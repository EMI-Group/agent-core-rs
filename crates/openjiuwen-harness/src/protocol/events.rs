//! Mapping from kernel agent frames to provider-neutral harness events.
//!
//! The native harness runs the kernel `DeepAgent`, whose stream is a sequence
//! of [`AgentChunk`] frames and whose terminal value is an [`AgentOutput`].
//! This module translates those kernel types into the payloads of
//! `openjiuwen-harness-protocol`:
//!
//! - [`chunk_payload`] maps one streaming frame (`llm_output`,
//!   `llm_reasoning`, `llm_usage`, or an unknown kind) to an output, usage, or
//!   provider payload;
//! - [`usage_from_metadata`] normalizes the kernel's `usage_metadata` object;
//! - [`item_events`] and [`failure_diagnostic`] turn the terminal output's tool
//!   observations and error into item and diagnostic payloads;
//! - [`messages_from_output`] and [`turn_result`] build the terminal
//!   [`TurnResult`] (messages, usage, timings, and status/Rules).
//!
//! The functions are pure: no clock, no I/O, and no validation. The caller
//! supplies timestamps ([`TurnTiming`]), the turn id, and validates the built
//! result once with [`Validate::validate`](openjiuwen_harness_protocol::Validate).
//!
//! The kernel stream carries no live tool frames, so [`item_events`] reports
//! each recorded observation as a coarse `Started` + `Completed` pair at turn
//! end (documented in the node `CONTEXT.md`).

use openjiuwen_agent::output::{
    AgentChunk, AgentOutput, ANSWER, CONTEXT_USAGE, INTERACTION, LLM_OUTPUT, LLM_REASONING,
    LLM_USAGE,
};
use openjiuwen_agent::ResultType;
use openjiuwen_foundation::prompt::Usage;
use openjiuwen_harness_protocol::{
    ContentBlock, DiagnosticEvent, DiagnosticLevel, HarnessEventPayload, ItemEventKind,
    ItemLifecycleEvent, JsonObject, JsonValue, MessageRole, OutputChannel, OutputEvent, OutputKind,
    OutputOperation, ProviderEvent, TurnError, TurnMessage, TurnResult, TurnStatus,
    TurnTermination, TurnUsage, UsageUpdateMode, UsageUpdatedEvent, EVENT_WIRE_SCHEMA_VERSION,
};

/// Wall-clock timing of one turn, supplied by the caller.
///
/// `started_at`/`completed_at` are Unix epoch seconds and `duration_ms` is the
/// monotonic elapsed time in milliseconds; the values are copied into the
/// terminal [`TurnResult`] verbatim.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct TurnTiming {
    /// Unix epoch seconds when the turn started.
    pub started_at: f64,
    /// Unix epoch seconds when the turn completed.
    pub completed_at: f64,
    /// Monotonic elapsed time in milliseconds.
    pub duration_ms: u64,
}

/// Maps one kernel streaming frame to a harness payload.
///
/// - [`LLM_OUTPUT`] becomes an answer [`OutputEvent`] delta;
/// - [`LLM_REASONING`] becomes a reasoning [`OutputEvent`] delta;
/// - [`LLM_USAGE`] becomes a [`UsageUpdatedEvent`] delta, or `None` when the
///   frame carries no usable `usage_metadata`;
/// - [`ANSWER`], [`INTERACTION`], and [`CONTEXT_USAGE`] produce no event;
/// - every other kind becomes a [`ProviderEvent`] preserving the frame kind and
///   payload.
///
/// `content_index` counts output events across both channels: each emitted
/// output event uses the current value and then increments it. A missing or
/// non-string `content` field is treated as an empty delta.
pub(crate) fn chunk_payload(
    chunk: &AgentChunk,
    turn_id: &str,
    content_index: &mut u64,
) -> Option<HarnessEventPayload> {
    match chunk.kind.as_str() {
        LLM_OUTPUT => Some(output_payload(
            chunk,
            turn_id,
            "answer",
            OutputChannel::Answer,
            content_index,
        )),
        LLM_REASONING => Some(output_payload(
            chunk,
            turn_id,
            "reasoning",
            OutputChannel::Reasoning,
            content_index,
        )),
        LLM_USAGE => {
            let usage = usage_from_metadata(chunk.payload.get("usage_metadata")?)?;
            Some(
                UsageUpdatedEvent {
                    usage,
                    mode: UsageUpdateMode::Delta,
                }
                .into(),
            )
        }
        ANSWER | INTERACTION | CONTEXT_USAGE => None,
        _ => Some(
            ProviderEvent {
                provider: "native".to_owned(),
                event_type: chunk.kind.clone(),
                schema_version: EVENT_WIRE_SCHEMA_VERSION.to_owned(),
                payload: chunk
                    .payload
                    .as_object()
                    .map(|object| {
                        object
                            .iter()
                            .map(|(key, value)| (key.clone(), value.clone()))
                            .collect()
                    })
                    .unwrap_or_default(),
            }
            .into(),
        ),
    }
}

/// Builds one delta [`OutputEvent`] for `chunk`, consuming a content index.
fn output_payload(
    chunk: &AgentChunk,
    turn_id: &str,
    suffix: &str,
    channel: OutputChannel,
    content_index: &mut u64,
) -> HarnessEventPayload {
    let delta = chunk
        .payload
        .get("content")
        .and_then(JsonValue::as_str)
        .unwrap_or_default()
        .to_owned();
    let index = *content_index;
    *content_index = content_index.saturating_add(1);
    OutputEvent {
        output_id: format!("native-output:{turn_id}:{suffix}"),
        kind: OutputKind::Text,
        content: JsonValue::String(delta),
        operation: OutputOperation::Delta,
        channel,
        content_index: index,
        content_type: Some("text/plain".to_owned()),
        data: JsonObject::new(),
    }
    .into()
}

/// Normalizes a kernel `usage_metadata` object into [`TurnUsage`].
///
/// Returns `None` when `value` is not a JSON object or is empty. Known counters
/// are copied into their normalized fields; provider-specific counters land in
/// [`TurnUsage::provider_data`] so they survive the translation.
pub(crate) fn usage_from_metadata(value: &JsonValue) -> Option<TurnUsage> {
    let object = value.as_object()?;
    if object.is_empty() {
        return None;
    }
    let usage: Usage = serde_json::from_value(value.clone()).ok()?;
    Some(TurnUsage {
        input_tokens: Some(usage.input_tokens),
        output_tokens: Some(usage.output_tokens),
        cached_input_tokens: Some(usage.cache_tokens),
        reasoning_output_tokens: Some(usage.reasoning_tokens),
        total_tokens: Some(usage.total_tokens),
        provider_data: usage.extra,
    })
}

/// Builds the `Started` + `Completed` item events for every tool observation.
///
/// Observations carry no error flag, so the completed payload is identical to
/// the started one: `{"tool_name", "content"}`. The caller attaches
/// `item_id` from the observation's `tool_call_id`.
pub(crate) fn item_events(output: &AgentOutput) -> Vec<HarnessEventPayload> {
    let mut events = Vec::with_capacity(output.observations.len().saturating_mul(2));
    for observation in &output.observations {
        let mut data = JsonObject::new();
        data.insert(
            "tool_name".to_owned(),
            JsonValue::String(observation.name.clone().unwrap_or_default()),
        );
        data.insert(
            "content".to_owned(),
            JsonValue::String(observation.content.clone()),
        );
        for kind in [ItemEventKind::Started, ItemEventKind::Completed] {
            events.push(
                ItemLifecycleEvent {
                    kind,
                    item_type: "tool".to_owned(),
                    data: data.clone(),
                }
                .into(),
            );
        }
    }
    events
}

/// Builds the error diagnostic for a failed terminal output.
///
/// Returns `Some` only when `output` failed ([`ResultType::Error`]) and carries
/// non-empty message text.
pub(crate) fn failure_diagnostic(output: &AgentOutput) -> Option<HarnessEventPayload> {
    if output.result_type != ResultType::Error || output.output.is_empty() {
        return None;
    }
    let mut data = JsonObject::new();
    data.insert(
        "result_type".to_owned(),
        JsonValue::String("error".to_owned()),
    );
    Some(
        DiagnosticEvent {
            level: DiagnosticLevel::Error,
            message: output.output.clone(),
            data,
        }
        .into(),
    )
}

/// Builds the normalized messages retained by a turn result.
///
/// The assistant message (`"<turn_id>:assistant"`) carries a reasoning block
/// (`"<turn_id>:reasoning"`, with `null` content because [`AgentOutput`] keeps
/// no separate reasoning text) when the run produced reasoning, plus a text
/// block (`"<turn_id>:text"`) holding the answer text. Each tool observation
/// becomes a [`MessageRole::Tool`] message (`"<turn_id>:tool:<call_id>"`) with
/// one text block whose id is the tool call id.
pub(crate) fn messages_from_output(output: &AgentOutput, turn_id: &str) -> Vec<TurnMessage> {
    let mut blocks = Vec::new();
    if output.reasoning_present {
        blocks.push(ContentBlock {
            block_id: format!("{turn_id}:reasoning"),
            kind: "reasoning".to_owned(),
            content: JsonValue::Null,
            data: JsonObject::new(),
        });
    }
    blocks.push(ContentBlock {
        block_id: format!("{turn_id}:text"),
        kind: "text".to_owned(),
        content: JsonValue::String(output.output.clone()),
        data: JsonObject::new(),
    });
    let mut messages = Vec::with_capacity(output.observations.len().saturating_add(1));
    messages.push(TurnMessage {
        message_id: format!("{turn_id}:assistant"),
        role: MessageRole::Assistant,
        content: blocks,
        data: JsonObject::new(),
    });
    for observation in &output.observations {
        let call_id = observation.tool_call_id.clone().unwrap_or_default();
        let mut data = JsonObject::new();
        data.insert(
            "tool_name".to_owned(),
            JsonValue::String(observation.name.clone().unwrap_or_default()),
        );
        messages.push(TurnMessage {
            message_id: format!("{turn_id}:tool:{call_id}"),
            role: MessageRole::Tool,
            content: vec![ContentBlock {
                block_id: call_id,
                kind: "text".to_owned(),
                content: JsonValue::String(observation.content.clone()),
                data,
            }],
            data: JsonObject::new(),
        });
    }
    messages
}

/// Builds the terminal [`TurnResult`] for one turn.
///
/// `termination` is `Some` for a turn aborted or stopped by the harness, which
/// always wins over the kernel output. Otherwise the output decides:
/// [`ResultType::Error`] → `Failed` with a `sdk_error` [`TurnError`] and the
/// provider finish reason, [`ResultType::Interrupt`] → `Completed` with
/// `stop_reason: "interrupt"` and the pending interrupt ids, [`ResultType::Answer`]
/// → `Completed` with the finish reason or `"answer"`. A missing output →
/// `Failed` with `"native turn ended without a result"`.
///
/// `usage`, the timing fields, and `provider_data` (`iterations_used`,
/// `result_type`) are always populated. `messages` starts empty: the caller
/// fills it with [`messages_from_output`] because this function does not
/// receive the turn id, and validates the result once with
/// [`Validate::validate`](openjiuwen_harness_protocol::Validate).
pub(crate) fn turn_result(
    output: Option<&AgentOutput>,
    usage: Option<TurnUsage>,
    timing: TurnTiming,
    termination: Option<TurnTermination>,
) -> TurnResult {
    let mut result = TurnResult {
        status: TurnStatus::Completed,
        messages: Vec::new(),
        final_output: JsonValue::Null,
        structured_output: JsonValue::Null,
        stop_reason: None,
        termination: None,
        error: None,
        usage,
        cost: None,
        started_at: Some(timing.started_at),
        completed_at: Some(timing.completed_at),
        duration_ms: Some(timing.duration_ms),
        provider_data: JsonObject::new(),
    };
    result.provider_data.insert(
        "iterations_used".to_owned(),
        output
            .and_then(|output| output.iterations_used)
            .map_or(JsonValue::Null, JsonValue::from),
    );
    result.provider_data.insert(
        "result_type".to_owned(),
        output.map_or(JsonValue::Null, |output| {
            JsonValue::String(result_type_str(output.result_type).to_owned())
        }),
    );

    if let Some(termination) = termination {
        result.status = TurnStatus::Interrupted;
        result.termination = Some(termination);
        return result;
    }

    let Some(output) = output else {
        result.status = TurnStatus::Failed;
        result.error = Some(turn_error("native turn ended without a result"));
        return result;
    };

    match output.result_type {
        ResultType::Answer => {
            result.stop_reason = Some(
                output
                    .finish_reason
                    .clone()
                    .unwrap_or_else(|| "answer".to_owned()),
            );
        }
        ResultType::Interrupt => {
            result.stop_reason = Some("interrupt".to_owned());
            result.final_output = JsonValue::String(output.output.clone());
            result.provider_data.insert(
                "pending_interrupt_ids".to_owned(),
                JsonValue::Array(
                    output
                        .interrupt_ids
                        .iter()
                        .cloned()
                        .map(JsonValue::String)
                        .collect(),
                ),
            );
        }
        ResultType::Error => {
            result.status = TurnStatus::Failed;
            result.stop_reason.clone_from(&output.finish_reason);
            result.error = Some(turn_error(output.output.clone()));
        }
    }
    result
}

/// Returns the normalized SDK failure with `code`/`category` `sdk_error`.
fn turn_error(message: impl Into<String>) -> TurnError {
    TurnError {
        message: message.into(),
        code: Some("sdk_error".to_owned()),
        category: Some("sdk_error".to_owned()),
        retryable: None,
        provider_data: JsonObject::new(),
    }
}

/// Returns the kernel result type as its wire string.
fn result_type_str(result_type: ResultType) -> &'static str {
    match result_type {
        ResultType::Answer => "answer",
        ResultType::Error => "error",
        ResultType::Interrupt => "interrupt",
    }
}

#[cfg(test)]
mod tests {
    // Tests assert by panicking on failure, which is the intent of a test.
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::{
        chunk_payload, failure_diagnostic, item_events, messages_from_output, turn_result,
        usage_from_metadata, TurnTiming,
    };
    use openjiuwen_agent::output::{AgentChunk, AgentOutput, Observation, LLM_OUTPUT, LLM_USAGE};
    use openjiuwen_agent::ResultType;
    use openjiuwen_harness_protocol::{
        DiagnosticLevel, HarnessEventPayload, ItemEventKind, JsonObject, MessageRole,
        OutputChannel, OutputKind, OutputOperation, TurnStatus, TurnTermination,
        TurnTerminationKind, TurnUsage,
    };
    use serde_json::{json, Value};

    /// Builds a streaming chunk with `kind` and `payload`.
    fn chunk(kind: &str, payload: Value) -> AgentChunk {
        AgentChunk {
            kind: kind.to_owned(),
            index: 0,
            payload,
        }
    }

    /// Builds a timing value.
    fn timing() -> TurnTiming {
        TurnTiming {
            started_at: 10.0,
            completed_at: 11.5,
            duration_ms: 1500,
        }
    }

    /// Extracts the answer output payload.
    fn answer_output(payload: &HarnessEventPayload) -> &openjiuwen_harness_protocol::OutputEvent {
        match payload {
            HarnessEventPayload::Output(output) => output,
            other => panic!("expected an output payload, got {other:?}"),
        }
    }

    #[test]
    fn llm_output_maps_to_an_answer_delta() {
        let mut index = 0;
        let payload = chunk_payload(
            &chunk(
                LLM_OUTPUT,
                json!({"content": "he", "result_type": "answer"}),
            ),
            "turn-2",
            &mut index,
        )
        .expect("output payload");
        let output = answer_output(&payload);
        assert_eq!(output.output_id, "native-output:turn-2:answer");
        assert_eq!(output.kind, OutputKind::Text);
        assert_eq!(output.content, Value::String("he".to_owned()));
        assert_eq!(output.operation, OutputOperation::Delta);
        assert_eq!(output.channel, OutputChannel::Answer);
        assert_eq!(output.content_index, 0);
        assert_eq!(output.content_type.as_deref(), Some("text/plain"));
        assert!(output.data.is_empty());
        assert_eq!(index, 1);
    }

    #[test]
    fn reasoning_output_has_its_own_id_and_channel() {
        let mut index = 4;
        let payload = chunk_payload(
            &chunk(
                openjiuwen_agent::output::LLM_REASONING,
                json!({"content": "hm"}),
            ),
            "turn-1",
            &mut index,
        )
        .expect("reasoning payload");
        let output = answer_output(&payload);
        assert_eq!(output.output_id, "native-output:turn-1:reasoning");
        assert_eq!(output.channel, OutputChannel::Reasoning);
        assert_eq!(output.content_index, 4);
        assert_eq!(index, 5);
    }

    #[test]
    fn missing_content_is_an_empty_delta() {
        let mut index = 0;
        let payload = chunk_payload(&chunk(LLM_OUTPUT, json!({})), "turn-1", &mut index)
            .expect("output payload");
        assert_eq!(
            answer_output(&payload).content,
            Value::String(String::new())
        );
    }

    #[test]
    fn usage_frames_map_to_delta_usage_updates() {
        let mut index = 0;
        let payload = chunk_payload(
            &chunk(
                LLM_USAGE,
                json!({
                    "usage_metadata": {
                        "input_tokens": 10,
                        "output_tokens": 4,
                        "total_tokens": 14,
                        "cache_tokens": 2,
                        "reasoning_tokens": 1,
                        "vendor_counter": 7
                    },
                    "result_type": "answer"
                }),
            ),
            "turn-1",
            &mut index,
        )
        .expect("usage payload");
        let usage = match payload {
            HarnessEventPayload::UsageUpdated(update) => update,
            other => panic!("expected a usage payload, got {other:?}"),
        };
        assert_eq!(
            usage.mode,
            openjiuwen_harness_protocol::UsageUpdateMode::Delta
        );
        assert_eq!(usage.usage.input_tokens, Some(10));
        assert_eq!(usage.usage.output_tokens, Some(4));
        assert_eq!(usage.usage.total_tokens, Some(14));
        assert_eq!(usage.usage.cached_input_tokens, Some(2));
        assert_eq!(usage.usage.reasoning_output_tokens, Some(1));
        assert_eq!(
            usage.usage.provider_data.get("vendor_counter"),
            Some(&json!(7))
        );
        assert_eq!(index, 0, "usage frames do not consume content indexes");
    }

    #[test]
    fn usage_metadata_must_be_a_non_empty_object() {
        assert!(usage_from_metadata(&json!(null)).is_none());
        assert!(usage_from_metadata(&json!([])).is_none());
        assert!(usage_from_metadata(&json!({})).is_none());
        assert!(usage_from_metadata(&json!(7)).is_none());
        let usage = usage_from_metadata(&json!({"input_tokens": 3})).expect("usage");
        assert_eq!(usage.input_tokens, Some(3));
        assert_eq!(usage.output_tokens, Some(0));
    }

    #[test]
    fn usage_frame_without_metadata_produces_no_event() {
        let mut index = 0;
        assert!(chunk_payload(&chunk(LLM_USAGE, json!({})), "turn-1", &mut index).is_none());
    }

    #[test]
    fn answer_and_interaction_frames_produce_no_event() {
        let mut index = 0;
        for kind in [
            openjiuwen_agent::output::ANSWER,
            openjiuwen_agent::output::INTERACTION,
            openjiuwen_agent::output::CONTEXT_USAGE,
        ] {
            assert!(chunk_payload(&chunk(kind, json!({})), "turn-1", &mut index).is_none());
        }
    }

    #[test]
    fn unknown_frames_become_provider_events() {
        let mut index = 0;
        let payload = chunk_payload(
            &chunk("custom.frame", json!({"alpha": 1, "beta": "two"})),
            "turn-1",
            &mut index,
        )
        .expect("provider payload");
        match payload {
            HarnessEventPayload::Provider(event) => {
                assert_eq!(event.provider, "native");
                assert_eq!(event.event_type, "custom.frame");
                assert_eq!(
                    event.schema_version,
                    openjiuwen_harness_protocol::EVENT_WIRE_SCHEMA_VERSION
                );
                assert_eq!(event.payload.get("alpha"), Some(&json!(1)));
                assert_eq!(event.payload.get("beta"), Some(&json!("two")));
            }
            other => panic!("expected a provider payload, got {other:?}"),
        }

        let non_object = chunk_payload(&chunk("custom.frame", json!("raw")), "turn-1", &mut index)
            .expect("provider payload");
        match non_object {
            HarnessEventPayload::Provider(event) => assert!(event.payload.is_empty()),
            other => panic!("expected a provider payload, got {other:?}"),
        }
    }

    #[test]
    fn item_events_pair_started_and_completed_per_observation() {
        let output = AgentOutput::answer("done").with_observations(vec![Observation {
            content: "42".to_owned(),
            name: Some("bash".to_owned()),
            tool_call_id: Some("call-1".to_owned()),
            ..Observation::default()
        }]);
        let events = item_events(&output);
        assert_eq!(events.len(), 2);
        for (event, expected_kind) in events
            .iter()
            .zip([ItemEventKind::Started, ItemEventKind::Completed])
        {
            match event {
                HarnessEventPayload::ItemLifecycle(item) => {
                    assert_eq!(item.kind, expected_kind);
                    assert_eq!(item.item_type, "tool");
                    assert_eq!(item.data.get("tool_name"), Some(&json!("bash")));
                    assert_eq!(item.data.get("content"), Some(&json!("42")));
                    assert_eq!(item.data.get("error"), None);
                }
                other => panic!("expected an item payload, got {other:?}"),
            }
        }
    }

    #[test]
    fn failure_diagnostic_only_for_non_empty_errors() {
        assert!(failure_diagnostic(&AgentOutput::answer("ok")).is_none());
        assert!(failure_diagnostic(&AgentOutput::error("")).is_none());
        let payload = failure_diagnostic(&AgentOutput::error("boom")).expect("diagnostic");
        match payload {
            HarnessEventPayload::Diagnostic(diagnostic) => {
                assert_eq!(diagnostic.level, DiagnosticLevel::Error);
                assert_eq!(diagnostic.message, "boom");
                assert_eq!(diagnostic.data.get("result_type"), Some(&json!("error")));
            }
            other => panic!("expected a diagnostic payload, got {other:?}"),
        }
    }

    #[test]
    fn messages_carry_assistant_and_tool_ids() {
        let mut output = AgentOutput::answer("hello").with_reasoning_present(true);
        output.observations = vec![Observation {
            content: "tool said hi".to_owned(),
            name: Some("bash".to_owned()),
            tool_call_id: Some("call-9".to_owned()),
            ..Observation::default()
        }];
        let messages = messages_from_output(&output, "turn-3");
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].message_id, "turn-3:assistant");
        assert_eq!(messages[0].role, MessageRole::Assistant);
        assert_eq!(messages[0].content.len(), 2);
        assert_eq!(messages[0].content[0].block_id, "turn-3:reasoning");
        assert_eq!(messages[0].content[1].block_id, "turn-3:text");
        assert_eq!(
            messages[0].content[1].content,
            Value::String("hello".to_owned())
        );
        assert_eq!(messages[1].message_id, "turn-3:tool:call-9");
        assert_eq!(messages[1].role, MessageRole::Tool);
        assert_eq!(messages[1].content[0].block_id, "call-9");
        assert_eq!(messages[1].content[0].kind, "text");
        assert_eq!(
            messages[1].content[0].content,
            Value::String("tool said hi".to_owned())
        );
        assert_eq!(
            messages[1].content[0].data.get("tool_name"),
            Some(&json!("bash"))
        );
    }

    #[test]
    fn answer_turn_completes_with_finish_reason_or_answer() {
        let output = AgentOutput::answer("all done");
        let result = turn_result(Some(&output), None, timing(), None);
        assert_eq!(result.status, TurnStatus::Completed);
        assert_eq!(result.stop_reason.as_deref(), Some("answer"));
        assert_eq!(result.started_at, Some(10.0));
        assert_eq!(result.completed_at, Some(11.5));
        assert_eq!(result.duration_ms, Some(1500));
        assert!(result.error.is_none());
        assert!(result.termination.is_none());
        assert!(result.messages.is_empty(), "the caller fills the messages");
        assert_eq!(
            result.provider_data.get("result_type"),
            Some(&json!("answer"))
        );
        assert_eq!(
            result.provider_data.get("iterations_used"),
            Some(&Value::Null)
        );

        let output = AgentOutput::answer("cut")
            .with_finish_reason("max_tokens")
            .with_iterations_used(3);
        let result = turn_result(Some(&output), None, timing(), None);
        assert_eq!(result.stop_reason.as_deref(), Some("max_tokens"));
        assert_eq!(result.provider_data.get("iterations_used"), Some(&json!(3)));
    }

    #[test]
    fn error_turn_fails_with_an_sdk_error() {
        let output = AgentOutput::error("model exploded").with_finish_reason("stop");
        let usage = TurnUsage {
            input_tokens: Some(3),
            ..TurnUsage::default()
        };
        let result = turn_result(Some(&output), Some(usage.clone()), timing(), None);
        assert_eq!(result.status, TurnStatus::Failed);
        assert_eq!(result.usage, Some(usage));
        assert_eq!(result.stop_reason.as_deref(), Some("stop"));
        assert_eq!(
            result.provider_data.get("result_type"),
            Some(&json!("error"))
        );
        let error = result.error.expect("error");
        assert_eq!(error.message, "model exploded");
        assert_eq!(error.code.as_deref(), Some("sdk_error"));
        assert_eq!(error.category.as_deref(), Some("sdk_error"));
        assert!(error.retryable.is_none());
        assert!(error.provider_data.is_empty());
    }

    #[test]
    fn interrupt_turn_completes_with_pending_ids() {
        let output = AgentOutput::interrupt(Vec::new(), vec!["interrupt-1".to_owned()]);
        let result = turn_result(Some(&output), None, timing(), None);
        assert_eq!(result.status, TurnStatus::Completed);
        assert_eq!(result.stop_reason.as_deref(), Some("interrupt"));
        assert_eq!(result.final_output, Value::String(String::new()));
        assert_eq!(
            result.provider_data.get("pending_interrupt_ids"),
            Some(&json!(["interrupt-1"]))
        );
        assert_eq!(
            result.provider_data.get("result_type"),
            Some(&json!("interrupt"))
        );
        assert!(result.error.is_none());
    }

    #[test]
    fn termination_wins_over_the_kernel_output() {
        let output = AgentOutput::answer("partial");
        let termination = TurnTermination {
            kind: TurnTerminationKind::UserAbort,
            message: Some("stopped".to_owned()),
            code: None,
            provider_data: JsonObject::new(),
        };
        let result = turn_result(Some(&output), None, timing(), Some(termination.clone()));
        assert_eq!(result.status, TurnStatus::Interrupted);
        assert_eq!(result.termination, Some(termination));
        assert!(result.error.is_none());
        assert_eq!(
            result.provider_data.get("result_type"),
            Some(&json!("answer"))
        );
    }

    #[test]
    fn missing_output_fails_the_turn() {
        let result = turn_result(None, None, timing(), None);
        assert_eq!(result.status, TurnStatus::Failed);
        let error = result.error.expect("error");
        assert_eq!(error.message, "native turn ended without a result");
        assert_eq!(error.code.as_deref(), Some("sdk_error"));
        assert_eq!(result.provider_data.get("result_type"), Some(&Value::Null));
    }

    #[test]
    fn result_type_strings_match_the_kernel_wire_names() {
        for (result_type, expected) in [
            (ResultType::Answer, "answer"),
            (ResultType::Error, "error"),
            (ResultType::Interrupt, "interrupt"),
        ] {
            let output = AgentOutput {
                result_type,
                ..AgentOutput::default()
            };
            let result = turn_result(Some(&output), None, timing(), None);
            assert_eq!(
                result.provider_data.get("result_type"),
                Some(&json!(expected))
            );
        }
    }
}
