//! Projection of harness events onto kernel agent chunks.
//!
//! The harness SPI emits rich, provider-neutral [`HarnessEvent`]s; the kernel consumes
//! [`AgentChunk`] frames with a `kind`, an `index`, and a JSON payload. [`Projector`] performs
//! that one-way mapping for the adapter's event pump.
//!
//! One event produces at most one chunk, and only these events produce chunks at all:
//!
//! - [`HarnessEventPayload::Output`] — `llm_output` (answer and system channels, and structured
//!   content) or `llm_reasoning` (reasoning channel);
//! - [`HarnessEventPayload::ItemLifecycle`] with `item_type == "tool_call"` — `tool_call` for
//!   `Started` and `tool_result` for every other kind;
//! - [`HarnessEventPayload::TurnLifecycle`] — `turn_lifecycle` when turn-lifecycle chunks are
//!   enabled, otherwise `task_failed` for a failed turn when turn-failure chunks are enabled.
//!
//! Usage, model-request, provider, diagnostic, state-change, hook and unknown events produce no
//! chunk: they are observations, and the observer sees them before projection.

use openjiuwen_agent::output::{AgentChunk, LLM_OUTPUT, LLM_REASONING};
use openjiuwen_harness_protocol::{
    HarnessEvent, HarnessEventPayload, ItemEventKind, ItemLifecycleEvent, JsonObject, JsonValue,
    OutputChannel, OutputEvent, OutputKind, TurnEventKind, TurnLifecycleEvent, TurnTerminationKind,
};
use serde_json::json;

use super::HarnessIOAdapterConfig;
use crate::input::{TASK_FAILED_KIND, TURN_LIFECYCLE_KIND};

/// Chunk kind of a projected tool call.
const TOOL_CALL_KIND: &str = "tool_call";
/// Chunk kind of a projected tool result.
const TOOL_RESULT_KIND: &str = "tool_result";
/// Item type the provider uses for a tool call.
const TOOL_CALL_ITEM_TYPE: &str = "tool_call";
/// Tool name reported when the provider data carries none.
const UNKNOWN_TOOL_NAME: &str = "unknown";

/// Projects harness events onto kernel agent chunks for one adapter configuration.
pub(super) struct Projector {
    /// Whether turn lifecycle events become chunks.
    emit_turn_lifecycle: bool,
    /// Whether a failed turn becomes a `task_failed` chunk.
    emit_turn_failure: bool,
    /// Provider card name, used in the fallback failure message.
    card_name: String,
}

impl Projector {
    /// Builds the projector for `config` and the provider named `card_name`.
    pub(super) fn new(config: &HarnessIOAdapterConfig, card_name: String) -> Self {
        Self {
            emit_turn_lifecycle: config.emit_turn_lifecycle,
            emit_turn_failure: config.emit_turn_failure,
            card_name,
        }
    }

    /// Projects one event to a chunk, or returns [`None`] when the event carries no chunk.
    ///
    /// `index` is called only when a chunk is emitted, so the adapter's chunk counter advances
    /// exactly once per emitted chunk and never for an event that projects to nothing.
    pub(super) fn project(
        &self,
        event: &HarnessEvent,
        index: impl FnOnce() -> i64,
    ) -> Option<AgentChunk> {
        match &event.event {
            HarnessEventPayload::Output(output) => Some(output_chunk(output, index())),
            HarnessEventPayload::ItemLifecycle(item) => {
                item_chunk(item, event.item_id.as_deref(), index)
            }
            HarnessEventPayload::TurnLifecycle(turn) => self.turn_chunk(event, turn, index),
            // Usage, model requests, provider extensions, diagnostics, state changes, hook
            // observations and unknown future events are observations only.
            _ => None,
        }
    }

    /// Projects a turn lifecycle event according to the configured switches.
    fn turn_chunk(
        &self,
        event: &HarnessEvent,
        turn: &TurnLifecycleEvent,
        index: impl FnOnce() -> i64,
    ) -> Option<AgentChunk> {
        if self.emit_turn_lifecycle {
            return Self::lifecycle_chunk(event, turn, index);
        }
        if self.emit_turn_failure && turn.kind == TurnEventKind::Failed {
            return Some(self.failed_chunk(turn, index));
        }
        None
    }

    /// Builds one `turn_lifecycle` chunk.
    ///
    /// `Paused` and `Resumed` are not projected: they do not change what a kernel consumer does
    /// with the turn.
    fn lifecycle_chunk(
        event: &HarnessEvent,
        turn: &TurnLifecycleEvent,
        index: impl FnOnce() -> i64,
    ) -> Option<AgentChunk> {
        let kind = match turn.kind {
            TurnEventKind::Started => "started",
            TurnEventKind::Finished => "finished",
            TurnEventKind::Failed => "failed",
            TurnEventKind::Aborted => "aborted",
            TurnEventKind::Paused | TurnEventKind::Resumed => return None,
        };
        let mut payload = json!({
            "turn_id": event.turn_id.clone().unwrap_or_default(),
            "kind": kind,
        });
        match turn.kind {
            TurnEventKind::Finished => {
                if let Some(result) = turn.result.as_ref() {
                    if let JsonValue::String(final_output) = &result.final_output {
                        payload["final_output"] = JsonValue::String(final_output.clone());
                    }
                }
            }
            TurnEventKind::Failed => {
                if let Some(error) = turn
                    .result
                    .as_ref()
                    .and_then(|result| result.error.as_ref())
                {
                    payload["error_message"] = JsonValue::String(error.message.clone());
                    payload["error_code"] = error
                        .code
                        .as_ref()
                        .map_or(JsonValue::Null, |code| JsonValue::String(code.clone()));
                }
            }
            TurnEventKind::Aborted => {
                if let Some(termination) = turn
                    .result
                    .as_ref()
                    .and_then(|result| result.termination.as_ref())
                {
                    payload["termination_kind"] =
                        JsonValue::String(termination_kind(termination.kind).to_owned());
                }
            }
            TurnEventKind::Started | TurnEventKind::Paused | TurnEventKind::Resumed => {}
        }
        Some(AgentChunk::new(TURN_LIFECYCLE_KIND, index(), payload))
    }

    /// Builds the `task_failed` chunk for a failed turn without a turn lifecycle.
    fn failed_chunk(&self, turn: &TurnLifecycleEvent, index: impl FnOnce() -> i64) -> AgentChunk {
        let error = turn
            .result
            .as_ref()
            .and_then(|result| result.error.as_ref());
        let message = error.map_or_else(
            || format!("{} turn failed", self.card_name),
            |error| error.message.clone(),
        );
        let code = error
            .and_then(|error| error.code.clone())
            .map_or(JsonValue::Null, JsonValue::String);
        AgentChunk::new(
            TASK_FAILED_KIND,
            index(),
            json!({"message": message, "code": code}),
        )
    }
}

/// Builds the chunk for one output event.
///
/// Structured content keeps its JSON shape; text content is carried as a string (a non-string
/// value falls back to its compact JSON text). The payload always carries `content`, `output_id`,
/// `operation` and `channel`; no delta/snapshot arithmetic is applied here.
fn output_chunk(output: &OutputEvent, index: i64) -> AgentChunk {
    let (kind, content) = match output.kind {
        OutputKind::Structured => (LLM_OUTPUT, output.content.clone()),
        OutputKind::Text => {
            let kind = if output.channel == OutputChannel::Reasoning {
                LLM_REASONING
            } else {
                LLM_OUTPUT
            };
            (kind, JsonValue::String(text_of(&output.content)))
        }
    };
    AgentChunk::new(
        kind,
        index,
        json!({
            "content": content,
            "output_id": output.output_id.clone(),
            "operation": output.operation.as_str(),
            "channel": output.channel.as_str(),
        }),
    )
}

/// Builds the chunk for one tool-call item lifecycle event, or [`None`] for other item types.
fn item_chunk(
    item: &ItemLifecycleEvent,
    envelope_item_id: Option<&str>,
    index: impl FnOnce() -> i64,
) -> Option<AgentChunk> {
    if item.item_type != TOOL_CALL_ITEM_TYPE {
        return None;
    }
    let tool_call_id = tool_call_id(&item.data, envelope_item_id);
    match item.kind {
        ItemEventKind::Started => {
            let arguments = match item.data.get("arguments") {
                Some(JsonValue::String(text)) => text.clone(),
                Some(other) => compact_json(other),
                None => "null".to_owned(),
            };
            Some(AgentChunk::new(
                TOOL_CALL_KIND,
                index(),
                json!({
                    "name": started_tool_name(&item.data),
                    "arguments": arguments,
                    "tool_call_id": tool_call_id,
                }),
            ))
        }
        ItemEventKind::Updated | ItemEventKind::Completed => {
            let mut payload = json!({
                "tool_call_id": tool_call_id,
                "tool_name": result_tool_name(&item.data),
                "result": item.data.get("result").cloned().unwrap_or(JsonValue::Null),
            });
            if let Some(rendered) = item.data.get("rendered_result").and_then(JsonValue::as_str) {
                payload["rendered_result"] = JsonValue::String(rendered.to_owned());
            }
            Some(AgentChunk::new(TOOL_RESULT_KIND, index(), payload))
        }
    }
}

/// Returns the tool-call id of an item: provider data first, then the envelope item id.
fn tool_call_id(data: &JsonObject, envelope_item_id: Option<&str>) -> String {
    data.get("tool_call_id")
        .and_then(JsonValue::as_str)
        .or(envelope_item_id)
        .unwrap_or_default()
        .to_owned()
}

/// Returns the tool name of a started call: `name` first, then `tool_name`.
fn started_tool_name(data: &JsonObject) -> String {
    data.get("name")
        .or_else(|| data.get("tool_name"))
        .and_then(JsonValue::as_str)
        .unwrap_or(UNKNOWN_TOOL_NAME)
        .to_owned()
}

/// Returns the tool name of a finished call: `tool_name` first, then `name`.
fn result_tool_name(data: &JsonObject) -> String {
    data.get("tool_name")
        .or_else(|| data.get("name"))
        .and_then(JsonValue::as_str)
        .unwrap_or(UNKNOWN_TOOL_NAME)
        .to_owned()
}

/// Returns the wire string of a termination kind.
///
/// The protocol type exposes no `as_str`, so this mirror keeps the payload on the wire values
/// (`user_abort`, `timeout`, `policy`, `provider`, `harness_stop`).
fn termination_kind(kind: TurnTerminationKind) -> &'static str {
    match kind {
        TurnTerminationKind::UserAbort => "user_abort",
        TurnTerminationKind::Timeout => "timeout",
        TurnTerminationKind::Policy => "policy",
        TurnTerminationKind::Provider => "provider",
        TurnTerminationKind::HarnessStop => "harness_stop",
    }
}

/// Returns the text of a content value, falling back to compact JSON.
fn text_of(value: &JsonValue) -> String {
    match value {
        JsonValue::String(text) => text.clone(),
        other => compact_json(other),
    }
}

/// Renders a JSON value as compact JSON, falling back to the empty string.
fn compact_json(value: &JsonValue) -> String {
    serde_json::to_string(value).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    // Tests assert by panicking on failure, which is the intent of a test.
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use openjiuwen_harness_protocol::{
        DiagnosticEvent, DiagnosticLevel, HarnessEvent, HarnessEventPayload, ItemEventKind,
        ItemLifecycleEvent, JsonObject, JsonValue, OutputChannel, OutputEvent, OutputKind,
        OutputOperation, TurnError, TurnEventKind, TurnLifecycleEvent, TurnResult, TurnStatus,
        TurnTermination, TurnTerminationKind,
    };
    use serde_json::json;

    use super::{
        Projector, TASK_FAILED_KIND, TOOL_CALL_KIND, TOOL_RESULT_KIND, TURN_LIFECYCLE_KIND,
    };
    use crate::io_adapter::HarnessIOAdapterConfig;

    fn projector() -> Projector {
        Projector::new(&HarnessIOAdapterConfig::default(), "native".to_owned())
    }

    fn lifecycle_projector() -> Projector {
        Projector::new(
            &HarnessIOAdapterConfig {
                emit_turn_lifecycle: true,
                ..HarnessIOAdapterConfig::default()
            },
            "native".to_owned(),
        )
    }

    fn failure_projector() -> Projector {
        Projector::new(
            &HarnessIOAdapterConfig {
                emit_turn_failure: true,
                ..HarnessIOAdapterConfig::default()
            },
            "native".to_owned(),
        )
    }

    fn event(payload: impl Into<HarnessEventPayload>) -> HarnessEvent {
        HarnessEvent::new(1, 0.0, payload, "host-session", "agent")
    }

    fn output_event(kind: OutputKind, content: JsonValue, channel: OutputChannel) -> HarnessEvent {
        event(OutputEvent {
            output_id: "out-1".to_owned(),
            kind,
            content,
            operation: OutputOperation::Delta,
            channel,
            content_index: 0,
            content_type: None,
            data: JsonObject::new(),
        })
    }

    fn item_event(kind: ItemEventKind, data: JsonObject, item_id: Option<&str>) -> HarnessEvent {
        let mut event = event(ItemLifecycleEvent {
            kind,
            item_type: "tool_call".to_owned(),
            data,
        });
        event.item_id = item_id.map(str::to_owned);
        event
    }

    fn turn_event(kind: TurnEventKind, result: Option<TurnResult>) -> HarnessEvent {
        let mut event = event(TurnLifecycleEvent { kind, result });
        event.turn_id = Some("turn-1".to_owned());
        event
    }

    fn turn_result(
        status: TurnStatus,
        final_output: JsonValue,
        error: Option<TurnError>,
        termination: Option<TurnTermination>,
    ) -> TurnResult {
        TurnResult {
            status,
            messages: Vec::new(),
            final_output,
            structured_output: JsonValue::Null,
            stop_reason: None,
            termination,
            error,
            usage: None,
            cost: None,
            started_at: None,
            completed_at: None,
            duration_ms: None,
            provider_data: JsonObject::new(),
        }
    }

    #[test]
    fn started_tool_call_falls_back_to_tool_name_and_envelope_id() {
        let mut data = JsonObject::new();
        data.insert("tool_name".to_owned(), json!("write_file"));
        data.insert("arguments".to_owned(), json!({"path": "a.txt"}));
        let chunk = projector()
            .project(
                &item_event(ItemEventKind::Started, data, Some("item-7")),
                || 3,
            )
            .unwrap();
        assert_eq!(chunk.kind, TOOL_CALL_KIND);
        assert_eq!(chunk.index, 3);
        assert_eq!(chunk.payload["name"], json!("write_file"));
        assert_eq!(chunk.payload["arguments"], json!(r#"{"path":"a.txt"}"#));
        assert_eq!(chunk.payload["tool_call_id"], json!("item-7"));
    }

    #[test]
    fn started_tool_call_without_data_uses_unknown_and_null_arguments() {
        let chunk = projector()
            .project(
                &item_event(ItemEventKind::Started, JsonObject::new(), None),
                || 0,
            )
            .unwrap();
        assert_eq!(chunk.payload["name"], json!("unknown"));
        assert_eq!(chunk.payload["arguments"], json!("null"));
        assert_eq!(chunk.payload["tool_call_id"], json!(""));
    }

    #[test]
    fn string_arguments_are_carried_verbatim() {
        let mut data = JsonObject::new();
        data.insert("name".to_owned(), json!("run"));
        data.insert("arguments".to_owned(), json!("{\"cmd\":\"ls\"}"));
        data.insert("tool_call_id".to_owned(), json!("call-1"));
        let chunk = projector()
            .project(&item_event(ItemEventKind::Started, data, None), || 1)
            .unwrap();
        assert_eq!(chunk.payload["name"], json!("run"));
        assert_eq!(chunk.payload["arguments"], json!("{\"cmd\":\"ls\"}"));
    }

    #[test]
    fn completed_tool_call_has_no_rendered_result_unless_the_provider_supplies_one() {
        let mut data = JsonObject::new();
        data.insert("name".to_owned(), json!("write_file"));
        data.insert("result".to_owned(), json!({"ok": true}));
        let chunk = projector()
            .project(
                &item_event(ItemEventKind::Completed, data, Some("item-1")),
                || 2,
            )
            .unwrap();
        assert_eq!(chunk.kind, TOOL_RESULT_KIND);
        assert_eq!(chunk.payload["tool_name"], json!("write_file"));
        assert_eq!(chunk.payload["result"], json!({"ok": true}));
        assert_eq!(chunk.payload["tool_call_id"], json!("item-1"));
        assert!(chunk.payload.get("rendered_result").is_none());
    }

    #[test]
    fn completed_tool_call_carries_a_rendered_result_and_a_null_result() {
        let mut data = JsonObject::new();
        data.insert("tool_name".to_owned(), json!("read_file"));
        data.insert("rendered_result".to_owned(), json!("hello"));
        let chunk = projector()
            .project(&item_event(ItemEventKind::Updated, data, None), || 0)
            .unwrap();
        assert_eq!(chunk.payload["rendered_result"], json!("hello"));
        assert_eq!(chunk.payload["result"], JsonValue::Null);
        assert_eq!(chunk.payload["tool_call_id"], json!(""));
    }

    #[test]
    fn non_tool_items_produce_no_chunk() {
        let event = event(ItemLifecycleEvent {
            kind: ItemEventKind::Started,
            item_type: "message".to_owned(),
            data: JsonObject::new(),
        });
        assert!(projector().project(&event, || 0).is_none());
    }

    #[test]
    fn structured_output_becomes_an_llm_output_chunk_with_raw_content() {
        let chunk = projector()
            .project(
                &output_event(
                    OutputKind::Structured,
                    json!({"answer": 42}),
                    OutputChannel::Reasoning,
                ),
                || 5,
            )
            .unwrap();
        assert_eq!(chunk.kind, "llm_output");
        assert_eq!(chunk.index, 5);
        assert_eq!(chunk.payload["content"], json!({"answer": 42}));
        assert_eq!(chunk.payload["output_id"], json!("out-1"));
        assert_eq!(chunk.payload["operation"], json!("delta"));
        assert_eq!(chunk.payload["channel"], json!("reasoning"));
    }

    #[test]
    fn text_output_uses_the_chunk_kind_of_its_channel() {
        let reasoning = projector()
            .project(
                &output_event(
                    OutputKind::Text,
                    json!("thinking"),
                    OutputChannel::Reasoning,
                ),
                || 0,
            )
            .unwrap();
        assert_eq!(reasoning.kind, "llm_reasoning");
        assert_eq!(reasoning.payload["content"], json!("thinking"));

        for channel in [OutputChannel::Answer, OutputChannel::System] {
            let chunk = projector()
                .project(
                    &output_event(OutputKind::Text, json!("hello"), channel),
                    || 1,
                )
                .unwrap();
            assert_eq!(chunk.kind, "llm_output");
            assert_eq!(chunk.payload["content"], json!("hello"));
            assert_eq!(chunk.payload["channel"], json!(channel.as_str()));
        }
    }

    #[test]
    fn non_string_text_content_falls_back_to_compact_json() {
        let chunk = projector()
            .project(
                &output_event(
                    OutputKind::Text,
                    json!({"text": "hi"}),
                    OutputChannel::Answer,
                ),
                || 0,
            )
            .unwrap();
        assert_eq!(chunk.payload["content"], json!(r#"{"text":"hi"}"#));
    }

    #[test]
    fn paused_and_resumed_turns_produce_no_chunk() {
        for kind in [TurnEventKind::Paused, TurnEventKind::Resumed] {
            assert!(lifecycle_projector()
                .project(&turn_event(kind, None), || 0)
                .is_none());
        }
    }

    #[test]
    fn turn_lifecycle_chunks_carry_the_terminal_details() {
        let started = lifecycle_projector()
            .project(&turn_event(TurnEventKind::Started, None), || 0)
            .unwrap();
        assert_eq!(started.kind, TURN_LIFECYCLE_KIND);
        assert_eq!(started.payload["turn_id"], json!("turn-1"));
        assert_eq!(started.payload["kind"], json!("started"));

        let finished = lifecycle_projector()
            .project(
                &turn_event(
                    TurnEventKind::Finished,
                    Some(turn_result(
                        TurnStatus::Completed,
                        json!("done"),
                        None,
                        None,
                    )),
                ),
                || 1,
            )
            .unwrap();
        assert_eq!(finished.payload["kind"], json!("finished"));
        assert_eq!(finished.payload["final_output"], json!("done"));

        let aborted = lifecycle_projector()
            .project(
                &turn_event(
                    TurnEventKind::Aborted,
                    Some(turn_result(
                        TurnStatus::Interrupted,
                        JsonValue::Null,
                        None,
                        Some(TurnTermination {
                            kind: TurnTerminationKind::UserAbort,
                            message: None,
                            code: None,
                            provider_data: JsonObject::new(),
                        }),
                    )),
                ),
                || 2,
            )
            .unwrap();
        assert_eq!(aborted.payload["termination_kind"], json!("user_abort"));
        assert!(aborted.payload.get("error_message").is_none());
    }

    #[test]
    fn failed_turn_lifecycle_carries_message_and_code() {
        let chunk = lifecycle_projector()
            .project(
                &turn_event(
                    TurnEventKind::Failed,
                    Some(turn_result(
                        TurnStatus::Failed,
                        JsonValue::Null,
                        Some(TurnError {
                            message: "boom".to_owned(),
                            code: None,
                            category: None,
                            retryable: None,
                            provider_data: JsonObject::new(),
                        }),
                        None,
                    )),
                ),
                || 0,
            )
            .unwrap();
        assert_eq!(chunk.payload["kind"], json!("failed"));
        assert_eq!(chunk.payload["error_message"], json!("boom"));
        assert_eq!(chunk.payload["error_code"], JsonValue::Null);
    }

    #[test]
    fn turn_lifecycle_is_silent_by_default() {
        assert!(projector()
            .project(&turn_event(TurnEventKind::Started, None), || 0)
            .is_none());
        assert!(projector()
            .project(
                &turn_event(
                    TurnEventKind::Failed,
                    Some(turn_result(
                        TurnStatus::Failed,
                        JsonValue::Null,
                        Some(TurnError {
                            message: "boom".to_owned(),
                            code: Some("E1".to_owned()),
                            category: None,
                            retryable: None,
                            provider_data: JsonObject::new(),
                        }),
                        None,
                    )),
                ),
                || 0,
            )
            .is_none());
    }

    #[test]
    fn failed_turn_becomes_a_task_failed_chunk_with_the_provider_message() {
        let chunk = failure_projector()
            .project(
                &turn_event(
                    TurnEventKind::Failed,
                    Some(turn_result(
                        TurnStatus::Failed,
                        JsonValue::Null,
                        Some(TurnError {
                            message: "rate limited".to_owned(),
                            code: Some("429".to_owned()),
                            category: None,
                            retryable: None,
                            provider_data: JsonObject::new(),
                        }),
                        None,
                    )),
                ),
                || 4,
            )
            .unwrap();
        assert_eq!(chunk.kind, TASK_FAILED_KIND);
        assert_eq!(chunk.index, 4);
        assert_eq!(chunk.payload["message"], json!("rate limited"));
        assert_eq!(chunk.payload["code"], json!("429"));
    }

    #[test]
    fn failed_turn_without_an_error_uses_the_card_name_fallback() {
        let chunk = failure_projector()
            .project(
                &turn_event(
                    TurnEventKind::Failed,
                    Some(turn_result(TurnStatus::Failed, JsonValue::Null, None, None)),
                ),
                || 0,
            )
            .unwrap();
        assert_eq!(chunk.kind, TASK_FAILED_KIND);
        assert_eq!(chunk.payload["message"], json!("native turn failed"));
        assert_eq!(chunk.payload["code"], JsonValue::Null);
    }

    #[test]
    fn turn_failure_is_silent_when_only_lifecycle_chunks_are_enabled() {
        let projector = Projector::new(
            &HarnessIOAdapterConfig {
                emit_turn_lifecycle: true,
                emit_turn_failure: true,
                ..HarnessIOAdapterConfig::default()
            },
            "native".to_owned(),
        );
        let chunk = projector
            .project(
                &turn_event(
                    TurnEventKind::Failed,
                    Some(turn_result(
                        TurnStatus::Failed,
                        JsonValue::Null,
                        Some(TurnError {
                            message: "boom".to_owned(),
                            code: None,
                            category: None,
                            retryable: None,
                            provider_data: JsonObject::new(),
                        }),
                        None,
                    )),
                ),
                || 0,
            )
            .unwrap();
        assert_eq!(chunk.kind, TURN_LIFECYCLE_KIND);
    }

    #[test]
    fn silent_events_do_not_consume_an_index() {
        let calls = std::cell::Cell::new(0);
        let next = || {
            calls.set(calls.get() + 1);
            0
        };
        let observation = event(DiagnosticEvent {
            level: DiagnosticLevel::Info,
            message: "noted".to_owned(),
            data: JsonObject::new(),
        });
        assert!(projector().project(&observation, next).is_none());
        assert!(lifecycle_projector()
            .project(&turn_event(TurnEventKind::Paused, None), next)
            .is_none());
        assert_eq!(
            calls.get(),
            0,
            "a silent event must not claim a chunk index"
        );
    }

    #[test]
    fn observation_only_events_produce_no_chunk() {
        let event = event(DiagnosticEvent {
            level: DiagnosticLevel::Info,
            message: "noted".to_owned(),
            data: JsonObject::new(),
        });
        assert!(projector().project(&event, || 0).is_none());
    }
}
