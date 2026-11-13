//! Session-event projection of the `DeepSeek` Harness transport.
//!
//! Every method here turns one decoded JSON-RPC notification into harness observations (outputs,
//! tool calls, tool results, usage, step/subagent items, provider events) or into the turn's
//! terminal result. The turn loop, process ownership and lifecycle live in `harness`; the argv and
//! environment live in `wire`.
//!
//! # Wire contract
//!
//! The runtime publishes one notification per line as `{"method": …, "params": …}`:
//!
//! - `session.event` with `params.{sessionId, event}` carries the session's own events. Only the
//!   root session drives the turn; an event of a child (subagent) session is forwarded as a
//!   `child.session.event` provider event and changes nothing else.
//! - `subagent.started` / `subagent.finished` mark one child session's lifetime.
//! - any other notification is forwarded verbatim as a provider event, so nothing is dropped.
//!
//! Root events are mapped as follows: `assistant/chunk` projects text/reasoning deltas, final
//! blocks and usage; `assistant/message` projects final block outputs plus one protocol message;
//! `tool/call` and `tool/result` publish the `tool_call` item lifecycle the IO adapter projects
//! into tool chunks; `step/start` and `step/end` publish `step` items; `turn/end` records the
//! native ending, publishes it redacted and ends the turn.
//!
//! # Privacy
//!
//! A `turn/end` carries provider diagnostics. Before it is surfaced, the reason keeps `kind`, its
//! cause minus `message`, and an error limited to `{code, status, requestId, retryable}` — the
//! Python `_redact_turn_end_event` / `_safe_reason_data` contract.

use std::collections::{BTreeMap, BTreeSet};

use openjiuwen_harness_protocol::{
    json_object_to_value, ContentBlock, DiagnosticLevel, ItemEventKind, ItemLifecycleEvent,
    JsonObject, JsonValue, MessageRole, OutputChannel, OutputEvent, OutputKind, OutputOperation,
    TurnError, TurnMessage, TurnResult, TurnTerminationKind, TurnUsage,
};
use serde_json::json;

use crate::vendor_cli::{completed_turn, failed_turn, interrupted_turn, turn_error, TurnContext};

use super::harness::{
    BLOCKED_MESSAGE, FAILED_TURN_MESSAGE, INTERRUPTED_MESSAGE, MISSING_TURN_END_CODE,
    MISSING_TURN_END_MESSAGE, PROVIDER_EVENT_NAME, STOPPED_MESSAGE, TRANSPORT_CODE,
    UNKNOWN_FINISH_CODE, UNKNOWN_FINISH_MESSAGE,
};

/// Item type of the step lifecycle events.
const STEP_ITEM_TYPE: &str = "step";

/// Item type of the subagent lifecycle events.
const SUBAGENT_ITEM_TYPE: &str = "subagent";

/// Provider event type of a child session's event.
const CHILD_EVENT_TYPE: &str = "child.session.event";

/// Tool name reported when the runtime states none.
const UNKNOWN_TOOL_NAME: &str = "unknown";

/// Tool call id reported when the runtime states none.
const UNKNOWN_CALL_ID: &str = "unknown-tool-call";

/// Reason keys kept when a `turn/end` error object is redacted.
const SAFE_ERROR_KEYS: [&str; 4] = ["code", "status", "requestId", "retryable"];

/// Mutable state of one externally driven DSH turn.
///
/// The accumulator owns the provider's own turn/step counters, the finalized output ids (so a
/// block is finalized once even when both a `block-end` chunk and the final message carry it), the
/// per-step usage (summed into the cumulative report) and the retained assistant messages.
pub(super) struct DshAccumulator {
    /// Turn id every observation is stamped with.
    turn_id: String,
    /// Root session id; an event of another session belongs to a child.
    root_session_id: String,
    /// Retained assistant messages of the turn.
    messages: Vec<TurnMessage>,
    /// Message ids already retained.
    message_ids: BTreeSet<String>,
    /// Output ids already finalized.
    finalized_outputs: BTreeSet<String>,
    /// Usage of every `(turn, step)` pair, summed into the cumulative report.
    usage_by_step: BTreeMap<(u64, u64), TurnUsage>,
    /// Tool name of every started call, keyed by call id.
    tool_names: BTreeMap<String, String>,
    /// Data of the last `turn/end`.
    last_turn_reason: Option<JsonObject>,
    /// Number of native `turn/end` events observed.
    native_turn_end_count: u64,
    /// Text of the last finalized answer block, the `final_output` projection.
    last_text_output: String,
}

impl DshAccumulator {
    /// Builds an accumulator for one turn of the root session.
    pub(super) fn new(turn_id: &str, root_session_id: &str) -> Self {
        Self {
            turn_id: turn_id.to_owned(),
            root_session_id: root_session_id.to_owned(),
            messages: Vec::new(),
            message_ids: BTreeSet::new(),
            finalized_outputs: BTreeSet::new(),
            usage_by_step: BTreeMap::new(),
            tool_names: BTreeMap::new(),
            last_turn_reason: None,
            native_turn_end_count: 0,
            last_text_output: String::new(),
        }
    }

    /// Returns whether `session_id` is the root session (a missing id counts as root).
    pub(super) fn is_root(&self, session_id: Option<&str>) -> bool {
        session_id.is_none_or(|session_id| session_id == self.root_session_id)
    }

    /// Maps one `session.event` of the root session, returning the terminal result on `turn/end`.
    pub(super) async fn consume(
        &mut self,
        session: &TurnContext,
        event: &JsonObject,
    ) -> Option<TurnResult> {
        let event_type = event
            .get("type")
            .and_then(JsonValue::as_str)
            .unwrap_or("unknown");
        let data = object_of(event.get("data"));
        match event_type {
            "assistant/chunk" => {
                self.assistant_chunk(session, event, &data).await;
                None
            }
            "assistant/message" => {
                self.assistant_message(session, &data).await;
                None
            }
            "tool/call" => {
                self.tool_call(session, &data).await;
                None
            }
            "tool/result" => {
                self.tool_result(session, &data).await;
                None
            }
            "step/start" | "step/end" => {
                self.step(session, event_type, &data).await;
                None
            }
            "turn/end" => {
                self.last_turn_reason = Some(object_of(data.get("reason")));
                self.native_turn_end_count = self.native_turn_end_count.saturating_add(1);
                session
                    .provider_event(
                        &self.turn_id,
                        PROVIDER_EVENT_NAME,
                        event_type,
                        json!({"event": redact_turn_end(event)}),
                    )
                    .await;
                Some(self.build_terminal())
            }
            other => {
                self.forward(
                    session,
                    other,
                    &json!({"event": json_object_to_value(event)}),
                )
                .await;
                None
            }
        }
    }

    /// Forwards one notification of any kind as a provider event.
    pub(super) async fn forward(
        &self,
        session: &TurnContext,
        event_type: &str,
        payload: &JsonValue,
    ) {
        session
            .provider_event(
                &self.turn_id,
                PROVIDER_EVENT_NAME,
                event_type,
                payload.clone(),
            )
            .await;
    }

    /// Maps one `subagent.started` / `subagent.finished` notification onto a subagent item.
    pub(super) async fn consume_subagent(
        &self,
        session: &TurnContext,
        method: &str,
        payload: &JsonObject,
    ) {
        let child_id = payload
            .get("childSessionId")
            .or_else(|| payload.get("sessionId"))
            .and_then(JsonValue::as_str)
            .unwrap_or("unknown-subagent");
        let kind = if method == "subagent.started" {
            ItemEventKind::Started
        } else {
            ItemEventKind::Completed
        };
        session
            .item(
                &self.turn_id,
                ItemLifecycleEvent {
                    kind,
                    item_type: SUBAGENT_ITEM_TYPE.to_owned(),
                    data: payload.clone(),
                },
                child_id,
            )
            .await;
    }

    /// Maps one child session's event onto a `child.session.event` provider event.
    pub(super) async fn consume_child(
        &self,
        session: &TurnContext,
        session_id: Option<&str>,
        event: &JsonObject,
    ) {
        let redacted = if event.get("type").and_then(JsonValue::as_str) == Some("turn/end") {
            redact_turn_end(event)
        } else {
            json_object_to_value(event)
        };
        let mut payload = JsonObject::new();
        payload.insert("event".to_owned(), redacted);
        if let Some(session_id) = session_id {
            payload.insert("sessionId".to_owned(), json!(session_id));
        }
        self.forward(session, CHILD_EVENT_TYPE, &json_object_to_value(&payload))
            .await;
    }

    /// Builds the terminal result of a turn whose stream ended before a `turn/end`.
    pub(super) fn build_open_ended(&self, stopped: bool) -> TurnResult {
        if stopped {
            return self.attach(interrupted_turn(
                TurnTerminationKind::HarnessStop,
                STOPPED_MESSAGE,
            ));
        }
        self.attach(failed_turn(turn_error(
            MISSING_TURN_END_MESSAGE,
            MISSING_TURN_END_CODE,
            "protocol",
            false,
        )))
    }

    /// Builds the terminal result of a turn the transport itself could not complete.
    pub(super) fn build_transport_failure(&self, message: &str) -> TurnResult {
        self.attach(failed_turn(turn_error(
            message,
            TRANSPORT_CODE,
            "protocol",
            false,
        )))
    }

    /// Builds the terminal result from the recorded `turn/end` reason.
    fn build_terminal(&self) -> TurnResult {
        let reason = self.last_turn_reason.clone().unwrap_or_default();
        let Some(finish_reason) = reason.get("kind").and_then(JsonValue::as_str) else {
            return self.attach(failed_turn(turn_error(
                MISSING_TURN_END_MESSAGE,
                MISSING_TURN_END_CODE,
                "protocol",
                false,
            )));
        };
        match finish_reason {
            "completed" | "max-tokens" => self.attach(completed_turn(
                JsonValue::String(self.last_text_output.clone()),
                self.messages.clone(),
                self.total_usage(),
            )),
            "blocked" => self.attach(interrupted_turn(
                TurnTerminationKind::Policy,
                BLOCKED_MESSAGE,
            )),
            "aborted" | "interrupted" => {
                let kind = termination_kind(finish_reason, &reason);
                let mut result = self.attach(interrupted_turn(kind, INTERRUPTED_MESSAGE));
                if let Some(termination) = result.termination.as_mut() {
                    termination.provider_data = safe_reason_data(&reason);
                }
                result
            }
            "error" => self.attach(failed_turn(TurnError {
                message: FAILED_TURN_MESSAGE.to_owned(),
                code: reason_error_code(&reason),
                category: Some("provider".to_owned()),
                retryable: Some(false),
                provider_data: safe_reason_data(&reason),
            })),
            other => {
                let mut provider_data = JsonObject::new();
                provider_data.insert("finish_reason".to_owned(), json!(other));
                self.attach(failed_turn(TurnError {
                    message: UNKNOWN_FINISH_MESSAGE.to_owned(),
                    code: Some(UNKNOWN_FINISH_CODE.to_owned()),
                    category: Some("protocol".to_owned()),
                    retryable: Some(false),
                    provider_data,
                }))
            }
        }
    }

    /// Attaches the accumulated messages, final output and usage to `result`.
    fn attach(&self, mut result: TurnResult) -> TurnResult {
        result.messages.clone_from(&self.messages);
        result.final_output = JsonValue::String(self.last_text_output.clone());
        result.usage.clone_from(&self.total_usage());
        result.provider_data.insert(
            "native_turn_end_count".to_owned(),
            json!(self.native_turn_end_count),
        );
        result
    }

    /// Returns the cumulative usage of every model-call step, when the runtime reported any.
    fn total_usage(&self) -> Option<TurnUsage> {
        if self.usage_by_step.is_empty() {
            return None;
        }
        let mut input_tokens = 0;
        let mut output_tokens = 0;
        let mut cached_input_tokens = 0;
        let mut reasoning_output_tokens = 0;
        let mut cache_write_tokens = 0;
        for usage in self.usage_by_step.values() {
            input_tokens += usage.input_tokens.unwrap_or(0);
            output_tokens += usage.output_tokens.unwrap_or(0);
            cached_input_tokens += usage.cached_input_tokens.unwrap_or(0);
            reasoning_output_tokens += usage.reasoning_output_tokens.unwrap_or(0);
            cache_write_tokens += usage
                .provider_data
                .get("cache_write_tokens")
                .and_then(JsonValue::as_u64)
                .unwrap_or(0);
        }
        let mut provider_data = JsonObject::new();
        provider_data.insert("cache_write_tokens".to_owned(), json!(cache_write_tokens));
        Some(TurnUsage {
            input_tokens: Some(input_tokens),
            output_tokens: Some(output_tokens),
            cached_input_tokens: Some(cached_input_tokens),
            reasoning_output_tokens: Some(reasoning_output_tokens),
            total_tokens: Some(
                input_tokens + output_tokens + cached_input_tokens + cache_write_tokens,
            ),
            provider_data,
        })
    }

    /// Records the usage of one step and publishes the new cumulative total.
    async fn update_usage(
        &mut self,
        session: &TurnContext,
        native_turn: u64,
        native_step: u64,
        raw: &JsonObject,
    ) {
        let usage = turn_usage(raw);
        let key = (native_turn, native_step);
        if self.usage_by_step.get(&key) == Some(&usage) {
            return;
        }
        self.usage_by_step.insert(key, usage);
        if let Some(total) = self.total_usage() {
            session.usage(&self.turn_id, total).await;
        }
    }

    /// Maps one `assistant/chunk` event.
    async fn assistant_chunk(
        &mut self,
        session: &TurnContext,
        event: &JsonObject,
        data: &JsonObject,
    ) {
        let chunk = object_of(data.get("chunk"));
        let chunk_type = chunk
            .get("type")
            .and_then(JsonValue::as_str)
            .unwrap_or("unknown");
        let native_turn = unsigned(data.get("turn")).unwrap_or(0);
        let native_step = unsigned(data.get("step")).unwrap_or(0);
        let content_index = unsigned(chunk.get("index")).unwrap_or(0);
        match chunk_type {
            "text-delta" | "reasoning-delta" => {
                let Some(text) = chunk.get("text").and_then(JsonValue::as_str) else {
                    return;
                };
                let channel = delta_channel(chunk_type);
                session
                    .output(
                        &self.turn_id,
                        output_event(
                            native_turn,
                            native_step,
                            content_index,
                            channel,
                            json!(text),
                            OutputOperation::Delta,
                        ),
                    )
                    .await;
            }
            "block-end" => {
                let block = object_of(chunk.get("block"));
                let Some(block_type) = block.get("type").and_then(JsonValue::as_str) else {
                    self.forward_unknown_chunk(session, event).await;
                    return;
                };
                if !matches!(block_type, "text" | "reasoning") {
                    self.forward_unknown_chunk(session, event).await;
                    return;
                }
                let channel = block_channel(block_type);
                let text = block
                    .get("text")
                    .and_then(JsonValue::as_str)
                    .unwrap_or_default();
                self.finalize_output(native_turn, native_step, content_index, channel, text);
                session
                    .output(
                        &self.turn_id,
                        output_event(
                            native_turn,
                            native_step,
                            content_index,
                            channel,
                            json!(text),
                            OutputOperation::Final,
                        ),
                    )
                    .await;
            }
            "usage" => {
                let usage = object_of(chunk.get("usage"));
                self.update_usage(session, native_turn, native_step, &usage)
                    .await;
            }
            _ => self.forward_unknown_chunk(session, event).await,
        }
    }

    /// Forwards an `assistant/chunk` the adapter does not project.
    async fn forward_unknown_chunk(&self, session: &TurnContext, event: &JsonObject) {
        self.forward(
            session,
            "assistant/chunk",
            &json!({"event": json_object_to_value(event)}),
        )
        .await;
    }

    /// Maps one `assistant/message` event: final block outputs plus the retained message.
    async fn assistant_message(&mut self, session: &TurnContext, data: &JsonObject) {
        let native_turn = unsigned(data.get("turn")).unwrap_or(0);
        let native_step = unsigned(data.get("step")).unwrap_or(0);
        let message = match data.get("message") {
            Some(message @ JsonValue::Object(_)) => object_of(Some(message)),
            _ => data.clone(),
        };
        let content: Vec<JsonValue> = message
            .get("content")
            .and_then(JsonValue::as_array)
            .cloned()
            .unwrap_or_default();
        let message_id = message.get("id").and_then(JsonValue::as_str).map_or_else(
            || {
                format!(
                    "dsh-message:{native_turn}:{native_step}:{}",
                    unsigned(data.get("seq")).unwrap_or(0)
                )
            },
            str::to_owned,
        );
        let mut blocks = Vec::with_capacity(content.len());
        let mut answer_parts = Vec::new();
        for (index, raw_block) in content.iter().enumerate() {
            let block = object_of(Some(raw_block));
            let block_type = block
                .get("type")
                .and_then(JsonValue::as_str)
                .unwrap_or("unknown");
            let block_index = u64::try_from(index).unwrap_or(u64::MAX);
            let block_id = block
                .get("id")
                .and_then(JsonValue::as_str)
                .map_or_else(|| format!("{message_id}:block:{index}"), str::to_owned);
            blocks.push(content_block(&block_id, block_type, &block));
            if !matches!(block_type, "text" | "reasoning") {
                continue;
            }
            let channel = block_channel(block_type);
            let text = block
                .get("text")
                .and_then(JsonValue::as_str)
                .unwrap_or_default();
            if channel == OutputChannel::Answer {
                answer_parts.push(text.to_owned());
            }
            if self.finalize_output(native_turn, native_step, block_index, channel, text) {
                session
                    .output(
                        &self.turn_id,
                        output_event(
                            native_turn,
                            native_step,
                            block_index,
                            channel,
                            json!(text),
                            OutputOperation::Final,
                        ),
                    )
                    .await;
            }
        }
        self.last_text_output = answer_parts.concat();
        if self.message_ids.insert(message_id.clone()) {
            let mut message_data = JsonObject::new();
            message_data.insert("native_turn".to_owned(), json!(native_turn));
            message_data.insert("native_step".to_owned(), json!(native_step));
            let source = object_of(message.get("source"));
            if !source.is_empty() {
                message_data.insert("source".to_owned(), json_object_to_value(&source));
            }
            self.messages.push(TurnMessage {
                message_id,
                role: MessageRole::Assistant,
                content: blocks,
                data: message_data,
            });
        }
        let usage = object_of(data.get("usage"));
        if !usage.is_empty() {
            self.update_usage(session, native_turn, native_step, &usage)
                .await;
        }
    }

    /// Records one finalized output, returning whether it was not finalized before.
    fn finalize_output(
        &mut self,
        native_turn: u64,
        native_step: u64,
        content_index: u64,
        channel: OutputChannel,
        text: &str,
    ) -> bool {
        let output_id = output_id(native_turn, native_step, content_index, channel);
        if !self.finalized_outputs.insert(output_id) {
            return false;
        }
        if channel == OutputChannel::Answer {
            text.clone_into(&mut self.last_text_output);
        }
        true
    }

    /// Maps one `tool/call` event onto the start of a `tool_call` item.
    async fn tool_call(&mut self, session: &TurnContext, data: &JsonObject) {
        let call_id = data
            .get("callId")
            .and_then(JsonValue::as_str)
            .unwrap_or(UNKNOWN_CALL_ID);
        let name = data
            .get("name")
            .and_then(JsonValue::as_str)
            .unwrap_or(UNKNOWN_TOOL_NAME);
        self.tool_names.insert(call_id.to_owned(), name.to_owned());
        let arguments = match data.get("arguments") {
            Some(JsonValue::String(text)) => {
                serde_json::from_str(text).unwrap_or_else(|_| json!(text))
            }
            Some(other) => other.clone(),
            None => JsonValue::Null,
        };
        session
            .tool_started(&self.turn_id, call_id, Some(call_id), name, arguments)
            .await;
    }

    /// Maps one `tool/result` event onto the completion of a `tool_call` item.
    async fn tool_result(&mut self, session: &TurnContext, data: &JsonObject) {
        let message = object_of(data.get("message"));
        let call_id = message
            .get("source")
            .and_then(JsonValue::as_object)
            .and_then(|source| source.get("callId"))
            .or_else(|| data.get("callId"))
            .and_then(JsonValue::as_str)
            .unwrap_or(UNKNOWN_CALL_ID);
        let content = message
            .get("content")
            .or_else(|| data.get("content"))
            .cloned()
            .unwrap_or(JsonValue::Null);
        let name = self
            .tool_names
            .get(call_id)
            .cloned()
            .unwrap_or_else(|| UNKNOWN_TOOL_NAME.to_owned());
        match data.get("error").filter(|error| !error.is_null()) {
            Some(error) => {
                session
                    .tool_completed(
                        &self.turn_id,
                        call_id,
                        Some(call_id),
                        &name,
                        json!(error_text(error)),
                        None,
                    )
                    .await;
                let mut diagnostic = JsonObject::new();
                diagnostic.insert("tool_call_id".to_owned(), json!(call_id));
                diagnostic.insert("tool_name".to_owned(), json!(name));
                diagnostic.insert("error".to_owned(), (*error).clone());
                session
                    .diagnostic(
                        &self.turn_id,
                        DiagnosticLevel::Warning,
                        "DeepSeek Harness reported a failed tool call",
                        diagnostic,
                    )
                    .await;
            }
            None => {
                session
                    .tool_completed(&self.turn_id, call_id, Some(call_id), &name, content, None)
                    .await;
            }
        }
    }

    /// Maps one `step/start` or `step/end` event onto a `step` item.
    async fn step(&self, session: &TurnContext, event_type: &str, data: &JsonObject) {
        let native_turn = unsigned(data.get("turn")).unwrap_or(0);
        let native_step = unsigned(data.get("step")).unwrap_or(0);
        let mut item_data = JsonObject::new();
        item_data.insert("provider_turn".to_owned(), json!(native_turn));
        item_data.insert("provider_step".to_owned(), json!(native_step));
        let kind = if event_type == "step/start" {
            ItemEventKind::Started
        } else {
            ItemEventKind::Completed
        };
        session
            .item(
                &self.turn_id,
                ItemLifecycleEvent {
                    kind,
                    item_type: STEP_ITEM_TYPE.to_owned(),
                    data: item_data,
                },
                &format!("dsh-step:{native_turn}:{native_step}"),
            )
            .await;
    }
}

/// Returns the object a JSON value holds, or an empty object.
fn object_of(value: Option<&JsonValue>) -> JsonObject {
    match value.and_then(JsonValue::as_object) {
        Some(object) => object
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect(),
        None => JsonObject::new(),
    }
}

/// Returns the unsigned value of `value`, when it is one.
fn unsigned(value: Option<&JsonValue>) -> Option<u64> {
    value.and_then(JsonValue::as_u64)
}

/// Returns the channel of a delta chunk.
fn delta_channel(chunk_type: &str) -> OutputChannel {
    if chunk_type == "reasoning-delta" {
        OutputChannel::Reasoning
    } else {
        OutputChannel::Answer
    }
}

/// Returns the channel of a finalized block.
fn block_channel(block_type: &str) -> OutputChannel {
    if block_type == "reasoning" {
        OutputChannel::Reasoning
    } else {
        OutputChannel::Answer
    }
}

/// Returns the stable output id of one turn/step/content position.
fn output_id(
    native_turn: u64,
    native_step: u64,
    content_index: u64,
    channel: OutputChannel,
) -> String {
    format!(
        "dsh-output:{native_turn}:{native_step}:{content_index}:{}",
        channel.as_str()
    )
}

/// Builds one output event.
fn output_event(
    native_turn: u64,
    native_step: u64,
    content_index: u64,
    channel: OutputChannel,
    content: JsonValue,
    operation: OutputOperation,
) -> OutputEvent {
    OutputEvent {
        output_id: output_id(native_turn, native_step, content_index, channel),
        kind: OutputKind::Text,
        content,
        operation,
        channel,
        content_index,
        content_type: None,
        data: JsonObject::new(),
    }
}

/// Builds one normalized content block of a retained message.
fn content_block(block_id: &str, block_type: &str, block: &JsonObject) -> ContentBlock {
    if matches!(block_type, "text" | "reasoning") {
        return ContentBlock {
            block_id: block_id.to_owned(),
            kind: block_type.to_owned(),
            content: json!(block
                .get("text")
                .and_then(JsonValue::as_str)
                .unwrap_or_default()),
            data: JsonObject::new(),
        };
    }
    if block_type == "tool-call" {
        let mut content = JsonObject::new();
        content.insert(
            "name".to_owned(),
            json!(block
                .get("name")
                .and_then(JsonValue::as_str)
                .unwrap_or(UNKNOWN_TOOL_NAME)),
        );
        content.insert(
            "arguments".to_owned(),
            block.get("arguments").cloned().unwrap_or(JsonValue::Null),
        );
        let mut data = JsonObject::new();
        data.insert(
            "call_id".to_owned(),
            json!(block
                .get("id")
                .and_then(JsonValue::as_str)
                .unwrap_or(block_id)),
        );
        return ContentBlock {
            block_id: block_id.to_owned(),
            kind: "tool_call".to_owned(),
            content: json_object_to_value(&content),
            data,
        };
    }
    ContentBlock {
        block_id: block_id.to_owned(),
        kind: block_type.to_owned(),
        content: json_object_to_value(block),
        data: JsonObject::new(),
    }
}

/// Maps a DSH usage object onto the protocol type.
fn turn_usage(raw: &JsonObject) -> TurnUsage {
    let input_tokens = non_negative(raw.get("inputTokens"));
    let output_tokens = non_negative(raw.get("outputTokens"));
    let cached_input_tokens = non_negative(raw.get("cacheReadTokens"));
    let cache_write_tokens = non_negative(raw.get("cacheWriteTokens"));
    let reasoning_output_tokens = non_negative(raw.get("reasoningTokens"));
    let mut provider_data = JsonObject::new();
    provider_data.insert(
        "cache_write_tokens".to_owned(),
        json!(cache_write_tokens.unwrap_or(0)),
    );
    TurnUsage {
        input_tokens,
        output_tokens,
        cached_input_tokens,
        reasoning_output_tokens,
        total_tokens: Some(
            input_tokens.unwrap_or(0)
                + output_tokens.unwrap_or(0)
                + cached_input_tokens.unwrap_or(0)
                + cache_write_tokens.unwrap_or(0),
        ),
        provider_data,
    }
}

/// Returns the counter as an unsigned value, when it is a non-negative integer.
fn non_negative(value: Option<&JsonValue>) -> Option<u64> {
    value.and_then(JsonValue::as_u64)
}

/// Returns the termination kind of an `aborted`/`interrupted` ending.
fn termination_kind(finish_reason: &str, reason: &JsonObject) -> TurnTerminationKind {
    if finish_reason == "interrupted" {
        return TurnTerminationKind::Provider;
    }
    let cause = reason.get("reason").and_then(JsonValue::as_object);
    match cause
        .and_then(|cause| cause.get("kind"))
        .and_then(JsonValue::as_str)
    {
        Some("user") => TurnTerminationKind::UserAbort,
        Some("disposed") => TurnTerminationKind::HarnessStop,
        Some("hook") => TurnTerminationKind::Policy,
        _ => TurnTerminationKind::Provider,
    }
}

/// Returns the error code of a failed ending.
fn reason_error_code(reason: &JsonObject) -> Option<String> {
    let error = reason.get("error").and_then(JsonValue::as_object)?;
    match error.get("code") {
        Some(JsonValue::String(code)) => Some(code.clone()),
        Some(JsonValue::Number(code)) => Some(code.to_string()),
        _ => None,
    }
}

/// Returns the non-message failure fields of a reason, without provider diagnostics.
fn safe_reason_data(reason: &JsonObject) -> JsonObject {
    let mut safe = JsonObject::new();
    if let Some(kind) = reason.get("kind").and_then(JsonValue::as_str) {
        safe.insert("kind".to_owned(), json!(kind));
    }
    if let Some(cause) = reason.get("reason").and_then(JsonValue::as_object) {
        let filtered: JsonObject = cause
            .iter()
            .filter(|(key, _)| key.as_str() != "message")
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        if !filtered.is_empty() {
            safe.insert("cause".to_owned(), json_object_to_value(&filtered));
        }
    }
    if let Some(error) = reason.get("error").and_then(JsonValue::as_object) {
        let filtered: JsonObject = error
            .iter()
            .filter(|(key, _)| SAFE_ERROR_KEYS.contains(&key.as_str()))
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        if !filtered.is_empty() {
            safe.insert("error".to_owned(), json_object_to_value(&filtered));
        }
    }
    safe
}

/// Returns the shape of a `turn/end` event without its provider error text.
fn redact_turn_end(event: &JsonObject) -> JsonValue {
    let mut safe = event.clone();
    let mut data = object_of(event.get("data"));
    let reason = redacted_reason(&object_of(data.get("reason")));
    data.insert("reason".to_owned(), json_object_to_value(&reason));
    safe.insert("data".to_owned(), json_object_to_value(&data));
    json_object_to_value(&safe)
}

/// Returns the redacted `reason` object of a `turn/end`.
fn redacted_reason(reason: &JsonObject) -> JsonObject {
    let mut safe = JsonObject::new();
    if let Some(kind) = reason.get("kind").and_then(JsonValue::as_str) {
        safe.insert("kind".to_owned(), json!(kind));
    }
    if let Some(cause) = reason.get("reason").and_then(JsonValue::as_object) {
        let filtered: JsonObject = cause
            .iter()
            .filter(|(key, _)| key.as_str() != "message")
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        if !filtered.is_empty() {
            safe.insert("reason".to_owned(), json_object_to_value(&filtered));
        }
    }
    if let Some(error) = reason.get("error").and_then(JsonValue::as_object) {
        let filtered: JsonObject = error
            .iter()
            .filter(|(key, _)| SAFE_ERROR_KEYS.contains(&key.as_str()))
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        if !filtered.is_empty() {
            safe.insert("error".to_owned(), json_object_to_value(&filtered));
        }
    }
    safe
}

/// Renders a tool error as the completion text.
fn error_text(error: &JsonValue) -> String {
    match error {
        JsonValue::String(text) => text.clone(),
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}
