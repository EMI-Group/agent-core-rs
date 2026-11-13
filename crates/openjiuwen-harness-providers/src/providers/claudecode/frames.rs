//! Frame projection of the Claude Code transport.
//!
//! Every method here turns one decoded CLI frame into harness observations (outputs, tool calls,
//! tool results, usage, provider events, permission prompts) or into the turn's terminal result.
//! The turn loop, process ownership and lifecycle live in `harness`; the argv and environment live
//! in `wire`.

use std::collections::BTreeMap;

use openjiuwen_harness_protocol::{
    ContentBlock, DiagnosticLevel, HarnessResult, JsonObject, JsonValue, MessageRole,
    OutputChannel, OutputEvent, OutputKind, OutputOperation, TurnError, TurnMessage, TurnResult,
    TurnUsage,
};
use serde_json::json;

use crate::vendor_cli::{
    ask_tool_approval, ask_user_input, completed_turn, failed_turn, turn_error, ApprovalDecision,
    ChildProcess, TurnContext,
};

use super::harness::{lock, PROVIDER_EVENT_NAME, SDK_ERROR_CATEGORY};
use super::ClaudeDriver;

/// Tool name the CLI uses for its user-input prompt.
const ASK_USER_QUESTION: &str = "AskUserQuestion";

impl ClaudeDriver {
    /// Maps one decoded stdout frame onto observations, or returns the terminal result.
    pub(super) async fn dispatch(
        &self,
        frame: &Frame,
        turn_id: &str,
        session: &TurnContext,
        accumulator: &mut TurnAccumulator,
        process: &mut ChildProcess,
    ) -> HarnessResult<FrameAction> {
        match frame.get("type").and_then(JsonValue::as_str) {
            Some("system") => {
                self.handle_system(frame, turn_id, session).await;
                Ok(FrameAction::Continue)
            }
            Some("assistant") => {
                self.handle_assistant(frame, turn_id, session, accumulator)
                    .await;
                Ok(FrameAction::Continue)
            }
            Some("user") => {
                self.handle_user(frame, turn_id, session, accumulator).await;
                Ok(FrameAction::Continue)
            }
            Some("stream_event") => {
                self.handle_stream_event(frame, turn_id, session, accumulator)
                    .await;
                Ok(FrameAction::Continue)
            }
            Some("control_request") => {
                self.handle_control_request(frame, turn_id, session, process)
                    .await?;
                Ok(FrameAction::Continue)
            }
            Some("control_response") => Ok(FrameAction::Continue),
            Some("result") => {
                let tail = process.stderr_tail();
                Ok(FrameAction::Terminal(Box::new(
                    self.handle_result(frame, turn_id, session, accumulator, &tail)
                        .await,
                )))
            }
            other => {
                let event_type = other.unwrap_or("unknown");
                session
                    .provider_event(
                        turn_id,
                        PROVIDER_EVENT_NAME,
                        event_type,
                        JsonValue::Object(frame.clone()),
                    )
                    .await;
                Ok(FrameAction::Continue)
            }
        }
    }

    /// Handles a `system` frame: captures the session id and forwards the frame.
    async fn handle_system(&self, frame: &Frame, turn_id: &str, session: &TurnContext) {
        let subtype = frame
            .get("subtype")
            .and_then(JsonValue::as_str)
            .unwrap_or("system");
        if let Some(id) = frame.get("session_id").and_then(JsonValue::as_str) {
            self.adopt_session(id);
        }
        session
            .provider_event(
                turn_id,
                PROVIDER_EVENT_NAME,
                &format!("system/{subtype}"),
                JsonValue::Object(frame.clone()),
            )
            .await;
    }

    /// Handles an `assistant` frame: outputs, tool calls, usage and message history.
    async fn handle_assistant(
        &self,
        frame: &Frame,
        turn_id: &str,
        session: &TurnContext,
        accumulator: &mut TurnAccumulator,
    ) {
        let Some(message) = frame.get("message").and_then(JsonValue::as_object) else {
            session
                .provider_event(
                    turn_id,
                    PROVIDER_EVENT_NAME,
                    "assistant",
                    JsonValue::Object(frame.clone()),
                )
                .await;
            return;
        };
        let message_index = accumulator.begin_message();
        let parent = parent_tool_use_id(frame.get("parent_tool_use_id"))
            .or_else(|| parent_tool_use_id(message.get("parent_tool_use_id")))
            .is_some();
        let mut blocks = Vec::new();
        if let Some(JsonValue::Array(content)) = message.get("content") {
            for (index, block) in content.iter().enumerate() {
                let content_index = u64::try_from(index).unwrap_or(u64::MAX);
                if let Some(block) = self
                    .assistant_block(
                        block,
                        message_index,
                        content_index,
                        parent,
                        turn_id,
                        session,
                        accumulator,
                    )
                    .await
                {
                    blocks.push(block);
                }
            }
        }
        if let Some(usage) = message.get("usage").and_then(JsonValue::as_object) {
            if let Some(mapped) =
                usage_from(usage, message.get("model").and_then(JsonValue::as_str))
            {
                accumulator.merge_usage(&mapped);
                session.usage(turn_id, mapped).await;
            }
        }
        if blocks.is_empty() {
            return;
        }
        let mut data = JsonObject::new();
        if let Some(model) = message.get("model").and_then(JsonValue::as_str) {
            data.insert("model".to_owned(), json!(model));
        }
        accumulator.messages.push(TurnMessage {
            message_id: frame
                .get("uuid")
                .and_then(JsonValue::as_str)
                .map_or_else(|| format!("claude-message:{message_index}"), str::to_owned),
            role: MessageRole::Assistant,
            content: blocks,
            data,
        });
    }
    /// Projects one `assistant` content block, publishing the observations it carries.
    #[allow(clippy::too_many_arguments)]
    async fn assistant_block(
        &self,
        block: &JsonValue,
        message_index: u64,
        content_index: u64,
        parent: bool,
        turn_id: &str,
        session: &TurnContext,
        accumulator: &mut TurnAccumulator,
    ) -> Option<ContentBlock> {
        let block = block.as_object()?;
        match block.get("type").and_then(JsonValue::as_str) {
            Some("text") => {
                let text = block
                    .get("text")
                    .and_then(JsonValue::as_str)
                    .unwrap_or_default();
                if !text.is_empty() && !parent {
                    session
                        .output(
                            turn_id,
                            output_event(
                                message_index,
                                content_index,
                                OutputChannel::Answer,
                                json!({"text": text}),
                                OutputOperation::Final,
                            ),
                        )
                        .await;
                }
                Some(text_block(message_index, content_index, "text", text))
            }
            Some("thinking") => {
                let thinking = block
                    .get("thinking")
                    .and_then(JsonValue::as_str)
                    .unwrap_or_default();
                if !thinking.is_empty() && !parent {
                    session
                        .output(
                            turn_id,
                            output_event(
                                message_index,
                                content_index,
                                OutputChannel::Reasoning,
                                json!({"thinking": thinking}),
                                OutputOperation::Final,
                            ),
                        )
                        .await;
                }
                Some(text_block(
                    message_index,
                    content_index,
                    "reasoning",
                    thinking,
                ))
            }
            Some("tool_use") => {
                let call_id = block.get("id").and_then(JsonValue::as_str).map_or_else(
                    || format!("claude-tool:{message_index}:{content_index}"),
                    str::to_owned,
                );
                let name = block
                    .get("name")
                    .and_then(JsonValue::as_str)
                    .unwrap_or_default()
                    .to_owned();
                let arguments = block.get("input").cloned().unwrap_or(JsonValue::Null);
                accumulator.tool_names.insert(call_id.clone(), name.clone());
                let mut data = JsonObject::new();
                data.insert("call_id".to_owned(), json!(call_id));
                if !parent {
                    session
                        .tool_started(turn_id, &call_id, Some(&call_id), &name, arguments.clone())
                        .await;
                }
                Some(ContentBlock {
                    block_id: format!("claude-block:{message_index}:{content_index}"),
                    kind: "tool_call".to_owned(),
                    content: json!({"name": name, "arguments": arguments}),
                    data,
                })
            }
            other => Some(ContentBlock {
                block_id: format!("claude-block:{message_index}:{content_index}"),
                kind: other.unwrap_or("unknown").to_owned(),
                content: JsonValue::Object(block.to_owned()),
                data: JsonObject::new(),
            }),
        }
    }
    /// Handles a `user` frame: tool results only.
    async fn handle_user(
        &self,
        frame: &Frame,
        turn_id: &str,
        session: &TurnContext,
        accumulator: &mut TurnAccumulator,
    ) {
        let Some(message) = frame.get("message").and_then(JsonValue::as_object) else {
            return;
        };
        let Some(JsonValue::Array(content)) = message.get("content") else {
            return;
        };
        let mut blocks = Vec::new();
        for (index, block) in content.iter().enumerate() {
            let Some(block) = block.as_object() else {
                continue;
            };
            if block.get("type").and_then(JsonValue::as_str) != Some("tool_result") {
                continue;
            }
            let Some(tool_use_id) = block.get("tool_use_id").and_then(JsonValue::as_str) else {
                continue;
            };
            let result = block.get("content").cloned().unwrap_or(JsonValue::Null);
            let mut data = JsonObject::new();
            data.insert("call_id".to_owned(), json!(tool_use_id));
            blocks.push(ContentBlock {
                block_id: format!("claude-result:{tool_use_id}:{index}"),
                kind: "tool_result".to_owned(),
                content: result.clone(),
                data,
            });
            // The name announced by the matching `tool_use` block; a result for a call the CLI
            // never announced (for example one issued before a resume) falls back to its id.
            let name = accumulator
                .tool_names
                .get(tool_use_id)
                .cloned()
                .unwrap_or_else(|| tool_use_id.to_owned());
            session
                .tool_completed(turn_id, tool_use_id, Some(tool_use_id), &name, result, None)
                .await;
        }
        if blocks.is_empty() {
            return;
        }
        accumulator.messages.push(TurnMessage {
            message_id: frame.get("uuid").and_then(JsonValue::as_str).map_or_else(
                || format!("claude-tool-message:{}", accumulator.messages.len()),
                str::to_owned,
            ),
            role: MessageRole::Tool,
            content: blocks,
            data: JsonObject::new(),
        });
    }

    /// Handles a `stream_event` frame: partial text/thinking deltas.
    async fn handle_stream_event(
        &self,
        frame: &Frame,
        turn_id: &str,
        session: &TurnContext,
        accumulator: &mut TurnAccumulator,
    ) {
        if !self.config.include_partial_messages
            || parent_tool_use_id(frame.get("parent_tool_use_id")).is_some()
        {
            return;
        }
        let Some(event) = frame.get("event").and_then(JsonValue::as_object) else {
            return;
        };
        match event.get("type").and_then(JsonValue::as_str) {
            Some("message_start") => accumulator.begin_stream_message(),
            Some("message_stop") => accumulator.stream_open = false,
            Some("content_block_delta") => {
                let Some(delta) = event.get("delta").and_then(JsonValue::as_object) else {
                    return;
                };
                let content_index = event.get("index").and_then(JsonValue::as_u64).unwrap_or(0);
                let (channel, content) = match delta.get("type").and_then(JsonValue::as_str) {
                    Some("text_delta") => match delta.get("text").and_then(JsonValue::as_str) {
                        Some(text) if !text.is_empty() => {
                            (OutputChannel::Answer, json!({"text": text}))
                        }
                        _ => return,
                    },
                    Some("thinking_delta") => {
                        match delta.get("thinking").and_then(JsonValue::as_str) {
                            Some(thinking) if !thinking.is_empty() => {
                                (OutputChannel::Reasoning, json!({"thinking": thinking}))
                            }
                            _ => return,
                        }
                    }
                    _ => return,
                };
                session
                    .output(
                        turn_id,
                        output_event(
                            accumulator.message_index,
                            content_index,
                            channel,
                            content,
                            OutputOperation::Delta,
                        ),
                    )
                    .await;
            }
            _ => {}
        }
    }

    /// Answers a `control_request/can_use_tool` permission prompt.
    async fn handle_control_request(
        &self,
        frame: &Frame,
        turn_id: &str,
        session: &TurnContext,
        process: &mut ChildProcess,
    ) -> HarnessResult<()> {
        let Some(request) = frame.get("request").and_then(JsonValue::as_object) else {
            return Ok(());
        };
        if request.get("subtype").and_then(JsonValue::as_str) != Some("can_use_tool") {
            session
                .provider_event(
                    turn_id,
                    PROVIDER_EVENT_NAME,
                    "control_request",
                    JsonValue::Object(frame.clone()),
                )
                .await;
            return Ok(());
        }
        let request_id = frame
            .get("request_id")
            .and_then(JsonValue::as_str)
            .unwrap_or_default()
            .to_owned();
        let tool_name = request
            .get("tool_name")
            .and_then(JsonValue::as_str)
            .unwrap_or_default()
            .to_owned();
        let tool_use_id = request
            .get("tool_use_id")
            .and_then(JsonValue::as_str)
            .map_or_else(|| request_id.clone(), str::to_owned);
        let input = request.get("input").cloned().unwrap_or(JsonValue::Null);
        let (policy, handler) = {
            let state = lock(&self.state);
            (state.policy, state.handler.clone())
        };
        let response = if !policy.interactive() {
            // The CLI only asks when the host declared a capability; answer it anyway so a stray
            // prompt can never wedge the turn.
            allow_response(None)
        } else if tool_name == ASK_USER_QUESTION {
            let provider_data = interaction_data(&tool_use_id, &tool_name);
            match ask_user_input(
                handler.as_ref(),
                &policy,
                format!("claude-ask:{tool_use_id}"),
                render_question(&input),
                question_choices(&input),
                provider_data,
            )
            .await
            {
                Some(answers) => {
                    let mut updated = input.as_object().cloned().unwrap_or_default();
                    updated.insert("answers".to_owned(), answers);
                    allow_response(Some(JsonValue::Object(updated)))
                }
                None => deny_response("user input was not provided", false),
            }
        } else {
            let provider_data = interaction_data(&tool_use_id, &tool_name);
            match ask_tool_approval(
                handler.as_ref(),
                &policy,
                format!("claude-approval:{tool_use_id}"),
                Some(tool_use_id.clone()),
                tool_name.clone(),
                input,
                provider_data,
            )
            .await
            {
                ApprovalDecision::Allow {
                    updated_arguments: Some(arguments),
                } => allow_response(Some(arguments)),
                ApprovalDecision::Allow {
                    updated_arguments: None,
                } => allow_response(None),
                ApprovalDecision::Deny { message, interrupt } => deny_response(&message, interrupt),
            }
        };
        let frame = json!({
            "type": "control_response",
            "response": {
                "subtype": "success",
                "request_id": request_id,
                "response": response,
            },
        });
        process.write_json(&frame).await
    }

    /// Folds a terminal `result` frame into the turn result.
    async fn handle_result(
        &self,
        frame: &Frame,
        turn_id: &str,
        session: &TurnContext,
        accumulator: &mut TurnAccumulator,
        stderr_tail: &str,
    ) -> TurnResult {
        let subtype = frame
            .get("subtype")
            .and_then(JsonValue::as_str)
            .unwrap_or("error")
            .to_owned();
        if let Some(id) = frame.get("session_id").and_then(JsonValue::as_str) {
            self.adopt_session(id);
        }
        if let Some(usage) = frame.get("usage").and_then(JsonValue::as_object) {
            if let Some(mapped) = usage_from(usage, None) {
                accumulator.merge_usage(&mapped);
                session.usage(turn_id, mapped).await;
            }
        }
        let is_error = frame
            .get("is_error")
            .and_then(JsonValue::as_bool)
            .unwrap_or(subtype != "success");
        if !is_error && subtype == "success" {
            let text = frame
                .get("result")
                .and_then(JsonValue::as_str)
                .unwrap_or_default()
                .to_owned();
            return completed_turn(
                JsonValue::String(text),
                std::mem::take(&mut accumulator.messages),
                accumulator.usage.clone(),
            );
        }
        let error = result_error(frame, &subtype, stderr_tail);
        let mut data = JsonObject::new();
        data.insert("subtype".to_owned(), json!(subtype));
        if let Some(status) = frame.get("api_error_status").and_then(JsonValue::as_u64) {
            data.insert("http_status".to_owned(), json!(status));
        }
        session
            .diagnostic(turn_id, DiagnosticLevel::Error, &error.message, data)
            .await;
        failed_turn(error)
    }
}

/// One decoded JSON object off the CLI's stdout.
type Frame = serde_json::Map<String, JsonValue>;

/// Outcome of one turn-loop iteration.
pub(super) enum FrameAction {
    /// Keep reading frames.
    Continue,
    /// The turn reached its terminal result; boxed because it dwarfs `Continue`.
    Terminal(Box<TurnResult>),
}

/// Per-turn accumulator: message history, usage and output indices.
#[derive(Default)]
pub(super) struct TurnAccumulator {
    /// Index of the current assistant message (`claude-output:<index>:…`).
    message_index: u64,
    /// Whether partial-message stream events opened a message that a final frame closes.
    stream_open: bool,
    /// Normalized message history of the turn.
    messages: Vec<TurnMessage>,
    /// Tool call names by call id.
    tool_names: BTreeMap<String, String>,
    /// Usage accumulated over every frame of the turn.
    usage: Option<TurnUsage>,
}

impl TurnAccumulator {
    /// Starts an assistant message and returns its index.
    fn begin_message(&mut self) -> u64 {
        if self.stream_open {
            self.stream_open = false;
        } else {
            self.message_index += 1;
        }
        self.message_index
    }

    /// Opens a streamed message.
    fn begin_stream_message(&mut self) {
        self.message_index += 1;
        self.stream_open = true;
    }

    /// Adds `usage` to the turn totals.
    fn merge_usage(&mut self, usage: &TurnUsage) {
        let current = self.usage.get_or_insert_with(TurnUsage::default);
        add_counter(&mut current.input_tokens, usage.input_tokens);
        add_counter(&mut current.output_tokens, usage.output_tokens);
        add_counter(&mut current.cached_input_tokens, usage.cached_input_tokens);
        add_counter(
            &mut current.reasoning_output_tokens,
            usage.reasoning_output_tokens,
        );
        current.total_tokens = match (current.input_tokens, current.output_tokens) {
            (Some(input), Some(output)) => Some(input.saturating_add(output)),
            (input, output) => input.or(output),
        };
        if !usage.provider_data.is_empty() {
            current.provider_data.extend(usage.provider_data.clone());
        }
    }
}

/// Adds `value` to `target` when both are present, keeping `target` otherwise.
fn add_counter(target: &mut Option<u64>, value: Option<u64>) {
    if let Some(value) = value {
        *target = Some(target.map_or(value, |current| current.saturating_add(value)));
    }
}

/// Returns the parent tool-use id of a frame or message, when non-null.
fn parent_tool_use_id(value: Option<&JsonValue>) -> Option<String> {
    value
        .filter(|value| !value.is_null())
        .and_then(JsonValue::as_str)
        .map(str::to_owned)
}

/// Builds one normalized text/reasoning content block.
fn text_block(message_index: u64, content_index: u64, kind: &str, text: &str) -> ContentBlock {
    ContentBlock {
        block_id: format!("claude-block:{message_index}:{content_index}"),
        kind: kind.to_owned(),
        content: JsonValue::String(text.to_owned()),
        data: JsonObject::new(),
    }
}

/// Builds one output event of `channel`.
fn output_event(
    message_index: u64,
    content_index: u64,
    channel: OutputChannel,
    content: JsonValue,
    operation: OutputOperation,
) -> OutputEvent {
    OutputEvent {
        output_id: format!(
            "claude-output:{message_index}:{content_index}:{}",
            channel_name(channel)
        ),
        kind: OutputKind::Text,
        content,
        operation,
        channel,
        content_index,
        content_type: None,
        data: JsonObject::new(),
    }
}

/// Returns the wire name of an output channel.
fn channel_name(channel: OutputChannel) -> &'static str {
    match channel {
        OutputChannel::Answer => "answer",
        OutputChannel::Reasoning => "reasoning",
        OutputChannel::System => "system",
    }
}

/// Maps a Claude `usage` object onto the protocol type.
fn usage_from(
    usage: &serde_json::Map<String, JsonValue>,
    model: Option<&str>,
) -> Option<TurnUsage> {
    let input = usage.get("input_tokens").and_then(JsonValue::as_u64);
    let output = usage.get("output_tokens").and_then(JsonValue::as_u64);
    let cached = usage
        .get("cache_read_input_tokens")
        .and_then(JsonValue::as_u64);
    if input.is_none() && output.is_none() && cached.is_none() {
        return None;
    }
    let mut provider_data = JsonObject::new();
    if let Some(model) = model {
        provider_data.insert("model".to_owned(), json!(model));
    }
    Some(TurnUsage {
        input_tokens: input,
        output_tokens: output,
        cached_input_tokens: cached,
        provider_data,
        ..TurnUsage::default()
    })
}

/// Builds an allow answer, optionally with rewritten arguments.
fn allow_response(updated_input: Option<JsonValue>) -> JsonValue {
    let mut response = json!({"behavior": "allow"});
    if let Some(updated) = updated_input {
        response["updatedInput"] = updated;
    }
    response
}

/// Builds a deny answer.
fn deny_response(message: &str, interrupt: bool) -> JsonValue {
    json!({"behavior": "deny", "message": message, "interrupt": interrupt})
}

/// Builds the provider data of one interaction request.
fn interaction_data(tool_use_id: &str, tool_name: &str) -> JsonObject {
    let mut data = JsonObject::new();
    data.insert("tool_use_id".to_owned(), json!(tool_use_id));
    data.insert("tool_name".to_owned(), json!(tool_name));
    data
}

/// Renders the prompt of an `AskUserQuestion` input.
fn render_question(input: &JsonValue) -> String {
    let Some(questions) = input.get("questions").and_then(JsonValue::as_array) else {
        return match input {
            JsonValue::String(text) => text.clone(),
            other => compact_json(other),
        };
    };
    let texts: Vec<&str> = questions
        .iter()
        .filter_map(|question| question.get("question").and_then(JsonValue::as_str))
        .collect();
    if texts.is_empty() {
        compact_json(input)
    } else {
        texts.join("\n")
    }
}

/// Returns the first question's option labels as user-input choices.
fn question_choices(input: &JsonValue) -> Vec<String> {
    input
        .get("questions")
        .and_then(JsonValue::as_array)
        .and_then(|questions| questions.first())
        .and_then(|question| question.get("options"))
        .and_then(JsonValue::as_array)
        .map(|options| {
            options
                .iter()
                .filter_map(|option| option.get("label").and_then(JsonValue::as_str))
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

/// Builds the normalized failure of a terminal `result` frame.
fn result_error(frame: &Frame, subtype: &str, stderr_tail: &str) -> TurnError {
    let status = frame.get("api_error_status").and_then(JsonValue::as_u64);
    let category = failure_category(subtype, status);
    let retryable = matches!(
        category,
        "rate_limited" | "server_unavailable" | "network_timeout"
    );
    let message = frame
        .get("result")
        .and_then(JsonValue::as_str)
        .filter(|text| !text.trim().is_empty())
        .map(str::to_owned)
        .or_else(|| errors_text(frame))
        .unwrap_or_else(|| match status {
            Some(status) => format!("Claude turn failed: HTTP {status}"),
            None => format!("Claude turn failed ({subtype})"),
        });
    turn_error(
        with_tail(message, stderr_tail),
        subtype,
        category,
        retryable,
    )
}

/// Joins the informative `errors` strings of a result frame.
fn errors_text(frame: &Frame) -> Option<String> {
    let errors = frame.get("errors").and_then(JsonValue::as_array)?;
    let informative: Vec<&str> = errors
        .iter()
        .filter_map(JsonValue::as_str)
        .filter(|item| !item.trim().is_empty() && !item.trim().eq_ignore_ascii_case("unknown"))
        .collect();
    if informative.is_empty() {
        None
    } else {
        Some(informative.join("\n"))
    }
}

/// Classifies a failure by HTTP status and subtype, using Python's maps.
fn failure_category(subtype: &str, http_status: Option<u64>) -> &'static str {
    if let Some(status) = http_status {
        let category = match status {
            400 => "request_rejected",
            401 | 403 => "auth_required",
            429 => "rate_limited",
            500 | 529 => "server_unavailable",
            _ => SDK_ERROR_CATEGORY,
        };
        if category != SDK_ERROR_CATEGORY {
            return category;
        }
    }
    match subtype {
        "authentication_failed" => "auth_required",
        "billing_error" => "quota_exceeded",
        "invalid_request" => "request_rejected",
        "rate_limit" => "rate_limited",
        "server_error" => "server_unavailable",
        _ => SDK_ERROR_CATEGORY,
    }
}

/// Appends `tail` to `message` when the CLI produced stderr output.
pub(super) fn with_tail(message: String, tail: &str) -> String {
    if tail.is_empty() {
        message
    } else {
        format!("{message}\n{tail}")
    }
}

/// Renders a JSON value as compact JSON, falling back to the empty string.
pub(super) fn compact_json(value: &JsonValue) -> String {
    serde_json::to_string(value).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    //! Unit tests of the `user`-frame tool-result projection.
    //!
    //! A `tool_result` block carries only the `tool_use_id`; the tool name has to be recovered from
    //! the `tool_use` block the same turn announced earlier. These tests drive the projection
    //! directly over an in-process event bus, so no child process is involved.
    // Tests assert by panicking on failure, which is the intent of a test.
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::Duration;

    use openjiuwen_harness_protocol::{
        HarnessEventPayload, ItemEventKind, ItemLifecycleEvent, JsonValue, MessageRole,
    };
    use serde_json::json;

    use crate::vendor_cli::{EnvelopeBuilder, EventBus, TurnContext};

    use super::super::ClaudeCodeHarnessConfig;
    use super::{ClaudeDriver, Frame, TurnAccumulator};

    /// Bounded await budget for every cursor read.
    const BOUND: Duration = Duration::from_secs(10);

    /// Turn id every projected frame is reported under.
    const TURN_ID: &str = "turn-1";

    /// Builds a driver over the default configuration; no binary is resolved or spawned.
    fn driver() -> ClaudeDriver {
        ClaudeDriver::new(
            ClaudeCodeHarnessConfig::default(),
            PathBuf::from("/nonexistent/claude"),
        )
    }

    /// Builds the turn context of one turn and the bus it publishes to.
    fn turn() -> (Arc<EventBus>, TurnContext) {
        let bus = EventBus::new(32);
        let envelope = Arc::new(EnvelopeBuilder::new("host-session", "agent-a"));
        (bus.clone(), TurnContext::new(bus, envelope))
    }

    /// Builds a decoded frame from a JSON literal.
    fn frame(value: &JsonValue) -> Frame {
        value
            .as_object()
            .cloned()
            .expect("the test frame is a JSON object")
    }

    /// Closes the bus and returns every completed item lifecycle event it retained.
    async fn completed_items(bus: &Arc<EventBus>) -> Vec<ItemLifecycleEvent> {
        bus.close();
        let mut cursor = bus.events().expect("no other consumer");
        let mut completed = Vec::new();
        while let Some(event) = tokio::time::timeout(BOUND, cursor.next_event())
            .await
            .expect("the cursor must not block")
            .expect("the cursor must not fail")
        {
            if let HarnessEventPayload::ItemLifecycle(item) = event.event {
                if item.kind == ItemEventKind::Completed {
                    completed.push(item);
                }
            }
        }
        completed
    }

    /// The name announced by the preceding `tool_use` block reaches the completed tool item.
    #[tokio::test]
    async fn a_tool_result_reports_the_name_announced_by_the_tool_use_block() {
        let (bus, session) = turn();
        let driver = driver();
        let mut accumulator = TurnAccumulator::default();
        let assistant = frame(&json!({
            "type": "assistant",
            "message": {"content": [{
                "type": "tool_use",
                "id": "tool-1",
                "name": "Bash",
                "input": {"command": "ls"},
            }]},
        }));
        driver
            .handle_assistant(&assistant, TURN_ID, &session, &mut accumulator)
            .await;
        let user = frame(&json!({
            "type": "user",
            "message": {"content": [{
                "type": "tool_result",
                "tool_use_id": "tool-1",
                "content": "listing",
            }]},
        }));
        driver
            .handle_user(&user, TURN_ID, &session, &mut accumulator)
            .await;

        let completed = completed_items(&bus).await;
        assert_eq!(completed.len(), 1, "one completed tool item");
        assert_eq!(completed[0].data["tool_name"], json!("Bash"));
        assert_eq!(completed[0].data["tool_call_id"], json!("tool-1"));
        assert_eq!(completed[0].data["result"], json!("listing"));

        let message = accumulator.messages.last().expect("the tool message");
        assert_eq!(message.role, MessageRole::Tool);
        assert_eq!(message.content[0].kind, "tool_result");
        assert_eq!(message.content[0].data["call_id"], json!("tool-1"));
    }

    /// A result for a call the turn never announced falls back to the call id as the name.
    #[tokio::test]
    async fn an_unannounced_tool_result_falls_back_to_the_call_id() {
        let (bus, session) = turn();
        let driver = driver();
        let mut accumulator = TurnAccumulator::default();
        let user = frame(&json!({
            "type": "user",
            "message": {"content": [{
                "type": "tool_result",
                "tool_use_id": "tool-9",
                "content": {"output": "resumed"},
            }]},
        }));
        driver
            .handle_user(&user, TURN_ID, &session, &mut accumulator)
            .await;

        let completed = completed_items(&bus).await;
        assert_eq!(completed.len(), 1, "one completed tool item");
        assert_eq!(completed[0].data["tool_name"], json!("tool-9"));
        assert_eq!(completed[0].data["tool_call_id"], json!("tool-9"));
        assert_eq!(completed[0].data["result"], json!({"output": "resumed"}));
    }
}
