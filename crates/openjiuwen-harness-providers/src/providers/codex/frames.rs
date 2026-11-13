//! Notification and server-request projection of the Codex app-server transport.
//!
//! Every method here turns one decoded JSON-RPC message into harness observations (outputs, tool
//! calls, tool results, usage, diagnostics, provider events, host prompts) or into the turn's
//! terminal result. The turn loop, process ownership and lifecycle live in `harness`; the argv and
//! thread options live in `wire`.
//!
//! # Message shapes
//!
//! A notification is `{"method": "<name>", "params": {…}}`; a server-to-client request is the same
//! message plus an `id`. The app-server spells its payload members in camelCase (`itemId`,
//! `tokenUsage`, `willRetry`), while the vendored data types use `snake_case`; both spellings are
//! accepted so a renamed field never drops an observation.

use std::collections::BTreeMap;
use std::sync::Arc;

use openjiuwen_harness_protocol::{
    json_object_from_value, json_object_to_value, ContentBlock, DiagnosticLevel, HarnessResult,
    JsonObject, JsonValue, MessageRole, OutputChannel, OutputEvent, OutputKind, OutputOperation,
    TurnError, TurnMessage, TurnResult, TurnTerminationKind, TurnUsage, UsageUpdateMode,
    UsageUpdatedEvent,
};
use serde_json::json;

use crate::vendor_cli::{
    ask_tool_approval, ask_user_input, completed_turn, failed_turn, interrupted_turn, turn_error,
    ApprovalDecision, JsonRpcClient, TurnContext,
};

use super::harness::{lock, PROVIDER_EVENT_NAME, SDK_ERROR_CATEGORY};
use super::CodexDriver;

/// Normalized failure code of a stream that ended without a terminal `turn/completed`.
pub(super) const MISSING_TURN_COMPLETED_CODE: &str = "CODEX_MISSING_TURN_COMPLETED";

/// Normalized failure code of a turn the app-server failed without an error payload.
pub(super) const TURN_FAILED_CODE: &str = "CODEX_TURN_FAILED";

/// Message of a turn the host interrupted.
pub(super) const INTERRUPTED_MESSAGE: &str = "Codex interrupted the turn";

/// Item types the transport projects as tool calls.
const TOOL_ITEM_TYPES: [&str; 4] = [
    "commandExecution",
    "dynamicToolCall",
    "fileChange",
    "mcpToolCall",
];

/// Reasoning notification methods, projected as reasoning deltas.
const REASONING_METHODS: [&str; 2] = [
    "item/reasoning/summaryTextDelta",
    "item/reasoning/textDelta",
];

/// Raw model-response notifications: large, and only useful to an observability bridge.
const SILENT_METHODS: [&str; 2] = ["rawResponseItem/completed", "rawResponse/completed"];

/// Server-request methods that ask the host to approve one tool call.
const APPROVAL_METHODS: [&str; 2] = [
    "item/commandExecution/requestApproval",
    "item/fileChange/requestApproval",
];

/// Server-request method of Codex's experimental `request_user_input` tool.
const USER_INPUT_METHOD: &str = "item/tool/requestUserInput";

/// Server-request method offering a dynamically registered host tool.
const DYNAMIC_TOOL_CALL_METHOD: &str = "item/tool/call";

/// Outcome of one turn-loop iteration.
pub(super) enum FrameAction {
    /// Keep reading messages.
    Continue,
    /// The turn reached its terminal result; boxed because it dwarfs [`FrameAction::Continue`].
    Terminal(Box<TurnResult>),
}

/// One decoded JSON-RPC message.
pub(super) type Frame = JsonObject;

/// Per-turn accumulator: message history, last text and terminal state.
#[derive(Default)]
pub(super) struct TurnAccumulator {
    /// Normalized message history of the turn.
    messages: Vec<TurnMessage>,
    /// Last completed assistant text, the turn's convenience `final_output`.
    last_text: String,
    /// Most specific failure seen during the turn, used by a generic terminal status.
    pending_error: Option<TurnError>,
    /// Usage accumulated over the turn's model calls.
    usage: Option<TurnUsage>,
    /// Tool call names by item id, so a completion repeats the started name.
    tool_names: BTreeMap<String, String>,
}

impl TurnAccumulator {
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
        add_counter(&mut current.total_tokens, usage.total_tokens);
    }

    /// Stores `error`, keeping an earlier more specific failure over a generic one.
    fn record_error(&mut self, error: TurnError) {
        self.pending_error = Some(match self.pending_error.take() {
            Some(previous) if is_generic(&error) && !is_generic(&previous) => previous,
            _ => error,
        });
    }

    /// Takes the turn's message history.
    fn take_messages(&mut self) -> Vec<TurnMessage> {
        std::mem::take(&mut self.messages)
    }
}

impl CodexDriver {
    /// Maps one decoded message onto observations, or returns the terminal result.
    pub(super) async fn dispatch(
        &self,
        frame: &Frame,
        turn_id: &str,
        session: &TurnContext,
        accumulator: &mut TurnAccumulator,
        client: &Arc<JsonRpcClient>,
    ) -> HarnessResult<FrameAction> {
        let method = get_str(frame, "method").unwrap_or("unknown").to_owned();
        let params = object_member(frame, "params").unwrap_or_default();
        if let Some(id) = frame.get("id").filter(|value| !value.is_null()) {
            self.handle_server_request(&method, id.clone(), &params, turn_id, session, client)
                .await?;
            return Ok(FrameAction::Continue);
        }
        match method.as_str() {
            "item/agentMessage/delta" => {
                self.delta(&params, OutputChannel::Answer, turn_id, session)
                    .await;
            }
            other if REASONING_METHODS.contains(&other) => {
                self.delta(&params, OutputChannel::Reasoning, turn_id, session)
                    .await;
            }
            "item/started" => {
                self.item_started(&params, turn_id, session, accumulator)
                    .await;
            }
            "item/completed" => {
                self.item_completed(&params, turn_id, session, accumulator)
                    .await;
            }
            "thread/tokenUsage/updated" => {
                self.token_usage(&params, turn_id, session, accumulator)
                    .await;
            }
            "error" => self.error(&params, turn_id, session, accumulator).await,
            "turn/completed" => {
                return Ok(FrameAction::Terminal(Box::new(
                    self.terminal(&params, accumulator),
                )))
            }
            other if SILENT_METHODS.contains(&other) => {}
            other => {
                session
                    .provider_event(
                        turn_id,
                        PROVIDER_EVENT_NAME,
                        if other.is_empty() { "unknown" } else { other },
                        json_object_to_value(&params),
                    )
                    .await;
            }
        }
        Ok(FrameAction::Continue)
    }

    /// Projects one text or reasoning delta.
    async fn delta(
        &self,
        params: &Frame,
        channel: OutputChannel,
        turn_id: &str,
        session: &TurnContext,
    ) {
        let Some(delta) = get_str(params, "delta").filter(|delta| !delta.is_empty()) else {
            return;
        };
        let item_id = get_str(params, "itemId")
            .or_else(|| get_str(params, "item_id"))
            .unwrap_or("unknown-item");
        let content = match channel {
            OutputChannel::Reasoning => json!({"thinking": delta}),
            OutputChannel::Answer | OutputChannel::System => json!({"text": delta}),
        };
        session
            .output(
                turn_id,
                OutputEvent {
                    output_id: format!("codex-output:{item_id}:{}", channel_name(channel)),
                    kind: OutputKind::Text,
                    content,
                    operation: OutputOperation::Delta,
                    channel,
                    content_index: 0,
                    content_type: None,
                    data: JsonObject::new(),
                },
            )
            .await;
    }

    /// Projects an `item/started` notification, which only tool items turn into observations.
    async fn item_started(
        &self,
        params: &Frame,
        turn_id: &str,
        session: &TurnContext,
        accumulator: &mut TurnAccumulator,
    ) {
        let Some(item) = object_member(params, "item") else {
            return;
        };
        let item_type = item_type(&item);
        if !TOOL_ITEM_TYPES.contains(&item_type.as_str()) {
            return;
        }
        let item_id = item_id(&item);
        let tool_name = tool_name(&item_type, &item);
        accumulator
            .tool_names
            .insert(item_id.clone(), tool_name.clone());
        session
            .tool_started(
                turn_id,
                &item_id,
                Some(&item_id),
                &tool_name,
                tool_args(&item_type, &item),
            )
            .await;
    }

    /// Projects an `item/completed` notification, reporting an unknown item as a provider event.
    async fn item_completed(
        &self,
        params: &Frame,
        turn_id: &str,
        session: &TurnContext,
        accumulator: &mut TurnAccumulator,
    ) {
        let Some(item) = object_member(params, "item") else {
            return;
        };
        let item_type = item_type(&item);
        let id = item_id(&item);
        if TOOL_ITEM_TYPES.contains(&item_type.as_str()) {
            let name = accumulator
                .tool_names
                .get(&id)
                .cloned()
                .unwrap_or_else(|| tool_name(&item_type, &item));
            let result = tool_result(&item_type, &item);
            session
                .tool_completed(turn_id, &id, Some(&id), &name, result.clone(), None)
                .await;
            let mut data = JsonObject::new();
            data.insert("call_id".to_owned(), json!(id));
            accumulator.messages.push(TurnMessage {
                message_id: format!("codex-tool:{id}"),
                role: MessageRole::Tool,
                content: vec![ContentBlock {
                    block_id: format!("codex-block:{id}"),
                    kind: "tool_result".to_owned(),
                    content: result,
                    data,
                }],
                data: JsonObject::new(),
            });
            return;
        }
        match item_type.as_str() {
            "agentMessage" => {
                let text = get_str(&item, "text").unwrap_or_default().to_owned();
                accumulator.last_text.clone_from(&text);
                if text.is_empty() {
                    return;
                }
                accumulator
                    .messages
                    .push(message(&id, "text", &text, MessageRole::Assistant));
                self.final_output(&id, OutputChannel::Answer, &text, turn_id, session)
                    .await;
            }
            "reasoning" => {
                let summary = summary_text(&item);
                if summary.is_empty() {
                    return;
                }
                accumulator.messages.push(message(
                    &id,
                    "reasoning",
                    &summary,
                    MessageRole::Assistant,
                ));
                self.final_output(&id, OutputChannel::Reasoning, &summary, turn_id, session)
                    .await;
            }
            _ => {
                session
                    .provider_event(
                        turn_id,
                        PROVIDER_EVENT_NAME,
                        "item/completed",
                        json_object_to_value(&item),
                    )
                    .await;
            }
        }
    }

    /// Publishes one completed text/reasoning block.
    async fn final_output(
        &self,
        item_id: &str,
        channel: OutputChannel,
        text: &str,
        turn_id: &str,
        session: &TurnContext,
    ) {
        let content = match channel {
            OutputChannel::Reasoning => json!({"thinking": text}),
            OutputChannel::Answer | OutputChannel::System => json!({"text": text}),
        };
        session
            .output(
                turn_id,
                OutputEvent {
                    output_id: format!("codex-output:{item_id}:{}", channel_name(channel)),
                    kind: OutputKind::Text,
                    content,
                    operation: OutputOperation::Final,
                    channel,
                    content_index: 0,
                    content_type: None,
                    data: JsonObject::new(),
                },
            )
            .await;
    }

    /// Publishes one delta usage update and accumulates it for the terminal result.
    async fn token_usage(
        &self,
        params: &Frame,
        turn_id: &str,
        session: &TurnContext,
        accumulator: &mut TurnAccumulator,
    ) {
        let Some(token_usage) =
            object_member(params, "tokenUsage").or_else(|| object_member(params, "token_usage"))
        else {
            return;
        };
        let last = object_member(&token_usage, "last");
        let source = last.as_ref().unwrap_or(&token_usage);
        let usage = TurnUsage {
            input_tokens: counter(source, "input_tokens", "inputTokens"),
            output_tokens: counter(source, "output_tokens", "outputTokens"),
            cached_input_tokens: counter(source, "cached_input_tokens", "cachedInputTokens"),
            reasoning_output_tokens: counter(
                source,
                "reasoning_output_tokens",
                "reasoningOutputTokens",
            ),
            total_tokens: counter(source, "total_tokens", "totalTokens"),
            provider_data: JsonObject::new(),
        };
        if usage.input_tokens.is_none()
            && usage.output_tokens.is_none()
            && usage.total_tokens.is_none()
            && usage.cached_input_tokens.is_none()
            && usage.reasoning_output_tokens.is_none()
        {
            return;
        }
        accumulator.merge_usage(&usage);
        session
            .emit(
                UsageUpdatedEvent {
                    usage,
                    mode: UsageUpdateMode::Delta,
                },
                Some(turn_id),
            )
            .await;
    }

    /// Records a Codex error notification and publishes its diagnostic.
    async fn error(
        &self,
        params: &Frame,
        turn_id: &str,
        session: &TurnContext,
        accumulator: &mut TurnAccumulator,
    ) {
        let (error, will_retry) = classify_notification(params);
        let category = error.category.clone().unwrap_or_default();
        let mut data = JsonObject::new();
        data.insert(
            "kind".to_owned(),
            json!(if will_retry { "retrying" } else { "reported" }),
        );
        data.insert("category".to_owned(), json!(category));
        if let Some(code) = error.code.as_deref() {
            data.insert("code".to_owned(), json!(code));
        }
        session
            .diagnostic(
                turn_id,
                if will_retry {
                    DiagnosticLevel::Warning
                } else {
                    DiagnosticLevel::Error
                },
                &if will_retry {
                    format!("Codex is retrying after {category}")
                } else {
                    format!("Codex reported {category}")
                },
                data,
            )
            .await;
        accumulator.record_error(error);
    }

    /// Folds the terminal `turn/completed` notification into the turn result.
    fn terminal(&self, params: &Frame, accumulator: &mut TurnAccumulator) -> TurnResult {
        let status = object_member(params, "turn")
            .and_then(|turn| get_str(&turn, "status").map(str::to_owned))
            .unwrap_or_default();
        let host_abort = self.take_host_abort();
        let messages = accumulator.take_messages();
        let usage = accumulator.usage.clone();
        let pending = accumulator.pending_error.clone();
        if host_abort {
            return interrupted_turn(TurnTerminationKind::UserAbort, INTERRUPTED_MESSAGE);
        }
        match status.as_str() {
            "completed" | "complete" => completed_turn(
                JsonValue::String(accumulator.last_text.clone()),
                messages,
                usage,
            ),
            "interrupted" => interrupted_turn(TurnTerminationKind::Provider, INTERRUPTED_MESSAGE),
            "failed" | "failure" => failed_turn(pending.unwrap_or_else(|| {
                turn_error(
                    "Codex turn failed",
                    TURN_FAILED_CODE,
                    SDK_ERROR_CATEGORY,
                    false,
                )
            })),
            _ => failed_turn(merge(
                pending,
                turn_error(
                    "Codex ended the turn stream without a terminal status",
                    MISSING_TURN_COMPLETED_CODE,
                    SDK_ERROR_CATEGORY,
                    false,
                ),
            )),
        }
    }

    /// Answers one server-to-client request.
    async fn handle_server_request(
        &self,
        method: &str,
        id: JsonValue,
        params: &Frame,
        turn_id: &str,
        session: &TurnContext,
        client: &Arc<JsonRpcClient>,
    ) -> HarnessResult<()> {
        if APPROVAL_METHODS.contains(&method) {
            return self.answer_approval(method, id, params, client).await;
        }
        match method {
            USER_INPUT_METHOD => self.answer_user_input(id, params, client).await,
            DYNAMIC_TOOL_CALL_METHOD => {
                client
                    .respond_error(id, -32601, "dynamic tools are not available")
                    .await
            }
            other => {
                session
                    .provider_event(
                        turn_id,
                        PROVIDER_EVENT_NAME,
                        "server_request",
                        json!({"method": other}),
                    )
                    .await;
                client
                    .respond_error(
                        id,
                        -32601,
                        &format!("unsupported Codex server request '{other}'"),
                    )
                    .await
            }
        }
    }

    /// Asks the host about one tool call and answers the approval request.
    async fn answer_approval(
        &self,
        method: &str,
        id: JsonValue,
        params: &Frame,
        client: &Arc<JsonRpcClient>,
    ) -> HarnessResult<()> {
        let item_id = get_str(params, "itemId")
            .or_else(|| get_str(params, "item_id"))
            .unwrap_or("codex-approval")
            .to_owned();
        let tool_name = if method == "item/fileChange/requestApproval" {
            "apply_patch"
        } else {
            "shell"
        };
        let mut arguments = params.clone();
        arguments.remove("threadId");
        arguments.remove("turnId");
        let (policy, handler) = {
            let state = lock(&self.state);
            (state.policy, state.handler.clone())
        };
        let mut provider_data = JsonObject::new();
        provider_data.insert("method".to_owned(), json!(method));
        let decision = ask_tool_approval(
            handler.as_ref(),
            &policy,
            format!("codex-approval:{item_id}"),
            Some(item_id),
            tool_name.to_owned(),
            json_object_to_value(&arguments),
            provider_data,
        )
        .await;
        let answer = match decision {
            ApprovalDecision::Allow { .. } => "accept",
            ApprovalDecision::Deny { .. } => "decline",
        };
        client.respond(id, json!({ "decision": answer })).await
    }

    /// Asks the host the Codex `request_user_input` questions and answers them.
    async fn answer_user_input(
        &self,
        id: JsonValue,
        params: &Frame,
        client: &Arc<JsonRpcClient>,
    ) -> HarnessResult<()> {
        let questions = question_list(params);
        let item_id = get_str(params, "itemId")
            .or_else(|| get_str(params, "item_id"))
            .unwrap_or("codex-user-input")
            .to_owned();
        let (policy, handler) = {
            let state = lock(&self.state);
            (state.policy, state.handler.clone())
        };
        let mut provider_data = JsonObject::new();
        provider_data.insert("tool_name".to_owned(), json!("request_user_input"));
        provider_data.insert("call_id".to_owned(), json!(item_id));
        provider_data.insert("questions".to_owned(), JsonValue::Array(questions.clone()));
        let response = ask_user_input(
            handler.as_ref(),
            &policy,
            format!("codex-ask:{item_id}"),
            render_questions(&questions),
            first_choices(&questions),
            provider_data,
        )
        .await;
        client
            .respond(id, answers_wire(response.as_ref(), &questions))
            .await
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

/// Returns the string member `key` of `object`.
fn get_str<'a>(object: &'a Frame, key: &str) -> Option<&'a str> {
    object.get(key).and_then(JsonValue::as_str)
}

/// Returns the object member `key` of `object`, decoded into a frame of its own.
fn object_member(object: &Frame, key: &str) -> Option<Frame> {
    object
        .get(key)
        .and_then(|value| json_object_from_value(value.clone()))
}

/// Returns a non-negative integer counter under either spelling.
fn counter(object: &Frame, snake: &str, camel: &str) -> Option<u64> {
    object
        .get(snake)
        .or_else(|| object.get(camel))
        .and_then(JsonValue::as_u64)
}

/// Adds `value` to `target` when both are present.
fn add_counter(target: &mut Option<u64>, value: Option<u64>) {
    if let Some(value) = value {
        *target = Some(target.map_or(value, |current| current.saturating_add(value)));
    }
}

/// Returns the `type` of one thread item.
fn item_type(item: &Frame) -> String {
    get_str(item, "type").unwrap_or("unknown").to_owned()
}

/// Returns the id of one thread item, or a stable placeholder.
fn item_id(item: &Frame) -> String {
    get_str(item, "id").unwrap_or("unknown-item").to_owned()
}

/// Returns the tool name the transport reports for one tool item.
fn tool_name(item_type: &str, item: &Frame) -> String {
    match item_type {
        "commandExecution" => "shell".to_owned(),
        "fileChange" => "apply_patch".to_owned(),
        "dynamicToolCall" => get_str(item, "tool").unwrap_or_default().to_owned(),
        "mcpToolCall" => {
            let server = get_str(item, "server").unwrap_or_default();
            let tool = get_str(item, "tool").unwrap_or_default();
            format!("{server}.{tool}").trim_matches('.').to_owned()
        }
        other => other.to_owned(),
    }
}

/// Returns the arguments of one tool item in the transport's shape.
fn tool_args(item_type: &str, item: &Frame) -> JsonValue {
    match item_type {
        "commandExecution" => json!({
            "command": item.get("command").cloned().unwrap_or(JsonValue::Null),
            "cwd": item.get("cwd").cloned().unwrap_or(JsonValue::Null),
        }),
        "fileChange" => json!({"changes": item.get("changes").cloned().unwrap_or(JsonValue::Null)}),
        _ => item.get("arguments").cloned().unwrap_or(JsonValue::Null),
    }
}

/// Returns the result of one completed tool item.
fn tool_result(item_type: &str, item: &Frame) -> JsonValue {
    match item_type {
        "commandExecution" => {
            let mut result = JsonObject::new();
            result.insert(
                "output".to_owned(),
                item.get("aggregatedOutput")
                    .or_else(|| item.get("aggregated_output"))
                    .cloned()
                    .unwrap_or(JsonValue::Null),
            );
            result.insert(
                "exit_code".to_owned(),
                item.get("exitCode")
                    .or_else(|| item.get("exit_code"))
                    .cloned()
                    .unwrap_or(JsonValue::Null),
            );
            json_object_to_value(&result)
        }
        "fileChange" => json!({"changes": item.get("changes").cloned().unwrap_or(JsonValue::Null)}),
        _ => item
            .get("result")
            .cloned()
            .or_else(|| item.get("output").cloned())
            .unwrap_or(JsonValue::Null),
    }
}

/// Joins the reasoning summary parts of one item.
fn summary_text(item: &Frame) -> String {
    let Some(JsonValue::Array(parts)) = item.get("summary") else {
        return String::new();
    };
    parts
        .iter()
        .filter_map(JsonValue::as_str)
        .filter(|part| !part.is_empty())
        .collect::<Vec<&str>>()
        .join("\n")
}

/// Builds one normalized assistant content block.
fn message(item_id: &str, kind: &str, text: &str, role: MessageRole) -> TurnMessage {
    TurnMessage {
        message_id: format!("codex-message:{item_id}"),
        role,
        content: vec![ContentBlock {
            block_id: format!("codex-block:{item_id}"),
            kind: kind.to_owned(),
            content: JsonValue::String(text.to_owned()),
            data: JsonObject::new(),
        }],
        data: JsonObject::new(),
    }
}

/// Returns whether `error` says nothing more than "the SDK failed".
fn is_generic(error: &TurnError) -> bool {
    error
        .category
        .as_deref()
        .is_none_or(|category| category == SDK_ERROR_CATEGORY)
}

/// Prefers a specific `pending` failure over a generic `terminal` one, like Python's
/// `merge_pending_error`.
fn merge(pending: Option<TurnError>, terminal: TurnError) -> TurnError {
    match pending {
        Some(pending) if is_generic(&terminal) && !is_generic(&pending) => pending,
        _ => terminal,
    }
}

/// Classifies an `error` notification into a normalized failure and whether Codex retries it.
fn classify_notification(params: &Frame) -> (TurnError, bool) {
    let will_retry = params
        .get("willRetry")
        .or_else(|| params.get("will_retry"))
        .and_then(JsonValue::as_bool)
        .unwrap_or(false);
    let error = object_member(params, "error");
    let error = error.as_ref();
    let message = error
        .and_then(|error| get_str(error, "message"))
        .filter(|message| !message.trim().is_empty())
        .map_or_else(|| "Codex reported an error".to_owned(), str::to_owned);
    let message = with_additional_details(message, error);
    let info = error.and_then(|error| {
        error
            .get("codexErrorInfo")
            .or_else(|| error.get("codex_error_info"))
            .or_else(|| {
                error.get("data").and_then(|data| {
                    data.get("codex_error_info")
                        .or_else(|| data.get("codexErrorInfo"))
                        .or_else(|| data.get("errorInfo"))
                })
            })
    });
    let http_status = error.and_then(error_http_status);
    let (category, code) = classify_info(info, http_status);
    let retryable = matches!(
        category.as_str(),
        "rate_limited" | "server_unavailable" | "network_timeout"
    );
    (turn_error(message, code, &category, retryable), will_retry)
}

/// Appends Codex's `additionalDetails` when it adds information.
fn with_additional_details(message: String, error: Option<&Frame>) -> String {
    let details = error
        .and_then(|error| {
            error
                .get("additionalDetails")
                .or_else(|| error.get("additional_details"))
        })
        .and_then(JsonValue::as_str)
        .map(str::trim)
        .unwrap_or_default();
    if details.is_empty() || message.contains(details) {
        message
    } else if message.is_empty() {
        details.to_owned()
    } else {
        format!("{message}\n{details}")
    }
}

/// Reads an HTTP status out of an error object.
fn error_http_status(error: &Frame) -> Option<u64> {
    let direct = error
        .get("httpStatusCode")
        .or_else(|| error.get("http_status_code"));
    if let Some(status) = direct {
        return status.as_u64();
    }
    match error
        .get("codexErrorInfo")
        .or_else(|| error.get("codex_error_info"))?
    {
        JsonValue::Object(info) => info.values().find_map(|value| {
            value
                .get("httpStatusCode")
                .or_else(|| value.get("http_status_code"))
                .and_then(JsonValue::as_u64)
        }),
        _ => None,
    }
}

/// Classifies a `CodexErrorInfo` value plus its HTTP status.
///
/// Returns the shared failure category and the normalized `codex_error_info` identifier.
fn classify_info(info: Option<&JsonValue>, http_status: Option<u64>) -> (String, String) {
    let (identifier, mut category) = match info {
        Some(JsonValue::String(identifier)) => {
            (identifier.clone(), info_category(identifier).to_owned())
        }
        Some(JsonValue::Object(variant)) => {
            let identifier = variant
                .keys()
                .next()
                .cloned()
                .unwrap_or_else(|| "other".to_owned());
            let category = variant_category(&identifier)
                .unwrap_or_else(|| info_category(&identifier))
                .to_owned();
            (identifier, category)
        }
        _ => (String::new(), SDK_ERROR_CATEGORY.to_owned()),
    };
    if let Some(status) = http_status {
        if let Some(mapped) = http_category(status) {
            mapped.clone_into(&mut category);
        }
    }
    let code = if identifier.is_empty() {
        http_status
            .map(|status| status.to_string())
            .unwrap_or_default()
    } else {
        identifier
    };
    (category, code)
}

/// Maps a `CodexErrorInfoValue` identifier onto a shared failure category.
fn info_category(identifier: &str) -> &'static str {
    match identifier {
        "unauthorized" => "auth_required",
        "usageLimitExceeded" | "sessionBudgetExceeded" => "quota_exceeded",
        "serverOverloaded" | "internalServerError" => "server_unavailable",
        "badRequest" => "request_rejected",
        _ => SDK_ERROR_CATEGORY,
    }
}

/// Maps a transport-level `CodexErrorInfo` variant onto a shared failure category.
fn variant_category(identifier: &str) -> Option<&'static str> {
    match identifier {
        "httpConnectionFailed"
        | "responseStreamConnectionFailed"
        | "responseStreamDisconnected"
        | "responseTooManyFailedAttempts" => Some("network_timeout"),
        "activeTurnNotSteerable" => Some(SDK_ERROR_CATEGORY),
        _ => None,
    }
}

/// Maps an HTTP status onto a shared failure category.
fn http_category(status: u64) -> Option<&'static str> {
    match status {
        400 => Some("request_rejected"),
        401 | 403 => Some("auth_required"),
        429 => Some("rate_limited"),
        500 | 529 => Some("server_unavailable"),
        _ => None,
    }
}

/// Returns the `request_user_input` questions as plain objects with ids.
fn question_list(params: &Frame) -> Vec<JsonValue> {
    let Some(JsonValue::Array(raw)) = params.get("questions") else {
        return Vec::new();
    };
    raw.iter()
        .enumerate()
        .filter_map(|(index, question)| {
            let mut question = question.as_object()?.clone();
            question
                .entry("id".to_owned())
                .or_insert_with(|| JsonValue::String(format!("q{index}")));
            Some(JsonValue::Object(question))
        })
        .collect()
}

/// Renders the questions as one prompt, listing the options by label.
fn render_questions(questions: &[JsonValue]) -> String {
    if questions.is_empty() {
        return "Input requested".to_owned();
    }
    let mut lines = Vec::new();
    for question in questions {
        let header = question
            .get("header")
            .and_then(JsonValue::as_str)
            .unwrap_or_default()
            .trim();
        let text = question
            .get("question")
            .and_then(JsonValue::as_str)
            .unwrap_or_default()
            .trim();
        lines.push(match (header.is_empty(), text.is_empty()) {
            (false, false) => format!("{header}: {text}"),
            (false, true) => header.to_owned(),
            (true, false) => text.to_owned(),
            (true, true) => "Input requested".to_owned(),
        });
        if let Some(JsonValue::Array(options)) = question.get("options") {
            for option in options {
                let Some(label) = option.get("label").and_then(JsonValue::as_str) else {
                    continue;
                };
                let description = option
                    .get("description")
                    .and_then(JsonValue::as_str)
                    .unwrap_or_default()
                    .trim();
                if description.is_empty() {
                    lines.push(format!("  - {label}"));
                } else {
                    lines.push(format!("  - {label}: {description}"));
                }
            }
        }
    }
    lines.join("\n")
}

/// Returns the first question's option labels as user-input choices.
fn first_choices(questions: &[JsonValue]) -> Vec<String> {
    questions
        .first()
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

/// Renders a host user-input answer in the app-server's `answers` shape.
fn answers_wire(response: Option<&JsonValue>, questions: &[JsonValue]) -> JsonValue {
    let ids: Vec<String> = questions
        .iter()
        .map(|question| {
            question
                .get("id")
                .and_then(JsonValue::as_str)
                .unwrap_or_default()
                .to_owned()
        })
        .collect();
    let by_text: BTreeMap<String, String> = questions
        .iter()
        .zip(ids.iter())
        .filter_map(|(question, id)| {
            question
                .get("question")
                .and_then(JsonValue::as_str)
                .map(|text| (text.to_owned(), id.clone()))
        })
        .collect();
    let mut answers: BTreeMap<String, Vec<String>> = BTreeMap::new();
    match response {
        None => {}
        Some(JsonValue::Object(object)) => {
            let source = object
                .get("answers")
                .and_then(JsonValue::as_object)
                .unwrap_or(object);
            for (key, value) in source {
                let id = ids
                    .iter()
                    .find(|id| *id == key)
                    .cloned()
                    .or_else(|| by_text.get(key).cloned());
                if let Some(id) = id {
                    answers.insert(id, answer_values(value));
                }
            }
            if answers.is_empty() {
                if let Some(first) = ids.first() {
                    let value = object
                        .get("answer")
                        .cloned()
                        .unwrap_or_else(|| JsonValue::Object(object.clone()));
                    answers.insert(first.clone(), answer_values(&value));
                }
            }
        }
        Some(JsonValue::Array(values)) => {
            for (id, value) in ids.iter().zip(values.iter()) {
                answers.insert(id.clone(), answer_values(value));
            }
        }
        Some(value) => {
            if let Some(first) = ids.first() {
                answers.insert(first.clone(), answer_values(value));
            }
        }
    }
    let wire: serde_json::Map<String, JsonValue> = answers
        .into_iter()
        .map(|(id, values)| (id, json!({"answers": values})))
        .collect();
    json!({ "answers": JsonValue::Object(wire) })
}

/// Flattens one host answer into the app-server's list of strings.
fn answer_values(value: &JsonValue) -> Vec<String> {
    match value {
        JsonValue::Array(values) => values.iter().map(as_text).collect(),
        JsonValue::Object(object) => match object.get("answers") {
            Some(JsonValue::Array(values)) => values.iter().map(as_text).collect(),
            _ => vec![JsonValue::Object(object.clone()).to_string()],
        },
        other => vec![as_text(other)],
    }
}

/// Renders one answer value as text.
fn as_text(value: &JsonValue) -> String {
    match value {
        JsonValue::String(text) => text.clone(),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    // Tests assert by panicking on failure, which is the intent of a test.
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use serde_json::json;

    use super::{
        answers_wire, classify_info, classify_notification, tool_args, tool_name, tool_result,
    };

    /// Decodes one JSON object literal.
    fn object(value: serde_json::Value) -> super::Frame {
        match value {
            serde_json::Value::Object(object) => object.into_iter().collect(),
            _ => panic!("expected an object"),
        }
    }

    /// Decodes the item of an `item/*` payload.
    fn item(value: serde_json::Value) -> super::Frame {
        object(value)
    }

    #[test]
    fn tool_names_and_arguments_follow_the_item_type() {
        let command = item(
            json!({"id": "i1", "type": "commandExecution", "command": "ls",
            "cwd": "/tmp"}),
        );
        assert_eq!(tool_name("commandExecution", &command), "shell");
        assert_eq!(
            tool_args("commandExecution", &command),
            json!({"command": "ls", "cwd": "/tmp"})
        );
        let mcp = item(
            json!({"id": "i2", "type": "mcpToolCall", "server": "srv", "tool": "read",
            "arguments": {"path": "a"}}),
        );
        assert_eq!(tool_name("mcpToolCall", &mcp), "srv.read");
        assert_eq!(tool_args("mcpToolCall", &mcp), json!({"path": "a"}));
        let patch = item(json!({"id": "i3", "type": "fileChange", "changes": [{"path": "a"}]}));
        assert_eq!(tool_name("fileChange", &patch), "apply_patch");
        assert_eq!(
            tool_args("fileChange", &patch),
            json!({"changes": [{"path": "a"}]})
        );
    }

    #[test]
    fn a_command_result_carries_the_output_and_exit_code() {
        let command = item(json!({"id": "i1", "type": "commandExecution",
            "aggregatedOutput": "done", "exitCode": 0}));
        assert_eq!(
            tool_result("commandExecution", &command),
            json!({"output": "done", "exit_code": 0})
        );
    }

    #[test]
    fn an_error_notification_is_classified_by_its_error_info() {
        let params = object(json!({
            "willRetry": true,
            "error": {"message": "rate limited", "codexErrorInfo": "usageLimitExceeded"},
        }));
        let (error, will_retry) = classify_notification(&params);
        assert!(will_retry);
        assert_eq!(error.category.as_deref(), Some("quota_exceeded"));
        assert_eq!(error.code.as_deref(), Some("usageLimitExceeded"));
        assert_eq!(error.retryable, Some(false));
        assert_eq!(error.message, "rate limited");
    }

    #[test]
    fn an_http_status_wins_over_the_error_info() {
        let params = object(json!({
            "error": {"message": "boom", "codexErrorInfo": {"internalServerError": {}}},
        }));
        let (error, will_retry) = classify_notification(&params);
        assert!(!will_retry);
        assert_eq!(error.category.as_deref(), Some("server_unavailable"));
        assert_eq!(error.retryable, Some(true));
        // The variant is unwrapped to its identifier and the message kept.
        assert_eq!(error.code.as_deref(), Some("internalServerError"));
    }

    #[test]
    fn a_connection_variant_is_a_retryable_network_timeout() {
        let (category, code) = classify_info(Some(&json!({"httpConnectionFailed": {}})), Some(503));
        // 503 is not one of Codex's mapped statuses, so the connection variant decides.
        assert_eq!(category, "network_timeout");
        assert_eq!(code, "httpConnectionFailed");
        let params = object(json!({
            "error": {"message": "offline", "codexErrorInfo": {"responseStreamDisconnected": {}}},
        }));
        let (error, _) = classify_notification(&params);
        assert_eq!(error.category.as_deref(), Some("network_timeout"));
        assert_eq!(error.retryable, Some(true));
    }

    #[test]
    fn answers_are_keyed_by_question_id() {
        let questions = vec![
            json!({"id": "q0", "question": "Name?"}),
            json!({"id": "q1", "question": "Colour?"}),
        ];
        let answers = answers_wire(
            Some(&json!({"answers": {"q0": "Ada", "q1": ["red", "blue"]}})),
            &questions,
        );
        assert_eq!(
            answers,
            json!({"answers": {
                "q0": {"answers": ["Ada"]},
                "q1": {"answers": ["red", "blue"]},
            }})
        );
        assert_eq!(
            answers_wire(Some(&json!("only")), &questions),
            json!({"answers": {"q0": {"answers": ["only"]}}})
        );
        assert_eq!(answers_wire(None, &questions), json!({"answers": {}}));
    }

    #[test]
    fn question_texts_are_matched_to_their_id() {
        let questions = vec![json!({"id": "q0", "question": "Name?"})];
        let answers = answers_wire(Some(&json!({"Name?": "Ada"})), &questions);
        assert_eq!(answers, json!({"answers": {"q0": {"answers": ["Ada"]}}}));
    }
}
