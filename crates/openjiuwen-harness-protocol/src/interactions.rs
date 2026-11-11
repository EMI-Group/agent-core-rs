//! Awaited host interactions requested by a third-party agent harness.
//!
//! Mirrors `openjiuwen.harness_protocol.interactions` (v0.1.20): the requests a harness raises
//! mid-turn ([`ToolApprovalRequest`], [`UserInputRequest`], [`McpElicitationRequest`],
//! [`DynamicToolCallRequest`], [`ProviderInteractionRequest`]), the paired responses, the
//! [`HarnessInteractionHandler`] host service, and [`validate_interaction_response`].
//!
//! Python's `typing` unions become the tagged enums [`HarnessInteractionRequest`] and
//! [`HarnessInteractionResponse`] (discriminator `interaction_type`); the five concrete request
//! and response types stay exported individually.
//!
//! Wire shape: `snake_case` keys, every key always present (`null` for [`Option::None`], `{}` for
//! empty objects, `[]` for empty sequences), unknown keys tolerated on decode. Python's
//! `freeze_json_object` has no Rust analogue — owned values and value semantics replace it.
//!
//! Constructors do not validate implicitly; decode paths and callers that need the Python
//! `__post_init__` invariants call [`Validate::validate`].

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::errors::{HarnessError, HarnessResult};
use crate::json::{JsonObject, JsonValue};
use crate::validation::{
    require_finite_timestamp, require_non_empty, require_optional_non_empty, Validate,
};

/// Completion status for an interaction that is not an approval.
///
/// Wire values are `completed`, `declined`, `cancelled`, and `failed`.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum InteractionResponseStatus {
    /// The host produced a usable answer.
    Completed,
    /// The user declined to answer or approve.
    Declined,
    /// The interaction was cancelled before an answer was produced.
    Cancelled,
    /// The interaction failed (host error, timeout propagated as a failure).
    Failed,
}

/// Host decision for a provider-requested tool execution.
///
/// Wire values are `allow`, `allow_for_session`, `deny`, and `abort`.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum ToolApprovalDecision {
    /// Allow this single tool call.
    Allow,
    /// Allow this tool call and every later call of the same tool for the rest of the session.
    AllowForSession,
    /// Deny this tool call; the harness reports the denial to the provider.
    Deny,
    /// Deny this tool call and abort the active turn.
    Abort,
}

/// Reason a pending provider-to-host interaction was cancelled.
///
/// Wire values are `provider_withdrew`, `turn_aborted`, `harness_stopped`, and
/// `deadline_exceeded`.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum InteractionCancelReason {
    /// The provider no longer needs the answer (Python's `cancel` keyword default).
    ProviderWithdrew,
    /// The active turn was aborted.
    TurnAborted,
    /// The harness is stopping.
    HarnessStopped,
    /// `deadline_at` elapsed before the host answered.
    DeadlineExceeded,
}

/// Checks the fields every interaction request shares, in Python's order.
///
/// # Errors
///
/// Returns [`HarnessError::Validation`] with the exact Python message for the first violated
/// invariant: `"interaction request_id must not be empty"`, `"interaction {field} must not be
/// empty"` for an empty `provider_session_id`/`turn_id`, or `"interaction deadline_at must be a
/// finite Unix timestamp"`.
fn validate_request_common(
    request_id: &str,
    provider_session_id: Option<&str>,
    turn_id: Option<&str>,
    deadline_at: Option<f64>,
) -> HarnessResult<()> {
    require_non_empty("interaction request_id", request_id)?;
    require_optional_non_empty("interaction provider_session_id", provider_session_id)?;
    require_optional_non_empty("interaction turn_id", turn_id)?;
    if let Some(deadline_at) = deadline_at {
        require_finite_timestamp("interaction deadline_at", deadline_at)?;
    }
    Ok(())
}

/// Checks the `request_id` every interaction response carries.
///
/// # Errors
///
/// Returns [`HarnessError::Validation`] with `"interaction response request_id must not be empty"`
/// when `request_id` is empty.
fn validate_response_id(request_id: &str) -> HarnessResult<()> {
    require_non_empty("interaction response request_id", request_id)
}

/// Request authorization before a provider executes a tool call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct ToolApprovalRequest {
    /// Harness-chosen identifier; the paired response must reuse it.
    pub request_id: String,
    /// Identifier of the tool call the provider wants to execute.
    pub call_id: String,
    /// Name of the tool the provider wants to execute.
    pub tool_name: String,
    /// Provider JSON arguments of the pending call; defaults to an empty object.
    #[serde(default)]
    pub arguments: JsonObject,
    /// Provider session the request belongs to, when the provider has one.
    #[serde(default)]
    pub provider_session_id: Option<String>,
    /// Turn the request belongs to, when the provider has one.
    #[serde(default)]
    pub turn_id: Option<String>,
    /// Optional host-UI title for the approval prompt.
    #[serde(default)]
    pub title: Option<String>,
    /// Optional host-UI description of the pending call.
    #[serde(default)]
    pub description: Option<String>,
    /// Optional reason the provider gives for requesting approval.
    #[serde(default)]
    pub reason: Option<String>,
    /// Unix-epoch-second deadline; when present it must be finite and non-negative.
    #[serde(default)]
    pub deadline_at: Option<f64>,
    /// Provider extension data; defaults to an empty object.
    #[serde(default)]
    pub provider_data: JsonObject,
}

impl Validate for ToolApprovalRequest {
    fn validate(&self) -> HarnessResult<()> {
        validate_request_common(
            &self.request_id,
            self.provider_session_id.as_deref(),
            self.turn_id.as_deref(),
            self.deadline_at,
        )?;
        if self.call_id.is_empty() || self.tool_name.is_empty() {
            return Err(HarnessError::validation(
                "tool approval call_id and tool_name must not be empty",
            ));
        }
        Ok(())
    }
}

/// Request additional input from the user while a turn is active.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct UserInputRequest {
    /// Harness-chosen identifier; the paired response must reuse it.
    pub request_id: String,
    /// Prompt shown to the user.
    pub prompt: String,
    /// Provider session the request belongs to, when the provider has one.
    #[serde(default)]
    pub provider_session_id: Option<String>,
    /// Turn the request belongs to, when the provider has one.
    #[serde(default)]
    pub turn_id: Option<String>,
    /// Allowed answers when the host should offer a closed choice; defaults to no restriction.
    #[serde(default)]
    pub choices: Vec<String>,
    /// Unix-epoch-second deadline; when present it must be finite and non-negative.
    #[serde(default)]
    pub deadline_at: Option<f64>,
    /// Provider extension data; defaults to an empty object.
    #[serde(default)]
    pub provider_data: JsonObject,
}

impl Validate for UserInputRequest {
    fn validate(&self) -> HarnessResult<()> {
        validate_request_common(
            &self.request_id,
            self.provider_session_id.as_deref(),
            self.turn_id.as_deref(),
            self.deadline_at,
        )?;
        if self.prompt.is_empty() {
            return Err(HarnessError::validation(
                "user input prompt must not be empty",
            ));
        }
        Ok(())
    }
}

/// Request structured user input for an MCP elicitation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct McpElicitationRequest {
    /// Harness-chosen identifier; the paired response must reuse it.
    pub request_id: String,
    /// Name of the MCP server that raised the elicitation.
    pub server_name: String,
    /// Prompt shown to the user.
    pub prompt: String,
    /// Optional JSON Schema describing the requested values.
    #[serde(default)]
    pub schema: Option<JsonObject>,
    /// Provider session the request belongs to, when the provider has one.
    #[serde(default)]
    pub provider_session_id: Option<String>,
    /// Turn the request belongs to, when the provider has one.
    #[serde(default)]
    pub turn_id: Option<String>,
    /// Unix-epoch-second deadline; when present it must be finite and non-negative.
    #[serde(default)]
    pub deadline_at: Option<f64>,
    /// Provider extension data; defaults to an empty object.
    #[serde(default)]
    pub provider_data: JsonObject,
}

impl Validate for McpElicitationRequest {
    fn validate(&self) -> HarnessResult<()> {
        validate_request_common(
            &self.request_id,
            self.provider_session_id.as_deref(),
            self.turn_id.as_deref(),
            self.deadline_at,
        )?;
        if self.server_name.is_empty() || self.prompt.is_empty() {
            return Err(HarnessError::validation(
                "MCP elicitation server_name and prompt must not be empty",
            ));
        }
        Ok(())
    }
}

/// Delegate a provider-originated dynamic tool call to the host.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct DynamicToolCallRequest {
    /// Harness-chosen identifier; the paired response must reuse it.
    pub request_id: String,
    /// Identifier of the dynamic tool call.
    pub call_id: String,
    /// Name of the dynamic tool.
    pub tool_name: String,
    /// Provider JSON arguments of the call; defaults to an empty object.
    #[serde(default)]
    pub arguments: JsonObject,
    /// Optional namespace that scopes the tool name.
    #[serde(default)]
    pub namespace: Option<String>,
    /// Provider session the request belongs to, when the provider has one.
    #[serde(default)]
    pub provider_session_id: Option<String>,
    /// Turn the request belongs to, when the provider has one.
    #[serde(default)]
    pub turn_id: Option<String>,
    /// Unix-epoch-second deadline; when present it must be finite and non-negative.
    #[serde(default)]
    pub deadline_at: Option<f64>,
    /// Provider extension data; defaults to an empty object.
    #[serde(default)]
    pub provider_data: JsonObject,
}

impl Validate for DynamicToolCallRequest {
    fn validate(&self) -> HarnessResult<()> {
        validate_request_common(
            &self.request_id,
            self.provider_session_id.as_deref(),
            self.turn_id.as_deref(),
            self.deadline_at,
        )?;
        if self.call_id.is_empty() || self.tool_name.is_empty() {
            return Err(HarnessError::validation(
                "dynamic tool call_id and tool_name must not be empty",
            ));
        }
        Ok(())
    }
}

/// Provider extension request not covered by a shared interaction type.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct ProviderInteractionRequest {
    /// Harness-chosen identifier; the paired response must reuse it.
    pub request_id: String,
    /// Provider namespace the request belongs to (validated as a non-empty "namespace").
    pub provider: String,
    /// Provider-defined request type within `provider`.
    pub request_type: String,
    /// Version of the provider-defined request payload.
    pub schema_version: String,
    /// Provider-defined payload; defaults to an empty object.
    #[serde(default)]
    pub payload: JsonObject,
    /// Provider session the request belongs to, when the provider has one.
    #[serde(default)]
    pub provider_session_id: Option<String>,
    /// Turn the request belongs to, when the provider has one.
    #[serde(default)]
    pub turn_id: Option<String>,
    /// Unix-epoch-second deadline; when present it must be finite and non-negative.
    #[serde(default)]
    pub deadline_at: Option<f64>,
}

impl Validate for ProviderInteractionRequest {
    fn validate(&self) -> HarnessResult<()> {
        validate_request_common(
            &self.request_id,
            self.provider_session_id.as_deref(),
            self.turn_id.as_deref(),
            self.deadline_at,
        )?;
        if self.provider.is_empty()
            || self.request_type.is_empty()
            || self.schema_version.is_empty()
        {
            return Err(HarnessError::validation(
                "provider interaction namespace, type, and version must not be empty",
            ));
        }
        Ok(())
    }
}

/// Decision returned for a [`ToolApprovalRequest`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct ToolApprovalResponse {
    /// Identifier of the request this response answers.
    pub request_id: String,
    /// Host decision for the pending call.
    pub decision: ToolApprovalDecision,
    /// Rewritten arguments to execute instead of the requested ones, when the host adjusted them.
    #[serde(default)]
    pub updated_arguments: Option<JsonObject>,
    /// Optional human-readable explanation of the decision.
    #[serde(default)]
    pub reason: Option<String>,
    /// Whether the decision interrupts the active turn (Python default `false`).
    #[serde(default)]
    pub interrupt: bool,
    /// Provider extension data; defaults to an empty object.
    #[serde(default)]
    pub provider_data: JsonObject,
}

impl Validate for ToolApprovalResponse {
    fn validate(&self) -> HarnessResult<()> {
        validate_response_id(&self.request_id)
    }
}

/// Response returned for a [`UserInputRequest`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct UserInputResponse {
    /// Identifier of the request this response answers.
    pub request_id: String,
    /// Completion status of the interaction.
    pub status: InteractionResponseStatus,
    /// The user's answer; defaults to JSON `null`.
    #[serde(default)]
    pub content: JsonValue,
    /// Provider extension data; defaults to an empty object.
    #[serde(default)]
    pub provider_data: JsonObject,
}

impl Validate for UserInputResponse {
    fn validate(&self) -> HarnessResult<()> {
        validate_response_id(&self.request_id)
    }
}

/// Response returned for a [`McpElicitationRequest`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct McpElicitationResponse {
    /// Identifier of the request this response answers.
    pub request_id: String,
    /// Completion status of the interaction.
    pub status: InteractionResponseStatus,
    /// Elicited values keyed by field name; defaults to an empty object.
    #[serde(default)]
    pub values: JsonObject,
    /// Provider extension data; defaults to an empty object.
    #[serde(default)]
    pub provider_data: JsonObject,
}

impl Validate for McpElicitationResponse {
    fn validate(&self) -> HarnessResult<()> {
        validate_response_id(&self.request_id)
    }
}

/// Response returned for a [`DynamicToolCallRequest`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct DynamicToolCallResponse {
    /// Identifier of the request this response answers.
    pub request_id: String,
    /// Completion status of the interaction.
    pub status: InteractionResponseStatus,
    /// The tool result; defaults to JSON `null`.
    #[serde(default)]
    pub result: JsonValue,
    /// Whether `result` describes a tool failure (Python default `false`).
    #[serde(default)]
    pub is_error: bool,
    /// Optional failure message, when the host reported an error.
    #[serde(default)]
    pub error_message: Option<String>,
    /// Provider extension data; defaults to an empty object.
    #[serde(default)]
    pub provider_data: JsonObject,
}

impl Validate for DynamicToolCallResponse {
    fn validate(&self) -> HarnessResult<()> {
        validate_response_id(&self.request_id)
    }
}

/// Response returned for a [`ProviderInteractionRequest`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct ProviderInteractionResponse {
    /// Identifier of the request this response answers.
    pub request_id: String,
    /// Completion status of the interaction.
    pub status: InteractionResponseStatus,
    /// Provider-defined response payload; defaults to an empty object.
    #[serde(default)]
    pub payload: JsonObject,
}

impl Validate for ProviderInteractionResponse {
    fn validate(&self) -> HarnessResult<()> {
        validate_response_id(&self.request_id)
    }
}

/// Any interaction request a harness can raise during a turn.
///
/// The `interaction_type` tag carries the `snake_case` variant name (`tool_approval`,
/// `user_input`, `mcp_elicitation`, `dynamic_tool_call`, `provider`). This tagged enum is the Rust
/// replacement for Python's `typing` union over the five request types.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "interaction_type", rename_all = "snake_case")]
pub enum HarnessInteractionRequest {
    /// See [`ToolApprovalRequest`].
    ToolApproval(ToolApprovalRequest),
    /// See [`UserInputRequest`].
    UserInput(UserInputRequest),
    /// See [`McpElicitationRequest`].
    McpElicitation(McpElicitationRequest),
    /// See [`DynamicToolCallRequest`].
    DynamicToolCall(DynamicToolCallRequest),
    /// See [`ProviderInteractionRequest`].
    Provider(ProviderInteractionRequest),
}

impl HarnessInteractionRequest {
    /// Returns the `request_id` of the wrapped request.
    #[must_use]
    pub fn request_id(&self) -> &str {
        match self {
            Self::ToolApproval(request) => &request.request_id,
            Self::UserInput(request) => &request.request_id,
            Self::McpElicitation(request) => &request.request_id,
            Self::DynamicToolCall(request) => &request.request_id,
            Self::Provider(request) => &request.request_id,
        }
    }

    /// Returns the Rust name of the wrapped request type.
    fn type_name(&self) -> &'static str {
        match self {
            Self::ToolApproval(_) => "ToolApprovalRequest",
            Self::UserInput(_) => "UserInputRequest",
            Self::McpElicitation(_) => "McpElicitationRequest",
            Self::DynamicToolCall(_) => "DynamicToolCallRequest",
            Self::Provider(_) => "ProviderInteractionRequest",
        }
    }

    /// Returns the Rust name of the response type Python pairs with this request.
    fn expected_response_type_name(&self) -> &'static str {
        match self {
            Self::ToolApproval(_) => "ToolApprovalResponse",
            Self::UserInput(_) => "UserInputResponse",
            Self::McpElicitation(_) => "McpElicitationResponse",
            Self::DynamicToolCall(_) => "DynamicToolCallResponse",
            Self::Provider(_) => "ProviderInteractionResponse",
        }
    }

    /// Checks `response` against Python's `validate_interaction_response` rules.
    ///
    /// The identity check runs first, then the response-kind check, exactly like the Python
    /// helper.
    ///
    /// # Errors
    ///
    /// Returns [`HarnessError::Protocol`] with `"interaction response id '{response_id}' does not
    /// match request '{request_id}'"` when the identifiers differ, or
    /// `"{RequestType} requires {ExpectedResponseType}, got {ResponseType}"` when the response is
    /// of the wrong kind.
    pub fn validate_response(&self, response: &HarnessInteractionResponse) -> HarnessResult<()> {
        let request_id = self.request_id();
        let response_id = response.request_id();
        if response_id != request_id {
            return Err(HarnessError::protocol(format!(
                "interaction response id '{response_id}' does not match request '{request_id}'"
            )));
        }
        let same_kind = matches!(
            (self, response),
            (
                Self::ToolApproval(_),
                HarnessInteractionResponse::ToolApproval(_)
            ) | (Self::UserInput(_), HarnessInteractionResponse::UserInput(_))
                | (
                    Self::McpElicitation(_),
                    HarnessInteractionResponse::McpElicitation(_)
                )
                | (
                    Self::DynamicToolCall(_),
                    HarnessInteractionResponse::DynamicToolCall(_)
                )
                | (Self::Provider(_), HarnessInteractionResponse::Provider(_))
        );
        if same_kind {
            return Ok(());
        }
        Err(HarnessError::protocol(format!(
            "{} requires {}, got {}",
            self.type_name(),
            self.expected_response_type_name(),
            response.type_name()
        )))
    }
}

impl Validate for HarnessInteractionRequest {
    fn validate(&self) -> HarnessResult<()> {
        match self {
            Self::ToolApproval(request) => request.validate(),
            Self::UserInput(request) => request.validate(),
            Self::McpElicitation(request) => request.validate(),
            Self::DynamicToolCall(request) => request.validate(),
            Self::Provider(request) => request.validate(),
        }
    }
}

impl From<ToolApprovalRequest> for HarnessInteractionRequest {
    fn from(request: ToolApprovalRequest) -> Self {
        Self::ToolApproval(request)
    }
}

impl From<UserInputRequest> for HarnessInteractionRequest {
    fn from(request: UserInputRequest) -> Self {
        Self::UserInput(request)
    }
}

impl From<McpElicitationRequest> for HarnessInteractionRequest {
    fn from(request: McpElicitationRequest) -> Self {
        Self::McpElicitation(request)
    }
}

impl From<DynamicToolCallRequest> for HarnessInteractionRequest {
    fn from(request: DynamicToolCallRequest) -> Self {
        Self::DynamicToolCall(request)
    }
}

impl From<ProviderInteractionRequest> for HarnessInteractionRequest {
    fn from(request: ProviderInteractionRequest) -> Self {
        Self::Provider(request)
    }
}

/// Any interaction response a host can return during a turn.
///
/// The `interaction_type` tag carries the `snake_case` variant name (`tool_approval`,
/// `user_input`, `mcp_elicitation`, `dynamic_tool_call`, `provider`). This tagged enum is the Rust
/// replacement for Python's `typing` union over the five response types.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "interaction_type", rename_all = "snake_case")]
pub enum HarnessInteractionResponse {
    /// See [`ToolApprovalResponse`].
    ToolApproval(ToolApprovalResponse),
    /// See [`UserInputResponse`].
    UserInput(UserInputResponse),
    /// See [`McpElicitationResponse`].
    McpElicitation(McpElicitationResponse),
    /// See [`DynamicToolCallResponse`].
    DynamicToolCall(DynamicToolCallResponse),
    /// See [`ProviderInteractionResponse`].
    Provider(ProviderInteractionResponse),
}

impl HarnessInteractionResponse {
    /// Returns the `request_id` the wrapped response answers.
    #[must_use]
    pub fn request_id(&self) -> &str {
        match self {
            Self::ToolApproval(response) => &response.request_id,
            Self::UserInput(response) => &response.request_id,
            Self::McpElicitation(response) => &response.request_id,
            Self::DynamicToolCall(response) => &response.request_id,
            Self::Provider(response) => &response.request_id,
        }
    }

    /// Returns the Rust name of the wrapped response type.
    fn type_name(&self) -> &'static str {
        match self {
            Self::ToolApproval(_) => "ToolApprovalResponse",
            Self::UserInput(_) => "UserInputResponse",
            Self::McpElicitation(_) => "McpElicitationResponse",
            Self::DynamicToolCall(_) => "DynamicToolCallResponse",
            Self::Provider(_) => "ProviderInteractionResponse",
        }
    }
}

impl Validate for HarnessInteractionResponse {
    fn validate(&self) -> HarnessResult<()> {
        match self {
            Self::ToolApproval(response) => response.validate(),
            Self::UserInput(response) => response.validate(),
            Self::McpElicitation(response) => response.validate(),
            Self::DynamicToolCall(response) => response.validate(),
            Self::Provider(response) => response.validate(),
        }
    }
}

impl From<ToolApprovalResponse> for HarnessInteractionResponse {
    fn from(response: ToolApprovalResponse) -> Self {
        Self::ToolApproval(response)
    }
}

impl From<UserInputResponse> for HarnessInteractionResponse {
    fn from(response: UserInputResponse) -> Self {
        Self::UserInput(response)
    }
}

impl From<McpElicitationResponse> for HarnessInteractionResponse {
    fn from(response: McpElicitationResponse) -> Self {
        Self::McpElicitation(response)
    }
}

impl From<DynamicToolCallResponse> for HarnessInteractionResponse {
    fn from(response: DynamicToolCallResponse) -> Self {
        Self::DynamicToolCall(response)
    }
}

impl From<ProviderInteractionResponse> for HarnessInteractionResponse {
    fn from(response: ProviderInteractionResponse) -> Self {
        Self::Provider(response)
    }
}

/// Validates request identity and the response shape required by its type.
///
/// The response is returned unchanged so callers can validate and forward it in one expression,
/// matching Python's `validate_interaction_response(request, response) -> response`.
///
/// # Errors
///
/// Returns [`HarnessError::Protocol`] when the response answers a different request or is not the
/// response type paired with `request`.
pub fn validate_interaction_response(
    request: &HarnessInteractionRequest,
    response: HarnessInteractionResponse,
) -> HarnessResult<HarnessInteractionResponse> {
    request.validate_response(&response)?;
    Ok(response)
}

/// Host service for request/response interactions during a live turn.
///
/// Implementations must return the response type paired with the request and the same
/// `request_id` before `deadline_at` when one is supplied.
/// `cancel` is idempotent and releases any pending host UI or policy operation for the request.
/// Harnesses must cancel every pending request before aborting their turn or completing `stop`.
#[async_trait]
pub trait HarnessInteractionHandler: Send + Sync {
    /// Waits for and returns the host response to `request`.
    ///
    /// # Errors
    ///
    /// Returns [`HarnessError`] when the request cannot be answered (host unavailable, deadline
    /// exceeded, or the provider cancelled it).
    async fn handle(
        &self,
        request: &HarnessInteractionRequest,
    ) -> HarnessResult<HarnessInteractionResponse>;

    /// Cancels a pending interaction, releasing the host UI or policy operation waiting on it.
    ///
    /// Implementations must be idempotent: cancelling an unknown or already settled request is
    /// not an error.
    ///
    /// # Errors
    ///
    /// Returns [`HarnessError`] only when the cancellation itself cannot be delivered.
    async fn cancel(&self, request_id: &str, reason: InteractionCancelReason) -> HarnessResult<()>;

    /// Cancels a pending interaction because the provider withdrew it.
    ///
    /// Convenience for Python's `reason=InteractionCancelReason.PROVIDER_WITHDREW` keyword default.
    ///
    /// # Errors
    ///
    /// Propagates the error from [`HarnessInteractionHandler::cancel`].
    async fn cancel_default(&self, request_id: &str) -> HarnessResult<()> {
        self.cancel(request_id, InteractionCancelReason::ProviderWithdrew)
            .await
    }
}

#[cfg(test)]
mod tests {
    // Tests assert by panicking on failure, which is the intent of a test.
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use serde_json::json;

    use super::{
        validate_interaction_response, HarnessInteractionRequest, HarnessInteractionResponse,
        InteractionCancelReason, InteractionResponseStatus, ToolApprovalDecision,
        ToolApprovalRequest, ToolApprovalResponse, UserInputRequest, UserInputResponse,
    };
    use crate::json::JsonObject;
    use crate::{
        empty_json_object, from_str_validated, from_value_validated, HarnessErrorKind, Validate,
    };

    fn approval_request() -> ToolApprovalRequest {
        ToolApprovalRequest {
            request_id: "req-1".to_owned(),
            call_id: "call-1".to_owned(),
            tool_name: "write_file".to_owned(),
            arguments: empty_json_object(),
            provider_session_id: None,
            turn_id: None,
            title: None,
            description: None,
            reason: None,
            deadline_at: None,
            provider_data: empty_json_object(),
        }
    }

    fn approval_response(decision: ToolApprovalDecision) -> ToolApprovalResponse {
        ToolApprovalResponse {
            request_id: "req-1".to_owned(),
            decision,
            updated_arguments: None,
            reason: None,
            interrupt: false,
            provider_data: empty_json_object(),
        }
    }

    #[test]
    fn request_common_validation_messages() {
        for (request, expected) in [
            (
                ToolApprovalRequest {
                    request_id: String::new(),
                    ..approval_request()
                },
                "interaction request_id must not be empty",
            ),
            (
                ToolApprovalRequest {
                    provider_session_id: Some(String::new()),
                    ..approval_request()
                },
                "interaction provider_session_id must not be empty",
            ),
            (
                ToolApprovalRequest {
                    turn_id: Some(String::new()),
                    ..approval_request()
                },
                "interaction turn_id must not be empty",
            ),
            (
                ToolApprovalRequest {
                    deadline_at: Some(-1.0),
                    ..approval_request()
                },
                "interaction deadline_at must be a finite Unix timestamp",
            ),
        ] {
            let error = request.validate().expect_err(expected);
            assert_eq!(error.kind(), HarnessErrorKind::Validation);
            assert_eq!(error.message(), expected);
        }

        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let request = ToolApprovalRequest {
                deadline_at: Some(bad),
                ..approval_request()
            };
            assert_eq!(
                request
                    .validate()
                    .expect_err("non-finite deadline")
                    .message(),
                "interaction deadline_at must be a finite Unix timestamp"
            );
        }

        ToolApprovalRequest {
            deadline_at: Some(0.0),
            ..approval_request()
        }
        .validate()
        .expect("a zero deadline is valid");
    }

    #[test]
    fn request_type_specific_validation_messages() {
        for (wire, expected) in [
            (
                r#"{"interaction_type": "tool_approval", "request_id": "req",
                    "call_id": "", "tool_name": "write_file"}"#,
                "tool approval call_id and tool_name must not be empty",
            ),
            (
                r#"{"interaction_type": "user_input", "request_id": "req", "prompt": ""}"#,
                "user input prompt must not be empty",
            ),
            (
                r#"{"interaction_type": "mcp_elicitation", "request_id": "req",
                    "server_name": "files", "prompt": ""}"#,
                "MCP elicitation server_name and prompt must not be empty",
            ),
            (
                r#"{"interaction_type": "dynamic_tool_call", "request_id": "req",
                    "call_id": "call", "tool_name": ""}"#,
                "dynamic tool call_id and tool_name must not be empty",
            ),
            (
                r#"{"interaction_type": "provider", "request_id": "req", "provider": "acme",
                    "request_type": "pick", "schema_version": ""}"#,
                "provider interaction namespace, type, and version must not be empty",
            ),
        ] {
            let error = from_str_validated::<HarnessInteractionRequest>(wire).expect_err(expected);
            assert_eq!(error.kind(), HarnessErrorKind::Validation);
            assert_eq!(error.message(), expected);
        }
    }

    #[test]
    fn decode_applies_python_defaults_and_requires_python_required_fields() {
        let request: ToolApprovalRequest =
            from_str_validated(r#"{"request_id": "req", "call_id": "call", "tool_name": "read"}"#)
                .expect("valid request");
        assert_eq!(request.arguments, JsonObject::new());
        assert_eq!(request.provider_data, JsonObject::new());
        assert!(request.provider_session_id.is_none());
        assert!(request.deadline_at.is_none());

        let input: UserInputRequest =
            from_str_validated(r#"{"request_id": "req", "prompt": "name?"}"#)
                .expect("valid request");
        assert_eq!(input.choices, Vec::<String>::new());

        let approval: ToolApprovalResponse =
            from_str_validated(r#"{"request_id": "req", "decision": "allow_for_session"}"#)
                .expect("valid response");
        assert_eq!(approval.decision, ToolApprovalDecision::AllowForSession);
        assert!(!approval.interrupt);
        assert!(approval.updated_arguments.is_none());

        let input: UserInputResponse =
            from_str_validated(r#"{"request_id": "req", "status": "declined"}"#)
                .expect("valid response");
        assert_eq!(input.status, InteractionResponseStatus::Declined);
        assert_eq!(input.content, serde_json::Value::Null);
        assert_eq!(input.provider_data, JsonObject::new());

        let error = from_str_validated::<ToolApprovalRequest>(r#"{"request_id": "req"}"#)
            .expect_err("call_id and tool_name are required");
        assert_eq!(error.kind(), HarnessErrorKind::Protocol);
        let error = from_str_validated::<UserInputResponse>(r#"{"status": "completed"}"#)
            .expect_err("request_id is required");
        assert_eq!(error.kind(), HarnessErrorKind::Protocol);
    }

    #[test]
    fn dispatch_enums_tag_request_id_and_validate() {
        let request = HarnessInteractionRequest::from(approval_request());
        assert_eq!(request.request_id(), "req-1");
        let encoded = serde_json::to_value(&request).expect("serializes");
        assert_eq!(encoded["interaction_type"], json!("tool_approval"));
        assert_eq!(
            from_value_validated::<HarnessInteractionRequest>(encoded).expect("round trip"),
            request
        );

        let response =
            HarnessInteractionResponse::from(approval_response(ToolApprovalDecision::Abort));
        assert_eq!(response.request_id(), "req-1");
        let encoded = serde_json::to_value(&response).expect("serializes");
        assert_eq!(encoded["interaction_type"], json!("tool_approval"));
        assert_eq!(
            from_value_validated::<HarnessInteractionResponse>(encoded).expect("round trip"),
            response
        );

        let broken = HarnessInteractionRequest::from(ToolApprovalRequest {
            request_id: String::new(),
            ..approval_request()
        });
        assert_eq!(
            broken.validate().expect_err("empty id").message(),
            "interaction request_id must not be empty"
        );
    }

    #[test]
    fn validate_interaction_response_checks_id_then_kind() {
        let request = HarnessInteractionRequest::from(approval_request());
        let matching =
            HarnessInteractionResponse::from(approval_response(ToolApprovalDecision::Deny));
        assert_eq!(
            validate_interaction_response(&request, matching.clone()).expect("matching response"),
            matching
        );

        let mismatched_id = HarnessInteractionResponse::from(UserInputResponse {
            request_id: "req-2".to_owned(),
            status: InteractionResponseStatus::Completed,
            content: json!("answer"),
            provider_data: empty_json_object(),
        });
        assert_eq!(
            validate_interaction_response(&request, mismatched_id)
                .expect_err("id mismatch")
                .message(),
            "interaction response id 'req-2' does not match request 'req-1'"
        );

        let mismatched_kind = HarnessInteractionResponse::from(UserInputResponse {
            request_id: "req-1".to_owned(),
            status: InteractionResponseStatus::Completed,
            content: json!("answer"),
            provider_data: empty_json_object(),
        });
        let error =
            validate_interaction_response(&request, mismatched_kind).expect_err("kind mismatch");
        assert_eq!(error.kind(), HarnessErrorKind::Protocol);
        assert_eq!(
            error.message(),
            "ToolApprovalRequest requires ToolApprovalResponse, got UserInputResponse"
        );
    }

    #[test]
    fn enum_wire_values_match_python() {
        for (status, wire) in [
            (InteractionResponseStatus::Completed, "completed"),
            (InteractionResponseStatus::Declined, "declined"),
            (InteractionResponseStatus::Cancelled, "cancelled"),
            (InteractionResponseStatus::Failed, "failed"),
        ] {
            assert_eq!(
                serde_json::to_value(status).expect("serializes"),
                json!(wire)
            );
        }
        for (reason, wire) in [
            (
                InteractionCancelReason::ProviderWithdrew,
                "provider_withdrew",
            ),
            (InteractionCancelReason::TurnAborted, "turn_aborted"),
            (InteractionCancelReason::HarnessStopped, "harness_stopped"),
            (
                InteractionCancelReason::DeadlineExceeded,
                "deadline_exceeded",
            ),
        ] {
            assert_eq!(
                serde_json::to_value(reason).expect("serializes"),
                json!(wire)
            );
        }
        for (decision, wire) in [
            (ToolApprovalDecision::Allow, "allow"),
            (ToolApprovalDecision::AllowForSession, "allow_for_session"),
            (ToolApprovalDecision::Deny, "deny"),
            (ToolApprovalDecision::Abort, "abort"),
        ] {
            assert_eq!(
                serde_json::to_value(decision).expect("serializes"),
                json!(wire)
            );
        }
    }
}
