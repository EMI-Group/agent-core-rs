//! Provider-neutral harness SPI between a host (the openJiuwen agent runtime) and a harness
//! provider (native openJiuwen harness, Claude Code, Codex, DSH, …).
//!
//! This crate is the single definition of the harness wire vocabulary: harness handles and
//! capability negotiation, events and the event wire codec, turn results, checkpoints, hooks,
//! interactions, tools and MCP configuration, event cursors/streams, and the behavioral protocol
//! traits (`HarnessProtocol`, `HarnessModelControl`, `HarnessProvider`, `ModelConfigProvider`).
//!
//! It has **zero workspace dependencies** — only `serde`, `serde_json`, `schemars`, `thiserror`,
//! `async-trait`, and `futures`. There is deliberately no runtime dependency: no `tokio`, no
//! clock, and no identifier generation; providers supply the `sequence` number and Unix-epoch-second
//! timestamps for every event.
//!
//! Parity source: the Python package `openjiuwen.harness_protocol` (v0.1.20). `DESIGN.md` in this
//! directory is the normative field-by-field contract; `CONTEXT.md` records the deliberate
//! deviations from Python.

pub mod checkpoints;
pub mod errors;
pub mod events;
pub mod hooks;
pub mod interactions;
pub mod json;
pub mod models;
pub mod protocol;
pub mod results;
pub mod serialization;
pub mod state;
pub mod stream;
pub mod tools;
pub mod validation;

pub use checkpoints::{
    CheckpointReason, CheckpointSaveReceipt, HarnessCheckpoint, HarnessCheckpointSink,
    MAX_CHECKPOINT_BYTES,
};
pub use errors::{HarnessError, HarnessErrorKind, HarnessResult};
pub use events::{
    event_retention, DiagnosticEvent, DiagnosticLevel, EventBufferConfig, EventOverflowPolicy,
    EventRetention, HarnessEvent, HarnessEventPayload, HookEventPhase, HookObservedEvent,
    ItemEventKind, ItemLifecycleEvent, ModelRequestEvent, ModelRequestStatus, OutputChannel,
    OutputEvent, OutputKind, OutputOperation, ProviderEvent, StateChangedEvent, TurnEventKind,
    TurnLifecycleEvent, UnknownEvent, UsageUpdateMode, UsageUpdatedEvent,
    TERMINAL_TURN_EVENT_KINDS,
};
pub use hooks::{
    AfterToolContext, AfterToolResult, BeforePromptContext, BeforePromptResult, BeforeToolContext,
    HarnessHookDispatcher, StopHookContext, ToolDecision, ToolDecisionKind,
};
pub use interactions::{
    validate_interaction_response, DynamicToolCallRequest, DynamicToolCallResponse,
    HarnessInteractionHandler, HarnessInteractionRequest, HarnessInteractionResponse,
    InteractionCancelReason, InteractionResponseStatus, McpElicitationRequest,
    McpElicitationResponse, ProviderInteractionRequest, ProviderInteractionResponse,
    ToolApprovalDecision, ToolApprovalRequest, ToolApprovalResponse, UserInputRequest,
    UserInputResponse,
};
pub use json::{
    empty_json_object, from_str_validated, from_value_validated, json_object_from_value,
    json_object_to_value, JsonObject, JsonValue,
};
pub use models::{
    default_compatible_protocol_versions, AbortMode, DeliveryMode, HarnessCapability, HarnessCard,
    HarnessContext, HarnessInput, HarnessTelemetry, HostCapability, ModelChoice, ModelEndpoint,
    ModelOption, ModelSelection, ResumePolicy, SendReceipt, AUTH_FALLBACK_EVENT_TYPE,
    AUTH_FALLBACK_REQUEST_TYPE, PROTOCOL_VERSION,
};
pub use protocol::{
    DynHarness, DynHarnessProvider, DynModelControl, HarnessModelControl, HarnessProtocol,
    HarnessProvider, ModelConfigProvider,
};
pub use results::{
    ContentBlock, MessageRole, MonetaryAmount, TurnError, TurnMessage, TurnResult, TurnStatus,
    TurnTermination, TurnTerminationKind, TurnUsage,
};
pub use serialization::{
    harness_event_from_dict, harness_event_from_json_str, harness_event_to_dict,
    harness_event_to_json_string, EVENT_WIRE_SCHEMA_VERSION,
};
pub use state::HarnessState;
pub use stream::{
    event_cursor_stream, EventCursorStream, HarnessEventCursor, HarnessEventStream, VecEventCursor,
};
pub use tools::{
    McpInstance, McpServerConfig, McpTransport, ToolDefinition, ToolExecutionResult, ToolGateway,
    ToolInvocation,
};
pub use validation::{
    join_sorted, require_finite_non_negative, require_finite_timestamp, require_non_empty,
    require_optional_non_empty, require_subset, require_unique_non_empty, Validate,
};

/// Returns the crate name.
#[must_use]
pub fn crate_name() -> &'static str {
    "openjiuwen-harness-protocol"
}
