//! The callback event vocabulary.
//!
//! Ports `openjiuwen/core/runner/callback/events.py`: every preset event is a
//! typed variant whose [`CallbackEvent::as_str`] is the exact Python wire value
//! (`"_framework:<name>"`), so a harness that subscribes by string, a guardrail
//! that binds through [`openjiuwen_security::ScanStage::event`], and this crate
//! all agree without a translation table.
//!
//! Custom event names stay expressible through [`CallbackEvent::Custom`]; their
//! name is used verbatim, so a host may write its own scoped names.
#![allow(clippy::module_name_repetitions)]
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::unnecessary_wraps
    )
)]

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;

/// Python's default event scope (`DEFAULT_SCOPE`).
pub const DEFAULT_SCOPE: &str = "_framework";

/// The `"<scope>:"` prefix every preset event name carries.
pub const SCOPE_PREFIX: &str = "_framework:";

/// One observable lifecycle point.
///
/// The variant list is the union of Python's `events.py` classes; the mapping
/// to wire names is in [`CallbackEvent::name`] (unscoped) and
/// [`CallbackEvent::as_str`] (scoped).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum CallbackEvent {
    /// Agent execution started (`AgentEvents.AGENT_STARTED`).
    AgentStarted,
    /// Arguments of an agent invocation, before the agent runs.
    AgentInvokeInput,
    /// Result of an agent invocation.
    AgentInvokeOutput,
    /// Arguments of a streaming agent invocation.
    AgentStreamInput,
    /// One frame yielded by a streaming agent invocation.
    AgentStreamOutput,
    /// An agent received a call from another agent over P2P.
    AgentP2pReceived,
    /// An agent received a call from another agent over pub/sub.
    AgentPubsubReceived,

    /// Workflow execution started.
    WorkflowStarted,
    /// Workflow execution completed successfully.
    WorkflowFinished,
    /// Workflow execution failed.
    WorkflowError,
    /// Workflow execution was cancelled.
    WorkflowCancelled,
    /// A workflow node completed execution.
    WorkflowNodeExecuted,
    /// A workflow node failed.
    WorkflowNodeError,
    /// A workflow edge was traversed.
    WorkflowEdgeTraversed,
    /// A workflow loop started.
    WorkflowLoopStarted,
    /// A workflow loop completed.
    WorkflowLoopFinished,
    /// Arguments of a workflow invocation.
    WorkflowInvokeInput,
    /// Result of a workflow invocation.
    WorkflowInvokeOutput,
    /// Arguments of a streaming workflow invocation.
    WorkflowStreamInput,
    /// One frame yielded by a streaming workflow invocation.
    WorkflowStreamOutput,
    /// Input batch of a workflow component.
    WorkflowComponentBatchInput,
    /// Output batch of a workflow component.
    WorkflowComponentBatchOutput,
    /// Input of a streaming workflow component.
    WorkflowComponentStreamInput,
    /// One frame yielded by a streaming workflow component.
    WorkflowComponentStreamOutput,

    /// An LLM call was initiated.
    LlmCallStarted,
    /// An LLM call failed.
    LlmCallError,
    /// An LLM response was received while streaming.
    LlmResponseReceived,
    /// Response headers arrived for a streaming request.
    LlmResponseStarted,
    /// Arguments of a model invocation.
    LlmInvokeInput,
    /// Result of a model invocation.
    LlmInvokeOutput,
    /// Arguments of a streaming model invocation.
    LlmStreamInput,
    /// One chunk yielded by a streaming model invocation.
    LlmStreamOutput,
    /// A model stream was exhausted; carries the accumulated message.
    LlmStreamCompleted,
    /// A model-group router selected a final route (or exhausted all routes).
    LlmRoute,
    /// A model-group router sent a request to one route.
    LlmRouteAttempt,
    /// The messages/tools request data of an LLM call.
    LlmInput,
    /// The response/usage data of an LLM call.
    LlmOutput,

    /// A tool call was initiated.
    ToolCallStarted,
    /// A tool call completed successfully.
    ToolCallFinished,
    /// A tool call failed.
    ToolCallError,
    /// A tool result was received.
    ToolResultReceived,
    /// Tool result parsing started.
    ToolParseStarted,
    /// Tool result parsing completed.
    ToolParseFinished,
    /// Arguments of a tool invocation.
    ToolInvokeInput,
    /// Result of a tool invocation.
    ToolInvokeOutput,
    /// Arguments of a streaming tool invocation.
    ToolStreamInput,
    /// One chunk yielded by a streaming tool invocation.
    ToolStreamOutput,
    /// A tool requests authentication configuration.
    ToolAuth,

    /// The context window was updated.
    ContextUpdated,
    /// Context was offloaded to storage.
    ContextOffloaded,
    /// Context was retrieved from storage.
    ContextRetrieved,
    /// Context was cleared.
    ContextCleared,
    /// Context compression changed state (Python keeps its literal dot).
    ContextCompressionState,

    /// A session was created.
    SessionCreated,
    /// An agent session was created by the runner.
    AgentSessionCreated,

    /// Knowledge retrieval started.
    RetrievalStarted,

    /// A memory entry was added.
    MemoryAdded,
    /// A memory search started.
    MemorySearchStarted,
    /// A memory search finished.
    MemorySearchFinished,
    /// A memory entry was updated.
    MemoryUpdated,
    /// A memory entry was deleted.
    MemoryDeleted,

    /// A task was created.
    TaskCreated,
    /// A task started running.
    TaskRunning,
    /// A task completed successfully.
    TaskCompleted,
    /// A task failed.
    TaskFailed,
    /// A task was cancelled.
    TaskCancelled,
    /// A task timed out.
    TaskTimeout,

    /// An event name this crate does not model.
    ///
    /// The string is used verbatim by [`CallbackEvent::as_str`] and
    /// [`CallbackEvent::name`]; conventionally it is already scoped
    /// (`"my_scope:my_event"`).
    Custom(String),
}

impl CallbackEvent {
    /// Every preset (non-[`Custom`](Self::Custom)) event, in declaration order.
    ///
    /// Used for iteration and as the round-trip corpus of the name mapping.
    pub const ALL: &'static [CallbackEvent] = &[
        Self::AgentStarted,
        Self::AgentInvokeInput,
        Self::AgentInvokeOutput,
        Self::AgentStreamInput,
        Self::AgentStreamOutput,
        Self::AgentP2pReceived,
        Self::AgentPubsubReceived,
        Self::WorkflowStarted,
        Self::WorkflowFinished,
        Self::WorkflowError,
        Self::WorkflowCancelled,
        Self::WorkflowNodeExecuted,
        Self::WorkflowNodeError,
        Self::WorkflowEdgeTraversed,
        Self::WorkflowLoopStarted,
        Self::WorkflowLoopFinished,
        Self::WorkflowInvokeInput,
        Self::WorkflowInvokeOutput,
        Self::WorkflowStreamInput,
        Self::WorkflowStreamOutput,
        Self::WorkflowComponentBatchInput,
        Self::WorkflowComponentBatchOutput,
        Self::WorkflowComponentStreamInput,
        Self::WorkflowComponentStreamOutput,
        Self::LlmCallStarted,
        Self::LlmCallError,
        Self::LlmResponseReceived,
        Self::LlmResponseStarted,
        Self::LlmInvokeInput,
        Self::LlmInvokeOutput,
        Self::LlmStreamInput,
        Self::LlmStreamOutput,
        Self::LlmStreamCompleted,
        Self::LlmRoute,
        Self::LlmRouteAttempt,
        Self::LlmInput,
        Self::LlmOutput,
        Self::ToolCallStarted,
        Self::ToolCallFinished,
        Self::ToolCallError,
        Self::ToolResultReceived,
        Self::ToolParseStarted,
        Self::ToolParseFinished,
        Self::ToolInvokeInput,
        Self::ToolInvokeOutput,
        Self::ToolStreamInput,
        Self::ToolStreamOutput,
        Self::ToolAuth,
        Self::ContextUpdated,
        Self::ContextOffloaded,
        Self::ContextRetrieved,
        Self::ContextCleared,
        Self::ContextCompressionState,
        Self::SessionCreated,
        Self::AgentSessionCreated,
        Self::RetrievalStarted,
        Self::MemoryAdded,
        Self::MemorySearchStarted,
        Self::MemorySearchFinished,
        Self::MemoryUpdated,
        Self::MemoryDeleted,
        Self::TaskCreated,
        Self::TaskRunning,
        Self::TaskCompleted,
        Self::TaskFailed,
        Self::TaskCancelled,
        Self::TaskTimeout,
    ];

    /// The unscoped event name, e.g. `"agent_invoke_input"`.
    ///
    /// For [`Custom`](Self::Custom) this is the stored string unchanged.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::AgentStarted => "agent_started",
            Self::AgentInvokeInput => "agent_invoke_input",
            Self::AgentInvokeOutput => "agent_invoke_output",
            Self::AgentStreamInput => "agent_stream_input",
            Self::AgentStreamOutput => "agent_stream_output",
            Self::AgentP2pReceived => "agent_p2p_received",
            Self::AgentPubsubReceived => "agent_pubsub_received",
            Self::WorkflowStarted => "workflow_started",
            Self::WorkflowFinished => "workflow_finished",
            Self::WorkflowError => "workflow_error",
            Self::WorkflowCancelled => "workflow_cancelled",
            Self::WorkflowNodeExecuted => "workflow_node_executed",
            Self::WorkflowNodeError => "workflow_node_error",
            Self::WorkflowEdgeTraversed => "workflow_edge_traversed",
            Self::WorkflowLoopStarted => "workflow_loop_started",
            Self::WorkflowLoopFinished => "workflow_loop_finished",
            Self::WorkflowInvokeInput => "workflow_invoke_input",
            Self::WorkflowInvokeOutput => "workflow_invoke_output",
            Self::WorkflowStreamInput => "workflow_stream_input",
            Self::WorkflowStreamOutput => "workflow_stream_output",
            Self::WorkflowComponentBatchInput => "workflow_component_batch_input",
            Self::WorkflowComponentBatchOutput => "workflow_component_batch_output",
            Self::WorkflowComponentStreamInput => "workflow_component_stream_input",
            Self::WorkflowComponentStreamOutput => "workflow_component_stream_output",
            Self::LlmCallStarted => "llm_call_started",
            Self::LlmCallError => "llm_call_error",
            Self::LlmResponseReceived => "llm_response_received",
            Self::LlmResponseStarted => "llm_response_started",
            Self::LlmInvokeInput => "llm_invoke_input",
            Self::LlmInvokeOutput => "llm_invoke_output",
            Self::LlmStreamInput => "llm_stream_input",
            Self::LlmStreamOutput => "llm_stream_output",
            Self::LlmStreamCompleted => "llm_stream_completed",
            Self::LlmRoute => "llm_route",
            Self::LlmRouteAttempt => "llm_route_attempt",
            Self::LlmInput => "llm_input",
            Self::LlmOutput => "llm_output",
            Self::ToolCallStarted => "tool_call_started",
            Self::ToolCallFinished => "tool_call_finished",
            Self::ToolCallError => "tool_call_error",
            Self::ToolResultReceived => "tool_result_received",
            Self::ToolParseStarted => "tool_parse_started",
            Self::ToolParseFinished => "tool_parse_finished",
            Self::ToolInvokeInput => "tool_invoke_input",
            Self::ToolInvokeOutput => "tool_invoke_output",
            Self::ToolStreamInput => "tool_stream_input",
            Self::ToolStreamOutput => "tool_stream_output",
            Self::ToolAuth => "tool_auth",
            Self::ContextUpdated => "context_updated",
            Self::ContextOffloaded => "context_offloaded",
            Self::ContextRetrieved => "context_retrieved",
            Self::ContextCleared => "context_cleared",
            Self::ContextCompressionState => "context.compression_state",
            Self::SessionCreated => "session_created",
            Self::AgentSessionCreated => "agent_session_created",
            Self::RetrievalStarted => "retrieval_started",
            Self::MemoryAdded => "memory_added",
            Self::MemorySearchStarted => "memory_search_started",
            Self::MemorySearchFinished => "memory_search_finished",
            Self::MemoryUpdated => "memory_updated",
            Self::MemoryDeleted => "memory_deleted",
            Self::TaskCreated => "task_created",
            Self::TaskRunning => "task_running",
            Self::TaskCompleted => "task_completed",
            Self::TaskFailed => "task_failed",
            Self::TaskCancelled => "task_cancelled",
            Self::TaskTimeout => "task_timeout",
            Self::Custom(name) => name,
        }
    }

    /// The scoped wire value, e.g. `"_framework:agent_invoke_input"`.
    ///
    /// This is the string subscription code uses and the value
    /// [`openjiuwen_security::ScanStage::event`] returns for a guarded stage.
    /// A [`Custom`](Self::Custom) name is returned unchanged: custom events own
    /// their scoping.
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::AgentStarted => "_framework:agent_started",
            Self::AgentInvokeInput => "_framework:agent_invoke_input",
            Self::AgentInvokeOutput => "_framework:agent_invoke_output",
            Self::AgentStreamInput => "_framework:agent_stream_input",
            Self::AgentStreamOutput => "_framework:agent_stream_output",
            Self::AgentP2pReceived => "_framework:agent_p2p_received",
            Self::AgentPubsubReceived => "_framework:agent_pubsub_received",
            Self::WorkflowStarted => "_framework:workflow_started",
            Self::WorkflowFinished => "_framework:workflow_finished",
            Self::WorkflowError => "_framework:workflow_error",
            Self::WorkflowCancelled => "_framework:workflow_cancelled",
            Self::WorkflowNodeExecuted => "_framework:workflow_node_executed",
            Self::WorkflowNodeError => "_framework:workflow_node_error",
            Self::WorkflowEdgeTraversed => "_framework:workflow_edge_traversed",
            Self::WorkflowLoopStarted => "_framework:workflow_loop_started",
            Self::WorkflowLoopFinished => "_framework:workflow_loop_finished",
            Self::WorkflowInvokeInput => "_framework:workflow_invoke_input",
            Self::WorkflowInvokeOutput => "_framework:workflow_invoke_output",
            Self::WorkflowStreamInput => "_framework:workflow_stream_input",
            Self::WorkflowStreamOutput => "_framework:workflow_stream_output",
            Self::WorkflowComponentBatchInput => "_framework:workflow_component_batch_input",
            Self::WorkflowComponentBatchOutput => "_framework:workflow_component_batch_output",
            Self::WorkflowComponentStreamInput => "_framework:workflow_component_stream_input",
            Self::WorkflowComponentStreamOutput => "_framework:workflow_component_stream_output",
            Self::LlmCallStarted => "_framework:llm_call_started",
            Self::LlmCallError => "_framework:llm_call_error",
            Self::LlmResponseReceived => "_framework:llm_response_received",
            Self::LlmResponseStarted => "_framework:llm_response_started",
            Self::LlmInvokeInput => "_framework:llm_invoke_input",
            Self::LlmInvokeOutput => "_framework:llm_invoke_output",
            Self::LlmStreamInput => "_framework:llm_stream_input",
            Self::LlmStreamOutput => "_framework:llm_stream_output",
            Self::LlmStreamCompleted => "_framework:llm_stream_completed",
            Self::LlmRoute => "_framework:llm_route",
            Self::LlmRouteAttempt => "_framework:llm_route_attempt",
            Self::LlmInput => "_framework:llm_input",
            Self::LlmOutput => "_framework:llm_output",
            Self::ToolCallStarted => "_framework:tool_call_started",
            Self::ToolCallFinished => "_framework:tool_call_finished",
            Self::ToolCallError => "_framework:tool_call_error",
            Self::ToolResultReceived => "_framework:tool_result_received",
            Self::ToolParseStarted => "_framework:tool_parse_started",
            Self::ToolParseFinished => "_framework:tool_parse_finished",
            Self::ToolInvokeInput => "_framework:tool_invoke_input",
            Self::ToolInvokeOutput => "_framework:tool_invoke_output",
            Self::ToolStreamInput => "_framework:tool_stream_input",
            Self::ToolStreamOutput => "_framework:tool_stream_output",
            Self::ToolAuth => "_framework:tool_auth",
            Self::ContextUpdated => "_framework:context_updated",
            Self::ContextOffloaded => "_framework:context_offloaded",
            Self::ContextRetrieved => "_framework:context_retrieved",
            Self::ContextCleared => "_framework:context_cleared",
            Self::ContextCompressionState => "_framework:context.compression_state",
            Self::SessionCreated => "_framework:session_created",
            Self::AgentSessionCreated => "_framework:agent_session_created",
            Self::RetrievalStarted => "_framework:retrieval_started",
            Self::MemoryAdded => "_framework:memory_added",
            Self::MemorySearchStarted => "_framework:memory_search_started",
            Self::MemorySearchFinished => "_framework:memory_search_finished",
            Self::MemoryUpdated => "_framework:memory_updated",
            Self::MemoryDeleted => "_framework:memory_deleted",
            Self::TaskCreated => "_framework:task_created",
            Self::TaskRunning => "_framework:task_running",
            Self::TaskCompleted => "_framework:task_completed",
            Self::TaskFailed => "_framework:task_failed",
            Self::TaskCancelled => "_framework:task_cancelled",
            Self::TaskTimeout => "_framework:task_timeout",
            Self::Custom(name) => name,
        }
    }

    /// Parses a preset event name, accepting it with or without the
    /// `"_framework:"` scope prefix.
    ///
    /// Returns `None` for a name this crate does not model; such a name is
    /// representable as [`Custom`](Self::Custom) instead. Every element of
    /// [`ALL`](Self::ALL) round-trips: `from_name(event.as_str()) == Some(event)`.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        let raw = name.strip_prefix(SCOPE_PREFIX).unwrap_or(name);
        Some(match raw {
            "agent_started" => Self::AgentStarted,
            "agent_invoke_input" => Self::AgentInvokeInput,
            "agent_invoke_output" => Self::AgentInvokeOutput,
            "agent_stream_input" => Self::AgentStreamInput,
            "agent_stream_output" => Self::AgentStreamOutput,
            "agent_p2p_received" => Self::AgentP2pReceived,
            "agent_pubsub_received" => Self::AgentPubsubReceived,
            "workflow_started" => Self::WorkflowStarted,
            "workflow_finished" => Self::WorkflowFinished,
            "workflow_error" => Self::WorkflowError,
            "workflow_cancelled" => Self::WorkflowCancelled,
            "workflow_node_executed" => Self::WorkflowNodeExecuted,
            "workflow_node_error" => Self::WorkflowNodeError,
            "workflow_edge_traversed" => Self::WorkflowEdgeTraversed,
            "workflow_loop_started" => Self::WorkflowLoopStarted,
            "workflow_loop_finished" => Self::WorkflowLoopFinished,
            "workflow_invoke_input" => Self::WorkflowInvokeInput,
            "workflow_invoke_output" => Self::WorkflowInvokeOutput,
            "workflow_stream_input" => Self::WorkflowStreamInput,
            "workflow_stream_output" => Self::WorkflowStreamOutput,
            "workflow_component_batch_input" => Self::WorkflowComponentBatchInput,
            "workflow_component_batch_output" => Self::WorkflowComponentBatchOutput,
            "workflow_component_stream_input" => Self::WorkflowComponentStreamInput,
            "workflow_component_stream_output" => Self::WorkflowComponentStreamOutput,
            "llm_call_started" => Self::LlmCallStarted,
            "llm_call_error" => Self::LlmCallError,
            "llm_response_received" => Self::LlmResponseReceived,
            "llm_response_started" => Self::LlmResponseStarted,
            "llm_invoke_input" => Self::LlmInvokeInput,
            "llm_invoke_output" => Self::LlmInvokeOutput,
            "llm_stream_input" => Self::LlmStreamInput,
            "llm_stream_output" => Self::LlmStreamOutput,
            "llm_stream_completed" => Self::LlmStreamCompleted,
            "llm_route" => Self::LlmRoute,
            "llm_route_attempt" => Self::LlmRouteAttempt,
            "llm_input" => Self::LlmInput,
            "llm_output" => Self::LlmOutput,
            "tool_call_started" => Self::ToolCallStarted,
            "tool_call_finished" => Self::ToolCallFinished,
            "tool_call_error" => Self::ToolCallError,
            "tool_result_received" => Self::ToolResultReceived,
            "tool_parse_started" => Self::ToolParseStarted,
            "tool_parse_finished" => Self::ToolParseFinished,
            "tool_invoke_input" => Self::ToolInvokeInput,
            "tool_invoke_output" => Self::ToolInvokeOutput,
            "tool_stream_input" => Self::ToolStreamInput,
            "tool_stream_output" => Self::ToolStreamOutput,
            "tool_auth" => Self::ToolAuth,
            "context_updated" => Self::ContextUpdated,
            "context_offloaded" => Self::ContextOffloaded,
            "context_retrieved" => Self::ContextRetrieved,
            "context_cleared" => Self::ContextCleared,
            "context.compression_state" => Self::ContextCompressionState,
            "session_created" => Self::SessionCreated,
            "agent_session_created" => Self::AgentSessionCreated,
            "retrieval_started" => Self::RetrievalStarted,
            "memory_added" => Self::MemoryAdded,
            "memory_search_started" => Self::MemorySearchStarted,
            "memory_search_finished" => Self::MemorySearchFinished,
            "memory_updated" => Self::MemoryUpdated,
            "memory_deleted" => Self::MemoryDeleted,
            "task_created" => Self::TaskCreated,
            "task_running" => Self::TaskRunning,
            "task_completed" => Self::TaskCompleted,
            "task_failed" => Self::TaskFailed,
            "task_cancelled" => Self::TaskCancelled,
            "task_timeout" => Self::TaskTimeout,
            _ => return None,
        })
    }

    /// The subsystem that owns the event.
    #[must_use]
    pub fn domain(&self) -> EventDomain {
        match self {
            Self::AgentStarted
            | Self::AgentInvokeInput
            | Self::AgentInvokeOutput
            | Self::AgentStreamInput
            | Self::AgentStreamOutput => EventDomain::Agent,
            Self::AgentP2pReceived | Self::AgentPubsubReceived => EventDomain::Team,
            Self::WorkflowStarted
            | Self::WorkflowFinished
            | Self::WorkflowError
            | Self::WorkflowCancelled
            | Self::WorkflowNodeExecuted
            | Self::WorkflowNodeError
            | Self::WorkflowEdgeTraversed
            | Self::WorkflowLoopStarted
            | Self::WorkflowLoopFinished
            | Self::WorkflowInvokeInput
            | Self::WorkflowInvokeOutput
            | Self::WorkflowStreamInput
            | Self::WorkflowStreamOutput
            | Self::WorkflowComponentBatchInput
            | Self::WorkflowComponentBatchOutput
            | Self::WorkflowComponentStreamInput
            | Self::WorkflowComponentStreamOutput => EventDomain::Workflow,
            Self::LlmCallStarted
            | Self::LlmCallError
            | Self::LlmResponseReceived
            | Self::LlmResponseStarted
            | Self::LlmInvokeInput
            | Self::LlmInvokeOutput
            | Self::LlmStreamInput
            | Self::LlmStreamOutput
            | Self::LlmStreamCompleted
            | Self::LlmRoute
            | Self::LlmRouteAttempt
            | Self::LlmInput
            | Self::LlmOutput => EventDomain::Llm,
            Self::ToolCallStarted
            | Self::ToolCallFinished
            | Self::ToolCallError
            | Self::ToolResultReceived
            | Self::ToolParseStarted
            | Self::ToolParseFinished
            | Self::ToolInvokeInput
            | Self::ToolInvokeOutput
            | Self::ToolStreamInput
            | Self::ToolStreamOutput
            | Self::ToolAuth => EventDomain::Tool,
            Self::ContextUpdated
            | Self::ContextOffloaded
            | Self::ContextRetrieved
            | Self::ContextCleared
            | Self::ContextCompressionState => EventDomain::Context,
            Self::SessionCreated | Self::AgentSessionCreated => EventDomain::Session,
            Self::RetrievalStarted => EventDomain::Retrieval,
            Self::MemoryAdded
            | Self::MemorySearchStarted
            | Self::MemorySearchFinished
            | Self::MemoryUpdated
            | Self::MemoryDeleted => EventDomain::Memory,
            Self::TaskCreated
            | Self::TaskRunning
            | Self::TaskCompleted
            | Self::TaskFailed
            | Self::TaskCancelled
            | Self::TaskTimeout => EventDomain::Task,
            Self::Custom(_) => EventDomain::Custom,
        }
    }

    /// Whether the event belongs to the model or tool seam.
    ///
    /// This is the predicate a host uses to decide whether a payload carries
    /// messages/tool arguments rather than agent-level state.
    #[must_use]
    pub fn is_llm_or_tool(&self) -> bool {
        matches!(self.domain(), EventDomain::Llm | EventDomain::Tool)
    }
}

impl fmt::Display for CallbackEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Serialize for CallbackEvent {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for CallbackEvent {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let name = String::deserialize(deserializer)?;
        Ok(Self::from_name(&name).unwrap_or(Self::Custom(name)))
    }
}

/// The subsystem an event belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum EventDomain {
    /// Single-agent lifecycle.
    Agent,
    /// Multi-agent team and agent-to-agent traffic.
    Team,
    /// Workflow execution.
    Workflow,
    /// Model invocation.
    Llm,
    /// Tool invocation.
    Tool,
    /// Context-window management.
    Context,
    /// Session management.
    Session,
    /// Knowledge retrieval.
    Retrieval,
    /// Long-term memory.
    Memory,
    /// Task-manager lifecycle.
    Task,
    /// A host-defined event.
    Custom,
}

impl EventDomain {
    /// The lowercase domain name used in logs and metrics.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Agent => "agent",
            Self::Team => "team",
            Self::Workflow => "workflow",
            Self::Llm => "llm",
            Self::Tool => "tool",
            Self::Context => "context",
            Self::Session => "session",
            Self::Retrieval => "retrieval",
            Self::Memory => "memory",
            Self::Task => "task",
            Self::Custom => "custom",
        }
    }
}

impl fmt::Display for EventDomain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn all_covers_every_preset_variant() {
        // 5 agent + 2 team + 17 workflow + 13 llm + 11 tool + 5 context
        // + 2 session + 1 retrieval + 5 memory + 6 task.
        assert_eq!(CallbackEvent::ALL.len(), 67);
        assert!(!CallbackEvent::ALL
            .iter()
            .any(|event| matches!(event, CallbackEvent::Custom(_))));
    }

    #[test]
    fn every_preset_name_round_trips_with_and_without_scope() {
        for event in CallbackEvent::ALL {
            assert_eq!(
                CallbackEvent::from_name(event.as_str()),
                Some(event.clone())
            );
            assert_eq!(CallbackEvent::from_name(event.name()), Some(event.clone()));
            assert_eq!(
                event.as_str(),
                format!("{SCOPE_PREFIX}{}", event.name()),
                "{event:?}"
            );
        }
    }

    #[test]
    fn names_are_unique() {
        let mut names: Vec<&str> = CallbackEvent::ALL.iter().map(CallbackEvent::name).collect();
        names.sort_unstable();
        let count = names.len();
        names.dedup();
        assert_eq!(names.len(), count);
    }

    #[test]
    fn compression_state_keeps_its_dot_and_scope() {
        let event = CallbackEvent::ContextCompressionState;
        assert_eq!(event.name(), "context.compression_state");
        assert_eq!(event.as_str(), "_framework:context.compression_state");
        assert_eq!(
            CallbackEvent::from_name("context.compression_state"),
            Some(event.clone())
        );
    }

    #[test]
    fn unknown_names_are_not_presets() {
        assert_eq!(CallbackEvent::from_name("made_up"), None);
        assert_eq!(CallbackEvent::from_name("_framework:made_up"), None);
        assert_eq!(CallbackEvent::from_name("other_scope:agent_started"), None);
    }

    #[test]
    fn custom_events_use_the_name_verbatim() {
        let event = CallbackEvent::Custom("host:thing".to_owned());
        assert_eq!(event.name(), "host:thing");
        assert_eq!(event.as_str(), "host:thing");
        assert_eq!(event.domain(), EventDomain::Custom);
        assert!(!event.is_llm_or_tool());
        assert_eq!(event.to_string(), "host:thing");
    }

    #[test]
    fn domains_classify_every_preset() {
        assert_eq!(
            CallbackEvent::AgentStreamOutput.domain(),
            EventDomain::Agent
        );
        assert_eq!(CallbackEvent::AgentP2pReceived.domain(), EventDomain::Team);
        assert_eq!(
            CallbackEvent::WorkflowLoopFinished.domain(),
            EventDomain::Workflow
        );
        assert_eq!(CallbackEvent::LlmRoute.domain(), EventDomain::Llm);
        assert_eq!(CallbackEvent::ToolAuth.domain(), EventDomain::Tool);
        assert_eq!(CallbackEvent::ContextCleared.domain(), EventDomain::Context);
        assert_eq!(CallbackEvent::SessionCreated.domain(), EventDomain::Session);
        assert_eq!(
            CallbackEvent::RetrievalStarted.domain(),
            EventDomain::Retrieval
        );
        assert_eq!(CallbackEvent::MemoryDeleted.domain(), EventDomain::Memory);
        assert_eq!(CallbackEvent::TaskTimeout.domain(), EventDomain::Task);
        for event in CallbackEvent::ALL {
            assert_ne!(event.domain(), EventDomain::Custom, "{event:?}");
        }
        assert_eq!(EventDomain::Task.as_str(), "task");
    }

    #[test]
    fn llm_and_tool_predicate_follows_the_domain() {
        assert!(CallbackEvent::LlmInvokeInput.is_llm_or_tool());
        assert!(CallbackEvent::ToolInvokeOutput.is_llm_or_tool());
        assert!(!CallbackEvent::AgentInvokeInput.is_llm_or_tool());
        assert!(!CallbackEvent::WorkflowStarted.is_llm_or_tool());
        assert!(!CallbackEvent::Custom("llm_x".to_owned()).is_llm_or_tool());
    }

    #[test]
    fn serde_uses_the_wire_name_and_tolerates_unknown_names() {
        let json = serde_json::to_value(CallbackEvent::ToolInvokeInput).unwrap();
        assert_eq!(json, json!("_framework:tool_invoke_input"));
        let event: CallbackEvent = serde_json::from_value(json).unwrap();
        assert_eq!(event, CallbackEvent::ToolInvokeInput);

        let custom: CallbackEvent = serde_json::from_value(json!("host:event")).unwrap();
        assert_eq!(custom, CallbackEvent::Custom("host:event".to_owned()));
    }
}
