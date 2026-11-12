//! One invoke of the `ReAct` loop: session resolution, interruption
//! load/resume, the iteration loop, rails, and termination.
//!
//! The loop drives the whole turn from [`run`]:
//!
//! - load any persisted [`InterruptionState`](crate::interrupt::InterruptionState)
//!   (or seed a fresh turn) and prepare the message list with
//!   `ContextEngine::create_context`;
//! - for each iteration in `1..=limit`, rebuild the system prompt, prepare the
//!   window with `ContextEngine::prepare`, fire `BeforeInvoke`/`BeforeModelCall`,
//!   call the model through [`call_model`](crate::react::model_step::call_model),
//!   and fire `AfterModelCall`;
//! - execute the model's tool calls in model order through
//!   [`AbilityManager::execute`](crate::abilities::AbilityManager::execute),
//!   batched by `parallel_safe`, firing `BeforeToolCall`/`AfterToolCall` around
//!   each call and appending the resulting tool messages with
//!   `ContextEngine::observe`;
//! - fire `AfterReactIteration` after each iteration;
//! - honour rail control flags: `force_finish` returns immediately, `retry`
//!   repeats the model call once per request, and `skip_tool` substitutes the
//!   tool result without executing it;
//! - persist contexts with `ContextEngine::save_contexts` under the session
//!   state key `"context"` and interruption state under
//!   `__react_agent_interruption__`;
//! - terminate with `termination_reason` `max_iterations`, `token_budget`, or
//!   `cancelled`, and observe cancellation through the `CancellationToken`
//!   carried by the session/tool context.
use crate::input::AgentInput;
use crate::output::{AgentChunk, AgentOutput, Observation};
use crate::react::ReActAgent;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::BaseMessage;
use openjiuwen_session::Session;
use std::sync::Arc;

/// Mutable state accumulated across the iterations of one `run`.
#[allow(dead_code)] // wired into the loop alongside `run`
#[derive(Debug, Clone, Default)]
pub(crate) struct RunState {
    /// Current loop iteration, `1`-based.
    pub iteration: usize,
    /// Messages of the conversation so far.
    pub messages: Vec<BaseMessage>,
    /// Tool observations recorded so far.
    pub observations: Vec<Observation>,
    /// Whether the run paused on an interrupt.
    pub interrupted: bool,
}

/// Runs one invoke of the `ReAct` loop to completion.
///
/// `agent` supplies the model, tools, rails, and context engine; `input` is the
/// parsed caller payload (a text query or an interactive resume); `session` is
/// the resolved session the run reads state from and writes histories to; and
/// `publisher`, when present, receives each [`AgentChunk`] as it is produced.
///
/// # Errors
///
/// Returns an error when the invocation cannot run at all (unconfigured model,
/// cancelled session) or when a rail aborts the run; recoverable model and tool
/// failures are reported inside the returned [`AgentOutput`].
#[allow(dead_code)] // wired into ReActAgent::invoke by the loop implementation
#[allow(clippy::unused_async)] // the implementation awaits the model, tools, and session
pub(crate) async fn run(
    agent: &ReActAgent,
    input: AgentInput,
    session: Arc<dyn Session>,
    publisher: Option<tokio::sync::mpsc::UnboundedSender<AgentChunk>>,
) -> Result<AgentOutput> {
    // The loop implementation will consume these; the placeholder keeps the
    // signature stable while silencing unused-parameter warnings.
    let _ = (agent, input, session, publisher);
    Err(Error::unsupported("react loop not implemented"))
}
