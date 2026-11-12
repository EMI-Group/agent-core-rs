//! Agent-as-tool: member agents exposed to a model as ordinary [`Tool`]s.
//!
//! Python parity: `P2PAbilityManager` in `openjiuwen/core/multi_agent/`, which
//! wraps a member `AgentCard` into an [`AgentTool`] the supervisor's model can
//! call. The Rust version is a plain [`Tool`] over a [`ToolCard`]: the card is
//! the model-facing projection of the member, and one invocation dispatches the
//! member through the team runtime in a child session, so the sub-agent run
//! stays traceable and isolated from the caller's session.
//!
//! The tool holds a [`Weak`] reference back to the runtime: an agent tool that
//! outlives its team fails the call instead of keeping the team alive.

use std::sync::{Arc, Weak};

use async_trait::async_trait;
use openjiuwen_agent::AgentCard;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::tool::{Tool, ToolCard, ToolContext};
use openjiuwen_session::{AgentSession, AgentSessionOptions, Session};
use serde_json::{json, Value};
use tokio::sync::Semaphore;

use crate::runtime::TeamRuntime;

/// Metadata key carrying the model's tool-call id for this invocation.
const TOOL_CALL_ID_KEY: &str = "tool_call_id";

/// A member agent exposed to a model as a [`Tool`].
///
/// The card mirrors the member's [`AgentCard`]; [`invoke`](Tool::invoke) sends
/// the model's arguments to the member through the team runtime and returns the
/// member's output value.
pub struct AgentTool {
    card: ToolCard,
    agent_id: String,
    runtime: Weak<TeamRuntime>,
    limiter: Option<Arc<Semaphore>>,
}

impl std::fmt::Debug for AgentTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentTool")
            .field("agent_id", &self.agent_id)
            .field("name", &self.card.name)
            .field("has_limiter", &self.limiter.is_some())
            .finish_non_exhaustive()
    }
}

impl AgentTool {
    /// Builds the tool for the member described by `card`.
    ///
    /// `runtime` is the team runtime that dispatches the dispatch; it is held
    /// weakly so the tool never extends the team's lifetime.
    #[must_use]
    pub fn new(card: &AgentCard, runtime: Weak<TeamRuntime>) -> Self {
        let input_params = card
            .input_params
            .clone()
            .unwrap_or_else(|| json!({"type": "object", "properties": {}}));
        let tool_card = ToolCard::new(card.name.clone(), card.description.clone(), input_params)
            .with_id(card.name.clone())
            .with_parallel_safe(true);
        Self {
            card: tool_card,
            agent_id: card.id.clone(),
            runtime,
            limiter: None,
        }
    }

    /// Caps how many sub-agent dispatches run concurrently.
    ///
    /// A supervisor shares one semaphore across all its agent tools, which is
    /// how the Python `P2PAbilityManager` bounds parallel sub-agents.
    #[must_use]
    pub fn with_limiter(mut self, limiter: Arc<Semaphore>) -> Self {
        self.limiter = Some(limiter);
        self
    }

    /// Returns the model-facing card of the tool.
    #[must_use]
    pub fn card(&self) -> &ToolCard {
        &self.card
    }

    /// Returns the id of the member this tool dispatches to.
    #[must_use]
    pub fn agent_id(&self) -> &str {
        &self.agent_id
    }

    /// Acquires a concurrency permit when the tool is limited.
    ///
    /// The permit is returned to the caller so it stays held for the whole
    /// invocation; `None` means the tool runs unbounded.
    async fn acquire_permit(&self) -> Result<Option<tokio::sync::SemaphorePermit<'_>>> {
        let Some(limiter) = &self.limiter else {
            return Ok(None);
        };
        limiter.acquire().await.map(Some).map_err(|_| {
            Error::tool(format!(
                "Agent execution error: concurrency limit closed for '{}'",
                self.agent_id
            ))
        })
    }

    /// Builds the child session one dispatch runs in.
    ///
    /// The id is `"{parent}:{tool_call_id}"` when the caller is session-scoped,
    /// else the tool-call id alone, matching Python's `conversation_id` scheme.
    fn child_session(&self, parent_id: Option<&str>, tool_call_id: &str) -> Arc<dyn Session> {
        let session_id = match parent_id {
            Some(parent) if !parent.is_empty() => format!("{parent}:{tool_call_id}"),
            _ => tool_call_id.to_owned(),
        };
        Arc::new(AgentSession::with_options(AgentSessionOptions {
            session_id: Some(session_id),
            agent_id: Some(self.agent_id.clone()),
            ..AgentSessionOptions::default()
        }))
    }

    /// Runs the member once, honouring the call deadline.
    async fn dispatch(
        &self,
        inputs: Value,
        tool_call_id: &str,
        ctx: &ToolContext,
    ) -> Result<Value> {
        let runtime = self
            .runtime
            .upgrade()
            .ok_or_else(|| Error::tool("Agent execution error: team runtime dropped".to_owned()))?;
        let session = self.child_session(ctx.session_id.as_deref(), tool_call_id);
        // The permit is held until the invocation finishes, so the cap counts
        // in-flight sub-agents rather than queued tool calls.
        let _permit = self.acquire_permit().await?;
        let pending = runtime.invoke_agent(&self.agent_id, inputs, Some(session));
        let output = match ctx.timeout {
            Some(timeout) => match tokio::time::timeout(timeout, pending).await {
                Ok(output) => output,
                Err(_) => {
                    return Err(Error::timeout(format!(
                        "agent tool '{}' timed out after {:?}",
                        self.agent_id, timeout
                    )))
                }
            },
            None => pending.await,
        };
        match output {
            Ok(output) => Ok(output.to_value()),
            Err(error) => Err(Error::tool(format!("Agent execution error: {error}"))),
        }
    }
}

#[async_trait]
impl Tool for AgentTool {
    fn card(&self) -> &ToolCard {
        &self.card
    }

    async fn invoke(&self, inputs: Value, ctx: &ToolContext) -> Result<Value> {
        ctx.check_cancelled()?;
        let tool_call_id = ctx
            .metadata
            .get(TOOL_CALL_ID_KEY)
            .and_then(Value::as_str)
            .unwrap_or(ctx.invocation_id.as_str());
        self.dispatch(inputs, tool_call_id, ctx).await
    }
}
