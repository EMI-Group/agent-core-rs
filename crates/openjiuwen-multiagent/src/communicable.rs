//! The communication mixin: one agent id bound to a team runtime.
//!
//! Python parity: `CommunicableAgent` in
//! `openjiuwen/core/multi_agent/team_runtime/communicable_agent.py`. Python
//! mixes the messaging methods into an agent class; Rust cannot express that
//! mixin, so this type owns the binding and exposes the same calls. An agent
//! keeps one instance and is bound explicitly through
//! [`CommunicableAgent::bind_runtime`] (in Python the runtime binds it during
//! `register_agent`).
//!
//! The binding sits behind a `parking_lot::RwLock`, and the guard is released
//! before any `.await`, so a bound agent can be shared across tasks.

use std::fmt;
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};
use parking_lot::RwLock;
use serde_json::Value;

use crate::runtime::TeamRuntime;

/// The identity and optional runtime binding of one [`CommunicableAgent`].
#[derive(Debug)]
struct Binding {
    /// The id messages are sent as and subscribed under.
    agent_id: Option<String>,
    /// The runtime the calls route through; `None` while unbound.
    runtime: Option<Arc<TeamRuntime>>,
}

/// One agent id bound to a [`TeamRuntime`], with P2P and pub/sub helpers.
///
/// Python parity: the `CommunicableAgent` mixin. Calling [`send`](Self::send),
/// [`publish`](Self::publish), [`subscribe`](Self::subscribe) or
/// [`unsubscribe`](Self::unsubscribe) before [`bind_runtime`](Self::bind_runtime)
/// fails with a validation error naming the agent id, matching Python's
/// "Agent not bound to a `TeamRuntime`" status error.
pub struct CommunicableAgent {
    binding: RwLock<Binding>,
}

impl fmt::Debug for CommunicableAgent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let binding = self.binding.read();
        f.debug_struct("CommunicableAgent")
            .field("agent_id", &binding.agent_id)
            .field("bound", &binding.runtime.is_some())
            .finish_non_exhaustive()
    }
}

impl CommunicableAgent {
    /// Creates an unbound agent whose identity is `agent_id`.
    ///
    /// The identity is known from construction (it is what
    /// [`agent_id`](Self::agent_id) reports), but no runtime is attached until
    /// [`bind_runtime`](Self::bind_runtime) is called.
    #[must_use]
    pub fn new(agent_id: impl Into<String>) -> Self {
        Self {
            binding: RwLock::new(Binding {
                agent_id: Some(agent_id.into()),
                runtime: None,
            }),
        }
    }

    /// Binds `runtime` and `agent_id` to this agent.
    ///
    /// Rebinding to the same `(runtime, agent_id)` pair is a silent no-op.
    /// Rebinding to a different pair logs a warning and proceeds, matching
    /// Python's "already bound ... rebinding may cause unexpected behavior"
    /// warning.
    ///
    /// Identity is compared by `Arc` pointer, so two equal-looking runtimes are
    /// different bindings.
    pub fn bind_runtime(&self, runtime: Arc<TeamRuntime>, agent_id: impl Into<String>) {
        let agent_id = agent_id.into();
        let mut binding = self.binding.write();
        if let Some(bound) = binding.runtime.as_ref() {
            if Arc::ptr_eq(bound, &runtime)
                && binding.agent_id.as_deref() == Some(agent_id.as_str())
            {
                return;
            }
            tracing::warn!(
                previous_agent_id = binding.agent_id.as_deref().unwrap_or_default(),
                agent_id = %agent_id,
                "agent is already bound to a team runtime; rebinding may cause unexpected behavior"
            );
        }
        binding.agent_id = Some(agent_id);
        binding.runtime = Some(runtime);
    }

    /// Returns `true` once a runtime is bound (and until [`unbind`](Self::unbind)).
    #[must_use]
    pub fn is_bound(&self) -> bool {
        let binding = self.binding.read();
        binding.runtime.is_some() && binding.agent_id.is_some()
    }

    /// Returns the bound runtime, or `None` while unbound.
    #[must_use]
    pub fn runtime(&self) -> Option<Arc<TeamRuntime>> {
        self.binding.read().runtime.clone()
    }

    /// Returns this agent's id, or `None` when it has no identity.
    ///
    /// The id is known from construction, so it stays available after
    /// [`unbind`](Self::unbind).
    #[must_use]
    pub fn agent_id(&self) -> Option<String> {
        self.binding.read().agent_id.clone()
    }

    /// Drops the runtime binding, keeping the agent identity.
    ///
    /// Every later call fails with the unbound validation error.
    pub fn unbind(&self) {
        self.binding.write().runtime = None;
    }

    /// Sends `message` point-to-point from this agent and waits for the reply.
    ///
    /// Python parity: `CommunicableAgent.send`, which forwards to
    /// `TeamRuntime.send` with this agent as the sender and no explicit
    /// deadline.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] when unbound, plus every error the runtime's `send`
    /// returns (an unknown recipient, an elapsed deadline, a member failure).
    pub async fn send(
        &self,
        message: Value,
        recipient: &str,
        session_id: Option<&str>,
    ) -> Result<Value> {
        let (runtime, agent_id) = self.bound()?;
        runtime
            .send(message, recipient, &agent_id, session_id, None)
            .await
    }

    /// Publishes `message` on `topic_id` from this agent and waits for fan-out.
    ///
    /// Python parity: `CommunicableAgent.publish`, which forwards to
    /// `TeamRuntime.publish` with this agent as the sender.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] when unbound, plus every error the runtime's
    /// `publish` returns.
    pub async fn publish(
        &self,
        message: Value,
        topic_id: &str,
        session_id: Option<&str>,
    ) -> Result<()> {
        let (runtime, agent_id) = self.bound()?;
        runtime
            .publish(message, topic_id, &agent_id, session_id)
            .await
    }

    /// Subscribes this agent to `topic` (an exact topic or a `*`/`?` pattern).
    ///
    /// Python parity: `CommunicableAgent.subscribe`, which forwards to
    /// `TeamRuntime.subscribe` with this agent's id.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] when unbound or `topic` is empty.
    pub async fn subscribe(&self, topic: &str) -> Result<()> {
        let (runtime, agent_id) = self.bound()?;
        runtime.subscribe(&agent_id, topic).await
    }

    /// Removes this agent's `topic` subscription; unknown pairs are a no-op.
    ///
    /// Python parity: `CommunicableAgent.unsubscribe`, which forwards to
    /// `TeamRuntime.unsubscribe` with this agent's id.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] when unbound or `topic` is empty.
    pub async fn unsubscribe(&self, topic: &str) -> Result<()> {
        let (runtime, agent_id) = self.bound()?;
        runtime.unsubscribe(&agent_id, topic).await
    }

    /// Returns the bound runtime and id, or the Python-parity validation error.
    ///
    /// The read guard is dropped before the returned pair is used, so callers
    /// never hold the lock across an `.await`.
    fn bound(&self) -> Result<(Arc<TeamRuntime>, String)> {
        let binding = self.binding.read();
        if let (Some(runtime), Some(agent_id)) =
            (binding.runtime.as_ref(), binding.agent_id.as_ref())
        {
            return Ok((Arc::clone(runtime), agent_id.clone()));
        }
        let agent_id = binding.agent_id.as_deref().unwrap_or_default();
        Err(Error::validation(format!(
            "agent '{agent_id}' is not bound to a team runtime"
        )))
    }
}
