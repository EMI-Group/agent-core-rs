//! The team runtime: member registry, team-session binding, and bus dispatch.
//!
//! Python parity: `TeamRuntime` in
//! `openjiuwen/core/multi_agent/team_runtime/`. The runtime is the bus's
//! [`MessageHandler`]: it resolves a recipient from its own registry, builds the
//! child [`AgentSession`] the delivery runs in, invokes the member, and maps an
//! empty P2P reply to [`TeamError::EmptyResponse`].
//!
//! A runtime is owned by exactly one team and shared as an `Arc`. The bus holds
//! a [`Weak`] reference back to it, so the two never form an `Arc` cycle and a
//! dropped runtime makes in-flight deliveries fail with
//! `Error::session("team runtime dropped")`.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, OnceLock, Weak};
use std::time::Duration;

use openjiuwen_agent::{AgentCard, AgentOutput};
use openjiuwen_common::error::{Error, Result};
use openjiuwen_session::{AgentSession, AgentSessionOptions, AgentTeamSession, Session};
use parking_lot::RwLock;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::envelope::MessageEnvelope;
use crate::error::TeamError;
use crate::member::{AgentProvider, TeamAgent};
use crate::message_bus::{MessageBus, MessageBusConfig, MessageHandler};

/// Team id used when the configured one is absent or empty.
const DEFAULT_TEAM_ID: &str = "default";
/// Default deadline of one P2P dispatch, in seconds (30 minutes).
const DEFAULT_P2P_TIMEOUT: f64 = 1800.0;

/// Configuration of a [`TeamRuntime`].
///
/// Every field has a Python-compatible default (`"default"` / no bus override /
/// `1800.0` seconds), unknown JSON fields are tolerated, and an empty object
/// deserializes to the defaults.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct RuntimeConfig {
    /// Team id embedded in transport topic names.
    pub team_id: String,
    /// Bus configuration override; `None` keeps [`MessageBusConfig::default`].
    pub message_bus: Option<MessageBusConfig>,
    /// Deadline of one P2P dispatch, in seconds.
    pub p2p_timeout: f64,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            team_id: DEFAULT_TEAM_ID.to_owned(),
            message_bus: None,
            p2p_timeout: DEFAULT_P2P_TIMEOUT,
        }
    }
}

impl RuntimeConfig {
    /// Creates the configuration with the Python defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the team id embedded in transport topic names.
    #[must_use]
    pub fn with_team_id(mut self, team_id: impl Into<String>) -> Self {
        self.team_id = team_id.into();
        self
    }

    /// Sets the bus configuration override.
    #[must_use]
    pub fn with_message_bus(mut self, message_bus: MessageBusConfig) -> Self {
        self.message_bus = Some(message_bus);
        self
    }

    /// Sets the P2P dispatch deadline, in seconds.
    #[must_use]
    pub fn with_p2p_timeout(mut self, p2p_timeout: f64) -> Self {
        self.p2p_timeout = p2p_timeout;
        self
    }
}

/// One registered member: its card, its provider, and an optional pinned instance.
struct RegisteredAgent {
    card: AgentCard,
    provider: AgentProvider,
    pinned: Option<TeamAgent>,
}

/// Members in registration order plus the id-keyed entries.
#[derive(Default)]
struct AgentRegistry {
    order: Vec<String>,
    entries: HashMap<String, RegisteredAgent>,
}

/// The mutable state of a [`TeamRuntime`].
struct RuntimeState {
    agents: RwLock<AgentRegistry>,
    team_sessions: RwLock<BTreeMap<String, Arc<AgentTeamSession>>>,
    p2p_timeout: RwLock<Duration>,
    self_ref: OnceLock<Weak<TeamRuntime>>,
}

impl RuntimeState {
    fn new(config: &RuntimeConfig) -> Self {
        Self {
            agents: RwLock::new(AgentRegistry::default()),
            team_sessions: RwLock::new(BTreeMap::new()),
            p2p_timeout: RwLock::new(seconds_to_duration(config.p2p_timeout)),
            self_ref: OnceLock::new(),
        }
    }
}

/// Resolves members, binds sessions, and dispatches bus deliveries for one team.
///
/// The runtime owns the member registry (in registration order), the bound team
/// sessions, and the P2P deadline. It implements [`MessageHandler`] for its own
/// bus, so a tool-free caller only needs [`TeamRuntime::send`] and
/// [`TeamRuntime::publish`].
pub struct TeamRuntime {
    config: RuntimeConfig,
    bus: Arc<MessageBus>,
    state: Arc<RuntimeState>,
}

impl std::fmt::Debug for TeamRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TeamRuntime")
            .field("team_id", &self.config.team_id)
            .field("agents", &self.agent_ids())
            .finish_non_exhaustive()
    }
}

impl TeamRuntime {
    /// Builds a runtime and the [`MessageBus`] that dispatches into it.
    ///
    /// The bus holds a [`Weak`] back-reference, so the returned `Arc` is the
    /// only owner of the runtime. Construction uses [`Arc::new_cyclic`] so the
    /// weak reference exists before the value does.
    #[must_use]
    pub fn new(config: RuntimeConfig) -> Arc<Self> {
        let state = Arc::new(RuntimeState::new(&config));
        // Python parity: the runtime stamps its team id onto the bus config, so
        // topic names isolate teams even without an explicit bus override.
        let mut bus_config = config.message_bus.clone().unwrap_or_default();
        bus_config.team_id = Some(config.team_id.clone());
        Arc::new_cyclic(move |weak| {
            let _ = state.self_ref.set(weak.clone());
            let handler = Arc::new(RuntimeHandler {
                runtime: weak.clone(),
            });
            Self {
                config,
                bus: Arc::new(MessageBus::new(bus_config, handler)),
                state,
            }
        })
    }

    /// Returns the team id embedded in transport topic names.
    #[must_use]
    pub fn team_id(&self) -> &str {
        &self.config.team_id
    }

    /// Returns the configuration the runtime was built from.
    #[must_use]
    pub fn config(&self) -> &RuntimeConfig {
        &self.config
    }

    /// Returns `true` when `agent_id` is registered.
    #[must_use]
    pub fn has_agent(&self, agent_id: &str) -> bool {
        self.state.agents.read().entries.contains_key(agent_id)
    }

    /// Returns a copy of the registered card of `agent_id`.
    #[must_use]
    pub fn agent_card(&self, agent_id: &str) -> Option<AgentCard> {
        self.state
            .agents
            .read()
            .entries
            .get(agent_id)
            .map(|entry| entry.card.clone())
    }

    /// Returns the provider of `agent_id`.
    #[must_use]
    pub fn agent_provider(&self, agent_id: &str) -> Option<AgentProvider> {
        self.state
            .agents
            .read()
            .entries
            .get(agent_id)
            .map(|entry| Arc::clone(&entry.provider))
    }

    /// Returns the registered agent ids in registration order.
    #[must_use]
    pub fn agent_ids(&self) -> Vec<String> {
        self.state.agents.read().order.clone()
    }

    /// Returns the number of registered agents.
    #[must_use]
    pub fn agent_count(&self) -> usize {
        self.state.agents.read().order.len()
    }

    /// Registers `card` with its provider, replacing an existing registration.
    ///
    /// A replacement clears the pinned instance: the new provider is the source
    /// of truth again.
    pub fn register_agent(&self, card: AgentCard, provider: AgentProvider) {
        let mut registry = self.state.agents.write();
        let agent_id = card.id.clone();
        if !registry.entries.contains_key(&agent_id) {
            registry.order.push(agent_id.clone());
        }
        registry.entries.insert(
            agent_id,
            RegisteredAgent {
                card,
                provider,
                pinned: None,
            },
        );
    }

    /// Removes `agent_id` and all its subscriptions, returning its card.
    ///
    /// An unknown agent is a no-op returning `None`.
    #[must_use]
    pub fn unregister_agent(&self, agent_id: &str) -> Option<AgentCard> {
        let card = {
            let mut registry = self.state.agents.write();
            let entry = registry.entries.remove(agent_id)?;
            registry.order.retain(|id| id != agent_id);
            entry.card
        };
        for topic in self.bus.agent_topics(agent_id) {
            self.bus.remove_subscription(agent_id, &topic);
        }
        Some(card)
    }

    /// Pins `agent` so later resolutions reuse this instance instead of calling
    /// the provider again.
    ///
    /// An unknown `agent_id` gets a placeholder card named after it, so a
    /// strategy can pin an agent it wired before registering it.
    pub fn pin_agent(&self, agent_id: &str, agent: TeamAgent) {
        let mut registry = self.state.agents.write();
        if let Some(entry) = registry.entries.get_mut(agent_id) {
            entry.pinned = Some(agent);
            return;
        }
        let pinned = agent.clone();
        registry.order.push(agent_id.to_owned());
        registry.entries.insert(
            agent_id.to_owned(),
            RegisteredAgent {
                card: AgentCard::new(agent_id).with_id(agent_id),
                provider: Arc::new(move || pinned.clone()),
                pinned: Some(agent),
            },
        );
    }

    /// Binds `session` so deliveries carrying its id run in its child sessions.
    pub fn bind_team_session(&self, session: Arc<AgentTeamSession>) {
        self.state
            .team_sessions
            .write()
            .insert(session.id().to_owned(), session);
    }

    /// Unbinds the team session named `session_id`.
    pub fn unbind_team_session(&self, session_id: &str) {
        self.state.team_sessions.write().remove(session_id);
    }

    /// Returns the bound team session named `session_id`.
    #[must_use]
    pub fn team_session(&self, session_id: &str) -> Option<Arc<AgentTeamSession>> {
        self.state.team_sessions.read().get(session_id).cloned()
    }

    /// Returns the default P2P dispatch deadline.
    #[must_use]
    pub fn p2p_timeout(&self) -> Duration {
        *self.state.p2p_timeout.read()
    }

    /// Sets the default P2P dispatch deadline.
    pub fn set_p2p_timeout(&self, timeout: Duration) {
        *self.state.p2p_timeout.write() = timeout;
    }

    /// Returns the bus this runtime dispatches through.
    #[must_use]
    pub fn message_bus(&self) -> Arc<MessageBus> {
        Arc::clone(&self.bus)
    }

    /// Starts the bus. Idempotent.
    ///
    /// # Errors
    ///
    /// Reserved for future transports; the in-process bus never fails.
    pub async fn start(&self) -> Result<()> {
        self.bus.start().await
    }

    /// Stops the bus and joins every topic worker. Idempotent.
    ///
    /// # Errors
    ///
    /// Reserved for future transports; the in-process bus never fails.
    pub async fn stop(&self) -> Result<()> {
        self.bus.stop().await
    }

    /// Sends `message` point-to-point and waits for the reply.
    ///
    /// The deadline is `timeout` when supplied, else [`TeamRuntime::p2p_timeout`].
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] for an empty sender or recipient, or an
    /// unregistered recipient; [`Error::Timeout`] when the deadline elapses; and
    /// the member's error unchanged when it fails.
    pub async fn send(
        &self,
        message: Value,
        recipient: &str,
        sender: &str,
        session_id: Option<&str>,
        timeout: Option<Duration>,
    ) -> Result<Value> {
        if sender.is_empty() {
            return Err(Error::validation("sender is required for message tracing"));
        }
        if recipient.is_empty() {
            return Err(Error::validation("recipient is required"));
        }
        if !self.has_agent(recipient) {
            return Err(Error::validation(format!(
                "Recipient '{recipient}' not registered in runtime"
            )));
        }
        let mut envelope = MessageEnvelope::p2p(message, sender, recipient);
        if let Some(session_id) = session_id {
            envelope = envelope.with_session(session_id);
        }
        let deadline = Some(timeout.unwrap_or_else(|| self.p2p_timeout()));
        self.bus.send(envelope, deadline).await
    }

    /// Publishes `message` on `topic_id` and waits for the fan-out.
    ///
    /// The topic does not need a registered subscriber: an empty subscriber set
    /// is a no-op, matching Python.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] for an empty sender; [`Error::Transport`] when the
    /// topic closes before the fan-out completes. Subscriber failures are
    /// isolated by the bus.
    pub async fn publish(
        &self,
        message: Value,
        topic_id: &str,
        sender: &str,
        session_id: Option<&str>,
    ) -> Result<()> {
        if sender.is_empty() {
            return Err(Error::validation("sender is required for message tracing"));
        }
        let mut envelope = MessageEnvelope::pubsub(message, sender, topic_id);
        if let Some(session_id) = session_id {
            envelope = envelope.with_session(session_id);
        }
        self.bus.publish(envelope).await
    }

    /// Subscribes `agent_id` to `topic` (an exact topic or a `*`/`?` pattern).
    ///
    /// This only updates the logical subscriber registry; the transport topic is
    /// created lazily by [`publish`](Self::publish).
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] for an empty `agent_id` or `topic`.
    // `async` matches the Python API and leaves room for a distributed transport.
    #[allow(clippy::unused_async, clippy::unused_async_trait_impl)]
    pub async fn subscribe(&self, agent_id: &str, topic: &str) -> Result<()> {
        if agent_id.is_empty() {
            return Err(Error::validation("agent_id is required"));
        }
        if topic.is_empty() {
            return Err(Error::validation("topic is required"));
        }
        self.bus.add_subscription(agent_id, topic);
        Ok(())
    }

    /// Removes the `(agent_id, topic)` subscription; unknown pairs are a no-op.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] for an empty `agent_id` or `topic`.
    // `async` matches the Python API and leaves room for a distributed transport.
    #[allow(clippy::unused_async, clippy::unused_async_trait_impl)]
    pub async fn unsubscribe(&self, agent_id: &str, topic: &str) -> Result<()> {
        if agent_id.is_empty() {
            return Err(Error::validation("agent_id is required"));
        }
        if topic.is_empty() {
            return Err(Error::validation("topic is required"));
        }
        self.bus.remove_subscription(agent_id, topic);
        Ok(())
    }

    /// Tears down the transport topics of `session_id`.
    pub async fn cleanup_session(&self, session_id: &str) {
        self.bus.cleanup_session(session_id).await;
    }

    /// Resolves `agent_id`, binds it, and runs one invocation.
    ///
    /// A pinned instance wins over the provider; the bind hook runs on the
    /// resolved member before the invocation.
    ///
    /// # Errors
    ///
    /// [`Error::NotFound`] when `agent_id` is not registered, and the member's
    /// error unchanged when the invocation fails.
    pub async fn invoke_agent(
        &self,
        agent_id: &str,
        inputs: Value,
        session: Option<Arc<dyn Session>>,
    ) -> Result<AgentOutput> {
        let (pinned, provider) = {
            let registry = self.state.agents.read();
            let entry = registry.entries.get(agent_id).ok_or_else(|| {
                Error::not_found(format!("Agent '{agent_id}' not registered in runtime"))
            })?;
            (entry.pinned.clone(), Arc::clone(&entry.provider))
        };
        let member = pinned.unwrap_or_else(|| provider());
        if let Some(runtime) = self.state.self_ref.get().and_then(Weak::upgrade) {
            member.bind(&runtime, agent_id);
        }
        member.agent().invoke(inputs, session).await
    }

    /// Builds the child session a delivery to `agent_id` runs in.
    ///
    /// A bound team session becomes the parent (so the child shares its config,
    /// stream, and trace id); otherwise an `AgentSession` keyed by the envelope
    /// session id is used. Without a session id the member runs sessionless.
    fn child_session(&self, agent_id: &str, session_id: Option<&str>) -> Option<Arc<dyn Session>> {
        let session_id = session_id?;
        if let Some(team) = self.team_session(session_id) {
            let child = team.create_agent_session(agent_id);
            return Some(Arc::new(child));
        }
        Some(Arc::new(AgentSession::with_options(AgentSessionOptions {
            session_id: Some(session_id.to_owned()),
            agent_id: Some(agent_id.to_owned()),
            ..AgentSessionOptions::default()
        })))
    }

    /// Handles one point-to-point delivery from the bus.
    async fn dispatch_p2p(&self, envelope: &MessageEnvelope) -> Result<Value> {
        let recipient = envelope
            .recipient
            .clone()
            .ok_or_else(|| Error::validation("recipient is required"))?;
        if !self.has_agent(&recipient) {
            return Err(Error::not_found(format!(
                "Recipient '{recipient}' not registered in runtime"
            )));
        }
        let session = self.child_session(&recipient, envelope.session_id.as_deref());
        let output = self
            .invoke_agent(&recipient, envelope.message.clone(), session)
            .await?;
        let sender = envelope.sender.clone().unwrap_or_default();
        ensure_non_empty(output.to_value(), &sender, &recipient)
    }

    /// Handles one pub/sub delivery from the bus; the output is discarded.
    async fn dispatch_pubsub(&self, envelope: &MessageEnvelope, subscriber: &str) -> Result<()> {
        if !self.has_agent(subscriber) {
            return Err(Error::not_found(format!(
                "Recipient '{subscriber}' not registered in runtime"
            )));
        }
        let session = self.child_session(subscriber, envelope.session_id.as_deref());
        let output = self
            .invoke_agent(subscriber, envelope.message.clone(), session)
            .await?;
        tracing::debug!(
            subscriber,
            recipient = ?envelope.recipient,
            "team pub/sub delivery completed"
        );
        drop(output);
        Ok(())
    }
}

/// Rejects an empty P2P reply, naming both ends for tracing.
fn ensure_non_empty(value: Value, sender: &str, recipient: &str) -> Result<Value> {
    if value.is_null() {
        Err(Error::external(TeamError::empty_response(
            sender, recipient,
        )))
    } else {
        Ok(value)
    }
}

/// Converts a configured number of seconds to a duration.
///
/// Non-finite and non-positive values collapse to [`Duration::ZERO`] rather than
/// panicking, matching the bus's own conversion.
fn seconds_to_duration(seconds: f64) -> Duration {
    if seconds.is_finite() && seconds > 0.0 {
        Duration::from_secs_f64(seconds)
    } else {
        Duration::ZERO
    }
}

/// Bridges the bus back into the runtime through a [`Weak`] reference.
struct RuntimeHandler {
    runtime: Weak<TeamRuntime>,
}

#[async_trait::async_trait]
impl MessageHandler for RuntimeHandler {
    async fn handle_p2p(&self, envelope: &MessageEnvelope) -> Result<Value> {
        let runtime = self.runtime.upgrade().ok_or_else(runtime_dropped)?;
        runtime.dispatch_p2p(envelope).await
    }

    async fn handle_pubsub(&self, envelope: &MessageEnvelope, subscriber: &str) -> Result<()> {
        let runtime = self.runtime.upgrade().ok_or_else(runtime_dropped)?;
        runtime.dispatch_pubsub(envelope, subscriber).await
    }
}

/// Error reported when a delivery outlives its runtime.
fn runtime_dropped() -> Error {
    Error::session("team runtime dropped")
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;

    use super::*;

    #[test]
    fn config_defaults_match_python() {
        let config = RuntimeConfig::new();
        assert_eq!(config.team_id, "default");
        assert_eq!(config.message_bus, None);
        assert_eq!(config.p2p_timeout, 1800.0);
        assert_eq!(RuntimeConfig::default(), config);
    }

    #[test]
    fn config_json_is_defaulted_and_tolerant() {
        let config: RuntimeConfig =
            serde_json::from_value(json!({"team_id": "t", "surprise": true})).unwrap();
        assert_eq!(config.team_id, "t");
        assert_eq!(config.p2p_timeout, 1800.0);

        let empty: RuntimeConfig = serde_json::from_value(json!({})).unwrap();
        assert_eq!(empty, RuntimeConfig::new());
    }

    #[test]
    fn config_builders_set_fields() {
        let config = RuntimeConfig::new()
            .with_team_id("team-1")
            .with_message_bus(MessageBusConfig::new().with_team_id("team-1"))
            .with_p2p_timeout(2.5);
        assert_eq!(config.team_id, "team-1");
        assert_eq!(
            config.message_bus.as_ref().unwrap().team_id.as_deref(),
            Some("team-1")
        );
        assert_eq!(config.p2p_timeout, 2.5);
    }

    #[test]
    fn seconds_collapse_safely() {
        assert_eq!(seconds_to_duration(0.0), Duration::ZERO);
        assert_eq!(seconds_to_duration(-1.0), Duration::ZERO);
        assert_eq!(seconds_to_duration(f64::NAN), Duration::ZERO);
        assert_eq!(seconds_to_duration(f64::INFINITY), Duration::ZERO);
        assert_eq!(seconds_to_duration(0.25), Duration::from_millis(250));
    }

    #[test]
    fn ensure_non_empty_rejects_null() {
        let error = ensure_non_empty(Value::Null, "boss", "worker").unwrap_err();
        assert_eq!(error.kind(), ErrorKind::External);
        assert!(error.message().contains("boss"));
        assert!(error.message().contains("worker"));
    }

    #[test]
    fn ensure_non_empty_passes_values_through() {
        let value = json!({"answer": "hi"});
        assert_eq!(
            ensure_non_empty(value.clone(), "boss", "worker").unwrap(),
            value
        );
    }

    #[test]
    fn new_runtime_uses_the_configured_team_id() {
        let runtime = TeamRuntime::new(RuntimeConfig::new().with_team_id("alpha"));
        assert_eq!(runtime.team_id(), "alpha");
        assert_eq!(runtime.message_bus().team_id(), "alpha");
        assert_eq!(runtime.agent_count(), 0);
    }
}
