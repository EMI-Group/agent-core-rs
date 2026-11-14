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
//!
//! # Team events
//!
//! Each runtime owns one [`TeamEventBus`] ([`TeamRuntime::events`]) and emits
//! synchronously, after the corresponding state change succeeded:
//!
//! - [`TeamEvent::member_joined`] for every [`TeamRuntime::register_agent`]
//!   (with a `replaced` detail when the id was already registered) and for a
//!   [`TeamRuntime::pin_agent`] that creates an entry;
//! - [`TeamEvent::member_left`] when [`TeamRuntime::unregister_agent`] removes a
//!   member (an unknown id emits nothing);
//! - [`TeamEvent::message_sent`] after a successful [`TeamRuntime::send`] or
//!   [`TeamRuntime::publish`], carrying the envelope's message id and the topic
//!   or session id when known;
//! - [`TeamEvent::message_received`] (the pub/sub form carries the topic id)
//!   before an inbound member is invoked, matching Python's
//!   `AGENT_P2P_RECEIVED`/`AGENT_PUBSUB_RECEIVED` triggers that run before
//!   `Runner.run_agent`;
//! - [`TeamEvent::error`] when an outbound or inbound dispatch fails, with an
//!   `operation` detail of `send`, `publish`, `p2p_dispatch`, or
//!   `pubsub_dispatch`.
//!
//! # Resource manager and transport seams
//!
//! A runtime can consult a [`TeamResourceMgr`] (typically owned by a runner) for
//! members the team never registered locally, and it can route outbound
//! deliveries through a [`TeamTransport`] instead of the built-in in-process
//! bus. The local member registry always wins over the resource manager.
//! Inbound dispatch stays bus-driven for the in-process transport; a backend
//! such as the Redis transport supplies its own inbound workers.

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, Ordering};
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
use crate::events::{TeamEvent, TeamEventBus};
use crate::inbox::InboxFilterRegistry;
use crate::member::{AgentProvider, TeamAgent};
use crate::member_checkpoint::MemberCheckpointer;
use crate::message_bus::{MessageBus, MessageBusConfig, MessageHandler};
use crate::resources::TeamResourceMgr;
use crate::routing::GroupRegistry;
use crate::transport::{InProcessTransport, TeamTransport};

mod routing;
mod seams;

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
    events: Arc<TeamEventBus>,
    transport: RwLock<Arc<dyn TeamTransport>>,
    resource_mgr: RwLock<Option<Arc<dyn TeamResourceMgr>>>,
    member_checkpointer: RwLock<Option<Arc<MemberCheckpointer>>>,
    /// Named agent groups used by [`TeamRuntime::broadcast_to_group`].
    groups: Arc<GroupRegistry>,
    /// Per-agent filters consulted before an inbound member is invoked.
    inbox_filters: Arc<InboxFilterRegistry>,
    /// Whether [`TeamRuntime::start`] succeeded and [`TeamRuntime::stop`] has
    /// not run since; Python parity for `TeamRuntime._running`.
    running: AtomicBool,
}

impl std::fmt::Debug for TeamRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TeamRuntime")
            .field("team_id", &self.config.team_id)
            .field("agents", &self.agent_ids())
            .field("running", &self.running.load(Ordering::SeqCst))
            .field("has_resource_mgr", &self.resource_mgr.read().is_some())
            .field(
                "has_member_checkpointer",
                &self.member_checkpointer.read().is_some(),
            )
            .finish_non_exhaustive()
    }
}

impl TeamRuntime {
    /// Builds a runtime and the [`MessageBus`] that dispatches into it.
    ///
    /// The bus holds a [`Weak`] back-reference, so the returned `Arc` is the
    /// only owner of the runtime. Construction uses [`Arc::new_cyclic`] so the
    /// weak reference exists before the value does.
    ///
    /// The runtime starts with its own [`TeamEventBus`], an in-process transport
    /// over the bus it just built, and no resource manager or member
    /// checkpointer.
    #[must_use]
    pub fn new(config: RuntimeConfig) -> Arc<Self> {
        Self::build(config, None)
    }

    /// Builds a runtime that also consults `resource_mgr` for members, tools,
    /// models, and team sessions it does not own.
    ///
    /// Registration forwards to the manager as well, matching Python's
    /// `Runner.resource_mgr.add_agent` call inside `register_agent`.
    #[must_use]
    pub fn new_with_resource_mgr(
        config: RuntimeConfig,
        resource_mgr: Arc<dyn TeamResourceMgr>,
    ) -> Arc<Self> {
        Self::build(config, Some(resource_mgr))
    }

    /// Builds the runtime and its bus; `resource_mgr` is optional.
    fn build(config: RuntimeConfig, resource_mgr: Option<Arc<dyn TeamResourceMgr>>) -> Arc<Self> {
        let state = Arc::new(RuntimeState::new(&config));
        let events = TeamEventBus::new();
        // Python parity: the runtime stamps its team id onto the bus config, so
        // topic names isolate teams even without an explicit bus override.
        let mut bus_config = config.message_bus.clone().unwrap_or_default();
        bus_config.team_id = Some(config.team_id.clone());
        Arc::new_cyclic(move |weak| {
            let _ = state.self_ref.set(weak.clone());
            let handler = Arc::new(RuntimeHandler {
                runtime: weak.clone(),
            });
            let bus = Arc::new(MessageBus::new(bus_config, handler));
            let transport: Arc<dyn TeamTransport> = InProcessTransport::from_bus(Arc::clone(&bus));
            Self {
                config,
                bus,
                state,
                events,
                transport: RwLock::new(transport),
                resource_mgr: RwLock::new(resource_mgr),
                member_checkpointer: RwLock::new(None),
                groups: Arc::new(GroupRegistry::new()),
                inbox_filters: Arc::new(InboxFilterRegistry::new()),
                running: AtomicBool::new(false),
            }
        })
    }

    /// Returns the team event bus this runtime emits on.
    ///
    /// The bus is process-local and shared as an `Arc`; it is not part of
    /// [`RuntimeConfig`] because it is neither serializable nor shareable across
    /// processes.
    #[must_use]
    pub fn events(&self) -> Arc<TeamEventBus> {
        Arc::clone(&self.events)
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
    /// of truth again. Registration is also forwarded to the configured
    /// [`TeamResourceMgr`], where a duplicate is logged at `debug` and the
    /// existing entry is reused (Python parity), so a `false` result never
    /// fails the local registration.
    ///
    /// Emits [`TeamEvent::member_joined`], with a `replaced` detail when the
    /// agent id was already registered.
    pub fn register_agent(&self, card: AgentCard, provider: AgentProvider) {
        let agent_id = card.id.clone();
        let replaced = {
            let mut registry = self.state.agents.write();
            let replaced = registry.entries.contains_key(&agent_id);
            if !replaced {
                registry.order.push(agent_id.clone());
            }
            registry.entries.insert(
                agent_id.clone(),
                RegisteredAgent {
                    card: card.clone(),
                    provider: Arc::clone(&provider),
                    pinned: None,
                },
            );
            replaced
        };
        if let Some(mgr) = self.resource_mgr() {
            if !mgr.register_agent(card, provider) {
                tracing::debug!(agent_id, "agent already in the resource manager; reusing");
            }
        }
        let mut event = TeamEvent::member_joined(self.team_id(), agent_id);
        if replaced {
            event = event.with_detail("replaced", Value::Bool(true));
        }
        self.events.emit(event);
    }

    /// Removes `agent_id` and all its subscriptions, returning its card.
    ///
    /// An unknown agent is a no-op returning `None`. The resource manager is
    /// deliberately left untouched, because an agent may be shared by several
    /// teams (Python parity).
    ///
    /// Emits [`TeamEvent::member_left`] only when a member was removed.
    #[must_use]
    pub fn unregister_agent(&self, agent_id: &str) -> Option<AgentCard> {
        let card = {
            let mut registry = self.state.agents.write();
            let entry = registry.entries.remove(agent_id)?;
            registry.order.retain(|id| id != agent_id);
            entry.card
        };
        self.bus.remove_all_subscriptions(agent_id);
        self.events
            .emit(TeamEvent::member_left(self.team_id(), agent_id));
        Some(card)
    }

    /// Pins `agent` so later resolutions reuse this instance instead of calling
    /// the provider again.
    ///
    /// An unknown `agent_id` gets a placeholder card named after it, so a
    /// strategy can pin an agent it wired before registering it; that path
    /// emits [`TeamEvent::member_joined`] because it creates a registry entry.
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
        drop(registry);
        self.events
            .emit(TeamEvent::member_joined(self.team_id(), agent_id));
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
    ///
    /// Outbound deliveries route through [`TeamRuntime::transport`], which
    /// wraps this bus by default. Subscriptions and session cleanup stay on the
    /// bus, because the in-process transport is bus-driven in both directions.
    #[must_use]
    pub fn message_bus(&self) -> Arc<MessageBus> {
        Arc::clone(&self.bus)
    }

    /// Sends `message` point-to-point and waits for the reply.
    ///
    /// The deadline is `timeout` when supplied, else [`TeamRuntime::p2p_timeout`].
    /// The delivery routes through [`TeamRuntime::transport`].
    ///
    /// Emits [`TeamEvent::message_sent`] with the envelope's message id on
    /// success, and [`TeamEvent::error`] (detail `operation = "send"`) when the
    /// transport fails; the error is returned unchanged.
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
        let message_id = envelope.message_id.clone();
        let deadline = Some(timeout.unwrap_or_else(|| self.p2p_timeout()));
        let result = self.transport().send(envelope, deadline).await;
        match result {
            Ok(value) => {
                self.emit_message_sent(sender, recipient, &message_id, session_id, None);
                Ok(value)
            }
            Err(error) => {
                self.emit_dispatch_error(recipient, &error, "send");
                Err(error)
            }
        }
    }

    /// Publishes `message` on `topic_id` and waits for the fan-out.
    ///
    /// The topic does not need a registered subscriber: an empty subscriber set
    /// is a no-op, matching Python. The delivery routes through
    /// [`TeamRuntime::transport`].
    ///
    /// Emits [`TeamEvent::message_sent`] with the envelope's message id and the
    /// topic on success, and [`TeamEvent::error`] (detail
    /// `operation = "publish"`) when the transport fails; the error is returned
    /// unchanged.
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
        let message_id = envelope.message_id.clone();
        let result = self.transport().publish(envelope).await;
        match result {
            Ok(()) => {
                self.emit_message_sent(sender, topic_id, &message_id, session_id, Some(topic_id));
                Ok(())
            }
            Err(error) => {
                self.emit_dispatch_error(topic_id, &error, "publish");
                Err(error)
            }
        }
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

    /// Returns `true` between a successful [`TeamRuntime::start`] and
    /// [`TeamRuntime::stop`].
    ///
    /// Python parity: `TeamRuntime.is_running`, the runtime's `_running` flag.
    /// Both lifecycle methods delegate to the transport, so this reports the
    /// started state the runtime observed; a freshly built runtime is stopped.
    #[must_use]
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Returns the transport's whole subscription registry as
    /// `pattern -> sorted agent ids`.
    ///
    /// Python parity: `TeamRuntime.list_subscriptions()` without an agent
    /// filter, which forwards to the message bus.
    #[must_use]
    pub fn list_subscriptions(&self) -> BTreeMap<String, Vec<String>> {
        self.transport().list_subscriptions()
    }

    /// Returns the sorted patterns `agent_id` is subscribed to.
    ///
    /// Python parity: the `topics` list of
    /// `TeamRuntime.list_subscriptions(agent_id)`.
    #[must_use]
    pub fn list_agent_subscriptions(&self, agent_id: &str) -> Vec<String> {
        self.transport().agent_subscriptions(agent_id)
    }

    /// Returns the total number of active topic subscriptions.
    ///
    /// Python parity: `TeamRuntime.get_subscription_count`, which forwards to
    /// the message bus.
    #[must_use]
    pub fn get_subscription_count(&self) -> usize {
        self.transport().subscription_count()
    }

    /// Resolves `agent_id`, binds it, and runs one invocation.
    ///
    /// A pinned instance wins over the provider; the bind hook runs on the
    /// resolved member before the invocation. A member the local registry does
    /// not hold is resolved through [`TeamRuntime::resolve_agent_provider`], so
    /// a runner-backed resource manager can supply members.
    ///
    /// With a [`MemberCheckpointer`] installed ([`TeamRuntime::set_member_checkpointer`])
    /// and a session, the member's own state is restored before the invocation
    /// and saved after a successful one — the Rust counterpart of Python's
    /// `pre_agent_execute` restore and `post_agent_execute` save over the
    /// `AgentStorage` keyed by the child session's agent id. A failed member is
    /// not saved.
    ///
    /// # Errors
    ///
    /// [`Error::NotFound`] when neither the local registry nor the resource
    /// manager knows `agent_id`; a checkpoint restore or save failure is
    /// propagated unchanged (Python's `pre_agent_execute` raises); a member
    /// error is propagated unchanged and never saved.
    pub async fn invoke_agent(
        &self,
        agent_id: &str,
        inputs: Value,
        session: Option<Arc<dyn Session>>,
    ) -> Result<AgentOutput> {
        let (pinned, provider) = self.resolve_member(agent_id)?;
        let member = pinned.unwrap_or_else(|| provider());
        if let Some(runtime) = self.state.self_ref.get().and_then(Weak::upgrade) {
            member.bind(&runtime, agent_id);
        }
        let checkpointer = self.member_checkpointer();
        if let (Some(checkpointer), Some(member_session)) =
            (checkpointer.as_ref(), session.as_deref())
        {
            let restored = checkpointer.restore(member_session).await?;
            tracing::debug!(agent_id, restored, "restored team member state");
        }
        let output = member.agent().invoke(inputs, session.clone()).await;
        match output {
            Ok(output) => {
                if let (Some(checkpointer), Some(member_session)) =
                    (checkpointer.as_ref(), session.as_deref())
                {
                    if let Some(checkpoint_id) = checkpointer.save(member_session).await? {
                        tracing::debug!(
                            agent_id,
                            checkpoint_id = %checkpoint_id,
                            "saved team member state"
                        );
                    }
                }
                Ok(output)
            }
            // A failed member is not checkpointed: its state may be partial.
            Err(error) => Err(error),
        }
    }

    /// Resolves the member pinned instance and provider for `agent_id`.
    ///
    /// The local registry wins; otherwise the provider (and, for tracing, the
    /// card) comes from the resource manager. Returns [`Error::NotFound`] with
    /// the Python message when neither resolves.
    fn resolve_member(&self, agent_id: &str) -> Result<(Option<TeamAgent>, AgentProvider)> {
        let local = {
            let registry = self.state.agents.read();
            registry
                .entries
                .get(agent_id)
                .map(|entry| (entry.pinned.clone(), Arc::clone(&entry.provider)))
        };
        if let Some(member) = local {
            return Ok(member);
        }
        let provider = self.resolve_agent_provider(agent_id).ok_or_else(|| {
            Error::not_found(format!("Agent '{agent_id}' not registered in runtime"))
        })?;
        match self.resolve_agent_card(agent_id) {
            Some(card) => {
                tracing::debug!(
                    agent_id,
                    card_id = %card.id,
                    "resolved team member from the resource manager"
                );
            }
            None => {
                tracing::debug!(
                    agent_id,
                    "resource manager supplied a member provider without a card"
                );
            }
        }
        Ok((None, provider))
    }

    /// Emits one outbound [`TeamEvent::message_sent`].
    fn emit_message_sent(
        &self,
        sender: &str,
        recipient: &str,
        message_id: &str,
        session_id: Option<&str>,
        topic_id: Option<&str>,
    ) {
        let mut event = TeamEvent::message_sent(self.team_id(), sender, recipient, message_id);
        if let Some(topic_id) = topic_id {
            event = event.with_topic(topic_id);
        }
        if let Some(session_id) = session_id {
            event = event.with_session(session_id);
        }
        self.events.emit(event);
    }

    /// Emits one [`TeamEvent::error`] for a failed dispatch.
    fn emit_dispatch_error(&self, subject: impl Into<String>, error: &Error, operation: &str) {
        let event = TeamEvent::error(self.team_id(), subject, error.to_string())
            .with_detail("operation", Value::String(operation.to_owned()));
        self.events.emit(event);
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
    ///
    /// Emits [`TeamEvent::message_received`] before the member is invoked,
    /// mirroring Python's `AGENT_P2P_RECEIVED` trigger that runs before
    /// `Runner.run_agent`.
    async fn dispatch_p2p(&self, envelope: &MessageEnvelope) -> Result<Value> {
        let recipient = envelope
            .recipient
            .clone()
            .ok_or_else(|| Error::validation("recipient is required"))?;
        let sender = envelope.sender.clone().unwrap_or_default();
        if !self.inbox_filters.accepts(&recipient, envelope) {
            return Err(Error::validation(format!(
                "inbox of '{recipient}' rejected the message from '{sender}'"
            )));
        }
        let mut received =
            TeamEvent::message_received(self.team_id(), &sender, &recipient, &envelope.message_id);
        if let Some(session_id) = &envelope.session_id {
            received = received.with_session(session_id.clone());
        }
        self.events.emit(received);
        if !self.has_agent(&recipient) {
            return Err(Error::not_found(format!(
                "Recipient '{recipient}' not registered in runtime"
            )));
        }
        let session = self.child_session(&recipient, envelope.session_id.as_deref());
        let output = self
            .invoke_agent(&recipient, envelope.message.clone(), session)
            .await?;
        ensure_non_empty(output.to_value(), &sender, &recipient)
    }

    /// Handles one pub/sub delivery from the bus; the output is discarded.
    ///
    /// Emits [`TeamEvent::pubsub_received`] before the subscriber is invoked,
    /// mirroring Python's `AGENT_PUBSUB_RECEIVED` trigger that runs before
    /// `Runner.run_agent`.
    async fn dispatch_pubsub(&self, envelope: &MessageEnvelope, subscriber: &str) -> Result<()> {
        let sender = envelope.sender.clone().unwrap_or_default();
        let topic = envelope.topic_id.clone().unwrap_or_default();
        if !self.inbox_filters.accepts(subscriber, envelope) {
            tracing::debug!(
                subscriber,
                sender = %sender,
                topic = %topic,
                "team pub/sub delivery rejected by the inbox filter"
            );
            return Ok(());
        }
        let mut received = TeamEvent::pubsub_received(
            self.team_id(),
            &sender,
            subscriber,
            &topic,
            &envelope.message_id,
        );
        if let Some(session_id) = &envelope.session_id {
            received = received.with_session(session_id.clone());
        }
        self.events.emit(received);
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
        let result = runtime.dispatch_p2p(envelope).await;
        if let Err(error) = &result {
            let recipient = envelope.recipient.clone().unwrap_or_default();
            runtime.emit_dispatch_error(recipient, error, "p2p_dispatch");
        }
        result
    }

    async fn handle_pubsub(&self, envelope: &MessageEnvelope, subscriber: &str) -> Result<()> {
        let runtime = self.runtime.upgrade().ok_or_else(runtime_dropped)?;
        let result = runtime.dispatch_pubsub(envelope, subscriber).await;
        if let Err(error) = &result {
            runtime.emit_dispatch_error(subscriber, error, "pubsub_dispatch");
        }
        result
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
