//! Team composition: the [`Team`] seam and its default implementation.
//!
//! Python parity: `BaseTeam` in `openjiuwen/core/multi_agent/team.py`. A team
//! owns the card (whose `agent_cards` list mutates as members come and go), the
//! [`TeamConfig`] limits, one [`TeamRuntime`] per team, and one
//! [`TeamStrategy`] that decides how an input becomes member invocations.
//!
//! The Python class hierarchy (`BaseTeam` plus a strategy subclass per topology)
//! collapses into one type: [`BaseTeam`] implements [`Team`] and delegates the
//! execution half of the contract to its strategy. A team built without a
//! strategy keeps the default [`NoStrategy`], which rejects every invocation
//! with [`openjiuwen_common::error::Error::Unsupported`].
//!
//! The trait is also the seam the runner drives: it calls
//! [`Team::invoke`]/[`Team::stream`] with the run's team session and binds that
//! session on [`Team::runtime`].
//!
//! # Auto-subscription
//!
//! Membership wires the bus automatically: with the default
//! [`AutoSubscriptionMode::Merge`], every added member subscribes to the patterns
//! [`crate::autosubscribe::derive_with_overrides`] derives from the *current*
//! team card — the `topic` broadcast (`topic`, `topic.*`, `topic.<member name>`),
//! `subscribe:` tags, and any installed overrides — and [`Team::start`] applies
//! the whole plan once. A card without a topic, a `subscribe:` tag, or an
//! override derives an empty plan, so legacy teams keep exactly their manual
//! wiring; [`BaseTeam::with_auto_subscriptions`] switches to
//! [`AutoSubscriptionMode::Replace`] (which also drops stale patterns) or
//! [`AutoSubscriptionMode::Disabled`] (which never touches the registry).

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::stream::{BoxStream, StreamExt};
use openjiuwen_agent::AgentCard;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_session::AgentTeamSession;
use parking_lot::RwLock;
use serde_json::Value;

use crate::autosubscribe::{self, AutoSubscriptionMode, AutoSubscriptionPlan};
use crate::card::TeamCard;
use crate::config::TeamConfig;
use crate::member::AgentProvider;
use crate::message_bus::MessageBusConfig;
use crate::runtime::{RuntimeConfig, TeamRuntime};
use crate::strategy::{MemberOptions, StrategyContext, TeamStrategy};

/// Message reported by a team that was built without a strategy.
const NO_STRATEGY: &str = "team has no strategy";

/// The public surface of a team of agents.
///
/// Membership mutates through [`add_agent`](Self::add_agent) and
/// [`remove_agent`](Self::remove_agent) while the card, configuration, and
/// runtime stay fixed; [`card`](Self::card) is therefore a snapshot, not a
/// borrow. Execution is delegated to the team's strategy.
#[async_trait]
pub trait Team: Send + Sync {
    /// Returns a snapshot of the team card, including the current member list.
    fn card(&self) -> TeamCard;

    /// Returns the team limits captured at construction.
    fn config(&self) -> TeamConfig;

    /// Returns the team id: the card *name* captured at construction.
    fn team_id(&self) -> &str;

    /// Returns the strategy kind (`"handoff"`, `"hierarchical_msgbus"`, …).
    fn kind(&self) -> &str;

    /// Returns the runtime that dispatches member invocations.
    fn runtime(&self) -> Arc<TeamRuntime>;

    /// Adds a member with the default [`MemberOptions`].
    ///
    /// A duplicate agent id is a warning and a no-op.
    ///
    /// # Errors
    ///
    /// [`openjiuwen_common::error::ErrorKind::Validation`] when the team already
    /// holds [`TeamConfig::max_agents`] members, and whatever
    /// [`TeamStrategy::on_member_added`] reports.
    fn add_agent(&self, card: AgentCard, provider: AgentProvider) -> Result<()>;

    /// Adds a member with explicit wiring options (a hierarchical parent, …).
    ///
    /// # Errors
    ///
    /// Same as [`add_agent`](Self::add_agent).
    fn add_member(
        &self,
        card: AgentCard,
        provider: AgentProvider,
        options: MemberOptions,
    ) -> Result<()>;

    /// Removes a member by agent id or by card name; an unknown member is a no-op.
    ///
    /// # Errors
    ///
    /// Propagates whatever [`TeamStrategy::on_member_removed`] reports.
    fn remove_agent(&self, agent: &str) -> Result<()>;

    /// Alias of [`Team::remove_agent`].
    ///
    /// Accepts an agent id or a card id; an unknown member is a no-op.
    ///
    /// # Errors
    ///
    /// Propagates whatever [`TeamStrategy::on_member_removed`] reports.
    fn remove_member(&self, agent: &str) -> Result<()> {
        self.remove_agent(agent)
    }

    /// Returns the registered card of `agent_id`, when it is a member.
    fn get_agent_card(&self, agent_id: &str) -> Option<AgentCard>;

    /// Returns the number of members.
    fn agent_count(&self) -> usize;

    /// Returns the member agent ids in registration order.
    fn list_agents(&self) -> Vec<String>;

    /// Validates that the team is complete enough to run.
    ///
    /// # Errors
    ///
    /// Propagates the strategy's readiness check (a missing supervisor or root
    /// agent, for instance).
    fn check_ready(&self) -> Result<()>;

    /// Sends `message` to `recipient` and waits for its reply.
    ///
    /// # Errors
    ///
    /// [`openjiuwen_common::error::ErrorKind::Validation`] when either end is
    /// not a member, and the transport/member errors of the runtime.
    async fn send(
        &self,
        message: Value,
        recipient: &str,
        sender: &str,
        session_id: Option<&str>,
        timeout: Option<Duration>,
    ) -> Result<Value>;

    /// Publishes `message` on `topic_id` and waits for the fan-out.
    ///
    /// # Errors
    ///
    /// [`openjiuwen_common::error::ErrorKind::Validation`] when the sender is
    /// not a member, and the transport errors of the runtime.
    async fn publish(
        &self,
        message: Value,
        topic_id: &str,
        sender: &str,
        session_id: Option<&str>,
    ) -> Result<()>;

    /// Subscribes `agent_id` to `topic` (an exact topic or a `*`/`?` pattern).
    ///
    /// # Errors
    ///
    /// Propagates the runtime's validation of the arguments.
    async fn subscribe(&self, agent_id: &str, topic: &str) -> Result<()>;

    /// Removes the `(agent_id, topic)` subscription.
    ///
    /// # Errors
    ///
    /// Propagates the runtime's validation of the arguments.
    async fn unsubscribe(&self, agent_id: &str, topic: &str) -> Result<()>;

    /// Starts the team's transport. Idempotent.
    ///
    /// A [`BaseTeam`] also applies its card-driven subscription plan once, so a
    /// member registered before the first start gets its planned subscriptions.
    ///
    /// # Errors
    ///
    /// Propagates a transport start failure.
    async fn start(&self) -> Result<()>;

    /// Stops the team's transport. Idempotent.
    ///
    /// # Errors
    ///
    /// Propagates a transport stop failure.
    async fn stop(&self) -> Result<()>;

    /// Runs one team invocation to completion.
    ///
    /// # Errors
    ///
    /// Propagates every strategy, member, transport, and deadline failure.
    async fn invoke(&self, input: Value, session: Option<Arc<AgentTeamSession>>) -> Result<Value>;

    /// Runs one team invocation as a stream of wire frames.
    fn stream(
        &self,
        input: Value,
        session: Option<Arc<AgentTeamSession>>,
    ) -> BoxStream<'static, Result<Value>>;
}

/// The strategy of a team built without one.
///
/// Every execution attempt is rejected with
/// [`openjiuwen_common::error::Error::Unsupported`], which makes an
/// unconfigured team a loud failure instead of a silent no-op.
struct NoStrategy;

#[async_trait]
impl TeamStrategy for NoStrategy {
    fn kind(&self) -> &'static str {
        "none"
    }

    async fn invoke(
        &self,
        _ctx: &StrategyContext,
        _input: Value,
        _session: Option<Arc<AgentTeamSession>>,
    ) -> Result<Value> {
        Err(Error::unsupported(NO_STRATEGY))
    }

    fn stream(
        &self,
        _ctx: &StrategyContext,
        _input: Value,
        _session: Option<Arc<AgentTeamSession>>,
    ) -> BoxStream<'static, Result<Value>> {
        futures::stream::once(async { Err(Error::unsupported(NO_STRATEGY)) }).boxed()
    }
}

/// The default [`Team`]: card, configuration, runtime, and one strategy.
///
/// Construction chooses the runtime: [`BaseTeam::new`] and
/// [`BaseTeam::with_strategy`] build one from the card id and the team config
/// (queue capacity and dispatch deadline), while [`BaseTeam::with_runtime`]
/// takes a runtime the caller already owns — the seam tests and hierarchical
/// strategies use to share wiring.
pub struct BaseTeam {
    card: RwLock<TeamCard>,
    config: TeamConfig,
    runtime: Arc<TeamRuntime>,
    strategy: Arc<dyn TeamStrategy>,
    team_id: String,
    started: AtomicBool,
    auto_subscriptions: RwLock<AutoSubscriptionMode>,
    subscription_overrides: RwLock<BTreeMap<String, Vec<String>>>,
}

impl std::fmt::Debug for BaseTeam {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BaseTeam")
            .field("team_id", &self.team_id)
            .field("kind", &self.strategy.kind())
            .field("agents", &self.list_agents())
            .field("auto_subscriptions", &self.auto_subscriptions())
            .finish_non_exhaustive()
    }
}

impl BaseTeam {
    /// Builds a strategy-free team: every invocation is rejected as unsupported.
    #[must_use]
    pub fn new(card: TeamCard, config: TeamConfig) -> Self {
        Self::build(card, config, None, Arc::new(NoStrategy))
    }

    /// Builds a team that executes through `strategy`.
    #[must_use]
    pub fn with_strategy(
        card: TeamCard,
        config: TeamConfig,
        strategy: Arc<dyn TeamStrategy>,
    ) -> Self {
        Self::build(card, config, None, strategy)
    }

    /// Builds a team around an existing `runtime`.
    #[must_use]
    pub fn with_runtime(
        card: TeamCard,
        config: TeamConfig,
        runtime: Arc<TeamRuntime>,
        strategy: Arc<dyn TeamStrategy>,
    ) -> Self {
        Self::build(card, config, Some(runtime), strategy)
    }

    /// Returns the strategy this team executes through.
    #[must_use]
    pub fn strategy(&self) -> Arc<dyn TeamStrategy> {
        Arc::clone(&self.strategy)
    }

    /// Switches card-driven auto-subscription to `mode` (builder form).
    #[must_use]
    pub fn with_auto_subscriptions(self, mode: AutoSubscriptionMode) -> Self {
        self.set_auto_subscriptions(mode);
        self
    }

    /// Returns the card-driven auto-subscription mode.
    ///
    /// The default is [`AutoSubscriptionMode::Merge`].
    #[must_use]
    pub fn auto_subscriptions(&self) -> AutoSubscriptionMode {
        *self.auto_subscriptions.read()
    }

    /// Changes the mode without touching the subscriptions already registered.
    pub fn set_auto_subscriptions(&self, mode: AutoSubscriptionMode) {
        *self.auto_subscriptions.write() = mode;
    }

    /// Installs explicit per-agent subscription overrides (builder form).
    ///
    /// An override replaces the derived pattern set of the named agent id
    /// verbatim (sorted and de-duplicated), and ids that are not members are kept,
    /// matching [`crate::autosubscribe::derive_with_overrides`].
    #[must_use]
    pub fn with_subscription_overrides(self, overrides: BTreeMap<String, Vec<String>>) -> Self {
        *self.subscription_overrides.write() = overrides;
        self
    }

    /// Returns the installed subscription overrides.
    #[must_use]
    pub fn subscription_overrides(&self) -> BTreeMap<String, Vec<String>> {
        self.subscription_overrides.read().clone()
    }

    /// Returns the subscription plan of the current card.
    ///
    /// The plan is derived from the member list captured at call time plus the
    /// card's `topic` and `subscribe:` tags, with the installed overrides applied
    /// last; every pattern list is sorted and de-duplicated, so iteration order and
    /// change counts are deterministic. A card without a topic, a `subscribe:` tag,
    /// or an override derives an empty plan.
    #[must_use]
    pub fn subscription_plan(&self) -> AutoSubscriptionPlan {
        autosubscribe::derive_with_overrides(&self.card(), &self.subscription_overrides())
    }

    /// Applies the whole plan to the runtime and returns the number of changes.
    ///
    /// [`AutoSubscriptionMode::Merge`] counts only the patterns that were not
    /// subscribed before, [`AutoSubscriptionMode::Replace`] also removes stale
    /// patterns (and counts them), and [`AutoSubscriptionMode::Disabled`] returns
    /// `0` and leaves the registry untouched.
    #[must_use]
    pub fn apply_subscriptions(&self) -> usize {
        let plan = self.subscription_plan();
        let mode = self.auto_subscriptions();
        autosubscribe::apply(&self.runtime, &plan, mode)
    }

    /// Applies the planned patterns of `agent_id` and returns the number of
    /// changes.
    ///
    /// The planned patterns are the ones [`subscription_plan`](Self::subscription_plan)
    /// holds for that agent, applied with the same ordering and counting rules as
    /// [`apply_subscriptions`](Self::apply_subscriptions). An agent absent from the
    /// plan derives nothing, so it gains nothing in
    /// [`AutoSubscriptionMode::Merge`] and loses its current subscriptions in
    /// [`AutoSubscriptionMode::Replace`].
    #[must_use]
    pub fn apply_member_subscriptions(&self, agent_id: &str) -> usize {
        let plan = self.subscription_plan();
        let mode = self.auto_subscriptions();
        autosubscribe::apply_agent(&self.runtime, &plan, agent_id, mode)
    }

    /// Shared constructor: captures the team id and resolves the runtime.
    fn build(
        card: TeamCard,
        config: TeamConfig,
        runtime: Option<Arc<TeamRuntime>>,
        strategy: Arc<dyn TeamStrategy>,
    ) -> Self {
        // Python parity: `team_id` is the card *name*, while topics, sessions,
        // and registries key on the card id.
        let team_id = card.name.clone();
        let runtime = runtime.unwrap_or_else(|| runtime_for(&card, &config));
        Self {
            card: RwLock::new(card),
            config,
            runtime,
            strategy,
            team_id,
            started: AtomicBool::new(false),
            auto_subscriptions: RwLock::new(AutoSubscriptionMode::default()),
            subscription_overrides: RwLock::new(BTreeMap::new()),
        }
    }

    /// Builds the immutable view handed to the strategy.
    ///
    /// The card lock is released before the context is returned, so no strategy
    /// hook runs while a guard is held.
    fn strategy_context(&self) -> StrategyContext {
        let card = self.card();
        StrategyContext::new(
            card.clone(),
            self.config.clone(),
            card.agent_cards,
            Arc::clone(&self.runtime),
        )
    }

    /// Resolves `agent` (an id or a card name) to a member id.
    fn resolve_agent_id(&self, agent: &str) -> Option<String> {
        if self.runtime.has_agent(agent) {
            return Some(agent.to_owned());
        }
        let card = self.card.read();
        card.agent_cards
            .iter()
            .find(|member| member.name == agent)
            .map(|member| member.id.clone())
    }

    /// Registers a member unless it is a duplicate or the team is full.
    fn insert_member(
        &self,
        card: &AgentCard,
        provider: AgentProvider,
        options: &MemberOptions,
    ) -> Result<()> {
        if self.runtime.agent_card(&card.id).is_some() {
            tracing::warn!(
                team = %self.team_id,
                agent_id = %card.id,
                "agent is already a member; ignoring the duplicate"
            );
            // Applying is idempotent, so the duplicate path only reconciles the
            // member's patterns (a repeated card may now derive more of them).
            let _ = self.apply_member_subscriptions(&card.id);
            return Ok(());
        }
        if self.agent_count() >= self.config.max_agents {
            return Err(Error::validation(format!(
                "Agent count exceeds max_agents ({})",
                self.config.max_agents
            )));
        }
        self.runtime.register_agent(card.clone(), provider);
        self.card.write().agent_cards.push(card.clone());
        let _ = self.apply_member_subscriptions(&card.id);
        self.strategy
            .on_member_added(&self.strategy_context(), card, options);
        Ok(())
    }

    /// Drops the planned subscriptions of `agent_id`.
    ///
    /// Only the patterns the plan holds for that agent are removed, so unrelated
    /// manual subscriptions of other members stay. A
    /// [`AutoSubscriptionMode::Disabled`] team never wired the plan, so this is a
    /// no-op for it.
    fn remove_member_subscriptions(&self, agent_id: &str) {
        if self.auto_subscriptions() == AutoSubscriptionMode::Disabled {
            return;
        }
        let plan = self.subscription_plan();
        let patterns: &[String] = plan.get(agent_id).unwrap_or_default();
        let bus = self.runtime.message_bus();
        for pattern in patterns {
            bus.remove_subscription(agent_id, pattern);
        }
    }
}

/// Builds the runtime a team owns: its topics are named after the card id and
/// its bus inherits the team's queue capacity and dispatch deadline.
fn runtime_for(card: &TeamCard, config: &TeamConfig) -> Arc<TeamRuntime> {
    TeamRuntime::new(RuntimeConfig {
        team_id: card.id.clone(),
        message_bus: Some(MessageBusConfig {
            max_queue_size: config.max_concurrent_messages,
            process_timeout: Some(config.message_timeout),
            team_id: Some(card.id.clone()),
        }),
        ..RuntimeConfig::default()
    })
}

#[async_trait]
impl Team for BaseTeam {
    fn card(&self) -> TeamCard {
        self.card.read().clone()
    }

    fn config(&self) -> TeamConfig {
        self.config.clone()
    }

    fn team_id(&self) -> &str {
        &self.team_id
    }

    fn kind(&self) -> &str {
        self.strategy.kind()
    }

    fn runtime(&self) -> Arc<TeamRuntime> {
        Arc::clone(&self.runtime)
    }

    fn add_agent(&self, card: AgentCard, provider: AgentProvider) -> Result<()> {
        self.insert_member(&card, provider, &MemberOptions::default())
    }

    fn add_member(
        &self,
        card: AgentCard,
        provider: AgentProvider,
        options: MemberOptions,
    ) -> Result<()> {
        self.insert_member(&card, provider, &options)
    }

    fn remove_agent(&self, agent: &str) -> Result<()> {
        let Some(agent_id) = self.resolve_agent_id(agent) else {
            return Ok(());
        };
        self.remove_member_subscriptions(&agent_id);
        if self.runtime.unregister_agent(&agent_id).is_none() {
            return Ok(());
        }
        self.card
            .write()
            .agent_cards
            .retain(|member| member.id != agent_id);
        self.strategy
            .on_member_removed(&self.strategy_context(), &agent_id);
        Ok(())
    }

    fn get_agent_card(&self, agent_id: &str) -> Option<AgentCard> {
        self.runtime.agent_card(agent_id)
    }

    fn agent_count(&self) -> usize {
        self.runtime.agent_count()
    }

    fn list_agents(&self) -> Vec<String> {
        self.runtime.agent_ids()
    }

    fn check_ready(&self) -> Result<()> {
        self.strategy.check_ready(&self.strategy_context())
    }

    async fn send(
        &self,
        message: Value,
        recipient: &str,
        sender: &str,
        session_id: Option<&str>,
        timeout: Option<Duration>,
    ) -> Result<Value> {
        if self.runtime.agent_card(sender).is_none() {
            return Err(Error::validation(format!(
                "Sender '{sender}' not found in team '{}'",
                self.team_id
            )));
        }
        if self.runtime.agent_card(recipient).is_none() {
            return Err(Error::validation(format!(
                "Recipient '{recipient}' not found in team '{}'",
                self.team_id
            )));
        }
        self.runtime
            .send(message, recipient, sender, session_id, timeout)
            .await
    }

    async fn publish(
        &self,
        message: Value,
        topic_id: &str,
        sender: &str,
        session_id: Option<&str>,
    ) -> Result<()> {
        if self.runtime.agent_card(sender).is_none() {
            return Err(Error::validation(format!(
                "Sender '{sender}' not found in team '{}'",
                self.team_id
            )));
        }
        self.runtime
            .publish(message, topic_id, sender, session_id)
            .await
    }

    async fn subscribe(&self, agent_id: &str, topic: &str) -> Result<()> {
        self.runtime.subscribe(agent_id, topic).await
    }

    async fn unsubscribe(&self, agent_id: &str, topic: &str) -> Result<()> {
        self.runtime.unsubscribe(agent_id, topic).await
    }

    async fn start(&self) -> Result<()> {
        if self.started.swap(true, Ordering::SeqCst) {
            return Ok(());
        }
        if let Err(error) = self.runtime.start().await {
            self.started.store(false, Ordering::SeqCst);
            return Err(error);
        }
        let changes = self.apply_subscriptions();
        if changes > 0 {
            tracing::debug!(
                team = %self.team_id,
                changes,
                "applied card-driven subscriptions on start"
            );
        }
        Ok(())
    }

    async fn stop(&self) -> Result<()> {
        if !self.started.swap(false, Ordering::SeqCst) {
            return Ok(());
        }
        if let Err(error) = self.runtime.stop().await {
            self.started.store(true, Ordering::SeqCst);
            return Err(error);
        }
        Ok(())
    }

    async fn invoke(&self, input: Value, session: Option<Arc<AgentTeamSession>>) -> Result<Value> {
        self.strategy
            .invoke(&self.strategy_context(), input, session)
            .await
    }

    fn stream(
        &self,
        input: Value,
        session: Option<Arc<AgentTeamSession>>,
    ) -> BoxStream<'static, Result<Value>> {
        self.strategy
            .stream(&self.strategy_context(), input, session)
    }
}
