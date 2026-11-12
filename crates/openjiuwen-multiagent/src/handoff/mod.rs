//! Handoff team: sequential agent-to-agent control transfer over the team bus.
//!
//! Python parity: `openjiuwen/core/multi_agent/teams/handoff/` in the openJiuwen
//! reference implementation. A handoff team runs one member at a time: that
//! member either answers, in which case the chain ends, or asks for a transfer,
//! in which case the next member continues with the accumulated history.
//!
//! ## Flow
//!
//! [`HandoffStrategy::setup`] registers one endpoint per member — an agent whose
//! id is `__handoff_ep_{team_id}_{agent_id}` — and subscribes it to the pub/sub
//! topic `container_{agent_id}`. To start a chain [`HandoffStrategy::invoke`]
//! publishes a [`HandoffRequest`] to the start member's endpoint. The endpoint's
//! [`ContainerAgent`] then:
//!
//! 1. resolves its target through the runtime and mounts the targets'
//!    `transfer_to_<id>` tools on it;
//! 2. invokes the target in a child team session;
//! 3. writes the target's result frames to the team stream and appends
//!    `{"agent", "output"}` to the request history;
//! 4. either completes the chain through the [`HandoffOrchestrator`], forwards a
//!    team interrupt, or publishes the next [`HandoffRequest`] hop.
//!
//! The coordinator enforces the hop cap, the termination condition, and the
//! route graph; every hop and the final result are keyed by the team session id
//! in a [`HandoffRegistry`], so a resumed chain picks up its coordinator state
//! from the session.
//!
//! ## Re-entrancy
//!
//! The first publish comes from the caller's task, but a container runs *inside*
//! the per-topic FIFO worker of the session's pub/sub transport topic. Awaiting
//! `publish` there would enqueue onto the same topic and deadlock, because the
//! worker cannot drain while it awaits the handler. Every follow-up hop is
//! therefore published from a spawned task, and a failed publish is logged
//! rather than raised (the chain then ends through its deadline).

/// Signal key carrying the id of the agent to hand off to.
pub const HANDOFF_TARGET_KEY: &str = "__handoff_to__";

/// Signal key carrying the context message passed to the next agent.
pub const HANDOFF_MESSAGE_KEY: &str = "__handoff_message__";

/// Signal key carrying the reason for the handoff.
pub const HANDOFF_REASON_KEY: &str = "__handoff_reason__";

/// Team-session state key holding the coordinator snapshot
/// (`current_agent_id`, `handoff_count`) for interrupt/resume.
pub const COORDINATOR_STATE_KEY: &str = "__handoff_coordinator__";

/// Team-session state key holding the accumulated hop history.
pub const HISTORY_STATE_KEY: &str = "__handoff_history__";

/// Team-session state key holding the conversation messages mirrored across
/// hops (the cleaned context messages of the hop sessions).
pub const CONTEXT_HISTORY_STATE_KEY: &str = "__handoff_ctx_history__";

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::FutureExt;
use openjiuwen_agent::Agent;
use openjiuwen_common::error::{Error, ErrorKind, Result};
use openjiuwen_session::{AgentTeamSession, Session};
use serde_json::Value;

use crate::card::TeamCard;
use crate::member::{AgentProvider, TeamAgent};
use crate::session::SessionLease;
use crate::strategy::{MemberOptions, StrategyContext, TeamStrategy};
use crate::team::BaseTeam;

mod config;
mod container;
mod interrupt;
mod orchestrator;
mod request;
mod signal;
mod tool;

pub use config::{
    HandoffConfig, HandoffRoute, HandoffSnapshot, HandoffTeamConfig, TerminationCondition,
};
pub use container::ContainerAgent;
pub use interrupt::{extract_interrupt_signal, flush_team_session, TeamInterruptSignal};
pub use orchestrator::{HandoffOrchestrator, HandoffRegistry};
pub use request::{HandoffHistoryEntry, HandoffRequest};
pub use signal::{
    extract_handoff_signal, extract_handoff_signal_from_observations,
    extract_handoff_signal_from_session, HandoffSignal,
};
pub use tool::HandoffTool;

/// The strategy kind of a handoff team.
const HANDOFF_KIND: &str = "handoff";

/// The execution strategy of a handoff team.
///
/// The strategy owns the handoff configuration, the coordinator registry of the
/// live chains, and a readiness flag that [`TeamStrategy::setup`] uses to wire
/// one endpoint per member exactly once per membership change.
#[derive(Clone)]
pub struct HandoffStrategy {
    /// The team limits and the handoff parameters.
    config: HandoffTeamConfig,
    /// Live coordinators, keyed by team session id.
    registry: Arc<HandoffRegistry>,
    /// Whether the per-member endpoints are registered.
    ready: Arc<AtomicBool>,
    /// Serializes endpoint wiring across concurrent invocations.
    setup_lock: Arc<tokio::sync::Mutex<()>>,
}

impl std::fmt::Debug for HandoffStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HandoffStrategy")
            .field("max_handoffs", &self.config.handoff.max_handoffs)
            .field("routes", &self.config.handoff.routes.len())
            .field("ready", &self.ready.load(Ordering::SeqCst))
            .finish_non_exhaustive()
    }
}

impl HandoffStrategy {
    /// Builds the strategy for `config`.
    #[must_use]
    pub fn new(config: HandoffTeamConfig) -> Arc<Self> {
        Arc::new(Self {
            config,
            registry: Arc::new(HandoffRegistry::new()),
            ready: Arc::new(AtomicBool::new(false)),
            setup_lock: Arc::new(tokio::sync::Mutex::new(())),
        })
    }

    /// Registers and subscribes the endpoint of every member.
    ///
    /// Members are wired in registration order: the card and the provider come
    /// from the runtime, the allowed targets from the route graph (so they are
    /// sorted), and the endpoint is registered under
    /// `__handoff_ep_{team_id}_{agent_id}` and subscribed to
    /// `container_{agent_id}`. A member without a registered card or provider is
    /// skipped with a warning instead of failing the run — a strategy must cope
    /// with a member that was unregistered after it was added.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Validation`] for a team
    /// without members, and the subscription error of the runtime.
    async fn wire_endpoints(&self, ctx: &StrategyContext) -> Result<()> {
        let members = ctx.member_ids();
        if members.is_empty() {
            return Err(Error::validation("handoff team has no members"));
        }
        let graph = HandoffOrchestrator::build_route_graph(&members, &self.config.handoff.routes);
        for agent_id in &members {
            let Some(target_card) = ctx.runtime.agent_card(agent_id) else {
                tracing::warn!(
                    team = %ctx.card.id,
                    agent_id = %agent_id,
                    "handoff member has no registered card; skipping its endpoint"
                );
                continue;
            };
            let Some(target_provider) = ctx.runtime.agent_provider(agent_id) else {
                tracing::warn!(
                    team = %ctx.card.id,
                    agent_id = %agent_id,
                    "handoff member has no provider; skipping its endpoint"
                );
                continue;
            };
            let allowed_targets: Vec<String> = graph
                .get(agent_id)
                .map(|targets| targets.iter().cloned().collect())
                .unwrap_or_default();
            let endpoint_id = format!("__handoff_ep_{}_{}", ctx.card.id, agent_id);
            let prototype = ContainerAgent::new(
                target_card,
                target_provider,
                allowed_targets,
                ctx.card.id.clone(),
                endpoint_id.clone(),
                Arc::downgrade(&ctx.runtime),
                Arc::clone(&self.registry),
            );
            let provider: AgentProvider = Arc::new(move || {
                let agent: Arc<dyn Agent> = Arc::new(prototype.clone());
                TeamAgent::new(agent)
            });
            ctx.runtime.register_agent(
                openjiuwen_agent::AgentCard::new(endpoint_id.clone()).with_id(endpoint_id.clone()),
                provider,
            );
            ctx.runtime
                .subscribe(&endpoint_id, &format!("container_{agent_id}"))
                .await?;
            tracing::debug!(
                team = %ctx.card.id,
                endpoint = %endpoint_id,
                agent_id = %agent_id,
                "handoff endpoint registered"
            );
        }
        Ok(())
    }

    /// Runs one handoff chain in an already-acquired team session.
    ///
    /// The chain restores its coordinator from the session, publishes the first
    /// hop, waits for the coordinator (bounded by
    /// [`crate::TeamConfig::message_timeout`] when that is positive), and always
    /// drops the registry entry and the session's transport topics afterwards.
    ///
    /// # Errors
    ///
    /// Propagates the publish, resume, and termination-condition failures, and
    /// maps a chain deadline to
    /// [`openjiuwen_common::error::ErrorKind::Timeout`] with the team context
    /// appended.
    async fn drive(
        &self,
        ctx: &StrategyContext,
        input: Value,
        session: Arc<AgentTeamSession>,
    ) -> Result<Value> {
        self.setup(ctx).await?;
        let session_id = session.id().to_owned();
        let snapshot = session.get_state(COORDINATOR_STATE_KEY).await?;
        let stored = session.get_state(HISTORY_STATE_KEY).await?;
        let mut history = match &stored {
            Some(Value::Array(_)) => {
                match serde_json::from_value::<Vec<HandoffHistoryEntry>>(
                    stored.clone().unwrap_or(Value::Null),
                ) {
                    Ok(history) => history,
                    Err(error) => {
                        tracing::warn!(
                            team = %ctx.card.id,
                            %error,
                            "handoff hop history state is malformed; starting empty"
                        );
                        Vec::new()
                    }
                }
            }
            Some(Value::Null) | None => Vec::new(),
            Some(_) => {
                tracing::warn!(
                    team = %ctx.card.id,
                    "handoff hop history state is not an array; starting empty"
                );
                Vec::new()
            }
        };
        if snapshot.is_some() {
            // A resumed chain drops the hop that was interrupted: it is re-run
            // from the restored coordinator, so its partial output must not
            // pollute the history.
            history.retain(|entry| !is_interrupt_output(&entry.output));
        }
        let members = ctx.member_ids();
        let start = self.config.handoff.start_agent.as_ref().map_or_else(
            || members.first().cloned().unwrap_or_default(),
            |card| card.id.clone(),
        );
        let coordinator = Arc::new(HandoffOrchestrator::restore_from_session(
            snapshot.as_ref(),
            start,
            &members,
            &self.config.handoff,
        )?);
        let current = coordinator.current_agent_id();

        self.registry
            .insert(session_id.clone(), Arc::clone(&coordinator));
        let seconds = self.config.team.message_timeout;
        let timeout = if seconds.is_finite() && seconds > 0.0 {
            Some(Duration::from_secs_f64(seconds))
        } else {
            None
        };
        tracing::debug!(
            team = %ctx.card.id,
            %session_id,
            current = %current,
            hops = history.len(),
            resume = snapshot.is_some(),
            "handoff chain started"
        );

        let published = ctx
            .runtime
            .publish(
                HandoffRequest::new(input, history).to_value(),
                &format!("container_{current}"),
                &ctx.card.id,
                Some(&session_id),
            )
            .await;
        let outcome = if published.is_err() {
            published.map(|()| Value::Null)
        } else {
            coordinator.wait(timeout).await
        };

        self.registry.remove(&session_id);
        ctx.runtime.cleanup_session(&session_id).await;

        match outcome {
            Err(error) if error.kind() == ErrorKind::Timeout => Err(Error::timeout(format!(
                "handoff chain timeout after {seconds}s, team='{}'",
                ctx.card.id
            ))),
            other => other,
        }
    }
}

/// Returns `true` when a hop output is an interrupt result.
fn is_interrupt_output(output: &Value) -> bool {
    output.get("result_type").and_then(Value::as_str) == Some("interrupt")
}

#[async_trait]
impl TeamStrategy for HandoffStrategy {
    fn kind(&self) -> &str {
        HANDOFF_KIND
    }

    /// Invalidates the wired endpoints: the next run registers the new member.
    fn on_member_added(
        &self,
        ctx: &StrategyContext,
        card: &openjiuwen_agent::AgentCard,
        options: &MemberOptions,
    ) {
        let _ = (ctx, card, options);
        self.ready.store(false, Ordering::SeqCst);
    }

    /// Invalidates the wired endpoints: the next run drops the stale member.
    fn on_member_removed(&self, ctx: &StrategyContext, agent_id: &str) {
        let _ = (ctx, agent_id);
        self.ready.store(false, Ordering::SeqCst);
    }

    /// Wires one endpoint per member, once per membership change.
    ///
    /// The readiness flag is checked before the setup lock and again after
    /// acquiring it, so concurrent runs wire the endpoints exactly once (the
    /// double-checked locking Python performs with `asyncio.Lock`).
    ///
    /// # Errors
    ///
    /// Returns the endpoint-wiring error: a validation error for a team without
    /// members, and the runtime's subscription error when an endpoint cannot be
    /// subscribed.
    async fn setup(&self, ctx: &StrategyContext) -> Result<()> {
        if self.ready.load(Ordering::SeqCst) {
            return Ok(());
        }
        let _guard = self.setup_lock.lock().await;
        if self.ready.load(Ordering::SeqCst) {
            return Ok(());
        }
        self.wire_endpoints(ctx).await?;
        self.ready.store(true, Ordering::SeqCst);
        Ok(())
    }

    async fn invoke(
        &self,
        ctx: &StrategyContext,
        input: Value,
        session: Option<Arc<AgentTeamSession>>,
    ) -> Result<Value> {
        let lease = SessionLease::acquire(&ctx.runtime, &ctx.card, &input, session).await;
        let team_session = Arc::clone(lease.session());
        let outcome = self.drive(ctx, input, team_session).await;
        let finished = lease.finish().await;
        match outcome {
            Err(error) => Err(error),
            Ok(value) => finished.map(|()| value),
        }
    }

    /// Streams the chain: the containers write every frame, the strategy none.
    fn stream(
        &self,
        ctx: &StrategyContext,
        input: Value,
        session: Option<Arc<AgentTeamSession>>,
    ) -> BoxStream<'static, Result<Value>> {
        let strategy = self.clone();
        let run_ctx = ctx.clone();
        let run_input = input.clone();
        crate::session::team_stream(
            Arc::clone(&ctx.runtime),
            ctx.card.clone(),
            input,
            session,
            move |team_session| {
                let strategy = strategy.clone();
                let ctx = run_ctx.clone();
                let input = run_input.clone();
                async move {
                    strategy
                        .drive(&ctx, input, team_session)
                        .await
                        .map(|_result| ())
                }
                .boxed()
            },
        )
    }
}

/// Builds a handoff team from its card and configuration.
#[must_use]
pub fn handoff_team(card: TeamCard, config: HandoffTeamConfig) -> BaseTeam {
    BaseTeam::with_strategy(card, config.team.clone(), HandoffStrategy::new(config))
}

#[cfg(test)]
pub(crate) mod support {
    //! Deterministic fakes shared by the handoff unit tests.
    //!
    //! Every fake is an in-process closure: no model, no network, no timers other
    //! than the explicit [`Duration`] delay of [`SlowAgent`].

    use std::sync::Arc;
    use std::time::Duration;

    use async_trait::async_trait;
    use futures::future::BoxFuture;
    use futures::stream::BoxStream;
    use futures::{FutureExt, StreamExt};
    use openjiuwen_agent::{Agent, AgentCard, AgentChunk, AgentOutput};
    use openjiuwen_common::error::Result;
    use openjiuwen_session::Session;
    use serde_json::Value;

    use crate::{AgentProvider, TeamAgent};

    /// The async handler of a [`FakeAgent`].
    pub(crate) type FakeHandler = dyn Fn(Value, Option<Arc<dyn Session>>) -> BoxFuture<'static, Result<AgentOutput>>
        + Send
        + Sync;

    /// Builds a card whose id and name are both `name`.
    pub(crate) fn agent_card(name: &str) -> AgentCard {
        AgentCard::new(name).with_id(name)
    }

    /// An [`Agent`] backed by a closure.
    pub(crate) struct FakeAgent {
        /// The card the agent answers with.
        card: AgentCard,
        /// The invoke handler.
        handler: Box<FakeHandler>,
    }

    impl FakeAgent {
        /// Builds an agent whose `invoke` awaits `handler`.
        pub(crate) fn new(
            card: AgentCard,
            handler: impl Fn(Value, Option<Arc<dyn Session>>) -> BoxFuture<'static, Result<AgentOutput>>
                + Send
                + Sync
                + 'static,
        ) -> Arc<Self> {
            Arc::new(Self {
                card,
                handler: Box::new(handler),
            })
        }

        /// Builds an agent whose `invoke` runs the synchronous `handler`.
        pub(crate) fn sync(
            card: AgentCard,
            handler: impl Fn(Value, Option<Arc<dyn Session>>) -> Result<AgentOutput>
                + Send
                + Sync
                + 'static,
        ) -> Arc<Self> {
            Self::new(card, move |inputs, session| {
                futures::future::ready(handler(inputs, session)).boxed()
            })
        }

        /// Builds an agent that answers `"{name}:{query}"`.
        pub(crate) fn echo(name: &str) -> Arc<Self> {
            let label = name.to_owned();
            Self::sync(agent_card(name), move |inputs, _session| {
                let text = inputs
                    .get("query")
                    .and_then(Value::as_str)
                    .map_or_else(|| inputs.to_string(), ToOwned::to_owned);
                Ok(AgentOutput::answer(format!("{label}:{text}")))
            })
        }
    }

    #[async_trait]
    impl Agent for FakeAgent {
        fn card(&self) -> &AgentCard {
            &self.card
        }

        async fn invoke(
            &self,
            inputs: Value,
            session: Option<Arc<dyn Session>>,
        ) -> Result<AgentOutput> {
            (self.handler)(inputs, session).await
        }

        fn stream(
            &self,
            _inputs: Value,
            _session: Option<Arc<dyn Session>>,
        ) -> BoxStream<'static, Result<AgentChunk>> {
            futures::stream::empty::<Result<AgentChunk>>().boxed()
        }
    }

    /// An [`Agent`] that answers only after a delay.
    pub(crate) struct SlowAgent {
        /// The card the agent answers with.
        card: AgentCard,
        /// How long `invoke` sleeps before answering.
        delay: Duration,
    }

    impl SlowAgent {
        /// Builds an agent named `name` that answers after `delay`.
        pub(crate) fn new(name: &str, delay: Duration) -> Arc<Self> {
            Arc::new(Self {
                card: agent_card(name),
                delay,
            })
        }
    }

    #[async_trait]
    impl Agent for SlowAgent {
        fn card(&self) -> &AgentCard {
            &self.card
        }

        async fn invoke(
            &self,
            _inputs: Value,
            _session: Option<Arc<dyn Session>>,
        ) -> Result<AgentOutput> {
            tokio::time::sleep(self.delay).await;
            Ok(AgentOutput::answer("slow"))
        }

        fn stream(
            &self,
            _inputs: Value,
            _session: Option<Arc<dyn Session>>,
        ) -> BoxStream<'static, Result<AgentChunk>> {
            futures::stream::empty::<Result<AgentChunk>>().boxed()
        }
    }

    /// Builds a member provider that returns a [`FakeAgent`].
    pub(crate) fn fake_provider(agent: Arc<FakeAgent>) -> AgentProvider {
        let agent: Arc<dyn Agent> = agent;
        Arc::new(move || TeamAgent::new(Arc::clone(&agent)))
    }

    /// Builds a member provider that returns a [`SlowAgent`].
    pub(crate) fn slow_provider(agent: Arc<SlowAgent>) -> AgentProvider {
        let agent: Arc<dyn Agent> = agent;
        Arc::new(move || TeamAgent::new(Arc::clone(&agent)))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use futures::StreamExt;
    use openjiuwen_agent::{AgentOutput, Observation};
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;

    use super::support::{agent_card, fake_provider, slow_provider, FakeAgent, SlowAgent};
    use super::*;
    use crate::config::TeamConfig;
    use crate::Team;

    /// Builds a team card with a unique id so tests never share bus topics.
    fn team_card(name: &str) -> TeamCard {
        TeamCard::new(name)
    }

    /// An agent that always asks to hand off to `target`.
    fn handoff_agent(name: &str, target: &str) -> Arc<FakeAgent> {
        let target = target.to_owned();
        let label = name.to_owned();
        FakeAgent::sync(agent_card(name), move |_inputs, _session| {
            Ok(AgentOutput {
                output: format!("{label} hands off"),
                observations: vec![Observation {
                    content: json!({"__handoff_to__": target, "__handoff_reason__": "route"})
                        .to_string(),
                    ..Observation::default()
                }],
                ..AgentOutput::default()
            })
        })
    }

    #[tokio::test]
    async fn a_single_hop_returns_the_member_output() {
        let team = handoff_team(team_card("single"), HandoffTeamConfig::default());
        assert_eq!(Team::kind(&team), "handoff");
        team.add_agent(agent_card("a"), fake_provider(FakeAgent::echo("a")))
            .unwrap();

        let result = team
            .invoke(json!({"query": "hi", "conversation_id": "single-1"}), None)
            .await
            .unwrap();
        assert_eq!(result["output"], "a:hi");
        assert_eq!(result["result_type"], "answer");
    }

    #[tokio::test]
    async fn a_two_hop_chain_returns_the_last_output() {
        let team = handoff_team(team_card("two-hop"), HandoffTeamConfig::default());
        team.add_agent(agent_card("a"), fake_provider(handoff_agent("a", "b")))
            .unwrap();
        team.add_agent(agent_card("b"), fake_provider(FakeAgent::echo("b")))
            .unwrap();

        let result = team
            .invoke(json!({"query": "hi", "conversation_id": "two-hop-1"}), None)
            .await
            .unwrap();
        assert_eq!(
            result["output"], "b:hi",
            "the second hop's output ends the chain"
        );
    }

    #[tokio::test]
    async fn a_blocked_route_completes_with_the_first_hop() {
        let config = HandoffTeamConfig {
            team: TeamConfig::default(),
            handoff: HandoffConfig {
                routes: vec![HandoffRoute::new("b", "a")],
                ..HandoffConfig::default()
            },
        };
        let team = handoff_team(team_card("blocked"), config);
        team.add_agent(agent_card("a"), fake_provider(handoff_agent("a", "b")))
            .unwrap();
        team.add_agent(agent_card("b"), fake_provider(FakeAgent::echo("b")))
            .unwrap();

        let result = team
            .invoke(json!({"query": "hi", "conversation_id": "blocked-1"}), None)
            .await
            .unwrap();
        assert_eq!(result["output"], "a hands off");
    }

    #[tokio::test]
    async fn a_zero_handoff_cap_blocks_every_handoff() {
        let config = HandoffTeamConfig {
            team: TeamConfig::default(),
            handoff: HandoffConfig {
                max_handoffs: 0,
                ..HandoffConfig::default()
            },
        };
        let team = handoff_team(team_card("capped"), config);
        team.add_agent(agent_card("a"), fake_provider(handoff_agent("a", "b")))
            .unwrap();
        team.add_agent(agent_card("b"), fake_provider(FakeAgent::echo("b")))
            .unwrap();

        let result = team
            .invoke(json!({"query": "hi", "conversation_id": "capped-1"}), None)
            .await
            .unwrap();
        assert_eq!(result["output"], "a hands off");
    }

    #[tokio::test]
    async fn stream_yields_the_hop_frames_and_ends() {
        let team = handoff_team(team_card("streamed"), HandoffTeamConfig::default());
        team.add_agent(agent_card("a"), fake_provider(FakeAgent::echo("a")))
            .unwrap();

        let frames: Vec<_> = team
            .stream(
                json!({"query": "hi", "conversation_id": "streamed-1"}),
                None,
            )
            .collect()
            .await;
        assert_eq!(frames.len(), 1, "one hop writes one frame");
        let frame = frames[0].as_ref().unwrap();
        assert_eq!(frame["type"], "message");
        assert_eq!(frame["payload"]["output"], "a:hi");
    }

    #[tokio::test]
    async fn a_short_message_timeout_is_rewritten_with_the_team() {
        let config = HandoffTeamConfig {
            team: TeamConfig::new().configure_timeout(0.01),
            ..HandoffTeamConfig::default()
        };
        let team = handoff_team(team_card("timing-out"), config);
        let slow = SlowAgent::new("a", Duration::from_millis(50));
        team.add_agent(agent_card("a"), slow_provider(slow))
            .unwrap();

        let error = team
            .invoke(json!({"query": "hi", "conversation_id": "timeout-1"}), None)
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Timeout);
        assert!(
            error.message().contains("team="),
            "the timeout names the team: {}",
            error.message()
        );
        assert!(error.message().contains("handoff chain timeout"));
    }

    #[tokio::test]
    async fn a_member_without_a_provider_is_skipped() {
        let team = handoff_team(team_card("skipping"), HandoffTeamConfig::default());
        team.add_agent(agent_card("a"), fake_provider(FakeAgent::echo("a")))
            .unwrap();
        team.add_agent(agent_card("b"), fake_provider(FakeAgent::echo("b")))
            .unwrap();
        assert!(team.runtime().unregister_agent("b").is_some());

        let result = team
            .invoke(
                json!({"query": "hi", "conversation_id": "skipping-1"}),
                None,
            )
            .await
            .unwrap();
        assert_eq!(
            result["output"], "a:hi",
            "the remaining member still runs the chain"
        );
    }

    #[tokio::test]
    async fn an_empty_team_is_a_validation_error() {
        let team = handoff_team(team_card("empty"), HandoffTeamConfig::default());
        let error = team
            .invoke(json!({"conversation_id": "empty-1"}), None)
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.message().contains("handoff team has no members"));
    }

    #[tokio::test]
    async fn a_caller_provided_session_is_used_and_left_bound() {
        let team = handoff_team(team_card("resume"), HandoffTeamConfig::default());
        team.add_agent(agent_card("a"), fake_provider(handoff_agent("a", "b")))
            .unwrap();
        team.add_agent(agent_card("b"), fake_provider(FakeAgent::echo("b")))
            .unwrap();

        let session =
            crate::make_team_session(&team.card(), &json!({"conversation_id": "resume-1"}));
        team.runtime().bind_team_session(Arc::clone(&session));

        let result = team
            .invoke(
                json!({"query": "hi", "conversation_id": "resume-1"}),
                Some(Arc::clone(&session)),
            )
            .await
            .unwrap();
        assert_eq!(result["output"], "b:hi");
        assert!(
            team.runtime().team_session("resume-1").is_some(),
            "a caller-owned session stays bound"
        );
    }
}
