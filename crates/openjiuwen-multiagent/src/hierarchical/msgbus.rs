//! Supervisor-over-message-bus hierarchical teams.
//!
//! Python parity: `openjiuwen/core/multi_agent/teams/hierarchical_msgbus/`. The
//! team registers a supervisor agent — normally the one built by
//! [`create_supervisor`](super::create_supervisor) — next to its sub-agents, and
//! every invocation is a single P2P request to that supervisor. The supervisor's
//! `ReActAgent` loop then calls the sub-agents through the same runtime, so the
//! whole hierarchy stays on the bus.
//!
//! The strategy itself is thin on purpose: it owns the supervisor id and the
//! per-dispatch deadline, validates that the supervisor was registered, and
//! delegates all member dispatch to
//! [`TeamRuntime::send`](crate::TeamRuntime::send).

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::FutureExt;
use openjiuwen_agent::AgentCard;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_session::{AgentTeamSession, Session, StreamItem};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::card::TeamCard;
use crate::config::TeamConfig;
use crate::session::SessionLease;
use crate::strategy::{StrategyContext, TeamStrategy};
use crate::team::BaseTeam;

/// The strategy kind reported by a message-bus hierarchical team.
const HIERARCHICAL_MSGBUS_KIND: &str = "hierarchical_msgbus";

/// Default P2P dispatch deadline of a message-bus hierarchical team, in seconds.
const DEFAULT_TIMEOUT_SECONDS: f64 = 1800.0;

/// Configuration of a message-bus hierarchical team.
///
/// Python parity: `HierarchicalTeamConfig`. It subclasses `TeamConfig`, so Rust
/// composes the team limits under [`HierarchicalMsgbusConfig::team`] while
/// serializing them flattened into the same JSON object. `timeout` bounds one
/// P2P dispatch to the supervisor (`None`, `0`, or a negative value means "no
/// per-invocation override": the runtime's own deadline applies).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct HierarchicalMsgbusConfig {
    /// The team limits shared by every strategy.
    #[serde(flatten)]
    pub team: TeamConfig,
    /// Card of the supervisor that receives every invocation.
    pub supervisor_agent: AgentCard,
    /// P2P dispatch deadline of one invocation, in seconds.
    pub timeout: Option<f64>,
}

impl Default for HierarchicalMsgbusConfig {
    fn default() -> Self {
        Self {
            team: TeamConfig::new(),
            supervisor_agent: AgentCard::default(),
            timeout: Some(DEFAULT_TIMEOUT_SECONDS),
        }
    }
}

impl HierarchicalMsgbusConfig {
    /// Builds a configuration with the Python defaults and `supervisor_agent`.
    #[must_use]
    pub fn new(supervisor_agent: AgentCard) -> Self {
        Self {
            supervisor_agent,
            ..Self::default()
        }
    }
}

/// Converts a configured timeout in seconds into a deadline.
///
/// `None`, a non-positive value, and a non-finite value all mean "no
/// per-invocation deadline", so the runtime's own P2P deadline applies.
fn deadline(seconds: Option<f64>) -> Option<Duration> {
    match seconds {
        Some(seconds) if seconds.is_finite() && seconds > 0.0 => {
            Some(Duration::from_secs_f64(seconds))
        }
        _ => None,
    }
}

/// The execution strategy of a message-bus hierarchical team.
///
/// One invocation is one P2P request to the supervisor; the supervisor's tools
/// dispatch the sub-agents. Nothing else is wired here, so the strategy is
/// stateless apart from the supervisor id and the deadline.
#[derive(Debug, Clone)]
pub struct HierarchicalMsgbusStrategy {
    /// Id of the supervisor agent registered in the team runtime.
    supervisor_id: String,
    /// Per-invocation P2P deadline; `None` defers to the runtime deadline.
    timeout: Option<Duration>,
}

impl HierarchicalMsgbusStrategy {
    /// Builds the strategy for `config`.
    #[must_use]
    pub fn new(config: HierarchicalMsgbusConfig) -> Arc<Self> {
        Arc::new(Self {
            supervisor_id: config.supervisor_agent.id,
            timeout: deadline(config.timeout),
        })
    }

    /// Sends `input` to the supervisor and awaits its result.
    ///
    /// # Errors
    ///
    /// Propagates the readiness check, the runtime's validation of the sender
    /// and recipient, and the supervisor's own failure.
    async fn dispatch(
        &self,
        ctx: &StrategyContext,
        input: Value,
        session_id: Option<&str>,
    ) -> Result<Value> {
        ctx.runtime
            .send(
                input,
                &self.supervisor_id,
                &ctx.card.id,
                session_id,
                self.timeout,
            )
            .await
    }
}

#[async_trait]
impl TeamStrategy for HierarchicalMsgbusStrategy {
    fn kind(&self) -> &str {
        HIERARCHICAL_MSGBUS_KIND
    }

    /// Adopts the configured deadline when the supervisor is registered.
    ///
    /// Python parity: registering the supervisor sets the runtime's P2P timeout,
    /// so a dispatch that does not pass an explicit deadline still honours the
    /// configuration.
    fn on_member_added(
        &self,
        ctx: &StrategyContext,
        card: &AgentCard,
        options: &crate::MemberOptions,
    ) {
        let _ = options;
        if card.id == self.supervisor_id {
            if let Some(timeout) = self.timeout {
                ctx.runtime.set_p2p_timeout(timeout);
            }
        }
    }

    /// Requires a configured supervisor that is registered in the runtime.
    ///
    /// # Errors
    ///
    /// [`openjiuwen_common::error::ErrorKind::Config`] when the supervisor id is
    /// empty or the supervisor was never registered.
    fn check_ready(&self, ctx: &StrategyContext) -> Result<()> {
        if self.supervisor_id.is_empty() {
            return Err(Error::config(
                "No supervisor configured in HierarchicalTeamConfig.",
            ));
        }
        if !ctx.runtime.has_agent(&self.supervisor_id) {
            return Err(Error::config(format!(
                "Supervisor '{}' is not registered in runtime. \
                 Call add_agent(supervisor_card, supervisor_provider) before invoke()/stream().",
                self.supervisor_id
            )));
        }
        Ok(())
    }

    async fn invoke(
        &self,
        ctx: &StrategyContext,
        input: Value,
        session: Option<Arc<AgentTeamSession>>,
    ) -> Result<Value> {
        self.check_ready(ctx)?;
        let lease = SessionLease::acquire(&ctx.runtime, &ctx.card, &input, session).await;
        let session_id = lease.session_id().to_owned();
        let outcome = self.dispatch(ctx, input, Some(&session_id)).await;
        let finished = lease.finish().await;
        match outcome {
            Err(error) => Err(error),
            Ok(value) => finished.map(|()| value),
        }
    }

    /// Sends the input to the supervisor and streams its single result frame.
    ///
    /// The run task writes exactly one `message` frame carrying
    /// `{"output": <result>}` when the supervisor answered with a non-null
    /// value; a send failure is surfaced after the frames like any other
    /// strategy error.
    fn stream(
        &self,
        ctx: &StrategyContext,
        input: Value,
        session: Option<Arc<AgentTeamSession>>,
    ) -> BoxStream<'static, Result<Value>> {
        let run_runtime = Arc::clone(&ctx.runtime);
        let supervisor_id = self.supervisor_id.clone();
        let sender = ctx.card.id.clone();
        let timeout = self.timeout;
        let run_input = input.clone();
        crate::session::team_stream(
            Arc::clone(&ctx.runtime),
            ctx.card.clone(),
            input,
            session,
            move |team_session| {
                let runtime = Arc::clone(&run_runtime);
                let supervisor_id = supervisor_id.clone();
                let sender = sender.clone();
                let message = run_input.clone();
                async move {
                    let result = runtime
                        .send(
                            message,
                            &supervisor_id,
                            &sender,
                            Some(team_session.id()),
                            timeout,
                        )
                        .await?;
                    if !result.is_null() {
                        let item =
                            StreamItem::output("message", Some(0), json!({"output": result}));
                        if let Err(error) = team_session.write_stream(item).await {
                            tracing::warn!(
                                team = %sender,
                                %error,
                                "hierarchical_msgbus: failed to write the supervisor result \
                                 to the team stream"
                            );
                        }
                    }
                    Ok(())
                }
                .boxed()
            },
        )
    }
}

/// Builds a message-bus hierarchical team from its card and configuration.
#[must_use]
pub fn hierarchical_msgbus_team(card: TeamCard, config: HierarchicalMsgbusConfig) -> BaseTeam {
    BaseTeam::with_strategy(
        card,
        config.team.clone(),
        HierarchicalMsgbusStrategy::new(config),
    )
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use openjiuwen_agent::{Agent, ReActAgent};
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;

    use super::*;
    use crate::member::TeamAgent;
    use crate::runtime::{RuntimeConfig, TeamRuntime};
    use crate::team::Team;
    use crate::{AgentProvider, MemberOptions};

    fn supervisor_card(id: &str) -> AgentCard {
        AgentCard::new(id).with_id(id)
    }

    fn config(id: &str) -> HierarchicalMsgbusConfig {
        HierarchicalMsgbusConfig::new(supervisor_card(id))
    }

    fn provider(agent: Arc<dyn Agent>) -> AgentProvider {
        Arc::new(move || TeamAgent::new(Arc::clone(&agent)))
    }

    fn context(strategy_config: &HierarchicalMsgbusConfig) -> StrategyContext {
        StrategyContext::new(
            TeamCard::new("team"),
            strategy_config.team.clone(),
            vec![],
            TeamRuntime::new(RuntimeConfig::new()),
        )
    }

    #[test]
    fn config_defaults_match_python() {
        let defaults = HierarchicalMsgbusConfig::default();
        assert_eq!(defaults.team, TeamConfig::new());
        assert_eq!(defaults.supervisor_agent.name, "");
        assert_eq!(defaults.supervisor_agent.description, "");
        assert_eq!(defaults.timeout, Some(1800.0));
    }

    #[test]
    fn config_new_sets_the_supervisor_and_keeps_defaults() {
        let card = supervisor_card("boss");
        let config = HierarchicalMsgbusConfig::new(card.clone());
        assert_eq!(config.supervisor_agent, card);
        assert_eq!(config.team, TeamConfig::new());
        assert_eq!(config.timeout, Some(1800.0));
    }

    #[test]
    fn config_json_is_flat_defaulted_and_tolerant() {
        let empty: HierarchicalMsgbusConfig = serde_json::from_value(json!({})).unwrap();
        assert_eq!(empty.team, TeamConfig::new());
        assert_eq!(empty.timeout, Some(1800.0));

        let parsed: HierarchicalMsgbusConfig = serde_json::from_value(json!({
            "max_agents": 4,
            "message_timeout": 2.5,
            "supervisor_agent": {"id": "boss", "name": "boss"},
            "timeout": 7.5,
            "surprise": true
        }))
        .unwrap();
        assert_eq!(parsed.team.max_agents, 4);
        assert_eq!(parsed.team.max_concurrent_messages, 100);
        assert!((parsed.team.message_timeout - 2.5).abs() < f64::EPSILON);
        assert_eq!(parsed.supervisor_agent.id, "boss");
        assert_eq!(parsed.timeout, Some(7.5));

        let wire = serde_json::to_value(&parsed).unwrap();
        assert_eq!(wire["max_agents"], json!(4));
        assert_eq!(wire["max_concurrent_messages"], json!(100));
        assert_eq!(wire["supervisor_agent"]["id"], json!("boss"));
        assert_eq!(wire["timeout"], json!(7.5));
        let reparsed: HierarchicalMsgbusConfig = serde_json::from_value(wire).unwrap();
        assert_eq!(reparsed, parsed);
    }

    #[test]
    fn deadline_semantics() {
        assert_eq!(deadline(None), None);
        assert_eq!(deadline(Some(0.0)), None);
        assert_eq!(deadline(Some(-3.0)), None);
        assert_eq!(deadline(Some(f64::NAN)), None);
        assert_eq!(deadline(Some(f64::INFINITY)), None);
        assert_eq!(deadline(Some(1.5)), Some(Duration::from_millis(1500)));
    }

    #[test]
    fn kind_is_the_python_strategy_name() {
        let strategy = HierarchicalMsgbusStrategy::new(config("boss"));
        assert_eq!(strategy.kind(), "hierarchical_msgbus");
    }

    #[test]
    fn check_ready_rejects_an_empty_supervisor() {
        let strategy = HierarchicalMsgbusStrategy::new(config(""));
        let ctx = context(&config(""));
        let error = strategy.check_ready(&ctx).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Config);
        assert_eq!(
            error.message(),
            "No supervisor configured in HierarchicalTeamConfig."
        );
    }

    #[test]
    fn check_ready_rejects_an_unregistered_supervisor() {
        let strategy = HierarchicalMsgbusStrategy::new(config("boss"));
        let ctx = context(&config("boss"));
        let error = strategy.check_ready(&ctx).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Config);
        assert_eq!(
            error.message(),
            "Supervisor 'boss' is not registered in runtime. Call \
             add_agent(supervisor_card, supervisor_provider) before invoke()/stream()."
        );
    }

    #[test]
    fn check_ready_accepts_a_registered_supervisor() {
        let strategy = HierarchicalMsgbusStrategy::new(config("boss"));
        let ctx = context(&config("boss"));
        let agent: Arc<dyn Agent> = Arc::new(ReActAgent::new(supervisor_card("boss")));
        ctx.runtime
            .register_agent(supervisor_card("boss"), provider(agent));
        strategy.check_ready(&ctx).unwrap();
    }

    #[test]
    fn on_member_added_adopts_the_configured_timeout() {
        let mut strategy_config = config("boss");
        strategy_config.timeout = Some(1.5);
        let strategy = HierarchicalMsgbusStrategy::new(strategy_config.clone());
        let ctx = context(&strategy_config);
        strategy.on_member_added(&ctx, &supervisor_card("boss"), &MemberOptions::default());
        assert_eq!(ctx.runtime.p2p_timeout(), Duration::from_millis(1500));

        strategy.on_member_added(&ctx, &supervisor_card("worker"), &MemberOptions::default());
        assert_eq!(ctx.runtime.p2p_timeout(), Duration::from_millis(1500));
    }

    #[test]
    fn on_member_added_without_a_deadline_keeps_the_runtime_timeout() {
        let mut strategy_config = config("boss");
        strategy_config.timeout = None;
        let strategy = HierarchicalMsgbusStrategy::new(strategy_config.clone());
        let ctx = context(&strategy_config);
        let original = ctx.runtime.p2p_timeout();
        strategy.on_member_added(&ctx, &supervisor_card("boss"), &MemberOptions::default());
        assert_eq!(ctx.runtime.p2p_timeout(), original);
    }

    #[test]
    fn team_builder_attaches_the_strategy() {
        let team = hierarchical_msgbus_team(TeamCard::new("team"), config("boss"));
        assert_eq!(team.kind(), "hierarchical_msgbus");
        assert_eq!(team.team_id(), "team");
        assert_eq!(team.config(), TeamConfig::new());
    }
}
