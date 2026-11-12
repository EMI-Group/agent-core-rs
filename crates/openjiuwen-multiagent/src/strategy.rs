//! The strategy seam: how a team turns one input into member invocations.
//!
//! Python parity: the strategy classes of `openjiuwen/core/multi_agent/`
//! (`handoff`, `hierarchical_msgbus`, `hierarchical_tools`). Rust models them as
//! objects implementing [`TeamStrategy`]: the team owns the card, configuration,
//! member registry, and runtime, and the strategy receives a
//! [`StrategyContext`] snapshot for every call.
//!
//! Every hook except [`TeamStrategy::invoke`] and [`TeamStrategy::stream`] has a
//! default, so a strategy implements only what it needs. The defaults are
//! deliberate no-ops: `setup` does nothing, `check_ready` accepts the team, and
//! the membership callbacks ignore the event.

use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::BoxStream;
use openjiuwen_agent::AgentCard;
use openjiuwen_common::error::Result;
use openjiuwen_session::AgentTeamSession;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::card::TeamCard;
use crate::config::TeamConfig;
use crate::runtime::TeamRuntime;

/// Per-member wiring options passed when a member is added.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct MemberOptions {
    /// Parent member id for hierarchical strategies (the supervisor or root).
    pub parent_agent_id: Option<String>,
}

/// The immutable view of a team handed to its strategy.
///
/// `members` is a snapshot of the member cards at call time; the live registry
/// lives in the [`runtime`](Self::runtime).
#[derive(Clone, Debug)]
pub struct StrategyContext {
    /// The team card (id, name, and the member list as of this snapshot).
    pub card: TeamCard,
    /// The team limits.
    pub config: TeamConfig,
    /// The member cards, in registration order.
    pub members: Vec<AgentCard>,
    /// The team runtime that dispatches member invocations.
    pub runtime: Arc<TeamRuntime>,
}

impl StrategyContext {
    /// Builds a context snapshot.
    #[must_use]
    pub fn new(
        card: TeamCard,
        config: TeamConfig,
        members: Vec<AgentCard>,
        runtime: Arc<TeamRuntime>,
    ) -> Self {
        Self {
            card,
            config,
            members,
            runtime,
        }
    }

    /// Returns the member ids in registration order.
    #[must_use]
    pub fn member_ids(&self) -> Vec<String> {
        self.members.iter().map(|card| card.id.clone()).collect()
    }
}

/// The team execution strategy.
///
/// Only [`invoke`](Self::invoke) and [`stream`](Self::stream) are required; the
/// rest are lifecycle hooks with harmless defaults.
#[async_trait]
pub trait TeamStrategy: Send + Sync {
    /// Returns the strategy kind (`"handoff"`, `"hierarchical_msgbus"`, …).
    fn kind(&self) -> &str;

    /// Called after a member was added to the team.
    ///
    /// Default: ignore the event.
    fn on_member_added(&self, ctx: &StrategyContext, card: &AgentCard, options: &MemberOptions) {
        let _ = (ctx, card, options);
    }

    /// Called after a member was removed from the team.
    ///
    /// Default: ignore the event.
    fn on_member_removed(&self, ctx: &StrategyContext, agent_id: &str) {
        let _ = (ctx, agent_id);
    }

    /// Validates that the team is complete enough to run.
    ///
    /// Default: accept everything.
    ///
    /// # Errors
    ///
    /// Implementations return [`openjiuwen_common::error::Error::Config`] when
    /// required configuration (a supervisor, a root agent) is missing.
    fn check_ready(&self, ctx: &StrategyContext) -> Result<()> {
        let _ = ctx;
        Ok(())
    }

    /// Wires the strategy's internal agents and hierarchy before the first run.
    ///
    /// Default: nothing to set up.
    ///
    /// # Errors
    ///
    /// Implementations propagate registration and subscription failures.
    async fn setup(&self, ctx: &StrategyContext) -> Result<()> {
        let _ = ctx;
        Ok(())
    }

    /// Runs one team invocation to completion.
    ///
    /// # Errors
    ///
    /// Propagates member, transport, and deadline failures.
    async fn invoke(
        &self,
        ctx: &StrategyContext,
        input: Value,
        session: Option<Arc<AgentTeamSession>>,
    ) -> Result<Value>;

    /// Runs one team invocation as a stream of wire frames.
    ///
    /// Frames are the strategy's own format; the team session driver forwards
    /// them unchanged.
    fn stream(
        &self,
        ctx: &StrategyContext,
        input: Value,
        session: Option<Arc<AgentTeamSession>>,
    ) -> BoxStream<'static, Result<Value>>;
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use futures::StreamExt;
    use serde_json::json;

    use super::*;
    use crate::{RuntimeConfig, TeamCard};

    /// A minimal strategy: the defaults apply, `invoke` echoes, `stream` yields
    /// exactly one frame.
    struct EchoStrategy;

    #[async_trait]
    impl TeamStrategy for EchoStrategy {
        fn kind(&self) -> &'static str {
            "echo"
        }

        async fn invoke(
            &self,
            _ctx: &StrategyContext,
            input: Value,
            _session: Option<Arc<AgentTeamSession>>,
        ) -> Result<Value> {
            Ok(input)
        }

        fn stream(
            &self,
            _ctx: &StrategyContext,
            input: Value,
            _session: Option<Arc<AgentTeamSession>>,
        ) -> BoxStream<'static, Result<Value>> {
            futures::stream::once(async move { Ok(input) }).boxed()
        }
    }

    fn context() -> StrategyContext {
        StrategyContext::new(
            TeamCard::new("team"),
            TeamConfig::new(),
            vec![],
            TeamRuntime::new(RuntimeConfig::new()),
        )
    }

    #[test]
    fn member_options_default_to_no_parent() {
        let options = MemberOptions::default();
        assert_eq!(options.parent_agent_id, None);
        let value = serde_json::to_value(&options).unwrap();
        assert_eq!(value, json!({"parent_agent_id": null}));
        let parsed: MemberOptions =
            serde_json::from_value(json!({"parent_agent_id": "boss"})).unwrap();
        assert_eq!(parsed.parent_agent_id.as_deref(), Some("boss"));
    }

    #[test]
    fn context_exposes_member_ids() {
        let mut ctx = context();
        ctx.members = vec![
            AgentCard::new("a").with_id("a"),
            AgentCard::new("b").with_id("b"),
        ];
        assert_eq!(ctx.member_ids(), vec!["a".to_string(), "b".to_string()]);
        assert_eq!(ctx.card.name, "team");
        assert_eq!(ctx.runtime.team_id(), "default");
    }

    #[tokio::test]
    async fn default_hooks_are_no_ops() {
        let strategy = EchoStrategy;
        let ctx = context();
        let card = AgentCard::new("a").with_id("a");
        strategy.on_member_added(&ctx, &card, &MemberOptions::default());
        strategy.on_member_removed(&ctx, "a");
        strategy.check_ready(&ctx).unwrap();
        strategy.setup(&ctx).await.unwrap();
        assert_eq!(strategy.kind(), "echo");
    }

    #[tokio::test]
    async fn invoke_and_stream_are_reachable_through_the_trait_object() {
        let strategy: Arc<dyn TeamStrategy> = Arc::new(EchoStrategy);
        let ctx = context();
        let input = json!({"query": "hi"});
        assert_eq!(
            strategy.invoke(&ctx, input.clone(), None).await.unwrap(),
            input
        );

        let mut stream = strategy.stream(&ctx, input.clone(), None);
        assert_eq!(stream.next().await.unwrap().unwrap(), input);
        assert!(stream.next().await.is_none());
    }
}
