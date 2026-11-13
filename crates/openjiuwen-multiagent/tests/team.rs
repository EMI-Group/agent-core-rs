#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Integration tests for the [`Team`] seam and [`BaseTeam`].
//!
//! The suite registers [`common::FakeAgent`] members through a team and checks
//! the membership contract (queries, duplicate no-ops, the `max_agents` cap,
//! removal by id or name), the exact membership-validation messages of
//! `send`/`publish`, transport start/stop idempotence, and the strategy seam
//! (a custom strategy drives invoke/stream, a strategy-free team is
//! unsupported). Every member is an in-process closure, so the suite is offline
//! and deterministic.

mod common;

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::StreamExt;
use openjiuwen_agent::AgentCard;
use openjiuwen_common::error::{ErrorKind, Result};
use openjiuwen_multiagent::autosubscribe::AutoSubscriptionMode;
use openjiuwen_multiagent::{
    AgentProvider, BaseTeam, MemberOptions, StrategyContext, Team, TeamAgent, TeamCard, TeamConfig,
    TeamStrategy,
};
use openjiuwen_session::AgentTeamSession;
use parking_lot::Mutex;
use serde_json::{json, Value};

use crate::common::{agent_card, FakeAgent};

/// Builds a provider that answers with a fresh echoing [`FakeAgent`].
fn echoing_provider(name: &str) -> AgentProvider {
    let name = name.to_owned();
    Arc::new(move || TeamAgent::new(FakeAgent::echoing(&name)))
}

/// A strategy that echoes the input back under a marker, so a test can tell the
/// team used the custom strategy rather than the unsupported default.
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
        Ok(json!({"strategy": "echo", "input": input}))
    }

    fn stream(
        &self,
        _ctx: &StrategyContext,
        input: Value,
        _session: Option<Arc<AgentTeamSession>>,
    ) -> BoxStream<'static, Result<Value>> {
        futures::stream::once(async move { Ok(json!({"strategy": "echo", "input": input})) })
            .boxed()
    }
}

/// Builds a team named `name` that executes through [`EchoStrategy`].
fn echo_team(name: &str) -> BaseTeam {
    BaseTeam::with_strategy(
        TeamCard::new(name).with_id(format!("{name}-id")),
        TeamConfig::new(),
        Arc::new(EchoStrategy),
    )
}

#[tokio::test]
async fn add_agent_and_queries() {
    let team = echo_team("team-1");
    assert_eq!(team.agent_count(), 0);
    assert_eq!(team.list_agents(), Vec::<String>::new());

    team.add_agent(agent_card("a"), echoing_provider("a"))
        .unwrap();
    team.add_agent(agent_card("b"), echoing_provider("b"))
        .unwrap();

    assert_eq!(team.agent_count(), 2);
    assert_eq!(team.list_agents(), vec!["a".to_string(), "b".to_string()]);
    assert_eq!(
        team.get_agent_card("a").map(|card| card.name),
        Some("a".to_string())
    );
    assert!(team.get_agent_card("ghost").is_none());

    let snapshot = team.card();
    assert_eq!(snapshot.id, "team-1-id");
    assert_eq!(snapshot.agent_cards, vec![agent_card("a"), agent_card("b")]);
    assert_eq!(team.team_id(), "team-1");
    assert_eq!(team.kind(), "echo");
    assert_eq!(team.config(), TeamConfig::new());
    assert_eq!(team.runtime().team_id(), "team-1-id");
    team.check_ready().unwrap();
}

#[tokio::test]
async fn duplicate_add_is_a_noop() {
    let team = echo_team("team");
    team.add_agent(agent_card("a"), echoing_provider("a"))
        .unwrap();
    team.add_agent(agent_card("a"), echoing_provider("a"))
        .unwrap();

    assert_eq!(team.agent_count(), 1);
    assert_eq!(team.card().agent_cards.len(), 1);
    assert_eq!(team.list_agents(), vec!["a".to_string()]);
}

#[tokio::test]
async fn max_agents_is_enforced() {
    let team = BaseTeam::with_strategy(
        TeamCard::new("team"),
        TeamConfig::default().configure_max_agents(1),
        Arc::new(EchoStrategy),
    );
    team.add_agent(agent_card("a"), echoing_provider("a"))
        .unwrap();

    let error = team
        .add_agent(agent_card("b"), echoing_provider("b"))
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(error.message(), "Agent count exceeds max_agents (1)");
    assert_eq!(team.agent_count(), 1);

    // Re-adding the existing member is still a no-op, not a cap violation.
    team.add_agent(agent_card("a"), echoing_provider("a"))
        .unwrap();
    assert_eq!(team.agent_count(), 1);
}

#[tokio::test]
async fn remove_agent_by_id_and_name() {
    let team = echo_team("team");
    team.add_agent(agent_card("a"), echoing_provider("a"))
        .unwrap();
    team.add_agent(
        AgentCard::new("worker").with_id("worker-id"),
        echoing_provider("worker"),
    )
    .unwrap();
    assert_eq!(team.agent_count(), 2);

    team.remove_agent("a").unwrap();
    assert_eq!(team.agent_count(), 1);
    assert_eq!(team.list_agents(), vec!["worker-id".to_string()]);

    // `worker` is the card name of `worker-id`.
    team.remove_agent("worker").unwrap();
    assert_eq!(team.agent_count(), 0);
    assert_eq!(team.card().agent_cards, Vec::<AgentCard>::new());

    // Unknown members are a no-op.
    team.remove_agent("ghost").unwrap();
    assert_eq!(team.agent_count(), 0);
}

#[tokio::test]
async fn remove_member_alias_drops_the_member() {
    let team = BaseTeam::new(TeamCard::new("team"), TeamConfig::new());
    team.add_agent(agent_card("alpha"), echoing_provider("alpha"))
        .unwrap();
    team.add_agent(agent_card("beta"), echoing_provider("beta"))
        .unwrap();
    assert_eq!(team.agent_count(), 2);
    team.remove_member("alpha").unwrap();
    assert_eq!(team.agent_count(), 1);
    assert!(team.get_agent_card("alpha").is_none());
    assert_eq!(team.list_agents(), vec!["beta".to_string()]);
    // Unknown members are a no-op.
    team.remove_member("alpha").unwrap();
    assert_eq!(team.agent_count(), 1);
}

#[tokio::test]
async fn send_and_publish_validate_membership() {
    let team = echo_team("team");
    team.add_agent(agent_card("a"), echoing_provider("a"))
        .unwrap();
    team.add_agent(agent_card("b"), echoing_provider("b"))
        .unwrap();

    let error = team
        .send(json!({"query": "hi"}), "b", "ghost", None, None)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(error.message(), "Sender 'ghost' not found in team 'team'");

    let error = team
        .send(json!({"query": "hi"}), "ghost", "a", None, None)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(
        error.message(),
        "Recipient 'ghost' not found in team 'team'"
    );

    let error = team
        .publish(json!({"headline": "hi"}), "news", "ghost", None)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(error.message(), "Sender 'ghost' not found in team 'team'");

    let reply = team
        .send(
            json!({"query": "hi"}),
            "b",
            "a",
            None,
            Some(Duration::from_secs(5)),
        )
        .await
        .unwrap();
    assert_eq!(reply["output"], "b:hi");
    assert_eq!(reply["result_type"], "answer");

    // An empty subscriber set is a no-op, like the runtime's publish.
    team.publish(json!({"headline": "hi"}), "news", "a", None)
        .await
        .unwrap();
}

#[tokio::test]
async fn start_stop_are_idempotent() {
    let team = echo_team("team");
    team.start().await.unwrap();
    team.start().await.unwrap();
    team.stop().await.unwrap();
    team.stop().await.unwrap();
    team.start().await.unwrap();
    team.stop().await.unwrap();
}

#[tokio::test]
async fn custom_strategy_is_used() {
    let team = echo_team("team");
    assert_eq!(team.kind(), "echo");
    assert_eq!(team.strategy().kind(), "echo");

    let input = json!({"query": "hi"});
    let result = team.invoke(input.clone(), None).await.unwrap();
    assert_eq!(result, json!({"strategy": "echo", "input": input}));

    let mut frames = team.stream(input.clone(), None);
    assert_eq!(
        frames.next().await.unwrap().unwrap(),
        json!({"strategy": "echo", "input": input})
    );
    assert!(frames.next().await.is_none());
}

#[tokio::test]
async fn strategy_free_team_is_unsupported() {
    let team = BaseTeam::new(TeamCard::new("team"), TeamConfig::new());
    assert_eq!(team.kind(), "none");

    let error = team.invoke(json!({"query": "hi"}), None).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert_eq!(error.message(), "team has no strategy");

    let mut frames = team.stream(json!({"query": "hi"}), None);
    let error = frames.next().await.unwrap().unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert_eq!(error.message(), "team has no strategy");
    assert!(frames.next().await.is_none());
}

#[tokio::test]
async fn auto_subscriptions_default_to_merge_and_derive_nothing() {
    let team = echo_team("team");
    assert_eq!(team.auto_subscriptions(), AutoSubscriptionMode::Merge);
    assert!(team.subscription_overrides().is_empty());
    assert!(team.subscription_plan().is_empty());

    team.add_agent(agent_card("a"), echoing_provider("a"))
        .unwrap();
    assert!(team.subscription_plan().is_empty());
    assert_eq!(team.apply_subscriptions(), 0);
    team.start().await.unwrap();
    assert_eq!(
        team.runtime().message_bus().agent_topics("a"),
        Vec::<String>::new()
    );
}

#[tokio::test]
async fn add_member_forwards_options_to_the_strategy() {
    /// Records the parent option of every added member.
    struct RecordingStrategy {
        parents: Mutex<Vec<Option<String>>>,
    }

    #[async_trait]
    impl TeamStrategy for RecordingStrategy {
        fn kind(&self) -> &'static str {
            "recording"
        }

        fn on_member_added(
            &self,
            _ctx: &StrategyContext,
            _card: &AgentCard,
            options: &MemberOptions,
        ) {
            self.parents.lock().push(options.parent_agent_id.clone());
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

    let strategy = Arc::new(RecordingStrategy {
        parents: Mutex::new(Vec::new()),
    });
    let team = BaseTeam::with_strategy(
        TeamCard::new("team"),
        TeamConfig::new(),
        Arc::clone(&strategy) as Arc<dyn TeamStrategy>,
    );
    team.add_member(
        agent_card("child"),
        echoing_provider("child"),
        MemberOptions {
            parent_agent_id: Some("boss".to_owned()),
        },
    )
    .unwrap();
    team.add_agent(agent_card("other"), echoing_provider("other"))
        .unwrap();

    assert_eq!(team.agent_count(), 2);
    assert_eq!(team.kind(), "recording");
    assert_eq!(
        strategy.parents.lock().clone(),
        vec![Some("boss".to_owned()), None]
    );
}
