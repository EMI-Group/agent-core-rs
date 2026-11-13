#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Integration tests for card-driven auto-subscription on [`BaseTeam`].
//!
//! The suite pins the wiring points of `openjiuwen_multiagent::autosubscribe` on
//! a team: the `topic` broadcast and the `subscribe:` tags applied on
//! `add_agent`/`add_member`, explicit per-agent overrides, the `Merge`/`Replace`
//! /`Disabled` modes, removal cleanup, whole-plan application on `start`, and one
//! end-to-end pub/sub delivery to an auto-subscribed member. Every member is an
//! in-process closure and every delivery is loopback, so the suite is offline and
//! deterministic.

mod common;

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use openjiuwen_agent::{AgentCard, AgentOutput};
use openjiuwen_multiagent::autosubscribe::AutoSubscriptionMode;
use openjiuwen_multiagent::{AgentProvider, BaseTeam, Team, TeamAgent, TeamCard, TeamConfig};
use parking_lot::Mutex;
use serde_json::{json, Value};

use crate::common::{agent_card, FakeAgent};

/// Builds a provider that answers with a fresh echoing [`FakeAgent`].
fn echoing_provider(name: &str) -> AgentProvider {
    let name = name.to_owned();
    Arc::new(move || TeamAgent::new(FakeAgent::echoing(&name)))
}

/// A member that records how often it was invoked and with which inputs.
struct Recorder {
    calls: Arc<AtomicUsize>,
    seen: Arc<Mutex<Vec<Value>>>,
}

impl Recorder {
    /// Returns the recorded invocation count.
    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    /// Returns the recorded inputs, in invocation order.
    fn seen(&self) -> Vec<Value> {
        self.seen.lock().clone()
    }
}

/// Builds a provider whose member records each invocation into a [`Recorder`].
fn recording_provider(card: AgentCard) -> (AgentProvider, Recorder) {
    let calls = Arc::new(AtomicUsize::new(0));
    let seen = Arc::new(Mutex::new(Vec::new()));
    let provider_calls = Arc::clone(&calls);
    let provider_seen = Arc::clone(&seen);
    let name = card.name.clone();
    let provider: AgentProvider = Arc::new(move || {
        let calls = Arc::clone(&provider_calls);
        let seen = Arc::clone(&provider_seen);
        let label = name.clone();
        TeamAgent::new(FakeAgent::new(card.clone(), move |inputs, _session| {
            calls.fetch_add(1, Ordering::SeqCst);
            seen.lock().push(inputs);
            Ok(AgentOutput::answer(format!("{label}:ok")))
        }))
    });
    (provider, Recorder { calls, seen })
}

/// Builds a strategy-free team whose card declares `topic`.
fn topic_team(topic: &str) -> BaseTeam {
    BaseTeam::new(
        TeamCard::new("team").with_id("team-id").with_topic(topic),
        TeamConfig::new(),
    )
}

/// Returns the bus subscriptions of `agent_id`.
fn topics(team: &BaseTeam, agent_id: &str) -> Vec<String> {
    team.runtime().message_bus().agent_topics(agent_id)
}

/// Returns the three patterns the `team.events` topic derives for `name`.
fn broadcast(name: &str) -> Vec<String> {
    vec![
        "team.events".to_owned(),
        "team.events.*".to_owned(),
        format!("team.events.{name}"),
    ]
}

#[tokio::test]
async fn topic_broadcast_subscribes_every_member() {
    let team = topic_team("team.events");
    team.add_agent(agent_card("alice"), echoing_provider("alice"))
        .unwrap();
    team.add_agent(agent_card("bob"), echoing_provider("bob"))
        .unwrap();

    assert_eq!(team.auto_subscriptions(), AutoSubscriptionMode::Merge);
    assert_eq!(topics(&team, "alice"), broadcast("alice"));
    assert_eq!(topics(&team, "bob"), broadcast("bob"));

    let plan = team.subscription_plan();
    assert_eq!(plan.get("alice"), Some(broadcast("alice").as_slice()));
    assert_eq!(plan.subscription_count(), 6);
}

#[tokio::test]
async fn subscribe_tags_fan_out_or_target_one_agent() {
    let card = TeamCard::new("team").with_id("team-id").with_tags(vec![
        "subscribe:alerts".to_owned(),
        "subscribe:alice:direct".to_owned(),
        "discovery".to_owned(),
    ]);
    let team = BaseTeam::new(card, TeamConfig::new());

    team.add_agent(agent_card("alice"), echoing_provider("alice"))
        .unwrap();
    team.add_agent(agent_card("bob"), echoing_provider("bob"))
        .unwrap();

    assert_eq!(
        topics(&team, "alice"),
        vec!["alerts".to_owned(), "direct".to_owned()]
    );
    assert_eq!(topics(&team, "bob"), vec!["alerts".to_owned()]);
}

#[tokio::test]
async fn overrides_replace_the_derived_patterns_of_an_agent() {
    let overrides = BTreeMap::from([
        ("alice".to_owned(), vec!["only.alice".to_owned()]),
        ("bob".to_owned(), Vec::new()),
    ]);
    let team = topic_team("team.events").with_subscription_overrides(overrides.clone());
    assert_eq!(team.subscription_overrides(), overrides);

    team.add_agent(agent_card("alice"), echoing_provider("alice"))
        .unwrap();
    team.add_agent(agent_card("bob"), echoing_provider("bob"))
        .unwrap();

    assert_eq!(topics(&team, "alice"), vec!["only.alice".to_owned()]);
    assert_eq!(topics(&team, "bob"), Vec::<String>::new());

    let plan = team.subscription_plan();
    assert_eq!(
        plan.get("alice"),
        Some(["only.alice".to_owned()].as_slice())
    );
    assert_eq!(plan.get("bob"), Some([].as_slice()));
}

#[tokio::test]
async fn disabled_leaves_the_bus_registry_untouched() {
    let team = topic_team("team.events").with_auto_subscriptions(AutoSubscriptionMode::Disabled);
    assert_eq!(team.auto_subscriptions(), AutoSubscriptionMode::Disabled);
    team.runtime()
        .message_bus()
        .add_subscription("alice", "manual");

    team.add_agent(agent_card("alice"), echoing_provider("alice"))
        .unwrap();
    assert_eq!(topics(&team, "alice"), vec!["manual".to_owned()]);
    assert_eq!(team.apply_member_subscriptions("alice"), 0);
    assert_eq!(team.apply_subscriptions(), 0);

    team.start().await.unwrap();
    assert_eq!(topics(&team, "alice"), vec!["manual".to_owned()]);
    assert!(!team.subscription_plan().is_empty());
}

#[tokio::test]
async fn replace_mode_drops_a_stale_manual_pattern() {
    let team = topic_team("team.events").with_auto_subscriptions(AutoSubscriptionMode::Replace);
    team.add_agent(agent_card("alice"), echoing_provider("alice"))
        .unwrap();
    team.runtime()
        .message_bus()
        .add_subscription("alice", "stale");

    assert_eq!(team.apply_subscriptions(), 1);
    assert_eq!(topics(&team, "alice"), broadcast("alice"));
    // Applying again changes nothing.
    assert_eq!(team.apply_subscriptions(), 0);
}

#[tokio::test]
async fn start_applies_the_whole_plan_once() {
    let team = topic_team("team.events").with_auto_subscriptions(AutoSubscriptionMode::Disabled);
    team.add_agent(agent_card("alice"), echoing_provider("alice"))
        .unwrap();
    team.add_agent(agent_card("bob"), echoing_provider("bob"))
        .unwrap();
    assert_eq!(topics(&team, "alice"), Vec::<String>::new());

    team.set_auto_subscriptions(AutoSubscriptionMode::Merge);
    team.start().await.unwrap();
    assert_eq!(topics(&team, "alice"), broadcast("alice"));
    assert_eq!(topics(&team, "bob"), broadcast("bob"));

    // A second start is idempotent.
    team.start().await.unwrap();
    assert_eq!(topics(&team, "alice"), broadcast("alice"));
}

#[tokio::test]
async fn removing_a_member_drops_only_its_planned_patterns() {
    let team = topic_team("team.events");
    team.add_agent(agent_card("alice"), echoing_provider("alice"))
        .unwrap();
    team.add_agent(agent_card("bob"), echoing_provider("bob"))
        .unwrap();
    team.runtime()
        .message_bus()
        .add_subscription("bob", "bob.extra");

    team.remove_agent("alice").unwrap();
    assert_eq!(topics(&team, "alice"), Vec::<String>::new());

    let mut expected = broadcast("bob");
    expected.push("bob.extra".to_owned());
    expected.sort_unstable();
    assert_eq!(topics(&team, "bob"), expected);
}

#[tokio::test]
async fn duplicate_add_keeps_the_wiring_idempotent() {
    let team = topic_team("team.events");
    team.add_agent(agent_card("alice"), echoing_provider("alice"))
        .unwrap();
    team.add_agent(agent_card("alice"), echoing_provider("alice"))
        .unwrap();

    assert_eq!(team.agent_count(), 1);
    assert_eq!(topics(&team, "alice"), broadcast("alice"));
    assert_eq!(team.apply_member_subscriptions("alice"), 0);
}

#[tokio::test]
async fn legacy_card_without_topic_or_tags_derives_nothing() {
    let team = BaseTeam::new(
        TeamCard::new("legacy").with_id("legacy-id"),
        TeamConfig::new(),
    );
    assert!(team.subscription_plan().is_empty());

    team.add_agent(agent_card("alice"), echoing_provider("alice"))
        .unwrap();
    team.add_agent(agent_card("bob"), echoing_provider("bob"))
        .unwrap();

    assert!(team.subscription_plan().is_empty());
    assert_eq!(topics(&team, "alice"), Vec::<String>::new());
    assert_eq!(topics(&team, "bob"), Vec::<String>::new());

    team.start().await.unwrap();
    assert_eq!(team.apply_subscriptions(), 0);
    assert_eq!(topics(&team, "bob"), Vec::<String>::new());
}

#[tokio::test]
async fn publish_delivers_to_the_auto_subscribed_member() {
    let team = topic_team("team.events");
    let (sender_provider, sender) = recording_provider(agent_card("sender"));
    let (alice_provider, alice) = recording_provider(agent_card("alice"));
    team.add_agent(agent_card("sender"), sender_provider)
        .unwrap();
    team.add_agent(agent_card("alice"), alice_provider).unwrap();
    team.start().await.unwrap();

    team.publish(json!({"headline": "hi"}), "team.events", "sender", None)
        .await
        .unwrap();

    assert_eq!(alice.calls(), 1);
    assert_eq!(alice.seen(), vec![json!({"headline": "hi"})]);
    // The sender is a member of the same broadcast topic, so it sees the
    // publication too; that is the expected fan-out, not a duplicate delivery.
    assert_eq!(sender.calls(), 1);
}
