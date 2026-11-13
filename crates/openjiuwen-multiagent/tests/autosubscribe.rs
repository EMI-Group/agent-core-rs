#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Integration tests for card-driven auto-subscription.
//!
//! The suite pins the derivation rules (broadcast, directed topics, tag-carried
//! capability topics, override replacement), the deterministic sorted/deduped
//! shape of a plan, and the three reconciliation modes against a real
//! [`TeamRuntime`] bus registry. No bus worker is started and no agent is
//! registered: logical subscriptions live in the registry the runtime exposes,
//! so every case is synchronous, offline and deterministic.
use std::collections::BTreeMap;
use std::sync::Arc;

use openjiuwen_agent::AgentCard;
use openjiuwen_multiagent::autosubscribe::{
    apply, apply_agent, derive, derive_from_event_card, derive_with_overrides, derived_for_member,
    AutoSubscriptionMode, AutoSubscriptionPlan,
};
use openjiuwen_multiagent::{EventDrivenTeamCard, RuntimeConfig, TeamCard, TeamRuntime};
use serde_json::json;

/// Builds a member card whose id and name are `name`.
fn member(name: &str) -> AgentCard {
    AgentCard::new(name).with_id(name)
}

/// Builds a team card with `topic`, `tags` and members named by `members`.
fn team(topic: &str, tags: &[&str], members: &[&str]) -> TeamCard {
    TeamCard::new("team")
        .with_id("team")
        .with_topic(topic)
        .with_tags(tags.iter().map(|tag| (*tag).to_owned()).collect())
        .with_agent_cards(members.iter().map(|name| member(name)).collect())
}

/// Builds a runtime whose subscriptions are used without starting the bus.
fn runtime() -> Arc<TeamRuntime> {
    TeamRuntime::new(RuntimeConfig::default())
}

#[test]
fn broadcast_topic_is_sorted_and_derived_for_every_member() {
    let plan = derive(&team("alerts", &[], &["writer", "reader"]));
    assert_eq!(
        plan.agents(),
        vec!["reader".to_owned(), "writer".to_owned()]
    );
    assert_eq!(
        plan.get("reader"),
        Some(
            &[
                "alerts".to_owned(),
                "alerts.*".to_owned(),
                "alerts.reader".to_owned()
            ][..]
        )
    );
    assert_eq!(plan.subscription_count(), 6);
}

#[test]
fn tag_capabilities_fan_out_to_all_or_one_member() {
    let card = team(
        "alerts",
        &[
            "subscribe:audit",
            "subscribe:writer:alerts.drafts",
            "subscribe:",
        ],
        &["writer", "reader"],
    );
    let plan = derive(&card);
    assert_eq!(
        plan.get("reader"),
        Some(
            &[
                "alerts".to_owned(),
                "alerts.*".to_owned(),
                "alerts.reader".to_owned(),
                "audit".to_owned()
            ][..]
        )
    );
    assert_eq!(
        plan.get("writer"),
        Some(
            &[
                "alerts".to_owned(),
                "alerts.*".to_owned(),
                "alerts.drafts".to_owned(),
                "alerts.writer".to_owned(),
                "audit".to_owned()
            ][..]
        )
    );
    assert_eq!(
        derived_for_member(&card, "writer"),
        plan.get("writer").unwrap()
    );
    assert_eq!(derived_for_member(&card, "ghost"), Vec::<String>::new());
}

#[test]
fn overrides_replace_the_derived_set_and_keep_unknown_agents() {
    let card = team("alerts", &["subscribe:audit"], &["writer"]);
    let overrides = BTreeMap::from([
        (
            "writer".to_owned(),
            vec!["only".to_owned(), "only".to_owned()],
        ),
        ("ghost".to_owned(), vec!["phantom".to_owned()]),
    ]);
    let plan = derive_with_overrides(&card, &overrides);
    assert_eq!(plan.get("writer"), Some(&["only".to_owned()][..]));
    assert_eq!(plan.get("ghost"), Some(&["phantom".to_owned()][..]));
    assert_eq!(plan.subscription_count(), 2);
}

#[test]
fn event_driven_card_subscriptions_are_the_overrides() {
    let card = EventDrivenTeamCard {
        card: team("alerts", &[], &["writer"]),
        subscriptions: BTreeMap::from([(
            "writer".to_owned(),
            vec!["alerts.reader".to_owned(), "audit".to_owned()],
        )]),
    };
    let plan = derive_from_event_card(&card);
    assert_eq!(
        plan.get("writer"),
        Some(&["alerts.reader".to_owned(), "audit".to_owned()][..])
    );
    assert!(derive(&card.card).get("writer").is_some());
}

#[test]
fn plan_round_trips_through_json_and_exposes_a_schema() {
    let plan = derive(&team("alerts", &[], &["writer"]));
    let value = serde_json::to_value(&plan).unwrap();
    assert_eq!(
        value["patterns"]["writer"],
        json!(["alerts", "alerts.*", "alerts.writer"])
    );
    let restored: AutoSubscriptionPlan = serde_json::from_value(value).unwrap();
    assert_eq!(restored, plan);
    let defaults: AutoSubscriptionPlan = serde_json::from_value(json!({"other": 1})).unwrap();
    assert!(defaults.is_empty());

    let schema = serde_json::to_value(schemars::schema_for!(AutoSubscriptionPlan)).unwrap();
    assert!(schema.to_string().contains("patterns"));
    let mode = serde_json::to_value(schemars::schema_for!(AutoSubscriptionMode)).unwrap();
    let mode = mode.to_string();
    assert!(mode.contains("\"merge\""));
    assert!(mode.contains("\"replace\""));
    assert!(mode.contains("\"disabled\""));
}

#[test]
fn merge_adds_only_the_missing_patterns() {
    let runtime = runtime();
    let bus = runtime.message_bus();
    let plan = derive(&team("alerts", &[], &["writer", "reader"]));
    bus.add_subscription("writer", "alerts");

    assert_eq!(apply(&runtime, &plan, AutoSubscriptionMode::Merge), 5);
    assert_eq!(
        bus.agent_topics("writer"),
        vec![
            "alerts".to_owned(),
            "alerts.*".to_owned(),
            "alerts.writer".to_owned()
        ]
    );
    assert_eq!(
        bus.agent_topics("reader"),
        vec![
            "alerts".to_owned(),
            "alerts.*".to_owned(),
            "alerts.reader".to_owned()
        ]
    );
    // Idempotent: nothing is counted twice.
    assert_eq!(apply(&runtime, &plan, AutoSubscriptionMode::Merge), 0);
    assert_eq!(
        bus.subscribers("alerts"),
        vec!["reader".to_owned(), "writer".to_owned()]
    );
    // The `*` pattern really matches a concrete directed topic.
    assert_eq!(
        bus.subscribers("alerts.anything"),
        vec!["reader".to_owned(), "writer".to_owned()]
    );
}

#[test]
fn replace_removes_stale_patterns_and_counts_both_sides() {
    let runtime = runtime();
    let bus = runtime.message_bus();
    let plan = derive(&team("alerts", &[], &["writer"]));
    bus.add_subscription("writer", "alerts");
    bus.add_subscription("writer", "stale");

    // Two patterns added (`alerts.*`, `alerts.writer`), one removed (`stale`).
    assert_eq!(apply(&runtime, &plan, AutoSubscriptionMode::Replace), 3);
    assert_eq!(
        bus.agent_topics("writer"),
        vec![
            "alerts".to_owned(),
            "alerts.*".to_owned(),
            "alerts.writer".to_owned()
        ]
    );
    assert_eq!(apply(&runtime, &plan, AutoSubscriptionMode::Replace), 0);
}

#[test]
fn replace_leaves_agents_outside_the_plan_alone() {
    let runtime = runtime();
    let bus = runtime.message_bus();
    let plan = derive(&team("alerts", &[], &["writer"]));
    bus.add_subscription("unrelated", "keeper");

    assert_eq!(apply(&runtime, &plan, AutoSubscriptionMode::Replace), 3);
    assert_eq!(bus.agent_topics("unrelated"), vec!["keeper".to_owned()]);
}

#[test]
fn disabled_never_touches_the_bus() {
    let runtime = runtime();
    let bus = runtime.message_bus();
    let plan = derive(&team("alerts", &[], &["writer"]));
    assert_eq!(apply(&runtime, &plan, AutoSubscriptionMode::Disabled), 0);
    assert_eq!(
        apply_agent(&runtime, &plan, "writer", AutoSubscriptionMode::Disabled),
        0
    );
    assert_eq!(bus.agent_topics("writer"), Vec::<String>::new());
    assert!(plan.patterns().contains_key("writer"));
}

#[test]
fn apply_agent_only_touches_one_agent() {
    let runtime = runtime();
    let bus = runtime.message_bus();
    let plan = derive(&team("alerts", &[], &["writer", "reader"]));

    assert_eq!(
        apply_agent(&runtime, &plan, "writer", AutoSubscriptionMode::Merge),
        3
    );
    assert_eq!(
        apply_agent(&runtime, &plan, "writer", AutoSubscriptionMode::Merge),
        0
    );
    assert_eq!(
        bus.agent_topics("writer"),
        vec![
            "alerts".to_owned(),
            "alerts.*".to_owned(),
            "alerts.writer".to_owned()
        ]
    );
    assert_eq!(bus.agent_topics("reader"), Vec::<String>::new());
}

#[test]
fn apply_agent_replace_clears_an_agent_absent_from_the_plan() {
    let runtime = runtime();
    let bus = runtime.message_bus();
    bus.add_subscription("ghost", "a");
    bus.add_subscription("ghost", "b");
    let plan = derive(&team("alerts", &[], &["writer"]));

    assert_eq!(
        apply_agent(&runtime, &plan, "ghost", AutoSubscriptionMode::Replace),
        2
    );
    assert_eq!(bus.agent_topics("ghost"), Vec::<String>::new());
    assert_eq!(
        apply_agent(&runtime, &plan, "", AutoSubscriptionMode::Merge),
        0
    );
}

#[test]
fn empty_patterns_are_stored_but_never_applied() {
    let runtime = runtime();
    let bus = runtime.message_bus();
    let mut plan = AutoSubscriptionPlan::new();
    plan.insert("writer", vec![String::new()]);
    assert_eq!(plan.subscription_count(), 1);

    assert_eq!(apply(&runtime, &plan, AutoSubscriptionMode::Merge), 0);
    assert_eq!(bus.agent_topics("writer"), Vec::<String>::new());
}

#[test]
fn empty_plan_and_empty_card_are_no_ops() {
    let runtime = runtime();
    let plan = derive(&TeamCard::default());
    assert!(plan.is_empty());
    assert_eq!(apply(&runtime, &plan, AutoSubscriptionMode::Merge), 0);
    assert_eq!(
        apply_agent(&runtime, &plan, "writer", AutoSubscriptionMode::Replace),
        0
    );
}
