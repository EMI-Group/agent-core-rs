#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Integration tests for per-agent inbox filters.
//!
//! Offline and deterministic: filters are pure data, members are in-process
//! fakes, and deliveries run over the default in-process transport. The suite
//! pins every rule (deny-wins, allow-list, topic patterns, metadata, session),
//! the registry's default-accept behaviour, and the runtime enforcement points:
//! a rejected P2P delivery fails the sender with a validation error, a rejected
//! pub/sub delivery is dropped without failing the publisher, and removing the
//! filter restores delivery.
mod common;

use std::sync::Arc;

use openjiuwen_agent::AgentOutput;
use openjiuwen_common::error::ErrorKind;
use openjiuwen_multiagent::{
    AgentProvider, InboxFilter, InboxFilterRegistry, MessageEnvelope, RuntimeConfig, TeamAgent,
    TeamRuntime,
};
use parking_lot::Mutex;
use serde_json::json;

use crate::common::{agent_card, FakeAgent};

/// Builds a provider that counts its deliveries.
fn counting_provider(name: &str, seen: &Arc<Mutex<usize>>) -> AgentProvider {
    let seen = Arc::clone(seen);
    let name = name.to_owned();
    Arc::new(move || {
        let seen = Arc::clone(&seen);
        TeamAgent::new(FakeAgent::new(
            agent_card(&name),
            move |_inputs, _session| {
                *seen.lock() += 1;
                Ok(AgentOutput::answer("ok"))
            },
        ))
    })
}

#[test]
fn default_filter_accepts_everything() {
    let filter = InboxFilter::new();
    assert!(filter.accepts(&MessageEnvelope::new(json!("hi"))));
    assert!(filter.accepts(&MessageEnvelope::p2p(json!("hi"), "a", "b")));
    assert!(filter.accepts(&MessageEnvelope::pubsub(json!("hi"), "a", "news")));
}

#[test]
fn allow_senders_restricts_deliveries() {
    let filter = InboxFilter::new().allow_senders(["ally", "boss"]);
    assert!(filter.accepts(&MessageEnvelope::p2p(json!("hi"), "ally", "worker")));
    assert!(filter.accepts(&MessageEnvelope::p2p(json!("hi"), "boss", "worker")));
    assert!(!filter.accepts(&MessageEnvelope::p2p(json!("hi"), "troll", "worker")));
    assert!(!filter.accepts(&MessageEnvelope::new(json!("hi"))));

    let filter = InboxFilter::new().allow_senders(["a"]).allow_senders(["b"]);
    assert!(filter.accepts(&MessageEnvelope::p2p(json!("hi"), "a", "c")));
    assert!(filter.accepts(&MessageEnvelope::p2p(json!("hi"), "b", "c")));
}

#[test]
fn deny_senders_wins_over_the_allow_list() {
    let filter = InboxFilter::new()
        .allow_senders(["ally", "troll"])
        .deny_senders(["troll"]);
    assert!(filter.accepts(&MessageEnvelope::p2p(json!("hi"), "ally", "worker")));
    assert!(!filter.accepts(&MessageEnvelope::p2p(json!("hi"), "troll", "worker")));
}

#[test]
fn topic_patterns_are_matched_with_wildcards() {
    let filter = InboxFilter::new().allow_topics(["news.*", "alerts"]);
    assert!(filter.accepts(&MessageEnvelope::pubsub(json!("hi"), "a", "news.tech")));
    assert!(filter.accepts(&MessageEnvelope::pubsub(json!("hi"), "a", "alerts")));
    assert!(!filter.accepts(&MessageEnvelope::pubsub(json!("hi"), "a", "other")));
    assert!(!filter.accepts(&MessageEnvelope::p2p(json!("hi"), "a", "worker")));
}

#[test]
fn metadata_pairs_must_match_exactly() {
    let filter = InboxFilter::new().require_metadata("channel", "ops");
    assert!(filter.accepts(
        &MessageEnvelope::p2p(json!("hi"), "a", "worker").with_metadata("channel", json!("ops"))
    ));
    assert!(!filter.accepts(
        &MessageEnvelope::p2p(json!("hi"), "a", "worker").with_metadata("channel", json!("dev"))
    ));
    assert!(!filter.accepts(&MessageEnvelope::p2p(json!("hi"), "a", "worker")));

    let filter = InboxFilter::new()
        .require_metadata("channel", "ops")
        .require_metadata("tenant", "t1");
    assert!(filter.accepts(
        &MessageEnvelope::p2p(json!("hi"), "a", "worker")
            .with_metadata("channel", json!("ops"))
            .with_metadata("tenant", json!("t1"))
    ));
    assert!(!filter.accepts(
        &MessageEnvelope::p2p(json!("hi"), "a", "worker").with_metadata("channel", json!("ops"))
    ));
}

#[test]
fn require_session_rejects_sessionless_envelopes() {
    let filter = InboxFilter::new().require_session();
    assert!(!filter.accepts(&MessageEnvelope::p2p(json!("hi"), "a", "worker")));
    assert!(filter.accepts(&MessageEnvelope::p2p(json!("hi"), "a", "worker").with_session("s1")));
}

#[test]
fn registry_defaults_to_accept_and_tracks_agents() {
    let registry = InboxFilterRegistry::new();
    assert_eq!(registry.len(), 0);
    assert!(registry.accepts(
        "worker",
        &MessageEnvelope::p2p(json!("hi"), "troll", "worker")
    ));

    registry.set("worker", InboxFilter::new().deny_senders(["troll"]));
    assert_eq!(registry.len(), 1);
    assert_ne!(registry.len(), 0);
    assert_eq!(registry.agents(), vec!["worker"]);
    assert!(registry.get("worker").is_some());
    assert!(!registry.accepts(
        "worker",
        &MessageEnvelope::p2p(json!("hi"), "troll", "worker")
    ));
    assert!(registry.accepts(
        "worker",
        &MessageEnvelope::p2p(json!("hi"), "ally", "worker")
    ));
    assert!(registry.accepts(
        "other",
        &MessageEnvelope::p2p(json!("hi"), "troll", "other")
    ));

    assert!(registry.remove("worker"));
    assert!(!registry.remove("worker"));
    assert!(registry.accepts(
        "worker",
        &MessageEnvelope::p2p(json!("hi"), "troll", "worker")
    ));

    registry.set("a", InboxFilter::new());
    registry.set("b", InboxFilter::new());
    registry.clear();
    assert_eq!(registry.len(), 0);
    assert_eq!(registry.agents(), Vec::<String>::new());
}

#[tokio::test]
async fn runtime_rejects_blocked_p2p_senders() {
    let runtime = TeamRuntime::new(RuntimeConfig::new().with_team_id("t"));
    let seen = Arc::new(Mutex::new(0usize));
    runtime.register_agent(agent_card("worker"), counting_provider("worker", &seen));

    runtime.set_inbox_filter("worker", InboxFilter::new().deny_senders(["boss"]));
    let error = runtime
        .send(json!({"query": "hi"}), "worker", "boss", None, None)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(
        error.message(),
        "inbox of 'worker' rejected the message from 'boss'"
    );
    assert_eq!(*seen.lock(), 0);

    // A sender the filter allows still delivers.
    runtime
        .send(json!({"query": "hi"}), "worker", "ally", None, None)
        .await
        .unwrap();
    assert_eq!(*seen.lock(), 1);

    // An allow-list rejects every other sender.
    runtime.set_inbox_filter("worker", InboxFilter::new().allow_senders(["ally"]));
    let error = runtime
        .send(json!({"query": "hi"}), "worker", "boss", None, None)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);

    // Removing the filter restores delivery.
    assert!(runtime.remove_inbox_filter("worker"));
    runtime
        .send(json!({"query": "hi"}), "worker", "boss", None, None)
        .await
        .unwrap();
    assert_eq!(*seen.lock(), 2);
    assert!(runtime.inbox_filter("worker").is_none());
}

#[tokio::test]
async fn runtime_drops_filtered_pubsub_without_failing_the_publisher() {
    let runtime = TeamRuntime::new(RuntimeConfig::new().with_team_id("t"));
    let seen = Arc::new(Mutex::new(0usize));
    runtime.register_agent(agent_card("worker"), counting_provider("worker", &seen));
    runtime.subscribe("worker", "news").await.unwrap();
    runtime.set_inbox_filter(
        "worker",
        InboxFilter::new()
            .deny_senders(["troll"])
            .allow_topics(["news*"]),
    );

    runtime
        .publish(json!({"query": "hi"}), "news", "troll", None)
        .await
        .unwrap();
    assert_eq!(*seen.lock(), 0);

    runtime
        .publish(json!({"query": "hi"}), "news", "ally", None)
        .await
        .unwrap();
    assert_eq!(*seen.lock(), 1);

    assert!(runtime.remove_inbox_filter("worker"));
    runtime
        .publish(json!({"query": "hi"}), "news", "troll", None)
        .await
        .unwrap();
    assert_eq!(*seen.lock(), 2);
}
