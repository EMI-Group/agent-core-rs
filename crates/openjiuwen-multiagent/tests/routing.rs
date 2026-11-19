#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Integration tests for group routing, broadcasts, and the routing table.
//!
//! The suite is offline and deterministic: members are in-process fakes, the
//! default in-process transport delivers the broadcasts, and one recording
//! transport observes the envelopes a broadcast emits. It pins the
//! `GroupRegistry` contract, the sender-exclusion rule, member-failure
//! isolation, skipped unregistered members, the `message_sent` group event, and
//! the serializable routing snapshot.
mod common;

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use openjiuwen_agent::AgentOutput;
use openjiuwen_common::error::{Error, ErrorKind, Result};
use openjiuwen_multiagent::events::{TeamEvent, TeamEventFilter, TeamEventKind};
use openjiuwen_multiagent::transport::{TeamTransport, TransportKind};
use openjiuwen_multiagent::{
    broadcast_targets, AgentProvider, GroupRegistry, MessageEnvelope, RuntimeConfig, TeamAgent,
    TeamRuntime, GROUP_METADATA_KEY,
};
use parking_lot::Mutex;
use serde_json::{json, Value};

use crate::common::{agent_card, FakeAgent};

/// Builds a provider that records every delivered payload.
fn recording_provider(name: &str, seen: &Arc<Mutex<Vec<Value>>>) -> AgentProvider {
    let seen = Arc::clone(seen);
    let name = name.to_owned();
    Arc::new(move || {
        let seen = Arc::clone(&seen);
        TeamAgent::new(FakeAgent::new(
            agent_card(&name),
            move |inputs, _session| {
                seen.lock().push(inputs);
                Ok(AgentOutput::answer("ok"))
            },
        ))
    })
}

/// Builds a provider whose member always fails.
fn failing_provider(name: &str) -> AgentProvider {
    let name = name.to_owned();
    Arc::new(move || {
        TeamAgent::new(FakeAgent::new(agent_card(&name), |_inputs, _session| {
            Err(Error::validation("member exploded"))
        }))
    })
}

/// A transport that records every outbound envelope without delivering it.
#[derive(Debug, Default)]
struct RecordingTransport {
    seen: Mutex<Vec<MessageEnvelope>>,
    failing: BTreeSet<String>,
}

#[async_trait]
impl TeamTransport for RecordingTransport {
    fn kind(&self) -> TransportKind {
        TransportKind::InProcess
    }

    fn team_id(&self) -> &'static str {
        "t"
    }

    async fn start(&self) -> Result<()> {
        Ok(())
    }

    async fn stop(&self) -> Result<()> {
        Ok(())
    }

    async fn send(&self, envelope: MessageEnvelope, _timeout: Option<Duration>) -> Result<Value> {
        let recipient = envelope.recipient.clone().unwrap_or_default();
        self.seen.lock().push(envelope);
        if self.failing.contains(&recipient) {
            return Err(Error::validation("transport boom"));
        }
        Ok(json!({"ok": true}))
    }

    async fn publish(&self, _envelope: MessageEnvelope) -> Result<()> {
        Ok(())
    }

    fn add_subscription(&self, _agent_id: &str, _topic: &str) -> Result<()> {
        Ok(())
    }

    fn remove_subscription(&self, _agent_id: &str, _topic: &str) -> Result<()> {
        Ok(())
    }

    fn subscribers(&self, _topic_id: &str) -> Vec<String> {
        Vec::new()
    }

    fn list_subscriptions(&self) -> BTreeMap<String, Vec<String>> {
        BTreeMap::new()
    }
}

#[test]
fn group_registry_registers_replaces_and_lists() {
    let registry = GroupRegistry::new();
    assert_eq!(
        registry
            .register_group("team", vec!["b".to_owned(), "a".to_owned(), "a".to_owned()])
            .unwrap(),
        2
    );
    assert_eq!(registry.group_members("team"), vec!["a", "b"]);
    assert_eq!(registry.groups(), vec!["team"]);
    assert_eq!(registry.group_count(), 1);
    assert_eq!(registry.member_count(), 2);
    assert_eq!(
        registry
            .register_group("team", vec!["c".to_owned()])
            .unwrap(),
        1
    );
    assert_eq!(registry.group_members("team"), vec!["c"]);
    assert_eq!(registry.member_count(), 1);
    assert_eq!(registry.group_members("missing"), Vec::<String>::new());
}

#[test]
fn group_registry_membership_and_lookups() {
    let registry = GroupRegistry::new();
    assert!(registry.add_member("g", "b").unwrap());
    assert!(registry.add_member("g", "a").unwrap());
    assert!(!registry.add_member("g", "a").unwrap());
    assert!(registry.add_member("h", "a").unwrap());
    assert_eq!(registry.group_members("g"), vec!["a", "b"]);
    assert_eq!(registry.groups(), vec!["g", "h"]);
    assert_eq!(registry.groups_of("a"), vec!["g", "h"]);
    assert_eq!(registry.groups_of("b"), vec!["g"]);
    assert_eq!(registry.groups_of("z"), Vec::<String>::new());
    assert_eq!(registry.member_count(), 3);
    assert!(registry.remove_member("g", "a"));
    assert!(!registry.remove_member("g", "a"));
    assert!(!registry.remove_member("missing", "a"));
    assert_eq!(registry.group_members("g"), vec!["b"]);
    assert!(registry.unregister_group("h"));
    assert!(!registry.unregister_group("h"));
    assert_eq!(registry.group_count(), 1);
    assert_eq!(registry.clear(), 1);
    assert_eq!(registry.group_count(), 0);
    assert_eq!(registry.member_count(), 0);
}

#[test]
fn group_registry_validates_empty_ids() {
    let registry = GroupRegistry::new();
    let error = registry
        .register_group("", Vec::<String>::new())
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("group_id is required"));
    assert_eq!(
        registry.add_member("", "a").unwrap_err().kind(),
        ErrorKind::Validation
    );
    assert_eq!(
        registry.add_member("g", "").unwrap_err().kind(),
        ErrorKind::Validation
    );
    assert_eq!(registry.group_count(), 0);
}

#[test]
fn broadcast_targets_exclude_the_sender_by_default() {
    let members = vec!["b".to_owned(), "a".to_owned(), "b".to_owned()];
    assert_eq!(
        broadcast_targets(members.clone(), Some("b"), false),
        vec!["a"]
    );
    assert_eq!(
        broadcast_targets(members.clone(), Some("b"), true),
        vec!["a", "b"]
    );
    assert_eq!(broadcast_targets(members, None, false), vec!["a", "b"]);
    assert_eq!(
        broadcast_targets(Vec::<String>::new(), Some("a"), false),
        Vec::<String>::new()
    );
}

#[tokio::test]
async fn broadcast_reaches_every_registered_member_but_not_the_sender() {
    let runtime = TeamRuntime::new(RuntimeConfig::new().with_team_id("t"));
    let seen_a = Arc::new(Mutex::new(Vec::new()));
    let seen_b = Arc::new(Mutex::new(Vec::new()));
    let seen_sender = Arc::new(Mutex::new(Vec::new()));
    runtime.register_agent(agent_card("a"), recording_provider("a", &seen_a));
    runtime.register_agent(agent_card("b"), recording_provider("b", &seen_b));
    runtime.register_agent(agent_card("boss"), recording_provider("boss", &seen_sender));
    runtime
        .register_group("team", ["a".to_owned(), "b".to_owned(), "boss".to_owned()])
        .unwrap();
    let broadcast = runtime
        .broadcast_to_group(json!({"query": "go"}), "team", "boss", None, None)
        .await
        .unwrap();
    assert_eq!(broadcast.group, "team");
    assert_eq!(broadcast.recipients, vec!["a", "b"]);
    assert_eq!(broadcast.skipped, Vec::<String>::new());
    assert_eq!(broadcast.succeeded(), 2);
    assert_eq!(broadcast.failed(), 0);
    assert_eq!(
        broadcast.value_of("a"),
        Some(&json!({"output": "ok", "result_type": "answer"}))
    );
    assert_eq!(seen_a.lock().len(), 1);
    assert_eq!(seen_b.lock().len(), 1);
    assert_eq!(seen_sender.lock().len(), 0);
    let results = broadcast.into_results();
    assert_eq!(results.len(), 2);
    assert!(results["a"].is_ok());
}

#[tokio::test]
async fn broadcast_isolation_skips_unregistered_members_and_failures() {
    let runtime = TeamRuntime::new(RuntimeConfig::new().with_team_id("t"));
    runtime.register_agent(agent_card("a"), {
        let seen = Arc::new(Mutex::new(Vec::new()));
        recording_provider("a", &seen)
    });
    runtime.register_agent(agent_card("broken"), failing_provider("broken"));
    runtime
        .register_group(
            "team",
            ["a".to_owned(), "broken".to_owned(), "ghost".to_owned()],
        )
        .unwrap();
    let broadcast = runtime
        .broadcast_to_group(json!("hello"), "team", "boss", None, None)
        .await
        .unwrap();
    assert_eq!(broadcast.recipients, vec!["a", "broken"]);
    assert_eq!(broadcast.skipped, vec!["ghost"]);
    assert_eq!(broadcast.succeeded(), 1);
    assert_eq!(broadcast.failed(), 1);
    assert!(broadcast.value_of("broken").is_none());
    let results = broadcast.into_results();
    let error = results["broken"].as_ref().unwrap_err();
    assert!(error.message().contains("member exploded"));
}

#[tokio::test]
async fn broadcast_validates_sender_and_group() {
    let runtime = TeamRuntime::new(RuntimeConfig::new().with_team_id("t"));
    runtime.register_group("team", ["a".to_owned()]).unwrap();
    let error = runtime
        .broadcast_to_group(json!("hi"), "team", "", None, None)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("sender is required"));

    let error = runtime
        .broadcast_to_group(json!("hi"), "missing", "boss", None, None)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert!(error
        .message()
        .contains("group 'missing' is not registered"));
}

#[tokio::test]
async fn broadcast_carries_the_group_metadata_and_emits_one_event() {
    let runtime = TeamRuntime::new(RuntimeConfig::new().with_team_id("t"));
    let transport = Arc::new(RecordingTransport::default());
    runtime
        .set_transport(Arc::clone(&transport) as Arc<dyn TeamTransport>)
        .unwrap();
    runtime.register_agent(
        agent_card("a"),
        recording_provider("a", &Arc::new(Mutex::new(Vec::new()))),
    );
    runtime.register_agent(
        agent_card("b"),
        recording_provider("b", &Arc::new(Mutex::new(Vec::new()))),
    );
    runtime
        .register_group("team", ["a".to_owned(), "b".to_owned()])
        .unwrap();

    let events = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&events);
    runtime.events().subscribe(
        TeamEventFilter::new().kind(TeamEventKind::MessageSent),
        Arc::new(move |event: &TeamEvent| sink.lock().push(event.clone())),
    );

    let broadcast = runtime
        .broadcast_to_group(json!({"query": "go"}), "team", "boss", None, None)
        .await
        .unwrap();
    assert_eq!(broadcast.succeeded(), 2);

    let envelopes = transport.seen.lock();
    assert_eq!(envelopes.len(), 2);
    for envelope in envelopes.iter() {
        assert_eq!(
            envelope.metadata.get(GROUP_METADATA_KEY),
            Some(&Value::String("team".to_owned()))
        );
        assert_eq!(envelope.sender.as_deref(), Some("boss"));
    }
    assert_eq!(
        envelopes
            .iter()
            .filter_map(|envelope| envelope.recipient.clone())
            .collect::<BTreeSet<_>>(),
        BTreeSet::from(["a".to_owned(), "b".to_owned()])
    );
    drop(envelopes);

    let events = events.lock();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].kind, TeamEventKind::MessageSent);
    assert_eq!(
        events[0].detail.get("kind"),
        Some(&Value::String("group".to_owned()))
    );
    assert_eq!(
        events[0].detail.get("group"),
        Some(&Value::String("team".to_owned()))
    );
    assert_eq!(events[0].detail.get("recipients"), Some(&json!(2)));
}

#[tokio::test]
async fn routing_table_resolves_groups_topics_and_destinations() {
    let runtime = TeamRuntime::new(RuntimeConfig::new().with_team_id("t"));
    runtime.register_agent(
        agent_card("a"),
        recording_provider("a", &Arc::new(Mutex::new(Vec::new()))),
    );
    runtime.register_agent(
        agent_card("b"),
        recording_provider("b", &Arc::new(Mutex::new(Vec::new()))),
    );
    runtime.subscribe("a", "news.*").await.unwrap();
    runtime.subscribe("b", "alerts").await.unwrap();
    runtime
        .register_group("team", ["a".to_owned(), "b".to_owned()])
        .unwrap();

    let table = runtime.routing_table();
    assert_eq!(table.team_id, "t");
    assert_eq!(table.route("team"), vec!["a", "b"]);
    assert_eq!(table.route("alerts"), vec!["b"]);
    assert_eq!(table.route("news.tech"), vec!["a"]);
    assert_eq!(table.route("unknown"), Vec::<String>::new());
    assert_eq!(table.destinations("a"), vec!["news.*", "team"]);
    assert_eq!(table.destinations("b"), vec!["alerts", "team"]);
    assert_eq!(table.destinations("ghost"), Vec::<String>::new());

    let encoded = serde_json::to_value(&table).unwrap();
    let decoded = serde_json::from_value(encoded).unwrap();
    assert_eq!(table, decoded);
}
