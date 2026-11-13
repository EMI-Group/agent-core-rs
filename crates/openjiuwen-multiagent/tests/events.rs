#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Integration tests for the typed team-event bus.
//!
//! Exercises only the public path `openjiuwen_multiagent::events::*`: wire names
//! and Python parity constants, filtered synchronous delivery, unsubscribe and
//! bounded history, serde/schema contracts, and concurrent emission from several
//! tokio tasks.
//! Everything is in-process and offline.
use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use openjiuwen_multiagent::events::{
    TeamEvent, TeamEventBus, TeamEventFilter, TeamEventKind, TeamEventListener,
    AGENT_P2P_RECEIVED_EVENT, AGENT_PUBSUB_RECEIVED_EVENT, TEAM_EVENT_SCOPE,
};
use serde_json::{json, Value};
/// Builds a listener that appends every delivered event to a shared log.
fn recording_listener() -> (TeamEventListener, Arc<Mutex<Vec<TeamEvent>>>) {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&seen);
    let listener: TeamEventListener = Arc::new(move |event: &TeamEvent| {
        sink.lock().unwrap().push(event.clone());
    });
    (listener, seen)
}
#[test]
fn constants_keep_python_parity() {
    assert_eq!(TEAM_EVENT_SCOPE, "agent_team");
    assert_eq!(AGENT_P2P_RECEIVED_EVENT, "_framework:agent_p2p_received");
    assert_eq!(
        AGENT_PUBSUB_RECEIVED_EVENT,
        "_framework:agent_pubsub_received"
    );
    assert_eq!(
        TeamEventKind::MessageReceived.python_event_name(),
        Some(AGENT_P2P_RECEIVED_EVENT)
    );
    assert_eq!(TeamEventKind::MessageSent.python_event_name(), None);
    let pubsub = TeamEvent::pubsub_received("team", "sender", "subscriber", "topic", "m-1");
    assert_eq!(
        pubsub.python_event_name(),
        Some(AGENT_PUBSUB_RECEIVED_EVENT)
    );
    assert_eq!(
        TeamEvent::message_received("team", "sender", "subscriber", "m-2").python_event_name(),
        Some(AGENT_P2P_RECEIVED_EVENT)
    );
}
#[test]
fn every_kind_has_a_scoped_event_name() {
    for kind in TeamEventKind::ALL {
        assert_eq!(kind.event_name(), format!("agent_team:{}", kind.as_str()));
        assert!(kind
            .as_str()
            .chars()
            .all(|ch| ch.is_ascii_lowercase() || ch == '_'));
    }
}
#[test]
fn bus_delivers_matching_events_with_sequence_numbers() {
    let bus = TeamEventBus::new();
    assert_eq!(bus.subscriber_count(), 0);
    assert_eq!(bus.last_sequence(), 0);
    let (listener, seen) = recording_listener();
    let subscription = bus.subscribe(
        TeamEventFilter::new().kind(TeamEventKind::MemberJoined),
        listener,
    );
    assert_eq!(subscription, 1);
    let delivered = bus.emit(TeamEvent::member_joined("team", "alpha"));
    assert_eq!(delivered.sequence, 1);
    bus.emit(TeamEvent::member_left("team", "alpha"));
    let second = bus.emit(TeamEvent::member_joined("team", "beta"));
    assert_eq!(second.sequence, 3);
    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 2);
    assert_eq!(seen[0].agent_id.as_deref(), Some("alpha"));
    assert_eq!(seen[1].agent_id.as_deref(), Some("beta"));
    // The filtered-out `member_left` still consumed sequence 2.
    assert_eq!(seen[1].sequence, 3);
}
#[test]
fn filters_match_team_session_and_peer_agents() {
    let bus = TeamEventBus::new();
    let (team_listener, team_seen) = recording_listener();
    let (session_listener, session_seen) = recording_listener();
    let (agent_listener, agent_seen) = recording_listener();
    let (kinds_listener, kinds_seen) = recording_listener();
    bus.subscribe(TeamEventFilter::all().team("team-a"), team_listener);
    bus.subscribe(TeamEventFilter::new().session("s-1"), session_listener);
    bus.subscribe(TeamEventFilter::new().agent("beta"), agent_listener);
    bus.subscribe(
        TeamEventFilter::new().kinds([TeamEventKind::Error, TeamEventKind::TaskAssigned]),
        kinds_listener,
    );
    bus.emit(TeamEvent::handoff_started("team-a", "alpha", "beta").with_session("s-1"));
    bus.emit(TeamEvent::handoff_started("team-b", "alpha", "gamma"));
    bus.emit(TeamEvent::task_assigned("team-a", "alpha", "write tests"));
    bus.emit(TeamEvent::error("team-b", "gamma", "boom"));
    assert_eq!(team_seen.lock().unwrap().len(), 2);
    assert_eq!(session_seen.lock().unwrap().len(), 1);
    // First event matches through the peer id; the fourth through the subject.
    let agent_ids: Vec<Option<String>> = agent_seen
        .lock()
        .unwrap()
        .iter()
        .map(|event| event.agent_id.clone())
        .collect();
    assert_eq!(agent_ids, [Some("alpha".to_owned())]);
    let kinds: Vec<TeamEventKind> = kinds_seen
        .lock()
        .unwrap()
        .iter()
        .map(|event| event.kind)
        .collect();
    assert_eq!(kinds, [TeamEventKind::TaskAssigned, TeamEventKind::Error]);
}
#[test]
fn unsubscribe_removes_a_listener_and_rejects_unknown_ids() {
    let bus = TeamEventBus::new();
    let (first, first_seen) = recording_listener();
    let (second, second_seen) = recording_listener();
    let first_id = bus.subscribe(TeamEventFilter::new(), first);
    let second_id = bus.subscribe(TeamEventFilter::new(), second);
    assert_ne!(first_id, second_id);
    assert!(bus.unsubscribe(first_id));
    assert!(!bus.unsubscribe(first_id));
    assert!(!bus.unsubscribe(0));
    assert_eq!(bus.subscriber_count(), 1);
    bus.emit(TeamEvent::member_joined("team", "alpha"));
    assert_eq!(*first_seen.lock().unwrap(), Vec::<TeamEvent>::new());
    assert_eq!(second_seen.lock().unwrap().len(), 1);
}
#[test]
fn history_is_a_bounded_snapshot() {
    let bus = TeamEventBus::with_capacity(4);
    for index in 0..10 {
        bus.emit(TeamEvent::member_joined("team", format!("agent-{index}")));
    }
    let history = bus.history();
    assert_eq!(history.len(), 4);
    assert_eq!(
        history
            .iter()
            .map(|event| event.sequence)
            .collect::<Vec<u64>>(),
        [7, 8, 9, 10]
    );
    assert_eq!(bus.history_of(TeamEventKind::MemberJoined).len(), 4);
    assert_eq!(
        bus.history_of(TeamEventKind::HandoffStarted),
        Vec::<TeamEvent>::new()
    );
    // The snapshot is detached from the bus: mutating it changes nothing.
    drop(history);
    assert_eq!(bus.history().len(), 4);
    let empty = TeamEventBus::with_capacity(0);
    empty.emit(TeamEvent::member_left("team", "alpha"));
    assert_eq!(empty.history(), Vec::<TeamEvent>::new());
    assert_eq!(empty.last_sequence(), 1);
}
#[test]
fn event_serde_and_schema_contract() {
    let event = TeamEvent::task_completed("team", "alpha", "write tests")
        .with_session("s-1")
        .with_message_id("m-1");
    let wire = serde_json::to_value(&event).unwrap();
    assert_eq!(
        wire,
        json!({
            "event_id": event.event_id,
            "sequence": 0,
            "kind": "task_completed",
            "team_id": "team",
            "session_id": "s-1",
            "agent_id": "alpha",
            "peer_id": null,
            "topic_id": null,
            "message_id": "m-1",
            "detail": {"task": "write tests"},
        })
    );
    let decoded: TeamEvent = serde_json::from_value(wire).unwrap();
    assert_eq!(decoded, event);
    let tolerant: TeamEvent = serde_json::from_value(json!({
        "kind": "handoff_started",
        "team_id": "team",
        "extra": {"ignored": true},
        "detail": null,
    }))
    .unwrap();
    assert_eq!(tolerant.kind, TeamEventKind::HandoffStarted);
    assert_eq!(tolerant.detail, json!({}));
    let schema = serde_json::to_value(schemars::schema_for!(TeamEvent)).unwrap();
    let mut names: Vec<String> = schema["properties"]
        .as_object()
        .unwrap()
        .keys()
        .cloned()
        .collect();
    names.sort();
    assert_eq!(
        names,
        [
            "agent_id",
            "detail",
            "event_id",
            "kind",
            "message_id",
            "peer_id",
            "sequence",
            "session_id",
            "team_id",
            "topic_id",
        ]
    );
    let detail_schema: Value = schema["properties"]["detail"].clone();
    assert!(
        detail_schema.to_string().contains("object"),
        "detail must be documented as an object: {detail_schema}"
    );
}
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_emits_stamp_unique_sequences() {
    const TASKS: u64 = 8;
    const PER_TASK: u64 = 50;
    let total = TASKS * PER_TASK;
    let total_count = usize::try_from(total).expect("test total fits in usize");
    let bus = TeamEventBus::with_capacity(64);
    let emitted = Arc::new(Mutex::new(Vec::with_capacity(total_count)));
    let (listener, delivered) = recording_listener();
    bus.subscribe(TeamEventFilter::new(), listener);
    let mut tasks = Vec::new();
    for task_index in 0..TASKS {
        let bus = Arc::clone(&bus);
        let emitted = Arc::clone(&emitted);
        tasks.push(tokio::spawn(async move {
            for step in 0..PER_TASK {
                let event = TeamEvent::message_sent(
                    "team",
                    format!("agent-{task_index}"),
                    "peer",
                    format!("m-{step}"),
                );
                let sequence = bus.emit(event).sequence;
                emitted.lock().unwrap().push(sequence);
            }
        }));
    }
    for task in tasks {
        task.await.expect("emit task joins");
    }
    let mut sequences = emitted.lock().unwrap().clone();
    sequences.sort_unstable();
    assert_eq!(sequences, (1..=total).collect::<Vec<u64>>());
    let unique: HashSet<u64> = sequences.iter().copied().collect();
    assert_eq!(unique.len(), total_count);
    assert_eq!(bus.last_sequence(), total);
    let mut delivered: Vec<u64> = delivered
        .lock()
        .unwrap()
        .iter()
        .map(|event| event.sequence)
        .collect();
    delivered.sort_unstable();
    assert_eq!(delivered, sequences);
    let history = bus.history();
    assert_eq!(history.len(), 64);
    assert_eq!(history.last().map(|event| event.sequence), Some(total));
    assert!(history
        .windows(2)
        .all(|window| window[0].sequence < window[1].sequence));
}
#[tokio::test]
async fn reentrant_emit_from_a_listener_is_ordered() {
    let bus = TeamEventBus::new();
    let bus_for_listener = Arc::clone(&bus);
    let (nested, nested_seen) = recording_listener();
    bus.subscribe(TeamEventFilter::new().kind(TeamEventKind::Error), nested);
    let reentrant: TeamEventListener = Arc::new(move |event: &TeamEvent| {
        if event.kind == TeamEventKind::MemberJoined {
            bus_for_listener.emit(TeamEvent::error("team", "listener", "nested"));
        }
    });
    bus.subscribe(
        TeamEventFilter::new().kind(TeamEventKind::MemberJoined),
        reentrant,
    );
    let outer = bus.emit(TeamEvent::member_joined("team", "alpha"));
    let inner = bus.emit(TeamEvent::member_left("team", "alpha"));
    assert_eq!(outer.sequence, 1);
    assert_eq!(inner.sequence, 3);
    let seen = nested_seen.lock().unwrap();
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0].sequence, 2);
    assert_eq!(bus.history().len(), 3);
}
#[test]
fn debug_summarizes_without_listener_data() {
    let bus = TeamEventBus::with_capacity(2);
    bus.subscribe(TeamEventFilter::new(), Arc::new(|_event: &TeamEvent| {}));
    bus.emit(TeamEvent::member_joined("team", "alpha"));
    let rendered = format!("{bus:?}");
    assert!(rendered.contains("subscriber_count: 1"), "{rendered}");
    assert!(rendered.contains("history_capacity: 2"), "{rendered}");
    assert!(rendered.contains("last_sequence: 1"), "{rendered}");
}
