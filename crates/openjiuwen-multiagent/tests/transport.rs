#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Integration tests for the team transport abstraction.
//!
//! This suite covers the in-process backend (1:1 delegation to the crate's
//! message bus) and backend selection. The Redis backend has its own suite in
//! `tests/transport_redis.rs`, which runs it over a loopback fake Redis server
//! and pins the Streams wire contract.

mod common;

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use openjiuwen_common::error::ErrorKind;
use openjiuwen_multiagent::transport::{
    build_transport, InProcessTransport, TeamTransport, TransportConfig, TransportKind,
};
use openjiuwen_multiagent::{MessageBusConfig, MessageEnvelope, MessageHandler};
use serde_json::json;

use crate::common::RecordingHandler;

/// Builds a recording handler plus the erased handler the transports take.
fn recording_handler() -> (Arc<RecordingHandler>, Arc<dyn MessageHandler>) {
    let recorder = RecordingHandler::new();
    let handler = Arc::clone(&recorder) as Arc<dyn MessageHandler>;
    (recorder, handler)
}

#[tokio::test]
async fn pulsar_is_rejected_as_unsupported() {
    let (_recorder, handler) = recording_handler();
    let config = TransportConfig::new().with_kind(TransportKind::Pulsar);
    let error = build_transport(&config, "team", MessageBusConfig::new(), handler).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    let message = error.message().to_string();
    assert!(message.contains("Pulsar"), "{message}");
    assert!(message.contains("not implemented"), "{message}");
}

#[tokio::test]
async fn redis_without_settings_is_a_config_error() {
    let (_recorder, handler) = recording_handler();
    let config = TransportConfig::new().with_kind(TransportKind::Redis);
    let error = build_transport(&config, "team", MessageBusConfig::new(), handler).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Config);
    assert!(error.message().contains("redis"), "{}", error.message());
}

#[tokio::test]
async fn build_transport_builds_the_in_process_backend() {
    let (recorder, handler) = recording_handler();
    recorder.set_reply(json!({"built": true}));
    let config = TransportConfig::new().with_key_prefix("unused");
    let transport = build_transport(
        &config,
        "team-built",
        MessageBusConfig::new().with_max_queue_size(4),
        handler,
    )
    .unwrap();
    assert_eq!(transport.kind(), TransportKind::InProcess);
    assert_eq!(transport.team_id(), "team-built");
    let reply = transport
        .send(
            MessageEnvelope::p2p(json!({"n": 1}), "a", "b"),
            Some(Duration::from_millis(500)),
        )
        .await
        .unwrap();
    assert_eq!(reply, json!({"built": true}));
    transport.stop().await.unwrap();
}

#[tokio::test]
async fn in_process_send_publish_and_subscriptions_match_the_bus() {
    let (recorder, handler) = recording_handler();
    let transport = InProcessTransport::new("team-inproc", MessageBusConfig::new(), handler);
    assert_eq!(transport.kind(), TransportKind::InProcess);
    assert_eq!(transport.team_id(), "team-inproc");
    transport.start().await.unwrap();

    recorder.set_reply(json!({"ok": true}));
    let reply = transport
        .send(
            MessageEnvelope::p2p(json!({"n": 1}), "a", "b").with_session("s1"),
            Some(Duration::from_millis(500)),
        )
        .await
        .unwrap();
    assert_eq!(reply, json!({"ok": true}));
    assert_eq!(recorder.p2p_payloads(), vec![json!({"n": 1})]);
    assert_eq!(recorder.p2p.lock()[0].session_id.as_deref(), Some("s1"));

    transport.add_subscription("z", "news");
    transport.add_subscription("a", "news");
    assert_eq!(transport.subscribers("news"), vec!["a", "z"]);
    transport
        .publish(MessageEnvelope::pubsub(
            json!({"story": 1}),
            "publisher",
            "news",
        ))
        .await
        .unwrap();
    assert_eq!(recorder.pubsub_order(), vec!["a", "z"]);

    transport.remove_subscription("a", "news");
    assert_eq!(transport.subscribers("news"), vec!["z"]);
    transport.stop().await.unwrap();
    assert!(!transport.bus().is_started());
}

#[tokio::test]
async fn in_process_transport_exposes_the_subscription_registry() {
    let (_recorder, handler) = recording_handler();
    let transport = InProcessTransport::new("team-introspect", MessageBusConfig::new(), handler);

    transport.add_subscription("z", "news");
    transport.add_subscription("a", "news");
    transport.add_subscription("a", "*__p2p__");

    assert_eq!(
        transport.list_subscriptions(),
        BTreeMap::from([
            ("*__p2p__".to_owned(), vec!["a".to_owned()]),
            ("news".to_owned(), vec!["a".to_owned(), "z".to_owned()]),
        ])
    );
    assert_eq!(transport.agent_subscriptions("a"), vec!["*__p2p__", "news"]);
    assert_eq!(transport.agent_subscriptions("ghost"), Vec::<String>::new());
    assert_eq!(transport.subscription_count(), 3);

    transport.remove_subscription("a", "news");
    assert_eq!(transport.subscription_count(), 2);
    assert_eq!(transport.agent_subscriptions("a"), vec!["*__p2p__"]);
}

#[tokio::test]
async fn in_process_send_passes_the_caller_deadline_through() {
    let (recorder, handler) = recording_handler();
    recorder.delay_first(Duration::from_millis(300));
    let transport = InProcessTransport::new("team-deadline", MessageBusConfig::new(), handler);
    let error = transport
        .send(
            MessageEnvelope::p2p(json!({}), "a", "b"),
            Some(Duration::from_millis(20)),
        )
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Timeout);
    transport.stop().await.unwrap();
}
