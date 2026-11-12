#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Integration tests for the in-process [`MessageBus`].
//!
//! The suite drives the bus directly with [`common::RecordingHandler`]: it pins
//! per-topic FIFO ordering, concurrent fan-out to sorted subscribers, subscriber
//! failure isolation, the Python-compatible timeout message, session cleanup,
//! lazy start after [`MessageBus::stop`], topic naming, and error forwarding.
//! Every deadline is well under a second, so the suite stays fast and offline.

mod common;

use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::future::join_all;
use openjiuwen_common::error::{Error, ErrorKind, Result};
use openjiuwen_multiagent::{MessageBus, MessageBusConfig, MessageEnvelope, MessageHandler};
use serde_json::{json, Value};

use crate::common::RecordingHandler;

/// Builds a started bus with `team_id` and the recording handler.
async fn started_bus(team_id: &str, handler: &Arc<RecordingHandler>) -> MessageBus {
    let config = MessageBusConfig::new().with_team_id(team_id);
    let bus = MessageBus::new(config, Arc::clone(handler) as Arc<dyn MessageHandler>);
    bus.start().await.unwrap();
    bus
}

#[tokio::test]
async fn p2p_ordered_per_topic() {
    let handler = RecordingHandler::new();
    handler.delay_first(Duration::from_millis(50));
    handler.set_reply(json!({"ok": true}));
    let bus = started_bus("order", &handler).await;

    // Three concurrent senders on the same session topic: the first delivery is
    // slow, so a per-message task would record messages 2 and 3 before it.
    let sends = (0..3).map(|index| {
        let envelope =
            MessageEnvelope::p2p(json!({"n": index}), "sender", "receiver").with_session("s1");
        bus.send(envelope, Some(Duration::from_millis(500)))
    });
    let replies = join_all(sends).await;

    for reply in replies {
        assert_eq!(reply.unwrap(), json!({"ok": true}));
    }
    assert_eq!(
        handler.p2p_payloads(),
        vec![json!({"n": 0}), json!({"n": 1}), json!({"n": 2})]
    );
    bus.stop().await.unwrap();
}

#[tokio::test]
async fn pubsub_fans_out_to_sorted_subscribers() {
    let handler = RecordingHandler::new();
    let bus = started_bus("fanout", &handler).await;
    for subscriber in ["z", "a", "m"] {
        bus.add_subscription(subscriber, "news");
    }

    bus.publish(MessageEnvelope::pubsub(
        json!({"story": 1}),
        "publisher",
        "news",
    ))
    .await
    .unwrap();

    // `publish` returns only after every subscriber was delivered to.
    assert_eq!(handler.pubsub_order(), vec!["a", "m", "z"]);
    assert_eq!(handler.pubsub.lock().len(), 3);
    bus.stop().await.unwrap();
}

#[tokio::test]
async fn pubsub_subscriber_failure_is_isolated() {
    let handler = RecordingHandler::new();
    handler.fail_for("b");
    let bus = started_bus("isolate", &handler).await;
    for subscriber in ["a", "b", "c"] {
        bus.add_subscription(subscriber, "news");
    }

    bus.publish(MessageEnvelope::pubsub(
        json!({"story": 2}),
        "publisher",
        "news",
    ))
    .await
    .unwrap();

    assert_eq!(handler.pubsub_order(), vec!["a", "c"]);
    bus.stop().await.unwrap();
}

#[tokio::test]
async fn publish_without_subscribers_is_a_noop() {
    let handler = RecordingHandler::new();
    let bus = started_bus("silent", &handler).await;

    bus.publish(MessageEnvelope::pubsub(
        json!({"story": 3}),
        "publisher",
        "news",
    ))
    .await
    .unwrap();

    assert_eq!(handler.pubsub_order(), Vec::<String>::new());
    bus.stop().await.unwrap();
}

/// A handler whose P2P delivery always outlives the test's deadline.
struct SlowHandler {
    delay: Duration,
}

#[async_trait::async_trait]
impl MessageHandler for SlowHandler {
    async fn handle_p2p(&self, _envelope: &MessageEnvelope) -> Result<Value> {
        tokio::time::sleep(self.delay).await;
        Ok(json!({"slow": true}))
    }

    async fn handle_pubsub(&self, _envelope: &MessageEnvelope, _subscriber: &str) -> Result<()> {
        Ok(())
    }
}

#[tokio::test]
async fn send_times_out_with_python_message() {
    // The config deadline applies when the caller passes none.
    let bus = MessageBus::new(
        MessageBusConfig::new()
            .with_team_id("timeout")
            .with_process_timeout(Some(0.05)),
        Arc::new(SlowHandler {
            delay: Duration::from_millis(400),
        }),
    );
    let error = bus
        .send(MessageEnvelope::p2p(json!({"n": 1}), "alice", "bob"), None)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Timeout);
    let text = error.to_string();
    assert!(text.contains("timed out after"), "unexpected error: {text}");
    assert!(
        text.contains("message from 'alice' to 'bob'"),
        "unexpected error: {text}"
    );
    assert!(text.contains("0.05s"), "unexpected error: {text}");

    // A caller-supplied deadline overrides the (much longer) config deadline.
    let bus = MessageBus::new(
        MessageBusConfig::new()
            .with_team_id("override")
            .with_process_timeout(Some(0.5)),
        Arc::new(SlowHandler {
            delay: Duration::from_millis(400),
        }),
    );
    let started = Instant::now();
    let error = bus
        .send(
            MessageEnvelope::p2p(json!({"n": 2}), "alice", "bob"),
            Some(Duration::from_millis(10)),
        )
        .await
        .unwrap_err();
    assert!(started.elapsed() < Duration::from_millis(300));
    assert_eq!(error.kind(), ErrorKind::Timeout);
    let text = error.to_string();
    assert!(text.contains("timed out after"), "unexpected error: {text}");
    assert!(text.contains("0.01s"), "unexpected error: {text}");
    // The workers are still sleeping; dropping the bus lets the runtime discard
    // them when the test ends.
}

#[tokio::test]
async fn duplicate_subscribe_and_cleanup() {
    let handler = RecordingHandler::new();
    let bus = started_bus("cleanup", &handler).await;

    bus.subscribe("t").await.unwrap();
    let error = bus.subscribe("t").await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(error.message(), "Topic 't' is already subscribed.");

    // Create both per-session topics so cleanup has workers to remove.
    let session_p2p = bus.p2p_topic(Some("s1"));
    let session_pubsub = bus.pubsub_topic(Some("s1"));
    assert_eq!(session_p2p, "cleanup_s1__p2p__");
    bus.subscribe(&session_p2p).await.unwrap();
    bus.subscribe(&session_pubsub).await.unwrap();

    bus.cleanup_session("s1").await;

    // The bookkeeping entry is gone, so the same topic can subscribe again, and
    // a send recreates its worker.
    bus.subscribe(&session_p2p).await.unwrap();
    bus.subscribe(&session_pubsub).await.unwrap();
    let reply = bus
        .send(
            MessageEnvelope::p2p(json!({"n": 1}), "alice", "bob").with_session("s1"),
            Some(Duration::from_millis(500)),
        )
        .await
        .unwrap();
    assert_eq!(reply, json!({"ok": true}));
    assert_eq!(handler.p2p_payloads(), vec![json!({"n": 1})]);
    bus.stop().await.unwrap();
}

#[tokio::test]
async fn unknown_unsubscribe_is_a_noop() {
    let handler = RecordingHandler::new();
    let bus = started_bus("unsubscribe", &handler).await;

    // Dropping a topic that was never subscribed is a no-op, not an error.
    bus.unsubscribe("never-subscribed").await.unwrap();

    // Unsubscribing is idempotent, and it only clears the bookkeeping entry, so
    // the topic can be subscribed again afterwards.
    bus.subscribe("t").await.unwrap();
    bus.unsubscribe("t").await.unwrap();
    bus.unsubscribe("t").await.unwrap();
    bus.subscribe("t").await.unwrap();

    let error = bus.subscribe("t").await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(error.message(), "Topic 't' is already subscribed.");
    bus.stop().await.unwrap();
}

#[tokio::test]
async fn start_stop_idempotent() {
    let handler = RecordingHandler::new();
    let config = MessageBusConfig::new().with_team_id("lifecycle");
    let bus = MessageBus::new(config, Arc::clone(&handler) as Arc<dyn MessageHandler>);
    assert!(!bus.is_started());

    bus.start().await.unwrap();
    bus.start().await.unwrap();
    assert!(bus.is_started());

    bus.stop().await.unwrap();
    bus.stop().await.unwrap();
    assert!(!bus.is_started());

    // `send` lazily restarts the bus after `stop`.
    let reply = bus
        .send(
            MessageEnvelope::p2p(json!({"n": 1}), "alice", "bob"),
            Some(Duration::from_millis(500)),
        )
        .await
        .unwrap();
    assert_eq!(reply, json!({"ok": true}));
    assert!(bus.is_started());
    bus.stop().await.unwrap();
}

#[tokio::test]
async fn session_topics_are_distinct() {
    let handler = RecordingHandler::new();
    let config = MessageBusConfig::new().with_team_id("t");
    let bus = MessageBus::new(config, Arc::clone(&handler) as Arc<dyn MessageHandler>);

    assert_eq!(bus.team_id(), "t");
    assert_eq!(bus.p2p_topic(None), "t__p2p__");
    assert_eq!(bus.p2p_topic(Some("s1")), "t_s1__p2p__");
    assert_eq!(bus.p2p_topic(Some("s2")), "t_s2__p2p__");
    assert_eq!(bus.pubsub_topic(None), "t__pubsub__");
    assert_eq!(bus.pubsub_topic(Some("s1")), "t_s1__pubsub__");
    assert_ne!(bus.p2p_topic(None), bus.p2p_topic(Some("s1")));
    assert_ne!(bus.pubsub_topic(None), bus.pubsub_topic(Some("s1")));
}

/// A handler that always fails P2P delivery.
struct FailingHandler;

#[async_trait::async_trait]
impl MessageHandler for FailingHandler {
    async fn handle_p2p(&self, _envelope: &MessageEnvelope) -> Result<Value> {
        Err(Error::validation("boom"))
    }

    async fn handle_pubsub(&self, _envelope: &MessageEnvelope, _subscriber: &str) -> Result<()> {
        Ok(())
    }
}

#[tokio::test]
async fn p2p_handler_error_propagates() {
    let bus = MessageBus::new(
        MessageBusConfig::new().with_team_id("errors"),
        Arc::new(FailingHandler),
    );
    let error = bus
        .send(
            MessageEnvelope::p2p(json!({"n": 1}), "alice", "bob"),
            Some(Duration::from_millis(500)),
        )
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(error.message(), "boom");
    bus.stop().await.unwrap();
}
