//! Behaviour suite for the in-process message queue, driven through the crate's
//! public API exactly as `openjiuwen-harness` and `openjiuwen-cli` will.
//!
//! Ports `tests/unit_tests/core/runner/test_message_queue.py` (pub/sub plus
//! request/reply against the in-memory fake) and the topic-ordering assertions
//! of `test_message_queue_inmemory.py`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]

use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use std::time::Duration;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, ErrorKind, Result};
use openjiuwen_runner::mq::{
    build_message_queue, create_message_queue, DeferredMessageQueue, InMemoryMessageQueue,
    MessageHandler, MessageQueue, MessageQueueConfig, MessageQueueKind, QueueMessage,
};
use openjiuwen_runner::{Runner, RunnerConfig};
use serde_json::{json, Value};
use tokio::sync::{mpsc, oneshot};

/// Records every payload it handles so a test can assert delivery order.
struct RecordingHandler {
    /// One entry per handled delivery, in handling order.
    seen: mpsc::UnboundedSender<Value>,
}

#[async_trait]
impl MessageHandler for RecordingHandler {
    async fn handle(&self, message: QueueMessage) -> Result<Value> {
        let _ = self.seen.send(message.payload);
        Ok(json!({"received": true}))
    }
}

/// Replies with the request payload, so the request path can be asserted.
struct EchoHandler;

#[async_trait]
impl MessageHandler for EchoHandler {
    async fn handle(&self, message: QueueMessage) -> Result<Value> {
        Ok(message.payload)
    }
}

/// Fails every delivery with an error the requester must see unchanged.
struct FailingHandler;

#[async_trait]
impl MessageHandler for FailingHandler {
    async fn handle(&self, _message: QueueMessage) -> Result<Value> {
        Err(Error::Validation("handler says no".to_owned()))
    }
}

/// Blocks until the test releases it, then signals completion.
struct GatedHandler {
    /// Released by the test after the caller's deadline expired.
    gate: StdMutex<Option<oneshot::Receiver<()>>>,
    /// Fires once the handler finished, proving it kept running.
    done: mpsc::UnboundedSender<Value>,
}

#[async_trait]
impl MessageHandler for GatedHandler {
    async fn handle(&self, message: QueueMessage) -> Result<Value> {
        let gate = self.gate.lock().expect("gate lock").take();
        if let Some(gate) = gate {
            let _ = gate.await;
        }
        let _ = self.done.send(message.payload);
        Ok(Value::Null)
    }
}

/// Builds a concrete queue (so `deliveries()` stays reachable) and starts it.
async fn started_queue(capacity: usize) -> Arc<InMemoryMessageQueue> {
    let queue = InMemoryMessageQueue::with_capacity(capacity);
    queue.start().await.expect("start");
    queue
}

#[tokio::test]
async fn publish_and_request_happy_paths() {
    let queue = started_queue(8).await;
    let (seen_tx, mut seen_rx) = mpsc::unbounded_channel();
    queue
        .add_handler("inbox", Arc::new(RecordingHandler { seen: seen_tx }))
        .await
        .expect("recording handler");
    queue
        .add_handler("echo", Arc::new(EchoHandler))
        .await
        .expect("echo handler");

    assert!(queue.is_started());
    assert_eq!(queue.topics(), vec!["echo".to_owned(), "inbox".to_owned()]);
    assert!(queue.has_subscriber("inbox"));
    assert!(queue.has_subscriber("echo"));

    queue
        .publish("inbox", QueueMessage::new(json!({"event": "ping"})))
        .await
        .expect("publish");
    assert_eq!(seen_rx.recv().await, Some(json!({"event": "ping"})));

    let reply = queue
        .request("echo", QueueMessage::new(json!({"query": "hi"})), None)
        .await
        .expect("request");
    assert_eq!(reply, json!({"query": "hi"}));
    assert_eq!(queue.deliveries(), 2);
    queue.stop().await.expect("stop");
}

#[tokio::test]
async fn deliveries_are_fifo_per_topic() {
    let queue = started_queue(8).await;
    let (seen_tx, mut seen_rx) = mpsc::unbounded_channel();
    queue
        .add_handler("inbox", Arc::new(RecordingHandler { seen: seen_tx }))
        .await
        .expect("handler");

    for seq in 0..4 {
        queue
            .publish("inbox", QueueMessage::new(json!(seq)))
            .await
            .expect("publish");
    }
    for seq in 0..4 {
        assert_eq!(seen_rx.recv().await, Some(json!(seq)));
    }
}

#[tokio::test]
async fn request_on_an_unsubscribed_topic_is_not_found() {
    let queue = started_queue(4).await;
    let error = queue
        .request("absent", QueueMessage::new(json!(1)), None)
        .await
        .expect_err("unsubscribed topic");
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert_eq!(
        error.to_string(),
        "not_found: message queue topic 'absent' is not subscribed"
    );
}

#[tokio::test]
async fn request_on_a_topic_without_a_handler_is_not_found() {
    let queue = started_queue(4).await;
    queue.subscribe("inbox").await.expect("subscribe");
    assert!(!queue.has_subscriber("inbox"));
    let error = queue
        .request("inbox", QueueMessage::new(json!(1)), None)
        .await
        .expect_err("no handler installed");
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert_eq!(
        error.to_string(),
        "not_found: message queue topic 'inbox' has no handler"
    );
}

#[tokio::test]
async fn handler_errors_reach_the_requester() {
    let queue = started_queue(4).await;
    queue
        .add_handler("strict", Arc::new(FailingHandler))
        .await
        .expect("handler");
    let error = queue
        .request("strict", QueueMessage::new(json!(1)), None)
        .await
        .expect_err("handler failure");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(error.to_string(), "validation: handler says no");
}

#[tokio::test]
async fn request_timeout_leaves_the_handler_running() {
    let queue = started_queue(4).await;
    let (release_tx, release_rx) = oneshot::channel();
    let (done_tx, mut done_rx) = mpsc::unbounded_channel();
    queue
        .add_handler(
            "slow",
            Arc::new(GatedHandler {
                gate: StdMutex::new(Some(release_rx)),
                done: done_tx,
            }),
        )
        .await
        .expect("handler");

    let error = queue
        .request(
            "slow",
            QueueMessage::new(json!("work")),
            Some(Duration::from_millis(20)),
        )
        .await
        .expect_err("deadline");
    assert_eq!(error.kind(), ErrorKind::Timeout);
    assert_eq!(
        error.to_string(),
        "timeout: message queue request on 'slow' timed out after 20ms"
    );
    assert_eq!(queue.deliveries(), 1);

    // The caller gave up, the worker did not: the handler runs to completion.
    release_tx.send(()).expect("release handler");
    assert_eq!(done_rx.recv().await, Some(json!("work")));
}

#[tokio::test]
async fn the_configured_default_deadline_applies_to_requests_without_one() {
    let queue = InMemoryMessageQueue::new(
        MessageQueueConfig::default()
            .with_capacity(4)
            .with_request_timeout_secs(Some(0.02)),
    );
    queue.start().await.expect("start");
    let (release_tx, release_rx) = oneshot::channel();
    let (done_tx, mut done_rx) = mpsc::unbounded_channel();
    queue
        .add_handler(
            "slow",
            Arc::new(GatedHandler {
                gate: StdMutex::new(Some(release_rx)),
                done: done_tx,
            }),
        )
        .await
        .expect("handler");

    let error = queue
        .request("slow", QueueMessage::new(json!("work")), None)
        .await
        .expect_err("configured deadline");
    assert_eq!(error.kind(), ErrorKind::Timeout);
    release_tx.send(()).expect("release handler");
    assert_eq!(done_rx.recv().await, Some(json!("work")));
}

#[tokio::test]
async fn operations_before_start_are_config_errors() {
    let queue = InMemoryMessageQueue::with_capacity(4);
    assert!(!queue.is_started());

    let publish = queue
        .publish("inbox", QueueMessage::new(json!(1)))
        .await
        .expect_err("publish before start");
    assert_eq!(publish.kind(), ErrorKind::Config);
    assert_eq!(publish.to_string(), "config: message queue is not started");

    let subscribe = queue.subscribe("inbox").await.expect_err("subscribe");
    assert_eq!(subscribe.kind(), ErrorKind::Config);

    let handler = queue
        .add_handler("inbox", Arc::new(EchoHandler))
        .await
        .expect_err("add_handler");
    assert_eq!(handler.kind(), ErrorKind::Config);

    let request = queue
        .request("inbox", QueueMessage::new(json!(1)), None)
        .await
        .expect_err("request");
    assert_eq!(request.kind(), ErrorKind::Config);
    assert!(queue.topics().is_empty());
}

#[tokio::test]
async fn start_and_stop_are_idempotent() {
    let queue = InMemoryMessageQueue::with_capacity(4);
    queue.start().await.expect("start");
    queue.start().await.expect("start again");
    queue.subscribe("inbox").await.expect("subscribe");
    assert_eq!(queue.topics(), vec!["inbox".to_owned()]);

    queue.stop().await.expect("stop");
    queue.stop().await.expect("stop again");
    assert!(!queue.is_started());
    assert!(queue.topics().is_empty(), "stop empties the topics");

    // Restarting re-enables the queue but does not resurrect old topics.
    queue.start().await.expect("restart");
    assert!(queue.is_started());
    assert!(queue.topics().is_empty());
    queue
        .subscribe("inbox")
        .await
        .expect("subscribe after restart");
    assert_eq!(queue.topics(), vec!["inbox".to_owned()]);
}

#[tokio::test]
async fn unsubscribe_joins_the_worker() {
    let queue = started_queue(4).await;
    let (seen_tx, mut seen_rx) = mpsc::unbounded_channel();
    queue
        .add_handler("inbox", Arc::new(RecordingHandler { seen: seen_tx }))
        .await
        .expect("handler");
    queue
        .publish("inbox", QueueMessage::new(json!("queued")))
        .await
        .expect("publish");
    assert_eq!(seen_rx.recv().await, Some(json!("queued")));

    queue.unsubscribe("inbox").await.expect("unsubscribe");
    assert!(queue.topics().is_empty());
    assert!(!queue.has_subscriber("inbox"));
    // An unknown topic is a no-op, so teardown is idempotent.
    queue.unsubscribe("inbox").await.expect("unsubscribe again");
    // The worker and its handler are gone: publishing is a no-op, requesting fails.
    queue
        .publish("inbox", QueueMessage::new(json!("dropped")))
        .await
        .expect("publish after unsubscribe");
    let error = queue
        .request("inbox", QueueMessage::new(json!("dropped")), None)
        .await
        .expect_err("unsubscribed");
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert_eq!(queue.deliveries(), 1);
    queue.stop().await.expect("stop");
}

#[tokio::test]
async fn duplicate_subscribe_is_a_validation_error() {
    let queue = started_queue(4).await;
    queue.subscribe("inbox").await.expect("first subscribe");
    let error = queue
        .subscribe("inbox")
        .await
        .expect_err("duplicate subscribe");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(queue.topics(), vec!["inbox".to_owned()]);
}

#[tokio::test]
async fn create_message_queue_builds_a_usable_in_memory_queue() {
    let queue = create_message_queue(&MessageQueueConfig::default())
        .await
        .expect("in-memory backend");
    assert!(!queue.is_started(), "the runner starts its queue");
    queue.start().await.expect("start");
    queue
        .add_handler("echo", Arc::new(EchoHandler))
        .await
        .expect("handler");
    let reply = queue
        .request(
            "echo",
            QueueMessage::new(json!("through the factory")),
            None,
        )
        .await
        .expect("request");
    assert_eq!(reply, json!("through the factory"));
    queue.stop().await.expect("stop");
}

#[tokio::test]
async fn create_message_queue_rejects_the_deferred_backends() {
    for kind in [MessageQueueKind::Pulsar, MessageQueueKind::ZeroMq] {
        let config = MessageQueueConfig {
            kind,
            url: Some("pulsar://broker:6650".to_owned()),
            ..MessageQueueConfig::default()
        };
        let error = create_message_queue(&config)
            .await
            .err()
            .expect("deferred backend");
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        let text = error.to_string();
        assert!(text.contains(kind.as_str()), "names the backend: {text}");
        assert!(text.contains("in_memory"), "points at the port: {text}");
    }
}

// ------------------------------------------- deferred transports (typed) --

#[test]
fn build_message_queue_never_reports_a_deferred_kind_as_in_memory() {
    let in_memory = build_message_queue(&MessageQueueConfig::default());
    assert!(!in_memory.is_started(), "the runner starts its queue");
    assert!(in_memory.topics().is_empty());

    for kind in [MessageQueueKind::Pulsar, MessageQueueKind::ZeroMq] {
        let config = MessageQueueConfig {
            kind,
            ..MessageQueueConfig::default()
        };
        let queue = build_message_queue(&config);
        assert!(!queue.is_started());
        assert!(queue.topics().is_empty());
        assert!(!queue.has_subscriber("inbox"));
    }
}

#[test]
fn the_deferred_queue_reports_the_kind_it_stands_in_for() {
    for kind in [MessageQueueKind::Pulsar, MessageQueueKind::ZeroMq] {
        let queue = DeferredMessageQueue::new(kind);
        assert_eq!(queue.kind(), kind);
        assert!(format!("{queue:?}").contains(kind.as_str()));
    }
}

/// `Runner::with_config` is infallible, but a configuration naming a deferred
/// transport must not be served by an in-memory queue behind the host's back.
#[tokio::test]
async fn a_runner_configured_for_a_deferred_transport_never_serves_in_memory() {
    for kind in [MessageQueueKind::Pulsar, MessageQueueKind::ZeroMq] {
        let config = RunnerConfig::default().with_message_queue(MessageQueueConfig {
            kind,
            url: Some("pulsar://broker:6650".to_owned()),
            ..MessageQueueConfig::default()
        });
        let runner = Runner::with_config(config);
        let queue = runner.message_queue();
        assert!(!queue.is_started());
        assert!(queue.topics().is_empty());
        assert!(!queue.has_subscriber("inbox"));

        let failures = [
            queue.start().await,
            queue.stop().await,
            queue.subscribe("inbox").await,
            queue.add_handler("inbox", Arc::new(EchoHandler)).await,
            queue.publish("inbox", QueueMessage::new(json!(1))).await,
            queue
                .request("inbox", QueueMessage::new(json!(1)), None)
                .await
                .map(|_| ()),
            queue.unsubscribe("inbox").await,
        ];
        for failure in failures {
            let error = failure.expect_err("deferred transport");
            assert_eq!(error.kind(), ErrorKind::Unsupported);
            let text = error.to_string();
            assert!(text.contains(kind.as_str()), "names the transport: {text}");
            assert!(text.contains("in_memory"), "points at the port: {text}");
        }
    }
}

#[tokio::test]
async fn a_deferred_transport_makes_start_unsupported() {
    for kind in [MessageQueueKind::Pulsar, MessageQueueKind::ZeroMq] {
        let runner = Runner::with_config(RunnerConfig::default().with_message_queue(
            MessageQueueConfig {
                kind,
                ..MessageQueueConfig::default()
            },
        ));
        let error = runner.start().await.expect_err("deferred transport");
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        let text = error.to_string();
        assert!(text.contains(kind.as_str()), "names the transport: {text}");
        assert!(text.contains("in_memory"), "points at the port: {text}");
        assert!(!runner.is_started(), "the runner stays stopped");

        // The slot still stands for the configured transport, not for a stopped
        // in-memory queue (which would answer `Error::Config`).
        let error = runner
            .message_queue()
            .publish("inbox", QueueMessage::new(json!(1)))
            .await
            .expect_err("deferred transport");
        assert_eq!(error.kind(), ErrorKind::Unsupported);
    }
}

#[tokio::test]
async fn a_default_runner_still_serves_the_in_memory_queue() {
    let runner = Runner::new();
    let queue = runner.message_queue();
    queue.start().await.expect("start");

    let (seen_tx, mut seen_rx) = mpsc::unbounded_channel();
    queue
        .add_handler("events", Arc::new(RecordingHandler { seen: seen_tx }))
        .await
        .expect("publish handler");
    queue
        .publish("events", QueueMessage::new(json!({"event": "ping"})))
        .await
        .expect("publish");
    assert_eq!(seen_rx.recv().await, Some(json!({"event": "ping"})));

    queue
        .add_handler("echo", Arc::new(EchoHandler))
        .await
        .expect("echo handler");
    let reply = queue
        .request("echo", QueueMessage::new(json!("ping")), None)
        .await
        .expect("request");
    assert_eq!(reply, json!("ping"));
    queue.stop().await.expect("stop");
}
