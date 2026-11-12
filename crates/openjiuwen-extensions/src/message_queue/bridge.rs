//! JSON bridge over any kernel [`MessageQueue`].
//!
//! Python's message-queue clients exchange `dict` payloads; the Rust kernel
//! keeps [`QueueMessage`] opaque and broker-neutral. [`QueueBridge`] is the
//! thin adapter in between: it wraps/unwraps [`serde_json::Value`] payloads, so
//! callers and handlers never touch the envelope, and it can expose a topic as
//! a [`BoxStream`] for streaming consumers.
//!
//! Error behaviour is deliberately transparent: a handler error returned by
//! [`JsonMessageHandler::handle_json`] reaches the `request` caller unchanged,
//! and a [`QueueMessage`] whose `error_code` is not [`SUCCESS`] becomes
//! [`Error::Transport`] on a topic stream.
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::stream::{self, BoxStream, StreamExt};
use openjiuwen_common::error::{Error, Result};
use openjiuwen_core::runner::mq::{MessageHandler, MessageQueue, QueueMessage, SUCCESS};
use serde_json::Value;
use tokio::sync::{mpsc, oneshot};

/// `tracing` target of every event this module emits.
const TRACING_TARGET: &str = "openjiuwen_extensions::message_queue";

/// Adapts a kernel [`MessageQueue`] to JSON payloads and topic streams.
///
/// The bridge owns nothing else: it clones the queue handle, so a bridge and the
/// queue (or `Runner`) can coexist. It holds no lock, hence no lock can be held
/// across an `.await`.
pub struct QueueBridge {
    /// The kernel queue every method delegates to.
    queue: Arc<dyn MessageQueue>,
}

impl QueueBridge {
    /// Wraps `queue`.
    #[must_use]
    pub fn new(queue: Arc<dyn MessageQueue>) -> Self {
        Self { queue }
    }

    /// Returns the wrapped queue (same allocation, not a copy).
    #[must_use]
    pub fn queue(&self) -> &Arc<dyn MessageQueue> {
        &self.queue
    }

    /// Publishes `payload` on `topic` and returns once it is enqueued.
    ///
    /// The handler's response is discarded, exactly like the kernel's
    /// `publish`; use [`request_json`](Self::request_json) to get one.
    ///
    /// # Errors
    ///
    /// The kernel queue's error unchanged: [`Error::Config`] when the queue is
    /// not started, [`Error::Transport`] when the topic worker is gone.
    pub async fn publish_json(&self, topic: &str, payload: Value) -> Result<()> {
        self.queue.publish(topic, QueueMessage::new(payload)).await
    }

    /// Sends `payload` on `topic` and returns the handler's JSON response.
    ///
    /// # Errors
    ///
    /// The handler's error unchanged; [`Error::Timeout`] when `timeout` (or the
    /// queue's configured default) elapses; [`Error::NotFound`] when the topic
    /// has no handler.
    pub async fn request_json(
        &self,
        topic: &str,
        payload: Value,
        timeout: Option<Duration>,
    ) -> Result<Value> {
        self.queue
            .request(topic, QueueMessage::new(payload), timeout)
            .await
    }

    /// Installs `handler` as the JSON handler of `topic`.
    ///
    /// Installing replaces any handler already registered for the topic, which
    /// is what the kernel's `add_handler` guarantees.
    ///
    /// # Errors
    ///
    /// The kernel queue's error unchanged ([`Error::Config`] when the queue is
    /// not started).
    pub async fn subscribe_json(
        &self,
        topic: &str,
        handler: Arc<dyn JsonMessageHandler>,
    ) -> Result<()> {
        self.queue
            .add_handler(topic, Arc::new(JsonHandlerAdapter { handler }))
            .await
    }

    /// Returns a stream of the JSON payloads published on `topic`.
    ///
    /// Registration is asynchronous: the returned stream awaits the handler
    /// installation before its first item, so a stream that cannot be installed
    /// yields that error once and ends. Deliveries are forwarded through a
    /// bounded channel of `capacity.max(1)`, so a slow consumer applies
    /// backpressure to the topic instead of growing an unbounded buffer. A
    /// delivery whose `error_code` is not [`SUCCESS`] becomes
    /// [`Error::Transport`] with the envelope's message.
    ///
    /// # Errors
    ///
    /// Returns the registration error on the stream (never from this call).
    ///
    /// # Panics
    ///
    /// Panics when called outside a tokio runtime, because registration is
    /// spawned on the current runtime.
    #[allow(clippy::unnecessary_wraps)]
    pub fn topic_stream(
        &self,
        topic: &str,
        capacity: usize,
    ) -> Result<BoxStream<'static, Result<Value>>> {
        let (sender, receiver) = mpsc::channel::<Result<Value>>(capacity.max(1));
        let (ready_tx, ready_rx) = oneshot::channel::<Result<()>>();
        let queue = Arc::clone(&self.queue);
        let topic_owned = topic.to_owned();
        tokio::spawn(async move {
            let registration = queue
                .add_handler(&topic_owned, Arc::new(StreamHandler { sender }))
                .await;
            if let Err(error) = &registration {
                tracing::warn!(
                    target: TRACING_TARGET,
                    topic = %topic_owned,
                    %error,
                    "message queue topic stream could not install its handler"
                );
            }
            // The receiver may already be gone; dropping the stream is not an
            // error worth reporting beyond the debug event in the handler.
            let _ = ready_tx.send(registration);
        });
        let state = StreamState {
            receiver,
            ready: Some(ready_rx),
            finished: false,
        };
        Ok(stream::unfold(state, next_stream_item).boxed())
    }
}

/// A JSON handler: the bridge's counterpart of [`MessageHandler`].
///
/// Implementations are shared across the topic worker, hence `Send + Sync`.
/// Publish deliverables may return [`Value::Null`]; a returned error is
/// forwarded unchanged to a `request` caller and logged-and-dropped on the
/// publish path.
#[async_trait]
pub trait JsonMessageHandler: Send + Sync {
    /// Handles one JSON payload and returns the request/reply response.
    ///
    /// # Errors
    ///
    /// Whatever the implementation fails with; the bridge never wraps or
    /// rewrites it.
    async fn handle_json(&self, payload: Value) -> Result<Value>;
}

/// Adapts a [`JsonMessageHandler`] to the kernel [`MessageHandler`] seam.
struct JsonHandlerAdapter {
    /// The JSON handler installed by [`QueueBridge::subscribe_json`].
    handler: Arc<dyn JsonMessageHandler>,
}

#[async_trait]
impl MessageHandler for JsonHandlerAdapter {
    async fn handle(&self, message: QueueMessage) -> Result<Value> {
        self.handler.handle_json(message.payload).await
    }
}

/// The state of one [`QueueBridge::topic_stream`].
struct StreamState {
    /// Deliveries forwarded by the stream handler.
    receiver: mpsc::Receiver<Result<Value>>,
    /// Registration outcome, awaited before the first delivery.
    ready: Option<oneshot::Receiver<Result<()>>>,
    /// Set once a registration error has been yielded (the stream then ends).
    finished: bool,
}

/// Produces one stream item: await registration first, then receive.
async fn next_stream_item(mut state: StreamState) -> Option<(Result<Value>, StreamState)> {
    if state.finished {
        return None;
    }
    if let Some(ready) = state.ready.take() {
        match ready.await {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                state.finished = true;
                return Some((Err(error), state));
            }
            Err(_) => {
                state.finished = true;
                return Some((
                    Err(Error::Transport(
                        "message queue topic stream registration task ended unexpectedly"
                            .to_owned(),
                    )),
                    state,
                ));
            }
        }
    }
    state.receiver.recv().await.map(|item| (item, state))
}

/// Forwards queue deliveries into a bounded stream channel.
struct StreamHandler {
    /// The stream's producer half.
    sender: mpsc::Sender<Result<Value>>,
}

#[async_trait]
impl MessageHandler for StreamHandler {
    async fn handle(&self, message: QueueMessage) -> Result<Value> {
        let item = if message.error_code == SUCCESS {
            Ok(message.payload)
        } else {
            Err(Error::Transport(message.error_msg))
        };
        if self.sender.send(item).await.is_err() {
            tracing::debug!(
                target: TRACING_TARGET,
                "message queue topic stream receiver was dropped"
            );
        }
        Ok(Value::Null)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]

    use std::sync::Arc;
    use std::time::Duration;

    use async_trait::async_trait;
    use futures::StreamExt;
    use openjiuwen_common::error::{Error, ErrorKind, Result};
    use openjiuwen_core::runner::mq::{InMemoryMessageQueue, MessageQueue, QueueMessage};
    use serde_json::{json, Value};
    use tokio::sync::mpsc;

    use super::{JsonMessageHandler, QueueBridge};

    /// Builds and starts an in-process queue with `capacity` slots per topic.
    async fn started_queue(capacity: usize) -> Arc<dyn MessageQueue> {
        let queue = InMemoryMessageQueue::with_capacity(capacity);
        queue.start().await.expect("start");
        queue
    }

    /// Waits until `topic` has a handler, with a deadline so a bug fails.
    async fn await_subscriber(bridge: &QueueBridge, topic: &str) {
        tokio::time::timeout(Duration::from_secs(1), async {
            while !bridge.queue().has_subscriber(topic) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("the topic stream installs its handler");
    }

    /// Returns the payload unchanged.
    struct EchoHandler;

    #[async_trait]
    impl JsonMessageHandler for EchoHandler {
        async fn handle_json(&self, payload: Value) -> Result<Value> {
            Ok(payload)
        }
    }

    /// Reports every payload it sees, then answers `null`.
    struct RecordingHandler {
        /// Receives each observed payload.
        seen: mpsc::UnboundedSender<Value>,
    }

    #[async_trait]
    impl JsonMessageHandler for RecordingHandler {
        async fn handle_json(&self, payload: Value) -> Result<Value> {
            let _ = self.seen.send(payload);
            Ok(Value::Null)
        }
    }

    /// Always fails, to prove errors propagate unchanged.
    struct FailingHandler;

    #[async_trait]
    impl JsonMessageHandler for FailingHandler {
        async fn handle_json(&self, _payload: Value) -> Result<Value> {
            Err(Error::Validation("handler rejected the payload".to_owned()))
        }
    }

    #[tokio::test]
    async fn publish_json_reaches_the_handler() {
        let bridge = QueueBridge::new(started_queue(8).await);
        let (seen_tx, mut seen_rx) = mpsc::unbounded_channel();
        bridge
            .subscribe_json("inbox", Arc::new(RecordingHandler { seen: seen_tx }))
            .await
            .expect("handler");

        bridge
            .publish_json("inbox", json!({"n": 1}))
            .await
            .expect("publish");
        let seen = tokio::time::timeout(Duration::from_secs(1), seen_rx.recv())
            .await
            .expect("the handler runs")
            .expect("payload");
        assert_eq!(seen, json!({"n": 1}));
    }

    #[tokio::test]
    async fn request_json_round_trips_the_handler_response() {
        let bridge = QueueBridge::new(started_queue(8).await);
        bridge
            .subscribe_json("echo", Arc::new(EchoHandler))
            .await
            .expect("handler");

        let reply = tokio::time::timeout(
            Duration::from_secs(1),
            bridge.request_json("echo", json!({"q": "hi"}), Some(Duration::from_secs(1))),
        )
        .await
        .expect("request completes")
        .expect("reply");
        assert_eq!(reply, json!({"q": "hi"}));
    }

    #[tokio::test]
    async fn handler_errors_propagate_unchanged() {
        let bridge = QueueBridge::new(started_queue(8).await);
        bridge
            .subscribe_json("failing", Arc::new(FailingHandler))
            .await
            .expect("handler");

        let error = tokio::time::timeout(
            Duration::from_secs(1),
            bridge.request_json("failing", json!(1), Some(Duration::from_secs(1))),
        )
        .await
        .expect("request completes")
        .expect_err("the handler fails");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("handler rejected the payload"));
    }

    #[tokio::test]
    async fn topic_stream_yields_published_payloads_in_order() {
        let bridge = QueueBridge::new(started_queue(8).await);
        let mut stream = bridge.topic_stream("events", 4).expect("stream");
        await_subscriber(&bridge, "events").await;

        for seq in 0..4 {
            bridge
                .publish_json("events", json!(seq))
                .await
                .expect("publish");
        }
        for seq in 0..4 {
            let item = tokio::time::timeout(Duration::from_secs(1), stream.next())
                .await
                .expect("item arrives")
                .expect("stream stays open");
            assert_eq!(item.expect("payload"), json!(seq));
        }
    }

    #[tokio::test]
    async fn topic_stream_maps_error_messages_to_transport_errors() {
        let bridge = QueueBridge::new(started_queue(8).await);
        let mut stream = bridge.topic_stream("errors", 2).expect("stream");
        await_subscriber(&bridge, "errors").await;

        bridge
            .queue()
            .publish(
                "errors",
                QueueMessage::new(json!("oops")).with_error(7, "boom"),
            )
            .await
            .expect("publish");
        let item = tokio::time::timeout(Duration::from_secs(1), stream.next())
            .await
            .expect("item arrives")
            .expect("stream stays open");
        let error = item.expect_err("an error envelope becomes a transport error");
        assert_eq!(error.kind(), ErrorKind::Transport);
        assert!(error.to_string().contains("boom"));
    }

    #[tokio::test]
    async fn queue_returns_the_same_handle() {
        let queue = started_queue(4).await;
        let bridge = QueueBridge::new(Arc::clone(&queue));
        assert!(Arc::ptr_eq(bridge.queue(), &queue));
        assert!(bridge.queue().is_started());
    }
}
