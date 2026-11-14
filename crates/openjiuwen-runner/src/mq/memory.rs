//! [`InMemoryMessageQueue`] — the tokio-native backend: one bounded channel and
//! one worker task per topic.
//!
//! Behavioural port of `MessageQueueInMemory` from
//! `openjiuwen/core/runner/message_queue_inmemory.py` and of
//! `FakeMessageQueue` in `openjiuwen/core/runner/dunner/dmessage_queue/message_queue_fake.py`,
//! minus the distributed wire (`DmqRequestMessage` / `DmqResponseMessage`, reply
//! topics).
//!
//! Deliberate differences from the Python fakes:
//!
//! - the worker awaits each handler before dequeuing the next delivery, so a
//!   topic is strictly FIFO (Python spawned one task per delivery and lost the
//!   order); a handler that requests **its own** topic therefore deadlocks —
//!   documented, not worked around;
//! - a request on a topic that is not subscribed (or has no handler) fails
//!   immediately instead of parking a future that nothing ever resolves
//!   (Python's "in-flight response orphaned forever" bug);
//! - `start` / `stop` are explicit and idempotent (Python's local queue answered
//!   `True` to both).

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use parking_lot::{Mutex, RwLock};
use serde_json::Value;
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;

use crate::config::MessageQueueConfig;
use crate::error::{not_started, request_timeout, transport};
use crate::mq::counters::Counters;
use crate::mq::message::{MessageHandler, QueueMessage};
use crate::mq::{MessageQueue, QueueStats};

/// `tracing` target of every event this module emits.
const TRACING_TARGET: &str = "openjiuwen_runner::mq";

/// The handler slot of one topic.
///
/// Shared with the worker task so a handler installed after `subscribe` takes
/// effect for the next delivery without rebuilding the topic.
type HandlerSlot = Arc<RwLock<Option<Arc<dyn MessageHandler>>>>;

/// One item in a topic's channel.
enum Delivery {
    /// A publish/subscribe delivery; the handler's response is discarded.
    Publish(QueueMessage),
    /// A request/reply delivery; the handler's result travels back on `reply`.
    Request {
        /// The request envelope.
        message: QueueMessage,
        /// The reply channel; dropped without a send when the worker stops.
        reply: oneshot::Sender<Result<Value>>,
    },
}

/// One live topic: its channel sender, its handler slot, and its worker task.
struct TopicState {
    /// Enqueue endpoint; cloning it is how `publish`/`request` hand work off.
    sender: mpsc::Sender<Delivery>,
    /// The handler the worker invokes, if one is installed.
    slot: HandlerSlot,
    /// The worker task; taken out of the map by `unsubscribe`/`stop` and awaited
    /// outside the table lock.
    task: JoinHandle<()>,
}

/// The mutable part of [`InMemoryMessageQueue`], guarded by one short-lived lock.
#[derive(Default)]
struct QueueState {
    /// Whether the queue accepts work.
    started: bool,
    /// Live topics, kept ordered so [`MessageQueue::topics`] is sorted for free.
    topics: BTreeMap<String, TopicState>,
}

/// The in-process queue: publish/subscribe plus request/reply, FIFO per topic.
///
/// Created by [`new`](Self::new) (or [`create_message_queue`](crate::mq::create_message_queue))
/// **unstarted**, because the runner starts its queue in `Runner::start`.
/// Sharing is by `Arc`; every method takes `&self`.
pub struct InMemoryMessageQueue {
    /// Per-topic capacity and the default request deadline.
    config: MessageQueueConfig,
    /// Topic table and start flag; held only for map operations, never across
    /// an `.await`.
    state: Mutex<QueueState>,
    /// Delivery and drop counters, shared with every topic worker.
    counters: Arc<Counters>,
}

impl InMemoryMessageQueue {
    /// Builds an unstarted queue from `config` (its `capacity` and
    /// `request_timeout_secs`; every other field is a broker concern).
    #[must_use]
    pub fn new(config: MessageQueueConfig) -> Arc<Self> {
        Arc::new(Self {
            config,
            state: Mutex::new(QueueState::default()),
            counters: Arc::new(Counters::default()),
        })
    }

    /// Builds an unstarted queue with `capacity` slots per topic and the default
    /// request deadline.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Arc<Self> {
        Self::new(MessageQueueConfig::default().with_capacity(capacity))
    }

    /// Returns how many deliveries a worker has started since the queue was
    /// created.
    ///
    /// Publish deliveries to a topic without a handler and requests rejected
    /// before enqueueing are not counted.
    #[must_use]
    pub fn deliveries(&self) -> u64 {
        self.stats().deliveries
    }

    /// Fails with `Error::Config` unless [`start`](MessageQueue::start) ran and
    /// [`stop`](MessageQueue::stop) did not.
    ///
    /// A refusal is counted in [`QueueStats::rejected_not_started`], so a host
    /// sampling the report sees calls made against a stopped queue.
    fn ensure_started(&self) -> Result<()> {
        if self.state.lock().started {
            Ok(())
        } else {
            Counters::bump(&self.counters.rejected_not_started);
            Err(not_started("message queue"))
        }
    }

    /// Returns the topic's sender and whether a handler is installed.
    fn topic_target(&self, topic: &str) -> Option<(mpsc::Sender<Delivery>, bool)> {
        let state = self.state.lock();
        state
            .topics
            .get(topic)
            .map(|entry| (entry.sender.clone(), entry.slot.read().is_some()))
    }

    /// Creates a topic: channel, handler slot, and the worker task draining it.
    ///
    /// The caller inserts the returned entry into the table; spawning here keeps
    /// the task owned by exactly one [`TopicState`].
    fn create_topic(
        topic: &str,
        config: &MessageQueueConfig,
        counters: Arc<Counters>,
    ) -> TopicState {
        let capacity = config.capacity.max(1);
        let (sender, receiver) = mpsc::channel(capacity);
        let slot: HandlerSlot = Arc::new(RwLock::new(None));
        let worker = TopicWorker {
            topic: topic.to_owned(),
            slot: Arc::clone(&slot),
            counters,
        };
        let task = tokio::spawn(worker.run(receiver));
        TopicState { sender, slot, task }
    }
}

#[async_trait]
impl MessageQueue for InMemoryMessageQueue {
    async fn start(&self) -> Result<()> {
        self.state.lock().started = true;
        Ok(())
    }

    async fn stop(&self) -> Result<()> {
        let topics = {
            let mut state = self.state.lock();
            state.started = false;
            std::mem::take(&mut state.topics)
        };
        for (topic, entry) in topics {
            // Dropping every sender closes the channel; the worker drains what is
            // already queued (in-flight requests still get their reply) and exits.
            drop(entry.sender);
            if let Err(error) = entry.task.await {
                tracing::warn!(
                    target: TRACING_TARGET,
                    topic = %topic,
                    %error,
                    "message queue topic worker did not stop cleanly"
                );
            }
        }
        Ok(())
    }

    fn is_started(&self) -> bool {
        self.state.lock().started
    }

    async fn subscribe(&self, topic: &str) -> Result<()> {
        self.ensure_started()?;
        let mut state = self.state.lock();
        if state.topics.contains_key(topic) {
            return Err(Error::Validation(format!(
                "message queue topic '{topic}' is already subscribed"
            )));
        }
        let entry = Self::create_topic(topic, &self.config, Arc::clone(&self.counters));
        state.topics.insert(topic.to_owned(), entry);
        Ok(())
    }

    async fn unsubscribe(&self, topic: &str) -> Result<()> {
        // Take the entry out under the lock, then close and join outside it: the
        // worker may still be awaiting a handler that touches the queue.
        let entry = self.state.lock().topics.remove(topic);
        let Some(entry) = entry else {
            tracing::debug!(
                target: TRACING_TARGET,
                topic = %topic,
                "message queue unsubscribe of an unknown topic"
            );
            return Ok(());
        };
        drop(entry.sender);
        if let Err(error) = entry.task.await {
            tracing::warn!(
                target: TRACING_TARGET,
                topic = %topic,
                %error,
                "message queue topic worker did not stop cleanly"
            );
        }
        Ok(())
    }

    async fn add_handler(&self, topic: &str, handler: Arc<dyn MessageHandler>) -> Result<()> {
        self.ensure_started()?;
        let mut state = self.state.lock();
        if let Some(existing) = state.topics.get(topic) {
            let slot = Arc::clone(&existing.slot);
            // The slot is installed before the guard is dropped, so the worker
            // never observes the topic as handled-but-empty.
            *slot.write() = Some(handler);
            return Ok(());
        }
        let entry = Self::create_topic(topic, &self.config, Arc::clone(&self.counters));
        *entry.slot.write() = Some(handler);
        state.topics.insert(topic.to_owned(), entry);
        Ok(())
    }

    async fn publish(&self, topic: &str, message: QueueMessage) -> Result<()> {
        self.ensure_started()?;
        let Some((sender, has_handler)) = self.topic_target(topic) else {
            Counters::bump(&self.counters.dropped_no_worker);
            tracing::debug!(
                target: TRACING_TARGET,
                topic = %topic,
                "message queue publish to an unsubscribed topic"
            );
            return Ok(());
        };
        if !has_handler {
            Counters::bump(&self.counters.dropped_no_handler);
            tracing::debug!(
                target: TRACING_TARGET,
                topic = %topic,
                "message queue publish to a topic without a handler"
            );
            return Ok(());
        }
        // A full topic applies backpressure here; a closed channel means the
        // worker stopped, and a publisher can neither observe nor fix that.
        if sender.send(Delivery::Publish(message)).await.is_err() {
            Counters::bump(&self.counters.dropped_no_worker);
            tracing::warn!(
                target: TRACING_TARGET,
                topic = %topic,
                "message queue topic worker stopped; published message dropped"
            );
        }
        Ok(())
    }

    async fn request(
        &self,
        topic: &str,
        message: QueueMessage,
        timeout: Option<Duration>,
    ) -> Result<Value> {
        self.ensure_started()?;
        let Some((sender, has_handler)) = self.topic_target(topic) else {
            return Err(Error::NotFound(format!(
                "message queue topic '{topic}' is not subscribed"
            )));
        };
        if !has_handler {
            return Err(Error::NotFound(format!(
                "message queue topic '{topic}' has no handler"
            )));
        }
        let (reply, reply_rx) = oneshot::channel();
        sender
            .send(Delivery::Request { message, reply })
            .await
            .map_err(|_| worker_stopped(topic))?;
        match timeout.or_else(|| self.config.request_timeout()) {
            Some(deadline) => match tokio::time::timeout(deadline, reply_rx).await {
                Ok(reply) => reply.map_err(|_| worker_stopped(topic))?,
                // The worker keeps running and its reply is discarded: the caller
                // owns the deadline, the handler owns the work.
                Err(_) => Err(request_timeout(topic, deadline)),
            },
            None => reply_rx.await.map_err(|_| worker_stopped(topic))?,
        }
    }

    fn topics(&self) -> Vec<String> {
        self.state.lock().topics.keys().cloned().collect()
    }

    fn has_subscriber(&self, topic: &str) -> bool {
        let state = self.state.lock();
        state
            .topics
            .get(topic)
            .is_some_and(|entry| entry.slot.read().is_some())
    }

    /// Reads the topic table (topic and handler counts) under one short lock and
    /// the counters without one.
    fn stats(&self) -> QueueStats {
        let (started, topics, handlers) = {
            let state = self.state.lock();
            let handlers = state
                .topics
                .values()
                .filter(|entry| entry.slot.read().is_some())
                .count();
            (state.started, state.topics.len(), handlers)
        };
        QueueStats {
            started,
            topics,
            handlers,
            deliveries: Counters::read(&self.counters.deliveries),
            dropped_no_worker: Counters::read(&self.counters.dropped_no_worker),
            dropped_no_handler: Counters::read(&self.counters.dropped_no_handler),
            handler_errors: Counters::read(&self.counters.handler_errors),
            rejected_not_started: Counters::read(&self.counters.rejected_not_started),
        }
    }
}

/// The task draining one topic in FIFO order.
struct TopicWorker {
    /// Topic name, for log events.
    topic: String,
    /// The handler installed by `add_handler`; looked up per delivery.
    slot: HandlerSlot,
    /// Shared delivery and drop counters.
    counters: Arc<Counters>,
}

impl TopicWorker {
    /// Drains the channel until it closes, awaiting each handler in turn.
    async fn run(self, mut receiver: mpsc::Receiver<Delivery>) {
        while let Some(delivery) = receiver.recv().await {
            match delivery {
                Delivery::Publish(message) => {
                    let Some(handler) = self.handler() else {
                        Counters::bump(&self.counters.dropped_no_handler);
                        tracing::debug!(
                            target: TRACING_TARGET,
                            topic = %self.topic,
                            "message queue delivery dropped: no handler"
                        );
                        continue;
                    };
                    Counters::bump(&self.counters.deliveries);
                    if let Err(error) = handler.handle(message).await {
                        Counters::bump(&self.counters.handler_errors);
                        // Publish-path errors are unobservable by the publisher.
                        tracing::warn!(
                            target: TRACING_TARGET,
                            topic = %self.topic,
                            %error,
                            "message queue publish delivery failed"
                        );
                    }
                }
                Delivery::Request { message, reply } => {
                    let response = match self.handler() {
                        Some(handler) => {
                            Counters::bump(&self.counters.deliveries);
                            let response = handler.handle(message).await;
                            if response.is_err() {
                                Counters::bump(&self.counters.handler_errors);
                            }
                            response
                        }
                        None => {
                            // The handler was removed after the request was queued.
                            Counters::bump(&self.counters.dropped_no_handler);
                            Err(Error::NotFound(format!(
                                "message queue topic '{}' has no handler",
                                self.topic
                            )))
                        }
                    };
                    // A failed send only means the caller timed out or went away.
                    let _ = reply.send(response);
                }
            }
        }
        tracing::debug!(
            target: TRACING_TARGET,
            topic = %self.topic,
            "message queue topic worker stopped"
        );
    }

    /// Returns the installed handler, if any.
    fn handler(&self) -> Option<Arc<dyn MessageHandler>> {
        self.slot.read().as_ref().map(Arc::clone)
    }
}

/// The transport error raised when a topic worker stops before replying.
fn worker_stopped(topic: &str) -> Error {
    transport(format!(
        "message queue topic '{topic}' worker stopped before replying"
    ))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]

    use super::*;
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;
    use std::sync::Mutex as StdMutex;

    /// Records every payload it handles and replies with an acknowledgement.
    struct RecordingHandler {
        seen: mpsc::UnboundedSender<Value>,
    }

    #[async_trait]
    impl MessageHandler for RecordingHandler {
        async fn handle(&self, message: QueueMessage) -> Result<Value> {
            let _ = self.seen.send(message.payload);
            Ok(json!({"received": true}))
        }
    }

    /// Echoes the payload back, so a request can assert the reply plumbing.
    struct EchoHandler;

    #[async_trait]
    impl MessageHandler for EchoHandler {
        async fn handle(&self, message: QueueMessage) -> Result<Value> {
            Ok(message.payload)
        }
    }

    /// Fails every delivery with a caller-visible error.
    struct FailingHandler;

    #[async_trait]
    impl MessageHandler for FailingHandler {
        async fn handle(&self, _message: QueueMessage) -> Result<Value> {
            Err(Error::Tool("handler rejected".to_owned()))
        }
    }

    /// Signals that it ran, then fails: the publish path has no reply channel, so
    /// the signal is how a test learns the delivery was handled.
    struct FailingReporter {
        seen: mpsc::UnboundedSender<Value>,
    }

    #[async_trait]
    impl MessageHandler for FailingReporter {
        async fn handle(&self, message: QueueMessage) -> Result<Value> {
            let _ = self.seen.send(message.payload);
            Err(Error::Tool("handler rejected".to_owned()))
        }
    }

    /// Blocks on a gate released by the test, then reports completion.
    struct GatedHandler {
        gate: StdMutex<Option<oneshot::Receiver<()>>>,
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

    /// Panics, proving a worker death is visible to a requester.
    struct PanickingHandler;

    #[async_trait]
    impl MessageHandler for PanickingHandler {
        async fn handle(&self, _message: QueueMessage) -> Result<Value> {
            panic!("handler exploded");
        }
    }

    /// Forwards a request to another topic, proving no global lock is held.
    struct ForwardingHandler {
        queue: Arc<dyn MessageQueue>,
        target: String,
    }

    #[async_trait]
    impl MessageHandler for ForwardingHandler {
        async fn handle(&self, message: QueueMessage) -> Result<Value> {
            self.queue
                .request(&self.target, message, Some(Duration::from_secs(5)))
                .await
        }
    }

    /// Builds a started queue with a `#`-sized capacity.
    async fn started_queue(capacity: usize) -> Arc<InMemoryMessageQueue> {
        let queue = InMemoryMessageQueue::with_capacity(capacity);
        queue.start().await.expect("start");
        queue
    }

    #[tokio::test]
    async fn publish_is_fifo_per_topic() {
        let queue = started_queue(8).await;
        let (seen_tx, mut seen_rx) = mpsc::unbounded_channel();
        queue
            .add_handler("inbox", Arc::new(RecordingHandler { seen: seen_tx }))
            .await
            .expect("handler");

        for seq in 0..5 {
            queue
                .publish("inbox", QueueMessage::new(json!(seq)))
                .await
                .expect("publish");
        }
        for seq in 0..5 {
            assert_eq!(seen_rx.recv().await, Some(json!(seq)));
        }
        assert_eq!(queue.deliveries(), 5);
        queue.stop().await.expect("stop");
    }

    #[tokio::test]
    async fn publish_to_an_unsubscribed_topic_is_a_no_op() {
        let queue = started_queue(8).await;
        queue
            .publish("nobody", QueueMessage::new(json!(1)))
            .await
            .expect("publish is a no-op");
        assert!(queue.topics().is_empty());
        assert_eq!(queue.deliveries(), 0);
    }

    #[tokio::test]
    async fn publish_to_a_topic_without_a_handler_is_a_no_op() {
        let queue = started_queue(8).await;
        queue.subscribe("inbox").await.expect("subscribe");
        queue
            .publish("inbox", QueueMessage::new(json!(1)))
            .await
            .expect("publish is a no-op");
        assert_eq!(queue.deliveries(), 0);
        assert!(!queue.has_subscriber("inbox"));
    }

    #[tokio::test]
    async fn request_on_an_unsubscribed_topic_is_not_found() {
        let queue = started_queue(8).await;
        let error = queue
            .request("nobody", QueueMessage::new(json!(1)), None)
            .await
            .expect_err("unsubscribed topic");
        assert_eq!(error.kind(), ErrorKind::NotFound);
        assert_eq!(
            error.to_string(),
            "not_found: message queue topic 'nobody' is not subscribed"
        );
    }

    #[tokio::test]
    async fn request_on_a_topic_without_a_handler_is_not_found() {
        let queue = started_queue(8).await;
        queue.subscribe("inbox").await.expect("subscribe");
        let error = queue
            .request("inbox", QueueMessage::new(json!(1)), None)
            .await
            .expect_err("no handler");
        assert_eq!(error.kind(), ErrorKind::NotFound);
        assert_eq!(
            error.to_string(),
            "not_found: message queue topic 'inbox' has no handler"
        );
    }

    #[tokio::test]
    async fn request_returns_the_handler_response() {
        let queue = started_queue(8).await;
        queue
            .add_handler("inbox", Arc::new(EchoHandler))
            .await
            .expect("handler");
        let reply = queue
            .request("inbox", QueueMessage::new(json!({"query": "hi"})), None)
            .await
            .expect("request");
        assert_eq!(reply, json!({"query": "hi"}));
        assert_eq!(queue.deliveries(), 1);
    }

    #[tokio::test]
    async fn request_propagates_the_handler_error_unchanged() {
        let queue = started_queue(8).await;
        queue
            .add_handler("inbox", Arc::new(FailingHandler))
            .await
            .expect("handler");
        let error = queue
            .request("inbox", QueueMessage::new(json!(1)), None)
            .await
            .expect_err("handler failure");
        assert_eq!(error.kind(), ErrorKind::Tool);
        assert_eq!(error.to_string(), "tool: handler rejected");
    }

    #[tokio::test]
    async fn request_times_out_and_the_handler_keeps_running() {
        let queue = started_queue(8).await;
        let (release_tx, release_rx) = oneshot::channel();
        let (done_tx, mut done_rx) = mpsc::unbounded_channel();
        let handler = Arc::new(GatedHandler {
            gate: StdMutex::new(Some(release_rx)),
            done: done_tx,
        });
        queue.add_handler("slow", handler).await.expect("handler");

        let error = queue
            .request(
                "slow",
                QueueMessage::new(json!("payload")),
                Some(Duration::from_millis(20)),
            )
            .await
            .expect_err("deadline");
        assert_eq!(error.kind(), ErrorKind::Timeout);
        assert_eq!(
            error.to_string(),
            "timeout: message queue request on 'slow' timed out after 20ms"
        );
        assert_eq!(queue.deliveries(), 1, "the worker started the delivery");

        // The handler was still running after the caller gave up.
        release_tx.send(()).expect("release the handler");
        assert_eq!(done_rx.recv().await, Some(json!("payload")));
    }

    #[tokio::test]
    async fn a_handler_may_request_another_topic() {
        let queue = started_queue(8).await;
        queue
            .add_handler("backend", Arc::new(EchoHandler))
            .await
            .expect("backend handler");
        let dyn_queue: Arc<dyn MessageQueue> = Arc::clone(&queue) as Arc<dyn MessageQueue>;
        queue
            .add_handler(
                "frontend",
                Arc::new(ForwardingHandler {
                    queue: dyn_queue,
                    target: "backend".to_owned(),
                }),
            )
            .await
            .expect("frontend handler");

        let reply = queue
            .request("frontend", QueueMessage::new(json!("ping")), None)
            .await
            .expect("forwarded request");
        assert_eq!(reply, json!("ping"));
        assert_eq!(queue.deliveries(), 2);
    }

    #[tokio::test]
    async fn a_stopped_worker_surfaces_as_a_transport_error() {
        let queue = started_queue(8).await;
        queue
            .add_handler("boom", Arc::new(PanickingHandler))
            .await
            .expect("handler");
        let error = queue
            .request("boom", QueueMessage::new(json!(1)), None)
            .await
            .expect_err("worker stopped");
        assert_eq!(error.kind(), ErrorKind::Transport);
        assert!(
            error.to_string().contains("worker stopped before replying"),
            "unexpected message: {error}"
        );
    }

    #[tokio::test]
    async fn operations_before_start_are_config_errors() {
        let queue = InMemoryMessageQueue::with_capacity(4);
        let errors = [
            queue
                .publish("inbox", QueueMessage::new(json!(1)))
                .await
                .expect_err("publish before start"),
            queue
                .subscribe("inbox")
                .await
                .expect_err("subscribe before start"),
            queue
                .add_handler("inbox", Arc::new(EchoHandler))
                .await
                .expect_err("add_handler before start"),
            queue
                .request("inbox", QueueMessage::new(json!(1)), None)
                .await
                .expect_err("request before start"),
        ];
        for error in errors {
            assert_eq!(error.kind(), ErrorKind::Config);
            assert_eq!(error.to_string(), "config: message queue is not started");
        }
        assert!(!queue.is_started());
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
        assert!(
            error.to_string().contains("is already subscribed"),
            "unexpected message: {error}"
        );
        assert_eq!(queue.topics(), vec!["inbox".to_owned()]);
    }

    #[tokio::test]
    async fn start_and_stop_are_idempotent() {
        let queue = InMemoryMessageQueue::with_capacity(4);
        queue.start().await.expect("start");
        queue.start().await.expect("start again");
        assert!(queue.is_started());
        queue.subscribe("inbox").await.expect("subscribe");

        queue.stop().await.expect("stop");
        queue.stop().await.expect("stop again");
        assert!(!queue.is_started());
        assert!(queue.topics().is_empty());
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
            .publish("inbox", QueueMessage::new(json!(1)))
            .await
            .expect("publish");
        assert_eq!(seen_rx.recv().await, Some(json!(1)));

        queue.unsubscribe("inbox").await.expect("unsubscribe");
        assert!(queue.topics().is_empty());
        assert!(!queue.has_subscriber("inbox"));
        // Unknown topics are a no-op, so teardown is idempotent too.
        queue.unsubscribe("inbox").await.expect("unsubscribe again");
        // The worker and its handler are gone: publishing is a logged no-op.
        queue
            .publish("inbox", QueueMessage::new(json!(2)))
            .await
            .expect("publish after unsubscribe");
        assert_eq!(queue.deliveries(), 1);
    }

    #[tokio::test]
    async fn start_after_stop_re_enables_the_queue_without_resurrecting_topics() {
        let queue = InMemoryMessageQueue::with_capacity(4);
        queue.start().await.expect("start");
        queue.subscribe("inbox").await.expect("subscribe");
        queue.stop().await.expect("stop");

        queue.start().await.expect("restart");
        assert!(queue.is_started());
        assert!(queue.topics().is_empty(), "topics are emptied by stop");

        let (seen_tx, mut seen_rx) = mpsc::unbounded_channel();
        queue
            .add_handler("inbox", Arc::new(RecordingHandler { seen: seen_tx }))
            .await
            .expect("handler after restart");
        queue
            .publish("inbox", QueueMessage::new(json!("again")))
            .await
            .expect("publish after restart");
        assert_eq!(seen_rx.recv().await, Some(json!("again")));
    }

    #[tokio::test]
    async fn topics_are_sorted_and_handlers_are_replaced() {
        let queue = started_queue(4).await;
        queue.subscribe("zeta").await.expect("subscribe");
        queue
            .add_handler("alpha", Arc::new(EchoHandler))
            .await
            .expect("handler");
        assert_eq!(queue.topics(), vec!["alpha".to_owned(), "zeta".to_owned()]);
        assert!(queue.has_subscriber("alpha"));
        assert!(!queue.has_subscriber("zeta"));

        // Replacing a handler takes effect on the next delivery.
        let (seen_tx, mut seen_rx) = mpsc::unbounded_channel();
        queue
            .add_handler("alpha", Arc::new(RecordingHandler { seen: seen_tx }))
            .await
            .expect("replace handler");
        queue
            .publish("alpha", QueueMessage::new(json!("replaced")))
            .await
            .expect("publish");
        assert_eq!(seen_rx.recv().await, Some(json!("replaced")));
    }

    #[tokio::test]
    async fn explicit_timeout_overrides_the_configured_default() {
        let config = MessageQueueConfig::default()
            .with_capacity(4)
            .with_request_timeout_secs(None);
        let queue = InMemoryMessageQueue::new(config);
        queue.start().await.expect("start");
        queue
            .add_handler("inbox", Arc::new(EchoHandler))
            .await
            .expect("handler");
        // No configured default: the explicit deadline is the only one applied.
        let reply = queue
            .request(
                "inbox",
                QueueMessage::new(json!("bounded")),
                Some(Duration::from_secs(5)),
            )
            .await
            .expect("request");
        assert_eq!(reply, json!("bounded"));
    }

    #[tokio::test]
    async fn stats_count_topics_handlers_and_deliveries() {
        let queue = started_queue(4).await;
        assert!(queue.stats().started);
        assert!(
            !queue.stats().is_empty(),
            "a started queue reports its state"
        );
        queue.subscribe("idle").await.expect("subscribe");
        queue
            .add_handler("echo", Arc::new(EchoHandler))
            .await
            .expect("handler");
        let reply = queue
            .request("echo", QueueMessage::new(json!(1)), None)
            .await
            .expect("request");
        assert_eq!(reply, json!(1));

        let stats = queue.stats();
        assert_eq!(stats.topics, 2);
        assert_eq!(stats.handlers, 1, "only the echo topic has a handler");
        assert_eq!(stats.deliveries, 1);
        assert_eq!(stats.deliveries, queue.deliveries(), "one counter");
        assert_eq!(stats.dropped(), 0);
        assert_eq!(stats.handler_errors, 0);
        assert_eq!(stats.rejected_not_started, 0);
    }

    #[tokio::test]
    async fn stats_count_drops_errors_and_refusals() {
        let queue = InMemoryMessageQueue::with_capacity(4);
        // Refused while stopped: rejected, not dropped, and no topic is created.
        let stopped = queue.publish("nobody", QueueMessage::new(json!(0))).await;
        assert!(stopped.is_err());
        let stats = queue.stats();
        assert!(!stats.started);
        assert_eq!(stats.rejected_not_started, 1);
        assert_eq!(stats.dropped(), 0);

        queue.start().await.expect("start");
        // A publish to a topic without a worker.
        queue
            .publish("nobody", QueueMessage::new(json!(1)))
            .await
            .expect("publish");
        // A publish to a worker without a handler.
        queue.subscribe("idle").await.expect("subscribe");
        queue
            .publish("idle", QueueMessage::new(json!(2)))
            .await
            .expect("publish");
        let stats = queue.stats();
        assert_eq!(stats.dropped_no_worker, 1);
        assert_eq!(stats.dropped_no_handler, 1);
        assert_eq!(stats.dropped(), 2);
        assert_eq!(stats.deliveries, 0, "a drop is not a delivery");

        // A handler error on the request path and on the publish path.
        let (seen_tx, mut seen_rx) = mpsc::unbounded_channel();
        queue
            .add_handler("boom", Arc::new(FailingReporter { seen: seen_tx }))
            .await
            .expect("handler");
        queue
            .request("boom", QueueMessage::new(json!(3)), None)
            .await
            .expect_err("handler failure");
        assert_eq!(
            seen_rx.recv().await,
            Some(json!(3)),
            "the request delivery ran"
        );
        queue
            .publish("boom", QueueMessage::new(json!(4)))
            .await
            .expect("publish");
        assert_eq!(seen_rx.recv().await, Some(json!(4)), "the delivery ran");
        let stats = queue.stats();
        assert_eq!(stats.handler_errors, 2);
        assert_eq!(stats.deliveries, 2);
        assert_eq!(stats.topics, 2);
        assert_eq!(stats.handlers, 1);

        // Stopping empties the topics and keeps the counters.
        queue.stop().await.expect("stop");
        let stats = queue.stats();
        assert!(!stats.started);
        assert_eq!(stats.topics, 0);
        assert_eq!(stats.handlers, 0);
        assert_eq!(stats.deliveries, 2);
        assert_eq!(stats.dropped(), 2);
    }
}
