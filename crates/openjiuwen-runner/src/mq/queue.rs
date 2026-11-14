//! The [`MessageQueue`] trait — the seam every queue backend implements — and
//! the backend factory [`create_message_queue`].
//!
//! Python parity: `MessageQueueBase` in
//! `openjiuwen/core/runner/message_queue_base.py`. The Python API returned a
//! `SubscriptionBase` object from `subscribe`; here the subscription is just the
//! topic name and the handler is attached with
//! [`add_handler`](MessageQueue::add_handler), which keeps the trait object-safe
//! for the broker backends a host plugs in.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::{MessageQueueConfig, MessageQueueKind};
use crate::error::unsupported;
use crate::mq::message::{MessageHandler, QueueMessage};
use crate::mq::InMemoryMessageQueue;

/// What a queue backend can report about the work it has done.
///
/// Every counter is a plain monotonic total since the backend was created, so a
/// host can sample the report repeatedly and derive rates itself. The report is
/// serializable (`snake_case`, unknown fields tolerated, missing fields default)
/// so a host can ship it to whatever observes its process.
///
/// A backend that cannot keep one of these counters leaves it at `0`; only
/// [`stats`](MessageQueue::stats) decides what it can honestly answer.
///
/// ```
/// use openjiuwen_runner::mq::QueueStats;
///
/// let stats = QueueStats { deliveries: 3, ..QueueStats::default() };
/// assert!(!stats.is_empty());
/// assert_eq!(stats.dropped(), 0);
/// assert_eq!(QueueStats::default(), QueueStats::default());
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct QueueStats {
    /// Whether the backend is between `start` and `stop`.
    pub started: bool,
    /// Topics the backend currently owns (subscribed or carrying a handler).
    pub topics: usize,
    /// Topics that currently have a handler installed.
    pub handlers: usize,
    /// Deliveries a worker actually started (a handler was present).
    pub deliveries: u64,
    /// Delivered messages dropped because their topic had no worker: a publish to
    /// an unsubscribed topic, or a publish whose worker had already stopped.
    pub dropped_no_worker: u64,
    /// Delivered messages dropped because their topic had no handler installed.
    pub dropped_no_handler: u64,
    /// Handler invocations that returned an error, on the publish and the request
    /// path alike.
    pub handler_errors: u64,
    /// Operations refused because the queue was stopped.
    pub rejected_not_started: u64,
}

impl QueueStats {
    /// Returns `true` when this is the empty report ([`QueueStats::default`]):
    /// the queue is stopped, owns nothing, and has recorded no counter.
    ///
    /// That is the report a backend performing no work (a deferred transport)
    /// answers, and the one a fresh [`InMemoryMessageQueue`] answers before
    /// [`start`](MessageQueue::start).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }

    /// The number of messages dropped instead of delivered
    /// ([`dropped_no_worker`](Self::dropped_no_worker) +
    /// [`dropped_no_handler`](Self::dropped_no_handler)).
    ///
    /// Handler failures are not drops: the handler ran and its error was reported
    /// (see [`handler_errors`](Self::handler_errors)).
    #[must_use]
    pub fn dropped(&self) -> u64 {
        self.dropped_no_worker + self.dropped_no_handler
    }
}

/// The runner-owned topic queue: publish/subscribe plus request/reply.
///
/// Topics are created by [`subscribe`](Self::subscribe) (or implicitly by
/// [`add_handler`](Self::add_handler)). A queue is created stopped and becomes
/// usable after [`start`](Self::start); operations on a stopped queue fail with
/// `Error::Config`, matching Python's explicit `Runner.start()` step.
///
/// Implementations are shared (`Arc<dyn MessageQueue>`), so every method takes
/// `&self`; interior mutability and task ownership are the backend's business.
#[async_trait::async_trait]
pub trait MessageQueue: Send + Sync {
    /// Starts the backend; idempotent.
    ///
    /// # Errors
    ///
    /// A backend that cannot start (a broker that refuses the connection) fails
    /// here; the in-process queue never does.
    async fn start(&self) -> Result<()>;

    /// Stops the backend and joins every topic worker; idempotent.
    ///
    /// Topics are emptied: a later [`start`](Self::start) re-enables the queue
    /// without resurrecting them.
    ///
    /// # Errors
    ///
    /// A backend that cannot stop cleanly fails here; the in-process queue logs
    /// join failures at `warn` instead of propagating them.
    async fn stop(&self) -> Result<()>;

    /// Returns `true` between [`start`](Self::start) and [`stop`](Self::stop).
    fn is_started(&self) -> bool;

    /// Creates the topic (its channel and worker) and marks it subscribed.
    ///
    /// # Errors
    ///
    /// `Error::Config` when the queue is not started, `Error::Validation` when
    /// the topic is already subscribed.
    async fn subscribe(&self, topic: &str) -> Result<()>;

    /// Tears the topic down: closes its channel, joins its worker, and drops its
    /// handler.
    ///
    /// An unknown topic is a no-op (`Ok(())`).
    ///
    /// # Errors
    ///
    /// A backend that cannot unsubscribe fails here; the in-process queue logs
    /// join failures at `warn` instead of propagating them.
    async fn unsubscribe(&self, topic: &str) -> Result<()>;

    /// Installs `handler` for `topic`, subscribing to the topic when needed.
    ///
    /// Replacing an existing handler is allowed (the new one takes effect for
    /// the next delivery).
    ///
    /// # Errors
    ///
    /// `Error::Config` when the queue is not started.
    async fn add_handler(&self, topic: &str, handler: Arc<dyn MessageHandler>) -> Result<()>;

    /// Enqueues `message` on `topic` and returns once it is queued.
    ///
    /// The delivery is not awaited: a full queue applies backpressure, and a
    /// handler error cannot be observed by the publisher (it is logged).
    /// Publishing to a topic with no worker or no handler is a logged no-op.
    ///
    /// # Errors
    ///
    /// `Error::Config` when the queue is not started.
    async fn publish(&self, topic: &str, message: QueueMessage) -> Result<()>;

    /// Enqueues `message` on `topic` and waits for the handler's response.
    ///
    /// `timeout` is the caller's deadline; the backend's configured default
    /// applies when it is `None`, and an absent default waits forever.
    ///
    /// # Errors
    ///
    /// `Error::Config` when the queue is not started, `Error::NotFound` when the
    /// topic has no worker or no handler, `Error::Timeout` when the deadline
    /// elapses (the handler keeps running), `Error::Transport` when the worker
    /// stops before replying, and otherwise the handler's error unchanged.
    async fn request(
        &self,
        topic: &str,
        message: QueueMessage,
        timeout: Option<Duration>,
    ) -> Result<Value>;

    /// Returns the subscribed topic names, sorted.
    fn topics(&self) -> Vec<String>;

    /// Returns `true` when `topic` has a handler installed.
    fn has_subscriber(&self, topic: &str) -> bool;

    /// Returns the counters this backend can honestly report.
    ///
    /// The default is the empty report ([`QueueStats::default`]): a backend that
    /// cannot report statistics returns it rather than fabricating counters — a
    /// report never claims work the backend did not do. The method is synchronous
    /// and object-safe on purpose, so a host holding `Arc<dyn MessageQueue>` can
    /// sample a queue from synchronous code without an `.await` or a downcast.
    fn stats(&self) -> QueueStats {
        QueueStats::default()
    }
}

/// The typed refusal of a transport this build does not compile.
///
/// The wording names the configured transport (`pulsar` / `zeromq`) and points at
/// the backend that does work (`in_memory`); [`super::DeferredMessageQueue`] and
/// [`create_message_queue`] share it so a host sees one message for one cause.
fn deferred_transport(kind: MessageQueueKind) -> Error {
    unsupported(
        &format!("{} message queue", kind.as_str()),
        "use `in_memory`",
    )
}

/// The typed stand-in for a queue transport this build does not compile.
///
/// [`crate::Runner::with_config`] cannot fail, yet a configuration may select a
/// transport this build cannot build ([`MessageQueueKind::Pulsar`],
/// [`MessageQueueKind::ZeroMq`]). Rather than silently serving an in-memory queue
/// under another backend's name, the runner installs this placeholder: it carries
/// the configured kind and answers
/// [`openjiuwen_common::error::Error::Unsupported`] — naming the missing
/// transport and pointing at `in_memory` — from **every** operation.
///
/// That matters for a host that never calls [`crate::Runner::start`] (pooled team
/// runs deliberately do not): it still gets the typed failure from
/// [`crate::Runner::message_queue`] instead of a working queue of the wrong kind.
///
/// [`MessageQueueKind::InMemory`] is never deferred by the runner
/// ([`create_message_queue`] builds the real backend); constructing this
/// placeholder for it is allowed and still refuses every operation.
pub struct DeferredMessageQueue {
    /// The configured transport this placeholder stands in for; reported by
    /// [`kind`](Self::kind) and `Debug`, and named in every error.
    kind: MessageQueueKind,
}

impl std::fmt::Debug for DeferredMessageQueue {
    /// Prints the configured transport under its wire name (`pulsar` /
    /// `zeromq`), the same word the errors use; the placeholder holds no other
    /// state.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeferredMessageQueue")
            .field("kind", &self.kind.as_str())
            .finish()
    }
}

impl DeferredMessageQueue {
    /// Builds the placeholder for `kind`.
    #[must_use]
    pub fn new(kind: MessageQueueKind) -> Arc<Self> {
        Arc::new(Self { kind })
    }

    /// The configured transport this queue stands in for.
    ///
    /// Never [`MessageQueueKind::InMemory`] when the runner built it; the error
    /// every operation returns names the same transport.
    #[must_use]
    pub fn kind(&self) -> MessageQueueKind {
        self.kind
    }

    /// The error every operation of this placeholder returns.
    fn error(&self) -> Error {
        deferred_transport(self.kind)
    }
}

#[async_trait]
impl MessageQueue for DeferredMessageQueue {
    /// Always `Error::Unsupported`: the transport is not compiled into this
    /// build.
    async fn start(&self) -> Result<()> {
        Err(self.error())
    }

    /// Always `Error::Unsupported`; there is nothing to stop.
    async fn stop(&self) -> Result<()> {
        Err(self.error())
    }

    /// Always `false`: a deferred queue never starts.
    fn is_started(&self) -> bool {
        false
    }

    /// Always `Error::Unsupported`.
    async fn subscribe(&self, _topic: &str) -> Result<()> {
        Err(self.error())
    }

    /// Always `Error::Unsupported`.
    async fn unsubscribe(&self, _topic: &str) -> Result<()> {
        Err(self.error())
    }

    /// Always `Error::Unsupported`.
    async fn add_handler(&self, _topic: &str, _handler: Arc<dyn MessageHandler>) -> Result<()> {
        Err(self.error())
    }

    /// Always `Error::Unsupported`.
    async fn publish(&self, _topic: &str, _message: QueueMessage) -> Result<()> {
        Err(self.error())
    }

    /// Always `Error::Unsupported`.
    async fn request(
        &self,
        _topic: &str,
        _message: QueueMessage,
        _timeout: Option<Duration>,
    ) -> Result<Value> {
        Err(self.error())
    }

    /// Always empty: a deferred queue owns no topics.
    fn topics(&self) -> Vec<String> {
        Vec::new()
    }

    /// Always `false`: a deferred queue has no handlers.
    fn has_subscriber(&self, _topic: &str) -> bool {
        false
    }

    /// Always the empty report: a deferred transport performs no work, so an
    /// empty [`QueueStats`] is the truth rather than a placeholder.
    fn stats(&self) -> QueueStats {
        QueueStats::default()
    }
}

/// Builds the backend `config` selects, **without failing**.
///
/// [`MessageQueueKind::InMemory`] yields an unstarted
/// [`InMemoryMessageQueue`] (the caller starts it, as Python's `Runner` does);
/// `pulsar` and `zeromq` yield the typed [`DeferredMessageQueue`] placeholder,
/// whose every operation is `Error::Unsupported`.
///
/// This is the infallible counterpart of [`create_message_queue`], used by
/// [`crate::Runner::with_config`] so the runner never reports an in-memory queue
/// for a configuration that names another transport.
#[must_use]
pub fn build_message_queue(config: &MessageQueueConfig) -> Arc<dyn MessageQueue> {
    match config.kind {
        MessageQueueKind::InMemory => InMemoryMessageQueue::new(config.clone()),
        kind => DeferredMessageQueue::new(kind),
    }
}

/// Builds the queue backend `config` selects.
///
/// [`MessageQueueKind::InMemory`] yields an unstarted
/// [`InMemoryMessageQueue`] (the caller starts it, as Python's `Runner` does);
/// `pulsar` and `zeromq` are deferred seams and fail with `Error::Unsupported`.
/// `config.url` / `config.max_workers` are carried for those backends and
/// ignored here.
///
/// # Errors
///
/// `Error::Unsupported` naming the requested backend when it is not compiled
/// into this build.
// `async` is part of the factory contract: a broker backend would connect here.
#[allow(clippy::unused_async)]
pub async fn create_message_queue(config: &MessageQueueConfig) -> Result<Arc<dyn MessageQueue>> {
    match config.kind {
        MessageQueueKind::InMemory => Ok(InMemoryMessageQueue::new(config.clone())),
        // A host asking for a broker gets the typed refusal, not the placeholder
        // `Runner::with_config` falls back to (it cannot fail).
        kind => Err(deferred_transport(kind)),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]

    use super::*;
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;

    #[tokio::test]
    async fn in_memory_backend_is_created_stopped() {
        let queue = create_message_queue(&MessageQueueConfig::default())
            .await
            .expect("in-memory queue");
        assert!(!queue.is_started(), "the runner starts its queue");
        assert!(queue.topics().is_empty());
    }

    #[tokio::test]
    async fn deferred_backends_are_unsupported() {
        for kind in [MessageQueueKind::Pulsar, MessageQueueKind::ZeroMq] {
            let config = MessageQueueConfig {
                kind,
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

    #[test]
    fn build_message_queue_never_misrepresents_the_configured_kind() {
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
        }
    }

    #[tokio::test]
    async fn the_deferred_queue_refuses_every_operation() {
        for kind in [MessageQueueKind::Pulsar, MessageQueueKind::ZeroMq] {
            let queue = DeferredMessageQueue::new(kind);
            assert_eq!(queue.kind(), kind);
            assert!(!queue.is_started());
            assert!(queue.topics().is_empty());
            assert!(!queue.has_subscriber("inbox"));
            assert!(
                format!("{queue:?}").contains(kind.as_str()),
                "debug names it"
            );

            // Every fallible operation is the same typed refusal.
            let failures = [
                queue.start().await,
                queue.stop().await,
                queue.subscribe("inbox").await,
                queue.add_handler("inbox", Arc::new(NoopHandler)).await,
                queue.publish("inbox", QueueMessage::new(Value::Null)).await,
                queue
                    .request("inbox", QueueMessage::new(Value::Null), None)
                    .await
                    .map(|_| ()),
                queue.unsubscribe("inbox").await,
            ];
            for result in failures {
                let error = result.expect_err("deferred transport");
                assert_eq!(error.kind(), ErrorKind::Unsupported);
                let text = error.to_string();
                assert!(text.contains(kind.as_str()), "names the backend: {text}");
                assert!(text.contains("in_memory"), "points at the port: {text}");
            }
        }
    }

    #[test]
    fn the_empty_report_is_the_default_and_exposes_helpers() {
        let empty = QueueStats::default();
        assert!(empty.is_empty());
        assert_eq!(empty.dropped(), 0);
        assert!(!empty.started);
        assert_eq!(empty.topics, 0);
        assert_eq!(empty.handlers, 0);

        let busy = QueueStats {
            started: true,
            topics: 2,
            handlers: 1,
            deliveries: 7,
            dropped_no_worker: 3,
            dropped_no_handler: 5,
            handler_errors: 2,
            rejected_not_started: 4,
        };
        assert!(
            !busy.is_empty(),
            "a started queue is never the empty report"
        );
        assert_eq!(busy.dropped(), 8);
        assert!(!QueueStats {
            rejected_not_started: 1,
            ..QueueStats::default()
        }
        .is_empty());
    }

    #[test]
    fn queue_stats_round_trip_with_a_tolerated_unknown_field() {
        let stats = QueueStats {
            started: true,
            topics: 3,
            handlers: 2,
            deliveries: 11,
            dropped_no_worker: 1,
            dropped_no_handler: 2,
            handler_errors: 3,
            rejected_not_started: 4,
        };
        let text = serde_json::to_string(&stats).expect("serialize");
        assert!(
            text.contains("\"dropped_no_worker\":1"),
            "snake_case: {text}"
        );
        let back: QueueStats = serde_json::from_str(&text).expect("deserialize");
        assert_eq!(back, stats);

        // An unknown (future) counter is tolerated and the missing ones default.
        let partial: QueueStats =
            serde_json::from_value(json!({"deliveries": 5, "future_counter": 9}))
                .expect("deserialize with an unknown field");
        assert_eq!(partial.deliveries, 5);
        assert!(!partial.started);
        assert_eq!(partial.topics, 0);
        assert_eq!(partial.dropped(), 0);
    }

    #[test]
    fn a_deferred_queue_reports_the_empty_report() {
        let queue = DeferredMessageQueue::new(MessageQueueKind::Pulsar);
        let stats = queue.stats();
        assert_eq!(stats, QueueStats::default());
        assert!(stats.is_empty());
    }

    /// A handler that is never reached by a deferred queue.
    struct NoopHandler;

    #[async_trait]
    impl MessageHandler for NoopHandler {
        async fn handle(&self, _message: QueueMessage) -> Result<Value> {
            Ok(Value::Null)
        }
    }
}
