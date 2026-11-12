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

use openjiuwen_common::error::Result;
use serde_json::Value;

use crate::config::{MessageQueueConfig, MessageQueueKind};
use crate::error::unsupported;
use crate::mq::message::{MessageHandler, QueueMessage};
use crate::mq::InMemoryMessageQueue;

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
        kind => {
            let feature = format!("{} message queue", kind.as_str());
            Err(unsupported(&feature, "use `in_memory`"))
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]

    use super::*;
    use openjiuwen_common::error::ErrorKind;

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
}
