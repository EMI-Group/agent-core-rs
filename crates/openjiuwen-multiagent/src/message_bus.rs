//! The in-process team message bus.
//!
//! Python parity: `MessageBus` in
//! `openjiuwen/core/multi_agent/team_runtime/message_bus.py`. The Python
//! `MessageQueueInMemory` transport collapses into one bounded FIFO worker task
//! per transport topic, with a `tokio` oneshot carrying the P2P reply.
//!
//! The bus owns topics, worker lifetime, request/response correlation, and
//! delivery deadlines. It never resolves agents itself: every delivery goes
//! through the [`MessageHandler`] it holds, so this module stays independent of
//! the team runtime that implements that trait.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Duration;

use futures::future::join_all;
use openjiuwen_common::error::{Error, Result};
use parking_lot::Mutex;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;

use crate::envelope::MessageEnvelope;
use crate::subscription::SubscriptionManager;

/// Default capacity of one topic's FIFO queue.
const DEFAULT_MAX_QUEUE_SIZE: usize = 1000;
/// Default deadline for processing one message, in seconds (30 minutes).
const DEFAULT_PROCESS_TIMEOUT: f64 = 1800.0;
/// Team id used when the configured one is absent or empty.
const DEFAULT_TEAM_ID: &str = "default";
/// Suffix of the point-to-point transport topic.
const P2P_TOPIC_SUFFIX: &str = "__p2p__";
/// Suffix of the pub/sub transport topic.
const PUBSUB_TOPIC_SUFFIX: &str = "__pubsub__";

/// Configuration of the in-process [`MessageBus`].
///
/// Every field has a Python-compatible default (`1000` / `Some(1800.0)` /
/// `Some("default")`), unknown JSON fields are tolerated, and an empty object
/// deserializes to the defaults.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct MessageBusConfig {
    /// Capacity of each topic's FIFO queue; the queue is never smaller than one.
    pub max_queue_size: usize,
    /// Deadline for one delivery, in seconds; `None` disables the timeout.
    pub process_timeout: Option<f64>,
    /// Team id embedded in transport topic names.
    pub team_id: Option<String>,
}

impl Default for MessageBusConfig {
    fn default() -> Self {
        Self {
            max_queue_size: DEFAULT_MAX_QUEUE_SIZE,
            process_timeout: Some(DEFAULT_PROCESS_TIMEOUT),
            team_id: Some(DEFAULT_TEAM_ID.to_owned()),
        }
    }
}

impl MessageBusConfig {
    /// Creates the configuration with the Python defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the capacity of each topic's FIFO queue.
    #[must_use]
    pub fn with_max_queue_size(mut self, max_queue_size: usize) -> Self {
        self.max_queue_size = max_queue_size;
        self
    }

    /// Sets the per-delivery deadline; `None` disables the process timeout.
    #[must_use]
    pub fn with_process_timeout(mut self, process_timeout: Option<f64>) -> Self {
        self.process_timeout = process_timeout;
        self
    }

    /// Sets the team id embedded in transport topic names.
    #[must_use]
    pub fn with_team_id(mut self, team_id: impl Into<String>) -> Self {
        self.team_id = Some(team_id.into());
        self
    }

    /// Returns the team id used in topic names.
    ///
    /// An absent or empty [`team_id`](Self::team_id) falls back to `"default"`.
    #[must_use]
    pub fn effective_team_id(&self) -> &str {
        self.team_id
            .as_deref()
            .filter(|team_id| !team_id.is_empty())
            .unwrap_or(DEFAULT_TEAM_ID)
    }
}

/// Delivers bus messages on behalf of the team runtime.
///
/// Implementations resolve the recipient (or the topic's subscribers), build the
/// delivery session, and invoke the agent. P2P failures propagate to the caller
/// of [`MessageBus::send`]; pub/sub failures are isolated by the bus.
#[async_trait::async_trait]
pub trait MessageHandler: Send + Sync {
    /// Handles one point-to-point message and returns the reply payload.
    ///
    /// # Errors
    ///
    /// Returns whatever the recipient produced; the error is forwarded unchanged
    /// to the waiting [`MessageBus::send`] caller.
    async fn handle_p2p(&self, envelope: &MessageEnvelope) -> Result<Value>;

    /// Handles one pub/sub delivery to `subscriber`.
    ///
    /// # Errors
    ///
    /// A failure is logged and swallowed by the bus and never fails the
    /// publisher.
    async fn handle_pubsub(&self, envelope: &MessageEnvelope, subscriber: &str) -> Result<()>;
}

/// One unit of work queued on a topic worker.
enum BusItem {
    /// A point-to-point request whose reply travels back over `reply`.
    P2p {
        /// The request envelope.
        envelope: MessageEnvelope,
        /// Reply channel; dropped when the caller stopped waiting.
        reply: oneshot::Sender<Result<Value>>,
    },
    /// A pub/sub fan-out; `done` fires once every subscriber was delivered to.
    Pubsub {
        /// The published envelope.
        envelope: MessageEnvelope,
        /// Completion signal awaited by [`MessageBus::publish`].
        done: oneshot::Sender<()>,
    },
    /// Terminates the worker loop.
    Shutdown,
}

/// The sender and worker task of one transport topic.
struct TopicHandle {
    sender: mpsc::Sender<BusItem>,
    task: JoinHandle<()>,
}

/// Bus lifecycle state, guarded by a short-held synchronous mutex.
#[derive(Default)]
struct BusState {
    started: bool,
    topics: BTreeMap<String, TopicHandle>,
    subscribed: BTreeSet<String>,
}

/// An in-process message bus with one FIFO worker per transport topic.
///
/// The bus is a transport only: it names topics, owns their workers, correlates
/// requests with replies, and enforces deadlines. Recipients, sessions, and
/// agent invocation all live behind the [`MessageHandler`] it is built with.
///
/// Workers are created lazily, so a P2P `send` on a fresh topic works without an
/// explicit [`subscribe`](Self::subscribe) (in-memory transport parity). A `send`
/// timeout never cancels the handler, matching Python.
pub struct MessageBus {
    config: MessageBusConfig,
    team_id: String,
    handler: Arc<dyn MessageHandler>,
    subscriptions: Arc<SubscriptionManager>,
    state: Mutex<BusState>,
}

impl MessageBus {
    /// Builds a stopped bus around `handler`.
    #[must_use]
    pub fn new(config: MessageBusConfig, handler: Arc<dyn MessageHandler>) -> Self {
        let team_id = config.effective_team_id().to_owned();
        Self {
            config,
            team_id,
            handler,
            subscriptions: Arc::new(SubscriptionManager::new()),
            state: Mutex::new(BusState::default()),
        }
    }

    /// Marks the bus started.
    ///
    /// Idempotent: calling it again only flips the flag. Workers are created
    /// lazily by [`send`](Self::send), [`publish`](Self::publish), and
    /// [`subscribe`](Self::subscribe), and [`send`](Self::send) and
    /// [`publish`](Self::publish) restart the bus after a
    /// [`stop`](Self::stop), matching Python's lazy start.
    ///
    /// # Errors
    ///
    /// Reserved for future transports; the in-process bus never fails.
    // `async` is part of the transport contract: a distributed backend would await here.
    #[allow(clippy::unused_async, clippy::unused_async_trait_impl)]
    pub async fn start(&self) -> Result<()> {
        self.state.lock().started = true;
        Ok(())
    }

    /// Stops every topic worker and joins them.
    ///
    /// Idempotent. Each worker is asked to shut down, its sender is dropped (so
    /// it also ends once the queue drains), and its task is awaited; join errors
    /// are logged, never propagated. The subscription set is cleared, so a later
    /// [`subscribe`](Self::subscribe) succeeds.
    ///
    /// # Errors
    ///
    /// Reserved for future transports; the in-process bus never fails.
    pub async fn stop(&self) -> Result<()> {
        let topics = {
            let mut state = self.state.lock();
            state.started = false;
            state.subscribed.clear();
            std::mem::take(&mut state.topics)
        };
        for (topic, handle) in topics {
            let TopicHandle { sender, task } = handle;
            // A full queue still ends the worker once `sender` is dropped, so the
            // signal is best-effort and never blocks `stop`.
            let _ = sender.try_send(BusItem::Shutdown);
            drop(sender);
            if let Err(error) = task.await {
                tracing::warn!(%topic, %error, "message bus worker did not shut down cleanly");
            }
        }
        Ok(())
    }

    /// Returns `true` between [`start`](Self::start) and [`stop`](Self::stop).
    #[must_use]
    pub fn is_started(&self) -> bool {
        self.state.lock().started
    }

    /// Declares `topic` subscribed.
    ///
    /// Ensures the topic's worker exists. Subscribing to an already-subscribed
    /// topic is an error, matching Python's `Topic '...' is already subscribed.`
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `topic` is already subscribed.
    pub async fn subscribe(&self, topic: &str) -> Result<()> {
        self.ensure_started().await?;
        if self.state.lock().subscribed.contains(topic) {
            return Err(Error::validation(format!(
                "Topic '{topic}' is already subscribed."
            )));
        }
        self.ensure_topic(topic)?;
        self.state.lock().subscribed.insert(topic.to_owned());
        Ok(())
    }

    /// Drops `topic` from the subscribed set.
    ///
    /// Only the bookkeeping entry is removed; the worker stays alive so queued
    /// messages are still drained. An unknown topic is a no-op.
    ///
    /// # Errors
    ///
    /// Reserved for future transports; the in-process bus never fails.
    // `async` is part of the transport contract: a distributed backend would await here.
    #[allow(clippy::unused_async, clippy::unused_async_trait_impl)]
    pub async fn unsubscribe(&self, topic: &str) -> Result<()> {
        self.state.lock().subscribed.remove(topic);
        Ok(())
    }

    /// Registers `agent_id` as a subscriber of `topic_id` (an exact topic or a
    /// `*`/`?` pattern).
    pub fn add_subscription(&self, agent_id: &str, topic_id: &str) {
        self.subscriptions.subscribe(agent_id, topic_id);
    }

    /// Removes the `(agent_id, topic_id)` subscription.
    pub fn remove_subscription(&self, agent_id: &str, topic_id: &str) {
        self.subscriptions.unsubscribe(agent_id, topic_id);
    }

    /// Returns the sorted, de-duplicated ids of the agents subscribed to
    /// `topic_id`.
    #[must_use]
    pub fn subscribers(&self, topic_id: &str) -> Vec<String> {
        self.subscriptions.subscribers(topic_id)
    }

    /// Returns the topics `agent_id` is subscribed to (sorted).
    ///
    /// Used by the runtime to drop every subscription of an unregistered agent.
    #[must_use]
    pub fn agent_topics(&self, agent_id: &str) -> Vec<String> {
        self.subscriptions.topics_of(agent_id)
    }

    /// Removes every subscription of `agent_id` and returns how many were removed.
    ///
    /// Python parity: `MessageBus.remove_all_subscriptions`, which delegates to
    /// `SubscriptionManager.unsubscribe_all`; the runtime calls it when a member
    /// is unregistered.
    pub fn remove_all_subscriptions(&self, agent_id: &str) -> usize {
        self.subscriptions.unsubscribe_all(agent_id)
    }

    /// Returns the whole registry as `pattern -> sorted agent ids`.
    ///
    /// Python parity: `MessageBus.list_subscriptions()` without an agent filter
    /// (Python's unordered sets are sorted here for determinism).
    #[must_use]
    pub fn list_subscriptions(&self) -> BTreeMap<String, Vec<String>> {
        self.subscriptions.list()
    }

    /// Returns the sorted patterns `agent_id` is subscribed to.
    ///
    /// Python parity: the `topics` list of
    /// `MessageBus.list_subscriptions(agent_id)`.
    #[must_use]
    pub fn list_agent_subscriptions(&self, agent_id: &str) -> Vec<String> {
        self.subscriptions.topics_of(agent_id)
    }

    /// Returns the total number of `(agent, pattern)` pairs.
    ///
    /// Python parity: `MessageBus.get_subscription_count`, the sum of the
    /// per-pattern subscriber counts.
    #[must_use]
    pub fn get_subscription_count(&self) -> usize {
        self.subscriptions.subscription_count()
    }

    /// Sends a point-to-point envelope and waits for the reply.
    ///
    /// The envelope's [`session_id`](MessageEnvelope::session_id) selects the
    /// transport topic. The deadline is `timeout` when supplied, else
    /// [`process_timeout`](MessageBusConfig::process_timeout); with neither, the
    /// call waits unbounded. On an elapsed deadline the handler keeps running
    /// (Python parity).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Timeout`] when the deadline elapses,
    /// [`Error::Transport`] when the topic closes before replying, and otherwise
    /// the handler's error unchanged.
    pub async fn send(
        &self,
        envelope: MessageEnvelope,
        timeout: Option<Duration>,
    ) -> Result<Value> {
        self.ensure_started().await?;
        let topic = self.p2p_topic(envelope.session_id.as_deref());
        let sender_id = envelope.sender.clone().unwrap_or_default();
        let recipient_id = envelope.recipient.clone().unwrap_or_default();
        let (reply, reply_rx) = oneshot::channel();
        let sender = self.ensure_topic(&topic)?;
        sender
            .send(BusItem::P2p { envelope, reply })
            .await
            .map_err(|_| closed_reply_error(&topic))?;
        drop(sender);

        let deadline = timeout.or_else(|| self.config.process_timeout.map(secs_to_duration));
        match deadline {
            Some(duration) => match tokio::time::timeout(duration, reply_rx).await {
                Ok(result) => result.map_err(|_| closed_reply_error(&topic))?,
                // The worker is not cancelled: the handler runs to completion and
                // its reply is discarded, matching Python's `wait_for` behaviour.
                Err(_) => Err(Error::timeout(format!(
                    "message from '{sender_id}' to '{recipient_id}' timed out after {}s",
                    duration.as_secs_f64()
                ))),
            },
            None => reply_rx.await.map_err(|_| closed_reply_error(&topic))?,
        }
    }

    /// Publishes an envelope to its topic and waits for the fan-out to complete.
    ///
    /// The envelope's [`session_id`](MessageEnvelope::session_id) selects the
    /// transport topic and its [`topic_id`](MessageEnvelope::topic_id) selects
    /// the subscribers (an empty subscriber set is a no-op). The call returns
    /// after every subscriber was delivered to; a failing subscriber is logged
    /// and never fails the publisher.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Transport`] when the topic closes before the fan-out
    /// completes.
    pub async fn publish(&self, envelope: MessageEnvelope) -> Result<()> {
        self.ensure_started().await?;
        let topic = self.pubsub_topic(envelope.session_id.as_deref());
        let (done, done_rx) = oneshot::channel();
        let sender = self.ensure_topic(&topic)?;
        sender
            .send(BusItem::Pubsub { envelope, done })
            .await
            .map_err(|_| closed_fan_out_error(&topic))?;
        drop(sender);
        done_rx.await.map_err(|_| closed_fan_out_error(&topic))
    }

    /// Tears down both transport topics of `session_id`.
    ///
    /// Each topic is unsubscribed (errors are logged) and its worker removed, so
    /// a later [`subscribe`](Self::subscribe) of the same topic does not report a
    /// duplicate and a later [`send`](Self::send) recreates the worker.
    pub async fn cleanup_session(&self, session_id: &str) {
        let topics = [
            self.p2p_topic(Some(session_id)),
            self.pubsub_topic(Some(session_id)),
        ];
        for topic in topics {
            if let Err(error) = self.unsubscribe(&topic).await {
                tracing::warn!(%topic, %error, "failed to unsubscribe session topic");
            }
            let handle = self.state.lock().topics.remove(&topic);
            if let Some(TopicHandle { sender, task }) = handle {
                let _ = sender.try_send(BusItem::Shutdown);
                drop(sender);
                if let Err(error) = task.await {
                    tracing::warn!(
                        %topic,
                        %error,
                        "session topic worker did not shut down cleanly"
                    );
                }
            }
        }
    }

    /// Returns the bus configuration.
    #[must_use]
    pub fn config(&self) -> &MessageBusConfig {
        &self.config
    }

    /// Returns the team id embedded in topic names.
    #[must_use]
    pub fn team_id(&self) -> &str {
        &self.team_id
    }

    /// Returns the P2P transport topic of `session_id`.
    ///
    /// `"{team_id}__p2p__"` without a session, `"{team_id}_{session_id}__p2p__"`
    /// with one. An empty session id is treated as absent (Python parity).
    #[must_use]
    pub fn p2p_topic(&self, session_id: Option<&str>) -> String {
        topic_name(&self.team_id, session_id, P2P_TOPIC_SUFFIX)
    }

    /// Returns the pub/sub transport topic of `session_id`.
    ///
    /// `"{team_id}__pubsub__"` without a session,
    /// `"{team_id}_{session_id}__pubsub__"` with one. An empty session id is
    /// treated as absent (Python parity).
    #[must_use]
    pub fn pubsub_topic(&self, session_id: Option<&str>) -> String {
        topic_name(&self.team_id, session_id, PUBSUB_TOPIC_SUFFIX)
    }

    /// Restarts the bus when a prior [`stop`](Self::stop) left it stopped.
    async fn ensure_started(&self) -> Result<()> {
        let started = self.state.lock().started;
        if started {
            return Ok(());
        }
        self.start().await
    }

    /// Returns the topic's sender, creating its worker on first use.
    ///
    /// Creation is independent of the subscribed set: a plain `send` to a fresh
    /// topic works without an explicit `subscribe`. The `Result` keeps the seam a
    /// distributed transport needs.
    #[allow(clippy::unnecessary_wraps)]
    fn ensure_topic(&self, topic: &str) -> Result<mpsc::Sender<BusItem>> {
        let mut state = self.state.lock();
        if let Some(handle) = state.topics.get(topic) {
            return Ok(handle.sender.clone());
        }
        let (sender, receiver) = mpsc::channel(self.config.max_queue_size.max(1));
        let worker = Worker {
            topic: topic.to_owned(),
            handler: Arc::clone(&self.handler),
            subscriptions: Arc::clone(&self.subscriptions),
            process_timeout: self.config.process_timeout,
        };
        let task = tokio::spawn(worker.run(receiver));
        state.topics.insert(
            topic.to_owned(),
            TopicHandle {
                sender: sender.clone(),
                task,
            },
        );
        Ok(sender)
    }
}

// No `Drop` impl: the join handles live in `BusState`, so dropping the bus drops
// the topic senders and the worker loops end on their own.

/// Builds a transport topic name from a team id, an optional session, and a
/// suffix.
fn topic_name(team_id: &str, session_id: Option<&str>, suffix: &str) -> String {
    match session_id.filter(|session_id| !session_id.is_empty()) {
        Some(session_id) => format!("{team_id}_{session_id}{suffix}"),
        None => format!("{team_id}{suffix}"),
    }
}

/// Converts a timeout in seconds into a [`Duration`] without panicking.
///
/// Non-finite and non-positive values collapse to zero, so a malformed config
/// deadline fails fast instead of panicking.
fn secs_to_duration(secs: f64) -> Duration {
    if secs.is_finite() && secs > 0.0 {
        Duration::from_secs_f64(secs)
    } else {
        Duration::ZERO
    }
}

/// Transport error raised when a P2P topic closes before replying.
fn closed_reply_error(topic: &str) -> Error {
    Error::transport(format!(
        "message bus topic '{topic}' closed before replying"
    ))
}

/// Transport error raised when a pub/sub topic closes mid-fan-out.
fn closed_fan_out_error(topic: &str) -> Error {
    Error::transport(format!(
        "message bus topic '{topic}' closed before fan-out completed"
    ))
}

/// The task draining one transport topic in FIFO order.
struct Worker {
    topic: String,
    handler: Arc<dyn MessageHandler>,
    subscriptions: Arc<SubscriptionManager>,
    process_timeout: Option<f64>,
}

impl Worker {
    /// Drains `receiver` until a [`BusItem::Shutdown`] arrives or every sender
    /// is dropped.
    async fn run(self, mut receiver: mpsc::Receiver<BusItem>) {
        while let Some(item) = receiver.recv().await {
            match item {
                BusItem::P2p { envelope, reply } => {
                    let result = self.handler.handle_p2p(&envelope).await;
                    // A dropped receiver means the caller timed out; the handler
                    // still ran to completion.
                    let _ = reply.send(result);
                }
                BusItem::Pubsub { envelope, done } => {
                    self.fan_out(&envelope).await;
                    let _ = done.send(());
                }
                BusItem::Shutdown => break,
            }
        }
        tracing::debug!(topic = %self.topic, "message bus worker stopped");
    }

    /// Delivers `envelope` to every sorted subscriber concurrently and waits for
    /// all of them.
    async fn fan_out(&self, envelope: &MessageEnvelope) {
        let Some(topic_id) = envelope.topic_id.as_deref() else {
            tracing::debug!(
                topic = %self.topic,
                "pub/sub envelope without a topic id was dropped"
            );
            return;
        };
        let subscribers = self.subscriptions.subscribers(topic_id);
        if subscribers.is_empty() {
            tracing::debug!(topic = %self.topic, topic_id, "pub/sub message has no subscribers");
            return;
        }
        let deliveries = subscribers
            .iter()
            .map(|subscriber| self.deliver_to(envelope, subscriber));
        join_all(deliveries).await;
    }

    /// Delivers one pub/sub message, isolating timeouts and failures.
    async fn deliver_to(&self, envelope: &MessageEnvelope, subscriber: &str) {
        let result = match self.process_timeout {
            Some(secs) => {
                let deadline = secs_to_duration(secs);
                if let Ok(result) =
                    tokio::time::timeout(deadline, self.handler.handle_pubsub(envelope, subscriber))
                        .await
                {
                    result
                } else {
                    tracing::warn!(
                        topic = %self.topic,
                        subscriber,
                        timeout = secs,
                        "pub/sub delivery timed out"
                    );
                    return;
                }
            }
            None => self.handler.handle_pubsub(envelope, subscriber).await,
        };
        if let Err(error) = result {
            tracing::warn!(
                topic = %self.topic,
                subscriber,
                %error,
                "pub/sub delivery failed"
            );
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use openjiuwen_common::error::Error;

    /// A handler that only answers P2P requests and records nothing.
    struct EchoHandler;

    #[async_trait::async_trait]
    impl MessageHandler for EchoHandler {
        async fn handle_p2p(&self, envelope: &MessageEnvelope) -> Result<Value> {
            Ok(envelope.message.clone())
        }

        async fn handle_pubsub(
            &self,
            _envelope: &MessageEnvelope,
            _subscriber: &str,
        ) -> Result<()> {
            Ok(())
        }
    }

    fn bus_with(config: MessageBusConfig) -> MessageBus {
        MessageBus::new(config, Arc::new(EchoHandler))
    }

    #[test]
    fn defaults_match_python() {
        let config = MessageBusConfig::default();
        assert_eq!(config.max_queue_size, 1000);
        assert_eq!(config.process_timeout, Some(1800.0));
        assert_eq!(config.team_id.as_deref(), Some("default"));
        assert_eq!(MessageBusConfig::new(), config);
    }

    #[test]
    fn json_uses_snake_case_keys() {
        let value = serde_json::to_value(MessageBusConfig::new()).unwrap();
        assert_eq!(
            value,
            serde_json::json!({
                "max_queue_size": 1000,
                "process_timeout": 1800.0,
                "team_id": "default"
            })
        );
    }

    #[test]
    fn empty_object_deserializes_to_defaults() {
        let config: MessageBusConfig = serde_json::from_value(serde_json::json!({})).unwrap();
        assert_eq!(config, MessageBusConfig::default());
    }

    #[test]
    fn unknown_keys_are_tolerated() {
        let config: MessageBusConfig = serde_json::from_value(serde_json::json!({
            "max_queue_size": 7,
            "surprise": {"nested": true}
        }))
        .unwrap();
        assert_eq!(config.max_queue_size, 7);
        assert_eq!(config.process_timeout, Some(1800.0));
        assert_eq!(config.team_id.as_deref(), Some("default"));
    }

    #[test]
    fn builders_set_fields() {
        let config = MessageBusConfig::new()
            .with_max_queue_size(3)
            .with_process_timeout(None)
            .with_team_id("team-1");
        assert_eq!(config.max_queue_size, 3);
        assert_eq!(config.process_timeout, None);
        assert_eq!(config.team_id.as_deref(), Some("team-1"));
        assert_eq!(config.effective_team_id(), "team-1");
    }

    #[test]
    fn effective_team_id_falls_back_to_default() {
        let absent = MessageBusConfig {
            team_id: None,
            ..MessageBusConfig::new()
        };
        assert_eq!(absent.effective_team_id(), "default");

        let empty = MessageBusConfig {
            team_id: Some(String::new()),
            ..MessageBusConfig::new()
        };
        assert_eq!(empty.effective_team_id(), "default");

        let set = MessageBusConfig::new().with_team_id("team-9");
        assert_eq!(set.effective_team_id(), "team-9");
    }

    #[test]
    fn bus_uses_the_effective_team_id() {
        let bus = bus_with(MessageBusConfig::new().with_team_id(""));
        assert_eq!(bus.team_id(), "default");
        assert_eq!(bus.config().max_queue_size, 1000);
    }

    #[test]
    fn topic_names_follow_the_python_shape() {
        let bus = bus_with(MessageBusConfig::new().with_team_id("t"));
        assert_eq!(bus.p2p_topic(None), "t__p2p__");
        assert_eq!(bus.p2p_topic(Some("s1")), "t_s1__p2p__");
        assert_eq!(bus.p2p_topic(Some("")), "t__p2p__");
        assert_eq!(bus.pubsub_topic(None), "t__pubsub__");
        assert_eq!(bus.pubsub_topic(Some("s1")), "t_s1__pubsub__");
    }

    #[test]
    fn non_positive_deadlines_collapse_to_zero() {
        assert_eq!(secs_to_duration(0.0), Duration::ZERO);
        assert_eq!(secs_to_duration(-1.0), Duration::ZERO);
        assert_eq!(secs_to_duration(f64::NAN), Duration::ZERO);
        assert_eq!(secs_to_duration(f64::INFINITY), Duration::ZERO);
        assert_eq!(secs_to_duration(0.25), Duration::from_millis(250));
    }

    #[tokio::test]
    async fn send_delivers_the_payload_and_reply() {
        let bus = bus_with(MessageBusConfig::new().with_team_id("unit"));
        let envelope = MessageEnvelope::p2p(serde_json::json!({"n": 1}), "a", "b");
        let reply = bus
            .send(envelope, Some(Duration::from_millis(500)))
            .await
            .unwrap();
        assert_eq!(reply, serde_json::json!({"n": 1}));
        assert!(bus.is_started());
        bus.stop().await.unwrap();
    }

    #[tokio::test]
    async fn p2p_error_is_forwarded() {
        struct Failing;

        #[async_trait::async_trait]
        impl MessageHandler for Failing {
            async fn handle_p2p(&self, _envelope: &MessageEnvelope) -> Result<Value> {
                Err(Error::validation("boom"))
            }

            async fn handle_pubsub(
                &self,
                _envelope: &MessageEnvelope,
                _subscriber: &str,
            ) -> Result<()> {
                Ok(())
            }
        }

        let bus = MessageBus::new(
            MessageBusConfig::new().with_team_id("unit"),
            Arc::new(Failing),
        );
        let error = bus
            .send(
                MessageEnvelope::p2p(serde_json::json!({}), "a", "b"),
                Some(Duration::from_millis(500)),
            )
            .await
            .unwrap_err();
        assert_eq!(error.message(), "boom");
        bus.stop().await.unwrap();
    }
}
