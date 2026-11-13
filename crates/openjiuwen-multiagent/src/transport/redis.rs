//! The Redis team transport.
//!
//! Envelopes travel as JSON documents inside Redis lists, so one team can be
//! served by several processes. The implementation speaks RESP2 through
//! [`openjiuwen_foundation`]'s existing client — no vendor SDK, no new
//! dependency, no blocking I/O on a task.
//!
//! # Key scheme
//!
//! Every key is `{key_prefix}:{team_id}:...`:
//!
//! | Key | Producer | Consumer |
//! | --- | --- | --- |
//! | `{prefix}:{team}:p2p:{recipient}` | [`RedisTransport::send`] (`RPUSH`) | the recipient's worker (`RPOP`) |
//! | `{prefix}:{team}:reply:{message_id}` | the recipient's worker (`RPUSH`) | the waiting [`RedisTransport::send`] (`RPOP`) |
//! | `{prefix}:{team}:pubsub:{subscriber}` | [`RedisTransport::publish`] (`RPUSH`) | the subscriber's worker (`RPOP`) |
//!
//! A P2P request carries its own [`MessageEnvelope`], so `sender`, `session_id`
//! and the correlation [`MessageEnvelope::message_id`] live in the payload; the
//! reply list is keyed by that id. A reply frame is
//! `{"value": <json>}` on success and `{"error": "<text>"}` when the handler
//! failed, so the waiting `send` can reproduce the recipient's failure.
//!
//! # Workers
//!
//! `RPOP` removes a message, so a transport must only poll queues it owns.
//! [`RedisTransport::watch_agent`] therefore registers an agent's two inboxes
//! (`p2p` and `pubsub`) with one background worker each; the worker polls its
//! queue every [`RedisTransportConfig::poll_interval_ms`], dispatches through the
//! [`MessageHandler`], pushes P2P replies, and logs handler failures instead of
//! stopping. Workers are gated on a [`CancellationToken`]:
//! [`RedisTransport::stop`] cancels and joins them, and dropping a started
//! transport cancels them without joining (an async join is impossible in
//! `Drop`), so they exit at their next poll instead of leaking forever.
//!
//! # Delivery semantics
//!
//! `send` is request/response: it pushes the request, then polls the reply list
//! until a reply arrives, the caller's deadline elapses ([`Error::Timeout`],
//! worded like Python's
//! `Message from '<sender>' to '<recipient>' timed out after <t>s`), or the
//! transport stops. `publish` is fire-and-forget per subscriber: it pushes one
//! copy per sorted subscriber and returns once every push succeeded, which —
//! unlike the in-process transport, whose `publish` awaits the fan-out — can be
//! before any handler ran. Subscriptions are registry entries of this transport
//! instance, so both sides of a cross-process pub/sub pair must register the
//! same `(agent, topic)` pairs.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::security::SecretString;
use openjiuwen_foundation::store::external::redis::client::RedisAuth;
use openjiuwen_foundation::store::external::redis::codec::RespValue;
use openjiuwen_foundation::store::external::redis::{RedisConnection, DEFAULT_ADDR};
use parking_lot::Mutex;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::Mutex as AsyncMutex;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::envelope::MessageEnvelope;
use crate::message_bus::MessageHandler;
use crate::subscription::SubscriptionManager;

use super::{TeamTransport, TransportKind, DEFAULT_KEY_PREFIX};

/// Default idle delay between two polls of one queue, in milliseconds.
pub const DEFAULT_POLL_INTERVAL_MS: u64 = 10;

/// Team id used when the configured one is empty.
const DEFAULT_TEAM_ID: &str = "default";

/// Connection settings of the Redis team transport.
///
/// Every field is defaulted and unknown JSON fields are tolerated, so an empty
/// object points at the local Redis on [`DEFAULT_ADDR`]. The password is held in
/// a [`SecretString`], which keeps the plaintext for serialization (a config
/// round-trip preserves it) but redacts it in `Debug` and `Display`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct RedisTransportConfig {
    /// `host:port` of the Redis server.
    pub addr: String,
    /// Username sent by `AUTH`, when the deployment requires one.
    pub username: Option<String>,
    /// Password sent by `AUTH`; redacted in `Debug` output.
    #[schemars(with = "Option<String>")]
    pub password: Option<SecretString>,
    /// Database selected after connecting.
    pub db: Option<u32>,
    /// Idle delay between two `RPOP` polls of one queue, in milliseconds.
    pub poll_interval_ms: u64,
}

impl Default for RedisTransportConfig {
    fn default() -> Self {
        Self {
            addr: DEFAULT_ADDR.to_owned(),
            username: None,
            password: None,
            db: None,
            poll_interval_ms: DEFAULT_POLL_INTERVAL_MS,
        }
    }
}

impl RedisTransportConfig {
    /// Creates the configuration of a local, unauthenticated Redis.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the `host:port` of the server.
    #[must_use]
    pub fn with_addr(mut self, addr: impl Into<String>) -> Self {
        self.addr = addr.into();
        self
    }

    /// Sets the `AUTH` username.
    #[must_use]
    pub fn with_username(mut self, username: impl Into<String>) -> Self {
        self.username = Some(username.into());
        self
    }

    /// Sets the `AUTH` password.
    #[must_use]
    pub fn with_password(mut self, password: impl Into<String>) -> Self {
        self.password = Some(SecretString::new(password));
        self
    }

    /// Sets the database selected after connecting.
    #[must_use]
    pub const fn with_db(mut self, db: u32) -> Self {
        self.db = Some(db);
        self
    }

    /// Sets the idle delay between two polls of one queue.
    #[must_use]
    pub const fn with_poll_interval_ms(mut self, poll_interval_ms: u64) -> Self {
        self.poll_interval_ms = poll_interval_ms;
        self
    }

    /// Returns the poll delay, never smaller than one millisecond.
    #[must_use]
    pub fn poll_interval(&self) -> Duration {
        Duration::from_millis(self.poll_interval_ms.max(1))
    }
}

/// Lifecycle state of a [`RedisTransport`], guarded by a short-held mutex.
#[derive(Default)]
struct State {
    /// Whether [`RedisTransport::start`] ran and [`RedisTransport::stop`] did not.
    started: bool,
    /// Cancellation token the current workers observe.
    cancel: Option<CancellationToken>,
    /// Agents whose inboxes this transport owns.
    watched: BTreeSet<String>,
    /// One worker task per queue key.
    workers: BTreeMap<String, JoinHandle<()>>,
}

/// A team transport that moves envelopes through Redis lists.
///
/// See the `transport` module documentation for the key scheme and the delivery
/// semantics.
pub struct RedisTransport {
    /// Connection and polling settings.
    config: RedisTransportConfig,
    /// Effective team id embedded in every key.
    team_id: String,
    /// Prefix of every key.
    key_prefix: String,
    /// Dispatcher of every delivery.
    handler: Arc<dyn MessageHandler>,
    /// Logical subscriber registry used for fan-out.
    subscriptions: Arc<SubscriptionManager>,
    /// The lazily connected RESP2 link shared by every task.
    link: Arc<AsyncMutex<RedisLink>>,
    /// Lifecycle state.
    state: Mutex<State>,
}

impl RedisTransport {
    /// Builds a stopped transport for `team_id`.
    ///
    /// An empty `team_id` falls back to `"default"` and an empty `key_prefix` to
    /// [`DEFAULT_KEY_PREFIX`], so key names never start with a bare separator.
    #[must_use]
    pub fn new(
        config: RedisTransportConfig,
        team_id: impl Into<String>,
        key_prefix: impl Into<String>,
        handler: Arc<dyn MessageHandler>,
    ) -> Arc<Self> {
        let team_id = team_id.into();
        let key_prefix = key_prefix.into();
        let link = RedisLink::new(&config);
        Arc::new(Self {
            config,
            team_id: if team_id.is_empty() {
                DEFAULT_TEAM_ID.to_owned()
            } else {
                team_id
            },
            key_prefix: if key_prefix.is_empty() {
                DEFAULT_KEY_PREFIX.to_owned()
            } else {
                key_prefix
            },
            handler,
            subscriptions: Arc::new(SubscriptionManager::new()),
            link: Arc::new(AsyncMutex::new(link)),
            state: Mutex::new(State::default()),
        })
    }

    /// Returns the connection and polling settings.
    #[must_use]
    pub fn config(&self) -> &RedisTransportConfig {
        &self.config
    }

    /// Returns the prefix of every key this transport creates.
    #[must_use]
    pub fn key_prefix(&self) -> &str {
        &self.key_prefix
    }

    /// Returns the P2P inbox key of `agent_id`.
    #[must_use]
    pub fn p2p_queue(&self, agent_id: &str) -> String {
        queue_key(&self.key_prefix, &self.team_id, "p2p", agent_id)
    }

    /// Returns the reply queue key of the request `message_id`.
    #[must_use]
    pub fn reply_queue(&self, message_id: &str) -> String {
        queue_key(&self.key_prefix, &self.team_id, "reply", message_id)
    }

    /// Returns the pub/sub inbox key of `subscriber`.
    #[must_use]
    pub fn pubsub_queue(&self, subscriber: &str) -> String {
        queue_key(&self.key_prefix, &self.team_id, "pubsub", subscriber)
    }

    /// Returns `true` between [`start`](TeamTransport::start) and
    /// [`stop`](TeamTransport::stop).
    #[must_use]
    pub fn is_started(&self) -> bool {
        self.state.lock().started
    }

    /// Registers `agent_id` as an agent this transport receives for.
    ///
    /// The agent's P2P and pub/sub inboxes get one worker each; a started
    /// transport spawns them immediately, a stopped one at the next
    /// [`start`](TeamTransport::start). Idempotent, and a transport must never
    /// watch an agent another process serves (both would compete for the same
    /// `RPOP`).
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] for an empty `agent_id`, and [`Error::Config`] when
    /// the transport is already started but the call happens outside a tokio
    /// runtime (workers are tokio tasks).
    pub fn watch_agent(&self, agent_id: &str) -> Result<()> {
        if agent_id.is_empty() {
            return Err(Error::validation("agent_id is required"));
        }
        let (started, cancel) = {
            let mut state = self.state.lock();
            state.watched.insert(agent_id.to_owned());
            (state.started, state.cancel.clone())
        };
        let Some(cancel) = cancel.filter(|_| started) else {
            return Ok(());
        };
        if tokio::runtime::Handle::try_current().is_err() {
            return Err(Error::config(
                "RedisTransport::watch_agent needs a tokio runtime once the transport is started",
            ));
        }
        self.spawn_workers(agent_id, &cancel);
        Ok(())
    }

    /// Spawns the two inbox workers of `agent_id` that are missing.
    fn spawn_workers(&self, agent_id: &str, cancel: &CancellationToken) {
        let inboxes = [
            (
                self.p2p_queue(agent_id),
                WorkerMode::P2p {
                    recipient: agent_id.to_owned(),
                },
            ),
            (
                self.pubsub_queue(agent_id),
                WorkerMode::Pubsub {
                    subscriber: agent_id.to_owned(),
                },
            ),
        ];
        for (queue, mode) in inboxes {
            if self.state.lock().workers.contains_key(&queue) {
                continue;
            }
            let worker = Worker {
                queue: queue.clone(),
                mode,
                link: Arc::clone(&self.link),
                handler: Arc::clone(&self.handler),
                reply_prefix: queue_key(&self.key_prefix, &self.team_id, "reply", ""),
                poll_interval: self.config.poll_interval(),
                cancel: cancel.clone(),
            };
            let handle = tokio::spawn(worker.run());
            self.state.lock().workers.insert(queue, handle);
        }
    }

    /// Starts the transport lazily when a caller skipped
    /// [`start`](TeamTransport::start).
    async fn ensure_started(&self) -> Result<()> {
        if self.is_started() {
            Ok(())
        } else {
            self.start().await
        }
    }
}

impl fmt::Debug for RedisTransport {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Read the state once: the guard of a `field(..)` temporary would live
        // until the end of the statement and deadlock the next lock of the same
        // non-reentrant mutex.
        let (watched, started) = {
            let state = self.state.lock();
            (state.watched.len(), state.started)
        };
        formatter
            .debug_struct("RedisTransport")
            .field("team_id", &self.team_id)
            .field("key_prefix", &self.key_prefix)
            .field("config", &self.config)
            .field("watched", &watched)
            .field("started", &started)
            .finish_non_exhaustive()
    }
}

impl Drop for RedisTransport {
    fn drop(&mut self) {
        // Workers hold no `Arc<Self>`, so dropping the transport really is the
        // last reference and the token cancels their next poll.
        let state = self.state.get_mut();
        state.started = false;
        if let Some(cancel) = state.cancel.take() {
            cancel.cancel();
        }
    }
}

#[async_trait]
impl TeamTransport for RedisTransport {
    fn kind(&self) -> TransportKind {
        TransportKind::Redis
    }

    fn team_id(&self) -> &str {
        &self.team_id
    }

    async fn start(&self) -> Result<()> {
        if self.is_started() {
            return Ok(());
        }
        // A handshake failure (bad address, rejected `AUTH`) surfaces here
        // instead of on the first delivery.
        {
            let mut link = self.link.lock().await;
            link.ping().await?;
        }
        let cancel = CancellationToken::new();
        let watched = {
            let mut state = self.state.lock();
            if state.started {
                return Ok(());
            }
            state.started = true;
            state.cancel = Some(cancel.clone());
            state.watched.clone()
        };
        for agent in watched {
            self.spawn_workers(&agent, &cancel);
        }
        Ok(())
    }

    async fn stop(&self) -> Result<()> {
        let (cancel, workers) = {
            let mut state = self.state.lock();
            state.started = false;
            (state.cancel.take(), std::mem::take(&mut state.workers))
        };
        let Some(cancel) = cancel else {
            return Ok(());
        };
        cancel.cancel();
        for (queue, handle) in workers {
            if let Err(error) = handle.await {
                tracing::warn!(queue = %queue, %error, "redis transport worker did not stop cleanly");
            }
        }
        Ok(())
    }

    async fn send(&self, envelope: MessageEnvelope, timeout: Option<Duration>) -> Result<Value> {
        let recipient = envelope
            .recipient
            .clone()
            .filter(|recipient| !recipient.is_empty())
            .ok_or_else(|| Error::validation("recipient is required"))?;
        self.ensure_started().await?;
        let sender = envelope.sender.clone().unwrap_or_default();
        let request_key = self.p2p_queue(&recipient);
        let reply_key = self.reply_queue(&envelope.message_id);
        let payload = encode_envelope(&envelope)?;
        rpush(&self.link, &request_key, &payload).await?;

        let deadline = timeout.map(|timeout| Instant::now() + timeout);
        let poll_interval = self.config.poll_interval();
        loop {
            if let Some(frame) = rpop(&self.link, &reply_key).await? {
                if let Some(frame) = decode_reply(&frame) {
                    return frame.into_result();
                }
            }
            let wait = match deadline {
                Some(deadline) => {
                    let now = Instant::now();
                    if now >= deadline {
                        return Err(Error::timeout(format!(
                            "Message from '{sender}' to '{recipient}' timed out after {}s",
                            timeout.map_or(0.0, |timeout| timeout.as_secs_f64())
                        )));
                    }
                    poll_interval.min(deadline - now)
                }
                None => poll_interval,
            };
            tokio::time::sleep(wait).await;
            if !self.is_started() {
                return Err(Error::transport(format!(
                    "redis transport for team '{}' stopped before '{recipient}' replied",
                    self.team_id
                )));
            }
        }
    }

    async fn publish(&self, envelope: MessageEnvelope) -> Result<()> {
        let topic_id = envelope
            .topic_id
            .clone()
            .filter(|topic_id| !topic_id.is_empty())
            .ok_or_else(|| Error::validation("topic_id is required"))?;
        self.ensure_started().await?;
        let subscribers = self.subscriptions.subscribers(&topic_id);
        if subscribers.is_empty() {
            tracing::debug!(topic_id, team = %self.team_id, "redis team publish has no subscribers");
            return Ok(());
        }
        let payload = encode_envelope(&envelope)?;
        for subscriber in subscribers {
            rpush(&self.link, &self.pubsub_queue(&subscriber), &payload).await?;
        }
        Ok(())
    }

    fn add_subscription(&self, agent_id: &str, topic: &str) {
        self.subscriptions.subscribe(agent_id, topic);
    }

    fn remove_subscription(&self, agent_id: &str, topic: &str) {
        self.subscriptions.unsubscribe(agent_id, topic);
    }

    fn subscribers(&self, topic_id: &str) -> Vec<String> {
        self.subscriptions.subscribers(topic_id)
    }
}

/// Builds one key of the `{prefix}:{team}:{kind}:{id}` scheme.
fn queue_key(key_prefix: &str, team_id: &str, kind: &str, id: &str) -> String {
    format!("{key_prefix}:{team_id}:{kind}:{id}")
}

/// Serializes an envelope for a Redis list.
fn encode_envelope(envelope: &MessageEnvelope) -> Result<Vec<u8>> {
    serde_json::to_vec(envelope).map_err(|error| {
        Error::transport(format!(
            "redis transport: cannot encode a message envelope: {error}"
        ))
    })
}

/// Decodes a reply frame, logging and skipping a malformed one.
fn decode_reply(payload: &[u8]) -> Option<ReplyFrame> {
    match serde_json::from_slice(payload) {
        Ok(frame) => Some(frame),
        Err(error) => {
            tracing::warn!(%error, "redis transport skipped a malformed reply frame");
            None
        }
    }
}

/// The reply one recipient's worker pushes onto `{prefix}:{team}:reply:{id}`.
#[derive(Debug, Serialize, Deserialize)]
struct ReplyFrame {
    /// The handler's reply value, when it succeeded.
    #[serde(default)]
    value: Option<Value>,
    /// The handler's error text, when it failed.
    #[serde(default)]
    error: Option<String>,
}

impl ReplyFrame {
    /// Builds the frame of a successful delivery.
    fn from_value(value: Value) -> Self {
        Self {
            value: Some(value),
            error: None,
        }
    }

    /// Builds the frame of a failed delivery.
    fn from_error(message: impl Into<String>) -> Self {
        Self {
            value: None,
            error: Some(message.into()),
        }
    }

    /// Turns the frame back into the waiting `send`'s result.
    fn into_result(self) -> Result<Value> {
        match self.error {
            Some(message) => Err(Error::transport(format!(
                "redis transport: remote handler failed: {message}"
            ))),
            None => Ok(self.value.unwrap_or(Value::Null)),
        }
    }
}

/// A lazily connected RESP2 link, shared by every task of one transport.
struct RedisLink {
    /// `host:port` of the server.
    addr: String,
    /// Credentials replayed by every handshake.
    auth: RedisAuth,
    /// Database selected by every handshake.
    db: Option<u32>,
    /// The live connection, or `None` before the first command.
    connection: Option<RedisConnection>,
}

impl RedisLink {
    /// Builds a disconnected link from `config`.
    fn new(config: &RedisTransportConfig) -> Self {
        let auth = config
            .password
            .as_ref()
            .map(|password| (config.username.clone(), password.expose_secret().to_owned()));
        Self {
            addr: config.addr.clone(),
            auth,
            db: config.db,
            connection: None,
        }
    }

    /// Sends one command, reconnecting once after a connection-level failure.
    async fn command(&mut self, args: &[&[u8]]) -> Result<RespValue> {
        if self.connection.is_none() {
            let connection =
                RedisConnection::connect(&self.addr, self.auth.clone(), self.db).await?;
            self.connection = Some(connection);
        }
        let Some(connection) = self.connection.as_mut() else {
            return Err(Error::transport("redis transport: no live connection"));
        };
        let result = connection.command(args).await;
        if result.is_err() {
            // The client already retried what is safe to retry; dropping the
            // socket makes the next call reconnect (and replay `AUTH`/`SELECT`).
            self.connection = None;
        }
        result
    }

    /// Verifies the connection with `PING`.
    async fn ping(&mut self) -> Result<()> {
        match self.command(&[b"PING"]).await? {
            RespValue::SimpleString(text) if text.eq_ignore_ascii_case("PONG") => Ok(()),
            RespValue::BulkString(Some(payload)) if payload.eq_ignore_ascii_case(b"PONG") => Ok(()),
            other => Err(Error::transport(format!(
                "redis transport: unexpected PING reply: {other:?}"
            ))),
        }
    }
}

/// Pushes one payload onto `queue`.
async fn rpush(link: &AsyncMutex<RedisLink>, queue: &str, payload: &[u8]) -> Result<()> {
    let mut link = link.lock().await;
    match link.command(&[b"RPUSH", queue.as_bytes(), payload]).await? {
        RespValue::Integer(_) => Ok(()),
        other => Err(Error::transport(format!(
            "redis transport: RPUSH '{queue}' returned {other:?}"
        ))),
    }
}

/// Pops one payload off `queue`; `None` means the queue is empty.
async fn rpop(link: &AsyncMutex<RedisLink>, queue: &str) -> Result<Option<Vec<u8>>> {
    let mut link = link.lock().await;
    match link.command(&[b"RPOP", queue.as_bytes()]).await? {
        RespValue::BulkString(payload) => Ok(payload.map(|bytes| bytes.to_vec())),
        RespValue::Null => Ok(None),
        other => Err(Error::transport(format!(
            "redis transport: RPOP '{queue}' returned {other:?}"
        ))),
    }
}

/// What one inbox worker does with the messages it pops.
enum WorkerMode {
    /// Reply to `recipient` through the reply list.
    P2p {
        /// The agent the inbox belongs to.
        recipient: String,
    },
    /// Deliver to `subscriber` through [`MessageHandler::handle_pubsub`].
    Pubsub {
        /// The agent the inbox belongs to.
        subscriber: String,
    },
}

/// The task draining one Redis queue in FIFO order.
struct Worker {
    /// The queue this worker owns.
    queue: String,
    /// What a delivered envelope means.
    mode: WorkerMode,
    /// The shared link.
    link: Arc<AsyncMutex<RedisLink>>,
    /// The dispatcher.
    handler: Arc<dyn MessageHandler>,
    /// `{prefix}:{team}:reply:` prefix of this team's reply lists.
    reply_prefix: String,
    /// Idle delay between two polls.
    poll_interval: Duration,
    /// Cancellation observed by every wait.
    cancel: CancellationToken,
}

impl Worker {
    /// Drains `self.queue` until the transport stops.
    async fn run(self) {
        loop {
            if self.cancel.is_cancelled() {
                break;
            }
            match rpop(&self.link, &self.queue).await {
                Ok(Some(payload)) => self.deliver(&payload).await,
                Ok(None) => {
                    if self.sleep_or_stop().await {
                        break;
                    }
                }
                Err(error) => {
                    tracing::warn!(queue = %self.queue, %error, "redis transport poll failed");
                    if self.sleep_or_stop().await {
                        break;
                    }
                }
            }
        }
        tracing::debug!(queue = %self.queue, "redis transport worker stopped");
    }

    /// Sleeps one poll interval; returns `true` when the transport stopped first.
    async fn sleep_or_stop(&self) -> bool {
        tokio::select! {
            () = self.cancel.cancelled() => true,
            () = tokio::time::sleep(self.poll_interval) => false,
        }
    }

    /// Dispatches one popped payload.
    async fn deliver(&self, payload: &[u8]) {
        let envelope: MessageEnvelope = match serde_json::from_slice(payload) {
            Ok(envelope) => envelope,
            Err(error) => {
                tracing::warn!(
                    queue = %self.queue,
                    %error,
                    "redis transport skipped a malformed envelope"
                );
                return;
            }
        };
        match &self.mode {
            WorkerMode::P2p { recipient } => self.deliver_p2p(&envelope, recipient).await,
            WorkerMode::Pubsub { subscriber } => {
                if let Err(error) = self.handler.handle_pubsub(&envelope, subscriber).await {
                    tracing::warn!(
                        queue = %self.queue,
                        subscriber,
                        %error,
                        "redis transport pub/sub handler failed"
                    );
                }
            }
        }
    }

    /// Runs one P2P delivery and pushes its reply frame.
    async fn deliver_p2p(&self, envelope: &MessageEnvelope, recipient: &str) {
        let frame = match self.handler.handle_p2p(envelope).await {
            Ok(value) => ReplyFrame::from_value(value),
            Err(error) => {
                tracing::warn!(
                    queue = %self.queue,
                    recipient,
                    %error,
                    "redis transport p2p handler failed"
                );
                ReplyFrame::from_error(error.to_string())
            }
        };
        let reply_key = format!("{}{}", self.reply_prefix, envelope.message_id);
        let encoded = match serde_json::to_vec(&frame) {
            Ok(encoded) => encoded,
            Err(error) => {
                tracing::warn!(%error, "redis transport cannot encode a reply frame");
                return;
            }
        };
        if let Err(error) = rpush(&self.link, &reply_key, &encoded).await {
            tracing::warn!(
                queue = %reply_key,
                %error,
                "redis transport failed to publish a reply"
            );
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults() {
        let config = RedisTransportConfig::new();
        assert_eq!(config.addr, DEFAULT_ADDR);
        assert_eq!(config.poll_interval_ms, DEFAULT_POLL_INTERVAL_MS);
        assert!(config.username.is_none());
        assert!(config.password.is_none());
        assert_eq!(config.db, None);
        assert_eq!(config.poll_interval(), Duration::from_millis(10));
    }

    #[test]
    fn builders_set_fields() {
        let config = RedisTransportConfig::new()
            .with_addr("redis:6380")
            .with_username("agent")
            .with_password("s3cret")
            .with_db(2)
            .with_poll_interval_ms(0);
        assert_eq!(config.addr, "redis:6380");
        assert_eq!(config.username.as_deref(), Some("agent"));
        assert_eq!(config.password.clone().unwrap().expose_secret(), "s3cret");
        assert_eq!(config.db, Some(2));
        // A zero interval would spin; the accessor clamps it to one millisecond.
        assert_eq!(config.poll_interval(), Duration::from_millis(1));
    }

    #[test]
    fn debug_redacts_the_password() {
        let config = RedisTransportConfig::new().with_password("s3cret");
        let rendered = format!("{config:?}");
        assert!(!rendered.contains("s3cret"), "{rendered}");
        assert!(rendered.contains("<REDACTED>"), "{rendered}");
    }

    #[test]
    fn json_is_snake_case_and_tolerates_unknown_fields() {
        let config: RedisTransportConfig = serde_json::from_value(serde_json::json!({
            "addr": "127.0.0.1:7000",
            "poll_interval_ms": 25,
            "future": true,
        }))
        .unwrap();
        assert_eq!(config.addr, "127.0.0.1:7000");
        assert_eq!(config.poll_interval_ms, 25);
        assert_eq!(config.db, None);

        let encoded = serde_json::to_value(RedisTransportConfig::new()).unwrap();
        assert_eq!(encoded["addr"], DEFAULT_ADDR);
        assert_eq!(encoded["poll_interval_ms"], 10);
    }

    #[test]
    fn key_scheme_is_stable() {
        assert_eq!(
            queue_key("openjiuwen:team", "t1", "p2p", "agent"),
            "openjiuwen:team:t1:p2p:agent"
        );
        assert_eq!(
            queue_key("openjiuwen:team", "t1", "reply", "m1"),
            "openjiuwen:team:t1:reply:m1"
        );
        assert_eq!(
            queue_key("openjiuwen:team", "t1", "pubsub", "sub"),
            "openjiuwen:team:t1:pubsub:sub"
        );
    }

    #[test]
    fn reply_frames_round_trip() {
        let ok = ReplyFrame::from_value(serde_json::json!({"a": 1}));
        let encoded = serde_json::to_vec(&ok).unwrap();
        let decoded = decode_reply(&encoded).unwrap();
        assert_eq!(decoded.into_result().unwrap(), serde_json::json!({"a": 1}));

        let failed = ReplyFrame::from_error("boom");
        let decoded = decode_reply(&serde_json::to_vec(&failed).unwrap()).unwrap();
        let error = decoded.into_result().unwrap_err();
        assert_eq!(
            error.message(),
            "redis transport: remote handler failed: boom"
        );

        assert!(decode_reply(b"not json").is_none());
    }

    #[test]
    fn encode_envelope_keeps_routing_metadata() {
        let envelope =
            MessageEnvelope::p2p(serde_json::json!({"n": 1}), "a", "b").with_session("s1");
        let encoded = encode_envelope(&envelope).unwrap();
        let decoded: MessageEnvelope = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded, envelope);
    }
}
