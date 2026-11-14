//! The Redis team transport.
//!
//! Envelopes travel as entries of Redis Streams, so several processes can serve
//! one team, delivery survives a reconnect, and a consumer that died mid-entry
//! leaves its work claimable instead of lost. The implementation speaks RESP2
//! through [`openjiuwen_foundation`]'s existing client — no vendor SDK, no new
//! dependency, no blocking I/O on a task.
//!
//! # Key scheme
//!
//! Every key is `{key_prefix}:{team_id}:...`:
//!
//! | Key | Producer | Consumer |
//! | --- | --- | --- |
//! | `{prefix}:{team}:p2p:{recipient}` | [`RedisTransport::send`] (`XADD envelope .. reply ..`) | the recipient's worker (`XREADGROUP`) |
//! | `{prefix}:{team}:reply:{message_id}` | the recipient's worker (`XADD frame ..`) | the waiting [`RedisTransport::send`] (`XREAD`, then `DEL`) |
//! | `{prefix}:{team}:pubsub:{subscriber}` | [`RedisTransport::publish`] (`XADD envelope ..`) | the subscriber's worker (`XREADGROUP`) |
//!
//! A P2P request carries its own [`MessageEnvelope`] plus the reply stream key,
//! so `sender`, `session_id` and the correlation
//! [`MessageEnvelope::message_id`] live in the payload; the reply stream is keyed
//! by that id and holds one entry whose `frame` field is
//! `{"value": <json>}` on success and `{"error": "<text>"}` when the handler
//! failed, so the waiting `send` can reproduce the recipient's failure.
//!
//! # Consumer groups
//!
//! Each inbox kind has one consumer group, derived as
//! `{prefix}:{team}:p2p:cg` and `{prefix}:{team}:pubsub:cg` unless
//! [`RedisTransportConfig::consumer_group`] names one explicitly. Every
//! transport instance joins with its own [`RedisTransport::consumer_name`], and
//! the group is created idempotently (`XGROUP CREATE key <group> 0 MKSTREAM`,
//! tolerating `BUSYGROUP`), so processes may start in any order and the same
//! agent may be watched by several of them without double-delivering an entry.
//!
//! # Workers
//!
//! [`RedisTransport::watch_agent`] registers an agent's two inboxes (`p2p` and
//! `pubsub`) with one background worker each; the worker polls its stream every
//! [`RedisTransportConfig::poll_interval_ms`], dispatches through the
//! [`MessageHandler`], writes P2P replies, and acknowledges entries. A worker
//! also runs a claim pass (`XAUTOCLAIM`) on start and after every recovered read
//! error, so entries left pending by a crashed or reconnected consumer are
//! re-dispatched. See [`worker`] for the delivery loop and its at-least-once
//! guarantee. Workers are gated on a [`CancellationToken`]:
//! [`RedisTransport::stop`] cancels and joins them, and dropping a started
//! transport cancels them without joining (an async join is impossible in
//! `Drop`), so they exit at their next poll instead of leaking forever.
//!
//! # Delivery semantics
//!
//! `send` is request/response: it appends the request, then reads the reply
//! stream until a frame arrives, the caller's deadline elapses
//! ([`Error::Timeout`], worded like Python's
//! `Message from '<sender>' to '<recipient>' timed out after <t>s`), or the
//! transport stops. `publish` is fire-and-forget per subscriber: it appends one
//! copy per sorted subscriber and returns once every append succeeded, which —
//! unlike the in-process transport, whose `publish` awaits the fan-out — can be
//! before any handler ran. Subscriptions are registry entries of this transport
//! instance, so both sides of a cross-process pub/sub pair must register the
//! same `(agent, topic)` pairs.
//!
//! # Topology
//!
//! Only [`RedisTopology::Single`] is implemented: the workspace has no cluster-
//! or Sentinel-aware RESP client, so [`RedisTransport::try_new`] and
//! [`TeamTransport::start`] reject the other variants with
//! [`Error::Unsupported`](openjiuwen_common::error::ErrorKind::Unsupported)
//! instead of silently degrading to a single node.

mod config;
mod streams;
mod worker;

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::utils::id::uuid_v4;
use openjiuwen_foundation::store::external::redis::client::RedisAuth;
use openjiuwen_foundation::store::external::redis::codec::RespValue;
use openjiuwen_foundation::store::external::redis::RedisConnection;
use parking_lot::Mutex;
use serde_json::Value;
use tokio::sync::Mutex as AsyncMutex;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::envelope::MessageEnvelope;
use crate::message_bus::MessageHandler;
use crate::subscription::SubscriptionManager;

use self::config::topology_unsupported;
use self::streams::{decode_reply, del, group_name, stream_key, xadd, xread};
use self::worker::{Worker, WorkerMode};
use super::{TeamTransport, TransportKind, DEFAULT_KEY_PREFIX};

pub use config::{
    RedisTopology, RedisTransportConfig, DEFAULT_CLAIM_IDLE_MS, DEFAULT_POLL_INTERVAL_MS,
};

/// Team id used when the configured one is empty.
const DEFAULT_TEAM_ID: &str = "default";

/// Entries one read or claim pass takes.
const BATCH: usize = 16;

/// Lifecycle state of a [`RedisTransport`], guarded by a short-held mutex.
#[derive(Default)]
struct State {
    /// Whether [`RedisTransport::start`] ran and [`RedisTransport::stop`] did not.
    started: bool,
    /// Cancellation token the current workers observe.
    cancel: Option<CancellationToken>,
    /// Agents whose inboxes this transport owns.
    watched: BTreeSet<String>,
    /// One worker task per stream key.
    workers: BTreeMap<String, JoinHandle<()>>,
}

/// A team transport that moves envelopes through Redis Streams.
///
/// See the `transport` module documentation for the key scheme, the consumer
/// groups and the delivery semantics.
pub struct RedisTransport {
    /// Connection and polling settings.
    config: RedisTransportConfig,
    /// Effective team id embedded in every key.
    team_id: String,
    /// Prefix of every key.
    key_prefix: String,
    /// Consumer identity of this instance inside every consumer group.
    consumer: String,
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
    /// The infallible constructor accepts every topology; [`TeamTransport::start`]
    /// then rejects a non-single one, and [`RedisTransport::try_new`] rejects it
    /// up front.
    #[must_use]
    pub fn new(
        config: RedisTransportConfig,
        team_id: impl Into<String>,
        key_prefix: impl Into<String>,
        handler: Arc<dyn MessageHandler>,
    ) -> Arc<Self> {
        Self::build(config, team_id, key_prefix, handler)
    }

    /// Builds a stopped transport, rejecting a topology this crate cannot serve.
    ///
    /// # Errors
    ///
    /// [`Error::Unsupported`] for [`RedisTopology::Cluster`] and
    /// [`RedisTopology::Sentinel`], whose message names the missing client.
    pub fn try_new(
        config: RedisTransportConfig,
        team_id: impl Into<String>,
        key_prefix: impl Into<String>,
        handler: Arc<dyn MessageHandler>,
    ) -> Result<Arc<Self>> {
        if !config.topology.is_single() {
            return Err(topology_unsupported(config.topology));
        }
        Ok(Self::build(config, team_id, key_prefix, handler))
    }

    /// The shared constructor behind [`RedisTransport::new`].
    fn build(
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
            consumer: uuid_v4(),
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

    /// Returns this transport's consumer name inside every consumer group.
    ///
    /// The name is a fresh uuid v4 per instance, so two processes watching the
    /// same agent are two distinct consumers of one group.
    #[must_use]
    pub fn consumer_name(&self) -> &str {
        &self.consumer
    }

    /// Returns the P2P inbox key of `agent_id`.
    #[must_use]
    pub fn p2p_queue(&self, agent_id: &str) -> String {
        stream_key(&self.key_prefix, &self.team_id, "p2p", agent_id)
    }

    /// Returns the reply stream key of the request `message_id`.
    #[must_use]
    pub fn reply_queue(&self, message_id: &str) -> String {
        stream_key(&self.key_prefix, &self.team_id, "reply", message_id)
    }

    /// Returns the pub/sub inbox key of `subscriber`.
    #[must_use]
    pub fn pubsub_queue(&self, subscriber: &str) -> String {
        stream_key(&self.key_prefix, &self.team_id, "pubsub", subscriber)
    }

    /// Returns the consumer group of the P2P inboxes.
    #[must_use]
    pub fn p2p_consumer_group(&self) -> String {
        self.consumer_group("p2p")
    }

    /// Returns the consumer group of the pub/sub inboxes.
    #[must_use]
    pub fn pubsub_consumer_group(&self) -> String {
        self.consumer_group("pubsub")
    }

    /// Derives the consumer group of one key family.
    fn consumer_group(&self, kind: &str) -> String {
        group_name(
            &self.key_prefix,
            &self.team_id,
            kind,
            self.config.consumer_group.as_deref(),
        )
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
    /// [`start`](TeamTransport::start). Idempotent. Several transports may watch
    /// the same agent as long as they share a consumer group: the group hands
    /// each entry to exactly one consumer, so the processes split the inbox
    /// instead of duplicating it.
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
                self.consumer_group("p2p"),
                WorkerMode::P2p {
                    recipient: agent_id.to_owned(),
                },
            ),
            (
                self.pubsub_queue(agent_id),
                self.consumer_group("pubsub"),
                WorkerMode::Pubsub {
                    subscriber: agent_id.to_owned(),
                },
            ),
        ];
        for (key, group, mode) in inboxes {
            if self.state.lock().workers.contains_key(&key) {
                continue;
            }
            let worker = Worker {
                key: key.clone(),
                group,
                consumer: self.consumer.clone(),
                mode,
                link: Arc::clone(&self.link),
                handler: Arc::clone(&self.handler),
                reply_prefix: stream_key(&self.key_prefix, &self.team_id, "reply", ""),
                poll_interval: self.config.poll_interval(),
                claim_idle_ms: self.config.claim_idle_ms,
                batch: BATCH,
                cancel: cancel.clone(),
            };
            let handle = tokio::spawn(worker.run());
            self.state.lock().workers.insert(key, handle);
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
            .field("consumer", &self.consumer)
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
        // Belt and braces: `try_new` already rejects these, `new` does not.
        if !self.config.topology.is_single() {
            return Err(topology_unsupported(self.config.topology));
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
        for (key, handle) in workers {
            if let Err(error) = handle.await {
                tracing::warn!(key = %key, %error, "redis transport worker did not stop cleanly");
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
        xadd(
            &self.link,
            &request_key,
            &[("envelope", &payload), ("reply", reply_key.as_bytes())],
        )
        .await?;

        let deadline = timeout.map(|timeout| Instant::now() + timeout);
        let poll_interval = self.config.poll_interval();
        loop {
            let entries = xread(&self.link, &reply_key, 1).await?;
            if let Some(entry) = entries.into_iter().next() {
                let frame = entry.field("frame").map(<[u8]>::to_vec);
                // The reply is consumed once: delete the stream so a later
                // request cannot read a stale frame and so the key expires.
                if let Err(error) = del(&self.link, &reply_key).await {
                    tracing::warn!(key = %reply_key, %error, "redis transport could not delete a consumed reply stream");
                }
                if let Some(frame) = frame.as_deref().and_then(decode_reply) {
                    return frame.into_result();
                }
                tracing::warn!(
                    key = %reply_key,
                    "redis transport skipped a reply entry without a decodable frame"
                );
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
            xadd(
                &self.link,
                &self.pubsub_queue(&subscriber),
                &[("envelope", &payload)],
            )
            .await?;
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

/// Serializes an envelope for a stream entry.
fn encode_envelope(envelope: &MessageEnvelope) -> Result<Vec<u8>> {
    serde_json::to_vec(envelope).map_err(|error| {
        Error::transport(format!(
            "redis transport: cannot encode a message envelope: {error}"
        ))
    })
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

#[cfg(test)]
mod tests;
