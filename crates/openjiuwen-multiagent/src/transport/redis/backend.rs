//! The [`TeamTransport`](super::super::TeamTransport) implementation of the
//! Redis backend.
//!
//! The type itself, its constructors, its keys and its worker lifecycle live in
//! [`super`]; this module is the trait surface a [`TeamRuntime`] drives:
//! `start`/`stop`, `send`/`publish`, and the subscription calls that mirror this
//! instance's pairs into the shared registry.
//!
//! [`TeamRuntime`]: crate::TeamRuntime

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use super::config::topology_unsupported;
use super::registry::SubscriptionRegistry;
use super::streams::{decode_reply, del, xadd, xread};
use super::{encode_envelope, RedisTransport, TeamTransport, TransportKind};
use crate::envelope::MessageEnvelope;

impl RedisTransport {
    /// Starts the transport lazily when a caller skipped
    /// [`start`](TeamTransport::start).
    pub(super) async fn ensure_started(&self) -> Result<()> {
        if self.is_started() {
            Ok(())
        } else {
            self.start().await
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
        // `stop` drops this instance's registry entries; put them back when the
        // same transport starts again. Best effort: a failure is logged and
        // leaves the local registry intact.
        self.registry.republish().await;
        let refresher = SubscriptionRegistry::spawn_refresher(
            Arc::clone(&self.registry),
            self.config.registry_interval(),
            cancel.clone(),
        );
        self.state.lock().refresher = refresher;
        Ok(())
    }

    async fn stop(&self) -> Result<()> {
        let (cancel, workers, refresher) = {
            let mut state = self.state.lock();
            state.started = false;
            (
                state.cancel.take(),
                std::mem::take(&mut state.workers),
                state.refresher.take(),
            )
        };
        let Some(cancel) = cancel else {
            return Ok(());
        };
        cancel.cancel();
        if let Some(refresher) = refresher {
            if let Err(error) = refresher.await {
                tracing::warn!(%error, "redis transport registry refresher did not stop cleanly");
            }
        }
        for (key, handle) in workers {
            if let Err(error) = handle.await {
                tracing::warn!(key = %key, %error, "redis transport worker did not stop cleanly");
            }
        }
        // Best effort: a clean shutdown leaves no stale cross-process routes.
        // A write spawned by an `add_subscription` that raced with this stop may
        // still land afterwards, like a crashed process would: a stale entry
        // only causes `XADD`s into a stream no worker consumes.
        self.registry.clear().await;
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
        self.registry.add(agent_id, topic);
    }

    fn remove_subscription(&self, agent_id: &str, topic: &str) {
        self.registry.remove(agent_id, topic);
    }

    fn subscribers(&self, topic_id: &str) -> Vec<String> {
        self.subscriptions.subscribers(topic_id)
    }

    fn list_subscriptions(&self) -> BTreeMap<String, Vec<String>> {
        self.subscriptions.list()
    }

    fn agent_subscriptions(&self, agent_id: &str) -> Vec<String> {
        self.subscriptions.topics_of(agent_id)
    }

    fn subscription_count(&self) -> usize {
        self.subscriptions.subscription_count()
    }
}
