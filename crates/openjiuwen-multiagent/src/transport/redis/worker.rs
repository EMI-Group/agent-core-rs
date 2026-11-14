//! The inbox worker of the Redis Streams transport.
//!
//! One [`Worker`] owns one inbox stream of one agent, either
//! `{prefix}:{team}:p2p:{agent}` or `{prefix}:{team}:pubsub:{agent}`, and is
//! identified in its consumer group by the transport's [`consumer_name`].
//!
//! # Delivery loop
//!
//! 1. Create the group idempotently (`XGROUP CREATE key <group> 0 MKSTREAM`).
//! 2. Claim entries another consumer left pending (`XAUTOCLAIM key <group>
//!    <consumer> <claim_idle_ms> 0 COUNT <batch>`), re-dispatch them, ack them.
//! 3. Poll `XREADGROUP GROUP <group> <consumer> COUNT <batch> STREAMS <key> >`,
//!    dispatch every entry, then ack it.
//!
//! A P2P entry is acknowledged only after its reply `XADD` succeeded, so a
//! failed reply write leaves the entry pending for [`xAUTOCLAIM`]-based recovery:
//! delivery is at-least-once, and a duplicate delivery is the price of never
//! losing a request. A malformed envelope (a poison entry) is logged and acked,
//! because a worker must never wedge its inbox on one bad payload.
//!
//! # Reconnect and backoff
//!
//! A failed command drops the shared socket (see [`super::RedisLink`]), so the
//! next command reconnects and replays `AUTH`/`SELECT`. Between two failed polls
//! the worker sleeps with a doubling delay capped at one second instead of
//! hammering an unreachable server, and it repeats the group-creation plus claim
//! pass once the read succeeds again, which is what recovers its own unacked
//! entries after a reconnect.
//!
//! [`consumer_name`]: super::RedisTransport::consumer_name
//! [`xAUTOCLAIM`]: super::streams::xautoclaim

use std::sync::Arc;
use std::time::Duration;

use openjiuwen_common::error::Result;
use tokio::sync::Mutex as AsyncMutex;
use tokio_util::sync::CancellationToken;

use crate::envelope::MessageEnvelope;
use crate::message_bus::MessageHandler;

use super::streams::{xack, xadd, xautoclaim, xgroup_create, xreadgroup, ReplyFrame, StreamEntry};
use super::RedisLink;

/// Lower bound of the delay between two failed polls.
const MIN_BACKOFF: Duration = Duration::from_millis(10);

/// Upper bound of the delay between two failed polls.
const MAX_BACKOFF: Duration = Duration::from_millis(1_000);

/// How many claim cursors one recovery pass follows before giving up until the
/// next poll, so a misbehaving cursor cannot spin the worker.
const MAX_CLAIM_PASSES: usize = 64;

/// What one inbox worker does with the entries it reads.
pub(super) enum WorkerMode {
    /// Reply to `recipient` through the request's reply stream.
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

/// The task draining one Redis stream consumer group.
pub(super) struct Worker {
    /// The stream this worker owns.
    pub(super) key: String,
    /// The consumer group shared by every transport of one deployment.
    pub(super) group: String,
    /// The consumer identity of this transport instance.
    pub(super) consumer: String,
    /// What a delivered envelope means.
    pub(super) mode: WorkerMode,
    /// The shared link.
    pub(super) link: Arc<AsyncMutex<RedisLink>>,
    /// The dispatcher.
    pub(super) handler: Arc<dyn MessageHandler>,
    /// `{prefix}:{team}:reply:` prefix of this team's reply streams.
    pub(super) reply_prefix: String,
    /// Idle delay between two polls.
    pub(super) poll_interval: Duration,
    /// Minimum idle time of a pending entry this worker may claim.
    pub(super) claim_idle_ms: u64,
    /// Entries requested by one read or claim pass.
    pub(super) batch: usize,
    /// Cancellation observed by every wait.
    pub(super) cancel: CancellationToken,
}

impl Worker {
    /// Drains the stream until the transport stops.
    pub(super) async fn run(self) {
        let mut backoff = self.poll_interval.max(MIN_BACKOFF);
        let mut ready = false;
        while !self.cancel.is_cancelled() {
            if !ready {
                if let Err(error) = xgroup_create(&self.link, &self.key, &self.group).await {
                    tracing::warn!(
                        key = %self.key,
                        group = %self.group,
                        %error,
                        "redis stream worker cannot create its consumer group"
                    );
                    if self.sleep_or_stop(backoff).await {
                        break;
                    }
                    backoff = doubled(backoff);
                    continue;
                }
                ready = true;
                backoff = self.poll_interval.max(MIN_BACKOFF);
                if let Err(error) = self.claim_pending().await {
                    tracing::warn!(
                        key = %self.key,
                        group = %self.group,
                        %error,
                        "redis stream worker could not claim pending entries"
                    );
                }
            }
            match xreadgroup(
                &self.link,
                &self.key,
                &self.group,
                &self.consumer,
                self.batch,
            )
            .await
            {
                Ok(entries) => {
                    backoff = self.poll_interval.max(MIN_BACKOFF);
                    if entries.is_empty() {
                        if self.sleep_or_stop(self.poll_interval).await {
                            break;
                        }
                    } else {
                        for entry in entries {
                            self.dispatch(entry).await;
                        }
                    }
                }
                Err(error) => {
                    tracing::warn!(
                        key = %self.key,
                        group = %self.group,
                        %error,
                        "redis stream worker poll failed"
                    );
                    // Re-run the group setup and the claim pass after recovery:
                    // the failure may have left this worker's own entries pending.
                    ready = false;
                    if self.sleep_or_stop(backoff).await {
                        break;
                    }
                    backoff = doubled(backoff);
                }
            }
        }
        tracing::debug!(key = %self.key, "redis stream worker stopped");
    }

    /// Sleeps `delay`, returning `true` when the transport stopped first.
    async fn sleep_or_stop(&self, delay: Duration) -> bool {
        tokio::select! {
            () = self.cancel.cancelled() => true,
            () = tokio::time::sleep(delay) => false,
        }
    }

    /// Re-dispatches every pending entry the server lets this consumer claim.
    async fn claim_pending(&self) -> Result<()> {
        let mut cursor = String::from("0-0");
        for _ in 0..MAX_CLAIM_PASSES {
            let (next, entries) = xautoclaim(
                &self.link,
                &self.key,
                &self.group,
                &self.consumer,
                self.claim_idle_ms,
                &cursor,
                self.batch,
            )
            .await?;
            for entry in entries {
                self.dispatch(entry).await;
            }
            if next.is_empty() || next == "0-0" {
                return Ok(());
            }
            cursor = next;
        }
        Ok(())
    }

    /// Dispatches one entry and acknowledges it where the contract allows.
    async fn dispatch(&self, entry: StreamEntry) {
        let Some(payload) = entry.field("envelope") else {
            tracing::warn!(
                key = %self.key,
                id = %entry.id,
                "redis stream entry carries no envelope field"
            );
            self.ack(&entry.id).await;
            return;
        };
        let Ok(envelope) = serde_json::from_slice::<MessageEnvelope>(payload) else {
            tracing::warn!(
                key = %self.key,
                id = %entry.id,
                "redis stream entry carries a malformed envelope"
            );
            self.ack(&entry.id).await;
            return;
        };
        match &self.mode {
            WorkerMode::P2p { recipient } => {
                self.deliver_p2p(&entry.id, &envelope, recipient).await;
            }
            WorkerMode::Pubsub { subscriber } => {
                if let Err(error) = self.handler.handle_pubsub(&envelope, subscriber).await {
                    tracing::warn!(
                        key = %self.key,
                        subscriber,
                        %error,
                        "redis transport pub/sub handler failed"
                    );
                }
                self.ack(&entry.id).await;
            }
        }
    }

    /// Runs one P2P delivery, writes its reply frame, then acknowledges it.
    ///
    /// The acknowledgement happens after the reply write: a failed write leaves
    /// the entry pending, so the request is retried instead of being lost.
    async fn deliver_p2p(&self, entry_id: &str, envelope: &MessageEnvelope, recipient: &str) {
        let frame = match self.handler.handle_p2p(envelope).await {
            Ok(value) => ReplyFrame::from_value(value),
            Err(error) => {
                tracing::warn!(
                    key = %self.key,
                    recipient,
                    %error,
                    "redis transport p2p handler failed"
                );
                ReplyFrame::from_error(error.to_string())
            }
        };
        let encoded = match frame.encode() {
            Ok(encoded) => encoded,
            Err(error) => {
                tracing::warn!(%error, "redis transport cannot encode a reply frame");
                return;
            }
        };
        let reply_key = format!("{}{}", self.reply_prefix, envelope.message_id);
        if let Err(error) = xadd(&self.link, &reply_key, &[("frame", &encoded)]).await {
            tracing::warn!(
                key = %reply_key,
                %error,
                "redis transport failed to publish a reply; the entry stays pending"
            );
            return;
        }
        self.ack(entry_id).await;
    }

    /// Acknowledges one entry, logging a failed acknowledgement.
    async fn ack(&self, id: &str) {
        if let Err(error) = xack(&self.link, &self.key, &self.group, id).await {
            tracing::warn!(
                key = %self.key,
                id,
                %error,
                "redis stream worker could not acknowledge an entry"
            );
        }
    }
}

/// Doubles `delay`, capped at [`MAX_BACKOFF`].
fn doubled(delay: Duration) -> Duration {
    (delay * 2).min(MAX_BACKOFF)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn backoff_doubles_and_caps() {
        assert_eq!(doubled(MIN_BACKOFF), Duration::from_millis(20));
        assert_eq!(doubled(Duration::from_millis(600)), MAX_BACKOFF);
        assert_eq!(doubled(MAX_BACKOFF), MAX_BACKOFF);
    }
}
