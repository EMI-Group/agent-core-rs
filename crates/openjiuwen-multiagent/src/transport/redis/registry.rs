//! The cross-process subscription registry of the Redis team transport.
//!
//! A transport's [`SubscriptionManager`] is a process-local snapshot: it decides
//! its own pub/sub fan-out. Sharing it needs a registry both processes can read,
//! so every `(agent, pattern)` pair is mirrored into two Redis SETs under the
//! transport's usual `{key_prefix}:{team}` namespace:
//!
//! | Key | Content |
//! | --- | --- |
//! | `{prefix}:{team}:subs` | every registered topic pattern |
//! | `{prefix}:{team}:subs:{pattern}` | the agent ids subscribed to that pattern |
//!
//! `add_subscription` writes through on a spawned best-effort task (`SADD` the
//! agent into the pattern set, `SADD` the pattern into the index) while the local
//! registry changes immediately, so same-instance behaviour stays deterministic
//! and a synchronous call never waits for the network. A removal mirrors the
//! same way (`SREM`, then `SCARD`, then `SREM` the index entry when the pattern
//! set emptied). A failed write is logged and leaves the local pair in place.
//!
//! [`SubscriptionRegistry::refresh`] re-reads the registry and replaces the local
//! view with `remote ∪ owned`, where `owned` is the set of pairs this instance
//! registered itself: a refresh can therefore never drop a local subscription
//! that has not reached Redis yet (or whose write failed). A refresher task that
//! [`RedisTransport::start`] spawns runs it every
//! [`RedisTransportConfig::registry_interval_ms`], and
//! [`RedisTransport::refresh_subscriptions`] runs it on demand.
//!
//! [`RedisTransport::stop`] removes the entries this instance owns, so a clean
//! shutdown leaves no stale routes. A crashed process may leave some behind: a
//! stale entry only makes a publisher `XADD` into a stream no worker consumes,
//! never a misdelivery.
//!
//! [`RedisTransport::start`]: super::RedisTransport::start
//! [`RedisTransport::stop`]: super::RedisTransport::stop
//! [`RedisTransport::refresh_subscriptions`]: super::RedisTransport::refresh_subscriptions
//! [`RedisTransportConfig::registry_interval_ms`]: super::RedisTransportConfig::registry_interval_ms

use std::collections::{BTreeMap, BTreeSet};
use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::store::external::redis::codec::RespValue;
use parking_lot::Mutex;
use tokio::sync::Mutex as AsyncMutex;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::subscription::SubscriptionManager;

use super::RedisLink;

/// Adds `member` to the Redis SET `key`.
async fn sadd(link: &AsyncMutex<RedisLink>, key: &str, member: &str) -> Result<()> {
    let args: [&[u8]; 3] = [b"SADD", key.as_bytes(), member.as_bytes()];
    let mut link = link.lock().await;
    match link.command(&args).await? {
        RespValue::Integer(_) => Ok(()),
        other => Err(unexpected("SADD", &other)),
    }
}

/// Removes `member` from the Redis SET `key`, returning whether it was there.
async fn srem(link: &AsyncMutex<RedisLink>, key: &str, member: &str) -> Result<bool> {
    let args: [&[u8]; 3] = [b"SREM", key.as_bytes(), member.as_bytes()];
    let mut link = link.lock().await;
    match link.command(&args).await? {
        RespValue::Integer(removed) => Ok(removed > 0),
        other => Err(unexpected("SREM", &other)),
    }
}

/// Returns the cardinality of the Redis SET `key`.
async fn scard(link: &AsyncMutex<RedisLink>, key: &str) -> Result<usize> {
    let args: [&[u8]; 2] = [b"SCARD", key.as_bytes()];
    let mut link = link.lock().await;
    match link.command(&args).await? {
        RespValue::Integer(count) => Ok(usize::try_from(count).unwrap_or(0)),
        other => Err(unexpected("SCARD", &other)),
    }
}

/// Returns every member of the Redis SET `key`.
async fn smembers(link: &AsyncMutex<RedisLink>, key: &str) -> Result<Vec<String>> {
    let args: [&[u8]; 2] = [b"SMEMBERS", key.as_bytes()];
    let mut link = link.lock().await;
    match link.command(&args).await? {
        RespValue::Array(Some(items)) | RespValue::Set(items) => Ok(items
            .into_iter()
            .filter_map(|item| item.as_string())
            .collect()),
        RespValue::Array(None) | RespValue::Null => Ok(Vec::new()),
        other => Err(unexpected("SMEMBERS", &other)),
    }
}

/// Builds the transport error used for a set-command reply of the wrong shape.
fn unexpected(command: &str, frame: &RespValue) -> Error {
    Error::transport(format!(
        "redis transport: unexpected {command} reply: {frame:?}"
    ))
}

/// The shared subscription registry of one team, mirrored into Redis.
///
/// See the module documentation for the key scheme, the write-through rules and
/// the `remote ∪ owned` merge.
pub(super) struct SubscriptionRegistry {
    /// The local view a transport fans out from.
    manager: Arc<SubscriptionManager>,
    /// The shared RESP2 link of the transport.
    link: Arc<AsyncMutex<RedisLink>>,
    /// Prefix of every key.
    key_prefix: String,
    /// Effective team id embedded in every key.
    team_id: String,
    /// The `(agent, pattern)` pairs this instance registered itself.
    ///
    /// Held while the local registry changes, so a concurrent refresh cannot
    /// interleave its wholesale replace with a local add and lose the pair.
    owned: Mutex<BTreeSet<(String, String)>>,
    /// Whether the "no runtime to write through" warning was already logged.
    no_runtime_logged: AtomicBool,
}

impl SubscriptionRegistry {
    /// Builds a registry over the transport's local manager and link.
    pub(super) fn new(
        manager: Arc<SubscriptionManager>,
        link: Arc<AsyncMutex<RedisLink>>,
        key_prefix: &str,
        team_id: &str,
    ) -> Self {
        Self {
            manager,
            link,
            key_prefix: key_prefix.to_owned(),
            team_id: team_id.to_owned(),
            owned: Mutex::new(BTreeSet::new()),
            no_runtime_logged: AtomicBool::new(false),
        }
    }

    /// Returns the key of the index SET holding every registered pattern.
    pub(super) fn index_key(&self) -> String {
        format!("{}:{}:subs", self.key_prefix, self.team_id)
    }

    /// Returns the key of the SET holding the agents of `pattern`.
    pub(super) fn pattern_key(&self, pattern: &str) -> String {
        format!("{}:{}", self.index_key(), pattern)
    }

    /// Registers `(agent_id, topic)` locally and mirrors it into Redis.
    pub(super) fn add(&self, agent_id: &str, topic: &str) {
        if agent_id.is_empty() || topic.is_empty() {
            return;
        }
        {
            let mut owned = self.owned.lock();
            self.manager.subscribe(agent_id, topic);
            owned.insert((agent_id.to_owned(), topic.to_owned()));
        }
        let link = Arc::clone(&self.link);
        let key = self.pattern_key(topic);
        let index = self.index_key();
        let agent = agent_id.to_owned();
        let pattern = topic.to_owned();
        let team = self.team_id.clone();
        self.spawn(async move {
            if let Err(error) = sadd(&link, &key, &agent).await {
                tracing::warn!(
                    team = %team,
                    key = %key,
                    %error,
                    "redis transport could not register a subscription"
                );
            }
            if let Err(error) = sadd(&link, &index, &pattern).await {
                tracing::warn!(
                    team = %team,
                    key = %index,
                    %error,
                    "redis transport could not index a subscription pattern"
                );
            }
        });
    }

    /// Removes `(agent_id, topic)` locally and from Redis.
    pub(super) fn remove(&self, agent_id: &str, topic: &str) {
        if agent_id.is_empty() || topic.is_empty() {
            return;
        }
        {
            let mut owned = self.owned.lock();
            self.manager.unsubscribe(agent_id, topic);
            owned.remove(&(agent_id.to_owned(), topic.to_owned()));
        }
        let link = Arc::clone(&self.link);
        let key = self.pattern_key(topic);
        let index = self.index_key();
        let agent = agent_id.to_owned();
        let pattern = topic.to_owned();
        let team = self.team_id.clone();
        self.spawn(async move {
            match srem(&link, &key, &agent).await {
                Ok(_) => {}
                Err(error) => {
                    tracing::warn!(
                        team = %team,
                        key = %key,
                        %error,
                        "redis transport could not remove a subscription"
                    );
                    return;
                }
            }
            match scard(&link, &key).await {
                // An empty pattern set would otherwise keep the index growing.
                Ok(0) => {
                    if let Err(error) = srem(&link, &index, &pattern).await {
                        tracing::warn!(
                            team = %team,
                            key = %index,
                            %error,
                            "redis transport could not drop an empty subscription pattern"
                        );
                    }
                }
                Ok(_) => {}
                Err(error) => {
                    tracing::warn!(
                        team = %team,
                        key = %key,
                        %error,
                        "redis transport could not count a subscription pattern"
                    );
                }
            }
        });
    }

    /// Rebuilds the local registry from the shared one plus the owned pairs.
    ///
    /// # Errors
    ///
    /// A transport error when a registry read fails; the local registry then
    /// keeps its previous content.
    pub(super) async fn refresh(&self) -> Result<()> {
        let mut remote: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for pattern in smembers(&self.link, &self.index_key()).await? {
            let agents = smembers(&self.link, &self.pattern_key(&pattern)).await?;
            remote.insert(pattern, agents);
        }
        self.apply(remote);
        Ok(())
    }

    /// Replaces the local registry with `remote ∪ owned`.
    fn apply(&self, remote: BTreeMap<String, Vec<String>>) {
        let owned = self.owned.lock();
        let mut entries: BTreeSet<(String, String)> = BTreeSet::new();
        for (pattern, agents) in remote {
            for agent in agents {
                entries.insert((agent, pattern.clone()));
            }
        }
        entries.extend(owned.iter().cloned());
        self.manager.replace_all(entries);
    }

    /// Re-publishes every owned pair into Redis, best effort.
    ///
    /// [`RedisTransport::stop`](super::RedisTransport::stop) removes them, so a
    /// later [`start`](super::super::TeamTransport::start) restores the routes
    /// without the caller re-adding every subscription. Failures are logged and
    /// never fail the start.
    pub(super) async fn republish(&self) {
        let owned: Vec<(String, String)> = self.owned.lock().iter().cloned().collect();
        let index = self.index_key();
        for (agent, pattern) in owned {
            let key = self.pattern_key(&pattern);
            if let Err(error) = sadd(&self.link, &key, &agent).await {
                tracing::warn!(key = %key, %error, "redis transport could not re-register a subscription");
                continue;
            }
            if let Err(error) = sadd(&self.link, &index, &pattern).await {
                tracing::warn!(key = %index, %error, "redis transport could not re-index a subscription pattern");
            }
        }
    }

    /// Removes every owned pair from Redis, best effort, so a clean shutdown
    /// leaves no stale routes.
    ///
    /// The local view is kept: a restarted transport re-publishes it through
    /// [`republish`](Self::republish).
    pub(super) async fn clear(&self) {
        let owned: Vec<(String, String)> = self.owned.lock().iter().cloned().collect();
        let index = self.index_key();
        for (agent, pattern) in owned {
            let key = self.pattern_key(&pattern);
            if let Err(error) = srem(&self.link, &key, &agent).await {
                tracing::warn!(key = %key, %error, "redis transport could not remove an owned subscription");
                continue;
            }
            match scard(&self.link, &key).await {
                Ok(0) => {
                    if let Err(error) = srem(&self.link, &index, &pattern).await {
                        tracing::warn!(
                            key = %index,
                            %error,
                            "redis transport could not drop an empty subscription pattern"
                        );
                    }
                }
                Ok(_) => {}
                Err(error) => {
                    tracing::warn!(key = %key, %error, "redis transport could not count a subscription pattern");
                }
            }
        }
    }

    /// Spawns the task that refreshes the registry every `interval`.
    ///
    /// Returns `None` when no tokio runtime is available (the transport then
    /// only learns remote subscriptions from
    /// [`RedisTransport::refresh_subscriptions`](super::RedisTransport::refresh_subscriptions)).
    pub(super) fn spawn_refresher(
        registry: Arc<Self>,
        interval: Duration,
        cancel: CancellationToken,
    ) -> Option<JoinHandle<()>> {
        let handle = tokio::runtime::Handle::try_current().ok()?;
        Some(handle.spawn(async move {
            loop {
                if let Err(error) = registry.refresh().await {
                    tracing::warn!(
                        team = %registry.team_id,
                        %error,
                        "redis transport could not refresh the subscription registry"
                    );
                }
                tokio::select! {
                    () = cancel.cancelled() => break,
                    () = tokio::time::sleep(interval) => {}
                }
            }
            tracing::debug!(team = %registry.team_id, "redis transport registry refresher stopped");
        }))
    }

    /// Spawns one best-effort registry write when a runtime is available.
    fn spawn<F>(&self, write: F)
    where
        F: Future<Output = ()> + Send + 'static,
    {
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => {
                // The task is detached on purpose: the caller is synchronous and
                // must neither block nor await, and the write is best effort.
                drop(handle.spawn(write));
            }
            Err(_) => {
                if !self.no_runtime_logged.swap(true, Ordering::Relaxed) {
                    tracing::warn!(
                        team = %self.team_id,
                        "redis transport subscription registry stays local: no tokio runtime to write through"
                    );
                }
            }
        }
    }
}

impl super::RedisTransport {
    /// Refreshes the local subscription view from the shared Redis registry.
    ///
    /// The local registry becomes `remote ∪ owned`: subscriptions this instance
    /// added itself survive even when their write has not reached Redis yet.
    /// [`start`](super::super::TeamTransport::start) runs the same refresh
    /// periodically; this method makes it deterministic for tests and for a
    /// caller that wants the view up to date before publishing.
    ///
    /// # Errors
    ///
    /// A transport error when a registry read fails; the local registry then
    /// keeps its previous content.
    pub async fn refresh_subscriptions(&self) -> Result<()> {
        self.registry.refresh().await
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::transport::redis::RedisTransportConfig;

    /// Builds a registry over a disconnected link (no command is issued here).
    fn registry() -> (Arc<SubscriptionRegistry>, Arc<SubscriptionManager>) {
        let manager = Arc::new(SubscriptionManager::new());
        let link = Arc::new(AsyncMutex::new(
            RedisLink::new(&RedisTransportConfig::new()),
        ));
        let registry = Arc::new(SubscriptionRegistry::new(
            Arc::clone(&manager),
            link,
            "team",
            "t1",
        ));
        (registry, manager)
    }

    #[test]
    fn keys_share_the_transport_namespace() {
        let (registry, _manager) = registry();
        assert_eq!(registry.index_key(), "team:t1:subs");
        assert_eq!(registry.pattern_key("news"), "team:t1:subs:news");
        assert_eq!(registry.pattern_key("team.*"), "team:t1:subs:team.*");
    }

    #[test]
    fn local_state_changes_immediately() {
        let (registry, manager) = registry();
        registry.add("a", "news");
        registry.add("b", "news");
        assert_eq!(manager.subscribers("news"), vec!["a", "b"]);
        assert_eq!(manager.list().len(), 1);

        registry.remove("a", "news");
        assert_eq!(manager.subscribers("news"), vec!["b"]);
        registry.remove("unknown", "news");
        assert_eq!(manager.subscription_count(), 1);
        registry.add("", "news");
        registry.add("b", "");
        assert_eq!(manager.subscription_count(), 1);
    }

    #[test]
    fn refresh_merge_keeps_owned_pairs_in_both_directions() {
        let (registry, manager) = registry();
        registry.add("a", "news");
        registry.apply(BTreeMap::from([(
            "news".to_owned(),
            vec!["b".to_owned(), "c".to_owned()],
        )]));
        assert_eq!(manager.subscribers("news"), vec!["a", "b", "c"]);

        registry.remove("a", "news");
        registry.apply(BTreeMap::from([("news".to_owned(), vec!["b".to_owned()])]));
        assert_eq!(manager.subscribers("news"), vec!["b"]);
        assert_eq!(manager.topics_of("c"), Vec::<String>::new());
        assert_eq!(manager.subscription_count(), 1);
    }
}
