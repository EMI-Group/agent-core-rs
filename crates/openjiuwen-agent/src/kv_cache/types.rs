//! KV-cache identities, entries, lookups, statistics, and the runtime protocol.
//!
//! Ports `openjiuwen/core/kv_cache/kv_cache_types.py` onto the Rust kernel: the
//! provider-facing prefix identity, the cached-prefix bookkeeping an in-process
//! runtime keeps, and the narrow async contract the model-call hook consumes.
//!
//! A [`KVCacheIdentity`] is what the runtime keys on: a session lineage
//! (`session_id` plus `parent_session_id`), the hash of the request prefix, the
//! model name, and the message/tool counts that were current when the entry was
//! recorded. [`KVCacheBackend`] is the factory seam — only the in-process
//! backend exists, and the other variants fail with [`Error::Unsupported`] so a
//! misconfigured distributed cache is loud instead of silently degrading into a
//! no-op.
use crate::kv_cache::config::KVCacheRuntimeConfig;
use crate::kv_cache::runtime::InProcessKVCacheRuntime;
use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::Arc;

/// Identity of one reusable request prefix.
///
/// Serializes with `snake_case` field names and tolerates unknown and missing
/// fields. `parent_session_id` is the lineage the prefix branches from; it
/// defaults to the resolved session id when the caller has no parent. An empty
/// `prefix_hash` marks a lineage-only identity (for example
/// [`crate::kv_cache::metadata::team_member_cache_identity`]) that the
/// model-call hook fills in with the observed prefix hash.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct KVCacheIdentity {
    /// Session (or derived cache id) that owns the prefix.
    pub session_id: String,
    /// Lineage the prefix branches from; `None` for an unset identity.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_session_id: Option<String>,
    /// Deterministic hash of the request prefix (see
    /// [`crate::kv_cache::runtime::stable_prefix_hash`]).
    pub prefix_hash: String,
    /// Model the prefix was built for.
    pub model: String,
    /// Number of leading messages the identity covers.
    pub message_count: usize,
    /// Number of tool definitions the identity covers.
    pub tool_count: usize,
}

impl KVCacheIdentity {
    /// Creates an identity for `session_id` whose prefix hashes to `prefix_hash`.
    ///
    /// The parent lineage defaults to `None` (a root identity) and both counts
    /// start at zero; use [`Self::with_parent`] and [`Self::with_counts`] to fill
    /// them in.
    #[must_use]
    pub fn new(
        session_id: impl Into<String>,
        prefix_hash: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            session_id: session_id.into(),
            prefix_hash: prefix_hash.into(),
            model: model.into(),
            ..Self::default()
        }
    }

    /// Sets the parent session id the identity branches from.
    #[must_use]
    pub fn with_parent(mut self, parent_session_id: impl Into<String>) -> Self {
        self.parent_session_id = Some(parent_session_id.into());
        self
    }

    /// Sets the covered message and tool counts.
    #[must_use]
    pub const fn with_counts(mut self, message_count: usize, tool_count: usize) -> Self {
        self.message_count = message_count;
        self.tool_count = tool_count;
        self
    }

    /// The parent session id, when the identity has one.
    #[must_use]
    pub fn parent(&self) -> Option<&str> {
        self.parent_session_id.as_deref()
    }

    /// Whether the identity is its own root (no parent, or a self parent).
    #[must_use]
    pub fn is_root(&self) -> bool {
        self.parent().is_none_or(|parent| parent == self.session_id)
    }
}

/// One cached request prefix.
///
/// `last_used_seq` is runtime-managed: [`KVCacheRuntimeProtocol::record`] stamps
/// it with the runtime's monotonic sequence so least-recently-used eviction is
/// deterministic. `hits` and `token_estimate` never regress on an upsert.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct KVCacheEntry {
    /// Identity of the cached prefix.
    pub identity: KVCacheIdentity,
    /// Hash of the prefix this entry covers.
    pub prefix_hash: String,
    /// Number of leading messages the reusable prefix covers.
    pub prefix_len: usize,
    /// Token estimate of the prefix, filled in by
    /// [`crate::kv_cache::hook::KVCacheModelCallHook::note_usage`].
    pub token_estimate: usize,
    /// Runtime sequence of the last record or lookup; runtime-managed.
    pub last_used_seq: u64,
    /// Number of lookups that reused this entry.
    pub hits: u64,
}

impl KVCacheEntry {
    /// Creates an entry for `identity` covering `prefix_len` messages.
    #[must_use]
    pub fn new(identity: KVCacheIdentity, prefix_len: usize) -> Self {
        Self {
            prefix_hash: identity.prefix_hash.clone(),
            identity,
            prefix_len,
            ..Self::default()
        }
    }

    /// Sets the token estimate of the prefix.
    #[must_use]
    pub const fn with_token_estimate(mut self, token_estimate: usize) -> Self {
        self.token_estimate = token_estimate;
        self
    }

    /// Sets the reuse counter.
    #[must_use]
    pub const fn with_hits(mut self, hits: u64) -> Self {
        self.hits = hits;
        self
    }

    /// Returns a copy whose hit counter is bumped by one.
    #[must_use]
    pub fn bumped(&self) -> Self {
        let mut bumped = self.clone();
        bumped.hits = bumped.hits.saturating_add(1);
        bumped
    }
}

/// Result of one runtime lookup.
///
/// A miss carries [`KVCacheLookup::miss`]'s zero `reused_prefix_len`; a hit
/// carries the matched [`KVCacheEntry`] and the number of messages the caller
/// may reuse.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct KVCacheLookup {
    /// Whether a reusable prefix was found.
    pub hit: bool,
    /// The matched entry, when [`Self::hit`] is `true`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entry: Option<KVCacheEntry>,
    /// Number of leading messages that can be reused (`0` on a miss).
    pub reused_prefix_len: usize,
}

impl KVCacheLookup {
    /// Builds a miss with no entry and no reusable prefix.
    #[must_use]
    pub const fn miss() -> Self {
        Self {
            hit: false,
            entry: None,
            reused_prefix_len: 0,
        }
    }

    /// Builds a hit for `entry` that reuses `reused_prefix_len` messages.
    #[must_use]
    pub fn hit(entry: KVCacheEntry, reused_prefix_len: usize) -> Self {
        Self {
            hit: true,
            entry: Some(entry),
            reused_prefix_len,
        }
    }

    /// Whether the lookup found a reusable prefix.
    #[must_use]
    pub const fn is_hit(&self) -> bool {
        self.hit
    }
}

/// Aggregate runtime counters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct KVCacheStats {
    /// Number of retained entries.
    pub entries: usize,
    /// Number of lookups that found an entry.
    pub hits: u64,
    /// Number of lookups that found no entry.
    pub misses: u64,
    /// Number of entries dropped by capacity, `TTL`, or session eviction.
    pub evictions: u64,
}

impl KVCacheStats {
    /// Total number of lookups served.
    #[must_use]
    pub const fn lookups(&self) -> u64 {
        self.hits.saturating_add(self.misses)
    }
}

/// Narrow async contract the model-call hook talks to.
///
/// The protocol deliberately omits lifecycle and provider-action methods: hosts
/// hold the concrete runtime when they need to clear it or destroy it, and
/// provider-side prefetch/offload/evict runs through
/// [`crate::kv_cache::action_runtime::KvCacheActionRuntime`], which is a separate
/// seam from this prefix bookkeeping.
#[async_trait]
pub trait KVCacheRuntimeProtocol: Send + Sync {
    /// Records (upserts) one cached prefix.
    ///
    /// An existing entry for the same identity keeps the greater `hits` and
    /// `token_estimate`; `last_used_seq` is always restamped.
    ///
    /// # Errors
    ///
    /// Returns the backend's record error; the in-process runtime never fails.
    async fn record(&self, entry: KVCacheEntry) -> Result<()>;

    /// Looks up the entry for `identity` and touches it for `LRU` ordering.
    ///
    /// # Errors
    ///
    /// Returns the backend's lookup error; the in-process runtime never fails.
    async fn lookup(&self, identity: &KVCacheIdentity) -> Result<KVCacheLookup>;

    /// Drops every entry owned by `session_id`, including its children.
    ///
    /// Returns the number of evicted entries.
    ///
    /// # Errors
    ///
    /// Returns the backend's eviction error; the in-process runtime never fails.
    async fn evict_session(&self, session_id: &str) -> Result<usize>;

    /// Returns the current counters.
    ///
    /// # Errors
    ///
    /// Returns the backend's statistics error; the in-process runtime never
    /// fails.
    async fn stats(&self) -> Result<KVCacheStats>;
}

/// Storage backend that serves [`KVCacheRuntimeProtocol`].
///
/// Only [`KVCacheBackend::InProcess`] is implemented. `Redis` and `Distributed`
/// exist so a host can express the intent and get a typed
/// [`Error::Unsupported`] from [`Self::runtime`] instead of an accidental
/// in-process fallback.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KVCacheBackend {
    /// Process-local prefix bookkeeping.
    #[default]
    InProcess,
    /// Redis-backed runtime (deferred).
    Redis,
    /// Distributed runtime (deferred).
    Distributed,
}

impl KVCacheBackend {
    /// The backend name used in configuration and error messages.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::InProcess => "in-process",
            Self::Redis => "redis",
            Self::Distributed => "distributed",
        }
    }

    /// Builds the runtime this backend selects.
    ///
    /// `config` supplies the capacity, `TTL`, and affinity settings; its own
    /// `backend` field is ignored so this method stays the explicit selector.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`] for every backend other than
    /// [`KVCacheBackend::InProcess`].
    pub fn runtime(self, config: &KVCacheRuntimeConfig) -> Result<Arc<dyn KVCacheRuntimeProtocol>> {
        match self {
            Self::InProcess => Ok(Arc::new(InProcessKVCacheRuntime::from_config(config))),
            Self::Redis | Self::Distributed => Err(Error::unsupported(format!(
                "kv-cache backend '{}' is not implemented; use the in-process runtime",
                self.as_str()
            ))),
        }
    }
}

impl fmt::Display for KVCacheBackend {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn identity() -> KVCacheIdentity {
        KVCacheIdentity::new("session-a", "abc123", "gpt-test")
            .with_parent("session-root")
            .with_counts(4, 2)
    }

    #[test]
    fn identity_round_trips_through_json() {
        let identity = identity();
        let value = serde_json::to_value(&identity).unwrap();
        assert_eq!(
            value,
            json!({
                "session_id": "session-a",
                "parent_session_id": "session-root",
                "prefix_hash": "abc123",
                "model": "gpt-test",
                "message_count": 4,
                "tool_count": 2,
            })
        );
        let parsed: KVCacheIdentity = serde_json::from_value(value).unwrap();
        assert_eq!(parsed, identity);
    }

    #[test]
    fn identity_tolerates_unknown_and_missing_fields() {
        let parsed: KVCacheIdentity =
            serde_json::from_value(json!({"session_id": "s", "future_field": 7})).unwrap();
        assert_eq!(parsed.session_id, "s");
        assert_eq!(parsed.model, "");
        assert!(parsed.parent().is_none());
        assert!(parsed.is_root());
        assert!(!identity().is_root());
    }

    #[test]
    fn entry_helpers_track_prefix_and_hits() {
        let entry = KVCacheEntry::new(identity(), 4).with_token_estimate(120);
        assert_eq!(entry.prefix_hash, "abc123");
        assert_eq!(entry.prefix_len, 4);
        assert_eq!(entry.hits, 0);
        assert_eq!(entry.bumped().hits, 1);
        assert_eq!(entry.hits, 0);
    }

    #[test]
    fn lookup_constructors_agree_with_flags() {
        let miss = KVCacheLookup::miss();
        assert!(!miss.is_hit());
        assert_eq!(miss.reused_prefix_len, 0);
        assert!(miss.entry.is_none());

        let hit = KVCacheLookup::hit(KVCacheEntry::new(identity(), 3), 3);
        assert!(hit.is_hit());
        assert_eq!(hit.reused_prefix_len, 3);
        assert!(hit.entry.is_some());
    }

    #[test]
    fn stats_count_lookups() {
        let stats = KVCacheStats {
            entries: 2,
            hits: 3,
            misses: 4,
            evictions: 1,
        };
        assert_eq!(stats.lookups(), 7);
    }

    #[test]
    fn backend_names_and_deferred_errors_are_typed() {
        assert_eq!(KVCacheBackend::default(), KVCacheBackend::InProcess);
        assert_eq!(KVCacheBackend::InProcess.to_string(), "in-process");
        assert_eq!(KVCacheBackend::Redis.to_string(), "redis");
        assert_eq!(KVCacheBackend::Distributed.to_string(), "distributed");

        let config = KVCacheRuntimeConfig::default();
        assert!(KVCacheBackend::InProcess.runtime(&config).is_ok());

        for backend in [KVCacheBackend::Redis, KVCacheBackend::Distributed] {
            let Err(error) = backend.runtime(&config) else {
                panic!("{backend} must be deferred");
            };
            assert_eq!(
                error.kind(),
                openjiuwen_common::error::ErrorKind::Unsupported
            );
            assert!(error.to_string().contains(&format!(
                "kv-cache backend '{}' is not implemented; use the in-process runtime",
                backend.as_str()
            )));
        }
    }

    #[test]
    fn backend_serializes_with_snake_case_names() {
        assert_eq!(
            serde_json::to_value(KVCacheBackend::InProcess).unwrap(),
            json!("in_process")
        );
        assert_eq!(
            serde_json::to_value(KVCacheBackend::Redis).unwrap(),
            json!("redis")
        );
    }
}
