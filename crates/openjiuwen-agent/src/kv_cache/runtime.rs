//! In-process prefix runtime: record, lookup, `LRU`/`TTL` eviction, and the
//! deterministic prefix hash.
//!
//! Ports the bookkeeping half of `openjiuwen/core/kv_cache/kv_cache_runtime.py`
//! without the provider action state machine: this runtime keeps the prefixes an
//! agent has already sent so the next model call can report how much of its
//! request is reusable. Provider-side prefetch/offload is deferred (see
//! [`crate::kv_cache::types::KVCacheBackend`]).
//!
//! All state sits behind one [`parking_lot::Mutex`] and no lock is held across an
//! `await`, so the async protocol methods stay cancellation-safe.
use crate::kv_cache::config::KVCacheRuntimeConfig;
use crate::kv_cache::types::{
    KVCacheEntry, KVCacheIdentity, KVCacheLookup, KVCacheRuntimeProtocol, KVCacheStats,
};
use async_trait::async_trait;
use openjiuwen_common::error::Result;
use openjiuwen_foundation::{BaseMessage, ToolCard};
use parking_lot::Mutex;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Domain separator of the prefix hash canonical form.
const HASH_DOMAIN: &[u8] = b"openjiuwen:kv-prefix:v1";
/// `FNV-1a` 64-bit offset basis.
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
/// `FNV-1a` 64-bit prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Process-local [`KVCacheRuntimeProtocol`] implementation.
///
/// Entries are keyed by [`KVCacheIdentity`] and evicted least-recently-used when
/// the entry count exceeds the configured capacity. A capacity of `0` disables
/// retention (every record is evicted immediately); a configured `TTL` expires
/// entries that have not been recorded or looked up for that long.
pub struct InProcessKVCacheRuntime {
    capacity: usize,
    ttl: Option<Duration>,
    state: Mutex<State>,
}

/// Mutable state guarded by the runtime's mutex.
#[derive(Default)]
struct State {
    entries: HashMap<KVCacheIdentity, KVCacheEntry>,
    touched: HashMap<KVCacheIdentity, Instant>,
    sequence: u64,
    hits: u64,
    misses: u64,
    evictions: u64,
}

impl InProcessKVCacheRuntime {
    /// Creates a runtime that retains at most `capacity` entries.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            ttl: None,
            state: Mutex::new(State::default()),
        }
    }

    /// Creates a runtime with an idle `TTL`; `None` keeps entries until evicted.
    #[must_use]
    pub fn with_ttl(capacity: usize, ttl: Option<Duration>) -> Self {
        Self {
            capacity,
            ttl,
            state: Mutex::new(State::default()),
        }
    }

    /// Creates a runtime from a [`KVCacheRuntimeConfig`].
    ///
    /// The backend field is not consulted — the caller has already selected this
    /// runtime (see [`KVCacheRuntimeConfig::build_runtime`]).
    #[must_use]
    pub(crate) fn from_config(config: &KVCacheRuntimeConfig) -> Self {
        Self::with_ttl(config.capacity, config.ttl)
    }

    /// Maximum number of retained entries.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of retained entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.state.lock().entries.len()
    }

    /// Whether no entry is retained.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the retained entries, most recently used first.
    #[must_use]
    pub fn entries(&self) -> Vec<KVCacheEntry> {
        let state = self.state.lock();
        let mut entries: Vec<KVCacheEntry> = state.entries.values().cloned().collect();
        entries.sort_by_key(|entry| std::cmp::Reverse(entry.last_used_seq));
        entries
    }

    /// Drops every entry and resets all counters.
    pub fn clear(&self) {
        let mut state = self.state.lock();
        *state = State::default();
    }
}

#[async_trait]
impl KVCacheRuntimeProtocol for InProcessKVCacheRuntime {
    async fn record(&self, entry: KVCacheEntry) -> Result<()> {
        let mut state = self.state.lock();
        purge_expired(&mut state, self.ttl);
        upsert(&mut state, self.capacity, entry);
        Ok(())
    }

    async fn lookup(&self, identity: &KVCacheIdentity) -> Result<KVCacheLookup> {
        let mut guard = self.state.lock();
        let state: &mut State = &mut guard;
        purge_expired(state, self.ttl);
        if let Some(stored) = state.entries.get_mut(identity) {
            state.sequence = state.sequence.saturating_add(1);
            stored.last_used_seq = state.sequence;
            let entry = stored.clone();
            let reused_prefix_len = entry.prefix_len;
            state.touched.insert(identity.clone(), Instant::now());
            state.hits = state.hits.saturating_add(1);
            return Ok(KVCacheLookup::hit(entry, reused_prefix_len));
        }
        state.misses = state.misses.saturating_add(1);
        Ok(KVCacheLookup::miss())
    }

    async fn evict_session(&self, session_id: &str) -> Result<usize> {
        let mut state = self.state.lock();
        purge_expired(&mut state, self.ttl);
        let keys: Vec<KVCacheIdentity> = state
            .entries
            .keys()
            .filter(|identity| {
                identity.session_id == session_id
                    || identity.parent_session_id.as_deref() == Some(session_id)
            })
            .cloned()
            .collect();
        for key in &keys {
            state.entries.remove(key);
            state.touched.remove(key);
        }
        let removed = keys.len();
        state.evictions = state
            .evictions
            .saturating_add(u64::try_from(removed).unwrap_or(u64::MAX));
        Ok(removed)
    }

    async fn stats(&self) -> Result<KVCacheStats> {
        let state = self.state.lock();
        Ok(KVCacheStats {
            entries: state.entries.len(),
            hits: state.hits,
            misses: state.misses,
            evictions: state.evictions,
        })
    }
}

/// Inserts `entry`, merging with an existing entry for the same identity, then
/// evicts least-recently-used entries down to `capacity`.
fn upsert(state: &mut State, capacity: usize, mut entry: KVCacheEntry) {
    state.sequence = state.sequence.saturating_add(1);
    entry.last_used_seq = state.sequence;
    if let Some(existing) = state.entries.get(&entry.identity) {
        entry.hits = entry.hits.max(existing.hits);
        entry.token_estimate = entry.token_estimate.max(existing.token_estimate);
    }
    let key = entry.identity.clone();
    state.entries.insert(key.clone(), entry);
    state.touched.insert(key, Instant::now());
    while state.entries.len() > capacity {
        if !evict_least_recently_used(state) {
            break;
        }
    }
}

/// Removes the entry with the smallest `last_used_seq`; `false` when empty.
fn evict_least_recently_used(state: &mut State) -> bool {
    let oldest = state
        .entries
        .iter()
        .min_by_key(|(_, entry)| entry.last_used_seq)
        .map(|(identity, _)| identity.clone());
    let Some(identity) = oldest else {
        return false;
    };
    state.entries.remove(&identity);
    state.touched.remove(&identity);
    state.evictions = state.evictions.saturating_add(1);
    true
}

/// Drops entries idle for at least the configured `TTL`.
fn purge_expired(state: &mut State, ttl: Option<Duration>) {
    let Some(ttl) = ttl else {
        return;
    };
    let now = Instant::now();
    let expired: Vec<KVCacheIdentity> = state
        .touched
        .iter()
        .filter(|(_, touched)| now.saturating_duration_since(**touched) >= ttl)
        .map(|(identity, _)| identity.clone())
        .collect();
    for identity in &expired {
        state.entries.remove(identity);
        state.touched.remove(identity);
    }
    state.evictions = state
        .evictions
        .saturating_add(u64::try_from(expired.len()).unwrap_or(u64::MAX));
}

/// Returns the deterministic hash of one request prefix.
///
/// The hash is `FNV-1a` (64-bit, lowercase hex) over a canonical projection of
/// `model`, the ordered `messages`, and the ordered `tools`. Message content is
/// hashed in its serialized form so multimodal parts count, while provider-only
/// message metadata is excluded. The projection is built by hand instead of
/// hashing a `std` collection: no randomized hasher seed, no external hashing
/// crate, and therefore the same inputs produce the same hash on every run and
/// platform. Any change to the messages, the tool definitions, or the model name
/// changes the hash; this is a reuse gate, not a security primitive.
///
/// `prefix_hash_states` returns the hashes of every shorter prefix from the
/// same canonical stream, so a hook can probe candidate prefixes in one pass.
#[must_use]
pub fn stable_prefix_hash(messages: &[BaseMessage], tools: &[ToolCard], model: &str) -> String {
    prefix_hash_states(messages, tools, model)
        .into_iter()
        .last()
        .unwrap_or_else(empty_prefix_hash)
}

/// Returns the prefix hashes for `messages[0..len]` for every `len`, in order.
///
/// The result has `messages.len() + 1` elements; index `len` equals
/// [`stable_prefix_hash`] of the first `len` messages with the same tools and
/// model. Index `0` hashes the empty prefix.
#[must_use]
pub(crate) fn prefix_hash_states(
    messages: &[BaseMessage],
    tools: &[ToolCard],
    model: &str,
) -> Vec<String> {
    let suffix = canonical_suffix(tools, model);
    let suffix_len = length_prefix(suffix.len());
    let mut states = Vec::with_capacity(messages.len().saturating_add(1));
    let mut hash = fnv_update(FNV_OFFSET_BASIS, HASH_DOMAIN);
    states.push(finish_hash(hash, suffix_len, &suffix));
    for message in messages {
        let token = message_token(message);
        hash = fnv_update(hash, &length_prefix(token.len()));
        hash = fnv_update(hash, token.as_bytes());
        states.push(finish_hash(hash, suffix_len, &suffix));
    }
    states
}

/// Hashes the tools and model trailer onto a prefix state.
fn finish_hash(state: u64, suffix_len: [u8; 8], suffix: &str) -> String {
    let hash = fnv_update(fnv_update(state, &suffix_len), suffix.as_bytes());
    format!("{hash:016x}")
}

/// Hash of the empty prefix.
fn empty_prefix_hash() -> String {
    let suffix = canonical_suffix(&[], "");
    finish_hash(
        fnv_update(FNV_OFFSET_BASIS, HASH_DOMAIN),
        length_prefix(suffix.len()),
        &suffix,
    )
}

/// Little-endian length prefix that keeps token boundaries unambiguous.
fn length_prefix(len: usize) -> [u8; 8] {
    u64::try_from(len).unwrap_or(u64::MAX).to_le_bytes()
}

/// Canonical JSON one message contributes to the hash.
fn message_token(message: &BaseMessage) -> String {
    let tool_calls: Vec<Value> = message
        .tool_calls
        .iter()
        .map(|call| {
            json!({
                "id": call.id,
                "name": call.name,
                "arguments": call.arguments,
            })
        })
        .collect();
    json!({
        "role": message.role.as_str(),
        "content": &message.content,
        "name": message.name,
        "tool_calls": tool_calls,
        "tool_call_id": message.tool_call_id,
    })
    .to_string()
}

/// Canonical JSON trailer covering the tool definitions and the model name.
fn canonical_suffix(tools: &[ToolCard], model: &str) -> String {
    let tools: Vec<Value> = tools
        .iter()
        .map(|tool| {
            json!({
                "name": tool.name,
                "description": tool.description,
                "input_params": tool.input_params,
            })
        })
        .collect();
    json!({"tools": tools, "model": model}).to_string()
}

/// `FNV-1a` over `bytes`, continuing from `hash`.
fn fnv_update(mut hash: u64, bytes: &[u8]) -> u64 {
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::config::DEFAULT_KV_CACHE_CAPACITY;
    use openjiuwen_foundation::{BaseMessage, ToolCall, ToolCard};
    use serde_json::json;
    use std::sync::Arc;

    fn identity(prefix_hash: &str) -> KVCacheIdentity {
        KVCacheIdentity::new("session-a", prefix_hash, "gpt-test")
            .with_parent("session-a")
            .with_counts(2, 1)
    }

    fn entry(prefix_hash: &str, prefix_len: usize) -> KVCacheEntry {
        KVCacheEntry::new(identity(prefix_hash), prefix_len)
    }

    fn tool(name: &str) -> ToolCard {
        ToolCard {
            name: name.to_owned(),
            description: format!("{name} tool"),
            ..ToolCard::default()
        }
    }

    #[test]
    fn prefix_hash_is_stable_and_sensitive_to_every_input() {
        let messages = vec![
            BaseMessage::system("be helpful"),
            BaseMessage::user("hello"),
        ];
        let tools = vec![tool("search")];
        let baseline = stable_prefix_hash(&messages, &tools, "gpt-test");
        assert_eq!(baseline.len(), 16);
        assert_eq!(baseline, stable_prefix_hash(&messages, &tools, "gpt-test"));

        let mut changed_messages = messages.clone();
        changed_messages[1] = BaseMessage::user("hello there");
        assert_ne!(
            baseline,
            stable_prefix_hash(&changed_messages, &tools, "gpt-test")
        );

        let changed_tools = vec![tool("read")];
        assert_ne!(
            baseline,
            stable_prefix_hash(&messages, &changed_tools, "gpt-test")
        );
        assert_ne!(baseline, stable_prefix_hash(&messages, &tools, "gpt-other"));
    }

    #[test]
    fn prefix_hash_states_agree_with_the_slice_hash() {
        let messages = vec![
            BaseMessage::system("be helpful"),
            BaseMessage::user("hello"),
            BaseMessage::user("again"),
        ];
        let tools = vec![tool("search")];
        let states = prefix_hash_states(&messages, &tools, "gpt-test");
        assert_eq!(states.len(), messages.len() + 1);
        for len in 0..=messages.len() {
            assert_eq!(
                states[len],
                stable_prefix_hash(&messages[..len], &tools, "gpt-test"),
                "prefix {len}"
            );
        }
    }

    #[test]
    fn prefix_hash_covers_tool_calls() {
        let plain = vec![BaseMessage::assistant("calling")];
        let mut calling = plain.clone();
        calling[0]
            .tool_calls
            .push(ToolCall::new("search").with_arguments(json!({"q": "x"})));
        assert_ne!(
            stable_prefix_hash(&plain, &[], "gpt-test"),
            stable_prefix_hash(&calling, &[], "gpt-test")
        );
    }

    #[tokio::test]
    async fn record_then_lookup_reports_a_hit() {
        let runtime = InProcessKVCacheRuntime::new(4);
        runtime.record(entry("hash-a", 2)).await.unwrap();

        let lookup = runtime.lookup(&identity("hash-a")).await.unwrap();
        assert!(lookup.is_hit());
        assert_eq!(lookup.reused_prefix_len, 2);
        assert_eq!(lookup.entry.unwrap().prefix_len, 2);

        let stats = runtime.stats().await.unwrap();
        assert_eq!(stats.entries, 1);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 0);
        assert_eq!(stats.evictions, 0);
    }

    #[tokio::test]
    async fn unknown_identity_counts_a_miss() {
        let runtime = InProcessKVCacheRuntime::new(4);
        let lookup = runtime.lookup(&identity("missing")).await.unwrap();
        assert!(!lookup.is_hit());
        assert_eq!(lookup.reused_prefix_len, 0);
        assert!(lookup.entry.is_none());
        assert_eq!(runtime.stats().await.unwrap().misses, 1);
    }

    #[tokio::test]
    async fn least_recently_used_entry_is_evicted_at_capacity() {
        let runtime = InProcessKVCacheRuntime::new(2);
        runtime.record(entry("hash-a", 1)).await.unwrap();
        runtime.record(entry("hash-b", 1)).await.unwrap();
        // Touch A so B becomes the least recently used entry.
        assert!(runtime.lookup(&identity("hash-a")).await.unwrap().is_hit());
        runtime.record(entry("hash-c", 1)).await.unwrap();

        let hashes: Vec<String> = runtime
            .entries()
            .into_iter()
            .map(|entry| entry.prefix_hash)
            .collect();
        assert_eq!(hashes.len(), 2);
        assert!(hashes.contains(&"hash-a".to_owned()));
        assert!(hashes.contains(&"hash-c".to_owned()));
        assert!(!hashes.contains(&"hash-b".to_owned()));
        assert_eq!(runtime.stats().await.unwrap().evictions, 1);
    }

    #[tokio::test]
    async fn capacity_zero_disables_retention() {
        let runtime = InProcessKVCacheRuntime::new(0);
        runtime.record(entry("hash-a", 1)).await.unwrap();
        assert!(runtime.is_empty());
        assert_eq!(runtime.stats().await.unwrap().evictions, 1);
    }

    #[tokio::test]
    async fn upsert_keeps_hits_and_token_estimate() {
        let runtime = InProcessKVCacheRuntime::new(4);
        runtime
            .record(entry("hash-a", 2).with_token_estimate(120).with_hits(3))
            .await
            .unwrap();
        // A later record with default counters must not regress the entry.
        runtime.record(entry("hash-a", 2)).await.unwrap();

        let stored = runtime.entries().remove(0);
        assert_eq!(stored.hits, 3);
        assert_eq!(stored.token_estimate, 120);
    }

    #[tokio::test]
    async fn evict_session_removes_children_too() {
        let runtime = InProcessKVCacheRuntime::new(8);
        runtime.record(entry("root", 1)).await.unwrap();
        runtime
            .record(KVCacheEntry::new(
                KVCacheIdentity::new("child", "child-hash", "gpt-test").with_parent("session-a"),
                1,
            ))
            .await
            .unwrap();
        runtime
            .record(KVCacheEntry::new(
                KVCacheIdentity::new("other", "other-hash", "gpt-test"),
                1,
            ))
            .await
            .unwrap();

        let removed = runtime.evict_session("session-a").await.unwrap();
        assert_eq!(removed, 2);
        assert_eq!(runtime.len(), 1);
        assert_eq!(runtime.stats().await.unwrap().evictions, 2);
    }

    #[tokio::test]
    async fn ttl_expiry_drops_idle_entries() {
        let runtime = InProcessKVCacheRuntime::with_ttl(4, Some(Duration::ZERO));
        runtime.record(entry("hash-a", 1)).await.unwrap();
        assert!(!runtime.lookup(&identity("hash-a")).await.unwrap().is_hit());
        assert!(runtime.is_empty());
        let stats = runtime.stats().await.unwrap();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.evictions, 1);
    }

    #[tokio::test]
    async fn clear_resets_entries_and_counters() {
        let runtime = InProcessKVCacheRuntime::new(4);
        runtime.record(entry("hash-a", 1)).await.unwrap();
        let _ = runtime.lookup(&identity("hash-a")).await.unwrap();
        runtime.clear();
        assert!(runtime.is_empty());
        assert_eq!(
            runtime.stats().await.unwrap(),
            KVCacheStats {
                entries: 0,
                hits: 0,
                misses: 0,
                evictions: 0,
            }
        );
    }

    #[tokio::test]
    async fn runtime_from_config_uses_its_capacity() {
        let config = KVCacheRuntimeConfig::default().with_capacity(2);
        let runtime = InProcessKVCacheRuntime::from_config(&config);
        assert_eq!(runtime.capacity(), 2);
        for index in 0..3 {
            runtime
                .record(entry(&format!("hash-{index}"), 1))
                .await
                .unwrap();
        }
        assert_eq!(runtime.len(), 2);
        assert_eq!(runtime.capacity(), 2);
        assert_eq!(DEFAULT_KV_CACHE_CAPACITY, 256);
    }

    #[tokio::test]
    async fn concurrent_lookups_stay_consistent() {
        let runtime = Arc::new(InProcessKVCacheRuntime::new(8));
        runtime.record(entry("hash-a", 2)).await.unwrap();

        let mut handles = Vec::new();
        for _ in 0..2 {
            let runtime = Arc::clone(&runtime);
            handles.push(tokio::spawn(async move {
                let mut hits = 0;
                for _ in 0..50 {
                    if runtime.lookup(&identity("hash-a")).await.unwrap().is_hit() {
                        hits += 1;
                    }
                }
                hits
            }));
        }
        let mut total = 0;
        for handle in handles {
            total += handle.await.unwrap();
        }
        assert_eq!(total, 100);
        let stats = runtime.stats().await.unwrap();
        assert_eq!(stats.hits, 100);
        assert_eq!(stats.entries, 1);
        assert_eq!(runtime.entries()[0].hits, 0, "hits are bumped by the hook");
    }
}
