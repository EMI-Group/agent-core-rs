//! Team-member KV-cache affinity: identity registry and per-member release.
//!
//! Every member of a team owns one provider-facing cache identity so its
//! requests reuse the same prefix cache across turns. This module binds a member
//! name to
//! [`team_member_cache_identity`]
//! (the persisted `team:{session}:team:{team}:member:{member}` shape, parented at
//! the team session), derives forked child identities from a parent member, and
//! renders the env bag / child-session kwargs a spawned session needs to inherit
//! that lineage.
//!
//! It owns no cache I/O itself: recording, lookup and eviction are delegated to
//! the dependency runtime through
//! [`KVCacheRuntimeProtocol`].
//!
//! Affinity is best-effort and opt-in, exactly like the agent-side primitives:
//! with a disabled [`KVCacheAffinityConfig`] nothing is registered and every
//! accessor returns `None`, so a host pays no cost and correctness never depends
//! on the cache.
use openjiuwen_agent::kv_cache::{
    team_member_cache_id, team_member_cache_identity, InProcessKVCacheRuntime,
    KVCacheAffinityConfig, KVCacheIdentity, KVCacheModelCallHook, KVCacheRuntimeProtocol,
    KV_CACHE_AFFINITY_PARENT_SESSION_ID_ENV, KV_CACHE_AFFINITY_SESSION_ID_ENV,
};
use openjiuwen_common::error::{Error, Result};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;
use tracing::warn;
/// Suffix inserted between a parent cache id and a forked member name.
///
/// The fork seam is local to this crate: the agent crate exposes no team-member
/// fork constructor, so the child session id is defined here and documented as
/// such.
pub const MEMBER_FORK_SEGMENT: &str = "fork";
/// Identity of one member's cache.
///
/// The triple is the routing key of the registry: `team_name` doubles as the
/// team id in the persisted cache id (the team schema has no separate team id),
/// `team_session_id` is the product-level team session and therefore the lineage
/// root, and `member_name` is the roster name.
#[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct MemberCacheKey {
    /// Team the member belongs to (used as the team id in the cache id).
    pub team_name: String,
    /// Product-level team session every member identity is parented at.
    pub team_session_id: String,
    /// Roster name of the member.
    pub member_name: String,
}
impl MemberCacheKey {
    /// Creates a cache key for `member_name` of `team_name` in `team_session_id`.
    #[must_use]
    pub fn new(
        team_name: impl Into<String>,
        team_session_id: impl Into<String>,
        member_name: impl Into<String>,
    ) -> Self {
        Self {
            team_name: team_name.into(),
            team_session_id: team_session_id.into(),
            member_name: member_name.into(),
        }
    }
    /// The persisted cache id of this member.
    ///
    /// Format `team:{team_session_id}:team:{team_name}:member:{member_name}`,
    /// identical to the Python contract (see
    /// [`team_member_cache_id`]).
    #[must_use]
    pub fn cache_id(&self) -> String {
        team_member_cache_id(&self.team_session_id, &self.team_name, &self.member_name)
    }
}
/// One registered member: its key, its cache identity and its fork parent.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct MemberCacheEntry {
    /// Routing key the entry was registered under.
    pub key: MemberCacheKey,
    /// Lineage-only identity (empty prefix hash, zero counts) the model-call
    /// hook fills in when it records a concrete request.
    pub identity: KVCacheIdentity,
    /// Member this entry was forked from, when it is a child.
    pub parent_member: Option<String>,
}
impl MemberCacheEntry {
    /// Creates a root entry (no fork parent).
    #[must_use]
    pub fn new(key: MemberCacheKey, identity: KVCacheIdentity) -> Self {
        Self {
            key,
            identity,
            parent_member: None,
        }
    }
    /// Marks the entry as a fork of `parent_member`.
    #[must_use]
    pub fn with_parent_member(mut self, parent_member: impl Into<String>) -> Self {
        self.parent_member = Some(parent_member.into());
        self
    }
}
/// Per-team registry of member cache identities.
///
/// One instance serves one team run: it holds the shared runtime handle, the
/// affinity configuration, and the member entries. Registrations are cheap and
/// idempotent; [`Self::release`] evicts the runtime entries of a member (and of
/// every member forked from it) before dropping the registry rows, so a
/// one-shot worker cannot leave provider-side cache behind.
///
/// The registry is `Send + Sync` and lock-safe: the internal mutex is only held
/// for short map operations, never across an `await`.
pub struct TeamKvCacheAffinity {
    /// Affinity configuration the registry was built from.
    config: KVCacheAffinityConfig,
    /// Shared runtime; `None` when affinity is disabled.
    runtime: Option<Arc<dyn KVCacheRuntimeProtocol>>,
    /// Registered members, keyed by member name (sorted).
    entries: Mutex<BTreeMap<String, MemberCacheEntry>>,
}
impl fmt::Debug for TeamKvCacheAffinity {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TeamKvCacheAffinity")
            .field("config", &self.config)
            .field("has_runtime", &self.runtime.is_some())
            .field("entries", &self.members())
            .finish_non_exhaustive()
    }
}
impl TeamKvCacheAffinity {
    /// Creates a registry for `config`.
    ///
    /// An enabled config builds a process-local
    /// [`InProcessKVCacheRuntime`] whose capacity is
    /// [`KVCacheAffinityConfig::max_entries`]; a disabled config keeps no
    /// runtime and therefore never touches a cache.
    #[must_use]
    pub fn new(config: KVCacheAffinityConfig) -> Self {
        let runtime = config.is_enabled().then(|| {
            Arc::new(InProcessKVCacheRuntime::new(config.max_entries))
                as Arc<dyn KVCacheRuntimeProtocol>
        });
        Self {
            config,
            runtime,
            entries: Mutex::new(BTreeMap::new()),
        }
    }
    /// Creates a disabled registry: nothing is stored, nothing is evicted.
    #[must_use]
    pub fn disabled() -> Self {
        Self::new(KVCacheAffinityConfig::default())
    }
    /// Replaces the runtime, for a host that shares one runtime across teams or
    /// injects a fake in tests.
    ///
    /// Whether affinity is enabled stays a property of the configuration, so
    /// injecting a runtime into a disabled registry does not turn registration
    /// on.
    #[must_use]
    pub fn with_runtime(mut self, runtime: Arc<dyn KVCacheRuntimeProtocol>) -> Self {
        self.runtime = Some(runtime);
        self
    }
    /// Whether prefix affinity is enabled for this registry.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.config.is_enabled()
    }
    /// The configuration the registry was built from.
    #[must_use]
    pub const fn config(&self) -> &KVCacheAffinityConfig {
        &self.config
    }
    /// The shared runtime, when affinity is enabled.
    #[must_use]
    pub fn runtime(&self) -> Option<Arc<dyn KVCacheRuntimeProtocol>> {
        self.runtime.clone()
    }
    /// Registers a root member and returns its cache identity.
    ///
    /// The identity is
    /// [`team_member_cache_identity`]:
    /// the persisted cache id, an empty prefix hash, zero counts and the team
    /// session as parent. Registration is idempotent — registering the same key
    /// twice returns the stored identity and keeps one entry, so a forked
    /// lineage is never flattened by a re-registration.
    ///
    /// A disabled registry returns the identity without storing anything.
    pub fn register_member(&self, key: MemberCacheKey, model: &str) -> KVCacheIdentity {
        let identity = team_member_cache_identity(
            &key.team_session_id,
            &key.team_name,
            &key.member_name,
            model,
        );
        if !self.is_enabled() || self.runtime.is_none() {
            return identity;
        }
        let mut entries = self.entries.lock();
        let member_name = key.member_name.clone();
        if let Some(existing) = entries.get(&member_name) {
            if existing.key == key {
                return existing.identity.clone();
            }
        }
        entries.insert(member_name, MemberCacheEntry::new(key, identity.clone()));
        identity
    }
    /// Registers `key` as a fork of `parent_member` and returns its identity.
    ///
    /// The child session id is `"{parent_cache_id}:fork:{child_member}"`, its
    /// parent session id is the parent member's cache id, and it keeps the same
    /// model with an empty prefix hash. Like the root path, re-registering the
    /// same key returns the stored identity.
    ///
    /// # Errors
    ///
    /// Returns [`Error::not_found`] when affinity is disabled or
    /// `parent_member` is not registered, and [`Error::validation`] when the
    /// child member name is blank.
    pub fn register_child(
        &self,
        parent_member: &str,
        key: MemberCacheKey,
        model: &str,
    ) -> Result<KVCacheIdentity> {
        if !self.is_enabled() || self.runtime.is_none() {
            return Err(Error::not_found(format!(
                "kv-cache affinity is disabled; cannot fork member '{parent_member}'"
            )));
        }
        if key.member_name.trim().is_empty() {
            return Err(Error::validation(
                "a forked team member needs a non-empty member name",
            ));
        }
        let parent = self.entries.lock().get(parent_member).cloned();
        let Some(parent) = parent else {
            return Err(Error::not_found(format!(
                "unknown kv-cache parent member '{parent_member}'"
            )));
        };
        let parent_cache_id = parent.identity.session_id;
        let mut identity = KVCacheIdentity::new(
            format!(
                "{parent_cache_id}:{MEMBER_FORK_SEGMENT}:{}",
                key.member_name.trim()
            ),
            String::new(),
            model.trim(),
        );
        identity.parent_session_id = Some(parent_cache_id);
        let mut entries = self.entries.lock();
        let member_name = key.member_name.trim().to_owned();
        if let Some(existing) = entries.get(&member_name) {
            if existing.key == key {
                return Ok(existing.identity.clone());
            }
        }
        entries.insert(
            member_name,
            MemberCacheEntry::new(key, identity.clone()).with_parent_member(parent_member),
        );
        Ok(identity)
    }
    /// The cache identity registered for `member_name`.
    #[must_use]
    pub fn identity(&self, member_name: &str) -> Option<KVCacheIdentity> {
        self.entries
            .lock()
            .get(member_name)
            .map(|entry| entry.identity.clone())
    }
    /// The affinity session id (cache id) of `member_name`.
    ///
    /// This is the value a session must carry as `kv_cache_affinity_session_id`;
    /// it equals [`Self::identity`]`(member_name).session_id`.
    #[must_use]
    pub fn affinity_session_id(&self, member_name: &str) -> Option<String> {
        self.entries
            .lock()
            .get(member_name)
            .map(|entry| entry.identity.session_id.clone())
    }
    /// The full registry entry of `member_name`.
    #[must_use]
    pub fn entry(&self, member_name: &str) -> Option<MemberCacheEntry> {
        self.entries.lock().get(member_name).cloned()
    }
    /// Registered member names, sorted.
    #[must_use]
    pub fn members(&self) -> Vec<String> {
        self.entries.lock().keys().cloned().collect()
    }
    /// Number of registered members.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.lock().len()
    }
    /// Whether no member is registered (always `true` when disabled).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.lock().is_empty()
    }
    /// The lineage env bag of `member_name`, or `None` when it is not registered.
    ///
    /// The bag carries exactly the two persisted keys: the member cache id under
    /// [`KV_CACHE_AFFINITY_SESSION_ID_ENV`] (`kv_cache_affinity_session_id`) and
    /// its lineage under [`KV_CACHE_AFFINITY_PARENT_SESSION_ID_ENV`]
    /// (`kv_cache_affinity_parent_session_id`). The parent key is omitted only
    /// for an identity without lineage, which a team member always has.
    #[must_use]
    pub fn env(&self, member_name: &str) -> Option<BTreeMap<String, String>> {
        let entry = self.entry(member_name)?;
        let mut env = BTreeMap::new();
        env.insert(
            KV_CACHE_AFFINITY_SESSION_ID_ENV.to_owned(),
            entry.identity.session_id.clone(),
        );
        if let Some(parent) = entry.identity.parent() {
            env.insert(
                KV_CACHE_AFFINITY_PARENT_SESSION_ID_ENV.to_owned(),
                parent.to_owned(),
            );
        }
        Some(env)
    }
    /// The child-session kwargs for `member_name`, or `None` when unregistered.
    ///
    /// Delegates to
    /// [`KVCacheModelCallHook::build_child_session_kwargs`](openjiuwen_agent::kv_cache::KVCacheModelCallHook::build_child_session_kwargs),
    /// so the shape is `{"parent_session_id": <cache id>, "envs":
    /// {"kv_cache_affinity_parent_session_id": <cache id>}}` — a session spawned
    /// for this member branches from the member's cache.
    #[must_use]
    pub fn child_session_kwargs(&self, member_name: &str) -> Option<Value> {
        let cache_id = self.affinity_session_id(member_name)?;
        Some(KVCacheModelCallHook::build_child_session_kwargs(&cache_id))
    }
    /// A model-call hook sharing this registry's runtime and configuration.
    ///
    /// `None` when affinity is disabled: a host must not schedule cache work.
    #[must_use]
    pub fn model_call_hook(&self) -> Option<KVCacheModelCallHook> {
        if !self.is_enabled() {
            return None;
        }
        let runtime = self.runtime.clone()?;
        Some(KVCacheModelCallHook::new(runtime, self.config.clone()))
    }
    /// Releases the cache of `member_name` and returns the evicted entry count.
    ///
    /// The registry rows of the member and of every member forked from it are
    /// dropped first, then the runtime is asked to evict each cache id — its
    /// `evict_session` also cascades to anything parented at those ids. Registry
    /// mutation happens under the lock and the eviction `await`s happen after it
    /// is released. An unknown member (or a disabled registry) evicts nothing.
    pub async fn release(&self, member_name: &str) -> usize {
        let targets = self.take_targets(member_name);
        if targets.is_empty() {
            return 0;
        }
        let Some(runtime) = self.runtime.clone() else {
            return 0;
        };
        let mut evicted: usize = 0;
        for session_id in targets {
            match runtime.evict_session(&session_id).await {
                Ok(count) => evicted = evicted.saturating_add(count),
                Err(error) => warn!(
                    member = %member_name,
                    session_id = %session_id,
                    error = %error,
                    "kv-cache release failed"
                ),
            }
        }
        evicted
    }
    /// Releases every member registered under `team_session_id`.
    ///
    /// Members of other teams sharing this registry are left untouched. Returns
    /// the total evicted entry count.
    pub async fn release_team(&self, team_session_id: &str) -> usize {
        let names: Vec<String> = {
            let entries = self.entries.lock();
            entries
                .iter()
                .filter(|(_, entry)| entry.key.team_session_id == team_session_id)
                .map(|(name, _)| name.clone())
                .collect()
        };
        let mut evicted: usize = 0;
        for name in names {
            evicted = evicted.saturating_add(self.release(&name).await);
        }
        evicted
    }
    /// Drops every registry row without evicting the runtime.
    ///
    /// This is the sync teardown for a host that is about to drop the whole
    /// runtime (or shares it and must not evict other teams' entries). Use
    /// [`Self::release`] or [`Self::release_team`] to evict provider-side cache.
    pub fn clear(&self) {
        self.entries.lock().clear();
    }
    /// Removes `member_name` and its forks from the registry, returning the
    /// cache ids to evict.
    ///
    /// Fork detection walks `parent_member` so grandchildren are covered; the
    /// runtime's own `parent_session_id` cascade on the intermediate id makes a
    /// second eviction pass unnecessary (and it is counted as `0`).
    fn take_targets(&self, member_name: &str) -> Vec<String> {
        let mut entries = self.entries.lock();
        let Some(root) = entries.get(member_name) else {
            return Vec::new();
        };
        let mut targets = vec![root.identity.session_id.clone()];
        let mut released: BTreeSet<String> = BTreeSet::from([member_name.to_owned()]);
        loop {
            let forked: Vec<(String, String)> = entries
                .iter()
                .filter(|(name, entry)| {
                    !released.contains(name.as_str())
                        && entry
                            .parent_member
                            .as_deref()
                            .is_some_and(|parent| released.contains(parent))
                })
                .map(|(name, entry)| (name.clone(), entry.identity.session_id.clone()))
                .collect();
            if forked.is_empty() {
                break;
            }
            for (name, session_id) in forked {
                released.insert(name);
                targets.push(session_id);
            }
        }
        for name in &released {
            entries.remove(name);
        }
        targets
    }
}
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use openjiuwen_agent::kv_cache::KVCacheEntry;
    use openjiuwen_common::error::ErrorKind;
    use openjiuwen_foundation::BaseMessage;
    use serde_json::json;
    fn enabled() -> KVCacheAffinityConfig {
        KVCacheAffinityConfig::enabled().with_max_entries(256)
    }
    fn key(team_session_id: &str, member: &str) -> MemberCacheKey {
        MemberCacheKey::new("team-1", team_session_id, member)
    }
    fn registered(registry: &TeamKvCacheAffinity, session: &str, member: &str) -> KVCacheIdentity {
        registry.register_member(key(session, member), "gpt-test")
    }
    async fn record(runtime: &Arc<InProcessKVCacheRuntime>, identity: KVCacheIdentity) {
        runtime
            .record(KVCacheEntry::new(identity, 1))
            .await
            .unwrap();
    }
    #[test]
    fn cache_key_serializes_with_snake_case_and_tolerates_unknown_fields() {
        let key = MemberCacheKey::new("team-1", "team-session", "dev-1");
        let value = serde_json::to_value(&key).unwrap();
        assert_eq!(
            value,
            json!({
                "team_name": "team-1",
                "team_session_id": "team-session",
                "member_name": "dev-1",
            })
        );
        let parsed: MemberCacheKey = serde_json::from_value(json!({
            "team_name": "team-1",
            "team_session_id": "team-session",
            "member_name": "dev-1",
            "future_field": 7,
        }))
        .unwrap();
        assert_eq!(parsed, key);
        assert_eq!(
            serde_json::from_value::<MemberCacheKey>(json!({})).unwrap(),
            MemberCacheKey::default()
        );
    }
    #[test]
    fn cache_id_keeps_the_persisted_team_member_format() {
        assert_eq!(
            key("team-session", "dev-1").cache_id(),
            "team:team-session:team:team-1:member:dev-1"
        );
    }
    #[test]
    fn entry_round_trips_through_json() {
        let identity = KVCacheIdentity::new("cache-id", "", "gpt-test").with_parent("team-session");
        let entry = MemberCacheEntry::new(key("team-session", "dev-1"), identity)
            .with_parent_member("team_leader");
        let value = serde_json::to_value(&entry).unwrap();
        assert_eq!(value["parent_member"], json!("team_leader"));
        assert_eq!(value["identity"]["session_id"], json!("cache-id"));
        let parsed: MemberCacheEntry = serde_json::from_value(value).unwrap();
        assert_eq!(parsed, entry);
    }
    #[tokio::test]
    async fn registering_a_member_is_idempotent() {
        let registry = TeamKvCacheAffinity::new(enabled());
        assert!(registry.is_enabled());
        let first = registered(&registry, "team-session", "dev-1");
        let second = registered(&registry, "team-session", "dev-1");
        assert_eq!(first, second);
        assert_eq!(
            first.session_id,
            "team:team-session:team:team-1:member:dev-1"
        );
        assert_eq!(registry.len(), 1);
        assert_eq!(registry.members(), vec!["dev-1".to_owned()]);
    }
    #[tokio::test]
    async fn member_identity_starts_empty_and_parents_at_the_team_session() {
        let registry = TeamKvCacheAffinity::new(enabled());
        let identity = registered(&registry, "team-session", "dev-1");
        assert_eq!(identity.prefix_hash, "");
        assert_eq!(identity.message_count, 0);
        assert_eq!(identity.tool_count, 0);
        assert_eq!(identity.model, "gpt-test");
        assert_eq!(identity.parent(), Some("team-session"));
        assert!(!identity.is_root());
    }
    #[tokio::test]
    async fn register_child_derives_from_the_parent_cache_id() {
        let registry = TeamKvCacheAffinity::new(enabled());
        let parent = registered(&registry, "team-session", "team_leader");
        let child = registry
            .register_child("team_leader", key("team-session", "dev-1"), "gpt-test")
            .unwrap();
        assert_eq!(
            child.session_id,
            format!("{}:fork:dev-1", parent.session_id)
        );
        assert_eq!(child.parent(), Some(parent.session_id.as_str()));
        assert_eq!(child.prefix_hash, "");
        assert_eq!(child.model, "gpt-test");
        let entry = registry.entry("dev-1").unwrap();
        assert_eq!(entry.parent_member.as_deref(), Some("team_leader"));
        assert_eq!(
            registry.members(),
            vec!["dev-1".to_owned(), "team_leader".to_owned()]
        );
    }
    #[tokio::test]
    async fn register_child_of_an_unknown_parent_is_not_found() {
        let registry = TeamKvCacheAffinity::new(enabled());
        let error = registry
            .register_child("nobody", key("team-session", "dev-1"), "gpt-test")
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::NotFound);
        assert!(registry.is_empty());
    }
    #[tokio::test]
    async fn register_child_rejects_a_blank_member_name() {
        let registry = TeamKvCacheAffinity::new(enabled());
        registered(&registry, "team-session", "team_leader");
        let error = registry
            .register_child("team_leader", key("team-session", "  "), "gpt-test")
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
    }
    #[tokio::test]
    async fn register_child_is_idempotent() {
        let registry = TeamKvCacheAffinity::new(enabled());
        registered(&registry, "team-session", "team_leader");
        let first = registry
            .register_child("team_leader", key("team-session", "dev-1"), "gpt-test")
            .unwrap();
        let second = registry
            .register_child("team_leader", key("team-session", "dev-1"), "gpt-test")
            .unwrap();
        assert_eq!(first, second);
        assert_eq!(registry.len(), 2);
    }
    #[tokio::test]
    async fn env_carries_the_two_persisted_keys() {
        let registry = TeamKvCacheAffinity::new(enabled());
        let identity = registered(&registry, "team-session", "dev-1");
        let env = registry.env("dev-1").unwrap();
        assert_eq!(env.len(), 2);
        assert_eq!(
            env.get(KV_CACHE_AFFINITY_SESSION_ID_ENV)
                .map(String::as_str),
            Some(identity.session_id.as_str())
        );
        assert_eq!(
            env.get(KV_CACHE_AFFINITY_PARENT_SESSION_ID_ENV)
                .map(String::as_str),
            Some("team-session")
        );
        assert!(registry.env("nobody").is_none());
    }
    #[tokio::test]
    async fn child_session_kwargs_render_the_python_shape() {
        let registry = TeamKvCacheAffinity::new(enabled());
        let identity = registered(&registry, "team-session", "dev-1");
        let kwargs = registry.child_session_kwargs("dev-1").unwrap();
        assert_eq!(
            kwargs,
            json!({
                "parent_session_id": identity.session_id,
                "envs": { KV_CACHE_AFFINITY_PARENT_SESSION_ID_ENV: identity.session_id },
            })
        );
        assert!(registry.child_session_kwargs("nobody").is_none());
    }
    #[tokio::test]
    async fn affinity_session_id_and_entry_expose_the_registration() {
        let registry = TeamKvCacheAffinity::new(enabled());
        let identity = registered(&registry, "team-session", "dev-1");
        assert_eq!(
            registry.affinity_session_id("dev-1").as_deref(),
            Some(identity.session_id.as_str())
        );
        let entry = registry.entry("dev-1").unwrap();
        assert_eq!(entry.key, key("team-session", "dev-1"));
        assert!(entry.parent_member.is_none());
        assert!(registry.identity("nobody").is_none());
        assert!(registry.affinity_session_id("nobody").is_none());
        assert!(registry.entry("nobody").is_none());
    }
    #[tokio::test]
    async fn disabled_registry_stores_nothing_and_evicts_nothing() {
        let registry = TeamKvCacheAffinity::disabled();
        assert!(!registry.is_enabled());
        assert!(registry.runtime().is_none());
        let identity = registered(&registry, "team-session", "dev-1");
        assert_eq!(
            identity.session_id,
            "team:team-session:team:team-1:member:dev-1"
        );
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
        assert_eq!(registry.members(), Vec::<String>::new());
        assert!(registry.identity("dev-1").is_none());
        assert!(registry.env("dev-1").is_none());
        assert!(registry.child_session_kwargs("dev-1").is_none());
        assert!(registry.model_call_hook().is_none());
        assert_eq!(registry.release("dev-1").await, 0);
        assert_eq!(registry.release_team("team-session").await, 0);
        let error = registry
            .register_child("dev-1", key("team-session", "dev-2"), "gpt-test")
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::NotFound);
    }
    #[tokio::test]
    async fn disabled_registry_ignores_an_injected_runtime() {
        let runtime = Arc::new(InProcessKVCacheRuntime::new(8));
        let registry = TeamKvCacheAffinity::disabled().with_runtime(runtime.clone());
        registered(&registry, "team-session", "dev-1");
        assert!(registry.is_empty());
        assert!(runtime.is_empty());
    }
    #[tokio::test]
    async fn registration_builds_an_in_process_runtime_at_max_entries() {
        let config = KVCacheAffinityConfig::enabled().with_max_entries(4);
        let registry = TeamKvCacheAffinity::new(config.clone());
        assert_eq!(registry.config(), &config);
        assert!(registry.runtime().is_some());
    }
    #[tokio::test]
    async fn release_evicts_the_member_and_its_forks() {
        let runtime = Arc::new(InProcessKVCacheRuntime::new(16));
        let registry = TeamKvCacheAffinity::new(enabled()).with_runtime(runtime.clone());
        let parent = registered(&registry, "team-session", "team_leader");
        let child = registry
            .register_child("team_leader", key("team-session", "dev-1"), "gpt-test")
            .unwrap();
        record(&runtime, parent).await;
        record(&runtime, child).await;
        assert_eq!(runtime.len(), 2);
        assert_eq!(registry.release("team_leader").await, 2);
        assert!(registry.is_empty());
        assert!(runtime.is_empty());
    }
    #[tokio::test]
    async fn release_keeps_unrelated_members() {
        let runtime = Arc::new(InProcessKVCacheRuntime::new(16));
        let registry = TeamKvCacheAffinity::new(enabled()).with_runtime(runtime.clone());
        let first = registered(&registry, "team-session", "dev-1");
        let second = registered(&registry, "team-session", "dev-2");
        record(&runtime, first).await;
        record(&runtime, second).await;
        assert_eq!(registry.release("dev-1").await, 1);
        assert_eq!(registry.members(), vec!["dev-2".to_owned()]);
        assert_eq!(runtime.len(), 1);
        assert!(registry.identity("dev-2").is_some());
    }
    #[tokio::test]
    async fn release_of_an_unknown_member_is_a_noop() {
        let registry = TeamKvCacheAffinity::new(enabled());
        assert_eq!(registry.release("nobody").await, 0);
        assert!(registry.is_empty());
    }
    #[tokio::test]
    async fn release_of_a_grandchild_keeps_the_parent() {
        let runtime = Arc::new(InProcessKVCacheRuntime::new(16));
        let registry = TeamKvCacheAffinity::new(enabled()).with_runtime(runtime.clone());
        let root = registered(&registry, "team-session", "team_leader");
        let child = registry
            .register_child("team_leader", key("team-session", "dev-1"), "gpt-test")
            .unwrap();
        let grandchild = registry
            .register_child("dev-1", key("team-session", "dev-2"), "gpt-test")
            .unwrap();
        record(&runtime, root).await;
        record(&runtime, child).await;
        record(&runtime, grandchild).await;
        assert_eq!(registry.release("dev-1").await, 2);
        assert_eq!(registry.members(), vec!["team_leader".to_owned()]);
        assert_eq!(runtime.len(), 1);
    }
    #[tokio::test]
    async fn release_team_only_drops_that_team() {
        let runtime = Arc::new(InProcessKVCacheRuntime::new(16));
        let registry = TeamKvCacheAffinity::new(enabled()).with_runtime(runtime.clone());
        let first = registered(&registry, "session-a", "dev-1");
        let second = registered(&registry, "session-b", "dev-2");
        record(&runtime, first).await;
        record(&runtime, second).await;
        assert_eq!(registry.release_team("session-a").await, 1);
        assert_eq!(registry.members(), vec!["dev-2".to_owned()]);
        assert_eq!(runtime.len(), 1);
        assert_eq!(registry.release_team("missing").await, 0);
    }
    #[tokio::test]
    async fn clear_drops_rows_without_touching_the_runtime() {
        let runtime = Arc::new(InProcessKVCacheRuntime::new(16));
        let registry = TeamKvCacheAffinity::new(enabled()).with_runtime(runtime.clone());
        let identity = registered(&registry, "team-session", "dev-1");
        record(&runtime, identity).await;
        registry.clear();
        assert!(registry.is_empty());
        assert_eq!(runtime.len(), 1);
    }
    #[tokio::test]
    async fn model_call_hook_records_and_reuses_a_member_prefix() {
        let registry = TeamKvCacheAffinity::new(enabled());
        let identity = registered(&registry, "team-session", "dev-1");
        let hook = registry.model_call_hook().unwrap();
        let messages = vec![
            BaseMessage::system("be helpful"),
            BaseMessage::user("hello"),
        ];
        let first = hook
            .affinity(
                &identity.session_id,
                identity.parent(),
                "gpt-test",
                &messages,
                &[],
            )
            .await
            .unwrap();
        assert!(!first.is_hit());
        let second = hook
            .affinity(
                &identity.session_id,
                identity.parent(),
                "gpt-test",
                &messages,
                &[],
            )
            .await
            .unwrap();
        assert!(second.is_hit());
        assert_eq!(second.reused_prefix_len, messages.len());
        assert_eq!(hook.stats().await.unwrap().hits, 1);
        assert_eq!(registry.release("dev-1").await, 1);
    }
    #[tokio::test]
    async fn concurrent_registration_and_release_leave_no_entries() {
        let runtime = Arc::new(InProcessKVCacheRuntime::new(256));
        let registry = Arc::new(TeamKvCacheAffinity::new(enabled()).with_runtime(runtime.clone()));
        let mut handles = Vec::new();
        for index in 0..32u32 {
            let registry = Arc::clone(&registry);
            let runtime = Arc::clone(&runtime);
            handles.push(tokio::spawn(async move {
                let identity = registry
                    .register_member(key("team-session", &format!("dev-{index}")), "gpt-test");
                runtime
                    .record(KVCacheEntry::new(identity, 1))
                    .await
                    .unwrap();
            }));
        }
        for handle in handles {
            handle.await.unwrap();
        }
        assert_eq!(registry.len(), 32);
        assert_eq!(runtime.len(), 32);
        let mut handles = Vec::new();
        for index in 0..32u32 {
            let registry = Arc::clone(&registry);
            handles.push(tokio::spawn(async move {
                registry.release(&format!("dev-{index}")).await
            }));
        }
        let mut evicted = 0;
        for handle in handles {
            evicted += handle.await.unwrap();
        }
        assert_eq!(evicted, 32);
        assert!(registry.is_empty());
        assert!(runtime.is_empty());
        assert_eq!(runtime.stats().await.unwrap().entries, 0);
    }
    #[tokio::test]
    async fn debug_lists_registered_members_without_exposing_the_runtime() {
        let registry = TeamKvCacheAffinity::new(enabled());
        registered(&registry, "team-session", "dev-1");
        let rendered = format!("{registry:?}");
        assert!(rendered.contains("has_runtime: true"));
        assert!(rendered.contains("dev-1"));
        assert!(!rendered.contains("InProcessKVCacheRuntime"));
    }
}
