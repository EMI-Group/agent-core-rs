//! Persisted scope↔user mapping (`scope_user_mapping/…` keys).
//!
//! Python's `core.memory` keeps a `scope_user_mapping/{scope_id}/{user_id}` key-value
//! namespace recording which users belong to a scope. This module ports that layout on
//! top of the [`KvStore`] trait: every row is the JSON `{"user_id": …, "scope_id": …}`
//! stored as UTF-8 text, and [`ScopeUserMappingStore`] is the only writer/reader of it.
//!
//! Key derivation always goes through [`super::keys::scope_user_mapping_key_in`] and
//! [`super::keys::scope_user_mapping_prefix_in`] with the store's
//! [`KvPrefixRegistry`], so renaming the `scope_user_mapping` family takes effect for
//! every subsequent read, write, and scan. Unlike the record and variable families
//! there is no alias fallback here (Python ships no legacy prefix for this family):
//! the current prefix is authoritative.
//!
//! Rules:
//!
//! - Both ids are validated before any store call: non-empty, at most 128 characters,
//!   no `/`, `\` or NUL, and no leading or trailing whitespace; a bad id is
//!   [`openjiuwen_common::error::Error::Validation`] naming the field and the value.
//! - A missing row is never an error: `get` is `Ok(None)`, `delete` is `Ok(false)`,
//!   `delete_scope`/`delete_user` are `Ok(0)`, and list-style reads are empty vectors.
//! - Decoding is tolerant: a malformed or non-text value is logged with
//!   `tracing::warn!` (target `openjiuwen_memory::store`) and skipped by list-style
//!   reads, so one corrupt row never fails a listing; `get` reports `Ok(None)` for it.
//! - Results are sorted (`BTreeSet`/`BTreeMap` based) and no wall clock is read anywhere.
//! - `add` is idempotent and uses the store's single conditional write
//!   (`exclusive_set`), so concurrent adds of the same pair cannot both report `true`.
//!
//! There is no reverse index: [`ScopeUserMappingStore::scopes_for_user`] scans the
//! family prefix and filters, exactly as documented for the Python keyspace.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::store::{KvStore, KvValue};
use serde::{Deserialize, Serialize};

use super::keys;
use super::prefix::{KvFamily, KvPrefixRegistry};
use super::MemoryStores;

/// Maximum accepted length of one mapping id, in characters.
const MAX_ID_CHARS: usize = 128;

/// One persisted scope→user link.
///
/// The JSON form is exactly `{"user_id": …, "scope_id": …}`, stored as text under
/// `<current scope_user_mapping prefix>/{scope_id}/{user_id}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScopeUserMapping {
    /// User that belongs to `scope_id`.
    pub user_id: String,
    /// Scope (workspace, tenant, or project) the user belongs to.
    pub scope_id: String,
}

/// KV-backed store for the `scope_user_mapping/{scope_id}/{user_id}` namespace.
///
/// The store is a plain value (there is no process-global registry) that owns the
/// key-value backend plus the prefix layout used for key derivation; it is cheap to
/// clone because both fields are cheaply shareable.
#[derive(Clone)]
pub struct ScopeUserMappingStore {
    kv: Arc<dyn KvStore>,
    prefixes: KvPrefixRegistry,
}

impl ScopeUserMappingStore {
    /// Creates a store over `kv` with the default key-prefix layout.
    #[must_use]
    pub fn new(kv: Arc<dyn KvStore>) -> Self {
        Self {
            kv,
            prefixes: KvPrefixRegistry::default(),
        }
    }

    /// Creates a store over the key-value backend of `stores` (default layout).
    #[must_use]
    pub fn from_stores(stores: &MemoryStores) -> Self {
        Self::new(Arc::clone(&stores.kv))
    }

    /// Returns the store with an explicit prefix layout.
    #[must_use]
    pub fn with_prefixes(mut self, prefixes: KvPrefixRegistry) -> Self {
        self.prefixes = prefixes;
        self
    }

    /// The prefix layout used for key derivation.
    #[must_use]
    pub fn prefixes(&self) -> &KvPrefixRegistry {
        &self.prefixes
    }

    /// Records that `user_id` belongs to `scope_id`.
    ///
    /// Returns `true` when the row was inserted and `false` when it already existed, so
    /// the call is idempotent and never overwrites an existing row. The write is the
    /// store's conditional `exclusive_set`, so two concurrent `add`s of the same pair
    /// cannot both report `true`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an invalid id (checked before any store call)
    /// and propagates store errors unchanged.
    pub async fn add(&self, user_id: &str, scope_id: &str) -> Result<bool> {
        validate_id("user_id", user_id)?;
        validate_id("scope_id", scope_id)?;
        let key = keys::scope_user_mapping_key_in(&self.prefixes, scope_id, user_id);
        let mapping = ScopeUserMapping {
            user_id: user_id.to_owned(),
            scope_id: scope_id.to_owned(),
        };
        let json = serde_json::to_string(&mapping).map_err(|error| {
            Error::external(error).with_context("serializing a scope-user mapping")
        })?;
        let inserted = self
            .kv
            .exclusive_set(&key, KvValue::Text(json), None)
            .await?;
        if inserted {
            tracing::debug!(
                target: "openjiuwen_memory::store",
                key = %key,
                "added a scope-user mapping"
            );
        }
        Ok(inserted)
    }

    /// Whether a row for `user_id` in `scope_id` exists.
    ///
    /// This is key presence: a row holding a malformed value still counts as existing,
    /// even though [`ScopeUserMappingStore::get`] reports `Ok(None)` for it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an invalid id and propagates store errors.
    pub async fn contains(&self, user_id: &str, scope_id: &str) -> Result<bool> {
        validate_id("user_id", user_id)?;
        validate_id("scope_id", scope_id)?;
        let key = keys::scope_user_mapping_key_in(&self.prefixes, scope_id, user_id);
        self.kv.exists(&key).await
    }

    /// Reads the row for `user_id` in `scope_id`, or `None` when it is absent.
    ///
    /// A non-text or malformed stored value logs a warning and yields `Ok(None)`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an invalid id and propagates store errors.
    pub async fn get(&self, user_id: &str, scope_id: &str) -> Result<Option<ScopeUserMapping>> {
        validate_id("user_id", user_id)?;
        validate_id("scope_id", scope_id)?;
        let key = keys::scope_user_mapping_key_in(&self.prefixes, scope_id, user_id);
        match self.kv.get(&key).await? {
            Some(value) => Ok(Self::decode(&key, value)),
            None => Ok(None),
        }
    }

    /// The users linked to `scope_id`, sorted and de-duplicated.
    ///
    /// Scans the `scope_id` prefix and keeps the rows whose stored `scope_id` matches,
    /// so a corrupt row belonging to another scope is never reported here.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an invalid `scope_id` and propagates store errors.
    pub async fn users_for_scope(&self, scope_id: &str) -> Result<Vec<String>> {
        validate_id("scope_id", scope_id)?;
        let prefix = keys::scope_user_mapping_prefix_in(&self.prefixes, scope_id);
        let users: BTreeSet<String> = self
            .entries_under(&prefix)
            .await?
            .into_iter()
            .filter(|mapping| mapping.scope_id == scope_id)
            .map(|mapping| mapping.user_id)
            .collect();
        Ok(users.into_iter().collect())
    }

    /// The scopes `user_id` is linked to, sorted and de-duplicated.
    ///
    /// The family prefix is scanned and filtered, because the key layout is ordered by
    /// scope first and there is no user index.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an invalid `user_id` and propagates store errors.
    pub async fn scopes_for_user(&self, user_id: &str) -> Result<Vec<String>> {
        validate_id("user_id", user_id)?;
        let scopes: BTreeSet<String> = self
            .entries_under(&self.family_prefix())
            .await?
            .into_iter()
            .filter(|mapping| mapping.user_id == user_id)
            .map(|mapping| mapping.scope_id)
            .collect();
        Ok(scopes.into_iter().collect())
    }

    /// Every row, sorted by `(scope_id, user_id)`.
    ///
    /// Malformed and non-text rows are skipped, so this is also the count behind
    /// [`ScopeUserMappingStore::len`].
    ///
    /// # Errors
    ///
    /// Propagates store errors.
    pub async fn list(&self) -> Result<Vec<ScopeUserMapping>> {
        let mut mappings: BTreeMap<(String, String), ScopeUserMapping> = BTreeMap::new();
        for mapping in self.entries_under(&self.family_prefix()).await? {
            mappings.insert((mapping.scope_id.clone(), mapping.user_id.clone()), mapping);
        }
        Ok(mappings.into_values().collect())
    }

    /// Number of well-formed rows.
    ///
    /// # Errors
    ///
    /// Propagates store errors.
    pub async fn len(&self) -> Result<usize> {
        Ok(self.list().await?.len())
    }

    /// Whether no well-formed row exists.
    ///
    /// # Errors
    ///
    /// Propagates store errors.
    pub async fn is_empty(&self) -> Result<bool> {
        Ok(self.len().await? == 0)
    }

    /// Removes the row for `user_id` in `scope_id`.
    ///
    /// Returns whether the row existed; removing a missing row is `Ok(false)`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an invalid id and propagates store errors.
    pub async fn delete(&self, user_id: &str, scope_id: &str) -> Result<bool> {
        validate_id("user_id", user_id)?;
        validate_id("scope_id", scope_id)?;
        let key = keys::scope_user_mapping_key_in(&self.prefixes, scope_id, user_id);
        if !self.kv.exists(&key).await? {
            return Ok(false);
        }
        self.kv.delete(&key).await?;
        tracing::debug!(
            target: "openjiuwen_memory::store",
            key = %key,
            "deleted a scope-user mapping"
        );
        Ok(true)
    }

    /// Removes every user of `scope_id`, returning how many live rows were deleted.
    ///
    /// The scope prefix is deleted as a whole, so malformed rows of that scope are
    /// removed too; other scopes (including prefixes that merely start with the same
    /// characters) are untouched because the prefix ends in `/`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an invalid `scope_id` and propagates store errors.
    pub async fn delete_scope(&self, scope_id: &str) -> Result<usize> {
        validate_id("scope_id", scope_id)?;
        let prefix = keys::scope_user_mapping_prefix_in(&self.prefixes, scope_id);
        let removed = self.kv.delete_by_prefix(&prefix).await?;
        if removed > 0 {
            tracing::debug!(
                target: "openjiuwen_memory::store",
                prefix = %prefix,
                removed,
                "deleted the scope-user mappings of a scope"
            );
        }
        Ok(removed)
    }

    /// Removes every scope of `user_id`, returning how many live rows were deleted.
    ///
    /// The family prefix is scanned and the keys whose final segment is `user_id` are
    /// deleted in one batch; malformed values are still removed, because the match is
    /// on the key layout.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an invalid `user_id` and propagates store errors.
    pub async fn delete_user(&self, user_id: &str) -> Result<usize> {
        validate_id("user_id", user_id)?;
        let stored = self.kv.get_by_prefix(&self.family_prefix()).await?;
        let keys: Vec<String> = stored
            .into_keys()
            .filter(|key| last_segment(key) == Some(user_id))
            .collect();
        if keys.is_empty() {
            return Ok(0);
        }
        let removed = self.kv.batch_delete(&keys).await?;
        if removed > 0 {
            tracing::debug!(
                target: "openjiuwen_memory::store",
                user_id,
                removed,
                "deleted the scope-user mappings of a user"
            );
        }
        Ok(removed)
    }

    /// The current prefix of the `scope_user_mapping` family, with its `/` separator.
    fn family_prefix(&self) -> String {
        format!("{}/", self.prefixes.current(KvFamily::ScopeUserMapping))
    }

    /// Decodes every well-formed row under `prefix`, warning and skipping bad rows.
    async fn entries_under(&self, prefix: &str) -> Result<Vec<ScopeUserMapping>> {
        let stored = self.kv.get_by_prefix(prefix).await?;
        let mut mappings = Vec::with_capacity(stored.len());
        for (key, value) in stored {
            if let Some(mapping) = Self::decode(&key, value) {
                mappings.push(mapping);
            }
        }
        Ok(mappings)
    }

    /// Decodes one stored value, logging and skipping a non-text or malformed row.
    fn decode(key: &str, value: KvValue) -> Option<ScopeUserMapping> {
        let Some(text) = value.as_str() else {
            tracing::warn!(
                target: "openjiuwen_memory::store",
                key = %key,
                "skipping a scope-user mapping that is not UTF-8 text"
            );
            return None;
        };
        match serde_json::from_str::<ScopeUserMapping>(text) {
            Ok(mapping) => Some(mapping),
            Err(error) => {
                tracing::warn!(
                    target: "openjiuwen_memory::store",
                    key = %key,
                    %error,
                    "skipping a malformed scope-user mapping"
                );
                None
            }
        }
    }
}

/// Validates one mapping id before any store call.
///
/// Ids are single storage-key segments: non-empty (and not whitespace-only), at most
/// [`MAX_ID_CHARS`] characters, free of `/`, `\` and NUL, and without leading or
/// trailing whitespace so a key cannot be confused with a neighbouring one.
fn validate_id(field: &str, value: &str) -> Result<()> {
    if value.trim().is_empty() {
        return Err(Error::validation(format!(
            "memory scope-user mapping `{field}` must not be empty (got {value:?})"
        )));
    }
    if value.trim() != value {
        return Err(Error::validation(format!(
            "memory scope-user mapping `{field}` must not have leading or trailing whitespace (got {value:?})"
        )));
    }
    if value.contains('\0') {
        return Err(Error::validation(format!(
            "memory scope-user mapping `{field}` must not contain NUL (got {value:?})"
        )));
    }
    if value.contains('/') {
        return Err(Error::validation(format!(
            "memory scope-user mapping `{field}` must not contain '/' (got {value:?})"
        )));
    }
    if value.contains('\\') {
        return Err(Error::validation(format!(
            "memory scope-user mapping `{field}` must not contain a backslash (got {value:?})"
        )));
    }
    let length = value.chars().count();
    if length > MAX_ID_CHARS {
        return Err(Error::validation(format!(
            "memory scope-user mapping `{field}` must be at most {MAX_ID_CHARS} characters (got {length})"
        )));
    }
    Ok(())
}

/// The final `/`-separated segment of `key`, or `None` when it is empty.
fn last_segment(key: &str) -> Option<&str> {
    key.rsplit('/').next().filter(|segment| !segment.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::tempdir;

    fn in_memory() -> (MemoryStores, ScopeUserMappingStore) {
        let stores = MemoryStores::in_memory();
        let store = ScopeUserMappingStore::from_stores(&stores);
        (stores, store)
    }

    fn mapping(user_id: &str, scope_id: &str) -> ScopeUserMapping {
        ScopeUserMapping {
            user_id: user_id.to_owned(),
            scope_id: scope_id.to_owned(),
        }
    }

    #[tokio::test]
    async fn add_get_contains_and_deletes_round_trip() {
        let (_stores, store) = in_memory();
        assert!(store.add("alice", "ws").await.expect("add alice/ws"));
        assert!(store.add("bob", "ws").await.expect("add bob/ws"));
        assert!(store.add("alice", "other").await.expect("add alice/other"));

        assert_eq!(
            store.get("alice", "ws").await.expect("get"),
            Some(mapping("alice", "ws"))
        );
        assert!(store.contains("bob", "ws").await.expect("contains"));
        assert!(!store.contains("bob", "other").await.expect("contains"));
        assert_eq!(store.len().await.expect("len"), 3);
        assert!(!store.is_empty().await.expect("is empty"));

        // A duplicate add is idempotent: it reports false and leaves the row untouched.
        assert!(!store.add("alice", "ws").await.expect("duplicate add"));
        assert_eq!(store.len().await.expect("len after duplicate"), 3);
        assert_eq!(
            store.get("alice", "ws").await.expect("get after duplicate"),
            Some(mapping("alice", "ws"))
        );

        assert_eq!(
            store.users_for_scope("ws").await.expect("users"),
            ["alice", "bob"]
        );
        assert_eq!(
            store.scopes_for_user("alice").await.expect("scopes"),
            ["other", "ws"]
        );
        assert_eq!(
            store.list().await.expect("list"),
            vec![
                mapping("alice", "other"),
                mapping("alice", "ws"),
                mapping("bob", "ws"),
            ]
        );

        assert!(store.delete("alice", "ws").await.expect("delete"));
        assert!(!store.delete("alice", "ws").await.expect("delete again"));

        // Only bob/ws is left in `ws`; the row of another scope survives.
        assert_eq!(store.delete_scope("ws").await.expect("delete scope"), 1);
        assert_eq!(
            store.users_for_scope("ws").await.expect("users"),
            Vec::<String>::new()
        );
        assert!(store.contains("alice", "other").await.expect("survivor"));

        assert_eq!(store.delete_user("alice").await.expect("delete user"), 1);
        assert!(store.is_empty().await.expect("empty"));
    }

    #[tokio::test]
    async fn add_writes_the_documented_key_and_json() {
        let (stores, store) = in_memory();
        assert!(store.add("alice", "ws").await.expect("add"));
        let raw = stores
            .kv
            .get("scope_user_mapping/ws/alice")
            .await
            .expect("read raw")
            .expect("key present");
        let text = raw.as_str().expect("text value");
        let value: serde_json::Value = serde_json::from_str(text).expect("valid JSON");
        assert_eq!(value, json!({"user_id": "alice", "scope_id": "ws"}));
    }

    #[tokio::test]
    async fn invalid_ids_are_validation_errors_and_write_nothing() {
        let (stores, store) = in_memory();
        let long = "x".repeat(MAX_ID_CHARS + 1);
        let bad = ["", "  ", "a/b", "a\\b", "a\0b", long.as_str()];
        for value in bad {
            let error = store.add(value, "ws").await.expect_err("bad user_id");
            assert!(matches!(error, Error::Validation(_)), "{error:?}");
            assert!(error.to_string().contains("user_id"), "{error}");

            let error = store.add("alice", value).await.expect_err("bad scope_id");
            assert!(matches!(error, Error::Validation(_)), "{error:?}");
            assert!(error.to_string().contains("scope_id"), "{error}");

            let error = store.get(value, "ws").await.expect_err("bad user_id read");
            assert!(matches!(error, Error::Validation(_)), "{error:?}");

            let error = store.delete("alice", value).await.expect_err("bad scope");
            assert!(matches!(error, Error::Validation(_)), "{error:?}");
        }
        assert_eq!(
            stores
                .kv
                .get_by_prefix("scope_user_mapping/")
                .await
                .expect("scan"),
            BTreeMap::new()
        );
    }

    #[tokio::test]
    async fn missing_rows_are_not_errors() {
        let (_stores, store) = in_memory();
        assert_eq!(store.get("alice", "ws").await.expect("get"), None);
        assert!(!store.delete("alice", "ws").await.expect("delete"));
        assert_eq!(store.delete_scope("ws").await.expect("delete scope"), 0);
        assert_eq!(store.delete_user("alice").await.expect("delete user"), 0);
        assert_eq!(
            store.list().await.expect("list"),
            Vec::<ScopeUserMapping>::new()
        );
        assert_eq!(store.len().await.expect("len"), 0);
        assert!(store.is_empty().await.expect("is empty"));
        assert_eq!(
            store.users_for_scope("ws").await.expect("users"),
            Vec::<String>::new()
        );
        assert_eq!(
            store.scopes_for_user("alice").await.expect("scopes"),
            Vec::<String>::new()
        );
    }

    #[tokio::test]
    async fn malformed_rows_are_skipped_not_fatal() {
        let (stores, store) = in_memory();
        stores
            .kv
            .set("scope_user_mapping/ws/alice", KvValue::text("not json"))
            .await
            .expect("seed malformed");
        stores
            .kv
            .set("scope_user_mapping/ws/bob", KvValue::bytes(vec![1, 2, 3]))
            .await
            .expect("seed bytes");
        assert!(store.add("carol", "ws").await.expect("add carol"));

        assert_eq!(store.get("alice", "ws").await.expect("get malformed"), None);
        assert!(store.contains("alice", "ws").await.expect("key presence"));
        assert_eq!(store.users_for_scope("ws").await.expect("users"), ["carol"]);
        assert_eq!(
            store.list().await.expect("list"),
            vec![mapping("carol", "ws")]
        );
        assert_eq!(store.len().await.expect("len"), 1);
        assert_eq!(
            store.scopes_for_user("alice").await.expect("scopes"),
            Vec::<String>::new()
        );

        // Malformed rows are removed by the key-layout deletes.
        assert_eq!(store.delete_user("alice").await.expect("delete user"), 1);
        assert_eq!(store.delete_scope("ws").await.expect("delete scope"), 2);
        assert!(store.is_empty().await.expect("empty"));
    }

    #[tokio::test]
    async fn a_renamed_prefix_takes_effect_for_reads_and_writes() {
        let (stores, default_store) = in_memory();
        assert!(default_store.add("alice", "ws").await.expect("add default"));

        let prefixes = KvPrefixRegistry::new()
            .with_current(KvFamily::ScopeUserMapping, "sum_v2")
            .expect("rename scope user mapping");
        let renamed = default_store.clone().with_prefixes(prefixes);
        assert_eq!(
            renamed.prefixes().current(KvFamily::ScopeUserMapping),
            "sum_v2"
        );
        assert!(renamed.add("bob", "ws").await.expect("add renamed"));

        assert!(stores
            .kv
            .exists("sum_v2/ws/bob")
            .await
            .expect("renamed key"));
        assert!(stores
            .kv
            .exists("scope_user_mapping/ws/alice")
            .await
            .expect("default key"));

        assert_eq!(
            renamed.users_for_scope("ws").await.expect("renamed users"),
            ["bob"]
        );
        assert_eq!(
            default_store
                .users_for_scope("ws")
                .await
                .expect("default users"),
            ["alice"]
        );
        assert_eq!(
            default_store.get("alice", "ws").await.expect("default get"),
            Some(mapping("alice", "ws"))
        );
        assert_eq!(renamed.get("alice", "ws").await.expect("renamed get"), None);
        assert!(renamed.delete("bob", "ws").await.expect("renamed delete"));
        assert!(stores
            .kv
            .exists("scope_user_mapping/ws/alice")
            .await
            .expect("default row survives"));
    }

    #[tokio::test]
    async fn rows_persist_across_a_local_file_reopen() {
        let dir = tempdir().expect("temp dir");
        {
            let stores = MemoryStores::local_file(dir.path()).await.expect("stores");
            let store = ScopeUserMappingStore::from_stores(&stores);
            assert!(store.add("alice", "ws").await.expect("add"));
        }
        let stores = MemoryStores::local_file(dir.path())
            .await
            .expect("reopen stores");
        let store = ScopeUserMappingStore::from_stores(&stores);
        assert_eq!(
            store.get("alice", "ws").await.expect("get"),
            Some(mapping("alice", "ws"))
        );
        assert_eq!(store.users_for_scope("ws").await.expect("users"), ["alice"]);
        assert_eq!(store.len().await.expect("len"), 1);
    }
}
