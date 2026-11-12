//! Record, embedding, variable, and scope-config persistence.
//!
//! Every operation is a thin, typed layer over [`MemoryStores`]: records live in the
//! KV store at the Python `UMD/...` keys, embeddings live in one vector collection per
//! `(user, scope, kind)` triple, variables are raw text KV entries, and scope
//! configuration is JSON under `memory_scope_config/{scope_id}`.
//!
//! No ranking, extraction, or engine policy lives here; the module only derives keys,
//! serializes values, and propagates store errors (a missing key/collection is never an
//! error, it is `None`/`false`/`0`/empty).
//!
//! Writes always use the current key layout (`keys`). Reads fall back to the legacy
//! layout (`keys::legacy_alternatives`) only when the current key is absent, and
//! deletes remove both variants; a legacy hit is logged at debug level and never
//! migrated implicitly. See the `keys` module docs for the family mapping.

use std::collections::{BTreeMap, BTreeSet};

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::store::{
    CollectionSchema, FieldSchema, KvValue, VectorDataType, VectorQuery, VectorStore,
};
use serde_json::{Map, Value};

use crate::config::MemoryScopeConfig;
use crate::record::{MemoryFilter, MemoryRecord, MemoryScope, MemoryType};

use super::json;
use super::keys;
use super::MemoryStores;

/// The `id` field of a vector collection.
const VECTOR_ID_FIELD: &str = "id";
/// The embedding field of a vector collection.
const VECTOR_FIELD: &str = "embedding";
/// Maximum length of the vector primary-key `Varchar` field.
const VECTOR_ID_MAX_LENGTH: usize = 256;

/// Typed persistence for every memory data family.
///
/// The struct owns the [`MemoryStores`] bundle by value (all fields are `Arc`s, so the
/// bundle is cheap to clone) and offers one method per storage operation. It is
/// crate-internal: callers use [`crate::facade::LongTermMemory`] or the engine.
pub(crate) struct MemoryStorage {
    /// The underlying KV, vector, document, and graph stores.
    stores: MemoryStores,
}

impl MemoryStorage {
    /// Builds storage over `stores`.
    pub(crate) fn new(stores: MemoryStores) -> Self {
        Self { stores }
    }

    /// Persists `record` under its scope/id key, without its embedding.
    pub(crate) async fn put_record(&self, record: &MemoryRecord) -> Result<()> {
        let text = json::record_to_text(record)?;
        let key = keys::record_key(&record.scope, &record.id);
        self.stores.kv.set(&key, KvValue::text(text)).await
    }

    /// Reads one record, or `None` when it does not exist.
    ///
    /// A stored value that is not UTF-8 text or not a valid record JSON is reported as
    /// `Error::Validation`: unlike a listing, a point read must not hide corruption.
    /// The current key wins; a record stored under the legacy layout is read only when
    /// the current key is absent.
    pub(crate) async fn get_record(
        &self,
        scope: &MemoryScope,
        id: &str,
    ) -> Result<Option<MemoryRecord>> {
        let key = keys::record_key(scope, id);
        let Some((found, text)) = self.get_text_with_legacy(key).await? else {
            return Ok(None);
        };
        json::record_from_text(&text)
            .map_err(|error| {
                Error::validation(format!(
                    "stored memory record `{found}` is not a valid record: {error}"
                ))
            })
            .map(Some)
    }

    /// Lists every record matching `filter`, newest first and by id ascending on ties.
    ///
    /// The scan uses the narrowest key prefix the filter allows. Malformed entries are
    /// logged and skipped: one corrupt record must not fail a whole listing. Records
    /// found only under the legacy layout are appended after the current-layout scan,
    /// and a record present in both layouts is reported once from the current one.
    pub(crate) async fn list_records(&self, filter: &MemoryFilter) -> Result<Vec<MemoryRecord>> {
        let prefix = match (&filter.user_id, &filter.scope_id) {
            (Some(user_id), Some(scope_id)) => {
                keys::record_prefix(&MemoryScope::new(user_id.clone(), scope_id.clone()))
            }
            (Some(user_id), None) => keys::record_prefix_for(user_id),
            (None, Some(scope_id)) => keys::record_prefix_for_scope(scope_id),
            (None, None) => keys::record_prefix_for_scope(""),
        };
        let mut records = self.scan_records(&prefix, filter).await?;
        let mut seen: BTreeSet<String> = records
            .iter()
            .map(|record| keys::record_key(&record.scope, &record.id))
            .collect();
        for legacy in keys::legacy_alternatives(&prefix) {
            for record in self.scan_records(&legacy, filter).await? {
                let current = keys::record_key(&record.scope, &record.id);
                if seen.insert(current.clone()) {
                    log_legacy_read(&current, &legacy);
                    records.push(record);
                }
            }
        }
        records.sort_by(|left, right| {
            right
                .timestamp
                .cmp(&left.timestamp)
                .then_with(|| left.id.cmp(&right.id))
        });
        Ok(records)
    }

    /// Scans `prefix`, returning the records that match `filter` (unsorted).
    ///
    /// Malformed or non-text entries are logged and skipped, never fatal.
    async fn scan_records(&self, prefix: &str, filter: &MemoryFilter) -> Result<Vec<MemoryRecord>> {
        let entries = self.stores.kv.get_by_prefix(prefix).await?;
        let mut records = Vec::with_capacity(entries.len());
        for (key, value) in entries {
            let Some(text) = value.as_str() else {
                tracing::warn!(
                    target: "openjiuwen_memory::store",
                    key = %key,
                    "skipping a stored memory record that is not UTF-8 text"
                );
                continue;
            };
            match json::record_from_text(text) {
                Ok(record) => {
                    if filter.matches(&record) {
                        records.push(record);
                    }
                }
                Err(error) => tracing::warn!(
                    target: "openjiuwen_memory::store",
                    key = %key,
                    %error,
                    "skipping a malformed stored memory record"
                ),
            }
        }
        Ok(records)
    }

    /// Deletes `record` and its embedding, returning whether the record row existed.
    ///
    /// Both the current and the legacy key are removed; a legacy row therefore reports
    /// `true` on its first delete and `false` afterwards.
    pub(crate) async fn delete_record(&self, record: &MemoryRecord) -> Result<bool> {
        let key = keys::record_key(&record.scope, &record.id);
        let existed = self.delete_with_legacy(&key).await?;
        self.delete_embedding(record).await?;
        Ok(existed)
    }

    /// Indexes `embedding` for `record` in its `(scope, kind)` collection.
    ///
    /// The collection is created on first use from the embedding's dimension. The
    /// foundation vector store is insert-only, so an existing entry for the record id is
    /// deleted before the new vector is added. A dimension mismatch against an existing
    /// collection propagates the foundation validation error unchanged.
    pub(crate) async fn index_embedding(
        &self,
        record: &MemoryRecord,
        embedding: &[f32],
    ) -> Result<()> {
        let collection = keys::vector_collection(&record.scope, record.kind);
        ensure_collection(self.stores.vector.as_ref(), &collection, embedding.len()).await?;
        let ids = std::slice::from_ref(&record.id);
        self.stores
            .vector
            .delete_docs_by_ids(&collection, ids)
            .await?;
        let vector = serde_json::to_value(embedding).map_err(|error| {
            Error::external(error).with_context(format!(
                "serializing the embedding of memory record `{}`",
                record.id
            ))
        })?;
        let mut document = Map::new();
        document.insert(VECTOR_ID_FIELD.to_owned(), Value::String(record.id.clone()));
        document.insert(VECTOR_FIELD.to_owned(), vector);
        self.stores
            .vector
            .add_docs(&collection, vec![document])
            .await?;
        Ok(())
    }

    /// Removes the embedding of `record`; a missing collection is a no-op.
    pub(crate) async fn delete_embedding(&self, record: &MemoryRecord) -> Result<()> {
        let collection = keys::vector_collection(&record.scope, record.kind);
        if !self.stores.vector.collection_exists(&collection).await? {
            return Ok(());
        }
        self.stores
            .vector
            .delete_docs_by_ids(&collection, std::slice::from_ref(&record.id))
            .await?;
        Ok(())
    }

    /// Searches the `(scope, kind)` collection for the nearest vectors to `vector`.
    ///
    /// Returns `(record_id, score)` pairs in the foundation's score order. A missing
    /// collection, or `top_k == 0`, yields an empty result rather than an error.
    pub(crate) async fn search_embeddings(
        &self,
        scope: &MemoryScope,
        kind: MemoryType,
        vector: &[f32],
        top_k: usize,
    ) -> Result<Vec<(String, f32)>> {
        if top_k == 0 {
            return Ok(Vec::new());
        }
        let collection = keys::vector_collection(scope, kind);
        if !self.stores.vector.collection_exists(&collection).await? {
            return Ok(Vec::new());
        }
        let query = VectorQuery::new(vector.to_vec()).with_top_k(top_k);
        let results = self.stores.vector.search(&collection, &query).await?;
        let mut hits = Vec::with_capacity(results.len());
        for result in results {
            if let Some(id) = result.fields.get(VECTOR_ID_FIELD).and_then(Value::as_str) {
                hits.push((id.to_owned(), result.score));
            } else {
                tracing::warn!(
                    target: "openjiuwen_memory::store",
                    collection = %collection,
                    score = result.score,
                    "skipping a vector search hit without a string id"
                );
            }
        }
        Ok(hits)
    }

    /// Stores `value` as the text variable `name` (user- or session-scoped).
    pub(crate) async fn set_variable(
        &self,
        scope: &MemoryScope,
        session_id: Option<&str>,
        name: &str,
        value: &str,
    ) -> Result<()> {
        let key = variable_key(scope, session_id, name);
        self.stores.kv.set(&key, KvValue::text(value)).await
    }

    /// Reads a text variable, or `None` when it is absent or not stored as text.
    ///
    /// The current key wins; the legacy key layout is consulted only when it is absent.
    pub(crate) async fn get_variable(
        &self,
        scope: &MemoryScope,
        session_id: Option<&str>,
        name: &str,
    ) -> Result<Option<String>> {
        let key = variable_key(scope, session_id, name);
        match self.get_text_with_legacy(key).await {
            Ok(found) => Ok(found.map(|(_, text)| text)),
            // A non-text value is "not a text variable", the documented contract here.
            Err(Error::Validation(_)) => Ok(None),
            Err(error) => Err(error),
        }
    }

    /// Lists every variable of the scope namespace, keyed by variable name.
    ///
    /// The current namespace is scanned first; names found only under the legacy layout
    /// are added afterwards, so a current-layout variable always wins on a name clash.
    pub(crate) async fn list_variables(
        &self,
        scope: &MemoryScope,
        session_id: Option<&str>,
    ) -> Result<BTreeMap<String, String>> {
        let prefix = variable_prefix(scope, session_id);
        let mut candidates = vec![(prefix.clone(), false)];
        candidates.extend(
            keys::legacy_alternatives(&prefix)
                .into_iter()
                .map(|legacy| (legacy, true)),
        );
        let mut variables = BTreeMap::new();
        for (candidate, from_legacy) in candidates {
            let entries = self.stores.kv.get_by_prefix(&candidate).await?;
            for (key, value) in entries {
                let Some(name) = key.strip_prefix(&candidate) else {
                    continue;
                };
                let Some(text) = value.as_str() else {
                    tracing::warn!(
                        target: "openjiuwen_memory::store",
                        key = %key,
                        "skipping a memory variable that is not UTF-8 text"
                    );
                    continue;
                };
                if from_legacy {
                    let current = keys::current_equivalent(&key).unwrap_or_else(|| prefix.clone());
                    log_legacy_read(&current, &key);
                    variables
                        .entry(name.to_owned())
                        .or_insert_with(|| text.to_owned());
                } else {
                    variables.insert(name.to_owned(), text.to_owned());
                }
            }
        }
        Ok(variables)
    }

    /// Deletes a variable, returning whether any of its keys existed.
    pub(crate) async fn delete_variable(
        &self,
        scope: &MemoryScope,
        session_id: Option<&str>,
        name: &str,
    ) -> Result<bool> {
        let key = variable_key(scope, session_id, name);
        self.delete_with_legacy(&key).await
    }

    /// Stores the configuration of `scope_id`, returning whether it replaced one.
    ///
    /// Writes only the current key: the legacy layout is read-only compatibility.
    pub(crate) async fn set_scope_config(
        &self,
        scope_id: &str,
        config: &MemoryScopeConfig,
    ) -> Result<bool> {
        let key = keys::scope_config_key(scope_id);
        let existed = self.stores.kv.exists(&key).await?;
        let text = serde_json::to_string(config).map_err(|error| {
            Error::external(error)
                .with_context(format!("serializing the memory scope config `{scope_id}`"))
        })?;
        self.stores.kv.set(&key, KvValue::text(text)).await?;
        Ok(existed)
    }

    /// Reads the configuration of `scope_id`, or `None` when it is absent.
    ///
    /// The current key wins; the legacy layout is consulted only when it is absent.
    /// Stored JSON that is not a valid [`MemoryScopeConfig`] is `Error::Validation`,
    /// never a silent default.
    pub(crate) async fn get_scope_config(
        &self,
        scope_id: &str,
    ) -> Result<Option<MemoryScopeConfig>> {
        let key = keys::scope_config_key(scope_id);
        let Some((found, text)) = self.get_text_with_legacy(key).await? else {
            return Ok(None);
        };
        let config = serde_json::from_str(&text).map_err(|error| {
            Error::validation(format!(
                "malformed memory scope config JSON for `{found}`: {error}"
            ))
        })?;
        Ok(Some(config))
    }

    /// Deletes the configuration of `scope_id`, returning whether it existed.
    ///
    /// Both the current and the legacy key are removed.
    pub(crate) async fn delete_scope_config(&self, scope_id: &str) -> Result<bool> {
        let key = keys::scope_config_key(scope_id);
        self.delete_with_legacy(&key).await
    }

    /// Reads the UTF-8 text stored under `key`, falling back to the legacy key layout.
    ///
    /// Returns the key that was actually found plus its text. The current key always
    /// wins: a legacy candidate is tried only when `key` is absent. A candidate holding
    /// a non-text value is `Error::Validation` naming that key, because a point read
    /// must not hide corruption behind a "missing" answer. `None` means no candidate
    /// key exists at all.
    async fn get_text_with_legacy(&self, key: String) -> Result<Option<(String, String)>> {
        let mut candidates = vec![key.clone()];
        candidates.extend(keys::legacy_alternatives(&key));
        for (position, candidate) in candidates.into_iter().enumerate() {
            let Some(value) = self.stores.kv.get(&candidate).await? else {
                continue;
            };
            let Some(text) = value.as_str() else {
                return Err(Error::validation(format!(
                    "stored memory value `{candidate}` is not UTF-8 text"
                )));
            };
            if position > 0 {
                log_legacy_read(&key, &candidate);
            }
            return Ok(Some((candidate, text.to_owned())));
        }
        Ok(None)
    }

    /// Deletes `key` and every legacy alternative, reporting whether any of them existed.
    async fn delete_with_legacy(&self, key: &str) -> Result<bool> {
        let mut candidates = vec![key.to_owned()];
        candidates.extend(keys::legacy_alternatives(key));
        let mut existed = false;
        for candidate in candidates {
            if self.stores.kv.exists(&candidate).await? {
                existed = true;
            }
            self.stores.kv.delete(&candidate).await?;
        }
        Ok(existed)
    }
}

/// Logs a read that was served from the legacy key layout.
fn log_legacy_read(current_key: &str, legacy_key: &str) {
    tracing::debug!(
        target: "openjiuwen_memory::store",
        current_key = %current_key,
        legacy_key = %legacy_key,
        "read from a legacy memory key"
    );
}

/// Creates the vector collection when it does not exist.
///
/// A failed `create_collection` is re-checked against `collection_exists`, so two
/// concurrent first writes do not turn one of them into an error.
async fn ensure_collection(vector: &dyn VectorStore, collection: &str, dim: usize) -> Result<()> {
    if vector.collection_exists(collection).await? {
        return Ok(());
    }
    let schema = CollectionSchema::new(vec![
        FieldSchema {
            name: VECTOR_ID_FIELD.to_owned(),
            dtype: VectorDataType::Varchar,
            is_primary: true,
            max_length: Some(VECTOR_ID_MAX_LENGTH),
            ..FieldSchema::default()
        },
        FieldSchema {
            name: VECTOR_FIELD.to_owned(),
            dtype: VectorDataType::FloatVector,
            dim: Some(dim),
            ..FieldSchema::default()
        },
    ]);
    if let Err(error) = vector.create_collection(collection, schema).await {
        if vector.collection_exists(collection).await? {
            return Ok(());
        }
        return Err(error.with_context(format!(
            "creating the memory vector collection `{collection}`"
        )));
    }
    Ok(())
}

/// The key of one variable, user-scoped when `session_id` is absent.
fn variable_key(scope: &MemoryScope, session_id: Option<&str>, name: &str) -> String {
    match session_id {
        Some(session_id) => keys::session_variable_key(scope, session_id, name),
        None => keys::variable_key(scope, name),
    }
}

/// The key prefix of one variable namespace, user-scoped when `session_id` is absent.
fn variable_prefix(scope: &MemoryScope, session_id: Option<&str>) -> String {
    match session_id {
        Some(session_id) => keys::session_variable_prefix(scope, session_id),
        None => keys::variable_prefix(scope),
    }
}

#[cfg(test)]
#[allow(clippy::assert_is_empty)] // `assert!(x.is_empty())` states the expectation directly
mod tests {
    use super::*;
    use chrono::{DateTime, TimeZone, Utc};
    use openjiuwen_common::error::ErrorKind;

    #[cfg(feature = "sqlite")]
    use crate::store::MessageStore;
    #[cfg(feature = "sqlite")]
    use openjiuwen_foundation::prompt::BaseMessage;

    fn storage() -> MemoryStorage {
        MemoryStorage::new(MemoryStores::in_memory())
    }

    fn scope() -> MemoryScope {
        MemoryScope::new("alice", "ws").session("s1")
    }

    fn timestamp(offset: i64) -> DateTime<Utc> {
        Utc.timestamp_opt(1_700_000_000 + offset, 0)
            .single()
            .expect("fixed time")
    }

    fn record(id: &str, content: &str, offset: i64) -> MemoryRecord {
        MemoryRecord::new(scope(), content, MemoryType::SemanticMemory)
            .with_id(id)
            .with_timestamp(timestamp(offset))
    }

    #[tokio::test]
    async fn records_round_trip_without_the_embedding() {
        let storage = storage();
        let record = record("rec-1", "Alice prefers dark mode", 0).with_embedding(vec![0.1, 0.2]);
        storage.put_record(&record).await.expect("put");
        let restored = storage
            .get_record(&scope(), "rec-1")
            .await
            .expect("get")
            .expect("record exists");
        assert_eq!(restored.id, "rec-1");
        assert_eq!(restored.content, record.content);
        assert_eq!(restored.kind, record.kind);
        assert_eq!(restored.scope, record.scope);
        assert_eq!(restored.embedding, None);
        let stored = storage
            .stores
            .kv
            .get(&keys::record_key(&scope(), "rec-1"))
            .await
            .expect("read key")
            .expect("key exists");
        let value: serde_json::Value =
            serde_json::from_str(stored.as_str().expect("text")).expect("valid JSON");
        assert_eq!(value.get("embedding"), None);
        assert_eq!(value["text"], serde_json::json!(record.content));
        assert!(storage
            .get_record(&scope(), "absent")
            .await
            .expect("get")
            .is_none());
    }

    #[tokio::test]
    async fn list_filters_and_orders_by_timestamp_then_id() {
        let storage = storage();
        for (id, content, offset) in [("a", "oldest", 0), ("b", "newest", 2), ("c", "middle", 1)] {
            storage
                .put_record(&record(id, content, offset))
                .await
                .expect("put");
        }
        storage
            .put_record(
                &MemoryRecord::new(
                    MemoryScope::new("alice", "other"),
                    "elsewhere",
                    MemoryType::SemanticMemory,
                )
                .with_id("d")
                .with_timestamp(timestamp(3)),
            )
            .await
            .expect("put other scope");
        let all = storage
            .list_records(&MemoryFilter::for_scope(&scope()))
            .await
            .expect("list scope");
        assert_eq!(
            all.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(),
            ["b", "c", "a"]
        );
        let mut by_id = MemoryFilter::for_scope(&scope());
        by_id.ids = vec!["a".to_owned(), "b".to_owned()];
        let filtered = storage.list_records(&by_id).await.expect("list ids");
        assert_eq!(
            filtered.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(),
            ["b", "a"]
        );
        let user_wide = storage
            .list_records(&MemoryFilter {
                user_id: Some("alice".to_owned()),
                ..MemoryFilter::default()
            })
            .await
            .expect("list user");
        assert_eq!(user_wide.len(), 4);
        let everything = storage
            .list_records(&MemoryFilter::new())
            .await
            .expect("list all");
        assert_eq!(everything.len(), 4);
        let kinds = storage
            .list_records(&MemoryFilter::new().kind(MemoryType::Coding))
            .await
            .expect("list kinds");
        assert!(kinds.is_empty());
    }

    #[tokio::test]
    async fn corrupt_records_are_skipped_by_list_but_rejected_by_get() {
        let storage = storage();
        let good = record("good", "healthy", 0);
        storage.put_record(&good).await.expect("put");
        let broken_key = keys::record_key(&scope(), "broken");
        storage
            .stores
            .kv
            .set(&broken_key, KvValue::text("{not json"))
            .await
            .expect("write corrupt");
        let bytes_key = keys::record_key(&scope(), "bytes");
        storage
            .stores
            .kv
            .set(&bytes_key, KvValue::bytes(vec![0_u8, 1, 2]))
            .await
            .expect("write bytes");
        let listed = storage
            .list_records(&MemoryFilter::for_scope(&scope()))
            .await
            .expect("list");
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].id, "good");
        let error = storage
            .get_record(&scope(), "broken")
            .await
            .expect_err("corrupt record");
        assert!(matches!(error, Error::Validation(_)), "{error:?}");
        let error = storage
            .get_record(&scope(), "bytes")
            .await
            .expect_err("byte record");
        assert_eq!(error.kind(), ErrorKind::Validation);
    }

    #[tokio::test]
    async fn delete_removes_the_record_and_reports_whether_it_existed() {
        let storage = storage();
        let record = record("gone", "delete me", 0);
        storage.put_record(&record).await.expect("put");
        assert!(storage.delete_record(&record).await.expect("delete"));
        assert!(storage
            .get_record(&scope(), "gone")
            .await
            .expect("get")
            .is_none());
        assert!(!storage.delete_record(&record).await.expect("delete again"));
    }

    #[tokio::test]
    async fn vectors_index_search_and_delete() {
        let storage = storage();
        let first = record("a", "near", 0);
        let second = record("b", "far", 1);
        storage.put_record(&first).await.expect("put first");
        storage.put_record(&second).await.expect("put second");
        storage
            .index_embedding(&first, &[1.0, 0.0])
            .await
            .expect("index first");
        storage
            .index_embedding(&second, &[0.0, 1.0])
            .await
            .expect("index second");
        let hits = storage
            .search_embeddings(&scope(), MemoryType::SemanticMemory, &[1.0, 0.0], 5)
            .await
            .expect("search");
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].0, "a");
        assert!(hits[0].1 >= hits[1].1);
        // Re-indexing replaces the entry instead of inserting a duplicate.
        storage
            .index_embedding(&first, &[1.0, 0.1])
            .await
            .expect("reindex");
        let hits = storage
            .search_embeddings(&scope(), MemoryType::SemanticMemory, &[1.0, 0.0], 5)
            .await
            .expect("search");
        assert_eq!(hits.len(), 2);
        assert!(storage
            .search_embeddings(&scope(), MemoryType::SemanticMemory, &[1.0, 0.0], 0)
            .await
            .expect("top_k zero")
            .is_empty());
        let missing = storage
            .search_embeddings(
                &MemoryScope::new("bob", "ws"),
                MemoryType::SemanticMemory,
                &[1.0, 0.0],
                5,
            )
            .await
            .expect("missing collection");
        assert!(missing.is_empty());
        storage
            .delete_embedding(&first)
            .await
            .expect("delete vector");
        let hits = storage
            .search_embeddings(&scope(), MemoryType::SemanticMemory, &[1.0, 0.0], 5)
            .await
            .expect("search");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, "b");
        let remaining = storage
            .list_records(&MemoryFilter::for_scope(&scope()))
            .await
            .expect("list remaining");
        for record in remaining {
            storage
                .delete_embedding(&record)
                .await
                .expect("delete embedding");
        }
        assert!(storage
            .search_embeddings(&scope(), MemoryType::SemanticMemory, &[1.0, 0.0], 5)
            .await
            .expect("search")
            .is_empty());
        // Deleting from a missing collection is a no-op.
        storage
            .delete_embedding(&record("absent", "absent", 2))
            .await
            .expect("missing collection");
    }

    #[tokio::test]
    async fn embedding_dimension_mismatch_is_rejected() {
        let storage = storage();
        storage
            .index_embedding(&record("a", "first", 0), &[1.0, 0.0, 0.0])
            .await
            .expect("index first");
        let error = storage
            .index_embedding(&record("b", "second", 1), &[1.0, 0.0])
            .await
            .expect_err("dimension mismatch");
        assert_eq!(error.kind(), ErrorKind::Validation);
    }

    #[tokio::test]
    async fn variables_are_text_scoped_and_prefix_listed() {
        let storage = storage();
        let scope = scope();
        storage
            .set_variable(&scope, None, "theme", "dark")
            .await
            .expect("set theme");
        storage
            .set_variable(&scope, None, "lang", "en")
            .await
            .expect("set lang");
        storage
            .set_variable(&scope, Some("s1"), "theme", "light")
            .await
            .expect("set session theme");
        storage
            .set_variable(&scope, Some("s2"), "theme", "blue")
            .await
            .expect("set other session");
        assert_eq!(
            storage
                .get_variable(&scope, None, "theme")
                .await
                .expect("get user"),
            Some("dark".to_owned())
        );
        assert_eq!(
            storage
                .get_variable(&scope, Some("s1"), "theme")
                .await
                .expect("get session"),
            Some("light".to_owned())
        );
        assert_eq!(
            storage
                .get_variable(&scope, None, "absent")
                .await
                .expect("get absent"),
            None
        );
        assert_eq!(
            storage
                .list_variables(&scope, None)
                .await
                .expect("list user"),
            BTreeMap::from([
                ("lang".to_owned(), "en".to_owned()),
                ("theme".to_owned(), "dark".to_owned()),
            ])
        );
        assert_eq!(
            storage
                .list_variables(&scope, Some("s1"))
                .await
                .expect("list session"),
            BTreeMap::from([("theme".to_owned(), "light".to_owned())])
        );
        assert!(storage
            .list_variables(&scope, Some("unknown"))
            .await
            .expect("list missing")
            .is_empty());
        assert!(storage
            .delete_variable(&scope, None, "theme")
            .await
            .expect("delete"));
        assert!(!storage
            .delete_variable(&scope, None, "theme")
            .await
            .expect("delete again"));
        assert_eq!(
            storage
                .list_variables(&scope, Some("s1"))
                .await
                .expect("session untouched")
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn scope_config_round_trips_and_flags_updates() {
        let storage = storage();
        let config = MemoryScopeConfig {
            user_profile_definition: Some("user facts".to_owned()),
            semantic_memory_definition: None,
            episodic_memory_definition: Some("events".to_owned()),
        };
        assert!(!storage
            .set_scope_config("ws", &config)
            .await
            .expect("create config"));
        assert_eq!(
            storage.get_scope_config("ws").await.expect("get config"),
            Some(config.clone())
        );
        assert!(storage
            .set_scope_config("ws", &config)
            .await
            .expect("replace config"));
        assert_eq!(
            storage
                .get_scope_config("other")
                .await
                .expect("absent config"),
            None
        );
        storage
            .stores
            .kv
            .set(&keys::scope_config_key("bad"), KvValue::text("not json"))
            .await
            .expect("write malformed");
        let error = storage
            .get_scope_config("bad")
            .await
            .expect_err("malformed config");
        assert!(matches!(error, Error::Validation(_)), "{error:?}");
        assert!(storage
            .delete_scope_config("ws")
            .await
            .expect("delete config"));
        assert!(!storage
            .delete_scope_config("ws")
            .await
            .expect("delete again"));
        assert_eq!(storage.get_scope_config("ws").await.expect("absent"), None);
    }

    async fn put_legacy(storage: &MemoryStorage, key: &str, value: &str) {
        storage
            .stores
            .kv
            .set(key, KvValue::text(value))
            .await
            .expect("write legacy value");
    }

    async fn kv_text(storage: &MemoryStorage, key: &str) -> Option<String> {
        storage
            .stores
            .kv
            .get(key)
            .await
            .expect("read key")
            .and_then(|value| value.as_str().map(str::to_owned))
    }

    async fn assert_key_gone(storage: &MemoryStorage, key: &str) {
        assert!(
            kv_text(storage, key).await.is_none(),
            "`{key}` still exists"
        );
    }

    fn variables(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
            .collect()
    }

    #[tokio::test]
    async fn legacy_record_keys_are_read_without_outranking_the_current_key() {
        let storage = storage();
        let scope = scope();
        let legacy_key = keys::legacy_record_key(&scope, "legacy");
        let legacy = record("legacy", "from the legacy layout", 0);
        let legacy_text = json::record_to_text(&legacy).expect("serialize record");
        put_legacy(&storage, &legacy_key, &legacy_text).await;
        let found = storage
            .get_record(&scope, "legacy")
            .await
            .expect("get legacy")
            .expect("legacy record readable");
        assert_eq!(found.content, "from the legacy layout");
        let listed = storage
            .list_records(&MemoryFilter::for_scope(&scope))
            .await
            .expect("list legacy");
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].id, "legacy");
        // A current-layout write of the same id wins for point reads and listings.
        let current = record("legacy", "from the current layout", 1);
        storage.put_record(&current).await.expect("put current");
        let found = storage
            .get_record(&scope, "legacy")
            .await
            .expect("get current")
            .expect("current record");
        assert_eq!(found.content, "from the current layout");
        let listed = storage
            .list_records(&MemoryFilter::for_scope(&scope))
            .await
            .expect("list both");
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].content, "from the current layout");
        // Reads never migrate implicitly: the legacy row is still stored.
        assert!(kv_text(&storage, &legacy_key).await.is_some());
    }

    #[tokio::test]
    async fn legacy_user_variables_are_read_and_current_writes_win() {
        let storage = storage();
        let scope = scope();
        let theme = keys::legacy_variable_key(&scope, "theme");
        put_legacy(&storage, &theme, "dark").await;
        put_legacy(&storage, &keys::legacy_variable_key(&scope, "lang"), "en").await;
        let value = storage
            .get_variable(&scope, None, "theme")
            .await
            .expect("get legacy");
        assert_eq!(value.as_deref(), Some("dark"));
        let listed = storage
            .list_variables(&scope, None)
            .await
            .expect("list legacy");
        assert_eq!(listed, variables(&[("theme", "dark"), ("lang", "en")]));
        storage
            .set_variable(&scope, None, "theme", "light")
            .await
            .expect("set theme");
        assert_eq!(
            kv_text(&storage, &theme).await.as_deref(),
            Some("dark"),
            "the write must land on the current key only"
        );
        let value = storage
            .get_variable(&scope, None, "theme")
            .await
            .expect("get current");
        assert_eq!(value.as_deref(), Some("light"));
        // The current-layout value wins over the legacy one, a legacy-only variable stays.
        let listed = storage
            .list_variables(&scope, None)
            .await
            .expect("list mixed");
        assert_eq!(listed, variables(&[("theme", "light"), ("lang", "en")]));
    }

    #[tokio::test]
    async fn legacy_session_variables_stay_in_their_session_namespace() {
        let storage = storage();
        let scope = scope();
        put_legacy(
            &storage,
            &keys::legacy_session_variable_key(&scope, "s1", "theme"),
            "light",
        )
        .await;
        let value = storage
            .get_variable(&scope, Some("s1"), "theme")
            .await
            .expect("get legacy session");
        assert_eq!(value.as_deref(), Some("light"));
        let listed = storage
            .list_variables(&scope, Some("s1"))
            .await
            .expect("list legacy session");
        assert_eq!(listed, variables(&[("theme", "light")]));
        // A session variable never leaks into the user-scoped namespace.
        let value = storage
            .get_variable(&scope, None, "theme")
            .await
            .expect("get user");
        assert_eq!(value, None);
        assert!(storage
            .list_variables(&scope, None)
            .await
            .expect("list user")
            .is_empty());
        assert!(storage
            .list_variables(&scope, Some("s2"))
            .await
            .expect("list other session")
            .is_empty());
        storage
            .set_variable(&scope, Some("s1"), "theme", "blue")
            .await
            .expect("set session");
        let value = storage
            .get_variable(&scope, Some("s1"), "theme")
            .await
            .expect("get current session");
        assert_eq!(value.as_deref(), Some("blue"));
    }

    #[tokio::test]
    async fn legacy_scope_config_is_read_and_current_writes_win() {
        let storage = storage();
        let legacy = MemoryScopeConfig {
            user_profile_definition: Some("legacy facts".to_owned()),
            semantic_memory_definition: None,
            episodic_memory_definition: None,
        };
        let legacy_json = serde_json::to_string(&legacy).expect("serialize config");
        put_legacy(&storage, &keys::legacy_scope_config_key("ws"), &legacy_json).await;
        let found = storage.get_scope_config("ws").await.expect("get legacy");
        assert_eq!(found, Some(legacy.clone()));
        let current = MemoryScopeConfig {
            user_profile_definition: Some("current facts".to_owned()),
            ..legacy
        };
        assert!(
            !storage
                .set_scope_config("ws", &current)
                .await
                .expect("set current"),
            "only the legacy key existed, so nothing was replaced"
        );
        let found = storage.get_scope_config("ws").await.expect("get current");
        assert_eq!(found, Some(current));
    }

    #[tokio::test]
    async fn deletes_remove_current_and_legacy_keys_and_report_existence() {
        let storage = storage();
        let scope = scope();
        let shared = record("shared", "record", 0);
        let shared_json = json::record_to_text(&shared).expect("serialize record");
        storage.put_record(&shared).await.expect("put record");
        put_legacy(
            &storage,
            &keys::legacy_record_key(&scope, "shared"),
            &shared_json,
        )
        .await;
        assert!(storage.delete_record(&shared).await.expect("delete record"));
        assert_key_gone(&storage, &keys::record_key(&scope, "shared")).await;
        assert_key_gone(&storage, &keys::legacy_record_key(&scope, "shared")).await;
        assert!(!storage.delete_record(&shared).await.expect("delete again"));
        // A legacy-only row is reported `true` on its first delete and `false` afterwards.
        let orphan = record("orphan", "legacy only", 1);
        let orphan_json = json::record_to_text(&orphan).expect("serialize orphan");
        put_legacy(
            &storage,
            &keys::legacy_record_key(&scope, "orphan"),
            &orphan_json,
        )
        .await;
        assert!(storage.delete_record(&orphan).await.expect("delete orphan"));
        assert!(!storage
            .delete_record(&orphan)
            .await
            .expect("delete orphan again"));

        storage
            .set_variable(&scope, None, "theme", "dark")
            .await
            .expect("set variable");
        put_legacy(
            &storage,
            &keys::legacy_variable_key(&scope, "theme"),
            "grey",
        )
        .await;
        assert!(storage
            .delete_variable(&scope, None, "theme")
            .await
            .expect("delete variable"));
        assert_key_gone(&storage, &keys::variable_key(&scope, "theme")).await;
        assert_key_gone(&storage, &keys::legacy_variable_key(&scope, "theme")).await;
        assert!(!storage
            .delete_variable(&scope, None, "theme")
            .await
            .expect("delete variable again"));

        let config = MemoryScopeConfig {
            user_profile_definition: None,
            semantic_memory_definition: Some("facts".to_owned()),
            episodic_memory_definition: None,
        };
        let config_json = serde_json::to_string(&config).expect("serialize config");
        storage
            .set_scope_config("ws", &config)
            .await
            .expect("set config");
        put_legacy(&storage, &keys::legacy_scope_config_key("ws"), &config_json).await;
        assert!(storage
            .delete_scope_config("ws")
            .await
            .expect("delete config"));
        assert_key_gone(&storage, &keys::scope_config_key("ws")).await;
        assert_key_gone(&storage, &keys::legacy_scope_config_key("ws")).await;
        assert!(!storage
            .delete_scope_config("ws")
            .await
            .expect("delete config again"));
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn sqlite_stores_persist_across_reopen() {
        let directory = tempfile::tempdir().expect("tempdir");
        let path = directory.path().join("memory.db");
        let record = record("persisted", "sqlite record", 0);
        let message_time = timestamp(0);
        {
            let storage =
                MemoryStorage::new(MemoryStores::sqlite(&path).await.expect("open sqlite"));
            storage.put_record(&record).await.expect("put record");
            let messages = MessageStore::new(storage.stores.document.clone());
            messages
                .append(&scope(), &[BaseMessage::user("hello")], message_time)
                .await
                .expect("append message");
        }
        let storage = MemoryStorage::new(MemoryStores::sqlite(&path).await.expect("reopen sqlite"));
        let restored = storage
            .get_record(&record.scope, "persisted")
            .await
            .expect("get record")
            .expect("record persisted");
        assert_eq!(restored.content, "sqlite record");
        let messages = MessageStore::new(storage.stores.document.clone());
        let history = messages.recent(&scope(), 10).await.expect("recent");
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].content, "hello");
        assert_eq!(history[0].timestamp, message_time);
    }
}
