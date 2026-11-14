//! The variable write path: a named variable is stored twice.
//!
//! Python's `manage/index/variable_manager.py` keeps variables in the KV keyspace only
//! (`user_var/{user}/{scope}/{name}` and `session_var/{user}/{scope}/{session}/{name}`).
//! This module keeps that entry — it stays the authoritative value every Python-shaped
//! reader consults — and adds a first-class [`MemoryType::Variable`] record projection,
//! so a variable is listed, counted, and (when a caller explicitly asks for the kind)
//! recalled like any other record.
//!
//! # Deterministic ids
//!
//! The projection id is derived from `(user_id, scope_id, session_id, name)` with the
//! same 64-bit FNV-1a hash family [`crate::record::new_memory_id`] uses, rendered as
//! `var-<16 hex digits>`. Writing the same variable again therefore updates the same
//! record in place (creation time preserved, `updated_at` advanced) instead of
//! accumulating one record per value, and no text-similarity dedupe is involved.
//! `agent_id` is not part of the id because it is not part of the KV layout either: the
//! projection mirrors one storage key, not one agent binding.
//!
//! `MemoryType::Variable` is not a searchable kind (the configuration rejects it in
//! `searchable_kinds`), so the projection is never a duplicate candidate of an ordinary
//! `add`, and an ordinary recall never returns it. It becomes recallable only when a
//! caller passes the kind explicitly in `SearchOptions::kinds`.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use openjiuwen_common::error::{Error, Result};

use crate::record::{AddOptions, MemoryFilter, MemoryRecord, MemoryScope, MemoryType};

/// The tracing target of engine diagnostics.
const TARGET: &str = "openjiuwen_memory::engine";
/// The metadata field holding the variable name.
const VARIABLE_NAME_FIELD: &str = "name";
/// The metadata field holding the variable value.
const VARIABLE_VALUE_FIELD: &str = "value";
/// The `source` marker of a variable projection record.
const VARIABLE_SOURCE: &str = "variable";
/// The prefix of every deterministic variable record id.
const VARIABLE_ID_PREFIX: &str = "var-";
/// The 64-bit FNV-1a offset basis (the id hash family of the record module).
const FNV1A_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
/// The 64-bit FNV-1a prime.
const FNV1A_PRIME: u64 = 0x0000_0100_0000_01b3;

/// The outcome of [`MemoryEngine::write_variable`](super::MemoryEngine::write_variable).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VariableWrite {
    /// The id of the `MemoryType::Variable` record projection.
    pub record_id: String,
    /// Whether this call created the projection (`false` means it was updated in place).
    pub created: bool,
}

impl super::MemoryEngine {
    /// Stores `name = value` as a variable entry plus its record projection.
    ///
    /// The KV entry is written in the session namespace when `scope.session_id` is set
    /// and in the user namespace otherwise, exactly like the facade's variable API.
    /// The projection keeps its creation `timestamp`, takes the newest content/value,
    /// and advances `updated_at`; it is embedded when an embedding model is attached
    /// (unless `options.skip_embedding`), so it can be recalled by content.
    ///
    /// `options.dedupe` is ignored: the projection is keyed by name+scope, never by
    /// text similarity, and it never merges with or absorbs another record.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an invalid scope, an empty name, a name with
    /// `/` or NUL, or a name listed in `MemoryEngineConfig::forbidden_variables` (all
    /// before any I/O), [`Error::Cancelled`]/[`Error::Timeout`] when the call is
    /// cancelled or exceeds its deadline, and the storage/embedding error when a write
    /// fails. A failed projection write rolls the projection back; the KV entry written
    /// by the same call stays, because it is the Python-authoritative representation.
    pub async fn write_variable(
        &self,
        scope: &MemoryScope,
        name: &str,
        value: &str,
        options: &AddOptions,
    ) -> Result<VariableWrite> {
        scope.validate()?;
        validate_variable_name(name)?;
        self.check_forbidden_variable(name)?;
        options.call.check()?;
        let lock = self.write_locks.lock_for(scope);
        let _guard = lock.lock().await;
        let now = Utc::now();
        let id = variable_record_id(scope, name);

        options.call.check()?;
        let previous = self.variable_projection(scope, &id).await?;
        let record = self.projection_record(scope, &id, name, value, previous.as_ref(), now);

        let embedding = if self.has_embedding() && !options.skip_embedding {
            Some(self.embed_text(&record.content, &options.call).await?)
        } else {
            None
        };
        options.call.check()?;
        self.storage()
            .set_variable(scope, scope.session_id.as_deref(), name, value)
            .await?;
        options.call.check()?;
        self.persist(&record, embedding.as_deref()).await?;
        Ok(VariableWrite {
            record_id: id,
            created: previous.is_none(),
        })
    }

    /// Removes both representations of the variable `name` of `scope`.
    ///
    /// Returns whether either representation existed. The projection removal also
    /// removes its vector entry (the same behaviour as deleting an ordinary record).
    /// A row at the deterministic id that is not a `Variable` projection, or one that
    /// cannot be decoded, keeps its place: its kind is unknown, so it is logged and
    /// never deleted here.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an invalid scope or name (checked before any
    /// I/O) and the storage error when a delete fails.
    pub async fn delete_variable_everywhere(
        &self,
        scope: &MemoryScope,
        name: &str,
    ) -> Result<bool> {
        scope.validate()?;
        validate_variable_name(name)?;
        let lock = self.write_locks.lock_for(scope);
        let _guard = lock.lock().await;
        let id = variable_record_id(scope, name);
        let removed_entry = self
            .storage()
            .delete_variable(scope, scope.session_id.as_deref(), name)
            .await?;
        let removed_projection = match self.variable_projection(scope, &id).await? {
            Some(record) => self.storage().delete_record(&record).await?,
            None => false,
        };
        Ok(removed_entry || removed_projection)
    }

    /// Lists the `MemoryType::Variable` records of `scope`, ascending by timestamp/id.
    ///
    /// Malformed stored rows are skipped with a warning by the storage scan, never
    /// fatal. A scope without `session_id` selects the variables of every session of
    /// `(user_id, scope_id)`, matching [`MemoryScope::matches`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an invalid scope and the storage error when the
    /// listing fails.
    pub async fn variable_records(&self, scope: &MemoryScope) -> Result<Vec<MemoryRecord>> {
        scope.validate()?;
        let filter = MemoryFilter::for_scope(scope).kind(MemoryType::Variable);
        let mut records = self.storage().list_records(&filter).await?;
        records.sort_by(|left, right| {
            left.timestamp
                .cmp(&right.timestamp)
                .then_with(|| left.id.cmp(&right.id))
        });
        Ok(records)
    }

    /// Reads the existing projection at `id`, or `None` when there is none.
    ///
    /// A row at the deterministic id that is not a `MemoryType::Variable` record is not
    /// touched: it is reported as absent with a warning, so the rewrite replaces only
    /// our own projection. An unreadable row at that id is treated as absent with a
    /// warning too, which lets the next write repair it.
    async fn variable_projection(
        &self,
        scope: &MemoryScope,
        id: &str,
    ) -> Result<Option<MemoryRecord>> {
        match self.storage().get_record(scope, id).await {
            Ok(Some(record)) if record.kind == MemoryType::Variable => Ok(Some(record)),
            Ok(Some(record)) => {
                tracing::warn!(
                    target: TARGET,
                    id = %record.id,
                    kind = %record.kind,
                    "the variable record id is held by another kind; rewriting the variable projection"
                );
                Ok(None)
            }
            Ok(None) => Ok(None),
            Err(Error::Validation(message)) => {
                tracing::warn!(
                    target: TARGET,
                    id,
                    %message,
                    "the stored variable projection is unreadable; rewriting it"
                );
                Ok(None)
            }
            Err(error) => Err(error),
        }
    }

    /// Builds the projection record of `name = value`, reusing `previous` when present.
    ///
    /// A reused record keeps its id, creation time, importance, and metadata; the
    /// content, kind, scope, `name`/`value` metadata fields, and `updated_at` are
    /// refreshed.
    fn projection_record(
        &self,
        scope: &MemoryScope,
        id: &str,
        name: &str,
        value: &str,
        previous: Option<&MemoryRecord>,
        now: DateTime<Utc>,
    ) -> MemoryRecord {
        let content = render_variable(name, value);
        let mut record = match previous {
            Some(previous) => previous.clone(),
            None => MemoryRecord::new(scope.clone(), content.clone(), MemoryType::Variable)
                .with_id(id.to_owned())
                .with_importance(self.config.importance_default)
                .with_timestamp(now)
                .with_source(VARIABLE_SOURCE),
        };
        record.kind = MemoryType::Variable;
        record.content = content;
        record.scope = scope.clone();
        if previous.is_none() {
            record.timestamp = now;
        }
        record.updated_at = now;
        record.metadata.insert(
            VARIABLE_NAME_FIELD.to_owned(),
            Value::String(name.to_owned()),
        );
        record.metadata.insert(
            VARIABLE_VALUE_FIELD.to_owned(),
            Value::String(value.to_owned()),
        );
        record
    }

    /// Rejects a variable name listed in `MemoryEngineConfig::forbidden_variables`.
    ///
    /// The comparison trims both sides and is ASCII case-insensitive, so a configured
    /// `api_key` also rejects `API_KEY ` and ` Api_Key`.
    fn check_forbidden_variable(&self, name: &str) -> Result<()> {
        let candidate = name.trim();
        let forbidden = self
            .config
            .forbidden_variables
            .iter()
            .any(|configured| configured.trim().eq_ignore_ascii_case(candidate));
        if forbidden {
            return Err(Error::validation(format!(
                "memory variable `{candidate}` is forbidden by `forbidden_variables`"
            )));
        }
        Ok(())
    }
}

/// Renders a variable as the record content: `{name}={value}`.
fn render_variable(name: &str, value: &str) -> String {
    format!("{name}={value}")
}

/// The deterministic projection id of `(user_id, scope_id, session_id, name)`.
fn variable_record_id(scope: &MemoryScope, name: &str) -> String {
    let mut input =
        String::with_capacity(scope.user_id.len() + scope.scope_id.len() + name.len() + 3);
    input.push_str(&scope.user_id);
    input.push('\0');
    input.push_str(&scope.scope_id);
    input.push('\0');
    input.push_str(scope.session_id.as_deref().unwrap_or_default());
    input.push('\0');
    input.push_str(name);
    format!("{VARIABLE_ID_PREFIX}{:016x}", fnv1a_64(input.as_bytes()))
}

/// Computes the 64-bit FNV-1a hash of `bytes` (the id hash family of the record module).
fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut hash = FNV1A_OFFSET_BASIS;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV1A_PRIME);
    }
    hash
}

/// Validates a variable name: non-blank, without `/` and without NUL.
fn validate_variable_name(name: &str) -> Result<()> {
    if name.trim().is_empty() {
        return Err(Error::validation(format!(
            "memory variable `name` must not be empty (got {name:?})"
        )));
    }
    if name.contains('/') {
        return Err(Error::validation(format!(
            "memory variable `name` must not contain `/` (got {name:?})"
        )));
    }
    if name.contains('\0') {
        return Err(Error::validation(format!(
            "memory variable `name` must not contain NUL (got {name:?})"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use chrono::{DateTime, Utc};
    use openjiuwen_common::error::Error;
    use openjiuwen_retrieval::embedding::HashEmbedding;

    use super::super::{MemoryEngine, MemoryEngineBuilder};
    use super::VariableWrite;
    use crate::config::MemoryEngineConfig;
    use crate::record::{
        AddOptions, MemoryFilter, MemoryScope, MemoryType, NewMemory, SearchOptions,
    };
    use crate::store::MemoryStores;

    fn engine() -> Arc<MemoryEngine> {
        MemoryEngine::new(MemoryEngineConfig::default(), MemoryStores::in_memory()).expect("engine")
    }

    fn embedding_engine() -> Arc<MemoryEngine> {
        MemoryEngineBuilder::new(MemoryEngineConfig::default())
            .embedding(Arc::new(HashEmbedding::new(16)))
            .build()
            .expect("engine")
    }

    fn scope() -> MemoryScope {
        MemoryScope::new("alice", "ws")
    }

    fn ts(offset: i64) -> DateTime<Utc> {
        DateTime::from_timestamp(1_700_000_000 + offset, 0).expect("fixed time")
    }

    /// The variable kind requested explicitly, as a caller asking for variables must.
    fn variable_only() -> SearchOptions {
        SearchOptions::default()
            .with_kinds(vec![MemoryType::Variable])
            .with_threshold(0.0)
    }

    async fn variable_record_count(engine: &MemoryEngine, scope: &MemoryScope) -> usize {
        engine
            .variable_records(scope)
            .await
            .expect("variable records")
            .len()
    }

    #[tokio::test]
    async fn a_variable_is_a_kv_entry_and_one_projection() {
        let engine = engine();
        let scope = scope();
        let write = engine
            .write_variable(&scope, "theme", "dark", &AddOptions::default())
            .await
            .expect("write");
        assert!(write.created);
        assert_eq!(
            engine
                .storage()
                .get_variable(&scope, None, "theme")
                .await
                .expect("kv"),
            Some("dark".to_owned())
        );
        let records = engine.variable_records(&scope).await.expect("records");
        assert_eq!(records.len(), 1);
        let record = &records[0];
        assert_eq!(record.id, write.record_id);
        assert_eq!(record.kind, MemoryType::Variable);
        assert_eq!(record.content, "theme=dark");
        assert_eq!(record.metadata_str("name"), Some("theme"));
        assert_eq!(record.metadata_str("value"), Some("dark"));
        assert_eq!(record.source.as_deref(), Some("variable"));
        assert_eq!(record.scope, scope);
        assert!(engine
            .get(&scope, &write.record_id)
            .await
            .expect("point read")
            .is_some());
    }

    #[tokio::test]
    async fn a_session_scoped_variable_lives_in_the_session_namespace_and_projection() {
        let engine = engine();
        let session = scope().session("s1");
        let write = engine
            .write_variable(&session, "theme", "light", &AddOptions::default())
            .await
            .expect("write");
        assert_eq!(
            engine
                .storage()
                .get_variable(&session, Some("s1"), "theme")
                .await
                .expect("session kv"),
            Some("light".to_owned())
        );
        assert_eq!(
            engine
                .storage()
                .get_variable(&session, None, "theme")
                .await
                .expect("user kv"),
            None
        );
        let records = engine.variable_records(&session).await.expect("records");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].id, write.record_id);
        assert_eq!(records[0].scope.session_id.as_deref(), Some("s1"));
    }

    #[tokio::test]
    async fn rewriting_a_variable_updates_one_record_in_place() {
        let engine = engine();
        let scope = scope();
        let first = engine
            .write_variable(&scope, "theme", "dark", &AddOptions::default())
            .await
            .expect("write");
        let created_at = engine
            .get(&scope, &first.record_id)
            .await
            .expect("get")
            .expect("present")
            .timestamp;
        tokio::time::sleep(Duration::from_millis(2)).await;
        let second = engine
            .write_variable(&scope, "theme", "light", &AddOptions::default())
            .await
            .expect("rewrite");
        assert!(!second.created);
        assert_eq!(second.record_id, first.record_id);
        let records = engine.variable_records(&scope).await.expect("records");
        assert_eq!(
            records.len(),
            1,
            "the projection is keyed by name, not by value"
        );
        assert_eq!(records[0].content, "theme=light");
        assert_eq!(records[0].metadata_str("value"), Some("light"));
        assert_eq!(
            records[0].timestamp, created_at,
            "creation time is preserved"
        );
        assert!(records[0].updated_at > created_at, "updated_at advances");
        assert_eq!(
            engine
                .storage()
                .get_variable(&scope, None, "theme")
                .await
                .expect("kv"),
            Some("light".to_owned())
        );
    }

    #[tokio::test]
    async fn a_forbidden_variable_is_rejected_before_any_write() {
        let config = MemoryEngineConfig {
            forbidden_variables: vec!["API_KEY".to_owned()],
            ..Default::default()
        };
        let engine = MemoryEngine::new(config, MemoryStores::in_memory()).expect("engine");
        let scope = scope();
        let error = engine
            .write_variable(&scope, " api_key ", "secret", &AddOptions::default())
            .await
            .expect_err("forbidden");
        match error {
            Error::Validation(message) => assert!(message.contains("api_key"), "{message}"),
            other => panic!("expected Validation, got {other:?}"),
        }
        assert!(engine
            .storage()
            .list_variables(&scope, None)
            .await
            .expect("variables")
            .is_empty());
        assert_eq!(variable_record_count(&engine, &scope).await, 0);
    }

    #[tokio::test]
    async fn invalid_names_are_rejected() {
        let engine = engine();
        let scope = scope();
        for name in ["", "   ", "a/b", "a\0b"] {
            let error = engine
                .write_variable(&scope, name, "value", &AddOptions::default())
                .await
                .expect_err("invalid name");
            assert!(matches!(error, Error::Validation(_)), "{name:?}: {error:?}");
            assert!(error.to_string().contains("name"), "{name:?}: {error}");
        }
        assert!(engine
            .storage()
            .list_variables(&scope, None)
            .await
            .expect("variables")
            .is_empty());
        assert_eq!(variable_record_count(&engine, &scope).await, 0);
        let error = engine
            .delete_variable_everywhere(&scope, "a/b")
            .await
            .expect_err("invalid name");
        assert!(matches!(error, Error::Validation(_)), "{error:?}");
    }

    #[tokio::test]
    async fn deleting_removes_the_entry_the_projection_and_its_vector() {
        let engine = embedding_engine();
        let scope = scope();
        let write = engine
            .write_variable(&scope, "theme", "dark", &AddOptions::default())
            .await
            .expect("write");
        assert!(
            !engine
                .search(&scope, "theme=dark", &variable_only())
                .await
                .expect("search")
                .is_empty(),
            "the embedded projection is recallable"
        );
        assert!(engine
            .delete_variable_everywhere(&scope, "theme")
            .await
            .expect("delete"));
        assert_eq!(
            engine
                .storage()
                .get_variable(&scope, None, "theme")
                .await
                .expect("kv"),
            None
        );
        assert_eq!(variable_record_count(&engine, &scope).await, 0);
        assert!(engine
            .get(&scope, &write.record_id)
            .await
            .expect("get")
            .is_none());
        assert!(
            engine
                .search(&scope, "theme=dark", &variable_only())
                .await
                .expect("search")
                .is_empty(),
            "the vector entry is gone"
        );
        assert!(!engine
            .delete_variable_everywhere(&scope, "theme")
            .await
            .expect("delete again"));
    }

    #[tokio::test]
    async fn variable_records_orders_ascending_and_is_scope_isolated() {
        let engine = engine();
        let scope = scope();
        engine
            .write_variable(&scope, "beta", "2", &AddOptions::default())
            .await
            .expect("write beta");
        tokio::time::sleep(Duration::from_millis(2)).await;
        engine
            .write_variable(&scope, "alpha", "1", &AddOptions::default())
            .await
            .expect("write alpha");
        let names = engine
            .variable_records(&scope)
            .await
            .expect("records")
            .into_iter()
            .map(|record| record.metadata_str("name").expect("name").to_owned())
            .collect::<Vec<_>>();
        assert_eq!(names, ["beta", "alpha"]);

        let other_user = MemoryScope::new("bob", "ws");
        let other_scope = MemoryScope::new("alice", "other");
        engine
            .write_variable(&other_user, "beta", "9", &AddOptions::default())
            .await
            .expect("write bob");
        engine
            .write_variable(&other_scope, "beta", "9", &AddOptions::default())
            .await
            .expect("write other scope");
        assert_eq!(variable_record_count(&engine, &scope).await, 2);
        assert_eq!(variable_record_count(&engine, &other_user).await, 1);
        assert_eq!(variable_record_count(&engine, &other_scope).await, 1);
    }

    #[tokio::test]
    async fn list_and_stats_include_the_projection() {
        let engine = engine();
        let scope = scope();
        let write = engine
            .write_variable(&scope, "theme", "dark", &AddOptions::default())
            .await
            .expect("write");
        engine
            .add(
                &scope,
                NewMemory::semantic("Alice prefers dark mode").with_timestamp(ts(0)),
                &AddOptions::default(),
            )
            .await
            .expect("add");
        let listed = engine
            .list(&MemoryFilter::for_scope(&scope))
            .await
            .expect("list");
        assert_eq!(listed.len(), 2);
        assert!(listed.iter().any(|record| record.id == write.record_id));
        let stats = engine
            .stats(&MemoryFilter::for_scope(&scope))
            .await
            .expect("stats");
        assert_eq!(stats.total, 2);
        assert_eq!(stats.by_kind.get("variable"), Some(&1));
        assert_eq!(stats.by_kind.get("semantic_memory"), Some(&1));
    }

    #[tokio::test]
    async fn an_embedded_projection_is_recallable_by_content() {
        let engine = embedding_engine();
        let scope = scope();
        let write = engine
            .write_variable(&scope, "theme", "dark", &AddOptions::default())
            .await
            .expect("write");
        engine
            .add(
                &scope,
                NewMemory::semantic("banana bread recipe"),
                &AddOptions::default(),
            )
            .await
            .expect("add");
        let hits = engine
            .search(&scope, "theme=dark", &variable_only())
            .await
            .expect("search");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].record.id, write.record_id);
        assert!(hits[0].score > 0.0);
        // A default recall (the searchable kinds of the configuration) never returns it.
        let default_hits = engine
            .search(
                &scope,
                "theme=dark",
                &SearchOptions::default().with_threshold(0.0),
            )
            .await
            .expect("search");
        assert!(default_hits
            .iter()
            .all(|hit| hit.record.kind != MemoryType::Variable));
    }

    #[tokio::test]
    async fn the_projection_is_not_merged_with_ordinary_records() {
        let engine = engine();
        let scope = scope();
        let write = engine
            .write_variable(&scope, "theme", "dark", &AddOptions::default())
            .await
            .expect("write");
        let outcome = engine
            .add(
                &scope,
                NewMemory::semantic("theme=dark"),
                &AddOptions::default(),
            )
            .await
            .expect("add same text");
        assert_eq!(
            outcome.added.len(),
            1,
            "the ordinary add is not a duplicate"
        );
        assert_eq!(outcome.merged.len(), 0);
        let records = engine.variable_records(&scope).await.expect("records");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].id, write.record_id);
        assert_eq!(records[0].content, "theme=dark");
        assert_eq!(
            engine
                .list(&MemoryFilter::for_scope(&scope))
                .await
                .expect("list")
                .len(),
            2
        );
    }

    #[tokio::test]
    async fn a_cancelled_write_leaves_both_representations_untouched() {
        let engine = engine();
        let scope = scope();
        let write = engine
            .write_variable(&scope, "theme", "dark", &AddOptions::default())
            .await
            .expect("write");
        let token = tokio_util::sync::CancellationToken::new();
        token.cancel();
        let error = engine
            .write_variable(
                &scope,
                "theme",
                "light",
                &AddOptions::default().with_cancellation(token),
            )
            .await
            .expect_err("cancelled");
        assert!(matches!(error, Error::Cancelled), "{error:?}");
        assert_eq!(
            engine
                .storage()
                .get_variable(&scope, None, "theme")
                .await
                .expect("kv"),
            Some("dark".to_owned())
        );
        let records = engine.variable_records(&scope).await.expect("records");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].id, write.record_id);
        assert_eq!(records[0].content, "theme=dark");
    }

    #[tokio::test]
    async fn an_unreadable_projection_is_rewritten_by_the_next_write() {
        let engine = engine();
        let scope = scope();
        let write = engine
            .write_variable(&scope, "theme", "dark", &AddOptions::default())
            .await
            .expect("write");
        let key = crate::store::keys::record_key(&scope, &write.record_id);
        engine
            .stores()
            .kv
            .set(
                &key,
                openjiuwen_foundation::store::KvValue::text("{not json"),
            )
            .await
            .expect("write corrupt");
        let again = engine
            .write_variable(&scope, "theme", "light", &AddOptions::default())
            .await
            .expect("rewrite");
        assert_eq!(again.record_id, write.record_id);
        let records = engine.variable_records(&scope).await.expect("records");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].content, "theme=light");
    }

    #[test]
    fn the_projection_id_is_deterministic_and_scope_specific() {
        let scope = scope();
        let first = super::variable_record_id(&scope, "theme");
        assert_eq!(first, super::variable_record_id(&scope, "theme"));
        assert_eq!(first.len(), 4 + 16);
        assert!(first.starts_with("var-"));
        assert!(first[4..]
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
        assert_ne!(first, super::variable_record_id(&scope, "other"));
        assert_ne!(
            first,
            super::variable_record_id(&scope.session("s1"), "theme")
        );
        assert_ne!(
            first,
            super::variable_record_id(&MemoryScope::new("bob", "ws"), "theme")
        );
        assert_ne!(
            first,
            super::variable_record_id(&MemoryScope::new("alice", "other"), "theme")
        );
    }

    #[test]
    fn variable_write_round_trips_through_json() {
        let write = VariableWrite {
            record_id: "var-0000000000000001".to_owned(),
            created: true,
        };
        let json = serde_json::to_value(&write).expect("serialize");
        assert_eq!(json["record_id"], serde_json::json!("var-0000000000000001"));
        assert_eq!(json["created"], serde_json::json!(true));
        let restored: VariableWrite = serde_json::from_value(json).expect("deserialize");
        assert_eq!(restored, write);
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn a_sqlite_projection_persists_across_reopen() {
        let directory = tempfile::tempdir().expect("tempdir");
        let path = directory.path().join("memory.db");
        let record_id = {
            let stores = MemoryStores::sqlite(&path).await.expect("open sqlite");
            let engine = MemoryEngine::new(MemoryEngineConfig::default(), stores).expect("engine");
            let scope = scope();
            let write = engine
                .write_variable(&scope, "theme", "dark", &AddOptions::default())
                .await
                .expect("write");
            assert!(write.created);
            write.record_id
        };
        let stores = MemoryStores::sqlite(&path).await.expect("reopen sqlite");
        let engine = MemoryEngine::new(MemoryEngineConfig::default(), stores).expect("engine");
        let scope = scope();
        let records = engine.variable_records(&scope).await.expect("records");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].id, record_id);
        assert_eq!(records[0].content, "theme=dark");
        assert_eq!(records[0].metadata_str("value"), Some("dark"));
        assert_eq!(
            engine
                .storage()
                .get_variable(&scope, None, "theme")
                .await
                .expect("kv"),
            Some("dark".to_owned())
        );
    }
}
