//! The write path: `add`/`add_many` (validation, embedding, dedupe, persistence) and
//! `update`/`delete`/`clear`/`get`/`list`.
//!
//! Every mutating operation is serialized per `(user_id, scope_id)` by
//! [`super::locks::WriteLocks`], so the dedupe scan and the write of one call cannot
//! interleave with another call of the same scope.
//! Ranking and duplicate detection live in the `recall` module; this module only
//! assembles candidates, applies the configured policy, and keeps persistence
//! consistent (a record is never left half-indexed).

use std::sync::Arc;

use chrono::{DateTime, Utc};
use openjiuwen_common::error::{Error, Result};
use openjiuwen_retrieval::embedding::Embedding;

use crate::recall::{find_duplicate, merge_records, Candidate};
use crate::record::{
    new_memory_id, AddOptions, AddOutcome, CallOptions, DedupePolicy, MemoryFilter, MemoryRecord,
    MemoryScope, MemoryUpdate, NewMemory, SearchMode, SearchOptions,
};

/// The tracing target of engine diagnostics.
const TARGET: &str = "openjiuwen_memory::engine";

/// The result of a duplicate candidate scan.
///
/// [`degraded`](DedupeScan::degraded) carries the reason the scan could not run, so the
/// caller can report "wrote without checking for duplicates" instead of silently treating
/// a failed scan as "no duplicates".
struct DedupeScan {
    /// The records a duplicate check compares against.
    candidates: Vec<Candidate>,
    /// The reason the scan failed and degraded to "no candidates", when it did.
    degraded: Option<String>,
}

impl super::MemoryEngine {
    /// Stores one memory, running duplicate detection according to `options`.
    ///
    /// Duplicate detection is best-effort: when the candidate scan fails for any reason
    /// other than cancellation/timeout, the memory is still stored and
    /// [`AddOutcome::degraded`] carries the reason, so the caller can tell a scope without
    /// duplicates apart from one that could not be scanned.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an invalid scope or memory,
    /// [`Error::Cancelled`]/[`Error::Timeout`] when the call is cancelled or exceeds its
    /// deadline, and the storage/embedding error when a write fails.
    pub async fn add(
        &self,
        scope: &MemoryScope,
        memory: NewMemory,
        options: &AddOptions,
    ) -> Result<AddOutcome> {
        self.add_with_embedding(scope, memory, options, None).await
    }

    /// Stores one memory with `embedding` as the embedder of this call.
    ///
    /// `embedding` is the handle a per-scope override resolved to; `None` means "use the
    /// engine's own model". A supplied handle is never replaced by the engine's, and an
    /// absent engine handle with no supplied handle stores the record without a vector
    /// (keyword-only recall), exactly like [`MemoryEngine::add`](super::MemoryEngine::add).
    ///
    /// # Errors
    ///
    /// See [`MemoryEngine::add`](super::MemoryEngine::add).
    pub async fn add_with_embedding(
        &self,
        scope: &MemoryScope,
        memory: NewMemory,
        options: &AddOptions,
        embedding: Option<&Arc<dyn Embedding>>,
    ) -> Result<AddOutcome> {
        self.add_many_with_embedding(scope, vec![memory], options, embedding)
            .await
    }

    /// Stores a batch sequentially under one scope lock, deduping against stored and
    /// earlier batch records.
    ///
    /// A failed item stops the batch; every item written before the failure stays
    /// committed and is reported in the returned error path only through storage.
    ///
    /// # Errors
    ///
    /// See [`MemoryEngine::add`](super::MemoryEngine::add); the first failing item's
    /// error is returned.
    pub async fn add_many(
        &self,
        scope: &MemoryScope,
        memories: Vec<NewMemory>,
        options: &AddOptions,
    ) -> Result<AddOutcome> {
        self.add_many_with_embedding(scope, memories, options, None)
            .await
    }

    /// Stores a batch under one scope lock with `embedding` as the embedder of this call.
    ///
    /// # Errors
    ///
    /// See [`MemoryEngine::add_many`](super::MemoryEngine::add_many).
    pub async fn add_many_with_embedding(
        &self,
        scope: &MemoryScope,
        memories: Vec<NewMemory>,
        options: &AddOptions,
        embedding: Option<&Arc<dyn Embedding>>,
    ) -> Result<AddOutcome> {
        scope.validate()?;
        let mut outcome = AddOutcome::default();
        if memories.is_empty() {
            return Ok(outcome);
        }
        let lock = self.write_locks.lock_for(scope);
        let _guard = lock.lock().await;
        let now = Utc::now();
        for memory in memories {
            options.call.check()?;
            memory.validate()?;
            let record = self.build_record(scope, &memory, now);
            record.validate()?;
            self.write_record(record, options, now, &mut outcome, embedding)
                .await?;
        }
        Ok(outcome)
    }

    /// Reads one record of `scope`, or `None` when it does not exist.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an invalid scope and [`Error::Validation`] when
    /// the stored JSON is corrupt (a point read never hides corruption).
    pub async fn get(&self, scope: &MemoryScope, id: &str) -> Result<Option<MemoryRecord>> {
        scope.validate()?;
        self.storage().get_record(scope, id).await
    }

    /// Applies `patch` to the stored record `id` and returns the updated record.
    ///
    /// Content changes are re-embedded before anything is written, so a failed
    /// embedding leaves the stored record unchanged. A kind change moves the record's
    /// vector to the new kind collection; when indexing that vector fails, the previous
    /// record and its vector are restored, so the record is never left persisted but
    /// unsearchable (see
    /// [`MemoryEngine::update_with_embedding`](super::MemoryEngine::update_with_embedding)).
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] when `id` does not exist, [`Error::Validation`] for
    /// an invalid scope or patch, and the storage/embedding error when the write fails.
    pub async fn update(
        &self,
        scope: &MemoryScope,
        id: &str,
        patch: MemoryUpdate,
        options: &CallOptions,
    ) -> Result<MemoryRecord> {
        self.update_with_embedding(scope, id, patch, options, None)
            .await
    }

    /// Applies `patch` with `embedding` as the embedder of this call.
    ///
    /// A kind change deletes the previous kind's vector before the new one is indexed. When
    /// that indexing fails, the previous record row is restored and its vector is
    /// regenerated from its content (one extra embedding call), and the error names that
    /// restore. If the restore itself fails, the row is still put back first and the error
    /// names the leftover state: the record is keyword-searchable only.
    ///
    /// # Errors
    ///
    /// See [`MemoryEngine::update`](super::MemoryEngine::update).
    pub async fn update_with_embedding(
        &self,
        scope: &MemoryScope,
        id: &str,
        patch: MemoryUpdate,
        options: &CallOptions,
        embedding: Option<&Arc<dyn Embedding>>,
    ) -> Result<MemoryRecord> {
        scope.validate()?;
        patch.validate()?;
        options.check()?;
        let lock = self.write_locks.lock_for(scope);
        let _guard = lock.lock().await;
        options.check()?;
        let Some(mut record) = self.storage().get_record(scope, id).await? else {
            return Err(Error::not_found(format!("memory record `{id}` not found")));
        };
        if patch.is_empty() {
            return Ok(record);
        }
        let previous = record.clone();
        let mut content_changed = false;
        if let Some(content) = patch.content {
            if content != record.content {
                record.content = content;
                content_changed = true;
            }
        }
        if let Some(importance) = patch.importance {
            record.importance = importance;
        }
        if let Some(kind) = patch.kind {
            record.kind = kind;
        }
        record.metadata.extend(patch.metadata);
        if let Some(source) = patch.source {
            record.source = Some(source);
        }
        record.updated_at = Utc::now();
        record.validate()?;

        let kind_changed = record.kind != previous.kind;
        let vector = if (content_changed || kind_changed) && self.has_embedding_with(embedding) {
            Some(
                self.embed_text_with(embedding, &record.content, options)
                    .await?,
            )
        } else {
            None
        };
        if kind_changed {
            // The old vector lives in the previous kind's collection; remove it before
            // the new vector is indexed under the same record id.
            self.storage().delete_embedding(&previous).await?;
        }
        self.storage().put_record(&record).await?;
        if let Some(vector) = vector.as_deref() {
            if let Err(error) = self.storage().index_embedding(&record, vector).await {
                if !kind_changed {
                    // The old vector was replaced lazily (same collection), so restoring
                    // the previous record keeps the store consistent with its JSON.
                    if let Err(restore) = self.storage().put_record(&previous).await {
                        tracing::warn!(target: TARGET, %restore, id = %record.id,
                            "failed to restore a memory record after an embedding index failure");
                    }
                    return Err(error);
                }
                // A kind change deleted the previous kind's vector before the new one was
                // indexed, so restoring the JSON row alone would leave the record without
                // a vector. Put the previous record back and re-index it, so a failed kind
                // change is undone instead of leaving the new content persisted but
                // unsearchable.
                let context = match self.restore_indexed(&previous, embedding, options).await {
                    Ok(()) => format!(
                        "re-indexing memory record `{}` as `{}` failed after changing its \
                             kind from `{}`; the previous record and its vector were restored",
                        record.id, record.kind, previous.kind
                    ),
                    Err(restore) => {
                        tracing::warn!(target: TARGET, %restore, id = %record.id,
                                "failed to restore a memory record after a kind-changing re-index failure");
                        format!(
                            "re-indexing memory record `{}` as `{}` failed after changing its \
                                 kind from `{}`, and restoring the previous record failed too \
                                 ({restore}); the stored record has no vector and is only \
                                 keyword-searchable",
                            record.id, record.kind, previous.kind
                        )
                    }
                };
                return Err(error.with_context(context));
            }
        }
        Ok(record)
    }

    /// Restores `previous` after a failed re-index: its record row and its vector.
    ///
    /// The stored record snapshot never carries an embedding (the vector collection is the
    /// authoritative copy), so the vector is regenerated by embedding
    /// `previous.content` again before it is indexed. The row is written first, so a
    /// failing re-embedding still leaves the previous JSON in place; the first error is
    /// returned for the caller to report.
    async fn restore_indexed(
        &self,
        previous: &MemoryRecord,
        embedder: Option<&Arc<dyn Embedding>>,
        options: &CallOptions,
    ) -> Result<()> {
        self.storage().put_record(previous).await?;
        if !self.has_embedding_with(embedder) {
            return Ok(());
        }
        let vector = self
            .embed_text_with(embedder, &previous.content, options)
            .await?;
        self.storage().index_embedding(previous, &vector).await
    }

    /// Deletes one record and its embedding, returning whether it existed.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an invalid scope and the storage error when the
    /// delete fails.
    pub async fn delete(&self, scope: &MemoryScope, id: &str) -> Result<bool> {
        scope.validate()?;
        let lock = self.write_locks.lock_for(scope);
        let _guard = lock.lock().await;
        let Some(record) = self.storage().get_record(scope, id).await? else {
            return Ok(false);
        };
        self.storage().delete_record(&record).await
    }

    /// Deletes every record matching `filter`, returning how many were removed.
    ///
    /// Only memory records are cleared; variables and conversation messages are owned
    /// by the facade and are untouched. An empty `filter.kinds` means every memory kind,
    /// which excludes the [`MemoryType::Variable`](crate::record::MemoryType::Variable)
    /// projections: they are removed only by a filter that names the kind explicitly
    /// (their KV entries stay either way).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Cancelled`]/[`Error::Timeout`] when the call is cancelled or
    /// exceeds its deadline and the storage error when a delete fails.
    pub async fn clear(&self, filter: &MemoryFilter, options: &CallOptions) -> Result<usize> {
        options.check()?;
        let guard = filter.scope_prefix().map(|(user_id, scope_id)| {
            self.write_locks
                .lock_for(&MemoryScope::new(user_id, scope_id))
        });
        let _guard = match guard {
            Some(lock) => Some(lock.lock_owned().await),
            None => None,
        };
        let records = self.storage().list_records(filter).await?;
        let mut removed = 0;
        for record in records {
            options.check()?;
            if !super::includes_record(filter, &record) {
                continue;
            }
            if self.storage().delete_record(&record).await? {
                removed += 1;
            }
        }
        Ok(removed)
    }

    /// Lists every record matching `filter`, newest first, without scoring.
    ///
    /// An empty `filter.kinds` lists every memory kind except
    /// [`MemoryType::Variable`](crate::record::MemoryType::Variable) projections, which
    /// belong to the variable namespace and appear only when the filter names the kind.
    ///
    /// # Errors
    ///
    /// Returns the storage error when the listing fails.
    pub async fn list(&self, filter: &MemoryFilter) -> Result<Vec<MemoryRecord>> {
        let records = self.storage().list_records(filter).await?;
        Ok(records
            .into_iter()
            .filter(|record| super::includes_record(filter, record))
            .collect())
    }

    /// Builds the record a [`NewMemory`] input describes.
    fn build_record(
        &self,
        scope: &MemoryScope,
        memory: &NewMemory,
        now: DateTime<Utc>,
    ) -> MemoryRecord {
        let timestamp = memory.timestamp.unwrap_or(now);
        MemoryRecord {
            id: new_memory_id(&scope.user_id, now),
            content: memory.content.clone(),
            kind: memory.kind,
            importance: memory.importance.unwrap_or(self.config.importance_default),
            timestamp,
            updated_at: timestamp,
            last_access: None,
            access_count: 0,
            metadata: memory.metadata.clone(),
            embedding: None,
            source: memory.source.clone(),
            scope: scope.clone(),
        }
    }

    /// Embeds, dedupes, and persists one prepared record, recording the outcome.
    async fn write_record(
        &self,
        mut record: MemoryRecord,
        options: &AddOptions,
        now: DateTime<Utc>,
        outcome: &mut AddOutcome,
        embedder: Option<&Arc<dyn Embedding>>,
    ) -> Result<()> {
        let embeds = self.has_embedding_with(embedder) && !options.skip_embedding;
        let embedding = if embeds {
            Some(
                self.embed_text_with(embedder, &record.content, &options.call)
                    .await?,
            )
        } else {
            None
        };
        record.embedding = embedding.clone();

        let policy = options.dedupe.unwrap_or(self.config.dedupe.policy);
        let dedupe = self.config.dedupe.enabled && options.dedupe != Some(DedupePolicy::Add);
        let candidates = if dedupe {
            let scan = self
                .dedupe_candidates(&record, embedding.as_deref(), options, embedder)
                .await?;
            if let Some(reason) = scan.degraded {
                // The first failure of a batch is the one worth reporting: it names both
                // the record that went unchecked and the reason every later item shares.
                if outcome.degraded.is_none() {
                    outcome.degraded = Some(reason);
                }
            }
            scan.candidates
        } else {
            Vec::new()
        };
        let Some(found) = find_duplicate(&record, &candidates, &self.config.dedupe) else {
            self.persist(&record, embedding.as_deref()).await?;
            outcome.added.push(record);
            return Ok(());
        };
        let existing = candidates[found.index].record.clone();
        match policy {
            DedupePolicy::Add => {
                self.persist(&record, embedding.as_deref()).await?;
                outcome.added.push(record);
            }
            DedupePolicy::Skip => outcome.skipped.push(record.id),
            DedupePolicy::Merge => {
                let merged = merge_records(&existing, &record, now);
                let merged_embedding = if embeds {
                    Some(
                        self.embed_text_with(embedder, &merged.content, &options.call)
                            .await?,
                    )
                } else {
                    None
                };
                if merged.kind != existing.kind {
                    self.storage().delete_embedding(&existing).await?;
                }
                self.persist(&merged, merged_embedding.as_deref()).await?;
                outcome.merged.push(merged);
            }
            DedupePolicy::Replace => {
                self.storage().delete_record(&existing).await?;
                record.id = existing.id;
                record.embedding = None;
                self.persist(&record, embedding.as_deref()).await?;
                outcome.merged.push(record);
            }
        }
        Ok(())
    }

    /// Collects the records a duplicate check of `incoming` compares against.
    ///
    /// A semantic scan is used when the incoming content was embedded; otherwise the
    /// scope is scanned and candidates are keyword-only. A failing scan degrades to "no
    /// candidates" with a warning (dedupe is best-effort, Python parity) and reports the
    /// reason in [`DedupeScan::degraded`], so the caller can surface it in
    /// [`AddOutcome::degraded`] instead of writing an unchecked duplicate as if the scope
    /// had no duplicates. Cancellation/timeout never degrade: they abort the write.
    async fn dedupe_candidates(
        &self,
        incoming: &MemoryRecord,
        vector: Option<&[f32]>,
        options: &AddOptions,
        embedder: Option<&Arc<dyn Embedding>>,
    ) -> Result<DedupeScan> {
        let search = SearchOptions {
            top_k: self.config.dedupe.candidate_top_k,
            threshold: 0.0,
            kinds: self.config.searchable_kinds.clone(),
            mode: if vector.is_some() {
                SearchMode::Hybrid
            } else {
                SearchMode::Keyword
            },
            token_budget: None,
            call: options.call.clone(),
        };
        match self
            .recall(
                &incoming.scope,
                &incoming.content,
                vector,
                &search,
                false,
                embedder,
            )
            .await
        {
            Ok(hits) => Ok(DedupeScan {
                candidates: hits
                    .into_iter()
                    .map(|hit| Candidate::new(hit.record, Some(hit.similarity)))
                    .collect(),
                degraded: None,
            }),
            Err(error @ (Error::Cancelled | Error::Timeout(_))) => Err(error),
            Err(error) => {
                tracing::warn!(
                    target: TARGET,
                    %error,
                    id = %incoming.id,
                    "duplicate candidate scan failed; writing without duplicate detection"
                );
                Ok(DedupeScan {
                    candidates: Vec::new(),
                    degraded: Some(format!(
                        "duplicate candidate scan failed, so memory record `{}` was written without a duplicate check: {error}",
                        incoming.id
                    )),
                })
            }
        }
    }

    /// Persists `record` and, when `embedding` is present, its vector.
    ///
    /// The record row is written first so a vector hit always resolves to a record. If
    /// indexing fails the row is removed again, so no half-indexed record remains.
    /// Shared with the variable projection (`super::variables`).
    pub(super) async fn persist(
        &self,
        record: &MemoryRecord,
        embedding: Option<&[f32]>,
    ) -> Result<()> {
        self.storage().put_record(record).await?;
        let Some(vector) = embedding else {
            return Ok(());
        };
        if let Err(error) = self.storage().index_embedding(record, vector).await {
            if let Err(rollback) = self.storage().delete_record(record).await {
                tracing::warn!(target: TARGET, %rollback, id = %record.id,
                    "failed to roll back a memory record after an embedding index failure");
            }
            return Err(error);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use async_trait::async_trait;
    use chrono::{DateTime, Utc};
    use openjiuwen_common::error::{Error, Result};
    use openjiuwen_retrieval::embedding::{Embedding, HashEmbedding};

    use super::super::{MemoryEngine, MemoryEngineBuilder};
    use crate::config::MemoryEngineConfig;
    use crate::record::{
        AddOptions, CallOptions, DedupePolicy, MemoryFilter, MemoryScope, MemoryType, MemoryUpdate,
        NewMemory,
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

    #[tokio::test]
    async fn add_get_update_delete_and_clear_round_trip() {
        let engine = engine();
        let scope = scope();
        let outcome = engine
            .add(
                &scope,
                NewMemory::semantic("Alice prefers dark mode").with_timestamp(ts(0)),
                &AddOptions::default(),
            )
            .await
            .expect("add");
        assert_eq!(outcome.added.len(), 1);
        assert_eq!(outcome.len(), 1);
        let id = outcome.added[0].id.clone();
        assert_eq!(outcome.added[0].timestamp, ts(0));
        assert_eq!(outcome.added[0].updated_at, ts(0));

        let fetched = engine
            .get(&scope, &id)
            .await
            .expect("get")
            .expect("present");
        assert_eq!(fetched.content, "Alice prefers dark mode");
        assert_eq!(
            engine
                .list(&MemoryFilter::for_scope(&scope))
                .await
                .expect("list")
                .len(),
            1
        );

        let updated = engine
            .update(
                &scope,
                &id,
                MemoryUpdate::content("Alice prefers light mode")
                    .with_importance(0.9)
                    .with_metadata("theme", serde_json::json!("light")),
                &CallOptions::new(),
            )
            .await
            .expect("update");
        assert_eq!(updated.id, id);
        assert_eq!(updated.content, "Alice prefers light mode");
        assert!((updated.importance - 0.9).abs() < f32::EPSILON);
        assert_eq!(
            updated.metadata.get("theme"),
            Some(&serde_json::json!("light"))
        );
        assert!(updated.updated_at > updated.timestamp);

        assert!(engine.delete(&scope, &id).await.expect("delete"));
        assert!(!engine.delete(&scope, &id).await.expect("delete again"));
        assert!(engine.get(&scope, &id).await.expect("get").is_none());

        for content in ["first", "second"] {
            engine
                .add(&scope, NewMemory::semantic(content), &AddOptions::default())
                .await
                .expect("add");
        }
        assert_eq!(
            engine
                .clear(&MemoryFilter::for_scope(&scope), &CallOptions::new())
                .await
                .expect("clear"),
            2
        );
        assert_eq!(
            engine
                .list(&MemoryFilter::for_scope(&scope))
                .await
                .expect("list")
                .len(),
            0
        );
    }

    #[tokio::test]
    async fn merge_keeps_the_existing_id_metadata_and_timestamp() {
        let engine = engine();
        let scope = scope();
        let first = engine
            .add(
                &scope,
                NewMemory::semantic("Alice prefers dark mode")
                    .with_timestamp(ts(0))
                    .with_importance(0.4)
                    .with_metadata("theme", serde_json::json!("dark")),
                &AddOptions::default(),
            )
            .await
            .expect("add");
        let id = first.added[0].id.clone();
        assert_eq!(first.added[0].metadata.len(), 1);
        let second = engine
            .add(
                &scope,
                NewMemory::semantic("alice   prefers dark mode")
                    .with_timestamp(ts(10))
                    .with_importance(0.9)
                    .with_metadata("language", serde_json::json!("en")),
                &AddOptions::default(),
            )
            .await
            .expect("add duplicate");
        assert_eq!(second.added.len(), 0);
        assert_eq!(second.merged.len(), 1);
        let merged = &second.merged[0];
        assert_eq!(merged.id, id, "merge keeps the existing id");
        assert_eq!(merged.timestamp, ts(0), "merge keeps the creation time");
        assert!((merged.importance - 0.9).abs() < f32::EPSILON);
        assert_eq!(
            merged.metadata.get("theme"),
            Some(&serde_json::json!("dark"))
        );
        assert_eq!(
            merged.metadata.get("language"),
            Some(&serde_json::json!("en"))
        );
        assert_eq!(
            engine
                .list(&MemoryFilter::for_scope(&scope))
                .await
                .expect("list")
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn skip_drops_the_incoming_record() {
        let engine = engine();
        let scope = scope();
        engine
            .add(
                &scope,
                NewMemory::semantic("duplicate content"),
                &AddOptions::default(),
            )
            .await
            .expect("add");
        let outcome = engine
            .add(
                &scope,
                NewMemory::semantic("duplicate content"),
                &AddOptions::default().with_dedupe(DedupePolicy::Skip),
            )
            .await
            .expect("skip");
        assert_eq!(outcome.skipped.len(), 1);
        assert_eq!(outcome.merged.len() + outcome.added.len(), 0);
        assert_eq!(
            engine
                .list(&MemoryFilter::for_scope(&scope))
                .await
                .expect("list")
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn replace_writes_the_incoming_content_under_the_existing_id() {
        let engine = engine();
        let scope = scope();
        let first = engine
            .add(
                &scope,
                NewMemory::semantic("Alice prefers dark mode").with_timestamp(ts(0)),
                &AddOptions::default(),
            )
            .await
            .expect("add");
        let existing_id = first.added[0].id.clone();
        let outcome = engine
            .add(
                &scope,
                NewMemory::semantic("Alice prefers dark mode!").with_timestamp(ts(5)),
                &AddOptions::default().with_dedupe(DedupePolicy::Replace),
            )
            .await
            .expect("replace");
        assert_eq!(outcome.merged.len(), 1);
        assert_eq!(outcome.merged[0].id, existing_id);
        assert_eq!(outcome.merged[0].content, "Alice prefers dark mode!");
        let stored = engine
            .list(&MemoryFilter::for_scope(&scope))
            .await
            .expect("list");
        assert_eq!(stored.len(), 1);
        assert_eq!(stored[0].id, existing_id);
        assert_eq!(stored[0].content, "Alice prefers dark mode!");
    }

    #[tokio::test]
    async fn add_policy_stores_the_duplicate_as_a_second_record() {
        let engine = engine();
        let scope = scope();
        for _ in 0..2 {
            engine
                .add(
                    &scope,
                    NewMemory::semantic("duplicate content"),
                    &AddOptions::default().with_dedupe(DedupePolicy::Add),
                )
                .await
                .expect("add");
        }
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
    async fn disabled_dedupe_stores_identical_memories() {
        let config = MemoryEngineConfig {
            dedupe: crate::config::DedupeConfig {
                enabled: false,
                ..Default::default()
            },
            ..Default::default()
        };
        let engine = MemoryEngine::new(config, MemoryStores::in_memory()).expect("engine");
        let scope = scope();
        for _ in 0..2 {
            engine
                .add(
                    &scope,
                    NewMemory::semantic("duplicate content"),
                    &AddOptions::default(),
                )
                .await
                .expect("add");
        }
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
    async fn add_many_dedupes_within_the_batch() {
        let engine = engine();
        let scope = scope();
        let outcome = engine
            .add_many(
                &scope,
                vec![
                    NewMemory::semantic("batch duplicate"),
                    NewMemory::semantic("batch duplicate"),
                ],
                &AddOptions::default(),
            )
            .await
            .expect("add many");
        assert_eq!(outcome.added.len(), 1);
        assert_eq!(outcome.merged.len(), 1);
        assert_eq!(
            engine
                .list(&MemoryFilter::for_scope(&scope))
                .await
                .expect("list")
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn blank_and_variable_inputs_are_rejected() {
        let engine = engine();
        let scope = scope();
        let error = engine
            .add(&scope, NewMemory::semantic("   "), &AddOptions::default())
            .await
            .expect_err("blank");
        assert!(matches!(error, Error::Validation(_)), "{error:?}");
        let error = engine
            .add(
                &scope,
                NewMemory::new("a variable", MemoryType::Variable),
                &AddOptions::default(),
            )
            .await
            .expect_err("variable");
        assert!(matches!(error, Error::Validation(_)), "{error:?}");
        assert_eq!(
            engine
                .list(&MemoryFilter::for_scope(&scope))
                .await
                .expect("list")
                .len(),
            0
        );
    }

    #[tokio::test]
    async fn cancelled_add_writes_nothing() {
        let engine = engine();
        let scope = scope();
        let token = tokio_util::sync::CancellationToken::new();
        token.cancel();
        let error = engine
            .add(
                &scope,
                NewMemory::semantic("never stored"),
                &AddOptions::default().with_cancellation(token),
            )
            .await
            .expect_err("cancelled");
        assert!(matches!(error, Error::Cancelled), "{error:?}");
        assert_eq!(
            engine
                .list(&MemoryFilter::for_scope(&scope))
                .await
                .expect("list")
                .len(),
            0
        );
    }

    #[tokio::test]
    async fn update_of_an_unknown_id_is_not_found() {
        let engine = engine();
        let error = engine
            .update(
                &scope(),
                "missing-id",
                MemoryUpdate::content("content"),
                &CallOptions::new(),
            )
            .await
            .expect_err("not found");
        match error {
            Error::NotFound(message) => assert!(message.contains("missing-id"), "{message}"),
            other => panic!("expected NotFound, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn update_merges_metadata_and_keeps_untouched_fields() {
        let engine = engine();
        let scope = scope();
        let added = engine
            .add(
                &scope,
                NewMemory::semantic("profile note")
                    .with_timestamp(ts(0))
                    .with_metadata("a", serde_json::json!(1)),
                &AddOptions::default(),
            )
            .await
            .expect("add");
        let id = added.added[0].id.clone();
        let updated = engine
            .update(
                &scope,
                &id,
                MemoryUpdate::default().with_metadata("b", serde_json::json!(2)),
                &CallOptions::new(),
            )
            .await
            .expect("update");
        assert_eq!(updated.content, "profile note");
        assert_eq!(updated.timestamp, ts(0));
        assert_eq!(updated.metadata.get("a"), Some(&serde_json::json!(1)));
        assert_eq!(updated.metadata.get("b"), Some(&serde_json::json!(2)));
    }

    #[tokio::test]
    async fn concurrent_skip_writes_leave_exactly_one_record() {
        let engine = engine();
        let scope = scope();
        let mut handles = Vec::new();
        for _ in 0..8 {
            let engine = Arc::clone(&engine);
            let scope = scope.clone();
            handles.push(tokio::spawn(async move {
                engine
                    .add(
                        &scope,
                        NewMemory::semantic("racy duplicate"),
                        &AddOptions::default().with_dedupe(DedupePolicy::Skip),
                    )
                    .await
                    .expect("add")
            }));
        }
        let mut added = 0;
        let mut skipped = 0;
        for handle in handles {
            let outcome = handle.await.expect("task");
            added += outcome.added.len();
            skipped += outcome.skipped.len();
        }
        assert_eq!(added, 1);
        assert_eq!(skipped, 7);
        assert_eq!(
            engine
                .list(&MemoryFilter::for_scope(&scope))
                .await
                .expect("list")
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn concurrent_add_policy_writes_keep_every_record() {
        let engine = engine();
        let scope = scope();
        let mut handles = Vec::new();
        for _ in 0..8 {
            let engine = Arc::clone(&engine);
            let scope = scope.clone();
            handles.push(tokio::spawn(async move {
                engine
                    .add(
                        &scope,
                        NewMemory::semantic("racy duplicate"),
                        &AddOptions::default().with_dedupe(DedupePolicy::Add),
                    )
                    .await
                    .expect("add")
            }));
        }
        for handle in handles {
            handle.await.expect("task");
        }
        assert_eq!(
            engine
                .list(&MemoryFilter::for_scope(&scope))
                .await
                .expect("list")
                .len(),
            8
        );
    }

    #[tokio::test]
    async fn a_failed_embedding_leaves_no_record() {
        struct FailingEmbedding;

        #[async_trait]
        impl Embedding for FailingEmbedding {
            fn model_name(&self) -> &'static str {
                "failing"
            }

            async fn embed_query(&self, _text: &str) -> Result<Vec<f32>> {
                Err(Error::llm("embedding backend unavailable"))
            }

            async fn embed_documents(
                &self,
                _texts: &[String],
                _batch_size: Option<usize>,
            ) -> Result<Vec<Vec<f32>>> {
                Err(Error::llm("embedding backend unavailable"))
            }
        }

        let engine = MemoryEngineBuilder::new(MemoryEngineConfig::default())
            .embedding(Arc::new(FailingEmbedding))
            .build()
            .expect("engine");
        let scope = scope();
        let error = engine
            .add(
                &scope,
                NewMemory::semantic("cannot be embedded"),
                &AddOptions::default(),
            )
            .await
            .expect_err("embedding failure");
        assert!(matches!(error, Error::Llm(_)), "{error:?}");
        assert_eq!(
            engine
                .list(&MemoryFilter::for_scope(&scope))
                .await
                .expect("list")
                .len(),
            0
        );
    }

    #[tokio::test]
    async fn embedding_engine_merges_duplicates_and_keeps_one_vector() {
        let engine = embedding_engine();
        let scope = scope();
        let first = engine
            .add(
                &scope,
                NewMemory::semantic("shared preference"),
                &AddOptions::default(),
            )
            .await
            .expect("add");
        let id = first.added[0].id.clone();
        let second = engine
            .add(
                &scope,
                NewMemory::semantic("shared preference"),
                &AddOptions::default(),
            )
            .await
            .expect("add duplicate");
        assert_eq!(second.merged.len(), 1);
        let hits = engine
            .search(
                &scope,
                "shared preference",
                &crate::record::SearchOptions::default().with_threshold(0.0),
            )
            .await
            .expect("search");
        assert!(hits.iter().any(|hit| hit.record.id == id));
        assert_eq!(
            engine
                .list(&MemoryFilter::for_scope(&scope))
                .await
                .expect("list")
                .len(),
            1
        );
    }
}
