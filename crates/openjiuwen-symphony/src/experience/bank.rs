//! The experience bank: the authoritative in-memory catalog with write-through store.
//!
//! [`ExperienceBank`](crate::experience::ExperienceBank) mirrors Python's
//! `ExperienceBank` in role but not in storage: it keeps every item in memory, assigns
//! `exp_NNNN` ids, dedupes on
//! [`ExperienceItem::dedupe_key`](crate::experience::ExperienceItem::dedupe_key), and
//! forwards each visible change to its
//! [`ExperienceStore`](crate::experience::ExperienceStore). Embeddings are computed on
//! demand by the optional
//! [`openjiuwen_retrieval::Embedding`] instead of being persisted with the items, so a
//! reopened bank is immediately searchable again (the vectors are re-derived from the
//! item text).
//!
//! Locking discipline: the item map lives behind a [`std::sync::RwLock`]. The guard is
//! only held for map mutations and snapshot clones, never across an `.await`, so store
//! and embedding I/O can never block the runtime's reactor. Writes persist first and
//! only then become visible in memory, so a store failure leaves the in-memory catalog
//! untouched.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

use openjiuwen_retrieval::Embedding;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::{validation, Result};
use crate::experience::ranking::{
    rank_by_embedding, rank_lexically, ExperienceHit, ExperienceRanking, SearchOptions,
};
use crate::experience::store::{ExperienceStore, JsonFileExperienceStore, MemoryExperienceStore};
use crate::experience::{merge_item, ordinal_of, ExperienceItem, ID_DIGITS, ID_PREFIX};

/// The outcome of [`ExperienceBank::add`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct AddOutcome {
    /// The id of the stored item: the generated id, the caller's id, or the id of the
    /// item a duplicate was merged into.
    pub id: String,
    /// Whether the item was merged into an existing record instead of inserted.
    pub merged: bool,
}

impl AddOutcome {
    /// An outcome for a freshly inserted item.
    #[must_use]
    pub fn inserted(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            merged: false,
        }
    }

    /// An outcome for an item merged into the existing record `id`.
    #[must_use]
    pub fn merged_into(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            merged: true,
        }
    }
}

/// The experience catalog: CRUD, generated ids, dedupe, and search.
///
/// Reads are served from memory; every mutation is persisted first (when a store is
/// configured) and only then applied in memory.
pub struct ExperienceBank {
    /// The persistence backend; every visible mutation is forwarded to it.
    store: Arc<dyn ExperienceStore>,
    /// The authoritative items, keyed and ordered by id.
    items: RwLock<BTreeMap<String, ExperienceItem>>,
    /// The next generated-id ordinal; `exp_NNNN` is rendered from it.
    next_ordinal: AtomicU64,
    /// The optional embedder; without one, search ranks lexically.
    embedding: Option<Arc<dyn Embedding>>,
}

impl std::fmt::Debug for ExperienceBank {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ExperienceBank")
            .field("items", &self.len())
            .field(
                "embedding",
                &self
                    .embedding
                    .as_ref()
                    .map(|embedding| embedding.model_name()),
            )
            .finish_non_exhaustive()
    }
}

impl ExperienceBank {
    /// Builds an empty bank backed by a fresh in-memory store.
    #[must_use]
    pub fn new() -> Self {
        Self::with_store(Arc::new(MemoryExperienceStore::new()))
    }

    /// Builds an empty bank that writes through to `store`.
    ///
    /// The in-memory catalog starts empty; use [`Self::load`] to hydrate it from the
    /// store's contents.
    #[must_use]
    pub fn with_store(store: Arc<dyn ExperienceStore>) -> Self {
        Self {
            store,
            items: RwLock::new(BTreeMap::new()),
            next_ordinal: AtomicU64::new(1),
            embedding: None,
        }
    }

    /// Builds a bank over `store` and hydrates it from the store's contents.
    ///
    /// The generated-id counter continues past the highest existing `exp_NNNN` id, so a
    /// reopened bank never reuses an id.
    ///
    /// # Errors
    ///
    /// Returns the backend's error when the store cannot be listed.
    pub async fn load(store: Arc<dyn ExperienceStore>) -> Result<Self> {
        let items = store.list().await?;
        let bank = Self::with_store(store);
        let mut next_ordinal = 1_u64;
        {
            let mut entries = bank.write_items();
            for item in items {
                if let Some(ordinal) = ordinal_of(&item.id) {
                    next_ordinal = next_ordinal.max(ordinal.saturating_add(1));
                }
                entries.insert(item.id.clone(), item);
            }
        }
        bank.next_ordinal.store(next_ordinal, Ordering::SeqCst);
        Ok(bank)
    }

    /// Opens (or creates) the JSON document at `path` and hydrates the bank from it.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::Error::External`] when the document cannot be read.
    pub async fn open_json_file(path: impl AsRef<Path>) -> Result<Self> {
        let store = JsonFileExperienceStore::open(path).await?;
        Self::load(Arc::new(store)).await
    }

    /// Opens (or creates) the `SQLite` database at `url` and hydrates the bank from it.
    ///
    /// # Errors
    ///
    /// Returns the foundation backend's error when the database cannot be opened.
    #[cfg(feature = "sqlite")]
    pub async fn open_sqlite(url: &str) -> Result<Self> {
        let store = crate::experience::SqliteExperienceStore::connect(url).await?;
        Self::load(Arc::new(store)).await
    }

    /// Configures the embedding model used for semantic search.
    ///
    /// Without a model, [`Self::search`] ranks lexically. Item embeddings are never
    /// persisted, so a bank loaded from a store re-embeds the items it searches.
    #[must_use]
    pub fn with_embedding(mut self, embedding: Arc<dyn Embedding>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// The persistence backend.
    #[must_use]
    pub fn store(&self) -> &Arc<dyn ExperienceStore> {
        &self.store
    }

    /// The configured embedding model, if any.
    #[must_use]
    pub fn embedding(&self) -> Option<&Arc<dyn Embedding>> {
        self.embedding.as_ref()
    }

    /// The number of items in the catalog.
    #[must_use]
    pub fn len(&self) -> usize {
        self.read_items().len()
    }

    /// Reports whether the catalog is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.read_items().is_empty()
    }

    /// Returns a copy of the item stored under `id`, or `None`.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<ExperienceItem> {
        self.read_items().get(id).cloned()
    }

    /// Lists every item, sorted by id.
    #[must_use]
    pub fn list(&self) -> Vec<ExperienceItem> {
        self.read_items().values().cloned().collect()
    }

    /// Reports whether `skill_ids` is the skill set of at least one item.
    ///
    /// The comparison is set-based: order and duplicate ids do not matter, so
    /// `["b", "a"]` matches an item with `["a", "b"]`. This is Python's
    /// `ExperienceBank.exist`.
    #[must_use]
    pub fn exists_for_skills(&self, skill_ids: &[String]) -> bool {
        let requested: std::collections::BTreeSet<&str> =
            skill_ids.iter().map(String::as_str).collect();
        self.read_items()
            .values()
            .any(|item| item.skill_set() == requested)
    }

    /// Generates the next `exp_NNNN` id, advancing the counter.
    ///
    /// The id is not checked against the catalog; [`Self::add`] skips ids that are
    /// already taken. Callers that mint ids up front and insert them later should
    /// prefer letting [`Self::add`] generate the id.
    #[must_use]
    pub fn generate_id(&self) -> String {
        let ordinal = self.next_ordinal.fetch_add(1, Ordering::SeqCst);
        let width = ID_DIGITS;
        format!("{ID_PREFIX}{ordinal:0width$}")
    }

    /// Adds `item`, merging it into an existing record when the dedupe key matches.
    ///
    /// - A blank id is replaced by the next free `exp_NNNN`; an explicit id is kept as
    ///   given (trimmed).
    /// - A duplicate (same normalized pattern and same skill set) is merged: the
    ///   success counters add up, the example lists union in first-seen order, the later
    ///   `last_hit_at` wins, and the stored embedding is kept unless it was empty. The
    ///   stored id is preserved and reported in [`AddOutcome::id`].
    /// - An explicit id that is already taken under a different dedupe key is replaced.
    ///
    /// The item is persisted before it becomes visible in memory.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::Error::Validation`] naming `query_pattern` when the
    /// normalized pattern is blank, or the backend's error when the write fails.
    pub async fn add(&self, item: ExperienceItem) -> Result<AddOutcome> {
        if item.normalized_pattern().is_empty() {
            return Err(validation("query_pattern", "must not be blank"));
        }
        let dedupe_key = item.dedupe_key();
        if let Some(mut existing) = self.duplicate(&dedupe_key) {
            merge_item(&mut existing, item);
            self.store.put(&existing).await?;
            let merged = existing;
            self.note_ordinal(&merged.id);
            self.write_items().insert(merged.id.clone(), merged.clone());
            return Ok(AddOutcome::merged_into(merged.id));
        }

        let mut item = item;
        item.id = if item.id.trim().is_empty() {
            self.fresh_id()
        } else {
            item.id.trim().to_string()
        };
        self.store.put(&item).await?;
        self.note_ordinal(&item.id);
        let id = item.id.clone();
        self.write_items().insert(item.id.clone(), item);
        Ok(AddOutcome::inserted(id))
    }

    /// Adds every item in `items`, returning one outcome per item in input order.
    ///
    /// A failure part-way through leaves the earlier items persisted (the batch is not
    /// transactional); their outcomes are lost with the error, but the store and the
    /// catalog agree.
    ///
    /// # Errors
    ///
    /// Returns the first item's validation error or the backend's error.
    pub async fn add_batch(&self, items: Vec<ExperienceItem>) -> Result<Vec<AddOutcome>> {
        let mut outcomes = Vec::with_capacity(items.len());
        for item in items {
            outcomes.push(self.add(item).await?);
        }
        Ok(outcomes)
    }

    /// Removes `id`, returning whether it existed.
    ///
    /// # Errors
    ///
    /// Returns the backend's error when the delete fails.
    pub async fn remove(&self, id: &str) -> Result<bool> {
        let removed = self.write_items().remove(id).is_some();
        if removed {
            self.store.remove(id).await?;
        }
        Ok(removed)
    }

    /// Removes every item from the catalog and the store.
    ///
    /// The generated-id counter is not reset, so ids of cleared items are never reused
    /// within the lifetime of this bank.
    ///
    /// # Errors
    ///
    /// Returns the backend's error when the clear fails.
    pub async fn clear(&self) -> Result<()> {
        self.store.clear().await?;
        self.write_items().clear();
        Ok(())
    }

    /// Searches the bank, returning at most `top_k` hits (`0` means unlimited).
    ///
    /// Hits are ranked by embedding similarity when an embedder is configured and the
    /// query is not blank, and lexically otherwise. Ties break on id ascending.
    ///
    /// # Errors
    ///
    /// Returns the embedding model's error when the query cannot be embedded.
    pub async fn search(&self, query: &str, top_k: usize) -> Result<Vec<ExperienceHit>> {
        self.search_with_options(query, &SearchOptions::new(top_k))
            .await
    }

    /// Searches with the ranking and skill filters of `options`.
    ///
    /// A blank query with an embedder configured falls back to lexical ranking, because
    /// embedding providers reject blank input.
    ///
    /// # Errors
    ///
    /// Returns the embedding model's error when the query or an item cannot be embedded.
    pub async fn search_with_options(
        &self,
        query: &str,
        options: &SearchOptions,
    ) -> Result<Vec<ExperienceHit>> {
        let mut items = self.list();
        if items.is_empty() {
            return Ok(Vec::new());
        }
        let Some(embedding) = &self.embedding else {
            return Ok(rank_lexically(&items, query, options));
        };
        if query.trim().is_empty() {
            return Ok(rank_lexically(&items, query, options));
        }
        let query_vector = embedding.embed_query(query).await?;
        embed_missing(embedding, &mut items, query_vector.len()).await?;
        Ok(rank_by_embedding(&items, &query_vector, options))
    }

    /// Searches with a [`ExperienceRanking`] (no skill filter).
    ///
    /// # Errors
    ///
    /// Returns the embedding model's error when the query or an item cannot be embedded.
    pub async fn search_with_ranking(
        &self,
        query: &str,
        ranking: &ExperienceRanking,
    ) -> Result<Vec<ExperienceHit>> {
        self.search_with_options(query, &ranking.options()).await
    }

    /// Returns the distinct skill ids of the hits, in first-seen order.
    ///
    /// This is Python's `search_with_skill_ids` convenience: the caller wants "which
    /// skills serve this query", not the whole items.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::search`].
    pub async fn search_with_skill_ids(&self, query: &str, top_k: usize) -> Result<Vec<String>> {
        let mut skill_ids = Vec::new();
        for hit in self.search(query, top_k).await? {
            for skill_id in hit.item.skill_ids {
                if !skill_ids.contains(&skill_id) {
                    skill_ids.push(skill_id);
                }
            }
        }
        Ok(skill_ids)
    }

    /// Returns the stored item whose dedupe key matches `dedupe_key`.
    ///
    /// The scan is linear and hashes each candidate lazily; the catalog is a distilled
    /// knowledge base of tens to hundreds of entries, not a document store.
    fn duplicate(&self, dedupe_key: &str) -> Option<ExperienceItem> {
        self.read_items()
            .values()
            .find(|item| item.dedupe_key() == dedupe_key)
            .cloned()
    }

    /// Generates an id that is not already taken.
    fn fresh_id(&self) -> String {
        loop {
            let candidate = self.generate_id();
            if !self.read_items().contains_key(&candidate) {
                return candidate;
            }
        }
    }

    /// Keeps the generated-id counter ahead of every `exp_NNNN` id ever seen.
    fn note_ordinal(&self, id: &str) {
        if let Some(ordinal) = ordinal_of(id) {
            self.next_ordinal
                .fetch_max(ordinal.saturating_add(1), Ordering::SeqCst);
        }
    }

    /// Locks the catalog for reading, recovering from a poisoned lock.
    fn read_items(&self) -> RwLockReadGuard<'_, BTreeMap<String, ExperienceItem>> {
        self.items.read().unwrap_or_else(PoisonError::into_inner)
    }

    /// Locks the catalog for writing, recovering from a poisoned lock.
    fn write_items(&self) -> RwLockWriteGuard<'_, BTreeMap<String, ExperienceItem>> {
        self.items.write().unwrap_or_else(PoisonError::into_inner)
    }
}

impl Default for ExperienceBank {
    fn default() -> Self {
        Self::new()
    }
}

/// Embeds every item whose stored embedding does not match `dimension`.
///
/// Items with blank text are left alone: the provider rejects blank input, and a
/// zero-length vector simply ranks at `0.0`.
async fn embed_missing(
    embedding: &Arc<dyn Embedding>,
    items: &mut [ExperienceItem],
    dimension: usize,
) -> Result<()> {
    let mut pending: Vec<(usize, String)> = Vec::new();
    for (index, item) in items.iter().enumerate() {
        if item.embedding.len() == dimension && dimension > 0 {
            continue;
        }
        let text = item.search_text();
        if text.is_empty() {
            continue;
        }
        pending.push((index, text));
    }
    if pending.is_empty() {
        return Ok(());
    }
    let texts: Vec<String> = pending.iter().map(|(_, text)| text.clone()).collect();
    let vectors = embedding.embed_documents(&texts, None).await?;
    for ((index, _), vector) in pending.into_iter().zip(vectors) {
        items[index].embedding = vector;
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn generated_ids_skip_taken_ones() {
        let bank = ExperienceBank::new();
        bank.add(ExperienceItem::new("first")).await.expect("add");
        bank.add(
            ExperienceItem::new("pinned")
                .with_id("exp_0002")
                .with_skill_ids(["pinned"]),
        )
        .await
        .expect("add pinned");
        let outcome = bank
            .add(ExperienceItem::new("third"))
            .await
            .expect("add third");
        assert_eq!(outcome.id, "exp_0003");
    }
}
