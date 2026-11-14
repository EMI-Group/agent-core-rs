//! The persisted, embedding-backed experience index.
//!
//! Python's `ExperienceBank` keeps its items in memory and persists an `index_dir`
//! directory of four files: a JSON manifest (`meta.json`), the scalar items
//! (`scalar/metadata.jsonl`), a faiss index (`vector/faiss_index.bin`), and the raw
//! embedding matrix (`vector/embeddings.npy`). This module ports that durable catalog
//! without faiss and without numpy:
//!
//! | Python | here |
//! | --- | --- |
//! | `meta.json` (`_KnowledgeMeta`) | `meta.json` ([`ExperienceIndexMeta`]) |
//! | `scalar/metadata.jsonl` | `scalar/metadata.jsonl` (one item JSON per line) |
//! | `vector/faiss_index.bin` + `vector/embeddings.npy` | `vector/embeddings.json` |
//!
//! The vector file is `{"dimension": n, "ids": [...], "rows": [[f32, ...], ...]}`. An exact
//! flat index is just a matrix plus an exhaustive rank, so the crate stores the matrix and
//! scores it with the crate's own `rank_by_embedding` helper (faiss' inner product over
//! L2-normalized vectors is the same cosine). JSON is the
//! persisted form because an offline build has neither numpy nor faiss to read or write
//! their binary formats.
//!
//! # Intentional differences from Python
//!
//! - **One vector file instead of a faiss index plus an `.npy` matrix.** The matrix *is*
//!   the exact index, so there is nothing else to keep in step; `meta.json` therefore has
//!   no `vector_sha256` counterpart, only `embeddings_sha256` and `scalar_sha256`.
//! - **Only the flat algorithm.** `"flat"` (any case) is implemented. Every IVF variant
//!   (`ivf`, `ivfpq`, `IVF4096,PQ16`, ...) needs faiss to train k-means centroids and fill
//!   inverted lists, so [`ExperienceIndex::open_with_algorithm`] reports it as a typed
//!   [`Error::Unsupported`](crate::error::Error::Unsupported) and the caller can fall back
//!   to flat cosine search, which returns exactly the same neighbours exhaustively.
//! - **Loud corruption instead of a `_backup` restore.** Python snapshots the four live
//!   files into `_backup/` before each publish and restores them when the manifest
//!   disagrees with the data. This port writes each file through a temp sibling plus a
//!   rename (vectors, then scalar, then manifest) and **rejects** an inconsistent
//!   directory with a precise error naming the offending file; a caller decides whether to
//!   rebuild, delete, or repair. Silent restore is the one thing a knowledge base must not
//!   do, because the restored snapshot may be older than the query that follows.
//! - **The embedded text is the whole record.** [`ExperienceItem::search_text`] (pattern,
//!   examples, skill ids) is embedded, where Python's `create_item` embeds only the query
//!   pattern. The examples carry the vocabulary a later query is phrased in, and
//!   [`ExperienceBank`](crate::experience::ExperienceBank) already ranks that same text
//!   lexically, so both catalogs agree on what an item is "about".
//! - **Dedupe instead of append.** A duplicate observation (same normalized pattern, same
//!   skill set) merges into the record stored first, exactly like
//!   [`ExperienceBank`](crate::experience::ExperienceBank); Python's list would keep both.
//!
//! `persist` (in `persist.rs`) owns the directory layout, the digests, and the load
//! validation; this module owns the catalog and its API.
//!
//! # Locking and write order
//!
//! The catalog lives behind a [`std::sync::RwLock`] that is only held for map operations,
//! never across an `.await`: items are embedded first, then applied, then persisted. A
//! persist failure rolls the touched entries back, so the in-memory catalog never claims a
//! change the directory does not hold.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

use openjiuwen_retrieval::Embedding;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::{unsupported, validation, Result};
use crate::experience::ranking::{
    rank_by_embedding, rank_lexically, ExperienceHit, ExperienceRanking,
};
use crate::experience::{merge_item, ordinal_of, AddOutcome, ExperienceItem, ID_DIGITS, ID_PREFIX};

mod persist;

/// The manifest schema version written by [`ExperienceIndex::persist`].
pub const MANIFEST_VERSION: u32 = 1;

/// The only vector algorithm this crate implements: exact flat cosine search.
pub const FLAT_VECTOR_ALGORITHM: &str = "flat";

/// The integrity manifest of a persisted index (`meta.json`).
///
/// Mirrors Python's `_KnowledgeMeta`: a schema version, the number of stored vectors, the
/// vector algorithm, the vector dimension, and the SHA-256 hex digests of the two data
/// files. The digests cover the exact bytes on disk, so a hand-edited scalar file or a
/// truncated vector file is detected instead of being loaded as a smaller catalog.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct ExperienceIndexMeta {
    /// The manifest schema version; [`MANIFEST_VERSION`] today.
    pub version: u32,
    /// The number of items, which is also the number of embedding rows.
    pub vector_count: usize,
    /// The vector algorithm; [`FLAT_VECTOR_ALGORITHM`].
    pub vector_algorithm: String,
    /// The length of every embedding row (`0` when the index is empty).
    pub vector_dimension: usize,
    /// The SHA-256 hex digest of `vector/embeddings.json`.
    pub embeddings_sha256: String,
    /// The SHA-256 hex digest of `scalar/metadata.jsonl`.
    pub scalar_sha256: String,
}

impl Default for ExperienceIndexMeta {
    fn default() -> Self {
        Self {
            version: MANIFEST_VERSION,
            vector_count: 0,
            vector_algorithm: FLAT_VECTOR_ALGORITHM.to_string(),
            vector_dimension: 0,
            embeddings_sha256: String::new(),
            scalar_sha256: String::new(),
        }
    }
}

/// The catalog: an item map plus one embedding per item, optionally persisted.
///
/// Without a directory ([`ExperienceIndex::new`]) the index is a purely in-memory
/// embedding catalog; with one ([`ExperienceIndex::open`]) every mutation is written
/// through `persist` so a reopen rehydrates the exact same vectors.
pub struct ExperienceIndex {
    /// The index directory, or `None` for an in-memory index.
    dir: Option<PathBuf>,
    /// The embedder that owns the vector space of this index.
    embedding: Arc<dyn Embedding>,
    /// The authoritative items, keyed and ordered by id.
    items: RwLock<BTreeMap<String, ExperienceItem>>,
    /// The next generated-id ordinal; `exp_NNNN` is rendered from it.
    next_ordinal: AtomicU64,
    /// The canonical vector algorithm; always [`FLAT_VECTOR_ALGORITHM`] today.
    algorithm: String,
    /// The manifest of the last successful load or persist.
    meta: RwLock<ExperienceIndexMeta>,
}

impl fmt::Debug for ExperienceIndex {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ExperienceIndex")
            .field("dir", &self.dir)
            .field("items", &self.count())
            .field("algorithm", &self.algorithm)
            .field("embedding", &self.embedding.model_name())
            .finish_non_exhaustive()
    }
}

impl ExperienceIndex {
    /// Builds an empty in-memory index that never touches a directory.
    ///
    /// Items keep their vectors in memory only; call [`Self::persist`] after opening a
    /// directory-backed index to write them out.
    #[must_use]
    pub fn new(embedding: Arc<dyn Embedding>) -> Self {
        Self::empty(None, embedding, FLAT_VECTOR_ALGORITHM.to_string())
    }

    /// Opens the index under `dir`, or starts an empty one when `dir` holds no index files.
    ///
    /// An existing index is loaded all-or-nothing: the manifest is read first, both data
    /// files are digest-checked against it, and the stored vectors are rehydrated instead
    /// of being recomputed. The generated-id counter resumes past the highest `exp_NNNN`
    /// id, so a reopened index never reuses an id.
    ///
    /// # Errors
    ///
    /// Returns [`Error::External`](crate::error::Error::External) when an index file exists
    /// but cannot be read, and [`Error::Validation`](crate::error::Error::Validation) naming
    /// the offending file when the directory is inconsistent
    /// (a missing file, an unsupported manifest version, a digest mismatch, a row/id count
    /// or dimension mismatch, or a malformed JSON record).
    pub async fn open(dir: impl AsRef<Path>, embedding: Arc<dyn Embedding>) -> Result<Self> {
        Self::open_with_algorithm(dir, embedding, FLAT_VECTOR_ALGORITHM).await
    }

    /// Opens the index under `dir` with an explicit vector algorithm.
    ///
    /// `"flat"` in any case selects the exact flat cosine index this crate implements.
    /// Every IVF variant (`"ivf"`, `"ivfpq"`, anything starting with `ivf`) needs faiss to
    /// train k-means centroids and populate inverted lists — a GPU/ML stack an offline
    /// build does not ship — so it is reported as a typed
    /// [`Error::Unsupported`](crate::error::Error::Unsupported) whose message names the
    /// deferred feature. Any other value is an [`Error::Validation`](crate::error::Error::Validation) naming
    /// `vector_algorithm`.
    ///
    /// # Errors
    ///
    /// Returns the algorithm errors above, plus the load errors of [`Self::open`].
    pub async fn open_with_algorithm(
        dir: impl AsRef<Path>,
        embedding: Arc<dyn Embedding>,
        algorithm: impl Into<String>,
    ) -> Result<Self> {
        let algorithm = normalize_algorithm(&algorithm.into())?;
        persist::load(dir.as_ref(), embedding, algorithm).await
    }

    /// The embedding model the index derives its vectors with.
    #[must_use]
    pub fn embedding(&self) -> &Arc<dyn Embedding> {
        &self.embedding
    }

    /// The index directory, or `None` for an in-memory index.
    #[must_use]
    pub fn dir(&self) -> Option<&Path> {
        self.dir.as_deref()
    }

    /// The canonical vector algorithm of this index ([`FLAT_VECTOR_ALGORITHM`]).
    #[must_use]
    pub fn algorithm(&self) -> &str {
        &self.algorithm
    }

    /// The manifest of the last successful load or persist.
    ///
    /// An index that was never persisted reports the default manifest (version
    /// [`MANIFEST_VERSION`], the flat algorithm, no vectors, and empty digests).
    #[must_use]
    pub fn meta(&self) -> ExperienceIndexMeta {
        self.read_meta().clone()
    }

    /// The number of items in the index.
    #[must_use]
    pub fn count(&self) -> usize {
        self.read_items().len()
    }

    /// Reports whether the index holds no items.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.read_items().is_empty()
    }

    /// Every item, sorted by id, with its stored embedding.
    #[must_use]
    pub fn items(&self) -> Vec<ExperienceItem> {
        self.read_items().values().cloned().collect()
    }

    /// Reports whether `skill_ids` is the skill set of at least one item.
    ///
    /// The comparison is set-based, so order and duplicate ids do not matter: `["b", "a"]`
    /// matches an item with `["a", "b"]`. This is Python's `ExperienceBank.exist`
    /// (`frozenset(skills) in self._skills`); the set is derived on demand from the
    /// catalog, which is a distilled knowledge base rather than a document store.
    #[must_use]
    pub fn exist(&self, skill_ids: &[&str]) -> bool {
        let requested: BTreeSet<&str> = skill_ids.iter().copied().collect();
        self.read_items()
            .values()
            .any(|item| item.skill_set() == requested)
    }

    /// Generates the next `exp_NNNN` id, advancing the counter.
    ///
    /// The id is not checked against the catalog; [`Self::add`] skips ids that are already
    /// taken. Callers that mint ids up front and insert them later should prefer letting
    /// [`Self::add`] generate the id.
    #[must_use]
    pub fn generate_id(&self) -> String {
        let ordinal = self.next_ordinal.fetch_add(1, Ordering::SeqCst);
        format!("{ID_PREFIX}{ordinal:0ID_DIGITS$}")
    }

    /// Adds `item`, embedding it and merging it into an existing record when it duplicates
    /// one.
    ///
    /// - A blank id is replaced by the next free `exp_NNNN`; an explicit id is kept as
    ///   given (trimmed).
    /// - A duplicate (same normalized pattern and same skill set) is merged: the success
    ///   counters add up, the example lists union in first-seen order, the later
    ///   `last_hit_at` wins, and the stored embedding and id are preserved. A merge into an
    ///   already-embedded duplicate skips the embedding call.
    /// - The item's vector is derived from [`ExperienceItem::search_text`] by
    ///   [`openjiuwen_retrieval::Embedding::embed_query`]; a caller-supplied `embedding` is
    ///   replaced, because the index owns its vector space.
    /// - The change is persisted before this call returns and rolled back if the write
    ///   fails.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](crate::error::Error::Validation) when the normalized
    /// `query_pattern` is blank or the
    /// embedding model returns an empty vector or a vector of a different dimension than
    /// the catalog, the model's error when the embedding call fails, and the backend's
    /// error when the persist fails.
    pub async fn add(&self, item: ExperienceItem) -> Result<AddOutcome> {
        validate_pattern(&item)?;
        let mut item = item;
        if self.needs_embedding(&item) {
            item.embedding = self.embed_item(&item).await?;
        }
        let (outcome, previous) = self.apply_insert(item)?;
        if let Err(error) = self.persist().await {
            self.restore(vec![previous]);
            return Err(error);
        }
        Ok(outcome)
    }

    /// Adds every item in `items` with a single `embed_documents` call.
    ///
    /// Outcomes are returned in input order and merge exactly like [`Self::add`]; the batch
    /// is persisted once, at the end. A failure part-way through (a model error, a
    /// dimension mismatch, or a persist failure) restores every entry the batch touched, so
    /// an all-or-nothing batch is the visible result.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::add`]; an empty input is `Ok(vec![])`.
    pub async fn add_batch(&self, items: Vec<ExperienceItem>) -> Result<Vec<AddOutcome>> {
        if items.is_empty() {
            return Ok(Vec::new());
        }
        for item in &items {
            validate_pattern(item)?;
        }
        let texts: Vec<String> = items.iter().map(ExperienceItem::search_text).collect();
        let vectors = self.embedding.embed_documents(&texts, None).await?;
        if vectors.len() != items.len() {
            return Err(validation(
                "embedding",
                format!(
                    "the model returned {} vectors for {} items",
                    vectors.len(),
                    items.len()
                ),
            ));
        }
        for (item, vector) in items.iter().zip(&vectors) {
            if vector.is_empty() {
                return Err(validation(
                    "embedding",
                    format!(
                        "the model returned an empty vector for `{}`",
                        item.query_pattern
                    ),
                ));
            }
        }

        let mut outcomes = Vec::with_capacity(items.len());
        let mut previous = Vec::with_capacity(items.len());
        for (item, vector) in items.into_iter().zip(vectors) {
            match self.apply_insert(item.with_embedding(vector)) {
                Ok((outcome, state)) => {
                    outcomes.push(outcome);
                    previous.push(state);
                }
                Err(error) => {
                    self.restore(previous);
                    return Err(error);
                }
            }
        }
        if let Err(error) = self.persist().await {
            self.restore(previous);
            return Err(error);
        }
        Ok(outcomes)
    }

    /// Removes `id`, returning whether it existed.
    ///
    /// The removal is persisted before this call returns and rolled back if the write
    /// fails.
    ///
    /// # Errors
    ///
    /// Returns the backend's error when the persist fails.
    pub async fn remove(&self, id: &str) -> Result<bool> {
        let Some(previous) = self.write_items().remove(id) else {
            return Ok(false);
        };
        if let Err(error) = self.persist().await {
            self.write_items().insert(id.to_string(), previous);
            return Err(error);
        }
        Ok(true)
    }

    /// Writes the catalog to its directory and refreshes [`Self::meta`].
    ///
    /// The three artifacts are written in dependency order — `vector/embeddings.json`,
    /// then `scalar/metadata.jsonl`, then the `meta.json` that vouches for both — each
    /// through a temp sibling plus a rename, so a reader never observes a partial file.
    /// The call is idempotent: persisting an unchanged catalog rewrites the same bytes and
    /// the same digests. An in-memory index ([`Self::dir`] is `None`) writes nothing but
    /// still refreshes the manifest.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](crate::error::Error::Validation) when an item has no
    /// embedding, and [`Error::External`](crate::error::Error::External) when a file cannot be
    /// encoded or written.
    pub async fn persist(&self) -> Result<()> {
        let encoded = persist::encode(&self.items(), &self.algorithm)?;
        if let Some(dir) = &self.dir {
            persist::write(dir, &encoded).await?;
        }
        *self.write_meta() = encoded.meta;
        Ok(())
    }

    /// Searches the index, returning at most `ranking.top_k` hits (`0` means unlimited).
    ///
    /// The query is embedded with the index's model and the hits are ranked by cosine
    /// similarity through the crate's `rank_by_embedding` helper, so ties break
    /// on id and `min_score` is applied after truncation, exactly like
    /// [`ExperienceBank::search`](crate::experience::ExperienceBank::search). A blank query
    /// falls back to the deterministic lexical ranking, because embedding providers reject
    /// blank input.
    ///
    /// # Errors
    ///
    /// Returns the embedding model's error when the query cannot be embedded.
    pub async fn search(
        &self,
        query: &str,
        ranking: &ExperienceRanking,
    ) -> Result<Vec<ExperienceHit>> {
        let items = self.items();
        if items.is_empty() {
            return Ok(Vec::new());
        }
        let options = ranking.options();
        if query.trim().is_empty() {
            return Ok(rank_lexically(&items, query, &options));
        }
        let query_vector = self.embedding.embed_query(query).await?;
        Ok(rank_by_embedding(&items, &query_vector, &options))
    }

    /// Returns the distinct skill ids of the hits, in first-seen order.
    ///
    /// This is Python's `search_with_skill_ids` convenience: the caller wants "which skills
    /// serve this query", not the whole items.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::search`].
    pub async fn search_with_skill_ids(
        &self,
        query: &str,
        ranking: &ExperienceRanking,
    ) -> Result<Vec<String>> {
        let mut skill_ids = Vec::new();
        for hit in self.search(query, ranking).await? {
            for skill_id in hit.item.skill_ids {
                if !skill_ids.contains(&skill_id) {
                    skill_ids.push(skill_id);
                }
            }
        }
        Ok(skill_ids)
    }

    /// Creates an empty catalog over `dir` (when given) with the given manifest.
    fn empty(dir: Option<PathBuf>, embedding: Arc<dyn Embedding>, algorithm: String) -> Self {
        Self::from_parts(
            dir,
            embedding,
            BTreeMap::new(),
            1,
            algorithm,
            ExperienceIndexMeta::default(),
        )
    }

    /// Assembles an index from a catalog the persistence layer already loaded.
    pub(super) fn from_parts(
        dir: Option<PathBuf>,
        embedding: Arc<dyn Embedding>,
        items: BTreeMap<String, ExperienceItem>,
        next_ordinal: u64,
        algorithm: String,
        meta: ExperienceIndexMeta,
    ) -> Self {
        Self {
            dir,
            embedding,
            items: RwLock::new(items),
            next_ordinal: AtomicU64::new(next_ordinal),
            algorithm,
            meta: RwLock::new(meta),
        }
    }

    /// Reports whether `item` still needs its vector computed.
    ///
    /// A merge into an already-embedded duplicate can reuse the stored vector, so the
    /// embedding call is skipped when it would be discarded. The stored vector is only
    /// reused when it matches the model's fixed dimension hint; a model swap therefore
    /// re-embeds instead of mixing vector spaces.
    fn needs_embedding(&self, item: &ExperienceItem) -> bool {
        let Some(existing) = self.duplicate(&item.dedupe_key()) else {
            return true;
        };
        if existing.embedding.is_empty() {
            return true;
        }
        match self.embedding.dimension_hint() {
            Some(hint) => existing.embedding.len() != hint,
            None => false,
        }
    }

    /// Embeds the text of `item`, rejecting an empty vector from the model.
    async fn embed_item(&self, item: &ExperienceItem) -> Result<Vec<f32>> {
        let vector = self.embedding.embed_query(&item.search_text()).await?;
        if vector.is_empty() {
            return Err(validation(
                "embedding",
                format!(
                    "the model returned an empty vector for `{}`",
                    item.query_pattern
                ),
            ));
        }
        Ok(vector)
    }

    /// Applies one already-embedded item, returning its outcome and the previous state of
    /// the touched entry so a failed persist can be rolled back.
    ///
    /// The write lock is held for the whole mutation and dropped before any I/O happens.
    fn apply_insert(
        &self,
        item: ExperienceItem,
    ) -> Result<(AddOutcome, (String, Option<ExperienceItem>))> {
        let dedupe_key = item.dedupe_key();
        let mut items = self.write_items();

        let dimension = items
            .values()
            .find_map(|existing| {
                (!existing.embedding.is_empty()).then_some(existing.embedding.len())
            })
            .unwrap_or(0);
        if !item.embedding.is_empty() && dimension > 0 && item.embedding.len() != dimension {
            return Err(validation(
                "embedding",
                format!(
                    "the item has {} dimensions but the index stores {dimension}; one index \
                     cannot mix embedding models",
                    item.embedding.len()
                ),
            ));
        }

        let duplicate_id = items
            .values()
            .find(|existing| existing.dedupe_key() == dedupe_key)
            .map(|existing| existing.id.clone());
        if let Some(id) = duplicate_id {
            if let Some(existing) = items.get_mut(&id) {
                let previous = existing.clone();
                merge_item(existing, item);
                let merged_id = existing.id.clone();
                drop(items);
                self.note_ordinal(&merged_id);
                return Ok((
                    AddOutcome::merged_into(merged_id.clone()),
                    (merged_id, Some(previous)),
                ));
            }
        }

        let mut item = item;
        if item.id.trim().is_empty() {
            item.id = fresh_id(&self.next_ordinal, &items);
        } else {
            item.id = item.id.trim().to_string();
        }
        if item.embedding.is_empty() {
            return Err(validation(
                "embedding",
                format!(
                    "the item `{}` was not embedded before it was stored",
                    item.id
                ),
            ));
        }
        let id = item.id.clone();
        let previous = items.insert(id.clone(), item);
        drop(items);
        self.note_ordinal(&id);
        Ok((AddOutcome::inserted(id.clone()), (id, previous)))
    }

    /// Restores the entries a failed mutation touched, newest first.
    fn restore(&self, entries: Vec<(String, Option<ExperienceItem>)>) {
        let mut items = self.write_items();
        for (id, previous) in entries.into_iter().rev() {
            match previous {
                Some(item) => {
                    items.insert(id, item);
                }
                None => {
                    items.remove(&id);
                }
            }
        }
    }

    /// The stored item whose dedupe key matches `dedupe_key`.
    fn duplicate(&self, dedupe_key: &str) -> Option<ExperienceItem> {
        self.read_items()
            .values()
            .find(|item| item.dedupe_key() == dedupe_key)
            .cloned()
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

    /// Locks the manifest for reading, recovering from a poisoned lock.
    fn read_meta(&self) -> RwLockReadGuard<'_, ExperienceIndexMeta> {
        self.meta.read().unwrap_or_else(PoisonError::into_inner)
    }

    /// Locks the manifest for writing, recovering from a poisoned lock.
    fn write_meta(&self) -> RwLockWriteGuard<'_, ExperienceIndexMeta> {
        self.meta.write().unwrap_or_else(PoisonError::into_inner)
    }
}

impl Default for ExperienceIndex {
    /// Returns an in-memory index over a [`HashEmbedding`](openjiuwen_retrieval::HashEmbedding),
    /// which keeps a default-constructed value usable offline.
    fn default() -> Self {
        Self::new(Arc::new(openjiuwen_retrieval::HashEmbedding::default()))
    }
}

/// Normalizes a requested vector algorithm.
///
/// `"flat"` in any case is the exact flat index this crate implements. Every IVF variant
/// (`"ivf"`, `"ivfpq"`, `"IVF4096,PQ16"`, ...) needs faiss to train k-means centroids and
/// populate inverted lists: that is a GPU/ML stack an offline build does not ship, so the
/// name is reported as a typed [`Error::Unsupported`] and the caller can fall back to flat
/// cosine search, which returns the same neighbours exhaustively.
fn normalize_algorithm(raw: &str) -> Result<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(validation("vector_algorithm", "must not be blank"));
    }
    let lowered = trimmed.to_ascii_lowercase();
    if lowered == FLAT_VECTOR_ALGORITHM {
        return Ok(FLAT_VECTOR_ALGORITHM.to_string());
    }
    if lowered.starts_with("ivf") {
        return Err(unsupported("faiss IVF index training"));
    }
    Err(validation(
        "vector_algorithm",
        format!("{trimmed:?} is not supported: only {FLAT_VECTOR_ALGORITHM:?} is implemented"),
    ))
}

/// Rejects an item whose normalized pattern is blank.
fn validate_pattern(item: &ExperienceItem) -> Result<()> {
    if item.normalized_pattern().is_empty() {
        return Err(validation("query_pattern", "must not be blank"));
    }
    Ok(())
}

/// Generates an id that is not already taken, advancing `counter` as it goes.
fn fresh_id(counter: &AtomicU64, items: &BTreeMap<String, ExperienceItem>) -> String {
    loop {
        let ordinal = counter.fetch_add(1, Ordering::SeqCst);
        let candidate = format!("{ID_PREFIX}{ordinal:0ID_DIGITS$}");
        if !items.contains_key(&candidate) {
            return candidate;
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use super::*;
    use crate::error::{is_unsupported, ErrorKind};

    #[test]
    fn the_manifest_defaults_to_the_flat_algorithm() {
        let meta = ExperienceIndexMeta::default();
        assert_eq!(meta.version, MANIFEST_VERSION);
        assert_eq!(meta.vector_algorithm, FLAT_VECTOR_ALGORITHM);
        assert_eq!(meta.vector_count, 0);
        assert_eq!(meta.vector_dimension, 0);
        assert!(meta.embeddings_sha256.is_empty());
        assert!(meta.scalar_sha256.is_empty());
    }

    #[test]
    fn the_manifest_round_trips_and_tolerates_unknown_fields() {
        let meta = ExperienceIndexMeta {
            vector_count: 3,
            vector_dimension: 8,
            embeddings_sha256: "aa".to_string(),
            scalar_sha256: "bb".to_string(),
            ..ExperienceIndexMeta::default()
        };
        let mut document = serde_json::to_value(&meta).expect("serialize");
        document["unknown_field"] = serde_json::json!(true);
        let parsed: ExperienceIndexMeta = serde_json::from_value(document).expect("deserialize");
        assert_eq!(parsed, meta);

        let partial: ExperienceIndexMeta =
            serde_json::from_str(r#"{"vector_count":2}"#).expect("deserialize partial");
        assert_eq!(partial.vector_count, 2);
        assert_eq!(partial.version, MANIFEST_VERSION);
        assert_eq!(partial.vector_algorithm, FLAT_VECTOR_ALGORITHM);
    }

    #[test]
    fn flat_is_case_insensitive() {
        for raw in ["flat", "Flat", "FLAT", " flat "] {
            assert_eq!(
                normalize_algorithm(raw).expect("flat"),
                FLAT_VECTOR_ALGORITHM
            );
        }
    }

    #[test]
    fn ivf_algorithms_are_reported_as_deferred() {
        for raw in ["ivf", "IVF", "ivfpq", "IVF4096,PQ16", "  IvfFlat  "] {
            let error = normalize_algorithm(raw).expect_err("ivf must not be accepted");
            assert_eq!(error.kind(), ErrorKind::Unsupported, "{raw}");
            assert!(is_unsupported(&error), "{raw}");
            assert!(error.message().contains("faiss IVF"), "{raw}");
            assert!(error.message().contains("deferred"), "{raw}");
        }
    }

    #[test]
    fn unknown_and_blank_algorithms_are_validation_errors() {
        let unknown = normalize_algorithm("hnsw").expect_err("unknown");
        assert_eq!(unknown.kind(), ErrorKind::Validation);
        assert!(unknown.message().contains("vector_algorithm"));
        assert!(unknown.message().contains("hnsw"));

        let blank = normalize_algorithm("   ").expect_err("blank");
        assert_eq!(blank.kind(), ErrorKind::Validation);
        assert!(blank.message().contains("vector_algorithm"));
    }

    #[tokio::test]
    async fn the_default_index_is_an_empty_in_memory_catalog() {
        let index = ExperienceIndex::default();
        assert!(index.is_empty());
        assert_eq!(index.count(), 0);
        assert!(index.dir().is_none());
        assert_eq!(index.algorithm(), FLAT_VECTOR_ALGORITHM);
        assert!(index.items().is_empty());
        assert!(!index.exist(&[]));
        assert_eq!(index.generate_id(), "exp_0001");
        assert_eq!(index.generate_id(), "exp_0002");
        assert!(index.persist().await.is_ok());
        assert_eq!(index.meta().vector_count, 0);
    }
}
