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
//! | `vector/embeddings.npy` | `vector/embeddings.json` |
//! | `vector/faiss_index.bin` | `vector/ivf.json` (IVF only; flat needs no separate index) |
//!
//! The vector file is `{"dimension": n, "ids": [...], "rows": [[f32, ...], ...]}`. An exact
//! flat index is just a matrix plus an exhaustive rank, so the crate stores the matrix and
//! scores it with the crate's own `rank_by_embedding` helper (faiss' inner product over
//! L2-normalized vectors is the same cosine). JSON is the persisted form because an offline
//! build has neither numpy nor faiss to read or write their binary formats.
//!
//! # Two search shapes: flat and IVF
//!
//! `"flat"` (any case) is the exact index: every stored vector is scored and the best
//! `top_k` come back. An IVF name (`"ivf"`, `"ivfflat"`, `"ivf4096"`, `"IVF4096,PQ16"`, ...)
//! adds the coarse-quantiser geometry: a deterministic k-means trains `nlist` centroids
//! (`DEFAULT_IVF_NLIST` = 64 when the name does not say, clamped to the item count), every item
//! is assigned to the list of its nearest centroid, and a query
//! 1. is embedded and L2-normalised,
//! 2. probes the [`ExperienceIndex::nprobe`] lists whose centroids have the highest inner
//!    product with the query (ties by list index, effective probe clamped to
//!    `1..=nlist`), and
//! 3. ranks exactly the gathered candidates with the very same flat cosine helper, so
//!    skill filters, `top_k` (`0` = unlimited), `min_score`-after-truncation, and tie-on-id
//!    ordering are unchanged.
//!
//! Probing fewer lists is a recall trade-off only: `nprobe >= nlist` probes every list and
//! therefore returns exactly the flat neighbour list, in the same order. A flat index ignores
//! `nlist` and `nprobe` entirely.
//!
//! **Product quantisation is parsed and recorded only.** The `pq<m>` part of a name survives
//! in the canonical name and in `meta.json`, but no vector is compressed: probed candidates
//! are scored against their exact stored vectors. Compressing them would need faiss' PQ code
//! books and would make every score an approximation, which this port does not do silently.
//!
//! # Intentional differences from Python
//!
//! - **One vector file instead of an `.npy` matrix.** The matrix *is* the exact index, so
//!   there is nothing else to keep in step; `meta.json` therefore has no `vector_sha256`
//!   counterpart, only `embeddings_sha256`, `scalar_sha256` and (for IVF) `ivf_sha256`.
//! - **A deterministic coarse quantiser instead of faiss' randomized k-means.** The seeding,
//!   the Lloyd cap and the tie-breaks are specified in [`crate::experience::index`]'s
//!   `ivf` module, so two builds over the same vectors produce byte-identical centroids,
//!   lists and digests.
//! - **Loud corruption instead of a `_backup` restore.** Python snapshots the four live
//!   files into `_backup/` before each publish and restores them when the manifest
//!   disagrees with the data. This port writes each file through a temp sibling plus a
//!   rename (vectors, then scalar, then coarse quantiser, then manifest) and **rejects** an
//!   inconsistent directory with a precise error naming the offending file; a caller decides
//!   whether to rebuild, delete, or repair. Silent restore is the one thing a knowledge base
//!   must not do, because the restored snapshot may be older than the query that follows.
//! - **The embedded text is the whole record.** [`ExperienceItem::search_text`] (pattern,
//!   examples, skill ids) is embedded, where Python's `create_item` embeds only the query
//!   pattern. The examples carry the vocabulary a later query is phrased in, and
//!   [`ExperienceBank`](crate::experience::ExperienceBank) already ranks that same text
//!   lexically, so both catalogs agree on what an item is "about".
//! - **Dedupe instead of append.** A duplicate observation (same normalized pattern, same
//!   skill set) merges into the record stored first, exactly like
//!   [`ExperienceBank`](crate::experience::ExperienceBank); Python's list would keep both.
//!
//! `algorithm` (in `algorithm.rs`) owns the factory-name grammar, `ivf` (in `ivf.rs`) the
//! coarse quantiser, and `persist` (in `persist.rs`) the directory layout, the digests, and
//! the load validation; this module owns the catalog, its search path and its API.
//!
//! # Locking and write order
//!
//! The catalog lives behind a [`std::sync::RwLock`] that is only held for map operations,
//! never across an `.await`: items are embedded first, then applied, then persisted. A
//! persist failure rolls the touched entries back, so the in-memory catalog never claims a
//! change the directory does not hold. The trained coarse quantiser is cached behind its own
//! lock, built lazily on the first IVF search and dropped on every mutation.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

use openjiuwen_retrieval::Embedding;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::{validation, Error, Result};
use crate::experience::ranking::{
    rank_by_embedding, rank_lexically, ExperienceHit, ExperienceRanking,
};
use crate::experience::{merge_item, ordinal_of, AddOutcome, ExperienceItem, ID_DIGITS, ID_PREFIX};

mod algorithm;
mod ivf;
mod persist;

use self::algorithm::{effective_nlist, AlgorithmSpec};
use self::ivf::CoarseIndex;
use self::persist::{EncodeCache, EncodeInput, EncodedIndex, WriteFault};

/// The manifest schema version written by [`ExperienceIndex::persist`].
pub const MANIFEST_VERSION: u32 = 1;

/// The canonical name of the exact flat cosine index ([`ExperienceIndex::new`]'s default).
pub const FLAT_VECTOR_ALGORITHM: &str = "flat";

/// The number of coarse lists an IVF index builds when its name does not say.
///
/// The *effective* count is [`ExperienceIndex::nlist`]: this default clamped to the current
/// item count, and at least one whenever the index is non-empty.
pub(crate) const DEFAULT_IVF_NLIST: usize = 64;

/// The number of coarse lists an IVF query probes by default.
///
/// The *effective* probe is this value clamped to `1..=nlist`; a flat index ignores it.
pub(crate) const DEFAULT_IVF_NPROBE: usize = 8;

/// The integrity manifest of a persisted index (`meta.json`).
///
/// Mirrors Python's `_KnowledgeMeta`: a schema version, the number of stored vectors, the
/// vector algorithm, the vector dimension, and the SHA-256 hex digests of the data files.
/// The digests cover the exact bytes on disk, so a hand-edited scalar file or a truncated
/// vector file is detected instead of being loaded as a smaller catalog.
///
/// `ivf_sha256` covers `vector/ivf.json` and is present only for an IVF index; a flat index
/// omits the field entirely, so a manifest written before IVF support still parses.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct ExperienceIndexMeta {
    /// The manifest schema version; [`MANIFEST_VERSION`] today.
    pub version: u32,
    /// The number of items, which is also the number of embedding rows.
    pub vector_count: usize,
    /// The canonical vector algorithm ([`FLAT_VECTOR_ALGORITHM`], or an `ivf` name).
    pub vector_algorithm: String,
    /// The length of every embedding row (`0` when the index is empty).
    pub vector_dimension: usize,
    /// The SHA-256 hex digest of `vector/embeddings.json`.
    pub embeddings_sha256: String,
    /// The SHA-256 hex digest of `scalar/metadata.jsonl`.
    pub scalar_sha256: String,
    /// The SHA-256 hex digest of `vector/ivf.json`, `None` for a flat index.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ivf_sha256: Option<String>,
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
            ivf_sha256: None,
        }
    }
}

/// The catalog: an item map plus one embedding per item, optionally persisted.
///
/// Without a directory ([`ExperienceIndex::new`]) the index is a purely in-memory
/// embedding catalog; with one ([`ExperienceIndex::open`]) every mutation is written
/// through `persist` so a reopen rehydrates the exact same vectors.
///
/// An IVF index additionally keeps a lazily trained `CoarseIndex` behind its own lock: the
/// first IVF search after a mutation trains it, and the persist immediately after a mutation
/// stores the trained state so a reopen reuses it instead of retraining.
pub struct ExperienceIndex {
    /// The index directory, or `None` for an in-memory index.
    dir: Option<PathBuf>,
    /// The embedder that owns the vector space of this index.
    embedding: Arc<dyn Embedding>,
    /// The authoritative items, keyed and ordered by id.
    items: RwLock<BTreeMap<String, ExperienceItem>>,
    /// The next generated-id ordinal; `exp_NNNN` is rendered from it.
    next_ordinal: AtomicU64,
    /// The canonical vector algorithm ([`FLAT_VECTOR_ALGORITHM`] or an `ivf` name).
    algorithm: AlgorithmSpec,
    /// The number of coarse lists a query probes, before clamping.
    nprobe: AtomicUsize,
    /// The trained coarse quantiser, or `None` before the first IVF search/persist.
    coarse: RwLock<Option<Arc<CoarseIndex>>>,
    /// The manifest of the last successful load or persist.
    meta: RwLock<ExperienceIndexMeta>,
    /// The per-item encoding cache that keeps a publish incremental.
    cache: Mutex<EncodeCache>,
    /// Serialises the persists of this index, so a slower, older snapshot cannot overwrite a
    /// newer one.
    persist_lock: tokio::sync::Mutex<()>,
    /// The number of individual items encoded since construction (diagnostics).
    encoded_items: AtomicU64,
    /// The number of completed publishes since construction (diagnostics).
    publishes: AtomicU64,
}

impl fmt::Debug for ExperienceIndex {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ExperienceIndex")
            .field("dir", &self.dir)
            .field("items", &self.count())
            .field("algorithm", &self.algorithm.canonical())
            .field("nprobe", &self.nprobe())
            .field("embedding", &self.embedding.model_name())
            .finish_non_exhaustive()
    }
}

impl ExperienceIndex {
    /// Builds an empty in-memory index that never touches a directory.
    ///
    /// Items keep their vectors in memory only; call [`Self::persist`] after opening a
    /// directory-backed index to write them out. The algorithm is
    /// [`FLAT_VECTOR_ALGORITHM`]; use [`Self::new_with_algorithm`] for IVF.
    #[must_use]
    pub fn new(embedding: Arc<dyn Embedding>) -> Self {
        Self::empty(None, embedding, AlgorithmSpec::flat())
    }

    /// Builds an empty in-memory index with an explicit vector algorithm.
    ///
    /// Accepts the same names as [`Self::open_with_algorithm`]: `flat`, `ivf`, `ivfflat`, or
    /// an IVF factory name such as `ivf4096` or `IVF4096,PQ16`. An IVF index trains its
    /// coarse quantiser lazily on the first search or [`Self::persist`]; product
    /// quantisation is parsed and recorded only, so vectors are never compressed.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](crate::error::Error::Validation) naming
    /// `vector_algorithm` for a blank or unimplemented name, or a zero `nlist`/`pq` count.
    pub fn new_with_algorithm(
        embedding: Arc<dyn Embedding>,
        algorithm: impl Into<String>,
    ) -> Result<Self> {
        let spec = AlgorithmSpec::parse(&algorithm.into())?;
        Ok(Self::empty(None, embedding, spec))
    }

    /// Opens the index under `dir`, or starts an empty one when `dir` holds no index files.
    ///
    /// An existing index is loaded all-or-nothing: the manifest is read first, every data
    /// file is digest-checked against it, and the stored vectors (and, for an IVF index, the
    /// trained coarse quantiser) are rehydrated instead of being recomputed. The
    /// generated-id counter resumes past the highest `exp_NNNN` id, so a reopened index never
    /// reuses an id.
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
    /// `"flat"` in any case selects the exact flat cosine index; `"ivf"`, `"ivfflat"`,
    /// `"ivf<nlist>"` and `"IVF<nlist>,PQ<m>"` select IVF probe search. The requested
    /// algorithm must be the one the directory was built with — an IVF directory holds a
    /// `vector/ivf.json` trained for exactly that name, so opening it as `flat` (or as a
    /// different IVF name) is an [`Error::Validation`](crate::error::Error::Validation)
    /// naming `meta.json` rather than a silent rebuild. Any other name is an
    /// [`Error::Validation`](crate::error::Error::Validation) naming `vector_algorithm`.
    ///
    /// # Errors
    ///
    /// Returns the algorithm errors above, plus the load errors of [`Self::open`].
    pub async fn open_with_algorithm(
        dir: impl AsRef<Path>,
        embedding: Arc<dyn Embedding>,
        algorithm: impl Into<String>,
    ) -> Result<Self> {
        let algorithm = AlgorithmSpec::parse(&algorithm.into())?;
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

    /// The canonical vector algorithm of this index ([`FLAT_VECTOR_ALGORITHM`] or `ivf…`).
    #[must_use]
    pub fn algorithm(&self) -> &str {
        self.algorithm.canonical()
    }

    /// The effective number of coarse lists, `0` for a flat or empty index.
    ///
    /// The value is derived from the *current* catalog: the requested count (from the
    /// algorithm name, 64 when it names none) clamped to the item count,
    /// and at least one whenever the index is non-empty. A loaded index reports the count it
    /// was persisted with, and an index whose items changed reports the count the next
    /// search or persist will train.
    #[must_use]
    pub fn nlist(&self) -> usize {
        if let Some(coarse) = self.read_coarse().as_ref() {
            return coarse.nlist();
        }
        if !self.is_ivf() {
            return 0;
        }
        effective_nlist(self.algorithm.nlist_request(), self.count())
    }

    /// The number of coarse lists a query probes (8 unless set).
    ///
    /// The effective probe is this value clamped to `1..=nlist` at query time; a flat index
    /// ignores it.
    #[must_use]
    pub fn nprobe(&self) -> usize {
        self.nprobe.load(Ordering::SeqCst)
    }

    /// Sets the number of coarse lists a query probes.
    ///
    /// The index is meant to be shared behind an `Arc`, so the value lives in an atomic and
    /// this takes `&self`. `0` is accepted and probes one list; a value above
    /// [`Self::nlist`] probes every list, which reproduces the flat result.
    pub fn set_nprobe(&self, nprobe: usize) {
        self.nprobe.store(nprobe, Ordering::SeqCst);
    }

    /// Sets the probe count on a builder, before the index is shared.
    #[must_use]
    pub fn with_nprobe(self, nprobe: usize) -> Self {
        self.set_nprobe(nprobe);
        self
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
        self.invalidate_coarse();
        self.forget_cached(id);
        if let Err(error) = self.persist().await {
            self.write_items().insert(id.to_string(), previous);
            self.invalidate_coarse();
            return Err(error);
        }
        Ok(true)
    }

    /// Writes the catalog to its directory and refreshes [`Self::meta`].
    ///
    /// The publish is incremental and atomic:
    ///
    /// - Only the items that changed since the last publish are serialized (a dirty set plus
    ///   the cache seeded by [`Self::open`]); an unchanged catalog re-encodes nothing, so `n`
    ///   appends cost `O(n)` item encodings in total rather than `O(n^2)`.
    /// - An IVF index keeps its trained coarse quantiser while the policy in
    ///   `persist::reusable_quantiser` allows it, assigning a new item to its nearest existing
    ///   centroid instead of retraining; the documented rebuild triggers are a missing
    ///   quantiser, a changed effective list count, a removal, or a catalog that grew by
    ///   `max(trained/4, 64)` items. This bounds the expensive rebuilds to `O(log n)`.
    /// - The CPU-bound serialization and quantiser work runs on
    ///   [`tokio::task::spawn_blocking`], so a large index cannot stall a runtime worker.
    /// - The four artifacts are written to unique temp files and flushed first, then renamed
    ///   with `meta.json` last, so a failure part-way leaves the previous snapshot loadable.
    ///
    /// The call is idempotent: persisting an unchanged catalog rewrites the same bytes and the
    /// same digests. An in-memory index ([`Self::dir`] is `None`) writes nothing but still
    /// refreshes the manifest and the cached quantiser.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](crate::error::Error::Validation) when an item has no
    /// embedding, and [`Error::External`](crate::error::Error::External) when a file cannot be
    /// encoded or written.
    pub async fn persist(&self) -> Result<()> {
        let _persisting = self.persist_lock.lock().await;
        let (input, drained) = self.begin_encode()?;
        let encoded = match tokio::task::spawn_blocking(move || persist::encode(input)).await {
            Ok(Ok(encoded)) => encoded,
            Ok(Err(error)) => {
                self.mark_dirty(drained);
                return Err(error);
            }
            Err(error) => {
                self.mark_dirty(drained);
                return Err(Error::external(error).with_context("encoding the experience index"));
            }
        };
        self.encoded_items
            .fetch_add(encoded.encoded_items as u64, Ordering::Relaxed);
        self.commit_cache(&encoded);
        if let Some(dir) = &self.dir {
            persist::write(dir, &encoded, WriteFault::default()).await?;
            self.publishes.fetch_add(1, Ordering::Relaxed);
        }
        self.write_coarse().clone_from(&encoded.coarse);
        *self.write_meta() = encoded.meta.clone();
        Ok(())
    }

    /// The number of individual items encoded and the number of publishes completed.
    ///
    /// A full-catalog re-encode (the old behaviour of every [`Self::persist`]) moves the first
    /// counter by the catalog size; an incremental publish moves it by one. The counters are
    /// diagnostic only and never affect a result.
    #[must_use]
    pub fn persist_counters(&self) -> (u64, u64) {
        (
            self.encoded_items.load(Ordering::Relaxed),
            self.publishes.load(Ordering::Relaxed),
        )
    }

    /// Searches the index, returning at most `ranking.top_k` hits (`0` means unlimited).
    ///
    /// The query is embedded with the index's model and the hits are ranked by cosine
    /// similarity through the crate's `rank_by_embedding` helper, so ties break on id and
    /// `min_score` is applied after truncation, exactly like
    /// [`ExperienceBank::search`](crate::experience::ExperienceBank::search). A blank query
    /// falls back to the deterministic lexical ranking, because embedding providers reject
    /// blank input.
    ///
    /// A flat index scores every stored vector. An IVF index first probes
    /// [`Self::nprobe`] lists with the normalised query (see `coarse_candidates`) and then
    /// ranks exactly those candidates with the same helper, so every filter and ordering rule
    /// is identical; `nprobe >= nlist` probes everything and therefore reproduces the flat
    /// result exactly.
    ///
    /// # Errors
    ///
    /// Returns the embedding model's error when the query cannot be embedded, and a
    /// [`Error::Validation`](crate::error::Error::Validation) error when the stored vectors
    /// of an IVF index cannot be quantised (a dimension mismatch or a missing embedding).
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
        if !self.is_ivf() {
            return Ok(rank_by_embedding(&items, &query_vector, &options));
        }
        let candidates = self.ivf_candidates(&items, &query_vector)?;
        Ok(rank_by_embedding(&candidates, &query_vector, &options))
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

    /// Creates an empty catalog over `dir` (when given) with the given algorithm.
    fn empty(
        dir: Option<PathBuf>,
        embedding: Arc<dyn Embedding>,
        algorithm: AlgorithmSpec,
    ) -> Self {
        Self::from_parts(
            dir,
            embedding,
            BTreeMap::new(),
            1,
            algorithm,
            ExperienceIndexMeta::default(),
            None,
        )
    }

    /// Assembles an index from a catalog the persistence layer already loaded.
    pub(super) fn from_parts(
        dir: Option<PathBuf>,
        embedding: Arc<dyn Embedding>,
        items: BTreeMap<String, ExperienceItem>,
        next_ordinal: u64,
        algorithm: AlgorithmSpec,
        meta: ExperienceIndexMeta,
        coarse: Option<Arc<CoarseIndex>>,
    ) -> Self {
        Self {
            dir,
            embedding,
            items: RwLock::new(items),
            next_ordinal: AtomicU64::new(next_ordinal),
            algorithm,
            nprobe: AtomicUsize::new(DEFAULT_IVF_NPROBE),
            coarse: RwLock::new(coarse),
            meta: RwLock::new(meta),
            cache: Mutex::new(EncodeCache::default()),
            persist_lock: tokio::sync::Mutex::new(()),
            encoded_items: AtomicU64::new(0),
            publishes: AtomicU64::new(0),
        }
    }

    /// Seeds the encoding cache with the entries a loaded catalog already has.
    ///
    /// The cache is not a constructor argument: only [`Self::load`](crate::experience::index)
    /// has entries to seed, so it sets them after assembly.
    pub(super) fn seed_encode_cache(self, cache: EncodeCache) -> Self {
        *self.lock_cache() = cache;
        self
    }
    /// Reports whether the index uses the IVF probe path.
    fn is_ivf(&self) -> bool {
        self.algorithm.is_ivf()
    }

    /// The candidates an IVF query has to score, in id order.
    ///
    /// The trained quantiser comes from the cache, or is trained from the current catalog on
    /// the first search after a mutation. Training is synchronous and the catalog is only
    /// locked to take a snapshot, so no lock is held across an `.await`.
    fn ivf_candidates(
        &self,
        items: &[ExperienceItem],
        query_vector: &[f32],
    ) -> Result<Vec<ExperienceItem>> {
        let Some(coarse) = self.coarse_state()? else {
            return Ok(items.to_vec());
        };
        let probe = self.nprobe().clamp(1, coarse.nlist().max(1));
        Ok(coarse_candidates(&coarse, items, query_vector, probe))
    }

    /// The cached coarse quantiser, training it when the catalog changed.
    fn coarse_state(&self) -> Result<Option<Arc<CoarseIndex>>> {
        if let Some(cached) = self.read_coarse().clone() {
            return Ok(Some(cached));
        }
        let items = self.items();
        if items.is_empty() {
            return Ok(None);
        }
        let ids: Vec<String> = items.iter().map(|item| item.id.clone()).collect();
        let vectors: Vec<Vec<f32>> = items.iter().map(|item| item.embedding.clone()).collect();
        let nlist = effective_nlist(self.algorithm.nlist_request(), ids.len());
        let coarse = Arc::new(CoarseIndex::build(&ids, &vectors, nlist)?);
        let mut cache = self.write_coarse();
        // Another search may have trained it while this one was building; keep whichever
        // state landed first so every caller sees the same lists.
        if let Some(existing) = cache.as_ref() {
            return Ok(Some(existing.clone()));
        }
        *cache = Some(coarse.clone());
        Ok(Some(coarse))
    }

    /// Drops the cached coarse quantiser after the catalog changed.
    fn invalidate_coarse(&self) {
        *self.write_coarse() = None;
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
                self.invalidate_coarse();
                self.mark_dirty_one(&merged_id);
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
        self.invalidate_coarse();
        self.mark_dirty_one(&id);
        Ok((AddOutcome::inserted(id.clone()), (id, previous)))
    }

    /// Restores the entries a failed mutation touched, newest first.
    ///
    /// The encoding cache is repaired with it: every restored id is dirty again, the entries
    /// of ids the rollback deleted are dropped, and the frozen quantiser is discarded (it may
    /// describe the rolled-back catalog).
    fn restore(&self, entries: Vec<(String, Option<ExperienceItem>)>) {
        let mut items = self.write_items();
        let mut dirty = Vec::with_capacity(entries.len());
        let mut dropped = Vec::new();
        for (id, previous) in entries.into_iter().rev() {
            if let Some(item) = previous {
                items.insert(id.clone(), item);
                dirty.push(id);
            } else {
                items.remove(&id);
                dropped.push(id);
            }
        }
        drop(items);
        self.invalidate_coarse();
        let mut cache = self.lock_cache();
        cache.ivf = None;
        cache.dirty.extend(dirty);
        if !dropped.is_empty() {
            let mut cached = (*cache.entries).clone();
            for id in &dropped {
                cached.remove(id);
            }
            cache.entries = Arc::new(cached);
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

    /// Snapshots the catalog and the work a publish has to do.
    ///
    /// The dirty set is drained *before* the catalog is read: a mutation racing with this call
    /// either lands in the snapshot (when it happened before the read) or re-marks its id
    /// dirty (when it happens after the drain), so no change is ever silently marked clean.
    /// Only dirty or not-yet-cached items are serialized here.
    fn begin_encode(&self) -> Result<(EncodeInput, BTreeSet<String>)> {
        let (base, reuse_ivf, drained) = {
            let mut cache = self.lock_cache();
            let base = Arc::clone(&cache.entries);
            let reuse_ivf = cache.ivf.clone();
            let drained = std::mem::take(&mut cache.dirty);
            (base, reuse_ivf, drained)
        };
        let items = self.read_items();
        let mut ids = Vec::with_capacity(items.len());
        let mut fresh = BTreeMap::new();
        for (id, item) in items.iter() {
            ids.push(id.clone());
            if !drained.contains(id) && base.contains_key(id) {
                continue;
            }
            match persist::cache_entry(item) {
                Ok(entry) => {
                    fresh.insert(id.clone(), entry);
                }
                Err(error) => {
                    drop(items);
                    self.mark_dirty(drained);
                    return Err(error);
                }
            }
        }
        drop(items);
        Ok((
            EncodeInput {
                algorithm: self.algorithm.clone(),
                ids,
                base,
                fresh,
                reuse_ivf,
            },
            drained,
        ))
    }

    /// Stores the entries a publish encoded, and the quantiser it produced.
    ///
    /// The cached map is unwrapped in place when no snapshot is outstanding, so the common
    /// case costs one pass over the new entries rather than a copy of the whole catalog.
    fn commit_cache(&self, encoded: &EncodedIndex) {
        let mut cache = self.lock_cache();
        let current = std::mem::replace(&mut cache.entries, Arc::new(BTreeMap::new()));
        let mut entries = match Arc::try_unwrap(current) {
            Ok(entries) => entries,
            Err(shared) => (*shared).clone(),
        };
        for (id, entry) in &encoded.fresh {
            entries.insert(id.clone(), entry.clone());
        }
        cache.entries = Arc::new(entries);
        cache.ivf.clone_from(&encoded.cached_ivf);
    }

    /// Marks `ids` as needing an encode on the next publish.
    fn mark_dirty(&self, ids: impl IntoIterator<Item = String>) {
        let mut cache = self.lock_cache();
        cache.dirty.extend(ids);
    }

    /// Marks one id as needing an encode on the next publish.
    fn mark_dirty_one(&self, id: &str) {
        self.lock_cache().dirty.insert(id.to_string());
    }

    /// Drops every cached trace of `id` after it was removed from the catalog.
    fn forget_cached(&self, id: &str) {
        let mut cache = self.lock_cache();
        cache.dirty.remove(id);
        cache.ivf = None;
        if cache.entries.contains_key(id) {
            let mut entries = (*cache.entries).clone();
            entries.remove(id);
            cache.entries = Arc::new(entries);
        }
    }

    /// Locks the encoding cache, recovering from a poisoned lock.
    fn lock_cache(&self) -> std::sync::MutexGuard<'_, EncodeCache> {
        self.cache.lock().unwrap_or_else(PoisonError::into_inner)
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

    /// Locks the cached coarse quantiser for reading, recovering from a poisoned lock.
    fn read_coarse(&self) -> RwLockReadGuard<'_, Option<Arc<CoarseIndex>>> {
        self.coarse.read().unwrap_or_else(PoisonError::into_inner)
    }

    /// Locks the cached coarse quantiser for writing, recovering from a poisoned lock.
    fn write_coarse(&self) -> RwLockWriteGuard<'_, Option<Arc<CoarseIndex>>> {
        self.coarse.write().unwrap_or_else(PoisonError::into_inner)
    }
}

impl Default for ExperienceIndex {
    /// Returns an in-memory index over a [`HashEmbedding`](openjiuwen_retrieval::HashEmbedding),
    /// which keeps a default-constructed value usable offline.
    fn default() -> Self {
        Self::new(Arc::new(openjiuwen_retrieval::HashEmbedding::default()))
    }
}

/// Rejects an item whose normalized pattern is blank.
fn validate_pattern(item: &ExperienceItem) -> Result<()> {
    if item.normalized_pattern().is_empty() {
        return Err(validation("query_pattern", "must not be blank"));
    }
    Ok(())
}

/// The items living in the `probe` lists nearest to `query_vector`, in id order.
///
/// Every id of those lists is resolved against the id-sorted catalog; an id the quantiser
/// knows but the catalog no longer holds is skipped rather than scored as a zero vector.
fn coarse_candidates(
    coarse: &CoarseIndex,
    items: &[ExperienceItem],
    query_vector: &[f32],
    probe: usize,
) -> Vec<ExperienceItem> {
    coarse
        .probed_ids(query_vector, probe)
        .into_iter()
        .filter_map(|id| {
            items
                .binary_search_by(|item| item.id.as_str().cmp(id))
                .ok()
                .map(|position| items[position].clone())
        })
        .collect()
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
    use crate::error::ErrorKind;

    #[test]
    fn the_manifest_defaults_to_the_flat_algorithm() {
        let meta = ExperienceIndexMeta::default();
        assert_eq!(meta.version, MANIFEST_VERSION);
        assert_eq!(meta.vector_algorithm, FLAT_VECTOR_ALGORITHM);
        assert_eq!(meta.vector_count, 0);
        assert_eq!(meta.vector_dimension, 0);
        assert!(meta.embeddings_sha256.is_empty());
        assert!(meta.scalar_sha256.is_empty());
        assert!(meta.ivf_sha256.is_none());
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
        // A flat manifest carries no ivf digest at all.
        assert!(document.get("ivf_sha256").is_none());
        document["unknown_field"] = serde_json::json!(true);
        let parsed: ExperienceIndexMeta = serde_json::from_value(document).expect("deserialize");
        assert_eq!(parsed, meta);

        let partial: ExperienceIndexMeta =
            serde_json::from_str(r#"{"vector_count":2}"#).expect("deserialize partial");
        assert_eq!(partial.vector_count, 2);
        assert_eq!(partial.version, MANIFEST_VERSION);
        assert_eq!(partial.vector_algorithm, FLAT_VECTOR_ALGORITHM);
        assert_eq!(partial.ivf_sha256, None);

        // A manifest written after IVF support round-trips its digest.
        let ivf = ExperienceIndexMeta {
            vector_algorithm: "ivf4".to_string(),
            ivf_sha256: Some("cc".to_string()),
            ..ExperienceIndexMeta::default()
        };
        let document = serde_json::to_value(&ivf).expect("serialize ivf");
        assert_eq!(document["ivf_sha256"], serde_json::json!("cc"));
        assert_eq!(
            serde_json::from_value::<ExperienceIndexMeta>(document).expect("deserialize ivf"),
            ivf
        );
    }

    #[test]
    fn the_algorithms_parse_into_their_canonical_names() {
        assert_eq!(
            AlgorithmSpec::parse(" FLAT ").expect("flat").canonical(),
            FLAT_VECTOR_ALGORITHM
        );
        assert_eq!(
            AlgorithmSpec::parse("IVF4096,PQ16")
                .expect("ivf")
                .canonical(),
            "ivf4096,pq16"
        );
        let unknown = AlgorithmSpec::parse("hnsw").expect_err("unknown");
        assert_eq!(unknown.kind(), ErrorKind::Validation);
        assert!(unknown.message().contains("vector_algorithm"));
        assert!(unknown.message().contains("hnsw"));
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
        // A flat index has no coarse lists and ignores the probe setting.
        assert_eq!(index.nlist(), 0);
        assert_eq!(index.nprobe(), DEFAULT_IVF_NPROBE);
        index.set_nprobe(3);
        assert_eq!(index.nprobe(), 3);
        assert!(index.persist().await.is_ok());
        assert_eq!(index.meta().vector_count, 0);
        assert!(index.meta().ivf_sha256.is_none());
    }

    #[tokio::test]
    async fn an_ivf_index_reports_its_effective_list_count() {
        let index = ExperienceIndex::new_with_algorithm(
            Arc::new(openjiuwen_retrieval::HashEmbedding::new(4)),
            "IVF4096",
        )
        .expect("ivf");
        assert_eq!(index.algorithm(), "ivf4096");
        // Nothing to quantise yet.
        assert_eq!(index.nlist(), 0);
        for (ordinal, pattern) in ["alpha", "beta", "gamma"].iter().enumerate() {
            let mut item = ExperienceItem::new(*pattern);
            item.id = format!("exp_{:04}", ordinal + 1);
            item.embedding = vec![1.0, 0.0, 0.0, 0.0];
            index.add(item).await.expect("add");
        }
        // The requested 4096 lists clamp to the three stored vectors.
        assert_eq!(index.nlist(), 3);
        assert_eq!(index.nprobe(), DEFAULT_IVF_NPROBE);
        // The probe is clamped to the list count at query time, not stored: `0` probes one
        // list, so only the items living there are scored.
        index.set_nprobe(0);
        assert_eq!(index.nprobe(), 0);
        let probed = index
            .search("alpha", &ExperienceRanking::new(0))
            .await
            .expect("search");
        assert!(!probed.is_empty());
        assert!(probed.len() <= 3);
        // An over-large probe is clamped to `nlist`, so every list is searched.
        index.set_nprobe(usize::MAX);
        let hits = index
            .search("alpha", &ExperienceRanking::new(0))
            .await
            .expect("search");
        assert_eq!(hits.len(), 3);
    }

    #[tokio::test]
    async fn the_with_nprobe_builder_sets_the_probe_count() {
        let index = ExperienceIndex::new_with_algorithm(
            Arc::new(openjiuwen_retrieval::HashEmbedding::new(4)),
            "ivf",
        )
        .expect("ivf")
        .with_nprobe(2);
        assert_eq!(index.nprobe(), 2);
        assert_eq!(index.nlist(), 0);
    }
}
