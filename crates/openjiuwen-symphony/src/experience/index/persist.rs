//! The on-disk layout of a persisted experience index.
//!
//! Three files under one directory — four when the index is an IVF index — written in
//! dependency order and read all-or-nothing:
//!
//! ```text
//! <dir>/
//! ├── meta.json              # integrity manifest (`ExperienceIndexMeta`)
//! ├── scalar/
//! │   └── metadata.jsonl     # one `ExperienceItem` JSON per line, sorted by id
//! └── vector/
//!     ├── embeddings.json    # {"dimension", "ids", "rows"}
//!     └── ivf.json           # {"algorithm", "dimension", "nlist", "ids", "centroids", "lists"}
//! ```
//!
//! Python's layout replaces the vector artifacts with `vector/faiss_index.bin` plus
//! `vector/embeddings.npy`; an exact flat index is the matrix itself, so one JSON document
//! carries both the ids and the vectors, and an IVF index adds the trained coarse quantiser
//! as a separate document.
//!
//! Every write goes through a temp sibling plus a rename — vectors, then scalar, then the
//! coarse quantiser, then the manifest that vouches for them — and every read verifies the
//! manifest's digests, counts, and dimension before it trusts a single row.
//!
//! An inconsistent directory is **rejected**, never repaired: Python restores a `_backup/`
//! snapshot when the manifest disagrees with the data, which can silently serve a catalog
//! older than the query that follows. The error names the offending file so a caller can
//! decide whether to rebuild or delete the directory.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, PoisonError, Weak};

use openjiuwen_retrieval::Embedding;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tracing::warn;

use crate::error::{validation, Error, Result};
use crate::experience::index::algorithm::{effective_nlist, AlgorithmSpec};
use crate::experience::index::ivf::CoarseIndex;
use crate::experience::index::{ExperienceIndex, ExperienceIndexMeta, MANIFEST_VERSION};
use crate::experience::store::{publish_temporary, sync_directory, write_temporary};
use crate::experience::{ordinal_of, ExperienceItem};

/// The manifest file name (`<dir>/meta.json`).
const META_FILE: &str = "meta.json";

/// The scalar directory name (`<dir>/scalar`).
const SCALAR_DIR: &str = "scalar";

/// The scalar file name (`<dir>/scalar/metadata.jsonl`).
const SCALAR_FILE: &str = "metadata.jsonl";

/// The vector directory name (`<dir>/vector`).
const VECTOR_DIR: &str = "vector";

/// The vector file name (`<dir>/vector/embeddings.json`).
const VECTORS_FILE: &str = "embeddings.json";

/// The coarse-quantiser file name (`<dir>/vector/ivf.json`).
const IVF_FILE: &str = "ivf.json";

/// The artifact paths of one index directory.
struct IndexPaths {
    /// `<dir>/meta.json`.
    meta: PathBuf,
    /// `<dir>/scalar/metadata.jsonl`.
    scalar: PathBuf,
    /// `<dir>/vector/embeddings.json`.
    vectors: PathBuf,
    /// `<dir>/vector/ivf.json`, written only by an IVF index.
    ivf: PathBuf,
}

impl IndexPaths {
    /// Resolves the paths under `dir`.
    fn new(dir: &Path) -> Self {
        Self {
            meta: dir.join(META_FILE),
            scalar: dir.join(SCALAR_DIR).join(SCALAR_FILE),
            vectors: dir.join(VECTOR_DIR).join(VECTORS_FILE),
            ivf: dir.join(VECTOR_DIR).join(IVF_FILE),
        }
    }

    /// The three artifacts every index stores, in write order.
    fn core(&self) -> [&Path; 3] {
        [&self.meta, &self.scalar, &self.vectors]
    }
}

/// The persisted vector matrix: one row per item, aligned with `ids`.
#[derive(Debug, Serialize, Deserialize)]
struct VectorFile {
    /// The length of every row (`0` for an empty index).
    dimension: usize,
    /// The item ids, one per row, sorted ascending.
    ids: Vec<String>,
    /// The embedding rows, aligned with `ids`.
    rows: Vec<Vec<f32>>,
}

/// The persisted coarse quantiser: the trained centroids plus one list index per id.
///
/// The field order is the serialized key order, which is what keeps repeated builds
/// byte-identical and therefore digest-stable.
#[derive(Debug, Serialize, Deserialize)]
struct IvfFile {
    /// The canonical algorithm name the quantiser was trained for.
    algorithm: String,
    /// The length of every centroid (`0` for an empty index).
    dimension: usize,
    /// The number of coarse lists; also the number of centroids.
    nlist: usize,
    /// The item ids, sorted ascending.
    ids: Vec<String>,
    /// The centroids, one per list.
    centroids: Vec<Vec<f32>>,
    /// The list index of each id, aligned with `ids`.
    lists: Vec<usize>,
}

/// One item's cached encoding: the exact scalar line and vector row it contributes.
///
/// Both halves are shared (`Arc`), so a catalog snapshot is a pointer clone rather than a
/// re-serialization of every item.
#[derive(Debug, Clone)]
pub(crate) struct CacheEntry {
    /// The item's `scalar/metadata.jsonl` line, without its newline.
    pub(crate) line: Arc<str>,
    /// The item's embedding, which is also its `vector/embeddings.json` row.
    pub(crate) row: Arc<[f32]>,
}

/// The coarse quantiser kept between publishes, with the bytes it serializes to.
#[derive(Debug, Clone)]
pub(crate) struct CachedQuantiser {
    /// The trained quantiser.
    pub(crate) coarse: Arc<CoarseIndex>,
    /// The `vector/ivf.json` bytes produced by [`serialize_ivf`] for `coarse`.
    pub(crate) bytes: Vec<u8>,
    /// The number of items the centroids were trained on.
    pub(crate) trained_count: usize,
}

/// The incremental encoding cache of one catalog.
///
/// `entries` mirrors the catalog: one [`CacheEntry`] per stored item. A mutation marks the
/// touched ids `dirty` (or drops the entry, when the item is removed), so `persist` only
/// serializes the items that actually changed instead of the whole catalog.
#[derive(Debug, Default)]
pub(crate) struct EncodeCache {
    /// The encoded pieces of the catalog, shared so a snapshot is `O(1)`.
    pub(crate) entries: Arc<BTreeMap<String, CacheEntry>>,
    /// The quantiser and bytes from the last encode, reused while the policy allows it.
    pub(crate) ivf: Option<CachedQuantiser>,
    /// Ids whose cached entry changed or appeared since it was cached.
    pub(crate) dirty: BTreeSet<String>,
}

/// The work one [`encode`] call has to do: the catalog order, the cached entries, and the
/// items that changed since the last publish.
#[derive(Debug)]
pub(super) struct EncodeInput {
    /// The vector algorithm of the index.
    pub(super) algorithm: AlgorithmSpec,
    /// Every stored id, ascending (the assembly order).
    pub(super) ids: Vec<String>,
    /// The cached entries of the previously encoded items.
    pub(super) base: Arc<BTreeMap<String, CacheEntry>>,
    /// The entries serialized for this publish (dirty or not yet cached ids).
    pub(super) fresh: BTreeMap<String, CacheEntry>,
    /// The quantiser from the previous publish, subject to [`reusable_quantiser`].
    pub(super) reuse_ivf: Option<CachedQuantiser>,
}

/// The encode-only form of the vector file, borrowing the cached rows.
///
/// Serializing this instead of [`VectorFile`] avoids copying every embedding just to publish
/// the matrix.
#[derive(Serialize)]
struct VectorFileRef<'a> {
    /// The length of every row (`0` for an empty index).
    dimension: usize,
    /// The item ids, one per row, sorted ascending.
    ids: &'a [String],
    /// The embedding rows, aligned with `ids`.
    rows: Vec<&'a [f32]>,
}

/// The encoded form of one catalog snapshot.
#[derive(Debug)]
pub(super) struct EncodedIndex {
    /// The `scalar/metadata.jsonl` bytes.
    pub(super) scalar: Vec<u8>,
    /// The `vector/embeddings.json` bytes.
    pub(super) vectors: Vec<u8>,
    /// The `vector/ivf.json` bytes, present only for an IVF index.
    pub(super) ivf: Option<Vec<u8>>,
    /// The manifest vouching for the data files.
    pub(super) meta: ExperienceIndexMeta,
    /// The trained coarse quantiser, kept so a search does not retrain it.
    pub(super) coarse: Option<Arc<CoarseIndex>>,
    /// The quantiser state to cache for the next publish.
    pub(super) cached_ivf: Option<CachedQuantiser>,
    /// The entries serialized by this publish.
    pub(super) fresh: BTreeMap<String, CacheEntry>,
    /// The number of individual items serialized by this publish.
    pub(super) encoded_items: usize,
}

/// The minimum number of new items that forces an IVF rebuild, even when the catalog is small.
///
/// This is the floor of the growth policy documented on [`reusable_quantiser`].
pub(super) const IVF_REBUILD_MIN_NEW_ITEMS: usize = 64;

/// The fraction of the trained catalog that has to be appended before a rebuild is worthwhile.
pub(super) const IVF_REBUILD_GROWTH_DIVISOR: usize = 4;

/// Serializes `item` into its cached scalar line and vector row.
///
/// # Errors
///
/// Returns [`Error::Validation`](crate::error::Error::Validation) naming `embedding` when the
/// item carries no vector (the matrix has no row for it), and
/// [`Error::External`](crate::error::Error::External) when the item cannot be serialized.
pub(super) fn cache_entry(item: &ExperienceItem) -> Result<CacheEntry> {
    let line = serde_json::to_string(item).map_err(|error| {
        Error::external(error).with_context(format!("encoding item `{}`", item.id))
    })?;
    if item.embedding.is_empty() {
        return Err(validation(
            "embedding",
            format!("the item `{}` has no embedding", item.id),
        ));
    }
    Ok(CacheEntry {
        line: Arc::from(line),
        row: Arc::from(item.embedding.clone()),
    })
}

/// Serializes a catalog snapshot into the persisted artifacts and their manifest.
///
/// Items are written in `ids` order. An item that is neither cached nor fresh is an internal
/// error: the caller's snapshot missed it, and the next persist re-encodes everything.
///
/// An IVF index reuses the previous quantiser whenever [`reusable_quantiser`] allows it —
/// new items are assigned to their nearest existing centroid — and retrains otherwise. That
/// keeps the centroids (and therefore the search results of an unchanged catalog) stable
/// while bounding the number of expensive rebuilds to `O(log n)` over `n` appends.
///
/// # Errors
///
/// Returns [`Error::Validation`](crate::error::Error::Validation) for a missing embedding, a
/// mixed dimension, or an internal gap in the snapshot, and
/// [`Error::External`](crate::error::Error::External) for a serialization failure.
pub(super) fn encode(input: EncodeInput) -> Result<EncodedIndex> {
    let EncodeInput {
        algorithm,
        ids,
        base,
        fresh,
        reuse_ivf,
    } = input;
    let encoded_items = fresh.len();
    let mut scalar = String::new();
    let mut rows: Vec<&[f32]> = Vec::with_capacity(ids.len());
    let mut dimension = 0_usize;
    for id in &ids {
        let entry = fresh.get(id).or_else(|| base.get(id)).ok_or_else(|| {
            validation(
                "experience",
                format!("the item `{id}` was not encoded; retry the persist"),
            )
        })?;
        scalar.push_str(&entry.line);
        scalar.push('\n');
        dimension = dimension.max(entry.row.len());
        rows.push(entry.row.as_ref());
    }
    if let Some((id, row)) = ids
        .iter()
        .zip(&rows)
        .find(|(_, row)| row.len() != dimension)
    {
        return Err(validation(
            "embedding",
            format!(
                "the item `{id}` has {} dimensions but the index dimension is {dimension}",
                row.len()
            ),
        ));
    }

    let vectors = serde_json::to_vec(&VectorFileRef {
        dimension,
        ids: &ids,
        rows,
    })
    .map_err(|error| Error::external(error).with_context("encoding vector/embeddings.json"))?;

    let (ivf, coarse, cached_ivf) = if algorithm.is_ivf() {
        let nlist = effective_nlist(algorithm.nlist_request(), ids.len());
        let (quantiser, bytes) =
            encode_quantiser(&algorithm, &ids, &base, &fresh, reuse_ivf, nlist)?;
        let cached = CachedQuantiser {
            coarse: Arc::clone(&quantiser),
            bytes: bytes.clone(),
            trained_count: ids.len(),
        };
        (Some(bytes), Some(quantiser), Some(cached))
    } else {
        (None, None, None)
    };

    let meta = ExperienceIndexMeta {
        version: MANIFEST_VERSION,
        vector_count: ids.len(),
        vector_algorithm: algorithm.canonical().to_string(),
        vector_dimension: dimension,
        embeddings_sha256: sha256_hex(&vectors),
        scalar_sha256: sha256_hex(scalar.as_bytes()),
        ivf_sha256: ivf.as_deref().map(sha256_hex),
    };
    Ok(EncodedIndex {
        scalar: scalar.into_bytes(),
        vectors,
        ivf,
        meta,
        coarse,
        cached_ivf,
        fresh,
        encoded_items,
    })
}

/// Builds the coarse quantiser of an IVF publish, reusing the cached one when allowed.
///
/// The returned quantiser and its serialized bytes are what the next publish caches.
fn encode_quantiser(
    algorithm: &AlgorithmSpec,
    ids: &[String],
    base: &BTreeMap<String, CacheEntry>,
    fresh: &BTreeMap<String, CacheEntry>,
    reuse_ivf: Option<CachedQuantiser>,
    nlist: usize,
) -> Result<(Arc<CoarseIndex>, Vec<u8>)> {
    let reusable = reuse_ivf.filter(|cached| reusable_quantiser(cached, ids, nlist));
    if let Some(cached) = reusable {
        // Nothing moved: the stored bytes and centroids are already correct, so the publish
        // stays byte-identical (and the retrieval results unchanged).
        if cached.coarse.ids() == ids {
            return Ok((cached.coarse, cached.bytes));
        }
        let vectors = owned_rows(ids, base, fresh)?;
        return match cached.coarse.assign_all(ids, &vectors) {
            Ok(coarse) => {
                let bytes = serialize_ivf(algorithm, &coarse)?;
                Ok((Arc::new(coarse), bytes))
            }
            Err(error) => {
                // The policy only reuses a quantiser whose ids all survive, so this is
                // unreachable; rebuilding is still the correct answer if it ever happens.
                warn!(
                    %error,
                    "reusing the IVF quantiser failed; rebuilding it from the catalog"
                );
                let coarse = CoarseIndex::build(ids, &vectors, nlist)?;
                let bytes = serialize_ivf(algorithm, &coarse)?;
                Ok((Arc::new(coarse), bytes))
            }
        };
    }
    let vectors = owned_rows(ids, base, fresh)?;
    let coarse = CoarseIndex::build(ids, &vectors, nlist)?;
    let bytes = serialize_ivf(algorithm, &coarse)?;
    Ok((Arc::new(coarse), bytes))
}

/// Reports whether an IVF publish may keep the cached centroids.
///
/// The documented policy is a rebuild when any of these holds:
///
/// - there is no cached quantiser (the first publish, or a removal dropped it);
/// - the list count the catalog calls for ([`effective_nlist`]) differs from the trained one,
///   which covers the ramp while fewer items than the requested `nlist` are stored;
/// - an id the quantiser knows is no longer stored (a removal, defensively);
/// - the catalog grew by at least `max(trained / 4, IVF_REBUILD_MIN_NEW_ITEMS)` items.
///
/// Otherwise a new item is assigned to the list of its nearest existing centroid, so `n`
/// appends cost `O(log n)` rebuilds rather than one per add.
fn reusable_quantiser(cached: &CachedQuantiser, ids: &[String], nlist: usize) -> bool {
    let trained = cached.trained_count;
    let growth_limit = trained.saturating_add(growth_step(trained));
    ids.len() >= trained
        && ids.len() < growth_limit
        && cached.coarse.nlist() == nlist
        && cached
            .coarse
            .ids()
            .iter()
            .all(|id| ids.binary_search(id).is_ok())
}

/// The number of appended items that makes an IVF rebuild worthwhile.
fn growth_step(trained: usize) -> usize {
    (trained / IVF_REBUILD_GROWTH_DIVISOR).max(IVF_REBUILD_MIN_NEW_ITEMS)
}

/// Owns the rows of `ids` for the quantiser API, which takes `&[Vec<f32>]`.
fn owned_rows(
    ids: &[String],
    base: &BTreeMap<String, CacheEntry>,
    fresh: &BTreeMap<String, CacheEntry>,
) -> Result<Vec<Vec<f32>>> {
    ids.iter()
        .map(|id| {
            fresh
                .get(id)
                .or_else(|| base.get(id))
                .map(|entry| entry.row.to_vec())
                .ok_or_else(|| {
                    validation(
                        "experience",
                        format!("the item `{id}` was not encoded; retry the persist"),
                    )
                })
        })
        .collect()
}

/// Serializes a trained quantiser as `vector/ivf.json`.
fn serialize_ivf(algorithm: &AlgorithmSpec, coarse: &CoarseIndex) -> Result<Vec<u8>> {
    let document = IvfFile {
        algorithm: algorithm.canonical().to_string(),
        dimension: coarse.dimension(),
        nlist: coarse.nlist(),
        ids: coarse.ids().to_vec(),
        centroids: coarse.centroids().to_vec(),
        lists: coarse.lists().to_vec(),
    };
    serde_json::to_vec(&document)
        .map_err(|error| Error::external(error).with_context("encoding vector/ivf.json"))
}

/// A publish fault injected by the regression tests (`None` = never).
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct WriteFault {
    /// Fail the n-th temp-file write (1-based); the destination files stay untouched.
    pub(super) fail_at: Option<usize>,
}

/// Writes one encoded snapshot under `dir`, publishing all files atomically as a set.
///
/// Every artifact — `vector/embeddings.json`, `scalar/metadata.jsonl`, `vector/ivf.json` when
/// there is one, and the `meta.json` that vouches for them — is first written to a unique
/// temp sibling and flushed. Only then are the temps renamed over their destinations, the
/// manifest last, and both directories synced. A failure while writing a temp therefore
/// leaves the previous snapshot fully intact and loadable, never a new data file next to an
/// old manifest.
///
/// Publishes into one directory are serialised by a process-wide lock keyed by the
/// directory's canonical path, so two indexes (or two concurrent `add`s) cannot interleave
/// their renames.
///
/// # Errors
///
/// Returns [`Error::External`] for an I/O failure, or the injected [`WriteFault`] error.
pub(super) async fn write(dir: &Path, encoded: &EncodedIndex, fault: WriteFault) -> Result<()> {
    let paths = IndexPaths::new(dir);
    let meta = serde_json::to_vec_pretty(&encoded.meta)
        .map_err(|error| Error::external(error).with_context("encoding meta.json"))?;
    let mut files: Vec<(&Path, &[u8])> = vec![
        (&paths.vectors, &encoded.vectors),
        (&paths.scalar, &encoded.scalar),
    ];
    if let Some(ivf) = &encoded.ivf {
        files.push((&paths.ivf, ivf));
    }
    files.push((&paths.meta, &meta));

    let lock = publish_lock(dir);
    let _guard = lock.lock().await;

    // Phase one: write every temp and flush it. Nothing reachable changes yet.
    let mut temporary_files: Vec<PathBuf> = Vec::with_capacity(files.len());
    for (number, (path, bytes)) in files.iter().enumerate() {
        if fault.fail_at == Some(number + 1) {
            discard(&temporary_files).await;
            return Err(Error::external(std::io::Error::other(format!(
                "injected publish failure before `{}`",
                path.display()
            ))));
        }
        match write_temporary(path, bytes).await {
            Ok(temporary) => {
                record_write(dir, path);
                temporary_files.push(temporary);
            }
            Err(error) => {
                discard(&temporary_files).await;
                return Err(error);
            }
        }
    }

    // Phase two: rename the temps over their destinations, the manifest last so it stays the
    // commit point of the snapshot.
    for ((path, _), temporary) in files.iter().zip(&temporary_files) {
        publish_temporary(temporary, path).await?;
    }
    sync_directory(dir).await?;
    if let Some(vector_dir) = paths.vectors.parent() {
        sync_directory(vector_dir).await?;
    }
    Ok(())
}

/// Removes the temp files of an abandoned publish, best effort.
async fn discard(temporary_files: &[PathBuf]) {
    for temporary in temporary_files {
        let _ = tokio::fs::remove_file(temporary).await;
    }
}

/// The process-wide publish lock of `dir`.
///
/// The registry is a documented piece of global state: it holds only `Weak` handles, so an
/// entry disappears with the last [`ExperienceIndex`] publishing into that directory. The key
/// is the canonical path when it exists, so two spellings of one directory share a lock.
fn publish_lock(dir: &Path) -> Arc<tokio::sync::Mutex<()>> {
    use std::sync::{Mutex, OnceLock};

    static LOCKS: OnceLock<Mutex<BTreeMap<PathBuf, Weak<tokio::sync::Mutex<()>>>>> =
        OnceLock::new();
    let key = std::fs::canonicalize(dir).unwrap_or_else(|_| {
        dir.parent().map_or_else(
            || dir.to_path_buf(),
            |parent| {
                let name = dir.file_name().unwrap_or_default();
                std::fs::canonicalize(parent).map_or_else(|_| dir.to_path_buf(), |p| p.join(name))
            },
        )
    });
    let registry = LOCKS.get_or_init(|| Mutex::new(BTreeMap::new()));
    let mut locks = registry.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(existing) = locks.get(&key).and_then(Weak::upgrade) {
        return existing;
    }
    let lock = Arc::new(tokio::sync::Mutex::new(()));
    locks.insert(key, Arc::downgrade(&lock));
    lock
}

/// The published-file counters the regression tests read, keyed by directory and file name.
#[cfg(test)]
static TEST_WRITES: std::sync::OnceLock<std::sync::Mutex<BTreeMap<(PathBuf, String), usize>>> =
    std::sync::OnceLock::new();

/// Records one published file, so the regression tests can bound the write count per catalog.
#[cfg(test)]
fn record_write(dir: &Path, path: &Path) {
    let name = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    let registry = TEST_WRITES.get_or_init(|| std::sync::Mutex::new(BTreeMap::new()));
    let mut writes = registry.lock().unwrap_or_else(PoisonError::into_inner);
    *writes.entry((dir.to_path_buf(), name)).or_default() += 1;
}

/// A no-op outside the crate's unit tests.
#[cfg(not(test))]
fn record_write(_dir: &Path, _path: &Path) {}

/// The file-write counts of `dir`, keyed by file name (unit-test observability).
#[cfg(test)]
pub(super) fn write_counts(dir: &Path) -> BTreeMap<String, usize> {
    let registry = TEST_WRITES.get_or_init(|| std::sync::Mutex::new(BTreeMap::new()));
    let writes = registry.lock().unwrap_or_else(PoisonError::into_inner);
    writes
        .iter()
        .filter(|((written, _), _)| written == dir)
        .map(|((_, name), count)| (name.clone(), *count))
        .collect()
}

/// Reads the index under `dir`; an absent or empty directory yields an empty index.
pub(super) async fn load(
    dir: &Path,
    embedding: Arc<dyn Embedding>,
    algorithm: AlgorithmSpec,
) -> Result<ExperienceIndex> {
    let paths = IndexPaths::new(dir);
    if !index_files_present(&paths).await? {
        return Ok(ExperienceIndex::from_parts(
            Some(dir.to_path_buf()),
            embedding,
            BTreeMap::new(),
            1,
            algorithm,
            ExperienceIndexMeta::default(),
            None,
        ));
    }

    let meta_bytes = read_bytes(&paths.meta).await?;
    let meta: ExperienceIndexMeta =
        serde_json::from_slice(&meta_bytes).map_err(|error| corrupt(&paths.meta, error))?;
    let persisted = validate_meta(&paths.meta, &meta)?;
    if persisted != algorithm {
        return Err(corrupt(
            &paths.meta,
            format!(
                "the directory stores the {:?} algorithm but {:?} was requested; open it with \
                 the algorithm it was built with",
                persisted.canonical(),
                algorithm.canonical()
            ),
        ));
    }

    let scalar_bytes = read_bytes(&paths.scalar).await?;
    let vector_bytes = read_bytes(&paths.vectors).await?;
    verify_digest(&paths.scalar, &meta.scalar_sha256, &scalar_bytes)?;
    verify_digest(&paths.vectors, &meta.embeddings_sha256, &vector_bytes)?;

    let mut items = decode_scalar(&paths.scalar, &scalar_bytes)?;
    let vector_ids = decode_vectors(&paths, &meta, &mut items, &vector_bytes)?;
    let (coarse, cached_ivf) = if persisted.is_ivf() {
        let (quantiser, bytes) = load_ivf(&paths, &meta, &vector_ids).await?;
        let quantiser = Arc::new(quantiser);
        let cached = CachedQuantiser {
            coarse: Arc::clone(&quantiser),
            bytes,
            trained_count: vector_ids.len(),
        };
        (Some(quantiser), Some(cached))
    } else {
        if meta.ivf_sha256.is_some() {
            return Err(corrupt(
                &paths.meta,
                format!(
                    "the {:?} algorithm records an ivf digest; delete the directory to rebuild \
                     it",
                    persisted.canonical()
                ),
            ));
        }
        (None, None)
    };

    let mut next_ordinal = 1_u64;
    for id in items.keys() {
        if let Some(ordinal) = ordinal_of(id) {
            next_ordinal = next_ordinal.max(ordinal.saturating_add(1));
        }
    }
    // Seed the cache from what was just loaded, so the first mutation after a reopen encodes
    // only the item it touched instead of the whole catalog.
    let entries = entries_from(&items)?;
    Ok(ExperienceIndex::from_parts(
        Some(dir.to_path_buf()),
        embedding,
        items,
        next_ordinal,
        persisted,
        meta,
        coarse,
    )
    .seed_encode_cache(EncodeCache {
        entries: Arc::new(entries),
        ivf: cached_ivf,
        dirty: BTreeSet::new(),
    }))
}

/// Rebuilds the per-item encoding cache from loaded items.
///
/// The scalar line is re-serialized rather than sliced out of the file, so the cache always
/// matches the parsed item exactly (and an identical catalog republishes identical bytes).
fn entries_from(items: &BTreeMap<String, ExperienceItem>) -> Result<BTreeMap<String, CacheEntry>> {
    let mut entries = BTreeMap::new();
    for (id, item) in items {
        entries.insert(id.clone(), cache_entry(item)?);
    }
    Ok(entries)
}

/// Reports whether the directory holds any index file, rejecting a partial set.
///
/// An absent (or empty) directory is a fresh index; a directory holding some but not all of
/// the core artifacts is a broken index that must not be repaired silently.
async fn index_files_present(paths: &IndexPaths) -> Result<bool> {
    let mut present = Vec::with_capacity(3);
    for path in paths.core() {
        present.push((exists(path).await?, path));
    }
    if present.iter().all(|(present, _)| !present) {
        return Ok(false);
    }
    for (present, path) in present {
        if !present {
            return Err(Error::external(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!(
                    "experience index {}: the file is missing while other index files are \
                     present; delete the directory to rebuild it",
                    path.display()
                ),
            )));
        }
    }
    Ok(true)
}

/// Validates the manifest version and returns its parsed algorithm.
///
/// The persisted name is parsed with the same grammar a caller uses, so a hand-edited
/// manifest naming an unimplemented algorithm is rejected here instead of at the first
/// search.
fn validate_meta(path: &Path, meta: &ExperienceIndexMeta) -> Result<AlgorithmSpec> {
    if meta.version != MANIFEST_VERSION {
        return Err(corrupt(
            path,
            format!(
                "unsupported manifest version {} (expected {MANIFEST_VERSION})",
                meta.version
            ),
        ));
    }
    AlgorithmSpec::parse(&meta.vector_algorithm).map_err(|error| corrupt(path, error.message()))
}

/// Decodes the scalar file: one item JSON per line, in id order.
fn decode_scalar(path: &Path, bytes: &[u8]) -> Result<BTreeMap<String, ExperienceItem>> {
    let text =
        String::from_utf8(bytes.to_vec()).map_err(|error| corrupt(path, error.utf8_error()))?;
    let mut items: BTreeMap<String, ExperienceItem> = BTreeMap::new();
    for (number, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let item: ExperienceItem = serde_json::from_str(line)
            .map_err(|error| corrupt(path, format!("line {}: {error}", number + 1)))?;
        if item.id.trim().is_empty() {
            return Err(corrupt(
                path,
                format!("line {}: the item has no id", number + 1),
            ));
        }
        if items.insert(item.id.clone(), item).is_some() {
            return Err(corrupt(
                path,
                format!("line {}: a duplicate item id", number + 1),
            ));
        }
    }
    Ok(items)
}

/// Decodes the vector file, rehydrates every embedding, and returns the persisted ids.
fn decode_vectors(
    paths: &IndexPaths,
    meta: &ExperienceIndexMeta,
    items: &mut BTreeMap<String, ExperienceItem>,
    bytes: &[u8],
) -> Result<Vec<String>> {
    let path = &paths.vectors;
    let text =
        String::from_utf8(bytes.to_vec()).map_err(|error| corrupt(path, error.utf8_error()))?;
    let vectors: VectorFile = serde_json::from_str(&text).map_err(|error| corrupt(path, error))?;
    if meta.vector_count != items.len() {
        return Err(corrupt(
            &paths.meta,
            format!(
                "vector_count {} does not match the {} items in {SCALAR_DIR}/{SCALAR_FILE}",
                meta.vector_count,
                items.len()
            ),
        ));
    }
    if vectors.ids.len() != items.len() {
        return Err(corrupt(
            path,
            format!(
                "{} ids do not match the {} items in {SCALAR_DIR}/{SCALAR_FILE}",
                vectors.ids.len(),
                items.len()
            ),
        ));
    }
    if vectors.rows.len() != vectors.ids.len() {
        return Err(corrupt(
            path,
            format!(
                "{} rows do not match the {} ids",
                vectors.rows.len(),
                vectors.ids.len()
            ),
        ));
    }
    if meta.vector_dimension != vectors.dimension {
        return Err(corrupt(
            path,
            format!(
                "the dimension {} does not match the manifest's {}",
                vectors.dimension, meta.vector_dimension
            ),
        ));
    }
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    for (id, row) in vectors.ids.iter().zip(&vectors.rows) {
        if row.len() != vectors.dimension {
            return Err(corrupt(
                path,
                format!(
                    "the row for `{id}` has {} values but the index dimension is {}",
                    row.len(),
                    vectors.dimension
                ),
            ));
        }
        if !seen.insert(id.as_str()) {
            return Err(corrupt(
                path,
                format!("the id `{id}` appears more than once"),
            ));
        }
        let Some(item) = items.get_mut(id) else {
            return Err(corrupt(
                path,
                format!("the id `{id}` is not present in {SCALAR_DIR}/{SCALAR_FILE}"),
            ));
        };
        item.embedding.clone_from(row);
    }
    Ok(vectors.ids)
}

/// Reads and validates the coarse quantiser of an IVF index, with its raw bytes.
///
/// The file must exist, match its manifest digest, name the same algorithm and dimension as
/// the manifest, and hold exactly the ids of the vector file; the coarse index then validates
/// its own geometry (list count, centroid dimension, list indexes). The bytes are returned so
/// the next publish of an unchanged catalog can republish them verbatim.
async fn load_ivf(
    paths: &IndexPaths,
    meta: &ExperienceIndexMeta,
    vector_ids: &[String],
) -> Result<(CoarseIndex, Vec<u8>)> {
    let path = &paths.ivf;
    let expected = meta.ivf_sha256.as_deref().ok_or_else(|| {
        corrupt(
            &paths.meta,
            format!(
                "the {:?} algorithm records no ivf_sha256 digest",
                meta.vector_algorithm
            ),
        )
    })?;
    let bytes = read_bytes(path).await?;
    verify_digest(path, expected, &bytes)?;
    let document: IvfFile = serde_json::from_slice(&bytes).map_err(|error| corrupt(path, error))?;
    if !document
        .algorithm
        .eq_ignore_ascii_case(&meta.vector_algorithm)
    {
        return Err(corrupt(
            path,
            format!(
                "the algorithm {:?} does not match the manifest's {:?}",
                document.algorithm, meta.vector_algorithm
            ),
        ));
    }
    if document.dimension != meta.vector_dimension {
        return Err(corrupt(
            path,
            format!(
                "the dimension {} does not match the manifest's {}",
                document.dimension, meta.vector_dimension
            ),
        ));
    }
    if document.ids != vector_ids {
        return Err(corrupt(
            path,
            format!(
                "the id set does not match the {} ids in {VECTOR_DIR}/{VECTORS_FILE}; delete the \
                 directory to rebuild it",
                vector_ids.len()
            ),
        ));
    }
    CoarseIndex::from_parts(
        document.dimension,
        document.nlist,
        document.ids,
        document.centroids,
        document.lists,
    )
    .map(move |coarse| (coarse, bytes))
    .map_err(|error| corrupt(path, error.message()))
}

/// Builds a validation error naming `path` for a corrupt or inconsistent artifact.
fn corrupt(path: &Path, reason: impl fmt::Display) -> Error {
    Error::validation(format!("experience index {}: {reason}", path.display()))
}

/// Verifies the SHA-256 digest the manifest records for `path`.
fn verify_digest(path: &Path, expected: &str, bytes: &[u8]) -> Result<()> {
    if expected.is_empty() {
        return Err(corrupt(
            path,
            "the manifest records no sha256 digest for this file",
        ));
    }
    let actual = sha256_hex(bytes);
    if !actual.eq_ignore_ascii_case(expected) {
        return Err(corrupt(
            path,
            format!(
                "sha256 mismatch: the manifest records {expected}, the file hashes to {actual}"
            ),
        ));
    }
    Ok(())
}

/// The SHA-256 hex digest of `bytes`.
fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

/// Reports whether `path` exists.
async fn exists(path: &Path) -> Result<bool> {
    tokio::fs::try_exists(path).await.map_err(|error| {
        Error::external(error).with_context(format!("checking `{}`", path.display()))
    })
}

/// Reads `path` as bytes.
async fn read_bytes(path: &Path) -> Result<Vec<u8>> {
    tokio::fs::read(path).await.map_err(|error| {
        Error::external(error).with_context(format!("reading `{}`", path.display()))
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use super::*;
    use crate::experience::index::FLAT_VECTOR_ALGORITHM;

    /// An item with a two-component vector, titled `id`.
    fn item(id: &str, pattern: &str) -> ExperienceItem {
        ExperienceItem::new(pattern)
            .with_id(id)
            .with_skill_ids(["skill"])
            .with_embedding(vec![1.0, 0.0])
    }

    /// Encodes an explicit item list as a fresh catalog (the "everything changed" path).
    fn encode_items(items: &[ExperienceItem], algorithm: &AlgorithmSpec) -> Result<EncodedIndex> {
        let mut ids = Vec::with_capacity(items.len());
        let mut fresh = BTreeMap::new();
        for item in items {
            ids.push(item.id.clone());
            fresh.insert(item.id.clone(), cache_entry(item)?);
        }
        encode(EncodeInput {
            algorithm: algorithm.clone(),
            ids,
            base: Arc::new(BTreeMap::new()),
            fresh,
            reuse_ivf: None,
        })
    }

    #[test]
    fn sha256_matches_the_reference_digest_of_the_empty_input() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn encoding_writes_one_line_per_item_and_records_the_dimension() {
        let items = vec![item("exp_0002", "second"), item("exp_0001", "first")];
        let encoded = encode_items(&items, &AlgorithmSpec::flat()).expect("encode");
        assert_eq!(encoded.meta.vector_count, 2);
        assert_eq!(encoded.meta.vector_dimension, 2);
        assert_eq!(encoded.meta.vector_algorithm, FLAT_VECTOR_ALGORITHM);
        assert_eq!(encoded.meta.embeddings_sha256, sha256_hex(&encoded.vectors));
        assert_eq!(encoded.meta.scalar_sha256, sha256_hex(&encoded.scalar));
        assert!(encoded.ivf.is_none());
        assert!(encoded.coarse.is_none());
        assert!(encoded.meta.ivf_sha256.is_none());

        let vectors: VectorFile = serde_json::from_slice(&encoded.vectors).expect("decode vectors");
        assert_eq!(vectors.ids, vec!["exp_0002", "exp_0001"]);
        assert_eq!(vectors.dimension, 2);
        assert_eq!(vectors.rows.len(), 2);
        assert_eq!(
            String::from_utf8(encoded.scalar)
                .expect("utf8")
                .lines()
                .count(),
            2
        );
    }

    #[test]
    fn an_ivf_index_encodes_its_coarse_quantiser_and_a_digest() {
        let spec = AlgorithmSpec::parse("IVF4").expect("ivf4");
        let items = vec![
            item("exp_0001", "first"),
            item("exp_0002", "second"),
            item("exp_0003", "third"),
        ];
        let encoded = encode_items(&items, &spec).expect("encode");
        assert_eq!(encoded.meta.vector_algorithm, "ivf4");
        assert_eq!(
            encoded.meta.ivf_sha256.as_deref(),
            encoded.ivf.as_deref().map(sha256_hex).as_deref()
        );
        let coarse = encoded.coarse.as_deref().expect("coarse");
        // `ivf4` over three items clamps to three lists.
        assert_eq!(coarse.nlist(), 3);
        let document: IvfFile =
            serde_json::from_slice(encoded.ivf.as_deref().expect("bytes")).expect("decode ivf");
        assert_eq!(document.algorithm, "ivf4");
        assert_eq!(document.dimension, 2);
        assert_eq!(document.nlist, 3);
        assert_eq!(document.ids, vec!["exp_0001", "exp_0002", "exp_0003"]);
        assert_eq!(document.centroids.len(), 3);
        assert_eq!(document.lists.len(), 3);
        // Repeated encodes are byte-identical, so the digest is stable.
        let repeated = encode_items(&items, &spec).expect("encode again");
        assert_eq!(repeated.ivf, encoded.ivf);
        assert_eq!(repeated.meta, encoded.meta);
    }

    #[test]
    fn an_empty_ivf_index_encodes_an_empty_quantiser() {
        let spec = AlgorithmSpec::parse("ivf64").expect("ivf64");
        let encoded = encode_items(&[], &spec).expect("encode");
        assert_eq!(encoded.meta.vector_count, 0);
        assert_eq!(encoded.meta.vector_dimension, 0);
        assert!(encoded.meta.ivf_sha256.is_some());
        let document: IvfFile =
            serde_json::from_slice(encoded.ivf.as_deref().expect("bytes")).expect("decode ivf");
        assert_eq!(document.nlist, 0);
        assert!(document.ids.is_empty());
        assert!(document.centroids.is_empty());
        assert!(document.lists.is_empty());
    }

    #[test]
    fn encoding_rejects_items_without_an_embedding_or_with_mixed_dimensions() {
        let missing = vec![ExperienceItem::new("orphan").with_id("exp_0001")];
        let error = encode_items(&missing, &AlgorithmSpec::flat()).expect_err("missing embedding");
        assert!(error.message().contains("embedding"));

        let mixed = vec![
            item("exp_0001", "first"),
            ExperienceItem::new("second")
                .with_id("exp_0002")
                .with_embedding(vec![1.0, 0.0, 0.0]),
        ];
        let error = encode_items(&mixed, &AlgorithmSpec::flat()).expect_err("mixed dimensions");
        assert!(
            error.message().contains("index dimension is 3"),
            "{}",
            error.message()
        );
    }

    #[test]
    fn corrupt_errors_name_the_file_and_keep_the_validation_kind() {
        let error = corrupt(Path::new("/tmp/meta.json"), "bad digest");
        assert_eq!(error.kind(), crate::error::ErrorKind::Validation);
        assert!(error.message().contains("/tmp/meta.json"));
        assert!(error.message().contains("bad digest"));
    }

    #[test]
    fn digest_verification_rejects_missing_and_mismatched_digests() {
        let path = Path::new("scalar/metadata.jsonl");
        assert!(verify_digest(path, &sha256_hex(b"payload"), b"payload").is_ok());
        assert!(verify_digest(path, "", b"payload").is_err());
        assert!(verify_digest(path, &sha256_hex(b"other"), b"payload").is_err());
    }

    #[test]
    fn the_manifest_validation_parses_the_algorithm_and_names_the_file() {
        let path = Path::new("meta.json");
        let mut meta = ExperienceIndexMeta::default();
        assert_eq!(
            validate_meta(path, &meta).expect("flat"),
            AlgorithmSpec::flat()
        );

        meta.vector_algorithm = "IVF4096,PQ16".to_string();
        assert_eq!(
            validate_meta(path, &meta).expect("ivf").canonical(),
            "ivf4096,pq16"
        );

        meta.vector_algorithm = "hnsw".to_string();
        let error = validate_meta(path, &meta).expect_err("unknown");
        assert!(error.message().contains("meta.json"));
        assert!(error.message().contains("vector_algorithm"));

        meta.vector_algorithm = "flat".to_string();
        meta.version = MANIFEST_VERSION + 1;
        let error = validate_meta(path, &meta).expect_err("version");
        assert!(error.message().contains("unsupported manifest version"));
    }

    /// The embedder the load tests hand to [`load`].
    fn embedding() -> Arc<dyn Embedding> {
        Arc::new(openjiuwen_retrieval::HashEmbedding::new(2))
    }

    /// Three two-axis items with distinct vectors, in id order.
    fn catalog() -> Vec<ExperienceItem> {
        vec![
            item("exp_0001", "deploy the service"),
            item("exp_0002", "summarize the report"),
            ExperienceItem::new("bake a cake")
                .with_id("exp_0003")
                .with_skill_ids(["skill"])
                .with_embedding(vec![0.0, 1.0]),
        ]
    }

    /// The parsed `ivf` specification the load tests reopen with.
    fn ivf_spec() -> AlgorithmSpec {
        AlgorithmSpec::parse("ivf2").expect("ivf2")
    }

    #[tokio::test]
    async fn an_ivf_publication_round_trips_through_the_disk() {
        let dir = tempfile::tempdir().expect("tempdir");
        let spec = ivf_spec();
        let items = catalog();
        let encoded = encode_items(&items, &spec).expect("encode");
        write(dir.path(), &encoded, WriteFault::default())
            .await
            .expect("write");
        let paths = IndexPaths::new(dir.path());
        assert!(tokio::fs::try_exists(&paths.ivf).await.expect("stat"));

        let loaded = load(dir.path(), embedding(), ivf_spec())
            .await
            .expect("load");
        assert_eq!(loaded.algorithm(), "ivf2");
        assert_eq!(loaded.nlist(), 2);
        assert_eq!(loaded.items(), items);
        assert_eq!(loaded.meta(), encoded.meta);

        // The quantiser came from the file, so republishing writes the same bytes.
        loaded.persist().await.expect("persist");
        let coarse = tokio::fs::read(&paths.ivf).await.expect("read coarse");
        assert_eq!(Some(coarse), encoded.ivf);
        assert_eq!(loaded.meta(), encoded.meta);

        // No temporary sibling survives a publication.
        let vector_dir = paths.vectors.parent().expect("vector directory");
        let mut entries = tokio::fs::read_dir(vector_dir)
            .await
            .expect("read vector dir");
        while let Some(entry) = entries.next_entry().await.expect("entry") {
            let name = entry.file_name().to_string_lossy().into_owned();
            let temporary = entry
                .path()
                .extension()
                .is_some_and(|extension| extension.eq_ignore_ascii_case("tmp"));
            assert!(!temporary, "{name}");
        }
    }

    #[tokio::test]
    async fn a_tampered_or_missing_coarse_file_is_rejected() {
        let dir = tempfile::tempdir().expect("tempdir");
        let encoded = encode_items(&catalog(), &ivf_spec()).expect("encode");
        write(dir.path(), &encoded, WriteFault::default())
            .await
            .expect("write");
        let paths = IndexPaths::new(dir.path());

        // A single flipped byte breaks the manifest digest.
        let mut tampered = encoded.ivf.clone().expect("coarse bytes");
        if let Some(last) = tampered.last_mut() {
            *last ^= 0x01;
        }
        tokio::fs::write(&paths.ivf, &tampered)
            .await
            .expect("write coarse");
        let error = load(dir.path(), embedding(), ivf_spec())
            .await
            .expect_err("digest");
        assert_eq!(error.kind(), crate::error::ErrorKind::Validation);
        assert!(error.message().contains("ivf.json"));
        assert!(error.message().contains("sha256"));

        // A missing file is an I/O failure that still names it.
        tokio::fs::remove_file(&paths.ivf)
            .await
            .expect("remove coarse");
        let error = load(dir.path(), embedding(), ivf_spec())
            .await
            .expect_err("missing");
        assert_eq!(error.kind(), crate::error::ErrorKind::External);
        assert!(error.message().contains("ivf.json"));
    }

    #[tokio::test]
    async fn a_flat_publication_writes_no_coarse_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let encoded = encode_items(&catalog(), &AlgorithmSpec::flat()).expect("encode");
        write(dir.path(), &encoded, WriteFault::default())
            .await
            .expect("write");
        let paths = IndexPaths::new(dir.path());
        assert!(!tokio::fs::try_exists(&paths.ivf).await.expect("stat"));

        let loaded = load(dir.path(), embedding(), AlgorithmSpec::flat())
            .await
            .expect("load");
        assert_eq!(loaded.count(), 3);
        assert_eq!(loaded.nlist(), 0);
        assert!(loaded.meta().ivf_sha256.is_none());

        // A manifest without the field is what an older build wrote; it still opens.
        let meta_text = tokio::fs::read_to_string(&paths.meta)
            .await
            .expect("read meta");
        assert!(!meta_text.contains("ivf_sha256"));

        // Requesting another algorithm than the manifest records is a loud error.
        let error = load(
            dir.path(),
            embedding(),
            AlgorithmSpec::parse("ivf").expect("ivf"),
        )
        .await
        .expect_err("mismatch");
        assert_eq!(error.kind(), crate::error::ErrorKind::Validation);
        assert!(error.message().contains("meta.json"));
        assert!(error.message().contains("flat"));
    }

    /// The embedder the publish regression tests hand to the index.
    fn test_embedding() -> Arc<dyn Embedding> {
        Arc::new(openjiuwen_retrieval::HashEmbedding::new(4))
    }

    /// A searchable item with a four-axis vector derived from its pattern.
    fn embedded(pattern: &str) -> ExperienceItem {
        let mut item = ExperienceItem::new(pattern).with_skill_ids(["skill"]);
        item.embedding = vec![
            if pattern.contains("alpha") { 1.0 } else { 0.0 },
            if pattern.contains("beta") { 1.0 } else { 0.0 },
            if pattern.contains("gamma") { 1.0 } else { 0.0 },
            0.25,
        ];
        item
    }

    #[tokio::test]
    async fn appends_encode_one_item_each_and_bound_the_publishes() {
        let dir = tempfile::tempdir().expect("tempdir");
        let index = ExperienceIndex::open_with_algorithm(dir.path(), test_embedding(), "ivf")
            .await
            .expect("open");
        let additions = 9_usize;
        for ordinal in 0..additions {
            let mut item = embedded(&format!("alpha pattern {ordinal}"));
            item.id = format!("exp_{:04}", ordinal + 1);
            index.add(item).await.expect("add");
        }
        let (encoded, publishes) = index.persist_counters();
        // One item serialized per add: `n` appends cost `O(n)` encoding work in total.
        assert_eq!(encoded, additions as u64);
        assert_eq!(publishes, additions as u64);
        // Each publish writes a bounded set of files, and the coarse quantiser is only
        // rewritten when the rebuild policy fires (here: while the list count ramps up).
        let writes = write_counts(dir.path());
        assert_eq!(writes.get("embeddings.json"), Some(&additions));
        assert_eq!(writes.get("metadata.jsonl"), Some(&additions));
        assert_eq!(writes.get("meta.json"), Some(&additions));
        let ivf_writes = writes.get("ivf.json").copied().unwrap_or_default();
        assert!(ivf_writes <= additions, "{writes:?}");
        // The newest item is still searched like any other.
        let hits = index
            .search(
                "alpha pattern 8",
                &crate::experience::ExperienceRanking::new(1),
            )
            .await
            .expect("search");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id(), "exp_0009");
        assert_eq!(index.meta().vector_count, additions);
    }

    #[tokio::test]
    async fn republishing_an_unchanged_catalog_reuses_the_cached_encoding() {
        let dir = tempfile::tempdir().expect("tempdir");
        let index = ExperienceIndex::open_with_algorithm(dir.path(), test_embedding(), "ivf")
            .await
            .expect("open");
        let mut item = embedded("alpha");
        item.id = "exp_0001".to_string();
        index.add(item).await.expect("add");
        let (encoded, publishes) = index.persist_counters();
        let vectors_before = tokio::fs::read(IndexPaths::new(dir.path()).vectors)
            .await
            .expect("read vectors");
        let ivf_before = tokio::fs::read(IndexPaths::new(dir.path()).ivf)
            .await
            .expect("read ivf");
        let hits_before = index
            .search("alpha", &crate::experience::ExperienceRanking::new(0))
            .await
            .expect("search");

        index.persist().await.expect("republish");
        // No item was re-serialized, and the artifacts are byte-identical.
        assert_eq!(index.persist_counters().0, encoded);
        assert_eq!(index.persist_counters().1, publishes + 1);
        assert_eq!(
            tokio::fs::read(IndexPaths::new(dir.path()).vectors)
                .await
                .expect("read vectors"),
            vectors_before
        );
        assert_eq!(
            tokio::fs::read(IndexPaths::new(dir.path()).ivf)
                .await
                .expect("read ivf"),
            ivf_before
        );
        // Retrieval results are preserved for an unchanged index.
        let hits_after = index
            .search("alpha", &crate::experience::ExperienceRanking::new(0))
            .await
            .expect("search after");
        assert_eq!(hits_after, hits_before);
    }

    #[tokio::test]
    async fn a_reopened_index_only_encodes_the_item_it_touches() {
        let dir = tempfile::tempdir().expect("tempdir");
        let index = ExperienceIndex::open(dir.path(), test_embedding())
            .await
            .expect("open");
        for pattern in ["alpha", "beta", "gamma"] {
            index.add(embedded(pattern)).await.expect("add");
        }
        let reopened = ExperienceIndex::open(dir.path(), test_embedding())
            .await
            .expect("reopen");
        reopened.add(embedded("alpha again")).await.expect("add");
        assert_eq!(reopened.persist_counters().0, 1);
    }

    #[tokio::test]
    async fn an_injected_publish_failure_leaves_the_previous_snapshot_loadable() {
        for fail_at in 1..=4 {
            let dir = tempfile::tempdir().expect("tempdir");
            let before = catalog();
            let encoded = encode_items(&before, &ivf_spec()).expect("encode");
            write(dir.path(), &encoded, WriteFault::default())
                .await
                .expect("first write");

            let mut after = before.clone();
            after.push(item("exp_0004", "fourth"));
            let encoded_after = encode_items(&after, &ivf_spec()).expect("encode after");
            let error = write(
                dir.path(),
                &encoded_after,
                WriteFault {
                    fail_at: Some(fail_at),
                },
            )
            .await
            .expect_err("injected failure");
            assert!(error.message().contains("injected publish failure"));

            // The directory still holds exactly the previous, complete snapshot.
            let loaded = load(dir.path(), embedding(), ivf_spec())
                .await
                .expect("load after failure");
            assert_eq!(loaded.count(), before.len());
            assert_eq!(
                loaded.meta().vector_dimension,
                encoded.meta.vector_dimension
            );
            assert_eq!(
                loaded.meta().embeddings_sha256,
                encoded.meta.embeddings_sha256
            );
            // No scratch file survives the failed publish.
            let vector_dir = IndexPaths::new(dir.path())
                .vectors
                .parent()
                .expect("dir")
                .to_path_buf();
            let mut entries = tokio::fs::read_dir(&vector_dir).await.expect("read dir");
            while let Some(entry) = entries.next_entry().await.expect("entry") {
                let name = entry.file_name().to_string_lossy().into_owned();
                let extension = std::path::Path::new(&name)
                    .extension()
                    .and_then(std::ffi::OsStr::to_str);
                assert_ne!(extension, Some("tmp"), "{name}");
            }
        }
    }
}
