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
use std::sync::Arc;

use openjiuwen_retrieval::Embedding;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{validation, Error, Result};
use crate::experience::index::algorithm::{effective_nlist, AlgorithmSpec};
use crate::experience::index::ivf::CoarseIndex;
use crate::experience::index::{ExperienceIndex, ExperienceIndexMeta, MANIFEST_VERSION};
use crate::experience::store::write_atomic;
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
}

/// Serializes a catalog into the persisted artifacts and their manifest.
///
/// Items are written in id order, and every item must carry an embedding: the vector file is
/// a dense matrix, so a missing row would misalign every later one. An IVF index additionally
/// trains its coarse quantiser here, which is the only place the centroids are derived —
/// loading reuses the persisted state instead of retraining.
pub(super) fn encode(items: &[ExperienceItem], algorithm: &AlgorithmSpec) -> Result<EncodedIndex> {
    let mut scalar = String::new();
    let mut ids = Vec::with_capacity(items.len());
    let mut rows = Vec::with_capacity(items.len());
    let mut dimension = 0_usize;
    for item in items {
        let line = serde_json::to_string(item).map_err(|error| {
            Error::external(error).with_context(format!("encoding item `{}`", item.id))
        })?;
        scalar.push_str(&line);
        scalar.push('\n');
        if item.embedding.is_empty() {
            return Err(validation(
                "embedding",
                format!("the item `{}` has no embedding", item.id),
            ));
        }
        dimension = dimension.max(item.embedding.len());
        ids.push(item.id.clone());
        rows.push(item.embedding.clone());
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

    let vectors = serde_json::to_vec(&VectorFile {
        dimension,
        ids: ids.clone(),
        rows: rows.clone(),
    })
    .map_err(|error| Error::external(error).with_context("encoding vector/embeddings.json"))?;

    let (ivf, coarse) = if algorithm.is_ivf() {
        let nlist = effective_nlist(algorithm.nlist_request(), items.len());
        let coarse = CoarseIndex::build(&ids, &rows, nlist)?;
        let document = IvfFile {
            algorithm: algorithm.canonical().to_string(),
            dimension: coarse.dimension(),
            nlist: coarse.nlist(),
            ids: coarse.ids().to_vec(),
            centroids: coarse.centroids().to_vec(),
            lists: coarse.lists().to_vec(),
        };
        let bytes = serde_json::to_vec(&document)
            .map_err(|error| Error::external(error).with_context("encoding vector/ivf.json"))?;
        (Some(bytes), Some(Arc::new(coarse)))
    } else {
        (None, None)
    };

    let meta = ExperienceIndexMeta {
        version: MANIFEST_VERSION,
        vector_count: items.len(),
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
    })
}

/// Writes one encoded snapshot under `dir`, in dependency order.
///
/// The vector file lands first, then the scalar file, then the coarse quantiser when there is
/// one, then the manifest: a crash therefore leaves a directory whose manifest still
/// describes the previous snapshot, which the next load rejects instead of mixing two
/// catalogs.
pub(super) async fn write(dir: &Path, encoded: &EncodedIndex) -> Result<()> {
    let paths = IndexPaths::new(dir);
    write_atomic(&paths.vectors, &encoded.vectors).await?;
    write_atomic(&paths.scalar, &encoded.scalar).await?;
    if let Some(ivf) = &encoded.ivf {
        write_atomic(&paths.ivf, ivf).await?;
    }
    let meta = serde_json::to_vec_pretty(&encoded.meta)
        .map_err(|error| Error::external(error).with_context("encoding meta.json"))?;
    write_atomic(&paths.meta, &meta).await
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
    let coarse = if persisted.is_ivf() {
        Some(Arc::new(load_ivf(&paths, &meta, &vector_ids).await?))
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
        None
    };

    let mut next_ordinal = 1_u64;
    for id in items.keys() {
        if let Some(ordinal) = ordinal_of(id) {
            next_ordinal = next_ordinal.max(ordinal.saturating_add(1));
        }
    }
    Ok(ExperienceIndex::from_parts(
        Some(dir.to_path_buf()),
        embedding,
        items,
        next_ordinal,
        persisted,
        meta,
        coarse,
    ))
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

/// Reads and validates the coarse quantiser of an IVF index.
///
/// The file must exist, match its manifest digest, name the same algorithm and dimension as
/// the manifest, and hold exactly the ids of the vector file; the coarse index then validates
/// its own geometry (list count, centroid dimension, list indexes).
async fn load_ivf(
    paths: &IndexPaths,
    meta: &ExperienceIndexMeta,
    vector_ids: &[String],
) -> Result<CoarseIndex> {
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
        let encoded = encode(&items, &AlgorithmSpec::flat()).expect("encode");
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
        let encoded = encode(&items, &spec).expect("encode");
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
        let repeated = encode(&items, &spec).expect("encode again");
        assert_eq!(repeated.ivf, encoded.ivf);
        assert_eq!(repeated.meta, encoded.meta);
    }

    #[test]
    fn an_empty_ivf_index_encodes_an_empty_quantiser() {
        let spec = AlgorithmSpec::parse("ivf64").expect("ivf64");
        let encoded = encode(&[], &spec).expect("encode");
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
        let error = encode(&missing, &AlgorithmSpec::flat()).expect_err("missing embedding");
        assert!(error.message().contains("embedding"));

        let mixed = vec![
            item("exp_0001", "first"),
            ExperienceItem::new("second")
                .with_id("exp_0002")
                .with_embedding(vec![1.0, 0.0, 0.0]),
        ];
        let error = encode(&mixed, &AlgorithmSpec::flat()).expect_err("mixed dimensions");
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
        let encoded = encode(&items, &spec).expect("encode");
        write(dir.path(), &encoded).await.expect("write");
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
        let encoded = encode(&catalog(), &ivf_spec()).expect("encode");
        write(dir.path(), &encoded).await.expect("write");
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
        let encoded = encode(&catalog(), &AlgorithmSpec::flat()).expect("encode");
        write(dir.path(), &encoded).await.expect("write");
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
}
