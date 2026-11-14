//! The on-disk layout of a persisted experience index.
//!
//! Three files under one directory, written in dependency order and read all-or-nothing:
//!
//! ```text
//! <dir>/
//! ├── meta.json              # integrity manifest (`ExperienceIndexMeta`)
//! ├── scalar/
//! │   └── metadata.jsonl     # one `ExperienceItem` JSON per line, sorted by id
//! └── vector/
//!     └── embeddings.json    # {"dimension", "ids", "rows"}
//! ```
//!
//! Python's layout replaces the last artifact with `vector/faiss_index.bin` plus
//! `vector/embeddings.npy`; an exact flat index is the matrix itself, so one JSON document
//! carries both the ids and the vectors. Every write goes through a temp sibling plus a
//! rename — vectors, then scalar, then the manifest that vouches for both — and every read
//! verifies the manifest's digests, counts, and dimension before it trusts a single row.
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
use crate::experience::index::{
    ExperienceIndex, ExperienceIndexMeta, FLAT_VECTOR_ALGORITHM, MANIFEST_VERSION,
};
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

/// The three artifact paths of one index directory.
struct IndexPaths {
    /// `<dir>/meta.json`.
    meta: PathBuf,
    /// `<dir>/scalar/metadata.jsonl`.
    scalar: PathBuf,
    /// `<dir>/vector/embeddings.json`.
    vectors: PathBuf,
}

impl IndexPaths {
    /// Resolves the three paths under `dir`.
    fn new(dir: &Path) -> Self {
        Self {
            meta: dir.join(META_FILE),
            scalar: dir.join(SCALAR_DIR).join(SCALAR_FILE),
            vectors: dir.join(VECTOR_DIR).join(VECTORS_FILE),
        }
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

/// The encoded form of one catalog snapshot.
#[derive(Debug)]
pub(super) struct EncodedIndex {
    /// The `scalar/metadata.jsonl` bytes.
    pub(super) scalar: Vec<u8>,
    /// The `vector/embeddings.json` bytes.
    pub(super) vectors: Vec<u8>,
    /// The manifest vouching for both files.
    pub(super) meta: ExperienceIndexMeta,
}

/// Serializes a catalog into the three persisted artifacts and their manifest.
///
/// Items are written in id order, and every item must carry an embedding: the vector file
/// is a dense matrix, so a missing row would misalign every later one.
pub(super) fn encode(items: &[ExperienceItem], algorithm: &str) -> Result<EncodedIndex> {
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
        ids,
        rows,
    })
    .map_err(|error| Error::external(error).with_context("encoding vector/embeddings.json"))?;
    let meta = ExperienceIndexMeta {
        version: MANIFEST_VERSION,
        vector_count: items.len(),
        vector_algorithm: algorithm.to_string(),
        vector_dimension: dimension,
        embeddings_sha256: sha256_hex(&vectors),
        scalar_sha256: sha256_hex(scalar.as_bytes()),
    };
    Ok(EncodedIndex {
        scalar: scalar.into_bytes(),
        vectors,
        meta,
    })
}

/// Writes one encoded snapshot under `dir`, in dependency order.
///
/// The vector file lands first, then the scalar file, then the manifest: a crash therefore
/// leaves a directory whose manifest still describes the previous snapshot, which the next
/// load rejects instead of mixing two catalogs.
pub(super) async fn write(dir: &Path, encoded: &EncodedIndex) -> Result<()> {
    let paths = IndexPaths::new(dir);
    write_atomic(&paths.vectors, &encoded.vectors).await?;
    write_atomic(&paths.scalar, &encoded.scalar).await?;
    let meta = serde_json::to_vec_pretty(&encoded.meta)
        .map_err(|error| Error::external(error).with_context("encoding meta.json"))?;
    write_atomic(&paths.meta, &meta).await
}

/// Reads the index under `dir`; an absent or empty directory yields an empty index.
pub(super) async fn load(
    dir: &Path,
    embedding: Arc<dyn Embedding>,
    algorithm: String,
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
        ));
    }

    let meta_bytes = read_bytes(&paths.meta).await?;
    let meta: ExperienceIndexMeta =
        serde_json::from_slice(&meta_bytes).map_err(|error| corrupt(&paths.meta, error))?;
    validate_meta(&paths.meta, &meta)?;

    let scalar_bytes = read_bytes(&paths.scalar).await?;
    let vector_bytes = read_bytes(&paths.vectors).await?;
    verify_digest(&paths.scalar, &meta.scalar_sha256, &scalar_bytes)?;
    verify_digest(&paths.vectors, &meta.embeddings_sha256, &vector_bytes)?;

    let mut items = decode_scalar(&paths.scalar, &scalar_bytes)?;
    decode_vectors(&paths, &meta, &mut items, &vector_bytes)?;

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
        algorithm,
        meta,
    ))
}

/// Reports whether the directory holds any index file, rejecting a partial set.
///
/// An absent (or empty) directory is a fresh index; a directory holding some but not all of
/// the three artifacts is a broken index that must not be repaired silently.
async fn index_files_present(paths: &IndexPaths) -> Result<bool> {
    let mut present = Vec::with_capacity(3);
    for path in [&paths.meta, &paths.scalar, &paths.vectors] {
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

/// Validates the manifest version and the persisted vector algorithm.
fn validate_meta(path: &Path, meta: &ExperienceIndexMeta) -> Result<()> {
    if meta.version != MANIFEST_VERSION {
        return Err(corrupt(
            path,
            format!(
                "unsupported manifest version {} (expected {MANIFEST_VERSION})",
                meta.version
            ),
        ));
    }
    if !algorithm_is_flat(&meta.vector_algorithm) {
        return Err(corrupt(
            path,
            format!(
                "vector_algorithm {:?} is not supported for loading; only {FLAT_VECTOR_ALGORITHM:?} \
                 is implemented",
                meta.vector_algorithm
            ),
        ));
    }
    Ok(())
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

/// Decodes the vector file and rehydrates the embedding of every item.
fn decode_vectors(
    paths: &IndexPaths,
    meta: &ExperienceIndexMeta,
    items: &mut BTreeMap<String, ExperienceItem>,
    bytes: &[u8],
) -> Result<()> {
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
    Ok(())
}

/// Reports whether a persisted algorithm name selects the flat index.
fn algorithm_is_flat(raw: &str) -> bool {
    raw.trim().eq_ignore_ascii_case(FLAT_VECTOR_ALGORITHM)
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
        let encoded = encode(&items, FLAT_VECTOR_ALGORITHM).expect("encode");
        assert_eq!(encoded.meta.vector_count, 2);
        assert_eq!(encoded.meta.vector_dimension, 2);
        assert_eq!(encoded.meta.vector_algorithm, FLAT_VECTOR_ALGORITHM);
        assert_eq!(encoded.meta.embeddings_sha256, sha256_hex(&encoded.vectors));
        assert_eq!(encoded.meta.scalar_sha256, sha256_hex(&encoded.scalar));

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
    fn encoding_rejects_items_without_an_embedding_or_with_mixed_dimensions() {
        let missing = vec![ExperienceItem::new("orphan").with_id("exp_0001")];
        let error = encode(&missing, FLAT_VECTOR_ALGORITHM).expect_err("missing embedding");
        assert!(error.message().contains("embedding"));

        let mixed = vec![
            item("exp_0001", "first"),
            ExperienceItem::new("second")
                .with_id("exp_0002")
                .with_embedding(vec![1.0, 0.0, 0.0]),
        ];
        let error = encode(&mixed, FLAT_VECTOR_ALGORITHM).expect_err("mixed dimensions");
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
}
