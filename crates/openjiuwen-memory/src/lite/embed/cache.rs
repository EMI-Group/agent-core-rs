//! Content-hash keyed embedding-vector cache.
//!
//! Ports the embedding cache of Python `core/memory/lite/` (its `embedding_cache`
//! table) onto two interchangeable backends: a process-local map and a `SQLite` table
//! (feature `sqlite`).
//!
//! # Keys
//!
//! An entry is keyed by the pair `CacheKey` (`{ model, hash }`):
//!
//! - `model` is the embedding model's own `model_name`,
//!   so vectors produced by different models never mix (two models can legitimately
//!   produce equally sized vectors for the same text).
//! - `hash` is `content_hash`, a 64-bit `FNV-1a` digest of the exact chunk text as 16
//!   lowercase hex characters. Content addressing keeps an entry valid across process
//!   restarts, renames, and re-chunking of *surrounding* text; a changed chunk text
//!   changes the hash and therefore misses the cache.
//!
//! The Python reference truncates a SHA-256 digest to 16 hex characters. This crate
//! cannot depend on a hashing crate here, so it uses `FNV-1a` and keeps the same
//! 16-hex-character shape; a 64-bit digest makes an accidental collision negligible for
//! a per-workspace cache, and an embedding is still only a recall signal.
//!
//! # Backends
//!
//! - `EmbeddingCache::in_memory` holds a `parking_lot::Mutex<BTreeMap<..>>` (never
//!   held across an `.await`) and is process-local and non-persistent.
//! - `EmbeddingCache::sqlite` holds one cheaply cloned `SqlitePool`. Vectors are stored
//!   as little-endian `f32` bytes beside their dimension, so a read reconstructs the
//!   exact vector; a row whose dimension and blob length disagree is reported as
//!   `Error::Memory` rather than decoded into a wrong vector.
//!
//! A clone of an `EmbeddingCache` shares the same backend (both backends are handles),
//! so an index and its caller observe the same entries. Nothing is cached on behalf of
//! a query: only document vectors are stored (see
//! `EmbeddingIndex::reindex`).
//!
//! No failure is swallowed: every `sqlx` error is propagated as `Error::External`
//! carrying the `sqlx` failure as its source, and a corrupt row is a typed
//! `Error::Memory`.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use openjiuwen_common::error::Result;

#[cfg(feature = "sqlite")]
use openjiuwen_common::error::Error;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

#[cfg(feature = "sqlite")]
use sqlx::{Row, SqlitePool};

#[cfg(feature = "sqlite")]
use super::TARGET;

/// The `FNV-1a` 64-bit offset basis.
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;

/// The `FNV-1a` 64-bit prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// The number of bytes in one little-endian `f32` component.
#[cfg(feature = "sqlite")]
const F32_BYTES: usize = 4;

/// The cache table, applied idempotently by [`EmbeddingCache::sqlite`].
#[cfg(feature = "sqlite")]
const SCHEMA: &str = "\
CREATE TABLE IF NOT EXISTS lite_embedding_cache (
    model TEXT NOT NULL,
    hash TEXT NOT NULL,
    dims INTEGER NOT NULL,
    vector BLOB NOT NULL,
    PRIMARY KEY (model, hash)
)";

/// Reads one cached vector.
#[cfg(feature = "sqlite")]
const GET_SQL: &str =
    "SELECT dims, vector FROM lite_embedding_cache WHERE model = ?1 AND hash = ?2";

/// Inserts or replaces one cached vector.
#[cfg(feature = "sqlite")]
const PUT_SQL: &str = "\
INSERT INTO lite_embedding_cache (model, hash, dims, vector) VALUES (?1, ?2, ?3, ?4)
ON CONFLICT(model, hash) DO UPDATE SET dims = excluded.dims, vector = excluded.vector";

/// Counts the cached vectors.
#[cfg(feature = "sqlite")]
const LEN_SQL: &str = "SELECT COUNT(*) AS total FROM lite_embedding_cache";

/// Empties the cache.
#[cfg(feature = "sqlite")]
const CLEAR_SQL: &str = "DELETE FROM lite_embedding_cache";

/// Hashes chunk text into a stable 16-character hex content id.
///
/// The digest is `FNV-1a` over the UTF-8 bytes, formatted as 16 lowercase hex
/// characters. It is deterministic across processes, machines, and builds (no random
/// seeding, no dependency on the standard library's hasher), which is what makes a
/// persisted cache usable after a restart.
///
/// ```
/// use openjiuwen_memory::lite::embed::content_hash;
///
/// assert_eq!(content_hash("hello").len(), 16);
/// assert_eq!(content_hash("hello"), "a430d84680aabd0b");
/// assert_eq!(content_hash(""), "cbf29ce484222325");
/// ```
#[must_use]
pub fn content_hash(text: &str) -> String {
    let mut hash = FNV_OFFSET_BASIS;
    for byte in text.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    format!("{hash:016x}")
}

/// The identity of one cached embedding: the model name plus the content hash.
///
/// The type is `Ord` and `Hash`, so it can key a [`BTreeMap`] or a `HashMap` directly.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct CacheKey {
    /// The embedding model's reported name.
    pub model: String,
    /// The 16-hex-character [`content_hash`] of the embedded text.
    pub hash: String,
}

/// The storage behind an [`EmbeddingCache`].
#[derive(Debug)]
enum Backend {
    /// A process-local map guarded by a short-lived `parking_lot` lock.
    Memory(Mutex<BTreeMap<CacheKey, Vec<f32>>>),
    /// A table in a `SQLite` database.
    #[cfg(feature = "sqlite")]
    Sqlite(SqlitePool),
}

/// What an [`EmbeddingCache`] actually holds.
#[derive(Debug)]
struct Inner {
    /// The selected backend.
    backend: Backend,
}

/// A content-hash keyed cache of embedding vectors.
///
/// Clone it freely: every clone shares the same backend.
#[derive(Clone)]
pub struct EmbeddingCache {
    /// The shared backend.
    inner: Arc<Inner>,
}

impl fmt::Debug for EmbeddingCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut debug = f.debug_struct("EmbeddingCache");
        debug.field("persistent", &self.is_persistent());
        match &self.inner.backend {
            Backend::Memory(entries) => {
                debug.field("entries", &entries.lock().len());
            }
            #[cfg(feature = "sqlite")]
            Backend::Sqlite(_) => {}
        }
        debug.finish_non_exhaustive()
    }
}

impl EmbeddingCache {
    /// Creates a process-local, non-persistent cache.
    #[must_use]
    pub fn in_memory() -> Self {
        Self {
            inner: Arc::new(Inner {
                backend: Backend::Memory(Mutex::new(BTreeMap::new())),
            }),
        }
    }

    /// Creates a cache in the `SQLite` database behind `pool`, applying the schema.
    ///
    /// The table `lite_embedding_cache` is created when absent, so calling this again
    /// over the same database is idempotent and keeps the stored vectors.
    ///
    /// # Errors
    ///
    /// Returns `Error::External` carrying the `sqlx` failure when the schema cannot be
    /// applied (for example a read-only database).
    #[cfg(feature = "sqlite")]
    pub async fn sqlite(pool: SqlitePool) -> Result<Self> {
        sqlx::query(SCHEMA)
            .execute(&pool)
            .await
            .map_err(db_error("create the cache table"))?;
        tracing::debug!(
            target: TARGET,
            "opened the sqlite embedding cache"
        );
        Ok(Self {
            inner: Arc::new(Inner {
                backend: Backend::Sqlite(pool),
            }),
        })
    }

    /// Whether the cache survives the process (i.e. is backed by `SQLite`).
    #[must_use]
    pub fn is_persistent(&self) -> bool {
        match &self.inner.backend {
            Backend::Memory(_) => false,
            #[cfg(feature = "sqlite")]
            Backend::Sqlite(_) => true,
        }
    }

    /// Looks up the vector stored for `(model, hash)`.
    ///
    /// # Errors
    ///
    /// Returns `Error::External` carrying the `sqlx` failure when the lookup fails, and
    /// `Error::Memory` when the stored row is corrupt (its blob length does not match
    /// its recorded dimension).
    // The API is async for the `SQLite` backend; the in-memory arm completes without
    // awaiting anything, which only happens when the feature is off.
    #[cfg_attr(
        not(feature = "sqlite"),
        allow(clippy::unused_async, clippy::unused_async_trait_impl)
    )]
    pub async fn get(&self, model: &str, hash: &str) -> Result<Option<Vec<f32>>> {
        match &self.inner.backend {
            Backend::Memory(entries) => {
                let key = CacheKey {
                    model: model.to_owned(),
                    hash: hash.to_owned(),
                };
                Ok(entries.lock().get(&key).cloned())
            }
            #[cfg(feature = "sqlite")]
            Backend::Sqlite(pool) => sqlite_get(pool, model, hash).await,
        }
    }

    /// Stores `vector` under `(model, hash)`, replacing an existing entry.
    ///
    /// # Errors
    ///
    /// Returns `Error::Memory` when the vector length cannot be stored as a `SQLite`
    /// integer, and `Error::External` carrying the `sqlx` failure when the write fails.
    // The API is async for the `SQLite` backend; the in-memory arm completes without
    // awaiting anything, which only happens when the feature is off.
    #[cfg_attr(
        not(feature = "sqlite"),
        allow(clippy::unused_async, clippy::unused_async_trait_impl)
    )]
    pub async fn put(&self, model: &str, hash: &str, vector: &[f32]) -> Result<()> {
        match &self.inner.backend {
            Backend::Memory(entries) => {
                let key = CacheKey {
                    model: model.to_owned(),
                    hash: hash.to_owned(),
                };
                entries.lock().insert(key, vector.to_vec());
                Ok(())
            }
            #[cfg(feature = "sqlite")]
            Backend::Sqlite(pool) => sqlite_put(pool, model, hash, vector).await,
        }
    }

    /// The number of cached vectors.
    ///
    /// # Errors
    ///
    /// Returns `Error::External` carrying the `sqlx` failure when the count fails, and
    /// `Error::Memory` when the count cannot be represented as a `usize`.
    // The API is async for the `SQLite` backend; the in-memory arm completes without
    // awaiting anything, which only happens when the feature is off.
    #[cfg_attr(
        not(feature = "sqlite"),
        allow(clippy::unused_async, clippy::unused_async_trait_impl)
    )]
    pub async fn len(&self) -> Result<usize> {
        match &self.inner.backend {
            Backend::Memory(entries) => Ok(entries.lock().len()),
            #[cfg(feature = "sqlite")]
            Backend::Sqlite(pool) => {
                let row = sqlx::query(LEN_SQL)
                    .fetch_one(pool)
                    .await
                    .map_err(db_error("count the cache entries"))?;
                let total: i64 = row
                    .try_get("total")
                    .map_err(db_error("read the cache entry count"))?;
                usize::try_from(total).map_err(|_| {
                    Error::memory(format!(
                        "lite embedding cache: stored entry count {total} is not a valid length"
                    ))
                })
            }
        }
    }

    /// Whether the cache holds no vector.
    ///
    /// # Errors
    ///
    /// Returns the [`EmbeddingCache::len`] errors.
    pub async fn is_empty(&self) -> Result<bool> {
        Ok(self.len().await? == 0)
    }

    /// Removes every cached vector.
    ///
    /// # Errors
    ///
    /// Returns `Error::External` carrying the `sqlx` failure when the delete fails.
    // The API is async for the `SQLite` backend; the in-memory arm completes without
    // awaiting anything, which only happens when the feature is off.
    #[cfg_attr(
        not(feature = "sqlite"),
        allow(clippy::unused_async, clippy::unused_async_trait_impl)
    )]
    pub async fn clear(&self) -> Result<()> {
        match &self.inner.backend {
            Backend::Memory(entries) => {
                entries.lock().clear();
                Ok(())
            }
            #[cfg(feature = "sqlite")]
            Backend::Sqlite(pool) => {
                sqlx::query(CLEAR_SQL)
                    .execute(pool)
                    .await
                    .map_err(db_error("clear the cache"))?;
                Ok(())
            }
        }
    }
}

/// Reads one row and decodes its vector.
#[cfg(feature = "sqlite")]
async fn sqlite_get(pool: &SqlitePool, model: &str, hash: &str) -> Result<Option<Vec<f32>>> {
    let row = sqlx::query(GET_SQL)
        .bind(model)
        .bind(hash)
        .fetch_optional(pool)
        .await
        .map_err(db_error("read a cached vector"))?;
    let Some(row) = row else {
        return Ok(None);
    };
    let dimensions: i64 = row
        .try_get("dims")
        .map_err(db_error("read a cached dimension"))?;
    let bytes: Vec<u8> = row
        .try_get("vector")
        .map_err(db_error("read a cached vector blob"))?;
    decode_vector(model, dimensions, &bytes).map(Some)
}

/// Encodes and upserts one vector row.
#[cfg(feature = "sqlite")]
async fn sqlite_put(pool: &SqlitePool, model: &str, hash: &str, vector: &[f32]) -> Result<()> {
    let dimensions = i64::try_from(vector.len()).map_err(|_| {
        Error::memory(format!(
            "lite embedding cache: vector length {} cannot be stored as a sqlite integer",
            vector.len()
        ))
    })?;
    sqlx::query(PUT_SQL)
        .bind(model)
        .bind(hash)
        .bind(dimensions)
        .bind(encode_vector(vector))
        .execute(pool)
        .await
        .map_err(db_error("store a vector"))?;
    Ok(())
}

/// Packs a vector into little-endian `f32` bytes.
#[cfg(feature = "sqlite")]
fn encode_vector(vector: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(vector.len().saturating_mul(F32_BYTES));
    for component in vector {
        bytes.extend_from_slice(&component.to_le_bytes());
    }
    bytes
}

/// Unpacks little-endian `f32` bytes, rejecting a length that disagrees with `dimensions`.
#[cfg(feature = "sqlite")]
fn decode_vector(model: &str, dimensions: i64, bytes: &[u8]) -> Result<Vec<f32>> {
    let dimensions = usize::try_from(dimensions).map_err(|_| {
        Error::memory(format!(
            "lite embedding cache: stored dimension {dimensions} for model `{model}` is not a length"
        ))
    })?;
    if bytes.len() != dimensions.saturating_mul(F32_BYTES) {
        return Err(Error::memory(format!(
            "lite embedding cache: corrupt vector for model `{model}` ({} bytes stored for {dimensions} dimensions)",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(F32_BYTES)
        .map(|component| {
            f32::from_le_bytes([component[0], component[1], component[2], component[3]])
        })
        .collect())
}

/// Maps a `sqlx` failure onto an `Error::External` naming the operation.
#[cfg(feature = "sqlite")]
fn db_error(operation: &'static str) -> impl FnOnce(sqlx::Error) -> Error {
    move |error| Error::external(error).with_context(format!("lite embedding cache: {operation}"))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn content_hash_is_sixteen_lowercase_hex_characters() {
        for text in ["", "a", "hello", "the quick brown fox", "你好世界"] {
            let hash = content_hash(text);
            assert_eq!(hash.len(), 16, "hash of {text:?} was {hash}");
            assert!(
                hash.chars()
                    .all(|character| character.is_ascii_digit() || ('a'..='f').contains(&character)),
                "hash of {text:?} was not lowercase hex: {hash}"
            );
        }
    }

    #[test]
    fn content_hash_matches_known_fnv1a_vectors() {
        assert_eq!(content_hash(""), "cbf29ce484222325");
        assert_eq!(content_hash("a"), "af63dc4c8601ec8c");
        assert_eq!(content_hash("hello"), "a430d84680aabd0b");
    }

    #[test]
    fn content_hash_is_stable_and_distinguishes_texts() {
        assert_eq!(content_hash("same text"), content_hash("same text"));
        assert_ne!(content_hash("a"), content_hash("b"));
        assert_ne!(content_hash("ab"), content_hash("ba"));
        assert_ne!(content_hash("line\n"), content_hash("line"));
        assert_ne!(content_hash("MEMORY.md"), content_hash("memory.md"));
    }

    #[tokio::test]
    async fn memory_cache_reports_hits_and_misses() {
        let cache = EmbeddingCache::in_memory();
        assert!(!cache.is_persistent());
        assert_eq!(cache.len().await.unwrap(), 0);
        assert_eq!(cache.len().await.unwrap(), 0);
        assert_eq!(cache.get("model", "deadbeefdeadbeef").await.unwrap(), None);
        cache
            .put("model", "deadbeefdeadbeef", &[0.25, -1.5, 3.0])
            .await
            .unwrap();
        assert_eq!(cache.len().await.unwrap(), 1);
        assert!(!cache.is_empty().await.unwrap());
        assert_eq!(
            cache.get("model", "deadbeefdeadbeef").await.unwrap(),
            Some(vec![0.25, -1.5, 3.0])
        );
    }

    #[tokio::test]
    async fn entries_are_keyed_by_both_model_and_hash() {
        let cache = EmbeddingCache::in_memory();
        cache
            .put("model-a", "aaaabbbbccccdddd", &[1.0])
            .await
            .unwrap();
        assert!(cache
            .get("model-b", "aaaabbbbccccdddd")
            .await
            .unwrap()
            .is_none());
        assert!(cache
            .get("model-a", "ddddccccbbbbaaaa")
            .await
            .unwrap()
            .is_none());
        assert_eq!(
            cache.get("model-a", "aaaabbbbccccdddd").await.unwrap(),
            Some(vec![1.0])
        );
    }

    #[tokio::test]
    async fn put_replaces_an_existing_entry() {
        let cache = EmbeddingCache::in_memory();
        cache.put("model", "hash", &[1.0, 2.0]).await.unwrap();
        cache.put("model", "hash", &[3.0]).await.unwrap();
        assert_eq!(cache.len().await.unwrap(), 1);
        assert_eq!(cache.get("model", "hash").await.unwrap(), Some(vec![3.0]));
    }

    #[tokio::test]
    async fn clear_empties_the_cache() {
        let cache = EmbeddingCache::in_memory();
        cache.put("model", "one", &[1.0]).await.unwrap();
        cache.put("model", "two", &[2.0]).await.unwrap();
        cache.clear().await.unwrap();
        assert_eq!(cache.len().await.unwrap(), 0);
        assert!(cache.get("model", "one").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn clones_share_the_backend() {
        let cache = EmbeddingCache::in_memory();
        let clone = cache.clone();
        clone.put("model", "hash", &[7.0]).await.unwrap();
        assert_eq!(cache.get("model", "hash").await.unwrap(), Some(vec![7.0]));
        assert_eq!(cache.len().await.unwrap(), 1);
    }

    #[test]
    fn cache_key_orders_and_hashes() {
        let mut keys = std::collections::BTreeMap::new();
        keys.insert(
            CacheKey {
                model: "b".to_owned(),
                hash: "2".to_owned(),
            },
            2,
        );
        keys.insert(
            CacheKey {
                model: "a".to_owned(),
                hash: "1".to_owned(),
            },
            1,
        );
        let ordered: Vec<i32> = keys.into_values().collect();
        assert_eq!(ordered, vec![1, 2]);
        let mut hashed = std::collections::HashSet::new();
        assert!(hashed.insert(CacheKey {
            model: "a".to_owned(),
            hash: "1".to_owned(),
        }));
        assert!(!hashed.insert(CacheKey {
            model: "a".to_owned(),
            hash: "1".to_owned(),
        }));
        assert!(hashed.insert(CacheKey {
            model: "a".to_owned(),
            hash: "2".to_owned(),
        }));
    }
}

#[cfg(all(test, feature = "sqlite"))]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]
mod sqlite_tests {
    use openjiuwen_common::error::ErrorKind;
    use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};

    use super::*;

    /// Opens a pool over `<dir>/cache.db`, creating the file when absent.
    async fn pool(dir: &tempfile::TempDir) -> SqlitePool {
        let options = SqliteConnectOptions::new()
            .filename(dir.path().join("cache.db"))
            .create_if_missing(true);
        SqlitePoolOptions::new()
            .max_connections(2)
            .connect_with(options)
            .await
            .expect("sqlite pool")
    }

    #[tokio::test]
    async fn sqlite_cache_round_trips_vectors_exactly() {
        let dir = tempfile::tempdir().unwrap();
        let cache = EmbeddingCache::sqlite(pool(&dir).await).await.unwrap();
        assert!(cache.is_persistent());
        cache
            .put(
                "hash-embedding",
                "0123456789abcdef",
                &[0.0, -0.5, f32::MAX, f32::MIN],
            )
            .await
            .unwrap();
        let vector = cache
            .get("hash-embedding", "0123456789abcdef")
            .await
            .unwrap()
            .expect("a stored vector");
        assert_eq!(vector, vec![0.0, -0.5, f32::MAX, f32::MIN]);
        assert_eq!(cache.len().await.unwrap(), 1);
        assert!(!cache.is_empty().await.unwrap());
    }

    #[tokio::test]
    async fn sqlite_cache_persists_across_handles() {
        let dir = tempfile::tempdir().unwrap();
        let first = EmbeddingCache::sqlite(pool(&dir).await).await.unwrap();
        first
            .put("model", "aaaabbbbccccdddd", &[1.5])
            .await
            .unwrap();
        drop(first);
        let second = EmbeddingCache::sqlite(pool(&dir).await).await.unwrap();
        assert_eq!(
            second.get("model", "aaaabbbbccccdddd").await.unwrap(),
            Some(vec![1.5])
        );
    }

    #[tokio::test]
    async fn sqlite_cache_replaces_and_clears() {
        let dir = tempfile::tempdir().unwrap();
        let cache = EmbeddingCache::sqlite(pool(&dir).await).await.unwrap();
        cache.put("model", "hash", &[1.0, 2.0]).await.unwrap();
        cache.put("model", "hash", &[3.0]).await.unwrap();
        assert_eq!(cache.len().await.unwrap(), 1);
        assert_eq!(cache.get("model", "hash").await.unwrap(), Some(vec![3.0]));
        cache.clear().await.unwrap();
        assert_eq!(cache.len().await.unwrap(), 0);
        assert!(cache.get("model", "hash").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn sqlite_cache_adopts_an_existing_table() {
        let dir = tempfile::tempdir().unwrap();
        let pool = pool(&dir).await;
        let first = EmbeddingCache::sqlite(pool.clone()).await.unwrap();
        first.put("model", "hash", &[9.0]).await.unwrap();
        let second = EmbeddingCache::sqlite(pool).await.unwrap();
        assert_eq!(second.len().await.unwrap(), 1);
        assert_eq!(second.get("model", "hash").await.unwrap(), Some(vec![9.0]));
    }

    #[tokio::test]
    async fn sqlite_cache_rejects_a_corrupt_blob() {
        let dir = tempfile::tempdir().unwrap();
        let pool = pool(&dir).await;
        let cache = EmbeddingCache::sqlite(pool.clone()).await.unwrap();
        // Three bytes for a stated dimension of two is a truncated vector.
        sqlx::query(PUT_SQL)
            .bind("model")
            .bind("hash")
            .bind(2_i64)
            .bind(vec![1_u8, 2, 3])
            .execute(&pool)
            .await
            .unwrap();
        let error = cache.get("model", "hash").await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Memory);
        assert!(error.to_string().contains("corrupt vector"));
    }

    #[tokio::test]
    async fn sqlite_cache_rejects_a_negative_dimension() {
        let dir = tempfile::tempdir().unwrap();
        let pool = pool(&dir).await;
        let cache = EmbeddingCache::sqlite(pool.clone()).await.unwrap();
        sqlx::query(PUT_SQL)
            .bind("model")
            .bind("hash")
            .bind(-1_i64)
            .bind(Vec::<u8>::new())
            .execute(&pool)
            .await
            .unwrap();
        assert_eq!(
            cache.get("model", "hash").await.unwrap_err().kind(),
            ErrorKind::Memory
        );
    }

    #[tokio::test]
    async fn a_failing_database_surfaces_the_error() {
        let dir = tempfile::tempdir().unwrap();
        let pool = pool(&dir).await;
        let cache = EmbeddingCache::sqlite(pool.clone()).await.unwrap();
        pool.close().await;
        assert_eq!(cache.len().await.unwrap_err().kind(), ErrorKind::External);
    }
}
