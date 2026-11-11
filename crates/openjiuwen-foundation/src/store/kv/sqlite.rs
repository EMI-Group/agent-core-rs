//! SQLite-backed key-value store (feature `sqlite`).
//!
//! One `kv_store` table holds every entry: `k` is the primary key, `value` holds the
//! raw text or the base64 payload of a byte value, `kind` distinguishes the two, and
//! `expiry_ms` is an absolute epoch-millisecond deadline (`NULL` never expires).
//!
//! Prefix scans use `substr(k, 1, length(?)) = ?` instead of `LIKE`, so no wildcard
//! escaping is ever needed; TTL checks are evaluated in SQL against the current
//! epoch time, and expired rows are simply invisible until an upsert replaces them.

use std::collections::BTreeMap;
use std::str::FromStr;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use base64::Engine as _;
use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions, SqliteRow};
use sqlx::{Row, SqlitePool};

use openjiuwen_common::error::{Error, Result};

use super::{KvPipeline, KvStore, KvValue, StorePipeline};

/// Table and index DDL, applied idempotently by every constructor.
const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS kv_store (
    k TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    kind TEXT NOT NULL,
    expiry_ms INTEGER
)";

/// Index on the TTL column; the reads filter on it.
const EXPIRY_INDEX: &str = "CREATE INDEX IF NOT EXISTS kv_store_expiry_idx ON kv_store (expiry_ms)";

/// `kind` tag for a text value.
const KIND_TEXT: &str = "text";
/// `kind` tag for a byte value.
const KIND_BYTES: &str = "bytes";

/// SQL predicate selecting only live rows, with the "now" placeholder explicit.
///
/// The placeholder index is spelled out (`?N`): `SQLite` numbers a bare `?`
/// relative to the numbered placeholders around it, so mixing `?1` with a bare
/// `?` makes the positional binding of "now" land on the key parameter.
fn live_at(index: usize) -> String {
    format!("(expiry_ms IS NULL OR expiry_ms > ?{index})")
}

/// A key-value store backed by a `SQLite` database.
///
/// Clones share the same connection pool.
#[derive(Debug, Clone)]
pub struct SqliteKvStore {
    /// The connection pool every operation borrows from.
    pool: SqlitePool,
}

impl SqliteKvStore {
    /// Connects to `url` (for example `sqlite::memory:` or `sqlite:kv.db`).
    ///
    /// In-memory URLs are pinned to a single pooled connection, otherwise every
    /// checkout would open a fresh, empty database.
    pub async fn connect(url: &str) -> Result<Self> {
        let options = SqliteConnectOptions::from_str(url)
            .map_err(Error::external)?
            .create_if_missing(true);
        let pool_options = SqlitePoolOptions::new();
        let pool_options = if url.contains("memory") {
            pool_options.max_connections(1)
        } else {
            pool_options
        };
        let pool = pool_options
            .connect_with(options)
            .await
            .map_err(Error::external)?;
        Self::from_pool(pool).await
    }

    /// Wraps an existing pool (created by a caller or another subsystem) and makes
    /// sure the schema exists.
    pub async fn from_pool(pool: SqlitePool) -> Result<Self> {
        sqlx::query(SCHEMA)
            .execute(&pool)
            .await
            .map_err(Error::external)?;
        sqlx::query(EXPIRY_INDEX)
            .execute(&pool)
            .await
            .map_err(Error::external)?;
        Ok(Self { pool })
    }

    /// The underlying pool, for callers that need to run companion queries.
    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }

    /// Number of live rows.
    pub async fn len(&self) -> Result<usize> {
        let row = sqlx::query(&format!(
            "SELECT COUNT(*) AS n FROM kv_store WHERE {}",
            live_at(1)
        ))
        .bind(now_epoch_ms())
        .fetch_one(&self.pool)
        .await
        .map_err(Error::external)?;
        let count: i64 = row.try_get("n").map_err(Error::external)?;
        Ok(usize::try_from(count).unwrap_or(0))
    }

    /// Whether the store holds no live row.
    pub async fn is_empty(&self) -> Result<bool> {
        Ok(self.len().await? == 0)
    }
}

#[async_trait]
impl KvStore for SqliteKvStore {
    async fn set(&self, key: &str, value: KvValue) -> Result<()> {
        let (kind, data) = encode_value(&value);
        sqlx::query(
            "INSERT INTO kv_store (k, value, kind, expiry_ms) VALUES (?1, ?2, ?3, NULL)
             ON CONFLICT(k) DO UPDATE SET value = excluded.value, kind = excluded.kind, expiry_ms = NULL",
        )
        .bind(key)
        .bind(data)
        .bind(kind)
        .execute(&self.pool)
        .await
        .map_err(Error::external)?;
        Ok(())
    }

    async fn get(&self, key: &str) -> Result<Option<KvValue>> {
        let row = sqlx::query(&format!(
            "SELECT value, kind FROM kv_store WHERE k = ?1 AND {}",
            live_at(2)
        ))
        .bind(key)
        .bind(now_epoch_ms())
        .fetch_optional(&self.pool)
        .await
        .map_err(Error::external)?;
        row.map(|row| decode_row(&row)).transpose()
    }

    async fn exists(&self, key: &str) -> Result<bool> {
        let row = sqlx::query(&format!(
            "SELECT 1 AS found FROM kv_store WHERE k = ?1 AND {}",
            live_at(2)
        ))
        .bind(key)
        .bind(now_epoch_ms())
        .fetch_optional(&self.pool)
        .await
        .map_err(Error::external)?;
        Ok(row.is_some())
    }

    async fn delete(&self, key: &str) -> Result<()> {
        sqlx::query("DELETE FROM kv_store WHERE k = ?1")
            .bind(key)
            .execute(&self.pool)
            .await
            .map_err(Error::external)?;
        Ok(())
    }

    async fn exclusive_set(
        &self,
        key: &str,
        value: KvValue,
        expiry: Option<Duration>,
    ) -> Result<bool> {
        let now = now_epoch_ms();
        let expiry_ms = expiry.map(|ttl| now.saturating_add(duration_ms(ttl)));
        let (kind, data) = encode_value(&value);
        // The upsert's `WHERE` clause only fires for an already-expired row, which
        // makes "write unless a live row exists" one atomic statement.
        let result = sqlx::query(
            "INSERT INTO kv_store (k, value, kind, expiry_ms) VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(k) DO UPDATE SET value = excluded.value, kind = excluded.kind,
                 expiry_ms = excluded.expiry_ms
             WHERE kv_store.expiry_ms IS NOT NULL AND kv_store.expiry_ms <= ?5",
        )
        .bind(key)
        .bind(data)
        .bind(kind)
        .bind(expiry_ms)
        .bind(now)
        .execute(&self.pool)
        .await
        .map_err(Error::external)?;
        Ok(result.rows_affected() > 0)
    }

    async fn get_by_prefix(&self, prefix: &str) -> Result<BTreeMap<String, KvValue>> {
        let rows = sqlx::query(&format!(
            "SELECT k, value, kind FROM kv_store
             WHERE substr(k, 1, length(?1)) = ?1 AND {}
             ORDER BY k",
            live_at(2)
        ))
        .bind(prefix)
        .bind(now_epoch_ms())
        .fetch_all(&self.pool)
        .await
        .map_err(Error::external)?;
        let mut entries = BTreeMap::new();
        for row in rows {
            let key: String = row.try_get("k").map_err(Error::external)?;
            entries.insert(key, decode_row(&row)?);
        }
        Ok(entries)
    }

    async fn delete_by_prefix(&self, prefix: &str) -> Result<usize> {
        let result = sqlx::query(&format!(
            "DELETE FROM kv_store WHERE substr(k, 1, length(?1)) = ?1 AND {}",
            live_at(2)
        ))
        .bind(prefix)
        .bind(now_epoch_ms())
        .execute(&self.pool)
        .await
        .map_err(Error::external)?;
        Ok(usize::try_from(result.rows_affected()).unwrap_or(usize::MAX))
    }

    async fn mget(&self, keys: &[String]) -> Result<Vec<Option<KvValue>>> {
        if keys.is_empty() {
            return Ok(Vec::new());
        }
        // One `IN` query, reordered in Rust so the input order is preserved.
        let placeholders = placeholders(1, keys.len());
        let now_index = keys.len() + 1;
        let sql = format!(
            "SELECT k, value, kind FROM kv_store WHERE k IN ({placeholders}) AND (expiry_ms IS NULL OR expiry_ms > ?{now_index})"
        );
        let mut query = sqlx::query(&sql);
        for key in keys {
            query = query.bind(key);
        }
        let rows = query
            .bind(now_epoch_ms())
            .fetch_all(&self.pool)
            .await
            .map_err(Error::external)?;
        let mut found = BTreeMap::new();
        for row in rows {
            let key: String = row.try_get("k").map_err(Error::external)?;
            found.insert(key, decode_row(&row)?);
        }
        Ok(keys.iter().map(|key| found.get(key).cloned()).collect())
    }

    async fn batch_delete(&self, keys: &[String]) -> Result<usize> {
        if keys.is_empty() {
            return Ok(0);
        }
        let placeholders = placeholders(1, keys.len());
        let now_index = keys.len() + 1;
        let sql = format!(
            "DELETE FROM kv_store WHERE k IN ({placeholders}) AND (expiry_ms IS NULL OR expiry_ms > ?{now_index})"
        );
        let mut query = sqlx::query(&sql);
        for key in keys {
            query = query.bind(key);
        }
        let result = query
            .bind(now_epoch_ms())
            .execute(&self.pool)
            .await
            .map_err(Error::external)?;
        Ok(usize::try_from(result.rows_affected()).unwrap_or(usize::MAX))
    }

    fn pipeline(&self) -> Box<dyn KvPipeline + '_> {
        Box::new(StorePipeline::new(self))
    }
}

/// Builds `?start, ?start+1, ...` for `count` parameters.
fn placeholders(start: usize, count: usize) -> String {
    (start..start + count)
        .map(|index| format!("?{index}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Splits a value into its `kind` tag and stored payload.
fn encode_value(value: &KvValue) -> (&'static str, String) {
    match value {
        KvValue::Text(text) => (KIND_TEXT, text.clone()),
        KvValue::Bytes(bytes) => (
            KIND_BYTES,
            base64::engine::general_purpose::STANDARD.encode(bytes),
        ),
    }
}

/// Decodes a `(value, kind)` row into a [`KvValue`].
fn decode_row(row: &SqliteRow) -> Result<KvValue> {
    let kind: String = row.try_get("kind").map_err(Error::external)?;
    let data: String = row.try_get("value").map_err(Error::external)?;
    match kind.as_str() {
        KIND_TEXT => Ok(KvValue::Text(data)),
        KIND_BYTES => {
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(data.as_bytes())
                .map_err(Error::external)?;
            Ok(KvValue::Bytes(bytes))
        }
        other => Err(Error::validation(format!(
            "unknown kv kind `{other}` in kv_store"
        ))),
    }
}

/// Milliseconds since the Unix epoch, the single clock every TTL uses.
fn now_epoch_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| {
            i64::try_from(elapsed.as_millis()).unwrap_or(i64::MAX)
        })
}

/// A duration in whole milliseconds, saturating on absurd values.
fn duration_ms(duration: Duration) -> i64 {
    i64::try_from(duration.as_millis()).unwrap_or(i64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn store() -> SqliteKvStore {
        SqliteKvStore::connect("sqlite::memory:").await.unwrap()
    }

    #[tokio::test]
    async fn text_and_bytes_round_trip() {
        let store = store().await;
        store.set("text", KvValue::text("hello")).await.unwrap();
        store
            .set("bytes", KvValue::bytes(vec![0_u8, 255]))
            .await
            .unwrap();

        assert_eq!(
            store.get("text").await.unwrap(),
            Some(KvValue::text("hello"))
        );
        assert_eq!(
            store.get("bytes").await.unwrap(),
            Some(KvValue::bytes(vec![0_u8, 255]))
        );
        assert_eq!(store.get("missing").await.unwrap(), None);
        assert!(store.exists("text").await.unwrap());
        assert!(!store.exists("missing").await.unwrap());
        assert_eq!(store.len().await.unwrap(), 2);
    }

    #[tokio::test]
    async fn set_overwrites_and_delete_is_idempotent() {
        let store = store().await;
        store.set("k", KvValue::text("1")).await.unwrap();
        store.set("k", KvValue::text("2")).await.unwrap();
        assert_eq!(store.get("k").await.unwrap(), Some(KvValue::text("2")));

        store.delete("k").await.unwrap();
        assert_eq!(store.get("k").await.unwrap(), None);
        store.delete("k").await.unwrap();
        assert!(store.is_empty().await.unwrap());
    }

    #[tokio::test]
    async fn exclusive_set_writes_only_when_absent_or_expired() {
        let store = store().await;
        assert!(store
            .exclusive_set("k", KvValue::text("first"), None)
            .await
            .unwrap());
        assert!(!store
            .exclusive_set("k", KvValue::text("second"), None)
            .await
            .unwrap());
        assert_eq!(store.get("k").await.unwrap(), Some(KvValue::text("first")));

        assert!(store
            .exclusive_set("ttl", KvValue::text("a"), Some(Duration::from_millis(20)))
            .await
            .unwrap());
        assert!(!store
            .exclusive_set("ttl", KvValue::text("b"), None)
            .await
            .unwrap());
        tokio::time::sleep(Duration::from_millis(80)).await;
        assert_eq!(store.get("ttl").await.unwrap(), None);
        assert!(!store.exists("ttl").await.unwrap());
        assert!(store
            .exclusive_set("ttl", KvValue::text("c"), None)
            .await
            .unwrap());
        assert_eq!(store.get("ttl").await.unwrap(), Some(KvValue::text("c")));
    }

    #[tokio::test]
    async fn get_by_prefix_supports_empty_prefix_and_is_sorted() {
        let store = store().await;
        for (key, value) in [("b", "2"), ("a", "1"), ("c/1", "3")] {
            store.set(key, KvValue::text(value)).await.unwrap();
        }

        let all = store.get_by_prefix("").await.unwrap();
        assert_eq!(
            all.keys().cloned().collect::<Vec<_>>(),
            vec!["a".to_string(), "b".to_string(), "c/1".to_string()]
        );
        let prefixed = store.get_by_prefix("c/").await.unwrap();
        assert_eq!(prefixed.len(), 1);
        assert_eq!(prefixed.get("c/1"), Some(&KvValue::text("3")));
        assert!(store.get_by_prefix("zzz").await.unwrap().is_empty());

        // Wildcards are literal data, not patterns.
        store.set("a%b", KvValue::text("percent")).await.unwrap();
        store.set("axb", KvValue::text("other")).await.unwrap();
        let wildcard = store.get_by_prefix("a%").await.unwrap();
        assert_eq!(wildcard.len(), 1);
        assert!(wildcard.contains_key("a%b"));
    }

    #[tokio::test]
    async fn delete_by_prefix_returns_count() {
        let store = store().await;
        for key in ["p/1", "p/2", "q/1"] {
            store.set(key, KvValue::text(key)).await.unwrap();
        }
        assert_eq!(store.delete_by_prefix("p/").await.unwrap(), 2);
        assert_eq!(store.delete_by_prefix("p/").await.unwrap(), 0);
        assert_eq!(store.get_by_prefix("q/").await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn mget_preserves_order_and_misses() {
        let store = store().await;
        store.set("a", KvValue::text("1")).await.unwrap();
        store.set("c", KvValue::text("3")).await.unwrap();

        let keys = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        assert_eq!(
            store.mget(&keys).await.unwrap(),
            vec![Some(KvValue::text("1")), None, Some(KvValue::text("3")),]
        );
        assert_eq!(store.mget(&[]).await.unwrap().len(), 0);
    }

    #[tokio::test]
    async fn batch_delete_counts_present_keys_only() {
        let store = store().await;
        store.set("a", KvValue::text("1")).await.unwrap();
        store.set("b", KvValue::text("2")).await.unwrap();

        let keys = vec![
            "a".to_string(),
            "b".to_string(),
            "a".to_string(),
            "missing".to_string(),
        ];
        assert_eq!(store.batch_delete(&keys).await.unwrap(), 2);
        assert_eq!(store.batch_delete(&[]).await.unwrap(), 0);
        assert!(store.is_empty().await.unwrap());
    }

    #[tokio::test]
    async fn pipeline_applies_operations_in_order() {
        let store = store().await;
        store.set("existing", KvValue::text("old")).await.unwrap();

        let mut pipeline = store.pipeline();
        pipeline.set("queued", KvValue::text("new"));
        pipeline.get("queued");
        pipeline.exists("queued");
        pipeline.exists("missing");
        pipeline.get("existing");

        assert_eq!(
            pipeline.execute().await.unwrap(),
            vec![
                None,
                Some(KvValue::text("new")),
                Some(KvValue::text("true")),
                Some(KvValue::text("false")),
                Some(KvValue::text("old")),
            ]
        );
    }

    #[tokio::test]
    async fn from_pool_shares_state_with_the_caller() {
        let store = store().await;
        let clone = SqliteKvStore::from_pool(store.pool().clone())
            .await
            .unwrap();
        store.set("shared", KvValue::text("v")).await.unwrap();
        assert_eq!(clone.get("shared").await.unwrap(), Some(KvValue::text("v")));
        // Schema creation is idempotent.
        let _again = SqliteKvStore::from_pool(store.pool().clone())
            .await
            .unwrap();
    }
}
