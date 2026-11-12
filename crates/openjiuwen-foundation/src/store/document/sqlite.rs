//! SQLite-backed document store (feature `sqlite`).
//!
//! Two fixed tables hold everything:
//!
//! - `document_collections(name TEXT PRIMARY KEY)` — the collection registry, so
//!   an empty collection is still visible after a restart.
//! - `documents(collection TEXT NOT NULL, id TEXT NOT NULL, doc_json TEXT NOT NULL,
//!   PRIMARY KEY(collection, id))` — one row per document, with the whole
//!   [`Document`] stored as its serde JSON.
//!
//! Collections share one `documents` table instead of getting a table each: the
//! schema then never changes, so no DDL runs on `create_collection`/`drop_collection`
//! (no migration and no identifier quoting for arbitrary collection names), the
//! `(collection, id)` primary key still guarantees id uniqueness per collection,
//! and one index serves every collection.
//!
//! Filtering, ordering, and paging happen in Rust on the collection's rows via
//! `super::run_query`, so `SQLite` rankings cannot differ from the other
//! backends. Rows are read in `id` order to match the in-memory and file
//! backends' iteration order, which makes ties stable across backends too.

use std::str::FromStr;

use async_trait::async_trait;
use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
use sqlx::{Row, SqlitePool};

use openjiuwen_common::error::{Error, Result};

use super::{matching, run_query, validate_collection, Document, DocumentStore, Query};

/// Collection registry DDL, applied idempotently by every constructor.
const SCHEMA_COLLECTIONS: &str = "
CREATE TABLE IF NOT EXISTS document_collections (
    name TEXT PRIMARY KEY
)";

/// Document DDL, applied idempotently by every constructor.
const SCHEMA_DOCUMENTS: &str = "
CREATE TABLE IF NOT EXISTS documents (
    collection TEXT NOT NULL,
    id TEXT NOT NULL,
    doc_json TEXT NOT NULL,
    PRIMARY KEY (collection, id)
)";

/// Index used by every per-collection read and delete.
const DOCUMENTS_INDEX: &str =
    "CREATE INDEX IF NOT EXISTS documents_collection_idx ON documents (collection)";

/// Registers a collection name, keeping an existing row untouched.
const INSERT_COLLECTION: &str =
    "INSERT INTO document_collections (name) VALUES (?1) ON CONFLICT(name) DO NOTHING";

/// Inserts or replaces one document.
const UPSERT_DOCUMENT: &str = "
INSERT INTO documents (collection, id, doc_json) VALUES (?1, ?2, ?3)
ON CONFLICT(collection, id) DO UPDATE SET doc_json = excluded.doc_json";

/// A [`DocumentStore`] backed by a `SQLite` database.
///
/// Clones share the same connection pool.
#[derive(Debug, Clone)]
pub struct SqliteDocumentStore {
    /// The connection pool every operation borrows from.
    pool: SqlitePool,
}

impl SqliteDocumentStore {
    /// Connects to `url` (for example `sqlite::memory:` or `sqlite:documents.db`).
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
        for statement in [SCHEMA_COLLECTIONS, SCHEMA_DOCUMENTS, DOCUMENTS_INDEX] {
            sqlx::query(statement)
                .execute(&pool)
                .await
                .map_err(Error::external)?;
        }
        Ok(Self { pool })
    }

    /// The underlying pool, for callers that need to run companion queries.
    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }

    /// Loads every document of `collection`, in `id` order.
    async fn load_documents(&self, collection: &str) -> Result<Vec<Document>> {
        let rows = sqlx::query("SELECT doc_json FROM documents WHERE collection = ?1 ORDER BY id")
            .bind(collection)
            .fetch_all(&self.pool)
            .await
            .map_err(Error::external)?;
        rows.iter().map(decode_row).collect()
    }
}

#[async_trait]
impl DocumentStore for SqliteDocumentStore {
    async fn create_collection(&self, collection: &str) -> Result<()> {
        validate_collection(collection)?;
        sqlx::query(INSERT_COLLECTION)
            .bind(collection)
            .execute(&self.pool)
            .await
            .map_err(Error::external)?;
        Ok(())
    }

    async fn list_collections(&self) -> Result<Vec<String>> {
        let rows = sqlx::query("SELECT name FROM document_collections ORDER BY name")
            .fetch_all(&self.pool)
            .await
            .map_err(Error::external)?;
        rows.iter()
            .map(|row| row.try_get("name").map_err(Error::external))
            .collect()
    }

    async fn drop_collection(&self, collection: &str) -> Result<()> {
        validate_collection(collection)?;
        let mut transaction = self.pool.begin().await.map_err(Error::external)?;
        sqlx::query("DELETE FROM documents WHERE collection = ?1")
            .bind(collection)
            .execute(&mut *transaction)
            .await
            .map_err(Error::external)?;
        sqlx::query("DELETE FROM document_collections WHERE name = ?1")
            .bind(collection)
            .execute(&mut *transaction)
            .await
            .map_err(Error::external)?;
        transaction.commit().await.map_err(Error::external)
    }

    async fn upsert(&self, collection: &str, documents: Vec<Document>) -> Result<usize> {
        validate_collection(collection)?;
        let written = documents.len();
        let mut transaction = self.pool.begin().await.map_err(Error::external)?;
        sqlx::query(INSERT_COLLECTION)
            .bind(collection)
            .execute(&mut *transaction)
            .await
            .map_err(Error::external)?;
        for document in documents {
            let encoded = serde_json::to_string(&document).map_err(Error::external)?;
            sqlx::query(UPSERT_DOCUMENT)
                .bind(collection)
                .bind(&document.id)
                .bind(encoded)
                .execute(&mut *transaction)
                .await
                .map_err(Error::external)?;
        }
        transaction.commit().await.map_err(Error::external)?;
        Ok(written)
    }

    async fn get(&self, collection: &str, id: &str) -> Result<Option<Document>> {
        validate_collection(collection)?;
        let row = sqlx::query("SELECT doc_json FROM documents WHERE collection = ?1 AND id = ?2")
            .bind(collection)
            .bind(id)
            .fetch_optional(&self.pool)
            .await
            .map_err(Error::external)?;
        row.as_ref().map(decode_row).transpose()
    }

    async fn query(&self, collection: &str, query: &Query) -> Result<Vec<Document>> {
        validate_collection(collection)?;
        let documents = self.load_documents(collection).await?;
        Ok(run_query(&documents, query).into_iter().cloned().collect())
    }

    async fn delete(&self, collection: &str, query: &Query) -> Result<usize> {
        validate_collection(collection)?;
        let documents = self.load_documents(collection).await?;
        let doomed: Vec<String> = matching(&documents, &query.filter)
            .into_iter()
            .map(|document| document.id.clone())
            .collect();
        if doomed.is_empty() {
            return Ok(0);
        }
        let removed = doomed.len();
        let mut transaction = self.pool.begin().await.map_err(Error::external)?;
        for id in &doomed {
            sqlx::query("DELETE FROM documents WHERE collection = ?1 AND id = ?2")
                .bind(collection)
                .bind(id)
                .execute(&mut *transaction)
                .await
                .map_err(Error::external)?;
        }
        transaction.commit().await.map_err(Error::external)?;
        Ok(removed)
    }

    async fn count(&self, collection: &str, query: &Query) -> Result<usize> {
        validate_collection(collection)?;
        let documents = self.load_documents(collection).await?;
        Ok(matching(&documents, &query.filter).len())
    }
}

/// Decodes a `doc_json` row into a [`Document`].
fn decode_row(row: &sqlx::sqlite::SqliteRow) -> Result<Document> {
    let encoded: String = row.try_get("doc_json").map_err(Error::external)?;
    serde_json::from_str(&encoded).map_err(Error::external)
}

#[cfg(test)]
mod tests {
    use super::super::contract;
    use super::*;
    use serde_json::json;

    async fn store() -> SqliteDocumentStore {
        SqliteDocumentStore::connect("sqlite::memory:")
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn satisfies_the_shared_contract() {
        contract::run(&store().await).await;
    }

    #[tokio::test]
    async fn documents_round_trip_through_the_json_column() {
        let store = store().await;
        let document = Document::new("nested")
            .with_field(
                "meta",
                json!({"owner": "ada", "list": [1, 2, {"deep": true}]}),
            )
            .with_field("nil", serde_json::Value::Null);
        store.upsert("docs", vec![document.clone()]).await.unwrap();

        let loaded = store.get("docs", "nested").await.unwrap().expect("stored");
        assert_eq!(loaded, document);
    }

    #[tokio::test]
    async fn one_table_holds_every_collection() {
        let store = store().await;
        store
            .upsert("first", vec![Document::new("a"), Document::new("b")])
            .await
            .unwrap();
        store.create_collection("second").await.unwrap();
        store
            .upsert("second", vec![Document::new("a")])
            .await
            .unwrap();

        let documents: i64 = sqlx::query("SELECT COUNT(*) AS n FROM documents")
            .fetch_one(store.pool())
            .await
            .unwrap()
            .try_get("n")
            .unwrap();
        assert_eq!(documents, 3);
        let collections: i64 = sqlx::query("SELECT COUNT(*) AS n FROM document_collections")
            .fetch_one(store.pool())
            .await
            .unwrap()
            .try_get("n")
            .unwrap();
        assert_eq!(collections, 2);

        store.drop_collection("first").await.unwrap();
        let remaining: i64 = sqlx::query("SELECT COUNT(*) AS n FROM documents")
            .fetch_one(store.pool())
            .await
            .unwrap()
            .try_get("n")
            .unwrap();
        assert_eq!(remaining, 1, "dropping a collection removes its rows only");
        assert_eq!(store.list_collections().await.unwrap(), vec!["second"]);
        // The same id in another collection is untouched.
        assert!(store.get("second", "a").await.unwrap().is_some());
    }

    #[tokio::test]
    async fn from_pool_shares_state() {
        let store = store().await;
        let sibling = SqliteDocumentStore::from_pool(store.pool().clone())
            .await
            .unwrap();
        sibling
            .upsert("docs", vec![Document::new("a").with_field("n", 1)])
            .await
            .unwrap();
        let document = store.get("docs", "a").await.unwrap().expect("shared");
        assert_eq!(document.field("n"), Some(&json!(1)));
    }

    #[tokio::test]
    async fn documents_survive_a_reconnect() {
        let dir = tempfile::tempdir().unwrap();
        let url = format!("sqlite:{}/documents.db", dir.path().display());
        {
            let store = SqliteDocumentStore::connect(&url).await.unwrap();
            store
                .upsert(
                    "docs",
                    vec![
                        Document::new("a").with_field("name", "Ada"),
                        Document::new("b").with_field("name", "Bob"),
                    ],
                )
                .await
                .unwrap();
            store.create_collection("empty").await.unwrap();
            store
                .delete(
                    "docs",
                    &Query {
                        filter: json!({"id": "b"}),
                        ..Query::default()
                    },
                )
                .await
                .unwrap();
        }

        let reopened = SqliteDocumentStore::connect(&url).await.unwrap();
        assert_eq!(
            reopened.list_collections().await.unwrap(),
            vec!["docs", "empty"]
        );
        assert!(reopened.get("docs", "a").await.unwrap().is_some());
        assert!(reopened.get("docs", "b").await.unwrap().is_none());
    }
}
