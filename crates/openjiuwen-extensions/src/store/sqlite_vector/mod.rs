//! `SQLite`-backed vector store (feature `sqlite`).
//!
//! The Python `extensions/store/` package ships no `SQLite` vector backend, so
//! this is a Rust-side member of the family: it implements the kernel
//! [`VectorStore`] contract on the kernel's own `SQLite` conventions — one
//! `SqlitePool`, idempotent `CREATE TABLE IF NOT EXISTS` DDL applied by every
//! constructor, one transaction per mutation, and `Error::external` for every
//! `SQLx` failure — so it sits next to the kernel's `SqliteKvStore` and
//! `SqliteDocumentStore` without introducing another connection style.
//!
//! # Row and blob layout
//!
//! Three fixed tables hold a store. The schema never changes, so no DDL runs per
//! collection: a collection name is data, never an identifier, and therefore
//! never needs quoting.
//!
//! - `openjiuwen_vector_collections(name TEXT PRIMARY KEY, schema_json TEXT NOT NULL,
//!   metric TEXT NOT NULL)` — one registry row per collection holding the
//!   serialized [`CollectionSchema`] (it round-trips exactly, including
//!   `is_primary`, `auto_id`, `max_length`, `dim` and `default_value`) plus the
//!   similarity metric the collection was created with.
//! - `openjiuwen_vector_documents(collection TEXT NOT NULL, id TEXT NOT NULL,
//!   doc_json TEXT NOT NULL, PRIMARY KEY(collection, id))` — one row per
//!   document: its scalar, JSON, and unknown fields as a JSON object, with the
//!   `FLOAT_VECTOR` fields stripped out (they live in the next table).
//! - `openjiuwen_vector_embeddings(collection TEXT NOT NULL, id TEXT NOT NULL,
//!   field TEXT NOT NULL, dim INTEGER NOT NULL, embedding BLOB NOT NULL,
//!   PRIMARY KEY(collection, id, field))` — one row per vector field of a
//!   document, so a schema with several `FLOAT_VECTOR` fields stores one blob
//!   each. `embedding` is exactly `dim` little-endian `f32` values written with
//!   `f32::to_le_bytes` (`4 * dim` bytes), so an embedding never round-trips
//!   through text and never loses precision.
//!
//! The `openjiuwen_` prefix on the three table names is load-bearing: `SQLite`
//! reserves the `sqlite_` prefix for its own internal objects.
//!
//! # Brute force
//!
//! There is no ANN index. A search reads every embedding row of the target
//! vector field (joined with its document row, ordered by primary key), scores
//! it against the query vector with the collection's [`Metric`], and sorts in
//! Rust. That is the same complexity as the kernel reference store and is the
//! documented trade-off of a self-contained local backend: a query is linear in
//! the collection size, and there is no quantization, no clustering and no
//! approximate recall.
//!
//! # Reference parity
//!
//! Scoring ([`Metric::score`], including the `[0, 1]` normalization and the
//! zero-vector convention), filter evaluation (the kernel's shared filter
//! language), ordering (score descending, primary key ascending for ties), and
//! `top_k` truncation all match the kernel reference store exactly.
//! `add_docs` delegates document preparation to a throwaway in-memory reference
//! store: that reference validates the batch, fills `default_value`s, mints
//! `auto_id`s, and type-checks every field before anything is written, and only
//! the *prepared* documents are persisted. A rejected batch therefore inserts
//! nothing and surfaces the reference error unchanged.
//!
//! # Metric
//!
//! The metric belongs to the collection, not to the query: `create_collection`
//! records [`SqliteVectorStore::metric`] (cosine unless
//! [`SqliteVectorStore::with_metric`] says otherwise) with the schema, and every
//! later operation compares that recorded value with the store's configured one.
//! A store opened with a different metric answers a validation error instead of
//! silently ranking the collection differently, and a recorded metric name this
//! backend does not recognize is a validation error too.
//!
//! # Errors
//!
//! - [`ErrorKind::NotFound`](openjiuwen_common::error::ErrorKind::NotFound) —
//!   `add_docs`, `search`, `delete_docs_by_ids`, `delete_docs_by_filters` and
//!   [`SqliteVectorStore::count`] on an unknown collection, while
//!   `delete_collection` on a missing collection is a no-op and
//!   `get_schema`/`collection_exists` report the absence instead.
//! - [`ErrorKind::Validation`](openjiuwen_common::error::ErrorKind::Validation) —
//!   an invalid schema or dimension, an empty or pathological collection name, a
//!   duplicate primary key, a mistyped or missing field, a query vector whose
//!   length disagrees with the collection's `dim`, a collection metric that
//!   disagrees with the store's (or one that cannot be decoded), and a filter
//!   that is neither `null` nor an object or that names an unknown operator.
//! - [`ErrorKind::External`](openjiuwen_common::error::ErrorKind::External) — a
//!   `SQLx` failure, or a stored row that does not match this layout (invalid
//!   JSON, a schema that no longer parses, or a blob whose length is not
//!   `4 * dim`). Nothing degrades silently.
#![allow(clippy::module_name_repetitions)] // `SqliteVectorStore` is the contract's name.
use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_core::foundation::store::{
    filter, CollectionSchema, FieldSchema, InMemoryVectorStore, Metric, VectorQuery,
    VectorSearchResult, VectorStore,
};
use serde_json::{Map, Value};
use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
use sqlx::{Row, SqlitePool};
use std::collections::BTreeMap;
use std::fmt;
use std::str::FromStr;
/// Collection registry DDL, applied idempotently by every constructor.
const SCHEMA_COLLECTIONS: &str = "
CREATE TABLE IF NOT EXISTS openjiuwen_vector_collections (
    name TEXT PRIMARY KEY,
    schema_json TEXT NOT NULL,
    metric TEXT NOT NULL
)";
/// Document DDL, applied idempotently by every constructor.
const SCHEMA_DOCUMENTS: &str = "
CREATE TABLE IF NOT EXISTS openjiuwen_vector_documents (
    collection TEXT NOT NULL,
    id TEXT NOT NULL,
    doc_json TEXT NOT NULL,
    PRIMARY KEY (collection, id)
)";
/// Embedding DDL: `embedding` is `4 * dim` little-endian `f32` bytes.
const SCHEMA_EMBEDDINGS: &str = "
CREATE TABLE IF NOT EXISTS openjiuwen_vector_embeddings (
    collection TEXT NOT NULL,
    id TEXT NOT NULL,
    field TEXT NOT NULL,
    dim INTEGER NOT NULL,
    embedding BLOB NOT NULL,
    PRIMARY KEY (collection, id, field)
)";
/// Index used by every per-collection document read and delete.
const DOCUMENTS_INDEX: &str =
    "CREATE INDEX IF NOT EXISTS openjiuwen_vector_documents_collection_idx ON openjiuwen_vector_documents (collection)";
/// Index used by every per-collection embedding read and delete.
const EMBEDDINGS_INDEX: &str = "CREATE INDEX IF NOT EXISTS openjiuwen_vector_embeddings_field_idx \
                                 ON openjiuwen_vector_embeddings (collection, field)";
/// Registers a collection name; the conflict arm makes duplicate detection atomic.
const INSERT_COLLECTION: &str = "
INSERT INTO openjiuwen_vector_collections (name, schema_json, metric) VALUES (?1, ?2, ?3)
ON CONFLICT(name) DO NOTHING";
/// Inserts one document row (its vector fields are stored separately).
const INSERT_DOCUMENT: &str =
    "INSERT INTO openjiuwen_vector_documents (collection, id, doc_json) VALUES (?1, ?2, ?3)";
/// Inserts one embedding blob.
const INSERT_EMBEDDING: &str =
    "INSERT INTO openjiuwen_vector_embeddings (collection, id, field, dim, embedding) \
     VALUES (?1, ?2, ?3, ?4, ?5)";
/// The collection name the in-memory reference is created under during an add.
const SCRATCH_COLLECTION: &str = "sqlite_vector_scratch";
/// Upper bound on the ids of one `IN` list, kept well below `SQLite`'s parameter limit.
const ID_CHUNK: usize = 500;
/// Filter operators this backend evaluates, named for its error messages.
const KNOWN_OPERATORS: [&str; 8] = [
    "$eq",
    "$ne",
    "$gt",
    "$gte",
    "$lt",
    "$lte",
    "$in",
    "$contains",
];
/// A [`VectorStore`] backed by a `SQLite` database.
///
/// Search is brute force over the stored embedding blobs; see the
/// [module docs](self) for the row layout and the metric policy. Clones share
/// the same connection pool.
#[derive(Debug, Clone)]
pub struct SqliteVectorStore {
    /// The connection pool every operation borrows from.
    pool: SqlitePool,
    /// The similarity metric newly created collections are registered with.
    metric: Metric,
}
/// One collection's registry row.
#[derive(Debug, Clone)]
struct CollectionRow {
    /// The schema the collection was created with.
    schema: CollectionSchema,
    /// The metric the collection was created with.
    metric: Metric,
}
/// One stored document: its non-vector fields plus its embeddings.
#[derive(Debug, Clone)]
struct StoredDocument {
    /// Primary key, as stored in the `id` column.
    id: String,
    /// Every non-vector field, exactly as stored.
    fields: Map<String, Value>,
    /// Embeddings, keyed by vector field name.
    vectors: BTreeMap<String, Vec<f32>>,
}
impl StoredDocument {
    /// The document with vector fields restored as arrays, i.e. the view the
    /// kernel reference store filters on.
    fn view(&self) -> Map<String, Value> {
        let mut fields = self.fields.clone();
        for (name, vector) in &self.vectors {
            fields.insert(
                name.clone(),
                Value::Array(vector.iter().map(|value| Value::from(*value)).collect()),
            );
        }
        fields
    }
}
/// A stored row that does not match the layout this backend writes.
///
/// Kept crate-private: callers only ever see it as the source of an
/// [`Error::External`] value.
#[derive(Debug)]
struct CorruptRow(String);
impl fmt::Display for CorruptRow {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}
impl std::error::Error for CorruptRow {}
impl SqliteVectorStore {
    /// Connects to `url` (for example `sqlite::memory:` or `sqlite:vectors.db`).
    ///
    /// In-memory URLs are pinned to a single pooled connection, otherwise every
    /// checkout would open a fresh, empty database. Collections are registered
    /// with the default metric ([`Metric::Cosine`]); chain
    /// [`SqliteVectorStore::with_metric`] to create them with another one.
    ///
    /// # Errors
    ///
    /// Returns the URL parse error or the connection error of `SQLx` as
    /// [`ErrorKind::External`](openjiuwen_common::error::ErrorKind::External).
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
    /// Wraps an existing pool (created by a caller or another subsystem) and
    /// makes sure the three tables and their indices exist.
    ///
    /// # Errors
    ///
    /// Returns the DDL failure of `SQLx` as
    /// [`ErrorKind::External`](openjiuwen_common::error::ErrorKind::External).
    pub async fn from_pool(pool: SqlitePool) -> Result<Self> {
        for statement in [
            SCHEMA_COLLECTIONS,
            SCHEMA_DOCUMENTS,
            SCHEMA_EMBEDDINGS,
            DOCUMENTS_INDEX,
            EMBEDDINGS_INDEX,
        ] {
            sqlx::query(statement)
                .execute(&pool)
                .await
                .map_err(Error::external)?;
        }
        Ok(Self {
            pool,
            metric: Metric::default(),
        })
    }
    /// The underlying pool, for callers that need to run companion queries.
    #[must_use]
    pub const fn pool(&self) -> &SqlitePool {
        &self.pool
    }
    /// The metric collections created through this store are registered with.
    #[must_use]
    pub const fn metric(&self) -> Metric {
        self.metric
    }
    /// Replaces the metric used for newly created collections.
    ///
    /// Collections keep the metric they were created with, so a store configured
    /// with a different one refuses to operate on them instead of ranking them
    /// differently.
    #[must_use]
    pub fn with_metric(mut self, metric: Metric) -> Self {
        self.metric = metric;
        self
    }
    /// Number of documents stored in `collection`.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::NotFound`](openjiuwen_common::error::ErrorKind::NotFound)
    /// for an unknown collection, a validation error for an unusable collection
    /// name, and the `SQLx` failure as
    /// [`ErrorKind::External`](openjiuwen_common::error::ErrorKind::External).
    pub async fn count(&self, collection: &str) -> Result<usize> {
        self.load_collection(collection).await?;
        let row = sqlx::query(
            "SELECT COUNT(*) AS n FROM openjiuwen_vector_documents WHERE collection = ?1",
        )
        .bind(collection)
        .fetch_one(&self.pool)
        .await
        .map_err(Error::external)?;
        let count: i64 = row.try_get("n").map_err(Error::external)?;
        Ok(usize::try_from(count).unwrap_or(0))
    }
    /// Loads a collection's registry row, or reports it missing.
    async fn load_collection(&self, collection: &str) -> Result<CollectionRow> {
        validate_collection(collection)?;
        let row = sqlx::query(
            "SELECT schema_json, metric FROM openjiuwen_vector_collections WHERE name = ?1",
        )
        .bind(collection)
        .fetch_optional(&self.pool)
        .await
        .map_err(Error::external)?;
        let row = row.ok_or_else(|| collection_not_found(collection))?;
        let schema_json: String = row.try_get("schema_json").map_err(Error::external)?;
        let metric: String = row.try_get("metric").map_err(Error::external)?;
        Ok(CollectionRow {
            schema: decode_schema(collection, &schema_json)?,
            metric: decode_metric(collection, &metric)?,
        })
    }
    /// Rejects a collection whose recorded metric differs from this store's.
    fn ensure_metric(&self, collection: &str, recorded: Metric) -> Result<()> {
        if recorded == self.metric {
            return Ok(());
        }
        Err(Error::validation(format!(
            "vector collection {collection:?} uses the {} metric but this store is configured \
             for the {} metric; open the store with the collection's metric",
            metric_name(recorded),
            metric_name(self.metric)
        )))
    }
    /// Loads every document of `collection`, in primary-key order.
    async fn load_documents(&self, collection: &str) -> Result<Vec<StoredDocument>> {
        let rows = sqlx::query(
            "SELECT d.id AS id, d.doc_json AS doc_json, e.field AS field, e.dim AS dim, \
                    e.embedding AS embedding
             FROM openjiuwen_vector_documents d
             LEFT JOIN openjiuwen_vector_embeddings e
                 ON e.collection = d.collection AND e.id = d.id
             WHERE d.collection = ?1
             ORDER BY d.id, e.field",
        )
        .bind(collection)
        .fetch_all(&self.pool)
        .await
        .map_err(Error::external)?;
        let mut documents: Vec<StoredDocument> = Vec::new();
        for row in &rows {
            let id: String = row.try_get("id").map_err(Error::external)?;
            if documents.last().map(|document| document.id.as_str()) != Some(id.as_str()) {
                let doc_json: String = row.try_get("doc_json").map_err(Error::external)?;
                let fields = decode_fields(collection, &id, &doc_json)?;
                documents.push(StoredDocument {
                    id: id.clone(),
                    fields,
                    vectors: BTreeMap::new(),
                });
            }
            let field: Option<String> = row.try_get("field").map_err(Error::external)?;
            let Some(field) = field else {
                // A document without any embedding: it still takes part in
                // filter matching, it just cannot be scored.
                continue;
            };
            let dim: Option<i64> = row.try_get("dim").map_err(Error::external)?;
            let embedding: Option<Vec<u8>> = row.try_get("embedding").map_err(Error::external)?;
            let (Some(dim), Some(embedding)) = (dim, embedding) else {
                return Err(Error::external(CorruptRow(format!(
                    "stored embedding {field:?} of a document in collection {collection:?} \
                     is missing its dim or payload"
                ))));
            };
            let dim = usize::try_from(dim).map_err(|_| {
                Error::external(CorruptRow(format!(
                    "stored embedding {field:?} of a document in collection {collection:?} \
                     has a negative dim ({dim})"
                )))
            })?;
            let vector = decode_vector(collection, &id, &field, &embedding, dim)?;
            if let Some(document) = documents.last_mut() {
                document.vectors.insert(field, vector);
            }
        }
        Ok(documents)
    }
    /// The ids of `ids` already stored in `collection`.
    async fn existing_ids(&self, collection: &str, ids: &[String]) -> Result<Vec<String>> {
        let mut existing = Vec::new();
        for chunk in ids.chunks(ID_CHUNK) {
            let sql = format!(
                "SELECT id FROM openjiuwen_vector_documents WHERE collection = ?1 AND id IN ({})",
                placeholders(2, chunk.len())
            );
            let mut query = sqlx::query(&sql).bind(collection);
            for id in chunk {
                query = query.bind(id);
            }
            let rows = query.fetch_all(&self.pool).await.map_err(Error::external)?;
            for row in &rows {
                existing.push(row.try_get::<String, _>("id").map_err(Error::external)?);
            }
        }
        Ok(existing)
    }
}
#[async_trait]
impl VectorStore for SqliteVectorStore {
    async fn create_collection(&self, collection: &str, schema: CollectionSchema) -> Result<()> {
        schema.validate()?;
        validate_collection(collection)?;
        let encoded = serde_json::to_string(&schema).map_err(Error::external)?;
        let result = sqlx::query(INSERT_COLLECTION)
            .bind(collection)
            .bind(encoded)
            .bind(metric_name(self.metric))
            .execute(&self.pool)
            .await
            .map_err(Error::external)?;
        if result.rows_affected() == 0 {
            return Err(Error::validation(format!(
                "vector collection {collection:?} already exists"
            )));
        }
        Ok(())
    }
    async fn delete_collection(&self, collection: &str) -> Result<()> {
        validate_collection(collection)?;
        let mut transaction = self.pool.begin().await.map_err(Error::external)?;
        for statement in [
            "DELETE FROM openjiuwen_vector_embeddings WHERE collection = ?1",
            "DELETE FROM openjiuwen_vector_documents WHERE collection = ?1",
            "DELETE FROM openjiuwen_vector_collections WHERE name = ?1",
        ] {
            sqlx::query(statement)
                .bind(collection)
                .execute(&mut *transaction)
                .await
                .map_err(Error::external)?;
        }
        transaction.commit().await.map_err(Error::external)
    }
    async fn collection_exists(&self, collection: &str) -> Result<bool> {
        validate_collection(collection)?;
        let row =
            sqlx::query("SELECT 1 AS found FROM openjiuwen_vector_collections WHERE name = ?1")
                .bind(collection)
                .fetch_optional(&self.pool)
                .await
                .map_err(Error::external)?;
        Ok(row.is_some())
    }
    async fn get_schema(&self, collection: &str) -> Result<Option<CollectionSchema>> {
        validate_collection(collection)?;
        let row =
            sqlx::query("SELECT schema_json FROM openjiuwen_vector_collections WHERE name = ?1")
                .bind(collection)
                .fetch_optional(&self.pool)
                .await
                .map_err(Error::external)?;
        let Some(row) = row else {
            return Ok(None);
        };
        let encoded: String = row.try_get("schema_json").map_err(Error::external)?;
        Ok(Some(decode_schema(collection, &encoded)?))
    }
    async fn add_docs(&self, collection: &str, docs: Vec<Map<String, Value>>) -> Result<usize> {
        let registry = self.load_collection(collection).await?;
        self.ensure_metric(collection, registry.metric)?;
        let prepared = prepare_batch(&registry.schema, docs).await?;
        if prepared.is_empty() {
            return Ok(0);
        }
        let primary = registry
            .schema
            .primary_key()
            .ok_or_else(|| Error::validation("collection schema declares no primary key field"))?;
        let ids = prepared
            .iter()
            .map(|document| primary_key_string(primary, document))
            .collect::<Result<Vec<String>>>()?;
        let existing = self.existing_ids(collection, &ids).await?;
        if let Some(duplicate) = existing.first() {
            return Err(Error::validation(format!(
                "document with primary key {duplicate:?} already exists in collection \
                 {collection:?}"
            )));
        }
        let mut transaction = self.pool.begin().await.map_err(Error::external)?;
        for (id, document) in ids.iter().zip(prepared) {
            let (fields, vectors) = split_document(&registry.schema, document)?;
            let encoded = serde_json::to_string(&Value::Object(fields)).map_err(Error::external)?;
            sqlx::query(INSERT_DOCUMENT)
                .bind(collection)
                .bind(id)
                .bind(encoded)
                .execute(&mut *transaction)
                .await
                .map_err(Error::external)?;
            for (field, vector) in &vectors {
                let dim = i64::try_from(vector.len())
                    .map_err(|_| Error::validation("vector is too large to store"))?;
                sqlx::query(INSERT_EMBEDDING)
                    .bind(collection)
                    .bind(id)
                    .bind(field)
                    .bind(dim)
                    .bind(encode_vector(vector))
                    .execute(&mut *transaction)
                    .await
                    .map_err(Error::external)?;
            }
        }
        transaction.commit().await.map_err(Error::external)?;
        Ok(ids.len())
    }
    async fn search(
        &self,
        collection: &str,
        query: &VectorQuery,
    ) -> Result<Vec<VectorSearchResult>> {
        let registry = self.load_collection(collection).await?;
        self.ensure_metric(collection, registry.metric)?;
        validate_filters(&query.filters)?;
        let field = resolve_vector_field(&registry.schema, collection, query)?;
        let dim = field.dim.unwrap_or(0);
        if query.query_vector.len() != dim {
            return Err(Error::validation(format!(
                "query vector dimension {} does not match vector field {:?} dimension {dim} of \
                 collection {collection:?}",
                query.query_vector.len(),
                field.name
            )));
        }
        let mut hits: Vec<(f32, String, Map<String, Value>)> = Vec::new();
        for document in self.load_documents(collection).await? {
            let Some(vector) = document.vectors.get(&field.name) else {
                // Only a row edited outside this backend can be missing its
                // embedding; it simply cannot match, exactly as in the reference.
                continue;
            };
            let view = document.view();
            if !filter::matches(&query.filters, &Value::Object(view.clone())) {
                continue;
            }
            hits.push((
                registry.metric.score(&query.query_vector, vector),
                document.id.clone(),
                view,
            ));
        }
        hits.sort_by(|left, right| {
            right
                .0
                .total_cmp(&left.0)
                .then_with(|| left.1.cmp(&right.1))
        });
        hits.truncate(query.top_k);
        Ok(hits
            .into_iter()
            .map(|(score, _, fields)| VectorSearchResult::new(score, fields))
            .collect())
    }
    async fn delete_docs_by_ids(&self, collection: &str, ids: &[String]) -> Result<usize> {
        self.load_collection(collection).await?;
        if ids.is_empty() {
            return Ok(0);
        }
        let mut transaction = self.pool.begin().await.map_err(Error::external)?;
        let mut deleted = 0_usize;
        for chunk in ids.chunks(ID_CHUNK) {
            let placeholders = placeholders(2, chunk.len());
            let embeddings = format!(
                "DELETE FROM openjiuwen_vector_embeddings WHERE collection = ?1 AND id IN \
                 ({placeholders})"
            );
            let mut query = sqlx::query(&embeddings).bind(collection);
            for id in chunk {
                query = query.bind(id);
            }
            query
                .execute(&mut *transaction)
                .await
                .map_err(Error::external)?;
            let documents = format!(
                "DELETE FROM openjiuwen_vector_documents WHERE collection = ?1 AND id IN \
                 ({placeholders})"
            );
            let mut query = sqlx::query(&documents).bind(collection);
            for id in chunk {
                query = query.bind(id);
            }
            let result = query
                .execute(&mut *transaction)
                .await
                .map_err(Error::external)?;
            deleted = deleted
                .saturating_add(usize::try_from(result.rows_affected()).unwrap_or(usize::MAX));
        }
        transaction.commit().await.map_err(Error::external)?;
        Ok(deleted)
    }
    async fn delete_docs_by_filters(&self, collection: &str, filters: &Value) -> Result<usize> {
        self.load_collection(collection).await?;
        validate_filters(filters)?;
        let doomed: Vec<String> = self
            .load_documents(collection)
            .await?
            .into_iter()
            .filter(|document| filter::matches(filters, &Value::Object(document.view())))
            .map(|document| document.id)
            .collect();
        self.delete_docs_by_ids(collection, &doomed).await
    }
    async fn list_collection_names(&self) -> Result<Vec<String>> {
        let rows = sqlx::query("SELECT name FROM openjiuwen_vector_collections ORDER BY name")
            .fetch_all(&self.pool)
            .await
            .map_err(Error::external)?;
        rows.iter()
            .map(|row| row.try_get("name").map_err(Error::external))
            .collect()
    }
}
/// The shared not-found error for a missing collection.
fn collection_not_found(collection: &str) -> Error {
    Error::not_found(format!("vector collection {collection:?} not found"))
}
/// Rejects collection names that could corrupt a backend's namespace.
///
/// The name is stored as a column value, never as an identifier, so this matches
/// the kernel's rule rather than quoting: a name must be non-empty, must not
/// contain `/`, `\`, or NUL, and must not be `.` or `..`.
fn validate_collection(collection: &str) -> Result<()> {
    let invalid = if collection.is_empty() {
        Some("must not be empty")
    } else if collection.contains('/') || collection.contains('\\') {
        Some("must not contain a path separator")
    } else if collection.contains('\0') {
        Some("must not contain a NUL byte")
    } else if collection == "." || collection == ".." {
        Some("must not be a dot segment")
    } else {
        None
    };
    match invalid {
        Some(reason) => Err(Error::validation(format!(
            "invalid vector collection name `{collection}`: {reason}"
        ))),
        None => Ok(()),
    }
}
/// The stored name of `metric`, also used in error messages.
const fn metric_name(metric: Metric) -> &'static str {
    match metric {
        Metric::Cosine => "cosine",
        Metric::Dot => "dot",
        Metric::Euclidean => "euclidean",
    }
}
/// Decodes a stored metric name, rejecting anything this backend cannot score.
fn decode_metric(collection: &str, raw: &str) -> Result<Metric> {
    match raw {
        "cosine" => Ok(Metric::Cosine),
        "dot" => Ok(Metric::Dot),
        "euclidean" => Ok(Metric::Euclidean),
        other => Err(Error::validation(format!(
            "vector collection {collection:?} records the unknown metric {other:?}; expected \
             cosine, dot, or euclidean"
        ))),
    }
}
/// Decodes a stored collection schema.
fn decode_schema(collection: &str, encoded: &str) -> Result<CollectionSchema> {
    serde_json::from_str(encoded).map_err(|error| {
        Error::external(CorruptRow(format!(
            "vector collection {collection:?} does not hold a valid schema: {error}"
        )))
    })
}
/// Decodes a stored document's non-vector fields.
fn decode_fields(collection: &str, id: &str, encoded: &str) -> Result<Map<String, Value>> {
    let corrupt = |detail: &str| {
        Error::external(CorruptRow(format!(
            "vector document {id:?} of collection {collection:?} {detail}"
        )))
    };
    match serde_json::from_str(encoded) {
        Ok(Value::Object(fields)) => Ok(fields),
        Ok(other) => Err(corrupt(&format!("is not a JSON object: {other}"))),
        Err(error) => Err(corrupt(&format!("does not hold valid JSON: {error}"))),
    }
}
/// Encodes an embedding as little-endian `f32` components.
fn encode_vector(vector: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(vector.len() * 4);
    for component in vector {
        bytes.extend_from_slice(&component.to_le_bytes());
    }
    bytes
}
/// Decodes an embedding blob, checking it against the row's recorded dimension.
fn decode_vector(
    collection: &str,
    id: &str,
    field: &str,
    bytes: &[u8],
    dim: usize,
) -> Result<Vec<f32>> {
    if bytes.len() != dim.saturating_mul(4) {
        return Err(Error::external(CorruptRow(format!(
            "embedding {field:?} of vector document {id:?} in collection {collection:?} is {} \
             bytes, expected {} for dim {dim}",
            bytes.len(),
            dim.saturating_mul(4)
        ))));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}
/// The string form of a prepared document's primary key, as stored in the `id`
/// column.
///
/// A numeric primary key keeps its number in the document fields (the reference
/// store does the same), so both forms are accepted here.
fn primary_key_string(primary: &FieldSchema, fields: &Map<String, Value>) -> Result<String> {
    match fields.get(&primary.name) {
        Some(Value::String(id)) => Ok(id.clone()),
        Some(value) if value.is_i64() || value.is_u64() => Ok(value.to_string()),
        Some(other) => Err(Error::validation(format!(
            "prepared document has primary key field {:?} of type {other}; a string or integer \
             is required",
            primary.name
        ))),
        None => Err(Error::validation(format!(
            "prepared document is missing its primary key field {:?}",
            primary.name
        ))),
    }
}
/// A prepared document split into its non-vector fields and its embeddings.
type SplitDocument = (Map<String, Value>, BTreeMap<String, Vec<f32>>);
/// Splits a prepared document into its non-vector fields and its embeddings.
fn split_document(
    schema: &CollectionSchema,
    document: Map<String, Value>,
) -> Result<SplitDocument> {
    let mut fields = document;
    let mut vectors: BTreeMap<String, Vec<f32>> = BTreeMap::new();
    for field in schema.vector_fields() {
        let Some(value) = fields.remove(&field.name) else {
            continue;
        };
        let vector = vector_components(&value).ok_or_else(|| {
            Error::validation(format!(
                "vector field {:?} is not an array of finite numbers: {value}",
                field.name
            ))
        })?;
        vectors.insert(field.name.clone(), vector);
    }
    Ok((fields, vectors))
}
/// Reads a vector field value into `f32` components.
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)] // Mirrors the reference's `f64`-to-`f32` narrowing, which is checked for finiteness.
fn vector_components(value: &Value) -> Option<Vec<f32>> {
    value
        .as_array()?
        .iter()
        .map(|item| {
            let component = item.as_f64()? as f32;
            component.is_finite().then_some(component)
        })
        .collect()
}
/// Validates a batch and returns the documents as the reference store prepares
/// them: defaults filled in, `auto_id`s minted, every field type-checked.
///
/// The throwaway reference store owns the rules, so a persisted document can
/// never differ from the one the kernel would have stored.
async fn prepare_batch(
    schema: &CollectionSchema,
    docs: Vec<Map<String, Value>>,
) -> Result<Vec<Map<String, Value>>> {
    if docs.is_empty() {
        return Ok(Vec::new());
    }
    let count = docs.len();
    let reference = InMemoryVectorStore::new();
    reference
        .create_collection(SCRATCH_COLLECTION, schema.clone())
        .await?;
    reference.add_docs(SCRATCH_COLLECTION, docs).await?;
    let dim = schema
        .default_vector_field()
        .and_then(|field| field.dim)
        .unwrap_or(0);
    let query = VectorQuery {
        query_vector: vec![0.0; dim],
        vector_field: None,
        top_k: count,
        filters: Value::Null,
    };
    let hits = reference.search_with_metric(SCRATCH_COLLECTION, &query, Metric::Cosine)?;
    Ok(hits.into_iter().map(|hit| hit.fields).collect())
}
/// Picks the vector field a query scores against.
fn resolve_vector_field<'a>(
    schema: &'a CollectionSchema,
    collection: &str,
    query: &VectorQuery,
) -> Result<&'a FieldSchema> {
    match &query.vector_field {
        Some(name) => schema.vector_field(name).ok_or_else(|| {
            Error::validation(format!(
                "field {name:?} is not a FLOAT_VECTOR field of collection {collection:?}"
            ))
        }),
        None => schema.default_vector_field().ok_or_else(|| {
            Error::validation(format!(
                "collection {collection:?} declares no FLOAT_VECTOR field"
            ))
        }),
    }
}
/// Rejects a filter this backend cannot evaluate literally.
///
/// The kernel filter language silently treats a malformed filter as "matches
/// nothing"; a vector search here is explicit instead, so a caller learns about a
/// mistyped filter rather than getting an empty result set.
fn validate_filters(filters: &Value) -> Result<()> {
    match filters {
        Value::Null => Ok(()),
        Value::Object(entries) => validate_filter_object(entries),
        other => Err(Error::validation(format!(
            "vector search filters must be a JSON object or null, got {other}"
        ))),
    }
}
/// Validates one filter object: combinators plus field conditions.
fn validate_filter_object(entries: &Map<String, Value>) -> Result<()> {
    for (key, condition) in entries {
        match key.as_str() {
            "$and" | "$or" => {
                let sub_filters = condition.as_array().ok_or_else(|| {
                    Error::validation(format!(
                        "filter operator `{key}` expects an array of filters, got {condition}"
                    ))
                })?;
                for sub_filter in sub_filters {
                    validate_filters(sub_filter)?;
                }
            }
            "$not" => {
                let Value::Object(sub_filter) = condition else {
                    return Err(Error::validation(format!(
                        "filter operator `$not` expects a filter object, got {condition}"
                    )));
                };
                validate_filter_object(sub_filter)?;
            }
            operator if operator.starts_with('$') => {
                return Err(unknown_operator(operator));
            }
            _field => validate_condition(condition)?,
        }
    }
    Ok(())
}
/// Validates a field condition's operator object, when it has one.
fn validate_condition(condition: &Value) -> Result<()> {
    let Value::Object(operators) = condition else {
        return Ok(());
    };
    if !operators.keys().any(|key| key.starts_with('$')) {
        // Implicit equality on a nested object: not an operator condition.
        return Ok(());
    }
    for operator in operators.keys() {
        if !KNOWN_OPERATORS.contains(&operator.as_str()) {
            return Err(unknown_operator(operator));
        }
    }
    Ok(())
}
/// The validation error for an operator this backend does not evaluate.
fn unknown_operator(operator: &str) -> Error {
    Error::validation(format!(
        "unknown vector filter operator `{operator}`; expected one of {}",
        KNOWN_OPERATORS.join(", ")
    ))
}
/// Builds `?start, ?start+1, ...` for `count` parameters.
fn placeholders(start: usize, count: usize) -> String {
    (start..start + count)
        .map(|index| format!("?{index}"))
        .collect::<Vec<_>>()
        .join(", ")
}
#[cfg(test)]
mod tests;
