//! Live `Redis` vector backend over the crate's RESP client.
//!
//! The Python `extensions/store` package ships only the Elasticsearch vector
//! store, so this backend is a Rust-side extension of the family: it implements
//! the kernel [`VectorStore`](openjiuwen_core::foundation::store::VectorStore)
//! contract on the key layout documented in [`super`]:
//!
//! - `openjiuwen:vec:collections` — every collection name,
//! - `openjiuwen:vec:schema:{collection}` — the serialized `CollectionSchema`
//!   (it round-trips exactly, including `is_primary`, `auto_id`, `max_length`,
//!   `default_value` and `dim`),
//! - `openjiuwen:vec:ids:{collection}` — the primary keys of the collection,
//! - `openjiuwen:vec:doc:{collection}:{id}` — one JSON document.
//!
//! # Delegation
//!
//! No `RediSearch` module is available offline (and adding one is not an option
//! for a self-contained build), so similarity ranking is brute force on the
//! client. Every operation hydrates a throwaway
//! [`InMemoryVectorStore`](openjiuwen_core::foundation::store::InMemoryVectorStore)
//! from `Redis` and delegates to it, then persists the effect:
//!
//! - `add_docs` lets the reference validate the batch, fill `default_value`s and
//!   mint `auto_id`s, then reads the prepared documents back with
//!   `InMemoryVectorStore::search_with_metric` and rewrites the collection with
//!   them. An add therefore rewrites every document of the collection so the
//!   persisted view is the reference-prepared one; a rejected batch persists
//!   nothing and surfaces the reference error unchanged (duplicate primary key,
//!   missing or mistyped field, dimension mismatch).
//! - `search` returns the reference hits unchanged, so score normalization,
//!   score-descending order, the primary-key tie-break and `top_k` truncation
//!   cannot drift.
//! - `delete_docs_by_ids`/`delete_docs_by_filters` delegate the deletion and
//!   then remove exactly the ids the reference dropped.
//!
//! # Metric and limits
//!
//! The metric is a store-level choice ([`RedisVectorStore::metric`], cosine by
//! default) because `CollectionSchema` carries no metric field. Collections are
//! addressed through their id set, never through `SCAN`, and collection names and
//! ids are percent-escaped before they become key components. `Redis` has no
//! empty sets, so a collection created without documents has no id-set key and
//! reads as empty. There is no cross-process locking (one writer per collection
//! is assumed) and no TTL (neither kernel contract has an expiry knob).
use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_core::foundation::store::{
    CollectionSchema, FieldSchema, InMemoryVectorStore, Metric, VectorQuery, VectorSearchResult,
    VectorStore,
};
use serde_json::{Map, Value};
use std::collections::BTreeSet;
use std::fmt;
use std::sync::Arc;

use super::{
    collections_key, document_key, expect_ok, ids_key, schema_key, validate_collection,
    RedisConnection, RedisStoreConfig, VECTOR_NAMESPACE,
};

/// A live `Redis` vector store over the in-crate RESP client.
///
/// [`RedisVectorStore::new`] only validates the configuration; the client is
/// opened on first use (or eagerly by [`RedisVectorStore::connect`]). The store
/// is `Send + Sync`, so it can be shared behind an `Arc` like every kernel
/// [`VectorStore`]. It is not `Clone` (a connection is shared, not duplicated);
/// use [`RedisVectorStore::connect`] to obtain the shared `Arc<dyn VectorStore>`.
pub struct RedisVectorStore {
    /// The shared connection layer.
    connection: RedisConnection,
    /// The similarity metric every search uses.
    metric: Metric,
}

impl fmt::Debug for RedisVectorStore {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The connection redacts credentials, so nesting it is safe.
        formatter
            .debug_struct("RedisVectorStore")
            .field("connection", &self.connection)
            .field("metric", &self.metric)
            .finish()
    }
}

impl RedisVectorStore {
    /// Validates `config` and wraps it in a store, without any I/O.
    ///
    /// The metric defaults to [`Metric::Cosine`]; chain
    /// [`RedisVectorStore::with_metric`] to change it.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::Validation`](openjiuwen_common::error::ErrorKind::Validation)
    /// when neither a `url` nor a non-empty `connection_args` is present, or
    /// when the URL does not use an accepted scheme.
    pub fn new(config: RedisStoreConfig) -> Result<Self> {
        Ok(Self {
            connection: RedisConnection::new(config)?,
            metric: Metric::default(),
        })
    }

    /// Replaces the similarity metric, returning the updated store.
    #[must_use]
    pub fn with_metric(mut self, metric: Metric) -> Self {
        self.metric = metric;
        self
    }

    /// Validates `config`, connects eagerly, and returns the shared store.
    ///
    /// The connection is verified with `PING`, so a bad URL, an unreachable
    /// host, or a server that only answers with cluster redirections fails here
    /// instead of on the caller's first operation.
    ///
    /// # Errors
    ///
    /// Returns the validation errors of [`RedisVectorStore::new`], followed by
    /// the client's connection error
    /// ([`ErrorKind::Transport`](openjiuwen_common::error::ErrorKind::Transport),
    /// [`ErrorKind::Timeout`](openjiuwen_common::error::ErrorKind::Timeout), or
    /// [`ErrorKind::Unsupported`](openjiuwen_common::error::ErrorKind::Unsupported)
    /// for TLS and cluster redirections).
    pub async fn connect(config: RedisStoreConfig) -> Result<Arc<dyn VectorStore>> {
        let store = Self::new(config)?;
        store.ping().await?;
        Ok(Arc::new(store))
    }

    /// The normalized configuration this store was built from.
    #[must_use]
    pub fn config(&self) -> &RedisStoreConfig {
        self.connection.config()
    }

    /// The similarity metric every search uses.
    #[must_use]
    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// Opens the connection (if needed) and verifies it with `PING`.
    ///
    /// # Errors
    ///
    /// Same as [`RedisVectorStore::connect`] after validation.
    pub async fn ping(&self) -> Result<()> {
        self.connection.ping().await
    }

    /// The primary keys the collection's id set holds.
    async fn stored_ids(&self, collection: &str) -> Result<Vec<String>> {
        self.connection
            .smembers(&ids_key(VECTOR_NAMESPACE, collection))
            .await
    }

    /// The stored schema of `collection`, or `NotFound` when it is absent.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::NotFound`](openjiuwen_common::error::ErrorKind::NotFound)
    /// for an unknown collection, plus the validation and transport errors of
    /// [`RedisVectorStore::get_schema`].
    async fn schema_or_not_found(&self, collection: &str) -> Result<CollectionSchema> {
        self.get_schema(collection)
            .await?
            .ok_or_else(|| Error::not_found(format!("vector collection {collection:?} not found")))
    }

    /// Loads `collection` into a throwaway reference store.
    ///
    /// A listed id whose value vanished is skipped: the id set can drift when a
    /// write is interrupted.
    async fn load(&self, collection: &str) -> Result<(CollectionSchema, InMemoryVectorStore)> {
        let schema = self.schema_or_not_found(collection).await?;
        let store = InMemoryVectorStore::new();
        store.create_collection(collection, schema.clone()).await?;
        let ids = self.stored_ids(collection).await?;
        let keys: Vec<String> = ids
            .iter()
            .map(|id| document_key(VECTOR_NAMESPACE, collection, id))
            .collect();
        let payloads = self.connection.mget_bytes(&keys).await?;
        let mut documents = Vec::with_capacity(payloads.len());
        for (key, payload) in keys.iter().zip(payloads) {
            if let Some(payload) = payload {
                documents.push(parse_document(key, &payload)?);
            }
        }
        if !documents.is_empty() {
            store.add_docs(collection, documents).await?;
        }
        Ok((schema, store))
    }

    /// Writes `documents` and registers their primary keys in one `SET` batch.
    async fn persist(
        &self,
        collection: &str,
        schema: &CollectionSchema,
        documents: &[Map<String, Value>],
    ) -> Result<()> {
        if documents.is_empty() {
            return Ok(());
        }
        let primary = schema
            .primary_key()
            .ok_or_else(|| Error::validation("collection schema declares no primary key field"))?;
        let mut commands = Vec::with_capacity(documents.len());
        let mut ids = Vec::with_capacity(documents.len());
        for fields in documents {
            let id = primary_key_string(primary, fields)?;
            let payload = serde_json::to_vec(fields).map_err(Error::external)?;
            let key = document_key(VECTOR_NAMESPACE, collection, &id);
            commands.push(vec![b"SET".to_vec(), key.into_bytes(), payload]);
            ids.push(id);
        }
        for reply in self.connection.pipeline(&commands).await? {
            expect_ok(&reply, "SET")?;
        }
        self.connection
            .sadd(&ids_key(VECTOR_NAMESPACE, collection), &ids)
            .await
    }

    /// Removes `ids` from the collection's keys and id set.
    async fn remove(&self, collection: &str, ids: &[String]) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let keys: Vec<String> = ids
            .iter()
            .map(|id| document_key(VECTOR_NAMESPACE, collection, id))
            .collect();
        self.connection.delete_keys(&keys).await?;
        self.connection
            .srem(&ids_key(VECTOR_NAMESPACE, collection), ids)
            .await
    }

    /// Removes every id of `before` that the reference store no longer holds.
    async fn remove_dropped(
        &self,
        collection: &str,
        schema: &CollectionSchema,
        store: &InMemoryVectorStore,
        before: &[String],
    ) -> Result<()> {
        let remaining: BTreeSet<String> = remaining_ids(store, collection, schema, before.len())?
            .into_iter()
            .collect();
        let doomed: Vec<String> = before
            .iter()
            .filter(|id| !remaining.contains(*id))
            .cloned()
            .collect();
        self.remove(collection, &doomed).await
    }
}

/// Parses one stored document, naming its key when the payload is corrupt.
fn parse_document(key: &str, payload: &[u8]) -> Result<Map<String, Value>> {
    serde_json::from_slice(payload).map_err(|error| {
        Error::transport(format!(
            "redis key `{key}` does not hold a valid vector document: {error}"
        ))
    })
}

/// The string form of a document's primary key, as used in keys and id sets.
///
/// The reference store keeps a numeric primary key as a number in the document
/// fields but addresses it by its string form, so both are accepted here.
fn primary_key_string(primary: &FieldSchema, fields: &Map<String, Value>) -> Result<String> {
    match fields.get(&primary.name) {
        Some(Value::String(id)) => Ok(id.clone()),
        Some(value) if value.is_i64() || value.is_u64() => Ok(value.to_string()),
        Some(other) => Err(Error::validation(format!(
            "primary key field {:?} of a stored document is not a string or integer: {other}",
            primary.name
        ))),
        None => Err(Error::validation(format!(
            "a stored document is missing its primary key field {:?}",
            primary.name
        ))),
    }
}

/// The primary keys the reference store still holds after a mutation.
///
/// The reference has no id listing, so `count` documents are read back through a
/// zero-vector search; a zero query scores every document (ties broken by
/// primary key), which is enough to enumerate them.
fn remaining_ids(
    store: &InMemoryVectorStore,
    collection: &str,
    schema: &CollectionSchema,
    count: usize,
) -> Result<Vec<String>> {
    if count == 0 {
        return Ok(Vec::new());
    }
    let dim = schema
        .default_vector_field()
        .and_then(|field| field.dim)
        .unwrap_or(0);
    let primary = schema
        .primary_key()
        .ok_or_else(|| Error::validation("collection schema declares no primary key field"))?;
    let query = VectorQuery {
        query_vector: vec![0.0; dim],
        vector_field: None,
        top_k: count,
        filters: Value::Null,
    };
    let hits = store.search_with_metric(collection, &query, Metric::Cosine)?;
    hits.iter()
        .map(|hit| primary_key_string(primary, &hit.fields))
        .collect()
}

#[async_trait]
impl VectorStore for RedisVectorStore {
    async fn create_collection(&self, collection: &str, schema: CollectionSchema) -> Result<()> {
        schema.validate()?;
        validate_collection(collection)?;
        if self.collection_exists(collection).await? {
            return Err(Error::validation(format!(
                "vector collection {collection:?} already exists"
            )));
        }
        let payload = serde_json::to_vec(&schema).map_err(Error::external)?;
        self.connection
            .set_bytes(&schema_key(collection), &payload)
            .await?;
        self.connection
            .sadd(
                &collections_key(VECTOR_NAMESPACE),
                &[collection.to_string()],
            )
            .await
    }

    async fn delete_collection(&self, collection: &str) -> Result<()> {
        validate_collection(collection)?;
        let ids = self.stored_ids(collection).await?;
        let mut keys = vec![schema_key(collection)];
        keys.extend(
            ids.iter()
                .map(|id| document_key(VECTOR_NAMESPACE, collection, id)),
        );
        self.connection.delete_keys(&keys).await?;
        self.connection
            .srem(&ids_key(VECTOR_NAMESPACE, collection), &ids)
            .await?;
        self.connection
            .srem(
                &collections_key(VECTOR_NAMESPACE),
                &[collection.to_string()],
            )
            .await
    }

    async fn collection_exists(&self, collection: &str) -> Result<bool> {
        validate_collection(collection)?;
        let reply = self
            .connection
            .run(&[b"EXISTS", schema_key(collection).as_bytes()])
            .await?;
        Ok(reply.as_integer()? > 0)
    }

    async fn get_schema(&self, collection: &str) -> Result<Option<CollectionSchema>> {
        validate_collection(collection)?;
        let key = schema_key(collection);
        let Some(payload) = self.connection.get_bytes(&key).await? else {
            return Ok(None);
        };
        serde_json::from_slice(&payload).map(Some).map_err(|error| {
            Error::transport(format!(
                "redis key `{key}` does not hold a valid collection schema: {error}"
            ))
        })
    }

    async fn add_docs(&self, collection: &str, docs: Vec<Map<String, Value>>) -> Result<usize> {
        validate_collection(collection)?;
        let (schema, store) = self.load(collection).await?;
        let existing = self.stored_ids(collection).await?.len();
        let added = store.add_docs(collection, docs).await?;
        if added == 0 {
            return Ok(0);
        }
        // Read the prepared documents back: validation minted `auto_id`s and
        // filled `default_value`s, so the persisted view must be the reference's.
        let dim = schema
            .default_vector_field()
            .and_then(|field| field.dim)
            .unwrap_or(0);
        let query = VectorQuery {
            query_vector: vec![0.0; dim],
            vector_field: None,
            top_k: existing.saturating_add(added),
            filters: Value::Null,
        };
        let prepared: Vec<Map<String, Value>> = store
            .search_with_metric(collection, &query, self.metric)?
            .into_iter()
            .map(|hit| hit.fields)
            .collect();
        self.persist(collection, &schema, &prepared).await?;
        Ok(added)
    }

    async fn search(
        &self,
        collection: &str,
        query: &VectorQuery,
    ) -> Result<Vec<VectorSearchResult>> {
        validate_collection(collection)?;
        let (_schema, store) = self.load(collection).await?;
        store.search_with_metric(collection, query, self.metric)
    }

    async fn delete_docs_by_ids(&self, collection: &str, ids: &[String]) -> Result<usize> {
        validate_collection(collection)?;
        let (schema, store) = self.load(collection).await?;
        let before = self.stored_ids(collection).await?;
        let deleted = store.delete_docs_by_ids(collection, ids).await?;
        if deleted > 0 {
            self.remove_dropped(collection, &schema, &store, &before)
                .await?;
        }
        Ok(deleted)
    }

    async fn delete_docs_by_filters(&self, collection: &str, filters: &Value) -> Result<usize> {
        validate_collection(collection)?;
        let (schema, store) = self.load(collection).await?;
        let before = self.stored_ids(collection).await?;
        let deleted = store.delete_docs_by_filters(collection, filters).await?;
        if deleted > 0 {
            self.remove_dropped(collection, &schema, &store, &before)
                .await?;
        }
        Ok(deleted)
    }

    async fn list_collection_names(&self) -> Result<Vec<String>> {
        let mut names = self
            .connection
            .smembers(&collections_key(VECTOR_NAMESPACE))
            .await?;
        names.sort();
        Ok(names)
    }
}
