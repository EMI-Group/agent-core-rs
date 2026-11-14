//! Live `Redis` document backend over the crate's RESP client.
//!
//! The Python `extensions/store` package has no document store (its `db/`
//! backend is `GaussDB`/`SQLAlchemy`), so this backend is a Rust-side extension of
//! the family: it implements the kernel
//! [`DocumentStore`](openjiuwen_core::foundation::store::DocumentStore) contract
//! on the key layout documented in [`super`]:
//!
//! - `openjiuwen:doc:collections` — every collection name,
//! - `openjiuwen:doc:ids:{collection}` — the ids stored in one collection,
//! - `openjiuwen:doc:doc:{collection}:{id}` — one serialized
//!   [`Document`](openjiuwen_core::foundation::store::Document).
//!
//! # Delegation
//!
//! `query`, `count` and `delete` load the collection into a throwaway
//! [`InMemoryDocumentStore`] and delegate to it; `delete` then removes the
//! documents the reference dropped. The kernel's `run_query` and matching
//! helpers are private to `openjiuwen-foundation`, so delegation is the only way
//! to keep the filter language, the ordering (`null` first ascending),
//! offset-then-limit paging, and the delete semantics byte-for-byte identical to
//! the behavioral reference. The cost is one read of the whole collection per
//! read or delete operation.
//!
//! # Key layout and limits
//!
//! Documents are addressed through the id set, never through `SCAN`, so ids that
//! contain glob metacharacters are safe. Collection names and ids are
//! percent-escaped before they become key components, so `a` + id `b:c` and
//! collection `a:b` + id `c` cannot collide. An empty collection has no id-set
//! key at all, because `Redis` has no empty sets; `SMEMBERS` on a missing key is
//! empty, so a freshly created collection reads as empty.
//!
//! There is no TTL: the [`DocumentStore`](openjiuwen_core::foundation::store::DocumentStore)
//! contract has no expiry knob, so an expiry is not invented here. Writes are
//! not transactional across keys, so an interrupted batch can leave the id set
//! and the document keys temporarily inconsistent; the store tolerates that on
//! read (a listed id whose value vanished is skipped). A single writer per
//! collection is assumed.
use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_core::foundation::store::{Document, DocumentStore, InMemoryDocumentStore, Query};
use std::collections::BTreeSet;
use std::fmt;
use std::sync::Arc;

use super::{
    collections_key, document_key, expect_ok, ids_key, validate_collection, RedisConnection,
    RedisStoreConfig, DOCUMENT_NAMESPACE,
};

/// A live `Redis` document store over the in-crate RESP client.
///
/// [`RedisDocumentStore::new`] only validates the configuration; the client is
/// opened on first use (or eagerly by [`RedisDocumentStore::connect`]). The
/// store is `Send + Sync`, so it can be shared behind an `Arc` like every kernel
/// [`DocumentStore`]. It is not `Clone` (a connection is shared, not
/// duplicated); use [`RedisDocumentStore::connect`] to obtain the shared
/// `Arc<dyn DocumentStore>`.
pub struct RedisDocumentStore {
    /// The shared connection layer.
    connection: RedisConnection,
}

impl fmt::Debug for RedisDocumentStore {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The connection redacts credentials, so nesting it is safe.
        formatter
            .debug_struct("RedisDocumentStore")
            .field("connection", &self.connection)
            .finish()
    }
}

impl RedisDocumentStore {
    /// Validates `config` and wraps it in a store, without any I/O.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::Validation`](openjiuwen_common::error::ErrorKind::Validation)
    /// when neither a `url` nor a non-empty `connection_args` is present, or
    /// when the URL does not use an accepted scheme.
    pub fn new(config: RedisStoreConfig) -> Result<Self> {
        Ok(Self {
            connection: RedisConnection::new(config)?,
        })
    }

    /// Validates `config`, connects eagerly, and returns the shared store.
    ///
    /// The connection is verified with `PING`, so a bad URL, an unreachable
    /// host, or a server that only answers with cluster redirections fails here
    /// instead of on the caller's first operation.
    ///
    /// # Errors
    ///
    /// Returns the validation errors of [`RedisDocumentStore::new`], followed by
    /// the client's connection error
    /// ([`ErrorKind::Transport`](openjiuwen_common::error::ErrorKind::Transport),
    /// [`ErrorKind::Timeout`](openjiuwen_common::error::ErrorKind::Timeout), or
    /// [`ErrorKind::Unsupported`](openjiuwen_common::error::ErrorKind::Unsupported)
    /// for TLS and cluster redirections).
    pub async fn connect(config: RedisStoreConfig) -> Result<Arc<dyn DocumentStore>> {
        let store = Self::new(config)?;
        store.ping().await?;
        Ok(Arc::new(store))
    }

    /// The normalized configuration this store was built from.
    #[must_use]
    pub fn config(&self) -> &RedisStoreConfig {
        self.connection.config()
    }

    /// Opens the connection (if needed) and verifies it with `PING`.
    ///
    /// # Errors
    ///
    /// Same as [`RedisDocumentStore::connect`] after validation.
    pub async fn ping(&self) -> Result<()> {
        self.connection.ping().await
    }

    /// The ids the collection's id set holds, in unspecified order.
    async fn stored_ids(&self, collection: &str) -> Result<Vec<String>> {
        self.connection
            .smembers(&ids_key(DOCUMENT_NAMESPACE, collection))
            .await
    }

    /// Loads `collection` into a throwaway reference store.
    ///
    /// A listed id whose value vanished is skipped: the id set can drift when a
    /// write is interrupted.
    async fn load(&self, collection: &str) -> Result<InMemoryDocumentStore> {
        let store = InMemoryDocumentStore::new();
        let ids = self.stored_ids(collection).await?;
        let keys: Vec<String> = ids
            .iter()
            .map(|id| document_key(DOCUMENT_NAMESPACE, collection, id))
            .collect();
        let payloads = self.connection.mget_bytes(&keys).await?;
        let mut documents = Vec::with_capacity(payloads.len());
        for (key, payload) in keys.iter().zip(payloads) {
            if let Some(payload) = payload {
                documents.push(parse_document(key, &payload)?);
            }
        }
        if !documents.is_empty() {
            store.upsert(collection, documents).await?;
        }
        Ok(store)
    }

    /// Writes `documents` and registers their ids in one `SET` batch.
    async fn persist(&self, collection: &str, documents: &[Document]) -> Result<()> {
        if documents.is_empty() {
            return Ok(());
        }
        let mut commands = Vec::with_capacity(documents.len());
        for document in documents {
            let key = document_key(DOCUMENT_NAMESPACE, collection, &document.id);
            let payload = serde_json::to_vec(document).map_err(Error::external)?;
            commands.push(vec![b"SET".to_vec(), key.into_bytes(), payload]);
        }
        for reply in self.connection.pipeline(&commands).await? {
            expect_ok(&reply, "SET")?;
        }
        let ids: Vec<String> = documents
            .iter()
            .map(|document| document.id.clone())
            .collect();
        self.connection
            .sadd(&ids_key(DOCUMENT_NAMESPACE, collection), &ids)
            .await
    }

    /// Removes `ids` from the collection's keys and id set.
    async fn remove(&self, collection: &str, ids: &[String]) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let keys: Vec<String> = ids
            .iter()
            .map(|id| document_key(DOCUMENT_NAMESPACE, collection, id))
            .collect();
        self.connection.delete_keys(&keys).await?;
        self.connection
            .srem(&ids_key(DOCUMENT_NAMESPACE, collection), ids)
            .await
    }

    /// Registers `collection` in the collection set.
    async fn register(&self, collection: &str) -> Result<()> {
        self.connection
            .sadd(
                &collections_key(DOCUMENT_NAMESPACE),
                &[collection.to_string()],
            )
            .await
    }
}

/// Parses one stored document, naming its key when the payload is corrupt.
fn parse_document(key: &str, payload: &[u8]) -> Result<Document> {
    serde_json::from_slice(payload).map_err(|error| {
        Error::transport(format!(
            "redis key `{key}` does not hold a valid document: {error}"
        ))
    })
}

#[async_trait]
impl DocumentStore for RedisDocumentStore {
    async fn create_collection(&self, collection: &str) -> Result<()> {
        validate_collection(collection)?;
        self.register(collection).await
    }

    async fn list_collections(&self) -> Result<Vec<String>> {
        let mut names = self
            .connection
            .smembers(&collections_key(DOCUMENT_NAMESPACE))
            .await?;
        names.sort();
        Ok(names)
    }

    async fn drop_collection(&self, collection: &str) -> Result<()> {
        validate_collection(collection)?;
        let ids = self.stored_ids(collection).await?;
        self.remove(collection, &ids).await?;
        self.connection
            .srem(
                &collections_key(DOCUMENT_NAMESPACE),
                &[collection.to_string()],
            )
            .await
    }

    async fn upsert(&self, collection: &str, documents: Vec<Document>) -> Result<usize> {
        validate_collection(collection)?;
        let written = documents.len();
        self.persist(collection, &documents).await?;
        // An empty upsert still creates the collection, like the reference store.
        self.register(collection).await?;
        Ok(written)
    }

    async fn get(&self, collection: &str, id: &str) -> Result<Option<Document>> {
        validate_collection(collection)?;
        let key = document_key(DOCUMENT_NAMESPACE, collection, id);
        match self.connection.get_bytes(&key).await? {
            Some(payload) => parse_document(&key, &payload).map(Some),
            None => Ok(None),
        }
    }

    async fn query(&self, collection: &str, query: &Query) -> Result<Vec<Document>> {
        validate_collection(collection)?;
        let store = self.load(collection).await?;
        store.query(collection, query).await
    }

    async fn delete(&self, collection: &str, query: &Query) -> Result<usize> {
        validate_collection(collection)?;
        let store = self.load(collection).await?;
        let before = self.stored_ids(collection).await?;
        let deleted = store.delete(collection, query).await?;
        if deleted > 0 {
            // Everything the reference kept stays; the difference is what to
            // remove from Redis. `delete` ignores paging and ordering, and so
            // does this diff.
            let remaining: BTreeSet<String> = store
                .query(collection, &Query::default())
                .await?
                .into_iter()
                .map(|document| document.id)
                .collect();
            let doomed: Vec<String> = before
                .into_iter()
                .filter(|id| !remaining.contains(id))
                .collect();
            self.remove(collection, &doomed).await?;
        }
        Ok(deleted)
    }

    async fn count(&self, collection: &str, query: &Query) -> Result<usize> {
        validate_collection(collection)?;
        let store = self.load(collection).await?;
        store.count(collection, query).await
    }
}
