//! [`MemoryStores`] and persistence for records, embeddings, variables, and messages.
//!
//! The store module keeps the logical layout of the Python `openjiuwen.core.memory`
//! package on top of the `openjiuwen-foundation` storage traits:
//!
//! - [`keys`] — KV key and vector collection naming (Python `UMD/...` layout).
//! - [`prefix`] — the runtime key-prefix registry over the logical memory families.
//! - [`backup`] — backup and restore of the memory key-value keyspace.
//! - [`kv_lock`] — the distributed KV lock seam (named `kv_lock` because `lite::lock`
//!   already owns the crate-root `lock` name).
//! - [`mapping`] — persisted scope↔user mapping keys.
//! - `MemoryStorage` — record/embedding/variable/scope-config persistence.
//! - `MessageStore` — conversation history in the `messages` document collection, read
//!   either as the newest N rows or as keyset pages resuming from a [`MessageCursor`].
//! - [`MemoryStores`] — the bundle of concrete backends every other module shares.
//!
//! A missing key or collection is never an error: reads yield `None`, deletes report
//! `false`/`0`, and searches return an empty vector. Store errors otherwise propagate
//! unchanged, annotated with [`openjiuwen_common::error::Error::with_context`] only
//! where the operation is not obvious from the call site.

use std::path::Path;
use std::sync::Arc;

#[cfg(feature = "sqlite")]
use openjiuwen_common::error::Error;
use openjiuwen_common::error::Result;
use openjiuwen_foundation::store::{
    DocumentStore, GraphStore, InMemoryDocumentStore, InMemoryGraphStore, InMemoryKvStore,
    InMemoryVectorStore, KvStore, LocalFileDocumentStore, LocalFileGraphStore, LocalFileKvStore,
    LocalFileVectorStore, VectorStore,
};
#[cfg(feature = "sqlite")]
use openjiuwen_foundation::store::{SqliteDocumentStore, SqliteKvStore};

pub mod backup;
pub mod keys;
pub mod kv_lock;
pub mod mapping;
pub mod migration;
pub mod prefix;

mod json;
mod legacy;
mod messages;
mod storage;

pub(crate) use messages::MessageStore;
pub use messages::{MessageCursor, MessagePage, StoredMessage};
pub(crate) use storage::MemoryStorage;

/// The concrete backends shared by the engine, facade, graph memory, and lite memory.
///
/// All fields are `Arc`s, so the bundle is `Clone` and cheap to share; a local-file,
/// SQLite, or future vendor store is injected here rather than selected by a global.
#[derive(Clone)]
pub struct MemoryStores {
    /// Key-value store holding records, variables, and scope configuration.
    pub kv: Arc<dyn KvStore>,
    /// Vector store holding one embedding collection per `(user, scope, kind)`.
    pub vector: Arc<dyn VectorStore>,
    /// Document store holding conversation message history.
    pub document: Arc<dyn DocumentStore>,
    /// Graph store holding entities and relations, when graph memory is enabled.
    pub graph: Option<Arc<dyn GraphStore>>,
}

impl MemoryStores {
    /// Builds ephemeral in-process stores, including an in-memory graph store.
    #[must_use]
    pub fn in_memory() -> Self {
        Self {
            kv: Arc::new(InMemoryKvStore::new()),
            vector: Arc::new(InMemoryVectorStore::new()),
            document: Arc::new(InMemoryDocumentStore::new()),
            graph: Some(Arc::new(InMemoryGraphStore::new())),
        }
    }

    /// Builds file-backed stores rooted at `root`.
    ///
    /// Key-value data lives in `<root>/kv`, documents in `<root>/documents`, vectors in
    /// `<root>/vector`, and graph data in `<root>/graph`; every directory is created
    /// when it does not exist yet.
    pub async fn local_file(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref();
        Ok(Self {
            kv: Arc::new(LocalFileKvStore::new(root.join("kv")).await?),
            vector: Arc::new(LocalFileVectorStore::new(root.join("vector")).await?),
            document: Arc::new(LocalFileDocumentStore::new(root.join("documents")).await?),
            graph: Some(Arc::new(
                LocalFileGraphStore::new(root.join("graph")).await?,
            )),
        })
    }

    /// Builds SQLite-backed key-value/document stores at `path`.
    ///
    /// The SQLite parent directory is created first; vectors live in the local-file
    /// store under `<path>.vectors` and graph data under `<path>.graph`, because the
    /// foundation has no SQLite vector/graph backend. Reopening the same path observes
    /// every committed record and message.
    #[cfg(feature = "sqlite")]
    pub async fn sqlite(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                tokio::fs::create_dir_all(parent).await.map_err(|error| {
                    Error::external(error).with_context(format!(
                        "creating the memory SQLite directory `{}`",
                        parent.display()
                    ))
                })?;
            }
        }
        let url = format!("sqlite:{}", path.display());
        let vector_root = format!("{}.vectors", path.display());
        let graph_root = format!("{}.graph", path.display());
        Ok(Self {
            kv: Arc::new(SqliteKvStore::connect(&url).await?),
            vector: Arc::new(LocalFileVectorStore::new(&vector_root).await?),
            document: Arc::new(SqliteDocumentStore::connect(&url).await?),
            graph: Some(Arc::new(LocalFileGraphStore::new(&graph_root).await?)),
        })
    }

    /// Builds SQLite-backed stores at `path` after applying the schema migrations.
    ///
    /// [`migration::migrate_sqlite`] runs first, so the version table and every builtin
    /// DDL step are in place before the backends open the database. Migrations are
    /// forward-only and idempotent, so reopening an already migrated path is a no-op.
    /// The migration error keeps its kind and is annotated with `path`.
    ///
    /// # Errors
    ///
    /// Returns the migration error annotated with the database path, or the
    /// [`MemoryStores::sqlite`] error when the backends cannot be opened.
    #[cfg(feature = "sqlite")]
    pub async fn sqlite_migrated(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        migration::migrate_sqlite(path).await.map_err(|error| {
            error.with_context(format!(
                "migrating the memory SQLite database at `{}`",
                path.display()
            ))
        })?;
        Self::sqlite(path).await
    }

    /// Builds a bundle from explicit stores; graph memory starts disabled.
    #[must_use]
    pub fn from_parts(
        kv: Arc<dyn KvStore>,
        vector: Arc<dyn VectorStore>,
        document: Arc<dyn DocumentStore>,
    ) -> Self {
        Self {
            kv,
            vector,
            document,
            graph: None,
        }
    }

    /// Returns the bundle with `graph` attached.
    #[must_use]
    pub fn with_graph(mut self, graph: Arc<dyn GraphStore>) -> Self {
        self.graph = Some(graph);
        self
    }
}

/// Test doubles for the store entry points that must react to a failing backend.
///
/// Compiled for tests only: a real store never fabricates these shapes, but the variable
/// read and the backup restore both have to tell a missing entry from a broken one, so
/// their tests need a store that can return each error shape on demand.
#[cfg(test)]
pub(crate) mod test_support {
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use async_trait::async_trait;
    use openjiuwen_common::error::{Error, Result};
    use openjiuwen_foundation::store::{InMemoryKvStore, KvPipeline, KvStore, KvValue};

    /// An in-memory KV store that can fail a chosen read or write.
    pub(crate) struct FaultyKvStore {
        /// The backing store serving every non-injected operation.
        inner: InMemoryKvStore,
        /// When set, every `get` fails with `Error::Validation`.
        fail_reads: bool,
        /// When set, the `set` call with this 1-based index and every later one fails.
        fail_writes_from: Option<usize>,
        /// How many `set` calls have happened so far.
        writes: AtomicUsize,
    }

    impl FaultyKvStore {
        /// A store that reads and writes normally.
        pub(crate) fn new() -> Self {
            Self {
                inner: InMemoryKvStore::new(),
                fail_reads: false,
                fail_writes_from: None,
                writes: AtomicUsize::new(0),
            }
        }

        /// A store whose every `get` fails with `Error::Validation`, the shape a backend
        /// returns for a corrupt stored envelope.
        pub(crate) fn failing_reads() -> Self {
            Self {
                fail_reads: true,
                ..Self::new()
            }
        }

        /// A store whose `set` fails from the `writes`-th call onwards (1-based).
        pub(crate) fn failing_writes_from(writes: usize) -> Self {
            Self {
                fail_writes_from: Some(writes),
                ..Self::new()
            }
        }

        /// How many `set` calls reached the backing store.
        pub(crate) fn write_count(&self) -> usize {
            self.writes.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl KvStore for FaultyKvStore {
        async fn set(&self, key: &str, value: KvValue) -> Result<()> {
            let write = self.writes.fetch_add(1, Ordering::SeqCst) + 1;
            if self.fail_writes_from.is_some_and(|limit| write >= limit) {
                return Err(Error::validation(format!(
                    "injected write failure for `{key}`"
                )));
            }
            self.inner.set(key, value).await
        }

        async fn get(&self, key: &str) -> Result<Option<KvValue>> {
            if self.fail_reads {
                return Err(Error::validation(format!(
                    "injected read failure for `{key}`"
                )));
            }
            self.inner.get(key).await
        }

        async fn exists(&self, key: &str) -> Result<bool> {
            self.inner.exists(key).await
        }

        async fn delete(&self, key: &str) -> Result<()> {
            self.inner.delete(key).await
        }

        async fn exclusive_set(
            &self,
            key: &str,
            value: KvValue,
            expiry: Option<Duration>,
        ) -> Result<bool> {
            self.inner.exclusive_set(key, value, expiry).await
        }

        async fn get_by_prefix(&self, prefix: &str) -> Result<BTreeMap<String, KvValue>> {
            self.inner.get_by_prefix(prefix).await
        }

        async fn delete_by_prefix(&self, prefix: &str) -> Result<usize> {
            self.inner.delete_by_prefix(prefix).await
        }

        async fn mget(&self, keys: &[String]) -> Result<Vec<Option<KvValue>>> {
            self.inner.mget(keys).await
        }

        async fn batch_delete(&self, keys: &[String]) -> Result<usize> {
            self.inner.batch_delete(keys).await
        }

        fn pipeline(&self) -> Box<dyn KvPipeline + '_> {
            self.inner.pipeline()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn in_memory_includes_a_graph_store() {
        let stores = MemoryStores::in_memory();
        assert!(stores.graph.is_some());
    }

    #[test]
    fn from_parts_has_no_graph_and_with_graph_attaches_one() {
        let stores = MemoryStores::from_parts(
            Arc::new(InMemoryKvStore::new()),
            Arc::new(InMemoryVectorStore::new()),
            Arc::new(InMemoryDocumentStore::new()),
        );
        assert!(stores.graph.is_none());
        let stores = stores.with_graph(Arc::new(InMemoryGraphStore::new()));
        assert!(stores.graph.is_some());
        let cloned = stores.clone();
        assert!(Arc::ptr_eq(&cloned.kv, &stores.kv));
    }
}
