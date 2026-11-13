//! [`MemoryStores`] and persistence for records, embeddings, variables, and messages.
//!
//! The store module keeps the logical layout of the Python `openjiuwen.core.memory`
//! package on top of the `openjiuwen-foundation` storage traits:
//!
//! - [`keys`] — KV key and vector collection naming (Python `UMD/...` layout).
//! - `MemoryStorage` — record/embedding/variable/scope-config persistence.
//! - `MessageStore` — conversation history in the `messages` document collection.
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

pub mod keys;
pub mod migration;

mod json;
mod legacy;
mod messages;
mod storage;

pub(crate) use messages::MessageStore;
pub use messages::StoredMessage;
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
