//! The [`LocalCodingMemory`] builder: the open sequence, the scope, and the optional
//! embedding model.
//!
//! [`LocalCodingMemory::open`] and [`LocalCodingMemory::open_in_scope`] are thin
//! delegations to this builder, so the keyword-only open sequence is identical whichever
//! entry point a caller uses. The builder is the only place local semantic search is
//! configured: without [`LocalCodingMemoryBuilder::embedding`] the store keeps its
//! keyword-only behaviour and needs no model and no network.
use std::path::PathBuf;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use openjiuwen_retrieval::embedding::Embedding;
use tokio::sync::Mutex;

use openjiuwen_common::error::{Error, Result};

use crate::config::CodingMemoryConfig;
use crate::lite::embed::{ChunkingConfig, EmbeddingCache};
use crate::lite::lock::FileLock;
use crate::record::MemoryScope;

use super::layout::{load_document, write_document, FileStamp, LOCK_FILE, MEMORY_FILE, STATE_DIR};
use super::state::{Inner, LocalCodingMemory, LocalState};

#[cfg(feature = "sqlite")]
use super::layout::INDEX_FILE;

#[cfg(feature = "sqlite")]
use crate::lite::fts::FtsIndex;

/// Configures and opens a [`LocalCodingMemory`].
///
/// ```
/// use std::sync::Arc;
///
/// use openjiuwen_memory::config::CodingMemoryConfig;
/// use openjiuwen_memory::lite::local::LocalCodingMemory;
/// use openjiuwen_retrieval::embedding::HashEmbedding;
///
/// let builder = LocalCodingMemory::builder("./workspace", CodingMemoryConfig::default())
///     .embedding(Arc::new(HashEmbedding::default()));
/// assert!(format!("{builder:?}").contains("has_embedding: true"));
/// ```
#[derive(Clone)]
pub struct LocalCodingMemoryBuilder {
    /// The directory that will hold `MEMORY.md`.
    root: PathBuf,
    /// The coding-memory settings of the store.
    config: CodingMemoryConfig,
    /// The scope every fact of the store belongs to.
    scope: MemoryScope,
    /// The embedding model backing semantic search, when one is configured.
    embedding: Option<Arc<dyn Embedding>>,
    /// The chunk shape of the embedding index.
    chunking: ChunkingConfig,
}

impl std::fmt::Debug for LocalCodingMemoryBuilder {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LocalCodingMemoryBuilder")
            .field("root", &self.root)
            .field("config", &self.config)
            .field("scope", &self.scope)
            .field("has_embedding", &self.embedding.is_some())
            .field("chunking", &self.chunking)
            .finish()
    }
}

impl LocalCodingMemoryBuilder {
    /// Creates a builder for the store rooted at `root`, in the default scope, with
    /// `config`, no embedding model, and the default chunking.
    #[must_use]
    pub fn new(root: impl Into<PathBuf>, config: CodingMemoryConfig) -> Self {
        Self {
            root: root.into(),
            config,
            scope: MemoryScope::default(),
            embedding: None,
            chunking: ChunkingConfig::default(),
        }
    }

    /// Sets the scope every fact of the store belongs to.
    #[must_use]
    pub fn scope(mut self, scope: MemoryScope) -> Self {
        self.scope = scope;
        self
    }

    /// Configures the embedding model that backs semantic recall.
    ///
    /// The model is only called when chunks have to be embedded (a cache miss) or a
    /// query is embedded; opening a store and searching it in keyword mode make no model
    /// call.
    #[must_use]
    pub fn embedding(mut self, embedding: Arc<dyn Embedding>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Sets the chunk shape of the embedding index (the default is the crate default).
    #[must_use]
    pub fn chunking(mut self, chunking: ChunkingConfig) -> Self {
        self.chunking = chunking;
        self
    }

    /// Opens (or initializes) the store.
    ///
    /// Creates the root and `.memory/` directories, takes the cross-process lock, loads
    /// (or writes) the `MEMORY.md` document, opens the derived FTS index, and rebuilds
    /// the derived indices from the document. With an embedding configured the chunk
    /// index is built once here, so the first semantic query needs no re-indexing pass.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an invalid scope or chunking config, the
    /// filesystem or lock error when the store cannot be initialized, and the index or
    /// embedding error when the derived indices cannot be rebuilt.
    pub async fn open(self) -> Result<LocalCodingMemory> {
        self.scope.validate()?;
        self.chunking.validate()?;
        let Self {
            root,
            config,
            scope,
            embedding,
            chunking,
        } = self;
        let state_dir = root.join(STATE_DIR);
        tokio::fs::create_dir_all(&state_dir)
            .await
            .map_err(|error| {
                Error::external(error).with_context(format!(
                    "creating the state directory {}",
                    state_dir.display()
                ))
            })?;
        let memory_path = root.join(MEMORY_FILE);
        let lock_path = state_dir.join(LOCK_FILE);
        let guard = FileLock::acquire(&lock_path).await?;
        let (document, stamp) = load_document(&memory_path, &scope).await?;
        if stamp.is_none() {
            write_document(&memory_path, &document).await?;
        }
        drop(guard);
        let stamp = FileStamp::of(&memory_path).await?;
        #[cfg(feature = "sqlite")]
        let fts = FtsIndex::open_path(&state_dir.join(INDEX_FILE)).await?;
        // A persistent cache is only worth opening when vectors are actually produced;
        // it shares the derived index database, so vectors survive a reopen.
        #[cfg(feature = "sqlite")]
        let cache = match embedding {
            Some(_) => EmbeddingCache::sqlite(fts.pool().clone()).await?,
            None => EmbeddingCache::in_memory(),
        };
        #[cfg(not(feature = "sqlite"))]
        let cache = EmbeddingCache::in_memory();
        let inner = Arc::new(Inner {
            root,
            memory_path,
            lock_path,
            config,
            scope,
            state: Mutex::new(LocalState { document, stamp }),
            reloads: AtomicU64::new(0),
            embedding,
            chunking,
            cache,
            embeddings: Mutex::new(None),
        });
        let store = LocalCodingMemory {
            #[cfg(feature = "sqlite")]
            fts,
            inner,
            watch: Arc::new(Mutex::new(None)),
        };
        let _indexed = store.rebuild_index().await?;
        if store.has_embedding() {
            let _stats = store.refresh_embeddings().await?;
        }
        Ok(store)
    }
}
