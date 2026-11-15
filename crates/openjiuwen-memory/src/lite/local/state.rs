//! The shared store state and the [`LocalCodingMemory`] handle: construction, cheap
//! accessors, and the parsed-document snapshot accessor.
//!
//! The mutation/read API lives in `crate::lite::local::store`, recall in
//! `crate::lite::local::search`, the embedding index in `crate::lite::local::embedding`,
//! configuration in `crate::lite::local::builder`, and the watcher lifecycle in
//! `crate::lite::local::watcher`; all of them are additional inherent `impl` blocks on the
//! same type and reach the state declared here through `pub(super)` items.
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use openjiuwen_retrieval::embedding::Embedding;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;

use openjiuwen_common::error::Result;

use crate::config::CodingMemoryConfig;
use crate::lite::embed::{ChunkingConfig, EmbeddingCache, EmbeddingIndex};
use crate::lite::facts::{DirStamp, FactFiles};
use crate::lite::markdown::MarkdownMemory;
use crate::lite::sync::IntervalScheduler;
use crate::lite::watch::MemoryWatcher;
use crate::record::MemoryScope;

use super::builder::LocalCodingMemoryBuilder;
use super::layout::FileStamp;
use super::sync_mode::IntervalTask;

#[cfg(feature = "sqlite")]
use crate::lite::fts::FtsIndex;

/// The cached document plus the stamps it was validated against.
///
/// In the default layout only [`LocalState::stamp`] (the `MEMORY.md` stamp) is used. With
/// the per-fact file layout enabled, `state` also records the stamps of the two rendered
/// index documents ([`LocalState::index_stamps`], indexed in
/// [`IndexLevel::ALL`](crate::lite::facts::IndexLevel::ALL) order, so `MEMORY.md` first
/// and `USER.md` second) and the stamp of the `facts/` directory
/// ([`LocalState::facts_stamp`]); `projections_dirty` records that the cached document is
/// ahead of the projection on disk because a refresh happened without the file lock.
#[derive(Debug)]
pub(super) struct LocalState {
    /// The parsed Markdown document.
    pub(super) document: MarkdownMemory,
    /// The stamp the document was loaded from (`None` when the file is absent).
    pub(super) stamp: Option<FileStamp>,
    /// The stamps of the rendered index documents, in [`IndexLevel::ALL`] order.
    ///
    /// [`IndexLevel::ALL`]: crate::lite::facts::IndexLevel::ALL
    pub(super) index_stamps: [Option<FileStamp>; 2],
    /// The stamp of the `facts/` directory (`None` before it was recorded).
    pub(super) facts_stamp: Option<DirStamp>,
    /// Whether the on-disk projection trails the cached document.
    pub(super) projections_dirty: bool,
    /// The version of the cached document, bumped on every change.
    ///
    /// The chunk index records the version it was built from, so a document that changed
    /// outside the store — an editor, another process, the watcher — is re-embedded before
    /// the next semantic recall instead of being scored with stale vectors.
    pub(super) version: u64,
}

impl LocalState {
    /// Builds the state of a freshly loaded document.
    pub(super) fn new(document: MarkdownMemory, stamp: Option<FileStamp>) -> Self {
        Self {
            document,
            stamp,
            index_stamps: [None; 2],
            facts_stamp: None,
            projections_dirty: false,
            version: 0,
        }
    }

    /// Records that the cached document changed, so derived state keyed on it is stale.
    pub(super) fn touch(&mut self) {
        self.version = self.version.wrapping_add(1);
    }

    /// The current document version.
    pub(super) fn version(&self) -> u64 {
        self.version
    }
}

/// The shared inner state; each [`LocalCodingMemory`] handle holds an `Arc` of it so a
/// watcher task can keep the store fresh.
pub(super) struct Inner {
    /// The directory holding `MEMORY.md`.
    pub(super) root: PathBuf,
    /// `<root>/MEMORY.md`.
    pub(super) memory_path: PathBuf,
    /// `<root>/.memory/index.lock`.
    pub(super) lock_path: PathBuf,
    /// Coding-memory settings (index cap, accepted fact types).
    pub(super) config: CodingMemoryConfig,
    /// The scope every fact of the document belongs to.
    pub(super) scope: MemoryScope,
    /// The cached document and its stamp.
    pub(super) state: Mutex<LocalState>,
    /// Number of times the document was reloaded from disk.
    pub(super) reloads: AtomicU64,
    /// The embedding model backing semantic search (`None` keeps keyword-only recall).
    pub(super) embedding: Option<Arc<dyn Embedding>>,
    /// The chunk shape of the embedding index.
    pub(super) chunking: ChunkingConfig,
    /// The content-hash keyed vector cache shared by the embedding index.
    pub(super) cache: EmbeddingCache,
    /// The chunked vector index, built lazily and refreshed on every mutation.
    ///
    /// The index is paired with the document version it was built from
    /// ([`LocalState::version`]), so a document changed outside the store is re-embedded
    /// before it is scored.
    pub(super) embeddings: Mutex<Option<EmbeddingIndexState>>,
    /// Whether the per-fact file layout (`facts/` plus `MEMORY.md`/`USER.md`) is enabled.
    pub(super) fact_files: bool,
    /// The per-fact file store of this root and scope.
    pub(super) facts: FactFiles,
    /// `<root>/USER.md`, the rendered index of the user-level facts.
    pub(super) user_path: PathBuf,
    /// The [`SyncTrigger::OnInterval`] gate, created from the first interval seen.
    ///
    /// [`SyncTrigger::OnInterval`]: crate::lite::sync::SyncTrigger::OnInterval
    pub(super) interval: Mutex<Option<IntervalScheduler>>,
    /// The opt-in background interval task, when one was started.
    pub(super) interval_task: Mutex<Option<IntervalTask>>,
}

/// An active watcher plus the task that applies its events.
pub(super) struct ActiveWatch {
    /// The debounced watcher.
    pub(super) watcher: MemoryWatcher,
    /// The reload task fed by the watcher.
    pub(super) task: JoinHandle<()>,
}

/// The chunk index together with the document version it was built from.
pub(super) struct EmbeddingIndexState {
    /// The chunked vector index over the fact bodies.
    pub(super) index: EmbeddingIndex,
    /// The [`LocalState::version`] the index was built from.
    pub(super) version: u64,
}

/// A local coding-memory store rooted at one directory.
///
/// Clone it freely: every clone shares the document cache, the lock file, and the
/// derived index. All mutation is serialized, and the Markdown file is authoritative, so
/// several instances (in one process or several) may share a root.
#[derive(Clone)]
pub struct LocalCodingMemory {
    /// Shared store state.
    pub(super) inner: Arc<Inner>,
    /// The derived FTS index (feature `sqlite`).
    #[cfg(feature = "sqlite")]
    pub(super) fts: FtsIndex,
    /// The running watcher, when [`LocalCodingMemory::start_watching`] was called.
    pub(super) watch: Arc<Mutex<Option<ActiveWatch>>>,
}

impl std::fmt::Debug for LocalCodingMemory {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LocalCodingMemory")
            .field("root", &self.inner.root)
            .field("memory_path", &self.inner.memory_path)
            .field("lock_path", &self.inner.lock_path)
            .field("scope", &self.inner.scope)
            .field("fact_files", &self.inner.fact_files)
            .finish_non_exhaustive()
    }
}

impl LocalCodingMemory {
    /// Opens (or initializes) the store rooted at `root` in the default scope.
    ///
    /// Creates the root and `.memory/` directories, writes an empty `MEMORY.md` when it
    /// does not exist, locks the store, loads the document, and rebuilds the derived
    /// index from it. The store has no embedding model, so recall is keyword-only; use
    /// [`LocalCodingMemory::builder`] to configure semantic search.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](openjiuwen_common::error::Error::Validation) for an
    /// invalid scope, and the filesystem, lock, or index error when the store cannot be
    /// initialized.
    pub async fn open(root: impl AsRef<Path>, config: CodingMemoryConfig) -> Result<Self> {
        Self::open_in_scope(root, config, MemoryScope::default()).await
    }

    /// Opens (or initializes) the store rooted at `root` in `scope`.
    ///
    /// # Errors
    ///
    /// Returns the [`LocalCodingMemory::open`] errors plus
    /// [`Error::Validation`](openjiuwen_common::error::Error::Validation) for an invalid
    /// scope.
    pub async fn open_in_scope(
        root: impl AsRef<Path>,
        config: CodingMemoryConfig,
        scope: MemoryScope,
    ) -> Result<Self> {
        Self::builder(root.as_ref().to_path_buf(), config)
            .scope(scope)
            .open()
            .await
    }

    /// Starts a builder for a store rooted at `root` with `config`.
    ///
    /// The builder is the only way to configure local semantic search: chain
    /// [`LocalCodingMemoryBuilder::embedding`] and/or
    /// [`LocalCodingMemoryBuilder::chunking`] before
    /// [`LocalCodingMemoryBuilder::open`].
    #[must_use]
    pub fn builder(
        root: impl Into<PathBuf>,
        config: CodingMemoryConfig,
    ) -> LocalCodingMemoryBuilder {
        LocalCodingMemoryBuilder::new(root, config)
    }

    /// The directory holding `MEMORY.md`.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.inner.root
    }

    /// The path of the Markdown source of truth.
    #[must_use]
    pub fn memory_path(&self) -> &Path {
        &self.inner.memory_path
    }

    /// The path of the cross-process lock file.
    #[must_use]
    pub fn lock_path(&self) -> &Path {
        &self.inner.lock_path
    }

    /// The coding-memory configuration in effect.
    #[must_use]
    pub fn config(&self) -> &CodingMemoryConfig {
        &self.inner.config
    }

    /// The scope every fact of this store belongs to.
    #[must_use]
    pub fn scope(&self) -> &MemoryScope {
        &self.inner.scope
    }

    /// How many times the document was reloaded from disk.
    ///
    /// Diagnostic counter: it grows when another process (or an editor) changed the
    /// Markdown file and the store noticed, either through a read or through an active
    /// watcher.
    #[must_use]
    pub fn reload_count(&self) -> u64 {
        self.inner.reloads.load(Ordering::Relaxed)
    }

    /// A snapshot of the parsed Markdown document.
    ///
    /// # Errors
    ///
    /// Returns the reload errors of [`LocalCodingMemory::list`].
    pub async fn markdown(&self) -> Result<MarkdownMemory> {
        let mut state = self.inner.state.lock().await;
        self.ensure_fresh(&mut state).await?;
        Ok(state.document.clone())
    }
}
