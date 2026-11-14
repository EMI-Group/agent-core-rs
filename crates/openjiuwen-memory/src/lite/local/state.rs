//! The shared store state and the [`LocalCodingMemory`] handle: construction, cheap
//! accessors, and the parsed-document snapshot accessor.
//!
//! The mutation/read API lives in `crate::lite::local::store`, recall in
//! `crate::lite::local::search`, and the watcher lifecycle in
//! `crate::lite::local::watcher`; all three are additional inherent `impl` blocks on the
//! same type and reach the state declared here through `pub(super)` items.
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use tokio::sync::Mutex;
use tokio::task::JoinHandle;

use openjiuwen_common::error::{Error, Result};

use crate::config::CodingMemoryConfig;
use crate::lite::lock::FileLock;
use crate::lite::markdown::MarkdownMemory;
use crate::lite::watch::MemoryWatcher;
use crate::record::MemoryScope;

use super::layout::{load_document, write_document, FileStamp, LOCK_FILE, MEMORY_FILE, STATE_DIR};

#[cfg(feature = "sqlite")]
use super::layout::INDEX_FILE;

#[cfg(feature = "sqlite")]
use crate::lite::fts::FtsIndex;

/// The cached document plus the stamp it was loaded from.
#[derive(Debug)]
pub(super) struct LocalState {
    /// The parsed Markdown document.
    pub(super) document: MarkdownMemory,
    /// The stamp the document was loaded from (`None` when the file is absent).
    pub(super) stamp: Option<FileStamp>,
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
}

/// An active watcher plus the task that applies its events.
pub(super) struct ActiveWatch {
    /// The debounced watcher.
    pub(super) watcher: MemoryWatcher,
    /// The reload task fed by the watcher.
    pub(super) task: JoinHandle<()>,
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
            .finish_non_exhaustive()
    }
}

impl LocalCodingMemory {
    /// Opens (or initializes) the store rooted at `root` in the default scope.
    ///
    /// Creates the root and `.memory/` directories, writes an empty `MEMORY.md` when it
    /// does not exist, locks the store, loads the document, and rebuilds the derived
    /// index from it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an invalid scope, and the filesystem, lock, or
    /// index error when the store cannot be initialized.
    pub async fn open(root: impl AsRef<Path>, config: CodingMemoryConfig) -> Result<Self> {
        Self::open_in_scope(root, config, MemoryScope::default()).await
    }

    /// Opens (or initializes) the store rooted at `root` in `scope`.
    ///
    /// # Errors
    ///
    /// Returns the [`LocalCodingMemory::open`] errors plus [`Error::Validation`] for an
    /// invalid scope.
    pub async fn open_in_scope(
        root: impl AsRef<Path>,
        config: CodingMemoryConfig,
        scope: MemoryScope,
    ) -> Result<Self> {
        scope.validate()?;
        let root = root.as_ref().to_path_buf();
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
        let inner = Arc::new(Inner {
            root,
            memory_path,
            lock_path,
            config,
            scope,
            state: Mutex::new(LocalState { document, stamp }),
            reloads: AtomicU64::new(0),
        });
        #[cfg(feature = "sqlite")]
        let fts = FtsIndex::open_path(&state_dir.join(INDEX_FILE)).await?;
        let store = Self {
            #[cfg(feature = "sqlite")]
            fts,
            inner,
            watch: Arc::new(Mutex::new(None)),
        };
        let _indexed = store.rebuild_index().await?;
        Ok(store)
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
