//! The local, file-backed coding-memory store: Markdown source of truth plus a derived
//! full-text index, a cross-process lock, and a debounced watcher.
//!
//! Ports the behaviour of Python's `MemoryIndexManager` for one coding-memory root:
//! `MEMORY.md` is the authoritative, human-editable document (see
//! [`MarkdownMemory`](super::markdown::MarkdownMemory)), a derived index database under
//! `.memory/` serves recall, and every mutation is serialized through a cross-process
//! advisory lock so several processes can share one root.
//!
//! # Layout
//!
//! ```text
//! <root>/
//!   MEMORY.md            the source of truth (fact sections + bodies)
//!   .memory/index.db     derived SQLite FTS5 index (feature `sqlite`)
//!   .memory/index.lock   cross-process advisory lock file
//! ```
//!
//! # Consistency model
//!
//! - The Markdown file wins. The index is derived and may always be rebuilt from it
//!   ([`LocalCodingMemory::reindex`]).
//! - A mutation takes the file lock ([`FileLock`]), reloads the Markdown file when its
//!   stamp (modification time plus length) differs from the cached one, applies the
//!   change, writes the document atomically (temporary file plus rename), and updates
//!   the index. A reader applies the same staleness check, so another process's
//!   committed fact becomes visible without an explicit refresh.
//! - Mutations are additionally serialized inside the process by an async mutex; the
//!   file lock alone would also work, but it is a blocking system call and the mutex
//!   keeps concurrent tasks from occupying blocking threads.
//! - `start_watching` additionally reloads (and reindexes) whenever the watcher reports a
//!   `.md` change, so an external editor's edit shows up in `search` without any call
//!   having to observe the file first. [`LocalCodingMemory::reload_count`] exposes how
//!   many reloads happened, which is what the watcher test observes.
//!
//! # Search
//!
//! With the `sqlite` feature the store queries the FTS5 index and merges the result with
//! a deterministic in-memory keyword scan; without the feature only the keyword scan
//! runs, so the type exists and works in both builds. The scan is needed in both builds
//! because the FTS `name` column is `UNINDEXED`, so a fact's name alone is not
//! reachable through `MATCH` (the Python implementation has the same property).
//!
//! The keyword score of a fact is the sum over query tokens of the strongest field it
//! matches (name 3, description or tags 2, body 1) divided by three times the token
//! count, capped at `1.0`; it is therefore in `0.0..=1.0` and higher is better. An FTS
//! hit contributes `1.0 - FtsHit::score` (the FTS score itself is higher for weaker
//! matches), and a fact keeps the maximum of its two signals. Only
//! [`SearchOptions::top_k`], [`SearchOptions::threshold`], [`SearchOptions::kinds`], and
//! [`SearchOptions::mode`] are honoured: `Semantic` mode returns
//! [`Error::Unsupported`](openjiuwen_common::error::Error::Unsupported) because the local
//! store holds no embeddings, and `token_budget` is ignored (there is no context window
//! here).
//!
//! Nothing in this module panics on user input: invalid facts, bad scopes, and missing
//! names are typed errors, and filesystem failures carry the failing path as context.
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, SystemTime};

use chrono::Utc;
use openjiuwen_common::error::{Error, Result};
use regex::Regex;
use tokio::sync::{mpsc, Mutex, MutexGuard};
use tokio::task::JoinHandle;

use super::fact::CodingFact;
use super::index::render_index;
use super::lock::FileLock;
use super::markdown::MarkdownMemory;
use super::watch::{MemoryWatcher, WatchEvent};
use crate::config::CodingMemoryConfig;
use crate::record::{MemoryScope, MemoryType, RecallHit, SearchMode, SearchOptions};

#[cfg(feature = "sqlite")]
use super::fts::FtsIndex;

/// Tracing target of the local-store diagnostics.
const TARGET: &str = "openjiuwen_memory::lite::local";

/// File name of the Markdown source of truth.
pub const MEMORY_FILE: &str = "MEMORY.md";

/// Directory (relative to the root) holding the derived index and the lock file.
pub const STATE_DIR: &str = ".memory";

/// File name of the derived full-text index (feature `sqlite`).
pub const INDEX_FILE: &str = "index.db";

/// File name of the cross-process advisory lock.
pub const LOCK_FILE: &str = "index.lock";

/// Debounce window used by [`LocalCodingMemory::start_watching`].
///
/// Matches the Python `sync.watchDebounceMs` intent at a smaller, test-friendly value:
/// filesystem bursts are coalesced into one reload.
pub const DEFAULT_DEBOUNCE: Duration = Duration::from_millis(200);

/// Maximum number of query tokens the keyword scan considers.
const MAX_QUERY_TOKENS: usize = 10;

/// Half-life of the recency component, in seconds (30 days).
const RECENCY_HALF_LIFE_SECS: f32 = 2_592_000.0;

/// One file's change stamp: modification time plus length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FileStamp {
    /// Last modification time, when the platform reports one.
    modified: Option<SystemTime>,
    /// File length in bytes.
    len: u64,
}

impl FileStamp {
    /// Reads the stamp of `path`, or `None` when the file does not exist.
    async fn of(path: &Path) -> Result<Option<Self>> {
        match tokio::fs::metadata(path).await {
            Ok(metadata) => Ok(Some(Self {
                modified: metadata.modified().ok(),
                len: metadata.len(),
            })),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => {
                Err(Error::external(error).with_context(format!("stamping {}", path.display())))
            }
        }
    }
}

/// The cached document plus the stamp it was loaded from.
#[derive(Debug)]
struct LocalState {
    /// The parsed Markdown document.
    document: MarkdownMemory,
    /// The stamp the document was loaded from (`None` when the file is absent).
    stamp: Option<FileStamp>,
}

/// The shared inner state; each [`LocalCodingMemory`] handle holds an `Arc` of it so a
/// watcher task can keep the store fresh.
struct Inner {
    /// The directory holding `MEMORY.md`.
    root: PathBuf,
    /// `<root>/MEMORY.md`.
    memory_path: PathBuf,
    /// `<root>/.memory/index.lock`.
    lock_path: PathBuf,
    /// Coding-memory settings (index cap, accepted fact types).
    config: CodingMemoryConfig,
    /// The scope every fact of the document belongs to.
    scope: MemoryScope,
    /// The cached document and its stamp.
    state: Mutex<LocalState>,
    /// Number of times the document was reloaded from disk.
    reloads: AtomicU64,
}

/// An active watcher plus the task that applies its events.
struct ActiveWatch {
    /// The debounced watcher.
    watcher: MemoryWatcher,
    /// The reload task fed by the watcher.
    task: JoinHandle<()>,
}

/// A local coding-memory store rooted at one directory.
///
/// Clone it freely: every clone shares the document cache, the lock file, and the
/// derived index. All mutation is serialized, and the Markdown file is authoritative, so
/// several instances (in one process or several) may share a root.
#[derive(Clone)]
pub struct LocalCodingMemory {
    /// Shared store state.
    inner: Arc<Inner>,
    /// The derived FTS index (feature `sqlite`).
    #[cfg(feature = "sqlite")]
    fts: FtsIndex,
    /// The running watcher, when [`LocalCodingMemory::start_watching`] was called.
    watch: Arc<Mutex<Option<ActiveWatch>>>,
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

    /// Writes `fact`, upserting the same-named fact of the store's scope.
    ///
    /// The fact type is resolved against [`CodingMemoryConfig::fact_types`] (an
    /// unconfigured type is replaced by [`CodingMemoryConfig::default_fact_type`]), the
    /// scope is the store's scope, and `updated_at` is set to now. Returns the stored
    /// fact.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the fact is invalid, and the lock, filesystem,
    /// or index error when the write fails.
    pub async fn write(&self, fact: CodingFact) -> Result<CodingFact> {
        fact.validate()?;
        let mut fact = fact;
        fact.fact_type = self.resolve_fact_type(fact.fact_type);
        fact.scope = self.inner.scope.clone();
        fact.updated_at = Utc::now();
        let (_guard, mut state) = self.lock_for_write().await?;
        self.ensure_fresh(&mut state).await?;
        let _replaced = state.document.upsert_fact(&fact);
        self.persist(&mut state).await?;
        #[cfg(feature = "sqlite")]
        self.fts.upsert(&fact).await?;
        Ok(fact)
    }

    /// Reads the fact named `name`, or `None` when it does not exist.
    ///
    /// # Errors
    ///
    /// Returns the filesystem/parse error when the Markdown file cannot be reloaded.
    pub async fn read(&self, name: &str) -> Result<Option<CodingFact>> {
        let mut state = self.inner.state.lock().await;
        self.ensure_fresh(&mut state).await?;
        Ok(state.document.fact(name))
    }

    /// Replaces the body of the fact named `name` and returns the updated fact.
    ///
    /// The fact's name, type, description, tags, and path are preserved; only the body
    /// and `updated_at` change.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] when no fact of that name exists,
    /// [`Error::Validation`] when `content` is blank (the Python lite memory rejects a
    /// fact without a body), and the lock, filesystem, or index error on failure.
    pub async fn edit(&self, name: &str, content: impl Into<String>) -> Result<CodingFact> {
        let content = content.into();
        if content.trim().is_empty() {
            return Err(Error::validation(format!(
                "coding fact `{name}` content must not be empty"
            )));
        }
        let (_guard, mut state) = self.lock_for_write().await?;
        self.ensure_fresh(&mut state).await?;
        let mut fact = state
            .document
            .fact(name)
            .ok_or_else(|| Error::not_found(format!("coding fact `{name}` not found")))?;
        fact.content = content;
        fact.updated_at = Utc::now();
        let _replaced = state.document.upsert_fact(&fact);
        self.persist(&mut state).await?;
        #[cfg(feature = "sqlite")]
        self.fts.upsert(&fact).await?;
        Ok(fact)
    }

    /// Deletes the fact named `name`, returning whether it existed.
    ///
    /// # Errors
    ///
    /// Returns the lock, filesystem, or index error when the delete fails.
    pub async fn delete(&self, name: &str) -> Result<bool> {
        let (_guard, mut state) = self.lock_for_write().await?;
        self.ensure_fresh(&mut state).await?;
        if !state.document.remove(name) {
            return Ok(false);
        }
        self.persist(&mut state).await?;
        #[cfg(feature = "sqlite")]
        self.fts.remove(name).await?;
        Ok(true)
    }

    /// Lists every fact, newest first and then by name.
    ///
    /// # Errors
    ///
    /// Returns the filesystem/parse error when the Markdown file cannot be reloaded.
    pub async fn list(&self) -> Result<Vec<CodingFact>> {
        let mut state = self.inner.state.lock().await;
        self.ensure_fresh(&mut state).await?;
        Ok(sorted_facts(&state.document))
    }

    /// Recalls the facts matching `query`.
    ///
    /// Hits are ordered by descending score and then by name ascending; see the module
    /// docs for the scoring rules. `kinds` filters the fact types (an empty list means
    /// "every kind", and a list without [`MemoryType::Coding`] yields no hit).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for a blank query, [`Error::Unsupported`] for
    /// semantic mode (the local store has no embeddings), the call's cancellation
    /// error, and the filesystem/index error when the search fails.
    pub async fn search(&self, query: &str, options: &SearchOptions) -> Result<Vec<RecallHit>> {
        let cleaned = query.trim();
        if cleaned.is_empty() {
            return Err(Error::validation(
                "search query must not be empty (got \"\")",
            ));
        }
        if options.mode == SearchMode::Semantic {
            return Err(Error::unsupported(
                "the local coding-memory store has no embedding index; use keyword or hybrid search",
            ));
        }
        if !options.kinds.is_empty() && !options.kinds.contains(&MemoryType::Coding) {
            return Ok(Vec::new());
        }
        options.call.check()?;
        let mut state = self.inner.state.lock().await;
        self.ensure_fresh(&mut state).await?;
        let facts = state.document.facts();
        drop(state);

        let limit = options.top_k.max(1);
        #[cfg_attr(not(feature = "sqlite"), allow(unused_mut))]
        let mut scores = keyword_scores(&facts, cleaned);
        #[cfg(feature = "sqlite")]
        {
            let candidates = self.fts.search(cleaned, limit.saturating_mul(4)).await?;
            for candidate in candidates {
                let score = 1.0 - candidate.score;
                let entry = scores
                    .iter_mut()
                    .find(|(name, _score)| *name == candidate.name);
                match entry {
                    Some((_name, score_slot)) => {
                        if score > *score_slot {
                            *score_slot = score;
                        }
                    }
                    None => scores.push((candidate.name, score)),
                }
            }
        }
        let mut ranked: Vec<(String, RecallHit)> = facts
            .iter()
            .filter_map(|fact| {
                let score = scores
                    .iter()
                    .find(|(name, _score)| *name == fact.name)
                    .map(|(_name, score)| *score)?;
                let score = score.clamp(0.0, 1.0);
                (score >= options.threshold).then(|| (fact.name.clone(), recall_hit(fact, score)))
            })
            .collect();
        ranked.sort_by(|(left_name, left), (right_name, right)| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left_name.cmp(right_name))
        });
        ranked.truncate(limit);
        Ok(ranked.into_iter().map(|(_name, hit)| hit).collect())
    }

    /// Renders the `MEMORY.md`-shaped summary index of the store.
    ///
    /// The summary is the prompt-facing shape (`- [Project] build — how to build`),
    /// capped at [`CodingMemoryConfig::max_index_lines`] lines including its title. It is
    /// not the on-disk document: `MEMORY.md` holds the full entries, this renders the
    /// compact list the harness can inject into a prompt.
    ///
    /// # Errors
    ///
    /// Returns the [`LocalCodingMemory::list`] errors.
    pub async fn render_index(&self) -> Result<String> {
        let facts = self.list().await?;
        Ok(render_index(&facts, self.inner.config.max_index_lines))
    }

    /// Rebuilds the derived index from the Markdown file and returns the fact count.
    ///
    /// The file is re-read unconditionally (it is authoritative), the cached document is
    /// replaced, and the FTS index is rebuilt. Without the `sqlite` feature there is no
    /// derived index, so only the reload happens.
    ///
    /// # Errors
    ///
    /// Returns the lock, filesystem, parse, or index error on failure.
    pub async fn reindex(&self) -> Result<usize> {
        let _guard = FileLock::acquire(&self.inner.lock_path).await?;
        let (document, stamp) = load_document(&self.inner.memory_path, &self.inner.scope).await?;
        {
            let mut state = self.inner.state.lock().await;
            state.document = document;
            state.stamp = stamp;
        }
        let _bumped = self.inner.reloads.fetch_add(1, Ordering::Relaxed);
        self.rebuild_index().await
    }

    /// The number of facts in the derived index (feature `sqlite`).
    ///
    /// # Errors
    ///
    /// Returns the index error when the count query fails.
    #[cfg(feature = "sqlite")]
    pub async fn indexed_count(&self) -> Result<usize> {
        self.fts.count().await
    }

    /// Starts watching the root for `.md` changes and reindexes on every debounced burst.
    ///
    /// Calling it again replaces the previous watcher. Must be called from within a tokio
    /// runtime. External edits then become visible to [`LocalCodingMemory::search`]
    /// without any call having to notice the file first.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`] when a watcher is already running is *not* an
    /// error (the old one is stopped first); the watcher's own errors are returned when
    /// the filesystem watch cannot be installed.
    pub async fn start_watching(&self) -> Result<()> {
        self.stop_watching().await;
        let (sender, mut receiver) = mpsc::unbounded_channel::<WatchEvent>();
        let watcher = MemoryWatcher::start([self.inner.root.clone()], DEFAULT_DEBOUNCE, sender)?;
        let store = self.clone();
        let task = tokio::spawn(async move {
            while let Some(event) = receiver.recv().await {
                tracing::debug!(
                    target: TARGET,
                    paths = event.paths.len(),
                    "memory watcher reported a markdown change"
                );
                let mut state = store.inner.state.lock().await;
                if let Err(error) = store.ensure_fresh(&mut state).await {
                    tracing::warn!(
                        target: TARGET,
                        error = %error,
                        "failed to reload the memory document after a filesystem change"
                    );
                }
            }
        });
        let mut slot = self.watch.lock().await;
        *slot = Some(ActiveWatch { watcher, task });
        Ok(())
    }

    /// Stops the watcher started by [`LocalCodingMemory::start_watching`].
    ///
    /// A no-op when no watcher is running.
    pub async fn stop_watching(&self) {
        let active = self.watch.lock().await.take();
        let Some(mut active) = active else {
            return;
        };
        active.watcher.stop().await;
        active.task.abort();
        let _joined = active.task.await;
    }

    /// Whether a watcher is currently running.
    pub async fn is_watching(&self) -> bool {
        self.watch
            .lock()
            .await
            .as_ref()
            .is_some_and(|active| active.watcher.is_running())
    }

    /// Acquires the cross-process lock, then the in-process state guard.
    ///
    /// Lock order is file lock before state guard; the watcher only takes the state
    /// guard, so the two can never deadlock.
    async fn lock_for_write(&self) -> Result<(FileLock, MutexGuard<'_, LocalState>)> {
        let guard = FileLock::acquire(&self.inner.lock_path).await?;
        let state = self.inner.state.lock().await;
        Ok((guard, state))
    }

    /// Reloads the document when the file stamp changed, returning whether it did.
    async fn ensure_fresh(&self, state: &mut LocalState) -> Result<bool> {
        let stamp = FileStamp::of(&self.inner.memory_path).await?;
        if stamp == state.stamp {
            return Ok(false);
        }
        let (document, stamp) = load_document(&self.inner.memory_path, &self.inner.scope).await?;
        state.document = document;
        state.stamp = stamp;
        let _bumped = self.inner.reloads.fetch_add(1, Ordering::Relaxed);
        let facts = state.document.facts();
        self.rebuild_index_from(&facts).await?;
        Ok(true)
    }

    /// Writes the cached document atomically and refreshes its stamp.
    async fn persist(&self, state: &mut LocalState) -> Result<()> {
        write_document(&self.inner.memory_path, &state.document).await?;
        state.stamp = FileStamp::of(&self.inner.memory_path).await?;
        Ok(())
    }

    /// Rebuilds the derived index from the cached document.
    async fn rebuild_index(&self) -> Result<usize> {
        let facts = {
            let state = self.inner.state.lock().await;
            state.document.facts()
        };
        self.rebuild_index_from(&facts).await
    }

    /// Rebuilds the derived index from an already-snapshotted fact list.
    ///
    /// Callers hold the state guard when they already own a document snapshot; taking
    /// the guard again here would deadlock the non-reentrant async mutex.
    #[allow(clippy::unused_async_trait_impl)] // inherent method; the lint misfires on the cfg split
    async fn rebuild_index_from(&self, facts: &[CodingFact]) -> Result<usize> {
        #[cfg(feature = "sqlite")]
        {
            self.fts.rebuild(facts.iter()).await
        }
        #[cfg(not(feature = "sqlite"))]
        {
            Ok(facts.len())
        }
    }

    /// Resolves the fact type to store against the configured type list.
    fn resolve_fact_type(&self, fact_type: super::CodingFactType) -> super::CodingFactType {
        let configured = self
            .inner
            .config
            .fact_types
            .iter()
            .any(|configured| configured.trim().eq_ignore_ascii_case(fact_type.as_str()));
        if configured {
            return fact_type;
        }
        tracing::debug!(
            target: TARGET,
            fact_type = fact_type.as_str(),
            default = %self.inner.config.default_fact_type,
            "coding fact type is not configured; storing the default fact type"
        );
        match self.inner.config.default_fact_type.parse() {
            Ok(resolved) => resolved,
            Err(_error) => super::CodingFactType::Project,
        }
    }
}

/// Loads (or initializes) the document at `path` for `scope`.
async fn load_document(
    path: &Path,
    scope: &MemoryScope,
) -> Result<(MarkdownMemory, Option<FileStamp>)> {
    let text = match tokio::fs::read_to_string(path).await {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok((MarkdownMemory::new().with_scope(scope.clone()), None));
        }
        Err(error) => {
            return Err(Error::external(error)
                .with_context(format!("reading the memory document {}", path.display())));
        }
    };
    let stamp = FileStamp::of(path).await?;
    if text.trim().is_empty() {
        return Ok((MarkdownMemory::new().with_scope(scope.clone()), stamp));
    }
    let document = MarkdownMemory::parse(&text)?.with_scope(scope.clone());
    Ok((document, stamp))
}

/// Writes `document` to `path` atomically (temporary file plus rename).
async fn write_document(path: &Path, document: &MarkdownMemory) -> Result<()> {
    let rendered = document.render();
    let temporary = path.with_extension("md.tmp");
    tokio::fs::write(&temporary, rendered.as_bytes())
        .await
        .map_err(|error| {
            Error::external(error).with_context(format!(
                "writing the temporary memory document {}",
                temporary.display()
            ))
        })?;
    tokio::fs::rename(&temporary, path).await.map_err(|error| {
        Error::external(error)
            .with_context(format!("replacing the memory document {}", path.display()))
    })
}

/// Returns the facts ordered newest first and then by name.
fn sorted_facts(document: &MarkdownMemory) -> Vec<CodingFact> {
    let mut facts = document.facts();
    facts.sort_by(|left, right| {
        right
            .updated_at
            .cmp(&left.updated_at)
            .then_with(|| left.name.cmp(&right.name))
    });
    facts
}

/// Splits a query into lowercased word tokens (at most [`MAX_QUERY_TOKENS`]).
fn query_tokens(query: &str) -> Vec<String> {
    static TOKENS: OnceLock<Option<Regex>> = OnceLock::new();
    let Some(regex) = TOKENS.get_or_init(|| Regex::new(r"\w+").ok()) else {
        return Vec::new();
    };
    let mut tokens: Vec<String> = Vec::new();
    for token in regex.find_iter(query) {
        let token = token.as_str().to_lowercase();
        if !tokens.contains(&token) {
            tokens.push(token);
        }
        if tokens.len() == MAX_QUERY_TOKENS {
            break;
        }
    }
    tokens
}

/// Scores every fact that matches at least one query token.
fn keyword_scores(facts: &[CodingFact], query: &str) -> Vec<(String, f32)> {
    let tokens = query_tokens(query);
    if tokens.is_empty() {
        return Vec::new();
    }
    let divisor = 3.0 * tokens.len() as f32;
    let mut scores = Vec::new();
    for fact in facts {
        let name = fact.name.to_lowercase();
        let description = fact.description.to_lowercase();
        let body = fact.content.to_lowercase();
        let tags: Vec<String> = fact.tags.iter().map(|tag| tag.to_lowercase()).collect();
        let mut matched = 0.0_f32;
        for token in &tokens {
            let mut weight = 0.0_f32;
            if name.contains(token.as_str()) {
                weight = 3.0;
            } else if description.contains(token.as_str())
                || tags.iter().any(|tag| tag.contains(token.as_str()))
            {
                weight = 2.0;
            } else if body.contains(token.as_str()) {
                weight = 1.0;
            }
            matched += weight;
        }
        let score = (matched / divisor).min(1.0);
        if score > 0.0 {
            scores.push((fact.name.clone(), score));
        }
    }
    scores
}

/// Builds a [`RecallHit`] for `fact` with `score` as its blended score.
fn recall_hit(fact: &CodingFact, score: f32) -> RecallHit {
    let mut record = fact.to_record();
    record.importance = 0.5;
    let importance = record.importance;
    RecallHit {
        record,
        score,
        similarity: 0.0,
        keyword: score,
        importance,
        recency: recency(fact),
    }
}

/// The recency component: `0.5^(age / 30 days)`, clamped to `0.0..=1.0`.
fn recency(fact: &CodingFact) -> f32 {
    let now = Utc::now();
    let age = match now.signed_duration_since(fact.updated_at).num_seconds() {
        age if age <= 0 => return 1.0,
        age => age as f32,
    };
    (0.5_f32).powf(age / RECENCY_HALF_LIFE_SECS).clamp(0.0, 1.0)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::lite::CodingFactType;
    use openjiuwen_common::error::Error as MemoryError;
    use tempfile::TempDir;

    fn scope() -> MemoryScope {
        MemoryScope::new("alice", "ws")
    }

    async fn store_at(temp: &TempDir) -> LocalCodingMemory {
        LocalCodingMemory::open_in_scope(temp.path(), CodingMemoryConfig::default(), scope())
            .await
            .unwrap()
    }

    fn fact(name: &str, content: &str) -> CodingFact {
        CodingFact::new(&scope(), name, content)
    }

    fn keyword_options() -> SearchOptions {
        SearchOptions::default().with_mode(SearchMode::Keyword)
    }

    #[tokio::test]
    async fn open_creates_the_document_and_the_state_directory() {
        let temp = TempDir::new().unwrap();
        let store = store_at(&temp).await;
        assert!(store.memory_path().exists());
        assert!(store.lock_path().exists());
        let text = std::fs::read_to_string(store.memory_path()).unwrap();
        assert!(text.contains("# Memory"), "{text}");
        assert_eq!(store.list().await.unwrap(), Vec::<CodingFact>::new());
        assert_eq!(store.reload_count(), 0);
        #[cfg(feature = "sqlite")]
        assert_eq!(store.indexed_count().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn write_read_edit_delete_round_trip() {
        let temp = TempDir::new().unwrap();
        let store = store_at(&temp).await;
        let written = store
            .write(fact("build", "cargo build").with_description("How to build"))
            .await
            .unwrap();
        assert_eq!(written.name, "build");
        assert_eq!(written.scope, scope());

        let read = store.read("build").await.unwrap().unwrap();
        assert_eq!(read.content, "cargo build");
        assert_eq!(read.description, "How to build");

        let edited = store.edit("build", "cargo build --release").await.unwrap();
        assert_eq!(edited.content, "cargo build --release");
        assert_eq!(
            store.read("build").await.unwrap().unwrap().content,
            "cargo build --release"
        );

        assert!(store.delete("build").await.unwrap());
        assert!(!store.delete("build").await.unwrap());
        assert!(store.read("build").await.unwrap().is_none());
        assert_eq!(store.list().await.unwrap(), Vec::<CodingFact>::new());
    }

    #[tokio::test]
    async fn edit_and_delete_missing_facts_are_typed() {
        let temp = TempDir::new().unwrap();
        let store = store_at(&temp).await;
        let error = store.edit("ghost", "body").await.unwrap_err();
        assert!(matches!(error, MemoryError::NotFound(_)), "{error:?}");
        assert!(!store.delete("ghost").await.unwrap());
        let error = store.edit("ghost", "   ").await.unwrap_err();
        assert!(matches!(error, MemoryError::Validation(_)), "{error:?}");
    }

    #[tokio::test]
    async fn search_finds_a_fact_by_name_body_description_and_tag() {
        let temp = TempDir::new().unwrap();
        let store = store_at(&temp).await;
        let _ = store
            .write(
                fact("release-checklist", "run the workspace test suite")
                    .with_description("how to ship")
                    .with_tags(vec!["ship".to_owned()]),
            )
            .await
            .unwrap();
        let _ = store
            .write(fact("unrelated", "nothing to see"))
            .await
            .unwrap();

        for query in ["release-checklist", "workspace", "ship"] {
            let hits = store.search(query, &keyword_options()).await.unwrap();
            assert!(!hits.is_empty(), "no hit for {query}");
            assert_eq!(
                hits[0]
                    .record
                    .metadata
                    .get("name")
                    .and_then(serde_json::Value::as_str),
                Some("release-checklist"),
                "query {query}"
            );
        }
        let hits = store.search("release", &keyword_options()).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert!(hits[0].score > 0.0 && hits[0].score <= 1.0);
        assert_eq!(
            store
                .search("absent-term", &keyword_options())
                .await
                .unwrap(),
            Vec::<RecallHit>::new()
        );
    }

    #[tokio::test]
    async fn search_rejects_blank_queries_and_semantic_mode() {
        let temp = TempDir::new().unwrap();
        let store = store_at(&temp).await;
        let error = store.search("   ", &keyword_options()).await.unwrap_err();
        assert!(matches!(error, MemoryError::Validation(_)));
        let error = store
            .search(
                "anything",
                &SearchOptions::default().with_mode(SearchMode::Semantic),
            )
            .await
            .unwrap_err();
        assert!(matches!(error, MemoryError::Unsupported(_)), "{error:?}");
        let hits = store
            .search(
                "anything",
                &keyword_options().with_kinds(vec![MemoryType::Summary]),
            )
            .await
            .unwrap();
        assert_eq!(hits, Vec::<RecallHit>::new());
    }

    #[tokio::test]
    async fn the_summary_index_uses_the_documented_shape() {
        let temp = TempDir::new().unwrap();
        let store = store_at(&temp).await;
        let _ = store
            .write(
                fact("build", "cargo build")
                    .with_description("How to build")
                    .with_path("docs/BUILD.md"),
            )
            .await
            .unwrap();
        let index = store.render_index().await.unwrap();
        assert!(index.starts_with("# Memory\n"), "{index}");
        assert_eq!(
            index.lines().nth(1),
            Some("- [Project] build — How to build (`docs/BUILD.md`)")
        );
    }

    #[tokio::test]
    async fn a_second_instance_sees_committed_facts() {
        let temp = TempDir::new().unwrap();
        let first = store_at(&temp).await;
        let second = store_at(&temp).await;
        let _ = first.write(fact("build", "cargo build")).await.unwrap();

        let seen = second.read("build").await.unwrap().unwrap();
        assert_eq!(seen.content, "cargo build");
        assert_eq!(second.reload_count(), 1);

        let _ = first.write(fact("test", "cargo test")).await.unwrap();
        assert_eq!(second.list().await.unwrap().len(), 2);
        assert_eq!(second.reload_count(), 2);
    }

    #[tokio::test]
    async fn external_markdown_edits_are_reloaded_and_reindexed() {
        let temp = TempDir::new().unwrap();
        let store = store_at(&temp).await;
        let _ = store.write(fact("build", "cargo build")).await.unwrap();

        let mut document =
            MarkdownMemory::parse(&std::fs::read_to_string(store.memory_path()).unwrap())
                .unwrap()
                .with_scope(scope());
        let _ = document.upsert_fact(&fact("hand-written", "edited outside the store"));
        std::fs::write(store.memory_path(), document.render()).unwrap();

        let read = store.read("hand-written").await.unwrap().unwrap();
        assert_eq!(read.content, "edited outside the store");
        #[cfg(feature = "sqlite")]
        assert_eq!(store.indexed_count().await.unwrap(), 2);
        assert_eq!(
            store
                .search("edited", &keyword_options())
                .await
                .unwrap()
                .len(),
            2
        );

        // A write after the external edit must keep the external fact.
        let _ = store.write(fact("test", "cargo test")).await.unwrap();
        let text = std::fs::read_to_string(store.memory_path()).unwrap();
        assert!(text.contains("hand-written"), "{text}");
        assert_eq!(store.list().await.unwrap().len(), 3);
    }

    #[tokio::test]
    async fn reindex_rebuilds_the_derived_index_from_the_file() {
        let temp = TempDir::new().unwrap();
        let store = store_at(&temp).await;
        let reindexed = store.reindex().await.unwrap();
        assert_eq!(reindexed, 0);
        assert_eq!(store.reload_count(), 1);

        let mut document = MarkdownMemory::new().with_scope(scope());
        let _ = document.upsert_fact(&fact("only-in-file", "written by an editor"));
        std::fs::write(store.memory_path(), document.render()).unwrap();
        assert_eq!(store.reindex().await.unwrap(), 1);
        assert_eq!(store.reload_count(), 2);
        #[cfg(feature = "sqlite")]
        assert_eq!(store.indexed_count().await.unwrap(), 1);
        assert_eq!(store.list().await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn the_markdown_document_round_trips_through_the_store() {
        let temp = TempDir::new().unwrap();
        let store = store_at(&temp).await;
        let _ = store
            .write(fact("build", "cargo build").with_type(CodingFactType::Project))
            .await
            .unwrap();
        let _ = store
            .write(
                fact("tone", "be concise")
                    .with_type(CodingFactType::Feedback)
                    .with_tags(vec!["style".to_owned()]),
            )
            .await
            .unwrap();
        let snapshot = store.markdown().await.unwrap();
        let text = std::fs::read_to_string(store.memory_path()).unwrap();
        let parsed = MarkdownMemory::parse(&text).unwrap().with_scope(scope());
        assert_eq!(parsed.render(), text);
        assert_eq!(parsed.facts().len(), 2);
        assert_eq!(snapshot.facts().len(), 2);
        assert!(text.contains("## Feedback"), "{text}");
    }

    #[tokio::test]
    async fn concurrent_writers_are_serialized() {
        let temp = TempDir::new().unwrap();
        let store = store_at(&temp).await;
        let mut tasks = Vec::new();
        for index in 0..8 {
            let handle = store.clone();
            tasks.push(tokio::spawn(async move {
                handle
                    .write(fact(&format!("fact-{index}"), &format!("body {index}")))
                    .await
                    .unwrap();
            }));
        }
        for task in tasks {
            task.await.unwrap();
        }
        let facts = store.list().await.unwrap();
        assert_eq!(facts.len(), 8);
        let text = std::fs::read_to_string(store.memory_path()).unwrap();
        assert_eq!(text.matches("### fact-").count(), 8, "{text}");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn watching_picks_up_external_edits() {
        let temp = TempDir::new().unwrap();
        let store = store_at(&temp).await;
        let _ = store.write(fact("build", "cargo build")).await.unwrap();
        store.start_watching().await.unwrap();
        assert!(store.is_watching().await);
        let before = store.reload_count();

        let mut document =
            MarkdownMemory::parse(&std::fs::read_to_string(store.memory_path()).unwrap())
                .unwrap()
                .with_scope(scope());
        let _ = document.upsert_fact(&fact("from-editor", "written while watching"));
        std::fs::write(store.memory_path(), document.render()).unwrap();

        let mut observed = false;
        for _ in 0..100 {
            if store.reload_count() > before {
                observed = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert!(observed, "the watcher never reloaded the document");
        assert_eq!(
            store.read("from-editor").await.unwrap().unwrap().content,
            "written while watching"
        );

        store.stop_watching().await;
        assert!(!store.is_watching().await);
        let after = store.reload_count();
        let _ = document.upsert_fact(&fact("after-stop", "ignored"));
        std::fs::write(store.memory_path(), document.render()).unwrap();
        tokio::time::sleep(Duration::from_millis(400)).await;
        assert_eq!(store.reload_count(), after);
    }

    #[test]
    fn the_store_handle_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<LocalCodingMemory>();
    }
}
