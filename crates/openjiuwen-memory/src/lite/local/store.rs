//! The local store's mutation and read API: writes, edits, deletes, listing, index
//! rendering, reindexing, and the locking/staleness/index helpers they share.
//!
//! Every mutation takes the cross-process file lock before the in-process state guard,
//! reloads the Markdown file when its stamp changed, writes it atomically, and refreshes
//! the derived index. When an embedding model is configured the chunk index is then
//! refreshed outside both guards, so the Markdown document is never locked while the
//! model runs (see `crate::lite::local::embedding`).
use std::sync::atomic::Ordering;

use chrono::Utc;
use tokio::sync::MutexGuard;

use openjiuwen_common::error::{Error, Result};

use crate::lite::fact::CodingFact;
use crate::lite::index::render_index;
use crate::lite::lock::FileLock;
use crate::lite::CodingFactType;

use super::layout::{load_document, write_document, FileStamp};
use super::search::sorted_facts;
use super::state::{LocalCodingMemory, LocalState};
use super::TARGET;

impl LocalCodingMemory {
    /// Writes `fact`, upserting the same-named fact of the store's scope.
    ///
    /// The fact type is resolved against
    /// [`CodingMemoryConfig::fact_types`](crate::config::CodingMemoryConfig::fact_types) (an
    /// unconfigured type is replaced by
    /// [`CodingMemoryConfig::default_fact_type`](crate::config::CodingMemoryConfig::default_fact_type)), the
    /// scope is the store's scope, and `updated_at` is set to now. Returns the stored
    /// fact. With an embedding model configured the chunk index is refreshed before the
    /// call returns.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the fact is invalid, and the lock, filesystem,
    /// index, or embedding error when the write fails.
    pub async fn write(&self, fact: CodingFact) -> Result<CodingFact> {
        fact.validate()?;
        let mut fact = fact;
        fact.fact_type = self.resolve_fact_type(fact.fact_type);
        fact.scope = self.inner.scope.clone();
        fact.updated_at = Utc::now();
        let (guard, mut state) = self.lock_for_write().await?;
        self.ensure_fresh_locked(&mut state).await?;
        let _replaced = state.document.upsert_fact(&fact);
        self.persist(&mut state).await?;
        #[cfg(feature = "sqlite")]
        self.fts.upsert(&fact).await?;
        drop(state);
        drop(guard);
        self.refresh_embedding_index().await?;
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
        let (guard, mut state) = self.lock_for_write().await?;
        self.ensure_fresh_locked(&mut state).await?;
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
        drop(state);
        drop(guard);
        self.refresh_embedding_index().await?;
        Ok(fact)
    }

    /// Deletes the fact named `name`, returning whether it existed.
    ///
    /// # Errors
    ///
    /// Returns the lock, filesystem, index, or embedding error when the delete fails.
    pub async fn delete(&self, name: &str) -> Result<bool> {
        let (guard, mut state) = self.lock_for_write().await?;
        self.ensure_fresh_locked(&mut state).await?;
        if !state.document.remove(name) {
            return Ok(false);
        }
        self.persist(&mut state).await?;
        #[cfg(feature = "sqlite")]
        self.fts.remove(name).await?;
        drop(state);
        drop(guard);
        self.refresh_embedding_index().await?;
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

    /// Renders the `MEMORY.md`-shaped summary index of the store.
    ///
    /// The summary is the prompt-facing shape (`- [Project] build — how to build`),
    /// capped at [`CodingMemoryConfig::max_index_lines`](crate::config::CodingMemoryConfig::max_index_lines)
    /// lines including its title. It is
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

    /// Rebuilds the derived indices from the Markdown file and returns the fact count.
    ///
    /// The file is re-read unconditionally (it is authoritative), the cached document is
    /// replaced, and the FTS index is rebuilt. Without the `sqlite` feature there is no
    /// full-text index, so only the reload happens. With an embedding model configured the
    /// chunk index is rebuilt too, after the file lock is released.
    ///
    /// # Errors
    ///
    /// Returns the lock, filesystem, parse, index, or embedding error on failure.
    pub async fn reindex(&self) -> Result<usize> {
        let guard = FileLock::acquire(&self.inner.lock_path).await?;
        let indexed = {
            let mut state = self.inner.state.lock().await;
            self.reindex_locked(&mut state).await?
        };
        drop(guard);
        self.refresh_embedding_index().await?;
        Ok(indexed)
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

    /// Acquires the cross-process lock, then the in-process state guard.
    ///
    /// Lock order is file lock before state guard; the watcher only takes the state
    /// guard, so the two can never deadlock.
    pub(super) async fn lock_for_write(&self) -> Result<(FileLock, MutexGuard<'_, LocalState>)> {
        let guard = FileLock::acquire(&self.inner.lock_path).await?;
        let state = self.inner.state.lock().await;
        Ok((guard, state))
    }

    /// Reloads the document when the authoritative file changed, returning whether it did.
    ///
    /// In the per-fact file layout the check covers the two index documents and the
    /// `facts/` directory (see `crate::lite::local::facts_mode`); this variant holds no
    /// file lock, so it may update the cached document and the derived index but it never
    /// rewrites the projection.
    pub(super) async fn ensure_fresh(&self, state: &mut LocalState) -> Result<bool> {
        if self.inner.fact_files {
            return self.refresh_facts(state, false).await;
        }
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

    /// Writes the cached document and refreshes its stamp.
    ///
    /// In the per-fact file layout the document is projected instead: every fact file is
    /// written (stale ones removed) and both index documents are re-rendered.
    pub(super) async fn persist(&self, state: &mut LocalState) -> Result<()> {
        if self.inner.fact_files {
            let _count = self.export_locked(state).await?;
            return Ok(());
        }
        write_document(&self.inner.memory_path, &state.document).await?;
        state.stamp = FileStamp::of(&self.inner.memory_path).await?;
        Ok(())
    }

    /// Refreshes the chunk index, when an embedding model is configured.
    ///
    /// Callers must have dropped the state guard and the file lock first: an embedding
    /// call may take arbitrarily long, and the document has to stay readable meanwhile.
    /// A store without an embedding model has no chunk index and does nothing here.
    pub(super) async fn refresh_embedding_index(&self) -> Result<()> {
        if !self.has_embedding() {
            return Ok(());
        }
        let _stats = self.refresh_embeddings().await?;
        Ok(())
    }

    /// Rebuilds the derived index from the cached document.
    pub(super) async fn rebuild_index(&self) -> Result<usize> {
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
    pub(super) async fn rebuild_index_from(&self, facts: &[CodingFact]) -> Result<usize> {
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
    pub(super) fn resolve_fact_type(&self, fact_type: CodingFactType) -> CodingFactType {
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
            Err(_error) => CodingFactType::Project,
        }
    }
}
