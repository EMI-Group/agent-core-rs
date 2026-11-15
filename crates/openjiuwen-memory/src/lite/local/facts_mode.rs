//! The per-fact file layout of the local store: `facts/`, `MEMORY.md`, and `USER.md`.
//!
//! [`LocalCodingMemoryBuilder::fact_files`](crate::lite::local::LocalCodingMemoryBuilder::fact_files)
//! opts into the layout of [`crate::lite::facts`]: every fact is persisted to its own
//! Markdown file, and the two top-level documents are rendered, human-editable projections
//! over those files.
//!
//! # Layout
//!
//! ```text
//! <root>/
//!   facts/<slug>-<hash>.md   one fact per file (front matter plus the Markdown body)
//!   MEMORY.md                rendered index of the non-user facts
//!   USER.md                  rendered index of the user-level facts
//!   .memory/index.db         derived SQLite index (feature `sqlite`), unchanged
//!   .memory/index.lock       cross-process advisory lock, unchanged
//! ```
//!
//! Both index documents are always written, even when their level is empty, so a caller
//! may edit either of them and [`LocalCodingMemory::import_index`] reads both back.
//!
//! # Consistency model
//!
//! - The per-fact files are authoritative for content; the two index documents are the
//!   human-editable projection. A fact file records every fact field except the scope (one
//!   root holds one scope), so a projection can be regenerated at any time.
//! - `LocalState` records the stamps it has seen: the two index-document stamps (in
//!   [`IndexLevel::ALL`] order) and the `facts/` directory stamp. Every read or mutation
//!   compares them with the files on disk.
//! - An index document that changed since the recorded stamp wins: it is imported, and
//!   the fact files are rewritten to match (a hand edit is the caller's intent). Otherwise
//!   a changed facts directory wins: every fact file is loaded, the cached document is
//!   rebuilt, and both index documents are re-rendered. When both changed the index wins
//!   and a warning records the fact.
//! - An index document that is empty, missing, or holds no entry is "no information", not
//!   "no facts": the fact files of that level are kept and the entry set of the other
//!   level still applies. Only a non-empty document expresses a removal, so a truncated or
//!   emptied `MEMORY.md` can never delete fact files. `import_index` follows the same
//!   rule; to remove a fact, edit it out of a non-empty document or call `delete`.
//! - A fact file that does not decode as a fact is left untouched by an export, and its
//!   path is reported: the store cannot tell a deleted fact from a file it failed to
//!   parse, so it never deletes what it could not read.
//! - Importing, exporting, and any refresh that rewrites files run under the cross-process
//!   [`FileLock`](crate::lite::lock::FileLock) and then the state guard, never the other
//!   way round. A refresh reached from a reader (or from the embedding refresh hook) holds
//!   only the state guard, so it updates the cached document and the derived full-text
//!   index but marks the projection dirty (`projections_dirty`) instead of writing without
//!   the lock; the next locked operation (a mutation, [`LocalCodingMemory::export_index`],
//!   [`LocalCodingMemory::import_index`], [`LocalCodingMemory::sync_now`], or
//!   `reindex`) flushes it.
//! - Every actual reload bumps
//!   [`reload_count`](crate::lite::local::LocalCodingMemory::reload_count) and rebuilds the
//!   derived full-text index, exactly like the single-document layout; the chunk index
//!   follows through the existing embedding refresh.
//!
//! # Fresh open
//!
//! Opening a root with the layout enabled reconciles the two representations: when `facts/`
//! holds at least one readable fact the files are adopted (and the index documents
//! re-rendered from them), otherwise the loaded `MEMORY.md` document is exported into
//! per-fact files plus both index documents. A root that was used in the default layout
//! therefore migrates on its first fact-file open, moving `user` facts out of `MEMORY.md`
//! into `USER.md`.
//!
//! # Idempotency
//!
//! Rendering is deterministic (fixed fact-type order, name-ascending inside a type), so
//! exporting twice with no change leaves both documents byte-identical, and importing a
//! rendered index reproduces the same bytes when it is exported again.
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;

use openjiuwen_common::error::{Error, Result};

use crate::lite::fact::CodingFact;
use crate::lite::facts::{level_of, parse_level, render_level, IndexLevel};
use crate::lite::lock::FileLock;
use crate::lite::markdown::MarkdownMemory;

use super::layout::{load_document, write_document, FileStamp};
use super::state::{LocalCodingMemory, LocalState};
use super::TARGET;

/// What the two rendered index documents say.
///
/// `absent` names the levels whose document was missing, empty, or unreadable as facts, so
/// the caller knows which levels carry no information this time round.
struct IndexDocuments {
    /// The facts both documents hold, merged by name (`USER.md` wins a duplicate).
    facts: Vec<CodingFact>,
    /// The levels whose document carries no importable entry.
    absent: Vec<IndexLevel>,
}

impl LocalCodingMemory {
    /// Whether the per-fact file layout is enabled.
    #[must_use]
    pub fn fact_files_enabled(&self) -> bool {
        self.inner.fact_files
    }

    /// The directory holding one Markdown file per fact (`<root>/facts`).
    ///
    /// The path is reported whether or not the layout is enabled and whether or not the
    /// directory exists.
    #[must_use]
    pub fn facts_dir(&self) -> PathBuf {
        self.inner.facts.dir()
    }

    /// The path of `<root>/USER.md`, the rendered index of the user-level facts.
    #[must_use]
    pub fn user_path(&self) -> PathBuf {
        self.inner.user_path.clone()
    }

    /// Writes every fact of the document to its own file and renders `MEMORY.md`/`USER.md`.
    ///
    /// Fact files whose fact no longer exists are removed, so a delete propagates. Both
    /// index documents are written atomically (temporary file plus rename) and the recorded
    /// stamps are refreshed. The call is idempotent: exporting twice with no change leaves
    /// every file byte-identical.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`] when the store was opened without
    /// [`LocalCodingMemoryBuilder::fact_files`](crate::lite::local::LocalCodingMemoryBuilder::fact_files),
    /// and the lock, filesystem, or parse error when a file cannot be written.
    pub async fn export_index(&self) -> Result<usize> {
        self.require_fact_files()?;
        let guard = FileLock::acquire(&self.inner.lock_path).await?;
        let count = {
            let mut state = self.inner.state.lock().await;
            let _reloaded = self.refresh_facts(&mut state, false).await?;
            self.export_locked(&mut state).await?
        };
        drop(guard);
        self.refresh_embedding_index().await?;
        Ok(count)
    }

    /// Imports `MEMORY.md`/`USER.md` (and any per-fact file changes) into the document.
    ///
    /// Both documents are parsed tolerantly (a missing file is empty), merged by fact name
    /// — a name present in both is taken from `USER.md`, with a warning — and written back
    /// to the per-fact files, after which both index documents are re-rendered and the
    /// derived indices rebuilt. Rendering the imported facts therefore reproduces the
    /// hand-edited document byte for byte.
    ///
    /// A level whose document is missing or empty carries no information, so the facts of
    /// that level that the per-fact files already hold are kept: an emptied (or truncated)
    /// index document never deletes fact files. A genuine removal is expressed by editing a
    /// non-empty document — delete the entry there, or call
    /// [`LocalCodingMemory::delete`](crate::lite::local::LocalCodingMemory::delete).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`] when the per-fact layout is disabled, the lock,
    /// filesystem, or parse error while reading or writing the files, and the index error
    /// when the derived index cannot be rebuilt.
    pub async fn import_index(&self) -> Result<usize> {
        self.require_fact_files()?;
        let guard = FileLock::acquire(&self.inner.lock_path).await?;
        let count = {
            let mut state = self.inner.state.lock().await;
            let facts = self.read_index_documents().await?;
            let count = self.adopt_facts(&mut state, facts);
            let _exported = self.export_locked(&mut state).await?;
            let _bumped = self.inner.reloads.fetch_add(1, Ordering::Relaxed);
            let facts = state.document.facts();
            let _indexed = self.rebuild_index_from(&facts).await?;
            count
        };
        drop(guard);
        self.refresh_embedding_index().await?;
        Ok(count)
    }

    /// Reconciles a freshly opened root: fact files win when they hold any fact, else the
    /// loaded `MEMORY.md` document is exported into them.
    ///
    /// The caller holds the cross-process file lock.
    pub(super) async fn initialize_fact_files(&self) -> Result<()> {
        let mut state = self.inner.state.lock().await;
        let (stored, skipped) = self.inner.facts.read_all_report().await?;
        if !skipped.is_empty() {
            tracing::warn!(
                target: TARGET,
                count = skipped.len(),
                "some fact files are not decodable and were skipped while opening the store"
            );
        }
        if !stored.is_empty() {
            let _count = self.adopt_facts(&mut state, stored);
        }
        let _count = self.export_locked(&mut state).await?;
        Ok(())
    }

    /// A staleness check for a caller that already holds the cross-process file lock.
    ///
    /// In the per-fact layout it writes the projection when it has to; in the default
    /// layout it is exactly `LocalCodingMemory::ensure_fresh`.
    pub(super) async fn ensure_fresh_locked(&self, state: &mut LocalState) -> Result<bool> {
        if self.inner.fact_files {
            return self.refresh_facts(state, true).await;
        }
        self.ensure_fresh(state).await
    }

    /// Rebuilds the derived indices (and the projection, when enabled) from scratch.
    pub(super) async fn reindex_locked(&self, state: &mut LocalState) -> Result<usize> {
        if !self.inner.fact_files {
            let (document, stamp) =
                load_document(&self.inner.memory_path, &self.inner.scope).await?;
            state.document = document;
            state.stamp = stamp;
            state.touch();
            let _bumped = self.inner.reloads.fetch_add(1, Ordering::Relaxed);
            let facts = state.document.facts();
            return self.rebuild_index_from(&facts).await;
        }
        let (stored, skipped) = self.inner.facts.read_all_report().await?;
        if !skipped.is_empty() {
            tracing::warn!(
                target: TARGET,
                count = skipped.len(),
                "some fact files are not decodable and were skipped by the reindex"
            );
        }
        let facts = if stored.is_empty() {
            self.read_index_documents().await?
        } else {
            stored
        };
        if facts.is_empty() {
            let (document, stamp) =
                load_document(&self.inner.memory_path, &self.inner.scope).await?;
            state.document = document;
            state.stamp = stamp;
            state.touch();
        } else {
            let _count = self.adopt_facts(state, facts);
        }
        let _count = self.export_locked(state).await?;
        let _bumped = self.inner.reloads.fetch_add(1, Ordering::Relaxed);
        let facts = state.document.facts();
        self.rebuild_index_from(&facts).await
    }

    /// Reloads the cached document from the authoritative representation of the enabled
    /// layout, returning whether it changed.
    ///
    /// `write` reports whether the caller holds the cross-process file lock: when it does,
    /// the projection (fact files plus both index documents) is rewritten to match the
    /// reloaded document; when it does not, the projection is marked dirty for the next
    /// locked operation and nothing is written.
    pub(super) async fn refresh_facts(&self, state: &mut LocalState, write: bool) -> Result<bool> {
        let index_changed = self.index_documents_changed(state).await?;
        let facts_stamp = self.inner.facts.stamp().await?;
        let facts_changed = state.facts_stamp != Some(facts_stamp);
        if !index_changed && !facts_changed {
            if write && state.projections_dirty {
                let _count = self.export_locked(state).await?;
            }
            return Ok(false);
        }
        if index_changed && facts_changed {
            tracing::warn!(
                target: TARGET,
                "both the fact files and an index document changed; the index document wins"
            );
        }
        let facts = if index_changed {
            self.read_index_documents().await?
        } else {
            let (facts, skipped) = self.inner.facts.read_all_report().await?;
            if !skipped.is_empty() {
                tracing::warn!(
                    target: TARGET,
                    count = skipped.len(),
                    "some fact files are not decodable and were skipped by the reload"
                );
            }
            facts
        };
        let _count = self.adopt_facts(state, facts);
        if write {
            let _count = self.export_locked(state).await?;
        } else {
            self.record_stamps(state).await?;
            state.projections_dirty = true;
        }
        let _bumped = self.inner.reloads.fetch_add(1, Ordering::Relaxed);
        let facts = state.document.facts();
        let _indexed = self.rebuild_index_from(&facts).await?;
        Ok(true)
    }

    /// Writes every fact file, removes the stale ones, renders both index documents, and
    /// records the stamps. The caller holds the file lock and the state guard.
    ///
    /// Only a fact file whose content decodes as a fact can be recognized as stale: a file
    /// that does not decode is left untouched, because the store cannot tell a deleted
    /// fact from a file it merely failed to parse.
    pub(super) async fn export_locked(&self, state: &mut LocalState) -> Result<usize> {
        let facts = state.document.facts();
        for fact in &facts {
            let _path = self.inner.facts.write(fact).await?;
        }
        let live: BTreeSet<String> = facts.iter().map(|fact| fact.name.clone()).collect();
        let (stored, skipped) = self.inner.facts.read_all_report().await?;
        if !skipped.is_empty() {
            tracing::warn!(
                target: TARGET,
                count = skipped.len(),
                "leaving undecodable fact files untouched during the export"
            );
        }
        for stored in stored {
            if !live.contains(&stored.name) {
                let _removed = self.inner.facts.remove(&stored.name).await?;
            }
        }
        self.render_index_documents(&facts).await?;
        self.record_stamps(state).await?;
        state.projections_dirty = false;
        Ok(facts.len())
    }

    /// Replaces the cached document with `facts`, returning the count.
    pub(super) fn adopt_facts(&self, state: &mut LocalState, facts: Vec<CodingFact>) -> usize {
        let mut document = MarkdownMemory::new().with_scope(self.inner.scope.clone());
        for fact in &facts {
            let mut fact = fact.clone();
            fact.scope = self.inner.scope.clone();
            let _replaced = document.upsert_fact(&fact);
        }
        state.document = document;
        state.touch();
        facts.len()
    }

    /// Reads and merges both index documents into the fact set to adopt.
    ///
    /// `USER.md` wins on a duplicate name. A level whose document is missing or empty holds
    /// no importable entry, so the facts of that level that the per-fact files already hold
    /// are added back: an empty document is "no information", never "no facts", which keeps
    /// a truncated or emptied index from deleting every fact file. A name present in both a
    /// document and the fact files is taken from the document (the hand edit wins).
    pub(super) async fn read_index_documents(&self) -> Result<Vec<CodingFact>> {
        let documents = self.index_documents().await?;
        if documents.absent.is_empty() {
            return Ok(documents.facts);
        }
        let (stored, skipped) = self.inner.facts.read_all_report().await?;
        if !skipped.is_empty() {
            tracing::warn!(
                target: TARGET,
                count = skipped.len(),
                "some fact files are not decodable and are left untouched by the import"
            );
        }
        let mut merged: BTreeMap<String, CodingFact> = documents
            .facts
            .into_iter()
            .map(|fact| (fact.name.clone(), fact))
            .collect();
        let mut kept = 0_usize;
        for stored in stored {
            if documents.absent.contains(&level_of(stored.fact_type))
                && !merged.contains_key(&stored.name)
            {
                kept += 1;
                let _previous = merged.insert(stored.name.clone(), stored);
            }
        }
        if kept > 0 {
            tracing::warn!(
                target: TARGET,
                count = kept,
                "an index document is missing or empty; keeping the fact files of that level \
                 instead of treating them as deleted"
            );
        }
        Ok(merged.into_values().collect())
    }

    /// Reads both index documents and reports which levels were missing or empty.
    async fn index_documents(&self) -> Result<IndexDocuments> {
        let mut merged: BTreeMap<String, CodingFact> = BTreeMap::new();
        let mut absent: Vec<IndexLevel> = Vec::new();
        for level in IndexLevel::ALL {
            let path = self.level_path(level);
            let text = match tokio::fs::read_to_string(path).await {
                Ok(text) => text,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    absent.push(level);
                    continue;
                }
                Err(error) => {
                    return Err(Error::external(error)
                        .with_context(format!("reading the index document {}", path.display())));
                }
            };
            if text.trim().is_empty() {
                absent.push(level);
                continue;
            }
            let facts = parse_level(&text)?;
            for mut fact in facts {
                fact.scope = self.inner.scope.clone();
                if level == IndexLevel::User && merged.contains_key(&fact.name) {
                    tracing::warn!(
                        target: TARGET,
                        name = %fact.name,
                        "a fact name appears in both index documents; USER.md wins"
                    );
                }
                let _previous = merged.insert(fact.name.clone(), fact);
            }
        }
        Ok(IndexDocuments {
            facts: merged.into_values().collect(),
            absent,
        })
    }

    /// Renders both index documents from `facts`, writing each atomically.
    async fn render_index_documents(&self, facts: &[CodingFact]) -> Result<()> {
        for level in IndexLevel::ALL {
            let rendered = render_level(facts, level);
            let document = MarkdownMemory::parse(&rendered)?;
            write_document(self.level_path(level), &document).await?;
        }
        Ok(())
    }

    /// Records the stamps of both index documents and of the facts directory.
    async fn record_stamps(&self, state: &mut LocalState) -> Result<()> {
        for (position, level) in IndexLevel::ALL.into_iter().enumerate() {
            state.index_stamps[position] = FileStamp::of(self.level_path(level)).await?;
        }
        state.stamp = state.index_stamps[0];
        state.facts_stamp = Some(self.inner.facts.stamp().await?);
        Ok(())
    }

    /// Whether either recorded index-document stamp differs from the file on disk.
    async fn index_documents_changed(&self, state: &LocalState) -> Result<bool> {
        for (position, level) in IndexLevel::ALL.into_iter().enumerate() {
            if FileStamp::of(self.level_path(level)).await? != state.index_stamps[position] {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// The path of the document `level` is rendered into.
    fn level_path(&self, level: IndexLevel) -> &Path {
        match level {
            IndexLevel::Memory => &self.inner.memory_path,
            IndexLevel::User => &self.inner.user_path,
        }
    }

    /// Rejects the per-fact APIs when the layout is disabled.
    fn require_fact_files(&self) -> Result<()> {
        if self.inner.fact_files {
            return Ok(());
        }
        Err(Error::unsupported(
            "per-fact files are disabled; open the store with `LocalCodingMemoryBuilder::fact_files(true)`",
        ))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use std::sync::Arc;

    use openjiuwen_retrieval::embedding::Embedding;
    use tempfile::TempDir;

    use openjiuwen_common::error::Error as MemoryError;

    use crate::config::CodingMemoryConfig;
    use crate::lite::fact::CodingFact;
    use crate::lite::facts::file_name_for;
    use crate::lite::markdown::MarkdownMemory;
    use crate::lite::CodingFactType;
    use crate::record::{MemoryScope, SearchMode, SearchOptions};

    use super::super::embedding::test_support::CountingEmbedding;
    use super::super::LocalCodingMemory;

    /// The scope every test store and test fact uses.
    fn scope() -> MemoryScope {
        MemoryScope::new("alice", "ws")
    }

    /// A fact of [`scope`] named `name` with `content` as its body.
    fn fact(name: &str, content: &str, fact_type: CodingFactType) -> CodingFact {
        CodingFact::new(&scope(), name, content).with_type(fact_type)
    }

    /// Opens a fact-file store rooted at `temp`.
    async fn open(temp: &TempDir) -> LocalCodingMemory {
        LocalCodingMemory::builder(temp.path().to_path_buf(), CodingMemoryConfig::default())
            .scope(scope())
            .fact_files(true)
            .open()
            .await
            .expect("the fact-file store opens")
    }

    /// Opens a fact-file store rooted at `temp` with `counter` as its embedding model.
    async fn open_embedded(temp: &TempDir, counter: &Arc<CountingEmbedding>) -> LocalCodingMemory {
        let embedding: Arc<dyn Embedding> = counter.clone();
        LocalCodingMemory::builder(temp.path().to_path_buf(), CodingMemoryConfig::default())
            .scope(scope())
            .fact_files(true)
            .embedding(embedding)
            .open()
            .await
            .expect("the embedded fact-file store opens")
    }

    /// Reads a file as text.
    fn read(path: &std::path::Path) -> String {
        std::fs::read_to_string(path).expect("the file is readable")
    }

    /// The facts of `facts` sorted by name, for order-insensitive comparison.
    fn sorted(mut facts: Vec<CodingFact>) -> Vec<CodingFact> {
        facts.sort_by(|left, right| left.name.cmp(&right.name));
        facts
    }

    /// Seeds a store with one project fact, one user fact, and a rendered index.
    async fn seeded(temp: &TempDir) -> LocalCodingMemory {
        let store = open(temp).await;
        store
            .write(
                fact("build", "cargo build", CodingFactType::Project)
                    .with_description("about build")
                    .with_tags(vec!["build".to_owned()])
                    .with_path("docs/BUILD.md"),
            )
            .await
            .expect("the project fact is written");
        store
            .write(fact("theme", "dark mode", CodingFactType::User))
            .await
            .expect("the user fact is written");
        store
    }

    #[tokio::test]
    async fn export_index_writes_one_file_per_fact_and_splits_the_index() {
        let temp = TempDir::new().expect("temp dir");
        let store = seeded(&temp).await;
        assert!(store.fact_files_enabled());
        assert_eq!(store.facts_dir(), temp.path().join("facts"));
        assert_eq!(store.user_path(), temp.path().join("USER.md"));

        let count = store.export_index().await.expect("the export succeeds");
        assert_eq!(count, 2);
        let build_file = store.facts_dir().join(file_name_for("build"));
        let theme_file = store.facts_dir().join(file_name_for("theme"));
        assert!(build_file.is_file(), "the build fact file exists");
        assert!(theme_file.is_file(), "the theme fact file exists");

        let front_matter = read(&build_file);
        for expected in [
            "name: build",
            "description: about build",
            "type: project",
            "tags:",
            "path: docs/BUILD.md",
            "updated_at:",
        ] {
            assert!(
                front_matter.contains(expected),
                "the fact file is missing {expected:?}: {front_matter}"
            );
        }

        let memory = read(store.memory_path());
        assert!(memory.contains("### build"), "{memory}");
        assert!(!memory.contains("theme"), "{memory}");
        let user = read(&store.user_path());
        assert!(user.contains("### theme"), "{user}");
        assert!(!user.contains("### build"), "{user}");
        assert!(user.contains("# User"), "{user}");
    }

    #[tokio::test]
    async fn a_reopened_store_reads_the_per_fact_files_losslessly() {
        let temp = TempDir::new().expect("temp dir");
        let original = {
            let store = seeded(&temp).await;
            store
                .write(fact("style", "be concise", CodingFactType::Feedback))
                .await
                .expect("the feedback fact is written");
            let count = store.export_index().await.expect("the export succeeds");
            assert_eq!(count, 3);
            store.list().await.expect("the list succeeds")
        };

        let reopened = open(&temp).await;
        let listed = reopened.list().await.expect("the list succeeds");
        assert_eq!(sorted(listed), sorted(original));
        assert!(read(&reopened.user_path()).contains("### theme"));
    }

    #[tokio::test]
    async fn a_hand_edited_index_imports_into_the_fact_files() {
        let temp = TempDir::new().expect("temp dir");
        let store = seeded(&temp).await;
        store.export_index().await.expect("the export succeeds");
        let rendered = read(store.memory_path());
        let edited = rendered
            .replace(
                "- description: about build",
                "- description: about build (edited)",
            )
            .replace("cargo build", "cargo build\nand a zephyr line");
        assert_ne!(edited, rendered, "the edit must change the document");
        std::fs::write(store.memory_path(), &edited).expect("the hand edit is written");

        let count = store.import_index().await.expect("the import succeeds");
        assert_eq!(count, 2);
        let build = store
            .read("build")
            .await
            .expect("the read succeeds")
            .expect("the fact exists");
        assert_eq!(build.description, "about build (edited)");
        assert!(build.content.contains("zephyr"), "{}", build.content);

        let fact_file = read(&store.facts_dir().join(file_name_for("build")));
        assert!(
            fact_file.contains("description: about build (edited)"),
            "{fact_file}"
        );
        assert!(fact_file.contains("zephyr"), "{fact_file}");

        let options = SearchOptions::default().with_mode(SearchMode::Keyword);
        let hits = store
            .search("zephyr", &options)
            .await
            .expect("the search succeeds");
        assert_eq!(hits.len(), 1, "the edited body is searchable");

        store.export_index().await.expect("the export succeeds");
        assert_eq!(read(store.memory_path()), edited);
    }

    #[tokio::test]
    async fn a_per_fact_file_edit_wins_over_the_index() {
        let temp = TempDir::new().expect("temp dir");
        let counter = Arc::new(CountingEmbedding::new());
        let store = open_embedded(&temp, &counter).await;
        store
            .write(fact("build", "cargo build", CodingFactType::Project))
            .await
            .expect("the fact is written");
        store.export_index().await.expect("the export succeeds");

        let path = store.facts_dir().join(file_name_for("build"));
        let original = read(&path);
        let edited = original.replace("cargo build", "cargo build --release");
        assert_ne!(edited, original, "the edit must change the file");
        std::fs::write(&path, &edited).expect("the edit is written");

        let report = store.sync_now().await.expect("the sync succeeds");
        assert!(report.ran);
        assert_eq!(report.facts, 1);
        assert!(report.embedded > 0, "the new body is embedded: {report:?}");

        let build = store
            .read("build")
            .await
            .expect("the read succeeds")
            .expect("the fact exists");
        assert_eq!(build.content, "cargo build --release");
        assert!(read(store.memory_path()).contains("--release"));
    }

    #[tokio::test]
    async fn a_name_present_in_both_indexes_imports_the_user_document() {
        let temp = TempDir::new().expect("temp dir");
        let store = open(&temp).await;
        store
            .write(fact("build", "cargo build", CodingFactType::Project))
            .await
            .expect("the fact is written");
        store.export_index().await.expect("the export succeeds");

        let mut memory = MarkdownMemory::new().with_scope(scope());
        let _replaced = memory.upsert_fact(&fact("theme", "from memory", CodingFactType::Project));
        std::fs::write(store.memory_path(), memory.render()).expect("the hand edit is written");
        let mut user = MarkdownMemory::new().with_title("User").with_scope(scope());
        let _replaced = user.upsert_fact(&fact("theme", "from user", CodingFactType::User));
        std::fs::write(store.user_path(), user.render()).expect("the hand edit is written");

        let count = store.import_index().await.expect("the import succeeds");
        assert_eq!(count, 1);
        let facts = store.list().await.expect("the list succeeds");
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].name, "theme");
        assert_eq!(facts[0].content, "from user");
        assert_eq!(facts[0].fact_type, CodingFactType::User);
        assert!(!read(store.memory_path()).contains("theme"));
    }

    #[tokio::test]
    async fn a_delete_removes_the_fact_file_and_both_index_entries() {
        let temp = TempDir::new().expect("temp dir");
        let store = seeded(&temp).await;
        store.export_index().await.expect("the export succeeds");
        let theme_file = store.facts_dir().join(file_name_for("theme"));
        assert!(theme_file.is_file());

        assert!(store.delete("theme").await.expect("the delete succeeds"));
        assert!(!theme_file.is_file(), "the deleted fact file is removed");
        assert!(!read(store.memory_path()).contains("theme"));
        assert!(!read(&store.user_path()).contains("### theme"));
        assert_eq!(store.list().await.expect("the list succeeds").len(), 1);
        assert!(store.facts_dir().join(file_name_for("build")).is_file());
    }

    #[tokio::test]
    async fn an_emptied_index_document_keeps_the_fact_files() {
        let temp = TempDir::new().expect("temp dir");
        let store = seeded(&temp).await;
        store.export_index().await.expect("the export succeeds");
        let build_file = store.facts_dir().join(file_name_for("build"));
        let theme_file = store.facts_dir().join(file_name_for("theme"));
        assert!(build_file.is_file(), "the build fact file exists");
        assert!(theme_file.is_file(), "the theme fact file exists");

        // A truncated (emptied) document is "no information", not "every fact was deleted".
        std::fs::write(store.memory_path(), "").expect("the truncation is written");
        let facts = store.list().await.expect("the list succeeds");
        assert_eq!(
            sorted(facts).len(),
            2,
            "the fact files survive an empty index"
        );
        assert!(build_file.is_file(), "the build fact file survives");
        assert!(theme_file.is_file(), "the theme fact file survives");
        assert_eq!(store.reload_count(), 1, "the reload happened");

        // The next locked operation restores the projection from the fact files.
        let count = store.export_index().await.expect("the export succeeds");
        assert_eq!(count, 2);
        assert!(read(store.memory_path()).contains("### build"));
        assert!(read(&store.user_path()).contains("### theme"));
        assert!(build_file.is_file() && theme_file.is_file());
    }

    #[tokio::test]
    async fn importing_an_emptied_index_document_keeps_the_fact_files() {
        let temp = TempDir::new().expect("temp dir");
        let store = seeded(&temp).await;
        store.export_index().await.expect("the export succeeds");
        let build_file = store.facts_dir().join(file_name_for("build"));

        std::fs::write(store.memory_path(), "   \n").expect("the truncation is written");
        let count = store.import_index().await.expect("the import succeeds");
        assert_eq!(count, 2, "the kept fact files are imported back");
        assert!(
            build_file.is_file(),
            "the build fact file survives the import"
        );
        assert!(read(store.memory_path()).contains("### build"));
        assert_eq!(store.list().await.expect("the list succeeds").len(), 2);
    }

    #[tokio::test]
    async fn removing_an_entry_from_a_non_empty_index_deletes_the_fact_file() {
        let temp = TempDir::new().expect("temp dir");
        let store = seeded(&temp).await;
        store.export_index().await.expect("the export succeeds");
        let build_file = store.facts_dir().join(file_name_for("build"));
        assert!(build_file.is_file());

        // A non-empty document lists the facts of its level, so dropping the entry is a
        // genuine removal: the fact file goes away and `USER.md` keeps its own fact.
        let rendered = read(store.memory_path());
        let without_build = rendered
            .split("### build")
            .next()
            .expect("the prefix before the entry")
            .to_owned();
        assert!(!without_build.contains("### build"));
        std::fs::write(store.memory_path(), &without_build).expect("the removal is written");

        let count = store.import_index().await.expect("the import succeeds");
        assert_eq!(count, 1);
        assert!(!build_file.is_file(), "the removed fact file is deleted");
        let facts = store.list().await.expect("the list succeeds");
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].name, "theme");
    }

    #[tokio::test]
    async fn an_undecodable_fact_file_is_reported_and_never_deleted() {
        let temp = TempDir::new().expect("temp dir");
        let store = seeded(&temp).await;
        store.export_index().await.expect("the export succeeds");
        let prose = store.facts_dir().join("hand-written.md");
        std::fs::write(&prose, "just prose, not a coding fact\n").expect("the stray file");

        let count = store.export_index().await.expect("the export succeeds");
        assert_eq!(count, 2);
        assert!(prose.is_file(), "an undecodable file is left untouched");

        // The delete path removes only the fact it names, never the stray file.
        assert!(store.delete("build").await.expect("the delete succeeds"));
        assert!(prose.is_file(), "the stray file survives the delete");
        assert_eq!(store.list().await.expect("the list succeeds").len(), 1);
    }

    #[tokio::test]
    async fn the_default_layout_stays_a_single_memory_document() {
        let temp = TempDir::new().expect("temp dir");
        let store =
            LocalCodingMemory::open_in_scope(temp.path(), CodingMemoryConfig::default(), scope())
                .await
                .expect("the default store opens");
        assert!(!store.fact_files_enabled());
        store
            .write(fact("build", "cargo build", CodingFactType::Project))
            .await
            .expect("the fact is written");
        store
            .write(fact("theme", "dark mode", CodingFactType::User))
            .await
            .expect("the user fact is written");

        assert!(
            !temp.path().join("facts").exists(),
            "the default layout creates no fact directory"
        );
        assert!(!store.user_path().exists());
        let memory = read(store.memory_path());
        assert!(memory.contains("### theme"), "{memory}");
        assert!(memory.contains("### build"), "{memory}");

        let error = store
            .export_index()
            .await
            .expect_err("the export is disabled");
        assert!(matches!(error, MemoryError::Unsupported(_)), "{error:?}");
        let error = store
            .import_index()
            .await
            .expect_err("the import is disabled");
        assert!(matches!(error, MemoryError::Unsupported(_)), "{error:?}");
    }
}
