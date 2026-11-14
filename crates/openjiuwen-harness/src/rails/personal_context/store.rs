//! Offline facade over the `PersonalContext` subtree.
//!
//! [`PersonalContextStore`] owns the durable layout under a caller-provided
//! home: the context root, the persisted entry index, the source-metadata
//! directory, the source-link book and the caller-published description. It
//! ingests local directories while deduplicating against the index, answers
//! keyword queries over the indexed text, and keeps the link book reachable
//! from one place.
//!
//! Scope notes:
//! - There is no fetch pipeline and no Markdown page compilation: remote
//!   vendors live behind the `provider` seam and stay `Unsupported`.
//! - The index is the durable entry ledger, rewritten atomically; it is not a
//!   generated page tree.
//! - Publication of `description.md` is caller-driven; the rail reads whatever
//!   the caller publishes here.
//!
//! Every path is derived from `home`, so a caller must pass a real home
//! directory: an empty [`PathBuf`] makes every derived path relative to the
//! process working directory.
use std::path::{Path, PathBuf};

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};

use super::atomic::{temp_path, write_json_atomic};
use super::ingest::{ingest_local_directory, IngestReport, IngestedDocument, LocalDirectoryConfig};
use super::link_book::SourceLinkBook;
use super::search::{search, SearchDocument, SearchHit, MAX_SEARCH_RESULTS};
use super::source::{is_source_id, observed_at_string, SourceMetadata, SourceMetadataStore};

/// Schema version written into [`ContextIndex`] documents.
pub const INDEX_VERSION: u32 = 1;

/// File name of the persisted entry index under the context root.
pub const INDEX_FILENAME: &str = "index.json";

/// File name of the caller-published description under the context root.
pub const DESCRIPTION_FILENAME: &str = "description.md";

/// File name of the persisted source-link book under the context root.
pub const LINK_BOOK_FILENAME: &str = "source_links.json";

/// Directory name of the per-source metadata directory under the context root.
pub const SOURCE_ROOT_DIRNAME: &str = "sources";

/// Default number of results returned by [`PersonalContextStore::search`].
pub const DEFAULT_SEARCH_LIMIT: usize = MAX_SEARCH_RESULTS;

/// Document label used in every index error message.
const INDEX_LABEL: &str = "personal context index";

/// Document label used in every description error message.
const DESCRIPTION_LABEL: &str = "personal context description";

/// One ingested source as persisted in the entry index.
///
/// A local plain-text file has no provider-side revision, so
/// [`ContextEntry::from_document`] uses the content hash as
/// [`ContextEntry::latest_revision`] too: for these sources the digest is the
/// revision identity.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ContextEntry {
    /// Stable `src_<32 hex>` source identifier.
    pub source_id: String,
    /// Service id the source was ingested under.
    pub service: String,
    /// Provider id (`"local_files"` for local scans).
    pub provider: String,
    /// Display title of the source.
    pub title: String,
    /// Normalised locator the source id was derived from.
    pub locator: String,
    /// Path relative to the scanned directory, with `/` separators.
    pub relative_path: String,
    /// SHA-256 hex digest of the indexed content.
    pub content_hash: String,
    /// Size of the source file in bytes.
    pub size_bytes: u64,
    /// Text snippet indexed for search.
    pub text: String,
    /// RFC3339 UTC timestamp of the first observation.
    pub first_seen: String,
    /// RFC3339 UTC timestamp of the latest observation.
    pub last_seen: String,
    /// Provider revision of the latest content; the content hash for local files.
    pub latest_revision: String,
}

impl ContextEntry {
    /// Builds an index entry from a freshly ingested document.
    ///
    /// The entry is stamped with `observed_at` for both `first_seen` and
    /// `last_seen`, and `latest_revision` mirrors `content_hash` because a
    /// local file carries no provider revision.
    #[must_use]
    pub fn from_document(document: &IngestedDocument, observed_at: &str) -> Self {
        Self {
            source_id: document.source_id.clone(),
            service: document.service.clone(),
            provider: document.provider.clone(),
            title: document.title.clone(),
            locator: document.locator.clone(),
            relative_path: document.relative_path.clone(),
            content_hash: document.content_hash.clone(),
            size_bytes: document.size_bytes,
            text: document.text.clone(),
            first_seen: observed_at.to_string(),
            last_seen: observed_at.to_string(),
            latest_revision: document.content_hash.clone(),
        }
    }
}

/// The persisted entry ledger.
///
/// A missing `version` falls back to [`INDEX_VERSION`] through the struct's
/// [`Default`] implementation, so documents written by this module and
/// partial documents both decode.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ContextIndex {
    /// Schema version of the document.
    pub version: u32,
    /// Entries in insertion order; [`PersonalContextStore::entries`] sorts them.
    pub entries: Vec<ContextEntry>,
}

impl Default for ContextIndex {
    fn default() -> Self {
        Self {
            version: INDEX_VERSION,
            entries: Vec::new(),
        }
    }
}

impl ContextIndex {
    /// Returns the index with `entry` appended.
    #[must_use]
    pub fn with_entry(mut self, entry: ContextEntry) -> Self {
        self.entries.push(entry);
        self
    }
}

/// Filesystem facade and durable entry ledger of the `PersonalContext` subtree.
///
/// The store holds only the home path, so several handles can point at the same
/// directory; every operation derives its paths from `home` and re-reads the
/// index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PersonalContextStore {
    home: PathBuf,
}

impl PersonalContextStore {
    /// Creates a store rooted at `home` without touching the filesystem.
    ///
    /// Callers pass a real home directory; an empty [`PathBuf`] yields relative
    /// derived paths.
    #[must_use]
    pub fn new(home: impl Into<PathBuf>) -> Self {
        Self { home: home.into() }
    }

    /// Returns the home directory every path is derived from.
    #[must_use]
    pub fn home(&self) -> &Path {
        &self.home
    }

    /// Returns `<home>/workspace/context`.
    #[must_use]
    pub fn context_root(&self) -> PathBuf {
        self.home.join("workspace").join("context")
    }

    /// Returns the caller-published description path.
    #[must_use]
    pub fn description_path(&self) -> PathBuf {
        self.context_root().join(DESCRIPTION_FILENAME)
    }

    /// Returns the persisted index path.
    #[must_use]
    pub fn index_path(&self) -> PathBuf {
        self.context_root().join(INDEX_FILENAME)
    }

    /// Returns the per-source metadata directory path.
    #[must_use]
    pub fn source_root(&self) -> PathBuf {
        self.context_root().join(SOURCE_ROOT_DIRNAME)
    }

    /// Returns the persisted source-link book path.
    #[must_use]
    pub fn link_book_path(&self) -> PathBuf {
        self.context_root().join(LINK_BOOK_FILENAME)
    }

    /// Returns a metadata store over [`PersonalContextStore::source_root`].
    #[must_use]
    pub fn metadata_store(&self) -> SourceMetadataStore {
        SourceMetadataStore::new(self.source_root())
    }

    /// Loads every indexed entry, sorted by `relative_path` then `source_id`.
    ///
    /// A missing index is an empty ledger.
    ///
    /// # Errors
    ///
    /// `Validation` when the document is malformed or its `version` is newer
    /// than [`INDEX_VERSION`]; `Config` when the index cannot be read.
    pub fn entries(&self) -> Result<Vec<ContextEntry>> {
        let mut entries = self.load_index()?.entries;
        sort_entries(&mut entries);
        Ok(entries)
    }

    /// Looks one entry up by `source_id`.
    ///
    /// # Errors
    ///
    /// `Validation` when `source_id` is not a well-formed source id; index
    /// errors are those of [`PersonalContextStore::entries`].
    pub fn entry(&self, source_id: &str) -> Result<Option<ContextEntry>> {
        if !is_source_id(source_id) {
            return Err(Error::validation("personal context source id is invalid"));
        }
        Ok(self
            .entries()?
            .into_iter()
            .find(|entry| entry.source_id == source_id))
    }

    /// Runs a keyword query over the indexed entries.
    ///
    /// `limit` is capped at [`MAX_SEARCH_RESULTS`]; a blank query yields an
    /// empty result rather than an error.
    ///
    /// # Errors
    ///
    /// Only index-loading errors, as described on
    /// [`PersonalContextStore::entries`].
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchHit>> {
        let documents: Vec<SearchDocument> = self
            .entries()?
            .into_iter()
            .map(|entry| {
                SearchDocument::new(
                    entry.source_id,
                    entry.relative_path,
                    entry.title,
                    entry.text,
                )
            })
            .collect();
        Ok(search(&documents, query, limit))
    }

    /// Ingests local directories against the persisted index.
    ///
    /// Every produced document is classified by `source_id`: absent from the
    /// index is `ingested`, a matching `content_hash` is `unchanged`, anything
    /// else is `updated`. Each document is also upserted into the metadata
    /// store so `first_seen` survives across runs, and the index is rewritten
    /// atomically when a document (or an observation timestamp) changed.
    ///
    /// The returned report aggregates the per-scan counters and warnings; its
    /// `documents` holds only this run's ingested and updated documents,
    /// sorted by `relative_path`, because unchanged documents are excluded.
    ///
    /// # Errors
    ///
    /// `Validation` for a bad scan configuration or a malformed index, and any
    /// error the ingest, metadata or atomic-write layers return.
    pub fn ingest_local(
        &self,
        configs: &[LocalDirectoryConfig],
        observed_at_ms: i64,
    ) -> Result<IngestReport> {
        let observed_at = observed_at_string(observed_at_ms);
        let mut index = self.load_index()?;
        let metadata_store = self.metadata_store();
        let mut aggregate = IngestReport::default();
        let mut dirty = false;
        let mut changed_documents: Vec<IngestedDocument> = Vec::new();

        for config in configs {
            let report = ingest_local_directory(config)?;
            aggregate.scanned += report.scanned;
            aggregate.skipped += report.skipped;
            aggregate.failed += report.failed;

            for document in &report.documents {
                match index
                    .entries
                    .iter_mut()
                    .find(|entry| entry.source_id == document.source_id)
                {
                    None => {
                        index
                            .entries
                            .push(ContextEntry::from_document(document, &observed_at));
                        aggregate.ingested += 1;
                        dirty = true;
                        changed_documents.push(document.clone());
                    }
                    Some(entry) if entry.content_hash == document.content_hash => {
                        aggregate.unchanged += 1;
                        if entry.last_seen != observed_at {
                            entry.last_seen.clone_from(&observed_at);
                            dirty = true;
                        }
                    }
                    Some(entry) => {
                        let mut replacement = ContextEntry::from_document(document, &observed_at);
                        replacement.first_seen = if entry.first_seen.is_empty() {
                            observed_at.clone()
                        } else {
                            entry.first_seen.clone()
                        };
                        *entry = replacement;
                        aggregate.updated += 1;
                        dirty = true;
                        changed_documents.push(document.clone());
                    }
                }

                metadata_store.upsert(SourceMetadata {
                    source_id: document.source_id.clone(),
                    source_type: document.provider.clone(),
                    title: document.title.clone(),
                    locator: document.locator.clone(),
                    provider: document.provider.clone(),
                    service: document.service.clone(),
                    first_seen: observed_at.clone(),
                    last_seen: observed_at.clone(),
                    latest_revision: document.content_hash.clone(),
                    latest_hash: document.content_hash.clone(),
                })?;
            }

            aggregate.warnings.extend(report.warnings);
        }

        if dirty {
            sort_entries(&mut index.entries);
            write_json_atomic(&self.index_path(), &index, INDEX_LABEL)?;
        }

        changed_documents.sort_by(|left, right| left.relative_path.cmp(&right.relative_path));
        aggregate.documents = changed_documents;
        Ok(aggregate)
    }

    /// Loads the source-link book, returning an empty book when absent.
    ///
    /// # Errors
    ///
    /// The errors of [`SourceLinkBook::load`].
    pub fn link_book(&self) -> Result<SourceLinkBook> {
        SourceLinkBook::load(&self.link_book_path())
    }

    /// Persists `book` atomically at [`PersonalContextStore::link_book_path`].
    ///
    /// # Errors
    ///
    /// The errors of [`SourceLinkBook::save`].
    pub fn save_link_book(&self, book: &SourceLinkBook) -> Result<()> {
        book.save(&self.link_book_path())
    }

    /// Publishes `description.md` atomically as UTF-8.
    ///
    /// The text goes to `description.md.tmp` first and is renamed over the
    /// destination, so a reader never sees a partial description.
    ///
    /// # Errors
    ///
    /// `Validation` for an empty or whitespace-only description; `Config` when
    /// the directory, temporary file or rename cannot be handled.
    pub fn write_description(&self, text: &str) -> Result<()> {
        if text.trim().is_empty() {
            return Err(Error::validation("personal context description is empty"));
        }
        let path = self.description_path();
        if let Some(dir) = path.parent() {
            if !dir.as_os_str().is_empty() {
                std::fs::create_dir_all(dir).map_err(|error| {
                    Error::config(format!(
                        "failed to create {DESCRIPTION_LABEL} directory {}: {error}",
                        dir.display()
                    ))
                })?;
            }
        }
        let temp = temp_path(&path);
        std::fs::write(&temp, text.as_bytes()).map_err(|error| {
            Error::config(format!(
                "failed to write {DESCRIPTION_LABEL} file {}: {error}",
                temp.display()
            ))
        })?;
        std::fs::rename(&temp, &path).map_err(|error| {
            Error::config(format!(
                "failed to replace {DESCRIPTION_LABEL} file {}: {error}",
                path.display()
            ))
        })
    }

    /// Reads and version-checks the persisted index.
    fn load_index(&self) -> Result<ContextIndex> {
        let path = self.index_path();
        let Some(index) = super::atomic::read_json_opt::<ContextIndex>(&path, INDEX_LABEL)? else {
            return Ok(ContextIndex::default());
        };
        if index.version > INDEX_VERSION {
            return Err(Error::validation(format!(
                "{INDEX_LABEL} {} has unsupported version {}",
                path.display(),
                index.version
            )));
        }
        Ok(index)
    }
}

/// Sorts entries by `relative_path` then `source_id`.
fn sort_entries(entries: &mut [ContextEntry]) {
    entries.sort_by(|left, right| {
        left.relative_path
            .cmp(&right.relative_path)
            .then_with(|| left.source_id.cmp(&right.source_id))
    });
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use openjiuwen_common::error::ErrorKind;

    use super::super::ingest::LocalDirectoryConfig;
    use super::super::source::observed_at_string;
    use super::*;

    const T0: i64 = 0;
    const T1: i64 = 86_400_000;
    const T2: i64 = 172_800_000;

    fn home_with_docs() -> tempfile::TempDir {
        let home = tempfile::tempdir().unwrap();
        let docs = home.path().join("docs");
        std::fs::create_dir_all(&docs).unwrap();
        std::fs::write(docs.join("alpha.md"), "# Alpha\n\nalpha keyword body\n").unwrap();
        std::fs::write(docs.join("beta.md"), "# Beta\n\nbeta body\n").unwrap();
        home
    }

    fn config(home: &tempfile::TempDir) -> LocalDirectoryConfig {
        LocalDirectoryConfig::new("notes", home.path().join("docs"))
    }

    fn text_of(store: &PersonalContextStore, path: &str) -> String {
        store
            .entries()
            .unwrap()
            .into_iter()
            .find(|entry| entry.relative_path == path)
            .map(|entry| entry.text)
            .unwrap()
    }

    #[test]
    fn first_ingest_indexes_every_document() {
        let home = home_with_docs();
        let store = PersonalContextStore::new(home.path());
        let report = store.ingest_local(&[config(&home)], T0).unwrap();

        assert_eq!(report.ingested, 2);
        assert_eq!(report.unchanged, 0);
        assert_eq!(report.updated, 0);
        assert_eq!(report.documents.len(), 2);
        assert_eq!(report.documents[0].relative_path, "alpha.md");

        let entries = store.entries().unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].relative_path, "alpha.md");
        assert_eq!(entries[1].relative_path, "beta.md");
        assert!(entries[0].source_id.starts_with("src_"));
        assert_eq!(entries[0].first_seen, observed_at_string(T0));
        assert_eq!(entries[0].last_seen, observed_at_string(T0));
        assert_eq!(entries[0].latest_revision, entries[0].content_hash);

        let hits = store.search("keyword", DEFAULT_SEARCH_LIMIT).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].path, "alpha.md");
        assert_eq!(hits[0].source_id, entries[0].source_id);
    }

    #[test]
    fn reingest_of_unchanged_files_keeps_entries_and_advances_last_seen() {
        let home = home_with_docs();
        let store = PersonalContextStore::new(home.path());
        store.ingest_local(&[config(&home)], T0).unwrap();
        let before = store.entries().unwrap();

        let report = store.ingest_local(&[config(&home)], T1).unwrap();
        assert_eq!(report.ingested, 0);
        assert_eq!(report.unchanged, 2);
        assert_eq!(report.updated, 0);
        assert!(report.documents.is_empty());

        let after = store.entries().unwrap();
        assert_eq!(after.len(), 2);
        for (old, new) in before.iter().zip(after.iter()) {
            assert_eq!(new.source_id, old.source_id);
            assert_eq!(new.first_seen, old.first_seen);
            assert_eq!(new.first_seen, observed_at_string(T0));
            assert_eq!(new.last_seen, observed_at_string(T1));
            assert_eq!(new.text, old.text);
            assert_eq!(new.content_hash, old.content_hash);
        }
    }

    #[test]
    fn changed_file_is_updated_and_keeps_first_seen() {
        let home = home_with_docs();
        let store = PersonalContextStore::new(home.path());
        store.ingest_local(&[config(&home)], T0).unwrap();
        let original = store
            .entry(&store.entries().unwrap()[0].source_id)
            .unwrap()
            .unwrap();

        std::fs::write(
            home.path().join("docs").join("alpha.md"),
            "# Alpha again\n\nrewritten alpha text\n",
        )
        .unwrap();
        let report = store.ingest_local(&[config(&home)], T2).unwrap();

        assert_eq!(report.updated, 1);
        assert_eq!(report.unchanged, 1);
        assert_eq!(report.ingested, 0);
        assert_eq!(report.documents.len(), 1);
        assert_eq!(report.documents[0].relative_path, "alpha.md");

        let updated = store.entry(&original.source_id).unwrap().unwrap();
        assert_eq!(updated.first_seen, observed_at_string(T0));
        assert_eq!(updated.last_seen, observed_at_string(T2));
        assert_ne!(updated.content_hash, original.content_hash);
        assert_eq!(updated.latest_revision, updated.content_hash);
        assert_ne!(updated.text, original.text);
        assert!(updated.text.contains("rewritten alpha text"));
        assert_eq!(text_of(&store, "beta.md"), "# Beta\n\nbeta body");
    }

    #[test]
    fn entry_lookup_and_invalid_ids() {
        let home = home_with_docs();
        let store = PersonalContextStore::new(home.path());
        store.ingest_local(&[config(&home)], T0).unwrap();
        let entries = store.entries().unwrap();

        let found = store.entry(&entries[0].source_id).unwrap().unwrap();
        assert_eq!(found, entries[0]);

        let error = store.entry("not-a-source").unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);

        let absent = store.entry("src_0123456789abcdef0123456789abcdef").unwrap();
        assert!(absent.is_none());
    }

    #[test]
    fn missing_index_is_an_empty_ledger() {
        let home = tempfile::tempdir().unwrap();
        let store = PersonalContextStore::new(home.path());

        assert!(store.entries().unwrap().is_empty());
        assert!(store
            .search("anything", DEFAULT_SEARCH_LIMIT)
            .unwrap()
            .is_empty());
        assert!(store.link_book().unwrap().is_empty());
    }

    #[test]
    fn malformed_index_is_a_validation_error() {
        let home = tempfile::tempdir().unwrap();
        let store = PersonalContextStore::new(home.path());
        std::fs::create_dir_all(store.context_root()).unwrap();
        std::fs::write(store.index_path(), "{not json").unwrap();

        let error = store.entries().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("personal context index"));
    }

    #[test]
    fn unsupported_index_version_is_a_validation_error() {
        let home = tempfile::tempdir().unwrap();
        let store = PersonalContextStore::new(home.path());
        std::fs::create_dir_all(store.context_root()).unwrap();
        std::fs::write(store.index_path(), "{\"version\": 99, \"entries\": []}").unwrap();

        let error = store.entries().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("99"));
    }

    #[test]
    fn blank_query_yields_no_hits() {
        let home = home_with_docs();
        let store = PersonalContextStore::new(home.path());
        store.ingest_local(&[config(&home)], T0).unwrap();

        assert!(store
            .search("   ", DEFAULT_SEARCH_LIMIT)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn link_book_round_trips_through_the_store() {
        let home = tempfile::tempdir().unwrap();
        let store = PersonalContextStore::new(home.path());
        let mut book = store.link_book().unwrap();
        book.register(
            "https://example.com/page",
            "https://example.com/target",
            "Target",
        )
        .unwrap();
        store.save_link_book(&book).unwrap();

        let loaded = store.link_book().unwrap();
        assert_eq!(loaded, book);
        assert_eq!(loaded.len(), 1);
        assert!(!store.link_book_path().with_extension("json.tmp").exists());
    }

    #[test]
    fn description_round_trips_and_rejects_blank_text() {
        let home = tempfile::tempdir().unwrap();
        let store = PersonalContextStore::new(home.path());

        store
            .write_description("# Personal context\n\nNotes.\n")
            .unwrap();
        let written = std::fs::read_to_string(store.description_path()).unwrap();
        assert_eq!(written, "# Personal context\n\nNotes.\n");
        assert!(!temp_path(&store.description_path()).exists());

        let error = store.write_description("  \n\t ").unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error
            .to_string()
            .contains("personal context description is empty"));
        assert_eq!(
            std::fs::read_to_string(store.description_path()).unwrap(),
            "# Personal context\n\nNotes.\n"
        );
    }

    #[test]
    fn metadata_store_tracks_both_sources_across_runs() {
        let home = home_with_docs();
        let store = PersonalContextStore::new(home.path());
        store.ingest_local(&[config(&home)], T0).unwrap();
        store.ingest_local(&[config(&home)], T1).unwrap();

        let entries = store.entries().unwrap();
        let metadata = store.metadata_store().list().unwrap();
        assert_eq!(metadata.len(), 2);
        let ids: Vec<&str> = metadata
            .iter()
            .map(|record| record.source_id.as_str())
            .collect();
        let mut expected: Vec<&str> = entries
            .iter()
            .map(|entry| entry.source_id.as_str())
            .collect();
        expected.sort_unstable();
        assert_eq!(ids, expected);
        for record in &metadata {
            assert_eq!(record.first_seen, observed_at_string(T0));
            assert_eq!(record.last_seen, observed_at_string(T1));
            assert_eq!(record.source_type, "local_files");
            assert_eq!(record.provider, "local_files");
            assert_eq!(record.service, "notes");
            assert_eq!(record.latest_revision, record.latest_hash);
            assert_eq!(record.latest_hash.len(), 64);
        }
    }
}
