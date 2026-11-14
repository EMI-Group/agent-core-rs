//! The incremental trace store: parsed records cached next to the session source.
//!
//! The layout mirrors Python's `experience/trace/store.py`:
//!
//! ```text
//! <root>/sessions/<session_id>/{metadata.json,history.json}   // external input
//! <root>/trace_store/records.jsonl                            // parsed records, one per line
//! <root>/trace_store/processed_index.json                     // {"processed_ids": [...]}
//! ```
//!
//! [`TraceStore::parse_all_sessions`] parses only sessions missing from the processed
//! index, appends their records, and then records the session ids as processed. Two
//! mechanisms keep that idempotent: the index filters whole sessions, and a record whose
//! `trace_id` is already in `records.jsonl` is skipped, so a crash between the append and
//! the index write cannot duplicate a record on the next run.
//!
//! Both files are replaced atomically (a `<file>.tmp` sibling plus a rename), so a reader
//! observes either the previous store or the complete new one. The records file keeps its
//! insertion order on disk; newly appended records are ordered by
//! `(timestamp, trace_id)` before the write.
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use serde_json::Value;
use tracing::warn;

use crate::error::{Error, Result};
use crate::experience::store::write_atomic;
use crate::experience::trace::models::TraceRecord;
use crate::experience::trace::parser::{list_session_ids, parse_session, sort_records};

/// The fixed directory names under a [`TraceStore`] root.
pub const SESSIONS_DIR: &str = "sessions";

/// The derived cache directory name, a sibling of [`SESSIONS_DIR`].
pub const TRACE_STORE_DIR: &str = "trace_store";

/// The JSONL file holding every parsed record.
pub const RECORDS_FILE: &str = "records.jsonl";

/// The JSON file tracking which sessions were already parsed.
pub const PROCESSED_INDEX_FILE: &str = "processed_index.json";

/// The paths of one trace workspace.
///
/// `root` holds the external `sessions/` tree and the derived `trace_store/` cache; the
/// two live side by side so a source scan and its parsed cache share one subtree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionPaths {
    /// The workspace root.
    root: PathBuf,
}

impl SessionPaths {
    /// Builds the path set for `root` without touching the filesystem.
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// The workspace root.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The directory whose subdirectories are sessions (`<root>/sessions`).
    #[must_use]
    pub fn sessions_dir(&self) -> PathBuf {
        self.root.join(SESSIONS_DIR)
    }

    /// One session directory (`<root>/sessions/<session_id>`).
    #[must_use]
    pub fn session_dir(&self, session_id: &str) -> PathBuf {
        self.sessions_dir().join(session_id)
    }

    /// The derived cache directory (`<root>/trace_store`).
    #[must_use]
    pub fn trace_store_dir(&self) -> PathBuf {
        self.root.join(TRACE_STORE_DIR)
    }

    /// The parsed-record file (`<root>/trace_store/records.jsonl`).
    #[must_use]
    pub fn records_path(&self) -> PathBuf {
        self.trace_store_dir().join(RECORDS_FILE)
    }

    /// The processed-session index (`<root>/trace_store/processed_index.json`).
    #[must_use]
    pub fn processed_index_path(&self) -> PathBuf {
        self.trace_store_dir().join(PROCESSED_INDEX_FILE)
    }
}

/// The incremental parsed-trace cache.
///
/// The store owns no in-memory state: every call reads the files it needs, so two stores
/// over the same root see the same records.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceStore {
    /// The paths this store reads and writes.
    paths: SessionPaths,
}

impl TraceStore {
    /// Builds a store for `root` without touching the filesystem.
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            paths: SessionPaths::new(root),
        }
    }

    /// Builds a store for `root` and creates the derived cache directory.
    ///
    /// The session directory is *not* created: it is external input, and a missing one
    /// simply means there is nothing to parse yet.
    ///
    /// # Errors
    ///
    /// Returns [`Error::External`] when the cache directory cannot be created.
    pub async fn open(root: impl Into<PathBuf>) -> Result<Self> {
        let store = Self::new(root);
        tokio::fs::create_dir_all(store.paths.trace_store_dir())
            .await
            .map_err(Error::external)?;
        Ok(store)
    }

    /// The workspace root.
    #[must_use]
    pub fn root(&self) -> &Path {
        self.paths.root()
    }

    /// The paths of this store.
    #[must_use]
    pub fn paths(&self) -> &SessionPaths {
        &self.paths
    }

    /// The derived cache directory.
    #[must_use]
    pub fn trace_store_dir(&self) -> PathBuf {
        self.paths.trace_store_dir()
    }

    /// The parsed-record file.
    #[must_use]
    pub fn records_path(&self) -> PathBuf {
        self.paths.records_path()
    }

    /// The processed-session index.
    #[must_use]
    pub fn processed_index_path(&self) -> PathBuf {
        self.paths.processed_index_path()
    }

    /// Parses every session not yet in the processed index and appends its records.
    ///
    /// Returns the records actually appended: a record whose `trace_id` is already in the
    /// store is dropped, and a session that yields no records still becomes processed (so
    /// a later run does not keep rescanning it). Re-running with no new sessions answers
    /// an empty list and leaves both files untouched.
    ///
    /// Unreadable sessions are skipped with a warning; the scan itself never aborts for
    /// one bad session.
    ///
    /// # Errors
    ///
    /// Returns [`Error::External`] when the index cannot be read or a write fails, and the
    /// store directory cannot be created.
    pub async fn parse_all_sessions(&self) -> Result<Vec<TraceRecord>> {
        let processed = self.load_processed_index().await?;
        let session_ids = list_session_ids(&self.paths).await?;
        let new_ids: Vec<String> = session_ids
            .into_iter()
            .filter(|session_id| !processed.contains(session_id))
            .collect();
        if new_ids.is_empty() {
            return Ok(Vec::new());
        }

        let mut parsed = Vec::new();
        for session_id in &new_ids {
            match parse_session(session_id, &self.paths).await {
                Ok(mut records) => parsed.append(&mut records),
                Err(error) => {
                    warn!(
                        session_id = %session_id,
                        %error,
                        "skipping unreadable trace session"
                    );
                }
            }
        }
        sort_records(&mut parsed);

        let mut seen = self.existing_trace_ids().await?;
        let mut fresh = Vec::new();
        for record in parsed {
            if record.trace_id.is_empty() {
                continue;
            }
            if seen.insert(record.trace_id.clone()) {
                fresh.push(record);
            }
        }
        if !fresh.is_empty() {
            self.append_records(&fresh).await?;
        }

        // The index is written after the records, so an interrupted run re-parses the
        // session and the trace-id dedupe above absorbs the duplicates.
        let mut processed_all = processed;
        processed_all.extend(new_ids);
        self.save_processed_index(&processed_all).await?;
        Ok(fresh)
    }

    /// Loads every stored record in file order (insertion order).
    ///
    /// Blank, malformed, and trace-id-less lines are warned about and skipped, so one bad
    /// line does not discard the rest of the cache.
    ///
    /// # Errors
    ///
    /// Returns [`Error::External`] when the records file cannot be read.
    pub async fn load_all_records(&self) -> Result<Vec<TraceRecord>> {
        let path = self.paths.records_path();
        let text = match tokio::fs::read_to_string(&path).await {
            Ok(text) => text,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(Error::external(error)),
        };
        let mut records = Vec::new();
        for (index, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            match serde_json::from_str::<TraceRecord>(line) {
                Ok(record) if !record.trace_id.is_empty() => records.push(record),
                Ok(_) => warn!(line = index + 1, "skipping trace record without a trace_id"),
                Err(error) => warn!(
                    line = index + 1,
                    %error,
                    "skipping malformed trace record"
                ),
            }
        }
        Ok(records)
    }

    /// Removes every stored record and the processed index, keeping the cache directory.
    ///
    /// # Errors
    ///
    /// Returns [`Error::External`] when a file exists but cannot be removed.
    pub async fn clear(&self) -> Result<()> {
        for path in [self.paths.records_path(), self.paths.processed_index_path()] {
            match tokio::fs::remove_file(&path).await {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(Error::external(error)),
            }
        }
        Ok(())
    }

    /// Reads the processed-session ids, degrading to an empty set on a corrupt index.
    async fn load_processed_index(&self) -> Result<BTreeSet<String>> {
        let path = self.paths.processed_index_path();
        let text = match tokio::fs::read_to_string(&path).await {
            Ok(text) => text,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(BTreeSet::new());
            }
            Err(error) => return Err(Error::external(error)),
        };
        match serde_json::from_str::<Value>(&text) {
            Ok(value) => Ok(value
                .get("processed_ids")
                .and_then(Value::as_array)
                .map(|ids| {
                    ids.iter()
                        .filter_map(Value::as_str)
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or_default()),
            Err(error) => {
                warn!(path = %path.display(), %error, "reading a corrupt processed index as empty");
                Ok(BTreeSet::new())
            }
        }
    }

    /// Writes the processed-session index atomically.
    async fn save_processed_index(&self, ids: &BTreeSet<String>) -> Result<()> {
        let document = serde_json::json!({ "processed_ids": ids.iter().collect::<Vec<_>>() });
        let bytes = serde_json::to_vec_pretty(&document).map_err(Error::external)?;
        write_atomic(&self.paths.processed_index_path(), &bytes).await
    }

    /// Returns the trace ids already present in the records file.
    ///
    /// Malformed lines are ignored here (they are reported by [`Self::load_all_records`]).
    async fn existing_trace_ids(&self) -> Result<BTreeSet<String>> {
        let path = self.paths.records_path();
        let text = match tokio::fs::read_to_string(&path).await {
            Ok(text) => text,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(BTreeSet::new());
            }
            Err(error) => return Err(Error::external(error)),
        };
        let mut ids = BTreeSet::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Ok(Value::Object(map)) = serde_json::from_str::<Value>(line) {
                if let Some(Value::String(trace_id)) = map.get("trace_id") {
                    ids.insert(trace_id.clone());
                }
            }
        }
        Ok(ids)
    }

    /// Appends `records` to the records file through one atomic replacement.
    ///
    /// Existing bytes are preserved verbatim (insertion order stays stable); only the new
    /// lines are added.
    async fn append_records(&self, records: &[TraceRecord]) -> Result<()> {
        let path = self.paths.records_path();
        let mut text = match tokio::fs::read_to_string(&path).await {
            Ok(text) => text,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => String::new(),
            Err(error) => return Err(Error::external(error)),
        };
        if !text.is_empty() && !text.ends_with('\n') {
            text.push('\n');
        }
        for record in records {
            let line = serde_json::to_string(record).map_err(Error::external)?;
            text.push_str(&line);
            text.push('\n');
        }
        write_atomic(&path, text.as_bytes()).await
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn paths_share_one_workspace_root() {
        let paths = SessionPaths::new("workspace");
        assert_eq!(paths.root(), Path::new("workspace"));
        assert_eq!(paths.sessions_dir(), Path::new("workspace/sessions"));
        assert_eq!(
            paths.session_dir("s-1"),
            Path::new("workspace/sessions/s-1")
        );
        assert_eq!(paths.trace_store_dir(), Path::new("workspace/trace_store"));
        assert_eq!(
            paths.records_path(),
            Path::new("workspace/trace_store/records.jsonl")
        );
        assert_eq!(
            paths.processed_index_path(),
            Path::new("workspace/trace_store/processed_index.json")
        );
    }

    #[test]
    fn new_never_touches_the_filesystem() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().join("does-not-exist");
        let store = TraceStore::new(&root);
        assert_eq!(store.root(), root.as_path());
        assert!(!root.exists());
    }
}
