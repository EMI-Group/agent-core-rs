//! Persisted worktree registry (`.worktree/index.json`).
//!
//! `git worktree list` is the source of truth for what exists, but it carries no
//! creation metadata: branch base, owner, timestamp. This module keeps that
//! metadata in a small JSON document under the workspace's `.worktree/`
//! directory (the same directory [`crate::workspace::Workspace`] uses for
//! worktree links), so `list()` and `status()` can report worktrees created by
//! an earlier process.
//!
//! The document is written atomically (sibling temporary file + rename), so a
//! concurrent reader never observes a half-written registry.
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use openjiuwen_common::error::{Error, Result};

use super::models::{WorktreeRecord, WorktreeState};

/// Directory holding the registry, relative to the workspace root.
pub const INDEX_DIR: &str = ".worktree";

/// File name of the registry inside [`INDEX_DIR`].
pub const INDEX_FILE: &str = "index.json";

/// Current registry document version.
pub const INDEX_VERSION: u32 = 1;

/// Returns `<workspace_root>/.worktree/index.json`.
#[must_use]
pub fn index_path(workspace_root: &Path) -> PathBuf {
    workspace_root.join(INDEX_DIR).join(INDEX_FILE)
}

/// The persisted registry document.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct WorktreeIndex {
    /// Document version; see [`INDEX_VERSION`].
    pub version: u32,
    /// Records, one per worktree this helper created.
    pub records: Vec<WorktreeRecord>,
}

impl Default for WorktreeIndex {
    fn default() -> Self {
        Self {
            version: INDEX_VERSION,
            records: Vec::new(),
        }
    }
}

impl WorktreeIndex {
    /// Builds an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Loads the registry at `path`.
    ///
    /// A missing file yields an empty registry; malformed JSON or an unreadable
    /// file is a validation error.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Validation`] when the file
    /// exists but cannot be read or parsed.
    pub fn load(path: &Path) -> Result<Self> {
        match std::fs::read(path) {
            Ok(bytes) => serde_json::from_slice::<Self>(&bytes).map_err(|error| {
                Error::validation(format!(
                    "invalid worktree index `{}`: {error}",
                    path.display()
                ))
            }),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(error) => Err(Error::validation(format!(
                "cannot read worktree index `{}`: {error}",
                path.display()
            ))),
        }
    }

    /// Writes the registry to `path` atomically.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Validation`] when the
    /// document cannot be serialized or written.
    pub fn save(&self, path: &Path) -> Result<()> {
        write_json_atomic(path, self)
    }

    /// Number of records.
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether the registry holds no record.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// All records, oldest first.
    #[must_use]
    pub fn records(&self) -> &[WorktreeRecord] {
        &self.records
    }

    /// The record for `name`, or for the record whose slug is `name`.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&WorktreeRecord> {
        self.records
            .iter()
            .find(|record| record.name == name || record.slug == name)
    }

    /// The record whose path is `path`.
    #[must_use]
    pub fn get_by_path(&self, path: &Path) -> Option<&WorktreeRecord> {
        self.records.iter().find(|record| record.path == path)
    }

    /// Inserts `record`, replacing any record with the same `name`.
    pub fn upsert(&mut self, record: WorktreeRecord) {
        if let Some(existing) = self
            .records
            .iter_mut()
            .find(|existing| existing.name == record.name)
        {
            *existing = record;
        } else {
            self.records.push(record);
        }
    }

    /// Removes the record for `name`, returning it.
    pub fn remove(&mut self, name: &str) -> Option<WorktreeRecord> {
        let index = self
            .records
            .iter()
            .position(|record| record.name == name || record.slug == name)?;
        Some(self.records.remove(index))
    }

    /// Drops active records whose path is not in `existing`, returning the count.
    ///
    /// Removed records are dropped too: a removed worktree has no directory to
    /// keep. This is the registry half of `cleanup_stale`.
    pub fn retain_existing(&mut self, existing: &[PathBuf]) -> usize {
        let before = self.records.len();
        self.records.retain(|record| match record.status {
            WorktreeState::Removed => false,
            WorktreeState::Active => existing.iter().any(|path| path == &record.path),
        });
        before.saturating_sub(self.records.len())
    }
}

/// Serializes `value` as pretty JSON and writes it to `path` atomically.
///
/// The temporary file is a sibling of the target (same filesystem), so the final
/// `rename` is atomic; a failed rename removes the temporary file.
pub(crate) fn write_json_atomic<T: Serialize + ?Sized>(path: &Path, value: &T) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value)
        .map_err(|error| Error::validation(format!("cannot serialize JSON document: {error}")))?;
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty());
    if let Some(parent) = parent {
        std::fs::create_dir_all(parent).map_err(|error| {
            Error::validation(format!(
                "cannot create directory `{}`: {error}",
                parent.display()
            ))
        })?;
    }
    let file_name = path.file_name().map_or_else(
        || INDEX_FILE.to_owned(),
        |name| name.to_string_lossy().into_owned(),
    );
    let temporary = parent.map_or_else(
        || PathBuf::from(format!("{file_name}.tmp")),
        |parent| parent.join(format!("{file_name}.tmp")),
    );
    std::fs::write(&temporary, bytes).map_err(|error| {
        Error::validation(format!("cannot write `{}`: {error}", temporary.display()))
    })?;
    if let Err(error) = std::fs::rename(&temporary, path) {
        let _ = std::fs::remove_file(&temporary);
        return Err(Error::validation(format!(
            "cannot publish `{}`: {error}",
            path.display()
        )));
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;

    fn record(name: &str) -> WorktreeRecord {
        WorktreeRecord::new(name, PathBuf::from(format!("/repo/.worktrees/{name}")))
            .with_branch(format!("worktree-{name}"))
            .with_created_at("2026-01-01T00:00:00+00:00")
    }

    #[test]
    fn index_path_lives_under_the_workspace() {
        assert_eq!(
            index_path(Path::new("/repo")),
            PathBuf::from("/repo/.worktree/index.json")
        );
    }

    #[test]
    fn missing_file_loads_empty_and_malformed_is_validation() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(INDEX_DIR).join(INDEX_FILE);
        let index = WorktreeIndex::load(&path).unwrap();
        assert!(index.is_empty());
        assert_eq!(index.version, INDEX_VERSION);

        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"{").unwrap();
        let error = WorktreeIndex::load(&path).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
    }

    #[test]
    fn index_round_trips_and_replaces_by_name() {
        let dir = tempfile::tempdir().unwrap();
        let path = index_path(dir.path());
        let mut index = WorktreeIndex::new();
        index.upsert(record("feature"));
        index.upsert(record("other"));
        index.upsert(record("feature").with_owner("agent-1"));
        index.save(&path).unwrap();

        let mut loaded = WorktreeIndex::load(&path).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(
            loaded
                .get("feature")
                .and_then(|record| record.owner.as_deref()),
            Some("agent-1")
        );
        assert_eq!(
            loaded.get_by_path(Path::new("/repo/.worktrees/other")),
            Some(&record("other"))
        );

        let removed = loaded.remove("feature").unwrap();
        assert_eq!(removed.name, "feature");
        assert!(loaded.remove("feature").is_none());
        assert_eq!(loaded.len(), 1);
    }

    #[test]
    fn retain_existing_drops_records_whose_directory_is_gone() {
        let mut index = WorktreeIndex::new();
        index.upsert(record("kept"));
        index.upsert(record("gone"));
        let mut removed = record("removed");
        removed.status = WorktreeState::Removed;
        index.upsert(removed);

        let existing = vec![PathBuf::from("/repo/.worktrees/kept")];
        assert_eq!(index.retain_existing(&existing), 2);
        assert_eq!(index.len(), 1);
        assert_eq!(index.records()[0].name, "kept");
    }

    #[test]
    fn saving_leaves_no_temporary_file_behind() {
        let dir = tempfile::tempdir().unwrap();
        let path = index_path(dir.path());
        WorktreeIndex::new().save(&path).unwrap();
        let entries: Vec<String> = std::fs::read_dir(dir.path().join(INDEX_DIR))
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(entries, [INDEX_FILE]);
    }
}
