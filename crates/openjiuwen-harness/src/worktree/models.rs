//! Worktree data models: git entries, creation results, status, and registry records.
//!
//! Port of the `pydantic` models in `openjiuwen/harness/tools/worktree/models.py`,
//! reduced to the shapes this helper actually produces. Every type is
//! `serde`-serializable, `snake_case`, and tolerant of unknown fields, so tool
//! payloads and the persisted registry stay readable across versions.
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::sparse::SparseConfig;

/// One entry of `git worktree list --porcelain`.
///
/// `recorded` attaches the registry record when this helper created the
/// worktree, which is how `list()` reports worktrees from earlier runs.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct WorktreeInfo {
    /// Absolute path of the worktree directory.
    pub path: PathBuf,
    /// Short branch name (`refs/heads/` stripped), absent when detached.
    pub branch: Option<String>,
    /// Full HEAD commit SHA as reported by git.
    pub head: Option<String>,
    /// Whether git reports this entry as a bare repository.
    pub bare: bool,
    /// Whether HEAD is detached.
    pub detached: bool,
    /// Whether the worktree is locked.
    pub locked: bool,
    /// Reason recorded with the lock, when git reports one.
    pub lock_reason: Option<String>,
    /// Prune reason, when git considers the entry prunable.
    pub prunable: Option<String>,
    /// Registry record of a worktree this helper created, when available.
    pub recorded: Option<WorktreeRecord>,
}

impl WorktreeInfo {
    /// Name of the worktree: its directory component.
    ///
    /// This is the slug the manager derives, not necessarily the name the
    /// caller originally passed.
    #[must_use]
    pub fn name(&self) -> String {
        self.path
            .file_name()
            .map_or_else(String::new, |name| name.to_string_lossy().into_owned())
    }

    /// Whether this entry is the main worktree of the repository.
    ///
    /// Heuristic only: the main worktree is the first entry of
    /// `git worktree list --porcelain`, so callers that need certainty track the
    /// index instead.
    #[must_use]
    pub fn is_detached(&self) -> bool {
        self.detached || self.branch.is_none()
    }
}

/// Result of [`crate::worktree::WorktreeManager::create`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct WorktreeCreateResult {
    /// Requested name, as the caller spelled it.
    pub name: String,
    /// Normalised slug used for the directory and the branch.
    pub slug: String,
    /// Absolute path of the worktree directory.
    pub path: PathBuf,
    /// Branch checked out in the worktree.
    pub branch: String,
    /// Base reference the branch was created from, when one was supplied.
    pub base: Option<String>,
    /// HEAD commit SHA after creation, when git reported one.
    pub head: Option<String>,
    /// Whether the worktree already existed and was recovered.
    pub existed: bool,
    /// Sparse-checkout selection applied to the worktree, when any.
    pub sparse: Option<SparseConfig>,
}

/// State of a worktree as seen by [`crate::worktree::WorktreeManager::status`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct WorktreeStatus {
    /// Name of the worktree (its directory component).
    pub name: String,
    /// Absolute path of the worktree directory.
    pub path: PathBuf,
    /// Checked-out branch, absent when detached.
    pub branch: Option<String>,
    /// HEAD commit SHA.
    pub head: Option<String>,
    /// Whether the worktree is locked.
    pub locked: bool,
    /// Reason recorded with the lock, when any.
    pub lock_reason: Option<String>,
    /// Whether the worktree has uncommitted changes.
    pub dirty: bool,
    /// Number of paths reported by `git status --porcelain`.
    pub changed_files: usize,
    /// Whether the registry holds a record for this worktree.
    pub tracked: bool,
    /// Sparse-checkout selection recorded for the worktree, when any.
    pub sparse: Option<SparseConfig>,
}

impl WorktreeStatus {
    /// Whether the worktree can be removed without `force`.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        !self.dirty
    }
}

/// Lifecycle state of a [`WorktreeRecord`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorktreeState {
    /// The worktree exists and is managed by this helper.
    #[default]
    Active,
    /// The worktree was removed but the record is kept for audit.
    Removed,
}

impl WorktreeState {
    /// The `snake_case` wire name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Removed => "removed",
        }
    }
}

/// One persisted registry entry (`.worktree/index.json`).
///
/// The record lets `list()` and `status()` report worktrees created by an
/// earlier process, and lets `cleanup_stale` drop entries whose directories are
/// gone.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct WorktreeRecord {
    /// Name the caller supplied at creation time.
    pub name: String,
    /// Normalised slug (directory component).
    pub slug: String,
    /// Absolute path of the worktree directory.
    pub path: PathBuf,
    /// Branch checked out in the worktree.
    pub branch: Option<String>,
    /// Base reference the branch was created from.
    pub base: Option<String>,
    /// Caller-defined owner id (agent id, team member, …), when supplied.
    pub owner: Option<String>,
    /// Lifecycle state of the record.
    pub status: WorktreeState,
    /// Creation timestamp, RFC 3339 in UTC.
    pub created_at: String,
    /// Sparse-checkout selection to re-apply when the worktree is re-entered.
    pub sparse: Option<SparseConfig>,
    /// HEAD commit SHA recorded at creation time, when git reported one.
    pub head: Option<String>,
}

impl WorktreeRecord {
    /// Builds an active record for `name` at `path`.
    #[must_use]
    pub fn new(name: impl Into<String>, path: PathBuf) -> Self {
        let name = name.into();
        let slug = path
            .file_name()
            .map_or_else(String::new, |slug| slug.to_string_lossy().into_owned());
        Self {
            name,
            slug,
            path,
            ..Self::default()
        }
    }

    /// Sets the branch.
    #[must_use]
    pub fn with_branch(mut self, branch: impl Into<String>) -> Self {
        self.branch = Some(branch.into());
        self
    }

    /// Sets the base reference.
    #[must_use]
    pub fn with_base(mut self, base: impl Into<String>) -> Self {
        self.base = Some(base.into());
        self
    }

    /// Sets the owner id.
    #[must_use]
    pub fn with_owner(mut self, owner: impl Into<String>) -> Self {
        self.owner = Some(owner.into());
        self
    }

    /// Sets the RFC 3339 creation timestamp.
    #[must_use]
    pub fn with_created_at(mut self, created_at: impl Into<String>) -> Self {
        self.created_at = created_at.into();
        self
    }

    /// Records the sparse-checkout selection of the worktree.
    #[must_use]
    pub fn with_sparse(mut self, sparse: SparseConfig) -> Self {
        self.sparse = Some(sparse);
        self
    }

    /// Records the HEAD commit SHA of the worktree.
    #[must_use]
    pub fn with_head(mut self, head: impl Into<String>) -> Self {
        self.head = Some(head.into());
        self
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn info_derives_its_name_and_detached_state() {
        let info = WorktreeInfo {
            path: PathBuf::from("/repo/.worktrees/feature"),
            branch: Some("worktree-feature".to_owned()),
            ..WorktreeInfo::default()
        };
        assert_eq!(info.name(), "feature");
        assert!(!info.is_detached());

        let detached = WorktreeInfo {
            path: PathBuf::from("/repo/.worktrees/feature"),
            ..WorktreeInfo::default()
        };
        assert!(detached.is_detached());
        assert_eq!(detached.name(), "feature");
    }

    #[test]
    fn records_default_to_active_and_round_trip() {
        let record = WorktreeRecord::new("feature", PathBuf::from("/repo/.worktrees/feature"))
            .with_branch("worktree-feature")
            .with_base("HEAD")
            .with_owner("agent-1")
            .with_created_at("2026-01-01T00:00:00+00:00");
        assert_eq!(record.status, WorktreeState::Active);
        assert_eq!(record.slug, "feature");

        let json = serde_json::to_string(&record).expect("serialize");
        let decoded: WorktreeRecord = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(decoded, record);
        assert_eq!(WorktreeState::Removed.as_str(), "removed");
    }

    #[test]
    fn models_tolerate_unknown_and_missing_fields() {
        let decoded: WorktreeInfo =
            serde_json::from_str(r#"{"path": "/repo/.worktrees/a", "future": true}"#)
                .expect("deserialize");
        assert_eq!(decoded.path, PathBuf::from("/repo/.worktrees/a"));
        assert!(decoded.branch.is_none());
        assert!(!decoded.locked);

        let status: WorktreeStatus = serde_json::from_str("{}").expect("deserialize");
        assert!(status.is_clean());
    }
}
