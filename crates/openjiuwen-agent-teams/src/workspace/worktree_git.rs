//! The git-backed [`WorktreeBackend`].
//!
//! Python parity source: `openjiuwen/harness/tools/worktree/backend.py` (`GitBackend`) plus the
//! lifecycle rules of `openjiuwen/agent_teams/worktree/lifecycle.py`.
//!
//! One teammate worktree is created by
//!
//! ```text
//! git worktree add -b worktree-<name> <session>/worktrees/<name> HEAD
//! ```
//!
//! and removed by `git worktree remove --force <path>`, `git worktree prune` and
//! `git branch -D worktree-<name>`. Every critical section runs under the repository lock
//! ([`super::lock`]) because two members can be allocated concurrently into the same repository and
//! `git worktree add` mutates shared `.git/worktrees` state.
//!
//! Two invariants matter for callers:
//!
//! - **A worktree is never reused.** [`WorktreeBackend::create`] fails with [`Error::Validation`]
//!   when the target directory or the branch it needs already exists; a caller that wants to keep
//!   working in an existing worktree must hold its [`WorktreeInfo`] instead of re-creating it.
//! - **Removal is complete.** A successful [`WorktreeBackend::remove`] has deregistered the
//!   worktree, pruned stale admin data and deleted its branch. When the branch is still checked out
//!   in another worktree the removal reports [`Error::Validation`] naming the branch — the worktree
//!   is gone but the branch is not, and the caller is told so instead of being told "removed".

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use tracing::debug;

use openjiuwen_common::error::{Error, Result};

use super::git::{
    normalize_path, BranchDeleteOutcome, GitRepository, WorktreeEntry, DEFAULT_BASE_REF,
};
use super::lock::{self, LockOptions, WorktreeLockGuard};
use super::worktree::{
    worktree_branch, WorktreeBackend, WorktreeCreateRequest, WorktreeInfo, BRANCH_PREFIX,
    NAME_PREFIX,
};

/// Creates teammate worktrees in a real git repository.
///
/// The repository is either pinned with [`Self::with_repo_root`] or discovered from the directory
/// the worktree is created in (walking up to the nearest existing ancestor). Discovery resolves the
/// *main* repository root even when it starts inside a worktree, so every teammate worktree of a
/// repository is registered in one `.git/worktrees` directory.
///
/// The backend is stateless apart from its configuration: all state lives in the repository and in
/// the lock file, so several instances (one per provider, or one per process) can cooperate.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GitWorktreeBackend {
    /// Repository to create worktrees in; `None` discovers it from the request path.
    repo_root: Option<PathBuf>,
    /// Bounds of the repository lock taken around every critical section.
    lock_options: LockOptions,
}

impl GitWorktreeBackend {
    /// Creates a backend that discovers the repository from the worktree location.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Pins the repository to `repo_root` instead of discovering it per request.
    ///
    /// A pinned root wins over discovery, which is what a caller that already resolved its project
    /// repository wants: worktrees then always belong to that repository, even when the session
    /// tree lives outside it.
    #[must_use]
    pub fn with_repo_root(mut self, repo_root: impl Into<PathBuf>) -> Self {
        self.repo_root = Some(normalize_path(&repo_root.into()));
        self
    }

    /// Sets the bounds of the repository lock.
    #[must_use]
    pub fn with_lock_options(mut self, lock_options: LockOptions) -> Self {
        self.lock_options = lock_options;
        self
    }

    /// Returns the pinned repository root, when the backend has one.
    #[must_use]
    pub fn repo_root(&self) -> Option<&Path> {
        self.repo_root.as_deref()
    }

    /// Returns the bounds of the repository lock.
    #[must_use]
    pub fn lock_options(&self) -> &LockOptions {
        &self.lock_options
    }

    /// Resolves the repository a worktree operation on `start` runs against.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when `start` is not inside a git repository and no repository is
    /// pinned, and whatever [`super::git::run_git`] reports when `git` is unavailable.
    async fn repository(&self, start: &Path) -> Result<GitRepository> {
        match self.repo_root.as_deref() {
            Some(root) => GitRepository::discover(root).await,
            None => GitRepository::discover(start).await,
        }
    }

    /// Takes the repository lock for one critical section.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Timeout`] when another member holds the lock for longer than
    /// [`LockOptions::wait`], and [`Error::SysOp`] when the lock file cannot be written.
    async fn lock_repository(&self, repository: &GitRepository) -> Result<WorktreeLockGuard> {
        lock::acquire(&repository.lock_file(), &self.lock_options).await
    }

    /// Returns the worktrees this crate owns in `repository`, sorted by path.
    ///
    /// # Errors
    ///
    /// Returns whatever [`GitRepository::worktree_entries`] reports.
    pub async fn list_worktrees(&self, repository: &GitRepository) -> Result<Vec<WorktreeInfo>> {
        let entries = repository.worktree_entries().await?;
        Ok(teammate_worktrees(entries))
    }
}

#[async_trait]
impl WorktreeBackend for GitWorktreeBackend {
    fn name(&self) -> &'static str {
        "git"
    }

    async fn create(&self, request: &WorktreeCreateRequest) -> Result<WorktreeInfo> {
        validate_request(request)?;
        let Some(parent) = request.path.parent() else {
            return Err(Error::validation(format!(
                "worktree path {} has no parent directory",
                request.path.display()
            )));
        };
        tokio::fs::create_dir_all(parent).await.map_err(|error| {
            Error::sys_op(format!(
                "creating worktree parent directory {}: {error}",
                parent.display()
            ))
        })?;
        let repository = self.repository(parent).await?;
        let _guard = self.lock_repository(&repository).await?;
        // `git worktree add` accepts an existing *empty* directory, so an existing directory is only
        // a conflict when it has content. A registered worktree is always a conflict: silently
        // reusing it would hand two members the same working tree.
        if let Some(existing) = repository.worktree_entry(&request.path).await? {
            return Err(Error::validation(format!(
                "worktree {} is already registered in {} on branch {}; refusing to reuse it",
                request.path.display(),
                repository.root.display(),
                existing.branch.as_deref().unwrap_or("<detached>")
            )));
        }
        if directory_has_entries(&request.path).await? {
            return Err(Error::validation(format!(
                "worktree target {} already exists and is not empty; refusing to reuse it",
                request.path.display()
            )));
        }
        let branch = worktree_branch(&request.name);
        let base_ref = request
            .base_ref
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or(DEFAULT_BASE_REF);
        repository
            .worktree_add(&request.path, &branch, base_ref)
            .await?;
        let path = normalize_path(&request.path);
        debug!(
            worktree = %request.name,
            branch = %branch,
            base = %base_ref,
            path = %path.display(),
            repository = %repository.root.display(),
            "created teammate worktree"
        );
        Ok(WorktreeInfo {
            name: request.name.clone(),
            path,
            branch,
        })
    }

    async fn remove(&self, worktree: &WorktreeInfo) -> Result<()> {
        if worktree.path.as_os_str().is_empty() {
            return Err(Error::validation("worktree path is empty"));
        }
        let start = worktree
            .path
            .parent()
            .map_or_else(|| PathBuf::from("."), Path::to_path_buf);
        let repository = self.repository(&start).await?;
        let _guard = self.lock_repository(&repository).await?;
        if repository.worktree_entry(&worktree.path).await?.is_none() {
            repository.worktree_prune().await?;
            return Err(Error::not_found(format!(
                "worktree {} is not registered in the repository at {}; it was already removed",
                worktree.path.display(),
                repository.root.display()
            )));
        }
        let removal = repository.worktree_remove(&worktree.path).await;
        // Prune regardless of the removal outcome: a failed removal can still leave admin data for
        // an already-deleted directory, and the next create must not trip over it.
        let prune = repository.worktree_prune().await;
        removal?;
        prune?;
        let branch = if worktree.branch.trim().is_empty() {
            worktree_branch(&worktree.name)
        } else {
            worktree.branch.clone()
        };
        match repository.delete_branch(&branch).await? {
            BranchDeleteOutcome::Deleted => {}
            BranchDeleteOutcome::Missing => {
                debug!(branch = %branch, "worktree branch was already gone");
            }
            BranchDeleteOutcome::InUse(detail) => {
                return Err(Error::validation(format!(
                    "removed worktree {} but branch {branch} is still checked out elsewhere: {detail}",
                    worktree.path.display()
                )));
            }
        }
        debug!(
            worktree = %worktree.name,
            branch = %branch,
            path = %worktree.path.display(),
            "removed teammate worktree"
        );
        Ok(())
    }

    async fn list(&self, repo_root: &Path) -> Result<Vec<WorktreeInfo>> {
        let repository = self.repository(repo_root).await?;
        self.list_worktrees(&repository).await
    }

    async fn lock(&self, worktree: &WorktreeInfo, reason: &str) -> Result<()> {
        let start = worktree
            .path
            .parent()
            .map_or_else(|| PathBuf::from("."), Path::to_path_buf);
        let repository = self.repository(&start).await?;
        let _guard = self.lock_repository(&repository).await?;
        repository.lock_worktree(&worktree.path, reason).await?;
        debug!(
            worktree = %worktree.name,
            reason = %reason,
            "locked teammate worktree"
        );
        Ok(())
    }

    async fn unlock(&self, worktree: &WorktreeInfo) -> Result<()> {
        let start = worktree
            .path
            .parent()
            .map_or_else(|| PathBuf::from("."), Path::to_path_buf);
        let repository = self.repository(&start).await?;
        let _guard = self.lock_repository(&repository).await?;
        repository.unlock_worktree(&worktree.path).await?;
        debug!(worktree = %worktree.name, "unlocked teammate worktree");
        Ok(())
    }
}

/// Maps repository entries to the teammate worktrees this crate owns.
///
/// A teammate worktree is recognized by both of its name markers: the directory stem starts with
/// [`NAME_PREFIX`] and the checked-out branch starts with [`BRANCH_PREFIX`]. Entries on a detached
/// `HEAD`, bare repository entries and worktrees an operator created by hand are therefore ignored.
/// The result is sorted by path so two calls never disagree about order.
#[must_use]
pub fn teammate_worktrees(entries: Vec<WorktreeEntry>) -> Vec<WorktreeInfo> {
    let mut worktrees: Vec<WorktreeInfo> = entries
        .into_iter()
        .filter(|entry| !entry.bare && !entry.detached)
        .filter_map(|entry| {
            let branch = entry.branch.as_deref()?;
            if !branch.starts_with(BRANCH_PREFIX) {
                return None;
            }
            let name = entry.path.file_name()?.to_str()?;
            if !name.starts_with(NAME_PREFIX) {
                return None;
            }
            Some(WorktreeInfo {
                name: name.to_owned(),
                path: entry.path.clone(),
                branch: branch.to_owned(),
            })
        })
        .collect();
    worktrees.sort_by(|left, right| left.path.cmp(&right.path));
    worktrees
}

/// Rejects a request the backend cannot act on before it touches the filesystem.
fn validate_request(request: &WorktreeCreateRequest) -> Result<()> {
    if request.name.trim().is_empty() {
        return Err(Error::validation("worktree name is empty"));
    }
    if !request.path.is_absolute() {
        return Err(Error::validation(format!(
            "worktree path {} must be absolute",
            request.path.display()
        )));
    }
    if request.path.to_str().is_none() {
        return Err(Error::validation(format!(
            "worktree path {} is not valid UTF-8 and cannot be passed to git",
            request.path.display()
        )));
    }
    Ok(())
}

/// Returns `true` when `path` exists and already contains at least one entry.
///
/// A missing path and an existing empty directory both yield `false`: `git worktree add` populates an
/// existing empty directory happily, and neither is a worktree this backend could reuse.
async fn directory_has_entries(path: &Path) -> Result<bool> {
    let mut handle = match tokio::fs::read_dir(path).await {
        Ok(handle) => handle,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => {
            return Err(Error::sys_op(format!(
                "inspecting worktree target {}: {error}",
                path.display()
            )));
        }
    };
    match handle.next_entry().await {
        Ok(entry) => Ok(entry.is_some()),
        Err(error) => Err(Error::sys_op(format!(
            "inspecting worktree target {}: {error}",
            path.display()
        ))),
    }
}
