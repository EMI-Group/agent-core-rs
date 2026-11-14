//! Worktree manager: layout, lifecycle operations, refusals, and bookkeeping.
//!
//! Port of `openjiuwen/harness/tools/worktree/{manager,cleanup,session}.py`,
//! reduced to the operations this helper needs: create, list, remove, prune,
//! lock, unlock, status, sparse setup, change summaries and cleanup. The manager
//! is the single owner of the layout and of the safety rules:
//!
//! - **Layout** — a worktree lives at `<repo-root>/.worktrees/<slug>`, matching
//!   the prompt guidance the model receives. The slug is derived by
//!   [`super::slug::slugify`]; a nested name (`user/feature`) is flattened.
//! - **Refusals** — a raw name with `..` segments, an absolute path, an
//!   occupied target directory, or a dirty worktree on a non-forced removal is
//!   rejected with a typed error instead of being "repaired".
//! - **Bookkeeping** — every created worktree is recorded in
//!   `<workspace>/.worktree/index.json` ([`super::store`]) and reported through
//!   [`WorktreeEventSink`], so `list()`/`status()` know what an earlier process
//!   created and a host can react to lifecycle changes. Records keep the HEAD
//!   commit and the sparse-checkout selection, so a re-entered worktree can be
//!   restored and a change summary can count from where the branch started.
//! - **Locking** — every mutating operation (`create`/`create_with_options`,
//!   `remove`, `prune`, `cleanup_stale`, `cleanup`, `lock`, `unlock`,
//!   `apply_sparse`) takes the repo-bucketed advisory lock of
//!   [`super::lock`] for its critical section, exactly like Python's
//!   `async with repo_lock(repo_root, timeout=config.repo_lock_timeout)`. The
//!   guard is an OS advisory lock, not a blocking in-process mutex, so it is
//!   safe to hold across the operation's git `.await` points; it is dropped
//!   before lifecycle events are emitted. Read-only paths (`list`, `status`,
//!   `change_summary`) stay outside the lock on purpose.
//! - **Sparse checkout** — a configured selection is applied on creation as
//!   `worktree add --no-checkout` → `sparse-checkout set` → `checkout HEAD`, and
//!   a half-created worktree is removed again on failure; a recovered worktree
//!   re-applies its recorded selection without touching its working tree
//!   ([`WorktreeManager::apply_sparse`]).
//! - **Cleanup** — [`cleanup`] reaps old ephemeral worktrees fail-closed;
//!   [`WorktreeManager::cleanup`] runs it under the lock and reports the pass.
//!
//! The manager performs no caching: the registry is read and written per
//! operation, so two managers over one workspace see each other's writes.
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use openjiuwen_common::error::{Error, Result};

use crate::permission::paths::normalize_lexically;

use super::changes::{self, WorktreeChangeSummary};
use super::cleanup::{self, CleanupPolicy, CleanupReport};
use super::events::{
    BroadcastWorktreeEvents, NoopWorktreeEvents, SharedWorktreeEvents, WorktreeEvent,
    WorktreeEventSink,
};
use super::git::{self, GitRunner, ProcessGitRunner};
use super::lock::{WorktreeLock, WorktreeLockConfig, WorktreeLockGuard};
use super::models::{
    WorktreeCreateResult, WorktreeInfo, WorktreeRecord, WorktreeState, WorktreeStatus,
};
use super::slug;
use super::sparse::SparseConfig;
use super::store::{self, WorktreeIndex};

/// Default base reference used when the caller supplies none.
pub const DEFAULT_BASE: &str = "HEAD";

/// Options of [`WorktreeManager::create_with_options`].
///
/// A per-call sparse selection overrides the manager-wide default configured
/// with [`WorktreeManager::with_sparse`]; everything else defaults.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct WorktreeCreateOptions {
    /// Sparse-checkout selection to apply, when any.
    pub sparse: Option<SparseConfig>,
}

impl WorktreeCreateOptions {
    /// Applies `sparse` to the worktree being created.
    #[must_use]
    pub fn with_sparse(mut self, sparse: SparseConfig) -> Self {
        self.sparse = Some(sparse);
        self
    }

    /// Whether no option deviates from the defaults.
    #[must_use]
    pub fn is_default(&self) -> bool {
        self.sparse.is_none()
    }
}

/// A git worktree lifecycle manager.
///
/// Construct with [`WorktreeManager::new`], which wires the real
/// [`ProcessGitRunner`]; tests inject a scripted [`GitRunner`] through
/// [`WorktreeManager::with_runner`] instead.
pub struct WorktreeManager {
    repo_root: PathBuf,
    workspace_root: PathBuf,
    owner: Option<String>,
    runner: Arc<dyn GitRunner>,
    events: SharedWorktreeEvents,
    lock_config: WorktreeLockConfig,
    sparse: Option<SparseConfig>,
}

impl std::fmt::Debug for WorktreeManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorktreeManager")
            .field("repo_root", &self.repo_root)
            .field("workspace_root", &self.workspace_root)
            .field("owner", &self.owner)
            .finish_non_exhaustive()
    }
}

impl WorktreeManager {
    /// Builds a manager for the repository at `repo_root` using the real git
    /// binary, with the registry under the same root.
    #[must_use]
    pub fn new(repo_root: impl Into<PathBuf>) -> Self {
        let repo_root = repo_root.into();
        Self {
            workspace_root: repo_root.clone(),
            repo_root,
            owner: None,
            runner: Arc::new(ProcessGitRunner::new()),
            events: Arc::new(NoopWorktreeEvents),
            lock_config: WorktreeLockConfig::default(),
            sparse: None,
        }
    }

    /// Builds a manager over `repo_root` with an explicit [`GitRunner`].
    #[must_use]
    pub fn with_runner(repo_root: impl Into<PathBuf>, runner: Arc<dyn GitRunner>) -> Self {
        let repo_root = repo_root.into();
        Self {
            workspace_root: repo_root.clone(),
            repo_root,
            owner: None,
            runner,
            events: Arc::new(NoopWorktreeEvents),
            lock_config: WorktreeLockConfig::default(),
            sparse: None,
        }
    }

    /// Places the registry (`.worktree/index.json`) under `workspace_root`.
    ///
    /// Worktrees still live under the repository root; only the registry moves.
    #[must_use]
    pub fn with_workspace_root(mut self, workspace_root: impl Into<PathBuf>) -> Self {
        self.workspace_root = workspace_root.into();
        self
    }

    /// Records `owner` on created worktrees and their events.
    #[must_use]
    pub fn with_owner(mut self, owner: impl Into<String>) -> Self {
        self.owner = Some(owner.into());
        self
    }

    /// Routes lifecycle events to `events`.
    #[must_use]
    pub fn with_events(mut self, events: Arc<dyn WorktreeEventSink>) -> Self {
        self.events = events;
        self
    }

    /// Routes lifecycle events to an in-process broadcast channel.
    #[must_use]
    pub fn with_broadcast_events(self, capacity: usize) -> (Self, Arc<BroadcastWorktreeEvents>) {
        let sink = Arc::new(BroadcastWorktreeEvents::new(capacity));
        (self.with_events(sink.clone()), sink)
    }

    /// Overrides the repository-lock policy (Python's `repo_lock` /
    /// `repo_lock_timeout`).
    #[must_use]
    pub fn with_lock_config(mut self, lock_config: WorktreeLockConfig) -> Self {
        self.lock_config = lock_config;
        self
    }

    /// Sets the manager-wide sparse-checkout selection applied on creation and
    /// re-applied when a worktree is recovered.
    ///
    /// The selection is not validated here; `create` re-validates it before any
    /// git command ([`WorktreeCreateOptions::with_sparse`] overrides it per
    /// call).
    #[must_use]
    pub fn with_sparse(mut self, sparse: SparseConfig) -> Self {
        self.sparse = Some(sparse);
        self
    }

    /// The repository-lock policy.
    #[must_use]
    pub fn lock_config(&self) -> &WorktreeLockConfig {
        &self.lock_config
    }

    /// A repository lock for this manager, built from the current roots and
    /// policy.
    ///
    /// Every handle over one repository root shares the sidecar lock file, so
    /// building one per operation is both cheap and correct.
    #[must_use]
    pub fn repo_lock(&self) -> WorktreeLock {
        WorktreeLock::new(
            &self.repo_root,
            &self.workspace_root,
            self.lock_config.clone(),
        )
    }

    /// The sidecar lock file of the repository lock.
    #[must_use]
    pub fn lock_path(&self) -> PathBuf {
        self.repo_lock().lock_path()
    }

    /// The manager-wide sparse-checkout selection, when one is configured.
    #[must_use]
    pub fn sparse(&self) -> Option<&SparseConfig> {
        self.sparse.as_ref()
    }

    /// The repository root the manager was built with.
    #[must_use]
    pub fn repo_root(&self) -> &Path {
        &self.repo_root
    }

    /// The workspace root holding the registry.
    #[must_use]
    pub fn workspace_root(&self) -> &Path {
        &self.workspace_root
    }

    /// The owner id recorded on new worktrees, when configured.
    #[must_use]
    pub fn owner(&self) -> Option<&str> {
        self.owner.as_deref()
    }

    /// The git runner.
    #[must_use]
    pub fn runner(&self) -> &Arc<dyn GitRunner> {
        &self.runner
    }

    /// The parent directory of every worktree (`<repo>/.worktrees`).
    #[must_use]
    pub fn worktrees_dir(&self) -> PathBuf {
        self.repo_root.join(slug::WORKTREES_DIR)
    }

    /// The registry file (`<workspace>/.worktree/index.json`).
    #[must_use]
    pub fn index_path(&self) -> PathBuf {
        store::index_path(&self.workspace_root)
    }

    /// Resolves the worktree path of `name`.
    ///
    /// The raw name is validated first (no `..` segments, no absolute path), then
    /// slugified and joined under [`Self::worktrees_dir`]. The result is
    /// normalised lexically and must stay inside the repository root.
    ///
    /// # Errors
    ///
    /// Returns [`Error::security`] when the name is absolute, contains a `..`
    /// segment, contains a NUL byte, or the resolved path escapes the
    /// repository; [`Error::validation`] when the name is empty.
    pub fn target_path(&self, name: &str) -> Result<PathBuf> {
        slug::validate_raw_name(name)?;
        let root = normalize_lexically(&self.repo_root);
        let target = normalize_lexically(&root.join(slug::WORKTREES_DIR).join(slug::slugify(name)));
        if !target.starts_with(&root) {
            return Err(Error::security(format!(
                "worktree path `{}` escapes the repository root `{}`",
                target.display(),
                root.display()
            )));
        }
        Ok(target)
    }

    /// Verifies that the repository root is inside a git repository.
    ///
    /// # Errors
    ///
    /// Returns [`Error::config`] when git is not installed or the root is not a
    /// repository.
    pub async fn ensure_repository(&self) -> Result<PathBuf> {
        git::find_git_root(self.runner.as_ref(), &self.repo_root).await
    }

    /// Creates (or recovers) the worktree `name`.
    ///
    /// `branch` defaults to `worktree-<slug>`; `base` defaults to
    /// [`DEFAULT_BASE`]. An existing branch is checked out instead of re-created.
    /// An existing worktree at the target path is recovered and reported with
    /// `existed = true`; a non-worktree directory at that path is refused.
    /// Shorthand of [`Self::create_with_options`] with the default options.
    ///
    /// # Errors
    ///
    /// Returns the name-validation errors of [`Self::target_path`],
    /// [`Error::validation`] for an occupied target, the lock error, or the git
    /// error.
    pub async fn create(
        &self,
        name: &str,
        branch: Option<&str>,
        base: Option<&str>,
    ) -> Result<WorktreeCreateResult> {
        self.create_with_options(name, branch, base, &WorktreeCreateOptions::default())
            .await
    }

    /// Creates (or recovers) the worktree `name` with explicit `options`.
    ///
    /// The effective sparse selection is `options.sparse` when set, otherwise
    /// the manager-wide [`Self::with_sparse`] selection; either way it is
    /// validated before the first git command. With a selection, creation runs
    /// `worktree add --no-checkout`, `sparse-checkout set` and `checkout HEAD`,
    /// and a failed sparse step removes the half-created worktree again
    /// ([`super::git::is_sparse_failure`] identifies that error). A recovered
    /// worktree re-applies the selection recorded in the registry — without a
    /// `checkout HEAD`, which would disturb the working tree — and the applied
    /// selection is reported with [`WorktreeEvent::SparseConfigured`].
    ///
    /// # Errors
    ///
    /// Returns the name-validation errors of [`Self::target_path`],
    /// [`Error::validation`] for an occupied target or an invalid sparse
    /// selection, the lock error, the git error, or the sparse-setup error.
    pub async fn create_with_options(
        &self,
        name: &str,
        branch: Option<&str>,
        base: Option<&str>,
        options: &WorktreeCreateOptions,
    ) -> Result<WorktreeCreateResult> {
        let sparse = options.sparse.clone().or_else(|| self.sparse.clone());
        if let Some(sparse) = &sparse {
            if let Err(error) = sparse.validate() {
                self.report_failure("create", &error).await;
                return Err(error);
            }
        }
        let guard = self.acquire_repo_lock("create").await?;
        let outcome = self.create_inner(name, branch, base, sparse.as_ref()).await;
        drop(guard);
        match outcome {
            Ok(result) => {
                self.events
                    .emit(WorktreeEvent::Created {
                        name: result.name.clone(),
                        path: result.path.clone(),
                        branch: result.branch.clone(),
                        existed: result.existed,
                        owner: self.owner.clone(),
                    })
                    .await;
                if let Some(sparse) = &result.sparse {
                    self.events
                        .emit(WorktreeEvent::SparseConfigured {
                            name: result.name.clone(),
                            path: result.path.clone(),
                            mode: sparse.mode,
                            paths: sparse.paths.clone(),
                        })
                        .await;
                }
                Ok(result)
            }
            Err(error) => {
                self.report_failure("create", &error).await;
                Err(error)
            }
        }
    }

    /// Implementation of [`Self::create_with_options`], without locking or event
    /// handling.
    async fn create_inner(
        &self,
        name: &str,
        branch: Option<&str>,
        base: Option<&str>,
        sparse: Option<&SparseConfig>,
    ) -> Result<WorktreeCreateResult> {
        let target = self.target_path(name)?;
        let repo = self.ensure_repository().await?;
        let slug = slug::slugify(name);
        let branch = branch.map_or_else(|| slug::branch_name(&slug), str::to_owned);
        let base = base.map(str::to_owned);

        let existing = git::worktree_list(self.runner.as_ref(), &repo)
            .await?
            .into_iter()
            .find(|info| same_path(&info.path, &target));
        if let Some(info) = existing {
            // Recovery never re-populates an existing working tree: only the
            // selection recorded for this worktree is re-applied.
            let recorded_sparse = WorktreeIndex::load(&self.index_path())?
                .get(name)
                .and_then(|record| record.sparse.clone());
            let mut record = self.record_for(name, &slug, &target, Some(&branch), base.as_deref());
            if let Some(head) = info.head.clone() {
                record = record.with_head(head);
            }
            if let Some(sparse) = &recorded_sparse {
                sparse.validate()?;
                git::sparse_checkout_set(self.runner.as_ref(), &target, sparse).await?;
                record = record.with_sparse(sparse.clone());
            }
            self.store_record(&record)?;
            return Ok(WorktreeCreateResult {
                name: name.to_owned(),
                slug,
                path: target,
                branch: info.branch.unwrap_or(branch),
                base,
                head: info.head,
                existed: true,
                sparse: recorded_sparse,
            });
        }

        if target.exists() {
            return Err(Error::validation(format!(
                "worktree path `{}` already exists and is not a git worktree; remove it or pick another name",
                target.display()
            )));
        }
        let base_ref = base.as_deref().unwrap_or(DEFAULT_BASE);
        let branch_exists = git::branch_exists(self.runner.as_ref(), &repo, &branch).await?;
        match (sparse.is_some(), branch_exists) {
            (false, false) => {
                git::worktree_add(self.runner.as_ref(), &repo, &target, &branch, base_ref).await?;
            }
            (false, true) => {
                git::worktree_add_existing(self.runner.as_ref(), &repo, &target, &branch).await?;
            }
            (true, false) => {
                git::worktree_add_no_checkout(
                    self.runner.as_ref(),
                    &repo,
                    &target,
                    &branch,
                    base_ref,
                )
                .await?;
            }
            (true, true) => {
                git::worktree_add_no_checkout_existing(
                    self.runner.as_ref(),
                    &repo,
                    &target,
                    &branch,
                )
                .await?;
            }
        }

        if let Some(sparse) = sparse {
            self.apply_sparse_on_create(&repo, &target, sparse).await?;
        }

        let head = git::rev_parse(self.runner.as_ref(), &target, "HEAD").await?;
        let mut record = self.record_for(name, &slug, &target, Some(&branch), base.as_deref());
        if let Some(head) = head.clone() {
            record = record.with_head(head);
        }
        if let Some(sparse) = sparse {
            record = record.with_sparse(sparse.clone());
        }
        self.store_record(&record)?;
        Ok(WorktreeCreateResult {
            name: name.to_owned(),
            slug,
            path: target,
            branch,
            base,
            head,
            existed: false,
            sparse: sparse.cloned(),
        })
    }

    /// Applies `sparse` to the freshly added worktree at `target`.
    ///
    /// Runs `sparse-checkout set` then `checkout HEAD`; when either fails the
    /// half-created worktree is removed again (best-effort, like Python) and a
    /// sparse-specific error is returned.
    async fn apply_sparse_on_create(
        &self,
        repo: &Path,
        target: &Path,
        sparse: &SparseConfig,
    ) -> Result<()> {
        let outcome = match git::sparse_checkout_set(self.runner.as_ref(), target, sparse).await {
            Ok(()) => git::checkout_head(self.runner.as_ref(), target).await,
            Err(error) => Err(error),
        };
        if let Err(error) = outcome {
            let _ = git::worktree_remove(self.runner.as_ref(), repo, target, true).await;
            return Err(git::sparse_failure_error(target, &error.to_string()));
        }
        Ok(())
    }
    /// Lists the repository's worktrees, attaching registry records.
    ///
    /// Entries created by this helper carry their [`WorktreeRecord`] in
    /// `recorded`, so callers see branch base, owner and creation time as well as
    /// the git state.
    ///
    /// # Errors
    ///
    /// Returns [`Error::config`] outside a repository, or a registry validation
    /// error when the index is corrupt.
    pub async fn list(&self) -> Result<Vec<WorktreeInfo>> {
        let repo = self.ensure_repository().await?;
        let mut infos = git::worktree_list(self.runner.as_ref(), &repo).await?;
        let index = WorktreeIndex::load(&self.index_path())?;
        for info in &mut infos {
            info.recorded = index
                .get_by_path(&info.path)
                .filter(|record| record.status == WorktreeState::Active)
                .cloned();
        }
        Ok(infos)
    }

    /// Removes the worktree `name`.
    ///
    /// A dirty worktree is refused unless `force` is set; a locked worktree is
    /// likewise left to git, which refuses to remove it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::not_found`] when git does not know the worktree,
    /// [`Error::validation`] when it is dirty and `force` is not set, or the git
    /// error.
    pub async fn remove(&self, name: &str, force: bool) -> Result<WorktreeInfo> {
        let guard = self.acquire_repo_lock("remove").await?;
        let outcome = self.remove_inner(name, force).await;
        drop(guard);
        match outcome {
            Ok(info) => {
                self.events
                    .emit(WorktreeEvent::Removed {
                        name: name.to_owned(),
                        path: info.path.clone(),
                        forced: force,
                    })
                    .await;
                Ok(info)
            }
            Err(error) => {
                self.report_failure("remove", &error).await;
                Err(error)
            }
        }
    }

    /// Implementation of [`Self::remove`], without event handling.
    async fn remove_inner(&self, name: &str, force: bool) -> Result<WorktreeInfo> {
        let target = self.target_path(name)?;
        let repo = self.ensure_repository().await?;
        let info = git::worktree_list(self.runner.as_ref(), &repo)
            .await?
            .into_iter()
            .find(|info| same_path(&info.path, &target))
            .ok_or_else(|| {
                Error::not_found(format!(
                    "worktree `{name}` is not registered with git at `{}`",
                    target.display()
                ))
            })?;
        if !force {
            let changes = git::status_porcelain(self.runner.as_ref(), &info.path).await?;
            if !changes.is_empty() {
                return Err(Error::validation(format!(
                    "worktree `{name}` has {} uncommitted change(s); pass force to remove it",
                    changes.len()
                )));
            }
        }
        git::worktree_remove(self.runner.as_ref(), &repo, &info.path, force).await?;
        let mut index = WorktreeIndex::load(&self.index_path())?;
        index.remove(name);
        index.save(&self.index_path())?;
        Ok(info)
    }

    /// Prunes prunable git worktrees and stale registry records.
    ///
    /// Returns the number of registry records dropped.
    ///
    /// # Errors
    ///
    /// Returns the git or registry error.
    pub async fn prune(&self) -> Result<usize> {
        let guard = self.acquire_repo_lock("prune").await?;
        let outcome = self.prune_inner().await;
        drop(guard);
        match outcome {
            Ok(removed) => {
                self.events.emit(WorktreeEvent::Pruned { removed }).await;
                Ok(removed)
            }
            Err(error) => {
                self.report_failure("prune", &error).await;
                Err(error)
            }
        }
    }

    /// Implementation of [`Self::prune`], without event handling.
    async fn prune_inner(&self) -> Result<usize> {
        let repo = self.ensure_repository().await?;
        git::worktree_prune(self.runner.as_ref(), &repo).await?;
        self.cleanup_stale_in(&repo).await
    }

    /// Drops registry records whose directory git no longer lists.
    ///
    /// # Errors
    ///
    /// Returns the lock, git or registry error.
    pub async fn cleanup_stale(&self) -> Result<usize> {
        let guard = self.acquire_repo_lock("cleanup_stale").await?;
        let outcome = async {
            let repo = self.ensure_repository().await?;
            self.cleanup_stale_in(&repo).await
        }
        .await;
        drop(guard);
        if let Err(error) = &outcome {
            self.report_failure("cleanup_stale", error).await;
        }
        outcome
    }

    /// Implementation of [`Self::cleanup_stale`] over an already-resolved root.
    async fn cleanup_stale_in(&self, repo: &Path) -> Result<usize> {
        let paths: Vec<PathBuf> = git::worktree_list(self.runner.as_ref(), repo)
            .await?
            .into_iter()
            .map(|info| info.path)
            .collect();
        let path = self.index_path();
        let mut index = WorktreeIndex::load(&path)?;
        let removed = index.retain_existing(&paths);
        if removed > 0 {
            index.save(&path)?;
        }
        Ok(removed)
    }

    /// Locks the worktree `name`, optionally recording a `reason`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::not_found`] when git does not know the worktree, or the
    /// git error.
    pub async fn lock(&self, name: &str, reason: Option<&str>) -> Result<()> {
        let target = self.target_path(name)?;
        let guard = self.acquire_repo_lock("lock").await?;
        let outcome = self.lock_inner(name, &target, reason).await;
        drop(guard);
        match outcome {
            Ok(()) => {
                self.events
                    .emit(WorktreeEvent::Locked {
                        name: name.to_owned(),
                        path: target,
                        reason: reason.map(str::to_owned),
                    })
                    .await;
                Ok(())
            }
            Err(error) => {
                self.report_failure("lock", &error).await;
                Err(error)
            }
        }
    }

    /// Implementation of [`Self::lock`], without event handling.
    async fn lock_inner(&self, name: &str, target: &Path, reason: Option<&str>) -> Result<()> {
        let repo = self.ensure_repository().await?;
        let info = self.require_listed(name, &repo, target).await?;
        git::worktree_lock(self.runner.as_ref(), &repo, &info.path, reason).await
    }

    /// Unlocks the worktree `name`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::not_found`] when git does not know the worktree, or the
    /// git error.
    pub async fn unlock(&self, name: &str) -> Result<()> {
        let target = self.target_path(name)?;
        let guard = self.acquire_repo_lock("unlock").await?;
        let outcome = self.unlock_inner(name, &target).await;
        drop(guard);
        match outcome {
            Ok(()) => {
                self.events
                    .emit(WorktreeEvent::Unlocked {
                        name: name.to_owned(),
                        path: target,
                    })
                    .await;
                Ok(())
            }
            Err(error) => {
                self.report_failure("unlock", &error).await;
                Err(error)
            }
        }
    }

    /// Implementation of [`Self::unlock`], without event handling.
    async fn unlock_inner(&self, name: &str, target: &Path) -> Result<()> {
        let repo = self.ensure_repository().await?;
        let info = self.require_listed(name, &repo, target).await?;
        git::worktree_unlock(self.runner.as_ref(), &repo, &info.path).await
    }

    /// Reports the current state of the worktree `name`.
    ///
    /// The sparse selection comes from the registry (an active record only), so
    /// status never invokes git beyond listing and `status --porcelain`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::not_found`] when git does not know the worktree, or the
    /// git or registry error.
    pub async fn status(&self, name: &str) -> Result<WorktreeStatus> {
        let target = self.target_path(name)?;
        let repo = self.ensure_repository().await?;
        let info = git::worktree_list(self.runner.as_ref(), &repo)
            .await?
            .into_iter()
            .find(|info| same_path(&info.path, &target))
            .ok_or_else(|| {
                Error::not_found(format!(
                    "worktree `{name}` is not registered with git at `{}`",
                    target.display()
                ))
            })?;
        let changes = git::status_porcelain(self.runner.as_ref(), &info.path).await?;
        let index = WorktreeIndex::load(&self.index_path())?;
        let record = index
            .get(name)
            .filter(|record| record.status == WorktreeState::Active);
        Ok(WorktreeStatus {
            name: name.to_owned(),
            path: info.path,
            branch: info.branch,
            head: info.head,
            locked: info.locked,
            lock_reason: info.lock_reason,
            dirty: !changes.is_empty(),
            changed_files: changes.len(),
            tracked: record.is_some(),
            sparse: record.and_then(|record| record.sparse.clone()),
        })
    }

    /// Re-applies the sparse-checkout selection recorded for the worktree
    /// `name`.
    ///
    /// Read-modify-write under the repository lock: git is the source of truth
    /// for the worktree, the registry for the selection. `Ok(None)` means the
    /// worktree has no recorded selection (or no record at all), so there is
    /// nothing to apply.
    ///
    /// # Errors
    ///
    /// Returns [`Error::not_found`] when git does not know the worktree, the
    /// sparse validation error, the lock error, or the git error.
    pub async fn apply_sparse(&self, name: &str) -> Result<Option<SparseConfig>> {
        let target = self.target_path(name)?;
        let guard = self.acquire_repo_lock("apply_sparse").await?;
        let outcome = self.apply_sparse_inner(name, &target).await;
        drop(guard);
        match outcome {
            Ok(Some(sparse)) => {
                self.events
                    .emit(WorktreeEvent::SparseConfigured {
                        name: name.to_owned(),
                        path: target,
                        mode: sparse.mode,
                        paths: sparse.paths.clone(),
                    })
                    .await;
                Ok(Some(sparse))
            }
            Ok(None) => Ok(None),
            Err(error) => {
                self.report_failure("apply_sparse", &error).await;
                Err(error)
            }
        }
    }

    /// Implementation of [`Self::apply_sparse`], without locking or events.
    async fn apply_sparse_inner(&self, name: &str, target: &Path) -> Result<Option<SparseConfig>> {
        let repo = self.ensure_repository().await?;
        let info = self.require_listed(name, &repo, target).await?;
        let index = WorktreeIndex::load(&self.index_path())?;
        let Some(sparse) = index.get(name).and_then(|record| record.sparse.clone()) else {
            return Ok(None);
        };
        sparse.validate()?;
        git::sparse_checkout_set(self.runner.as_ref(), &info.path, &sparse).await?;
        Ok(Some(sparse))
    }

    /// Summarises the uncommitted and unpushed work in the worktree `name`.
    ///
    /// Read-only and deliberately outside the repository lock (Python's
    /// `count_changes` is lock-free too). The base reference is the HEAD commit
    /// recorded at creation, falling back to the recorded base when that is not
    /// [`DEFAULT_BASE`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::not_found`] when git does not know the worktree, the git
    /// error of the `status`/`diff` calls, or the registry error.
    pub async fn change_summary(&self, name: &str) -> Result<WorktreeChangeSummary> {
        let target = self.target_path(name)?;
        let repo = self.ensure_repository().await?;
        let info = self.require_listed(name, &repo, &target).await?;
        let index = WorktreeIndex::load(&self.index_path())?;
        let base = index.get(name).and_then(|record| {
            record
                .head
                .clone()
                .or_else(|| record.base.clone().filter(|base| base != DEFAULT_BASE))
        });
        changes::change_summary(self.runner.as_ref(), &info.path, base.as_deref()).await
    }

    /// Reaps expired ephemeral worktrees under `policy`.
    ///
    /// Runs the cleanup pass under the repository lock and reports the pass with
    /// [`WorktreeEvent::Cleaned`]. The pass is fail-closed: see the cleanup
    /// module for the exact skip rules.
    ///
    /// # Errors
    ///
    /// Returns the lock error, the git error of listing/removing worktrees or of
    /// the final prune, or the registry error.
    pub async fn cleanup(&self, policy: &CleanupPolicy) -> Result<CleanupReport> {
        let guard = self.acquire_repo_lock("cleanup").await?;
        let outcome = cleanup::run(self, policy).await;
        drop(guard);
        match outcome {
            Ok(report) => {
                self.events
                    .emit(WorktreeEvent::Cleaned {
                        dry_run: report.dry_run,
                        removed: report.removed.clone(),
                        skipped: report.skipped.len(),
                    })
                    .await;
                Ok(report)
            }
            Err(error) => {
                self.report_failure("cleanup", &error).await;
                Err(error)
            }
        }
    }

    /// Builds the registry record of a worktree.
    fn record_for(
        &self,
        name: &str,
        slug: &str,
        path: &Path,
        branch: Option<&str>,
        base: Option<&str>,
    ) -> WorktreeRecord {
        let mut record = WorktreeRecord::new(name, path.to_path_buf())
            .with_created_at(chrono::Utc::now().to_rfc3339());
        slug.clone_into(&mut record.slug);
        if let Some(branch) = branch {
            record.branch = Some(branch.to_owned());
        }
        if let Some(base) = base {
            record.base = Some(base.to_owned());
        }
        if let Some(owner) = self.owner.clone() {
            record.owner = Some(owner);
        }
        record
    }

    /// Inserts `record` into the registry and saves it.
    fn store_record(&self, record: &WorktreeRecord) -> Result<()> {
        let path = self.index_path();
        let mut index = WorktreeIndex::load(&path)?;
        index.upsert(record.clone());
        index.save(&path)
    }

    /// Resolves the git entry of `target`, refusing one git does not list.
    async fn require_listed(&self, name: &str, repo: &Path, target: &Path) -> Result<WorktreeInfo> {
        git::worktree_list(self.runner.as_ref(), repo)
            .await?
            .into_iter()
            .find(|info| same_path(&info.path, target))
            .ok_or_else(|| {
                Error::not_found(format!(
                    "worktree `{name}` is not registered with git at `{}`",
                    target.display()
                ))
            })
    }

    /// Takes the repository lock for `action`, reporting a failure event.
    ///
    /// The returned guard releases the advisory lock on drop; callers hold it
    /// only for the operation's critical section and release it before emitting
    /// lifecycle events.
    async fn acquire_repo_lock(&self, action: &str) -> Result<WorktreeLockGuard> {
        match self.repo_lock().acquire().await {
            Ok(guard) => Ok(guard),
            Err(error) => {
                let error = Error::from(error);
                self.report_failure(action, &error).await;
                Err(error)
            }
        }
    }

    /// Emits a failure event; the error is still returned to the caller.
    async fn report_failure(&self, action: &str, error: &Error) {
        self.events
            .emit(WorktreeEvent::Error {
                action: action.to_owned(),
                message: error.to_string(),
            })
            .await;
    }
}

/// Compares two paths after lexical normalisation.
fn same_path(left: &Path, right: &Path) -> bool {
    normalize_lexically(left) == normalize_lexically(right)
}
