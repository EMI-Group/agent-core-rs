//! Git-worktree helper: create, list, lock and clean up worktrees through `git`.
//!
//! Port of `openjiuwen/harness/tools/worktree/`: the harness-side helper that
//! gives an agent (or a team member) an isolated checkout without cloning the
//! repository. Python drives `git` with `asyncio.create_subprocess_exec` and
//! persists session state through pydantic models; this port keeps the observable
//! behaviour — layout, refusals, porcelain parsing, lifecycle events — and is
//! idiomatic Rust otherwise.
//!
//! # Layout
//!
//! A worktree named `feature-x` lives at `<repo-root>/.worktrees/feature-x` and
//! tracks the branch `worktree-feature-x`. That is the exact layout the harness
//! prompt guidance gives the model
//! ([`crate::tools::descriptions`]), so a model that follows the prompt and this
//! helper agree on where worktrees go.
//!
//! The registry of created worktrees lives at
//! `<workspace>/.worktree/index.json` — the directory
//! [`crate::workspace::Workspace`] already uses for worktree links — and holds
//! `created_at`, `branch`, `base`, `owner` and `status` per worktree, so a later
//! process can still report what was created and clean up stale entries.
//!
//! # Surface
//!
//! - [`GitRunner`] — the seam; [`ProcessGitRunner`] spawns `git`, tests inject a
//!   scripted runner. A non-zero exit is a typed error carrying git's stderr.
//! - [`WorktreeManager`] — `create`, `list`, `remove`, `prune`, `lock`,
//!   `unlock`, `status`, `cleanup_stale`, plus the layout/registry accessors.
//! - [`WorktreeEvent`] / [`WorktreeEventSink`] — lifecycle events with a
//!   broadcast sink and a recording sink.
//! - [`WorktreeLock`] — the repo-bucketed cross-process advisory lock
//!   (Python's `lock.py`): one exclusive lock per repository root, a bounded
//!   wait, and diagnostic `pid`/`acquired_at` metadata in the lock file.
//! - [`WorktreeTool`] — the model-facing `worktree` tool (actions `create`,
//!   `list`, `remove`, `lock`, `unlock`, `prune`, `status`), mounted by
//!   [`crate::tools::ToolRegistry`] as
//!   [`BuiltinTool::Worktree`](crate::tools::BuiltinTool::Worktree), the last of
//!   the extended tools, and built from a [`WorktreeManager`] plus the registry
//!   language and permission engine.
//! - [`SparseConfig`] / [`SparseMode`] — the sparse-checkout selection (cone by
//!   default, `--no-cone` patterns otherwise) with host-side path validation;
//!   `git::sparse_checkout_set` and friends are the command helpers, and
//!   [`WorktreeRecord::sparse`] persists the selection for re-entry.
//! - [`WorktreeChangeSummary`] / [`change_summary`] / [`render_change_summary`]
//!   — bounded change detection (`status`, `<base>..HEAD` diffstat and count,
//!   unpushed probe) and its deterministic bilingual rendering.
//!
//! # Constraints
//!
//! - Offline and deterministic: no network, no wall-clock ordering in tests
//!   (the registry timestamps are informational only), and the one test that
//!   runs real git skips when `which::which("git")` fails.
//! - Typed errors only: `openjiuwen_common::error::{Error, Result}`; the tool
//!   turns expected failures into `success = false` payloads and never panics.
//! - No `unsafe`, no `println!` (tracing only), no new dependencies.
//! - Deferred (documented in this module's `CONTEXT.md`): automatic sparse
//!   setup inside `create`/`status` (the helpers and the persisted record exist,
//!   the manager wiring is pending), symlinked heavy directories,
//!   include-pattern copying, hook-path configuration, lifecycle policies and
//!   auto-cleanup timers, and the Python `enter_worktree`/`exit_worktree` tools
//!   plus the rails integration.
mod changes;
mod events;
mod git;
mod lock;
mod manager;
mod models;
mod slug;
mod sparse;
mod store;
mod tool;

pub use changes::{
    change_summary, parse_diff_stat, render_change_summary, status_paths, WorktreeChangeSummary,
    SUMMARY_MAX_CHARS, SUMMARY_MAX_FILES, SUMMARY_MAX_LINES,
};

pub use events::{
    BroadcastWorktreeEvents, NoopWorktreeEvents, RecordingWorktreeEvents, SharedWorktreeEvents,
    WorktreeEvent, WorktreeEventSink, DEFAULT_EVENT_CAPACITY,
};
pub use git::{
    branch_exists, checkout_head, current_branch, diff_stat, find_git_root, git_available,
    git_failure_error, has_unpushed_commits, is_not_a_repository, is_not_installed,
    is_sparse_failure, not_a_repository_error, not_installed_error, parse_status_porcelain,
    parse_worktree_list, rev_list_count, rev_parse, sparse_checkout_disable,
    sparse_checkout_reapply, sparse_checkout_set, sparse_failure_error, status_porcelain,
    worktree_add, worktree_add_existing, worktree_add_no_checkout,
    worktree_add_no_checkout_existing, worktree_list, worktree_lock, worktree_prune,
    worktree_remove, worktree_unlock, GitOutput, GitRunner, ProcessGitRunner, GIT_BINARY,
    NOT_A_REPOSITORY_MARKER, NOT_INSTALLED_MARKER, SPARSE_FAILURE_MARKER,
};
pub use lock::{
    is_lock_contention, is_lock_timeout, WorktreeLock, WorktreeLockConfig, WorktreeLockError,
    WorktreeLockGuard, WorktreeLockMetadata, DEFAULT_LOCK_TIMEOUT, LOCK_CONTENTION_MARKER,
    LOCK_TIMEOUT_MARKER, STALE_LOCK_AGE,
};
pub use manager::{WorktreeManager, DEFAULT_BASE};
pub use models::{
    WorktreeCreateResult, WorktreeInfo, WorktreeRecord, WorktreeState, WorktreeStatus,
};
pub use slug::{
    branch_name, slugify, unique_slug, validate_raw_name, BRANCH_PREFIX, DEFAULT_SLUG,
    MAX_SLUG_LENGTH, WORKTREES_DIR,
};
pub use sparse::{validate_sparse_path, SparseConfig, SparseMode, MAX_SPARSE_PATHS};
pub use store::{index_path, WorktreeIndex, INDEX_DIR, INDEX_FILE, INDEX_VERSION};
pub use tool::{WorktreeArgs, WorktreeTool, ACTIONS, TOOL_NAME};

#[cfg(test)]
mod tests;
