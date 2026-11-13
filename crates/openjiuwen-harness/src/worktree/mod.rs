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
//! - [`WorktreeTool`] — the model-facing `worktree` tool (actions `create`,
//!   `list`, `remove`, `lock`, `unlock`, `prune`, `status`). It is not mounted by
//!   [`crate::tools::ToolRegistry`] yet; the registry wiring is a separate change.
//!
//! # Constraints
//!
//! - Offline and deterministic: no network, no wall-clock ordering in tests
//!   (the registry timestamps are informational only), and the one test that
//!   runs real git skips when `which::which("git")` fails.
//! - Typed errors only: `openjiuwen_common::error::{Error, Result}`; the tool
//!   turns expected failures into `success = false` payloads and never panics.
//! - No `unsafe`, no `println!` (tracing only), no new dependencies.
//! - Deferred (documented in this module's `CONTEXT.md`): the repo-wide
//!   cross-process file lock, sparse checkout, symlinked heavy directories,
//!   include-pattern copying, hook-path configuration, lifecycle policies and
//!   auto-cleanup timers, and the Python `enter_worktree`/`exit_worktree` tools
//!   plus the rails integration.
mod events;
mod git;
mod manager;
mod models;
mod slug;
mod store;
mod tool;

pub use events::{
    BroadcastWorktreeEvents, NoopWorktreeEvents, RecordingWorktreeEvents, SharedWorktreeEvents,
    WorktreeEvent, WorktreeEventSink, DEFAULT_EVENT_CAPACITY,
};
pub use git::{
    branch_exists, current_branch, find_git_root, git_available, git_failure_error,
    is_not_a_repository, is_not_installed, not_a_repository_error, not_installed_error,
    parse_status_porcelain, parse_worktree_list, rev_parse, status_porcelain, worktree_add,
    worktree_add_existing, worktree_list, worktree_lock, worktree_prune, worktree_remove,
    worktree_unlock, GitOutput, GitRunner, ProcessGitRunner, GIT_BINARY, NOT_A_REPOSITORY_MARKER,
    NOT_INSTALLED_MARKER,
};
pub use manager::{WorktreeManager, DEFAULT_BASE};
pub use models::{
    WorktreeCreateResult, WorktreeInfo, WorktreeRecord, WorktreeState, WorktreeStatus,
};
pub use slug::{
    branch_name, slugify, unique_slug, validate_raw_name, BRANCH_PREFIX, DEFAULT_SLUG,
    MAX_SLUG_LENGTH, WORKTREES_DIR,
};
pub use store::{index_path, WorktreeIndex, INDEX_DIR, INDEX_FILE, INDEX_VERSION};
pub use tool::{WorktreeArgs, WorktreeTool, ACTIONS, TOOL_NAME};

#[cfg(test)]
mod tests;
