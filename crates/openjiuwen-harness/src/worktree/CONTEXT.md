# worktree
## Intent
A real git-worktree helper for the DeepAgent harness, ported from `openjiuwen/harness/tools/worktree/`: create, list, lock, unlock, remove, prune and inspect worktrees through `git`, with a persisted registry and lifecycle events.
It gives an agent (or a team member) an isolated checkout at `<repo-root>/.worktrees/<slug>` without cloning the repository.
The model-facing `worktree` tool is the only model entry point; everything else is host API.
## API Surface
`WorktreeManager` — `new(repo_root)`, `with_runner(repo_root, Arc<dyn GitRunner>)`, `with_workspace_root`, `with_owner`, `with_events`, `with_broadcast_events(capacity)`, `repo_root`, `workspace_root`, `owner`, `runner`, `worktrees_dir`, `index_path`, `target_path(name)`, `ensure_repository`, `create(name, branch, base) -> Result<WorktreeCreateResult>`, `list() -> Result<Vec<WorktreeInfo>>`, `remove(name, force) -> Result<WorktreeInfo>`, `prune() -> Result<usize>`, `cleanup_stale() -> Result<usize>`, `lock(name, reason)`, `unlock(name)`, `status(name) -> Result<WorktreeStatus>`.
`GitRunner` (`#[async_trait]`) — `async fn run(&self, args: &[&str], cwd: &Path) -> Result<GitOutput>`; `run` resolves only on exit code 0.
`ProcessGitRunner` — `new()`, `with_program(program)`, `program()`; spawns git with `GIT_TERMINAL_PROMPT=0`, `GIT_ASKPASS=""`, stdin null and checks `which::which` first.
`GitOutput { stdout, stderr }`, `git_available()`, `parse_worktree_list(text) -> Vec<WorktreeInfo>`, `parse_status_porcelain(text) -> Vec<String>`.
Git helpers — `find_git_root`, `current_branch`, `rev_parse`, `branch_exists`, `worktree_list`, `worktree_add`, `worktree_add_existing`, `worktree_remove`, `worktree_prune`, `worktree_lock`, `worktree_unlock`, `status_porcelain`.
Typed errors — `not_installed_error(program)` / `NOT_INSTALLED_MARKER` (`Error::config`, "git is not installed"), `not_a_repository_error(cwd)` / `NOT_A_REPOSITORY_MARKER` (`Error::config`), `git_failure_error(args, stderr)` (`Error::tool`, carries git stderr), `is_not_installed`, `is_not_a_repository`.
Slug helpers — `validate_raw_name(raw)` (security gate), `slugify(raw)` (normaliser), `unique_slug(slug, taken)`, `branch_name(raw)`, `MAX_SLUG_LENGTH` (64), `DEFAULT_SLUG` ("worktree"), `BRANCH_PREFIX` ("worktree-"), `WORKTREES_DIR` (".worktrees").
Models — `WorktreeInfo { path, branch, head, bare, detached, locked, lock_reason, prunable, recorded }` with `name()` / `is_detached()`; `WorktreeCreateResult { name, slug, path, branch, base, head, existed }`; `WorktreeStatus { name, path, branch, head, locked, lock_reason, dirty, changed_files, tracked }` with `is_clean()`; `WorktreeRecord { name, slug, path, branch, base, owner, status, created_at }`; `WorktreeState { Active, Removed }`.
Registry — `WorktreeIndex { version, records }` with `load(path)`, `save(path)`, `get(name)`, `get_by_path`, `upsert`, `remove`, `records`, `retain_existing(paths)`, plus `index_path(workspace_root)`, `INDEX_DIR` (".worktree"), `INDEX_FILE` ("index.json"), `INDEX_VERSION` (1).
Events — `WorktreeEvent::{Created, Removed, Locked, Unlocked, Pruned, Error}` (serde tag `type`, `kind()`); `WorktreeEventSink` (`async fn emit(&self, WorktreeEvent)`); `NoopWorktreeEvents`, `BroadcastWorktreeEvents` (`new(capacity)`, `subscribe()`, `receiver_count()`), `RecordingWorktreeEvents` (`events()`, `clear()`), `SharedWorktreeEvents`, `DEFAULT_EVENT_CAPACITY` (64).
Tool — `WorktreeTool::new(manager: Arc<WorktreeManager>, language: &str) -> WorktreeTool`, `WorktreeTool::card_for(language) -> ToolCard`, `with_permission(engine: Arc<PermissionEngine>)`, `manager()`; `WorktreeArgs { action, name, branch, base, force, reason }` with `new(action)`, `with_name(name)`, `is_known_action()`; `TOOL_NAME` ("worktree"), `ACTIONS` (`create`, `list`, `remove`, `lock`, `unlock`, `prune`, `status`).
## Lock (`lock.rs`)

`WorktreeLock` — `new(repo_root, workspace_root, config)`, `config()`, `repo_root()`, `root()`, `lock_path()`, `is_enabled()`, `acquire()`, `try_acquire()`, `read_metadata()`, `stale_metadata(now)`.
`WorktreeLockConfig { enabled, timeout }` — `default()` (enabled, 120 s), `disabled()`, `with_timeout(duration)`; `DEFAULT_LOCK_TIMEOUT` (120 s), `STALE_LOCK_AGE` (30 min).
`WorktreeLockGuard` — `lock_path()`, `is_active()`, `metadata()`; dropping it releases the `flock`.
`WorktreeLockMetadata { pid, acquired_at }` — RFC 3339 `acquired_at`, diagnostics only.
`WorktreeLockError` — `Contended`/`TimedOut`/`Io` with `target()`, `lock_path()`, `waited()`, `is_timeout()`, `is_contention()`, `Display`, `std::error::Error`, `From<WorktreeLockError> for Error` (`Error::tool`).
`is_lock_timeout(&Error)` / `is_lock_contention(&Error)` — marker checks; `LOCK_TIMEOUT_MARKER` ("worktree lock timeout") and `LOCK_CONTENTION_MARKER` ("worktree lock is held").
The advisory `flock` itself comes from `openjiuwen_sysop::fs::lock::{FileLockManager, LockMode, LockError}`; the module only adds layout, the timeout policy, metadata and the typed error.

## Layout
A worktree named `Feature Login!` becomes `slug = "Feature-Login"`, directory `<repo-root>/.worktrees/Feature-Login`, branch `worktree-Feature-Login`.
The registry is `<workspace-root>/.worktree/index.json` (default workspace root is the repo root), the directory `crate::workspace::Workspace` already uses for worktree links.
`create` recovers an existing worktree at the target path (`existed = true`) and refuses a target directory that git does not list.
`create` checks out an existing branch instead of re-creating it; `base` defaults to `HEAD`.
`remove` refuses a dirty worktree without `force` and reports an unlisted worktree as `Error::not_found`.
`prune` runs `git worktree prune` and then drops registry records whose directory git no longer lists.

## Lock layout
The lock is repo-bucketed: the hashed target is `repo_root`, so two worktrees of one repository share one lock and different repositories never contend.
The sidecar lives in `<repo_root>/.git` when that is an existing directory, otherwise in `<workspace_root>/.worktree`; its name is `<root>/<sha256-of-repo-root>.lock`.
The lock is always exclusive (write), never read/write, because every worktree critical section mutates shared git state.
A configured zero timeout becomes a 1 ms budget so one attempt always happens (the seam reads `None` as "wait forever").
Acquisition writes `pid=<u32>` and `acquired_at=<rfc3339>` into the sidecar best-effort, and only while the exclusive lock is held.
`flock` is released by the kernel when the holder exits, so a crashed holder never leaves a blocking stale lock; the metadata is what outlives it and `stale_metadata` reports it. The lock file is never broken or unlinked while another process may hold it.

## Constraints
Offline and deterministic by default: tests use `ScriptedGitRunner` (queued stdout/failures, recorded calls) and temp directories; the single real-git test skips when `which::which("git")` fails.
No new dependencies, no `unsafe`, no `println!` (tracing only); errors are `openjiuwen_common::error::{Error, Result}`.
`slug::validate_raw_name` runs before `slugify`, so `..`, absolute paths, `~` and drive prefixes are refused instead of normalised.
`target_path` normalises lexically and asserts the result stays under the repo root.
The registry document is written atomically (sibling `.tmp` + rename) and a missing file loads as empty while malformed JSON is `ErrorKind::Validation`.
The tool never returns `Err` for expected failures: it returns a `ToolPayload` with `success = false`; only cancellation propagates.
The tool consults `crate::permission` path guards for the target path on mutating actions and blocks only on `PermissionLevel::Deny` (`ask`/`allow` pass, no engine fails open).
The tool is mounted by `crate::tools::ToolRegistry` as `BuiltinTool::Worktree`, the last of the extended tools, built from a `WorktreeManager` plus the registry language and permission engine.
`WorktreeLock` supplies the repo-scoped cross-process advisory lock, but `WorktreeManager` operations do not take it: callers wrap create/remove in `acquire` themselves, so unmodified managers over one workspace are still uncoordinated.
The lock guard is an OS advisory lock on a `Send` descriptor, not a `parking_lot`/`std` mutex, so it may be held across the manager's git `.await` points; no `parking_lot`/`std` lock is ever held across `.await`.
Where `flock` is missing (non-Unix) or the lock is disabled, `acquire` returns an inactive guard and writes no metadata: exclusion is best-effort and `is_enabled()` is the check for a caller that needs it.
## Deferred
Python's module-level `asyncio.Lock` bucket in `lock.py`: the OS advisory lock already excludes two handles in one process, so the bucket has no extra work to do.
Python's `filelock`-specific lock file name (`openjiuwen-worktree.lock`) and its beside-the-repo fallback: this port uses the sysop sidecar naming and the workspace `.worktree` fallback.
Sparse checkout, `symlink_directories`, `include_patterns` copying, `core.longpaths`, hook-path configuration.
Lifecycle policies (`ephemeral`/`durable`/`auto`), `cleanup_after_days`, auto-cleanup on shutdown, event-log recovery of sessions.
`enter_worktree`/`exit_worktree` tools, the worktree rails, backend hooks (`WorktreeBackend`), change summaries (commits since creation, unpushed commits), and the `vcs` module.
## Routing Table
- `./git.rs` → `GitRunner` seam, `ProcessGitRunner`, porcelain parsers, typed git errors, command helpers
- `./slug.rs` → raw-name validation, `slugify`, `unique_slug`, branch naming, layout constants
- `./models.rs` → `WorktreeInfo`, `WorktreeCreateResult`, `WorktreeStatus`, `WorktreeRecord`, `WorktreeState`
- `./store.rs` → `WorktreeIndex` registry and atomic JSON writes
- `./events.rs` → `WorktreeEvent`, `WorktreeEventSink`, noop/broadcast/recording sinks
- `./lock.rs` → `WorktreeLock`, config, guard, metadata and typed contention/timeout errors
- `./manager.rs` → `WorktreeManager` lifecycle, layout and refusal rules
- `./tool.rs` → `WorktreeTool` and `WorktreeArgs` (the model-facing `worktree` tool)
- `./tests/` → `mod.rs` (scripted runner and fixtures), `manager.rs` (lifecycle scenarios, real-git round trip), `lock.rs` (lock layout, contention, timeout, metadata), `tool.rs` (card, dispatch, payload failures)
