# worktree
## Intent
A real git-worktree helper for the DeepAgent harness, ported from `openjiuwen/harness/tools/worktree/`: create, list, lock, unlock, remove, prune and inspect worktrees through `git`, with a persisted registry and lifecycle events.
It gives an agent (or a team member) an isolated checkout at `<repo-root>/.worktrees/<slug>` without cloning the repository.
The model-facing entry points are the `worktree`, `enter_worktree` and `exit_worktree` tools; the `WorktreeRail` publishes the bounded worktree context section, and everything else is host API.
`enter_worktree`/`exit_worktree` are not mounted by `ToolRegistry` yet (the registry still mounts only the action-dispatching `worktree` tool); a host wires them directly.
## API Surface
`WorktreeManager` — `new(repo_root)`, `with_runner(repo_root, Arc<dyn GitRunner>)`, `with_workspace_root`, `with_owner`, `with_events`, `with_broadcast_events(capacity)`, `with_lock_config(WorktreeLockConfig)`, `with_sparse(SparseConfig)`, `repo_root`, `workspace_root`, `owner`, `runner`, `lock_config()`, `repo_lock() -> WorktreeLock`, `lock_path() -> PathBuf`, `sparse()`, `worktrees_dir`, `index_path`, `target_path(name)`, `ensure_repository`, `create(name, branch, base) -> Result<WorktreeCreateResult>`, `create_with_options(name, branch, base, &WorktreeCreateOptions) -> Result<WorktreeCreateResult>`, `list() -> Result<Vec<WorktreeInfo>>`, `remove(name, force) -> Result<WorktreeInfo>`, `prune() -> Result<usize>`, `cleanup_stale() -> Result<usize>`, `cleanup(&CleanupPolicy) -> Result<CleanupReport>`, `lock(name, reason)`, `unlock(name)`, `status(name) -> Result<WorktreeStatus>`, `apply_sparse(name) -> Result<Option<SparseConfig>>`, `change_summary(name) -> Result<WorktreeChangeSummary>`.
`WorktreeCreateOptions { sparse: Option<SparseConfig> }` (serde default) — `with_sparse(sparse)`, `is_default()`; a per-call selection overrides `WorktreeManager::with_sparse`.
`GitRunner` (`#[async_trait]`) — `async fn run(&self, args: &[&str], cwd: &Path) -> Result<GitOutput>`; `run` resolves only on exit code 0.
`ProcessGitRunner` — `new()`, `with_program(program)`, `program()`; spawns git with `GIT_TERMINAL_PROMPT=0`, `GIT_ASKPASS=""`, stdin null and checks `which::which` first.
`GitOutput { stdout, stderr }`, `git_available()`, `parse_worktree_list(text) -> Vec<WorktreeInfo>`, `parse_status_porcelain(text) -> Vec<String>`.
Git helpers — `find_git_root`, `current_branch`, `rev_parse`, `branch_exists`, `worktree_list`, `worktree_add`, `worktree_add_existing`, `worktree_remove`, `worktree_prune`, `worktree_lock`, `worktree_unlock`, `status_porcelain`.
Sparse git helpers — `worktree_add_no_checkout(runner, repo, path, branch, base)`, `worktree_add_no_checkout_existing(runner, repo, path, branch)`, `sparse_checkout_set(runner, cwd, &SparseConfig)`, `checkout_head`, `sparse_checkout_reapply`, `sparse_checkout_disable`.
Change-summary git helpers — `diff_stat(runner, cwd, base: Option<&str>) -> Result<String>`, `rev_list_count(runner, cwd, range) -> Result<Option<u64>>`, `has_unpushed_commits(runner, cwd) -> Result<Option<bool>>`.
Typed errors — `not_installed_error(program)` / `NOT_INSTALLED_MARKER` (`Error::config`, "git is not installed"), `not_a_repository_error(cwd)` / `NOT_A_REPOSITORY_MARKER` (`Error::config`), `git_failure_error(args, stderr)` (`Error::tool`, carries git stderr), `is_not_installed`, `is_not_a_repository`, `sparse_failure_error(path, stderr)` / `SPARSE_FAILURE_MARKER` ("sparse checkout failed", `Error::tool`) / `is_sparse_failure`.
Sparse config (`sparse.rs`) — `SparseConfig { paths, mode }` with `new(paths, mode)`, `cone(paths)`, `patterns(paths)`, `is_empty()`, `validate()`, `normalized()`; `SparseMode { Cone (default), Patterns }` with `as_str()` / `git_flag()`; `validate_sparse_path(path)`; `MAX_SPARSE_PATHS` (256).
Change summaries (`changes.rs`) — `WorktreeChangeSummary { changed_files, files, commits, unpushed, diffstat, truncated }` with `is_dirty()` / `is_clean()`; `change_summary(runner, worktree, base)`, `render_change_summary(summary, language)`, `status_paths(text)`, `parse_diff_stat(text)`; `SUMMARY_MAX_FILES` (20), `SUMMARY_MAX_LINES` (40), `SUMMARY_MAX_CHARS` (2000).
Slug helpers — `validate_raw_name(raw)` (security gate), `slugify(raw)` (normaliser), `unique_slug(slug, taken)`, `branch_name(raw)`, `MAX_SLUG_LENGTH` (64), `DEFAULT_SLUG` ("worktree"), `BRANCH_PREFIX` ("worktree-"), `WORKTREES_DIR` (".worktrees").
Models — `WorktreeInfo { path, branch, head, bare, detached, locked, lock_reason, prunable, recorded }` with `name()` / `is_detached()`; `WorktreeCreateResult { name, slug, path, branch, base, head, existed, sparse }`; `WorktreeStatus { name, path, branch, head, locked, lock_reason, dirty, changed_files, tracked, sparse }` with `is_clean()`; `WorktreeRecord { name, slug, path, branch, base, owner, status, created_at, sparse, head }` with `with_sparse` / `with_head`; `WorktreeState { Active, Removed }`.
Registry — `WorktreeIndex { version, records }` with `load(path)`, `save(path)`, `get(name)`, `get_by_path`, `upsert`, `remove`, `records`, `retain_existing(paths)`, plus `index_path(workspace_root)`, `INDEX_DIR` (".worktree"), `INDEX_FILE` ("index.json"), `INDEX_VERSION` (1).
Events — `WorktreeEvent::{Created, Removed, Locked, Unlocked, Entered { name, path, branch, existed }, Exited { name, path, action }, Pruned, SparseConfigured { name, path, mode, paths }, Cleaned { dry_run, removed, skipped }, Error}` (serde tag `type`, `kind()`); `WorktreeEventSink` (`async fn emit(&self, WorktreeEvent)`); `NoopWorktreeEvents`, `BroadcastWorktreeEvents` (`new(capacity)`, `subscribe()`, `receiver_count()`), `RecordingWorktreeEvents` (`events()`, `clear()`), `SharedWorktreeEvents`, `DEFAULT_EVENT_CAPACITY` (64).
Cleanup (`cleanup.rs`) — `CleanupPolicy { max_age_days, ephemeral_only, dry_run, current_worktree, now }` with `with_max_age_days`, `with_ephemeral_only`, `with_dry_run`, `with_current_worktree`, `with_now`, `reference_time()`, `max_age_secs()`; `CleanupCandidate { name, path, age_secs }`, `CleanupSkip { name, path, reason }`, `CleanupReport { dry_run, candidates, removed, skipped }` with `removed_count()`; `is_ephemeral_slug(slug)`, `EPHEMERAL_PATTERNS`, `DEFAULT_CLEANUP_AFTER_DAYS` (30).
Tool — `WorktreeTool::new(manager: Arc<WorktreeManager>, language: &str) -> WorktreeTool`, `WorktreeTool::card_for(language) -> ToolCard`, `with_permission(engine: Arc<PermissionEngine>)`, `manager()`; `WorktreeArgs { action, name, branch, base, force, reason }` with `new(action)`, `with_name(name)`, `is_known_action()`; `TOOL_NAME` ("worktree"), `ACTIONS` (`create`, `list`, `remove`, `lock`, `unlock`, `prune`, `status`).
Session tools (`tools.rs`) — `EnterWorktreeTool::new(Arc<WorktreeManager>, language)`, `with_sessions(SharedWorktreeSessions)`, `with_permission(Arc<PermissionEngine>)`, `with_original_cwd(cwd)`, `manager()`, `card_for(language)`; `ExitWorktreeTool::new(Arc<WorktreeManager>, language)`, `with_sessions`, `with_permission`, `manager()`, `card_for(language)`; both implement `openjiuwen_foundation::tool::Tool` (`card`, `invoke`) and dispatch through a private async `run`.
Session arguments — `EnterWorktreeArgs { name: Option<String> }` (`#[serde(deny_unknown_fields)]`, `new()`, `with_name(name)`); `ExitWorktreeArgs { action: String, discard_changes: bool }` (`new(action)`, `is_known_action()`).
Session state (`tools.rs`) — `WorktreeSessionState { original_cwd, worktree_path, worktree_name, worktree_branch, original_branch, original_head_commit, member_name, team_name, hook_based, lifecycle_policy ("auto" default), team_lifecycle, creation_duration_ms, used_sparse_paths }` with `Default`, `new(name, path, branch: Option<String>)`, `to_json()`, `from_json(&Value)`; all fields serde-default and tolerant.
Session store (`tools.rs`) — `WorktreeSessionStore` (`session()`, `set_session(Option<…>)`, `default_name()`, `set_default_name(Option<String>)`), `MemoryWorktreeSessionStore` (`new()`, `with_session(state)`, `with_default_name(name)`), `SharedWorktreeSessions = Arc<dyn WorktreeSessionStore>`.
Session constants (`tools.rs`) — `ENTER_TOOL_NAME` ("enter_worktree"), `EXIT_TOOL_NAME` ("exit_worktree"), `EXIT_ACTIONS` (`keep`, `remove`), `WORKTREE_SESSION_KEY` ("_worktree_session"), `WORKTREE_DEFAULT_NAME_KEY` ("_worktree_default_name").
Rail (`rails.rs`) — `WorktreeRail` (`new(manager, sessions, language)`, `with_worktree(name)`, `with_priority(priority)`, `manager()`, `language()`, `worktree_name()`, impl `HarnessRail`); `WorktreeContext { name, path, branch, summary }` (serde snake_case/default) with `render(language)`; `build_worktree_notice(parent_cwd, worktree_cwd, language)`; `WORKTREE_SECTION_NAME` ("worktree"), `WORKTREE_SECTION_PRIORITY` (85), `WORKTREE_SECTION_MAX_CHARS` (1200), `WORKTREE_RAIL_PRIORITY` (100), `WORKTREE_SUMMARY_EXTRA_KEY` ("harness.worktree_summary").
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
`create_with_options`/`create` record the HEAD commit and the effective sparse selection in the registry record and in `WorktreeCreateResult`.
A recovered worktree re-applies the selection recorded in the registry (never `checkout HEAD`, which would disturb uncommitted work) and reports `SparseConfigured`.
`status` reports the sparse selection of the active registry record; `list` keeps attaching records through `WorktreeInfo::recorded`.
`remove` refuses a dirty worktree without `force` and reports an unlisted worktree as `Error::not_found`.
`prune` runs `git worktree prune` and then drops registry records whose directory git no longer lists.

## Locking policy
Every mutating operation — `create`/`create_with_options`, `remove`, `prune`, `cleanup_stale`, `cleanup`, `lock`, `unlock`, `apply_sparse` — takes the repo lock for its critical section, matching Python's `async with repo_lock(repo_root, timeout=config.repo_lock_timeout)`; lock acquisition issues no git command, so command sequences are unchanged.
A lock failure becomes the typed `Error` (still recognised by `is_lock_timeout`/`is_lock_contention`) and is reported as `WorktreeEvent::Error` with the operation name.
The guard is dropped before lifecycle events are emitted, and read-only paths (`list`, `status`, `change_summary`) stay outside the lock on purpose.
`WorktreeLockConfig::disabled()` opts a manager out without changing any behaviour beyond the lock.

## Lock layout
The lock is repo-bucketed: the hashed target is `repo_root`, so two worktrees of one repository share one lock and different repositories never contend.
The sidecar lives in `<repo_root>/.git` when that is an existing directory, otherwise in `<workspace_root>/.worktree`; its name is `<root>/<sha256-of-repo-root>.lock`.
The lock is always exclusive (write), never read/write, because every worktree critical section mutates shared git state.
A configured zero timeout becomes a 1 ms budget so one attempt always happens (the seam reads `None` as "wait forever").
Acquisition writes `pid=<u32>` and `acquired_at=<rfc3339>` into the sidecar best-effort, and only while the exclusive lock is held.
`flock` is released by the kernel when the holder exits, so a crashed holder never leaves a blocking stale lock; the metadata is what outlives it and `stale_metadata` reports it. The lock file is never broken or unlinked while another process may hold it.

## Sparse checkout and change summaries
A sparse creation runs `worktree add --no-checkout` (with `-b` for a new branch, without it for an existing one), then `sparse-checkout set`, then `checkout HEAD`.
If either sparse step fails, the half-created worktree is removed with `worktree remove --force` (errors ignored) and the caller gets `sparse_failure_error` (`is_sparse_failure` holds).
The effective selection is `WorktreeCreateOptions::sparse`, else the manager-wide `with_sparse`; it is validated before the first git command.
`apply_sparse(name)` re-validates and re-applies the recorded selection under the lock, emits `SparseConfigured`, and answers `Ok(None)` when the worktree has no record or no selection.
`WorktreeManager::change_summary` resolves the worktree through `worktree list`, uses the record `head` (falling back to `base` when that is not `HEAD`) as the diff base, and runs outside the lock.
Cone mode is the default and maps to `git sparse-checkout set --cone -- <paths>`; `SparseMode::Patterns` maps to `--no-cone` with the entries passed through as git patterns.
Sparse paths are repository-relative; `validate_sparse_path` refuses absolute paths, `~`, drive prefixes, `.`/`..` segments and NUL with `Error::security`, and empty/control input with `Error::validation`, before any normalisation happens.
`SparseConfig::normalized` only trims, converts `\` to `/`, strips a trailing `/`, drops empty entries and collapses duplicates in first-seen order, so a hostile path is refused rather than rewritten.
`sparse_checkout_set` refuses an empty selection and an invalid path without running git, then uses the normalised paths.
`change_summary` issues `status --porcelain`, then (only with a base) `diff --stat <base>..HEAD` and `rev-list --count <base>..HEAD`, then always `rev-list --max-count=1 HEAD --not --remotes`.
A failing `status` or `diff` propagates the typed error; a failing or unparsable count degrades to `commits = 0` and a failing unpushed probe degrades to `unpushed = None` (fail-closed for cleanup).
`render_change_summary` is deterministic and bilingual (`language == "en"` exactly, otherwise Chinese), reads no clock or environment, and stays within `SUMMARY_MAX_CHARS` with a trailing `...` when anything was truncated.

## Cleanup policy
`CleanupPolicy::default()` is thirty days, ephemeral-only, a real removal pass, no active worktree and the wall clock; `with_now` freezes the reference so tests stay deterministic.
The pass only considers worktrees git lists under `.worktrees` (never the main worktree) whose directory mtime is at least `max_age_days * 86_400` seconds old; an unreadable mtime is a `cannot read worktree metadata` skip.
Ephemeral slugs match `teammate-<8hex>`, `agent-<7hex>` or `agent-<team>-<member>-<8hex>` exactly; other slugs are skipped as `not an ephemeral worktree` unless `ephemeral_only` is off.
The active worktree (`current_worktree`, compared lexically) is never reaped, a dirty worktree is skipped as `uncommitted changes`, and unpushed state is fail-closed: `Some(true)` is `unpushed commits`, `None` is `cannot verify unpushed commits`.
A dry run reports the candidates and performs no removal, no prune and no registry write.
A removal pass runs `worktree remove` per candidate (a failure becomes a skip carrying the git message), then drops registry records git no longer lists, then runs exactly one `worktree prune` when something was removed, and reports `WorktreeEvent::Cleaned`.
## Session tools (`enter_worktree`/`exit_worktree`)
Python keeps the active session in a `ContextVar` and calls `set_cwd(worktree_path)`/`set_original_cwd(...)`; Rust has no safe process-wide cwd switch, so the tools never touch the cwd. They record the session and report `worktree_path`/`original_cwd` in the payload, and the host applies the change in its own cwd abstraction. The messages keep Python's "CWD switched to worktree." wording.
`enter_worktree` rejects unknown fields and resolves the name from a trimmed non-empty `name`, else the store's default name, else fails with "name is required when no worktree session or default name is recorded" (Python generated a random slug instead; this port refuses rather than inventing one).
An active session refuses a second enter with "Already in worktree '<name>'. Exit first with exit_worktree."
The target path comes from `WorktreeManager::target_path`, so traversal, `~`, drive prefixes and absolute names are refused before any git command; the permission engine's `write_file` guard blocks on `deny` (`ask`/`allow` pass, no engine fails open).
A target git already lists is recovered (`existed = true`, `Entered` event with `existed`); a target that only the registry lists as `Active` is a failure naming the `worktree` tool's `prune` action and runs no `worktree add`; otherwise `create_with_options` runs with the default options, so the manager-wide sparse selection still applies.
Enter stores a `WorktreeSessionState` (original cwd, path, name, branch, HEAD as the change baseline, `used_sparse_paths`), records the name as the session default, emits `Entered`, and answers `{worktree_path, worktree_branch, message, existed, sparse}` with the exact Python `Created worktree at …`/`Entered existing worktree at …` message.
`exit_worktree` requires an active session, trims/lowercases `action`, and rejects anything but `keep`/`remove` with "'action' must be 'keep' or 'remove'."; the session's worktree path goes through the same permission guard.
`keep` clears the session (the default name stays), emits `Exited { action: "keep" }`, and answers `{action, original_cwd, worktree_path, worktree_branch, worktree_name, message}` with Python's "Kept worktree …" message.
`remove` runs `change_summary` first and is fail-closed: an unverifiable worktree answers "Could not verify worktree state at …", a dirty one without `discard_changes` answers with the uncommitted-file/commit counts, and only a clean one or an explicitly confirmed one reaches `WorktreeManager::remove` (force only when `discard_changes` is set).
The removal payload carries `discarded_files`/`discarded_commits` from the pre-removal summary; the session is cleared and `Exited { action: "remove" }` is emitted after the manager's `Removed` event.
Both tools turn every expected failure into a `ToolPayload` with `success = false` and only propagate cancellation; their cards are bilingual (`"en"` exactly selects English) and state that `remove` is destructive and needs explicit user confirmation.

## Rail (`rails.rs`)
`WorktreeRail` is a `HarnessRail` named `"worktree"` with priority `WORKTREE_RAIL_PRIORITY` (100, Python's `WorktreeRail.priority`); `with_priority` overrides it and `priority()` reports it.
It publishes one `PromptSection` named `WORKTREE_SECTION_NAME`, priority `WORKTREE_SECTION_PRIORITY` (85), category `SystemPrompt`, carrier `system_message`, and bilingual content (`cn` plus `en`) rendered by `WorktreeContext::render`, from `before_invoke`, `before_model_call` and `after_invoke` through one private async `publish`; publishing is an upsert by name, so repeated firings keep one section.
The active name is the explicit `with_worktree(name)` (trimmed, non-empty), else the `WorktreeSessionStore` session's `worktree_name`; no name means no section, no `extra` write and no git call.
Path and branch come from a session naming the same worktree with a non-empty path, then from `WorktreeManager::list()` matched by directory name or its slug, then from `WorktreeManager::target_path(name)` with no branch.
The summary is `WorktreeManager::change_summary(name)` (lock-free); a failure is logged at `debug` and the section is published without it.
`WorktreeContext::render` names the worktree, its path and its branch, then either the bounded `render_change_summary` text or a note that no summary is available (the worktree is not listed by git, or `git status` failed).
The rendering is capped at `WORKTREE_SECTION_MAX_CHARS` and ends with `...` when anything was cut; it reads no clock and no environment.
The structured summary (serde JSON of `WorktreeChangeSummary`) is written into `HarnessRailContext::extra_mut()` under `WORKTREE_SUMMARY_EXTRA_KEY`; a firing without a summary, or an impossible serialization failure, removes the key instead of leaving a stale entry.
Hooks never return `Err`: resolution, git and serialization failures are logged at `debug` and swallowed, so a rail never aborts a run.
`build_worktree_notice` keeps Python's `notice.py` English text (`"en"` exactly) plus a Chinese counterpart and is a host-side helper: the section carries no parent cwd.
`WorktreeRail`, `WorktreeContext`, `build_worktree_notice` and the five constants are re-exported from `openjiuwen_harness::worktree`, not from the crate root (`src/lib.rs` is untouched).

## Constraints
Offline and deterministic by default: tests use `ScriptedGitRunner` (queued stdout/failures, recorded calls) and temp directories; the single real-git test skips when `which::which("git")` fails.
No new dependencies, no `unsafe`, no `println!` (tracing only); errors are `openjiuwen_common::error::{Error, Result}`.
`slug::validate_raw_name` runs before `slugify`, so `..`, absolute paths, `~` and drive prefixes are refused instead of normalised.
`target_path` normalises lexically and asserts the result stays under the repo root.
The registry document is written atomically (sibling `.tmp` + rename) and a missing file loads as empty while malformed JSON is `ErrorKind::Validation`.
The tool never returns `Err` for expected failures: it returns a `ToolPayload` with `success = false`; only cancellation propagates.
`enter_worktree`/`exit_worktree` are not mounted by `crate::tools::ToolRegistry`; the registry still mounts only the `worktree` tool, so a host wires the session tools (and their shared `WorktreeSessionStore`) directly.
The session tools are re-exported from `openjiuwen_harness::worktree`, not from the crate root: `src/lib.rs` flat re-exports only a subset of the worktree items (`slugify`, `GitRunner`, `ProcessGitRunner`, `WorktreeEvent`, `WorktreeInfo`, `WorktreeManager`, `WorktreeRecord`, `WorktreeTool`).
`tools.rs` never changes the process cwd; a caller must apply the reported `worktree_path`/`original_cwd` in its own cwd stack.
The session tools consult `crate::permission` path guards for the target path on mutating actions and block only on `PermissionLevel::Deny` (`ask`/`allow` pass, no engine fails open).
The tool is mounted by `crate::tools::ToolRegistry` as `BuiltinTool::Worktree`, the last of the extended tools, built from a `WorktreeManager` plus the registry language and permission engine.
`WorktreeManager` acquires the repo lock itself for every mutating operation, so two managers over one workspace are coordinated; `repo_lock()` is exposed for a caller that needs the lock around its own critical section.
The lock guard is an OS advisory lock on a `Send` descriptor, not a `parking_lot`/`std` mutex, so it may be held across the manager's git `.await` points; no `parking_lot`/`std` lock is ever held across `.await`.
Where `flock` is missing (non-Unix) or the lock is disabled, `acquire` returns an inactive guard and writes no metadata: exclusion is best-effort and `is_enabled()` is the check for a caller that needs it.
## Deferred
Python's module-level `asyncio.Lock` bucket in `lock.py`: the OS advisory lock already excludes two handles in one process, so the bucket has no extra work to do.
Python's `filelock`-specific lock file name (`openjiuwen-worktree.lock`) and its beside-the-repo fallback: this port uses the sysop sidecar naming and the workspace `.worktree` fallback.
`symlink_directories`, `include_patterns` copying, `core.longpaths`, hook-path configuration.
Lifecycle policies (`ephemeral`/`durable`/`auto`), auto-cleanup timers on shutdown, event-log recovery of sessions (age-based reaping itself lives in `cleanup::run`/`CleanupPolicy`).
The Python lifecycle rails (`WorktreeLifecycleRail`, `AutoSetupRail` setup commands, commit/sync hooks) and their manager hook chain stay deferred; the prompt-side `WorktreeRail` and the `DiffSummaryRail` summary are implemented in `rails.rs`.
Backend hooks (`WorktreeBackend`) and the `vcs` module remain deferred; the session tools (`enter_worktree`/`exit_worktree`) and the session store are implemented in `tools.rs`.
Mounted through the registry: nothing yet — `enter_worktree`/`exit_worktree` are host-wired, and the rail only reads their session store; `WorktreeRail` is host-registered too.
## Routing Table
- `./git.rs` → `GitRunner` seam, `ProcessGitRunner`, porcelain parsers, typed git errors, command helpers
- `./slug.rs` → raw-name validation, `slugify`, `unique_slug`, branch naming, layout constants
- `./models.rs` → `WorktreeInfo`, `WorktreeCreateResult`, `WorktreeStatus`, `WorktreeRecord`, `WorktreeState`
- `./store.rs` → `WorktreeIndex` registry and atomic JSON writes
- `./events.rs` → `WorktreeEvent`, `WorktreeEventSink`, noop/broadcast/recording sinks
- `./lock.rs` → `WorktreeLock`, config, guard, metadata and typed contention/timeout errors
- `./manager.rs` → `WorktreeManager` lifecycle, layout, refusal rules, locking, sparse wiring, change summaries and cleanup entry point
- `./cleanup.rs` → `CleanupPolicy`, candidate/skip/report models, ephemeral patterns and the fail-closed reaping pass
- `./tool.rs` → `WorktreeTool` and `WorktreeArgs` (the model-facing `worktree` tool)
- `./tools.rs` → `EnterWorktreeTool`/`ExitWorktreeTool`, their arguments, `WorktreeSessionState`, the `WorktreeSessionStore` seam and `MemoryWorktreeSessionStore`
- `./sparse.rs` → `SparseConfig`, `SparseMode`, path validation, normalisation and `MAX_SPARSE_PATHS`
- `./changes.rs` → `WorktreeChangeSummary`, `change_summary`, bounding, porcelain/diffstat parsing and bilingual rendering
- `./rails.rs` → `WorktreeRail`, `WorktreeContext`, `build_worktree_notice`, the section/summary constants and the bounded bilingual rendering
- `./tests/` → `mod.rs` (scripted runner and fixtures), `manager.rs` (lifecycle scenarios, real-git round trip), `lock.rs` (lock layout, contention, timeout, metadata), `tool.rs` (card, dispatch, payload failures), `tools.rs` (enter/exit cards, argument/session/failure paths, removal confirmation, permission refusal), `sparse.rs` (path validation, command construction), `changes.rs` (parsing, call order, bounding, rendering), `rails.rs` (inert path, section/summary payloads, explicit override, failing summary, unlisted worktree, bounds, notice text, hook coverage)
