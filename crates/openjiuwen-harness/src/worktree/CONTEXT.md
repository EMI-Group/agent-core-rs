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
## Layout
A worktree named `Feature Login!` becomes `slug = "Feature-Login"`, directory `<repo-root>/.worktrees/Feature-Login`, branch `worktree-Feature-Login`.
The registry is `<workspace-root>/.worktree/index.json` (default workspace root is the repo root), the directory `crate::workspace::Workspace` already uses for worktree links.
`create` recovers an existing worktree at the target path (`existed = true`) and refuses a target directory that git does not list.
`create` checks out an existing branch instead of re-creating it; `base` defaults to `HEAD`.
`remove` refuses a dirty worktree without `force` and reports an unlisted worktree as `Error::not_found`.
`prune` runs `git worktree prune` and then drops registry records whose directory git no longer lists.
## Constraints
Offline and deterministic by default: tests use `ScriptedGitRunner` (queued stdout/failures, recorded calls) and temp directories; the single real-git test skips when `which::which("git")` fails.
No new dependencies, no `unsafe`, no `println!` (tracing only); errors are `openjiuwen_common::error::{Error, Result}`.
`slug::validate_raw_name` runs before `slugify`, so `..`, absolute paths, `~` and drive prefixes are refused instead of normalised.
`target_path` normalises lexically and asserts the result stays under the repo root.
The registry document is written atomically (sibling `.tmp` + rename) and a missing file loads as empty while malformed JSON is `ErrorKind::Validation`.
The tool never returns `Err` for expected failures: it returns a `ToolPayload` with `success = false`; only cancellation propagates.
The tool consults `crate::permission` path guards for the target path on mutating actions and blocks only on `PermissionLevel::Deny` (`ask`/`allow` pass, no engine fails open).
The tool is mounted by `crate::tools::ToolRegistry` as `BuiltinTool::Worktree`, the last of the extended tools, built from a `WorktreeManager` plus the registry language and permission engine.
No repo-wide cross-process file lock and no operation serialisation: concurrent managers over one workspace are not coordinated.
## Deferred
Python's `lock.py` cross-process `filelock` + per-repo in-process lock and timeout.
Sparse checkout, `symlink_directories`, `include_patterns` copying, `core.longpaths`, hook-path configuration.
Lifecycle policies (`ephemeral`/`durable`/`auto`), `cleanup_after_days`, auto-cleanup on shutdown, event-log recovery of sessions.
`enter_worktree`/`exit_worktree` tools, the worktree rails, backend hooks (`WorktreeBackend`), change summaries (commits since creation, unpushed commits), and the `vcs` module.
## Routing Table
- `./git.rs` → `GitRunner` seam, `ProcessGitRunner`, porcelain parsers, typed git errors, command helpers
- `./slug.rs` → raw-name validation, `slugify`, `unique_slug`, branch naming, layout constants
- `./models.rs` → `WorktreeInfo`, `WorktreeCreateResult`, `WorktreeStatus`, `WorktreeRecord`, `WorktreeState`
- `./store.rs` → `WorktreeIndex` registry and atomic JSON writes
- `./events.rs` → `WorktreeEvent`, `WorktreeEventSink`, noop/broadcast/recording sinks
- `./manager.rs` → `WorktreeManager` lifecycle, layout and refusal rules
- `./tool.rs` → `WorktreeTool` and `WorktreeArgs` (the model-facing `worktree` tool)
- `./tests/` → `mod.rs` (scripted runner and fixtures), `manager.rs` (lifecycle scenarios, real-git round trip), `tool.rs` (card, dispatch, payload failures)
