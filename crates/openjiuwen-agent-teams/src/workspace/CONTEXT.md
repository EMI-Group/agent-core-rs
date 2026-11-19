# openjiuwen-agent-teams/src/workspace

## Intent
Member workspace allocation: where a teammate's files live, how untrusted names become path segments, and the backends that give a member its own working tree.
Python parity sources: `openjiuwen/agent_teams/workspace_layout.py`, `openjiuwen/agent_teams/paths.py`, `openjiuwen/agent_teams/worktree/{lifecycle,naming}.py`, `openjiuwen/agent_teams/worktree_remote.py` and `openjiuwen/harness/tools/worktree/{git,backend,lock}.py`.
Everything here is local and in-process: allocation creates directories and git worktrees with `tokio::fs` and the `git` CLI, release removes a worktree through its backend, and a shared workspace is never deleted (fail-closed, Python parity).
The module holds no global state, reads no environment variables and reads no clock except for lock staleness.

## API Surface
```rust
// mod.rs — seam types
pub enum IsolationMode { None, Worktree }        // Default = None; as_str() -> "none"|"worktree"
pub struct MemberWorkspace {                      // serde snake_case
    pub team_name: String, pub session_id: String, pub member_name: String,
    pub root: PathBuf, pub isolation: IsolationMode, pub worktree: Option<WorktreeInfo>,
}
impl MemberWorkspace {
    pub fn new(team_name, session_id, member_name, root: impl Into<PathBuf>) -> Self;
    pub fn root_str(&self) -> String;             // lossy UTF-8
    pub fn is_shared(&self) -> bool;              // isolation == IsolationMode::None
}
pub struct WorkspaceRequest {                     // serde snake_case
    pub team_name: String, pub session_id: String, pub member_name: String,
    pub spec: TeamWorkspaceSpec, pub member_spec: Option<WorkspaceSpec>, pub isolation: IsolationMode,
}
impl WorkspaceRequest { pub fn new(team, session, member) -> Self; }  // default spec, no isolation
#[async_trait]
pub trait WorkspaceProvider: Send + Sync {
    async fn allocate(&self, request: &WorkspaceRequest) -> Result<MemberWorkspace>;
    async fn release(&self, workspace: &MemberWorkspace) -> Result<()>;
}

// layout.rs — sanitization + the session-scoped tree
pub const MAX_SEGMENT_CHARS: usize = 96;
pub fn sanitize_segment(value: &str) -> Result<String>;
pub struct MemberWorkspaceLayout { pub base: PathBuf }
impl MemberWorkspaceLayout {
    pub fn new(base) -> Self;  pub fn base(&self) -> &Path;
    pub fn team_dir(&self, team) -> Result<PathBuf>;
    pub fn session_dir(&self, team, session_id) -> Result<PathBuf>;
    pub fn member_workspace(&self, team, session_id, member) -> Result<PathBuf>;
    pub fn member_dir(&self, team, session_id, member) -> Result<PathBuf>;
    pub fn worktrees_dir(&self, team, session_id) -> Result<PathBuf>;
}

// local.rs — the provider
pub struct LocalWorkspaceProvider;                 // manual Debug (backends expose only name())
impl LocalWorkspaceProvider {
    pub fn new(layout: MemberWorkspaceLayout) -> Self;
    pub fn with_worktree_backend(self, backend: Arc<dyn WorktreeBackend>, mode: IsolationMode) -> Self;
    pub fn layout(&self) -> &MemberWorkspaceLayout;
    pub fn worktree_backend(&self) -> Arc<dyn WorktreeBackend>;   // injected when it serves worktrees, else git
}

// worktree.rs — naming + the backend seam
pub const TEAM_PART_CHARS: usize = 14;  pub const MEMBER_PART_CHARS: usize = 16;  pub const DIGEST_CHARS: usize = 10;
pub const NAME_PREFIX: &str = "agent-";  pub const BRANCH_PREFIX: &str = "worktree-";
pub struct WorktreeInfo { pub name: String, pub path: PathBuf, pub branch: String }
pub fn build_teammate_worktree_name(team, member, sha) -> String;
pub fn worktree_branch(name: &str) -> String;      // "worktree-<name>", "/" -> "+"
pub fn worktree_digest(seed: &str) -> String;      // 10 hex chars, FNV-1a 64
pub struct WorktreeCreateRequest {                 // serde snake_case
    pub team_name: String, pub session_id: String, pub member_name: String,
    pub name: String, pub path: PathBuf, pub base_ref: Option<String>,
}
#[async_trait]
pub trait WorktreeBackend: Send + Sync {
    fn name(&self) -> &str;
    async fn create(&self, request: &WorktreeCreateRequest) -> Result<WorktreeInfo>;
    async fn remove(&self, worktree: &WorktreeInfo) -> Result<()>;
    async fn list(&self, repo_root: &Path) -> Result<Vec<WorktreeInfo>>;            // default: Unsupported
    async fn lock(&self, worktree: &WorktreeInfo, reason: &str) -> Result<()>;      // default: Unsupported
    async fn unlock(&self, worktree: &WorktreeInfo) -> Result<()>;                  // default: Unsupported
}
pub struct UnsupportedWorktreeBackend;             // name() == "unsupported"; every op -> Unsupported

// git.rs — the git CLI layer
pub const GIT_TIMEOUT: Duration = 10s;  pub const GIT_BINARY: &str = "git";  pub const DEFAULT_BASE_REF: &str = "HEAD";
pub struct GitOutput { pub code: Option<i32>, pub stdout: String, pub stderr: String }
impl GitOutput { pub fn succeeded(&self) -> bool; pub fn stderr_summary(&self) -> String; pub fn first_line(&self) -> Option<&str>; }
pub fn resolve_git() -> Result<PathBuf>;           // which::which, Config when missing
pub async fn run_git(cwd: &Path, args: &[&str]) -> Result<GitOutput>;
pub fn normalize_path(path: &Path) -> PathBuf;     // lexical: drops ".", pops ".."
pub fn existing_ancestor(path: &Path) -> PathBuf;  // path or its nearest existing ancestor
pub struct GitRepository { pub root: PathBuf, pub git_dir: PathBuf, pub common_dir: PathBuf }
impl GitRepository {
    pub async fn discover(start: &Path) -> Result<Self>;
    pub fn lock_file(&self) -> PathBuf;
    pub async fn resolve_ref(&self, refspec: &str) -> Result<Option<String>>;
    pub async fn worktree_entries(&self) -> Result<Vec<WorktreeEntry>>;
    pub async fn worktree_entry(&self, path: &Path) -> Result<Option<WorktreeEntry>>;
    pub async fn worktree_add(&self, path: &Path, branch: &str, base_ref: &str) -> Result<()>;
    pub async fn worktree_remove(&self, path: &Path) -> Result<()>;
    pub async fn worktree_prune(&self) -> Result<()>;
    pub async fn lock_worktree(&self, path: &Path, reason: &str) -> Result<()>;
    pub async fn unlock_worktree(&self, path: &Path) -> Result<()>;
    pub async fn delete_branch(&self, branch: &str) -> Result<BranchDeleteOutcome>;
}
pub enum BranchDeleteOutcome { Deleted, Missing, InUse(String) }
pub struct WorktreeEntry { pub path: PathBuf, pub head: Option<String>, pub branch: Option<String>,
                           pub locked: Option<String>, pub bare: bool, pub detached: bool }
pub fn parse_worktree_list(porcelain: &str) -> Vec<WorktreeEntry>;

// worktree_git.rs — the shipped backend
pub struct GitWorktreeBackend;                     // Default = discover per request
impl GitWorktreeBackend {
    pub fn new() -> Self;
    pub fn with_repo_root(self, repo_root: impl Into<PathBuf>) -> Self;   // pins the repository
    pub fn with_lock_options(self, options: LockOptions) -> Self;
    pub fn repo_root(&self) -> Option<&Path>;
    pub fn lock_options(&self) -> &LockOptions;
    pub async fn list_worktrees(&self, repository: &GitRepository) -> Result<Vec<WorktreeInfo>>;
}
pub fn teammate_worktrees(entries: Vec<WorktreeEntry>) -> Vec<WorktreeInfo>;  // agent-* dir + worktree-* branch

// lock.rs — the repository lock
pub const LOCK_FILE_NAME: &str = "openjiuwen-worktree.lock";
pub const DEFAULT_LOCK_WAIT: Duration = 10s;  pub const DEFAULT_STALE_AFTER: Duration = 60s;  pub const DEFAULT_POLL_INTERVAL: Duration = 25ms;
pub struct LockOptions { pub wait: Duration, pub stale_after: Duration, pub poll: Duration }   // Default = the three consts
impl LockOptions { pub fn new(wait: Duration) -> Self; pub fn with_stale_after(self, d) -> Self; pub fn with_poll(self, d) -> Self; }
pub struct WorktreeLockGuard;                      // removes its own lock file on drop
impl WorktreeLockGuard {
    pub fn path(&self) -> &Path;
    pub fn token(&self) -> &str;                   // "<pid>-<32 lowercase hex>", stamped into the file
    pub async fn release(self) -> Result<()>;
}
pub async fn acquire(path: &Path, options: &LockOptions) -> Result<WorktreeLockGuard>;

// worktree_remote.rs — the remote protocol
pub enum RemoteWorktreeAction { Create, Remove, Exists }            // "create" | "remove" | "exists"
pub struct RemoteWorktreeRequest { action, slug, repo_url, base_branch, worktree_path, config }   // Option fields
impl RemoteWorktreeRequest {
    pub fn create(slug: impl Into<String>, base_branch: Option<String>) -> Self;
    pub fn remove(path: impl Into<String>) -> Self;
    pub fn exists(path: impl Into<String>) -> Self;
    pub fn validate(&self) -> Result<()>;
}
pub struct RemoteWorktreeResponse { success /*default true*/, worktree_path, worktree_branch, head_commit, existed, exists, error }
impl RemoteWorktreeResponse { pub fn failure(error) -> Self; pub fn into_success(self, node_id, action) -> Result<Self>; }
#[async_trait] pub trait RemoteDispatch: Send + Sync {
    fn name(&self) -> &str;
    async fn send_and_wait(&self, node_id: &str, request: Value) -> Result<Value>;
}
pub struct UnavailableRemoteDispatch;              // every send -> Unsupported
pub struct RemoteWorkspaceAdapter;                 // manual Debug; impl WorktreeBackend
impl RemoteWorkspaceAdapter {
    pub fn new(node_id, dispatch: Arc<dyn RemoteDispatch>) -> Self;
    pub fn unavailable(node_id) -> Self;
    pub fn with_repo_url(self, repo_url) -> Self;  pub fn with_base_branch(self, branch) -> Self;
    pub fn node_id(&self) -> &str;
    pub async fn exists(&self, worktree_path: &Path) -> Result<bool>;
}
```

## Backend matrix
| Backend | `name()` | Used by | Implemented operations | Fails with |
| --- | --- | --- | --- | --- |
| `GitWorktreeBackend` | `git` | `LocalWorkspaceProvider::worktree_backend()` by default | `create`, `remove`, `list`, `lock`, `unlock` against a real repository | `Config` (no repository / no `git`), `Validation` (bad request, name collision), `NotFound` (unknown worktree), `SysOp` (git failed), `Timeout` (git too slow, lock contended) |
| `UnsupportedWorktreeBackend` | `unsupported` | injected by a host that wants no worktrees | none | `Unsupported` naming `agent_teams.worktree.lifecycle` |
| `RemoteWorkspaceAdapter` | `remote` | a leader whose members live on another node | `create`, `remove`, inherent `exists` over an injected `RemoteDispatch` | `SysOp` (node reported failure, malformed reply), `Unsupported` (transport unavailable), plus the transport's own error |

`LocalWorkspaceProvider::worktree_backend()` returns the injected backend when it was registered for `IsolationMode::Worktree`, else a fresh `GitWorktreeBackend::new()`; a backend registered under another mode is diagnostic only.

## git command contract
Discovery (`GitRepository::discover`, from the nearest existing ancestor of the start path):
- `git rev-parse --show-toplevel --absolute-git-dir --git-common-dir` — one invocation, three lines.
- `--git-common-dir` is relative to the directory the command ran in, so it is joined and normalized; `root` is the parent of the common dir when it is named `.git`, else the toplevel, so discovery from inside a teammate worktree resolves the **main** repository (Python `find_canonical_git_root`).
- Commands always run in `root`, never inside the worktree they act on.

Operations (all `10 s` bounded, stderr captured, `GIT_TERMINAL_PROMPT=0`, empty `GIT_ASKPASS`, `GIT_DIR`/`GIT_WORK_TREE`/`GIT_INDEX_FILE` removed, stdin `/dev/null`):
- create: `create_dir_all(parent)` → discover → lock → `git worktree add -b <worktree-…> <path> <base_ref|HEAD>`; the worktree is never reused (a registered worktree or a non-empty target directory is `Error::Validation`), and a git name collision (`already exists`) is also `Validation`.
- remove: discover → lock → `git worktree remove --force <path>` → `git worktree prune` → `git branch -D <branch>`; prune runs even when the removal failed.
- list: `git worktree list --porcelain`, filtered to entries whose directory stem starts with `agent-` **and** whose branch starts with `worktree-`, sorted by path.
- lock/unlock: `git worktree lock --reason <reason> <path>` / `git worktree unlock <path>`; "not a working tree"/"not locked" becomes `Error::NotFound`.
- branch deletion: `git branch -D <branch>` → `Deleted`, `Missing` ("not found", tolerated) or `InUse` ("used by worktree"/"checked out at"); a successfully removed worktree whose branch is still checked out elsewhere reports `Error::Validation` naming the branch, because the state is only half-cleaned and the caller must know.

## Locking
`<common-dir>/openjiuwen-worktree.lock` (Python `<repo_root>/.git/openjiuwen-worktree.lock`), held around every critical section of one repository (`create`, `remove`, `lock`, `unlock`) so two members cannot race on shared `.git` state.
The file lives inside `.git`, where a leading dot would be meaningless, so the name has none; Python only uses `.openjiuwen-worktree.lock` beside a repository that has no `.git` directory, a case this port does not reach.
- Exclusivity is `create_new` file creation, which the filesystem arbitrates atomically across tasks and processes; the file records `pid=<n>` and `token=<pid>-<32 lowercase hex>`, a token unique to one acquisition.
- A loser polls every `poll` (25 ms) until `wait` (10 s) expires, then fails with `Error::Timeout` — never lock-free.
- A file older than `stale_after` (60 s) is treated as abandoned: it is removed with `warn!` and the acquisition retries; an unreadable modification time is left alone (fail closed).
  The decision is re-checked over the content: the file must still be stale after it was read and its bytes must be identical across two reads, so a holder that refreshed its stamp (or another waiter that replaced the file) is never stolen.
- A guard keeps its file's modification time fresh — a heartbeat task every `stale_after / 3` (minimum 10 ms; a window shorter than 30 ms gets no heartbeat) re-reads the file first and stops as soon as it no longer names its token — so a critical section that provably outlives `stale_after` is never declared abandoned.
- Ownership is checked on every release path: `WorktreeLockGuard` removes the file on drop and on `release()` **only while the content still carries its own token**, because a stale guard that was reclaimed must not delete the new holder's file (that would admit a third writer next to the second). A file that already vanished is a successful release, the takeover is logged with `warn!`, and `release()` reports removal failures that are not `NotFound`.

## Remote adapter status
Python's remote contract is JSON over the team `Messager` (`send_and_wait`), **not** HTTP: the leader sends a `WorktreeRemoteRequest` and the node answers with a `WorktreeRemoteResponse` (`openjiuwen/agent_teams/worktree_remote.py`).
`reqwest` would therefore not talk to an openJiuwen node, and this crate does not depend on it.
The port splits the contract:
- implemented and tested: the wire types (`create`/`remove`/`exists`), request defaults and validation, translation into `WorktreeInfo`/`Error`, and the adapter over an injected `RemoteDispatch` seam (the Rust counterpart of `Messager.send_and_wait`);
- deferred: a live distributed transport. `UnavailableRemoteDispatch` (and `RemoteWorkspaceAdapter::unavailable`) fails with `Error::Unsupported` naming `agent_teams.worktree_remote.RemoteWorktreeBackend`;
- `list`/`lock`/`unlock` use the trait defaults (`Unsupported`); Python's remote backend has no such operations either.

## Layout rules
`base` is the agent-teams home (Python `get_agent_teams_home()` = `<openjiuwen_home>/.agent_teams`); every segment below is sanitized.
- `team_dir("my-team")` = `<base>/my-team`
- `session_dir("my-team", "session-1")` = `<base>/my-team/sessions/session-1`
- `member_workspace(...)` = `<base>/my-team/sessions/session-1/workspaces/dev-1_workspace`
- `member_dir(...)` = `<base>/my-team/sessions/session-1/members/dev-1`
- `worktrees_dir(...)` = `<base>/my-team/sessions/session-1/worktrees`, and a teammate worktree is that directory joined with the worktree name
`members/` sits outside `workspaces/` so resetting or re-creating a member workspace never touches the member's runtime state.
Python's stable `<team_home>/workspaces/<member>_workspace` has no session component; this port is session-scoped by design (see Notes for Agents).

## Sanitization rules (`sanitize_segment`)
1. Trim surrounding whitespace.
2. Reject with `Error::Validation` (message carries the offending value): empty/whitespace-only, anything containing `/`, `\` or NUL, absolute paths, and exactly `.` or `..`.
3. Replace every character outside `[A-Za-z0-9._-]` with `_`, per Unicode scalar value (one `_` per multi-byte character).
4. Strip leading/trailing `.`, `_`, `-`; truncate to `MAX_SEGMENT_CHARS` (96) characters; strip again.
5. Reject when the result is empty, `.` or `..`.
Consequences: `"dev-1"` is unchanged, `"café-日本"` becomes `"caf_-"`, `"Team A/2024"` and `"../evil"` are rejected, and a sanitized value joined to the base always stays below the base.
Team, session and member names are all sanitized (Python joins team/member names raw); `member_workspace` appends `_workspace` after sanitization.

## Worktree naming rules
`build_teammate_worktree_name("my-team", "dev-1", "abcdef1234567890")` = `agent-my-team-dev-1-abcdef1234`.
- team part: lowercased, non-`[a-z0-9]` -> `-`, runs collapsed, `-` trimmed, truncated to 14 chars, fallback `team`;
- member part: same rules, truncated to 16 chars, fallback `member`;
- digest part: first 10 lowercase alphanumerics of the supplied digest, fallback `0000000000`.
The digest is computed by the caller over the raw `team:member:session` triple (`worktree_digest`), so names that truncate to the same slug still produce different worktree names.
Branch: `worktree_branch(name)` = `worktree-<name>` with `/` flattened to `+`.

## Constraints
- Only files under `src/workspace/` are edited; `lib.rs`, `Cargo.toml`, `src/schema/` and sibling modules are out of scope.
- Errors are `openjiuwen_common::error::{Error, Result}` only: `Validation` for names, paths and name collisions, `Config` for a missing repository or `git` binary, `NotFound` for unknown worktrees, `SysOp` for directory and git failures, `Timeout` for git and lock deadlines, `Unsupported` for a refused capability.
- No panics on user input; no `unwrap`/`expect` outside `#[cfg(test)]`; no `unsafe`.
- Locks are never held across `.await` except the file lock, which is held by value across the critical section and never by a shared reference; the provider is stateless apart from the injected backend.
- `tracing` only (`debug!` on success, `warn!` for a disabled shared workspace and a reclaimed stale lock).
- `Default`/`Copy` on `IsolationMode`; every public data type is `serde` (snake_case, unknown fields tolerated).
- Files stay well under 1000 lines; pure helpers, the CLI layer, the backend, the lock and the remote protocol are split.
- No new dependency: the backends use `tokio::process`, `tokio::time`, `which`, `serde`, `serde_json`, `tempfile` (test) and `tracing`, all already declared by the crate.
- `--no-default-features` must build (nothing here is feature gated).

## Routing Table
- `./mod.rs` -> `IsolationMode`, `MemberWorkspace`, `WorkspaceRequest`, `WorkspaceProvider` seam, module wiring and the module doctest
- `./layout.rs` -> `sanitize_segment`, `MemberWorkspaceLayout` (team/session/member/worktree paths)
- `./local.rs` -> `LocalWorkspaceProvider` (shared allocation, project sub-directory, worktree delegation through `worktree_backend()`, release)
- `./worktree.rs` -> `WorktreeInfo`, name/branch/digest helpers, `WorktreeCreateRequest`, the `WorktreeBackend` seam, `UnsupportedWorktreeBackend`
- `./git.rs` -> `run_git`/`GitOutput`, `resolve_git`, `normalize_path`, `existing_ancestor`, `GitRepository` (discovery, worktree verbs, branch deletion), `WorktreeEntry` + `parse_worktree_list`
- `./worktree_git.rs` -> `GitWorktreeBackend`, `teammate_worktrees` (the owned-worktree filter)
- `./lock.rs` -> `LockOptions`, `WorktreeLockGuard` (owner token + heartbeat), `acquire`, stale reclamation
- `./worktree_remote.rs` -> `RemoteWorktreeAction`/`Request`/`Response`, `RemoteDispatch`, `UnavailableRemoteDispatch`, `RemoteWorkspaceAdapter`
- `./git_tests.rs` -> porcelain parsing, path helpers, discovery, branch deletion, the command runner
- `./lock_tests.rs` -> exclusivity, bounded wait, stale reclamation, directory creation, ownership on release after a steal, the heartbeat keeping a live holder
- `./worktree_git_tests.rs` -> real repositories: round trip, no reuse, concurrent create, base ref, pinned root, list filter, provider allocation
- `./worktree_remote_tests.rs` -> wire shapes, validation, dispatch seam, remote failure, unavailable transport
- `./test_repo.rs` -> shared fixture: a temporary repository with one commit and helpers for commits/branches/heads

## Test Strategy
Unit tests live beside the code (`#[cfg(test)]` inline) and in the sibling `*_tests.rs` suites, all offline and deterministic; filesystem and repository tests use `tempfile::TempDir` and never touch the network.
The git-backed suites spawn the real `git` binary (resolved through `which`) and are not `#[cfg(unix)]`- or feature-gated, so a host without `git` fails `test_repo::git_ok`'s "git is available" assertion instead of skipping.
- `layout.rs`: exact path equality for team/session/workspace/member/worktrees, sanitization accept/reject matrix, Unicode-per-scalar replacement, 96-char truncation, hostile input never escaping `base`.
- `worktree.rs`: the documented name example, case/space/unicode slugging and fallbacks, 14/16 truncation with digest-distinguished long names, branch flattening, digest determinism/shape, `UnsupportedWorktreeBackend` failing closed with the Python capability name.
- `local.rs`: enabled spec creates the workspace plus `members/<member>`; disabled spec creates nothing; `member_spec.root_path` creates `<workspace>/src/x` and rejects `/abs` and `../escape`; a worktree outside a repository is `Config`; an injected `UnsupportedWorktreeBackend` keeps worktrees off; a fake backend records one `create` (name starts `agent-`, path under `<session>/worktrees/`) and surfaces on the `MemberWorkspace`; a failing backend propagates; release retains a shared workspace and calls `remove` for a worktree.
- `git_tests.rs` (11): full and partial porcelain parsing (locked reason, detached, bare, unknown lines), lexical normalization, ancestor walk, a failed command captured rather than raised, discovery from a missing sub-directory and from inside a worktree, missing repository as `Config`, deleted/missing/in-use branch classification, unsafe branch names, `resolve_ref` both answers.
- `lock_tests.rs` (5): a held lock is not granted twice and releasing or dropping it re-opens it, a waiter acquires after the holder releases, a stale file is reclaimed, a fresh file is respected, `acquire` creates the lock directory.
- `worktree_git_tests.rs` (12): create → list → lock → unlock → remove round trip (directory, branch, list, locked removal refused), no reuse of an existing worktree, two concurrent creates with exactly one winner and one worktree registered, `Config` outside a repository, request validation (empty name, relative path), `base_ref` honored, `NotFound` for an unknown worktree, a pinned root beating discovery, the owned-worktree filter (unit + against a hand-made worktree), provider allocation of two distinct real worktrees plus release, and the lock file being gone after a critical section.
- `worktree_remote_tests.rs` (12): Python wire shape and unknown-field tolerance, response defaults, per-action validation, remote failure → `SysOp`, create/remove/exists payloads, request `base_ref` winning over the adapter default, branch fallback, missing reported path, remote failure surfaced on remove, the unavailable transport naming the Python capability, and the default trait operations failing closed.
Definition of done: `cargo fmt -p openjiuwen-agent-teams`, `cargo test -p openjiuwen-agent-teams`, `cargo test -p openjiuwen-agent-teams --no-default-features`, `cargo clippy -p openjiuwen-agent-teams --all-targets -- -D warnings` all green.

## Deferred (documented, `Error::Unsupported` through the seams)
- Live distributed worktree transport (Python `Messager`): only the `RemoteDispatch` seam and the protocol types ship; the `ZeroMQ`/`Pulsar` transports are deferred crate-wide.
- The shared team-workspace manager: `TeamWorkspaceSpec::{root_path, artifact_dirs, version_control, conflict_strategy, remote_url}` are accepted but not acted on; no `team-workspace/` directory, no artifact directories, no `git init`.
- Symlinked/ref-counted member workspace linking (`ensure_team_member_workspace_link`) and independent DeepAgent workspaces.
- Worktree lifecycle automation of Python's `TeammateWorktreeLifecycle`: contribution classification (`rev-list`/`merge-base`), change counting and finalize-on-clean policies. The primitives (`resolve_ref`, `worktree_entries`, `branch_delete`, porcelain parsing) exist, the policy layer does not.
- Sparse checkout, `core.longpaths`, hook-based worktrees and remote-repository cloning on the node (`WorktreeRemoteHandler._ensure_repo`).

## Notes for Agents
- Member workspaces are session-scoped (`<team>/sessions/<session>/workspaces/<member>_workspace`); Python's stable `<team>/workspaces/<member>_workspace` is deliberately not reproduced, so concurrent sessions never share mutable state.
- Sanitization is fail-closed: Python joins team and member names into the path raw and only sanitizes session ids/workflow names/run ids; this port sanitizes every segment and rejects values it cannot make safe (including `.`/`..` and separators) instead of rewriting them.
- The worktree suffix uses FNV-1a 64 truncated to 10 hex chars instead of Python's `sha256(...)[:10]`, because `sha2` is not a dependency of this crate; it is deterministic across runs and is not a security primitive.
- `release` retains every shared workspace (Python cleans up on team delete or session clean) and removes a worktree through the same backend `allocate` used; do not add a delete for shared workspaces.
- `member_spec.root_path` is resolved inside the workspace root: `""`, `"."` and `"./"` mean the root itself, every other value must be relative (`Component::Normal` only) and is sanitized per component; `TeamWorkspaceSpec.root_path` is ignored (deferred).
- `enabled=false` returns a `MemberWorkspace` whose directory does not exist, so callers see Python's "no workspace manager" state rather than a missing record.
- `IsolationMode::Worktree` no longer needs an injected backend: the provider defaults to `GitWorktreeBackend`, which discovers the repository from the worktree target's nearest existing ancestor. Inject a backend to override it, and inject `UnsupportedWorktreeBackend` to refuse worktrees outright.
- A pinned `GitWorktreeBackend::with_repo_root` wins over discovery, so a caller that already resolved its project repository gets worktrees in that repository even when the session tree lives elsewhere.
- `git` is resolved through `which` on every invocation (no cached path, no global state); a host without `git` gets `Error::Config` naming the binary, never a silent fallback to a shared directory.
- Non-UTF-8 paths are rejected with `Error::Validation` before a subprocess starts, because `git` arguments are passed as `&str`; worktree names this crate generates are always ASCII.
- The lock is per repository, not per worktree: it serializes the whole worktree critical section (git mutates shared `.git/worktrees` state), so a create waits for an unrelated member's removal in the same repository. It is held only across git commands, never across the member's lifetime.
- Two concurrent creates of the same worktree serialize on the lock; the loser re-checks registration and fails with `Error::Validation` ("refusing to reuse it"), so exactly one caller owns a worktree.
- Removing a worktree is deliberately not idempotent: an unregistered path is `Error::NotFound` after a prune. A release path that must tolerate a double release should check `list` first.
- The remote adapter ignores `WorktreeCreateRequest::path` (the node owns its layout) and defaults its branch to the local `worktree-<name>` shape when the node omits `worktree_branch`.
- `RemoteWorktreeRequest::config` stays `null`: this port ships no `WorktreeConfig` equivalent, and forwarding a foreign schema would mislead a node.
- The session-scoped worktrees directory is created inside the repository working tree (`.agent_teams/...`), which git accepts; teams that must keep the repository clean should point `MemberWorkspaceLayout` outside it and pin `GitWorktreeBackend::with_repo_root` at the project repository.
