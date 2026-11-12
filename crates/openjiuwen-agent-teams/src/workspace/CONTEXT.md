# openjiuwen-agent-teams/src/workspace

## Intent
Member workspace allocation: where a teammate's files live, how untrusted names become path segments, and the seam that will hold a git worktree backend.
Python parity sources: `openjiuwen/agent_teams/workspace_layout.py`, `openjiuwen/agent_teams/paths.py` and `openjiuwen/agent_teams/worktree/naming.py` (plus `openjiuwen/harness/tools/worktree/slug.py` for branch flattening).
Everything here is local and in-process: allocation creates directories with `tokio::fs`, release never deletes them (fail-closed, Python parity), and a worktree request without an injected backend fails with `Error::Unsupported`.
The module is pure-path + `async` directory work only: no global state, no environment lookups, no clocks.

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
    pub fn new(base: impl Into<PathBuf>) -> Self;
    pub fn base(&self) -> &Path;
    pub fn team_dir(&self, team: &str) -> Result<PathBuf>;
    pub fn session_dir(&self, team: &str, session_id: &str) -> Result<PathBuf>;
    pub fn member_workspace(&self, team: &str, session_id: &str, member: &str) -> Result<PathBuf>;
    pub fn member_dir(&self, team: &str, session_id: &str, member: &str) -> Result<PathBuf>;
    pub fn worktrees_dir(&self, team: &str, session_id: &str) -> Result<PathBuf>;
}

// worktree.rs — deterministic names + backend seam
pub const TEAM_PART_CHARS: usize = 14;  pub const MEMBER_PART_CHARS: usize = 16;
pub const DIGEST_CHARS: usize = 10;
pub struct WorktreeInfo { pub name: String, pub path: PathBuf, pub branch: String }   // serde snake_case
pub fn build_teammate_worktree_name(team: &str, member: &str, sha: &str) -> String;
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
}
pub struct UnsupportedWorktreeBackend;             // name() == "unsupported"; create/remove -> Unsupported

// local.rs — the shipped provider
pub struct LocalWorkspaceProvider;                 // manual Debug (backend exposes only name())
impl LocalWorkspaceProvider {
    pub fn new(layout: MemberWorkspaceLayout) -> Self;
    pub fn with_worktree_backend(self, backend: Arc<dyn WorktreeBackend>, mode: IsolationMode) -> Self;
    pub fn layout(&self) -> &MemberWorkspaceLayout;
}
#[async_trait] impl WorkspaceProvider for LocalWorkspaceProvider { /* allocate, release */ }
```

## Layout rules
`base` is the agent-teams home (Python `get_agent_teams_home()` = `<openjiuwen_home>/.agent_teams`); every segment below is sanitized.
- `team_dir("my-team")` = `<base>/my-team`
- `session_dir("my-team", "session-1")` = `<base>/my-team/sessions/session-1`
- `member_workspace("my-team", "session-1", "dev-1")` = `<base>/my-team/sessions/session-1/workspaces/dev-1_workspace`
- `member_dir("my-team", "session-1", "dev-1")` = `<base>/my-team/sessions/session-1/members/dev-1`
- `worktrees_dir("my-team", "session-1")` = `<base>/my-team/sessions/session-1/worktrees` (a teammate worktree is that directory joined with the worktree name)
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
- Errors are `openjiuwen_common::error::{Error, Result}` only: `Error::validation` for names and project roots, `Error::sys_op` for directory creation, `Error::unsupported` for the worktree backend.
- No panics on user input; no `unwrap`/`expect` outside `#[cfg(test)]`.
- Locks are never held across `.await`; the provider is stateless apart from the injected backend.
- `tracing` only (`debug!` on success, `warn!` when the shared workspace is disabled).
- `Default`/`Copy` on `IsolationMode`; every public data type is `serde` (snake_case).
- Files stay well under 1000 lines; pure helpers and the async provider are split.
- `--no-default-features` must build (nothing here is feature gated).

## Routing Table
- `./mod.rs` -> `IsolationMode`, `MemberWorkspace`, `WorkspaceRequest`, `WorkspaceProvider` seam and the module doctest
- `./layout.rs` -> `sanitize_segment`, `MemberWorkspaceLayout` (team/session/member/worktree paths)
- `./local.rs` -> `LocalWorkspaceProvider` (shared allocation, project sub-directory, worktree delegation, release)
- `./worktree.rs` -> `WorktreeInfo`, name/branch/digest helpers, `WorktreeCreateRequest`, `WorktreeBackend`, `UnsupportedWorktreeBackend`

## Test Strategy
Unit tests live beside the code (`#[cfg(test)]`, `#[allow(clippy::unwrap_used, clippy::expect_used)]`), all offline and deterministic; filesystem tests use `tempfile::TempDir`.
- `layout.rs`: exact path equality for team/session/workspace/member/worktrees, sanitization accept/reject matrix, Unicode-per-scalar replacement, 96-char truncation, hostile input never escaping `base`.
- `worktree.rs`: the documented name example, case/space/unicode slugging and fallbacks, 14/16 truncation with digest-distinguished long names, branch flattening, digest determinism/shape, `UnsupportedWorktreeBackend` failing closed with the Python capability name.
- `local.rs`: enabled spec creates the workspace plus `members/<member>`; disabled spec creates nothing; `member_spec.root_path` creates `<workspace>/src/x` and rejects `/abs` and `../escape`; worktree isolation without a backend is `Unsupported`; a fake backend records one `create` (name starts `agent-`, path under `<session>/worktrees/`) and surfaces on the `MemberWorkspace`; a failing backend propagates; release retains a shared workspace and calls `remove` for a worktree.
- `mod.rs`: `IsolationMode` wire values, `MemberWorkspace` round-trip (shared and worktree), `WorkspaceRequest` defaults.
Definition of done: `cargo fmt -p openjiuwen-agent-teams`, `cargo test -p openjiuwen-agent-teams`, `cargo test -p openjiuwen-agent-teams --no-default-features`, `cargo clippy -p openjiuwen-agent-teams --all-targets -- -D warnings` all green.

## Deferred (documented, `Error::Unsupported` through the seams)
- The git worktree backend (`WorktreeBackend` / `UnsupportedWorktreeBackend`; Python `agent_teams.worktree.lifecycle`): creation, removal, commit contribution and cleanup are not ported.
- The shared team-workspace manager: `TeamWorkspaceSpec::{root_path, artifact_dirs, version_control, conflict_strategy, remote_url}` are accepted but not acted on; no `team-workspace/` directory, no artifact directories, no `git init`.
- Symlinked/ref-counted member workspace linking (`ensure_team_member_workspace_link`) and independent DeepAgent workspaces.
- Remote/distributed workspaces and worktree backends.

## Notes for Agents
- Member workspaces are session-scoped (`<team>/sessions/<session>/workspaces/<member>_workspace`); Python's stable `<team>/workspaces/<member>_workspace` is deliberately not reproduced, so concurrent sessions never share mutable state.
- Sanitization is fail-closed: Python joins team and member names into the path raw and only sanitizes session ids/workflow names/run ids; this port sanitizes every segment and rejects values it cannot make safe (including `.`/`..` and separators) instead of rewriting them.
- The worktree suffix uses FNV-1a 64 truncated to 10 hex chars instead of Python's `sha256(...)[:10]`, because `sha2` is not a dependency of this crate; it is deterministic across runs and is not a security primitive.
- `release` retains every shared workspace (Python cleans up on team delete or session clean); do not add a delete here.
- `member_spec.root_path` is resolved inside the workspace root: `""`, `"."` and `"./"` mean the root itself, every other value must be relative (`Component::Normal` only) and is sanitized per component; `TeamWorkspaceSpec.root_path` is ignored (deferred).
- `enabled=false` returns a `MemberWorkspace` whose directory does not exist, so callers see Python's "no workspace manager" state rather than a missing record.
- `IsolationMode::Worktree` requires a backend injected with `with_worktree_backend(..., IsolationMode::Worktree)`; a backend recorded under another mode is debug-visible but not used.
