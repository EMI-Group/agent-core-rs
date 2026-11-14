//! The model-facing `enter_worktree` and `exit_worktree` tools and the worktree
//! session store.
//!
//! Port of `openjiuwen/harness/tools/worktree/{tools,session}.py`: Python keeps
//! the active session in a `ContextVar` and answers `enter_worktree`/`exit_worktree`
//! with a lifecycle message. This port keeps the observable behaviour — argument
//! validation, the two-phase removal confirmation, the exact messages, the
//! session-scoped default name — and stores the session behind the
//! [`WorktreeSessionStore`] seam, so a host can keep it per agent or per process.
//!
//! # Cwd deviation
//!
//! Python also calls `set_cwd(worktree_path)`/`set_original_cwd(...)` to switch
//! the process working directory. Rust cannot change a process-wide cwd safely
//! (there is no per-task `ContextVar` equivalent), so the tools never touch the
//! cwd: they record the session and report `worktree_path`/`original_cwd` in the
//! payload, and the host applies the change in whatever cwd abstraction it owns.
//! The messages still say "CWD switched to worktree." for Python parity.
//!
//! # Failure policy
//!
//! Every expected failure — unknown arguments, a blank name without a stored
//! default, a refused name, an occupied session, an unverifiable or dirty
//! worktree — is a [`ToolPayload`] with `success = false`. Only cancellation
//! propagates as `Err`.
#![allow(clippy::result_large_err)]
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::Mutex;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use openjiuwen_common::error::Result;
use openjiuwen_foundation::tool::{Tool, ToolCard, ToolContext};

use crate::permission::paths::normalize_lexically;
use crate::permission::{PermissionEngine, PermissionLevel};
use crate::tools::{schema_for, tool_card, ToolPayload};

use super::events::WorktreeEvent;
use super::manager::{WorktreeCreateOptions, WorktreeManager};
use super::models::WorktreeState;
use super::store::WorktreeIndex;

/// Name of the tool that enters a worktree session.
pub const ENTER_TOOL_NAME: &str = "enter_worktree";

/// Name of the tool that exits a worktree session.
pub const EXIT_TOOL_NAME: &str = "exit_worktree";

/// Actions `exit_worktree` accepts, in card order.
pub const EXIT_ACTIONS: [&str; 2] = ["keep", "remove"];

/// Session key Python stores the active worktree session under.
pub const WORKTREE_SESSION_KEY: &str = "_worktree_session";

/// Session key Python stores the session-scoped default worktree name under.
pub const WORKTREE_DEFAULT_NAME_KEY: &str = "_worktree_default_name";

/// Lifecycle policy recorded when the caller supplies none.
const DEFAULT_LIFECYCLE_POLICY: &str = "auto";

/// The active worktree session, mirroring Python's `WorktreeSession`.
///
/// Every field is optional on the wire so a host can persist the struct and
/// read it back with a later version. `lifecycle_policy` defaults to `"auto"`
/// and `original_cwd`/`worktree_path` are empty paths when unset.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct WorktreeSessionState {
    /// Directory the agent worked in before entering the worktree.
    pub original_cwd: PathBuf,
    /// Absolute path of the worktree the agent is working in.
    pub worktree_path: PathBuf,
    /// Name the worktree was entered under.
    pub worktree_name: String,
    /// Branch checked out in the worktree.
    pub worktree_branch: Option<String>,
    /// Branch that was checked out before entering.
    pub original_branch: Option<String>,
    /// HEAD commit at creation time, used as the change-detection baseline.
    pub original_head_commit: Option<String>,
    /// Owner identifier (team member name, or any caller-defined owner id).
    pub member_name: Option<String>,
    /// Owner grouping label (team name, or any caller-defined label).
    pub team_name: Option<String>,
    /// Whether the worktree came from a backend hook rather than git.
    pub hook_based: bool,
    /// Resolved lifecycle policy of the worktree (`"auto"` by default).
    pub lifecycle_policy: String,
    /// Owner lifecycle marker at creation time (`temporary`/`persistent`).
    pub team_lifecycle: Option<String>,
    /// Time spent creating the worktree, for performance tracking.
    pub creation_duration_ms: Option<f64>,
    /// Whether a sparse-checkout selection was applied.
    pub used_sparse_paths: bool,
}

impl Default for WorktreeSessionState {
    fn default() -> Self {
        Self {
            original_cwd: PathBuf::new(),
            worktree_path: PathBuf::new(),
            worktree_name: String::new(),
            worktree_branch: None,
            original_branch: None,
            original_head_commit: None,
            member_name: None,
            team_name: None,
            hook_based: false,
            lifecycle_policy: DEFAULT_LIFECYCLE_POLICY.to_owned(),
            team_lifecycle: None,
            creation_duration_ms: None,
            used_sparse_paths: false,
        }
    }
}

impl WorktreeSessionState {
    /// Builds a session for the worktree `name` at `path` on `branch`.
    ///
    /// `original_cwd` stays empty until the caller sets it, like the optional
    /// Python fields it maps to.
    #[must_use]
    pub fn new(name: impl Into<String>, path: impl Into<PathBuf>, branch: Option<String>) -> Self {
        Self {
            worktree_name: name.into(),
            worktree_path: path.into(),
            worktree_branch: branch,
            ..Self::default()
        }
    }

    /// Serializes the session to JSON.
    ///
    /// # Errors
    ///
    /// Returns the serializer error (never expected for this plain data type).
    pub fn to_json(&self) -> serde_json::Result<Value> {
        serde_json::to_value(self)
    }

    /// Deserializes a session from JSON, tolerating missing fields.
    ///
    /// # Errors
    ///
    /// Returns the deserializer error when `value` is not a session object.
    pub fn from_json(value: &Value) -> serde_json::Result<Self> {
        serde_json::from_value(value.clone())
    }
}

/// The session seam shared by [`EnterWorktreeTool`] and [`ExitWorktreeTool`].
///
/// Python keeps both the active session and the session-scoped default name in
/// one `ContextVar`; a host that needs a different scope (per agent, per
/// process, persisted) implements this trait and wires it with
/// [`EnterWorktreeTool::with_sessions`]. Implementations must be cheap and
/// non-blocking: the tools call these methods inline.
pub trait WorktreeSessionStore: Send + Sync {
    /// The active worktree session, when the agent is inside a worktree.
    fn session(&self) -> Option<WorktreeSessionState>;
    /// Sets or clears the active worktree session.
    fn set_session(&self, session: Option<WorktreeSessionState>);
    /// The session-scoped default worktree name, when one was recorded.
    fn default_name(&self) -> Option<String>;
    /// Sets or clears the session-scoped default worktree name.
    fn set_default_name(&self, name: Option<String>);
}

/// The two slots of a [`WorktreeSessionStore`], behind one lock.
#[derive(Debug, Default)]
struct StoredSessions {
    /// The active session.
    session: Option<WorktreeSessionState>,
    /// The recorded default name.
    default_name: Option<String>,
}

/// An in-process [`WorktreeSessionStore`] guarded by a `parking_lot::Mutex`.
///
/// The default store of both tools; two tools sharing one store share the
/// session, which is what a single agent needs. No lock is held across an
/// `.await`.
#[derive(Debug, Default)]
pub struct MemoryWorktreeSessionStore {
    slots: Mutex<StoredSessions>,
}

impl MemoryWorktreeSessionStore {
    /// Builds an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Builds a store already holding `session`.
    #[must_use]
    pub fn with_session(mut self, session: WorktreeSessionState) -> Self {
        self.slots.get_mut().session = Some(session);
        self
    }

    /// Builds a store already holding the default name `name`.
    #[must_use]
    pub fn with_default_name(mut self, name: impl Into<String>) -> Self {
        self.slots.get_mut().default_name = Some(name.into());
        self
    }
}

impl WorktreeSessionStore for MemoryWorktreeSessionStore {
    fn session(&self) -> Option<WorktreeSessionState> {
        self.slots.lock().session.clone()
    }

    fn set_session(&self, session: Option<WorktreeSessionState>) {
        self.slots.lock().session = session;
    }

    fn default_name(&self) -> Option<String> {
        self.slots.lock().default_name.clone()
    }

    fn set_default_name(&self, name: Option<String>) {
        self.slots.lock().default_name = name;
    }
}

/// A shared [`WorktreeSessionStore`] handle.
pub type SharedWorktreeSessions = Arc<dyn WorktreeSessionStore>;

/// Arguments of the `enter_worktree` tool.
///
/// Unknown fields are rejected, so a model that passes an action or a path
/// finds out instead of having it silently ignored.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EnterWorktreeArgs {
    /// Worktree name; the session default is reused when omitted.
    #[serde(default)]
    pub name: Option<String>,
}

impl EnterWorktreeArgs {
    /// Builds arguments without a name.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Requests the worktree `name`.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
}

/// Arguments of the `exit_worktree` tool.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ExitWorktreeArgs {
    /// Exit action: `keep` or `remove`.
    pub action: String,
    /// Confirm that `remove` may discard uncommitted work and commits.
    #[serde(default)]
    pub discard_changes: bool,
}

impl ExitWorktreeArgs {
    /// Builds arguments for `action`.
    #[must_use]
    pub fn new(action: &str) -> Self {
        Self {
            action: action.to_owned(),
            discard_changes: false,
        }
    }

    /// Whether `action` is one of [`EXIT_ACTIONS`].
    #[must_use]
    pub fn is_known_action(&self) -> bool {
        EXIT_ACTIONS.contains(&self.action.as_str())
    }
}

/// The `enter_worktree` tool: creates or recovers a worktree and records the
/// session.
pub struct EnterWorktreeTool {
    card: ToolCard,
    manager: Arc<WorktreeManager>,
    sessions: SharedWorktreeSessions,
    permission: Option<Arc<PermissionEngine>>,
    original_cwd: Option<PathBuf>,
}

impl std::fmt::Debug for EnterWorktreeTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EnterWorktreeTool")
            .field("manager", &self.manager)
            .field("permission", &self.permission.is_some())
            .field("original_cwd", &self.original_cwd)
            .finish_non_exhaustive()
    }
}

impl EnterWorktreeTool {
    /// Builds the tool over `manager` with a card in `language` and a private
    /// in-memory session store.
    #[must_use]
    pub fn new(manager: Arc<WorktreeManager>, language: &str) -> Self {
        Self {
            card: Self::card_for(language),
            manager,
            sessions: Arc::new(MemoryWorktreeSessionStore::new()),
            permission: None,
            original_cwd: None,
        }
    }

    /// Shares the session store with another tool (or the host).
    #[must_use]
    pub fn with_sessions(mut self, sessions: SharedWorktreeSessions) -> Self {
        self.sessions = sessions;
        self
    }

    /// Wires the permission engine consulted before entering.
    #[must_use]
    pub fn with_permission(mut self, engine: Arc<PermissionEngine>) -> Self {
        self.permission = Some(engine);
        self
    }

    /// Sets the directory reported as the pre-worktree cwd.
    ///
    /// Defaults to the manager's workspace root; a host with its own cwd
    /// abstraction passes the real one.
    #[must_use]
    pub fn with_original_cwd(mut self, cwd: impl Into<PathBuf>) -> Self {
        self.original_cwd = Some(cwd.into());
        self
    }

    /// The manager the tool dispatches to.
    #[must_use]
    pub fn manager(&self) -> &Arc<WorktreeManager> {
        &self.manager
    }

    /// Builds the model-facing card in `language`.
    ///
    /// Unknown languages fall back to Chinese, like the other harness tools.
    #[must_use]
    pub fn card_for(language: &str) -> ToolCard {
        tool_card(
            ENTER_TOOL_NAME,
            enter_description(language),
            schema_for::<EnterWorktreeArgs>(),
            false,
        )
    }

    /// Validates the arguments and enters the worktree.
    async fn run(&self, inputs: &Value) -> ToolPayload {
        let args: EnterWorktreeArgs = match serde_json::from_value(inputs.clone()) {
            Ok(args) => args,
            Err(error) => return ToolPayload::fail(format!("invalid arguments: {error}")),
        };
        self.enter(&args).await
    }

    /// Implementation of [`Self::run`].
    async fn enter(&self, args: &EnterWorktreeArgs) -> ToolPayload {
        if let Some(session) = self.sessions.session() {
            return ToolPayload::fail(format!(
                "Already in worktree '{}'. Exit first with exit_worktree.",
                session.worktree_name
            ));
        }
        let requested = args
            .name
            .as_deref()
            .map(str::trim)
            .filter(|name| !name.is_empty());
        let name = match requested {
            Some(name) => name.to_owned(),
            None => match self.stored_default_name() {
                Some(name) => name,
                None => {
                    return ToolPayload::fail(
                        "name is required when no worktree session or default name is recorded",
                    )
                }
            },
        };
        let target = match self.manager.target_path(&name) {
            Ok(target) => target,
            Err(error) => return ToolPayload::fail(error.to_string()),
        };
        if let Some(reason) = deny_reason(self.permission.as_deref(), &target) {
            return ToolPayload::fail(reason);
        }

        // A registry record without a live git worktree is stale state: the
        // `worktree` tool's `prune` action reconciles it, entering would hide it.
        if self.listed_target(&target).await == Some(false)
            && self.has_registry_record(&name, &target)
        {
            return ToolPayload::fail(format!(
                "registered worktree `{name}` at `{}` is not live (git does not list it); run the `worktree` tool's `prune` action",
                target.display()
            ));
        }

        let result = match self
            .manager
            .create_with_options(&name, None, None, &WorktreeCreateOptions::default())
            .await
        {
            Ok(result) => result,
            Err(error) => {
                return ToolPayload::fail(format!("Failed to create worktree: {error}"));
            }
        };

        let session = WorktreeSessionState {
            original_cwd: self
                .original_cwd
                .clone()
                .unwrap_or_else(|| self.manager.workspace_root().to_path_buf()),
            worktree_path: result.path.clone(),
            worktree_name: name.clone(),
            worktree_branch: Some(result.branch.clone()),
            original_head_commit: result.head.clone(),
            used_sparse_paths: result.sparse.is_some(),
            ..WorktreeSessionState::default()
        };
        self.sessions.set_session(Some(session));
        self.sessions.set_default_name(Some(name.clone()));
        self.manager
            .emit(WorktreeEvent::Entered {
                name,
                path: result.path.clone(),
                branch: result.branch.clone(),
                existed: result.existed,
            })
            .await;

        let message = if result.existed {
            format!(
                "Entered existing worktree at {} on branch {}. CWD switched to worktree.",
                result.path.display(),
                result.branch
            )
        } else {
            format!(
                "Created worktree at {} on branch {}. CWD switched to worktree.",
                result.path.display(),
                result.branch
            )
        };
        ToolPayload::ok(json!({
            "worktree_path": result.path,
            "worktree_branch": result.branch,
            "message": message,
            "existed": result.existed,
            "sparse": result.sparse,
        }))
    }

    /// Whether git lists `target`, or `None` when the listing failed.
    async fn listed_target(&self, target: &Path) -> Option<bool> {
        let infos = self.manager.list().await.ok()?;
        Some(infos.iter().any(|info| same_path(&info.path, target)))
    }

    /// Whether an active registry record of `name` points at `target`.
    fn has_registry_record(&self, name: &str, target: &Path) -> bool {
        WorktreeIndex::load(&self.manager.index_path()).is_ok_and(|index| {
            index
                .get(name)
                .filter(|record| record.status == WorktreeState::Active)
                .is_some_and(|record| same_path(&record.path, target))
        })
    }
    /// The stored default name, trimmed and non-empty.
    fn stored_default_name(&self) -> Option<String> {
        self.sessions
            .default_name()
            .map(|name| name.trim().to_owned())
            .filter(|name| !name.is_empty())
    }
}

#[async_trait]
impl Tool for EnterWorktreeTool {
    fn card(&self) -> &ToolCard {
        &self.card
    }

    async fn invoke(&self, inputs: Value, ctx: &ToolContext) -> Result<Value> {
        ctx.check_cancelled()?;
        Ok(self.run(&inputs).await.to_json())
    }
}

/// The `exit_worktree` tool: keeps or removes the session's worktree.
pub struct ExitWorktreeTool {
    card: ToolCard,
    manager: Arc<WorktreeManager>,
    sessions: SharedWorktreeSessions,
    permission: Option<Arc<PermissionEngine>>,
}

impl std::fmt::Debug for ExitWorktreeTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExitWorktreeTool")
            .field("manager", &self.manager)
            .field("permission", &self.permission.is_some())
            .finish_non_exhaustive()
    }
}

impl ExitWorktreeTool {
    /// Builds the tool over `manager` with a card in `language` and a private
    /// in-memory session store.
    #[must_use]
    pub fn new(manager: Arc<WorktreeManager>, language: &str) -> Self {
        Self {
            card: Self::card_for(language),
            manager,
            sessions: Arc::new(MemoryWorktreeSessionStore::new()),
            permission: None,
        }
    }

    /// Shares the session store with another tool (or the host).
    #[must_use]
    pub fn with_sessions(mut self, sessions: SharedWorktreeSessions) -> Self {
        self.sessions = sessions;
        self
    }

    /// Wires the permission engine consulted before exiting.
    #[must_use]
    pub fn with_permission(mut self, engine: Arc<PermissionEngine>) -> Self {
        self.permission = Some(engine);
        self
    }

    /// The manager the tool dispatches to.
    #[must_use]
    pub fn manager(&self) -> &Arc<WorktreeManager> {
        &self.manager
    }

    /// Builds the model-facing card in `language`.
    ///
    /// Unknown languages fall back to Chinese, like the other harness tools.
    #[must_use]
    pub fn card_for(language: &str) -> ToolCard {
        tool_card(
            EXIT_TOOL_NAME,
            exit_description(language),
            schema_for::<ExitWorktreeArgs>(),
            false,
        )
    }

    /// Validates the arguments and exits the session.
    async fn run(&self, inputs: &Value) -> ToolPayload {
        let args: ExitWorktreeArgs = match serde_json::from_value(inputs.clone()) {
            Ok(args) => args,
            Err(error) => return ToolPayload::fail(format!("invalid arguments: {error}")),
        };
        self.exit(&args).await
    }

    /// Implementation of [`Self::run`].
    async fn exit(&self, args: &ExitWorktreeArgs) -> ToolPayload {
        let Some(session) = self.sessions.session() else {
            return ToolPayload::fail("No active worktree session to exit.");
        };
        let action = args.action.trim().to_ascii_lowercase();
        if !EXIT_ACTIONS.contains(&action.as_str()) {
            return ToolPayload::fail("'action' must be 'keep' or 'remove'.");
        }
        if let Some(reason) = deny_reason(self.permission.as_deref(), &session.worktree_path) {
            return ToolPayload::fail(reason);
        }
        let name = session.worktree_name.clone();
        let branch = session
            .worktree_branch
            .clone()
            .unwrap_or_else(|| "unknown".to_owned());
        let original_cwd = session.original_cwd.clone();

        if action == "keep" {
            self.sessions.set_session(None);
            self.manager
                .emit(WorktreeEvent::Exited {
                    name: name.clone(),
                    path: session.worktree_path.clone(),
                    action: "keep".to_owned(),
                })
                .await;
            let message = format!(
                "Kept worktree '{name}' (branch {branch}). In this session, enter_worktree without a name will re-enter it; from another session, pass name='{name}'. Returned to {}",
                original_cwd.display()
            );
            return ToolPayload::ok(json!({
                "action": "keep",
                "original_cwd": original_cwd,
                "worktree_path": session.worktree_path,
                "worktree_branch": session.worktree_branch,
                "worktree_name": name,
                "message": message,
            }));
        }

        let Ok(summary) = self.manager.change_summary(&name).await else {
            return ToolPayload::fail(format!(
                "Could not verify worktree state at {}. Refusing to remove without explicit confirmation. Set discard_changes=True to proceed, or use action='keep' to preserve the worktree.",
                session.worktree_path.display()
            ));
        };
        if summary.is_dirty() && !args.discard_changes {
            let mut parts: Vec<String> = Vec::new();
            if summary.changed_files > 0 {
                parts.push(format!("{} uncommitted files", summary.changed_files));
            }
            if summary.commits > 0 {
                parts.push(format!("{} commits on {branch}", summary.commits));
            }
            return ToolPayload::fail(format!(
                "Worktree has {}. Removing will discard this work permanently. Confirm with the user, then set discard_changes=True to proceed, or use action='keep' to preserve the work.",
                parts.join(" and ")
            ));
        }
        if let Err(error) = self.manager.remove(&name, args.discard_changes).await {
            return ToolPayload::fail(format!("Failed to exit worktree: {error}"));
        }
        self.sessions.set_session(None);
        self.manager
            .emit(WorktreeEvent::Exited {
                name: name.clone(),
                path: session.worktree_path.clone(),
                action: "remove".to_owned(),
            })
            .await;
        let message = format!(
            "Removed worktree '{name}' (branch {branch}). Returned to {}",
            original_cwd.display()
        );
        ToolPayload::ok(json!({
            "action": "remove",
            "original_cwd": original_cwd,
            "worktree_path": session.worktree_path,
            "worktree_branch": session.worktree_branch,
            "worktree_name": name,
            "message": message,
            "discarded_files": summary.changed_files,
            "discarded_commits": summary.commits,
        }))
    }
}

#[async_trait]
impl Tool for ExitWorktreeTool {
    fn card(&self) -> &ToolCard {
        &self.card
    }

    async fn invoke(&self, inputs: Value, ctx: &ToolContext) -> Result<Value> {
        ctx.check_cancelled()?;
        Ok(self.run(&inputs).await.to_json())
    }
}

/// The reason a `deny` decision rejects an operation on `target`.
///
/// The guard is consulted with the write-tool argument shape, because entering
/// a worktree writes into the repository. `ask` and `allow` pass, and an absent
/// engine fails open — the same policy the built-in tools apply.
fn deny_reason(engine: Option<&PermissionEngine>, target: &Path) -> Option<String> {
    let engine = engine?;
    let args = json!({ "file_path": target.to_string_lossy() });
    let result = engine.check_permission("write_file", &args);
    if result.permission == PermissionLevel::Deny {
        Some(
            result
                .reason
                .clone()
                .unwrap_or_else(|| "Permission denied: worktree".to_owned()),
        )
    } else {
        None
    }
}

/// Compares two paths after lexical normalisation.
fn same_path(left: &Path, right: &Path) -> bool {
    normalize_lexically(left) == normalize_lexically(right)
}

/// The bilingual `enter_worktree` card description.
fn enter_description(language: &str) -> &'static str {
    if language == "en" {
        "Create or enter an isolated git worktree under `.worktrees/<name>` in the current repository. A worktree is the same repository with a separate working copy, so file changes there do not affect the main checkout or other worktrees. Only use this tool when the user explicitly mentions worktrees. `name` is optional (at most 64 characters, no `..`, no absolute paths); without it the session's recorded default name is reused."
    } else {
        "在当前仓库中创建或进入位于 `.worktrees/<name>` 下的隔离 git worktree。worktree 是同一个仓库的独立工作副本，其中的文件改动不会影响主检出或其他 worktree。仅在用户明确提到 worktree 时使用此工具。`name` 可选（最长 64 个字符，不得包含 `..`，也不能是绝对路径）；省略时复用当前会话记录的默认名称。"
    }
}

/// The bilingual `exit_worktree` card description.
fn exit_description(language: &str) -> &'static str {
    if language == "en" {
        "Exit the current worktree session. `action='keep'` preserves the worktree for a later `enter_worktree`; `action='remove'` deletes it, discarding all uncommitted work unless `discard_changes=true` is also set. Removing is destructive: get explicit user confirmation before passing `discard_changes=true`."
    } else {
        "退出当前 worktree 会话。`action='keep'` 保留 worktree，稍后可再次 `enter_worktree`；`action='remove'` 删除 worktree，除非同时设置 `discard_changes=true`，否则会丢弃全部未提交改动。删除不可恢复：传入 `discard_changes=true` 前必须先获得用户明确确认。"
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn session_defaults_to_the_auto_lifecycle_policy() {
        let session = WorktreeSessionState::default();
        assert_eq!(session.lifecycle_policy, "auto");
        assert_eq!(session.worktree_name, "");
        assert!(!session.used_sparse_paths);
    }

    #[test]
    fn session_round_trips_through_json_with_missing_fields() {
        let session = WorktreeSessionState::new(
            "feature",
            PathBuf::from("/repo/.worktrees/feature"),
            Some("worktree-feature".to_owned()),
        );
        let value = session.to_json().unwrap();
        assert_eq!(WorktreeSessionState::from_json(&value).unwrap(), session);

        let partial = json!({
            "worktree_name": "feature",
            "worktree_path": "/repo/.worktrees/feature",
        });
        let restored = WorktreeSessionState::from_json(&partial).unwrap();
        assert_eq!(restored.lifecycle_policy, "auto");
        assert!(restored.original_cwd.as_os_str().is_empty());
    }

    #[test]
    fn memory_store_holds_session_and_default_name() {
        let store = MemoryWorktreeSessionStore::new();
        assert!(store.session().is_none());
        assert!(store.default_name().is_none());
        let session = WorktreeSessionState::new("feature", PathBuf::from("/tmp/w"), None);
        store.set_session(Some(session.clone()));
        store.set_default_name(Some("feature".to_owned()));
        assert_eq!(store.session(), Some(session));
        assert_eq!(store.default_name().as_deref(), Some("feature"));
        store.set_session(None);
        store.set_default_name(None);
        assert!(store.session().is_none());
        assert!(store.default_name().is_none());
    }

    #[test]
    fn exit_actions_are_keep_and_remove() {
        assert!(ExitWorktreeArgs::new("keep").is_known_action());
        assert!(ExitWorktreeArgs::new("remove").is_known_action());
        assert!(!ExitWorktreeArgs::new("drop").is_known_action());
    }
}
