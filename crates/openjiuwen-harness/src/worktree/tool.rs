//! The model-facing `worktree` tool.
//!
//! One tool, seven actions (`create`, `list`, `remove`, `lock`, `unlock`,
//! `prune`, `status`), following the Python `enter_worktree`/`exit_worktree`
//! model surface but exposed as a single action-dispatching card. The tool is a
//! thin payload layer over [`WorktreeManager`]:
//!
//! - every expected failure — bad action, missing name, a refused name, an
//!   occupied target, a dirty worktree, a missing git binary — comes back as a
//!   [`ToolPayload`] with `success = false`, never as `Err` and never as a panic;
//! - when a [`PermissionEngine`] is wired, mutating actions consult the path
//!   guards for the target path first and refuse on a `deny`; `ask` and `allow`
//!   do not block the tool (the interrupt rail owns asking a human).
//!
//! The tool is mounted by [`crate::tools::ToolRegistry`] as
//! [`BuiltinTool::Worktree`](crate::tools::BuiltinTool::Worktree), the last of the
//! extended tools; the registry builds it from a [`WorktreeManager`] plus its
//! language and permission engine, and a host outside the registry builds it with
//! [`WorktreeTool::new`].
#![allow(clippy::result_large_err)]
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use openjiuwen_common::error::Result;
use openjiuwen_foundation::tool::{Tool, ToolCard, ToolContext};

use crate::permission::{PermissionEngine, PermissionLevel};
use crate::tools::{schema_for, tool_card, ToolPayload};

use super::manager::WorktreeManager;

/// Name of the model-facing tool.
pub const TOOL_NAME: &str = "worktree";

/// Actions the tool accepts, in card order.
pub const ACTIONS: [&str; 7] = [
    "create", "list", "remove", "lock", "unlock", "prune", "status",
];

/// Arguments of the `worktree` tool.
///
/// Only `action` is required; every other field is action-specific and has a
/// serde default, so models can omit what an action does not use.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct WorktreeArgs {
    /// Action to run: `create`, `list`, `remove`, `lock`, `unlock`, `prune` or `status`.
    pub action: String,
    /// Worktree name (required by every action except `list` and `prune`).
    #[serde(default)]
    pub name: Option<String>,
    /// Branch to create for `create` (default: `worktree-<slug>`).
    #[serde(default)]
    pub branch: Option<String>,
    /// Base reference for `create` (default: `HEAD`).
    #[serde(default)]
    pub base: Option<String>,
    /// Skip the dirty-worktree check for `remove`.
    #[serde(default)]
    pub force: bool,
    /// Lock reason for `lock`.
    #[serde(default)]
    pub reason: Option<String>,
}

impl WorktreeArgs {
    /// Builds arguments for `action`.
    #[must_use]
    pub fn new(action: &str) -> Self {
        Self {
            action: action.to_owned(),
            ..Self::default()
        }
    }

    /// Sets the worktree name.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Whether `action` is one of [`ACTIONS`].
    #[must_use]
    pub fn is_known_action(&self) -> bool {
        ACTIONS.contains(&self.action.as_str())
    }
}

/// The `worktree` tool: action dispatch over a [`WorktreeManager`].
pub struct WorktreeTool {
    card: ToolCard,
    manager: Arc<WorktreeManager>,
    permission: Option<Arc<PermissionEngine>>,
}

impl std::fmt::Debug for WorktreeTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorktreeTool")
            .field("manager", &self.manager)
            .field("permission", &self.permission.is_some())
            .finish_non_exhaustive()
    }
}

impl WorktreeTool {
    /// Builds the tool over `manager` with a card in `language`.
    #[must_use]
    pub fn new(manager: Arc<WorktreeManager>, language: &str) -> Self {
        Self {
            card: Self::card_for(language),
            manager,
            permission: None,
        }
    }

    /// Wires the permission engine consulted by mutating actions.
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
            TOOL_NAME,
            description(language),
            schema_for::<WorktreeArgs>(),
            false,
        )
    }

    /// Resolves the target path of `name` and applies the permission guards.
    fn checked_target(&self, name: &str) -> std::result::Result<PathBuf, ToolPayload> {
        let target = self
            .manager
            .target_path(name)
            .map_err(|error| ToolPayload::fail(error.to_string()))?;
        if let Some(reason) = deny_reason(self.permission.as_ref(), TOOL_NAME, &target) {
            return Err(ToolPayload::fail(reason));
        }
        Ok(target)
    }

    /// Dispatches one invocation and returns the payload.
    async fn dispatch(&self, inputs: &Value) -> ToolPayload {
        let args: WorktreeArgs = match serde_json::from_value(inputs.clone()) {
            Ok(args) => args,
            Err(error) => return ToolPayload::fail(format!("invalid arguments: {error}")),
        };
        let action = args.action.trim().to_ascii_lowercase();
        match action.as_str() {
            "create" => self.create(&args).await,
            "list" => self.list().await,
            "remove" => self.remove(&args).await,
            "lock" => self.lock(&args).await,
            "unlock" => self.unlock(&args).await,
            "prune" => self.prune().await,
            "status" => self.status(&args).await,
            other => ToolPayload::fail(format!(
                "unknown worktree action `{other}`: expected one of {}",
                ACTIONS.join(", ")
            )),
        }
    }

    /// Runs `create`.
    async fn create(&self, args: &WorktreeArgs) -> ToolPayload {
        let Some(name) = required_name(args) else {
            return ToolPayload::fail("name is required for the create action");
        };
        if let Err(payload) = self.checked_target(name) {
            return payload;
        }
        match self
            .manager
            .create(name, args.branch.as_deref(), args.base.as_deref())
            .await
        {
            Ok(result) => match serde_json::to_value(&result) {
                Ok(value) => ToolPayload::ok(json!({ "action": "create", "worktree": value })),
                Err(error) => ToolPayload::fail(format!("cannot serialize result: {error}")),
            },
            Err(error) => ToolPayload::fail(error.to_string()),
        }
    }

    /// Runs `list`.
    async fn list(&self) -> ToolPayload {
        match self.manager.list().await {
            Ok(infos) => match serde_json::to_value(&infos) {
                Ok(value) => ToolPayload::ok(json!({ "action": "list", "worktrees": value })),
                Err(error) => ToolPayload::fail(format!("cannot serialize result: {error}")),
            },
            Err(error) => ToolPayload::fail(error.to_string()),
        }
    }

    /// Runs `remove`.
    async fn remove(&self, args: &WorktreeArgs) -> ToolPayload {
        let Some(name) = required_name(args) else {
            return ToolPayload::fail("name is required for the remove action");
        };
        if let Err(payload) = self.checked_target(name) {
            return payload;
        }
        match self.manager.remove(name, args.force).await {
            Ok(info) => ToolPayload::ok(json!({
                "action": "remove",
                "name": name,
                "path": info.path,
                "branch": info.branch,
            })),
            Err(error) => ToolPayload::fail(error.to_string()),
        }
    }

    /// Runs `lock`.
    async fn lock(&self, args: &WorktreeArgs) -> ToolPayload {
        let Some(name) = required_name(args) else {
            return ToolPayload::fail("name is required for the lock action");
        };
        let target = match self.checked_target(name) {
            Ok(target) => target,
            Err(payload) => return payload,
        };
        match self.manager.lock(name, args.reason.as_deref()).await {
            Ok(()) => ToolPayload::ok(json!({
                "action": "lock",
                "name": name,
                "path": target,
                "reason": args.reason,
            })),
            Err(error) => ToolPayload::fail(error.to_string()),
        }
    }

    /// Runs `unlock`.
    async fn unlock(&self, args: &WorktreeArgs) -> ToolPayload {
        let Some(name) = required_name(args) else {
            return ToolPayload::fail("name is required for the unlock action");
        };
        let target = match self.checked_target(name) {
            Ok(target) => target,
            Err(payload) => return payload,
        };
        match self.manager.unlock(name).await {
            Ok(()) => ToolPayload::ok(json!({
                "action": "unlock",
                "name": name,
                "path": target,
            })),
            Err(error) => ToolPayload::fail(error.to_string()),
        }
    }

    /// Runs `prune`.
    async fn prune(&self) -> ToolPayload {
        match self.manager.prune().await {
            Ok(removed) => ToolPayload::ok(json!({ "action": "prune", "removed": removed })),
            Err(error) => ToolPayload::fail(error.to_string()),
        }
    }

    /// Runs `status`.
    async fn status(&self, args: &WorktreeArgs) -> ToolPayload {
        let Some(name) = required_name(args) else {
            return ToolPayload::fail("name is required for the status action");
        };
        match self.manager.status(name).await {
            Ok(status) => match serde_json::to_value(&status) {
                Ok(value) => ToolPayload::ok(json!({ "action": "status", "worktree": value })),
                Err(error) => ToolPayload::fail(format!("cannot serialize result: {error}")),
            },
            Err(error) => ToolPayload::fail(error.to_string()),
        }
    }
}

/// The trimmed, non-empty `name` argument, when present.
fn required_name(args: &WorktreeArgs) -> Option<&str> {
    args.name
        .as_deref()
        .map(str::trim)
        .filter(|name| !name.is_empty())
}

/// The reason a `deny` decision rejects an operation on `target`.
///
/// The guard is consulted with the write-tool argument shape, because a worktree
/// operation writes into the repository. `ask` and `allow` pass, and an absent
/// engine fails open — the same policy the built-in tools apply. `tool_name` only
/// names the tool in the fallback reason, used when the engine supplies no
/// reason of its own.
pub(crate) fn deny_reason(
    engine: Option<&Arc<PermissionEngine>>,
    tool_name: &str,
    target: &Path,
) -> Option<String> {
    let engine = engine?;
    let args = json!({ "file_path": target.to_string_lossy() });
    let result = engine.check_permission("write_file", &args);
    if result.permission == PermissionLevel::Deny {
        Some(
            result
                .reason
                .clone()
                .unwrap_or_else(|| format!("Permission denied: {tool_name}")),
        )
    } else {
        None
    }
}

/// The bilingual card description.
fn description(language: &str) -> &'static str {
    if language == "en" {
        "Create and manage git worktrees for this repository. Worktrees live under `.worktrees/<name>` inside the project; never use `..` or paths outside it. Actions: `create` (name, optional branch/base), `list`, `remove` (name, optional force), `lock` (name, optional reason), `unlock` (name), `prune`, `status` (name)."
    } else {
        "在当前仓库中创建和管理 git worktree。worktree 位于项目内的 `.worktrees/<name>`，禁止使用 `..` 或项目之外的路径。操作：`create`（name，可选 branch/base）、`list`、`remove`（name，可选 force）、`lock`（name，可选 reason）、`unlock`（name）、`prune`、`status`（name）。"
    }
}

#[async_trait]
impl Tool for WorktreeTool {
    fn card(&self) -> &ToolCard {
        &self.card
    }

    async fn invoke(&self, inputs: Value, ctx: &ToolContext) -> Result<Value> {
        ctx.check_cancelled()?;
        Ok(self.dispatch(&inputs).await.to_json())
    }
}
