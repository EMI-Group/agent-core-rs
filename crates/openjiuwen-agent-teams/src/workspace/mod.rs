//! Member workspace allocation.
//!
//! A teammate needs a directory to write artifacts into. This module owns the on-disk contract:
//!
//! - [`layout`] builds the session-scoped team tree
//!   (`<base>/<team>/sessions/<session>/workspaces/<member>_workspace`) and sanitizes every
//!   untrusted path segment;
//! - [`local`] implements [`WorkspaceProvider`] against a local directory (and, when a host injects
//!   a backend, against git worktrees);
//! - [`worktree`] owns deterministic worktree names, branch names, digests and the
//!   [`WorktreeBackend`] seam.
//!
//! The git worktree backend is deferred: without an injected backend a worktree request fails
//! with [`Error::Unsupported`](openjiuwen_common::error::Error::Unsupported) instead of silently
//! sharing a directory.
//!
//! # Examples
//!
//! ```
//! use std::path::Path;
//!
//! use openjiuwen_agent_teams::workspace::MemberWorkspaceLayout;
//!
//! let layout = MemberWorkspaceLayout::new("/srv/openjiuwen/.agent_teams");
//! let root = layout.member_workspace("my-team", "session-1", "dev-1").unwrap();
//! assert_eq!(
//!     root,
//!     Path::new("/srv/openjiuwen/.agent_teams/my-team/sessions/session-1/workspaces/dev-1_workspace"),
//! );
//!
//! // Hostile input can never escape the layout base.
//! assert!(layout.team_dir("../../etc").is_err());
//! ```

use std::path::PathBuf;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use openjiuwen_common::error::Result;

use crate::schema::{TeamWorkspaceSpec, WorkspaceSpec};

pub mod layout;
pub mod local;
pub mod worktree;

pub use layout::*;
pub use local::*;
pub use worktree::*;

/// How a member workspace is isolated from the teammates of the same team.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IsolationMode {
    /// One session-scoped directory per member (Python `isolation="none"`); the default.
    #[default]
    None,
    /// One git worktree per member (Python `isolation="worktree"`); deferred.
    Worktree,
}

impl IsolationMode {
    /// Returns the wire value of the mode (`"none"` / `"worktree"`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Worktree => "worktree",
        }
    }
}

/// The workspace a member was allocated.
///
/// The record is plain data: allocating one does not imply the directory exists (a disabled shared
/// workspace, or a backend that creates worktrees lazily, both return a record for a not-yet-existing
/// path).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct MemberWorkspace {
    /// Team the member belongs to.
    pub team_name: String,
    /// Session that owns the workspace.
    pub session_id: String,
    /// Member the workspace belongs to.
    pub member_name: String,
    /// Absolute root directory of the workspace.
    pub root: PathBuf,
    /// Isolation mode the workspace was allocated with.
    pub isolation: IsolationMode,
    /// Worktree details, present only for [`IsolationMode::Worktree`].
    pub worktree: Option<WorktreeInfo>,
}

impl MemberWorkspace {
    /// Creates the record of a shared (non-worktree) member workspace rooted at `root`.
    #[must_use]
    pub fn new(
        team_name: impl Into<String>,
        session_id: impl Into<String>,
        member_name: impl Into<String>,
        root: impl Into<PathBuf>,
    ) -> Self {
        Self {
            team_name: team_name.into(),
            session_id: session_id.into(),
            member_name: member_name.into(),
            root: root.into(),
            isolation: IsolationMode::None,
            worktree: None,
        }
    }

    /// Returns [`Self::root`] as a display string, replacing invalid UTF-8 lossily.
    #[must_use]
    pub fn root_str(&self) -> String {
        self.root.to_string_lossy().into_owned()
    }

    /// Returns `true` when the member shares the plain local directory layout (no worktree).
    #[must_use]
    pub fn is_shared(&self) -> bool {
        matches!(self.isolation, IsolationMode::None)
    }
}

/// What a caller asks a [`WorkspaceProvider`] to allocate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct WorkspaceRequest {
    /// Team the member belongs to.
    pub team_name: String,
    /// Session that owns the workspace.
    pub session_id: String,
    /// Member the workspace belongs to.
    pub member_name: String,
    /// Team-level workspace configuration (Python `TeamWorkspaceConfig`).
    pub spec: TeamWorkspaceSpec,
    /// Member-level workspace configuration, when the member declares one.
    pub member_spec: Option<WorkspaceSpec>,
    /// Isolation mode to allocate with.
    pub isolation: IsolationMode,
}

impl WorkspaceRequest {
    /// Creates a request with the default (disabled) team workspace spec and no isolation.
    #[must_use]
    pub fn new(
        team_name: impl Into<String>,
        session_id: impl Into<String>,
        member_name: impl Into<String>,
    ) -> Self {
        Self {
            team_name: team_name.into(),
            session_id: session_id.into(),
            member_name: member_name.into(),
            spec: TeamWorkspaceSpec::default(),
            member_spec: None,
            isolation: IsolationMode::None,
        }
    }
}

/// Allocates and releases member workspaces.
///
/// Allocation is idempotent per `(team, session, member)`: allocating twice yields the same
/// directory, and an existing directory is reused. Release is fail-closed — the Python runtime
/// keeps member directories until the team is deleted or the session is cleaned, so a released
/// shared workspace stays on disk; only a worktree is handed back to its backend.
#[async_trait]
pub trait WorkspaceProvider: Send + Sync {
    /// Allocates the workspace described by `request`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](openjiuwen_common::error::Error::Validation) for an unsafe
    /// team/session/member name or project root, [`Error::Unsupported`](openjiuwen_common::error::Error::Unsupported)
    /// when the requested isolation mode is not implemented, and the provider's own error when the
    /// workspace cannot be created.
    async fn allocate(&self, request: &WorkspaceRequest) -> Result<MemberWorkspace>;

    /// Releases `workspace`.
    ///
    /// # Errors
    ///
    /// Returns the backend's error when a worktree cannot be removed; a shared workspace is always
    /// retained, so releasing one cannot fail.
    async fn release(&self, workspace: &MemberWorkspace) -> Result<()>;
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn isolation_mode_wire_values() {
        assert_eq!(IsolationMode::default(), IsolationMode::None);
        assert_eq!(IsolationMode::None.as_str(), "none");
        assert_eq!(IsolationMode::Worktree.as_str(), "worktree");
        assert_eq!(
            serde_json::to_value(IsolationMode::None).unwrap(),
            json!("none")
        );
        assert_eq!(
            serde_json::to_value(IsolationMode::Worktree).unwrap(),
            json!("worktree")
        );
        assert_eq!(
            serde_json::from_value::<IsolationMode>(json!("worktree")).unwrap(),
            IsolationMode::Worktree
        );
    }

    #[test]
    fn member_workspace_round_trips_and_reports_shared_state() {
        let workspace = MemberWorkspace::new("my-team", "session-1", "dev-1", "/tmp/ws");
        assert!(workspace.is_shared());
        assert_eq!(workspace.root_str(), "/tmp/ws");
        let value = serde_json::to_value(&workspace).unwrap();
        assert_eq!(value["isolation"], json!("none"));
        assert_eq!(value["worktree"], json!(null));
        let parsed: MemberWorkspace = serde_json::from_value(value).unwrap();
        assert_eq!(parsed, workspace);
    }

    #[test]
    fn worktree_workspace_round_trips() {
        let workspace = MemberWorkspace {
            worktree: Some(WorktreeInfo {
                name: "agent-my-team-dev-1-abcdef1234".to_owned(),
                path: PathBuf::from("/tmp/worktrees/agent-my-team-dev-1-abcdef1234"),
                branch: "worktree-agent-my-team-dev-1-abcdef1234".to_owned(),
            }),
            isolation: IsolationMode::Worktree,
            ..MemberWorkspace::new("my-team", "session-1", "dev-1", "/tmp/worktrees")
        };
        assert!(!workspace.is_shared());
        let parsed: MemberWorkspace =
            serde_json::from_value(serde_json::to_value(&workspace).unwrap()).unwrap();
        assert_eq!(parsed, workspace);
    }

    #[test]
    fn request_defaults_match_python() {
        let request = WorkspaceRequest::new("my-team", "session-1", "dev-1");
        assert!(!request.spec.enabled);
        assert!(request.spec.version_control);
        assert!(request.member_spec.is_none());
        assert_eq!(request.isolation, IsolationMode::None);
    }
}
