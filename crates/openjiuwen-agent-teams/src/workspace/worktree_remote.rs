//! The remote (cross-node) [`WorktreeBackend`].
//!
//! Python parity source: `openjiuwen/agent_teams/worktree_remote.py`
//! (`RemoteWorktreeBackend` on the leader side, `WorktreeRemoteHandler` on the node side).
//!
//! # Wire contract
//!
//! The Python contract is **not** HTTP: the leader serializes a `WorktreeRemoteRequest`, sends it to
//! one node over the team `Messager` (`send_and_wait`) and parses the `WorktreeRemoteResponse` that
//! comes back. Three actions exist — `create`, `remove`, `exists` — and the response carries
//! `success`, `worktree_path`, `worktree_branch`, `head_commit`, `existed`, `exists` and `error`.
//!
//! Because the transport is the (deferred) distributed `Messager`, this port splits the contract in
//! two:
//!
//! - the **protocol layer** is implemented and tested here: request/response wire types, the action
//!   set, request validation and the translation into [`WorktreeInfo`] / [`Error`];
//! - the **transport** is an injected [`RemoteDispatch`] seam (the Rust counterpart of
//!   `Messager.send_and_wait`). A node that speaks `ZeroMQ`/`Pulsar` registers its own dispatch
//!   implementation; [`UnavailableRemoteDispatch`] is the fail-closed implementation whose error
//!   names the Python capability.
//!
//! Using `reqwest` here would be a protocol change, not a port: Python's handler is reached through a
//! P2P broker, so an HTTP client would not talk to an openJiuwen node.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::debug;

use openjiuwen_common::error::{Error, Result};

use super::worktree::{WorktreeBackend, WorktreeCreateRequest, WorktreeInfo};

/// Operation a remote node is asked to perform.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RemoteWorktreeAction {
    /// Clone if needed, then create a worktree on the node.
    #[default]
    Create,
    /// Remove a worktree on the node.
    Remove,
    /// Report whether a worktree exists on the node.
    Exists,
}

impl RemoteWorktreeAction {
    /// Returns the wire value of the action (`"create"` / `"remove"` / `"exists"`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Create => "create",
            Self::Remove => "remove",
            Self::Exists => "exists",
        }
    }
}

/// Request sent to a remote node to manage a worktree (Python `WorktreeRemoteRequest`).
///
/// Unknown fields are tolerated and every optional field defaults to `None`, so a node written
/// against a newer Python version still parses.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct RemoteWorktreeRequest {
    /// Operation to perform.
    pub action: RemoteWorktreeAction,
    /// Worktree slug; required for [`RemoteWorktreeAction::Create`].
    pub slug: Option<String>,
    /// Git remote URL of the source repository; required by a node that has to clone.
    pub repo_url: Option<String>,
    /// Branch the node bases the new worktree on (defaults to the node's `main`).
    pub base_branch: Option<String>,
    /// Absolute path on the remote node; required for `remove` / `exists`.
    pub worktree_path: Option<String>,
    /// Opaque backend configuration forwarded to the node.
    ///
    /// Python sends its serialized `WorktreeConfig`. This port ships no equivalent type, so the
    /// field is forwarded verbatim when a caller sets it and stays `null` otherwise.
    pub config: Option<Value>,
}

impl RemoteWorktreeRequest {
    /// Builds a `create` request for `slug`, branching off `base_branch`.
    #[must_use]
    pub fn create(slug: impl Into<String>, base_branch: Option<String>) -> Self {
        Self {
            action: RemoteWorktreeAction::Create,
            slug: Some(slug.into()),
            base_branch,
            ..Self::default()
        }
    }

    /// Builds a `remove` request for the worktree at `path`.
    #[must_use]
    pub fn remove(path: impl Into<String>) -> Self {
        Self {
            action: RemoteWorktreeAction::Remove,
            worktree_path: Some(path.into()),
            ..Self::default()
        }
    }

    /// Builds an `exists` request for the worktree at `path`.
    #[must_use]
    pub fn exists(path: impl Into<String>) -> Self {
        Self {
            action: RemoteWorktreeAction::Exists,
            worktree_path: Some(path.into()),
            ..Self::default()
        }
    }

    /// Checks that the request carries everything its action needs.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when a `create` request has no slug or a `remove`/`exists`
    /// request has no worktree path.
    pub fn validate(&self) -> Result<()> {
        match self.action {
            RemoteWorktreeAction::Create => {
                if self
                    .slug
                    .as_deref()
                    .is_none_or(|slug| slug.trim().is_empty())
                {
                    return Err(Error::validation(
                        "remote worktree create requires a worktree slug",
                    ));
                }
            }
            RemoteWorktreeAction::Remove | RemoteWorktreeAction::Exists => {
                if self
                    .worktree_path
                    .as_deref()
                    .is_none_or(|path| path.trim().is_empty())
                {
                    return Err(Error::validation(format!(
                        "remote worktree {} requires a worktree path",
                        self.action.as_str()
                    )));
                }
            }
        }
        Ok(())
    }
}

/// Response of a remote node (Python `WorktreeRemoteResponse`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct RemoteWorktreeResponse {
    /// Whether the operation succeeded; Python defaults this to `true`.
    #[serde(default = "default_true")]
    pub success: bool,
    /// Absolute path of the worktree on the remote node.
    #[serde(default)]
    pub worktree_path: Option<String>,
    /// Branch checked out in the worktree on the remote node.
    #[serde(default)]
    pub worktree_branch: Option<String>,
    /// `HEAD` commit id after creation.
    #[serde(default)]
    pub head_commit: Option<String>,
    /// Whether the worktree already existed (the node recovered an existing one).
    #[serde(default)]
    pub existed: bool,
    /// Whether the worktree exists, for [`RemoteWorktreeAction::Exists`].
    #[serde(default)]
    pub exists: bool,
    /// Failure detail reported by the node.
    #[serde(default)]
    pub error: Option<String>,
}

/// Default value of [`RemoteWorktreeResponse::success`] (Python `success: bool = True`).
const fn default_true() -> bool {
    true
}

impl RemoteWorktreeResponse {
    /// Builds the failure response a node returns when it cannot serve a request.
    #[must_use]
    pub fn failure(error: impl Into<String>) -> Self {
        Self {
            success: false,
            worktree_path: None,
            worktree_branch: None,
            head_commit: None,
            existed: false,
            exists: false,
            error: Some(error.into()),
        }
    }

    /// Converts a node response into an [`Error`] when it reports failure.
    ///
    /// # Errors
    ///
    /// Returns [`Error::SysOp`] carrying the node's own `error` text when `success` is `false`.
    pub fn into_success(self, node_id: &str, action: RemoteWorktreeAction) -> Result<Self> {
        if self.success {
            Ok(self)
        } else {
            Err(Error::sys_op(format!(
                "remote node {node_id} reported failure for worktree {}: {}",
                action.as_str(),
                self.error.as_deref().unwrap_or("no error detail")
            )))
        }
    }
}

/// Transport seam that carries a [`RemoteWorktreeRequest`] to one node and back.
///
/// The Rust counterpart of Python's `Messager.send_and_wait`: implementations shipped by the
/// distributed transports (`ZeroMQ`, `Pulsar`) forward the JSON payload and await the reply payload.
#[async_trait]
pub trait RemoteDispatch: Send + Sync {
    /// Returns the dispatch name used in logs.
    fn name(&self) -> &str;

    /// Sends `request` to `node_id` and awaits the reply payload.
    ///
    /// # Errors
    ///
    /// Returns the transport error when the node is unreachable, the request cannot be encoded, or
    /// no reply arrives in time.
    async fn send_and_wait(&self, node_id: &str, request: Value) -> Result<Value>;
}

/// A [`RemoteDispatch`] of a deployment without a distributed transport.
///
/// Every send fails with [`Error::Unsupported`] naming the Python capability, so a caller learns
/// that remote worktrees need a distributed broker instead of waiting for a reply that never comes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UnavailableRemoteDispatch;

#[async_trait]
impl RemoteDispatch for UnavailableRemoteDispatch {
    fn name(&self) -> &'static str {
        "unavailable"
    }

    async fn send_and_wait(&self, node_id: &str, _request: Value) -> Result<Value> {
        Err(Error::unsupported(format!(
            "remote worktree requests for node {node_id} need a distributed Messager transport, \
             which is not shipped (Python capability: agent_teams.worktree_remote.RemoteWorktreeBackend)"
        )))
    }
}

/// A [`WorktreeBackend`] that drives teammate worktrees on another node.
///
/// The adapter is the leader side of the protocol: it turns a [`WorktreeCreateRequest`] into a
/// `create` request for `node_id`, and reports the node's own path/branch back in
/// [`WorktreeInfo`]. The local [`WorktreeCreateRequest::path`] is ignored, exactly as in Python: the
/// remote node owns its filesystem layout.
#[derive(Clone)]
pub struct RemoteWorkspaceAdapter {
    /// Node that owns the worktrees.
    node_id: String,
    /// Transport used to reach the node.
    dispatch: Arc<dyn RemoteDispatch>,
    /// Git remote URL sent with `create` requests, when the caller knows it.
    repo_url: Option<String>,
    /// Branch `create` requests base on, when the caller pins one.
    base_branch: Option<String>,
}

impl std::fmt::Debug for RemoteWorkspaceAdapter {
    /// Prints the node, the dispatch name and the pinned request defaults; the dispatch itself is an
    /// opaque trait object.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RemoteWorkspaceAdapter")
            .field("node_id", &self.node_id)
            .field("dispatch", &self.dispatch.name())
            .field("repo_url", &self.repo_url)
            .field("base_branch", &self.base_branch)
            .finish()
    }
}

impl RemoteWorkspaceAdapter {
    /// Creates an adapter for `node_id` over `dispatch`.
    #[must_use]
    pub fn new(node_id: impl Into<String>, dispatch: Arc<dyn RemoteDispatch>) -> Self {
        Self {
            node_id: node_id.into(),
            dispatch,
            repo_url: None,
            base_branch: None,
        }
    }

    /// Creates an adapter whose every request fails with [`Error::Unsupported`].
    ///
    /// Used when a team declares remote worktrees but the process runs without a distributed
    /// transport.
    #[must_use]
    pub fn unavailable(node_id: impl Into<String>) -> Self {
        Self::new(node_id, Arc::new(UnavailableRemoteDispatch))
    }

    /// Sets the git remote URL sent with `create` requests.
    #[must_use]
    pub fn with_repo_url(mut self, repo_url: impl Into<String>) -> Self {
        self.repo_url = Some(repo_url.into());
        self
    }

    /// Sets the branch `create` requests base on.
    #[must_use]
    pub fn with_base_branch(mut self, base_branch: impl Into<String>) -> Self {
        self.base_branch = Some(base_branch.into());
        self
    }

    /// Returns the node the adapter drives.
    #[must_use]
    pub fn node_id(&self) -> &str {
        &self.node_id
    }

    /// Asks the node whether `worktree_path` exists there.
    ///
    /// # Errors
    ///
    /// Returns the transport error when the node cannot be reached, and [`Error::SysOp`] when the
    /// node reports failure.
    pub async fn exists(&self, worktree_path: &Path) -> Result<bool> {
        let request = self.prepare(RemoteWorktreeRequest::exists(path_string(worktree_path)?))?;
        let response = self.send(&request).await?;
        Ok(response.exists)
    }

    /// Fills in the adapter-level defaults of `request` and validates it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the request cannot be served (see
    /// [`RemoteWorktreeRequest::validate`]).
    fn prepare(&self, mut request: RemoteWorktreeRequest) -> Result<RemoteWorktreeRequest> {
        if matches!(request.action, RemoteWorktreeAction::Create) {
            if request.repo_url.is_none() {
                request.repo_url.clone_from(&self.repo_url);
            }
            if request.base_branch.is_none() {
                request.base_branch.clone_from(&self.base_branch);
            }
        }
        request.validate()?;
        Ok(request)
    }

    /// Serializes `request`, sends it to the node and parses the reply.
    ///
    /// # Errors
    ///
    /// Returns the transport error, [`Error::SysOp`] when the request cannot be encoded or the reply
    /// cannot be decoded, and [`Error::SysOp`] carrying the node's own failure detail.
    async fn send(&self, request: &RemoteWorktreeRequest) -> Result<RemoteWorktreeResponse> {
        let payload = serde_json::to_value(request)
            .map_err(|error| Error::sys_op(format!("encoding remote worktree request: {error}")))?;
        debug!(
            node = %self.node_id,
            dispatch = self.dispatch.name(),
            action = request.action.as_str(),
            "sending remote worktree request"
        );
        let reply = self.dispatch.send_and_wait(&self.node_id, payload).await?;
        let response: RemoteWorktreeResponse = serde_json::from_value(reply).map_err(|error| {
            Error::sys_op(format!(
                "decoding the remote worktree reply of node {}: {error}",
                self.node_id
            ))
        })?;
        response.into_success(&self.node_id, request.action)
    }
}

#[async_trait]
impl WorktreeBackend for RemoteWorkspaceAdapter {
    fn name(&self) -> &'static str {
        "remote"
    }

    async fn create(&self, request: &WorktreeCreateRequest) -> Result<WorktreeInfo> {
        let prepared = self.prepare(RemoteWorktreeRequest::create(
            request.name.clone(),
            request.base_ref.clone(),
        ))?;
        let response = self.send(&prepared).await?;
        let Some(path) = response.worktree_path.as_deref() else {
            return Err(Error::sys_op(format!(
                "remote node {} created worktree {} without reporting its path",
                self.node_id, request.name
            )));
        };
        if response.existed {
            debug!(
                node = %self.node_id,
                worktree = %request.name,
                "remote node reused an existing worktree"
            );
        }
        let branch = response
            .worktree_branch
            .clone()
            .unwrap_or_else(|| super::worktree::worktree_branch(&request.name));
        Ok(WorktreeInfo {
            name: request.name.clone(),
            path: PathBuf::from(path),
            branch,
        })
    }

    async fn remove(&self, worktree: &WorktreeInfo) -> Result<()> {
        let request = self.prepare(RemoteWorktreeRequest::remove(path_string(&worktree.path)?))?;
        self.send(&request).await?;
        debug!(
            node = %self.node_id,
            worktree = %worktree.name,
            "remote node removed the worktree"
        );
        Ok(())
    }
}

/// Renders a path for a JSON payload, refusing non-UTF-8 paths.
fn path_string(path: &Path) -> Result<String> {
    path.to_str().map(str::to_owned).ok_or_else(|| {
        Error::validation(format!(
            "worktree path {} is not valid UTF-8 and cannot be sent to a remote node",
            path.display()
        ))
    })
}
