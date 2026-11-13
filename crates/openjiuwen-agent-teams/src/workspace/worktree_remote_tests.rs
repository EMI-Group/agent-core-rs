//! Tests of the remote worktree adapter: wire types, request validation, the dispatch seam and the
//! fail-closed transport.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::Mutex;
use serde_json::{json, Value};

use super::worktree::{WorktreeBackend, WorktreeInfo};
use super::worktree_remote::{
    RemoteDispatch, RemoteWorkspaceAdapter, RemoteWorktreeAction, RemoteWorktreeRequest,
    RemoteWorktreeResponse,
};
use openjiuwen_common::error::{Error, Result};

/// A dispatch that records what it was asked and replays queued replies.
struct RecordingDispatch {
    /// Requests seen, in order, as `(node_id, payload)`.
    sent: Mutex<Vec<(String, Value)>>,
    /// Replies handed out, in order; a default success reply is used when empty.
    replies: Mutex<VecDeque<Value>>,
}

impl RecordingDispatch {
    fn new(replies: Vec<Value>) -> Arc<Self> {
        Arc::new(Self {
            sent: Mutex::new(Vec::new()),
            replies: Mutex::new(replies.into()),
        })
    }

    fn sent(&self) -> Vec<(String, Value)> {
        self.sent.lock().clone()
    }
}

#[async_trait]
impl RemoteDispatch for RecordingDispatch {
    fn name(&self) -> &'static str {
        "recording"
    }

    async fn send_and_wait(&self, node_id: &str, request: Value) -> Result<Value> {
        self.sent.lock().push((node_id.to_owned(), request));
        Ok(self
            .replies
            .lock()
            .pop_front()
            .unwrap_or_else(|| json!({ "success": true })))
    }
}

fn adapter(dispatch: Arc<RecordingDispatch>) -> RemoteWorkspaceAdapter {
    RemoteWorkspaceAdapter::new("node-1", dispatch)
        .with_repo_url("git@example.invalid:team/repo.git")
        .with_base_branch("main")
}

fn create_request(name: &str) -> super::worktree::WorktreeCreateRequest {
    super::worktree::WorktreeCreateRequest {
        team_name: "my-team".to_owned(),
        session_id: "session-1".to_owned(),
        member_name: "dev-1".to_owned(),
        name: name.to_owned(),
        path: PathBuf::from("/local/worktrees").join(name),
        base_ref: None,
    }
}

#[test]
fn request_wire_shape_matches_python() {
    let request = RemoteWorktreeRequest::create("agent-team-dev-1", Some("main".to_owned()));
    let value = serde_json::to_value(&request).unwrap();
    assert_eq!(value["action"], json!("create"));
    assert_eq!(value["slug"], json!("agent-team-dev-1"));
    assert_eq!(value["base_branch"], json!("main"));
    assert_eq!(value["repo_url"], Value::Null);
    assert_eq!(value["worktree_path"], Value::Null);
    assert_eq!(value["config"], Value::Null);

    // Tolerance: unknown fields are ignored and absent fields default.
    let parsed: RemoteWorktreeRequest = serde_json::from_value(
        json!({ "action": "remove", "worktree_path": "/remote/wt", "extra": 1 }),
    )
    .unwrap();
    assert_eq!(parsed.action, RemoteWorktreeAction::Remove);
    assert_eq!(parsed.worktree_path.as_deref(), Some("/remote/wt"));
    assert!(parsed.slug.is_none());
}

#[test]
fn response_defaults_match_python() {
    let response: RemoteWorktreeResponse = serde_json::from_value(json!({})).unwrap();
    assert!(response.success, "success defaults to true");
    assert!(!response.existed);
    assert!(!response.exists);
    assert!(response.error.is_none());

    let value = serde_json::to_value(RemoteWorktreeResponse::failure("boom")).unwrap();
    assert_eq!(value["success"], json!(false));
    assert_eq!(value["error"], json!("boom"));
}

#[test]
fn validate_requires_the_field_each_action_needs() {
    let err = RemoteWorktreeRequest::default()
        .validate()
        .expect_err("a create request needs a slug");
    assert!(matches!(err, Error::Validation(_)), "{err:?}");
    assert!(err.to_string().contains("worktree slug"), "{err}");

    let err = RemoteWorktreeRequest::remove("")
        .validate()
        .expect_err("a remove request needs a path");
    assert!(matches!(err, Error::Validation(_)), "{err:?}");
    assert!(err.to_string().contains("worktree path"), "{err}");

    RemoteWorktreeRequest::create("agent-x", None)
        .validate()
        .expect("slug is enough for create");
    RemoteWorktreeRequest::exists("/remote/wt")
        .validate()
        .expect("path is enough for exists");
}

#[test]
fn remote_failure_becomes_a_sys_op_error() {
    let err = RemoteWorktreeResponse::failure("clone failed")
        .into_success("node-1", RemoteWorktreeAction::Create)
        .expect_err("success=false is an error");
    assert!(matches!(err, Error::SysOp(_)), "{err:?}");
    assert!(err.to_string().contains("node-1"), "{err}");
    assert!(err.to_string().contains("clone failed"), "{err}");
}

#[tokio::test]
async fn adapter_create_sends_the_protocol_request() {
    let dispatch = RecordingDispatch::new(vec![json!({
        "success": true,
        "worktree_path": "/remote/worktrees/agent-team-dev-1",
        "worktree_branch": "worktree-agent-team-dev-1",
        "head_commit": "abc123",
    })]);
    let adapter = adapter(Arc::clone(&dispatch));
    let info = adapter
        .create(&create_request("agent-team-dev-1"))
        .await
        .expect("the node reports a worktree");

    assert_eq!(info.name, "agent-team-dev-1");
    assert_eq!(info.path, Path::new("/remote/worktrees/agent-team-dev-1"));
    assert_eq!(info.branch, "worktree-agent-team-dev-1");

    let sent = dispatch.sent();
    assert_eq!(sent.len(), 1);
    assert_eq!(sent[0].0, "node-1");
    assert_eq!(sent[0].1["action"], json!("create"));
    assert_eq!(sent[0].1["slug"], json!("agent-team-dev-1"));
    assert_eq!(
        sent[0].1["repo_url"],
        json!("git@example.invalid:team/repo.git")
    );
    assert_eq!(sent[0].1["base_branch"], json!("main"));
}

#[tokio::test]
async fn adapter_create_prefers_the_request_base_ref_over_the_default() {
    let dispatch = RecordingDispatch::new(vec![json!({
        "success": true,
        "worktree_path": "/remote/wt",
    })]);
    let adapter = adapter(Arc::clone(&dispatch));
    let mut request = create_request("agent-team-dev-1");
    request.base_ref = Some("v1.2.3".to_owned());
    adapter.create(&request).await.unwrap();
    assert_eq!(dispatch.sent()[0].1["base_branch"], json!("v1.2.3"));
}

#[tokio::test]
async fn adapter_create_falls_back_to_the_local_branch_shape() {
    let dispatch = RecordingDispatch::new(vec![json!({
        "success": true,
        "worktree_path": "/remote/wt",
    })]);
    let adapter = adapter(Arc::clone(&dispatch));
    let info = adapter
        .create(&create_request("agent-team-dev-1"))
        .await
        .unwrap();
    assert_eq!(info.branch, "worktree-agent-team-dev-1");
}

#[tokio::test]
async fn adapter_create_without_a_reported_path_is_an_error() {
    let dispatch = RecordingDispatch::new(vec![json!({ "success": true })]);
    let adapter = adapter(Arc::clone(&dispatch));
    let err = adapter
        .create(&create_request("agent-team-dev-1"))
        .await
        .expect_err("a path is required");
    assert!(matches!(err, Error::SysOp(_)), "{err:?}");
    assert!(
        err.to_string().contains("without reporting its path"),
        "{err}"
    );
}

#[tokio::test]
async fn adapter_remove_and_exists_use_the_matching_actions() {
    let dispatch = RecordingDispatch::new(vec![
        json!({ "success": true }),
        json!({ "success": true, "exists": true }),
    ]);
    let adapter = adapter(Arc::clone(&dispatch));
    let info = WorktreeInfo {
        name: "agent-team-dev-1".to_owned(),
        path: PathBuf::from("/remote/wt"),
        branch: "worktree-agent-team-dev-1".to_owned(),
    };
    adapter.remove(&info).await.expect("the node removed it");
    assert!(adapter.exists(Path::new("/remote/wt")).await.unwrap());

    let sent = dispatch.sent();
    assert_eq!(sent[0].1["action"], json!("remove"));
    assert_eq!(sent[0].1["worktree_path"], json!("/remote/wt"));
    assert_eq!(sent[1].1["action"], json!("exists"));
}

#[tokio::test]
async fn adapter_surfaces_a_remote_failure() {
    let dispatch = RecordingDispatch::new(vec![json!({
        "success": false,
        "error": "cannot find repo root for worktree",
    })]);
    let adapter = adapter(Arc::clone(&dispatch));
    let err = adapter
        .remove(&WorktreeInfo {
            name: "agent-team-dev-1".to_owned(),
            path: PathBuf::from("/remote/wt"),
            branch: "worktree-agent-team-dev-1".to_owned(),
        })
        .await
        .expect_err("the node failed");
    assert!(matches!(err, Error::SysOp(_)), "{err:?}");
    assert!(err.to_string().contains("cannot find repo root"), "{err}");
}

#[tokio::test]
async fn unavailable_adapter_reports_the_deferred_transport() {
    let adapter = RemoteWorkspaceAdapter::unavailable("node-7");
    assert_eq!(adapter.node_id(), "node-7");
    let err = adapter
        .create(&create_request("agent-team-dev-1"))
        .await
        .expect_err("no distributed transport");
    assert!(matches!(err, Error::Unsupported(_)), "{err:?}");
    let message = err.to_string();
    assert!(
        message.contains("distributed Messager transport"),
        "{message}"
    );
    assert!(message.contains("node-7"), "{message}");
    assert!(
        message.contains("agent_teams.worktree_remote.RemoteWorktreeBackend"),
        "{message}"
    );

    let err = adapter.exists(Path::new("/remote/wt")).await.unwrap_err();
    assert!(matches!(err, Error::Unsupported(_)), "{err:?}");
}

#[tokio::test]
async fn default_trait_operations_of_the_adapter_fail_closed() {
    let adapter = RemoteWorkspaceAdapter::unavailable("node-7");
    assert_eq!(adapter.name(), "remote");
    let info = WorktreeInfo {
        name: "agent-team-dev-1".to_owned(),
        path: PathBuf::from("/remote/wt"),
        branch: "worktree-agent-team-dev-1".to_owned(),
    };
    let err = adapter
        .list(Path::new("/remote"))
        .await
        .expect_err("unsupported");
    assert!(matches!(err, Error::Unsupported(_)), "{err:?}");
    let err = adapter
        .lock(&info, "reason")
        .await
        .expect_err("unsupported");
    assert!(matches!(err, Error::Unsupported(_)), "{err:?}");
    let err = adapter.unlock(&info).await.expect_err("unsupported");
    assert!(matches!(err, Error::Unsupported(_)), "{err:?}");
}
