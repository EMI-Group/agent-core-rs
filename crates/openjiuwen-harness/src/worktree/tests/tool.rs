//! `WorktreeTool` scenarios: card shape, action dispatch, expected failures as
//! payloads, permission refusal, and the missing-git path.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use openjiuwen_foundation::tool::{Tool, ToolContext};
use serde_json::{json, Value};

use super::{listing, temp_root, ScriptedGitRunner};
use crate::permission::{
    PermissionEngine, PermissionEngineConfig, PermissionLevel, PermissionsSection,
};
use crate::tools::ToolPayload;
use crate::worktree::{
    ProcessGitRunner, WorktreeManager, WorktreeTool, ACTIONS, NOT_INSTALLED_MARKER, TOOL_NAME,
};

/// An engine whose `write_file` baseline is `deny`.
fn deny_write_engine() -> Arc<PermissionEngine> {
    let section = PermissionsSection {
        package_builtin_rules: false,
        tools: BTreeMap::from([("write_file".to_owned(), PermissionLevel::Deny)]),
        ..PermissionsSection::default()
    };
    Arc::new(PermissionEngine::new(PermissionEngineConfig::new(section)))
}

/// Builds a scripted manager rooted at a temp directory.
fn scripted(repo: &Path) -> (Arc<WorktreeManager>, Arc<ScriptedGitRunner>) {
    let runner = Arc::new(ScriptedGitRunner::new());
    let manager = Arc::new(WorktreeManager::with_runner(repo, runner.clone()));
    (manager, runner)
}

/// Invokes the tool and unwraps the payload envelope.
async fn invoke(tool: &WorktreeTool, inputs: Value) -> ToolPayload {
    let value = tool.invoke(inputs, &ToolContext::new()).await.unwrap();
    serde_json::from_value(value).unwrap()
}

#[test]
fn card_is_named_worktree_and_exposes_the_action_arguments() {
    let card = WorktreeTool::card_for("en");
    assert_eq!(card.name, TOOL_NAME);
    assert_eq!(card.name, "worktree");
    assert!(
        !card.parallel_safe,
        "worktree operations mutate the repository"
    );
    assert!(card.description.contains("create"));
    assert_ne!(WorktreeTool::card_for("cn").description, card.description);

    let properties = card.input_params["properties"].as_object().unwrap();
    let mut keys: Vec<&str> = properties.keys().map(String::as_str).collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        ["action", "base", "branch", "force", "name", "reason"]
    );
    assert_eq!(properties["force"]["type"], json!("boolean"));
    assert_eq!(card.input_params["required"], json!(["action"]));
    assert_eq!(ACTIONS.len(), 7);
}

#[tokio::test]
async fn create_action_returns_the_created_worktree() {
    let (_dir, repo) = temp_root();
    let (manager, runner) = scripted(&repo);
    let target = manager.target_path("feature").unwrap();
    runner.push_ok(&repo.to_string_lossy());
    runner.push_ok(&listing(&repo, None));
    runner.push_fail("no branch");
    runner.push_success();
    runner.push_ok("ffff");
    let tool = WorktreeTool::new(manager, "en");

    let payload = invoke(&tool, json!({"action": "create", "name": "feature"})).await;
    assert!(payload.success, "{:?}", payload.error);
    let data = payload.data.unwrap();
    assert_eq!(data["action"], "create");
    assert_eq!(data["worktree"]["path"], target.to_string_lossy().as_ref());
    assert_eq!(data["worktree"]["branch"], "worktree-feature");
    assert_eq!(data["worktree"]["existed"], false);
}

#[tokio::test]
async fn list_action_serializes_git_entries() {
    let (_dir, repo) = temp_root();
    let (manager, runner) = scripted(&repo);
    runner.push_ok(&repo.to_string_lossy());
    runner.push_ok(&listing(&repo, None));
    let tool = WorktreeTool::new(manager, "en");

    let payload = invoke(&tool, json!({"action": "list"})).await;
    assert!(payload.success, "{:?}", payload.error);
    let worktrees = payload.data.unwrap()["worktrees"].clone();
    assert_eq!(worktrees.as_array().unwrap().len(), 1);
    assert_eq!(worktrees[0]["branch"], "main");
}

#[tokio::test]
async fn expected_failures_are_payloads_not_errors() {
    let (_dir, repo) = temp_root();
    let (manager, _runner) = scripted(&repo);
    let tool = WorktreeTool::new(manager, "en");

    let unknown = invoke(&tool, json!({"action": "explode"})).await;
    assert!(!unknown.success);
    assert!(unknown
        .error
        .unwrap()
        .contains("unknown worktree action `explode`"));

    let missing = invoke(&tool, json!({"action": "create"})).await;
    assert!(!missing.success);
    assert!(missing.error.unwrap().contains("name is required"));

    let invalid = invoke(&tool, json!({"action": 5})).await;
    assert!(!invalid.success);
    assert!(invalid.error.unwrap().contains("invalid arguments:"));

    let escape = invoke(&tool, json!({"action": "remove", "name": "../escape"})).await;
    assert!(!escape.success);
    assert!(escape.error.unwrap().contains("not allowed"));
}

#[tokio::test]
async fn remove_action_forwards_force() {
    let (_dir, repo) = temp_root();
    let (manager, runner) = scripted(&repo);
    let target = manager.target_path("feature").unwrap();
    runner.push_ok(&repo.to_string_lossy());
    runner.push_ok(&listing(&repo, Some((&target, "worktree-feature"))));
    runner.push_success();
    let tool = WorktreeTool::new(manager, "en");

    let payload = invoke(
        &tool,
        json!({"action": "remove", "name": "feature", "force": true}),
    )
    .await;
    assert!(payload.success, "{:?}", payload.error);
    let data = payload.data.unwrap();
    assert_eq!(data["action"], "remove");
    assert_eq!(data["path"], target.to_string_lossy().as_ref());
    assert_eq!(
        runner.commands()[2],
        format!("worktree remove --force {}", target.display())
    );
}

#[tokio::test]
async fn lock_status_and_prune_actions_dispatch_to_the_manager() {
    let (_dir, repo) = temp_root();
    let (manager, runner) = scripted(&repo);
    let target = manager.target_path("feature").unwrap();
    runner.push_ok(&repo.to_string_lossy());
    runner.push_ok(&listing(&repo, Some((&target, "worktree-feature"))));
    runner.push_success();
    runner.push_ok(&repo.to_string_lossy());
    runner.push_ok(&listing(&repo, Some((&target, "worktree-feature"))));
    runner.push_ok("");
    runner.push_ok(&repo.to_string_lossy());
    runner.push_success();
    runner.push_ok(&listing(&repo, None));
    let tool = WorktreeTool::new(manager, "en");

    let locked = invoke(
        &tool,
        json!({"action": "lock", "name": "feature", "reason": "busy"}),
    )
    .await;
    assert!(locked.success, "{:?}", locked.error);
    assert_eq!(locked.data.unwrap()["reason"], "busy");

    let status = invoke(&tool, json!({"action": "status", "name": "feature"})).await;
    assert!(status.success, "{:?}", status.error);
    assert_eq!(status.data.unwrap()["worktree"]["dirty"], false);

    let pruned = invoke(&tool, json!({"action": "prune"})).await;
    assert!(pruned.success, "{:?}", pruned.error);
    assert_eq!(pruned.data.unwrap()["removed"], 0);
}

#[tokio::test]
async fn unlock_action_reports_success() {
    let (_dir, repo) = temp_root();
    let (manager, runner) = scripted(&repo);
    let target = manager.target_path("feature").unwrap();
    runner.push_ok(&repo.to_string_lossy());
    runner.push_ok(&listing(&repo, Some((&target, "worktree-feature"))));
    runner.push_success();
    let tool = WorktreeTool::new(manager, "en");

    let payload = invoke(&tool, json!({"action": "unlock", "name": "feature"})).await;
    assert!(payload.success, "{:?}", payload.error);
    assert_eq!(payload.data.unwrap()["action"], "unlock");
}

#[tokio::test]
async fn a_deny_permission_refuses_before_git_runs() {
    let (_dir, repo) = temp_root();
    let (manager, runner) = scripted(&repo);
    let tool = WorktreeTool::new(manager, "en").with_permission(deny_write_engine());

    for action in ["create", "remove", "lock", "unlock"] {
        let payload = invoke(&tool, json!({"action": action, "name": "feature"})).await;
        assert!(!payload.success, "{action} must be refused");
        assert!(payload.error.is_some());
    }
    assert_eq!(runner.calls().len(), 0, "deny must short-circuit git");
}

#[tokio::test]
async fn a_missing_git_binary_is_a_failure_payload() {
    let (_dir, repo) = temp_root();
    let runner = Arc::new(ProcessGitRunner::with_program("openjiuwen-no-such-git"));
    let manager = Arc::new(WorktreeManager::with_runner(&repo, runner));
    let tool = WorktreeTool::new(manager, "en");

    let payload = invoke(&tool, json!({"action": "create", "name": "feature"})).await;
    assert!(!payload.success);
    assert!(payload.error.unwrap().contains(NOT_INSTALLED_MARKER));
}

#[test]
fn worktree_args_defaults_are_action_specific() {
    let args = crate::worktree::WorktreeArgs::new("list");
    assert!(args.is_known_action());
    assert!(args.name.is_none());
    assert!(!args.force);
    assert!(!crate::worktree::WorktreeArgs::new("nope").is_known_action());

    let args = crate::worktree::WorktreeArgs::new("create").with_name("feature");
    assert_eq!(args.name.as_deref(), Some("feature"));
}

#[test]
fn tool_payload_is_always_the_six_key_envelope() {
    // Guards the documented contract that a failure still renders the full
    // `ToolOutput` shape (all six keys, `null` for unset optionals).
    let payload = ToolPayload::fail("boom").to_json();
    let object: BTreeMap<String, Value> = serde_json::from_value(payload).unwrap();
    assert_eq!(object.len(), 6);
    assert_eq!(object["success"], json!(false));
}
