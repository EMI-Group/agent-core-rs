//! `enter_worktree`/`exit_worktree` scenarios: card shape, argument validation,
//! session bookkeeping, recovery, the two-phase removal confirmation, and the
//! permission refusal.
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
    EnterWorktreeTool, ExitWorktreeTool, MemoryWorktreeSessionStore, RecordingWorktreeEvents,
    WorktreeEvent, WorktreeIndex, WorktreeManager, WorktreeRecord, WorktreeSessionState,
    WorktreeSessionStore, ENTER_TOOL_NAME, EXIT_TOOL_NAME,
};

/// A manager over a scripted runner sharing a recording sink and a session store.
struct Fixture {
    manager: Arc<WorktreeManager>,
    runner: Arc<ScriptedGitRunner>,
    store: Arc<MemoryWorktreeSessionStore>,
    events: Arc<RecordingWorktreeEvents>,
}

impl Fixture {
    fn new(repo: &Path) -> Self {
        let runner = Arc::new(ScriptedGitRunner::new());
        let events = Arc::new(RecordingWorktreeEvents::new());
        let manager = Arc::new(
            WorktreeManager::with_runner(repo, runner.clone()).with_events(events.clone()),
        );
        Self {
            manager,
            runner,
            store: Arc::new(MemoryWorktreeSessionStore::new()),
            events,
        }
    }

    fn enter(&self, language: &str) -> EnterWorktreeTool {
        EnterWorktreeTool::new(self.manager.clone(), language).with_sessions(self.store.clone())
    }

    fn exit(&self, language: &str) -> ExitWorktreeTool {
        ExitWorktreeTool::new(self.manager.clone(), language).with_sessions(self.store.clone())
    }
}

/// An engine whose `write_file` baseline is `deny`.
fn deny_write_engine() -> Arc<PermissionEngine> {
    let section = PermissionsSection {
        package_builtin_rules: false,
        tools: BTreeMap::from([("write_file".to_owned(), PermissionLevel::Deny)]),
        ..PermissionsSection::default()
    };
    Arc::new(PermissionEngine::new(PermissionEngineConfig::new(section)))
}

/// The stdout of `rev-parse --show-toplevel` for `repo`.
fn root(repo: &Path) -> String {
    repo.to_string_lossy().into_owned()
}

/// A session for the worktree `name` of `repo`.
fn session(repo: &Path, name: &str) -> WorktreeSessionState {
    let mut session = WorktreeSessionState::new(
        name,
        repo.join(".worktrees").join(name),
        Some(format!("worktree-{name}")),
    );
    session.original_cwd = repo.to_path_buf();
    session
}

/// Saves one registry record for `repo`.
fn store_record(repo: &Path, record: WorktreeRecord) {
    let path = crate::worktree::index_path(repo);
    let mut index = WorktreeIndex::load(&path).unwrap();
    index.upsert(record);
    index.save(&path).unwrap();
}

/// Invokes `enter_worktree` and unwraps the payload envelope.
async fn invoke_enter(tool: &EnterWorktreeTool, inputs: Value) -> ToolPayload {
    let value = tool.invoke(inputs, &ToolContext::new()).await.unwrap();
    serde_json::from_value(value).unwrap()
}

/// Invokes `exit_worktree` and unwraps the payload envelope.
async fn invoke_exit(tool: &ExitWorktreeTool, inputs: Value) -> ToolPayload {
    let value = tool.invoke(inputs, &ToolContext::new()).await.unwrap();
    serde_json::from_value(value).unwrap()
}

#[test]
fn enter_card_is_named_enter_worktree_with_an_optional_name() {
    let card = EnterWorktreeTool::card_for("en");
    assert_eq!(card.name, ENTER_TOOL_NAME);
    assert_eq!(card.name, "enter_worktree");
    assert!(!card.parallel_safe, "entering mutates the repository");
    assert!(card.description.to_lowercase().contains("worktree"));
    assert!(card.description.contains(".worktrees/<name>"));
    assert_ne!(
        EnterWorktreeTool::card_for("cn").description,
        card.description
    );

    let properties = card.input_params["properties"].as_object().unwrap();
    let keys: Vec<&str> = properties.keys().map(String::as_str).collect();
    assert_eq!(keys, ["name"]);
    let required = card.input_params.get("required").and_then(Value::as_array);
    assert!(required.is_none_or(Vec::is_empty), "name must be optional");
}

#[test]
fn exit_card_requires_action_and_takes_discard_changes() {
    let card = ExitWorktreeTool::card_for("en");
    assert_eq!(card.name, EXIT_TOOL_NAME);
    assert_eq!(card.name, "exit_worktree");
    assert!(!card.parallel_safe);
    assert_ne!(
        ExitWorktreeTool::card_for("cn").description,
        card.description
    );

    let properties = card.input_params["properties"].as_object().unwrap();
    let mut keys: Vec<&str> = properties.keys().map(String::as_str).collect();
    keys.sort_unstable();
    assert_eq!(keys, ["action", "discard_changes"]);
    assert_eq!(properties["discard_changes"]["type"], json!("boolean"));
    assert_eq!(card.input_params["required"], json!(["action"]));
}

#[tokio::test]
async fn enter_rejects_unknown_fields_and_missing_names_before_git() {
    let (_dir, repo) = temp_root();
    let fixture = Fixture::new(&repo);
    let tool = fixture.enter("en");

    let unknown = invoke_enter(&tool, json!({"name": "feature", "action": "create"})).await;
    assert!(!unknown.success);
    assert!(unknown
        .error
        .as_deref()
        .unwrap()
        .starts_with("invalid arguments:"));

    let absent = invoke_enter(&tool, json!({})).await;
    assert!(!absent.success);
    assert_eq!(
        absent.error.as_deref(),
        Some("name is required when no worktree session or default name is recorded")
    );

    let blank = invoke_enter(&tool, json!({"name": "   "})).await;
    assert!(!blank.success);
    assert_eq!(
        blank.error.as_deref(),
        Some("name is required when no worktree session or default name is recorded")
    );
    assert_eq!(
        fixture.runner.calls().len(),
        0,
        "validation runs before git"
    );
}

#[tokio::test]
async fn enter_refuses_a_traversal_name_without_running_git() {
    let (_dir, repo) = temp_root();
    let fixture = Fixture::new(&repo);
    let tool = fixture.enter("en");

    let payload = invoke_enter(&tool, json!({"name": "../escape"})).await;
    assert!(!payload.success);
    assert!(payload
        .error
        .as_deref()
        .unwrap()
        .contains("`..` path segments are not allowed"));
    assert_eq!(fixture.runner.calls().len(), 0);
}

#[tokio::test]
async fn enter_refuses_when_a_session_is_already_active() {
    let (_dir, repo) = temp_root();
    let fixture = Fixture::new(&repo);
    fixture.store.set_session(Some(session(&repo, "old")));
    let tool = fixture.enter("en");

    let payload = invoke_enter(&tool, json!({"name": "feature"})).await;
    assert!(!payload.success);
    assert_eq!(
        payload.error.as_deref(),
        Some("Already in worktree 'old'. Exit first with exit_worktree.")
    );
    assert_eq!(fixture.runner.calls().len(), 0);
}

#[tokio::test]
async fn enter_creates_the_worktree_and_records_the_session() {
    let (_dir, repo) = temp_root();
    let fixture = Fixture::new(&repo);
    let target = fixture.manager.target_path("feature").unwrap();
    fixture.runner.push_ok(&root(&repo));
    fixture.runner.push_ok(&listing(&repo, None));
    fixture.runner.push_ok(&root(&repo));
    fixture.runner.push_ok(&listing(&repo, None));
    fixture.runner.push_fail("fatal: Needed a single revision");
    fixture.runner.push_success();
    fixture
        .runner
        .push_ok("cccccccccccccccccccccccccccccccccccccccc");
    let tool = fixture.enter("en");

    let payload = invoke_enter(&tool, json!({"name": "feature"})).await;
    assert!(payload.success, "{:?}", payload.error);
    let data = payload.data.unwrap();
    assert_eq!(data["worktree_path"], target.to_string_lossy().as_ref());
    assert_eq!(data["worktree_branch"], "worktree-feature");
    assert_eq!(data["existed"], false);
    assert_eq!(data["sparse"], Value::Null);
    assert_eq!(
        data["message"],
        format!(
            "Created worktree at {} on branch worktree-feature. CWD switched to worktree.",
            target.display()
        )
    );

    let stored = fixture.store.session().expect("session recorded");
    assert_eq!(stored.worktree_name, "feature");
    assert_eq!(stored.worktree_path, target);
    assert_eq!(stored.worktree_branch.as_deref(), Some("worktree-feature"));
    assert_eq!(
        stored.original_head_commit.as_deref(),
        Some("cccccccccccccccccccccccccccccccccccccccc")
    );
    assert!(!stored.used_sparse_paths);
    assert_eq!(stored.original_cwd, repo);
    assert_eq!(fixture.store.default_name().as_deref(), Some("feature"));
    assert_eq!(
        fixture.events.events(),
        [
            WorktreeEvent::Created {
                name: "feature".to_owned(),
                path: target.clone(),
                branch: "worktree-feature".to_owned(),
                existed: false,
                owner: None,
            },
            WorktreeEvent::Entered {
                name: "feature".to_owned(),
                path: target,
                branch: "worktree-feature".to_owned(),
                existed: false,
            },
        ]
    );
}

#[tokio::test]
async fn enter_recovers_a_worktree_git_already_lists() {
    let (_dir, repo) = temp_root();
    let fixture = Fixture::new(&repo);
    let target = fixture.manager.target_path("feature").unwrap();
    fixture.runner.push_ok(&root(&repo));
    fixture
        .runner
        .push_ok(&listing(&repo, Some((&target, "worktree-feature"))));
    fixture.runner.push_ok(&root(&repo));
    fixture
        .runner
        .push_ok(&listing(&repo, Some((&target, "worktree-feature"))));
    let tool = fixture.enter("en");

    let payload = invoke_enter(&tool, json!({"name": "feature"})).await;
    assert!(payload.success, "{:?}", payload.error);
    let data = payload.data.unwrap();
    assert_eq!(data["existed"], true);
    assert_eq!(
        data["message"],
        format!(
            "Entered existing worktree at {} on branch worktree-feature. CWD switched to worktree.",
            target.display()
        )
    );
    assert_eq!(fixture.store.session().unwrap().worktree_path, target);
    assert!(fixture
        .runner
        .commands()
        .iter()
        .all(|command| !command.starts_with("worktree add")));
}

#[tokio::test]
async fn enter_refuses_a_registry_record_git_does_not_list() {
    let (_dir, repo) = temp_root();
    let fixture = Fixture::new(&repo);
    let target = fixture.manager.target_path("feature").unwrap();
    store_record(
        &repo,
        WorktreeRecord::new("feature", target.clone()).with_branch("worktree-feature"),
    );
    fixture.runner.push_ok(&root(&repo));
    fixture.runner.push_ok(&listing(&repo, None));
    let tool = fixture.enter("en");

    let payload = invoke_enter(&tool, json!({"name": "feature"})).await;
    assert!(!payload.success);
    let error = payload.error.unwrap();
    assert!(error.contains("is not live"), "{error}");
    assert!(error.contains("prune"), "{error}");
    assert!(fixture
        .runner
        .commands()
        .iter()
        .all(|command| !command.starts_with("worktree add")));
}

#[tokio::test]
async fn an_omitted_name_reenters_the_recorded_default_name() {
    let (_dir, repo) = temp_root();
    let fixture = Fixture::new(&repo);
    fixture.store.set_default_name(Some("feature".to_owned()));
    let target = fixture.manager.target_path("feature").unwrap();
    fixture.runner.push_ok(&root(&repo));
    fixture
        .runner
        .push_ok(&listing(&repo, Some((&target, "worktree-feature"))));
    fixture.runner.push_ok(&root(&repo));
    fixture
        .runner
        .push_ok(&listing(&repo, Some((&target, "worktree-feature"))));
    let tool = fixture.enter("en");

    let payload = invoke_enter(&tool, json!({})).await;
    assert!(payload.success, "{:?}", payload.error);
    assert_eq!(payload.data.unwrap()["existed"], true);
    assert_eq!(fixture.store.session().unwrap().worktree_name, "feature");
}

#[tokio::test]
async fn enter_short_circuits_on_a_deny_permission() {
    let (_dir, repo) = temp_root();
    let fixture = Fixture::new(&repo);
    let tool = fixture.enter("en").with_permission(deny_write_engine());

    let payload = invoke_enter(&tool, json!({"name": "feature"})).await;
    assert!(!payload.success);
    assert!(payload.error.is_some());
    assert_eq!(
        fixture.runner.calls().len(),
        0,
        "deny must short-circuit git"
    );
    assert!(fixture.store.session().is_none());
}

#[tokio::test]
async fn exit_without_a_session_is_a_failure_payload() {
    let (_dir, repo) = temp_root();
    let fixture = Fixture::new(&repo);
    let tool = fixture.exit("en");

    let payload = invoke_exit(&tool, json!({"action": "keep"})).await;
    assert!(!payload.success);
    assert_eq!(
        payload.error.as_deref(),
        Some("No active worktree session to exit.")
    );
}

#[tokio::test]
async fn exit_rejects_an_unknown_action() {
    let (_dir, repo) = temp_root();
    let fixture = Fixture::new(&repo);
    fixture.store.set_session(Some(session(&repo, "feature")));
    let tool = fixture.exit("en");

    let payload = invoke_exit(&tool, json!({"action": "drop"})).await;
    assert!(!payload.success);
    assert_eq!(
        payload.error.as_deref(),
        Some("'action' must be 'keep' or 'remove'.")
    );
    assert_eq!(fixture.runner.calls().len(), 0);
}

#[tokio::test]
async fn exit_keep_clears_the_session_and_keeps_the_default_name() {
    let (_dir, repo) = temp_root();
    let fixture = Fixture::new(&repo);
    let session = session(&repo, "feature");
    let worktree_path = session.worktree_path.clone();
    fixture.store.set_session(Some(session));
    fixture.store.set_default_name(Some("feature".to_owned()));
    let tool = fixture.exit("en");

    let payload = invoke_exit(&tool, json!({"action": "keep"})).await;
    assert!(payload.success, "{:?}", payload.error);
    let data = payload.data.unwrap();
    assert_eq!(data["action"], "keep");
    assert_eq!(data["worktree_name"], "feature");
    assert_eq!(data["worktree_branch"], "worktree-feature");
    assert_eq!(data["original_cwd"], repo.to_string_lossy().as_ref());
    assert_eq!(
        data["worktree_path"],
        worktree_path.to_string_lossy().as_ref()
    );
    assert_eq!(
        data["message"],
        format!(
            "Kept worktree 'feature' (branch worktree-feature). In this session, enter_worktree without a name will re-enter it; from another session, pass name='feature'. Returned to {}",
            repo.display()
        )
    );

    assert!(fixture.store.session().is_none());
    assert_eq!(fixture.store.default_name().as_deref(), Some("feature"));
    assert_eq!(
        fixture.events.events(),
        [WorktreeEvent::Exited {
            name: "feature".to_owned(),
            path: worktree_path,
            action: "keep".to_owned(),
        }]
    );
}

#[tokio::test]
async fn exit_remove_refuses_dirty_work_without_discard_changes() {
    let (_dir, repo) = temp_root();
    let fixture = Fixture::new(&repo);
    let session = session(&repo, "feature");
    let worktree_path = session.worktree_path.clone();
    fixture.store.set_session(Some(session));
    store_record(
        &repo,
        WorktreeRecord::new("feature", worktree_path.clone())
            .with_branch("worktree-feature")
            .with_head("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
    );
    fixture.runner.push_ok(&root(&repo));
    fixture
        .runner
        .push_ok(&listing(&repo, Some((&worktree_path, "worktree-feature"))));
    fixture.runner.push_ok(" M src/a.rs\n M src/b.rs\n");
    fixture.runner.push_ok("");
    fixture.runner.push_ok("2");
    fixture.runner.push_ok("");
    let tool = fixture.exit("en");

    let payload = invoke_exit(&tool, json!({"action": "remove"})).await;
    assert!(!payload.success);
    let error = payload.error.unwrap();
    assert!(
        error.starts_with("Worktree has 2 uncommitted files and 2 commits on worktree-feature."),
        "{error}"
    );
    assert!(error.contains("set discard_changes=True"), "{error}");
    assert!(fixture
        .runner
        .commands()
        .iter()
        .all(|command| !command.starts_with("worktree remove")));
    assert!(fixture.store.session().is_some(), "session must survive");
}

#[tokio::test]
async fn exit_remove_with_discard_changes_forces_the_removal() {
    let (_dir, repo) = temp_root();
    let fixture = Fixture::new(&repo);
    let session = session(&repo, "feature");
    let worktree_path = session.worktree_path.clone();
    fixture.store.set_session(Some(session));
    store_record(
        &repo,
        WorktreeRecord::new("feature", worktree_path.clone())
            .with_branch("worktree-feature")
            .with_head("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
    );
    fixture.runner.push_ok(&root(&repo));
    fixture
        .runner
        .push_ok(&listing(&repo, Some((&worktree_path, "worktree-feature"))));
    fixture.runner.push_ok(" M src/a.rs\n M src/b.rs\n");
    fixture.runner.push_ok("");
    fixture.runner.push_ok("2");
    fixture.runner.push_ok("");
    fixture.runner.push_ok(&root(&repo));
    fixture
        .runner
        .push_ok(&listing(&repo, Some((&worktree_path, "worktree-feature"))));
    fixture.runner.push_success();
    let tool = fixture.exit("en");

    let payload = invoke_exit(&tool, json!({"action": "remove", "discard_changes": true})).await;
    assert!(payload.success, "{:?}", payload.error);
    let data = payload.data.unwrap();
    assert_eq!(data["action"], "remove");
    assert_eq!(data["discarded_files"], 2);
    assert_eq!(data["discarded_commits"], 2);
    assert_eq!(
        data["message"],
        format!(
            "Removed worktree 'feature' (branch worktree-feature). Returned to {}",
            repo.display()
        )
    );
    assert_eq!(
        fixture.runner.last_call().unwrap().command(),
        format!("worktree remove --force {}", worktree_path.display())
    );
    assert!(fixture.store.session().is_none());
    let events = fixture.events.events();
    assert!(events.contains(&WorktreeEvent::Removed {
        name: "feature".to_owned(),
        path: worktree_path.clone(),
        forced: true,
    }));
    assert_eq!(
        events.last(),
        Some(&WorktreeEvent::Exited {
            name: "feature".to_owned(),
            path: worktree_path,
            action: "remove".to_owned(),
        })
    );
}

#[tokio::test]
async fn exit_remove_on_a_clean_worktree_does_not_force() {
    let (_dir, repo) = temp_root();
    let fixture = Fixture::new(&repo);
    let session = session(&repo, "feature");
    let worktree_path = session.worktree_path.clone();
    fixture.store.set_session(Some(session));
    fixture.runner.push_ok(&root(&repo));
    fixture
        .runner
        .push_ok(&listing(&repo, Some((&worktree_path, "worktree-feature"))));
    fixture.runner.push_ok("");
    fixture.runner.push_ok("");
    fixture.runner.push_ok(&root(&repo));
    fixture
        .runner
        .push_ok(&listing(&repo, Some((&worktree_path, "worktree-feature"))));
    fixture.runner.push_ok("");
    fixture.runner.push_success();
    let tool = fixture.exit("en");

    let payload = invoke_exit(&tool, json!({"action": "remove"})).await;
    assert!(payload.success, "{:?}", payload.error);
    let data = payload.data.unwrap();
    assert_eq!(data["discarded_files"], 0);
    assert_eq!(data["discarded_commits"], 0);
    assert_eq!(
        fixture.runner.last_call().unwrap().command(),
        format!("worktree remove {}", worktree_path.display())
    );
    assert!(fixture.store.session().is_none());
}

#[tokio::test]
async fn exit_remove_fails_closed_when_the_state_cannot_be_verified() {
    let (_dir, repo) = temp_root();
    let fixture = Fixture::new(&repo);
    let session = session(&repo, "feature");
    fixture.store.set_session(Some(session));
    fixture.runner.push_ok(&root(&repo));
    fixture.runner.push_ok(&listing(&repo, None));
    let tool = fixture.exit("en");

    let payload = invoke_exit(&tool, json!({"action": "remove", "discard_changes": true})).await;
    assert!(!payload.success);
    let error = payload.error.unwrap();
    assert!(
        error.starts_with("Could not verify worktree state at "),
        "{error}"
    );
    assert!(error.contains("action='keep'"), "{error}");
    assert!(fixture.store.session().is_some());
}
