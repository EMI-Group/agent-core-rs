//! Session persistence of the `DeepAgent` state.
//!
//! Covers the Python parity contract for the `deepagent` session payload: the
//! exact five-key `DeepAgentState` shape, plan-mode bookkeeping, pending
//! follow-ups, the documented default for a session without the key, and the
//! todo store that lives beside (never inside) the state.
//!
//! Parity assertions here are intentionally literal: the key sets below are the
//! Python ones (`iteration`, `task_plan`, `stop_condition_state`,
//! `pending_follow_ups`, `plan_mode`; `activeForm` on todo items).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use openjiuwen_agent::{Agent, AgentCard};
use openjiuwen_common::error::Error;
use openjiuwen_foundation::llm::Model;
use openjiuwen_foundation::tool::{Tool, ToolContext};
use openjiuwen_harness::agent::{
    clear_state, load_state, save_state, DeepAgent, DeepAgentBuilder, SESSION_STATE_KEY,
};
use openjiuwen_harness::schema::{
    AgentMode, DeepAgentConfig, DeepAgentState, PlanModeState, TaskPlan, TodoItem, TodoStatus,
};
use openjiuwen_harness::tools::{session_key, TodoCreateTool, TodoStore};
use openjiuwen_harness::workspace::{Workspace, WorkspaceNode};
use openjiuwen_session::{AgentSession, Session};
use openjiuwen_test_support::{ScriptedModel, TempWorkspace};
use serde_json::{json, Value};

/// A two-step plan with a completed and an in-progress task.
fn sample_plan() -> TaskPlan {
    TaskPlan {
        goal: "port the state store".to_owned(),
        tasks: vec![
            TodoItem {
                id: "read".to_owned(),
                content: "read state.rs".to_owned(),
                active_form: "reading state.rs".to_owned(),
                description: "Read the state schema".to_owned(),
                status: TodoStatus::Completed,
                ..TodoItem::default()
            },
            TodoItem {
                id: "write".to_owned(),
                content: "write the suite".to_owned(),
                active_form: "writing the suite".to_owned(),
                description: "Write state_persistence.rs".to_owned(),
                status: TodoStatus::InProgress,
                ..TodoItem::default()
            },
        ],
        current_task_id: Some("write".to_owned()),
    }
}

/// A fully populated state: non-zero iteration, a plan, a queued follow-up, plan mode.
fn sample_state() -> DeepAgentState {
    DeepAgentState {
        iteration: 3,
        task_plan: Some(sample_plan()),
        stop_condition_state: Some(json!({"iteration": 3, "token_usage": 0})),
        pending_follow_ups: vec!["next".to_owned()],
        plan_mode: PlanModeState {
            mode: "plan".to_owned(),
            pre_plan_mode: Some("normal".to_owned()),
            plan_slug: Some("brave-otter".to_owned()),
            prompt_context: None,
        },
    }
}

/// The object keys of `value`, sorted (`serde_json` map order is not guaranteed).
fn keys_sorted(value: &Value) -> Vec<&str> {
    let mut keys: Vec<&str> = value
        .as_object()
        .expect("expected a JSON object")
        .keys()
        .map(String::as_str)
        .collect();
    keys.sort_unstable();
    keys
}

/// A task-loop agent over `workspace` driven by `model`.
fn build_task_loop_agent(model: &ScriptedModel, workspace: &Workspace) -> Arc<DeepAgent> {
    let config = DeepAgentConfig::default()
        .with_language("en".to_owned())
        .with_workspace(workspace.clone())
        .with_enable_task_loop(true);
    DeepAgentBuilder::new(AgentCard::new("state-agent").with_description("State parity agent"))
        .config(config)
        .model(Arc::new(model.clone()) as Arc<dyn Model>)
        .build()
        .expect("agent builds")
}

#[tokio::test]
async fn state_round_trips_through_an_agent_session() {
    let session = AgentSession::new("state-round-trip");
    let state = sample_state();

    save_state(&session, &state).await.expect("save state");
    let loaded = load_state(&session).await.expect("load state");

    assert_eq!(loaded, state, "load(save(state)) == state");
    assert_eq!(loaded.iteration, 3);
    assert_eq!(
        loaded.task_plan.as_ref().map(|plan| plan.tasks.len()),
        Some(2)
    );
    assert_eq!(loaded.pending_follow_ups, vec!["next".to_owned()]);
    assert_eq!(loaded.plan_mode.mode, "plan");
}

#[tokio::test]
async fn persisted_payload_keeps_the_exact_python_key_set() {
    let session = AgentSession::new("state-json-parity");
    let state = sample_state();
    save_state(&session, &state).await.expect("save state");

    // Read the persisted payload back through the real session API: an empty
    // key returns the whole global state lane.
    let global = session
        .get_state("")
        .await
        .expect("read global state")
        .expect("global state present");
    assert_eq!(
        keys_sorted(&global),
        vec![SESSION_STATE_KEY],
        "the payload lives under exactly the `deepagent` key"
    );
    let payload = global
        .get(SESSION_STATE_KEY)
        .expect("payload under `deepagent`");
    assert_eq!(
        keys_sorted(payload),
        vec![
            "iteration",
            "pending_follow_ups",
            "plan_mode",
            "stop_condition_state",
            "task_plan",
        ],
        "the `deepagent` object has exactly Python's five keys"
    );
    assert_eq!(payload["iteration"], json!(3));
    assert_eq!(payload["pending_follow_ups"], json!(["next"]));
    assert_eq!(
        payload["stop_condition_state"],
        json!({"iteration": 3, "token_usage": 0})
    );

    // `plan_mode` is an object with snake_case keys and nulls preserved.
    let plan_mode = &payload["plan_mode"];
    assert_eq!(
        keys_sorted(plan_mode),
        vec!["mode", "plan_slug", "pre_plan_mode", "prompt_context"]
    );
    assert_eq!(plan_mode["mode"], json!("plan"));
    assert_eq!(plan_mode["pre_plan_mode"], json!("normal"));
    assert_eq!(plan_mode["plan_slug"], json!("brave-otter"));
    assert_eq!(plan_mode["prompt_context"], Value::Null);

    // `task_plan` uses Python's item keys, including camel-case `activeForm`.
    let task_plan = &payload["task_plan"];
    assert_eq!(
        keys_sorted(task_plan),
        vec!["current_task_id", "goal", "tasks"]
    );
    assert_eq!(task_plan["goal"], json!("port the state store"));
    assert_eq!(task_plan["current_task_id"], json!("write"));
    assert_eq!(
        keys_sorted(&task_plan["tasks"][0]),
        vec![
            "activeForm",
            "content",
            "depends_on",
            "description",
            "id",
            "meta_data",
            "result_summary",
            "selected_model_id",
            "status",
        ]
    );
    assert_eq!(
        task_plan["tasks"][0]["activeForm"],
        json!("reading state.rs")
    );
    assert_eq!(task_plan["tasks"][0]["status"], json!("completed"));

    // The helpers are the exact inverse of each other, on the persisted payload
    // and directly.
    assert_eq!(DeepAgentState::from_session_dict(Some(payload)), state);
    assert_eq!(
        DeepAgentState::from_session_dict(Some(&state.to_session_dict())),
        state
    );
    assert_eq!(
        PlanModeState::from_session_dict(Some(&state.plan_mode.to_session_dict())),
        state.plan_mode
    );
}

#[tokio::test]
async fn invocation_then_follow_up_and_plan_mode_survive_save_and_load() {
    let temp = TempWorkspace::new().expect("temp workspace");
    let workspace = Workspace::for_language(temp.workspace(), "en");
    let model = ScriptedModel::text("first answer");
    let agent = build_task_loop_agent(&model, &workspace);
    let session: Arc<dyn Session> = Arc::new(AgentSession::new("invoke-session"));

    let output = agent
        .invoke(json!({"query": "start"}), Some(session.clone()))
        .await
        .expect("invocation succeeds");
    assert_eq!(output.output, "first answer");
    assert_eq!(model.call_count(), 1);

    // Queue input and enter plan mode after the run, then persist the live view.
    agent.follow_up("check the tests", None);
    agent.switch_mode(AgentMode::Plan).expect("switch to plan");
    agent.save_state().await.expect("save state");

    let loaded = agent.load_state().await.expect("load state");
    assert_eq!(loaded.iteration, 1, "one round ran in the invocation");
    assert_eq!(loaded.plan_mode.mode, "plan");
    assert_eq!(loaded.plan_mode.pre_plan_mode.as_deref(), Some("normal"));
    assert_eq!(
        loaded.pending_follow_ups,
        vec!["check the tests".to_owned()],
        "the queued follow-up is persisted verbatim"
    );

    // The same payload is visible through the free loader.
    let raw = load_state(session.as_ref()).await.expect("free load");
    assert_eq!(raw, loaded);
    let payload = session
        .get_state(SESSION_STATE_KEY)
        .await
        .expect("read payload")
        .expect("payload present");
    assert_eq!(payload["iteration"], json!(1));
    assert_eq!(payload["pending_follow_ups"], json!(["check the tests"]));
    assert_eq!(payload["plan_mode"]["mode"], json!("plan"));
    assert_eq!(
        payload["plan_mode"]["pre_plan_mode"],
        json!("normal"),
        "entering plan mode remembers the previous mode"
    );
    // The exact key set after repeated writes is covered by
    // `repeated_saves_keep_python_null_keys`.
}

#[tokio::test]
async fn persisted_plan_mode_is_visible_to_a_fresh_agent_over_the_same_session() {
    let temp = TempWorkspace::new().expect("temp workspace");
    let workspace = Workspace::for_language(temp.workspace(), "en");
    let first_model = ScriptedModel::text("first answer");
    let first = build_task_loop_agent(&first_model, &workspace);
    let session = Arc::new(AgentSession::new("reload-session"));

    first
        .invoke(json!({"query": "start"}), Some(session.clone()))
        .await
        .expect("first invocation succeeds");
    first.follow_up("keep this", None);
    first.switch_mode(AgentMode::Plan).expect("switch to plan");
    first.save_state().await.expect("save state");

    let second_model = ScriptedModel::text("second answer");
    let second = build_task_loop_agent(&second_model, &workspace);
    // `DeepAgent::load_state` reads the session bound by the last invocation, so a
    // fresh instance has no public seam to read another agent's session; the
    // equivalent assertion uses the module-level loader the agent itself calls.
    assert!(matches!(
        second.load_state().await,
        Err(Error::Validation(_))
    ));

    let restored = load_state(session.as_ref())
        .await
        .expect("load persisted state");
    assert_eq!(restored.iteration, 1);
    assert_eq!(restored.plan_mode.mode, "plan");
    assert_eq!(restored.plan_mode.pre_plan_mode.as_deref(), Some("normal"));
    assert_eq!(restored.plan_mode.plan_slug, None);
    assert_eq!(restored.pending_follow_ups, vec!["keep this".to_owned()]);

    // A second instance drives the same session and keeps the payload shape.
    // Its round counter is per invocation, so `iteration` restarts at 1.
    second
        .invoke(json!({"query": "resume"}), Some(session.clone()))
        .await
        .expect("second invocation succeeds");
    let after = load_state(session.as_ref())
        .await
        .expect("reload after the second invocation");
    assert_eq!(after.iteration, 1);

    let payload = session
        .get_state(SESSION_STATE_KEY)
        .await
        .expect("read payload")
        .expect("payload present");
    assert_eq!(payload["iteration"], json!(1));
    for key in ["iteration", "pending_follow_ups", "plan_mode"] {
        assert!(
            payload.get(key).is_some(),
            "the rewritten payload keeps `{key}`"
        );
    }
    // Note: the second instance's live plan mode comes from its own config; plan
    // mode is not re-applied from the persisted payload. The exact five-key shape
    // after repeated writes is covered by `repeated_saves_keep_python_null_keys`.
}

#[tokio::test]
async fn missing_and_cleared_sessions_load_the_default_state() {
    let session = AgentSession::new("no-state");
    let loaded = load_state(&session).await.expect("load default state");
    assert_eq!(loaded, DeepAgentState::default());
    assert_eq!(loaded.iteration, 0);
    assert_eq!(loaded.pending_follow_ups, Vec::<String>::new());
    assert_eq!(loaded.task_plan, None);
    assert_eq!(loaded.plan_mode.mode, "normal");
    assert_eq!(loaded.plan_mode.pre_plan_mode.as_deref(), Some("normal"));

    save_state(&session, &sample_state())
        .await
        .expect("save state");
    assert_eq!(
        load_state(&session).await.expect("load saved state"),
        sample_state()
    );

    clear_state(&session).await.expect("clear state");
    assert_eq!(
        load_state(&session).await.expect("load after clear"),
        DeepAgentState::default()
    );
    // Clearing writes `null`; a null patch may delete the path in the store.
    let cleared = session
        .get_state(SESSION_STATE_KEY)
        .await
        .expect("read cleared state");
    assert!(cleared.is_none() || cleared.as_ref().is_some_and(Value::is_null));

    // The agent-method form clears the session it is bound to.
    let temp = TempWorkspace::new().expect("temp workspace");
    let workspace = Workspace::for_language(temp.workspace(), "en");
    let model = ScriptedModel::text("hello");
    let agent = build_task_loop_agent(&model, &workspace);
    let bound = Arc::new(AgentSession::new("agent-clear"));
    agent
        .invoke(json!({"query": "hi"}), Some(bound.clone()))
        .await
        .expect("invocation succeeds");
    agent.save_state().await.expect("save state");
    let saved = load_state(bound.as_ref()).await.expect("load saved");
    assert_eq!(saved.iteration, 1);
    assert_eq!(saved.pending_follow_ups, Vec::<String>::new());
    assert_eq!(saved.plan_mode, PlanModeState::default());
    assert!(
        saved.stop_condition_state.is_some(),
        "the task loop snapshots its stop-condition evaluators"
    );
    assert!(saved.task_plan.is_none());

    agent.clear_state().await.expect("agent clear state");
    assert_eq!(
        agent.load_state().await.expect("agent load after clear"),
        DeepAgentState::default()
    );
    assert_eq!(
        load_state(bound.as_ref())
            .await
            .expect("free load after clear"),
        DeepAgentState::default()
    );
}

#[tokio::test]
async fn todos_persist_beside_the_state_in_the_workspace_not_inside_it() {
    let temp = TempWorkspace::new().expect("temp workspace");
    let workspace = Workspace::for_language(temp.workspace(), "en");

    // Todos are not part of the `deepagent` payload.
    assert!(
        DeepAgentState::default()
            .to_session_dict()
            .get("todos")
            .is_none(),
        "todos never live inside the DeepAgentState payload"
    );

    let store = TodoStore::for_workspace(&workspace);
    let tool = TodoCreateTool::new(store.clone(), "en");
    let ctx = ToolContext::new().with_session("session-1");
    assert_eq!(session_key(&ctx), "session-1");

    let payload = tool
        .invoke(
            json!({"todos": [
                {
                    "id": "read",
                    "content": "Read the spec",
                    "activeForm": "Reading the spec",
                    "description": "Read the specification",
                },
                {
                    "id": "write",
                    "content": "Write the tests",
                    "activeForm": "Writing the tests",
                    "description": "Write the integration tests",
                },
            ]}),
            &ctx,
        )
        .await
        .expect("todo_create succeeds");
    assert_eq!(payload["success"], json!(true));

    // The documented layout is `<workspace>/todo/<session>/todo.json`.
    let todo_node = workspace
        .get_node_path_for(WorkspaceNode::Todo)
        .expect("the en schema has a todo node");
    let path = store.path_for("session-1");
    assert_eq!(path, todo_node.join("session-1").join("todo.json"));
    assert!(path.is_file(), "todo_create writes {}", path.display());

    let text = tokio::fs::read_to_string(&path)
        .await
        .expect("read todo.json");
    let raw: Value = serde_json::from_str(&text).expect("todo.json is JSON");
    let items = raw.as_array().expect("todo.json is a JSON array");
    assert_eq!(items.len(), 2);
    assert_eq!(
        keys_sorted(&items[0]),
        vec![
            "activeForm",
            "content",
            "depends_on",
            "description",
            "id",
            "meta_data",
            "result_summary",
            "selected_model_id",
            "status",
        ],
        "persisted items use Python's camel-case `activeForm`"
    );
    assert_eq!(items[0]["id"], json!("read"));
    assert_eq!(items[0]["content"], json!("Read the spec"));
    assert_eq!(items[0]["activeForm"], json!("Reading the spec"));
    assert_eq!(items[0]["status"], json!("in_progress"));
    assert_eq!(items[1]["id"], json!("write"));
    assert_eq!(items[1]["status"], json!("pending"));

    // A fresh store over the same workspace reads the same list back.
    let reloaded = TodoStore::for_workspace(&workspace)
        .load("session-1")
        .await
        .expect("reload todos");
    assert_eq!(reloaded.len(), 2);
    assert_eq!(reloaded[0].id, "read");
    assert_eq!(reloaded[0].status, TodoStatus::InProgress);
    assert_eq!(reloaded[0].active_form, "Reading the spec");
    assert_eq!(reloaded[1].status, TodoStatus::Pending);

    // Another session does not see this list.
    assert_eq!(
        TodoStore::for_workspace(&workspace)
            .load("other-session")
            .await
            .expect("load other session"),
        Vec::<TodoItem>::new()
    );
}

/// Python's `DeepAgentState.to_session_dict` writes all five keys (absent values
/// as `null`) and `PlanModeState.to_session_dict` writes all four keys.
/// `save_state` reproduces Python's `update_dict` replace semantics, so a second
/// write over an existing `deepagent` object keeps every key, explicit nulls
/// included.
#[tokio::test]
async fn repeated_saves_keep_python_null_keys() {
    let session = AgentSession::new("state-null-keys");
    save_state(&session, &sample_state())
        .await
        .expect("first save");

    let without_plan = DeepAgentState {
        iteration: 1,
        task_plan: None,
        stop_condition_state: None,
        pending_follow_ups: Vec::new(),
        plan_mode: PlanModeState {
            mode: "normal".to_owned(),
            pre_plan_mode: None,
            plan_slug: None,
            prompt_context: None,
        },
    };
    save_state(&session, &without_plan)
        .await
        .expect("second save");

    let payload = session
        .get_state(SESSION_STATE_KEY)
        .await
        .expect("read payload")
        .expect("payload present");
    assert_eq!(
        keys_sorted(&payload),
        vec![
            "iteration",
            "pending_follow_ups",
            "plan_mode",
            "stop_condition_state",
            "task_plan",
        ]
    );
    assert_eq!(
        keys_sorted(&payload["plan_mode"]),
        vec!["mode", "plan_slug", "pre_plan_mode", "prompt_context"]
    );
    assert_eq!(payload["task_plan"], Value::Null);
}
