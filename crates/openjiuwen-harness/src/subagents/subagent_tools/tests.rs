//! Unit tests for the six persistent `subagent_*` tools.

use super::*;
use crate::subagents::test_support::{build_runtime, echo_runtime, Behavior, FakeFactory};

async fn call(tool: &impl Tool, inputs: Value) -> Value {
    tool.invoke(inputs, &ToolContext::new()).await.unwrap()
}

fn sorted(values: &[&str]) -> Vec<String> {
    let mut values: Vec<String> = values.iter().map(|value| (*value).to_owned()).collect();
    values.sort();
    values
}

fn sorted_strings(value: &Value) -> Vec<String> {
    let mut values: Vec<String> = value
        .as_array()
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default();
    values.sort();
    values
}

fn property_keys(card: &ToolCard) -> Vec<String> {
    let mut keys: Vec<String> = card.input_params["properties"]
        .as_object()
        .map(|properties| properties.keys().cloned().collect())
        .unwrap_or_default();
    keys.sort();
    keys
}

async fn spawn_one(runtime: &Arc<SubAgentRuntime>) -> (String, String) {
    let tool = SubagentSpawnTool::new(Arc::clone(runtime));
    let spawned = call(
        &tool,
        json!({
            "subagent_type": "general-purpose",
            "task_description": "task",
            "display_name": "Worker",
            "role": "tester",
        }),
    )
    .await;
    (
        spawned["data"]["subagent_id"].as_str().unwrap().to_owned(),
        spawned["data"]["task_id"].as_str().unwrap().to_owned(),
    )
}

async fn wait_one(runtime: &Arc<SubAgentRuntime>, id: &str) -> Value {
    let tool = SubagentWaitTool::new(Arc::clone(runtime));
    call(&tool, json!({"subagent_ids": [id]})).await
}

#[test]
fn cards_expose_the_python_names_and_schemas() {
    let cases: Vec<(ToolCard, &str, &[&str], &[&str])> = vec![
        (
            SubagentSpawnTool::card_for("en"),
            "subagent_spawn",
            &["subagent_type", "task_description", "display_name", "role"],
            &["subagent_type", "task_description", "display_name", "role"],
        ),
        (
            SubagentWaitTool::card_for("en"),
            "subagent_wait",
            &["subagent_ids"],
            &["subagent_ids", "timeout_ms"],
        ),
        (SubagentListTool::card_for("en"), "subagent_list", &[], &[]),
        (
            SubagentSendInputTool::card_for("en"),
            "subagent_send_input",
            &["subagent_id", "query"],
            &["subagent_id", "query", "interrupt"],
        ),
        (
            SubagentCloseTool::card_for("en"),
            "subagent_close",
            &["subagent_id"],
            &["subagent_id"],
        ),
        (
            SubagentResumeTool::card_for("en"),
            "subagent_resume",
            &["subagent_id"],
            &["subagent_id"],
        ),
    ];
    for (card, name, required, properties) in cases {
        assert_eq!(card.name, name);
        assert_ne!(card.description, "", "{name} has no description");
        assert_eq!(card.input_params["type"], "object", "{name}");
        assert_eq!(
            sorted_strings(&card.input_params["required"]),
            sorted(required),
            "{name}"
        );
        assert_eq!(property_keys(&card), sorted(properties), "{name}");
    }
    let wait = SubagentWaitTool::card_for("en");
    assert_eq!(
        wait.input_params["properties"]["subagent_ids"]["items"]["type"],
        "string"
    );
    assert_eq!(
        wait.input_params["properties"]["subagent_ids"]["minItems"],
        1
    );
    assert_eq!(
        wait.input_params["properties"]["timeout_ms"]["default"],
        2_400_000
    );
    assert_eq!(
        SubagentSendInputTool::card_for("en").input_params["properties"]["interrupt"]["default"],
        false
    );
}

#[tokio::test]
async fn subagent_tools_returns_the_python_order() {
    let (runtime, _factory) = echo_runtime();
    let tools = subagent_tools(runtime);
    let names: Vec<&str> = tools.iter().map(|tool| tool.card().name.as_str()).collect();
    assert_eq!(
        names,
        vec![
            "subagent_spawn",
            "subagent_wait",
            "subagent_list",
            "subagent_send_input",
            "subagent_close",
            "subagent_resume",
        ]
    );
    for tool in &tools {
        assert_ne!(tool.card().description, "");
    }
    let listed = tools[2]
        .invoke(json!({}), &ToolContext::new())
        .await
        .unwrap();
    assert_eq!(listed["success"], true);
}

#[tokio::test]
async fn spawn_returns_ids_and_initial_status() {
    let (runtime, factory) = echo_runtime();
    let tool = SubagentSpawnTool::new(Arc::clone(&runtime));
    let output = call(
        &tool,
        json!({
            "subagent_type": "general-purpose",
            "task_description": "task",
            "display_name": "Worker",
            "role": "tester",
        }),
    )
    .await;
    assert_eq!(output["success"], true);
    let data = &output["data"];
    let id = data["subagent_id"].as_str().unwrap();
    assert_eq!(data["sub_session_id"], id);
    assert_eq!(data["task_id"].as_str().unwrap().len(), 32);
    assert!(matches!(
        data["status"].as_str().unwrap(),
        "pending_init" | "running"
    ));
    let recorded = factory.recorded();
    assert_eq!(recorded.len(), 1);
    assert_eq!(recorded[0].1, id);
    assert_eq!(recorded[0].0.display_name.as_deref(), Some("Worker"));
    assert_eq!(recorded[0].0.role.as_deref(), Some("tester"));
    assert_eq!(runtime.capacity().0, 1);
}

#[tokio::test]
async fn spawn_rejects_incomplete_payloads_and_unknown_types() {
    let (runtime, _factory) = echo_runtime();
    let tool = SubagentSpawnTool::new(runtime);
    let incomplete = call(
        &tool,
        json!({
            "subagent_type": "general-purpose",
            "task_description": "task",
            "display_name": "Worker",
        }),
    )
    .await;
    assert_eq!(incomplete["success"], false);
    assert_eq!(incomplete["error"], SPAWN_REQUIRED_MESSAGE);

    let unknown = call(
        &tool,
        json!({
            "subagent_type": "nope",
            "task_description": "task",
            "display_name": "Worker",
            "role": "tester",
        }),
    )
    .await;
    assert_eq!(unknown["success"], false);
    assert!(unknown["error"]
        .as_str()
        .unwrap()
        .contains("subagent type not found"));
}

#[tokio::test]
async fn wait_returns_statuses_and_results_after_a_spawn() {
    let (runtime, _factory) = echo_runtime();
    let (id, _task_id) = spawn_one(&runtime).await;
    let tool = SubagentWaitTool::new(Arc::clone(&runtime));
    // `timeout_ms` 1 exercises the minimum clamp without waiting ten seconds.
    let output = call(
        &tool,
        json!({"subagent_ids": [id.as_str()], "timeout_ms": 1}),
    )
    .await;
    assert_eq!(output["success"], true);
    let data = &output["data"];
    assert_eq!(data["statuses"][id.as_str()], "completed");
    assert_eq!(data["results"][id.as_str()], "echo:task");
    assert_eq!(data["output_files"], json!({}));
    assert_eq!(data["timed_out"], false);
}

#[tokio::test]
async fn wait_collects_a_blocked_subagent_once_the_gate_opens() {
    let gate = Arc::new(tokio::sync::Notify::new());
    let factory = FakeFactory::new(Behavior::Blocked(Arc::clone(&gate)));
    let runtime = build_runtime(factory);
    let (id, _task_id) = spawn_one(&runtime).await;
    gate.notify_one();
    let output = wait_one(&runtime, &id).await;
    assert_eq!(output["data"]["statuses"][id.as_str()], "completed");
    assert_eq!(output["data"]["timed_out"], false);
}

#[tokio::test]
async fn wait_rejects_an_empty_list_and_a_non_integer_timeout() {
    let (runtime, _factory) = echo_runtime();
    let wait = SubagentWaitTool::new(runtime);

    for payload in [json!({"subagent_ids": []}), json!({"subagent_ids": ["  "]})] {
        let output = call(&wait, payload).await;
        assert_eq!(output["success"], false);
        assert_eq!(output["error"], "'subagent_ids' must be a non-empty list");
    }

    let fractional = call(&wait, json!({"subagent_ids": ["x"], "timeout_ms": 1.5})).await;
    assert_eq!(fractional["success"], false);
    assert_eq!(fractional["error"], "'timeout_ms' must be an integer");
}

#[test]
fn timeout_resolution_clamps_to_the_python_bounds() {
    assert_eq!(
        resolve_timeout_ms(&json!({})).ok(),
        Some(DEFAULT_WAIT_TIMEOUT_MS)
    );
    assert_eq!(
        resolve_timeout_ms(&json!({"timeout_ms": 1})).ok(),
        Some(10_000)
    );
    assert_eq!(
        resolve_timeout_ms(&json!({"timeout_ms": 9_999_999})).ok(),
        Some(3_600_000)
    );
    assert!(resolve_timeout_ms(&json!({"timeout_ms": 1.5})).is_err());
    assert!(resolve_timeout_ms(&json!({"timeout_ms": "1000"})).is_err());
}

#[tokio::test]
async fn list_reports_capacity_and_subagent_buckets() {
    let (runtime, _factory) = echo_runtime();
    let (id, _task_id) = spawn_one(&runtime).await;
    wait_one(&runtime, &id).await;

    let list = SubagentListTool::new(runtime);
    let output = call(&list, json!({})).await;
    assert_eq!(output["success"], true);
    let data = &output["data"];
    for key in [
        "capacity",
        "subagents",
        "live_subagents",
        "closed_subagents",
        "summary",
    ] {
        assert!(data.get(key).is_some(), "missing {key}");
    }
    assert_eq!(data["capacity"]["used"], 1);
    assert_eq!(data["capacity"]["max"], 10);
    assert_eq!(data["summary"]["live_count"], 1);
    assert_eq!(data["summary"]["closed_count"], 0);
    assert!(list
        .render_for_llm(&output)
        .starts_with("Capacity: 1/10 subagents in use."));
}

#[tokio::test]
async fn send_input_queues_a_new_task_on_a_completed_subagent() {
    let (runtime, _factory) = echo_runtime();
    let (id, first_task) = spawn_one(&runtime).await;
    wait_one(&runtime, &id).await;

    let send = SubagentSendInputTool::new(runtime);
    let output = call(&send, json!({"subagent_id": id.as_str(), "query": "again"})).await;
    assert_eq!(output["success"], true);
    let data = &output["data"];
    assert_eq!(data["subagent_id"], id);
    assert_ne!(data["task_id"].as_str().unwrap(), first_task);
    assert_eq!(data["task_id"].as_str().unwrap().len(), 32);
    assert!(matches!(
        data["status"].as_str().unwrap(),
        "pending_init" | "running" | "completed"
    ));
}

#[tokio::test]
async fn send_input_rejects_missing_fields_and_a_closed_target() {
    let (runtime, _factory) = echo_runtime();
    let (id, _task_id) = spawn_one(&runtime).await;
    wait_one(&runtime, &id).await;
    let close = SubagentCloseTool::new(Arc::clone(&runtime));
    call(&close, json!({"subagent_id": id.as_str()})).await;

    let send = SubagentSendInputTool::new(runtime);
    let missing_id = call(&send, json!({"query": "again"})).await;
    assert_eq!(missing_id["success"], false);
    assert_eq!(missing_id["error"], "'subagent_id' is required");

    let missing_query = call(&send, json!({"subagent_id": id.as_str()})).await;
    assert_eq!(missing_query["success"], false);
    assert_eq!(missing_query["error"], "'query' is required");

    let bad_interrupt = call(
        &send,
        json!({"subagent_id": id.as_str(), "query": "again", "interrupt": "yes"}),
    )
    .await;
    assert_eq!(bad_interrupt["success"], false);
    assert_eq!(bad_interrupt["error"], "'interrupt' must be a boolean");

    let closed = call(&send, json!({"subagent_id": id.as_str(), "query": "again"})).await;
    assert_eq!(closed["success"], false);
    assert!(closed["error"]
        .as_str()
        .unwrap()
        .contains("subagent closed or not found"));
}

#[tokio::test]
async fn close_is_idempotent_and_resume_restores_the_instance() {
    let (runtime, _factory) = echo_runtime();
    let (id, _task_id) = spawn_one(&runtime).await;
    wait_one(&runtime, &id).await;

    let close = SubagentCloseTool::new(Arc::clone(&runtime));
    let missing = call(&close, json!({})).await;
    assert_eq!(missing["success"], false);
    assert_eq!(missing["error"], "'subagent_id' is required");
    let first = call(&close, json!({"subagent_id": id.as_str()})).await;
    assert_eq!(first["success"], true);
    assert_eq!(first["data"]["previous_status"], "completed");
    let second = call(&close, json!({"subagent_id": id.as_str()})).await;
    assert_eq!(second["success"], true);
    assert_eq!(second["data"]["previous_status"], "closed");

    let resume = SubagentResumeTool::new(runtime);
    let blank = call(&resume, json!({"subagent_id": "  "})).await;
    assert_eq!(blank["success"], false);
    assert_eq!(blank["error"], "'subagent_id' is required");
    let restored = call(&resume, json!({"subagent_id": id.as_str()})).await;
    assert_eq!(restored["success"], true);
    assert_eq!(restored["data"]["restored"], true);
    assert_eq!(restored["data"]["status"], "completed");

    let unknown = call(&resume, json!({"subagent_id": "missing"})).await;
    assert_eq!(unknown["success"], false);
    assert!(unknown["error"]
        .as_str()
        .unwrap()
        .contains("subagent not found"));
}

#[test]
fn renderers_describe_payloads_and_errors() {
    let (runtime, _factory) = echo_runtime();
    let spawn = SubagentSpawnTool::new(Arc::clone(&runtime));
    assert_eq!(
        spawn.render_for_llm(&json!({
            "success": true,
            "data": {
                "subagent_id": "sub-1",
                "sub_session_id": "sub-1",
                "task_id": "abc",
                "status": "pending_init",
            },
            "error": null,
        })),
        "subagent_id: sub-1\nsub_session_id: sub-1\ntask_id: abc\nstatus: pending_init"
    );
    assert_eq!(
        spawn.render_for_llm(&json!({"success": false, "data": null, "error": "boom"})),
        "boom"
    );
}
