//! Unit tests for the additive `subagent_transcript` tool.

use std::sync::Arc;

use openjiuwen_foundation::tool::Tool;
use serde_json::{json, Value};

use super::super::SubagentTranscriptTool;
use super::{call, property_keys, sorted, sorted_strings, spawn_one, wait_one};
use crate::subagents::test_support::echo_runtime;

#[test]
fn transcript_card_exposes_the_name_schema_and_required_list() {
    let card = SubagentTranscriptTool::card_for("en");
    assert_eq!(card.name, "subagent_transcript");
    assert_ne!(card.description, "");
    assert_eq!(card.input_params["type"], "object");
    assert_eq!(
        sorted_strings(&card.input_params["required"]),
        sorted(&["subagent_id"])
    );
    assert_eq!(
        property_keys(&card),
        sorted(&["subagent_id", "limit", "after_seq"])
    );
    let properties = &card.input_params["properties"];
    assert_eq!(properties["subagent_id"]["type"], "string");
    assert_eq!(properties["limit"]["type"], "integer");
    assert_eq!(properties["limit"]["default"], 20);
    assert_eq!(properties["after_seq"]["type"], "integer");
    assert_eq!(properties["after_seq"]["default"], 0);
    assert_ne!(
        SubagentTranscriptTool::card_for("en").description,
        SubagentTranscriptTool::card_for("cn").description
    );
}

#[tokio::test]
async fn transcript_payload_reads_the_recorded_entries() {
    let (runtime, _factory) = echo_runtime();
    let (id, _task_id) = spawn_one(&runtime).await;
    wait_one(&runtime, &id).await;
    runtime
        .record_tool_call(&id, "read_file", "src/lib.rs", Some(true))
        .expect("the transcript accepts a recorded tool call");

    let tool = SubagentTranscriptTool::new(Arc::clone(&runtime));
    let output = call(&tool, json!({"subagent_id": id.as_str()})).await;
    assert_eq!(output["success"], true);
    let data = &output["data"];
    assert_eq!(data["subagent_id"], id);
    assert_eq!(data["status"], "completed");
    assert_eq!(data["activity"], "idle");
    assert_eq!(data["dropped"], 0);
    assert_eq!(data["last_kind"], "tool_call");
    assert_eq!(data["has_more"], false);
    let count = data["count"].as_u64().unwrap();
    assert!(count >= 2);
    assert_eq!(data["total"], count);
    let entries = data["entries"].as_array().unwrap();
    assert_eq!(entries.len() as u64, count);
    assert_eq!(entries[0]["kind"], "spawn");
    assert_eq!(entries[0]["content"], "task");
    let last = entries.last().unwrap();
    assert_eq!(last["kind"], "tool_call");
    assert_eq!(last["tool_name"], "read_file");
    assert_eq!(last["success"], true);
}

#[tokio::test]
async fn transcript_limit_is_clamped_and_validated() {
    let (runtime, _factory) = echo_runtime();
    let (id, _task_id) = spawn_one(&runtime).await;
    wait_one(&runtime, &id).await;
    let tool = SubagentTranscriptTool::new(Arc::clone(&runtime));

    let clamped = call(&tool, json!({"subagent_id": id.as_str(), "limit": 10_000})).await;
    assert_eq!(clamped["success"], true);
    let total = clamped["data"]["total"].as_u64().unwrap();
    assert_eq!(clamped["data"]["count"], total);

    let smallest = call(&tool, json!({"subagent_id": id.as_str(), "limit": 0})).await;
    assert_eq!(smallest["success"], true);
    assert_eq!(smallest["data"]["count"], 1);
    assert_eq!(smallest["data"]["has_more"], true);

    for payload in [
        json!({"subagent_id": id.as_str(), "limit": -1}),
        json!({"subagent_id": id.as_str(), "limit": "5"}),
        json!({"subagent_id": id.as_str(), "limit": 1.5}),
    ] {
        let output = call(&tool, payload).await;
        assert_eq!(output["success"], false);
        assert_eq!(output["error"], "'limit' must be an integer");
    }

    for payload in [
        json!({"subagent_id": id.as_str(), "after_seq": -1}),
        json!({"subagent_id": id.as_str(), "after_seq": "1"}),
        json!({"subagent_id": id.as_str(), "after_seq": 0.5}),
    ] {
        let output = call(&tool, payload).await;
        assert_eq!(output["success"], false);
        assert_eq!(
            output["error"],
            "'after_seq' must be a non-negative integer"
        );
    }
}

#[tokio::test]
async fn transcript_pages_with_after_seq() {
    let (runtime, _factory) = echo_runtime();
    let (id, _task_id) = spawn_one(&runtime).await;
    wait_one(&runtime, &id).await;
    for step in 0..5 {
        runtime
            .record_tool_call(&id, "grep", &format!("hit-{step}"), Some(true))
            .expect("the transcript accepts a recorded tool call");
    }
    let tool = SubagentTranscriptTool::new(Arc::clone(&runtime));

    let first = call(&tool, json!({"subagent_id": id.as_str(), "limit": 2})).await;
    assert_eq!(first["success"], true);
    assert_eq!(first["data"]["count"], 2);
    assert_eq!(first["data"]["has_more"], true);
    let entries = first["data"]["entries"].as_array().unwrap();
    assert_eq!(entries[0]["seq"], 1);
    let first_cursor = entries[1]["seq"].as_u64().unwrap();

    let second = call(
        &tool,
        json!({"subagent_id": id.as_str(), "after_seq": first_cursor, "limit": 2}),
    )
    .await;
    let second_entries = second["data"]["entries"].as_array().unwrap();
    assert_eq!(second_entries.len(), 2);
    assert_eq!(second_entries[0]["seq"].as_u64().unwrap(), first_cursor + 1);
    let second_cursor = second_entries[1]["seq"].as_u64().unwrap();

    let rest = call(
        &tool,
        json!({"subagent_id": id.as_str(), "after_seq": second_cursor, "limit": 200}),
    )
    .await;
    assert_eq!(rest["data"]["has_more"], false);
    assert!(rest["data"]["count"].as_u64().unwrap() >= 1);
    assert!(rest["data"]["entries"][0]["seq"].as_u64().unwrap() > second_cursor);
}

#[tokio::test]
async fn transcript_rejects_unknown_and_invalid_ids() {
    let (runtime, _factory) = echo_runtime();
    let tool = SubagentTranscriptTool::new(runtime);

    for payload in [
        json!({}),
        json!({"subagent_id": "   "}),
        json!({"subagent_id": 7}),
    ] {
        let output = call(&tool, payload).await;
        assert_eq!(output["success"], false);
        assert_eq!(output["error"], "'subagent_id' must be a non-empty string");
    }

    let unknown = call(&tool, json!({"subagent_id": "missing"})).await;
    assert_eq!(unknown["success"], false);
    assert_eq!(unknown["error"], "subagent not found: missing");
    assert_eq!(tool.render_for_llm(&unknown), "subagent not found: missing");
}

#[tokio::test]
async fn transcript_render_lists_entries_and_the_more_marker() {
    let (runtime, _factory) = echo_runtime();
    let (id, _task_id) = spawn_one(&runtime).await;
    wait_one(&runtime, &id).await;
    runtime
        .record_tool_call(&id, "read_file", "src/lib.rs", Some(true))
        .expect("the transcript accepts a recorded tool call");
    runtime
        .record_tool_call(&id, "bash", "false", Some(false))
        .expect("the transcript accepts a recorded tool call");

    let tool = SubagentTranscriptTool::new(Arc::clone(&runtime));
    let output = call(&tool, json!({"subagent_id": id.as_str()})).await;
    let rendered = tool.render_for_llm(&output);
    assert!(
        rendered.starts_with(&format!("Transcript of {id} (completed,")),
        "{rendered}"
    );
    assert!(rendered.contains("\n[1] spawn system: task"), "{rendered}");
    assert!(rendered.contains("(read_file, ok)"), "{rendered}");
    assert!(rendered.contains("(bash, failed)"), "{rendered}");
    assert!(!rendered.contains("more entries"), "{rendered}");

    let paged = call(&tool, json!({"subagent_id": id.as_str(), "limit": 1})).await;
    let total = paged["data"]["total"].as_u64().unwrap();
    let rendered = tool.render_for_llm(&paged);
    assert!(
        rendered.contains(&format!("… {} more entries", total - 1)),
        "{rendered}"
    );
}

#[test]
fn transcript_render_degrades_without_a_data_payload() {
    let (runtime, _factory) = echo_runtime();
    let tool = SubagentTranscriptTool::new(runtime);
    assert_eq!(tool.render_for_llm(&Value::Null), "");
    assert_eq!(tool.render_for_llm(&json!({"success": true})), "");
    let sparse = json!({"success": true, "data": {"entries": [], "has_more": false}});
    assert_eq!(
        tool.render_for_llm(&sparse),
        "Transcript of  (unknown, 0 entries)"
    );
}

#[tokio::test]
async fn transcript_payload_keeps_the_documented_key_set() {
    // Guards the documented key set: adding fields is an API change other tools mirror.
    let (runtime, _factory) = echo_runtime();
    let (id, _task_id) = spawn_one(&runtime).await;
    wait_one(&runtime, &id).await;
    let tool = SubagentTranscriptTool::new(runtime);
    let output = call(&tool, json!({"subagent_id": id.as_str()})).await;
    let data = output["data"].as_object().unwrap();
    assert_eq!(
        sorted(&data.keys().map(String::as_str).collect::<Vec<_>>()),
        sorted(&[
            "subagent_id",
            "status",
            "activity",
            "entries",
            "count",
            "total",
            "dropped",
            "last_kind",
            "has_more",
        ])
    );
}
