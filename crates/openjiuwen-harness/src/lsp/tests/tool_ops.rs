//! `LspTool` tests for the actions added in the final integration phase: implementation,
//! call hierarchy and diagnostics (push, pull, capability failures and "no server").
//!
//! Everything runs against the in-process fake server, so no language server is required.
use std::path::Path;
use std::time::Duration;

use serde_json::{json, Value};

use openjiuwen_foundation::tool::{Tool, ToolContext};

use crate::lsp::tool::{LspTool, LspToolConfig};
use crate::lsp::transport::ChannelTransport;

use super::fake_server::{self, FakeServer};

/// Builds a tool wired to a fresh fake server, rooted at `workspace`.
fn build_tool(workspace: &Path, timeout: Duration) -> (LspTool, FakeServer) {
    build_tool_advertising(workspace, timeout, fake_server::capabilities())
}

/// Builds a tool whose fake server advertises `capabilities` instead of the default set.
fn build_tool_advertising(
    workspace: &Path,
    timeout: Duration,
    capabilities: Value,
) -> (LspTool, FakeServer) {
    let (transport, peer) = ChannelTransport::new(64 * 1024);
    let server = fake_server::spawn_with_capabilities(peer, capabilities);
    let config = LspToolConfig::default()
        .with_language("en")
        .with_workspace_root(workspace)
        .with_language_id("rust")
        .with_request_timeout(timeout);
    (
        LspTool::new(config).with_transport(Box::new(transport)),
        server,
    )
}

/// Invokes the tool once and returns the raw JSON payload.
async fn call(tool: &LspTool, args: Value) -> Value {
    tool.invoke(args, &ToolContext::new()).await.unwrap()
}

/// Writes `content` to `workspace/src/lib.rs` and returns the relative path.
async fn write_source(workspace: &Path, content: &str) -> String {
    write_file(workspace, "lib.rs", content).await
}

/// Writes `content` to `workspace/src/<name>` and returns the relative path.
async fn write_file(workspace: &Path, name: &str, content: &str) -> String {
    let path = workspace.join("src");
    tokio::fs::create_dir_all(&path).await.unwrap();
    tokio::fs::write(path.join(name), content).await.unwrap();
    format!("src/{name}")
}

/// Hovers line 1, column 1 of `path`, opening the document for the fake server.
async fn open_document(tool: &LspTool, path: &str) -> Value {
    call(
        tool,
        json!({"action": "hover", "path": path, "line": 1, "column": 1}),
    )
    .await
}

#[tokio::test]
async fn implementation_renders_definition_text_and_locations() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_source(dir.path(), "fn main() {}\n").await;
    let (tool, _server) = build_tool(dir.path(), Duration::from_secs(2));
    let payload = call(
        &tool,
        json!({"action": "implementation", "path": path, "line": 2, "column": 1}),
    )
    .await;
    assert_eq!(payload["success"], json!(true), "{payload}");
    assert_eq!(payload["data"]["action"], json!("implementation"));
    let result = payload["data"]["result"].as_str().unwrap();
    assert!(result.contains("Defined in"), "{result}");
    assert!(result.contains("lib.rs:6:1"), "{result}");
    assert_eq!(
        payload["data"]["locations"][0]["range"]["start"]["line"],
        json!(5)
    );
    assert!(tool.render_for_llm(&payload).contains("Defined in"));
}

#[tokio::test]
async fn implementation_maps_method_not_found_to_a_clear_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_file(dir.path(), "unsupported_impl.rs", "fn main() {}\n").await;
    let (tool, _server) = build_tool(dir.path(), Duration::from_secs(2));
    let payload = call(
        &tool,
        json!({"action": "implementation", "path": path, "line": 1, "column": 1}),
    )
    .await;
    assert_eq!(payload["success"], json!(false));
    assert!(
        payload["error"].as_str().unwrap().contains(
            "Operation not supported by this language server: textDocument/implementation"
        ),
        "{payload}"
    );
}

#[tokio::test]
async fn prepare_call_hierarchy_renders_every_item() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_source(dir.path(), "fn main() {}\n").await;
    let (tool, _server) = build_tool(dir.path(), Duration::from_secs(2));
    let payload = call(
        &tool,
        json!({"action": "prepare_call_hierarchy", "path": path, "line": 1, "column": 1}),
    )
    .await;
    assert_eq!(payload["success"], json!(true), "{payload}");
    assert_eq!(payload["data"]["action"], json!("prepare_call_hierarchy"));
    let result = payload["data"]["result"].as_str().unwrap();
    assert!(result.starts_with("2 call hierarchy items:"), "{result}");
    assert!(result.contains("caller.rs:2: caller"), "{result}");
    assert!(result.contains("callee.rs:8: callee"), "{result}");
    assert_eq!(payload["data"]["items"].as_array().unwrap().len(), 2);
}

#[tokio::test]
async fn prepare_call_hierarchy_maps_method_not_found() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_file(dir.path(), "unsupported_calls.rs", "fn main() {}\n").await;
    let (tool, _server) = build_tool(dir.path(), Duration::from_secs(2));
    let payload = call(
        &tool,
        json!({"action": "prepare_call_hierarchy", "path": path, "line": 1, "column": 1}),
    )
    .await;
    assert_eq!(payload["success"], json!(false));
    assert!(
        payload["error"].as_str().unwrap().contains(
            "Operation not supported by this language server: textDocument/prepareCallHierarchy"
        ),
        "{payload}"
    );
}

#[tokio::test]
async fn incoming_calls_are_grouped_by_caller() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_source(dir.path(), "fn main() {}\n").await;
    let (tool, _server) = build_tool(dir.path(), Duration::from_secs(2));
    let payload = call(
        &tool,
        json!({"action": "incoming_calls", "path": path, "line": 1, "column": 1}),
    )
    .await;
    assert_eq!(payload["success"], json!(true), "{payload}");
    assert_eq!(payload["data"]["action"], json!("incoming_calls"));
    let result = payload["data"]["result"].as_str().unwrap();
    assert!(result.contains("caller.rs:"), "{result}");
    assert!(result.contains("caller.rs:2 -> call site 3:5"), "{result}");
    assert_eq!(payload["data"]["calls"][0]["from"]["name"], json!("caller"));
    assert_eq!(payload["data"]["item"]["name"], json!("caller"));
}

#[tokio::test]
async fn outgoing_calls_render_call_sites_and_the_callee() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_source(dir.path(), "fn main() {}\n").await;
    let (tool, _server) = build_tool(dir.path(), Duration::from_secs(2));
    let payload = call(
        &tool,
        json!({"action": "outgoing_calls", "path": path, "line": 1, "column": 1}),
    )
    .await;
    assert_eq!(payload["success"], json!(true), "{payload}");
    assert_eq!(payload["data"]["action"], json!("outgoing_calls"));
    let result = payload["data"]["result"].as_str().unwrap();
    assert!(
        result.contains("call site 4:3 -> callee (") && result.contains("callee.rs:6)"),
        "{result}"
    );
    assert_eq!(payload["data"]["calls"][0]["to"]["name"], json!("callee"));
}

#[tokio::test]
async fn a_missing_call_hierarchy_item_is_a_success_with_the_empty_text() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_file(dir.path(), "noitems.rs", "fn main() {}\n").await;
    let (tool, _server) = build_tool(dir.path(), Duration::from_secs(2));
    let payload = call(
        &tool,
        json!({"action": "incoming_calls", "path": path, "line": 1, "column": 1}),
    )
    .await;
    assert_eq!(payload["success"], json!(true), "{payload}");
    assert_eq!(
        payload["data"]["result"],
        json!("No call hierarchy available.")
    );
    assert_eq!(payload["data"]["calls"], json!([]));
    assert_eq!(payload["data"]["item"], Value::Null);
}

#[tokio::test]
async fn call_hierarchy_requests_map_method_not_found() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_file(dir.path(), "nocalls.rs", "fn main() {}\n").await;
    let (tool, _server) = build_tool(dir.path(), Duration::from_secs(2));

    let incoming = call(
        &tool,
        json!({"action": "incoming_calls", "path": &path, "line": 1, "column": 1}),
    )
    .await;
    assert_eq!(incoming["success"], json!(false));
    assert!(
        incoming["error"].as_str().unwrap().contains(
            "Operation not supported by this language server: callHierarchy/incomingCalls"
        ),
        "{incoming}"
    );

    let outgoing = call(
        &tool,
        json!({"action": "outgoing_calls", "path": path, "line": 1, "column": 1}),
    )
    .await;
    assert_eq!(outgoing["success"], json!(false));
    assert!(
        outgoing["error"].as_str().unwrap().contains(
            "Operation not supported by this language server: callHierarchy/outgoingCalls"
        ),
        "{outgoing}"
    );
}

#[tokio::test]
async fn diagnostics_prefer_push_data_then_pull_and_reuse_unchanged_reports() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_source(dir.path(), "fn main() {}\n").await;
    let (tool, server) = build_tool(dir.path(), Duration::from_secs(2));

    // The first `didOpen` publishes one warning, buffered by the registry.
    let opened = open_document(&tool, &path).await;
    assert_eq!(opened["success"], json!(true));
    let pushed = call(&tool, json!({"action": "diagnostics", "path": &path})).await;
    assert_eq!(pushed["success"], json!(true), "{pushed}");
    assert_eq!(pushed["data"]["source"], json!("push"));
    assert_eq!(pushed["data"]["diagnostics"].as_array().unwrap().len(), 1);
    assert!(
        pushed["data"]["result"]
            .as_str()
            .unwrap()
            .contains("fake warning"),
        "{pushed}"
    );

    // Rewriting the file with `clean` makes the fake publish an empty list, clearing the push
    // buffer; the next call has to pull.
    write_source(dir.path(), "// clean\nfn main() {}\n").await;
    let reopened = open_document(&tool, &path).await;
    assert_eq!(reopened["success"], json!(true));

    let pulled = call(&tool, json!({"action": "diagnostics", "path": &path})).await;
    assert_eq!(pulled["success"], json!(true), "{pulled}");
    assert_eq!(pulled["data"]["source"], json!("pull"));
    assert_eq!(pulled["data"]["diagnostics"].as_array().unwrap().len(), 1);
    assert!(
        pulled["data"]["result"]
            .as_str()
            .unwrap()
            .contains("cannot find name"),
        "{pulled}"
    );

    // The same pull with the cached result id answers `unchanged`: the previous items are
    // returned once, not duplicated.
    let unchanged = call(&tool, json!({"action": "diagnostics", "path": &path})).await;
    assert_eq!(unchanged["success"], json!(true), "{unchanged}");
    assert_eq!(unchanged["data"]["source"], json!("pull"));
    assert_eq!(
        unchanged["data"]["diagnostics"].as_array().unwrap().len(),
        1
    );

    let pulls: Vec<Value> = server
        .messages()
        .into_iter()
        .filter(|message| {
            message.get("method").and_then(Value::as_str) == Some("textDocument/diagnostic")
        })
        .collect();
    assert_eq!(pulls.len(), 2, "one full pull and one unchanged pull");
    assert!(pulls[0]["params"].get("previousResultId").is_none());
    assert_eq!(pulls[1]["params"]["previousResultId"], json!("r1"));
}

#[tokio::test]
async fn diagnostics_without_a_path_drain_every_buffered_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_source(dir.path(), "fn main() {}\n").await;
    let (tool, _server) = build_tool(dir.path(), Duration::from_secs(2));
    let opened = open_document(&tool, &path).await;
    assert_eq!(opened["success"], json!(true));

    let drained = call(&tool, json!({"action": "diagnostics"})).await;
    assert_eq!(drained["success"], json!(true), "{drained}");
    assert_eq!(drained["data"]["count"], json!(1));
    assert_eq!(drained["data"]["files"].as_array().unwrap().len(), 1);
    let result = drained["data"]["result"].as_str().unwrap();
    assert!(result.contains("lib.rs:"), "{result}");
    assert!(result.contains("fake warning"), "{result}");

    let empty = call(&tool, json!({"action": "diagnostics"})).await;
    assert_eq!(empty["success"], json!(true));
    assert_eq!(empty["data"]["count"], json!(0));
    assert_eq!(empty["data"]["result"], json!("No diagnostics found."));
}

#[tokio::test]
async fn diagnostics_without_a_server_report_the_diagnostics_as_unknown() {
    let dir = tempfile::tempdir().unwrap();
    write_source(dir.path(), "fn main() {}\n").await;
    let tool = LspTool::new(
        LspToolConfig::default()
            .with_language("en")
            .with_workspace_root(dir.path()),
    );
    let payload = call(
        &tool,
        json!({"action": "diagnostics", "path": "src/lib.rs"}),
    )
    .await;
    // Informational, not an error: nothing failed, there is nothing to retry. The payload still
    // has to say that the diagnostics are unknown, and `source` must never claim a pull.
    assert_eq!(payload["success"], json!(true), "{payload}");
    assert_eq!(payload["data"]["source"], json!("none"));
    assert_eq!(payload["data"]["diagnostics"], json!([]));
    let result = payload["data"]["result"].as_str().unwrap();
    assert!(result.contains("No language server"), "{result}");
    assert!(result.contains("unknown"), "{result}");
    assert!(
        !result.contains("No diagnostics found."),
        "no server must not read as a clean file: {result}"
    );
    assert!(tool.render_for_llm(&payload).contains("are unknown"));
}

#[tokio::test]
async fn a_clean_full_report_is_a_success_with_an_empty_list() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_file(dir.path(), "pristine.rs", "fn main() {}\n").await;
    let (tool, _server) = build_tool(dir.path(), Duration::from_secs(2));
    // The fake publishes one warning on `didOpen`; drop it so the call has to pull.
    let opened = open_document(&tool, &path).await;
    assert_eq!(opened["success"], json!(true), "{opened}");
    tool.diagnostics().clear();

    let payload = call(&tool, json!({"action": "diagnostics", "path": &path})).await;
    assert_eq!(payload["success"], json!(true), "{payload}");
    assert_eq!(payload["data"]["source"], json!("pull"));
    assert_eq!(payload["data"]["diagnostics"], json!([]));
    assert_eq!(payload["data"]["result"], json!("No diagnostics found."));
    assert!(payload["data"].get("stale").is_none(), "{payload}");
    assert!(tool
        .render_for_llm(&payload)
        .contains("No diagnostics found."));
}

#[tokio::test]
async fn an_unadvertised_diagnostic_capability_is_a_typed_failure() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_file(dir.path(), "lib.rs", "fn main() {}\n").await;
    let mut capabilities = fake_server::capabilities();
    capabilities
        .as_object_mut()
        .unwrap()
        .remove("diagnosticProvider");
    let (tool, server) = build_tool_advertising(dir.path(), Duration::from_secs(2), capabilities);

    let payload = call(&tool, json!({"action": "diagnostics", "path": &path})).await;
    assert_eq!(
        payload["success"],
        json!(false),
        "a missing capability must not read as a clean file: {payload}"
    );
    let error = payload["error"].as_str().unwrap();
    assert!(
        error.contains("not supported")
            && error.contains("does not advertise textDocument/diagnostic"),
        "{error}"
    );
    assert!(error.contains("unknown, not clean"), "{error}");
    assert_eq!(payload["data"]["action"], json!("diagnostics"));
    assert!(
        payload["data"]["file_path"]
            .as_str()
            .unwrap()
            .ends_with("lib.rs"),
        "{payload}"
    );
    assert_eq!(
        payload["data"]["capability"],
        json!("textDocument/diagnostic")
    );
    assert!(
        payload["data"].get("diagnostics").is_none(),
        "no diagnostics are fabricated for an unsupported capability: {payload}"
    );
    assert!(!tool
        .render_for_llm(&payload)
        .contains("No diagnostics found."));
    assert!(
        !server
            .methods()
            .iter()
            .any(|method| method == "textDocument/diagnostic"),
        "the tool asks no server for a method it does not advertise"
    );
}

#[tokio::test]
async fn a_rejected_diagnostic_method_is_a_typed_failure() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_file(dir.path(), "rejects_pull.rs", "fn main() {}\n").await;
    let (tool, server) = build_tool(dir.path(), Duration::from_secs(2));
    let opened = open_document(&tool, &path).await;
    assert_eq!(opened["success"], json!(true), "{opened}");
    tool.diagnostics().clear();

    // The server advertises `diagnosticProvider` but answers `-32601` with no textual hint, so
    // only the typed code classifies it; either way it must not read as "no problems".
    let payload = call(&tool, json!({"action": "diagnostics", "path": &path})).await;
    assert_eq!(payload["success"], json!(false), "{payload}");
    let error = payload["error"].as_str().unwrap();
    assert!(error.contains("textDocument/diagnostic"), "{error}");
    assert!(error.contains("unknown, not clean"), "{error}");
    assert_eq!(payload["data"]["action"], json!("diagnostics"));
    assert_eq!(
        payload["data"]["capability"],
        json!("textDocument/diagnostic")
    );
    assert!(payload["data"].get("diagnostics").is_none(), "{payload}");
    assert!(!tool
        .render_for_llm(&payload)
        .contains("No diagnostics found."));
    assert_eq!(pull_requests(&server).len(), 1, "exactly one pull attempt");
}

/// The pull requests the fake server received, in order.
fn pull_requests(server: &FakeServer) -> Vec<Value> {
    server
        .messages()
        .into_iter()
        .filter(|message| {
            message.get("method").and_then(Value::as_str) == Some("textDocument/diagnostic")
        })
        .collect()
}

#[tokio::test]
async fn unchanged_pull_reuses_the_cached_snapshot_without_a_stale_marker() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_source(dir.path(), "// clean\nfn main() {}\n").await;
    let (tool, _server) = build_tool(dir.path(), Duration::from_secs(2));
    // The first pull is a full report; the second one carries its `resultId` and is answered
    // with `unchanged`, so the cached items are reused.
    let full = call(&tool, json!({"action": "diagnostics", "path": &path})).await;
    assert_eq!(full["data"]["diagnostics"].as_array().unwrap().len(), 1);
    assert!(full["data"].get("stale").is_none(), "{full}");
    // Drop the push notification the fake published on `didOpen` so the next call pulls again;
    // only the pull caches matter here.
    tool.diagnostics().clear();
    let unchanged = call(&tool, json!({"action": "diagnostics", "path": &path})).await;
    assert_eq!(unchanged["success"], json!(true), "{unchanged}");
    assert_eq!(unchanged["data"]["source"], json!("pull"));
    assert_eq!(
        unchanged["data"]["diagnostics"].as_array().unwrap().len(),
        1,
        "the cached full report answers the unchanged pull"
    );
    assert!(
        unchanged["data"].get("stale").is_none(),
        "a cached snapshot is authoritative: {unchanged}"
    );
}

#[tokio::test]
async fn unchanged_pull_without_a_snapshot_is_marked_stale() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_file(dir.path(), "stale_pull.rs", "fn main() {}\n").await;
    let (tool, server) = build_tool(dir.path(), Duration::from_secs(2));
    let payload = call(&tool, json!({"action": "diagnostics", "path": &path})).await;
    assert_eq!(payload["success"], json!(true), "{payload}");
    assert_eq!(payload["data"]["source"], json!("pull"));
    assert_eq!(
        payload["data"]["stale"],
        json!(true),
        "an unchanged answer without a baseline must not read as \"no problems\": {payload}"
    );
    assert_eq!(payload["data"]["diagnostics"], json!([]));
    let result = payload["data"]["result"].as_str().unwrap();
    assert!(result.contains("stale"), "{result}");
    assert!(
        !result.contains("No diagnostics found."),
        "the payload must not claim the file is clean: {result}"
    );
    let pulls = pull_requests(&server);
    assert_eq!(
        pulls.len(),
        2,
        "one pull plus exactly one forced full re-pull"
    );
    assert!(pulls
        .iter()
        .all(|pull| pull["params"].get("previousResultId").is_none()));
}

#[tokio::test]
async fn unchanged_pull_without_a_snapshot_forces_one_full_pull() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_file(dir.path(), "late_full.rs", "fn main() {}\n").await;
    let (tool, server) = build_tool(dir.path(), Duration::from_secs(2));
    let payload = call(&tool, json!({"action": "diagnostics", "path": &path})).await;
    assert_eq!(payload["success"], json!(true), "{payload}");
    assert_eq!(
        payload["data"]["stale"],
        json!(null),
        "the forced re-pull resolved the report: {payload}"
    );
    assert_eq!(
        payload["data"]["diagnostics"].as_array().unwrap().len(),
        1,
        "{payload}"
    );
    assert!(payload["data"]["result"]
        .as_str()
        .unwrap()
        .contains("cannot find name"));
    let pulls = pull_requests(&server);
    assert_eq!(
        pulls.len(),
        2,
        "the bogus unchanged answer triggers one retry, never a loop"
    );
    assert!(pulls[1]["params"].get("previousResultId").is_none());
}

#[tokio::test]
async fn method_not_found_is_classified_from_the_typed_code() {
    let dir = tempfile::tempdir().unwrap();
    // The fake answers with code -32601 but neither "-32601" nor "method not found" in the
    // message, so only a typed check classifies the answer.
    let path = write_file(dir.path(), "typedonly.rs", "fn main() {}\n").await;
    let (tool, _server) = build_tool(dir.path(), Duration::from_secs(2));
    let payload = call(
        &tool,
        json!({"action": "completion", "path": path, "line": 1, "column": 1}),
    )
    .await;
    assert_eq!(payload["success"], json!(false));
    assert_eq!(
        payload["error"],
        json!("Operation not supported by this language server: textDocument/completion")
    );
}

#[tokio::test]
async fn method_not_found_message_is_the_documented_last_resort() {
    let dir = tempfile::tempdir().unwrap();
    // A non-standard code with the text: the typed code does not match, the fallback does.
    let path = write_file(dir.path(), "sniffonly.rs", "fn main() {}\n").await;
    let (tool, _server) = build_tool(dir.path(), Duration::from_secs(2));
    let payload = call(
        &tool,
        json!({"action": "completion", "path": path, "line": 1, "column": 1}),
    )
    .await;
    assert_eq!(payload["success"], json!(false));
    assert_eq!(
        payload["error"],
        json!("Operation not supported by this language server: textDocument/completion")
    );
}

#[test]
fn method_not_found_detection_prefers_the_typed_code() {
    use crate::lsp::protocol::{is_method_not_found, server_error, LspError};

    let typed = LspError::new(-32601, "no handler registered").into_error();
    assert!(
        !typed
            .message()
            .to_ascii_lowercase()
            .contains("method not found"),
        "the message carries no textual hint: {}",
        typed.message()
    );
    assert!(
        server_error(&typed).is_some_and(LspError::is_method_not_found),
        "the typed code survives the conversion"
    );
    assert!(is_method_not_found(&typed));

    let mis_coded = LspError::new(0, "method not found: textDocument/hover").into_error();
    assert!(
        !server_error(&mis_coded).is_some_and(LspError::is_method_not_found),
        "a non-standard code is not a typed match"
    );
    assert!(
        is_method_not_found(&mis_coded),
        "the message fallback still classifies a server that omits the code"
    );

    let unrelated = LspError::new(-32000, "internal error").into_error();
    assert!(!is_method_not_found(&unrelated));
    assert!(server_error(&unrelated).is_some(), "the error stays typed");
}
