//! End-to-end `LspTool` tests against the in-process fake server.
//!
//! Every failure path is asserted to be a `success=false` payload with an `error` string; no
//! test needs a language server process.
use std::path::Path;
use std::time::Duration;

use serde_json::{json, Value};

use openjiuwen_foundation::tool::{Tool, ToolContext};

use crate::lsp::config::LspServerConfig;
use crate::lsp::tool::{LspAction, LspTool, LspToolConfig};
use crate::lsp::transport::ChannelTransport;

use super::fake_server::{self, FakeServer};

/// Builds a tool wired to a fresh fake server, rooted at `workspace`.
fn build_tool(workspace: &Path, timeout: Duration) -> (LspTool, FakeServer) {
    let (transport, peer) = ChannelTransport::new(64 * 1024);
    let server = fake_server::spawn(peer);
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

/// Writes `content` to `workspace/src/lib.rs`.
async fn write_source(workspace: &Path, content: &str) -> String {
    let path = workspace.join("src");
    tokio::fs::create_dir_all(&path).await.unwrap();
    tokio::fs::write(path.join("lib.rs"), content)
        .await
        .unwrap();
    "src/lib.rs".to_owned()
}

#[test]
fn card_has_the_python_name_uuid_schema_and_bilingual_description() {
    let card = LspTool::card_for("en");
    assert_eq!(card.name, "lsp");
    assert_eq!(card.id.len(), 32);
    assert!(card.description.contains("LSP"));
    assert!(card.parallel_safe);

    let schema = &card.input_params;
    assert_eq!(schema["type"], "object");
    let properties = schema["properties"].as_object().unwrap();
    let mut keys: Vec<&str> = properties.keys().map(String::as_str).collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        ["action", "column", "context_lines", "line", "path", "query"]
    );
    let actions: Vec<&str> = properties["action"]["enum"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_str().unwrap())
        .collect();
    assert_eq!(
        actions,
        [
            "hover",
            "definition",
            "references",
            "completion",
            "symbols",
            "workspace_symbols",
            "diagnostics",
            "implementation",
            "prepare_call_hierarchy",
            "incoming_calls",
            "outgoing_calls"
        ],
        "the card lists every action in ALL order"
    );
    assert_eq!(LspAction::ALL.len(), 11);
    assert!(!LspAction::Diagnostics.requires_path());
    assert!(!LspAction::WorkspaceSymbols.requires_path());
    for action in LspAction::ALL {
        assert_eq!(
            action.requires_path(),
            !matches!(action, LspAction::WorkspaceSymbols | LspAction::Diagnostics),
            "{} path requirement",
            action.as_str()
        );
    }
    assert_eq!(schema["required"], json!(["action"]));

    assert_ne!(LspTool::card_for("cn").description, card.description);
}

#[tokio::test]
async fn hover_returns_short_text_and_the_normalised_result() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_source(dir.path(), "fn main() {\n    let value = 1;\n}\n").await;
    let (tool, _server) = build_tool(dir.path(), Duration::from_secs(2));
    let payload = call(
        &tool,
        json!({"action": "hover", "path": path, "line": 1, "column": 1}),
    )
    .await;
    assert_eq!(payload["success"], json!(true));
    assert_eq!(payload["data"]["action"], json!("hover"));
    assert!(payload["data"]["result"]
        .as_str()
        .unwrap()
        .contains("fake hover"));
    assert_eq!(
        payload["data"]["hover"]["contents"],
        json!("**fake hover**")
    );
    assert_eq!(
        payload["data"]["file_path"].as_str().unwrap(),
        dir.path().join("src/lib.rs").to_string_lossy()
    );
    assert!(tool.render_for_llm(&payload).contains("fake hover"));
}

#[tokio::test]
async fn definition_includes_a_source_excerpt_when_context_lines_is_set() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_source(
        dir.path(),
        "fn main() {\n    let x = 1;\n    let y = 2;\n}\n",
    )
    .await;
    let (tool, _server) = build_tool(dir.path(), Duration::from_secs(2));
    let payload = call(
        &tool,
        json!({"action": "definition", "path": path, "line": 3, "column": 9, "context_lines": 1}),
    )
    .await;
    assert_eq!(payload["success"], json!(true));
    let result = payload["data"]["result"].as_str().unwrap();
    assert!(result.contains("Defined in"), "{result}");
    assert!(result.contains("lib.rs:3:5"), "{result}");
    assert!(result.contains("let y = 2;"), "{result}");
    assert!(result.contains("-- "), "{result}");
    assert_eq!(
        payload["data"]["locations"][0]["range"]["start"]["line"],
        json!(2)
    );
}

#[tokio::test]
async fn references_are_grouped_by_file_and_one_based() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_source(dir.path(), "fn main() {}\n").await;
    let (tool, _server) = build_tool(dir.path(), Duration::from_secs(2));
    let payload = call(
        &tool,
        json!({"action": "references", "path": path, "line": 1, "column": 1}),
    )
    .await;
    assert_eq!(payload["success"], json!(true));
    let result = payload["data"]["result"].as_str().unwrap();
    assert!(result.contains("lib.rs:"), "{result}");
    assert!(result.contains("  2:3"), "{result}");
    assert!(result.contains("  4:5"), "{result}");
}

#[tokio::test]
async fn completion_and_document_symbols_render_names_and_kinds() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_source(dir.path(), "fn main() {}\nfn helper() {}\n").await;
    let (tool, _server) = build_tool(dir.path(), Duration::from_secs(2));

    let completion = call(
        &tool,
        json!({"action": "completion", "path": path, "line": 2, "column": 3}),
    )
    .await;
    assert_eq!(completion["success"], json!(true));
    let text = completion["data"]["result"].as_str().unwrap();
    assert!(text.contains("alpha (Function) - fake detail"), "{text}");
    assert!(text.contains("beta (Variable)"), "{text}");

    let symbols = call(&tool, json!({"action": "symbols", "path": path})).await;
    assert_eq!(symbols["success"], json!(true));
    let text = symbols["data"]["result"].as_str().unwrap();
    assert!(text.contains("Function main - fn main()"), "{text}");
    assert!(text.contains("  Function helper"), "{text}");
    assert_eq!(
        symbols["data"]["symbols"][0]["children"][0]["name"],
        json!("helper")
    );
}

#[tokio::test]
async fn workspace_symbols_use_the_query_and_group_by_file() {
    let dir = tempfile::tempdir().unwrap();
    write_source(dir.path(), "fn main() {}\n").await;
    let (tool, _server) = build_tool(dir.path(), Duration::from_secs(2));
    let payload = call(
        &tool,
        json!({"action": "workspace_symbols", "query": "main"}),
    )
    .await;
    assert_eq!(payload["success"], json!(true));
    let result = payload["data"]["result"].as_str().unwrap();
    assert!(result.contains("lib.rs:"), "{result}");
    assert!(result.contains("  3: Function crate.main"), "{result}");
}

#[tokio::test]
async fn definition_positions_are_one_based_and_zero_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_source(dir.path(), "fn main() {}\n").await;
    let (tool, _server) = build_tool(dir.path(), Duration::from_secs(2));
    let payload = call(
        &tool,
        json!({"action": "definition", "path": path, "line": 0, "column": 0}),
    )
    .await;
    assert_eq!(payload["success"], json!(false));
    assert!(
        payload["error"].as_str().unwrap().contains("1-based"),
        "{payload}"
    );
}

#[tokio::test]
async fn missing_or_escaping_paths_return_success_false() {
    let dir = tempfile::tempdir().unwrap();
    write_source(dir.path(), "fn main() {}\n").await;
    let (tool, _server) = build_tool(dir.path(), Duration::from_secs(2));

    let missing = call(&tool, json!({"action": "hover"})).await;
    assert_eq!(missing["success"], json!(false));
    assert!(missing["error"]
        .as_str()
        .unwrap()
        .contains("path is required"));

    let escape = call(&tool, json!({"action": "hover", "path": "../escape.rs"})).await;
    assert_eq!(escape["success"], json!(false));
    assert!(
        escape["error"]
            .as_str()
            .unwrap()
            .contains("outside the workspace"),
        "{escape}"
    );

    let absent = call(&tool, json!({"action": "hover", "path": "src/missing.rs"})).await;
    assert_eq!(absent["success"], json!(false));
    assert!(
        absent["error"].as_str().unwrap().contains("File not found"),
        "{absent}"
    );
}

#[tokio::test]
async fn oversized_files_are_rejected_before_reading() {
    let dir = tempfile::tempdir().unwrap();
    write_source(dir.path(), "fn main() { let value = 12345; }\n").await;
    let (transport, peer) = ChannelTransport::new(64 * 1024);
    let _server = fake_server::spawn(peer);
    let config = LspToolConfig::default()
        .with_workspace_root(dir.path())
        .with_max_file_size(4);
    let tool = LspTool::new(config).with_transport(Box::new(transport));
    let payload = call(
        &tool,
        json!({"action": "hover", "path": "src/lib.rs", "line": 1}),
    )
    .await;
    assert_eq!(payload["success"], json!(false));
    assert!(
        payload["error"].as_str().unwrap().contains("too large"),
        "{payload}"
    );
}

#[tokio::test]
async fn diagnostics_report_pushed_warnings_after_a_document_event() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_source(dir.path(), "fn main() {}\n").await;
    let (tool, _server) = build_tool(dir.path(), Duration::from_secs(2));
    let opened = call(
        &tool,
        json!({"action": "hover", "path": &path, "line": 1, "column": 1}),
    )
    .await;
    assert_eq!(opened["success"], json!(true));
    let payload = call(&tool, json!({"action": "diagnostics", "path": path})).await;
    assert_eq!(payload["success"], json!(true));
    assert_eq!(payload["data"]["source"], json!("push"));
    assert!(
        payload["data"]["result"]
            .as_str()
            .unwrap()
            .contains("fake warning"),
        "{payload}"
    );
}

#[tokio::test]
async fn unknown_actions_and_missing_servers_fail_without_panicking() {
    let dir = tempfile::tempdir().unwrap();
    let (tool, _server) = build_tool(dir.path(), Duration::from_secs(2));
    let invalid = call(&tool, json!({"action": "teleport"})).await;
    assert_eq!(invalid["success"], json!(false));
    assert!(
        invalid["error"].as_str().unwrap().contains("Invalid input"),
        "{invalid}"
    );

    let unconfigured = LspTool::new(LspToolConfig::default().with_workspace_root(dir.path()));
    let payload = call(
        &unconfigured,
        json!({"action": "hover", "path": "src/lib.rs", "line": 1}),
    )
    .await;
    assert_eq!(payload["success"], json!(false));
    assert!(
        payload["error"]
            .as_str()
            .unwrap()
            .contains("no LSP server configured"),
        "{payload}"
    );
}

#[tokio::test]
async fn a_missing_server_binary_is_a_payload_error() {
    let dir = tempfile::tempdir().unwrap();
    write_source(dir.path(), "fn main() {}\n").await;
    let config = LspToolConfig::default()
        .with_workspace_root(dir.path())
        .with_server(LspServerConfig::new(
            "openjiuwen-definitely-not-a-real-lsp-binary",
        ));
    let tool = LspTool::new(config);
    let payload = call(
        &tool,
        json!({"action": "hover", "path": "src/lib.rs", "line": 1}),
    )
    .await;
    assert_eq!(payload["success"], json!(false));
    assert!(
        payload["error"]
            .as_str()
            .unwrap()
            .contains("failed to start LSP server"),
        "{payload}"
    );
}

#[tokio::test]
async fn a_missing_server_binary_still_breaks_workspace_symbols() {
    let dir = tempfile::tempdir().unwrap();
    write_source(dir.path(), "fn main() {}\n").await;
    let config = LspToolConfig::default()
        .with_workspace_root(dir.path())
        .with_server(LspServerConfig::new(
            "openjiuwen-definitely-missing-lsp-server",
        ));
    let tool = LspTool::new(config);
    let payload = call(&tool, json!({"action": "workspace_symbols", "query": "x"})).await;
    assert_eq!(payload["success"], json!(false));
    let error = payload["error"].as_str().unwrap();
    assert!(error.contains("failed to start LSP server"), "{payload}");
    assert!(
        error.contains("openjiuwen-definitely-missing-lsp-server"),
        "the launch error names the command: {payload}"
    );
}

#[tokio::test]
async fn a_server_timeout_is_a_payload_error() {
    let dir = tempfile::tempdir().unwrap();
    write_source(dir.path(), "fn main() {}\n").await;
    tokio::fs::write(dir.path().join("src/slow.rs"), "fn slow() {}\n")
        .await
        .unwrap();
    let (tool, _server) = build_tool(dir.path(), Duration::from_millis(100));
    let payload = call(
        &tool,
        json!({"action": "definition", "path": "src/slow.rs", "line": 1, "column": 1}),
    )
    .await;
    assert_eq!(payload["success"], json!(false));
    assert!(
        payload["error"].as_str().unwrap().contains("timed out"),
        "{payload}"
    );
}

#[tokio::test]
async fn unsupported_methods_map_to_a_clear_payload_error() {
    let dir = tempfile::tempdir().unwrap();
    write_source(dir.path(), "fn main() {}\n").await;
    tokio::fs::write(dir.path().join("src/unsupported.rs"), "fn main() {}\n")
        .await
        .unwrap();
    let (tool, _server) = build_tool(dir.path(), Duration::from_secs(2));
    let payload = call(
        &tool,
        json!({"action": "completion", "path": "src/unsupported.rs", "line": 1}),
    )
    .await;
    assert_eq!(payload["success"], json!(false));
    assert!(
        payload["error"]
            .as_str()
            .unwrap()
            .contains("Operation not supported by this language server: textDocument/completion"),
        "{payload}"
    );
}

#[tokio::test]
async fn workspace_symbol_errors_are_wrapped() {
    let dir = tempfile::tempdir().unwrap();
    write_source(dir.path(), "fn main() {}\n").await;
    let (tool, _server) = build_tool(dir.path(), Duration::from_secs(2));
    let payload = call(
        &tool,
        json!({"action": "workspace_symbols", "query": "boom"}),
    )
    .await;
    assert_eq!(payload["success"], json!(false));
    assert!(
        payload["error"]
            .as_str()
            .unwrap()
            .contains("workspace/symbol"),
        "{payload}"
    );
}

#[tokio::test]
async fn failure_payloads_render_their_error_text() {
    let dir = tempfile::tempdir().unwrap();
    let (tool, _server) = build_tool(dir.path(), Duration::from_secs(2));
    let payload = call(&tool, json!({"action": "hover", "path": "nope.rs"})).await;
    let rendered = tool.render_for_llm(&payload);
    assert_eq!(rendered, payload["error"].as_str().unwrap());
}

#[tokio::test]
async fn the_client_is_lazily_created_once_and_reused() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_source(dir.path(), "fn main() {}\n").await;
    let (tool, server) = build_tool(dir.path(), Duration::from_secs(2));
    for _ in 0..2 {
        let payload = call(
            &tool,
            json!({"action": "hover", "path": &path, "line": 1, "column": 1}),
        )
        .await;
        assert_eq!(payload["success"], json!(true));
    }
    let initializes = server
        .methods()
        .iter()
        .filter(|method| method.as_str() == "initialize")
        .count();
    assert_eq!(initializes, 1, "one lazy client per tool");
    let opens = server
        .methods()
        .iter()
        .filter(|method| method.as_str() == "textDocument/didOpen")
        .count();
    assert_eq!(
        opens, 1,
        "didOpen is sent once; later calls reuse the document"
    );
}
