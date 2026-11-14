//! Cross-module parity tests for the tool registry, the tool cards and the rendered descriptions.
//!
//! Per-tool behaviour is covered beside each module; these tests wire the whole registry
//! together: the model-facing name order, the card shapes, the parallel-safety flags, the
//! deferred-tool discovery API, and end-to-end calls through one `build()` (so the filesystem
//! tools share their read state and the extended tools share their seams).
//!
//! Size note: this file is long by design — it is one parity suite over all 33 tools rather
//! than per-module behaviour tests, which stay beside each tool module.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::tool::{Tool, ToolContext};
use serde_json::{json, Value};
use tempfile::TempDir;

use super::{
    descriptions, discovery, goal, mcp_resources, tool_names, web, BuiltinTool, ToolRegistry,
};
use crate::permission::{
    PermissionEngine, PermissionEngineConfig, PermissionLevel, PermissionsSection,
};
use crate::workspace::Workspace;
use crate::worktree::{WorktreeManager, NOT_A_REPOSITORY_MARKER, NOT_INSTALLED_MARKER};

/// The twenty default Python tool names, in registry order.
const EXPECTED_NAMES: [&str; 20] = [
    "bash",
    "read_file",
    "write_file",
    "edit_file",
    "glob",
    "list_files",
    "grep",
    "todo_create",
    "todo_list",
    "todo_get",
    "todo_modify",
    "ask_user",
    "memory_search",
    "memory_get",
    "write_memory",
    "read_memory",
    "edit_memory",
    "coding_memory_read",
    "coding_memory_write",
    "coding_memory_edit",
];

/// The thirteen extended tool names, in registry order (twelve Python tools plus the Rust-only
/// `mcp_resource_subscription`).
const EXPECTED_EXTENDED_NAMES: [&str; 13] = [
    "fetch_webpage",
    "web_search",
    "image_metadata",
    "audio_metadata",
    "cron",
    "list_mcp_resources",
    "read_mcp_resource",
    "mcp_resource_subscription",
    "tool_search",
    "get_current_goal",
    "submit_goal_report",
    "lsp",
    "worktree",
];

/// The `required` list of each Python card; the empty slices have no `required` key at all.
const EXPECTED_REQUIRED: [(&str, &[&str]); 33] = [
    ("bash", &["command"]),
    ("read_file", &["file_path"]),
    ("write_file", &["file_path", "content"]),
    ("edit_file", &["file_path", "old_string", "new_string"]),
    ("glob", &["pattern"]),
    ("list_files", &[]),
    ("grep", &["pattern"]),
    ("todo_create", &["todos"]),
    ("todo_list", &[]),
    ("todo_get", &["id"]),
    ("todo_modify", &["action"]),
    ("ask_user", &["questions"]),
    ("memory_search", &["query"]),
    ("memory_get", &["path"]),
    ("write_memory", &["path", "content"]),
    ("read_memory", &["path"]),
    ("edit_memory", &["path", "old_text", "new_text"]),
    ("coding_memory_read", &["path"]),
    ("coding_memory_write", &["path", "content"]),
    ("coding_memory_edit", &["path", "old_text", "new_text"]),
    ("fetch_webpage", &["url"]),
    ("web_search", &["query"]),
    ("image_metadata", &["image_path"]),
    ("audio_metadata", &["audio_path"]),
    ("cron", &["action"]),
    ("list_mcp_resources", &[]),
    ("read_mcp_resource", &["uri"]),
    ("mcp_resource_subscription", &["action"]),
    ("tool_search", &["query"]),
    ("get_current_goal", &[]),
    ("submit_goal_report", &["status", "evidence"]),
    ("lsp", &["action"]),
    ("worktree", &["action"]),
];

/// Builds a registry over `dir` with English card text.
fn registry(dir: &TempDir) -> ToolRegistry {
    ToolRegistry::new(Workspace::new(dir.path())).with_language("en")
}

/// Builds a registry over `dir` mounting every built-in tool, with English card text.
fn extended_registry(dir: &TempDir) -> ToolRegistry {
    registry(dir).with_tools(BuiltinTool::EXTENDED)
}

/// Builds the full tool set of a registry over `dir`.
fn built(dir: &TempDir) -> Vec<Arc<dyn Tool>> {
    registry(dir).build()
}

/// Builds the tool set of every built-in tool over `dir`.
fn built_extended(dir: &TempDir) -> Vec<Arc<dyn Tool>> {
    extended_registry(dir).build()
}

/// Looks a built tool up by its model-facing name.
fn tool_by_name<'a>(tools: &'a [Arc<dyn Tool>], name: &str) -> &'a Arc<dyn Tool> {
    tools
        .iter()
        .find(|tool| tool.card().name == name)
        .unwrap_or_else(|| panic!("tool {name} is missing from the registry"))
}

/// Invokes a built tool with `ctx` and unwraps the payload envelope.
async fn invoke_with(
    tools: &[Arc<dyn Tool>],
    name: &str,
    inputs: Value,
    ctx: &ToolContext,
) -> Value {
    tool_by_name(tools, name)
        .invoke(inputs, ctx)
        .await
        .unwrap_or_else(|error| panic!("tool {name} returned an error: {error}"))
}

/// Invokes a built tool with a fresh, uncancelled context.
async fn invoke(tools: &[Arc<dyn Tool>], name: &str, inputs: Value) -> Value {
    invoke_with(tools, name, inputs, &ToolContext::new()).await
}

#[test]
fn tool_names_are_exactly_the_twenty_python_names() {
    assert_eq!(tool_names(), &EXPECTED_NAMES);
    assert_eq!(BuiltinTool::DEFAULT.len(), EXPECTED_NAMES.len());
    assert_eq!(BuiltinTool::ALL.len(), 33);
    assert_eq!(BuiltinTool::ALL, BuiltinTool::EXTENDED);

    let names: Vec<&str> = BuiltinTool::DEFAULT
        .iter()
        .map(|tool| tool.name())
        .collect();
    assert_eq!(names, EXPECTED_NAMES.to_vec());
    let extended: Vec<&str> = BuiltinTool::EXTENDED
        .iter()
        .map(|tool| tool.name())
        .collect();
    assert_eq!(
        extended,
        EXPECTED_NAMES
            .iter()
            .chain(EXPECTED_EXTENDED_NAMES.iter())
            .copied()
            .collect::<Vec<&str>>()
    );
}

#[test]
fn from_name_round_trips_every_name_and_rejects_unknown_tools() {
    for tool in BuiltinTool::EXTENDED {
        let found = BuiltinTool::from_name(tool.name())
            .unwrap_or_else(|| panic!("unknown name {}", tool.name()));
        assert_eq!(found, tool, "round-trip mismatch for {}", tool.name());
    }
    assert_eq!(BuiltinTool::from_name("task"), None);
    assert_eq!(BuiltinTool::from_name(""), None);
    assert_eq!(BuiltinTool::from_name("Bash"), None);
}

#[test]
fn cards_match_the_python_names_schemas_and_languages() {
    let dir = tempfile::tempdir().expect("temp workspace");
    let registry = registry(&dir);

    let cards = registry.cards("en");
    assert_eq!(cards.len(), BuiltinTool::DEFAULT.len());
    let names: Vec<&str> = cards.iter().map(|card| card.name.as_str()).collect();
    assert_eq!(names, tool_names().to_vec());

    for tool in BuiltinTool::DEFAULT {
        let card = registry.card(tool, "en");
        assert_eq!(card.name, tool.name(), "card name mismatch");

        let english = descriptions::description(tool.name(), "en");
        let chinese = descriptions::description(tool.name(), "cn");
        assert!(!english.is_empty(), "{} has no English text", tool.name());
        assert!(!chinese.is_empty(), "{} has no Chinese text", tool.name());
        assert_ne!(english, chinese, "{} has identical en/cn text", tool.name());
        assert_eq!(card.description, english, "{} card text", tool.name());
        assert_eq!(
            registry.card(tool, "cn").description,
            chinese,
            "{} Chinese card text",
            tool.name()
        );

        assert_eq!(
            card.input_params["type"],
            json!("object"),
            "{} input_params type",
            tool.name()
        );
        let properties = card.input_params["properties"]
            .as_object()
            .unwrap_or_else(|| panic!("{} properties is not an object", tool.name()));
        assert_eq!(
            properties.len(),
            properties.iter().count(),
            "{} has duplicate property names",
            tool.name()
        );

        if let Some(required) = card.input_params.get("required") {
            let required = required
                .as_array()
                .unwrap_or_else(|| panic!("{} required is not an array", tool.name()));
            for field in required {
                let field = field
                    .as_str()
                    .unwrap_or_else(|| panic!("{} required entry is not a string", tool.name()));
                assert!(
                    properties.contains_key(field),
                    "{} requires unknown field {field}",
                    tool.name()
                );
            }
        }
    }
}

#[test]
fn extended_cards_have_bilingual_descriptions() {
    let dir = tempfile::tempdir().expect("temp workspace");
    let registry = extended_registry(&dir);

    let cards = registry.cards("en");
    assert_eq!(cards.len(), BuiltinTool::EXTENDED.len());
    let names: Vec<&str> = cards.iter().map(|card| card.name.as_str()).collect();
    assert_eq!(names, super::extended_tool_names().to_vec());

    for tool in BuiltinTool::EXTENDED {
        let english = descriptions::description(tool.name(), "en");
        let chinese = descriptions::description(tool.name(), "cn");
        assert!(!english.is_empty(), "{} has no English text", tool.name());
        assert!(!chinese.is_empty(), "{} has no Chinese text", tool.name());

        let card = registry.card(tool, "en");
        assert_eq!(card.name, tool.name());
        assert_eq!(card.description, english, "{} card text", tool.name());
        assert_eq!(
            registry.card(tool, "cn").description,
            chinese,
            "{} Chinese card text",
            tool.name()
        );
    }
}

#[test]
fn required_lists_match_the_python_cards() {
    let dir = tempfile::tempdir().expect("temp workspace");
    let registry = registry(&dir);

    for (name, expected) in EXPECTED_REQUIRED {
        let tool = BuiltinTool::from_name(name).unwrap_or_else(|| panic!("unknown name {name}"));
        let card = registry.card(tool, "en");
        match card.input_params.get("required") {
            Some(Value::Array(required)) => {
                let required: Vec<&str> = required
                    .iter()
                    .map(|field| field.as_str().expect("required entry is a string"))
                    .collect();
                assert_eq!(required, expected.to_vec(), "{name} required list");
            }
            None => assert!(
                expected.is_empty(),
                "{name} has no required key; expected {expected:?}"
            ),
            Some(other) => panic!("{name} has an unexpected required value: {other}"),
        }
    }
}

#[test]
fn no_card_schema_has_a_dangling_reference() {
    let dir = tempfile::tempdir().expect("temp workspace");
    let registry = registry(&dir);

    for tool in BuiltinTool::EXTENDED {
        let card = registry.card(tool, "en");
        let serialized = serde_json::to_string(&card.input_params).expect("schema json");
        // A reference can only appear as a `"$ref"` key; the quoted form keeps prose that
        // merely mentions `$ref` (the `bash` description) from matching.
        assert!(
            !serialized.contains("\"$ref\""),
            "{} keeps a dangling $ref: {serialized}",
            tool.name()
        );
        assert!(
            !serialized.contains("\"$defs\"") && !serialized.contains("\"definitions\""),
            "{} keeps its definition map: {serialized}",
            tool.name()
        );

        let properties = card.input_params["properties"]
            .as_object()
            .unwrap_or_else(|| panic!("{} properties is not an object", tool.name()));
        if let Some(required) = card.input_params.get("required") {
            let required = required
                .as_array()
                .unwrap_or_else(|| panic!("{} required is not an array", tool.name()));
            for field in required {
                let field = field
                    .as_str()
                    .unwrap_or_else(|| panic!("{} required entry is not a string", tool.name()));
                assert!(
                    properties.contains_key(field),
                    "{} requires unknown field {field}",
                    tool.name()
                );
            }
        }
    }
}

#[test]
fn parallel_safe_flags_match_the_registry() {
    for tool in BuiltinTool::EXTENDED {
        let expected = !matches!(
            tool,
            BuiltinTool::WriteFile
                | BuiltinTool::EditFile
                | BuiltinTool::Cron
                | BuiltinTool::SubmitGoalReport
                | BuiltinTool::McpResourceSubscription
                | BuiltinTool::Worktree
        );
        assert_eq!(tool.parallel_safe(), expected, "for {}", tool.name());
    }
    for name in [
        "write_file",
        "edit_file",
        "cron",
        "submit_goal_report",
        "mcp_resource_subscription",
        "worktree",
    ] {
        let tool = BuiltinTool::from_name(name).expect("known tool");
        assert!(!tool.parallel_safe(), "{name} must be serialised");
    }

    // The mounted default tools report the same flag through the built tool and its card.
    let dir = tempfile::tempdir().expect("temp workspace");
    let tools = built(&dir);
    for tool in BuiltinTool::DEFAULT {
        let built_tool = tool_by_name(&tools, tool.name());
        assert_eq!(
            built_tool.is_parallel_safe(),
            tool.parallel_safe(),
            "{} built parallel safety",
            tool.name()
        );
        assert_eq!(
            built_tool.card().parallel_safe,
            tool.parallel_safe(),
            "{} card parallel safety",
            tool.name()
        );
    }
}

#[test]
fn registry_builds_every_tool_in_registry_order() {
    let dir = tempfile::tempdir().expect("temp workspace");
    let registry = registry(&dir);

    let tools = registry.build();
    assert_eq!(tools.len(), EXPECTED_NAMES.len());
    let names: Vec<&str> = tools.iter().map(|tool| tool.card().name.as_str()).collect();
    assert_eq!(names, tool_names().to_vec());
    assert_eq!(registry.cards("en").len(), EXPECTED_NAMES.len());
    assert!(registry.sys_operation().is_some(), "local backend missing");
    assert_eq!(registry.tools().len(), EXPECTED_NAMES.len());

    // The explicit backend wiring produces the same tool set as the derived local backend.
    let operation = openjiuwen_sysop::SysOperation::local(
        openjiuwen_sysop::LocalWorkConfig::unrestricted(dir.path().to_string_lossy()),
    )
    .expect("local sys operation");
    let explicit = registry.with_sys_operation(Arc::new(operation));
    assert_eq!(explicit.build().len(), EXPECTED_NAMES.len());
}

#[test]
fn with_tools_restricts_the_registry_and_keeps_order() {
    let dir = tempfile::tempdir().expect("temp workspace");
    let registry = registry(&dir).with_tools([BuiltinTool::Bash, BuiltinTool::ReadFile]);

    let tools = registry.build();
    let names: Vec<&str> = tools.iter().map(|tool| tool.card().name.as_str()).collect();
    assert_eq!(names, vec!["bash", "read_file"]);
    assert_eq!(registry.cards("en").len(), 2);
    assert_eq!(
        registry.tools(),
        &[BuiltinTool::Bash, BuiltinTool::ReadFile]
    );
}

#[tokio::test]
async fn read_then_edit_round_trips_through_the_registry() {
    let dir = tempfile::tempdir().expect("temp workspace");
    std::fs::write(dir.path().join("notes.txt"), "alpha\nbeta\ngamma\n").expect("fixture write");
    let tools = built(&dir);

    let read = invoke(&tools, "read_file", json!({"file_path": "notes.txt"})).await;
    assert_eq!(read["success"], true, "read failed: {read}");
    let content = read["data"]["content"].as_str().expect("content string");
    assert!(content.contains("     1\talpha"), "unexpected: {content}");
    assert_eq!(read["data"]["line_count"], json!(3));

    let edited = invoke(
        &tools,
        "edit_file",
        json!({
            "file_path": "notes.txt",
            "old_string": "beta",
            "new_string": "delta",
        }),
    )
    .await;
    assert_eq!(edited["success"], true, "edit failed: {edited}");
    assert_eq!(edited["data"]["replacements"], json!(1));
    assert_eq!(
        std::fs::read_to_string(dir.path().join("notes.txt")).expect("read back"),
        "alpha\ndelta\ngamma\n"
    );
}

#[tokio::test]
async fn write_file_creates_a_new_file() {
    let dir = tempfile::tempdir().expect("temp workspace");
    let tools = built(&dir);

    let out = invoke(
        &tools,
        "write_file",
        json!({"file_path": "fresh.txt", "content": "hello\n"}),
    )
    .await;

    assert_eq!(out["success"], true, "write failed: {out}");
    assert_eq!(out["data"]["type"], json!("create"));
    assert_eq!(out["data"]["created"], json!(true));
    assert_eq!(out["data"]["bytes_written"], json!(6));
    assert_eq!(
        std::fs::read_to_string(dir.path().join("fresh.txt")).expect("read back"),
        "hello\n"
    );
}

#[tokio::test]
async fn glob_list_files_and_grep_find_the_written_files() {
    let dir = tempfile::tempdir().expect("temp workspace");
    std::fs::create_dir(dir.path().join("nested")).expect("nested dir");
    std::fs::write(dir.path().join("alpha.txt"), "one needle here\n").expect("fixture write");
    std::fs::write(dir.path().join("nested").join("beta.txt"), "haystack\n")
        .expect("fixture write");
    let tools = built(&dir);

    let glob = invoke(&tools, "glob", json!({"pattern": "**/*.txt"})).await;
    assert_eq!(glob["success"], true, "glob failed: {glob}");
    assert_eq!(glob["data"]["numFiles"], json!(2));
    assert_eq!(glob["data"]["count"], json!(2));
    assert_eq!(glob["data"]["truncated"], json!(false));
    let filenames = glob["data"]["filenames"]
        .as_array()
        .expect("filenames array");
    assert!(
        filenames.iter().any(|name| name == "alpha.txt"),
        "unexpected: {filenames:?}"
    );
    assert!(
        filenames.iter().any(|name| name == "nested/beta.txt"),
        "unexpected: {filenames:?}"
    );

    let list = invoke(&tools, "list_files", json!({})).await;
    assert_eq!(list["success"], true, "list failed: {list}");
    assert_eq!(list["data"]["files"], json!(["alpha.txt"]));
    assert_eq!(list["data"]["dirs"], json!(["nested"]));

    let grep = invoke(&tools, "grep", json!({"pattern": "needle"})).await;
    assert_eq!(grep["success"], true, "grep failed: {grep}");
    let stdout = grep["data"]["stdout"].as_str().expect("stdout string");
    assert!(stdout.contains("alpha.txt"), "unexpected: {stdout}");
    assert!(stdout.contains("needle"), "unexpected: {stdout}");

    let files = invoke(
        &tools,
        "grep",
        json!({"pattern": "haystack", "output_mode": "files_with_matches"}),
    )
    .await;
    assert_eq!(files["success"], true, "grep failed: {files}");
    assert_eq!(files["data"]["filenames"], json!(["nested/beta.txt"]));
}

#[tokio::test]
async fn todos_persist_across_registries_for_the_same_session() {
    let dir = tempfile::tempdir().expect("temp workspace");
    let ctx = ToolContext::new().with_session("s1");
    let tools = built(&dir);

    let created = invoke_with(
        &tools,
        "todo_create",
        json!({
            "todos": [
                {
                    "id": "first",
                    "content": "Write tests",
                    "activeForm": "Writing tests",
                    "description": "Add the cross-module suite",
                },
                {
                    "id": "second",
                    "content": "Run tests",
                    "activeForm": "Running tests",
                    "description": "Run the workspace suite",
                }
            ]
        }),
        &ctx,
    )
    .await;
    assert_eq!(created["success"], true, "todo_create failed: {created}");

    let listed = invoke_with(&tools, "todo_list", json!({}), &ctx).await;
    assert_eq!(listed["success"], true, "todo_list failed: {listed}");
    let todos = listed["data"]["todos"].as_array().expect("todos array");
    assert_eq!(todos.len(), 2);
    assert_eq!(todos[0]["id"], json!("first"));
    assert_eq!(todos[0]["status"], json!("in_progress"));
    assert_eq!(todos[0]["content"], json!("Write tests"));
    assert_eq!(todos[1]["id"], json!("second"));
    assert_eq!(todos[1]["status"], json!("pending"));

    let persisted = dir.path().join("todo").join("s1").join("todo.json");
    assert!(persisted.is_file(), "missing {}", persisted.display());

    // A second registry over the same workspace reads the same session file.
    let second = built(&dir);
    let relisted = invoke_with(&second, "todo_list", json!({}), &ctx).await;
    assert_eq!(relisted["data"]["todos"], listed["data"]["todos"]);

    // An unrelated session starts empty and does not see the first session's items.
    let other_ctx = ToolContext::new().with_session("s2");
    let other = invoke_with(&tools, "todo_list", json!({}), &other_ctx).await;
    assert_eq!(other["data"]["todos"], json!([]));
}

#[tokio::test]
async fn bash_runs_echo_through_the_local_backend() {
    let dir = tempfile::tempdir().expect("temp workspace");
    let tools = built(&dir);

    let out = invoke(&tools, "bash", json!({"command": "echo hi"})).await;
    assert_eq!(out["success"], true, "bash failed: {out}");
    let content = out["data"]["content"].as_str().expect("content string");
    assert!(content.contains("hi"), "unexpected: {content}");
    assert!(content.contains("Exit Code: 0"), "unexpected: {content}");
}

#[tokio::test]
async fn bash_background_renders_the_pid_line() {
    let dir = tempfile::tempdir().expect("temp workspace");
    let tools = built(&dir);

    let out = invoke(
        &tools,
        "bash",
        json!({"command": "sleep 1", "run_in_background": true}),
    )
    .await;
    assert_eq!(out["success"], true, "background bash failed: {out}");
    assert_eq!(out["data"]["status"], json!("started"));
    assert!(out["data"]["pid"].as_u64().is_some(), "missing pid: {out}");

    let rendered = tool_by_name(&tools, "bash").render_for_llm(&out);
    assert!(
        rendered.starts_with("Command started in background (pid "),
        "unexpected: {rendered}"
    );
    assert!(rendered.ends_with(")."), "unexpected: {rendered}");
}

#[tokio::test]
async fn empty_results_render_the_python_messages() {
    let dir = tempfile::tempdir().expect("temp workspace");
    let tools = built(&dir);

    let glob = invoke(&tools, "glob", json!({"pattern": "**/*.txt"})).await;
    assert_eq!(glob["success"], true, "glob failed: {glob}");
    assert_eq!(
        tool_by_name(&tools, "glob").render_for_llm(&glob),
        "No files found."
    );

    let list = invoke(&tools, "list_files", json!({})).await;
    assert_eq!(list["success"], true, "list failed: {list}");
    assert_eq!(
        tool_by_name(&tools, "list_files").render_for_llm(&list),
        "No entries found."
    );

    let grep = invoke(&tools, "grep", json!({"pattern": "needle"})).await;
    assert_eq!(grep["success"], true, "grep failed: {grep}");
    assert_eq!(
        tool_by_name(&tools, "grep").render_for_llm(&grep),
        "No matches found."
    );

    let ctx = ToolContext::new().with_session("empty");
    let todos = invoke_with(&tools, "todo_list", json!({}), &ctx).await;
    assert_eq!(todos["success"], true, "todo_list failed: {todos}");
    assert_eq!(
        tool_by_name(&tools, "todo_list").render_for_llm(&todos),
        "No active tasks."
    );
}

#[tokio::test]
async fn write_and_edit_results_render_the_python_messages() {
    let dir = tempfile::tempdir().expect("temp workspace");
    let tools = built(&dir);
    let write_tool = tool_by_name(&tools, "write_file");

    let created = invoke(
        &tools,
        "write_file",
        json!({"file_path": "rendered.txt", "content": "alpha\n"}),
    )
    .await;
    assert_eq!(created["success"], true, "write failed: {created}");
    let path = created["data"]["file_path"]
        .as_str()
        .expect("file_path string")
        .to_owned();
    assert_eq!(
        write_tool.render_for_llm(&created),
        format!("Created {path} (6 bytes written).")
    );

    let updated = invoke(
        &tools,
        "write_file",
        json!({"file_path": "rendered.txt", "content": "alpha\nbeta\n"}),
    )
    .await;
    assert_eq!(updated["success"], true, "update failed: {updated}");
    assert_eq!(updated["data"]["type"], json!("update"));
    assert_eq!(
        write_tool.render_for_llm(&updated),
        format!("Updated {path} (11 bytes written).")
    );

    let read = invoke(&tools, "read_file", json!({"file_path": "rendered.txt"})).await;
    assert_eq!(read["success"], true, "read failed: {read}");

    let edited = invoke(
        &tools,
        "edit_file",
        json!({"file_path": "rendered.txt", "old_string": "beta", "new_string": "gamma"}),
    )
    .await;
    assert_eq!(edited["success"], true, "edit failed: {edited}");
    assert_eq!(
        tool_by_name(&tools, "edit_file").render_for_llm(&edited),
        format!("Edited {path} (1 replacement(s)).")
    );
}

#[tokio::test]
async fn ask_user_without_a_handler_reports_the_python_failure() {
    let dir = tempfile::tempdir().expect("temp workspace");
    let tools = built(&dir);

    let out = invoke(
        &tools,
        "ask_user",
        json!({
            "questions": [{
                "header": "Confirm",
                "question": "Proceed?",
                "options": [{"label": "Yes", "description": "Go ahead"}]
            }]
        }),
    )
    .await;

    assert_eq!(out["success"], false);
    assert_eq!(
        out["error"],
        "ask_user is not available: no user interaction handler is configured"
    );
    assert_eq!(out["data"], Value::Null);
}

#[tokio::test]
async fn memory_search_without_a_backend_reports_the_python_failure() {
    let dir = tempfile::tempdir().expect("temp workspace");
    let tools = built(&dir);

    let out = invoke(&tools, "memory_search", json!({"query": "anything"})).await;

    assert_eq!(out["success"], false);
    assert_eq!(out["error"], "memory tool backend is not configured");
    assert_eq!(out["data"], Value::Null);
}

/// An [`web::HttpFetcher`] returning one canned HTML page.
struct StaticFetcher {
    /// The HTML body handed back for every request.
    body: Vec<u8>,
}

#[async_trait]
impl web::HttpFetcher for StaticFetcher {
    async fn fetch(&self, request: web::FetchRequest) -> Result<web::FetchResponse> {
        if !request.url.starts_with("https://") {
            return Err(Error::transport("unexpected scheme"));
        }
        Ok(web::FetchResponse {
            status: 200,
            final_url: request.url,
            headers: vec![(
                "content-type".to_owned(),
                "text/html; charset=utf-8".to_owned(),
            )],
            body: self.body.clone(),
        })
    }
}

/// A minimal 2x2 RGBA PNG header (`IHDR` only), built inline so no fixture is shared.
fn png_header() -> Vec<u8> {
    let mut bytes = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    bytes.extend_from_slice(&13u32.to_be_bytes());
    bytes.extend_from_slice(b"IHDR");
    bytes.extend_from_slice(&2u32.to_be_bytes());
    bytes.extend_from_slice(&2u32.to_be_bytes());
    bytes.extend_from_slice(&[8, 6, 0, 0, 0]);
    bytes.extend_from_slice(&[0, 0, 0, 0]);
    bytes
}

/// A minimal 1 kHz 16-bit mono WAV header with a one-second `data` chunk.
fn wav_header() -> Vec<u8> {
    let mut body = b"WAVE".to_vec();
    body.extend_from_slice(b"fmt ");
    body.extend_from_slice(&16u32.to_le_bytes());
    for value in [1u16, 1, 0, 0] {
        let _ = value;
    }
    body.extend_from_slice(&1u16.to_le_bytes());
    body.extend_from_slice(&1u16.to_le_bytes());
    body.extend_from_slice(&1_000u32.to_le_bytes());
    body.extend_from_slice(&2_000u32.to_le_bytes());
    body.extend_from_slice(&2u16.to_le_bytes());
    body.extend_from_slice(&16u16.to_le_bytes());
    body.extend_from_slice(b"data");
    body.extend_from_slice(&2_000u32.to_le_bytes());
    let mut bytes = b"RIFF".to_vec();
    bytes.extend_from_slice(&u32::try_from(body.len()).unwrap_or_default().to_le_bytes());
    bytes.extend_from_slice(&body);
    bytes
}

#[test]
fn default_registry_excludes_extended_tools() {
    let dir = tempfile::tempdir().expect("temp workspace");
    let registry = registry(&dir);
    let tools = registry.build();
    assert_eq!(tools.len(), 20);
    for name in [
        "fetch_webpage",
        "web_search",
        "cron",
        "submit_goal_report",
        "lsp",
        "worktree",
    ] {
        assert!(
            tools.iter().all(|tool| tool.card().name != name),
            "{name} must not be mounted by default"
        );
        assert!(registry.cards("en").iter().all(|card| card.name != name));
    }
}

#[test]
fn extended_registry_builds_every_tool_in_registry_order() {
    let dir = tempfile::tempdir().expect("temp workspace");
    let registry = extended_registry(&dir);
    let tools = registry.build();
    assert_eq!(tools.len(), 33);
    let names: Vec<&str> = tools.iter().map(|tool| tool.card().name.as_str()).collect();
    assert_eq!(
        names,
        BuiltinTool::EXTENDED
            .iter()
            .map(|tool| tool.name())
            .collect::<Vec<&str>>()
    );
    assert_eq!(registry.cards("en").len(), 33);
    assert_eq!(registry.tools(), &BuiltinTool::EXTENDED);
}

#[tokio::test]
async fn fetch_webpage_uses_the_injected_fetcher() {
    let dir = tempfile::tempdir().expect("temp workspace");
    let registry = extended_registry(&dir).with_http_fetcher(Arc::new(StaticFetcher {
        body: b"<html><head><title>Doc</title></head><body><p>Hello</p></body></html>".to_vec(),
    }));
    assert!(registry.http_fetcher().is_some());
    let tools = registry.build();
    let out = invoke(
        &tools,
        "fetch_webpage",
        json!({"url": "https://example.com"}),
    )
    .await;
    assert_eq!(out["success"], true, "fetch failed: {out}");
    assert_eq!(out["data"]["title"], json!("Doc"));
    assert_eq!(out["data"]["content"], json!("Hello"));
    assert_eq!(out["data"]["status"], json!(200));
}

#[tokio::test]
async fn web_search_returns_the_provider_results() {
    let dir = tempfile::tempdir().expect("temp workspace");
    let backend = web::FakeSearchBackend::with_results(vec![web::SearchResult::new(
        "Rust",
        "https://rust.example",
        "a language",
    )]);
    let registry = extended_registry(&dir).with_search_backend(Arc::new(backend));
    assert!(registry.search_backend().is_some());
    let tools = registry.build();
    let out = invoke(&tools, "web_search", json!({"query": "rust"})).await;
    assert_eq!(out["success"], true, "search failed: {out}");
    assert_eq!(out["data"]["success"], true);
    assert_eq!(out["data"]["results"][0]["title"], json!("Rust"));
    assert_eq!(
        out["data"]["results"][0]["url"],
        json!("https://rust.example")
    );
    assert_eq!(out["data"]["results"][0]["snippet"], json!("a language"));
}

#[tokio::test]
async fn media_tools_report_the_image_and_audio_headers() {
    let dir = tempfile::tempdir().expect("temp workspace");
    std::fs::write(dir.path().join("pic.png"), png_header()).expect("png fixture");
    std::fs::write(dir.path().join("clip.wav"), wav_header()).expect("wav fixture");
    let tools = built_extended(&dir);
    let image = invoke(&tools, "image_metadata", json!({"image_path": "pic.png"})).await;
    assert_eq!(image["success"], true, "image failed: {image}");
    assert!(image["data"].is_object(), "unexpected: {image}");
    let audio = invoke(&tools, "audio_metadata", json!({"audio_path": "clip.wav"})).await;
    assert_eq!(audio["success"], true, "audio failed: {audio}");
    assert!(audio["data"].is_object(), "unexpected: {audio}");
}

#[tokio::test]
async fn cron_creates_lists_and_cancels_a_job() {
    let dir = tempfile::tempdir().expect("temp workspace");
    let tools = built_extended(&dir);
    let created = invoke(
        &tools,
        "cron",
        json!({"action": "create", "cron": "0 9 * * *", "prompt": "morning"}),
    )
    .await;
    assert_eq!(created["success"], true, "create failed: {created}");
    let job_id = created["data"]["job"]["id"]
        .as_str()
        .expect("job id")
        .to_owned();
    let listed = invoke(&tools, "cron", json!({"action": "list"})).await;
    assert_eq!(listed["success"], true, "list failed: {listed}");
    assert_eq!(listed["data"]["jobs"].as_array().map(Vec::len), Some(1));
    let cancelled = invoke(
        &tools,
        "cron",
        json!({"action": "cancel", "job_id": job_id}),
    )
    .await;
    assert_eq!(cancelled["success"], true, "cancel failed: {cancelled}");
    assert_eq!(cancelled["data"]["job_id"], json!(job_id));
    let empty = invoke(&tools, "cron", json!({"action": "list"})).await;
    assert_eq!(empty["data"]["jobs"].as_array().map(Vec::len), Some(0));
}

#[tokio::test]
async fn goal_tools_round_trip_a_report() {
    let dir = tempfile::tempdir().expect("temp workspace");
    let workspace = Workspace::new(dir.path());
    let store = goal::GoalStore::for_workspace(&workspace);
    store
        .start("ship the release", chrono::Utc::now())
        .await
        .expect("goal start");
    let tools = extended_registry(&dir).with_goal_store(store).build();

    let empty_dir = tempfile::tempdir().expect("second workspace");
    let empty = built_extended(&empty_dir);
    let no_goal = invoke(&empty, "get_current_goal", json!({})).await;
    assert_eq!(no_goal["success"], true, "goal read failed: {no_goal}");
    assert_eq!(no_goal["data"]["goal"], Value::Null);

    let current = invoke(&tools, "get_current_goal", json!({})).await;
    assert_eq!(current["success"], true, "goal read failed: {current}");
    assert_eq!(
        current["data"]["goal"]["objective"],
        json!("ship the release")
    );

    let reported = invoke(
        &tools,
        "submit_goal_report",
        json!({"status": "complete", "evidence": "shipped"}),
    )
    .await;
    assert_eq!(reported["success"], true, "report failed: {reported}");
    assert_eq!(reported["data"]["status"], json!("complete"));

    let after = invoke(&tools, "get_current_goal", json!({})).await;
    assert_eq!(
        after["data"]["goal"]["last_assessment"]["status"],
        json!("complete")
    );
}

#[tokio::test]
async fn mcp_resource_tools_use_the_provider() {
    let dir = tempfile::tempdir().expect("temp workspace");
    let provider = mcp_resources::FakeMcpResourceProvider::new();
    provider.set_resources(vec![mcp_resources::McpResource {
        uri: "mem://notes".to_owned(),
        name: "notes".to_owned(),
        mime_type: Some("text/markdown".to_owned()),
        description: None,
    }]);
    provider.set_resource(mcp_resources::McpResourceContent {
        uri: "mem://notes".to_owned(),
        mime_type: Some("text/markdown".to_owned()),
        text: Some("# notes".to_owned()),
        blob_base64: None,
        byte_size: 7,
    });
    let tools = extended_registry(&dir)
        .with_mcp_resources(Arc::new(provider))
        .build();

    let listed = invoke(&tools, "list_mcp_resources", json!({})).await;
    assert_eq!(listed["success"], true, "list failed: {listed}");
    assert_eq!(listed["data"]["resources"][0]["uri"], json!("mem://notes"));
    let read = invoke(&tools, "read_mcp_resource", json!({"uri": "mem://notes"})).await;
    assert_eq!(read["success"], true, "read failed: {read}");
    assert_eq!(read["data"]["resource"]["text"], json!("# notes"));
}

#[tokio::test]
async fn mcp_resource_tools_without_a_provider_report_the_typed_failure() {
    let dir = tempfile::tempdir().expect("temp workspace");
    let tools = built_extended(&dir);
    let listed = invoke(&tools, "list_mcp_resources", json!({})).await;
    assert_eq!(listed["success"], false);
    assert_eq!(
        listed["error"],
        json!("mcp resource provider is not configured")
    );
    let read = invoke(&tools, "read_mcp_resource", json!({"uri": "mem://x"})).await;
    assert_eq!(read["success"], false);
    assert_eq!(
        read["error"],
        json!("mcp resource provider is not configured")
    );
}

#[tokio::test]
async fn mcp_resource_subscription_tool_uses_the_registry() {
    let dir = tempfile::tempdir().expect("temp workspace");
    let provider = Arc::new(super::mcp_subscriptions::FakeMcpResourceSubscriptionProvider::new());
    let subscriptions = Arc::new(
        super::mcp_subscriptions::McpResourceSubscriptions::new()
            .with_provider(provider.clone())
            .with_capacity(4),
    );
    let tools = extended_registry(&dir)
        .with_mcp_subscriptions(subscriptions)
        .build();

    let subscribed = invoke(
        &tools,
        "mcp_resource_subscription",
        json!({"action": "subscribe", "uri": "mem://notes"}),
    )
    .await;
    assert_eq!(
        subscribed["success"], true,
        "subscribe failed: {subscribed}"
    );
    assert_eq!(subscribed["data"]["subscription"]["capacity"], json!(4));

    assert!(provider.push_notification("mem://notes", "changed"));
    let polled = invoke(
        &tools,
        "mcp_resource_subscription",
        json!({"action": "poll", "uri": "mem://notes"}),
    )
    .await;
    assert_eq!(polled["data"]["count"], json!(1));
    assert_eq!(polled["data"]["events"][0]["reason"], "changed");

    let listed = invoke(
        &tools,
        "mcp_resource_subscription",
        json!({"action": "list"}),
    )
    .await;
    assert_eq!(listed["data"]["count"], json!(1));
    assert_eq!(listed["data"]["capacity"], json!(4));

    let closed = invoke(
        &tools,
        "mcp_resource_subscription",
        json!({"action": "unsubscribe", "uri": "mem://notes"}),
    )
    .await;
    assert_eq!(closed["success"], true, "unsubscribe failed: {closed}");
    assert_eq!(provider.unsubscribe_calls(), vec!["mem://notes".to_owned()]);
}

#[tokio::test]
async fn mcp_resource_subscription_tool_without_a_registry_reports_the_typed_failure() {
    let dir = tempfile::tempdir().expect("temp workspace");
    let tools = built_extended(&dir);
    let payload = invoke(
        &tools,
        "mcp_resource_subscription",
        json!({"action": "list"}),
    )
    .await;
    assert_eq!(payload["success"], false);
    let error = payload["error"].as_str().expect("error text");
    assert!(
        error.contains("no subscription registry is configured"),
        "error: {error}"
    );
}

#[test]
fn mcp_resource_subscription_card_is_mounted_and_serialisable() {
    let dir = tempfile::tempdir().expect("temp workspace");
    let registry = extended_registry(&dir);
    let card = registry.card(BuiltinTool::McpResourceSubscription, "en");
    assert_eq!(card.name, "mcp_resource_subscription");
    assert!(!card.parallel_safe);
    assert_eq!(card.input_params["required"], json!(["action"]));
    assert_eq!(
        card.description,
        descriptions::description("mcp_resource_subscription", "en")
    );
    assert!(registry.mcp_subscriptions().is_none());
    let subscriptions = Arc::new(super::mcp_subscriptions::McpResourceSubscriptions::new());
    assert!(registry
        .with_mcp_subscriptions(subscriptions)
        .mcp_subscriptions()
        .is_some());
}

#[tokio::test]
async fn tool_search_finds_the_deferred_tool() {
    let dir = tempfile::tempdir().expect("temp workspace");
    let tools = extended_registry(&dir)
        .with_deferred_tools([BuiltinTool::FetchWebpage, BuiltinTool::Cron])
        .build();
    let out = invoke(&tools, "tool_search", json!({"query": "fetch web page"})).await;
    assert_eq!(out["success"], true, "search failed: {out}");
    assert_eq!(out["data"]["results"][0]["name"], json!("fetch_webpage"));
    assert_eq!(out["data"]["count"], json!(1));
}

#[test]
fn deferred_tools_are_hidden_until_discovered() {
    let dir = tempfile::tempdir().expect("temp workspace");
    let registry =
        registry(&dir).with_deferred_tools([BuiltinTool::FetchWebpage, BuiltinTool::Cron]);

    assert_eq!(
        registry.deferred_tools(),
        &[BuiltinTool::FetchWebpage, BuiltinTool::Cron]
    );
    let tools = registry.build();
    assert_eq!(tools.len(), 20);
    for name in ["fetch_webpage", "cron"] {
        assert!(tools.iter().all(|tool| tool.card().name != name));
        assert!(registry.cards("en").iter().all(|card| card.name != name));
    }

    let hits = registry.discover("fetch a web page", 5);
    assert_eq!(
        hits.first().map(|hit| hit.name.as_str()),
        Some("fetch_webpage")
    );
    let revealed = registry.reveal(&["fetch_webpage"]);
    assert_eq!(revealed.len(), 1);
    assert_eq!(revealed[0].name, "fetch_webpage");
    assert_eq!(
        revealed[0].description,
        descriptions::description("fetch_webpage", "en")
    );
    let _: &discovery::ToolSearchHit = &hits[0];

    let deferred = registry.deferred_cards("en");
    assert_eq!(deferred.len(), 2);
    assert_eq!(
        deferred
            .iter()
            .map(|card| card.name.as_str())
            .collect::<Vec<&str>>(),
        vec!["fetch_webpage", "cron"]
    );
    assert_eq!(registry.reveal(&["nope"]).len(), 0);
}

#[test]
fn extended_names_match_the_variant_table() {
    let names: Vec<&str> = BuiltinTool::EXTENDED
        .iter()
        .map(|tool| tool.name())
        .collect();
    assert_eq!(names, super::extended_tool_names().to_vec());
    assert_eq!(BuiltinTool::ALL.len(), 33);
    for (index, name) in super::extended_tool_names().iter().enumerate() {
        assert_eq!(
            BuiltinTool::EXTENDED[index].name(),
            *name,
            "name at {index}"
        );
        assert_eq!(
            BuiltinTool::from_name(name),
            Some(BuiltinTool::EXTENDED[index]),
            "from_name({name})"
        );
    }
}

#[test]
fn lsp_and_worktree_cards_are_registry_members() {
    let dir = tempfile::tempdir().expect("temp workspace");
    let registry = extended_registry(&dir);

    for (tool, name, parallel_safe) in [
        (BuiltinTool::Lsp, "lsp", true),
        (BuiltinTool::Worktree, "worktree", false),
    ] {
        let card = registry.card(tool, "en");
        assert_eq!(card.name, name);
        assert_eq!(card.id.len(), 32, "{name} card id is a uuid hex");
        assert!(card.id.chars().all(|c| c.is_ascii_hexdigit()), "{name} id");
        assert_eq!(card.parallel_safe, parallel_safe, "{name} parallel safety");
        assert_eq!(card.input_params["type"], json!("object"), "{name} schema");
        let properties = card.input_params["properties"]
            .as_object()
            .unwrap_or_else(|| panic!("{name} properties is not an object"));
        assert!(properties.contains_key("action"), "{name} has no action");
        assert_eq!(
            card.input_params["required"],
            json!(["action"]),
            "{name} required"
        );
        assert_eq!(card.description, descriptions::description(name, "en"));
        assert_eq!(
            registry.card(tool, "cn").description,
            descriptions::description(name, "cn")
        );
    }

    let tools = registry.build();
    let names: Vec<&str> = tools.iter().map(|tool| tool.card().name.as_str()).collect();
    assert!(names.contains(&"lsp"), "lsp is not mounted: {names:?}");
    assert!(
        names.contains(&"worktree"),
        "worktree is not mounted: {names:?}"
    );
}

#[tokio::test]
async fn lsp_without_a_configuration_reports_the_missing_server() {
    let dir = tempfile::tempdir().expect("temp workspace");
    std::fs::write(dir.path().join("main.rs"), "fn main() {}\n").expect("fixture");
    let registry = extended_registry(&dir);
    assert!(registry.lsp_config().is_none(), "no LSP config is wired");

    let tools = registry.build();
    let out = invoke(
        &tools,
        "lsp",
        json!({"action": "hover", "path": "main.rs", "line": 1, "column": 1}),
    )
    .await;

    assert_eq!(out["success"], false, "unexpected success: {out}");
    let error = out["error"].as_str().expect("error string");
    assert!(
        error.contains("no LSP server configured"),
        "the failure must name the missing configuration: {error}"
    );
    assert!(error.contains("LspToolConfig::server"), "{error}");
}

#[tokio::test]
async fn a_configured_lsp_server_is_reached_by_the_built_tool() {
    let dir = tempfile::tempdir().expect("temp workspace");
    // A command that cannot exist keeps the test offline: the spawn failure proves the built
    // tool used the configured server descriptor instead of the "not configured" default.
    let config = crate::lsp::LspToolConfig::default()
        .with_language("en")
        .with_server(crate::lsp::LspServerConfig::new(
            "openjiuwen-definitely-missing-lsp-server",
        ));
    let registry = extended_registry(&dir).with_lsp_config(config);
    assert!(registry.lsp_config().is_some(), "the config is not stored");

    let tools = registry.build();
    let out = invoke(
        &tools,
        "lsp",
        json!({"action": "workspace_symbols", "query": "x"}),
    )
    .await;

    assert_eq!(out["success"], false, "unexpected success: {out}");
    let error = out["error"].as_str().expect("error string");
    assert!(
        error.contains("failed to start LSP server"),
        "the configured command was not used: {error}"
    );
    assert!(
        error.contains("openjiuwen-definitely-missing-lsp-server"),
        "{error}"
    );
}

/// A minimal in-process language server answering the `initialize` and hover requests.
///
/// Spawned over one half of a [`tokio::io::duplex`] pair by
/// [`crate::lsp::ChannelTransport`], so no language server binary is needed.
fn spawn_fake_lsp(stream: tokio::io::DuplexStream) -> tokio::task::JoinHandle<()> {
    use crate::lsp::protocol::{error_response, response_message, write_message, MessageReader};

    tokio::spawn(async move {
        let (read, mut write) = tokio::io::split(stream);
        let mut reader = MessageReader::new(read);
        while let Ok(Some(message)) = reader.next().await {
            let Some(id) = message.get("id").cloned() else {
                continue;
            };
            let method = message
                .get("method")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let response = match method {
                "initialize" => {
                    response_message(id, json!({"capabilities": {"hoverProvider": true}}))
                }
                "textDocument/hover" => response_message(
                    id,
                    json!({
                        "contents": { "kind": "markdown", "value": "**fake hover**" },
                        "range": {
                            "start": { "line": 0, "character": 0 },
                            "end": { "line": 0, "character": 1 },
                        },
                    }),
                ),
                "shutdown" => response_message(id, Value::Null),
                other => error_response(id, &crate::lsp::LspError::new(-32601, other)),
            };
            if write_message(&mut write, &response).await.is_err() {
                break;
            }
        }
    })
}

#[tokio::test]
async fn lsp_tool_with_an_injected_transport_reaches_the_client() {
    let dir = tempfile::tempdir().expect("temp workspace");
    std::fs::write(dir.path().join("main.rs"), "fn main() {}\n").expect("fixture write");
    let (transport, peer) = crate::lsp::ChannelTransport::new(64 * 1024);
    let server = spawn_fake_lsp(peer);
    // The registry builds exactly this shape of tool (`LspTool::new(config)`), so the injected
    // transport proves the model-facing call reaches the JSON-RPC client.
    let config = crate::lsp::LspToolConfig::default()
        .with_workspace_root(dir.path())
        .with_language("en")
        .with_language_id("rust");
    let tool = crate::lsp::LspTool::new(config).with_transport(Box::new(transport));

    let out = tool
        .invoke(
            json!({"action": "hover", "path": "main.rs", "line": 1, "column": 1}),
            &ToolContext::new(),
        )
        .await
        .expect("lsp invoke returns a payload");

    assert_eq!(out["success"], true, "hover failed: {out}");
    assert_eq!(out["data"]["action"], json!("hover"));
    assert!(
        out["data"]["file_path"]
            .as_str()
            .unwrap_or_default()
            .ends_with("main.rs"),
        "unexpected file path: {out}"
    );
    assert_eq!(out["data"]["result"], json!("**fake hover**"));
    server.abort();
}

/// A [`crate::worktree::GitRunner`] replaying queued responses, so no git binary is needed.
struct ScriptedGitRunner {
    /// Queued `Ok(stdout)`/`Err(stderr)` responses, oldest first.
    responses: std::sync::Mutex<std::collections::VecDeque<std::result::Result<String, String>>>,
}

impl ScriptedGitRunner {
    /// Builds a runner with no queued response.
    fn new() -> Self {
        Self {
            responses: std::sync::Mutex::new(std::collections::VecDeque::new()),
        }
    }

    /// Queues a successful invocation printing `stdout`.
    fn push_ok(&self, stdout: &str) {
        self.responses
            .lock()
            .expect("runner lock")
            .push_back(Ok(stdout.to_owned()));
    }
}

#[async_trait]
impl crate::worktree::GitRunner for ScriptedGitRunner {
    async fn run(
        &self,
        args: &[&str],
        _cwd: &std::path::Path,
    ) -> Result<crate::worktree::GitOutput> {
        let next = self.responses.lock().expect("runner lock").pop_front();
        match next {
            Some(Ok(stdout)) => Ok(crate::worktree::GitOutput::new(stdout)),
            Some(Err(message)) => Err(Error::tool(message)),
            None => Err(Error::tool(format!(
                "scripted git runner exhausted while running `git {}`",
                args.join(" ")
            ))),
        }
    }
}

#[tokio::test]
async fn worktree_tool_uses_the_scripted_manager() {
    let dir = tempfile::tempdir().expect("temp workspace");
    let repo = dir.path().to_path_buf();
    let runner = Arc::new(ScriptedGitRunner::new());
    runner.push_ok(&repo.to_string_lossy());
    runner.push_ok(&format!(
        "worktree {}\nHEAD {}\nbranch refs/heads/main\n",
        repo.display(),
        "1".repeat(40)
    ));
    let manager = Arc::new(WorktreeManager::with_runner(&repo, runner));
    let registry = extended_registry(&dir).with_worktree_manager(Arc::clone(&manager));
    assert!(
        registry.worktree_manager().is_some(),
        "the manager is not stored"
    );

    let tools = registry.build();
    let out = invoke(&tools, "worktree", json!({"action": "list"})).await;

    assert_eq!(out["success"], true, "list failed: {out}");
    assert_eq!(out["data"]["action"], json!("list"));
    let worktrees = out["data"]["worktrees"].as_array().expect("worktree array");
    assert_eq!(worktrees.len(), 1);
    assert_eq!(worktrees[0]["branch"], json!("main"));
}

#[tokio::test]
async fn worktree_tool_defaults_to_the_workspace_root_and_reports_git_failures() {
    let dir = tempfile::tempdir().expect("temp workspace");
    let registry = extended_registry(&dir);
    assert!(registry.worktree_root().is_none(), "no root override");

    let tools = registry.build();
    let out = invoke(&tools, "worktree", json!({"action": "list"})).await;

    assert_eq!(out["success"], false, "unexpected success: {out}");
    let error = out["error"].as_str().expect("error string");
    assert!(
        error.contains(NOT_A_REPOSITORY_MARKER) || error.contains(NOT_INSTALLED_MARKER),
        "the manager's typed git failure must surface: {error}"
    );
}

#[tokio::test]
async fn worktree_tool_forwards_the_permission_engine() {
    let dir = tempfile::tempdir().expect("temp workspace");
    let section = PermissionsSection {
        package_builtin_rules: false,
        tools: BTreeMap::from([("write_file".to_owned(), PermissionLevel::Deny)]),
        ..PermissionsSection::default()
    };
    let engine = Arc::new(PermissionEngine::new(PermissionEngineConfig::new(section)));
    let runner = Arc::new(ScriptedGitRunner::new());
    let manager = Arc::new(WorktreeManager::with_runner(dir.path(), runner));
    let registry = extended_registry(&dir)
        .with_permissions(engine)
        .with_worktree_manager(manager);

    let tools = registry.build();
    let out = invoke(
        &tools,
        "worktree",
        json!({"action": "create", "name": "feature"}),
    )
    .await;

    assert_eq!(out["success"], false, "unexpected success: {out}");
    let error = out["error"].as_str().expect("error string");
    assert!(
        error.to_ascii_lowercase().contains("denied"),
        "the deny decision must reject the worktree: {error}"
    );
}

#[test]
fn worktree_root_override_is_reported() {
    let dir = tempfile::tempdir().expect("temp workspace");
    let root = dir.path().join("repo");
    std::fs::create_dir_all(&root).expect("repo dir");
    let registry = extended_registry(&dir).with_worktree_root(root.clone());
    assert_eq!(registry.worktree_root(), Some(root.as_path()));
}
