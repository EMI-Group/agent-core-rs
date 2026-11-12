//! Cross-module parity tests for the tool registry, the tool cards and the rendered descriptions.
//!
//! Per-tool behaviour is covered beside each module; these tests wire the whole registry
//! together: the model-facing name order, the card shapes, the parallel-safety flags, and
//! end-to-end calls through one `build()` (so the filesystem tools share their read state).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use openjiuwen_foundation::tool::{Tool, ToolContext};
use serde_json::{json, Value};
use tempfile::TempDir;

use super::{descriptions, tool_names, BuiltinTool, ToolRegistry};
use crate::workspace::Workspace;

/// The twenty Python tool names, in registry order.
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

/// The `required` list of each Python card; the empty slices have no `required` key at all.
const EXPECTED_REQUIRED: [(&str, &[&str]); 20] = [
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
];

/// Builds a registry over `dir` with English card text.
fn registry(dir: &TempDir) -> ToolRegistry {
    ToolRegistry::new(Workspace::new(dir.path())).with_language("en")
}

/// Builds the full tool set of a registry over `dir`.
fn built(dir: &TempDir) -> Vec<Arc<dyn Tool>> {
    registry(dir).build()
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
    assert_eq!(BuiltinTool::ALL.len(), EXPECTED_NAMES.len());

    let names: Vec<&str> = BuiltinTool::ALL.iter().map(|tool| tool.name()).collect();
    assert_eq!(names, EXPECTED_NAMES.to_vec());
}

#[test]
fn from_name_round_trips_every_name_and_rejects_unknown_tools() {
    for name in EXPECTED_NAMES {
        let tool = BuiltinTool::from_name(name).unwrap_or_else(|| panic!("unknown name {name}"));
        assert_eq!(tool.name(), name, "round-trip mismatch for {name}");
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
    assert_eq!(cards.len(), BuiltinTool::ALL.len());
    let names: Vec<&str> = cards.iter().map(|card| card.name.as_str()).collect();
    assert_eq!(names, tool_names().to_vec());

    for tool in BuiltinTool::ALL {
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

    for tool in BuiltinTool::ALL {
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
fn only_the_write_tools_are_not_parallel_safe() {
    for tool in BuiltinTool::ALL {
        let expected = !matches!(tool, BuiltinTool::WriteFile | BuiltinTool::EditFile);
        assert_eq!(tool.parallel_safe(), expected, "for {}", tool.name());
    }

    let dir = tempfile::tempdir().expect("temp workspace");
    let tools = built(&dir);
    for tool in BuiltinTool::ALL {
        let expected = !matches!(tool, BuiltinTool::WriteFile | BuiltinTool::EditFile);
        let built_tool = tool_by_name(&tools, tool.name());
        assert_eq!(
            built_tool.is_parallel_safe(),
            expected,
            "{} built parallel safety",
            tool.name()
        );
        assert_eq!(
            built_tool.card().parallel_safe,
            expected,
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
