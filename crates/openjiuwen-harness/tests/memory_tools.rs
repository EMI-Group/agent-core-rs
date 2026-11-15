//! Integration suite of the eight memory tools over the file-backed workspace backend.
//!
//! Every test drives the *mounted tools* (never the backend directly) through
//! [`Tool::invoke`](openjiuwen_foundation::tool::Tool::invoke) on a `tempfile` workspace, so the
//! suite proves the model-facing contract end to end: the Python path mapping, the payload
//! shapes, the edit failures, the keyword-search result object and the typed traversal failures.
//! The last test builds a real `DeepAgent` through [`DeepAgentBuilder`] and asserts the backend is
//! mounted by default.
//!
//! Nothing here touches the network, reads the wall clock or depends on directory ordering.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use openjiuwen_agent::{Agent, AgentCard};
use openjiuwen_foundation::tool::{Tool, ToolContext};
use openjiuwen_harness::agent::{DeepAgent, DeepAgentBuilder};
use openjiuwen_harness::schema::DeepAgentConfig;
use openjiuwen_harness::tools::ToolRegistry;
use openjiuwen_harness::workspace::Workspace;
use openjiuwen_test_support::{MockCompletion, ScriptedModel};
use serde_json::{json, Value};
use tempfile::TempDir;

/// The traversal rejection message of `validate_memory_path`.
const TRAVERSAL_MESSAGE: &str = "Invalid path: directory traversal not allowed";

/// A throwaway workspace root with the memory tree directories created.
///
/// `Workspace::for_language` only carries the schema; the backend creates write targets on
/// demand, but a read of a seeded file needs the directory to exist first.
fn temp_workspace() -> (TempDir, Workspace) {
    let temp = TempDir::new().expect("temp workspace directory");
    for directory in ["memory", "memory/daily_memory", "coding_memory"] {
        std::fs::create_dir_all(temp.path().join(directory)).expect("memory directory");
    }
    let workspace = Workspace::for_language(temp.path(), "en");
    (temp, workspace)
}

/// The default tool set with the workspace memory backend mounted, in English.
fn memory_tools(workspace: &Workspace) -> Vec<Arc<dyn Tool>> {
    ToolRegistry::new(workspace.clone())
        .with_workspace_memory()
        .with_language("en")
        .build()
}

/// Invokes the mounted tool called `name` with `args`.
async fn invoke(tools: &[Arc<dyn Tool>], name: &str, args: Value) -> Value {
    let tool = tools
        .iter()
        .find(|tool| tool.card().name == name)
        .unwrap_or_else(|| panic!("{name} must be mounted"));
    tool.invoke(args, &ToolContext::new())
        .await
        .expect("a memory tool never fails the invocation")
}

/// Asserts a payload is a success and returns its `data` object.
fn data(output: &Value) -> &Value {
    assert_eq!(output["success"], json!(true), "payload: {output}");
    assert_eq!(output["error"], Value::Null, "payload: {output}");
    &output["data"]
}

/// Asserts a payload is a failure with `error` and returns the message.
fn failure(output: &Value) -> String {
    assert_eq!(output["success"], json!(false), "payload: {output}");
    output["error"]
        .as_str()
        .unwrap_or_else(|| panic!("failure carries a message: {output}"))
        .to_owned()
}

#[tokio::test]
async fn write_and_read_round_trip_with_line_windows() {
    let (_temp, workspace) = temp_workspace();
    let tools = memory_tools(&workspace);

    let written = invoke(
        &tools,
        "write_memory",
        json!({"path": "notes.md", "content": "alpha\nbeta\ngamma"}),
    )
    .await;
    let written = data(&written);
    assert_eq!(written["path"], json!("memory/notes.md"));
    assert_eq!(written["appended"], json!(false));
    assert_eq!(written["fileExisted"], json!(false));
    assert!(
        written["fullPath"]
            .as_str()
            .is_some_and(|path| path.ends_with("memory/notes.md")),
        "the absolute path is reported: {written}"
    );

    let read = invoke(&tools, "read_memory", json!({"path": "notes.md"})).await;
    let read = data(&read);
    assert_eq!(read["path"], json!("memory/notes.md"));
    assert_eq!(read["content"], json!("alpha\nbeta\ngamma"));
    assert_eq!(read["totalLines"], json!(3));
    assert_eq!(read["start_line"], json!(1));
    assert_eq!(read["end_line"], json!(3));
    assert_eq!(read["truncated"], json!(false));

    let window = invoke(
        &tools,
        "memory_get",
        json!({"path": "notes.md", "from_line": 2, "lines": 1}),
    )
    .await;
    let window = data(&window);
    assert_eq!(window["text"], json!("beta"));
    assert_eq!(window["totalLines"], json!(3));
    assert_eq!(window["fromLine"], json!(2));
    assert_eq!(window["toLine"], json!(2));
    assert_eq!(window["disabled"], json!(false));

    let appended = invoke(
        &tools,
        "write_memory",
        json!({"path": "notes.md", "content": "delta", "append": true}),
    )
    .await;
    let appended = data(&appended);
    assert_eq!(appended["appended"], json!(true));
    assert_eq!(appended["fileExisted"], json!(true));
    assert_eq!(
        std::fs::read_to_string(workspace.root_path.join("memory/notes.md")).expect("file"),
        "alpha\nbeta\ngamma\ndelta",
        "append inserts the missing newline"
    );
}

#[tokio::test]
async fn read_memory_reports_a_missing_file() {
    let (_temp, workspace) = temp_workspace();
    let tools = memory_tools(&workspace);

    let output = invoke(&tools, "read_memory", json!({"path": "absent.md"})).await;

    let message = failure(&output);
    assert!(
        message.contains("memory file not found: memory/absent.md"),
        "message: {message}"
    );
}

#[tokio::test]
async fn edit_memory_replaces_one_occurrence_and_rejects_ambiguity() {
    let (_temp, workspace) = temp_workspace();
    let tools = memory_tools(&workspace);

    invoke(
        &tools,
        "write_memory",
        json!({"path": "notes.md", "content": "one\ntwo\nthree"}),
    )
    .await;

    let edited = invoke(
        &tools,
        "edit_memory",
        json!({"path": "notes.md", "old_text": "two", "new_text": "2"}),
    )
    .await;
    let edited = data(&edited);
    assert_eq!(edited["path"], json!("memory/notes.md"));
    assert_eq!(edited["replaced"], json!(1));

    let read = invoke(&tools, "read_memory", json!({"path": "notes.md"})).await;
    assert_eq!(data(&read)["content"], json!("one\n2\nthree"));

    let missing = invoke(
        &tools,
        "edit_memory",
        json!({"path": "notes.md", "old_text": "absent", "new_text": "x"}),
    )
    .await;
    assert_eq!(
        failure(&missing),
        "old_text not found in file. Use read_memory tool to check exact content."
    );

    invoke(
        &tools,
        "write_memory",
        json!({"path": "dupes.md", "content": "same\nsame"}),
    )
    .await;
    let ambiguous = invoke(
        &tools,
        "edit_memory",
        json!({"path": "dupes.md", "old_text": "same", "new_text": "x"}),
    )
    .await;
    assert_eq!(
        failure(&ambiguous),
        "old_text appears 2 times in file. Be more specific."
    );
}

#[tokio::test]
async fn the_python_path_mapping_covers_the_index_daily_files_and_user_profile() {
    let (_temp, workspace) = temp_workspace();
    let tools = memory_tools(&workspace);

    let index = invoke(
        &tools,
        "write_memory",
        json!({"path": "MEMORY.md", "content": "# Memory\n\nArgoCD deploys the service."}),
    )
    .await;
    assert_eq!(data(&index)["path"], json!("memory/MEMORY.md"));

    let daily = invoke(
        &tools,
        "write_memory",
        json!({"path": "2026-11-15.md", "content": "- shipped the release"}),
    )
    .await;
    assert_eq!(
        data(&daily)["path"],
        json!("memory/daily_memory/2026-11-15.md")
    );

    let user = invoke(
        &tools,
        "write_memory",
        json!({"path": "USER.md", "content": "# User\n\nBill."}),
    )
    .await;
    assert_eq!(data(&user)["path"], json!("USER.md"));

    assert!(workspace.root_path.join("memory/MEMORY.md").is_file());
    assert!(workspace
        .root_path
        .join("memory/daily_memory/2026-11-15.md")
        .is_file());
    assert!(workspace.root_path.join("USER.md").is_file());

    let read = invoke(&tools, "read_memory", json!({"path": "MEMORY.md"})).await;
    assert_eq!(
        data(&read)["content"],
        json!("# Memory\n\nArgoCD deploys the service.")
    );

    let window = invoke(&tools, "memory_get", json!({"path": "MEMORY.md"})).await;
    assert_eq!(data(&window)["totalLines"], json!(3));
}

#[tokio::test]
async fn memory_search_returns_the_python_result_object() {
    let (_temp, workspace) = temp_workspace();
    let tools = memory_tools(&workspace);

    invoke(
        &tools,
        "write_memory",
        json!({"path": "MEMORY.md", "content": "# Memory\n\nThe deployment pipeline uses ArgoCD."}),
    )
    .await;
    invoke(
        &tools,
        "write_memory",
        json!({"path": "2026-11-15.md", "content": "- unrelated groceries"}),
    )
    .await;

    let found = invoke(
        &tools,
        "memory_search",
        json!({"query": "deployment pipeline"}),
    )
    .await;
    let found = data(&found);

    assert_eq!(found["provider"], json!("local"));
    assert_eq!(found["search_mode"], json!("keyword_only"));
    assert_eq!(found["embedding_available"], json!(false));
    assert_eq!(found["embedding_error"], Value::Null);
    assert_eq!(found["model"], Value::Null);
    assert_eq!(found["disabled"], json!(false));

    let results = found["results"].as_array().expect("results array");
    assert_eq!(results.len(), 1, "results: {results:?}");
    let hit = &results[0];
    assert_eq!(hit["path"], json!("memory/MEMORY.md"));
    assert_eq!(hit["citation"], json!("memory/MEMORY.md#L3"));
    assert_eq!(hit["start_line"], json!(3));
    assert_eq!(hit["end_line"], json!(3));
    assert_eq!(hit["score"], json!(1.0));
    assert_eq!(hit["source"], json!("memory"));
    assert!(
        hit["snippet"]
            .as_str()
            .is_some_and(|snippet| snippet.contains("ArgoCD")),
        "hit: {hit}"
    );

    let empty = invoke(&tools, "memory_search", json!({"query": "kubernetes"})).await;
    assert_eq!(
        data(&empty)["results"].as_array().map(Vec::len),
        Some(0),
        "an unmatched query yields no hits"
    );
}

#[tokio::test]
async fn coding_memory_requires_valid_front_matter_and_edits_the_body() {
    let (_temp, workspace) = temp_workspace();
    let tools = memory_tools(&workspace);

    let content =
        "---\nname: build\ndescription: how to build the crate\ntype: project\n---\n\ncargo build";
    let written = invoke(
        &tools,
        "coding_memory_write",
        json!({"path": "build.md", "content": content}),
    )
    .await;
    let written = data(&written);
    assert_eq!(written["path"], json!("coding_memory/build.md"));

    let stored = std::fs::read_to_string(workspace.root_path.join("coding_memory/build.md"))
        .expect("coding memory file");
    assert!(stored
        .starts_with("---\nname: build\ndescription: how to build the crate\ntype: project\n"));
    assert!(stored.contains("created_at: "), "stored: {stored}");
    assert!(stored.contains("updated_at: "), "stored: {stored}");
    assert!(stored.ends_with("---\n\ncargo build"), "stored: {stored}");

    let read = invoke(&tools, "coding_memory_read", json!({"path": "build.md"})).await;
    let read = data(&read);
    assert_eq!(read["path"], json!("coding_memory/build.md"));
    assert!(
        read["content"]
            .as_str()
            .is_some_and(|content| content.ends_with("cargo build")),
        "content: {}",
        read["content"]
    );

    let edited = invoke(
        &tools,
        "coding_memory_edit",
        json!({"path": "build.md", "old_text": "cargo build", "new_text": "cargo build --release"}),
    )
    .await;
    let edited = data(&edited);
    assert_eq!(edited["replaced"], json!(1));
    let stored = std::fs::read_to_string(workspace.root_path.join("coding_memory/build.md"))
        .expect("coding memory file");
    assert!(
        stored.ends_with("cargo build --release"),
        "stored: {stored}"
    );

    let missing = invoke(
        &tools,
        "coding_memory_edit",
        json!({"path": "build.md", "old_text": "cargo test", "new_text": "x"}),
    )
    .await;
    assert_eq!(failure(&missing), "old_text not found in file");
}

#[tokio::test]
async fn coding_memory_write_reports_the_python_validation_messages() {
    let (_temp, workspace) = temp_workspace();
    let tools = memory_tools(&workspace);

    let bare = invoke(
        &tools,
        "coding_memory_write",
        json!({"path": "notes.md", "content": "no front matter"}),
    )
    .await;
    assert_eq!(
        failure(&bare),
        "must contain frontmatter(name/description/type)"
    );

    let unknown_type = invoke(
        &tools,
        "coding_memory_write",
        json!({"path": "notes.md", "content": "---\nname: x\ndescription: y\ntype: other\n---\n\nbody"}),
    )
    .await;
    assert_eq!(
        failure(&unknown_type),
        "type must be one of: ('user', 'feedback', 'project', 'reference')"
    );

    let wrong_extension = invoke(
        &tools,
        "coding_memory_write",
        json!({"path": "notes.txt", "content": "x"}),
    )
    .await;
    assert_eq!(failure(&wrong_extension), "Path must end with .md");
}

#[tokio::test]
async fn traversal_paths_fail_with_a_typed_payload() {
    let (temp, workspace) = temp_workspace();
    let tools = memory_tools(&workspace);

    for (tool, path) in [
        ("write_memory", "../escape.md"),
        ("read_memory", "/etc/passwd"),
        ("memory_get", "../escape.md"),
        ("edit_memory", "../../etc/passwd"),
        ("coding_memory_write", "../escape.md"),
        ("coding_memory_read", "/etc/shadow.md"),
    ] {
        let output = if tool == "write_memory" || tool == "coding_memory_write" {
            invoke(&tools, tool, json!({"path": path, "content": "x"})).await
        } else if tool == "edit_memory" {
            invoke(
                &tools,
                tool,
                json!({"path": path, "old_text": "a", "new_text": "b"}),
            )
            .await
        } else {
            invoke(&tools, tool, json!({"path": path})).await
        };
        assert_eq!(
            failure(&output),
            TRAVERSAL_MESSAGE,
            "{tool} must reject {path}"
        );
    }

    let parent = temp.path().parent().expect("temp parent");
    assert!(
        !parent.join("escape.md").exists(),
        "nothing may be written outside the workspace"
    );
}

#[tokio::test]
async fn a_deep_agent_mounts_the_workspace_memory_backend_by_default() {
    let (temp, workspace) = temp_workspace();
    let model = ScriptedModel::builder()
        .completion(MockCompletion::tool_call(
            "write_memory",
            json!({"path": "MEMORY.md", "content": "# Memory\n\nArgoCD deploys it."}),
        ))
        .completion(MockCompletion::tool_call(
            "read_memory",
            json!({"path": "MEMORY.md"}),
        ))
        .respond_text("memory recorded")
        .build();
    let config = DeepAgentConfig::default()
        .with_language("en".to_string())
        .with_workspace(workspace.clone());
    let agent: Arc<DeepAgent> = DeepAgentBuilder::new(
        AgentCard::new("memory-agent").with_description("Memory round-trip agent"),
    )
    .config(config)
    .model(Arc::new(model))
    .build()
    .expect("agent builds");

    let output = agent
        .invoke(json!({"query": "remember the deployment"}), None)
        .await
        .expect("invocation succeeds");

    assert_eq!(output.output, "memory recorded");
    let stored = std::fs::read_to_string(temp.path().join("memory/MEMORY.md"))
        .expect("the write_memory tool created the file");
    assert_eq!(stored, "# Memory\n\nArgoCD deploys it.");
    let read = output
        .observations
        .iter()
        .find(|observation| observation.name.as_deref() == Some("read_memory"))
        .expect("the read_memory call must be observed");
    assert!(
        read.content.contains("ArgoCD"),
        "the read must return the written text: {}",
        read.content
    );
}
