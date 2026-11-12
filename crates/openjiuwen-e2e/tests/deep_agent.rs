#![allow(clippy::unwrap_used, clippy::expect_used)]
//! `DeepAgent` harness cross-crate scenario: the task loop over a real temp workspace,
//! the same round over a loopback HTTP model, permission denial surfaced to the model,
//! and an `Ask` permission interrupt resumed on the same session.
//!
//! Every scenario builds a real [`DeepAgent`] through the umbrella crate
//! (`openjiuwen::harness`) over a `tempfile::TempDir`-backed [`Workspace`] — never the
//! real home directory — and asserts recorded evidence, not timing:
//!
//! - the scripted model's calls (`ScriptedModel::call_count`/`calls`),
//! - the mock server's recordings (`MockLlmServer::requests`),
//! - the tool observations on `AgentOutput`,
//! - the bytes actually left on disk inside the scenario's own temp directory.
//!
//! Fully offline and deterministic: `ScriptedModel` or the loopback `MockLlmServer`
//! only, no credentials, no sleeps for ordering, and no assertions on timestamps,
//! durations or generated ids (a resumable interrupt id is read back from the output
//! instead of being predicted).

mod support;

use std::sync::Arc;

use openjiuwen::core::foundation::{init_model, Model, Role, Tool};
use openjiuwen::core::single_agent::{Agent, AgentCard, ResultType};
use openjiuwen::harness::agent::{DeepAgent, DeepAgentBuilder};
use openjiuwen::harness::permission::{PermissionLevel, PermissionsSection, DENIED_PREFIX};
use openjiuwen::harness::schema::DeepAgentConfig;
use openjiuwen::harness::tools::{BuiltinTool, ToolRegistry};
use openjiuwen::harness::workspace::Workspace;
use openjiuwen_test_support::{
    agent_session, MockCompletion, MockLlmServer, ScriptedModel, ScriptedResponse,
};
use serde_json::{json, Value};
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// A throwaway workspace root; the caller keeps the handle alive for the test.
fn temp_workspace() -> (TempDir, Workspace) {
    let temp = TempDir::new().expect("temp workspace directory");
    let workspace = Workspace::for_language(temp.path(), "en");
    (temp, workspace)
}

/// The stock configuration: English prompts over `workspace`, default task loop.
fn base_config(workspace: &Workspace) -> DeepAgentConfig {
    DeepAgentConfig::default()
        .with_language("en".to_string())
        .with_workspace(workspace.clone())
}

/// Builds the built-in tools of `workspace` for one agent (one registry per agent).
fn tool_set(
    workspace: &Workspace,
    builtins: impl IntoIterator<Item = BuiltinTool>,
) -> Vec<Arc<dyn Tool>> {
    ToolRegistry::new(workspace.clone())
        .with_tools(builtins)
        .build()
}

/// Builds a `DeepAgent` with an injected model, so no lazy `init_model` reaches the network.
fn agent_with(
    model: Arc<dyn Model>,
    config: DeepAgentConfig,
    tools: Vec<Arc<dyn Tool>>,
) -> Arc<DeepAgent> {
    DeepAgentBuilder::new(AgentCard::new("e2e-deep-agent").with_description("e2e scenario"))
        .config(config)
        .model(model)
        .tools(tools)
        .build()
        .expect("agent builds")
}

/// The text content of every `role` message in a recorded wire request.
fn wire_texts(messages: &[Value], role: &str) -> Vec<String> {
    messages
        .iter()
        .filter(|message| message.get("role").and_then(Value::as_str) == Some(role))
        .filter_map(|message| message.get("content").and_then(Value::as_str))
        .map(str::to_owned)
        .collect()
}

/// The content of the recorded wire message answering `tool_call_id`, if present.
fn wire_tool_result(messages: &[Value], tool_call_id: &str) -> Option<String> {
    messages
        .iter()
        .find(|message| {
            message.get("role").and_then(Value::as_str) == Some("tool")
                && message.get("tool_call_id").and_then(Value::as_str) == Some(tool_call_id)
        })
        .and_then(|message| message.get("content").and_then(Value::as_str))
        .map(str::to_owned)
}

/// A `bash` permission section that only pins `bash` to `level`.
fn bash_permissions(level: PermissionLevel) -> PermissionsSection {
    PermissionsSection {
        package_builtin_rules: false,
        tools: std::collections::BTreeMap::from([("bash".to_owned(), level)]),
        ..PermissionsSection::default()
    }
}

// ---------------------------------------------------------------------------
// 1. the task loop over a real temp workspace
// ---------------------------------------------------------------------------

/// A read → write round through the real workspace tools answers with the third completion.
#[tokio::test]
async fn deep_agent_reads_a_workspace_file_then_writes_a_new_one() {
    let (temp, workspace) = temp_workspace();
    std::fs::write(temp.path().join("a.txt"), "seed text\n").expect("seed a.txt");
    let model = ScriptedModel::builder()
        .completion(MockCompletion::tool_call(
            "read_file",
            json!({"file_path": "a.txt"}),
        ))
        .completion(MockCompletion::tool_call(
            "write_file",
            json!({"file_path": "b.txt", "content": "copied"}),
        ))
        .respond_text("done")
        .build();
    let scripted = model.clone();
    let agent = agent_with(
        Arc::new(model),
        base_config(&workspace),
        tool_set(&workspace, [BuiltinTool::ReadFile, BuiltinTool::WriteFile]),
    );
    let session = agent_session("deep-agent-workspace");

    let output = support::within(
        "workspace tool round",
        agent.invoke(json!({"query": "copy a.txt to b.txt"}), Some(session)),
    )
    .await
    .expect("invocation succeeds");

    assert_eq!(output.result_type, ResultType::Answer);
    assert_eq!(output.output, "done");
    assert_eq!(
        scripted.call_count(),
        3,
        "one call per scripted turn: read, write, answer"
    );

    let written = std::fs::read_to_string(temp.path().join("b.txt")).expect("b.txt exists");
    assert_eq!(
        written, "copied",
        "the write tool must leave its content on disk"
    );

    let read = output
        .observations
        .iter()
        .find(|observation| observation.name.as_deref() == Some("read_file"))
        .expect("the read_file call must produce an observation");
    assert!(
        read.content.contains("seed text"),
        "the read observation must carry the file text: {}",
        read.content
    );
    assert!(
        output
            .observations
            .iter()
            .any(|observation| observation.name.as_deref() == Some("write_file")),
        "the write_file call must produce an observation: {:?}",
        output.observations
    );

    let second = &scripted.calls()[1].messages;
    let tool_result = second
        .iter()
        .find(|message| {
            message.role == Role::Tool && message.tool_call_id.as_deref() == Some("call_read_file")
        })
        .expect("the second model call must carry the read_file result for call_read_file");
    assert!(
        tool_result.text().contains("seed text"),
        "the tool result must reach the model window: {}",
        tool_result.text()
    );
}

// ---------------------------------------------------------------------------
// 2. the same round over a loopback HTTP model
// ---------------------------------------------------------------------------

/// The wire path drives the same read → write round through `init_model`.
#[tokio::test]
async fn deep_agent_runs_a_tool_round_over_http() {
    let (temp, workspace) = temp_workspace();
    std::fs::write(temp.path().join("a.txt"), "wire seed\n").expect("seed a.txt");
    let server = MockLlmServer::with_script([
        ScriptedResponse::completion(MockCompletion::tool_call(
            "read_file",
            json!({"file_path": "a.txt"}),
        )),
        ScriptedResponse::completion(MockCompletion::tool_call(
            "write_file",
            json!({"file_path": "b.txt", "content": "copied"}),
        )),
        ScriptedResponse::text("http done"),
    ])
    .await;
    let model = init_model(server.client_config(), server.request_config("mock-model"))
        .await
        .expect("model initialises against the loopback server");
    // The injected model is called with the inner kernel agent's per-call request, so
    // the model identity has to reach it through the harness configuration as well.
    let config = base_config(&workspace)
        .with_model_name(Some("mock-model".to_string()))
        .with_model_config(Some(server.request_config("mock-model")));
    let agent = agent_with(
        model,
        config,
        tool_set(&workspace, [BuiltinTool::ReadFile, BuiltinTool::WriteFile]),
    );
    let session = agent_session("deep-agent-http");

    let output = support::within(
        "http tool round",
        agent.invoke(json!({"query": "copy a.txt to b.txt"}), Some(session)),
    )
    .await
    .expect("invocation succeeds");

    assert_eq!(output.result_type, ResultType::Answer);
    assert_eq!(output.output, "http done");
    let written = std::fs::read_to_string(temp.path().join("b.txt")).expect("b.txt exists");
    assert_eq!(
        written, "copied",
        "the write tool must run over the wire path too"
    );

    assert_eq!(
        server.request_count(),
        3,
        "one request per scripted turn: read, write, answer"
    );
    let recorded = server.requests();
    assert!(
        recorded
            .iter()
            .all(|request| request.path.ends_with("/chat/completions")),
        "every request must hit the OpenAI-compatible endpoint: {recorded:?}"
    );

    // The third request is the answer turn: its window carries both tool results.
    let answer_window = recorded[2].messages();
    let read_result = wire_tool_result(&answer_window, "call_read_file")
        .expect("the answer turn must see the read_file result");
    assert!(
        read_result.contains("wire seed"),
        "the read result must reach the model window: {read_result}"
    );
    let write_result = wire_tool_result(&answer_window, "call_write_file")
        .expect("the answer turn must see the write_file result");
    assert!(
        write_result.contains("b.txt"),
        "the write result must reach the model window: {write_result}"
    );

    // The second request is the write turn: only the read result is available yet.
    let write_window = recorded[1].messages();
    assert!(
        wire_tool_result(&write_window, "call_read_file").is_some(),
        "the write turn must see the read_file result: {write_window:?}"
    );
    assert!(
        wire_tool_result(&write_window, "call_write_file").is_none(),
        "the write turn cannot see its own result: {write_window:?}"
    );
    assert!(
        wire_texts(&recorded[0].messages(), "tool").is_empty(),
        "the first turn has no tool result yet: {:?}",
        recorded[0].messages()
    );
}

// ---------------------------------------------------------------------------
// 3. permission denial is reported to the model
// ---------------------------------------------------------------------------

/// A `Deny` rule skips the real `bash` tool and shows the denial to the model.
#[tokio::test]
async fn permission_denial_is_reported_to_the_model() {
    let (_temp, workspace) = temp_workspace();
    let model = ScriptedModel::builder()
        .completion(MockCompletion::tool_call(
            "bash",
            json!({"command": "echo hi"}),
        ))
        .respond_text("denied and done")
        .build();
    let scripted = model.clone();
    let config =
        base_config(&workspace).with_permissions(Some(bash_permissions(PermissionLevel::Deny)));
    let agent = agent_with(
        Arc::new(model),
        config,
        tool_set(&workspace, [BuiltinTool::Bash]),
    );
    let session = agent_session("deep-agent-deny");

    let output = support::within(
        "denied bash round",
        agent.invoke(json!({"query": "run bash"}), Some(session)),
    )
    .await
    .expect("a denial is reported to the model, not raised as an error");

    assert_eq!(output.result_type, ResultType::Answer);
    assert_eq!(output.output, "denied and done");
    assert_eq!(scripted.call_count(), 2);

    let denied = output
        .observations
        .iter()
        .find(|observation| observation.name.as_deref() == Some("bash"))
        .expect("the denied tool call must produce an observation");
    assert!(
        denied.content.starts_with(DENIED_PREFIX),
        "the observation must carry the denial surface: {}",
        denied.content
    );

    let second = &scripted.calls()[1].messages;
    assert!(
        second
            .iter()
            .any(|message| message.role == Role::Tool && message.text().contains(DENIED_PREFIX)),
        "the denial must reach the model window: {second:?}"
    );

    // A denied call never starts a process, so no execution report may appear anywhere.
    assert!(
        !denied.content.contains("Exit Code:"),
        "the denied bash call must not have executed: {}",
        denied.content
    );
    assert!(
        !second
            .iter()
            .any(|message| message.text().contains("Exit Code:")),
        "no model-window message may carry an execution report: {second:?}"
    );
    assert!(
        !output.output.contains("Exit Code:"),
        "the answer must not carry an execution report: {}",
        output.output
    );
    assert!(
        !output
            .observations
            .iter()
            .any(|observation| observation.content.contains("Exit Code:")),
        "no observation may carry an execution report: {:?}",
        output.observations
    );
}

// ---------------------------------------------------------------------------
// 4. an `Ask` permission interrupt resumes on the same session
// ---------------------------------------------------------------------------

/// An `Ask` rule pauses the run; approving on the same session executes the tool.
#[tokio::test]
async fn ask_permission_interrupts_and_resumes_on_the_same_session() {
    let (_temp, workspace) = temp_workspace();
    let model = ScriptedModel::builder()
        .completion(MockCompletion::tool_call(
            "bash",
            json!({"command": "echo hi"}),
        ))
        .respond_text("approved and done")
        .build();
    let scripted = model.clone();
    let config =
        base_config(&workspace).with_permissions(Some(bash_permissions(PermissionLevel::Ask)));
    let agent = agent_with(
        Arc::new(model),
        config,
        tool_set(&workspace, [BuiltinTool::Bash]),
    );
    let session = agent_session("deep-agent-ask");

    let paused = support::within(
        "ask permission pause",
        agent.invoke(json!({"query": "run bash"}), Some(session.clone())),
    )
    .await
    .expect("the paused invocation succeeds");
    assert_eq!(paused.result_type, ResultType::Interrupt);
    assert!(
        !paused.interrupt_ids.is_empty(),
        "an `Ask` rule must raise at least one interrupt: {paused:?}"
    );
    let interrupt_id = paused.interrupt_ids[0].clone();

    let resumed = support::within(
        "approved resume",
        agent.invoke(
            json!({
                "query": {
                    "interrupt_answers": {interrupt_id.clone(): {"approved": true}},
                    "user_inputs": {},
                    "raw_inputs": null,
                },
            }),
            Some(session),
        ),
    )
    .await
    .expect("the approved resume succeeds");
    assert_eq!(resumed.result_type, ResultType::Answer);
    assert_eq!(resumed.output, "approved and done");
    assert_eq!(
        scripted.call_count(),
        2,
        "one call before the pause and one follow-up after the approved tool ran"
    );

    let last = scripted
        .last_call()
        .expect("a resumed turn calls the model again");
    let tool_result = last
        .messages
        .iter()
        .find(|message| {
            message.role == Role::Tool && message.tool_call_id.as_deref() == Some("call_bash")
        })
        .expect("the follow-up call must carry the bash result for call_bash");
    assert!(
        !tool_result.text().is_empty(),
        "the approved tool must report its output to the model"
    );
}
