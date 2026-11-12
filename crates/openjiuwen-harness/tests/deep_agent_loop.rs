//! End-to-end suite of the `DeepAgent` task loop through the public API.
//!
//! Every test builds a real [`DeepAgent`] with [`openjiuwen_test_support::ScriptedModel`]
//! over a `tempfile::TempDir`-backed [`Workspace`] — never the real home directory —
//! and asserts the cross-module contract the loop must keep:
//!
//! - a tool-free turn returns the scripted answer and reports no termination reason;
//! - a tool round really invokes a mounted [`FunctionTool`] and feeds its result back
//!   to the model under the same tool-call id;
//! - a permission `deny` short-circuits the `bash` call, surfaces
//!   [`DENIED_PREFIX`] and never executes the command;
//! - exhausting `max_iterations` reports `termination_reason = "max_iterations"` with
//!   `truncated_by_max_iterations = true`;
//! - a queued follow-up is drained into the next task-loop round and the persisted
//!   state has no pending follow-ups left;
//! - `abort` is idempotent while idle and after a run, and a later invocation works.
//!
//! Nothing here touches the network, sleeps, or depends on wall-clock ordering.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use openjiuwen_agent::{Agent, AgentCard, ResultType};
use openjiuwen_foundation::tool::{FunctionTool, Tool};
use openjiuwen_foundation::Role;
use openjiuwen_harness::agent::{DeepAgent, DeepAgentBuilder};
use openjiuwen_harness::permission::{PermissionLevel, PermissionsSection, DENIED_PREFIX};
use openjiuwen_harness::schema::{DeepAgentConfig, InteractionPhase};
use openjiuwen_harness::tools::{BuiltinTool, ToolRegistry};
use openjiuwen_harness::workspace::Workspace;
use openjiuwen_session::{AgentSession, Session};
use openjiuwen_test_support::{MockCompletion, ScriptedModel, ScriptedTurn};
use serde_json::{json, Value};
use tempfile::TempDir;

/// A throwaway workspace root; the caller keeps the handle alive for the test.
fn temp_workspace() -> (TempDir, Workspace) {
    let temp = TempDir::new().expect("temp workspace directory");
    let workspace = Workspace::for_language(temp.path(), "en");
    (temp, workspace)
}

/// The stock configuration: English prompts over `workspace`, task loop off.
fn base_config(workspace: &Workspace) -> DeepAgentConfig {
    DeepAgentConfig::default()
        .with_language("en".to_string())
        .with_workspace(workspace.clone())
}

/// Builds a `DeepAgent` with the base config and no extra tools.
fn scripted_agent(model: ScriptedModel, workspace: &Workspace) -> Arc<DeepAgent> {
    DeepAgentBuilder::new(AgentCard::new("loop-agent").with_description("Loop test agent"))
        .config(base_config(workspace))
        .model(Arc::new(model))
        .build()
        .expect("agent builds")
}

/// An echo `FunctionTool` plus the log of every invocation it accepted.
fn recording_echo() -> (Arc<dyn Tool>, Arc<Mutex<Vec<Value>>>) {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let recorder = seen.clone();
    let tool = FunctionTool::builder()
        .name("echo")
        .description("Echo the input text back to the model")
        .input_schema(json!({
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }))
        .handler(move |inputs: Value, _ctx| {
            let recorder = recorder.clone();
            async move {
                recorder.lock().expect("tool log").push(inputs.clone());
                // `ToolOutput`-shaped so the model-facing text carries the input.
                Ok(json!({"success": true, "data": {"content": inputs}}))
            }
        })
        .build()
        .expect("the echo tool builds");
    (tool.boxed(), seen)
}

/// A counting echo `FunctionTool` for loop-exhaustion tests.
fn counting_echo() -> (Arc<dyn Tool>, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = calls.clone();
    let tool = FunctionTool::builder()
        .name("echo")
        .description("Echo the input text back to the model")
        .input_schema(json!({
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }))
        .handler(move |inputs: Value, _ctx| {
            let counter = counter.clone();
            async move {
                counter.fetch_add(1, Ordering::SeqCst);
                Ok(json!({"success": true, "echo": inputs}))
            }
        })
        .build()
        .expect("the echo tool builds");
    (tool.boxed(), calls)
}

/// One turn with no tool call returns the scripted text and reports no termination.
#[tokio::test]
async fn no_tool_answer_returns_the_scripted_text() {
    let (_temp, workspace) = temp_workspace();
    let model = ScriptedModel::text("hello from the script");
    let scripted = model.clone();
    let agent = scripted_agent(model, &workspace);

    let session: Arc<dyn Session> = Arc::new(AgentSession::new("plain-session"));
    let output = agent
        .invoke(json!({"query": "hi"}), Some(session))
        .await
        .expect("invocation succeeds");

    assert_eq!(output.result_type, ResultType::Answer);
    assert_eq!(output.output, "hello from the script");
    assert_eq!(output.termination_reason, None);
    assert!(!output.truncated_by_max_iterations);
    assert_eq!(scripted.call_count(), 1);
    assert_eq!(scripted.calls()[0].last_user_text().as_deref(), Some("hi"));
}

/// A tool round runs the mounted function tool and answers with the second completion.
#[tokio::test]
async fn tool_round_executes_the_function_tool_and_returns_the_second_completion() {
    let (_temp, workspace) = temp_workspace();
    let (echo, seen) = recording_echo();
    let model = ScriptedModel::builder()
        .completion(MockCompletion::tool_call("echo", json!({"text": "ping"})))
        .respond_text("tool round done")
        .build();
    let scripted = model.clone();

    let agent = DeepAgentBuilder::new(AgentCard::new("tool-agent").with_description("test"))
        .config(base_config(&workspace))
        .model(Arc::new(model))
        .tool(echo)
        .build()
        .expect("agent builds");

    let session: Arc<dyn Session> = Arc::new(AgentSession::new("tool-session"));
    let output = agent
        .invoke(json!({"query": "echo it"}), Some(session))
        .await
        .expect("invocation succeeds");

    assert_eq!(output.result_type, ResultType::Answer);
    assert_eq!(output.output, "tool round done");
    assert_eq!(scripted.call_count(), 2);

    let invoked = seen.lock().expect("tool log").clone();
    assert_eq!(
        invoked,
        vec![json!({"text": "ping"})],
        "the tool must run once with the model arguments"
    );

    let calls = scripted.calls();
    assert_eq!(calls[0].last_user_text().as_deref(), Some("echo it"));
    let second = &calls[1].messages;
    assert!(
        second.iter().any(|message| message.role == Role::Assistant
            && message.tool_calls.iter().any(|call| call.name == "echo")),
        "the second model call must carry the assistant tool call: {second:?}"
    );
    let tool_result = second
        .iter()
        .find(|message| {
            message.role == Role::Tool && message.tool_call_id.as_deref() == Some("call_echo")
        })
        .expect("the second model call must carry the echo result for call_echo");
    assert!(
        tool_result.text().contains("ping"),
        "the tool result must echo the arguments: {}",
        tool_result.text()
    );
    assert!(output
        .observations
        .iter()
        .any(|observation| observation.name.as_deref() == Some("echo")));
}

/// A `deny` rule stops the `bash` call before it runs and shows the denial surface.
#[tokio::test]
async fn permission_denied_bash_call_is_skipped_and_reported() {
    let (_temp, workspace) = temp_workspace();
    // The real built-in `bash` tool; a denial must skip it before any process starts.
    let bash_tools = ToolRegistry::new(workspace.clone())
        .with_tools([BuiltinTool::Bash])
        .build();
    let section = PermissionsSection {
        package_builtin_rules: false,
        tools: std::collections::BTreeMap::from([("bash".to_string(), PermissionLevel::Deny)]),
        ..PermissionsSection::default()
    };
    let config = base_config(&workspace).with_permissions(Some(section));
    let model = ScriptedModel::builder()
        .completion(MockCompletion::tool_call(
            "bash",
            json!({"command": "echo hi"}),
        ))
        .respond_text("denied and done")
        .build();
    let scripted = model.clone();

    let agent = DeepAgentBuilder::new(AgentCard::new("deny-agent").with_description("test"))
        .config(config)
        .model(Arc::new(model))
        .tools(bash_tools)
        .build()
        .expect("agent builds");

    let session: Arc<dyn Session> = Arc::new(AgentSession::new("deny-session"));
    let output = tokio::time::timeout(
        Duration::from_secs(2),
        agent.invoke(json!({"query": "run bash"}), Some(session)),
    )
    .await
    .expect("the denied invocation must not hang")
    .expect("invocation succeeds");

    assert_eq!(output.output, "denied and done");
    assert_eq!(scripted.call_count(), 2);
    let denied = output
        .observations
        .iter()
        .find(|observation| observation.name.as_deref() == Some("bash"))
        .expect("the denied tool call must produce an observation");
    assert!(
        denied.content.starts_with(DENIED_PREFIX),
        "the observation must carry the denial: {}",
        denied.content
    );
    assert!(
        !denied.content.contains("Exit Code:"),
        "the denied bash call must not have executed: {}",
        denied.content
    );
    let second = &scripted.calls()[1].messages;
    assert!(
        second
            .iter()
            .any(|message| message.role == Role::Tool && message.text().contains(DENIED_PREFIX)),
        "the denial must reach the model window: {second:?}"
    );
}

/// Exhausting the round budget marks the output truncated with the documented reason.
#[tokio::test]
async fn max_iterations_marks_the_output_truncated() {
    let (_temp, workspace) = temp_workspace();
    let (echo, calls) = counting_echo();
    let completion = MockCompletion::tool_call("echo", json!({"text": "again"}));
    let model = ScriptedModel::builder()
        .completion(completion.clone())
        .fallback(ScriptedTurn::Response(completion.to_chat_response("")))
        .build();
    let config = base_config(&workspace).with_max_iterations(Some(2));

    let agent = DeepAgentBuilder::new(AgentCard::new("limit-agent").with_description("test"))
        .config(config)
        .model(Arc::new(model))
        .tool(echo)
        .build()
        .expect("agent builds");

    let session: Arc<dyn Session> = Arc::new(AgentSession::new("limit-session"));
    let output = agent
        .invoke(json!({"query": "loop forever"}), Some(session))
        .await
        .expect("invocation succeeds");

    assert_eq!(output.termination_reason.as_deref(), Some("max_iterations"));
    assert!(output.truncated_by_max_iterations);
    assert_eq!(output.result_type, ResultType::Error);
    assert_eq!(output.iterations_used, Some(2));
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "each round runs the tool once"
    );
}

/// A queued follow-up is consumed by the next task-loop round and drained from state.
#[tokio::test]
async fn follow_up_queue_is_drained_into_the_next_task_loop_round() {
    let (_temp, workspace) = temp_workspace();
    let model = ScriptedModel::new([
        ScriptedTurn::text("first answer"),
        ScriptedTurn::text("second answer"),
    ]);
    let scripted = model.clone();
    let config = base_config(&workspace).with_enable_task_loop(true);
    let agent = DeepAgentBuilder::new(AgentCard::new("follow-up-agent").with_description("test"))
        .config(config)
        .model(Arc::new(model))
        .build()
        .expect("agent builds");

    agent.follow_up("keep going", None);

    let session: Arc<dyn Session> = Arc::new(AgentSession::new("follow-up-session"));
    let output = agent
        .invoke(json!({"query": "start"}), Some(session))
        .await
        .expect("task-loop invocation succeeds");

    assert_eq!(output.result_type, ResultType::Answer);
    assert_eq!(output.output, "second answer");
    assert_eq!(scripted.call_count(), 2);

    let calls = scripted.calls();
    assert!(
        calls[0]
            .messages
            .iter()
            .any(|message| message.role == Role::User && message.text() == "start"),
        "the initial query must reach the first round"
    );
    assert_eq!(
        calls[1].last_user_text().as_deref(),
        Some("keep going"),
        "the follow-up must be the next round's query"
    );

    let state = agent.load_state().await.expect("state loads");
    assert!(
        state.pending_follow_ups.is_empty(),
        "the queued follow-up must be drained: {:?}",
        state.pending_follow_ups
    );
    assert_eq!(state.iteration, 2);
}

/// `abort` is safe while idle and after a run; later invocations still work.
#[tokio::test]
async fn abort_is_idempotent_and_a_later_invocation_still_runs() {
    let (_temp, workspace) = temp_workspace();
    let model = ScriptedModel::new([
        ScriptedTurn::text("after first abort"),
        ScriptedTurn::text("after second abort"),
    ]);
    let agent = scripted_agent(model, &workspace);

    // Twice with no run in flight: synchronous, idempotent, no panic.
    agent.abort();
    agent.abort();
    assert_eq!(agent.phase(), InteractionPhase::Terminated);

    let first_session: Arc<dyn Session> = Arc::new(AgentSession::new("abort-session-1"));
    let first = agent
        .invoke(json!({"query": "first"}), Some(first_session))
        .await
        .expect("a fresh invocation succeeds after an idle abort");
    assert_eq!(first.output, "after first abort");
    assert_ne!(agent.phase(), InteractionPhase::Terminated);

    // Twice after a completed invocation: still safe, and the next run works.
    agent.abort();
    agent.abort();
    assert_eq!(agent.phase(), InteractionPhase::Terminated);

    let second_session: Arc<dyn Session> = Arc::new(AgentSession::new("abort-session-2"));
    let second = agent
        .invoke(json!({"query": "second"}), Some(second_session))
        .await
        .expect("a later invocation succeeds after an abort");
    assert_eq!(second.output, "after second abort");
    assert_eq!(second.result_type, ResultType::Answer);
    assert_ne!(agent.phase(), InteractionPhase::Terminated);
}
