#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Integration tests for the interrupt/resume contract of [`ReActAgent`].
//!
//! The suite pins the human-in-the-loop surface end to end over an in-process
//! `AgentSession` and a `ScriptedModel`: a confirm interrupt pauses before the
//! tool runs, an approving resume executes it, a rejecting resume reports the
//! feedback without executing it, auto-confirm skips the pause, an ask-user
//! answer becomes the tool result, and two pending calls of one model turn pause
//! together and resume one at a time.
//!
//! Everything is offline and deterministic: tools are in-process closures that
//! record their inputs, sessions are `AgentSession`s, and assertions read the
//! recorded model calls, the persisted session state, and the tool log.
use futures::StreamExt;
use openjiuwen_agent::output::ANSWER;
use openjiuwen_agent::{
    Agent, AgentCard, AskUserRail, ConfirmInterruptRail, OutputFrame, ReActAgent, ReActAgentConfig,
    ResultType, AUTO_CONFIRM_STATE_KEY, DEFAULT_REJECT_FEEDBACK, INTERRUPTION_STATE_KEY,
};
use openjiuwen_foundation::{BaseMessage, FunctionTool, Model, Role, Tool, ToolContext};
use openjiuwen_session::{AgentSession, Session, StreamEvent};
use openjiuwen_test_support::{MockCompletion, MockToolCall, ScriptedModel};
use parking_lot::Mutex;
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Duration;
/// Session id every test uses; also the `conversation_id` a resume carries.
const SESSION_ID: &str = "s1";
/// Owner string `ConfirmInterruptRail` records on its requests.
const CONFIRM_OWNER: &str = "openjiuwen.agent.rails.interrupt.ConfirmInterruptRail";
/// Hands a clone of `model` to the agent while the test keeps the original for
/// inspecting the recorded calls.
fn shared(model: &ScriptedModel) -> Arc<dyn Model> {
    Arc::new(model.clone())
}
/// Builds a `ReActAgent` over `model` with the default configuration.
fn agent(model: &ScriptedModel) -> ReActAgent {
    let agent = ReActAgent::new(AgentCard::new("interrupt-test"));
    agent.configure(ReActAgentConfig::new());
    agent.set_model(shared(model));
    agent
}
/// Returns a fresh log a [`recording_tool`] writes into.
fn log() -> Arc<Mutex<Vec<Value>>> {
    Arc::new(Mutex::new(Vec::new()))
}
/// Builds a tool named `name` that records its parsed inputs into `log` and
/// answers with the fixed text `"tool-output"`.
fn recording_tool(name: &str, log: &Arc<Mutex<Vec<Value>>>) -> Arc<dyn Tool> {
    let log = Arc::clone(log);
    let tool = FunctionTool::builder()
        .name(name)
        .description("recording tool")
        .input_schema(json!({"type": "object"}))
        .handler(move |inputs: Value, _ctx: ToolContext| {
            // `handler` requires an `Fn`, so the log is cloned per call.
            let log = Arc::clone(&log);
            async move {
                log.lock().push(inputs);
                Ok(Value::String("tool-output".to_owned()))
            }
        })
        .build()
        .expect("the recording tool fixture builds");
    Arc::new(tool)
}
/// Builds an in-process session with the shared id.
fn new_session() -> Arc<dyn Session> {
    Arc::new(AgentSession::new(SESSION_ID))
}
/// The text of every message in `messages` that has `role`.
fn texts_with_role(messages: &[BaseMessage], role: &Role) -> Vec<String> {
    messages
        .iter()
        .filter(|message| &message.role == role)
        .map(|message| message.text().into_owned())
        .collect()
}
/// The serialized `InterruptRequest` carried by an `__interaction__` frame.
fn request_of(frame: &OutputFrame) -> &Value {
    &frame.payload["value"]
}
/// The `tool_call_id` of every interaction frame, in frame order.
fn frame_call_ids(frames: &[OutputFrame]) -> Vec<String> {
    frames
        .iter()
        .map(|frame| {
            request_of(frame)["tool_call_id"]
                .as_str()
                .expect("every frame carries a tool call id")
                .to_owned()
        })
        .collect()
}
/// A resume payload answering the tool calls in `answers`.
fn resume(answers: &Value) -> Value {
    json!({
        "query": {
            "user_inputs": answers,
            "raw_inputs": null,
            "interrupt_answers": {},
        },
        "conversation_id": SESSION_ID,
    })
}

/// A confirm rail pauses the turn before the approved tool ever runs.
#[tokio::test]
async fn confirm_interrupt_pauses_before_the_tool_runs() {
    let model = ScriptedModel::builder()
        .completion(MockCompletion::tool_call("write", json!({"path": "x"})))
        .respond_text("done")
        .build();
    let agent = agent(&model);
    let log = log();
    agent.add_tool(recording_tool("write", &log));
    agent.register_rail(Arc::new(ConfirmInterruptRail::for_tools(["write"])));
    let session = new_session();

    let paused = agent
        .invoke(json!({"query": "do it"}), Some(Arc::clone(&session)))
        .await
        .unwrap();

    assert_eq!(paused.result_type, ResultType::Interrupt);
    assert_eq!(paused.interrupt_ids.len(), 1);
    assert_eq!(paused.state.len(), 1);
    assert_eq!(paused.state[0].kind, "__interaction__");
    assert_eq!(
        paused.state[0].payload["id"],
        json!(paused.interrupt_ids[0])
    );

    let request = request_of(&paused.state[0]);
    let interrupt_id = request["interrupt_id"]
        .as_str()
        .expect("the request carries its id");
    assert!(interrupt_id.starts_with("intr_"), "{interrupt_id}");
    assert_eq!(interrupt_id, paused.interrupt_ids[0]);
    assert_eq!(request["interrupt_kind"], json!("confirm"));
    assert_eq!(request["interrupt_owner"], json!(CONFIRM_OWNER));
    assert_eq!(request["tool_name"], json!("write"));
    assert_eq!(request["tool_call_id"], json!("call_write"));
    assert_eq!(request["tool_args"], json!("{\"path\":\"x\"}"));

    assert!(log.lock().is_empty(), "the paused tool never runs");
    assert_eq!(paused.output, "");
    assert!(paused.data.is_none());
    assert!(
        session
            .get_state(INTERRUPTION_STATE_KEY)
            .await
            .unwrap()
            .is_some(),
        "the pending interruption is persisted on the session"
    );
}

/// Approving a paused call executes it and feeds its output back to the model.
#[tokio::test]
async fn approval_resume_executes_the_tool_and_feeds_the_answer() {
    let model = ScriptedModel::builder()
        .completion(MockCompletion::tool_call("write", json!({"path": "x"})))
        .respond_text("done")
        .build();
    let agent = agent(&model);
    let log = log();
    agent.add_tool(recording_tool("write", &log));
    agent.register_rail(Arc::new(ConfirmInterruptRail::for_tools(["write"])));
    let session = new_session();

    let paused = agent
        .invoke(json!({"query": "do it"}), Some(Arc::clone(&session)))
        .await
        .unwrap();
    assert_eq!(paused.result_type, ResultType::Interrupt);

    let resumed = agent
        .invoke(
            resume(&json!({"call_write": {"approved": true}})),
            Some(Arc::clone(&session)),
        )
        .await
        .unwrap();

    assert_eq!(resumed.result_type, ResultType::Answer);
    assert_eq!(resumed.output, "done");
    let recorded = log.lock().clone();
    assert_eq!(recorded.len(), 1);
    assert_eq!(recorded[0], json!({"path": "x"}));

    let calls = model.calls();
    assert_eq!(calls.len(), 2, "one call before the pause, one after");
    let tool_texts = texts_with_role(&calls[1].messages, &Role::Tool);
    assert!(
        tool_texts.iter().any(|text| text.contains("tool-output")),
        "the tool output reaches the model: {tool_texts:?}"
    );
    assert_eq!(
        session.get_state(INTERRUPTION_STATE_KEY).await.unwrap(),
        None,
        "a resumed turn clears the interruption"
    );
}

/// Rejecting a paused call reports the feedback instead of running the tool.
#[tokio::test]
async fn rejection_resume_reports_the_feedback() {
    let model = ScriptedModel::builder()
        .completion(MockCompletion::tool_call("write", json!({"path": "x"})))
        .respond_text("done")
        .build();
    let agent = agent(&model);
    let log = log();
    agent.add_tool(recording_tool("write", &log));
    agent.register_rail(Arc::new(ConfirmInterruptRail::for_tools(["write"])));
    let session = new_session();

    let paused = agent
        .invoke(json!({"query": "do it"}), Some(Arc::clone(&session)))
        .await
        .unwrap();
    assert_eq!(paused.result_type, ResultType::Interrupt);

    let resumed = agent
        .invoke(
            resume(&json!({"call_write": {"approved": false, "feedback": "not now"}})),
            Some(Arc::clone(&session)),
        )
        .await
        .unwrap();

    assert_eq!(resumed.result_type, ResultType::Answer);
    assert_eq!(resumed.output, "done");
    assert!(log.lock().is_empty(), "a rejected tool never runs");
    let calls = model.calls();
    assert_eq!(calls.len(), 2);
    assert_eq!(
        texts_with_role(&calls[1].messages, &Role::Tool),
        vec!["not now".to_owned()],
        "the rejection feedback becomes the tool result"
    );
}

/// Rejecting without a reason uses the crate's default feedback text.
#[tokio::test]
async fn rejection_without_feedback_uses_the_default() {
    let model = ScriptedModel::builder()
        .completion(MockCompletion::tool_call("write", json!({"path": "x"})))
        .respond_text("done")
        .build();
    let agent = agent(&model);
    let log = log();
    agent.add_tool(recording_tool("write", &log));
    agent.register_rail(Arc::new(ConfirmInterruptRail::for_tools(["write"])));
    let session = new_session();

    let paused = agent
        .invoke(json!({"query": "do it"}), Some(Arc::clone(&session)))
        .await
        .unwrap();
    assert_eq!(paused.result_type, ResultType::Interrupt);

    let resumed = agent
        .invoke(
            resume(&json!({"call_write": {"approved": false}})),
            Some(Arc::clone(&session)),
        )
        .await
        .unwrap();

    assert_eq!(resumed.result_type, ResultType::Answer);
    assert_eq!(resumed.output, "done");
    assert!(log.lock().is_empty(), "a rejected tool never runs");
    let calls = model.calls();
    assert_eq!(
        texts_with_role(&calls[1].messages, &Role::Tool),
        vec![DEFAULT_REJECT_FEEDBACK.to_owned()]
    );
}

/// A host auto-confirm decision seeded on the session skips the pause entirely.
#[tokio::test]
async fn auto_confirm_skips_the_interruption() {
    let model = ScriptedModel::builder()
        .completion(MockCompletion::tool_call("write", json!({"path": "x"})))
        .respond_text("done")
        .build();
    let agent = agent(&model);
    let log = log();
    agent.add_tool(recording_tool("write", &log));
    agent.register_rail(Arc::new(ConfirmInterruptRail::for_tools(["write"])));
    let session = new_session();
    session
        .update_state(json!({AUTO_CONFIRM_STATE_KEY: {"write": true}}))
        .await
        .unwrap();

    let output = agent
        .invoke(json!({"query": "do it"}), Some(Arc::clone(&session)))
        .await
        .unwrap();

    assert_eq!(output.result_type, ResultType::Answer);
    assert_eq!(output.output, "done");
    assert_eq!(log.lock().len(), 1, "the auto-confirmed tool runs");
    assert_eq!(
        session.get_state(INTERRUPTION_STATE_KEY).await.unwrap(),
        None,
        "no interruption is persisted"
    );
}

/// An ask-user answer becomes the tool result without executing the tool.
#[tokio::test]
async fn ask_user_answer_becomes_the_tool_result() {
    let model = ScriptedModel::builder()
        .completion(MockCompletion::tool_call(
            "write",
            json!({"question": "Which city?"}),
        ))
        .respond_text("settled")
        .build();
    let agent = agent(&model);
    let log = log();
    agent.add_tool(recording_tool("write", &log));
    agent.register_rail(Arc::new(AskUserRail::new()));
    let session = new_session();

    let paused = agent
        .invoke(json!({"query": "do it"}), Some(Arc::clone(&session)))
        .await
        .unwrap();
    assert_eq!(paused.result_type, ResultType::Interrupt);
    assert_eq!(paused.state.len(), 1);
    let request = request_of(&paused.state[0]);
    assert_eq!(request["interrupt_kind"], json!("ask_user"));
    assert_eq!(request["message"], json!("Which city?"));

    let resumed = agent
        .invoke(
            resume(&json!({"call_write": "Paris"})),
            Some(Arc::clone(&session)),
        )
        .await
        .unwrap();

    assert_eq!(resumed.result_type, ResultType::Answer);
    assert_eq!(resumed.output, "settled");
    assert!(log.lock().is_empty(), "the ask-user tool never runs");
    let calls = model.calls();
    assert_eq!(calls.len(), 2);
    assert_eq!(
        texts_with_role(&calls[1].messages, &Role::Tool),
        vec!["Paris".to_owned()],
        "the answer becomes the tool result"
    );
}

/// Two matching calls of one model turn pause together and resume one by one.
#[tokio::test]
async fn two_pending_calls_pause_once_and_resume_individually() {
    let model = ScriptedModel::builder()
        .completion(MockCompletion::tool_calls(vec![
            MockToolCall::new("write", json!({"path": "a"})).with_id("call_a"),
            MockToolCall::new("write", json!({"path": "b"})).with_id("call_b"),
        ]))
        .respond_text("done")
        .build();
    let agent = agent(&model);
    let log = log();
    agent.add_tool(recording_tool("write", &log));
    agent.register_rail(Arc::new(ConfirmInterruptRail::for_tools(["write"])));
    let session = new_session();

    let paused = agent
        .invoke(json!({"query": "do both"}), Some(Arc::clone(&session)))
        .await
        .unwrap();

    assert_eq!(paused.result_type, ResultType::Interrupt);
    assert_eq!(paused.interrupt_ids.len(), 2, "one request per call");
    assert_eq!(paused.state.len(), 2, "one frame per call");
    assert_eq!(
        frame_call_ids(&paused.state),
        vec!["call_a".to_owned(), "call_b".to_owned()]
    );
    assert_ne!(
        paused.interrupt_ids[0], paused.interrupt_ids[1],
        "each pending call has its own interrupt id"
    );
    assert_eq!(
        paused.state[0].payload["id"],
        json!(paused.interrupt_ids[0])
    );
    assert_eq!(
        paused.state[1].payload["id"],
        json!(paused.interrupt_ids[1])
    );
    assert!(log.lock().is_empty(), "no call runs while both are pending");
    let call_b_interrupt = paused.interrupt_ids[1].clone();

    // Answering only the first call replays it and pauses on the second one.
    let again = agent
        .invoke(
            resume(&json!({"call_a": {"approved": true}})),
            Some(Arc::clone(&session)),
        )
        .await
        .unwrap();

    assert_eq!(again.result_type, ResultType::Interrupt);
    assert_eq!(
        again.interrupt_ids,
        vec![call_b_interrupt],
        "the remaining request is replayed, not re-minted"
    );
    assert_eq!(again.state.len(), 1);
    assert_eq!(again.state[0].payload["id"], json!(again.interrupt_ids[0]));
    assert_eq!(frame_call_ids(&again.state), vec!["call_b".to_owned()]);
    let executed = log.lock().clone();
    assert_eq!(executed.len(), 1, "only the approved call ran");
    assert_eq!(executed[0], json!({"path": "a"}));

    // Answering the second call finishes the turn.
    let finished = agent
        .invoke(
            resume(&json!({"call_b": {"approved": true}})),
            Some(Arc::clone(&session)),
        )
        .await
        .unwrap();

    assert_eq!(finished.result_type, ResultType::Answer);
    assert_eq!(finished.output, "done");
    let executed = log.lock().clone();
    assert_eq!(executed.len(), 2, "each call ran exactly once");
    assert_eq!(
        executed
            .iter()
            .filter(|arg| **arg == json!({"path": "a"}))
            .count(),
        1
    );
    assert_eq!(
        executed
            .iter()
            .filter(|arg| **arg == json!({"path": "b"}))
            .count(),
        1
    );
    assert_eq!(
        session.get_state(INTERRUPTION_STATE_KEY).await.unwrap(),
        None,
        "the finished turn clears the interruption"
    );
}

/// The streaming path publishes the pause frames and mirrors them into the
/// session before the terminal `answer` frame.
#[tokio::test]
async fn streaming_pause_emits_the_interaction_frame_and_mirrors_it() {
    let model = ScriptedModel::builder()
        .completion(MockCompletion::tool_call("write", json!({"path": "x"})))
        .respond_text("done")
        .build();
    let agent = agent(&model);
    let log = log();
    agent.add_tool(recording_tool("write", &log));
    agent.register_rail(Arc::new(ConfirmInterruptRail::for_tools(["write"])));
    let session = new_session();
    // Subscribe before the run so the mirrored frames are not missed.
    let mut events = session.subscribe();
    let chunks = agent
        .stream(json!({"query": "do it"}), Some(Arc::clone(&session)))
        .map(|chunk| chunk.unwrap())
        .collect::<Vec<_>>()
        .await;
    let interactions = chunks
        .iter()
        .filter(|chunk| chunk.kind == "__interaction__")
        .collect::<Vec<_>>();
    assert_eq!(interactions.len(), 1, "{chunks:?}");
    let frame = interactions[0];
    assert_eq!(frame.payload["value"]["interrupt_kind"], json!("confirm"));
    assert_eq!(frame.payload["value"]["tool_call_id"], json!("call_write"));
    let terminal = chunks.last().expect("the stream has a terminal frame");
    assert_eq!(terminal.kind, ANSWER, "{chunks:?}");
    assert_eq!(terminal.payload["result_type"], json!("interrupt"));
    assert!(
        !terminal.payload["state"]
            .as_array()
            .expect("the paused output carries its frames")
            .is_empty(),
        "the terminal frame carries the pause state: {terminal:?}"
    );
    assert!(log.lock().is_empty(), "the paused tool never runs");
    let mut mirrored = Vec::new();
    let deadline = tokio::time::Instant::now() + Duration::from_millis(500);
    loop {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            break;
        }
        match tokio::time::timeout(remaining, events.next()).await {
            Ok(Some(Ok(StreamEvent::Item(item)))) => {
                let terminal_item = item.item_type == ANSWER;
                mirrored.push(item.item_type);
                if terminal_item {
                    break;
                }
            }
            Ok(Some(Ok(StreamEvent::End)) | None) | Err(_) => break,
            Ok(Some(Err(error))) => panic!("session stream failed: {error}"),
        }
    }
    assert!(
        mirrored.iter().any(|kind| kind == "__interaction__"),
        "the interaction frame is mirrored into the session: {mirrored:?}"
    );
}
