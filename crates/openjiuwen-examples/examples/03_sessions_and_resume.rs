//! `03_sessions_and_resume` — persistence and human-in-the-loop in one session.
//!
//! Two seams that make an agent run survive a restart:
//!
//! - **Checkpoints.** An `AgentSession` with a `FileCheckpointer` writes its state
//!   to a temporary directory. A second session built with the same id and the
//!   same checkpointer root resumes by checkpoint id and reads the state back, so
//!   the state really crossed process-equivalent boundaries (nothing is shared in
//!   memory between the two sessions).
//! - **Interrupt and resume.** A `ConfirmInterruptRail` pauses the run before a
//!   tool call: the first `invoke` returns `ResultType::Interrupt` with the pending
//!   interrupt ids and does *not* execute the tool. Re-invoking the same agent with
//!   the same session and an interactive payload (`{"query": {"user_inputs":
//!   {"<tool_call_id>": {"approved": true}}}}`) executes the tool and finishes with
//!   `ResultType::Answer`.
//!
//! Python counterparts: `examples/client_tool/client_tool_demo.py` (session
//! checkpointing) and `examples/harness/worktree/single_deepagent_e2e.py`
//! (interrupt/resume in a harness run).

#![allow(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::items_after_statements
)]

mod common;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{ensure, Context, Result};
use openjiuwen::core::foundation::{FunctionTool, ToolContext};
use openjiuwen::core::session::{
    AgentSession, AgentSessionOptions, Checkpointer, FileCheckpointer, Session,
};
use openjiuwen::core::single_agent::{
    Agent, AgentCard, ConfirmInterruptRail, ReActAgent, ReActAgentConfig, ResultType,
};
use serde_json::{json, Value};

/// Session id used by the checkpoint half of the example.
const CHECKPOINT_SESSION_ID: &str = "checkpoint-demo";

/// Session id used by the interrupt half of the example.
const INTERRUPT_SESSION_ID: &str = "resume-demo";

/// Tool name the confirm rail intercepts.
const WRITE_TOOL: &str = "write_file";

/// Model name written into the agent config; the scripted model ignores it.
const MODEL_NAME: &str = "gpt-4o-mini";

/// Provider written into the agent config.
const MODEL_PROVIDER: &str = "openai";

/// Upper bound on one model turn.
const TURN_TIMEOUT: Duration = Duration::from_secs(60);

/// Builds a session that persists into `checkpointer` under `session_id`.
fn persistent_session(session_id: &str, checkpointer: Arc<dyn Checkpointer>) -> AgentSession {
    AgentSession::with_options(AgentSessionOptions {
        session_id: Some(session_id.to_owned()),
        checkpointer: Some(checkpointer),
        ..AgentSessionOptions::default()
    })
}

/// A tool that records that it ran, so the example can prove when it executed.
fn write_tool(executions: Arc<AtomicUsize>) -> Result<Arc<dyn openjiuwen::core::foundation::Tool>> {
    let tool = FunctionTool::builder()
        .name(WRITE_TOOL)
        .description("Write a note into the demo workspace.")
        .input_schema(json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" },
                "content": { "type": "string" }
            },
            "required": ["path", "content"]
        }))
        .handler(move |args: Value, _ctx: ToolContext| {
            let executions = Arc::clone(&executions);
            async move {
                executions.fetch_add(1, Ordering::SeqCst);
                let path = args.get("path").and_then(Value::as_str).unwrap_or_default();
                Ok(json!({ "written": true, "path": path }))
            }
        })
        .build()?;
    Ok(tool.boxed())
}

/// The checkpoint half: save a named checkpoint, then resume it from a new session.
async fn demonstrate_checkpoints() -> Result<()> {
    common::banner::title("part A: checkpoint persistence");
    let dir = common::tempdir()?;
    let checkpointer: Arc<dyn Checkpointer> = Arc::new(FileCheckpointer::new(dir.path()));
    common::banner::step(&format!("checkpoint root: {}", dir.path().display()));

    let session = persistent_session(CHECKPOINT_SESSION_ID, Arc::clone(&checkpointer));
    session
        .update_state(json!({ "topic": "capitals", "step": 1 }))
        .await
        .context("updating the session state failed")?;
    let checkpoint_id = session
        .save_checkpoint(Some("cp-1"))
        .await
        .context("saving the checkpoint failed")?;

    let before = session.get_state("").await?.unwrap_or(Value::Null);
    common::banner::note(&format!(
        "session {} saved checkpoint {checkpoint_id}",
        session.id()
    ));
    common::banner::note(&format!("state before restart: {before}"));

    // A brand-new session over the same on-disk root: nothing is carried over in
    // memory, so whatever comes back was read from the checkpoint store.
    let restored = persistent_session(CHECKPOINT_SESSION_ID, checkpointer);
    ensure!(
        restored.id() == CHECKPOINT_SESSION_ID,
        "the restored session must reuse the saved session id"
    );
    ensure!(
        restored.resume(&checkpoint_id).await?,
        "checkpoint {checkpoint_id} was not found by the second session"
    );

    let after = restored.get_state("").await?.unwrap_or(Value::Null);
    common::banner::note(&format!("state after resume: {after}"));
    ensure!(
        restored.get_state("topic").await? == Some(json!("capitals")),
        "the saved topic did not survive the checkpoint"
    );
    ensure!(
        restored.get_state("step").await? == Some(json!(1)),
        "the saved step did not survive the checkpoint"
    );
    common::banner::answer("checkpoint state restored by a second session");

    Ok(())
}

/// The interrupt half: pause before the tool, approve, then let the run finish.
async fn demonstrate_interrupt() -> Result<()> {
    common::banner::title("part B: interrupt and resume");

    let executions = Arc::new(AtomicUsize::new(0));
    let agent = ReActAgent::new(AgentCard::new("resume-agent"));
    let mut config = ReActAgentConfig::new();
    config
        .configure_model(MODEL_NAME, MODEL_PROVIDER)
        .configure_max_iterations(4);
    agent.configure(config);
    // Turn one asks for the tool, turn two answers once the result is fed back.
    agent.set_model(common::scripted_tool_round(
        WRITE_TOOL,
        json!({ "path": "notes.txt", "content": "hello from openjiuwen" }),
        "The note is written.",
    ));
    let mounted = agent.add_tool(write_tool(Arc::clone(&executions))?);
    common::banner::note(&format!("mounted {} ({})", mounted.name, mounted.reason));
    agent.register_rail(Arc::new(ConfirmInterruptRail::for_tools([WRITE_TOOL])));
    common::banner::note("the confirm rail pauses every write_file call for approval");

    let session: Arc<dyn Session> = Arc::new(AgentSession::new(INTERRUPT_SESSION_ID));

    common::banner::step("first invoke: expect a pause instead of a tool execution");
    let paused = tokio::time::timeout(
        TURN_TIMEOUT,
        agent.invoke(
            json!({ "query": "Write a note to notes.txt" }),
            Some(Arc::clone(&session)),
        ),
    )
    .await
    .context("the first invoke timed out")??;

    ensure!(
        paused.result_type == ResultType::Interrupt,
        "expected the confirm rail to pause the run, got {:?}",
        paused.result_type
    );
    ensure!(
        executions.load(Ordering::SeqCst) == 0,
        "the tool must not run before the approval"
    );
    common::banner::note(&format!(
        "pending interrupt ids: {}",
        if paused.interrupt_ids.is_empty() {
            "(none reported)".to_owned()
        } else {
            paused.interrupt_ids.join(", ")
        }
    ));

    // The scripted model keys tool calls as `call_<tool_name>`, so the approval is
    // addressed by tool-call id. `interrupt_answers` would also accept the
    // interrupt id from the list above.
    let answers = json!({ "call_write_file": { "approved": true } });
    common::banner::step("second invoke: resume the same session with the approval");
    let resumed = tokio::time::timeout(
        TURN_TIMEOUT,
        agent.invoke(
            json!({
                "query": {
                    "user_inputs": answers,
                    "raw_inputs": null,
                    "interrupt_answers": {}
                },
                "conversation_id": INTERRUPT_SESSION_ID
            }),
            Some(session),
        ),
    )
    .await
    .context("the resume invoke timed out")??;

    ensure!(
        resumed.result_type == ResultType::Answer,
        "expected an answer after the approval, got {:?}",
        resumed.result_type
    );
    ensure!(
        executions.load(Ordering::SeqCst) == 1,
        "the tool must run exactly once after the approval"
    );
    common::banner::note(&format!(
        "iterations used: {}",
        resumed.iterations_used.unwrap_or_default()
    ));
    common::banner::answer(&resumed.output);

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let _mode = common::setup()?;
    common::banner::title("03 sessions and resume");

    demonstrate_checkpoints().await?;
    demonstrate_interrupt().await?;

    Ok(())
}
