//! Shared fixtures for the agent integration suites.
//!
//! Every suite is its own crate, so helpers used by only one of them still
//! trigger dead-code warnings here; that is expected and silenced below.
//! Nothing in this module touches the network: sessions are in-process
//! `AgentSession`s, models are `ScriptedModel`s, and tools are
//! `FunctionTool`s over plain closures.
#![allow(dead_code, unreachable_pub, clippy::unwrap_used, clippy::expect_used)]
use std::sync::Arc;

use openjiuwen_agent::{AgentCard, ReActAgent, ReActAgentConfig};
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::{FunctionTool, Model, Tool, ToolContext};
use openjiuwen_session::{AgentSession, Session};
use parking_lot::Mutex;
use serde_json::{json, Value};

/// The session id every fixture agent uses when a suite does not care.
pub const SESSION_ID: &str = "s1";

/// Creates an in-process agent session with `id`.
pub fn session(id: &str) -> Arc<dyn Session> {
    Arc::new(AgentSession::new(id))
}

/// Builds a `ReActAgent` configured with `config` and driven by `model`.
pub fn agent(config: ReActAgentConfig, model: Arc<dyn Model>) -> ReActAgent {
    let agent = ReActAgent::new(AgentCard::new("react-test"));
    agent.configure(config);
    agent.set_model(model);
    agent
}

/// Builds a tool named `name` that records its parsed inputs into `log` and
/// answers with the fixed text `"tool-output"`.
///
/// The returned tool is parallel-safe, so two of them in one model turn run as
/// one concurrent batch.
pub fn recording_tool(name: &str, log: &Arc<Mutex<Vec<Value>>>) -> Arc<dyn Tool> {
    let log = Arc::clone(log);
    let tool = FunctionTool::builder()
        .name(name)
        .description("recording tool")
        .input_schema(json!({"type": "object"}))
        .handler(move |inputs: Value, _ctx: ToolContext| {
            // `handler` requires an `Fn`, so the log is cloned per call rather
            // than moved out of the closure.
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

/// Builds a tool named `name` whose handler always fails with `Error::tool`.
pub fn failing_tool(name: &str) -> Arc<dyn Tool> {
    let tool = FunctionTool::builder()
        .name(name)
        .description("failing tool")
        .input_schema(json!({"type": "object"}))
        .handler(move |_inputs: Value, _ctx: ToolContext| async move {
            Err(Error::tool("boom")) as Result<Value>
        })
        .build()
        .expect("the failing tool fixture builds");
    Arc::new(tool)
}

/// Returns a fresh log a [`recording_tool`] writes into.
pub fn log() -> Arc<Mutex<Vec<Value>>> {
    Arc::new(Mutex::new(Vec::new()))
}
