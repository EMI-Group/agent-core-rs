//! Client-tool wiring: pause, resume, and the handler dispatch.
//!
//! A mounted tool whose card declares
//! [`ToolExecution::Client`] never runs in this runtime: the turn pauses with a
//! `client_tool` interrupt, and the resumed turn completes the call through its
//! registered [`FnClientToolHandler`]. A handler failure becomes a failed tool
//! result, so the model still sees what happened.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::{Tool, ToolCard, ToolContext, ToolExecution};
use openjiuwen_session::{AgentSession, Session};
use openjiuwen_test_support::{MockCompletion, ScriptedModel};
use serde_json::{json, Value};

use crate::agent::Agent;
use crate::interrupt::InterruptionState;
use crate::output::ResultType;
use crate::policy::{ClientToolRegistry, FnClientToolHandler};

use super::agent;

/// A tool the connected client is supposed to execute.
struct ClientTool {
    card: ToolCard,
    calls: Arc<AtomicUsize>,
}

#[async_trait::async_trait]
impl Tool for ClientTool {
    fn card(&self) -> &ToolCard {
        &self.card
    }

    async fn invoke(&self, _inputs: Value, _ctx: &ToolContext) -> Result<Value> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Err(Error::tool("client tool must not run locally"))
    }
}

/// The mounted `ask_client` tool: a client-executed card with a call counter.
fn client_tool(calls: &Arc<AtomicUsize>) -> Arc<dyn Tool> {
    Arc::new(ClientTool {
        card: ToolCard::new("ask_client", "asks the client", json!({"type": "object"}))
            .with_execution(ToolExecution::Client),
        calls: Arc::clone(calls),
    })
}

/// A registry wiring `ask_client` to `handler`.
fn registry_with(handler: FnClientToolHandler) -> Arc<ClientToolRegistry> {
    let registry = ClientToolRegistry::new();
    registry.register("ask_client", Arc::new(handler));
    Arc::new(registry)
}

#[tokio::test]
async fn client_tool_pauses_and_resume_runs_the_handler() {
    let calls = Arc::new(AtomicUsize::new(0));
    let model = ScriptedModel::builder()
        .completion(MockCompletion::tool_call(
            "ask_client",
            json!({"q": "name?"}),
        ))
        .respond_text("settled")
        .build();
    let agent = agent(Arc::new(model));
    agent.add_tool(client_tool(&calls));
    agent.set_client_tools(registry_with(FnClientToolHandler::new(
        "handler",
        |_call, answer| Box::pin(async move { Ok(json!({"answer": answer})) }),
    )));
    let session: Arc<dyn Session> = Arc::new(AgentSession::new("ct1"));

    let paused = agent
        .invoke(json!("go"), Some(Arc::clone(&session)))
        .await
        .expect("the turn pauses");
    assert!(paused.is_interrupt());
    assert_eq!(paused.state.len(), 1, "one request is surfaced");

    let request = &paused.state[0].payload["value"];
    assert_eq!(request["interrupt_kind"], json!("client_tool"));
    assert_eq!(
        request["message"],
        json!("Please run this tool and return its result")
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        0,
        "the tool never ran locally"
    );
    let call_id = request["tool_call_id"]
        .as_str()
        .expect("the request carries its tool call id")
        .to_owned();
    assert_ne!(call_id, "");

    let resume = json!({
        "query": {"user_inputs": {call_id.clone(): {"approved": true, "name": "Ada"}}},
    });
    let resumed = agent
        .invoke(resume, Some(Arc::clone(&session)))
        .await
        .expect("the resumed turn completes");
    assert_eq!(resumed.result_type, ResultType::Answer);
    assert_eq!(resumed.output, "settled");
    let observation = resumed.observations.first().expect("one observation");
    assert!(
        observation.content.contains("Ada"),
        "the client's answer reaches the model: {}",
        observation.content
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        0,
        "the mounted tool still never ran"
    );
    assert_eq!(
        InterruptionState::load(&*session)
            .await
            .expect("state readable"),
        None,
        "a resumed turn clears the persisted interruption"
    );
}

#[tokio::test]
async fn client_tool_handler_error_becomes_a_failed_result() {
    let calls = Arc::new(AtomicUsize::new(0));
    let model = ScriptedModel::builder()
        .completion(MockCompletion::tool_call(
            "ask_client",
            json!({"q": "name?"}),
        ))
        .respond_text("settled")
        .build();
    let agent = agent(Arc::new(model));
    agent.add_tool(client_tool(&calls));
    agent.set_client_tools(registry_with(FnClientToolHandler::new(
        "handler",
        |_call, _answer| Box::pin(async move { Err(Error::tool("client failed")) }),
    )));
    let session: Arc<dyn Session> = Arc::new(AgentSession::new("ct2"));

    let paused = agent
        .invoke(json!("go"), Some(Arc::clone(&session)))
        .await
        .expect("the turn pauses");
    assert!(paused.is_interrupt());
    let frame = paused.state.first().expect("one request is surfaced");
    let call_id = frame.payload["value"]["tool_call_id"]
        .as_str()
        .expect("the request carries its tool call id")
        .to_owned();

    let resume = json!({
        "query": {"user_inputs": {call_id.clone(): {"approved": true, "name": "Ada"}}},
    });
    let resumed = agent
        .invoke(resume, Some(Arc::clone(&session)))
        .await
        .expect("the resumed turn completes");
    assert_eq!(resumed.output, "settled");

    let observation = resumed.observations.first().expect("one observation");
    assert!(
        observation.content.contains("client failed"),
        "the handler error becomes the tool result: {}",
        observation.content
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}
