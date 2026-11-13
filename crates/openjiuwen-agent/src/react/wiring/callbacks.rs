//! Lifecycle-callback wiring: order, payloads, error path, failure isolation.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::Arc;

use openjiuwen_common::error::{Error, ErrorKind, Result};
use openjiuwen_test_support::{MockCompletion, ScriptedError, ScriptedModel, ScriptedTurn};
use parking_lot::Mutex;
use serde::Serialize;
use serde_json::{json, Value};

use crate::agent::Agent;
use crate::callbacks::{
    AgentCallback, AgentCallbackEvent, InvokeEndPayload, InvokeStartPayload, StepEndPayload,
    StepStartPayload, ToolCallEndPayload, ToolCallPayload,
};

use super::{agent, echo_tool};

/// Fired events and their serialized payloads, in firing order.
type Recorded = Arc<Mutex<Vec<(AgentCallbackEvent, Value)>>>;

/// A callback that records `(event, payload)` for every hook it implements.
struct Recorder {
    events: Recorded,
}

impl Recorder {
    /// Wraps a fresh event log in a callback handle.
    fn new(events: &Recorded) -> Arc<Self> {
        Arc::new(Self {
            events: Arc::clone(events),
        })
    }

    /// Serializes `payload` and appends it under `event`.
    fn push<T: Serialize>(&self, event: AgentCallbackEvent, payload: &T) {
        self.events.lock().push((
            event,
            serde_json::to_value(payload).expect("payload serializes"),
        ));
    }
}

#[async_trait::async_trait]
impl AgentCallback for Recorder {
    fn name(&self) -> &'static str {
        "recorder"
    }

    async fn on_invoke_start(&self, payload: &InvokeStartPayload) -> Result<()> {
        self.push(AgentCallbackEvent::InvokeStart, payload);
        Ok(())
    }

    async fn on_step_start(&self, payload: &StepStartPayload) -> Result<()> {
        self.push(AgentCallbackEvent::StepStart, payload);
        Ok(())
    }

    async fn on_step_end(&self, payload: &StepEndPayload) -> Result<()> {
        self.push(AgentCallbackEvent::StepEnd, payload);
        Ok(())
    }

    async fn on_tool_call_start(&self, payload: &ToolCallPayload) -> Result<()> {
        self.push(AgentCallbackEvent::ToolCallStart, payload);
        Ok(())
    }

    async fn on_tool_call_end(&self, payload: &ToolCallEndPayload) -> Result<()> {
        self.push(AgentCallbackEvent::ToolCallEnd, payload);
        Ok(())
    }

    async fn on_invoke_end(&self, payload: &InvokeEndPayload) -> Result<()> {
        self.push(AgentCallbackEvent::InvokeEnd, payload);
        Ok(())
    }
}

/// A callback whose invoke/step hooks always fail.
struct Failing;

#[async_trait::async_trait]
impl AgentCallback for Failing {
    fn name(&self) -> &'static str {
        "failing"
    }

    async fn on_invoke_start(&self, _payload: &InvokeStartPayload) -> Result<()> {
        Err(Error::tool("callback boom"))
    }

    async fn on_step_start(&self, _payload: &StepStartPayload) -> Result<()> {
        Err(Error::tool("callback boom"))
    }
}

/// Creates an empty event log.
fn event_log() -> Recorded {
    Arc::new(Mutex::new(Vec::new()))
}

/// Registers `recorder` for every lifecycle event.
fn record_all(agent: &super::ReActAgent, recorder: &Arc<Recorder>) {
    let callback = Arc::clone(recorder);
    let callback: Arc<dyn AgentCallback> = callback;
    for event in AgentCallbackEvent::ALL {
        agent.register_callback(event, Arc::clone(&callback));
    }
}

#[tokio::test]
async fn event_order_and_payloads_across_two_iterations() {
    let model = ScriptedModel::builder()
        .completion(MockCompletion::tool_call("echo", json!({"x": 1})))
        .respond_text("done")
        .build();
    let agent = agent(Arc::new(model));
    agent.add_tool(echo_tool());
    let log = event_log();
    let recorder = Recorder::new(&log);
    record_all(&agent, &recorder);

    let output = agent.invoke(json!("hi"), None).await.expect("run succeeds");
    assert_eq!(output.output, "done");

    let entries = log.lock().clone();
    let order = entries.iter().map(|(event, _)| *event).collect::<Vec<_>>();
    assert_eq!(
        order,
        vec![
            AgentCallbackEvent::InvokeStart,
            AgentCallbackEvent::StepStart,
            AgentCallbackEvent::StepEnd,
            AgentCallbackEvent::ToolCallStart,
            AgentCallbackEvent::ToolCallEnd,
            AgentCallbackEvent::StepStart,
            AgentCallbackEvent::StepEnd,
            AgentCallbackEvent::InvokeEnd,
        ]
    );

    let invoke_start = &entries[0].1;
    assert_eq!(invoke_start["query"], json!("hi"));
    assert_eq!(invoke_start["max_iterations"], json!(5));
    assert_eq!(invoke_start["agent_id"], json!("react-test"));

    let first_step_start = &entries[1].1;
    assert_eq!(first_step_start["iteration"], json!(1));
    assert!(
        !first_step_start["messages"]
            .as_array()
            .expect("messages serialized")
            .is_empty(),
        "the first step sends a non-empty window"
    );
    assert!(
        first_step_start["tools"]
            .as_array()
            .expect("tools serialized")
            .iter()
            .any(|tool| tool["name"] == json!("echo")),
        "the mounted tool is offered to the model"
    );

    assert_eq!(entries[2].1["tool_call_count"], json!(1));

    assert_eq!(entries[3].1["call"]["name"], json!("echo"));

    let tool_call_end = &entries[4].1;
    assert_eq!(tool_call_end["call"]["name"], json!("echo"));
    assert_eq!(tool_call_end["success"], json!(true));
    assert!(
        !tool_call_end["text"]
            .as_str()
            .expect("text serialized")
            .is_empty(),
        "a successful tool call carries its result text"
    );

    let invoke_end = &entries[7].1;
    assert_eq!(invoke_end["output"]["output"], json!("done"));
    assert_eq!(invoke_end["output"]["result_type"], json!("answer"));
}

#[tokio::test]
async fn error_path_still_fires_invoke_end() {
    let model = ScriptedModel::new([ScriptedTurn::fail_with(ScriptedError::llm("boom"))]);
    let agent = agent(Arc::new(model));
    let log = event_log();
    let recorder = Recorder::new(&log);
    record_all(&agent, &recorder);

    let error = agent
        .invoke(json!("hi"), None)
        .await
        .expect_err("the model failure propagates");
    assert_eq!(error.kind(), ErrorKind::Llm);

    let entries = log.lock().clone();
    let (event, payload) = entries.last().expect("invoke end recorded");
    assert_eq!(*event, AgentCallbackEvent::InvokeEnd);
    assert_eq!(payload["output"]["result_type"], json!("error"));
    assert!(
        payload["output"]["output"]
            .as_str()
            .expect("output serialized")
            .contains("boom"),
        "the terminal output reports the failure"
    );
}

#[tokio::test]
async fn callback_error_does_not_abort_the_run() {
    let model = ScriptedModel::builder().respond_text("done").build();
    let agent = agent(Arc::new(model));
    let failing: Arc<dyn AgentCallback> = Arc::new(Failing);
    agent.register_callback(AgentCallbackEvent::InvokeStart, Arc::clone(&failing));
    agent.register_callback(AgentCallbackEvent::StepStart, Arc::clone(&failing));
    let log = event_log();
    let recorder = Recorder::new(&log);
    agent.register_callback(AgentCallbackEvent::InvokeEnd, recorder);

    let output = agent
        .invoke(json!("hi"), None)
        .await
        .expect("callback errors are logged, never propagated");
    assert_eq!(output.output, "done");
    assert!(
        log.lock()
            .iter()
            .any(|(event, _)| *event == AgentCallbackEvent::InvokeEnd),
        "the run still reaches its invoke-end callback"
    );
}
