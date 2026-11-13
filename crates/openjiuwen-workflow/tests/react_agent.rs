#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(
    dead_code,
    unreachable_pub,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::new_without_default,
    clippy::module_name_repetitions,
    clippy::unnecessary_literal_bound,
    clippy::assert_is_empty
)]
//! `ReAct` component behavior through the public workflow API.
//!
//! Pins the `react_agent` node contract: the resolved inputs reach the
//! [`AgentRunner`] unchanged, the answer is normalized to
//! `{"output", "result_type"}`, a runner failure becomes `result_type: "error"`
//! data (the workflow still completes), and a HITL question becomes
//! `result_type: "interrupt"` data. The bundled [`ModelReActRunner`] is pinned
//! against a `ScriptedModel`: one tool round feeds the tool message back to the
//! model, and each `on_max_iterations` policy reports its own shape.
//!
//! Everything is offline and deterministic: the model is `ScriptedModel`, tools
//! are in-process closures, sessions are in-process `WorkflowSession`s, and the
//! assertions read the recorded call history instead of timing anything.
mod common;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, ErrorKind, Result};
use openjiuwen_foundation::{FunctionTool, Model, Role, Tool};
use openjiuwen_test_support::{MockCompletion, ScriptedModel, ScriptedTurn};
use parking_lot::Mutex;
use serde_json::{json, Value};

use openjiuwen_workflow::components::react_agent::{
    AgentRunner, ModelReActRunner, ReActAgentComponent, ReActAgentConfig,
};
use openjiuwen_workflow::{
    Component, ComponentContext, ComponentOptions, EndComponent, StartComponent, Workflow,
    WorkflowExecutionState,
};

use common::{b_of_a_tool, card, session};

/// A component context for direct (non-workflow) node invocations.
fn context(id: &str) -> ComponentContext {
    ComponentContext::new(session(id), "react", id)
}

/// `start → react → end`, where the react node resolves `{"query": "${start.query}"}`
/// and the end node reports the node's `output` under `answer`.
fn react_workflow(id: &str, component: ReActAgentComponent) -> Workflow {
    let mut workflow = Workflow::new(card(id));
    workflow
        .set_start_comp(
            "start",
            StartComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"query": "${inputs.query}"})),
        )
        .add_workflow_comp(
            "react",
            component,
            ComponentOptions::new().with_inputs_schema(json!({"query": "${start.query}"})),
        )
        .set_end_comp(
            "end",
            EndComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"answer": "${react.output}"})),
        )
        .add_connection("start", "react")
        .add_connection("react", "end");
    workflow
}

/// A runner that records its inputs and always answers `answer`.
struct RecordingRunner {
    answer: Value,
    seen: Mutex<Vec<Value>>,
}

impl RecordingRunner {
    fn new(answer: Value) -> Self {
        Self {
            answer,
            seen: Mutex::new(Vec::new()),
        }
    }

    fn seen(&self) -> Vec<Value> {
        self.seen.lock().clone()
    }
}

#[async_trait]
impl AgentRunner for RecordingRunner {
    async fn invoke(&self, inputs: Value, _ctx: &ComponentContext) -> Result<Value> {
        self.seen.lock().push(inputs);
        Ok(self.answer.clone())
    }
}

/// A runner that fails every invocation with `error`.
struct FailingRunner {
    error: fn() -> Error,
}

#[async_trait]
impl AgentRunner for FailingRunner {
    async fn invoke(&self, _inputs: Value, _ctx: &ComponentContext) -> Result<Value> {
        Err((self.error)())
    }
}

/// A runner that asks a HITL question and returns the interrupt unchanged.
struct InterruptingRunner;

#[async_trait]
impl AgentRunner for InterruptingRunner {
    async fn invoke(&self, _inputs: Value, ctx: &ComponentContext) -> Result<Value> {
        let question = json!({"id": "q", "text": "continue?"});
        // `?` propagates the interrupt error unchanged, like `InteractingComponent`.
        let answer = ctx.interact(question).await?;
        Ok(json!({"answer": answer}))
    }
}

/// A `b_of_a` tool whose handler counts its invocations.
fn counting_b_of_a(calls: Arc<AtomicUsize>) -> Arc<dyn Tool> {
    FunctionTool::builder()
        .name("b_of_a")
        .description("returns b")
        .input_schema(json!({
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "required": ["a"]
        }))
        .handler(move |inputs, _ctx| {
            let calls = Arc::clone(&calls);
            async move {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(json!({"b": 2, "got": inputs["a"]}))
            }
        })
        .build()
        .expect("the b_of_a tool has a handler")
        .boxed()
}

/// A runner over `model` with the given tools and iteration budget.
fn runner(
    model: &ScriptedModel,
    tools: Vec<Arc<dyn Tool>>,
    config: ReActAgentConfig,
) -> ModelReActRunner {
    let model: Arc<dyn Model> = Arc::new(model.clone());
    ModelReActRunner::new(config.with_model(model).with_tools(tools))
        .expect("the runner config is valid")
}

/// An always-tool-calling model: one scripted call, then the same call forever.
fn always_calls_a_tool() -> ScriptedModel {
    let call = || MockCompletion::tool_call("b_of_a", json!({"a": 41}));
    ScriptedModel::builder()
        .completion(call())
        .fallback(ScriptedTurn::Response(call().to_chat_response("")))
        .build()
}

/// The text of every `role` message in `messages`.
fn texts_with_role(messages: &[openjiuwen_foundation::BaseMessage], role: &Role) -> Vec<String> {
    messages
        .iter()
        .filter(|message| &message.role == role)
        .map(|message| message.text().into_owned())
        .collect()
}

#[tokio::test]
async fn a_runner_receives_the_resolved_inputs_and_answers_with_data() {
    let runner = Arc::new(RecordingRunner::new(json!("done")));

    // Node-level contract: a scalar answer is wrapped under `output`.
    let node_output = ReActAgentComponent::new(runner.clone())
        .invoke(json!({"query": "hi"}), &context("react-direct"))
        .await
        .unwrap();
    assert_eq!(
        node_output,
        json!({"output": "done", "result_type": "answer"})
    );

    // Workflow-level: the node's `inputs_schema` selects the resolved keys.
    let output = react_workflow("react-inputs", ReActAgentComponent::new(runner.clone()))
        .invoke(
            json!({"query": "hi", "ignored": true}),
            session("react-inputs"),
        )
        .await
        .unwrap();
    assert_eq!(output.state, WorkflowExecutionState::Completed);
    assert_eq!(output.result, json!({"output": {"answer": "done"}}));
    assert_eq!(
        runner.seen(),
        vec![json!({"query": "hi"}), json!({"query": "hi"})],
        "the runner sees the schema-resolved inputs, never the raw run inputs"
    );
}

#[tokio::test]
async fn a_runner_failure_becomes_error_data_and_the_run_completes() {
    let node_output = ReActAgentComponent::new(Arc::new(FailingRunner {
        error: || Error::workflow("boom"),
    }))
    .invoke(json!({"query": "hi"}), &context("react-error"))
    .await
    .unwrap();
    assert_eq!(
        node_output,
        json!({"output": "Error in ReAct execution: boom", "result_type": "error"})
    );

    let output = react_workflow(
        "react-error-flow",
        ReActAgentComponent::new(Arc::new(FailingRunner {
            error: || Error::workflow("boom"),
        })),
    )
    .invoke(json!({"query": "hi"}), session("react-error-flow"))
    .await
    .unwrap();
    assert_eq!(
        output.state,
        WorkflowExecutionState::Completed,
        "a runner failure is data, so the workflow does not fail"
    );
    assert_eq!(
        output.result,
        json!({"output": {"answer": "Error in ReAct execution: boom"}})
    );
}

#[tokio::test]
async fn an_interrupt_becomes_the_question_as_data() {
    let node_output = ReActAgentComponent::new(Arc::new(InterruptingRunner))
        .invoke(json!({"query": "hi"}), &context("react-interrupt"))
        .await
        .unwrap();
    assert_eq!(
        node_output,
        json!({
            "output": {"id": "q", "text": "continue?"},
            "result_type": "interrupt",
        })
    );

    // The `__interaction__` frame was already written by `ctx.interact`, and the
    // component maps the interrupt to data, so the run completes instead of
    // reporting INPUT_REQUIRED.
    let output = react_workflow(
        "react-interrupt-flow",
        ReActAgentComponent::new(Arc::new(InterruptingRunner)),
    )
    .invoke(json!({"query": "hi"}), session("react-interrupt-flow"))
    .await
    .unwrap();
    assert_eq!(output.state, WorkflowExecutionState::Completed);
    assert_eq!(
        output.result,
        json!({"output": {"answer": {"id": "q", "text": "continue?"}}})
    );
}

#[tokio::test]
async fn a_tool_round_feeds_the_tool_message_back_to_the_model() {
    let calls = Arc::new(AtomicUsize::new(0));
    let model = ScriptedModel::builder()
        .completion(MockCompletion::tool_call("b_of_a", json!({"a": 41})))
        .respond_text("final answer")
        .build();
    let runner = runner(
        &model,
        vec![counting_b_of_a(Arc::clone(&calls))],
        ReActAgentConfig::default(),
    );

    let output = ReActAgentComponent::new(Arc::new(runner))
        .invoke(json!({"query": "hi"}), &context("react-tool"))
        .await
        .unwrap();

    assert_eq!(calls.load(Ordering::SeqCst), 1, "the tool ran exactly once");
    assert_eq!(model.call_count(), 2, "one tool round, then the answer");

    let calls = model.calls();
    let first_request = calls[0].request.tools.clone().expect("tools are sent");
    let tools = first_request.as_array().expect("tools are an array");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0]["function"]["name"], json!("b_of_a"));

    let assistant_calls: Vec<_> = calls[1]
        .messages
        .iter()
        .filter(|message| message.role == Role::Assistant && !message.tool_calls.is_empty())
        .collect();
    assert_eq!(
        assistant_calls.len(),
        1,
        "the assistant's tool call is echoed back: {:?}",
        calls[1].messages
    );
    let tool_texts = texts_with_role(&calls[1].messages, &Role::Tool);
    assert_eq!(tool_texts.len(), 1, "{:?}", calls[1].messages);
    assert!(
        tool_texts[0].contains('b'),
        "the rendered tool output reaches the model: {}",
        tool_texts[0]
    );
    assert_eq!(
        output,
        json!({"output": "final answer", "result_type": "answer"})
    );
}

#[tokio::test]
async fn an_exhausted_budget_reports_an_error_by_default() {
    let model = always_calls_a_tool();
    let runner = runner(
        &model,
        vec![b_of_a_tool()],
        ReActAgentConfig::default().with_max_iterations(2),
    );
    let output = ReActAgentComponent::new(Arc::new(runner))
        .invoke(json!({"query": "hi"}), &context("react-max-error"))
        .await
        .unwrap();
    assert_eq!(model.call_count(), 2);
    assert_eq!(
        output,
        json!({
            "output": "Error in ReAct execution: Max iterations reached without completion",
            "result_type": "error",
        })
    );
}

#[tokio::test]
async fn an_exhausted_budget_can_return_the_observations() {
    let model = always_calls_a_tool();
    let runner = runner(
        &model,
        vec![b_of_a_tool()],
        ReActAgentConfig::default()
            .with_max_iterations(2)
            .with_on_max_iterations("return_observations"),
    );
    let output = ReActAgentComponent::new(Arc::new(runner))
        .invoke(json!({"query": "hi"}), &context("react-max-observations"))
        .await
        .unwrap();
    assert_eq!(output["result_type"], json!("answer"));
    let observations = output["output"].as_array().expect("output is an array");
    assert_eq!(observations.len(), 2);
    for observation in observations {
        assert_eq!(observation["tool_name"], json!("b_of_a"));
        assert_eq!(observation["tool_call_id"], json!("call_b_of_a"));
        assert!(
            observation["content"]
                .as_str()
                .is_some_and(|text| !text.is_empty()),
            "{observation:?}"
        );
    }
}

#[tokio::test]
async fn an_exhausted_budget_can_return_the_last_answer() {
    let model = always_calls_a_tool();
    let runner = runner(
        &model,
        vec![b_of_a_tool()],
        ReActAgentConfig::default()
            .with_max_iterations(2)
            .with_on_max_iterations("final_answer"),
    );
    let output = ReActAgentComponent::new(Arc::new(runner))
        .invoke(json!({"query": "hi"}), &context("react-max-final"))
        .await
        .unwrap();
    // The scripted tool-call turns carry no text, so the answer is the empty
    // string; the shape is what this test pins.
    assert_eq!(output["result_type"], json!("answer"));
    assert!(output["output"].is_string(), "{output:?}");
}

#[tokio::test]
async fn a_full_workflow_resolves_the_react_output() {
    let model = ScriptedModel::builder()
        .completion(MockCompletion::tool_call("b_of_a", json!({"a": 41})))
        .respond_text("final answer")
        .build();
    let runner = runner(&model, vec![b_of_a_tool()], ReActAgentConfig::default());
    let output = react_workflow("react-flow", ReActAgentComponent::new(Arc::new(runner)))
        .invoke(json!({"query": "hi"}), session("react-flow"))
        .await
        .unwrap();
    assert_eq!(output.state, WorkflowExecutionState::Completed);
    assert_eq!(output.result, json!({"output": {"answer": "final answer"}}));
}

#[test]
fn an_illegal_max_iteration_policy_is_a_validation_error() {
    let model: Arc<dyn Model> = Arc::new(ScriptedModel::text("x"));
    let error = ModelReActRunner::new(
        ReActAgentConfig::default()
            .with_model(model)
            .with_on_max_iterations("explode"),
    )
    .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(
        error.message(),
        "invalid on_max_iterations 'explode': expected one of error, final_answer, return_observations"
    );

    let error = ReActAgentComponent::from_model(ReActAgentConfig::default()).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(error.message(), "react_agent component requires a model");
}
