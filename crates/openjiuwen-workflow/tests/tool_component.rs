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
//! Tool component behavior through the public workflow API.
//!
//! A tool failure is data, not an exception: `ToolComponent::invoke` always returns
//! `{"error_code", "error_message", "data"}` and the workflow still completes. The codes
//! follow Python's split — `102000` for a rejected call (`Tool`/`Validation`) and
//! `182012` for any other kind.
//!
//! Ports of `tests/unit_tests/core/component/test_tool_comp.py`, rewritten against
//! `Workflow`/`ComponentOptions`. Everything is offline: the tools are local closures or
//! an in-test recording tool, and no network or sleep is involved.
mod common;

use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::{FunctionTool, Tool, ToolCard, ToolContext};
use parking_lot::Mutex;
use serde_json::{json, Value};

use openjiuwen_workflow::{
    ComponentOptions, EndComponent, StartComponent, ToolComponent, Workflow, WorkflowExecutionState,
};

use common::{b_of_a_tool, card, session};

/// `start → tool → end`, where the tool node resolves `{"a": "${start.a}"}` and the end
/// node reports the tool's batch result verbatim.
fn tool_workflow(id: &str, tool: Arc<dyn Tool>) -> Workflow {
    let mut workflow = Workflow::new(card(id));
    workflow
        .set_start_comp(
            "start",
            StartComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"a": "${inputs.a}"})),
        )
        .add_workflow_comp(
            "tool",
            ToolComponent::new(tool),
            ComponentOptions::new().with_inputs_schema(json!({"a": "${start.a}"})),
        )
        .set_end_comp(
            "end",
            EndComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!("${tool}")),
        )
        .add_connection("start", "tool")
        .add_connection("tool", "end");
    workflow
}

/// A tool that always fails with `error`.
fn failing_tool(error: fn() -> Error) -> Arc<dyn Tool> {
    FunctionTool::builder()
        .name("failing")
        .description("always fails")
        .handler(move |_inputs, _ctx| async move { Err(error()) })
        .build()
        .expect("the failing tool has a handler")
        .boxed()
}

/// A tool that records every input object it receives.
struct RecordingTool {
    card: ToolCard,
    seen: Arc<Mutex<Vec<Value>>>,
}

impl RecordingTool {
    fn new() -> Self {
        Self {
            card: ToolCard::new("recording", "records its inputs", json!({"type": "object"})),
            seen: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn seen(&self) -> Vec<Value> {
        self.seen.lock().clone()
    }
}

#[async_trait]
impl Tool for RecordingTool {
    fn card(&self) -> &ToolCard {
        &self.card
    }

    async fn invoke(&self, inputs: Value, _ctx: &ToolContext) -> Result<Value> {
        self.seen.lock().push(inputs.clone());
        Ok(json!({"seen": inputs}))
    }
}

// port of tests/unit_tests/core/component/test_tool_comp.py::test_invoke_workflow_with_start_tool_end
#[tokio::test]
async fn a_successful_tool_reports_code_zero_and_its_data() {
    let output = tool_workflow("tool-success", b_of_a_tool())
        .invoke(json!({"a": 41}), session("tool-success"))
        .await
        .unwrap();

    assert_eq!(output.state, WorkflowExecutionState::Completed);
    assert_eq!(
        output.result,
        json!({
            "output": {
                "error_code": 0,
                "error_message": "",
                "data": {"b": 2, "got": 41},
            }
        })
    );
}

// port of tests/unit_tests/core/component/test_tool_comp.py::test_tool_comp_in_workflow
#[tokio::test]
async fn a_tool_error_is_captured_with_code_102000() {
    let output = tool_workflow("tool-error", failing_tool(|| Error::tool("kaboom")))
        .invoke(json!({"a": 1}), session("tool-error"))
        .await
        .unwrap();

    assert_eq!(
        output.state,
        WorkflowExecutionState::Completed,
        "a tool failure must not fail the workflow"
    );
    assert_eq!(output.result["output"]["error_code"], json!(102_000));
    assert_eq!(output.result["output"]["error_message"], json!("kaboom"));
    assert_eq!(output.result["output"]["data"], Value::Null);
}

// port of tests/unit_tests/core/component/test_tool_comp.py::test_tool_comp_invoke
#[tokio::test]
async fn another_error_kind_is_reported_with_code_182012() {
    let output = tool_workflow(
        "tool-transport",
        failing_tool(|| Error::transport("offline")),
    )
    .invoke(json!({"a": 1}), session("tool-transport"))
    .await
    .unwrap();

    assert_eq!(output.state, WorkflowExecutionState::Completed);
    assert_eq!(output.result["output"]["error_code"], json!(182_012));
    assert_eq!(output.result["output"]["error_message"], json!("offline"));
    assert_eq!(output.result["output"]["data"], Value::Null);
}

// port of tests/unit_tests/core/component/test_tool_comp.py::test_invoke_workflow_with_start_tool_end_with_compatible_interface
#[tokio::test]
async fn the_tool_receives_the_inputs_resolved_from_its_schema() {
    let tool = Arc::new(RecordingTool::new());
    let handle: Arc<dyn Tool> = tool.clone();

    let output = tool_workflow("tool-inputs", handle)
        .invoke(json!({"a": 7, "ignored": true}), session("tool-inputs"))
        .await
        .unwrap();

    assert_eq!(
        tool.seen(),
        vec![json!({"a": 7})],
        "only the keys named by the tool node's inputs_schema reach the tool"
    );
    assert_eq!(output.result["output"]["data"], json!({"seen": {"a": 7}}));
}
