#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(
    unreachable_pub,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::new_without_default,
    clippy::module_name_repetitions,
    clippy::unnecessary_literal_bound
)]
//! `ExceptionConfig` / `default_outputs` error recovery.
//!
//! After the graph's retries are exhausted, a failed `Invoke`/`Collect` node whose
//! `ExceptionConfig` declares `default_outputs` under a non-`interrupt` `handle_type`
//! recovers: the defaults are projected through the node's `outputs_schema` and written
//! to graph state under the node id, so downstream `${node.field}` references resolve.
//! An `interrupt` policy (Python's default), a node without an `ExceptionConfig`, and a
//! failed `Stream`/`Transform` keep Python's propagation path (`[100054]`).
//!
//! Ports of `tests/unit_tests/core/workflow/test_workflow.py::test_workflow_error_recovery_with_trigger`
//! and `openjiuwen/core/graph/vertex.py::_run_error_recovery` / `_post_invoke`. Every
//! test is offline and deterministic: in-process components, in-process sessions, no
//! sleeps and no network.
mod common;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::StreamExt;
use openjiuwen_common::error::{Error, ErrorKind, Result};
use serde_json::{json, Value};

use openjiuwen_workflow::{
    Component, ComponentAbility, ComponentCard, ComponentContext, ComponentOptions, EndComponent,
    ExceptionConfig, StartComponent, Workflow, WorkflowConfig, WorkflowExecutionState,
};

use common::{card, end_with_template, session};

/// Fails every invocation and counts them.
#[derive(Debug, Clone, Default)]
struct CountingFailure {
    calls: Arc<AtomicUsize>,
}

impl CountingFailure {
    /// Creates a component that has not failed yet.
    fn new() -> Self {
        Self::default()
    }

    /// How many times the component ran.
    fn count(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl Component for CountingFailure {
    fn card(&self) -> ComponentCard {
        ComponentCard::new("counting_failure", "counting_failure").with_abilities(self.abilities())
    }

    fn abilities(&self) -> Vec<ComponentAbility> {
        vec![ComponentAbility::Invoke]
    }

    fn component_type(&self) -> &str {
        "counting_failure"
    }

    async fn invoke(&self, _inputs: Value, _ctx: &ComponentContext) -> Result<Value> {
        let attempt = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        Err(Error::workflow(format!("attempt {attempt} failed")))
    }
}

/// A `Stream` producer that yields one frame and then fails.
#[derive(Debug, Clone, Default)]
struct FailingStream;

#[async_trait]
impl Component for FailingStream {
    fn card(&self) -> ComponentCard {
        ComponentCard::new("failing_stream", "failing_stream").with_abilities(self.abilities())
    }

    fn abilities(&self) -> Vec<ComponentAbility> {
        vec![ComponentAbility::Stream]
    }

    fn component_type(&self) -> &str {
        "failing_stream"
    }

    fn stream(
        self: Arc<Self>,
        _inputs: Value,
        _ctx: Arc<ComponentContext>,
    ) -> BoxStream<'static, Result<Value>> {
        futures::stream::iter(vec![
            Ok(json!({"delta": "a"})),
            Err(Error::workflow("stream boom")),
        ])
        .boxed()
    }
}

/// A recovery policy that substitutes `defaults`.
fn recovery_config(handle_type: &str, defaults: Value) -> ExceptionConfig {
    ExceptionConfig {
        handle_type: handle_type.to_string(),
        default_outputs: Some(defaults),
        metadata: std::collections::BTreeMap::new(),
    }
}

/// `start → failing → end`, where the failing node carries `options` and the end node
/// resolves `end_inputs`.
fn failing_workflow(
    id: &str,
    failing: impl Component + 'static,
    options: ComponentOptions,
    end_inputs: Value,
) -> Workflow {
    let mut workflow = Workflow::new(card(id));
    workflow
        .set_start_comp("start", StartComponent::new(), ComponentOptions::new())
        .add_workflow_comp("failing", failing, options)
        .set_end_comp(
            "end",
            EndComponent::new(),
            ComponentOptions::new().with_inputs_schema(end_inputs),
        )
        .add_connection("start", "failing")
        .add_connection("failing", "end");
    workflow
}

/// Asserts the failure is Python's component-execution-error path.
fn assert_component_execution_error(error: &Error) {
    assert_eq!(error.kind(), ErrorKind::Workflow);
    let message = error.message();
    assert!(message.contains("[100054]"), "{message}");
}

// port of tests/unit_tests/core/workflow/test_workflow.py::test_workflow_error_recovery_with_trigger
#[tokio::test]
async fn default_outputs_recover_a_failing_invoke_after_retries() {
    let failing = CountingFailure::new();
    let workflow = failing_workflow(
        "recovery",
        failing.clone(),
        ComponentOptions::new()
            .with_max_retries(2)
            .with_exception_config(Some(recovery_config(
                "defaultOutputs",
                json!({"result": "recovery_result"}),
            ))),
        json!({"result": "${failing.result}"}),
    );

    let output = workflow
        .invoke(json!({"query": "test"}), session("recovery"))
        .await
        .unwrap();

    assert_eq!(
        failing.count(),
        3,
        "one initial attempt plus the two configured retries"
    );
    assert_eq!(output.state, WorkflowExecutionState::Completed);
    assert_eq!(
        output.result,
        json!({"output": {"result": "recovery_result"}}),
        "the end node read the recovered value through ${{failing.result}}"
    );
}

#[tokio::test]
async fn an_interrupt_policy_still_propagates_the_failure() {
    let failing = CountingFailure::new();
    let workflow = failing_workflow(
        "interrupt",
        failing.clone(),
        ComponentOptions::new()
            .with_max_retries(2)
            .with_exception_config(Some(recovery_config(
                "interrupt",
                json!({"result": "recovery_result"}),
            ))),
        json!({"result": "${failing.result}"}),
    );

    let failure = workflow
        .invoke(json!({}), session("interrupt"))
        .await
        .unwrap_err();

    assert_eq!(failing.count(), 3, "retries still run before the failure");
    assert_component_execution_error(&failure);
}

#[tokio::test]
async fn a_node_without_an_exception_config_propagates_the_failure() {
    let failing = CountingFailure::new();
    let workflow = failing_workflow(
        "no-config",
        failing.clone(),
        ComponentOptions::new().with_max_retries(2),
        json!({"result": "${failing.result}"}),
    );

    let failure = workflow
        .invoke(json!({}), session("no-config"))
        .await
        .unwrap_err();

    assert_eq!(failing.count(), 3);
    assert_component_execution_error(&failure);
}

#[tokio::test]
async fn recovery_output_is_projected_through_the_outputs_schema() {
    let workflow = failing_workflow(
        "projected",
        CountingFailure::new(),
        ComponentOptions::new()
            .with_outputs_schema(json!({"answer": "${output.result}"}))
            .with_exception_config(Some(recovery_config(
                "defaultOutputs",
                json!({"result": "recovery_result"}),
            ))),
        json!({"answer": "${failing.answer}"}),
    );

    let output = workflow
        .invoke(json!({}), session("projected"))
        .await
        .unwrap();

    assert_eq!(output.state, WorkflowExecutionState::Completed);
    assert_eq!(
        output.result,
        json!({"output": {"answer": "recovery_result"}}),
        "the recovered defaults go through the node's outputs_schema"
    );
}

#[tokio::test]
async fn a_streaming_failure_is_never_recovered() {
    let mut workflow = Workflow::new(card("stream-recovery"));
    workflow
        .set_start_comp("start", StartComponent::new(), ComponentOptions::new())
        .add_workflow_comp(
            "failing",
            FailingStream,
            ComponentOptions::new().with_exception_config(Some(recovery_config(
                "defaultOutputs",
                json!({"result": "recovery_result"}),
            ))),
        )
        .set_end_comp(
            "end",
            end_with_template("{{whole}}"),
            ComponentOptions::new()
                .with_inputs_schema(json!({"first": "${failing[0].delta}"}))
                .with_stream_inputs_schema(json!({"whole": "${failing}"}))
                .streaming(),
        )
        .add_connection("start", "failing")
        .add_stream_connection("failing", "end");

    let failure = workflow
        .invoke(json!({}), session("stream-recovery"))
        .await
        .unwrap_err();

    assert_component_execution_error(&failure);
    assert!(
        failure.message().contains("stream boom"),
        "the streaming error surfaces instead of the defaults: {failure}"
    );
}

#[tokio::test]
async fn an_exception_config_round_trips_through_the_spec() {
    let config = recovery_config("defaultOutputs", json!({"result": "recovery_result"}));
    let workflow = failing_workflow(
        "round-trip",
        CountingFailure::new(),
        ComponentOptions::new().with_exception_config(Some(config.clone())),
        json!({"result": "${failing.result}"}),
    );

    let spec = workflow.spec();
    assert_eq!(
        spec.comp_configs.get("failing").unwrap().exception_config,
        Some(config.clone()),
        "the built spec carries the node's exception config"
    );

    let workflow_config = WorkflowConfig {
        card: workflow.card().clone(),
        spec,
        workflow_max_nesting_depth: 5,
    };
    let wire = serde_json::to_value(&workflow_config).unwrap();
    assert_eq!(
        wire["spec"]["comp_configs"]["failing"]["exception_config"],
        json!({
            "handle_type": "defaultOutputs",
            "default_outputs": {"result": "recovery_result"},
        }),
        "{wire}"
    );

    let parsed: WorkflowConfig = serde_json::from_value(wire).unwrap();
    assert_eq!(
        parsed
            .spec
            .comp_configs
            .get("failing")
            .unwrap()
            .exception_config,
        Some(config)
    );
}
