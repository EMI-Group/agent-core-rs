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
//! Loop component behavior through the public workflow API.
//!
//! The loop component runs a flat body per iteration and returns
//! `{"loop_results": [<iteration output>, …]}`. Iteration inputs are the loop's resolved
//! inputs overlaid with `index`/`item`/`previous`; the effective cap is the tighter of
//! the component's `max_iterations` and the session's `loop_number_max_limit`.
//!
//! Ports of `tests/unit_tests/core/component/test_loop_comp.py`, rewritten against
//! `Workflow`/`ComponentOptions`. Everything is offline and deterministic: the body is a
//! local component and no wall-clock or sleep is used.
mod common;

use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::{ErrorKind, Result};
use serde_json::{json, Value};

use openjiuwen_workflow::{
    Component, ComponentCard, ComponentContext, ComponentOptions, EndComponent, LoopComponent,
    LoopConfig, LoopType, StartComponent, Workflow, WorkflowExecutionState,
};

use common::{card, session};

/// The loop body: reports the iteration position it was handed.
#[derive(Debug, Clone, Default)]
struct IndexingComponent;

#[async_trait]
impl Component for IndexingComponent {
    fn card(&self) -> ComponentCard {
        ComponentCard::new("indexing", "indexing")
    }

    fn component_type(&self) -> &str {
        "indexing"
    }

    async fn invoke(&self, inputs: Value, _ctx: &ComponentContext) -> Result<Value> {
        Ok(json!({"index": inputs["index"], "item": inputs["item"]}))
    }
}

/// `start → loop → end`, with the loop driven by `config` over one [`IndexingComponent`].
fn loop_workflow(
    id: &str,
    config: LoopConfig,
    start_schema: Value,
    loop_schema: Value,
) -> Workflow {
    let body: Vec<Arc<dyn Component>> = vec![Arc::new(IndexingComponent) as Arc<dyn Component>];
    let mut workflow = Workflow::new(card(id));
    workflow
        .set_start_comp(
            "start",
            StartComponent::new(),
            ComponentOptions::new().with_inputs_schema(start_schema),
        )
        .add_workflow_comp(
            "loop",
            LoopComponent::new(config, body),
            ComponentOptions::new().with_inputs_schema(loop_schema),
        )
        .set_end_comp(
            "end",
            EndComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!("${loop}")),
        )
        .add_connection("start", "loop")
        .add_connection("loop", "end");
    workflow
}

// port of tests/unit_tests/core/component/test_loop_comp.py::test_loop_number
#[tokio::test]
async fn a_number_loop_runs_the_configured_iteration_count() {
    let workflow = loop_workflow(
        "loop-number",
        LoopConfig::new(LoopType::Number).with_loop_number(3),
        json!({}),
        json!({}),
    );

    let output = workflow
        .invoke(json!({}), session("loop-number"))
        .await
        .unwrap();

    assert_eq!(output.state, WorkflowExecutionState::Completed);
    assert_eq!(
        output.result,
        json!({
            "output": {
                "loop_results": [
                    {"index": 0, "item": null},
                    {"index": 1, "item": null},
                    {"index": 2, "item": null},
                ]
            }
        })
    );
}

// port of tests/unit_tests/core/component/test_loop_comp.py::test_loop_group_component_stream
#[tokio::test]
async fn an_array_loop_iterates_over_the_referenced_array() {
    let workflow = loop_workflow(
        "loop-array",
        LoopConfig::new(LoopType::Array).with_loop_array("${start.items}"),
        json!({"items": "${inputs.items}"}),
        json!({"start": "${start}"}),
    );

    let output = workflow
        .invoke(json!({"items": ["a", "b", "c"]}), session("loop-array"))
        .await
        .unwrap();

    assert_eq!(
        output.result,
        json!({
            "output": {
                "loop_results": [
                    {"index": 0, "item": "a"},
                    {"index": 1, "item": "b"},
                    {"index": 2, "item": "c"},
                ]
            }
        })
    );
}

// port of tests/unit_tests/core/component/test_loop_comp.py::test_loop_number
#[tokio::test]
async fn a_condition_loop_stops_on_the_first_false_condition() {
    let workflow = loop_workflow(
        "loop-condition",
        LoopConfig::new(LoopType::Condition).with_bool_expression("${index} < 3"),
        json!({}),
        json!({}),
    );
    let output = workflow
        .invoke(json!({}), session("loop-condition"))
        .await
        .unwrap();
    assert_eq!(
        output.result["output"]["loop_results"],
        json!([
            {"index": 0, "item": null},
            {"index": 1, "item": null},
            {"index": 2, "item": null},
        ])
    );

    let never = loop_workflow(
        "loop-never",
        LoopConfig::new(LoopType::Condition).with_bool_expression("${index} < 0"),
        json!({}),
        json!({}),
    );
    let output = never
        .invoke(json!({}), session("loop-never"))
        .await
        .unwrap();
    assert_eq!(output.result, json!({"output": {"loop_results": []}}));
}

// port of tests/unit_tests/core/component/test_loop_comp.py::test_loop_number_exceeds_max_limit
#[tokio::test]
async fn the_session_limit_caps_the_iteration_count() {
    let session = session("loop-cap");
    session
        .config()
        .expect("the session exposes a configuration")
        .write()
        .loop_number_max_limit = 2;

    let workflow = loop_workflow(
        "loop-cap",
        LoopConfig::new(LoopType::Number).with_loop_number(3),
        json!({}),
        json!({}),
    );
    let failure = workflow.invoke(json!({}), session).await.unwrap_err();

    assert_eq!(failure.kind(), ErrorKind::Workflow);
    assert!(failure.message().contains("[101040]"), "{failure}");
    assert!(failure.message().contains("maximum limit 2"), "{failure}");
}

// port of tests/unit_tests/core/component/test_loop_comp.py::test_loop_number
#[tokio::test]
async fn a_missing_loop_number_is_a_loop_execution_error() {
    // Python raises `COMPONENT_LOOP_PARAM_INVALID` (`101030`) while parsing the loop
    // config; this component validates at invoke time and reports the shared loop
    // execution code (`101040`).
    let workflow = loop_workflow(
        "loop-missing",
        LoopConfig::new(LoopType::Number),
        json!({}),
        json!({}),
    );

    let failure = workflow
        .invoke(json!({}), session("loop-missing"))
        .await
        .unwrap_err();

    assert_eq!(failure.kind(), ErrorKind::Workflow);
    assert!(failure.message().contains("[101040]"), "{failure}");
    assert!(
        failure.message().contains("loop_number variable not found"),
        "{failure}"
    );
}
