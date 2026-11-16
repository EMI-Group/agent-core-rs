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
//! Loop-coordination components through the public workflow API.
//!
//! Covers the four components Python keeps in
//! `openjiuwen/core/workflow/components/flow/loop/loop_comp.py`:
//! `LoopGroupComponent` (`loop_group`), `AdvancedLoopComponent` (`advanced_loop`),
//! `LoopBreakComponent` (`loop_break`) and `LoopSetVariableComponent`
//! (`loop_set_variable`). The shared driver makes every container report the documented
//! wrapper `{"loop_results", "loop_variables", "loop_count", "broken"}`, fold a
//! `.__loop_set_variables__` marker into its loop variables, and stop on a
//! `.__loop_break__` marker without recording the interrupted round.
//!
//! Everything is offline and deterministic: bodies are local components, no model, no
//! clock, no sleep.
mod common;

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::{ErrorKind, Result};
use serde_json::{json, Value};

use openjiuwen_workflow::{
    AdvancedLoopComponent, AdvancedLoopConfig, Component, ComponentCard, ComponentContext,
    ComponentOptions, EndComponent, LoopBound, LoopBreakComponent, LoopComponent, LoopConfig,
    LoopGroupComponent, LoopGroupConfig, LoopSetVariableComponent, LoopType, StartComponent,
    Workflow, WorkflowExecutionState, LOOP_BREAK_MARKER, LOOP_SET_VARIABLES_MARKER,
};

use common::{card, session};

/// Increments the loop variable `total`, so a variable-carrying loop counts its rounds.
#[derive(Debug, Clone, Default)]
struct IncrementComponent;

#[async_trait]
impl Component for IncrementComponent {
    fn card(&self) -> ComponentCard {
        ComponentCard::new("increment", "increment")
    }

    fn component_type(&self) -> &str {
        "increment"
    }

    async fn invoke(&self, inputs: Value, _ctx: &ComponentContext) -> Result<Value> {
        let total = inputs["total"].as_u64().unwrap_or(0) + 1;
        Ok(json!({"total": total, "index": inputs["index"], "item": inputs["item"]}))
    }
}

/// The variable mapping Python writes: `{"{\loop.total}": "<the body output>"}`.
fn total_mapping() -> BTreeMap<String, Value> {
    let mut mapping = BTreeMap::new();
    let _ = mapping.insert("${loop.total}".to_owned(), json!("${total}"));
    mapping
}

/// `[<IncrementComponent>, <LoopSetVariableComponent>]`, optionally prefixed by a break
/// component with `condition`.
fn body(break_at: Option<&str>) -> Vec<Arc<dyn Component>> {
    let mut body: Vec<Arc<dyn Component>> = Vec::new();
    if let Some(condition) = break_at {
        body.push(Arc::new(
            LoopBreakComponent::new().with_condition(condition),
        ));
    }
    body.push(Arc::new(IncrementComponent));
    body.push(Arc::new(LoopSetVariableComponent::new(total_mapping())));
    body
}

/// `start → <container> → end`.
fn container_workflow(
    id: &str,
    component: impl Component + 'static,
    start_schema: Value,
    container_schema: Value,
) -> Workflow {
    let mut workflow = Workflow::new(card(id));
    workflow
        .set_start_comp(
            "start",
            StartComponent::new(),
            ComponentOptions::new().with_inputs_schema(start_schema),
        )
        .add_workflow_comp(
            "loop",
            component,
            ComponentOptions::new().with_inputs_schema(container_schema),
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

#[tokio::test]
async fn a_number_group_reports_the_documented_wrapper() {
    let workflow = container_workflow(
        "group-number",
        LoopGroupComponent::new(LoopGroupConfig::new().with_loop_number(3), body(None)),
        json!({}),
        json!({}),
    );
    let output = workflow
        .invoke(json!({}), session("group-number"))
        .await
        .unwrap();

    assert_eq!(output.state, WorkflowExecutionState::Completed);
    let result = &output.result["output"];
    assert_eq!(result["loop_count"], json!(3));
    assert_eq!(result["broken"], json!(false));
    assert_eq!(result["loop_variables"], json!({"total": 3}));
    let results = result["loop_results"].as_array().unwrap();
    assert_eq!(results.len(), 3);
    assert_eq!(results[2], json!({"total": 3, "index": 2, "item": null}));
}

#[tokio::test]
async fn an_array_group_iterates_over_the_referenced_array() {
    let workflow = container_workflow(
        "group-array",
        LoopGroupComponent::new(
            LoopGroupConfig::new()
                .with_loop_type(LoopBound::Array)
                .with_loop_array("${start.items}"),
            body(None),
        ),
        json!({"items": "${inputs.items}"}),
        json!({"start": "${start}"}),
    );
    let output = workflow
        .invoke(json!({"items": ["a", "b", "c"]}), session("group-array"))
        .await
        .unwrap();

    let result = &output.result["output"];
    assert_eq!(result["loop_count"], json!(3));
    assert_eq!(result["loop_results"][1]["item"], json!("b"));
}

#[tokio::test]
async fn a_condition_group_reevaluates_the_condition_with_the_updated_variables() {
    let workflow = container_workflow(
        "group-condition",
        LoopGroupComponent::new(
            LoopGroupConfig::new()
                .with_loop_type(LoopBound::Condition)
                .with_bool_expression("${total} < 2")
                .with_variable("total", json!(0)),
            body(None),
        ),
        json!({}),
        json!({}),
    );
    let output = workflow
        .invoke(json!({}), session("group-condition"))
        .await
        .unwrap();

    // `total` starts at 0, is incremented by the body, and stops the loop at 2.
    assert_eq!(output.result["output"]["loop_count"], json!(2));
    assert_eq!(
        output.result["output"]["loop_variables"],
        json!({"total": 2})
    );
}

#[tokio::test]
async fn a_set_variable_component_accumulates_a_variable_across_iterations() {
    // The body reads `total` from the loop variables and writes it back, so every round
    // sees the previous round's value.
    let group = LoopGroupComponent::new(
        LoopGroupConfig::new()
            .with_loop_number(4)
            .with_variable("total", json!(0)),
        vec![
            Arc::new(IncrementComponent),
            Arc::new(LoopSetVariableComponent::new(total_mapping())),
        ],
    );
    let output = group.invoke(json!({}), &context()).await.unwrap();

    assert_eq!(output["loop_variables"], json!({"total": 4}));
    assert_eq!(
        output["loop_results"],
        json!([
            {"total": 1, "index": 0, "item": null},
            {"total": 2, "index": 1, "item": null},
            {"total": 3, "index": 2, "item": null},
            {"total": 4, "index": 3, "item": null},
        ])
    );
}

#[tokio::test]
async fn a_break_component_stops_an_unbounded_group_without_recording_the_round() {
    let group = LoopGroupComponent::new(
        LoopGroupConfig::new()
            .with_loop_type(LoopBound::UntilBreak)
            .with_max_iterations(10),
        body(Some("${index} >= 2")),
    );
    let output = group.invoke(json!({}), &context()).await.unwrap();

    assert_eq!(output["broken"], json!(true));
    assert_eq!(output["loop_count"], json!(2));
    assert_eq!(output["loop_results"].as_array().unwrap().len(), 2);
    assert_eq!(output["loop_variables"], json!({"total": 2}));
}

#[tokio::test]
async fn an_unconditional_break_stops_the_first_round() {
    let group = LoopGroupComponent::new(
        LoopGroupConfig::new()
            .with_loop_type(LoopBound::UntilBreak)
            .with_max_iterations(10),
        vec![
            Arc::new(LoopBreakComponent::new()),
            Arc::new(IncrementComponent),
        ],
    );
    let output = group.invoke(json!({}), &context()).await.unwrap();

    assert_eq!(output["broken"], json!(true));
    assert_eq!(output["loop_count"], json!(0));
    assert_eq!(output["loop_results"], json!([]));
}

#[tokio::test]
async fn an_advanced_loop_terminates_on_its_condition_or_on_the_cap() {
    // The condition ends the loop once `total` reaches 2.
    let condition = AdvancedLoopComponent::new(
        AdvancedLoopConfig::new()
            .with_bool_expression("${total} < 2")
            .with_max_iterations(50)
            .with_variable("total", json!(0)),
        body(None),
    );
    let output = condition.invoke(json!({}), &context()).await.unwrap();
    assert_eq!(output["loop_count"], json!(2));
    assert_eq!(output["broken"], json!(false));

    // A condition that never goes false is bounded by `max_iterations`.
    let capped = AdvancedLoopComponent::new(
        AdvancedLoopConfig::new()
            .with_bool_expression("true")
            .with_max_iterations(2),
        body(None),
    );
    let error = capped.invoke(json!({}), &context()).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Workflow);
    assert!(error.message().contains("[101040]"), "{error}");
    assert!(error.message().contains("maximum limit 2"), "{error}");
}

#[tokio::test]
async fn an_advanced_loop_applies_variable_updates_before_the_next_round() {
    let component = AdvancedLoopComponent::new(
        AdvancedLoopConfig::new()
            .with_loop_number(3)
            .with_bool_expression("${total} < 10")
            .with_variable("total", json!(0)),
        body(None),
    );
    let output = component.invoke(json!({}), &context()).await.unwrap();
    assert_eq!(output["loop_count"], json!(3));
    assert_eq!(output["loop_variables"], json!({"total": 3}));
}

#[tokio::test]
async fn an_advanced_loop_stops_on_a_break_component() {
    let component = AdvancedLoopComponent::new(
        AdvancedLoopConfig::new().with_max_iterations(10),
        body(Some("${index} == 1")),
    );
    let output = component.invoke(json!({}), &context()).await.unwrap();
    assert_eq!(output["broken"], json!(true));
    assert_eq!(output["loop_count"], json!(1));
}

#[tokio::test]
async fn a_group_nested_in_a_loop_component_runs_per_outer_iteration() {
    let inner = LoopGroupComponent::new(LoopGroupConfig::new().with_loop_number(2), body(None));
    let outer = LoopComponent::new(
        LoopConfig::new(LoopType::Number).with_loop_number(2),
        vec![Arc::new(inner) as Arc<dyn Component>],
    );
    let output = outer.invoke(json!({}), &context()).await.unwrap();

    let results = output["loop_results"].as_array().unwrap();
    assert_eq!(results.len(), 2);
    for round in results {
        assert_eq!(round["loop_count"], json!(2));
        assert_eq!(round["loop_variables"], json!({"total": 2}));
    }
}

#[tokio::test]
async fn a_group_inside_a_workflow_is_bounded_by_the_session_limit() {
    let session = session("group-cap");
    session
        .config()
        .expect("the session exposes a configuration")
        .write()
        .loop_number_max_limit = 1;
    let workflow = container_workflow(
        "group-cap",
        LoopGroupComponent::new(LoopGroupConfig::new().with_loop_number(3), body(None)),
        json!({}),
        json!({}),
    );
    let failure = workflow.invoke(json!({}), session).await.unwrap_err();
    assert!(failure.message().contains("[101040]"), "{failure}");
    assert!(failure.message().contains("maximum limit 1"), "{failure}");
}

#[tokio::test]
async fn a_standalone_set_variable_component_returns_its_marker() {
    let component = LoopSetVariableComponent::new(total_mapping());
    let output = component
        .invoke(json!({"total": 7}), &context())
        .await
        .unwrap();
    assert_eq!(output[LOOP_SET_VARIABLES_MARKER], json!({"total": 7}));

    let break_component = LoopBreakComponent::new();
    let output = break_component.invoke(json!({}), &context()).await.unwrap();
    assert_eq!(output[LOOP_BREAK_MARKER], json!(true));

    let gated = LoopBreakComponent::new().with_condition("${index} > 1");
    let output = gated.invoke(json!({"index": 1}), &context()).await.unwrap();
    assert_eq!(output[LOOP_BREAK_MARKER], json!(false));
}

/// A bare context for the direct-invoke tests.
fn context() -> ComponentContext {
    ComponentContext::new(session("loop-coordination"), "loop", "wf")
}
