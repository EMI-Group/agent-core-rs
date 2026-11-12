#![allow(clippy::unwrap_used, clippy::expect_used)]
//! `${node.field}` reference resolution through the public workflow API.
//!
//! A component's `inputs_schema` is a JSON template: a string that is exactly `${…}`
//! resolves against workflow state (any JSON type), everything else is a literal, and a
//! missing path resolves to `null` without failing the run. These tests pin the path
//! grammar, the literal rule, and the `outputs_schema` projection.
//!
//! Ports of `tests/unit_tests/core/workflow/test_workflow.py::test_workflow_references`
//! and the reference cases of `tests/unit_tests/core/workflow/test_reference.py`.
mod common;

use serde_json::{json, Value};

use openjiuwen_workflow::{
    ComponentOptions, EndComponent, StartComponent, Workflow, WorkflowExecutionState,
};

use common::{card, session, EchoComponent};

/// `start → echo → end`, where `echo` resolves `schema` and `end` reports the echo output.
fn reference_workflow(schema: Value) -> Workflow {
    let mut workflow = Workflow::new(card("references"));
    workflow
        .set_start_comp(
            "start",
            StartComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"a": "${inputs.a}"})),
        )
        .add_workflow_comp(
            "echo",
            EchoComponent::new(),
            ComponentOptions::new().with_inputs_schema(schema),
        )
        .set_end_comp(
            "end",
            EndComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"echo": "${echo.echo}"})),
        )
        .add_connection("start", "echo")
        .add_connection("echo", "end");
    workflow
}

/// The `echo` payload of one run of [`reference_workflow`].
fn echo_of(output: &openjiuwen_workflow::WorkflowOutput) -> Value {
    output.result["output"]["echo"].clone()
}

/// The run inputs every reference test starts from.
fn inputs() -> Value {
    json!({"a": {"b": [{"c": 7}, {"c": 8}], "k": "v"}})
}

// port of tests/unit_tests/core/workflow/test_workflow.py::test_workflow_references
#[tokio::test]
async fn dotted_nested_indexed_and_bracket_paths_resolve() {
    let workflow = reference_workflow(json!({
        "direct": "${start.a.k}",
        "nested": "${start.a.b[0]}",
        "deep": "${start.a.b[0].c}",
        "bracket": "${start.a['k']}",
        "negative": "${start.a.b[-1].c}",
    }));

    let output = workflow
        .invoke(inputs(), session("refs-paths"))
        .await
        .unwrap();

    assert_eq!(output.state, WorkflowExecutionState::Completed);
    assert_eq!(
        echo_of(&output),
        json!({
            "direct": "v",
            "nested": {"c": 7},
            "deep": 7,
            "bracket": "v",
            "negative": 8,
        })
    );
}

// port of tests/unit_tests/core/workflow/test_reference.py::test_whole_node_reference
#[tokio::test]
async fn a_whole_node_reference_returns_the_node_output() {
    let workflow = reference_workflow(json!({"whole": "${start}"}));

    let output = workflow
        .invoke(inputs(), session("refs-whole"))
        .await
        .unwrap();

    assert_eq!(
        echo_of(&output),
        json!({"whole": {"a": {"b": [{"c": 7}, {"c": 8}], "k": "v"}}})
    );
}

// port of tests/unit_tests/core/workflow/test_reference.py::test_missing_path_is_null
#[tokio::test]
async fn a_missing_path_resolves_to_null_without_failing() {
    let workflow = reference_workflow(json!({
        "missing": "${start.nope.deep}",
        "index_out_of_range": "${start.a.b[9]}",
        "after_missing": "${start.a.k}",
    }));

    let output = workflow
        .invoke(inputs(), session("refs-missing"))
        .await
        .unwrap();

    assert_eq!(output.state, WorkflowExecutionState::Completed);
    assert_eq!(
        echo_of(&output),
        json!({"missing": null, "index_out_of_range": null, "after_missing": "v"})
    );
}

// port of tests/unit_tests/core/workflow/test_reference.py::test_literal_strings_are_not_resolved
#[tokio::test]
async fn literals_pass_through_unchanged() {
    let workflow = reference_workflow(json!({
        "label": "static",
        "value": "${start.a.k}",
        "impure": "prefix ${start.a.k}",
        "braces": "${not a path",
        "number": 3,
        "flag": true,
    }));

    let output = workflow
        .invoke(inputs(), session("refs-literal"))
        .await
        .unwrap();

    assert_eq!(
        echo_of(&output),
        json!({
            "label": "static",
            "value": "v",
            // Only a string that is exactly `${…}` is a reference.
            "impure": "prefix ${start.a.k}",
            "braces": "${not a path",
            "number": 3,
            "flag": true,
        })
    );
}

// port of tests/unit_tests/core/workflow/test_workflow.py::test_outputs_schema_projection
#[tokio::test]
async fn outputs_schema_projects_the_raw_output() {
    let middle = common::FixedComponent::new(json!({"value": "raw", "nested": {"n": 1}}));
    let mut workflow = Workflow::new(card("projection"));
    workflow
        .set_start_comp(
            "start",
            StartComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"query": "${inputs.query}"})),
        )
        .add_workflow_comp(
            "mid",
            middle,
            ComponentOptions::new().with_outputs_schema(json!({
                // The projection scope is workflow state plus `output` and the node id;
                // there is no `self`/bare-key shorthand (`${value}` resolves to null).
                "answer": "${output.value}",
                "nested": "${mid.nested.n}",
                "bare": "${value}",
            })),
        )
        .set_end_comp(
            "end",
            EndComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"mid": "${mid}"})),
        )
        .add_connection("start", "mid")
        .add_connection("mid", "end");

    let output = workflow
        .invoke(json!({"query": "hello"}), session("refs-projection"))
        .await
        .unwrap();

    assert_eq!(
        output.result,
        json!({"output": {"mid": {"answer": "raw", "nested": 1, "bare": null}}})
    );
}

// port of tests/unit_tests/core/workflow/test_workflow.py::test_root_and_node_references
#[tokio::test]
async fn root_references_resolve_before_and_after_the_start_node() {
    let mut workflow = Workflow::new(card("root-refs"));
    workflow
        .set_start_comp(
            "start",
            StartComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"query": "${inputs.query}"})),
        )
        .add_workflow_comp(
            "echo",
            EchoComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({
                "root": "${query}",
                "node": "${start.query}",
                "reserved": "${inputs.query}",
            })),
        )
        .set_end_comp(
            "end",
            EndComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"echo": "${echo.echo}"})),
        )
        .add_connection("start", "echo")
        .add_connection("echo", "end");

    let output = workflow
        .invoke(json!({"query": "hi"}), session("refs-root"))
        .await
        .unwrap();

    assert_eq!(
        echo_of(&output),
        json!({"root": "hi", "node": "hi", "reserved": "hi"})
    );
}

// port of tests/unit_tests/core/workflow/test_reference.py::test_structured_values_round_trip
#[tokio::test]
async fn arrays_and_objects_round_trip_without_stringification() {
    let nested = json!({
        "list": [1, "two", true, null, {"deep": [1, 2, 3]}],
        "empty_object": {},
        "empty_array": [],
    });
    let workflow = reference_workflow(json!({
        "array": "${start.a.b}",
        "object": "${start.a}",
        "nested": "${start.a.b[1]}",
    }));

    let output = workflow
        .invoke(
            json!({"a": {"b": [nested, "plain"]}}),
            session("refs-structured"),
        )
        .await
        .unwrap();

    let echoed = echo_of(&output);
    assert_eq!(echoed["array"], json!([nested, "plain"]));
    assert_eq!(echoed["object"], json!({"b": [nested, "plain"]}));
    // Arrays and objects stay JSON values; nothing is rendered as text.
    assert_eq!(echoed["nested"], json!("plain"));
    assert_eq!(echoed["array"][0], nested);
}
