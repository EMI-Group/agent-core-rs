#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Cards, specs, and options: the serializable JSON contract of a workflow.
//!
//! `WorkflowCard`/`WorkflowSpec` round-trip through JSON, tolerate unknown fields, and
//! match Python's field names and defaults; `Workflow::spec()` reflects the builder's
//! state and agrees with `Workflow::config().spec`; `to_mermaid()` renders a flowchart
//! containing every component id.
//!
//! Ports of `tests/unit_tests/core/workflow/test_workflow_config.py` (card/spec
//! round-trips and defaults) and `test_workflow.py::test_workflow_draw`.
mod common;

use std::collections::BTreeMap;

use serde_json::json;

use openjiuwen_workflow::{
    CompIoConfig, ComponentAbility, ComponentOptions, EndComponent, StartComponent, Workflow,
    WorkflowCard, WorkflowSpec,
};

use common::{card, session, StreamingProducer};

/// `start → mid → end`, where `mid` streams into the end node (so the spec carries a
/// stream edge, a stream source group, and stream abilities on `mid`).
fn spec_workflow() -> Workflow {
    let mut workflow = Workflow::new(card("spec-workflow"));
    workflow
        .set_start_comp(
            "start",
            StartComponent::new(),
            ComponentOptions::new()
                .with_inputs_schema(json!({"query": "${inputs.query}"}))
                .with_timeout_ms(1500)
                .with_max_retries(2)
                .with_name("Start"),
        )
        .add_workflow_comp(
            "mid",
            StreamingProducer::new(vec![json!("frame")]),
            ComponentOptions::new(),
        )
        .set_end_comp("end", EndComponent::new(), ComponentOptions::new())
        .add_connection("start", "mid")
        .add_stream_connection("mid", "end");
    workflow
}

// port of tests/unit_tests/core/workflow/test_workflow_config.py::test_workflow_card_json
#[test]
fn workflow_card_round_trips_and_tolerates_unknown_fields() {
    let card = WorkflowCard {
        id: "greeting".to_owned(),
        name: "Greeting".to_owned(),
        description: "says hello".to_owned(),
        version: "1.2.3".to_owned(),
        input_params: Some(json!({"type": "object"})),
    };

    let wire = serde_json::to_value(&card).unwrap();
    assert_eq!(
        wire,
        json!({
            "id": "greeting",
            "name": "Greeting",
            "description": "says hello",
            "version": "1.2.3",
            "input_params": {"type": "object"},
        })
    );
    assert_eq!(serde_json::from_value::<WorkflowCard>(wire).unwrap(), card);

    // Unknown keys are ignored and missing keys fall back to the card defaults.
    let tolerant: WorkflowCard =
        serde_json::from_value(json!({"id": "tolerant", "future_key": true})).unwrap();
    assert_eq!(tolerant.id, "tolerant");
    assert_eq!(tolerant.name, "");
    assert_eq!(tolerant.description, "");
    assert_eq!(tolerant.version, "");
    assert_eq!(tolerant.input_params, None);
}

// port of tests/unit_tests/core/workflow/test_workflow_config.py::test_workflow_spec_json
#[test]
fn workflow_spec_round_trips_and_tolerates_unknown_fields() {
    let spec: WorkflowSpec = serde_json::from_value(json!({
        "edges": {"start": ["mid"]},
        "stream_edges": {"mid": ["end"]},
        "comp_configs": {
            "start": {"abilities": ["invoke"], "timeout": -1.0, "max_retries": 0},
            "mid": {"abilities": ["stream"], "timeout": 0.25, "max_retries": 1},
        },
        "stream_source_groups": {"end": [["mid"]]},
        "start_nodes": ["start"],
        "future_spec_key": {"ignored": true},
    }))
    .unwrap();

    assert_eq!(
        spec.edges,
        BTreeMap::from([("start".to_owned(), vec!["mid".to_owned()])])
    );
    assert_eq!(
        spec.stream_edges,
        BTreeMap::from([("mid".to_owned(), vec!["end".to_owned()])])
    );
    assert_eq!(
        spec.stream_source_groups,
        BTreeMap::from([("end".to_owned(), vec![vec!["mid".to_owned()]])])
    );
    assert_eq!(spec.start_nodes, vec!["start".to_owned()]);
    assert_eq!(
        spec.comp_configs["mid"].abilities,
        vec![ComponentAbility::Stream]
    );
    assert_eq!(spec.comp_configs["mid"].max_retries, 1);
    assert!((spec.comp_configs["mid"].timeout - 0.25).abs() < f64::EPSILON);
    // An absent `io_configs` pair stays `None` after deserialization.
    assert_eq!(spec.comp_configs["start"].io_configs, None);

    let wire = serde_json::to_value(&spec).unwrap();
    let back: WorkflowSpec = serde_json::from_value(wire).unwrap();
    assert_eq!(back, spec);
}

// port of tests/unit_tests/core/workflow/test_workflow.py::test_workflow_spec
#[test]
fn the_builder_spec_records_edges_and_component_configs() {
    let workflow = spec_workflow();
    let spec = workflow.spec();

    assert_eq!(
        spec.edges,
        BTreeMap::from([("start".to_owned(), vec!["mid".to_owned()])])
    );
    assert_eq!(
        spec.stream_edges,
        BTreeMap::from([("mid".to_owned(), vec!["end".to_owned()])])
    );
    assert_eq!(
        spec.stream_source_groups,
        BTreeMap::from([("end".to_owned(), vec![vec!["mid".to_owned()]])])
    );
    assert_eq!(spec.start_nodes, vec!["start".to_owned()]);
    assert_eq!(
        spec.comp_configs
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        vec!["end", "mid", "start"]
    );

    let start = &spec.comp_configs["start"];
    assert_eq!(
        start.io_configs,
        Some(CompIoConfig {
            inputs_schema: Some(json!({"query": "${inputs.query}"})),
            outputs_schema: None,
        })
    );
    assert_eq!(start.stream_io_configs, None);
    assert_eq!(start.abilities, vec![ComponentAbility::Invoke]);
    assert_eq!(start.max_retries, 2);
    // Milliseconds are exposed as Python's seconds.
    assert!((start.timeout - 1.5).abs() < f64::EPSILON);
    assert_eq!(start.name.as_deref(), Some("Start"));
    assert_eq!(start.exception_config, None);
    assert_eq!(workflow.start_comp_id(), Some("start"));
    assert_eq!(workflow.end_comp_id(), Some("end"));
    assert_eq!(workflow.component_ids(), vec!["start", "mid", "end"]);
}

// port of tests/unit_tests/core/workflow/test_workflow.py::test_component_options_defaults
#[test]
fn component_options_defaults_appear_in_the_spec() {
    let workflow = spec_workflow();
    let spec = workflow.spec();

    let mid = &spec.comp_configs["mid"];
    // `None` abilities mean "infer from the component": the stream producer advertises
    // `Stream` only.
    assert_eq!(mid.abilities, vec![ComponentAbility::Stream]);
    assert_eq!(mid.max_retries, 0);
    assert!((mid.timeout + 1.0).abs() < f64::EPSILON);
    assert_eq!(mid.io_configs, None);
    assert_eq!(mid.stream_io_configs, None);
    assert_eq!(mid.name, None);

    // The end component advertises every ability; the builder does not narrow it.
    assert_eq!(
        spec.comp_configs["end"].abilities,
        vec![
            ComponentAbility::Invoke,
            ComponentAbility::Stream,
            ComponentAbility::Collect,
            ComponentAbility::Transform,
        ]
    );
}

// port of tests/unit_tests/core/workflow/test_workflow.py::test_component_options_streaming
#[tokio::test]
async fn streaming_options_override_the_abilities_in_the_spec() {
    let mut workflow = Workflow::new(card("streaming-options"));
    workflow
        .set_start_comp(
            "start",
            StartComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"query": "${inputs.query}"})),
        )
        .set_end_comp(
            "end",
            EndComponent::new(),
            ComponentOptions::new().streaming(),
        )
        .add_connection("start", "end");

    let spec = workflow.spec();
    assert_eq!(
        spec.comp_configs["end"].abilities,
        vec![ComponentAbility::Invoke, ComponentAbility::Stream]
    );
    assert!((spec.comp_configs["end"].timeout + 1.0).abs() < f64::EPSILON);

    // A direct `invoke` on a streaming end collects the end frames instead of a value.
    let output = workflow
        .invoke(json!({"query": "hi"}), session("streaming-options"))
        .await
        .unwrap();
    assert_eq!(
        output.state,
        openjiuwen_workflow::WorkflowExecutionState::Completed
    );
    assert!(
        output.result.is_array(),
        "unexpected result: {}",
        output.result
    );
}

// port of tests/unit_tests/core/workflow/test_workflow.py::test_config_matches_spec
#[test]
fn config_spec_matches_the_builder_spec() {
    let workflow = spec_workflow();

    assert_eq!(workflow.config().spec, workflow.spec());
    assert_eq!(workflow.config().card, *workflow.card());
    assert_eq!(workflow.config().workflow_max_nesting_depth, 5);
}

// port of tests/unit_tests/core/workflow/test_workflow.py::test_workflow_draw
#[test]
fn to_mermaid_renders_a_flowchart_with_every_component() {
    let mermaid = spec_workflow().to_mermaid().unwrap();

    assert!(mermaid.contains("flowchart"), "{mermaid}");
    for id in ["start", "mid", "end"] {
        assert!(mermaid.contains(id), "missing `{id}` in: {mermaid}");
    }
    assert!(
        mermaid.contains("-->"),
        "expected at least one rendered edge: {mermaid}"
    );
}

// port of tests/unit_tests/core/workflow/test_workflow.py::test_workflow_draw_invalid
#[test]
fn to_mermaid_reports_a_compile_error() {
    let failure = Workflow::new(card("empty")).to_mermaid().unwrap_err();

    assert!(failure.to_string().contains("no start component"));
}
