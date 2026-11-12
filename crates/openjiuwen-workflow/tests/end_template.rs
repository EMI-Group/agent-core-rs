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
//! End-node response templates through the public workflow API.
//!
//! The end component has two modes: flat (`{"output": …}`) and template
//! (`{"response": "<rendered text>"}` for `invoke`, one frame per template piece for
//! `stream`). These tests pin the template grammar (`{{path}}` segments), the rule that
//! missing segments contribute nothing, the JSON-text rendering of non-string values, and
//! the streaming frame order of an array-valued segment over a buffered producer.
//!
//! Ports of `tests/unit_tests/core/component/test_end.py`, rewritten against `Workflow`
//! and `ComponentOptions`. Everything is offline and deterministic: no sleeps, no
//! network, no stdout.
mod common;

use std::sync::Arc;

use futures::StreamExt;
use openjiuwen_session::Session;
use openjiuwen_test_support::ScriptedModel;
use serde_json::{json, Value};

use openjiuwen_workflow::{
    ComponentAbility, ComponentOptions, EndComponent, EndConfig, StartComponent, Workflow,
    WorkflowChunk, WorkflowExecutionState, END_NODE_STREAM, WORKFLOW_FINAL,
};

use common::{assistant_text, card, end_with_template, llm_component, session, StreamingProducer};

/// `start → end`, where `start` republishes the whole run input under `values` and the
/// end node resolves its template against that object.
fn template_workflow(id: &str, template: &str) -> Workflow {
    let mut workflow = Workflow::new(card(id));
    workflow
        .set_start_comp(
            "start",
            StartComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"values": "${inputs}"})),
        )
        .set_end_comp(
            "end",
            end_with_template(template),
            ComponentOptions::new().with_inputs_schema(json!("${start.values}")),
        )
        .add_connection("start", "end");
    workflow
}

/// Collects every frame of a run, panicking on the first failure.
async fn run_stream(
    workflow: &Workflow,
    inputs: Value,
    session: Arc<dyn Session>,
) -> Vec<WorkflowChunk> {
    workflow
        .stream(inputs, session)
        .map(|frame| frame.unwrap())
        .collect()
        .await
}

// port of tests/unit_tests/core/component/test_end.py::test_simple_template_workflow
#[tokio::test]
async fn a_batch_template_renders_the_llm_answer() {
    let model = Arc::new(ScriptedModel::builder().respond_text("hello there").build());

    let mut workflow = Workflow::new(card("end-template"));
    workflow
        .set_start_comp(
            "start",
            StartComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"query": "${inputs.query}"})),
        )
        .add_workflow_comp(
            "llm",
            llm_component(model, "{{query}}"),
            ComponentOptions::new().with_inputs_schema(json!({"query": "${start.query}"})),
        )
        .set_end_comp(
            "end",
            EndComponent::with_config(EndConfig::new("Answer: {{llm.result}}")),
            ComponentOptions::new().with_inputs_schema(json!({"llm": "${llm}"})),
        )
        .add_connection("start", "llm")
        .add_connection("llm", "end");

    let output = workflow
        .invoke(json!({"query": "hi"}), session("end-template-batch"))
        .await
        .unwrap();

    assert_eq!(output.state, WorkflowExecutionState::Completed);
    // Template mode returns `{"response": …}` directly: the end component's contract has
    // no `{"output": …}` wrapper here (that wrapper belongs to flat mode).
    assert_eq!(output.result, json!({"response": "Answer: hello there"}));
}

// port of tests/unit_tests/core/component/test_end.py::test_end_template_001
#[tokio::test]
async fn missing_segments_contribute_nothing_and_values_render_as_json_text() {
    let missing = template_workflow("end-missing", "{{missing.path}}!");
    let output = missing
        .invoke(json!({"other": 1}), session("end-template-missing"))
        .await
        .unwrap();
    assert_eq!(output.result, json!({"response": "!"}));

    let values = template_workflow("end-values", "n={{n}} obj={{obj}}");
    let output = values
        .invoke(
            json!({"n": 3, "obj": {"a": 1}}),
            session("end-template-values"),
        )
        .await
        .unwrap();
    // Non-string batch values render as JSON text (`serde_json::to_string`), the
    // documented deviation from Python's `str()`/`repr`.
    assert_eq!(output.result, json!({"response": r#"n=3 obj={"a":1}"#}));
}

// port of tests/unit_tests/core/component/test_end.py::test_end_stream_template
#[tokio::test]
async fn an_array_template_streams_one_frame_per_element() {
    let mut workflow = Workflow::new(card("end-array-stream"));
    workflow
        .set_start_comp(
            "start",
            StartComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"items": "${inputs.items}"})),
        )
        .set_end_comp(
            "end",
            end_with_template("{{items}}"),
            ComponentOptions::new()
                .with_inputs_schema(json!({"items": "${start.items}"}))
                .streaming(),
        )
        .add_connection("start", "end");

    let chunks = run_stream(
        &workflow,
        json!({"items": ["a", "b", "c"]}),
        session("end-array-stream"),
    )
    .await;

    let final_chunk = chunks.last().unwrap();
    assert_eq!(final_chunk.item_type, WORKFLOW_FINAL);

    let frames = &chunks[..chunks.len() - 1];
    assert_eq!(frames.len(), 3, "one frame per array element");
    for (index, chunk) in frames.iter().enumerate() {
        assert_eq!(chunk.item_type, END_NODE_STREAM);
        assert_eq!(chunk.index, Some(i64::try_from(index).unwrap()));
    }
    let payloads: Vec<Value> = frames.iter().map(|chunk| chunk.payload.clone()).collect();
    assert_eq!(
        payloads,
        vec![
            json!({"response": "a"}),
            json!({"response": "b"}),
            json!({"response": "c"}),
        ]
    );

    // The final payload is the wire list of the frames the end node produced.
    assert_eq!(
        final_chunk.payload,
        json!([
            {"type": END_NODE_STREAM, "index": 0, "payload": {"response": "a"}},
            {"type": END_NODE_STREAM, "index": 1, "payload": {"response": "b"}},
            {"type": END_NODE_STREAM, "index": 2, "payload": {"response": "c"}},
        ])
    );
}

// port of tests/unit_tests/core/component/test_end.py::test_end_stream_workflow
#[tokio::test]
async fn a_template_projects_a_streaming_producers_buffered_frames() {
    let mut workflow = Workflow::new(card("end-producer-stream"));
    workflow
        .set_start_comp(
            "start",
            StartComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"query": "${inputs.query}"})),
        )
        .add_workflow_comp(
            "producer",
            // The producer only implements `Stream`, so the ability is declared instead
            // of derived from a stream edge; the edge stays regular because the end node
            // consumes the producer's buffered output, not a live stream.
            StreamingProducer::new(vec![
                json!({"output": "Hello "}),
                json!({"output": "world"}),
            ]),
            ComponentOptions::new().with_abilities(vec![ComponentAbility::Stream]),
        )
        .set_end_comp(
            "end",
            end_with_template("{{producer.output}}"),
            ComponentOptions::new()
                .with_inputs_schema(json!({"producer": "${producer}"}))
                .streaming(),
        )
        .add_connection("start", "producer")
        .add_connection("producer", "end");

    let chunks = run_stream(
        &workflow,
        json!({"query": "hi"}),
        session("end-producer-stream"),
    )
    .await;

    let final_chunk = chunks.last().unwrap();
    assert_eq!(final_chunk.item_type, WORKFLOW_FINAL);

    let payloads: Vec<Value> = chunks[..chunks.len() - 1]
        .iter()
        .map(|chunk| chunk.payload.clone())
        .collect();
    assert_eq!(
        payloads,
        vec![json!({"response": "Hello "}), json!({"response": "world"})],
        "`${{producer.output}}` projects the field over the producer's buffered frames"
    );
    assert_eq!(assistant_text(&payloads), "Hello world");
}

// port of tests/unit_tests/core/component/test_end.py::test_end_template_002
#[tokio::test]
async fn a_template_can_reference_the_end_nodes_own_resolved_inputs() {
    let mut workflow = Workflow::new(card("end-own-inputs"));
    workflow
        .set_start_comp(
            "start",
            StartComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"query": "${inputs.query}"})),
        )
        .set_end_comp(
            "end",
            EndComponent::with_config(EndConfig::new("Q: {{query}}")),
            ComponentOptions::new().with_inputs_schema(json!({"query": "${start.query}"})),
        )
        .add_connection("start", "end");

    let output = workflow
        .invoke(json!({"query": "hello"}), session("end-own-inputs"))
        .await
        .unwrap();

    assert_eq!(output.result, json!({"response": "Q: hello"}));
}
