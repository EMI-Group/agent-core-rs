#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(
    unreachable_pub,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::new_without_default,
    clippy::module_name_repetitions,
    clippy::unnecessary_literal_bound
)]
//! Workflow streaming: end-node frames, the terminal `workflow_final` frame, custom
//! frames, mode filtering, failure delivery, and buffered producer composition.
//!
//! A `Stream`/`Transform` producer is drained into an array under its node id; the end
//! node replays its own template frames as indexed `end node stream` output frames and
//! `invoke` returns their `{"type", "index", "payload"}` wires. `stream` forwards those
//! frames in production order and finishes with one `workflow_final` frame carrying the
//! end output. These tests pin that contract, including mode filtering and drop safety.
//!
//! Ports of `tests/unit_tests/core/workflow/test_workflow_with_comp_stream.py`
//! (`test_workflow_stream`, `test_end_stream*`). Every test is offline and deterministic:
//! frames are produced from in-memory streams, never from sleeps.
mod common;

use std::time::Duration;

use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::StreamExt;
use openjiuwen_common::error::{ErrorKind, Result};
use openjiuwen_session::stream::{StreamItem, StreamMode};
use serde_json::{json, Value};

use openjiuwen_workflow::{
    Component, ComponentAbility, ComponentCard, ComponentContext, ComponentOptions, EndComponent,
    StartComponent, Workflow, WorkflowExecutionState, WorkflowRunOptions, END_NODE_STREAM,
    WORKFLOW_FINAL,
};

use common::{
    card, end_with_template, session, CollectorComponent, FailingComponent, StreamingProducer,
};

/// Upper bound on how long a deterministic in-memory stream may take to yield.
const STREAM_TIMEOUT: Duration = Duration::from_secs(5);

/// Publishes a custom frame, an output frame, then another custom frame from `invoke`.
#[derive(Debug, Clone, Default)]
struct CustomEmitter;

#[async_trait]
impl Component for CustomEmitter {
    fn card(&self) -> ComponentCard {
        ComponentCard::new("custom", "custom").with_abilities(self.abilities())
    }

    fn abilities(&self) -> Vec<ComponentAbility> {
        vec![ComponentAbility::Invoke]
    }

    fn component_type(&self) -> &str {
        "custom"
    }

    async fn invoke(&self, _inputs: Value, ctx: &ComponentContext) -> Result<Value> {
        ctx.write_custom_stream(json!({"progress": 1})).await?;
        ctx.write_stream(json!({"chunk": "a"})).await?;
        ctx.write_custom_stream(json!({"progress": 2})).await?;
        Ok(json!({"done": true}))
    }
}

/// `start → producer → end`, where `end` streams `{{first}}{{second}}`.
///
/// The end node is a `Transform` (`ComponentOptions::streaming()` plus a stream input
/// schema), so it consumes the producer's buffered array and renders one frame per
/// template segment.
fn streaming_workflow(producer: StreamingProducer) -> Workflow {
    let mut workflow = Workflow::new(card("streaming"));
    workflow
        .set_start_comp("start", StartComponent::new(), ComponentOptions::new())
        .add_workflow_comp("producer", producer, ComponentOptions::new())
        .set_end_comp(
            "end",
            end_with_template("{{first}}{{second}}"),
            ComponentOptions::new()
                .with_inputs_schema(json!({
                    "first": "${producer[0].delta}",
                    "second": "${producer[1].delta}",
                }))
                .with_stream_inputs_schema(json!({"whole": "${producer}"}))
                .streaming(),
        )
        .add_connection("start", "producer")
        .add_stream_connection("producer", "end");
    workflow
}

/// The producer frames the streaming tests feed into the end node.
fn producer_frames() -> Vec<Value> {
    vec![json!({"delta": "a"}), json!({"delta": "b"})]
}

/// Drains a frame stream, panicking on an error item.
async fn collect_frames(stream: BoxStream<'static, Result<StreamItem>>) -> Vec<StreamItem> {
    stream
        .map(|item| item.expect("the workflow stream yielded an error"))
        .collect()
        .await
}

/// `start → emitter → end`, where `emitter` writes custom and output frames.
fn custom_workflow() -> Workflow {
    let mut workflow = Workflow::new(card("custom"));
    workflow
        .set_start_comp("start", StartComponent::new(), ComponentOptions::new())
        .add_workflow_comp("emitter", CustomEmitter, ComponentOptions::new())
        .set_end_comp("end", EndComponent::new(), ComponentOptions::new())
        .add_connection("start", "emitter")
        .add_connection("emitter", "end");
    workflow
}

// port of tests/unit_tests/core/workflow/test_workflow_with_comp_stream.py::test_end_stream
#[tokio::test]
async fn a_streaming_end_yields_indexed_frames_then_the_final_frame() {
    let workflow = streaming_workflow(StreamingProducer::new(producer_frames()));
    let frames = collect_frames(workflow.stream(json!({}), session("stream-frames"))).await;

    assert_eq!(frames.len(), 3, "two end frames plus the final frame");
    for (index, frame) in frames[..2].iter().enumerate() {
        assert_eq!(frame.mode, StreamMode::Output, "frame {index}");
        assert_eq!(frame.item_type, END_NODE_STREAM, "frame {index}");
        assert_eq!(frame.index, Some(i64::try_from(index).unwrap()));
    }
    assert_eq!(frames[0].payload, json!({"response": "a"}));
    assert_eq!(frames[1].payload, json!({"response": "b"}));

    let final_frame = &frames[2];
    assert_eq!(final_frame.mode, StreamMode::Output);
    assert_eq!(final_frame.item_type, WORKFLOW_FINAL);
    assert_eq!(final_frame.index, Some(0));
    assert_eq!(
        final_frame.payload,
        json!([
            {"type": END_NODE_STREAM, "index": 0, "payload": {"response": "a"}},
            {"type": END_NODE_STREAM, "index": 1, "payload": {"response": "b"}},
        ])
    );
}

// port of tests/unit_tests/core/workflow/test_workflow_with_comp_stream.py::test_workflow_stream
#[tokio::test]
async fn invoke_returns_the_collected_end_wires() {
    let workflow = streaming_workflow(StreamingProducer::new(producer_frames()));
    let output = workflow
        .invoke(json!({}), session("stream-invoke"))
        .await
        .unwrap();

    assert_eq!(output.state, WorkflowExecutionState::Completed);
    assert_eq!(
        output.result,
        json!([
            {"type": END_NODE_STREAM, "index": 0, "payload": {"response": "a"}},
            {"type": END_NODE_STREAM, "index": 1, "payload": {"response": "b"}},
        ])
    );
}

// port of tests/unit_tests/core/workflow/test_workflow_with_comp_stream.py::test_custom_stream
#[tokio::test]
async fn custom_frames_are_forwarded_in_production_order() {
    let frames = collect_frames(custom_workflow().stream(json!({}), session("custom-order"))).await;

    assert_eq!(
        frames.len(),
        4,
        "three component frames plus the final frame"
    );
    assert_eq!(frames[0].mode, StreamMode::Custom);
    assert_eq!(frames[0].payload, json!({"progress": 1}));
    assert_eq!(frames[1].mode, StreamMode::Output);
    assert_eq!(frames[1].item_type, "output");
    assert_eq!(frames[1].index, Some(0));
    assert_eq!(frames[1].payload, json!({"chunk": "a"}));
    assert_eq!(frames[2].mode, StreamMode::Custom);
    assert_eq!(frames[2].payload, json!({"progress": 2}));
    assert_eq!(frames[3].item_type, WORKFLOW_FINAL);
}

// port of tests/unit_tests/core/workflow/test_workflow_with_comp_stream.py::test_stream_mode_filter
#[tokio::test]
async fn run_options_filter_the_forwarded_modes() {
    let workflow = custom_workflow();

    let custom: Vec<StreamItem> = workflow
        .stream_with(
            json!({}),
            session("custom-only"),
            WorkflowRunOptions::default().with_stream_modes(vec![StreamMode::Custom]),
        )
        .map(|item| item.expect("the filtered stream yielded an error"))
        .collect()
        .await;
    assert_eq!(custom.len(), 2, "{custom:?}");
    assert!(custom.iter().all(|frame| frame.mode == StreamMode::Custom));
    assert_eq!(custom[0].payload, json!({"progress": 1}));
    assert_eq!(custom[1].payload, json!({"progress": 2}));

    let output: Vec<StreamItem> = workflow
        .stream_with(
            json!({}),
            session("output-only"),
            WorkflowRunOptions::default().with_stream_modes(vec![StreamMode::Output]),
        )
        .map(|item| item.expect("the filtered stream yielded an error"))
        .collect()
        .await;
    assert_eq!(output.len(), 2, "{output:?}");
    assert!(output.iter().all(|frame| frame.mode == StreamMode::Output));
    assert_eq!(output[0].item_type, "output");
    assert_eq!(output[1].item_type, WORKFLOW_FINAL);
}

// port of tests/unit_tests/core/workflow/test_workflow_with_comp_stream.py::test_stream_failure
#[tokio::test]
async fn a_failing_component_yields_exactly_one_error() {
    let mut workflow = Workflow::new(card("stream-failure"));
    workflow
        .set_start_comp("start", StartComponent::new(), ComponentOptions::new())
        .add_workflow_comp(
            "boom",
            FailingComponent::new("kaboom"),
            ComponentOptions::new(),
        )
        .set_end_comp("end", EndComponent::new(), ComponentOptions::new())
        .add_connection("start", "boom")
        .add_connection("boom", "end");

    let items: Vec<_> = workflow
        .stream(json!({}), session("stream-failure"))
        .collect()
        .await;

    assert_eq!(items.len(), 1, "a failure is the stream's only item");
    let failure = items.into_iter().next().unwrap().unwrap_err();
    assert_eq!(failure.kind(), ErrorKind::Workflow);
    assert!(failure.message().contains("kaboom"), "{failure}");
}

// port of tests/unit_tests/core/workflow/test_workflow_with_comp_stream.py::test_buffered_stream_composition
#[tokio::test]
async fn a_collector_reads_the_producers_buffered_frames() {
    let mut workflow = Workflow::new(card("composition"));
    workflow
        .set_start_comp("start", StartComponent::new(), ComponentOptions::new())
        .add_workflow_comp(
            "producer",
            StreamingProducer::new(vec![json!({"output": "x"}), json!({"output": "y"})]),
            ComponentOptions::new(),
        )
        .add_workflow_comp(
            "collector",
            CollectorComponent::new(),
            ComponentOptions::new().with_stream_inputs_schema(json!({
                "frames": "${producer.output}",
            })),
        )
        .set_end_comp(
            "end",
            EndComponent::new(),
            ComponentOptions::new()
                .with_inputs_schema(json!({"frames": "${collector.collected.frames}"})),
        )
        .add_connection("start", "producer")
        .add_stream_connection("producer", "collector")
        .add_connection("collector", "end");

    let output = workflow
        .invoke(json!({}), session("composition"))
        .await
        .unwrap();

    assert_eq!(output.state, WorkflowExecutionState::Completed);
    // `${producer.output}` projects the `output` field over the buffered frames, so the
    // collector receives them as an array of raw payloads.
    assert_eq!(output.result, json!({"output": {"frames": ["x", "y"]}}));
}

// port of tests/unit_tests/core/workflow/test_workflow_with_comp_stream.py::test_stream_drop
#[tokio::test]
async fn dropping_the_stream_after_one_frame_does_not_hang() {
    let mut stream = streaming_workflow(StreamingProducer::new(producer_frames()))
        .stream(json!({}), session("stream-drop"));

    let first = tokio::time::timeout(STREAM_TIMEOUT, stream.next())
        .await
        .expect("the first frame must arrive");
    let first = first.expect("the stream must yield a frame").unwrap();
    assert_eq!(first.item_type, END_NODE_STREAM);

    // Dropping the stream aborts the in-flight run; the guard must not block.
    let dropped = tokio::time::timeout(STREAM_TIMEOUT, async move {
        drop(stream);
        tokio::task::yield_now().await;
    })
    .await;
    assert!(dropped.is_ok(), "dropping the stream must not block");
}

// port of tests/unit_tests/core/workflow/test_workflow_with_comp_stream.py::test_end_stream_empty
#[tokio::test]
async fn an_empty_producer_still_yields_the_final_frame() {
    // The engine stores an empty frame array under the end node and only omits the final
    // frame when that value is `null`, so an empty stream yields `workflow_final` with an
    // empty wire array rather than no frame at all.
    let workflow = streaming_workflow(StreamingProducer::new(Vec::new()));
    let frames = collect_frames(workflow.stream(json!({}), session("stream-empty"))).await;

    assert_eq!(frames.len(), 1, "{frames:?}");
    assert_eq!(frames[0].mode, StreamMode::Output);
    assert_eq!(frames[0].item_type, WORKFLOW_FINAL);
    assert_eq!(frames[0].index, Some(0));
    assert_eq!(frames[0].payload, json!([]));
}
