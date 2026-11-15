//! Tests for the streaming form of the staged pipeline.
//!
//! Every case runs offline on scripted models and asserts the events, the
//! stage order, and the fact that the stream never touches builder sessions.

use std::sync::Arc;

use futures::StreamExt;
use openjiuwen_common::error::ErrorKind;
use openjiuwen_test_support::{ScriptedModel, ScriptedTurn};

use crate::agent_builder::outcome::RepairReport;

use super::*;

/// The request every streaming test submits.
const QUERY: &str = "Create a workflow that summarizes the document and returns a summary.";

/// A clarification answer declaring the request complete.
const CLARIFY_READY: &str = r#"```json
{"ready": true, "question": "", "summary": "goal, output"}
```"#;

/// A model answer holding a complete workflow design.
const DESIGN_FENCE: &str = r#"```json
{"name": "demo", "description": "demo design",
 "nodes": [{"id": "start", "type": "start", "title": "Start"},
           {"id": "llm_1", "type": "llm", "title": "Draft", "description": "draft it"},
           {"id": "end", "type": "end", "title": "End"}],
 "edges": [{"source": "start", "target": "llm_1"}, {"source": "llm_1", "target": "end"}]}
```"#;

/// A model answer holding a valid workflow document.
const GENERATED_FENCE: &str = r#"```json
{"name": "generated", "description": "generated workflow",
 "nodes": [{"id": "start", "type": "start", "title": "Start"},
           {"id": "llm_1", "type": "llm", "title": "Draft"},
           {"id": "end", "type": "end", "title": "End"}],
 "edges": [{"source": "start", "target": "llm_1"}, {"source": "llm_1", "target": "end"}]}
```"#;

/// A structurally valid workflow document with a self-loop.
const CYCLIC_ONE: &str = r#"```json
{"name": "looped", "description": "a loop",
 "nodes": [{"id": "start", "type": "start", "title": "Start"},
           {"id": "a", "type": "llm", "title": "A"},
           {"id": "end", "type": "end", "title": "End"}],
 "edges": [{"source": "start", "target": "a"}, {"source": "a", "target": "a"}]}
```"#;

/// A structurally valid workflow document with a three-node cycle.
const CYCLIC_TWO: &str = r#"```json
{"name": "looped two", "description": "another loop",
 "nodes": [{"id": "start", "type": "start", "title": "Start"},
           {"id": "a", "type": "llm", "title": "A"},
           {"id": "b", "type": "llm", "title": "B"},
           {"id": "end", "type": "end", "title": "End"}],
 "edges": [{"source": "start", "target": "a"}, {"source": "a", "target": "b"},
           {"source": "b", "target": "a"}]}
```"#;

/// A third cyclic document, distinct from the repaired form of [`CYCLIC_TWO`].
const CYCLIC_THREE: &str = r#"```json
{"name": "looped three", "description": "yet another loop",
 "nodes": [{"id": "start", "type": "start", "title": "Start"},
           {"id": "a", "type": "llm", "title": "A"},
           {"id": "b", "type": "llm", "title": "B"},
           {"id": "c", "type": "llm", "title": "C"},
           {"id": "end", "type": "end", "title": "End"}],
 "edges": [{"source": "start", "target": "a"}, {"source": "a", "target": "b"},
           {"source": "b", "target": "c"}, {"source": "c", "target": "a"}]}
```"#;

/// The five scripted answers of a fully model-driven workflow build.
fn workflow_turns() -> Vec<ScriptedTurn> {
    vec![
        ScriptedTurn::text(CLARIFY_READY),
        ScriptedTurn::text(DESIGN_FENCE),
        ScriptedTurn::text(DESIGN_FENCE),
        ScriptedTurn::text(DESIGN_FENCE),
        ScriptedTurn::text(GENERATED_FENCE),
    ]
}

/// Coerces a scripted model into the trait object the builder takes.
fn dynamic(model: &Arc<ScriptedModel>) -> Arc<dyn Model> {
    let cloned: Arc<ScriptedModel> = Arc::clone(model);
    cloned
}

/// Collects every event of one streamed build.
async fn collect(stream: BoxStream<'static, Result<PipelineEvent>>) -> Vec<Result<PipelineEvent>> {
    stream.collect().await
}

/// Returns the single final result of a collected run.
fn final_of(events: &[Result<PipelineEvent>]) -> &BuildResult {
    let finals: Vec<&BuildResult> = events
        .iter()
        .filter_map(|event| match event {
            Ok(PipelineEvent::Final(result)) => Some(result),
            _ => None,
        })
        .collect();
    assert_eq!(finals.len(), 1, "exactly one final event");
    assert!(
        matches!(events.last(), Some(Ok(PipelineEvent::Final(_)))),
        "the final event is last"
    );
    finals[0]
}

#[tokio::test]
async fn stream_reports_the_stage_order_and_matches_the_buffered_build() {
    let streamed = Arc::new(ScriptedModel::new(workflow_turns()));
    let buffered = Arc::new(ScriptedModel::new(workflow_turns()));
    let streaming_builder = AgentBuilder::new(dynamic(&streamed));
    let mut buffered_builder = AgentBuilder::new(dynamic(&buffered));

    let events = collect(streaming_builder.build_with_pipeline_stream(
        QUERY,
        "w1",
        AgentType::Workflow,
        1_700_000_000_000,
    ))
    .await;
    assert!(
        events.iter().all(Result::is_ok),
        "no stage errors: {events:?}"
    );

    let started: Vec<ProgressStage> = events
        .iter()
        .filter_map(|event| match event {
            Ok(PipelineEvent::StageStarted { stage }) => Some(*stage),
            _ => None,
        })
        .collect();
    assert_eq!(
        started,
        vec![
            ProgressStage::DetectingIntention,
            ProgressStage::Clarifying,
            ProgressStage::GeneratingWorkflowDesign,
            ProgressStage::GeneratingWorkflowDesign,
            ProgressStage::ValidatingDl,
            ProgressStage::GeneratingDl,
            ProgressStage::ValidatingDl,
        ]
    );
    let completed = events
        .iter()
        .filter(|event| matches!(event, Ok(PipelineEvent::StageCompleted { .. })))
        .count();
    assert_eq!(
        completed,
        started.len(),
        "every stage started and completed"
    );
    let clarified = events
        .iter()
        .filter(|event| matches!(event, Ok(PipelineEvent::Clarified(_))))
        .count();
    assert_eq!(clarified, 1, "one clarification round settles");

    let streamed_final = final_of(&events).clone();

    let buffered_final = buffered_builder
        .build_with_pipeline(QUERY, "w1", AgentType::Workflow, 1_700_000_000_000)
        .await
        .expect("the buffered build succeeds");
    assert_eq!(streamed_final, buffered_final);
    assert_eq!(streamed.call_count(), buffered.call_count());
}

#[tokio::test]
async fn stream_reports_one_repaired_event_per_repair_attempt() {
    let model = Arc::new(ScriptedModel::new([
        ScriptedTurn::text(CLARIFY_READY),
        ScriptedTurn::text(DESIGN_FENCE),
        ScriptedTurn::text(DESIGN_FENCE),
        ScriptedTurn::text(DESIGN_FENCE),
        ScriptedTurn::text(CYCLIC_ONE),
        ScriptedTurn::text(CYCLIC_TWO),
        ScriptedTurn::text(CYCLIC_THREE),
    ]));
    let builder = AgentBuilder::new(dynamic(&model));

    let events = collect(builder.build_with_pipeline_stream(
        QUERY,
        "w1",
        AgentType::Workflow,
        1_700_000_000_000,
    ))
    .await;

    let reports: Vec<&RepairReport> = events
        .iter()
        .filter_map(|event| match event {
            Ok(PipelineEvent::Repaired(report)) => Some(report),
            _ => None,
        })
        .collect();
    assert_eq!(reports.len(), 3, "one repaired event per attempt");
    assert_eq!(reports[0].attempts, 1);
    assert_eq!(reports[2].attempts, 3);
    assert!(reports[2].notes[0].contains("cycle"));

    let final_result = final_of(&events);
    assert_eq!(final_result.repair.attempts, 3);
    assert_eq!(final_result.repair.notes.len(), 2);
}

#[tokio::test]
async fn blank_query_stream_yields_a_single_validation_error() {
    let model = Arc::new(ScriptedModel::new([]));
    let builder = AgentBuilder::new(dynamic(&model));

    let events =
        collect(builder.build_with_pipeline_stream("   ", "w1", AgentType::Workflow, 1)).await;

    assert_eq!(events.len(), 1, "one error and no final event: {events:?}");
    match &events[0] {
        Err(error) => assert_eq!(error.kind(), ErrorKind::Validation),
        other => panic!("expected a validation error, got {other:?}"),
    }
    assert_eq!(model.call_count(), 0);
    assert!(builder.status("w1").is_none(), "no session was created");
}

#[tokio::test]
async fn stream_is_send_and_static_and_leaves_sessions_untouched() {
    let model = Arc::new(ScriptedModel::new(workflow_turns()));
    let mut builder = AgentBuilder::new(dynamic(&model));

    let stream =
        builder.build_with_pipeline_stream(QUERY, "w1", AgentType::Workflow, 1_700_000_000_000);
    // The stream owns its data, so the builder stays usable and untouched.
    assert!(builder.status("w1").is_none());
    let handle = tokio::spawn(async move { collect(stream).await });
    let events = handle.await.expect("the streaming task finishes");
    assert!(
        builder.status("w1").is_none(),
        "the stream persists nothing"
    );

    let result = final_of(&events).clone();
    builder
        .record_result("w1", AgentType::Workflow, &result)
        .expect("the result persists");

    let status = builder.status("w1").expect("the session now exists");
    assert_eq!(status.state, BuildState::Completed);
    assert_eq!(status.resource_count, 1, "one assistant turn recorded");
    let history = builder.history("w1", None);
    assert_eq!(history.len(), 1, "only the assistant turn was recorded");
    let workflow =
        WorkflowDsl::from_json_str(&history[0].content).expect("the document round-trips");
    assert_eq!(workflow.name, "generated");
}
