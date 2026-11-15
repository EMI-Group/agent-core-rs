//! Cross-module integration suite for the public staged-pipeline surface.
//!
//! Everything here is public API: [`AgentBuilder::build_with_pipeline`],
//! [`AgentBuilder::build_with_pipeline_stream`],
//! [`AgentBuilder::record_result`], [`AgentBuilder::status`],
//! [`AgentBuilder::history`], the [`BuildResult`] provenance/repair vocabulary
//! ([`GenerationSource`], [`RepairReport`], [`DEFAULT_MAX_RETRIES`]) and the
//! wire shape [`BuildResult::to_json`] produces, plus the [`PipelineEvent`]
//! stream.
//!
//! The suite pins five behaviours a caller can rely on:
//!
//! 1. a fully scripted build records one provenance entry per stage, with the
//!    intention detector deterministic and every model stage `model`;
//! 2. a model that never answers usable text still yields a completed,
//!    validating document whose provenance says `fallback`;
//! 3. the bounded repair loop converges inside [`DEFAULT_MAX_RETRIES`] and
//!    reports its attempts and notes;
//! 4. the stream and the buffered entry point produce the same result;
//! 5. [`AgentBuilder::record_result`] persists a streamed result into a
//!    session, and a blank query is rejected before any model call.
//!
//! Offline and deterministic: scripted models only, fixed timestamps passed by
//! the caller, no clock, environment, network, or sleep.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::Arc;

use futures::stream::BoxStream;
use futures::StreamExt;
use openjiuwen_common::error::{ErrorKind, Result};
use openjiuwen_dev_tools::agent_builder::{
    AgentBuilder, AgentType, BuildOutput, BuildResult, BuildState, GenerationSource, PipelineEvent,
    ProgressStage, RepairReport, WorkflowDsl, DEFAULT_MAX_RETRIES,
};
use openjiuwen_foundation::llm::Model;
use openjiuwen_test_support::{ScriptedModel, ScriptedTurn};
use serde_json::{json, Value};

/// The fixed timestamp every build here is stamped with.
const STAMP: i64 = 1_700_000_000_000;

/// The workflow request every build submits.
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

/// A structurally valid document whose `end` node has no inbound edge, so it is
/// unreachable and the deterministic reflector has to connect it.
const UNREACHABLE_END_FENCE: &str = r#"```json
{"name": "needs-repair", "description": "a draft that forgot its final edge",
 "nodes": [{"id": "start", "type": "start", "title": "Start"},
           {"id": "draft", "type": "llm", "title": "Draft"},
           {"id": "end", "type": "end", "title": "End"}],
 "edges": [{"source": "start", "target": "draft"}]}
```"#;

/// The revision the model answers the repair feedback with: a different, fully
/// connected document.
const REPAIRED_FENCE: &str = r#"```json
{"name": "repaired", "description": "the repaired workflow",
 "nodes": [{"id": "start", "type": "start", "title": "Start"},
           {"id": "draft", "type": "llm", "title": "Draft"},
           {"id": "review", "type": "llm", "title": "Review"},
           {"id": "end", "type": "end", "title": "End"}],
 "edges": [{"source": "start", "target": "draft"}, {"source": "draft", "target": "review"},
           {"source": "review", "target": "end"}]}
```"#;

/// A document naming a node that does not exist; `WorkflowDsl::validate`
/// rejects it, so it can never reach the reflector.
const DANGLING_EDGE_FENCE: &str = r#"```json
{"name": "dangling", "description": "an edge to nowhere",
 "nodes": [{"id": "start", "type": "start", "title": "Start"},
           {"id": "draft", "type": "llm", "title": "Draft"},
           {"id": "end", "type": "end", "title": "End"}],
 "edges": [{"source": "start", "target": "draft"}, {"source": "draft", "target": "ghost"}]}
```"#;

/// Coerces a scripted model into the trait object the builder takes.
fn dynamic(model: &Arc<ScriptedModel>) -> Arc<dyn Model> {
    let cloned: Arc<ScriptedModel> = Arc::clone(model);
    cloned
}

/// A model whose five answers drive a fully model-generated workflow build:
/// clarifier, three designer passes, DL generator.
fn model_driven_workflow() -> Arc<ScriptedModel> {
    Arc::new(
        ScriptedModel::builder()
            .respond_text(CLARIFY_READY)
            .respond_text(DESIGN_FENCE)
            .respond_text(DESIGN_FENCE)
            .respond_text(DESIGN_FENCE)
            .respond_text(GENERATED_FENCE)
            .build(),
    )
}

/// The workflow document of a result.
fn workflow_of(result: &BuildResult) -> &WorkflowDsl {
    match &result.output {
        BuildOutput::Workflow(workflow) => workflow,
        other => panic!("expected a workflow output, got {other:?}"),
    }
}

/// The `(stage, source)` pairs of a result's provenance, oldest first.
fn stage_sources(result: &BuildResult) -> Vec<(ProgressStage, GenerationSource)> {
    result
        .provenance
        .stages
        .iter()
        .map(|record| (record.stage, record.source))
        .collect()
}

/// The `(stage, source)` pairs of the `provenance` object of `to_json`.
fn json_stage_sources(value: &Value) -> Vec<(String, String)> {
    value["provenance"]["stages"]
        .as_array()
        .expect("provenance holds a stages array")
        .iter()
        .map(|stage| {
            (
                stage["stage"]
                    .as_str()
                    .expect("stage is a string")
                    .to_string(),
                stage["source"]
                    .as_str()
                    .expect("source is a string")
                    .to_string(),
            )
        })
        .collect()
}

/// Collects a whole pipeline stream.
async fn collect(stream: BoxStream<'static, Result<PipelineEvent>>) -> Vec<Result<PipelineEvent>> {
    stream.collect().await
}

/// The single final result of a collected stream, asserting it comes last.
fn final_of(events: &[Result<PipelineEvent>]) -> &BuildResult {
    let finals: Vec<&BuildResult> = events
        .iter()
        .filter_map(|event| match event {
            Ok(PipelineEvent::Final(result)) => Some(result),
            _ => None,
        })
        .collect();
    assert_eq!(finals.len(), 1, "exactly one final event: {events:?}");
    assert!(
        matches!(events.last(), Some(Ok(PipelineEvent::Final(_)))),
        "the final event is last"
    );
    finals[0]
}

/// Every repair report the stream reported, oldest first.
fn repaired_of(events: &[Result<PipelineEvent>]) -> Vec<RepairReport> {
    events
        .iter()
        .filter_map(|event| match event {
            Ok(PipelineEvent::Repaired(report)) => Some(report.clone()),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn model_driven_pipeline_records_provenance_and_the_wire_shape() {
    let model = model_driven_workflow();
    let mut builder = AgentBuilder::new(dynamic(&model));

    let result = builder
        .build_with_pipeline(QUERY, "w1", AgentType::Workflow, STAMP)
        .await
        .expect("the fully scripted pipeline succeeds");

    assert_eq!(result.status, "completed");
    assert_eq!(result.session_id, "w1");
    assert_eq!(result.agent_type, AgentType::Workflow);
    let workflow = workflow_of(&result);
    workflow
        .validate()
        .expect("the produced workflow validates");
    assert_eq!(workflow.name, "generated");
    // Five model round trips: clarifier, three designer passes, generator.
    assert_eq!(model.call_count(), 5);

    // One record per stage, oldest first: the first request is detected
    // deterministically, everything after it came from an accepted answer.
    assert_eq!(
        stage_sources(&result),
        vec![
            (
                ProgressStage::DetectingIntention,
                GenerationSource::Deterministic
            ),
            (ProgressStage::Clarifying, GenerationSource::Model),
            (
                ProgressStage::GeneratingWorkflowDesign,
                GenerationSource::Model
            ),
            (
                ProgressStage::GeneratingWorkflowDesign,
                GenerationSource::Model
            ),
            (ProgressStage::ValidatingDl, GenerationSource::Model),
            (ProgressStage::GeneratingDl, GenerationSource::Model),
        ]
    );
    assert!(result.provenance.has_deviation(), "the detector deviated");
    assert!(!result.provenance.has_fallback(), "no model answer failed");
    assert_eq!(
        result
            .provenance
            .stages_matching(GenerationSource::Model)
            .len(),
        5
    );

    // The wire shape: the four Python keys plus `provenance` and `repair`,
    // every stage spelled with the snake_case stage/source/detail keys.
    let value = result.to_json().expect("the result serializes");
    for key in [
        "status",
        "session_id",
        "agent_type",
        "workflow",
        "provenance",
        "repair",
    ] {
        assert!(value.get(key).is_some(), "missing key `{key}`: {value}");
    }
    assert_eq!(value["status"], json!("completed"));
    assert_eq!(value["agent_type"], json!("workflow"));
    let records = value["provenance"]["stages"]
        .as_array()
        .expect("stages is an array");
    assert_eq!(records.len(), 6);
    for record in records {
        let keys: Vec<&String> = record
            .as_object()
            .expect("a stage record is an object")
            .keys()
            .collect();
        assert_eq!(keys, vec!["stage", "source", "detail"]);
        assert!(record["detail"].is_string());
    }
    assert_eq!(
        json_stage_sources(&value),
        vec![
            (
                "detecting_intention".to_string(),
                "deterministic".to_string()
            ),
            ("clarifying".to_string(), "model".to_string()),
            (
                "generating_workflow_design".to_string(),
                "model".to_string()
            ),
            (
                "generating_workflow_design".to_string(),
                "model".to_string()
            ),
            ("validating_dl".to_string(), "model".to_string()),
            ("generating_dl".to_string(), "model".to_string()),
        ]
    );
    // A single passing attempt is still reported.
    assert_eq!(value["repair"], json!({"attempts": 1, "notes": []}));

    // The document on the wire is the same valid document.
    let embedded = WorkflowDsl::from_json_str(&value["workflow"].to_string())
        .expect("the embedded document parses");
    assert_eq!(&embedded, workflow);
    embedded
        .validate()
        .expect("the embedded document validates");
}

#[tokio::test]
async fn unusable_model_answers_fall_back_to_a_validating_document() {
    // No scripted answers: every call receives the model's fallback text, which
    // no stage can parse, so each stage runs its documented offline path.
    let model = Arc::new(ScriptedModel::new([]));
    let mut builder = AgentBuilder::new(dynamic(&model));

    let result = builder
        .build_with_pipeline(QUERY, "w1", AgentType::Workflow, STAMP)
        .await
        .expect("the deterministic stages still succeed");

    assert_eq!(result.status, "completed");
    let workflow = workflow_of(&result);
    workflow
        .validate()
        .expect("the fallback workflow validates");
    assert_eq!(
        workflow
            .nodes
            .iter()
            .map(|node| node.id.as_str())
            .collect::<Vec<_>>(),
        ["start", "llm_1", "end"]
    );

    // The detector never calls a model, so it stays `deterministic`; every
    // stage that did call one and got an unusable answer is a `fallback` with a
    // non-empty reason.
    let sources = stage_sources(&result);
    assert_eq!(
        sources.first(),
        Some(&(
            ProgressStage::DetectingIntention,
            GenerationSource::Deterministic
        ))
    );
    assert!(result.provenance.has_fallback(), "{sources:?}");
    assert!(result.provenance.fallback_count() >= 4, "{sources:?}");
    let fallbacks = result
        .provenance
        .stages_matching(GenerationSource::Fallback);
    assert!(fallbacks
        .iter()
        .all(|record| !record.detail.trim().is_empty()));
    assert!(sources
        .iter()
        .all(|(_, source)| *source == GenerationSource::Deterministic
            || *source == GenerationSource::Fallback));

    // The fallback provenance reaches the wire, and the DSL still validates.
    let value = result.to_json().expect("the result serializes");
    assert_eq!(value["status"], json!("completed"));
    let wire = json_stage_sources(&value);
    assert!(
        wire.contains(&("clarifying".to_string(), "fallback".to_string())),
        "{wire:?}"
    );
    assert!(
        wire.contains(&("generating_dl".to_string(), "fallback".to_string())),
        "{wire:?}"
    );
    let embedded = WorkflowDsl::from_json_str(&value["workflow"].to_string())
        .expect("the embedded document parses");
    embedded
        .validate()
        .expect("the embedded document validates");
    assert_eq!(
        builder.status("w1").expect("the session exists").state,
        BuildState::Completed
    );
    assert!(model.call_count() > 0, "the model was consulted");
}

#[tokio::test]
async fn repair_loop_converges_within_the_retry_budget() {
    // The generated document is structurally valid but leaves `end`
    // unreachable, so the deterministic reflection repairs it, the bounded loop
    // asks for a revision, and the revision passes.
    let model = Arc::new(ScriptedModel::new([
        ScriptedTurn::text(CLARIFY_READY),
        ScriptedTurn::text(DESIGN_FENCE),
        ScriptedTurn::text(DESIGN_FENCE),
        ScriptedTurn::text(DESIGN_FENCE),
        ScriptedTurn::text(UNREACHABLE_END_FENCE),
        ScriptedTurn::text(REPAIRED_FENCE),
    ]));
    let mut builder = AgentBuilder::new(dynamic(&model));

    let result = builder
        .build_with_pipeline(QUERY, "w1", AgentType::Workflow, STAMP)
        .await
        .expect("the repair loop converges");

    assert_eq!(result.status, "completed");
    let workflow = workflow_of(&result);
    workflow
        .validate()
        .expect("the repaired workflow validates");
    assert_eq!(workflow.name, "repaired");
    // The unreachable `end` node was connected before the revision arrived.
    assert!(workflow
        .edges
        .iter()
        .any(|edge| edge.source == "review" && edge.target == "end"));

    // Attempt one repaired the unreachable node, attempt two passed.
    assert_eq!(result.repair.attempts, 2);
    assert!(result.repair.attempts <= DEFAULT_MAX_RETRIES);
    assert_eq!(result.repair.notes.len(), 1);
    assert!(
        result.repair.notes[0].contains("unreachable_node"),
        "{:?}",
        result.repair.notes
    );
    assert!(result.repair.notes[0].contains("end"));
    assert!(!result.repair.is_empty());

    // The repair episode and the accepted revision are both on the wire.
    let value = result.to_json().expect("the result serializes");
    assert_eq!(value["repair"]["attempts"], json!(2));
    assert_eq!(value["repair"]["notes"], json!([result.repair.notes[0]]));
    let wire = json_stage_sources(&value);
    assert!(
        wire.contains(&("refining_dl".to_string(), "model".to_string())),
        "{wire:?}"
    );
    assert_eq!(model.call_count(), 6, "five stages plus one revision");

    // A document with a dangling edge never reaches the reflector: structure
    // validation rejects it during generation, so the baseline document is
    // used instead of a repair.
    let dangling = Arc::new(ScriptedModel::new([
        ScriptedTurn::text(CLARIFY_READY),
        ScriptedTurn::text(DESIGN_FENCE),
        ScriptedTurn::text(DESIGN_FENCE),
        ScriptedTurn::text(DESIGN_FENCE),
        ScriptedTurn::text(DANGLING_EDGE_FENCE),
    ]));
    let mut fallback_builder = AgentBuilder::new(dynamic(&dangling));
    let fallback = fallback_builder
        .build_with_pipeline(QUERY, "w2", AgentType::Workflow, STAMP)
        .await
        .expect("the dangling answer is absorbed");

    assert_eq!(fallback.status, "completed");
    workflow_of(&fallback)
        .validate()
        .expect("the baseline workflow validates");
    let generation = fallback
        .provenance
        .stages
        .iter()
        .find(|record| record.stage == ProgressStage::GeneratingDl)
        .expect("the generation stage is recorded");
    assert_eq!(generation.source, GenerationSource::Fallback);
    assert_ne!(generation.detail.trim(), "");
    assert_eq!(fallback.repair.attempts, 1, "the baseline needs no repair");
    assert!(fallback.repair.attempts <= DEFAULT_MAX_RETRIES);
}

#[tokio::test]
async fn stream_matches_the_buffered_build_event_for_event() {
    let streamed = model_driven_workflow();
    let buffered = model_driven_workflow();
    let builder = AgentBuilder::new(dynamic(&streamed));
    let mut buffered_builder = AgentBuilder::new(dynamic(&buffered));

    let events =
        collect(builder.build_with_pipeline_stream(QUERY, "w1", AgentType::Workflow, STAMP)).await;
    assert!(
        events.iter().all(Result::is_ok),
        "no stage errors: {events:?}"
    );

    // Stage starts are announced in pipeline order and every one completes.
    let started: Vec<ProgressStage> = events
        .iter()
        .filter_map(|event| match event {
            Ok(PipelineEvent::StageStarted { stage }) => Some(*stage),
            _ => None,
        })
        .collect();
    let completed: Vec<ProgressStage> = events
        .iter()
        .filter_map(|event| match event {
            Ok(PipelineEvent::StageCompleted { stage, .. }) => Some(*stage),
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
    assert_eq!(completed, started, "every started stage completes");

    // The clarification round and the repair attempt are reported once each,
    // and the repair report matches the result.
    let clarified = events
        .iter()
        .filter(|event| matches!(event, Ok(PipelineEvent::Clarified(_))))
        .count();
    assert_eq!(clarified, 1, "one clarification round settles");
    let repairs = repaired_of(&events);
    assert_eq!(repairs.len(), 1, "one repair attempt");

    // The streamed final result is the buffered result for the same script.
    let streamed_final = final_of(&events);
    assert_eq!(repairs[0], streamed_final.repair);
    assert_eq!(streamed_final.status, "completed");
    assert_eq!(streamed_final.session_id, "w1");
    assert_eq!(streamed_final.agent_type, AgentType::Workflow);
    assert_eq!(
        stage_sources(streamed_final),
        vec![
            (
                ProgressStage::DetectingIntention,
                GenerationSource::Deterministic
            ),
            (ProgressStage::Clarifying, GenerationSource::Model),
            (
                ProgressStage::GeneratingWorkflowDesign,
                GenerationSource::Model
            ),
            (
                ProgressStage::GeneratingWorkflowDesign,
                GenerationSource::Model
            ),
            (ProgressStage::ValidatingDl, GenerationSource::Model),
            (ProgressStage::GeneratingDl, GenerationSource::Model),
        ]
    );

    let buffered_final = buffered_builder
        .build_with_pipeline(QUERY, "w1", AgentType::Workflow, STAMP)
        .await
        .expect("the buffered build succeeds");
    assert_eq!(streamed_final.status, buffered_final.status);
    assert_eq!(streamed_final.output, buffered_final.output);
    assert_eq!(
        stage_sources(streamed_final),
        stage_sources(&buffered_final)
    );
    assert_eq!(streamed_final.repair, buffered_final.repair);
    assert_eq!(*streamed_final, buffered_final);
    assert_eq!(streamed.call_count(), buffered.call_count());
}

#[tokio::test]
async fn record_result_persists_a_streamed_build_into_its_session() {
    let model = model_driven_workflow();
    let mut builder = AgentBuilder::new(dynamic(&model));

    let events =
        collect(builder.build_with_pipeline_stream(QUERY, "w1", AgentType::Workflow, STAMP)).await;
    let result = final_of(&events).clone();
    // The stream mutates nothing: a caller decides when to persist.
    assert!(builder.status("w1").is_none());
    assert_eq!(builder.history("w1", None).len(), 0);

    builder
        .record_result("w1", AgentType::Workflow, &result)
        .expect("the streamed result persists");

    let status = builder.status("w1").expect("the session now exists");
    assert_eq!(status.session_id, "w1");
    assert_eq!(status.agent_type, AgentType::Workflow);
    assert_eq!(status.state, BuildState::Completed);
    assert_eq!(status.resource_count, 1, "the assistant turn was recorded");

    let history = builder.history("w1", None);
    assert_eq!(history.len(), 1);
    assert_eq!(history[0].role, "assistant");
    let persisted =
        WorkflowDsl::from_json_str(&history[0].content).expect("the document round-trips");
    assert_eq!(&persisted, workflow_of(&result));
    assert_eq!(builder.history("w1", Some(1)), history);
    assert_eq!(builder.history("w1", Some(0)).len(), 0);
    assert_eq!(builder.history("unknown", None).len(), 0);

    // A blank query is rejected before any session or model call.
    let blank_model = Arc::new(ScriptedModel::new([]));
    let mut blank_builder = AgentBuilder::new(dynamic(&blank_model));
    let error = blank_builder
        .build_with_pipeline("   ", "w2", AgentType::Workflow, STAMP)
        .await
        .expect_err("a blank query is invalid");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("query must not be blank"));
    assert_eq!(blank_model.call_count(), 0, "the model was never called");
    assert!(blank_builder.status("w2").is_none(), "no session was made");
}
