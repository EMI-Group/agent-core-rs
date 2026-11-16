//! Tests for `AgentBuilder::build_with_pipeline`: the staged clarifier,
//! designer, DL generator, and reflector path and its deterministic fallbacks.
//!
//! Every case runs offline on scripted models: no network, no environment reads,
//! and no wall-clock dependence.
use super::*;
use openjiuwen_common::error::ErrorKind;
use openjiuwen_test_support::{ScriptedModel, ScriptedTurn};

/// The request every pipeline test in this module submits.
const QUERY: &str = "Create a workflow that summarizes the document and returns a summary.";

/// A clarification answer declaring the request complete.
const CLARIFY_READY: &str = r#"```json
{"ready": true, "question": "", "summary": "goal, output"}
```"#;

/// A clarification answer asking for the expected output.
const CLARIFY_NEEDED: &str = r#"```json
{"ready": false, "question": "What should the agent produce?", "summary": "goal"}
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

/// Coerces a scripted model into the trait object the builder takes.
fn dynamic(model: &Arc<ScriptedModel>) -> Arc<dyn Model> {
    let cloned: Arc<ScriptedModel> = Arc::clone(model);
    cloned
}

/// The workflow document of a successful build result.
fn workflow_of(result: &BuildResult) -> &WorkflowDsl {
    match &result.output {
        BuildOutput::Workflow(workflow) => workflow,
        other => panic!("expected workflow, got {other:?}"),
    }
}

/// The DSL document of a successful build result.
fn dsl_of(result: &BuildResult) -> &LlmAgentDsl {
    match &result.output {
        BuildOutput::Dsl(dsl) => dsl,
        other => panic!("expected DSL, got {other:?}"),
    }
}

#[tokio::test]
async fn workflow_pipeline_runs_clarifier_designer_and_generator() {
    let model = Arc::new(ScriptedModel::new([
        ScriptedTurn::text(CLARIFY_READY),
        ScriptedTurn::text(DESIGN_FENCE),
        ScriptedTurn::text(DESIGN_FENCE),
        ScriptedTurn::text(DESIGN_FENCE),
        ScriptedTurn::text(GENERATED_FENCE),
    ]));
    let mut builder = AgentBuilder::new(dynamic(&model));

    let result = builder
        .build_with_pipeline(QUERY, "w1", AgentType::Workflow, 1_700_000_000_000)
        .await
        .expect("the pipeline succeeds");

    assert_eq!(result.status, "completed");
    assert_eq!(result.agent_type, AgentType::Workflow);
    let workflow = workflow_of(&result);
    workflow.validate().expect("the workflow is valid");
    assert_eq!(workflow.name, "generated");
    assert_eq!(
        workflow
            .nodes
            .iter()
            .map(|node| node.id.as_str())
            .collect::<Vec<_>>(),
        ["start", "llm_1", "end"]
    );

    // Intention detection runs deterministically on a first request, so the
    // five model calls are: clarifier, three designer passes, DL generator.
    assert_eq!(model.call_count(), 5);
    let calls = model.calls();
    assert_eq!(calls[0].last_user_text().as_deref(), Some(QUERY));
    assert!(calls[1]
        .last_user_text()
        .is_some_and(|text| text.contains("summary")));
    assert!(calls[4]
        .last_user_text()
        .is_some_and(|text| text.contains("demo design")));

    let status = builder.status("w1").expect("the session exists");
    assert_eq!(status.state, BuildState::Completed);
    assert_eq!(status.resource_count, 2);

    let history = builder.history("w1", None);
    assert_eq!(history.len(), 2);
    assert_eq!(history[1].role, "assistant");
    let persisted =
        WorkflowDsl::from_json_str(&history[1].content).expect("the document round-trips");
    assert_eq!(&persisted, workflow);
}

#[tokio::test]
async fn workflow_pipeline_falls_back_to_the_deterministic_stages() {
    let model = Arc::new(ScriptedModel::new([]));
    let mut builder = AgentBuilder::new(dynamic(&model));

    let result = builder
        .build_with_pipeline(QUERY, "w1", AgentType::Workflow, 1_700_000_000_000)
        .await
        .expect("the deterministic stages succeed");

    assert_eq!(result.status, "completed");
    let workflow = workflow_of(&result);
    workflow.validate().expect("the workflow is valid");
    assert_eq!(
        workflow.nodes.first().map(|node| node.id.as_str()),
        Some("start")
    );
    assert_eq!(
        workflow.nodes.last().map(|node| node.id.as_str()),
        Some("end")
    );
    assert!(workflow.nodes.iter().any(|node| node.id == "llm_1"));
    assert_eq!(
        builder.status("w1").expect("the session exists").state,
        BuildState::Completed
    );
}

#[tokio::test]
async fn incomplete_request_stops_with_the_clarification_question() {
    let model = Arc::new(ScriptedModel::text(CLARIFY_NEEDED));
    let mut builder = AgentBuilder::new(dynamic(&model));

    let result = builder
        .build_with_pipeline("do something", "s1", AgentType::LlmAgent, 1)
        .await
        .expect("the pipeline succeeds");

    assert_eq!(result.status, "clarifying");
    match &result.output {
        BuildOutput::Response(text) => assert_eq!(text, "What should the agent produce?"),
        other => panic!("expected a response, got {other:?}"),
    }
    assert_eq!(
        builder.status("s1").expect("the session exists").state,
        BuildState::Clarifying
    );
    assert_eq!(model.call_count(), 1);

    let history = builder.history("s1", None);
    assert_eq!(history.len(), 2);
    assert_eq!(history[0].role, "user");
    assert_eq!(history[0].content, "do something");
    assert_eq!(history[1].role, "assistant");
    assert_eq!(history[1].content, "What should the agent produce?");
}

#[tokio::test]
async fn llm_agent_pipeline_repairs_an_unparseable_answer() {
    let model = Arc::new(ScriptedModel::new([
        ScriptedTurn::text(CLARIFY_READY),
        ScriptedTurn::text("I cannot help with that."),
    ]));
    let mut builder = AgentBuilder::new(dynamic(&model));

    let result = builder
        .build_with_pipeline(QUERY, "s1", AgentType::LlmAgent, 1_700_000_000_000)
        .await
        .expect("the pipeline succeeds");

    assert_eq!(result.status, "completed");
    let dsl = dsl_of(&result);
    assert_ne!(dsl.agent_id.trim(), "");
    assert_ne!(dsl.name.trim(), "");
    assert_ne!(dsl.configs.system_prompt.trim(), "");
    assert_eq!(dsl.create_time, 1_700_000_000_000);
    assert_eq!(dsl.update_time, 1_700_000_000_000);

    let history = builder.history("s1", None);
    assert_eq!(history.len(), 2);
    assert_eq!(history[1].role, "assistant");
    let persisted =
        LlmAgentDsl::from_json_str(&history[1].content).expect("the document round-trips");
    assert_eq!(&persisted, dsl);
    assert_eq!(model.call_count(), 2);
}

#[tokio::test]
async fn llm_agent_pipeline_absorbs_a_model_failure() {
    let model = Arc::new(ScriptedModel::new([
        ScriptedTurn::text(CLARIFY_READY),
        ScriptedTurn::fail("boom"),
    ]));
    let mut builder = AgentBuilder::new(dynamic(&model));

    let result = builder
        .build_with_pipeline(QUERY, "s1", AgentType::LlmAgent, 1_700_000_000_000)
        .await
        .expect("the model failure is absorbed");

    assert_eq!(result.status, "completed");
    let dsl = dsl_of(&result);
    assert_ne!(dsl.agent_id.trim(), "");
    assert_eq!(dsl.create_time, 1_700_000_000_000);
    assert_eq!(model.call_count(), 2);
    assert_eq!(
        builder.status("s1").expect("the session exists").state,
        BuildState::Completed
    );
}

#[tokio::test]
async fn blank_query_is_rejected_before_any_session_or_model_call() {
    let model = Arc::new(ScriptedModel::new([]));
    let mut builder = AgentBuilder::new(dynamic(&model));

    for query in ["", "   ", "\n\t"] {
        let error = builder
            .build_with_pipeline(query, "s1", AgentType::LlmAgent, 1)
            .await
            .expect_err("a blank query is invalid");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("query must not be blank"));
    }

    assert_eq!(model.call_count(), 0);
    assert!(builder.status("s1").is_none());
}

#[tokio::test]
async fn deterministic_fallback_is_reproducible() {
    let first_model = Arc::new(ScriptedModel::new([]));
    let mut first = AgentBuilder::new(dynamic(&first_model));
    let second_model = Arc::new(ScriptedModel::new([]));
    let mut second = AgentBuilder::new(dynamic(&second_model));

    let first_result = first
        .build_with_pipeline(QUERY, "s1", AgentType::LlmAgent, 1_700_000_000_000)
        .await
        .expect("first pipeline succeeds");
    let second_result = second
        .build_with_pipeline(QUERY, "s2", AgentType::LlmAgent, 1_700_000_000_000)
        .await
        .expect("second pipeline succeeds");

    let first_dsl = dsl_of(&first_result);
    let second_dsl = dsl_of(&second_result);
    assert_eq!(first_dsl.name, second_dsl.name);
    assert_eq!(first_dsl.description, second_dsl.description);
    assert_eq!(
        first_dsl.configs.system_prompt,
        second_dsl.configs.system_prompt
    );
    assert_eq!(first_dsl.create_time, second_dsl.create_time);
    assert_eq!(first_dsl.update_time, second_dsl.update_time);
}

#[tokio::test]
async fn legacy_build_agent_at_keeps_its_single_round_trip() {
    let model = Arc::new(ScriptedModel::text("I cannot help with that."));
    let mut builder = AgentBuilder::new(dynamic(&model));

    let result = builder
        .build_agent_at("build me an agent", "s1", AgentType::LlmAgent, 1)
        .await
        .expect("build succeeds");

    assert_eq!(result.status, "clarifying");
    match &result.output {
        BuildOutput::Response(text) => assert_eq!(text, "I cannot help with that."),
        other => panic!("expected a response, got {other:?}"),
    }
    assert_eq!(model.call_count(), 1);
    assert_eq!(
        builder.status("s1").expect("the session exists").state,
        BuildState::Clarifying
    );
}

/// A refinement-intention answer declaring that the diagram must change.
const NEED_REFINED: &str = r#"```json
{"need_refined": true}
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

/// An acyclic workflow document that passes the deterministic reflection.
const FIXED_FENCE: &str = r#"```json
{"name": "fixed", "description": "a fixed workflow",
 "nodes": [{"id": "start", "type": "start", "title": "Start"},
           {"id": "a", "type": "llm", "title": "A"},
           {"id": "end", "type": "end", "title": "End"}],
 "edges": [{"source": "start", "target": "a"}, {"source": "a", "target": "end"}]}
```"#;

/// The document a refinement request answers with on the second turn.
const REFINED_FENCE: &str = r#"```json
{"name": "refined", "description": "the refined workflow",
 "nodes": [{"id": "start", "type": "start", "title": "Start"},
           {"id": "validate", "type": "llm", "title": "Validate"},
           {"id": "end", "type": "end", "title": "End"}],
 "edges": [{"source": "start", "target": "validate"}, {"source": "validate", "target": "end"}]}
```"#;

/// A generation answer whose edge names a node that does not exist, so the
/// document parses but fails `WorkflowDsl::validate`.
const DANGLING_FENCE: &str = r#"```json
{"name": "dangling", "description": "an edge to nowhere",
 "nodes": [{"id": "start", "type": "start", "title": "Start"},
           {"id": "end", "type": "end", "title": "End"}],
 "edges": [{"source": "start", "target": "ghost"}]}
```"#;

/// A revision that is invalid too: a parseable document with no nodes.
const EMPTY_FENCE: &str = r#"```json
{"name": "empty", "description": "no nodes", "nodes": [], "edges": []}
```"#;

/// A complete LLM-agent document with explicit identity and timestamps.
const AGENT_FENCE: &str = r#"```json
{"agent_id": "agent-1", "agent_type": "react", "name": "Demo",
 "description": "a demo agent", "configs": {"system_prompt": "You are helpful."},
 "create_time": 1, "update_time": 1}
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

#[tokio::test]
async fn workflow_provenance_records_every_stage() {
    let model = Arc::new(ScriptedModel::new(workflow_turns()));
    let mut builder = AgentBuilder::new(dynamic(&model));

    let result = builder
        .build_with_pipeline(QUERY, "w1", AgentType::Workflow, 1_700_000_000_000)
        .await
        .expect("the pipeline succeeds");

    let stages: Vec<(ProgressStage, GenerationSource)> = result
        .provenance
        .stages
        .iter()
        .map(|record| (record.stage, record.source))
        .collect();
    assert_eq!(
        stages,
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
    assert!(result.provenance.has_deviation());
    assert!(!result.provenance.has_fallback());
    assert_eq!(result.repair.attempts, 1);
    assert_eq!(result.repair.notes, Vec::<String>::new());

    // The degraded intention stage is visible on the wire, and the four Python
    // keys are unchanged.
    let value = result.to_json().expect("serialize");
    assert!(value.get("provenance").is_some());
    // The repair report is carried even when a single attempt passed.
    assert_eq!(value["repair"]["attempts"], 1);
    assert_eq!(value["repair"]["notes"], serde_json::json!([]));
    assert_eq!(value["status"], serde_json::json!("completed"));
}

#[tokio::test]
async fn repair_loop_refines_until_the_document_passes() {
    let model = Arc::new(ScriptedModel::new([
        ScriptedTurn::text(CLARIFY_READY),
        ScriptedTurn::text(DESIGN_FENCE),
        ScriptedTurn::text(DESIGN_FENCE),
        ScriptedTurn::text(DESIGN_FENCE),
        ScriptedTurn::text(CYCLIC_ONE),
        ScriptedTurn::text(CYCLIC_TWO),
        ScriptedTurn::text(FIXED_FENCE),
    ]));
    let mut builder = AgentBuilder::new(dynamic(&model));

    let result = builder
        .build_with_pipeline(QUERY, "w1", AgentType::Workflow, 1_700_000_000_000)
        .await
        .expect("the repair loop succeeds");

    assert_eq!(result.status, "completed");
    let workflow = workflow_of(&result);
    workflow.validate().expect("the repaired workflow is valid");
    assert_eq!(workflow.name, "fixed");
    assert_eq!(result.repair.attempts, 3);
    assert_eq!(result.repair.notes.len(), 2);
    assert!(result.repair.notes[0].contains("cycle"));
    assert!(result.repair.notes[1].contains("cycle"));

    // Both refinements are recorded, and the model was called once each.
    let refinements = result
        .provenance
        .stages
        .iter()
        .filter(|record| record.stage == ProgressStage::RefiningDl)
        .count();
    assert_eq!(refinements, 2);
    assert_eq!(model.call_count(), 7);
}

#[tokio::test]
async fn repair_loop_exhausts_its_budget_without_error() {
    let model = Arc::new(ScriptedModel::new([
        ScriptedTurn::text(CLARIFY_READY),
        ScriptedTurn::text(DESIGN_FENCE),
        ScriptedTurn::text(DESIGN_FENCE),
        ScriptedTurn::text(DESIGN_FENCE),
        ScriptedTurn::text(CYCLIC_ONE),
        ScriptedTurn::text(CYCLIC_TWO),
        ScriptedTurn::text(CYCLIC_THREE),
    ]));
    let mut builder = AgentBuilder::new(dynamic(&model));

    let result = builder
        .build_with_pipeline(QUERY, "w1", AgentType::Workflow, 1_700_000_000_000)
        .await
        .expect("exhaustion is not an error");

    let workflow = workflow_of(&result);
    workflow.validate().expect("the last document is valid");
    assert_eq!(result.repair.attempts, 3);
    assert_eq!(result.repair.notes.len(), 2);
    assert_eq!(model.call_count(), 7);
    assert_eq!(
        builder.status("w1").expect("the session exists").state,
        BuildState::Completed
    );
}

#[tokio::test]
async fn repair_loop_rescues_an_invalid_model_document() {
    // The generation answer names a node that does not exist, so it fails
    // validation; the loop repairs it and the revision is accepted.
    let model = Arc::new(ScriptedModel::new([
        ScriptedTurn::text(CLARIFY_READY),
        ScriptedTurn::text(DESIGN_FENCE),
        ScriptedTurn::text(DESIGN_FENCE),
        ScriptedTurn::text(DESIGN_FENCE),
        ScriptedTurn::text(DANGLING_FENCE),
        ScriptedTurn::text(REFINED_FENCE),
    ]));
    let mut builder = AgentBuilder::new(dynamic(&model));

    let result = builder
        .build_with_pipeline(QUERY, "w1", AgentType::Workflow, 1_700_000_000_000)
        .await
        .expect("the repair loop rescues the invalid answer");

    assert_eq!(result.status, "completed");
    let workflow = workflow_of(&result);
    workflow.validate().expect("the rescued workflow validates");
    assert_eq!(workflow.name, "refined", "the revision is used");
    assert_ne!(
        workflow.name, "dangling",
        "the invalid answer is not used as-is"
    );
    assert!(workflow.nodes.iter().any(|node| node.id == "validate"));

    // Attempt one repaired the dangling edge, attempt two needed no repair.
    assert!(result.repair.attempts > 0);
    assert_eq!(result.repair.attempts, 2);
    assert_eq!(result.repair.notes.len(), 1, "{:?}", result.repair.notes);
    assert!(
        result.repair.notes[0].contains("ghost"),
        "{:?}",
        result.repair.notes
    );
    assert!(
        result.repair.notes[0].contains("dangling"),
        "{:?}",
        result.repair.notes
    );

    // The model answered throughout, so nothing fell back.
    assert!(
        !result.provenance.has_fallback(),
        "{:?}",
        result.provenance.stages
    );
    let generation = result
        .provenance
        .stages
        .iter()
        .find(|record| record.stage == ProgressStage::GeneratingDl)
        .expect("the generation stage is recorded");
    assert_eq!(generation.source, GenerationSource::Model);
    assert!(generation.detail.contains("needs repair"), "{generation:?}");
    let refinements: Vec<_> = result
        .provenance
        .stages
        .iter()
        .filter(|record| record.stage == ProgressStage::RefiningDl)
        .collect();
    assert_eq!(refinements.len(), 1);
    assert_eq!(refinements[0].source, GenerationSource::Model);

    // Five stages plus the one revision the rescue needed.
    assert_eq!(model.call_count(), 6);
    let calls = model.calls();
    let feedback = calls[5]
        .last_user_text()
        .expect("the revision has a prompt");
    assert!(
        feedback.contains("ghost"),
        "the refine prompt names the dangling node: {feedback}"
    );
    assert!(feedback.contains("dangling"), "{feedback}");
}

#[tokio::test]
async fn repair_loop_rejects_a_model_document_that_stays_invalid() {
    // The revision is invalid too, so the model's document never validates and
    // the build stands on the deterministic baseline.
    let model = Arc::new(ScriptedModel::new([
        ScriptedTurn::text(CLARIFY_READY),
        ScriptedTurn::text(DESIGN_FENCE),
        ScriptedTurn::text(DESIGN_FENCE),
        ScriptedTurn::text(DESIGN_FENCE),
        ScriptedTurn::text(DANGLING_FENCE),
        ScriptedTurn::text(EMPTY_FENCE),
    ]));
    let mut builder = AgentBuilder::new(dynamic(&model));

    let result = builder
        .build_with_pipeline(QUERY, "w1", AgentType::Workflow, 1_700_000_000_000)
        .await
        .expect("an unrepaired answer is not an error");

    let workflow = workflow_of(&result);
    workflow
        .validate()
        .expect("the baseline workflow validates");
    assert_eq!(workflow.name, "demo", "the design-derived baseline is used");
    assert_ne!(workflow.name, "dangling");

    assert!(result.repair.attempts >= 1);
    assert_eq!(result.repair.attempts, 1);
    assert!(result.provenance.has_fallback());
    let fallback = result
        .provenance
        .stages
        .iter()
        .find(|record| {
            record.stage == ProgressStage::ValidatingDl
                && record.source == GenerationSource::Fallback
        })
        .expect("the discarded model document is recorded");
    assert!(fallback.detail.contains("stayed invalid"), "{fallback:?}");

    assert_eq!(model.call_count(), 6);
    let calls = model.calls();
    let feedback = calls[5]
        .last_user_text()
        .expect("the revision has a prompt");
    assert!(
        feedback.contains("ghost"),
        "the refine prompt names the dangling node: {feedback}"
    );
    assert!(
        builder.status("w1").expect("the session exists").state == BuildState::Completed,
        "the build still completes"
    );
}

#[tokio::test]
async fn refine_path_on_a_second_turn_updates_the_stored_artifact() {
    let model = Arc::new(ScriptedModel::new([
        ScriptedTurn::text(CLARIFY_READY),
        ScriptedTurn::text(DESIGN_FENCE),
        ScriptedTurn::text(DESIGN_FENCE),
        ScriptedTurn::text(DESIGN_FENCE),
        ScriptedTurn::text(GENERATED_FENCE),
        ScriptedTurn::text(NEED_REFINED),
        ScriptedTurn::text(REFINED_FENCE),
    ]));
    let mut builder = AgentBuilder::new(dynamic(&model));

    builder
        .build_with_pipeline(QUERY, "w1", AgentType::Workflow, 1_700_000_000_000)
        .await
        .expect("the first build succeeds");

    let result = builder
        .build_with_pipeline(
            "add a validation step",
            "w1",
            AgentType::Workflow,
            1_700_000_000_001,
        )
        .await
        .expect("the refinement succeeds");

    assert_eq!(result.status, "completed");
    let workflow = workflow_of(&result);
    assert_eq!(workflow.name, "refined");
    assert!(result
        .provenance
        .stages
        .iter()
        .any(|record| record.stage == ProgressStage::RefiningDl
            && record.source == GenerationSource::Model));

    assert_eq!(model.call_count(), 7);
    let calls = model.calls();
    // The intention call reads the previous diagram.
    assert!(calls[5]
        .last_user_text()
        .is_some_and(|text| text.contains("graph TD")));
    // The refinement prompt carries the diagram and the request.
    assert!(calls[6]
        .last_user_text()
        .is_some_and(|text| text.contains("graph TD")));
    assert!(calls[6]
        .last_user_text()
        .is_some_and(|text| text.contains("add a validation step")));

    let history = builder.history("w1", None);
    assert_eq!(history.len(), 4);
    assert_eq!(history[2].content, "add a validation step");
    let persisted =
        WorkflowDsl::from_json_str(&history[3].content).expect("the refined document round-trips");
    assert_eq!(persisted.name, "refined");
}

#[tokio::test]
async fn first_turn_refine_intent_is_ignored_without_a_previous_artifact() {
    let model = Arc::new(ScriptedModel::new([]));
    let mut builder = AgentBuilder::new(dynamic(&model));

    let result = builder
        .build_with_pipeline(
            "modify the workflow to add a step",
            "w1",
            AgentType::Workflow,
            1,
        )
        .await
        .expect("the pipeline succeeds");

    // The first request detects the intention deterministically, so the only
    // model call is the clarifier — the refinement path was never entered.
    assert_eq!(model.call_count(), 1);
    assert!(model.calls()[0].messages[0]
        .text()
        .contains("requirement clarifier"));
    assert!(matches!(result.output, BuildOutput::Response(_)));
    assert_eq!(
        result.provenance.stages.first().map(|record| record.stage),
        Some(ProgressStage::DetectingIntention)
    );
    assert_eq!(
        result.provenance.stages.first().map(|record| record.source),
        Some(GenerationSource::Deterministic)
    );
    assert!(result
        .provenance
        .stages
        .iter()
        .all(|record| record.stage != ProgressStage::RefiningDl));
}

#[tokio::test]
async fn llm_agent_provenance_distinguishes_model_and_fallback() {
    for (answer, expected, fallback) in [
        (AGENT_FENCE, GenerationSource::Model, false),
        ("I cannot help with that.", GenerationSource::Fallback, true),
    ] {
        let model = Arc::new(ScriptedModel::new([
            ScriptedTurn::text(CLARIFY_READY),
            ScriptedTurn::text(answer),
        ]));
        let mut builder = AgentBuilder::new(dynamic(&model));

        let result = builder
            .build_with_pipeline(QUERY, "s1", AgentType::LlmAgent, 1_700_000_000_000)
            .await
            .expect("the pipeline succeeds");

        let generation = result
            .provenance
            .stages
            .iter()
            .find(|record| record.stage == ProgressStage::GeneratingConfig)
            .expect("the generation stage is recorded");
        assert_eq!(generation.source, expected, "{answer}");
        assert_eq!(result.provenance.has_fallback(), fallback, "{answer}");
        if fallback {
            assert!(generation.detail.contains("LLM-agent document"));
        }
        assert_eq!(model.call_count(), 2);
    }
}
