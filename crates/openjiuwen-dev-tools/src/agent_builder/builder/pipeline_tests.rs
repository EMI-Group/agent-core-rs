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

    assert_eq!(model.call_count(), 3);
    let calls = model.calls();
    assert_eq!(calls[0].last_user_text().as_deref(), Some(QUERY));
    assert!(calls[1]
        .last_user_text()
        .is_some_and(|text| text.contains("summary")));
    assert!(calls[2]
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
