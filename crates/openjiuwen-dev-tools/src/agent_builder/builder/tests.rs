//! Tests for the session-based `AgentBuilder`: the legacy single round trip,
//! its result vocabulary, and the deterministic LLM-agent repair.
//!
//! Every case runs offline on scripted models: no network, no environment reads,
//! and no wall-clock dependence.
use super::*;
use openjiuwen_common::error::ErrorKind;
use openjiuwen_test_support::{ScriptedModel, ScriptedTurn};

/// A model answer holding an LLM-agent document without timestamps.
const AGENT_FENCE: &str = r#"Here is the agent you asked for.
```json
{"agent_id": "agent-1", "agent_type": "react", "name": "Demo Agent", "description": "a demo agent",
 "configs": {"system_prompt": "You are helpful.", "opening_remarks": "Hi!"},
 "plugins": [{"id": "plugin-1", "name": "search"}], "dependencies": {"plugins": ["plugin-1"]}}
```"#;

/// A model answer holding a valid two-node workflow document.
const WORKFLOW_FENCE: &str = r#"```json
{"name": "demo", "description": "a demo workflow",
 "nodes": [{"id": "start", "type": "start", "title": "Start"}, {"id": "end", "type": "end"}],
 "edges": [{"source": "start", "target": "end"}]}
```"#;

/// Coerces a scripted model into the trait object the builder takes.
fn dynamic(model: &Arc<ScriptedModel>) -> Arc<dyn Model> {
    let cloned: Arc<ScriptedModel> = Arc::clone(model);
    cloned
}

/// The DSL document of a successful build result.
fn dsl_of(result: &BuildResult) -> &LlmAgentDsl {
    match &result.output {
        BuildOutput::Dsl(dsl) => dsl,
        other => panic!("expected DSL, got {other:?}"),
    }
}

/// The workflow document of a successful build result.
fn workflow_of(result: &BuildResult) -> &WorkflowDsl {
    match &result.output {
        BuildOutput::Workflow(workflow) => workflow,
        other => panic!("expected workflow, got {other:?}"),
    }
}

#[test]
fn agent_type_round_trips_through_display_and_from_str() {
    for agent_type in [AgentType::LlmAgent, AgentType::Workflow] {
        assert_eq!(agent_type.as_str(), agent_type.to_string());
        assert_eq!(
            agent_type
                .as_str()
                .parse::<AgentType>()
                .expect("round trip"),
            agent_type
        );
    }
    assert_eq!(AgentType::LlmAgent.to_string(), "llm_agent");
    assert_eq!(AgentType::Workflow.to_string(), "workflow");
    for value in ["", "LLMAgent", "llmagent", "workflow_dsl"] {
        let error = value.parse::<AgentType>().expect_err("unknown name");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("unknown agent type"));
        assert!(error.to_string().contains("llm_agent"));
        assert!(error.to_string().contains("workflow"));
    }
}

#[tokio::test]
async fn build_agent_returns_the_parsed_dsl() {
    let model = Arc::new(ScriptedModel::text(AGENT_FENCE));
    let mut builder = AgentBuilder::new(dynamic(&model));

    let result = builder
        .build_agent("build me an agent", "s1", AgentType::LlmAgent)
        .await
        .expect("build succeeds");

    assert_eq!(result.status, "completed");
    assert_eq!(result.session_id, "s1");
    assert_eq!(result.agent_type, AgentType::LlmAgent);
    let dsl = dsl_of(&result);
    assert_eq!(dsl.agent_id, "agent-1");
    assert_eq!(dsl.configs.opening_remarks, "Hi!");
    assert!(dsl.create_time > 0, "the clock fills the create time");
    assert_eq!(dsl.create_time, dsl.update_time);

    assert_eq!(model.call_count(), 1);
    let call = model.last_call().expect("one call");
    assert_eq!(call.messages.len(), 2);
    assert_eq!(call.messages[0].text(), LLM_AGENT_SYSTEM_PROMPT);
    assert_eq!(call.messages[1].text(), "build me an agent");
    assert_eq!(call.request.model_name, "scripted-model");

    let status = builder.status("s1").expect("session is known");
    assert_eq!(status.session_id, "s1");
    assert_eq!(status.agent_type, AgentType::LlmAgent);
    assert_eq!(status.state, BuildState::Completed);
    assert_eq!(status.resource_count, 2);
    let history = builder.history("s1", None);
    assert_eq!(history.len(), 2);
    assert_eq!(
        history[0],
        DialogueMessage {
            role: "user".to_string(),
            content: "build me an agent".to_string(),
        }
    );
    assert_eq!(
        history[1],
        DialogueMessage {
            role: "assistant".to_string(),
            content: AGENT_FENCE.to_string(),
        }
    );
}

#[tokio::test]
async fn build_agent_at_stamps_missing_metadata_only() {
    let model = Arc::new(ScriptedModel::text(AGENT_FENCE));
    let mut builder = AgentBuilder::new(dynamic(&model));
    let result = builder
        .build_agent_at("go", "s1", AgentType::LlmAgent, 1_700_000_000_000)
        .await
        .expect("build succeeds");
    let dsl = dsl_of(&result);
    assert_eq!(dsl.agent_id, "agent-1");
    assert_eq!(dsl.create_time, 1_700_000_000_000);
    assert_eq!(dsl.update_time, 1_700_000_000_000);

    let nameless = Arc::new(ScriptedModel::text(
        "```json\n{\"name\": \"nameless\"}\n```",
    ));
    let mut builder = AgentBuilder::new(dynamic(&nameless));
    let result = builder
        .build_agent_at("go", "s2", AgentType::LlmAgent, 42)
        .await
        .expect("build succeeds");
    let dsl = dsl_of(&result);
    assert_eq!(dsl.agent_type, "react");
    assert!(!dsl.agent_id.is_empty(), "a fallback id is generated");
    assert_eq!(dsl.create_time, 42);
    assert_eq!(dsl.update_time, 42);
}

#[tokio::test]
async fn sessions_are_isolated_and_clearable() {
    let model = Arc::new(ScriptedModel::new([
        ScriptedTurn::text(AGENT_FENCE),
        ScriptedTurn::text("which workflow do you want?"),
    ]));
    let mut builder = AgentBuilder::new(dynamic(&model));

    builder
        .build_agent("build me an agent", "s1", AgentType::LlmAgent)
        .await
        .expect("first build succeeds");
    assert_eq!(builder.history("s2", None).len(), 0);
    assert!(builder.status("s2").is_none());
    assert_eq!(builder.history("s1", None).len(), 2);

    let result = builder
        .build_agent("build me a workflow", "s2", AgentType::Workflow)
        .await
        .expect("second build succeeds");
    assert_eq!(result.status, "clarifying");
    assert_eq!(result.session_id, "s2");
    assert_eq!(builder.history("s2", None).len(), 2);
    assert_eq!(builder.history("s1", None).len(), 2);
    assert_eq!(model.calls()[1].messages[0].text(), WORKFLOW_SYSTEM_PROMPT);

    builder.clear_session("s1");
    assert!(builder.status("s1").is_none());
    assert_eq!(builder.history("s1", None).len(), 0);
    builder.clear_session("never-seen");
    assert_eq!(builder.history("s2", None).len(), 2);
}

#[tokio::test]
async fn history_k_limits_the_returned_turns() {
    let model = Arc::new(ScriptedModel::new([
        ScriptedTurn::text(AGENT_FENCE),
        ScriptedTurn::text("and now?"),
    ]));
    let mut builder = AgentBuilder::new(dynamic(&model));

    for query in ["build me an agent", "add a plugin"] {
        builder
            .build_agent(query, "s1", AgentType::LlmAgent)
            .await
            .expect("build succeeds");
    }
    assert_eq!(builder.history("s1", None).len(), 4);
    assert_eq!(builder.history("s1", Some(1))[0].content, "and now?");
    assert_eq!(builder.history("s1", Some(0)).len(), 0);
    assert_eq!(builder.history("s1", Some(99)).len(), 4);
    assert_eq!(model.call_count(), 2);
}

#[tokio::test]
async fn plain_text_answer_is_a_clarifying_response() {
    let model = Arc::new(ScriptedModel::text("Which system prompt should I use?"));
    let mut builder = AgentBuilder::new(dynamic(&model));

    let result = builder
        .build_agent("make an agent", "s1", AgentType::LlmAgent)
        .await
        .expect("build succeeds");

    assert_eq!(result.status, "clarifying");
    assert_eq!(
        result.output,
        BuildOutput::Response("Which system prompt should I use?".to_string())
    );
    let status = builder.status("s1").expect("session is known");
    assert_eq!(status.state, BuildState::Clarifying);
    assert_eq!(status.resource_count, 2);
}

#[tokio::test]
async fn blank_query_is_rejected_before_a_session_exists() {
    let model = Arc::new(ScriptedModel::text(AGENT_FENCE));
    let mut builder = AgentBuilder::new(dynamic(&model));

    for query in ["", "   ", "\n\t "] {
        let error = builder
            .build_agent(query, "s1", AgentType::LlmAgent)
            .await
            .expect_err("blank query is invalid");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("must not be blank"));
    }
    assert!(builder.status("s1").is_none());
    assert_eq!(builder.history("s1", None).len(), 0);
    assert_eq!(model.call_count(), 0);
}

#[tokio::test]
async fn valid_workflow_fence_completes_the_session() {
    let model = Arc::new(ScriptedModel::text(WORKFLOW_FENCE));
    let mut builder = AgentBuilder::new(dynamic(&model));

    let result = builder
        .build_agent("build a workflow", "w1", AgentType::Workflow)
        .await
        .expect("build succeeds");

    assert_eq!(result.status, "completed");
    assert_eq!(result.agent_type, AgentType::Workflow);
    let workflow = workflow_of(&result);
    assert_eq!(workflow.nodes.len(), 2);
    assert_eq!(workflow.edges[0].target, "end");
    assert_eq!(
        builder.status("w1").expect("session is known").state,
        BuildState::Completed
    );
}

#[tokio::test]
async fn invalid_workflow_falls_back_to_a_response() {
    for answer in [
        r#"```json
{"name": "broken", "nodes": [{"id": "start", "type": "start"}], "edges": [{"source": "start", "target": "ghost"}]}
```"#,
        "```json\n{\"name\": \"empty\"}\n```",
    ] {
        let model = Arc::new(ScriptedModel::text(answer));
        let mut builder = AgentBuilder::new(dynamic(&model));

        let result = builder
            .build_agent("build a workflow", "w1", AgentType::Workflow)
            .await
            .expect("build succeeds");

        assert_eq!(result.status, "clarifying");
        assert!(matches!(result.output, BuildOutput::Response(_)));
        assert_eq!(
            builder.status("w1").expect("session is known").state,
            BuildState::Clarifying
        );
    }
}

#[tokio::test]
async fn model_failure_propagates_and_keeps_the_session_processing() {
    let model = Arc::new(ScriptedModel::new([ScriptedTurn::fail("boom")]));
    let mut builder = AgentBuilder::new(dynamic(&model));

    let error = builder
        .build_agent("build me an agent", "s1", AgentType::LlmAgent)
        .await
        .expect_err("the model failed");
    assert_eq!(error.kind(), ErrorKind::Llm);
    assert!(error.to_string().contains("boom"));

    let status = builder.status("s1").expect("the session was created");
    assert_eq!(status.state, BuildState::Processing);
    assert_eq!(status.resource_count, 1);
    let history = builder.history("s1", None);
    assert_eq!(history.len(), 1);
    assert_eq!(history[0].role, "user");
    assert_eq!(history[0].content, "build me an agent");
}

#[tokio::test]
async fn build_agent_uses_the_clock_for_missing_timestamps() {
    let model = Arc::new(ScriptedModel::text(
        "```json\n{\"agent_id\": \"agent-1\", \"name\": \"Demo\"}\n```",
    ));
    let mut builder = AgentBuilder::new(dynamic(&model));

    let result = builder
        .build_agent("go", "s1", AgentType::LlmAgent)
        .await
        .expect("build succeeds");

    let dsl = dsl_of(&result);
    assert!(dsl.create_time > 0);
    assert_eq!(dsl.create_time, dsl.update_time);
}

#[test]
fn model_clock_and_debug_surface() {
    let model = Arc::new(ScriptedModel::text("hi"));
    let builder = AgentBuilder::new(dynamic(&model));
    assert_eq!(builder.model().provider(), "scripted");
    assert_eq!(builder.model().model_name(), "scripted-model");
    assert!(now_millis() > 0);

    let rendered = format!("{builder:?}");
    assert!(rendered.contains("provider: \"scripted\""), "{rendered}");
    assert!(
        rendered.contains("model_name: \"scripted-model\""),
        "{rendered}"
    );
    assert!(rendered.contains("sessions: 0"), "{rendered}");
}
