//! Cross-module integration suite for the agent-asset seam.
//!
//! Exercises the composition the per-module unit tests deliberately leave out:
//! an [`LlmAgentSpec`] becomes a serialized [`LlmAgentDsl`], that document
//! survives JSON and is bridged into an `openjiuwen_agent::LlmAgent`; the
//! session-based [`AgentBuilder`] drives the same document through a model round
//! trip, falling back to a clarifying response, and keeps sessions isolated; a
//! [`WorkflowSpec`] converts into a validated [`WorkflowDsl`].
//!
//! Everything runs offline and deterministically on
//! [`openjiuwen_test_support::ScriptedModel`]; ids and timestamps are explicit.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::BTreeMap;
use std::sync::Arc;

use openjiuwen_agent::Agent;
use openjiuwen_common::error::ErrorKind;
use openjiuwen_dev_tools::agent_builder::{
    map_state_to_status, AgentBuilder, AgentType, BuildOutput, BuildState, DslEdge, DslModelInfo,
    DslNode, LlmAgentDsl, LlmAgentSpec, WorkflowDsl, WorkflowSpec,
};
use openjiuwen_foundation::llm::Model;
use openjiuwen_test_support::ScriptedModel;
use serde_json::json;

/// The fixed timestamp every DSL assertion uses, so no test reads the clock.
const STAMP: i64 = 1_700_000_000_000;

/// A model answer holding a complete LLM-agent document in a JSON fence.
const AGENT_FENCE: &str = r#"```json
{
  "agent_id": "agent-42",
  "agent_type": "react",
  "name": "Session Agent",
  "description": "an agent produced by one model round trip"
}
```"#;

/// A plain-text answer that carries no fenced document.
const CLARIFYING_ANSWER: &str = "Which model provider should the agent use?";

/// Coerces a scripted model into the `Arc<dyn Model>` the builder takes.
fn dynamic(model: &Arc<ScriptedModel>) -> Arc<dyn Model> {
    let cloned: Arc<ScriptedModel> = Arc::clone(model);
    cloned
}

/// The DSL carried by a successful build, asserting the variant.
fn dsl_of<'a>(output: &'a BuildOutput, invariant: &str) -> &'a LlmAgentDsl {
    match output {
        BuildOutput::Dsl(dsl) => dsl,
        other => panic!("{invariant}: expected a DSL output, got {other:?}"),
    }
}

#[test]
fn llm_agent_spec_round_trips_to_a_kernel_agent() {
    let spec = LlmAgentSpec::new(
        "Round Trip Agent",
        "an agent authored from a specification",
        "You are helpful.",
    )
    .with_provider("anthropic")
    .with_model(DslModelInfo {
        api_base: "https://example.test/v1".to_string(),
        model_name: "demo-model".to_string(),
        temperature: Some(0.5),
        top_p: Some(0.9),
    })
    .with_max_iteration(5);

    let dsl = spec.to_dsl("agent-1", STAMP);
    let encoded = dsl
        .to_json_string()
        .expect("a DSL built from a spec must serialize");
    let decoded = LlmAgentDsl::from_json_str(&encoded)
        .expect("a serialized DSL document must parse back unchanged");

    assert_eq!(
        decoded.agent_id, "agent-1",
        "the spec conversion must keep the caller-supplied agent id"
    );
    assert_eq!(
        decoded.create_time, STAMP,
        "the caller-supplied timestamp must survive as create_time"
    );
    assert_eq!(
        decoded.update_time, STAMP,
        "the caller-supplied timestamp must survive as update_time"
    );
    assert_eq!(
        decoded.name, "Round Trip Agent",
        "the spec name must survive the JSON round trip"
    );
    assert_eq!(
        decoded.description, "an agent authored from a specification",
        "the spec description must survive the JSON round trip"
    );
    assert_eq!(
        decoded.agent_type, "react",
        "an agent spec must convert into the documented `react` agent type"
    );

    let scripted = Arc::new(ScriptedModel::text("hi"));
    let model = dynamic(&scripted);
    let agent = decoded
        .build_llm_agent(Some(model))
        .expect("a complete DSL document must build a kernel agent");

    assert_eq!(
        agent.card().id(),
        "agent-1",
        "the kernel agent card must carry the DSL agent id"
    );
    assert_eq!(
        agent.card().name(),
        "Round Trip Agent",
        "the kernel agent card must carry the DSL name"
    );
    assert_eq!(
        agent.card().description(),
        "an agent authored from a specification",
        "the kernel agent card must carry the DSL description"
    );
    let installed = agent
        .react()
        .model()
        .expect("build_llm_agent must install the supplied model");
    assert_eq!(
        installed.model_name(),
        "scripted-model",
        "the built agent must report the supplied model's name"
    );
}

#[tokio::test]
async fn agent_builder_completes_a_session_from_a_json_fence() {
    let scripted = Arc::new(ScriptedModel::builder().respond_text(AGENT_FENCE).build());
    let mut builder = AgentBuilder::new(dynamic(&scripted));

    let result = builder
        .build_agent_at(
            "build me an LLM agent",
            "session-complete",
            AgentType::LlmAgent,
            STAMP,
        )
        .await
        .expect("a fenced DSL answer must not be an error");

    let dsl = dsl_of(
        &result.output,
        "a fenced document must complete the session",
    );
    assert_eq!(
        result.status, "completed",
        "a parsed document must report the `completed` status string"
    );
    assert_eq!(
        scripted.call_count(),
        1,
        "one request must consume exactly one model round trip"
    );
    assert_eq!(
        dsl.agent_id, "agent-42",
        "the parsed DSL must keep the scripted agent id"
    );
    assert_eq!(
        dsl.name, "Session Agent",
        "the parsed DSL must keep the scripted agent name"
    );

    let history = builder.history("session-complete", None);
    assert_eq!(
        history.len(),
        2,
        "a completed session must retain exactly one user and one assistant turn"
    );
    assert_eq!(
        history[0].role, "user",
        "the first retained turn must be the caller's request"
    );
    assert_eq!(
        history[1].role, "assistant",
        "the second retained turn must be the model's answer"
    );
    assert_eq!(
        history[1].content, AGENT_FENCE,
        "the retained assistant turn must be the scripted answer verbatim"
    );

    let json = result
        .to_json()
        .expect("a successful build result must serialize");
    assert!(
        json.get("dsl").is_some(),
        "a completed build must expose a top-level `dsl` key"
    );
    assert!(
        json.get("response").is_none(),
        "a completed build must not expose a `response` key"
    );
}

#[tokio::test]
async fn agent_builder_returns_a_clarifying_response_for_plain_text() {
    let scripted = Arc::new(ScriptedModel::text(CLARIFYING_ANSWER));
    let mut builder = AgentBuilder::new(dynamic(&scripted));

    let result = builder
        .build_agent_at(
            "build me an LLM agent",
            "session-clarify",
            AgentType::LlmAgent,
            STAMP,
        )
        .await
        .expect("an unparseable answer is a clarifying response, not an error");

    let expected_status = map_state_to_status(BuildState::Clarifying, AgentType::LlmAgent);
    assert_eq!(
        result.status, expected_status,
        "a clarifying answer must report the status mapped from the clarifying state"
    );
    match &result.output {
        BuildOutput::Response(answer) => assert_eq!(
            answer, CLARIFYING_ANSWER,
            "the clarifying response must be the model's text verbatim"
        ),
        other => panic!("plain text must yield a response output, got {other:?}"),
    }
    assert_eq!(
        scripted.call_count(),
        1,
        "a clarifying request must still consume exactly one model round trip"
    );
    assert_eq!(
        builder.history("session-clarify", None).len(),
        2,
        "even a clarifying answer must be retained as an assistant turn"
    );

    let json = result
        .to_json()
        .expect("a clarifying build result must serialize");
    assert!(
        json.get("response").is_some(),
        "a clarifying build must expose a top-level `response` key"
    );
    assert!(
        json.get("dsl").is_none(),
        "a clarifying build must not expose a `dsl` key"
    );
}

#[tokio::test]
async fn clear_session_drops_history_and_status() {
    let scripted = Arc::new(
        ScriptedModel::builder()
            .respond_text(AGENT_FENCE)
            .respond_text(AGENT_FENCE)
            .build(),
    );
    let mut builder = AgentBuilder::new(dynamic(&scripted));

    for session_id in ["session-a", "session-b"] {
        let result = builder
            .build_agent_at(
                "build me an LLM agent",
                session_id,
                AgentType::LlmAgent,
                STAMP,
            )
            .await
            .expect("both sessions must complete from the scripted fence");
        dsl_of(
            &result.output,
            "each scripted fence must complete its session",
        );
    }

    builder.clear_session("session-a");

    assert!(
        builder.history("session-a", None).is_empty(),
        "clearing a session must drop its retained dialogue"
    );
    assert!(
        builder.status("session-a").is_none(),
        "clearing a session must drop its status snapshot"
    );

    let untouched = builder
        .status("session-b")
        .expect("clearing one session must leave every other session registered in the builder");
    assert_eq!(
        untouched.state,
        BuildState::Completed,
        "the other session must keep its completed build state"
    );
    assert_eq!(
        untouched.resource_count, 2,
        "the other session must keep both retained dialogue turns"
    );
    assert_eq!(
        builder.history("session-b", None).len(),
        2,
        "the other session must keep its history across another session's clear"
    );
    assert_eq!(
        scripted.call_count(),
        2,
        "one round trip per session must be recorded on the shared model"
    );
}

#[test]
fn workflow_spec_converts_to_a_validated_dsl() {
    let spec = WorkflowSpec {
        name: "brief-flow".to_string(),
        description: "extract then summarize".to_string(),
        nodes: vec![
            DslNode {
                id: "extract".to_string(),
                node_type: "llm".to_string(),
                title: "Extract".to_string(),
                description: "pull the claims out of the document".to_string(),
                inputs: BTreeMap::from([("document".to_string(), json!({"type": "string"}))]),
                outputs: BTreeMap::from([("claims".to_string(), json!({"type": "array"}))]),
                data: json!({"prompt": "Extract the claims."}),
            },
            DslNode {
                id: "summarize".to_string(),
                node_type: "end".to_string(),
                title: "Summarize".to_string(),
                description: "render the brief".to_string(),
                inputs: BTreeMap::new(),
                outputs: BTreeMap::new(),
                data: json!({"prompt": "Write the brief."}),
            },
        ],
        edges: vec![DslEdge {
            source: "extract".to_string(),
            target: "summarize".to_string(),
            condition: None,
        }],
    };

    let dsl = spec.to_dsl();
    dsl.validate()
        .expect("a two-node, one-edge spec must convert into a structurally valid DSL");

    let node_ids: Vec<&str> = dsl.nodes.iter().map(|node| node.id.as_str()).collect();
    assert_eq!(
        node_ids,
        vec!["extract", "summarize"],
        "the conversion must preserve the node ids in document order"
    );
    assert_eq!(
        (dsl.edges[0].source.as_str(), dsl.edges[0].target.as_str()),
        ("extract", "summarize"),
        "the conversion must preserve the edge endpoints"
    );
    assert!(
        dsl.node("extract").is_some(),
        "node(\"extract\") must resolve the first node by id"
    );
    assert!(
        dsl.node("summarize").is_some(),
        "node(\"summarize\") must resolve the second node by id"
    );
    assert!(
        dsl.node("missing").is_none(),
        "node() must not invent a node the document does not contain"
    );

    let encoded = spec
        .to_json_string()
        .expect("a populated workflow spec must serialize");
    let decoded =
        WorkflowDsl::from_json_str(&encoded).expect("the serialized spec must parse as a DSL");
    assert_eq!(
        decoded, dsl,
        "the workflow document must round trip through JSON unchanged"
    );
    assert_eq!(
        decoded
            .node("summarize")
            .map(|node| node.node_type.as_str()),
        Some("end"),
        "the round-tripped document must keep the node type under its `type` wire name"
    );

    let broken = WorkflowSpec {
        name: "broken-flow".to_string(),
        description: "an edge that names a missing node".to_string(),
        nodes: spec.nodes.clone(),
        edges: vec![DslEdge {
            source: "extract".to_string(),
            target: "unknown-node".to_string(),
            condition: None,
        }],
    };
    let error = broken
        .to_dsl()
        .validate()
        .expect_err("an edge naming a non-existent node must fail validation");
    assert_eq!(
        error.kind(),
        ErrorKind::Validation,
        "an unresolvable edge endpoint must be a validation error, not a runtime failure"
    );
    assert!(
        error.to_string().contains("unknown-node"),
        "the validation error must name the offending node id: {error}"
    );
}
