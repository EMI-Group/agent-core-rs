//! Offline, deterministic test suite of the `agent` node.
//!
//! Every test drives the real `DeepAgent` with
//! [`openjiuwen_test_support::ScriptedModel`] and either the default workspace
//! or a temporary one; no network, no provider and no real sys-operation work.
//!
//! Covered here (first batch): defaults and round-tripping, prompt content,
//! single-round answers, tool execution, max-iteration mapping, permission
//! denial, mode switching, input dispatch/phases, and the persisted-state entry
//! points.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_agent::{Agent, AgentCard, ResultType};
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::Model;
use openjiuwen_foundation::tool::{Tool, ToolCard, ToolContext};
use openjiuwen_session::{AgentSession, Session};
use openjiuwen_test_support::{
    MockCompletion, MockToolCall, ScriptedModel, ScriptedTurn, TempWorkspace,
};
use serde_json::{json, Value};

use super::{DeepAgent, DeepAgentBuilder};
use crate::permission::{PermissionLevel, PermissionsSection};
use crate::schema::{
    AgentMode, DeepAgentConfig, DeepAgentState, InputDispatchMode, InteractionPhase,
    OnMaxIterations, SendInputRequest,
};
use crate::tools::tool_card;
use crate::workspace::Workspace;

/// A tool echoing its inputs back and counting invocations.
struct EchoTool {
    card: ToolCard,
    calls: Arc<AtomicUsize>,
}

impl EchoTool {
    fn new() -> Self {
        Self {
            card: tool_card(
                "echo",
                "Echo the input text back to the model",
                json!({
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                }),
                true,
            ),
            calls: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl Tool for EchoTool {
    fn card(&self) -> &ToolCard {
        &self.card
    }

    async fn invoke(&self, inputs: Value, _ctx: &ToolContext) -> Result<Value> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(json!({"success": true, "echo": inputs}))
    }
}

/// Builds a scripted-model agent over the default workspace.
fn build_agent(model: impl Model + 'static) -> Arc<DeepAgent> {
    DeepAgentBuilder::new(AgentCard::new("test-agent").with_description("Test DeepAgent"))
        .config(DeepAgentConfig::default().with_language("en".to_string()))
        .model(Arc::new(model))
        .build()
        .expect("agent builds")
}

#[test]
fn new_and_configure_round_trip() {
    let agent = DeepAgent::new(AgentCard::new("facade"));
    assert_eq!(agent.phase(), InteractionPhase::Idle);
    assert!(agent
        .rails()
        .names()
        .iter()
        .any(|name| name == "harness_prompt"));

    let config = DeepAgentConfig::default().with_language("en".to_string());
    agent.configure(config.clone());
    assert_eq!(agent.config(), config);
}

#[test]
fn prompt_contains_identity_workspace_and_tools() {
    let temp = TempWorkspace::new().unwrap();
    let workspace = Workspace::for_language(temp.workspace(), "en");
    let config = DeepAgentConfig::default()
        .with_language("en".to_string())
        .with_workspace(workspace.clone());
    let agent = DeepAgentBuilder::new(
        AgentCard::new("echo-agent").with_description("Identity from the card"),
    )
    .config(config)
    .model(Arc::new(ScriptedModel::text("hi")))
    .tool(Arc::new(EchoTool::new()))
    .build()
    .unwrap();

    let prompt = agent.prompt.system_text();
    assert!(prompt.contains("Identity from the card"));
    assert!(prompt.contains(&workspace.root_path.display().to_string()));
    assert!(prompt.contains("echo"));
}

#[tokio::test]
async fn single_round_returns_the_model_answer() {
    let agent = build_agent(ScriptedModel::text("hello"));
    let output = agent
        .invoke(json!({"query": "hi"}), None)
        .await
        .expect("invocation succeeds");
    assert_eq!(output.result_type, ResultType::Answer);
    assert_eq!(output.output, "hello");
}

#[tokio::test]
async fn tool_round_executes_the_mounted_tool() {
    let echo = Arc::new(EchoTool::new());
    let model = ScriptedModel::builder()
        .completion(MockCompletion::tool_calls(vec![MockToolCall::new(
            "echo",
            json!({"text": "x"}),
        )]))
        .respond_text("done")
        .build();
    let agent = DeepAgentBuilder::new(AgentCard::new("tool-agent"))
        .config(DeepAgentConfig::default().with_language("en".to_string()))
        .model(Arc::new(model))
        .tool(echo.clone())
        .build()
        .unwrap();

    let output = agent
        .invoke(json!({"query": "echo it"}), None)
        .await
        .unwrap();
    assert_eq!(output.output, "done");
    assert_eq!(echo.calls(), 1);
}

#[tokio::test]
async fn max_iterations_maps_to_an_error_output() {
    let call = MockToolCall::new("echo", json!({"text": "x"}));
    let completion = MockCompletion::tool_calls(vec![call]);
    let model = ScriptedModel::builder()
        .completion(completion.clone())
        .fallback(ScriptedTurn::Response(completion.to_chat_response("")))
        .build();
    let config = DeepAgentConfig::default()
        .with_language("en".to_string())
        .with_max_iterations(Some(1))
        .with_on_max_iterations(OnMaxIterations::Error);
    let agent = DeepAgentBuilder::new(AgentCard::new("limit-agent"))
        .config(config)
        .model(Arc::new(model))
        .tool(Arc::new(EchoTool::new()))
        .build()
        .unwrap();

    let output = agent
        .invoke(json!({"query": "loop forever"}), None)
        .await
        .unwrap();
    assert_eq!(output.result_type, ResultType::Error);
    assert!(output.output.to_lowercase().contains("iteration"));
}

#[tokio::test]
async fn permission_denial_skips_the_tool() {
    let echo = Arc::new(EchoTool::new());
    let mut permissions = PermissionsSection::default();
    permissions
        .tools
        .insert("echo".to_string(), PermissionLevel::Deny);
    let model = ScriptedModel::builder()
        .completion(MockCompletion::tool_calls(vec![MockToolCall::new(
            "echo",
            json!({"text": "x"}),
        )]))
        .respond_text("denied and done")
        .build();
    let config = DeepAgentConfig::default()
        .with_language("en".to_string())
        .with_permissions(Some(permissions));
    let agent = DeepAgentBuilder::new(AgentCard::new("permission-agent"))
        .config(config)
        .model(Arc::new(model))
        .tool(echo.clone())
        .build()
        .unwrap();

    let output = agent
        .invoke(json!({"query": "run echo"}), None)
        .await
        .unwrap();
    assert_eq!(echo.calls(), 0);
    assert_eq!(output.output, "denied and done");
}

#[test]
fn switch_mode_updates_state_and_prompt() {
    let agent = build_agent(ScriptedModel::text("unused"));
    agent.switch_mode(AgentMode::Plan).unwrap();
    assert_eq!(agent.config().default_mode, AgentMode::Plan);
    assert_eq!(agent.plan_mode.read().mode, "plan");
    assert_eq!(
        agent.plan_mode.read().pre_plan_mode.as_deref(),
        Some("normal")
    );
    assert!(agent.prompt.system_text().contains("Plan Mode"));

    agent.switch_mode(AgentMode::Normal).unwrap();
    assert_eq!(agent.config().default_mode, AgentMode::Normal);
    assert!(!agent.prompt.system_text().contains("Plan Mode"));
}

#[test]
fn send_input_fills_the_queues_and_phases_transition() {
    let agent = build_agent(ScriptedModel::text("unused"));
    let steer = SendInputRequest {
        request_id: "r1".to_string(),
        inputs: BTreeMap::from([("query".to_string(), json!("steer this"))]),
        mode: Some(InputDispatchMode::Steer),
    };
    agent.send_input(steer).unwrap();
    assert_eq!(agent.queues.steering_len(), 1);

    let follow_up = SendInputRequest {
        request_id: "r2".to_string(),
        inputs: BTreeMap::from([("query".to_string(), json!("do this later"))]),
        mode: Some(InputDispatchMode::FollowUp),
    };
    agent.send_input(follow_up).unwrap();
    assert_eq!(agent.queues.follow_up_len(), 1);

    let empty = SendInputRequest {
        request_id: "r3".to_string(),
        inputs: BTreeMap::from([("query".to_string(), json!(""))]),
        mode: None,
    };
    assert!(matches!(agent.send_input(empty), Err(Error::Validation(_))));

    agent.stop().unwrap();
    assert_eq!(agent.phase(), InteractionPhase::Terminated);
    assert!(matches!(agent.start(), Err(Error::Validation(_))));
    assert_eq!(
        agent.start().unwrap_err().to_string(),
        "validation: interaction_terminated"
    );
}

#[tokio::test]
async fn state_entry_points_round_trip_and_require_a_session() {
    let agent = build_agent(ScriptedModel::text("hello"));
    assert!(matches!(
        agent.save_state().await,
        Err(Error::Validation(_))
    ));
    assert!(matches!(
        agent.load_state().await,
        Err(Error::Validation(_))
    ));
    assert!(matches!(
        agent.clear_state().await,
        Err(Error::Validation(_))
    ));

    let session = Arc::new(AgentSession::new("agent-state-test"));
    agent
        .invoke(json!({"query": "hi"}), Some(session.clone()))
        .await
        .unwrap();
    assert_eq!(agent.load_state().await.unwrap(), DeepAgentState::default());

    agent.save_state().await.unwrap();
    assert_eq!(agent.load_state().await.unwrap(), DeepAgentState::default());
    agent.clear_state().await.unwrap();
    assert_eq!(agent.load_state().await.unwrap(), DeepAgentState::default());
    assert_eq!(session.id(), "agent-state-test");
}
