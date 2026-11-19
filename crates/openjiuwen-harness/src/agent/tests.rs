//! Offline, deterministic test suite of the `agent` node.
//!
//! Every test drives the real `DeepAgent` with
//! [`openjiuwen_test_support::ScriptedModel`] and either the default workspace
//! or a temporary one; no network, no provider and no real sys-operation work.
//!
//! Covered here: defaults and round-tripping, prompt content, single-round
//! answers, tool execution, max-iteration mapping, permission denial, mode
//! switching, input dispatch/phases, the persisted-state entry points, the outer
//! task loop (follow-up rounds, steering, abort), stream frames, the single-run
//! guard, rail-published prompt sections, and the factory assembly.
// Size note: this file is the `agent` node's test suite only; its length is test cases,
// not production code.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::StreamExt;
use openjiuwen_agent::{Agent, AgentCard, AgentChunk, AgentOutput, ResultType};
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::{ChatResponse, Model, ModelRequestConfig, StreamChunk};
use openjiuwen_foundation::tool::{Tool, ToolCard, ToolContext};
use openjiuwen_foundation::{BaseMessage, Role};
use openjiuwen_session::{AgentSession, Session};
use openjiuwen_test_support::{
    MockCompletion, MockToolCall, ScriptedModel, ScriptedTurn, TempWorkspace,
};
use serde_json::{json, Value};
use tokio::sync::Semaphore;

use super::{create_deep_agent, DeepAgent, DeepAgentBuilder};
use crate::agent::RAIL_EVENTS_EXTRA_PREFIX;
use crate::permission::{PermissionLevel, PermissionsSection};
use crate::prompts::{PromptSection, SectionName};
use crate::rails::{publish_prompt_section, HarnessRail, HarnessRailContext};
use crate::schema::{
    AgentMode, DeepAgentConfig, DeepAgentState, InputDispatchMode, InteractionEvent,
    InteractionEventType, InteractionPhase, OnMaxIterations, SendInputRequest, SubAgentConfig,
};
use crate::tools::goal::GOAL_EVENT_EXTRA_KEY;
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

/// A rail publishing one prompt section before every model call.
struct SectionRail {
    name: &'static str,
    section: PromptSection,
}

#[async_trait]
impl HarnessRail for SectionRail {
    fn name(&self) -> &str {
        self.name
    }

    fn priority(&self) -> i32 {
        crate::rails::rail_priority::DEFAULT
    }

    async fn before_model_call(&self, ctx: &mut HarnessRailContext) -> Result<()> {
        publish_prompt_section(ctx, &self.section);
        Ok(())
    }
}

/// A model whose streamed call blocks until the test opens the gate.
///
/// The gate proves the single-run guard: the first invocation is parked inside
/// the model while the test issues the second one.
struct GateModel {
    inner: ScriptedModel,
    /// One permit is added as soon as a streamed call is entered.
    entered: Arc<Semaphore>,
    /// The call proceeds after the test adds one permit here.
    release: Arc<Semaphore>,
}

#[async_trait]
impl Model for GateModel {
    fn provider(&self) -> &str {
        self.inner.provider()
    }

    fn model_name(&self) -> &str {
        self.inner.model_name()
    }

    async fn invoke(
        &self,
        messages: &[BaseMessage],
        request: &ModelRequestConfig,
    ) -> Result<ChatResponse> {
        self.inner.invoke(messages, request).await
    }

    fn stream(
        &self,
        messages: Vec<BaseMessage>,
        request: ModelRequestConfig,
    ) -> BoxStream<'static, Result<StreamChunk>> {
        let inner = self.inner.clone();
        let entered = self.entered.clone();
        let release = self.release.clone();
        async_stream::stream! {
            entered.add_permits(1);
            let _permit = release.acquire().await;
            let mut frames = inner.stream(messages, request);
            while let Some(frame) = frames.next().await {
                yield frame;
            }
        }
        .boxed()
    }
}

/// A model whose every call suspends forever, signalling when a call is entered.
///
/// Proves that cancellation and the completion timeout end a turn that is parked inside the
/// model, instead of waiting for an answer that never comes.
struct HangModel {
    entered: Arc<Semaphore>,
}

#[async_trait]
impl Model for HangModel {
    fn provider(&self) -> &'static str {
        "hang"
    }

    fn model_name(&self) -> &'static str {
        "hang"
    }

    async fn invoke(
        &self,
        _messages: &[BaseMessage],
        _request: &ModelRequestConfig,
    ) -> Result<ChatResponse> {
        self.entered.add_permits(1);
        std::future::pending().await
    }

    fn stream(
        &self,
        _messages: Vec<BaseMessage>,
        _request: ModelRequestConfig,
    ) -> BoxStream<'static, Result<StreamChunk>> {
        let entered = self.entered.clone();
        async_stream::stream! {
            entered.add_permits(1);
            std::future::pending::<()>().await;
            yield Ok(StreamChunk::Done);
        }
        .boxed()
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

/// Builds a scripted-model agent running the outer task loop.
fn build_task_loop_agent(model: impl Model + 'static) -> Arc<DeepAgent> {
    let config = DeepAgentConfig::default()
        .with_language("en".to_string())
        .with_enable_task_loop(true);
    DeepAgentBuilder::new(AgentCard::new("loop-agent"))
        .config(config)
        .model(Arc::new(model))
        .build()
        .expect("agent builds")
}

/// Returns the system message of `call`, or an empty string when absent.
fn system_text(call: &openjiuwen_test_support::ModelCall) -> String {
    call.messages
        .first()
        .filter(|message| message.role == Role::System)
        .map(|message| message.text().into_owned())
        .unwrap_or_default()
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
    agent
        .configure(config.clone())
        .expect("the default configuration is valid");
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

#[tokio::test]
async fn task_loop_runs_a_follow_up_round_and_persists_the_iteration() {
    let model = ScriptedModel::new([
        ScriptedTurn::text("first answer"),
        ScriptedTurn::text("second answer"),
    ]);
    let agent = build_task_loop_agent(model.clone());
    agent.follow_up("keep going", None);

    let session = Arc::new(AgentSession::new("task-loop-session"));
    let output = agent
        .invoke(json!({"query": "start"}), Some(session))
        .await
        .expect("task loop invocation succeeds");

    assert_eq!(output.output, "second answer");
    assert_eq!(model.call_count(), 2);
    let state = agent.load_state().await.unwrap();
    assert_eq!(state.iteration, 2);
}

#[tokio::test]
async fn follow_ups_reach_the_model_and_the_state_round_trips() {
    let model = ScriptedModel::new([
        ScriptedTurn::text("first answer"),
        ScriptedTurn::text("second answer"),
        ScriptedTurn::text("third answer"),
    ]);
    let agent = build_task_loop_agent(model.clone());
    agent.follow_up("second", None);
    agent.follow_up("third", None);

    let session = Arc::new(AgentSession::new("follow-up-session"));
    let output = agent
        .invoke(json!({"query": "start"}), Some(session))
        .await
        .unwrap();
    assert_eq!(output.output, "third answer");

    let user_texts: Vec<String> = model
        .calls()
        .iter()
        .filter_map(openjiuwen_test_support::ModelCall::last_user_text)
        .collect();
    assert!(
        user_texts.iter().any(|text| text == "second"),
        "the follow-up text must reach the model: {user_texts:?}"
    );
    assert!(user_texts.iter().any(|text| text == "third"));

    // A queued follow-up and the plan-mode bookkeeping survive save/load.
    agent.switch_mode(AgentMode::Plan).unwrap();
    agent.follow_up("later", None);
    agent.save_state().await.unwrap();
    let loaded = agent.load_state().await.unwrap();
    assert_eq!(loaded.pending_follow_ups, vec!["later".to_string()]);
    assert_eq!(loaded.plan_mode.mode, "plan");
    assert_eq!(loaded.plan_mode.pre_plan_mode.as_deref(), Some("normal"));
    assert_eq!(loaded.iteration, 3);
}

#[tokio::test]
async fn steering_is_injected_into_the_next_model_call() {
    let model = ScriptedModel::text("steered answer");
    let agent = build_agent(model.clone());
    agent.steer("please focus");

    let output = agent
        .invoke(json!({"query": "hi"}), None)
        .await
        .expect("invocation succeeds");
    assert_eq!(output.output, "steered answer");

    let call = model.last_call().expect("the model was called");
    assert!(
        call.messages
            .iter()
            .any(|message| message.role == Role::User
                && message.text().as_ref() == "User steering: please focus"),
        "the steering message must reach the model window"
    );
    assert_eq!(agent.queues.steering_len(), 0);
}

#[tokio::test]
async fn a_rail_published_section_reaches_the_system_prompt_once() {
    let model = ScriptedModel::text("answered");
    let agent =
        DeepAgentBuilder::new(AgentCard::new("section-agent").with_description("Section identity"))
            .config(DeepAgentConfig::default().with_language("en".to_string()))
            .model(Arc::new(model.clone()))
            .rail(Arc::new(SectionRail {
                name: "custom_skill",
                section: PromptSection::bilingual(
                    "skill_demo",
                    "技能演示",
                    "Custom skill demo text",
                    50,
                ),
            }))
            .build()
            .unwrap();

    agent.invoke(json!({"query": "hi"}), None).await.unwrap();

    let system = system_text(&model.last_call().unwrap());
    assert_eq!(system.matches("Custom skill demo text").count(), 1);
    assert!(!system.contains("技能演示"));
}

#[tokio::test]
async fn a_published_identity_section_never_duplicates_the_static_one() {
    let model = ScriptedModel::text("answered");
    let agent =
        DeepAgentBuilder::new(AgentCard::new("section-agent").with_description("Section identity"))
            .config(DeepAgentConfig::default().with_language("en".to_string()))
            .model(Arc::new(model.clone()))
            .rail(Arc::new(SectionRail {
                name: "custom_identity",
                section: PromptSection::bilingual(
                    SectionName::IDENTITY,
                    "中文身份",
                    "Published identity override",
                    10,
                ),
            }))
            .build()
            .unwrap();

    agent.invoke(json!({"query": "hi"}), None).await.unwrap();

    let system = system_text(&model.last_call().unwrap());
    assert_eq!(system.matches("Section identity").count(), 1);
    assert!(!system.contains("Published identity override"));
    assert!(!system.contains("中文身份"));
}

#[tokio::test]
async fn stream_yields_frames_ending_with_the_answer() {
    let agent = build_agent(ScriptedModel::text("streamed answer"));
    let chunks: Vec<Result<openjiuwen_agent::AgentChunk>> =
        agent.stream(json!({"query": "hi"}), None).collect().await;

    assert!(!chunks.is_empty());
    assert!(chunks.iter().all(Result::is_ok), "no frame may fail");
    let last = chunks.last().unwrap().as_ref().unwrap();
    assert_eq!(last.kind, openjiuwen_agent::output::ANSWER);
    let output = AgentOutput::from_value(last.payload.clone()).unwrap();
    assert_eq!(output.output, "streamed answer");
}

#[tokio::test]
async fn a_second_invocation_is_rejected_while_one_is_running() {
    let entered = Arc::new(Semaphore::new(0));
    let release = Arc::new(Semaphore::new(0));
    let model = GateModel {
        inner: ScriptedModel::text("gated answer"),
        entered: entered.clone(),
        release: release.clone(),
    };
    let agent = Arc::new(
        DeepAgentBuilder::new(AgentCard::new("gate-agent"))
            .config(DeepAgentConfig::default().with_language("en".to_string()))
            .model(Arc::new(model))
            .build()
            .unwrap(),
    );

    let first = {
        let agent = agent.clone();
        tokio::spawn(async move { agent.invoke(json!({"query": "hi"}), None).await })
    };
    // Park the first invocation inside the model call.
    let permit = entered.acquire().await.unwrap();
    drop(permit);

    let error = agent
        .invoke(json!({"query": "again"}), None)
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("invocation already running"),
        "unexpected error: {error}"
    );

    let streamed: Vec<Result<openjiuwen_agent::AgentChunk>> = agent
        .stream(json!({"query": "again"}), None)
        .collect()
        .await;
    assert_eq!(streamed.len(), 1);
    assert!(streamed[0]
        .as_ref()
        .unwrap_err()
        .to_string()
        .contains("invocation already running"));

    release.add_permits(1);
    let output = first.await.unwrap().unwrap();
    assert_eq!(output.output, "gated answer");
}

/// Abort semantics: `abort` is synchronous and idempotent, a stopped agent
/// reports [`InteractionPhase::Terminated`], and the next invocation refreshes
/// the cancellation token and runs normally.
///
/// The in-round abort outcome (`RoundOutcome::abort()` after cancellation) is
/// covered by the `round.rs` tests and is not duplicated here.
#[tokio::test]
async fn abort_is_idempotent_and_a_later_invocation_still_runs() {
    let agent = build_agent(ScriptedModel::text("after abort"));
    agent.abort();
    agent.abort();
    assert_eq!(agent.phase(), InteractionPhase::Terminated);

    let output = agent
        .invoke(json!({"query": "hi"}), None)
        .await
        .expect("a fresh invocation succeeds after an abort");
    assert_eq!(output.output, "after abort");
    assert_ne!(agent.phase(), InteractionPhase::Terminated);
}

#[tokio::test]
async fn abort_interrupts_a_hung_model_promptly() {
    let entered = Arc::new(Semaphore::new(0));
    let agent = Arc::new(
        DeepAgentBuilder::new(AgentCard::new("hung-agent"))
            .config(DeepAgentConfig::default().with_language("en".to_string()))
            .model(Arc::new(HangModel {
                entered: entered.clone(),
            }))
            .build()
            .unwrap(),
    );
    let running = {
        let agent = agent.clone();
        tokio::spawn(async move { agent.invoke(json!({"query": "hang"}), None).await })
    };
    // Park the turn inside the model before aborting it.
    let permit = tokio::time::timeout(std::time::Duration::from_secs(10), entered.acquire())
        .await
        .expect("the model call must be entered")
        .unwrap();
    drop(permit);

    agent.abort();

    let error = tokio::time::timeout(std::time::Duration::from_secs(10), running)
        .await
        .expect("abort must end a hung turn")
        .unwrap()
        .expect_err("an aborted turn reports an error");
    assert!(
        error.to_string().contains("aborted"),
        "unexpected error: {error}"
    );
}

#[tokio::test]
async fn dropping_the_stream_cancels_the_run_and_frees_the_slot() {
    let entered = Arc::new(Semaphore::new(0));
    let release = Arc::new(Semaphore::new(0));
    let model = GateModel {
        inner: ScriptedModel::text("second answer"),
        entered: entered.clone(),
        release: release.clone(),
    };
    let agent = Arc::new(
        DeepAgentBuilder::new(AgentCard::new("dropped-stream-agent"))
            .config(DeepAgentConfig::default().with_language("en".to_string()))
            .model(Arc::new(model))
            .build()
            .unwrap(),
    );

    let stream = agent.stream(json!({"query": "stream"}), None);
    let permit = tokio::time::timeout(std::time::Duration::from_secs(10), entered.acquire())
        .await
        .expect("the streamed model call must be entered")
        .unwrap();
    drop(permit);
    drop(stream);
    // The gate only stays closed for the abandoned call; later calls (the retry) proceed.
    release.add_permits(16);

    // The cancelled run must release the single-run slot so a later invoke is accepted.
    let output = tokio::time::timeout(std::time::Duration::from_secs(10), async {
        loop {
            match agent.invoke(json!({"query": "again"}), None).await {
                Ok(output) => break output,
                Err(error) if error.to_string().contains("already running") => {
                    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                }
                Err(error) => panic!("unexpected error: {error}"),
            }
        }
    })
    .await
    .expect("a dropped stream must end its run");
    assert_eq!(output.output, "second answer");
}

#[tokio::test]
async fn the_single_round_path_honours_the_completion_timeout() {
    let entered = Arc::new(Semaphore::new(0));
    let agent = DeepAgentBuilder::new(AgentCard::new("timeout-agent"))
        .config(
            DeepAgentConfig::default()
                .with_language("en".to_string())
                .with_completion_timeout_secs(0.05),
        )
        .model(Arc::new(HangModel { entered }))
        .build()
        .unwrap();

    let error = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        agent.invoke(json!({"query": "hang"}), None),
    )
    .await
    .expect("the completion timeout must fire")
    .expect_err("a timed-out turn reports an error");
    assert!(
        error.to_string().contains("completion timeout"),
        "unexpected error: {error}"
    );
}

#[tokio::test]
async fn factory_wires_subagents_permissions_and_a_temp_workspace() {
    let temp = TempWorkspace::new().unwrap();
    let workspace = Workspace::for_language(temp.workspace(), "en");
    let mut permissions = PermissionsSection::default();
    permissions
        .tools
        .insert("echo".to_string(), PermissionLevel::Deny);
    let echo = Arc::new(EchoTool::new());
    let model = ScriptedModel::builder()
        .completion(MockCompletion::tool_calls(vec![MockToolCall::new(
            "echo",
            json!({"text": "x"}),
        )]))
        .respond_text("factory answer")
        .build();

    let agent = create_deep_agent(
        Arc::new(model),
        None,
        Some(
            DeepAgentConfig::default()
                .with_add_general_purpose_agent(true)
                // The persistent runtime surface is opt-in; only `task_tool` is mounted by default.
                .with_enable_subagent_runtime(true),
        ),
        vec![echo.clone() as Arc<dyn Tool>],
        Vec::new(),
        Some(workspace.clone()),
        Vec::new(),
        vec![SubAgentConfig::new(AgentCard::new("helper"))],
        true,
        None,
        Some("en".to_string()),
        None,
        true,
        Some(permissions),
        None,
    )
    .await
    .expect("the factory builds the agent");

    let mounted: Vec<String> = agent
        .tool_cards
        .read()
        .iter()
        .map(|card| card.name.clone())
        .collect();
    assert!(mounted.iter().any(|name| name == "task_tool"));
    for tool in [
        "subagent_spawn",
        "subagent_wait",
        "subagent_list",
        "subagent_send_input",
        "subagent_close",
        "subagent_resume",
    ] {
        assert!(
            mounted.iter().any(|name| name == tool),
            "missing mounted tool {tool}: {mounted:?}"
        );
    }
    assert_eq!(agent.config().workspace.root_path, workspace.root_path);
    assert!(agent.config().enable_task_loop);
    assert!(agent.rails().names().iter().any(|name| name == "subagent"));

    let session = Arc::new(AgentSession::new("factory-session"));
    let output = agent
        .invoke(json!({"query": "answer"}), Some(session))
        .await
        .unwrap();
    assert_eq!(output.output, "factory answer");
    assert_eq!(echo.calls(), 0, "the denied tool must never run");
}

/// Guards the stale abort latch: an idle abort leaves the stored controller
/// aborted, so the next task-loop invocation must not reuse it.
#[tokio::test]
async fn abort_while_idle_does_not_poison_the_next_task_loop_invocation() {
    let model = ScriptedModel::new([
        ScriptedTurn::text("first answer"),
        ScriptedTurn::text("second answer"),
    ]);
    let agent = build_task_loop_agent(model.clone());

    let first_session = Arc::new(AgentSession::new("abort-reuse-session"));
    let first = agent
        .invoke(json!({"query": "start"}), Some(first_session))
        .await
        .expect("the first task-loop invocation succeeds");
    assert_eq!(first.result_type, ResultType::Answer);
    assert_eq!(first.output, "first answer");

    agent.abort();

    let second_session = Arc::new(AgentSession::new("abort-reuse-session-2"));
    let second = agent
        .invoke(json!({"query": "again"}), Some(second_session))
        .await
        .expect("the second task-loop invocation succeeds");
    assert_eq!(second.result_type, ResultType::Answer);
    assert_eq!(second.output, "second answer");
    assert_ne!(agent.phase(), InteractionPhase::Terminated);
}

/// A `None` session is resolved by the agent, mirroring Python's
/// `owns_session`: the run creates and finishes its own session.
#[tokio::test]
async fn a_task_loop_invocation_without_a_session_creates_one() {
    let model = ScriptedModel::text("sessionless answer");
    let agent = build_task_loop_agent(model.clone());

    let output = agent
        .invoke(json!("hi"), None)
        .await
        .expect("a sessionless task-loop invocation succeeds");
    assert_eq!(output.result_type, ResultType::Answer);
    assert_eq!(output.output, "sessionless answer");
    assert_eq!(model.call_count(), 1);
    assert_ne!(agent.phase(), InteractionPhase::Terminated);
}

/// A rail that queues interaction events at the hooks the test selects.
///
/// Rails have no event sink, so the rail appends
/// [`InteractionEvent::to_output_value`] frames to an array in `ctx.extra`
/// under `channel`; the drain rail of the agent is what surfaces them.
struct QueueRail {
    name: &'static str,
    channel: String,
    before_task: Option<InteractionEvent>,
    after_task: Option<InteractionEvent>,
    after_invoke: Option<InteractionEvent>,
    scratch: bool,
}

impl QueueRail {
    fn new(name: &'static str, channel: &str) -> Self {
        Self {
            name,
            channel: channel.to_string(),
            before_task: None,
            after_task: None,
            after_invoke: None,
            scratch: false,
        }
    }

    fn before_task(mut self, event: InteractionEvent) -> Self {
        self.before_task = Some(event);
        self
    }

    fn after_task(mut self, event: InteractionEvent) -> Self {
        self.after_task = Some(event);
        self
    }

    fn after_invoke(mut self, event: InteractionEvent) -> Self {
        self.after_invoke = Some(event);
        self
    }

    fn with_scratch(mut self) -> Self {
        self.scratch = true;
        self
    }

    fn queue(&self, ctx: &mut HarnessRailContext, event: &InteractionEvent) {
        let channel = ctx
            .extra_mut()
            .entry(self.channel.clone())
            .or_insert_with(|| json!([]));
        if let Some(entries) = channel.as_array_mut() {
            entries.push(event.to_output_value());
        }
        if self.scratch {
            ctx.extra_mut()
                .insert("scratch".to_string(), json!({"keep": true}));
        }
    }
}

#[async_trait]
impl HarnessRail for QueueRail {
    fn name(&self) -> &str {
        self.name
    }

    fn priority(&self) -> i32 {
        crate::rails::rail_priority::DEFAULT
    }

    async fn before_task_iteration(&self, ctx: &mut HarnessRailContext) -> Result<()> {
        if let Some(event) = &self.before_task {
            self.queue(ctx, event);
        }
        Ok(())
    }

    async fn after_task_iteration(&self, ctx: &mut HarnessRailContext) -> Result<()> {
        if let Some(event) = &self.after_task {
            self.queue(ctx, event);
        }
        Ok(())
    }

    async fn after_invoke(&self, ctx: &mut HarnessRailContext) -> Result<()> {
        if let Some(event) = &self.after_invoke {
            self.queue(ctx, event);
        }
        Ok(())
    }
}

/// Builds a task-loop agent with `rails` mounted.
fn build_event_agent(
    model: impl Model + 'static,
    rails: Vec<Arc<dyn HarnessRail>>,
) -> Arc<DeepAgent> {
    let config = DeepAgentConfig::default()
        .with_language("en".to_string())
        .with_enable_task_loop(true);
    let mut builder = DeepAgentBuilder::new(AgentCard::new("event-agent"))
        .config(config)
        .model(Arc::new(model));
    for rail in rails {
        builder = builder.rail(rail);
    }
    builder.build().expect("agent builds")
}

/// Returns the payloads of the `__interaction__` frames of `chunks`, in order.
fn interaction_frames(chunks: &[Result<AgentChunk>]) -> Vec<Value> {
    chunks
        .iter()
        .filter_map(|chunk| {
            let chunk = chunk.as_ref().ok()?;
            (chunk.kind == openjiuwen_agent::output::INTERACTION).then(|| chunk.payload.clone())
        })
        .collect()
}

#[tokio::test]
async fn rail_events_of_both_task_iterations_reach_the_stream_in_order() {
    let rail = QueueRail::new("queue_goal", GOAL_EVENT_EXTRA_KEY)
        .before_task(InteractionEvent::goal_updated(Some(
            json!({"id": "before"}),
        )))
        .after_task(InteractionEvent::goal_updated(Some(json!({"id": "after"}))))
        .with_scratch();
    let agent = build_event_agent(ScriptedModel::text("answered"), vec![Arc::new(rail)]);

    let chunks: Vec<Result<AgentChunk>> =
        agent.stream(json!({"query": "hi"}), None).collect().await;
    let frames = interaction_frames(&chunks);
    assert_eq!(
        frames.len(),
        2,
        "both queued events must surface: {frames:?}"
    );
    assert_eq!(frames[0]["type"], json!("goal.updated"));
    assert_eq!(frames[0]["index"], json!(0));
    assert_eq!(frames[0]["payload"]["goal"], json!({"id": "before"}));
    assert_eq!(frames[1]["payload"]["goal"], json!({"id": "after"}));
    assert_eq!(
        chunks.last().unwrap().as_ref().unwrap().kind,
        openjiuwen_agent::output::ANSWER
    );

    // The same events are retained for a host that does not read the stream,
    // and each one is drained exactly once.
    let retained = agent.drain_rail_events();
    assert_eq!(retained.len(), 2);
    assert_eq!(retained[0].event_type, InteractionEventType::GoalUpdated);
    assert_eq!(agent.drain_rail_events().len(), 0);
}

#[tokio::test]
async fn a_terminal_goal_event_is_collected_once_by_drain_rail_events() {
    let rail = QueueRail::new("queue_goal", GOAL_EVENT_EXTRA_KEY).after_task(
        InteractionEvent::goal_updated(Some(json!({"id": "g-1", "status": "complete"}))),
    );
    let agent = build_event_agent(ScriptedModel::text("answered"), vec![Arc::new(rail)]);

    let output = agent
        .invoke(json!({"query": "hi"}), None)
        .await
        .expect("invocation succeeds");
    assert_eq!(output.result_type, ResultType::Answer);

    let events = agent.drain_rail_events();
    assert_eq!(
        events.len(),
        1,
        "the settled goal is reported once: {events:?}"
    );
    assert_eq!(events[0].event_type, InteractionEventType::GoalUpdated);
    assert_eq!(
        events[0].payload.get("goal"),
        Some(&json!({"id": "g-1", "status": "complete"}))
    );
    assert_eq!(agent.drain_rail_events().len(), 0);
}

#[tokio::test]
async fn a_rail_event_queued_in_after_invoke_is_drained() {
    // `after_invoke` is fired by the inner kernel agent through the harness
    // adapter chain, so this covers the kernel firing points as well.
    let rail = QueueRail::new("queue_after_invoke", GOAL_EVENT_EXTRA_KEY).after_invoke(
        InteractionEvent::goal_updated(Some(json!({"id": "kernel"}))),
    );
    let agent = DeepAgentBuilder::new(AgentCard::new("after-invoke-agent"))
        .config(DeepAgentConfig::default().with_language("en".to_string()))
        .model(Arc::new(ScriptedModel::text("answered")))
        .rail(Arc::new(rail))
        .build()
        .unwrap();

    agent.invoke(json!({"query": "hi"}), None).await.unwrap();

    let events = agent.drain_rail_events();
    assert_eq!(
        events.len(),
        1,
        "the after_invoke event must survive: {events:?}"
    );
    assert_eq!(
        events[0].payload.get("goal"),
        Some(&json!({"id": "kernel"}))
    );
}

#[tokio::test]
async fn several_rails_drain_in_a_deterministic_key_order() {
    let goal = QueueRail::new("queue_goal", GOAL_EVENT_EXTRA_KEY)
        .before_task(InteractionEvent::goal_updated(Some(json!({"id": "g"}))));
    let custom = QueueRail::new("queue_custom", &format!("{RAIL_EVENTS_EXTRA_PREFIX}alpha"))
        .before_task(InteractionEvent::execution_error("code-1", "first", None));
    let agent = build_event_agent(
        ScriptedModel::text("answered"),
        vec![Arc::new(goal), Arc::new(custom)],
    );

    let chunks: Vec<Result<AgentChunk>> =
        agent.stream(json!({"query": "hi"}), None).collect().await;
    let frames = interaction_frames(&chunks);
    assert_eq!(frames.len(), 2, "every rail event must drain: {frames:?}");
    // `harness.events.alpha` sorts before `harness.goal_events`.
    assert_eq!(frames[0]["type"], json!("execution.error"));
    assert_eq!(frames[0]["payload"]["code"], json!("code-1"));
    assert_eq!(frames[1]["type"], json!("goal.updated"));

    let retained: Vec<InteractionEventType> = agent
        .drain_rail_events()
        .iter()
        .map(|event| event.event_type)
        .collect();
    assert_eq!(
        retained,
        vec![
            InteractionEventType::ExecutionError,
            InteractionEventType::GoalUpdated
        ]
    );
}

#[tokio::test]
async fn a_round_without_queued_events_drains_nothing() {
    let agent = build_agent(ScriptedModel::text("plain"));
    agent.invoke(json!({"query": "hi"}), None).await.unwrap();
    assert_eq!(agent.drain_rail_events().len(), 0);

    let agent = build_agent(ScriptedModel::text("plain"));
    let chunks: Vec<Result<AgentChunk>> =
        agent.stream(json!({"query": "hi"}), None).collect().await;
    assert_eq!(interaction_frames(&chunks).len(), 0);
    assert_eq!(
        chunks.last().unwrap().as_ref().unwrap().kind,
        openjiuwen_agent::output::ANSWER
    );
    assert_eq!(agent.drain_rail_events().len(), 0);
}
