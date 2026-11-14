#![allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
//! Usage/telemetry seam tests for the implemented `ReAct` loop.
//!
//! `openjiuwen-context` measures; the agent only connects the measurement to
//! the loop. These tests pin that wiring against a `ScriptedModel`:
//!
//! - with a `ContextUsageAnalyzer` installed, the pre-call `context.usage` frame
//!   carries the measured block, and every model call publishes a post-call
//!   frame (provider token counts plus its own snapshot) that the session
//!   mirror also receives;
//! - a non-streaming `invoke` buffers the same snapshots for
//!   `last_usage`/`drain_usage_events`;
//! - the provider report lands in the engine's KV-cache rollup under the scope
//!   the seam's helper builds;
//! - without an analyzer nothing is measured: no post-call frame, no buffered
//!   snapshot, and `usage_report` returns `Error::Config`;
//! - `usage_report(session, Some(n))` reconciles the window total to `n`, and a
//!   later `configure` keeps an installed analyzer.
//!
//! Everything is offline and deterministic: models are scripted, sessions are
//! in-process, and only the session-mirror read uses a bounded (500 ms) wait.
mod common;
use futures::StreamExt;
use openjiuwen_agent::output::{ANSWER, CONTEXT_USAGE};
use openjiuwen_agent::usage::cache_scope_key;
use openjiuwen_agent::{default_usage_analyzer, Agent, AgentChunk, ReActAgent, ReActAgentConfig};
use openjiuwen_common::error::ErrorKind;
use openjiuwen_context::ContextProcessorStateRecorder;
use openjiuwen_foundation::{Model, ModelRequestConfig, Usage};
use openjiuwen_session::{Session, StreamEvent};
use openjiuwen_test_support::{MockCompletion, ScriptedModel};
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;

/// The provider the scripted model reports; the KV-cache scope uses it.
const PROVIDER: &str = "openai";
/// The model name the agent's configuration asks for.
const MODEL_NAME: &str = "gpt-4o";

/// Hands a clone of `model` to the agent while the test keeps the original.
fn shared(model: &ScriptedModel) -> Arc<dyn Model> {
    Arc::new(model.clone())
}

/// Builds an agent whose request configuration names [`MODEL_NAME`].
fn agent(model: Arc<dyn Model>) -> ReActAgent {
    let mut config = ReActAgentConfig::new();
    config.configure_model(MODEL_NAME, PROVIDER);
    common::agent(config, model)
}

/// Installs the default (approximate) analyzer on `agent`.
fn install_analyzer(agent: &ReActAgent) {
    agent.set_usage_analyzer(Some(Arc::new(default_usage_analyzer())));
}

/// Runs `agent` to completion, returning every frame it published.
async fn collect_frames(agent: &ReActAgent, session: Arc<dyn Session>) -> Vec<AgentChunk> {
    agent
        .stream(json!("hi"), Some(session))
        .map(|frame| frame.unwrap())
        .collect()
        .await
}

/// Drains the session mirror until the terminal `answer` frame.
async fn mirrored_events<S>(mut events: S) -> Vec<openjiuwen_session::StreamItem>
where
    S: futures::Stream<Item = openjiuwen_common::error::Result<StreamEvent>> + Unpin,
{
    let mut items = Vec::new();
    let deadline = tokio::time::Instant::now() + Duration::from_millis(500);
    loop {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            break;
        }
        match tokio::time::timeout(remaining, events.next()).await {
            Ok(Some(Ok(StreamEvent::Item(item)))) => {
                let terminal = item.item_type == ANSWER;
                items.push(item);
                if terminal {
                    break;
                }
            }
            Ok(Some(Ok(StreamEvent::End)) | None) | Err(_) => break,
            Ok(Some(Err(error))) => panic!("session stream failed: {error}"),
        }
    }
    items
}

#[tokio::test]
async fn analyzer_installed_publishes_pre_and_post_call_usage_frames() {
    let model = ScriptedModel::builder()
        .provider(PROVIDER)
        .completion(MockCompletion::text("hello").with_usage(Usage::new(11, 5)))
        .build();
    let agent = agent(shared(&model));
    install_analyzer(&agent);
    let session = common::session(common::SESSION_ID);

    let frames = collect_frames(&agent, Arc::clone(&session)).await;
    let usage_frames = frames
        .iter()
        .filter(|frame| frame.kind == CONTEXT_USAGE)
        .collect::<Vec<_>>();
    assert_eq!(usage_frames.len(), 2, "{usage_frames:?}");

    // The pre-call frame keeps every window key and gains the measured block.
    let pre_call = &usage_frames[0];
    assert_eq!(pre_call.payload["phase"], "pre_call");
    assert_eq!(pre_call.payload["iteration"], 1);
    assert!(pre_call.payload["message_count"].is_number());
    assert!(pre_call.payload["tool_count"].is_number());
    assert!(!pre_call.payload["usage"].is_null(), "{pre_call:?}");
    assert_eq!(pre_call.payload["usage"]["type"], "context.usage");

    // The post-call frame reports the provider's accounting and its snapshot.
    let post_call = &usage_frames[1];
    assert_eq!(post_call.payload["phase"], "post_call");
    assert_eq!(post_call.payload["iteration"], 1);
    assert_eq!(post_call.payload["input_tokens"], 11);
    assert_eq!(post_call.payload["output_tokens"], 5);
    assert_eq!(post_call.payload["total_tokens"], 16);
    assert_eq!(post_call.payload["tokens"], 11);
    assert!(!post_call.payload["usage"].is_null(), "{post_call:?}");
    assert_eq!(post_call.payload["result_type"], "answer");
}

#[tokio::test]
async fn usage_frames_are_mirrored_into_the_session() {
    let model = ScriptedModel::builder()
        .provider(PROVIDER)
        .completion(MockCompletion::text("hello").with_usage(Usage::new(7, 2)))
        .build();
    let agent = agent(shared(&model));
    install_analyzer(&agent);
    let session = common::session("mirror");
    // Subscribe before the run so no frame is produced before the subscriber.
    let events = session.subscribe();
    collect_frames(&agent, Arc::clone(&session)).await;

    let items = mirrored_events(events).await;
    let post_call = items
        .iter()
        .find(|item| item.item_type == CONTEXT_USAGE && item.payload["phase"] == "post_call")
        .expect("the session mirror receives the post-call frame");
    assert_eq!(post_call.payload["input_tokens"], 7);
    assert_eq!(post_call.payload["output_tokens"], 2);
    assert!(!post_call.payload["usage"].is_null());
    assert!(
        items.iter().any(|item| item.item_type == ANSWER),
        "{:?}",
        items
            .iter()
            .map(|item| item.item_type.clone())
            .collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn a_plain_invoke_buffers_one_snapshot_per_model_call() {
    let model = ScriptedModel::builder()
        .provider(PROVIDER)
        .completion(MockCompletion::tool_call("echo", json!({"value": "hi"})))
        .respond_text("final answer")
        .build();
    let agent = agent(shared(&model));
    install_analyzer(&agent);
    agent.add_tool(common::recording_tool("echo", &common::log()));

    // A non-streaming invoke has no publisher, so nothing is framed.
    let output = agent
        .invoke(json!("hi"), Some(common::session(common::SESSION_ID)))
        .await
        .unwrap();
    assert_eq!(output.output, "final answer");
    assert_eq!(model.call_count(), 2);

    let last = agent.last_usage().expect("a snapshot was buffered");
    assert_eq!(last.phase, "post_call");
    assert_eq!(last.request_id, format!("{}#2", common::SESSION_ID));

    let drained = agent.drain_usage_events();
    assert_eq!(drained.len(), 2);
    assert_eq!(drained[0].request_id, format!("{}#1", common::SESSION_ID));
    assert_eq!(drained[1].request_id, format!("{}#2", common::SESSION_ID));
    // Draining empties the buffer without losing the count.
    assert!(agent.last_usage().is_none());
    assert!(agent.drain_usage_events().is_empty());
    assert_eq!(agent.usage_telemetry().recorded_total(), 2);
}

#[tokio::test]
async fn provider_usage_reaches_the_kv_cache_rollup() {
    let model = ScriptedModel::builder()
        .provider(PROVIDER)
        .completion(MockCompletion::text("hello").with_usage(Usage::new(9, 3)))
        .build();
    let agent = agent(shared(&model));
    install_analyzer(&agent);
    let session = common::session(common::SESSION_ID);
    agent
        .invoke(json!("hi"), Some(Arc::clone(&session)))
        .await
        .unwrap();

    let scope = cache_scope_key(session.id(), PROVIDER, &ModelRequestConfig::new(MODEL_NAME));
    let rollup = agent
        .context_engine()
        .kv_cache_usage(&scope)
        .expect("the run recorded its provider usage under the scope");
    assert_eq!(rollup.calls_total, 1, "{rollup:?}");

    // A different model name is a different scope, so nothing is recorded there.
    let other = cache_scope_key(
        session.id(),
        PROVIDER,
        &ModelRequestConfig::new("other-model"),
    );
    assert!(agent.context_engine().kv_cache_usage(&other).is_none());
}

#[tokio::test]
async fn no_analyzer_means_no_post_call_frame_and_a_config_error() {
    let model = ScriptedModel::builder()
        .provider(PROVIDER)
        .completion(MockCompletion::text("hello").with_usage(Usage::new(4, 1)))
        .build();
    let agent = agent(shared(&model));
    let session = common::session(common::SESSION_ID);

    let frames = collect_frames(&agent, Arc::clone(&session)).await;
    let usage_frames = frames
        .iter()
        .filter(|frame| frame.kind == CONTEXT_USAGE)
        .collect::<Vec<_>>();
    assert_eq!(usage_frames.len(), 1, "{usage_frames:?}");
    assert_eq!(usage_frames[0].payload["phase"], "pre_call");
    assert!(usage_frames[0].payload["usage"].is_null());

    // Nothing was measured, so nothing was buffered and the run still answered.
    assert_eq!(frames.last().unwrap().payload["result_type"], "answer");
    assert!(agent.last_usage().is_none());
    assert!(agent.drain_usage_events().is_empty());

    let error = agent.usage_report(&session, None).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Config, "{error}");
}

#[tokio::test]
async fn usage_report_reconciles_the_supplied_provider_total() {
    let model = ScriptedModel::builder()
        .provider(PROVIDER)
        .completion(MockCompletion::text("hello").with_usage(Usage::new(11, 5)))
        .build();
    let agent = agent(shared(&model));
    install_analyzer(&agent);
    let session = common::session(common::SESSION_ID);
    agent
        .invoke(json!("hi"), Some(Arc::clone(&session)))
        .await
        .unwrap();

    // Without a provider total the report is the local measurement.
    let local = agent.usage_report(&session, None).await.unwrap();
    assert!(local.context_window.tokens > 0);
    assert_eq!(local.context_id, "main");

    // With one, it is authoritative for the window total.
    let reconciled = agent.usage_report(&session, Some(1234)).await.unwrap();
    assert_eq!(reconciled.context_window.tokens, 1234);
    assert_eq!(reconciled.session_id, session.id());
}

#[tokio::test]
async fn configure_keeps_an_installed_analyzer_and_recorder() {
    let model = ScriptedModel::builder()
        .provider(PROVIDER)
        .completion(MockCompletion::text("hello").with_usage(Usage::new(6, 2)))
        .build();
    let agent = agent(shared(&model));
    install_analyzer(&agent);
    agent.set_context_recorder(Some(Arc::new(ContextProcessorStateRecorder::new(
        common::SESSION_ID,
        "main",
    ))));
    assert!(agent.usage_analyzer().is_some());
    assert!(agent.context_recorder().is_some());

    // A later `configure` rebuilds the engine but keeps the instrumentation.
    let mut config = ReActAgentConfig::new();
    config.configure_model(MODEL_NAME, PROVIDER);
    agent.configure(config);
    assert!(agent.usage_analyzer().is_some());
    assert!(agent.context_recorder().is_some());

    agent
        .invoke(json!("hi"), Some(common::session(common::SESSION_ID)))
        .await
        .unwrap();
    assert!(
        agent.last_usage().is_some(),
        "the rebuilt engine measured the run"
    );

    // Removing both rebuilds the engine without them again.
    agent.set_usage_analyzer(None);
    agent.set_context_recorder(None);
    assert!(agent.usage_analyzer().is_none());
    assert!(agent.context_recorder().is_none());
    let error = agent
        .usage_report(&common::session(common::SESSION_ID), None)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Config, "{error}");
}
