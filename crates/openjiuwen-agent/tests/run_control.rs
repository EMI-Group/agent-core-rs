#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Cancellation and deadline regression tests for the `ReAct` loop.
//!
//! A hung model stream or a tool that never returns used to block a turn
//! forever, and dropping the returned stream never stopped the run. These
//! suites pin the two escape hatches: the run's own deadline, and the
//! cancellation token that `abort()` (or dropping the stream) trips.
mod common;
use async_trait::async_trait;
use futures::stream::{self, BoxStream, StreamExt};
use openjiuwen_agent::{Agent, ReActAgentConfig, ResultType};
use openjiuwen_common::error::{Error, ErrorKind, Result};
use openjiuwen_foundation::{
    BaseMessage, ChatResponse, Model, ModelRequestConfig, StreamChunk, Tool, ToolCard, ToolContext,
};
use openjiuwen_test_support::{MockCompletion, ScriptedModel};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
/// A model whose stream accepts the request and then never yields.
struct HangingModel {
    calls: Arc<AtomicUsize>,
}
#[async_trait]
impl Model for HangingModel {
    fn provider(&self) -> &'static str {
        "fake"
    }
    fn model_name(&self) -> &'static str {
        "hanging"
    }
    async fn invoke(
        &self,
        _messages: &[BaseMessage],
        _request: &ModelRequestConfig,
    ) -> Result<ChatResponse> {
        Err(Error::validation("this fake only streams"))
    }
    fn stream(
        &self,
        _messages: Vec<BaseMessage>,
        _request: ModelRequestConfig,
    ) -> BoxStream<'static, Result<StreamChunk>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Box::pin(stream::pending())
    }
}
/// A tool that never returns.
struct HangingTool {
    card: ToolCard,
}
#[async_trait]
impl Tool for HangingTool {
    fn card(&self) -> &ToolCard {
        &self.card
    }
    async fn invoke(&self, _args: Value, _ctx: &ToolContext) -> Result<Value> {
        futures::future::pending::<()>().await;
        Ok(json!(null))
    }
}
/// A model that always asks for `hang`, so the loop reaches the tool.
fn tool_calling_model() -> ScriptedModel {
    ScriptedModel::builder()
        .completion(MockCompletion::tool_call("hang", json!({})))
        .respond_text("done")
        .build()
}
#[tokio::test]
async fn a_hung_model_stream_ends_the_run_on_the_deadline() {
    let calls = Arc::new(AtomicUsize::new(0));
    let model: Arc<dyn Model> = Arc::new(HangingModel {
        calls: Arc::clone(&calls),
    });
    let mut config = ReActAgentConfig::new();
    config.completion_timeout_secs = 0.05;
    let agent = common::agent(config, model);
    let error = tokio::time::timeout(
        Duration::from_secs(5),
        agent.invoke(json!("hi"), Some(common::session("hang"))),
    )
    .await
    .expect("the run must not hang")
    .expect_err("the deadline must fail the run");
    assert_eq!(error.kind(), ErrorKind::Timeout, "{error}");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}
#[tokio::test]
async fn abort_cancels_a_run_stuck_on_the_model_stream() {
    let calls = Arc::new(AtomicUsize::new(0));
    let model: Arc<dyn Model> = Arc::new(HangingModel {
        calls: Arc::clone(&calls),
    });
    let mut config = ReActAgentConfig::new();
    // No deadline: only the abort can end this run.
    config.completion_timeout_secs = 0.0;
    let agent = Arc::new(common::agent(config, model));
    let running = Arc::clone(&agent);
    let handle = tokio::spawn(async move {
        running
            .invoke(json!("hi"), Some(common::session("abort")))
            .await
    });
    // Give the loop time to enter the model stream before aborting it.
    while calls.load(Ordering::SeqCst) == 0 {
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    agent.abort();
    let error = tokio::time::timeout(Duration::from_secs(5), handle)
        .await
        .expect("the aborted run must end")
        .expect("the task must not panic")
        .expect_err("an aborted run reports cancellation");
    assert_eq!(error.kind(), ErrorKind::Cancelled, "{error}");
}
#[tokio::test]
async fn a_hung_tool_ends_its_call_with_a_timeout_result() {
    let model = tool_calling_model();
    let mut config = ReActAgentConfig::new();
    config.tool_timeout_secs = 0.05;
    config.completion_timeout_secs = 0.0;
    let agent = common::agent(config, Arc::new(model.clone()));
    agent.add_tool(Arc::new(HangingTool {
        card: ToolCard::new("hang", "blocks forever", json!({"type": "object"})),
    }));
    let output = tokio::time::timeout(
        Duration::from_secs(5),
        agent.invoke(json!("hi"), Some(common::session("tool"))),
    )
    .await
    .expect("a hung tool must not hang the turn")
    .expect("the turn still answers after the tool timeout");
    assert_eq!(output.result_type, ResultType::Answer);
    let observation = output
        .observations
        .first()
        .expect("the tool timeout is reported as an observation");
    assert!(
        observation.content.contains("timed out after"),
        "{}",
        observation.content
    );
    // The model ran twice: the tool call, then the answer after the failure.
    assert_eq!(model.call_count(), 2);
}
#[tokio::test]
async fn dropping_the_stream_stops_the_run() {
    let model = ScriptedModel::builder().stream_text("hello").build();
    let agent = common::agent(ReActAgentConfig::new(), Arc::new(model.clone()));
    {
        let mut frames = agent.stream(json!("hi"), Some(common::session("drop")));
        // Consume one frame, then drop the stream mid-run.
        let first = frames.next().await.expect("a first frame").unwrap();
        assert_ne!(first.kind, "");
    }
    // The dropped consumer cancelled the run before it could call the model
    // again for a second iteration.
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(model.call_count() <= 1, "{}", model.call_count());
}
