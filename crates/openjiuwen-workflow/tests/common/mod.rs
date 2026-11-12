//! Shared fixtures for the `openjiuwen-workflow` integration tests.
//!
//! Every suite in `tests/` includes this module (`mod common;`), so items some suites do
//! not use are allowed to be dead. Everything here is deterministic and offline: no
//! sleeps, no wall-clock reads, no network, and no stdout.
//!
//! The fakes follow the recipe the in-crate unit tests use (`src/engine/tests.rs`'s
//! `test_support`): a small `Component` implementation per behavior, built through a
//! constructor so a test reads as topology plus assertions.
#![allow(dead_code)]
#![allow(
    unreachable_pub,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::new_without_default,
    clippy::module_name_repetitions,
    clippy::unnecessary_literal_bound
)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::StreamExt;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::{FunctionTool, Model, Tool};
use openjiuwen_session::{Checkpointer, Session, WorkflowSession, WorkflowSessionOptions};
use parking_lot::Mutex;
use serde_json::{json, Value};

use openjiuwen_workflow::{
    Component, ComponentAbility, ComponentCard, ComponentContext, EndComponent, EndConfig,
    LlmCompConfig, LlmComponent, WorkflowCard,
};

/// A fresh in-process session with no checkpointer.
pub fn session(id: &str) -> Arc<dyn Session> {
    Arc::new(WorkflowSession::new(id))
}

/// A fresh in-process session that snapshots through `checkpointer`.
pub fn session_with_checkpointer(
    id: &str,
    checkpointer: Arc<dyn Checkpointer>,
) -> Arc<dyn Session> {
    Arc::new(WorkflowSession::with_options(WorkflowSessionOptions {
        session_id: Some(id.to_owned()),
        checkpointer: Some(checkpointer),
        ..WorkflowSessionOptions::default()
    }))
}

/// A workflow card whose id and name are both `id`.
pub fn card(id: &str) -> WorkflowCard {
    WorkflowCard::new(id)
}

/// Echoes its inputs under `echo`.
#[derive(Debug, Clone, Default)]
pub struct EchoComponent;

impl EchoComponent {
    /// Creates an echo component.
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Component for EchoComponent {
    fn card(&self) -> ComponentCard {
        ComponentCard::new("echo", "echo").with_abilities(self.abilities())
    }

    fn abilities(&self) -> Vec<ComponentAbility> {
        vec![ComponentAbility::Invoke]
    }

    fn component_type(&self) -> &str {
        "echo"
    }

    async fn invoke(&self, inputs: Value, _ctx: &ComponentContext) -> Result<Value> {
        Ok(json!({"echo": inputs}))
    }
}

/// Returns a caller-supplied value and records every input object it received.
#[derive(Debug, Clone)]
pub struct FixedComponent {
    output: Value,
    calls: Arc<Mutex<Vec<Value>>>,
}

impl FixedComponent {
    /// Creates a component that always answers `output`.
    pub fn new(output: Value) -> Self {
        Self {
            output,
            calls: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// The inputs of every invocation, in call order.
    pub fn calls(&self) -> Vec<Value> {
        self.calls.lock().clone()
    }

    /// How many times the component was invoked.
    pub fn call_count(&self) -> usize {
        self.calls.lock().len()
    }
}

#[async_trait]
impl Component for FixedComponent {
    fn card(&self) -> ComponentCard {
        ComponentCard::new("fixed", "fixed").with_abilities(self.abilities())
    }

    fn abilities(&self) -> Vec<ComponentAbility> {
        vec![ComponentAbility::Invoke]
    }

    fn component_type(&self) -> &str {
        "fixed"
    }

    async fn invoke(&self, inputs: Value, _ctx: &ComponentContext) -> Result<Value> {
        self.calls.lock().push(inputs);
        Ok(self.output.clone())
    }
}

/// Counts its invocations and answers `{"count": n}` (one-based).
#[derive(Debug, Clone, Default)]
pub struct CounterComponent {
    calls: Arc<AtomicUsize>,
}

impl CounterComponent {
    /// Creates a counter component at zero.
    pub fn new() -> Self {
        Self::default()
    }

    /// How many times the component ran.
    pub fn count(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl Component for CounterComponent {
    fn card(&self) -> ComponentCard {
        ComponentCard::new("counter", "counter").with_abilities(self.abilities())
    }

    fn abilities(&self) -> Vec<ComponentAbility> {
        vec![ComponentAbility::Invoke]
    }

    fn component_type(&self) -> &str {
        "counter"
    }

    async fn invoke(&self, _inputs: Value, _ctx: &ComponentContext) -> Result<Value> {
        let count = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        Ok(json!({"count": count}))
    }
}

/// A `Stream` producer that yields its configured frames verbatim.
#[derive(Debug, Clone, Default)]
pub struct StreamingProducer {
    frames: Vec<Value>,
}

impl StreamingProducer {
    /// Creates a producer yielding `frames` in order.
    pub fn new(frames: Vec<Value>) -> Self {
        Self { frames }
    }

    /// The frames this producer yields.
    pub fn frames(&self) -> &[Value] {
        &self.frames
    }
}

#[async_trait]
impl Component for StreamingProducer {
    fn card(&self) -> ComponentCard {
        ComponentCard::new("streaming", "streaming").with_abilities(self.abilities())
    }

    fn abilities(&self) -> Vec<ComponentAbility> {
        vec![ComponentAbility::Stream]
    }

    fn component_type(&self) -> &str {
        "streaming"
    }

    fn stream(
        self: Arc<Self>,
        _inputs: Value,
        _ctx: Arc<ComponentContext>,
    ) -> BoxStream<'static, Result<Value>> {
        let frames = self.frames.clone();
        futures::stream::iter(frames.into_iter().map(Ok)).boxed()
    }
}

/// A `Transform` producer that yields its configured frames verbatim.
#[derive(Debug, Clone, Default)]
pub struct TransformProducer {
    frames: Vec<Value>,
}

impl TransformProducer {
    /// Creates a transform yielding `frames` in order.
    pub fn new(frames: Vec<Value>) -> Self {
        Self { frames }
    }

    /// The frames this transform yields.
    pub fn frames(&self) -> &[Value] {
        &self.frames
    }
}

#[async_trait]
impl Component for TransformProducer {
    fn card(&self) -> ComponentCard {
        ComponentCard::new("transform", "transform").with_abilities(self.abilities())
    }

    fn abilities(&self) -> Vec<ComponentAbility> {
        vec![ComponentAbility::Transform]
    }

    fn component_type(&self) -> &str {
        "transform"
    }

    fn transform(
        self: Arc<Self>,
        _inputs: Value,
        _ctx: Arc<ComponentContext>,
    ) -> BoxStream<'static, Result<Value>> {
        let frames = self.frames.clone();
        futures::stream::iter(frames.into_iter().map(Ok)).boxed()
    }
}

/// A `Collect` consumer that wraps its resolved inputs under `collected`.
#[derive(Debug, Clone, Default)]
pub struct CollectorComponent;

impl CollectorComponent {
    /// Creates a collector component.
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Component for CollectorComponent {
    fn card(&self) -> ComponentCard {
        ComponentCard::new("collector", "collector").with_abilities(self.abilities())
    }

    fn abilities(&self) -> Vec<ComponentAbility> {
        vec![ComponentAbility::Collect]
    }

    fn component_type(&self) -> &str {
        "collector"
    }

    async fn collect(&self, inputs: Value, _ctx: &ComponentContext) -> Result<Value> {
        Ok(json!({"collected": inputs}))
    }
}

/// Asks a question through the component context (HITL) and returns the answer.
#[derive(Debug, Clone)]
pub struct InteractingComponent {
    question: Value,
}

impl InteractingComponent {
    /// Creates a component asking `question`.
    pub fn new(question: Value) -> Self {
        Self { question }
    }
}

#[async_trait]
impl Component for InteractingComponent {
    fn card(&self) -> ComponentCard {
        ComponentCard::new("interacting", "interacting").with_abilities(self.abilities())
    }

    fn abilities(&self) -> Vec<ComponentAbility> {
        vec![ComponentAbility::Invoke]
    }

    fn component_type(&self) -> &str {
        "interacting"
    }

    async fn invoke(&self, _inputs: Value, ctx: &ComponentContext) -> Result<Value> {
        let answer = ctx.interact(self.question.clone()).await?;
        Ok(json!({"answer": answer}))
    }
}

/// Fails its invocation with a coded workflow error.
#[derive(Debug, Clone)]
pub struct FailingComponent {
    reason: String,
}

impl FailingComponent {
    /// Creates a component failing with `reason`.
    pub fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
        }
    }
}

#[async_trait]
impl Component for FailingComponent {
    fn card(&self) -> ComponentCard {
        ComponentCard::new("failing", "failing").with_abilities(self.abilities())
    }

    fn abilities(&self) -> Vec<ComponentAbility> {
        vec![ComponentAbility::Invoke]
    }

    fn component_type(&self) -> &str {
        "failing"
    }

    async fn invoke(&self, _inputs: Value, _ctx: &ComponentContext) -> Result<Value> {
        Err(Error::workflow(self.reason.clone()))
    }
}

/// Waits until the run is cancelled, then reports [`Error::cancelled`].
#[derive(Debug, Clone, Default)]
pub struct WaitOnCancellationComponent;

impl WaitOnCancellationComponent {
    /// Creates a cancellation-blocking component.
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Component for WaitOnCancellationComponent {
    fn card(&self) -> ComponentCard {
        ComponentCard::new("wait_cancel", "wait_cancel").with_abilities(self.abilities())
    }

    fn abilities(&self) -> Vec<ComponentAbility> {
        vec![ComponentAbility::Invoke]
    }

    fn component_type(&self) -> &str {
        "wait_cancel"
    }

    async fn invoke(&self, _inputs: Value, ctx: &ComponentContext) -> Result<Value> {
        ctx.cancellation().cancelled().await;
        Err(Error::cancelled())
    }
}

/// A tool returning `{"b": 2, "got": inputs["a"]}`.
pub fn b_of_a_tool() -> Arc<dyn Tool> {
    FunctionTool::builder()
        .name("b_of_a")
        .description("returns b")
        .input_schema(json!({
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "required": ["a"]
        }))
        .handler(|inputs, _ctx| async move { Ok(json!({"b": 2, "got": inputs["a"]})) })
        .build()
        .expect("the b_of_a tool has a handler")
        .boxed()
}

/// An `llm` component using `model` with a user template `user`.
pub fn llm_component(model: Arc<dyn Model>, user: &str) -> LlmComponent {
    LlmComponent::new(LlmCompConfig {
        model: Some(model),
        user_prompt_template: Some(user.to_owned()),
        ..LlmCompConfig::default()
    })
    .expect("the llm component has a model")
}

/// An end component rendering `template` (`{{path}}` segments).
pub fn end_with_template(template: &str) -> EndComponent {
    EndComponent::with_config(EndConfig::new(template))
}

/// Joins the assistant text carried by `chunks`.
///
/// A chunk is either a bare string or an object whose `result` or `response` field is
/// the assistant text; anything else contributes nothing.
pub fn assistant_text(chunks: &[Value]) -> String {
    let mut text = String::new();
    for chunk in chunks {
        let piece = match chunk {
            Value::String(value) => Some(value.as_str()),
            Value::Object(map) => map
                .get("result")
                .or_else(|| map.get("response"))
                .and_then(Value::as_str),
            _ => None,
        };
        if let Some(piece) = piece {
            text.push_str(piece);
        }
    }
    text
}
