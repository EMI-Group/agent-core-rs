#![allow(clippy::unwrap_used, clippy::expect_used)]
//! End-to-end coverage of the rails seam inside the `ReAct` loop.
//!
//! The suite proves that rails are not merely dispatched but actually steer a
//! running agent: ordering follows priority for every hook, a force-finish on
//! `before_model_call` skips the model, a force-finish on `before_tool_call`
//! skips the tool, the model-backup rail retries with a working model, the
//! guardrail rail aborts on blocked user input, the memory rail recalls into the
//! prompt and writes the turn back, and a rail can replace a tool result.
//! Everything is offline and deterministic: models are `ScriptedModel`s, tools
//! are in-process `FunctionTool`s, and memory is an in-memory `LongTermMemory`.
//!
//! Helpers are local to this file on purpose so the suite stays self-contained.

use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_agent::{
    Agent, AgentCard, AgentOutput, AgentRail, GuardrailRail, MemoryRail, ModelBackupRail,
    RailContext, RailEvent, ReActAgent, ReActAgentConfig, ResultType,
};
use openjiuwen_common::error::{ErrorKind, Result as OJResult};
use openjiuwen_foundation::{BaseMessage, FunctionTool, Model, Role, Tool, ToolContext};
use openjiuwen_memory::{
    AddOptions, AgentMemoryConfig, LongTermMemory, MemoryEngineConfig, MemoryScope, SearchOptions,
};
use openjiuwen_security::{
    Guardrail, GuardrailContext, GuardrailPipeline, GuardrailResult, RiskLevel,
};
use openjiuwen_test_support::{MockCompletion, ScriptedError, ScriptedModel};
use parking_lot::Mutex;
use serde_json::{json, Value};

/// Every hook the ordering test subscribes its recording rails to.
const ORDERED_EVENTS: &[RailEvent] = &[
    RailEvent::BeforeInvoke,
    RailEvent::BeforeModelCall,
    RailEvent::BeforeToolCall,
    RailEvent::AfterInvoke,
];

/// Hands a clone of `model` to the agent while the test keeps the original for
/// inspecting the recorded calls.
fn shared(model: &ScriptedModel) -> Arc<dyn Model> {
    Arc::new(model.clone())
}

/// Builds a `ReActAgent` configured with `config` and driven by `model`.
fn agent(config: ReActAgentConfig, model: Arc<dyn Model>) -> ReActAgent {
    let agent = ReActAgent::new(AgentCard::new("rails-test"));
    agent.configure(config);
    agent.set_model(model);
    agent
}

/// Builds a tool named `name` that records its parsed inputs into `log` and
/// answers with the fixed text `"tool-output"`.
fn recording_tool(name: &str, log: &Arc<Mutex<Vec<Value>>>) -> Arc<dyn Tool> {
    let log = Arc::clone(log);
    let tool = FunctionTool::builder()
        .name(name)
        .description("recording tool")
        .input_schema(json!({"type": "object"}))
        .handler(move |inputs: Value, _ctx: ToolContext| {
            // `handler` requires an `Fn`, so the log is cloned per call rather
            // than moved out of the closure.
            let log = Arc::clone(&log);
            async move {
                log.lock().push(inputs);
                Ok(Value::String("tool-output".to_owned()))
            }
        })
        .build()
        .expect("the recording tool fixture builds");
    Arc::new(tool)
}

/// Returns a fresh log a [`recording_tool`] writes into.
fn log() -> Arc<Mutex<Vec<Value>>> {
    Arc::new(Mutex::new(Vec::new()))
}

/// A rail that records `"{name}:{event}"` for the events it subscribes to.
struct RecordingRail {
    /// Short distinct label of this rail (`"high"`, `"low"`).
    name: &'static str,
    /// Dispatch priority; higher runs first.
    priority: i32,
    /// Hooks this rail records.
    events: &'static [RailEvent],
    /// Flat invocation-wide log the test reads after the run.
    log: Arc<Mutex<Vec<String>>>,
}

impl RecordingRail {
    /// Builds a recording rail over `events`, writing into `log`.
    fn new(
        name: &'static str,
        priority: i32,
        events: &'static [RailEvent],
        log: Arc<Mutex<Vec<String>>>,
    ) -> Self {
        Self {
            name,
            priority,
            events,
            log,
        }
    }

    /// Records `event` under `label` when this rail subscribed to it.
    fn record(&self, event: RailEvent, label: &str) {
        if self.events.contains(&event) {
            self.log.lock().push(format!("{}:{label}", self.name));
        }
    }
}

#[async_trait]
impl AgentRail for RecordingRail {
    fn name(&self) -> &str {
        self.name
    }

    fn priority(&self) -> i32 {
        self.priority
    }

    async fn before_invoke(&self, _ctx: &mut RailContext) -> OJResult<()> {
        self.record(RailEvent::BeforeInvoke, "before_invoke");
        Ok(())
    }

    async fn before_model_call(&self, _ctx: &mut RailContext) -> OJResult<()> {
        self.record(RailEvent::BeforeModelCall, "before_model_call");
        Ok(())
    }

    async fn before_tool_call(&self, _ctx: &mut RailContext) -> OJResult<()> {
        self.record(RailEvent::BeforeToolCall, "before_tool_call");
        Ok(())
    }

    async fn after_invoke(&self, _ctx: &mut RailContext) -> OJResult<()> {
        self.record(RailEvent::AfterInvoke, "after_invoke");
        Ok(())
    }
}

/// A rail that force-finishes the run at one configured hook.
struct ForceFinishRail {
    /// Hook the rail reacts to; the other hook is a no-op.
    event: RailEvent,
    /// Terminal output handed to the loop.
    output: AgentOutput,
}

impl ForceFinishRail {
    /// Builds a force-finish rail for `event`.
    fn new(event: RailEvent, output: AgentOutput) -> Self {
        Self { event, output }
    }

    /// Finishes the run when `event` is the configured hook.
    fn maybe_finish(&self, event: RailEvent, ctx: &mut RailContext) {
        if self.event == event {
            ctx.request_force_finish(self.output.clone());
        }
    }
}

#[async_trait]
impl AgentRail for ForceFinishRail {
    fn name(&self) -> &'static str {
        "force-finish"
    }

    async fn before_model_call(&self, ctx: &mut RailContext) -> OJResult<()> {
        self.maybe_finish(RailEvent::BeforeModelCall, ctx);
        Ok(())
    }

    async fn before_tool_call(&self, ctx: &mut RailContext) -> OJResult<()> {
        self.maybe_finish(RailEvent::BeforeToolCall, ctx);
        Ok(())
    }
}

/// A rail that replaces every tool result with `"replacement"`.
struct SkipToolRail;

#[async_trait]
impl AgentRail for SkipToolRail {
    fn name(&self) -> &'static str {
        "skip-tool"
    }

    async fn before_tool_call(&self, ctx: &mut RailContext) -> OJResult<()> {
        ctx.skip_tool_with("replacement");
        Ok(())
    }
}

/// A guardrail that blocks everything as a critical finding.
struct LocalBlockingGuardrail;

#[async_trait]
impl Guardrail for LocalBlockingGuardrail {
    fn name(&self) -> &'static str {
        "local-block"
    }

    async fn check(&self, _ctx: &GuardrailContext) -> OJResult<GuardrailResult> {
        Ok(GuardrailResult::block(
            RiskLevel::Critical,
            "test_risk",
            None,
            None,
        ))
    }
}

/// Returns the position of the first entry equal to `needle`.
fn position_of(entries: &[String], needle: &str) -> usize {
    entries
        .iter()
        .position(|entry| entry == needle)
        .unwrap_or_else(|| panic!("missing rail log entry {needle:?} in {entries:?}"))
}

/// Returns the position of the first entry whose suffix is `event`.
fn first_event(entries: &[String], event: &str) -> usize {
    let suffix = format!(":{event}");
    entries
        .iter()
        .position(|entry| entry.ends_with(&suffix))
        .unwrap_or_else(|| panic!("missing event {event:?} in {entries:?}"))
}

/// Asserts that `high` precedes `low` for every ordered hook, and that the
/// hooks themselves fire in `before_invoke → before_model_call → before_tool_call
/// → after_invoke` order.
#[tokio::test]
async fn ordering_follows_priority_for_every_event() {
    let model = ScriptedModel::builder()
        .completion(MockCompletion::tool_call("echo", json!({})))
        .respond_text("done")
        .build();
    let entries = Arc::new(Mutex::new(Vec::new()));
    let tool_log = log();
    let agent = agent(ReActAgentConfig::new(), shared(&model));
    agent.register_rail(Arc::new(RecordingRail::new(
        "high",
        90,
        ORDERED_EVENTS,
        Arc::clone(&entries),
    )));
    agent.register_rail(Arc::new(RecordingRail::new(
        "low",
        10,
        ORDERED_EVENTS,
        Arc::clone(&entries),
    )));
    agent.add_tool(recording_tool("echo", &tool_log));

    let output = agent.invoke(json!("hi"), None).await.unwrap();

    assert_eq!(output.result_type, ResultType::Answer);
    assert_eq!(output.output, "done");
    let recorded = entries.lock().clone();
    for event in [
        "before_invoke",
        "before_model_call",
        "before_tool_call",
        "after_invoke",
    ] {
        let high = position_of(&recorded, &format!("high:{event}"));
        let low = position_of(&recorded, &format!("low:{event}"));
        assert!(
            high < low,
            "high must precede low for {event}: {recorded:?}"
        );
    }
    let first_before_invoke = first_event(&recorded, "before_invoke");
    let first_before_model = first_event(&recorded, "before_model_call");
    let first_before_tool = first_event(&recorded, "before_tool_call");
    let first_after_invoke = first_event(&recorded, "after_invoke");
    assert!(
        first_before_invoke < first_before_model
            && first_before_model < first_before_tool
            && first_before_tool < first_after_invoke,
        "hooks must fire in loop order: {recorded:?}"
    );
}

/// A `before_model_call` force-finish must return before the model is asked.
#[tokio::test]
async fn before_model_call_force_finish_skips_the_model() {
    let model = ScriptedModel::builder().respond_text("unexpected").build();
    let agent = agent(ReActAgentConfig::new(), shared(&model));
    agent.register_rail(Arc::new(ForceFinishRail::new(
        RailEvent::BeforeModelCall,
        AgentOutput::answer("short"),
    )));

    let output = agent.invoke(json!("hi"), None).await.unwrap();

    assert_eq!(model.call_count(), 0);
    assert_eq!(output.result_type, ResultType::Answer);
    assert_eq!(output.output, "short");
    // The loop's `with_progress` helper fills the iteration count in.
    assert_eq!(output.iterations_used, Some(1));
}

/// A `before_tool_call` force-finish must stop before any tool runs.
#[tokio::test]
async fn before_tool_call_force_finish_skips_the_tool() {
    let model = ScriptedModel::builder()
        .completion(MockCompletion::tool_call("echo", json!({})))
        .respond_text("unused")
        .build();
    let tool_log = log();
    let agent = agent(ReActAgentConfig::new(), shared(&model));
    agent.add_tool(recording_tool("echo", &tool_log));
    agent.register_rail(Arc::new(ForceFinishRail::new(
        RailEvent::BeforeToolCall,
        AgentOutput::answer("stopped"),
    )));

    let output = agent.invoke(json!("hi"), None).await.unwrap();

    assert!(tool_log.lock().is_empty());
    assert_eq!(model.call_count(), 1);
    assert_eq!(output.result_type, ResultType::Answer);
    assert_eq!(output.output, "stopped");
}

/// The model-backup rail retries once with the next model after an exception.
#[tokio::test]
async fn model_backup_rail_recovers_from_a_failing_primary() {
    let primary = ScriptedModel::builder()
        .fail(ScriptedError::llm("boom"))
        .build();
    let backup = ScriptedModel::builder().respond_text("from backup").build();
    let agent = agent(ReActAgentConfig::new(), shared(&primary));
    agent.register_rail(Arc::new(ModelBackupRail::new(vec![Arc::new(
        backup.clone(),
    )])));

    let output = agent.invoke(json!("hi"), None).await.unwrap();

    assert_eq!(output.output, "from backup");
    assert_eq!(output.result_type, ResultType::Answer);
    assert_eq!(primary.call_count(), 1);
    assert_eq!(backup.call_count(), 1);
}

/// A blocking guardrail aborts the run with a security error at `BeforeInvoke`.
#[tokio::test]
async fn guardrail_rail_blocks_user_input() {
    let model = ScriptedModel::builder().respond_text("unused").build();
    let agent = agent(ReActAgentConfig::new(), shared(&model));
    let mut pipeline = GuardrailPipeline::new();
    pipeline.add(LocalBlockingGuardrail);
    agent.register_rail(Arc::new(GuardrailRail::new(Arc::new(pipeline))));

    let error = agent.invoke(json!("hi"), None).await.unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Security);
    // The run aborted before the model was ever asked.
    assert_eq!(model.call_count(), 0);
}

/// The memory rail injects recalled memory into the prompt and writes the turn
/// back into the store.
#[tokio::test]
async fn memory_rail_recalls_and_writes_back() {
    let store = LongTermMemory::builder(MemoryEngineConfig::default())
        .build()
        .unwrap();
    let scope = MemoryScope::user("alice");
    store
        .add_messages(
            &scope,
            &[BaseMessage::user("the user's favourite editor is helix")],
            &AgentMemoryConfig::default(),
            &AddOptions::default(),
        )
        .await
        .unwrap();
    let query = "which editor does the user favour? favourite editor helix";
    // The seeded record is recallable with the rail's own search options; this
    // proves the recall half of the rail has candidates to inject.
    let seeded_hits = store
        .search(&scope, query, &SearchOptions::default())
        .await
        .unwrap();
    assert!(
        !seeded_hits.is_empty(),
        "the seeded memory must be recallable"
    );
    let model = ScriptedModel::builder().respond_text("helix it is").build();
    let agent = agent(ReActAgentConfig::new(), shared(&model));
    agent.register_rail(Arc::new(
        MemoryRail::new(Arc::clone(&store)).with_scope(MemoryScope::user("alice")),
    ));

    let output = agent.invoke(json!(query), None).await.unwrap();

    assert_eq!(output.result_type, ResultType::Answer);
    assert_eq!(output.output, "helix it is");
    // (b) The turn must be written back: a search finds a hit, and the recent
    // history holds the user query plus the assistant answer.
    let hits = store
        .search(&scope, query, &SearchOptions::default())
        .await
        .unwrap();
    assert!(!hits.is_empty(), "the write-back must be searchable");
    let recent = store.get_recent_messages(&scope, 10).await.unwrap();
    assert!(
        recent
            .iter()
            .any(|message| message.role == Role::User && message.text().contains(query)),
        "the user query must be stored: {recent:?}"
    );
    assert!(
        recent.iter().any(
            |message| message.role == Role::Assistant && message.text().contains("helix it is")
        ),
        "the assistant answer must be stored: {recent:?}"
    );
    // (a) The recall block must be visible to the model as a system message.
    // This depends on the loop passing post-rail `ctx.messages` to the model
    // (the parallel `agent_loop.rs` fix); the assertion is intentionally kept
    // strict because model visibility is the point of the rail.
    let system_texts: Vec<String> = model
        .calls()
        .iter()
        .flat_map(|call| call.messages.iter())
        .filter(|message| message.role == Role::System)
        .map(|message| message.text().into_owned())
        .collect();
    assert!(
        system_texts.iter().any(|text| text.contains("helix")),
        "the recall block must reach the model: {system_texts:?}"
    );
    assert!(
        system_texts
            .iter()
            .any(|text| text.contains("Long-term memory")),
        "the recall block must use the memory header: {system_texts:?}"
    );
}

/// A rail can replace a tool result on `before_tool_call`, and the replacement
/// is what the model sees on the next call.
#[tokio::test]
async fn a_rail_can_skip_a_tool_call() {
    let model = ScriptedModel::builder()
        .completion(MockCompletion::tool_call("echo", json!({})))
        .respond_text("done")
        .build();
    let tool_log = log();
    let agent = agent(ReActAgentConfig::new(), shared(&model));
    agent.add_tool(recording_tool("echo", &tool_log));
    agent.register_rail(Arc::new(SkipToolRail));

    let output = agent.invoke(json!("hi"), None).await.unwrap();

    assert!(tool_log.lock().is_empty());
    assert_eq!(output.result_type, ResultType::Answer);
    assert_eq!(output.output, "done");
    let calls = model.calls();
    assert_eq!(calls.len(), 2);
    let tool_texts: Vec<String> = calls[1]
        .messages
        .iter()
        .filter(|message| message.role == Role::Tool)
        .map(|message| message.text().into_owned())
        .collect();
    assert!(
        tool_texts.iter().any(|text| text.contains("replacement")),
        "the replacement must reach the model: {tool_texts:?}"
    );
}
