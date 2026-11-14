#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Public-API tests for the compression telemetry the processor pipeline emits.
//!
//! Every processor pass of a pipeline that carries a
//! [`ContextProcessorStateRecorder`] is bracketed by a `started` record and the
//! record of its outcome (`completed`, `noop`, `skipped`, `failed`,
//! `cancelled`), and the processor runs inside a compaction-operation scope, so
//! the model calls it makes are attributable. The suite pins the per-processor
//! status sequence and its content, the failure and cancellation paths, the
//! recorder's history cap and its delivery tolerance, the operation id a model
//! call observes, the correlation the shared summariser stamps onto its request,
//! the usage it reports back, and the completely uninstrumented path.
//!
//! Everything is offline and deterministic: the counter is
//! [`ApproxTokenCounter::with_chars_per_token`] at one character per token,
//! offload storage is in memory, models are the fake [`AmbientModel`] and
//! [`openjiuwen_test_support::ScriptedModel`], and no test reads a wall clock
//! (durations are only asserted to be present).
//!
//! [`ContextProcessorStateRecorder`]:
//!     openjiuwen_context::telemetry::ContextProcessorStateRecorder
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::stream::BoxStream;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_context::processor::summarize;
use openjiuwen_context::telemetry::{
    self, CompressionStatePhase, CompressionStateStatus, ContextProcessorStateRecorder,
    RecordingSink,
};
use openjiuwen_context::{
    compute_stats, stamp_message_id, ContextConfig, ContextEvent, ContextEventKind,
    ContextPipeline, ContextProcessor, InMemoryOffloadStore, OffloadScope, PipelineDeps,
    ProcessorConfig, ProcessorContext, TruncationConfig,
};
use openjiuwen_foundation::llm::{ChatResponse, Choice, Model, ModelRequestConfig, StreamChunk};
use openjiuwen_foundation::prompt::{
    ApproxTokenCounter, BaseMessage, MessageContent, TokenCounter, Usage,
};
use openjiuwen_test_support::ScriptedModel;
use tokio_util::sync::CancellationToken;

/// The one-character-per-token counter every test uses.
fn counter() -> Arc<ApproxTokenCounter> {
    Arc::new(ApproxTokenCounter::with_chars_per_token(1))
}

/// The state a [`ProcessorContext`] borrows from, owned per test.
struct Fixture {
    config: ContextConfig,
    counter: Arc<ApproxTokenCounter>,
    offload: InMemoryOffloadStore,
    scope: OffloadScope,
    cancellation: CancellationToken,
    model: Option<Arc<dyn Model>>,
}

impl Fixture {
    /// A fixture with a 1 000-token budget and no model.
    fn new() -> Self {
        Self {
            config: ContextConfig::default(),
            counter: counter(),
            offload: InMemoryOffloadStore::new(),
            scope: OffloadScope::new("session", "main"),
            cancellation: CancellationToken::new(),
            model: None,
        }
    }

    /// A fixture whose processors can call `model`.
    fn with_model(mut self, model: Arc<dyn Model>) -> Self {
        self.model = Some(model);
        self
    }

    /// A processor context borrowing from the fixture.
    fn context(&self) -> ProcessorContext<'_> {
        ProcessorContext {
            config: &self.config,
            counter: self.counter.as_ref(),
            model: self.model.as_ref(),
            offload: &self.offload,
            scope: &self.scope,
            budget: 1_000,
            trigger_tokens: 800,
            manual: false,
            cancellation: &self.cancellation,
            recall: None,
            session_memory: None,
        }
    }
}

/// A recorder whose records are also collected by the returned sink.
fn recorder(
    counter: &Arc<ApproxTokenCounter>,
) -> (Arc<ContextProcessorStateRecorder>, Arc<RecordingSink>) {
    let sink = Arc::new(RecordingSink::new());
    let recorder = Arc::new(
        ContextProcessorStateRecorder::new("session", "main")
            .with_sink(sink.clone())
            .with_token_counter(counter.clone()),
    );
    (recorder, sink)
}

/// Rewrites the first message in place and reports the change.
struct Rewriter;

#[async_trait]
impl ContextProcessor for Rewriter {
    fn name(&self) -> &'static str {
        "rewriter"
    }

    async fn process(
        &self,
        messages: &mut Vec<BaseMessage>,
        _cx: &ProcessorContext<'_>,
    ) -> Result<Option<ContextEvent>> {
        if let Some(message) = messages.first_mut() {
            message.content = MessageContent::Text("short".to_owned());
        }
        Ok(Some(ContextEvent {
            processor: "rewriter".to_owned(),
            kind: ContextEventKind::ToolResultTrimmed,
            messages_changed: vec![0],
            summary: Some("rewrote the first message".to_owned()),
            ..ContextEvent::default()
        }))
    }
}

/// Runs and changes nothing.
struct Noop;

#[async_trait]
impl ContextProcessor for Noop {
    fn name(&self) -> &'static str {
        "noop"
    }

    async fn process(
        &self,
        _messages: &mut Vec<BaseMessage>,
        _cx: &ProcessorContext<'_>,
    ) -> Result<Option<ContextEvent>> {
        Ok(None)
    }
}

/// Fails with a provider error.
struct Failing;

#[async_trait]
impl ContextProcessor for Failing {
    fn name(&self) -> &'static str {
        "failing"
    }

    async fn process(
        &self,
        _messages: &mut Vec<BaseMessage>,
        _cx: &ProcessorContext<'_>,
    ) -> Result<Option<ContextEvent>> {
        Err(Error::Llm("boom".to_owned()))
    }
}

/// Fails because the run was cancelled.
struct Cancelling;

#[async_trait]
impl ContextProcessor for Cancelling {
    fn name(&self) -> &'static str {
        "cancelling"
    }

    async fn process(
        &self,
        _messages: &mut Vec<BaseMessage>,
        _cx: &ProcessorContext<'_>,
    ) -> Result<Option<ContextEvent>> {
        Err(Error::Cancelled)
    }
}

/// Counts its invocations and records the operation id it observed.
struct Observer {
    runs: Arc<AtomicUsize>,
    seen: Arc<Mutex<Vec<Option<String>>>>,
}

#[async_trait]
impl ContextProcessor for Observer {
    fn name(&self) -> &'static str {
        "observer"
    }

    async fn process(
        &self,
        _messages: &mut Vec<BaseMessage>,
        _cx: &ProcessorContext<'_>,
    ) -> Result<Option<ContextEvent>> {
        self.runs.fetch_add(1, Ordering::SeqCst);
        self.seen
            .lock()
            .unwrap()
            .push(telemetry::current_compression_operation_id());
        Ok(None)
    }
}

/// Summarises with the context model and rewrites the first message.
struct Summarizer;

#[async_trait]
impl ContextProcessor for Summarizer {
    fn name(&self) -> &'static str {
        "summarizer"
    }

    async fn process(
        &self,
        messages: &mut Vec<BaseMessage>,
        cx: &ProcessorContext<'_>,
    ) -> Result<Option<ContextEvent>> {
        let summary = summarize(cx.model, "system", "user", 32, || "offline".to_owned()).await;
        if let Some(message) = messages.first_mut() {
            message.content = MessageContent::Text(summary.clone());
        }
        Ok(Some(ContextEvent {
            processor: "summarizer".to_owned(),
            kind: ContextEventKind::Summarized,
            summary: Some(summary),
            ..ContextEvent::default()
        }))
    }
}

/// A [`Model`] that answers immediately, reports `usage`, and records the
/// compaction operation id it ran under.
struct AmbientModel {
    usage: Usage,
    seen: Mutex<Vec<Option<String>>>,
}

impl AmbientModel {
    /// A model reporting a 7/3-token usage.
    fn new() -> Self {
        Self {
            usage: Usage::new(7, 3),
            seen: Mutex::new(Vec::new()),
        }
    }

    /// A model that reports no usage at all.
    fn without_usage() -> Self {
        Self {
            usage: Usage::default(),
            seen: Mutex::new(Vec::new()),
        }
    }

    /// The operation ids observed so far, one per call.
    fn seen(&self) -> Vec<Option<String>> {
        self.seen.lock().unwrap().clone()
    }
}

#[async_trait]
impl Model for AmbientModel {
    fn provider(&self) -> &'static str {
        "ambient"
    }

    fn model_name(&self) -> &'static str {
        "ambient-model"
    }

    async fn invoke(
        &self,
        _messages: &[BaseMessage],
        _request: &ModelRequestConfig,
    ) -> Result<ChatResponse> {
        self.seen
            .lock()
            .unwrap()
            .push(telemetry::current_compression_operation_id());
        Ok(ChatResponse {
            id: "request-1".to_owned(),
            model: self.model_name().to_owned(),
            choices: vec![Choice {
                index: 0,
                message: BaseMessage::assistant("compressed"),
                finish_reason: None,
            }],
            usage: self.usage.clone(),
        })
    }

    fn stream(
        &self,
        _messages: Vec<BaseMessage>,
        _request: ModelRequestConfig,
    ) -> BoxStream<'static, Result<StreamChunk>> {
        Box::pin(futures::stream::empty())
    }
}

#[tokio::test]
async fn every_pass_records_started_then_its_outcome_in_order() {
    let fixture = Fixture::new();
    let (recorder, sink) = recorder(&fixture.counter);
    let pipeline = ContextPipeline::new(vec![Arc::new(Rewriter), Arc::new(Noop)])
        .with_recorder(Some(Arc::clone(&recorder)));

    let mut first = BaseMessage::user("a long user message");
    stamp_message_id(&mut first);
    let first_id = first.metadata["context_message_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let mut messages = vec![first, BaseMessage::assistant("answer")];
    let original = messages.clone();
    let events = pipeline
        .run_in_phase(
            &mut messages,
            &fixture.context(),
            CompressionStatePhase::AddMessages,
        )
        .await
        .unwrap();

    assert_eq!(events.len(), 1);
    assert_eq!(
        sink.statuses(),
        vec![
            CompressionStateStatus::Started,
            CompressionStateStatus::Completed,
            CompressionStateStatus::Started,
            CompressionStateStatus::Noop,
        ]
    );
    let states = sink.states();
    assert_eq!(
        states
            .iter()
            .map(|state| state.processor.as_str())
            .collect::<Vec<_>>(),
        vec!["rewriter", "rewriter", "noop", "noop"]
    );
    assert!(states
        .iter()
        .all(|state| state.phase == CompressionStatePhase::AddMessages));

    // The two records of one pass share an operation id; different passes do not.
    assert_eq!(states[0].operation_id, states[1].operation_id);
    assert_ne!(states[1].operation_id, states[3].operation_id);
    // A simple uuid: 32 hex characters.
    assert_eq!(states[0].operation_id.len(), 32);

    // Before/after metrics of the rewriting pass.
    let before_tokens = fixture.counter.count_messages(&original);
    let after_tokens = fixture.counter.count_messages(&messages);
    assert_eq!(states[0].before.messages, 2);
    assert_eq!(states[0].before.tokens, before_tokens);
    assert_eq!(states[0].context_max, Some(1_000));
    assert!(states[0].after.is_none());
    let completed = &states[1];
    let after = completed.after.as_ref().unwrap();
    assert_eq!(after.messages, 2);
    assert_eq!(after.tokens, after_tokens);
    assert_eq!(
        completed.saved.as_ref().unwrap().tokens,
        i64::try_from(before_tokens - after_tokens).unwrap()
    );
    assert_eq!(
        completed.statistic,
        compute_stats(&messages, fixture.counter.as_ref())
    );
    assert!(completed.duration_ms.is_some());
    assert_eq!(completed.modified_messages.len(), 1);
    assert_eq!(completed.modified_messages[0].message_id, first_id);
    assert_eq!(completed.modified_messages[0].role, "user");
    assert_eq!(completed.compact_summary, "rewrote the first message");

    // Reasons follow Python's spellings, and a noop pass reports no change.
    assert_eq!(states[0].reason, "processor_triggered");
    assert_eq!(completed.reason, "processor_completed");
    assert_eq!(states[3].reason, "processor_noop");
    assert_eq!(states[3].modified_messages.len(), 0);
    assert_eq!(recorder.history().len(), states.len());
}

#[tokio::test]
async fn an_empty_history_is_recorded_as_skipped() {
    let fixture = Fixture::new();
    let (recorder, sink) = recorder(&fixture.counter);
    let pipeline = ContextPipeline::new(vec![Arc::new(Noop)]).with_recorder(Some(recorder));
    let mut messages = Vec::new();

    let events = pipeline
        .run_in_phase(
            &mut messages,
            &fixture.context(),
            CompressionStatePhase::GetContextWindow,
        )
        .await
        .unwrap();

    assert_eq!(events.len(), 0);
    assert_eq!(
        sink.statuses(),
        vec![
            CompressionStateStatus::Started,
            CompressionStateStatus::Skipped,
        ]
    );
    let states = sink.states();
    assert_eq!(states[1].reason, "processor_skipped");
    assert_eq!(states[1].phase, CompressionStatePhase::GetContextWindow);
    assert_eq!(states[1].before.messages, 0);
}

#[tokio::test]
async fn a_failing_processor_is_recorded_and_propagated() {
    let fixture = Fixture::new();
    let (recorder, sink) = recorder(&fixture.counter);
    let pipeline = ContextPipeline::new(vec![Arc::new(Failing)]).with_recorder(Some(recorder));
    let mut messages = vec![BaseMessage::user("hi")];

    let error = pipeline
        .run(&mut messages, &fixture.context())
        .await
        .unwrap_err();

    assert!(matches!(error, Error::Llm(_)));
    assert_eq!(
        sink.statuses(),
        vec![
            CompressionStateStatus::Started,
            CompressionStateStatus::Failed,
        ]
    );
    let states = sink.states();
    assert_eq!(states[1].error.as_deref(), Some("llm: boom"));
    assert_eq!(states[1].reason, "processor_error");
    // `run` records the compaction phase.
    assert_eq!(states[1].phase, CompressionStatePhase::ActiveCompress);
    // The history a failing pass leaves behind is still reported.
    assert_eq!(states[1].after.as_ref().unwrap().messages, 1);
}

#[tokio::test]
async fn a_cancelled_processor_is_recorded_and_propagated() {
    let fixture = Fixture::new();
    let (recorder, sink) = recorder(&fixture.counter);
    let pipeline = ContextPipeline::new(vec![Arc::new(Cancelling)]).with_recorder(Some(recorder));
    let mut messages = vec![BaseMessage::user("hi")];

    let error = pipeline
        .run(&mut messages, &fixture.context())
        .await
        .unwrap_err();

    assert!(matches!(error, Error::Cancelled));
    assert_eq!(
        sink.statuses(),
        vec![
            CompressionStateStatus::Started,
            CompressionStateStatus::Cancelled,
        ]
    );
    let states = sink.states();
    assert_eq!(states[1].reason, "user_cancel");
    assert!(states[1].error.is_none());
}

#[tokio::test]
async fn a_cancelled_token_before_the_pass_records_nothing() {
    let fixture = Fixture::new();
    fixture.cancellation.cancel();
    let (recorder, sink) = recorder(&fixture.counter);
    let pipeline = ContextPipeline::new(vec![Arc::new(Noop)]).with_recorder(Some(recorder));
    let mut messages = vec![BaseMessage::user("hi")];

    let error = pipeline
        .run(&mut messages, &fixture.context())
        .await
        .unwrap_err();

    assert!(matches!(error, Error::Cancelled));
    assert_eq!(sink.len(), 0);
}

#[tokio::test]
async fn the_history_cap_is_respected_and_a_failing_sink_never_fails_the_pass() {
    let fixture = Fixture::new();
    let sink = Arc::new(RecordingSink::failing());
    let recorder = Arc::new(
        ContextProcessorStateRecorder::new("session", "main")
            .with_history_limit(2)
            .with_sink(sink.clone())
            .with_token_counter(fixture.counter.clone()),
    );
    let pipeline = ContextPipeline::new(vec![Arc::new(Noop), Arc::new(Noop), Arc::new(Noop)])
        .with_recorder(Some(Arc::clone(&recorder)));
    let mut messages = vec![BaseMessage::user("hi")];

    let events = pipeline
        .run(&mut messages, &fixture.context())
        .await
        .unwrap();

    assert_eq!(events.len(), 0);
    // Three passes emitted six records; the sink rejected every one of them.
    assert_eq!(sink.len(), 0);
    assert_eq!(recorder.history().len(), 2);
    assert_eq!(
        recorder.last().unwrap().status,
        CompressionStateStatus::Noop
    );
}

#[tokio::test]
async fn a_processor_model_call_sees_the_pass_operation_and_reports_usage() {
    let model = Arc::new(AmbientModel::new());
    let ambient: Arc<dyn Model> = model.clone();
    let fixture = Fixture::new().with_model(ambient);
    let (recorder, sink) = recorder(&fixture.counter);
    let pipeline = ContextPipeline::new(vec![Arc::new(Summarizer)]).with_recorder(Some(recorder));
    let mut messages = vec![BaseMessage::user("a long question")];

    let events = pipeline
        .run(&mut messages, &fixture.context())
        .await
        .unwrap();

    let seen = model.seen();
    assert_eq!(seen.len(), 1);
    let operation_id = seen[0].clone().unwrap();
    let states = sink.states();
    assert_eq!(states.len(), 2);
    assert_eq!(states[0].operation_id, operation_id);
    assert_eq!(states[1].operation_id, operation_id);
    assert_eq!(states[1].model, "ambient-model");
    assert!(states[0].compression_usage.is_none());

    let usage = states[1].compression_usage.as_ref().unwrap();
    assert_eq!(usage.calls, 1);
    assert_eq!(usage.input_tokens, 7);
    assert_eq!(usage.output_tokens, 3);
    assert_eq!(usage.total_tokens, 10);
    assert_eq!(usage.model_name, "ambient-model");

    assert_eq!(events.len(), 1);
    assert_eq!(messages[0].text().as_ref(), "compressed");
    // The scope is closed once the pass is over.
    assert!(telemetry::current_compression_operation_id().is_none());
}

#[tokio::test]
async fn unavailable_usage_is_left_empty() {
    let model = Arc::new(AmbientModel::without_usage());
    let ambient: Arc<dyn Model> = model.clone();
    let fixture = Fixture::new().with_model(ambient);
    let (recorder, sink) = recorder(&fixture.counter);
    let pipeline = ContextPipeline::new(vec![Arc::new(Summarizer)]).with_recorder(Some(recorder));
    let mut messages = vec![BaseMessage::user("a long question")];

    pipeline
        .run(&mut messages, &fixture.context())
        .await
        .unwrap();

    // A provider that reports no usage is not fabricated as zeros.
    assert!(sink.states()[1].compression_usage.is_none());
}

#[tokio::test]
async fn the_summariser_stamps_the_compaction_correlation_inside_a_scope() {
    let inside = ScriptedModel::text("summary");
    let model: Arc<dyn Model> = Arc::new(inside.clone());
    let answer = telemetry::with_operation("op-telemetry", async {
        summarize(Some(&model), "system", "user", 16, || "offline".to_owned()).await
    })
    .await;

    assert_eq!(answer, "summary");
    let calls = inside.calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(
        calls[0].request.extra.get("request_purpose"),
        Some(&serde_json::json!("compaction"))
    );
    assert_eq!(
        calls[0].request.extra.get("context_operation_id"),
        Some(&serde_json::json!("op-telemetry"))
    );

    // Outside a scope the request carries no correlation at all.
    let outside = ScriptedModel::text("summary");
    let model: Arc<dyn Model> = Arc::new(outside.clone());
    let answer = summarize(Some(&model), "system", "user", 16, || "offline".to_owned()).await;

    assert_eq!(answer, "summary");
    let calls = outside.calls();
    assert_eq!(calls.len(), 1);
    assert!(calls[0].request.extra.get("request_purpose").is_none());
    assert!(calls[0].request.extra.get("context_operation_id").is_none());
}

#[tokio::test]
async fn a_pipeline_without_a_recorder_emits_no_telemetry() {
    let fixture = Fixture::new();
    let runs = Arc::new(AtomicUsize::new(0));
    let seen = Arc::new(Mutex::new(Vec::new()));
    let pipeline = ContextPipeline::new(vec![Arc::new(Observer {
        runs: Arc::clone(&runs),
        seen: Arc::clone(&seen),
    })]);
    assert!(pipeline.recorder().is_none());
    let mut messages = vec![BaseMessage::user("hi")];

    for _ in 0..2 {
        let events = pipeline
            .run(&mut messages, &fixture.context())
            .await
            .unwrap();
        assert_eq!(events.len(), 0);
    }

    // The processor still ran exactly once per pass, and no operation scope was
    // entered around it.
    assert_eq!(runs.load(Ordering::SeqCst), 2);
    assert_eq!(seen.lock().unwrap().clone(), vec![None::<String>, None]);
}

#[tokio::test]
async fn pipeline_deps_wire_the_recorder_into_the_built_pipeline() {
    let fixture = Fixture::new();
    let (recorder, sink) = recorder(&fixture.counter);
    let store: Arc<dyn openjiuwen_context::OffloadStore> = Arc::new(InMemoryOffloadStore::new());
    let deps = PipelineDeps::new(store).with_recorder(Some(recorder));
    let pipeline = ContextPipeline::from_configs_with(
        &[ProcessorConfig::Truncation(TruncationConfig::default())],
        &deps,
    )
    .unwrap();
    assert!(pipeline.recorder().is_some());

    let mut messages = vec![BaseMessage::user("short"), BaseMessage::assistant("answer")];
    pipeline
        .run_in_phase(
            &mut messages,
            &fixture.context(),
            CompressionStatePhase::GetContextWindow,
        )
        .await
        .unwrap();

    let states = sink.states();
    assert_eq!(
        sink.statuses(),
        vec![
            CompressionStateStatus::Started,
            CompressionStateStatus::Noop,
        ]
    );
    assert_eq!(states[0].processor, "truncation");
    assert_eq!(states[1].phase, CompressionStatePhase::GetContextWindow);
}
