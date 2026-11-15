#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Public-API integration tests for the processor pipeline.
//!
//! These tests drive [`ContextPipeline`] and the three built-in processors the
//! way an agent or the harness does: ordered execution, post-run tool-pair
//! repair, truncation, tool-result trimming/offload/rehydration, summarisation,
//! failure and cancellation paths, and the wire format of [`ContextEvent`].
//!
//! Everything is offline and deterministic: token counts come from the
//! `characters` counter with a ratio of one (one character plus chat framing per
//! message), the only model is [`openjiuwen_test_support::ScriptedModel`], and
//! offload storage is the in-process stores shipped with the crate.

use std::sync::{Arc, Mutex, PoisonError};

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_context::{
    count_message, dialogue_round_ranges, rehydrate, ContextConfig, ContextEvent, ContextEventKind,
    ContextPipeline, ContextProcessor, InMemoryOffloadStore, KvOffloadStore, OffloadKind,
    OffloadMarker, OffloadScope, OffloadStore, ProcessorConfig, ProcessorContext,
    SummarizationConfig, SummarizationProcessor, TokenCounterConfig, TokenCounterKind,
    ToolResultConfig, ToolResultProcessor, TruncationConfig, TruncationProcessor,
    CONTEXT_SUMMARY_PREFIX, PERSISTED_OUTPUT_TAG, TRUNCATED_MARKER,
};
use openjiuwen_foundation::llm::Model;
use openjiuwen_foundation::prompt::{ApproxTokenCounter, BaseMessage, Role, ToolCall};
use openjiuwen_foundation::store::InMemoryKvStore;
use openjiuwen_test_support::{ScriptedError, ScriptedModel};
use serde_json::json;
use tokio_util::sync::CancellationToken;

/// A deterministic configuration: one character per token.
fn char_config() -> ContextConfig {
    ContextConfig {
        token_counter: TokenCounterConfig {
            kind: TokenCounterKind::Characters,
            chars_per_token: 1,
        },
        ..ContextConfig::default()
    }
}

/// The one-character-per-token counter every test uses.
fn counter() -> ApproxTokenCounter {
    ApproxTokenCounter::with_chars_per_token(1)
}

/// Builds one tool exchange: a user turn, a tool-calling assistant, and its result.
fn tool_exchange(id: &str, tool: &str, output: &str) -> Vec<BaseMessage> {
    vec![
        BaseMessage::user("run"),
        BaseMessage::assistant("").with_tool_calls(vec![ToolCall::new(tool).with_id(id)]),
        BaseMessage::tool(id, output),
    ]
}

/// Shared processor inputs, so each test only spells out what it is testing.
struct Fixture {
    config: ContextConfig,
    counter: ApproxTokenCounter,
    store: Arc<dyn OffloadStore>,
    scope: OffloadScope,
    cancellation: CancellationToken,
}

impl Fixture {
    /// A fixture writing offloaded records to `store`.
    fn new(store: Arc<dyn OffloadStore>) -> Self {
        Self {
            config: char_config(),
            counter: counter(),
            store,
            scope: OffloadScope::new("session-1", "main"),
            cancellation: CancellationToken::new(),
        }
    }

    /// A fixture backed by the in-memory offload store.
    fn memory() -> Self {
        Self::new(Arc::new(InMemoryOffloadStore::new()))
    }

    /// A context over the fixture, without a model.
    fn context(&self, budget: usize, trigger_tokens: usize, manual: bool) -> ProcessorContext<'_> {
        ProcessorContext {
            config: &self.config,
            counter: &self.counter,
            model: None,
            offload: &*self.store,
            scope: &self.scope,
            budget,
            trigger_tokens,
            manual,
            cancellation: &self.cancellation,
            recall: None,
            session_memory: None,
            session_memory_agent: None,
        }
    }

    /// A context that offers `model` to the processors of one request.
    fn model_context<'a>(
        &'a self,
        model: &'a Arc<dyn Model>,
        trigger_tokens: usize,
    ) -> ProcessorContext<'a> {
        ProcessorContext {
            config: &self.config,
            counter: &self.counter,
            model: Some(model),
            offload: &*self.store,
            scope: &self.scope,
            budget: 100_000,
            trigger_tokens,
            manual: false,
            cancellation: &self.cancellation,
            recall: None,
            session_memory: None,
            session_memory_agent: None,
        }
    }
}

/// A processor that records its name and reports a truncation event.
struct Recorder {
    name: &'static str,
    log: Arc<Mutex<Vec<&'static str>>>,
}

#[async_trait]
impl ContextProcessor for Recorder {
    fn name(&self) -> &'static str {
        self.name
    }

    async fn process(
        &self,
        _messages: &mut Vec<BaseMessage>,
        _cx: &ProcessorContext<'_>,
    ) -> Result<Option<ContextEvent>> {
        self.log
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(self.name);
        Ok(Some(ContextEvent {
            processor: self.name.to_owned(),
            kind: ContextEventKind::Truncated,
            ..ContextEvent::default()
        }))
    }
}

/// A processor that appends a message and reports nothing.
struct Appender;

#[async_trait]
impl ContextProcessor for Appender {
    fn name(&self) -> &'static str {
        "appender"
    }

    async fn process(
        &self,
        messages: &mut Vec<BaseMessage>,
        _cx: &ProcessorContext<'_>,
    ) -> Result<Option<ContextEvent>> {
        messages.push(BaseMessage::user("added"));
        Ok(None)
    }
}

/// A processor that always fails with an LLM error.
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

/// A processor that removes the assistant messages owning a tool call.
struct DropToolCallOwner;

#[async_trait]
impl ContextProcessor for DropToolCallOwner {
    fn name(&self) -> &'static str {
        "drop_tool_call_owner"
    }

    async fn process(
        &self,
        messages: &mut Vec<BaseMessage>,
        _cx: &ProcessorContext<'_>,
    ) -> Result<Option<ContextEvent>> {
        messages.retain(|message| message.role != Role::Assistant || message.tool_calls.is_empty());
        Ok(None)
    }
}

/// Custom processors run in vector order, and their events come back in order.
#[tokio::test]
async fn pipeline_runs_processors_in_order_and_returns_events_in_order() {
    let fixture = Fixture::memory();
    let log = Arc::new(Mutex::new(Vec::new()));
    let pipeline = ContextPipeline::new(vec![
        Arc::new(Recorder {
            name: "first",
            log: Arc::clone(&log),
        }),
        Arc::new(Recorder {
            name: "second",
            log: Arc::clone(&log),
        }),
    ]);
    assert_eq!(pipeline.processors().len(), 2);

    let mut messages = vec![BaseMessage::user("hi")];
    let events = pipeline
        .run(&mut messages, &fixture.context(1_000, 800, false))
        .await
        .unwrap();

    assert_eq!(
        *log.lock().unwrap_or_else(PoisonError::into_inner),
        vec!["first", "second"]
    );
    assert_eq!(events.len(), 2);
    assert_eq!(events[0].processor, "first");
    assert_eq!(events[1].processor, "second");
    assert_eq!(events[0].kind, ContextEventKind::Truncated);
    assert_eq!(messages.len(), 1);
}

/// A failing processor propagates its error while earlier mutations stay applied.
#[tokio::test]
async fn pipeline_propagates_errors_and_keeps_earlier_mutations() {
    let fixture = Fixture::memory();
    let pipeline = ContextPipeline::new(vec![Arc::new(Appender), Arc::new(Failing)]);
    let mut messages = vec![BaseMessage::user("hi")];

    let error = pipeline
        .run(&mut messages, &fixture.context(1_000, 800, false))
        .await
        .unwrap_err();

    assert!(matches!(error, Error::Llm(_)));
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[1].text(), "added");
}

/// After the pipeline runs, tool results orphaned by an earlier processor go away.
#[tokio::test]
async fn pipeline_repairs_orphan_tool_results_after_a_processor_drops_the_owner() {
    let fixture = Fixture::memory();
    let pipeline = ContextPipeline::new(vec![Arc::new(DropToolCallOwner)]);
    let mut messages = vec![
        BaseMessage::user("q"),
        BaseMessage::assistant("").with_tool_calls(vec![ToolCall::new("shell").with_id("call_1")]),
        BaseMessage::tool("call_1", "result"),
    ];

    let events = pipeline
        .run(&mut messages, &fixture.context(1_000, 800, false))
        .await
        .unwrap();

    assert!(events.is_empty(), "unexpected events: {events:?}");
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].role, Role::User);
    assert_eq!(messages[0].text(), "q");
}

/// A cancelled request stops before the first processor runs.
#[tokio::test]
async fn pipeline_stops_before_the_first_processor_when_cancelled() {
    let fixture = Fixture::memory();
    fixture.cancellation.cancel();
    let log = Arc::new(Mutex::new(Vec::new()));
    let pipeline = ContextPipeline::new(vec![Arc::new(Recorder {
        name: "never",
        log: Arc::clone(&log),
    })]);
    let mut messages = vec![BaseMessage::user("hi")];

    let error = pipeline
        .run(&mut messages, &fixture.context(1_000, 800, false))
        .await
        .unwrap_err();

    assert!(matches!(error, Error::Cancelled));
    assert!(log
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .is_empty());
    assert_eq!(messages.len(), 1);
}

/// Configs deserialize by their `type` tag, fill defaults, and build the default pipeline.
#[test]
fn processor_configs_deserialize_by_tag_and_use_the_default_pipeline() {
    let parsed: ProcessorConfig =
        serde_json::from_value(json!({"type": "summarization", "keep_recent_messages": 2}))
            .unwrap();
    assert_eq!(
        parsed,
        ProcessorConfig::Summarization(SummarizationConfig {
            keep_recent_messages: 2,
            ..SummarizationConfig::default()
        })
    );
    assert_eq!(
        serde_json::from_value::<ProcessorConfig>(serde_json::to_value(&parsed).unwrap()).unwrap(),
        parsed
    );

    let truncation: ProcessorConfig =
        serde_json::from_value(json!({"type": "truncation", "keep_recent_messages": 3})).unwrap();
    assert_eq!(
        truncation,
        ProcessorConfig::Truncation(TruncationConfig {
            keep_recent_messages: 3,
            ..TruncationConfig::default()
        })
    );

    let tool_result: ProcessorConfig =
        serde_json::from_value(json!({"type": "tool_result"})).unwrap();
    assert_eq!(
        tool_result,
        ProcessorConfig::ToolResult(ToolResultConfig::default())
    );

    let defaults = ContextConfig::default().processors;
    assert_eq!(defaults.len(), 2);
    assert!(matches!(defaults[0], ProcessorConfig::ToolResult(_)));
    assert!(matches!(defaults[1], ProcessorConfig::Truncation(_)));
}

/// A summarisation config without a model is a configuration error.
#[test]
fn from_configs_requires_a_model_for_summarization() {
    let store: Arc<dyn OffloadStore> = Arc::new(InMemoryOffloadStore::new());
    let configs = vec![ProcessorConfig::Summarization(
        SummarizationConfig::default(),
    )];

    let error = ContextPipeline::from_configs(&configs, None, Arc::clone(&store)).unwrap_err();
    assert!(matches!(error, Error::Config(_)));

    let model: Arc<dyn Model> = Arc::new(ScriptedModel::builder().respond_text("unused").build());
    let pipeline = ContextPipeline::from_configs(
        &[
            ProcessorConfig::ToolResult(ToolResultConfig::default()),
            ProcessorConfig::Truncation(TruncationConfig::default()),
            ProcessorConfig::Summarization(SummarizationConfig::default()),
        ],
        Some(model),
        store,
    )
    .unwrap();
    assert_eq!(pipeline.processors().len(), 3);
    assert_eq!(pipeline.processors()[0].name(), "tool_result");
    assert_eq!(pipeline.processors()[1].name(), "truncation");
    assert_eq!(pipeline.processors()[2].name(), "summarization");
}

/// Over budget, truncation drops whole oldest rounds, keeps systems, protects the tail.
#[tokio::test]
async fn truncation_drops_whole_oldest_rounds_and_keeps_system_messages() {
    let fixture = Fixture::memory();
    let mut messages = vec![
        BaseMessage::system("persona"),
        BaseMessage::user("q1"),
        BaseMessage::assistant("a1"),
        BaseMessage::user("q2"),
        BaseMessage::assistant("a2"),
        BaseMessage::user("q3"),
        BaseMessage::assistant("a3"),
    ];
    assert_eq!(dialogue_round_ranges(&messages).len(), 3);
    let tokens = openjiuwen_context::count_messages(&fixture.counter, &messages);
    let budget = tokens - 12;
    let processor = TruncationProcessor::new(TruncationConfig {
        keep_recent_messages: 4,
        ..TruncationConfig::default()
    });

    let event = processor
        .process(&mut messages, &fixture.context(budget, 100_000, false))
        .await
        .unwrap()
        .unwrap();

    assert_eq!(event.kind, ContextEventKind::Truncated);
    assert_eq!(event.messages_changed, vec![1, 2]);
    assert!(event.tokens_before > event.tokens_after);
    assert!(openjiuwen_context::count_messages(&fixture.counter, &messages) <= budget);
    assert_eq!(messages[0].role, Role::System);
    assert_eq!(messages[0].text(), "persona");
    assert_eq!(messages[1].text(), "q2");
    assert_eq!(messages[3].text(), "q3");
    assert_eq!(messages[4].text(), "a3");
}

/// A history below the threshold reports no event; `manual` forces the pass.
#[tokio::test]
async fn truncation_is_a_noop_below_the_threshold_and_manual_bypasses_it() {
    let fixture = Fixture::memory();
    let mut messages = vec![
        BaseMessage::system("persona"),
        BaseMessage::user("q1"),
        BaseMessage::assistant("a1"),
        BaseMessage::user("q2"),
        BaseMessage::assistant("a2"),
        BaseMessage::user("q3"),
        BaseMessage::assistant("a3"),
    ];
    // With no processor budget of its own, a request far below the trigger is a
    // no-op even though the history is longer than the manual budget below.
    let idle = TruncationProcessor::new(TruncationConfig {
        max_tokens: None,
        keep_recent_messages: 2,
        ..TruncationConfig::default()
    });
    let events = ContextPipeline::new(vec![Arc::new(idle)])
        .run(&mut messages, &fixture.context(100_000, 100_000, false))
        .await
        .unwrap();
    assert!(events.is_empty(), "unexpected events: {events:?}");
    assert_eq!(messages.len(), 7);

    let manual = TruncationProcessor::new(TruncationConfig {
        max_tokens: Some(23),
        keep_recent_messages: 2,
        ..TruncationConfig::default()
    });
    let events = ContextPipeline::new(vec![Arc::new(manual)])
        .run(&mut messages, &fixture.context(100_000, 100_000, true))
        .await
        .unwrap();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].kind, ContextEventKind::Truncated);
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[1].text(), "q3");
    assert_eq!(messages[2].text(), "a3");
}

/// Old long results are offloaded; newer, short, and protected results are not.
#[tokio::test]
async fn tool_result_offloads_a_long_old_result_and_rehydrates_it() {
    let fixture = Fixture::memory();
    let long = "L".repeat(2_000);
    let short = "s".repeat(40);
    let protected = "r".repeat(80);
    let newest = "n".repeat(80);
    let mut messages = Vec::new();
    messages.extend(tool_exchange("call_long", "shell", &long));
    messages.extend(tool_exchange("call_short", "grep", &short));
    messages.extend(tool_exchange("call_read", "read_file", &protected));
    messages.extend(tool_exchange("call_new", "shell", &newest));
    let processor = ToolResultProcessor::new(
        ToolResultConfig {
            keep_recent_results: 1,
            max_result_tokens: 10,
            head_chars: 20,
            tail_chars: 10,
            offload_over_chars: Some(50),
            ..ToolResultConfig::default()
        },
        Arc::clone(&fixture.store),
    );

    let event = processor
        .process(&mut messages, &fixture.context(100_000, 80_000, false))
        .await
        .unwrap()
        .unwrap();

    assert_eq!(event.kind, ContextEventKind::Offloaded);
    assert_eq!(event.messages_changed, vec![2, 5]);
    assert!(event.tokens_after < event.tokens_before);

    let handle = event.offload_handle.clone().expect("an offload handle");
    let text = messages[2].text().to_string();
    assert!(text.contains(PERSISTED_OUTPUT_TAG));
    assert_eq!(messages[2].role, Role::Tool);
    assert_eq!(messages[2].tool_call_id.as_deref(), Some("call_long"));
    let marker = OffloadMarker::parse(&text).expect("a parseable marker");
    assert_eq!(marker.handle, handle);
    assert_eq!(marker.kind, OffloadKind::InMemory);

    let restored = rehydrate(&*fixture.store, &fixture.scope, &text)
        .await
        .unwrap()
        .expect("the offloaded record");
    assert_eq!(restored.len(), 1);
    assert_eq!(restored[0].text(), long);
    assert_eq!(restored[0].role, Role::Tool);
    assert_eq!(restored[0].tool_call_id.as_deref(), Some("call_long"));

    assert!(messages[5].text().contains(TRUNCATED_MARKER));
    assert_eq!(messages[8].text(), protected);
    assert_eq!(messages[11].text(), newest);
}

/// Old short results are trimmed in place, `read_file` output is never touched.
#[tokio::test]
async fn tool_result_trims_old_short_results_and_protects_read_file() {
    let fixture = Fixture::memory();
    let short = "s".repeat(40);
    let protected = "r".repeat(80);
    let mut messages = Vec::new();
    messages.extend(tool_exchange("call_0", "grep", &short));
    messages.extend(tool_exchange("call_1", "grep", &short));
    messages.extend(tool_exchange("call_read", "read_file", &protected));
    messages.extend(tool_exchange("call_new", "grep", &short));
    let processor = ToolResultProcessor::new(
        ToolResultConfig {
            keep_recent_results: 1,
            max_result_tokens: 10,
            head_chars: 20,
            tail_chars: 10,
            offload_over_chars: Some(50),
            ..ToolResultConfig::default()
        },
        Arc::clone(&fixture.store),
    );

    let event = processor
        .process(&mut messages, &fixture.context(100_000, 80_000, false))
        .await
        .unwrap()
        .unwrap();

    assert_eq!(event.kind, ContextEventKind::ToolResultTrimmed);
    assert_eq!(event.offload_handle, None);
    assert_eq!(event.messages_changed, vec![2, 5]);
    assert!(messages[2].text().contains(TRUNCATED_MARKER));
    assert!(messages[5].text().contains(TRUNCATED_MARKER));
    assert_eq!(messages[8].text(), protected);
    assert_eq!(messages[11].text(), short);
    assert_eq!(messages[2].tool_call_id.as_deref(), Some("call_0"));
}

/// The total tool-output budget shrinks the oldest unprotected results first.
#[tokio::test]
async fn tool_result_enforces_the_total_tool_budget_oldest_first() {
    let fixture = Fixture::memory();
    let mut messages = Vec::new();
    for index in 0..4 {
        messages.extend(tool_exchange(
            &format!("call_{index}"),
            "grep",
            &format!("{}{index}", "q".repeat(200)),
        ));
    }
    let processor = ToolResultProcessor::new(
        ToolResultConfig {
            max_result_tokens: 1_000,
            max_total_tool_tokens: Some(600),
            head_chars: 20,
            tail_chars: 10,
            offload_over_chars: None,
            keep_recent_results: 1,
            ..ToolResultConfig::default()
        },
        Arc::clone(&fixture.store),
    );

    let event = processor
        .process(&mut messages, &fixture.context(100_000, 80_000, false))
        .await
        .unwrap()
        .unwrap();

    assert_eq!(event.kind, ContextEventKind::ToolResultTrimmed);
    assert!(event.messages_changed.len() >= 2);
    assert_eq!(event.messages_changed.last(), Some(&8));
    let total: usize = messages
        .iter()
        .filter(|message| message.role == Role::Tool)
        .map(|message| count_message(&fixture.counter, message))
        .sum();
    assert!(total <= 600, "tool output still weighs {total} tokens");
    assert_eq!(messages[11].text(), format!("{}{}", "q".repeat(200), 3));
}

/// The same offload/rehydrate contract holds over a key-value backed store.
#[tokio::test]
async fn tool_result_offloads_through_the_kv_store_and_rehydrates() {
    let store: Arc<dyn OffloadStore> =
        Arc::new(KvOffloadStore::new(Arc::new(InMemoryKvStore::new())));
    let fixture = Fixture::new(store);
    let original = "k".repeat(120);
    let mut messages = tool_exchange("call_kv", "shell", &original);
    let processor = ToolResultProcessor::new(
        ToolResultConfig {
            keep_recent_results: 0,
            max_result_tokens: 10,
            head_chars: 20,
            tail_chars: 10,
            offload_over_chars: Some(50),
            ..ToolResultConfig::default()
        },
        Arc::clone(&fixture.store),
    );

    let event = processor
        .process(&mut messages, &fixture.context(100_000, 80_000, false))
        .await
        .unwrap()
        .unwrap();

    assert_eq!(event.kind, ContextEventKind::Offloaded);
    let text = messages[2].text().to_string();
    let marker = OffloadMarker::parse(&text).expect("a parseable marker");
    assert_eq!(marker.kind, OffloadKind::Filesystem);
    assert_eq!(marker.path, None);
    assert_eq!(Some(marker.handle.clone()), event.offload_handle);
    let restored = rehydrate(&*fixture.store, &fixture.scope, &text)
        .await
        .unwrap()
        .expect("the offloaded record");
    assert_eq!(restored[0].text(), original);
}

/// One summary message replaces the old rounds; the model is called exactly once.
#[tokio::test]
async fn summarization_replaces_old_rounds_with_one_summary_and_calls_the_model_once() {
    let fixture = Fixture::memory();
    let scripted = ScriptedModel::builder().respond_text("the summary").build();
    let model: Arc<dyn Model> = Arc::new(scripted.clone());
    let processor = SummarizationProcessor::new(
        SummarizationConfig {
            keep_recent_messages: 2,
            keep_last_rounds: 1,
            tokens_threshold: Some(1),
            min_messages_to_summarize: 2,
            ..SummarizationConfig::default()
        },
        Arc::clone(&model),
    );
    let mut messages = vec![
        BaseMessage::system("persona"),
        BaseMessage::user("q1"),
        BaseMessage::assistant("a1"),
        BaseMessage::user("q2"),
        BaseMessage::assistant("a2"),
        BaseMessage::user("q3"),
        BaseMessage::assistant("a3"),
        BaseMessage::user("q4"),
    ];

    let event = processor
        .process(&mut messages, &fixture.model_context(&model, 1_000))
        .await
        .unwrap()
        .unwrap();

    assert_eq!(event.kind, ContextEventKind::Summarized);
    assert_eq!(event.summary.as_deref(), Some("the summary"));
    assert_eq!(event.messages_changed, vec![1, 2, 3, 4, 5]);
    assert_eq!(scripted.call_count(), 1);

    assert_eq!(messages.len(), 4);
    assert_eq!(messages[0].role, Role::System);
    assert_eq!(messages[0].text(), "persona");
    assert!(messages[1].text().contains(CONTEXT_SUMMARY_PREFIX));
    assert!(messages[1].text().contains("the summary"));
    assert_eq!(
        messages
            .iter()
            .filter(|message| message.text().contains(CONTEXT_SUMMARY_PREFIX))
            .count(),
        1
    );
    assert_eq!(messages[2].text(), "a3");
    assert_eq!(messages[3].text(), "q4");

    let call = scripted.last_call().expect("one recorded call");
    assert_eq!(call.messages.len(), 2);
    assert_eq!(call.messages[0].role, Role::System);
    assert_eq!(call.request.max_tokens, Some(900));
}

/// A failing model surfaces as an LLM error and never corrupts the history.
#[tokio::test]
async fn summarization_failure_is_an_llm_error_and_leaves_the_history_unchanged() {
    let fixture = Fixture::memory();
    let model: Arc<dyn Model> = Arc::new(
        ScriptedModel::builder()
            .fail(ScriptedError::llm("provider down"))
            .build(),
    );
    let processor = SummarizationProcessor::new(
        SummarizationConfig {
            keep_recent_messages: 2,
            keep_last_rounds: 1,
            tokens_threshold: Some(1),
            min_messages_to_summarize: 2,
            ..SummarizationConfig::default()
        },
        Arc::clone(&model),
    );
    let mut messages = vec![
        BaseMessage::system("persona"),
        BaseMessage::user("q1"),
        BaseMessage::assistant("a1"),
        BaseMessage::user("q2"),
        BaseMessage::assistant("a2"),
        BaseMessage::user("q3"),
        BaseMessage::assistant("a3"),
        BaseMessage::user("q4"),
    ];
    let before = messages.clone();

    let error = processor
        .process(&mut messages, &fixture.model_context(&model, 1_000))
        .await
        .unwrap_err();

    assert!(matches!(error, Error::Llm(_)));
    assert_eq!(messages, before);
}

/// A cancelled request reports `Cancelled` and never reaches the model.
#[tokio::test]
async fn summarization_cancellation_returns_cancelled_before_the_model_call() {
    let fixture = Fixture::memory();
    let scripted = ScriptedModel::builder().respond_text("unused").build();
    let model: Arc<dyn Model> = Arc::new(scripted.clone());
    let processor = SummarizationProcessor::new(
        SummarizationConfig {
            keep_recent_messages: 2,
            keep_last_rounds: 1,
            tokens_threshold: Some(1),
            min_messages_to_summarize: 2,
            ..SummarizationConfig::default()
        },
        Arc::clone(&model),
    );
    fixture.cancellation.cancel();
    let mut messages = vec![
        BaseMessage::system("persona"),
        BaseMessage::user("q1"),
        BaseMessage::assistant("a1"),
        BaseMessage::user("q2"),
        BaseMessage::assistant("a2"),
        BaseMessage::user("q3"),
        BaseMessage::assistant("a3"),
        BaseMessage::user("q4"),
    ];
    let before = messages.clone();

    let error = processor
        .process(&mut messages, &fixture.model_context(&model, 1_000))
        .await
        .unwrap_err();

    assert!(matches!(error, Error::Cancelled));
    assert_eq!(messages, before);
    assert_eq!(scripted.call_count(), 0);
}

/// Every event kind keeps its snake-case wire name and round-trips through JSON.
#[test]
fn context_event_kinds_round_trip_on_the_wire() {
    for (kind, wire) in [
        (ContextEventKind::Truncated, "truncated"),
        (ContextEventKind::Summarized, "summarized"),
        (ContextEventKind::Offloaded, "offloaded"),
        (ContextEventKind::ToolResultTrimmed, "tool_result_trimmed"),
    ] {
        assert_eq!(serde_json::to_value(kind).unwrap(), json!(wire));
        assert_eq!(
            serde_json::from_value::<ContextEventKind>(json!(wire)).unwrap(),
            kind
        );

        let event = ContextEvent {
            processor: "truncation".to_owned(),
            kind,
            messages_changed: vec![0, 2],
            summary: (kind == ContextEventKind::Summarized).then_some("the summary".to_owned()),
            offload_handle: (kind == ContextEventKind::Offloaded).then_some("handle-1".to_owned()),
            tokens_before: 100,
            tokens_after: 40,
        };
        let encoded = serde_json::to_value(&event).unwrap();
        assert_eq!(encoded["kind"], json!(wire));
        assert_eq!(
            serde_json::from_value::<ContextEvent>(encoded).unwrap(),
            event
        );
    }

    let minimal: ContextEvent = serde_json::from_value(json!({"kind": "summarized"})).unwrap();
    assert_eq!(minimal.kind, ContextEventKind::Summarized);
    assert_eq!(minimal.tokens_before, 0);
    assert_eq!(minimal.summary, None);
}
