#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Public-API tests for the per-processor checkpoint state.
//!
//! [`ProcessorState`] is the persisted side of
//! [`ContextProcessor::save_state`]: a stateful processor counts what it did,
//! [`ContextPipeline::save_state`] keys those counts by processor name, and
//! [`ContextState`] carries the map through the session so a new engine can
//! restore it. The suite pins the bag's own semantics, the pipeline key scheme,
//! one counter test per stateful processor, the end-to-end session round trip,
//! and the backward-compatibility and corrupt-state paths.
//!
//! Everything is offline and deterministic: token counts come from the
//! one-character-per-token counter, offload storage is in memory, sessions are
//! in-process `AgentSession`s, and no model is configured, so every summary uses
//! the deterministic offline path.
//!
//! [`ContextProcessor::save_state`]: openjiuwen_context::processor::ContextProcessor::save_state
//! [`ContextState`]: openjiuwen_context::ContextState
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::Result;
use openjiuwen_context::processor::{
    ContextPipeline, ContextProcessor, CurrentRoundCompressor, CurrentRoundCompressorConfig,
    DialogueCompressor, DialogueCompressorConfig, FullCompactProcessor, FullCompactProcessorConfig,
    MessageOffloader, MessageOffloaderConfig, MessageSummaryOffloader,
    MessageSummaryOffloaderConfig, MicroCompactProcessor, MicroCompactProcessorConfig,
    ProcessorContext, RoundLevelCompressor, RoundLevelCompressorConfig, ToolResultBudgetProcessor,
    ToolResultBudgetProcessorConfig, ToolResultWindowProcessor, ToolResultWindowProcessorConfig,
    CURRENT_ROUND_MEMORY_BLOCK,
};
use openjiuwen_context::{
    CompressionStatus, ContextConfig, ContextEngine, ContextEvent, ContextState,
    InMemoryOffloadStore, MemoryBlockKind, ModelContext, OffloadScope, OffloadStore,
    ProcessorConfig, ProcessorState, TokenCounterConfig, TokenCounterKind,
};
use openjiuwen_foundation::prompt::{ApproxTokenCounter, BaseMessage, ToolCall};
use openjiuwen_session::{AgentSession, Session};
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

/// The one-character-per-token counter every test uses.
fn counter() -> ApproxTokenCounter {
    ApproxTokenCounter::with_chars_per_token(1)
}

/// A body long enough that replacing it with a block or a summary is a saving.
fn long(text: &str) -> String {
    format!("{text}: {}", "detail ".repeat(300))
}

/// One round with a tool exchange, so it is longer than two messages.
fn tool_round(index: usize, tool: &str) -> Vec<BaseMessage> {
    vec![
        BaseMessage::user(format!("u{index}")),
        BaseMessage::assistant("")
            .with_tool_calls(vec![ToolCall::new(tool).with_id(format!("call_{index}"))]),
        BaseMessage::tool(
            format!("call_{index}"),
            long(&format!("tool output {index}")),
        ),
        BaseMessage::assistant(format!("final{index}")),
    ]
}

/// A persona plus `rounds` tool rounds of `tool`.
fn history(rounds: usize, tool: &str) -> Vec<BaseMessage> {
    let mut messages = vec![BaseMessage::system("persona")];
    for index in 1..=rounds {
        messages.extend(tool_round(index, tool));
    }
    messages
}

/// Shared processor inputs.
struct Fixture {
    config: ContextConfig,
    counter: ApproxTokenCounter,
    store: Arc<dyn OffloadStore>,
    scope: OffloadScope,
    cancellation: CancellationToken,
}

impl Fixture {
    fn new() -> Self {
        Self {
            config: ContextConfig::default(),
            counter: counter(),
            store: Arc::new(InMemoryOffloadStore::new()),
            scope: OffloadScope::new("session-1", "main"),
            cancellation: CancellationToken::new(),
        }
    }

    fn context(&self, manual: bool) -> ProcessorContext<'_> {
        ProcessorContext {
            config: &self.config,
            counter: &self.counter,
            model: None,
            offload: &*self.store,
            scope: &self.scope,
            budget: 100_000,
            trigger_tokens: 80_000,
            manual,
            cancellation: &self.cancellation,
            recall: None,
            session_memory: None,
        }
    }
}

#[test]
fn counters_snapshots_and_clones_are_independent() {
    let state = ProcessorState::default();
    assert_eq!(state.snapshot(), Value::Null);
    assert_eq!(state.counter("compressions"), 0);

    state.bump("compressions");
    state.bump("compressions");
    assert_eq!(state.counter("compressions"), 2);

    state.set_u64("blocks", 7);
    assert_eq!(state.counter("blocks"), 7);
    state.record_tokens("last_tokens_before", 1_234);
    assert_eq!(state.counter("last_tokens_before"), 1_234);
    // Counts saturate instead of wrapping.
    state.record_tokens("last_tokens_after", usize::MAX);
    assert_eq!(state.counter("last_tokens_after"), u64::MAX);

    let snapshot = state.snapshot();
    assert_eq!(snapshot["compressions"], 2);

    let restored = ProcessorState::default();
    restored.restore(&snapshot);
    assert_eq!(restored.snapshot(), snapshot);

    // `Null` and non-object values are ignored, so corrupt or foreign entries
    // never replace real counters.
    restored.restore(&Value::Null);
    restored.restore(&json!("not a bag"));
    restored.restore(&json!([1, 2, 3]));
    restored.restore(&json!(7));
    assert_eq!(restored.snapshot(), snapshot);

    let copy = restored.clone();
    copy.bump("compressions");
    assert_eq!(copy.counter("compressions"), 3);
    assert_eq!(restored.counter("compressions"), 2);
}

/// A processor whose counters are the state it was constructed with.
struct Named {
    name: &'static str,
    state: ProcessorState,
}

impl Named {
    fn new(name: &'static str, state: Option<Value>) -> Self {
        let processor = Self {
            name,
            state: ProcessorState::default(),
        };
        if let Some(state) = state {
            processor.state.restore(&state);
        }
        processor
    }
}

#[async_trait]
impl ContextProcessor for Named {
    fn name(&self) -> &'static str {
        self.name
    }

    async fn process(
        &self,
        _messages: &mut Vec<BaseMessage>,
        _cx: &ProcessorContext<'_>,
    ) -> Result<Option<ContextEvent>> {
        Ok(None)
    }

    fn save_state(&self) -> Value {
        self.state.snapshot()
    }

    fn load_state(&self, state: &Value) {
        self.state.restore(state);
    }
}

#[tokio::test]
async fn pipeline_keys_are_positional_and_null_states_are_skipped() {
    let pipeline = ContextPipeline::new(vec![
        Arc::new(Named::new("compress", Some(json!({"compressions": 1}))))
            as Arc<dyn ContextProcessor>,
        Arc::new(Named::new("compress", Some(json!({"compressions": 2})))),
        Arc::new(Named::new("quiet", None)),
    ]);

    let state = pipeline.save_state();
    assert_eq!(state.len(), 2);
    assert_eq!(state["compress"], json!({"compressions": 1}));
    assert_eq!(state["compress#2"], json!({"compressions": 2}));
    assert!(!state.contains_key("quiet"));
    assert!(!state.contains_key("quiet#2"));

    let target = ContextPipeline::new(vec![
        Arc::new(Named::new("compress", None)) as Arc<dyn ContextProcessor>,
        Arc::new(Named::new("compress", None)),
        Arc::new(Named::new("quiet", None)),
    ]);
    target.load_state(&state);
    let restored = target.save_state();
    assert_eq!(restored, state);
    assert!(target.processors()[2].save_state().is_null());
}

#[tokio::test]
async fn unknown_and_foreign_state_keys_are_ignored() {
    let pipeline = ContextPipeline::new(vec![
        Arc::new(Named::new("compress", None)) as Arc<dyn ContextProcessor>,
        Arc::new(Named::new("other", Some(json!({"n": 3})))),
    ]);
    let mut state = pipeline.save_state();
    state.insert("ghost".to_owned(), json!({"n": 9}));
    state.insert("compress#2".to_owned(), json!({"n": 4}));

    pipeline.load_state(&state);
    // Only the entry that names the first processor reached it.
    assert_eq!(pipeline.processors()[0].save_state(), Value::Null);
    assert_eq!(pipeline.processors()[1].save_state(), json!({"n": 3}));
}

#[tokio::test]
async fn dialogue_compressor_counts_the_blocks_it_applied() {
    let fixture = Fixture::new();
    let compressor = DialogueCompressor::new(
        DialogueCompressorConfig {
            tokens_threshold: 100,
            compression_target_tokens: 200,
            ..DialogueCompressorConfig::default()
        },
        None,
    );
    let mut messages = history(3, "read_file");
    let event = compressor
        .process(&mut messages, &fixture.context(true))
        .await
        .unwrap()
        .unwrap();

    assert_eq!(compressor.state.counter("compressions"), 2);
    assert_eq!(compressor.state.counter("rounds_dropped"), 6);
    assert_eq!(
        compressor.state.counter("last_tokens_before"),
        event.tokens_before as u64
    );
    assert_eq!(
        compressor.state.counter("last_tokens_after"),
        event.tokens_after as u64
    );

    // A history with no compressible round bumps nothing.
    let mut short = vec![BaseMessage::user("q"), BaseMessage::assistant("a")];
    assert!(compressor
        .process(&mut short, &fixture.context(true))
        .await
        .unwrap()
        .is_none());
    assert_eq!(compressor.state.counter("compressions"), 2);
    assert_eq!(compressor.state.counter("rounds_dropped"), 6);
}

#[tokio::test]
async fn current_round_compressor_counts_a_stage_one_block() {
    let fixture = Fixture::new();
    let compressor = CurrentRoundCompressor::new(
        CurrentRoundCompressorConfig {
            tokens_threshold: 50,
            messages_to_keep: 1,
            min_selected_tokens_for_compression: 10,
            compression_target_tokens: 100,
            ..CurrentRoundCompressorConfig::default()
        },
        None,
    );
    let mut messages = vec![
        BaseMessage::user("q"),
        BaseMessage::assistant("").with_tool_calls(vec![ToolCall::new("read_file").with_id("c1")]),
        BaseMessage::tool("c1", long("result")),
        BaseMessage::assistant("halfway"),
        BaseMessage::assistant("done"),
    ];
    let event = compressor
        .process(&mut messages, &fixture.context(true))
        .await
        .unwrap()
        .unwrap();

    assert_eq!(compressor.state.counter("compressions"), 1);
    assert_eq!(compressor.state.counter("merges"), 0);
    assert_eq!(
        compressor.state.counter("last_tokens_after"),
        event.tokens_after as u64
    );

    // Without a user message there is nothing to compress.
    let mut headless = vec![BaseMessage::assistant("a"), BaseMessage::assistant("b")];
    assert!(compressor
        .process(&mut headless, &fixture.context(true))
        .await
        .unwrap()
        .is_none());
    assert_eq!(compressor.state.counter("compressions"), 1);
}

#[tokio::test]
async fn current_round_compressor_counts_a_merge() {
    let fixture = Fixture::new();
    let compressor = CurrentRoundCompressor::new(
        CurrentRoundCompressorConfig {
            compression_target_tokens: 100,
            summary_merge_min_blocks: 3,
            ..CurrentRoundCompressorConfig::default()
        },
        None,
    );
    let body = "detail ".repeat(40);
    let block = |suffix: &str| {
        BaseMessage::user(MemoryBlockKind::CurrentRound.render(&format!("{body} {suffix}")))
    };
    let mut messages = vec![
        BaseMessage::user("q"),
        block("one"),
        block("two"),
        block("three"),
        BaseMessage::assistant("done"),
    ];
    assert!(compressor
        .process(&mut messages, &fixture.context(true))
        .await
        .unwrap()
        .is_some());

    assert_eq!(compressor.state.counter("compressions"), 0);
    assert_eq!(compressor.state.counter("merges"), 1);
    // The three blocks collapsed into exactly one block.
    assert_eq!(
        messages
            .iter()
            .filter(|message| message.text().starts_with(CURRENT_ROUND_MEMORY_BLOCK))
            .count(),
        1
    );
}

#[tokio::test]
async fn round_level_compressor_counts_its_ladder() {
    let fixture = Fixture::new();
    let compressor = RoundLevelCompressor::new(
        RoundLevelCompressorConfig {
            target_total_tokens: 400,
            keep_recent_messages: 1,
            first_pass_target_tokens: 200,
            second_pass_target_tokens: 150,
            third_pass_target_tokens: 150,
            ..RoundLevelCompressorConfig::default()
        },
        None,
    );
    let mut messages = history(3, "read_file");
    let event = compressor
        .process(&mut messages, &fixture.context(true))
        .await
        .unwrap()
        .unwrap();

    assert_eq!(compressor.state.counter("compressions"), 1);
    assert!(compressor.state.counter("blocks") >= 1);
    assert_eq!(
        compressor.state.counter("last_tokens_before"),
        event.tokens_before as u64
    );
    assert!(
        compressor.state.counter("last_tokens_after")
            < compressor.state.counter("last_tokens_before")
    );

    // A history already inside the target is a no-op.
    let mut small = history(1, "read_file");
    let before = small.clone();
    let tight = RoundLevelCompressor::new(
        RoundLevelCompressorConfig {
            target_total_tokens: 10_000_000,
            ..RoundLevelCompressorConfig::default()
        },
        None,
    );
    assert!(tight
        .process(&mut small, &fixture.context(true))
        .await
        .unwrap()
        .is_none());
    assert_eq!(small, before);
    assert_eq!(tight.save_state(), Value::Null);
}

#[tokio::test]
async fn full_compact_compressor_counts_a_compaction() {
    let fixture = Fixture::new();
    let compressor = FullCompactProcessor::new(
        FullCompactProcessorConfig {
            trigger_total_tokens: 500,
            compression_call_max_tokens: 200,
            messages_to_keep: 0,
            ..FullCompactProcessorConfig::default()
        },
        None,
        None,
    );
    // Six long messages: longer than the deterministic offline summary, so the
    // compaction is a real saving.
    let mut messages = Vec::new();
    for index in 1..=3 {
        messages.push(BaseMessage::user(long(&format!("u{index}"))));
        messages.push(BaseMessage::assistant(long(&format!("a{index}"))));
    }
    let event = compressor
        .process(&mut messages, &fixture.context(false))
        .await
        .unwrap()
        .unwrap();

    assert_eq!(compressor.state.counter("compactions"), 1);
    assert_eq!(compressor.state.counter("session_memory_compactions"), 0);
    assert_eq!(
        compressor.state.counter("last_tokens_before"),
        event.tokens_before as u64
    );

    // A history under the trigger is left alone.
    let mut short = vec![BaseMessage::user("q"), BaseMessage::assistant("a")];
    assert!(compressor
        .process(&mut short, &fixture.context(false))
        .await
        .unwrap()
        .is_none());
    assert_eq!(compressor.state.counter("compactions"), 1);
}

#[tokio::test]
async fn micro_compact_counts_cleared_results_and_groups() {
    let fixture = Fixture::new();
    let compressor = MicroCompactProcessor::new(MicroCompactProcessorConfig {
        trigger_threshold: 1,
        compactable_tool_names: vec!["grep".to_owned()],
        keep_recent_per_tool: 1,
        ..MicroCompactProcessorConfig::default()
    });
    let mut messages = history(3, "grep");
    assert!(compressor
        .process(&mut messages, &fixture.context(true))
        .await
        .unwrap()
        .is_some());

    assert_eq!(compressor.state.counter("cleared"), 2);
    assert_eq!(compressor.state.counter("groups"), 1);

    // A history whose results already fit the window is a no-op.
    let mut small = history(1, "grep");
    let before = small.clone();
    assert!(compressor
        .process(&mut small, &fixture.context(true))
        .await
        .unwrap()
        .is_none());
    assert_eq!(small, before);
    assert_eq!(compressor.state.counter("cleared"), 2);
    assert_eq!(compressor.state.counter("groups"), 1);
}

#[tokio::test]
async fn message_offloader_counts_offloaded_records() {
    let fixture = Fixture::new();
    let offloader = MessageOffloader::new(
        MessageOffloaderConfig {
            tokens_threshold: 10,
            large_message_threshold: 100,
            protected_tool_names: vec!["read_file".to_owned()],
            trim_size: 20,
            ..MessageOffloaderConfig::default()
        },
        Arc::clone(&fixture.store),
    );
    let mut messages = history(1, "grep");
    let event = offloader
        .process(&mut messages, &fixture.context(true))
        .await
        .unwrap()
        .unwrap();

    assert_eq!(offloader.state.counter("offloaded"), 1);
    assert_eq!(offloader.state.counter("trimmed"), 0);
    assert_eq!(
        offloader.state.counter("last_tokens_before"),
        event.tokens_before as u64
    );
    assert_eq!(
        offloader.state.counter("last_tokens_after"),
        event.tokens_after as u64
    );

    // The second pass finds every candidate already marked.
    assert!(offloader
        .process(&mut messages, &fixture.context(true))
        .await
        .unwrap()
        .is_none());
    assert_eq!(offloader.state.counter("offloaded"), 1);
}

#[tokio::test]
async fn tool_result_budget_counts_offloaded_results() {
    let fixture = Fixture::new();
    let budget = ToolResultBudgetProcessor::new(
        ToolResultBudgetProcessorConfig {
            tokens_threshold: 5_000,
            large_message_threshold: 500,
            trim_size: 50,
            tool_name_allowlist: None,
        },
        Arc::clone(&fixture.store),
    );
    let mut messages = vec![
        BaseMessage::user("run"),
        BaseMessage::assistant("").with_tool_calls(vec![
            ToolCall::new("grep").with_id("c1"),
            ToolCall::new("grep").with_id("c2"),
        ]),
        BaseMessage::tool("c1", "a".repeat(2_000)),
        BaseMessage::tool("c2", "b".repeat(6_000)),
        BaseMessage::assistant("done"),
    ];
    let event = budget
        .process(&mut messages, &fixture.context(false))
        .await
        .unwrap()
        .unwrap();

    assert_eq!(budget.state.counter("offloaded"), 1);
    assert_eq!(budget.state.counter("trimmed"), 0);
    assert_eq!(
        budget.state.counter("last_tokens_before"),
        event.tokens_before as u64
    );

    // The round now fits, so a second pass changes nothing.
    assert!(budget
        .process(&mut messages, &fixture.context(false))
        .await
        .unwrap()
        .is_none());
    assert_eq!(budget.state.counter("offloaded"), 1);
}

#[tokio::test]
async fn tool_result_window_counts_persisted_and_compacted_results() {
    let fixture = Fixture::new();
    let window = ToolResultWindowProcessor::new(
        ToolResultWindowProcessorConfig {
            tool_names: vec!["grep".to_owned()],
            keep_last_k: 1,
            trim_size: 50,
            min_offload_chars: 0,
            small_result_trim_size: 100,
        },
        Arc::clone(&fixture.store),
    );
    let mut messages = history(2, "grep");
    assert!(window
        .process(&mut messages, &fixture.context(true))
        .await
        .unwrap()
        .is_some());
    assert_eq!(window.state.counter("offloaded"), 1);
    assert_eq!(window.state.counter("trimmed"), 0);

    // A result below the offload threshold is compacted in place instead.
    let compacting = ToolResultWindowProcessor::new(
        ToolResultWindowProcessorConfig {
            tool_names: vec!["grep".to_owned()],
            keep_last_k: 1,
            trim_size: 50,
            min_offload_chars: 100_000,
            small_result_trim_size: 100,
        },
        Arc::clone(&fixture.store),
    );
    let mut small = history(2, "grep");
    assert!(compacting
        .process(&mut small, &fixture.context(true))
        .await
        .unwrap()
        .is_some());
    assert_eq!(compacting.state.counter("offloaded"), 0);
    assert_eq!(compacting.state.counter("trimmed"), 1);

    // Nothing above the window: no counters.
    let mut single = history(1, "grep");
    assert!(window
        .process(&mut single, &fixture.context(true))
        .await
        .unwrap()
        .is_none());
    assert_eq!(window.state.counter("offloaded"), 1);
}

#[tokio::test]
async fn summary_offloader_counts_summaries() {
    let fixture = Fixture::new();
    let offloader = MessageSummaryOffloader::new(
        MessageSummaryOffloaderConfig {
            large_message_threshold: 100,
            protected_tool_names: vec!["read_file".to_owned()],
            summary_max_tokens: 900,
            content_max_chars_for_compression: 300,
            ..MessageSummaryOffloaderConfig::default()
        },
        None,
        Arc::clone(&fixture.store),
    );
    let mut messages = history(1, "grep");
    let event = offloader
        .process(&mut messages, &fixture.context(false))
        .await
        .unwrap()
        .unwrap();

    assert_eq!(offloader.state.counter("offloaded"), 1);
    assert_eq!(offloader.state.counter("summaries"), 1);
    assert_eq!(
        offloader.state.counter("last_tokens_after"),
        event.tokens_after as u64
    );

    // The rewritten message carries a marker, so a second pass skips it.
    assert!(offloader
        .process(&mut messages, &fixture.context(false))
        .await
        .unwrap()
        .is_none());
    assert_eq!(offloader.state.counter("offloaded"), 1);
    assert_eq!(offloader.state.counter("summaries"), 1);
}

/// The engine configuration the persistence test uses.
///
/// Two stateful processors: micro compaction clears stale tool results first and
/// the round-level compressor then shrinks what is left.
fn engine_config() -> ContextConfig {
    ContextConfig {
        max_context_tokens: Some(200),
        token_counter: TokenCounterConfig {
            kind: TokenCounterKind::Characters,
            chars_per_token: 1,
        },
        processors: vec![
            ProcessorConfig::MicroCompact(MicroCompactProcessorConfig {
                trigger_threshold: 1,
                compactable_tool_names: vec!["grep".to_owned()],
                keep_recent_per_tool: 1,
                ..MicroCompactProcessorConfig::default()
            }),
            ProcessorConfig::RoundLevel(RoundLevelCompressorConfig {
                target_total_tokens: 400,
                keep_recent_messages: 1,
                first_pass_target_tokens: 200,
                second_pass_target_tokens: 150,
                third_pass_target_tokens: 150,
                ..RoundLevelCompressorConfig::default()
            }),
        ],
        ..ContextConfig::default()
    }
}

#[tokio::test]
async fn engine_persists_processor_counters_through_the_session() {
    let session: Arc<dyn Session> = Arc::new(AgentSession::new("s"));
    let engine = ContextEngine::new(engine_config());
    let context = engine
        .create_context("main", Some(Arc::clone(&session)))
        .await
        .unwrap();
    for message in history(3, "grep") {
        engine.observe(&context, message).await.unwrap();
    }

    let report = engine.compress(&context).await.unwrap();
    assert_eq!(report.status, CompressionStatus::Compressed);

    let saved = context.save_state().await;
    assert!(saved.processors.contains_key("micro_compact"));
    assert!(saved.processors.contains_key("round_level"));
    assert!(
        saved.processors["micro_compact"]["cleared"]
            .as_u64()
            .unwrap()
            >= 1
    );
    assert!(
        saved.processors["round_level"]["compressions"]
            .as_u64()
            .unwrap()
            >= 1
    );

    assert_eq!(engine.save_contexts(session.as_ref()).await.unwrap(), 1);

    // A brand-new engine restores both the history and the counters.
    let restored_engine = ContextEngine::new(engine_config());
    assert_eq!(
        restored_engine
            .load_contexts(session.as_ref())
            .await
            .unwrap(),
        1
    );
    let restored = restored_engine.get_context("s", "main").await.unwrap();
    assert_eq!(restored.messages().await, context.messages().await);
    let restored_state = restored.save_state().await;
    assert_eq!(restored_state.processors, saved.processors);
    assert_eq!(restored_state.revision, saved.revision);
}

#[test]
fn context_state_without_processors_still_loads() {
    let state: ContextState =
        serde_json::from_str(r#"{"messages": [{"role": "user", "content": "hi"}], "revision": 7}"#)
            .unwrap();
    assert_eq!(state.revision, 7);
    assert_eq!(state.messages.len(), 1);
    assert!(state.processors.is_empty());

    let mut populated = state.clone();
    populated
        .processors
        .insert("round_level".to_owned(), json!({"compressions": 3}));
    let json = serde_json::to_string(&populated).unwrap();
    let round_tripped: ContextState = serde_json::from_str(&json).unwrap();
    assert_eq!(round_tripped, populated);
    assert_eq!(
        round_tripped.processors["round_level"],
        json!({"compressions": 3})
    );
}

#[tokio::test]
async fn context_load_state_ignores_corrupt_and_foreign_entries() {
    let engine = ContextEngine::new(engine_config());
    let context = engine.create_context("main", None).await.unwrap();

    let mut state = ContextState {
        messages: vec![BaseMessage::user("hi")],
        revision: 3,
        ..ContextState::default()
    };
    state
        .processors
        .insert("round_level".to_owned(), json!("not a bag"));
    state.processors.insert("ghost".to_owned(), json!({"n": 1}));
    context.load_state(state).await.unwrap();

    assert_eq!(context.messages().await.len(), 1);
    assert_eq!(context.revision(), 3);
    assert_eq!(context.processors().len(), 2);
    for processor in context.processors() {
        assert_eq!(processor.save_state(), Value::Null);
    }
}

#[tokio::test]
async fn a_corrupt_state_leaves_the_processor_working() {
    let fixture = Fixture::new();
    let compressor = DialogueCompressor::new(
        DialogueCompressorConfig {
            tokens_threshold: 100,
            compression_target_tokens: 200,
            ..DialogueCompressorConfig::default()
        },
        None,
    );
    compressor.load_state(&json!("not a bag"));
    compressor.load_state(&Value::Null);
    assert_eq!(compressor.save_state(), Value::Null);

    compressor.load_state(&json!({"compressions": 5, "rounds_dropped": 12}));
    assert_eq!(compressor.state.counter("compressions"), 5);

    let mut messages = history(3, "read_file");
    assert!(compressor
        .process(&mut messages, &fixture.context(true))
        .await
        .unwrap()
        .is_some());
    assert_eq!(compressor.state.counter("compressions"), 7);
    assert_eq!(compressor.state.counter("rounds_dropped"), 18);
}
