#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Public-API tests for [`CurrentRoundCompressor`].
//!
//! The contract under test: the last user message and the newest
//! `messages_to_keep` messages stay raw, the compressible span is clamped back to
//! the last completed API round so a tool exchange is never split, the span must
//! clear `min_selected_tokens_for_compression`, the replacement must be smaller
//! than the span, and the merge stage collapses a run of at least
//! `summary_merge_min_blocks` memory blocks even when no new increment was
//! compressed.
//!
//! Everything is offline and deterministic: token counts come from the
//! one-character-per-token counter, and the only model is
//! [`openjiuwen_test_support::ScriptedModel`].
//!
//! [`CurrentRoundCompressor`]: openjiuwen_context::processor::CurrentRoundCompressor

use std::sync::Arc;

use openjiuwen_context::processor::{
    is_memory_block, ContextProcessor, CurrentRoundCompressor, CurrentRoundCompressorConfig,
    MemoryBlockKind, ProcessorContext, CURRENT_ROUND_MEMORY_BLOCK, DIALOGUE_MEMORY_BLOCK,
};
use openjiuwen_context::{
    count_messages, validate_tool_pairs, ContextConfig, ContextEventKind, InMemoryOffloadStore,
    OffloadScope, OffloadStore,
};
use openjiuwen_foundation::llm::Model;
use openjiuwen_foundation::prompt::{ApproxTokenCounter, BaseMessage, Role, ToolCall};
use openjiuwen_test_support::ScriptedModel;
use tokio_util::sync::CancellationToken;

/// The one-character-per-token counter every test uses.
fn counter() -> ApproxTokenCounter {
    ApproxTokenCounter::with_chars_per_token(1)
}

/// A body long enough that replacing it with a memory block is a real saving.
fn long(text: &str) -> String {
    format!("{text}: {}", "detail ".repeat(300))
}

/// A current-round block carrying `body`.
fn block(body: &str) -> BaseMessage {
    BaseMessage::user(MemoryBlockKind::CurrentRound.render(body))
}

/// One user request, a tool exchange, then a tail of three messages.
fn history() -> Vec<BaseMessage> {
    vec![
        BaseMessage::user("task"),
        BaseMessage::assistant("")
            .with_tool_calls(vec![ToolCall::new("read_file").with_id("call_1")]),
        BaseMessage::tool("call_1", long("tool output")),
        BaseMessage::assistant("closing"),
        BaseMessage::assistant(long("analysis")),
        BaseMessage::assistant("progress"),
        BaseMessage::assistant("check"),
        BaseMessage::assistant("latest"),
    ]
}

/// A configuration that compresses aggressively, so tests exercise structure
/// rather than arithmetic.
fn config() -> CurrentRoundCompressorConfig {
    CurrentRoundCompressorConfig {
        tokens_threshold: 100,
        messages_to_keep: 3,
        min_selected_tokens_for_compression: 100,
        compression_target_tokens: 200,
        ..CurrentRoundCompressorConfig::default()
    }
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

    fn context(&self, trigger_tokens: usize, manual: bool) -> ProcessorContext<'_> {
        ProcessorContext {
            config: &self.config,
            counter: &self.counter,
            model: None,
            offload: &*self.store,
            scope: &self.scope,
            budget: 100_000,
            trigger_tokens,
            manual,
            cancellation: &self.cancellation,
            recall: None,
            session_memory: None,
            session_memory_agent: None,
        }
    }

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

#[tokio::test]
async fn compresses_the_current_round_before_the_raw_tail() {
    let fixture = Fixture::new();
    let compressor = CurrentRoundCompressor::new(config(), None);
    let mut messages = history();
    let before_tokens = count_messages(&fixture.counter, &messages);

    let event = compressor
        .process(&mut messages, &fixture.context(100, false))
        .await
        .unwrap()
        .unwrap();

    assert_eq!(event.processor, "current_round");
    assert_eq!(event.kind, ContextEventKind::Summarized);
    // The span runs to the raw tail: the tool exchange and everything after it
    // that forms a completed round are compressed together.
    assert_eq!(event.messages_changed, vec![1, 2, 3, 4]);
    assert_eq!(event.tokens_before, before_tokens);
    assert!(event.tokens_after < event.tokens_before);

    assert_eq!(messages.len(), 5);
    // The newest user message survives raw.
    assert_eq!(messages[0].role, Role::User);
    assert_eq!(messages[0].text().as_ref(), "task");
    // One memory block replaces the span.
    assert!(is_memory_block(&messages[1], CURRENT_ROUND_MEMORY_BLOCK));
    let block_text = messages[1].text().into_owned();
    let summary = MemoryBlockKind::CurrentRound
        .summary_body(&block_text)
        .unwrap();
    assert!(summary.contains("tool output"));
    assert_eq!(event.summary.as_deref(), Some(summary));
    // The raw tail is intact, and the tool exchange left no orphan behind.
    assert_eq!(messages[2].text().as_ref(), "progress");
    assert_eq!(messages[3].text().as_ref(), "check");
    assert_eq!(messages[4].text().as_ref(), "latest");
    assert_eq!(validate_tool_pairs(&messages), Vec::<usize>::new());
    assert!(messages
        .iter()
        .all(|message| message.tool_calls.is_empty() && message.tool_call_id.is_none()));
}

#[tokio::test]
async fn a_span_ending_inside_a_tool_round_is_never_split() {
    let fixture = Fixture::new();
    let compressor = CurrentRoundCompressor::new(config(), None);
    // messages_to_keep = 3 leaves [assistant tool_calls, tool result] as the
    // only span; that exchange has no closing assistant yet.
    let mut messages = vec![
        BaseMessage::user("task"),
        BaseMessage::assistant("")
            .with_tool_calls(vec![ToolCall::new("read_file").with_id("call_1")]),
        BaseMessage::tool("call_1", long("tool output")),
        BaseMessage::assistant("progress"),
        BaseMessage::assistant("check"),
        BaseMessage::assistant("latest"),
    ];
    let before = messages.clone();
    let event = compressor
        .process(&mut messages, &fixture.context(100, true))
        .await
        .unwrap();
    assert!(event.is_none());
    assert_eq!(messages, before);
}

#[tokio::test]
async fn the_min_selected_gate_skips_a_small_span() {
    let fixture = Fixture::new();
    let compressor = CurrentRoundCompressor::new(
        CurrentRoundCompressorConfig {
            min_selected_tokens_for_compression: 20_000,
            ..config()
        },
        None,
    );
    let mut messages = history();
    let before = messages.clone();
    let event = compressor
        .process(&mut messages, &fixture.context(100, true))
        .await
        .unwrap();
    assert!(event.is_none(), "the span is far below 20k tokens");
    assert_eq!(messages, before);
}

#[tokio::test]
async fn a_replacement_larger_than_the_span_is_rejected() {
    let fixture = Fixture::new();
    let compressor = CurrentRoundCompressor::new(
        CurrentRoundCompressorConfig {
            min_selected_tokens_for_compression: 1,
            ..config()
        },
        None,
    );
    // The span holds a short tool exchange: the envelope costs more than it.
    let mut messages = vec![
        BaseMessage::user("task"),
        BaseMessage::assistant("")
            .with_tool_calls(vec![ToolCall::new("read_file").with_id("call_1")]),
        BaseMessage::tool("call_1", "r"),
        BaseMessage::assistant("closing"),
        BaseMessage::assistant("a"),
        BaseMessage::assistant("b"),
        BaseMessage::assistant("c"),
    ];
    let before = messages.clone();
    let event = compressor
        .process(&mut messages, &fixture.context(100, true))
        .await
        .unwrap();
    assert!(event.is_none());
    assert_eq!(messages, before);
}

#[tokio::test]
async fn the_merge_stage_collapses_a_run_of_blocks() {
    let fixture = Fixture::new();
    let compressor = CurrentRoundCompressor::new(config(), None);
    // The blocks belong to an older round; the newest user request opens the
    // current one, so stage 1 has nothing to compress and only stage 2 runs.
    let mut messages = vec![
        BaseMessage::user("earlier task"),
        block("a"),
        block("b"),
        block("c"),
        BaseMessage::user("new task"),
        BaseMessage::assistant("working"),
        BaseMessage::assistant("check"),
        BaseMessage::assistant("latest"),
    ];
    let event = compressor
        .process(&mut messages, &fixture.context(100, true))
        .await
        .unwrap()
        .unwrap();

    assert_eq!(event.processor, "current_round");
    assert_eq!(event.messages_changed, vec![1, 2, 3]);
    assert_eq!(event.summary.as_deref(), Some("a\n---\nb\n---\nc"));
    assert_eq!(messages.len(), 6, "three blocks collapse into one message");
    assert_eq!(messages[0].text().as_ref(), "earlier task");
    let block_text = messages[1].text().into_owned();
    let merged = MemoryBlockKind::CurrentRound
        .summary_body(&block_text)
        .unwrap();
    assert_eq!(merged, "a\n---\nb\n---\nc");
    assert_eq!(messages[2].text().as_ref(), "new task");
    assert_eq!(messages[5].text().as_ref(), "latest");
}

#[tokio::test]
async fn the_merge_gate_skips_runs_that_are_already_small() {
    let fixture = Fixture::new();
    let compressor = CurrentRoundCompressor::new(
        CurrentRoundCompressorConfig {
            compression_target_tokens: 10_000,
            ..config()
        },
        None,
    );
    let mut messages = vec![
        BaseMessage::user("earlier task"),
        block("a"),
        block("b"),
        block("c"),
        BaseMessage::user("new task"),
        BaseMessage::assistant("working"),
        BaseMessage::assistant("check"),
        BaseMessage::assistant("latest"),
    ];
    let before = messages.clone();
    let event = compressor
        .process(&mut messages, &fixture.context(100, true))
        .await
        .unwrap();
    assert!(event.is_none(), "three tiny blocks do not exceed the gate");
    assert_eq!(messages, before);
}

#[tokio::test]
async fn below_the_threshold_nothing_happens_and_manual_forces() {
    let fixture = Fixture::new();
    let compressor = CurrentRoundCompressor::new(
        CurrentRoundCompressorConfig {
            tokens_threshold: 10_000_000,
            ..config()
        },
        None,
    );
    let mut messages = history();

    let idle = compressor
        .process(&mut messages, &fixture.context(1_000_000, false))
        .await
        .unwrap();
    assert!(idle.is_none());
    assert_eq!(messages, history());

    let forced = compressor
        .process(&mut messages, &fixture.context(1_000_000, true))
        .await
        .unwrap();
    assert!(forced.is_some());
}

#[tokio::test]
async fn a_history_without_a_usable_boundary_is_left_alone() {
    let fixture = Fixture::new();
    let compressor = CurrentRoundCompressor::new(config(), None);

    // No user message at all.
    let mut no_user = vec![BaseMessage::assistant("a"), BaseMessage::assistant("b")];
    assert!(compressor
        .process(&mut no_user, &fixture.context(100, true))
        .await
        .unwrap()
        .is_none());

    // The newest user message is the last message.
    let mut user_last = vec![
        BaseMessage::user("a"),
        BaseMessage::assistant("b"),
        BaseMessage::user("c"),
    ];
    let before = user_last.clone();
    assert!(compressor
        .process(&mut user_last, &fixture.context(100, true))
        .await
        .unwrap()
        .is_none());
    assert_eq!(user_last, before);
}

#[tokio::test]
async fn the_model_summary_lands_inside_the_memory_block() {
    let fixture = Fixture::new();
    let scripted = ScriptedModel::builder()
        .respond_text("MODEL-SUMMARY")
        .build();
    let model: Arc<dyn Model> = Arc::new(scripted.clone());
    let compressor = CurrentRoundCompressor::new(config(), Some(Arc::clone(&model)));
    let mut messages = history();
    let event = compressor
        .process(&mut messages, &fixture.model_context(&model, 100))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(scripted.call_count(), 1);
    let block_text = messages[1].text().into_owned();
    let summary = MemoryBlockKind::CurrentRound
        .summary_body(&block_text)
        .unwrap();
    assert_eq!(summary, "MODEL-SUMMARY");
    assert_eq!(event.summary.as_deref(), Some("MODEL-SUMMARY"));
    assert!(is_memory_block(&messages[1], CURRENT_ROUND_MEMORY_BLOCK));
    assert!(!is_memory_block(&messages[1], DIALOGUE_MEMORY_BLOCK));
}

#[tokio::test]
async fn compression_is_deterministic() {
    let fixture = Fixture::new();
    let compressor = CurrentRoundCompressor::new(config(), None);
    let mut first = history();
    let mut second = history();
    let first_event = compressor
        .process(&mut first, &fixture.context(100, false))
        .await
        .unwrap()
        .unwrap();
    let second_event = compressor
        .process(&mut second, &fixture.context(100, false))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(first, second);
    assert_eq!(first_event, second_event);
}

#[test]
fn config_round_trips_through_json_and_validates() {
    let config = CurrentRoundCompressorConfig {
        tokens_threshold: 500,
        messages_to_keep: 2,
        min_selected_tokens_for_compression: 50,
        compression_target_tokens: 100,
        summary_merge_min_blocks: 4,
        prior_context_window_size: 5,
        custom_compression_prompt: Some("summarise".to_owned()),
        ..CurrentRoundCompressorConfig::default()
    };
    assert!(config.validate().is_ok());
    let json = serde_json::to_string(&config).unwrap();
    assert_eq!(
        serde_json::from_str::<CurrentRoundCompressorConfig>(&json).unwrap(),
        config
    );
    assert_eq!(
        serde_json::from_str::<CurrentRoundCompressorConfig>("{}").unwrap(),
        CurrentRoundCompressorConfig::default()
    );

    let single_block = CurrentRoundCompressorConfig {
        summary_merge_min_blocks: 1,
        ..CurrentRoundCompressorConfig::default()
    };
    assert!(single_block.validate().is_err());
    let zero_tokens = CurrentRoundCompressorConfig {
        tokens_threshold: 0,
        ..CurrentRoundCompressorConfig::default()
    };
    assert!(zero_tokens.validate().is_err());
    let zero_target = CurrentRoundCompressorConfig {
        compression_target_tokens: 0,
        ..CurrentRoundCompressorConfig::default()
    };
    assert!(zero_target.validate().is_err());
}
