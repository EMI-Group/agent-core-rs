#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Public-API tests for [`RoundLevelCompressor`].
//!
//! The suite pins the fallback ladder: completed rounds become one block each,
//! phase 2 merges contiguous blocks, phase 3 compresses the rounds the keep
//! window protected, and phase 4 collapses what is left into one truncated
//! block. Every step must be a real saving, system messages must survive, tool
//! pairs must never split, and a second run on the same input must be stable.
//!
//! Everything is offline and deterministic: token counts come from the
//! one-character-per-token counter, and the only model is
//! [`openjiuwen_test_support::ScriptedModel`].
//!
//! [`RoundLevelCompressor`]: openjiuwen_context::processor::RoundLevelCompressor
use std::sync::Arc;

use openjiuwen_context::processor::{
    is_memory_block, ContextProcessor, ProcessorConfig, ProcessorContext, RoundLevelCompressor,
    RoundLevelCompressorConfig, ROUND_LEVEL_MEMORY_BLOCK, TRUNCATED_FULL_CONTEXT_SCOPE,
};
use openjiuwen_context::{
    count_messages, validate_tool_pairs, ContextConfig, ContextEventKind, InMemoryOffloadStore,
    OffloadScope, OffloadStore,
};
use openjiuwen_foundation::llm::Model;
use openjiuwen_foundation::prompt::{ApproxTokenCounter, BaseMessage, Role, ToolCall};
use openjiuwen_test_support::{ScriptedModel, ScriptedTurn};
use tokio_util::sync::CancellationToken;

/// The one-character-per-token counter every test uses.
fn counter() -> ApproxTokenCounter {
    ApproxTokenCounter::with_chars_per_token(1)
}

/// A body long enough that replacing it with a memory block is a real saving.
fn long(text: &str) -> String {
    format!("{text}: {}", "detail ".repeat(300))
}

/// One round that exercises a tool exchange, so it is longer than two messages.
fn tool_round(index: usize) -> Vec<BaseMessage> {
    vec![
        BaseMessage::user(format!("u{index}")),
        BaseMessage::assistant("").with_tool_calls(vec![
            ToolCall::new("read_file").with_id(format!("call_{index}"))
        ]),
        BaseMessage::tool(
            format!("call_{index}"),
            long(&format!("tool output {index}")),
        ),
        BaseMessage::assistant(format!("final{index}")),
    ]
}

/// A persona plus `rounds` tool rounds.
fn history(rounds: usize) -> Vec<BaseMessage> {
    let mut messages = vec![BaseMessage::system("persona")];
    for index in 1..=rounds {
        messages.extend(tool_round(index));
    }
    messages
}

/// A configuration with small pass targets, so the offline summary is a saving.
fn config(target: usize, keep_recent: usize) -> RoundLevelCompressorConfig {
    RoundLevelCompressorConfig {
        target_total_tokens: target,
        keep_recent_messages: keep_recent,
        first_pass_target_tokens: 200,
        second_pass_target_tokens: 150,
        third_pass_target_tokens: 150,
        ..RoundLevelCompressorConfig::default()
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

    fn context(&self, budget: usize, manual: bool) -> ProcessorContext<'_> {
        ProcessorContext {
            config: &self.config,
            counter: &self.counter,
            model: None,
            offload: &*self.store,
            scope: &self.scope,
            budget,
            trigger_tokens: budget,
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
        budget: usize,
    ) -> ProcessorContext<'a> {
        ProcessorContext {
            config: &self.config,
            counter: &self.counter,
            model: Some(model),
            offload: &*self.store,
            scope: &self.scope,
            budget,
            trigger_tokens: budget,
            manual: false,
            cancellation: &self.cancellation,
            recall: None,
            session_memory: None,
            session_memory_agent: None,
        }
    }
}

/// The blocks (user messages with the round-level marker) in `messages`.
fn blocks(messages: &[BaseMessage]) -> Vec<&BaseMessage> {
    messages
        .iter()
        .filter(|message| is_memory_block(message, ROUND_LEVEL_MEMORY_BLOCK))
        .collect()
}

/// The `scope:` line of a block.
fn scope_of(message: &BaseMessage) -> String {
    message
        .text()
        .lines()
        .find(|line| line.starts_with("scope: "))
        .unwrap_or_default()
        .trim_start_matches("scope: ")
        .to_owned()
}

#[tokio::test]
async fn completed_rounds_become_one_block_each() {
    let fixture = Fixture::new();
    let compressor = RoundLevelCompressor::new(config(5_000, 0), None);
    let mut messages = history(3);
    let before = count_messages(&fixture.counter, &messages);

    let event = compressor
        .process(&mut messages, &fixture.context(1_000, false))
        .await
        .unwrap()
        .expect("the ladder must run above the target");

    // Persona plus one block per round; system messages survive.
    assert_eq!(messages.len(), 4);
    assert_eq!(messages[0].role, Role::System);
    assert_eq!(messages[0].text(), "persona");
    let blocks = blocks(&messages);
    assert_eq!(blocks.len(), 3);
    for block in &blocks {
        assert_eq!(scope_of(block), "round_level");
    }
    // No tool-call/tool-result pair was split, because every one of them was
    // inside a replaced round.
    assert_eq!(validate_tool_pairs(&messages), Vec::<usize>::new());

    assert_eq!(event.processor, "round_level");
    assert_eq!(event.kind, ContextEventKind::Summarized);
    assert_eq!(event.tokens_before, before);
    assert_eq!(
        event.tokens_after,
        count_messages(&fixture.counter, &messages)
    );
    assert!(event.tokens_after < event.tokens_before);
    assert_ne!(event.messages_changed, Vec::<usize>::new());
    assert!(event
        .summary
        .as_deref()
        .is_some_and(|text| !text.is_empty()));
}

#[tokio::test]
async fn a_history_under_the_target_is_left_alone() {
    let fixture = Fixture::new();
    let compressor = RoundLevelCompressor::new(RoundLevelCompressorConfig::default(), None);
    let mut messages = history(2);
    let untouched = messages.clone();
    assert!(compressor
        .process(&mut messages, &fixture.context(1_000, false))
        .await
        .unwrap()
        .is_none());
    assert_eq!(messages, untouched);
}

#[tokio::test]
async fn the_ratio_gates_a_pass_but_manual_bypasses_it() {
    let fixture = Fixture::new();
    let compressor = RoundLevelCompressor::new(config(2_000, 0), None);
    let mut messages = history(3);
    let untouched = messages.clone();

    // A huge budget keeps the ratio threshold above the history size.
    assert!(compressor
        .process(&mut messages, &fixture.context(100_000_000, false))
        .await
        .unwrap()
        .is_none());
    assert_eq!(messages, untouched);

    // Manual compression ignores the ratio but still honours the target.
    let event = compressor
        .process(&mut messages, &fixture.context(100_000_000, true))
        .await
        .unwrap();
    assert!(event.is_some());
    assert!(
        count_messages(&fixture.counter, &messages) < count_messages(&fixture.counter, &untouched)
    );

    // Manual is not a free pass: under the target nothing happens.
    let mut small = vec![BaseMessage::user("hi"), BaseMessage::assistant("ho")];
    let untouched = small.clone();
    assert!(compressor
        .process(&mut small, &fixture.context(1, true))
        .await
        .unwrap()
        .is_none());
    assert_eq!(small, untouched);
}

#[tokio::test]
async fn a_summary_longer_than_its_round_is_rejected() {
    let fixture = Fixture::new();
    // Every call answers with a summary that dwarfs its round.
    let scripted = ScriptedModel::new([] as [ScriptedTurn; 0]);
    scripted.set_fallback(ScriptedTurn::text("x".repeat(50_000)));
    let model: Arc<dyn Model> = Arc::new(scripted);
    let compressor = RoundLevelCompressor::new(config(3_000, 0), None);
    let mut messages = history(3);

    let event = compressor
        .process(&mut messages, &fixture.model_context(&model, 1_000))
        .await
        .unwrap()
        .expect("the ladder still hard-truncates");

    // The oversized summaries were rejected by the benefit gate, so the only
    // block left is the phase-4 truncation of the whole prefix.
    let blocks = blocks(&messages);
    assert_eq!(blocks.len(), 1);
    assert_eq!(scope_of(blocks[0]), TRUNCATED_FULL_CONTEXT_SCOPE);
    assert!(!blocks[0].text().contains("xxxxx"));
    assert_eq!(messages[0].role, Role::System);
    assert_eq!(validate_tool_pairs(&messages), Vec::<usize>::new());
    assert!(event.tokens_after <= 3_000);
    assert!(event.tokens_after < event.tokens_before);
}

#[tokio::test]
async fn contiguous_blocks_are_merged_in_phase_two() {
    let fixture = Fixture::new();
    let mut open = config(6_000, 0);
    // A generous merge cap keeps the joined bodies, so the separator survives.
    open.third_pass_target_tokens = 10_000;
    let compressor = RoundLevelCompressor::new(open, None);
    let mut messages = history(6);

    let event = compressor
        .process(&mut messages, &fixture.context(1_000, false))
        .await
        .unwrap()
        .expect("the ladder must run");

    assert_eq!(messages[0].role, Role::System);
    let blocks = blocks(&messages);
    assert_eq!(blocks.len(), 1);
    assert_eq!(scope_of(blocks[0]), "round_level");
    assert!(blocks[0].text().contains("\n---\n"));
    assert!(event.tokens_after < event.tokens_before);
    assert_eq!(validate_tool_pairs(&messages), Vec::<usize>::new());
}

#[tokio::test]
async fn phase_three_compresses_rounds_the_keep_window_protected() {
    let fixture = Fixture::new();
    // Five messages of the newest round are protected by phase 1; the earlier
    // completed round that survives in that tail is phase 3's target.
    let compressor = RoundLevelCompressor::new(config(8_000, 13), None);
    let mut messages = history(5);

    let event = compressor
        .process(&mut messages, &fixture.context(1_000, false))
        .await
        .unwrap()
        .expect("the ladder must run");

    let blocks = blocks(&messages);
    assert_eq!(blocks.len(), 4);
    assert!(blocks.iter().all(|block| scope_of(block) == "round_level"));
    assert!(event.tokens_after <= 8_000);
    assert!(event.tokens_after < event.tokens_before);
    assert_eq!(validate_tool_pairs(&messages), Vec::<usize>::new());
}

#[tokio::test]
async fn phase_four_truncates_the_prefix_and_survives_the_target() {
    let fixture = Fixture::new();
    // A tiny first-pass target makes every block larger than its round, so only
    // the hard truncation can bring the history down.
    let mut tight = config(2_000, 0);
    tight.first_pass_target_tokens = 250_000;
    tight.second_pass_target_tokens = 250_000;
    let compressor = RoundLevelCompressor::new(tight, None);
    let mut messages = history(4);

    let event = compressor
        .process(&mut messages, &fixture.context(1_000, false))
        .await
        .unwrap()
        .expect("hard truncation must run");

    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].role, Role::System);
    let block = &messages[1];
    assert!(is_memory_block(block, ROUND_LEVEL_MEMORY_BLOCK));
    assert_eq!(scope_of(block), TRUNCATED_FULL_CONTEXT_SCOPE);
    assert!(block.text().contains("...[TRUNCATED]..."));
    assert!(event.tokens_after <= 2_000);
    assert!(event.tokens_after < event.tokens_before);
    assert_eq!(validate_tool_pairs(&messages), Vec::<usize>::new());
}

#[tokio::test]
async fn the_same_input_produces_the_same_output() {
    let fixture = Fixture::new();
    let compressor = RoundLevelCompressor::new(config(5_000, 0), None);
    let mut first = history(3);
    let mut second = first.clone();

    let first_event = compressor
        .process(&mut first, &fixture.context(1_000, false))
        .await
        .unwrap();
    let second_event = compressor
        .process(&mut second, &fixture.context(1_000, false))
        .await
        .unwrap();

    assert_eq!(first, second);
    assert_eq!(first_event, second_event);

    // A second pass over the already compressed history is a no-op.
    let before = first.clone();
    assert!(compressor
        .process(&mut first, &fixture.context(1_000, false))
        .await
        .unwrap()
        .is_none());
    assert_eq!(first, before);
}

#[test]
fn config_round_trips_and_validation_rejects_bad_values() {
    let config = RoundLevelCompressorConfig::default();
    assert!(config.validate().is_ok());
    let json = serde_json::to_string(&config).unwrap();
    assert_eq!(
        serde_json::from_str::<RoundLevelCompressorConfig>(&json).unwrap(),
        config
    );

    let bad_ratio = RoundLevelCompressorConfig {
        trigger_context_ratio: 1.5,
        ..RoundLevelCompressorConfig::default()
    };
    assert!(bad_ratio.validate().is_err());
    let bad_head = RoundLevelCompressorConfig {
        truncate_head_ratio: 0.0,
        ..RoundLevelCompressorConfig::default()
    };
    assert!(bad_head.validate().is_err());
    let bad_size = RoundLevelCompressorConfig {
        second_pass_target_tokens: 0,
        ..RoundLevelCompressorConfig::default()
    };
    assert!(bad_size.validate().is_err());
}

#[test]
fn the_processor_config_tag_builds_the_compressor() {
    let config = ProcessorConfig::RoundLevel(RoundLevelCompressorConfig::default());
    let json = serde_json::to_string(&config).unwrap();
    assert!(json.contains("\"type\":\"round_level\""));
    let decoded: ProcessorConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded, config);
}
