#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Public-API tests for the compressor support module and [`DialogueCompressor`].
//!
//! The suite covers the shared round grouping, the memory-block envelopes, the
//! offline summary fallback, and the dialogue compressor's contract: only older
//! complete rounds are replaced, the anchor user message survives raw, the
//! newest round and every system message are untouched, tool-call/tool-result
//! pairs never split, and the replacement must be a real saving.
//!
//! Everything is offline and deterministic: token counts come from the
//! one-character-per-token counter, and the only model is
//! [`openjiuwen_test_support::ScriptedModel`].
//!
//! [`DialogueCompressor`]: openjiuwen_context::processor::DialogueCompressor

use std::sync::Arc;

use openjiuwen_context::processor::{
    find_last_completed_round_end, group_completed_rounds, is_memory_block,
    iter_summary_merge_ranges, summarize_offline, ContextPipeline, ContextProcessor,
    CurrentRoundCompressor, CurrentRoundCompressorConfig, DialogueCompressor,
    DialogueCompressorConfig, MemoryBlockKind, ProcessorConfig, ProcessorContext,
    CURRENT_ROUND_MEMORY_BLOCK, DIALOGUE_MEMORY_BLOCK, ROUND_LEVEL_MEMORY_BLOCK,
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

/// Two compressible rounds, then the protected newest round.
fn history() -> Vec<BaseMessage> {
    let mut messages = vec![BaseMessage::system("persona")];
    for index in 1..=3 {
        messages.extend(tool_round(index));
    }
    messages
}

/// A configuration that compresses aggressively, so tests exercise structure
/// rather than arithmetic.
fn config() -> DialogueCompressorConfig {
    DialogueCompressorConfig {
        tokens_threshold: 100,
        compression_target_tokens: 200,
        ..DialogueCompressorConfig::default()
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

#[test]
fn rounds_are_split_by_user_turns_and_tool_rounds_count() {
    let messages = history();
    let rounds = group_completed_rounds(&messages);
    // Each round carries a tool exchange; the leading system message belongs to
    // none of them.
    assert_eq!(rounds, vec![1..5, 5..9, 9..13]);

    let with_trailing_user = {
        let mut messages = messages.clone();
        messages.push(BaseMessage::user("u4"));
        messages
    };
    // The trailing user turn is incomplete and therefore not a round.
    assert_eq!(
        group_completed_rounds(&with_trailing_user),
        vec![1..5, 5..9, 9..13]
    );

    // A contiguous user run opens one round at its first member.
    let run = vec![
        BaseMessage::user("q1"),
        BaseMessage::user("q2"),
        BaseMessage::assistant("a"),
    ];
    assert_eq!(group_completed_rounds(&run), vec![0..3]);
}

#[test]
fn the_last_completed_round_end_clamps_a_span() {
    let messages = history();
    // Inside a tool exchange: no completed round, so no safe end.
    assert_eq!(find_last_completed_round_end(&messages, 2, 3), None);
    assert_eq!(find_last_completed_round_end(&messages, 10, 11), None);
    // Through the closing assistant of the first round.
    assert_eq!(find_last_completed_round_end(&messages, 2, 4), Some(5));
    // Past the second round, which closes at index 8.
    assert_eq!(find_last_completed_round_end(&messages, 2, 10), Some(9));
    assert_eq!(find_last_completed_round_end(&messages, 20, 20), None);
}

#[test]
fn envelopes_render_the_contract_and_round_trip_the_summary() {
    let text = MemoryBlockKind::Dialogue.render("carry over");
    assert_eq!(text.lines().next(), Some(DIALOGUE_MEMORY_BLOCK));
    assert!(text.contains("processor: DialogueCompressor"));
    assert!(text.contains("scope: historical_dialogue_block"));
    assert!(text.ends_with("\nSummary:\ncarry over"));
    assert_eq!(
        MemoryBlockKind::Dialogue.summary_body(&text),
        Some("carry over")
    );
    assert_eq!(MemoryBlockKind::CurrentRound.summary_body(&text), None);
    assert!(is_memory_block(
        &BaseMessage::user(text.clone()),
        DIALOGUE_MEMORY_BLOCK
    ));
    assert!(!is_memory_block(
        &BaseMessage::assistant(text.clone()),
        DIALOGUE_MEMORY_BLOCK
    ));

    let current = MemoryBlockKind::CurrentRound.render("state");
    assert_eq!(current.lines().next(), Some(CURRENT_ROUND_MEMORY_BLOCK));
    assert!(current.contains("processor: CurrentRoundCompressor"));
    assert!(current.contains("scope: current_round_increment"));
    assert!(current.contains("strategy_status: "));
    assert!(current.contains("tool_action_state_status: "));
    assert_eq!(
        MemoryBlockKind::CurrentRound.summary_body(&current),
        Some("state")
    );

    let round = MemoryBlockKind::RoundLevel.render("short");
    assert_eq!(round.lines().next(), Some(ROUND_LEVEL_MEMORY_BLOCK));
    assert!(round.contains("scope: round_level"));
    assert!(round.contains("Historical fallback context only."));
    let scoped = MemoryBlockKind::RoundLevel.render_with_scope("short", "custom_scope");
    assert!(scoped.contains("scope: custom_scope"));
}

#[test]
fn block_runs_and_offline_summaries_are_deterministic() {
    let block = |text: &str| BaseMessage::user(MemoryBlockKind::CurrentRound.render(text));
    let messages = vec![
        BaseMessage::user("raw"),
        block("a"),
        block("b"),
        BaseMessage::assistant("milestone"),
        block("c"),
        block("d"),
        block("e"),
    ];
    assert_eq!(
        iter_summary_merge_ranges(&messages, CURRENT_ROUND_MEMORY_BLOCK, 3),
        vec![4..7]
    );
    assert_eq!(
        iter_summary_merge_ranges(&messages, CURRENT_ROUND_MEMORY_BLOCK, 2),
        vec![1..3, 4..7]
    );

    let transcript = vec![
        BaseMessage::user("hello"),
        BaseMessage::assistant("").with_tool_calls(vec![ToolCall::new("read_file").with_id("c1")]),
        BaseMessage::tool("c1", "body"),
    ];
    let first = summarize_offline(&transcript, 10_000);
    assert_eq!(first, summarize_offline(&transcript, 10_000));
    assert!(first.starts_with("[1] user: hello"));
    assert!(first.contains("[2] assistant:  | tool_calls=read_file()"));
    assert!(first.contains("[3] tool: body | tool_call_id=c1"));
    let truncated = summarize_offline(&transcript, 30);
    assert!(truncated.chars().count() < first.chars().count());
    assert_eq!(summarize_offline(&[], 30), "");
}

#[tokio::test]
async fn compresses_older_rounds_only() {
    let fixture = Fixture::new();
    let compressor = DialogueCompressor::new(config(), None);
    let mut messages = history();
    let before_tokens = count_messages(&fixture.counter, &messages);

    let event = compressor
        .process(&mut messages, &fixture.context(100, false))
        .await
        .unwrap()
        .unwrap();

    assert_eq!(event.processor, "dialogue");
    assert_eq!(event.kind, ContextEventKind::Summarized);
    // The tool exchange of rounds one and two, without their anchor user turns.
    assert_eq!(event.messages_changed, vec![2, 3, 4, 6, 7, 8]);
    assert_eq!(event.tokens_before, before_tokens);
    let summary = event.summary.clone().unwrap();
    assert!(summary.starts_with(DIALOGUE_MEMORY_BLOCK));

    assert_eq!(messages.len(), 9);
    // Every system message survives.
    assert_eq!(messages[0].role, Role::System);
    // The anchor user message of each compressed round survives raw.
    assert_eq!(messages[1].text().as_ref(), "u1");
    assert_eq!(messages[3].text().as_ref(), "u2");
    // Each replaced span is exactly one memory block.
    assert!(is_memory_block(&messages[2], DIALOGUE_MEMORY_BLOCK));
    assert!(is_memory_block(&messages[4], DIALOGUE_MEMORY_BLOCK));
    // The newest round is untouched, tool exchange included.
    assert_eq!(messages[5].text().as_ref(), "u3");
    assert_eq!(messages[6].tool_calls.len(), 1);
    assert_eq!(messages[7].text().as_ref(), long("tool output 3"));
    assert_eq!(messages[8].text().as_ref(), "final3");

    // The replaced tool exchanges left no orphans behind: both sides went
    // together.
    assert_eq!(validate_tool_pairs(&messages), Vec::<usize>::new());
    // Only the newest round's tool result is still a raw tool message.
    assert_eq!(
        messages
            .iter()
            .filter(|message| message.role == Role::Tool)
            .count(),
        1
    );
    assert!(messages.iter().all(|message| {
        message.text().as_ref() != long("tool output 1")
            && message.text().as_ref() != long("tool output 2")
    }));
    assert!(event.tokens_after < event.tokens_before);
}

#[tokio::test]
async fn the_benefit_gate_rejects_a_longer_replacement() {
    let fixture = Fixture::new();
    let compressor = DialogueCompressor::new(config(), None);
    // One short tool round that is a valid target, but the envelope costs more
    // than the messages it would replace.
    let mut messages = vec![
        BaseMessage::user("u1"),
        BaseMessage::assistant("").with_tool_calls(vec![ToolCall::new("read_file").with_id("c1")]),
        BaseMessage::tool("c1", "short"),
        BaseMessage::assistant("final1"),
        BaseMessage::user("u2"),
        BaseMessage::assistant("newest"),
    ];
    let before = messages.clone();
    let event = compressor
        .process(&mut messages, &fixture.context(100, true))
        .await
        .unwrap();
    assert!(
        event.is_none(),
        "the envelope is larger than the round body"
    );
    assert_eq!(messages, before);
}

#[tokio::test]
async fn below_the_threshold_nothing_happens_and_manual_forces() {
    let fixture = Fixture::new();
    let compressor = DialogueCompressor::new(
        DialogueCompressorConfig {
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
async fn a_message_count_threshold_triggers_the_pass() {
    let fixture = Fixture::new();
    let compressor = DialogueCompressor::new(
        DialogueCompressorConfig {
            messages_threshold: Some(5),
            tokens_threshold: 10_000_000,
            compression_target_tokens: 200,
            ..DialogueCompressorConfig::default()
        },
        None,
    );
    let mut messages = history();
    let event = compressor
        .process(&mut messages, &fixture.context(1, false))
        .await
        .unwrap();
    assert!(event.is_some(), "13 messages exceed the threshold of 5");
}

#[tokio::test]
async fn the_model_summary_lands_inside_the_memory_block() {
    let fixture = Fixture::new();
    let scripted = ScriptedModel::builder()
        .respond_text("MODEL-SUMMARY")
        .build();
    let model: Arc<dyn Model> = Arc::new(scripted.clone());
    let compressor = DialogueCompressor::new(config(), Some(Arc::clone(&model)));
    let mut messages = history();
    let event = compressor
        .process(&mut messages, &fixture.model_context(&model, 100))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(scripted.call_count(), 2);
    // Targets are compressed newest first, so the newest compressed round
    // consumes the first scripted response.
    let block_text = messages[4].text().into_owned();
    assert!(is_memory_block(&messages[4], DIALOGUE_MEMORY_BLOCK));
    assert_eq!(
        MemoryBlockKind::Dialogue.summary_body(&block_text),
        Some("MODEL-SUMMARY")
    );
    assert!(event.summary.unwrap().starts_with(DIALOGUE_MEMORY_BLOCK));
}

#[tokio::test]
async fn compression_is_deterministic() {
    let fixture = Fixture::new();
    let compressor = DialogueCompressor::new(config(), None);
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
    let config = DialogueCompressorConfig {
        messages_threshold: Some(8),
        tokens_threshold: 500,
        messages_to_keep: Some(4),
        keep_last_round: false,
        compression_target_tokens: 400,
        custom_compression_prompt: Some("summarise".to_owned()),
        ..DialogueCompressorConfig::default()
    };
    assert!(config.validate().is_ok());
    let json = serde_json::to_string(&config).unwrap();
    assert_eq!(
        serde_json::from_str::<DialogueCompressorConfig>(&json).unwrap(),
        config
    );
    assert_eq!(
        serde_json::from_str::<DialogueCompressorConfig>("{}").unwrap(),
        DialogueCompressorConfig::default()
    );

    let zero_tokens = DialogueCompressorConfig {
        tokens_threshold: 0,
        ..DialogueCompressorConfig::default()
    };
    assert!(zero_tokens.validate().is_err());
    let inverted = DialogueCompressorConfig {
        messages_threshold: Some(4),
        messages_to_keep: Some(4),
        ..DialogueCompressorConfig::default()
    };
    assert!(inverted.validate().is_err());
    let zero_target = DialogueCompressorConfig {
        compression_target_tokens: 0,
        ..DialogueCompressorConfig::default()
    };
    assert!(zero_target.validate().is_err());
}

#[test]
fn the_pipeline_accepts_both_compressor_configs() {
    let configs = vec![
        ProcessorConfig::DialogueCompressor(DialogueCompressorConfig::default()),
        ProcessorConfig::CurrentRoundCompressor(CurrentRoundCompressorConfig::default()),
    ];
    let json = serde_json::to_string(&configs).unwrap();
    assert!(json.contains("\"type\":\"dialogue_compressor\""));
    assert!(json.contains("\"type\":\"current_round_compressor\""));
    let decoded: Vec<ProcessorConfig> = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded, configs);

    let pipeline =
        ContextPipeline::from_configs(&configs, None, Arc::new(InMemoryOffloadStore::new()))
            .unwrap();
    assert_eq!(pipeline.processors().len(), 2);
    assert_eq!(pipeline.processors()[0].name(), "dialogue");
    assert_eq!(pipeline.processors()[1].name(), "current_round");

    let invalid = vec![ProcessorConfig::DialogueCompressor(
        DialogueCompressorConfig {
            tokens_threshold: 0,
            ..DialogueCompressorConfig::default()
        },
    )];
    assert!(
        ContextPipeline::from_configs(&invalid, None, Arc::new(InMemoryOffloadStore::new()),)
            .is_err()
    );

    // Both compressors are dyn-compatible alongside the existing pipeline.
    let processors: Vec<Arc<dyn ContextProcessor>> = vec![
        Arc::new(DialogueCompressor::new(
            DialogueCompressorConfig::default(),
            None,
        )),
        Arc::new(CurrentRoundCompressor::new(
            CurrentRoundCompressorConfig::default(),
            None,
        )),
    ];
    assert_eq!(processors[0].name(), "dialogue");
    assert_eq!(processors[1].name(), "current_round");
}
