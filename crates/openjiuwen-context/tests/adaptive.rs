#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Public-API tests for adaptive compression strategy selection.
//!
//! The suite covers four layers:
//!
//! - the deterministic ladder ([`AdaptiveStrategy::deterministic`]) — every
//!   branch, its [`StrategySource::DeterministicFallback`] and a non-empty
//!   reason;
//! - the model path ([`AdaptiveStrategy::select`]) — a valid JSON reply, a
//!   fenced/chatty reply, an unknown strategy, garbage, an empty reply and a
//!   provider error, all through [`openjiuwen_test_support::ScriptedModel`],
//!   plus allowlist filtering, the decision-token budget, and the disabled-flag
//!   and small-payload gates;
//! - the memory-block plan — serde, JSON schema, lenient parsing, deterministic
//!   task extraction, and the rendered envelope and recall marker placement;
//! - the compressor integration — flag off is unchanged, `noop` reports nothing,
//!   another strategy is reported without touching the history, and the same
//!   strategy writes the structured sections into the block.
//!
//! Everything is offline and deterministic: token counts come from the
//! one-character-per-token counter and the only model is the scripted fake.
//!
//! [`AdaptiveStrategy::deterministic`]: openjiuwen_context::processor::compressor::adaptive::AdaptiveStrategy::deterministic
//! [`AdaptiveStrategy::select`]: openjiuwen_context::processor::compressor::adaptive::AdaptiveStrategy::select
use std::sync::Arc;

use openjiuwen_context::processor::compressor::adaptive::{
    render_memory_block_from_plan, render_memory_block_from_plan_with_marker, AdaptiveConfig,
    AdaptiveContext, AdaptiveStrategy, AdaptiveStrategyConfig, CompressionStrategy,
    MemoryBlockPlan, StrategyInput, StrategySource,
};
use openjiuwen_context::processor::compressor::RecallArchiveMarker;
use openjiuwen_context::processor::{
    is_memory_block, ContextProcessor, CurrentRoundCompressor, CurrentRoundCompressorConfig,
    DialogueCompressor, DialogueCompressorConfig, MemoryBlockKind, ProcessorContext,
    CURRENT_ROUND_MEMORY_BLOCK, DIALOGUE_MEMORY_BLOCK,
};
use openjiuwen_context::{
    ContextConfig, ContextEventKind, InMemoryOffloadStore, OffloadScope, OffloadStore,
};
use openjiuwen_foundation::llm::Model;
use openjiuwen_foundation::prompt::{ApproxTokenCounter, BaseMessage, ToolCall};
use openjiuwen_test_support::{CallMode, ScriptedModel, ScriptedTurn};
use tokio_util::sync::CancellationToken;

/// The one-character-per-token counter every test uses.
fn counter() -> ApproxTokenCounter {
    ApproxTokenCounter::with_chars_per_token(1)
}

/// A body long enough that replacing it with a memory block is a real saving.
fn long(text: &str) -> String {
    format!("{text}: {}", "detail ".repeat(300))
}

/// One completed round with a tool exchange of `read_file`.
fn tool_round(index: usize) -> Vec<BaseMessage> {
    vec![
        BaseMessage::user(format!("u{index}")),
        BaseMessage::assistant("").with_tool_calls(vec![
            ToolCall::new("read_file").with_id(format!("call_{index}"))
        ]),
        BaseMessage::tool(format!("call_{index}"), long(&format!("output {index}"))),
        BaseMessage::assistant(format!("final{index}")),
    ]
}

/// Three completed tool rounds behind a system message.
///
/// Every round calls `read_file`, so two of the three results are stale.
fn tool_history() -> Vec<BaseMessage> {
    let mut messages = vec![BaseMessage::system("persona")];
    for index in 1..=3 {
        messages.extend(tool_round(index));
    }
    messages
}

/// Two completed rounds without tools.
fn plain_rounds() -> Vec<BaseMessage> {
    vec![
        BaseMessage::user("q1"),
        BaseMessage::assistant("a1"),
        BaseMessage::user("q2"),
        BaseMessage::assistant("a2"),
    ]
}

/// Three tool rounds whose first result starts with a TODO line.
fn todo_history() -> Vec<BaseMessage> {
    let mut messages = vec![BaseMessage::system("persona")];
    messages.extend([
        BaseMessage::user("u0"),
        BaseMessage::assistant("")
            .with_tool_calls(vec![ToolCall::new("read_file").with_id("call_0")]),
        BaseMessage::tool(
            "call_0",
            format!(
                "TODO: a task the extractor would find. {}",
                long("output 0")
            ),
        ),
        BaseMessage::assistant("final0"),
    ]);
    for index in 1..=2 {
        messages.extend(tool_round(index));
    }
    messages
}

/// A history whose last message leaves the round in flight.
fn in_flight_history() -> Vec<BaseMessage> {
    vec![
        BaseMessage::user("q"),
        BaseMessage::assistant("").with_tool_calls(vec![ToolCall::new("read_file").with_id("c1")]),
        BaseMessage::tool("c1", long("body")),
        BaseMessage::assistant("").with_tool_calls(vec![ToolCall::new("grep").with_id("c2")]),
        BaseMessage::tool("c2", long("hits")),
    ]
}

/// An adaptive configuration with the flag on and nothing else changed.
fn adaptive_on() -> AdaptiveConfig {
    AdaptiveConfig {
        enabled: true,
        ..AdaptiveConfig::default()
    }
}

/// An enabled strategy configuration that may consult `model`.
fn model_config(model: Arc<dyn Model>) -> AdaptiveStrategyConfig {
    AdaptiveStrategyConfig {
        enabled: true,
        model: Some(model),
        ..AdaptiveStrategyConfig::default()
    }
}

/// A measured context built from `messages`.
fn measured(messages: &[BaseMessage]) -> AdaptiveContext {
    StrategyInput::new(messages, &counter()).measure()
}

// ---------------------------------------------------------------------------
// The deterministic ladder
// ---------------------------------------------------------------------------

#[test]
fn the_ladder_covers_every_branch_with_a_reason() {
    let strategy = AdaptiveStrategy::new(AdaptiveStrategyConfig::default());
    let cases: Vec<(&str, AdaptiveContext, CompressionStrategy)> = vec![
        (
            "empty",
            AdaptiveContext::default(),
            CompressionStrategy::Noop,
        ),
        (
            "stale tool results",
            measured(&tool_history()),
            CompressionStrategy::MicroCompact,
        ),
        (
            "in-flight round",
            measured(&in_flight_history()),
            CompressionStrategy::CurrentRound,
        ),
        (
            "many rounds",
            measured(&three_plain_rounds()),
            CompressionStrategy::RoundLevel,
        ),
        (
            "few rounds",
            measured(&plain_rounds()),
            CompressionStrategy::DialogueSummary,
        ),
        (
            "over budget",
            AdaptiveContext {
                total_tokens: 10_000,
                messages: 2,
                system_messages: 2,
                budget_tokens: 9_000,
                ..AdaptiveContext::default()
            },
            CompressionStrategy::FullCompact,
        ),
        (
            "nothing dominates",
            AdaptiveContext {
                total_tokens: 10_000,
                messages: 3,
                system_messages: 1,
                completed_rounds: 1,
                completed_round_tokens: 10,
                budget_tokens: 20_000,
                ..AdaptiveContext::default()
            },
            CompressionStrategy::Noop,
        ),
    ];
    for (name, context, expected) in cases {
        let decision = strategy.deterministic(&context);
        assert_eq!(decision.strategy, expected, "ladder branch: {name}");
        assert_eq!(decision.source, StrategySource::DeterministicFallback);
        assert!(
            !decision.reason.is_empty(),
            "ladder branch {name} must explain itself"
        );
        assert!(decision.plan.is_none());
    }
}

/// Three completed rounds without tool results, so no stale result dominates.
fn three_plain_rounds() -> Vec<BaseMessage> {
    vec![
        BaseMessage::user("q1"),
        BaseMessage::assistant("a1"),
        BaseMessage::user("q2"),
        BaseMessage::assistant("a2"),
        BaseMessage::user("q3"),
        BaseMessage::assistant("a3"),
    ]
}

#[test]
fn the_ladder_is_measure_driven_and_configurable() {
    let default = AdaptiveStrategy::new(AdaptiveStrategyConfig::default());
    let strict = AdaptiveStrategy::new(AdaptiveStrategyConfig {
        min_savings_ratio: 0.99,
        ..AdaptiveStrategyConfig::default()
    });
    // The rounds carry half the tokens, so the 15% default already calls that a
    // saving while a 99% ratio refuses it; the budget is not exceeded either way.
    let context = AdaptiveContext {
        total_tokens: 10_000,
        messages: 5,
        system_messages: 1,
        completed_rounds: 2,
        completed_round_tokens: 5_000,
        budget_tokens: 20_000,
        ..AdaptiveContext::default()
    };
    assert_eq!(
        default.deterministic(&context).strategy,
        CompressionStrategy::DialogueSummary
    );
    assert_eq!(
        strict.deterministic(&context).strategy,
        CompressionStrategy::Noop
    );

    let fallback = AdaptiveStrategy::new(AdaptiveStrategyConfig {
        min_savings_ratio: 0.99,
        fallback_strategy: CompressionStrategy::FullCompact,
        ..AdaptiveStrategyConfig::default()
    });
    let decision = fallback.deterministic(&context);
    assert_eq!(decision.strategy, CompressionStrategy::FullCompact);
    assert!(decision.reason.contains("configured fallback"));
}

#[tokio::test]
async fn select_without_a_model_stays_on_the_ladder() {
    let strategy = AdaptiveStrategy::new(AdaptiveStrategyConfig::default());
    let context = measured(&plain_rounds());
    let decision = strategy.select(&context).await;
    assert_eq!(decision.strategy, CompressionStrategy::DialogueSummary);
    assert_eq!(decision.source, StrategySource::DeterministicFallback);
    assert!(!strategy.has_model());
}

// ---------------------------------------------------------------------------
// Measuring the history
// ---------------------------------------------------------------------------

#[test]
fn measurement_reports_rounds_tools_and_memory_blocks() {
    let messages = tool_history();
    let context = StrategyInput::new(&messages, &counter())
        .with_budget(100_000)
        .with_trigger(500)
        .with_history(2, Some(CompressionStrategy::RoundLevel))
        .measure();

    assert_eq!(context.messages, messages.len());
    assert!(context.total_tokens > 0);
    assert_eq!(context.system_messages, 1);
    assert_eq!(context.user_messages, 3);
    assert_eq!(context.assistant_messages, 6);
    assert_eq!(context.tool_messages, 3);
    assert_eq!(context.completed_rounds, 3);
    assert!(context.completed_round_tokens > 0);
    assert!(!context.assistant_in_flight);
    assert_eq!(context.in_flight_round_tokens, 0);
    assert_eq!(context.tool_result_count, 3);
    assert!(context.tool_result_tokens > 0);
    assert_eq!(context.stale_tool_result_count, 2);
    assert!(context.stale_tool_result_tokens > 0);
    assert_eq!(context.memory_blocks, 0);
    assert_eq!(context.recent_compressions, 2);
    assert_eq!(context.last_strategy, Some(CompressionStrategy::RoundLevel));
    assert_eq!(context.effective_budget(), 100_000);
    assert!(!context.is_empty());

    let mut with_block = messages.clone();
    with_block.insert(
        1,
        BaseMessage::user(MemoryBlockKind::Dialogue.render("old")),
    );
    assert_eq!(
        AdaptiveContext::from_messages(&with_block, &counter()).memory_blocks,
        1
    );
    assert_eq!(
        AdaptiveContext::from_messages(&[], &counter()).effective_budget(),
        0
    );
}

// ---------------------------------------------------------------------------
// The model path
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_valid_reply_is_honoured_and_carries_the_plan() {
    let model = ScriptedModel::text(
        r#"{"strategy": "full_compact", "reason": "the bulk is stale",
            "summary": "durable state", "tasks": ["write the docs"],
            "files": ["src/lib.rs"]}"#,
    );
    let strategy = AdaptiveStrategy::new(model_config(Arc::new(model)));
    let decision = strategy.select(&measured(&plain_rounds())).await;

    assert_eq!(decision.strategy, CompressionStrategy::FullCompact);
    assert_eq!(decision.source, StrategySource::Model);
    assert_eq!(decision.reason, "the bulk is stale");
    let plan = decision.plan.expect("the reply carried a plan");
    assert_eq!(plan.summary, "durable state");
    assert_eq!(plan.tasks, vec!["write the docs".to_owned()]);
    assert_eq!(plan.files, vec!["src/lib.rs".to_owned()]);
}

#[tokio::test]
async fn fenced_and_chatty_replies_are_parsed() {
    let model = ScriptedModel::text(
        "Sure — here is my choice:\n```json\n{\"strategy\": \"micro_compact\"}\n```\nDone.",
    );
    let strategy = AdaptiveStrategy::new(model_config(Arc::new(model)));
    let decision = strategy.select(&measured(&plain_rounds())).await;
    assert_eq!(decision.strategy, CompressionStrategy::MicroCompact);
    assert_eq!(decision.source, StrategySource::Model);
    assert!(decision.reason.contains("micro_compact"));
    assert!(decision.plan.is_none());
}

#[tokio::test]
async fn every_model_failure_falls_back_to_the_ladder() {
    let cases: Vec<(&str, ScriptedTurn, &str)> = vec![
        (
            "unknown strategy",
            ScriptedTurn::text("{\"strategy\": \"summarise_everything\"}"),
            "unknown compression strategy",
        ),
        (
            "garbage",
            ScriptedTurn::text("I cannot decide that."),
            "no JSON object",
        ),
        ("empty", ScriptedTurn::text("   "), "empty"),
        (
            "provider error",
            ScriptedTurn::fail("connection reset"),
            "strategy model call failed",
        ),
    ];
    for (name, turn, expected) in cases {
        let model = ScriptedModel::new([turn]);
        let strategy = AdaptiveStrategy::new(model_config(Arc::new(model)));
        let decision = strategy.select(&measured(&plain_rounds())).await;
        assert_eq!(
            decision.strategy,
            CompressionStrategy::DialogueSummary,
            "fallback branch: {name}"
        );
        assert_eq!(decision.source, StrategySource::DeterministicFallback);
        assert!(
            decision.reason.contains(expected),
            "fallback branch {name} reason was {:?}",
            decision.reason
        );
        assert!(decision.plan.is_none());
    }
}

#[tokio::test]
async fn the_allowlist_filters_the_models_choice() {
    let model = ScriptedModel::text("{\"strategy\": \"full_compact\"}");
    let strategy = AdaptiveStrategy::new(AdaptiveStrategyConfig {
        enabled: true,
        model: Some(Arc::new(model)),
        strategy_allowlist: vec![
            CompressionStrategy::DialogueSummary,
            CompressionStrategy::CurrentRound,
        ],
        ..AdaptiveStrategyConfig::default()
    });
    let decision = strategy.select(&measured(&plain_rounds())).await;
    assert_eq!(decision.strategy, CompressionStrategy::DialogueSummary);
    assert_eq!(decision.source, StrategySource::DeterministicFallback);
    assert!(decision.reason.contains("outside the configured allowlist"));

    let noop = ScriptedModel::text("{\"strategy\": \"noop\"}");
    let strategy = AdaptiveStrategy::new(AdaptiveStrategyConfig {
        enabled: true,
        model: Some(Arc::new(noop)),
        strategy_allowlist: vec![CompressionStrategy::FullCompact],
        ..AdaptiveStrategyConfig::default()
    });
    let decision = strategy.select(&measured(&plain_rounds())).await;
    assert_eq!(decision.strategy, CompressionStrategy::Noop);
    assert_eq!(decision.source, StrategySource::Model);
    assert!(strategy.config().allows(CompressionStrategy::Noop));
    assert!(!strategy
        .config()
        .allows(CompressionStrategy::DialogueSummary));
}

#[tokio::test]
async fn the_decision_call_is_bounded_and_never_streams() {
    let model = Arc::new(ScriptedModel::text("{\"strategy\": \"current_round\"}"));
    let strategy = AdaptiveStrategy::new(AdaptiveStrategyConfig {
        enabled: true,
        max_decision_tokens: 64,
        model: Some(Arc::clone(&model) as Arc<dyn Model>),
        ..AdaptiveStrategyConfig::default()
    });
    let decision = strategy.select(&measured(&plain_rounds())).await;
    assert_eq!(decision.strategy, CompressionStrategy::CurrentRound);
    let call = model.last_call().expect("one decision call");
    assert_eq!(call.mode, CallMode::Invoke);
    assert_eq!(call.request.max_tokens, Some(64));
    let prompt = call.last_user_text().expect("a user prompt");
    assert!(prompt.contains("Memory block JSON schema"));
    assert!(prompt.contains("Allowed strategies: every strategy."));
    assert!(call
        .messages
        .first()
        .is_some_and(|message| message.text().contains("dialogue_summary")));
}

#[tokio::test]
async fn a_disabled_flag_or_a_small_payload_never_calls_the_model() {
    let disabled = Arc::new(ScriptedModel::text("{\"strategy\": \"full_compact\"}"));
    let strategy = AdaptiveStrategy::new(AdaptiveStrategyConfig {
        enabled: false,
        model: Some(Arc::clone(&disabled) as Arc<dyn Model>),
        ..AdaptiveStrategyConfig::default()
    });
    let decision = strategy.select(&measured(&plain_rounds())).await;
    assert_eq!(decision.source, StrategySource::DeterministicFallback);
    assert_eq!(disabled.call_count(), 0);

    let small = Arc::new(ScriptedModel::text("{\"strategy\": \"full_compact\"}"));
    let strategy = AdaptiveStrategy::new(AdaptiveStrategyConfig {
        enabled: true,
        model: Some(Arc::clone(&small) as Arc<dyn Model>),
        ..AdaptiveStrategyConfig::default()
    });
    let decision = strategy.select(&AdaptiveContext::default()).await;
    assert_eq!(decision.strategy, CompressionStrategy::Noop);
    assert_eq!(decision.source, StrategySource::DeterministicFallback);
    assert_eq!(small.call_count(), 0);
}

// ---------------------------------------------------------------------------
// The memory-block plan
// ---------------------------------------------------------------------------

#[test]
fn the_plan_round_trips_and_describes_its_schema() {
    let plan = MemoryBlockPlan {
        summary: "carried state".to_owned(),
        tasks: vec!["ship the flag".to_owned()],
        decisions: vec!["keep the envelope".to_owned()],
        constraints: vec!["no new crates".to_owned()],
        files: vec!["src/lib.rs".to_owned()],
        open_questions: vec!["is the ratio right?".to_owned()],
    };
    let json = serde_json::to_string(&plan).unwrap();
    assert_eq!(
        serde_json::from_str::<MemoryBlockPlan>(&json).unwrap(),
        plan
    );
    assert_eq!(
        serde_json::from_str::<MemoryBlockPlan>("{}").unwrap(),
        MemoryBlockPlan::default()
    );
    assert!(!plan.is_empty());
    assert!(MemoryBlockPlan::default().is_empty());

    let schema = MemoryBlockPlan::json_schema();
    assert_eq!(schema["type"], "object");
    for field in [
        "summary",
        "tasks",
        "decisions",
        "constraints",
        "files",
        "open_questions",
    ] {
        assert!(
            schema["properties"].get(field).is_some(),
            "schema field {field}"
        );
    }
    assert_eq!(schema["properties"]["tasks"]["type"], "array");
    assert_eq!(schema["properties"]["tasks"]["items"]["type"], "string");
    assert_eq!(schema["additionalProperties"], true);
    assert_eq!(schema["required"].as_array().map(Vec::len), Some(0));
}

#[test]
fn the_plan_parses_lenient_replies() {
    let reply = "Here you go:\n```json\n{\
        \"summary\": \"state\",\
        \"tasks\": [\"one\", {\"text\": \"two\"}, \"one\"],\
        \"files\": [{\"path\": \"crates/a.rs\"}],\
        \"decisions\": \"a single string decision\",\
        \"unknown\": {\"nested\": true}\
    }\n```\n";
    let plan = MemoryBlockPlan::from_model_reply(reply).unwrap();
    assert_eq!(plan.summary, "state");
    assert_eq!(plan.tasks, vec!["one".to_owned(), "two".to_owned()]);
    assert_eq!(plan.files, vec!["crates/a.rs".to_owned()]);
    assert_eq!(plan.decisions, vec!["a single string decision".to_owned()]);

    let prose = MemoryBlockPlan::from_model_reply("  just prose  ").unwrap();
    assert_eq!(prose.summary, "just prose");
    assert_eq!(prose.tasks, Vec::<String>::new());
}

#[test]
fn the_plan_rejects_unusable_replies() {
    for reply in [
        "",
        "   ",
        "{}",
        "{\"strategy\": \"noop\"}",
        "{\"tasks\": []}",
    ] {
        let error = MemoryBlockPlan::from_model_reply(reply).unwrap_err();
        assert!(
            matches!(error, openjiuwen_common::error::Error::Validation(_)),
            "reply {reply:?} produced {error:?}"
        );
    }
}

#[test]
fn offline_extraction_finds_tasks_decisions_constraints_and_files() {
    let messages = vec![
        BaseMessage::user("TODO: wire the adaptive flag through the compressor"),
        BaseMessage::assistant("We decided to keep the memory-block envelope unchanged."),
        BaseMessage::user("You must keep the recall marker last."),
        BaseMessage::assistant("Open question: does the pipeline retry the chosen pass?"),
        BaseMessage::tool(
            "c1",
            "edited crates/openjiuwen-context/src/processor/compressor/adaptive/mod.rs",
        ),
    ];
    let plan = MemoryBlockPlan::from_messages(&messages, true);
    assert!(plan
        .tasks
        .iter()
        .any(|task| task.contains("wire the adaptive flag")));
    assert!(plan
        .decisions
        .iter()
        .any(|decision| decision.contains("keep the memory-block envelope")));
    assert!(plan
        .constraints
        .iter()
        .any(|constraint| constraint.contains("keep the recall marker last")));
    assert!(plan
        .open_questions
        .iter()
        .any(|question| question.contains("does the pipeline retry")));
    assert!(plan
        .files
        .iter()
        .any(|file| file.ends_with("adaptive/mod.rs")));
    assert_eq!(plan.summary, "");
    assert!(MemoryBlockPlan::from_messages(&messages, false).is_empty());

    let unstructured = vec![BaseMessage::user("hello"), BaseMessage::assistant("hi")];
    let plan = MemoryBlockPlan::from_messages(&unstructured, true);
    assert!(plan.tasks.is_empty() && plan.files.is_empty());
    assert!(plan.summary.starts_with("[1] user: hello"));
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

#[test]
fn a_plan_renders_inside_the_existing_envelope() {
    let plan = MemoryBlockPlan {
        summary: "carried state".to_owned(),
        tasks: vec!["ship it".to_owned()],
        decisions: vec!["keep the envelope".to_owned()],
        constraints: vec!["no new crates".to_owned()],
        files: vec!["src/lib.rs".to_owned()],
        open_questions: vec!["is it fast enough?".to_owned()],
    };
    let text = render_memory_block_from_plan(MemoryBlockKind::Dialogue, &plan);
    assert!(text.starts_with(DIALOGUE_MEMORY_BLOCK));
    assert!(text.contains("processor: DialogueCompressor"));
    assert!(text.contains("authority: This block is reference memory"));
    assert!(text.contains("\nSummary:\ncarried state\n\ntasks:\n- ship it"));
    let body = MemoryBlockKind::Dialogue.summary_body(&text).unwrap();
    let order: Vec<usize> = [
        "tasks:",
        "decisions:",
        "constraints:",
        "files:",
        "open_questions:",
    ]
    .iter()
    .map(|label| body.find(label).unwrap())
    .collect();
    assert!(order.windows(2).all(|pair| pair[0] < pair[1]));
    assert!(!text.contains("[[COMPRESSION_RECALL"));
    assert!(is_memory_block(
        &BaseMessage::user(text),
        DIALOGUE_MEMORY_BLOCK
    ));

    let bare = render_memory_block_from_plan(
        MemoryBlockKind::CurrentRound,
        &MemoryBlockPlan {
            summary: "only a summary".to_owned(),
            ..MemoryBlockPlan::default()
        },
    );
    assert_eq!(bare, MemoryBlockKind::CurrentRound.render("only a summary"));
    assert!(bare.starts_with(CURRENT_ROUND_MEMORY_BLOCK));
}

#[test]
fn a_recall_marker_is_the_last_line_of_the_block() {
    let plan = MemoryBlockPlan {
        summary: "carried state".to_owned(),
        tasks: vec!["ship it".to_owned()],
        ..MemoryBlockPlan::default()
    };
    let marker = RecallArchiveMarker {
        memory_id: "abc123".to_owned(),
    };
    let text = render_memory_block_from_plan_with_marker(
        MemoryBlockKind::RoundLevel,
        &plan,
        Some(&marker),
    );
    assert_eq!(
        text.lines().last(),
        Some("[[COMPRESSION_RECALL: id=abc123]]")
    );
    assert_eq!(text.matches("[[COMPRESSION_RECALL").count(), 1);
    assert!(text.contains("tasks:\n- ship it"));
}

// ---------------------------------------------------------------------------
// Compressor integration
// ---------------------------------------------------------------------------

/// Shared processor inputs for the compressor integration tests.
struct Fixture {
    config: ContextConfig,
    counter: ApproxTokenCounter,
    store: InMemoryOffloadStore,
    scope: OffloadScope,
    cancellation: CancellationToken,
}

impl Fixture {
    fn new() -> Self {
        Self {
            config: ContextConfig::default(),
            counter: counter(),
            store: InMemoryOffloadStore::new(),
            scope: OffloadScope::new("session-adaptive", "main"),
            cancellation: CancellationToken::new(),
        }
    }

    fn context(&self) -> ProcessorContext<'_> {
        ProcessorContext {
            config: &self.config,
            counter: &self.counter,
            model: None,
            offload: &self.store as &dyn OffloadStore,
            scope: &self.scope,
            budget: 100_000,
            trigger_tokens: 100,
            manual: true,
            cancellation: &self.cancellation,
            recall: None,
            session_memory: None,
        }
    }
}

/// A dialogue configuration that compresses aggressively.
fn dialogue_config(adaptive: AdaptiveConfig) -> DialogueCompressorConfig {
    DialogueCompressorConfig {
        tokens_threshold: 100,
        compression_target_tokens: 200,
        adaptive,
        ..DialogueCompressorConfig::default()
    }
}

/// A current-round configuration that compresses aggressively.
fn current_round_config(adaptive: AdaptiveConfig) -> CurrentRoundCompressorConfig {
    CurrentRoundCompressorConfig {
        tokens_threshold: 100,
        messages_to_keep: 1,
        min_selected_tokens_for_compression: 1,
        compression_target_tokens: 200,
        adaptive,
        ..CurrentRoundCompressorConfig::default()
    }
}

/// The block the dialogue compressor wrote, if any.
fn dialogue_block(messages: &[BaseMessage]) -> Option<String> {
    messages
        .iter()
        .find(|message| is_memory_block(message, DIALOGUE_MEMORY_BLOCK))
        .map(|message| message.text().into_owned())
}

/// The block the current-round compressor wrote, if any.
fn current_round_block(messages: &[BaseMessage]) -> Option<String> {
    messages
        .iter()
        .find(|message| is_memory_block(message, CURRENT_ROUND_MEMORY_BLOCK))
        .map(|message| message.text().into_owned())
}

#[tokio::test]
async fn the_flag_off_behaviour_is_unchanged() {
    let fixture = Fixture::new();
    let cx = fixture.context();

    // The default configuration (flag off) and an explicit `enabled: false`
    // produce the same history, the same event, and exactly one summary call
    // per compressed span — never a decision call.
    let baseline_model = Arc::new(ScriptedModel::new([
        ScriptedTurn::text("first summary"),
        ScriptedTurn::text("second summary"),
    ]));
    let baseline = DialogueCompressor::new(
        dialogue_config(AdaptiveConfig::default()),
        Some(Arc::clone(&baseline_model) as Arc<dyn Model>),
    );
    let mut left = tool_history();
    let left_event = baseline.process(&mut left, &cx).await.unwrap();

    let off_model = Arc::new(ScriptedModel::new([
        ScriptedTurn::text("first summary"),
        ScriptedTurn::text("second summary"),
    ]));
    let off = DialogueCompressor::new(
        dialogue_config(AdaptiveConfig {
            enabled: false,
            ..AdaptiveConfig::default()
        }),
        Some(Arc::clone(&off_model) as Arc<dyn Model>),
    );
    let mut right = tool_history();
    let right_event = off.process(&mut right, &cx).await.unwrap();

    assert_eq!(left, right);
    assert_eq!(left_event, right_event);
    assert!(dialogue_block(&right).is_some());
    assert_eq!(baseline_model.call_count(), 2);
    assert_eq!(off_model.call_count(), 2);
    assert_eq!(off.state.counter("compressions"), 2);
}

#[tokio::test]
async fn an_adaptive_noop_reports_nothing_and_keeps_the_counters() {
    let fixture = Fixture::new();
    let cx = fixture.context();
    let model = Arc::new(ScriptedModel::text(
        "{\"strategy\": \"noop\", \"reason\": \"the newest round is still growing\"}",
    ));
    let compressor = DialogueCompressor::new(
        dialogue_config(adaptive_on()),
        Some(Arc::clone(&model) as Arc<dyn Model>),
    );

    let mut messages = tool_history();
    let original = messages.clone();
    assert!(compressor
        .process(&mut messages, &cx)
        .await
        .unwrap()
        .is_none());
    assert_eq!(messages, original);
    assert_eq!(model.call_count(), 1);
    assert!(compressor.state.snapshot().is_null());
}

#[tokio::test]
async fn another_strategy_is_reported_without_touching_the_history() {
    let fixture = Fixture::new();
    let cx = fixture.context();
    let model = Arc::new(ScriptedModel::text(
        "{\"strategy\": \"micro_compact\", \"reason\": \"stale tool results dominate\"}",
    ));
    let compressor = DialogueCompressor::new(
        dialogue_config(adaptive_on()),
        Some(Arc::clone(&model) as Arc<dyn Model>),
    );

    let mut messages = tool_history();
    let original = messages.clone();
    let event = compressor
        .process(&mut messages, &cx)
        .await
        .unwrap()
        .expect("a recommendation event");
    assert_eq!(messages, original);
    assert_eq!(event.processor, "dialogue");
    assert_eq!(event.kind, ContextEventKind::Summarized);
    assert_eq!(event.messages_changed, Vec::<usize>::new());
    assert_eq!(event.tokens_before, event.tokens_after);
    let summary = event.summary.expect("the recommendation is in the summary");
    assert!(summary.contains("micro_compact"));
    assert!(summary.contains("stale tool results dominate"));
    assert!(compressor.state.snapshot().is_null());
}

#[tokio::test]
async fn the_same_strategy_writes_the_structured_sections() {
    let fixture = Fixture::new();
    let cx = fixture.context();
    let model = Arc::new(ScriptedModel::new([
        ScriptedTurn::text(
            r#"{"strategy": "dialogue_summary", "reason": "older rounds dominate",
                "tasks": ["ship the adaptive path"],
                "decisions": ["keep the envelope"],
                "files": ["crates/openjiuwen-context/src/lib.rs"]}"#,
        ),
        ScriptedTurn::text("summary of the older rounds"),
        ScriptedTurn::text("summary of the oldest round"),
    ]));
    let compressor = DialogueCompressor::new(
        dialogue_config(adaptive_on()),
        Some(Arc::clone(&model) as Arc<dyn Model>),
    );

    let mut messages = tool_history();
    let event = compressor
        .process(&mut messages, &cx)
        .await
        .unwrap()
        .expect("the dialogue pass ran");
    let block = dialogue_block(&messages).expect("a dialogue memory block");
    assert!(block.contains("tasks:\n- ship the adaptive path"));
    assert!(block.contains("decisions:\n- keep the envelope"));
    assert!(block.contains("files:\n- crates/openjiuwen-context/src/lib.rs"));
    assert!(block.contains("summary of the"));
    assert!(event.summary.is_some());
    assert_eq!(compressor.state.counter("compressions"), 2);
    // The decision call came first, then one summary call per compressed span.
    assert_eq!(model.call_count(), 3);
}

#[tokio::test]
async fn the_current_round_honours_the_same_contract() {
    let fixture = Fixture::new();
    let cx = fixture.context();

    let noop = Arc::new(ScriptedModel::text("{\"strategy\": \"noop\"}"));
    let compressor = CurrentRoundCompressor::new(
        current_round_config(adaptive_on()),
        Some(Arc::clone(&noop) as Arc<dyn Model>),
    );
    let mut messages = in_flight_history();
    messages.push(BaseMessage::assistant("progress"));
    messages.push(BaseMessage::assistant("latest"));
    let original = messages.clone();
    assert!(compressor
        .process(&mut messages, &cx)
        .await
        .unwrap()
        .is_none());
    assert_eq!(messages, original);
    assert_eq!(noop.call_count(), 1);
    assert!(compressor.state.snapshot().is_null());

    let other = Arc::new(ScriptedModel::text(
        "{\"strategy\": \"round_level\", \"reason\": \"several rounds\"}",
    ));
    let compressor = CurrentRoundCompressor::new(
        current_round_config(adaptive_on()),
        Some(Arc::clone(&other) as Arc<dyn Model>),
    );
    let mut messages = original.clone();
    let event = compressor
        .process(&mut messages, &cx)
        .await
        .unwrap()
        .expect("a recommendation event");
    assert_eq!(messages, original);
    assert_eq!(event.messages_changed, Vec::<usize>::new());
    assert!(event.summary.unwrap_or_default().contains("round_level"));

    let same = Arc::new(ScriptedModel::new([
        ScriptedTurn::text(
            r#"{"strategy": "current_round", "reason": "the round dominates",
                "tasks": ["finish the in-flight step"]}"#,
        ),
        ScriptedTurn::text("summary of the earlier steps"),
    ]));
    let compressor = CurrentRoundCompressor::new(
        current_round_config(adaptive_on()),
        Some(Arc::clone(&same) as Arc<dyn Model>),
    );
    let mut messages = original.clone();
    let event = compressor
        .process(&mut messages, &cx)
        .await
        .unwrap()
        .expect("the current-round pass ran");
    let block = current_round_block(&messages).expect("a current-round memory block");
    assert!(block.contains("tasks:\n- finish the in-flight step"));
    assert!(block.contains("summary of the earlier steps"));
    assert!(event
        .messages_changed
        .iter()
        .any(|index| *index < original.len()));
    assert_eq!(compressor.state.counter("compressions"), 1);
}

#[tokio::test]
async fn adaptive_blocks_can_omit_the_sections_or_the_extraction() {
    let fixture = Fixture::new();
    let cx = fixture.context();

    // Structured blocks off: the body is the plain summary, exactly as the
    // non-adaptive path renders it.
    let model = Arc::new(ScriptedModel::new([
        ScriptedTurn::text(
            r#"{"strategy": "dialogue_summary",
                "tasks": ["model task"]}"#,
        ),
        ScriptedTurn::text("first summary"),
        ScriptedTurn::text("second summary"),
    ]));
    let compressor = DialogueCompressor::new(
        dialogue_config(AdaptiveConfig {
            enabled: true,
            json_schema_blocks: false,
            ..AdaptiveConfig::default()
        }),
        Some(Arc::clone(&model) as Arc<dyn Model>),
    );
    let mut messages = tool_history();
    assert!(compressor
        .process(&mut messages, &cx)
        .await
        .unwrap()
        .is_some());
    let block = dialogue_block(&messages).expect("a dialogue memory block");
    assert!(!block.contains("tasks:"));
    assert!(block.contains("first summary") || block.contains("second summary"));

    // Extraction on (the default): a TODO line inside a compressed span reaches
    // the block. Conversion off: it stays out.
    let run = |task_extraction: bool| {
        let model = Arc::new(ScriptedModel::new([
            ScriptedTurn::text("{\"strategy\": \"dialogue_summary\"}"),
            ScriptedTurn::text("first summary"),
            ScriptedTurn::text("second summary"),
        ]));
        let compressor = DialogueCompressor::new(
            dialogue_config(AdaptiveConfig {
                enabled: true,
                task_extraction,
                ..AdaptiveConfig::default()
            }),
            Some(Arc::clone(&model) as Arc<dyn Model>),
        );
        (compressor, model)
    };

    let (compressor, _model) = run(true);
    let mut messages = todo_history();
    assert!(compressor
        .process(&mut messages, &cx)
        .await
        .unwrap()
        .is_some());
    let block = dialogue_block(&messages).expect("a dialogue memory block");
    assert!(block.contains("a task the extractor would find"));

    let (compressor, _model) = run(false);
    let mut messages = todo_history();
    assert!(compressor
        .process(&mut messages, &cx)
        .await
        .unwrap()
        .is_some());
    let block = dialogue_block(&messages).expect("a dialogue memory block");
    assert!(!block.contains("a task the extractor would find"));
}

#[test]
fn the_adaptive_sub_config_round_trips_and_validates() {
    let config = AdaptiveConfig::default();
    assert!(!config.enabled);
    assert!(config.task_extraction);
    assert!(config.json_schema_blocks);
    assert!(config.validate().is_ok());
    let json = serde_json::to_string(&config).unwrap();
    assert_eq!(
        serde_json::from_str::<AdaptiveConfig>(&json).unwrap(),
        config
    );

    let invalid = AdaptiveConfig {
        enabled: true,
        max_decision_tokens: 0,
        ..AdaptiveConfig::default()
    };
    assert!(invalid.validate().is_err());
    assert!(AdaptiveConfig {
        enabled: false,
        max_decision_tokens: 0,
        ..AdaptiveConfig::default()
    }
    .validate()
    .is_ok());
}
