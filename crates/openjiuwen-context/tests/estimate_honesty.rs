#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Honesty of every token-count fallback: the typed "estimated" fact and its reason.
//!
//! These tests pin the rule that an estimated count never looks exact:
//!
//! - every non-native [`TokenizerSource`] reports `estimated = true` and names
//!   why, both through `fallback_reason` (a degraded selection) and through
//!   `estimation_note` (the prose explanation of the estimate);
//! - a native artifact keeps counting exactly;
//! - the usage analyzer publishes the provenance of the counter that actually
//!   runs, so a native identity over an approximate counter is an estimate;
//! - `ContextTokenCount` carries the same provenance into a compression-trigger
//!   decision;
//! - `RecallChunk` marks its character-based size estimate.
//!
//! Everything is offline and deterministic: `ApproxTokenCounter`, an in-memory
//! native artifact and no model call.
use std::sync::Arc;

use openjiuwen_context::recall::chunk::{estimate_tokens, split_text};
use openjiuwen_context::recall::{archive_from_messages, RecallChunk};
use openjiuwen_context::tokenizers::{
    ModelTokenCounter, NativeTokenizer, SelectedTokenizer, TokenizerEngine, TokenizerSelector,
    TokenizerSource, TokenizerSpec,
};
use openjiuwen_context::usage::analyzer::ContextUsageAnalyzer;
use openjiuwen_context::usage::context_tokens::{
    count_context_window_tokens, count_context_window_tokens_with_provenance,
    count_messages_tokens_with_provenance, count_tools_tokens_with_provenance, ContextTokenCount,
    STRING_LENGTH_NOTE, USAGE_METADATA_KEY, USAGE_TAIL_NOTE,
};
use openjiuwen_context::usage::models::CounterProvenance;
use openjiuwen_context::usage::{AnalysisInput, ContextCategory, CounterIdentity};
use openjiuwen_context::{
    CompressionRecallConfig, ContextConfig, ContextEngine, PrepareOptions, TokenCounterConfig,
    TokenCounterKind, TokenizerSource as RootSource,
};
use openjiuwen_foundation::prompt::{ApproxTokenCounter, BaseMessage, TokenCounter};
use serde_json::json;

/// The reason published for a native identity the analyzer cannot verify.
const UNVERIFIED_NATIVE_REASON: &str = "unverified_native_tokenizer_identity";

/// The reason published when a declared identity disagrees with the counter.
const DISAGREEING_IDENTITY_REASON: &str = "declared_identity_disagrees_with_counter";

/// A tiny BPE artifact: `a`, `b`, `c` merge into `abc`, nothing else is known.
const BPE_FIXTURE: &str = r#"{
    "name": "tiny-bpe",
    "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "c": 2, "ab": 3, "abc": 4}, "merges": ["a b", "ab c"]},
    "pre_tokenizer": {"type": "Whitespace"}
}"#;

/// A window input with one system message and one user message.
fn sample_input<'a>(system: &'a [BaseMessage], messages: &'a [BaseMessage]) -> AnalysisInput<'a> {
    let mut input = AnalysisInput::new("req-1", "session-1");
    input.system_messages = Some(system);
    input.context_messages = messages;
    input
}

/// The part of `category` in `snapshot`.
fn part_estimated(
    snapshot: &openjiuwen_context::ContextUsageSnapshot,
    category: ContextCategory,
) -> bool {
    snapshot
        .parts
        .get(category.as_str())
        .expect("category part")
        .estimated
}

/// A loaded native artifact, counting exactly.
fn native_selection() -> SelectedTokenizer {
    let native = NativeTokenizer::from_json_str(BPE_FIXTURE).expect("fixture loads");
    SelectedTokenizer::from_native(
        Arc::new(native),
        Some("gpt-4o".to_owned()),
        Some("tiny-bpe".to_owned()),
    )
}

// ---------------------------------------------------------------------------
// The named reason of every rung
// ---------------------------------------------------------------------------

#[test]
fn every_estimated_source_names_why_it_is_an_estimate() {
    assert_eq!(TokenizerSource::NativeTokenizer.estimation_note(), None);
    assert!(TokenizerSource::NativeTokenizer.is_native());
    for source in [
        TokenizerSource::FamilyTokenizerFallback,
        TokenizerSource::Tiktoken,
        TokenizerSource::TiktokenFallback,
        TokenizerSource::StringLengthFallback,
    ] {
        assert!(!source.is_native(), "{source:?} is not native");
        let note = source
            .estimation_note()
            .expect("an estimate names its reason");
        assert!(!note.trim().is_empty(), "{source:?} has a non-empty note");
    }
}

#[test]
fn a_resolved_tiktoken_vocabulary_is_an_estimate_without_a_degradation() {
    let counter = ModelTokenCounter::new("gpt-4o");
    assert_eq!(counter.source(), TokenizerSource::Tiktoken);
    // The vocabulary is exact for itself but is not the model's own tokenizer.
    assert!(counter.estimated());
    assert_eq!(counter.fallback_reason(), None);
    assert!(counter
        .estimation_note()
        .expect("a note")
        .contains("tiktoken vocabulary"));
    let measured = counter.measure("hello world");
    assert!(measured.estimated);
    assert_eq!(measured.source, TokenizerSource::Tiktoken);
    assert_eq!(measured.fallback_reason, None);
    assert_eq!(measured.estimation_note(), counter.estimation_note());
}

#[test]
fn an_unregistered_model_names_the_fallback_reason() {
    let counter = ModelTokenCounter::new("claude-3-5-sonnet");
    assert_eq!(counter.source(), TokenizerSource::TiktokenFallback);
    assert!(counter.estimated());
    assert_eq!(counter.fallback_reason(), Some("model_not_registered"));
    let measured = counter.measure_messages(&[BaseMessage::user("hello")]);
    assert!(measured.estimated);
    assert_eq!(
        measured.fallback_reason.as_deref(),
        Some("model_not_registered")
    );
    assert!(measured.estimation_note().is_some());
}

#[test]
fn an_unavailable_native_engine_degrades_with_a_named_reason() {
    let spec = TokenizerSpec {
        engine: TokenizerEngine::Native,
        ..TokenizerSpec::default()
    };
    let mut spec = spec;
    spec.model = "gpt-4o".to_owned();
    let selected = TokenizerSelector::new("openai", "gpt-4o", Some(spec), None).select();
    assert_eq!(selected.source(), TokenizerSource::TiktokenFallback);
    assert!(selected.estimated());
    assert_eq!(
        selected.fallback_reason(),
        Some("native_tokenizer_unavailable")
    );
    assert!(selected.estimation_note().is_some());
    let measured = selected.measure("hello world");
    assert!(measured.estimated);
    assert_eq!(
        measured.fallback_reason.as_deref(),
        Some("native_tokenizer_unavailable")
    );
}

#[test]
fn the_string_length_rung_is_estimated_and_names_its_reason() {
    let selected = TokenizerSelector::new("ollama", "qwen3-72b", None, None).select();
    assert_eq!(selected.source(), TokenizerSource::StringLengthFallback);
    assert!(selected.estimated());
    assert_eq!(selected.fallback_reason(), Some("model_not_registered"));
    assert!(selected
        .estimation_note()
        .expect("a note")
        .contains("string length"));
    // Three characters per token.
    // Three characters per token, rounded up by the shared approximate counter.
    assert_eq!(selected.count_text("abcdef"), 2);
    assert_eq!(selected.count_text("abcde"), 2);
    let measured = selected.measure("abcdef");
    assert!(measured.estimated);
    assert_eq!(measured.source, TokenizerSource::StringLengthFallback);
    assert_eq!(
        measured.fallback_reason.as_deref(),
        Some("model_not_registered")
    );
}

#[test]
fn a_selector_without_a_model_reports_the_disabled_reason() {
    let selected = TokenizerSelector::new("", "", None, None).select();
    assert_eq!(selected.source(), TokenizerSource::StringLengthFallback);
    assert_eq!(selected.fallback_reason(), Some("tiktoken_disabled"));
    assert!(selected.estimated());
}

#[test]
fn a_declared_fallback_source_names_a_reason_even_without_a_degradation() {
    // `gpt-4o` resolves cleanly, so encoding resolution produces no reason; the
    // declared source still has to name why its counts are estimates.
    let spec = TokenizerSpec {
        model: "gpt-4o".to_owned(),
        source: Some("tiktoken_fallback".to_owned()),
        tokenizer_id: Some("declared".to_owned()),
        ..TokenizerSpec::default()
    };
    let selected = TokenizerSelector::new("openai", "gpt-4o", Some(spec), None).select();
    assert_eq!(selected.source(), TokenizerSource::TiktokenFallback);
    assert!(selected.estimated());
    assert_eq!(selected.fallback_reason(), Some("tiktoken_unavailable"));
    assert!(selected.estimation_note().is_some());
}

#[test]
fn a_native_artifact_keeps_counting_exactly() {
    let selected = native_selection();
    assert_eq!(selected.source(), RootSource::NativeTokenizer);
    assert!(!selected.estimated());
    assert_eq!(selected.fallback_reason(), None);
    assert_eq!(selected.estimation_note(), None);
    let measured = selected.measure("abc");
    assert_eq!(measured.tokens, 1);
    assert_eq!(measured.source, TokenizerSource::NativeTokenizer);
    assert!(!measured.estimated);
    assert_eq!(measured.fallback_reason, None);
    assert_eq!(measured.estimation_note(), None);
}

// ---------------------------------------------------------------------------
// The analyzer publishes the provenance of the counter that runs
// ---------------------------------------------------------------------------

#[test]
fn an_unverified_native_identity_can_no_longer_publish_an_exact_count() {
    let analyzer = ContextUsageAnalyzer::new(
        Arc::new(ApproxTokenCounter::with_chars_per_token(1)),
        CounterIdentity::new("TestNativeCounter", TokenizerSource::NativeTokenizer),
    )
    .with_model("test-model");
    assert!(!analyzer.identity_verified());
    let system = vec![BaseMessage::system("abc")];
    let messages = vec![BaseMessage::user("hello")];
    let snapshot = analyzer.analyze(&sample_input(&system, &messages));
    // The declared source is republished, but the count is an estimate and says why.
    assert_eq!(snapshot.measurement.category_source, "native_tokenizer");
    assert!(snapshot.measurement.estimated);
    let system_part = snapshot
        .parts
        .get(ContextCategory::SystemPrompt.as_str())
        .expect("system part");
    assert!(system_part.estimated);
    assert_eq!(
        system_part.fallback_reason.as_deref(),
        Some(UNVERIFIED_NATIVE_REASON)
    );
    let published = analyzer.published_identity();
    assert_eq!(published.source_name(), "native_tokenizer");
    assert_eq!(
        published.fallback_reason.as_deref(),
        Some(UNVERIFIED_NATIVE_REASON)
    );
}

#[test]
fn an_analyzer_built_from_its_counter_publishes_an_exact_native_count() {
    let analyzer = ContextUsageAnalyzer::from_counter(Arc::new(native_selection()));
    assert!(analyzer.identity_verified());
    assert_eq!(
        analyzer.counter_identity().source_name(),
        "native_tokenizer"
    );
    let system = vec![BaseMessage::system("abc")];
    let messages = vec![BaseMessage::user("abc")];
    let snapshot = analyzer.analyze(&sample_input(&system, &messages));
    assert!(!snapshot.measurement.estimated);
    assert_eq!(snapshot.measurement.category_source, "native_tokenizer");
    assert!(!part_estimated(&snapshot, ContextCategory::SystemPrompt));
    assert!(!part_estimated(&snapshot, ContextCategory::Messages));
    assert_eq!(
        snapshot
            .parts
            .get(ContextCategory::Messages.as_str())
            .expect("messages part")
            .fallback_reason,
        None
    );
}

#[test]
fn an_analyzer_built_from_an_approximate_counter_publishes_an_estimate() {
    let analyzer = ContextUsageAnalyzer::from_counter(Arc::new(ApproxTokenCounter::new()));
    assert!(analyzer.identity_verified());
    assert_eq!(
        analyzer.published_identity().source_name(),
        "string_length_fallback"
    );
    assert!(analyzer.counter_identity().estimated());
    let messages = vec![BaseMessage::user("hello")];
    let snapshot = analyzer.analyze(&sample_input(&[], &messages));
    assert!(snapshot.measurement.estimated);
    assert!(part_estimated(&snapshot, ContextCategory::Messages));
}

#[test]
fn a_disagreeing_declared_identity_is_recorded_not_published() {
    let analyzer = ContextUsageAnalyzer::from_counter_with(
        Arc::new(ApproxTokenCounter::new()),
        CounterIdentity::new("ClaimedNative", TokenizerSource::NativeTokenizer),
    );
    // The counter wins, so the published source is the approximate one.
    let published = analyzer.published_identity();
    assert_eq!(published.source_name(), "string_length_fallback");
    assert_eq!(published.class, "ApproxTokenCounter");
    assert_eq!(
        published.fallback_reason.as_deref(),
        Some(DISAGREEING_IDENTITY_REASON)
    );
    assert!(published.estimated());
}

#[tokio::test]
async fn an_engine_analyzer_cannot_publish_native_provenance_for_an_approximate_context() {
    // Bug 2: the engine-level analyzer declares a native identity while the
    // context counts with `TokenCounterKind::Approx`. The window limit is
    // substituted; the counter is not, so the published count keeps the
    // provenance of the counter that ran and never claims to be exact.
    let engine = ContextEngine::new(ContextConfig {
        token_counter: TokenCounterConfig {
            kind: TokenCounterKind::Approx,
            chars_per_token: 4,
        },
        ..ContextConfig::default()
    })
    .with_usage_analyzer(Some(Arc::new(ContextUsageAnalyzer::new(
        Arc::new(ApproxTokenCounter::new()),
        CounterIdentity::new("ClaimedNative", TokenizerSource::NativeTokenizer),
    ))));
    let context = engine.create_context("main", None).await.unwrap();
    engine
        .observe(&context, BaseMessage::user("hello, explain the engine"))
        .await
        .unwrap();
    let options = PrepareOptions {
        system_messages: vec![BaseMessage::system("you are a careful assistant")],
        ..PrepareOptions::default()
    };
    let prepared = engine.prepare(&context, &options).await.unwrap();
    let usage = prepared.usage.as_ref().expect("the usage block is filled");
    assert!(usage.measurement.estimated);
    let system_part = &usage.parts[ContextCategory::SystemPrompt.as_str()];
    assert!(system_part.estimated);
    assert_eq!(
        system_part.fallback_reason.as_deref(),
        Some(UNVERIFIED_NATIVE_REASON)
    );
}

#[test]
fn counter_provenance_is_derived_from_every_counter_kind() {
    let approx = CounterProvenance::counter_identity(&ApproxTokenCounter::new());
    assert!(approx.is_fallback());
    let model = CounterProvenance::counter_identity(&ModelTokenCounter::new("gpt-4o"));
    assert_eq!(model.source_name(), "tiktoken");
    assert!(model.estimated());
    assert!(!model.is_fallback());
    let native = CounterProvenance::counter_identity(&native_selection());
    assert_eq!(native.source_name(), "native_tokenizer");
    assert!(!native.estimated());
}

// ---------------------------------------------------------------------------
// ContextTokenCount: provenance through a compression-gating call
// ---------------------------------------------------------------------------

#[test]
fn a_usage_baseline_with_a_tail_gates_compression_as_an_estimate() {
    let baseline = BaseMessage::assistant("answer")
        .with_metadata(USAGE_METADATA_KEY, json!({"total_tokens": 100}));
    let messages = vec![
        BaseMessage::user("question"),
        baseline,
        BaseMessage::user("abcdef"),
    ];
    // The compressor compares this total with its budget.
    let counted = count_context_window_tokens_with_provenance(&messages, &[], None);
    assert_eq!(counted.tokens, 100 + 2);
    assert!(counted.estimated);
    assert_eq!(counted.note.as_deref(), Some(USAGE_TAIL_NOTE));
    // The compatibility wrapper returns the same number.
    assert_eq!(
        counted.tokens,
        count_context_window_tokens(&messages, &[], None)
    );
}

#[test]
fn a_usage_baseline_without_a_tail_gates_as_a_provider_count() {
    let baseline = BaseMessage::assistant("answer")
        .with_metadata(USAGE_METADATA_KEY, json!({"total_tokens": 42}));
    let counted = count_context_window_tokens_with_provenance(&[baseline], &[], None);
    assert_eq!(counted.tokens, 42);
    assert!(!counted.estimated);
    assert_eq!(counted.note, None);
}

#[test]
fn a_window_counted_without_a_counter_reports_the_string_length_estimate() {
    let messages = vec![BaseMessage::user("abcdef")];
    let tools = vec![json!({"name": "x"})];
    let counted = count_messages_tokens_with_provenance(&messages, None, false);
    assert!(counted.estimated);
    assert_eq!(counted.note.as_deref(), Some(STRING_LENGTH_NOTE));
    let tools_counted = count_tools_tokens_with_provenance(&tools, None);
    assert!(tools_counted.estimated);
    assert_eq!(tools_counted.note.as_deref(), Some(STRING_LENGTH_NOTE));
    let window = count_context_window_tokens_with_provenance(&messages, &tools, None);
    assert!(window.estimated);
    assert_eq!(window.note.as_deref(), Some(STRING_LENGTH_NOTE));
    assert_eq!(window.tokens, counted.tokens + tools_counted.tokens);
}

#[test]
fn a_window_counted_by_a_counter_is_reported_as_a_counter_count() {
    let character_counter = ApproxTokenCounter::with_chars_per_token(1);
    let messages = vec![BaseMessage::user("abcdef")];
    let window =
        count_context_window_tokens_with_provenance(&messages, &[], Some(&character_counter));
    assert!(!window.estimated);
    assert_eq!(window.note, None);
    // An empty window and an empty tool list are zero, not an estimate.
    let empty = count_messages_tokens_with_provenance(&[], None, false);
    assert_eq!(empty, ContextTokenCount::counted(0));
    assert!(!empty.estimated);
    assert_eq!(window.tokens, 4 + 6);
}

#[test]
fn a_token_count_can_be_marked_estimated_by_its_counter() {
    let counted = ContextTokenCount::counted(10).with_estimated_counter(STRING_LENGTH_NOTE);
    assert_eq!(counted.tokens, 10);
    assert!(counted.estimated);
    assert_eq!(counted.note.as_deref(), Some(STRING_LENGTH_NOTE));
}

// ---------------------------------------------------------------------------
// RecallChunk: a marked, character-based size estimate
// ---------------------------------------------------------------------------

#[test]
fn recall_chunk_sizes_use_characters_not_bytes() {
    assert_eq!(estimate_tokens(""), 0);
    assert_eq!(estimate_tokens("abc"), 1);
    assert_eq!(estimate_tokens("abcdef"), 2);
    // Four CJK characters are 12 UTF-8 bytes: `bytes / 4` would say 3.
    assert_eq!(estimate_tokens("你好世界"), 1);
    assert_eq!(estimate_tokens(&"x".repeat(300)), 100);
}

#[test]
fn archived_chunks_are_marked_estimated_and_char_sized() {
    let text = "你好世界".repeat(20);
    let messages = vec![
        BaseMessage::user(text.as_str()),
        BaseMessage::assistant("好的"),
    ];
    let archive = archive_from_messages(
        "memory-1",
        "session-1",
        "context-1",
        "2026-01-01T00:00:00Z",
        &messages,
        &CompressionRecallConfig::default(),
    );
    assert_eq!(archive.chunks.len(), 1);
    for chunk in &archive.chunks {
        assert!(
            chunk.estimated,
            "every archived chunk carries the estimate flag"
        );
        assert_eq!(chunk.tokens, estimate_tokens(&chunk.text));
        assert_eq!(chunk.tokens, (chunk.text.chars().count() / 3).max(1));
        assert!(chunk.tokens * 3 <= chunk.text.len() * 3);
    }
    // A chunk estimate is bounded by the character count, never the byte count.
    let first = &archive.chunks[0];
    assert!(first.tokens < first.text.len().div_ceil(4) + 1);
}

#[test]
fn recall_chunks_default_to_estimated_and_deserialize_legacy_archives() {
    assert!(RecallChunk::default().estimated);
    let built = RecallChunk::estimated("c1", "turn_0001", 0, "abcdef");
    assert_eq!(built.tokens, 2);
    assert!(built.estimated);
    assert_eq!(built.chunk_index, 0);
    // An archive written before the field existed still reports an estimate.
    let legacy: RecallChunk = serde_json::from_value(json!({
        "chunk_id": "c1",
        "turn_id": "turn_0001",
        "chunk_index": 0,
        "text": "abcdef",
        "tokens": 2,
    }))
    .unwrap();
    assert!(legacy.estimated);
    assert_eq!(legacy, built);
    // The flag round-trips.
    let mut measured = built.clone();
    measured.estimated = false;
    let value = serde_json::to_value(&measured).unwrap();
    assert_eq!(value["estimated"], json!(false));
    let restored: RecallChunk = serde_json::from_value(value).unwrap();
    assert!(!restored.estimated);
    assert_eq!(restored.tokens, 2);
}

#[test]
fn chunk_windows_stay_whole_while_their_sizes_stay_estimates() {
    let text = "你好世界".repeat(10);
    let chunks = split_text(&text, 2, 0).unwrap();
    assert!(chunks.len() > 1);
    for chunk in &chunks {
        assert_eq!(chunk.chars().count(), 2);
        assert_eq!(estimate_tokens(chunk), 1);
    }
    assert_eq!(chunks.concat(), text);
}
