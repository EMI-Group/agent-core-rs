#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Public-API integration tests for prompt-token and KV-cache usage accounting.
//!
//! These tests pin the `context.usage` contract: the four-category
//! measurement (sections, attachments, fragment splitting), provider
//! reconciliation for native and fallback counters, the measurement cache,
//! provider `usage_metadata` normalization, session-scoped cache aggregation,
//! and the shared `context_tokens` sizing policy.
//!
//! Everything is offline and deterministic: counters are `ApproxTokenCounter`
//! instances (one character per token) or small fakes, no model is involved,
//! and no test depends on the wall clock.

use std::collections::BTreeMap;
use std::sync::Arc;

use openjiuwen_context::tokenizers::TokenizerSource;
use openjiuwen_context::usage::context_tokens::{
    canonical_json, count_context_window_tokens, count_messages_tokens, count_tools_tokens,
    count_usage_tokens_with_tail, estimate_content_tokens, estimate_message_tokens,
    has_valid_usage_metadata, USAGE_METADATA_KEY, USAGE_STALE_METADATA_KEY,
};
use openjiuwen_context::usage::{
    is_fallback_source, request_usage_from_metadata, AnalysisInput, Attribution,
    CacheAggregationKey, ContextCategory, ContextPartUsage, ContextUsageAnalyzer,
    ContextUsageSnapshot, ContextWindowTokenReport, CounterIdentity, KvCacheUsage,
    MeasurementCache, MeasurementCacheHit, PromptSection, RequestKVCacheUsage,
    SessionKVCacheAggregator, SessionKVCacheUsage, SnapshotOptions, WindowSignatures,
    CONTEXT_USAGE_EVENT_TYPE, NOT_REPORTED, SOURCE_PROVIDER_USAGE,
    SOURCE_PROVIDER_USAGE_PROPORTIONAL, SOURCE_PROVIDER_USAGE_RESIDUAL,
    SOURCE_PROVIDER_USAGE_UNATTRIBUTED, STATUS_FULL_HIT, STATUS_INVALID, STATUS_MISS,
    STATUS_NOT_REPORTED, STATUS_PARTIAL_HIT,
};
use openjiuwen_foundation::prompt::{
    ApproxTokenCounter, BaseMessage, MessageContent, Role, TokenCounter,
};
use serde_json::{json, Value};

/// The one-character-per-token counter every measurement test uses.
fn counter() -> ApproxTokenCounter {
    ApproxTokenCounter::with_chars_per_token(1)
}

/// An analyzer using a fallback (string-length) counter.
fn fallback_analyzer() -> ContextUsageAnalyzer {
    ContextUsageAnalyzer::new(Arc::new(counter()), CounterIdentity::approx())
        .with_model("test-model")
        .with_context_window_limit(Some(1_000))
}

/// An analyzer claiming a native tokenizer identity over the same counter.
fn native_analyzer() -> ContextUsageAnalyzer {
    ContextUsageAnalyzer::new(
        Arc::new(counter()),
        CounterIdentity::new("TestNativeCounter", TokenizerSource::NativeTokenizer),
    )
    .with_model("test-model")
    .with_context_window_limit(Some(1_000))
}

/// A counter that measures nothing, so the local total is zero.
struct ZeroCounter;

impl TokenCounter for ZeroCounter {
    fn count_text(&self, _text: &str) -> usize {
        0
    }

    fn count_messages(&self, _messages: &[BaseMessage]) -> usize {
        0
    }
}

/// A message with one extra metadata entry.
fn message_with(mut message: BaseMessage, key: &str, value: Value) -> BaseMessage {
    message.metadata.insert(key.to_owned(), value);
    message
}

/// The part of `category` in `snapshot`.
fn part(snapshot: &ContextUsageSnapshot, category: ContextCategory) -> &ContextPartUsage {
    snapshot
        .parts
        .get(category.as_str())
        .expect("category part")
}

/// The sum of every part's tokens.
fn part_sum(snapshot: &ContextUsageSnapshot) -> u64 {
    snapshot.parts.values().map(|part| part.tokens).sum()
}

/// An input window with a system message, two tools and one exchange.
fn sample_input<'a>(
    system: &'a [BaseMessage],
    messages: &'a [BaseMessage],
    tools: &'a [Value],
) -> AnalysisInput<'a> {
    let mut input = AnalysisInput::new("req-1", "session-1");
    input.system_messages = Some(system);
    input.context_messages = messages;
    input.tools = tools;
    input
}

// ---------------------------------------------------------------------------
// Category measurement
// ---------------------------------------------------------------------------

#[test]
fn measures_the_four_categories_with_sources_and_carriers() {
    let system = vec![BaseMessage::system("abc")];
    let messages = vec![BaseMessage::user("hello"), BaseMessage::assistant("hi")];
    let tools = vec![json!({"name": "x"})];
    let input = sample_input(&system, &messages, &tools);

    let snapshot = native_analyzer().analyze(&input);

    assert_eq!(snapshot.event_type, CONTEXT_USAGE_EVENT_TYPE);
    assert_eq!(snapshot.parts.len(), 4);
    assert_eq!(part(&snapshot, ContextCategory::SystemPrompt).tokens, 7);
    assert_eq!(part(&snapshot, ContextCategory::Tools).tokens, 12);
    assert_eq!(part(&snapshot, ContextCategory::Messages).tokens, 15);
    assert_eq!(part(&snapshot, ContextCategory::Skills).tokens, 0);

    let system_part = part(&snapshot, ContextCategory::SystemPrompt);
    assert_eq!(system_part.carrier.as_deref(), Some("system_message"));
    assert_eq!(system_part.source, "native_tokenizer");
    assert_eq!(system_part.measurement_source, "native_tokenizer");
    let skills_part = part(&snapshot, ContextCategory::Skills);
    assert_eq!(skills_part.source, NOT_REPORTED);
    assert_eq!(skills_part.carrier.as_deref(), Some("system_message"));
    assert_eq!(
        part(&snapshot, ContextCategory::Tools).carrier.as_deref(),
        Some("tools")
    );
    assert_eq!(
        part(&snapshot, ContextCategory::Messages)
            .carrier
            .as_deref(),
        Some("context_message")
    );

    assert_eq!(snapshot.context_window.local_estimated_input_tokens, 34);
    assert_eq!(snapshot.context_window.tokens, 34);
    assert_eq!(snapshot.context_window.tokens_source, "native_tokenizer");
    assert!(snapshot
        .context_window
        .reconciliation_delta_tokens
        .is_none());
    assert_eq!(snapshot.measurement.category_source, "native_tokenizer");
    assert!(!snapshot.measurement.authoritative_total);
    assert_eq!(snapshot.measurement.measurement_version, "1");
    assert_eq!(snapshot.model.as_deref(), Some("test-model"));
    assert_eq!(snapshot.request_id, "req-1");
    assert_eq!(snapshot.session_id, "session-1");
    assert_eq!(snapshot.phase, "pre_call");
    assert_ne!(snapshot.timestamp.as_str(), "");

    // Without a provider total the local total is the denominator.
    let messages_part = part(&snapshot, ContextCategory::Messages);
    assert!((messages_part.percentage_of_input.unwrap() - 15.0 / 34.0).abs() < 1e-9);
    assert!((messages_part.percentage_of_window.unwrap() - 0.015).abs() < 1e-9);
    assert!((snapshot.context_window.occupancy_rate.unwrap() - 0.034).abs() < 1e-9);
}

#[test]
fn sections_replace_the_system_messages_measurement() {
    let system = vec![BaseMessage::system("ignored entirely")];
    let messages = vec![BaseMessage::user("hello")];
    let sections = [
        PromptSection::new(ContextCategory::SystemPrompt, "abc"),
        PromptSection::new(ContextCategory::Skills, "defgh").with_carrier("skill:demo"),
    ];
    let mut input = sample_input(&system, &messages, &[]);
    input.system_prompt_sections = Some(&sections);

    let snapshot = native_analyzer().analyze(&input);

    assert_eq!(part(&snapshot, ContextCategory::SystemPrompt).tokens, 3);
    assert_eq!(part(&snapshot, ContextCategory::Skills).tokens, 5);
    assert_eq!(
        part(&snapshot, ContextCategory::Skills).carrier.as_deref(),
        Some("skill:demo")
    );
    assert_eq!(part(&snapshot, ContextCategory::Messages).tokens, 9);
    assert_eq!(snapshot.context_window.local_estimated_input_tokens, 17);

    // A `tools` or `messages` section would double count, so it folds into the
    // system prompt instead.
    let folded = [PromptSection::new(ContextCategory::Tools, "abcd")];
    input.system_prompt_sections = Some(&folded);
    let folded_snapshot = native_analyzer().analyze(&input);
    assert_eq!(
        part(&folded_snapshot, ContextCategory::SystemPrompt).tokens,
        4
    );
    assert_eq!(part(&folded_snapshot, ContextCategory::Tools).tokens, 0);
}

#[test]
fn attachments_route_whole_messages_into_accepting_categories() {
    let attachment = message_with(
        BaseMessage::user("abcdef"),
        "_context_usage_category",
        json!("system_prompt"),
    );
    let attachment = message_with(attachment, "_context_usage_carrier", json!("persona"));
    let memory = message_with(
        BaseMessage::user("wxyz"),
        "_context_usage_category",
        json!("memory"),
    );
    let routed_away = message_with(
        BaseMessage::user("stays"),
        "_context_usage_category",
        json!("tools"),
    );
    let messages = vec![attachment, memory, routed_away];
    let input = sample_input(&[], &messages, &[]);

    let snapshot = native_analyzer().analyze(&input);

    // "abcdef" (10 tokens with framing) and "wxyz" (8 tokens) are both
    // system-prompt attachments; "stays" remains history.
    assert_eq!(
        part(&snapshot, ContextCategory::SystemPrompt).tokens,
        10 + 8
    );
    assert_eq!(
        part(&snapshot, ContextCategory::SystemPrompt)
            .carrier
            .as_deref(),
        // carrier list is "system_message" (empty system messages) + "persona"
        Some("mixed")
    );
    assert_eq!(part(&snapshot, ContextCategory::Tools).tokens, 0);
    assert_eq!(part(&snapshot, ContextCategory::Messages).tokens, 9);
}

#[test]
fn fragments_split_one_carrier_by_text_length_with_remainder_last() {
    let message = message_with(
        BaseMessage::user("abcdef"),
        "_context_usage_fragments",
        json!([
            {"category": "system_prompt", "text": "abcd", "carrier": "persona"},
            {"category": "skills", "text": "ef"},
        ]),
    );
    let messages = [message];
    let input = sample_input(&[], &messages, &[]);

    let snapshot = native_analyzer().analyze(&input);

    // The carrier measures 10 tokens; weights 4:2 give 6 then the remainder 4.
    assert_eq!(part(&snapshot, ContextCategory::SystemPrompt).tokens, 6);
    assert_eq!(part(&snapshot, ContextCategory::Skills).tokens, 4);
    assert_eq!(
        part(&snapshot, ContextCategory::SystemPrompt)
            .carrier
            .as_deref(),
        Some("mixed")
    );
    assert_eq!(part(&snapshot, ContextCategory::Messages).tokens, 0);
    assert_eq!(
        part(&snapshot, ContextCategory::SystemPrompt).tokens
            + part(&snapshot, ContextCategory::Skills).tokens,
        10
    );

    // A fragment category that cannot accept attachments folds into the system
    // prompt, and a non-object fragment list falls back to a whole message.
    let folded = message_with(
        BaseMessage::user("abcdef"),
        "_context_usage_fragments",
        json!([{"category": "tools", "text": "abcdef"}]),
    );
    let folded_messages = [folded];
    let folded_input = sample_input(&[], &folded_messages, &[]);
    let folded_snapshot = native_analyzer().analyze(&folded_input);
    assert_eq!(
        part(&folded_snapshot, ContextCategory::SystemPrompt).tokens,
        10
    );
    assert_eq!(part(&folded_snapshot, ContextCategory::Skills).tokens, 0);
}

// ---------------------------------------------------------------------------
// Provider reconciliation
// ---------------------------------------------------------------------------

#[test]
fn native_reconciliation_keeps_fixed_categories_and_puts_the_rest_on_messages() {
    let system = vec![BaseMessage::system("abc")];
    let messages = vec![BaseMessage::user("hello"), BaseMessage::assistant("hi")];
    let tools = vec![json!({"name": "x"})];
    let mut input = sample_input(&system, &messages, &tools);
    input.provider_input_tokens = Some(100);

    let snapshot = native_analyzer().analyze(&input);

    assert_eq!(part(&snapshot, ContextCategory::SystemPrompt).tokens, 7);
    assert_eq!(part(&snapshot, ContextCategory::Tools).tokens, 12);
    let messages_part = part(&snapshot, ContextCategory::Messages);
    assert_eq!(messages_part.tokens, 81);
    assert_eq!(messages_part.source, SOURCE_PROVIDER_USAGE_RESIDUAL);
    assert_eq!(
        messages_part.fallback_reason.as_deref(),
        Some("provider_total_residual")
    );
    assert_eq!(snapshot.context_window.tokens, 100);
    assert_eq!(snapshot.context_window.tokens_source, SOURCE_PROVIDER_USAGE);
    assert_eq!(snapshot.context_window.local_estimated_input_tokens, 34);
    assert_eq!(
        snapshot.context_window.reconciliation_delta_tokens,
        Some(66)
    );
    assert!(snapshot.measurement.authoritative_total);
    assert_eq!(part_sum(&snapshot), 100);
    assert!((snapshot.context_window.occupancy_rate.unwrap() - 0.1).abs() < 1e-9);
}

#[test]
fn native_reconciliation_never_goes_negative() {
    let system = vec![BaseMessage::system("abc")];
    let tools = vec![json!({"name": "x"})];
    let mut input = sample_input(&system, &[], &tools);
    input.provider_input_tokens = Some(10);

    let snapshot = native_analyzer().analyze(&input);

    let messages_part = part(&snapshot, ContextCategory::Messages);
    assert_eq!(messages_part.tokens, 0);
    assert_eq!(messages_part.source, SOURCE_PROVIDER_USAGE_RESIDUAL);
    assert_eq!(
        messages_part.fallback_reason.as_deref(),
        Some("provider_total_below_fixed_categories")
    );
    assert_eq!(
        snapshot.context_window.reconciliation_delta_tokens,
        Some(-9)
    );
}

#[test]
fn fallback_reconciliation_scales_every_category_to_the_provider_total() {
    let system = vec![BaseMessage::system("abc")];
    let messages = vec![BaseMessage::user("hello"), BaseMessage::assistant("hi")];
    let tools = vec![json!({"name": "x"})];
    let mut input = sample_input(&system, &messages, &tools);
    input.provider_input_tokens = Some(100);

    let snapshot = fallback_analyzer().analyze(&input);

    // 7/34 -> 20 (+1 remainder), 12/34 -> 35, 15/34 -> 44, 0 stays.
    assert_eq!(part(&snapshot, ContextCategory::SystemPrompt).tokens, 21);
    assert_eq!(part(&snapshot, ContextCategory::Tools).tokens, 35);
    assert_eq!(part(&snapshot, ContextCategory::Messages).tokens, 44);
    assert_eq!(part(&snapshot, ContextCategory::Skills).tokens, 0);
    assert_eq!(part_sum(&snapshot), 100);
    for category in [
        ContextCategory::SystemPrompt,
        ContextCategory::Tools,
        ContextCategory::Messages,
    ] {
        let usage = part(&snapshot, category);
        assert_eq!(usage.source, SOURCE_PROVIDER_USAGE_PROPORTIONAL);
        assert_eq!(usage.measurement_source, SOURCE_PROVIDER_USAGE_PROPORTIONAL);
        assert!(usage.estimated);
        assert_eq!(
            usage.fallback_reason.as_deref(),
            Some("provider_total_proportional_reconciliation")
        );
    }
    assert_eq!(
        part(&snapshot, ContextCategory::Skills).source,
        NOT_REPORTED
    );
    assert_eq!(
        snapshot.measurement.category_source,
        SOURCE_PROVIDER_USAGE_PROPORTIONAL
    );
    assert_eq!(snapshot.context_window.tokens, 100);
    assert_eq!(part_sum(&snapshot), 100);
}

#[test]
fn proportional_reconciliation_is_exact_for_awkward_ratios() {
    let system = vec![BaseMessage::system("a")];
    let messages = vec![BaseMessage::user("bb"), BaseMessage::user("ccc")];
    let tools = vec![json!({"name": "x"}), json!({"name": "y"})];
    for provider_total in [1u64, 7, 13, 97, 10_000] {
        let mut input = sample_input(&system, &messages, &tools);
        input.provider_input_tokens = Some(provider_total);
        let snapshot = fallback_analyzer().analyze(&input);
        assert_eq!(
            part_sum(&snapshot),
            provider_total,
            "parts must sum to the provider total {provider_total}"
        );
    }
}

#[test]
fn a_zero_local_total_is_unattributed_instead_of_fake_history() {
    // Python only reaches the unattributed branch on the fallback path, where
    // a custom counter can legitimately report nothing at all.
    let analyzer = ContextUsageAnalyzer::new(Arc::new(ZeroCounter), CounterIdentity::approx())
        .with_context_window_limit(Some(100));
    let system = vec![BaseMessage::system("abc")];
    let messages = vec![BaseMessage::user("hello")];
    let mut input = sample_input(&system, &messages, &[]);
    input.provider_input_tokens = Some(50);

    let snapshot = analyzer.analyze(&input);

    let messages_part = part(&snapshot, ContextCategory::Messages);
    assert_eq!(messages_part.tokens, 50);
    assert_eq!(messages_part.source, SOURCE_PROVIDER_USAGE_UNATTRIBUTED);
    assert_eq!(
        messages_part.fallback_reason.as_deref(),
        Some("provider_total_without_local_measurement")
    );
    assert_eq!(snapshot.context_window.local_estimated_input_tokens, 0);
    assert_eq!(
        snapshot.context_window.reconciliation_delta_tokens,
        Some(50)
    );
    assert_eq!(part_sum(&snapshot), 50);
}

// ---------------------------------------------------------------------------
// Measurement cache
// ---------------------------------------------------------------------------

#[test]
fn measurement_cache_reports_miss_full_and_partial() {
    let analyzer = fallback_analyzer();
    let system = vec![BaseMessage::system("abc")];
    let first_messages = vec![BaseMessage::user("hello")];
    let second_messages = vec![BaseMessage::user("hello there")];
    let tools = vec![json!({"name": "x"})];

    let first = analyzer.analyze(&sample_input(&system, &first_messages, &tools));
    assert_eq!(first.measurement_cache.hit, MeasurementCacheHit::Miss);
    assert_eq!(first.measurement_cache.reused_parts, Vec::<String>::new());
    assert_eq!(
        first.measurement_cache.recomputed_parts,
        vec![
            "messages".to_owned(),
            "system_prompt".to_owned(),
            "tools".to_owned()
        ]
    );

    let second = analyzer.analyze(&sample_input(&system, &first_messages, &tools));
    assert_eq!(second.measurement_cache.hit, MeasurementCacheHit::Full);
    assert_eq!(
        second.measurement_cache.recomputed_parts,
        Vec::<String>::new()
    );
    assert_eq!(
        second.measurement_cache.reused_parts,
        vec![
            "messages".to_owned(),
            "system_prompt".to_owned(),
            "tools".to_owned()
        ]
    );
    assert_eq!(first.context_window.tokens, second.context_window.tokens);

    let third = analyzer.analyze(&sample_input(&system, &second_messages, &tools));
    assert_eq!(third.measurement_cache.hit, MeasurementCacheHit::Partial);
    assert_eq!(
        third.measurement_cache.reused_parts,
        vec!["system_prompt".to_owned(), "tools".to_owned()]
    );
    assert_eq!(
        third.measurement_cache.recomputed_parts,
        vec!["messages".to_owned()]
    );
    assert_ne!(
        third.context_window.tokens, first.context_window.tokens,
        "a new message list must be re-measured"
    );
}

#[test]
fn measurement_cache_is_bounded_and_can_be_disabled_or_cleared() {
    let analyzer = fallback_analyzer().with_measurement_cache_capacity(2);
    let system = vec![BaseMessage::system("abc")];
    let messages = vec![BaseMessage::user("hello")];
    let tools = vec![json!({"name": "x"})];

    let _ = analyzer.analyze(&sample_input(&system, &messages, &tools));
    assert_eq!(
        analyzer.measurement_cache_len(),
        2,
        "the cache is bounded even when the window has three parts"
    );

    // A smaller cache than the window evicts entries during the same pass, so
    // a repeated pass is not guaranteed to be a full hit.
    let second = analyzer.analyze(&sample_input(&system, &messages, &tools));
    assert_eq!(second.measurement_cache.hit, MeasurementCacheHit::Miss);

    analyzer.clear_measurement_cache();
    assert_eq!(analyzer.measurement_cache_len(), 0);

    let disabled = fallback_analyzer().with_measurement_cache_capacity(0);
    let _ = disabled.analyze(&sample_input(&system, &messages, &tools));
    assert_eq!(disabled.measurement_cache_len(), 0);
    let again = disabled.analyze(&sample_input(&system, &messages, &tools));
    assert_eq!(again.measurement_cache.hit, MeasurementCacheHit::Miss);
}

#[test]
fn the_measurement_cache_type_stores_and_evicts_entries() {
    let mut cache = MeasurementCache::new(2);
    assert!(cache.is_empty());
    assert_eq!(cache.capacity(), 2);
    let measurement = openjiuwen_context::TokenMeasurement::default();
    cache.insert("a".to_owned(), measurement.clone());
    cache.insert("b".to_owned(), measurement.clone());
    assert_eq!(cache.len(), 2);
    cache.insert("c".to_owned(), measurement);
    assert_eq!(cache.len(), 2);
    assert!(cache.get("a").is_none(), "the oldest entry is evicted");
    assert!(cache.get("c").is_some());
    cache.clear();
    assert!(cache.is_empty());
}

// ---------------------------------------------------------------------------
// Report reuse
// ---------------------------------------------------------------------------

#[test]
fn reports_freeze_the_measurement_and_rebuild_snapshots_without_measuring() {
    let analyzer = fallback_analyzer();
    let system = vec![BaseMessage::system("abc")];
    let messages = vec![BaseMessage::user("hello"), BaseMessage::assistant("hi")];
    let tools = vec![json!({"name": "x"})];
    let mut signatures = WindowSignatures {
        message_revision: 4,
        prompt_revision: 2,
        skills_revision: 1,
        attachment_revision: 3,
        tools_signature: Some("tools-v1"),
        ..WindowSignatures::default()
    };
    let components = BTreeMap::from([("skills".to_owned(), "skills-v1".to_owned())]);
    signatures.component_signatures = Some(&components);
    let mut input = sample_input(&system, &messages, &tools);
    input.context_id = Some("ctx-1");
    input.context_snapshot_id = Some("snap-1");
    input.signatures = Some(&signatures);
    let stats = openjiuwen_context::ContextStats {
        total_messages: 3,
        ..openjiuwen_context::ContextStats::default()
    };
    input.context_stats = Some(&stats);
    let attribution = Attribution {
        team_id: Some("team-1".to_owned()),
        member_name: Some("lead".to_owned()),
        depth: 2,
        ..Attribution::default()
    };
    input.attribution = Some(&attribution);

    let report = analyzer.build_report(&input);
    assert_eq!(report.context_id, "ctx-1");
    assert_eq!(report.session_id, "session-1");
    assert_eq!(report.context_snapshot_id, "snap-1");
    assert_eq!(report.message_revision, 4);
    assert_eq!(report.prompt_revision, 2);
    assert_eq!(report.skills_revision, 1);
    assert_eq!(report.attachment_revision, 3);
    assert_eq!(report.tools_signature.as_deref(), Some("tools-v1"));
    assert_eq!(
        report.tokenizer_identity.as_deref(),
        Some("ApproxTokenCounter:")
    );
    assert_eq!(report.component_signatures.len(), 1);
    assert_eq!(report.context_stats.total_messages, 3);
    assert_eq!(report.team_id.as_deref(), Some("team-1"));
    assert_eq!(report.depth, 2);
    assert_eq!(report.context_window.local_estimated_input_tokens, 34);
    assert_eq!(report.measurement.category_source, "string_length_fallback");
    assert_eq!(report.parts.len(), 4);

    let mut options = SnapshotOptions::new("req-2", "session-1");
    options.phase = "post_call";
    options.provider_input_tokens = Some(100);
    let snapshot = analyzer.analyze_from_report(&report, &options);

    assert_eq!(snapshot.phase, "post_call");
    assert_eq!(snapshot.request_id, "req-2");
    assert_eq!(snapshot.context_window.tokens, 100);
    assert!(snapshot.measurement.authoritative_total);
    assert_eq!(part_sum(&snapshot), 100);
    assert_eq!(snapshot.measurement_cache, report.measurement_cache);
    assert_eq!(snapshot.context_changes, report.context_changes);
    assert!(snapshot.context_stats.is_none());
    // The report carries the attribution and the agent id forward.
    assert_eq!(snapshot.team_id.as_deref(), Some("team-1"));
    assert_eq!(snapshot.member_name.as_deref(), Some("lead"));
    assert_eq!(snapshot.depth, 2);
    assert_eq!(snapshot.context_snapshot_id.as_deref(), Some("snap-1"));
    assert_eq!(snapshot.agent_id, None);
}

// ---------------------------------------------------------------------------
// Provider usage normalization
// ---------------------------------------------------------------------------

/// One table-driven provider-metadata expectation.
struct UsageCase {
    /// The case name, used in assertion messages.
    name: &'static str,
    /// The provider metadata.
    metadata: Value,
    /// The expected status.
    status: &'static str,
    /// The expected input tokens.
    input: Option<u64>,
    /// The expected cache-read tokens.
    read: Option<u64>,
    /// The expected cache-miss tokens.
    miss: Option<u64>,
    /// The expected cache-write tokens.
    write: Option<u64>,
    /// The expected invalid reason.
    invalid: Option<&'static str>,
    /// Whether the miss counter should be flagged as derived.
    derived: bool,
}

#[test]
#[allow(clippy::too_many_lines)] // one row per provider-metadata contract case
fn provider_metadata_normalization_is_table_driven() {
    let cases = vec![
        UsageCase {
            name: "empty",
            metadata: json!({}),
            status: STATUS_NOT_REPORTED,
            input: None,
            read: None,
            miss: None,
            write: None,
            invalid: None,
            derived: false,
        },
        UsageCase {
            name: "derived miss",
            metadata: json!({"input_tokens": 100, "cache_read_tokens": 80}),
            status: STATUS_PARTIAL_HIT,
            input: Some(100),
            read: Some(80),
            miss: Some(20),
            write: None,
            invalid: None,
            derived: true,
        },
        UsageCase {
            name: "reported miss",
            metadata: json!({"input_tokens": 100, "cache_read_tokens": 80, "cache_miss_tokens": 20}),
            status: STATUS_PARTIAL_HIT,
            input: Some(100),
            read: Some(80),
            miss: Some(20),
            write: None,
            invalid: None,
            derived: false,
        },
        UsageCase {
            name: "full hit",
            metadata: json!({"input_tokens": 100, "cache_read_tokens": 100}),
            status: STATUS_FULL_HIT,
            input: Some(100),
            read: Some(100),
            miss: Some(0),
            write: None,
            invalid: None,
            derived: true,
        },
        UsageCase {
            name: "miss",
            metadata: json!({"input_tokens": 100, "cache_read_tokens": 0, "cache_write_tokens": 0}),
            status: STATUS_MISS,
            input: Some(100),
            read: Some(0),
            miss: Some(100),
            write: Some(0),
            invalid: None,
            derived: true,
        },
        UsageCase {
            name: "read exceeds input",
            metadata: json!({"input_tokens": 10, "cache_read_tokens": 11}),
            status: STATUS_INVALID,
            input: Some(10),
            read: Some(11),
            miss: None,
            write: None,
            invalid: Some("cache_read_tokens_exceeds_input_tokens"),
            derived: false,
        },
        UsageCase {
            name: "read plus miss mismatch",
            metadata: json!({"input_tokens": 10, "cache_read_tokens": 4, "cache_miss_tokens": 5}),
            status: STATUS_INVALID,
            input: Some(10),
            read: Some(4),
            miss: Some(5),
            write: None,
            invalid: Some("cache_read_plus_miss_does_not_equal_input"),
            derived: false,
        },
        UsageCase {
            name: "write exceeds miss",
            metadata: json!({"input_tokens": 10, "cache_read_tokens": 4, "cache_write_tokens": 7}),
            status: STATUS_INVALID,
            input: Some(10),
            read: Some(4),
            miss: Some(6),
            write: Some(7),
            invalid: Some("cache_write_tokens_exceeds_cache_miss_tokens"),
            derived: true,
        },
        UsageCase {
            name: "boolean input",
            metadata: json!({"input_tokens": true, "cache_read_tokens": 4}),
            status: STATUS_INVALID,
            input: None,
            read: Some(4),
            miss: None,
            write: None,
            invalid: Some("input_tokens_is_not_an_integer"),
            derived: false,
        },
        UsageCase {
            name: "negative read",
            metadata: json!({"input_tokens": 10, "cache_read_tokens": -1}),
            status: STATUS_INVALID,
            input: Some(10),
            read: None,
            miss: None,
            write: None,
            invalid: Some("cache_read_tokens_is_negative"),
            derived: false,
        },
        UsageCase {
            name: "fractional miss",
            metadata: json!({"input_tokens": 10, "cache_read_tokens": 4, "cache_miss_tokens": 1.5}),
            status: STATUS_INVALID,
            input: Some(10),
            read: Some(4),
            miss: None,
            write: None,
            invalid: Some("cache_miss_tokens_is_not_an_integer"),
            derived: false,
        },
        UsageCase {
            name: "integral float and digit strings",
            metadata: json!({
                "input_tokens": "12",
                "cache_read_tokens": 6.0,
                "cache_miss_tokens": "6"
            }),
            status: STATUS_PARTIAL_HIT,
            input: Some(12),
            read: Some(6),
            miss: Some(6),
            write: None,
            invalid: None,
            derived: false,
        },
        UsageCase {
            name: "legacy cache tokens",
            metadata: json!({"input_tokens": 10, "cache_tokens": 4}),
            status: STATUS_PARTIAL_HIT,
            input: Some(10),
            read: Some(4),
            miss: Some(6),
            write: None,
            invalid: None,
            derived: true,
        },
        UsageCase {
            name: "legacy zero means not reported",
            metadata: json!({"input_tokens": 10, "cache_tokens": 0}),
            status: STATUS_NOT_REPORTED,
            input: Some(10),
            read: None,
            miss: None,
            write: None,
            invalid: None,
            derived: false,
        },
        UsageCase {
            name: "explicit null read falls back to legacy",
            metadata: json!({"input_tokens": 10, "cache_read_tokens": null, "cache_tokens": 3}),
            status: STATUS_PARTIAL_HIT,
            input: Some(10),
            read: Some(3),
            miss: Some(7),
            write: None,
            invalid: None,
            derived: true,
        },
        UsageCase {
            name: "trusted provider miss verdict",
            metadata: json!({"input_tokens": 10, "cache_read_tokens": 4, "cache_status": "miss"}),
            status: STATUS_MISS,
            input: Some(10),
            read: Some(4),
            miss: Some(6),
            write: None,
            invalid: None,
            derived: true,
        },
        UsageCase {
            name: "observed verdict is recomputed",
            metadata: json!({"input_tokens": 10, "cache_read_tokens": 4, "cache_status": "observed"}),
            status: STATUS_PARTIAL_HIT,
            input: Some(10),
            read: Some(4),
            miss: Some(6),
            write: None,
            invalid: None,
            derived: true,
        },
    ];
    for case in cases {
        let usage = request_usage_from_metadata(&case.metadata);
        assert_eq!(usage.status, case.status, "case {}", case.name);
        assert_eq!(usage.input_tokens, case.input, "case {}", case.name);
        assert_eq!(usage.cache_read_tokens, case.read, "case {}", case.name);
        assert_eq!(usage.cache_miss_tokens, case.miss, "case {}", case.name);
        assert_eq!(usage.cache_write_tokens, case.write, "case {}", case.name);
        assert_eq!(
            usage.invalid_reason.as_deref(),
            case.invalid,
            "case {}",
            case.name
        );
        assert_eq!(
            usage.miss_tokens_derived, case.derived,
            "case {}",
            case.name
        );
        if case.invalid.is_some() {
            assert!(usage.hit_rate.is_none(), "case {}", case.name);
        }
        assert!(usage.cache_mode.is_none(), "case {}", case.name);
        assert!(usage.cache_scope.is_none(), "case {}", case.name);
    }
}

#[test]
fn provider_metadata_keeps_raw_values_and_reports_the_source() {
    let usage = request_usage_from_metadata(&json!({
        "input_tokens": 100,
        "cache_read_tokens": 80,
        "cache_source": "vendor_cache",
        "cache_authoritative": true,
    }));
    assert_eq!(usage.source, "vendor_cache");
    assert!(usage.authoritative);
    assert!((usage.hit_rate.unwrap() - 0.8).abs() < 1e-9);

    let derived = request_usage_from_metadata(&json!({
        "input_tokens": 100,
        "cache_read_tokens": 80,
    }));
    assert_eq!(derived.source, SOURCE_PROVIDER_USAGE);
    assert!(!derived.authoritative);

    let invalid = request_usage_from_metadata(&json!({
        "input_tokens": 10,
        "cache_read_tokens": 20,
        "cache_miss_tokens": 5,
    }));
    assert_eq!(invalid.status, STATUS_INVALID);
    assert!(invalid.invalid_reason.is_some());
    assert_eq!(invalid.input_tokens, Some(10));
    assert_eq!(invalid.cache_read_tokens, Some(20));
    assert_eq!(invalid.cache_miss_tokens, Some(5));
    assert!(invalid.hit_rate.is_none());

    let no_cache = request_usage_from_metadata(&json!({
        "input_tokens": 10,
        "cache_read_tokens": 10,
    }));
    assert_eq!(no_cache.status, STATUS_FULL_HIT);

    // A non-object record carries no fields at all.
    let odd = request_usage_from_metadata(&json!("no usage"));
    assert_eq!(odd.status, STATUS_NOT_REPORTED);
    assert_eq!(odd.source, NOT_REPORTED);
}

// ---------------------------------------------------------------------------
// Session aggregation
// ---------------------------------------------------------------------------

/// A request usage with the given counters and no invalid reason.
fn usage(input: u64, read: u64, miss: Option<u64>, write: Option<u64>) -> RequestKVCacheUsage {
    RequestKVCacheUsage {
        input_tokens: Some(input),
        cache_read_tokens: Some(read),
        cache_miss_tokens: miss,
        cache_write_tokens: write,
        ..RequestKVCacheUsage::default()
    }
}

/// The default aggregation key of the tests.
fn scope_key() -> CacheAggregationKey {
    CacheAggregationKey::new("session-1", "openai", "gpt-test")
}

#[test]
fn aggregation_is_token_weighted_and_idempotent_per_request() {
    let mut aggregator = SessionKVCacheAggregator::new();
    let key = scope_key();
    let first = aggregator.record("req-1", &key, &usage(100, 80, Some(20), None));
    assert_eq!(first.calls_total, 1);
    assert_eq!(first.calls_observed, 1);
    assert_eq!(first.input_tokens_total, 100);
    assert!((first.weighted_hit_rate.unwrap() - 0.8).abs() < 1e-9);

    let repeated = aggregator.record("req-1", &key, &usage(999, 0, Some(999), None));
    assert_eq!(repeated.calls_total, 1, "a repeated id must be ignored");
    assert_eq!(repeated.input_tokens_total, 100);
    assert_eq!(repeated, first);

    let second = aggregator.record("req-2", &key, &usage(100, 0, Some(100), None));
    assert_eq!(second.calls_total, 2);
    assert_eq!(second.calls_observed, 2);
    assert_eq!(second.input_tokens_total, 200);
    assert_eq!(second.cache_read_tokens_total, 80);
    assert_eq!(second.cache_miss_tokens_total, 120);
    assert!((second.weighted_hit_rate.unwrap() - 0.4).abs() < 1e-9);
    assert!((second.average_input_per_call.unwrap() - 100.0).abs() < 1e-9);
    assert!((second.average_cache_read_per_call.unwrap() - 40.0).abs() < 1e-9);
    assert!((second.average_cache_miss_per_call.unwrap() - 60.0).abs() < 1e-9);
    assert_eq!(second.status, STATUS_PARTIAL_HIT);
    assert_eq!(second.data_quality, "authoritative");
    assert_eq!(second.aggregation_key, key);
    assert_eq!(second.scope, "session");
    assert_eq!(second.aggregation_method, "token_weighted");
}

#[test]
fn aggregation_excludes_partial_and_invalid_calls_from_the_totals() {
    let mut aggregator = SessionKVCacheAggregator::new();
    let key = scope_key();

    let invalid_reason = RequestKVCacheUsage {
        invalid_reason: Some("cache_read_tokens_exceeds_input_tokens".to_owned()),
        input_tokens: Some(10),
        cache_read_tokens: Some(20),
        ..RequestKVCacheUsage::default()
    };
    let snapshot = aggregator.record("req-bad", &key, &invalid_reason);
    assert_eq!(snapshot.calls_partial, 1);
    assert_eq!(snapshot.calls_observed, 0);
    assert_eq!(snapshot.data_quality, "invalid");
    assert_eq!(snapshot.status, STATUS_INVALID);

    let bounds = RequestKVCacheUsage {
        input_tokens: Some(10),
        cache_read_tokens: Some(5),
        cache_miss_tokens: Some(9),
        ..RequestKVCacheUsage::default()
    };
    let snapshot = aggregator.record("req-bounds", &key, &bounds);
    assert_eq!(snapshot.calls_partial, 2);
    assert_eq!(snapshot.calls_observed, 0);
    assert_eq!(snapshot.data_quality, "invalid");

    let write_bounds = RequestKVCacheUsage {
        input_tokens: Some(10),
        cache_read_tokens: Some(5),
        cache_miss_tokens: Some(5),
        cache_write_tokens: Some(6),
        ..RequestKVCacheUsage::default()
    };
    let snapshot = aggregator.record("req-write", &key, &write_bounds);
    assert_eq!(snapshot.calls_partial, 3);

    let partial = RequestKVCacheUsage {
        input_tokens: Some(10),
        cache_miss_tokens: Some(3),
        ..RequestKVCacheUsage::default()
    };
    let snapshot = aggregator.record("req-partial", &key, &partial);
    assert_eq!(snapshot.calls_partial, 4);
    assert_eq!(snapshot.data_quality, "mixed");

    let nothing = RequestKVCacheUsage::default();
    let snapshot = aggregator.record("req-none", &key, &nothing);
    assert_eq!(snapshot.calls_partial, 4);
    assert_eq!(snapshot.calls_not_reported, 1);
    assert_eq!(snapshot.calls_total, 5);
    assert_eq!(snapshot.calls_observed, 0);
    assert_eq!(snapshot.input_tokens_total, 0);
    assert_eq!(snapshot.status, STATUS_NOT_REPORTED);
}

#[test]
fn derived_misses_are_quality_derived_and_mix_with_authoritative_calls() {
    let mut aggregator = SessionKVCacheAggregator::new();
    let key = scope_key();
    let snapshot = aggregator.record("req-1", &key, &usage(100, 80, None, None));
    assert_eq!(snapshot.data_quality, "derived");
    let snapshot = aggregator.record("req-2", &key, &usage(100, 80, Some(20), None));
    assert_eq!(snapshot.data_quality, "mixed");
    assert_eq!(snapshot.calls_observed, 2);
}

#[test]
fn cache_write_coverage_distinguishes_none_partial_and_complete() {
    let mut none = SessionKVCacheAggregator::new();
    let key = scope_key();
    let snapshot = none.record("req-1", &key, &usage(10, 4, Some(6), None));
    assert_eq!(snapshot.cache_write_coverage, "none");
    assert!(snapshot.cache_write_tokens_total.is_none());

    let mut complete = SessionKVCacheAggregator::new();
    complete.record("req-1", &key, &usage(10, 4, Some(6), Some(2)));
    let snapshot = complete.record("req-2", &key, &usage(10, 6, Some(4), Some(1)));
    assert_eq!(snapshot.cache_write_coverage, "complete");
    assert_eq!(snapshot.cache_write_tokens_total, Some(3));
    assert_eq!(snapshot.input_tokens_total, 20);

    let mut partial = SessionKVCacheAggregator::new();
    partial.record("req-1", &key, &usage(10, 4, Some(6), Some(2)));
    let snapshot = partial.record("req-2", &key, &usage(10, 6, Some(4), None));
    assert_eq!(snapshot.cache_write_coverage, "partial");
    assert!(snapshot.cache_write_tokens_total.is_none());
}

#[test]
fn aggregation_status_moves_from_not_reported_to_full_hit_to_partial() {
    let mut aggregator = SessionKVCacheAggregator::new();
    let key = scope_key();
    let empty = aggregator.snapshot(&key);
    assert_eq!(empty.status, STATUS_NOT_REPORTED);
    assert_eq!(empty.data_quality, "not_reported");
    assert_eq!(empty.calls_total, 0);
    assert!(empty.weighted_hit_rate.is_none());

    let full = aggregator.record("req-1", &key, &usage(10, 10, None, None));
    assert_eq!(full.status, STATUS_FULL_HIT);

    let partial = aggregator.record("req-2", &key, &usage(10, 0, None, None));
    assert_eq!(partial.status, STATUS_PARTIAL_HIT);
    assert_eq!(partial.cache_read_tokens_total, 10);
    assert_eq!(partial.cache_miss_tokens_total, 10);
    assert!((partial.weighted_hit_rate.unwrap() - 0.5).abs() < 1e-9);

    let mut miss_only = SessionKVCacheAggregator::new();
    let snapshot = miss_only.record("req-1", &key, &usage(10, 0, None, None));
    assert_eq!(snapshot.status, STATUS_MISS);
}

#[test]
fn distinct_deployments_and_cache_identities_aggregate_separately() {
    let mut aggregator = SessionKVCacheAggregator::new();
    let base = scope_key();
    let mut other_deployment = base.clone();
    other_deployment.deployment = "https://api.example".to_owned();
    let mut other_identity = base.clone();
    other_identity.cache_identity = Some("cache-b".to_owned());
    let mut teammate = base.clone();
    teammate.member_name = Some("worker".to_owned());

    aggregator.record("req-1", &base, &usage(100, 40, None, None));
    aggregator.record("req-2", &other_deployment, &usage(100, 90, None, None));
    let teammate_snapshot = aggregator.record("req-3", &teammate, &usage(10, 0, None, None));
    aggregator.record("req-4", &other_identity, &usage(100, 100, None, None));

    assert_eq!(aggregator.scope_count(), 4);
    assert!(!aggregator.is_empty());
    assert_eq!(aggregator.snapshot(&base).calls_total, 1);
    assert_eq!(
        aggregator
            .snapshot(&other_deployment)
            .cache_read_tokens_total,
        90
    );
    assert_eq!(teammate_snapshot.member_name(), Some("worker"));
    assert_eq!(aggregator.snapshot(&other_identity).status, STATUS_FULL_HIT);
    assert_eq!(aggregator.snapshot(&teammate).status, STATUS_MISS);
}

/// Reads the aggregation key's member name, to prove the key travels with the
/// snapshot.
trait MemberName {
    /// The member name of the aggregation key.
    fn member_name(&self) -> Option<&str>;
}

impl MemberName for SessionKVCacheUsage {
    fn member_name(&self) -> Option<&str> {
        self.aggregation_key.member_name.as_deref()
    }
}

// ---------------------------------------------------------------------------
// Shared sizing policy
// ---------------------------------------------------------------------------

#[test]
fn content_estimates_floor_at_one_token_and_ignore_media_payloads() {
    assert_eq!(
        estimate_content_tokens(&MessageContent::Text(String::new())),
        1
    );
    assert_eq!(
        estimate_content_tokens(&MessageContent::Text("abcdef".to_owned())),
        2
    );
    let multimodal = BaseMessage::user(MessageContent::Parts(vec![
        openjiuwen_foundation::prompt::ContentPart::Text {
            text: "abcd".to_owned(),
        },
    ]));
    assert_eq!(estimate_message_tokens(&multimodal), 1);
    assert_eq!(multimodal.text(), "abcd");
}

#[test]
fn usage_baseline_wins_and_tails_are_estimated() {
    let baseline = message_with(
        BaseMessage::assistant("hi"),
        USAGE_METADATA_KEY,
        json!({"total_tokens": 100}),
    );
    let messages = vec![baseline.clone(), BaseMessage::user("abcdef")];
    assert!(has_valid_usage_metadata(&baseline));
    assert_eq!(count_usage_tokens_with_tail(&messages), Some(102));
    assert_eq!(
        count_context_window_tokens(&messages, &[json!({"name": "x"})], Some(&counter())),
        102,
        "the baseline already includes the system prompt and tools"
    );

    let stale = message_with(baseline.clone(), USAGE_STALE_METADATA_KEY, json!(true));
    assert!(!has_valid_usage_metadata(&stale));
    let stale_messages = vec![stale, BaseMessage::user("abcdef")];
    assert!(count_usage_tokens_with_tail(&stale_messages).is_none());

    let user_usage = message_with(
        BaseMessage::user("hi"),
        USAGE_METADATA_KEY,
        json!({"total_tokens": 100}),
    );
    assert!(!has_valid_usage_metadata(&user_usage));

    let zero = message_with(
        BaseMessage::assistant("hi"),
        USAGE_METADATA_KEY,
        json!({"total_tokens": 0}),
    );
    assert!(!has_valid_usage_metadata(&zero));

    let without_baseline = vec![BaseMessage::user("abcdef")];
    assert!(count_usage_tokens_with_tail(&without_baseline).is_none());
    assert_eq!(
        count_messages_tokens(&without_baseline, Some(&counter()), false),
        4 + 6
    );
    assert_eq!(
        count_context_window_tokens(&without_baseline, &[json!({"name": "x"})], Some(&counter())),
        4 + 6 + 12
    );
}

#[test]
fn counts_fall_back_to_estimates_without_a_counter() {
    let messages = vec![BaseMessage::user("abcdef")];
    assert_eq!(count_messages_tokens(&messages, None, false), 2);
    assert_eq!(count_messages_tokens(&[], None, false), 0);
    assert_eq!(count_tools_tokens(&[json!({"name": "x"})], None), 4);
    assert_eq!(
        canonical_json(&json!({"b": 1, "a": [2, {"d": 3, "c": 4}]})),
        r#"{"a":[2,{"c":4,"d":3}],"b":1}"#
    );
    assert_eq!(
        count_tools_tokens(&[json!({"b": 1, "a": 2})], Some(&counter())),
        r#"{"a":2,"b":1}"#.chars().count() as u64
    );
    assert_eq!(estimate_message_tokens(&BaseMessage::user("abcdefgh")), 2);
}

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------

#[test]
fn snapshots_round_trip_and_tolerate_unknown_fields() {
    let system = vec![BaseMessage::system("abc")];
    let messages = vec![BaseMessage::user("hello"), BaseMessage::assistant("hi")];
    let tools = vec![json!({"name": "x"})];
    let mut input = sample_input(&system, &messages, &tools);
    input.provider_input_tokens = Some(100);
    input.provider = Some("openai");
    input.deployment = Some("api-base");
    let kv_cache = KvCacheUsage::new(
        Some(request_usage_from_metadata(
            &json!({"input_tokens": 100, "cache_read_tokens": 80}),
        )),
        None,
    );
    input.kv_cache = Some(&kv_cache);
    let snapshot = native_analyzer().analyze(&input);

    let value = serde_json::to_value(&snapshot).unwrap();
    assert_eq!(value["type"], json!("context.usage"));
    assert!(value.get("event_type").is_none());
    assert_eq!(value["schema_version"], json!("context-usage.v1"));
    assert_eq!(value["parts"]["messages"]["tokens"], json!(81));
    assert_eq!(value["kv_cache"]["request"]["status"], json!("partial_hit"));
    let restored: ContextUsageSnapshot = serde_json::from_value(value.clone()).unwrap();
    assert_eq!(restored, snapshot);

    let mut tolerant = value.clone();
    tolerant["future_field"] = json!({"anything": true});
    tolerant["parts"]["messages"]["future_part_field"] = json!(1);
    tolerant["kv_cache"]["future_block"] = json!(1);
    let restored: ContextUsageSnapshot = serde_json::from_value(tolerant).unwrap();
    assert_eq!(restored.parts, snapshot.parts);
    assert_eq!(restored.request_id, snapshot.request_id);

    let minimal: ContextUsageSnapshot = serde_json::from_value(json!({
        "event_type": "context.usage",
        "request_id": "r",
        "session_id": "s",
    }))
    .unwrap();
    assert_eq!(minimal.event_type, CONTEXT_USAGE_EVENT_TYPE);
    assert_eq!(minimal.phase, "pre_call");
    assert_eq!(minimal.measurement.measurement_version, "1");
    assert!(minimal.parts.is_empty());
    assert!(minimal.session_kv_cache_hit_rate.is_none());
}

#[test]
fn reports_round_trip_through_json() {
    let analyzer = fallback_analyzer();
    let system = vec![BaseMessage::system("abc")];
    let messages = vec![BaseMessage::user("hello")];
    let input = sample_input(&system, &messages, &[]);
    let report = analyzer.build_report(&input);
    let value = serde_json::to_value(&report).unwrap();
    let restored: ContextWindowTokenReport = serde_json::from_value(value).unwrap();
    assert_eq!(restored, report);
    let empty: ContextWindowTokenReport = serde_json::from_value(json!({})).unwrap();
    assert_eq!(empty.measurement_version, "1");
    assert!(empty.parts.is_empty());
}

#[test]
fn categories_normalize_legacy_names_and_serialize_snake_case() {
    let from_memory: ContextCategory = serde_json::from_value(json!("memory")).unwrap();
    assert_eq!(from_memory, ContextCategory::SystemPrompt);
    let unknown: ContextCategory = serde_json::from_value(json!("something-new")).unwrap();
    assert_eq!(unknown, ContextCategory::SystemPrompt);
    let skills: ContextCategory = serde_json::from_value(json!("skills")).unwrap();
    assert_eq!(skills, ContextCategory::Skills);
    assert_eq!(serde_json::to_value(skills).unwrap(), json!("skills"));
    assert_eq!(ContextCategory::ALL.len(), 4);
    assert!(ContextCategory::SystemPrompt.accepts_attachment());
    assert!(ContextCategory::Skills.accepts_attachment());
    assert!(!ContextCategory::Tools.accepts_attachment());
    assert!(!ContextCategory::Messages.accepts_attachment());
    assert!(is_fallback_source("string_length_fallback"));
    assert!(is_fallback_source("tiktoken_fallback"));
    assert!(!is_fallback_source("native_tokenizer"));
    assert!(!is_fallback_source("tiktoken"));
}

#[test]
fn counter_identity_marks_fallbacks_and_keeps_a_stable_string() {
    let fallback = CounterIdentity::approx();
    assert!(fallback.is_fallback());
    assert_eq!(fallback.source_name(), "string_length_fallback");
    assert_eq!(fallback.identity_string(), "ApproxTokenCounter:");

    let native = CounterIdentity::new("TestNative", TokenizerSource::NativeTokenizer)
        .with_tokenizer("cl100k_base");
    assert!(!native.is_fallback());
    assert_eq!(native.identity_string(), "TestNative:cl100k_base");

    let model = openjiuwen_context::ModelTokenCounter::new("gpt-4o");
    let identity = CounterIdentity::model(&model);
    assert_eq!(identity.class, "ModelTokenCounter");
    assert!(!identity.is_fallback());
    assert!(identity.tokenizer.is_some());

    // An unknown model resolves to a fallback vocabulary.
    let unknown = openjiuwen_context::ModelTokenCounter::new("totally-unknown-model");
    assert!(CounterIdentity::model(&unknown).is_fallback());

    // The analyzer exposes its identity and limit.
    let analyzer = fallback_analyzer();
    assert_eq!(analyzer.context_window_limit(), Some(1_000));
    assert_eq!(analyzer.model(), "test-model");
    assert!(analyzer.counter_identity().is_fallback());
    assert_eq!(analyzer.measurement_cache_len(), 0);
}

#[test]
fn role_and_metadata_helpers_stay_consistent() {
    let assistant = BaseMessage::assistant("hi");
    assert_eq!(assistant.role, Role::Assistant);
    assert!(count_usage_tokens_with_tail(&[assistant]).is_none());
    assert_eq!(
        part_sum(&native_analyzer().analyze(&AnalysisInput::new("r", "s"))),
        0
    );
}
