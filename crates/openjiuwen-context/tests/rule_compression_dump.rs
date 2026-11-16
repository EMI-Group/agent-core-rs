#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
// Fixture builders append `format!` output to a growing `String` on purpose.
#![allow(
    clippy::format_push_string,
    clippy::format_collect,
    clippy::useless_format
)]
//! Dump durability and honesty tests for the rule-compression tier.
//!
//! The suites drive the public dump surface end to end: the JSONL writer
//! (one appended line per application), the pretty-JSON writer, both readers,
//! malformed-input reporting, the `{session_id}`/`{context_id}`/`{date}` template
//! expansion, the real session/context ids a pass records, the `lossy` flag in
//! message metadata, and Python's three offload strategies through
//! [`MessageSummaryOffloader`].
//!
//! Everything is offline and deterministic: the pipelines use an injected clock,
//! dump directories are `tempfile` trees, the counter is one character per token,
//! offload storage is [`InMemoryOffloadStore`], and the one model-backed test uses
//! [`ScriptedModel`] only to prove it is *not* called.
use std::sync::Arc;

use openjiuwen_context::processor::rule_compression::{
    jsonl_file_name, read_dump_dir, read_dump_file, read_dump_text, utc_date_stamp,
    PlainTextCompressor, RuleCompressionDumpFormat, RuleCompressionPipeline,
    RuleCompressionStrategy, RuleCompressor, RuleContext, ADD_PASS, RULE_COMPRESSED_AT_KEY,
    RULE_COMPRESSION_DUMP_PATH_KEY, RULE_COMPRESSION_LOSSY_KEY, RULE_COMPRESSION_TYPE_KEY,
};
use openjiuwen_context::processor::summary_offloader::RuleCompressionTierConfig;
use openjiuwen_context::processor::{
    ContextProcessor, MessageOffloadRole, MessageSummaryOffloader, MessageSummaryOffloaderConfig,
    ProcessorContext, OFFLOADED_INFO_TAG,
};
use openjiuwen_context::{
    ContextConfig, InMemoryOffloadStore, OffloadScope, OffloadStore, TokenCounterConfig,
    TokenCounterKind,
};
use openjiuwen_foundation::llm::Model;
use openjiuwen_foundation::prompt::{ApproxTokenCounter, BaseMessage, ToolCall};
use openjiuwen_test_support::ScriptedModel;
use tokio_util::sync::CancellationToken;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// 2024-02-29T12:00:00Z, so `{date}` must expand to `20240229`.
const LEAP_DAY: f64 = 1_709_208_000.0;

/// A clock that advances one second per read, so two applications of the same
/// tool call cannot overwrite each other's pretty-printed dump file.
fn ticking_clock(start: f64) -> impl Fn() -> f64 + Send + Sync + 'static {
    let ticks = Arc::new(std::sync::atomic::AtomicU32::new(0));
    move || {
        let tick = ticks.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        start + f64::from(tick)
    }
}

/// A plain-text body: repeated prose lines, blank runs, no routing signals.
fn plain_text_body() -> String {
    let mut body = String::new();
    for _ in 0..5 {
        for index in 0..40 {
            body.push_str(&format!(
                "the quick brown fox line {index} keeps jumping over the lazy dog\n"
            ));
        }
        body.push_str("\n\n");
    }
    body
}

/// A conversation of one user turn, one tool call, and the tool result.
fn conversation(user: &str, tool_name: &str, content: &str) -> Vec<BaseMessage> {
    let call = ToolCall {
        id: Some("call_1".to_owned()),
        name: tool_name.to_owned(),
        arguments: "{\"pattern\": \"fox\"}".to_owned(),
        ..ToolCall::default()
    };
    vec![
        BaseMessage::user(user),
        BaseMessage::assistant("calling").with_tool_calls(vec![call]),
        BaseMessage::tool("call_1", content),
    ]
}

/// The scope a processor runs under.
fn scope() -> OffloadScope {
    OffloadScope::new("session-dump", "main")
}

/// A character-per-token context configuration, so budgets are character budgets.
fn character_config() -> ContextConfig {
    ContextConfig {
        token_counter: TokenCounterConfig {
            kind: TokenCounterKind::Characters,
            chars_per_token: 1,
        },
        ..ContextConfig::default()
    }
}

/// The processor context a pass runs with.
fn processor_context<'a>(
    config: &'a ContextConfig,
    counter: &'a ApproxTokenCounter,
    store: &'a InMemoryOffloadStore,
    scope: &'a OffloadScope,
    cancellation: &'a CancellationToken,
) -> ProcessorContext<'a> {
    ProcessorContext {
        config,
        counter,
        model: None,
        offload: store,
        scope,
        budget: 100_000,
        trigger_tokens: 80_000,
        manual: false,
        cancellation,
        recall: None,
        session_memory: None,
        session_memory_agent: None,
    }
}

/// An offloader that rule-compresses into `max_chars` before summarising.
fn offloader_config(max_chars: usize) -> MessageSummaryOffloaderConfig {
    MessageSummaryOffloaderConfig {
        large_message_threshold: 100,
        offload_message_types: vec![MessageOffloadRole::Tool],
        protected_tool_names: Vec::new(),
        summary_max_tokens: 900,
        content_max_chars_for_compression: 300,
        enable_rule_compression: true,
        rule_compression_max_chars: max_chars,
        enable_rule_compression_dump: false,
        rule_compression_dump_dir: None,
    }
}

// ---------------------------------------------------------------------------
// JSONL writer and readers
// ---------------------------------------------------------------------------

#[test]
fn jsonl_appends_one_line_per_application_and_round_trips() {
    let dir = tempfile::tempdir().expect("temp dir");
    let pipeline = RuleCompressionPipeline::with_clock(|| LEAP_DAY)
        .with_dump(true, Some(dir.path().to_string_lossy().into_owned()))
        .with_dump_format(RuleCompressionDumpFormat::Jsonl);
    let messages = conversation("find the fox", "grep", &plain_text_body());
    let scope = scope();
    let first = pipeline
        .compress_in_messages_with_scope(&messages, 2, ADD_PASS, 2_000, false, &scope)
        .expect("first pass");
    let second = pipeline
        .compress_in_messages_with_scope(&messages, 2, ADD_PASS, 1_500, false, &scope)
        .expect("second pass");
    let first_path = first.metadata[RULE_COMPRESSION_DUMP_PATH_KEY]
        .as_str()
        .expect("dump path")
        .to_owned();
    assert_eq!(
        first.metadata[RULE_COMPRESSION_DUMP_PATH_KEY],
        second.metadata[RULE_COMPRESSION_DUMP_PATH_KEY]
    );
    let expected_name = jsonl_file_name(&scope.session_id, &scope.context_id);
    assert_eq!(
        std::path::Path::new(&first_path)
            .file_name()
            .and_then(|name| name.to_str()),
        Some(expected_name.as_str())
    );
    // One file for the whole conversation, one line per application.
    assert_eq!(std::fs::read_dir(dir.path()).expect("read dir").count(), 1);
    let text = std::fs::read_to_string(&first_path).expect("dump text");
    assert_eq!(text.lines().count(), 2);
    assert!(text.lines().all(|line| !line.trim().is_empty()));

    let records = read_dump_file(std::path::Path::new(&first_path)).expect("reader");
    assert_eq!(records.len(), 2);
    assert_eq!(records[0].original_content, plain_text_body());
    assert!(records[1].compressed_content.len() < records[1].original_content.len());
    assert_eq!(records[0].session_id, "session-dump");
    assert_eq!(records[0].context_id, "main");
    assert_eq!(records[0].rule_compression_pass, ADD_PASS);
    assert_eq!(records[0].rule_compression_type, "PLAIN_TEXT");
    assert_eq!(records[0].original_chars, plain_text_body().chars().count());
    assert!(records.iter().all(|record| record.lossy));
    assert_eq!(records[0].path, std::path::Path::new(&first_path));
}

#[test]
fn the_pretty_form_also_round_trips_through_the_directory_reader() {
    let dir = tempfile::tempdir().expect("temp dir");
    let pipeline = RuleCompressionPipeline::with_clock(ticking_clock(LEAP_DAY))
        .with_dump(true, Some(dir.path().to_string_lossy().into_owned()));
    assert_eq!(pipeline.dump_format(), RuleCompressionDumpFormat::Json);
    let messages = conversation("find the fox", "grep", &plain_text_body());
    let compressed = pipeline
        .compress_in_messages_with_scope(&messages, 2, ADD_PASS, 2_000, false, &scope())
        .expect("pass");
    let path = compressed.metadata[RULE_COMPRESSION_DUMP_PATH_KEY]
        .as_str()
        .expect("dump path")
        .to_owned();
    assert!(std::path::Path::new(&path)
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("json")));
    // Two applications produce two files, both readable through the directory.
    pipeline
        .compress_in_messages_with_scope(&messages, 2, ADD_PASS, 1_500, false, &scope())
        .expect("second pass");
    let records = read_dump_dir(dir.path()).expect("reader");
    assert_eq!(records.len(), 2);
    assert!(records
        .iter()
        .all(|record| record.rule_compression_type == "PLAIN_TEXT"));
    assert!(records
        .iter()
        .all(|record| record.path.parent() == Some(dir.path())));
    // The fields the pretty dump writes are the fields the reader exposes.
    assert_eq!(records[0].session_id, "session-dump");
    assert_eq!(records[0].context_id, "main");
    assert_eq!(
        records[0].compressed_chars,
        records[0].compressed_content.chars().count()
    );
}

#[test]
fn malformed_dump_input_is_reported_with_its_line_number() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("broken.jsonl");
    std::fs::write(
        &path,
        "{\"timestamp\": 1.0, \"tool_call_id\": \"call_1\"}\n\n{this is not json}\n",
    )
    .expect("write");
    let error = read_dump_file(&path).expect_err("malformed line");
    assert!(error.to_string().contains("line"));
    assert!(error.to_string().contains(":3"));
    // A blank-line-only file is empty, not an error.
    assert_eq!(read_dump_text("\n\n", "memory").expect("empty"), Vec::new());
}

#[test]
fn template_expansion_places_dumps_under_the_session_context_and_date() {
    let dir = tempfile::tempdir().expect("temp dir");
    let template = dir
        .path()
        .join("{date}/{session_id}-{context_id}")
        .to_string_lossy()
        .into_owned();
    let pipeline = RuleCompressionPipeline::with_clock(|| LEAP_DAY)
        .with_dump(true, Some(template))
        .with_dump_format(RuleCompressionDumpFormat::Jsonl);
    let messages = conversation("find the fox", "grep", &plain_text_body());
    let scope = OffloadScope::new("session/one", "ctx:one");
    let compressed = pipeline
        .compress_in_messages_with_scope(&messages, 2, ADD_PASS, 2_000, false, &scope)
        .expect("pass");
    let path = compressed.metadata[RULE_COMPRESSION_DUMP_PATH_KEY]
        .as_str()
        .expect("dump path");
    let expected = dir
        .path()
        .join(utc_date_stamp(LEAP_DAY))
        .join("session-one-ctx-one");
    assert_eq!(
        std::path::Path::new(path).parent(),
        Some(expected.as_path())
    );
    assert!(expected.is_dir());
    assert_eq!(
        std::path::Path::new(path)
            .file_name()
            .and_then(|name| name.to_str()),
        Some("session-one-ctx-one.jsonl")
    );
    assert_eq!(utc_date_stamp(LEAP_DAY), "20240229");
}

#[test]
fn dumps_carry_the_real_scope_ids_and_the_placeholders_only_without_a_scope() {
    let dir = tempfile::tempdir().expect("temp dir");
    let pipeline = RuleCompressionPipeline::with_clock(|| LEAP_DAY)
        .with_dump(true, Some(dir.path().to_string_lossy().into_owned()))
        .with_dump_format(RuleCompressionDumpFormat::Jsonl);
    let messages = conversation("find the fox", "grep", &plain_text_body());
    let scoped = pipeline
        .compress_in_messages_with_scope(&messages, 2, ADD_PASS, 2_000, false, &scope())
        .expect("scoped pass");
    let unscoped = pipeline
        .compress_in_messages(&messages, 2, ADD_PASS, 2_000, false)
        .expect("unscoped pass");
    assert_ne!(
        scoped.metadata[RULE_COMPRESSION_DUMP_PATH_KEY],
        unscoped.metadata[RULE_COMPRESSION_DUMP_PATH_KEY],
        "a scopeless pass lands in its own conversation file"
    );
    let records = read_dump_dir(dir.path()).expect("reader");
    assert_eq!(records.len(), 2);
    assert_eq!(records[0].session_id, "session-dump");
    assert_eq!(records[0].context_id, "main");
    assert_eq!(records[1].session_id, "unknown_session");
    assert_eq!(records[1].context_id, "unknown_context");
}

// ---------------------------------------------------------------------------
// `lossy` honesty
// ---------------------------------------------------------------------------

#[test]
fn plain_text_compression_is_marked_lossy_in_the_metadata() {
    let body = plain_text_body();
    let direct = PlainTextCompressor.compress(&body, &RuleContext::new(2_000));
    assert!(direct.modified);
    assert!(direct.lossy, "dropped duplicate and blank lines are a loss");

    let pipeline = RuleCompressionPipeline::with_clock(|| LEAP_DAY);
    let messages = conversation("find the fox", "grep", &body);
    let compressed = pipeline
        .compress(
            &messages[2],
            &RuleContext::new(2_000),
            ADD_PASS,
            2_000,
            true,
        )
        .expect("pass");
    assert_eq!(compressed.metadata[RULE_COMPRESSION_LOSSY_KEY], true);
    assert_eq!(compressed.metadata[RULE_COMPRESSION_TYPE_KEY], "PLAIN_TEXT");
    assert!(compressed.metadata.contains_key(RULE_COMPRESSED_AT_KEY));
}

#[test]
fn a_faithful_uniqueness_preserving_rewrite_is_not_marked_lossy() {
    // Unique lines with irregular spacing: nothing is dropped, so the flag stays
    // false while the body still shrinks.
    let mut body = String::new();
    for index in 0..60 {
        body.push_str(&format!(
            "alpha     bravo\t\tcharlie        delta {index}\n"
        ));
    }
    let direct = PlainTextCompressor.compress(&body, &RuleContext::new(2_000));
    assert!(direct.modified);
    assert!(direct.lossy, "collapsed internal spacing is a loss");
    let unique = (0..60)
        .map(|index| format!("unique line {index}\n"))
        .collect::<String>();
    let unchanged = PlainTextCompressor.compress(&unique, &RuleContext::new(200));
    assert!(!unchanged.modified);
    assert!(!unchanged.lossy);
}

// ---------------------------------------------------------------------------
// Strategy switch through `MessageSummaryOffloader`
// ---------------------------------------------------------------------------

#[tokio::test]
async fn rule_then_truncate_falls_through_to_the_summary_tier() {
    let config = character_config();
    let counter = ApproxTokenCounter::with_chars_per_token(1);
    let store = InMemoryOffloadStore::new();
    let scope = scope();
    let cancellation = CancellationToken::new();
    let processor = MessageSummaryOffloader::new(
        offloader_config(2_000),
        None,
        Arc::new(InMemoryOffloadStore::new()) as Arc<dyn OffloadStore>,
    );
    assert_eq!(
        processor.rule_compression_tier().strategy,
        RuleCompressionStrategy::RuleThenTruncate
    );
    let mut messages = conversation("find the fox", "grep", &plain_text_body());
    let event = processor
        .process(
            &mut messages,
            &processor_context(&config, &counter, &store, &scope, &cancellation),
        )
        .await
        .expect("process")
        .expect("an event");
    assert_eq!(event.messages_changed, vec![2]);
    let text = messages[2].content.as_text().expect("text");
    assert!(text.contains(OFFLOADED_INFO_TAG));
    assert!(messages[2].metadata.contains_key(RULE_COMPRESSED_AT_KEY));
    assert_eq!(processor.state.counter("rule_compressed"), 1);
    assert_eq!(processor.state.counter("offloaded"), 1);
}

#[tokio::test]
async fn rule_only_keeps_the_oversized_rule_result_and_never_calls_the_model() {
    let config = character_config();
    let counter = ApproxTokenCounter::with_chars_per_token(1);
    let store = InMemoryOffloadStore::new();
    let scope = scope();
    let cancellation = CancellationToken::new();
    let scripted = ScriptedModel::builder().respond_text("summary").build();
    let model: Arc<dyn Model> = Arc::new(scripted.clone());
    let processor = MessageSummaryOffloader::new(
        offloader_config(2_000),
        Some(model),
        Arc::new(InMemoryOffloadStore::new()) as Arc<dyn OffloadStore>,
    )
    .with_rule_compression_tier(RuleCompressionTierConfig::new(
        RuleCompressionStrategy::RuleOnly,
    ));
    let mut messages = conversation("find the fox", "grep", &plain_text_body());
    let event = processor
        .process(
            &mut messages,
            &processor_context(&config, &counter, &store, &scope, &cancellation),
        )
        .await
        .expect("process")
        .expect("an event");
    assert_eq!(event.messages_changed, vec![2]);
    let text = messages[2].content.as_text().expect("text");
    assert!(!text.contains(OFFLOADED_INFO_TAG));
    assert!(!text.contains("[[OFFLOAD:"));
    assert!(
        text.chars().count() > 2_000,
        "the oversized body is kept as is"
    );
    assert_eq!(processor.state.counter("rule_compressed"), 1);
    assert_eq!(processor.state.counter("offloaded"), 0);
    assert_eq!(scripted.call_count(), 0, "rule_only never summarises");
}

#[tokio::test]
async fn rule_only_leaves_a_declined_candidate_untouched() {
    let config = character_config();
    let counter = ApproxTokenCounter::with_chars_per_token(1);
    let store = InMemoryOffloadStore::new();
    let scope = scope();
    let cancellation = CancellationToken::new();
    let processor = MessageSummaryOffloader::new(
        offloader_config(2_000),
        None,
        Arc::new(InMemoryOffloadStore::new()) as Arc<dyn OffloadStore>,
    )
    .with_rule_compression_tier(RuleCompressionTierConfig::new(
        RuleCompressionStrategy::RuleOnly,
    ));
    // Unique characters leave the plain-text compressor nothing to drop, so the
    // rule tier declines and `rule_only` must not summarise either.
    let mut messages = conversation("huge blob", "grep", &"z".repeat(3_000));
    assert!(processor
        .process(
            &mut messages,
            &processor_context(&config, &counter, &store, &scope, &cancellation),
        )
        .await
        .expect("process")
        .is_none());
    assert_eq!(messages[2].content.as_text().map(str::len), Some(3_000));
    assert_eq!(processor.state.counter("offloaded"), 0);
}

#[tokio::test]
async fn truncate_only_skips_rule_compression() {
    let config = character_config();
    let counter = ApproxTokenCounter::with_chars_per_token(1);
    let store = InMemoryOffloadStore::new();
    let scope = scope();
    let cancellation = CancellationToken::new();
    let processor = MessageSummaryOffloader::new(
        offloader_config(2_000),
        None,
        Arc::new(InMemoryOffloadStore::new()) as Arc<dyn OffloadStore>,
    )
    .with_rule_compression_tier(RuleCompressionTierConfig::new(
        RuleCompressionStrategy::TruncateOnly,
    ));
    let mut messages = conversation("find the fox", "grep", &plain_text_body());
    let event = processor
        .process(
            &mut messages,
            &processor_context(&config, &counter, &store, &scope, &cancellation),
        )
        .await
        .expect("process")
        .expect("an event");
    assert_eq!(event.messages_changed, vec![2]);
    let text = messages[2].content.as_text().expect("text");
    assert!(text.contains(OFFLOADED_INFO_TAG));
    assert!(!messages[2].metadata.contains_key(RULE_COMPRESSED_AT_KEY));
    assert_eq!(processor.state.counter("rule_compressed"), 0);
    assert_eq!(processor.state.counter("offloaded"), 1);
    // The pipeline itself is a no-op under the same strategy.
    let pipeline =
        RuleCompressionPipeline::new().with_strategy(RuleCompressionStrategy::TruncateOnly);
    let untouched = pipeline
        .compress_in_messages(&messages, 2, ADD_PASS, 10, true)
        .expect("pass");
    assert!(untouched.metadata.is_empty());
}

#[tokio::test]
async fn the_tier_config_selects_the_dump_format_and_real_ids_reach_the_dump() {
    let config = character_config();
    let counter = ApproxTokenCounter::with_chars_per_token(1);
    let store = InMemoryOffloadStore::new();
    let dir = tempfile::tempdir().expect("temp dir");
    let scope = OffloadScope::new("session-tier", "main");
    let cancellation = CancellationToken::new();
    let processor = MessageSummaryOffloader::new(
        MessageSummaryOffloaderConfig {
            enable_rule_compression_dump: true,
            rule_compression_dump_dir: Some(dir.path().to_string_lossy().into_owned()),
            ..offloader_config(2_000)
        },
        None,
        Arc::new(InMemoryOffloadStore::new()) as Arc<dyn OffloadStore>,
    )
    .with_rule_compression_tier(RuleCompressionTierConfig {
        strategy: RuleCompressionStrategy::RuleThenTruncate,
        dump_format: RuleCompressionDumpFormat::Jsonl,
    });
    assert_eq!(
        processor.rule_compression_tier().dump_format,
        RuleCompressionDumpFormat::Jsonl
    );
    let mut messages = conversation("find the fox", "grep", &plain_text_body());
    processor
        .process(
            &mut messages,
            &processor_context(&config, &counter, &store, &scope, &cancellation),
        )
        .await
        .expect("process")
        .expect("an event");
    let records = read_dump_dir(dir.path()).expect("reader");
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].session_id, "session-tier");
    assert_eq!(records[0].context_id, "main");
    assert_eq!(records[0].tool_call_id, "call_1");
    assert!(records[0].lossy);
}

#[test]
fn an_unknown_strategy_or_format_is_a_typed_config_error() {
    let unknown_strategy = serde_json::json!({"strategy": "rule_and_truncate"});
    let error = RuleCompressionTierConfig::from_json_value(&unknown_strategy)
        .expect_err("unknown strategy");
    assert!(error.to_string().contains("rule_and_truncate"));
    let unknown_format = serde_json::json!({"dump_format": "yaml"});
    assert!(RuleCompressionTierConfig::from_json_value(&unknown_format).is_err());
    assert!(serde_json::from_value::<RuleCompressionTierConfig>(unknown_strategy).is_err());
    let parsed = RuleCompressionTierConfig::from_json_value(&serde_json::json!({
        "strategy": "rule_only",
        "dump_format": "jsonl",
    }))
    .expect("valid");
    assert_eq!(parsed.strategy, RuleCompressionStrategy::RuleOnly);
    assert_eq!(parsed.dump_format, RuleCompressionDumpFormat::Jsonl);
    assert!(!parsed.runs_fallback_offload());
    // The default matches Python's `_OFFLOAD_STRATEGY`.
    assert_eq!(
        RuleCompressionTierConfig::default().strategy,
        RuleCompressionStrategy::RuleThenTruncate
    );
}
