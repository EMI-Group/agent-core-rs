#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
// Fixture builders append `format!` output to a growing `String` on purpose.
#![allow(clippy::format_push_string, clippy::useless_format)]
//! Public-API tests for deterministic rule compression.
//!
//! The suites drive the router, every compressor, the pipeline and the
//! `MessageSummaryOffloader` tier-1 integration through their public surface:
//! they assert the detection matrix, the savings/no-op gates, the `details`
//! payloads, metadata stamping, the `force` bypass, dump files, query-term
//! extraction, and the end-to-end "rule compress, then offload the remainder"
//! path with its stored payload and marker shape.
//!
//! Everything is offline and deterministic: the pipelines use an injected clock
//! or no clock at all, the counter is one character per token, offload storage is
//! [`InMemoryOffloadStore`], and no model is ever configured.
use std::sync::Arc;

use openjiuwen_context::processor::rule_compression::{
    count_tokens, extract_query_terms, meets_savings_ratio, ContentType, DiffCompressor,
    HtmlCompressor, JsonArrayCompressor, LogCompressor, PlainTextCompressor,
    RuleCompressionPipeline, RuleCompressor, RuleContentRouter, RuleContext,
    SearchResultsCompressor, SourceFileCompressor, ADD_PASS, RULE_COMPRESSED_AT_KEY,
    RULE_COMPRESSION_DETAILS_KEY, RULE_COMPRESSION_DUMP_PATH_KEY, RULE_COMPRESSION_PASS_KEY,
    RULE_COMPRESSION_TYPE_KEY, TTL_PASS,
};
use openjiuwen_context::processor::{
    ContextProcessor, MessageOffloadRole, MessageSummaryOffloader, MessageSummaryOffloaderConfig,
    ProcessorContext, OFFLOADED_INFO_TAG,
};
use openjiuwen_context::{
    ContextConfig, InMemoryOffloadStore, OffloadMarker, OffloadScope, OffloadStore,
    TokenCounterConfig, TokenCounterKind,
};
use openjiuwen_foundation::prompt::{ApproxTokenCounter, BaseMessage, Role, ToolCall};
use tokio_util::sync::CancellationToken;

/// Whether the pipeline stamped the message as rule-compressed.
fn is_rule_compressed(message: &BaseMessage) -> bool {
    message.metadata.contains_key(RULE_COMPRESSED_AT_KEY)
}

/// A context with `max_tokens` and every other field at its default.
fn ctx(max_tokens: usize) -> RuleContext {
    RuleContext::new(max_tokens)
}

/// `count` copies of one line, so plain-text dedupe has work to do.
fn repeated_lines(count: usize) -> String {
    let mut lines = String::new();
    for index in 0..count {
        lines.push_str(&format!("the same progress line {}\n", index % 3));
    }
    lines
}

/// A synthetic HTML page that keeps real readable structure.
fn html_page(paragraphs: usize) -> String {
    let mut page = String::from(
        "<!doctype html><html><head><title>Payment report</title>\
         <style>body { color: red; }</style><script>var tracking = 1;</script></head><body>",
    );
    page.push_str("<nav>home | docs | pricing</nav><h1>Payment report</h1>");
    for index in 0..paragraphs {
        page.push_str(&format!(
            "<section><h2>Section {index}</h2><p>Paragraph {index} explains how the payment \
             pipeline retries a failed capture and reports the error to the caller.</p>\
             <ul><li>first item {index}</li><li>second item {index}</li></ul></section>"
        ));
    }
    page.push_str("</body></html>");
    page
}

/// A pytest-shaped log with `errors` failing tests.
fn pytest_log(errors: usize) -> String {
    let mut lines = vec![
        "= test session starts =".to_owned(),
        "collected 120 items".to_owned(),
    ];
    for index in 0..errors {
        lines.push(format!(
            "2024-01-01T00:00:00Z ERROR test_payment.py::test_capture_{index} failed"
        ));
        lines.push(format!(
            "E   AssertionError: expected payment {index} to settle, got a timeout"
        ));
    }
    lines.push("= 12 failed, 108 passed in 3.11s =".to_owned());
    lines.join("\n")
}

/// A unified diff with `files` files of `changes` changed lines each.
fn unified_diff(files: usize, changes: usize) -> String {
    let mut diff = String::new();
    for file in 0..files {
        diff.push_str(&format!("diff --git a/src/f{file}.rs b/src/f{file}.rs\n"));
        diff.push_str(&format!("index 1111111..2222222 100644\n"));
        diff.push_str(&format!("--- a/src/f{file}.rs\n"));
        diff.push_str(&format!("+++ b/src/f{file}.rs\n"));
        diff.push_str(&format!("@@ -1,{changes} +1,{changes} @@\n"));
        for line in 0..changes {
            diff.push_str(&format!("-old line {line} of file {file}\n"));
            diff.push_str(&format!("+new line {line} of file {file}\n"));
        }
    }
    diff
}

/// `lines` search hits spread over `files` files.
fn search_output(files: usize, lines: usize) -> String {
    (0..lines)
        .map(|index| {
            format!(
                "src/module{}.rs:{}:    fn duplicate_handle_{index}() -> Result<()> {{",
                index % files,
                index + 1
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

// ---------------------------------------------------------------------------
// Detection matrix
// ---------------------------------------------------------------------------

#[test]
fn detection_types_every_tool_result_shape() {
    let router = RuleContentRouter::new();
    let plain = ctx(10_000);
    assert_eq!(
        router.detect("The quick brown fox jumps over the lazy dog.", &plain),
        ContentType::PlainText
    );
    assert_eq!(
        router.detect(
            r#"[{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]"#,
            &plain
        ),
        ContentType::JsonArray
    );
    assert_eq!(
        router.detect(
            "<!doctype html><html><body><p>hello</p></body></html>",
            &plain
        ),
        ContentType::Html
    );
    assert_eq!(router.detect(&pytest_log(3), &plain), ContentType::Log);
    assert_eq!(
        router.detect(&unified_diff(2, 4), &plain),
        ContentType::GitDiff
    );
    assert_eq!(
        router.detect(&search_output(2, 6), &plain),
        ContentType::SearchResults
    );
    // BUILD_OUTPUT is Python's alias of LOG.
    assert_eq!(ContentType::BUILD_OUTPUT, ContentType::Log);
    assert_eq!(ContentType::BUILD_OUTPUT.as_str(), "LOG");
}

#[test]
fn numbered_read_file_output_is_stripped_before_detection() {
    let router = RuleContentRouter::new();
    let plain = ctx(10_000);
    let numbered_json = "     1\t[\n     2\t  {\"id\": 1}\n     3\t]\n";
    assert_eq!(router.detect(numbered_json, &plain), ContentType::JsonArray);
    let numbered_diff = "     1\tdiff --git a/src/a.rs b/src/a.rs\n\
                         2\t--- a/src/a.rs\n\
                         3\t+++ b/src/a.rs\n\
                         4\t@@ -1,2 +1,2 @@\n\
                         5\t-old\n\
                         6\t+new\n";
    assert_eq!(router.detect(numbered_diff, &plain), ContentType::GitDiff);
}

#[test]
fn a_named_source_file_wins_over_content_sniffing() {
    let router = RuleContentRouter::new();
    let mut source = ctx(10_000);
    source.file_path = Some("src/App.tsx".to_owned());
    let component = "<div className=\"row\"><p>hello</p></div>";
    assert_eq!(router.detect(component, &source), ContentType::SourceFile);
    assert_eq!(router.detect(component, &ctx(10_000)), ContentType::Html);
    assert_eq!(ContentType::SourceFile.as_str(), "SOURCE_FILE");
}

#[test]
fn compress_reports_the_type_detection_chooses() {
    let router = RuleContentRouter::new();
    let plain = ctx(10_000);
    for content in [
        pytest_log(3),
        unified_diff(2, 4),
        search_output(2, 6),
        html_page(2),
        r#"[{"id": 1}, {"id": 2}]"#.to_owned(),
        repeated_lines(80),
    ] {
        let detected = router.detect(&content, &plain);
        assert_eq!(router.compress(&content, &plain).content_type, detected);
    }
}

// ---------------------------------------------------------------------------
// Per-compressor gates
// ---------------------------------------------------------------------------

#[test]
fn compressors_decline_when_they_cannot_improve_the_input() {
    let roomy = ctx(100_000);
    assert!(
        PlainTextCompressor
            .compress(&repeated_lines(200), &roomy)
            .modified
    );
    assert!(
        !PlainTextCompressor
            .compress("alpha\nbeta\ngamma\ndelta", &roomy)
            .modified
    );

    assert!(
        HtmlCompressor::default()
            .compress(&html_page(6), &roomy)
            .modified
    );
    let tiny = HtmlCompressor::default().compress("<p>tiny</p>", &roomy);
    assert!(!tiny.modified);
    assert_eq!(tiny.content, "<p>tiny</p>");

    let rows: Vec<serde_json::Value> = (0..40)
        .map(|index| serde_json::json!({"id": index, "name": format!("row {index}"), "ok": true}))
        .collect();
    let array = serde_json::Value::Array(rows).to_string();
    let compressed = JsonArrayCompressor.compress(&array, &roomy);
    assert!(compressed.modified);
    assert!(!compressed.lossy);
    // Already-compact JSON cannot be improved on.
    assert!(
        !JsonArrayCompressor
            .compress(r#"[{"a":1},{"a":2}]"#, &roomy)
            .modified
    );

    let log = pytest_log(30);
    let compressed_log = LogCompressor.compress(&log, &roomy);
    assert!(compressed_log.modified);
    assert!(compressed_log
        .content
        .starts_with("[LOG compressed: format=pytest"));
    assert!(
        !LogCompressor
            .compress("line one\nline two\n", &roomy)
            .modified
    );

    let diff = unified_diff(3, 40);
    let compressed_diff = DiffCompressor.compress(&diff, &roomy);
    assert!(compressed_diff.modified);
    assert!(compressed_diff.content.contains("diff --git a/src/f0.rs"));
    assert!(
        !DiffCompressor
            .compress("diff --git a/x b/x\n", &roomy)
            .modified
    );

    let search = search_output(3, 60);
    let compressed_search = SearchResultsCompressor.compress(&search, &roomy);
    assert!(compressed_search.modified);
    assert!(compressed_search
        .content
        .starts_with("[SEARCH_RESULTS compressed]"));
    assert!(
        !SearchResultsCompressor
            .compress("one\ntwo\nthree", &roomy)
            .modified
    );

    let mut source_ctx = ctx(100_000);
    source_ctx.file_path = Some("src/report.rs".to_owned());
    let mut doc_comments = String::new();
    for index in 0..40 {
        doc_comments.push_str(&format!("/// doc comment {index}\n"));
    }
    let source = format!(
        "// generated file, do not edit\n{doc_comments}\nfn report() {{\n    println!(\"report\");\n}}\n"
    );
    let compressed_source = SourceFileCompressor.compress(&source, &source_ctx);
    assert!(compressed_source.modified);
    assert!(compressed_source.content.contains("fn report()"));
    assert!(
        !SourceFileCompressor
            .compress("plain text\nlines\n", &ctx(100_000))
            .modified
    );
}

#[test]
fn token_accounting_and_savings_match_the_python_formula() {
    let plain = ctx(100);
    assert_eq!(count_tokens("", &plain), 1);
    assert_eq!(count_tokens(&"a".repeat(30), &plain), 10);
    assert!(meets_savings_ratio(
        &"a".repeat(300),
        &"a".repeat(30),
        &plain
    ));
    assert!(!meets_savings_ratio(
        &"a".repeat(300),
        &"a".repeat(295),
        &plain
    ));
}

#[test]
fn diff_details_report_the_original_offload_decision() {
    let roomy = ctx(100_000);
    let trimmed = DiffCompressor.compress(&unified_diff(3, 40), &roomy);
    let details = trimmed
        .details
        .expect("details for a rejected or applied diff");
    assert_eq!(details["files_affected"], 3);
    assert_eq!(details["should_offload_original"], serde_json::json!(true));
    assert!(details["changed_lines_omitted"].as_u64().unwrap() > 0);

    // Twelve small files: nothing is omitted, so the diff is kept whole.
    let untrimmed = DiffCompressor.compress(&unified_diff(12, 4), &roomy);
    assert!(!untrimmed.modified);
    let details = untrimmed
        .details
        .expect("details are reported for declined diffs");
    assert_eq!(details["should_offload_original"], serde_json::json!(false));
    assert_eq!(details["changed_lines_omitted"], serde_json::json!(0));

    // A diff shorter than `diff_min_lines` exits before parsing: no details.
    let short = DiffCompressor.compress("diff --git a/x b/x\n@@ -1 +1 @@\n-a\n+b", &roomy);
    assert!(!short.modified);
    assert!(short.details.is_none());
}

#[test]
fn log_details_count_the_classified_levels() {
    let roomy = ctx(100_000);
    let result = LogCompressor.compress(&pytest_log(30), &roomy);
    let details = result.details.expect("details");
    assert_eq!(details["format_detected"], "pytest");
    assert!(details["errors"].as_u64().unwrap() >= 30);
    assert_eq!(details["total_lines"], 63);
    assert!(details["omitted_lines"].as_u64().unwrap() > 0);
    assert!(result.content.contains("12 failed, 108 passed"));
}

#[test]
fn html_details_describe_the_extraction() {
    let roomy = ctx(100_000);
    let result = HtmlCompressor::default().compress(&html_page(6), &roomy);
    assert!(result.modified);
    assert!(!result.content.contains("var tracking"));
    let details = result.details.expect("details");
    assert_eq!(details["extractor"], "builtin");
    assert_eq!(details["title"], "Payment report");
    assert!(details["removed_node_count"].as_u64().unwrap() >= 3);
    assert!(details["reduction_percent"].as_f64().unwrap() > 0.0);
}

// ---------------------------------------------------------------------------
// Pipeline: stamping, force, dumps
// ---------------------------------------------------------------------------

/// A conversation whose tool result is `output`, requested by `tool`.
fn conversation(user: &str, tool: &str, output: &str) -> Vec<BaseMessage> {
    vec![
        BaseMessage::user(user),
        BaseMessage::assistant("").with_tool_calls(vec![ToolCall::new(tool)
            .with_id("call_1")
            .with_arguments(serde_json::json!({"pattern": "duplicate"}))]),
        BaseMessage::tool("call_1", output),
    ]
}

#[test]
fn a_compressed_message_carries_the_rule_compression_metadata() {
    let pipeline = RuleCompressionPipeline::with_clock(|| 1_700_000_000.5);
    let messages = conversation(
        "find the duplicate handlers",
        "grep",
        &search_output(3, 400),
    );
    let compressed = pipeline
        .compress_in_messages(&messages, 2, ADD_PASS, 2_000, false)
        .expect("infallible compression");
    assert!(is_rule_compressed(&compressed));
    assert_eq!(
        compressed.metadata[RULE_COMPRESSION_TYPE_KEY],
        "SEARCH_RESULTS"
    );
    assert_eq!(compressed.metadata[RULE_COMPRESSION_PASS_KEY], ADD_PASS);
    assert_eq!(compressed.metadata[RULE_COMPRESSED_AT_KEY], 1_700_000_000.5);
    let details = compressed.metadata[RULE_COMPRESSION_DETAILS_KEY]
        .as_object()
        .expect("details object");
    assert_eq!(details["files_affected"], 3);
    assert!(details["retained_match_count"].as_u64().unwrap() > 0);
    // Identity and pairing survive the rewrite.
    assert_eq!(compressed.role, Role::Tool);
    assert_eq!(compressed.tool_call_id.as_deref(), Some("call_1"));
    assert!(compressed.content.as_text().unwrap().len() < 10_000);
}

#[test]
fn messages_below_the_budget_are_left_alone_unless_forced() {
    let pipeline = RuleCompressionPipeline::new();
    let messages = conversation("look at the log", "shell", &pytest_log(30));
    let untouched = pipeline
        .compress_in_messages(&messages, 2, ADD_PASS, 1_000_000, false)
        .expect("ok");
    assert!(!is_rule_compressed(&untouched));
    assert_eq!(untouched.content.as_text(), messages[2].content.as_text());

    let forced = pipeline
        .compress_in_messages(&messages, 2, TTL_PASS, 1_000_000, true)
        .expect("ok");
    assert!(is_rule_compressed(&forced));
    assert_eq!(forced.metadata[RULE_COMPRESSION_PASS_KEY], TTL_PASS);
    assert_eq!(forced.metadata[RULE_COMPRESSION_TYPE_KEY], "LOG");

    // Non-tool messages are never touched, even with `force`.
    let user = BaseMessage::user("x".repeat(50_000));
    let skipped = pipeline
        .compress(&user, &ctx(10), TTL_PASS, 10, true)
        .expect("ok");
    assert!(!is_rule_compressed(&skipped));
}

#[test]
fn dumps_are_written_only_when_enabled() {
    let dir = tempfile::tempdir().expect("temp dir");
    let messages = conversation("look at the log", "shell", &pytest_log(30));
    let dumping = RuleCompressionPipeline::new()
        .with_dump(true, Some(dir.path().to_string_lossy().into_owned()));
    let compressed = dumping
        .compress_in_messages(&messages, 2, ADD_PASS, 1_000, false)
        .expect("ok");
    let dump_path = compressed.metadata[RULE_COMPRESSION_DUMP_PATH_KEY]
        .as_str()
        .expect("dump path")
        .to_owned();
    assert!(std::path::Path::new(&dump_path)
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("json")));
    let payload: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&dump_path).expect("dump file"))
            .expect("json");
    assert_eq!(payload["rule_compression_type"], "LOG");
    assert_eq!(payload["rule_compression_pass"], ADD_PASS);
    assert_eq!(payload["tool_call_id"], "call_1");
    assert!(
        payload["original_chars"].as_u64().unwrap() > payload["compressed_chars"].as_u64().unwrap()
    );
    assert!(payload["compressed_content"]
        .as_str()
        .unwrap()
        .starts_with("[LOG compressed"));

    let quiet = RuleCompressionPipeline::new()
        .with_dump(false, Some(dir.path().to_string_lossy().into_owned()));
    let plain = quiet
        .compress_in_messages(&messages, 2, ADD_PASS, 1_000, false)
        .expect("ok");
    assert!(!plain.metadata.contains_key(RULE_COMPRESSION_DUMP_PATH_KEY));
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
}

// ---------------------------------------------------------------------------
// Query terms
// ---------------------------------------------------------------------------

#[test]
fn query_terms_come_from_the_request_and_the_tool_call() {
    let terms = extract_query_terms(
        "Investigate the timeout in payment.rs",
        Some("grep_files"),
        Some(r#"{"file_path": "/abs/payment_router.rs", "limit": 2000, "query": "retry timeout"}"#),
    );
    assert!(terms.contains("investigate"));
    assert!(terms.contains("timeout"));
    assert!(terms.contains("payment"));
    assert!(terms.contains("grep_files"));
    assert!(terms.contains("retry"));
    // Path values and non-user-input values contribute nothing.
    assert!(!terms.contains("payment_router"));
    assert!(!terms.contains("2000"));
    // Two-character terms never qualify.
    assert!(!terms.contains("in"));
}

#[test]
fn cjk_text_contributes_only_its_ascii_terms() {
    let terms = extract_query_terms("修复支付超时 bug", None, None);
    assert_eq!(
        terms.iter().cloned().collect::<Vec<_>>(),
        vec!["bug".to_owned()]
    );
}

// ---------------------------------------------------------------------------
// End to end through MessageSummaryOffloader
// ---------------------------------------------------------------------------

fn summary_offloader_config(max_chars: usize) -> MessageSummaryOffloaderConfig {
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
    }
}

#[tokio::test]
async fn a_rule_compressed_result_that_fits_is_kept_without_offloading() {
    let context_config = ContextConfig {
        token_counter: TokenCounterConfig {
            kind: TokenCounterKind::Characters,
            chars_per_token: 1,
        },
        ..ContextConfig::default()
    };
    let counter = ApproxTokenCounter::with_chars_per_token(1);
    let store = InMemoryOffloadStore::new();
    let scope = OffloadScope::new("session-rule", "main");
    let cancellation = CancellationToken::new();
    let processor = MessageSummaryOffloader::new(
        summary_offloader_config(2_000),
        None,
        Arc::new(InMemoryOffloadStore::new()) as Arc<dyn OffloadStore>,
    );
    let mut messages = conversation(
        "find the duplicate handlers",
        "grep",
        &search_output(3, 400),
    );
    let event = processor
        .process(
            &mut messages,
            &processor_context(&context_config, &counter, &store, &scope, &cancellation),
        )
        .await
        .expect("process")
        .expect("an event");
    assert_eq!(event.messages_changed, vec![2]);
    let text = messages[2].content.as_text().expect("text");
    assert!(text.starts_with("[SEARCH_RESULTS compressed]"));
    assert!(!text.contains(OFFLOADED_INFO_TAG));
    assert!(!text.contains("[[OFFLOAD:"));
    assert_eq!(
        messages[2].metadata[RULE_COMPRESSION_TYPE_KEY],
        "SEARCH_RESULTS"
    );
    assert_eq!(messages[2].metadata[RULE_COMPRESSION_PASS_KEY], ADD_PASS);
    assert_eq!(processor.state.counter("rule_compressed"), 1);
    assert_eq!(processor.state.counter("offloaded"), 0);
}

#[tokio::test]
async fn a_still_oversized_remainder_is_offloaded_with_a_marker() {
    let context_config = ContextConfig {
        token_counter: TokenCounterConfig {
            kind: TokenCounterKind::Characters,
            chars_per_token: 1,
        },
        ..ContextConfig::default()
    };
    let counter = ApproxTokenCounter::with_chars_per_token(1);
    let store = InMemoryOffloadStore::new();
    let scope = OffloadScope::new("session-rule", "main");
    let cancellation = CancellationToken::new();
    let processor = MessageSummaryOffloader::new(
        summary_offloader_config(500),
        None,
        Arc::new(InMemoryOffloadStore::new()) as Arc<dyn OffloadStore>,
    );
    let mut messages = conversation("why does the capture fail", "shell", &pytest_log(200));
    let event = processor
        .process(
            &mut messages,
            &processor_context(&context_config, &counter, &store, &scope, &cancellation),
        )
        .await
        .expect("process")
        .expect("an event");
    assert_eq!(event.messages_changed, vec![2]);
    let text = messages[2].content.as_text().expect("text");
    assert!(text.contains(OFFLOADED_INFO_TAG));
    let marker = OffloadMarker::parse(text).expect("offload marker");
    assert_eq!(event.offload_handle, Some(marker.handle.clone()));
    // Tier 1 ran first and stamped the message.
    assert_eq!(messages[2].metadata[RULE_COMPRESSION_TYPE_KEY], "LOG");
    assert_eq!(processor.state.counter("rule_compressed"), 1);
    assert_eq!(processor.state.counter("offloaded"), 1);
    // The stored payload is the rule-compressed body, not the raw log.
    let stored = processor
        .store
        .get(&scope, &marker.handle)
        .await
        .expect("store lookup")
        .expect("stored message");
    let stored_text = stored.messages[0].content.as_text().expect("stored text");
    assert!(stored_text.starts_with("[LOG compressed"));
    assert!(stored_text.len() > 500);
}

#[tokio::test]
async fn rule_compression_can_be_switched_off() {
    let context_config = ContextConfig {
        token_counter: TokenCounterConfig {
            kind: TokenCounterKind::Characters,
            chars_per_token: 1,
        },
        ..ContextConfig::default()
    };
    let counter = ApproxTokenCounter::with_chars_per_token(1);
    let store = InMemoryOffloadStore::new();
    let scope = OffloadScope::new("session-rule", "main");
    let cancellation = CancellationToken::new();
    let processor = MessageSummaryOffloader::new(
        MessageSummaryOffloaderConfig {
            enable_rule_compression: false,
            ..summary_offloader_config(2_000)
        },
        None,
        Arc::new(InMemoryOffloadStore::new()) as Arc<dyn OffloadStore>,
    );
    let mut messages = conversation(
        "find the duplicate handlers",
        "grep",
        &search_output(3, 400),
    );
    processor
        .process(
            &mut messages,
            &processor_context(&context_config, &counter, &store, &scope, &cancellation),
        )
        .await
        .expect("process")
        .expect("an event");
    assert!(!messages[2].metadata.contains_key(RULE_COMPRESSED_AT_KEY));
    assert!(messages[2]
        .content
        .as_text()
        .expect("text")
        .contains(OFFLOADED_INFO_TAG));
}
