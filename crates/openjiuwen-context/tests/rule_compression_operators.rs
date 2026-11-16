#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
// Fixtures append `format!` output to a growing `String` on purpose.
#![allow(clippy::format_push_string, clippy::uninlined_format_args)]
//! Public-API tests for the rule-compression content operators.
//!
//! Three operator families are covered through the crate's public surface:
//!
//! - the tolerant JSON repair ([`repair_json`], [`repair_json_array`],
//!   [`repair_json_array_checked`]) with one case per repair rule plus a body
//!   that no candidate can repair,
//! - the content sniff ([`sniff_content`], [`sniff_bytes`]) for UTF-16, NUL and
//!   replacement-heavy payloads, and the Unicode-aware [`is_timestamp_line`],
//! - the HTML readability ladder as the router exposes it: a page whose article
//!   must survive while the menu, the related-links sidebar and the footer are
//!   dropped.
//!
//! Everything is offline and deterministic: no model, no clock, no filesystem
//! and no network.
// Everything is reached through the module root, so the re-exports a caller
// depends on are the imports this suite exercises.
use openjiuwen_context::processor::rule_compression::{
    is_timestamp_line, repair_json, repair_json_array, repair_json_array_checked, sniff_bytes,
    sniff_content, ContentType, JsonRepairFailure, RuleContentRouter, RuleContext, SniffKind,
    BINARY_SNIFF_BYTES, MAX_CONTROL_RATIO_PERCENT, REPLACEMENT_CHAR_MIN,
    REPLACEMENT_CHAR_RATIO_PERCENT,
};

/// A rule-compression context with room for every fixture.
fn ctx() -> RuleContext {
    RuleContext::new(20_000)
}

/// The router under test; it is stateless.
fn router() -> RuleContentRouter {
    RuleContentRouter::new()
}

/// A repaired JSON value, as it prints.
fn array_text(value: &serde_json::Value) -> String {
    value.to_string()
}

// ---------------------------------------------------------------------------
// JSON repair: one case per rule
// ---------------------------------------------------------------------------

#[test]
fn a_fenced_array_is_repaired() {
    let value = repair_json_array("```json\n[{\"a\": 1}]\n```").expect("fenced array");
    assert_eq!(array_text(&value), r#"[{"a":1}]"#);

    let bare_fence = repair_json_array("```[1, 2]```").expect("fence without a language");
    assert_eq!(array_text(&bare_fence), "[1,2]");
}

#[test]
fn prose_around_the_json_is_dropped() {
    let value = repair_json_array("Here are the rows: [{\"a\": 1}] — done, thanks").expect("prose");
    assert_eq!(array_text(&value), r#"[{"a":1}]"#);
}

#[test]
fn trailing_commas_are_removed_from_arrays_and_objects() {
    let value = repair_json_array("[{\"a\": 1,}, {\"b\": 2,},]").expect("trailing commas");
    assert_eq!(value.as_array().map(Vec::len), Some(2));

    let object = repair_json("{\"a\": [1, 2,],}").expect("nested trailing commas");
    assert_eq!(array_text(&object), r#"{"a":[1,2]}"#);

    // A comma inside a string is data, not syntax.
    let data = repair_json_array("[\"a,]\"]").expect("string payload");
    assert_eq!(array_text(&data), r#"["a,]"]"#);
}

#[test]
fn single_quoted_strings_are_rewritten() {
    let value = repair_json_array("[{'note': 'stop'}]").expect("single quotes");
    assert_eq!(array_text(&value), r#"[{"note":"stop"}]"#);

    // An escaped quote inside the literal is unescaped, not a string end.
    let escaped = repair_json_array(r"[{'note': 'don\'t stop'}]").expect("escaped quote");
    assert_eq!(
        escaped.as_array().and_then(|items| items[0].get("note")),
        Some(&serde_json::Value::String("don't stop".to_owned()))
    );

    // A single-quoted Python literal keeps its spelling; it is a string.
    let literal = repair_json_array("[{'flag': 'True'}]").expect("quoted literal");
    assert_eq!(array_text(&literal), r#"[{"flag":"True"}]"#);
}

#[test]
fn unquoted_keys_are_quoted() {
    let value = repair_json_array("[{a: 1, b_c: 2}]").expect("unquoted keys");
    assert_eq!(array_text(&value), r#"[{"a":1,"b_c":2}]"#);

    // A bare value is not a key, so it is left alone.
    let bare = repair_json("{\"a\": wide}").expect_err("a bare word is not JSON");
    assert_eq!(bare, JsonRepairFailure::Unparsable);
}

#[test]
fn python_literals_are_rewritten() {
    let value = repair_json_array("[True, False, None]").expect("python literals");
    assert_eq!(array_text(&value), "[true,false,null]");

    // A literal-looking word inside a string is data.
    let data = repair_json_array("[\"True story\"]").expect("string payload");
    assert_eq!(array_text(&data), r#"["True story"]"#);
}

#[test]
fn smart_quotes_are_folded() {
    let value =
        repair_json_array("[{\u{201c}a\u{201d}: \u{2018}b\u{2019}}]").expect("smart quotes");
    assert_eq!(array_text(&value), r#"[{"a":"b"}]"#);

    // Smart quotes inside a real JSON string stay as the author wrote them.
    let data = repair_json_array("[\"he said \u{201c}hi\u{201d}\"]").expect("string payload");
    assert_eq!(array_text(&data), "[\"he said \u{201c}hi\u{201d}\"]");
}

#[test]
fn a_truncated_final_element_is_dropped() {
    let value = repair_json_array("[{\"a\": 1}, {\"b\":").expect("truncated tail");
    assert_eq!(array_text(&value), r#"[{"a":1}]"#);
}

#[test]
fn a_truncated_array_is_closed() {
    let value = repair_json_array("[\"abc").expect("truncated string");
    assert_eq!(array_text(&value), "[\"abc\"]");

    let nested = repair_json_array("[{\"a\": 1").expect("truncated object");
    assert_eq!(array_text(&nested), r#"[{"a":1}]"#);
}

#[test]
fn a_bare_object_body_is_wrapped() {
    let value = repair_json("\"a\": 1, \"b\": 2").expect("bare members");
    assert_eq!(array_text(&value), r#"{"a":1,"b":2}"#);
}

#[test]
fn a_json_lines_body_is_wrapped() {
    let value = repair_json_array("{\"a\": 1}\n{\"b\": 2}").expect("json lines");
    assert_eq!(array_text(&value), r#"[{"a":1},{"b":2}]"#);

    let comma_separated = repair_json_array("{\"a\": 1}, {\"b\": 2}").expect("comma separated");
    assert_eq!(comma_separated.as_array().map(Vec::len), Some(2));
}

#[test]
fn an_unrepairable_body_is_reported() {
    // A value that is not JSON, inside brackets that balance.
    assert_eq!(
        repair_json_array_checked("[{\"a\": oops}]"),
        Err(JsonRepairFailure::Unparsable)
    );
    // Nothing to anchor on, and no bare members either.
    assert_eq!(
        repair_json_array_checked("just prose about cats"),
        Err(JsonRepairFailure::NoJsonBody)
    );
    // A repairable document that is an object, not an array.
    assert_eq!(
        repair_json_array_checked("{\"a\": 1}"),
        Err(JsonRepairFailure::NotAnArray)
    );
    // The `Option` entry point reports the same body as `None`.
    assert_eq!(repair_json_array("[{\"a\": oops}]"), None);
    assert_eq!(JsonRepairFailure::Unparsable.as_str(), "unparsable");
    assert_eq!(JsonRepairFailure::NotAnArray.to_string(), "not_an_array");
}

#[test]
fn a_repaired_array_is_still_detected_and_compressed() {
    let mut body = String::from("[\n");
    for index in 0..40 {
        body.push_str(&format!(
            "  {{'id': {}, 'status': 'ok', 'detail': 'the payment capture for merchant {} \
             settled after retry {}'}},\n",
            index, index, index
        ));
    }
    body.push_str("] and that is the whole report");
    let result = router().compress(&body, &ctx());
    assert_eq!(result.content_type, ContentType::JsonArray);
    assert!(result.modified, "the repaired array compresses");
}

// ---------------------------------------------------------------------------
// Sniffing: UTF-16, NUL bytes, replacement characters
// ---------------------------------------------------------------------------

/// An HTML page of ASCII text, so it survives a UTF-16 round trip byte for byte.
fn article_page() -> String {
    String::from(
        "<!doctype html><html><head><title>Retry policy</title></head><body>\
         <div class=\"menu\"><a href=\"/home\">Home</a> <a href=\"/docs\">Docs</a> \
         <a href=\"/pricing\">Pricing</a> <a href=\"/blog\">Blog</a></div>\
         <article><h1>Retry policy</h1>\
         <p>A failed capture is retried three times before the pipeline reports the error \
         to the caller.</p>\
         <p>Every retry is logged with the request id, so an operator can follow one payment \
         through the gateway.</p>\
         <p>The retry budget is configurable per merchant, so a slow gateway never exhausts \
         the default allowance.</p>\
         </article>\
         <div class=\"sidebar\"><a href=\"/a\">Related: gateway limits</a> \
         <a href=\"/b\">Related: refunds</a> <a href=\"/c\">Related: settlements</a></div>\
         <div class=\"footer-notes\">&copy; 2026 Example Inc. All rights reserved. \
         Privacy policy &middot; Terms of service</div>\
         </body></html>",
    )
}

/// `text` as a UTF-16 byte string, with the matching byte-order mark.
fn utf16_bytes(text: &str, little_endian: bool) -> Vec<u8> {
    let mut bytes = if little_endian {
        vec![0xFF, 0xFE]
    } else {
        vec![0xFE, 0xFF]
    };
    for unit in text.encode_utf16() {
        let low = u8::try_from(unit).expect("the fixture is ASCII");
        if little_endian {
            bytes.extend([low, 0]);
        } else {
            bytes.extend([0, low]);
        }
    }
    bytes
}

#[test]
fn a_utf16le_body_is_recoded() {
    let page = article_page();
    let sniffed = sniff_bytes(&utf16_bytes(&page, true));
    assert_eq!(sniffed.kind, SniffKind::Utf16Le);
    assert!(sniffed.converted, "the body was re-encoded");
    assert_eq!(sniffed.text, page);
    assert_eq!(sniffed.source_encoding(), Some("utf-16le"));
    assert!(!sniffed.is_passthrough());

    let details = sniffed.details().expect("a non-UTF-8 body reports details");
    assert_eq!(details["content_kind"], "utf16_le");
    assert_eq!(details["source_encoding"], "utf-16le");
    assert_eq!(details["recoded"], true);
}

#[test]
fn a_utf16be_body_is_recoded() {
    let page = article_page();
    let sniffed = sniff_bytes(&utf16_bytes(&page, false));
    assert_eq!(sniffed.kind, SniffKind::Utf16Be);
    assert_eq!(sniffed.text, page);
    assert_eq!(sniffed.source_encoding(), Some("utf-16be"));
}

#[test]
fn a_latin1_projection_of_utf16_is_recovered() {
    let page = article_page();
    let projected: String = utf16_bytes(&page, true)
        .iter()
        .map(|byte| char::from(*byte))
        .collect();
    let sniffed = sniff_content(&projected);
    assert_eq!(sniffed.kind, SniffKind::Utf16Le);
    assert_eq!(sniffed.text, page);
}

#[test]
fn a_utf16_body_is_compressed_as_the_text_it_holds() {
    let page = article_page();
    let projected: String = utf16_bytes(&page, true)
        .iter()
        .map(|byte| char::from(*byte))
        .collect();
    let result = router().compress(&projected, &ctx());
    assert_eq!(result.content_type, ContentType::Html);
    assert!(result.modified, "the re-encoded page compresses");
    assert!(result
        .content
        .contains("A failed capture is retried three times"));
    let details = result.details.expect("the compressor reports details");
    assert_eq!(details["source_encoding"], "utf-16le");
    assert_eq!(details["content_kind"], "utf16_le");
}

#[test]
fn a_utf8_bom_is_stripped() {
    let sniffed = sniff_content("\u{feff}[1, 2]");
    assert_eq!(sniffed.kind, SniffKind::Utf8Bom);
    assert_eq!(sniffed.text, "[1, 2]");
    assert!(!sniffed.is_passthrough());
}

#[test]
fn a_nul_payload_is_passed_through() {
    let binary = "GIF89a\u{0}\u{1}\u{0}payload\u{0}more bytes";
    let sniffed = sniff_content(binary);
    assert_eq!(sniffed.kind, SniffKind::Binary);
    assert!(sniffed.is_passthrough());
    assert!(sniffed.nul_bytes >= 3);

    let result = router().compress(binary, &ctx());
    assert_eq!(result.content, binary, "a binary body is left untouched");
    assert!(!result.modified);
    assert_eq!(result.content_type, ContentType::PlainText);
    let details = result.details.expect("the pass-through is reported");
    assert_eq!(details["content_kind"], "binary");
    assert_eq!(details["nul_bytes"], 3);
}

#[test]
fn a_control_heavy_payload_is_passed_through() {
    let controls = format!("{}\u{0}tail", "\u{1}".repeat(40));
    let sniffed = sniff_content(&controls);
    assert_eq!(sniffed.kind, SniffKind::Binary);
    assert!(sniffed.control_ratio > 5.0);

    let result = router().compress(&controls, &ctx());
    assert!(!result.modified);
    assert_eq!(result.content, controls);
}

#[test]
fn replacement_heavy_text_is_passed_through() {
    let mangled = format!(
        "{}broken text{}",
        "\u{fffd}".repeat(30),
        "\u{fffd}".repeat(30)
    );
    let sniffed = sniff_content(&mangled);
    assert_eq!(sniffed.kind, SniffKind::ReplacementHeavy);
    assert!(sniffed.is_passthrough());
    assert!(sniffed.replacement_characters >= 8);

    let result = router().compress(&mangled, &ctx());
    assert!(!result.modified);
    assert_eq!(result.content_type, ContentType::PlainText);
    let details = result.details.expect("the pass-through is reported");
    assert_eq!(details["content_kind"], "replacement_heavy");
}

#[test]
fn tabs_and_newlines_stay_text() {
    let sniffed = sniff_content("a\tb\nc\r\nd");
    assert_eq!(sniffed.kind, SniffKind::Utf8);
    assert_eq!(sniffed.text, "a\tb\nc\r\nd");
    assert!(sniffed.details().is_none(), "UTF-8 reports no details");
}

#[test]
fn the_sniff_thresholds_are_public_and_pinned() {
    // A caller budgets against these limits, so they are part of the surface.
    assert_eq!(BINARY_SNIFF_BYTES, 8 * 1024);
    assert_eq!(MAX_CONTROL_RATIO_PERCENT, 5);
    assert_eq!(REPLACEMENT_CHAR_MIN, 8);
    assert_eq!(REPLACEMENT_CHAR_RATIO_PERCENT, 5);

    // The control ratio is strict: exactly the limit is text, one over is binary.
    let controls = "\u{1}".repeat(MAX_CONTROL_RATIO_PERCENT);
    let at_limit = format!("{controls}{}", "a".repeat(100 - MAX_CONTROL_RATIO_PERCENT));
    assert_eq!(sniff_content(&at_limit).kind, SniffKind::Utf8);
    let over = format!(
        "{}{}",
        "\u{1}".repeat(MAX_CONTROL_RATIO_PERCENT + 1),
        "a".repeat(100 - MAX_CONTROL_RATIO_PERCENT - 1)
    );
    assert_eq!(sniff_content(&over).kind, SniffKind::Binary);

    // The replacement gate needs the count and the ratio, so the count alone
    // keeps a body as text.
    let few = format!(
        "{}{}",
        "\u{fffd}".repeat(REPLACEMENT_CHAR_MIN - 1),
        "a".repeat(100 - REPLACEMENT_CHAR_MIN + 1)
    );
    assert_eq!(sniff_content(&few).kind, SniffKind::Utf8);
    let many = format!(
        "{}{}",
        "\u{fffd}".repeat(REPLACEMENT_CHAR_MIN),
        "a".repeat(100 - REPLACEMENT_CHAR_MIN)
    );
    assert_eq!(sniff_content(&many).kind, SniffKind::ReplacementHeavy);

    // The binary scan stops at its limit, so a clean prefix hides a dirty tail.
    let mut long = "a".repeat(BINARY_SNIFF_BYTES);
    long.push_str(&"\u{1}".repeat(200));
    assert_eq!(sniff_content(&long).kind, SniffKind::Utf8);
}

// ---------------------------------------------------------------------------
// Unicode-aware timestamp recognition
// ---------------------------------------------------------------------------

#[test]
fn full_width_timestamps_are_recognised() {
    assert!(is_timestamp_line(
        "\u{ff12}\u{ff10}\u{ff12}\u{ff14}-\u{ff10}\u{ff11}-\u{ff10}\u{ff11}\
         \u{ff34}\u{ff10}\u{ff19}\u{ff1a}\u{ff13}\u{ff10}\u{ff1a}\u{ff10}\u{ff10} ERROR boom"
    ));
    // The ASCII shape is still a timestamp.
    assert!(is_timestamp_line("2024-01-01T09:30:00 ERROR boom"));
    assert!(is_timestamp_line("2024-01-01 09:30:00 WARN slow"));
}

#[test]
fn arabic_indic_timestamps_are_recognised() {
    // `٢٠٢٤-٠١-٠١T09:30:00`, with the date in Arabic-Indic digits.
    assert!(is_timestamp_line(
        "\u{0662}\u{0660}\u{0662}\u{0664}-\u{0660}\u{0661}-\u{0660}\u{0661}T09:30:00 ERROR boom"
    ));
    // A `path:line:` hit is not a timestamp.
    assert!(!is_timestamp_line("src/a.rs:12:fn one()"));
    // A short line is not a timestamp however it is spelled.
    assert!(!is_timestamp_line("2024-01-01"));
}

#[test]
fn a_localized_log_is_not_routed_as_search_results() {
    let mut log = String::new();
    for index in 0..20 {
        log.push_str(&format!(
            "\u{ff12}\u{ff10}\u{ff12}\u{ff14}-\u{ff10}\u{ff11}-\u{ff10}\u{ff11}\
             \u{ff34}\u{ff10}\u{ff19}\u{ff1a}\u{ff13}\u{ff10}\u{ff1a}\u{ff10}\u{ff10} \
             ERROR src/a.rs:{}:failed to flush\n",
            index
        ));
    }
    let result = router().compress(&log, &ctx());
    assert_ne!(result.content_type, ContentType::SearchResults);
}

// ---------------------------------------------------------------------------
// The HTML readability ladder
// ---------------------------------------------------------------------------

#[test]
fn the_readability_ladder_keeps_the_article_over_the_chrome() {
    let page = article_page();
    let result = router().compress(&page, &ctx());
    assert_eq!(result.content_type, ContentType::Html);
    assert!(result.modified, "the page compresses");

    let body = &result.content;
    for sentence in [
        "A failed capture is retried three times",
        "Every retry is logged with the request id",
        "The retry budget is configurable per merchant",
    ] {
        assert!(body.contains(sentence), "kept article sentence: {sentence}");
    }
    for chrome in [
        "Related: gateway limits",
        "Related: refunds",
        "All rights reserved",
        "Privacy policy",
        "[Home]",
    ] {
        assert!(!body.contains(chrome), "dropped chrome: {chrome}");
    }

    let details = result.details.expect("the compressor reports details");
    assert_eq!(details["extractor"], "builtin");
    assert_eq!(details["title"], "Retry policy");
    assert!(
        details["boilerplate_block_count"].as_u64().unwrap_or(0) >= 2,
        "the ladder reports the chrome it dropped: {details}"
    );
    assert!(
        details["selected_block_count"].as_u64().unwrap_or(0) >= 1,
        "the ladder reports the blocks it kept: {details}"
    );
    assert!(
        details["ladder_fallback"].is_null(),
        "the ladder kept a region instead of falling back: {details}"
    );
}

#[test]
fn a_page_of_pure_chrome_still_yields_text() {
    let page = format!(
        "<html><body><nav><a href=\"/a\">A</a> <a href=\"/b\">B</a></nav>{}</body></html>",
        "<p>Short.</p>".repeat(8)
    );
    let result = router().compress(&page, &ctx());
    assert_eq!(result.content_type, ContentType::Html);
    // Whether the ladder keeps the region or falls back, text survives.
    assert!(result.content.contains("Short."));
}

// ---------------------------------------------------------------------------
// Attribution: a decline, a pass-through or a skipped message names its reason
// ---------------------------------------------------------------------------
use openjiuwen_common::error::Error;
use openjiuwen_context::processor::rule_compression::{
    DiffCompressor, JsonArrayCompressor, LogCompressor, RuleCompressionPipeline,
    RuleCompressionResult, RuleCompressor, SearchResultsCompressor, SourceFileCompressor, ADD_PASS,
};
use openjiuwen_foundation::prompt::{BaseMessage, ContentPart, MessageContent};
use std::fmt;
use std::sync::{Arc, Mutex};
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id, Record};
use tracing::{Event, Level, Metadata, Subscriber};
/// A thread-local capture of every tracing event, message and field rendered.
#[derive(Clone, Default)]
struct EventCapture {
    /// The rendered events, in emission order.
    events: Arc<Mutex<Vec<String>>>,
}
impl EventCapture {
    /// An empty capture.
    fn new() -> Self {
        Self::default()
    }
    /// Every captured event whose rendering contains `needle`.
    fn matching(&self, needle: &str) -> Vec<String> {
        self.events
            .lock()
            .map(|events| {
                events
                    .iter()
                    .filter(|event| event.contains(needle))
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }
}
impl Subscriber for EventCapture {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        metadata.level() <= &Level::DEBUG
    }
    fn new_span(&self, _span: &Attributes<'_>) -> Id {
        Id::from_u64(1)
    }
    fn record(&self, _span: &Id, _values: &Record<'_>) {}
    fn record_follows_from(&self, _span: &Id, _follows: &Id) {}
    fn event(&self, event: &Event<'_>) {
        let mut visitor = EventVisitor(String::new());
        event.record(&mut visitor);
        if let Ok(mut events) = self.events.lock() {
            events.push(visitor.0);
        }
    }
    fn enter(&self, _span: &Id) {}
    fn exit(&self, _span: &Id) {}
}
/// Renders one event's message and fields as `key=value` pairs.
struct EventVisitor(String);
impl Visit for EventVisitor {
    fn record_str(&mut self, field: &Field, value: &str) {
        self.0.push_str(&format!("{}={} ", field.name(), value));
    }
    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        self.0.push_str(&format!("{}={:?} ", field.name(), value));
    }
}
#[test]
fn invalid_utf8_bytes_are_reported_as_decode_damage() {
    // `caf\xe9` is not UTF-8; the lossy decode substituted one `U+FFFD`, so the
    // body is damaged text and never clean UTF-8.
    let sniffed = sniff_bytes(b"caf\xe9 au lait");
    assert_eq!(sniffed.kind, SniffKind::ReplacementHeavy);
    assert_eq!(sniffed.decode_replacements, 1);
    assert!(sniffed.is_passthrough());
    let details = sniffed.details().expect("the decode damage is reported");
    assert_eq!(details["content_kind"], "replacement_heavy");
    assert_eq!(details["decode_replacement_characters"], 1);
    // A payload that encoded `U+FFFD` itself is data, not damage.
    let clean = sniff_bytes("a \u{fffd} byte".as_bytes());
    assert_eq!(clean.kind, SniffKind::Utf8);
    assert_eq!(clean.decode_replacements, 0);
    assert!(clean.details().is_none());
}
#[test]
fn a_truncated_utf16_payload_reports_its_dropped_byte_and_surrogate() {
    // A LE BOM, one unpaired high surrogate and a trailing half unit.
    let sniffed = sniff_bytes(b"\xFF\xFE\x00\xD8A");
    assert_eq!(sniffed.kind, SniffKind::Utf16Le);
    assert_eq!(sniffed.dropped_bytes, 1);
    assert_eq!(sniffed.decode_replacements, 1);
    let details = sniffed.details().expect("details");
    assert_eq!(details["source_encoding"], "utf-16le");
    assert_eq!(details["dropped_bytes"], 1);
    assert_eq!(details["decode_replacement_characters"], 1);
}
#[test]
fn a_utf16_surrogate_rides_the_compression_details() {
    // The latin-1 projection of a UTF-16LE page with one unpaired surrogate.
    let mut bytes = utf16_bytes(&article_page(), true);
    bytes.extend([0x00, 0xD8]);
    let projected: String = bytes.iter().map(|byte| char::from(*byte)).collect();
    let result = router().compress(&projected, &ctx());
    assert!(result.modified, "the page still compresses");
    let details = result.details.expect("details");
    assert_eq!(details["source_encoding"], "utf-16le");
    assert_eq!(details["decode_replacement_characters"], 1);
    assert_eq!(details["dropped_bytes"], 0);
}
#[test]
fn a_latin1_projection_that_would_corrupt_a_character_is_refused() {
    let capture = EventCapture::new();
    // A character above `U+00FF` would have to become `?`.
    let above = "\u{ff}\u{fe}a\u{4e00}";
    let sniffed = tracing::subscriber::with_default(capture.clone(), || sniff_content(above));
    assert_eq!(sniffed.kind, SniffKind::Utf8);
    assert_eq!(sniffed.text, above);
    assert!(sniffed.details().is_none());
    assert_eq!(capture.matching("reason=character_above_latin1").len(), 1);
    // An odd byte count cannot be UTF-16 either.
    let odd = "\u{ff}\u{fe}h";
    let sniffed = tracing::subscriber::with_default(capture.clone(), || sniff_content(odd));
    assert_eq!(sniffed.kind, SniffKind::Utf8);
    assert_eq!(sniffed.text, odd);
    assert_eq!(capture.matching("reason=odd_byte_count").len(), 1);
}
#[test]
fn detecting_a_binary_payload_reports_the_evidence() {
    let capture = EventCapture::new();
    let binary = "GIF89a\u{0}nul";
    let content_type =
        tracing::subscriber::with_default(capture.clone(), || router().detect(binary, &ctx()));
    assert_eq!(content_type, ContentType::PlainText);
    let events = capture.matching("payload through untouched");
    assert_eq!(events.len(), 1, "{events:?}");
    assert!(events[0].contains("content_kind=binary"));
    assert!(events[0].contains("reason=binary_payload"));
    // Replacement-heavy text names the other reason.
    let mangled = "\u{fffd}".repeat(REPLACEMENT_CHAR_MIN);
    let capture = EventCapture::new();
    let content_type =
        tracing::subscriber::with_default(capture.clone(), || router().detect(&mangled, &ctx()));
    assert_eq!(content_type, ContentType::PlainText);
    let events = capture.matching("payload through untouched");
    assert_eq!(events.len(), 1, "{events:?}");
    assert!(events[0].contains("content_kind=replacement_heavy"));
    assert!(events[0].contains("reason=replacement_damaged"));
}
#[test]
fn an_out_of_range_index_is_a_typed_error() {
    let messages = vec![BaseMessage::user("hi"), BaseMessage::assistant("hello")];
    let error = RuleCompressionPipeline::new()
        .compress_in_messages(&messages, 9, ADD_PASS, 100, false)
        .expect_err("an index the conversation does not own");
    assert!(matches!(error, Error::Validation(_)), "{error}");
    assert!(error.to_string().contains("index 9"), "{error}");
    assert!(error.to_string().contains("2 messages"), "{error}");
}
#[test]
fn a_message_without_text_reports_the_reason_it_was_skipped() {
    let mut message = BaseMessage::tool("call_1", "");
    message.content = MessageContent::Parts(vec![ContentPart::Text {
        text: "x".repeat(400),
    }]);
    let messages = vec![message.clone()];
    let capture = EventCapture::new();
    let unchanged = tracing::subscriber::with_default(capture.clone(), || {
        RuleCompressionPipeline::new().compress_in_messages(&messages, 0, ADD_PASS, 10, true)
    })
    .expect("an owned index");
    assert!(unchanged.metadata.is_empty(), "nothing was stamped");
    assert!(unchanged.content.as_text().is_none(), "content untouched");
    let events = capture.matching("reason=non_text_content");
    assert_eq!(events.len(), 1, "{events:?}");
    assert!(events[0].contains("content_kind=parts"));
    assert!(events[0].contains("message_index=0"));
}
#[test]
fn an_empty_query_term_set_reports_which_case_it_was() {
    let pipeline = RuleCompressionPipeline::new();
    let capture = EventCapture::new();
    let none = tracing::subscriber::with_default(capture.clone(), || {
        pipeline.context_for(900, &[], None, None)
    });
    assert!(none.query_terms.is_empty());
    assert_eq!(capture.matching("reason=no_user_message").len(), 1);
    let empty = tracing::subscriber::with_default(capture.clone(), || {
        pipeline.context_for(900, &[BaseMessage::user("   ")], None, None)
    });
    assert!(empty.query_terms.is_empty());
    assert_eq!(capture.matching("reason=empty_user_message").len(), 1);
    let mut user = BaseMessage::user("");
    user.content = MessageContent::Parts(vec![ContentPart::Text {
        text: "payout".to_owned(),
    }]);
    let non_text = tracing::subscriber::with_default(capture.clone(), || {
        pipeline.context_for(900, &[user], None, None)
    });
    assert!(non_text.query_terms.is_empty());
    assert_eq!(capture.matching("reason=non_text_user_message").len(), 1);
}
#[test]
fn a_mid_pipeline_decline_is_logged_with_its_reason() {
    // A small array the router types as JSON but the ladder cannot shrink.
    let message = BaseMessage::tool("call_1", r#"[{"a":1},{"a":2}]"#);
    let messages = vec![message];
    let capture = EventCapture::new();
    let unchanged = tracing::subscriber::with_default(capture.clone(), || {
        RuleCompressionPipeline::new().compress_in_messages(&messages, 0, ADD_PASS, 300, true)
    })
    .expect("an owned index");
    assert!(unchanged.metadata.is_empty());
    let events = capture.matching("compressor declined the body");
    assert_eq!(events.len(), 1, "{events:?}");
    assert!(events[0].contains("content_type=JSON_ARRAY"));
    assert!(events[0].contains("reason=candidate_does_not_fit"));
    assert!(events[0].contains("message_index=0"));
}
#[test]
fn a_short_log_reports_the_reason_it_was_declined() {
    let result = LogCompressor.compress("line one\nline two", &ctx());
    assert!(!result.modified);
    let details = result.details.expect("the decline reports why");
    assert_eq!(details["reason"], "below_min_lines");
    assert_eq!(details["total_lines"], 2);
    assert_eq!(details["min_lines"], 50);
}
#[test]
fn a_truncated_stack_trace_reports_its_dropped_tail_lines() {
    let mut lines = vec!["Traceback (most recent call last):".to_owned()];
    for index in 0..40 {
        lines.push(format!(
            "  File \"mod{index}.py\", line {index}, in <module>"
        ));
    }
    lines.push("ERROR: the pipeline failed".to_owned());
    let mut log_ctx = ctx();
    log_ctx.log_min_lines = 10;
    log_ctx.log_stack_trace_max_lines = 5;
    let result = LogCompressor.compress(&lines.join("\n"), &log_ctx);
    assert!(result.modified);
    let details = result.details.expect("details");
    assert!(
        details["stack_traces_truncated"].as_u64().unwrap_or(0) > 0,
        "{details}"
    );
    assert!(
        details["stack_trace_truncated_lines"].as_u64().unwrap_or(0) > 0,
        "{details}"
    );
    assert!(
        details["stack_trace_dropped_lines"].as_u64().unwrap_or(0) > 0,
        "{details}"
    );
}
#[test]
fn every_json_array_decline_carries_a_reason() {
    let small = RuleContext::new(100);
    let unparsable = JsonArrayCompressor.compress("[{\"a\": oops}]", &small);
    assert!(!unparsable.modified);
    assert_eq!(
        unparsable.details.expect("details")["reason"],
        "unparsable_array"
    );
    let empty = JsonArrayCompressor.compress("[]", &small);
    assert!(!empty.modified);
    assert_eq!(empty.details.expect("details")["reason"], "empty_array");
    let rows = serde_json::Value::Array(vec![serde_json::json!({"a": 1}); 2]);
    let small_array = JsonArrayCompressor.compress(&array_text(&rows), &small);
    assert!(!small_array.modified);
    let details = small_array.details.expect("details");
    assert_eq!(details["reason"], "candidate_does_not_fit");
    assert_eq!(details["rows"], 2);
}
#[test]
fn an_unmeasurable_json_row_is_attributed() {
    let content =
        serde_json::Value::Array((0..40).map(serde_json::Value::from).collect()).to_string();
    // One token of budget: neither the compact nor the salient candidate fits.
    let result = JsonArrayCompressor.compress(&content, &RuleContext::new(1));
    assert!(!result.modified);
    let details = result.details.expect("details");
    assert_eq!(details["reason"], "candidate_does_not_fit");
    // Every row parses as a number, so the unscoreable-row count is reported as
    // zero instead of a row silently scored `0.0`.
    assert_eq!(details["unmeasurable_rows"], 0);
    assert_eq!(details["rows"], 40);
}
#[test]
fn search_output_without_parsable_hits_reports_the_reason() {
    let result = SearchResultsCompressor.compress("just prose\nno markers here", &ctx());
    assert!(!result.modified);
    let details = result.details.expect("the decline reports why");
    assert_eq!(details["reason"], "no_parsable_hits");
    assert_eq!(details["unparsed_line_count"], 2);
}
#[test]
fn a_diff_without_a_file_header_reports_why_it_was_declined() {
    let mut diff_ctx = ctx();
    diff_ctx.diff_min_lines = 1;
    let content = "@@ -1,2 +1,2 @@\n-a\n+b";
    let result = DiffCompressor.compress(content, &diff_ctx);
    assert!(!result.modified);
    assert_eq!(result.content, content);
    let details = result.details.expect("the decline reports why");
    assert_eq!(details["reason"], "no_files_parsed");
    assert_eq!(details["original_line_count"], 3);
}
#[test]
fn a_blank_source_file_reports_its_digest_stage() {
    let mut source_ctx = ctx();
    source_ctx.file_path = Some("src/blank.rs".to_owned());
    let result = SourceFileCompressor.compress("  \n\n", &source_ctx);
    assert!(!result.modified);
    let details = result.details.expect("the decline reports why");
    assert_eq!(details["digest_stage"], "none");
    assert_eq!(details["reason"], "blank_content");
}
#[test]
fn an_unchanged_result_can_carry_a_reason() {
    let bare = RuleCompressionResult::unchanged("body", ContentType::Log);
    assert!(bare.details.is_none());
    let attributed = bare.with_details(Some(serde_json::json!({"reason": "nothing_to_do"})));
    assert!(!attributed.modified);
    assert_eq!(attributed.content, "body");
    assert_eq!(
        attributed.details.expect("details")["reason"],
        "nothing_to_do"
    );
}
