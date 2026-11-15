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
use openjiuwen_context::processor::rule_compression::router::{
    is_timestamp_line, repair_json, repair_json_array, repair_json_array_checked, sniff_bytes,
    sniff_content, JsonRepairFailure, SniffKind,
};
use openjiuwen_context::processor::rule_compression::{
    ContentType, RuleContentRouter, RuleContext,
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
