//! Unit tests for the record helpers and the diff of rewrites.
use super::metrics::compact_number;
use super::*;
use serde_json::json;
use std::time::UNIX_EPOCH;

fn stamped(mut message: BaseMessage, id: &str) -> BaseMessage {
    message
        .metadata
        .insert(CONTEXT_MESSAGE_ID_KEY.to_owned(), json!(id));
    message
}

#[test]
fn timestamps_are_iso_8601_utc() {
    assert_eq!(format_timestamp(UNIX_EPOCH), "1970-01-01T00:00:00.000Z");
    let later = UNIX_EPOCH + std::time::Duration::from_millis(1_704_164_645_006);
    assert_eq!(format_timestamp(later), "2024-01-02T03:04:05.006Z");
    // A leap day, a year boundary, and a negative (pre-epoch) day.
    let leap = UNIX_EPOCH + std::time::Duration::from_secs(1_709_164_800);
    assert_eq!(format_timestamp(leap), "2024-02-29T00:00:00.000Z");
    assert_eq!(
        format_timestamp(UNIX_EPOCH + std::time::Duration::from_secs(1)),
        "1970-01-01T00:00:01.000Z"
    );
}

#[test]
fn compact_numbers_match_python() {
    assert_eq!(compact_number(0), "0");
    assert_eq!(compact_number(999), "999");
    assert_eq!(compact_number(1_000), "1k");
    assert_eq!(compact_number(12_345), "12.3k");
    assert_eq!(compact_number(1_500_000), "1.5m");
    assert_eq!(compact_number(2_000_000), "2m");
}

#[test]
fn summaries_follow_python() {
    let before = ContextCompressionMetric {
        messages: 4,
        tokens: 1_200,
        ..ContextCompressionMetric::default()
    };
    let after = ContextCompressionMetric {
        messages: 2,
        tokens: 300,
        ..ContextCompressionMetric::default()
    };
    let saved = saved_delta(&before, Some(&after)).unwrap();
    assert_eq!(saved.messages, 2);
    assert_eq!(saved.tokens, 900);
    assert!((saved.percent - 75.0).abs() < 1e-9);

    let started = build_summary(CompressionStateStatus::Started, &before, None, None, "", 0);
    assert_eq!(started, "Compressing 4 messages, ~1.2k tokens");

    let completed = build_summary(
        CompressionStateStatus::Completed,
        &before,
        Some(&after),
        Some(&saved),
        "",
        1,
    );
    assert_eq!(
        completed,
        "Compressed 4 -> 2 messages, ~1.2k -> ~300 tokens, \
         saved ~900 tokens (75.0%), modified 1 messages"
    );

    let noop = build_summary(
        CompressionStateStatus::Noop,
        &before,
        Some(&before),
        saved_delta(&before, Some(&before)).as_ref(),
        "",
        0,
    );
    assert_eq!(noop, "Context unchanged at ~1.2k tokens (saved 0.0%)");

    let skipped = build_summary(
        CompressionStateStatus::Skipped,
        &before,
        None,
        None,
        "below threshold",
        0,
    );
    assert_eq!(skipped, "Context processor skipped: below threshold");
}

#[test]
fn diffs_report_only_in_place_rewrites() {
    let unchanged = stamped(BaseMessage::user("same"), "a");
    let rewritten = stamped(BaseMessage::assistant("abcdef"), "b");
    let removed = stamped(BaseMessage::user("gone"), "c");
    let before = vec![unchanged.clone(), rewritten.clone(), removed];

    let rewritten_after = stamped(BaseMessage::assistant("abcXef"), "b");
    let added = stamped(BaseMessage::user("new"), "d");
    let after = vec![unchanged, rewritten_after, added];

    let modified = modified_messages(&before, &after);
    assert_eq!(modified.len(), 1);
    assert_eq!(modified[0].message_id, "b");
    assert_eq!(modified[0].role, "assistant");
    assert!(modified[0].offload_handle.is_none());

    let offloaded = stamped(
        BaseMessage::tool("call_1", "[[OFFLOAD: handle=h1, type=in_memory]]"),
        "t",
    );
    let original = stamped(BaseMessage::tool("call_1", "long output"), "t");
    let modified = modified_messages(std::slice::from_ref(&original), &[offloaded]);
    assert_eq!(modified.len(), 1);
    assert_eq!(modified[0].message_id, "t");
    assert_eq!(modified[0].role, "tool");
    assert_eq!(modified[0].tool_call_id.as_deref(), Some("call_1"));
    assert_eq!(modified[0].offload_handle.as_deref(), Some("h1"));
    assert_eq!(modified[0].offload_type.as_deref(), Some("in_memory"));

    // An identical offload marker is not a change, and messages that only
    // appear on one side are never reported as rewrites.
    let same_marker = stamped(
        BaseMessage::tool("call_1", "[[OFFLOAD: handle=h1, type=in_memory]]"),
        "t",
    );
    let original = stamped(BaseMessage::tool("call_1", "long output"), "t");
    assert_eq!(
        modified_messages(
            std::slice::from_ref(&same_marker),
            std::slice::from_ref(&same_marker)
        ),
        Vec::new()
    );
    assert_eq!(modified_messages(&[original], &[]), Vec::new());
}
