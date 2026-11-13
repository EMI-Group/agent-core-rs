//! Cache-breakpoint tests for the projected request body.
//!
//! Covers the three anchors (tool list, system block array, last stable
//! message), the transient-tail skip, the plain-string anchor conversion, and
//! the `disable_cache_control` / `cache_control_ttl` overrides in `extra`.

use super::tests::{long_marker, marker, messages, request, system_blocks, uncached_request};
use super::*;
use crate::prompt::ToolCall;
use serde_json::json;

#[test]
fn a_tool_use_tail_gets_the_cache_marker() {
    let body = build_messages_body(
        &[
            BaseMessage::assistant("checking").with_tool_calls(vec![ToolCall::new("get_weather")
                .with_id("call_1")
                .with_arguments(json!({"city": "Paris"}))]),
        ],
        &request("claude-sonnet-4"),
        false,
    )
    .expect("body");

    assert_eq!(
        messages(&body),
        [json!({
            "role": "assistant",
            "content": [
                {"type": "text", "text": "checking"},
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "get_weather",
                    "input": {"city": "Paris"},
                    "cache_control": marker(),
                },
            ],
        })]
    );
}

#[test]
fn a_tool_result_tail_gets_the_cache_marker() {
    let body = build_messages_body(
        &[
            BaseMessage::user("go"),
            BaseMessage::assistant("").with_tool_calls(vec![ToolCall::new("a").with_id("call_1")]),
            BaseMessage::tool("call_1", "first"),
        ],
        &request("claude-sonnet-4"),
        false,
    )
    .expect("body");

    assert_eq!(
        messages(&body).last(),
        Some(&json!({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": "call_1",
                "content": "first",
                "cache_control": marker(),
            }],
        }))
    );
}

#[test]
fn a_transient_tail_is_skipped_for_the_previous_stable_message() {
    let mut transient = BaseMessage::user("runtime budget: 5 minutes");
    transient
        .metadata
        .insert("transient".to_owned(), json!(true));

    let body = build_messages_body(
        &[
            BaseMessage::user("go"),
            BaseMessage::assistant("").with_tool_calls(vec![ToolCall::new("a").with_id("call_1")]),
            BaseMessage::tool("call_1", "first"),
            transient,
        ],
        &request("claude-sonnet-4"),
        false,
    )
    .expect("body");

    let messages = messages(&body);
    assert_eq!(messages.len(), 4);
    assert_eq!(
        messages[2]["content"][0]["cache_control"],
        marker(),
        "the stable tool result is the anchor"
    );
    assert_eq!(
        messages[3],
        json!({"role": "user", "content": "runtime budget: 5 minutes"}),
        "the transient tail stays uncached and unconverted"
    );
}

#[test]
fn a_string_content_assistant_anchor_becomes_a_block_array() {
    let body = build_messages_body(
        &[BaseMessage::user("hi"), BaseMessage::assistant("done")],
        &request("claude-sonnet-4"),
        false,
    )
    .expect("body");

    assert_eq!(
        messages(&body),
        [
            json!({"role": "user", "content": "hi"}),
            json!({
                "role": "assistant",
                "content": [{
                    "type": "text",
                    "text": "done",
                    "cache_control": marker(),
                }],
            }),
        ]
    );
}

#[test]
fn disable_cache_control_writes_no_markers_and_keeps_block_arrays() {
    let body = build_messages_body(
        &[
            BaseMessage::system("be brief"),
            BaseMessage::user("hi"),
            BaseMessage::assistant("").with_tool_calls(vec![ToolCall::new("a").with_id("call_1")]),
            BaseMessage::tool("call_1", "first"),
        ],
        &uncached_request("claude-sonnet-4").with_tools(vec![json!({
            "type": "function",
            "function": {"name": "a", "parameters": {"type": "object"}},
        })]),
        false,
    )
    .expect("body");

    assert_eq!(
        body.get("system"),
        Some(&json!([{"type": "text", "text": "be brief"}]))
    );
    assert_eq!(
        body.get("tools"),
        Some(&json!([{
            "name": "a",
            "description": "",
            "input_schema": {"type": "object"},
        }])),
        "the tool definition keeps its shape, just without a marker"
    );
    assert_eq!(
        messages(&body).last(),
        Some(&json!({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": "call_1",
                "content": "first",
            }],
        }))
    );
    let rendered = body.to_string();
    assert!(
        !rendered.contains("cache_control"),
        "no anchor may survive the opt-out: {rendered}"
    );
    assert!(!rendered.contains("disable_cache_control"));
}

#[test]
fn cache_control_ttl_one_hour_marks_all_three_anchors() {
    let request = request("claude-sonnet-4")
        .with_extra(json!({"cache_control_ttl": "1h"}))
        .with_tools(vec![json!({
            "type": "function",
            "function": {"name": "a", "parameters": {"type": "object"}},
        })]);
    let body = build_messages_body(
        &[BaseMessage::system("be brief"), BaseMessage::user("hi")],
        &request,
        false,
    )
    .expect("body");

    assert_eq!(
        system_blocks(&body),
        [json!({"type": "text", "text": "be brief", "cache_control": long_marker()})]
    );
    assert_eq!(body["tools"][0]["cache_control"], long_marker());
    assert_eq!(
        messages(&body),
        [json!({
            "role": "user",
            "content": [{
                "type": "text",
                "text": "hi",
                "cache_control": long_marker(),
            }],
        })]
    );
    assert!(
        body.get("cache_control_ttl").is_none(),
        "the control key must not leak into the body"
    );
}
