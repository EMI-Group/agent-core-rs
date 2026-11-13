//! Body-level tests for the Anthropic request projection.
//!
//! These assert the exact JSON of [`build_messages_body`](super::build_messages_body)
//! for the system block array, the mid-conversation `system` placement rule, the
//! tool/tool-result projection, multimodal parts, sampling fields, and the
//! `max_tokens` caps. Cache-breakpoint placement lives in the sibling
//! `cache_tests` module.

use super::*;
use crate::prompt::{ContentPart, ImageUrl, MessageContent, ToolCall};
use serde_json::json;

/// The `cache_control` marker written by default.
pub(super) fn marker() -> Value {
    json!({"type": "ephemeral"})
}

/// The `cache_control` marker written when `cache_control_ttl` is `"1h"`.
pub(super) fn long_marker() -> Value {
    json!({"type": "ephemeral", "ttl": "1h"})
}

pub(super) fn system_blocks(body: &Value) -> &[Value] {
    body.get("system")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default()
}

pub(super) fn messages(body: &Value) -> &[Value] {
    body.get("messages")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default()
}

pub(super) fn request(model: &str) -> ModelRequestConfig {
    ModelRequestConfig::new(model)
}

/// A request whose cache anchors are switched off, for tests that assert a
/// projection shape rather than the breakpoints (covered separately).
pub(super) fn uncached_request(model: &str) -> ModelRequestConfig {
    ModelRequestConfig::new(model).with_extra(json!({"disable_cache_control": true}))
}

#[test]
fn normalize_api_base_strips_the_endpoint_and_version() {
    assert_eq!(
        normalize_api_base(" https://api.anthropic.com/v1/messages/ "),
        "https://api.anthropic.com"
    );
    assert_eq!(
        normalize_api_base("https://api.anthropic.com/v1/"),
        "https://api.anthropic.com"
    );
    assert_eq!(
        normalize_api_base("https://api.anthropic.com/"),
        "https://api.anthropic.com"
    );
    assert_eq!(
        normalize_api_base("https://api.anthropic.com"),
        "https://api.anthropic.com"
    );
    assert_eq!(
        normalize_api_base("https://gateway.test/anthropic/v1"),
        "https://gateway.test/anthropic"
    );
    assert_eq!(
        normalize_api_base("https://gateway.test/anthropic/v1/messages"),
        "https://gateway.test/anthropic"
    );
    assert_eq!(normalize_api_base("   "), "");
}

#[test]
fn an_empty_model_name_is_a_validation_error() {
    let error = build_messages_body(&[], &ModelRequestConfig::default(), false)
        .expect_err("the model name is required");
    assert_eq!(
        error.kind(),
        openjiuwen_common::error::ErrorKind::Validation
    );
    assert!(error.to_string().contains("model_name"));
}

#[test]
fn the_body_carries_the_model_messages_max_tokens_and_stream_flag() {
    let body = build_messages_body(
        &[BaseMessage::user("hello")],
        &uncached_request("claude-sonnet-4"),
        false,
    )
    .expect("body");

    assert_eq!(
        body,
        json!({
            "model": "claude-sonnet-4",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 32768,
            "stream": false,
        })
    );
}

#[test]
fn the_stream_flag_is_written_verbatim() {
    let body = build_messages_body(
        &[BaseMessage::user("hi")],
        &request("claude-sonnet-4"),
        true,
    )
    .expect("body");
    assert_eq!(body.get("stream"), Some(&Value::Bool(true)));
}

#[test]
fn sampling_top_p_and_stop_are_mapped() {
    let request = request("claude-sonnet-4")
        .with_temperature(0.2)
        .with_top_p(0.9)
        .with_stop(["END".to_owned(), "STOP".to_owned()]);
    let body = build_messages_body(&[BaseMessage::user("hi")], &request, false).expect("body");

    assert_eq!(body.get("temperature"), Some(&json!(0.2_f32)));
    assert_eq!(body.get("top_p"), Some(&json!(0.9_f32)));
    assert_eq!(body.get("stop_sequences"), Some(&json!(["END", "STOP"])));
    assert!(
        body.get("stop").is_none(),
        "the OpenAI key must not be sent"
    );
}

#[test]
fn a_single_system_turn_becomes_one_marked_block() {
    let body = build_messages_body(
        &[BaseMessage::system("be brief"), BaseMessage::user("hello")],
        &request("claude-sonnet-4"),
        false,
    )
    .expect("body");

    assert_eq!(
        body.get("system"),
        Some(&json!([{
            "type": "text",
            "text": "be brief",
            "cache_control": marker(),
        }])),
        "a single system turn is a one-element block array"
    );
    assert_eq!(
        messages(&body),
        [json!({
            "role": "user",
            "content": [{
                "type": "text",
                "text": "hello",
                "cache_control": marker(),
            }],
        })]
    );
}

#[test]
fn two_system_turns_become_two_blocks_and_only_the_last_is_marked() {
    let body = build_messages_body(
        &[
            BaseMessage::system("be brief"),
            BaseMessage::system(vec![
                ContentPart::Text {
                    text: "and kind".to_owned(),
                },
                ContentPart::Audio(crate::prompt::AudioPart {
                    data: "ignored".to_owned(),
                    format: None,
                }),
            ]),
            BaseMessage::user("hello"),
        ],
        &request("claude-sonnet-4"),
        false,
    )
    .expect("body");

    assert_eq!(
        body.get("system"),
        Some(&json!([
            {"type": "text", "text": "be brief"},
            {
                "type": "text",
                "text": "and kind",
                "cache_control": marker(),
            },
        ])),
        "each turn keeps its own block and only the last carries the marker"
    );
    assert_eq!(
        messages(&body),
        [json!({
            "role": "user",
            "content": [{
                "type": "text",
                "text": "hello",
                "cache_control": marker(),
            }],
        })]
    );
}

#[test]
fn an_empty_system_turn_omits_the_field() {
    let body = build_messages_body(
        &[
            BaseMessage::system(""),
            BaseMessage::system(vec![ContentPart::Other(json!({"type": "image"}))]),
            BaseMessage::user("hi"),
        ],
        &request("claude-sonnet-4"),
        false,
    )
    .expect("body");

    assert!(body.get("system").is_none());
    assert_eq!(messages(&body).len(), 1);
}

#[test]
fn no_system_key_is_written_without_a_system_turn() {
    let body = build_messages_body(
        &[BaseMessage::user("hi"), BaseMessage::assistant("hello")],
        &request("claude-sonnet-4"),
        false,
    )
    .expect("body");

    assert!(body.get("system").is_none());
}

#[test]
fn an_inline_system_turn_stays_in_messages_for_a_supported_family() {
    let body = build_messages_body(
        &[
            BaseMessage::system("lead"),
            BaseMessage::user("hi"),
            BaseMessage::system("extra"),
            BaseMessage::assistant("ok"),
        ],
        &request("claude-opus-4-8"),
        false,
    )
    .expect("body");

    assert_eq!(
        messages(&body),
        [
            json!({"role": "user", "content": "hi"}),
            json!({"role": "system", "content": "extra"}),
            json!({
                "role": "assistant",
                "content": [{
                    "type": "text",
                    "text": "ok",
                    "cache_control": marker(),
                }],
            }),
        ],
        "the mid-conversation turn keeps its position"
    );
    assert_eq!(
        system_blocks(&body),
        [json!({"type": "text", "text": "lead", "cache_control": marker()})],
        "only the leading prompt is hoisted"
    );
}

#[test]
fn an_inline_system_turn_at_index_zero_is_hoisted() {
    let body = build_messages_body(
        &[BaseMessage::system("lead"), BaseMessage::user("hi")],
        &request("claude-opus-4-8"),
        false,
    )
    .expect("body");

    assert_eq!(
        messages(&body),
        [json!({
            "role": "user",
            "content": [{
                "type": "text",
                "text": "hi",
                "cache_control": marker(),
            }],
        })]
    );
    assert_eq!(
        system_blocks(&body),
        [json!({"type": "text", "text": "lead", "cache_control": marker()})]
    );
}

#[test]
fn an_inline_system_turn_after_an_assistant_turn_is_hoisted() {
    let body = build_messages_body(
        &[
            BaseMessage::user("hi"),
            BaseMessage::assistant("ok"),
            BaseMessage::system("extra"),
            BaseMessage::user("more"),
        ],
        &request("claude-opus-4-8"),
        false,
    )
    .expect("body");

    assert_eq!(
        messages(&body),
        [
            json!({"role": "user", "content": "hi"}),
            json!({"role": "assistant", "content": "ok"}),
            json!({
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": "more",
                    "cache_control": marker(),
                }],
            }),
        ],
        "no message may carry a system role here"
    );
    assert_eq!(
        system_blocks(&body),
        [json!({"type": "text", "text": "extra", "cache_control": marker()})]
    );
}

#[test]
fn a_hoisted_system_turn_flushes_pending_tool_results() {
    let body = build_messages_body(
        &[
            BaseMessage::user("run both"),
            BaseMessage::assistant("").with_tool_calls(vec![
                ToolCall::new("a").with_id("call_1"),
                ToolCall::new("b").with_id("call_2"),
            ]),
            BaseMessage::tool("call_1", "first"),
            BaseMessage::system("be careful"),
            BaseMessage::tool("call_2", "second"),
        ],
        &request("claude-sonnet-4"),
        false,
    )
    .expect("body");

    assert_eq!(
        messages(&body),
        [
            json!({"role": "user", "content": "run both"}),
            json!({
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "call_1", "name": "a", "input": {}},
                    {"type": "tool_use", "id": "call_2", "name": "b", "input": {}},
                ],
            }),
            json!({
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "first"}],
            }),
            json!({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "call_2",
                    "content": "second",
                    "cache_control": marker(),
                }],
            }),
        ],
        "a hoisted system turn splits the tool results instead of merging across it"
    );
    assert_eq!(
        system_blocks(&body),
        [json!({"type": "text", "text": "be careful", "cache_control": marker()})]
    );
}

#[test]
fn the_body_stays_valid_with_a_system_block_array() {
    let body = build_messages_body(
        &[
            BaseMessage::system("be brief"),
            BaseMessage::system("and kind"),
            BaseMessage::user("hi"),
        ],
        &request("claude-sonnet-4"),
        true,
    )
    .expect("body");

    assert!(
        body["system"].is_array(),
        "`system` is always a block array"
    );
    assert!(
        body["system"]
            .as_array()
            .expect("array")
            .iter()
            .all(|block| block["type"] == "text"),
        "every hoisted turn is a text block"
    );
    assert_eq!(body["model"], json!("claude-sonnet-4"));
    assert_eq!(body["max_tokens"], json!(32768));
    assert_eq!(body["stream"], json!(true));
    assert!(
        messages(&body)
            .iter()
            .all(|message| message["role"] != "system"),
        "a model without inline system support sees no system role"
    );
}

#[test]
fn multimodal_parts_become_anthropic_image_blocks() {
    let body = build_messages_body(
        &[BaseMessage::user(vec![
            ContentPart::Text {
                text: "what is this?".to_owned(),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "data:image/png;base64,AAAA".to_owned(),
                    detail: None,
                },
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "https://example.test/cat.jpg".to_owned(),
                    detail: Some("high".to_owned()),
                },
            },
        ])],
        &request("claude-sonnet-4"),
        false,
    )
    .expect("body");

    assert_eq!(
        messages(&body),
        [json!({
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this?"},
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"},
                },
                {
                    "type": "image",
                    "source": {"type": "url", "url": "https://example.test/cat.jpg"},
                    "cache_control": marker(),
                },
            ],
        })]
    );
}

#[test]
fn unsupported_image_urls_are_validation_errors() {
    for url in [
        "ftp://example.test/cat.png",
        "data:image/png;base64,",
        "data:image/png,not-base64",
        "relative/cat.png",
    ] {
        let error = build_messages_body(
            &[BaseMessage::user(vec![ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: url.to_owned(),
                    detail: None,
                },
            }])],
            &request("claude-sonnet-4"),
            false,
        )
        .expect_err("the image cannot be represented");

        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation,
            "url: {url}"
        );
        let rendered = error.to_string();
        assert!(rendered.contains("Anthropic image"), "error: {rendered}");
    }
}

#[test]
fn audio_file_and_other_parts_pass_through_verbatim() {
    let vendor = json!({"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": "x"}});
    let body = build_messages_body(
        &[BaseMessage::user(vec![
            ContentPart::Audio(crate::prompt::AudioPart {
                data: "AAAA".to_owned(),
                format: Some("wav".to_owned()),
            }),
            ContentPart::File(crate::prompt::FilePart {
                url: Some("https://example.test/a.pdf".to_owned()),
                data: None,
                mime_type: Some("application/pdf".to_owned()),
                filename: None,
            }),
            ContentPart::Other(vendor.clone()),
        ])],
        &uncached_request("claude-sonnet-4"),
        false,
    )
    .expect("body");

    assert_eq!(
        messages(&body)[0]["content"],
        json!([
            {"type": "audio", "audio": {"data": "AAAA", "format": "wav"}},
            {
                "type": "file",
                "file": {
                    "url": "https://example.test/a.pdf",
                    "mime_type": "application/pdf",
                },
            },
            vendor,
        ])
    );
}

#[test]
fn a_system_turn_accepts_parts_but_only_keeps_text() {
    let body = build_messages_body(
        &[BaseMessage::system(MessageContent::Parts(vec![
            ContentPart::Text {
                text: "first".to_owned(),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "https://example.test/cat.jpg".to_owned(),
                    detail: None,
                },
            },
        ]))],
        &request("claude-sonnet-4"),
        false,
    )
    .expect("body");

    assert_eq!(
        body.get("system"),
        Some(&json!([{
            "type": "text",
            "text": "first",
            "cache_control": marker(),
        }]))
    );
}

#[test]
fn assistant_tool_calls_become_tool_use_blocks() {
    let body = build_messages_body(
        &[BaseMessage::assistant("let me check").with_tool_calls(vec![
            ToolCall::new("get_weather")
                .with_id("call_1")
                .with_arguments(json!({"city": "Paris"})),
            ToolCall::new("get_time").with_id("call_2"),
        ])],
        &request("claude-sonnet-4"),
        false,
    )
    .expect("body");

    assert_eq!(
        messages(&body),
        [json!({
            "role": "assistant",
            "content": [
                {"type": "text", "text": "let me check"},
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "get_weather",
                    "input": {"city": "Paris"},
                },
                {
                    "type": "tool_use",
                    "id": "call_2",
                    "name": "get_time",
                    "input": {},
                    "cache_control": marker(),
                },
            ],
        })]
    );
}

#[test]
fn unparseable_tool_arguments_are_preserved() {
    let mut call = ToolCall::new("broken").with_id("call_1");
    call.arguments = "{not json".to_owned();
    let body = build_messages_body(
        &[BaseMessage::assistant("").with_tool_calls(vec![call])],
        &uncached_request("claude-sonnet-4"),
        false,
    )
    .expect("body");

    assert_eq!(
        messages(&body)[0]["content"],
        json!([{
            "type": "tool_use",
            "id": "call_1",
            "name": "broken",
            "input": {"_raw_arguments": "{not json"},
        }])
    );
}

#[test]
fn a_tool_call_without_an_id_is_a_validation_error() {
    let error = build_messages_body(
        &[BaseMessage::assistant("").with_tool_calls(vec![ToolCall::new("nameless")])],
        &request("claude-sonnet-4"),
        false,
    )
    .expect_err("the API requires an id");

    assert_eq!(
        error.kind(),
        openjiuwen_common::error::ErrorKind::Validation
    );
    assert!(error.to_string().contains("id"));
}

#[test]
fn consecutive_tool_messages_merge_into_one_user_message() {
    let body = build_messages_body(
        &[
            BaseMessage::user("run both"),
            BaseMessage::assistant("").with_tool_calls(vec![
                ToolCall::new("a").with_id("call_1"),
                ToolCall::new("b").with_id("call_2"),
            ]),
            BaseMessage::tool("call_1", "first"),
            BaseMessage::tool(
                "call_2",
                vec![ContentPart::Text {
                    text: "second".to_owned(),
                }],
            ),
        ],
        &request("claude-sonnet-4"),
        false,
    )
    .expect("body");

    let messages = messages(&body);
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[0], json!({"role": "user", "content": "run both"}));
    assert_eq!(messages[1]["role"], "assistant");
    assert_eq!(
        messages[2],
        json!({
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call_1", "content": "first"},
                {
                    "type": "tool_result",
                    "tool_use_id": "call_2",
                    "content": "second",
                    "cache_control": marker(),
                },
            ],
        })
    );
}

#[test]
fn a_tool_message_without_a_call_id_is_a_validation_error() {
    let mut message = BaseMessage::tool("call_1", "result");
    message.tool_call_id = None;
    let error = build_messages_body(&[message], &request("claude-sonnet-4"), false)
        .expect_err("tool_call_id is required");

    assert_eq!(
        error.kind(),
        openjiuwen_common::error::ErrorKind::Validation
    );
    assert!(error.to_string().contains("tool_call_id"));
}

#[test]
fn tools_are_converted_from_the_openai_shape() {
    let body = build_messages_body(
        &[BaseMessage::user("hi")],
        &request("claude-sonnet-4").with_tools(vec![
            json!({
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Weather lookup",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }),
            json!({
                "type": "function",
                "function": {
                    "name": "get_time",
                    "input_schema": {"type": "object", "properties": {}},
                },
            }),
            json!({"name": "already_anthropic", "input_schema": {"type": "object"}}),
        ]),
        false,
    )
    .expect("body");

    assert_eq!(
        body.get("tools"),
        Some(&json!([
            {
                "name": "get_weather",
                "description": "Weather lookup",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
            {
                "name": "get_time",
                "description": "",
                "input_schema": {"type": "object", "properties": {}},
            },
            {
                "name": "already_anthropic",
                "input_schema": {"type": "object"},
                "cache_control": marker(),
            },
        ]))
    );
}

#[test]
fn an_empty_tool_list_is_omitted() {
    let body = build_messages_body(
        &[BaseMessage::user("hi")],
        &request("claude-sonnet-4").with_tools(Vec::new()),
        false,
    )
    .expect("body");
    assert!(body.get("tools").is_none());
}

#[test]
fn tool_choice_values_are_mapped() {
    let cases = [
        (json!("auto"), None),
        (json!("none"), Some(json!({"type": "none"}))),
        (json!("required"), Some(json!({"type": "any"}))),
        (json!("any"), Some(json!({"type": "any"}))),
        (
            json!({"type": "function", "function": {"name": "get_weather"}}),
            Some(json!({"type": "tool", "name": "get_weather"})),
        ),
        (
            json!({"type": "tool", "name": "get_time"}),
            Some(json!({"type": "tool", "name": "get_time"})),
        ),
        (json!({"type": "unknown"}), None),
        (json!(42), None),
    ];

    for (choice, expected) in cases {
        let body = build_messages_body(
            &[BaseMessage::user("hi")],
            &request("claude-sonnet-4").with_tool_choice(choice.clone()),
            false,
        )
        .expect("body");
        assert_eq!(
            body.get("tool_choice"),
            expected.as_ref(),
            "tool_choice: {choice}"
        );
    }
}

#[test]
fn max_tokens_uses_the_default_and_the_family_caps() {
    let cases: [(&str, Option<u32>, u32); 8] = [
        ("claude-sonnet-4", None, 32768),
        ("claude-3-5-sonnet-latest", None, 8192),
        ("claude-3.5-sonnet", None, 8192),
        ("Claude-3-Opus-20240229", None, 4096),
        ("claude-3-sonnet-20240229", Some(16_384), 4096),
        ("claude-3-haiku-20240307", Some(1024), 1024),
        ("claude-3-5-sonnet", Some(32_000), 8192),
        ("claude-sonnet-4", Some(64_000), 64_000),
    ];

    for (model, requested, expected) in cases {
        let request = ModelRequestConfig {
            max_tokens: requested,
            ..ModelRequestConfig::new(model)
        };
        let body = build_messages_body(&[BaseMessage::user("hi")], &request, false).expect("body");
        assert_eq!(
            body.get("max_tokens"),
            Some(&Value::from(expected)),
            "model: {model}, requested: {requested:?}"
        );
    }
}

#[test]
fn extra_is_merged_last_and_wins() {
    let request = request("claude-sonnet-4")
        .with_temperature(0.2)
        .with_extra(json!({"temperature": 0.7, "custom_flag": true, "model": "override"}));
    let body = build_messages_body(&[BaseMessage::user("hi")], &request, false).expect("body");

    assert_eq!(body.get("temperature"), Some(&json!(0.7)));
    assert_eq!(body.get("custom_flag"), Some(&json!(true)));
    assert_eq!(body.get("model"), Some(&json!("override")));
}

#[test]
fn non_object_extra_is_ignored() {
    let request = request("claude-sonnet-4").with_extra(json!([1, 2, 3]));
    let body = build_messages_body(&[BaseMessage::user("hi")], &request, false).expect("body");
    assert_eq!(body.get("model"), Some(&json!("claude-sonnet-4")));
    assert_eq!(
        messages(&body),
        [json!({
            "role": "user",
            "content": [{
                "type": "text",
                "text": "hi",
                "cache_control": marker(),
            }],
        })],
        "a non-object extra must not disturb the anchors"
    );
}
