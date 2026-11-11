#![allow(clippy::unwrap_used, clippy::expect_used)]
// ports tests/unit_tests/core/foundation/prompt/ and llm/test_message_chunk.py behaviors

//! Wire-shape tests for [`BaseMessage`] and the types it carries.
//!
//! Every assertion compares whole `serde_json::Value`s, so the tests pin the
//! field names, the omission of empty optional fields, the nested tool-call
//! shape, and the tolerance for unknown content parts — the properties the
//! OpenAI-compatible and Anthropic clients rely on.

use openjiuwen_foundation::prompt::content::{ContentPart, ImageUrl, MessageContent};
use openjiuwen_foundation::prompt::message::BaseMessage;
use openjiuwen_foundation::prompt::role::Role;
use openjiuwen_foundation::prompt::tool_call::ToolCall;
use openjiuwen_foundation::prompt::usage::Usage;
use serde_json::json;

#[test]
fn string_content_round_trips_as_a_plain_json_string() {
    let message = BaseMessage::user("hi");
    assert_eq!(
        serde_json::to_value(&message).unwrap(),
        json!({"role": "user", "content": "hi"})
    );
    assert_eq!(
        serde_json::to_string(&message).unwrap(),
        r#"{"role":"user","content":"hi"}"#
    );

    let back: BaseMessage =
        serde_json::from_value(json!({"role": "user", "content": "hi"})).unwrap();
    assert_eq!(back, message);
    assert_eq!(back.text(), "hi");
}

#[test]
fn parts_content_round_trips_with_image_url_detail() {
    let message = BaseMessage::user(vec![
        ContentPart::Text {
            text: "look".to_owned(),
        },
        ContentPart::ImageUrl {
            image_url: ImageUrl {
                url: "https://x/y.png".to_owned(),
                detail: None,
            },
        },
    ]);
    let value = serde_json::to_value(&message).unwrap();
    assert_eq!(
        value,
        json!({
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
            ],
        }),
        "a missing `detail` must not be serialized as null"
    );
    assert_eq!(
        serde_json::from_value::<BaseMessage>(value).unwrap(),
        message
    );

    let with_detail = BaseMessage::user(vec![ContentPart::ImageUrl {
        image_url: ImageUrl {
            url: "https://x/y.png".to_owned(),
            detail: Some("high".to_owned()),
        },
    }]);
    assert_eq!(
        serde_json::to_value(&with_detail).unwrap(),
        json!({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "https://x/y.png", "detail": "high"}},
            ],
        })
    );
}

#[test]
fn unknown_content_parts_survive_a_message_round_trip_verbatim() {
    let raw = json!({
        "role": "user",
        "content": [
            {"type": "text", "text": "watch this"},
            {"type": "video", "source": {"url": "v"}},
        ],
    });
    let message: BaseMessage = serde_json::from_value(raw.clone()).unwrap();
    assert_eq!(
        message.content,
        MessageContent::Parts(vec![
            ContentPart::Text {
                text: "watch this".to_owned()
            },
            ContentPart::Other(json!({"type": "video", "source": {"url": "v"}})),
        ])
    );
    assert_eq!(serde_json::to_value(&message).unwrap(), raw);
}

#[test]
fn null_content_becomes_empty_text() {
    let message: BaseMessage =
        serde_json::from_value(json!({"role": "assistant", "content": null})).unwrap();
    assert_eq!(message.content, MessageContent::Text(String::new()));
    assert!(message.content.is_empty());
    assert_eq!(message.text(), "");
    assert_eq!(message.content.to_plain_text(), "");
}

#[test]
fn flat_tool_calls_convert_to_and_from_the_openai_nested_shape() {
    let message = BaseMessage::assistant("").with_tool_calls(vec![ToolCall::new("get_weather")
        .with_id("call_1")
        .with_index(0)
        .with_arguments(json!({"city": "Paris"}))]);
    let value = serde_json::to_value(&message).unwrap();
    // The crate's own tool-call serde is flat (`type` at the top level).
    assert_eq!(
        value,
        json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "name": "get_weather",
                "arguments": "{\"city\":\"Paris\"}",
                "index": 0,
            }],
        })
    );

    let call = &message.tool_calls[0];
    let nested = call.to_openai();
    assert_eq!(
        nested,
        json!({
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"},
        })
    );
    assert_eq!(
        nested.get("index"),
        None,
        "index stays outside the OpenAI shape"
    );

    let parsed = ToolCall::from_openai(&nested).unwrap();
    assert_eq!(parsed.id.as_deref(), Some("call_1"));
    assert_eq!(parsed.kind, "function");
    assert_eq!(parsed.name, "get_weather");
    assert_eq!(parsed.arguments, "{\"city\":\"Paris\"}");
    assert_eq!(
        parsed.index, None,
        "from_openai reads no index from the OpenAI shape"
    );
    assert_eq!(parsed.arguments_value().unwrap(), json!({"city": "Paris"}));

    let mut expected_with_index = call.clone();
    expected_with_index.index = Some(3);
    let indexed = ToolCall::from_openai_with_index(&nested, 3).unwrap();
    assert_eq!(indexed, expected_with_index);

    let flat_back = ToolCall::from_openai(&value["tool_calls"][0]).unwrap();
    assert_eq!(flat_back, *call);
}

#[test]
fn usage_and_tool_call_serde_use_snake_case_wire_names() {
    let usage = Usage {
        input_tokens: 11,
        output_tokens: 5,
        total_tokens: 16,
        cache_tokens: 3,
        reasoning_tokens: 2,
        extra: std::collections::BTreeMap::new(),
    };
    assert_eq!(
        serde_json::to_value(&usage).unwrap(),
        json!({
            "input_tokens": 11,
            "output_tokens": 5,
            "total_tokens": 16,
            "cache_tokens": 3,
            "reasoning_tokens": 2,
        })
    );

    let anonymous = ToolCall::new("noop");
    let value = serde_json::to_value(&anonymous).unwrap();
    assert_eq!(
        value,
        json!({"type": "function", "name": "noop", "arguments": ""})
    );
    assert!(value.get("kind").is_none(), "the wire field is `type`");
    assert!(value.get("id").is_none(), "a missing id is omitted");
    assert!(value.get("index").is_none(), "a missing index is omitted");
}

#[test]
fn unknown_message_fields_are_ignored() {
    let parsed: BaseMessage = serde_json::from_value(json!({
        "role": "assistant",
        "content": "x",
        "vendor_field": 1,
    }))
    .unwrap();
    assert_eq!(parsed, BaseMessage::assistant("x"));

    let value = serde_json::to_value(&parsed).unwrap();
    assert_eq!(value, json!({"role": "assistant", "content": "x"}));
    assert!(value.get("vendor_field").is_none());
}

#[test]
fn metadata_round_trips_arbitrary_json_and_is_omitted_when_empty() {
    let message = BaseMessage::assistant("x")
        .with_metadata("trace", json!("t1"))
        .with_metadata("score", json!(0.5))
        .with_metadata("tags", json!(["a", "b"]))
        .with_metadata("nested", json!({"k": [1, 2]}));
    let value = serde_json::to_value(&message).unwrap();
    assert_eq!(
        value["metadata"],
        json!({
            "trace": "t1",
            "score": 0.5,
            "tags": ["a", "b"],
            "nested": {"k": [1, 2]},
        })
    );
    assert_eq!(
        serde_json::from_value::<BaseMessage>(value).unwrap(),
        message
    );

    let without = serde_json::to_value(BaseMessage::assistant("x")).unwrap();
    assert!(without.get("metadata").is_none());
}

#[test]
fn assistant_reasoning_and_tool_calls_round_trip() {
    let message = BaseMessage::assistant("using a tool")
        .with_reasoning("the user asked for the weather")
        .with_tool_calls(vec![ToolCall::new("get_weather").with_id("call_9")]);
    let value = serde_json::to_value(&message).unwrap();
    assert_eq!(
        value,
        json!({
            "role": "assistant",
            "content": "using a tool",
            "tool_calls": [{
                "id": "call_9",
                "type": "function",
                "name": "get_weather",
                "arguments": "",
            }],
            "reasoning_content": "the user asked for the weather",
        })
    );

    let back: BaseMessage = serde_json::from_value(value).unwrap();
    assert_eq!(back, message);
    assert_eq!(back.role, Role::Assistant);
    assert_eq!(back.tool_calls[0].arguments_value().unwrap(), json!(null));
}

#[test]
fn tool_message_carries_the_answered_call_id() {
    let message = BaseMessage::tool("call_9", "22C");
    assert_eq!(
        serde_json::to_value(&message).unwrap(),
        json!({"role": "tool", "content": "22C", "tool_call_id": "call_9"})
    );
    let back: BaseMessage =
        serde_json::from_value(serde_json::to_value(&message).unwrap()).unwrap();
    assert_eq!(back, message);
    assert_eq!(back.role, Role::Tool);
}
