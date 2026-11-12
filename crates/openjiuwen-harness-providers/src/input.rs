//! Input conversion and chunk-kind constants for the harness IO surface.
//!
//! [`to_harness_input`] lifts a JSON payload into the
//! [`HarnessInput`] envelope the harness SPI accepts; [`harness_input_text`] extracts the
//! human-readable text projection Python uses for logging and interaction routing.
//!
//! The kind constants name the chunk and delivery kinds the adapter projects:
//! [`INTERACTIVE_INPUT_KIND`] is the delivery kind of a host answer to a pending interaction,
//! [`INTERACTION_KIND`] the chunk kind of an interaction request, [`TURN_LIFECYCLE_KIND`] the
//! chunk kind of a turn lifecycle event, and [`TASK_FAILED_KIND`] the chunk kind of a failed task.

use openjiuwen_harness_protocol::{empty_json_object, HarnessInput, JsonValue};

/// Delivery kind of an interactive host input (an answer to a pending interaction).
pub const INTERACTIVE_INPUT_KIND: &str = "interactive_input";
/// Chunk kind of a harness interaction request (Python `__interaction__`).
pub const INTERACTION_KIND: &str = "__interaction__";
/// Chunk kind of a turn lifecycle event.
pub const TURN_LIFECYCLE_KIND: &str = "turn_lifecycle";
/// Chunk kind of a failed harness task.
pub const TASK_FAILED_KIND: &str = "task_failed";

/// Wraps a JSON payload as a [`HarnessInput`] with empty metadata.
///
/// Python's `to_harness_input` may infer metadata from the payload; the Rust port keeps the
/// content verbatim and always supplies an empty metadata object (documented deviation).
#[must_use]
pub fn to_harness_input(content: JsonValue) -> HarnessInput {
    HarnessInput {
        content,
        metadata: empty_json_object(),
    }
}

/// Extracts the text projection of a harness input.
///
/// Python parity:
/// - a JSON string is returned unchanged;
/// - an array whose elements are **all** JSON objects carrying a string `text` field joins those
///   texts with a blank line (`"\n\n"`); an empty array is not eligible and falls through;
/// - any other value (object, number, bool, null, mixed array) is rendered as compact JSON, or as
///   the empty string if serialization fails (it cannot for a [`JsonValue`]).
#[must_use]
pub fn harness_input_text(input: &HarnessInput) -> String {
    match &input.content {
        JsonValue::String(text) => text.clone(),
        JsonValue::Array(items) => match joined_text_blocks(items) {
            Some(text) => text,
            None => compact_json(&input.content),
        },
        other => compact_json(other),
    }
}

/// Joins the `text` fields of an array of text blocks.
///
/// Returns [`None`] when the array is empty or any element is not an object with a string `text`
/// field, which makes the caller fall back to compact JSON.
fn joined_text_blocks(items: &[JsonValue]) -> Option<String> {
    if items.is_empty() {
        return None;
    }
    let mut texts = Vec::with_capacity(items.len());
    for item in items {
        let object = item.as_object()?;
        texts.push(object.get("text")?.as_str()?);
    }
    Some(texts.join("\n\n"))
}

/// Renders a JSON value as compact JSON, falling back to the empty string.
fn compact_json(value: &JsonValue) -> String {
    serde_json::to_string(value).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    // Tests assert by panicking on failure, which is the intent of a test.
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use openjiuwen_harness_protocol::{empty_json_object, HarnessInput};
    use serde_json::json;

    use super::{
        harness_input_text, to_harness_input, INTERACTION_KIND, INTERACTIVE_INPUT_KIND,
        TASK_FAILED_KIND, TURN_LIFECYCLE_KIND,
    };

    #[test]
    fn kind_constants_match_python_wire_values() {
        assert_eq!(INTERACTIVE_INPUT_KIND, "interactive_input");
        assert_eq!(INTERACTION_KIND, "__interaction__");
        assert_eq!(TURN_LIFECYCLE_KIND, "turn_lifecycle");
        assert_eq!(TASK_FAILED_KIND, "task_failed");
    }

    #[test]
    fn to_harness_input_keeps_content_and_empties_metadata() {
        let input = to_harness_input(json!({"prompt": "hi"}));
        assert_eq!(input.content, json!({"prompt": "hi"}));
        assert!(input.metadata.is_empty());
        assert_eq!(input.metadata, empty_json_object());
    }

    #[test]
    fn to_harness_input_keeps_string_content_verbatim() {
        let input = to_harness_input(json!("hello"));
        assert_eq!(input.content, json!("hello"));
    }

    #[test]
    fn text_of_string_content_is_the_string() {
        assert_eq!(harness_input_text(&HarnessInput::text("hello")), "hello");
        assert_eq!(harness_input_text(&HarnessInput::text("")), "");
    }

    #[test]
    fn text_of_all_text_blocks_joins_with_blank_line() {
        let input = HarnessInput {
            content: json!([{"text": "first"}, {"text": "second"}]),
            metadata: empty_json_object(),
        };
        assert_eq!(harness_input_text(&input), "first\n\nsecond");
    }

    #[test]
    fn text_of_single_text_block_is_the_block_text() {
        let input = HarnessInput {
            content: json!([{"text": "only"}]),
            metadata: empty_json_object(),
        };
        assert_eq!(harness_input_text(&input), "only");
    }

    #[test]
    fn text_of_empty_array_falls_back_to_compact_json() {
        let input = HarnessInput {
            content: json!([]),
            metadata: empty_json_object(),
        };
        assert_eq!(harness_input_text(&input), "[]");
    }

    #[test]
    fn text_of_mixed_array_falls_back_to_compact_json() {
        let input = HarnessInput {
            content: json!([{"text": "first"}, {"not_text": "second"}]),
            metadata: empty_json_object(),
        };
        assert_eq!(
            harness_input_text(&input),
            r#"[{"text":"first"},{"not_text":"second"}]"#
        );
    }

    #[test]
    fn text_of_non_string_text_field_falls_back_to_compact_json() {
        let input = HarnessInput {
            content: json!([{"text": 42}]),
            metadata: empty_json_object(),
        };
        assert_eq!(harness_input_text(&input), r#"[{"text":42}]"#);
    }

    #[test]
    fn text_of_object_is_compact_json() {
        let input = HarnessInput {
            content: json!({"prompt": "hi", "turn": 1}),
            metadata: empty_json_object(),
        };
        assert_eq!(harness_input_text(&input), r#"{"prompt":"hi","turn":1}"#);
    }

    #[test]
    fn text_of_scalars_is_compact_json() {
        let number = HarnessInput {
            content: json!(12),
            metadata: empty_json_object(),
        };
        assert_eq!(harness_input_text(&number), "12");
        let null = HarnessInput {
            content: serde_json::Value::Null,
            metadata: empty_json_object(),
        };
        assert_eq!(harness_input_text(&null), "null");
    }
}
