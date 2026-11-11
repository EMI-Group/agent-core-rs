//! Internal (de)serialization helpers of the store module.
//!
//! Records are persisted as the Python-compatible [`MemoryRecord`] JSON, minus the
//! embedding: the vector collection is the authoritative copy of the embedding, so
//! the KV snapshot must not carry a second, potentially stale vector.
//!
//! [`plain_text`] renders one [`MessageContent`] as the flat text stored for a
//! conversation message; a non-text part is preserved as compact JSON instead of
//! being dropped, so image/audio/file attachments stay inspectable.

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::prompt::{ContentPart, MessageContent};
#[cfg(test)]
use serde_json::Value;

use crate::record::MemoryRecord;

/// Serializes `record` for the record KV store, with `embedding` forced to `None`.
///
/// The record is cloned so the caller's value is untouched; the resulting JSON uses
/// the Python field names (`text`, `type`, `fields`) and flattens the scope.
pub(crate) fn record_to_text(record: &MemoryRecord) -> Result<String> {
    let mut record = record.clone();
    record.embedding = None;
    serde_json::to_string(&record)
        .map_err(|error| Error::external(error).with_context("serializing a memory record"))
}

/// Parses the JSON stored by [`record_to_text`] back into a [`MemoryRecord`].
///
/// An absent `embedding` member deserializes to `None` through the record's serde
/// default; malformed JSON or a wrong shape is reported as `Error::Validation`.
pub(crate) fn record_from_text(text: &str) -> Result<MemoryRecord> {
    serde_json::from_str(text)
        .map_err(|error| Error::validation(format!("malformed memory record JSON: {error}")))
}

/// Renders `content` as plain text.
///
/// [`MessageContent::Text`] is returned verbatim. [`MessageContent::Parts`] renders
/// every part and joins the results with a single space: a text part contributes its
/// text, any other part its compact JSON. Empty text and an empty part list give `""`.
pub(crate) fn plain_text(content: &MessageContent) -> String {
    match content {
        MessageContent::Text(text) => text.clone(),
        MessageContent::Parts(parts) => parts.iter().map(render_part).collect::<Vec<_>>().join(" "),
    }
}

/// Renders one content part: its text, or its compact JSON for any non-text part.
fn render_part(part: &ContentPart) -> String {
    match part {
        ContentPart::Text { text } => text.clone(),
        other => render_json(other),
    }
}

/// Serializes a non-text content part as compact JSON.
///
/// A serialization failure cannot be represented in the flat message text, so it is
/// logged and rendered as an empty part rather than failing the whole message write.
fn render_json(part: &ContentPart) -> String {
    match serde_json::to_string(part) {
        Ok(json) => json,
        Err(error) => {
            tracing::warn!(
                target: "openjiuwen_memory::store",
                %error,
                "failed to render a non-text message part as JSON"
            );
            String::new()
        }
    }
}

/// The compact JSON of one part, used by the tests to state the expected rendering.
#[cfg(test)]
fn part_json(part: &ContentPart) -> Value {
    serde_json::to_value(part).expect("a content part serializes to JSON")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::{MemoryScope, MemoryType};
    use chrono::DateTime;
    use openjiuwen_foundation::prompt::{ContentPart, ImageUrl, MessageContent};
    use serde_json::json;

    fn record() -> MemoryRecord {
        MemoryRecord::new(
            MemoryScope::new("alice", "ws").session("s1"),
            "Alice prefers dark mode",
            MemoryType::SemanticMemory,
        )
        .with_id("rec-1")
        .with_importance(0.7)
        .with_metadata("source", json!("conversation"))
        .with_timestamp(DateTime::from_timestamp(1_700_000_000, 0).expect("fixed time"))
    }

    #[test]
    fn record_text_omits_the_embedding() {
        let text = record_to_text(&record().with_embedding(vec![0.1, 0.2, 0.3]))
            .expect("serialize record");
        let value: Value = serde_json::from_str(&text).expect("valid JSON");
        assert_eq!(value.get("embedding"), None);
        assert_eq!(value["text"], json!("Alice prefers dark mode"));
        assert_eq!(value["type"], json!("semantic_memory"));
    }

    #[test]
    fn record_text_round_trips_through_python_field_names() {
        let original = record()
            .with_metadata("source", json!("conversation"))
            .with_embedding(vec![1.0, 0.0]);
        let restored =
            record_from_text(&record_to_text(&original).expect("serialize")).expect("deserialize");
        assert_eq!(restored.id, original.id);
        assert_eq!(restored.content, original.content);
        assert_eq!(restored.kind, original.kind);
        assert_eq!(restored.scope, original.scope);
        assert_eq!(restored.metadata, original.metadata);
        assert_eq!(restored.embedding, None);
        let value: Value = serde_json::from_str(&record_to_text(&original).expect("serialize"))
            .expect("valid JSON");
        assert_eq!(value["user_id"], json!("alice"));
        assert_eq!(value["scope_id"], json!("ws"));
        assert_eq!(value["session_id"], json!("s1"));
        assert_eq!(value["fields"], json!({"source": "conversation"}));
    }

    #[test]
    fn record_text_tolerates_unknown_fields() {
        let mut value: Value =
            serde_json::from_str(&record_to_text(&record()).expect("serialize")).expect("JSON");
        value["an_extra_field"] = json!({"nested": true});
        let restored = record_from_text(&value.to_string()).expect("tolerant parse");
        assert_eq!(restored.content, "Alice prefers dark mode");
        assert_eq!(restored.scope.user_id, "alice");
    }

    #[test]
    fn malformed_record_json_is_a_validation_error() {
        let error = record_from_text("not json").expect_err("malformed record");
        assert!(matches!(error, Error::Validation(_)), "{error:?}");
        let error = record_from_text(r#"{"id": 42}"#).expect_err("wrong shape");
        assert!(matches!(error, Error::Validation(_)), "{error:?}");
    }

    #[test]
    fn plain_text_returns_text_content_verbatim() {
        assert_eq!(
            plain_text(&MessageContent::Text("hello".to_owned())),
            "hello"
        );
        assert_eq!(plain_text(&MessageContent::Text(String::new())), "");
    }

    #[test]
    fn plain_text_joins_text_and_json_parts_with_a_space() {
        let part = ContentPart::ImageUrl {
            image_url: ImageUrl {
                url: "https://example.test/cat.png".to_owned(),
                detail: None,
            },
        };
        let content = MessageContent::Parts(vec![
            ContentPart::Text {
                text: "describe".to_owned(),
            },
            part.clone(),
        ]);
        let rendered = plain_text(&content);
        assert_eq!(rendered, format!("describe {}", part_json(&part)));
        assert!(rendered.contains("\"type\":\"image_url\""));
        assert!(rendered.contains("https://example.test/cat.png"));
    }

    #[test]
    fn plain_text_of_empty_parts_is_empty() {
        assert_eq!(plain_text(&MessageContent::Parts(Vec::new())), "");
        assert_eq!(plain_text(&MessageContent::default()), "");
    }
}
