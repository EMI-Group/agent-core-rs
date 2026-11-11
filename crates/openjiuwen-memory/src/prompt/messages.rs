//! Recall results as `BaseMessage`s and as the Python `MemResult` JSON array.
//!
//! Messages carry the same body the string renderers produce, so a caller that
//! appends a system message and a caller that interpolates the
//! [`MEMORY_PLACEHOLDER`](super::MEMORY_PLACEHOLDER) template slot see identical text.
use super::render::render_snippets;
use crate::record::RecallHit;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::prompt::BaseMessage;
use serde_json::{Map, Value};
use std::collections::BTreeMap;

/// Collapses every whitespace run in `value` to a single space and trims it.
fn one_line(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Renders memory hits as one system message, or `None` when nothing fits.
#[must_use]
pub fn hits_to_system_message(hits: &[RecallHit], max_tokens: usize) -> Option<BaseMessage> {
    let body = render_snippets(hits, max_tokens);
    if body.is_empty() {
        return None;
    }
    Some(BaseMessage::system(body))
}

/// Renders memory hits as a message list of zero or one system message.
#[must_use]
pub fn hits_to_messages(hits: &[RecallHit], max_tokens: usize) -> Vec<BaseMessage> {
    hits_to_system_message(hits, max_tokens)
        .into_iter()
        .collect()
}

/// Renders memory variables as one system message, or `None` for an empty map.
///
/// The body is a `## Memory variables` header followed by `key: value` lines in
/// `BTreeMap` order; each value is collapsed to a single line.
#[must_use]
pub fn variables_to_system_message(variables: &BTreeMap<String, String>) -> Option<BaseMessage> {
    if variables.is_empty() {
        return None;
    }
    let mut body = String::from("## Memory variables");
    for (name, value) in variables {
        body.push('\n');
        body.push_str(name);
        body.push_str(": ");
        body.push_str(&one_line(value));
    }
    Some(BaseMessage::system(body))
}

/// Projects hits onto the Python `MemResult` JSON shape.
///
/// Each element is `{"id", "mem", "mem_type", "timestamp", "score"}` with the
/// timestamp in RFC 3339 and the input order preserved.
#[must_use]
pub fn to_memory_values(hits: &[RecallHit]) -> Vec<Value> {
    hits.iter()
        .map(|hit| {
            let mut object = Map::new();
            object.insert("id".to_owned(), Value::String(hit.record.id.clone()));
            object.insert("mem".to_owned(), Value::String(hit.record.content.clone()));
            object.insert(
                "mem_type".to_owned(),
                Value::String(hit.record.kind.as_str().to_owned()),
            );
            object.insert(
                "timestamp".to_owned(),
                Value::String(hit.record.timestamp.to_rfc3339()),
            );
            object.insert(
                "score".to_owned(),
                serde_json::Number::from_f64(f64::from(hit.score))
                    .map_or(Value::Null, Value::Number),
            );
            Value::Object(object)
        })
        .collect()
}

/// Serializes hits as the Python `sys_long_term_memory` JSON array.
///
/// The values are exactly [`to_memory_values`]; serialization of these types
/// cannot fail, and the error is mapped to [`Error::Validation`] for that reason.
pub fn to_memory_json(hits: &[RecallHit]) -> Result<String> {
    serde_json::to_string(&to_memory_values(hits))
        .map_err(|error| Error::validation(format!("failed to serialize memory results: {error}")))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::record::{MemoryRecord, MemoryScope, MemoryType};
    use openjiuwen_foundation::prompt::Role;
    use serde_json::json;

    fn hit(id: &str, content: &str, kind: MemoryType, score: f32) -> RecallHit {
        RecallHit {
            record: MemoryRecord::new(MemoryScope::new("alice", "ws"), content, kind).with_id(id),
            score,
            similarity: score,
            keyword: 0.0,
            importance: 0.5,
            recency: 0.5,
        }
    }

    fn sample_hits() -> Vec<RecallHit> {
        vec![
            hit(
                "id-1",
                "Alice prefers dark mode",
                MemoryType::SemanticMemory,
                0.87,
            ),
            hit(
                "id-2",
                "Alice lives in Berlin",
                MemoryType::UserProfile,
                0.42,
            ),
        ]
    }

    #[test]
    fn empty_hits_yield_no_message() {
        assert!(hits_to_system_message(&[], 100).is_none());
        assert_eq!(hits_to_messages(&[], 100), Vec::<BaseMessage>::new());
        // A non-empty input that cannot fit the budget is also empty.
        assert!(hits_to_system_message(&sample_hits(), 0).is_none());
        assert_eq!(
            hits_to_messages(&sample_hits(), 0),
            Vec::<BaseMessage>::new()
        );
    }

    #[test]
    fn system_message_body_matches_render_snippets() {
        let hits = sample_hits();
        let body = render_snippets(&hits, 1000);
        let message = hits_to_system_message(&hits, 1000).unwrap();
        assert_eq!(message.role, Role::System);
        assert_eq!(&*message.text(), body.as_str());
        assert!(body.starts_with("## Long-term memory\n"));
    }

    #[test]
    fn messages_hold_zero_or_one_system_message() {
        let hits = sample_hits();
        let messages = hits_to_messages(&hits, 1000);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0], hits_to_system_message(&hits, 1000).unwrap());
    }

    #[test]
    fn variables_render_in_key_order() {
        let variables = BTreeMap::from([
            ("theme".to_owned(), "dark  mode".to_owned()),
            ("city".to_owned(), " Berlin \n tiergarten ".to_owned()),
        ]);
        let message = variables_to_system_message(&variables).unwrap();
        assert_eq!(message.role, Role::System);
        assert_eq!(
            &*message.text(),
            "## Memory variables\ncity: Berlin tiergarten\ntheme: dark mode"
        );
    }

    #[test]
    fn empty_variables_yield_no_message() {
        assert!(variables_to_system_message(&BTreeMap::new()).is_none());
    }

    #[test]
    fn memory_values_use_python_field_names_and_order() {
        let hits = sample_hits();
        let values = to_memory_values(&hits);
        assert_eq!(values.len(), 2);

        let first = values[0].as_object().unwrap();
        let mut keys: Vec<&str> = first.keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(keys, vec!["id", "mem", "mem_type", "score", "timestamp"]);

        assert_eq!(values[0]["id"], json!("id-1"));
        assert_eq!(values[0]["mem"], json!("Alice prefers dark mode"));
        assert_eq!(values[0]["mem_type"], json!("semantic_memory"));
        assert_eq!(values[0]["score"], json!(0.87_f32));
        assert_eq!(
            values[0]["timestamp"],
            json!(hits[0].record.timestamp.to_rfc3339())
        );
        assert_eq!(values[1]["id"], json!("id-2"));
        assert_eq!(values[1]["mem_type"], json!("user_profile"));

        assert_eq!(to_memory_values(&[]), Vec::<Value>::new());
    }

    #[test]
    fn memory_json_round_trips_and_preserves_order() {
        let hits = sample_hits();
        let json_text = to_memory_json(&hits).unwrap();
        let parsed: Value = serde_json::from_str(&json_text).unwrap();
        assert_eq!(parsed, Value::Array(to_memory_values(&hits)));
        let array = parsed.as_array().unwrap();
        assert_eq!(array[0]["id"], json!("id-1"));
        assert_eq!(array[1]["id"], json!("id-2"));

        assert_eq!(to_memory_json(&[]).unwrap(), "[]");
    }
}
