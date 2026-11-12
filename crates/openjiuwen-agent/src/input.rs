//! Parsing of agent invocation payloads.
//!
//! The caller passes a JSON value; [`AgentInput::parse`] normalizes it into a
//! text query or an interactive (resume) payload, extracts the reserved
//! fields, and keeps every other key as a template variable in `extra`.
use openjiuwen_common::error::{Error, Result};
use openjiuwen_session::InteractiveInput;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
/// The question an agent is asked.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum InputQuery {
    /// A plain text question.
    Text(String),
    /// Answers to a previous interrupt, resuming the run.
    Interactive(InteractiveInput),
}
impl Default for InputQuery {
    fn default() -> Self {
        Self::Text(String::new())
    }
}
/// Parsed agent invocation input.
///
/// Reserved keys (`query`, `conversation_id`, `parent_session_id`, `user_id`,
/// `run_kind`, `_host_max_iterations`) are consumed; every other object key is
/// copied into `extra` and can be used as a prompt template variable.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct AgentInput {
    /// The question or resume payload.
    pub query: InputQuery,
    /// Conversation id; selects the session when no session is supplied.
    pub conversation_id: Option<String>,
    /// Parent session id for child sessions.
    pub parent_session_id: Option<String>,
    /// End-user id.
    pub user_id: Option<String>,
    /// Host-defined run kind.
    pub run_kind: Option<String>,
    /// Host-imposed iteration limit (Python `_host_max_iterations`).
    pub max_iterations: Option<usize>,
    /// Unreserved keys, usable as prompt template variables.
    pub extra: BTreeMap<String, Value>,
}
/// Object keys consumed by [`AgentInput::parse`].
const RESERVED_KEYS: [&str; 6] = [
    "query",
    "conversation_id",
    "parent_session_id",
    "user_id",
    "run_kind",
    "_host_max_iterations",
];
impl AgentInput {
    /// Parses the caller's JSON payload.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] when a string or object payload is
    /// required but a different JSON type is given, when the object has no
    /// `query`, or when an interactive query is malformed.
    pub fn parse(inputs: &Value) -> Result<Self> {
        match inputs {
            Value::String(text) => Ok(Self {
                query: InputQuery::Text(text.clone()),
                ..Self::default()
            }),
            Value::Object(map) => {
                let query = match map.get("query") {
                    Some(Value::String(text)) => InputQuery::Text(text.clone()),
                    Some(value @ Value::Object(_)) => serde_json::from_value(value.clone())
                        .map(InputQuery::Interactive)
                        .map_err(|error| Error::validation(error.to_string()))?,
                    _ => return Err(Error::validation("Input must contain 'query'")),
                };
                let extra = map
                    .iter()
                    .filter(|(key, _)| !RESERVED_KEYS.contains(&key.as_str()))
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect();
                Ok(Self {
                    query,
                    conversation_id: string_field(map, "conversation_id"),
                    parent_session_id: string_field(map, "parent_session_id"),
                    user_id: string_field(map, "user_id"),
                    run_kind: string_field(map, "run_kind"),
                    max_iterations: map
                        .get("_host_max_iterations")
                        .and_then(Value::as_u64)
                        .map(|value| usize::try_from(value).unwrap_or(usize::MAX)),
                    extra,
                })
            }
            _ => Err(Error::validation("Input must be dict with 'query' or str")),
        }
    }
    /// Returns the query text, or `None` for an interactive query.
    #[must_use]
    pub fn query_text(&self) -> Option<&str> {
        match &self.query {
            InputQuery::Text(text) => Some(text),
            InputQuery::Interactive(_) => None,
        }
    }
    /// Returns the interactive payload, or `None` for a text query.
    #[must_use]
    pub fn interactive(&self) -> Option<&InteractiveInput> {
        match &self.query {
            InputQuery::Text(_) => None,
            InputQuery::Interactive(input) => Some(input),
        }
    }
    /// Builds a text-query input.
    #[must_use]
    pub fn from_text(text: impl Into<String>) -> Self {
        Self {
            query: InputQuery::Text(text.into()),
            ..Self::default()
        }
    }
}
/// Reads `key` as an owned string when it holds a JSON string.
fn string_field(map: &serde_json::Map<String, Value>, key: &str) -> Option<String> {
    map.get(key).and_then(Value::as_str).map(str::to_owned)
}
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn plain_string_becomes_text_query() {
        let input = AgentInput::parse(&json!("hello")).unwrap();
        assert_eq!(input.query_text(), Some("hello"));
        assert!(input.interactive().is_none());
        assert!(input.extra.is_empty());
        assert!(input.max_iterations.is_none());
    }
    #[test]
    fn object_with_query_and_extra_keys() {
        let input = AgentInput::parse(&json!({
            "query": "hi",
            "conversation_id": "c1",
            "parent_session_id": "p1",
            "user_id": "u1",
            "run_kind": "chat",
            "_host_max_iterations": 7,
            "city": "Paris",
            "count": 2
        }))
        .unwrap();
        assert_eq!(input.query_text(), Some("hi"));
        assert_eq!(input.conversation_id.as_deref(), Some("c1"));
        assert_eq!(input.parent_session_id.as_deref(), Some("p1"));
        assert_eq!(input.user_id.as_deref(), Some("u1"));
        assert_eq!(input.run_kind.as_deref(), Some("chat"));
        assert_eq!(input.max_iterations, Some(7));
        assert_eq!(input.extra.len(), 2);
        assert_eq!(input.extra["city"], json!("Paris"));
        assert_eq!(input.extra["count"], json!(2));
    }
    #[test]
    fn non_numeric_max_iterations_is_ignored() {
        let input =
            AgentInput::parse(&json!({"query": "hi", "_host_max_iterations": "9"})).unwrap();
        assert!(input.max_iterations.is_none());
    }
    #[test]
    fn object_with_interactive_query() {
        let input =
            AgentInput::parse(&json!({"query": {"user_inputs": {"node": {"x": 1}}}})).unwrap();
        let interactive = input.interactive().unwrap();
        assert_eq!(interactive.user_inputs["node"], json!({"x": 1}));
        assert!(input.query_text().is_none());
    }
    #[test]
    fn missing_query_is_an_error() {
        let error = AgentInput::parse(&json!({"conversation_id": "c1"})).unwrap_err();
        assert!(error.to_string().contains("Input must contain 'query'"));
    }
    #[test]
    fn wrong_query_type_is_an_error() {
        assert!(AgentInput::parse(&json!({"query": 12})).is_err());
    }
    #[test]
    fn wrong_top_level_type_is_an_error() {
        assert!(AgentInput::parse(&json!([1, 2, 3])).is_err());
        assert!(AgentInput::parse(&json!(null)).is_err());
        assert!(AgentInput::parse(&json!(4)).is_err());
    }
    #[test]
    fn from_text_builds_a_text_query() {
        let input = AgentInput::from_text("question");
        assert_eq!(input.query_text(), Some("question"));
        assert_eq!(input, AgentInput::parse(&json!("question")).unwrap());
    }
    #[test]
    fn default_query_is_empty_text() {
        assert_eq!(AgentInput::default().query, InputQuery::Text(String::new()));
    }
}
