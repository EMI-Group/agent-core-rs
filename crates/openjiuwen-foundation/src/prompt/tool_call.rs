//! Tool calls, streaming fragments, and delta accumulation.
//!
//! Ports `openjiuwen/core/foundation/llm/schema/tool_call.py` (`ToolCall`) and
//! the tool-call half of `AssistantMessageChunk.__add__` from
//! `openjiuwen/core/foundation/llm/schema/message_chunk.py`.
//!
//! Providers deliver a tool call either as one complete object (non-streaming
//! responses) or as a series of fragments whose `arguments` are plain string
//! concatenations (streaming responses). [`ToolCallAccumulator`] implements the
//! fragment merge rule in one place so every client assembles calls identically.
//!
//! ```
//! use openjiuwen_foundation::prompt::tool_call::{ToolCall, ToolCallDelta, ToolCallAccumulator};
//!
//! let mut accumulator = ToolCallAccumulator::new();
//! accumulator.push_delta(ToolCallDelta {
//!     index: Some(0),
//!     id: Some("call_1".to_owned()),
//!     name: Some("get_weather".to_owned()),
//!     arguments: Some("{\"city\":".to_owned()),
//! });
//! accumulator.push_delta(ToolCallDelta {
//!     arguments: Some("\"Paris\"}".to_owned()),
//!     ..ToolCallDelta::default()
//! });
//!
//! let calls = accumulator.into_calls();
//! assert_eq!(calls.len(), 1);
//! assert_eq!(calls[0].arguments, "{\"city\":\"Paris\"}");
//! ```

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// The only tool-call kind the built-in providers emit.
const FUNCTION_KIND: &str = "function";

/// The default `type` field of a [`ToolCall`].
fn default_kind() -> String {
    FUNCTION_KIND.to_owned()
}

/// One model-requested tool invocation.
///
/// `arguments` is the raw JSON text exactly as the provider sent it (possibly
/// incomplete while streaming); use [`ToolCall::arguments_value`] to parse it.
/// The `type` field keeps its wire name and defaults to `"function"` when a
/// payload omits it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ToolCall {
    /// Provider-assigned call id, absent on the leading fragments of a stream.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Wire kind; always `"function"` for the built-in providers.
    #[serde(rename = "type", default = "default_kind")]
    pub kind: String,
    /// Name of the tool the model wants to call.
    pub name: String,
    /// JSON text of the arguments, accumulated verbatim across stream fragments.
    pub arguments: String,
    /// Position of the call in the model's `tool_calls` array, when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<u32>,
}

impl Default for ToolCall {
    fn default() -> Self {
        Self {
            id: None,
            kind: default_kind(),
            name: String::new(),
            arguments: String::new(),
            index: None,
        }
    }
}

impl ToolCall {
    /// Creates a `"function"` call with no id, no index, and empty arguments.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Self::default()
        }
    }

    /// Sets the provider call id.
    #[must_use]
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Sets the arguments from a JSON value, serialized compactly.
    ///
    /// Serialization of a [`Value`] cannot fail; the unreachable failure falls
    /// back to the value's own JSON text so this never panics.
    #[must_use]
    pub fn with_arguments(mut self, value: Value) -> Self {
        self.arguments = serde_json::to_string(&value).unwrap_or_else(|_| value.to_string());
        self
    }

    /// Sets the position of this call in the model's `tool_calls` array.
    #[must_use]
    pub fn with_index(mut self, index: u32) -> Self {
        self.index = Some(index);
        self
    }

    /// Parses [`ToolCall::arguments`] as JSON.
    ///
    /// Empty or whitespace-only arguments mean "no arguments" and yield
    /// [`Value::Null`]; malformed JSON is a validation error naming the tool.
    pub fn arguments_value(&self) -> Result<Value> {
        let trimmed = self.arguments.trim();
        if trimmed.is_empty() {
            return Ok(Value::Null);
        }
        serde_json::from_str(trimmed).map_err(|error| {
            Error::validation(format!(
                "tool call `{}` has invalid JSON arguments: {error}",
                self.name
            ))
        })
    }

    /// Renders the call in the `OpenAI` `tool_calls` wire shape.
    ///
    /// The result is `{"id", "type", "function": {"name", "arguments"}}`, with
    /// `id` omitted when absent. `index` stays out: the `OpenAI` message format
    /// carries positions on the enclosing array, not inside each call.
    #[must_use]
    pub fn to_openai(&self) -> Value {
        let kind = if self.kind.is_empty() {
            FUNCTION_KIND
        } else {
            self.kind.as_str()
        };
        let mut function = Map::new();
        function.insert("name".to_owned(), Value::String(self.name.clone()));
        function.insert(
            "arguments".to_owned(),
            Value::String(self.arguments.clone()),
        );

        let mut object = Map::new();
        if let Some(id) = &self.id {
            object.insert("id".to_owned(), Value::String(id.clone()));
        }
        object.insert("type".to_owned(), Value::String(kind.to_owned()));
        object.insert("function".to_owned(), Value::Object(function));
        Value::Object(object)
    }

    /// Parses one `OpenAI` `tool_calls` entry, nested or flat.
    ///
    /// The nested shape (`{"id", "type", "function": {"name", "arguments"}}`) is
    /// what providers send; a flat object with `name`/`arguments` at the top
    /// level is accepted as well. `arguments` may be a JSON string or an inline
    /// JSON object/array, which is stringified compactly. `index` is read from
    /// the top level when present.
    pub fn from_openai(value: &Value) -> Result<Self> {
        Self::from_openai_impl(value, None)
    }

    /// Like [`ToolCall::from_openai`] but forces `index`, typically the array
    /// position during stream parsing.
    pub fn from_openai_with_index(value: &Value, index: u32) -> Result<Self> {
        Self::from_openai_impl(value, Some(index))
    }

    fn from_openai_impl(value: &Value, forced_index: Option<u32>) -> Result<Self> {
        let object = value.as_object().ok_or_else(|| {
            Error::validation(format!("tool call must be a JSON object, got: {value}"))
        })?;

        let nested = match object.get("function") {
            Some(function) => Some(function.as_object().ok_or_else(|| {
                Error::validation("tool call `function` field must be a JSON object")
            })?),
            None => None,
        };

        let (name, arguments) = match nested {
            Some(function) => (
                required_name(function.get("name"), "function.name")?,
                stringify_arguments(function.get("arguments"))?,
            ),
            None => (
                required_name(object.get("name"), "name")?,
                stringify_arguments(object.get("arguments"))?,
            ),
        };

        let id = object
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
            .map(str::to_owned);
        let kind = object
            .get("type")
            .and_then(Value::as_str)
            .filter(|kind| !kind.is_empty())
            .map_or_else(default_kind, str::to_owned);
        let index = forced_index.or_else(|| {
            object
                .get("index")
                .and_then(Value::as_u64)
                .and_then(|raw| u32::try_from(raw).ok())
        });

        Ok(Self {
            id,
            kind,
            name,
            arguments,
            index,
        })
    }
}

/// One incremental fragment of a [`ToolCall`] from a streaming response.
///
/// Every field is optional: providers split the id, name, and arguments across
/// chunks in vendor-specific ways, and [`ToolCallAccumulator`] stitches them
/// back together.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ToolCallDelta {
    /// Position of the call in the model's `tool_calls` array.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<u32>,
    /// Provider call id, usually sent only on the first fragment.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Tool name, usually sent only on the first fragment.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Arguments text appended to the previously accumulated arguments.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

/// Accumulates [`ToolCallDelta`] fragments and whole [`ToolCall`]s into calls.
///
/// Fragment merging mirrors Python `AssistantMessageChunk.__add__`: a fragment
/// joins the previous call when their ids are compatible (equal non-empty ids,
/// or either side without an id) and both are `"function"` calls; `arguments`
/// then concatenate. Any other fragment starts a new call, whose `index`
/// defaults to its position in the accumulated list.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ToolCallAccumulator {
    calls: Vec<ToolCall>,
}

impl ToolCallAccumulator {
    /// Creates an empty accumulator.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns `true` when no call has been accumulated.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.calls.is_empty()
    }

    /// The calls accumulated so far.
    #[must_use]
    pub fn calls(&self) -> &[ToolCall] {
        &self.calls
    }

    /// Merges one streaming fragment.
    pub fn push_delta(&mut self, delta: ToolCallDelta) {
        let incoming = ToolCall {
            id: delta.id,
            kind: default_kind(),
            name: delta.name.unwrap_or_default(),
            arguments: delta.arguments.unwrap_or_default(),
            index: delta.index,
        };
        self.merge(incoming);
    }

    /// Merges one complete call as a single fragment.
    ///
    /// A whole call goes through the same merge rule as a fragment, so feeding
    /// a sequence of non-streaming responses accumulates a conversation's calls.
    pub fn push_call(&mut self, call: ToolCall) {
        self.merge(call);
    }

    /// Consumes the accumulator and returns the assembled calls.
    #[must_use]
    pub fn into_calls(self) -> Vec<ToolCall> {
        self.calls
    }

    fn merge(&mut self, incoming: ToolCall) {
        let position = self.calls.len();
        if let Some(last) = self.calls.last_mut() {
            let same_id = (last.id.is_some() && incoming.id.is_some() && last.id == incoming.id)
                || last.id.is_none()
                || incoming.id.is_none();
            if same_id && last.kind == FUNCTION_KIND && incoming.kind == FUNCTION_KIND {
                let id = last.id.take().or(incoming.id);
                let kind = if last.kind.is_empty() {
                    incoming.kind
                } else {
                    std::mem::take(&mut last.kind)
                };
                let name = if last.name.is_empty() {
                    incoming.name
                } else {
                    std::mem::take(&mut last.name)
                };
                let arguments = std::mem::take(&mut last.arguments) + &incoming.arguments;
                let index = last.index.or(incoming.index);
                *last = ToolCall {
                    id,
                    kind,
                    name,
                    arguments,
                    index,
                };
                return;
            }
        }

        let mut call = incoming;
        if call.index.is_none() {
            call.index = Some(u32::try_from(position).unwrap_or(u32::MAX));
        }
        self.calls.push(call);
    }
}

/// Reads a required, non-empty name string.
fn required_name(value: Option<&Value>, location: &str) -> Result<String> {
    value
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| Error::validation(format!("tool call is missing `{location}`")))
}

/// Normalizes a wire `arguments` field into JSON text.
fn stringify_arguments(value: Option<&Value>) -> Result<String> {
    match value {
        None | Some(Value::Null) => Ok(String::new()),
        Some(Value::String(text)) => Ok(text.clone()),
        Some(other) => serde_json::to_string(other)
            .map_err(|error| Error::validation(format!("invalid tool call arguments: {error}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;

    fn delta(
        index: Option<u32>,
        id: Option<&str>,
        name: Option<&str>,
        arguments: &str,
    ) -> ToolCallDelta {
        ToolCallDelta {
            index,
            id: id.map(str::to_owned),
            name: name.map(str::to_owned),
            arguments: Some(arguments.to_owned()),
        }
    }

    #[test]
    fn id_less_fragments_merge_and_concatenate_arguments() {
        let mut accumulator = ToolCallAccumulator::new();
        assert!(accumulator.is_empty());
        accumulator.push_delta(delta(
            Some(0),
            Some("call_1"),
            Some("get_weather"),
            "{\"city\":",
        ));
        accumulator.push_delta(delta(Some(0), None, None, "\"Paris\""));
        accumulator.push_delta(delta(Some(0), None, None, "}"));

        let calls = accumulator.into_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id.as_deref(), Some("call_1"));
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments, "{\"city\":\"Paris\"}");
        assert_eq!(calls[0].kind, "function");
        assert_eq!(
            calls[0].arguments_value().unwrap(),
            json!({"city": "Paris"})
        );
    }

    #[test]
    fn matching_non_empty_ids_merge() {
        let mut accumulator = ToolCallAccumulator::new();
        accumulator.push_delta(delta(Some(0), Some("call_1"), Some("f"), "{\"a\":"));
        accumulator.push_delta(delta(Some(0), Some("call_1"), Some("f"), "1}"));

        let calls = accumulator.into_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments, "{\"a\":1}");
    }

    #[test]
    fn differing_non_empty_ids_append_a_new_call() {
        let mut accumulator = ToolCallAccumulator::new();
        accumulator.push_delta(delta(Some(0), Some("call_1"), Some("f"), "{}"));
        accumulator.push_delta(delta(Some(1), Some("call_2"), Some("g"), "{}"));

        let calls = accumulator.into_calls();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].id.as_deref(), Some("call_1"));
        assert_eq!(calls[1].id.as_deref(), Some("call_2"));
        assert_eq!(calls[1].index, Some(1));
    }

    #[test]
    fn fragment_with_id_after_id_less_call_merges() {
        let mut accumulator = ToolCallAccumulator::new();
        accumulator.push_delta(delta(None, None, Some("lookup"), "{\"q\":"));
        accumulator.push_delta(delta(None, Some("call_9"), None, "{"));
        accumulator.push_delta(delta(None, None, None, "}"));

        let calls = accumulator.into_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id.as_deref(), Some("call_9"));
        assert_eq!(calls[0].name, "lookup");
        assert_eq!(calls[0].arguments, "{\"q\":{}");
    }

    #[test]
    fn name_and_index_keep_the_first_non_empty_value() {
        let mut accumulator = ToolCallAccumulator::new();
        accumulator.push_delta(delta(Some(4), Some("call_1"), Some("first_name"), "{}"));
        accumulator.push_delta(delta(Some(7), Some("call_1"), Some("second_name"), "{}"));

        let calls = accumulator.into_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "first_name");
        assert_eq!(calls[0].index, Some(4));
    }

    #[test]
    fn non_function_kinds_are_appended_never_merged() {
        let mut accumulator = ToolCallAccumulator::new();
        accumulator.push_call(ToolCall {
            id: Some("call_1".to_owned()),
            kind: "custom".to_owned(),
            name: "f".to_owned(),
            arguments: "{}".to_owned(),
            index: None,
        });
        accumulator.push_call(ToolCall {
            id: Some("call_1".to_owned()),
            kind: "custom".to_owned(),
            name: "f".to_owned(),
            arguments: "{}".to_owned(),
            index: None,
        });

        let calls = accumulator.into_calls();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].kind, "custom");
        assert_eq!(calls[1].index, Some(1));
    }

    #[test]
    fn whole_calls_and_deltas_can_be_mixed() {
        let mut accumulator = ToolCallAccumulator::new();
        accumulator.push_call(ToolCall::new("f").with_id("call_1").with_index(0));
        accumulator.push_delta(delta(None, Some("call_1"), None, "{\"x\":1}"));

        let calls = accumulator.into_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments, "{\"x\":1}");
    }

    #[test]
    fn calls_accessor_exposes_the_accumulated_calls() {
        let mut accumulator = ToolCallAccumulator::new();
        accumulator.push_delta(delta(None, None, Some("f"), "{}"));
        assert_eq!(accumulator.calls().len(), 1);
        assert!(!accumulator.is_empty());
    }

    #[test]
    fn from_openai_reads_the_nested_shape() {
        let value = json!({
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"},
            "index": 3
        });
        let call = ToolCall::from_openai(&value).unwrap();
        assert_eq!(call.id.as_deref(), Some("call_1"));
        assert_eq!(call.kind, "function");
        assert_eq!(call.name, "get_weather");
        assert_eq!(call.arguments, "{\"city\":\"Paris\"}");
        assert_eq!(call.index, Some(3));
    }

    #[test]
    fn from_openai_reads_the_flat_shape_and_unknown_type_defaults() {
        let call = ToolCall::from_openai(&json!({
            "id": "call_2",
            "name": "lookup",
            "arguments": {"q": "rust"}
        }))
        .unwrap();
        assert_eq!(call.id.as_deref(), Some("call_2"));
        assert_eq!(call.kind, "function");
        assert_eq!(call.name, "lookup");
        assert_eq!(call.arguments, "{\"q\":\"rust\"}");
        assert_eq!(call.index, None);
    }

    #[test]
    fn from_openai_inlines_json_object_arguments_compactly() {
        let call = ToolCall::from_openai(&json!({
            "type": "function",
            "function": {"name": "f", "arguments": {"nested": [1, 2]}}
        }))
        .unwrap();
        assert_eq!(call.arguments, "{\"nested\":[1,2]}");
        assert_eq!(call.arguments_value().unwrap(), json!({"nested": [1, 2]}));
    }

    #[test]
    fn from_openai_with_index_forces_the_position() {
        let value = json!({"function": {"name": "f", "arguments": "{}"}, "index": 1});
        let call = ToolCall::from_openai_with_index(&value, 7).unwrap();
        assert_eq!(call.index, Some(7));
    }

    #[test]
    fn from_openai_reports_missing_names_and_bad_shapes() {
        let missing_name = ToolCall::from_openai(&json!({"function": {"arguments": "{}"}}))
            .expect_err("missing name must fail");
        assert_eq!(missing_name.kind(), ErrorKind::Validation);
        assert!(missing_name.to_string().contains("function.name"));

        let missing_flat_name =
            ToolCall::from_openai(&json!({"arguments": "{}"})).expect_err("missing name must fail");
        assert_eq!(missing_flat_name.kind(), ErrorKind::Validation);

        let not_an_object =
            ToolCall::from_openai(&json!("nope")).expect_err("non-object must fail");
        assert_eq!(not_an_object.kind(), ErrorKind::Validation);

        let bad_function = ToolCall::from_openai(&json!({"function": "nope"}))
            .expect_err("non-object function must fail");
        assert_eq!(bad_function.kind(), ErrorKind::Validation);
    }

    #[test]
    fn to_openai_matches_the_expected_wire_shape() {
        let call = ToolCall::new("get_weather")
            .with_id("call_1")
            .with_arguments(json!({"city": "Paris"}))
            .with_index(2);
        assert_eq!(
            call.to_openai(),
            json!({
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}
            })
        );

        let anonymous = ToolCall::new("noop");
        assert_eq!(
            anonymous.to_openai(),
            json!({
                "type": "function",
                "function": {"name": "noop", "arguments": ""}
            })
        );
    }

    #[test]
    fn arguments_value_handles_empty_and_malformed_text() {
        let mut call = ToolCall::new("f");
        assert_eq!(call.arguments_value().unwrap(), Value::Null);

        call.arguments = "   ".to_owned();
        assert_eq!(call.arguments_value().unwrap(), Value::Null);

        call.arguments = "{not json".to_owned();
        let error = call
            .arguments_value()
            .expect_err("malformed JSON must fail");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("`f`"));
    }

    #[test]
    fn tool_call_serde_uses_wire_names_and_defaults() {
        let call = ToolCall::new("f").with_index(2);
        assert_eq!(
            serde_json::to_value(&call).unwrap(),
            json!({"type": "function", "name": "f", "arguments": "", "index": 2})
        );

        let parsed: ToolCall =
            serde_json::from_value(json!({"name": "g", "arguments": "{\"x\":1}"})).unwrap();
        assert_eq!(parsed.id, None);
        assert_eq!(parsed.kind, "function");
        assert_eq!(parsed.index, None);

        let tolerant: ToolCall = serde_json::from_value(json!({
            "id": "i",
            "type": "function",
            "name": "g",
            "arguments": "{}",
            "index": 1,
            "response_item_id": "ignored"
        }))
        .unwrap();
        assert_eq!(
            tolerant,
            ToolCall::new("g")
                .with_id("i")
                .with_index(1)
                .with_arguments(json!({}))
        );

        let round_trip: ToolCall =
            serde_json::from_value(serde_json::to_value(&call).unwrap()).unwrap();
        assert_eq!(round_trip, call);
    }

    #[test]
    fn tool_call_delta_serde_round_trips_and_skips_absent_fields() {
        assert_eq!(
            serde_json::to_value(ToolCallDelta::default()).unwrap(),
            json!({})
        );

        let fragment = ToolCallDelta {
            index: Some(0),
            id: Some("call_1".to_owned()),
            name: None,
            arguments: Some("{\"a\":".to_owned()),
        };
        let value = serde_json::to_value(&fragment).unwrap();
        assert_eq!(
            value,
            json!({"index": 0, "id": "call_1", "arguments": "{\"a\":"})
        );
        assert_eq!(
            serde_json::from_value::<ToolCallDelta>(value).unwrap(),
            fragment
        );

        let empty: ToolCallDelta = serde_json::from_value(json!({})).unwrap();
        assert_eq!(empty, ToolCallDelta::default());
    }
}
