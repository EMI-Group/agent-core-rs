//! Scan stages, guardrail event names, and the [`GuardrailContext`] payload.
//!
//! Ports `openjiuwen/core/security/guardrail/context.py`: Python passes the
//! callback event plus positional arguments to every guardrail, here the
//! pipeline builds a typed [`GuardrailContext`] instead.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fmt;

use openjiuwen_common::security::REDACTED;
use openjiuwen_foundation::prompt::BaseMessage;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;

/// Event name of the model-input callback (`_framework:llm_invoke_input`).
pub const EVENT_LLM_INVOKE_INPUT: &str = "_framework:llm_invoke_input";
/// Event name of the model-output callback (`_framework:llm_invoke_output`).
pub const EVENT_LLM_INVOKE_OUTPUT: &str = "_framework:llm_invoke_output";
/// Event name of the tool-input callback (`_framework:tool_invoke_input`).
pub const EVENT_TOOL_INVOKE_INPUT: &str = "_framework:tool_invoke_input";
/// Event name of the tool-output callback (`_framework:tool_invoke_output`).
pub const EVENT_TOOL_INVOKE_OUTPUT: &str = "_framework:tool_invoke_output";
/// Event name used when scanning direct user input (`user_input`).
pub const EVENT_USER_INPUT: &str = "user_input";

/// The point in a call at which guardrails run.
///
/// The known variants carry Python's exact callback event strings, so a runner
/// callback adapter can bind by string without a translation table;
/// [`ScanStage::Other`] accepts event names this crate does not know.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ScanStage {
    /// The prompt sent to the model.
    LlmInvokeInput,
    /// The model's response.
    LlmInvokeOutput,
    /// The arguments of a tool call.
    ToolInvokeInput,
    /// The result a tool produced.
    ToolInvokeOutput,
    /// Raw user input, before it reaches a model.
    UserInput,
    /// An event name this crate does not model.
    Other(String),
}

impl ScanStage {
    /// The callback event name of the stage.
    #[must_use]
    pub fn event(&self) -> &str {
        match self {
            Self::LlmInvokeInput => EVENT_LLM_INVOKE_INPUT,
            Self::LlmInvokeOutput => EVENT_LLM_INVOKE_OUTPUT,
            Self::ToolInvokeInput => EVENT_TOOL_INVOKE_INPUT,
            Self::ToolInvokeOutput => EVENT_TOOL_INVOKE_OUTPUT,
            Self::UserInput => EVENT_USER_INPUT,
            Self::Other(event) => event.as_str(),
        }
    }

    /// Maps a callback event name back to a stage; unknown names become
    /// [`Self::Other`].
    #[must_use]
    pub fn from_event(event: &str) -> Self {
        match event {
            EVENT_LLM_INVOKE_INPUT => Self::LlmInvokeInput,
            EVENT_LLM_INVOKE_OUTPUT => Self::LlmInvokeOutput,
            EVENT_TOOL_INVOKE_INPUT => Self::ToolInvokeInput,
            EVENT_TOOL_INVOKE_OUTPUT => Self::ToolInvokeOutput,
            EVENT_USER_INPUT => Self::UserInput,
            other => Self::Other(other.to_owned()),
        }
    }

    /// The payload shape usually carried by this stage.
    #[must_use]
    pub fn content_type(&self) -> crate::GuardrailContentType {
        match self {
            Self::LlmInvokeInput | Self::LlmInvokeOutput | Self::UserInput => {
                crate::GuardrailContentType::Text
            }
            Self::ToolInvokeInput | Self::ToolInvokeOutput => crate::GuardrailContentType::ToolCall,
            Self::Other(_) => crate::GuardrailContentType::Raw,
        }
    }
}

impl fmt::Display for ScanStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.event())
    }
}

impl Serialize for ScanStage {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.event())
    }
}

impl<'de> Deserialize<'de> for ScanStage {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        match value {
            Value::String(event) => Ok(Self::from_event(&event)),
            other => Err(serde::de::Error::custom(format!(
                "expected a scan-stage event string, got {other}"
            ))),
        }
    }
}

/// Everything a guardrail may inspect for one scan.
///
/// The payload is JSON so the same value can come from a model request, a tool
/// call, or a parsed wire message. [`GuardrailContext::text_view`] projects it
/// to the string the built-in guardrails scan.
#[derive(Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct GuardrailContext {
    /// Shape of [`Self::content`].
    pub content_type: crate::GuardrailContentType,
    /// The payload itself, normally a string, a message array, or an object.
    pub content: Value,
    /// The callback event name, see [`ScanStage::event`].
    pub event: String,
    /// Extra facts about the scan (`tool_name`, `message_count`, ...).
    pub metadata: BTreeMap<String, Value>,
}

impl GuardrailContext {
    /// A plain-text context.
    ///
    /// The stage is taken by value so call sites read naturally
    /// (`ScanStage::UserInput`); it is only read for its event name.
    #[must_use]
    #[allow(clippy::needless_pass_by_value)]
    pub fn text(text: impl Into<String>, stage: ScanStage) -> Self {
        Self {
            content_type: crate::GuardrailContentType::Text,
            content: Value::String(text.into()),
            event: stage.event().to_owned(),
            metadata: BTreeMap::new(),
        }
    }

    /// A chat-message context carrying the serialized messages and their count.
    #[must_use]
    #[allow(clippy::needless_pass_by_value)]
    pub fn messages(messages: &[BaseMessage], stage: ScanStage) -> Self {
        let mut metadata = BTreeMap::new();
        metadata.insert("message_count".to_owned(), Value::from(messages.len()));
        Self {
            content_type: crate::GuardrailContentType::Messages,
            content: serde_json::to_value(messages).unwrap_or(Value::Null),
            event: stage.event().to_owned(),
            metadata,
        }
    }

    /// A tool-call context carrying the tool name and its JSON arguments.
    #[must_use]
    #[allow(clippy::needless_pass_by_value)]
    pub fn tool_call(tool_name: impl Into<String>, arguments: &Value, stage: ScanStage) -> Self {
        let mut metadata = BTreeMap::new();
        metadata.insert("tool_name".to_owned(), Value::String(tool_name.into()));
        Self {
            content_type: crate::GuardrailContentType::ToolCall,
            content: arguments.clone(),
            event: stage.event().to_owned(),
            metadata,
        }
    }

    /// A tool-result context carrying the tool name and the JSON output.
    ///
    /// Identical to [`Self::tool_call`]; callers normally pass
    /// [`ScanStage::ToolInvokeOutput`].
    #[must_use]
    pub fn tool_output(tool_name: impl Into<String>, output: &Value, stage: ScanStage) -> Self {
        Self::tool_call(tool_name, output, stage)
    }

    /// An opaque context for an arbitrary event.
    #[must_use]
    pub fn raw(event: impl Into<String>, content: Value) -> Self {
        Self {
            content_type: crate::GuardrailContentType::Raw,
            content,
            event: event.into(),
            metadata: BTreeMap::new(),
        }
    }

    /// Inserts one metadata entry; a later insert with the same key replaces the
    /// previous value.
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// The content as a string, when it is a JSON string.
    #[must_use]
    pub fn get_text(&self) -> Option<&str> {
        self.content.as_str()
    }

    /// The content as a JSON array, when it is one.
    #[must_use]
    pub fn get_messages(&self) -> Option<&[Value]> {
        self.content.as_array().map(Vec::as_slice)
    }

    /// The `tool_name` metadata entry, when it is a string.
    #[must_use]
    pub fn get_tool_name(&self) -> Option<&str> {
        self.metadata.get("tool_name").and_then(Value::as_str)
    }

    /// The text projection every built-in guardrail scans.
    ///
    /// Text content is returned borrowed. Message arrays contribute every
    /// message's text (string content, or the `"text"` parts of multimodal
    /// content) plus every tool call's argument string, joined with `\n`.
    /// Tool-call, raw, and non-string contexts fall back to compact JSON.
    #[must_use]
    pub fn text_view(&self) -> Cow<'_, str> {
        if self.content_type == crate::GuardrailContentType::Messages {
            return Cow::Owned(self.messages_text());
        }
        match self.content.as_str() {
            Some(text) => Cow::Borrowed(text),
            None => Cow::Owned(compact_json(&self.content)),
        }
    }

    /// Flattens a serialized message array into one newline-joined string.
    fn messages_text(&self) -> String {
        let Some(messages) = self.content.as_array() else {
            return compact_json(&self.content);
        };
        let mut pieces: Vec<&str> = Vec::new();
        for message in messages {
            match message.get("content") {
                Some(Value::String(text)) => pieces.push(text),
                Some(Value::Array(parts)) => {
                    for part in parts {
                        if part.get("type").and_then(Value::as_str) == Some("text") {
                            if let Some(text) = part.get("text").and_then(Value::as_str) {
                                pieces.push(text);
                            }
                        }
                    }
                }
                _ => {}
            }
            if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
                for call in calls {
                    if let Some(arguments) = call.get("arguments").and_then(Value::as_str) {
                        pieces.push(arguments);
                    }
                }
            }
        }
        pieces
            .into_iter()
            .filter(|piece| !piece.is_empty())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

impl fmt::Debug for GuardrailContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let keys: Vec<&str> = self.metadata.keys().map(String::as_str).collect();
        f.debug_struct("GuardrailContext")
            .field("content_type", &self.content_type)
            .field("event", &self.event)
            .field("metadata_keys", &keys)
            .field("content", &REDACTED)
            .finish()
    }
}

/// Compact JSON of a value; an empty string when it cannot be serialized.
fn compact_json(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn text_view_borrows_plain_text() {
        let context = GuardrailContext::text("hello", ScanStage::UserInput);
        assert!(matches!(context.text_view(), Cow::Borrowed("hello")));
        assert_eq!(context.get_text(), Some("hello"));
        assert_eq!(context.event, EVENT_USER_INPUT);
        assert_eq!(context.content_type, crate::GuardrailContentType::Text);
    }

    #[test]
    fn text_view_joins_message_text_and_tool_arguments() {
        let messages = vec![
            BaseMessage::system("be nice"),
            BaseMessage::user("hello there"),
        ];
        let context = GuardrailContext::messages(&messages, ScanStage::LlmInvokeInput);
        assert_eq!(context.text_view(), "be nice\nhello there");
        assert_eq!(
            context
                .metadata
                .get("message_count")
                .and_then(Value::as_u64),
            Some(2)
        );
        assert_eq!(context.get_messages().map(<[Value]>::len), Some(2));
    }

    #[test]
    fn text_view_handles_multimodal_parts_and_tool_calls() {
        let messages = GuardrailContext {
            content_type: crate::GuardrailContentType::Messages,
            content: json!([
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "a"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}
                    ]
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "1", "type": "function", "name": "run", "arguments": "{\"cmd\":\"ls\"}"}
                    ]
                }
            ]),
            event: EVENT_LLM_INVOKE_INPUT.to_owned(),
            metadata: BTreeMap::new(),
        };
        assert_eq!(messages.text_view(), "a\n{\"cmd\":\"ls\"}");
    }

    #[test]
    fn text_view_falls_back_to_compact_json() {
        let context = GuardrailContext::tool_call(
            "shell",
            &json!({"cmd": "rm -rf /"}),
            ScanStage::ToolInvokeInput,
        );
        assert_eq!(context.text_view(), "{\"cmd\":\"rm -rf /\"}");
        assert_eq!(context.get_tool_name(), Some("shell"));
        assert_eq!(context.event, EVENT_TOOL_INVOKE_INPUT);
        assert_eq!(context.content_type, crate::GuardrailContentType::ToolCall);

        let output = GuardrailContext::tool_output(
            "shell",
            &Value::String("done".to_owned()),
            ScanStage::ToolInvokeOutput,
        );
        assert!(matches!(output.text_view(), Cow::Borrowed("done")));

        let raw = GuardrailContext::raw("custom_event", Value::String("plain".to_owned()));
        assert_eq!(raw.content_type, crate::GuardrailContentType::Raw);
        assert_eq!(raw.event, "custom_event");
        assert_eq!(raw.text_view(), "plain");

        let object = GuardrailContext::raw("custom_event", json!({"a": 1}));
        assert_eq!(object.text_view(), "{\"a\":1}");
    }

    #[test]
    fn with_metadata_adds_keys() {
        let context = GuardrailContext::text("x", ScanStage::LlmInvokeOutput)
            .with_metadata("tool_name", json!("t"));
        assert_eq!(context.get_tool_name(), Some("t"));
        assert_eq!(context.event, EVENT_LLM_INVOKE_OUTPUT);
    }

    #[test]
    fn debug_redacts_content() {
        let context = GuardrailContext::text("super-secret-token", ScanStage::UserInput)
            .with_metadata("b", json!(2))
            .with_metadata("a", json!(1));
        let debug = format!("{context:?}");
        assert!(debug.contains(REDACTED), "{debug}");
        assert!(!debug.contains("super-secret-token"), "{debug}");
        assert!(debug.contains("[\"a\", \"b\"]"), "{debug}");
    }

    #[test]
    fn scan_stage_serde_round_trips_including_other() {
        for stage in [
            ScanStage::LlmInvokeInput,
            ScanStage::LlmInvokeOutput,
            ScanStage::ToolInvokeInput,
            ScanStage::ToolInvokeOutput,
            ScanStage::UserInput,
        ] {
            let value = serde_json::to_value(&stage).unwrap();
            assert_eq!(value, json!(stage.event()));
            let back: ScanStage = serde_json::from_value(value).unwrap();
            assert_eq!(back, stage);
        }

        let other = ScanStage::Other("custom".to_owned());
        let value = serde_json::to_value(&other).unwrap();
        assert_eq!(value, json!("custom"));
        let back: ScanStage = serde_json::from_value(value).unwrap();
        assert_eq!(back, other);

        let error = serde_json::from_value::<ScanStage>(json!(42)).unwrap_err();
        assert!(error
            .to_string()
            .contains("expected a scan-stage event string"));
    }

    #[test]
    fn scan_stage_event_and_content_type() {
        assert_eq!(
            ScanStage::from_event(EVENT_TOOL_INVOKE_OUTPUT),
            ScanStage::ToolInvokeOutput
        );
        assert_eq!(
            ScanStage::from_event("unknown.event"),
            ScanStage::Other("unknown.event".to_owned())
        );
        assert_eq!(
            ScanStage::ToolInvokeInput.to_string(),
            EVENT_TOOL_INVOKE_INPUT
        );
        assert_eq!(
            ScanStage::ToolInvokeOutput.content_type(),
            crate::GuardrailContentType::ToolCall
        );
        assert_eq!(
            ScanStage::Other("x".to_owned()).content_type(),
            crate::GuardrailContentType::Raw
        );
        assert_eq!(
            ScanStage::UserInput.content_type(),
            crate::GuardrailContentType::Text
        );
    }
}
