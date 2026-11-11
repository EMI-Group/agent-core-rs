//! Chat messages: one struct for every role.
//!
//! Ports the message schema of `openjiuwen/core/foundation/llm/schema/message.py`,
//! where Python defines a `BaseMessage` plus `SystemMessage`/`UserMessage`/
//! `AssistantMessage`/`ToolMessage` subclasses. Rust keeps a single struct and
//! moves the role-specific part into constructors ([`BaseMessage::system`],
//! [`BaseMessage::user`], [`BaseMessage::assistant`], [`BaseMessage::tool`]), so
//! providers and agents never need downcasting to read a message.
//!
//! ```
//! use openjiuwen_foundation::prompt::message::BaseMessage;
//!
//! let message = BaseMessage::user("what is the weather?").with_name("alice");
//! assert_eq!(message.text(), "what is the weather?");
//! ```

use std::borrow::Cow;
use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::content::{ContentPart, MessageContent};
use super::role::Role;
use super::tool_call::ToolCall;

/// One message in a conversation.
///
/// The wire shape keeps the Python field names and order (`role`, `content`,
/// `name`, `tool_calls`, `tool_call_id`, `reasoning_content`, `metadata`), omits
/// empty optional fields, and tolerates unknown fields coming from a provider.
/// [`BaseMessage::role`] and [`BaseMessage::content`] are always serialized
/// because every provider requires them.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct BaseMessage {
    /// Who authored the message; defaults to [`Role::User`].
    pub role: Role,
    /// The text or multimodal payload of the message.
    pub content: MessageContent,
    /// Optional speaker name, used by some providers to distinguish participants.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Tool calls requested by an assistant message.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    /// Id of the call a [`Role::Tool`] message answers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Reasoning/thinking text some providers return alongside the answer.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    /// Free-form per-message annotations; keys are stable and serialized sorted.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, Value>,
}

impl BaseMessage {
    /// Creates a [`Role::System`] message.
    #[must_use]
    pub fn system(content: impl Into<MessageContent>) -> Self {
        Self {
            role: Role::System,
            content: content.into(),
            ..Self::default()
        }
    }

    /// Creates a [`Role::User`] message.
    #[must_use]
    pub fn user(content: impl Into<MessageContent>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
            ..Self::default()
        }
    }

    /// Creates a [`Role::Assistant`] message.
    #[must_use]
    pub fn assistant(content: impl Into<MessageContent>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
            ..Self::default()
        }
    }

    /// Creates a [`Role::Tool`] message answering the call `tool_call_id`.
    #[must_use]
    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<MessageContent>) -> Self {
        Self {
            role: Role::Tool,
            content: content.into(),
            tool_call_id: Some(tool_call_id.into()),
            ..Self::default()
        }
    }

    /// Sets [`BaseMessage::name`].
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Replaces [`BaseMessage::tool_calls`] with the given calls.
    #[must_use]
    pub fn with_tool_calls(mut self, calls: Vec<ToolCall>) -> Self {
        self.tool_calls = calls;
        self
    }

    /// Sets [`BaseMessage::reasoning_content`].
    #[must_use]
    pub fn with_reasoning(mut self, reasoning: impl Into<String>) -> Self {
        self.reasoning_content = Some(reasoning.into());
        self
    }

    /// Inserts one [`BaseMessage::metadata`] entry; a later insert with the same
    /// key replaces the previous value.
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Appends one content part, promoting text content to a part list.
    ///
    /// For [`MessageContent::Text`] the existing text becomes the first
    /// [`ContentPart::Text`] part — an empty text contributes no part at all, so
    /// the content becomes `Parts(vec![part])`; for
    /// [`MessageContent::Parts`] the part is pushed onto the existing list.
    pub fn push_part(&mut self, part: ContentPart) {
        let content = std::mem::take(&mut self.content);
        self.content = match content {
            MessageContent::Text(text) if text.is_empty() => MessageContent::Parts(vec![part]),
            MessageContent::Text(text) => {
                MessageContent::Parts(vec![ContentPart::Text { text }, part])
            }
            MessageContent::Parts(mut parts) => {
                parts.push(part);
                MessageContent::Parts(parts)
            }
        };
    }

    /// The plain text view of the message.
    ///
    /// [`MessageContent::Text`] is returned borrowed, so the common case copies
    /// nothing; [`MessageContent::Parts`] is flattened by
    /// [`MessageContent::to_plain_text`] (text parts joined with `\n`, other
    /// parts skipped).
    #[must_use]
    pub fn text(&self) -> Cow<'_, str> {
        match &self.content {
            MessageContent::Text(text) => Cow::Borrowed(text.as_str()),
            MessageContent::Parts(_) => Cow::Owned(self.content.to_plain_text()),
        }
    }
}

impl From<&str> for BaseMessage {
    /// Builds a [`BaseMessage::user`] message.
    fn from(value: &str) -> Self {
        Self::user(value)
    }
}

impl From<String> for BaseMessage {
    /// Builds a [`BaseMessage::user`] message.
    fn from(value: String) -> Self {
        Self::user(value)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::prompt::content::ImageUrl;

    fn image(url: &str) -> ContentPart {
        ContentPart::ImageUrl {
            image_url: ImageUrl {
                url: url.to_owned(),
                detail: None,
            },
        }
    }

    #[test]
    fn constructors_set_role_and_content() {
        assert_eq!(BaseMessage::system("s").role, Role::System);
        assert_eq!(
            BaseMessage::system("s").content,
            MessageContent::Text("s".to_owned())
        );

        assert_eq!(BaseMessage::user("u").role, Role::User);
        assert_eq!(
            BaseMessage::user("u".to_owned()).content.as_text(),
            Some("u")
        );

        assert_eq!(BaseMessage::assistant("a").role, Role::Assistant);
        let parts = vec![image("https://x/y.png")];
        assert_eq!(
            BaseMessage::assistant(parts.clone()).content,
            MessageContent::Parts(parts)
        );

        let tool = BaseMessage::tool("call_1", "42");
        assert_eq!(tool.role, Role::Tool);
        assert_eq!(tool.tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(tool.content.as_text(), Some("42"));
    }

    #[test]
    fn constructors_leave_optional_fields_empty() {
        let message = BaseMessage::user("u");
        assert_eq!(message.name, None);
        assert_eq!(message.tool_calls, Vec::new());
        assert_eq!(message.tool_call_id, None);
        assert_eq!(message.reasoning_content, None);
        assert!(message.metadata.is_empty());
    }

    #[test]
    fn builders_attach_optional_fields() {
        let call = ToolCall::new("get_weather").with_id("call_1");
        let message = BaseMessage::assistant("thinking")
            .with_name("alice")
            .with_tool_calls(vec![call.clone()])
            .with_reasoning("step by step");
        assert_eq!(message.name.as_deref(), Some("alice"));
        assert_eq!(message.tool_calls, vec![call]);
        assert_eq!(message.reasoning_content.as_deref(), Some("step by step"));
        assert_eq!(message.role, Role::Assistant);
    }

    #[test]
    fn with_metadata_accumulates_and_replaces_one_key() {
        let message = BaseMessage::user("u")
            .with_metadata("trace", json!("abc"))
            .with_metadata("turn", json!(1))
            .with_metadata("turn", json!(2));
        assert_eq!(message.metadata.len(), 2);
        assert_eq!(message.metadata.get("trace"), Some(&json!("abc")));
        assert_eq!(message.metadata.get("turn"), Some(&json!(2)));
    }

    #[test]
    fn push_part_replaces_empty_text() {
        let mut message = BaseMessage::user("");
        message.push_part(image("https://x/y.png"));
        assert_eq!(
            message.content,
            MessageContent::Parts(vec![image("https://x/y.png")])
        );
    }

    #[test]
    fn push_part_keeps_existing_text_as_the_first_part() {
        let mut message = BaseMessage::user("describe");
        message.push_part(image("https://x/y.png"));
        assert_eq!(
            message.content,
            MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "describe".to_owned()
                },
                image("https://x/y.png"),
            ])
        );
    }

    #[test]
    fn push_part_appends_to_an_existing_part_list() {
        let mut message = BaseMessage::assistant(vec![image("https://x/a.png")]);
        message.push_part(ContentPart::Text {
            text: "and this".to_owned(),
        });
        assert_eq!(
            message.content,
            MessageContent::Parts(vec![
                image("https://x/a.png"),
                ContentPart::Text {
                    text: "and this".to_owned()
                },
            ])
        );
    }

    #[test]
    fn text_borrows_text_content_and_joins_parts() {
        let text = BaseMessage::user("plain");
        assert_eq!(text.text(), "plain");
        assert!(matches!(text.text(), Cow::Borrowed(_)));

        let parts = BaseMessage::user(vec![
            ContentPart::Text {
                text: "first".to_owned(),
            },
            image("https://x/y.png"),
            ContentPart::Text {
                text: "second".to_owned(),
            },
        ]);
        assert_eq!(parts.text(), "first\nsecond");
        assert!(matches!(parts.text(), Cow::Owned(_)));

        assert_eq!(BaseMessage::user("").text(), "");
    }

    #[test]
    fn default_is_a_user_message_with_empty_content() {
        let message = BaseMessage::default();
        assert_eq!(message.role, Role::User);
        assert_eq!(message.content, MessageContent::Text(String::new()));
        assert!(message.content.is_empty());
        assert_eq!(message.text(), "");
        assert_eq!(message, BaseMessage::user(""));
    }

    #[test]
    fn full_message_round_trips_through_serde() {
        let call = ToolCall::new("get_weather")
            .with_id("call_1")
            .with_index(0)
            .with_arguments(json!({"city": "Paris"}));
        let message = BaseMessage {
            role: Role::Assistant,
            content: MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "checking".to_owned(),
                },
                image("https://x/y.png"),
            ]),
            name: Some("alice".to_owned()),
            tool_calls: vec![call],
            tool_call_id: Some("call_0".to_owned()),
            reasoning_content: Some("because".to_owned()),
            metadata: BTreeMap::from([("trace".to_owned(), json!("t1"))]),
        };

        let value = serde_json::to_value(&message).unwrap();
        assert_eq!(
            value,
            json!({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "checking"},
                    {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
                ],
                "name": "alice",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "name": "get_weather",
                    "arguments": "{\"city\":\"Paris\"}",
                    "index": 0,
                }],
                "tool_call_id": "call_0",
                "reasoning_content": "because",
                "metadata": {"trace": "t1"},
            })
        );

        let back: BaseMessage = serde_json::from_value(value).unwrap();
        assert_eq!(back, message);
    }

    #[test]
    fn minimal_message_round_trips_and_omits_empty_fields() {
        let message = BaseMessage::user("hi");
        assert_eq!(
            serde_json::to_value(&message).unwrap(),
            json!({"role": "user", "content": "hi"})
        );

        let parsed: BaseMessage =
            serde_json::from_value(json!({"role": "user", "content": "hi"})).unwrap();
        assert_eq!(parsed, message);
        assert_eq!(parsed.tool_calls, Vec::new());
        assert!(parsed.metadata.is_empty());
    }

    #[test]
    fn unknown_fields_are_ignored() {
        let parsed: BaseMessage = serde_json::from_value(json!({
            "role": "assistant",
            "content": "x",
            "vendor_field": 1,
            "cache_control": {"type": "ephemeral"},
        }))
        .unwrap();
        assert_eq!(parsed, BaseMessage::assistant("x"));
        let value = serde_json::to_value(&parsed).unwrap();
        assert!(value.get("vendor_field").is_none());
        assert!(value.get("cache_control").is_none());
    }

    #[test]
    fn null_content_becomes_empty_text() {
        let parsed: BaseMessage =
            serde_json::from_value(json!({"role": "assistant", "content": null})).unwrap();
        assert_eq!(parsed.content, MessageContent::Text(String::new()));
        assert_eq!(parsed.text(), "");
        assert_eq!(parsed.content.to_plain_text(), "");
    }

    #[test]
    fn tool_calls_serialize_with_the_nested_wire_shape() {
        let message =
            BaseMessage::assistant("").with_tool_calls(vec![ToolCall::new("noop").with_id("c1")]);
        assert_eq!(
            serde_json::to_value(&message).unwrap(),
            json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "c1",
                    "type": "function",
                    "name": "noop",
                    "arguments": "",
                }],
            })
        );
    }

    #[test]
    fn from_conversions_build_user_messages() {
        assert_eq!(BaseMessage::from("hi"), BaseMessage::user("hi"));
        assert_eq!(
            BaseMessage::from("hi".to_owned()),
            BaseMessage::user("hi".to_owned())
        );
    }
}
