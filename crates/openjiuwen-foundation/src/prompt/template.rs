//! Interpolatable prompt templates with configurable placeholders.
//!
//! Ports `openjiuwen/core/foundation/prompt/template.py` (`PromptTemplate`) and
//! the string handling of `openjiuwen/core/foundation/prompt/assemble/`. A
//! template holds either plain text or a fixed message list plus a pair of
//! delimiters (default `{{` / `}}`), and [`PromptTemplate::format`] returns a new
//! template with every resolvable placeholder substituted.
//!
//! ```
//! use openjiuwen_foundation::prompt::template::PromptTemplate;
//!
//! let template = PromptTemplate::named("greet", "hi {{name}}!");
//! assert_eq!(template.input_keys(), vec!["name".to_owned()]);
//! let filled = template.format_one("name", "bob");
//! assert_eq!(filled.to_messages()[0].text(), "hi bob!");
//! ```
//!
//! Rendering rules (the binding contract, see `prompt/CONTEXT.md`):
//! - Only `prefix + name + suffix` is a placeholder; the name is
//!   whitespace-trimmed and may not contain `{`/`}`. Empty names (`{{}}`) and
//!   single braces stay literal.
//! - Extra values are ignored; an unresolvable placeholder is kept verbatim, so
//!   a partially formatted template stays usable and can be formatted again.
//! - A dotted name (`{{user.name}}`) resolves through nested JSON objects; a
//!   missing head, a missing intermediate, or a `null` leaf keeps the
//!   placeholder verbatim (and a `null` never renders as `"null"`).
//! - Substitution is single pass: an inserted value is never re-scanned, so
//!   values cannot inject further placeholders.
//! - For a message-list template only text content is rendered; other content
//!   parts and every other field are preserved byte-identically.

use std::collections::BTreeMap;
use std::path::Path;

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::assembler;
use super::content::{ContentPart, MessageContent};
use super::message::BaseMessage;

/// The content of a [`PromptTemplate`].
///
/// Serializes as a JSON string for [`TemplateContent::Text`] and as a JSON array
/// of messages for [`TemplateContent::Messages`]; deserialization is untagged, so
/// the two shapes are told apart by their JSON type alone (Python types the field
/// as `Union[str, List[BaseMessage]]`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TemplateContent {
    /// A single template string.
    Text(String),
    /// A message list whose text content is templated.
    Messages(Vec<BaseMessage>),
}

impl Default for TemplateContent {
    /// Empty text; Python's `Union[str, ...]` default is `""`.
    fn default() -> Self {
        Self::Text(String::new())
    }
}

impl From<&str> for TemplateContent {
    /// Wraps a borrowed string as [`TemplateContent::Text`].
    fn from(value: &str) -> Self {
        Self::Text(value.to_owned())
    }
}

impl From<String> for TemplateContent {
    /// Wraps an owned string as [`TemplateContent::Text`].
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<Vec<BaseMessage>> for TemplateContent {
    /// Wraps a message list as [`TemplateContent::Messages`].
    fn from(value: Vec<BaseMessage>) -> Self {
        Self::Messages(value)
    }
}

/// An interpolatable prompt template.
///
/// Field names and defaults match the Python model (`name`, `content`,
/// `placeholder_prefix`, `placeholder_suffix`), so a template serializes to and
/// from the same JSON/YAML documents. Unknown fields are ignored and every
/// missing field falls back to [`PromptTemplate::default`] — in particular the
/// delimiters default to `"{{"`/`"}}"`, never to empty strings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PromptTemplate {
    /// The template name (empty when unnamed).
    pub name: String,
    /// The templated text or message list.
    pub content: TemplateContent,
    /// Opening placeholder delimiter.
    pub placeholder_prefix: String,
    /// Closing placeholder delimiter.
    pub placeholder_suffix: String,
}

impl Default for PromptTemplate {
    /// An unnamed, empty text template with the `{{` / `}}` delimiters.
    ///
    /// Implemented manually (not derived) because the delimiter defaults are
    /// non-empty; `#[serde(default)]` on the struct uses this impl, so a document
    /// that omits `placeholder_prefix`/`placeholder_suffix` keeps `{{`/`}}`.
    fn default() -> Self {
        Self {
            name: String::new(),
            content: TemplateContent::default(),
            placeholder_prefix: "{{".to_owned(),
            placeholder_suffix: "}}".to_owned(),
        }
    }
}

impl PromptTemplate {
    /// Creates an unnamed template from text or a message list.
    #[must_use]
    pub fn new(content: impl Into<TemplateContent>) -> Self {
        Self {
            content: content.into(),
            ..Self::default()
        }
    }

    /// Creates a named template from text or a message list.
    #[must_use]
    pub fn named(name: impl Into<String>, content: impl Into<TemplateContent>) -> Self {
        Self {
            name: name.into(),
            content: content.into(),
            ..Self::default()
        }
    }

    /// Creates an unnamed template over an existing message list.
    #[must_use]
    pub fn from_messages(messages: Vec<BaseMessage>) -> Self {
        Self::new(messages)
    }

    /// Replaces both placeholder delimiters (builder style).
    ///
    /// The delimiters are used literally and are not escaped, matching Python's
    /// `re.escape`-based implementation; an empty delimiter disables placeholder
    /// detection instead of matching everywhere.
    #[must_use]
    pub fn with_placeholders(self, prefix: impl Into<String>, suffix: impl Into<String>) -> Self {
        Self {
            placeholder_prefix: prefix.into(),
            placeholder_suffix: suffix.into(),
            ..self
        }
    }

    /// Every distinct placeholder name in the content, in first-seen order.
    ///
    /// Names are the full dotted paths as written (`"user.name"`), whitespace
    /// already trimmed. For a message list, the text content of each message is
    /// scanned in order — the whole [`MessageContent::Text`] value, and for
    /// [`MessageContent::Parts`] only the [`ContentPart::Text`] parts.
    #[must_use]
    pub fn placeholders(&self) -> Vec<String> {
        match &self.content {
            TemplateContent::Text(text) => self.placeholders_in(text),
            TemplateContent::Messages(messages) => {
                let mut names: Vec<String> = Vec::new();
                for message in messages {
                    for text in message_texts(message) {
                        dedupe_into(&mut names, self.placeholders_in(text));
                    }
                }
                names
            }
        }
    }

    /// The distinct head segments of [`PromptTemplate::placeholders`].
    ///
    /// The head is the part before the first `.`, so `"user.name"` and
    /// `"user.age"` both require the caller to supply `"user"`. Order is
    /// first-seen.
    #[must_use]
    pub fn input_keys(&self) -> Vec<String> {
        match &self.content {
            TemplateContent::Text(text) => self.input_keys_in(text),
            TemplateContent::Messages(messages) => {
                let mut keys: Vec<String> = Vec::new();
                for message in messages {
                    for text in message_texts(message) {
                        dedupe_into(&mut keys, self.input_keys_in(text));
                    }
                }
                keys
            }
        }
    }

    /// Renders the template with `values` and returns a new template.
    ///
    /// `self` is never mutated. The name and both delimiters are copied, so the
    /// result can be formatted again (for example to fill the slots a first pass
    /// could not resolve). An empty `values` map returns a clone, as Python does.
    /// For a message list only the text content is rendered; image, audio, file,
    /// and unknown parts plus `role`, `name`, `tool_calls`, `tool_call_id`,
    /// `reasoning_content`, and `metadata` are preserved exactly.
    #[must_use]
    pub fn format(&self, values: &BTreeMap<String, Value>) -> Self {
        if values.is_empty() {
            return self.clone();
        }

        let prefix = self.placeholder_prefix.as_str();
        let suffix = self.placeholder_suffix.as_str();
        let content = match &self.content {
            TemplateContent::Text(text) => {
                TemplateContent::Text(assembler::render(text, values, prefix, suffix))
            }
            TemplateContent::Messages(messages) => TemplateContent::Messages(
                messages
                    .iter()
                    .map(|message| render_message(message, values, prefix, suffix))
                    .collect(),
            ),
        };

        Self {
            name: self.name.clone(),
            content,
            placeholder_prefix: self.placeholder_prefix.clone(),
            placeholder_suffix: self.placeholder_suffix.clone(),
        }
    }

    /// Renders the template with a single `key`/`value` pair.
    ///
    /// Convenience wrapper over [`PromptTemplate::format`] for the common
    /// one-slot case.
    #[must_use]
    pub fn format_one(&self, key: impl Into<String>, value: impl Into<Value>) -> Self {
        self.format(&BTreeMap::from([(key.into(), value.into())]))
    }

    /// Converts the content to a message list.
    ///
    /// Empty text yields no messages (Python returns `[]`), non-empty text yields
    /// one user message, and a message list is returned as a deep copy.
    #[must_use]
    pub fn to_messages(&self) -> Vec<BaseMessage> {
        match &self.content {
            TemplateContent::Text(text) if text.is_empty() => Vec::new(),
            TemplateContent::Text(text) => vec![BaseMessage::user(text.clone())],
            TemplateContent::Messages(messages) => messages.clone(),
        }
    }

    /// Parses a template from a YAML document.
    ///
    /// The document is normally a mapping with `name`, `content`,
    /// `placeholder_prefix`, and `placeholder_suffix` (all optional). A bare
    /// string document (`"hello {{x}}"`) is accepted as the `content` of an
    /// otherwise default template, which is friendlier than Python's strict model
    /// and never loses a template file. Every parse failure becomes
    /// `Error::Validation` carrying the parser message — this function never
    /// panics.
    pub fn from_yaml_str(yaml: &str) -> Result<Self> {
        let error = match serde_yaml_ng::from_str::<Self>(yaml) {
            Ok(template) => return Ok(template),
            Err(error) => error,
        };

        match serde_yaml_ng::from_str::<String>(yaml) {
            Ok(text) => Ok(Self::new(text)),
            Err(_) => Err(Error::validation(format!(
                "invalid prompt template YAML: {error}"
            ))),
        }
    }

    /// Reads and parses a YAML template file.
    ///
    /// I/O failures are reported as `Error::Validation` — a template path is
    /// configuration, not a transport — with the path as context; parse failures
    /// keep the [`PromptTemplate::from_yaml_str`] classification and gain the same
    /// context.
    pub fn load_yaml(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let context = format!("prompt template {}", path.display());
        let text = std::fs::read_to_string(path)
            .map_err(|error| Error::validation(format!("cannot read template: {error}")))
            .map_err(|error| error.with_context(&context))?;
        Self::from_yaml_str(&text).map_err(|error| error.with_context(&context))
    }

    /// The distinct placeholder names of one string, under this template's delimiters.
    fn placeholders_in(&self, text: &str) -> Vec<String> {
        assembler::placeholders(
            text,
            self.placeholder_prefix.as_str(),
            self.placeholder_suffix.as_str(),
        )
    }

    /// The distinct placeholder head segments of one string.
    fn input_keys_in(&self, text: &str) -> Vec<String> {
        assembler::input_keys(
            text,
            self.placeholder_prefix.as_str(),
            self.placeholder_suffix.as_str(),
        )
    }
}

/// Appends the names in `names` that `target` does not hold yet, order preserved.
fn dedupe_into(target: &mut Vec<String>, names: Vec<String>) {
    for name in names {
        if !target.contains(&name) {
            target.push(name);
        }
    }
}

/// The text segments of `message` that a template renders.
///
/// A [`MessageContent::Text`] contributes itself; [`MessageContent::Parts`]
/// contributes only its [`ContentPart::Text`] parts, so binary payloads are never
/// scanned.
fn message_texts(message: &BaseMessage) -> Vec<&str> {
    match &message.content {
        MessageContent::Text(text) => vec![text.as_str()],
        MessageContent::Parts(parts) => parts.iter().filter_map(ContentPart::text).collect(),
    }
}

/// Renders a copy of `message` with only its text content substituted.
fn render_message(
    message: &BaseMessage,
    values: &BTreeMap<String, Value>,
    prefix: &str,
    suffix: &str,
) -> BaseMessage {
    let mut rendered = message.clone();
    rendered.content = match &message.content {
        MessageContent::Text(text) => {
            MessageContent::Text(assembler::render(text, values, prefix, suffix))
        }
        MessageContent::Parts(parts) => MessageContent::Parts(
            parts
                .iter()
                .map(|part| render_part(part, values, prefix, suffix))
                .collect(),
        ),
    };
    rendered
}

/// Renders one content part: [`ContentPart::Text`] is templated, everything else
/// is cloned unchanged.
fn render_part(
    part: &ContentPart,
    values: &BTreeMap<String, Value>,
    prefix: &str,
    suffix: &str,
) -> ContentPart {
    match part {
        ContentPart::Text { text } => ContentPart::Text {
            text: assembler::render(text, values, prefix, suffix),
        },
        other => other.clone(),
    }
}

#[cfg(test)]
mod tests {
    //! Ports the `PromptTemplate` behaviors of
    //! `tests/unit_tests/core/foundation/prompt/`, with the crate's documented
    //! deviations: unresolvable slots stay verbatim instead of raising, and
    //! non-string values render as JSON rather than Python `str()`.

    use std::collections::BTreeMap;

    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;

    use super::*;
    use crate::prompt::content::ImageUrl;
    use crate::prompt::tool_call::ToolCall;

    fn map(pairs: &[(&str, Value)]) -> BTreeMap<String, Value> {
        pairs
            .iter()
            .map(|(key, value)| ((*key).to_owned(), value.clone()))
            .collect()
    }

    fn image(url: &str) -> ContentPart {
        ContentPart::ImageUrl {
            image_url: ImageUrl {
                url: url.to_owned(),
                detail: None,
            },
        }
    }

    #[test]
    fn default_has_braces_delimiters_and_empty_text() {
        let template = PromptTemplate::default();
        assert_eq!(template.name, "");
        assert_eq!(template.content, TemplateContent::Text(String::new()));
        assert_eq!(template.placeholder_prefix, "{{");
        assert_eq!(template.placeholder_suffix, "}}");
        assert_eq!(
            TemplateContent::default(),
            TemplateContent::Text(String::new())
        );
    }

    #[test]
    fn constructors_set_name_and_content() {
        let text = PromptTemplate::new("hi {{name}}");
        assert_eq!(text.name, "");
        assert_eq!(
            text.content,
            TemplateContent::Text("hi {{name}}".to_owned())
        );

        let owned = PromptTemplate::new("hi".to_owned());
        assert_eq!(owned.content, TemplateContent::Text("hi".to_owned()));

        let named = PromptTemplate::named("greet", "hi");
        assert_eq!(named.name, "greet");
        assert_eq!(named.content, TemplateContent::Text("hi".to_owned()));

        let messages = vec![BaseMessage::system("sys"), BaseMessage::user("u")];
        let from_list = PromptTemplate::from_messages(messages.clone());
        assert_eq!(from_list.name, "");
        assert_eq!(from_list.content, TemplateContent::Messages(messages));

        let custom = PromptTemplate::new("hi [[name]]").with_placeholders("[[", "]]");
        assert_eq!(custom.placeholder_prefix, "[[");
        assert_eq!(custom.placeholder_suffix, "]]");
        assert_eq!(custom.placeholders(), vec!["name".to_owned()]);
    }

    #[test]
    fn placeholders_and_input_keys_for_text() {
        let template = PromptTemplate::new("{{b}} {{a}} {{b}} {{a.b}}");
        assert_eq!(template.placeholders(), vec!["b", "a", "a.b"]);
        assert_eq!(template.input_keys(), vec!["b", "a"]);

        let plain = PromptTemplate::new("no slots");
        assert_eq!(plain.placeholders(), Vec::<String>::new());
        assert_eq!(plain.input_keys(), Vec::<String>::new());

        let trimmed = PromptTemplate::new("{{ user.name }} {{ x }}");
        assert_eq!(trimmed.placeholders(), vec!["user.name", "x"]);
        assert_eq!(trimmed.input_keys(), vec!["user", "x"]);
    }

    #[test]
    fn placeholders_for_message_lists_scan_only_text_parts() {
        let messages = vec![
            BaseMessage::system("{{system_prompt}}"),
            BaseMessage::user(vec![
                ContentPart::Text {
                    text: "look at {{image.alt}}".to_owned(),
                },
                image("{{not_a_slot}}"),
                ContentPart::Other(json!({"type": "video", "url": "{{also_not}}"})),
            ]),
            BaseMessage::assistant("{{system_prompt}} {{answer}}"),
        ];
        let template = PromptTemplate::from_messages(messages);
        assert_eq!(
            template.placeholders(),
            vec!["system_prompt", "image.alt", "answer"]
        );
        assert_eq!(
            template.input_keys(),
            vec!["system_prompt", "image", "answer"]
        );
    }

    #[test]
    fn format_returns_a_new_template_and_leaves_the_original_untouched() {
        let template = PromptTemplate::named("greet", "hi {{name}} and {{other}}");
        let formatted = template.format(&map(&[("name", json!("bob"))]));

        assert_eq!(
            formatted.content,
            TemplateContent::Text("hi bob and {{other}}".to_owned())
        );
        assert_eq!(
            template.content,
            TemplateContent::Text("hi {{name}} and {{other}}".to_owned()),
            "format must not mutate the receiver"
        );
        assert_ne!(formatted, template);
        assert_eq!(template.format(&BTreeMap::new()), template);
    }

    #[test]
    fn format_preserves_name_and_custom_delimiters() {
        let template = PromptTemplate::named("greet", "hi <<name>>").with_placeholders("<<", ">>");
        let formatted = template.format_one("name", "bob");
        assert_eq!(formatted.name, "greet");
        assert_eq!(formatted.placeholder_prefix, "<<");
        assert_eq!(formatted.placeholder_suffix, ">>");
        assert_eq!(
            formatted.content,
            TemplateContent::Text("hi bob".to_owned())
        );
        // The default delimiters stay inert under the custom ones.
        assert_eq!(
            template.format_one("name", "{{x}}").to_messages()[0].text(),
            "hi {{x}}"
        );
    }

    #[test]
    fn format_renders_message_lists_preserving_metadata() {
        let call = ToolCall::new("get_weather")
            .with_id("call_1")
            .with_arguments(json!({"city": "Paris"}));
        let messages = vec![
            BaseMessage::system("you are {{persona}}"),
            BaseMessage::assistant("")
                .with_tool_calls(vec![call.clone()])
                .with_reasoning("because {{persona}}"),
            BaseMessage::tool("call_1", "42")
                .with_name("tool")
                .with_metadata("trace", json!("{{persona}}")),
        ];
        let template = PromptTemplate::named("agent", messages.clone());
        let formatted = template.format_one("persona", "a pirate");

        assert_eq!(
            formatted.content,
            TemplateContent::Messages(vec![
                BaseMessage::system("you are a pirate"),
                BaseMessage::assistant("")
                    .with_tool_calls(vec![call])
                    .with_reasoning("because {{persona}}"),
                BaseMessage::tool("call_1", "42")
                    .with_name("tool")
                    .with_metadata("trace", json!("{{persona}}")),
            ])
        );
        assert_eq!(template.content, TemplateContent::Messages(messages));
    }

    #[test]
    fn to_messages_covers_the_three_cases() {
        assert_eq!(
            PromptTemplate::new("").to_messages(),
            Vec::<BaseMessage>::new()
        );
        assert_eq!(
            PromptTemplate::default().to_messages(),
            Vec::<BaseMessage>::new()
        );
        assert_eq!(
            PromptTemplate::new("hi {{name}}").to_messages(),
            vec![BaseMessage::user("hi {{name}}")]
        );

        let messages = vec![BaseMessage::system("s"), BaseMessage::user("u")];
        let template = PromptTemplate::from_messages(messages.clone());
        let mut copy = template.to_messages();
        assert_eq!(copy, messages);
        copy[0] = BaseMessage::system("mutated");
        assert_eq!(
            template.to_messages(),
            messages,
            "to_messages must deep-copy"
        );
    }

    #[test]
    fn serde_round_trips_text_and_message_content() {
        let text = PromptTemplate::named("greet", "hi {{name}}");
        let value = serde_json::to_value(&text).unwrap();
        assert_eq!(
            value,
            json!({
                "name": "greet",
                "content": "hi {{name}}",
                "placeholder_prefix": "{{",
                "placeholder_suffix": "}}",
            })
        );
        assert_eq!(
            serde_json::from_value::<PromptTemplate>(value).unwrap(),
            text
        );

        let messages = PromptTemplate::from_messages(vec![BaseMessage::user("{{q}}")]);
        let value = serde_json::to_value(&messages).unwrap();
        assert_eq!(
            value,
            json!({
                "name": "",
                "content": [{"role": "user", "content": "{{q}}"}],
                "placeholder_prefix": "{{",
                "placeholder_suffix": "}}",
            })
        );
        assert_eq!(
            serde_json::from_value::<PromptTemplate>(value).unwrap(),
            messages
        );

        // Missing fields fall back to the manual `Default` (braces delimiters).
        let parsed: PromptTemplate = serde_json::from_value(json!({"content": "x"})).unwrap();
        assert_eq!(parsed, PromptTemplate::new("x"));
    }

    #[test]
    fn template_content_is_untagged() {
        assert_eq!(
            serde_json::from_value::<TemplateContent>(json!("hi")).unwrap(),
            TemplateContent::Text("hi".to_owned())
        );
        assert_eq!(
            serde_json::from_value::<TemplateContent>(json!([{"role": "user", "content": "hi"}]))
                .unwrap(),
            TemplateContent::Messages(vec![BaseMessage::user("hi")])
        );
        assert_eq!(
            serde_json::to_value(TemplateContent::Text("hi".to_owned())).unwrap(),
            json!("hi")
        );
        assert!(serde_json::from_value::<TemplateContent>(json!(7)).is_err());
    }

    #[test]
    fn from_yaml_str_parses_mappings_and_bare_strings() {
        let yaml = "name: greet\ncontent: \"hi <<name>>\"\nplaceholder_prefix: \"<<\"\nplaceholder_suffix: \">>\"\n";
        let parsed = PromptTemplate::from_yaml_str(yaml).unwrap();
        assert_eq!(
            parsed,
            PromptTemplate::named("greet", "hi <<name>>").with_placeholders("<<", ">>")
        );

        let bare = PromptTemplate::from_yaml_str("hello {{x}}").unwrap();
        assert_eq!(bare, PromptTemplate::new("hello {{x}}"));

        let minimal = PromptTemplate::from_yaml_str("content: hi").unwrap();
        assert_eq!(minimal, PromptTemplate::new("hi"));
    }

    #[test]
    fn from_yaml_str_reports_invalid_yaml_as_validation() {
        let error = PromptTemplate::from_yaml_str("name: [unclosed").unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(
            error.to_string().contains("invalid prompt template YAML"),
            "unexpected message: {error}"
        );
    }

    #[test]
    fn load_yaml_reads_a_file_and_reports_missing_paths() {
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(file.path(), "name: greet\ncontent: hi {{name}}\n").unwrap();
        let loaded = PromptTemplate::load_yaml(file.path()).unwrap();
        assert_eq!(loaded, PromptTemplate::named("greet", "hi {{name}}"));

        let missing = file.path().with_extension("does-not-exist.yaml");
        let error = PromptTemplate::load_yaml(&missing).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(
            error.to_string().contains("cannot read template"),
            "unexpected message: {error}"
        );
    }

    #[test]
    fn format_one_and_numeric_values_round_trip() {
        let template = PromptTemplate::new("{{n}} {{flag}} {{empty}}");
        let formatted = template.format(&map(&[
            ("n", json!(42)),
            ("flag", json!(true)),
            ("empty", json!("")),
        ]));
        assert_eq!(
            formatted.content,
            TemplateContent::Text("42 true ".to_owned())
        );
        assert_eq!(template.input_keys(), vec!["n", "flag", "empty"]);
        assert_eq!(
            template.format_one("n", 7).content,
            TemplateContent::Text("7 {{flag}} {{empty}}".to_owned())
        );
        assert_eq!(
            PromptTemplate::new("{{u}}")
                .format_one("u", json!({"name": "bob"}))
                .content,
            TemplateContent::Text(r#"{"name":"bob"}"#.to_owned())
        );
    }
}
