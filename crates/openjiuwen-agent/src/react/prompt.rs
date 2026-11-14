//! Identity prompt rendering for the `ReAct` agent.
//!
//! The identity section is the system-role text of
//! [`ReActAgentConfig::prompt_template`], joined with a blank line and rendered
//! against the invocation's template variables: the caller's extra keys,
//! `query`, `conversation_id`, `user_id`, `run_kind` when present, and finally
//! any extra render fields supplied by the caller (the memory rail's
//! `ctx.extra["memory_variables"]`), which override same-named fields.
use crate::input::AgentInput;
use crate::react::config::ReActAgentConfig;
use openjiuwen_foundation::{BaseMessage, PromptTemplate, TemplateContent};
use serde_json::{Map, Value};
use std::collections::BTreeMap;

/// Renders the identity section of the system prompt for one invocation.
///
/// Returns an empty string when the template has no system-role messages.
/// `{{placeholder}}` variables are resolved with
/// [`PromptTemplate::format`](openjiuwen_foundation::PromptTemplate); unknown
/// placeholders are left untouched.
///
/// `extra_fields` is merged after every other render field, so it wins on a
/// name collision; only its string values become template variables (mirroring
/// Python's `_render_system_messages`).
#[allow(dead_code)] // used by the loop's prompt construction
pub(crate) fn build_identity(
    config: &ReActAgentConfig,
    input: &AgentInput,
    extra_fields: Option<&Map<String, Value>>,
) -> String {
    let raw = config
        .prompt_template
        .iter()
        .filter(|message| message.role.is_system())
        .map(|message| message.text().into_owned())
        .collect::<Vec<_>>()
        .join("\n\n");
    if raw.is_empty() {
        return raw;
    }

    let mut values: BTreeMap<String, Value> = input.extra.clone();
    if let Some(query) = input.query_text() {
        values.insert("query".to_owned(), Value::String(query.to_owned()));
    }
    if let Some(conversation_id) = &input.conversation_id {
        values.insert(
            "conversation_id".to_owned(),
            Value::String(conversation_id.clone()),
        );
    }
    if let Some(user_id) = &input.user_id {
        values.insert("user_id".to_owned(), Value::String(user_id.clone()));
    }
    if let Some(run_kind) = &input.run_kind {
        values.insert("run_kind".to_owned(), Value::String(run_kind.clone()));
    }
    if let Some(extra_fields) = extra_fields {
        values.extend(
            extra_fields
                .iter()
                .filter(|(_, value)| value.is_string())
                .map(|(key, value)| (key.clone(), value.clone())),
        );
    }

    let rendered = PromptTemplate::new(raw.clone()).format(&values);
    match rendered.content {
        TemplateContent::Text(text) => text,
        // `raw` came from text messages, so this variant is unreachable in
        // practice; falling back to the unrendered text keeps it usable.
        TemplateContent::Messages(_) => raw,
    }
}

/// Wraps [`build_identity`] in the single system message the loop prepends.
///
/// The content may be empty when the template carries no system-role messages;
/// the loop still passes one system message so the window shape is stable.
#[allow(dead_code)] // used by the loop's prompt construction
pub(crate) fn build_system(
    config: &ReActAgentConfig,
    input: &AgentInput,
    extra_fields: Option<&Map<String, Value>>,
) -> BaseMessage {
    BaseMessage::system(build_identity(config, input, extra_fields))
}

#[cfg(test)]
mod tests {
    use super::*;
    use openjiuwen_foundation::BaseMessage;

    #[test]
    fn no_template_renders_empty() {
        let config = ReActAgentConfig::new();
        assert_eq!(build_identity(&config, &AgentInput::default(), None), "");
    }

    #[test]
    fn query_and_extra_variables_are_rendered() {
        let config = ReActAgentConfig {
            prompt_template: vec![BaseMessage::system("hello {{query}} from {{city}}")],
            ..ReActAgentConfig::new()
        };
        let mut input = AgentInput::from_text("bob");
        input
            .extra
            .insert("city".to_owned(), Value::String("Paris".to_owned()));
        assert_eq!(
            build_identity(&config, &input, None),
            "hello bob from Paris"
        );
    }

    #[test]
    fn unknown_placeholder_is_left_untouched() {
        let config = ReActAgentConfig {
            prompt_template: vec![BaseMessage::system("hi {{missing}}")],
            ..ReActAgentConfig::new()
        };
        assert_eq!(
            build_identity(&config, &AgentInput::from_text("q"), None),
            "hi {{missing}}"
        );
    }

    #[test]
    fn user_messages_do_not_feed_the_identity() {
        let config = ReActAgentConfig {
            prompt_template: vec![
                BaseMessage::system("system text"),
                BaseMessage::user("user text"),
            ],
            ..ReActAgentConfig::new()
        };
        assert_eq!(
            build_identity(&config, &AgentInput::default(), None),
            "system text"
        );
    }

    #[test]
    fn build_system_wraps_the_identity_in_one_system_message() {
        let config = ReActAgentConfig {
            prompt_template: vec![BaseMessage::system("hello {{query}}")],
            ..ReActAgentConfig::new()
        };
        let message = build_system(&config, &AgentInput::from_text("bob"), None);
        assert!(message.role.is_system());
        assert_eq!(message.text(), "hello bob");
    }

    #[test]
    fn build_system_allows_empty_content() {
        let message = build_system(&ReActAgentConfig::new(), &AgentInput::default(), None);
        assert!(message.role.is_system());
        assert_eq!(message.text(), "");
    }

    #[test]
    fn memory_variables_render_into_the_system_template() {
        let config = ReActAgentConfig {
            prompt_template: vec![BaseMessage::system(
                "memory: {{sys_long_term_memory}} vars: {{sys_memory_variables}}",
            )],
            ..ReActAgentConfig::new()
        };
        let fields = serde_json::json!({
            "sys_memory_variables": "{\"name\":\"Alice\"}",
            "sys_long_term_memory": "[\"用户画像记忆：\",\"likes helix\"]",
        });
        let Some(fields) = fields.as_object() else {
            panic!("the fixture is an object");
        };
        let rendered = build_identity(&config, &AgentInput::from_text("q"), Some(fields));
        assert_eq!(
            rendered,
            "memory: [\"用户画像记忆：\",\"likes helix\"] vars: {\"name\":\"Alice\"}"
        );
    }

    #[test]
    fn memory_variables_override_same_named_input_fields() {
        let config = ReActAgentConfig {
            prompt_template: vec![BaseMessage::system("{{user_id}} / {{city}}")],
            ..ReActAgentConfig::new()
        };
        let mut input = AgentInput::from_text("q");
        input.user_id = Some("alice".to_owned());
        input
            .extra
            .insert("city".to_owned(), Value::String("Paris".to_owned()));
        let fields = serde_json::json!({
            "user_id": "override",
            "city": "Lyon",
            "ignored": {"nested": true},
        });
        let Some(fields) = fields.as_object() else {
            panic!("the fixture is an object");
        };
        assert_eq!(
            build_identity(&config, &input, Some(fields)),
            "override / Lyon"
        );
    }

    #[test]
    fn unknown_placeholders_stay_literal_with_memory_variables() {
        let config = ReActAgentConfig {
            prompt_template: vec![BaseMessage::system("{{sys_long_term_memory}} {{missing}}")],
            ..ReActAgentConfig::new()
        };
        let fields = serde_json::json!({"sys_long_term_memory": "[]"});
        let Some(fields) = fields.as_object() else {
            panic!("the fixture is an object");
        };
        assert_eq!(
            build_identity(&config, &AgentInput::from_text("q"), Some(fields)),
            "[] {{missing}}"
        );
    }
}
