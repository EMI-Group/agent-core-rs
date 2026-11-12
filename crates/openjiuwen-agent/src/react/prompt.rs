//! Identity prompt rendering for the `ReAct` agent.
//!
//! The identity section is the system-role text of
//! [`ReActAgentConfig::prompt_template`], joined with a blank line and rendered
//! against the invocation's template variables: the caller's extra keys plus
//! `query`, `conversation_id`, `user_id`, and `run_kind` when present.
use crate::input::AgentInput;
use crate::react::config::ReActAgentConfig;
use openjiuwen_foundation::{PromptTemplate, TemplateContent};
use serde_json::Value;
use std::collections::BTreeMap;

/// Renders the identity section of the system prompt for one invocation.
///
/// Returns an empty string when the template has no system-role messages.
/// `{{placeholder}}` variables are resolved with
/// [`PromptTemplate::format`](openjiuwen_foundation::PromptTemplate); unknown
/// placeholders are left untouched.
#[allow(dead_code)] // used by the loop's prompt construction
pub(crate) fn build_identity(config: &ReActAgentConfig, input: &AgentInput) -> String {
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

    let rendered = PromptTemplate::new(raw.clone()).format(&values);
    match rendered.content {
        TemplateContent::Text(text) => text,
        // `raw` came from text messages, so this variant is unreachable in
        // practice; falling back to the unrendered text keeps it usable.
        TemplateContent::Messages(_) => raw,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openjiuwen_foundation::BaseMessage;

    #[test]
    fn no_template_renders_empty() {
        let config = ReActAgentConfig::new();
        assert_eq!(build_identity(&config, &AgentInput::default()), "");
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
        assert_eq!(build_identity(&config, &input), "hello bob from Paris");
    }

    #[test]
    fn unknown_placeholder_is_left_untouched() {
        let config = ReActAgentConfig {
            prompt_template: vec![BaseMessage::system("hi {{missing}}")],
            ..ReActAgentConfig::new()
        };
        assert_eq!(
            build_identity(&config, &AgentInput::from_text("q")),
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
            build_identity(&config, &AgentInput::default()),
            "system text"
        );
    }
}
