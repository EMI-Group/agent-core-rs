//! Summarisation prompts used by the summarisation processor.
//!
//! The templates are the Python `context_engine` prompts reduced to their
//! stable parts: a system instruction and a user template with one
//! `{{conversation}}` placeholder. Rendering goes through foundation's
//! [`PromptTemplate`], so a conversation containing braces can never inject a
//! placeholder of its own.
use openjiuwen_foundation::prompt::PromptTemplate;

/// System instruction for the summarisation model call.
pub const SUMMARY_SYSTEM_PROMPT: &str = "You compress conversation history for an AI agent. \
Write a concise summary of the conversation below. Preserve: the user's requests and goals, \
decisions already made, facts and tool results the agent still needs, and tasks that remain open. \
Drop greetings, repetition, and failed attempts that led nowhere. \
Reply with the summary only — no preamble, no headings, no commentary.";

/// User template; the literal `{{conversation}}` slot receives the transcript.
pub const SUMMARY_USER_TEMPLATE: &str = "Summarise this conversation history:\n\n\
<conversation>\n{{conversation}}\n</conversation>";

/// Renders [`SUMMARY_USER_TEMPLATE`] with `conversation` substituted.
///
/// Substitution is single pass: placeholders inside the transcript are kept
/// verbatim rather than expanded.
#[must_use]
pub fn render_summary_prompt(conversation: &str) -> String {
    let template = PromptTemplate::new(SUMMARY_USER_TEMPLATE);
    let rendered = template.format_one("conversation", conversation);
    rendered
        .to_messages()
        .first()
        .map(|message| message.text().into_owned())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_the_conversation_into_the_template() {
        let rendered = render_summary_prompt("user: hi\nassistant: hello");
        assert!(rendered.contains("user: hi\nassistant: hello"));
        assert!(!rendered.contains("{{conversation}}"));
        assert!(rendered.starts_with("Summarise this conversation history:"));
    }

    #[test]
    fn placeholders_inside_the_transcript_are_not_expanded() {
        let rendered = render_summary_prompt("literal {{conversation}} inside");
        assert!(rendered.contains("literal {{conversation}} inside"));
    }

    #[test]
    fn system_prompt_describes_the_summary_contract() {
        assert!(SUMMARY_SYSTEM_PROMPT.contains("summary"));
        assert!(SUMMARY_SYSTEM_PROMPT.contains("tool results"));
        assert!(SUMMARY_SYSTEM_PROMPT.contains("open"));
    }
}
