//! Turn grouping and byte-window chunking for compression recall.
//!
//! The chunker is deliberately offline and provider-free: a window is
//! `chunk_size_tokens * 4` UTF-8 bytes (the same heuristic Python uses to avoid
//! initializing a tokenizer that may download assets), and overlapping windows
//! step back at most `chunk_overlap_tokens * 4` bytes. Characters are never
//! split, so multibyte CJK text survives chunking intact.
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::prompt::{BaseMessage, Role};
use serde::{Deserialize, Serialize};

/// Leading-text markers that identify internal user messages.
///
/// A user message whose text (after leading whitespace) starts with one of
/// these prefixes was injected by the harness rather than typed by a human, so
/// it does not open a new conversation turn.
pub const INTERNAL_USER_PREFIXES: [&str; 7] = [
    "<system-reminder>",
    "<memory_block_current>",
    "<memory_block_dialogue>",
    "<memory_block_round>",
    "<memory_block_session>",
    "<recovered_context>",
    "[STATE_REINJECTION]",
];

/// One conversation turn detected in a message history.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TurnRecord {
    /// Stable turn identifier (`turn_0001`, `turn_0002`, …).
    pub turn_id: String,
    /// The turn's user query text; empty for an implicit leading turn.
    pub query: String,
    /// Text of the turn's last assistant message; empty when there is none.
    pub answer: String,
    /// Indices into the source history that belong to this turn.
    pub message_indices: Vec<usize>,
}

/// Whether `message` was authored by a human and should open a new turn.
///
/// Only [`Role::User`] messages qualify, and their text must not begin with an
/// internal prefix after leading-whitespace trimming.
#[must_use]
pub fn is_real_user_message(message: &BaseMessage) -> bool {
    if !message.role.is_user() {
        return false;
    }
    let text = message.text();
    let trimmed = text.trim_start();
    !INTERNAL_USER_PREFIXES
        .iter()
        .any(|prefix| trimmed.starts_with(prefix))
}

/// Groups `messages` into conversation turns.
///
/// A new turn starts at every real user message; assistant messages, tool
/// calls, and tool results attach to the current turn. A history that begins
/// with an assistant or tool message forms one implicit turn with an empty
/// query. Internal-prefixed user messages also attach to the current turn.
#[must_use]
pub fn group_turns(messages: &[BaseMessage]) -> Vec<TurnRecord> {
    let mut groups: Vec<(String, Vec<usize>)> = Vec::new();
    let mut current_indices: Vec<usize> = Vec::new();
    let mut current_query = String::new();
    for (index, message) in messages.iter().enumerate() {
        if is_real_user_message(message) {
            if !current_indices.is_empty() {
                groups.push((
                    std::mem::take(&mut current_query),
                    std::mem::take(&mut current_indices),
                ));
            }
            current_query = message.text().into_owned();
        }
        current_indices.push(index);
    }
    if !current_indices.is_empty() {
        groups.push((current_query, current_indices));
    }
    groups
        .into_iter()
        .enumerate()
        .map(|(position, (query, message_indices))| {
            let answer = message_indices
                .iter()
                .rev()
                .map(|index| &messages[*index])
                .find(|message| message.role.is_assistant())
                .map(|message| message.text().into_owned())
                .unwrap_or_default();
            TurnRecord {
                turn_id: format!("turn_{:04}", position + 1),
                query,
                answer,
                message_indices,
            }
        })
        .collect()
}

/// Capitalizes the first character of a role name for a section header.
fn section_role(role: &str) -> String {
    let mut characters = role.chars();
    characters.next().map_or_else(String::new, |first| {
        first.to_uppercase().chain(characters).collect()
    })
}

/// Renders one message as zero or more transcript sections.
fn render_message(message: &BaseMessage) -> Vec<String> {
    let text = message.text();
    let text = text.as_ref();
    match &message.role {
        Role::User => vec![format!("## User\n{text}")],
        Role::Assistant => {
            let mut sections = Vec::new();
            if let Some(reasoning) = message
                .reasoning_content
                .as_deref()
                .filter(|reasoning| !reasoning.is_empty())
            {
                sections.push(format!("## Assistant Reasoning\n{reasoning}"));
            }
            if !text.is_empty() {
                sections.push(format!("## Assistant\n{text}"));
            }
            for call in &message.tool_calls {
                let name = if call.name.is_empty() {
                    "unknown"
                } else {
                    call.name.as_str()
                };
                sections.push(format!("## Tool Call: {name}\n{}", call.arguments));
            }
            sections
        }
        Role::Tool => vec![format!(
            "## Tool Result: {}\n{text}",
            message.tool_call_id.as_deref().unwrap_or_default()
        )],
        Role::System | Role::Other(_) => {
            vec![format!(
                "## {}\n{text}",
                section_role(message.role.as_str())
            )]
        }
    }
}

/// Renders the messages referenced by `message_indices` as a section transcript.
///
/// Sections are `## User`, `## Assistant Reasoning`, `## Assistant`,
/// `## Tool Call: <name>`, and `## Tool Result: <tool_call_id>` blocks separated
/// by blank lines; the result is trimmed.
#[must_use]
pub fn render_turn(messages: &[BaseMessage], message_indices: &[usize]) -> String {
    let mut sections = Vec::new();
    for index in message_indices {
        if let Some(message) = messages.get(*index) {
            sections.extend(render_message(message));
        }
    }
    sections.join("\n\n").trim().to_owned()
}

/// Estimates tokens as `(utf8_bytes + 3) / 4`.
#[must_use]
pub fn estimate_tokens(text: &str) -> usize {
    text.len().div_ceil(4)
}

/// Splits `text` into overlapping byte-window chunks.
///
/// The window is `chunk_size_tokens * 4` UTF-8 bytes and the overlap is at most
/// `chunk_overlap_tokens * 4` bytes. Characters are never split; a character
/// wider than the whole window still becomes its own one-character chunk.
///
/// # Errors
///
/// Returns [`Error::Config`] when `chunk_overlap_tokens >= chunk_size_tokens`.
pub fn split_text(
    text: &str,
    chunk_size_tokens: usize,
    chunk_overlap_tokens: usize,
) -> Result<Vec<String>> {
    if chunk_overlap_tokens >= chunk_size_tokens {
        return Err(Error::Config(format!(
            "chunk_overlap_tokens ({chunk_overlap_tokens}) must be smaller than \
             chunk_size_tokens ({chunk_size_tokens})"
        )));
    }
    if text.is_empty() {
        return Ok(Vec::new());
    }
    let byte_size = chunk_size_tokens.saturating_mul(4);
    let byte_overlap = chunk_overlap_tokens.saturating_mul(4);
    if text.len() <= byte_size {
        return Ok(vec![text.to_owned()]);
    }
    let characters: Vec<char> = text.chars().collect();
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < characters.len() {
        let mut end = start;
        let mut used_bytes = 0;
        while end < characters.len() {
            let character_bytes = characters[end].len_utf8();
            if used_bytes + character_bytes > byte_size {
                break;
            }
            used_bytes += character_bytes;
            end += 1;
        }
        if end == start {
            end += 1;
        }
        chunks.push(characters[start..end].iter().collect::<String>());
        if end >= characters.len() {
            break;
        }
        let mut next_start = end;
        let mut overlap_bytes = 0;
        while next_start > start {
            let character_bytes = characters[next_start - 1].len_utf8();
            if overlap_bytes + character_bytes > byte_overlap {
                break;
            }
            overlap_bytes += character_bytes;
            next_start -= 1;
        }
        start = if next_start > start { next_start } else { end };
    }
    Ok(chunks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use openjiuwen_foundation::prompt::ToolCall;

    #[test]
    fn groups_real_user_turns_and_attaches_tool_rounds() {
        let messages = vec![
            BaseMessage::system("persona"),
            BaseMessage::user("<system-reminder>ignore me"),
            BaseMessage::user("first question"),
            BaseMessage::assistant("calling")
                .with_tool_calls(vec![ToolCall::new("read_file").with_id("call_1")]),
            BaseMessage::tool("call_1", "file contents"),
            BaseMessage::user("  second question"),
            BaseMessage::assistant("second answer"),
        ];
        let turns = group_turns(&messages);
        assert_eq!(turns.len(), 3);
        assert_eq!(turns[0].turn_id, "turn_0001");
        assert_eq!(turns[0].query, "");
        assert_eq!(turns[0].message_indices, vec![0, 1]);
        assert_eq!(turns[1].query, "first question");
        assert_eq!(turns[1].answer, "calling");
        assert_eq!(turns[1].message_indices, vec![2, 3, 4]);
        assert_eq!(turns[2].query, "  second question");
        assert_eq!(turns[2].answer, "second answer");
    }

    #[test]
    fn leading_non_user_messages_form_an_implicit_turn() {
        let messages = vec![
            BaseMessage::assistant("prologue"),
            BaseMessage::tool("call_0", "output"),
            BaseMessage::user("question"),
        ];
        let turns = group_turns(&messages);
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0].query, "");
        assert_eq!(turns[0].answer, "prologue");
        assert_eq!(turns[0].message_indices, vec![0, 1]);
        assert_eq!(turns[1].query, "question");
    }

    #[test]
    fn renders_sections_for_every_message_kind() {
        let messages = vec![
            BaseMessage::user("ask"),
            BaseMessage::assistant("answer").with_reasoning("because"),
            BaseMessage::assistant("").with_tool_calls(vec![ToolCall::new("grep")
                .with_id("c1")
                .with_arguments(serde_json::json!({"pattern": "x"}))]),
            BaseMessage::tool("c1", "hit"),
        ];
        let rendered = render_turn(&messages, &[0, 1, 2, 3]);
        assert!(rendered.starts_with("## User\nask\n\n## Assistant Reasoning\nbecause"));
        assert!(rendered.contains("## Assistant\nanswer"));
        assert!(rendered.contains("## Tool Call: grep"));
        assert!(rendered.contains("## Tool Result: c1\nhit"));
    }

    #[test]
    fn splits_ascii_text_without_splitting_characters() {
        let chunks = split_text("abcdefghij", 2, 1).unwrap();
        assert_eq!(chunks, vec!["abcdefgh", "efghij"]);
    }

    #[test]
    fn keeps_multibyte_characters_whole() {
        let chunks = split_text("你好世界你好", 2, 0).unwrap();
        assert_eq!(chunks, vec!["你好", "世界", "你好"]);
        assert_eq!(chunks.concat(), "你好世界你好");
    }

    #[test]
    fn emits_a_single_character_when_it_is_wider_than_the_window() {
        let chunks = split_text("😀😀", 1, 0).unwrap();
        assert_eq!(chunks, vec!["😀", "😀"]);
    }

    #[test]
    fn rejects_overlap_at_or_above_the_window_size() {
        assert!(matches!(split_text("text", 10, 10), Err(Error::Config(_))));
        assert!(matches!(split_text("text", 10, 11), Err(Error::Config(_))));
        assert_eq!(split_text("", 10, 1).unwrap(), Vec::<String>::new());
    }

    #[test]
    fn token_estimate_uses_the_documented_formula() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("abc"), 1);
        assert_eq!(estimate_tokens("abcde"), 2);
        assert_eq!(estimate_tokens("你好"), 2);
    }
}
