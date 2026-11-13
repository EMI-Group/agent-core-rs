//! Prompt text and deterministic renderers for the analyzer.
//!
//! The prompts mirror the intent of the Python `memory_analysis_prompt.md` and
//! `memory_update_check.md` templates: JSON-only output, the memory kinds the store can
//! write, the redundant/conflicting/coexist definitions, and "keep the input language".
//! The renderers are pure and length-bounded so a hostile or oversized input can never
//! blow up a request.
use crate::analyzer::fact::{truncate_chars, ConversationTurn, ExtractionInput};
use crate::record::{MemoryRecord, NewMemory};

/// The memory kinds an extractor may emit, in schema order.
pub const MEMORY_KINDS: [&str; 5] = [
    "user_profile",
    "semantic_memory",
    "episodic_memory",
    "summary",
    "coding",
];

/// The JSON schema an extraction answer must follow.
pub const EXTRACTION_JSON_SCHEMA: &str = concat!(
    "{\"facts\":[{\"content\":\"...\",\"type\":\"semantic_memory\",",
    "\"importance\":0.0,\"tags\":[\"...\"],\"source\":\"...\"}]}"
);

/// System prompt for [`crate::analyzer::LlmExtractor`].
pub const EXTRACTION_SYSTEM_PROMPT: &str = concat!(
    "You extract long-term memory facts from a conversation. ",
    "Reply with JSON only, using exactly this shape and no other keys: ",
    "{\"facts\":[{\"content\":\"...\",\"type\":\"semantic_memory\",",
    "\"importance\":0.0,\"tags\":[\"...\"],\"source\":\"...\"}]}. ",
    "`type` must be one of user_profile, semantic_memory, episodic_memory, ",
    "summary, or coding. Use user_profile for durable facts about the user ",
    "(identity, preferences, constraints), semantic_memory for general facts and ",
    "project/technology knowledge, and episodic_memory for time-stamped events. ",
    "`importance` must be a number between 0.0 and 1.0, higher for facts that ",
    "matter to future conversations. `content` must be one self-contained ",
    "sentence in the language of the input. Do not invent facts, do not record ",
    "secrets or credentials, and do not add prose, Markdown fences, or explanations."
);

/// The JSON schema a conflict-classification answer must follow.
pub const CONFLICT_JSON_SCHEMA: &str = concat!(
    "{\"verdicts\":[{\"info_id\":\"...\",\"info_text\":\"...\",",
    "\"result\":\"redundant|conflicting|none\",\"related_infos\":{},",
    "\"reason\":\"...\",\"merged_content\":\"...\"}]}"
);

/// System prompt for [`crate::analyzer::LlmClassifier`].
pub const CONFLICT_SYSTEM_PROMPT: &str = concat!(
    "You are a strict information-consistency checker. Compare the NEW memory with ",
    "each EXISTING memory and classify it:\n",
    "- `redundant`: both describe the same fact about the same entity with no ",
    "information added or lost;\n",
    "- `conflicting`: both describe the same entity but state opposite things, or give ",
    "different values for a single-valued attribute (name, role, current state);\n",
    "- `none`: different entities, different attributes of one entity, or a new value ",
    "of a multi-valued attribute (hobbies, goals).\n",
    "Reply with JSON only, using exactly this shape and no other keys: ",
    "{\"verdicts\":[{\"info_id\":\"...\",\"info_text\":\"...\",",
    "\"result\":\"redundant|conflicting|none\",\"related_infos\":{},",
    "\"reason\":\"...\",\"merged_content\":\"...\"}]}. ",
    "Emit exactly one verdict per NEW memory, keep `info_id` unchanged, fill ",
    "`related_infos` with {existing_id: existing_text} for redundant or conflicting ",
    "verdicts, put the fact that should be stored in `merged_content`, and answer in ",
    "the language of the input. Do not add prose, Markdown fences, or explanations."
);

/// Renders a conversation snapshot for extraction.
///
/// The format is fixed: an optional `HISTORY` block, then a `CURRENT` block, one
/// `role: text` line per turn. Blank turns are skipped, and the result is truncated to
/// at most `max_chars` characters on a character boundary.
#[must_use]
pub fn render_conversation(input: &ExtractionInput, max_chars: usize) -> String {
    let mut rendered = String::new();
    append_section(&mut rendered, "HISTORY", &input.history);
    append_section(&mut rendered, "CURRENT", &input.turns);
    truncate_chars(rendered.trim_end(), max_chars)
}

/// Appends one labelled block of turns, skipping blank turns and empty blocks.
fn append_section(rendered: &mut String, label: &str, turns: &[ConversationTurn]) {
    let mut body = String::new();
    for turn in turns {
        if turn.is_blank() {
            continue;
        }
        body.push_str(turn.role.trim());
        body.push_str(": ");
        body.push_str(turn.text.trim());
        body.push('\n');
    }
    if body.is_empty() {
        return;
    }
    if !rendered.is_empty() {
        rendered.push('\n');
    }
    rendered.push_str(label);
    rendered.push('\n');
    rendered.push_str(&body);
}

/// Renders the existing records and the incoming memory for conflict classification.
///
/// Existing records come first, one `id: content` line each, because a verdict refers
/// back to them; the new memory comes last as `incoming: content`. Blank contents are
/// skipped and the result is truncated to at most `max_chars` characters.
#[must_use]
pub fn render_conflict_input(
    incoming: &NewMemory,
    existing: &[MemoryRecord],
    max_chars: usize,
) -> String {
    let mut rendered = String::from("EXISTING\n");
    for record in existing {
        let content = record.content.trim();
        if content.is_empty() {
            continue;
        }
        rendered.push_str(record.id.trim());
        rendered.push_str(": ");
        rendered.push_str(content);
        rendered.push('\n');
    }
    rendered.push_str("NEW\nincoming: ");
    rendered.push_str(incoming.content.trim());
    truncate_chars(rendered.trim_end(), max_chars)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::{MemoryScope, MemoryType};

    fn record(id: &str, content: &str) -> MemoryRecord {
        MemoryRecord::new(
            MemoryScope::new("alice", "ws"),
            content,
            MemoryType::SemanticMemory,
        )
        .with_id(id)
    }

    #[test]
    fn schemas_are_embedded_in_the_prompts() {
        assert!(EXTRACTION_SYSTEM_PROMPT.contains(EXTRACTION_JSON_SCHEMA));
        assert!(CONFLICT_SYSTEM_PROMPT.contains(CONFLICT_JSON_SCHEMA));
        for kind in MEMORY_KINDS {
            assert!(EXTRACTION_SYSTEM_PROMPT.contains(kind), "missing {kind}");
        }
    }

    #[test]
    fn schemas_are_valid_json() {
        let schema: serde_json::Value = serde_json::from_str(EXTRACTION_JSON_SCHEMA).unwrap();
        assert!(schema.get("facts").is_some());
        let schema: serde_json::Value = serde_json::from_str(CONFLICT_JSON_SCHEMA).unwrap();
        assert!(schema.get("verdicts").is_some());
    }

    #[test]
    fn render_conversation_includes_history_then_turns() {
        let input = ExtractionInput {
            turns: vec![ConversationTurn::new("user", "my name is Alice")],
            history: vec![
                ConversationTurn::new("user", "hi"),
                ConversationTurn::new("assistant", "hello"),
            ],
        };
        let rendered = render_conversation(&input, 4096);
        assert!(rendered.starts_with("HISTORY\nuser: hi\nassistant: hello"));
        assert!(rendered.contains("CURRENT\nuser: my name is Alice"));
        assert!(rendered.find("HISTORY").unwrap() < rendered.find("CURRENT").unwrap());
    }

    #[test]
    fn render_conversation_skips_history_and_blank_turns() {
        let input = ExtractionInput {
            turns: vec![
                ConversationTurn::new("user", "   "),
                ConversationTurn::new("user", "a fact"),
            ],
            history: Vec::new(),
        };
        let rendered = render_conversation(&input, 4096);
        assert_eq!(rendered, "CURRENT\nuser: a fact");
        assert!(!rendered.contains("HISTORY"));

        let empty = render_conversation(&ExtractionInput::default(), 4096);
        assert_eq!(empty.len(), 0);
    }

    #[test]
    fn render_conversation_is_bounded_and_char_safe() {
        let input = ExtractionInput::from_text("日本語のテキストです");
        let rendered = render_conversation(&input, 12);
        assert_eq!(rendered.chars().count(), 12);
        let rendered = render_conversation(&input, 0);
        assert_eq!(rendered.len(), 0);
        let huge = ExtractionInput::from_text("x".repeat(100_000));
        assert_eq!(render_conversation(&huge, 8192).chars().count(), 8192);
    }

    #[test]
    fn render_conflict_input_lists_existing_then_incoming() {
        let incoming = NewMemory::semantic("Alice works at Acme");
        let existing = vec![
            record("id-1", "Alice works at Globex"),
            record("id-2", "  "),
        ];
        let rendered = render_conflict_input(&incoming, &existing, 4096);
        assert_eq!(
            rendered,
            "EXISTING\nid-1: Alice works at Globex\nNEW\nincoming: Alice works at Acme"
        );
        assert!(!rendered.contains("id-2"));
    }

    #[test]
    fn render_conflict_input_is_bounded() {
        let incoming = NewMemory::semantic("x".repeat(500));
        let rendered = render_conflict_input(&incoming, &[], 20);
        assert_eq!(rendered.chars().count(), 20);
        assert_eq!(render_conflict_input(&incoming, &[], 0).len(), 0);
        let rendered = render_conflict_input(&incoming, &[], 4096);
        assert!(rendered.starts_with("EXISTING\nNEW\nincoming: x"));
    }
}
