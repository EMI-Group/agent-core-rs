//! Shared support for the compressor family.
//!
//! The compressors replace spans of a conversation with a *memory block*: a
//! user-role message whose text starts with a stable marker and carries a fixed
//! envelope telling the model that the content is historical reference memory
//! rather than a fresh instruction. Ports the protocolised block markers of
//! Python's `processor/compressor/` (`_wrap_memory_block`, the summary markers)
//! without transliterating the individual compressors.
//!
//! This module owns the pieces every compressor needs:
//!
//! - [`MemoryBlockKind`] and its [`MemoryBlockKind::render`] envelope.
//! - [`group_completed_rounds`] — the round unit a compressor may replace, and
//!   [`find_last_completed_round_end`] to clamp a span back to a round boundary.
//! - [`iter_summary_merge_ranges`] — contiguous block runs a merge stage can
//!   collapse further.
//! - [`summarize_offline`] and [`summarize`] — the deterministic fallback and
//!   the (optional) model-backed summary. Neither ever fails, so the pipeline
//!   keeps working offline.
pub mod compact;
pub mod current_round;
pub mod dialogue;
pub mod micro_compact;
pub mod round_level;

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::ops::Range;
use std::sync::Arc;

use openjiuwen_foundation::llm::{Model, ModelRequestConfig};
use openjiuwen_foundation::prompt::{BaseMessage, Role};

pub use self::compact::{
    FullCompactProcessor, FullCompactProcessorConfig, FULL_COMPACT_BOUNDARY,
    FULL_COMPACT_BOUNDARY_TEXT, FULL_COMPACT_SUMMARY_INTRO,
};
pub use self::current_round::{CurrentRoundCompressor, CurrentRoundCompressorConfig};
pub use self::dialogue::{DialogueCompressor, DialogueCompressorConfig};
pub use self::micro_compact::{
    MicroCompactProcessor, MicroCompactProcessorConfig, MICRO_COMPACT_CLEARED_MARKER,
};
pub use self::round_level::{
    RoundLevelCompressor, RoundLevelCompressorConfig, TRUNCATED_FULL_CONTEXT_SCOPE,
};

/// Marker that opens a historical dialogue memory block.
pub const DIALOGUE_MEMORY_BLOCK: &str = "[DIALOGUE_MEMORY_BLOCK]";

/// Marker that opens a current-round memory block.
pub const CURRENT_ROUND_MEMORY_BLOCK: &str = "[CURRENT_ROUND_MEMORY_BLOCK]";

/// Marker that opens a round-level fallback memory block.
pub const ROUND_LEVEL_MEMORY_BLOCK: &str = "[ROUND_LEVEL_MEMORY_BLOCK]";

/// Inserted between the head and the tail of a truncated offline summary.
pub const TRUNCATED_MARKER: &str = "...[TRUNCATED]...";

/// Separator between merged memory-block bodies.
pub const SUMMARY_JOIN_SEPARATOR: &str = "\n---\n";

/// Which compressor produced a memory block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryBlockKind {
    /// Written by the dialogue compressor.
    Dialogue,
    /// Written by the current-round compressor.
    CurrentRound,
    /// Written by the round-level fallback compressor.
    RoundLevel,
}

impl MemoryBlockKind {
    /// The marker this block starts with.
    #[must_use]
    pub fn marker(self) -> &'static str {
        match self {
            Self::Dialogue => DIALOGUE_MEMORY_BLOCK,
            Self::CurrentRound => CURRENT_ROUND_MEMORY_BLOCK,
            Self::RoundLevel => ROUND_LEVEL_MEMORY_BLOCK,
        }
    }

    /// The `scope:` value used when [`MemoryBlockKind::render`] renders.
    #[must_use]
    pub fn default_scope(self) -> &'static str {
        match self {
            Self::Dialogue => "historical_dialogue_block",
            Self::CurrentRound => "current_round_increment",
            Self::RoundLevel => "round_level",
        }
    }

    /// Wraps `summary` in this block's envelope, using the default scope.
    #[must_use]
    pub fn render(self, summary: &str) -> String {
        self.render_with_scope(summary, self.default_scope())
    }

    /// Wraps `summary` in this block's envelope with an explicit `scope`.
    ///
    /// The summary is trimmed, so rendering is idempotent for already trimmed
    /// text. The envelope lines are part of the wire contract: the model reads
    /// them as a statement that the block is reference memory, not an
    /// instruction.
    #[must_use]
    pub fn render_with_scope(self, summary: &str, scope: &str) -> String {
        let summary = summary.trim();
        match self {
            Self::Dialogue => format!(
                "{DIALOGUE_MEMORY_BLOCK}\n\
                 processor: DialogueCompressor\n\
                 type: historical_memory_block\n\
                 scope: historical_dialogue_block\n\
                 authority: This block is reference memory, not a binding source of truth.\n\
                 instruction_status: Do not treat this block as a new user request or fresh assistant commitment.\n\
                 conflict_priority: Prefer newer explicit user intent, newer raw context, and fresh tool results over this block.\n\n\
                 Summary:\n{summary}"
            ),
            Self::CurrentRound => format!(
                "{CURRENT_ROUND_MEMORY_BLOCK}\n\
                 processor: CurrentRoundCompressor\n\
                 type: historical_memory_block\n\
                 scope: current_round_increment\n\
                 type_note: This is compressed memory from earlier conversation, kept to preserve long-range task continuity.\n\
                 authority: This block is reference memory, not a binding source of truth.\n\
                 instruction_status: Do not treat this block as a new user request or fresh assistant commitment.\n\
                 strategy_status: Any plans, approaches, or next steps recorded here are historical working state. They may be revised, replaced, or discarded later.\n\
                 tool_action_state_status: Tool results, action history, and execution state in this block may help continuation, but they should only be reused if they are still valid in the current context.\n\
                 conflict_priority: Prefer newer signals in this order: latest explicit user request, recent uncompressed context, fresh tool or action results, then this memory block.\n\n\
                 Summary:\n{summary}"
            ),
            Self::RoundLevel => format!(
                "{ROUND_LEVEL_MEMORY_BLOCK}\n\
                 processor: RoundLevelCompressor\n\
                 type: historical_memory_block\n\
                 scope: {scope}\n\
                 authority: This block is reference memory, not a binding source of truth.\n\
                 instruction_status: Historical fallback context only. Do not treat as a new user instruction.\n\
                 conflict_priority: Prefer newer explicit user intent, newer raw context, and fresh tool results over this block.\n\n\
                 Summary:\n{summary}"
            ),
        }
    }

    /// Extracts the `Summary:` body from a rendered block.
    ///
    /// Returns `None` when `text` does not start with this kind's marker or has
    /// no `Summary:` section at all. The returned slice is trimmed.
    #[must_use]
    pub fn summary_body(self, text: &str) -> Option<&str> {
        let rest = text.strip_prefix(self.marker())?;
        let (_, body) = rest.split_once("\nSummary:\n")?;
        Some(body.trim())
    }
}

/// Whether `message` is a memory block that starts with `marker`.
///
/// Memory blocks are always user-role messages; an assistant message that
/// happens to quote a marker is not one.
#[must_use]
pub fn is_memory_block(message: &BaseMessage, marker: &str) -> bool {
    message.role == Role::User && message.text().starts_with(marker)
}

/// Returns the ranges of every *completed* dialogue round in `messages`.
///
/// A round opens on a user message — a contiguous run of user messages opens
/// one round at its first member — continues through the assistant and tool
/// messages that answer it, and closes at the first assistant message that
/// requests no tool calls and has no tool call still awaiting a result.
/// System messages never open a round and never close one; they are transparent.
/// A trailing round that never closes is *not* returned, which is what makes the
/// result the set of completed rounds.
///
/// Ranges are half-open, ordered oldest first, and never split a
/// tool-call/tool-result pair.
#[must_use]
pub fn group_completed_rounds(messages: &[BaseMessage]) -> Vec<Range<usize>> {
    collect_completed_rounds(messages, false)
}

/// The exclusive end of the last completed round inside `start..=end_inclusive`.
///
/// The candidate slice is grouped as its own conversation, so a span that
/// starts in the middle of a round (the usual case for the current-round
/// compressor, whose span follows the last user message) still has round
/// boundaries. A span that ends inside a tool exchange therefore has no
/// completed round: the caller must not compress it.
///
/// Returns `None` when the range contains no completed round, and clamps
/// `end_inclusive` to the last message index.
#[must_use]
pub fn find_last_completed_round_end(
    messages: &[BaseMessage],
    start: usize,
    end_inclusive: usize,
) -> Option<usize> {
    if start >= messages.len() || end_inclusive < start {
        return None;
    }
    let end_inclusive = end_inclusive.min(messages.len() - 1);
    let slice = &messages[start..=end_inclusive];
    let rounds = collect_completed_rounds(slice, true);
    let last = rounds.last()?;
    Some(start + last.end)
}

/// Ranges of contiguous runs of at least `min_blocks` memory blocks.
///
/// A run is broken by any message that is not a block of the given marker, so
/// the returned ranges are always consecutive indices. Used by the merge stage
/// of the current-round compressor, which collapses neighbouring blocks into
/// one.
#[must_use]
pub fn iter_summary_merge_ranges(
    messages: &[BaseMessage],
    marker: &str,
    min_blocks: usize,
) -> Vec<Range<usize>> {
    let mut ranges = Vec::new();
    let mut run_start: Option<usize> = None;
    let mut run_end: Option<usize> = None;

    for (index, message) in messages.iter().enumerate() {
        if is_memory_block(message, marker) {
            if run_start.is_none() {
                run_start = Some(index);
            }
            run_end = Some(index);
            continue;
        }
        if let (Some(begin), Some(end)) = (run_start.take(), run_end.take()) {
            if end - begin + 1 >= min_blocks {
                ranges.push(begin..end + 1);
            }
        }
    }
    if let (Some(begin), Some(end)) = (run_start, run_end) {
        if end - begin + 1 >= min_blocks {
            ranges.push(begin..end + 1);
        }
    }
    ranges
}

/// Renders a message as one transcript line.
fn transcript_line(index: usize, message: &BaseMessage) -> String {
    let mut line = format!(
        "[{}] {}: {}",
        index + 1,
        message.role.as_str(),
        message.text()
    );
    for call in &message.tool_calls {
        let _ = write!(line, " | tool_calls={}({})", call.name, call.arguments);
    }
    if let Some(id) = message.tool_call_id.as_deref() {
        let _ = write!(line, " | tool_call_id={id}");
    }
    line
}

/// Deterministic offline summary of `messages`.
///
/// Each message becomes one `[i] role: text` line (1-based index inside the
/// slice), with ` | tool_calls=name(arguments)` appended per call and
/// ` | tool_call_id=id` for tool results. When the joined text is longer than
/// `max_chars` characters, the head (~2/3 of the budget) and the tail (~1/3) are
/// kept and joined by [`TRUNCATED_MARKER`]. Empty input yields an empty string.
#[must_use]
pub fn summarize_offline(messages: &[BaseMessage], max_chars: usize) -> String {
    if messages.is_empty() {
        return String::new();
    }
    let text = messages
        .iter()
        .enumerate()
        .map(|(index, message)| transcript_line(index, message))
        .collect::<Vec<_>>()
        .join("\n");

    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= max_chars {
        return text;
    }
    let head_len = max_chars - max_chars / 3;
    let tail_len = max_chars.saturating_sub(head_len);
    let head: String = chars[..head_len].iter().collect();
    let tail: String = chars[chars.len() - tail_len..].iter().collect();
    format!("{head}\n{TRUNCATED_MARKER}\n{tail}")
}

/// Summarizes `user_prompt` with `model`, falling back to `fallback`.
///
/// A missing model, a failed call, and an empty answer all produce the fallback
/// text: a compressor must never turn a provider hiccup into a failed
/// compression pass, and the offline fallback keeps the pipeline functional
/// without credentials.
pub async fn summarize(
    model: Option<&Arc<dyn Model>>,
    system_prompt: &str,
    user_prompt: &str,
    max_tokens: u32,
    fallback: impl FnOnce() -> String,
) -> String {
    let Some(model) = model else {
        return fallback();
    };
    let request = ModelRequestConfig::default()
        .with_model_name(model.model_name())
        .with_max_tokens(max_tokens);
    let messages = [
        BaseMessage::system(system_prompt),
        BaseMessage::user(user_prompt),
    ];
    match model.invoke(&messages, &request).await {
        Ok(response) => {
            let text = response
                .message()
                .map(|message| message.text().into_owned())
                .unwrap_or_default();
            if text.trim().is_empty() {
                tracing::warn!(
                    "compressor model returned an empty summary; using the offline summary"
                );
                fallback()
            } else {
                text.trim().to_owned()
            }
        }
        Err(error) => {
            tracing::warn!(
                error = %error,
                "compressor model call failed; using the offline summary"
            );
            fallback()
        }
    }
}

/// Groups completed rounds, optionally opening one at the slice's first message.
///
/// `open_anywhere` mirrors Python's block grouping for a *candidate span*: the
/// first message of the slice opens a round even when it is not a user message.
/// Without it, only user messages open rounds, which is the public
/// [`group_completed_rounds`] contract.
fn collect_completed_rounds(messages: &[BaseMessage], open_anywhere: bool) -> Vec<Range<usize>> {
    let mut rounds = Vec::new();
    let mut start: Option<usize> = None;
    let mut pending: BTreeSet<String> = BTreeSet::new();
    let mut previous_role: Option<&Role> = None;

    for (index, message) in messages.iter().enumerate() {
        // Opening and re-anchoring happens before the message is consumed.
        if start.is_none() {
            let opens = match &message.role {
                Role::System => false,
                Role::User => true,
                _ => open_anywhere,
            };
            if opens {
                start = Some(index);
            }
        } else if message.role == Role::User
            && previous_role != Some(&Role::User)
            && pending.is_empty()
        {
            // A user turn after a completed exchange opens a new round; the
            // previous one never closed and is therefore discarded.
            start = Some(index);
        }

        match &message.role {
            Role::Assistant => {
                if message.tool_calls.is_empty() {
                    if pending.is_empty() {
                        if let Some(begin) = start.take() {
                            rounds.push(begin..index + 1);
                        }
                    }
                } else {
                    pending.extend(message.tool_calls.iter().filter_map(|call| call.id.clone()));
                }
            }
            Role::Tool => {
                if let Some(id) = message.tool_call_id.as_deref() {
                    pending.remove(id);
                }
            }
            Role::System | Role::User | Role::Other(_) => {}
        }
        previous_role = Some(&message.role);
    }
    rounds
}

#[cfg(test)]
mod tests {
    use super::*;
    use openjiuwen_foundation::prompt::ToolCall;

    fn tool_exchange(id: &str, tool: &str, output: &str) -> Vec<BaseMessage> {
        vec![
            BaseMessage::user("run"),
            BaseMessage::assistant("").with_tool_calls(vec![ToolCall::new(tool).with_id(id)]),
            BaseMessage::tool(id, output),
            BaseMessage::assistant("done"),
        ]
    }

    #[test]
    fn completed_rounds_exclude_system_and_incomplete_rounds() {
        let mut messages = vec![BaseMessage::system("persona")];
        messages.extend(tool_exchange("call_1", "read_file", "contents"));
        messages.push(BaseMessage::user("q2"));
        assert_eq!(group_completed_rounds(&messages), vec![1..5]);

        // The trailing user-only turn is incomplete.
        messages.push(BaseMessage::assistant("a2"));
        assert_eq!(group_completed_rounds(&messages), vec![1..5, 5..7]);
    }

    #[test]
    fn contiguous_user_runs_open_one_round_at_the_first_member() {
        let messages = vec![
            BaseMessage::user("q1"),
            BaseMessage::user("q2"),
            BaseMessage::assistant("a"),
        ];
        assert_eq!(group_completed_rounds(&messages), vec![0..3]);
    }

    #[test]
    fn an_abandoned_round_is_dropped() {
        let messages = vec![
            BaseMessage::user("q1"),
            BaseMessage::assistant("").with_tool_calls(vec![ToolCall::new("t").with_id("c1")]),
            BaseMessage::tool("c1", "r"),
            BaseMessage::user("q2"),
            BaseMessage::assistant("a2"),
        ];
        // The first exchange never got its closing assistant, so only the
        // second round is complete.
        assert_eq!(group_completed_rounds(&messages), vec![3..5]);
    }

    #[test]
    fn last_completed_round_end_clamps_spans() {
        let messages = vec![
            BaseMessage::user("q"),
            BaseMessage::assistant("").with_tool_calls(vec![ToolCall::new("t").with_id("c1")]),
            BaseMessage::tool("c1", "r"),
            BaseMessage::assistant("done"),
            BaseMessage::assistant("tail"),
        ];
        assert_eq!(find_last_completed_round_end(&messages, 1, 2), None);
        assert_eq!(find_last_completed_round_end(&messages, 1, 3), Some(4));
        assert_eq!(find_last_completed_round_end(&messages, 1, 4), Some(5));
        assert_eq!(find_last_completed_round_end(&messages, 9, 9), None);
        assert_eq!(find_last_completed_round_end(&[], 0, 0), None);
    }

    #[test]
    fn merge_ranges_find_contiguous_block_runs() {
        let block = |text: &str| BaseMessage::user(MemoryBlockKind::Dialogue.render(text));
        let messages = vec![
            BaseMessage::user("raw"),
            block("a"),
            block("b"),
            BaseMessage::assistant("milestone"),
            block("c"),
            block("d"),
            block("e"),
        ];
        let ranges = iter_summary_merge_ranges(&messages, DIALOGUE_MEMORY_BLOCK, 3);
        assert_eq!(ranges, vec![4..7]);
        let ranges = iter_summary_merge_ranges(&messages, DIALOGUE_MEMORY_BLOCK, 2);
        assert_eq!(ranges, vec![1..3, 4..7]);
        assert_eq!(
            iter_summary_merge_ranges(&messages, CURRENT_ROUND_MEMORY_BLOCK, 2),
            Vec::<Range<usize>>::new()
        );
    }

    #[test]
    fn offline_summary_renders_tool_metadata_and_truncates() {
        let messages = vec![
            BaseMessage::user("hello"),
            BaseMessage::assistant("").with_tool_calls(vec![ToolCall::new("read_file")
                .with_id("c1")
                .with_arguments(serde_json::json!({"path": "a"}))]),
            BaseMessage::tool("c1", "body"),
        ];
        let summary = summarize_offline(&messages, 10_000);
        assert!(summary.starts_with("[1] user: hello"));
        assert!(summary.contains("[2] assistant:  | tool_calls=read_file({\"path\":\"a\"})"));
        assert!(summary.contains("[3] tool: body | tool_call_id=c1"));

        let truncated = summarize_offline(&messages, 40);
        assert!(truncated.contains(TRUNCATED_MARKER));
        assert!(truncated.chars().count() < summary.chars().count());
        assert_eq!(summarize_offline(&[], 10), "");
    }

    #[test]
    fn envelopes_render_and_parse_back() {
        for kind in [
            MemoryBlockKind::Dialogue,
            MemoryBlockKind::CurrentRound,
            MemoryBlockKind::RoundLevel,
        ] {
            let text = kind.render("  task done  ");
            assert!(text.starts_with(kind.marker()));
            assert_eq!(kind.summary_body(&text), Some("task done"));
            assert!(!is_memory_block(
                &BaseMessage::assistant(text.clone()),
                kind.marker()
            ));
            assert!(is_memory_block(
                &BaseMessage::user(text.clone()),
                kind.marker()
            ));
        }
        let scoped = MemoryBlockKind::RoundLevel.render_with_scope("s", "custom_scope");
        assert!(scoped.contains("\nscope: custom_scope\n"));
        assert_eq!(MemoryBlockKind::Dialogue.summary_body("no envelope"), None);
    }

    #[tokio::test]
    async fn summarize_without_a_model_uses_the_fallback() {
        let text = summarize(None, "sys", "user", 10, || "fallback".to_owned()).await;
        assert_eq!(text, "fallback");
    }
}
