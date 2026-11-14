//! Prompts and lenient JSON extraction for adaptive compression decisions.
//!
//! The strategy request is deliberately small: it carries the measured view of
//! [`AdaptiveContext`] (tokens, rounds, tool results) plus the memory-block JSON
//! schema, so one bounded, non-streaming model call can both pick the pass and
//! extract the durable task state.
//!
//! Replies are parsed leniently. A model that wraps the object in a fenced code
//! block, prefixes it with prose, or adds unknown keys still yields the object;
//! [`extract_json_object`] scans for the first balanced, valid JSON object.
use std::fmt::Write as _;

use super::plan::MemoryBlockPlan;
use super::{AdaptiveContext, CompressionStrategy};

/// System prompt of the strategy chooser.
///
/// The prompt is part of the wire contract: it names the six strategies, asks
/// for exactly one JSON object, and forbids tool calls, so the caller can parse
/// the reply without a streaming parser.
pub const STRATEGY_SYSTEM_PROMPT: &str = concat!(
    "You are the compression strategy selector of an LLM context engine.\n",
    "Choose exactly one compression pass for the conversation state in the user message.\n",
    "Answer with exactly one JSON object and nothing else: no prose, no code fence, no tool calls.\n",
    "{\"strategy\": \"<name>\", \"reason\": \"<short justification>\", ",
    "\"summary\": \"<durable state summary>\", \"tasks\": [\"<pending task>\"], ",
    "\"decisions\": [\"<decision>\"], \"constraints\": [\"<constraint>\"], ",
    "\"files\": [\"<path>\"], \"open_questions\": [\"<question>\"]}\n",
    "Strategies:\n",
    "- dialogue_summary: replace older completed dialogue rounds with one memory block.\n",
    "- current_round: compress the earlier part of the in-flight round.\n",
    "- round_level: build per-round blocks and merge them.\n",
    "- full_compact: summarise the whole history down to its boundary.\n",
    "- micro_compact: clear the bodies of stale tool results in place.\n",
    "- noop: nothing in this history is worth compressing yet.\n",
    "Every list is optional; omit it or send an empty list when the history has nothing to report.\n",
    "Preserve exact file paths, identifiers, decisions and open questions. Never invent details.\n",
    "Never include API keys, tokens, passwords or other secrets.\n",
);

/// Builds the user message of the strategy request.
///
/// Every measured field of `context` is rendered on its own line, the allowlist
/// is spelled out (an empty allowlist permits every strategy), and the
/// memory-block schema is appended so the reply's structured sections match
/// [`MemoryBlockPlan`].
#[must_use]
pub fn strategy_request(context: &AdaptiveContext, allowlist: &[CompressionStrategy]) -> String {
    let mut prompt = String::new();
    let _ = writeln!(prompt, "Measured history:");
    let _ = writeln!(prompt, "- total tokens: {}", context.total_tokens);
    let _ = writeln!(prompt, "- budget tokens: {}", context.budget_tokens);
    let _ = writeln!(prompt, "- trigger tokens: {}", context.trigger_tokens);
    let _ = writeln!(
        prompt,
        "- messages: {} (system {}, user {}, assistant {}, tool {})",
        context.messages,
        context.system_messages,
        context.user_messages,
        context.assistant_messages,
        context.tool_messages
    );
    let _ = writeln!(
        prompt,
        "- completed rounds: {} holding {} tokens",
        context.completed_rounds, context.completed_round_tokens
    );
    let _ = writeln!(
        prompt,
        "- assistant turn in flight: {} holding {} tokens",
        if context.assistant_in_flight {
            "yes"
        } else {
            "no"
        },
        context.in_flight_round_tokens
    );
    let _ = writeln!(
        prompt,
        "- tool results: {} holding {} tokens, stale {} holding {} tokens",
        context.tool_result_count,
        context.tool_result_tokens,
        context.stale_tool_result_count,
        context.stale_tool_result_tokens
    );
    let _ = writeln!(
        prompt,
        "- memory blocks already present: {}",
        context.memory_blocks
    );
    let _ = writeln!(
        prompt,
        "- recent compressions: {}, last strategy: {}",
        context.recent_compressions,
        context.last_strategy.map_or_else(
            || "none".to_owned(),
            |strategy| strategy.as_str().to_owned()
        )
    );
    let allowed = if allowlist.is_empty() {
        "every strategy".to_owned()
    } else {
        allowlist
            .iter()
            .map(|strategy| strategy.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    };
    let _ = writeln!(prompt, "Allowed strategies: {allowed}.");
    let schema = MemoryBlockPlan::json_schema();
    let _ = writeln!(
        prompt,
        "Memory block JSON schema: {}",
        serde_json::to_string(&schema).unwrap_or_else(|_| "{}".to_owned())
    );
    let _ = writeln!(prompt, "Reply with one JSON object only.");
    prompt
}

/// Returns the first balanced JSON object inside `text`.
///
/// The scan is tolerant of fenced code blocks (` ```json `), leading prose and
/// trailing commentary: every `{` is tried in order, its matching `}` found with
/// string/escape awareness, and the slice accepted only when it parses as a JSON
/// object. `None` means the reply carries no object at all.
#[must_use]
pub fn extract_json_object(text: &str) -> Option<&str> {
    let mut index = 0;
    while let Some(offset) = text.get(index..)?.find('{') {
        let start = index + offset;
        if let Some(end) = matching_brace(text, start) {
            if let Some(candidate) = text.get(start..=end) {
                if serde_json::from_str::<serde_json::Value>(candidate).is_ok() {
                    return Some(candidate);
                }
            }
        }
        index = start + 1;
    }
    None
}

/// Byte offset of the `}` closing the object opened at `start`.
fn matching_brace(text: &str, start: usize) -> Option<usize> {
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (offset, ch) in text.get(start..)?.char_indices() {
        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        match ch {
            '"' => in_string = true,
            '{' => depth += 1,
            '}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return Some(start + offset);
                }
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fenced_and_chatty_replies_yield_the_object() {
        let fenced = "Sure, here it is:\n```json\n{\"strategy\": \"noop\"}\n```\nThanks!";
        assert_eq!(
            extract_json_object(fenced),
            Some("{\"strategy\": \"noop\"}")
        );
        let prose = "I think {\"strategy\": \"micro_compact\", \"reason\": \"a {brace} in a string\"} is best.";
        assert_eq!(
            extract_json_object(prose),
            Some("{\"strategy\": \"micro_compact\", \"reason\": \"a {brace} in a string\"}")
        );
        assert_eq!(extract_json_object("no json here"), None);
        assert_eq!(extract_json_object("{\"unbalanced\": true"), None);
        assert_eq!(extract_json_object("{\"not json\": tru}"), None);
    }
}
