//! Window assembly and tool-pair helpers.
//!
//! A *dialogue round* is Python's unit of eviction: a contiguous user group,
//! every tool-call assistant and tool result that follows, and the assistant
//! message that closes the round (an assistant without `tool_calls`). A round
//! ends when the next user message starts or the history ends, so an incomplete
//! trailing turn still counts as one round. System messages are transparent:
//! they are never part of a round and are never evicted.
//!
//! Round detection mirrors `ContextUtils.find_all_dialogue_round`, scanning
//! forward instead of backward but producing the same boundaries.
use std::collections::BTreeSet;
use std::ops::Range;

use openjiuwen_foundation::prompt::{BaseMessage, Role, TokenCounter};
use serde::{Deserialize, Serialize};

use crate::config::ContextConfig;
use crate::tokens::count_messages;

/// Returns the index ranges of every dialogue round in `messages`.
///
/// Ranges are ordered oldest first, never contain a leading or trailing system
/// message, and cover every non-system message exactly once. A mid-round system
/// message falls inside the span (system boundaries break nothing) but is never
/// removed by [`truncate_keeping_rounds`].
#[must_use]
pub fn dialogue_round_ranges(messages: &[BaseMessage]) -> Vec<Range<usize>> {
    let mut rounds = Vec::new();
    let mut index = 0;
    while index < messages.len() {
        if messages[index].role == Role::System {
            index += 1;
            continue;
        }
        let start = index;
        // The contiguous user group that opens the round.
        while index < messages.len() && messages[index].role == Role::User {
            index += 1;
        }
        // Everything up to the next user message, or up to the assistant that
        // answers without requesting a tool.
        while index < messages.len() {
            if messages[index].role == Role::System {
                index += 1;
                continue;
            }
            if messages[index].role == Role::User {
                break;
            }
            let closes =
                messages[index].role == Role::Assistant && messages[index].tool_calls.is_empty();
            index += 1;
            if closes {
                break;
            }
        }
        rounds.push(start..index);
    }
    rounds
}

/// Returns the start index of the `n`-th round from the end (`n = 1` is the
/// last round, clamped when fewer rounds exist).
///
/// Returns `None` when `n` is zero or the history has no rounds.
#[must_use]
pub fn find_last_n_rounds_start(messages: &[BaseMessage], n: usize) -> Option<usize> {
    if n == 0 {
        return None;
    }
    let rounds = dialogue_round_ranges(messages);
    if rounds.is_empty() {
        return None;
    }
    let target = rounds.len().saturating_sub(n);
    rounds.get(target).map(|range| range.start)
}

/// Per-request overrides for window assembly.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct WindowOptions {
    /// Maximum messages to send, system messages included.
    pub window_size: Option<usize>,
    /// Maximum dialogue rounds to keep.
    pub dialogue_rounds: Option<usize>,
}

/// Assembles the messages to send to the model.
///
/// The steps are Python's order: round truncation first (never splitting a
/// round), then message-count truncation where system messages consume the same
/// budget, then leading tool messages are stripped so a window never starts
/// mid-tool-call. The result is `system_messages` followed by the selected
/// context messages.
#[must_use]
pub fn assemble_window(
    history: &[BaseMessage],
    system_messages: &[BaseMessage],
    options: &WindowOptions,
    config: &ContextConfig,
) -> Vec<BaseMessage> {
    // (1) Round truncation.
    let mut context: Vec<BaseMessage> =
        match options.dialogue_rounds.or(config.default_window_rounds) {
            Some(0) => Vec::new(),
            Some(rounds) => match find_last_n_rounds_start(history, rounds) {
                Some(start) => history[start..].to_vec(),
                None => Vec::new(),
            },
            None => history.to_vec(),
        };

    // (2) Message-count truncation, system messages sharing the budget.
    let mut systems: Vec<BaseMessage> = system_messages.to_vec();
    if let Some(window) = options.window_size.or(config.default_window_messages) {
        let keep_systems = systems.len().min(window);
        let drop_systems = systems.len() - keep_systems;
        systems.drain(..drop_systems);
        let keep_context = window.saturating_sub(keep_systems);
        let drop_context = context.len().saturating_sub(keep_context);
        context.drain(..drop_context);
    }

    // (3) Never start a window in the middle of a tool exchange.
    strip_leading_tool_messages(&mut context);

    // (4) System messages come first.
    let mut assembled = systems;
    assembled.append(&mut context);
    assembled
}

/// Removes leading [`Role::Tool`] messages, returning how many were removed.
pub fn strip_leading_tool_messages(messages: &mut Vec<BaseMessage>) -> usize {
    let leading = messages
        .iter()
        .take_while(|message| message.role == Role::Tool)
        .count();
    if leading > 0 {
        messages.drain(..leading);
    }
    leading
}

/// Indices of [`Role::Tool`] messages whose `tool_call_id` has no matching id in
/// a *preceding* assistant message's `tool_calls`.
#[must_use]
pub fn validate_tool_pairs(messages: &[BaseMessage]) -> Vec<usize> {
    let mut known: BTreeSet<&str> = BTreeSet::new();
    let mut orphans = Vec::new();
    for (index, message) in messages.iter().enumerate() {
        match message.role {
            Role::Assistant => {
                known.extend(
                    message
                        .tool_calls
                        .iter()
                        .filter_map(|call| call.id.as_deref()),
                );
            }
            Role::Tool => {
                let matched = message
                    .tool_call_id
                    .as_deref()
                    .is_some_and(|id| known.contains(id));
                if !matched {
                    orphans.push(index);
                }
            }
            _ => {}
        }
    }
    orphans
}

/// Removes orphan tool results, returning how many were removed.
pub fn drop_orphan_tool_results(messages: &mut Vec<BaseMessage>) -> usize {
    let orphans = validate_tool_pairs(messages);
    if orphans.is_empty() {
        return 0;
    }
    let orphan_set: BTreeSet<usize> = orphans.iter().copied().collect();
    let mut index = 0;
    messages.retain(|_| {
        let keep = !orphan_set.contains(&index);
        index += 1;
        keep
    });
    orphans.len()
}

/// Drops the oldest complete rounds until `messages` fits `max_tokens`.
///
/// System messages and the newest `keep_recent` messages are never dropped, and
/// only whole rounds are removed, so tool-call/tool-result pairs stay intact by
/// construction. Returns the number of removed messages.
pub fn truncate_keeping_rounds(
    messages: &mut Vec<BaseMessage>,
    max_tokens: usize,
    counter: &dyn TokenCounter,
    keep_recent: usize,
) -> usize {
    let mut tokens = count_messages(counter, messages);
    if tokens <= max_tokens {
        return 0;
    }

    let rounds = dialogue_round_ranges(messages);
    let protected_start = messages.len().saturating_sub(keep_recent);
    let mut removed: BTreeSet<usize> = BTreeSet::new();

    for round in rounds {
        if tokens <= max_tokens {
            break;
        }
        // A round touching the protected tail (or still being written) stays.
        if round.end > protected_start {
            break;
        }
        let mut freed = 0usize;
        for index in round.clone() {
            // System messages are never evicted, even when a round spans one.
            if messages[index].role == Role::System {
                continue;
            }
            freed += crate::tokens::count_message(counter, &messages[index]);
            removed.insert(index);
        }
        if freed == 0 {
            // Nothing removable in this round: continue with the next one
            // instead of spinning on it.
            continue;
        }
        tokens = tokens.saturating_sub(freed);
    }

    if removed.is_empty() {
        return 0;
    }
    let removed_count = removed.len();
    let mut index = 0;
    messages.retain(|_| {
        let keep = !removed.contains(&index);
        index += 1;
        keep
    });
    removed_count
}

#[cfg(test)]
mod tests {
    use super::*;
    use openjiuwen_foundation::prompt::{ApproxTokenCounter, ToolCall};

    fn counter() -> ApproxTokenCounter {
        ApproxTokenCounter::with_chars_per_token(1)
    }

    fn tool_call_assistant(id: &str, tool: &str) -> BaseMessage {
        BaseMessage::assistant("").with_tool_calls(vec![ToolCall::new(tool).with_id(id)])
    }

    #[test]
    fn round_ranges_cover_plain_and_incomplete_rounds() {
        let messages = vec![
            BaseMessage::user("q1"),
            BaseMessage::assistant("a1"),
            BaseMessage::user("q2"),
        ];
        assert_eq!(dialogue_round_ranges(&messages), vec![0..2, 2..3]);
        assert_eq!(find_last_n_rounds_start(&messages, 1), Some(2));
        assert_eq!(find_last_n_rounds_start(&messages, 2), Some(0));
        assert_eq!(find_last_n_rounds_start(&messages, 9), Some(0));
        assert_eq!(find_last_n_rounds_start(&messages, 0), None);
        assert_eq!(find_last_n_rounds_start(&[], 1), None);
    }

    #[test]
    fn tool_call_rounds_include_results_and_the_closing_assistant() {
        let messages = vec![
            BaseMessage::user("read"),
            tool_call_assistant("call_1", "read_file"),
            BaseMessage::tool("call_1", "contents"),
            BaseMessage::assistant("done"),
            BaseMessage::user("next"),
        ];
        assert_eq!(dialogue_round_ranges(&messages), vec![0..4, 4..5]);

        // A trailing tool exchange without a final assistant is one round.
        let unfinished = vec![
            BaseMessage::user("go"),
            tool_call_assistant("call_1", "read_file"),
            BaseMessage::tool("call_1", "contents"),
        ];
        assert_eq!(dialogue_round_ranges(&unfinished), vec![0..3]);
    }

    #[test]
    fn a_user_after_tool_results_starts_a_new_round() {
        let messages = vec![
            BaseMessage::user("q1"),
            tool_call_assistant("call_1", "read_file"),
            BaseMessage::tool("call_1", "contents"),
            BaseMessage::user("q2"),
            BaseMessage::assistant("a2"),
        ];
        assert_eq!(dialogue_round_ranges(&messages), vec![0..3, 3..5]);
    }

    #[test]
    fn system_messages_are_transparent() {
        let messages = vec![
            BaseMessage::system("persona"),
            BaseMessage::user("q1"),
            BaseMessage::assistant("a1"),
            BaseMessage::system("reminder"),
            BaseMessage::user("q2"),
            BaseMessage::assistant("a2"),
        ];
        assert_eq!(dialogue_round_ranges(&messages), vec![1..3, 4..6]);
    }

    #[test]
    fn assemble_applies_round_then_message_budgets() {
        let history = vec![
            BaseMessage::user("q1"),
            BaseMessage::assistant("a1"),
            BaseMessage::user("q2"),
            BaseMessage::assistant("a2"),
            BaseMessage::user("q3"),
            BaseMessage::assistant("a3"),
        ];
        let config = ContextConfig::default();
        let systems = vec![BaseMessage::system("persona")];

        let one_round = assemble_window(
            &history,
            &systems,
            &WindowOptions {
                dialogue_rounds: Some(1),
                window_size: None,
            },
            &config,
        );
        assert_eq!(one_round.len(), 3);
        assert_eq!(one_round[0].role, Role::System);
        assert_eq!(one_round[1].text(), "q3");

        // A window of two messages leaves room for one system message and one
        // context message from the tail.
        let narrow = assemble_window(
            &history,
            &systems,
            &WindowOptions {
                dialogue_rounds: None,
                window_size: Some(2),
            },
            &config,
        );
        assert_eq!(narrow.len(), 2);
        assert_eq!(narrow[0].role, Role::System);
        assert_eq!(narrow[1].text(), "a3");

        // Zero rounds keeps only the system messages.
        let none = assemble_window(
            &history,
            &systems,
            &WindowOptions {
                dialogue_rounds: Some(0),
                window_size: None,
            },
            &config,
        );
        assert_eq!(none, systems);
    }

    #[test]
    fn assemble_honours_the_config_defaults_and_strips_leading_tools() {
        let history = vec![
            BaseMessage::tool("call_1", "orphan"),
            BaseMessage::user("q1"),
            BaseMessage::assistant("a1"),
        ];
        let config = ContextConfig {
            default_window_messages: Some(3),
            default_window_rounds: Some(1),
            ..ContextConfig::default()
        };
        let assembled = assemble_window(&history, &[], &WindowOptions::default(), &config);
        assert_eq!(assembled.len(), 2);
        assert_eq!(assembled[0].text(), "q1");

        let mut with_leading_tools = vec![
            BaseMessage::tool("a", "1"),
            BaseMessage::tool("b", "2"),
            BaseMessage::user("q"),
        ];
        assert_eq!(strip_leading_tool_messages(&mut with_leading_tools), 2);
        assert_eq!(with_leading_tools.len(), 1);
    }

    #[test]
    fn orphan_tool_results_are_detected_and_dropped() {
        let mut messages = vec![
            BaseMessage::user("q"),
            tool_call_assistant("call_1", "read_file"),
            BaseMessage::tool("call_1", "ok"),
            BaseMessage::tool("call_missing", "stray"),
            BaseMessage::tool("call_2", "unpaired"),
            BaseMessage::assistant("done"),
        ];
        assert_eq!(validate_tool_pairs(&messages), vec![3, 4]);
        assert_eq!(drop_orphan_tool_results(&mut messages), 2);
        assert_eq!(messages.len(), 4);
        assert_eq!(validate_tool_pairs(&messages), Vec::<usize>::new());
    }

    #[test]
    fn truncation_keeps_system_and_recent_and_stays_under_budget() {
        let mut messages = vec![
            BaseMessage::system("persona"),
            BaseMessage::user("q1"),
            BaseMessage::assistant("a1"),
            BaseMessage::user("q2"),
            BaseMessage::assistant("a2"),
            BaseMessage::user("q3"),
            BaseMessage::assistant("a3"),
        ];
        let counter = counter();
        // Room for the persona, the second round, and the protected third one.
        let budget =
            count_messages(&counter, &messages[3..]) + count_messages(&counter, &messages[..1]);
        let removed = truncate_keeping_rounds(&mut messages, budget, &counter, 2);
        assert_eq!(removed, 2);
        assert_eq!(messages.len(), 5);
        assert_eq!(messages[0].role, Role::System);
        assert_eq!(messages[1].text(), "q2");
        assert_eq!(messages[2].text(), "a2");
        assert_eq!(messages[3].text(), "q3");
        assert!(count_messages(&counter, &messages) <= budget);
        assert_eq!(validate_tool_pairs(&messages), Vec::<usize>::new());
    }

    #[test]
    fn truncation_never_drops_the_protected_tail_or_loops() {
        let mut messages = vec![BaseMessage::user("q1"), BaseMessage::assistant("a1")];
        let counter = counter();
        // Everything is inside the protected tail.
        assert_eq!(truncate_keeping_rounds(&mut messages, 1, &counter, 2), 0);
        assert_eq!(messages.len(), 2);

        // With no protection the whole history can go, and the call terminates.
        assert_eq!(truncate_keeping_rounds(&mut messages, 1, &counter, 0), 2);
        assert_eq!(messages.len(), 0);

        // Already inside the budget is a no-op.
        let mut fits = vec![BaseMessage::user("q")];
        assert_eq!(truncate_keeping_rounds(&mut fits, 100, &counter, 0), 0);
    }

    #[test]
    fn truncation_drops_whole_tool_rounds_without_orphans() {
        let mut messages = vec![
            BaseMessage::user("q1"),
            tool_call_assistant("call_1", "read_file"),
            BaseMessage::tool("call_1", "contents"),
            BaseMessage::assistant("a1"),
            BaseMessage::user("q2"),
            BaseMessage::assistant("a2"),
        ];
        let counter = counter();
        let removed = truncate_keeping_rounds(&mut messages, 12, &counter, 0);
        assert_eq!(removed, 4);
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].text(), "q2");
        assert_eq!(validate_tool_pairs(&messages), Vec::<usize>::new());
    }
}
