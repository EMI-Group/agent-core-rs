//! Conversation statistics and the prepared-context payload.
//!
//! [`ContextStats`] is the numeric view of a history: message counts and token
//! counts per role, the number of dialogue rounds, and the compression counters
//! the engine accumulates. [`PreparedContext`] bundles the messages to send with
//! those statistics and any processor events.
use openjiuwen_foundation::prompt::{BaseMessage, Role, TokenCounter};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::events::ContextEvent;
use crate::tokens::{count_message, count_messages};
use crate::usage::ContextUsageSnapshot;
use crate::window::dialogue_round_ranges;

/// Per-role message and token counts plus compression counters.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ContextStats {
    /// Total messages in the history.
    pub total_messages: usize,
    /// Total tokens in the history.
    pub total_tokens: usize,
    /// Number of dialogue rounds.
    pub total_dialogues: usize,
    /// Number of system messages.
    pub system_messages: usize,
    /// Number of user messages.
    pub user_messages: usize,
    /// Number of assistant messages.
    pub assistant_messages: usize,
    /// Number of tool messages.
    pub tool_messages: usize,
    /// Tokens in system messages.
    pub system_tokens: usize,
    /// Tokens in user messages.
    pub user_tokens: usize,
    /// Tokens in assistant messages.
    pub assistant_tokens: usize,
    /// Tokens in tool messages.
    pub tool_tokens: usize,
    /// How many compression passes ran.
    pub compressions: usize,
    /// Token count before the most recent compression.
    pub tokens_before_compression: usize,
    /// Token count after the most recent compression.
    pub tokens_after_compression: usize,
}

/// Computes the statistics of `messages`.
///
/// Roles this crate does not know are counted in the totals only, mirroring
/// Python's tolerant `role.value` checks.
#[must_use]
pub fn compute_stats(messages: &[BaseMessage], counter: &dyn TokenCounter) -> ContextStats {
    let mut stats = ContextStats {
        total_messages: messages.len(),
        total_tokens: count_messages(counter, messages),
        total_dialogues: dialogue_round_ranges(messages).len(),
        ..ContextStats::default()
    };
    for message in messages {
        let tokens = count_message(counter, message);
        match message.role {
            Role::System => {
                stats.system_messages += 1;
                stats.system_tokens += tokens;
            }
            Role::User => {
                stats.user_messages += 1;
                stats.user_tokens += tokens;
            }
            Role::Assistant => {
                stats.assistant_messages += 1;
                stats.assistant_tokens += tokens;
            }
            Role::Tool => {
                stats.tool_messages += 1;
                stats.tool_tokens += tokens;
            }
            Role::Other(_) => {}
        }
    }
    stats
}

/// The payload a context engine hands to a model call.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PreparedContext {
    /// System messages first, then the selected context messages.
    pub messages: Vec<BaseMessage>,
    /// Tool schemas passed through to the provider.
    pub tools: Vec<Value>,
    /// Statistics of the full stored history.
    pub stats: ContextStats,
    /// Tokens in [`PreparedContext::messages`].
    pub token_count: usize,
    /// Whether [`PreparedContext::token_count`] exceeds the configured budget.
    pub overflow: bool,
    /// The window size actually applied, when one was requested.
    pub window_size: Option<usize>,
    /// Processor events produced while preparing.
    pub events: Vec<ContextEvent>,
    /// The `context.usage` payload of the prepared window.
    ///
    /// `Some` only when a [`ContextUsageAnalyzer`] is configured on the engine
    /// or the context; the snapshot explains where the window's prompt tokens
    /// went and is never part of the request. Telemetry can never fail a
    /// preparation, so a missing analyzer leaves this `None` rather than
    /// returning an error.
    ///
    /// [`ContextUsageAnalyzer`]: crate::usage::ContextUsageAnalyzer
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<ContextUsageSnapshot>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use openjiuwen_foundation::prompt::{ApproxTokenCounter, ToolCall};

    fn counter() -> ApproxTokenCounter {
        ApproxTokenCounter::with_chars_per_token(1)
    }

    #[test]
    fn counts_each_role_and_the_rounds() {
        let messages = vec![
            BaseMessage::system("abc"),
            BaseMessage::user("abcd"),
            BaseMessage::assistant("").with_tool_calls(vec![ToolCall::new("read_file")]),
            BaseMessage::tool("call_1", "xy"),
            BaseMessage::assistant("ok"),
        ];
        let stats = compute_stats(&messages, &counter());
        assert_eq!(stats.total_messages, 5);
        assert_eq!(stats.system_messages, 1);
        assert_eq!(stats.user_messages, 1);
        assert_eq!(stats.assistant_messages, 2);
        assert_eq!(stats.tool_messages, 1);
        assert_eq!(stats.system_tokens, 7);
        assert_eq!(stats.user_tokens, 8);
        assert_eq!(stats.tool_tokens, 6);
        assert_eq!(
            stats.assistant_tokens,
            count_message(&counter(), &messages[2]) + count_message(&counter(), &messages[4])
        );
        assert_eq!(
            stats.total_tokens,
            stats.system_tokens + stats.user_tokens + stats.assistant_tokens + stats.tool_tokens
        );
        assert_eq!(stats.total_dialogues, 1);
    }

    #[test]
    fn unknown_roles_only_count_in_the_totals() {
        let mut custom = BaseMessage::user("abcd");
        custom.role = Role::Other("narrator".to_owned());
        let stats = compute_stats(&[custom], &counter());
        assert_eq!(stats.total_messages, 1);
        assert_eq!(stats.total_tokens, 8);
        assert_eq!(stats.user_messages, 0);
        assert_eq!(
            stats.system_tokens + stats.user_tokens + stats.assistant_tokens + stats.tool_tokens,
            0
        );
    }

    #[test]
    fn stats_and_prepared_context_round_trip() {
        let stats = compute_stats(&[BaseMessage::user("hi")], &counter());
        let json = serde_json::to_string(&stats).unwrap();
        assert_eq!(serde_json::from_str::<ContextStats>(&json).unwrap(), stats);

        let prepared = PreparedContext {
            messages: vec![BaseMessage::user("hi")],
            tools: vec![serde_json::json!({"name": "read_file"})],
            stats,
            token_count: 6,
            overflow: false,
            window_size: Some(10),
            events: Vec::new(),
            usage: None,
        };
        let json = serde_json::to_string(&prepared).unwrap();
        assert_eq!(
            serde_json::from_str::<PreparedContext>(&json).unwrap(),
            prepared
        );
        assert_eq!(
            serde_json::from_str::<PreparedContext>("{}").unwrap(),
            PreparedContext::default()
        );
    }
}
