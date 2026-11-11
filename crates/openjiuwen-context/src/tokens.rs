//! Token accounting built on foundation's [`TokenCounter`].
//!
//! This module is the context engine's seam over counting: it adds per-message
//! and per-request helpers plus a [`TokenBudget`] comparison, but every count comes
//! from `openjiuwen_foundation::prompt::TokenCounter`, so the engine, agents, and
//! providers share one definition of a token.
//!
//! Python's `len(text) // 3` fallback is available as
//! [`TokenCounterKind::Characters`]:
//!
//! ```
//! use openjiuwen_context::{TokenCounterConfig, TokenCounterKind};
//!
//! let config = TokenCounterConfig {
//!     kind: TokenCounterKind::Characters,
//!     chars_per_token: 3,
//! };
//! assert_eq!(config.chars_per_token, 3);
//! ```
//!
//! [`TokenCounterKind::Characters`]: crate::config::TokenCounterKind::Characters
use openjiuwen_foundation::prompt::{BaseMessage, TokenCounter};

/// Counts one message, including its chat framing and tool calls.
#[must_use]
pub fn count_message(counter: &dyn TokenCounter, message: &BaseMessage) -> usize {
    counter.count_messages(std::slice::from_ref(message))
}

/// Counts a whole conversation.
#[must_use]
pub fn count_messages(counter: &dyn TokenCounter, messages: &[BaseMessage]) -> usize {
    counter.count_messages(messages)
}

/// A token count compared against a budget.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct TokenBudget {
    /// Tokens used by the measured messages.
    pub tokens: usize,
    /// The budget they were measured against.
    pub budget: usize,
    /// Whether `tokens > budget`.
    pub overflow: bool,
    /// Head-room left; zero once the budget is exceeded.
    pub remaining: usize,
}

/// Measures `messages` against `budget`.
#[must_use]
pub fn check_budget(
    counter: &dyn TokenCounter,
    messages: &[BaseMessage],
    budget: usize,
) -> TokenBudget {
    let tokens = count_messages(counter, messages);
    TokenBudget {
        tokens,
        budget,
        overflow: tokens > budget,
        remaining: budget.saturating_sub(tokens),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openjiuwen_foundation::prompt::ApproxTokenCounter;

    fn counter() -> ApproxTokenCounter {
        ApproxTokenCounter::with_chars_per_token(1)
    }

    #[test]
    fn counts_include_framing_and_tool_calls() {
        let counter = counter();
        let plain = BaseMessage::user("abcd");
        assert_eq!(count_message(&counter, &plain), 4 + 4);

        let with_call = BaseMessage::assistant("").with_tool_calls(vec![
            openjiuwen_foundation::prompt::ToolCall::new("read_file").with_id("call_1"),
        ]);
        assert!(count_message(&counter, &with_call) > 4);
        assert_eq!(count_messages(&counter, &[]), 0);
    }

    #[test]
    fn budget_reports_overflow_and_saturates_remaining() {
        let counter = counter();
        let messages = vec![BaseMessage::user("abcd")];

        let fits = check_budget(&counter, &messages, 100);
        assert_eq!(fits.tokens, 8);
        assert!(!fits.overflow);
        assert_eq!(fits.remaining, 92);

        let over = check_budget(&counter, &messages, 3);
        assert!(over.overflow);
        assert_eq!(over.remaining, 0);
    }
}
