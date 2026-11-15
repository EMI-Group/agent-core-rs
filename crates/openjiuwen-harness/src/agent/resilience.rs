//! The bounded tool-call retry budget behind
//! [`DeepAgentConfig::enable_tool_resilience_rail`].
//!
//! Python parity: `openjiuwen/harness/rails/tool_call_resilience_rail.py`
//! (`ToolCallResilienceRail`, priority 70). Python implements the retry as a
//! *rail*: its `on_tool_exception` hook decides whether a failure is retryable and
//! calls `ctx.request_retry()`, which the kernel decorator around the tool call
//! consumes.
//!
//! The Rust kernel retries inside the tool execution path instead
//! ([`openjiuwen_agent::policy::ToolResiliencePolicy`], consumed by
//! `execute_calls_with`), and never fires `RailEvent::OnToolException` from the
//! loop — a rail hook there would be dead. This module therefore maps the same
//! decision onto the policy the kernel really honours:
//!
//! - the attempt budget is [`DEFAULT_MAX_ATTEMPTS`] per tool call (Python's
//!   `DEFAULT_MAX_ATTEMPTS = 3`) with an exponential [`DEFAULT_INITIAL_BACKOFF`];
//! - a non-idempotent tool never retries (Python layer 0): re-running a write, a
//!   shell command or a sub-agent spawn after a failure would repeat the side
//!   effect;
//! - the kernel's own retryability rule applies to the rest: timeouts and tool
//!   errors retry, cancellation and validation errors do not;
//! - `enable_tool_resilience_rail = false` installs an empty policy, which is the
//!   kernel's single-attempt, no-per-tool-timeout behaviour.
use std::time::Duration;

use openjiuwen_agent::policy::{ToolResilience, ToolResiliencePolicy};
use openjiuwen_agent::ReActAgent;
use openjiuwen_foundation::tool::ToolCard;

use crate::schema::DeepAgentConfig;

/// Attempt budget of one tool call, counting the first try.
///
/// Python parity: `ToolCallResilienceRail.DEFAULT_MAX_ATTEMPTS`.
pub(crate) const DEFAULT_MAX_ATTEMPTS: u32 = 3;

/// Delay before the first retry of a tool call; doubled per further attempt.
pub(crate) const DEFAULT_INITIAL_BACKOFF: Duration = Duration::from_millis(100);

/// Builds the per-tool retry budgets of `config` for the mounted `cards`.
///
/// An empty policy when [`DeepAgentConfig::enable_tool_resilience_rail`] is off;
/// otherwise the default budget applies to every tool except the non-idempotent
/// ones, which get the single-attempt budget.
#[must_use]
pub(crate) fn tool_resilience_policy(
    config: &DeepAgentConfig,
    cards: &[ToolCard],
) -> ToolResiliencePolicy {
    let mut policy = ToolResiliencePolicy::new();
    if !config.enable_tool_resilience_rail {
        return policy;
    }
    policy.set_default(
        ToolResilience::new()
            .with_attempts(DEFAULT_MAX_ATTEMPTS)
            .with_backoff(DEFAULT_INITIAL_BACKOFF),
    );
    for card in cards.iter().filter(|card| !card.idempotent) {
        policy.set(card.name.clone(), ToolResilience::new());
    }
    policy
}

/// Installs the retry budget of `config` on the inner kernel agent.
///
/// Idempotent: the policy is plain data, so re-applying it after a tool mount or a
/// reconfiguration replaces the previous budgets instead of accumulating them.
pub(crate) fn apply(inner: &ReActAgent, config: &DeepAgentConfig, cards: &[ToolCard]) {
    inner.set_tool_resilience(tool_resilience_policy(config, cards));
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use openjiuwen_foundation::tool::ToolCard;

    use super::{tool_resilience_policy, DEFAULT_INITIAL_BACKOFF, DEFAULT_MAX_ATTEMPTS};
    use crate::schema::DeepAgentConfig;

    /// Builds a card with the given name and idempotency.
    fn card(name: &str, idempotent: bool) -> ToolCard {
        ToolCard {
            name: name.to_string(),
            idempotent,
            ..ToolCard::default()
        }
    }

    #[test]
    fn the_default_configuration_retries_retryable_tools() {
        let cards = vec![card("read_file", true), card("bash", true)];
        let policy = tool_resilience_policy(&DeepAgentConfig::default(), &cards);
        assert!(!policy.is_empty());
        let budget = policy.for_tool("read_file");
        assert_eq!(budget.attempts(), DEFAULT_MAX_ATTEMPTS);
        assert_eq!(budget.initial_backoff, Some(DEFAULT_INITIAL_BACKOFF));
        assert_eq!(budget.backoff_for(1), Some(DEFAULT_INITIAL_BACKOFF));
    }

    #[test]
    fn non_idempotent_tools_are_never_retried() {
        let cards = vec![card("read_file", true), card("write_file", false)];
        let policy = tool_resilience_policy(&DeepAgentConfig::default(), &cards);
        assert_eq!(policy.for_tool("write_file").attempts(), 1);
        assert_eq!(policy.for_tool("write_file").initial_backoff, None);
        assert_eq!(
            policy.for_tool("read_file").attempts(),
            DEFAULT_MAX_ATTEMPTS
        );
    }

    #[test]
    fn the_switch_off_installs_an_empty_policy() {
        let cards = vec![card("read_file", true)];
        let config = DeepAgentConfig::default().with_enable_tool_resilience_rail(false);
        let policy = tool_resilience_policy(&config, &cards);
        assert!(policy.is_empty());
        assert_eq!(policy.for_tool("read_file").attempts(), 1);
    }
}
