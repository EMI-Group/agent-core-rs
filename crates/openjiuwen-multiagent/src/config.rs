//! Team configuration: member and message limits.
//!
//! Python parity: `TeamConfig` in `openjiuwen/core/multi_agent/team.py`, with
//! the member cap (`max_agents`), the message-queue capacity
//! (`max_concurrent_messages`), and the chain deadline (`message_timeout`,
//! seconds).

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Default cap on the number of member agents in one team.
const DEFAULT_MAX_AGENTS: usize = 10;
/// Default capacity of a per-topic message queue.
const DEFAULT_MAX_CONCURRENT_MESSAGES: usize = 100;
/// Default deadline for one message dispatch, in seconds.
const DEFAULT_MESSAGE_TIMEOUT: f64 = 30.0;

/// Limits applied to a team's membership and messaging.
///
/// Every field has a Python-compatible default (`10` / `100` / `30.0`), and
/// unknown JSON fields are tolerated and dropped on deserialization.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct TeamConfig {
    /// Maximum number of member agents; further additions are rejected.
    pub max_agents: usize,
    /// Capacity of each message-queue topic.
    pub max_concurrent_messages: usize,
    /// Deadline for one message dispatch, in seconds.
    pub message_timeout: f64,
}

impl Default for TeamConfig {
    fn default() -> Self {
        Self {
            max_agents: DEFAULT_MAX_AGENTS,
            max_concurrent_messages: DEFAULT_MAX_CONCURRENT_MESSAGES,
            message_timeout: DEFAULT_MESSAGE_TIMEOUT,
        }
    }
}

impl TeamConfig {
    /// Creates the configuration with the Python defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the member cap.
    #[must_use]
    pub fn configure_max_agents(mut self, max_agents: usize) -> Self {
        self.max_agents = max_agents;
        self
    }

    /// Sets the per-topic message-queue capacity.
    #[must_use]
    pub fn configure_concurrency(mut self, max_concurrent_messages: usize) -> Self {
        self.max_concurrent_messages = max_concurrent_messages;
        self
    }

    /// Sets the dispatch deadline, in seconds.
    #[must_use]
    pub fn configure_timeout(mut self, message_timeout: f64) -> Self {
        self.message_timeout = message_timeout;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_python() {
        let config = TeamConfig::default();
        assert_eq!(config.max_agents, 10);
        assert_eq!(config.max_concurrent_messages, 100);
        assert_eq!(config.message_timeout, 30.0);
        assert_eq!(TeamConfig::new(), config);
    }

    #[test]
    fn json_uses_snake_case_keys() {
        let value = serde_json::to_value(TeamConfig::new()).unwrap();
        assert_eq!(
            value,
            serde_json::json!({
                "max_agents": 10,
                "max_concurrent_messages": 100,
                "message_timeout": 30.0
            })
        );
    }

    #[test]
    fn empty_object_deserializes_to_defaults() {
        let config: TeamConfig = serde_json::from_value(serde_json::json!({})).unwrap();
        assert_eq!(config, TeamConfig::default());
    }

    #[test]
    fn unknown_keys_are_tolerated() {
        let config: TeamConfig = serde_json::from_value(serde_json::json!({
            "max_agents": 3,
            "surprise": true
        }))
        .unwrap();
        assert_eq!(config.max_agents, 3);
        assert_eq!(config.max_concurrent_messages, 100);
    }

    #[test]
    fn builders_set_fields() {
        let config = TeamConfig::new()
            .configure_max_agents(4)
            .configure_concurrency(8)
            .configure_timeout(1.5);
        assert_eq!(config.max_agents, 4);
        assert_eq!(config.max_concurrent_messages, 8);
        assert!((config.message_timeout - 1.5).abs() < f64::EPSILON);
    }
}
