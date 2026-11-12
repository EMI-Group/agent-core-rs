//! LLM-agent configuration and composition over [`ReActAgent`](crate::react::ReActAgent).
//!
//! Routing table for this module:
//!
//! - [`ConstrainConfig`] — chat-round and iteration limits (this round).
//! - `LlmAgentConfig` / `create_llm_agent_config` — configuration surface
//!   (later round).
//! - `LlmAgent` — the `ReActAgent` composition implementing [`Agent`](crate::agent::Agent)
//!   (later round).
//!
//! Python parity: `openjiuwen/core/application/llm_agent/`.
use serde::{Deserialize, Serialize};

/// Budget limits applied to an LLM agent.
///
/// The agent keeps at most `reserved_max_chat_rounds` conversation rounds in its
/// context window and allows up to `max_iteration` `ReAct` iterations per
/// invocation; both defaults match the Python `ConstrainConfig`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ConstrainConfig {
    /// Conversation rounds reserved in the context window; defaults to 10.
    pub reserved_max_chat_rounds: usize,
    /// Maximum `ReAct` iterations per invocation; defaults to 5.
    pub max_iteration: usize,
}
impl Default for ConstrainConfig {
    fn default() -> Self {
        Self {
            reserved_max_chat_rounds: 10,
            max_iteration: 5,
        }
    }
}
impl ConstrainConfig {
    /// Creates the default constraint budget (10 rounds, 5 iterations).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

// TODO: add `ModelInfo`, `ModelConfig`, `LlmAgentConfig`,
// `create_llm_agent_config`, and `LlmAgent` (the composition over `ReActAgent`)
// in the follow-up round.

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn defaults_match_python() {
        let constrain = ConstrainConfig::default();
        assert_eq!(constrain.reserved_max_chat_rounds, 10);
        assert_eq!(constrain.max_iteration, 5);
        assert_eq!(ConstrainConfig::new(), constrain);
    }
    #[test]
    fn json_round_trip() {
        let constrain = ConstrainConfig {
            reserved_max_chat_rounds: 3,
            max_iteration: 2,
        };
        let value = serde_json::to_value(&constrain).unwrap();
        let restored: ConstrainConfig = serde_json::from_value(value).unwrap();
        assert_eq!(restored, constrain);
    }
    #[test]
    fn unknown_keys_and_missing_fields_are_tolerated() {
        let constrain: ConstrainConfig =
            serde_json::from_value(serde_json::json!({"max_iteration": 7, "surprise": true}))
                .unwrap();
        assert_eq!(constrain.max_iteration, 7);
        assert_eq!(constrain.reserved_max_chat_rounds, 10);
    }
}
