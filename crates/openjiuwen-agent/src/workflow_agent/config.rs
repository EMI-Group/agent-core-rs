//! Configuration of [`WorkflowAgent`](super::WorkflowAgent).
//!
//! Python parity: `WorkflowAgentConfig`, `ConstrainConfig`, and
//! `DefaultResponse` in `openjiuwen/core/application/workflow_agent/`.
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

pub use crate::llm_agent::ConstrainConfig;

/// Response returned when no workflow can be selected.
///
/// Python's `DefaultResponse`; `text` is `None` when the agent should surface the
/// selection failure as an error instead.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct DefaultResponse {
    /// Predefined answer text; omitted from JSON when `None`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

/// Configuration of a [`WorkflowAgent`](super::WorkflowAgent).
///
/// The identity fields (`id`, `name`, `description`, `version`) become the
/// agent's [`AgentCard`](crate::card::AgentCard); `constrain` limits the chat
/// rounds and iterations a delegated model run may use; `default_response` is
/// returned when workflow selection fails; `global_params` are merged into the
/// workflow start inputs on every run.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct WorkflowAgentConfig {
    /// Unique agent id; also the id of the derived [`AgentCard`](crate::card::AgentCard).
    pub id: String,
    /// Human-readable agent name.
    pub name: String,
    /// Free-form description used for discovery and intent matching.
    pub description: String,
    /// Agent version string; empty when unversioned.
    pub version: String,
    /// Chat-round/iteration budget applied to model-assisted runs.
    pub constrain: ConstrainConfig,
    /// Answer returned when workflow selection fails; `None` text disables it.
    pub default_response: DefaultResponse,
    /// Parameters merged into the workflow start inputs on every run.
    pub global_params: BTreeMap<String, Value>,
}

impl WorkflowAgentConfig {
    /// Creates a configuration for `id` with every other field at its default.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            ..Self::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn default_values_match_python() {
        let config = WorkflowAgentConfig::default();
        assert_eq!(config.constrain.reserved_max_chat_rounds, 10);
        assert_eq!(config.constrain.max_iteration, 5);
        assert_eq!(config.id, "");
        assert_eq!(config.name, "");
        assert_eq!(config.description, "");
        assert_eq!(config.version, "");
        assert!(config.default_response.text.is_none());
        assert!(config.global_params.is_empty());
        assert_eq!(WorkflowAgentConfig::new(String::new()), config);
    }
    #[test]
    fn new_sets_id_only() {
        let config = WorkflowAgentConfig::new("wf-agent");
        assert_eq!(config.id, "wf-agent");
        assert_eq!(config.constrain, ConstrainConfig::new());
    }
    #[test]
    fn json_round_trip() {
        let mut config = WorkflowAgentConfig::new("a1");
        config.name = "Agent".to_owned();
        config.description = "runs workflows".to_owned();
        config.version = "1.0".to_owned();
        config.constrain.max_iteration = 3;
        config.default_response.text = Some("fallback".to_owned());
        config
            .global_params
            .insert("city".to_owned(), Value::String("Paris".to_owned()));
        let value = serde_json::to_value(&config).unwrap();
        let restored: WorkflowAgentConfig = serde_json::from_value(value).unwrap();
        assert_eq!(restored, config);
    }
    #[test]
    fn default_response_omits_none_text() {
        let value = serde_json::to_value(DefaultResponse::default()).unwrap();
        assert!(value.get("text").is_none());
        assert_eq!(value, serde_json::json!({}));
    }
    #[test]
    fn unknown_keys_and_missing_fields_are_tolerated() {
        let config: WorkflowAgentConfig =
            serde_json::from_value(serde_json::json!({"id": "x", "surprise": 1})).unwrap();
        assert_eq!(config.id, "x");
        assert_eq!(config.constrain, ConstrainConfig::new());
    }
}
