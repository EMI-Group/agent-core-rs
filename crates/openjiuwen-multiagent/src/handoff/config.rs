//! Handoff configuration: routes, hop limits, and the early-termination hook.
//!
//! Python parity: `HandoffRoute`, `HandoffConfig`, and `HandoffTeamConfig` from
//! `openjiuwen/core/multi_agent/teams/handoff/handoff_config.py`. The Python
//! `HandoffTeamConfig` subclasses `TeamConfig`; Rust composes instead, keeping
//! the team limits under [`HandoffTeamConfig::team`] while serializing them
//! flattened, so the JSON shape stays the Python one.
//!
//! [`HandoffConfig::termination_condition`] is a process-local closure and is
//! neither serializable nor schema-visible: a deserialized config has no
//! termination hook, matching Python's `arbitrary_types_allowed` field.

use std::sync::Arc;

use futures::future::BoxFuture;
use openjiuwen_agent::AgentCard;
use openjiuwen_common::error::Result;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::config::TeamConfig;

/// Default number of handoff transfers after the initial hop.
const DEFAULT_MAX_HANDOFFS: usize = 10;

/// One routing rule: `source` may hand off to `target`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct HandoffRoute {
    /// Id of the agent that may start the handoff.
    pub source: String,
    /// Id of the agent the handoff is directed at.
    pub target: String,
}

impl HandoffRoute {
    /// Builds a routing rule from `source` to `target`.
    #[must_use]
    pub fn new(source: impl Into<String>, target: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            target: target.into(),
        }
    }
}

/// A snapshot of the coordinator handed to a termination condition.
///
/// It carries only the state a condition needs to decide whether the chain
/// should stop, so a condition cannot mutate the coordinator.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct HandoffSnapshot {
    /// Id of the agent that would run the next hop.
    pub current_agent_id: String,
    /// Number of handoffs approved so far.
    pub handoff_count: usize,
}

/// An asynchronous early-termination predicate.
///
/// The condition receives the current [`HandoffSnapshot`] and returns `true`
/// when the chain must stop. Python passes the orchestrator itself and accepts a
/// plain or awaitable result; Rust takes the immutable snapshot and always
/// returns a future, which is the async-only equivalent.
pub type TerminationCondition =
    Arc<dyn Fn(HandoffSnapshot) -> BoxFuture<'static, Result<bool>> + Send + Sync>;

/// Handoff orchestration parameters.
///
/// `start_agent` defaults to the first member the strategy registers, `routes`
/// empty means a full mesh, and `max_handoffs` caps the transfers after the
/// initial hop.
#[derive(Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct HandoffConfig {
    /// Card of the agent that runs the first hop; `None` uses the first member.
    pub start_agent: Option<AgentCard>,
    /// Maximum number of handoff transfers after the initial hop.
    pub max_handoffs: usize,
    /// Explicit routing rules; an empty list allows every member to hand off to
    /// every other member.
    pub routes: Vec<HandoffRoute>,
    /// Optional early-termination predicate; never serialized.
    #[serde(skip)]
    #[schemars(skip)]
    pub termination_condition: Option<TerminationCondition>,
}

impl Default for HandoffConfig {
    fn default() -> Self {
        Self {
            start_agent: None,
            max_handoffs: DEFAULT_MAX_HANDOFFS,
            routes: Vec::new(),
            termination_condition: None,
        }
    }
}

impl std::fmt::Debug for HandoffConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HandoffConfig")
            .field(
                "start_agent",
                &self.start_agent.as_ref().map(|card| card.id.as_str()),
            )
            .field("max_handoffs", &self.max_handoffs)
            .field("routes", &self.routes.len())
            .field("has_termination", &self.termination_condition.is_some())
            .finish()
    }
}

/// The full configuration of a handoff team.
///
/// The team limits are flattened into the same JSON object, so a handoff team
/// accepts exactly the `TeamConfig` keys plus `handoff`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct HandoffTeamConfig {
    /// The team limits shared by every strategy.
    #[serde(flatten)]
    pub team: TeamConfig,
    /// Handoff-specific orchestration parameters.
    pub handoff: HandoffConfig,
}

impl HandoffTeamConfig {
    /// Builds a configuration with the default team limits.
    #[must_use]
    pub fn new(handoff: HandoffConfig) -> Self {
        Self {
            team: TeamConfig::new(),
            handoff,
        }
    }

    /// Sets the team's member cap.
    #[must_use]
    pub fn configure_max_agents(mut self, max_agents: usize) -> Self {
        self.team = self.team.configure_max_agents(max_agents);
        self
    }

    /// Sets the per-topic message-queue capacity.
    #[must_use]
    pub fn configure_concurrency(mut self, max_concurrent_messages: usize) -> Self {
        self.team = self.team.configure_concurrency(max_concurrent_messages);
        self
    }

    /// Sets the dispatch deadline, in seconds.
    #[must_use]
    pub fn configure_timeout(mut self, message_timeout: f64) -> Self {
        self.team = self.team.configure_timeout(message_timeout);
        self
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use futures::FutureExt;
    use serde_json::json;

    use super::*;

    #[test]
    fn defaults_match_python() {
        let config = HandoffConfig::default();
        assert!(config.start_agent.is_none());
        assert_eq!(config.max_handoffs, 10);
        assert!(config.routes.is_empty(), "routes default to empty");
        assert!(config.termination_condition.is_none());

        let team = HandoffTeamConfig::default();
        assert_eq!(team.team, TeamConfig::new());
        assert_eq!(team.handoff.max_handoffs, 10);
    }

    #[test]
    fn empty_object_deserializes_to_defaults() {
        let config: HandoffConfig = serde_json::from_value(json!({})).unwrap();
        assert_eq!(config.max_handoffs, 10);
        assert!(config.routes.is_empty(), "routes default to empty");

        let team: HandoffTeamConfig = serde_json::from_value(json!({})).unwrap();
        assert_eq!(team.team, TeamConfig::new());
        assert_eq!(team.handoff.max_handoffs, 10);
        assert!(
            team.handoff.routes.is_empty(),
            "handoff routes default to empty"
        );
    }

    #[test]
    fn team_config_json_is_flat_both_ways() {
        let wire = json!({
            "max_agents": 4,
            "message_timeout": 2.5,
            "handoff": {
                "max_handoffs": 3,
                "routes": [{"source": "a", "target": "b"}]
            }
        });
        let config: HandoffTeamConfig = serde_json::from_value(wire.clone()).unwrap();
        assert_eq!(config.team.max_agents, 4);
        assert_eq!(config.team.max_concurrent_messages, 100);
        assert!((config.team.message_timeout - 2.5).abs() < f64::EPSILON);
        assert_eq!(config.handoff.max_handoffs, 3);
        assert_eq!(config.handoff.routes, vec![HandoffRoute::new("a", "b")]);

        // Serialization is flat and always emits the full shape.
        let round_trip = serde_json::to_value(&config).unwrap();
        assert_eq!(
            round_trip,
            json!({
                "max_agents": 4,
                "max_concurrent_messages": 100,
                "message_timeout": 2.5,
                "handoff": {
                    "start_agent": null,
                    "max_handoffs": 3,
                    "routes": [{"source": "a", "target": "b"}]
                }
            })
        );
        let reparsed: HandoffTeamConfig = serde_json::from_value(round_trip).unwrap();
        assert_eq!(reparsed.team, config.team);
        assert_eq!(reparsed.handoff.routes, config.handoff.routes);
    }

    #[test]
    fn unknown_keys_are_tolerated() {
        let config: HandoffTeamConfig = serde_json::from_value(json!({
            "max_agents": 2,
            "surprise": true,
            "handoff": {"max_handoffs": 1, "extra": {"nested": 1}}
        }))
        .unwrap();
        assert_eq!(config.team.max_agents, 2);
        assert_eq!(config.handoff.max_handoffs, 1);
    }

    #[test]
    fn termination_condition_is_never_serialized() {
        let condition: TerminationCondition =
            Arc::new(|_snapshot| futures::future::ready(Ok(false)).boxed());
        let config = HandoffConfig {
            termination_condition: Some(Arc::clone(&condition)),
            ..HandoffConfig::default()
        };
        let value = serde_json::to_value(&config).unwrap();
        assert!(value.get("termination_condition").is_none());
        assert_eq!(
            value,
            json!({"start_agent": null, "max_handoffs": 10, "routes": []})
        );

        let debug = format!("{config:?}");
        assert!(debug.contains("has_termination: true"));

        let parsed: HandoffConfig = serde_json::from_value(value).unwrap();
        assert!(parsed.termination_condition.is_none());
    }

    #[test]
    fn snapshot_round_trips() {
        let snapshot = HandoffSnapshot {
            current_agent_id: "b".to_owned(),
            handoff_count: 2,
        };
        let value = serde_json::to_value(&snapshot).unwrap();
        assert_eq!(value, json!({"current_agent_id": "b", "handoff_count": 2}));
        let parsed: HandoffSnapshot = serde_json::from_value(value).unwrap();
        assert_eq!(parsed, snapshot);
        assert_eq!(HandoffSnapshot::default().handoff_count, 0);
    }

    #[test]
    fn team_config_builders_delegate() {
        let config = HandoffTeamConfig::new(HandoffConfig::default())
            .configure_max_agents(3)
            .configure_concurrency(7)
            .configure_timeout(1.25);
        assert_eq!(config.team.max_agents, 3);
        assert_eq!(config.team.max_concurrent_messages, 7);
        assert!((config.team.message_timeout - 1.25).abs() < f64::EPSILON);
    }
}
