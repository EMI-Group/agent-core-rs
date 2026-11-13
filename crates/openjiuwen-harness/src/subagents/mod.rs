//! Sub-agent definitions, presets, an in-process runtime, and the model-facing task tools.
//!
//! Ports `openjiuwen/harness/subagents/` (preset factories), the control subset of
//! `openjiuwen/harness/subagent_runtime/`, `subagent_lifecycle.py`, and
//! `openjiuwen/harness/tools/subagent/{task_tool,subagent_tools}.py`.
//!
//! The registry owns the sub-agent definitions the parent agent can delegate to, the
//! runtime owns the live child instances of one parent session, and [`TaskTool`] plus the
//! six `subagent_*` tools are the model-facing surface over them. Everything is in-process
//! and in-memory: child agents are built lazily by a [`SubAgentFactory`], so this module
//! never depends on [`crate::agent`].
mod presets;
mod runtime;
mod subagent_tools;
mod task_tool;
#[cfg(test)]
mod test_support;
use std::collections::BTreeMap;
use std::sync::Arc;

use parking_lot::RwLock;

pub use crate::schema::SubAgentConfig;
pub use presets::{build_preset, general_purpose_config, SubAgentPreset, GENERAL_PURPOSE_NAME};
pub use runtime::clock::{rfc3339_from_ms, Clock, ManualClock, SystemClock};
pub use runtime::quota::{
    select_lru_candidate, LruCandidate, SpawnBudget, SubAgentQuota, SubagentActivity,
};
pub use runtime::transcript::{
    TranscriptBuffer, TranscriptChannel, TranscriptEntry, TranscriptKind, TranscriptSubscription,
    TranscriptSummary, DEFAULT_TRANSCRIPT_CAPACITY, DEFAULT_TRANSCRIPT_CHANNEL_CAPACITY,
};
pub use runtime::{clock, quota, transcript};
pub use runtime::{
    ResumeReport, SubAgentFactory, SubAgentId, SubAgentOutcome, SubAgentRuntime,
    SubAgentSpawnRequest, SubagentMetadata, SubagentStatus,
};
pub use subagent_tools::{
    subagent_tools, SubagentCloseTool, SubagentListTool, SubagentResumeTool, SubagentSendInputTool,
    SubagentSpawnTool, SubagentWaitTool,
};
pub use task_tool::TaskTool;

/// One registered sub-agent definition: its configuration plus the preset it came from.
///
/// Python parity: the `SubAgentConfig` entries of the Python sub-agent registry. The preset
/// is kept so callers can tell a built-in preset from a custom definition; it is `None` for
/// definitions registered by name only.
#[derive(Debug, Clone, PartialEq)]
pub struct SubAgentSpec {
    /// The configuration the runtime builds the child agent from.
    pub config: SubAgentConfig,
    /// The built-in preset this definition was derived from, when any.
    pub preset: Option<SubAgentPreset>,
}

impl SubAgentSpec {
    /// Wraps `config` as a custom (non-preset) definition.
    #[must_use]
    pub fn new(config: SubAgentConfig) -> Self {
        Self {
            config,
            preset: None,
        }
    }

    /// Wraps `config` as the definition of `preset`.
    #[must_use]
    pub fn with_preset(config: SubAgentConfig, preset: SubAgentPreset) -> Self {
        Self {
            config,
            preset: Some(preset),
        }
    }

    /// Returns the registry key of this definition (`agent_card.name`).
    #[must_use]
    pub fn name(&self) -> &str {
        &self.config.agent_card.name
    }
}

/// The sub-agent definitions one parent agent can delegate to.
///
/// Python parity: the sub-agent registry of `openjiuwen.harness`. Keys are the
/// `agent_card.name` values, so registering a second definition with the same name
/// replaces the first.
///
/// Cloning is cheap: every clone shares the same definition map through an `Arc`, so a
/// runtime and the tools built over it observe one registry.
#[derive(Debug, Clone, Default)]
pub struct SubAgentRegistry {
    specs: Arc<RwLock<BTreeMap<String, SubAgentSpec>>>,
}

impl SubAgentRegistry {
    /// Creates an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a registry holding the built-in presets in `language`.
    ///
    /// The five always-available presets (`code`, `explore`, `plan`, `research`,
    /// `verification`) are registered; presets whose cargo feature is disabled are not
    /// enumerated at all, so this cannot fail for a valid language.
    #[must_use]
    pub fn with_builtin_presets(language: &str) -> Self {
        let registry = Self::new();
        for preset in SubAgentPreset::builtin() {
            match build_preset(*preset, language) {
                Ok(config) => registry.register(SubAgentSpec::with_preset(config, *preset)),
                Err(error) => tracing::warn!(
                    target: "openjiuwen_harness::subagents",
                    preset = preset.as_str(),
                    %error,
                    "skipping unavailable built-in sub-agent preset"
                ),
            }
        }
        registry
    }

    /// Registers `spec`, replacing any definition with the same `agent_card.name`.
    pub fn register(&self, spec: SubAgentSpec) {
        self.specs.write().insert(spec.name().to_owned(), spec);
    }

    /// Returns a copy of the configuration registered as `name`.
    ///
    /// The configuration is cloned because the registry is shared; callers that need the
    /// preset as well use [`SubAgentRegistry::get_spec`].
    #[must_use]
    pub fn get(&self, name: &str) -> Option<SubAgentConfig> {
        self.get_spec(name).map(|spec| spec.config)
    }

    /// Returns a copy of the definition registered as `name`.
    #[must_use]
    pub fn get_spec(&self, name: &str) -> Option<SubAgentSpec> {
        self.specs.read().get(name).cloned()
    }

    /// Whether a definition is registered as `name`.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.specs.read().contains_key(name)
    }

    /// Returns every definition, ordered by name.
    #[must_use]
    pub fn list_specs(&self) -> Vec<SubAgentSpec> {
        self.specs.read().values().cloned().collect()
    }

    /// Returns every configuration, ordered by name.
    #[must_use]
    pub fn list(&self) -> Vec<SubAgentConfig> {
        self.list_specs()
            .into_iter()
            .map(|spec| spec.config)
            .collect()
    }

    /// Returns every registered name, ordered.
    #[must_use]
    pub fn names(&self) -> Vec<String> {
        self.specs.read().keys().cloned().collect()
    }

    /// Returns the number of registered definitions.
    #[must_use]
    pub fn len(&self) -> usize {
        self.specs.read().len()
    }

    /// Whether no definition is registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.specs.read().is_empty()
    }

    /// Returns the `general-purpose` configuration, registering it on first use.
    ///
    /// Python parity: the general-purpose sub-agent is always available as a fallback, so
    /// the definition is inserted lazily instead of being part of the preset set.
    #[must_use]
    pub fn general_purpose(&self) -> SubAgentConfig {
        if let Some(config) = self.get(GENERAL_PURPOSE_NAME) {
            return config;
        }
        let config = general_purpose_config();
        self.register(SubAgentSpec::new(config.clone()));
        config
    }
}

#[cfg(test)]
mod tests {
    use openjiuwen_agent::AgentCard;

    use super::*;

    fn spec(name: &str) -> SubAgentSpec {
        SubAgentSpec::new(SubAgentConfig::new(
            AgentCard::new(name).with_description("test"),
        ))
    }

    #[test]
    fn registering_the_same_name_replaces_the_definition() {
        let registry = SubAgentRegistry::new();
        registry.register(spec("worker"));
        registry.register(SubAgentSpec::new(SubAgentConfig::new(
            AgentCard::new("worker").with_description("replacement"),
        )));
        assert_eq!(registry.len(), 1);
        assert_eq!(
            registry
                .get("worker")
                .map(|config| config.agent_card.description),
            Some("replacement".to_owned())
        );
    }

    #[test]
    fn builtin_presets_are_registered_by_python_name() {
        let registry = SubAgentRegistry::with_builtin_presets("en");
        assert_eq!(
            registry.names(),
            vec![
                "code_agent".to_owned(),
                "explore_agent".to_owned(),
                "plan_agent".to_owned(),
                "research_agent".to_owned(),
                "verification_agent".to_owned(),
            ]
        );
        assert!(registry
            .get_spec("code_agent")
            .and_then(|spec| spec.preset)
            .is_some());
        assert_eq!(
            registry
                .get("code_agent")
                .and_then(|config| config.subagent_type),
            Some("code_agent".to_owned())
        );
    }

    #[test]
    fn general_purpose_is_inserted_once() {
        let registry = SubAgentRegistry::new();
        let first = registry.general_purpose();
        let second = registry.general_purpose();
        assert_eq!(first, second);
        assert_eq!(first.agent_card.name, GENERAL_PURPOSE_NAME);
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn clones_share_one_definition_map() {
        let registry = SubAgentRegistry::new();
        let clone = registry.clone();
        registry.register(spec("shared"));
        assert!(clone.contains("shared"));
        assert_eq!(clone.list().len(), 1);
    }

    #[test]
    fn list_is_ordered_by_name() {
        let registry = SubAgentRegistry::new();
        registry.register(spec("zeta"));
        registry.register(spec("alpha"));
        let names: Vec<String> = registry
            .list()
            .into_iter()
            .map(|config| config.agent_card.name)
            .collect();
        assert_eq!(names, vec!["alpha".to_owned(), "zeta".to_owned()]);
    }
}
