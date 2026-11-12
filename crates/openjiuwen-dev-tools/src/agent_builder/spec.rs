//! Programmatic agent and workflow specifications.
//!
//! Ports the specification layer of `openjiuwen.dev_tools.agent_builder`: an
//! [`LlmAgentSpec`] describes an agent in plain Rust (name, prompt, model,
//! plugins, workflows) and converts into the platform document
//! [`LlmAgentDsl`]; a [`WorkflowSpec`] does the same for a graph of
//! [`DslNode`]s and [`DslEdge`]s.
//!
//! Both specs are `serde` models, so a specification can be authored in JSON,
//! stored, and re-parsed. The conversion into the DSL is deterministic: the
//! agent id and the clock are arguments, never read from the environment.
//!
//! The typed DSL subset has no provider or constraint fields, so
//! [`LlmAgentSpec::to_dsl`] keeps them in [`LlmAgentDsl::extra`] under the
//! Python key names (`model_provider`, `constraints`); the bridge in
//! [`crate::agent_builder::bridge`] reads them back.

use std::collections::BTreeMap;

use openjiuwen_common::error::{Error, Result};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::dsl::{
    DslConfigs, DslDependencies, DslModel, DslModelInfo, LlmAgentDsl, PluginRef, WorkflowRef,
};
use super::workflow_dsl::{DslEdge, DslNode, WorkflowDsl};

/// The provider assumed when a specification does not set one explicitly.
const DEFAULT_PROVIDER: &str = "openai";

/// The chat rounds reported alongside a `max_iteration` constraint.
///
/// Python's DSL carries both limits in one `constraints` block; the port has no
/// typed field for the round count, so it reports the platform default.
const DEFAULT_RESERVED_MAX_CHAT_ROUNDS: usize = 10;

/// A programmatic description of an LLM agent.
///
/// Build one with [`LlmAgentSpec::new`] and the `with_*` builders, then convert
/// it into the platform document with [`LlmAgentSpec::to_dsl`]. Every field is
/// optional on the wire: [`Default::default`] yields an empty specification
/// whose `provider` is already the documented [`DEFAULT_PROVIDER`] (`"openai"`),
/// so `new` and a deserialized document agree on the fallback.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct LlmAgentSpec {
    /// The agent's display name.
    pub name: String,
    /// A human-readable agent description.
    pub description: String,
    /// The system prompt sent ahead of every conversation.
    pub system_prompt: String,
    /// The greeting the agent sends before the first user turn.
    pub opening_remarks: String,
    /// The model provider name; defaults to [`DEFAULT_PROVIDER`].
    pub provider: String,
    /// The model endpoint; `None` produces a DSL without a model block.
    pub model: Option<DslModelInfo>,
    /// The plugins the agent may call.
    pub plugins: Vec<PluginRef>,
    /// The workflows the agent may invoke.
    pub workflows: Vec<WorkflowRef>,
    /// The resources the agent depends on.
    pub dependencies: DslDependencies,
    /// The `ReAct` iteration budget; `None` leaves the platform default.
    pub max_iteration: Option<usize>,
}

impl Default for LlmAgentSpec {
    /// An empty specification with the default provider, `"openai"`.
    fn default() -> Self {
        Self {
            name: String::new(),
            description: String::new(),
            system_prompt: String::new(),
            opening_remarks: String::new(),
            provider: DEFAULT_PROVIDER.to_string(),
            model: None,
            plugins: Vec::new(),
            workflows: Vec::new(),
            dependencies: DslDependencies::default(),
            max_iteration: None,
        }
    }
}

impl LlmAgentSpec {
    /// Creates a specification with a name, a description, and a system prompt.
    ///
    /// Every other field takes its [`Default`] value, including the `"openai"`
    /// provider.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        system_prompt: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            system_prompt: system_prompt.into(),
            ..Self::default()
        }
    }

    /// Sets the model endpoint.
    #[must_use]
    pub fn with_model(mut self, model: DslModelInfo) -> Self {
        self.model = Some(model);
        self
    }

    /// Sets the model provider name.
    #[must_use]
    pub fn with_provider(mut self, provider: impl Into<String>) -> Self {
        self.provider = provider.into();
        self
    }

    /// Sets the `ReAct` iteration budget.
    #[must_use]
    pub fn with_max_iteration(mut self, max_iteration: usize) -> Self {
        self.max_iteration = Some(max_iteration);
        self
    }

    /// Converts the specification into the platform DSL document.
    ///
    /// The agent id and the clock are supplied by the caller: `agent_id` becomes
    /// [`LlmAgentDsl::agent_id`] and `timestamp_ms` (epoch milliseconds) is
    /// written to both [`LlmAgentDsl::create_time`] and
    /// [`LlmAgentDsl::update_time`], so converting the same specification twice
    /// with the same arguments yields an identical document.
    ///
    /// The typed DSL has no provider or constraint fields, so both are preserved
    /// in [`LlmAgentDsl::extra`] under their Python key names:
    ///
    /// - `extra["model_provider"]` — the [`Self::provider`] value, omitted when
    ///   the provider is empty;
    /// - `extra["constraints"]` — `{"max_iterations": n,
    ///   "reserved_max_chat_rounds": 10}` when [`Self::max_iteration`] is set.
    #[must_use]
    pub fn to_dsl(&self, agent_id: impl Into<String>, timestamp_ms: i64) -> LlmAgentDsl {
        let mut extra = BTreeMap::new();
        if !self.provider.is_empty() {
            extra.insert("model_provider".to_string(), json!(self.provider));
        }
        if let Some(max_iteration) = self.max_iteration {
            extra.insert(
                "constraints".to_string(),
                json!({
                    "max_iterations": max_iteration,
                    "reserved_max_chat_rounds": DEFAULT_RESERVED_MAX_CHAT_ROUNDS,
                }),
            );
        }

        LlmAgentDsl {
            agent_id: agent_id.into(),
            agent_type: "react".to_string(),
            name: self.name.clone(),
            description: self.description.clone(),
            configs: DslConfigs {
                system_prompt: self.system_prompt.clone(),
                opening_remarks: self.opening_remarks.clone(),
            },
            model: self.model.clone().map(|model_info| DslModel { model_info }),
            plugins: self.plugins.clone(),
            workflows: self.workflows.clone(),
            dependencies: self.dependencies.clone(),
            create_time: timestamp_ms,
            update_time: timestamp_ms,
            extra,
        }
    }

    /// Serializes the specification as pretty-printed JSON.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the specification cannot be
    /// serialized, with the serde error in the message.
    pub fn to_json_string(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|error| Error::validation(format!("failed to serialize agent spec: {error}")))
    }

    /// Parses a specification from JSON.
    ///
    /// Unknown keys are ignored (`#[serde(default)]` tolerance), so a document
    /// authored against a newer platform version still parses; absent keys take
    /// their [`Default`] value.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the text is not an agent
    /// specification, with the serde error in the message.
    pub fn from_json_str(text: &str) -> Result<Self> {
        serde_json::from_str(text)
            .map_err(|error| Error::validation(format!("failed to parse agent spec JSON: {error}")))
    }
}

/// A programmatic description of a workflow.
///
/// The spec holds the graph itself (nodes and edges, in document order) and
/// converts one-to-one into [`WorkflowDsl`]; structural checks live on the DSL
/// ([`WorkflowDsl::validate`]).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct WorkflowSpec {
    /// The workflow name.
    pub name: String,
    /// A human-readable workflow description.
    pub description: String,
    /// The nodes of the graph, in document order.
    pub nodes: Vec<DslNode>,
    /// The edges of the graph, in document order.
    pub edges: Vec<DslEdge>,
}

impl WorkflowSpec {
    /// Converts the specification into the workflow DSL document.
    #[must_use]
    pub fn to_dsl(&self) -> WorkflowDsl {
        WorkflowDsl {
            name: self.name.clone(),
            description: self.description.clone(),
            nodes: self.nodes.clone(),
            edges: self.edges.clone(),
        }
    }

    /// Serializes the specification as pretty-printed JSON.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the specification cannot be
    /// serialized, with the serde error in the message.
    pub fn to_json_string(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(|error| {
            Error::validation(format!("failed to serialize workflow spec: {error}"))
        })
    }

    /// Parses a specification from JSON.
    ///
    /// Unknown keys are ignored and absent keys take their [`Default`] value;
    /// call [`WorkflowDsl::validate`] on the converted document before executing
    /// it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the text is not a workflow
    /// specification, with the serde error in the message.
    pub fn from_json_str(text: &str) -> Result<Self> {
        serde_json::from_str(text).map_err(|error| {
            Error::validation(format!("failed to parse workflow spec JSON: {error}"))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;
    use serde_json::Value;

    /// A specification that populates every field.
    fn populated() -> LlmAgentSpec {
        LlmAgentSpec {
            name: "Demo Agent".to_string(),
            description: "a demo".to_string(),
            system_prompt: "You are helpful.".to_string(),
            opening_remarks: "Hi!".to_string(),
            provider: "anthropic".to_string(),
            model: Some(DslModelInfo {
                api_base: "https://example.test/v1".to_string(),
                model_name: "demo-model".to_string(),
                temperature: Some(0.5),
                top_p: Some(0.9),
            }),
            plugins: vec![PluginRef {
                id: "plugin-1".to_string(),
                name: "search".to_string(),
            }],
            workflows: vec![WorkflowRef {
                id: "workflow-1".to_string(),
                name: "flow".to_string(),
            }],
            dependencies: DslDependencies {
                plugins: vec!["plugin-1".to_string()],
                workflows: vec!["workflow-1".to_string()],
                knowledge_bases: vec!["kb-1".to_string()],
                prompt_templates: vec!["tpl-1".to_string()],
            },
            max_iteration: Some(7),
        }
    }

    /// A two-node, one-edge workflow specification: `start` -> `end`.
    fn workflow() -> WorkflowSpec {
        WorkflowSpec {
            name: "demo".to_string(),
            description: "a demo workflow".to_string(),
            nodes: vec![
                DslNode {
                    id: "start".to_string(),
                    node_type: "start".to_string(),
                    title: "Start".to_string(),
                    description: "entry".to_string(),
                    inputs: BTreeMap::new(),
                    outputs: BTreeMap::new(),
                    data: json!({"prompt": "hello"}),
                },
                DslNode {
                    id: "end".to_string(),
                    node_type: "end".to_string(),
                    title: "End".to_string(),
                    description: "exit".to_string(),
                    inputs: BTreeMap::new(),
                    outputs: BTreeMap::new(),
                    data: Value::Null,
                },
            ],
            edges: vec![DslEdge {
                source: "start".to_string(),
                target: "end".to_string(),
                condition: Some("score > 0.5".to_string()),
            }],
        }
    }

    #[test]
    fn new_uses_documented_defaults() {
        let spec = LlmAgentSpec::new("Agent", "desc", "prompt");
        assert_eq!(spec.name, "Agent");
        assert_eq!(spec.description, "desc");
        assert_eq!(spec.system_prompt, "prompt");
        assert_eq!(spec.opening_remarks, "");
        assert_eq!(spec.provider, "openai");
        assert!(spec.model.is_none());
        assert_eq!(spec.plugins, Vec::<PluginRef>::new());
        assert_eq!(spec.workflows, Vec::<WorkflowRef>::new());
        assert_eq!(spec.dependencies, DslDependencies::default());
        assert!(spec.max_iteration.is_none());

        let default = LlmAgentSpec::default();
        assert_eq!(default.provider, "openai");
        assert_eq!(default, LlmAgentSpec::new("", "", ""));
    }

    #[test]
    fn builders_chain() {
        let model = DslModelInfo {
            api_base: "https://example.test/v1".to_string(),
            model_name: "demo-model".to_string(),
            temperature: Some(0.2),
            top_p: None,
        };
        let spec = LlmAgentSpec::new("Agent", "desc", "prompt")
            .with_provider("azure")
            .with_model(model.clone())
            .with_max_iteration(3);

        assert_eq!(spec.provider, "azure");
        assert_eq!(spec.model, Some(model));
        assert_eq!(spec.max_iteration, Some(3));
    }

    #[test]
    fn to_dsl_is_deterministic() {
        let spec = populated();
        let first = spec.to_dsl("agent-1", 1_700_000_000_000);
        let second = spec.to_dsl("agent-1", 1_700_000_000_000);

        assert_eq!(first, second);
        assert_eq!(
            first.to_json_string().expect("serialize"),
            second.to_json_string().expect("serialize")
        );
    }

    #[test]
    fn to_dsl_writes_the_supplied_timestamp() {
        let spec = populated();
        let early = spec.to_dsl("agent-1", 1_000);
        let late = spec.to_dsl("agent-1", 2_000);

        assert_eq!(early.create_time, 1_000);
        assert_eq!(early.update_time, 1_000);
        assert_eq!(late.create_time, 2_000);
        assert_eq!(late.update_time, 2_000);
        assert_ne!(early, late);
    }

    #[test]
    fn to_dsl_maps_every_spec_field() {
        let dsl = populated().to_dsl("agent-1", 1_700_000_000_000);

        assert_eq!(dsl.agent_id, "agent-1");
        assert_eq!(dsl.agent_type, "react");
        assert_eq!(dsl.name, "Demo Agent");
        assert_eq!(dsl.description, "a demo");
        assert_eq!(dsl.configs.system_prompt, "You are helpful.");
        assert_eq!(dsl.configs.opening_remarks, "Hi!");
        assert_eq!(
            dsl.model,
            Some(DslModel {
                model_info: DslModelInfo {
                    api_base: "https://example.test/v1".to_string(),
                    model_name: "demo-model".to_string(),
                    temperature: Some(0.5),
                    top_p: Some(0.9),
                },
            })
        );
        assert_eq!(dsl.plugins.len(), 1);
        assert_eq!(dsl.plugins[0].name, "search");
        assert_eq!(dsl.workflows.len(), 1);
        assert_eq!(dsl.workflows[0].id, "workflow-1");
        assert_eq!(dsl.dependencies.knowledge_bases, vec!["kb-1"]);
    }

    #[test]
    fn to_dsl_preserves_provider_and_constraints_in_extra() {
        let dsl = populated().to_dsl("agent-1", 1_000);
        assert_eq!(dsl.extra.get("model_provider"), Some(&json!("anthropic")));
        assert_eq!(
            dsl.extra.get("constraints"),
            Some(&json!({
                "max_iterations": 7,
                "reserved_max_chat_rounds": 10,
            }))
        );

        let value = dsl.to_value().expect("to value");
        assert_eq!(value["model_provider"], json!("anthropic"));
        assert_eq!(value["constraints"]["max_iterations"], json!(7));
        assert_eq!(value["constraints"]["reserved_max_chat_rounds"], json!(10));
    }

    #[test]
    fn to_dsl_omits_unset_provider_and_constraints() {
        let mut spec = populated();
        spec.provider = String::new();
        spec.max_iteration = None;
        let dsl = spec.to_dsl("agent-1", 1_000);

        assert!(!dsl.extra.contains_key("model_provider"));
        assert!(!dsl.extra.contains_key("constraints"));
        assert!(dsl.extra.is_empty());
    }

    #[test]
    fn spec_json_round_trip_preserves_known_fields() {
        let spec = populated();
        let text = spec.to_json_string().expect("serialize");
        let value: Value = serde_json::from_str(&text).expect("reparse");

        assert_eq!(value["name"], json!("Demo Agent"));
        assert_eq!(value["provider"], json!("anthropic"));
        assert_eq!(value["model"]["model_name"], json!("demo-model"));
        assert_eq!(value["max_iteration"], json!(7));
        assert_eq!(value["opening_remarks"], json!("Hi!"));

        assert_eq!(LlmAgentSpec::from_json_str(&text).expect("parse"), spec);
    }

    #[test]
    fn spec_json_tolerates_unknown_fields() {
        let text = r#"{
            "name": "Agent",
            "description": "desc",
            "system_prompt": "prompt",
            "provider": "azure",
            "unknown_platform_field": {"nested": true}
        }"#;
        let spec = LlmAgentSpec::from_json_str(text).expect("unknown keys are tolerated");

        assert_eq!(spec.name, "Agent");
        assert_eq!(spec.description, "desc");
        assert_eq!(spec.system_prompt, "prompt");
        assert_eq!(spec.provider, "azure");
        assert!(spec.model.is_none());

        let restored: Value =
            serde_json::from_str(&spec.to_json_string().expect("serialize")).expect("reparse");
        assert!(restored.get("unknown_platform_field").is_none());
    }

    #[test]
    fn from_json_str_reports_validation_errors() {
        for text in ["nope", "[1]"] {
            let error = LlmAgentSpec::from_json_str(text).expect_err("invalid document");
            assert_eq!(error.kind(), ErrorKind::Validation);
            assert!(error.to_string().contains("agent spec"));
            assert!(error.to_string().contains("failed to parse"));
        }
    }

    #[test]
    fn workflow_spec_to_dsl_keeps_nodes_and_edges() {
        let spec = workflow();
        let dsl = spec.to_dsl();

        assert_eq!(dsl.name, "demo");
        assert_eq!(dsl.description, "a demo workflow");
        assert_eq!(dsl.nodes, spec.nodes);
        assert_eq!(dsl.edges, spec.edges);
        assert_eq!(dsl.nodes.len(), 2);
        assert_eq!(dsl.edges[0].condition.as_deref(), Some("score > 0.5"));
        dsl.validate().expect("converted graph is valid");
    }

    #[test]
    fn workflow_spec_json_round_trip() {
        let spec = workflow();
        let text = spec.to_json_string().expect("serialize");
        let value: Value = serde_json::from_str(&text).expect("reparse");

        assert_eq!(value["name"], json!("demo"));
        assert_eq!(value["nodes"][0]["type"], json!("start"));
        assert_eq!(value["edges"][0]["condition"], json!("score > 0.5"));

        assert_eq!(WorkflowSpec::from_json_str(&text).expect("parse"), spec);
    }

    #[test]
    fn workflow_spec_from_json_str_reports_validation_errors() {
        for text in ["nope", "[1]"] {
            let error = WorkflowSpec::from_json_str(text).expect_err("invalid document");
            assert_eq!(error.kind(), ErrorKind::Validation);
            assert!(error.to_string().contains("workflow spec"));
        }
    }

    #[test]
    fn empty_spec_converts_to_default_dsl_document() {
        let dsl = LlmAgentSpec::default().to_dsl("agent-1", 0);
        assert_eq!(dsl.agent_id, "agent-1");
        assert_eq!(dsl.agent_type, "react");
        assert!(dsl.model.is_none());
        assert_eq!(dsl.create_time, 0);
        assert_eq!(dsl.update_time, 0);
        assert_eq!(dsl.extra.get("model_provider"), Some(&json!("openai")));
        assert!(!dsl.extra.contains_key("constraints"));
    }
}
