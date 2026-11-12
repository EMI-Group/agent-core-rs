//! The serialized LLM-agent DSL consumed by the platform.
//!
//! Ports the document produced by the Python
//! `openjiuwen.dev_tools.agent_builder.builders.llm_agent` transformer: agent
//! identity, configuration, model, plugin/workflow references, dependency
//! lists, and the create/update timestamps.
//!
//! The JSON field names are Python's (`agent_id`, `agent_type`, `configs`,
//! `model_info`, `create_time`, ...). Unknown top-level keys are captured in
//! [`LlmAgentDsl::extra`] and written back unchanged, so a document that passes
//! through this crate keeps every platform field it arrived with.

use std::collections::BTreeMap;

use openjiuwen_common::error::{Error, Result};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The `agent_type` used when a document does not carry one.
fn default_agent_type() -> String {
    "react".to_string()
}

/// The agent's prompt configuration.
///
/// Python keeps `opening_remarks` at the top level of the DSL; this port nests
/// it in `configs` so all configuration lives in one struct. The field name is
/// unchanged.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct DslConfigs {
    /// The agent's system prompt.
    pub system_prompt: String,
    /// The greeting the agent sends before the first user turn.
    ///
    /// Python serializes this key at the DSL top level; the Rust port nests it
    /// here, keeping the wire name `opening_remarks` unchanged.
    pub opening_remarks: String,
}

/// The model endpoint an agent talks to.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct DslModelInfo {
    /// The provider base URL.
    pub api_base: String,
    /// The provider's model name.
    pub model_name: String,
    /// The sampling temperature; unset means the provider default.
    pub temperature: Option<f32>,
    /// The nucleus-sampling threshold; unset means the provider default.
    pub top_p: Option<f32>,
}

/// The model block of an agent DSL document.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct DslModel {
    /// The endpoint and sampling settings.
    pub model_info: DslModelInfo,
}

/// A reference to a plugin the agent may call.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct PluginRef {
    /// The plugin id in the platform registry.
    pub id: String,
    /// The plugin's display name.
    pub name: String,
}

/// A reference to a workflow the agent may invoke.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct WorkflowRef {
    /// The workflow id in the platform registry.
    pub id: String,
    /// The workflow's display name.
    pub name: String,
}

/// The resources the agent depends on, as id lists.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct DslDependencies {
    /// Plugin ids.
    pub plugins: Vec<String>,
    /// Workflow ids.
    pub workflows: Vec<String>,
    /// Knowledge-base ids.
    pub knowledge_bases: Vec<String>,
    /// Prompt-template ids.
    pub prompt_templates: Vec<String>,
}

/// A complete LLM-agent DSL document.
///
/// Construct it directly, parse it with [`Self::from_json_str`] or
/// [`Self::from_value`], and write it back with [`Self::to_json_string`] or
/// [`Self::to_value`]. All unreferenced top-level keys round-trip through
/// [`Self::extra`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct LlmAgentDsl {
    /// The agent's platform id.
    pub agent_id: String,
    /// The agent kind; defaults to `"react"` when the document omits it.
    #[serde(default = "default_agent_type")]
    pub agent_type: String,
    /// The agent's display name.
    pub name: String,
    /// A human-readable agent description.
    pub description: String,
    /// The prompt configuration.
    pub configs: DslConfigs,
    /// The model block; a document may omit it entirely.
    pub model: Option<DslModel>,
    /// The plugins the agent references.
    pub plugins: Vec<PluginRef>,
    /// The workflows the agent references.
    pub workflows: Vec<WorkflowRef>,
    /// The resources the agent depends on.
    pub dependencies: DslDependencies,
    /// Creation time, epoch milliseconds, supplied by the caller.
    pub create_time: i64,
    /// Last update time, epoch milliseconds, supplied by the caller.
    pub update_time: i64,
    /// Any other platform keys, preserved verbatim across a round trip.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

impl Default for LlmAgentDsl {
    /// An empty document with Python's default agent type, `"react"`.
    fn default() -> Self {
        Self {
            agent_id: String::new(),
            agent_type: default_agent_type(),
            name: String::new(),
            description: String::new(),
            configs: DslConfigs::default(),
            model: None,
            plugins: Vec::new(),
            workflows: Vec::new(),
            dependencies: DslDependencies::default(),
            create_time: 0,
            update_time: 0,
            extra: BTreeMap::new(),
        }
    }
}

impl LlmAgentDsl {
    /// Serializes the document as pretty-printed JSON.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the document cannot be serialized,
    /// with the serde error in the message.
    pub fn to_json_string(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|error| Error::validation(format!("failed to serialize agent DSL: {error}")))
    }

    /// Parses an agent DSL document from JSON.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the text is not an agent DSL
    /// document, with the serde error in the message.
    pub fn from_json_str(text: &str) -> Result<Self> {
        serde_json::from_str(text)
            .map_err(|error| Error::validation(format!("failed to parse agent DSL JSON: {error}")))
    }

    /// Converts the document into a JSON value.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the document cannot be serialized,
    /// with the serde error in the message.
    pub fn to_value(&self) -> Result<Value> {
        serde_json::to_value(self)
            .map_err(|error| Error::validation(format!("failed to convert agent DSL: {error}")))
    }

    /// Builds the document from a JSON value.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the value is not an agent DSL
    /// document, with the serde error in the message.
    pub fn from_value(value: Value) -> Result<Self> {
        serde_json::from_value(value)
            .map_err(|error| Error::validation(format!("failed to read agent DSL: {error}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;

    /// A document that populates every field, including unknown extra keys.
    fn populated() -> LlmAgentDsl {
        LlmAgentDsl {
            agent_id: "agent-1".to_string(),
            agent_type: "workflow".to_string(),
            name: "Demo Agent".to_string(),
            description: "a demo".to_string(),
            configs: DslConfigs {
                system_prompt: "You are helpful.".to_string(),
                opening_remarks: "Hi!".to_string(),
            },
            model: Some(DslModel {
                model_info: DslModelInfo {
                    api_base: "https://example.test/v1".to_string(),
                    model_name: "demo-model".to_string(),
                    temperature: Some(0.5),
                    top_p: Some(0.9),
                },
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
            create_time: 1_700_000_000_000,
            update_time: 1_700_000_001_000,
            extra: BTreeMap::from([
                ("space_id".to_string(), json!("s1")),
                ("triggers".to_string(), json!([])),
            ]),
        }
    }

    #[test]
    fn round_trip_preserves_python_field_names() {
        let dsl = populated();
        let text = dsl.to_json_string().expect("serialize");
        let value: Value = serde_json::from_str(&text).expect("reparse");

        for key in [
            "agent_id",
            "agent_type",
            "name",
            "description",
            "configs",
            "model",
            "plugins",
            "workflows",
            "dependencies",
            "create_time",
            "update_time",
        ] {
            assert!(value.get(key).is_some(), "missing top-level key {key}");
        }

        assert_eq!(value["agent_id"], json!("agent-1"));
        assert_eq!(value["agent_type"], json!("workflow"));
        assert_eq!(value["configs"]["system_prompt"], json!("You are helpful."));
        assert_eq!(value["configs"]["opening_remarks"], json!("Hi!"));
        assert_eq!(
            value["model"]["model_info"]["api_base"],
            json!("https://example.test/v1")
        );
        assert_eq!(
            value["model"]["model_info"]["model_name"],
            json!("demo-model")
        );
        assert_eq!(value["model"]["model_info"]["temperature"], json!(0.5));
        assert_eq!(value["model"]["model_info"]["top_p"], json!(0.9));
        assert_eq!(value["plugins"][0]["id"], json!("plugin-1"));
        assert_eq!(value["plugins"][0]["name"], json!("search"));
        assert_eq!(value["workflows"][0]["id"], json!("workflow-1"));
        assert_eq!(value["workflows"][0]["name"], json!("flow"));
        assert_eq!(value["dependencies"]["plugins"], json!(["plugin-1"]));
        assert_eq!(value["dependencies"]["workflows"], json!(["workflow-1"]));
        assert_eq!(value["dependencies"]["knowledge_bases"], json!(["kb-1"]));
        assert_eq!(value["dependencies"]["prompt_templates"], json!(["tpl-1"]));
        assert_eq!(value["create_time"], json!(1_700_000_000_000_i64));
        assert_eq!(value["update_time"], json!(1_700_000_001_000_i64));
        assert_eq!(value["space_id"], json!("s1"));
        assert_eq!(value["triggers"], json!([]));

        assert_eq!(LlmAgentDsl::from_json_str(&text).expect("parse"), dsl);
    }

    #[test]
    fn unknown_fields_survive_from_str_to_value() {
        let text = r#"{
            "agent_id": "a",
            "agent_type": "react",
            "space_id": "s1",
            "triggers": []
        }"#;
        let dsl = LlmAgentDsl::from_json_str(text).expect("parse");
        assert_eq!(dsl.extra.get("space_id"), Some(&json!("s1")));
        assert_eq!(dsl.extra.get("triggers"), Some(&json!([])));

        let value = dsl.to_value().expect("to value");
        assert_eq!(value["agent_id"], json!("a"));
        assert_eq!(value["space_id"], json!("s1"));
        assert_eq!(value["triggers"], json!([]));
    }

    #[test]
    fn agent_type_defaults_to_react() {
        let dsl = LlmAgentDsl::from_json_str(r#"{"agent_id": "a"}"#).expect("parse");
        assert_eq!(dsl.agent_type, "react");
        assert_eq!(
            dsl,
            LlmAgentDsl {
                agent_id: "a".to_string(),
                ..LlmAgentDsl::default()
            }
        );
        assert_eq!(
            LlmAgentDsl::from_json_str("{}").expect("parse"),
            LlmAgentDsl::default()
        );
    }

    #[test]
    fn explicit_agent_type_is_kept() {
        let dsl = LlmAgentDsl::from_json_str(r#"{"agent_type": "workflow"}"#).expect("parse");
        assert_eq!(dsl.agent_type, "workflow");
    }

    #[test]
    fn malformed_json_is_a_validation_error() {
        for text in ["not json", "[1,2]"] {
            let error = LlmAgentDsl::from_json_str(text).expect_err("invalid document");
            assert_eq!(error.kind(), ErrorKind::Validation);
            assert!(error.to_string().contains("agent DSL"));
            assert!(error.to_string().contains("failed to parse"));
        }
    }

    #[test]
    fn from_value_rejects_non_objects() {
        let error = LlmAgentDsl::from_value(json!([1, 2])).expect_err("not a document");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("agent DSL"));
    }

    #[test]
    fn value_round_trip_is_exact() {
        let dsl = populated();
        let value = dsl.to_value().expect("to value");
        let restored = LlmAgentDsl::from_value(value.clone()).expect("from value");
        assert_eq!(restored, dsl);
        assert_eq!(restored.to_value().expect("to value"), value);
    }

    #[test]
    fn default_document_serializes_with_empty_blocks() {
        let value = LlmAgentDsl::default().to_value().expect("to value");
        assert_eq!(value["agent_type"], json!("react"));
        assert_eq!(value["name"], json!(""));
        assert_eq!(value["configs"]["system_prompt"], json!(""));
        assert_eq!(value["configs"]["opening_remarks"], json!(""));
        assert!(value["model"].is_null());
        assert_eq!(value["create_time"], json!(0));
        assert_eq!(value["update_time"], json!(0));
        assert_eq!(value["plugins"], json!([]));
        assert_eq!(value["workflows"], json!([]));
        assert_eq!(
            LlmAgentDsl::default().extra,
            BTreeMap::<String, Value>::new()
        );
    }
}
