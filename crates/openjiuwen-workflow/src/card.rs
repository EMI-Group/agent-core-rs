//! Serializable workflow definition types.
//!
//! These types are the JSON contract of a workflow: [`WorkflowCard`] identifies it,
//! [`WorkflowSpec`] describes its nodes and edges, and [`WorkflowConfig`] bundles the
//! two with the nesting limit. Field names, defaults, and card keys match the Python
//! `openjiuwen.core.workflow.workflow_config` module; every type is `snake_case`,
//! tolerates unknown fields, and round-trips through JSON.
//!
//! ```
//! use openjiuwen_workflow::card::{WorkflowCard, WorkflowConfig};
//!
//! let card = WorkflowCard::new("greeting");
//! assert_eq!(card.name, "greeting");
//! assert_eq!(WorkflowConfig::default().workflow_max_nesting_depth, 5);
//! ```

#![allow(clippy::module_name_repetitions)]

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use openjiuwen_common::error::Result;

use crate::ability::ComponentAbility;
use crate::error::component_id_invalid;

/// The longest component id accepted by [`validate_component_id`].
pub const MAX_COMPONENT_ID_LEN: usize = 100;

/// The reason reported when a component id contains an unsupported character.
pub const INVALID_ID_CHARS_REASON: &str =
    "only support letters (a\u{2013}z, A\u{2013}Z), digits (0\u{2013}9), underscores (_) or hyphens (-)";

/// A workflow's identity and metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct WorkflowCard {
    /// The unique workflow id.
    pub id: String,
    /// The human-readable workflow name.
    pub name: String,
    /// A free-form description.
    pub description: String,
    /// The workflow version string; empty when unversioned.
    pub version: String,
    /// Optional JSON Schema of the workflow's inputs.
    pub input_params: Option<Value>,
}

impl WorkflowCard {
    /// Creates a card whose id and name are both `id`, with an empty description.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        let id = id.into();
        Self {
            name: id.clone(),
            id,
            description: String::new(),
            version: String::new(),
            input_params: None,
        }
    }
}

impl Default for WorkflowCard {
    /// Mints a fresh v4 UUID id, mirroring Python's `uuid.uuid4()` card default.
    fn default() -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            name: String::new(),
            description: String::new(),
            version: String::new(),
            input_params: None,
        }
    }
}

/// How a component maps incoming and outgoing data.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct CompIoConfig {
    /// The JSON template resolved against workflow state before the component runs.
    pub inputs_schema: Option<Value>,
    /// The JSON template describing the component's output keys.
    pub outputs_schema: Option<Value>,
}

/// What to do when a node fails; Python only implements `handle_type="interrupt"`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ExceptionConfig {
    /// The failure policy; `"interrupt"` propagates the error to the caller.
    pub handle_type: String,
    /// Outputs substituted when the policy recovers from a failure.
    pub default_outputs: Option<Value>,
    /// Any other Python-side keys, preserved verbatim.
    #[serde(flatten)]
    pub metadata: BTreeMap<String, Value>,
}

impl Default for ExceptionConfig {
    fn default() -> Self {
        Self {
            handle_type: "interrupt".to_string(),
            default_outputs: None,
            metadata: BTreeMap::new(),
        }
    }
}

/// The per-component configuration recorded in a [`WorkflowSpec`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct NodeSpec {
    /// How batch inputs and outputs are mapped.
    pub io_configs: Option<CompIoConfig>,
    /// How streamed inputs and outputs are mapped.
    pub stream_io_configs: Option<CompIoConfig>,
    /// The abilities the component declares.
    pub abilities: Vec<ComponentAbility>,
    /// How many times a failing invocation is retried.
    pub max_retries: u32,
    /// The per-component timeout in seconds; a negative value means no timeout.
    pub timeout: f64,
    /// The failure policy for this component.
    pub exception_config: Option<ExceptionConfig>,
    /// An optional display name.
    pub name: Option<String>,
}

/// The static shape of a workflow: nodes, edges, and their configuration.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct WorkflowSpec {
    /// Batch edges, source id to target ids.
    pub edges: BTreeMap<String, Vec<String>>,
    /// Streaming edges, source id to target ids.
    pub stream_edges: BTreeMap<String, Vec<String>>,
    /// The configuration of every component, keyed by component id.
    pub comp_configs: BTreeMap<String, NodeSpec>,
    /// The source groups of every stream consumer, keyed by component id.
    pub stream_source_groups: BTreeMap<String, Vec<Vec<String>>>,
    /// The component ids that act as run entries (the start component).
    pub start_nodes: Vec<String>,
}

/// A workflow's card plus its spec plus the nesting limit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct WorkflowConfig {
    /// The workflow's identity and metadata.
    pub card: WorkflowCard,
    /// The workflow's static shape.
    pub spec: WorkflowSpec,
    /// How deeply sub-workflows may nest; Python defaults to 5.
    pub workflow_max_nesting_depth: u32,
}

impl Default for WorkflowConfig {
    /// A fresh card (fresh UUID id), an empty spec, and Python's nesting depth of 5.
    fn default() -> Self {
        Self {
            card: WorkflowCard::default(),
            spec: WorkflowSpec::default(),
            workflow_max_nesting_depth: 5,
        }
    }
}

/// Builds the storage key of one workflow version.
///
/// Ports Python's `generate_workflow_key`
/// (`openjiuwen/core/workflow/base.py`): the key is `"{workflow_id}_{workflow_version}"`,
/// so an unversioned workflow keys as `"<id>_"`. Bare concatenation is deliberate — an
/// id that itself contains `_` is not escaped, exactly like the Python helper.
///
/// ```
/// use openjiuwen_workflow::card::generate_workflow_key;
///
/// assert_eq!(generate_workflow_key("greeting", "1.0.0"), "greeting_1.0.0");
/// assert_eq!(generate_workflow_key("greeting", ""), "greeting_");
/// ```
#[must_use]
pub fn generate_workflow_key(workflow_id: &str, workflow_version: &str) -> String {
    format!("{workflow_id}_{workflow_version}")
}

/// Validates a component id the way Python's `Workflow._validate_comp_id` does.
///
/// An id must be non-empty, at most [`MAX_COMPONENT_ID_LEN`] characters, and consist
/// only of ASCII letters, digits, underscores, and hyphens.
///
/// # Errors
///
/// Returns the coded workflow error `WORKFLOW_COMPONENT_ID_INVALID` (100010) with the
/// Python reason text, so callers can surface it verbatim.
pub fn validate_component_id(id: &str) -> Result<()> {
    if id.is_empty() {
        return Err(component_id_invalid(id, "is None or empty", ""));
    }
    if id.chars().count() > MAX_COMPONENT_ID_LEN {
        return Err(component_id_invalid(
            id,
            "length must not between [1, 100]",
            "",
        ));
    }
    if !id
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        return Err(component_id_invalid(id, INVALID_ID_CHARS_REASON, ""));
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn new_card_copies_id_into_name() {
        let card = WorkflowCard::new("greeting");
        assert_eq!(card.id, "greeting");
        assert_eq!(card.name, "greeting");
        assert_eq!(card.description, "");
        assert_eq!(card.version, "");
        assert_eq!(card.input_params, None);
    }

    #[test]
    fn default_card_mints_uuid() {
        let card = WorkflowCard::default();
        assert_eq!(
            uuid::Uuid::parse_str(&card.id).unwrap().get_version_num(),
            4
        );
        assert_eq!(card.name, "");
        assert_ne!(card.id, WorkflowCard::default().id);
    }

    #[test]
    fn default_config_uses_python_defaults() {
        let config = WorkflowConfig::default();
        assert_eq!(config.workflow_max_nesting_depth, 5);
        assert!(config.spec.edges.is_empty());
        assert_eq!(
            uuid::Uuid::parse_str(&config.card.id)
                .unwrap()
                .get_version_num(),
            4
        );
    }

    #[test]
    fn default_exception_config_interrupts() {
        let config = ExceptionConfig::default();
        assert_eq!(config.handle_type, "interrupt");
        assert_eq!(config.default_outputs, None);
        assert!(config.metadata.is_empty());
    }

    #[test]
    fn default_node_spec_is_inert() {
        let spec = NodeSpec::default();
        assert!(spec.abilities.is_empty());
        assert_eq!(spec.max_retries, 0);
        assert!((spec.timeout - 0.0).abs() < f64::EPSILON);
        assert_eq!(spec.io_configs, None);
        assert_eq!(spec.stream_io_configs, None);
        assert_eq!(spec.exception_config, None);
        assert_eq!(spec.name, None);
    }

    #[test]
    fn generate_workflow_key_concatenates_id_and_version() {
        // Python: openjiuwen/core/workflow/base.py::generate_workflow_key
        assert_eq!(generate_workflow_key("wf", "1.0.0"), "wf_1.0.0");
        assert_eq!(generate_workflow_key("wf", ""), "wf_");
        assert_eq!(generate_workflow_key("", "v1"), "_v1");
        // Neither side is escaped or normalized.
        assert_eq!(generate_workflow_key("a_b", "1_2"), "a_b_1_2");
    }

    #[test]
    fn card_round_trips_through_json() {
        let card = WorkflowCard {
            id: "wf".to_string(),
            name: "Workflow".to_string(),
            description: "demo".to_string(),
            version: "1.0.0".to_string(),
            input_params: Some(json!({"type": "object"})),
        };
        let wire = serde_json::to_string(&card).unwrap();
        let parsed: WorkflowCard = serde_json::from_str(&wire).unwrap();
        assert_eq!(parsed, card);
    }

    #[test]
    fn spec_round_trips_through_json() {
        let mut spec = WorkflowSpec::default();
        spec.edges
            .insert("start".to_string(), vec!["end".to_string()]);
        spec.stream_edges
            .insert("llm".to_string(), vec!["end".to_string()]);
        spec.comp_configs.insert(
            "llm".to_string(),
            NodeSpec {
                abilities: vec![ComponentAbility::Stream, ComponentAbility::Collect],
                max_retries: 2,
                timeout: 30.0,
                name: Some("LLM".to_string()),
                exception_config: Some(ExceptionConfig {
                    default_outputs: Some(json!({"result": null})),
                    ..ExceptionConfig::default()
                }),
                ..NodeSpec::default()
            },
        );
        spec.stream_source_groups.insert(
            "end".to_string(),
            vec![vec!["llm".to_string(), "tool".to_string()]],
        );
        spec.start_nodes.push("start".to_string());

        let wire = serde_json::to_string(&spec).unwrap();
        let parsed: WorkflowSpec = serde_json::from_str(&wire).unwrap();
        assert_eq!(parsed, spec);
    }

    #[test]
    fn config_round_trips_through_json() {
        let config = WorkflowConfig {
            card: WorkflowCard::new("wf"),
            spec: WorkflowSpec::default(),
            workflow_max_nesting_depth: 2,
        };
        let wire = serde_json::to_string(&config).unwrap();
        let parsed: WorkflowConfig = serde_json::from_str(&wire).unwrap();
        assert_eq!(parsed, config);
    }

    #[test]
    fn missing_fields_default() {
        let config: WorkflowConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(config.workflow_max_nesting_depth, 5);
        assert_eq!(
            uuid::Uuid::parse_str(&config.card.id)
                .unwrap()
                .get_version_num(),
            4
        );

        let spec: WorkflowSpec = serde_json::from_str("{}").unwrap();
        assert_eq!(spec, WorkflowSpec::default());

        let node: NodeSpec = serde_json::from_str("{}").unwrap();
        assert_eq!(node, NodeSpec::default());

        let card: WorkflowCard = serde_json::from_str(r#"{"id": "wf"}"#).unwrap();
        assert_eq!(card.id, "wf");
        assert_eq!(card.name, "");
        assert_eq!(card.input_params, None);
    }

    #[test]
    fn unknown_fields_are_tolerated() {
        let config: WorkflowConfig = serde_json::from_str(
            r#"{"card": {"id": "wf", "extra": 1}, "spec": {"nodes": []}, "later": true,
                "workflow_max_nesting_depth": 1}"#,
        )
        .unwrap();
        assert_eq!(config.card.id, "wf");
        assert_eq!(config.workflow_max_nesting_depth, 1);

        let node: NodeSpec =
            serde_json::from_str(r#"{"timeout": 12.5, "unknown_key": {"a": 1}}"#).unwrap();
        assert!((node.timeout - 12.5).abs() < f64::EPSILON);
    }

    #[test]
    fn exception_config_keeps_unknown_keys() {
        let config: ExceptionConfig =
            serde_json::from_str(r#"{"handle_type": "skip", "retries": 3}"#).unwrap();
        assert_eq!(config.handle_type, "skip");
        assert_eq!(config.metadata.get("retries"), Some(&json!(3)));
        let wire = serde_json::to_value(&config).unwrap();
        assert_eq!(wire["retries"], json!(3));
    }

    #[test]
    fn ability_wire_names_are_snake_case() {
        let node: NodeSpec =
            serde_json::from_str(r#"{"abilities": ["invoke", "transform"]}"#).unwrap();
        assert_eq!(
            node.abilities,
            vec![ComponentAbility::Invoke, ComponentAbility::Transform]
        );
    }

    #[test]
    fn validate_component_id_accepts_valid_ids() {
        for id in ["a", "start", "start_1", "node-2", "A9_-", &"n".repeat(100)] {
            validate_component_id(id).unwrap();
        }
    }

    #[test]
    fn validate_component_id_rejects_empty() {
        let error = validate_component_id("").unwrap_err();
        assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::Workflow);
        assert_eq!(
            error.message(),
            "[100010] the component id is invalid for component '', \
             reason='is None or empty', workflow=''"
        );
    }

    #[test]
    fn validate_component_id_rejects_long_ids() {
        let error = validate_component_id(&"n".repeat(101)).unwrap_err();
        assert_eq!(
            error.message(),
            format!(
                "[100010] the component id is invalid for component '{}', \
                 reason='length must not between [1, 100]', workflow=''",
                "n".repeat(101)
            )
        );
    }

    #[test]
    fn validate_component_id_rejects_bad_characters() {
        for id in ["a b", "a.b", "a/b", "节点"] {
            let error = validate_component_id(id).unwrap_err();
            assert_eq!(
                error.message(),
                format!(
                    "[100010] the component id is invalid for component '{id}', \
                     reason='{INVALID_ID_CHARS_REASON}', workflow=''"
                )
            );
        }
    }

    #[test]
    fn invalid_chars_reason_uses_python_text() {
        assert_eq!(
            INVALID_ID_CHARS_REASON,
            "only support letters (a–z, A–Z), digits (0–9), underscores (_) or hyphens (-)"
        );
    }
}
