//! The platform workflow document produced by the DL transformer.
//!
//! Ports the output half of Python's `dl_transformer/models.py`: a workflow of
//! [`PlatformNode`]s and [`PlatformEdge`]s whose wire shape is
//! `{"nodes": [{"id", "type", "meta", "data"}], "edges": [{"source_node_id",
//! "target_node_id", "source_port_id"?}]}`.
//!
//! Unlike the DL document, the platform document is *derived*: [`PlatformNode`]
//! carries the numeric platform `type` id, a `meta` position, and a `data` object
//! that mixes the node title with the per-type configuration built by the
//! converters. [`PlatformWorkflow::to_value`] applies Python's
//! `ConverterUtils.convert_to_dict` null policy to the whole document, so an
//! unset optional leaves no key behind while `0`, `""`, and `false` survive.
//!
//! [`PlatformWorkflow::to_workflow_dsl`] adapts the document to the simplified
//! [`WorkflowDsl`] of [`crate::agent_builder::workflow_dsl`] and validates it, so
//! a transformed document is immediately usable by the rest of the builder.

use openjiuwen_common::error::{Error, Result};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::agent_builder::workflow_dsl::{DslEdge, DslNode, WorkflowDsl};

/// A complete platform workflow document.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
pub struct PlatformWorkflow {
    /// The nodes of the workflow, in DL document order.
    pub nodes: Vec<PlatformNode>,
    /// The edges of the workflow, in generation order.
    pub edges: Vec<PlatformEdge>,
}

impl PlatformWorkflow {
    /// Serializes the workflow to a JSON value, dropping every `null`.
    ///
    /// This is Python's `ConverterUtils.convert_to_dict` policy: `null` object
    /// entries and `null` array elements are removed recursively, while scalar
    /// values such as `0`, `""`, and `false` are kept verbatim.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the document cannot be serialized, with
    /// the serde error in the message.
    pub fn to_value(&self) -> Result<Value> {
        let value = serde_json::to_value(self).map_err(|error| {
            Error::validation(format!("failed to serialize platform workflow: {error}"))
        })?;
        Ok(drop_nulls(value))
    }

    /// Serializes the workflow to pretty-printed JSON (see [`Self::to_value`]).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the document cannot be serialized, with
    /// the serde error in the message.
    pub fn to_json_string(&self) -> Result<String> {
        let value = self.to_value()?;
        serde_json::to_string_pretty(&value).map_err(|error| {
            Error::validation(format!("failed to serialize platform workflow: {error}"))
        })
    }

    /// Adapts the workflow to a simplified [`WorkflowDsl`] document.
    ///
    /// Each [`PlatformNode`] keeps its id, uses the platform `type` id as its
    /// `node_type`, takes `data.title` for both `title` and `description`, copies
    /// the whole `data` object into `data`, and leaves the typed `inputs`/`outputs`
    /// maps empty because the platform configuration lives inside `data`.
    /// Each [`PlatformEdge`] becomes an unconditional `source -> target` edge.
    /// The resulting document is validated before it is returned.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the document has no nodes or an edge
    /// names a node that does not exist.
    pub fn to_workflow_dsl(&self) -> Result<WorkflowDsl> {
        let nodes = self.nodes.iter().map(PlatformNode::to_dsl_node).collect();
        let edges = self
            .edges
            .iter()
            .map(|edge| DslEdge {
                source: edge.source_node_id.clone(),
                target: edge.target_node_id.clone(),
                condition: None,
            })
            .collect();

        let workflow = WorkflowDsl {
            name: String::new(),
            description: String::new(),
            nodes,
            edges,
        };
        workflow.validate()?;
        Ok(workflow)
    }
}

/// One node of a [`PlatformWorkflow`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
pub struct PlatformNode {
    /// The node id, copied from the DL node.
    pub id: String,
    /// The platform node type id (`"1"` for start, `"19"` for a plugin, ...),
    /// serialized as `"type"`.
    #[serde(rename = "type")]
    pub node_type: String,
    /// Layout metadata; always `{"position": {"x": ..., "y": ...}}` here.
    pub meta: Value,
    /// The node configuration: `title` plus the per-type platform fields.
    pub data: Value,
}

impl PlatformNode {
    /// Adapts the node to a [`DslNode`], keeping the whole platform `data` object.
    fn to_dsl_node(&self) -> DslNode {
        let title = self
            .data
            .get("title")
            .and_then(Value::as_str)
            .unwrap_or_default();
        DslNode {
            id: self.id.clone(),
            node_type: self.node_type.clone(),
            title: title.to_string(),
            description: title.to_string(),
            inputs: std::collections::BTreeMap::new(),
            outputs: std::collections::BTreeMap::new(),
            data: self.data.clone(),
        }
    }
}

/// A directed connection between two [`PlatformNode`]s.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
pub struct PlatformEdge {
    /// The id of the node the edge leaves.
    pub source_node_id: String,
    /// The id of the node the edge enters.
    pub target_node_id: String,
    /// The source port the edge leaves (`"0"`, a branch name, or an intent id);
    /// absent for an unconditional edge.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_port_id: Option<String>,
}

/// Removes every `null` from `value`, recursively.
fn drop_nulls(value: Value) -> Value {
    match value {
        Value::Object(map) => Value::Object(
            map.into_iter()
                .filter(|(_, entry)| !entry.is_null())
                .map(|(key, entry)| (key, drop_nulls(entry)))
                .collect::<Map<String, Value>>(),
        ),
        Value::Array(items) => Value::Array(
            items
                .into_iter()
                .filter(|item| !item.is_null())
                .map(drop_nulls)
                .collect(),
        ),
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn node(id: &str, node_type: &str, data: Value) -> PlatformNode {
        PlatformNode {
            id: id.to_string(),
            node_type: node_type.to_string(),
            meta: json!({"position": {"x": 0, "y": 0}}),
            data,
        }
    }

    #[test]
    fn to_value_drops_nulls_and_keeps_falsy_scalars() {
        let workflow = PlatformWorkflow {
            nodes: vec![PlatformNode {
                id: "start".to_string(),
                node_type: "1".to_string(),
                meta: json!({"position": {"x": 0, "y": 0}, "note": null}),
                data: json!({
                    "title": "",
                    "outputs": {"properties": {}, "required": []},
                    "flag": false,
                    "count": 0,
                    "missing": null,
                    "list": [null, 1, "", false, null],
                }),
            }],
            edges: vec![PlatformEdge {
                source_node_id: "start".to_string(),
                target_node_id: "end".to_string(),
                source_port_id: None,
            }],
        };

        let value = workflow.to_value().expect("serializes");
        let rendered = value["nodes"][0]["data"].clone();
        assert_eq!(rendered["title"], json!(""));
        assert_eq!(rendered["flag"], json!(false));
        assert_eq!(rendered["count"], json!(0));
        assert_eq!(rendered["list"], json!([1, "", false]));
        assert_eq!(
            rendered["outputs"],
            json!({"properties": {}, "required": []})
        );
        assert!(rendered.get("missing").is_none());
        assert!(value["nodes"][0]["meta"].get("note").is_none());
        // `source_port_id` is absent rather than null.
        assert!(value["edges"][0].get("source_port_id").is_none());
        assert_eq!(value["edges"][0]["source_node_id"], json!("start"));

        let text = workflow.to_json_string().expect("serializes");
        assert!(text.contains("\"source_node_id\": \"start\""));
        assert!(!text.contains("null"));
    }

    #[test]
    fn to_value_keeps_source_port_when_present() {
        let workflow = PlatformWorkflow {
            nodes: vec![node("code", "10", json!({"title": "code"}))],
            edges: vec![PlatformEdge {
                source_node_id: "code".to_string(),
                target_node_id: "code".to_string(),
                source_port_id: Some("0".to_string()),
            }],
        };
        let value = workflow.to_value().expect("serializes");
        assert_eq!(value["edges"][0]["source_port_id"], json!("0"));
    }

    #[test]
    fn to_workflow_dsl_maps_nodes_and_edges() {
        let workflow = PlatformWorkflow {
            nodes: vec![
                node("start", "1", json!({"title": "entry"})),
                node("end", "2", json!({"title": "exit", "inputs": {}})),
            ],
            edges: vec![PlatformEdge {
                source_node_id: "start".to_string(),
                target_node_id: "end".to_string(),
                source_port_id: Some("0".to_string()),
            }],
        };

        let dsl = workflow.to_workflow_dsl().expect("valid DSL");
        assert_eq!(dsl.nodes.len(), 2);
        assert_eq!(dsl.nodes[0].id, "start");
        assert_eq!(dsl.nodes[0].node_type, "1");
        assert_eq!(dsl.nodes[0].title, "entry");
        assert_eq!(dsl.nodes[0].description, "entry");
        assert_eq!(dsl.nodes[0].data, json!({"title": "entry"}));
        assert!(dsl.nodes[0].inputs.is_empty());
        assert!(dsl.nodes[0].outputs.is_empty());
        assert_eq!(dsl.nodes[1].data, json!({"title": "exit", "inputs": {}}));
        assert_eq!(
            dsl.edges,
            vec![DslEdge {
                source: "start".to_string(),
                target: "end".to_string(),
                condition: None,
            }]
        );
        dsl.validate().expect("dsl validates");
    }

    #[test]
    fn to_workflow_dsl_rejects_dangling_edges() {
        let workflow = PlatformWorkflow {
            nodes: vec![node("start", "1", json!({"title": "entry"}))],
            edges: vec![PlatformEdge {
                source_node_id: "start".to_string(),
                target_node_id: "ghost".to_string(),
                source_port_id: None,
            }],
        };
        let error = workflow.to_workflow_dsl().expect_err("dangling edge");
        assert!(error.to_string().contains("ghost"), "{error}");
    }

    #[test]
    fn platform_node_serde_keeps_python_field_names() {
        let value =
            serde_json::to_value(node("n", "3", json!({"title": "t"}))).expect("serializes");
        assert_eq!(
            value,
            json!({
                "id": "n",
                "type": "3",
                "meta": {"position": {"x": 0, "y": 0}},
                "data": {"title": "t"},
            })
        );

        let parsed: PlatformNode =
            serde_json::from_value(json!({"id": "n", "type": "3", "meta": {}, "data": {}}))
                .expect("parses");
        assert_eq!(parsed.node_type, "3");

        let edge: PlatformEdge = serde_json::from_value(json!({
            "source_node_id": "a",
            "target_node_id": "b",
        }))
        .expect("parses");
        assert_eq!(edge.source_port_id, None);
    }
}
