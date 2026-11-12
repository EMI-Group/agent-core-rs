//! The serialized workflow DSL: nodes, edges, and graph-level validation.
//!
//! Ports the workflow-document shape consumed by the Python
//! `openjiuwen.dev_tools.agent_builder` workflow builders. The DSL is a plain
//! JSON document (`{"name", "description", "nodes", "edges"}`) that can be
//! handed to the platform or converted into a kernel workflow; the node `type`
//! keeps the Python wire name, everything else is `snake_case`.
//!
//! Validation is deterministic and reports the offending id:
//! a document must contain at least one node, node ids must be unique, and every
//! edge endpoint must name an existing node.

use std::collections::{BTreeMap, BTreeSet};

use openjiuwen_common::error::{Error, Result};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// One node of a workflow DSL document.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct DslNode {
    /// The node id, unique inside the document and referenced by [`DslEdge`].
    pub id: String,
    /// The node kind; serialized as `"type"` to match the Python wire format.
    #[serde(rename = "type")]
    pub node_type: String,
    /// A human-readable node title.
    pub title: String,
    /// A human-readable node description.
    pub description: String,
    /// Declared inputs, keyed by input name.
    pub inputs: BTreeMap<String, Value>,
    /// Declared outputs, keyed by output name.
    pub outputs: BTreeMap<String, Value>,
    /// Free-form node parameters, preserved verbatim.
    pub data: Value,
}

/// A directed connection between two [`DslNode`]s.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct DslEdge {
    /// The id of the node the edge leaves.
    pub source: String,
    /// The id of the node the edge enters.
    pub target: String,
    /// An optional condition that must hold for the edge to be taken.
    pub condition: Option<String>,
}

/// A complete workflow DSL document.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct WorkflowDsl {
    /// The workflow name.
    pub name: String,
    /// A human-readable workflow description.
    pub description: String,
    /// The nodes of the graph, in document order.
    pub nodes: Vec<DslNode>,
    /// The edges of the graph, in document order.
    pub edges: Vec<DslEdge>,
}

impl WorkflowDsl {
    /// Checks the structural invariants of the document.
    ///
    /// The document must contain at least one node, every node id must be
    /// unique, and every edge endpoint must resolve to a node. Error messages
    /// name the offending id.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the document has no nodes
    /// (`workflow has no nodes`), a repeated node id
    /// (`duplicate node id: <id>`), or an edge whose source or target is not a
    /// node (`edge references unknown node: <id>`).
    pub fn validate(&self) -> Result<()> {
        if self.nodes.is_empty() {
            return Err(Error::validation("workflow has no nodes"));
        }

        let mut ids = BTreeSet::new();
        for node in &self.nodes {
            if !ids.insert(node.id.as_str()) {
                return Err(Error::validation(format!("duplicate node id: {}", node.id)));
            }
        }

        for edge in &self.edges {
            for endpoint in [edge.source.as_str(), edge.target.as_str()] {
                if !ids.contains(endpoint) {
                    return Err(Error::validation(format!(
                        "edge references unknown node: {endpoint}"
                    )));
                }
            }
        }

        Ok(())
    }

    /// Looks up a node by id.
    #[must_use]
    pub fn node(&self, id: &str) -> Option<&DslNode> {
        self.nodes.iter().find(|node| node.id == id)
    }

    /// Serializes the document as pretty-printed JSON.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the document cannot be serialized,
    /// with the serde error in the message.
    pub fn to_json_string(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(|error| {
            Error::validation(format!("failed to serialize workflow DSL: {error}"))
        })
    }

    /// Parses a workflow DSL document from JSON.
    ///
    /// Structural validation is not performed here; call [`Self::validate`] on
    /// the parsed document when it is about to be executed or persisted.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the text is not a workflow DSL
    /// document, with the serde error in the message.
    pub fn from_json_str(text: &str) -> Result<Self> {
        serde_json::from_str(text).map_err(|error| {
            Error::validation(format!("failed to parse workflow DSL JSON: {error}"))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;

    /// A two-node, one-edge graph: `start` -> `end`.
    fn sample() -> WorkflowDsl {
        WorkflowDsl {
            name: "demo".to_string(),
            description: "a demo workflow".to_string(),
            nodes: vec![
                DslNode {
                    id: "start".to_string(),
                    node_type: "start".to_string(),
                    title: "Start".to_string(),
                    description: "entry".to_string(),
                    inputs: BTreeMap::from([("query".to_string(), json!({"type": "string"}))]),
                    outputs: BTreeMap::from([("text".to_string(), json!({"type": "string"}))]),
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
                condition: None,
            }],
        }
    }

    #[test]
    fn round_trip_preserves_wire_shape() {
        let dsl = sample();
        let text = dsl.to_json_string().expect("serialize");
        let value: Value = serde_json::from_str(&text).expect("reparse");

        assert_eq!(value["name"], json!("demo"));
        assert_eq!(value["nodes"][0]["id"], json!("start"));
        assert_eq!(value["nodes"][0]["type"], json!("start"));
        assert_eq!(
            value["nodes"][0]["inputs"]["query"]["type"],
            json!("string")
        );
        assert_eq!(value["nodes"][1]["data"], Value::Null);
        assert!(
            value["edges"][0]["condition"].is_null(),
            "an absent condition serializes as null"
        );

        let parsed = WorkflowDsl::from_json_str(&text).expect("parse");
        assert_eq!(parsed, dsl);
    }

    #[test]
    fn condition_survives_round_trip() {
        let mut dsl = sample();
        dsl.edges[0].condition = Some("score > 0.5".to_string());
        let text = dsl.to_json_string().expect("serialize");
        let value: Value = serde_json::from_str(&text).expect("reparse");
        assert_eq!(value["edges"][0]["condition"], json!("score > 0.5"));
        assert_eq!(WorkflowDsl::from_json_str(&text).expect("parse"), dsl);
    }

    #[test]
    fn validate_accepts_connected_graph() {
        sample().validate().expect("graph is valid");
    }

    #[test]
    fn validate_rejects_no_nodes() {
        let dsl = WorkflowDsl::default();
        let error = dsl.validate().expect_err("empty graph is invalid");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert_eq!(error.to_string(), "validation: workflow has no nodes");
    }

    #[test]
    fn validate_rejects_duplicate_ids() {
        let mut dsl = sample();
        dsl.nodes[1].id = "start".to_string();
        dsl.edges.clear();
        let error = dsl.validate().expect_err("duplicate id is invalid");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("duplicate node id: start"));
    }

    #[test]
    fn validate_rejects_dangling_source() {
        let mut dsl = sample();
        dsl.edges[0].source = "ghost-source".to_string();
        let error = dsl.validate().expect_err("unknown source is invalid");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error
            .to_string()
            .contains("edge references unknown node: ghost-source"));
    }

    #[test]
    fn validate_rejects_dangling_target() {
        let mut dsl = sample();
        dsl.edges[0].target = "ghost-target".to_string();
        let error = dsl.validate().expect_err("unknown target is invalid");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error
            .to_string()
            .contains("edge references unknown node: ghost-target"));
    }

    #[test]
    fn node_lookup_hits_and_misses() {
        let dsl = sample();
        assert_eq!(dsl.node("end").map(|node| node.title.as_str()), Some("End"));
        assert!(dsl.node("missing").is_none());
        assert!(WorkflowDsl::default().node("start").is_none());
    }

    #[test]
    fn from_json_str_reports_validation_errors() {
        for text in ["not json", "[1,2]"] {
            let error = WorkflowDsl::from_json_str(text).expect_err("invalid document");
            assert_eq!(error.kind(), ErrorKind::Validation);
            assert!(error.to_string().contains("workflow DSL"));
            assert!(error.to_string().contains("failed to parse"));
        }
    }

    #[test]
    fn from_json_str_tolerates_absent_fields() {
        let dsl = WorkflowDsl::from_json_str(r#"{"name": "only-name"}"#).expect("parse");
        assert_eq!(dsl.name, "only-name");
        assert_eq!(dsl.nodes.len(), 0);
        assert_eq!(dsl.edges.len(), 0);
    }
}
