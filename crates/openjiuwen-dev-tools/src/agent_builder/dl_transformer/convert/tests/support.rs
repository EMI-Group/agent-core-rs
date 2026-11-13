//! Shared fixtures for the DL transformer tests.

use super::super::*;
use serde_json::json;

pub(super) fn variable(
    name: &str,
    var_type: Option<&str>,
    description: Option<&str>,
    value: Option<Value>,
) -> DlVariable {
    DlVariable {
        name: name.to_string(),
        var_type: var_type.map(str::to_string),
        description: description.map(str::to_string),
        value,
        ..DlVariable::default()
    }
}

/// Builds one input variable bound to `value`.
pub(super) fn input(name: &str, var_type: Option<&str>, value: Value) -> DlVariable {
    variable(name, var_type, None, Some(value))
}

/// Builds one output variable.
pub(super) fn output(name: &str, var_type: Option<&str>, description: Option<&str>) -> DlVariable {
    variable(name, var_type, description, None)
}

/// Builds a node without a successor or parameters.
pub(super) fn node(id: &str, node_type: &str, description: &str) -> DlNode {
    DlNode {
        id: id.to_string(),
        node_type: node_type.to_string(),
        description: description.to_string(),
        ..DlNode::default()
    }
}

/// Sets a node's successor.
pub(super) fn with_next(mut node: DlNode, next: &str) -> DlNode {
    node.next = Some(next.to_string());
    node
}

/// Sets a node's declared inputs.
pub(super) fn with_inputs(mut node: DlNode, inputs: Vec<DlVariable>) -> DlNode {
    node.parameters.inputs = inputs;
    node
}

/// Sets a node's declared outputs.
pub(super) fn with_outputs(mut node: DlNode, outputs: Vec<DlVariable>) -> DlNode {
    node.parameters.outputs = outputs;
    node
}

/// Sets one `parameters.configs` entry.
pub(super) fn with_config(mut node: DlNode, key: &str, value: Value) -> DlNode {
    node.parameters.configs.insert(key.to_string(), value);
    node
}

/// Sets a node's branch conditions.
pub(super) fn with_conditions(mut node: DlNode, conditions: Vec<DlCondition>) -> DlNode {
    node.parameters.conditions = conditions;
    node
}

/// Builds a condition with a branch name and a successor.
pub(super) fn condition(branch: &str, next: &str) -> DlCondition {
    DlCondition {
        branch: branch.to_string(),
        next: Some(next.to_string()),
        ..DlCondition::default()
    }
}

/// Transforms a document, panicking on error.
pub(super) fn transform(nodes: Vec<DlNode>) -> PlatformWorkflow {
    DlTransformer::new()
        .transform(&DlDocument { nodes })
        .expect("document transforms")
}

/// Transforms a document and returns its platform JSON.
pub(super) fn transform_value(nodes: Vec<DlNode>) -> Value {
    transform(nodes).to_value().expect("document serializes")
}

/// The platform JSON of one node.
pub(super) fn node_json(nodes: Vec<DlNode>, id: &str) -> Value {
    let value = transform_value(nodes);
    value["nodes"]
        .as_array()
        .expect("nodes array")
        .iter()
        .find(|node| node["id"] == json!(id))
        .unwrap_or_else(|| panic!("node {id} is present"))
        .clone()
}

/// The platform `data` of one node.
pub(super) fn node_data(nodes: Vec<DlNode>, id: &str) -> Value {
    node_json(nodes, id)["data"].clone()
}

/// The platform edges of a document.
pub(super) fn edges_json(nodes: Vec<DlNode>) -> Value {
    transform_value(nodes)["edges"].clone()
}
