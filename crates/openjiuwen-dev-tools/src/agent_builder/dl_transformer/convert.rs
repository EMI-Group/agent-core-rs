//! The DL -> platform workflow DSL transformer.
//!
//! Ports `openjiuwen.dev_tools.agent_builder.builders.workflow.dl_transformer`:
//! [`DlTransformer`] walks a [`DlDocument`] in wire order and turns every node
//! into one [`PlatformNode`] plus its outgoing [`PlatformEdge`]s, using the
//! Python converter registry's per-type rules.
//!
//! # Conversion contract
//!
//! - The document is validated first, so a blank id, a duplicate id, or a
//!   dangling `next` link is rejected before any node is touched.
//! - Nodes keep their document order; the platform `type` is Python's numeric
//!   [`DlNodeType::dsl_id`] and `meta.position` steps by
//!   `POSITION_STEP` (`0`, `20`, `40`, ...) on both axes.
//! - Unknown node types are a validation error; Python logs a warning and drops
//!   the node, which would silently lose part of a workflow.
//! - The shared variable index restarts at `0` for every node: inputs consume
//!   indices first, outputs afterwards, one index per declared variable.
//! - An intent without a [`DlCondition::intent_id`] receives a fresh uuid4, so
//!   generation is non-deterministic per conversion exactly as in Python while
//!   an id supplied by the DL document is used verbatim.
//! - Intent ports reuse the resolved id, so a generated id always equals the
//!   `source_port_id` of its intent edge.
//!
//! Python's `ConverterUtils.convert_to_dict` null policy is applied later, in
//! [`PlatformWorkflow::to_value`]; the converters here simply leave optional
//! fields unset.
//!
//! [`PlatformWorkflow::to_value`]: super::platform::PlatformWorkflow::to_value

use openjiuwen_common::error::{Error, Result};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::model::{DlCondition, DlDocument, DlNode, DlNodeType, DlVariable};
use super::platform::{PlatformEdge, PlatformNode, PlatformWorkflow};

mod nodes;

#[cfg(test)]
mod tests;

/// The `SourceType` of a `${node.variable}` reference.
pub(super) const REF: &str = "ref";
/// The `SourceType` of a literal value.
pub(super) const CONSTANT: &str = "constant";
/// The platform layout step between consecutive nodes (Python's `x += 20`).
pub(super) const POSITION_STEP: usize = 20;
/// The expression marking a branch as the fallback branch.
pub(super) const DEFAULT_EXPRESSION: &str = "default";
/// The plugin version used when neither the matched resource nor the node's
/// `configs.plugin_version` supplies one.
pub(super) const DRAFT_PLUGIN_VERSION: &str = "draft";

/// The plugin information a plugin node resolves against.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct PluginResource {
    /// The tool id the DL node's `configs.tool_id` is matched against.
    pub tool_id: String,
    /// The human-readable tool name.
    pub tool_name: String,
    /// The plugin id the tool belongs to.
    pub plugin_id: String,
    /// The human-readable plugin name.
    pub plugin_name: String,
    /// The plugin version; blank means the node's `configs.plugin_version`, then
    /// `"draft"`, applies.
    pub plugin_version: String,
}

/// Transforms DL documents into platform workflow documents.
///
/// The transformer is stateless, so one value can be reused for any number of
/// documents and [`DlTransformer::default`] is equivalent to
/// [`DlTransformer::new`].
#[derive(Debug, Clone, Copy, Default)]
pub struct DlTransformer;

impl DlTransformer {
    /// Creates a transformer.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Transforms a parsed DL document, resolving no plugin resources.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an invalid document, an unknown node
    /// type, a missing or malformed node configuration, or an unresolvable
    /// reference expression.
    pub fn transform(&self, document: &DlDocument) -> Result<PlatformWorkflow> {
        self.transform_with_resources(document, &[])
    }

    /// Transforms a parsed DL document, resolving plugin nodes against
    /// `resources`.
    ///
    /// A plugin node matches the first resource whose
    /// [`PluginResource::tool_id`] equals its non-blank `configs.tool_id`.
    /// Each plugin parameter resolves per field to the first non-blank value of
    /// the matched resource, the node's matching `configs` entry, and the field
    /// default, which is `"draft"` for `pluginVersion` and the empty string for
    /// the other four.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] under the same conditions as
    /// [`Self::transform`].
    pub fn transform_with_resources(
        &self,
        document: &DlDocument,
        resources: &[PluginResource],
    ) -> Result<PlatformWorkflow> {
        document.validate()?;

        let mut workflow = PlatformWorkflow::default();
        for (position, node) in document.nodes.iter().enumerate() {
            let offset = position * POSITION_STEP;
            let (platform_node, edges) = convert_node(node, resources, offset)?;
            workflow.nodes.push(platform_node);
            workflow.edges.extend(edges);
        }

        Ok(workflow)
    }

    /// Parses `dl_text` (fenced or bare JSON), validates it, and transforms it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the text is not a DL array of nodes,
    /// when the document is structurally invalid, or when a node cannot be
    /// converted.
    pub fn transform_str(&self, dl_text: &str) -> Result<PlatformWorkflow> {
        let document = DlDocument::from_json_str(dl_text)?;
        document.validate()?;
        self.transform(&document)
    }
}

/// Converts one DL node into its platform node and outgoing edges.
fn convert_node(
    node: &DlNode,
    resources: &[PluginResource],
    position: usize,
) -> Result<(PlatformNode, Vec<PlatformEdge>)> {
    let kind = node.kind()?;
    let mut conversion = NodeConversion::new(node, resources);
    let data = nodes::node_data(&mut conversion, kind)?;
    let edges = conversion.edges(kind)?;

    let platform_node = PlatformNode {
        id: node.id.clone(),
        node_type: kind.dsl_id().to_string(),
        meta: serde_json::json!({"position": {"x": position, "y": position}}),
        data: Value::Object(data),
    };

    Ok((platform_node, edges))
}

/// The per-node conversion state: the node, the plugin resources, and the
/// variable index counter.
pub(super) struct NodeConversion<'a> {
    /// The DL node being converted.
    pub(super) node: &'a DlNode,
    /// The plugin resources available to a plugin node.
    pub(super) resources: &'a [PluginResource],
    /// The next variable index handed out.
    index: usize,
    /// The intent id per branch, filled while an intent-detection node's data is
    /// built and consumed by its edges.
    intent_ports: Vec<(String, String)>,
}

impl<'a> NodeConversion<'a> {
    /// Creates the conversion state for `node`.
    pub(super) fn new(node: &'a DlNode, resources: &'a [PluginResource]) -> Self {
        Self {
            node,
            resources,
            index: 0,
            intent_ports: Vec::new(),
        }
    }

    /// The DL node id, used in every validation message.
    pub(super) fn id(&self) -> &str {
        &self.node.id
    }

    /// Reads a required string configuration entry.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `configs.<key>` is absent
    /// (`configs.<key> is required`) or is not a JSON string.
    pub(super) fn config(&self, key: &str) -> Result<String> {
        self.optional_config(key)?.ok_or_else(|| {
            Error::validation(format!("node {}: configs.{key} is required", self.node.id))
        })
    }

    /// Reads an optional string configuration entry.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `configs.<key>` is present but is not a
    /// JSON string.
    pub(super) fn optional_config(&self, key: &str) -> Result<Option<String>> {
        match self.node.parameters.configs.get(key) {
            Some(Value::String(text)) => Ok(Some(text.clone())),
            Some(_) => Err(Error::validation(format!(
                "node {}: configs.{key} must be a string",
                self.node.id
            ))),
            None => Ok(None),
        }
    }

    /// Reads a required non-blank string configuration entry.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `configs.<key>` is absent or is not a
    /// JSON string (see [`Self::config`]) or when it is blank.
    pub(super) fn prompt(&self, key: &str) -> Result<String> {
        let value = self.config(key)?;
        if value.trim().is_empty() {
            return Err(Error::validation(format!(
                "node {}: configs.{key} must not be blank",
                self.node.id
            )));
        }
        Ok(value)
    }

    /// Hands out the next variable index.
    fn take_index(&mut self) -> usize {
        let index = self.index;
        self.index += 1;
        index
    }

    /// Converts the node's declared inputs into `(name, variable)` pairs, in
    /// document order.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an input without a value or for a value
    /// that looks like a reference but is not `${node.variable}`.
    fn input_pairs(&mut self) -> Result<Vec<(String, Value)>> {
        let mut pairs = Vec::with_capacity(self.node.parameters.inputs.len());
        for item in &self.node.parameters.inputs {
            let index = self.take_index();
            let value = match item.value.as_ref() {
                Some(Value::String(text)) if text.contains("${") => {
                    let reference = convert_reference(&self.node.id, text)?;
                    with_index(reference, index)
                }
                Some(value) => Value::Object(object_with([
                    ("type", Value::String(CONSTANT.to_string())),
                    ("content", value.clone()),
                    (
                        "schema",
                        Value::Object(object_with([("type", schema_type(item))])),
                    ),
                    ("extra", index_value(index)),
                ])),
                None => {
                    return Err(Error::validation(format!(
                        "node {}: input {} has no value",
                        self.node.id, item.name
                    )))
                }
            };
            pairs.push((item.name.clone(), value));
        }
        Ok(pairs)
    }

    /// Converts the node's declared inputs into the `input_parameters` object.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] under the same conditions as the input
    /// conversion it wraps.
    pub(super) fn inputs(&mut self) -> Result<Map<String, Value>> {
        let pairs = self.input_pairs()?;
        Ok(pairs.into_iter().collect())
    }

    /// Converts the node's declared outputs into an `object` schema.
    pub(super) fn outputs(&mut self) -> Map<String, Value> {
        let outputs = self.node.parameters.outputs.clone();
        self.outputs_for(&outputs)
    }

    /// Converts `outputs` into an `object` schema, consuming one index each.
    pub(super) fn outputs_for(&mut self, outputs: &[DlVariable]) -> Map<String, Value> {
        let mut field = object_with([
            ("type", Value::String("object".to_string())),
            ("properties", Value::Object(Map::new())),
            ("required", Value::Array(Vec::new())),
        ]);
        for item in outputs {
            let index = self.take_index();
            add_output_property(&mut field, &output_names(&item.name), item, index);
        }
        field
    }

    /// Builds the outgoing edges of the node, per its type.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when a branch or intent condition names no
    /// next node; the document validation already rejects unknown targets.
    fn edges(&self, kind: DlNodeType) -> Result<Vec<PlatformEdge>> {
        let source = self.node.id.as_str();
        let mut edges = Vec::new();

        match kind {
            DlNodeType::Branch => {
                for condition in &self.node.parameters.conditions {
                    let target = self.require_target(condition)?;
                    edges.push(edge(source, target, Some(condition.branch.as_str())));
                }
            }
            DlNodeType::IntentDetection => {
                for condition in &self.node.parameters.conditions {
                    let target = self.require_target(condition)?;
                    if condition.expression.as_deref() == Some(DEFAULT_EXPRESSION) {
                        edges.push(edge(source, target, Some("0")));
                    } else if let Some((_, port)) = self
                        .intent_ports
                        .iter()
                        .find(|(branch, _)| *branch == condition.branch)
                    {
                        edges.push(edge(source, target, Some(port.as_str())));
                    }
                }
            }
            DlNodeType::Code => {
                if let Some(target) = non_blank(self.node.next.as_deref()) {
                    edges.push(edge(source, target, Some("0")));
                }
            }
            _ => {
                if let Some(target) = non_blank(self.node.next.as_deref()) {
                    edges.push(edge(source, target, None));
                }
            }
        }

        Ok(edges)
    }

    /// The target of a branching node's condition.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the condition carries no `next` node.
    fn require_target<'c>(&self, condition: &'c DlCondition) -> Result<&'c str> {
        non_blank(condition.next.as_deref()).ok_or_else(|| {
            Error::validation(format!(
                "node {}: condition {} has no next node",
                self.node.id, condition.branch
            ))
        })
    }
}

/// Builds Python's `LLM_MODEL_CONFIG` platform value.
pub(super) fn llm_model_config() -> Value {
    serde_json::json!({
        "id": "52",
        "name": "siliconf-qwen3-8b",
        "type": "Qwen/Qwen3-8B",
    })
}

/// Builds one `{"type": "template", "content": ...}` platform prompt.
pub(super) fn template(content: &str) -> Value {
    Value::Object(object_with([
        ("type", Value::String("template".to_string())),
        ("content", Value::String(content.to_string())),
    ]))
}

/// Builds Python's `convert_llm_param` result.
pub(super) fn llm_param(system_prompt: &str, user_prompt: &str) -> Map<String, Value> {
    object_with([
        ("systemPrompt", template(system_prompt)),
        ("prompt", template(user_prompt)),
        ("mode", llm_model_config()),
    ])
}

/// Builds a JSON object from ordered entries, preserving insertion order.
pub(super) fn object_with<const N: usize>(entries: [(&str, Value); N]) -> Map<String, Value> {
    entries
        .into_iter()
        .map(|(key, value)| (key.to_string(), value))
        .collect()
}

/// The schema `type` of an input variable, defaulting to `string`.
pub(super) fn schema_type(item: &DlVariable) -> Value {
    Value::String(
        item.var_type
            .clone()
            .unwrap_or_else(|| "string".to_string()),
    )
}

/// Splits an output variable name into its nested property path.
pub(super) fn output_names(name: &str) -> Vec<String> {
    let mut names: Vec<String> = name.split("_of_").map(str::to_string).collect();
    names.reverse();
    names
}

/// The `{"index": n}` extra block of one variable.
pub(super) fn index_value(index: usize) -> Value {
    Value::Object(object_with([("index", serde_json::json!(index))]))
}

/// Splits `${node.variable}` into `(node, variable)`.
///
/// Matching mirrors Python's `_VAR_PATTERN.match`: surrounding whitespace is
/// ignored, both parts are word characters, and trailing text after the closing
/// brace is not inspected.
pub(super) fn parse_variable(expr: &str) -> Option<(String, String)> {
    let body = expr.trim().strip_prefix("${")?;
    let inner = &body[..body.find('}')?];
    let (node, variable) = inner.split_once('.')?;
    let node = node.trim();
    let variable = variable.trim();
    if is_word(node) && is_word(variable) {
        Some((node.to_string(), variable.to_string()))
    } else {
        None
    }
}

/// Whether `text` is non-empty and made of word characters only.
fn is_word(text: &str) -> bool {
    !text.is_empty()
        && text
            .chars()
            .all(|character| character.is_alphanumeric() || character == '_')
}

/// Builds Python's `convert_ref_variable` value for `${node.variable}`.
///
/// # Errors
///
/// Returns [`Error::Validation`] naming the node and the expression when `expr`
/// is not a reference.
pub(super) fn convert_reference(node_id: &str, expr: &str) -> Result<Value> {
    let Some((node, variable)) = parse_variable(expr) else {
        return Err(Error::validation(format!(
            "node {node_id}: invalid reference variable expression: {expr}"
        )));
    };

    let parts = output_names(&variable);
    let mut content = Vec::with_capacity(parts.len() + 1);
    content.push(Value::String(node));
    for part in parts {
        content.push(Value::String(part));
    }

    Ok(Value::Object(object_with([
        ("type", Value::String(REF.to_string())),
        ("content", Value::Array(content)),
    ])))
}

/// Attaches the `{"index": n}` extra block to a reference value.
fn with_index(mut value: Value, index: usize) -> Value {
    if let Value::Object(map) = &mut value {
        map.insert("extra".to_string(), index_value(index));
    }
    value
}

/// Adds one output variable to the `object` schema `field`.
fn add_output_property(
    field: &mut Map<String, Value>,
    names: &[String],
    item: &DlVariable,
    index: usize,
) {
    let Some((key, rest)) = names.split_first() else {
        return;
    };
    let leaf = rest.is_empty();

    let Some(properties) = field.get_mut("properties").and_then(Value::as_object_mut) else {
        return;
    };
    if !properties.contains_key(key.as_str()) {
        let child = if leaf {
            output_leaf(item, index)
        } else {
            Value::Object(object_with([
                ("type", Value::String("object".to_string())),
                ("properties", Value::Object(Map::new())),
                ("required", Value::Array(Vec::new())),
            ]))
        };
        properties.insert(key.clone(), child);
    }

    let Some(child) = properties
        .get_mut(key.as_str())
        .and_then(Value::as_object_mut)
    else {
        return;
    };
    add_output_property(child, rest, item, index);
}

/// Builds the leaf schema of one output variable.
fn output_leaf(item: &DlVariable, index: usize) -> Value {
    let mut leaf = object_with([("type", schema_type(item))]);
    if let Some(description) = &item.description {
        leaf.insert(
            "description".to_string(),
            Value::String(description.clone()),
        );
    }
    leaf.insert("extra".to_string(), index_value(index));

    if item.var_type.as_deref() == Some("object") {
        let properties = item
            .properties
            .clone()
            .unwrap_or_else(|| Value::Object(Map::new()));
        leaf.insert("properties".to_string(), properties);
        let required = item.required.clone().unwrap_or_default();
        leaf.insert("required".to_string(), serde_json::json!(required));
    } else if let Some(items) = &item.items {
        leaf.insert("items".to_string(), items.clone());
    }

    Value::Object(leaf)
}

/// Builds one platform edge.
fn edge(source: &str, target: &str, port: Option<&str>) -> PlatformEdge {
    PlatformEdge {
        source_node_id: source.to_string(),
        target_node_id: target.to_string(),
        source_port_id: port.map(str::to_string),
    }
}

/// The value of `target` unless it is missing or blank.
fn non_blank(target: Option<&str>) -> Option<&str> {
    target.filter(|value| !value.trim().is_empty())
}
