//! The Python-shaped DL (`SimpleIR`) document model.
//!
//! Ports the wire shape of
//! `openjiuwen.dev_tools.agent_builder.builders.workflow.dl_transformer`: the DL
//! content is a bare JSON **array of nodes**, not a document object. Every node
//! is `{id, type, description, next?, parameters}` and every `parameters` block
//! carries `inputs`, `outputs`, `configs`, and `conditions`.
//!
//! The node `type` keeps Python's capitalized wire name (`"Start"`, `"LLM"`,
//! `"IntentDetection"`, ...); [`DlNodeType`] maps each name to the platform DSL
//! node id Python's `_dsl_converter_registry` uses. The model is deliberately
//! permissive: unknown node, parameter, variable, and condition keys are kept in
//! each type's `extra` map, so a document round-trips through this port without
//! losing fields the model does not name yet.
//!
//! [`DlDocument`] serializes as the bare array (`#[serde(transparent)]`) and
//! parses text through Python's `extract_json_from_text` contract, so a
//! `json`-tagged fence, an untagged fence, and a raw array all parse.
//!
//! Structural validation ([`DlDocument::validate`]) is graph bookkeeping only: it
//! checks node presence, non-blank unique ids, and resolvable `next` links.
//! Per-node-type field rules belong to the reflector built on top of this model.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::agent_builder::utils::extract_json_from_text;

/// A complete DL document: Python's bare JSON array of nodes.
///
/// The type is transparent over its [`DlDocument::nodes`] field, so serializing
/// yields `[{...}, {...}]` and parsing expects the same array (never an object
/// with a `nodes` key).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct DlDocument {
    /// The nodes of the document, in wire order.
    pub nodes: Vec<DlNode>,
}

impl DlDocument {
    /// Parses a DL document from text.
    ///
    /// The first fenced code block is extracted with
    /// [`extract_json_from_text`] (a `json`-tagged fence and an untagged fence are
    /// both accepted, and text without a fence is used verbatim), then parsed as
    /// a JSON array.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] mentioning `DL JSON` when the extracted
    /// text is not a JSON array of DL nodes, with the serde error in the message.
    pub fn from_json_str(text: &str) -> Result<Self> {
        let body = extract_json_from_text(text);
        let value: Value = serde_json::from_str(&body)
            .map_err(|error| Error::validation(format!("failed to parse DL JSON: {error}")))?;
        Self::from_json_value(value)
    }

    /// Parses a DL document from a JSON value.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] mentioning `DL JSON` unless `value` is a
    /// JSON array whose elements are DL nodes.
    pub fn from_json_value(value: Value) -> Result<Self> {
        serde_json::from_value(value)
            .map_err(|error| Error::validation(format!("failed to parse DL JSON: {error}")))
    }

    /// Serializes the document as a JSON array.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the document cannot be serialized,
    /// with the serde error in the message.
    pub fn to_value(&self) -> Result<Value> {
        serde_json::to_value(self)
            .map_err(|error| Error::validation(format!("failed to serialize DL JSON: {error}")))
    }

    /// Serializes the document as pretty-printed JSON.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the document cannot be serialized,
    /// with the serde error in the message.
    pub fn to_json_string(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|error| Error::validation(format!("failed to serialize DL JSON: {error}")))
    }

    /// Looks up a node by id.
    #[must_use]
    pub fn node(&self, id: &str) -> Option<&DlNode> {
        self.nodes.iter().find(|node| node.id == id)
    }

    /// Collects every node id, in sorted order.
    #[must_use]
    pub fn ids(&self) -> BTreeSet<String> {
        self.nodes.iter().map(|node| node.id.clone()).collect()
    }

    /// Checks the structural invariants of the document.
    ///
    /// The document must contain at least one node, every node id must be
    /// non-blank and unique, and every present `next` link — a node's
    /// [`DlNode::next`] and each [`DlCondition::next`] inside
    /// [`DlParameters::conditions`] — must name an existing node. Node types are
    /// not checked here; callers use [`DlNodeType::from_wire`] for that.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with the message `DL document has no nodes`,
    /// `blank node id at position <n>` (zero-based position),
    /// `duplicate node id: <id>`, or `next references unknown node: <id>`.
    pub fn validate(&self) -> Result<()> {
        if self.nodes.is_empty() {
            return Err(Error::validation("DL document has no nodes"));
        }

        let mut ids = BTreeSet::new();
        for (position, node) in self.nodes.iter().enumerate() {
            if node.id.trim().is_empty() {
                return Err(Error::validation(format!(
                    "blank node id at position {position}"
                )));
            }
            if !ids.insert(node.id.as_str()) {
                return Err(Error::validation(format!("duplicate node id: {}", node.id)));
            }
        }

        let dangling = self
            .nodes
            .iter()
            .flat_map(|node| {
                std::iter::once(node.next.as_deref()).chain(
                    node.parameters
                        .conditions
                        .iter()
                        .map(|condition| condition.next.as_deref()),
                )
            })
            .flatten()
            .find(|next| !ids.contains(*next));

        if let Some(next) = dangling {
            return Err(Error::validation(format!(
                "next references unknown node: {next}"
            )));
        }

        Ok(())
    }
}

/// One node of a [`DlDocument`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct DlNode {
    /// The node id, unique inside the document and referenced by `next` links.
    pub id: String,
    /// The Python DL node type, serialized as `"type"`.
    #[serde(rename = "type")]
    pub node_type: String,
    /// A human-readable node description.
    pub description: String,
    /// The id of the node this one continues to, when the type has a single
    /// successor (branching nodes use [`DlParameters::conditions`] instead).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next: Option<String>,
    /// The node parameters: variables, configuration, and branch conditions.
    pub parameters: DlParameters,
    /// Unknown node-level keys, preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

impl DlNode {
    /// Parses the node's wire type into a [`DlNodeType`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] naming the unknown type when the node's
    /// `type` is not one of Python's DL node types.
    pub fn kind(&self) -> Result<DlNodeType> {
        DlNodeType::from_wire(&self.node_type)
    }
}

/// The parameter block of a [`DlNode`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct DlParameters {
    /// Declared input variables.
    pub inputs: Vec<DlVariable>,
    /// Declared output variables.
    pub outputs: Vec<DlVariable>,
    /// Free-form per-node configuration (for example `system_prompt`,
    /// `user_prompt`, `tool_id`, `code`, or `template`), preserved verbatim.
    pub configs: BTreeMap<String, Value>,
    /// Branch conditions, for the node types that route to several successors.
    pub conditions: Vec<DlCondition>,
    /// Unknown parameter keys, preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// One input or output variable of a [`DlNode`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct DlVariable {
    /// The variable name.
    pub name: String,
    /// The bound value or `${node.field}` reference, when present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<Value>,
    /// The DL variable type (for example `String` or `Array<Object>`),
    /// serialized as `"type"`.
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub var_type: Option<String>,
    /// A human-readable variable description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// The element schema of an array-typed variable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub items: Option<Value>,
    /// The nested property schema of an object-typed variable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<Value>,
    /// The required nested property names of an object-typed variable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required: Option<Vec<String>>,
    /// Unknown variable keys, preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// One branch condition of a [`DlParameters`] block.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct DlCondition {
    /// The branch name.
    pub branch: String,
    /// A human-readable condition description.
    pub description: String,
    /// The id of the node this branch continues to.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next: Option<String>,
    /// A single branch expression; the marker `default` names the fallback.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expression: Option<String>,
    /// A list of branch expressions, combined by [`DlCondition::operator`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expressions: Option<Vec<String>>,
    /// The logical operator combining [`DlCondition::expressions`] (for example
    /// `and` or `or`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operator: Option<String>,
    /// The intent identifier of an intent-detection branch.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub intent_id: Option<String>,
    /// Unknown condition keys, preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// A Python DL node type.
///
/// Serialization uses the capitalized wire name ([`DlNodeType::wire_name`]), and
/// [`DlNodeType::dsl_id`] is the platform DSL node id Python assigns.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub enum DlNodeType {
    /// The workflow entry point (`"1"`).
    #[default]
    Start,
    /// The workflow exit (`"2"`).
    End,
    /// A model call (`"3"`).
    Llm,
    /// A conditional split (`"4"`).
    Branch,
    /// Intent classification (`"6"`).
    IntentDetection,
    /// A user-facing question step (`"7"`).
    Questioner,
    /// A formatted output step (`"9"`).
    Output,
    /// A code-execution step (`"10"`).
    Code,
    /// A tool or plugin call (`"19"`).
    Plugin,
}

impl DlNodeType {
    /// Every node type in Python's `_dsl_converter_registry` order.
    pub const ALL: [DlNodeType; 9] = [
        DlNodeType::Start,
        DlNodeType::End,
        DlNodeType::Llm,
        DlNodeType::IntentDetection,
        DlNodeType::Questioner,
        DlNodeType::Code,
        DlNodeType::Plugin,
        DlNodeType::Output,
        DlNodeType::Branch,
    ];

    /// The capitalized Python wire name of the type.
    #[must_use]
    pub const fn wire_name(self) -> &'static str {
        match self {
            Self::Start => "Start",
            Self::End => "End",
            Self::Llm => "LLM",
            Self::Branch => "Branch",
            Self::IntentDetection => "IntentDetection",
            Self::Questioner => "Questioner",
            Self::Output => "Output",
            Self::Code => "Code",
            Self::Plugin => "Plugin",
        }
    }

    /// The platform DSL node id of the type, as Python's `NodeType` assigns it.
    #[must_use]
    pub const fn dsl_id(self) -> &'static str {
        match self {
            Self::Start => "1",
            Self::End => "2",
            Self::Llm => "3",
            Self::Branch => "4",
            Self::IntentDetection => "6",
            Self::Questioner => "7",
            Self::Output => "9",
            Self::Code => "10",
            Self::Plugin => "19",
        }
    }

    /// Parses a node type from its wire name.
    ///
    /// Matching trims surrounding whitespace and ignores ASCII case, but performs
    /// no aliasing: `"Questioner"` and `"LLM"` stay distinct types.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with `unknown DL node type: <value>` when no
    /// wire name matches.
    pub fn from_wire(value: &str) -> Result<Self> {
        let trimmed = value.trim();
        for kind in Self::ALL {
            if trimmed.eq_ignore_ascii_case(kind.wire_name()) {
                return Ok(kind);
            }
        }
        Err(Error::validation(format!("unknown DL node type: {value}")))
    }
}

impl fmt::Display for DlNodeType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.wire_name())
    }
}

impl TryFrom<String> for DlNodeType {
    type Error = Error;

    fn try_from(value: String) -> Result<Self> {
        Self::from_wire(&value)
    }
}

impl From<DlNodeType> for String {
    fn from(kind: DlNodeType) -> Self {
        kind.wire_name().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;

    /// A five-node DL array mirroring the Python `dl_transformer` fixtures.
    const PYTHON_FIXTURE: &str = r#"[
      {
        "id": "start",
        "type": "Start",
        "description": "entry",
        "next": "llm",
        "parameters": {
          "inputs": [],
          "outputs": [
            {"name": "query", "type": "String", "description": "用户输入"},
            {"name": "用户输入", "type": "String", "description": "用户输入"}
          ],
          "configs": {}
        }
      },
      {
        "id": "llm",
        "type": "LLM",
        "description": "generate",
        "next": "branch",
        "parameters": {
          "inputs": [
            {
              "name": "query",
              "value": "${start.query}",
              "type": "String",
              "description": "user query"
            }
          ],
          "outputs": [
            {"name": "result", "type": "String", "description": "model output"}
          ],
          "configs": {
            "system_prompt": "You are helpful.",
            "user_prompt": "${start.query}"
          }
        }
      },
      {
        "id": "branch",
        "type": "Branch",
        "description": "route",
        "parameters": {
          "inputs": [],
          "outputs": [],
          "configs": {},
          "conditions": [
            {
              "branch": "positive",
              "description": "positive case",
              "next": "plugin",
              "expression": "${llm.result} contain 'good'"
            },
            {
              "branch": "combined",
              "description": "combined case",
              "next": "plugin",
              "expressions": [
                "${llm.result} contain 'a'",
                "${llm.result} contain 'b'"
              ],
              "operator": "or"
            },
            {
              "branch": "default",
              "description": "fallback",
              "next": "end",
              "expression": "default",
              "intent_id": "intent-1"
            }
          ]
        }
      },
      {
        "id": "plugin",
        "type": "Plugin",
        "description": "call tool",
        "next": "end",
        "parameters": {
          "inputs": [],
          "outputs": [],
          "configs": {"tool_id": "tool-1"}
        }
      },
      {
        "id": "end",
        "type": "End",
        "description": "exit",
        "parameters": {
          "inputs": [],
          "outputs": [],
          "configs": {"template": "${plugin.result}"}
        }
      }
    ]"#;

    /// A two-node document that names every modelled key, so a round trip is
    /// byte-for-byte (value-for-value) stable.
    fn full_fixture() -> Value {
        json!([
            {
                "id": "start",
                "type": "Start",
                "description": "entry",
                "next": "llm",
                "parameters": {
                    "inputs": [],
                    "outputs": [
                        {
                            "name": "query",
                            "value": 0,
                            "type": "String",
                            "description": "用户输入",
                            "items": false,
                            "properties": "",
                            "required": ["query"]
                        }
                    ],
                    "configs": {},
                    "conditions": []
                }
            },
            {
                "id": "llm",
                "type": "LLM",
                "description": "call",
                "parameters": {
                    "inputs": [],
                    "outputs": [],
                    "configs": {
                        "system_prompt": "hi",
                        "user_prompt": "${start.query}"
                    },
                    "conditions": [
                        {
                            "branch": "b",
                            "description": "d",
                            "next": "start",
                            "expression": "x",
                            "expressions": [],
                            "operator": "or",
                            "intent_id": "i"
                        }
                    ]
                }
            }
        ])
    }

    #[test]
    fn parses_python_shaped_array() {
        let doc = DlDocument::from_json_str(PYTHON_FIXTURE).expect("fixture parses");
        assert_eq!(doc.nodes.len(), 5);
        assert_eq!(doc.ids().len(), 5);
        assert!(doc.ids().contains("llm"));
        assert_eq!(
            doc.node("plugin").map(|node| node.id.as_str()),
            Some("plugin")
        );
        assert!(doc.node("missing").is_none());

        let start = &doc.nodes[0];
        assert_eq!(start.kind().expect("start kind"), DlNodeType::Start);
        assert_eq!(start.next.as_deref(), Some("llm"));
        assert_eq!(start.parameters.outputs.len(), 2);
        assert_eq!(start.parameters.outputs[0].name, "query");
        assert_eq!(
            start.parameters.outputs[0].var_type.as_deref(),
            Some("String")
        );
        assert_eq!(
            start.parameters.outputs[0].description.as_deref(),
            Some("用户输入")
        );
        assert_eq!(start.parameters.outputs[1].name, "用户输入");

        let llm = &doc.nodes[1];
        assert_eq!(llm.kind().expect("llm kind"), DlNodeType::Llm);
        assert_eq!(
            llm.parameters.configs.get("system_prompt"),
            Some(&json!("You are helpful."))
        );
        assert_eq!(
            llm.parameters.inputs[0].value,
            Some(json!("${start.query}"))
        );

        let branch = &doc.nodes[2];
        assert_eq!(branch.kind().expect("branch kind"), DlNodeType::Branch);
        assert!(branch.next.is_none());
        assert_eq!(branch.parameters.conditions.len(), 3);
        let positive = &branch.parameters.conditions[0];
        assert_eq!(positive.branch, "positive");
        assert_eq!(positive.next.as_deref(), Some("plugin"));
        assert_eq!(
            positive.expression.as_deref(),
            Some("${llm.result} contain 'good'")
        );
        let combined = &branch.parameters.conditions[1];
        assert_eq!(
            combined.expressions,
            Some(vec![
                "${llm.result} contain 'a'".to_string(),
                "${llm.result} contain 'b'".to_string()
            ])
        );
        assert_eq!(combined.operator.as_deref(), Some("or"));
        let fallback = &branch.parameters.conditions[2];
        assert_eq!(fallback.expression.as_deref(), Some("default"));
        assert_eq!(fallback.intent_id.as_deref(), Some("intent-1"));

        let plugin = &doc.nodes[3];
        assert_eq!(plugin.kind().expect("plugin kind"), DlNodeType::Plugin);
        assert_eq!(
            plugin.parameters.configs.get("tool_id"),
            Some(&json!("tool-1"))
        );
        assert_eq!(doc.nodes[4].kind().expect("end kind"), DlNodeType::End);
    }

    #[test]
    fn parses_fenced_and_unfenced_documents() {
        let raw = PYTHON_FIXTURE.trim();
        let expected = DlDocument::from_json_str(raw).expect("raw array parses");
        for text in [
            format!("```json\n{raw}\n```"),
            format!("```\n{raw}\n```"),
            format!("Here is the DL:\n\n```json\n{raw}\n```\n\nDone."),
        ] {
            let parsed = DlDocument::from_json_str(&text).expect("fenced document parses");
            assert_eq!(parsed, expected);
        }
    }

    #[test]
    fn rejects_non_array_input() {
        for value in [json!({}), json!([1]), json!("text"), json!(42)] {
            let error = DlDocument::from_json_value(value).expect_err("non-array is rejected");
            assert_eq!(error.kind(), ErrorKind::Validation);
            assert!(
                error.to_string().contains("failed to parse DL JSON"),
                "{error}"
            );
        }

        for text in ["{}", "not json", ""] {
            let error = DlDocument::from_json_str(text).expect_err("non-array text is rejected");
            assert_eq!(error.kind(), ErrorKind::Validation);
            assert!(
                error.to_string().contains("failed to parse DL JSON"),
                "{error}"
            );
        }
    }

    #[test]
    fn empty_array_parses_but_fails_validation() {
        let doc = DlDocument::from_json_str("[]").expect("empty array parses");
        assert_eq!(doc.nodes, Vec::new());
        assert!(doc.ids().is_empty());

        let error = doc.validate().expect_err("empty document is invalid");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert_eq!(error.to_string(), "validation: DL document has no nodes");
    }

    #[test]
    fn round_trips_modelled_keys() {
        let value = full_fixture();
        let doc = DlDocument::from_json_value(value.clone()).expect("fixture parses");
        assert_eq!(doc.to_value().expect("serializes"), value);

        let text = doc.to_json_string().expect("pretty serializes");
        let reparsed: Value = serde_json::from_str(&text).expect("pretty output is JSON");
        assert_eq!(reparsed, value);
        assert_eq!(
            DlDocument::from_json_value(reparsed).expect("reparses"),
            doc
        );
    }

    #[test]
    fn unknown_keys_survive_round_trip() {
        let mut value = full_fixture();
        value[0]["custom_node_key"] = json!({"nested": true});
        value[0]["parameters"]["custom_param"] = json!([1, 2]);
        value[0]["parameters"]["configs"]["custom_config"] = json!("kept");
        value[0]["parameters"]["outputs"][0]["custom_var"] = json!(7);
        value[1]["parameters"]["conditions"][0]["custom_condition"] = json!(true);

        let doc = DlDocument::from_json_value(value.clone()).expect("fixture parses");
        assert_eq!(
            doc.nodes[0].extra.get("custom_node_key"),
            Some(&json!({"nested": true}))
        );
        assert_eq!(
            doc.nodes[0].parameters.extra.get("custom_param"),
            Some(&json!([1, 2]))
        );
        assert_eq!(
            doc.nodes[1].parameters.conditions[0]
                .extra
                .get("custom_condition"),
            Some(&json!(true))
        );
        assert_eq!(doc.to_value().expect("serializes"), value);
    }

    #[test]
    fn absent_optional_keys_stay_absent() {
        let text = r#"[
          {
            "id": "start",
            "type": "Start",
            "description": "entry",
            "parameters": {
              "inputs": [{"name": "q"}],
              "outputs": [],
              "configs": {},
              "conditions": []
            }
          },
          {
            "id": "branch",
            "type": "Branch",
            "description": "route",
            "parameters": {
              "inputs": [],
              "outputs": [],
              "configs": {},
              "conditions": [{"branch": "b", "description": "d", "next": "start"}]
            }
          }
        ]"#;
        let doc = DlDocument::from_json_str(text).expect("document parses");
        let value = doc.to_value().expect("serializes");

        assert!(value[0].get("next").is_none());
        let variable = &value[0]["parameters"]["inputs"][0];
        assert_eq!(variable["name"], json!("q"));
        for key in [
            "value",
            "type",
            "description",
            "items",
            "properties",
            "required",
        ] {
            assert!(
                variable.get(key).is_none(),
                "`{key}` should stay absent for a variable"
            );
        }

        let condition = &value[1]["parameters"]["conditions"][0];
        for key in ["expression", "expressions", "operator", "intent_id"] {
            assert!(
                condition.get(key).is_none(),
                "`{key}` should stay absent for a condition"
            );
        }
    }

    #[test]
    fn present_zero_empty_and_false_values_are_kept() {
        let value = json!([
            {
                "id": "a",
                "type": "LLM",
                "description": "",
                "parameters": {
                    "inputs": [
                        {
                            "name": "q",
                            "value": false,
                            "type": "",
                            "description": "",
                            "items": [],
                            "properties": {},
                            "required": []
                        }
                    ],
                    "outputs": [],
                    "configs": {},
                    "conditions": []
                }
            }
        ]);
        let doc = DlDocument::from_json_value(value.clone()).expect("document parses");
        assert_eq!(doc.to_value().expect("serializes"), value);
    }

    #[test]
    fn validate_accepts_unique_resolvable_graph() {
        DlDocument::from_json_str(PYTHON_FIXTURE)
            .expect("fixture parses")
            .validate()
            .expect("graph is valid");
    }

    #[test]
    fn validate_rejects_duplicate_ids() {
        let mut doc = DlDocument::from_json_str(PYTHON_FIXTURE).expect("fixture parses");
        doc.nodes[1].id = "start".to_string();
        let error = doc.validate().expect_err("duplicate id is invalid");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(
            error.to_string().contains("duplicate node id: start"),
            "{error}"
        );
    }

    #[test]
    fn validate_rejects_blank_ids() {
        let mut doc = DlDocument::from_json_str(PYTHON_FIXTURE).expect("fixture parses");
        doc.nodes[0].id = "   ".to_string();
        let error = doc.validate().expect_err("blank id is invalid");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(
            error.to_string().contains("blank node id at position 0"),
            "{error}"
        );
    }

    #[test]
    fn validate_rejects_dangling_next() {
        let mut doc = DlDocument::from_json_str(PYTHON_FIXTURE).expect("fixture parses");
        doc.nodes[0].next = Some("ghost".to_string());
        let error = doc.validate().expect_err("dangling next is invalid");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(
            error
                .to_string()
                .contains("next references unknown node: ghost"),
            "{error}"
        );
    }

    #[test]
    fn validate_rejects_dangling_condition_next() {
        let mut doc = DlDocument::from_json_str(PYTHON_FIXTURE).expect("fixture parses");
        doc.nodes[2].parameters.conditions[0].next = Some("ghost-condition".to_string());
        let error = doc
            .validate()
            .expect_err("dangling condition next is invalid");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(
            error
                .to_string()
                .contains("next references unknown node: ghost-condition"),
            "{error}"
        );
    }

    #[test]
    fn node_type_registry_matches_python() {
        let expected = [
            (DlNodeType::Start, "Start", "1"),
            (DlNodeType::End, "End", "2"),
            (DlNodeType::Llm, "LLM", "3"),
            (DlNodeType::Branch, "Branch", "4"),
            (DlNodeType::IntentDetection, "IntentDetection", "6"),
            (DlNodeType::Questioner, "Questioner", "7"),
            (DlNodeType::Output, "Output", "9"),
            (DlNodeType::Code, "Code", "10"),
            (DlNodeType::Plugin, "Plugin", "19"),
        ];
        assert_eq!(DlNodeType::ALL.len(), expected.len());

        for (kind, wire, dsl_id) in expected {
            assert_eq!(kind.wire_name(), wire);
            assert_eq!(kind.dsl_id(), dsl_id);
            assert_eq!(kind.to_string(), wire);
            assert_eq!(DlNodeType::from_wire(wire).expect("wire name parses"), kind);
        }

        assert_eq!(
            DlNodeType::ALL,
            [
                DlNodeType::Start,
                DlNodeType::End,
                DlNodeType::Llm,
                DlNodeType::IntentDetection,
                DlNodeType::Questioner,
                DlNodeType::Code,
                DlNodeType::Plugin,
                DlNodeType::Output,
                DlNodeType::Branch,
            ]
        );
    }

    #[test]
    fn node_type_from_wire_is_case_insensitive() {
        let cases = [
            ("llm", DlNodeType::Llm),
            ("Branch", DlNodeType::Branch),
            (" intentdetection ", DlNodeType::IntentDetection),
            ("pLuGiN", DlNodeType::Plugin),
            ("START", DlNodeType::Start),
            ("code", DlNodeType::Code),
        ];
        for (text, kind) in cases {
            assert_eq!(
                DlNodeType::from_wire(text).expect("case-insensitive parse"),
                kind,
                "{text}"
            );
        }
    }

    #[test]
    fn node_type_rejects_unknown_names() {
        for name in ["Starter", "", "Node", "LLMNode"] {
            let error = DlNodeType::from_wire(name).expect_err("unknown type is rejected");
            assert_eq!(error.kind(), ErrorKind::Validation);
            assert!(
                error.to_string().contains("unknown DL node type"),
                "{error}"
            );
            assert!(error.to_string().contains(name), "{error}");
        }
    }

    #[test]
    fn node_type_serde_uses_wire_name() {
        assert_eq!(
            serde_json::to_value(DlNodeType::Llm).expect("serializes"),
            json!("LLM")
        );
        assert_eq!(
            serde_json::from_value::<DlNodeType>(json!("Plugin")).expect("parses"),
            DlNodeType::Plugin
        );
        assert_eq!(
            serde_json::from_value::<DlNodeType>(json!("intentdetection")).expect("parses"),
            DlNodeType::IntentDetection
        );
        let error =
            serde_json::from_value::<DlNodeType>(json!("bogus")).expect_err("unknown wire name");
        assert!(
            error.to_string().contains("unknown DL node type"),
            "{error}"
        );
    }

    #[test]
    fn node_kind_delegates_to_from_wire() {
        let mut node = DlNode {
            id: "n".to_string(),
            node_type: "questioner".to_string(),
            ..DlNode::default()
        };
        assert_eq!(node.kind().expect("kind"), DlNodeType::Questioner);

        node.node_type = "Mystery".to_string();
        let error = node.kind().expect_err("unknown node type");
        assert!(error.to_string().contains("unknown DL node type: Mystery"));
    }
}
