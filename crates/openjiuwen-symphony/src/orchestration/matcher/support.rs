//! The data layer of the matcher response protocol: diagnostics and supporting fields.
//!
//! Ports the payload types of Python
//! `openjiuwen.symphony.orchestration.graph.matcher.protocol`. Nothing here decides a
//! verdict: [`GraphDiagnostic`] carries one machine-readable record about the pipeline,
//! [`PortMappingPair`] one requested output→input wiring, and [`SupportingFields`] the
//! wiring and IO names a `can_feed` verdict rests on, read with Python's tolerant
//! `_validation_field_names` rules.
//!
//! [`crate::orchestration::matcher::validate`] produces and consumes these types; the
//! reply pipeline itself lives there.

use std::collections::BTreeSet;

use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{json, Map, Value};

use crate::orchestration::matcher::protocol::DirectionEvidence;

/// One machine-readable record about the protocol pipeline.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct GraphDiagnostic {
    /// The stable diagnostic code ([`DIAGNOSTIC_UNKNOWN_CANDIDATE_ID`], …).
    ///
    /// [`DIAGNOSTIC_UNKNOWN_CANDIDATE_ID`]: crate::orchestration::matcher::DIAGNOSTIC_UNKNOWN_CANDIDATE_ID
    pub code: String,
    /// The human-readable explanation.
    pub message: String,
    /// The machine-readable context; carries `candidate_id` when one is known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<Value>,
}

impl GraphDiagnostic {
    /// Builds a diagnostic without context.
    #[must_use]
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            detail: None,
        }
    }

    /// Attaches the machine-readable context.
    #[must_use]
    pub fn with_detail(mut self, detail: Value) -> Self {
        self.detail = Some(detail);
        self
    }

    /// The candidate id this diagnostic is attributable to, when one is known.
    #[must_use]
    pub fn candidate_id(&self) -> Option<&str> {
        self.detail
            .as_ref()
            .and_then(|detail| detail.get("candidate_id"))
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
    }
}

/// One requested output→input wiring, as the model reported it.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct PortMappingPair {
    /// The producing output name.
    pub source_output: String,
    /// The consuming input name.
    pub target_input: String,
}

/// The supporting fields a `can_feed` verdict rests on.
///
/// Deserialization is tolerant and mirrors Python's `_validation_field_names`: a field may
/// be a plain string (`"text"`, `"text: string"`, `"text(string)"`) or an object with a
/// `name`. Every value is trimmed, and empty collections are skipped when serializing, so
/// an empty instance renders as `{}`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct SupportingFields {
    /// The requested output→input wiring, deduplicated in report order.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub port_mappings: Vec<PortMappingPair>,
    /// The producing output names the verdict relied on.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub source_outputs: Vec<String>,
    /// The consuming input names the verdict relied on.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub target_inputs: Vec<String>,
}

impl SupportingFields {
    /// Reads the fields from a JSON object, normalizing names and port pairs.
    ///
    /// A non-object value yields empty fields.
    #[must_use]
    pub fn from_value(value: &Value) -> Self {
        let Some(object) = value.as_object() else {
            return Self::default();
        };
        Self {
            port_mappings: object
                .get("port_mappings")
                .map(port_pairs)
                .unwrap_or_default(),
            source_outputs: object
                .get("source_outputs")
                .map(field_names)
                .unwrap_or_default(),
            target_inputs: object
                .get("target_inputs")
                .map(field_names)
                .unwrap_or_default(),
        }
    }

    /// Derives the fields from one direction's evidence.
    ///
    /// Every port pair becomes a wiring entry; a direction with no wiring falls back to the
    /// evidence output/input names.
    #[must_use]
    pub fn from_evidence(evidence: &DirectionEvidence) -> Self {
        let mut pairs: Vec<PortMappingPair> = Vec::new();
        for (output, input) in evidence.port_pairs() {
            let pair = PortMappingPair {
                source_output: output.to_string(),
                target_input: input.to_string(),
            };
            if !pairs.contains(&pair) {
                pairs.push(pair);
            }
        }
        if !pairs.is_empty() {
            return Self {
                source_outputs: sorted_names(pairs.iter().map(|pair| pair.source_output.as_str())),
                target_inputs: sorted_names(pairs.iter().map(|pair| pair.target_input.as_str())),
                port_mappings: pairs,
            };
        }
        Self {
            port_mappings: Vec::new(),
            source_outputs: sorted_names(evidence.output_names()),
            target_inputs: sorted_names(evidence.input_names()),
        }
    }

    /// The requested wiring as a pair set.
    #[must_use]
    pub fn pairs(&self) -> BTreeSet<(String, String)> {
        self.port_mappings
            .iter()
            .map(|pair| (pair.source_output.clone(), pair.target_input.clone()))
            .collect()
    }

    /// Reports whether nothing is recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.port_mappings.is_empty()
            && self.source_outputs.is_empty()
            && self.target_inputs.is_empty()
    }

    /// Completes the output/input names from the requested wiring when they are absent.
    #[must_use]
    pub fn completed_from_pairs(mut self, pairs: &BTreeSet<(String, String)>) -> Self {
        if self.source_outputs.is_empty() {
            self.source_outputs = sorted_names(pairs.iter().map(|(output, _)| output.as_str()));
        }
        if self.target_inputs.is_empty() {
            self.target_inputs = sorted_names(pairs.iter().map(|(_, input)| input.as_str()));
        }
        self
    }

    /// Renders the fields as a pruned JSON object.
    #[must_use]
    pub fn to_value(&self) -> Value {
        let mut object = Map::new();
        if !self.port_mappings.is_empty() {
            let pairs: Vec<Value> = self
                .port_mappings
                .iter()
                .map(|pair| json!({"source_output": pair.source_output, "target_input": pair.target_input}))
                .collect();
            object.insert("port_mappings".to_string(), Value::Array(pairs));
        }
        if !self.source_outputs.is_empty() {
            object.insert(
                "source_outputs".to_string(),
                Value::from(self.source_outputs.clone()),
            );
        }
        if !self.target_inputs.is_empty() {
            object.insert(
                "target_inputs".to_string(),
                Value::from(self.target_inputs.clone()),
            );
        }
        Value::Object(object)
    }
}

impl<'de> Deserialize<'de> for SupportingFields {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        Ok(Self::from_value(&value))
    }
}

/// Renders a scalar JSON value as text; containers and null render as empty.
pub(super) fn scalar_text(value: Option<&Value>) -> String {
    match value {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Number(number)) => number.to_string(),
        Some(Value::Bool(flag)) => flag.to_string(),
        _ => String::new(),
    }
}

/// Reads the output→input wiring of a supporting-fields object.
fn port_pairs(value: &Value) -> Vec<PortMappingPair> {
    let Some(items) = value.as_array() else {
        return Vec::new();
    };
    let mut pairs: Vec<PortMappingPair> = Vec::new();
    for item in items {
        let Some(object) = item.as_object() else {
            continue;
        };
        let source_output = scalar_text(object.get("source_output")).trim().to_string();
        let target_input = scalar_text(object.get("target_input")).trim().to_string();
        if source_output.is_empty() || target_input.is_empty() {
            continue;
        }
        let pair = PortMappingPair {
            source_output,
            target_input,
        };
        if !pairs.contains(&pair) {
            pairs.push(pair);
        }
    }
    pairs
}

/// Reads IO names: plain strings (`"text"`, `"text: string"`) or objects with a `name`.
fn field_names(value: &Value) -> Vec<String> {
    let Some(items) = value.as_array() else {
        return Vec::new();
    };
    let mut names: Vec<String> = Vec::new();
    for item in items {
        let name = match item {
            Value::String(text) => field_name_from_string(text),
            Value::Object(object) => scalar_text(object.get("name")).trim().to_string(),
            _ => String::new(),
        };
        if !name.is_empty() && !names.contains(&name) {
            names.push(name);
        }
    }
    names
}

/// Reduces `"text: string"` or `"text(string)"` to `"text"`.
fn field_name_from_string(value: &str) -> String {
    let text = value.trim();
    if text.is_empty() {
        return String::new();
    }
    let head = text.split(':').next().unwrap_or(text).trim();
    head.split('(').next().unwrap_or(head).trim().to_string()
}

/// Collects a name stream into a sorted, deduplicated list.
fn sorted_names<'a>(names: impl IntoIterator<Item = &'a str>) -> Vec<String> {
    let set: BTreeSet<String> = names.into_iter().map(str::to_string).collect();
    set.into_iter().collect()
}
