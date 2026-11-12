//! Capability cards: identity, typed inputs/outputs, derived schemas, and hashing.
//!
//! Ports Python `openjiuwen.symphony.models.capability`. A [`CapabilityDescriptor`] is
//! the provider-owned metadata for one discoverable capability: its identity
//! (`capability_type` + `capability_id`), display fields, typed inputs and outputs,
//! classification/tags, and free-form metadata. [`CapabilityIo`] describes one input
//! or output.
//!
//! Everything derived from a descriptor ([`CapabilityDescriptor::input_schema`],
//! [`CapabilityDescriptor::output_schema`],
//! [`CapabilityDescriptor::compute_content_hash`], and
//! [`CapabilityDescriptor::fingerprint`]) is deterministic: no clock, no hash-map
//! iteration order, and no allocation-dependent ordering leaks into the result.

use std::collections::BTreeMap;
use std::fmt;
use std::str::FromStr;

use openjiuwen_common::card::Card;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::error::{validation, Error, Result};
use crate::model::datatype::normalize_type;
use crate::model::fingerprint::{CapabilityFingerprint, SemanticProfile, FINGERPRINT_VERSION};

/// The canonical IO type used when a card omits `type`.
pub const DEFAULT_IO_TYPE: &str = "any";

/// The identity of a capability: its type plus its provider-assigned id.
///
/// The pair is the registry key and the graph node id. [`CapabilityKey::qualified`]
/// renders it as `"{capability_type}:{capability_id}"`.
#[derive(
    Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "snake_case", default)]
pub struct CapabilityKey {
    /// The capability kind, for example `tool`, `workflow`, or `skill`.
    pub capability_type: String,
    /// The provider-assigned identifier, unique inside `capability_type`.
    pub capability_id: String,
}

impl CapabilityKey {
    /// Builds a key from a capability type and a capability id.
    #[must_use]
    pub fn new(capability_type: impl Into<String>, capability_id: impl Into<String>) -> Self {
        Self {
            capability_type: capability_type.into(),
            capability_id: capability_id.into(),
        }
    }

    /// Renders the key as `"{capability_type}:{capability_id}"`.
    #[must_use]
    pub fn qualified(&self) -> String {
        format!("{}:{}", self.capability_type, self.capability_id)
    }
}

impl fmt::Display for CapabilityKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}:{}", self.capability_type, self.capability_id)
    }
}

impl FromStr for CapabilityKey {
    type Err = Error;

    /// Parses `"{capability_type}:{capability_id}"`, splitting on the *first* colon so
    /// ids may themselves contain colons.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the separator is missing or either part is
    /// blank.
    fn from_str(raw: &str) -> Result<Self> {
        let trimmed = raw.trim();
        let Some((capability_type, capability_id)) = trimmed.split_once(':') else {
            return Err(validation(
                "capability key",
                format!("expected `<capability_type>:<capability_id>`, got `{trimmed}`"),
            ));
        };
        let capability_type = capability_type.trim();
        let capability_id = capability_id.trim();
        if capability_type.is_empty() {
            return Err(validation("capability_type", "must not be blank"));
        }
        if capability_id.is_empty() {
            return Err(validation("capability_id", "must not be blank"));
        }
        Ok(Self::new(capability_type, capability_id))
    }
}

/// One semantic input or output exposed by a capability.
///
/// The JSON key for the type is `"type"` (not `"io_type"`) to match the Python wire
/// format; use [`CapabilityIo::io_type`] to read it in Rust.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct CapabilityIo {
    /// The IO name, used for data-flow matching against other capabilities.
    pub name: String,
    /// The data type; normalized by the data-type vocabulary when matched.
    #[serde(rename = "type")]
    pub io_type: String,
    /// Human-readable description of the IO.
    pub description: String,
    /// Whether a plan must supply the value; `None` means "unspecified".
    pub required: Option<bool>,
    /// Optional default value, shown in the derived JSON Schema.
    pub default: Option<Value>,
    /// Free-form metadata attached to the IO.
    pub metadata: BTreeMap<String, Value>,
}

impl Default for CapabilityIo {
    fn default() -> Self {
        Self {
            name: String::new(),
            io_type: DEFAULT_IO_TYPE.to_string(),
            description: String::new(),
            required: None,
            default: None,
            metadata: BTreeMap::new(),
        }
    }
}

impl CapabilityIo {
    /// Builds an IO with `name` and `io_type` (the other fields defaulted).
    #[must_use]
    pub fn new(name: impl Into<String>, io_type: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            io_type: io_type.into(),
            ..Self::default()
        }
    }

    /// The data type of this IO (the JSON `"type"` member).
    #[must_use]
    pub fn io_type(&self) -> &str {
        &self.io_type
    }

    /// Sets the description.
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Sets whether the value is required.
    #[must_use]
    pub fn with_required(mut self, required: bool) -> Self {
        self.required = Some(required);
        self
    }

    /// Sets the default value.
    #[must_use]
    pub fn with_default(mut self, value: Value) -> Self {
        self.default = Some(value);
        self
    }

    /// Inserts one metadata entry.
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

/// Provider-owned metadata for one discoverable capability.
///
/// Identity fields are `capability_id`, `capability_type`, `name`, `description`,
/// `source`, `inputs`, `outputs`, `classification`, and `tags`; those — and only
/// those — feed [`CapabilityDescriptor::compute_content_hash`]. Operational fields
/// (`available`, `metadata`, `content_hash`, `semantic_content`) are deliberately
/// excluded so that availability flips and bookkeeping do not look like content
/// changes.
///
/// Unknown JSON fields are ignored on deserialization, and every field has a default.
/// `semantic_content` is accepted on input but excluded from output (Python marks it
/// `exclude=True` because it is a private retrieval aid, not public metadata).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct CapabilityDescriptor {
    /// Provider-assigned identifier, unique inside `capability_type`.
    pub capability_id: String,
    /// The capability kind, for example `tool`, `workflow`, or `skill`.
    pub capability_type: String,
    /// Display name.
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// Where the capability came from (provider name or artifact path).
    pub source: String,
    /// Whether the capability can currently be planned and executed.
    pub available: bool,
    /// Typed inputs consumed by the capability.
    pub inputs: Vec<CapabilityIo>,
    /// Typed outputs produced by the capability.
    pub outputs: Vec<CapabilityIo>,
    /// Coarse classification label used for grouping and search.
    pub classification: String,
    /// Retrieval tags.
    pub tags: Vec<String>,
    /// Cached content hash; populate with
    /// [`CapabilityDescriptor::compute_content_hash`].
    pub content_hash: String,
    /// Internal semantic definition used for retrieval; never serialized.
    #[serde(default, skip_serializing)]
    pub semantic_content: String,
    /// Free-form metadata attached to the capability.
    pub metadata: BTreeMap<String, Value>,
}

impl CapabilityDescriptor {
    /// Builds a descriptor with the required identity fields and everything else
    /// defaulted (`available` is `false` until [`Self::with_available`] is called).
    #[must_use]
    pub fn new(
        capability_type: impl Into<String>,
        capability_id: impl Into<String>,
        name: impl Into<String>,
    ) -> Self {
        Self {
            capability_id: capability_id.into(),
            capability_type: capability_type.into(),
            name: name.into(),
            ..Self::default()
        }
    }

    /// Sets the description.
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Sets the source.
    #[must_use]
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = source.into();
        self
    }

    /// Sets whether the capability is available.
    #[must_use]
    pub fn with_available(mut self, available: bool) -> Self {
        self.available = available;
        self
    }

    /// Sets the classification label.
    #[must_use]
    pub fn with_classification(mut self, classification: impl Into<String>) -> Self {
        self.classification = classification.into();
        self
    }

    /// Replaces the tags.
    #[must_use]
    pub fn with_tags(mut self, tags: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.tags = tags.into_iter().map(Into::into).collect();
        self
    }

    /// Replaces the typed inputs.
    #[must_use]
    pub fn with_inputs(mut self, inputs: Vec<CapabilityIo>) -> Self {
        self.inputs = inputs;
        self
    }

    /// Replaces the typed outputs.
    #[must_use]
    pub fn with_outputs(mut self, outputs: Vec<CapabilityIo>) -> Self {
        self.outputs = outputs;
        self
    }

    /// Sets the internal semantic content (never serialized).
    #[must_use]
    pub fn with_semantic_content(mut self, text: impl Into<String>) -> Self {
        self.semantic_content = text.into();
        self
    }

    /// Inserts one metadata entry.
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// The registry key of this capability.
    #[must_use]
    pub fn key(&self) -> CapabilityKey {
        CapabilityKey::new(self.capability_type.as_str(), self.capability_id.as_str())
    }

    /// The qualified id (`"{capability_type}:{capability_id}"`).
    #[must_use]
    pub fn qualified_id(&self) -> String {
        self.key().qualified()
    }

    /// Checks that the identity fields are non-blank.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] naming the first blank field among
    /// `capability_id`, `capability_type`, and `name`.
    pub fn validate(&self) -> Result<()> {
        if self.capability_id.trim().is_empty() {
            return Err(validation("capability_id", "must not be blank"));
        }
        if self.capability_type.trim().is_empty() {
            return Err(validation("capability_type", "must not be blank"));
        }
        if self.name.trim().is_empty() {
            return Err(validation("name", "must not be blank"));
        }
        Ok(())
    }

    /// The text indexed for retrieval: name, description, classification, tags, and
    /// the semantic content, joined by single spaces with blanks skipped.
    #[must_use]
    pub fn retrieval_text(&self) -> String {
        let mut parts: Vec<&str> = [&self.name, &self.description, &self.classification]
            .into_iter()
            .map(|part| part.trim())
            .filter(|part| !part.is_empty())
            .collect();
        parts.extend(
            self.tags
                .iter()
                .map(|tag| tag.trim())
                .filter(|tag| !tag.is_empty()),
        );
        let semantic = self.semantic_content.trim();
        if !semantic.is_empty() {
            parts.push(semantic);
        }
        parts.join(" ")
    }

    /// The JSON Schema of the inputs: an object whose properties carry the normalized
    /// JSON type and description, with `required` listing the `required: true` names.
    #[must_use]
    pub fn input_schema(&self) -> Value {
        io_schema(&self.inputs)
    }

    /// The JSON Schema of the outputs; see [`Self::input_schema`].
    #[must_use]
    pub fn output_schema(&self) -> Value {
        io_schema(&self.outputs)
    }

    /// The SHA-256 hex digest of the canonical JSON of the identity fields.
    ///
    /// The identity fields are `capability_id`, `capability_type`, `name`,
    /// `description`, `source`, `inputs`, `outputs`, `classification`, and `tags`.
    /// Object keys are sorted recursively before hashing, so the digest does not
    /// depend on JSON map order; `available`, `metadata`, `content_hash`, and
    /// `semantic_content` are excluded.
    #[must_use]
    pub fn compute_content_hash(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(canonical_json(&identity_value(self)).as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Derives a deterministic fingerprint: the descriptor plus static evidence and a
    /// [`SemanticProfile`] built from the card.
    #[must_use]
    pub fn fingerprint(&self) -> CapabilityFingerprint {
        let mut static_data = BTreeMap::new();
        static_data.insert("available".to_string(), Value::Bool(self.available));
        static_data.insert(
            "content_hash".to_string(),
            Value::String(self.compute_content_hash()),
        );
        static_data.insert("source".to_string(), Value::String(self.source.clone()));

        let mut keywords: Vec<String> = self
            .tags
            .iter()
            .flat_map(|tag| keyword_tokens(tag))
            .chain(keyword_tokens(&self.name))
            .chain(keyword_tokens(&self.classification))
            .collect();
        keywords.sort_unstable();
        keywords.dedup();

        let summary = if self.description.trim().is_empty() {
            self.name.clone()
        } else {
            self.description.clone()
        };
        let limitations = if self.available {
            Vec::new()
        } else {
            vec!["capability is unavailable".to_string()]
        };

        CapabilityFingerprint {
            descriptor: self.clone(),
            version: FINGERPRINT_VERSION.to_string(),
            static_data,
            semantic_profile: SemanticProfile {
                summary,
                capabilities: vec![self.qualified_id()],
                use_cases: self.tags.clone(),
                limitations,
                keywords,
            },
            failures: Vec::new(),
            evidence: Vec::new(),
            suggestions: Vec::new(),
        }
    }
}

impl Card for CapabilityDescriptor {
    fn id(&self) -> &str {
        &self.capability_id
    }

    fn name(&self) -> &str {
        &self.name
    }

    /// Returns `None` for a blank description, so an unset description never renders
    /// as an empty string.
    fn description(&self) -> Option<&str> {
        let trimmed = self.description.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed)
        }
    }
}

/// Builds a JSON Schema object from a list of IO declarations.
///
/// Nameless declarations cannot become properties and are skipped. Later duplicates
/// overwrite earlier ones. The `required` array is sorted and deduplicated.
fn io_schema(ios: &[CapabilityIo]) -> Value {
    let mut properties: BTreeMap<String, Value> = BTreeMap::new();
    let mut required: Vec<String> = Vec::new();
    for io in ios {
        let name = io.name.trim();
        if name.is_empty() {
            continue;
        }
        let mut schema = Map::new();
        if let Some(json_type) = json_schema_type(&io.io_type) {
            schema.insert("type".to_string(), Value::String(json_type.to_string()));
        }
        let description = io.description.trim();
        if !description.is_empty() {
            schema.insert(
                "description".to_string(),
                Value::String(description.to_string()),
            );
        }
        if let Some(default) = &io.default {
            schema.insert("default".to_string(), default.clone());
        }
        properties.insert(name.to_string(), Value::Object(schema));
        if io.required == Some(true) {
            required.push(name.to_string());
        }
    }
    required.sort_unstable();
    required.dedup();

    let mut root = Map::new();
    root.insert("type".to_string(), Value::String("object".to_string()));
    root.insert(
        "properties".to_string(),
        Value::Object(properties.into_iter().collect()),
    );
    root.insert(
        "required".to_string(),
        Value::Array(required.into_iter().map(Value::String).collect()),
    );
    Value::Object(root)
}

/// Maps a card data type to a JSON Schema `type`.
///
/// Returns `None` for `any` and unknown types, which leaves the property untyped
/// (valid JSON Schema meaning "any value").
fn json_schema_type(raw: &str) -> Option<&'static str> {
    match normalize_type(raw).as_str() {
        "string" | "text" | "file" | "image" | "audio" | "bytes" => Some("string"),
        "number" => Some("number"),
        "integer" => Some("integer"),
        "boolean" => Some("boolean"),
        "object" => Some("object"),
        "array" => Some("array"),
        "null" => Some("null"),
        _ => None,
    }
}

/// Splits `text` into lowercase alphanumeric keyword tokens.
fn keyword_tokens(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for character in text.chars() {
        if character.is_alphanumeric() {
            current.extend(character.to_lowercase());
        } else if !current.is_empty() {
            tokens.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

/// The identity subset of a descriptor, as JSON.
fn identity_value(descriptor: &CapabilityDescriptor) -> Value {
    let mut map = Map::new();
    map.insert(
        "capability_id".to_string(),
        Value::String(descriptor.capability_id.clone()),
    );
    map.insert(
        "capability_type".to_string(),
        Value::String(descriptor.capability_type.clone()),
    );
    map.insert("name".to_string(), Value::String(descriptor.name.clone()));
    map.insert(
        "description".to_string(),
        Value::String(descriptor.description.clone()),
    );
    map.insert(
        "source".to_string(),
        Value::String(descriptor.source.clone()),
    );
    map.insert(
        "inputs".to_string(),
        Value::Array(descriptor.inputs.iter().map(io_value).collect()),
    );
    map.insert(
        "outputs".to_string(),
        Value::Array(descriptor.outputs.iter().map(io_value).collect()),
    );
    map.insert(
        "classification".to_string(),
        Value::String(descriptor.classification.clone()),
    );
    map.insert(
        "tags".to_string(),
        Value::Array(
            descriptor
                .tags
                .iter()
                .map(|tag| Value::String(tag.clone()))
                .collect(),
        ),
    );
    Value::Object(map)
}

/// One IO declaration as JSON, with the `type` key used by the wire format.
fn io_value(io: &CapabilityIo) -> Value {
    let mut map = Map::new();
    map.insert("name".to_string(), Value::String(io.name.clone()));
    map.insert("type".to_string(), Value::String(io.io_type.clone()));
    map.insert(
        "description".to_string(),
        Value::String(io.description.clone()),
    );
    if let Some(required) = io.required {
        map.insert("required".to_string(), Value::Bool(required));
    }
    if let Some(default) = &io.default {
        map.insert("default".to_string(), default.clone());
    }
    if !io.metadata.is_empty() {
        let metadata: Map<String, Value> = io
            .metadata
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        map.insert("metadata".to_string(), Value::Object(metadata));
    }
    Value::Object(map)
}

/// Renders `value` with every object's keys sorted, recursively.
fn canonical_json(value: &Value) -> String {
    canonicalize(value.clone()).to_string()
}

/// Recursively rebuilds `value` with object keys in sorted order.
fn canonicalize(value: Value) -> Value {
    match value {
        Value::Object(map) => {
            let sorted: BTreeMap<String, Value> = map
                .into_iter()
                .map(|(key, item)| (key, canonicalize(item)))
                .collect();
            Value::Object(sorted.into_iter().collect())
        }
        Value::Array(items) => Value::Array(items.into_iter().map(canonicalize).collect()),
        other => other,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn sample() -> CapabilityDescriptor {
        CapabilityDescriptor::new("tool", "sum", "Sum")
            .with_description("adds numbers")
            .with_source("builtin")
            .with_available(true)
            .with_classification("math")
            .with_tags(["math", "Numbers"])
            .with_inputs(vec![
                CapabilityIo::new("a", "number")
                    .with_description("first addend")
                    .with_required(true),
                CapabilityIo::new("b", "int").with_required(false),
            ])
            .with_outputs(vec![CapabilityIo::new("total", "float")])
    }

    #[test]
    fn content_hash_is_stable_under_irrelevant_changes() {
        let base = sample();
        let hash = base.compute_content_hash();
        assert_eq!(hash.len(), 64);
        assert!(hash
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
        assert_eq!(hash, base.compute_content_hash());

        let mutated = base
            .clone()
            .with_available(false)
            .with_metadata("owner", Value::String("ops".to_string()))
            .with_semantic_content("internal notes");
        assert_eq!(hash, mutated.compute_content_hash());

        let described = base.with_description("adds two numbers");
        assert_ne!(hash, described.compute_content_hash());
    }

    #[test]
    fn content_hash_ignores_json_map_order() {
        let one = CapabilityIo::new("x", "object")
            .with_metadata("b", Value::Bool(true))
            .with_metadata("a", Value::Bool(false));
        let two = CapabilityIo::new("x", "object")
            .with_metadata("a", Value::Bool(false))
            .with_metadata("b", Value::Bool(true));
        let left = CapabilityDescriptor::new("tool", "t", "T").with_inputs(vec![one]);
        let right = CapabilityDescriptor::new("tool", "t", "T").with_inputs(vec![two]);
        assert_eq!(left.compute_content_hash(), right.compute_content_hash());
    }

    #[test]
    fn schemas_map_types_and_filter_required_names() {
        let schema = sample().input_schema();
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["properties"]["a"]["type"], "number");
        assert_eq!(schema["properties"]["a"]["description"], "first addend");
        assert_eq!(schema["properties"]["b"]["type"], "integer");
        assert_eq!(schema["required"], serde_json::json!(["a"]));
    }

    #[test]
    fn untyped_ios_omit_the_type_key() {
        let descriptor = CapabilityDescriptor::new("tool", "t", "T")
            .with_inputs(vec![CapabilityIo::new("raw", "any")]);
        assert!(descriptor.input_schema()["properties"]["raw"]
            .get("type")
            .is_none());
        assert_eq!(
            descriptor.output_schema()["properties"],
            Value::Object(Map::new())
        );
    }

    #[test]
    fn keywords_and_profile_are_deterministic() {
        let fingerprint = sample().fingerprint();
        let mut expected = fingerprint.semantic_profile.keywords.clone();
        expected.sort_unstable();
        expected.dedup();
        assert_eq!(fingerprint.semantic_profile.keywords, expected);
        assert_eq!(
            fingerprint.semantic_profile.capabilities,
            vec!["tool:sum".to_string()]
        );
        assert_eq!(
            fingerprint.semantic_profile.limitations,
            Vec::<String>::new()
        );
        assert_eq!(fingerprint.version, FINGERPRINT_VERSION);
    }

    #[test]
    fn unnamed_ios_are_skipped_in_schemas() {
        let descriptor = CapabilityDescriptor::new("tool", "t", "T")
            .with_inputs(vec![CapabilityIo::new("  ", "string")]);
        assert_eq!(
            descriptor.input_schema()["properties"],
            Value::Object(Map::new())
        );
        assert_eq!(descriptor.input_schema()["required"], serde_json::json!([]));
    }
}
