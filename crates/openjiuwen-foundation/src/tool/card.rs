//! Tool cards: the model-facing, serializable description of a tool.
//!
//! Ports `openjiuwen/core/foundation/tool/base.py` (`ToolCard`) plus the
//! `ToolExposure` and `ToolExecution` policy enums from `exposure.py` and
//! `execution.py`. A card is data only: the transport details (REST url, MCP
//! server) live in the tool implementation, never in the card.
//!
//! The struct is flat instead of embedding `openjiuwen_common::card::CardBase`
//! (the deliberate deviation documented in `src/tool/CONTEXT.md`), so consumers
//! read `card.name`/`card.id` directly while the card still implements the
//! shared [`Card`](openjiuwen_common::card::Card) trait and stays registry-safe.
//!
//! ```
//! use openjiuwen_foundation::tool::card::{ToolCard, ToolExposure};
//!
//! let card = ToolCard::new("calculator", "does math", serde_json::json!({}));
//! assert_eq!(card.exposure, ToolExposure::Direct);
//! assert_eq!(card.tool_info().function.name, "calculator");
//! ```
//!
//! # Wire shapes
//!
//! [`ToolInfo`] serializes to the nested `OpenAI` chat-completions tools shape
//! `{"type": "function", "function": {"name", "description", "parameters"}}`,
//! and that is what `ModelRequestConfig::tools` carries into every transport:
//!
//! - **`OpenAI` chat-completions** (`llm::openai`) sends the nested object
//!   verbatim.
//! - **`OpenAI` Responses** (`llm::responses`) flattens one level itself, so the
//!   body receives `{"type": "function", "name", "description", "parameters"}`.
//! - **Anthropic Messages** (`llm::anthropic`) maps it to
//!   `{"name", "description", "input_schema"}`.
//!
//! Python's `ToolInfo` (`openjiuwen/core/foundation/tool/schema.py`) is flat:
//! `{"type": "function", "name", "description", "parameters"}`. That flat shape
//! is also the tool-card/program JSON contract here — [`ToolInfo::flat`] renders
//! it, [`ToolInfo::from_flat`] parses it, and [`ToolInfo::from_wire`] accepts
//! either shape by auto-detection, all without changing the serde
//! representation (which stays nested for the crates that already depend on it).

use std::collections::BTreeMap;

use openjiuwen_common::card::Card;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::utils::id::uuid_hex;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

/// When a callable tool is included in the model-visible tool list.
///
/// Python parity: `ToolExposure` in `openjiuwen/core/foundation/tool/exposure.py`.
/// This is runtime policy metadata and never enters the model-facing
/// [`ToolInfo`] projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ToolExposure {
    /// The tool is always visible to the model.
    #[default]
    Direct,
    /// The tool is hidden until tool search discovers it.
    Deferred,
}

/// Who executes a call to a tool.
///
/// Python parity: `ToolExecution` in `openjiuwen/core/foundation/tool/execution.py`.
/// Like [`ToolExposure`], this is runtime policy metadata and never enters
/// [`ToolInfo`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ToolExecution {
    /// The agent runtime calls `Tool::invoke`.
    #[default]
    Server,
    /// The connected client runs the call; the runtime pauses with a
    /// `client_tool` interrupt and turns the client's answer into the result.
    Client,
}

/// The serializable definition of a tool.
///
/// Python parity: `ToolCard` in `openjiuwen/core/foundation/tool/base.py`.
/// Unknown JSON fields are tolerated and dropped, every field has a serde
/// default, and an entirely empty object deserializes to [`ToolCard::default`]
/// (with a freshly generated id).
///
/// The serialized key set matches Python for a default card: `id`, `name`,
/// `description`, `input_params`, `properties`, `exposure`, `execution`,
/// `parallel_safe`, `stateless`, `idempotent`. The Rust-only extensions
/// `version` and `metadata` are omitted while they are unset, so a default card
/// serializes to exactly the Python key set.
///
/// Cards are immutable after construction: the `with_*` builders consume and
/// return the card, and no accessor hands out `&mut`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
#[schemars(transform = strip_generated_id_default)]
pub struct ToolCard {
    /// Unique identifier; Python's `uuid.uuid4().hex` form by default.
    pub id: String,
    /// Name of the tool, as the model sees it.
    pub name: String,
    /// Human-readable description shown to the model.
    pub description: String,
    /// JSON Schema of the accepted arguments; `{}` when unset.
    pub input_params: Value,
    /// Free-form, program-only properties carried with the card; `{}` when
    /// unset.
    ///
    /// Python parity: `properties: Dict[str, Any] = Field(default_factory=dict)`.
    /// Registration and discovery bookkeeping lives here; the map never enters
    /// the model-facing [`ToolInfo`] projection.
    #[serde(default)]
    pub properties: BTreeMap<String, Value>,
    /// Whether the tool is exposed directly or deferred behind tool search.
    pub exposure: ToolExposure,
    /// Whether the runtime or the connected client executes the call.
    pub execution: ToolExecution,
    /// Whether the tool may run concurrently with sibling tool calls.
    pub parallel_safe: bool,
    /// Whether the tool holds no per-agent/session state.
    pub stateless: bool,
    /// Whether repeated invocations with the same inputs have no side effects.
    pub idempotent: bool,
    /// Optional version string of the tool.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    /// Free-form metadata attached to the card; omitted when empty so a default
    /// card serializes to exactly the Python key set.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, Value>,
}

impl Default for ToolCard {
    /// Python defaults: a fresh `uuid4().hex` id, empty name/description,
    /// input schema and properties, `direct`/`server` policy, parallel-safe and
    /// stateful.
    fn default() -> Self {
        Self {
            id: uuid_hex(),
            name: String::new(),
            description: String::new(),
            input_params: json!({}),
            properties: BTreeMap::new(),
            exposure: ToolExposure::Direct,
            execution: ToolExecution::Server,
            parallel_safe: true,
            stateless: false,
            idempotent: false,
            version: None,
            metadata: BTreeMap::new(),
        }
    }
}

impl ToolCard {
    /// Creates a card with `name`, `description`, `input_params` and a freshly
    /// generated id; all policy flags keep their Python defaults.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        input_params: Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            input_params,
            ..Self::default()
        }
    }

    /// Returns this card with the given id, replacing the generated one.
    #[must_use]
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }

    /// Returns this card with the given name.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Returns this card with the given description.
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Returns this card with the given input JSON Schema.
    #[must_use]
    pub fn with_input_params(mut self, input_params: Value) -> Self {
        self.input_params = input_params;
        self
    }

    /// Returns this card with `properties`, replacing the previous map.
    #[must_use]
    pub fn with_properties(mut self, properties: BTreeMap<String, Value>) -> Self {
        self.properties = properties;
        self
    }

    /// Returns this card with the given exposure policy.
    #[must_use]
    pub fn with_exposure(mut self, exposure: ToolExposure) -> Self {
        self.exposure = exposure;
        self
    }

    /// Returns this card with the given execution policy.
    #[must_use]
    pub fn with_execution(mut self, execution: ToolExecution) -> Self {
        self.execution = execution;
        self
    }

    /// Returns this card with the given parallel-safety flag.
    #[must_use]
    pub fn with_parallel_safe(mut self, parallel_safe: bool) -> Self {
        self.parallel_safe = parallel_safe;
        self
    }

    /// Returns this card with the given stateless flag.
    #[must_use]
    pub fn with_stateless(mut self, stateless: bool) -> Self {
        self.stateless = stateless;
        self
    }

    /// Returns this card with the given idempotency flag.
    #[must_use]
    pub fn with_idempotent(mut self, idempotent: bool) -> Self {
        self.idempotent = idempotent;
        self
    }

    /// Returns this card with the given version string.
    #[must_use]
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = Some(version.into());
        self
    }

    /// Returns this card with `metadata`, replacing the previous map.
    #[must_use]
    pub fn with_metadata(mut self, metadata: BTreeMap<String, Value>) -> Self {
        self.metadata = metadata;
        self
    }

    /// Returns this card with one additional metadata entry.
    #[must_use]
    pub fn with_metadata_entry(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Projects the card onto the model-facing [`ToolInfo`] shape.
    ///
    /// The parameters are normalized through
    /// [`crate::tool::schema::ensure_object_schema`], so an empty
    /// `input_params` becomes `{"type": "object"}` exactly as the `OpenAI`
    /// function-tool payload requires.
    #[must_use]
    pub fn tool_info(&self) -> ToolInfo {
        ToolInfo::new(
            &self.name,
            &self.description,
            crate::tool::schema::ensure_object_schema(self.input_params.clone()),
        )
    }

    /// Builds the `OpenAI` function-tool object for this card.
    ///
    /// The result is byte-for-byte equal to `serde_json::to_value(self.tool_info())`
    /// and is built with `json!` so it cannot fail.
    #[must_use]
    pub fn openai_tool(&self) -> Value {
        json!({
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": crate::tool::schema::ensure_object_schema(
                    self.input_params.clone(),
                ),
            }
        })
    }

    /// Builds the Python-flat `ToolInfo` object for this card.
    ///
    /// This is the tool-card/program-facing shape —
    /// `{"type": "function", "name": ..., "description": ..., "parameters": ...}`
    /// — and is exactly `self.tool_info().flat()`. It is deliberately distinct
    /// from [`ToolCard::openai_tool`], which returns the nested wire shape that
    /// `ModelRequestConfig::tools` and the provider clients expect.
    #[must_use]
    pub fn flat_tool_info(&self) -> Value {
        self.tool_info().flat()
    }
}

impl Card for ToolCard {
    fn id(&self) -> &str {
        &self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    /// Returns [`None`] when the description is empty.
    ///
    /// The card stores Python's `""` default (so the field always serializes),
    /// while the shared [`Card`] contract uses `Option<&str>`; an unset
    /// description is therefore reported as absent rather than as `Some("")`.
    fn description(&self) -> Option<&str> {
        if self.description.is_empty() {
            None
        } else {
            Some(&self.description)
        }
    }
}

/// Removes the generated `id` default from a [`ToolCard`] schema.
///
/// The struct-level serde default makes `schemars` bake a freshly generated UUID
/// into every generated schema; Python's `default_factory` emits no `default` at
/// all, so the property is stripped after generation to keep schemas
/// deterministic (same trick as `openjiuwen_common::card`).
fn strip_generated_id_default(schema: &mut schemars::Schema) {
    let Some(id) = schema
        .get_mut("properties")
        .and_then(|properties| properties.get_mut("id"))
    else {
        return;
    };
    if let Some(object) = id.as_object_mut() {
        object.remove("default");
    }
}

/// The model-facing projection of a [`ToolCard`].
///
/// Python parity: `ToolInfo` in `openjiuwen/core/foundation/tool/schema.py`.
/// Python serializes it **flat**
/// (`{"type": "function", "name": ..., "description": ..., "parameters": ...}`);
/// Rust keeps the nested `OpenAI` chat-completions tools shape
/// (`{"type": "function", "function": {"name": ..., "description": ...,
/// "parameters": ...}}`) as its serde representation, because
/// `openjiuwen-dev-tools`, `openjiuwen-agent-evolving` and the examples all
/// consume the nested shape.
///
/// Which shape each protocol sends:
///
/// | Consumer | Shape |
/// | --- | --- |
/// | `ModelRequestConfig::tools` (`Vec<Value>`) | nested, as produced by [`ToolInfo::openai`] |
/// | `OpenAI` chat-completions (`llm::openai`) | nested, sent verbatim |
/// | `OpenAI` Responses (`llm::responses`) | flat `{type, name, description, parameters}`, flattened by the Responses builder |
/// | Anthropic Messages (`llm::anthropic`) | `{name, description, input_schema}` |
/// | Python program/tool-card JSON | flat, via [`ToolInfo::flat`] / [`ToolInfo::from_flat`] |
///
/// Conversions: [`ToolInfo::flat`] renders the Python shape,
/// [`ToolInfo::from_flat`] parses it, [`ToolInfo::openai`] renders the canonical
/// nested shape, [`ToolInfo::from_openai`] parses it, and
/// [`ToolInfo::from_wire`] accepts either shape by auto-detection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct ToolInfo {
    /// The tool kind; `"function"` for every built-in tool.
    #[serde(rename = "type")]
    pub kind: String,
    /// The nested function description.
    pub function: ToolFunctionInfo,
}

impl Default for ToolInfo {
    /// A function-kind info with empty name, description, and parameters.
    fn default() -> Self {
        Self {
            kind: "function".to_owned(),
            function: ToolFunctionInfo::default(),
        }
    }
}

impl ToolInfo {
    /// Creates a function-tool projection.
    #[must_use]
    pub fn new(name: impl Into<String>, description: impl Into<String>, parameters: Value) -> Self {
        Self {
            kind: "function".to_owned(),
            function: ToolFunctionInfo::new(name, description, parameters),
        }
    }

    /// Renders Python's flat `ToolInfo` JSON.
    ///
    /// The result is `{"type": <kind>, "name": ..., "description": ...,
    /// "parameters": ...}`. `parameters` is always a JSON object: when the
    /// stored parameters are `null` or a non-object (a scalar, an array, a
    /// boolean — shapes Python's `default_factory=dict` field can never hold),
    /// `{}` is emitted instead, so the flat projection is always a valid
    /// tool-info payload. The `type` field is copied verbatim.
    #[must_use]
    pub fn flat(&self) -> Value {
        json!({
            "type": self.kind,
            "name": self.function.name,
            "description": self.function.description,
            "parameters": object_or_empty(&self.function.parameters),
        })
    }

    /// Parses Python's flat `ToolInfo` JSON.
    ///
    /// Unknown fields are tolerated and dropped, a missing `type` (or a
    /// non-string one) defaults to `"function"`, and a missing `name` /
    /// `description` defaults to `""`. A missing or `null` `parameters` becomes
    /// `{}`; any other non-object `parameters` is
    /// [`Error::Validation`] naming the field, because the flat contract
    /// declares it as a JSON Schema object.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the value is not a JSON object or
    /// carries a non-object `parameters` field.
    pub fn from_flat(value: Value) -> Result<Self> {
        let object = expect_object(&value, "a flat tool info")?.clone();

        let kind = object
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("function")
            .to_owned();
        let name = string_field(&object, "name");
        let description = string_field(&object, "description");
        let parameters = match object.get("parameters") {
            None | Some(Value::Null) => json!({}),
            Some(value @ Value::Object(_)) => value.clone(),
            Some(other) => {
                return Err(Error::validation(format!(
                    "flat tool info field `parameters` must be a JSON object, got {}",
                    json_type_name(other)
                )));
            }
        };

        Ok(Self {
            kind,
            function: ToolFunctionInfo::new(name, description, parameters),
        })
    }

    /// Renders the nested `OpenAI` chat-completions tools shape.
    ///
    /// This is the canonical serialization of [`ToolInfo`]: the result is
    /// byte-for-byte equal to `serde_json::to_value(self)`, and it is built with
    /// `json!` so it cannot fail. `ModelRequestConfig::tools` carries exactly
    /// this shape — the `OpenAI` client sends it verbatim, the Responses builder
    /// flattens it, and the Anthropic client maps it to
    /// `{name, description, input_schema}`.
    ///
    /// Unlike [`ToolInfo::flat`], a non-object `parameters` value is copied
    /// verbatim: this is a faithful serialization, not a normalization.
    #[must_use]
    pub fn openai(&self) -> Value {
        json!({
            "type": self.kind,
            "function": {
                "name": self.function.name,
                "description": self.function.description,
                "parameters": self.function.parameters,
            }
        })
    }

    /// Parses the nested `OpenAI` chat-completions tools shape.
    ///
    /// Deserialize-then-normalize: unknown fields are tolerated (the serde
    /// container default applies), a missing `type` defaults to `"function"`,
    /// and an empty `type` is rewritten to `"function"`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the value is not a JSON object, when
    /// it has no `function` object (pointing the caller at
    /// [`ToolInfo::from_flat`] / [`ToolInfo::from_wire`]), or when the nested
    /// fields have the wrong JSON types.
    pub fn from_openai(value: Value) -> Result<Self> {
        let object = expect_object(&value, "a nested OpenAI tool info")?;
        if !object.contains_key("function") {
            return Err(Error::validation(
                "nested OpenAI tool info has no `function` object: use ToolInfo::from_flat \
                 for Python's flat shape, or ToolInfo::from_wire to accept either shape",
            ));
        }

        let mut info: Self = serde_json::from_value(value).map_err(|error| {
            Error::validation(format!("invalid nested OpenAI tool info: {error}"))
        })?;
        if info.kind.trim().is_empty() {
            "function".clone_into(&mut info.kind);
        }
        Ok(info)
    }

    /// Parses either wire shape, auto-detecting which one the value uses.
    ///
    /// An object with a `function` key is parsed as the nested `OpenAI` shape
    /// ([`ToolInfo::from_openai`]); every other object must carry a top-level
    /// `name` and is parsed as Python's flat shape ([`ToolInfo::from_flat`]).
    /// Unknown fields are tolerated in both shapes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the value is not a JSON object, has
    /// neither a `function` key nor a top-level `name` (so neither shape can be
    /// assumed), or fails the checks of the detected shape.
    pub fn from_wire(value: Value) -> Result<Self> {
        let object = expect_object(&value, "a tool info")?;
        if object.contains_key("function") {
            return Self::from_openai(value);
        }
        if !object.contains_key("name") {
            return Err(Error::validation(
                "tool info is neither shape: expected a nested `function` object (OpenAI) \
                 or a top-level `name` (Python flat)",
            ));
        }
        Self::from_flat(value)
    }
}

/// Renders `infos` as the Python-flat tool-info JSON array.
///
/// Mirrors [`crate::tool::schema::openai_tools`] for the flat shape: one
/// [`ToolInfo::flat`] object per info, in slice order. An empty slice renders
/// `[]`.
#[must_use]
pub fn flat_tools(infos: &[ToolInfo]) -> Value {
    Value::Array(infos.iter().map(ToolInfo::flat).collect())
}

/// Parses a JSON array of tool infos in either wire shape.
///
/// Each element goes through [`ToolInfo::from_wire`], so nested `OpenAI` objects
/// and Python-flat objects can be mixed in one array and unknown fields are
/// ignored. This is the batch counterpart of [`flat_tools`].
///
/// # Errors
///
/// Returns [`Error::Validation`] when the value is not a JSON array, or when one
/// element is not a valid tool info (the error names the element index).
pub fn from_wire_tools(value: Value) -> Result<Vec<ToolInfo>> {
    let Value::Array(items) = value else {
        return Err(Error::validation(format!(
            "tool infos must be a JSON array, got {}",
            json_type_name(&value)
        )));
    };

    items
        .into_iter()
        .enumerate()
        .map(|(index, item)| {
            ToolInfo::from_wire(item).map_err(|error| {
                Error::validation(format!("tool info at index {index} is invalid: {error}"))
            })
        })
        .collect()
}

/// Returns the parameters as an object, normalizing non-objects to `{}`.
///
/// Python's flat `parameters` field is `default_factory=dict`, so a non-object
/// (including `null`) is rendered as an empty schema object.
fn object_or_empty(parameters: &Value) -> Value {
    match parameters {
        Value::Object(_) => parameters.clone(),
        _ => json!({}),
    }
}

/// Borrows `value` as an object or reports [`Error::Validation`].
fn expect_object<'a>(value: &'a Value, expected: &str) -> Result<&'a Map<String, Value>> {
    value.as_object().ok_or_else(|| {
        Error::validation(format!(
            "{expected} must be a JSON object, got {}",
            json_type_name(value)
        ))
    })
}

/// Reads an optional string field, defaulting to `""`.
fn string_field(object: &Map<String, Value>, key: &str) -> String {
    object
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned()
}

/// Names a JSON value's type for error messages.
fn json_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

/// The nested `function` object of a [`ToolInfo`].
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct ToolFunctionInfo {
    /// Function name, unique among the tools offered in one request.
    pub name: String,
    /// Human-readable description shown to the model.
    pub description: String,
    /// JSON Schema of the accepted arguments.
    pub parameters: Value,
}

impl ToolFunctionInfo {
    /// Creates the nested function object.
    #[must_use]
    pub fn new(name: impl Into<String>, description: impl Into<String>, parameters: Value) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    // port of openjiuwen/core/foundation/tool/base.py ToolCard + exposure.py/execution.py

    #[test]
    fn default_card_has_python_defaults_and_a_fresh_id() {
        let first = ToolCard::default();
        let second = ToolCard::default();

        assert_eq!(first.id.len(), 32);
        assert!(first
            .id
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
        assert_ne!(first.id, second.id);

        assert_eq!(first.name, "");
        assert_eq!(first.description, "");
        assert_eq!(first.input_params, json!({}));
        assert!(first.properties.is_empty());
        assert_eq!(first.exposure, ToolExposure::Direct);
        assert_eq!(first.execution, ToolExecution::Server);
        assert!(first.parallel_safe);
        assert!(!first.stateless);
        assert!(!first.idempotent);
        assert_eq!(first.version, None);
        assert!(first.metadata.is_empty());
    }

    #[test]
    fn default_card_serializes_exactly_the_python_key_set() {
        let json = serde_json::to_value(ToolCard::default()).unwrap();
        let object = json.as_object().unwrap();

        let keys: std::collections::BTreeSet<&str> = object.keys().map(String::as_str).collect();
        assert_eq!(
            keys,
            std::collections::BTreeSet::from([
                "id",
                "name",
                "description",
                "input_params",
                "properties",
                "exposure",
                "execution",
                "parallel_safe",
                "stateless",
                "idempotent",
            ]),
            "a default card must serialize to the Python key set exactly"
        );

        let id = object["id"].as_str().unwrap();
        assert_eq!(id.len(), 32, "the id is Python's uuid4().hex form");
        assert!(id
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
        assert_eq!(object["name"], json!(""));
        assert_eq!(object["description"], json!(""));
        assert_eq!(object["input_params"], json!({}));
        assert_eq!(object["properties"], json!({}));
        assert_eq!(object["exposure"], json!("direct"));
        assert_eq!(object["execution"], json!("server"));
        assert_eq!(object["parallel_safe"], json!(true));
        assert_eq!(object["stateless"], json!(false));
        assert_eq!(object["idempotent"], json!(false));
        assert!(
            object.get("version").is_none(),
            "an unset version must be omitted"
        );
        assert!(
            object.get("metadata").is_none(),
            "an empty metadata map must be omitted"
        );
    }

    #[test]
    fn card_round_trips_python_shaped_json() {
        let wire = json!({
            "id": "abc",
            "name": "t",
            "description": "d",
            "input_params": {"type": "object"},
            "properties": {"x": 1},
            "exposure": "deferred",
            "execution": "client",
            "parallel_safe": false,
            "stateless": true,
            "idempotent": true,
            "version": "1",
            "metadata": {"k": "v"},
            "unexpected": 42,
        });

        let card: ToolCard = serde_json::from_value(wire.clone()).unwrap();
        assert_eq!(card.id, "abc");
        assert_eq!(card.name, "t");
        assert_eq!(card.description, "d");
        assert_eq!(card.input_params, json!({"type": "object"}));
        assert_eq!(card.properties.len(), 1);
        assert_eq!(card.properties["x"], json!(1));
        assert_eq!(card.exposure, ToolExposure::Deferred);
        assert_eq!(card.execution, ToolExecution::Client);
        assert!(!card.parallel_safe);
        assert!(card.stateless);
        assert!(card.idempotent);
        assert_eq!(card.version.as_deref(), Some("1"));
        assert_eq!(card.metadata["k"], json!("v"));

        let mut expected = wire.as_object().unwrap().clone();
        assert!(
            expected.remove("unexpected").is_some(),
            "the unknown key must be dropped on the way in"
        );
        let reserialized = serde_json::to_value(&card).unwrap();
        assert_eq!(Value::Object(expected), reserialized);

        let again: ToolCard = serde_json::from_value(reserialized.clone()).unwrap();
        assert_eq!(again, card);
        assert_eq!(serde_json::to_value(&again).unwrap(), reserialized);
    }

    #[test]
    fn card_serializes_python_field_names() {
        let card = ToolCard::new("calc", "does math", json!({"type": "object"}))
            .with_id("abc")
            .with_exposure(ToolExposure::Deferred)
            .with_execution(ToolExecution::Client);

        let json = serde_json::to_value(&card).unwrap();
        assert_eq!(json["id"], "abc");
        assert_eq!(json["name"], "calc");
        assert_eq!(json["description"], "does math");
        assert_eq!(json["input_params"], json!({"type": "object"}));
        assert_eq!(json["properties"], json!({}));
        assert_eq!(json["exposure"], "deferred");
        assert_eq!(json["execution"], "client");
        assert_eq!(json["parallel_safe"], true);
        assert_eq!(json["stateless"], false);
        assert_eq!(json["idempotent"], false);
        assert!(
            json.get("version").is_none(),
            "an absent version must be omitted"
        );
    }

    #[test]
    fn card_tolerates_unknown_fields_and_empty_objects() {
        let card: ToolCard = serde_json::from_value(json!({
            "id": "abc",
            "unexpected": {"nested": true},
            "another": 1,
        }))
        .unwrap();
        assert_eq!(card.id, "abc");
        assert_eq!(card.name, "");
        assert!(card.metadata.is_empty());

        let bare: ToolCard = serde_json::from_value(json!({})).unwrap();
        assert_eq!(bare.id.len(), 32);
        assert_eq!(bare.input_params, json!({}));
        assert_eq!(bare.exposure, ToolExposure::Direct);
        assert_eq!(bare.execution, ToolExecution::Server);
    }

    #[test]
    fn card_trait_exposes_identity_and_optional_description() {
        let card = ToolCard::new("calc", "does math", json!({})).with_id("abc");
        assert_eq!(Card::id(&card), "abc");
        assert_eq!(Card::name(&card), "calc");
        assert_eq!(Card::description(&card), Some("does math"));

        let undescribed = ToolCard::new("calc", "", json!({}));
        assert_eq!(Card::description(&undescribed), None);
    }

    #[test]
    fn tool_info_and_openai_tool_match_the_openai_shape() {
        let card = ToolCard::new("calc", "does math", json!({}));
        let info = card.tool_info();

        assert_eq!(info.kind, "function");
        assert_eq!(info.function.name, "calc");
        assert_eq!(info.function.description, "does math");
        assert_eq!(info.function.parameters, json!({"type": "object"}));

        let json = serde_json::to_value(&info).unwrap();
        assert_eq!(json["type"], "function");
        assert_eq!(json["function"]["name"], "calc");
        assert_eq!(json["function"]["description"], "does math");
        assert_eq!(json["function"]["parameters"], json!({"type": "object"}));
        assert!(
            json.get("name").is_none(),
            "the function fields must stay nested"
        );

        assert_eq!(card.openai_tool(), json);
        assert_eq!(ToolInfo::default().kind, "function");
        assert_eq!(ToolInfo::default().function.name, "");
    }

    #[test]
    fn openai_tool_keeps_declared_input_schema() {
        let schema = json!({
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        });
        let card = ToolCard::new("calc", "does math", schema.clone());
        assert_eq!(card.openai_tool()["function"]["parameters"], schema);
    }

    #[test]
    fn tool_info_projection_is_the_exact_function_shape() {
        // Python's `ToolInfo` is flat (`type`/`name`/`description`/`parameters`);
        // the documented Rust deviation nests the function fields in the OpenAI
        // tools shape, so the model-facing projection is asserted here as-is.
        let input_params = json!({
            "title": "Point",
            "type": "object",
            "properties": {
                "x": {"title": "X", "type": "integer"},
                "y": {"title": "Y", "type": "integer"},
            },
            "required": ["x", "y"],
        });
        let card = ToolCard::new("calc", "does math", input_params.clone())
            .with_properties(BTreeMap::from([("owner".to_owned(), json!("math"))]));

        let json = serde_json::to_value(card.tool_info()).unwrap();
        assert_eq!(
            json,
            json!({
                "type": "function",
                "function": {
                    "name": "calc",
                    "description": "does math",
                    "parameters": input_params,
                },
            }),
            "the projection carries exactly type/function(name, description, parameters)"
        );
        assert!(
            !json.to_string().contains("owner"),
            "properties are program-only and never reach the model"
        );
        assert_eq!(
            card.openai_tool(),
            json,
            "openai_tool must stay byte-identical to the ToolInfo projection"
        );
    }

    #[test]
    fn tool_info_keeps_a_pydantic_model_shaped_input_params() {
        // Python's `input_params` may be a `BaseModel` subclass; its JSON form is
        // the model-dumped schema, which must pass through unchanged.
        let dumped = json!({
            "title": "WeatherArgs",
            "type": "object",
            "properties": {
                "city": {"title": "City", "type": "string"},
                "days": {"default": 1, "title": "Days", "type": "integer"},
            },
            "required": ["city"],
        });
        let card = ToolCard::new("weather", "looks up the weather", dumped.clone());

        assert_eq!(card.tool_info().function.parameters, dumped);
        assert_eq!(card.openai_tool()["function"]["parameters"], dumped);
    }

    #[test]
    fn policy_flags_serialize_with_no_hidden_fields() {
        let deferred = ToolCard::new("t", "", json!({}))
            .with_exposure(ToolExposure::Deferred)
            .with_execution(ToolExecution::Client);

        let keys: Vec<String> = serde_json::to_value(&deferred)
            .unwrap()
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect();
        let default_keys: Vec<String> = serde_json::to_value(ToolCard::default())
            .unwrap()
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect();

        assert_eq!(keys, default_keys);
        assert!(keys.iter().any(|key| key == "exposure"));
        assert!(keys.iter().any(|key| key == "execution"));
    }

    #[test]
    fn builders_override_every_field() {
        let mut metadata = BTreeMap::new();
        metadata.insert("owner".to_owned(), json!("math-team"));
        let mut properties = BTreeMap::new();
        properties.insert("registry".to_owned(), json!("kernel"));

        let card = ToolCard::new("a", "b", json!({}))
            .with_id("id")
            .with_name("renamed")
            .with_description("described")
            .with_input_params(json!({"type": "string"}))
            .with_properties(properties)
            .with_exposure(ToolExposure::Deferred)
            .with_execution(ToolExecution::Client)
            .with_parallel_safe(false)
            .with_stateless(true)
            .with_idempotent(true)
            .with_version("1.2.3")
            .with_metadata(metadata)
            .with_metadata_entry("team", json!("kernel"));

        assert_eq!(card.id, "id");
        assert_eq!(card.name, "renamed");
        assert_eq!(card.description, "described");
        assert_eq!(card.input_params, json!({"type": "string"}));
        assert_eq!(card.properties["registry"], json!("kernel"));
        assert_eq!(card.exposure, ToolExposure::Deferred);
        assert_eq!(card.execution, ToolExecution::Client);
        assert!(!card.parallel_safe);
        assert!(card.stateless);
        assert!(card.idempotent);
        assert_eq!(card.version.as_deref(), Some("1.2.3"));
        assert_eq!(card.metadata["owner"], json!("math-team"));
        assert_eq!(card.metadata["team"], json!("kernel"));
    }

    #[test]
    fn generated_schema_does_not_bake_the_id_default() {
        let schema = schemars::SchemaGenerator::default().into_root_schema_for::<ToolCard>();
        let value = serde_json::to_value(&schema).unwrap();

        assert_eq!(value["type"], "object");
        assert!(value["properties"]["id"].get("default").is_none());
        assert!(value["properties"].get("input_params").is_some());
    }
}

// The flat/nested wire-conversion tests live in their own file to keep this
// module under the workspace file-size baseline.
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod flat_tests;
