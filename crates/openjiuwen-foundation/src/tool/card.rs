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

use std::collections::BTreeMap;

use openjiuwen_common::card::Card;
use openjiuwen_common::utils::id::uuid_hex;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

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
    /// Free-form metadata attached to the card.
    pub metadata: BTreeMap<String, Value>,
}

impl Default for ToolCard {
    /// Python defaults: a fresh `uuid4().hex` id, empty name/description and
    /// input schema, `direct`/`server` policy, parallel-safe and stateful.
    fn default() -> Self {
        Self {
            id: uuid_hex(),
            name: String::new(),
            description: String::new(),
            input_params: json!({}),
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
/// Python parity: `ToolInfo` in `openjiuwen/core/foundation/tool/schema.py`,
/// serialized in the `OpenAI` tools shape
/// `{"type": "function", "function": {"name": ..., "description": ...,
/// "parameters": ...}}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct ToolInfo {
    /// The tool kind; always `"function"`.
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
        assert_eq!(first.exposure, ToolExposure::Direct);
        assert_eq!(first.execution, ToolExecution::Server);
        assert!(first.parallel_safe);
        assert!(!first.stateless);
        assert!(!first.idempotent);
        assert_eq!(first.version, None);
        assert!(first.metadata.is_empty());
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

        let card = ToolCard::new("a", "b", json!({}))
            .with_id("id")
            .with_name("renamed")
            .with_description("described")
            .with_input_params(json!({"type": "string"}))
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
