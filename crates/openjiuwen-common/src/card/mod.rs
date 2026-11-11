//! Cards: serialized identity and metadata shared across process boundaries.
//!
//! A card is the registry-safe description of an agent, tool, workflow, or model: it
//! round-trips through JSON, exposes a JSON Schema, and carries an id plus a name that
//! is unique inside its namespace.
//!
//! This module ports the Python `openjiuwen.core.common.schema.card` module:
//! `BaseCard` (id/name/description plus `to_str()`) becomes [`CardBase`], the pydantic
//! base class becomes the [`Card`] trait (so concrete cards embed [`CardBase`] with
//! `#[serde(flatten)]` and still serialize flat), and `uuid.uuid4().hex` ids are
//! produced by [`crate::utils::id::uuid_hex`].
//!
//! ```
//! use openjiuwen_common::card::{Card, CardBase};
//!
//! let card = CardBase::new("calculator").with_description("does math");
//! assert_eq!(card.name(), "calculator");
//! assert_eq!(card.description(), Some("does math"));
//! ```

use std::collections::BTreeMap;
use std::fmt;

use schemars::JsonSchema;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Value;

use crate::error::{Error, Result};
use crate::utils::id::uuid_hex;

/// Serialized identity and metadata for a registry entry.
///
/// The trait mirrors `docs/ARCHITECTURE.md` §2.1: implementors are serializable,
/// schema-bearing, cloneable, and safe to share across threads. Concrete cards embed
/// [`CardBase`] (usually with `#[serde(flatten)]`) and add their own fields.
pub trait Card: Serialize + DeserializeOwned + JsonSchema + Clone + Send + Sync + 'static {
    /// Returns the unique identifier of this card.
    fn id(&self) -> &str;

    /// Returns the name of this card, unique inside its namespace.
    fn name(&self) -> &str;

    /// Returns the human-readable description, if any.
    fn description(&self) -> Option<&str>;
}

/// The common card fields reused by all concrete cards.
///
/// Python parity: `BaseCard` in `openjiuwen/core/common/schema/card.py` with
/// `id` (default `uuid4().hex`), `name` (default `""`), and `description`
/// (default `""`). In Rust `description` is an `Option<String>` so that an absent
/// value serializes as an omitted key instead of an empty string.
///
/// Unknown JSON fields are tolerated and dropped on deserialization; every field has a
/// serde default, so `{}` deserializes to a card with a freshly generated id.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct CardBase {
    /// Unique identifier; Python's `uuid.uuid4().hex` form by default.
    pub id: String,
    /// Name of the card, unique inside its namespace.
    pub name: String,
    /// Human-readable description of the capability.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Optional version string of the described capability.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    /// Free-form metadata attached to the card.
    pub metadata: BTreeMap<String, Value>,
}

impl CardBase {
    /// Creates a card with `name` and a freshly generated id.
    ///
    /// `description` and `version` stay unset and `metadata` is empty.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            id: uuid_hex(),
            name: name.into(),
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
        self.description = Some(description.into());
        self
    }

    /// Returns this card with the given version.
    #[must_use]
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = Some(version.into());
        self
    }

    /// Returns this card with the given metadata, replacing any existing entries.
    #[must_use]
    pub fn with_metadata(mut self, metadata: BTreeMap<String, Value>) -> Self {
        self.metadata = metadata;
        self
    }

    /// Returns this card with one extra metadata entry inserted.
    #[must_use]
    pub fn with_metadata_entry(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Renders the compact identity form used in logs and error messages.
    ///
    /// Python parity: `BaseCard.to_str()` returns `f"id={id},name={name}"`.
    #[must_use]
    pub fn to_str(&self) -> String {
        format!("id={},name={}", self.id, self.name)
    }
}

impl Default for CardBase {
    /// Creates an anonymous card with a freshly generated id (never an empty id).
    fn default() -> Self {
        Self {
            id: uuid_hex(),
            name: String::new(),
            description: None,
            version: None,
            metadata: BTreeMap::new(),
        }
    }
}

impl Card for CardBase {
    fn id(&self) -> &str {
        &self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }
}

/// A typed card identity.
///
/// Serializes transparently as the inner string and generates a Python-style
/// `uuid4().hex` value on [`CardId::generate`] / [`Default`].
#[derive(
    Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, JsonSchema,
)]
#[serde(transparent)]
pub struct CardId(String);

impl CardId {
    /// Wraps an existing identifier string.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Generates a fresh identifier in Python `uuid.uuid4().hex` form.
    #[must_use]
    pub fn generate() -> Self {
        Self(uuid_hex())
    }

    /// Returns the identifier as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for CardId {
    /// Generates a fresh identifier (never an empty string).
    fn default() -> Self {
        Self::generate()
    }
}

impl fmt::Display for CardId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<&str> for CardId {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

impl From<String> for CardId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<CardId> for String {
    fn from(value: CardId) -> Self {
        value.0
    }
}

/// Returns the JSON Schema of a concrete card type.
///
/// The schema is generated with `schemars` from the `JsonSchema` bound carried by
/// [`Card`], so subclasses that flatten [`CardBase`] and add fields get the extra
/// properties automatically. Unknown fields stay allowed (no `additionalProperties:
/// false`), matching serde's tolerant deserialization.
///
/// # Errors
///
/// Returns [`Error::Validation`] if the generated schema cannot be converted into a
/// [`serde_json::Value`].
pub fn card_schema<C: Card>() -> Result<Value> {
    let schema = schemars::SchemaGenerator::default().into_root_schema_for::<C>();
    serde_json::to_value(&schema)
        .map_err(|e| Error::validation(format!("failed to serialize card schema: {e}")))
}

/// Returns the JSON Schema of [`CardBase`].
///
/// A convenience wrapper over [`card_schema`] for callers that only need the shared
/// card fields.
///
/// # Errors
///
/// Same conditions as [`card_schema`].
pub fn card_base_schema() -> Result<Value> {
    card_schema::<CardBase>()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    // port of openjiuwen/core/common/schema/card.py

    /// A concrete card that embeds `CardBase` and adds one field, the way the Python
    /// subclasses (`ToolCard`, `AgentCard`, ...) add their own fields.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
    struct TestCard {
        #[serde(flatten)]
        base: CardBase,
        input_params: Option<Value>,
    }

    impl Card for TestCard {
        fn id(&self) -> &str {
            &self.base.id
        }

        fn name(&self) -> &str {
            &self.base.name
        }

        fn description(&self) -> Option<&str> {
            self.base.description.as_deref()
        }
    }

    fn sample_base() -> CardBase {
        CardBase::new("calculator")
            .with_id("0123456789abcdef0123456789abcdef")
            .with_description("does math")
            .with_version("1.2.3")
            .with_metadata_entry("owner", Value::from("math-team"))
    }

    #[test]
    fn card_base_round_trips_with_metadata_and_version() {
        let card = sample_base();
        let json = serde_json::to_value(&card).unwrap();

        assert_eq!(json["id"], "0123456789abcdef0123456789abcdef");
        assert_eq!(json["name"], "calculator");
        assert_eq!(json["description"], "does math");
        assert_eq!(json["version"], "1.2.3");
        assert_eq!(json["metadata"]["owner"], "math-team");

        let back: CardBase = serde_json::from_value(json).unwrap();
        assert_eq!(back, card);
    }

    #[test]
    fn card_base_skips_absent_optional_fields() {
        let card = CardBase::new("nameless").with_id("abc");
        let json = serde_json::to_value(&card).unwrap();

        assert_eq!(json.as_object().unwrap().len(), 3);
        assert!(json.get("description").is_none());
        assert!(json.get("version").is_none());
        assert_eq!(json["metadata"], serde_json::json!({}));
    }

    #[test]
    fn card_base_tolerates_unknown_fields() {
        let json = serde_json::json!({
            "id": "abc",
            "name": "tool",
            "unexpected": {"nested": true},
        });

        let card: CardBase = serde_json::from_value(json).unwrap();
        assert_eq!(card.id, "abc");
        assert_eq!(card.name, "tool");
        assert_eq!(card.description, None);
        assert!(card.metadata.is_empty());
    }

    #[test]
    fn card_base_defaults_fill_missing_fields() {
        let card: CardBase = serde_json::from_value(serde_json::json!({})).unwrap();

        assert_eq!(card.id.len(), 32);
        assert_eq!(card.name, "");
        assert_eq!(card.description, None);
        assert_eq!(card.version, None);
        assert!(card.metadata.is_empty());

        // Python parity: an explicit JSON `null` description is treated as absent.
        let with_null: CardBase =
            serde_json::from_value(serde_json::json!({"description": null})).unwrap();
        assert_eq!(with_null.description, None);
    }

    #[test]
    fn default_and_new_generate_distinct_ids() {
        let a = CardBase::default();
        let b = CardBase::new("other");

        assert_eq!(a.id.len(), 32);
        assert_ne!(a.id, b.id);
    }

    #[test]
    fn to_str_matches_python() {
        let card = CardBase::new("calculator").with_id("abc");
        assert_eq!(card.to_str(), "id=abc,name=calculator");
    }

    #[test]
    fn card_id_generates_hex_and_is_unique() {
        let first = CardId::generate();
        let second = CardId::generate();

        assert_eq!(first.as_str().len(), 32);
        assert!(first
            .as_str()
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
        assert_ne!(first, second);
        assert_ne!(CardId::default(), first);
    }

    #[test]
    fn card_id_round_trips_transparently() {
        let id = CardId::from("abc123");
        assert_eq!(serde_json::to_value(&id).unwrap(), Value::from("abc123"));
        assert_eq!(
            serde_json::from_value::<CardId>(Value::from("abc123")).unwrap(),
            id
        );
        assert_eq!(id.to_string(), "abc123");
        assert_eq!(String::from(id.clone()), "abc123".to_owned());
        assert_eq!(CardId::new("x"), CardId::from("x".to_owned()));
    }

    #[test]
    fn flattened_test_card_serializes_flat_and_round_trips() {
        let card = TestCard {
            base: sample_base(),
            input_params: Some(serde_json::json!({"type": "object"})),
        };

        let json = serde_json::to_value(&card).unwrap();
        let object = json.as_object().unwrap();
        assert!(object.contains_key("id"));
        assert!(object.contains_key("name"));
        assert!(object.contains_key("input_params"));
        assert!(
            object.get("base").is_none(),
            "flatten must not nest the base card"
        );
        assert_eq!(json["metadata"]["owner"], "math-team");

        let back: TestCard = serde_json::from_value(json).unwrap();
        assert_eq!(back, card);

        assert_eq!(Card::id(&card), "0123456789abcdef0123456789abcdef");
        assert_eq!(Card::name(&card), "calculator");
        assert_eq!(Card::description(&card), Some("does math"));

        let without_params: TestCard =
            serde_json::from_value(serde_json::json!({"name": "bare"})).unwrap();
        assert_eq!(without_params.input_params, None);
        assert_eq!(without_params.base.name, "bare");
    }

    #[test]
    fn schemas_expose_card_properties_and_allow_unknown_fields() {
        let schema = card_base_schema().unwrap();
        let properties = schema["properties"].as_object().unwrap();

        assert!(properties.contains_key("id"));
        assert!(properties.contains_key("name"));
        assert!(properties.contains_key("metadata"));
        assert_ne!(
            schema.get("additionalProperties"),
            Some(&Value::Bool(false)),
            "unknown fields must stay allowed"
        );

        let generic = card_schema::<TestCard>().unwrap();
        let generic_props = generic["properties"].as_object().unwrap();
        assert!(generic_props.contains_key("id"));
        assert!(generic_props.contains_key("input_params"));
    }

    #[test]
    fn card_base_implements_card_and_is_shareable() {
        fn assert_card<C: Card + Send + Sync + 'static>() {}
        assert_card::<CardBase>();

        // Accessors go through the trait as well.
        let card = CardBase::new("trait-access");
        assert_eq!(Card::name(&card), "trait-access");
        assert_eq!(Card::description(&card), None);
    }
}
