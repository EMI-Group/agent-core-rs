//! External-consumer integration tests for card JSON parity.
//!
//! These tests only touch the crate's public surface (`openjiuwen_common::card::*`),
//! exactly the way downstream crates embed cards, and pin the wire format of the
//! Python SDK's `BaseCard` (`openjiuwen/core/common/schema/card.py`):
//! `to_str()` renders `id=<id>,name=<name>`, absent optional fields are omitted,
//! unknown fields are tolerated, and ids are `uuid.uuid4().hex` strings.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeSet;

use openjiuwen_common::card::{card_base_schema, card_schema, Card, CardBase, CardId};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// A downstream-style card: the shared `CardBase` fields flattened in, plus its own
/// fields, the way the Python subclasses (`ToolCard`, `AgentCard`, ...) extend `BaseCard`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
struct ToolLikeCard {
    #[serde(flatten)]
    base: CardBase,
    #[serde(default)]
    kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    input_schema: Option<Value>,
}

impl Card for ToolLikeCard {
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

/// Compile-time proof that a card type satisfies the trait's thread-safety contract.
fn assert_card<C: Card + Send + Sync + 'static>() {}

/// Python `uuid.uuid4().hex` form: 32 lowercase ASCII hex characters.
fn is_uuid_hex(value: &str) -> bool {
    value.len() == 32
        && value
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
}

fn sample_tool_card() -> ToolLikeCard {
    ToolLikeCard {
        base: CardBase::new("calculator")
            .with_id("0123456789abcdef0123456789abcdef")
            .with_description("does math")
            .with_version("1.2.3")
            .with_metadata_entry("owner", Value::from("math-team"))
            .with_metadata_entry("limits", json!({"max_ops": 10, "enabled": true})),
        kind: "function".to_owned(),
        input_schema: Some(json!({"type": "object"})),
    }
}

#[test]
fn card_base_serializes_only_core_keys_when_optionals_absent() {
    let card = CardBase::new("calculator").with_id("abc");
    let json = serde_json::to_value(&card).unwrap();
    let object = json.as_object().unwrap();

    let keys: BTreeSet<&str> = object.keys().map(String::as_str).collect();
    assert_eq!(keys, BTreeSet::from(["id", "metadata", "name"]));

    assert_eq!(json["id"], "abc");
    assert_eq!(json["name"], "calculator");
    assert_eq!(json["metadata"], json!({}));

    // Python parity: unset `description` and `version` are omitted, not `null`.
    assert!(json.get("description").is_none());
    assert!(json.get("version").is_none());
}

#[test]
fn card_base_round_trips_with_metadata_and_version() {
    let card = CardBase::new("calculator")
        .with_id("0123456789abcdef0123456789abcdef")
        .with_description("does math")
        .with_version("1.2.3")
        .with_metadata_entry("owner", Value::from("math-team"))
        .with_metadata_entry("limits", json!({"max_ops": 10, "enabled": true}));

    let encoded = serde_json::to_string(&card).unwrap();
    let decoded: CardBase = serde_json::from_str(&encoded).unwrap();
    assert_eq!(decoded, card, "JSON round-trip must preserve every field");

    let json = serde_json::to_value(&decoded).unwrap();
    assert_eq!(json["description"], "does math");
    assert_eq!(json["version"], "1.2.3");

    // Metadata values keep their JSON types, including nested structures.
    let limits = &json["metadata"]["limits"];
    assert_eq!(limits["max_ops"].as_u64(), Some(10));
    assert!(limits["max_ops"].is_number());
    assert_eq!(limits["enabled"].as_bool(), Some(true));
    assert_eq!(json["metadata"]["owner"], "math-team");
}

#[test]
fn card_base_tolerates_unknown_fields() {
    let json = json!({
        "id": "abc",
        "name": "tool",
        "unexpected": {"nested": true},
    });

    let card: CardBase = serde_json::from_value(json).unwrap();
    assert_eq!(card.id, "abc");
    assert_eq!(card.name, "tool");
    assert_eq!(card.description, None);
    assert_eq!(card.version, None);
    assert!(card.metadata.is_empty());

    // The unknown key is dropped, never re-serialized.
    let reserialized = serde_json::to_value(&card).unwrap();
    assert!(reserialized.get("unexpected").is_none());
}

#[test]
fn card_base_defaults_fill_missing_fields() {
    let card: CardBase = serde_json::from_value(json!({})).unwrap();

    assert!(
        is_uuid_hex(&card.id),
        "a missing id must be generated, got {:?}",
        card.id
    );
    assert_eq!(card.name, "");
    assert_eq!(card.description, None);
    assert_eq!(card.version, None);
    assert!(card.metadata.is_empty());

    // Python parity: an explicit JSON `null` description is treated as absent.
    let with_null: CardBase = serde_json::from_value(json!({"description": null})).unwrap();
    assert_eq!(with_null.description, None);
    assert_eq!(with_null.name, "");
    assert!(is_uuid_hex(&with_null.id));
}

#[test]
fn generated_ids_are_uuid_hex_and_unique() {
    let default_card = CardBase::default();
    let named_card = CardBase::new("x");
    let generated = CardId::generate();
    let default_id = CardId::default();

    for id in [
        default_card.id.as_str(),
        named_card.id.as_str(),
        generated.as_str(),
        default_id.as_str(),
    ] {
        assert!(is_uuid_hex(id), "not a uuid4-hex id: {id:?}");
    }

    assert_ne!(CardBase::default().id, CardBase::default().id);
    assert_ne!(CardId::generate(), CardId::generate());
}

#[test]
fn to_str_matches_python_base_card() {
    // Python: `f"id={self.id},name={self.name}"`.
    let card = CardBase::new("calculator").with_id("abc");
    assert_eq!(card.to_str(), "id=abc,name=calculator");
    assert_eq!(card.to_str(), format!("id={},name={}", card.id, card.name));

    let unnamed = CardBase::new("").with_id("abc");
    assert_eq!(unnamed.to_str(), "id=abc,name=");
}

#[test]
fn flattened_downstream_card_serializes_flat() {
    let card = sample_tool_card();
    let json = serde_json::to_value(&card).unwrap();
    let object = json.as_object().unwrap();

    assert!(object.contains_key("id"));
    assert!(object.contains_key("name"));
    assert!(object.contains_key("kind"));
    assert!(object.contains_key("input_schema"));
    assert!(
        object.get("base").is_none(),
        "flatten must not nest the base card"
    );
    assert_eq!(json["id"], "0123456789abcdef0123456789abcdef");
    assert_eq!(json["kind"], "function");
    assert_eq!(json["metadata"]["owner"], "math-team");
}

#[test]
fn flattened_downstream_card_round_trips_and_exposes_accessors() {
    let card = sample_tool_card();

    let encoded = serde_json::to_string(&card).unwrap();
    let decoded: ToolLikeCard = serde_json::from_str(&encoded).unwrap();
    assert_eq!(decoded, card);

    assert_eq!(Card::id(&decoded), "0123456789abcdef0123456789abcdef");
    assert_eq!(Card::name(&decoded), "calculator");
    assert_eq!(Card::description(&decoded), Some("does math"));

    let minimal: ToolLikeCard = serde_json::from_value(json!({"name": "bare"})).unwrap();
    assert_eq!(minimal.kind, "");
    assert_eq!(minimal.input_schema, None);
    assert_eq!(minimal.base.name, "bare");
    assert_eq!(minimal.base.description, None);

    assert_card::<ToolLikeCard>();
    assert_card::<CardBase>();
}

#[test]
fn card_base_schema_exposes_core_properties() {
    let schema = card_base_schema().unwrap();
    assert_eq!(schema["type"], "object");

    let properties = schema["properties"].as_object().unwrap();
    assert!(properties.contains_key("id"));
    assert!(properties.contains_key("name"));
    assert!(properties.contains_key("metadata"));

    assert_ne!(
        schema.get("additionalProperties"),
        Some(&Value::Bool(false)),
        "unknown fields must stay allowed"
    );

    // `card_schema` is the generic form of `card_base_schema`; note that the derived
    // schema embeds a freshly generated `id` default, so only the shape is compared.
    let generic = card_schema::<CardBase>().unwrap();
    let generic_properties = generic["properties"].as_object().unwrap();
    assert!(generic_properties.contains_key("id"));
    assert!(generic_properties.contains_key("name"));
    assert_eq!(generic["type"], "object");
}

#[test]
fn downstream_card_schema_includes_extra_properties() {
    let schema = card_schema::<ToolLikeCard>().unwrap();
    let properties = schema["properties"].as_object().unwrap();

    assert!(properties.contains_key("id"));
    assert!(properties.contains_key("name"));
    assert!(properties.contains_key("kind"));
    assert!(properties.contains_key("input_schema"));
    assert_ne!(
        schema.get("additionalProperties"),
        Some(&Value::Bool(false))
    );
}

#[test]
fn card_id_serializes_transparently() {
    let id = CardId::new("x");
    assert_eq!(serde_json::to_string(&id).unwrap(), "\"x\"");
    assert_eq!(serde_json::to_value(&id).unwrap(), json!("x"));

    let decoded: CardId = serde_json::from_str("\"x\"").unwrap();
    assert_eq!(decoded, id);
}

#[test]
fn card_id_conversions_behave() {
    let from_str = CardId::from("abc123");
    let from_string = CardId::from("abc123".to_owned());
    assert_eq!(from_str, from_string);
    assert_eq!(from_str, CardId::new("abc123"));

    assert_eq!(from_str.as_str(), "abc123");
    assert_eq!(from_str.to_string(), "abc123");
    assert_eq!(String::from(from_str.clone()), "abc123".to_owned());

    assert_ne!(CardId::new("a"), CardId::new("b"));
}
