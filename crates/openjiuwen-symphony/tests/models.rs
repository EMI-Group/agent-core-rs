//! Behaviour tests for the model layer and the KV asset store.
//!
//! These pin the wire format (`"type"` keys, skipped `semantic_content`, unknown-field
//! tolerance), the derived artifacts (JSON Schema, content hash, [`Card`] accessors),
//! and the deterministic comparison helpers used by the graph and planner. Everything
//! is in-process and deterministic: no network, no clock, no temp files outside the
//! store tests.
#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(clippy::float_cmp)]

use std::str::FromStr;

use openjiuwen_common::card::Card;
use openjiuwen_common::error::ErrorKind;
use openjiuwen_symphony::model::{
    fuzzy_name_distance, fuzzy_name_similarity, normalize_name_key, normalize_type,
    CapabilityDescriptor, CapabilityIo, CapabilityKey, DataTypeVocabulary,
};
use openjiuwen_symphony::store::{AssetStore, KvAssetStore};
use serde_json::{json, Value};

/// A fully populated descriptor used by several tests.
fn sample() -> CapabilityDescriptor {
    CapabilityDescriptor::new("tool", "sum", "Sum")
        .with_description("adds two numbers")
        .with_source("builtin")
        .with_available(true)
        .with_classification("math")
        .with_tags(["math", "numbers"])
        .with_inputs(vec![
            CapabilityIo::new("a", "int")
                .with_description("first addend")
                .with_required(true),
            CapabilityIo::new("b", "number").with_required(false),
        ])
        .with_outputs(vec![CapabilityIo::new("total", "float")])
        .with_semantic_content("internal semantic definition")
        .with_metadata("owner", json!("platform"))
}

#[test]
fn descriptor_round_trips_with_type_keys() {
    let descriptor = sample();
    let value = serde_json::to_value(&descriptor).expect("serializes");
    assert_eq!(value["capability_id"], "sum");
    assert_eq!(value["capability_type"], "tool");
    assert_eq!(value["available"], Value::Bool(true));
    assert_eq!(value["inputs"][0]["type"], "int");
    assert_eq!(value["inputs"][1]["type"], "number");
    assert_eq!(value["outputs"][0]["type"], "float");
    assert!(
        value["inputs"][0].get("io_type").is_none(),
        "the JSON key must be `type`, never `io_type`"
    );
    assert_eq!(value["metadata"]["owner"], "platform");

    let decoded: CapabilityDescriptor = serde_json::from_value(value).expect("deserializes");
    // `semantic_content` is excluded from output, so it cannot survive a round trip.
    let expected = descriptor.clone().with_semantic_content("");
    assert_eq!(decoded, expected);
    assert_eq!(decoded.inputs[0].io_type(), "int");
    assert_eq!(decoded.inputs[0].required, Some(true));
    assert_eq!(decoded.inputs[1].required, Some(false));
}

#[test]
fn available_defaults_to_true_for_python_parity() {
    let built = CapabilityDescriptor::new("tool", "sum", "Sum");
    assert!(built.available, "`new` must default `available` to true");
    assert!(CapabilityDescriptor::default().available);

    let decoded: CapabilityDescriptor = serde_json::from_value(json!({
        "capability_id": "sum",
        "capability_type": "tool",
        "name": "Sum",
    }))
    .expect("deserializes without an `available` key");
    assert!(
        decoded.available,
        "a missing `available` key must deserialize to true"
    );

    let explicit: CapabilityDescriptor = serde_json::from_value(json!({
        "capability_id": "sum",
        "capability_type": "tool",
        "name": "Sum",
        "available": false,
    }))
    .expect("deserializes an explicit `available`");
    assert!(!explicit.available);

    let withdrawn = built.with_available(false);
    assert!(!withdrawn.available);
    assert!(withdrawn.with_available(true).available);
}

#[test]
fn semantic_content_is_skipped_but_accepted() {
    let descriptor = sample();
    let value = serde_json::to_value(&descriptor).expect("serializes");
    assert!(
        value.get("semantic_content").is_none(),
        "Python excludes semantic_content from dumps"
    );

    let with_semantic: CapabilityDescriptor = serde_json::from_value(json!({
        "capability_id": "sum",
        "capability_type": "tool",
        "name": "Sum",
        "semantic_content": "private notes",
    }))
    .expect("deserializes");
    assert_eq!(with_semantic.semantic_content, "private notes");

    let absent: CapabilityDescriptor =
        serde_json::from_value(json!({"name": "Sum"})).expect("deserializes with defaults");
    assert_eq!(absent.semantic_content, String::new());
}

#[test]
fn unknown_fields_are_tolerated_everywhere() {
    let descriptor: CapabilityDescriptor = serde_json::from_value(json!({
        "capability_id": "sum",
        "capability_type": "tool",
        "name": "Sum",
        "future_descriptor_field": {"nested": true},
        "inputs": [{"name": "a", "type": "string", "future_io_field": 7}],
        "outputs": [],
    }))
    .expect("deserializes");
    assert_eq!(descriptor.capability_id, "sum");
    assert_eq!(descriptor.inputs[0].io_type(), "string");

    let io: CapabilityIo =
        serde_json::from_value(json!({"name": "x", "type": "text", "x": 1})).expect("deserializes");
    assert_eq!(io.name, "x");
}

#[test]
fn card_accessors_match_the_contract() {
    let descriptor = sample();
    assert_eq!(Card::id(&descriptor), "sum");
    assert_eq!(Card::name(&descriptor), "Sum");
    assert_eq!(Card::description(&descriptor), Some("adds two numbers"));

    let undescribed = CapabilityDescriptor::new("tool", "blank", "Blank");
    assert_eq!(Card::description(&undescribed), None);
    assert_eq!(
        Card::description(&undescribed.clone().with_description("   ")),
        None
    );
}

#[test]
fn descriptor_schema_derives_from_schemars() {
    let schema = schemars::schema_for!(CapabilityDescriptor);
    let value = serde_json::to_value(&schema).expect("schema serializes");
    let properties = &value["properties"];
    for field in [
        "capability_id",
        "capability_type",
        "name",
        "description",
        "source",
        "available",
        "inputs",
        "outputs",
        "classification",
        "tags",
        "content_hash",
        "metadata",
    ] {
        assert!(
            properties.get(field).is_some(),
            "schema must describe `{field}`"
        );
    }
    let io_schema =
        serde_json::to_value(schemars::schema_for!(CapabilityIo)).expect("io schema serializes");
    assert!(io_schema["properties"].get("type").is_some());
    assert!(io_schema["properties"].get("io_type").is_none());
}

#[test]
fn content_hash_is_stable_and_ignores_operational_fields() {
    let descriptor = sample();
    let hash = descriptor.compute_content_hash();
    assert_eq!(hash.len(), 64);
    assert!(hash
        .chars()
        .all(|character| character.is_ascii_hexdigit() && !character.is_ascii_uppercase()));
    assert_eq!(hash, descriptor.compute_content_hash());
    assert_eq!(hash, sample().compute_content_hash());

    let operational = descriptor
        .clone()
        .with_available(false)
        .with_semantic_content("different internal text")
        .with_metadata("owner", json!("someone-else"))
        .with_metadata("extra", json!([1, 2, 3]));
    assert_eq!(
        hash,
        operational.compute_content_hash(),
        "availability, metadata, and semantic content are not identity"
    );

    let reordered_metadata = descriptor
        .clone()
        .with_metadata("zeta", json!({"b": 2, "a": 1}))
        .with_metadata("alpha", json!({"d": 4, "c": 3}));
    assert_eq!(hash, reordered_metadata.compute_content_hash());

    assert_ne!(
        hash,
        descriptor
            .clone()
            .with_description("adds three numbers")
            .compute_content_hash()
    );
    assert_ne!(
        hash,
        descriptor
            .clone()
            .with_tags(["math"])
            .compute_content_hash()
    );
}

#[test]
fn descriptor_validation_names_the_offending_field() {
    sample().validate().expect("valid descriptor");

    let error = CapabilityDescriptor::new("tool", "  ", "Sum")
        .validate()
        .expect_err("blank id");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("capability_id"));

    let error = CapabilityDescriptor::new("", "sum", "Sum")
        .validate()
        .expect_err("blank type");
    assert!(error.to_string().contains("capability_type"));

    let error = CapabilityDescriptor::new("tool", "sum", "")
        .validate()
        .expect_err("blank name");
    assert!(error.to_string().contains("name:"));
}

#[test]
fn key_qualifies_displays_and_parses() {
    let key = CapabilityKey::new("tool", "sum");
    assert_eq!(key.qualified(), "tool:sum");
    assert_eq!(key.to_string(), "tool:sum");

    assert_eq!(CapabilityKey::from_str("tool:sum").unwrap(), key);
    assert_eq!(
        CapabilityKey::from_str("  tool : sum  ").unwrap(),
        CapabilityKey::new("tool", "sum")
    );
    assert_eq!(
        CapabilityKey::from_str("tool:a:b").unwrap(),
        CapabilityKey::new("tool", "a:b"),
        "ids may contain colons; split on the first one"
    );

    for invalid in ["", "noseparator", ":sum", "tool:", "   "] {
        let error = CapabilityKey::from_str(invalid).expect_err("malformed keys must be rejected");
        assert_eq!(error.kind(), ErrorKind::Validation, "input `{invalid}`");
    }
    assert!(CapabilityKey::from_str(":sum")
        .expect_err("blank type")
        .to_string()
        .contains("capability_type"));
    assert!(CapabilityKey::from_str("tool:")
        .expect_err("blank id")
        .to_string()
        .contains("capability_id"));
}

#[test]
fn data_type_vocabulary_normalizes_aliases() {
    let vocabulary = DataTypeVocabulary::standard();
    let cases = [
        ("str", "string"),
        ("int", "integer"),
        ("float", "number"),
        ("double", "number"),
        ("bool", "boolean"),
        ("dict", "object"),
        ("map", "object"),
        ("record", "object"),
        ("json", "object"),
        ("list", "array"),
        ("tuple", "array"),
        ("bytes", "bytes"),
        ("text", "text"),
        ("", "any"),
        ("   ", "any"),
        ("STRING", "string"),
        ("  InT ", "integer"),
        ("unknown-type", "unknown-type"),
    ];
    for (raw, expected) in cases {
        assert_eq!(vocabulary.normalize(raw), expected, "raw `{raw}`");
    }

    assert_eq!(
        vocabulary.canonical_types(),
        vec![
            "any", "string", "text", "number", "integer", "boolean", "object", "array", "null",
            "file", "image", "audio",
        ]
    );
}

#[test]
fn normalize_type_is_the_standard_shorthand() {
    assert_eq!(normalize_type("BOOL"), "boolean");
    assert_eq!(normalize_type(" dict "), "object");
    assert_eq!(normalize_type(""), "any");
    assert_eq!(normalize_type("whatever"), "whatever");
}

#[test]
fn can_feed_matrix_matches_the_contract() {
    let vocabulary = DataTypeVocabulary::standard();

    // Equal types feed each other, and everything feeds `any`.
    for canonical in vocabulary.canonical_types() {
        assert!(vocabulary.can_feed(canonical, canonical), "{canonical}");
        assert!(vocabulary.can_feed(canonical, "any"), "{canonical} -> any");
        assert!(
            !vocabulary.can_feed("any", canonical) || canonical == "any",
            "`any` must not feed `{canonical}`"
        );
        assert!(
            !vocabulary.can_feed(canonical, "string") || matches!(canonical, "string" | "text"),
            "{canonical} -> string"
        );
    }

    // Documented widenings, spelled through aliases.
    assert!(vocabulary.can_feed("int", "number"));
    assert!(vocabulary.can_feed("integer", "float"));
    assert!(vocabulary.can_feed("text", "string"));
    assert!(vocabulary.can_feed("text", "str"));
    assert!(vocabulary.is_subtype("integer", "number"));
    assert!(vocabulary.is_subtype("text", "string"));

    // Wrong direction and unrelated types do not feed.
    assert!(!vocabulary.can_feed("number", "integer"));
    assert!(!vocabulary.can_feed("string", "text"));
    assert!(!vocabulary.can_feed("boolean", "string"));
    assert!(!vocabulary.can_feed("object", "array"));
    assert!(!vocabulary.can_feed("image", "audio"));
    assert!(!vocabulary.can_feed("any", "string"));
    assert!(!vocabulary.can_feed("any", "object"));

    // Aliases of the same canonical type are interchangeable.
    for alias in ["str", "STRING", " String "] {
        assert!(vocabulary.can_feed(alias, "string"));
        assert!(vocabulary.can_feed("string", alias));
    }
    assert!(vocabulary.can_feed("dict", "json"));
    assert!(vocabulary.can_feed("list", "tuple"));
    assert!(vocabulary.can_feed("bool", "boolean"));
    assert!(vocabulary.can_feed("bytes", "any"));
    assert!(vocabulary.can_feed("bytes", "bytes"));
    assert!(!vocabulary.can_feed("bytes", "string"));

    // Blank types normalize to `any`.
    assert!(vocabulary.can_feed("", "any"));
    assert!(!vocabulary.can_feed("", "text"));

    // Unknown types only feed themselves.
    assert!(vocabulary.can_feed("Fancy", "fancy"));
    assert!(!vocabulary.can_feed("fancy", "string"));
}

#[test]
fn normalize_name_key_collapses_and_trims() {
    let cases = [
        ("Read__File", "read_file"),
        ("__read_file__", "read_file"),
        ("a--b..c", "a_b_c"),
        ("A B", "a_b"),
        ("CamelCase", "camelcase"),
        ("foo-bar.baz", "foo_bar_baz"),
        ("  ", ""),
        ("!!!", ""),
        ("", ""),
        ("Über-Ärger", "über_ärger"),
        ("already_snake", "already_snake"),
    ];
    for (raw, expected) in cases {
        assert_eq!(normalize_name_key(raw), expected, "raw `{raw}`");
    }
}

#[test]
fn fuzzy_name_distance_and_similarity_are_deterministic() {
    assert_eq!(fuzzy_name_distance("Read File", "read_file"), 0);
    assert_eq!(fuzzy_name_distance("kitten", "sitting"), 3);
    assert_eq!(fuzzy_name_distance("", "abc"), 3);
    assert_eq!(fuzzy_name_distance("abc", ""), 3);
    assert_eq!(fuzzy_name_distance("", ""), 0);
    assert_eq!(
        fuzzy_name_distance("sum", "Sum"),
        0,
        "keys are case-insensitive"
    );
    assert_eq!(
        fuzzy_name_distance("read_file", "write_file"),
        fuzzy_name_distance("write_file", "read_file"),
        "distance is symmetric"
    );

    assert_eq!(fuzzy_name_similarity("same", "SAME"), 1.0);
    assert_eq!(fuzzy_name_similarity("", "  "), 1.0);
    assert_eq!(fuzzy_name_similarity("", "abc"), 0.0);
    let partial = fuzzy_name_similarity("kitten", "sitting");
    assert!(
        partial > 0.0 && partial < 1.0,
        "unexpected similarity {partial}"
    );
    assert!(fuzzy_name_similarity("sum", "sums") < 1.0);
}

#[test]
fn input_and_output_schemas_shape_required_names() {
    let descriptor = sample();
    let input = descriptor.input_schema();
    assert_eq!(input["type"], "object");
    assert_eq!(input["properties"]["a"]["type"], "integer");
    assert_eq!(input["properties"]["a"]["description"], "first addend");
    assert_eq!(input["properties"]["b"]["type"], "number");
    assert_eq!(
        input["required"],
        json!(["a"]),
        "only required: true names are listed"
    );

    let output = descriptor.output_schema();
    assert_eq!(output["type"], "object");
    assert_eq!(output["properties"]["total"]["type"], "number");
    assert_eq!(output["required"], json!([]));
}

#[test]
fn schemas_include_defaults_omit_untyped_and_skip_nameless_ios() {
    let descriptor = CapabilityDescriptor::new("tool", "t", "T").with_inputs(vec![
        CapabilityIo::new("raw", "any"),
        CapabilityIo::new("custom", "FancyType"),
        CapabilityIo::new("limit", "integer")
            .with_default(json!(10))
            .with_required(true),
        CapabilityIo::new("limited", "integer")
            .with_default(json!(5))
            .with_required(false),
        CapabilityIo::new("   ", "string"),
    ]);
    let schema = descriptor.input_schema();
    let properties = schema["properties"].as_object().expect("properties object");

    assert!(
        properties["raw"].get("type").is_none(),
        "any omits its type"
    );
    assert!(properties["custom"].get("type").is_none());
    assert_eq!(properties["limit"]["default"], json!(10));
    assert_eq!(properties["limited"]["default"], json!(5));
    assert!(!properties.contains_key(""), "nameless IO is skipped");
    assert_eq!(schema["required"], json!(["limit"]));
}

#[tokio::test]
async fn descriptors_look_up_through_the_asset_store() {
    let store = KvAssetStore::in_memory();
    let descriptor = sample();
    // Stored values are the serialized descriptor, so `semantic_content` is dropped.
    let persisted = descriptor.clone().with_semantic_content("");
    store.put(&descriptor).await.expect("put");
    assert_eq!(
        store.get(&descriptor.key()).await.expect("get"),
        Some(persisted.clone())
    );
    assert_eq!(store.list().await.expect("list"), vec![persisted]);
    assert!(store.remove(&descriptor.key()).await.expect("remove"));
    assert_eq!(store.list().await.expect("list").len(), 0);
}

#[test]
fn descriptor_metadata_round_trips_as_a_sorted_map() {
    let descriptor = sample().with_metadata("alpha", json!(1));
    let decoded: CapabilityDescriptor =
        serde_json::from_value(serde_json::to_value(&descriptor).expect("serializes"))
            .expect("deserializes");
    assert_eq!(decoded.metadata.len(), 2);
    assert_eq!(decoded.metadata["alpha"], json!(1));
    assert_eq!(decoded.metadata["owner"], json!("platform"));
    let keys: Vec<&str> = decoded.metadata.keys().map(String::as_str).collect();
    assert_eq!(keys, vec!["alpha", "owner"], "BTreeMap keys are sorted");
}
