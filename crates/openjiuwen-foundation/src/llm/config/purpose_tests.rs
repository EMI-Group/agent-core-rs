//! Unit tests for the client-side request purpose, the internal metadata map, and
//! the legacy-`extra` compatibility rules shared by the three protocol builders.
//!
//! The purpose and metadata surface is telemetry/routing bookkeeping, so every
//! assertion here is about what *stays off* the wire as much as about what is
//! stored: the JSON shape of a request without them is unchanged, and the builder
//! tests in `openai/request.rs`, `anthropic/tests.rs` and `responses/request.rs`
//! prove the bodies never carry the keys.

use super::*;
use serde_json::json;

/// A request with a model name and nothing else set beyond the test's own fields.
fn request() -> ModelRequestConfig {
    ModelRequestConfig::new("gpt-4o")
}

#[test]
fn purpose_and_metadata_round_trip_through_json() {
    let request = request()
        .with_purpose_compaction()
        .with_metadata_entry("run_id", json!("eval-7"))
        .with_context_operation_id("op-1");

    let encoded = serde_json::to_value(&request).expect("serializes");
    assert_eq!(encoded["purpose"], json!("compaction"));
    assert_eq!(
        encoded["metadata"],
        json!({"run_id": "eval-7", "context_operation_id": "op-1"})
    );

    let restored: ModelRequestConfig = serde_json::from_value(encoded).expect("round-trips");
    assert_eq!(restored, request);
    assert!(restored.is_compaction());
    assert_eq!(
        restored.metadata.get("context_operation_id"),
        Some(&json!("op-1"))
    );
}

#[test]
fn a_python_shaped_config_decodes_purpose_and_metadata() {
    // Python's `ModelRequestConfig` has no `purpose` field: the hint travels as an
    // internal kwarg. A caller that persisted the Rust field still decodes it, and
    // unknown Python keys stay out of `extra` as before.
    let request: ModelRequestConfig = serde_json::from_value(json!({
        "model": "gpt-4o",
        "purpose": "compaction",
        "metadata": {"context_operation_id": "op-9", "run_id": "eval-3"},
        "request_purpose": "ignored-unknown-key",
    }))
    .expect("the Python-shaped config decodes");

    assert_eq!(request.effective_purpose(), Some(PURPOSE_COMPACTION));
    assert!(request.is_compaction());
    assert_eq!(request.metadata.get("run_id"), Some(&json!("eval-3")));
    assert_eq!(
        request.extra,
        Value::Null,
        "unknown top-level keys are still ignored"
    );
}

#[test]
fn the_default_request_serializes_without_the_new_keys() {
    let encoded = serde_json::to_value(request()).expect("serializes");
    assert_eq!(
        encoded,
        json!({
            "model_name": "gpt-4o",
            "temperature": null,
            "top_p": null,
            "max_tokens": null,
            "stop": [],
            "tools": null,
            "tool_choice": null,
            "stream": false,
            "extra": null,
        }),
        "the unset purpose/metadata fields do not change the JSON shape"
    );
}

#[test]
fn the_purpose_builders_set_the_well_known_values() {
    assert_eq!(PURPOSE_ASSISTANT, "assistant");
    assert_eq!(PURPOSE_COMPACTION, "compaction");

    assert_eq!(
        request().with_purpose_assistant().purpose.as_deref(),
        Some(PURPOSE_ASSISTANT)
    );
    assert_eq!(
        request().with_purpose_compaction().purpose.as_deref(),
        Some(PURPOSE_COMPACTION)
    );
    assert_eq!(
        request().with_purpose("routing-probe").purpose.as_deref(),
        Some("routing-probe"),
        "the purpose is an open-ended client-side string"
    );
}

#[test]
fn the_metadata_builders_store_diagnostic_entries() {
    let mut metadata = BTreeMap::new();
    metadata.insert("run_id".to_owned(), json!("eval-7"));
    let request = request()
        .with_metadata(metadata)
        .with_context_operation_id("op-3");
    assert_eq!(request.metadata.get("run_id"), Some(&json!("eval-7")));
    assert_eq!(
        request.metadata.get("context_operation_id"),
        Some(&json!("op-3"))
    );

    let replaced = request.with_metadata_entry("run_id", json!("eval-8"));
    assert_eq!(replaced.metadata.get("run_id"), Some(&json!("eval-8")));
    assert_eq!(
        replaced.metadata.get("context_operation_id"),
        Some(&json!("op-3")),
        "other entries survive"
    );
}

#[test]
fn effective_purpose_prefers_the_typed_field() {
    let request = request()
        .with_purpose_assistant()
        .with_extra(json!({"request_purpose": "compaction", "purpose": "compaction"}));
    assert_eq!(request.effective_purpose(), Some(PURPOSE_ASSISTANT));
    assert!(
        !request.is_compaction(),
        "the field wins over the legacy spellings"
    );
}

#[test]
fn effective_purpose_falls_back_to_the_legacy_spellings() {
    let legacy_kwarg = request().with_extra(json!({"request_purpose": "compaction"}));
    assert_eq!(legacy_kwarg.effective_purpose(), Some(PURPOSE_COMPACTION));
    assert!(legacy_kwarg.is_compaction());

    let legacy_field = request().with_extra(json!({"purpose": "compaction"}));
    assert_eq!(legacy_field.effective_purpose(), Some(PURPOSE_COMPACTION));
    assert!(legacy_field.is_compaction());

    let both = request().with_extra(json!({
        "request_purpose": "assistant",
        "purpose": "compaction",
    }));
    assert_eq!(
        both.effective_purpose(),
        Some(PURPOSE_ASSISTANT),
        "Python's internal kwarg spelling wins over the field spelling"
    );
}

#[test]
fn effective_purpose_ignores_blank_and_non_string_values() {
    assert_eq!(
        request().with_purpose("   ").effective_purpose(),
        None,
        "a blank field is treated as unset"
    );
    assert_eq!(
        request()
            .with_extra(json!({"request_purpose": "  "}))
            .effective_purpose(),
        None
    );
    assert_eq!(
        request()
            .with_extra(json!({"request_purpose": " ", "purpose": "compaction"}))
            .effective_purpose(),
        Some(PURPOSE_COMPACTION),
        "a blank legacy value falls through to the next spelling"
    );
    assert_eq!(
        request()
            .with_extra(json!({"request_purpose": " compaction "}))
            .effective_purpose(),
        Some(PURPOSE_COMPACTION),
        "the resolved purpose is trimmed"
    );

    let non_string = request().with_extra(json!({
        "request_purpose": 7,
        "purpose": {"mode": "compaction"},
    }));
    assert_eq!(
        non_string.effective_purpose(),
        None,
        "non-string legacy values are ignored, not forwarded"
    );
    assert!(!non_string.is_compaction());
}

#[test]
fn effective_purpose_tolerates_a_non_object_extra() {
    for extra in [
        Value::Null,
        json!("compaction"),
        json!([1, 2, 3]),
        json!(7),
        json!(true),
    ] {
        let request = request().with_extra(extra.clone());
        assert_eq!(
            request.effective_purpose(),
            None,
            "extra {extra} carries no purpose"
        );
    }
}

#[test]
fn is_compaction_recognises_only_the_compaction_purpose() {
    assert!(!request().is_compaction());
    assert!(!request().with_purpose_assistant().is_compaction());
    assert!(!request().with_purpose("Compaction").is_compaction());
    assert!(request().with_purpose_compaction().is_compaction());
    assert!(request()
        .with_extra(json!({"request_purpose": "compaction"}))
        .is_compaction());
}

#[test]
fn extra_without_internal_keys_keeps_every_other_key() {
    let request = request().with_extra(json!({
        "top_k": 5,
        "request_purpose": "compaction",
        "purpose": "compaction",
        "metadata": {"run_id": "eval-7"},
        "context_operation_id": "op-1",
    }));
    assert_eq!(
        request.extra_without_internal_keys(),
        json!({"top_k": 5}),
        "only the internal keys are stripped"
    );
    assert_eq!(
        request.extra.as_object().expect("object").len(),
        5,
        "the request's own extra map is untouched"
    );

    let only_internal =
        ModelRequestConfig::new("gpt-4o").with_extra(json!({"purpose": "compaction"}));
    assert_eq!(only_internal.extra_without_internal_keys(), json!({}));
}

#[test]
fn extra_without_internal_keys_passes_non_objects_through() {
    for extra in [Value::Null, json!("compaction"), json!([1, 2]), json!(3)] {
        let request = request().with_extra(extra.clone());
        assert_eq!(request.extra_without_internal_keys(), extra);
    }
}
