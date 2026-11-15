//! Behaviour suite for `model::normalization`.
//!
//! Covers the three layers the module ports from Python: name folding, the
//! corpus-derived vocabulary (with its validated content hash), and I/O-spec
//! normalization with its full audit trail. Everything is offline and deterministic.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]

use openjiuwen_symphony::model::{
    builtin_io_name_vocabulary, normalize_io_name, normalize_io_specs,
    normalize_io_specs_for_direction, normalize_io_specs_with_audit,
    normalize_io_specs_with_issues, resolve_data_type, sanitize_io_specs, CapabilityDescriptor,
    CapabilityIo, DataTypeSource, IONameVocabulary, IoDirection, MAX_IO_NAME_LEN, MAX_TERM_VALUES,
    MAX_VOCABULARY_ALIASES, MAX_VOCABULARY_TERMS,
};
use openjiuwen_symphony::redaction::REDACTED;
use serde_json::{json, Value};

/// Builds a descriptor with the given `(name, type)` inputs and outputs.
fn descriptor(id: &str, inputs: &[(&str, &str)], outputs: &[(&str, &str)]) -> CapabilityDescriptor {
    CapabilityDescriptor::new("tool", id, id)
        .with_inputs(
            inputs
                .iter()
                .map(|(name, io_type)| CapabilityIo::new(*name, *io_type))
                .collect(),
        )
        .with_outputs(
            outputs
                .iter()
                .map(|(name, io_type)| CapabilityIo::new(*name, *io_type))
                .collect(),
        )
}

/// Returns the issue codes of `issues`, for readable assertions.
fn issue_codes(issues: &[openjiuwen_symphony::model::NormalizationIssue]) -> Vec<&str> {
    issues.iter().map(|issue| issue.code.as_str()).collect()
}

/// Returns the deepest container nesting of `value` (a scalar has depth 1).
fn nesting(value: &Value) -> usize {
    match value {
        Value::Object(map) => 1 + map.values().map(nesting).max().unwrap_or(0),
        Value::Array(items) => 1 + items.iter().map(nesting).max().unwrap_or(0),
        _ => 1,
    }
}

#[test]
fn names_fold_unicode_punctuation_and_separators() {
    assert_eq!(normalize_io_name("  Read File (v2)! ", ""), "read_file_v2");
    assert_eq!(
        normalize_io_name("ＦＵＬＬ－ＷＩＤＴＨ　Ｔｅｘｔ", ""),
        "full_width_text"
    );
    assert_eq!(normalize_io_name("\u{00A0}query\u{00A0}", ""), "query");
    assert_eq!(
        normalize_io_name("__lead--and_trail__", ""),
        "lead_and_trail"
    );
    assert_eq!(normalize_io_name("a.b/c\\d", ""), "a_b_c_d");
    assert_eq!(normalize_io_name("Über Ärger", ""), "über_ärger");
    assert_eq!(normalize_io_name("Prompt", ""), "prompt");
}

#[test]
fn names_fall_back_are_bounded_and_idempotent() {
    assert_eq!(normalize_io_name("", "Result"), "result");
    assert_eq!(normalize_io_name("!!!", "Result"), "result");
    assert_eq!(normalize_io_name("", ""), "");

    let long = "W".repeat(MAX_IO_NAME_LEN * 2);
    let bounded = normalize_io_name(&long, "");
    assert_eq!(bounded.chars().count(), MAX_IO_NAME_LEN);
    assert_eq!(normalize_io_name(&bounded, ""), bounded);

    // Cutting a word-separated token may land on a separator; it is trimmed and the
    // result stays a fixed point.
    let words = "ab ".repeat(MAX_IO_NAME_LEN);
    let cut = normalize_io_name(&words, "");
    assert!(cut.chars().count() <= MAX_IO_NAME_LEN);
    assert!(!cut.ends_with('_'));
    assert_eq!(normalize_io_name(&cut, ""), cut);
}

#[test]
fn names_redact_credentials_before_folding() {
    let name = normalize_io_name("api_key=sk-live-0123456789abcdef", "");
    assert!(!name.contains("sk-live"), "unexpected {name}");
    assert!(name.contains("redacted"), "unexpected {name}");
}

#[test]
fn vocabulary_build_records_counts_directions_and_types() {
    let vocabulary = IONameVocabulary::build(&[
        descriptor(
            "a",
            &[("Prompt", "str"), ("prompt", "text")],
            &[("Summary", "str")],
        ),
        descriptor("b", &[("Query", "string")], &[("Prompt", "text")]),
    ]);
    assert_eq!(vocabulary.term_names(), vec!["prompt", "query", "summary"]);
    let prompt = &vocabulary.terms[0];
    assert_eq!(prompt.count, 3);
    assert_eq!(prompt.directions, vec!["input", "output"]);
    assert_eq!(prompt.data_types, vec!["string", "text"]);
    assert!(prompt.aliases.is_empty());

    let summary = &vocabulary.terms[2];
    assert_eq!(summary.directions, vec!["output"]);
    assert_eq!(summary.count, 1);
    assert_eq!(vocabulary.version, "io-name-vocab-v1");
    assert_eq!(vocabulary.content_hash.len(), 64);
}

#[test]
fn vocabulary_content_hash_is_order_independent() {
    let first = descriptor("a", &[("prompt", "text")], &[("summary", "str")]);
    let second = descriptor("b", &[("query", "str")], &[("result", "text")]);
    let forward = IONameVocabulary::build(&[first.clone(), second.clone()]);
    let reverse = IONameVocabulary::build(&[second, first]);
    assert_eq!(forward, reverse);
    assert_eq!(forward.content_hash, reverse.content_hash);
    // Evidence order inside a term does not matter either.
    let mixed = IONameVocabulary::build(&[
        descriptor("c", &[("prompt", "str")], &[("summary", "text")]),
        descriptor("d", &[("prompt", "text")], &[("summary", "str")]),
    ]);
    assert_eq!(mixed.terms[0].data_types, vec!["string", "text"]);
    assert_ne!(mixed.content_hash, forward.content_hash);
    assert_eq!(IONameVocabulary::empty().resolve("prompt").normalized, None);
}

#[test]
fn vocabulary_round_trips_and_rejects_tampering_and_oversize() {
    let vocabulary = builtin_io_name_vocabulary();
    let payload = vocabulary.to_value();
    assert_eq!(IONameVocabulary::from_value(&payload).unwrap(), vocabulary);

    let mut tampered = payload.clone();
    tampered["version"] = json!("io-name-vocab-v2");
    let error = IONameVocabulary::from_value(&tampered).unwrap_err();
    assert!(error.to_string().contains("version"), "{error}");

    let mut tampered = payload.clone();
    tampered["content_hash"] = json!("0".repeat(64));
    let error = IONameVocabulary::from_value(&tampered).unwrap_err();
    assert!(error.to_string().contains("content_hash"), "{error}");

    // Editing a term without re-hashing is the same rejection.
    let mut tampered = payload.clone();
    tampered["terms"][0]["count"] = json!(999);
    let error = IONameVocabulary::from_value(&tampered).unwrap_err();
    assert!(error.to_string().contains("content_hash"), "{error}");

    let mut tampered = payload.clone();
    tampered["terms"] = json!("prompt");
    let error = IONameVocabulary::from_value(&tampered).unwrap_err();
    assert!(error.to_string().contains("terms"), "{error}");

    let mut tampered = payload.clone();
    tampered["terms"] = json!([3]);
    let error = IONameVocabulary::from_value(&tampered).unwrap_err();
    assert!(error.to_string().contains("terms[0]"), "{error}");

    let mut tampered = payload.clone();
    tampered["terms"] = json!([{"name": "   ", "count": 1}]);
    let error = IONameVocabulary::from_value(&tampered).unwrap_err();
    assert!(error.to_string().contains("terms[0].name"), "{error}");

    let mut tampered = payload.clone();
    tampered["terms"] = json!([{"name": "prompt", "count": -1}]);
    let error = IONameVocabulary::from_value(&tampered).unwrap_err();
    assert!(error.to_string().contains("terms[0].count"), "{error}");

    let mut tampered = payload.clone();
    tampered["terms"] = json!([{"name": "prompt", "count": true}]);
    let error = IONameVocabulary::from_value(&tampered).unwrap_err();
    assert!(error.to_string().contains("terms[0].count"), "{error}");

    let too_many_terms: Vec<Value> = (0..=MAX_VOCABULARY_TERMS)
        .map(|index| json!({"name": format!("term{index}"), "count": 1}))
        .collect();
    let error = IONameVocabulary::from_value(&json!({
        "version": "io-name-vocab-v1",
        "content_hash": "0".repeat(64),
        "terms": too_many_terms,
    }))
    .unwrap_err();
    assert!(error.to_string().contains("terms"), "{error}");

    let too_many_directions: Vec<Value> = (0..=MAX_TERM_VALUES).map(|_| json!("input")).collect();
    let error = IONameVocabulary::from_value(&json!({
        "version": "io-name-vocab-v1",
        "content_hash": "0".repeat(64),
        "terms": [{"name": "prompt", "count": 1, "directions": too_many_directions}],
    }))
    .unwrap_err();
    assert!(error.to_string().contains("directions"), "{error}");

    let too_many_aliases: Vec<Value> = (0..=MAX_VOCABULARY_ALIASES)
        .map(|index| json!(format!("alias{index}")))
        .collect();
    let error = IONameVocabulary::from_value(&json!({
        "version": "io-name-vocab-v1",
        "content_hash": "0".repeat(64),
        "terms": [{"name": "prompt", "count": 1, "aliases": too_many_aliases}],
    }))
    .unwrap_err();
    assert!(error.to_string().contains("aliases"), "{error}");

    let error = IONameVocabulary::from_value(&json!([])).unwrap_err();
    assert!(error.to_string().contains("io_name_vocabulary"), "{error}");
}

#[test]
fn vocabulary_learns_only_unambiguous_aliases() {
    let vocabulary = IONameVocabulary::build(&[descriptor(
        "a",
        &[("body", "text"), ("bodie", "text"), ("prompt", "text")],
        &[("summary", "text")],
    )]);
    let learned =
        vocabulary.with_observed_specs(&json!(["prompts", "bodies", "unknowns"]), &json!([]));

    // `prompts` folds to exactly one canonical term, so it becomes an alias.
    assert_eq!(
        learned.resolve("prompts").normalized.as_deref(),
        Some("prompt")
    );
    assert!(!learned.term_names().contains(&"prompts"));
    // `bodies` folds to both `body` and `bodie`, so the tie keeps it its own term.
    assert!(learned.term_names().contains(&"bodies"));
    assert_eq!(
        learned.resolve("bodies").normalized.as_deref(),
        Some("bodies")
    );
    // A token with no canonical relative becomes a term of its own.
    assert!(learned.term_names().contains(&"unknowns"));
    // The evidence merge is max-based, so re-learning never inflates counts.
    let relearned = learned.with_observed_specs(&json!(["prompts"]), &json!([]));
    assert_eq!(relearned, learned);
}

#[test]
fn builtin_vocabulary_resolves_generic_names() {
    let vocabulary = builtin_io_name_vocabulary();
    assert_eq!(vocabulary, builtin_io_name_vocabulary());
    assert_eq!(
        vocabulary.resolve("Query").normalized.as_deref(),
        Some("query")
    );
    assert_eq!(
        vocabulary.resolve(" result ").normalized.as_deref(),
        Some("result")
    );
    assert_eq!(vocabulary.resolve("nope").normalized, None);
    // Resolution is exact-or-recorded-alias: a spelling variant is only folded into a
    // term by `build`/`with_observed_specs`, never by `resolve` alone.
    assert_eq!(vocabulary.resolve("Results").normalized, None);
    assert!(!vocabulary.terms.is_empty());
}

#[test]
fn io_specs_from_a_list_accept_every_alias_key() {
    let vocabulary = builtin_io_name_vocabulary();
    let (items, issues, decisions) = normalize_io_specs_with_audit(
        &json!([
            {"id": "Query", "data_type": "str", "help": "the search query", "optional": "yes"},
            {"name": "limit", "datatype": "int", "required": "required"},
            {"name": "notes", "format": "text/markdown"},
            {"name": "preview", "mime_type": "image/png"},
        ]),
        &vocabulary,
        true,
    );

    assert_eq!(items[0].name, "query");
    assert_eq!(items[0].io_type, "string");
    assert_eq!(items[0].description, "the search query");
    assert_eq!(items[0].required, Some(false));

    assert_eq!(items[1].io_type, "integer");
    assert_eq!(items[1].required, Some(true));

    assert_eq!(items[2].io_type, "text_markdown");
    assert_eq!(
        items[2].metadata.get("raw_type"),
        Some(&json!("text/markdown"))
    );
    assert_eq!(items[3].io_type, "image_png");

    // The three soft coercions are audited, and the two unknown names are reported.
    assert!(decisions
        .iter()
        .any(|decision| decision.field == "required" && decision.method == "coerced"));
    assert_eq!(
        issue_codes(&issues)
            .iter()
            .filter(|code| **code == "io_name_not_in_vocabulary")
            .count(),
        3
    );
    assert_eq!(
        issue_codes(&issues)
            .iter()
            .filter(|code| **code == "unknown_data_type")
            .count(),
        2
    );
    assert!(issues.iter().all(|issue| issue.direction == "input"));
}

#[test]
fn io_specs_accept_single_mappings_and_name_keyed_mappings() {
    let vocabulary = builtin_io_name_vocabulary();
    let single = normalize_io_specs(&json!({"name": "Query", "type": "text"}), &vocabulary, true);
    assert_eq!(single.len(), 1);
    assert_eq!(single[0].name, "query");

    let keyed = normalize_io_specs(
        &json!({"Query": "text", "limit": {"type": "int", "required": true}}),
        &vocabulary,
        false,
    );
    assert_eq!(keyed.len(), 2);
    // Name-keyed mappings are visited in sorted key order (`Query` sorts before `limit`).
    assert_eq!(keyed[0].name, "query");
    assert_eq!(keyed[0].required, Some(false));
    assert_eq!(keyed[1].name, "limit");
    assert_eq!(keyed[1].io_type, "integer");
    assert_eq!(keyed[1].required, Some(true));

    // A list of plain strings uses the string as both name and type.
    let plain = normalize_io_specs(&json!(["Quantity", "price"]), &vocabulary, true);
    assert_eq!(plain.len(), 2);
    assert_eq!(plain[0].name, "quantity");
    assert_eq!(plain[0].io_type, "quantity");
}

#[test]
fn io_specs_honour_default_required_and_the_direction_default() {
    let vocabulary = builtin_io_name_vocabulary();
    let raw = json!([{"name": "query", "type": "text"}]);
    assert_eq!(
        normalize_io_specs(&raw, &vocabulary, true)[0].required,
        Some(true)
    );
    assert_eq!(
        normalize_io_specs(&raw, &vocabulary, false)[0].required,
        Some(false)
    );
    // An explicit `default` makes a declaration optional regardless of the fallback.
    let with_default = normalize_io_specs(
        &json!([{"name": "query", "type": "text", "default": "all"}]),
        &vocabulary,
        true,
    );
    assert_eq!(with_default[0].required, Some(false));
    assert_eq!(with_default[0].default, Some(json!("all")));

    // Outputs keep Python's direction defaults: `result` naming, no required flag.
    let outputs = normalize_io_specs_for_direction(
        &json!([{"type": "text"}]),
        &vocabulary,
        IoDirection::Output,
        IoDirection::Output.default_required(),
    );
    assert_eq!(outputs.items[0].name, "result");
    assert_eq!(outputs.items[0].required, None);
    assert!(outputs
        .issues
        .iter()
        .any(|issue| issue.code == "missing_io_name" && issue.direction == "output"));
}

#[test]
fn io_specs_sort_and_redact_metadata_with_audit() {
    let vocabulary = builtin_io_name_vocabulary();
    let (items, issues, decisions) = normalize_io_specs_with_audit(
        &json!([{
            "name": "query",
            "type": "text",
            "metadata": {"z": 1, "api_key": "sk-live-0123456789abcdef", "a": {"nested": "token: abcdefgh"}},
        }]),
        &vocabulary,
        true,
    );
    let metadata = &items[0].metadata;
    let keys: Vec<&str> = metadata.keys().map(String::as_str).collect();
    assert_eq!(keys, vec!["a", "api_key", "z"]);
    assert_eq!(metadata["api_key"], json!(REDACTED));
    assert_eq!(metadata["a"], json!({"nested": "token=<redacted>"}));
    assert_eq!(metadata["z"], json!(1));

    let audit = issues
        .iter()
        .find(|issue| issue.code == "sensitive_metadata_redacted")
        .expect("metadata redaction must be audited");
    assert_eq!(audit.details.get("key"), Some(&json!("api_key")));
    assert!(decisions
        .iter()
        .any(|decision| decision.field == "metadata" && decision.method == "redacted"));
}

#[test]
fn io_specs_redact_sensitive_names_and_their_defaults() {
    let vocabulary = builtin_io_name_vocabulary();
    let (items, issues, decisions) = normalize_io_specs_with_audit(
        &json!([{"name": "api_key", "type": "text", "default": "hunter2"}]),
        &vocabulary,
        true,
    );
    assert_eq!(items[0].name, REDACTED);
    assert_eq!(items[0].default, None);
    assert_eq!(
        items[0].metadata.get("default_redacted"),
        Some(&json!(true))
    );
    assert!(issue_codes(&issues).contains(&"sensitive_io_name_redacted"));
    assert!(issue_codes(&issues).contains(&"sensitive_default_redacted"));
    let redaction = decisions
        .iter()
        .find(|decision| decision.field == "name" && decision.method == "redacted")
        .expect("the redacted name must be audited");
    assert_eq!(redaction.token, REDACTED);
    assert_eq!(redaction.normalized_value, REDACTED);
    // The credential never survives anywhere in the normalized card.
    assert!(!serde_json::to_string(&items).unwrap().contains("hunter2"));
}

#[test]
fn io_specs_report_every_dropped_coerced_or_defaulted_field() {
    let vocabulary = builtin_io_name_vocabulary();
    let (items, issues) = normalize_io_specs_with_issues(
        &json!([
            {"type": "text"},
            {"name": "thing", "type": "banana"},
            {"name": "query", "type": "text", "required": "maybe"},
            7
        ]),
        &vocabulary,
        true,
    );
    assert_eq!(items.len(), 3);
    assert_eq!(items[0].name, "input");
    assert_eq!(items[1].name, "thing");
    assert_eq!(items[1].io_type, "banana");
    assert_eq!(items[2].name, "query");
    assert_eq!(items[2].required, Some(true));
    assert_eq!(
        issues
            .iter()
            .map(|issue| (issue.code.as_str(), issue.index))
            .collect::<Vec<_>>(),
        vec![
            ("missing_io_name", Some(0)),
            ("io_name_not_in_vocabulary", Some(1)),
            ("unknown_data_type", Some(1)),
            ("invalid_required", Some(2)),
            ("invalid_io_entry", Some(3)),
        ]
    );

    // A value that is not a collection has nothing to normalize and says so.
    let (items, issues) = normalize_io_specs_with_issues(&json!(7), &vocabulary, true);
    assert!(items.is_empty());
    assert_eq!(issue_codes(&issues), vec!["invalid_io_collection"]);
    assert_eq!(issues[0].direction, "input");
}

#[test]
fn io_specs_deduplicate_repeated_names() {
    let vocabulary = builtin_io_name_vocabulary();
    let (items, issues, decisions) = normalize_io_specs_with_audit(
        &json!([
            {"name": "query", "type": "text"},
            {"name": "Query", "type": "text"},
            {"name": "QUERY", "type": "text"}
        ]),
        &vocabulary,
        true,
    );
    assert_eq!(
        items
            .iter()
            .map(|item| item.name.as_str())
            .collect::<Vec<_>>(),
        vec!["query", "query_2", "query_3"]
    );
    assert_eq!(
        issue_codes(&issues)
            .iter()
            .filter(|code| **code == "duplicate_io_name")
            .count(),
        2
    );
    assert_eq!(
        decisions
            .iter()
            .filter(|decision| decision.method == "deduplicate")
            .count(),
        2
    );
}

#[test]
fn io_specs_without_a_corpus_use_the_syntax_path() {
    let empty = IONameVocabulary::empty();
    let (items, issues, decisions) = normalize_io_specs_with_audit(
        &json!([{"name": "Free Form Name", "type": "text"}]),
        &empty,
        true,
    );
    assert_eq!(items[0].name, "free_form_name");
    // No corpus means no "not in vocabulary" noise; a blank name is still reported.
    assert!(issues.is_empty(), "unexpected {issues:?}");
    let name_decision = decisions
        .iter()
        .find(|decision| decision.field == "name")
        .expect("a name decision is always recorded");
    assert_eq!(name_decision.method, "syntax");
    assert!(name_decision.details.is_empty());
    assert_eq!(name_decision.vocabulary_version, "io-name-vocab-v1");
}

#[test]
fn data_type_resolutions_follow_the_embedded_vocabulary() {
    assert_eq!(resolve_data_type(" text ").normalized, "text");
    assert_eq!(resolve_data_type(" text ").source, DataTypeSource::Exact);
    assert_eq!(resolve_data_type("BYTES").normalized, "bytes");
    assert_eq!(resolve_data_type("bytes").source, DataTypeSource::Exact);
    assert_eq!(resolve_data_type("str").normalized, "string");
    assert_eq!(resolve_data_type("str").source, DataTypeSource::Alias);
    assert_eq!(resolve_data_type("float").normalized, "number");
    assert_eq!(resolve_data_type("dict").normalized, "object");
    assert_eq!(resolve_data_type("").normalized, "any");
    assert_eq!(resolve_data_type("").source, DataTypeSource::Empty);
    assert!(resolve_data_type("").raw.is_empty());
    let unknown = resolve_data_type("image/png");
    assert_eq!(unknown.raw, "image/png");
    assert_eq!(unknown.normalized, "image_png");
    assert_eq!(unknown.source, DataTypeSource::Unknown);
    assert!(!unknown.known());

    // The audit records the same resolution.
    let vocabulary = builtin_io_name_vocabulary();
    let (_, _, decisions) = normalize_io_specs_with_audit(
        &json!([{"name": "query", "type": {"type": "string", "format": "date-time"}}]),
        &vocabulary,
        true,
    );
    let type_decision = decisions
        .iter()
        .find(|decision| decision.field == "type")
        .expect("a type decision is always recorded");
    assert_eq!(type_decision.token, "date_time");
    assert_eq!(type_decision.vocabulary, "data_type_vocab");
    assert_eq!(type_decision.vocabulary_version, "data-type-v1");
}

#[test]
fn sanitize_produces_a_bounded_json_safe_replay_state() {
    assert_eq!(sanitize_io_specs(&json!("Quantity")), json!(["Quantity"]));
    assert_eq!(sanitize_io_specs(&json!(null)), json!([]));
    assert_eq!(
        sanitize_io_specs(&json!({"name": "q", "type": "text", "extra": 1})),
        json!([{"name": "q", "type": "text"}])
    );
    assert_eq!(
        sanitize_io_specs(&json!({"alpha": "text", "beta": {"type": "int"}})),
        json!([{"name": "alpha", "type": "text"}, {"name": "beta", "type": "int"}])
    );
    assert_eq!(
        sanitize_io_specs(&json!([{"name": "q"}, 3, null])),
        json!([{"name": "q"}, 0, 0])
    );
    assert_eq!(
        sanitize_io_specs(&json!([{"name": "q", "metadata": {"api_key": "abc"}}])),
        json!([{"name": "q", "metadata": {"api_key": REDACTED}}])
    );
    assert_eq!(sanitize_io_specs(&json!(true)), json!(0));

    let mut deep = json!({"name": "q", "metadata": {"leaf": 1}});
    for _ in 0..70 {
        deep = json!({"metadata": deep});
    }
    let safe = sanitize_io_specs(&deep);
    assert!(nesting(&safe) <= 68, "unexpected depth {}", nesting(&safe));
    let mut cursor = &safe[0];
    while let Some(next) = cursor.get("metadata") {
        cursor = next;
    }
    assert!(
        cursor.is_null() || cursor.as_str() == Some(REDACTED),
        "unexpected tail {cursor}"
    );
}

#[test]
fn normalization_is_deterministic_and_audit_complete() {
    let vocabulary = builtin_io_name_vocabulary();
    let raw = json!([
        {"name": "Query", "type": "str", "optional": "yes", "metadata": {"b": 1, "a": 2}},
        {"name": "", "type": "banana"},
        {"name": "api_key", "default": "hunter2"},
        "Quantity"
    ]);

    let (items, issues, decisions) = normalize_io_specs_with_audit(&raw, &vocabulary, true);
    let repeat = normalize_io_specs_with_audit(&raw, &vocabulary, true);
    assert_eq!(
        serde_json::to_string(&items).unwrap(),
        serde_json::to_string(&repeat.0).unwrap()
    );
    assert_eq!(
        serde_json::to_string(&issues).unwrap(),
        serde_json::to_string(&repeat.1).unwrap()
    );
    assert_eq!(
        serde_json::to_string(&decisions).unwrap(),
        serde_json::to_string(&repeat.2).unwrap()
    );

    // Every declaration produces a name and a type decision, plus a required decision;
    // the sensitive name adds a second (redaction) name decision.
    assert_eq!(decisions.iter().filter(|d| d.field == "name").count(), 5);
    assert_eq!(decisions.iter().filter(|d| d.field == "type").count(), 4);
    assert_eq!(
        decisions.iter().filter(|d| d.field == "required").count(),
        4
    );
    assert!(decisions
        .iter()
        .all(|decision| decision.index.is_some() && !decision.method.is_empty()));

    // Metadata order in the input never reaches the output.
    let reordered = normalize_io_specs(
        &json!([{"name": "query", "metadata": {"a": 2, "b": 1}}]),
        &vocabulary,
        true,
    );
    let original = normalize_io_specs(
        &json!([{"name": "query", "metadata": {"b": 1, "a": 2}}]),
        &vocabulary,
        true,
    );
    assert_eq!(reordered, original);
}
