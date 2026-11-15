//! The matcher request/response protocol: request shape, reply expansion, validation.
//!
//! Every test is offline and deterministic: no model is called, timestamps never appear,
//! and the only ordering used is the payload order or a sorted collection.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]

use openjiuwen_symphony::graph::CAN_FEED;
use openjiuwen_symphony::llm::ontology::{OntologyMatch, METHOD_LLM};
use openjiuwen_symphony::model::{CapabilityDescriptor, CapabilityIo};
use openjiuwen_symphony::orchestration::matcher::{
    build_match_request, diagnostics_by_candidate, matches_by_candidate, parse_match_response,
    parse_match_response_with, parse_match_response_with_evidence, CandidateEvidence,
    GraphDiagnostic, MatchThresholds, SupportingFields, DIAGNOSTIC_DUPLICATE_CANDIDATE_ID,
    DIAGNOSTIC_INVALID_CANDIDATE_DIRECTION, DIAGNOSTIC_INVALID_MATCHES_PAYLOAD,
    DIAGNOSTIC_INVALID_MATCH_ITEM, DIAGNOSTIC_LOW_CONFIDENCE_LLM_MATCH,
    DIAGNOSTIC_REJECTED_LLM_MATCH, DIAGNOSTIC_UNKNOWN_CANDIDATE_ID,
    DIAGNOSTIC_UNSUPPORTED_CANDIDATE_DIRECTION, MAX_PORT_DESCRIPTION_LENGTH, MAX_REASON_LENGTH,
    MAX_SKILL_DESCRIPTION_LENGTH,
};
use openjiuwen_symphony::orchestration::{MatchStrategy, RelationCandidate};
use serde_json::{json, Value};

/// An IO with a name, a type and a description.
fn io(name: &str, io_type: &str, description: &str) -> CapabilityIo {
    let mut parameter = CapabilityIo::new(name, io_type);
    parameter.description = description.to_string();
    parameter
}

/// One capability card.
fn capability(
    id: &str,
    name: &str,
    inputs: Vec<CapabilityIo>,
    outputs: Vec<CapabilityIo>,
) -> CapabilityDescriptor {
    CapabilityDescriptor::new("tool", id, name)
        .with_description(format!("{name} capability"))
        .with_inputs(inputs)
        .with_outputs(outputs)
}

/// The fixture catalog: `extract` produces `text` and consumes `notes`; `summarise` does
/// the opposite, so both candidate directions have evidence.
fn catalog() -> Vec<CapabilityDescriptor> {
    vec![
        capability(
            "extract",
            "Extract",
            vec![io("notes", "string", "Notes to fold in")],
            vec![io("text", "string", "Extracted text")],
        ),
        capability(
            "summarise",
            "Summarise",
            vec![io("text", "string", "Text to summarise")],
            vec![io("notes", "string", "Notes produced")],
        ),
    ]
}

/// The `extract -> summarise` candidate with its matched `text`/`text` pair.
fn candidate() -> RelationCandidate {
    RelationCandidate {
        source: "tool:extract".to_string(),
        target: "tool:summarise".to_string(),
        relation: CAN_FEED.to_string(),
        confidence: 1.0,
        strategy: MatchStrategy::ExactIo,
        matched_input: Some("text".to_string()),
        matched_output: Some("text".to_string()),
    }
}

/// A second candidate (`summarise -> extract`) with its matched `notes`/`notes` pair.
fn reverse_candidate() -> RelationCandidate {
    RelationCandidate {
        source: "tool:summarise".to_string(),
        target: "tool:extract".to_string(),
        relation: CAN_FEED.to_string(),
        confidence: 1.0,
        strategy: MatchStrategy::ExactIo,
        matched_input: Some("notes".to_string()),
        matched_output: Some("notes".to_string()),
    }
}

/// The evidence table of a request built over `candidates`.
fn evidence_of(
    capabilities: &[CapabilityDescriptor],
    candidates: &[RelationCandidate],
) -> CandidateEvidence {
    build_match_request(capabilities, candidates, false).evidence()
}

#[test]
fn request_emits_the_compact_document() {
    let capabilities = catalog();
    let request = build_match_request(&capabilities, &[candidate()], false);
    assert_eq!(
        request.to_value(),
        json!({"candidates": [{
            "id": "c1",
            "source": {"name": "Extract", "description": "Extract capability"},
            "target": {"name": "Summarise", "description": "Summarise capability"},
            "directions": {
                "forward": {
                    "outputs": [{"name": "text", "type": "string", "description": "Extracted text"}],
                    "inputs": [{"name": "text", "type": "string", "description": "Text to summarise"}],
                    "ports": [{
                        "output": "text",
                        "output_type": "string",
                        "input": "text",
                        "input_type": "string"
                    }]
                },
                "reverse": {
                    "outputs": [{"name": "notes", "type": "string", "description": "Notes produced"}],
                    "inputs": [{"name": "notes", "type": "string", "description": "Notes to fold in"}],
                    "ports": [{
                        "output": "notes",
                        "output_type": "string",
                        "input": "notes",
                        "input_type": "string"
                    }]
                }
            }
        }]})
    );
    // The derived value and the derived serializer agree, so neither can drift.
    assert_eq!(serde_json::to_value(&request).unwrap(), request.to_value());
}

#[test]
fn positional_ids_follow_the_input_order_and_reverse() {
    let capabilities = catalog();
    let candidates = [candidate(), reverse_candidate()];
    let forward = build_match_request(&capabilities, &candidates, false);
    let ids: Vec<&str> = forward
        .candidates()
        .iter()
        .map(|context| context.id.as_str())
        .collect();
    assert_eq!(ids, ["c1", "c2"]);
    assert_eq!(
        forward.candidates()[0].candidate_key,
        "tool:extract->tool:summarise"
    );

    let reversed = build_match_request(&capabilities, &candidates, true);
    assert_eq!(
        reversed.candidates()[0].id,
        "c1",
        "the last input candidate becomes c1"
    );
    assert_eq!(
        reversed.candidates()[0].candidate_key,
        "tool:summarise->tool:extract"
    );
    assert_eq!(reversed.candidates()[1].id, "c2");
    // Parsing a reversed request against the reversed order round-trips the ids.
    let outcome = parse_match_response_with_evidence(
        &json!({"matches": [{"id": "c1", "confidence": 0.95}]}),
        &candidates.iter().rev().cloned().collect::<Vec<_>>(),
        &reversed.evidence(),
        &MatchThresholds::default(),
    );
    assert_eq!(outcome.matches.len(), 1);
    assert_eq!(outcome.matches[0].source_id, "tool:summarise");
    assert_eq!(outcome.matches[0].target_id, "tool:extract");
}

#[test]
fn descriptions_truncate_on_character_boundaries() {
    // 300 CJK characters: a byte-boundary slice would panic, a char slice keeps 240 chars.
    let long_description = "能力".repeat(150);
    let capabilities = vec![
        CapabilityDescriptor::new("tool", "source", "Source")
            .with_description(long_description.clone())
            .with_outputs(vec![io("值", "string", &long_description)]),
        capability("target", "Target", vec![io("值", "string", "")], Vec::new()),
    ];
    let request = build_match_request(&capabilities, &[single_candidate()], false);
    let value = request.to_value();
    let entry = &value["candidates"][0];
    let source = entry["source"]["description"].as_str().unwrap();
    assert_eq!(source.chars().count(), MAX_SKILL_DESCRIPTION_LENGTH);
    assert!(source
        .chars()
        .all(|character| character == '能' || character == '力'));

    let port = &entry["directions"]["forward"]["outputs"][0]["description"];
    let port = port.as_str().unwrap();
    assert_eq!(port.chars().count(), MAX_PORT_DESCRIPTION_LENGTH);
    assert_eq!(port.chars().count() * 3, port.len(), "no split code point");
}

/// The `source -> target` candidate over the truncation fixture.
fn single_candidate() -> RelationCandidate {
    RelationCandidate {
        source: "tool:source".to_string(),
        target: "tool:target".to_string(),
        relation: CAN_FEED.to_string(),
        confidence: 1.0,
        strategy: MatchStrategy::ExactIo,
        matched_input: Some("值".to_string()),
        matched_output: Some("值".to_string()),
    }
}

#[test]
fn a_direction_without_a_pair_is_omitted() {
    // `target` has no inputs and `source` has no inputs, so neither orientation is
    // supported by the data-type vocabulary.
    let capabilities = vec![
        capability(
            "source",
            "Source",
            Vec::new(),
            vec![io("blob", "object", "")],
        ),
        capability(
            "target",
            "Target",
            Vec::new(),
            vec![io("blob", "object", "")],
        ),
    ];
    let request = build_match_request(
        &capabilities,
        &[RelationCandidate {
            source: "tool:source".to_string(),
            target: "tool:target".to_string(),
            ..RelationCandidate::default()
        }],
        false,
    );
    let entry = &request.to_value()["candidates"][0];
    assert!(entry["directions"]
        .as_object()
        .unwrap_or(&serde_json::Map::new())
        .is_empty());
    // Without evidence the compact reply cannot be answered at all.
    let outcome = parse_match_response(
        &json!({"matches": [{"id": "c1", "confidence": 0.9}]}),
        &[RelationCandidate {
            source: "tool:source".to_string(),
            target: "tool:target".to_string(),
            ..RelationCandidate::default()
        }],
    );
    assert!(outcome.matches.is_empty());
    assert_eq!(
        outcome.diagnostics[0].code,
        DIAGNOSTIC_UNSUPPORTED_CANDIDATE_DIRECTION
    );
}

#[test]
fn a_candidate_with_an_unknown_capability_is_skipped() {
    let capabilities = vec![capability(
        "extract",
        "Extract",
        Vec::new(),
        vec![io("text", "string", "")],
    )];
    let request = build_match_request(&capabilities, &[candidate()], false);
    assert!(request.candidates().is_empty());
    assert_eq!(request.to_value(), json!({"candidates": []}));
}

#[test]
fn compact_replies_expand_in_both_directions() {
    let capabilities = catalog();
    let candidates = [candidate()];
    let evidence = evidence_of(&capabilities, &candidates);

    let forward = parse_match_response_with_evidence(
        &json!({"matches": [{"id": "c1", "direction": "forward", "confidence": 0.9,
                             "reason": "  both produce text  "}]}),
        &candidates,
        &evidence,
        &MatchThresholds::default(),
    );
    assert!(forward.diagnostics.is_empty());
    let item = &forward.matches[0];
    assert_eq!(
        item.candidate_id.as_deref(),
        Some("tool:extract->tool:summarise")
    );
    assert_eq!(item.source_id, "tool:extract");
    assert_eq!(item.target_id, "tool:summarise");
    assert_eq!(item.relation, CAN_FEED);
    assert_eq!(item.method, METHOD_LLM);
    assert_eq!(item.reasons, ["both produce text"]);
    assert!(item.accepted);
    assert_eq!(item.supporting_fields.source_outputs, ["text".to_string()]);
    assert_eq!(item.supporting_fields.target_inputs, ["text".to_string()]);
    assert_eq!(item.supporting_fields.port_mappings.len(), 1);

    let reverse = parse_match_response_with_evidence(
        &json!({"matches": [{"id": "c1", "direction": " REVERSE ", "confidence": 0.8}]}),
        &candidates,
        &evidence,
        &MatchThresholds::default(),
    );
    let swapped = &reverse.matches[0];
    assert_eq!(swapped.source_id, "tool:summarise", "reverse swaps the ids");
    assert_eq!(swapped.target_id, "tool:extract");
    assert_eq!(
        swapped.supporting_fields.source_outputs,
        ["notes".to_string()]
    );
    assert!(swapped.accepted);

    // Without the evidence table only the candidate's matched pair is known, so the
    // reverse direction is unsupported rather than guessed.
    let without = parse_match_response(
        &json!({"matches": [{"id": "c1", "direction": "reverse", "confidence": 0.8}]}),
        &candidates,
    );
    assert!(without.matches.is_empty());
    assert_eq!(
        without.diagnostics[0].code,
        DIAGNOSTIC_UNSUPPORTED_CANDIDATE_DIRECTION
    );
}

#[test]
fn explicit_rejection_forces_the_confidence_to_zero() {
    for flag in [
        json!(false),
        json!(0),
        json!("0"),
        json!("false"),
        json!("no"),
        json!(""),
    ] {
        let payload = json!({"matches": [{"id": "c1", "confidence": 0.99, "accepted": flag}]});
        let outcome = parse_match_response(&payload, &[candidate()]);
        assert_eq!(outcome.matches.len(), 1, "the rejection flag is {flag}");
        assert!((outcome.matches[0].confidence - 0.0).abs() < f64::EPSILON);
        assert!(!outcome.matches[0].accepted);
        assert_eq!(
            outcome.diagnostics[0].code,
            DIAGNOSTIC_LOW_CONFIDENCE_LLM_MATCH
        );
    }
    // A real acceptance leaves the confidence alone.
    let outcome = parse_match_response(
        &json!({"matches": [{"id": "c1", "confidence": 0.9, "accepted": "yes"}]}),
        &[candidate()],
    );
    assert!((outcome.matches[0].confidence - 0.9).abs() < f64::EPSILON);
    assert!(outcome.matches[0].accepted);
}

#[test]
fn malformed_compact_items_produce_protocol_diagnostics() {
    let candidates = [candidate()];

    let not_an_object = parse_match_response(&json!({"matches": ["c1"]}), &candidates);
    assert!(not_an_object.matches.is_empty());
    assert_eq!(
        not_an_object.diagnostics[0].code,
        DIAGNOSTIC_INVALID_MATCH_ITEM
    );

    let unknown = parse_match_response(&json!({"matches": [{"id": "c9"}]}), &candidates);
    assert!(unknown.matches.is_empty());
    assert_eq!(unknown.diagnostics[0].code, DIAGNOSTIC_UNKNOWN_CANDIDATE_ID);
    assert_eq!(
        unknown.diagnostics[0].candidate_id(),
        Some("c9"),
        "the named id is recorded for attribution"
    );

    let duplicate = parse_match_response(
        &json!({"matches": [
            {"id": "c1", "confidence": 0.9},
            {"id": "c1", "confidence": 0.1}
        ]}),
        &candidates,
    );
    assert_eq!(duplicate.matches.len(), 1, "the first answer wins");
    assert!((duplicate.matches[0].confidence - 0.9).abs() < f64::EPSILON);
    assert_eq!(
        duplicate.diagnostics[0].code,
        DIAGNOSTIC_DUPLICATE_CANDIDATE_ID
    );

    let bad_direction = parse_match_response(
        &json!({"matches": [{"id": "c1", "direction": "sideways", "confidence": 0.9}]}),
        &candidates,
    );
    assert!(bad_direction.matches.is_empty());
    assert_eq!(
        bad_direction.diagnostics[0].code,
        DIAGNOSTIC_INVALID_CANDIDATE_DIRECTION
    );

    let not_a_list = parse_match_response(&json!({"matches": {"id": "c1"}}), &candidates);
    assert!(not_a_list.matches.is_empty());
    assert_eq!(
        not_a_list.diagnostics[0].code,
        DIAGNOSTIC_INVALID_MATCHES_PAYLOAD
    );

    let missing = parse_match_response(&json!({}), &candidates);
    assert!(missing.matches.is_empty());
    assert!(missing.diagnostics.is_empty());
}

#[test]
fn validation_rejects_unknown_endpoints_and_bad_confidences() {
    let candidates = [candidate()];

    let unknown_endpoint = parse_match_response(
        &json!({"matches": [{
            "source_id": "tool:extract",
            "target_id": "tool:ghost",
            "relation_type": "can_feed",
            "confidence": 0.9
        }]}),
        &candidates,
    );
    let item = &unknown_endpoint.matches[0];
    assert!(!item.accepted);
    assert!(item
        .errors
        .contains(&"target_id does not exist".to_string()));
    assert!(item
        .errors
        .contains(&"match does not correspond to an input candidate".to_string()));
    assert_eq!(
        unknown_endpoint.diagnostics[0].code,
        DIAGNOSTIC_REJECTED_LLM_MATCH
    );

    let unparseable = parse_match_response(
        &json!({"matches": [{"id": "c1", "confidence": "high"}]}),
        &candidates,
    );
    assert!(unparseable.matches[0]
        .errors
        .contains(&"confidence is not numeric".to_string()));
    assert!((unparseable.matches[0].confidence - 0.0).abs() < f64::EPSILON);

    let out_of_range = parse_match_response(
        &json!({"matches": [{"id": "c1", "confidence": 1.4}]}),
        &candidates,
    );
    let item = &out_of_range.matches[0];
    assert!(item
        .errors
        .contains(&"confidence must be between 0 and 1".to_string()));
    assert!(
        (item.confidence - 1.0).abs() < f64::EPSILON,
        "the clamped value is kept"
    );
    assert!(!item.accepted);

    let negative = parse_match_response(
        &json!({"matches": [{"id": "c1", "confidence": -0.5}]}),
        &candidates,
    );
    assert!((negative.matches[0].confidence - 0.0).abs() < f64::EPSILON);

    // A compact item's relation is always `can_feed`; a full-form item may not claim
    // another relation.
    let compact_relation = parse_match_response(
        &json!({"matches": [{"id": "c1", "confidence": 0.9, "relation_type": "overrides"}]}),
        &candidates,
    );
    assert_eq!(compact_relation.matches[0].relation, CAN_FEED);
    assert!(compact_relation.matches[0].errors.is_empty());

    let wrong_relation = parse_match_response(
        &json!({"matches": [{
            "source_id": "tool:extract",
            "target_id": "tool:summarise",
            "confidence": 0.9,
            "relation_type": "overrides"
        }]}),
        &candidates,
    );
    assert_eq!(wrong_relation.matches[0].relation, CAN_FEED);
    assert!(wrong_relation.matches[0]
        .errors
        .contains(&"relation_type is not allowed".to_string()));
}

#[test]
fn thresholds_decide_acceptance_and_report_low_confidence() {
    let candidates = [candidate()];
    let payload = json!({"matches": [{"id": "c1", "confidence": 0.69}]});
    let outcome = parse_match_response(&payload, &candidates);
    assert_eq!(outcome.matches.len(), 1);
    assert!(!outcome.matches[0].accepted);
    assert_eq!(
        outcome.diagnostics[0].code,
        DIAGNOSTIC_LOW_CONFIDENCE_LLM_MATCH
    );
    assert_eq!(
        outcome.diagnostics[0].detail.as_ref().unwrap()["threshold"],
        json!(0.7)
    );

    let accepted = parse_match_response(
        &json!({"matches": [{"id": "c1", "confidence": 0.7}]}),
        &candidates,
    );
    assert!(accepted.matches[0].accepted, "the threshold is inclusive");
    assert!(accepted.diagnostics.is_empty());

    let strict = parse_match_response_with(
        &json!({"matches": [{"id": "c1", "confidence": 0.8}]}),
        &candidates,
        &MatchThresholds::new(0.9),
    );
    assert!(!strict.matches[0].accepted);
    assert_eq!(MatchThresholds::default().threshold(CAN_FEED), 0.7);
    assert_eq!(MatchThresholds::default().threshold("overrides"), 1.0);
}

#[test]
fn reasons_are_trimmed_and_truncated() {
    let candidates = [candidate()];
    let long = format!("  {}  ", "原".repeat(MAX_REASON_LENGTH + 40));
    let outcome = parse_match_response(
        &json!({"matches": [{"id": "c1", "confidence": 0.9, "reason": long}]}),
        &candidates,
    );
    let reason = &outcome.matches[0].reasons[0];
    assert_eq!(reason.chars().count(), MAX_REASON_LENGTH);
    assert!(!reason.starts_with(' '), "the reason is trimmed first");

    let blank = parse_match_response(
        &json!({"matches": [{"id": "c1", "confidence": 0.9, "reason": "   "}]}),
        &candidates,
    );
    assert!(blank.matches[0].reasons.is_empty());
}

#[test]
fn supporting_fields_complete_from_the_requested_port_mappings() {
    let candidates = [candidate()];
    let payload = json!({"matches": [{
        "source_id": "tool:extract",
        "target_id": "tool:summarise",
        "relation_type": "can_feed",
        "confidence": 0.9,
        "supporting_fields": {"port_mappings": [{"source_output": "text", "target_input": "text"}]}
    }]});
    let outcome = parse_match_response(&payload, &candidates);
    let item = &outcome.matches[0];
    assert!(item.accepted, "errors: {:?}", item.errors);
    assert_eq!(item.supporting_fields.source_outputs, ["text".to_string()]);
    assert_eq!(item.supporting_fields.target_inputs, ["text".to_string()]);

    // A wiring the candidate never proposed fails the subset check. The compact path
    // derives supporting_fields from the evidence, so this needs a full-form item.
    let unrelated = json!({"matches": [{
        "source_id": "tool:extract",
        "target_id": "tool:summarise",
        "confidence": 0.9,
        "supporting_fields": {"port_mappings": [{"source_output": "blob", "target_input": "text"}]}
    }]});
    let outcome = parse_match_response(&unrelated, &candidates);
    assert!(outcome.matches[0]
        .errors
        .contains(&"port_mappings do not match candidate evidence".to_string()));

    // Named fields that do not intersect the evidence fail the intersection check.
    let drifted = json!({"matches": [{
        "source_id": "tool:extract",
        "target_id": "tool:summarise",
        "supporting_fields": {"source_outputs": ["blob"], "target_inputs": ["text"]}
    }]});
    let outcome = parse_match_response(&drifted, &candidates);
    assert!(outcome.matches[0]
        .errors
        .contains(&"supporting_fields do not match candidate evidence".to_string()));

    // Supporting-field names accept Python's `"name: type"` / `"name(type)"` spellings.
    let supporting = SupportingFields::from_value(&json!({
        "source_outputs": ["text: string", {"name": "text"}],
        "target_inputs": ["text(string)"]
    }));
    assert_eq!(supporting.source_outputs, ["text".to_string()]);
    assert_eq!(supporting.target_inputs, ["text".to_string()]);
}

#[test]
fn supporting_fields_project_onto_ontology_matches() {
    let candidates = [candidate()];
    let outcome = parse_match_response(
        &json!({"matches": [{"id": "c1", "confidence": 0.85, "reason": "ok"}]}),
        &candidates,
    );
    let projected: OntologyMatch = outcome.matches[0].to_ontology_match();
    assert_eq!(projected.source_id, "tool:extract");
    assert_eq!(projected.target_id, "tool:summarise");
    assert_eq!(projected.relation, CAN_FEED);
    assert!((projected.confidence - 0.85).abs() < f64::EPSILON);
    assert_eq!(projected.method, METHOD_LLM);
    assert_eq!(projected.reasons, ["ok"]);
    assert!(projected.accepted);
    assert_eq!(outcome.ontology_matches(), vec![projected]);
    assert_eq!(outcome.accepted_matches().len(), 1);
}

#[test]
fn grouping_helpers_attribute_matches_and_diagnostics() {
    let candidates = [candidate()];
    let outcome = parse_match_response(
        &json!({"matches": [
            {"id": "c1", "confidence": 0.4},
            {"id": "c7"}
        ]}),
        &candidates,
    );
    let grouped = matches_by_candidate(&outcome.matches);
    assert_eq!(
        grouped.keys().cloned().collect::<Vec<_>>(),
        ["tool:extract->tool:summarise"],
        "a sorted, deterministic key order"
    );
    assert_eq!(grouped["tool:extract->tool:summarise"].len(), 1);

    // A match without a candidate key falls back to its endpoint pair.
    let orphan = openjiuwen_symphony::orchestration::matcher::ValidatedMatch {
        source_id: "tool:a".to_string(),
        target_id: "tool:b".to_string(),
        ..Default::default()
    };
    let orphans = [orphan];
    let fallback = matches_by_candidate(&orphans);
    assert!(fallback.contains_key("tool:a->tool:b"));

    let by_candidate = diagnostics_by_candidate(&outcome.diagnostics);
    assert_eq!(
        by_candidate.keys().cloned().collect::<Vec<_>>(),
        ["c7", "tool:extract->tool:summarise"]
    );
    assert_eq!(
        by_candidate["tool:extract->tool:summarise"][0].code,
        DIAGNOSTIC_LOW_CONFIDENCE_LLM_MATCH
    );

    // A diagnostic without a candidate id is dropped from the grouping, never lost.
    let anonymous = GraphDiagnostic::new(DIAGNOSTIC_INVALID_MATCH_ITEM, "not an object");
    assert!(diagnostics_by_candidate(&[anonymous]).is_empty());
}

#[test]
fn parsing_the_same_payload_twice_is_byte_identical() {
    let capabilities = catalog();
    let candidates = [candidate(), reverse_candidate()];
    let evidence = evidence_of(&capabilities, &candidates);
    let payload: Value = json!({"matches": [
        {"id": "c1", "direction": "sideways"},
        {"id": "c2", "confidence": 0.4, "reason": " notes align "},
        {"id": "c2", "confidence": 0.9},
        {"id": "c7"},
        "not an object",
        {
            "source_id": "tool:extract",
            "target_id": "tool:summarise",
            "relation_type": "overrides",
            "confidence": 1.5
        }
    ]});
    let first = serde_json::to_string(&parse_match_response_with_evidence(
        &payload,
        &candidates,
        &evidence,
        &MatchThresholds::default(),
    ))
    .unwrap();
    let second = serde_json::to_string(&parse_match_response_with_evidence(
        &payload,
        &candidates,
        &evidence,
        &MatchThresholds::default(),
    ))
    .unwrap();
    assert_eq!(first, second);

    let outcome = parse_match_response_with_evidence(
        &payload,
        &candidates,
        &evidence,
        &MatchThresholds::default(),
    );
    let codes: Vec<&str> = outcome
        .diagnostics
        .iter()
        .map(|diagnostic| diagnostic.code.as_str())
        .collect();
    assert_eq!(
        codes,
        [
            DIAGNOSTIC_INVALID_CANDIDATE_DIRECTION,
            DIAGNOSTIC_DUPLICATE_CANDIDATE_ID,
            DIAGNOSTIC_UNKNOWN_CANDIDATE_ID,
            DIAGNOSTIC_INVALID_MATCH_ITEM,
            DIAGNOSTIC_LOW_CONFIDENCE_LLM_MATCH,
            DIAGNOSTIC_REJECTED_LLM_MATCH,
        ],
        "protocol diagnostics first, then validation diagnostics, both in payload order"
    );
    assert_eq!(outcome.matches.len(), 2);
    assert_eq!(outcome.matches[0].reasons, ["notes align"]);
    assert!((outcome.matches[1].confidence - 1.0).abs() < f64::EPSILON);
}
