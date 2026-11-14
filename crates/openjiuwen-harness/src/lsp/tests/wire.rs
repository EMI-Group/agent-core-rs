//! Unit tests for the normalised diagnostic and call-hierarchy wire layer.
//!
//! Every parser is exercised with the malformed shapes real servers send: `null`, missing
//! optional fields, string versus numeric diagnostic codes, empty diagnostic lists, unknown
//! pull-report kinds, `LocationLink` implementation results, and flat versus nested
//! call-hierarchy items.
use serde_json::json;

use crate::lsp::protocol::{parse_locations, Location, Position, Range};
use crate::lsp::wire::{
    parse_call_hierarchy_incoming_calls, parse_call_hierarchy_items,
    parse_call_hierarchy_outgoing_calls, parse_diagnostics, parse_diagnostics_event,
    parse_document_diagnostic_report, Diagnostic, DiagnosticCode, DiagnosticRelatedInformation,
    DiagnosticReportKind, DocumentDiagnosticReport,
};

/// Builds a range literal for assertions.
fn range(start_line: u32, start_character: u32, end_line: u32, end_character: u32) -> Range {
    Range {
        start: Position::new(start_line, start_character),
        end: Position::new(end_line, end_character),
    }
}

#[test]
fn diagnostics_parse_from_arrays_and_tolerate_malformed_entries() {
    assert_eq!(parse_diagnostics(&json!(null)), Vec::new());
    assert_eq!(parse_diagnostics(&json!({})), Vec::new());
    assert_eq!(parse_diagnostics(&json!([])), Vec::new());
    assert_eq!(parse_diagnostics(&json!("not a list")), Vec::new());

    let diagnostics = parse_diagnostics(&json!([
        null,
        {
            "range": {
                "start": { "line": 2, "character": 3 },
                "end": { "line": 2, "character": 9 },
            },
            "message": "boom",
            "severity": 1,
        },
        { "message": "no range at all" },
        { "range": {}, "message": 7 },
    ]));
    assert_eq!(diagnostics.len(), 2);
    assert_eq!(diagnostics[0].message, "boom");
    assert_eq!(diagnostics[0].severity, Some(1));
    assert_eq!(diagnostics[0].range, range(2, 3, 2, 9));
    assert_eq!(diagnostics[1].message, "no range at all");
    assert_eq!(diagnostics[1].severity, None);
    assert_eq!(diagnostics[1].range, Range::default());
    assert_eq!(diagnostics[1].tags, Vec::<u32>::new());
    assert_eq!(
        diagnostics[1].related_information,
        Vec::<DiagnosticRelatedInformation>::new()
    );
}

#[test]
fn diagnostics_parse_from_an_object_carrying_items() {
    let diagnostics =
        parse_diagnostics(&json!({ "items": [{ "message": "from items" }], "kind": "full" }));
    assert_eq!(diagnostics.len(), 1);
    assert_eq!(diagnostics[0].message, "from items");

    let wrapped = parse_diagnostics(&json!({ "diagnostics": [{ "message": "wrapped" }] }));
    assert_eq!(wrapped.len(), 1);
    assert_eq!(wrapped[0].message, "wrapped");
}

#[test]
fn diagnostic_codes_accept_strings_numbers_and_nulls() {
    let parsed = parse_diagnostics(&json!([
        { "message": "a", "code": "unused_variable" },
        { "message": "b", "code": 2304 },
        { "message": "c", "code": null },
        { "message": "d" },
        { "message": "e", "code": { "nested": true } },
    ]));
    assert_eq!(
        parsed[0].code,
        Some(DiagnosticCode::String("unused_variable".to_owned()))
    );
    assert_eq!(parsed[1].code, Some(DiagnosticCode::Number(2304)));
    assert_eq!(parsed[2].code, None);
    assert_eq!(parsed[3].code, None);
    assert_eq!(parsed[4].code, None);

    let numeric = serde_json::to_value(&parsed[1]).expect("diagnostic serializes");
    assert_eq!(numeric["code"], json!(2304));
    let named = serde_json::to_value(&parsed[0]).expect("diagnostic serializes");
    assert_eq!(named["code"], json!("unused_variable"));
    let round_trip: Diagnostic =
        serde_json::from_value(numeric).expect("serialized diagnostic deserializes");
    assert_eq!(round_trip, parsed[1]);
}

#[test]
fn diagnostic_related_information_uses_the_lsp_camel_case_shape() {
    let diagnostic = Diagnostic {
        message: "main".to_owned(),
        severity: Some(2),
        range: Range::default(),
        source: Some("fake".to_owned()),
        code: Some(DiagnosticCode::Number(7)),
        tags: vec![1, 2],
        related_information: vec![DiagnosticRelatedInformation {
            location: Location {
                uri: "file:///workspace/src/lib.rs".to_owned(),
                range: range(1, 0, 1, 4),
            },
            message: "declared here".to_owned(),
        }],
    };
    let value = serde_json::to_value(&diagnostic).expect("diagnostic serializes");
    assert_eq!(
        value["relatedInformation"][0]["location"]["uri"],
        json!("file:///workspace/src/lib.rs")
    );
    assert_eq!(
        value["relatedInformation"][0]["message"],
        json!("declared here")
    );
    assert_eq!(value["tags"], json!([1, 2]));
    assert_eq!(value["code"], json!(7));
    assert!(value.get("related_information").is_none());

    let round_trip: Diagnostic =
        serde_json::from_value(value).expect("serialized diagnostic deserializes");
    assert_eq!(round_trip, diagnostic);
}

#[test]
fn diagnostics_events_require_a_uri_and_a_list() {
    assert!(parse_diagnostics_event(&json!(null)).is_none());
    assert!(parse_diagnostics_event(&json!("textDocument/publishDiagnostics")).is_none());
    assert!(parse_diagnostics_event(&json!({ "uri": "", "diagnostics": [] })).is_none());
    assert!(parse_diagnostics_event(&json!({ "uri": "file:///a.rs" })).is_none());
    assert!(parse_diagnostics_event(&json!({ "uri": "file:///a.rs", "diagnostics": 3 })).is_none());

    let event = parse_diagnostics_event(&json!({
        "uri": "file:///a.rs",
        "version": 4,
        "diagnostics": [{ "message": "warn", "severity": 2 }],
    }))
    .expect("a complete event parses");
    assert_eq!(event.uri, "file:///a.rs");
    assert_eq!(event.version, Some(4));
    assert_eq!(event.diagnostics.len(), 1);
    assert_eq!(event.diagnostics[0].severity, Some(2));

    let resolved = parse_diagnostics_event(&json!({
        "uri": "file:///a.rs",
        "version": 5,
        "diagnostics": [],
    }))
    .expect("an empty list means the diagnostics are resolved");
    assert_eq!(resolved.diagnostics, Vec::new());

    let items_shape = parse_diagnostics_event(&json!({ "uri": "file:///a.rs", "items": [] }))
        .expect("the items spelling is accepted");
    assert_eq!(items_shape.version, None);
    assert_eq!(items_shape.diagnostics, Vec::new());
}

#[test]
fn document_diagnostic_reports_cover_full_unchanged_and_unknown_kinds() {
    let full = parse_document_diagnostic_report(&json!({
        "kind": "full",
        "resultId": "r1",
        "items": [{ "message": "cannot find name" }],
    }));
    assert_eq!(full.kind, DiagnosticReportKind::Full);
    assert_eq!(full.result_id.as_deref(), Some("r1"));
    assert_eq!(full.items.len(), 1);

    let unchanged = parse_document_diagnostic_report(&json!({
        "kind": "unchanged",
        "resultId": "r1",
    }));
    assert_eq!(unchanged.kind, DiagnosticReportKind::Unchanged);
    assert_eq!(unchanged.result_id.as_deref(), Some("r1"));
    assert_eq!(unchanged.items, Vec::new());

    let unknown = parse_document_diagnostic_report(&json!({
        "kind": "partial",
        "items": [{ "message": "x" }],
    }));
    assert_eq!(unknown.kind, DiagnosticReportKind::Full);
    assert_eq!(unknown.items.len(), 1);

    assert_eq!(
        parse_document_diagnostic_report(&json!(null)),
        DocumentDiagnosticReport::default()
    );
}

#[test]
fn call_hierarchy_items_flatten_flat_and_nested_uris() {
    let items = parse_call_hierarchy_items(&json!([
        {
            "name": "flat",
            "kind": 12,
            "detail": "fn flat()",
            "uri": "file:///a.rs",
            "range": {
                "start": { "line": 1, "character": 0 },
                "end": { "line": 2, "character": 1 },
            },
            "selectionRange": {
                "start": { "line": 1, "character": 3 },
                "end": { "line": 1, "character": 7 },
            },
            "data": { "k": 1 },
        },
        {
            "name": "nested",
            "kind": 13,
            "textDocument": { "uri": "file:///b.rs" },
        },
        { "kind": 12, "uri": "file:///c.rs" },
        { "name": "no-uri", "kind": 12 },
        { "name": "no-kind", "uri": "file:///d.rs" },
        null,
    ]));
    assert_eq!(items.len(), 3);
    assert_eq!(items[0].name, "flat");
    assert_eq!(items[0].kind, 12);
    assert_eq!(items[0].detail.as_deref(), Some("fn flat()"));
    assert_eq!(items[0].uri, "file:///a.rs");
    assert_eq!(items[0].range, range(1, 0, 2, 1));
    assert_eq!(items[0].selection_range, range(1, 3, 1, 7));
    assert_eq!(items[0].data, Some(json!({ "k": 1 })));

    assert_eq!(items[1].name, "nested");
    assert_eq!(items[1].kind, 13);
    assert_eq!(items[1].uri, "file:///b.rs");
    assert_eq!(items[1].range, Range::default());
    assert_eq!(items[1].selection_range, Range::default());
    assert_eq!(items[1].detail, None);
    assert_eq!(items[1].data, None);

    assert_eq!(items[2].name, "no-kind");
    assert_eq!(items[2].kind, 0);

    let single = parse_call_hierarchy_items(&json!({ "name": "solo", "uri": "file:///solo.rs" }));
    assert_eq!(single.len(), 1);
    assert_eq!(single[0].uri, "file:///solo.rs");
    assert_eq!(parse_call_hierarchy_items(&json!(null)), Vec::new());
}

#[test]
fn call_hierarchy_calls_parse_both_directions() {
    let incoming = parse_call_hierarchy_incoming_calls(&json!([
        {
            "from": { "name": "caller", "kind": 12, "uri": "file:///caller.rs" },
            "fromRanges": [
                {
                    "start": { "line": 2, "character": 4 },
                    "end": { "line": 2, "character": 9 },
                },
            ],
        },
        { "from": { "name": "no-uri" } },
        null,
    ]));
    assert_eq!(incoming.len(), 1);
    assert_eq!(incoming[0].from.name, "caller");
    assert_eq!(incoming[0].from.uri, "file:///caller.rs");
    assert_eq!(incoming[0].from_ranges, vec![range(2, 4, 2, 9)]);

    let outgoing = parse_call_hierarchy_outgoing_calls(&json!([
        {
            "to": {
                "name": "callee",
                "kind": 12,
                "textDocument": { "uri": "file:///callee.rs" },
            },
            "fromRanges": [],
        },
    ]));
    assert_eq!(outgoing.len(), 1);
    assert_eq!(outgoing[0].to.name, "callee");
    assert_eq!(outgoing[0].to.uri, "file:///callee.rs");
    assert_eq!(outgoing[0].from_ranges, Vec::new());

    assert_eq!(
        parse_call_hierarchy_incoming_calls(&json!(null)),
        Vec::new()
    );
    assert_eq!(parse_call_hierarchy_outgoing_calls(&json!({})), Vec::new());
}

#[test]
fn implementation_location_links_are_normalised() {
    let locations = parse_locations(&json!([
        {
            "targetUri": "file:///impl.rs",
            "targetRange": {
                "start": { "line": 5, "character": 0 },
                "end": { "line": 6, "character": 1 },
            },
            "targetSelectionRange": {
                "start": { "line": 5, "character": 4 },
                "end": { "line": 5, "character": 9 },
            },
        },
        {
            "uri": "file:///impl.rs",
            "range": {
                "start": { "line": 0, "character": 0 },
                "end": { "line": 0, "character": 3 },
            },
        },
    ]));
    assert_eq!(locations.len(), 2);
    assert_eq!(locations[0].uri, "file:///impl.rs");
    assert_eq!(locations[0].range, range(5, 0, 6, 1));
    assert_eq!(locations[1].range, range(0, 0, 0, 3));
    assert_eq!(parse_locations(&json!(null)), Vec::new());
}
