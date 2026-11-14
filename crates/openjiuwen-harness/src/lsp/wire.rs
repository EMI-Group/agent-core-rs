//! Normalised pull-diagnostics and call-hierarchy wire types and their parsers.
//!
//! The types `protocol` keeps to framing and navigation now live here, so `protocol.rs` stays
//! a pure framing/wire-shape module. Everything is normalised to the harness shapes used by
//! later phases: diagnostics keep their optional severity/source/code/tags, call-hierarchy
//! items are flat and URI-based (`uri`, `range`, `selectionRange`) instead of nesting
//! `textDocument`, and pull reports collapse `full`/`unchanged` into
//! [`DiagnosticReportKind`].
//!
//! Every parser is tolerant by construction: `null`, missing fields and malformed entries
//! yield empty vectors or `None` instead of an error, because language servers disagree about
//! optional fields and a diagnostic stream must never break the agent loop.
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::protocol::{Location, Position, Range};

/// A diagnostic code, which the protocol allows to be a string or a number.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum DiagnosticCode {
    /// A string code, for example a lint name.
    String(String),
    /// A numeric code, for example a compiler error number.
    Number(i64),
}

/// One diagnostic attached to a document range.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct Diagnostic {
    /// Human-readable diagnostic message.
    pub message: String,
    /// LSP `DiagnosticSeverity` number (1 error, 2 warning, 3 information, 4 hint).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub severity: Option<u32>,
    /// Range the diagnostic applies to; `0,0`-`0,0` when the server omits it.
    #[serde(default)]
    pub range: Range,
    /// Tool or server that produced the diagnostic.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// Optional code identifying the diagnostic kind.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<DiagnosticCode>,
    /// LSP `DiagnosticTag` numbers (1 unnecessary, 2 deprecated).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<u32>,
    /// Additional locations the diagnostic refers to.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub related_information: Vec<DiagnosticRelatedInformation>,
}

/// A secondary location attached to a [`Diagnostic`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct DiagnosticRelatedInformation {
    /// Location of the related information.
    pub location: Location,
    /// Message explaining the relation.
    pub message: String,
}

/// One `textDocument/publishDiagnostics` notification, normalised.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct DiagnosticsEvent {
    /// `file://` URI of the document the diagnostics belong to.
    pub uri: String,
    /// Document version the server computed the diagnostics for, when it reports one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<i64>,
    /// Diagnostics for the document; empty means "all previous diagnostics are resolved".
    #[serde(default)]
    pub diagnostics: Vec<Diagnostic>,
}

/// Which kind of report a `textDocument/diagnostic` result is.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "camelCase")]
pub enum DiagnosticReportKind {
    /// A complete diagnostic report, replacing anything cached for the document.
    #[default]
    Full,
    /// The previous report with the same result id is still valid.
    Unchanged,
}

/// One normalised `textDocument/diagnostic` pull report.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct DocumentDiagnosticReport {
    /// Whether this is a full report or an unchanged marker.
    #[serde(default)]
    pub kind: DiagnosticReportKind,
    /// Opaque id the server can use to answer `unchanged` next time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result_id: Option<String>,
    /// Diagnostics of a full report; always empty for an unchanged marker.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub items: Vec<Diagnostic>,
}

/// One call-hierarchy node, normalised to a flat URI-based item.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct CallHierarchyItem {
    /// Symbol name shown to the model.
    pub name: String,
    /// LSP `SymbolKind` number.
    pub kind: u32,
    /// Signature or type detail, when the server provides one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    /// `file://` URI of the file declaring the symbol.
    pub uri: String,
    /// Full range of the symbol.
    #[serde(default)]
    pub range: Range,
    /// Range of the identifier itself.
    #[serde(default)]
    pub selection_range: Range,
    /// LSP `SymbolTag` numbers.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<u32>,
    /// Opaque server payload that must be sent back with `callHierarchy/*` requests.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

/// One incoming call of a [`CallHierarchyItem`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct CallHierarchyIncomingCall {
    /// The caller.
    pub from: CallHierarchyItem,
    /// Ranges inside the caller that call the item.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub from_ranges: Vec<Range>,
}

/// One outgoing call of a [`CallHierarchyItem`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct CallHierarchyOutgoingCall {
    /// The callee.
    pub to: CallHierarchyItem,
    /// Ranges inside the caller that call the callee.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub from_ranges: Vec<Range>,
}

/// Parses a diagnostic list, accepting a bare array or an object carrying `diagnostics`/`items`.
#[must_use]
pub fn parse_diagnostics(value: &Value) -> Vec<Diagnostic> {
    diagnostic_array(value)
        .map(|items| items.iter().filter_map(diagnostic).collect())
        .unwrap_or_default()
}

/// Parses one `textDocument/publishDiagnostics` notification.
///
/// Returns `None` when the payload is not an object, carries an empty `uri`, or has neither a
/// `diagnostics` nor an `items` array: a notification without a document cannot be attributed.
#[must_use]
pub fn parse_diagnostics_event(value: &Value) -> Option<DiagnosticsEvent> {
    let object = value.as_object()?;
    let uri = object.get("uri").and_then(Value::as_str)?;
    if uri.is_empty() {
        return None;
    }
    let diagnostics = diagnostic_array(value)?;
    Some(DiagnosticsEvent {
        uri: uri.to_owned(),
        version: object.get("version").and_then(Value::as_i64),
        diagnostics: diagnostics.iter().filter_map(diagnostic).collect(),
    })
}

/// Parses a `textDocument/diagnostic` result.
///
/// `{ "kind": "full", "resultId": .., "items": [..] }` becomes [`DiagnosticReportKind::Full`],
/// `{ "kind": "unchanged", "resultId": .. }` becomes [`DiagnosticReportKind::Unchanged`], and
/// any other or missing `kind` falls back to a full report with the items that are present.
#[must_use]
pub fn parse_document_diagnostic_report(value: &Value) -> DocumentDiagnosticReport {
    let kind = match value.get("kind").and_then(Value::as_str) {
        Some("unchanged") => DiagnosticReportKind::Unchanged,
        _ => DiagnosticReportKind::Full,
    };
    DocumentDiagnosticReport {
        kind,
        result_id: value
            .get("resultId")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        items: parse_diagnostics(value),
    }
}

/// Parses a `textDocument/prepareCallHierarchy` result into flat items.
///
/// Accepts an array of items, a single item object, and items that either carry a flat `uri`
/// or nest `{ "textDocument": { "uri": .. } }`.
#[must_use]
pub fn parse_call_hierarchy_items(value: &Value) -> Vec<CallHierarchyItem> {
    match value {
        Value::Array(items) => items.iter().filter_map(call_hierarchy_item).collect(),
        Value::Object(_) => call_hierarchy_item(value).into_iter().collect(),
        _ => Vec::new(),
    }
}

/// Parses a `callHierarchy/incomingCalls` result.
#[must_use]
pub fn parse_call_hierarchy_incoming_calls(value: &Value) -> Vec<CallHierarchyIncomingCall> {
    let Some(entries) = value.as_array() else {
        return Vec::new();
    };
    entries
        .iter()
        .filter_map(|entry| {
            let from = call_hierarchy_item(entry.get("from")?)?;
            Some(CallHierarchyIncomingCall {
                from,
                from_ranges: ranges(entry.get("fromRanges")),
            })
        })
        .collect()
}

/// Parses a `callHierarchy/outgoingCalls` result.
#[must_use]
pub fn parse_call_hierarchy_outgoing_calls(value: &Value) -> Vec<CallHierarchyOutgoingCall> {
    let Some(entries) = value.as_array() else {
        return Vec::new();
    };
    entries
        .iter()
        .filter_map(|entry| {
            let to = call_hierarchy_item(entry.get("to")?)?;
            Some(CallHierarchyOutgoingCall {
                to,
                from_ranges: ranges(entry.get("fromRanges")),
            })
        })
        .collect()
}

/// Returns the object's `diagnostics`/`items` array, or the value itself when it is an array.
fn diagnostic_array(value: &Value) -> Option<&Vec<Value>> {
    match value {
        Value::Array(items) => Some(items),
        Value::Object(map) => map
            .get("diagnostics")
            .or_else(|| map.get("items"))
            .and_then(Value::as_array),
        _ => None,
    }
}

/// Parses one raw `Diagnostic`; a diagnostic without a message is dropped.
fn diagnostic(value: &Value) -> Option<Diagnostic> {
    let message = value.get("message").and_then(Value::as_str)?.to_owned();
    Some(Diagnostic {
        message,
        severity: value
            .get("severity")
            .and_then(Value::as_u64)
            .and_then(|severity| u32::try_from(severity).ok()),
        range: range(value.get("range")),
        source: value
            .get("source")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        code: value.get("code").and_then(diagnostic_code),
        tags: u32_list(value.get("tags")),
        related_information: value
            .get("relatedInformation")
            .and_then(Value::as_array)
            .map(|items| items.iter().filter_map(related_information).collect())
            .unwrap_or_default(),
    })
}

/// Parses one `DiagnosticRelatedInformation` entry.
fn related_information(value: &Value) -> Option<DiagnosticRelatedInformation> {
    let message = value.get("message").and_then(Value::as_str)?.to_owned();
    let location = location(value.get("location")?)?;
    Some(DiagnosticRelatedInformation { location, message })
}

/// Parses one raw `Diagnostic.code`, which the protocol allows to be a string or a number.
fn diagnostic_code(value: &Value) -> Option<DiagnosticCode> {
    match value {
        Value::String(code) => Some(DiagnosticCode::String(code.clone())),
        Value::Number(number) => number.as_i64().map(DiagnosticCode::Number),
        _ => None,
    }
}

/// Parses one `Location`-shaped object, defaulting a missing range to `0,0`-`0,0`.
fn location(value: &Value) -> Option<Location> {
    let uri = value.get("uri").and_then(Value::as_str)?;
    if uri.is_empty() {
        return None;
    }
    Some(Location {
        uri: uri.to_owned(),
        range: range(value.get("range")),
    })
}

/// Parses one flat call-hierarchy item, tolerating a nested `textDocument.uri` shape.
fn call_hierarchy_item(value: &Value) -> Option<CallHierarchyItem> {
    let object = value.as_object()?;
    let name = object.get("name").and_then(Value::as_str)?.to_owned();
    let uri = object
        .get("uri")
        .and_then(Value::as_str)
        .or_else(|| {
            object
                .get("textDocument")
                .and_then(|document| document.get("uri"))
                .and_then(Value::as_str)
        })
        .filter(|uri| !uri.is_empty())?;
    let full_range = range(object.get("range"));
    let selection_range = object
        .get("selectionRange")
        .map_or(full_range, |selection| range(Some(selection)));
    Some(CallHierarchyItem {
        name,
        kind: object
            .get("kind")
            .and_then(Value::as_u64)
            .and_then(|kind| u32::try_from(kind).ok())
            .unwrap_or(0),
        detail: object
            .get("detail")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        uri: uri.to_owned(),
        range: full_range,
        selection_range,
        tags: u32_list(object.get("tags")),
        data: object.get("data").filter(|data| !data.is_null()).cloned(),
    })
}

/// Parses a `fromRanges` array, dropping malformed entries and keeping the rest.
fn ranges(value: Option<&Value>) -> Vec<Range> {
    value
        .and_then(Value::as_array)
        .map(|items| items.iter().map(|item| range(Some(item))).collect())
        .unwrap_or_default()
}

/// Parses a range object, defaulting missing positions to `0,0`.
fn range(value: Option<&Value>) -> Range {
    let Some(value) = value else {
        return Range::default();
    };
    Range {
        start: position(value.get("start")),
        end: position(value.get("end")),
    }
}

/// Parses a position object, defaulting missing coordinates to `0`.
fn position(value: Option<&Value>) -> Position {
    let Some(value) = value else {
        return Position::default();
    };
    Position {
        line: coordinate(value.get("line")),
        character: coordinate(value.get("character")),
    }
}

/// Narrows one JSON number to a `u32` coordinate, defaulting malformed values to `0`.
fn coordinate(value: Option<&Value>) -> u32 {
    value
        .and_then(Value::as_u64)
        .and_then(|number| u32::try_from(number).ok())
        .unwrap_or(0)
}

/// Collects the `u32` entries of a JSON array, ignoring other entries.
fn u32_list(value: Option<&Value>) -> Vec<u32> {
    value
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_u64)
                .filter_map(|number| u32::try_from(number).ok())
                .collect()
        })
        .unwrap_or_default()
}
