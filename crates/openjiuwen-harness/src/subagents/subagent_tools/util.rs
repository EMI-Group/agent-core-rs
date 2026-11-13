//! Shared parsing, task-id and payload-rendering helpers for the `subagent_*` tools.

use serde_json::Value;

use super::{DEFAULT_WAIT_TIMEOUT_MS, MAX_WAIT_TIMEOUT_MS, MIN_WAIT_TIMEOUT_MS, SUBAGENT_ROW_KEYS};
use crate::tools::{compact_json, render_fields};

/// Returns the trimmed, non-empty string value of `key`, if any.
pub(crate) fn non_empty(inputs: &Value, key: &str) -> Option<String> {
    inputs
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

/// Normalizes `subagent_ids`, accepting a singular `subagent_id` as a fallback.
pub(crate) fn parse_subagent_ids(inputs: &Value) -> Vec<String> {
    let raw = match inputs.get("subagent_ids") {
        Some(Value::Null) | None => inputs.get("subagent_id"),
        Some(value) => Some(value),
    };
    match raw {
        Some(Value::String(id)) => normalize_ids(std::iter::once(id.as_str())),
        Some(Value::Array(items)) => normalize_ids(items.iter().filter_map(Value::as_str)),
        _ => Vec::new(),
    }
}

/// Trims IDs, drops blanks, and keeps input order.
pub(crate) fn normalize_ids<'a>(ids: impl Iterator<Item = &'a str>) -> Vec<String> {
    ids.map(str::trim)
        .filter(|id| !id.is_empty())
        .map(str::to_owned)
        .collect()
}

/// Resolves and clamps the `subagent_wait` deadline.
///
/// An absent deadline keeps [`DEFAULT_WAIT_TIMEOUT_MS`]; a present but non-integer value is
/// a failed payload, matching Python's `isinstance(timeout_ms, int)` check.
pub(crate) fn resolve_timeout_ms(inputs: &Value) -> std::result::Result<u64, &'static str> {
    match inputs.get("timeout_ms") {
        None => Ok(DEFAULT_WAIT_TIMEOUT_MS),
        Some(value) => value
            .as_i64()
            .map(|timeout_ms| {
                timeout_ms
                    .clamp(MIN_WAIT_TIMEOUT_MS, MAX_WAIT_TIMEOUT_MS)
                    .unsigned_abs()
            })
            .ok_or("'timeout_ms' must be an integer"),
    }
}

/// Returns a new Python-style `task_id` (32 hex characters).
pub(crate) fn new_task_id() -> String {
    uuid::Uuid::new_v4().simple().to_string()
}

/// Returns the failure message of a payload, or `None` on success.
pub(crate) fn payload_error(output: &Value) -> Option<&str> {
    if output.get("success").and_then(Value::as_bool) == Some(false) {
        output.get("error").and_then(Value::as_str)
    } else {
        None
    }
}

/// Renders the listed fields of a successful payload in Python `render_fields` style.
pub(crate) fn render_payload(output: &Value, keys: &[&str]) -> String {
    if let Some(error) = payload_error(output) {
        return error.to_owned();
    }
    let Some(data) = output.get("data").and_then(Value::as_object) else {
        return String::new();
    };
    let fields: Vec<(&str, Value)> = keys
        .iter()
        .filter_map(|key| data.get(*key).map(|value| (*key, value.clone())))
        .collect();
    render_fields(&fields)
}

/// Renders one `transcripts` map entry of a `subagent_wait` payload.
///
/// The first line is `transcript: <TranscriptSummary::display()>`; a non-empty transcript
/// adds its newest entry's content on a second `last:` line. Missing keys degrade to a zero
/// summary instead of panicking.
pub(crate) fn render_transcript_summary(summary: &Value) -> String {
    let count = summary
        .get("count")
        .and_then(Value::as_u64)
        .unwrap_or_default();
    let last_kind = summary.get("last_kind").and_then(Value::as_str);
    let last_content = summary
        .get("last_content")
        .and_then(Value::as_str)
        .filter(|content| !content.is_empty());
    let noun = if count == 1 { "entry" } else { "entries" };
    let digest = match (last_kind, last_content) {
        (Some(kind), Some(content)) => format!("{count} {noun}, last: {kind}: {content}"),
        (Some(kind), None) => format!("{count} {noun}, last: {kind}"),
        _ => format!("{count} {noun}"),
    };
    let mut rendered = format!("transcript: {digest}");
    if let Some(content) = last_content {
        rendered = format!("{rendered}\nlast: {content}");
    }
    rendered
}

/// Renders one live/closed sub-agent list, Python `_render_subagent_rows` style.
pub(crate) fn render_subagent_rows(title: &str, rows: &[Value]) -> String {
    if rows.is_empty() {
        return format!("{title}: none");
    }
    let lines: Vec<String> = rows
        .iter()
        .map(|row| {
            let fields: Vec<String> = SUBAGENT_ROW_KEYS
                .iter()
                .filter_map(|key| {
                    let value = row.get(*key)?;
                    let text = match value {
                        Value::Null => return None,
                        Value::String(text) if text.is_empty() => return None,
                        Value::String(text) => text.clone(),
                        other => compact_json(other),
                    };
                    Some(format!("{key}: {text}"))
                })
                .collect();
            format!("- {}", fields.join(" | "))
        })
        .collect();
    format!("{title}:\n{}", lines.join("\n"))
}
