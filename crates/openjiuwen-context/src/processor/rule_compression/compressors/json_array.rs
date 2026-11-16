//! JSON-array compression: CSV/JSONL rendering, then a salient-row digest.
//!
//! Ports `rule_compression/compressors/json_array_compressor.py`. Only plain
//! JSON arrays reach this compressor; a JSON object is never routed here.
//!
//! - Object arrays: schema density `filled_cells / (rows * keys)` decides CSV
//!   (`>= json_csv_min_density`) versus JSONL; both keep every row and are
//!   lossless in meaning.
//! - When the row-preserving candidate does not fit `ctx.max_tokens`, the
//!   compressor degrades to at most 12 salient rows followed by a
//!   `{"_omitted": n, "_summary": {...}}` marker.
//! - String, number and mixed arrays render compact JSON when it fits, and
//!   otherwise keep their first, last, extreme and error-matching rows.
//!
//! This is the only compressor that enforces [`fits_budget_and_saves`].
//!
//! `details` are never emitted for an accepted candidate, matching Python, but
//! every decline names its reason: `unparsable_array`, `empty_array`,
//! `no_object_keys`, `too_few_rows`, `candidate_does_not_fit` (the row-preserving
//! and the salient-row candidate both failed the budget/savings gate) and
//! `candidate_unchanged`. A number array also reports `unmeasurable_rows`, the
//! rows `f64` measurement could not score (always 0 for a parsed JSON number
//! array, so a future measurement gap is visible rather than silently scored
//! `0.0`).
//!
//! [`fits_budget_and_saves`]: crate::processor::rule_compression::common::fits_budget_and_saves
use super::{RuleCompressionResult, RuleCompressor};
use crate::processor::rule_compression::common::{fits_budget_and_saves, is_error_text};
use crate::processor::rule_compression::types::{ContentType, RuleContext};
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
/// An array rendered as CSV, JSONL or a salient-row digest.
#[derive(Debug, Clone, Copy, Default)]
pub struct JsonArrayCompressor;
impl RuleCompressor for JsonArrayCompressor {
    fn compress(&self, content: &str, ctx: &RuleContext) -> RuleCompressionResult {
        match ladder(content, ctx) {
            Ladder::Applied { candidate, lossy } if candidate != content => {
                RuleCompressionResult::applied(candidate, ContentType::JsonArray, lossy, None)
            }
            Ladder::Applied { .. } => {
                tracing::debug!(
                    reason = "candidate_unchanged",
                    "rule compression: JSON array declined, the candidate is the input"
                );
                RuleCompressionResult::declined(
                    content,
                    ContentType::JsonArray,
                    Some(json!({ "reason": "candidate_unchanged" })),
                )
            }
            Ladder::Declined(details) => {
                tracing::debug!(
                    reason = details["reason"].as_str().unwrap_or("declined"),
                    details = %details,
                    "rule compression: JSON array declined"
                );
                RuleCompressionResult::declined(content, ContentType::JsonArray, Some(details))
            }
        }
    }
}
/// What the JSON-array ladder decided.
enum Ladder {
    /// A candidate to accept.
    Applied {
        /// The rendered candidate.
        candidate: String,
        /// Whether rows were dropped.
        lossy: bool,
    },
    /// A decline carrying the reason and the facts behind it.
    Declined(Value),
}
/// A decline with `reason` plus the `facts` object merged into one report.
fn declined(reason: &'static str, facts: Value) -> Ladder {
    let mut map = Map::new();
    map.insert("reason".to_owned(), Value::from(reason));
    if let Value::Object(facts) = facts {
        for (key, value) in facts {
            map.insert(key, value);
        }
    }
    Ladder::Declined(Value::Object(map))
}
/// Runs the ladder: parse, render and gate `content`.
fn ladder(content: &str, ctx: &RuleContext) -> Ladder {
    let Some(rows) = load_json_array(content) else {
        return declined(
            "unparsable_array",
            json!({ "characters": content.chars().count() }),
        );
    };
    if rows.is_empty() {
        return declined("empty_array", json!({ "rows": 0 }));
    }
    if rows.iter().all(Value::is_object) {
        return object_array(content, &rows, ctx);
    }
    if rows.iter().all(Value::is_string) {
        return string_array(content, &rows, ctx);
    }
    if rows.iter().all(is_number) {
        return number_array(content, &rows, ctx);
    }
    mixed_array(content, &rows, ctx)
}
/// Whether `value` is a JSON number (booleans excluded).
fn is_number(value: &Value) -> bool {
    value.is_number()
}
/// Object arrays: CSV or JSONL, then a salient-row digest.
fn object_array(content: &str, rows: &[Value], ctx: &RuleContext) -> Ladder {
    let mut keys: Vec<String> = rows
        .iter()
        .filter_map(Value::as_object)
        .flat_map(|row| row.keys().cloned())
        .collect();
    keys.sort_unstable();
    keys.dedup();
    if keys.is_empty() {
        return declined("no_object_keys", json!({ "rows": rows.len() }));
    }
    let candidate = if schema_density(rows, keys.len()) >= ctx.json_csv_min_density {
        to_csv(rows, &keys)
    } else {
        to_jsonl(rows)
    };
    if fits_budget_and_saves(content, &candidate, ctx) {
        return Ladder::Applied {
            candidate,
            lossy: false,
        };
    }
    let selected = select_salient_rows(rows);
    let candidate = format_lossy_array(
        &selected,
        rows.len() - selected.len(),
        Some(object_summary(rows)),
    );
    if fits_budget_and_saves(content, &candidate, ctx) {
        return Ladder::Applied {
            lossy: selected.len() < rows.len(),
            candidate,
        };
    }
    declined(
        "candidate_does_not_fit",
        json!({
            "rows": rows.len(),
            "resolution": "salient_rows",
            "omitted_rows": rows.len() - selected.len(),
        }),
    )
}
/// String arrays: compact JSON, then first/last/error/longest rows.
fn string_array(content: &str, rows: &[Value], ctx: &RuleContext) -> Ladder {
    let compact = compact_json(rows);
    if fits_budget_and_saves(content, &compact, ctx) {
        return Ladder::Applied {
            candidate: compact,
            lossy: false,
        };
    }
    if rows.len() <= 6 {
        return declined(
            "too_few_rows",
            json!({ "rows": rows.len(), "resolution": "compact_json" }),
        );
    }
    let mut indexes = vec![0_usize, rows.len() - 1];
    for (index, value) in rows.iter().enumerate() {
        if value.as_str().is_some_and(is_error_text) {
            indexes.push(index);
        }
    }
    if let Some((index, _)) = rows
        .iter()
        .enumerate()
        .map(|(index, value)| (index, value.as_str().map_or(0, str::len)))
        .max_by_key(|(index, length)| (*length, std::cmp::Reverse(*index)))
    {
        indexes.push(index);
    }
    indexes.sort_unstable();
    indexes.dedup();
    let selected: Vec<Value> = indexes.iter().map(|index| rows[*index].clone()).collect();
    let candidate = format_lossy_array(&selected, rows.len() - selected.len(), None);
    if fits_budget_and_saves(content, &candidate, ctx) {
        return Ladder::Applied {
            candidate,
            lossy: true,
        };
    }
    declined(
        "candidate_does_not_fit",
        json!({
            "rows": rows.len(),
            "resolution": "salient_rows",
            "omitted_rows": rows.len() - selected.len(),
        }),
    )
}
/// Number arrays: compact JSON, then first/last/min/max/largest-delta rows.
fn number_array(content: &str, rows: &[Value], ctx: &RuleContext) -> Ladder {
    let compact = compact_json(rows);
    if fits_budget_and_saves(content, &compact, ctx) {
        return Ladder::Applied {
            candidate: compact,
            lossy: false,
        };
    }
    let unmeasurable_rows = rows.iter().filter(|value| value.as_f64().is_none()).count();
    if rows.len() <= 6 {
        return declined(
            "too_few_rows",
            json!({
                "rows": rows.len(),
                "resolution": "compact_json",
                "unmeasurable_rows": unmeasurable_rows,
            }),
        );
    }
    let mut indexes = vec![0_usize, rows.len() - 1];
    if let Some((index, _)) = first_extreme(rows, false) {
        indexes.push(index);
    }
    if let Some((index, _)) = first_extreme(rows, true) {
        indexes.push(index);
    }
    let mut change_index = 1_usize;
    let mut largest = f64::NEG_INFINITY;
    for index in 1..rows.len() {
        let (Some(current), Some(previous)) = (number_at(rows, index), number_at(rows, index - 1))
        else {
            // An unmeasurable row cannot contribute a delta; it is reported
            // instead of being scored as `0.0`.
            continue;
        };
        let delta = (current - previous).abs();
        if delta > largest {
            largest = delta;
            change_index = index;
        }
    }
    indexes.push(change_index.saturating_sub(1));
    indexes.push(change_index);
    indexes.sort_unstable();
    indexes.dedup();
    let selected: Vec<Value> = indexes.iter().map(|index| rows[*index].clone()).collect();
    let candidate = format_lossy_array(&selected, rows.len() - selected.len(), None);
    if fits_budget_and_saves(content, &candidate, ctx) {
        return Ladder::Applied {
            candidate,
            lossy: true,
        };
    }
    declined(
        "candidate_does_not_fit",
        json!({
            "rows": rows.len(),
            "resolution": "salient_rows",
            "omitted_rows": rows.len() - selected.len(),
            "unmeasurable_rows": unmeasurable_rows,
        }),
    )
}
/// Mixed arrays: compact JSON, then the first row of each type plus errors.
fn mixed_array(content: &str, rows: &[Value], ctx: &RuleContext) -> Ladder {
    let compact = compact_json(rows);
    if fits_budget_and_saves(content, &compact, ctx) {
        return Ladder::Applied {
            candidate: compact,
            lossy: false,
        };
    }
    if rows.len() <= 6 {
        return declined(
            "too_few_rows",
            json!({ "rows": rows.len(), "resolution": "compact_json" }),
        );
    }
    let mut indexes = vec![0_usize, rows.len() - 1];
    let mut seen_types: Vec<&'static str> = Vec::new();
    for (index, value) in rows.iter().enumerate() {
        let type_name = json_value_type(value);
        if !seen_types.contains(&type_name) {
            seen_types.push(type_name);
            indexes.push(index);
        }
        if is_error_text(&value.to_string()) {
            indexes.push(index);
        }
    }
    indexes.sort_unstable();
    indexes.dedup();
    let selected: Vec<Value> = indexes.iter().map(|index| rows[*index].clone()).collect();
    let candidate = format_lossy_array(&selected, rows.len() - selected.len(), None);
    if fits_budget_and_saves(content, &candidate, ctx) {
        return Ladder::Applied {
            candidate,
            lossy: true,
        };
    }
    declined(
        "candidate_does_not_fit",
        json!({
            "rows": rows.len(),
            "resolution": "salient_rows",
            "omitted_rows": rows.len() - selected.len(),
        }),
    )
}
/// Python's `_json_value_type` names.
fn json_value_type(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Object(_) => "object",
        Value::Array(_) => "array",
    }
}
/// `rows[index]` as `f64`, or `None` when the row carries no measurable number.
fn number_at(rows: &[Value], index: usize) -> Option<f64> {
    rows.get(index).and_then(Value::as_f64)
}
/// The first index with the smallest (`largest == false`) or largest value.
fn first_extreme(rows: &[Value], largest: bool) -> Option<(usize, f64)> {
    let mut best: Option<(usize, f64)> = None;
    for (index, value) in rows.iter().enumerate() {
        let Some(number) = value.as_f64() else {
            continue;
        };
        let better = match best {
            None => true,
            Some((_, current)) => {
                if largest {
                    number > current
                } else {
                    number < current
                }
            }
        };
        if better {
            best = Some((index, number));
        }
    }
    best
}
/// The fraction of schema cells a row set fills.
fn schema_density(rows: &[Value], key_count: usize) -> f64 {
    if rows.is_empty() || key_count == 0 {
        return 0.0;
    }
    let filled: usize = rows.iter().filter_map(Value::as_object).map(Map::len).sum();
    filled as f64 / (rows.len() * key_count) as f64
}
/// Renders the rows as a CSV table with a Python-compatible header.
fn to_csv(rows: &[Value], keys: &[String]) -> String {
    let mut out = String::from("[JSON_ARRAY compressed to CSV]\n");
    out.push_str(&csv_line(keys.iter().map(String::as_str)));
    for row in rows {
        let object = row.as_object();
        let cells: Vec<String> = keys
            .iter()
            .map(|key| csv_cell(object.and_then(|row| row.get(key))))
            .collect();
        out.push('\n');
        out.push_str(&csv_line(cells.iter().map(String::as_str)));
    }
    out.trim_end().to_owned()
}
/// Joins `cells` with commas, quoting the cells that need it.
fn csv_line<'a>(cells: impl Iterator<Item = &'a str>) -> String {
    let quoted: Vec<String> = cells.map(csv_quote).collect();
    quoted.join(",")
}
/// Quotes one CSV cell the way Python's `QUOTE_MINIMAL` writer does.
fn csv_quote(cell: &str) -> String {
    if cell.contains(',') || cell.contains('"') || cell.contains('\n') || cell.contains('\r') {
        format!("\"{}\"", cell.replace('"', "\"\""))
    } else {
        cell.to_owned()
    }
}
/// Renders one CSV cell the way `csv.DictWriter` would.
fn csv_cell(value: Option<&Value>) -> String {
    match value {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(text)) => text.clone(),
        Some(Value::Bool(flag)) => {
            if *flag {
                "True".to_owned()
            } else {
                "False".to_owned()
            }
        }
        Some(other) => other.to_string(),
    }
}
/// Renders the rows as JSON Lines with a Python-compatible header.
fn to_jsonl(rows: &[Value]) -> String {
    let lines: Vec<String> = rows.iter().map(Value::to_string).collect();
    format!("[JSON_ARRAY compressed to JSONL]\n{}", lines.join("\n"))
}
/// The up-to-12 salient rows, in their original order.
fn select_salient_rows(rows: &[Value]) -> Vec<Value> {
    if rows.len() <= 12 {
        return rows.to_vec();
    }
    let mut scored: Vec<(i32, usize)> = Vec::with_capacity(rows.len());
    for (index, row) in rows.iter().enumerate() {
        let mut score = 0_i32;
        if index < 5 || index >= rows.len() - 5 {
            score += 10;
        }
        if is_error_text(&row.to_string().to_lowercase()) {
            score += 20;
        }
        scored.push((score, index));
    }
    scored.sort_unstable_by(|left, right| right.cmp(left));
    scored.truncate(12);
    scored.sort_unstable_by_key(|(_, index)| *index);
    scored
        .into_iter()
        .filter_map(|(_, index)| rows.get(index).cloned())
        .collect()
}
/// Renders a lossy array with the `_omitted` / `_summary` marker.
fn format_lossy_array(selected: &[Value], omitted: usize, summary: Option<Value>) -> String {
    let mut output: Vec<Value> = selected.to_vec();
    if omitted > 0 {
        let mut marker = Map::new();
        marker.insert("_omitted".to_owned(), Value::from(omitted));
        if let Some(summary) = summary {
            marker.insert("_summary".to_owned(), summary);
        }
        output.push(Value::Object(marker));
    }
    compact_json(&output)
}
/// Per-key value counts for the keys with wide coverage and few distinct values.
fn object_summary(rows: &[Value]) -> Value {
    let mut keys: Vec<String> = rows
        .iter()
        .filter_map(Value::as_object)
        .flat_map(|row| row.keys().cloned())
        .collect();
    keys.sort_unstable();
    keys.dedup();
    let minimum_coverage = (rows.len() * 8 / 10).max(1);
    let mut value_counts = Map::new();
    for key in keys {
        let mut counts: BTreeMap<String, u64> = BTreeMap::new();
        let mut covered = 0_usize;
        for row in rows {
            let Some(object) = row.as_object() else {
                continue;
            };
            let Some(value) = object.get(&key) else {
                continue;
            };
            if !matches!(
                value,
                Value::Null | Value::String(_) | Value::Number(_) | Value::Bool(_)
            ) {
                continue;
            }
            covered += 1;
            let bucket = match value {
                Value::String(text) => text.clone(),
                other => other.to_string(),
            };
            *counts.entry(bucket).or_insert(0) += 1;
        }
        if covered < minimum_coverage || counts.len() > 10 {
            continue;
        }
        let bucket_map: Map<String, Value> = counts
            .into_iter()
            .map(|(value, count)| (value, Value::from(count)))
            .collect();
        value_counts.insert(key, Value::Object(bucket_map));
    }
    let mut summary = Map::new();
    summary.insert("total_rows".to_owned(), Value::from(rows.len()));
    if !value_counts.is_empty() {
        summary.insert("value_counts".to_owned(), Value::Object(value_counts));
    }
    Value::Object(summary)
}
/// Compact JSON of `value` (`serde_json`'s default separator-free rendering).
fn compact_json(value: &[Value]) -> String {
    Value::Array(value.to_vec()).to_string()
}
/// Parses `content` as a JSON array, repairing a truncated array when needed.
fn load_json_array(content: &str) -> Option<Vec<Value>> {
    match serde_json::from_str::<Value>(content) {
        Ok(Value::Array(rows)) => Some(rows),
        Ok(_) => None,
        Err(_) => {
            let parsed = super::super::router::repair_json_array(content)?;
            parsed.as_array().cloned()
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn object_arrays_render_as_csv_when_dense() {
        let rows: Vec<Value> = (0..40)
            .map(|index| {
                serde_json::json!({"id": index, "name": format!("item {index}"), "ok": true})
            })
            .collect();
        let content = Value::Array(rows).to_string();
        let ctx = RuleContext::new(10_000);
        let result = JsonArrayCompressor.compress(&content, &ctx);
        assert!(result.modified);
        assert!(result
            .content
            .starts_with("[JSON_ARRAY compressed to CSV]\nid,name,ok\n"));
        assert!(!result.lossy);
        assert!(result.content.contains("0,item 0,True"));
    }
    #[test]
    fn oversized_arrays_fall_back_to_salient_rows() {
        let rows: Vec<Value> = (0..200)
            .map(|index| serde_json::json!({"id": index, "message": format!("event {index}")}))
            .collect();
        let content = Value::Array(rows).to_string();
        // The CSV rendering does not fit, the 12-row digest does.
        let ctx = RuleContext::new(600);
        let result = JsonArrayCompressor.compress(&content, &ctx);
        assert!(result.modified);
        assert!(result.lossy);
        assert!(result.content.contains("\"_omitted\":188"));
        assert!(result.content.contains("\"_summary\":{\"total_rows\":200"));
    }
    #[test]
    fn small_arrays_are_declined_with_the_reason() {
        let content = r#"[{"a":1},{"a":2}]"#;
        let result = JsonArrayCompressor.compress(content, &RuleContext::new(100));
        assert!(!result.modified);
        assert_eq!(result.content, content);
        let details = result.details.expect("the decline reports why");
        assert_eq!(details["reason"], "candidate_does_not_fit");
        assert_eq!(details["rows"], 2);
    }
    #[test]
    fn every_decline_reason_is_reported() {
        let ctx = RuleContext::new(100);
        let unparsable = JsonArrayCompressor.compress("[{\"a\": oops}]", &ctx);
        assert!(!unparsable.modified);
        assert_eq!(
            unparsable.details.expect("details")["reason"],
            "unparsable_array"
        );
        let empty = JsonArrayCompressor.compress("[]", &ctx);
        assert!(!empty.modified);
        assert_eq!(empty.details.expect("details")["reason"], "empty_array");
        let numbers = JsonArrayCompressor.compress("[1,2,3]", &ctx);
        assert!(!numbers.modified);
        let details = numbers.details.expect("details");
        assert_eq!(details["reason"], "too_few_rows");
        // A parsed number array is measurable, so the count is zero rather than
        // an unscoreable row silently scored `0.0`.
        assert_eq!(details["unmeasurable_rows"], 0);
    }
}
