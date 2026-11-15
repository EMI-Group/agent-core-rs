//! Tolerant JSON accessors shared by every REST vendor spec.
//!
//! Vendors answer in slightly different shapes: a bare array, or an array wrapped in
//! `results`/`memories`/`items`/`data`/`entries`; a record id under `id`/`memory_id`/
//! `unit_id`/`uuid`; memory text under `memory`/`content`/`text`/`data`; timestamps as
//! RFC 3339 strings or unix seconds. These helpers absorb that variation so a spec's
//! `*_result` method stays a few lines long and never panics on an unexpected shape:
//! an absent field yields `None`.
//!
//! # Tolerance versus shape drift
//!
//! There are two kinds of "unexpected" body, and they must not look alike:
//!
//! - **An optional field is missing** — a row without a timestamp, importance, or id. Here
//!   tolerance is right: [`vendor_record`] keeps what the vendor sent and leaves the rest
//!   at its documented default.
//! - **The body's shape changed** — a renamed wrapper, an HTML page from a proxy, a row
//!   that is not a memory at all. Here tolerance is a silent failure: a renamed wrapper
//!   would read as "no rows", which is indistinguishable from an empty collection.
//!
//! So a *listing* decoder uses [`results_array_checked`] (a non-empty body with no
//! recognised row array is [`Error::Validation`]) and [`listing_record`] (a row with no
//! memory content is skipped with a `tracing::warn!` instead of becoming a record with an
//! invented id and an empty body). [`results_array`] and [`vendor_record`] stay tolerant
//! because a write or a point read legitimately answers a single object rather than an
//! array.
//!
//! The module is declared next to [`rest`](super::rest) and re-exported from it, so a
//! spec only needs `use super::rest::*`.
//!
//! Nothing here performs I/O: the helpers accept a [`Value`] (or an already-buffered
//! [`HttpResponse`] for [`body_value`]) and are pure and infallible apart from
//! [`body_value`] and [`results_array_checked`], which report an unusable body as a typed
//! error.

use chrono::{DateTime, Utc};
use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::http::HttpResponse;
use serde_json::Value;

use crate::record::{MemoryRecord, MemoryScope, MemoryType, RecallHit};

/// `tracing` target of the shape-drift diagnostics this module emits.
const LOG_TARGET: &str = "openjiuwen_memory::external::rest_json";
/// Longest list of row keys quoted in one diagnostic.
const QUOTED_KEYS: usize = 8;

/// Object keys a vendor body may wrap its row array in.
const RESULT_KEYS: &[&str] = &["results", "memories", "items", "data", "entries"];
/// Record id aliases, in priority order.
const ID_KEYS: &[&str] = &["id", "memory_id", "unit_id", "uuid"];
/// Record content aliases, in priority order.
const CONTENT_KEYS: &[&str] = &["memory", "content", "text", "data"];
/// Creation-timestamp aliases, in priority order.
const CREATED_KEYS: &[&str] = &["created_at", "timestamp", "created"];
/// Update-timestamp aliases, in priority order.
const UPDATED_KEYS: &[&str] = &["updated_at", "modified_at", "updated"];
/// Metadata-object aliases, in priority order.
const METADATA_KEYS: &[&str] = &["metadata", "fields"];
/// Importance/score aliases, in priority order.
const IMPORTANCE_KEYS: &[&str] = &["importance", "score"];
/// Top-level error-message keys of a vendor error body.
const ERROR_KEYS: &[&str] = &["error", "message", "detail"];
/// Nested error-message keys inside one entry of a `errors` array.
const NESTED_ERROR_KEYS: &[&str] = &["message", "detail", "error"];
/// Marker appended to a detail string that had to be shortened.
const ELLIPSIS: char = '…';

/// Decodes a response body as a JSON value.
///
/// An empty (or whitespace-only) body is [`Value::Null`] — several vendors answer
/// `204 No Content` on a successful delete. Anything else must be JSON: a plain-text or
/// HTML body becomes a typed transport error naming the response URL, because a vendor
/// that answers non-JSON has not honoured the spec at all.
///
/// # Errors
///
/// Returns [`Error::Transport`] when the body is not valid JSON.
pub fn body_value(response: &HttpResponse) -> Result<Value> {
    let text = response.text();
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Ok(Value::Null);
    }
    serde_json::from_str(trimmed).map_err(|error| {
        Error::transport(format!(
            "the memory provider returned a non-JSON body for {}: {error}",
            response.url
        ))
    })
}

/// Returns the row array of a vendor body, tolerantly.
///
/// Accepts a top-level array or one wrapped in `RESULT_KEYS`. An unrecognized shape yields
/// an empty slice rather than an error, because several vendors answer a *single* object
/// on a write or a point read (`{"id": …}`, `{"memory": …}`), where "no row array" is the
/// documented shape and not drift. A scalar body (a bare string, number, or boolean) is
/// never a documented answer, so it is logged as drift; an object body is not, because it
/// may legitimately be a single record.
///
/// A listing decoder must not use this function: it fills
/// [`results_array_checked`], which reports a renamed wrapper instead of reading it as an
/// empty collection.
#[must_use]
pub fn results_array(value: &Value) -> &[Value] {
    if let Value::Array(items) = value {
        return items;
    }
    for key in RESULT_KEYS {
        if let Some(Value::Array(items)) = value.get(*key) {
            return items;
        }
    }
    if value.is_boolean() || value.is_number() || value.is_string() {
        tracing::warn!(
            target: LOG_TARGET,
            "a vendor body that is a bare scalar carries no row array; reading it as no rows"
        );
    }
    &[]
}

/// Returns the row array of a listing answer, reporting shape drift.
///
/// Accepts everything [`results_array`] accepts, plus an empty body ([`Value::Null`],
/// which several vendors answer on an empty collection) as "no rows". Every other
/// non-empty body — a renamed wrapper such as `{"payload": […]}` (a pure object with no
/// recognised array key), a scalar, or an object whose `results`-style key holds a
/// non-array — is reported as a typed [`Error::Validation`] naming the response URL and
/// the keys the vendor answered with, after a `tracing::warn!`. That keeps a vendor rename
/// or a proxy error page from being read as an empty collection.
///
/// # Errors
///
/// Returns [`Error::Validation`] when a non-empty body carries no recognised row array.
pub fn results_array_checked<'a>(response: &HttpResponse, value: &'a Value) -> Result<&'a [Value]> {
    if value.is_null() {
        return Ok(&[]);
    }
    if let Value::Array(items) = value {
        return Ok(items);
    }
    for key in RESULT_KEYS {
        if let Some(Value::Array(items)) = value.get(*key) {
            return Ok(items);
        }
    }
    let keys = quoted_keys(value);
    tracing::warn!(
        target: LOG_TARGET,
        %keys,
        url = %response.url,
        "a vendor listing body carries no recognised row array; refusing to read it as no rows"
    );
    Err(Error::validation(format!(
        "the memory provider returned a body with no recognised row array for {} (answered keys: {keys}); expected a JSON array or one wrapped in {}",
        response.url,
        RESULT_KEYS.join("/")
    )))
}

/// Returns the first present, non-null value under `keys`.
#[must_use]
pub fn field_value<'a>(value: &'a Value, keys: &[&str]) -> Option<&'a Value> {
    keys.iter()
        .find_map(|key| value.get(*key).filter(|field| !field.is_null()))
}

/// Returns the first non-blank string under `keys`, trimmed.
#[must_use]
pub fn field_str<'a>(value: &'a Value, keys: &[&str]) -> Option<&'a str> {
    keys.iter().find_map(|key| {
        value
            .get(*key)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|text| !text.is_empty())
    })
}

/// Returns the first numeric value under `keys`.
///
/// A numeric string (`"0.75"`) is accepted because several vendors serialize numbers as
/// strings.
#[must_use]
pub fn field_f64(value: &Value, keys: &[&str]) -> Option<f64> {
    keys.iter().find_map(|key| {
        let field = value.get(*key)?;
        field
            .as_f64()
            .or_else(|| field.as_str().and_then(|raw| raw.trim().parse().ok()))
    })
}

/// Returns the first integer value under `keys`, accepting a numeric string.
#[must_use]
pub fn field_i64(value: &Value, keys: &[&str]) -> Option<i64> {
    keys.iter().find_map(|key| {
        let field = value.get(*key)?;
        field
            .as_i64()
            .or_else(|| field.as_str().and_then(|raw| raw.trim().parse().ok()))
    })
}

/// Returns the first numeric value under `keys` narrowed to `f32`.
///
/// Recall scores are `f32` on [`RecallHit`], so this avoids an `as` cast at every spec
/// call site.
#[must_use]
pub fn field_f32(value: &Value, keys: &[&str]) -> Option<f32> {
    field_f64(value, keys).and_then(seconds_to_f32)
}

/// Narrows an `f64` to `f32`, rejecting values that cannot be represented.
fn seconds_to_f32(value: f64) -> Option<f32> {
    if value.is_finite() {
        Some(value as f32)
    } else {
        None
    }
}

/// Parses a timestamp held under `keys`.
///
/// Accepts an RFC 3339 string, an integer/float unix-seconds value, or a string holding
/// unix seconds. An unparseable or out-of-range value yields `None` so
/// [`vendor_record`] falls back to "now" instead of failing.
#[must_use]
pub fn parse_timestamp(value: &Value, keys: &[&str]) -> Option<DateTime<Utc>> {
    let field = field_value(value, keys)?;
    if let Some(raw) = field.as_str() {
        let raw = raw.trim();
        if let Ok(parsed) = DateTime::parse_from_rfc3339(raw) {
            return Some(parsed.with_timezone(&Utc));
        }
        if let Ok(seconds) = raw.parse::<i64>() {
            return DateTime::from_timestamp(seconds, 0);
        }
        return raw.parse::<f64>().ok().and_then(from_seconds_f64);
    }
    if let Some(seconds) = field.as_i64() {
        return DateTime::from_timestamp(seconds, 0);
    }
    field.as_f64().and_then(from_seconds_f64)
}

/// Builds a UTC timestamp from fractional unix seconds.
#[allow(clippy::cast_possible_truncation)]
fn from_seconds_f64(seconds: f64) -> Option<DateTime<Utc>> {
    if !seconds.is_finite() {
        return None;
    }
    let whole = seconds.trunc();
    let fraction = ((seconds - whole) * 1_000_000_000.0).round();
    let nanos = if fraction.is_finite() && (0.0..1_000_000_000.0).contains(&fraction) {
        fraction as u32
    } else {
        0
    };
    DateTime::from_timestamp(whole as i64, nanos)
}

/// Maps one vendor row onto a [`MemoryRecord`], best effort.
///
/// The id, content, creation/update timestamps, importance, and metadata object are read
/// through their alias lists; anything absent falls back to a generated id, empty
/// content, "now", importance `0.5`, and no metadata. The function never fails — a
/// vendor row that lacks a documented field must not break a whole listing — but it also
/// never invents content: an empty result is visible to the caller.
///
/// This is the accessor for a *single* record (a point read or a write answer), where
/// returning something is required. A listing uses [`listing_record`], which drops a row
/// that carries no content at all.
#[must_use]
pub fn vendor_record(value: &Value, scope: &MemoryScope, kind: MemoryType) -> MemoryRecord {
    let content = field_str(value, CONTENT_KEYS)
        .unwrap_or_default()
        .to_owned();
    let mut record = MemoryRecord::new(scope.clone(), content, kind);
    if let Some(id) = field_str(value, ID_KEYS) {
        record = record.with_id(id.to_owned());
    }
    if let Some(timestamp) = parse_timestamp(value, CREATED_KEYS) {
        record = record.with_timestamp(timestamp);
        record.updated_at = timestamp;
    }
    if let Some(updated) = parse_timestamp(value, UPDATED_KEYS) {
        record.updated_at = updated;
    }
    if let Some(importance) = field_f32(value, IMPORTANCE_KEYS) {
        record.importance = importance.clamp(0.0, 1.0);
    }
    if let Some(metadata) = field_value(value, METADATA_KEYS).and_then(Value::as_object) {
        for (key, field) in metadata {
            record.metadata.insert(key.clone(), field.clone());
        }
    }
    record
}

/// Maps one row of a *listing* onto a [`MemoryRecord`], skipping a row that is not a memory.
///
/// A row is a memory when it carries at least one content alias (`memory`/`content`/
/// `text`/`data`). A row without one is skipped, after a `tracing::warn!` naming
/// `operation` and the row's own keys, because reporting it would invent an id and an
/// empty body the vendor never stored. An id-less but content-bearing row stays usable:
/// mem0 and `LakeBase` recall answers legitimately omit the id, and a listing is read-only
/// anyway. Everything else about the row is decoded by [`vendor_record`], so an absent
/// timestamp, importance, or metadata object stays tolerated.
///
/// `None` means "this row is not a memory"; the caller's listing simply omits it.
#[must_use]
pub fn listing_record(
    row: &Value,
    scope: &MemoryScope,
    kind: MemoryType,
    operation: &str,
) -> Option<MemoryRecord> {
    if field_str(row, CONTENT_KEYS).is_none() {
        let keys = quoted_keys(row);
        tracing::warn!(
            target: LOG_TARGET,
            operation,
            %keys,
            "skipping a vendor row that carries no memory content"
        );
        return None;
    }
    Some(vendor_record(row, scope, kind))
}

/// Wraps `record` and `score` into a [`RecallHit`].
///
/// A remote backend returns one blended score and nothing else, so `similarity` mirrors
/// that score clamped to `0.0..=1.0`, `keyword` and `recency` are `0.0`, and `importance`
/// comes from the record. A non-finite score becomes `0.0` instead of poisoning the
/// caller's ranking arithmetic.
#[must_use]
pub fn vendor_hit(record: MemoryRecord, score: f32) -> RecallHit {
    let similarity = if score.is_finite() {
        score.clamp(0.0, 1.0)
    } else {
        0.0
    };
    RecallHit {
        importance: record.importance,
        record,
        score: if score.is_finite() { score } else { 0.0 },
        similarity,
        keyword: 0.0,
        recency: 0.0,
    }
}

/// Filters, ranks, and truncates vendor recall candidates into hits.
///
/// Each candidate carries the record and the score the vendor reported for it, or `None`
/// when the row carried no score at all. An absent score is *absent*, not zero: the vendor
/// matched the row but did not rank it, so `threshold` never drops it (otherwise a vendor
/// whose recall answer omits `score` would always look like "nothing matched" under the
/// default `0.3`). A scored row keeps the documented `score >= threshold` semantics, and a
/// non-positive `threshold` keeps everything.
///
/// Scored rows come first, ordered by their score descending; unscored rows follow in the
/// vendor's own order, since they have no rank to sort by. `top_k` then truncates the
/// merged list, so a vendor that sends both kinds still honours the caller's page size.
#[must_use]
pub fn rank_hits(
    candidates: Vec<(MemoryRecord, Option<f32>)>,
    threshold: f32,
    top_k: usize,
) -> Vec<RecallHit> {
    let mut kept: Vec<(MemoryRecord, Option<f32>)> = candidates
        .into_iter()
        .filter(|(_, score)| match score {
            Some(score) => threshold <= 0.0 || *score >= threshold,
            None => true,
        })
        .collect();
    kept.sort_by(|left, right| match (left.1, right.1) {
        (Some(left), Some(right)) => right.total_cmp(&left),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => std::cmp::Ordering::Equal,
    });
    kept.truncate(top_k);
    kept.into_iter()
        .map(|(record, score)| vendor_hit(record, score.unwrap_or(0.0)))
        .collect()
}

/// Collapses whitespace and shortens `body` to at most `limit` characters.
///
/// The result is safe for an error message or a log line: newlines and repeated spaces
/// fold into single spaces and truncation happens on a character boundary, with a `…`
/// appended so a shortened body is never mistaken for the whole answer.
#[must_use]
pub fn truncate_detail(body: &str, limit: usize) -> String {
    let collapsed = body.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.chars().count() <= limit {
        return collapsed;
    }
    let mut truncated = collapsed.chars().take(limit).collect::<String>();
    truncated.push(ELLIPSIS);
    truncated
}

/// Extracts the vendor's own error message from a JSON error body.
///
/// Looks at the top-level `error` (a string or an object with `message`), `message`, and
/// `detail`, then at the first entry of an `errors` array. A non-JSON or message-free
/// body yields `None`, leaving the caller to fall back to
/// [`truncate_detail`] of the raw body.
#[must_use]
pub fn error_detail(body: &str) -> Option<String> {
    let trimmed = body.trim();
    if trimmed.is_empty() {
        return None;
    }
    let value: Value = serde_json::from_str(trimmed).ok()?;
    for key in ERROR_KEYS {
        match value.get(*key) {
            Some(Value::String(message)) => {
                let message = message.trim();
                if !message.is_empty() {
                    return Some(message.to_owned());
                }
            }
            Some(Value::Object(object)) => {
                if let Some(message) = nested_message(object) {
                    return Some(message);
                }
            }
            _ => {}
        }
    }
    let first = value.get("errors")?.as_array()?.first()?;
    match first {
        Value::String(message) => {
            let message = message.trim();
            (!message.is_empty()).then(|| message.to_owned())
        }
        Value::Object(object) => nested_message(object),
        _ => None,
    }
}

/// Reads the first non-blank nested message of an error object.
fn nested_message(object: &serde_json::Map<String, Value>) -> Option<String> {
    for key in NESTED_ERROR_KEYS {
        if let Some(message) = object.get(*key).and_then(Value::as_str) {
            let message = message.trim();
            if !message.is_empty() {
                return Some(message.to_owned());
            }
        }
    }
    None
}

/// The top-level keys of a vendor body, comma-joined for a diagnostic.
///
/// Quotes key *names* only, never a value, so a log line can never carry memory content.
/// A non-object body is described by its JSON kind instead.
fn quoted_keys(value: &Value) -> String {
    match value {
        Value::Object(object) => {
            let mut keys: Vec<&str> = object.keys().map(String::as_str).collect();
            keys.sort_unstable();
            keys.truncate(QUOTED_KEYS);
            keys.join(", ")
        }
        Value::Array(_) => "[]".to_owned(),
        Value::String(_) => "a string".to_owned(),
        Value::Number(_) => "a number".to_owned(),
        Value::Bool(_) => "a boolean".to_owned(),
        Value::Null => "an empty body".to_owned(),
    }
}

/// The scope id of `scope`, as a `&str`.
#[must_use]
pub fn scope_id(scope: &MemoryScope) -> &str {
    &scope.scope_id
}

/// The user id of `scope`, as a `&str`.
#[must_use]
pub fn user_id(scope: &MemoryScope) -> &str {
    &scope.user_id
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use chrono::TimeZone;
    use serde_json::json;

    use super::*;
    use crate::record::SearchOptions;

    /// Builds a buffered response carrying `body` as the payload.
    #[allow(clippy::default_trait_access)] // `HeaderMap` is not a named dependency here.
    fn response(body: &str) -> HttpResponse {
        HttpResponse {
            status: 200,
            headers: Default::default(),
            url: "http://vendor.test/items".to_owned(),
            body: body.as_bytes().to_vec().into(),
        }
    }

    #[test]
    fn body_value_treats_an_empty_body_as_null_and_rejects_non_json() {
        assert_eq!(body_value(&response("")).expect("empty"), Value::Null);
        assert_eq!(body_value(&response("   \n")).expect("blank"), Value::Null);
        assert_eq!(
            body_value(&response(r#"{"ok":true}"#)).expect("json"),
            json!({"ok": true})
        );
        let error = body_value(&response("<html>nope</html>")).expect_err("non-json");
        assert!(matches!(error, Error::Transport(_)), "{error:?}");
        assert!(
            error.to_string().contains("http://vendor.test/items"),
            "{error}"
        );
    }

    #[test]
    fn results_array_accepts_every_wrapper_and_ignores_unknown_shapes() {
        let rows = json!([{"id": "a"}, {"id": "b"}]);
        assert_eq!(results_array(&rows).len(), 2);
        for key in RESULT_KEYS {
            let mut object = serde_json::Map::new();
            object.insert((*key).to_owned(), json!([{"id": "a"}]));
            let wrapped = Value::Object(object);
            assert_eq!(results_array(&wrapped).len(), 1, "{key}");
        }
        assert_eq!(results_array(&json!({"ok": true})), [] as [Value; 0]);
        assert_eq!(results_array(&json!(null)), [] as [Value; 0]);
    }

    #[test]
    fn results_array_checked_reads_an_empty_body_as_no_rows() {
        let empty = response("");
        let value = body_value(&empty).expect("empty body");
        assert_eq!(
            results_array_checked(&empty, &value).expect("an empty body is no rows"),
            [] as [Value; 0]
        );
        let wrapped = response(r#"{"results": []}"#);
        let value = body_value(&wrapped).expect("json");
        assert_eq!(
            results_array_checked(&wrapped, &value).expect("an empty array"),
            [] as [Value; 0]
        );
    }

    #[test]
    fn results_array_checked_rejects_shape_drift_instead_of_reporting_no_rows() {
        for body in [
            r#"{"payload": [{"id": "a"}]}"#,
            r#"{"ok": true}"#,
            r#"{"results": {"id": "a"}}"#,
            r#""a bare string""#,
        ] {
            let http = response(body);
            let value = body_value(&http).expect("json");
            let error = results_array_checked(&http, &value)
                .expect_err("shape drift must not read as no rows");
            assert!(matches!(error, Error::Validation(_)), "{body}: {error:?}");
            let message = error.to_string();
            assert!(
                message.contains("http://vendor.test/items"),
                "{body}: {message}"
            );
            assert!(message.contains("no recognised row array"), "{body}");
        }
        let drifted = response(r#"{"payload": []}"#);
        let value = body_value(&drifted).expect("json");
        let error = results_array_checked(&drifted, &value).expect_err("drift");
        assert!(
            error.to_string().contains("payload"),
            "the answered keys must be named: {error}"
        );
    }

    #[test]
    fn listing_record_skips_a_row_without_content() {
        let scope = MemoryScope::new("alice", "ws");
        let usable = listing_record(
            &json!({"id": "a", "memory": "hello"}),
            &scope,
            MemoryType::SemanticMemory,
            "list",
        )
        .expect("a content-bearing row is a record");
        assert_eq!(usable.id, "a");
        assert_eq!(usable.content, "hello");
        let idless = listing_record(
            &json!({"content": "no id"}),
            &scope,
            MemoryType::SemanticMemory,
            "list",
        )
        .expect("an id-less row stays usable");
        assert_eq!(idless.content, "no id");
        assert_ne!(idless.id, "", "the generated id is only a placeholder");
        for row in [
            json!({}),
            json!({"id": "a"}),
            json!({"memory_id": 7, "score": 0.9}),
            json!({"content": "   "}),
        ] {
            assert!(
                listing_record(&row, &scope, MemoryType::SemanticMemory, "list").is_none(),
                "{row}"
            );
        }
    }

    #[test]
    fn rank_hits_keeps_an_unscored_row_and_filters_a_scored_one() {
        let scope = MemoryScope::new("alice", "ws");
        let record =
            |content: &str| MemoryRecord::new(scope.clone(), content, MemoryType::SemanticMemory);
        let candidates = vec![
            (record("scored low"), Some(0.1)),
            (record("unscored"), None),
            (record("scored high"), Some(0.9)),
        ];
        let hits = rank_hits(candidates.clone(), 0.3, 5);
        assert_eq!(
            hits.len(),
            2,
            "the low score is dropped, the unscored row stays"
        );
        assert_eq!(hits[0].record.content, "scored high");
        assert_eq!(hits[0].score, 0.9_f32);
        assert_eq!(hits[1].record.content, "unscored");
        assert!(hits[1].score.abs() < f32::EPSILON, "no score is invented");
        assert_eq!(
            rank_hits(candidates.clone(), 0.0, 5).len(),
            3,
            "a non-positive threshold keeps a low score but still never drops an unscored row"
        );
        assert_eq!(
            rank_hits(
                vec![(record("only unscored"), None)],
                SearchOptions::default().threshold,
                5
            )
            .len(),
            1,
            "the default threshold never hides an unscored row"
        );
        assert_eq!(rank_hits(candidates, 0.3, 1).len(), 1, "top_k truncates");
    }

    #[test]
    fn field_helpers_prefer_the_first_alias_and_skip_blanks() {
        let row = json!({
            "memory_id": "mem-1",
            "memory": "  hello  ",
            "text": "ignored",
            "score": "0.75",
            "created_at": 1_700_000_000_i64,
            "metadata": {"source": "api"},
        });
        assert_eq!(field_str(&row, ID_KEYS), Some("mem-1"));
        assert_eq!(field_str(&row, CONTENT_KEYS), Some("hello"));
        assert_eq!(field_f64(&row, &["score"]), Some(0.75));
        assert_eq!(field_f32(&row, &["score"]), Some(0.75_f32));
        assert_eq!(field_i64(&row, &["created_at"]), Some(1_700_000_000));
        assert_eq!(
            field_value(&row, METADATA_KEYS),
            Some(&json!({"source": "api"}))
        );
        assert_eq!(field_str(&row, &["missing"]), None);
        assert_eq!(field_f64(&row, &["text"]), None);
    }

    #[test]
    fn parse_timestamp_accepts_rfc_3339_and_unix_seconds() {
        let expected = Utc.with_ymd_and_hms(2023, 11, 14, 22, 13, 20).unwrap();
        for row in [
            json!({"created_at": "2023-11-14T22:13:20Z"}),
            json!({"created_at": 1_700_000_000_i64}),
            json!({"created_at": 1_700_000_000_u64}),
            json!({"created_at": "1700000000"}),
            json!({"created_at": 1_700_000_000.0_f64}),
            json!({"timestamp": "2023-11-14T22:13:20.000Z"}),
        ] {
            assert_eq!(parse_timestamp(&row, CREATED_KEYS), Some(expected), "{row}");
        }
        assert_eq!(
            parse_timestamp(&json!({"created_at": "soon"}), CREATED_KEYS),
            None
        );
        assert_eq!(parse_timestamp(&json!({}), CREATED_KEYS), None);
    }

    #[test]
    fn vendor_record_maps_aliases_and_generates_a_missing_id() {
        let scope = MemoryScope::new("alice", "ws");
        let row = json!({
            "memory_id": "mem-1",
            "memory": "alice prefers dark mode",
            "created_at": 1_700_000_000_i64,
            "updated_at": 1_700_000_100_i64,
            "score": 0.9,
            "metadata": {"origin": "import"},
            "unexpected": [1, 2, 3],
        });
        let record = vendor_record(&row, &scope, MemoryType::SemanticMemory);
        assert_eq!(record.id, "mem-1");
        assert_eq!(record.content, "alice prefers dark mode");
        assert_eq!(record.kind, MemoryType::SemanticMemory);
        assert_eq!(record.scope, scope);
        assert_eq!(
            record.timestamp,
            DateTime::from_timestamp(1_700_000_000, 0).unwrap()
        );
        assert_eq!(
            record.updated_at,
            DateTime::from_timestamp(1_700_000_100, 0).unwrap()
        );
        assert!((record.importance - 0.9).abs() < f32::EPSILON);
        assert_eq!(record.metadata_str("origin"), Some("import"));

        let bare = vendor_record(&json!({"content": "no id"}), &scope, MemoryType::Summary);
        assert_eq!(bare.id.len(), 24, "a generated id is 24 hex characters");
        assert_eq!(bare.content, "no id");
        assert!((bare.importance - 0.5).abs() < f32::EPSILON);

        let empty = vendor_record(&json!({}), &scope, MemoryType::Unknown);
        assert_eq!(empty.content, "");
        assert_ne!(empty.id, "");
    }

    #[test]
    fn vendor_hit_keeps_the_blended_score_and_zeroes_the_components() {
        let record = MemoryRecord::new(
            MemoryScope::new("alice", "ws"),
            "hello",
            MemoryType::SemanticMemory,
        );
        let hit = vendor_hit(record.clone(), 0.75);
        assert!((hit.score - 0.75).abs() < f32::EPSILON);
        assert!((hit.similarity - 0.75).abs() < f32::EPSILON);
        assert!(hit.keyword.abs() < f32::EPSILON);
        assert!(hit.recency.abs() < f32::EPSILON);
        assert!((hit.importance - record.importance).abs() < f32::EPSILON);
        assert_eq!(hit.record, record);

        assert!(vendor_hit(record.clone(), -2.0).similarity.abs() < f32::EPSILON);
        assert!((vendor_hit(record.clone(), 4.0).similarity - 1.0).abs() < f32::EPSILON);
        let nan = vendor_hit(record, f32::NAN);
        assert!(nan.score.abs() < f32::EPSILON);
        assert!(nan.similarity.abs() < f32::EPSILON);
    }

    #[test]
    fn truncate_detail_collapses_whitespace_and_stays_on_char_boundaries() {
        assert_eq!(truncate_detail("  a \n b\tc  ", 32), "a b c");
        let long = "é".repeat(20);
        let shortened = truncate_detail(&long, 5);
        assert_eq!(shortened, "ééééé…");
        assert_eq!(truncate_detail("", 10), "");
    }

    #[test]
    fn error_detail_reads_every_documented_shape() {
        assert_eq!(
            error_detail(r#"{"error":"bad key"}"#).as_deref(),
            Some("bad key")
        );
        assert_eq!(
            error_detail(r#"{"error":{"message":"bad key"}}"#).as_deref(),
            Some("bad key")
        );
        assert_eq!(
            error_detail(r#"{"message":"too many"}"#).as_deref(),
            Some("too many")
        );
        assert_eq!(
            error_detail(r#"{"detail":" nope "}"#).as_deref(),
            Some("nope")
        );
        assert_eq!(
            error_detail(r#"{"errors":[{"message":"first failure"}]}"#).as_deref(),
            Some("first failure")
        );
        assert_eq!(
            error_detail(r#"{"errors":["plain"]}"#).as_deref(),
            Some("plain")
        );
        assert_eq!(error_detail(""), None);
        assert_eq!(error_detail("not json"), None);
        assert_eq!(error_detail(r#"{"error":""}"#), None);
    }

    #[test]
    fn id_accessors_return_the_scope_segments() {
        let scope = MemoryScope::new("alice", "workspace").session("s1");
        assert_eq!(user_id(&scope), "alice");
        assert_eq!(scope_id(&scope), "workspace");
    }
}
