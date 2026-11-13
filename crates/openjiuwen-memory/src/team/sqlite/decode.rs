//! Row decoding for the `SQLite` team memory store, with its documented degradation.
//!
//! A well-formed row always decodes exactly; the interesting part is what happens to a
//! row that is not well formed (written by an older release, edited by hand, or left
//! behind by a failed migration). The rule is: a caller asking for a listing must not
//! lose the whole listing because one row is odd, but nothing is dropped silently.
//!
//! - `visibility_json` that is not valid JSON, or names an unknown variant, decodes as
//!   [`TeamVisibility::Private`] (fail closed: an unreadable visibility must never widen
//!   access) and logs a warning.
//! - `tags_json` that is not a JSON array of strings decodes as an empty tag list, with a
//!   warning.
//! - `metadata_json` that is not valid JSON, or is valid JSON but not an object, decodes
//!   as `{}`, with a warning.
//! - `importance` outside `0.0..=1.0` is clamped into the range, a non-finite value
//!   becomes [`DEFAULT_IMPORTANCE`], and both log a warning.
//! - `expires_at` that cannot be parsed is treated as "no expiry" (again with a warning),
//!   which keeps the record visible instead of hiding it behind a broken deadline.
//! - A row whose `created_at` or `updated_at` cannot be parsed is skipped: those are the
//!   sort keys of every listing, and inventing a timestamp would silently reorder rows.
//!
//! Every other column is required by the schema, so a read failure there is a real error
//! and is propagated instead of degraded.
use super::schema::parse_ts;
use crate::team::model::{TeamMemoryKind, TeamMemoryRecord, TeamVisibility, DEFAULT_IMPORTANCE};
use openjiuwen_common::error::{Error, Result};
use sqlx::sqlite::SqliteRow;
use sqlx::Row;
use tracing::warn;
/// The tracing target of every log line this module emits.
const LOG_TARGET: &str = "openjiuwen_memory::team";
/// Reads a non-null text column.
fn text(row: &SqliteRow, column: &str) -> Result<String> {
    row.try_get(column).map_err(Error::external)
}
/// Reads a nullable text column.
fn optional_text(row: &SqliteRow, column: &str) -> Result<Option<String>> {
    row.try_get(column).map_err(Error::external)
}
/// Reads the `importance` column as the `REAL` SQLite stores.
fn real(row: &SqliteRow, column: &str) -> Result<f64> {
    row.try_get(column).map_err(Error::external)
}
/// Decodes `visibility_json`, failing closed to [`TeamVisibility::Private`].
fn decode_visibility(raw: &str, id: &str) -> TeamVisibility {
    match serde_json::from_str::<TeamVisibility>(raw) {
        Ok(visibility) => visibility,
        Err(error) => {
            warn!(
                target: LOG_TARGET,
                id = %id,
                error = %error,
                "malformed team memory `visibility_json`; falling back to private"
            );
            TeamVisibility::Private
        }
    }
}
/// Decodes `tags_json`, degrading a malformed payload to an empty tag list.
fn decode_tags(raw: &str, id: &str) -> Vec<String> {
    match serde_json::from_str::<Vec<String>>(raw) {
        Ok(tags) => tags,
        Err(error) => {
            warn!(
                target: LOG_TARGET,
                id = %id,
                error = %error,
                "malformed team memory `tags_json`; falling back to no tags"
            );
            Vec::new()
        }
    }
}
/// Decodes `metadata_json`, degrading a malformed payload or a non-object to `{}`.
fn decode_metadata(raw: &str, id: &str) -> serde_json::Value {
    let empty = || serde_json::Value::Object(serde_json::Map::new());
    match serde_json::from_str::<serde_json::Value>(raw) {
        Ok(value) if value.is_object() => value,
        Ok(value) => {
            warn!(
                target: LOG_TARGET,
                id = %id,
                value = %value,
                "team memory `metadata_json` is not an object; falling back to an empty object"
            );
            empty()
        }
        Err(error) => {
            warn!(
                target: LOG_TARGET,
                id = %id,
                error = %error,
                "malformed team memory `metadata_json`; falling back to an empty object"
            );
            empty()
        }
    }
}
/// Clamps a stored `importance` into `0.0..=1.0`.
fn decode_importance(raw: f64, id: &str) -> f32 {
    let importance = raw as f32;
    if !importance.is_finite() {
        warn!(
            target: LOG_TARGET,
            id = %id,
            importance = %raw,
            "non-finite team memory `importance`; falling back to the default"
        );
        return DEFAULT_IMPORTANCE;
    }
    let clamped = importance.clamp(0.0, 1.0);
    if (clamped - importance).abs() > f32::EPSILON {
        warn!(
            target: LOG_TARGET,
            id = %id,
            importance = %importance,
            "out-of-range team memory `importance`; clamping into 0.0..=1.0"
        );
    }
    clamped
}
/// Decodes one row.
///
/// Returns `Ok(None)` when the row's `created_at`/`updated_at` cannot be parsed (the row
/// is skipped with a warning); malformed JSON columns and an out-of-range importance
/// degrade as documented on the module. A failure to read a required column is returned
/// as an error.
pub(super) fn decode_row(row: &SqliteRow) -> Result<Option<TeamMemoryRecord>> {
    let id = text(row, "id")?;
    let created_raw = text(row, "created_at")?;
    let updated_raw = text(row, "updated_at")?;
    let (Some(created_at), Some(updated_at)) = (parse_ts(&created_raw), parse_ts(&updated_raw))
    else {
        warn!(
            target: LOG_TARGET,
            id = %id,
            created_at = %created_raw,
            updated_at = %updated_raw,
            "unparseable team memory timestamps; skipping the row"
        );
        return Ok(None);
    };
    let expires_at = decode_expiry(optional_text(row, "expires_at")?, &id);
    let record = TeamMemoryRecord {
        id: id.clone(),
        team_id: text(row, "team_id")?,
        member_id: optional_text(row, "member_id")?,
        session_id: optional_text(row, "session_id")?,
        kind: TeamMemoryKind::parse(&text(row, "kind")?),
        content: text(row, "content")?,
        title: optional_text(row, "title")?,
        importance: decode_importance(real(row, "importance")?, &id),
        tags: decode_tags(&text(row, "tags_json")?, &id),
        metadata: decode_metadata(&text(row, "metadata_json")?, &id),
        created_at,
        updated_at,
        source: optional_text(row, "source")?,
        visibility: decode_visibility(&text(row, "visibility_json")?, &id),
        expires_at,
    };
    Ok(Some(record))
}
/// Decodes the optional expiry column; an unparseable deadline becomes "no expiry".
fn decode_expiry(raw: Option<String>, id: &str) -> Option<chrono::DateTime<chrono::Utc>> {
    let raw = raw?;
    let parsed = parse_ts(&raw);
    if parsed.is_none() {
        warn!(
            target: LOG_TARGET,
            id = %id,
            expires_at = %raw,
            "unparseable team memory `expires_at`; treating the record as unexpiring"
        );
    }
    parsed
}
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use super::*;
    use chrono::{DateTime, Utc};
    #[test]
    fn importance_is_clamped_and_non_finite_falls_back() {
        assert!((decode_importance(0.25, "id") - 0.25).abs() < f32::EPSILON);
        assert!((decode_importance(-1.0, "id") - 0.0).abs() < f32::EPSILON);
        assert!((decode_importance(4.0, "id") - 1.0).abs() < f32::EPSILON);
        assert!((decode_importance(f64::NAN, "id") - DEFAULT_IMPORTANCE).abs() < f32::EPSILON);
        assert!((decode_importance(f64::INFINITY, "id") - DEFAULT_IMPORTANCE).abs() < f32::EPSILON);
    }
    #[test]
    fn malformed_json_columns_degrade() {
        assert_eq!(
            decode_visibility("{\"members\":", "id"),
            TeamVisibility::Private
        );
        assert_eq!(
            decode_visibility("\"shared\"", "id"),
            TeamVisibility::Private
        );
        assert_eq!(decode_visibility("\"team\"", "id"), TeamVisibility::Team);
        assert!(decode_tags("not-json", "id").is_empty());
        assert_eq!(decode_tags("[\"a\",\"b\"]", "id"), vec!["a", "b"]);
        assert_eq!(decode_metadata("\"text\"", "id"), serde_json::json!({}));
        assert_eq!(decode_metadata("[1]", "id"), serde_json::json!({}));
        assert_eq!(decode_metadata("{oops", "id"), serde_json::json!({}));
        assert_eq!(
            decode_metadata("{\"issue\":42}", "id"),
            serde_json::json!({"issue": 42})
        );
    }
    #[test]
    fn expiry_degrades_to_none() {
        let at: DateTime<Utc> = DateTime::from_timestamp(1_700_000_000, 0).unwrap();
        assert_eq!(decode_expiry(None, "id"), None);
        assert_eq!(decode_expiry(Some("soon".to_owned()), "id"), None);
        assert_eq!(
            decode_expiry(Some("2023-11-14T22:13:20.000Z".to_owned()), "id"),
            Some(at)
        );
    }
}
