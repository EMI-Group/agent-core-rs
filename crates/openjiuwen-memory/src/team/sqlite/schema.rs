//! Schema DDL, column list, and timestamp encoding of the `SQLite` team memory store.
//!
//! The table is one row per [`TeamMemoryRecord`](crate::team::model::TeamMemoryRecord),
//! with the three structured columns (`visibility_json`, `tags_json`, `metadata_json`)
//! stored as JSON text and every timestamp stored as an RFC 3339 UTC string.
//!
//! # Timestamp encoding
//!
//! Timestamps are rendered with [`DateTime::to_rfc3339_opts`] using
//! [`SecondsFormat::Millis`] and a literal `Z` (`2023-11-14T22:13:20.000Z`) and parsed
//! back with [`DateTime::parse_from_rfc3339`]. A fixed-width UTC representation keeps
//! the strings lexicographically ordered the same way the instants are, so `ORDER BY`,
//! `MIN`/`MAX`, and the expiry comparisons (`expires_at <= ?`) are plain text
//! comparisons — no SQLite date functions and no per-row conversion. Time terms are
//! always bound as text for the same reason: a numeric epoch would not compare against
//! the stored strings.
//!
//! Millisecond precision is the deliberate trade-off: sub-millisecond digits would be
//! preserved by the database but are irrelevant for team memory, and fixing the width
//! keeps the strings sortable. `upsert` therefore returns the millisecond-truncated
//! record, which is exactly what a following `get` reads back.
use chrono::{DateTime, SecondsFormat, Utc};
/// Table DDL, applied idempotently by every constructor.
pub(super) const TABLE: &str = "
CREATE TABLE IF NOT EXISTS team_memories (
    id TEXT PRIMARY KEY,
    team_id TEXT NOT NULL,
    member_id TEXT,
    session_id TEXT,
    kind TEXT NOT NULL,
    visibility_json TEXT NOT NULL,
    title TEXT,
    content TEXT NOT NULL,
    importance REAL NOT NULL,
    tags_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    source TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    expires_at TEXT
)";
/// Index backing the `updated_at` pagination of every listing.
pub(super) const UPDATED_INDEX: &str = "
CREATE INDEX IF NOT EXISTS team_memories_team_updated_idx ON team_memories (team_id, updated_at DESC)";
/// Index backing the kind narrowing of `list` and `search`.
pub(super) const KIND_INDEX: &str =
    "CREATE INDEX IF NOT EXISTS team_memories_team_kind_idx ON team_memories (team_id, kind)";
/// Index backing the member narrowing of `list`.
pub(super) const MEMBER_INDEX: &str = "
CREATE INDEX IF NOT EXISTS team_memories_team_member_idx ON team_memories (team_id, member_id)";
/// Every DDL statement, in creation order.
pub(super) const DDL: [&str; 4] = [TABLE, UPDATED_INDEX, KIND_INDEX, MEMBER_INDEX];
/// The column list every `SELECT` reads, in the order [`decode`](super::decode)
/// expects.
pub(super) const COLUMNS: &str = "id, team_id, member_id, session_id, kind, visibility_json, \
     title, content, importance, tags_json, metadata_json, source, created_at, updated_at, \
     expires_at";
/// Renders a timestamp as a lexicographically sortable RFC 3339 UTC string.
pub(super) fn render_ts(timestamp: DateTime<Utc>) -> String {
    timestamp.to_rfc3339_opts(SecondsFormat::Millis, true)
}
/// Parses a stored timestamp back into UTC.
///
/// Returns `None` for a value that is not an RFC 3339 timestamp; the caller decides
/// whether that skips the row or drops the value. This store only ever writes the `Z`
/// form, but any numeric offset is accepted and normalized to UTC, so a row written by
/// hand with `+08:00` still decodes (its text simply does not sort with the `Z` strings,
/// which matters only for such foreign rows).
pub(super) fn parse_ts(raw: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(raw)
        .ok()
        .map(|parsed| parsed.with_timezone(&Utc))
}
/// Truncates a timestamp to the millisecond precision the table stores.
pub(super) fn truncate_ts(timestamp: DateTime<Utc>) -> DateTime<Utc> {
    DateTime::from_timestamp_millis(timestamp.timestamp_millis()).unwrap_or(timestamp)
}
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use super::*;
    #[test]
    fn timestamps_round_trip_and_sort() {
        let early = DateTime::from_timestamp(1_700_000_000, 0).unwrap();
        let late = DateTime::from_timestamp(1_700_000_001, 0).unwrap();
        let rendered = render_ts(early);
        assert_eq!(rendered, "2023-11-14T22:13:20.000Z");
        assert_eq!(parse_ts(&rendered), Some(early));
        assert!(render_ts(early) < render_ts(late));
        assert_eq!(
            render_ts(early),
            render_ts(truncate_ts(early + chrono::Duration::microseconds(123)))
        );
    }
    #[test]
    fn malformed_timestamps_do_not_parse() {
        for raw in ["", "now", "2023-11-14", "not-a-timestamp"] {
            assert_eq!(parse_ts(raw), None, "{raw}");
        }
        assert!(parse_ts("2023-11-14T22:13:20.123Z").is_some());
        // A foreign offset form is accepted and normalized to UTC.
        assert_eq!(
            parse_ts("2023-11-14T22:13:20+08:00"),
            Some(DateTime::from_timestamp(1_700_000_000 - 8 * 3_600, 0).unwrap())
        );
    }
    #[test]
    fn truncate_keeps_milliseconds() {
        let base = DateTime::from_timestamp(1_700_000_000, 0).unwrap();
        let precise = base + chrono::Duration::nanoseconds(123_456_789);
        assert_eq!(
            truncate_ts(precise),
            base + chrono::Duration::milliseconds(123)
        );
    }
    #[test]
    fn column_list_names_every_column() {
        let columns: Vec<&str> = COLUMNS.split(',').map(str::trim).collect();
        assert_eq!(columns.len(), 15);
        assert!(columns.contains(&"visibility_json"));
        assert!(columns.contains(&"expires_at"));
        assert_eq!(DDL.len(), 4);
    }
}
