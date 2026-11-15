//! Tests of the `SQLite` team memory store.
//!
//! Every test is deterministic: timestamps come from [`DateTime::from_timestamp`], the
//! database is a `tempfile` directory or `sqlite::memory:`, and no assertion depends on
//! the wall clock. Malformed rows are injected straight through
//! [`SqliteTeamMemoryStore::pool`] so the decoder is exercised against data a normal write
//! path cannot produce.
//!
//! - `crud` — round-trip fidelity, reopening a file, the upsert contract, team isolation,
//!   delete/clear/set-expiry return values, and pool sharing.
//! - `filters` — `list` filter combinations, ordering, pagination, `stats`, and expiry.
//! - `search` — visibility and keyword ranking through the shared ranker.
//! - `decode` — malformed rows degrade or skip instead of failing the listing.
//! - `embedding` — the additive `embedding_json` column: round trip, the upgrade of a
//!   table written before it existed, semantic ranking from stored vectors, and the
//!   degradation of a malformed vector.
use super::*;
use crate::team::model::{TeamMemoryKind, TeamMemoryRecord, TeamMemoryScope};
use chrono::{DateTime, Utc};
mod crud;
mod decode;
mod embedding;
mod filters;
mod search;
/// A fixed instant used as "now" in fixture records.
pub(super) fn at(seconds: i64) -> DateTime<Utc> {
    DateTime::from_timestamp(seconds, 0).unwrap()
}
/// Builds a record with every field at a deterministic value.
pub(super) fn record(
    team: &str,
    member: Option<&str>,
    id: &str,
    content: &str,
    kind: TeamMemoryKind,
) -> TeamMemoryRecord {
    let scope = member.map_or_else(
        || TeamMemoryScope::new(team),
        |member| TeamMemoryScope::member(team, member),
    );
    let mut record = TeamMemoryRecord::new(&scope, content, kind)
        .with_id(id)
        .with_importance(0.5);
    record.member_id = member.map(str::to_owned);
    record.created_at = at(1_700_000_000);
    record.updated_at = at(1_700_000_000);
    record
}
/// Returns `record` with `updated_at` set to `seconds` after the epoch.
pub(super) fn updated_at(mut record: TeamMemoryRecord, seconds: i64) -> TeamMemoryRecord {
    record.updated_at = at(seconds);
    record
}
/// Opens an in-memory store.
pub(super) async fn store() -> SqliteTeamMemoryStore {
    SqliteTeamMemoryStore::connect("sqlite::memory:")
        .await
        .unwrap()
}
/// Inserts every record of `records` into `store`.
pub(super) async fn seed(store: &SqliteTeamMemoryStore, records: &[TeamMemoryRecord]) {
    for record in records {
        store.upsert(record).await.unwrap();
    }
}
