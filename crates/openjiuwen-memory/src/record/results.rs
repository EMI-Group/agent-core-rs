//! Result types of the write and recall paths.
//!
//! These are plain serde values so callers can log, persist, or forward them; the
//! outcome types add small accessors for "how much happened" and "which records".
use super::record::MemoryRecord;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
/// One scored recall result.
///
/// The record is flattened onto the top level (so a hit has the same JSON fields as a
/// record plus the score components). `score` is the blended score the engine filtered
/// on; `similarity`, `keyword`, `importance`, and `recency` are the exact `0.0..=1.0`
/// components it was computed from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecallHit {
    /// The recalled record, flattened into this object.
    #[serde(flatten)]
    pub record: MemoryRecord,
    /// Final blended score in `0.0..=1.0`.
    pub score: f32,
    /// Normalized semantic similarity component.
    pub similarity: f32,
    /// Normalized BM25 keyword component.
    pub keyword: f32,
    /// The record's importance component.
    pub importance: f32,
    /// Exponential recency component.
    pub recency: f32,
}
/// What a write did with each input memory.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct AddOutcome {
    /// Records written as new memories.
    pub added: Vec<MemoryRecord>,
    /// Records merged into or replaced by an existing memory.
    pub merged: Vec<MemoryRecord>,
    /// Ids of existing records that were removed.
    ///
    /// Only a replace-style write populates this list: it removes the row it supersedes
    /// before storing the replacement, so the superseded id appears here, while the
    /// replacement record (stored under that same id) appears in
    /// [`AddOutcome::merged`]. The deletion is therefore visible even though the id
    /// survives. Merge, skip, and add never remove a row.
    pub deleted: Vec<String>,
    /// Ids of inputs dropped by the dedupe policy.
    pub skipped: Vec<String>,
    /// Reason duplicate detection did not run, when it degraded.
    ///
    /// `None` means the duplicate check ran (or was not requested), so the outcome is
    /// complete. `Some(reason)` means the write had to continue without duplicate
    /// detection — the candidate scan failed — so the records in this outcome were stored
    /// unchecked and a duplicate of an existing record may exist. Cancellation and
    /// timeouts still abort the write; they never degrade it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub degraded: Option<String>,
}
impl AddOutcome {
    /// Total number of entries across `added`, `merged`, `deleted`, and `skipped`.
    ///
    /// [`AddOutcome::degraded`] is a diagnostic, not an entry, so it is not counted. A
    /// replace-style write contributes one entry to both `merged` (the replacement) and
    /// `deleted` (the row it superseded), because it really did both.
    #[must_use]
    pub fn len(&self) -> usize {
        self.added.len() + self.merged.len() + self.deleted.len() + self.skipped.len()
    }
    /// Returns `true` when the write changed nothing.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
/// Aggregate statistics of a filtered record set.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct MemoryStats {
    /// Number of matching records.
    pub total: usize,
    /// Record count per kind, keyed by [`super::MemoryType::as_str`].
    pub by_kind: BTreeMap<String, usize>,
    /// Creation time of the oldest matching record.
    pub oldest: Option<DateTime<Utc>>,
    /// Creation time of the newest matching record.
    pub newest: Option<DateTime<Utc>>,
    /// Total token count of the matching record contents.
    pub total_tokens: usize,
}
/// The records produced by one Python-parity `add_mem` call.
///
/// Python returns a dict of per-kind unit lists plus extracted variables; this type
/// keeps that shape so callers can inspect one kind at a time.
///
/// Besides the records, the result reports what did *not* produce a record, so a call
/// that stored nothing because everything was a duplicate or because model extraction
/// failed is distinguishable from a call that found nothing to store:
/// [`AddMemResult::skipped`], [`AddMemResult::deleted`], and [`AddMemResult::degraded`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct AddMemResult {
    /// Variables extracted from the input (name to value).
    pub variables: BTreeMap<String, String>,
    /// User-profile records.
    pub user_profile: Vec<MemoryRecord>,
    /// Semantic-memory records.
    pub semantic_memory: Vec<MemoryRecord>,
    /// Episodic-memory records.
    pub episodic_memory: Vec<MemoryRecord>,
    /// Summary records.
    pub summary: Vec<MemoryRecord>,
    /// Coding-memory records.
    pub coding: Vec<MemoryRecord>,
    /// Number of candidate memories the dedupe policy dropped (`DedupePolicy::Skip`).
    ///
    /// These inputs were valid memories that were already stored, so nothing was written
    /// for them; a nonzero count with no records still means the call did something.
    pub skipped: usize,
    /// Number of existing records a write deleted before storing the replacement.
    ///
    /// A replace-style write removes the row it supersedes; a nonzero count makes that
    /// deletion visible even when only the replacement record is returned.
    pub deleted: usize,
    /// Why model-assisted extraction degraded to the deterministic heuristic path.
    ///
    /// `None` when extraction never needed a fallback: no model is attached, extraction
    /// is disabled, the model payload was applied, or there was nothing to extract.
    /// `Some(reason)` when the model call or its payload failed (including a payload
    /// carrying neither a fact nor a variable) and the heuristics produced the records.
    /// The reason names the failure so a degraded turn is visible in logs and results.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub degraded: Option<String>,
}
impl AddMemResult {
    /// Iterates every record in kind order (`user_profile`, `semantic`, `episodic`,
    /// `summary`, `coding`).
    pub fn records(&self) -> impl Iterator<Item = &MemoryRecord> {
        self.user_profile
            .iter()
            .chain(self.semantic_memory.iter())
            .chain(self.episodic_memory.iter())
            .chain(self.summary.iter())
            .chain(self.coding.iter())
    }
    /// Total number of records across all kinds.
    ///
    /// The [`AddMemResult::skipped`], [`AddMemResult::deleted`], and
    /// [`AddMemResult::degraded`] bookkeeping does not contribute: they describe what
    /// happened around the records, not records themselves.
    #[must_use]
    pub fn len(&self) -> usize {
        self.user_profile.len()
            + self.semantic_memory.len()
            + self.episodic_memory.len()
            + self.summary.len()
            + self.coding.len()
    }
    /// Returns `true` when no record was produced (variables may still be present).
    ///
    /// A call that skipped duplicates or deleted replaced rows still counts as empty
    /// here; read [`AddMemResult::skipped`] and [`AddMemResult::deleted`] for that.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::record::{MemoryScope, MemoryType};
    fn hit(id: &str, score: f32) -> RecallHit {
        RecallHit {
            record: MemoryRecord::new(
                MemoryScope::new("alice", "ws"),
                "some content",
                MemoryType::SemanticMemory,
            )
            .with_id(id),
            score,
            similarity: 0.8,
            keyword: 0.4,
            importance: 0.5,
            recency: 0.9,
        }
    }
    #[test]
    fn recall_hit_flattens_the_record() {
        let hit = hit("000000000000000000000001", 0.75);
        let json = serde_json::to_value(&hit).unwrap();
        assert_eq!(json["id"], serde_json::json!("000000000000000000000001"));
        assert_eq!(json["text"], serde_json::json!("some content"));
        assert_eq!(json["user_id"], serde_json::json!("alice"));
        assert_eq!(json["score"], serde_json::json!(0.75));
        assert_eq!(json["similarity"], serde_json::json!(0.8_f32));
        assert_eq!(json["keyword"], serde_json::json!(0.4_f32));
        assert_eq!(json["importance"], serde_json::json!(0.5_f32));
        assert_eq!(json["recency"], serde_json::json!(0.9_f32));
        let restored: RecallHit = serde_json::from_value(json).unwrap();
        assert_eq!(restored, hit);
    }
    #[test]
    fn add_outcome_len_counts_every_bucket() {
        let outcome = AddOutcome::default();
        assert!(outcome.is_empty());
        assert_eq!(outcome.len(), 0);
        let outcome = AddOutcome {
            added: vec![hit("a", 0.1).record],
            merged: vec![hit("b", 0.1).record],
            deleted: vec!["c".to_owned()],
            skipped: vec!["d".to_owned()],
            degraded: None,
        };
        assert_eq!(outcome.len(), 4);
        assert!(!outcome.is_empty());
        let json = serde_json::to_value(&outcome).unwrap();
        assert_eq!(json["skipped"], serde_json::json!(["d"]));
        assert!(
            json.get("degraded").is_none(),
            "a complete outcome must not serialize a degraded reason"
        );
        let restored: AddOutcome = serde_json::from_value(json).unwrap();
        assert_eq!(restored, outcome);
    }
    #[test]
    fn add_outcome_degraded_reason_is_optional_and_round_trips() {
        let outcome = AddOutcome {
            added: vec![hit("a", 0.1).record],
            degraded: Some("candidate scan failed".to_owned()),
            ..AddOutcome::default()
        };
        assert_eq!(outcome.len(), 1, "degraded is a diagnostic, not an entry");
        let json = serde_json::to_value(&outcome).unwrap();
        assert_eq!(
            json["degraded"],
            serde_json::json!("candidate scan failed"),
            "a degraded outcome must serialize its reason"
        );
        let restored: AddOutcome = serde_json::from_value(json).unwrap();
        assert_eq!(restored, outcome);
        let tolerant: AddOutcome =
            serde_json::from_value(serde_json::json!({ "unknown_field": true })).unwrap();
        assert_eq!(tolerant, AddOutcome::default());
    }
    #[test]
    fn memory_stats_serde_round_trip() {
        let stats = MemoryStats {
            total: 2,
            by_kind: BTreeMap::from([("semantic_memory".to_owned(), 2)]),
            oldest: Some(DateTime::from_timestamp(1_700_000_000, 0).unwrap()),
            newest: Some(DateTime::from_timestamp(1_700_000_100, 0).unwrap()),
            total_tokens: 12,
        };
        let json = serde_json::to_value(&stats).unwrap();
        assert_eq!(json["by_kind"]["semantic_memory"], serde_json::json!(2));
        let restored: MemoryStats = serde_json::from_value(json).unwrap();
        assert_eq!(restored, stats);
        let empty = MemoryStats::default();
        assert_eq!(empty.total, 0);
        assert!(empty.by_kind.is_empty());
        assert!(empty.oldest.is_none());
    }
    #[test]
    fn add_mem_result_iterates_and_counts_records() {
        let mut result = AddMemResult::default();
        assert!(result.is_empty());
        assert_eq!(result.records().count(), 0);
        result
            .variables
            .insert("theme".to_owned(), "dark".to_owned());
        assert!(result.is_empty());
        result.user_profile.push(hit("a", 0.1).record);
        result.summary.push(hit("b", 0.1).record);
        result.coding.push(hit("c", 0.1).record);
        assert_eq!(result.len(), 3);
        assert!(!result.is_empty());
        let ids: Vec<&str> = result.records().map(|record| record.id.as_str()).collect();
        assert_eq!(ids, vec!["a", "b", "c"]);
        let json = serde_json::to_value(&result).unwrap();
        assert_eq!(json["variables"]["theme"], serde_json::json!("dark"));
        let restored: AddMemResult = serde_json::from_value(json).unwrap();
        assert_eq!(restored, result);
    }
    #[test]
    fn add_mem_result_bookkeeping_round_trips_and_stays_backward_compatible() {
        let mut result = AddMemResult {
            skipped: 2,
            deleted: 1,
            degraded: Some("model extraction failed".to_owned()),
            ..Default::default()
        };
        let json = serde_json::to_value(&result).unwrap();
        assert_eq!(json["skipped"], serde_json::json!(2));
        assert_eq!(json["deleted"], serde_json::json!(1));
        assert_eq!(
            json["degraded"],
            serde_json::json!("model extraction failed")
        );
        // Bookkeeping is not a record: `len`/`is_empty` keep their documented meaning.
        assert_eq!(result.len(), 0);
        assert!(result.is_empty());
        let restored: AddMemResult = serde_json::from_value(json).unwrap();
        assert_eq!(restored, result);

        // A healthy result omits the reason and decodes a legacy payload that lacks
        // every bookkeeping field.
        result.degraded = None;
        let json = serde_json::to_value(&result).unwrap();
        assert!(json.get("degraded").is_none(), "expected no degraded field");
        let legacy: AddMemResult =
            serde_json::from_value(serde_json::json!({"variables": {"theme": "dark"}})).unwrap();
        assert_eq!(legacy.skipped, 0);
        assert_eq!(legacy.deleted, 0);
        assert!(legacy.degraded.is_none());
        assert!(legacy.is_empty());
    }
}
