//! Persisted subagent buckets: per-instance transcript, status, and configuration.
//!
//! Ports the bucket shape of Python
//! `openjiuwen/harness/subagent_runtime/persistence.py`: one parent session owns a
//! `"subagents"` namespace holding a revision counter and the records of every child it
//! spawned. The Python bucket keeps separate `records`, `turns` and `activities` maps; the
//! Rust runtime flattens turns and activity milestones into one bounded transcript per
//! record, so a bucket here is `{"revision": u64, "records": {"<id>": record}}`.
//!
//! Record retention follows the Python FIFO bound: [`max_persisted_records`] derives the cap
//! from the live-instance limit and [`PersistedSubAgentBuckets::trim`] drops closed records
//! before live ones, oldest first. There is no session object in this layer, so the bucket is
//! a plain serde value that callers persist with [`save_bucket_file`] and restore with
//! [`load_bucket_file`].
use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::Path;

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::quota::SubagentActivity;
use super::transcript::TranscriptEntry;
use super::SubagentStatus;
use crate::schema::SubAgentConfig;

/// Namespace key of the subagent bucket (Python `SUBAGENTS_KEY`).
pub const SUBAGENTS_KEY: &str = "subagents";

/// Upper bound on persisted records for one parent session (`max(max_subagents * 4, 40)`).
///
/// Python parity: `max_persisted_records`. The floor of 40 keeps a short history even when a
/// parent allows very few live children.
#[must_use]
pub fn max_persisted_records(max_subagents: usize) -> usize {
    (max_subagents * 4).max(40)
}

/// One persisted subagent: identity, quota bookkeeping, status/activity, the configuration the
/// child was built from, and its bounded transcript. All timestamps are epoch milliseconds.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct PersistedSubAgentRecord {
    /// Instance id (doubles as the child session id prefix).
    pub subagent_id: String,
    /// Session id of the child agent.
    pub sub_session_id: String,
    /// Registered sub-agent type the child was built from.
    pub subagent_type: String,
    /// Human-readable name shown in prompts and transcripts.
    pub display_name: Option<String>,
    /// Role description given to the child in the parent's prompt.
    pub role: Option<String>,
    /// The task the child was spawned with.
    pub task_description: String,
    /// Lifecycle status of the instance when the bucket was written.
    pub status: SubagentStatus,
    /// Derived lifecycle phase of the instance when the bucket was written.
    pub activity: SubagentActivity,
    /// Nesting depth (`1` for a top-level spawn).
    pub depth: usize,
    /// The instance that spawned this one, when nested.
    pub parent_subagent_id: Option<String>,
    /// Spawn sequence number (stable ordering tie-break).
    pub spawn_seq: u64,
    /// Id of the child agent built by the factory.
    pub agent_id: String,
    /// Last final answer the child produced.
    pub output: Option<String>,
    /// Last error the child reported.
    pub error: Option<String>,
    /// When the instance was reserved, in epoch milliseconds.
    pub created_at_ms: i64,
    /// When the instance last changed, in epoch milliseconds.
    pub updated_at_ms: i64,
    /// When the instance was last used for a turn, in epoch milliseconds.
    pub last_used_at_ms: i64,
    /// When the activity phase last changed, in epoch milliseconds.
    pub last_activity_at_ms: i64,
    /// When the instance was closed, in epoch milliseconds.
    pub closed_at_ms: Option<i64>,
    /// Why the instance was closed (Python `manual`, `parent_ended`, `evicted`).
    pub closed_reason: Option<String>,
    /// The configuration the child agent was built from.
    pub config: SubAgentConfig,
    /// Bounded transcript of the instance (turns and activity milestones).
    pub transcript: Vec<TranscriptEntry>,
}

impl PersistedSubAgentRecord {
    /// Whether the instance is closed, by activity phase or by status.
    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.activity == SubagentActivity::Closed || self.status == SubagentStatus::Closed
    }

    /// The ordering timestamp [`PersistedSubAgentBuckets::trim`] uses.
    ///
    /// `closed_at_ms` wins when set; otherwise the first non-zero of `last_activity_at_ms`,
    /// `updated_at_ms` and `created_at_ms`, so a record with unset timestamps falls back to
    /// the remaining ones instead of sorting as epoch zero.
    #[must_use]
    pub fn sort_key_ms(&self) -> i64 {
        if let Some(closed_at_ms) = self.closed_at_ms {
            return closed_at_ms;
        }
        if self.last_activity_at_ms != 0 {
            return self.last_activity_at_ms;
        }
        if self.updated_at_ms != 0 {
            return self.updated_at_ms;
        }
        self.created_at_ms
    }
}

/// Trim ordering of one record: sort-key timestamp, then `updated_at_ms`, then `created_at_ms`.
type TrimOrder = (i64, i64, i64);

fn trim_order(record: &PersistedSubAgentRecord) -> TrimOrder {
    (
        record.sort_key_ms(),
        record.updated_at_ms,
        record.created_at_ms,
    )
}

/// Persisted subagent namespace of one parent session.
///
/// Python parity: the `"subagents"` bucket written under [`SUBAGENTS_KEY`]. `revision` counts
/// mutating operations so a later merge can tell which side is newer.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct PersistedSubAgentBuckets {
    /// Monotone revision counter, bumped by every mutation.
    pub revision: u64,
    /// Persisted records, ordered by instance id.
    pub records: BTreeMap<String, PersistedSubAgentRecord>,
}

impl PersistedSubAgentBuckets {
    /// Creates an empty bucket at revision `0`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the number of persisted records.
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether no record is persisted.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Returns the persisted ids in sorted (map) order.
    #[must_use]
    pub fn ids(&self) -> Vec<String> {
        self.records.keys().cloned().collect()
    }

    /// Returns the record persisted as `id`.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<&PersistedSubAgentRecord> {
        self.records.get(id)
    }

    /// Inserts `record` under its `subagent_id`, replacing any record with that id.
    ///
    /// The revision is bumped on every call, replacement included.
    pub fn insert(&mut self, record: PersistedSubAgentRecord) {
        self.records.insert(record.subagent_id.clone(), record);
        self.revision = self.revision.wrapping_add(1);
    }

    /// Removes and returns the record persisted as `id`.
    ///
    /// The revision is bumped only when a record was actually removed.
    pub fn remove(&mut self, id: &str) -> Option<PersistedSubAgentRecord> {
        let removed = self.records.remove(id);
        if removed.is_some() {
            self.revision = self.revision.wrapping_add(1);
        }
        removed
    }

    /// Merges `other` into `self`: incoming records replace same-id records, other ids are added;
    /// `revision` becomes `max(self.revision, other.revision) + 1`.
    pub fn merge(&mut self, other: Self) {
        self.records.extend(other.records);
        self.revision = self.revision.max(other.revision).wrapping_add(1);
    }

    /// Keeps at most `max_records` records: drops `Closed` records first (oldest by
    /// `closed_at_ms`, then `updated_at_ms`, then `created_at_ms`, then id), then, if still over
    /// the limit, the oldest remaining records by the same key. Returns the number dropped.
    ///
    /// `max_records == 0` clears the bucket. A bucket within the limit is left untouched, and
    /// the revision is not bumped.
    pub fn trim(&mut self, max_records: usize) -> usize {
        if self.records.len() <= max_records {
            return 0;
        }
        let mut closed: Vec<(TrimOrder, String)> = Vec::new();
        let mut live: Vec<(TrimOrder, String)> = Vec::new();
        for (id, record) in &self.records {
            let entry = (trim_order(record), id.clone());
            if record.is_closed() {
                closed.push(entry);
            } else {
                live.push(entry);
            }
        }
        let by_order = |a: &(TrimOrder, String), b: &(TrimOrder, String)| {
            a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1))
        };
        closed.sort_by(by_order);
        live.sort_by(by_order);
        let excess = self.records.len() - max_records;
        let mut dropped = 0;
        for (_, id) in closed.into_iter().chain(live).take(excess) {
            if self.records.remove(&id).is_some() {
                dropped += 1;
            }
        }
        dropped
    }

    /// Serializes the bucket to JSON (`{"revision": u64, "records": {"<id>": {...}}}`).
    ///
    /// # Errors
    /// `Error::Validation("failed to encode subagent bucket: <detail>")` when serialization fails.
    pub fn to_json(&self) -> Result<Value> {
        serde_json::to_value(self).map_err(|error| {
            Error::validation(format!("failed to encode subagent bucket: {error}"))
        })
    }

    /// Parses a bucket from JSON, tolerating unknown and missing keys.
    ///
    /// # Errors
    /// `Error::Validation("invalid subagent bucket: <detail>")` when the value is not a bucket.
    pub fn from_json(value: &Value) -> Result<Self> {
        serde_json::from_value(value.clone())
            .map_err(|error| Error::validation(format!("invalid subagent bucket: {error}")))
    }

    /// Serializes the bucket to pretty-printed JSON.
    ///
    /// # Errors
    /// `Error::Validation("failed to encode subagent bucket: <detail>")` when serialization fails.
    pub fn to_json_string(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(|error| {
            Error::validation(format!("failed to encode subagent bucket: {error}"))
        })
    }

    /// Parses a bucket from JSON text, tolerating unknown and missing keys.
    ///
    /// # Errors
    /// `Error::Validation("invalid subagent bucket: <detail>")` when the text is not a bucket.
    pub fn from_json_string(json: &str) -> Result<Self> {
        serde_json::from_str(json)
            .map_err(|error| Error::validation(format!("invalid subagent bucket: {error}")))
    }
}

/// Reads a bucket from `path`, returning an empty bucket when the file does not exist.
///
/// The whole file is read and parsed here; a malformed file is reported against its path.
///
/// # Errors
/// `Error::Session("failed to read subagent bucket <path>: <detail>")` on IO or parse failure.
pub fn load_bucket_file(path: &Path) -> Result<PersistedSubAgentBuckets> {
    let fail = |detail: &dyn std::fmt::Display| {
        Error::session(format!(
            "failed to read subagent bucket {}: {detail}",
            path.display()
        ))
    };
    match fs::read_to_string(path) {
        Ok(json) => PersistedSubAgentBuckets::from_json_string(&json).map_err(|error| fail(&error)),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            Ok(PersistedSubAgentBuckets::default())
        }
        Err(error) => Err(fail(&error)),
    }
}

/// Writes `buckets` to `path` as pretty JSON, creating parent directories.
///
/// # Errors
/// `Error::Session("failed to write subagent bucket <path>: <detail>")` on IO or encode failure.
pub fn save_bucket_file(path: &Path, buckets: &PersistedSubAgentBuckets) -> Result<()> {
    let fail = |detail: &dyn std::fmt::Display| {
        Error::session(format!(
            "failed to write subagent bucket {}: {detail}",
            path.display()
        ))
    };
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|error| fail(&error))?;
        }
    }
    let json = buckets.to_json_string().map_err(|error| fail(&error))?;
    fs::write(path, format!("{json}\n")).map_err(|error| fail(&error))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use serde_json::json;

    use super::*;

    fn record(id: &str) -> PersistedSubAgentRecord {
        PersistedSubAgentRecord {
            subagent_id: id.to_owned(),
            sub_session_id: format!("parent_{id}"),
            subagent_type: "code_agent".to_owned(),
            display_name: Some("Coder".to_owned()),
            role: Some("implementer".to_owned()),
            task_description: "do the work".to_owned(),
            status: SubagentStatus::Completed,
            activity: SubagentActivity::Idle,
            depth: 1,
            parent_subagent_id: None,
            spawn_seq: 3,
            agent_id: format!("agent_{id}"),
            output: Some("done".to_owned()),
            error: None,
            created_at_ms: 1_000,
            updated_at_ms: 2_000,
            last_used_at_ms: 2_000,
            last_activity_at_ms: 2_000,
            closed_at_ms: None,
            closed_reason: None,
            config: SubAgentConfig::default(),
            transcript: Vec::new(),
        }
    }

    fn closed(mut record: PersistedSubAgentRecord, closed_at_ms: i64) -> PersistedSubAgentRecord {
        record.status = SubagentStatus::Closed;
        record.activity = SubagentActivity::Closed;
        record.closed_at_ms = Some(closed_at_ms);
        record.closed_reason = Some("evicted".to_owned());
        record
    }

    fn full_record() -> PersistedSubAgentRecord {
        let mut record = record("sub_1");
        record.parent_subagent_id = Some("root".to_owned());
        record.closed_reason = Some("evicted".to_owned());
        record.closed_at_ms = Some(3_000);
        record.config.factory_name = Some("code_agent".to_owned());
        record.transcript = vec![
            TranscriptEntry::spawn("sub_1", "do the work", 1_100),
            TranscriptEntry::assistant_turn("sub_1", "done", 2_900),
        ];
        record
    }

    fn bucket_with(records: Vec<PersistedSubAgentRecord>) -> PersistedSubAgentBuckets {
        let mut buckets = PersistedSubAgentBuckets::new();
        for record in records {
            buckets.insert(record);
        }
        buckets
    }

    #[test]
    fn json_round_trip_keeps_the_documented_keys() {
        let mut buckets = PersistedSubAgentBuckets::new();
        buckets.insert(full_record());

        let value = buckets.to_json().expect("encode");
        assert!(value.get("revision").is_some());
        assert!(value.get("records").is_some());
        let record = &value["records"]["sub_1"];
        for key in [
            "subagent_id",
            "sub_session_id",
            "subagent_type",
            "display_name",
            "role",
            "task_description",
            "status",
            "activity",
            "depth",
            "parent_subagent_id",
            "spawn_seq",
            "agent_id",
            "output",
            "error",
            "created_at_ms",
            "updated_at_ms",
            "last_used_at_ms",
            "last_activity_at_ms",
            "closed_at_ms",
            "closed_reason",
            "config",
            "transcript",
        ] {
            assert!(record.get(key).is_some(), "missing record key {key}");
        }
        assert_eq!(record["status"], json!("completed"));
        assert_eq!(record["activity"], json!("idle"));
        assert_eq!(record["config"]["factory_name"], json!("code_agent"));
        assert_eq!(record["transcript"].as_array().map(Vec::len), Some(2));

        let restored = PersistedSubAgentBuckets::from_json(&value).expect("decode");
        assert_eq!(restored, buckets);
        let restored_text = PersistedSubAgentBuckets::from_json_string(
            &buckets.to_json_string().expect("encode string"),
        )
        .expect("decode string");
        assert_eq!(restored_text, buckets);
    }

    #[test]
    fn from_json_tolerates_missing_and_unknown_keys() {
        let empty = PersistedSubAgentBuckets::from_json(&json!({ "records": {} })).expect("decode");
        assert_eq!(empty, PersistedSubAgentBuckets::default());
        assert!(empty.is_empty());
        assert_eq!(empty.revision, 0);

        let bare = PersistedSubAgentBuckets::from_json(&json!({
            "records": { "sub_1": {} },
            "unknown": [1, 2, 3],
        }))
        .expect("decode");
        assert_eq!(bare.len(), 1);
        let record = bare.get("sub_1").expect("record");
        assert_eq!(record.subagent_id, "");
        assert_eq!(record.status, SubagentStatus::PendingInit);
        assert_eq!(record.activity, SubagentActivity::Queued);
        assert_eq!(record.config.agent_card.name, "");
        assert_eq!(record.transcript.len(), 0);
    }

    #[test]
    fn from_json_rejects_malformed_buckets() {
        for bad in [json!({ "records": 7 }), json!("not a bucket")] {
            let error = PersistedSubAgentBuckets::from_json(&bad).expect_err("must reject");
            assert!(matches!(error, Error::Validation(_)));
            assert!(
                error
                    .to_string()
                    .starts_with("validation: invalid subagent bucket:"),
                "unexpected message: {error}"
            );
        }
    }

    #[test]
    fn insert_replaces_by_id_and_bumps_the_revision() {
        let mut buckets = PersistedSubAgentBuckets::new();
        assert!(buckets.is_empty());
        assert_eq!(buckets.revision, 0);

        buckets.insert(record("worker"));
        assert_eq!(buckets.len(), 1);
        assert_eq!(buckets.revision, 1);

        let mut replacement = record("worker");
        replacement.task_description = "replacement".to_owned();
        buckets.insert(replacement);
        assert_eq!(buckets.len(), 1);
        assert_eq!(buckets.revision, 2);
        assert_eq!(
            buckets
                .get("worker")
                .map(|record| record.task_description.as_str()),
            Some("replacement")
        );
    }

    #[test]
    fn ids_are_sorted_and_remove_bumps_only_when_present() {
        let mut buckets = bucket_with(vec![record("b"), record("a"), record("c")]);
        assert_eq!(buckets.ids(), vec!["a", "b", "c"]);
        let revision = buckets.revision;

        assert!(buckets.remove("missing").is_none());
        assert_eq!(buckets.revision, revision);

        let removed = buckets.remove("b").expect("removed");
        assert_eq!(removed.subagent_id, "b");
        assert_eq!(buckets.revision, revision + 1);
        assert_eq!(buckets.ids(), vec!["a", "c"]);
    }

    #[test]
    fn merge_replaces_by_id_and_takes_the_newer_revision_plus_one() {
        let mut left = bucket_with(vec![record("shared")]);
        left.revision = 5;
        let mut right = bucket_with(vec![record("shared"), record("extra")]);
        right.revision = 2;
        if let Some(record) = right.records.get_mut("shared") {
            record.output = Some("from right".to_owned());
        }

        left.merge(right);

        assert_eq!(left.revision, 6);
        assert_eq!(left.ids(), vec!["extra", "shared"]);
        assert_eq!(
            left.get("shared")
                .and_then(|record| record.output.as_deref()),
            Some("from right")
        );

        let mut backwards = bucket_with(vec![record("shared")]);
        backwards.revision = 9;
        let mut forwards = PersistedSubAgentBuckets::new();
        forwards.revision = 1;
        backwards.merge(forwards);
        assert_eq!(backwards.revision, 10);
    }

    #[test]
    fn trim_drops_closed_records_before_live_ones() {
        let mut live_old = record("live_old");
        live_old.last_activity_at_ms = 5;
        let mut live_new = record("live_new");
        live_new.last_activity_at_ms = 50;
        let buckets = bucket_with(vec![
            live_old,
            closed(record("closed_old"), 10),
            live_new,
            closed(record("closed_new"), 30),
        ]);
        assert_eq!(buckets.len(), 4);

        let mut trimmed = buckets.clone();
        assert_eq!(trimmed.trim(3), 1);
        assert_eq!(trimmed.ids(), vec!["closed_new", "live_new", "live_old"]);

        let mut trimmed = buckets.clone();
        assert_eq!(trimmed.trim(2), 2);
        assert!(trimmed.ids().iter().all(|id| !id.starts_with("closed")));
        assert_eq!(trimmed.ids(), vec!["live_new", "live_old"]);

        let mut trimmed = buckets.clone();
        assert_eq!(trimmed.trim(1), 3);
        assert_eq!(trimmed.ids(), vec!["live_new"]);
    }

    #[test]
    fn trim_orders_by_closed_then_updated_then_created_then_id() {
        let mut by_created = bucket_with(vec![closed(record("c1"), 10), closed(record("c2"), 10)]);
        for (id, created_at_ms) in [("c1", 30), ("c2", 31)] {
            if let Some(record) = by_created.records.get_mut(id) {
                record.updated_at_ms = 20;
                record.created_at_ms = created_at_ms;
            }
        }
        assert_eq!(by_created.trim(1), 1);
        assert_eq!(by_created.ids(), vec!["c2"]);

        let mut by_updated = bucket_with(vec![
            closed(record("u_old"), 10),
            closed(record("u_new"), 10),
        ]);
        for record in by_updated.records.values_mut() {
            record.updated_at_ms = 20;
            record.created_at_ms = 30;
        }
        if let Some(record) = by_updated.records.get_mut("u_new") {
            record.updated_at_ms = 21;
        }
        assert_eq!(by_updated.trim(1), 1);
        assert_eq!(by_updated.ids(), vec!["u_new"]);

        let mut by_id = bucket_with(vec![
            closed(record("b_dup"), 10),
            closed(record("a_dup"), 10),
        ]);
        for record in by_id.records.values_mut() {
            record.updated_at_ms = 20;
            record.created_at_ms = 30;
        }
        assert_eq!(by_id.trim(1), 1);
        assert_eq!(by_id.ids(), vec!["b_dup"]);
    }

    #[test]
    fn trim_zero_clears_and_trim_within_the_limit_is_a_no_op() {
        let buckets = bucket_with(vec![record("a"), record("b")]);

        let mut untouched = buckets.clone();
        let revision = untouched.revision;
        assert_eq!(untouched.trim(2), 0);
        assert_eq!(untouched.trim(10), 0);
        assert_eq!(untouched, buckets);
        assert_eq!(untouched.revision, revision);

        let mut cleared = buckets.clone();
        assert_eq!(cleared.trim(0), 2);
        assert!(cleared.is_empty());
    }

    #[test]
    fn sort_key_falls_back_through_the_timestamps() {
        let mut record = PersistedSubAgentRecord::default();
        assert_eq!(record.sort_key_ms(), 0);
        record.created_at_ms = 7;
        assert_eq!(record.sort_key_ms(), 7);
        record.updated_at_ms = 8;
        assert_eq!(record.sort_key_ms(), 8);
        record.last_activity_at_ms = 9;
        assert_eq!(record.sort_key_ms(), 9);
        record.closed_at_ms = Some(10);
        assert_eq!(record.sort_key_ms(), 10);
    }

    #[test]
    fn max_persisted_records_matches_the_python_bound() {
        for max_subagents in 0..=20 {
            assert_eq!(
                max_persisted_records(max_subagents),
                (max_subagents * 4).max(40)
            );
        }
        assert_eq!(max_persisted_records(0), 40);
        assert_eq!(max_persisted_records(10), 40);
        assert_eq!(max_persisted_records(20), 80);
    }

    #[test]
    fn save_and_load_round_trip_creates_parent_directories() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir
            .path()
            .join("parent")
            .join("nested")
            .join("subagents.json");
        let buckets = bucket_with(vec![full_record(), record("sub_2")]);

        save_bucket_file(&path, &buckets).expect("save");
        assert!(path.exists());
        assert_eq!(load_bucket_file(&path).expect("load"), buckets);
    }

    #[test]
    fn load_of_a_missing_file_yields_an_empty_bucket() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("absent.json");
        let buckets = load_bucket_file(&path).expect("load missing");
        assert_eq!(buckets, PersistedSubAgentBuckets::default());
        assert!(buckets.is_empty());
    }

    #[test]
    fn load_of_an_invalid_file_reports_the_path_as_a_session_error() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("broken.json");
        std::fs::write(&path, "{ not json").expect("write");

        let error = load_bucket_file(&path).expect_err("must reject");
        assert!(matches!(error, Error::Session(_)), "unexpected: {error}");
        let message = error.to_string();
        assert!(message.starts_with("session: "), "unexpected: {message}");
        assert!(
            message.contains("failed to read subagent bucket"),
            "unexpected: {message}"
        );
        assert!(
            message.contains(&path.display().to_string()),
            "unexpected: {message}"
        );
    }
}
