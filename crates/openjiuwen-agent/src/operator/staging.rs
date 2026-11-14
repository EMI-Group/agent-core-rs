//! Caller-owned pending-change lifecycle for staged skill-experience records.
//!
//! Ports the approval half of `openjiuwen/agent_evolving/experience/`:
//! [`LocalApplyPreview`] (`lifecycle.py`), `PendingChange` (`types.py`), and
//! `commit_pending_change` / `reject_pending_change` (`common.py`).
//!
//! The split follows Python: the
//! [`SkillExperienceOperator`](super::skill_experience::SkillExperienceOperator)
//! only produces a preview, the [`PendingChangeStore`] owns the staged snapshot
//! and the approve/reject/retry transitions, and a [`SkillExperienceStore`] sink
//! performs the per-record writes. A failed commit keeps the unwritten tail so a
//! retry resumes exactly where the store stopped.
//!
//! The pending snapshot map is process-local and deliberately not serialized,
//! matching Python's `ExperienceManager._pending_approval_snapshots`.
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::base::ApplyResult;
use super::protocol::{LOCAL_APPLY_COMPLETED, SKILL_EXPERIENCE_ENTRY};

/// Prefix of every generated pending-change identifier.
const CHANGE_ID_PREFIX: &str = "skill_evolve_";

/// Stable local-apply preview contract (`LocalApplyPreview` in Python).
///
/// [`Self::records`] holds only the records reported by applied results, while
/// [`Self::apply_results`] keeps the full result list (rejections included) so
/// a host can explain what was dropped.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LocalApplyPreview {
    /// Skill the records belong to.
    pub skill_name: String,
    /// Records produced by the applied results, in result order.
    #[serde(default)]
    pub records: Vec<Value>,
    /// Record kind of the produced records.
    #[serde(default = "default_change_type")]
    pub change_type: String,
    /// Lifecycle stage the preview reached.
    #[serde(default = "default_lifecycle_stage")]
    pub lifecycle_stage: String,
    /// Every apply result the preview was built from.
    #[serde(default)]
    pub apply_results: Vec<ApplyResult>,
}

/// Default record kind of a skill-experience preview.
fn default_change_type() -> String {
    SKILL_EXPERIENCE_ENTRY.to_owned()
}

/// Lifecycle stage a finished local apply reaches.
fn default_lifecycle_stage() -> String {
    LOCAL_APPLY_COMPLETED.to_owned()
}

/// Builds the preview contract from `apply_results`.
///
/// Results with `applied == false` contribute nothing; the last non-`None`
/// `change_type` among the applied results wins, defaulting to
/// [`SKILL_EXPERIENCE_ENTRY`].
///
/// # Errors
///
/// Returns [`Error::Validation`] when an applied result reports a
/// `lifecycle_stage` other than [`LOCAL_APPLY_COMPLETED`].
pub fn build_local_apply_preview(
    skill_name: &str,
    apply_results: &[ApplyResult],
) -> Result<LocalApplyPreview> {
    let mut records = Vec::new();
    let mut change_type = default_change_type();
    for result in apply_results.iter().filter(|result| result.applied) {
        if let Some(stage) = &result.lifecycle_stage {
            if stage != LOCAL_APPLY_COMPLETED {
                return Err(Error::validation(format!(
                    "unsupported apply lifecycle stage for {skill_name}: {stage}"
                )));
            }
        }
        records.extend(result.records.iter().cloned());
        if let Some(kind) = &result.change_type {
            change_type.clone_from(kind);
        }
    }
    Ok(LocalApplyPreview {
        skill_name: skill_name.to_owned(),
        records,
        change_type,
        lifecycle_stage: default_lifecycle_stage(),
        apply_results: apply_results.to_vec(),
    })
}

/// Sink for approved experience records (`EvolutionStore.append_record` in
/// Python).
#[async_trait]
pub trait SkillExperienceStore: Send + Sync {
    /// Appends one record to `skill_name`'s experience list.
    ///
    /// # Errors
    ///
    /// Returns an error when the record cannot be persisted; the caller keeps
    /// the record staged for retry.
    async fn append_record(&self, skill_name: &str, record: &Value) -> Result<()>;
}

/// Deterministic in-memory sink for tests and offline hosts.
///
/// Writes are recorded in call order. [`Self::with_failure_after`] injects a
/// permanent failure once `limit` records were written, which exercises the
/// partial-commit and retry paths.
#[derive(Debug, Default)]
pub struct InMemorySkillExperienceStore {
    records: Mutex<Vec<(String, Value)>>,
    successes: AtomicUsize,
    failure_after: Option<usize>,
}

impl InMemorySkillExperienceStore {
    /// Creates an empty sink that accepts every write.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a sink that fails every append after `limit` successful writes.
    #[must_use]
    pub fn with_failure_after(limit: usize) -> Self {
        Self {
            failure_after: Some(limit),
            ..Self::default()
        }
    }

    /// The written `(skill_name, record)` pairs in call order.
    #[must_use]
    pub fn records(&self) -> Vec<(String, Value)> {
        self.records.lock().clone()
    }

    /// Number of written records.
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.lock().len()
    }

    /// Whether nothing was written yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.lock().is_empty()
    }
}

#[async_trait]
impl SkillExperienceStore for InMemorySkillExperienceStore {
    async fn append_record(&self, skill_name: &str, record: &Value) -> Result<()> {
        if let Some(limit) = self.failure_after {
            if self.successes.load(Ordering::Relaxed) >= limit {
                return Err(Error::memory(format!(
                    "injected append failure for skill '{skill_name}' after {limit} record(s)"
                )));
            }
        }
        self.records
            .lock()
            .push((skill_name.to_owned(), record.clone()));
        self.successes.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

/// Lifecycle status of one staged pending change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PendingChangeStatus {
    /// Staged and waiting for the host to approve or reject it.
    PendingApproval,
    /// Every record was written and the staged entry was removed.
    Persisted,
    /// Some records were written; the unwritten tail stays staged.
    Partial,
    /// The host rejected the change; nothing was written.
    Rejected,
}

impl PendingChangeStatus {
    /// The `snake_case` wire name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PendingApproval => "pending_approval",
            Self::Persisted => "persisted",
            Self::Partial => "partial",
            Self::Rejected => "rejected",
        }
    }
}

/// Snapshot of staged experience records awaiting approval.
#[derive(Debug, Clone, PartialEq)]
pub struct PendingChange {
    /// Identifier of the staged change (`skill_evolve_<8 hex digits>`).
    pub change_id: String,
    /// Skill the records belong to.
    pub skill_name: String,
    /// Records still waiting to be written.
    pub records: Vec<Value>,
    /// Record kind of the staged records.
    pub change_type: String,
    /// Attribution copied from the preview's applied results.
    pub metadata: BTreeMap<String, Value>,
    /// Monotonic staging sequence number (strictly increasing per store).
    pub created_seq: u64,
    /// Current lifecycle status.
    pub status: PendingChangeStatus,
}

/// Outcome of one approve/reject/retry transition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PendingChangeOutcome {
    /// Identifier of the staged change.
    pub change_id: String,
    /// Skill the records belong to.
    pub skill_name: String,
    /// Status reached by the transition.
    pub status: PendingChangeStatus,
    /// Records written by this transition.
    pub applied_count: usize,
    /// Records still staged afterwards.
    pub pending_count: usize,
    /// Records rejected by this transition.
    pub rejected_count: usize,
    /// One message per failed write.
    pub errors: Vec<String>,
}

/// Caller-owned store of staged pending changes.
///
/// The snapshot map is process-local and not serialized; the write sink is a
/// [`SkillExperienceStore`] handed to [`PendingChangeStore::with_store`] or
/// swapped through [`PendingChangeStore::set_store`].
#[derive(Default)]
pub struct PendingChangeStore {
    pending: Mutex<BTreeMap<String, PendingChange>>,
    sink: RwLock<Option<Arc<dyn SkillExperienceStore>>>,
    counter: AtomicU64,
}

impl PendingChangeStore {
    /// Creates an empty store without a write sink.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Installs the write sink used by [`Self::approve`] and [`Self::retry`].
    #[must_use]
    pub fn with_store(self, sink: Arc<dyn SkillExperienceStore>) -> Self {
        *self.sink.write() = Some(sink);
        self
    }

    /// Replaces the write sink (for example after a failed commit).
    pub fn set_store(&self, sink: Arc<dyn SkillExperienceStore>) {
        *self.sink.write() = Some(sink);
    }

    /// Stages `preview` as a pending change and returns the stored snapshot.
    pub fn stage(&self, preview: &LocalApplyPreview) -> PendingChange {
        let seq = self.counter.fetch_add(1, Ordering::Relaxed) + 1;
        let change = PendingChange {
            change_id: format!("{CHANGE_ID_PREFIX}{seq:08x}"),
            skill_name: preview.skill_name.clone(),
            records: preview.records.clone(),
            change_type: preview.change_type.clone(),
            metadata: preview_metadata(preview),
            created_seq: seq,
            status: PendingChangeStatus::PendingApproval,
        };
        self.pending
            .lock()
            .insert(change.change_id.clone(), change.clone());
        change
    }

    /// Every staged change, oldest first.
    #[must_use]
    pub fn list(&self) -> Vec<PendingChange> {
        let mut pending: Vec<PendingChange> = self.pending.lock().values().cloned().collect();
        pending.sort_by_key(|change| change.created_seq);
        pending
    }

    /// One staged change by identifier.
    #[must_use]
    pub fn get(&self, change_id: &str) -> Option<PendingChange> {
        self.pending.lock().get(change_id).cloned()
    }

    /// Writes every remaining record of `change_id` through the sink.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] for an unknown identifier and
    /// [`Error::Config`] when no sink was installed. A failed write is not an
    /// error: the unwritten tail stays staged and is reported as
    /// [`PendingChangeStatus::Partial`].
    pub async fn approve(&self, change_id: &str) -> Result<PendingChangeOutcome> {
        self.commit(change_id).await
    }

    /// Discards `change_id` without writing anything.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] for an unknown identifier.
    #[allow(clippy::unused_async, clippy::unused_async_trait_impl)] // mirrors Python's `async` approve/retry surface
    pub async fn reject(&self, change_id: &str) -> Result<PendingChangeOutcome> {
        let pending = self
            .pending
            .lock()
            .remove(change_id)
            .ok_or_else(|| unknown_change(change_id))?;
        Ok(PendingChangeOutcome {
            change_id: pending.change_id,
            skill_name: pending.skill_name,
            status: PendingChangeStatus::Rejected,
            applied_count: 0,
            pending_count: 0,
            rejected_count: pending.records.len(),
            errors: Vec::new(),
        })
    }

    /// Retries the unwritten tail of a partially committed `change_id`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] for an unknown identifier and
    /// [`Error::Config`] when no sink was installed. Write failures follow
    /// [`Self::approve`] semantics.
    pub async fn retry(&self, change_id: &str) -> Result<PendingChangeOutcome> {
        self.commit(change_id).await
    }

    /// Writes the records of `change_id` one at a time, keeping the tail staged
    /// when a write fails (`commit_pending_change` in Python).
    async fn commit(&self, change_id: &str) -> Result<PendingChangeOutcome> {
        let Some(sink) = self.sink() else {
            return Err(Error::config(
                "PendingChangeStore has no SkillExperienceStore sink; call with_store/set_store",
            ));
        };
        let mut pending = self
            .get(change_id)
            .ok_or_else(|| unknown_change(change_id))?;

        let mut applied_count = 0_usize;
        let mut errors = Vec::new();
        let mut tail: Vec<Value> = Vec::new();
        for (index, record) in pending.records.iter().enumerate() {
            if let Err(error) = sink.append_record(&pending.skill_name, record).await {
                tracing::warn!(
                    change_id,
                    error = %error,
                    "pending change commit failed; unwritten tail stays staged"
                );
                errors.push(error.to_string());
                tail = pending.records[index..].to_vec();
                break;
            }
            applied_count += 1;
        }

        let change_id = pending.change_id.clone();
        let skill_name = pending.skill_name.clone();
        if tail.is_empty() {
            self.pending.lock().remove(&change_id);
            return Ok(PendingChangeOutcome {
                change_id,
                skill_name,
                status: PendingChangeStatus::Persisted,
                applied_count,
                pending_count: 0,
                rejected_count: 0,
                errors,
            });
        }
        let pending_count = tail.len();
        pending.records = tail;
        pending.status = PendingChangeStatus::Partial;
        self.pending.lock().insert(change_id.clone(), pending);
        Ok(PendingChangeOutcome {
            change_id,
            skill_name,
            status: PendingChangeStatus::Partial,
            applied_count,
            pending_count,
            rejected_count: 0,
            errors,
        })
    }

    /// The configured write sink, if any.
    fn sink(&self) -> Option<Arc<dyn SkillExperienceStore>> {
        self.sink.read().clone()
    }
}

/// Error reported for an unknown pending-change identifier.
fn unknown_change(change_id: &str) -> Error {
    Error::not_found(format!("unknown pending change '{change_id}'"))
}

/// Metadata aggregated from the preview's applied results.
fn preview_metadata(preview: &LocalApplyPreview) -> BTreeMap<String, Value> {
    let mut metadata = BTreeMap::new();
    for result in preview.apply_results.iter().filter(|result| result.applied) {
        for (key, value) in &result.metadata {
            metadata.insert(key.clone(), value.clone());
        }
    }
    metadata
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::assert_is_empty
    )]

    use super::*;
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;

    use crate::operator::base::UpdateValue;

    fn preview(skill_name: &str, records: Vec<Value>) -> LocalApplyPreview {
        LocalApplyPreview {
            skill_name: skill_name.to_owned(),
            records,
            change_type: SKILL_EXPERIENCE_ENTRY.to_owned(),
            lifecycle_stage: LOCAL_APPLY_COMPLETED.to_owned(),
            apply_results: Vec::new(),
        }
    }

    fn working_store() -> Arc<dyn SkillExperienceStore> {
        Arc::new(InMemorySkillExperienceStore::new())
    }

    fn as_sink(store: &Arc<InMemorySkillExperienceStore>) -> Arc<dyn SkillExperienceStore> {
        Arc::clone(store) as Arc<dyn SkillExperienceStore>
    }

    #[test]
    fn staged_changes_are_ordered_and_identified() {
        let store = PendingChangeStore::new();
        let first = store.stage(&preview("alpha", vec![json!({"text": "a"})]));
        let second = store.stage(&preview("beta", vec![json!({"text": "b"})]));

        assert!(first.change_id.starts_with("skill_evolve_"));
        let suffix = first.change_id.strip_prefix("skill_evolve_").unwrap();
        assert_eq!(suffix.len(), 8);
        assert!(suffix.chars().all(|c| c.is_ascii_hexdigit()));
        assert_eq!(first.created_seq, 1);
        assert_eq!(second.created_seq, 2);
        assert_ne!(first.change_id, second.change_id);
        assert_eq!(first.status, PendingChangeStatus::PendingApproval);
        assert_eq!(first.change_type, SKILL_EXPERIENCE_ENTRY);

        let listed = store.list();
        assert_eq!(listed.len(), 2);
        assert_eq!(listed[0].change_id, first.change_id);
        assert_eq!(listed[1].change_id, second.change_id);
        assert_eq!(store.get(&second.change_id), Some(second));
        assert!(store.get("missing").is_none());
    }

    #[test]
    fn stage_copies_preview_metadata_from_applied_results() {
        let store = PendingChangeStore::new();
        let mut preview = preview("alpha", vec![json!({"text": "a"})]);
        preview.apply_results = vec![
            ApplyResult::applied("experiences")
                .with_metadata_entry("skill_name", json!("alpha"))
                .with_metadata_entry("source", json!("online")),
            ApplyResult::skipped("experiences", "nope")
                .with_metadata_entry("source", json!("dropped")),
        ];
        let staged = store.stage(&preview);
        assert_eq!(staged.metadata.get("skill_name"), Some(&json!("alpha")));
        assert_eq!(staged.metadata.get("source"), Some(&json!("online")));
    }

    #[tokio::test]
    async fn approve_persists_every_record_and_removes_the_entry() {
        let sink = Arc::new(InMemorySkillExperienceStore::new());
        let store = PendingChangeStore::new().with_store(as_sink(&sink));
        let staged = store.stage(&preview(
            "alpha",
            vec![json!({"text": "a"}), json!({"text": "b"})],
        ));

        let outcome = store.approve(&staged.change_id).await.unwrap();
        assert_eq!(outcome.status, PendingChangeStatus::Persisted);
        assert_eq!(outcome.status.as_str(), "persisted");
        assert_eq!(outcome.applied_count, 2);
        assert_eq!(outcome.pending_count, 0);
        assert_eq!(outcome.rejected_count, 0);
        assert!(outcome.errors.is_empty());
        assert_eq!(sink.len(), 2);
        assert_eq!(
            sink.records(),
            vec![
                ("alpha".to_owned(), json!({"text": "a"})),
                ("alpha".to_owned(), json!({"text": "b"})),
            ]
        );
        assert!(store.get(&staged.change_id).is_none());
        assert!(store.list().is_empty());
    }

    #[tokio::test]
    async fn approve_with_no_records_is_persisted_and_removed() {
        let sink = Arc::new(InMemorySkillExperienceStore::new());
        let store = PendingChangeStore::new().with_store(as_sink(&sink));
        let staged = store.stage(&preview("alpha", Vec::new()));

        let outcome = store.approve(&staged.change_id).await.unwrap();
        assert_eq!(outcome.status, PendingChangeStatus::Persisted);
        assert_eq!(outcome.applied_count, 0);
        assert!(sink.is_empty());
        assert!(store.get(&staged.change_id).is_none());
    }

    #[tokio::test]
    async fn reject_removes_the_entry_and_counts_records() {
        let sink = Arc::new(InMemorySkillExperienceStore::new());
        let store = PendingChangeStore::new().with_store(as_sink(&sink));
        let staged = store.stage(&preview(
            "alpha",
            vec![
                json!({"text": "a"}),
                json!({"text": "b"}),
                json!({"text": "c"}),
            ],
        ));

        let outcome = store.reject(&staged.change_id).await.unwrap();
        assert_eq!(outcome.status, PendingChangeStatus::Rejected);
        assert_eq!(outcome.status.as_str(), "rejected");
        assert_eq!(outcome.rejected_count, 3);
        assert_eq!(outcome.applied_count, 0);
        assert_eq!(outcome.pending_count, 0);
        assert!(outcome.errors.is_empty());
        assert!(sink.is_empty());
        assert!(store.get(&staged.change_id).is_none());
    }

    #[tokio::test]
    async fn partial_approval_keeps_the_tail_and_retry_finishes_it() {
        let failing = Arc::new(InMemorySkillExperienceStore::with_failure_after(1));
        let store = PendingChangeStore::new();
        store.set_store(as_sink(&failing));
        let staged = store.stage(&preview(
            "alpha",
            vec![
                json!({"text": "a"}),
                json!({"text": "b"}),
                json!({"text": "c"}),
            ],
        ));

        let outcome = store.approve(&staged.change_id).await.unwrap();
        assert_eq!(outcome.status, PendingChangeStatus::Partial);
        assert_eq!(outcome.status.as_str(), "partial");
        assert_eq!(outcome.applied_count, 1);
        assert_eq!(outcome.pending_count, 2);
        assert_eq!(outcome.rejected_count, 0);
        assert_eq!(outcome.errors.len(), 1);
        assert!(outcome.errors[0].contains("injected append failure"));
        assert_eq!(failing.len(), 1);

        let retained = store.get(&staged.change_id).unwrap();
        assert_eq!(retained.status, PendingChangeStatus::Partial);
        assert_eq!(
            retained.records,
            vec![json!({"text": "b"}), json!({"text": "c"})]
        );

        // A working sink commits only the two-record tail.
        let working = Arc::new(InMemorySkillExperienceStore::new());
        store.set_store(as_sink(&working));
        let retried = store.retry(&staged.change_id).await.unwrap();
        assert_eq!(retried.status, PendingChangeStatus::Persisted);
        assert_eq!(retried.applied_count, 2);
        assert_eq!(retried.pending_count, 0);
        assert!(retried.errors.is_empty());
        // The working sink receives only the two-record tail, never the record
        // the first attempt already wrote.
        assert_eq!(
            working.records(),
            vec![
                ("alpha".to_owned(), json!({"text": "b"})),
                ("alpha".to_owned(), json!({"text": "c"})),
            ]
        );

        let mut combined = failing.records();
        combined.extend(working.records());
        assert_eq!(
            combined,
            vec![
                ("alpha".to_owned(), json!({"text": "a"})),
                ("alpha".to_owned(), json!({"text": "b"})),
                ("alpha".to_owned(), json!({"text": "c"})),
            ]
        );
        assert!(store.get(&staged.change_id).is_none());
    }

    #[tokio::test]
    async fn retry_without_records_is_persisted() {
        let store = PendingChangeStore::new().with_store(working_store());
        let staged = store.stage(&preview("alpha", Vec::new()));
        let outcome = store.retry(&staged.change_id).await.unwrap();
        assert_eq!(outcome.status, PendingChangeStatus::Persisted);
        assert_eq!(outcome.applied_count, 0);
        assert!(store.get(&staged.change_id).is_none());
    }

    #[tokio::test]
    async fn unknown_change_is_not_found() {
        let store = PendingChangeStore::new().with_store(working_store());
        for error in [
            store.approve("missing").await.unwrap_err(),
            store.reject("missing").await.unwrap_err(),
            store.retry("missing").await.unwrap_err(),
        ] {
            assert_eq!(error.kind(), ErrorKind::NotFound);
            assert!(error
                .to_string()
                .contains("unknown pending change 'missing'"));
        }
    }

    #[tokio::test]
    async fn committing_without_a_sink_is_a_config_error() {
        let store = PendingChangeStore::new();
        let staged = store.stage(&preview("alpha", vec![json!({"text": "a"})]));
        let error = store.approve(&staged.change_id).await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Config);
        assert!(store.get(&staged.change_id).is_some());
    }

    #[test]
    fn build_preview_drops_unapplied_and_tracks_change_type() {
        let results = vec![
            ApplyResult::applied("experiences")
                .with_records(vec![json!({"text": "a"})])
                .with_lifecycle_stage(LOCAL_APPLY_COMPLETED),
            ApplyResult::rejection(
                "skill_experience_alpha",
                "experiences",
                &UpdateValue::set(json!([])),
                vec!["nope".to_owned()],
            )
            .with_records(vec![json!({"text": "dropped"})]),
            ApplyResult::applied("experiences")
                .with_records(vec![json!({"text": "c"})])
                .with_change_type("experience_entry")
                .with_lifecycle_stage(LOCAL_APPLY_COMPLETED),
        ];
        let built = build_local_apply_preview("alpha", &results).unwrap();
        assert_eq!(built.skill_name, "alpha");
        assert_eq!(
            built.records,
            vec![json!({"text": "a"}), json!({"text": "c"})]
        );
        assert_eq!(built.change_type, "experience_entry");
        assert_eq!(built.lifecycle_stage, LOCAL_APPLY_COMPLETED);
        assert_eq!(built.apply_results.len(), 3);

        let plain = build_local_apply_preview(
            "beta",
            &[ApplyResult::applied("experiences").with_records(vec![json!({"text": "x"})])],
        )
        .unwrap();
        assert_eq!(plain.change_type, SKILL_EXPERIENCE_ENTRY);
        assert!(build_local_apply_preview("beta", &[])
            .unwrap()
            .records
            .is_empty());
    }

    #[test]
    fn build_preview_rejects_other_lifecycle_stages() {
        let results =
            vec![ApplyResult::applied("experiences").with_lifecycle_stage("pending_approval")];
        let error = build_local_apply_preview("alpha", &results).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("pending_approval"));
        assert!(error.to_string().contains("alpha"));
    }

    #[test]
    fn preview_round_trips_through_json() {
        let preview = preview("alpha", vec![json!({"text": "a"})]);
        let value = serde_json::to_value(&preview).unwrap();
        assert_eq!(value["change_type"], json!(SKILL_EXPERIENCE_ENTRY));
        assert_eq!(value["lifecycle_stage"], json!(LOCAL_APPLY_COMPLETED));
        let round: LocalApplyPreview = serde_json::from_value(value).unwrap();
        assert_eq!(round, preview);
    }

    #[test]
    fn outcome_serde_uses_snake_case_status() {
        let outcome = PendingChangeOutcome {
            change_id: "skill_evolve_00000001".to_owned(),
            skill_name: "alpha".to_owned(),
            status: PendingChangeStatus::Partial,
            applied_count: 1,
            pending_count: 2,
            rejected_count: 0,
            errors: vec!["boom".to_owned()],
        };
        let value = serde_json::to_value(&outcome).unwrap();
        assert_eq!(value["status"], json!("partial"));
        let round: PendingChangeOutcome = serde_json::from_value(value).unwrap();
        assert_eq!(round, outcome);
    }
}
