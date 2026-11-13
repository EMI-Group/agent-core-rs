//! Snapshot, save, restore, and child-agent rebuild of the sub-agent runtime's instances.
//!
//! Completes the bucket layer of Python `openjiuwen/harness/subagent_runtime/persistence.py`:
//! [`SubAgentRuntime::snapshot`] flattens every instance — live ones and closed tombstones —
//! into one [`PersistedSubAgentBuckets`], [`SubAgentRuntime::save`] trims that snapshot to the
//! retention bound of [`max_persisted_records`], and [`SubAgentRuntime::load`] rebuilds
//! instances from a bucket.
//!
//! A restored instance has no child agent: [`SubAgentRuntime::resume`] asks the
//! [`SubAgentFactory`](super::SubAgentFactory) for one again from the configuration the record
//! kept, so a reload survives a process restart. Restore itself never resumes work — a record
//! saved mid-turn is downgraded, because no turn survives reconstruction.
//!
//! The runtime state lock is never held across an `.await`: every bucket method here is
//! synchronous, and the factory call of [`SubAgentRuntime::resume`] runs between two short
//! locked sections.

use std::collections::BTreeMap;
use std::path::Path;

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};

use super::evict;
use super::instance::Instance;
use super::persistence::{
    load_bucket_file, max_persisted_records, save_bucket_file, PersistedSubAgentBuckets,
    PersistedSubAgentRecord,
};
use super::quota::SubagentActivity;
use super::transcript::TranscriptEntry;
use super::{not_found, ResumeReport, SubAgentId, SubAgentRuntime, SubagentStatus};

/// Report of one restore (see [`SubAgentRuntime::load`]).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct LoadReport {
    /// Records present in the bucket.
    pub records: usize,
    /// Records restored into this runtime.
    pub restored: usize,
    /// Records skipped because the id was already live or the live cap was reached.
    pub skipped: usize,
}

impl SubAgentRuntime {
    /// Snapshot of every instance (live and closed tombstones) as a persisted bucket; the bucket
    /// revision advances on every snapshot.
    #[must_use]
    pub fn snapshot(&self) -> PersistedSubAgentBuckets {
        let mut state = self.shared.state.lock();
        state.revision = state.revision.wrapping_add(1);
        let revision = state.revision;
        let mut records = BTreeMap::new();
        for (id, instance) in &state.instances {
            records.insert(id.clone(), instance.persisted());
        }
        PersistedSubAgentBuckets { revision, records }
    }

    /// Trimmed snapshot (`max_persisted_records(quota.max_subagents)`) ready to persist.
    ///
    /// The trim drops closed records before live ones, oldest first (see
    /// [`PersistedSubAgentBuckets::trim`]), so a long-lived session keeps its history bounded
    /// without losing a running child.
    #[must_use]
    pub fn save(&self) -> PersistedSubAgentBuckets {
        let mut buckets = self.snapshot();
        buckets.trim(max_persisted_records(self.quota.max_subagents));
        buckets
    }

    /// Writes [`SubAgentRuntime::save`] to `path` as pretty JSON and returns the written bucket.
    ///
    /// # Errors
    /// `Error::Session` (see [`save_bucket_file`]) when the file cannot be written.
    pub fn save_to_path(&self, path: &Path) -> Result<PersistedSubAgentBuckets> {
        let buckets = self.save();
        save_bucket_file(path, &buckets)?;
        Ok(buckets)
    }

    /// Restores `buckets` into this runtime and reports what happened.
    ///
    /// Records are visited in `spawn_seq` order, then by id, so a bucket always restores into the
    /// same shape. A record whose id is already present is skipped, which makes `load`
    /// idempotent, and so is every record once the runtime holds `quota.max_subagents` live
    /// instances. Each restored instance keeps its identity, configuration, transcript (replayed
    /// into a ring of `quota.transcript_capacity` entries, sequence numbers included), status,
    /// activity and timestamps, but has no child agent and no running turn: a `Running`/
    /// `PendingInit` status becomes `Interrupted`, a `Running`/`Queued` activity becomes `Idle`,
    /// and `closed_reason` is kept only on a `Closed` record. The cumulative spawn ledger becomes
    /// `max(spawned, records)` so the budget still reflects the reloaded history. An empty bucket
    /// restores nothing.
    ///
    /// # Errors
    /// `Error::Validation("invalid subagent record: <id> has status not_found")` for a record
    /// whose status is `NotFound`; the whole bucket is validated first, so nothing is restored in
    /// that case.
    pub fn load(&self, buckets: &PersistedSubAgentBuckets) -> Result<LoadReport> {
        let mut report = LoadReport {
            records: buckets.len(),
            ..LoadReport::default()
        };
        if buckets.is_empty() {
            return Ok(report);
        }
        let mut order: Vec<(&SubAgentId, &PersistedSubAgentRecord)> =
            buckets.records.iter().collect();
        order.sort_by(|(left_id, left), (right_id, right)| {
            left.spawn_seq
                .cmp(&right.spawn_seq)
                .then_with(|| left_id.cmp(right_id))
        });
        for (id, record) in &order {
            if record.status == SubagentStatus::NotFound {
                return Err(Error::validation(format!(
                    "invalid subagent record: {id} has status not_found"
                )));
            }
        }
        {
            let mut state = self.shared.state.lock();
            let cap = self.quota.max_subagents;
            let mut live = evict::live_count(&state);
            for (id, record) in order {
                if live >= cap || state.instances.contains_key(id) {
                    report.skipped += 1;
                    continue;
                }
                let seq = state.next_seq;
                state.next_seq = state.next_seq.wrapping_add(1);
                let instance = Instance::restore(
                    seq,
                    id.clone(),
                    record,
                    self.quota.transcript_capacity,
                    self.quota.transcript_channel_capacity,
                );
                if instance.is_live() {
                    live += 1;
                }
                state.instances.insert(id.clone(), instance);
                report.restored += 1;
            }
            state.spawned = state.spawned.max(report.records);
        }
        self.shared.notify.notify_waiters();
        tracing::debug!(
            target: "openjiuwen_harness::subagents",
            records = report.records,
            restored = report.restored,
            skipped = report.skipped,
            "subagent bucket restored"
        );
        Ok(report)
    }

    /// Reads a bucket file (missing file = empty bucket) and restores it.
    ///
    /// # Errors
    /// `Error::Session` for IO/parse failures (see [`load_bucket_file`]), `Error::Validation`
    /// per [`SubAgentRuntime::load`].
    pub fn load_from_path(&self, path: &Path) -> Result<LoadReport> {
        let buckets = load_bucket_file(path)?;
        self.load(&buckets)
    }

    /// Revives a closed or restored instance so new input can be sent to it.
    ///
    /// A closed in-process instance keeps its child agent and is revived in place (`Closed` goes
    /// back to `Completed`). An instance without a child — one rebuilt from a persisted bucket by
    /// [`SubAgentRuntime::load`] — asks the [`SubAgentFactory`](super::SubAgentFactory) for a new
    /// child built from the configuration the instance kept; the factory runs outside the state
    /// lock, and the instance status is preserved unless it was `Closed`. Either way the instance
    /// ends `Idle` with `closed_reason` cleared, a fresh turn generation, and a `status("idle")`
    /// transcript entry.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the id was never registered, and propagates the factory
    /// error unchanged when the child cannot be rebuilt — the instance then stays exactly as it
    /// was.
    // Kept async for parity with the Python control surface; the tools await it.
    #[allow(clippy::unused_async, clippy::unused_async_trait_impl)]
    pub async fn resume(&self, id: &str) -> Result<ResumeReport> {
        let (config, sub_session_id, rebuild) = {
            let state = self.shared.state.lock();
            let Some(instance) = state.instances.get(id) else {
                return Err(not_found(id));
            };
            let rebuild = instance.agent.is_none();
            if !rebuild && instance.status != SubagentStatus::Closed {
                return Ok(already_live(instance.status));
            }
            let config = instance.config.clone();
            let sub_session_id = instance.sub_session_id.clone();
            (config, sub_session_id, rebuild)
        };
        let agent = if rebuild {
            Some(self.shared.factory.create(&config, &sub_session_id).await?)
        } else {
            None
        };
        let now_ms = self.clock.now_ms();
        let report = {
            let mut state = self.shared.state.lock();
            let Some(instance) = state.instances.get_mut(id) else {
                return Err(not_found(id));
            };
            if let Some(agent) = agent {
                instance.agent_id.clone_from(&agent.card().id);
                instance.agent = Some(agent);
            }
            if instance.status == SubagentStatus::Closed {
                instance.status = SubagentStatus::Completed;
            }
            instance.running = false;
            instance.generation = instance.generation.wrapping_add(1);
            instance.handle = None;
            instance.closed_reason = None;
            instance.touch(now_ms, SubagentActivity::Idle);
            instance.record(TranscriptEntry::status(id, "idle", now_ms));
            ResumeReport {
                status: instance.status,
                restored: true,
                message: None,
            }
        };
        self.shared.notify.notify_waiters();
        tracing::debug!(
            target: "openjiuwen_harness::subagents",
            subagent_id = %id,
            status = report.status.as_str(),
            rebuilt = rebuild,
            "subagent resumed"
        );
        Ok(report)
    }
}

/// The report of a `resume` on an instance that is already live with its child agent.
fn already_live(status: SubagentStatus) -> ResumeReport {
    ResumeReport {
        status,
        restored: false,
        message: Some("Instance is already live; use subagent_send_input directly.".to_owned()),
    }
}
