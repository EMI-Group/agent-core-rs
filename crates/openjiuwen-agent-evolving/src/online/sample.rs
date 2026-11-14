//! The sample ledger: scored RL samples waiting for, inside, or past training.
//!
//! Python parity: `agent_rl/storage/trajectory_store.py`
//! (`TrajectorySampleStore` + `InMemoryTrajectoryStore`). The ledger keeps the
//! same contract — one record per `sample_id`, a status index per status, and
//! first-wins publication — on top of [`OnlineStore`] instead of an in-process
//! dictionary, so the in-memory and future Redis backends share every rule.
//!
//! # Deviation from Python's score semantics
//!
//! Python orders pending samples by a Redis ZSET score, which the online gateway
//! sets to `datetime.now(timezone.utc).timestamp()`. This module never reads the
//! wall clock: [`SampleRecord::created_seq`] is a monotonic counter allocated
//! from `rl:v1:sample:seq` on the first write, and every queue decision orders
//! by `(created_seq, sample_id)`. Creation order is therefore reproduced
//! exactly, and replaying a test produces the same batch on every machine.
//! Sequence numbers are allocated before the write is attempted, so a duplicate
//! id that loses the first-wins reservation burns one number (a documented gap).
//!
//! # Status machine
//!
//! | From | Event | To |
//! |---|---|---|
//! | — | `save_once` (new id) | `pending` |
//! | — | `save_once` (existing id) | unchanged, not returned |
//! | `pending` | `fetch_and_mark_training` | `training` |
//! | `training` | `mark_trained` | `trained` |
//! | `training` | `mark_failed` | `failed` |
//! | `training` | `reset_to_pending` | `pending` |
//! | any | `mark_*`/`reset_to_pending` replayed | idempotent |
//!
//! Every transition follows the record's *current* status, so replaying a
//! transition never leaves an id in two index sets.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::error::{OnlineError, Result};
use super::json;
use super::store::{keys, OnlineStore};

/// Where a sample sits in the training pipeline.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SampleStatus {
    /// Scored and waiting for a training batch.
    #[default]
    Pending,
    /// Claimed by the training run that is currently executing.
    Training,
    /// Consumed by a succeeded training run.
    Trained,
    /// Abandoned by a failed training run.
    Failed,
}

impl SampleStatus {
    /// The Python wire literal of the status.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Training => "training",
            Self::Trained => "trained",
            Self::Failed => "failed",
        }
    }
}

/// One scored RL sample and its pipeline position.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SampleRecord {
    /// Caller-supplied sample id; the first writer of an id owns the record.
    pub sample_id: String,
    /// Sample owner the training runner batches for.
    pub user_id: String,
    /// Agent session the sample was captured from.
    pub session_id: String,
    /// RL task the sample belongs to.
    pub task_id: String,
    /// Terminal (or delayed-feedback) reward of the sample.
    pub reward: f64,
    /// Pipeline position.
    pub status: SampleStatus,
    /// The recorded trajectory/material payload, opaque to this module.
    pub payload: Value,
    /// Monotonic creation sequence allocated by the ledger (never a clock).
    pub created_seq: u64,
}

impl Default for SampleRecord {
    fn default() -> Self {
        Self {
            sample_id: String::new(),
            user_id: String::new(),
            session_id: String::new(),
            task_id: String::new(),
            reward: 0.0,
            status: SampleStatus::Pending,
            payload: Value::Null,
            created_seq: 0,
        }
    }
}

impl SampleRecord {
    /// Builds a pending sample; the ledger assigns `created_seq`.
    #[must_use]
    pub fn pending(
        sample_id: impl Into<String>,
        user_id: impl Into<String>,
        reward: f64,
        payload: Value,
    ) -> Self {
        Self {
            sample_id: sample_id.into(),
            user_id: user_id.into(),
            reward,
            payload,
            ..Self::default()
        }
    }

    /// The `policy_version` the sample was rolled out on, `"base"` when absent.
    ///
    /// Python reads `sample["policy_version"]`; the port keeps it inside the
    /// opaque [`SampleRecord::payload`] so the ledger stays schema-free.
    #[must_use]
    pub fn policy_version(&self) -> String {
        self.payload
            .get("policy_version")
            .and_then(Value::as_str)
            .filter(|version| !version.is_empty())
            .map_or_else(|| "base".to_owned(), ToOwned::to_owned)
    }

    /// Rejects a blank id and a non-finite reward.
    pub fn validate(&self) -> Result<()> {
        if self.sample_id.trim().is_empty() {
            return Err(OnlineError::InvalidRequest(
                "sample_id is required".to_owned(),
            ));
        }
        if !self.reward.is_finite() {
            return Err(OnlineError::InvalidRequest(format!(
                "sample reward must be finite: {}",
                self.sample_id
            )));
        }
        Ok(())
    }

    /// Deserializes a record from its store representation.
    pub fn from_json(payload: &str) -> Result<Self> {
        json::decode(payload)
    }

    /// Serializes the record into its store representation.
    pub fn to_json(&self) -> Result<String> {
        json::encode(self)
    }
}

/// The per-status sample queue, backed by an [`OnlineStore`].
///
/// The ledger is stateless — every call reads and writes through the store — so
/// two ledgers over one backend observe identical state.
pub struct SampleLedger<S: OnlineStore> {
    store: S,
}

/// Python's name for the same contract (`TrajectorySampleStore`).
pub type SampleStore<S> = SampleLedger<S>;

impl<S: OnlineStore> SampleLedger<S> {
    /// Creates a ledger over `store`.
    pub fn new(store: S) -> Self {
        Self { store }
    }

    /// The backing store.
    #[must_use]
    pub fn store(&self) -> &S {
        &self.store
    }

    /// Saves every sample not already present and returns the newly saved ids.
    ///
    /// Publication is first-wins: an id that already exists keeps its original
    /// record — including its status, reward and sequence — and the new sample
    /// is dropped rather than resurrected. That matches Python's
    /// `save_samples_once`, which makes a replayed capture a no-op.
    pub async fn save_once(&self, samples: &[SampleRecord]) -> Result<Vec<String>> {
        let mut saved = Vec::new();
        for sample in samples {
            sample.validate()?;
            // Allocate first: a duplicate id then burns a sequence number
            // instead of colliding with the record it lost to.
            let created_seq = self.next_seq().await?;
            let stored = SampleRecord {
                status: SampleStatus::Pending,
                created_seq,
                ..sample.clone()
            };
            let key = keys::sample(&stored.sample_id);
            if !self
                .store
                .set_if_absent(&key, &stored.to_json()?)
                .await
                .map_err(OnlineError::store)?
            {
                continue;
            }
            self.index(&stored.sample_id, SampleStatus::Pending).await?;
            saved.push(stored.sample_id);
        }
        Ok(saved)
    }

    /// Returns one sample by id.
    pub async fn get(&self, sample_id: &str) -> Result<Option<SampleRecord>> {
        self.load(sample_id).await
    }

    /// Returns the samples in one status, ordered by creation.
    ///
    /// An index entry whose record disappeared (a partially applied write) is
    /// skipped, as Python's `list_samples` does.
    pub async fn list_by_status(&self, status: SampleStatus) -> Result<Vec<SampleRecord>> {
        let mut records = Vec::new();
        for sample_id in self.members(status).await? {
            if let Some(record) = self.load(&sample_id).await? {
                records.push(record);
            }
        }
        records.sort_by(|left, right| {
            left.created_seq
                .cmp(&right.created_seq)
                .then_with(|| left.sample_id.cmp(&right.sample_id))
        });
        Ok(records)
    }

    /// Counts the samples in one status.
    pub async fn count_by_status(&self, status: SampleStatus) -> Result<usize> {
        Ok(self.members(status).await?.len())
    }

    /// Counts the samples waiting for a training batch.
    pub async fn pending_count(&self) -> Result<usize> {
        self.count_by_status(SampleStatus::Pending).await
    }

    /// Moves up to `limit` pending samples to `training` and returns them.
    ///
    /// The batch is the `limit` oldest pending samples by `(created_seq,
    /// sample_id)`, matching Python's `zrange` over the pending ZSET. The
    /// transition is atomic enough for one connection: the training runner is
    /// single-flight, so no second fetch drains the same ids concurrently.
    pub async fn fetch_and_mark_training(&self, limit: usize) -> Result<Vec<SampleRecord>> {
        let mut batch = self.list_by_status(SampleStatus::Pending).await?;
        batch.truncate(limit.max(1));
        for record in &mut batch {
            self.unindex(&record.sample_id, SampleStatus::Pending)
                .await?;
            record.status = SampleStatus::Training;
            self.store_record(record).await?;
            self.index(&record.sample_id, SampleStatus::Training)
                .await?;
        }
        Ok(batch)
    }

    /// Marks the supplied samples `trained`.
    pub async fn mark_trained(&self, sample_ids: &[String]) -> Result<()> {
        self.transition(sample_ids, SampleStatus::Trained).await
    }

    /// Marks the supplied samples `failed`.
    pub async fn mark_failed(&self, sample_ids: &[String]) -> Result<()> {
        self.transition(sample_ids, SampleStatus::Failed).await
    }

    /// Returns the supplied samples to `pending`.
    pub async fn reset_to_pending(&self, sample_ids: &[String]) -> Result<()> {
        self.transition(sample_ids, SampleStatus::Pending).await
    }

    /// Counters in Python's `stats()` shape.
    ///
    /// `total_samples` counts everything the ledger knows; the per-status
    /// counters are the index sizes.
    pub async fn stats(&self) -> Result<BTreeMap<String, usize>> {
        let mut stats = BTreeMap::new();
        let mut total = 0;
        for (name, status) in [
            ("pending_samples", SampleStatus::Pending),
            ("training_samples", SampleStatus::Training),
            ("trained_samples", SampleStatus::Trained),
            ("failed_samples", SampleStatus::Failed),
        ] {
            let count = self.count_by_status(status).await?;
            total += count;
            stats.insert(name.to_owned(), count);
        }
        stats.insert("total_samples".to_owned(), total);
        Ok(stats)
    }

    async fn load(&self, sample_id: &str) -> Result<Option<SampleRecord>> {
        match self.read(&keys::sample(sample_id)).await? {
            Some(payload) => Ok(Some(SampleRecord::from_json(&payload)?)),
            None => Ok(None),
        }
    }

    /// Moves each known sample to `status`, following its current status.
    async fn transition(&self, sample_ids: &[String], status: SampleStatus) -> Result<()> {
        for sample_id in sample_ids {
            let Some(mut record) = self.load(sample_id).await? else {
                continue;
            };
            if record.status == status {
                self.index(sample_id, status).await?;
                continue;
            }
            self.unindex(sample_id, record.status).await?;
            record.status = status;
            self.store_record(&record).await?;
            self.index(sample_id, status).await?;
        }
        Ok(())
    }

    async fn store_record(&self, record: &SampleRecord) -> Result<()> {
        self.write(&keys::sample(&record.sample_id), &record.to_json()?)
            .await
    }

    async fn index(&self, sample_id: &str, status: SampleStatus) -> Result<()> {
        self.store
            .set_add(&keys::samples_by_status(status.as_str()), sample_id)
            .await
            .map_err(OnlineError::store)?;
        Ok(())
    }

    async fn unindex(&self, sample_id: &str, status: SampleStatus) -> Result<()> {
        self.store
            .set_remove(&keys::samples_by_status(status.as_str()), sample_id)
            .await
            .map_err(OnlineError::store)?;
        Ok(())
    }

    async fn members(&self, status: SampleStatus) -> Result<Vec<String>> {
        self.store
            .set_members(&keys::samples_by_status(status.as_str()))
            .await
            .map_err(OnlineError::store)
    }

    /// Allocates the next creation sequence number.
    ///
    /// The counter is one store key, so a concurrent writer can lose an
    /// increment; the ordering of two samples created in the same instant is
    /// then a tie broken by `sample_id`, never a correctness problem. Redis
    /// `INCR` is the atomic upgrade a later backend can make.
    async fn next_seq(&self) -> Result<u64> {
        let key = keys::sample_seq();
        let next = match self.read(&key).await? {
            Some(raw) => raw
                .trim()
                .parse::<u64>()
                .map_err(|error| {
                    OnlineError::store(openjiuwen_common::error::Error::external(error))
                })?
                .saturating_add(1),
            None => 1,
        };
        self.write(&key, &next.to_string()).await?;
        Ok(next)
    }

    async fn read(&self, key: &str) -> Result<Option<String>> {
        self.store.get(key).await.map_err(OnlineError::store)
    }

    async fn write(&self, key: &str, value: &str) -> Result<()> {
        self.store.set(key, value).await.map_err(OnlineError::store)
    }
}
