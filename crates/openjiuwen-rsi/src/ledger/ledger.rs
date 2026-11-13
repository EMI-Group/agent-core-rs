//! The run-scoped ledger: deduplicated model usage plus persisted history.
//!
//! [`RunLedger`] is the Rust counterpart of Python's `ModelUsageObserver`: it
//! binds a task's persisted events (replaying them so a resumed run never
//! double-charges a call), records new usage and stage events through the
//! injected [`LedgerStore`], keeps cumulative [`UsageTotals`], and tracks
//! calls that were started but never completed.
//!
//! The ledger never invents data. A call whose provider omitted token counters
//! contributes nothing but still counts as one call; a call in flight is only
//! finished when the caller says so, and [`RunLedger::check_delivery`] surfaces
//! the remaining ones instead of silently dropping them.
use crate::ledger::record::{
    ModelCallStatus, RunEvent, StageRecord, UsageRecord, UsageSnapshot, UsageTokens,
};
use crate::ledger::store::LedgerStore;
use crate::ledger::usage::{aggregate, LedgerMetrics, UsageTotals};
use openjiuwen_common::error::{Error, Result};
use std::collections::BTreeSet;
use std::fmt;
use std::sync::Arc;

/// The task's model-call ledger.
///
/// Construct one per run with [`RunLedger::new`], optionally attach a store with
/// [`RunLedger::with_store`], call [`RunLedger::bind`] to replay persisted
/// events, then feed it every completed call through
/// [`RunLedger::record_usage`]. All state is per-instance; there is no global
/// registry and no ambient context, so two runs can never charge each other.
pub struct RunLedger {
    task_id: String,
    store: Option<Arc<dyn LedgerStore>>,
    bound: bool,
    records: Vec<RunEvent>,
    seen_calls: BTreeSet<String>,
    pending: BTreeSet<String>,
    totals: UsageTotals,
}

impl fmt::Debug for RunLedger {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RunLedger")
            .field("task_id", &self.task_id)
            .field("has_store", &self.store.is_some())
            .field("bound", &self.bound)
            .field("records", &self.records.len())
            .field("seen_calls", &self.seen_calls)
            .field("pending", &self.pending)
            .field("totals", &self.totals)
            .finish()
    }
}

impl RunLedger {
    /// Creates an empty ledger for `task_id` with no persistence.
    #[must_use]
    pub fn new(task_id: impl Into<String>) -> Self {
        Self {
            task_id: task_id.into(),
            store: None,
            bound: false,
            records: Vec::new(),
            seen_calls: BTreeSet::new(),
            pending: BTreeSet::new(),
            totals: UsageTotals::default(),
        }
    }

    /// Attaches a persistence backend (builder style).
    #[must_use]
    pub fn with_store(mut self, store: Arc<dyn LedgerStore>) -> Self {
        self.store = Some(store);
        self
    }

    /// Returns the owning task id.
    #[must_use]
    pub fn task_id(&self) -> &str {
        &self.task_id
    }

    /// Loads this task's persisted events and replays them into the ledger.
    ///
    /// Replay verifies every usage record's `task_id` (a mismatch means the
    /// caller resumed the wrong run) and counts each `call_id` once, so a
    /// resumed run continues with the persisted totals and never double-charges
    /// a call. Binding is idempotent: a second call is a no-op, even after
    /// events were recorded in between.
    ///
    /// # Errors
    ///
    /// Returns [`Error::session`] when the store cannot read or parse the
    /// history, and [`Error::validation`] when a persisted record belongs to a
    /// different task.
    pub async fn bind(&mut self) -> Result<()> {
        if self.bound {
            return Ok(());
        }
        let loaded = match &self.store {
            Some(store) => store.load(&self.task_id).await?,
            None => Vec::new(),
        };
        for event in &loaded {
            if let RunEvent::Usage(record) = event {
                if record.task_id != self.task_id {
                    return Err(Error::validation(
                        "model usage ledger task_id does not match this run",
                    ));
                }
            }
        }
        self.reset();
        for event in loaded {
            if let RunEvent::Usage(record) = &event {
                if self.seen_calls.insert(record.call_id.clone()) {
                    self.totals.observe_usage(record);
                }
            }
            self.records.push(event);
        }
        self.bound = true;
        Ok(())
    }

    /// Records one completed model call.
    ///
    /// Returns `true` when the call was new and persisted, `false` when its
    /// `call_id` was already seen (the duplicate is ignored, matching Python's
    /// idempotent resume). A completed call also leaves the pending set, so a
    /// call that eventually reports usage is never finished as incomplete.
    ///
    /// # Errors
    ///
    /// Returns [`Error::session`] when the store cannot append the event (the
    /// in-memory state is then left untouched) and [`Error::validation`] when
    /// the record belongs to a different task.
    pub async fn record_usage(&mut self, record: UsageRecord) -> Result<bool> {
        if self.seen_calls.contains(&record.call_id) {
            tracing::debug!(
                task_id = %self.task_id,
                call_id = %record.call_id,
                "skipping duplicate model usage record"
            );
            return Ok(false);
        }
        if record.task_id != self.task_id {
            return Err(Error::validation(
                "model usage ledger task_id does not match this run",
            ));
        }
        if let Some(store) = &self.store {
            store
                .append(&self.task_id, &RunEvent::Usage(record.clone()))
                .await?;
        }
        self.pending.remove(&record.call_id);
        self.seen_calls.insert(record.call_id.clone());
        self.totals.observe_usage(&record);
        self.records.push(RunEvent::Usage(record));
        Ok(true)
    }

    /// Marks a call as started but not yet completed.
    ///
    /// Python's observer tracks this while an LLM request is in flight; the Rust
    /// ledger lets the caller declare it explicitly. A later
    /// [`RunLedger::record_usage`] with the same `call_id` clears the marker.
    pub fn mark_pending(&mut self, call_id: impl Into<String>) {
        self.pending.insert(call_id.into());
    }

    /// Finishes every pending call with `status` and clears the pending set.
    ///
    /// The store is append-only, so the rewrite applies to the in-memory
    /// records only (a call that never reached the store has nothing to
    /// rewrite); persisted reconciliation stays the caller's job, exactly like
    /// Python's `finish_pending` / `check_delivery` split. The call is a no-op
    /// when nothing is pending.
    pub fn finish_pending(&mut self, status: ModelCallStatus) {
        if self.pending.is_empty() {
            return;
        }
        let pending = std::mem::take(&mut self.pending);
        for call_id in &pending {
            let mut rewritten = false;
            for event in &mut self.records {
                if let RunEvent::Usage(record) = event {
                    if &record.call_id == call_id {
                        record.model_call.status = status;
                        rewritten = true;
                    }
                }
            }
            if !rewritten {
                tracing::debug!(
                    task_id = %self.task_id,
                    call_id = %call_id,
                    status = status.as_str(),
                    "finishing a pending model usage call with no persisted record"
                );
            }
        }
        tracing::debug!(
            task_id = %self.task_id,
            calls = pending.len(),
            status = status.as_str(),
            "finished pending model usage calls"
        );
    }

    /// Records one node stage transition.
    ///
    /// # Errors
    ///
    /// Returns [`Error::session`] when the store cannot append the event.
    pub async fn record_stage(&mut self, record: StageRecord) -> Result<()> {
        if let Some(store) = &self.store {
            store
                .append(&self.task_id, &RunEvent::Stage(record.clone()))
                .await?;
        }
        self.records.push(RunEvent::Stage(record));
        Ok(())
    }

    /// Returns the cumulative usage totals.
    #[must_use]
    pub fn totals(&self) -> UsageTotals {
        self.totals
    }

    /// Returns every event the ledger holds, in replay/write order.
    #[must_use]
    pub fn records(&self) -> &[RunEvent] {
        &self.records
    }

    /// Aggregates the ledger into overall, per-model, and per-stage metrics.
    #[must_use]
    pub fn metrics(&self) -> LedgerMetrics {
        aggregate(&self.records)
    }

    /// Projects the cumulative totals onto a [`UsageSnapshot`].
    ///
    /// Once at least one call has been observed the token counters and cost are
    /// reported (unreported per-call counters having been folded to zero); with
    /// no calls at all the counters stay unknown and only `call_count` is `0`.
    #[must_use]
    pub fn snapshot(&self) -> UsageSnapshot {
        let reported = self.totals.calls > 0;
        UsageSnapshot {
            tokens: UsageTokens::new(
                reported.then_some(self.totals.input_tokens),
                reported.then_some(self.totals.output_tokens),
                reported.then_some(self.totals.cache_hit_tokens),
            ),
            cost_estimate: reported.then_some(self.totals.cost_usd),
            call_count: Some(self.totals.calls),
        }
    }

    /// Fails when a call was started but never completed.
    ///
    /// Port of Python's `ModelUsageObserver.check_delivery`: observation
    /// failures must surface outside the model's retry/error handler, because a
    /// silently lost call would corrupt the run's cost accounting.
    ///
    /// # Errors
    ///
    /// Returns [`Error::session`] while any call is still pending.
    pub fn check_delivery(&self) -> Result<()> {
        if self.pending.is_empty() {
            Ok(())
        } else {
            Err(Error::session(
                "RSI model usage delivery failed; persisted calls can be reconciled",
            ))
        }
    }

    /// Removes this task's persisted history and resets all in-memory state.
    ///
    /// # Errors
    ///
    /// Returns [`Error::session`] when the store cannot delete the history; the
    /// in-memory state is then left untouched so a failed clear can be retried.
    pub async fn clear(&mut self) -> Result<()> {
        if let Some(store) = &self.store {
            store.remove(&self.task_id).await?;
        }
        self.reset();
        self.bound = false;
        Ok(())
    }

    /// Resets the replayed and recorded state while keeping the task id and store.
    fn reset(&mut self) {
        self.records.clear();
        self.seen_calls.clear();
        self.pending.clear();
        self.totals = UsageTotals::default();
    }
}

#[cfg(test)]
mod tests {
    use crate::ledger::record::{ModelCall, UsageTokens};
    use crate::ledger::store::InMemoryLedgerStore;
    use std::sync::Arc;

    use super::*;

    fn record(task_id: &str, call_id: &str, tokens: UsageTokens) -> UsageRecord {
        UsageRecord::new(
            task_id,
            call_id,
            ModelCall::new("model").with_tokens(tokens),
        )
    }

    #[tokio::test]
    async fn recording_without_a_store_updates_totals() {
        let mut ledger = RunLedger::new("task");
        assert!(ledger
            .record_usage(record(
                "task",
                "c1",
                UsageTokens::new(Some(3), Some(2), None)
            ))
            .await
            .unwrap());
        assert_eq!(ledger.totals().calls, 1);
        assert_eq!(ledger.totals().input_tokens, 3);
    }

    #[tokio::test]
    async fn clear_resets_the_ledger() {
        let store = Arc::new(InMemoryLedgerStore::new());
        let mut ledger = RunLedger::new("task").with_store(store.clone());
        ledger.bind().await.unwrap();
        ledger
            .record_usage(record("task", "c1", UsageTokens::new(Some(3), None, None)))
            .await
            .unwrap();
        ledger.clear().await.unwrap();
        assert_eq!(ledger.records().len(), 0);
        assert_eq!(ledger.totals(), UsageTotals::default());
        assert_eq!(store.events("task").len(), 0);
    }
}
