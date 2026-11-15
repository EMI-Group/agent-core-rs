//! [`AdoptMemory`]: what this run adopted, plus the persistent prompt memory.
//!
//! Python's `AdoptOptimizer` keeps no record of what it changed: it rewrites
//! every `LLMCall` in place and returns. The port records every verdict, so a
//! caller can report what an adoption run decided, and — when a
//! [`PromptMemory`] is injected — persist the adopted prompts so a later run can
//! use them as a baseline.
//!
//! Two rules keep this honest:
//!
//! - **An adopted record is marked applied** with the caller-supplied `now`
//!   (never a clock read), so it stops appearing in
//!   [`PromptMemory::pending`].
//! - **A rejected record is never persisted**: it is kept only in this run's
//!   [`AdoptMemory::rejected`] list, because a prompt that did not clear the
//!   policy must not become history a later run reasons from.
//!
//! A failing [`PromptMemory::store`] is logged with `tracing::warn!` and never
//! fails the caller, matching the crate's prompt-search memory convention: the
//! adoption verdict has already been reached, and losing the archive must not
//! undo it.

use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use crate::tune::prompt_search::{NullPromptMemory, PromptMemory, PromptRecord};

/// The adoption bookkeeping of one run, backed by an optional prompt memory.
pub struct AdoptMemory {
    /// The persistent store (a [`NullPromptMemory`] when none was injected).
    memory: Arc<dyn PromptMemory>,
    /// The records this run adopted, in decision order.
    adopted: Mutex<Vec<PromptRecord>>,
    /// The records this run rejected, in decision order.
    rejected: Mutex<Vec<PromptRecord>>,
}

impl std::fmt::Debug for AdoptMemory {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AdoptMemory")
            .field("adopted", &self.adopted().len())
            .field("rejected", &self.rejected().len())
            .finish()
    }
}

impl Default for AdoptMemory {
    /// A memory with no persistent store behind it.
    fn default() -> Self {
        Self::new()
    }
}

impl AdoptMemory {
    /// Creates the bookkeeping over a [`NullPromptMemory`].
    #[must_use]
    pub fn new() -> Self {
        Self::with_memory(Arc::new(NullPromptMemory::new()))
    }

    /// Creates the bookkeeping over an injected store.
    #[must_use]
    pub fn with_memory(memory: Arc<dyn PromptMemory>) -> Self {
        Self {
            memory,
            adopted: Mutex::new(Vec::new()),
            rejected: Mutex::new(Vec::new()),
        }
    }

    /// The persistent store in force.
    #[must_use]
    pub fn memory(&self) -> &Arc<dyn PromptMemory> {
        &self.memory
    }

    /// Marks `record` applied at `now`, persists it, and remembers the decision.
    ///
    /// The returned record is the one this run holds — `applied == true` and
    /// `applied_at == Some(now)` — so a caller can report exactly what was
    /// installed. A store failure is logged and does not affect the returned
    /// record.
    pub fn record_adopted(&self, record: PromptRecord, now: f64) -> PromptRecord {
        let stored = record.with_applied(true, Some(now));
        if let Err(error) = self.memory.store(stored.clone()) {
            tracing::warn!(
                %error,
                "adopt memory store failed; the adoption verdict is unaffected"
            );
        }
        lock(&self.adopted).push(stored.clone());
        stored
    }

    /// Remembers a rejected decision without persisting it.
    ///
    /// The record keeps `applied == false` and `applied_at == None`; `now` fills
    /// [`PromptRecord::created_at`] when the caller did not supply a creation
    /// timestamp, so the run list stays ordered in time.
    pub fn record_rejected(&self, record: PromptRecord, now: f64) -> PromptRecord {
        let mut rejected = record.with_applied(false, None);
        if rejected.created_at == 0.0 {
            rejected.created_at = now;
        }
        lock(&self.rejected).push(rejected.clone());
        rejected
    }

    /// The records this run adopted, in decision order.
    #[must_use]
    pub fn adopted(&self) -> Vec<PromptRecord> {
        lock(&self.adopted).clone()
    }

    /// The records this run rejected, in decision order.
    #[must_use]
    pub fn rejected(&self) -> Vec<PromptRecord> {
        lock(&self.rejected).clone()
    }

    /// The best reward known for `objective`, before or after this run.
    ///
    /// The persistent store's best record for the objective wins; this run's own
    /// adoptions are considered too, so a run backed by the default
    /// [`NullPromptMemory`] still reports a baseline for its second decision.
    /// Only finite rewards count.
    #[must_use]
    pub fn previous_best(&self, objective: &str) -> Option<f64> {
        let mut best = self
            .memory
            .best_for_objective(objective)
            .map(|record| record.reward)
            .filter(|reward| reward.is_finite());
        for record in lock(&self.adopted)
            .iter()
            .filter(|record| record.objective == objective && record.reward.is_finite())
        {
            best = Some(match best {
                Some(current) => current.max(record.reward),
                None => record.reward,
            });
        }
        best
    }

    /// The persistent store's unapplied records with a gain above `threshold`.
    #[must_use]
    pub fn pending(&self, threshold: f64) -> Vec<PromptRecord> {
        self.memory.pending(threshold)
    }
}

/// Locks `mutex`, recovering the data when a previous holder panicked.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::super::support::RecordingMemory;
    use super::*;

    fn record(id: &str, objective: &str, reward: f64) -> PromptRecord {
        PromptRecord::new(id, format!("prompt {id}"), reward, objective)
    }

    #[test]
    fn default_memory_stores_nothing_but_still_records_the_run() {
        let memory = AdoptMemory::new();
        assert_eq!(memory.pending(0.0).len(), 0);
        assert!(memory.previous_best("objective").is_none());
        let stored = memory.record_adopted(record("r1", "objective", 0.8), 5.0);
        assert!(stored.applied);
        assert_eq!(stored.applied_at, Some(5.0));
        assert_eq!(memory.adopted().len(), 1);
        assert_eq!(memory.adopted()[0].record_id, "r1");
        assert_eq!(memory.rejected().len(), 0);
        // The run's own adoption supplies the baseline even without a store.
        assert_eq!(memory.previous_best("objective"), Some(0.8));
        assert!(memory.previous_best("other").is_none());
        assert_eq!(memory.pending(0.0).len(), 0);
    }

    #[test]
    fn adopted_records_are_persisted_marked_applied() {
        let store = Arc::new(RecordingMemory::new());
        let memory = AdoptMemory::with_memory(store.clone());
        assert_eq!(memory.memory().records().len(), 0);
        memory.record_adopted(record("r1", "objective", 0.8), 5.0);
        let stored = store.stored();
        assert_eq!(stored.len(), 1);
        assert!(stored[0].applied);
        assert_eq!(stored[0].applied_at, Some(5.0));
        assert_eq!(memory.pending(0.0).len(), 0);
        // A previously stored, unapplied record with a gain is still pending.
        store
            .store(
                record("r0", "objective", 0.4)
                    .with_baseline_reward(0.1)
                    .with_created_at(1.0),
            )
            .unwrap();
        assert_eq!(memory.pending(0.1).len(), 1);
        assert_eq!(memory.previous_best("objective"), Some(0.8));
    }

    #[test]
    fn rejected_records_stay_in_the_run_and_never_reach_the_store() {
        let store = Arc::new(RecordingMemory::new());
        let memory = AdoptMemory::with_memory(store.clone());
        let rejected = memory.record_rejected(record("r1", "objective", 0.2), 7.0);
        assert!(!rejected.applied);
        assert!(rejected.applied_at.is_none());
        assert_eq!(rejected.created_at, 7.0);
        assert_eq!(memory.rejected().len(), 1);
        assert_eq!(memory.adopted().len(), 0);
        assert_eq!(store.stored().len(), 0);
        assert!(memory.previous_best("objective").is_none());
        // An existing creation timestamp is preserved.
        let stamped =
            memory.record_rejected(record("r2", "objective", 0.1).with_created_at(3.0), 9.0);
        assert_eq!(stamped.created_at, 3.0);
    }

    #[test]
    fn decisions_keep_their_order() {
        let memory = AdoptMemory::new();
        memory.record_adopted(record("a", "objective", 0.9), 1.0);
        memory.record_rejected(record("b", "objective", 0.2), 2.0);
        memory.record_adopted(record("c", "objective", 0.95), 3.0);
        let adopted: Vec<String> = memory
            .adopted()
            .into_iter()
            .map(|record| record.record_id)
            .collect();
        assert_eq!(adopted, vec!["a".to_owned(), "c".to_owned()]);
        assert_eq!(memory.rejected()[0].record_id, "b");
        assert_eq!(memory.previous_best("objective"), Some(0.95));
    }

    #[test]
    fn a_failing_store_is_logged_and_not_fatal() {
        let store = Arc::new(RecordingMemory::failing());
        let memory = AdoptMemory::with_memory(store.clone());
        let stored = memory.record_adopted(record("r1", "objective", 0.9), 2.0);
        assert!(stored.applied);
        assert_eq!(memory.adopted().len(), 1);
        // `best_for_objective` reads what was written; nothing was.
        assert_eq!(memory.previous_best("objective"), Some(0.9));
        assert_eq!(store.stored().len(), 0);
    }

    #[test]
    fn debug_reports_the_decision_counts() {
        let memory = AdoptMemory::new();
        assert_eq!(
            format!("{memory:?}"),
            "AdoptMemory { adopted: 0, rejected: 0 }"
        );
        memory.record_adopted(record("a", "objective", 0.9), 1.0);
        memory.record_rejected(record("b", "objective", 0.1), 2.0);
        assert_eq!(
            format!("{memory:?}"),
            "AdoptMemory { adopted: 1, rejected: 1 }"
        );
        assert_eq!(AdoptMemory::default().adopted().len(), 0);
    }
}
