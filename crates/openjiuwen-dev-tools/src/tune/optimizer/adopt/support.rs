//! Shared fixtures for the adopt test suites.
//!
//! Two fakes are needed by more than one suite, so they live here instead of
//! being copied per file: [`RecordingMemory`] (a [`PromptMemory`] that keeps its
//! records in memory and can be made to fail) and [`FixedDriftJudge`] (a
//! [`DriftJudge`] returning a constant deviation). [`evaluated`] builds an
//! [`EvaluatedCase`] deterministically.

use std::collections::BTreeMap;
use std::sync::{Mutex, MutexGuard, PoisonError};

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use serde_json::json;

use crate::tune::prompt_search::{DriftJudge, PromptMemory, PromptRecord, PromptTaskSpec};
use crate::tune::{Case, EvaluatedCase, PromptCandidate};

/// A prompt memory that keeps records in memory and can fail on `store`.
pub(crate) struct RecordingMemory {
    /// The stored records, in insertion order.
    records: Mutex<Vec<PromptRecord>>,
    /// Whether `store` reports an error.
    fail: bool,
}

impl RecordingMemory {
    /// A working store.
    pub(crate) fn new() -> Self {
        Self {
            records: Mutex::new(Vec::new()),
            fail: false,
        }
    }

    /// A store whose `store` call always fails.
    pub(crate) fn failing() -> Self {
        Self {
            records: Mutex::new(Vec::new()),
            fail: true,
        }
    }

    /// The records currently stored.
    pub(crate) fn stored(&self) -> Vec<PromptRecord> {
        lock(&self.records).clone()
    }
}

impl PromptMemory for RecordingMemory {
    fn store(&self, record: PromptRecord) -> Result<()> {
        if self.fail {
            return Err(Error::validation("store is closed"));
        }
        lock(&self.records).push(record);
        Ok(())
    }

    fn search_similar(&self, _task: &PromptTaskSpec, top_k: usize) -> Vec<PromptRecord> {
        self.stored().into_iter().take(top_k).collect()
    }

    fn best_for_objective(&self, objective: &str) -> Option<PromptRecord> {
        self.stored()
            .into_iter()
            .filter(|record| record.objective == objective)
            .fold(None, |best: Option<PromptRecord>, record| match best {
                Some(current) if current.reward >= record.reward => Some(current),
                _ => Some(record),
            })
    }

    fn pending(&self, threshold: f64) -> Vec<PromptRecord> {
        self.stored()
            .into_iter()
            .filter(|record| record.is_pending(threshold))
            .collect()
    }

    fn mark_applied(&self, record_id: &str) -> Result<bool> {
        let mut found = false;
        for record in lock(&self.records).iter_mut() {
            if record.record_id == record_id {
                record.applied = true;
                found = true;
            }
        }
        Ok(found)
    }

    fn records(&self) -> Vec<PromptRecord> {
        self.stored()
    }
}

/// A drift judge that reports a fixed deviation.
pub(crate) struct FixedDriftJudge {
    /// The deviation every measurement returns.
    pub(crate) deviation: f64,
}

impl FixedDriftJudge {
    /// Creates a judge reporting `deviation`.
    pub(crate) fn new(deviation: f64) -> Self {
        Self { deviation }
    }
}

#[async_trait]
impl DriftJudge for FixedDriftJudge {
    async fn deviation(&self, _objective: &str, _candidate: &PromptCandidate) -> f64 {
        self.deviation
    }
}

/// An evaluated case with `id`, `score`, a fixed answer, and `reason`.
pub(crate) fn evaluated(id: &str, score: f64, reason: &str) -> EvaluatedCase {
    let inputs = BTreeMap::from([("query".to_owned(), json!(format!("question {id}")))]);
    let label = BTreeMap::from([("output".to_owned(), json!(format!("answer {id}")))]);
    let case = Case::new(inputs, label).unwrap().with_case_id(id);
    let answer = BTreeMap::from([("output".to_owned(), json!(format!("given {id}")))]);
    EvaluatedCase::new(case, answer, score, reason).unwrap()
}

/// Locks `mutex`, recovering the data when a previous holder panicked.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}
