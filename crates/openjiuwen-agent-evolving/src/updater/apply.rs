//! Applying normalized updates to evolution targets.
//!
//! Ports Python `openjiuwen.agent_evolving.update_execution`: [`execute_updates`]
//! normalizes and applies a batch of updates, and [`summarize_apply_results`]
//! aggregates the outcome. The seam differs from Python only where Python is
//! duck-typed: an operator is any [`UpdateTarget`], and the target registry is an
//! [`OperatorMap`] keyed by operator id.
//!
//! Execution keeps Python's two-pass contract: every non-`None` value is applied
//! first in map order, then every `None`-valued key is reported last. A missing
//! operator, an unsupported `mode`/`effect` pair, and a target write failure each
//! produce an [`ApplyResult`] with `applied == false` and a message in `errors` —
//! nothing is dropped and nothing fails silently.
//!
//! The seam is replace-only, matching Python's `Operator.apply_update`
//! compatibility default: `replace` + `state` reaches [`UpdateTarget::set_parameter`]
//! and any other combination is rejected with the same message Python uses. A
//! target that can honour `append`/`merge`/`pending_change` needs a richer seam
//! on the Python side (`preview_update`), which this port does not model.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use serde_json::Value;

use openjiuwen_common::error::{Error, Result};

use super::types::{
    normalize_update_value, ApplyResult, ApplySummary, Update, UpdateDomain, UpdateEffect,
    UpdateKey, UpdateMode, Updates,
};

/// Anything that can receive a normalized parameter write.
///
/// Implementations own their state; the updater crate never persists or approves
/// anything, so a write is the whole contract.
pub trait UpdateTarget: Send + Sync {
    /// Writes `value` to `target` of `operator_id` (replace semantics).
    ///
    /// # Errors
    ///
    /// Any error the target raises is reported as a failed [`ApplyResult`] rather
    /// than aborting the whole batch.
    fn set_parameter(&self, operator_id: &str, target: &str, value: Value) -> Result<()>;

    /// The domain `operator_id` belongs to; defaults to [`UpdateDomain::Llm`].
    ///
    /// [`crate::updater::MultiDimUpdater`] routes operators with this method, so
    /// tool and memory targets must override it.
    fn domain(&self, _operator_id: &str) -> UpdateDomain {
        UpdateDomain::Llm
    }
}

/// The targets a batch is applied to, keyed by operator id.
pub type OperatorMap = BTreeMap<String, Arc<dyn UpdateTarget>>;

/// Normalizes and executes a batch of updates (Python `execute_updates`).
///
/// Non-`None` values are applied in map order, then `None`-valued keys are
/// appended as failures with `errors = ["update value is None"]`. The returned
/// vector is therefore ordered: applied-or-attempted updates first, `None` keys
/// last.
#[must_use]
pub fn execute_updates(operators: &OperatorMap, updates: &Updates) -> Vec<ApplyResult> {
    let mut results = Vec::with_capacity(updates.len());
    for (operator_id, target, value) in updates.iter() {
        let Some(value) = value else {
            continue;
        };
        results.push(apply_one(
            operators,
            &(operator_id.to_owned(), target.to_owned()),
            value,
        ));
    }
    for (operator_id, target, value) in updates.iter() {
        if value.is_some() {
            continue;
        }
        results.push(ApplyResult::rejected(
            operator_id,
            target,
            "update value is None",
        ));
    }
    results
}

/// Applies one normalized update to the operator it addresses.
fn apply_one(operators: &OperatorMap, key: &UpdateKey, value: &Update) -> ApplyResult {
    let normalized = normalize_update_value(value, &key.1);
    let mut result = ApplyResult {
        operator_id: key.0.clone(),
        target: key.1.clone(),
        mode: normalized.mode,
        effect: normalized.effect,
        value: normalized.payload.clone(),
        change_type: normalized.change_type.clone(),
        metadata: normalized.metadata.clone(),
        ..ApplyResult::default()
    };
    let Some(target) = operators.get(&key.0) else {
        result.errors.push(format!("operator not found: {}", key.0));
        return result;
    };
    if normalized.mode != UpdateMode::Replace || normalized.effect != UpdateEffect::State {
        result.errors.push(format!(
            "unsupported update mode/effect for compatibility operator: {}/{}",
            normalized.mode.as_str(),
            normalized.effect.as_str()
        ));
        return result;
    }
    match target.set_parameter(&key.0, &key.1, normalized.payload) {
        Ok(()) => result.applied = true,
        Err(error) => result.errors.push(error.to_string()),
    }
    result
}

/// Compatibility alias for [`execute_updates`] (Python `apply_updates`).
#[must_use]
pub fn apply_updates(operators: &OperatorMap, updates: &Updates) -> Vec<ApplyResult> {
    execute_updates(operators, updates)
}

/// Counts total/applied/failed over `results` (Python `summarize_apply_results`).
#[must_use]
pub fn summarize_apply_results(results: &[ApplyResult]) -> ApplySummary {
    let applied = results.iter().filter(|result| result.applied).count();
    ApplySummary {
        total: results.len(),
        applied,
        failed: results.len() - applied,
    }
}

/// A deterministic [`UpdateTarget`] that records every write.
///
/// Used by tests and examples: [`Self::writes`] replays the `(operator_id,
/// target, value)` triples in call order, and [`Self::with_rejected`] makes a
/// chosen target fail so the failure path is exercisable without a real operator.
#[derive(Debug, Default)]
pub struct RecordingTarget {
    domain: UpdateDomain,
    rejected: Mutex<Vec<String>>,
    writes: Mutex<Vec<(String, String, Value)>>,
}

impl RecordingTarget {
    /// Creates a target in the [`UpdateDomain::Llm`] domain that accepts every write.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a target that reports `domain` from [`UpdateTarget::domain`].
    #[must_use]
    pub fn with_domain(domain: UpdateDomain) -> Self {
        Self {
            domain,
            ..Self::default()
        }
    }

    /// Makes writes to `target` fail with an [`Error::validation`].
    #[must_use]
    pub fn with_rejected(self, target: impl Into<String>) -> Self {
        self.rejected
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(target.into());
        self
    }

    /// Every accepted write, in call order.
    #[must_use]
    pub fn writes(&self) -> Vec<(String, String, Value)> {
        lock(&self.writes).clone()
    }

    /// The targets written to, in call order.
    #[must_use]
    pub fn written_targets(&self) -> Vec<String> {
        lock(&self.writes)
            .iter()
            .map(|(_, target, _)| target.clone())
            .collect()
    }

    /// The last value written to `(operator_id, target)`, if any.
    #[must_use]
    pub fn value(&self, operator_id: &str, target: &str) -> Option<Value> {
        lock(&self.writes)
            .iter()
            .rev()
            .find(|(operator, written, _)| operator == operator_id && written == target)
            .map(|(_, _, value)| value.clone())
    }

    /// Number of accepted writes.
    #[must_use]
    pub fn len(&self) -> usize {
        lock(&self.writes).len()
    }

    /// Whether nothing was written yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        lock(&self.writes).is_empty()
    }
}

impl UpdateTarget for RecordingTarget {
    fn set_parameter(&self, operator_id: &str, target: &str, value: Value) -> Result<()> {
        if lock(&self.rejected).iter().any(|rejected| rejected == target) {
            return Err(Error::validation(format!(
                "RecordingTarget rejects target {target:?} for operator {operator_id:?}"
            )));
        }
        lock(&self.writes).push((operator_id.to_owned(), target.to_owned(), value));
        Ok(())
    }

    fn domain(&self, _operator_id: &str) -> UpdateDomain {
        self.domain
    }
}

/// Locks without panicking on poisoning: a poisoned recorder still holds data.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::updater::types::{UpdateValue, EXPERIENCES_TARGET};
    use serde_json::json;

    fn operators(target: Arc<RecordingTarget>) -> OperatorMap {
        let mut operators = OperatorMap::new();
        operators.insert("op".to_owned(), target);
        operators
    }

    #[test]
    fn applies_non_none_updates_then_reports_none_ones_last() {
        let target = Arc::new(RecordingTarget::new());
        let mut updates = Updates::new();
        updates.insert(("op".to_owned(), "body".to_owned()), None);
        updates.insert_update("op", "description", json!("new"));
        updates.insert(("missing".to_owned(), "body".to_owned()), None);
        updates.insert_update("missing", "body", json!("x"));

        let results = execute_updates(&operators(Arc::clone(&target)), &updates);
        let order: Vec<(&str, &str)> = results
            .iter()
            .map(|result| (result.operator_id.as_str(), result.target.as_str()))
            .collect();
        assert_eq!(
            order,
            [
                ("op", "description"),
                ("missing", "body"),
                ("op", "body"),
                ("missing", "body"),
            ]
        );
        assert!(results[0].ok());
        assert_eq!(results[0].mode, UpdateMode::Replace);
        assert_eq!(results[0].value, json!("new"));
        assert_eq!(results[1].errors, ["operator not found: missing"]);
        assert!(!results[1].applied);
        assert_eq!(results[2].errors, ["update value is None"]);
        assert_eq!(results[3].errors, ["update value is None"]);
        assert_eq!(target.writes(), [(
            "op".to_owned(),
            "description".to_owned(),
            json!("new")
        )]);
        let summary = summarize_apply_results(&results);
        assert_eq!(summary, ApplySummary {
            total: 4,
            applied: 1,
            failed: 3
        });
    }

    #[test]
    fn experiences_updates_are_rejected_by_the_replace_only_seam() {
        let target = Arc::new(RecordingTarget::new());
        let mut updates = Updates::new();
        updates.insert_update("op", EXPERIENCES_TARGET, json!(["entry"]));
        let results = execute_updates(&operators(Arc::clone(&target)), &updates);
        assert_eq!(results.len(), 1);
        assert!(!results[0].applied);
        assert_eq!(results[0].mode, UpdateMode::Append);
        assert_eq!(results[0].change_type.as_deref(), Some(super::super::types::SKILL_EXPERIENCE_ENTRY));
        assert_eq!(
            results[0].errors,
            ["unsupported update mode/effect for compatibility operator: append/pending_change"]
        );
        assert!(target.is_empty());
    }

    #[test]
    fn target_errors_become_failed_results_instead_of_aborting_the_batch() {
        let target = Arc::new(RecordingTarget::new().with_rejected("body"));
        let mut updates = Updates::new();
        updates.insert_update("op", "body", json!("x"));
        updates.insert_update("op", "description", json!("y"));
        let results = execute_updates(&operators(Arc::clone(&target)), &updates);
        assert_eq!(results.len(), 2);
        assert!(!results[0].applied);
        assert!(results[0].errors[0].contains("rejects target"));
        assert!(results[1].ok());
        assert_eq!(summarize_apply_results(&results).failed, 1);
        assert_eq!(target.written_targets(), ["description"]);
    }

    #[test]
    fn structured_values_keep_their_metadata_on_the_result() {
        let target = Arc::new(RecordingTarget::new());
        let mut value = UpdateValue::new(json!("text"));
        value.metadata.insert("scope".to_owned(), json!("s1"));
        let mut updates = Updates::new();
        updates.insert_update("op", "body", value);
        let results = execute_updates(&operators(Arc::clone(&target)), &updates);
        assert_eq!(results[0].metadata.get("scope"), Some(&json!("s1")));
        assert!(results[0].ok());
    }

    #[test]
    fn apply_updates_is_an_alias() {
        let target = Arc::new(RecordingTarget::new());
        let mut updates = Updates::new();
        updates.insert_update("op", "body", json!(1));
        assert_eq!(
            apply_updates(&operators(target), &updates),
            execute_updates(&operators(Arc::new(RecordingTarget::new())), &updates)
        );
    }

    #[test]
    fn empty_batch_summarises_to_zeroes() {
        let summary = summarize_apply_results(&[]);
        assert_eq!(summary, ApplySummary::default());
        assert!(summary.is_ok());
    }
}
