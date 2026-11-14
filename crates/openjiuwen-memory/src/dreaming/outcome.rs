//! Dream outcomes and run summaries: what one consolidation attempt produced.
//!
//! [`DreamOutcome`] is what a consolidator reports (or what the planner synthesises when
//! an attempt fails or is cancelled) and what the repository stores on completion;
//! [`DreamRunSummary`] aggregates the outcomes of one sweep. Both are plain `serde` data
//! with the same tolerance rules as [`super::model`].
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};

use super::model::{validate_non_empty, DreamMetrics, DreamState};
use super::sweep_lock::SweepLockSkip;

/// The result of one completed dream attempt.
///
/// A [`DreamRepository::complete`](crate::dreaming::DreamRepository::complete) call
/// stores the outcome's state and metrics on the job; the notes stay with the caller
/// (they may be logged or merged into a [`DreamRunSummary`]).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct DreamOutcome {
    /// Id of the job this outcome belongs to.
    pub job_id: String,
    /// Terminal state the job finished in.
    pub state: DreamState,
    /// Counters describing the attempt.
    pub metrics: DreamMetrics,
    /// Human-readable notes, e.g. why the job was skipped.
    pub notes: Vec<String>,
}

impl DreamOutcome {
    /// Builds an outcome for `job_id` in `state`.
    #[must_use]
    pub fn new(job_id: impl Into<String>, state: DreamState) -> Self {
        Self {
            job_id: job_id.into(),
            state,
            ..Self::default()
        }
    }

    /// Builds a successful outcome carrying `metrics`.
    #[must_use]
    pub fn succeeded(job_id: impl Into<String>, metrics: DreamMetrics) -> Self {
        Self::new(job_id, DreamState::Succeeded).with_metrics(metrics)
    }

    /// Builds a skipped outcome with one explanatory note.
    #[must_use]
    pub fn skipped(job_id: impl Into<String>, note: impl Into<String>) -> Self {
        Self::new(job_id, DreamState::Skipped).with_note(note)
    }

    /// Builds a cancelled outcome.
    #[must_use]
    pub fn cancelled(job_id: impl Into<String>) -> Self {
        Self::new(job_id, DreamState::Cancelled)
    }

    /// Builds a failed outcome with one explanatory note.
    #[must_use]
    pub fn failed(job_id: impl Into<String>, note: impl Into<String>) -> Self {
        Self::new(job_id, DreamState::Failed).with_note(note)
    }

    /// Returns the outcome carrying `metrics`.
    #[must_use]
    pub const fn with_metrics(mut self, metrics: DreamMetrics) -> Self {
        self.metrics = metrics;
        self
    }

    /// Returns the outcome with one more note.
    #[must_use]
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    /// Checks that this is a usable completion: a non-empty NUL-free job id and a
    /// terminal state.
    ///
    /// `Pending`/`Running` are rejected because a completed run must describe where the
    /// job ended up; use [`DreamRepository::fail`](crate::dreaming::DreamRepository::fail)
    /// for an attempt that should be retried.
    pub fn validate(&self) -> Result<()> {
        validate_non_empty("dream outcome `job_id`", &self.job_id)?;
        if !self.state.is_terminal() {
            return Err(Error::validation(format!(
                "dream outcome state `{}` is not terminal: a completed run needs a terminal state",
                self.state
            )));
        }
        Ok(())
    }
}

/// Aggregate over one [`DreamRepository`](crate::dreaming::DreamRepository) sweep.
///
/// `failed` counts failed *attempts*, including attempts that were rescheduled for a
/// retry, so it can exceed the number of jobs that ended up in
/// [`DreamState::Failed`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct DreamRunSummary {
    /// Attempts recorded by the sweep.
    pub scanned: usize,
    /// Attempts that finished successfully.
    pub succeeded: usize,
    /// Attempts that failed (including retries that were requeued).
    pub failed: usize,
    /// Attempts that found nothing to do.
    pub skipped: usize,
    /// Attempts that were cancelled.
    pub cancelled: usize,
    /// The individual outcomes, in recording order.
    pub outcomes: Vec<DreamOutcome>,
    /// Why this run skipped its sweep because the sweep lock was unavailable; default `None`.
    ///
    /// A run that reports a skip did no work at all (`scanned == 0`): no job was claimed and
    /// no retention pass ran. The field is additive and `serde` omits it when absent, so
    /// summaries written before it existed still round-trip.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lock_skip: Option<SweepLockSkip>,
}

impl DreamRunSummary {
    /// Builds an empty summary.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Builds a summary that reports a skipped sweep: no work done, and `skip` as the reason.
    #[must_use]
    pub fn lock_skipped(skip: SweepLockSkip) -> Self {
        Self {
            lock_skip: Some(skip),
            ..Self::default()
        }
    }

    /// Whether this summary reports a skipped sweep rather than sweep work.
    #[must_use]
    pub const fn is_lock_skip(&self) -> bool {
        self.lock_skip.is_some()
    }

    /// Returns `true` when no outcome has been recorded.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.scanned == 0
    }

    /// Records one outcome, updating the counters and appending it to
    /// [`DreamRunSummary::outcomes`].
    ///
    /// Non-terminal outcomes (which [`DreamOutcome::validate`] rejects) only bump
    /// `scanned`, since there is no counter to attribute them to.
    pub fn record(&mut self, outcome: DreamOutcome) {
        self.scanned = self.scanned.saturating_add(1);
        match outcome.state {
            DreamState::Succeeded => self.succeeded = self.succeeded.saturating_add(1),
            DreamState::Failed => self.failed = self.failed.saturating_add(1),
            DreamState::Skipped => self.skipped = self.skipped.saturating_add(1),
            DreamState::Cancelled => self.cancelled = self.cancelled.saturating_add(1),
            DreamState::Pending | DreamState::Running => {}
        }
        self.outcomes.push(outcome);
    }

    /// Folds `other` into `self`, concatenating the outcome lists.
    ///
    /// A skip marker is sticky: `self` keeps its own reason, and takes `other`'s when it has
    /// none, so a merged total still remembers why part of it did nothing.
    pub fn merge(&mut self, other: Self) {
        self.scanned = self.scanned.saturating_add(other.scanned);
        self.succeeded = self.succeeded.saturating_add(other.succeeded);
        self.failed = self.failed.saturating_add(other.failed);
        self.skipped = self.skipped.saturating_add(other.skipped);
        self.cancelled = self.cancelled.saturating_add(other.cancelled);
        self.outcomes.extend(other.outcomes);
        if self.lock_skip.is_none() {
            self.lock_skip = other.lock_skip;
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use serde_json::json;

    fn assert_validation(error: Error, needle: &str) {
        assert!(matches!(error, Error::Validation(_)), "got {error:?}");
        assert!(
            error.to_string().contains(needle),
            "`{error}` should mention `{needle}`"
        );
    }

    #[test]
    fn outcome_constructors_validate_and_serde() {
        let metrics = DreamMetrics {
            records_scanned: 1,
            ..DreamMetrics::default()
        };
        assert_eq!(
            DreamOutcome::succeeded("j", metrics),
            DreamOutcome::new("j", DreamState::Succeeded).with_metrics(metrics)
        );
        assert_eq!(
            DreamOutcome::skipped("j", "nothing to do"),
            DreamOutcome::new("j", DreamState::Skipped).with_note("nothing to do")
        );
        assert_eq!(
            DreamOutcome::failed("j", "boom"),
            DreamOutcome::new("j", DreamState::Failed).with_note("boom")
        );
        assert_eq!(
            DreamOutcome::cancelled("j"),
            DreamOutcome::new("j", DreamState::Cancelled)
        );
        assert_eq!(
            DreamOutcome::cancelled("j")
                .with_note("first")
                .with_note("second")
                .notes,
            vec!["first".to_owned(), "second".to_owned()]
        );
        for state in [DreamState::Pending, DreamState::Running] {
            assert_validation(
                DreamOutcome::new("j", state).validate().unwrap_err(),
                "not terminal",
            );
        }
        assert_validation(
            DreamOutcome::new("", DreamState::Succeeded)
                .validate()
                .unwrap_err(),
            "job_id",
        );
        assert_validation(
            DreamOutcome::new("a\0b", DreamState::Succeeded)
                .validate()
                .unwrap_err(),
            "NUL",
        );
        assert!(DreamOutcome::succeeded("j", DreamMetrics::default())
            .validate()
            .is_ok());

        let restored: DreamOutcome = serde_json::from_value(json!({})).unwrap();
        assert_eq!(restored, DreamOutcome::default());
        let outcome = DreamOutcome::failed("j", "boom").with_metrics(DreamMetrics {
            records_deleted: 2,
            ..DreamMetrics::default()
        });
        let value = serde_json::to_value(&outcome).unwrap();
        assert_eq!(value["state"], json!("failed"));
        assert_eq!(value["metrics"]["records_deleted"], json!(2));
        assert_eq!(
            serde_json::from_value::<DreamOutcome>(value).unwrap(),
            outcome
        );
    }

    #[test]
    fn summary_records_merges_and_serde() {
        let mut summary = DreamRunSummary::new();
        assert!(summary.is_empty());
        summary.record(DreamOutcome::succeeded("a", DreamMetrics::default()));
        summary.record(DreamOutcome::failed("b", "boom"));
        summary.record(DreamOutcome::skipped("c", "nothing"));
        summary.record(DreamOutcome::cancelled("d"));
        assert_eq!(
            (summary.scanned, summary.succeeded, summary.failed),
            (4, 1, 1)
        );
        assert_eq!((summary.skipped, summary.cancelled), (1, 1));
        assert_eq!(summary.outcomes.len(), 4);

        // Non-terminal outcomes only move `scanned`.
        summary.record(DreamOutcome::new("e", DreamState::Running));
        assert_eq!(summary.scanned, 5);
        assert_eq!(
            summary.succeeded + summary.failed + summary.skipped + summary.cancelled,
            4
        );

        let mut right = DreamRunSummary::new();
        right.record(DreamOutcome::failed("f", "boom"));
        right.record(DreamOutcome::cancelled("g"));
        summary.merge(right);
        assert_eq!(
            (summary.scanned, summary.failed, summary.cancelled),
            (7, 2, 2)
        );
        assert_eq!(summary.outcomes.len(), 7);
        assert_eq!(
            serde_json::from_value::<DreamRunSummary>(json!({})).unwrap(),
            DreamRunSummary::default()
        );
    }
}
