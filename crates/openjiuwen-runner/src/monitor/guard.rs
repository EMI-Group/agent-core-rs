//! The run-scoped status guard the execution paths drive.
//!
//! A run path builds one [`RunStatusGuard`] when it admits a run and calls
//! `running` just before the invocation, then exactly one of `ok`/`fail`/
//! `cancelled`. Dropping an unfinished guard finishes the run as cancelled, so a
//! dropped future, an `?`-returned error, or a panicking task never leaves a live
//! run behind in the monitor.
//!
//! Every method is synchronous and best-effort: the guard is bookkeeping, not
//! control flow, so a mirror failure (the run was finished elsewhere, or already
//! evicted) is logged at `warn` and never propagated to the run.

use std::sync::Arc;

use openjiuwen_common::error::{Error, ErrorKind};
use serde_json::{json, Value};
use tracing::warn;

use super::status::{RunOutcome, RunPhase, RunStart};
use super::RunMonitor;

/// The tracing target every guard warning uses.
const TARGET: &str = "openjiuwen_runner::monitor";

/// The run-dropped detail recorded when an unfinished guard is dropped.
const DROPPED_DETAIL: &str = "run dropped";

/// Mirrors one run's lifecycle into a [`RunMonitor`], if one is attached.
///
/// Constructed with [`RunStatusGuard::begin`]; `monitor == None` yields an inert
/// guard whose methods do nothing (the runner without a monitor pays nothing).
pub(crate) struct RunStatusGuard {
    /// The monitor to mirror into; `None` makes the guard inert.
    monitor: Option<Arc<RunMonitor>>,
    /// The run id the transitions are published under.
    run_id: String,
    /// Whether a terminal transition (explicit, on drop, or attempted) happened.
    finished: bool,
}

impl RunStatusGuard {
    /// Adopts `run_id` as [`RunPhase::Queued`], creating it when it is unknown.
    ///
    /// An already-tracked id keeps the status it has (no event is published), which
    /// is what a run path that pre-registered the run wants.
    pub(crate) fn begin(monitor: Option<Arc<RunMonitor>>, run_id: &str, start: RunStart) -> Self {
        if let Some(active) = monitor.as_ref() {
            let _ = active.track(run_id, start);
        }
        Self {
            monitor,
            run_id: run_id.to_string(),
            finished: false,
        }
    }

    /// Marks the run [`RunPhase::Running`], just before the invocation.
    pub(crate) fn running(&mut self, detail: Value) {
        self.advance(RunPhase::Running, detail);
    }

    /// Finishes the run successfully.
    pub(crate) fn ok(&mut self, detail: Value) {
        self.finish(RunOutcome::completed().with_detail(detail));
    }

    /// Finishes the run with `error`: [`ErrorKind::Cancelled`] maps to
    /// [`RunPhase::Cancelled`], everything else to [`RunPhase::Failed`], and in both
    /// cases the recorded message is `error.to_string()`.
    pub(crate) fn fail(&mut self, error: &Error, detail: Value) {
        let message = error.to_string();
        let outcome = if error.kind() == ErrorKind::Cancelled {
            RunOutcome {
                error: Some(message),
                ..RunOutcome::cancelled()
            }
        } else {
            RunOutcome::failed(message)
        };
        self.finish(outcome.with_detail(detail));
    }

    /// Finishes the run as cancelled.
    pub(crate) fn cancelled(&mut self, detail: Value) {
        self.finish(RunOutcome::cancelled().with_detail(detail));
    }

    /// Publishes a non-terminal transition, logging a mirror failure.
    fn advance(&mut self, phase: RunPhase, detail: Value) {
        let Some(monitor) = self.monitor.as_ref() else {
            return;
        };
        if let Err(error) = monitor.update(&self.run_id, phase, detail) {
            warn!(
                target: TARGET,
                run_id = %self.run_id,
                phase = ?phase,
                %error,
                "run status update failed"
            );
        }
    }

    /// Publishes a terminal transition and marks the guard finished.
    ///
    /// Marking happens even when the mirror rejects the transition, so dropping the
    /// guard afterwards neither retries nor warns twice.
    fn finish(&mut self, outcome: RunOutcome) {
        if let Some(monitor) = self.monitor.as_ref() {
            let phase = outcome.phase;
            if let Err(error) = monitor.finish(&self.run_id, outcome) {
                warn!(
                    target: TARGET,
                    run_id = %self.run_id,
                    phase = ?phase,
                    %error,
                    "run status finish failed"
                );
            }
        }
        self.finished = true;
    }
}

impl Drop for RunStatusGuard {
    /// Finishes an unfinished run as [`RunPhase::Cancelled`] with
    /// `{"reason": "run dropped"}`.
    ///
    /// Uses no runtime and never panics: a closed channel and an untracked run are
    /// both silent best-effort failures.
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        self.finish(RunOutcome::cancelled().with_detail(json!({"reason": DROPPED_DETAIL})));
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]

    use super::*;
    use crate::monitor::status::RunKind;
    use crate::monitor::{RunPhase, RunStatus};

    fn start() -> RunStart {
        RunStart::new(RunKind::Agent, "assistant").with_session("s-1")
    }

    fn monitor() -> Arc<RunMonitor> {
        Arc::new(RunMonitor::new())
    }

    fn snapshot(monitor: &RunMonitor, run_id: &str) -> RunStatus {
        monitor.snapshot(run_id).expect("tracked run")
    }

    #[test]
    fn inert_guard_without_a_monitor_is_a_no_op() {
        let mut guard = RunStatusGuard::begin(None, "run-1", start());
        guard.running(json!({"step": 1}));
        guard.fail(&Error::Validation("boom".to_string()), Value::Null);
        drop(guard);

        let mut fresh = RunStatusGuard::begin(None, "run-2", start());
        fresh.cancelled(Value::Null);
        drop(fresh);
    }

    #[test]
    fn running_then_ok_records_the_success() {
        let monitor = monitor();
        let mut guard = RunStatusGuard::begin(Some(Arc::clone(&monitor)), "run-1", start());
        assert_eq!(snapshot(&monitor, "run-1").phase, RunPhase::Queued);
        assert_eq!(
            snapshot(&monitor, "run-1").session_id.as_deref(),
            Some("s-1")
        );

        guard.running(json!({"step": 1}));
        let running = snapshot(&monitor, "run-1");
        assert_eq!(running.phase, RunPhase::Running);
        assert_eq!(running.detail, json!({"step": 1}));
        assert!(running.finished_at.is_none());

        guard.ok(json!({"answer": 42}));
        let done = snapshot(&monitor, "run-1");
        assert_eq!(done.phase, RunPhase::Completed);
        assert_eq!(done.detail, json!({"answer": 42}));
        assert!(done.finished_at.is_some());
        assert!(done.error.is_none());

        drop(guard);
        assert_eq!(snapshot(&monitor, "run-1").phase, RunPhase::Completed);
    }

    #[test]
    fn cancelled_error_maps_to_the_cancelled_phase() {
        let monitor = monitor();
        let mut guard = RunStatusGuard::begin(Some(Arc::clone(&monitor)), "run-1", start());
        guard.running(Value::Null);
        guard.fail(&Error::Cancelled, json!({"reason": "user"}));

        let done = snapshot(&monitor, "run-1");
        assert_eq!(done.phase, RunPhase::Cancelled);
        assert_eq!(done.error.as_deref(), Some("cancelled"));
        assert_eq!(done.detail, json!({"reason": "user"}));
        assert!(done.finished_at.is_some());
    }

    #[test]
    fn other_errors_fail_the_run_with_the_display_form() {
        let monitor = monitor();
        let mut guard = RunStatusGuard::begin(Some(Arc::clone(&monitor)), "run-1", start());
        guard.running(Value::Null);
        guard.fail(&Error::Validation("boom".to_string()), Value::Null);

        let done = snapshot(&monitor, "run-1");
        assert_eq!(done.phase, RunPhase::Failed);
        assert_eq!(done.error.as_deref(), Some("validation: boom"));
    }

    #[test]
    fn cancelled_finishes_without_an_error() {
        let monitor = monitor();
        let mut guard = RunStatusGuard::begin(Some(Arc::clone(&monitor)), "run-1", start());
        guard.cancelled(json!({"reason": "shutdown"}));

        let done = snapshot(&monitor, "run-1");
        assert_eq!(done.phase, RunPhase::Cancelled);
        assert!(done.error.is_none());
        assert_eq!(done.detail, json!({"reason": "shutdown"}));
    }

    #[test]
    fn dropping_an_unfinished_guard_cancels_the_run() {
        let monitor = monitor();
        let guard = RunStatusGuard::begin(Some(Arc::clone(&monitor)), "run-1", start());
        drop(guard);

        let done = snapshot(&monitor, "run-1");
        assert_eq!(done.phase, RunPhase::Cancelled);
        assert_eq!(done.detail, json!({"reason": "run dropped"}));
        assert!(done.finished_at.is_some());
    }

    #[test]
    fn dropping_after_an_explicit_finish_changes_nothing() {
        let monitor = monitor();
        let mut guard = RunStatusGuard::begin(Some(Arc::clone(&monitor)), "run-1", start());
        guard.running(Value::Null);
        guard.ok(Value::Null);
        let before = snapshot(&monitor, "run-1");
        let history_before = monitor.history("run-1").expect("history").len();

        drop(guard);

        assert_eq!(snapshot(&monitor, "run-1"), before);
        assert_eq!(
            monitor.history("run-1").expect("history").len(),
            history_before
        );
    }

    #[test]
    fn mirror_failures_are_logged_not_propagated() {
        let monitor = monitor();
        let mut guard = RunStatusGuard::begin(Some(Arc::clone(&monitor)), "run-1", start());
        monitor
            .finish("run-1", RunOutcome::completed())
            .expect("finish elsewhere");

        // Both transitions are rejected by the monitor and silently ignored here.
        guard.running(Value::Null);
        guard.cancelled(Value::Null);
        assert_eq!(snapshot(&monitor, "run-1").phase, RunPhase::Completed);

        drop(guard);
        assert_eq!(snapshot(&monitor, "run-1").phase, RunPhase::Completed);
    }

    #[test]
    fn guard_adopts_a_pre_registered_run_without_a_second_event() {
        let monitor = monitor();
        let tracked = monitor.track("run-1", start());
        assert_eq!(tracked.phase, RunPhase::Queued);
        let events = monitor.history("run-1").expect("history").len();
        assert_eq!(events, 1, "track publishes the queued event once");

        let mut guard = RunStatusGuard::begin(Some(Arc::clone(&monitor)), "run-1", start());
        assert_eq!(
            monitor.history("run-1").expect("history").len(),
            events,
            "re-admitting an existing run publishes nothing"
        );
        guard.running(Value::Null);
        assert_eq!(snapshot(&monitor, "run-1").phase, RunPhase::Running);
    }
}
