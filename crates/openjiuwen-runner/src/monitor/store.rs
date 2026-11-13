//! The shared run table behind [`RunMonitor`](super::RunMonitor): one mutex over
//! the live statuses, the per-run event histories, and the finished-retention
//! queue, plus the broadcast sender transitions publish on.
//!
//! Every transition (admit, update, finish) appends its [`RunEvent`] to the run's
//! bounded history **and** publishes it on the channel inside the same
//! `parking_lot::Mutex` step, then drops the guard before returning; there is no
//! await point anywhere in the module.
//!
//! Retention: a live run is never evicted. A finished run is retained until
//! [`LIVE_RUN_LIMIT`] newer completions exist, so the table cannot grow without
//! bound; evicting a run drops both its status and its history.

use std::collections::{HashMap, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH};

use openjiuwen_common::error::{Error, Result};
use parking_lot::Mutex;
use serde_json::Value;
use tokio::sync::broadcast;

use super::event::RunEvent;
use super::status::{RunOutcome, RunPhase, RunStart, RunStatus};
use super::{EVENT_HISTORY_LIMIT, LIVE_RUN_LIMIT};

/// Epoch milliseconds of the current clock reading.
///
/// A pre-epoch clock yields `0` and an out-of-range reading saturates at
/// [`i64::MAX`]; neither panics.
pub(super) fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| {
            i64::try_from(elapsed.as_millis()).unwrap_or(i64::MAX)
        })
}

/// `run '<id>' is not tracked`: the id was never admitted or has been evicted.
fn not_tracked(run_id: &str) -> Error {
    Error::NotFound(format!("run '{run_id}' is not tracked"))
}

/// `run '<id>' has already finished`: it accepts no further transitions.
fn already_finished(run_id: &str) -> Error {
    Error::Validation(format!("run '{run_id}' has already finished"))
}

/// The mutable run table, guarded as one unit so a status and its history never
/// diverge.
#[derive(Default)]
struct State {
    /// The live statuses plus the retained finished ones.
    runs: HashMap<String, RunStatus>,
    /// Per-run transition history, oldest first, bounded to [`EVENT_HISTORY_LIMIT`].
    history: HashMap<String, VecDeque<RunEvent>>,
    /// Finished run ids in completion order, oldest first (the retention queue).
    finished: VecDeque<String>,
}

/// The shared state plus the event channel; `Arc<Inner>` is what
/// [`RunMonitor`](super::RunMonitor) clones.
pub(super) struct Inner {
    state: Mutex<State>,
    sender: broadcast::Sender<RunEvent>,
}

impl Inner {
    /// Builds an empty table and a channel buffering `event_capacity` events.
    ///
    /// A capacity below `1` is raised to `1`: `broadcast::channel(0)` panics, and
    /// the monitor never panics on a caller-supplied capacity.
    pub(super) fn new(event_capacity: usize) -> Self {
        let (sender, _receiver) = broadcast::channel(event_capacity.max(1));
        Self {
            state: Mutex::new(State::default()),
            sender,
        }
    }

    /// Admits a new run under `run_id`, or returns the existing status unchanged.
    pub(super) fn track(&self, run_id: String, start: RunStart) -> RunStatus {
        let mut state = self.state.lock();
        if let Some(existing) = state.runs.get(&run_id) {
            return existing.clone();
        }
        let status = RunStatus {
            run_id,
            kind: start.kind,
            label: start.label,
            phase: RunPhase::Queued,
            session_id: start.session_id,
            started_at: now_ms(),
            finished_at: None,
            error: None,
            detail: start.detail,
        };
        state.runs.insert(status.run_id.clone(), status.clone());
        publish(&mut state, &self.sender, &status);
        status
    }

    /// Advances a live run to `phase` and publishes the transition.
    pub(super) fn update(&self, run_id: &str, phase: RunPhase, detail: Value) -> Result<RunStatus> {
        let mut state = self.state.lock();
        let status = {
            let status = state
                .runs
                .get_mut(run_id)
                .ok_or_else(|| not_tracked(run_id))?;
            if status.phase.is_terminal() {
                return Err(already_finished(run_id));
            }
            status.phase = phase;
            status.detail = detail;
            status.clone()
        };
        publish(&mut state, &self.sender, &status);
        Ok(status)
    }

    /// Finishes a live run with `outcome`, retains it, and publishes the transition.
    pub(super) fn finish(&self, run_id: &str, outcome: RunOutcome) -> Result<RunStatus> {
        let mut state = self.state.lock();
        let status = {
            let status = state
                .runs
                .get_mut(run_id)
                .ok_or_else(|| not_tracked(run_id))?;
            if status.phase.is_terminal() {
                return Err(already_finished(run_id));
            }
            status.phase = outcome.phase;
            status.finished_at = Some(now_ms());
            status.error = outcome.error;
            status.detail = outcome.detail;
            status.clone()
        };
        publish(&mut state, &self.sender, &status);
        // The queue is bounded by completions, never by admissions: a live run is
        // excluded from it and therefore never evicted.
        state.finished.push_back(status.run_id.clone());
        while state.finished.len() > LIVE_RUN_LIMIT {
            if let Some(evicted) = state.finished.pop_front() {
                state.runs.remove(&evicted);
                state.history.remove(&evicted);
            }
        }
        Ok(status)
    }

    /// Returns the live or retained-finished status of `run_id`.
    pub(super) fn snapshot(&self, run_id: &str) -> Result<RunStatus> {
        let state = self.state.lock();
        state
            .runs
            .get(run_id)
            .cloned()
            .ok_or_else(|| not_tracked(run_id))
    }

    /// Returns the non-terminal runs, sorted by `(started_at, run_id)`.
    pub(super) fn list_active(&self) -> Vec<RunStatus> {
        let state = self.state.lock();
        let mut active: Vec<RunStatus> = state
            .runs
            .values()
            .filter(|status| !status.phase.is_terminal())
            .cloned()
            .collect();
        active.sort_by(|left, right| {
            left.started_at
                .cmp(&right.started_at)
                .then_with(|| left.run_id.cmp(&right.run_id))
        });
        active
    }

    /// Returns the retained transition history of `run_id`, oldest first.
    pub(super) fn history(&self, run_id: &str) -> Result<Vec<RunEvent>> {
        let state = self.state.lock();
        state
            .history
            .get(run_id)
            .map(|events| events.iter().cloned().collect())
            .ok_or_else(|| not_tracked(run_id))
    }

    /// Drops every run, history, and retention entry; publishes nothing.
    pub(super) fn clear(&self) {
        let mut state = self.state.lock();
        state.runs.clear();
        state.history.clear();
        state.finished.clear();
    }

    /// Returns a fresh receiver on the event channel.
    pub(super) fn subscribe(&self) -> broadcast::Receiver<RunEvent> {
        self.sender.subscribe()
    }
}

/// Appends `status`'s event to its bounded history and publishes it.
///
/// Called with the state lock held, so the history append and the publish are one
/// step; a send failure means "no subscribers", which is not an error.
fn publish(state: &mut State, sender: &broadcast::Sender<RunEvent>, status: &RunStatus) {
    let event = RunEvent {
        run_id: status.run_id.clone(),
        kind: status.kind,
        label: status.label.clone(),
        phase: status.phase,
        session_id: status.session_id.clone(),
        at_ms: now_ms(),
        detail: status.detail.clone(),
    };
    let history = state.history.entry(status.run_id.clone()).or_default();
    history.push_back(event.clone());
    while history.len() > EVENT_HISTORY_LIMIT {
        history.pop_front();
    }
    let _ = sender.send(event);
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]

    use super::*;
    use crate::monitor::status::RunKind;
    use crate::monitor::RunMonitor;

    #[test]
    fn history_keeps_the_newest_events() {
        let monitor = RunMonitor::new();
        let status = monitor.begin(RunStart::new(RunKind::Workflow, "wf"));
        for _ in 0..(EVENT_HISTORY_LIMIT + 10) {
            monitor
                .update(&status.run_id, RunPhase::Running, Value::Null)
                .expect("update");
        }
        let history = monitor.history(&status.run_id).expect("history");
        assert_eq!(history.len(), EVENT_HISTORY_LIMIT);
        assert_eq!(
            history.last().map(|event| event.phase),
            Some(RunPhase::Running)
        );
        assert!(
            history.iter().all(|event| event.phase == RunPhase::Running),
            "the initial queued event is the first one dropped"
        );
    }

    #[test]
    fn finished_runs_are_retained_up_to_the_live_run_limit() {
        let monitor = RunMonitor::new();
        let live = monitor.begin(RunStart::new(RunKind::Agent, "live"));
        let mut finished = Vec::new();
        for index in 0..(LIVE_RUN_LIMIT + 2) {
            let status = monitor.begin(RunStart::new(RunKind::Agent, format!("run-{index}")));
            monitor
                .finish(&status.run_id, RunOutcome::completed())
                .expect("finish");
            finished.push(status.run_id);
        }

        assert!(monitor.snapshot(&finished[0]).is_err(), "oldest evicted");
        assert!(
            monitor.history(&finished[0]).is_err(),
            "history evicted too"
        );
        assert!(monitor.snapshot(&finished[1]).is_err(), "second oldest too");
        let newest = monitor
            .snapshot(&finished[LIVE_RUN_LIMIT + 1])
            .expect("newest retained");
        assert_eq!(newest.phase, RunPhase::Completed);
        assert!(newest.finished_at.is_some());

        assert!(monitor.snapshot(&live.run_id).is_ok(), "live runs survive");
        assert_eq!(monitor.list_active().len(), 1);
        assert_eq!(
            monitor
                .history(&finished[LIVE_RUN_LIMIT + 1])
                .expect("retained history")
                .len(),
            2,
            "a retained run keeps its begin and finish events"
        );
    }

    #[test]
    fn clear_drops_every_run_and_history() {
        let monitor = RunMonitor::new();
        let live = monitor.begin(RunStart::new(RunKind::Team, "crew"));
        let done = monitor.begin(RunStart::new(RunKind::Agent, "solo"));
        monitor
            .finish(&done.run_id, RunOutcome::completed())
            .expect("finish");

        monitor.clear();

        assert!(monitor.snapshot(&live.run_id).is_err());
        assert!(monitor.snapshot(&done.run_id).is_err());
        assert!(monitor.history(&live.run_id).is_err());
        assert!(monitor.list_active().is_empty());
    }

    #[test]
    fn zero_capacity_is_raised_to_one() {
        let monitor = RunMonitor::with_capacity(0);
        let mut receiver = monitor.subscribe();
        let status = monitor.begin(RunStart::new(RunKind::Agent, "a"));
        assert_eq!(receiver.try_recv().expect("event").run_id, status.run_id);
    }

    #[test]
    fn clock_is_non_negative() {
        assert!(now_ms() >= 0);
        let start = now_ms();
        let end = now_ms();
        assert!(start <= end);
    }
}
