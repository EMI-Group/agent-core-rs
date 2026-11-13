//! Run status and event monitoring for the runs the runner executes.
//!
//! [`RunMonitor`] keeps a [`RunStatus`] snapshot per run and publishes one
//! [`RunEvent`] per state transition on a [`tokio::sync::broadcast`] channel, so a
//! host can render live progress and a final outcome without polling the session:
//!
//! - [`RunMonitor::begin`] admits a run under a fresh UUID v4 id in
//!   [`RunPhase::Queued`] and publishes the initial event;
//!   [`RunMonitor::track`] does the same for a caller-supplied id and adopts an
//!   already-tracked run unchanged (publishing nothing).
//! - [`RunMonitor::update`] advances a live run ([`RunPhase::Running`],
//!   [`RunPhase::Paused`], ...) and publishes the phase it was given.
//! - [`RunMonitor::finish`] records a terminal [`RunOutcome`]; a terminal run
//!   rejects every later transition with `Error::Validation`.
//!
//! **Bounds.** A run's transition history keeps the newest
//! [`EVENT_HISTORY_LIMIT`] events (oldest dropped). The table retains every live
//! run plus the [`LIVE_RUN_LIMIT`] most recently finished ones; finishing one more
//! evicts the oldest finished run together with its history, so memory is bounded
//! by `LIVE_RUN_LIMIT + live runs` statuses. Live runs are never evicted.
//!
//! **One event per transition.** `begin`/`track` publish the [`RunPhase::Queued`]
//! event, `update` publishes its phase, `finish` publishes the outcome phase, and
//! the history append and the channel publish happen in the same lock step — a
//! subscriber sees exactly the transitions that changed the status.
//!
//! **Synchronous.** The module is `await`-free: `parking_lot::Mutex` guards one
//! short critical section per transition and is always dropped before the call
//! returns, so a monitor call can be made from a run path, a drop guard, or a
//! plain test without a runtime. Only [`RunMonitor::subscribe`] hands out a tokio
//! receiver, and receiving is the subscriber's business.
//!
//! The monitor is in-memory only: nothing is persisted, events are not shipped
//! over the message queue or a socket, and no aggregation/metrics are computed.
//!
//! ```no_run
//! use openjiuwen_runner::monitor::{RunKind, RunMonitor, RunOutcome, RunStart};
//!
//! let monitor = RunMonitor::new();
//! let mut events = monitor.subscribe();
//! let status = monitor.begin(RunStart::new(RunKind::Agent, "assistant"));
//! let _ = events.try_recv();
//! monitor.finish(&status.run_id, RunOutcome::completed()).unwrap();
//! assert!(monitor.list_active().is_empty());
//! ```

mod event;
// The run paths and the team pool drive this guard (`crate::monitor::RunStatusGuard`);
// the `dead_code` allow is kept as it is.
#[allow(dead_code)]
mod guard;
mod status;
mod store;

use std::sync::Arc;

use openjiuwen_common::error::Result;
use serde_json::Value;
use tokio::sync::broadcast;

pub use event::RunEvent;
pub use status::{RunKind, RunOutcome, RunPhase, RunStart, RunStatus};

// The run paths reach the guard through this re-export; it is a crate-internal
// seam, so it is deliberately not part of the public API.
#[allow(unused_imports)]
pub(crate) use guard::RunStatusGuard;

use store::Inner;

/// How many transition events a single run's history retains (newest kept).
pub const EVENT_HISTORY_LIMIT: usize = 128;

/// How many finished runs the table retains, most recent completions first.
///
/// A live (non-terminal) run is never subject to this bound.
pub const LIVE_RUN_LIMIT: usize = 64;

/// The default number of [`RunEvent`]s buffered for each subscriber.
pub const EVENT_CHANNEL_CAPACITY: usize = 256;

/// A cheap-to-clone handle to the shared run table and its event channel.
///
/// Clones share one table, so a monitor handed to a run path and a monitor held by
/// a host observe the same statuses and events. See the [module docs](self) for the
/// retention bounds and the one-event-per-transition contract.
#[derive(Clone)]
pub struct RunMonitor {
    /// The shared table plus the broadcast sender.
    inner: Arc<Inner>,
}

impl RunMonitor {
    /// Creates a monitor with the default limits ([`EVENT_CHANNEL_CAPACITY`] events
    /// buffered per subscriber, [`LIVE_RUN_LIMIT`] finished runs retained,
    /// [`EVENT_HISTORY_LIMIT`] events of history per run).
    #[must_use]
    pub fn new() -> Self {
        Self::with_capacity(EVENT_CHANNEL_CAPACITY)
    }

    /// Creates a monitor whose channel buffers `event_capacity` events for each
    /// subscriber.
    ///
    /// A capacity below `1` is raised to `1`; a slow subscriber then observes
    /// `Err(RecvError::Lagged(n))` as soon as it falls further behind.
    #[must_use]
    pub fn with_capacity(event_capacity: usize) -> Self {
        Self {
            inner: Arc::new(Inner::new(event_capacity)),
        }
    }

    /// Admits a new run under a fresh UUID v4 id, in [`RunPhase::Queued`], and
    /// publishes its initial event.
    #[must_use]
    pub fn begin(&self, start: RunStart) -> RunStatus {
        self.inner
            .track(openjiuwen_common::utils::id::uuid_v4(), start)
    }

    /// Admits `run_id` in [`RunPhase::Queued`] and publishes its initial event, or
    /// returns the existing status unchanged (publishing nothing) when the id is
    /// already tracked.
    ///
    /// Use this when the run path already minted an id (for example the id it uses
    /// for tracing); use [`begin`](Self::begin) to let the monitor mint one.
    #[must_use]
    pub fn track(&self, run_id: impl Into<String>, start: RunStart) -> RunStatus {
        self.inner.track(run_id.into(), start)
    }

    /// Advances a live run to `phase`, records `detail`, and publishes one event.
    ///
    /// # Errors
    ///
    /// [`Error::NotFound`](openjiuwen_common::error::Error::NotFound) when the id is
    /// unknown (or was evicted), and
    /// [`Error::Validation`](openjiuwen_common::error::Error::Validation) when the
    /// run already reached a terminal phase, or when `phase` is itself terminal
    /// (a terminal phase must go through [`finish`](Self::finish) so
    /// `finished_at` is stamped and the run is retained).
    pub fn update(&self, run_id: &str, phase: RunPhase, detail: Value) -> Result<RunStatus> {
        self.inner.update(run_id, phase, detail)
    }

    /// Finishes a live run with `outcome`, records `finished_at`/`error`/`detail`,
    /// publishes the terminal event, and retains the run for the
    /// [`LIVE_RUN_LIMIT`] most recent completions.
    ///
    /// The outcome's phase is recorded as given; [`RunOutcome`]'s constructors
    /// produce the three terminal phases.
    ///
    /// # Errors
    ///
    /// [`Error::NotFound`](openjiuwen_common::error::Error::NotFound) when the id is
    /// unknown (or was evicted), and
    /// [`Error::Validation`](openjiuwen_common::error::Error::Validation) when the
    /// run has already finished.
    pub fn finish(&self, run_id: &str, outcome: RunOutcome) -> Result<RunStatus> {
        self.inner.finish(run_id, outcome)
    }

    /// Returns the current status of a live run or of a retained finished one.
    ///
    /// # Errors
    ///
    /// [`Error::NotFound`](openjiuwen_common::error::Error::NotFound) when the id is
    /// unknown or its finished status has been evicted.
    pub fn snapshot(&self, run_id: &str) -> Result<RunStatus> {
        self.inner.snapshot(run_id)
    }

    /// Returns the non-terminal runs, sorted by `(started_at, run_id)` so the order
    /// is deterministic even when two runs started in the same millisecond.
    #[must_use]
    pub fn list_active(&self) -> Vec<RunStatus> {
        self.inner.list_active()
    }

    /// Returns the retained transition history of `run_id`, oldest event first.
    ///
    /// # Errors
    ///
    /// [`Error::NotFound`](openjiuwen_common::error::Error::NotFound) when the id is
    /// unknown or was evicted; a live run always has at least its `begin` event.
    pub fn history(&self, run_id: &str) -> Result<Vec<RunEvent>> {
        self.inner.history(run_id)
    }

    /// Subscribes to the transition events published from now on.
    ///
    /// A slow subscriber that does not keep up with the channel capacity receives
    /// `Err(RecvError::Lagged(n))` — `n` frames were dropped **for that subscriber
    /// only**, the monitor and the other subscribers are unaffected — and can keep
    /// receiving the frames after the dropped ones. `RecvError::Closed` is only
    /// observed once every sender is gone, i.e. when the last [`RunMonitor`] (and
    /// every clone) has been dropped.
    #[must_use]
    pub fn subscribe(&self) -> broadcast::Receiver<RunEvent> {
        self.inner.subscribe()
    }

    /// Drops every run, every history, and the finished-retention queue.
    ///
    /// Nothing is published: existing subscribers simply stop seeing events for the
    /// cleared runs and keep working for runs begun afterwards.
    pub fn clear(&self) {
        self.inner.clear();
    }
}

impl Default for RunMonitor {
    /// Same as [`RunMonitor::new`].
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]

    use super::*;
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;
    use tokio::sync::broadcast::error::TryRecvError;

    fn start(kind: RunKind, label: &str) -> RunStart {
        RunStart::new(kind, label)
    }

    #[test]
    fn begin_update_finish_records_the_whole_sequence() {
        let monitor = RunMonitor::new();
        let created = monitor.begin(start(RunKind::Workflow, "wf"));
        assert_eq!(created.phase, RunPhase::Queued);
        assert_eq!(created.kind, RunKind::Workflow);
        assert_eq!(created.label, "wf");
        assert!(created.started_at > 0);
        assert!(created.finished_at.is_none());
        assert!(created.error.is_none());

        let running = monitor
            .update(&created.run_id, RunPhase::Running, json!({"step": 1}))
            .expect("running");
        assert_eq!(running.phase, RunPhase::Running);
        assert_eq!(running.detail, json!({"step": 1}));
        assert_eq!(running.started_at, created.started_at);
        assert!(running.finished_at.is_none());

        let paused = monitor
            .update(
                &created.run_id,
                RunPhase::Paused,
                json!({"awaiting": "human"}),
            )
            .expect("paused");
        assert_eq!(paused.phase, RunPhase::Paused);

        let done = monitor
            .finish(
                &created.run_id,
                RunOutcome::completed().with_detail(json!({"steps": 2})),
            )
            .expect("finish");
        assert_eq!(done.phase, RunPhase::Completed);
        assert_eq!(done.detail, json!({"steps": 2}));
        assert!(done.finished_at.is_some());
        assert!(done.finished_at >= Some(done.started_at));
        assert!(done.error.is_none());
        assert_eq!(monitor.snapshot(&created.run_id).expect("snapshot"), done);

        let phases: Vec<RunPhase> = monitor
            .history(&created.run_id)
            .expect("history")
            .into_iter()
            .map(|event| event.phase)
            .collect();
        assert_eq!(
            phases,
            vec![
                RunPhase::Queued,
                RunPhase::Running,
                RunPhase::Paused,
                RunPhase::Completed
            ]
        );
    }

    #[test]
    fn every_transition_publishes_exactly_one_event() {
        let monitor = RunMonitor::new();
        let mut receiver = monitor.subscribe();

        let created = monitor.begin(start(RunKind::Team, "crew"));
        monitor
            .update(&created.run_id, RunPhase::Running, Value::Null)
            .expect("running");
        monitor
            .finish(&created.run_id, RunOutcome::cancelled())
            .expect("finish");

        let mut phases = Vec::new();
        while let Ok(event) = receiver.try_recv() {
            assert_eq!(event.run_id, created.run_id);
            assert_eq!(event.kind, RunKind::Team);
            assert_eq!(event.label, "crew");
            assert!(event.at_ms > 0);
            phases.push(event.phase);
        }
        assert_eq!(
            phases,
            vec![RunPhase::Queued, RunPhase::Running, RunPhase::Cancelled]
        );
    }

    #[test]
    fn track_on_an_existing_id_adopts_it_and_publishes_nothing() {
        let monitor = RunMonitor::new();
        let first = monitor.track("run-1", start(RunKind::Agent, "assistant"));
        assert_eq!(first.run_id, "run-1");
        monitor
            .update("run-1", RunPhase::Running, json!({"step": 1}))
            .expect("running");
        let history_len = monitor.history("run-1").expect("history").len();

        let adopted = monitor.track("run-1", start(RunKind::Agent, "other"));
        assert_eq!(adopted.phase, RunPhase::Running, "adopted unchanged");
        assert_eq!(adopted.label, "assistant", "the original start is kept");
        assert_eq!(adopted.detail, json!({"step": 1}));
        assert_eq!(
            monitor.history("run-1").expect("history").len(),
            history_len,
            "no event for an adopted run"
        );

        let new = monitor.track("run-2", start(RunKind::Agent, "solo"));
        assert_eq!(new.run_id, "run-2");
        assert_eq!(new.phase, RunPhase::Queued);
        assert_eq!(monitor.history("run-2").expect("history").len(), 1);
    }

    #[test]
    fn list_active_returns_only_live_runs_in_a_deterministic_order() {
        let monitor = RunMonitor::new();
        let statuses: Vec<RunStatus> = ["a", "b", "c"]
            .into_iter()
            .map(|label| monitor.begin(start(RunKind::Agent, label)))
            .collect();
        let finished = statuses[1].run_id.clone();
        monitor
            .finish(&finished, RunOutcome::failed("boom"))
            .expect("finish");

        let mut expected: Vec<RunStatus> = statuses
            .into_iter()
            .filter(|status| status.run_id != finished)
            .collect();
        expected.sort_by(|left, right| {
            left.started_at
                .cmp(&right.started_at)
                .then_with(|| left.run_id.cmp(&right.run_id))
        });

        let active = monitor.list_active();
        assert_eq!(active.len(), 2);
        assert_eq!(active, expected);
        assert!(
            active
                .windows(2)
                .all(|pair| (pair[0].started_at, &pair[0].run_id)
                    <= (pair[1].started_at, &pair[1].run_id)),
            "sorted by (started_at, run_id)"
        );
        assert!(
            active.iter().all(|status| status.run_id != finished),
            "the finished run is excluded"
        );
    }

    #[test]
    fn unknown_ids_are_not_found() {
        let monitor = RunMonitor::new();
        for error in [
            monitor.snapshot("missing").expect_err("snapshot"),
            monitor.history("missing").expect_err("history"),
            monitor
                .update("missing", RunPhase::Running, Value::Null)
                .expect_err("update"),
            monitor
                .finish("missing", RunOutcome::completed())
                .expect_err("finish"),
        ] {
            assert_eq!(error.kind(), ErrorKind::NotFound);
            assert!(
                error.to_string().contains("run 'missing' is not tracked"),
                "unexpected message: {error}"
            );
        }
    }

    #[test]
    fn double_finish_and_a_terminal_update_are_validation_errors() {
        let monitor = RunMonitor::new();
        let created = monitor.begin(start(RunKind::Agent, "assistant"));
        monitor
            .finish(&created.run_id, RunOutcome::completed())
            .expect("first finish");

        let second = monitor
            .finish(&created.run_id, RunOutcome::completed())
            .expect_err("second finish");
        assert_eq!(second.kind(), ErrorKind::Validation);
        assert!(
            second
                .to_string()
                .contains(&format!("run '{}' has already finished", created.run_id)),
            "unexpected message: {second}"
        );

        let late = monitor
            .update(&created.run_id, RunPhase::Running, Value::Null)
            .expect_err("terminal update");
        assert_eq!(late.kind(), ErrorKind::Validation);
        assert!(late.to_string().contains("has already finished"));
        assert_eq!(
            monitor.snapshot(&created.run_id).expect("snapshot").phase,
            RunPhase::Completed,
            "the rejected transitions changed nothing"
        );
    }

    #[test]
    fn a_terminal_phase_must_go_through_finish() {
        let monitor = RunMonitor::new();
        let created = monitor.begin(start(RunKind::Workflow, "wf"));
        monitor
            .update(&created.run_id, RunPhase::Running, Value::Null)
            .expect("running");
        for phase in [RunPhase::Completed, RunPhase::Failed, RunPhase::Cancelled] {
            let error = monitor
                .update(&created.run_id, phase, Value::Null)
                .expect_err("terminal update");
            assert_eq!(error.kind(), ErrorKind::Validation);
            assert!(
                error.to_string().contains("must be finished with `finish`"),
                "unexpected message: {error}"
            );
        }
        let status = monitor.snapshot(&created.run_id).expect("still live");
        assert_eq!(status.phase, RunPhase::Running);
        assert!(status.finished_at.is_none());
        assert_eq!(
            monitor.list_active().len(),
            1,
            "a rejected terminal update keeps the run live"
        );
        assert_eq!(monitor.history(&created.run_id).expect("history").len(), 2);
    }
    #[test]
    fn subscribers_receive_events_in_order() {
        let monitor = RunMonitor::new();
        let mut first = monitor.subscribe();
        let mut second = monitor.subscribe();
        let created = monitor.begin(start(RunKind::Agent, "assistant"));
        monitor
            .update(&created.run_id, RunPhase::Running, Value::Null)
            .expect("running");
        monitor
            .update(&created.run_id, RunPhase::Paused, Value::Null)
            .expect("paused");

        for receiver in [&mut first, &mut second] {
            let phases: Vec<RunPhase> = std::iter::from_fn(|| receiver.try_recv().ok())
                .map(|event| event.phase)
                .collect();
            assert_eq!(
                phases,
                vec![RunPhase::Queued, RunPhase::Running, RunPhase::Paused]
            );
        }
    }

    #[test]
    fn slow_subscriber_lags_then_keeps_receiving() {
        let monitor = RunMonitor::with_capacity(1);
        let mut receiver = monitor.subscribe();
        let created = monitor.begin(start(RunKind::Agent, "assistant"));
        monitor
            .update(&created.run_id, RunPhase::Running, Value::Null)
            .expect("running");
        monitor
            .update(&created.run_id, RunPhase::Paused, Value::Null)
            .expect("paused");

        match receiver.try_recv() {
            Err(TryRecvError::Lagged(dropped)) => assert!(dropped >= 1),
            other => panic!("expected Lagged, got {other:?}"),
        }
        // The cursor jumped to the oldest stored frame (the capacity is one frame),
        // so the receiver resumes from there instead of losing the stream.
        let buffered = receiver.try_recv().expect("oldest buffered frame");
        assert_eq!(buffered.phase, RunPhase::Paused);

        monitor
            .update(&created.run_id, RunPhase::Running, Value::Null)
            .expect("running again");
        let event = receiver.try_recv().expect("event after the lag");
        assert_eq!(event.phase, RunPhase::Running);
        assert_eq!(event.run_id, created.run_id);
    }

    #[test]
    fn clear_empties_the_monitor_and_keeps_subscribers_working() {
        let monitor = RunMonitor::default();
        let mut receiver = monitor.subscribe();
        let created = monitor.begin(start(RunKind::Workflow, "wf"));
        let _ = receiver.try_recv();

        monitor.clear();
        assert!(monitor.snapshot(&created.run_id).is_err());
        assert!(monitor.history(&created.run_id).is_err());
        assert!(monitor.list_active().is_empty());
        assert_eq!(
            receiver.try_recv(),
            Err(TryRecvError::Empty),
            "nothing published"
        );

        let next = monitor.begin(start(RunKind::Workflow, "wf-2"));
        assert_eq!(monitor.list_active().len(), 1);
        assert_eq!(receiver.try_recv().expect("event").run_id, next.run_id);
    }

    #[test]
    fn clones_share_one_table() {
        let monitor = RunMonitor::new();
        let clone = monitor.clone();
        let created = monitor.begin(start(RunKind::Agent, "assistant"));
        assert_eq!(clone.snapshot(&created.run_id).expect("shared"), created);
    }
}
