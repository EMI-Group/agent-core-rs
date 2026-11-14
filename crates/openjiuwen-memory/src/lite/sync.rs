//! Caller-driven sync triggers (`on_session_start`, `on_interval`, `on_write`) and a
//! deterministic interval scheduler.
//!
//! Ports the deferred sync triggers of Python `core/memory/lite/`: the store exposes points
//! at which the derived state (indexes, embeddings, rollups) is reconciled with the
//! Markdown source of truth, and the caller decides when those points occur.
//!
//! # Triggers
//!
//! - `on_session_start` — reconcile once when a session begins, so a store touched by
//!   another process is current before the first recall.
//! - `on_interval` — reconcile after a configured elapsed interval, measured on an injected
//!   clock so tests are deterministic and no wall-clock ordering leaks in.
//! - `on_write` — reconcile immediately after a mutation, for callers that need the derived
//!   state to be current as soon as a write returns.
//!
//! Triggers are additive and each is independently enableable; the default configuration
//! enables none of them, so the store does nothing the caller did not ask for.
//!
//! # Scheduler
//!
//! The interval scheduler is deterministic: it computes the next due instant from the last
//! completed run plus the configured interval, reports whether a run is due for a given
//! `now`, and never depends on a timer firing promptly. A due check is idempotent within the
//! same instant, so two callers cannot schedule the same run twice.
//!
//! # No background daemon
//!
//! Nothing here starts a background loop. Reconciliation runs on the caller's task, and any
//! optional background task is explicit opt-in, lives on the caller's tokio runtime, wakes
//! no more often than the configured interval, and is cancellation-aware: it holds a
//! cancellation token, stops promptly when the token is cancelled or its handle is dropped,
//! and returns without leaving a detached task behind. [`OnSessionStart`](crate::lite::sync::SyncTrigger::OnSessionStart)
//! and [`OnWrite`](crate::lite::sync::SyncTrigger::OnWrite) have no timer at all: they are points the caller
//! reaches on its own, and [`SyncTrigger::interval`](crate::lite::sync::SyncTrigger::interval) is `None` for both.
//!
//! # Errors
//!
//! A failing reconciliation surfaces the underlying typed error (filesystem, index, or
//! embedding) to the caller. A trigger that cannot run (disabled, not due, already
//! running) reports that explicitly instead of silently succeeding: the interval gate
//! answers `false` and a skipped run is a [`SyncReport::skipped`](crate::lite::sync::SyncReport::skipped).
use std::fmt;
use std::time::{Duration, Instant};

use openjiuwen_common::error::{Error, Result};

/// What caused a sync to run.
///
/// The labels are the wire names ([`SyncTrigger::ALL_KINDS`]) and are what a
/// [`SyncReport`] records, so a report can be logged or serialized without carrying the
/// enum's payload (the interval duration).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SyncTrigger {
    /// Reconcile once when a session begins; no timer is involved.
    OnSessionStart,
    /// Reconcile when at least the carried interval elapsed since the last run.
    OnInterval(Duration),
    /// Reconcile immediately after a mutation; no timer is involved.
    OnWrite,
}

impl SyncTrigger {
    /// Every trigger label, in declaration order.
    pub const ALL_KINDS: [&'static str; 3] = ["on_session_start", "on_interval", "on_write"];

    /// The stable `snake_case` label of this trigger.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::OnSessionStart => Self::ALL_KINDS[0],
            Self::OnInterval(_) => Self::ALL_KINDS[1],
            Self::OnWrite => Self::ALL_KINDS[2],
        }
    }

    /// The interval this trigger runs on, or `None` when it has no timer.
    ///
    /// Only [`SyncTrigger::OnInterval`] carries a duration; the session-start and
    /// write triggers are caller-driven points.
    #[must_use]
    pub fn interval(&self) -> Option<Duration> {
        match self {
            Self::OnInterval(interval) => Some(*interval),
            Self::OnSessionStart | Self::OnWrite => None,
        }
    }
}

impl fmt::Display for SyncTrigger {
    /// Prints [`SyncTrigger::label`], so a trigger renders as its wire name.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.label())
    }
}

/// What one sync run did.
///
/// `ran` is `false` only when a trigger's precondition was not met (the interval
/// scheduler was not due); every count is zero in that case.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SyncReport {
    /// Whether a reconciliation actually ran.
    pub ran: bool,
    /// The label of the trigger that produced this report.
    pub trigger: &'static str,
    /// Number of coding facts considered.
    pub facts: usize,
    /// Number of Markdown chunks considered.
    pub chunks: usize,
    /// Number of chunks embedded during the run.
    pub embedded: usize,
    /// Number of session logs considered.
    pub sessions: usize,
}

impl SyncReport {
    /// Builds the report of a trigger that did not run.
    #[must_use]
    pub fn skipped(trigger: &'static str) -> Self {
        Self {
            ran: false,
            trigger,
            facts: 0,
            chunks: 0,
            embedded: 0,
            sessions: 0,
        }
    }

    /// Builds the report of a run that reconciled the given counts.
    #[must_use]
    pub fn new(
        trigger: &'static str,
        facts: usize,
        chunks: usize,
        embedded: usize,
        sessions: usize,
    ) -> Self {
        Self {
            ran: true,
            trigger,
            facts,
            chunks,
            embedded,
            sessions,
        }
    }
}

/// A deterministic interval gate for the [`SyncTrigger::OnInterval`] trigger.
///
/// The caller supplies `now` on every query, so the gate is fully testable without
/// sleeping and never reads a wall clock. It never schedules anything: a caller — or an
/// explicitly started, cancellation-aware task — polls it and performs the sync itself.
#[derive(Debug, Clone)]
pub struct IntervalScheduler {
    /// Minimum elapsed time between two runs.
    interval: Duration,
    /// The instant of the last completed run, if any.
    last: Option<Instant>,
}

impl IntervalScheduler {
    /// Builds a scheduler that runs at most once per `interval`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] naming the value when `interval` is zero.
    pub fn new(interval: Duration) -> Result<Self> {
        if interval.is_zero() {
            return Err(Error::validation(format!(
                "sync interval must be greater than zero; got {interval:?}"
            )));
        }
        Ok(Self {
            interval,
            last: None,
        })
    }

    /// The minimum elapsed time between two runs.
    #[must_use]
    pub fn interval(&self) -> Duration {
        self.interval
    }

    /// Whether a run is due at `now`.
    ///
    /// A scheduler that never ran is always due. Otherwise the run is due once at least
    /// the interval elapsed since the last run, using saturating subtraction, so a `now`
    /// before [`IntervalScheduler::last`] is never due.
    #[must_use]
    pub fn is_due(&self, now: Instant) -> bool {
        self.last
            .is_none_or(|last| now.saturating_duration_since(last) >= self.interval)
    }

    /// Records `now` as the instant of a completed run.
    pub fn mark(&mut self, now: Instant) {
        self.last = Some(now);
    }

    /// Marks the run at `now` when it is due, reporting whether it was.
    ///
    /// This is the single-call form of [`IntervalScheduler::is_due`] plus
    /// [`IntervalScheduler::mark`]: a due poll advances the gate, so polling again at the
    /// same instant is not due.
    pub fn poll(&mut self, now: Instant) -> bool {
        if self.is_due(now) {
            self.mark(now);
            true
        } else {
            false
        }
    }

    /// The instant of the last run, if any.
    #[must_use]
    pub fn last(&self) -> Option<Instant> {
        self.last
    }

    /// Forgets the last run, so the next [`IntervalScheduler::is_due`] is due.
    pub fn reset(&mut self) {
        self.last = None;
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// A fixed base instant; every test derives its instants from it without sleeping.
    fn base() -> Instant {
        Instant::now()
    }

    #[test]
    fn a_zero_interval_is_rejected_and_names_the_value() {
        let error = IntervalScheduler::new(Duration::ZERO).expect_err("zero is rejected");
        let message = error.to_string();
        assert!(message.contains("sync interval"), "{message}");
        assert!(message.contains("0ns"), "{message}");
    }

    #[test]
    fn a_positive_interval_is_accepted_and_reported() {
        let interval = Duration::from_millis(250);
        let scheduler = IntervalScheduler::new(interval).expect("positive interval");
        assert_eq!(scheduler.interval(), interval);
        assert_eq!(scheduler.last(), None);
    }

    #[test]
    fn a_fresh_scheduler_is_due_immediately() {
        let scheduler = IntervalScheduler::new(Duration::from_secs(1)).expect("interval");
        assert!(scheduler.is_due(base()));
    }

    #[test]
    fn a_scheduler_is_due_exactly_at_the_interval_and_after_it() {
        let interval = Duration::from_millis(100);
        let mut scheduler = IntervalScheduler::new(interval).expect("interval");
        let start = base();
        scheduler.mark(start);
        assert_eq!(scheduler.last(), Some(start));
        assert!(!scheduler.is_due(start));
        assert!(!scheduler.is_due(start + Duration::from_millis(99)));
        assert!(scheduler.is_due(start + interval));
        assert!(scheduler.is_due(start + Duration::from_millis(101)));
    }

    #[test]
    fn a_now_before_the_last_run_is_not_due() {
        let mut scheduler = IntervalScheduler::new(Duration::from_millis(1)).expect("interval");
        let start = base();
        let last = start + Duration::from_secs(10);
        scheduler.mark(last);
        assert!(!scheduler.is_due(start));
        assert!(!scheduler.is_due(
            last.checked_sub(Duration::from_secs(1))
                .expect("a representable earlier instant")
        ));
        assert!(scheduler.is_due(last + Duration::from_millis(1)));
    }

    #[test]
    fn poll_runs_exactly_once_per_interval_across_deterministic_instants() {
        let interval = Duration::from_millis(100);
        let mut scheduler = IntervalScheduler::new(interval).expect("interval");
        let start = base();
        let polled: Vec<bool> = [0, 50, 100, 150, 200, 300]
            .into_iter()
            .map(|offset| scheduler.poll(start + Duration::from_millis(offset)))
            .collect();
        assert_eq!(polled, vec![true, false, true, false, true, true]);
        assert_eq!(scheduler.last(), Some(start + Duration::from_millis(300)));
    }

    #[test]
    fn polling_twice_at_the_same_instant_runs_once() {
        let mut scheduler = IntervalScheduler::new(Duration::from_secs(5)).expect("interval");
        let now = base();
        assert!(scheduler.poll(now));
        assert!(!scheduler.poll(now));
        assert_eq!(scheduler.last(), Some(now));
    }

    #[test]
    fn reset_makes_the_next_check_due_again() {
        let mut scheduler = IntervalScheduler::new(Duration::from_secs(60)).expect("interval");
        let now = base();
        scheduler.mark(now);
        assert!(!scheduler.is_due(now));
        scheduler.reset();
        assert_eq!(scheduler.last(), None);
        assert!(scheduler.is_due(now));
    }

    #[test]
    fn trigger_labels_are_the_documented_wire_names() {
        assert_eq!(
            SyncTrigger::ALL_KINDS,
            ["on_session_start", "on_interval", "on_write"]
        );
        let triggers = [
            SyncTrigger::OnSessionStart,
            SyncTrigger::OnInterval(Duration::from_secs(3)),
            SyncTrigger::OnWrite,
        ];
        for (trigger, expected) in triggers.iter().zip(SyncTrigger::ALL_KINDS) {
            assert_eq!(trigger.label(), expected);
            assert_eq!(trigger.to_string(), expected);
        }
    }

    #[test]
    fn only_the_interval_trigger_carries_a_duration() {
        assert_eq!(SyncTrigger::OnSessionStart.interval(), None);
        assert_eq!(SyncTrigger::OnWrite.interval(), None);
        let interval = Duration::from_millis(750);
        assert_eq!(SyncTrigger::OnInterval(interval).interval(), Some(interval));
    }

    #[test]
    fn reports_record_whether_the_run_happened() {
        let skipped = SyncReport::skipped(SyncTrigger::OnSessionStart.label());
        assert!(!skipped.ran);
        assert_eq!(skipped.trigger, "on_session_start");
        assert_eq!(
            skipped.facts + skipped.chunks + skipped.embedded + skipped.sessions,
            0
        );

        let report = SyncReport::new(
            SyncTrigger::OnInterval(Duration::from_secs(1)).label(),
            3,
            4,
            2,
            5,
        );
        assert!(report.ran);
        assert_eq!(report.trigger, "on_interval");
        assert_eq!(report.facts, 3);
        assert_eq!(report.chunks, 4);
        assert_eq!(report.embedded, 2);
        assert_eq!(report.sessions, 5);
    }
}
