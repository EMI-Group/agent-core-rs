//! The settle policy: when a run counts as finished, and how often the board is read.
//!
//! Python parity source: the idle debounce of the `CoordinationKernel` in
//! `openjiuwen/agent_teams/runtime/coordination.py`.
//!
//! A run ends when the leader is idle **and** the task board is settled on two consecutive polls,
//! so a between-turn idle never finishes a team that is about to start its next turn. That rule is
//! [`SettleTracker::poll`]; the state it needs (the two-tick counter, the adaptive interval and the
//! last probe) lives here instead of in the run loop, so the loop stays one linear `select!`.
//!
//! # Adaptive polling
//!
//! A fixed 25 ms poll rebuilt the whole board forty times a second no matter what the team was
//! doing. The tracker keeps that rate while the board is moving and doubles the interval up to
//! [`SETTLE_MAX_POLL_INTERVAL`] while two consecutive probes are identical:
//!
//! * the first poll happens immediately ([`SettleTracker::initial_delay`]), because a run that has
//!   nothing to wait for should settle without a delay;
//! * a changed probe — a task created, claimed, completed, or re-stamped — resets the interval to
//!   [`SETTLE_POLL_INTERVAL`], and so does any chunk or bus event ([`SettleTracker::activity`]),
//!   because both mean the team is working;
//! * a poll that observes "leader idle && board settled" always schedules the *fast* interval, so
//!   the confirming poll arrives within one [`SETTLE_POLL_INTERVAL`] even after a long backoff.
//!
//! The settle *decision* is time independent: it depends only on the sequence of
//! `(idle, probe)` observations, so the same board settles on the same observation, not on the
//! interval in force. The interval only decides how much wall-clock time passes between those
//! observations.
use std::time::Duration;

use crate::taskboard::board::SettleProbe;

/// How often an idle run re-reads the task board while the board is moving.
pub const SETTLE_POLL_INTERVAL: Duration = Duration::from_millis(25);

/// Longest interval the settle poll backs off to while the board does not change.
///
/// A settled observation always schedules [`SETTLE_POLL_INTERVAL`] instead, so this only bounds how
/// late a *change* is noticed by a quiet run.
pub const SETTLE_MAX_POLL_INTERVAL: Duration = Duration::from_millis(250);

/// How many consecutive settle polls must observe "leader idle && board settled".
///
/// Two polls (one interval apart) keep a between-turn idle from ending the run before the leader
/// starts its next turn.
const REQUIRED_SETTLE_TICKS: u32 = 2;

/// The settle decision of one run plus its adaptive poll schedule.
///
/// Construct one per run ([`SettleTracker::new`]), call [`SettleTracker::activity`] whenever the run
/// produced output or consumed a bus event, and call [`SettleTracker::poll`] once per poll with the
/// leader's state and the board's [`SettleProbe`].
/// [`SettleTracker::interval`] is the delay before the next poll.
#[derive(Debug, Clone)]
pub(crate) struct SettleTracker {
    /// Interval used while the board is moving.
    fast: Duration,
    /// Interval the backoff stops at.
    max: Duration,
    /// Interval scheduled for the next poll.
    current: Duration,
    /// Probe observed by the previous poll, when there was one.
    last: Option<SettleProbe>,
    /// Consecutive polls that observed "leader idle && board settled".
    idle_ticks: u32,
}

impl SettleTracker {
    /// Builds a tracker with the default intervals ([`SETTLE_POLL_INTERVAL`], [`SETTLE_MAX_POLL_INTERVAL`]).
    pub(crate) const fn new() -> Self {
        Self::with_intervals(SETTLE_POLL_INTERVAL, SETTLE_MAX_POLL_INTERVAL)
    }

    /// Builds a tracker with explicit intervals; `max` below `fast` is clamped up to `fast`.
    pub(crate) const fn with_intervals(fast: Duration, max: Duration) -> Self {
        let max = if max.as_millis() < fast.as_millis() {
            fast
        } else {
            max
        };
        Self {
            fast,
            max,
            current: fast,
            last: None,
            idle_ticks: 0,
        }
    }

    /// The delay before the first poll: none, so an already-settled board settles without waiting.
    pub(crate) const fn initial_delay(&self) -> Duration {
        Duration::ZERO
    }

    /// The delay scheduled before the next poll.
    pub(crate) fn interval(&self) -> Duration {
        self.current
    }

    /// Records that the run did something: output, a forwarded chunk, a bus event, a delivery.
    ///
    /// This breaks the two-tick streak (a run that just spoke is not idle) and schedules the next
    /// poll at [`SettleTracker::fast`], so a busy run keeps observing the board closely.
    pub(crate) fn activity(&mut self) {
        self.idle_ticks = 0;
        self.current = self.fast;
    }

    /// Records one poll and reports whether the run has settled.
    ///
    /// `leader_idle` is the leader's harness state; `probe` is the board read of this poll.
    /// Returns `true` exactly when `leader_idle && probe.settled` held on
    /// [`REQUIRED_SETTLE_TICKS`] consecutive polls.
    pub(crate) fn poll(&mut self, leader_idle: bool, probe: SettleProbe) -> bool {
        let moved = self.last != Some(probe);
        self.last = Some(probe);
        if !(leader_idle && probe.settled) {
            self.idle_ticks = 0;
            // Back off only while the board sits still; a changed board is work in progress.
            self.current = if moved { self.fast } else { self.slowed() };
            return false;
        }
        self.idle_ticks = self.idle_ticks.saturating_add(1);
        // A settle is imminent: the confirming poll must not wait out a backoff.
        self.current = self.fast;
        self.idle_ticks >= REQUIRED_SETTLE_TICKS
    }

    /// The next backoff step: double the current interval, capped at `max`.
    fn slowed(&self) -> Duration {
        self.current.saturating_mul(2).min(self.max)
    }
}

impl Default for SettleTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::pedantic)]
mod tests {
    use super::*;

    /// Short intervals, so the schedule assertions are readable.
    const FAST: Duration = Duration::from_millis(10);
    const MAX: Duration = Duration::from_millis(80);

    fn tracker() -> SettleTracker {
        SettleTracker::with_intervals(FAST, MAX)
    }

    fn probe(settled: bool, fingerprint: u64) -> SettleProbe {
        SettleProbe {
            settled,
            tasks: 1,
            fingerprint,
        }
    }

    #[test]
    fn defaults_match_the_public_constants() {
        let tracker = SettleTracker::new();
        assert_eq!(tracker.interval(), SETTLE_POLL_INTERVAL);
        assert_eq!(SETTLE_MAX_POLL_INTERVAL, Duration::from_millis(250));
        assert_eq!(tracker.initial_delay(), Duration::ZERO);
        assert_eq!(SettleTracker::default().interval(), SETTLE_POLL_INTERVAL);
        assert_eq!(
            SettleTracker::with_intervals(MAX, FAST).interval(),
            MAX,
            "a max below fast is clamped to fast"
        );
    }

    #[test]
    fn settling_needs_two_consecutive_idle_and_settled_polls() {
        let mut tracker = tracker();
        assert!(
            !tracker.poll(true, probe(true, 1)),
            "one poll is not enough"
        );
        assert_eq!(
            tracker.interval(),
            FAST,
            "the confirming poll must not wait out a backoff"
        );
        assert!(
            tracker.poll(true, probe(true, 1)),
            "the second poll settles"
        );
    }

    #[test]
    fn a_single_settled_observation_does_not_end_the_run() {
        let mut tracker = tracker();
        assert!(!tracker.poll(true, probe(true, 1)));
        tracker.activity();
        assert!(!tracker.poll(true, probe(true, 1)));
        assert!(
            !tracker.poll(false, probe(true, 1)),
            "a busy leader is not idle"
        );
        assert!(
            !tracker.poll(true, probe(false, 2)),
            "an open task keeps the run open"
        );
        assert!(!tracker.poll(true, probe(true, 1)), "the streak restarted");
        assert!(tracker.poll(true, probe(true, 1)));
    }

    #[test]
    fn the_interval_backs_off_while_the_board_is_unchanged() {
        let mut tracker = tracker();
        let mut observed = Vec::new();
        for _ in 0..5 {
            assert!(!tracker.poll(true, probe(false, 7)));
            observed.push(tracker.interval());
        }
        assert_eq!(observed, vec![FAST, FAST * 2, FAST * 4, MAX, MAX]);
    }

    #[test]
    fn a_changed_probe_and_activity_reset_the_backoff() {
        let mut tracker = tracker();
        for _ in 0..4 {
            assert!(!tracker.poll(true, probe(false, 7)));
        }
        assert_eq!(tracker.interval(), MAX);
        assert!(!tracker.poll(true, probe(false, 8)), "the board moved");
        assert_eq!(tracker.interval(), FAST);
        for _ in 0..4 {
            assert!(!tracker.poll(true, probe(false, 8)));
        }
        assert_eq!(tracker.interval(), MAX);
        tracker.activity();
        assert_eq!(tracker.interval(), FAST);
    }

    #[test]
    fn work_finishing_after_a_long_backoff_still_settles_promptly() {
        let mut tracker = tracker();
        for _ in 0..6 {
            assert!(!tracker.poll(true, probe(false, 7)));
        }
        assert_eq!(tracker.interval(), MAX, "the quiet run backed off");
        // The last task completed: the first observation of the moved board is still open-polled
        // (the probe changed), then the decision holds for two polls on the fast interval.
        assert!(!tracker.poll(true, probe(true, 9)));
        assert_eq!(tracker.interval(), FAST);
        assert!(tracker.poll(true, probe(true, 9)));
        assert_eq!(tracker.interval(), FAST);
    }

    #[test]
    fn an_empty_board_settles_on_two_polls() {
        let mut tracker = tracker();
        let empty = SettleProbe {
            settled: true,
            tasks: 0,
            fingerprint: 0,
        };
        assert!(!tracker.poll(true, empty));
        assert!(tracker.poll(true, empty));
    }
}
