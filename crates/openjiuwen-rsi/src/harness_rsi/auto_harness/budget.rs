//! Session budget accounting: wall-clock and API-cost limits.
//!
//! Ported from Python `auto_harness.infra.session_budget.SessionBudgetController`.
//! A [`Budget`] is process-local and clock-backed through [`std::time::Instant`]:
//! it is created with [`BudgetLimits`], started by [`Budget::start`] (or the
//! [`Budget::new`] constructor), and observed through [`Budget::elapsed`],
//! [`Budget::remaining`], [`Budget::remaining_cost`], [`Budget::should_stop`],
//! and [`Budget::exhausted`].
//!
//! Python parity rules:
//!
//! - the budget is exhausted when elapsed time reaches the wall clock **or**
//!   spent cost reaches the cost limit;
//! - each dimension emits exactly one `tracing::warn!` when it first crosses
//!   80% of its limit (target `openjiuwen_rsi::auto_harness::budget`).
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use tracing::warn;

use crate::harness_rsi::auto_harness::task::StopReason;

/// `tracing` target shared by every budget warning.
const LOG_TARGET: &str = "openjiuwen_rsi::auto_harness::budget";

/// Python `_WARN_THRESHOLD`: warn once a dimension passes 80% usage.
const WARN_THRESHOLD: f64 = 0.8;

/// The limits a [`Budget`] enforces.
///
/// [`BudgetLimits::default`] is all-zero, which means "already exhausted";
/// callers must always construct real limits for a session.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct BudgetLimits {
    /// Maximum session wall-clock time.
    pub wall_clock: Duration,
    /// Maximum accumulated API cost, in US dollars.
    pub cost_limit_usd: f64,
    /// Default timeout of a single task.
    pub task_timeout: Duration,
}

/// Snapshot of a [`Budget`]'s counters, as reported by [`Budget::status`].
///
/// This is a process-local value object and deliberately not serialisable:
/// it describes a live clock, not session state.
#[derive(Debug, Clone, PartialEq)]
pub struct BudgetStatus {
    /// Seconds elapsed since the budget clock started (0 before it starts).
    pub elapsed_secs: f64,
    /// Seconds left before the wall clock is exhausted (never negative).
    pub remaining_secs: f64,
    /// Cost accumulated so far, in US dollars.
    pub spent_usd: f64,
    /// Cost left before the cost limit is exhausted (never negative).
    pub remaining_usd: f64,
    /// Whether elapsed time reached the wall clock.
    pub wall_clock_exhausted: bool,
    /// Whether spent cost reached the cost limit.
    pub cost_exhausted: bool,
}

/// Wall-clock and cost budget of one optimization session.
///
/// Construct with [`Budget::start`] for a clock that is already running, or
/// with [`Budget::new`] for a clock that has not started yet (`elapsed` is
/// zero). [`Budget::with_spent`] exists so tests can inject cost without
/// sleeping.
#[derive(Debug)]
pub struct Budget {
    limits: BudgetLimits,
    started: Option<Instant>,
    spent_usd: f64,
    time_warned: AtomicBool,
    cost_warned: AtomicBool,
}

impl Budget {
    /// Create a budget whose clock has not started yet.
    ///
    /// `elapsed` stays zero until [`Self::begin`] is called; the cost limit and
    /// `spent_usd` are still enforced.
    #[must_use]
    pub fn new(limits: BudgetLimits) -> Self {
        Self {
            limits,
            started: None,
            spent_usd: 0.0,
            time_warned: AtomicBool::new(false),
            cost_warned: AtomicBool::new(false),
        }
    }

    /// Create a budget and start its wall clock immediately.
    #[must_use]
    pub fn start(limits: BudgetLimits) -> Self {
        let mut budget = Self::new(limits);
        budget.begin();
        budget
    }

    /// Create a started budget with an injected amount of spent cost.
    ///
    /// Intended for tests and for resuming a persisted session where the cost
    /// already spent is known: it avoids sleeping to reach the cost limit.
    #[must_use]
    pub fn with_spent(limits: BudgetLimits, spent_usd: f64) -> Self {
        let mut budget = Self::start(limits);
        if spent_usd.is_finite() && spent_usd >= 0.0 {
            budget.spent_usd = spent_usd;
        }
        budget
    }

    /// Start (or restart) the wall clock at the current instant.
    ///
    /// Restarting also resets the one-shot 80% warnings, mirroring Python where
    /// `start()` begins a fresh session.
    pub fn begin(&mut self) {
        self.started = Some(Instant::now());
        self.time_warned.store(false, Ordering::Relaxed);
        self.cost_warned.store(false, Ordering::Relaxed);
    }

    /// The configured limits.
    #[must_use]
    pub fn limits(&self) -> &BudgetLimits {
        &self.limits
    }

    /// Time elapsed since the clock started; zero when it has not started.
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.started
            .map_or(Duration::ZERO, |started| started.elapsed())
    }

    /// Wall-clock time left, saturating at zero.
    #[must_use]
    pub fn remaining(&self) -> Duration {
        self.limits.wall_clock.saturating_sub(self.elapsed())
    }

    /// Cost budget left, saturating at zero.
    #[must_use]
    pub fn remaining_cost(&self) -> f64 {
        (self.limits.cost_limit_usd - self.spent_usd).max(0.0)
    }

    /// Cost accumulated so far, in US dollars.
    #[must_use]
    pub fn spent_usd(&self) -> f64 {
        self.spent_usd
    }

    /// Accumulate `amount_usd` of API cost.
    ///
    /// Non-finite and negative amounts are rejected with a `debug` log instead
    /// of poisoning the accounting, so a provider returning garbage cannot make
    /// the budget `NaN` and silently disable the cost limit.
    pub fn record_cost(&mut self, amount_usd: f64) {
        if !amount_usd.is_finite() || amount_usd < 0.0 {
            tracing::debug!(
                target: LOG_TARGET,
                amount_usd,
                "ignoring invalid cost amount"
            );
            return;
        }
        self.spent_usd += amount_usd;
        self.check_warnings();
    }

    /// Snapshot of the budget counters, including the 80% warning check.
    #[must_use]
    pub fn status(&self) -> BudgetStatus {
        self.check_warnings();
        let elapsed = self.elapsed();
        BudgetStatus {
            elapsed_secs: elapsed.as_secs_f64(),
            remaining_secs: self.remaining().as_secs_f64(),
            spent_usd: self.spent_usd,
            remaining_usd: self.remaining_cost(),
            wall_clock_exhausted: elapsed >= self.limits.wall_clock,
            cost_exhausted: self.spent_usd >= self.limits.cost_limit_usd,
        }
    }

    /// Whether either dimension is exhausted.
    #[must_use]
    pub fn should_stop(&self) -> bool {
        self.exhausted().is_some()
    }

    /// Whether a task starting now with `task_timeout` still fits in the
    /// remaining wall clock.
    ///
    /// `None` uses [`BudgetLimits::task_timeout`]. Cost is not considered here;
    /// [`Self::should_stop`] is the full check.
    #[must_use]
    pub fn allows_task(&self, task_timeout: Option<Duration>) -> bool {
        let timeout = task_timeout.unwrap_or(self.limits.task_timeout);
        self.remaining() >= timeout
    }

    /// Which limit is exhausted, if any.
    ///
    /// Wall clock and cost both map to [`StopReason::BudgetExhausted`].
    #[must_use]
    pub fn exhausted(&self) -> Option<StopReason> {
        self.check_warnings();
        if self.elapsed() >= self.limits.wall_clock || self.spent_usd >= self.limits.cost_limit_usd
        {
            return Some(StopReason::BudgetExhausted);
        }
        None
    }

    /// Emit at most one warning per dimension once it crosses 80% of its limit.
    fn check_warnings(&self) {
        let elapsed = self.elapsed();
        if self.limits.wall_clock > Duration::ZERO {
            let ratio = elapsed.as_secs_f64() / self.limits.wall_clock.as_secs_f64();
            if ratio >= WARN_THRESHOLD && !self.time_warned.swap(true, Ordering::Relaxed) {
                warn!(
                    target: LOG_TARGET,
                    ratio_pct = ratio * 100.0,
                    elapsed_secs = elapsed.as_secs_f64(),
                    limit_secs = self.limits.wall_clock.as_secs_f64(),
                    "wall-clock budget mostly consumed"
                );
            }
        }
        if self.limits.cost_limit_usd > 0.0 {
            let ratio = self.spent_usd / self.limits.cost_limit_usd;
            if ratio >= WARN_THRESHOLD && !self.cost_warned.swap(true, Ordering::Relaxed) {
                warn!(
                    target: LOG_TARGET,
                    ratio_pct = ratio * 100.0,
                    spent_usd = self.spent_usd,
                    limit_usd = self.limits.cost_limit_usd,
                    "cost budget mostly consumed"
                );
            }
        }
    }

    /// Test-only view of the one-shot wall-clock warning latch.
    #[cfg(test)]
    fn time_warned(&self) -> bool {
        self.time_warned.load(Ordering::Relaxed)
    }

    /// Test-only view of the one-shot cost warning latch.
    #[cfg(test)]
    fn cost_warned(&self) -> bool {
        self.cost_warned.load(Ordering::Relaxed)
    }

    /// Test-only constructor with a clock that started `elapsed` ago.
    #[cfg(test)]
    fn with_elapsed(limits: BudgetLimits, elapsed: Duration) -> Self {
        let started = Instant::now()
            .checked_sub(elapsed)
            .expect("test durations stay within the process clock");
        Self {
            limits,
            started: Some(started),
            spent_usd: 0.0,
            time_warned: AtomicBool::new(false),
            cost_warned: AtomicBool::new(false),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits(wall_clock: Duration, cost_limit_usd: f64) -> BudgetLimits {
        BudgetLimits {
            wall_clock,
            cost_limit_usd,
            task_timeout: Duration::from_secs(60),
        }
    }

    fn assert_close(left: f64, right: f64) {
        assert!((left - right).abs() < 1e-9, "expected {right}, got {left}");
    }

    #[test]
    fn stops_at_cost_limit() {
        let mut budget = Budget::start(limits(Duration::from_secs(3_600), 10.0));
        assert!(!budget.should_stop());
        budget.record_cost(9.99);
        assert!(!budget.should_stop());
        budget.record_cost(0.01);
        assert_eq!(budget.exhausted(), Some(StopReason::BudgetExhausted));
        assert!(budget.should_stop());
        let status = budget.status();
        assert!(status.cost_exhausted);
        assert!(!status.wall_clock_exhausted);
        assert_close(status.remaining_usd, 0.0);
    }

    #[test]
    fn stops_at_wall_clock_limit() {
        let budget = Budget::start(limits(Duration::ZERO, 10.0));
        assert_eq!(budget.exhausted(), Some(StopReason::BudgetExhausted));
        let status = budget.status();
        assert!(status.wall_clock_exhausted);
        assert!(!status.cost_exhausted);
        assert_close(status.remaining_secs, 0.0);
    }

    #[test]
    fn default_limits_are_exhausted() {
        let budget = Budget::start(BudgetLimits::default());
        assert!(budget.should_stop());
    }

    #[test]
    fn remaining_saturates_at_zero() {
        let budget = Budget::with_spent(limits(Duration::ZERO, 10.0), 25.0);
        assert_eq!(budget.remaining(), Duration::ZERO);
        assert_close(budget.remaining_cost(), 0.0);
    }

    #[test]
    fn allows_task_compares_against_remaining_clock() {
        let budget = Budget::new(BudgetLimits {
            wall_clock: Duration::from_secs(300),
            cost_limit_usd: 10.0,
            task_timeout: Duration::from_secs(60),
        });
        assert!(budget.allows_task(None));
        assert!(budget.allows_task(Some(Duration::from_secs(300))));
        assert!(!budget.allows_task(Some(Duration::from_secs(301))));

        let short = Budget::new(BudgetLimits {
            wall_clock: Duration::from_secs(30),
            cost_limit_usd: 10.0,
            task_timeout: Duration::from_secs(60),
        });
        assert!(!short.allows_task(None));
        assert!(short.allows_task(Some(Duration::from_secs(30))));
    }

    #[test]
    fn record_cost_ignores_invalid_amounts() {
        let mut budget = Budget::start(limits(Duration::from_secs(3_600), 10.0));
        budget.record_cost(1.0);
        budget.record_cost(f64::NAN);
        budget.record_cost(f64::INFINITY);
        budget.record_cost(-5.0);
        assert_close(budget.spent_usd(), 1.0);
        assert!(!budget.should_stop());
    }

    #[test]
    fn cost_warning_latches_once_at_eighty_percent() {
        let mut budget = Budget::start(limits(Duration::from_secs(3_600), 10.0));
        budget.record_cost(7.9);
        assert!(!budget.cost_warned());
        budget.record_cost(0.1);
        assert!(budget.cost_warned());
        budget.record_cost(1.0);
        assert!(budget.cost_warned());
    }

    #[test]
    fn time_warning_latches_once_at_eighty_percent() {
        let budget = Budget::with_elapsed(
            limits(Duration::from_secs(10), 10.0),
            Duration::from_secs(9),
        );
        assert!(budget.status().elapsed_secs > 9.0);
        assert!(budget.time_warned());
        let again = Budget::with_elapsed(
            limits(Duration::from_secs(10), 10.0),
            Duration::from_secs(1),
        );
        assert!(!again.time_warned());
    }

    #[test]
    fn begin_resets_warning_latches() {
        let mut budget = Budget::with_elapsed(
            limits(Duration::from_secs(10), 10.0),
            Duration::from_secs(9),
        );
        assert!(budget.status().elapsed_secs >= 9.0);
        assert!(budget.time_warned());
        budget.begin();
        assert!(!budget.time_warned());
        assert!(!budget.cost_warned());
    }

    #[test]
    fn unstarted_budget_reports_zero_elapsed() {
        let budget = Budget::new(limits(Duration::from_secs(3_600), 10.0));
        assert_eq!(budget.elapsed(), Duration::ZERO);
        assert_eq!(budget.remaining(), Duration::from_secs(3_600));
        assert!(!budget.should_stop());
        assert_close(budget.status().elapsed_secs, 0.0);
    }

    #[test]
    fn with_spent_ignores_invalid_injected_cost() {
        let budget = Budget::with_spent(limits(Duration::from_secs(3_600), 10.0), f64::NAN);
        assert_close(budget.spent_usd(), 0.0);
        let budget = Budget::with_spent(limits(Duration::from_secs(3_600), 10.0), -1.0);
        assert_close(budget.spent_usd(), 0.0);
    }

    #[test]
    fn limits_are_exposed() {
        let limits = limits(Duration::from_secs(10), 2.5);
        let budget = Budget::new(limits.clone());
        assert_eq!(budget.limits(), &limits);
    }
}
