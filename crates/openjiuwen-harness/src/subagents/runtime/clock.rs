//! Injectable clock for subagent activity timestamps.
//!
//! The runtime records when a sub-agent was spawned, last ran a turn and was closed. Taking
//! those timestamps through a [`Clock`] keeps the runtime deterministic in tests: inject a
//! [`ManualClock`] to freeze or advance time instead of reading the system clock.
use std::fmt;
use std::sync::atomic::{AtomicI64, Ordering};
/// Source of the activity timestamps the sub-agent runtime persists.
///
/// Implementations must be cheap to call from async code; [`SystemClock`] reads the wall
/// clock and [`ManualClock`] a relaxed atomic, so neither blocks.
pub trait Clock: Send + Sync + fmt::Debug {
    /// Milliseconds since the Unix epoch.
    fn now_ms(&self) -> i64;
    /// RFC 3339 rendering of [`Clock::now_ms`] with millisecond precision.
    fn now_rfc3339(&self) -> String {
        rfc3339_from_ms(self.now_ms())
    }
}
/// The wall clock (`chrono::Utc::now()`), used when no clock is injected.
#[derive(Debug, Default, Clone, Copy)]
pub struct SystemClock;
impl SystemClock {
    /// Creates the system clock.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}
impl Clock for SystemClock {
    fn now_ms(&self) -> i64 {
        chrono::Utc::now().timestamp_millis()
    }
}
/// A clock that only moves when a test tells it to.
///
/// The current time lives in an [`AtomicI64`], so cloning is not offered: share one clock
/// through an `Arc` and every holder observes the same instant.
#[derive(Debug)]
pub struct ManualClock {
    /// Milliseconds since the Unix epoch.
    now_ms: AtomicI64,
}
impl ManualClock {
    /// Creates a manual clock frozen at `start_ms` milliseconds since the Unix epoch.
    #[must_use]
    pub fn new(start_ms: i64) -> Self {
        Self {
            now_ms: AtomicI64::new(start_ms),
        }
    }
    /// Advances the clock by `delta_ms` (saturating) and returns the new value.
    pub fn advance(&self, delta_ms: i64) -> i64 {
        let mut current = self.now_ms.load(Ordering::SeqCst);
        loop {
            let next = current.saturating_add(delta_ms);
            match self.now_ms.compare_exchange_weak(
                current,
                next,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => return next,
                Err(observed) => current = observed,
            }
        }
    }
    /// Sets the clock to `ms` milliseconds since the Unix epoch.
    pub fn set(&self, ms: i64) {
        self.now_ms.store(ms, Ordering::SeqCst);
    }
}
impl Clock for ManualClock {
    fn now_ms(&self) -> i64 {
        self.now_ms.load(Ordering::SeqCst)
    }
}
/// Formats `ms` as RFC 3339 in UTC with millisecond precision, e.g.
/// `1700000000000` -> `"2023-11-14T22:13:20.000Z"`; out-of-range values fall back to the epoch.
#[must_use]
pub fn rfc3339_from_ms(ms: i64) -> String {
    chrono::DateTime::from_timestamp_millis(ms)
        .unwrap_or_default()
        .to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::{rfc3339_from_ms, Clock, ManualClock, SystemClock};
    #[test]
    fn manual_clock_reports_the_injected_instant() {
        let clock = ManualClock::new(1_700_000_000_000);
        assert_eq!(clock.now_ms(), 1_700_000_000_000);
        assert_eq!(clock.now_rfc3339(), "2023-11-14T22:13:20.000Z");
    }
    #[test]
    fn manual_clock_advance_returns_the_new_value() {
        let clock = ManualClock::new(1_000);
        assert_eq!(clock.advance(500), 1_500);
        assert_eq!(clock.now_ms(), 1_500);
        assert_eq!(clock.advance(0), 1_500);
    }
    #[test]
    fn manual_clock_advance_saturates() {
        let clock = ManualClock::new(i64::MAX - 5);
        assert_eq!(clock.advance(10), i64::MAX);
        assert_eq!(clock.now_ms(), i64::MAX);
        let clock = ManualClock::new(i64::MIN + 5);
        assert_eq!(clock.advance(-10), i64::MIN);
        assert_eq!(clock.now_ms(), i64::MIN);
    }
    #[test]
    fn manual_clock_set_replaces_the_instant() {
        let clock = ManualClock::new(1_000);
        clock.set(42);
        assert_eq!(clock.now_ms(), 42);
        assert_eq!(clock.now_rfc3339(), "1970-01-01T00:00:00.042Z");
    }
    #[test]
    fn default_now_rfc3339_follows_the_injected_time() {
        let clock = ManualClock::new(0);
        assert_eq!(clock.now_rfc3339(), "1970-01-01T00:00:00.000Z");
        clock.advance(1_700_000_000_000);
        assert_eq!(clock.now_rfc3339(), "2023-11-14T22:13:20.000Z");
    }
    #[test]
    fn rfc3339_from_ms_renders_utc_millis() {
        assert_eq!(rfc3339_from_ms(0), "1970-01-01T00:00:00.000Z");
        assert_eq!(
            rfc3339_from_ms(1_700_000_000_000),
            "2023-11-14T22:13:20.000Z"
        );
        assert_eq!(
            rfc3339_from_ms(1_700_000_000_123),
            "2023-11-14T22:13:20.123Z"
        );
    }
    #[test]
    fn rfc3339_from_ms_falls_back_to_the_epoch_when_out_of_range() {
        assert_eq!(rfc3339_from_ms(i64::MAX), "1970-01-01T00:00:00.000Z");
        assert_eq!(rfc3339_from_ms(i64::MIN), "1970-01-01T00:00:00.000Z");
    }
    #[test]
    fn system_clock_reads_the_wall_clock() {
        let clock = SystemClock::new();
        assert!(clock.now_ms() > 1_600_000_000_000);
        assert!(clock.now_rfc3339().ends_with('Z'));
    }
}
