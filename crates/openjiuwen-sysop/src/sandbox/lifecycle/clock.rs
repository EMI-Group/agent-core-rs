//! Time sources for the sandbox lifecycle.
//!
//! Every lifecycle decision takes `now_epoch_secs` as an argument or reads the injected
//! [`Clock`], so idle-stop is deterministic in tests. Only [`SystemClock`] and
//! [`now_epoch_secs`] touch [`SystemTime`]; use
//! [`SandboxLifecycle::with_clock`](crate::sandbox::lifecycle::SandboxLifecycle::with_clock)
//! and [`FixedClock`] to drive time by hand.
use std::{
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{SystemTime, UNIX_EPOCH},
};

/// A source of Unix-epoch seconds.
///
/// The lifecycle never reads the wall clock directly: it asks its clock, which a test
/// replaces with [`FixedClock`].
pub trait Clock: Send + Sync {
    /// Seconds since the Unix epoch.
    fn now_epoch_secs(&self) -> u64;
}

/// The wall clock ([`SystemTime`]); the default clock of
/// [`SandboxLifecycle::new`](crate::sandbox::lifecycle::SandboxLifecycle::new).
#[derive(Debug, Clone, Copy, Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now_epoch_secs(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |elapsed| elapsed.as_secs())
    }
}

/// A clock that only moves when a caller moves it, for deterministic idle-stop tests.
#[derive(Debug, Clone, Default)]
pub struct FixedClock {
    epoch_secs: Arc<AtomicU64>,
}

impl FixedClock {
    /// A clock pinned to `epoch_secs`.
    #[must_use]
    pub fn new(epoch_secs: u64) -> Self {
        Self {
            epoch_secs: Arc::new(AtomicU64::new(epoch_secs)),
        }
    }

    /// Moves the clock to `epoch_secs`.
    pub fn set(&self, epoch_secs: u64) {
        self.epoch_secs.store(epoch_secs, Ordering::SeqCst);
    }

    /// Advances the clock by `seconds`.
    pub fn advance(&self, seconds: u64) {
        let current = self.epoch_secs.load(Ordering::SeqCst);
        self.set(current.saturating_add(seconds));
    }
}

impl Clock for FixedClock {
    fn now_epoch_secs(&self) -> u64 {
        self.epoch_secs.load(Ordering::SeqCst)
    }
}

/// Seconds since the Unix epoch, for hosts that do not inject a [`Clock`].
#[must_use]
pub fn now_epoch_secs() -> u64 {
    SystemClock.now_epoch_secs()
}

#[cfg(test)]
mod tests {
    use super::{now_epoch_secs, Clock, FixedClock};

    #[test]
    fn fixed_clock_moves_only_when_told() {
        let clock = FixedClock::new(100);
        assert_eq!(clock.now_epoch_secs(), 100);
        clock.advance(5);
        assert_eq!(clock.now_epoch_secs(), 105);
        clock.set(7);
        assert_eq!(clock.now_epoch_secs(), 7);
        assert!(now_epoch_secs() > 1_600_000_000);
    }
}
