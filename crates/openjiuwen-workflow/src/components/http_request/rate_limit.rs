//! Per-component rate limiting for the `http_request` component.
//!
//! A component with `rate_limit_config.enabled` holds one [`RateLimiter`] (shared by every
//! clone of that component, and by nothing else — there is no process-global state). Before
//! each send the limiter awaits the delay its sliding window requires.
//!
//! # Sliding window
//!
//! The limiter records the *scheduled* send instant of every request and keeps at most
//! `requests_per_unit` of them. A new request may be sent at `max(now, oldest + window)`,
//! so no window ever contains more than `requests_per_unit` sends.
//!
//! [`RateLimiter::wait_for`] is the pure schedule computation (it takes the caller's clock
//! reading and returns the delay to await); [`RateLimiter::acquire`] is the thin async
//! wrapper that honours a [`CancellationToken`] while waiting.

use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use tokio_util::sync::CancellationToken;

use openjiuwen_common::error::{Error, Result};

use super::config::{HttpRateLimitConfig, HttpRateLimitUnit};

/// The seconds in each rate-limit window unit.
fn window_of(unit: HttpRateLimitUnit) -> Duration {
    match unit {
        HttpRateLimitUnit::Second => Duration::from_secs(1),
        HttpRateLimitUnit::Minute => Duration::from_secs(60),
        HttpRateLimitUnit::Hour => Duration::from_secs(3_600),
    }
}

/// A sliding-window request limiter.
#[derive(Debug)]
pub(crate) struct RateLimiter {
    /// The window length.
    window: Duration,
    /// The number of sends allowed per window.
    limit: usize,
    /// The scheduled instants of the most recent sends, ascending.
    sent: Mutex<Vec<Instant>>,
}

impl RateLimiter {
    /// Builds the limiter of `config`; `None` when rate limiting is off or unusable.
    pub(crate) fn new(config: &HttpRateLimitConfig) -> Option<Arc<Self>> {
        if !config.enabled || config.requests_per_unit == 0 {
            return None;
        }
        Some(Arc::new(Self {
            window: window_of(config.unit),
            limit: usize::try_from(config.requests_per_unit).unwrap_or(usize::MAX),
            sent: Mutex::new(Vec::new()),
        }))
    }

    /// The number of sends allowed per window.
    #[cfg(test)]
    fn limit(&self) -> usize {
        self.limit
    }

    /// The window length.
    #[cfg(test)]
    fn window(&self) -> Duration {
        self.window
    }

    /// Records one send at `now` and returns how long it must wait first.
    pub(crate) fn wait_for(&self, now: Instant) -> Duration {
        // The lock is never held across an await: this function is synchronous.
        let mut sent = self.sent.lock();
        let scheduled = match sent.len().checked_sub(self.limit) {
            Some(oldest) => (sent[oldest] + self.window).max(now),
            None => now,
        };
        sent.push(scheduled);
        let excess = sent.len().saturating_sub(self.limit);
        if excess > 0 {
            sent.drain(..excess);
        }
        scheduled.saturating_duration_since(now)
    }

    /// Waits until the window allows the next send.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Cancelled`] when `cancellation` fires while waiting.
    pub(crate) async fn acquire(&self, cancellation: &CancellationToken) -> Result<()> {
        let delay = self.wait_for(Instant::now());
        if delay.is_zero() {
            return Ok(());
        }
        tokio::select! {
            biased;
            () = cancellation.cancelled() => Err(Error::cancelled()),
            () = tokio::time::sleep(delay) => Ok(()),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// A limiter allowing `limit` requests per second.
    fn limiter(limit: u32) -> Arc<RateLimiter> {
        RateLimiter::new(&HttpRateLimitConfig {
            enabled: true,
            requests_per_unit: limit,
            unit: HttpRateLimitUnit::Second,
        })
        .expect("an enabled config yields a limiter")
    }

    #[test]
    fn disabled_or_empty_configurations_have_no_limiter() {
        assert!(RateLimiter::new(&HttpRateLimitConfig::default()).is_none());
        assert!(RateLimiter::new(&HttpRateLimitConfig {
            enabled: true,
            requests_per_unit: 0,
            ..HttpRateLimitConfig::default()
        })
        .is_none());
        let limiter = limiter(3);
        assert_eq!(limiter.limit(), 3);
        assert_eq!(limiter.window(), Duration::from_secs(1));
    }

    #[test]
    fn windows_follow_their_unit() {
        for (unit, expected) in [
            (HttpRateLimitUnit::Second, 1),
            (HttpRateLimitUnit::Minute, 60),
            (HttpRateLimitUnit::Hour, 3_600),
        ] {
            let limiter = RateLimiter::new(&HttpRateLimitConfig {
                enabled: true,
                requests_per_unit: 1,
                unit,
            })
            .unwrap();
            assert_eq!(limiter.window(), Duration::from_secs(expected));
        }
    }

    #[test]
    fn the_first_burst_is_immediate_and_the_next_send_waits() {
        let limiter = limiter(1);
        let base = Instant::now();
        assert_eq!(limiter.wait_for(base), Duration::ZERO);
        assert_eq!(
            limiter.wait_for(base + Duration::from_millis(100)),
            Duration::from_millis(900)
        );
        assert_eq!(
            limiter.wait_for(base + Duration::from_millis(200)),
            Duration::from_millis(1_800),
            "each scheduled send pushes the window forward"
        );
        // Once the window has fully passed, a send is immediate again.
        let later = base + Duration::from_secs(10);
        assert_eq!(limiter.wait_for(later), Duration::ZERO);
    }

    #[test]
    fn a_burst_of_n_is_allowed_immediately() {
        let limiter = limiter(3);
        let base = Instant::now();
        for _ in 0..3 {
            assert_eq!(limiter.wait_for(base), Duration::ZERO);
        }
        assert_eq!(limiter.wait_for(base), Duration::from_secs(1));
        assert_eq!(
            limiter.wait_for(base + Duration::from_millis(500)),
            Duration::from_millis(500),
            "the fourth and later sends are spaced one window apart"
        );
    }

    #[test]
    fn no_window_ever_exceeds_the_limit() {
        let limiter = limiter(2);
        let base = Instant::now();
        let mut scheduled: Vec<Instant> = Vec::new();
        let mut now = base;
        for _ in 0..10 {
            let delay = limiter.wait_for(now);
            scheduled.push(now + delay);
            now += Duration::from_millis(100);
        }
        for window_start in 0..scheduled.len() {
            let end = scheduled[window_start] + limiter.window();
            let in_window = scheduled
                .iter()
                .filter(|instant| **instant > scheduled[window_start] && **instant < end)
                .count();
            assert!(
                in_window < limiter.limit(),
                "a window starting at {window_start} held {} sends",
                in_window + 1
            );
        }
    }

    #[tokio::test]
    async fn a_cancelled_token_stops_the_wait() {
        let limiter = limiter(1);
        let cancellation = CancellationToken::new();
        // The first send needs no wait, so the token is only consulted once it fires.
        limiter.acquire(&cancellation).await.unwrap();
        cancellation.cancel();
        let error = limiter.acquire(&cancellation).await.unwrap_err();
        assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::Cancelled);
    }

    // timing: a generous lower bound (0.9 × window) on one real 1-second window.
    #[tokio::test]
    async fn acquire_waits_about_one_window() {
        let limiter = limiter(1);
        let cancellation = CancellationToken::new();
        limiter.acquire(&cancellation).await.unwrap();
        let started = Instant::now();
        limiter.acquire(&cancellation).await.unwrap();
        let elapsed = started.elapsed();
        assert!(
            elapsed >= Duration::from_millis(900),
            "the second send waited only {elapsed:?}"
        );
        assert!(
            elapsed < Duration::from_secs(5),
            "the second send waited unreasonably long: {elapsed:?}"
        );
    }
}
