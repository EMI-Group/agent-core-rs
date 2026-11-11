//! Retry policy for the provider-agnostic HTTP client.
//!
//! The Python `clients/` layer does not retry; the Rust transport adds retries
//! because every LLM client relies on it. [`RetryPolicy`] is shared by
//! [`crate::http::HttpClient`] and can be used directly by callers that drive
//! their own transports.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::error::Error;

/// Default statuses retried by [`RetryPolicy::default`].
fn default_retry_statuses() -> Vec<u16> {
    vec![429, 500, 502, 503, 504]
}

/// Retry schedule and classification for HTTP requests.
///
/// `max_retries` counts the *additional* attempts after the first one, so a
/// policy with `max_retries == 2` performs at most three attempts.
///
/// ```
/// use std::time::Duration;
/// use openjiuwen_common::http::RetryPolicy;
///
/// let policy = RetryPolicy {
///     max_retries: 2,
///     initial_backoff: Duration::from_millis(100),
///     max_backoff: Duration::from_secs(1),
///     retry_statuses: vec![503],
///     jitter: false,
/// };
///
/// assert!(policy.is_retryable_status(503));
/// assert!(!policy.is_retryable_status(404));
/// assert_eq!(policy.backoff(2), Duration::from_millis(200));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct RetryPolicy {
    /// Number of retries attempted after the first request.
    pub max_retries: u32,
    /// Backoff before the first retry.
    pub initial_backoff: Duration,
    /// Upper bound on the exponential backoff.
    pub max_backoff: Duration,
    /// Statuses that trigger a retry.
    pub retry_statuses: Vec<u16>,
    /// Apply a randomized factor in `[0.5, 1.0]` to each backoff.
    pub jitter: bool,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 0,
            initial_backoff: Duration::from_millis(200),
            max_backoff: Duration::from_secs(5),
            retry_statuses: default_retry_statuses(),
            jitter: true,
        }
    }
}

impl RetryPolicy {
    /// Returns `true` when `status` is configured as retryable.
    #[must_use]
    pub fn is_retryable_status(&self, status: u16) -> bool {
        self.retry_statuses.contains(&status)
    }

    /// Decides whether another attempt should be made.
    ///
    /// Retries are exhausted once `attempt` reaches
    /// [`max_retries`](Self::max_retries). Otherwise the request is retried when
    /// the error is retryable ([`Error::is_retryable`], i.e. transport errors and
    /// timeouts) or the status is in [`retry_statuses`](Self::retry_statuses).
    #[must_use]
    pub fn should_retry(&self, attempt: u32, error: Option<&Error>, status: Option<u16>) -> bool {
        if attempt >= self.max_retries {
            return false;
        }

        error.is_some_and(Error::is_retryable)
            || status.is_some_and(|status| self.is_retryable_status(status))
    }

    /// Computes the backoff before retry number `attempt`.
    ///
    /// The base delay is `initial_backoff * 2^(attempt - 1)` (so attempt 0 and 1
    /// both use the initial delay) capped at [`max_backoff`](Self::max_backoff).
    /// With jitter enabled the result is multiplied by a pseudo-random factor in
    /// `[0.5, 1.0]` derived from the system clock, so it never exceeds the cap.
    #[must_use]
    pub fn backoff(&self, attempt: u32) -> Duration {
        let exponent = attempt.saturating_sub(1);
        let factor = 1_u32.checked_shl(exponent).unwrap_or(u32::MAX);
        let base = self
            .initial_backoff
            .saturating_mul(factor)
            .min(self.max_backoff);

        if !self.jitter {
            return base;
        }

        let jitter = jitter_factor();
        base.mul_f64(jitter).min(self.max_backoff)
    }

    /// Returns the configured retryable statuses.
    #[must_use]
    pub fn statuses(&self) -> &[u16] {
        &self.retry_statuses
    }
}

/// A clock-derived factor in `[0.5, 1.0]`, avoiding a dependency on `rand`.
fn jitter_factor() -> f64 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.subsec_nanos());
    f64::from(nanos % 1000) / 1000.0 / 2.0 + 0.5
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn policy_with(initial: Duration, max: Duration, jitter: bool) -> RetryPolicy {
        RetryPolicy {
            max_retries: 4,
            initial_backoff: initial,
            max_backoff: max,
            retry_statuses: vec![429, 503],
            jitter,
        }
    }

    #[test]
    fn defaults_match_the_documented_values() {
        let policy = RetryPolicy::default();
        assert_eq!(policy.max_retries, 0);
        assert_eq!(policy.initial_backoff, Duration::from_millis(200));
        assert_eq!(policy.max_backoff, Duration::from_secs(5));
        assert_eq!(policy.statuses(), &[429, 500, 502, 503, 504]);
        assert!(policy.jitter);
    }

    #[test]
    fn exponential_backoff_grows_and_saturates() {
        let policy = policy_with(Duration::from_millis(100), Duration::from_secs(10), false);

        assert_eq!(policy.backoff(0), Duration::from_millis(100));
        assert_eq!(policy.backoff(1), Duration::from_millis(100));
        assert_eq!(policy.backoff(2), Duration::from_millis(200));
        assert_eq!(policy.backoff(3), Duration::from_millis(400));
        assert_eq!(policy.backoff(4), Duration::from_millis(800));
    }

    #[test]
    fn backoff_is_capped() {
        let policy = policy_with(Duration::from_secs(1), Duration::from_millis(2500), false);
        assert_eq!(policy.backoff(5), Duration::from_millis(2500));
        assert_eq!(policy.backoff(64), Duration::from_millis(2500));
        assert!(policy.backoff(500) <= policy.max_backoff);
    }

    #[test]
    fn jitter_stays_within_bounds() {
        let policy = policy_with(Duration::from_millis(100), Duration::from_secs(10), true);

        for attempt in 0_u32..50 {
            let exponent = attempt.saturating_sub(1);
            let factor = 1_u32.checked_shl(exponent).unwrap_or(u32::MAX);
            let base = Duration::from_millis(100)
                .saturating_mul(factor)
                .min(Duration::from_secs(10));
            let delay = policy.backoff(attempt);
            assert!(delay >= base.div_f64(2.0), "attempt {attempt}: {delay:?}");
            assert!(delay <= base, "attempt {attempt}: {delay:?}");
        }
    }

    #[test]
    fn jitter_on_a_capped_delay_never_exceeds_the_cap() {
        let policy = policy_with(Duration::from_secs(1), Duration::from_millis(1500), true);

        for _ in 0..50 {
            let delay = policy.backoff(10);
            assert!(delay >= Duration::from_millis(750));
            assert!(delay <= Duration::from_millis(1500));
        }
    }

    #[test]
    fn zero_initial_backoff_is_tolerated() {
        let policy = policy_with(Duration::ZERO, Duration::from_secs(1), true);
        assert_eq!(policy.backoff(0), Duration::ZERO);
        assert_eq!(policy.backoff(3), Duration::ZERO);
    }

    #[test]
    fn retryable_statuses_are_exact_members() {
        let policy = RetryPolicy::default();
        assert!(policy.is_retryable_status(429));
        assert!(policy.is_retryable_status(500));
        assert!(policy.is_retryable_status(502));
        assert!(policy.is_retryable_status(503));
        assert!(policy.is_retryable_status(504));
        assert!(!policy.is_retryable_status(400));
        assert!(!policy.is_retryable_status(404));
        assert!(!policy.is_retryable_status(501));
    }

    #[test]
    fn should_retry_respects_the_attempt_budget() {
        let policy = RetryPolicy {
            max_retries: 2,
            ..RetryPolicy::default()
        };

        assert!(policy.should_retry(0, None, Some(503)));
        assert!(policy.should_retry(1, None, Some(503)));
        assert!(!policy.should_retry(2, None, Some(503)));
    }

    #[test]
    fn should_retry_classifies_statuses_and_errors() {
        let policy = RetryPolicy {
            max_retries: 3,
            ..RetryPolicy::default()
        };

        assert!(policy.should_retry(0, None, Some(500)));
        assert!(!policy.should_retry(0, None, Some(404)));
        assert!(!policy.should_retry(0, None, None));

        assert!(policy.should_retry(0, Some(&Error::transport("reset")), None));
        assert!(policy.should_retry(0, Some(&Error::timeout("deadline")), None));
        assert!(!policy.should_retry(0, Some(&Error::validation("bad")), None));
        assert!(!policy.should_retry(0, Some(&Error::cancelled()), None));
    }

    #[test]
    fn should_retry_with_zero_retries_is_never_true() {
        let policy = RetryPolicy::default();
        assert!(!policy.should_retry(0, Some(&Error::transport("reset")), None));
        assert!(!policy.should_retry(0, None, Some(503)));
    }
}
