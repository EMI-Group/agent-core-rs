//! Bounded asynchronous polling helpers.
//!
//! Tests that observe state written by a background task (a spawned agent turn,
//! an MQ consumer, a file watcher) need to wait for eventual consistency without
//! sleeping for a fixed period. These helpers poll a predicate until it is
//! satisfied or a deadline expires, and report the expiry as
//! [`ErrorKind::Timeout`](openjiuwen_common::error::ErrorKind::Timeout) so a
//! regression fails fast instead of hanging or passing late.

use std::future::Future;
use std::time::Duration;

use openjiuwen_common::error::{Error, Result};
use tokio::time::{sleep, Instant};

/// Default budget for [`eventually`] and [`eventually_with_default_timeout`].
///
/// Two seconds is long enough for an in-process task to be scheduled on a
/// loaded CI machine and short enough that a regression fails quickly.
pub const DEFAULT_EVENTUALLY_TIMEOUT: Duration = Duration::from_secs(2);

/// Default pause between two predicate evaluations (10 ms).
pub const DEFAULT_POLL_INTERVAL: Duration = Duration::from_millis(10);

/// Polls `predicate` until it returns `true` or `timeout` elapses.
///
/// The predicate is evaluated at least once. Between attempts the task sleeps
/// [`DEFAULT_POLL_INTERVAL`] (capped at the remaining budget), so a failing
/// wait neither busy-spins nor oversleeps its deadline.
///
/// # Errors
///
/// Returns [`Error::timeout`] once the budget is exhausted:
/// `eventually: predicate did not become true within <timeout>`.
pub async fn eventually<F, Fut>(timeout: Duration, predicate: F) -> Result<()>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = bool>,
{
    eventually_with_interval(DEFAULT_POLL_INTERVAL, timeout, predicate).await
}

/// Polls `predicate` every `interval` until it returns `true` or `timeout`
/// elapses.
///
/// The predicate is evaluated at least once, and the sleep between attempts is
/// capped at the remaining budget, so the total wait never exceeds `timeout` by
/// more than one predicate evaluation.
///
/// # Errors
///
/// Returns [`Error::timeout`] once the budget is exhausted.
pub async fn eventually_with_interval<F, Fut>(
    interval: Duration,
    timeout: Duration,
    mut predicate: F,
) -> Result<()>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = bool>,
{
    let deadline = Instant::now() + timeout;
    let mut polled = false;
    loop {
        if polled {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(Error::timeout(format!(
                    "eventually: predicate did not become true within {timeout:?}"
                )));
            }
            sleep(interval.min(remaining)).await;
        }
        polled = true;
        if predicate().await {
            return Ok(());
        }
    }
}

/// Polls `predicate` until it yields `Some` or `timeout` elapses.
///
/// This is the variant for waits whose success carries a value (for example the
/// first recorded request, or the first message published on a bus). The
/// predicate is evaluated at least once and the poll interval is
/// [`DEFAULT_POLL_INTERVAL`].
///
/// # Errors
///
/// Returns [`Error::timeout`] once the budget is exhausted:
/// `eventually_some: no value became available within <timeout>`.
pub async fn eventually_some<F, Fut, T>(timeout: Duration, mut predicate: F) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Option<T>>,
{
    let deadline = Instant::now() + timeout;
    let mut polled = false;
    loop {
        if polled {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(Error::timeout(format!(
                    "eventually_some: no value became available within {timeout:?}"
                )));
            }
            sleep(DEFAULT_POLL_INTERVAL.min(remaining)).await;
        }
        polled = true;
        if let Some(value) = predicate().await {
            return Ok(value);
        }
    }
}

/// [`eventually`] with [`DEFAULT_EVENTUALLY_TIMEOUT`].
///
/// # Errors
///
/// Returns [`Error::timeout`] once the default budget is exhausted.
pub async fn eventually_with_default_timeout<F, Fut>(predicate: F) -> Result<()>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = bool>,
{
    eventually(DEFAULT_EVENTUALLY_TIMEOUT, predicate).await
}

#[cfg(test)]
mod tests {
    use std::future::ready;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use openjiuwen_common::error::ErrorKind;

    use super::*;

    #[test]
    fn defaults_are_short() {
        assert_eq!(DEFAULT_EVENTUALLY_TIMEOUT, Duration::from_secs(2));
        assert_eq!(DEFAULT_POLL_INTERVAL, Duration::from_millis(10));
        assert!(DEFAULT_POLL_INTERVAL < DEFAULT_EVENTUALLY_TIMEOUT);
    }

    #[tokio::test]
    async fn succeeds_on_the_first_poll() {
        let mut calls = 0;

        eventually(Duration::from_secs(1), || {
            calls += 1;
            ready(true)
        })
        .await
        .expect("first poll succeeds");

        assert_eq!(calls, 1);
    }

    #[tokio::test]
    async fn succeeds_after_a_few_polls() {
        let polls = AtomicUsize::new(0);

        eventually(Duration::from_secs(2), || {
            ready(polls.fetch_add(1, Ordering::SeqCst) >= 2)
        })
        .await
        .expect("third poll succeeds");

        assert_eq!(polls.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn times_out_when_the_predicate_stays_false() {
        let polls = AtomicUsize::new(0);

        let error =
            eventually_with_interval(Duration::from_millis(5), Duration::from_millis(20), || {
                polls.fetch_add(1, Ordering::SeqCst);
                ready(false)
            })
            .await
            .expect_err("never true");

        assert_eq!(error.kind(), ErrorKind::Timeout);
        assert!(error.to_string().contains("did not become true"));
        assert!(error.to_string().contains("20ms"));
        assert!(polls.load(Ordering::SeqCst) >= 1);
    }

    #[tokio::test]
    async fn missing_value_times_out_with_its_own_message() {
        let error = eventually_some(Duration::from_millis(20), || ready(None::<u32>))
            .await
            .expect_err("no value");

        assert_eq!(error.kind(), ErrorKind::Timeout);
        assert!(error.to_string().contains("no value became available"));
    }

    #[tokio::test]
    async fn some_returns_the_first_available_value() {
        let polls = AtomicUsize::new(0);

        let value = eventually_some(Duration::from_secs(2), || {
            let poll = polls.fetch_add(1, Ordering::SeqCst);
            ready((poll >= 1).then_some(42_u32))
        })
        .await
        .expect("value becomes available");

        assert_eq!(value, 42);
        assert_eq!(polls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn default_timeout_helper_succeeds() {
        eventually_with_default_timeout(|| ready(true))
            .await
            .expect("ready predicate");
    }
}
