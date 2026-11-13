//! Cancellation-safe release-then-dispose for a terminal team session.
//!
//! Ports `openjiuwen/agent_teams/kv_cache/kv_cache_cleanup.py`: when a session
//! (or a one-shot worker harness) tears down it must evict the provider-side
//! cache *before* the runtime that owns the cache handle is disposed, and that
//! cleanup must survive the caller being cancelled.
//!
//! Python reaches for `asyncio.create_task` plus `asyncio.shield` because a
//! cancelled `await` would abandon a network-bound release. In Rust dropping a
//! future is cancellation, so the equivalent guarantee is to run the whole
//! release-then-dispose sequence inside one [`tokio::spawn`]ed task: dropping
//! the future returned by
//! [`cancellation_safe_release_then_dispose`] does not abandon that task, which
//! keeps running (and is still bounded by its own timeout) after the caller is
//! gone.
//!
//! Cleanup is best effort by contract: a failing or panicking step is logged and
//! reported in the [`CleanupReport`], never propagated, so a teardown can never
//! fail because of the cache.
use futures::future::BoxFuture;
use futures::FutureExt;
use openjiuwen_agent::kv_cache::KVC_TERMINAL_CLEANUP_TIMEOUT_SECONDS;
use openjiuwen_common::error::Result;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use std::time::Duration;
use tracing::warn;
/// Outcome of one release-then-dispose cleanup.
///
/// Every field is a report, never an error return: the cleanup path logs and
/// records failures so a session teardown always completes.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct CleanupReport {
    /// Whether the release step ran and reported that it released something.
    ///
    /// A release future returning `Ok(false)` means "nothing to release": the
    /// step succeeded but `released` stays `false`. No release future at all
    /// also leaves it `false`.
    pub released: bool,
    /// Whether the dispose step completed successfully.
    pub disposed: bool,
    /// Whether the timeout expired before the cleanup finished.
    pub timed_out: bool,
    /// The release failure message, when the release step failed or panicked.
    pub release_error: Option<String>,
    /// The dispose failure message, when dispose failed or panicked.
    pub dispose_error: Option<String>,
}
/// The default bounded budget of a terminal cleanup.
///
/// [`KVC_TERMINAL_CLEANUP_TIMEOUT_SECONDS`] (5 s) is the whole release+dispose
/// budget, matching Python's default argument.
#[must_use]
pub fn terminal_cleanup_timeout() -> Duration {
    Duration::from_secs_f64(KVC_TERMINAL_CLEANUP_TIMEOUT_SECONDS)
}
/// Releases the session cache, then disposes the runtime that owns it.
///
/// `release` is optional (a session without a cache has nothing to evict);
/// `dispose` always runs, even when release failed or panicked. Both steps run
/// in order inside one spawned task, so:
///
/// - a release failure is logged and reported in
///   [`CleanupReport::release_error`], and never skips `dispose`;
/// - a panic in either step is caught, reported in the matching `*_error` field
///   and logged — this function never panics;
/// - the whole sequence is bounded by `timeout`; on expiry the task is aborted,
///   [`CleanupReport::timed_out`] is set, and whatever progress the task already
///   reported is kept;
/// - dropping the returned future does **not** abandon the cleanup task, which
///   is the Rust equivalent of Python's `asyncio.shield` — a cancelled caller
///   still leaves the cache released.
///
/// `owner_id` only labels the log lines. The function returns a report instead
/// of an error, because a teardown must not fail on cache work.
pub async fn cancellation_safe_release_then_dispose(
    release: Option<BoxFuture<'static, Result<bool>>>,
    dispose: BoxFuture<'static, Result<()>>,
    owner_id: &str,
    timeout: Duration,
) -> CleanupReport {
    let owner = owner_id.to_owned();
    let report = Arc::new(Mutex::new(CleanupReport::default()));
    let task_report = Arc::clone(&report);
    let task_owner = owner.clone();
    let mut handle = tokio::spawn(async move {
        if let Some(release) = release {
            match AssertUnwindSafe(release).catch_unwind().await {
                Ok(Ok(released)) => task_report.lock().released = released,
                Ok(Err(error)) => {
                    let message = error.to_string();
                    warn!(
                        owner = %task_owner,
                        error = %message,
                        "kv-cache release failed before dispose"
                    );
                    task_report.lock().release_error = Some(message);
                }
                Err(payload) => {
                    let message = panic_message(payload.as_ref());
                    warn!(
                        owner = %task_owner,
                        error = %message,
                        "kv-cache release panicked before dispose"
                    );
                    task_report.lock().release_error = Some(message);
                }
            }
        }
        match AssertUnwindSafe(dispose).catch_unwind().await {
            Ok(Ok(())) => task_report.lock().disposed = true,
            Ok(Err(error)) => {
                let message = error.to_string();
                warn!(
                    owner = %task_owner,
                    error = %message,
                    "kv-cache runtime dispose failed"
                );
                task_report.lock().dispose_error = Some(message);
            }
            Err(payload) => {
                let message = panic_message(payload.as_ref());
                warn!(
                    owner = %task_owner,
                    error = %message,
                    "kv-cache runtime dispose panicked"
                );
                task_report.lock().dispose_error = Some(message);
            }
        }
    });
    match tokio::time::timeout(timeout, &mut handle).await {
        Ok(Ok(())) => report.lock().clone(),
        Ok(Err(error)) => {
            warn!(
                owner = %owner,
                error = %error,
                "kv-cache cleanup task ended without a report"
            );
            report.lock().clone()
        }
        Err(_elapsed) => {
            handle.abort();
            let _ = handle.await;
            warn!(
                owner = %owner,
                timeout = ?timeout,
                "kv-cache cleanup timed out; remaining work cancelled"
            );
            let mut report = report.lock().clone();
            report.timed_out = true;
            report
        }
    }
}
/// Renders a panic payload as a message, for the report and the log line.
fn panic_message(payload: &(dyn Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_owned()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "panicked with a non-string payload".to_owned()
    }
}
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use openjiuwen_common::error::Error;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Instant;
    type Log = Arc<Mutex<Vec<&'static str>>>;
    fn log() -> Log {
        Arc::new(Mutex::new(Vec::new()))
    }
    fn recording(log: &Log, label: &'static str) -> BoxFuture<'static, Result<()>> {
        let log = Arc::clone(log);
        Box::pin(async move {
            log.lock().push(label);
            Ok(())
        })
    }
    fn release_ok(log: &Log, released: bool) -> BoxFuture<'static, Result<bool>> {
        let log = Arc::clone(log);
        Box::pin(async move {
            log.lock().push("release");
            Ok(released)
        })
    }
    #[test]
    fn report_defaults_to_nothing_done() {
        let report = CleanupReport::default();
        assert!(!report.released);
        assert!(!report.disposed);
        assert!(!report.timed_out);
        assert!(report.release_error.is_none());
        assert!(report.dispose_error.is_none());
    }
    #[test]
    fn report_serializes_every_field() {
        let report = CleanupReport {
            released: true,
            disposed: false,
            timed_out: true,
            release_error: Some("release boom".to_owned()),
            dispose_error: None,
        };
        let value = serde_json::to_value(&report).unwrap();
        assert_eq!(
            value,
            json!({
                "released": true,
                "disposed": false,
                "timed_out": true,
                "release_error": "release boom",
                "dispose_error": null,
            })
        );
        let parsed: CleanupReport = serde_json::from_value(value).unwrap();
        assert_eq!(parsed, report);
    }
    #[test]
    fn terminal_cleanup_timeout_matches_the_python_constant() {
        assert_eq!(terminal_cleanup_timeout(), Duration::from_secs(5));
        assert_eq!(
            terminal_cleanup_timeout(),
            Duration::from_secs_f64(KVC_TERMINAL_CLEANUP_TIMEOUT_SECONDS)
        );
    }
    #[tokio::test]
    async fn release_runs_before_dispose() {
        let log = log();
        let report = cancellation_safe_release_then_dispose(
            Some(release_ok(&log, true)),
            recording(&log, "dispose"),
            "session-a",
            Duration::from_secs(1),
        )
        .await;
        assert_eq!(*log.lock(), vec!["release", "dispose"]);
        assert!(report.released);
        assert!(report.disposed);
        assert!(!report.timed_out);
        assert!(report.release_error.is_none());
        assert!(report.dispose_error.is_none());
    }
    #[tokio::test]
    async fn without_release_only_dispose_runs() {
        let log = log();
        let report = cancellation_safe_release_then_dispose(
            None,
            recording(&log, "dispose"),
            "session-a",
            Duration::from_secs(1),
        )
        .await;
        assert_eq!(*log.lock(), vec!["dispose"]);
        assert!(!report.released);
        assert!(report.disposed);
        assert!(report.release_error.is_none());
    }
    #[tokio::test]
    async fn release_failure_still_disposes() {
        let log = log();
        let release: BoxFuture<'static, Result<bool>> =
            Box::pin(async { Err(Error::config("release boom")) });
        let report = cancellation_safe_release_then_dispose(
            Some(release),
            recording(&log, "dispose"),
            "session-a",
            Duration::from_secs(1),
        )
        .await;
        assert_eq!(*log.lock(), vec!["dispose"]);
        assert!(!report.released);
        assert!(report.disposed);
        assert!(!report.timed_out);
        assert!(report.release_error.unwrap().contains("release boom"));
    }
    #[tokio::test]
    async fn release_reporting_nothing_released_is_not_an_error() {
        let log = log();
        let report = cancellation_safe_release_then_dispose(
            Some(release_ok(&log, false)),
            recording(&log, "dispose"),
            "session-a",
            Duration::from_secs(1),
        )
        .await;
        assert!(!report.released);
        assert!(report.disposed);
        assert!(report.release_error.is_none());
    }
    #[tokio::test]
    async fn dispose_failure_is_reported() {
        let log = log();
        let dispose: BoxFuture<'static, Result<()>> =
            Box::pin(async { Err(Error::config("dispose boom")) });
        let report = cancellation_safe_release_then_dispose(
            Some(release_ok(&log, true)),
            dispose,
            "session-a",
            Duration::from_secs(1),
        )
        .await;
        assert!(report.released);
        assert!(!report.disposed);
        assert!(!report.timed_out);
        assert!(report.dispose_error.unwrap().contains("dispose boom"));
    }
    #[tokio::test]
    async fn release_panic_still_disposes() {
        let log = log();
        let release: BoxFuture<'static, Result<bool>> =
            Box::pin(async { panic!("release exploded") });
        let report = cancellation_safe_release_then_dispose(
            Some(release),
            recording(&log, "dispose"),
            "session-a",
            Duration::from_secs(1),
        )
        .await;
        assert_eq!(*log.lock(), vec!["dispose"]);
        assert!(report.disposed);
        assert!(!report.released);
        assert!(report.release_error.unwrap().contains("release exploded"));
    }
    #[tokio::test]
    async fn dispose_panic_is_reported() {
        let log = log();
        let dispose: BoxFuture<'static, Result<()>> =
            Box::pin(async { panic!("dispose exploded") });
        let report = cancellation_safe_release_then_dispose(
            Some(release_ok(&log, true)),
            dispose,
            "session-a",
            Duration::from_secs(1),
        )
        .await;
        assert!(report.released);
        assert!(!report.disposed);
        assert!(report.dispose_error.unwrap().contains("dispose exploded"));
    }
    #[tokio::test]
    async fn timeout_cancels_the_slow_dispose_and_keeps_partial_progress() {
        let log = log();
        let dispose_log = Arc::clone(&log);
        let dispose: BoxFuture<'static, Result<()>> = Box::pin(async move {
            tokio::time::sleep(Duration::from_millis(500)).await;
            dispose_log.lock().push("dispose");
            Ok(())
        });
        let started = Instant::now();
        let report = cancellation_safe_release_then_dispose(
            Some(release_ok(&log, true)),
            dispose,
            "session-a",
            Duration::from_millis(100),
        )
        .await;
        assert!(report.timed_out);
        assert!(!report.disposed);
        assert!(
            report.released,
            "the release step finished before the timeout"
        );
        assert!(report.release_error.is_none());
        assert!(!log.lock().contains(&"dispose"));
        assert!(started.elapsed() < Duration::from_secs(1));
    }
    #[tokio::test]
    async fn timeout_without_release_reports_no_progress() {
        let dispose: BoxFuture<'static, Result<()>> = Box::pin(async {
            tokio::time::sleep(Duration::from_millis(500)).await;
            Ok(())
        });
        let report = cancellation_safe_release_then_dispose(
            None,
            dispose,
            "session-a",
            Duration::from_millis(50),
        )
        .await;
        assert!(report.timed_out);
        assert!(!report.released);
        assert!(!report.disposed);
    }
    #[tokio::test]
    async fn dropping_the_returned_future_does_not_abandon_the_cleanup() {
        let log = log();
        let dispose_log = Arc::clone(&log);
        let dispose: BoxFuture<'static, Result<()>> = Box::pin(async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            dispose_log.lock().push("dispose");
            Ok(())
        });
        let owner = tokio::spawn(async move {
            cancellation_safe_release_then_dispose(
                None,
                dispose,
                "session-a",
                Duration::from_secs(5),
            )
            .await
        });
        tokio::task::yield_now().await;
        owner.abort();
        let _ = owner.await;
        let deadline = Instant::now() + Duration::from_millis(200);
        while log.lock().is_empty() && Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert_eq!(*log.lock(), vec!["dispose"]);
    }
    #[tokio::test]
    async fn cleanup_runs_once_per_call() {
        let calls = Arc::new(AtomicUsize::new(0));
        let release_calls = Arc::clone(&calls);
        let dispose_calls = Arc::clone(&calls);
        let release: BoxFuture<'static, Result<bool>> = Box::pin(async move {
            release_calls.fetch_add(1, Ordering::SeqCst);
            Ok(true)
        });
        let dispose: BoxFuture<'static, Result<()>> = Box::pin(async move {
            dispose_calls.fetch_add(1, Ordering::SeqCst);
            Ok(())
        });
        let report = cancellation_safe_release_then_dispose(
            Some(release),
            dispose,
            "session-a",
            Duration::from_secs(1),
        )
        .await;
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert!(report.disposed);
    }
}
