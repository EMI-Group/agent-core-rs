//! Per-run cancellation and deadline control for the `ReAct` loop.
//!
//! One [`RunControl`] is created per invocation. It carries the run's
//! [`CancellationToken`] (cancelled by `ReActAgent::abort` or by dropping the
//! returned stream) and the run-wide completion deadline, so every await that
//! can hang — a model stream poll, the non-streaming finalization call, or the
//! loop itself — is raced against both.
//!
//! The deadline follows the workspace's `completion_timeout_secs` convention
//! (`openjiuwen-harness`'s `DeepAgentConfig`): a positive finite number of
//! seconds, `0` (or a non-finite value) meaning no deadline.
use openjiuwen_common::error::{Error, Result};
use std::future::Future;
use std::time::Duration;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

/// Upper bound applied to a configured deadline, so an absurd value cannot
/// overflow [`Duration`].
const MAX_TIMEOUT_SECS: f64 = 1_000_000.0;

/// Cancellation and deadline of one run.
///
/// Cloning shares the same [`CancellationToken`]; the deadline is an absolute
/// instant, so all clones expire together.
#[derive(Debug, Clone)]
pub(crate) struct RunControl {
    /// The run's cancellation signal.
    pub(crate) cancellation: CancellationToken,
    /// Absolute completion deadline, `None` when disabled.
    deadline: Option<Instant>,
    /// Configured deadline in seconds, reported in the timeout message.
    timeout_secs: f64,
}

impl RunControl {
    /// Builds a control for `cancellation` with a `timeout_secs` run deadline.
    ///
    /// A non-positive or non-finite `timeout_secs` disables the deadline.
    pub(crate) fn new(cancellation: CancellationToken, timeout_secs: f64) -> Self {
        Self {
            cancellation,
            deadline: deadline_for(timeout_secs),
            timeout_secs,
        }
    }

    /// Returns the typed error when the run is already cancelled or expired.
    ///
    /// # Errors
    ///
    /// [`Error::Cancelled`] once the token is cancelled, or [`Error::Timeout`]
    /// once the deadline elapsed.
    pub(crate) fn check(&self) -> Result<()> {
        if self.cancellation.is_cancelled() {
            return Err(Error::Cancelled);
        }
        if self
            .deadline
            .is_some_and(|deadline| Instant::now() >= deadline)
        {
            return Err(self.timeout_error());
        }
        Ok(())
    }

    /// Awaits `future`, failing with the typed cancellation or timeout error
    /// when either fires first.
    ///
    /// Cancellation is biased so a dropped consumer aborts the run even when a
    /// frame or stream item is already ready.
    ///
    /// # Errors
    ///
    /// [`Error::Cancelled`] or [`Error::Timeout`]; the future's own output is
    /// returned unchanged otherwise.
    pub(crate) async fn race<F: Future>(&self, future: F) -> Result<F::Output> {
        self.check()?;
        if let Some(deadline) = self.deadline {
            tokio::select! {
                biased;
                () = self.cancellation.cancelled() => Err(Error::Cancelled),
                () = tokio::time::sleep_until(deadline) => Err(self.timeout_error()),
                output = future => Ok(output),
            }
        } else {
            tokio::select! {
                biased;
                () = self.cancellation.cancelled() => Err(Error::Cancelled),
                output = future => Ok(output),
            }
        }
    }

    /// The typed error a deadline expiry reports.
    fn timeout_error(&self) -> Error {
        Error::timeout(format!(
            "agent run exceeded the {}s completion timeout",
            self.timeout_secs
        ))
    }
}

/// Maps a configured `secs` deadline onto an absolute instant.
pub(crate) fn deadline_for(secs: f64) -> Option<Instant> {
    if secs.is_finite() && secs > 0.0 {
        Some(Instant::now() + Duration::from_secs_f64(secs.min(MAX_TIMEOUT_SECS)))
    } else {
        None
    }
}

/// Maps a configured `secs` per-call deadline onto a [`Duration`].
pub(crate) fn timeout_duration(secs: f64) -> Option<Duration> {
    if secs.is_finite() && secs > 0.0 {
        Some(Duration::from_secs_f64(secs.min(MAX_TIMEOUT_SECS)))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn an_active_control_runs_the_future() {
        let control = RunControl::new(CancellationToken::new(), 0.0);
        assert_eq!(control.race(async { 7 }).await.unwrap(), 7);
        control.check().unwrap();
    }

    #[tokio::test]
    async fn a_cancelled_control_reports_cancellation() {
        let token = CancellationToken::new();
        let control = RunControl::new(token.clone(), 0.0);
        token.cancel();
        let error = control.race(async { 7 }).await.unwrap_err();
        assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::Cancelled);
        assert_eq!(
            control.check().unwrap_err().kind(),
            openjiuwen_common::error::ErrorKind::Cancelled
        );
    }

    #[tokio::test]
    async fn an_expired_deadline_reports_a_timeout() {
        let control = RunControl::new(CancellationToken::new(), 0.01);
        let error = control
            .race(std::future::pending::<()>())
            .await
            .unwrap_err();
        assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::Timeout);
        assert!(error.message().contains("0.01s completion timeout"));
        assert_eq!(
            control.check().unwrap_err().kind(),
            openjiuwen_common::error::ErrorKind::Timeout
        );
    }

    #[tokio::test]
    async fn a_non_positive_or_non_finite_deadline_is_disabled() {
        for secs in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(deadline_for(secs).is_none(), "{secs}");
            assert!(timeout_duration(secs).is_none(), "{secs}");
        }
        assert_eq!(timeout_duration(2.5), Some(Duration::from_secs_f64(2.5)));
    }
}
