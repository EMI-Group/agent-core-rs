//! The monitored spawn decorator: one spawned child's lifecycle reported into a
//! [`crate::monitor::RunMonitor`].
//!
//! A [`crate::spawn::SpawnedAgent`] returned by a backend reports liveness and
//! health when asked, but it has no way to tell a host that the child it
//! supervises finished, was killed, or was abandoned. A
//! [`MonitoredSpawnedAgent`] is that report: it wraps an inner handle, forwards
//! every call verbatim, and mirrors the transitions a spawn goes through into the
//! runner's monitor under [`crate::monitor::RunKind::Spawn`].
//!
//! **One run per handle.** [`MonitoredSpawnedAgent::track`] admits the handle's
//! [`crate::spawn::SpawnedAgent::id`] into the monitor as a
//! [`crate::monitor::RunPhase::Running`] spawn; [`MonitoredSpawnedAgent::new`]
//! wraps without admitting, for a caller that already tracks the id itself.
//!
//! **Observation never fails the spawn.** Every monitor call is best effort: a
//! monitor error (an unknown id, an already-terminal run) is discarded with
//! `let _ =`, because a child must not fail to shut down because the observation
//! trail did.
//!
//! **Health is pull-based.** [`crate::spawn::SpawnedAgent::is_healthy`] stays a
//! cheap pure read and records nothing; [`crate::spawn::SpawnedAgent::health`] is
//! the probe path and records each observation (a `healthy`/`alive`/`pid` detail
//! on success, a `healthy: false` detail carrying the error on failure) while
//! still returning the probe's own result to the caller.
//!
//! **A dead child is terminal.** Liveness is the one thing the decorator can
//! decide without a backend-specific probe, so it is what ends a run whose child
//! exited on its own: after every probe, and after a `send`/`stream` that returned
//! an error, a child reporting [`crate::spawn::SpawnedAgent::is_alive`] `false`
//! finishes the run as failed with `{"reason": "child_exited", "backend": …}`.
//! The call itself is unaffected — the probe result and the error are returned
//! unchanged — and a failure against a *live* child (a protocol desync, a
//! timed-out probe) leaves the run active, because the child is still there.
//! Nothing is polled in the background; a host that never probes or calls a dead
//! child keeps seeing it as active.
//!
//! **Finished exactly once.** `shutdown`, `force_kill`, [`Drop`] and the dead-child
//! rule all finish the run, and a [`std::sync::atomic::AtomicBool`] latches the
//! first terminal transition, so a shutdown followed by a drop records one
//! terminal event. A dropped handle finishes as completed with
//! `{"reason": "handle_dropped"}`, so an abandoned spawn never leaves a live status
//! behind.

use crate::monitor::{RunKind, RunMonitor, RunOutcome, RunPhase, RunStart};
use crate::spawn::{SpawnBackend, SpawnHealth, SpawnedAgent};
use async_trait::async_trait;
use futures::stream::BoxStream;
use openjiuwen_agent::output::{AgentChunk, AgentOutput};
use openjiuwen_common::error::Result;
use serde_json::{json, Value};
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// A [`SpawnedAgent`] decorator that reports its child's lifecycle into a
/// [`RunMonitor`].
///
/// See the [module docs](self) for the transitions it records; every
/// [`SpawnedAgent`] method forwards to the wrapped handle unchanged.
pub struct MonitoredSpawnedAgent {
    /// The wrapped child handle.
    inner: Arc<dyn SpawnedAgent>,
    /// The monitor the lifecycle transitions are reported into.
    monitor: Arc<RunMonitor>,
    /// The monitor run id; the inner handle's id.
    run_id: String,
    /// Latches once a terminal transition was recorded (or attempted).
    ///
    /// Shared with the [`SpawnWatch`] a returned stream carries, so a stream frame
    /// and the handle itself contend for the same latch.
    closed: Arc<AtomicBool>,
}

/// The observation state a returned stream needs, detached from the handle.
///
/// [`MonitoredSpawnedAgent::stream`] must report a terminal error frame arriving
/// after its `&self` borrow ended, so it captures one of these instead: the same
/// child, monitor, run id, and `closed` latch the handle itself uses.
#[derive(Clone)]
struct SpawnWatch {
    /// The wrapped child handle.
    inner: Arc<dyn SpawnedAgent>,
    /// The monitor the lifecycle transitions are reported into.
    monitor: Arc<RunMonitor>,
    /// The monitor run id; the inner handle's id.
    run_id: String,
    /// Latches once a terminal transition was recorded (or attempted).
    closed: Arc<AtomicBool>,
}

impl SpawnWatch {
    /// Records `outcome` unless a terminal transition was already recorded.
    ///
    /// A monitor error is discarded: observation never fails the child.
    fn finish_once(&self, outcome: RunOutcome) {
        if self.closed.swap(true, Ordering::AcqRel) {
            return;
        }
        let _ = self.monitor.finish(&self.run_id, outcome);
    }

    /// Finishes the run as failed when the wrapped child is no longer alive.
    ///
    /// The detail is `{"reason": "child_exited", "backend": …}`; a live child
    /// leaves the run untouched.
    fn finish_if_child_exited(&self) {
        if self.inner.is_alive() {
            return;
        }
        self.finish_once(RunOutcome::failed("spawned child exited").with_detail(
            json!({"reason": "child_exited", "backend": self.inner.backend().as_str()}),
        ));
    }
}

impl MonitoredSpawnedAgent {
    /// Wraps `inner` and reports into `monitor` under `run_id`.
    ///
    /// Nothing is admitted into the monitor: use [`Self::track`] when the
    /// monitor does not know the run yet.
    #[must_use]
    pub fn new(
        inner: Arc<dyn SpawnedAgent>,
        monitor: Arc<RunMonitor>,
        run_id: impl Into<String>,
    ) -> Self {
        Self {
            inner,
            monitor,
            run_id: run_id.into(),
            closed: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Admits `inner` into `monitor` as a running [`RunKind::Spawn`], then wraps
    /// it.
    ///
    /// The run id is [`SpawnedAgent::id`] and the label is the same id, so a host
    /// can address the status without holding the handle. `session_id` binds the
    /// status to the session the child was spawned for, and `detail` is recorded
    /// on both the initial event and the
    /// [`RunPhase::Running`] transition.
    ///
    /// An id the monitor already tracks is adopted unchanged (the monitor
    /// publishes nothing for it); the transition to
    /// [`RunPhase::Running`] is still attempted and its error discarded.
    #[must_use]
    pub fn track(
        inner: Arc<dyn SpawnedAgent>,
        monitor: Arc<RunMonitor>,
        session_id: Option<&str>,
        detail: Value,
    ) -> Self {
        let run_id = inner.id().to_owned();
        let mut start = RunStart::new(RunKind::Spawn, run_id.clone()).with_detail(detail.clone());
        if let Some(session_id) = session_id {
            start = start.with_session(session_id);
        }
        let _ = monitor.track(run_id.clone(), start);
        let agent = Self::new(inner, monitor, run_id);
        let _ = agent
            .monitor
            .update(&agent.run_id, RunPhase::Running, detail);
        agent
    }

    /// The wrapped child handle.
    #[must_use]
    pub fn inner(&self) -> &Arc<dyn SpawnedAgent> {
        &self.inner
    }

    /// The monitor run id this handle reports under.
    #[must_use]
    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    /// Whether this handle already recorded a terminal transition.
    ///
    /// `shutdown`, `force_kill` and drop each try to finish the run exactly once;
    /// this reports whether that attempt has been made.
    #[must_use]
    pub fn is_finished(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }

    /// Records `outcome` unless a terminal transition was already recorded.
    ///
    /// A monitor error is discarded: observation never fails the child.
    fn finish_once(&self, outcome: RunOutcome) {
        self.watch().finish_once(outcome);
    }

    /// Finishes the run as failed when the wrapped child is no longer alive.
    ///
    /// Called after a failed `send`/`stream` and after every probe, so a call that
    /// failed because the child is gone ends the run while the same error against a
    /// live child leaves it active.
    fn finish_if_child_exited(&self) {
        self.watch().finish_if_child_exited();
    }

    /// The observation state for a detached task, such as a returned stream.
    fn watch(&self) -> SpawnWatch {
        SpawnWatch {
            inner: Arc::clone(&self.inner),
            monitor: Arc::clone(&self.monitor),
            run_id: self.run_id.clone(),
            closed: Arc::clone(&self.closed),
        }
    }
}

impl fmt::Debug for MonitoredSpawnedAgent {
    /// Reports the run id, the child's backend, and whether it finished.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MonitoredSpawnedAgent")
            .field("run_id", &self.run_id)
            .field("backend", &self.inner.backend())
            .field("finished", &self.is_finished())
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl SpawnedAgent for MonitoredSpawnedAgent {
    /// Forwards the inner handle's id.
    fn id(&self) -> &str {
        self.inner.id()
    }

    /// Forwards the inner handle's liveness.
    fn is_alive(&self) -> bool {
        self.inner.is_alive()
    }

    /// Forwards the inner handle's process id.
    fn pid(&self) -> Option<u32> {
        self.inner.pid()
    }

    /// Forwards one invocation to the inner handle.
    ///
    /// An error from a child that is no longer alive also finishes the run as
    /// failed with `{"reason": "child_exited"}`; an error from a live child is
    /// returned without a terminal transition.
    ///
    /// # Errors
    ///
    /// Whatever the inner handle reports.
    async fn send(&self, inputs: Value) -> Result<AgentOutput> {
        let result = self.inner.send(inputs).await;
        if result.is_err() {
            self.finish_if_child_exited();
        }
        result
    }

    /// Forwards the inner handle's frame stream, observing a terminal error frame.
    ///
    /// The returned stream reports a dead child exactly as `send` does: an `Err`
    /// frame from a child that is no longer alive finishes the run, while an `Err`
    /// from a live child is forwarded as-is.
    fn stream(&self, inputs: Value) -> BoxStream<'static, Result<AgentChunk>> {
        let watch = self.watch();
        let inner = self.inner.stream(inputs);
        Box::pin(futures::stream::unfold(
            (inner, watch),
            |(mut inner, watch)| async move {
                let frame = futures::StreamExt::next(&mut inner).await?;
                if frame.is_err() {
                    watch.finish_if_child_exited();
                }
                Some((frame, (inner, watch)))
            },
        ))
    }

    /// Forwards the graceful shutdown and records its result.
    ///
    /// A graceful exit finishes the run as
    /// [`RunOutcome::completed`] with `{"graceful": graceful}`; an error finishes
    /// it as [`RunOutcome::failed`] with the message and is returned unchanged.
    ///
    /// # Errors
    ///
    /// Whatever the inner handle reports.
    async fn shutdown(&self, timeout: Duration) -> Result<bool> {
        match self.inner.shutdown(timeout).await {
            Ok(graceful) => {
                self.finish_once(
                    RunOutcome::completed().with_detail(json!({"graceful": graceful})),
                );
                Ok(graceful)
            }
            Err(error) => {
                self.finish_once(RunOutcome::failed(error.to_string()));
                Err(error)
            }
        }
    }

    /// Forwards the kill and records its result.
    ///
    /// A successful kill finishes the run as [`RunOutcome::cancelled`] with
    /// `{"reason": "force_kill"}`; an error finishes it as failed with the message
    /// and is returned unchanged.
    ///
    /// # Errors
    ///
    /// Whatever the inner handle reports.
    async fn force_kill(&self) -> Result<()> {
        match self.inner.force_kill().await {
            Ok(()) => {
                self.finish_once(
                    RunOutcome::cancelled().with_detail(json!({"reason": "force_kill"})),
                );
                Ok(())
            }
            Err(error) => {
                self.finish_once(RunOutcome::failed(error.to_string()));
                Err(error)
            }
        }
    }

    /// Forwards the inner handle's backend.
    fn backend(&self) -> SpawnBackend {
        self.inner.backend()
    }

    /// Forwards the inner handle's health flag and records nothing.
    ///
    /// The probe path is [`Self::health`]; this accessor stays a cheap pure read.
    fn is_healthy(&self) -> bool {
        self.inner.is_healthy()
    }

    /// Forwards the inner handle's snapshot unchanged.
    fn health_snapshot(&self) -> SpawnHealth {
        self.inner.health_snapshot()
    }

    /// Runs the inner handle's health probe and records the observation.
    ///
    /// A successful probe records `{"backend", "alive", "healthy", "pid"}` on a
    /// [`RunPhase::Running`] transition; a failed probe records
    /// `{"healthy": false, "error"}`. Either way, a probe that finds the child gone
    /// ([`SpawnedAgent::is_alive`] `false`) then finishes the run as failed with
    /// `{"reason": "child_exited", "backend": …}`; the probe's own result is
    /// returned unchanged.
    ///
    /// # Errors
    ///
    /// Whatever the inner handle's probe reports.
    async fn health(&self) -> Result<bool> {
        let result = self.inner.health().await;
        match &result {
            Ok(healthy) => {
                let _ = self.monitor.update(
                    &self.run_id,
                    RunPhase::Running,
                    json!({
                        "backend": self.inner.backend().as_str(),
                        "alive": self.inner.is_alive(),
                        "healthy": healthy,
                        "pid": self.inner.pid(),
                    }),
                );
            }
            Err(error) => {
                let _ = self.monitor.update(
                    &self.run_id,
                    RunPhase::Running,
                    json!({"healthy": false, "error": error.to_string()}),
                );
            }
        }
        self.finish_if_child_exited();
        result
    }

    /// Forwards task cancellation to the inner handle.
    ///
    /// # Errors
    ///
    /// Whatever the inner handle reports.
    async fn cancel_task(&self, task_id: &str) -> Result<()> {
        self.inner.cancel_task(task_id).await
    }
}

impl Drop for MonitoredSpawnedAgent {
    /// Finishes the run as completed with `{"reason": "handle_dropped"}` unless a
    /// terminal transition was already recorded.
    fn drop(&mut self) {
        self.finish_once(RunOutcome::completed().with_detail(json!({"reason": "handle_dropped"})));
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use openjiuwen_common::error::{Error, Result};
    use std::sync::atomic::{AtomicBool, AtomicUsize};

    /// A child handle with configurable identity, backend and probe behaviour.
    struct FakeChild {
        /// Stable handle id.
        id: String,
        /// Backend the handle reports.
        backend: SpawnBackend,
        /// Whether `health` answers `Ok(true)` or an error.
        probe_ok: bool,
        /// Whether `send` answers an error instead of an echo.
        send_fails: bool,
        /// How many `cancel_task` calls arrived.
        cancelled: Arc<AtomicUsize>,
        /// Whether the handle reports itself alive.
        alive: AtomicBool,
    }

    impl FakeChild {
        /// An alive, healthy child reporting `backend` under `id`.
        fn new(id: &str, backend: SpawnBackend) -> Self {
            Self {
                id: id.to_owned(),
                backend,
                probe_ok: true,
                send_fails: false,
                cancelled: Arc::new(AtomicUsize::new(0)),
                alive: AtomicBool::new(true),
            }
        }
    }

    #[async_trait]
    impl SpawnedAgent for FakeChild {
        fn id(&self) -> &str {
            &self.id
        }

        fn is_alive(&self) -> bool {
            self.alive.load(Ordering::Acquire)
        }

        fn pid(&self) -> Option<u32> {
            Some(4242)
        }

        async fn send(&self, inputs: Value) -> Result<AgentOutput> {
            if self.send_fails {
                return Err(Error::transport("the child did not answer"));
            }
            Ok(AgentOutput {
                output: format!("echo: {inputs}"),
                ..AgentOutput::default()
            })
        }

        fn stream(&self, _inputs: Value) -> BoxStream<'static, Result<AgentChunk>> {
            Box::pin(futures::stream::empty())
        }

        async fn shutdown(&self, _timeout: Duration) -> Result<bool> {
            self.alive.store(false, Ordering::Release);
            Ok(true)
        }

        async fn force_kill(&self) -> Result<()> {
            self.alive.store(false, Ordering::Release);
            Ok(())
        }

        fn backend(&self) -> SpawnBackend {
            self.backend
        }

        async fn health(&self) -> Result<bool> {
            if self.probe_ok {
                Ok(true)
            } else {
                Err(Error::transport("probe failed"))
            }
        }

        async fn cancel_task(&self, _task_id: &str) -> Result<()> {
            self.cancelled.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    fn tracked(
        id: &str,
        probe_ok: bool,
    ) -> (MonitoredSpawnedAgent, Arc<FakeChild>, Arc<RunMonitor>) {
        let monitor = Arc::new(RunMonitor::new());
        let child = Arc::new(FakeChild {
            probe_ok,
            ..FakeChild::new(id, SpawnBackend::InProcess)
        });
        let handle = MonitoredSpawnedAgent::track(
            Arc::clone(&child) as Arc<dyn SpawnedAgent>,
            Arc::clone(&monitor),
            Some("s-1"),
            json!({"backend": "in_process"}),
        );
        (handle, child, monitor)
    }

    #[test]
    fn forwards_the_child_surface() {
        let (handle, child, _monitor) = tracked("child-1", true);
        assert_eq!(handle.id(), "child-1");
        assert_eq!(handle.backend(), SpawnBackend::InProcess);
        assert_eq!(handle.pid(), Some(4242));
        assert!(handle.is_alive());
        assert!(handle.is_healthy());
        assert_eq!(handle.health_snapshot().pid, Some(4242));
        assert!(!handle.is_finished());
        assert_eq!(handle.inner().id(), child.id());
        assert!(format!("{handle:?}").contains("child-1"));
    }

    #[tokio::test]
    async fn forwards_send_stream_and_cancel_task() {
        let (handle, child, _monitor) = tracked("child-2", true);
        let output = handle.send(json!({"query": "hi"})).await.unwrap();
        assert_eq!(output.output, "echo: {\"query\":\"hi\"}");

        let mut frames = handle.stream(json!({}));
        assert!(futures::StreamExt::next(&mut frames).await.is_none());

        handle.cancel_task("task-7").await.unwrap();
        assert_eq!(child.cancelled.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn track_admits_a_live_spawn_run() {
        let (handle, _child, monitor) = tracked("child-3", true);
        let status = monitor.snapshot(handle.run_id()).unwrap();
        assert_eq!(status.run_id, "child-3");
        assert_eq!(status.kind, RunKind::Spawn);
        assert_eq!(status.label, "child-3");
        assert_eq!(status.phase, RunPhase::Running);
        assert_eq!(status.session_id.as_deref(), Some("s-1"));
        assert_eq!(status.detail, json!({"backend": "in_process"}));
        assert!(status.finished_at.is_none());
        assert_eq!(monitor.list_active().len(), 1);
    }

    #[tokio::test]
    async fn health_records_the_observation_and_returns_the_probe() {
        let (handle, _child, monitor) = tracked("child-4", true);
        assert!(handle.health().await.unwrap());
        let status = monitor.snapshot(handle.run_id()).unwrap();
        assert_eq!(status.phase, RunPhase::Running);
        assert_eq!(status.detail["backend"], json!("in_process"));
        assert_eq!(status.detail["alive"], json!(true));
        assert_eq!(status.detail["healthy"], json!(true));
        assert_eq!(status.detail["pid"], json!(4242));
    }

    #[tokio::test]
    async fn a_failing_probe_is_recorded_and_propagated() {
        let (handle, _child, monitor) = tracked("child-5", false);
        let error = handle.health().await.unwrap_err();
        assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::Transport);
        let status = monitor.snapshot(handle.run_id()).unwrap();
        assert_eq!(status.phase, RunPhase::Running);
        assert_eq!(status.detail["healthy"], json!(false));
        assert!(
            status.detail["error"]
                .as_str()
                .unwrap()
                .contains("probe failed"),
            "{:?}",
            status.detail
        );
    }

    #[tokio::test]
    async fn a_probe_of_a_dead_child_finishes_the_run() {
        let (handle, child, monitor) = tracked("child-11", true);
        child.alive.store(false, Ordering::Release);

        assert!(
            handle.health().await.unwrap(),
            "the probe result is returned unchanged"
        );
        let status = monitor.snapshot(handle.run_id()).unwrap();
        assert_eq!(status.phase, RunPhase::Failed);
        assert_eq!(status.error.as_deref(), Some("spawned child exited"));
        assert_eq!(
            status.detail,
            json!({"reason": "child_exited", "backend": "in_process"})
        );
        assert!(handle.is_finished());
        assert_eq!(monitor.list_active().len(), 0);
    }

    #[tokio::test]
    async fn a_failed_send_from_a_live_child_is_not_terminal() {
        let monitor = Arc::new(RunMonitor::new());
        let child = Arc::new(FakeChild {
            send_fails: true,
            ..FakeChild::new("child-12", SpawnBackend::InProcess)
        });
        let handle = MonitoredSpawnedAgent::track(
            Arc::clone(&child) as Arc<dyn SpawnedAgent>,
            Arc::clone(&monitor),
            None,
            json!({"backend": "in_process"}),
        );

        let error = handle.send(json!({})).await.unwrap_err();
        assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::Transport);
        assert_eq!(
            monitor.snapshot(handle.run_id()).unwrap().phase,
            RunPhase::Running,
            "a live child's failure only observes"
        );
        assert!(!handle.is_finished());
        assert_eq!(monitor.list_active().len(), 1);
    }

    #[tokio::test]
    async fn a_failed_send_from_a_dead_child_finishes_the_run() {
        let monitor = Arc::new(RunMonitor::new());
        let child = Arc::new(FakeChild {
            send_fails: true,
            ..FakeChild::new("child-13", SpawnBackend::InProcess)
        });
        let handle = MonitoredSpawnedAgent::track(
            Arc::clone(&child) as Arc<dyn SpawnedAgent>,
            Arc::clone(&monitor),
            None,
            json!({"backend": "in_process"}),
        );

        child.alive.store(false, Ordering::Release);
        let error = handle.send(json!({})).await.unwrap_err();
        assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::Transport);
        let status = monitor.snapshot(handle.run_id()).unwrap();
        assert_eq!(status.phase, RunPhase::Failed);
        assert_eq!(
            status.detail,
            json!({"reason": "child_exited", "backend": "in_process"})
        );
        assert!(handle.is_finished());
        assert_eq!(monitor.list_active().len(), 0);
    }

    #[tokio::test]
    async fn shutdown_finishes_the_run_as_completed() {
        let (handle, _child, monitor) = tracked("child-6", true);
        assert!(handle.shutdown(Duration::from_secs(1)).await.unwrap());
        assert!(handle.is_finished());
        let status = monitor.snapshot(handle.run_id()).unwrap();
        assert_eq!(status.phase, RunPhase::Completed);
        assert_eq!(status.detail, json!({"graceful": true}));
        assert!(status.finished_at.is_some());
        assert_eq!(monitor.list_active().len(), 0);
    }

    #[tokio::test]
    async fn force_kill_finishes_the_run_as_cancelled() {
        let (handle, _child, monitor) = tracked("child-7", true);
        handle.force_kill().await.unwrap();
        assert!(handle.is_finished());
        let status = monitor.snapshot(handle.run_id()).unwrap();
        assert_eq!(status.phase, RunPhase::Cancelled);
        assert_eq!(status.detail, json!({"reason": "force_kill"}));
        assert_eq!(monitor.list_active().len(), 0);
    }

    #[test]
    fn dropping_an_unfinished_handle_finishes_it() {
        let monitor = Arc::new(RunMonitor::new());
        let child: Arc<dyn SpawnedAgent> =
            Arc::new(FakeChild::new("child-8", SpawnBackend::InProcess));
        let handle = MonitoredSpawnedAgent::track(
            child,
            Arc::clone(&monitor),
            None,
            json!({"backend": "in_process"}),
        );
        let run_id = handle.run_id().to_owned();
        drop(handle);
        let status = monitor.snapshot(&run_id).unwrap();
        assert_eq!(status.phase, RunPhase::Completed);
        assert_eq!(status.detail, json!({"reason": "handle_dropped"}));
        assert_eq!(monitor.list_active().len(), 0);
    }

    #[tokio::test]
    async fn a_second_finish_attempt_is_harmless() {
        let (handle, _child, monitor) = tracked("child-9", true);
        handle.shutdown(Duration::from_secs(1)).await.unwrap();
        let finished_at = monitor.snapshot(handle.run_id()).unwrap().finished_at;
        assert!(handle.shutdown(Duration::from_secs(1)).await.unwrap());
        handle.force_kill().await.unwrap();
        let status = monitor.snapshot(handle.run_id()).unwrap();
        assert_eq!(status.phase, RunPhase::Completed, "the first finish wins");
        assert_eq!(status.detail, json!({"graceful": true}));
        assert_eq!(status.finished_at, finished_at);
        assert_eq!(
            monitor.history(handle.run_id()).unwrap().len(),
            3,
            "queued, running, and one terminal event"
        );
    }

    #[test]
    fn new_wraps_without_admitting_a_run() {
        let monitor = Arc::new(RunMonitor::new());
        let child: Arc<dyn SpawnedAgent> =
            Arc::new(FakeChild::new("child-10", SpawnBackend::Subprocess));
        let handle = MonitoredSpawnedAgent::new(child, Arc::clone(&monitor), "child-10");
        assert_eq!(monitor.list_active().len(), 0);
        assert!(monitor.snapshot("child-10").is_err());
        drop(handle);
        assert!(
            monitor.snapshot("child-10").is_err(),
            "no run to finish either: the drop is a discarded NotFound"
        );
    }
}
