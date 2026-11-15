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
//! **A cadence the backend cannot honour is supervised here.**
//! [`crate::spawn::SpawnedAgent::set_health_interval`] forwards to the wrapped
//! handle first: a backend that runs its own loop keeps owning its cadence and the
//! decorator records nothing extra, because a cadence is configuration, not a
//! lifecycle transition. When the backend answers
//! [`openjiuwen_common::error::Error::Unsupported`] the decorator starts — and on
//! the next call re-tunes — one supervised loop of its own over the wrapped handle
//! and returns `Ok(())`: a monitored handle honours the cadence even when the
//! backend cannot, and a [`Duration::ZERO`] interval stops that loop (the zero is
//! still forwarded to the backend). The loop is one owned task, so a re-tune
//! reschedules instead of accumulating, and `shutdown`, `force_kill`, a terminal
//! finish, [`Drop`] and the run reaching a terminal phase all stop it.
//!
//! Each supervised tick records exactly what
//! [`crate::spawn::SpawnedAgent::health`] records — the same probe, the same
//! `backend`/`alive`/`healthy`/`pid` detail on success, the same `healthy: false`
//! plus `error` detail on failure, and the same dead-child rule — and an
//! observation error is discarded with `let _ =`, so it never fails the spawn. The
//! one difference is deliberate: a backend whose own probe is
//! [`openjiuwen_common::error::Error::Unsupported`] (a bare host handle) is
//! observed by liveness, because `is_alive`/`pid` are real data while a fabricated
//! probe result would not be; that tick records
//! `{"backend", "alive", "healthy", "pid", "probe": "liveness_only"}`.
//!
//! **A dead child is terminal.** Liveness is the one thing the decorator can
//! decide without a backend-specific probe, so it is what ends a run whose child
//! exited on its own: after every probe, and after a `send`/`stream` (or their
//! bounded `send_with_timeout`/`stream_with_timeout` forms) that returned an
//! error, a child reporting [`crate::spawn::SpawnedAgent::is_alive`] `false`
//! finishes the run as failed with `{"reason": "child_exited", "backend": …}`.
//! The call itself is unaffected — the probe result and the error are returned
//! unchanged — and a failure against a *live* child (a protocol desync, a
//! timed-out probe) leaves the run active, because the child is still there.
//! Nothing is polled unless a cadence was asked for; a host that never probes,
//! never calls a dead child, and never sets an interval keeps seeing it as active.
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
use openjiuwen_common::error::{ErrorKind, Result};
use serde_json::{json, Value};
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

/// A [`SpawnedAgent`] decorator that reports its child's lifecycle into a
/// [`RunMonitor`].
///
/// See the [module docs](self) for the transitions it records and for the health
/// cadence it supervises when the wrapped backend cannot; every
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
    /// The supervised health loop, when the wrapped backend cannot re-tune its own.
    ///
    /// Nothing runs until a caller asks for a cadence the backend rejects with
    /// [`openjiuwen_common::error::Error::Unsupported`].
    supervisor: Arc<HealthSupervisor>,
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

    /// Probes the wrapped child and records the observation into the monitor.
    ///
    /// Both the explicit probe path
    /// ([`SpawnedAgent::health`](crate::spawn::SpawnedAgent::health)) and the
    /// supervised tick funnel through here, so they record the same details: a
    /// successful probe records `{"backend", "alive", "healthy", "pid"}`, a failed
    /// one `{"healthy": false, "error"}`.
    ///
    /// `mode` decides what an *unimplemented* probe means: the explicit path
    /// records it as the failure it is, while a supervised tick records the
    /// liveness it can still read (`{"backend", "alive", "healthy", "pid",
    /// "probe": "liveness_only"}`), because there is a real difference between "the
    /// child is up but cannot answer a probe" and "the child is gone".
    ///
    /// Every monitor call is discarded with `let _ =` — an observation error must
    /// never fail the spawn — and the probe's own result is returned unchanged,
    /// including the dead-child rule that finishes the run after it.
    async fn record_probe(&self, mode: ProbeMode) -> Result<bool> {
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
            Err(error) if mode == ProbeMode::Tick && error.kind() == ErrorKind::Unsupported => {
                let _ = self.monitor.update(
                    &self.run_id,
                    RunPhase::Running,
                    json!({
                        "backend": self.inner.backend().as_str(),
                        "alive": self.inner.is_alive(),
                        "healthy": self.inner.is_healthy(),
                        "pid": self.inner.pid(),
                        "probe": "liveness_only",
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
}

/// Which observation a probe result is recorded as.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProbeMode {
    /// An explicit [`crate::spawn::SpawnedAgent::health`] call: record the probe's
    /// own answer, and an unimplemented probe as the failed observation it is.
    Probe,
    /// A supervised tick: an unimplemented probe is recorded by liveness instead of
    /// as a fabricated failure.
    Tick,
}

/// One supervised health loop, plus the signal that stops it.
struct SupervisedLoop {
    /// Cancels the loop between observations, so a re-tune never abandons a probe.
    stop: CancellationToken,
    /// The loop task; joined by [`HealthSupervisor::stop`], aborted on drop.
    task: tokio::task::JoinHandle<()>,
}

/// Owns the decorator's one supervised health loop.
///
/// A backend that implements
/// [`SpawnedAgent::set_health_interval`](crate::spawn::SpawnedAgent::set_health_interval)
/// supervises itself, so the decorator never starts a loop for it. A backend that
/// answers [`openjiuwen_common::error::Error::Unsupported`] cannot, so the
/// decorator owns the cadence here: one slot, which means a re-tune cancels and
/// joins the previous loop before arming the next (no task is ever leaked), and a
/// [`Duration::ZERO`] interval leaves the slot empty.
///
/// The loop task holds only a [`SpawnWatch`] and a stop token, never an [`Arc`] of
/// this supervisor, so it can never keep its own owner alive.
#[derive(Default)]
struct HealthSupervisor {
    /// The running loop, if a cadence was requested and the backend refused it.
    slot: parking_lot::Mutex<Option<SupervisedLoop>>,
}

impl HealthSupervisor {
    /// A supervisor with no loop running.
    fn new() -> Self {
        Self::default()
    }

    /// Cancels and joins the running loop, if any.
    async fn stop(&self) {
        let Some(running) = self.slot.lock().take() else {
            return;
        };
        running.stop.cancel();
        let _ = running.task.await;
    }

    /// Replaces the running loop with one observing every `interval`.
    ///
    /// A [`Duration::ZERO`] interval stops the loop and arms nothing.
    async fn retune(&self, interval: Duration, watch: SpawnWatch) {
        self.stop().await;
        if interval.is_zero() {
            return;
        }
        let stop = CancellationToken::new();
        let task = tokio::spawn(supervised_health_loop(interval, watch, stop.clone()));
        *self.slot.lock() = Some(SupervisedLoop { stop, task });
    }
}

impl Drop for HealthSupervisor {
    /// Aborts the supervised loop when the last handle goes away.
    fn drop(&mut self) {
        if let Some(running) = self.slot.get_mut().take() {
            running.stop.cancel();
            running.task.abort();
        }
    }
}

/// The supervised loop: one observation per interval until the run finishes.
///
/// The loop ends on the stop token, on the crate's own latch (a terminal finish
/// from a shutdown, a kill, a dead child, or a dropped handle), and therefore
/// always within one interval of the run finishing.
async fn supervised_health_loop(interval: Duration, watch: SpawnWatch, stop: CancellationToken) {
    let mut ticker = tokio::time::interval(interval);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    // The first tick completes immediately; consume it so the first observation
    // comes one whole interval after the re-tune.
    ticker.tick().await;
    loop {
        if watch.closed.load(Ordering::Acquire) {
            return;
        }
        tokio::select! {
            () = stop.cancelled() => return,
            _ = ticker.tick() => {}
        }
        let _ = watch.record_probe(ProbeMode::Tick).await;
    }
}

/// Wraps `inner` so a terminal error frame from a dead child finishes the run.
///
/// Shared by both stream overrides ([`SpawnedAgent::stream`] and
/// [`SpawnedAgent::stream_with_timeout`]), so the bounded form is observed
/// exactly like the unbounded one: frames pass through unchanged, and an `Err`
/// frame only ends the run when the child is no longer alive.
fn observed_stream(
    inner: BoxStream<'static, Result<AgentChunk>>,
    watch: SpawnWatch,
) -> BoxStream<'static, Result<AgentChunk>> {
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
            supervisor: Arc::new(HealthSupervisor::new()),
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
        observed_stream(self.inner.stream(inputs), self.watch())
    }

    /// Forwards the bounded send, including the caller's deadline.
    ///
    /// The deadline reaches the wrapped handle unchanged (a backend that honours
    /// it bounds that one exchange; the trait default ignores it), and the
    /// observation rule is `send`'s: an error from a child that is no longer alive
    /// finishes the run as failed with `{"reason": "child_exited"}`, while an error
    /// from a live child — a timed-out request, for instance — leaves it running.
    ///
    /// # Errors
    ///
    /// Whatever the inner handle reports, including
    /// [`openjiuwen_common::error::Error::Timeout`] from a backend that honours the
    /// deadline.
    async fn send_with_timeout(
        &self,
        inputs: Value,
        timeout: Option<Duration>,
    ) -> Result<AgentOutput> {
        let result = self.inner.send_with_timeout(inputs, timeout).await;
        if result.is_err() {
            self.finish_if_child_exited();
        }
        result
    }

    /// Forwards the bounded stream, including the caller's deadline.
    ///
    /// Frames and the terminal error are observed exactly as `stream` observes
    /// them; the deadline is only passed on to the wrapped handle.
    fn stream_with_timeout(
        &self,
        inputs: Value,
        timeout: Option<Duration>,
    ) -> BoxStream<'static, Result<AgentChunk>> {
        observed_stream(
            self.inner.stream_with_timeout(inputs, timeout),
            self.watch(),
        )
    }

    /// Forwards the graceful shutdown and records its result.
    ///
    /// The supervised health loop — if one is running — is cancelled and joined
    /// before the shutdown exchange, so no probe can race it. A graceful exit then
    /// finishes the run as [`RunOutcome::completed`] with `{"graceful": graceful}`;
    /// an error finishes it as [`RunOutcome::failed`] with the message and is
    /// returned unchanged.
    ///
    /// # Errors
    ///
    /// Whatever the inner handle reports.
    async fn shutdown(&self, timeout: Duration) -> Result<bool> {
        self.supervisor.stop().await;
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
    /// The supervised health loop — if one is running — is cancelled and joined
    /// first. A successful kill finishes the run as [`RunOutcome::cancelled`] with
    /// `{"reason": "force_kill"}`; an error finishes it as failed with the message
    /// and is returned unchanged.
    ///
    /// # Errors
    ///
    /// Whatever the inner handle reports.
    async fn force_kill(&self) -> Result<()> {
        self.supervisor.stop().await;
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
        self.watch().record_probe(ProbeMode::Probe).await
    }

    /// Forwards a health-interval re-tune to the inner handle, supervising the
    /// cadence here when the backend cannot.
    ///
    /// The forward happens first and its answer decides:
    ///
    /// - the backend accepted, so its own loop owns the cadence (a decorator loop
    ///   would report every observation twice) and any supervised loop is stopped;
    /// - the backend answered [`openjiuwen_common::error::Error::Unsupported`], so
    ///   the decorator starts — or re-tunes — one supervised loop over the wrapped
    ///   handle and returns `Ok(())`: the handle honours the cadence, and a
    ///   [`Duration::ZERO`] interval stops that loop without cancelling a request;
    /// - anything else is the backend's own failure (a stopped child, say) and is
    ///   returned unchanged, with no supervised loop started.
    ///
    /// Nothing is recorded either way: a cadence change is a configuration choice,
    /// not a lifecycle transition.
    ///
    /// # Errors
    ///
    /// Whatever the inner handle reports other than
    /// [`openjiuwen_common::error::Error::Unsupported`].
    async fn set_health_interval(&self, interval: Duration) -> Result<()> {
        match self.inner.set_health_interval(interval).await {
            Ok(()) => {
                self.supervisor.stop().await;
                Ok(())
            }
            Err(error) if error.kind() == ErrorKind::Unsupported => {
                self.supervisor.retune(interval, self.watch()).await;
                Ok(())
            }
            Err(error) => Err(error),
        }
    }

    /// Forwards task cancellation to the inner handle.
    ///
    /// Task cancellation is not a lifecycle transition: the run keeps its status
    /// until the cancelled request actually ends (a terminal frame, a failed call,
    /// a dead child, or a shutdown). A backend with no cancellation keeps the
    /// trait's typed `Unsupported`, which is returned unchanged.
    ///
    /// # Errors
    ///
    /// Whatever the inner handle reports.
    async fn cancel_task(&self, task_id: &str) -> Result<()> {
        self.inner.cancel_task(task_id).await
    }

    /// Forwards a reason-carrying, id-correlated cancel to the inner handle.
    ///
    /// Forwarded with the reason so a backend whose wire carries it (the process
    /// backends' `cancel` frame, the HTTP client's `/cancel` body) delivers both;
    /// a backend that only knows [`SpawnedAgent::cancel_task`] gets the trait's
    /// default, which is that method with the same id.
    ///
    /// # Errors
    ///
    /// Whatever the inner handle reports.
    async fn cancel_task_with_request_id(
        &self,
        request_id: &str,
        reason: Option<&str>,
    ) -> Result<()> {
        self.inner
            .cancel_task_with_request_id(request_id, reason)
            .await
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
    use crate::monitor::RunEvent;
    use openjiuwen_common::error::{Error, Result};
    use openjiuwen_test_support::eventually;
    use std::sync::atomic::{AtomicBool, AtomicUsize};
    use std::sync::Mutex;

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
        /// Deadlines `send_with_timeout` received, in call order.
        send_timeouts: Arc<Mutex<Vec<Option<Duration>>>>,
        /// Deadlines `stream_with_timeout` received, in call order.
        stream_timeouts: Arc<Mutex<Vec<Option<Duration>>>>,
        /// Health intervals `set_health_interval` received, in call order.
        health_intervals: Arc<Mutex<Vec<Duration>>>,
        /// How many frames `stream`/`stream_with_timeout` yield.
        stream_frames: i64,
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
                send_timeouts: Arc::new(Mutex::new(Vec::new())),
                stream_timeouts: Arc::new(Mutex::new(Vec::new())),
                health_intervals: Arc::new(Mutex::new(Vec::new())),
                stream_frames: 0,
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
            let frames: Vec<Result<AgentChunk>> = (0..self.stream_frames)
                .map(|index| Ok(AgentChunk::llm_output(index, "tick")))
                .collect();
            Box::pin(futures::stream::iter(frames))
        }

        async fn send_with_timeout(
            &self,
            inputs: Value,
            timeout: Option<Duration>,
        ) -> Result<AgentOutput> {
            self.send_timeouts.lock().unwrap().push(timeout);
            self.send(inputs).await
        }

        fn stream_with_timeout(
            &self,
            inputs: Value,
            timeout: Option<Duration>,
        ) -> BoxStream<'static, Result<AgentChunk>> {
            self.stream_timeouts.lock().unwrap().push(timeout);
            self.stream(inputs)
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

        async fn set_health_interval(&self, interval: Duration) -> Result<()> {
            self.health_intervals.lock().unwrap().push(interval);
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
    async fn forwards_the_health_interval_re_tune_without_recording_a_transition() {
        let (handle, child, monitor) = tracked("child-15", true);
        let before = monitor.history(handle.run_id()).unwrap().len();

        handle.set_health_interval(Duration::ZERO).await.unwrap();
        handle
            .set_health_interval(Duration::from_millis(250))
            .await
            .unwrap();
        assert_eq!(
            child.health_intervals.lock().unwrap().clone(),
            vec![Duration::ZERO, Duration::from_millis(250)],
            "the decorator forwards every re-tune verbatim"
        );
        assert_eq!(
            monitor.history(handle.run_id()).unwrap().len(),
            before,
            "a cadence change is not a lifecycle transition"
        );
        assert!(!handle.is_finished(), "the run is still active");
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

    #[tokio::test]
    async fn forwards_the_bounded_invocation_deadlines_verbatim() {
        let (handle, child, _monitor) = tracked("child-14", true);

        handle
            .send_with_timeout(json!({}), Some(Duration::from_millis(25)))
            .await
            .unwrap();
        handle.send_with_timeout(json!({}), None).await.unwrap();
        assert_eq!(
            *child.send_timeouts.lock().unwrap(),
            vec![Some(Duration::from_millis(25)), None],
            "both deadlines reached the wrapped handle, so the decorator's own \
             overrides ran instead of the defaulting trait bodies"
        );

        let _: Vec<Result<AgentChunk>> = futures::StreamExt::collect(
            handle.stream_with_timeout(json!({}), Some(Duration::from_millis(25))),
        )
        .await;
        let _: Vec<Result<AgentChunk>> =
            futures::StreamExt::collect(handle.stream_with_timeout(json!({}), None)).await;
        assert_eq!(
            *child.stream_timeouts.lock().unwrap(),
            vec![Some(Duration::from_millis(25)), None],
            "the stream variant forwards its deadline too"
        );
    }

    #[tokio::test]
    async fn a_bounded_stream_forwards_frames_and_terminates_like_stream() {
        let monitor = Arc::new(RunMonitor::new());
        let child = Arc::new(FakeChild {
            stream_frames: 2,
            ..FakeChild::new("child-15", SpawnBackend::InProcess)
        });
        let handle = MonitoredSpawnedAgent::track(
            Arc::clone(&child) as Arc<dyn SpawnedAgent>,
            Arc::clone(&monitor),
            None,
            json!({"backend": "in_process"}),
        );

        let plain: Vec<Result<AgentChunk>> =
            futures::StreamExt::collect(handle.stream(json!({}))).await;
        let bounded: Vec<Result<AgentChunk>> = futures::StreamExt::collect(
            handle.stream_with_timeout(json!({}), Some(Duration::from_millis(25))),
        )
        .await;

        assert_eq!(plain.len(), 2);
        assert_eq!(
            bounded.len(),
            plain.len(),
            "the bounded stream terminates exactly like `stream`"
        );
        for (bounded, plain) in bounded.iter().zip(&plain) {
            assert_eq!(
                bounded.as_ref().unwrap().payload,
                plain.as_ref().unwrap().payload
            );
        }
        assert_eq!(
            monitor.snapshot(handle.run_id()).unwrap().phase,
            RunPhase::Running,
            "a bounded stream is not a terminal transition"
        );
    }

    #[tokio::test]
    async fn a_failed_bounded_send_from_a_dead_child_finishes_the_run() {
        let monitor = Arc::new(RunMonitor::new());
        let child = Arc::new(FakeChild {
            send_fails: true,
            ..FakeChild::new("child-16", SpawnBackend::InProcess)
        });
        let handle = MonitoredSpawnedAgent::track(
            Arc::clone(&child) as Arc<dyn SpawnedAgent>,
            Arc::clone(&monitor),
            None,
            json!({"backend": "in_process"}),
        );

        // A live child's failed bounded request — a timed-out exchange — only
        // observes.
        let error = handle
            .send_with_timeout(json!({}), Some(Duration::from_millis(25)))
            .await
            .unwrap_err();
        assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::Transport);
        assert_eq!(
            monitor.snapshot(handle.run_id()).unwrap().phase,
            RunPhase::Running,
            "a live child's failure only observes"
        );
        assert!(!handle.is_finished());

        child.alive.store(false, Ordering::Release);
        let error = handle.send_with_timeout(json!({}), None).await.unwrap_err();
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

    // --- the supervised health loop -----------------------------------------

    /// A child that inherits the default `set_health_interval`, so the decorator
    /// must supervise the cadence, and answers a configurable probe.
    ///
    /// `probe` of `None` is a probe the backend does not implement (the shape of
    /// the trait default), which the supervised tick must observe by liveness.
    struct TickChild {
        /// Stable handle id.
        id: String,
        /// Whether the handle reports itself alive.
        alive: AtomicBool,
        /// The probe answer, or `None` for "not implemented".
        probe: Option<bool>,
        /// How many probes the decorator ran.
        probes: AtomicUsize,
    }

    impl TickChild {
        /// An alive child under `id` answering `probe`.
        fn new(id: &str, probe: Option<bool>) -> Arc<Self> {
            Arc::new(Self {
                id: id.to_owned(),
                alive: AtomicBool::new(true),
                probe,
                probes: AtomicUsize::new(0),
            })
        }

        /// How many probes arrived.
        fn probes(&self) -> usize {
            self.probes.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl SpawnedAgent for TickChild {
        fn id(&self) -> &str {
            &self.id
        }

        fn is_alive(&self) -> bool {
            self.alive.load(Ordering::Acquire)
        }

        fn pid(&self) -> Option<u32> {
            Some(5151)
        }

        async fn send(&self, inputs: Value) -> Result<AgentOutput> {
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
            SpawnBackend::InProcess
        }

        async fn health(&self) -> Result<bool> {
            self.probes.fetch_add(1, Ordering::SeqCst);
            match self.probe {
                Some(healthy) => Ok(healthy),
                None => Err(crate::error::unsupported(
                    "agent health probe",
                    "the tick fake has no probe",
                )),
            }
        }
    }

    /// A tracked decorator over a [`TickChild`] and its monitor.
    fn tracked_tick(
        id: &str,
        probe: Option<bool>,
    ) -> (MonitoredSpawnedAgent, Arc<TickChild>, Arc<RunMonitor>) {
        let monitor = Arc::new(RunMonitor::new());
        let child = TickChild::new(id, probe);
        let handle = MonitoredSpawnedAgent::track(
            Arc::clone(&child) as Arc<dyn SpawnedAgent>,
            Arc::clone(&monitor),
            Some("s-1"),
            json!({"backend": "in_process"}),
        );
        (handle, child, monitor)
    }

    /// Collects up to `want` events, giving up at `within`.
    async fn observations(
        events: &mut tokio::sync::broadcast::Receiver<RunEvent>,
        want: usize,
        within: Duration,
    ) -> Vec<RunEvent> {
        let deadline = tokio::time::Instant::now() + within;
        let mut collected = Vec::new();
        while collected.len() < want {
            match tokio::time::timeout_at(deadline, events.recv()).await {
                Ok(Ok(event)) => collected.push(event),
                _ => break,
            }
        }
        collected
    }

    #[tokio::test]
    async fn a_backend_without_a_loop_is_supervised_at_the_requested_cadence() {
        let (handle, child, monitor) = tracked_tick("child-17", Some(true));
        let mut events = monitor.subscribe();

        handle
            .set_health_interval(Duration::from_millis(5))
            .await
            .expect("the decorator honours a cadence the backend refuses");
        assert!(
            !handle.is_finished(),
            "a cadence change is not a lifecycle transition"
        );

        let observed = observations(&mut events, 3, Duration::from_secs(2)).await;
        assert_eq!(observed.len(), 3, "the supervised loop must observe");
        for event in &observed {
            assert_eq!(event.run_id, "child-17");
            assert_eq!(event.phase, RunPhase::Running);
            assert_eq!(
                event.detail,
                json!({
                    "backend": "in_process",
                    "alive": true,
                    "healthy": true,
                    "pid": 5151,
                }),
                "a supervised tick records the probe's own observation"
            );
        }
        assert!(
            child.probes() >= 3,
            "each tick calls the inner probe: {} probes",
            child.probes()
        );
        assert_eq!(
            monitor.snapshot(handle.run_id()).unwrap().detail,
            observed.last().unwrap().detail,
            "the status follows the last observation"
        );
        handle.shutdown(Duration::from_secs(1)).await.unwrap();
    }

    #[tokio::test]
    async fn a_re_tune_reschedules_the_supervised_loop_and_zero_stops_it() {
        let (handle, child, monitor) = tracked_tick("child-18", Some(true));
        let mut events = monitor.subscribe();

        handle
            .set_health_interval(Duration::from_millis(5))
            .await
            .unwrap();
        assert_eq!(
            observations(&mut events, 2, Duration::from_secs(2))
                .await
                .len(),
            2,
            "the first cadence observes"
        );

        // A second cadence replaces the loop; it does not add a second one.
        handle
            .set_health_interval(Duration::from_millis(20))
            .await
            .unwrap();
        assert_eq!(
            observations(&mut events, 2, Duration::from_secs(2))
                .await
                .len(),
            2,
            "the re-tuned loop keeps observing"
        );

        handle.set_health_interval(Duration::ZERO).await.unwrap();
        while events.try_recv().is_ok() {}
        let probes = child.probes();
        assert!(
            tokio::time::timeout(Duration::from_millis(120), events.recv())
                .await
                .is_err(),
            "a zero interval stops the supervised loop"
        );
        assert_eq!(
            child.probes(),
            probes,
            "nothing probes once the cadence is stopped"
        );
        assert!(!handle.is_finished(), "stopping a cadence is not terminal");
        handle.shutdown(Duration::from_secs(1)).await.unwrap();
    }

    #[tokio::test]
    async fn a_supervised_tick_escalates_a_dead_child_to_failed() {
        let (handle, child, monitor) = tracked_tick("child-19", Some(true));
        let mut events = monitor.subscribe();
        handle
            .set_health_interval(Duration::from_millis(5))
            .await
            .unwrap();
        assert_eq!(
            observations(&mut events, 1, Duration::from_secs(2))
                .await
                .len(),
            1,
            "the loop is running before the child is killed"
        );

        child.alive.store(false, Ordering::Release);
        eventually(Duration::from_secs(2), || {
            let status = monitor.snapshot("child-19");
            std::future::ready(matches!(status, Ok(status) if status.phase == RunPhase::Failed))
        })
        .await
        .expect("the supervised loop must finish a dead child");

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
    async fn a_supervised_tick_of_an_unimplemented_probe_observes_liveness_only() {
        let (handle, _child, monitor) = tracked_tick("child-20", None);
        let mut events = monitor.subscribe();
        handle
            .set_health_interval(Duration::from_millis(5))
            .await
            .unwrap();

        let observed = observations(&mut events, 2, Duration::from_secs(2)).await;
        assert_eq!(observed.len(), 2);
        let detail = &observed[0].detail;
        assert_eq!(detail["probe"], json!("liveness_only"));
        assert_eq!(detail["backend"], json!("in_process"));
        assert_eq!(detail["alive"], json!(true));
        assert_eq!(
            detail["healthy"],
            json!(true),
            "the default health answer follows liveness"
        );
        assert_eq!(detail["pid"], json!(5151));
        assert!(
            detail.get("error").is_none(),
            "a missing probe is not a fabricated failure: {detail:?}"
        );
        handle.shutdown(Duration::from_secs(1)).await.unwrap();
    }

    #[tokio::test]
    async fn shutdown_and_drop_stop_the_supervised_loop() {
        let (handle, _child, monitor) = tracked_tick("child-21", Some(true));
        let mut events = monitor.subscribe();
        handle
            .set_health_interval(Duration::from_millis(5))
            .await
            .unwrap();
        assert_eq!(
            observations(&mut events, 2, Duration::from_secs(2))
                .await
                .len(),
            2
        );
        handle.shutdown(Duration::from_secs(1)).await.unwrap();
        while events.try_recv().is_ok() {}
        assert!(
            tokio::time::timeout(Duration::from_millis(120), events.recv())
                .await
                .is_err(),
            "a shutdown joins the supervised loop"
        );

        // A dropped handle takes its loop with it.
        let (handle, _child, monitor) = tracked_tick("child-22", Some(true));
        let mut events = monitor.subscribe();
        handle
            .set_health_interval(Duration::from_millis(5))
            .await
            .unwrap();
        assert_eq!(
            observations(&mut events, 2, Duration::from_secs(2))
                .await
                .len(),
            2
        );
        drop(handle);
        let terminal = tokio::time::timeout(Duration::from_secs(1), events.recv())
            .await
            .expect("the drop reports its terminal transition")
            .expect("an event");
        assert_eq!(terminal.phase, RunPhase::Completed);
        assert!(
            tokio::time::timeout(Duration::from_millis(120), events.recv())
                .await
                .is_err(),
            "a dropped handle stops observing"
        );
    }
}
