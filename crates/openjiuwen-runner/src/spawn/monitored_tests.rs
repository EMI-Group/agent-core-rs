#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Unit tests of the monitored spawn decorator: the `RunKind::Spawn` lifecycle
//! it reports into a `RunMonitor`, the dead-child terminal rule, the bounded
//! invocation deadlines it forwards, and the health cadence it supervises when
//! the wrapped backend cannot.

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

fn tracked(id: &str, probe_ok: bool) -> (MonitoredSpawnedAgent, Arc<FakeChild>, Arc<RunMonitor>) {
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
fn dropping_an_unfinished_handle_finishes_it_as_cancelled() {
    let monitor = Arc::new(RunMonitor::new());
    let child: Arc<dyn SpawnedAgent> = Arc::new(FakeChild::new("child-8", SpawnBackend::InProcess));
    let handle = MonitoredSpawnedAgent::track(
        child,
        Arc::clone(&monitor),
        None,
        json!({"backend": "in_process"}),
    );
    let run_id = handle.run_id().to_owned();
    drop(handle);
    let status = monitor.snapshot(&run_id).unwrap();
    assert_eq!(
        status.phase,
        RunPhase::Cancelled,
        "an abandoned handle is not a successful run"
    );
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
    assert_eq!(
        terminal.phase,
        RunPhase::Cancelled,
        "an abandoned handle is not a successful run"
    );
    assert!(
        tokio::time::timeout(Duration::from_millis(120), events.recv())
            .await
            .is_err(),
        "a dropped handle stops observing"
    );
}
