//! The team runtime pool's lifecycle: `stop`, `stop_all`, the concurrency bound
//! with queued admission, and the typed `pause`/`resume` refusals.
//!
//! Every case runs a real in-process handoff team whose single member blocks on a
//! gate nobody opens, so a "live run" is genuinely live (it holds the key, the
//! permit, the monitor entry, and the HITL binding) and cancellation has to come
//! from the pool.
//!
//! Ports `team_runner.py::_TeamRunnerMixin.stop_agent_team` /
//! `pause_agent_team` (kept as a typed refusal: an in-process strategy has no
//! mid-flight suspension seam) / `_maybe_release_team_session`, the
//! `TeamRuntimeManager.activate` admission guard (Python's pool bound), and
//! `TeamRuntimeManager.finalize` (the exactly-once teardown).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::sync::Arc;
use std::time::Duration;

use openjiuwen_agent::Agent;
use openjiuwen_common::error::ErrorKind;
use openjiuwen_multiagent::{handoff_team, BaseTeam, HandoffTeamConfig, Team};
use openjiuwen_runner::config::TeamPoolConfig;
use openjiuwen_runner::hitl::InteractionHub;
use openjiuwen_runner::monitor::{RunMonitor, RunPhase};
use openjiuwen_runner::pool::{PoolError, PoolFrame, PoolKey, PoolRequest, TeamPool};
use openjiuwen_runner::run::TeamRef;
use serde_json::json;
use tokio::sync::Notify;

use common::{
    agent_card, handoff_team_fixture, provider, team_calls, team_card, ScriptedAgent,
    HANDOFF_SECOND_ANSWER,
};

/// Upper bound on a deterministic in-memory wait (a deadline, never an ordering
/// device).
const BOUND: Duration = Duration::from_secs(10);

/// Polls `check` until it reports `true`, bounded by [`BOUND`].
async fn until(mut check: impl FnMut() -> bool) {
    tokio::time::timeout(BOUND, async {
        while !check() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("the condition is reached before the deadline");
}

/// A one-member handoff team whose only member blocks on the returned gate.
fn blocked_team(team_id: &str) -> (Arc<BaseTeam>, Arc<Notify>) {
    let gate = Arc::new(Notify::new());
    let member: Arc<dyn Agent> = Arc::new(
        ScriptedAgent::new("worker")
            .with_answer("released")
            .blocking_on(Arc::clone(&gate)),
    );
    let team = handoff_team(team_card(team_id), HandoffTeamConfig::default());
    team.add_agent(agent_card("worker"), provider(member))
        .expect("the only member registers");
    (Arc::new(team), gate)
}

/// A two-member handoff team that answers as soon as it is admitted.
fn quick_team(team_id: &str) -> Arc<BaseTeam> {
    handoff_team_fixture(team_id, &team_calls())
}

/// The request a suite sends for `team` in session `session`.
fn request(team: &Arc<BaseTeam>, session: &str) -> PoolRequest {
    PoolRequest::new(
        TeamRef::of(Arc::clone(team) as Arc<dyn Team>),
        json!({"query": "hi", "conversation_id": session}),
    )
}

// ---------------------------------------------------------------------------
// stop
// ---------------------------------------------------------------------------

// ports team_runner.py::_TeamRunnerMixin.stop_agent_team
#[tokio::test]
async fn stop_cancels_a_blocked_run_and_releases_every_seam() {
    let hub = Arc::new(InteractionHub::new());
    let monitor = Arc::new(RunMonitor::new());
    let services = common::services()
        .with_monitor(Arc::clone(&monitor))
        .with_hitl(Arc::clone(&hub));
    let (team, _gate) = blocked_team("stop-crew");

    let pool = TeamPool::new();
    let run = pool
        .start(services, request(&team, "stop-session"))
        .await
        .unwrap();
    let key = run.key().clone();
    let run_id = run.run_id().to_owned();
    let mut frames = run.subscribe();

    // The worker is admitted, so the run owns both observation seams.
    until(|| run.state() == RunPhase::Running).await;
    assert_eq!(hub.list_active().len(), 1, "the hub knows the live run");
    assert_eq!(hub.list_active()[0].run_id, run_id);

    assert!(pool.stop(&key).await.unwrap(), "the run was live");
    assert_eq!(run.state(), RunPhase::Cancelled);
    assert!(run.is_finished());
    assert_eq!(
        run.wait().await.unwrap_err().kind(),
        ErrorKind::Cancelled,
        "a cancelled run reports cancellation from `wait`"
    );

    // Frames: the announcement, then exactly one terminal cancellation.
    let published: Vec<PoolFrame> = std::iter::from_fn(|| frames.try_recv().ok()).collect();
    assert!(
        matches!(published.first(), Some(PoolFrame::Started { .. })),
        "the run announced itself: {published:?}"
    );
    assert_eq!(
        published.last(),
        Some(&PoolFrame::Cancelled {
            reason: "stopped".to_owned()
        }),
        "`stop` records its own reason: {published:?}"
    );

    // The monitor saw the terminal phase, and the run left the active table.
    let status = monitor.snapshot(&run_id).expect("the status is retained");
    assert_eq!(status.phase, RunPhase::Cancelled);
    assert_eq!(monitor.list_active(), Vec::new(), "no live run is listed");

    // The hub binding is gone, the entry is removed, and the permit is back.
    assert!(
        hub.is_empty(),
        "a finalised run is unregistered from the hub"
    );
    assert!(pool.get(&key).is_none(), "`stop` removes the entry");
    assert_eq!(pool.list_active(), Vec::new(), "no live run is listed");
    assert_eq!(pool.available_permits(), pool.capacity());

    // An unknown key is `false`, not an error, so a host can call it blindly.
    assert!(!pool.stop(&PoolKey::new("ghost", "nowhere")).await.unwrap());
    assert!(pool.stop_all().await.is_ok());
}

// ports team_runner.py::TeamRuntimeManager.finalize + pool shutdown
#[tokio::test]
async fn stop_all_sweeps_every_run_and_closes_the_pool() {
    let pool = TeamPool::new();
    let (team_a, _gate_a) = blocked_team("sweep-a");
    let (team_b, _gate_b) = blocked_team("sweep-b");
    let run_a = pool
        .start(common::services(), request(&team_a, "sweep-session-a"))
        .await
        .unwrap();
    let run_b = pool
        .start(common::services(), request(&team_b, "sweep-session-b"))
        .await
        .unwrap();
    assert_eq!(pool.list_active().len(), 2);
    assert_eq!(pool.available_permits(), pool.capacity() - 2);

    let shutdown = pool.stop_all().await;
    assert!(shutdown.is_ok(), "the sweep reported no failure");
    assert_eq!(shutdown.stopped, 2, "both live runs were stopped");
    assert_eq!(shutdown.into_result().unwrap(), 2);
    for run in [&run_a, &run_b] {
        assert_eq!(run.state(), RunPhase::Cancelled);
        assert!(run.is_finished(), "every run is terminal");
    }

    assert!(pool.is_closed());
    assert_eq!(pool.list_active(), Vec::new(), "no live run is listed");
    assert_eq!(pool.len(), 0);
    assert_eq!(pool.len(), 0, "the pool retains nothing");
    assert_eq!(pool.available_permits(), pool.capacity());

    // A closed pool refuses every admission, and none of them creates an entry.
    let error = pool
        .start(common::services(), request(&team_a, "after-close"))
        .await
        .expect_err("a closed pool admits nothing");
    assert_eq!(error.kind(), ErrorKind::Config);
    assert_eq!(error.to_string(), "config: team pool is closed");
    assert_eq!(error.message(), PoolError::Closed.to_string());
    let error = pool
        .try_start(common::services(), request(&team_a, "after-close"))
        .expect_err("a closed pool admits nothing");
    assert_eq!(error.to_string(), "config: team pool is closed");
    let error = pool
        .acquire(common::services(), request(&team_a, "after-close"))
        .await
        .expect_err("a closed pool admits nothing");
    assert_eq!(error.kind(), ErrorKind::Config);
    assert_eq!(pool.len(), 0);

    // No worker outlives the sweep: after a few yield rounds nothing reappears.
    for _ in 0..8 {
        tokio::task::yield_now().await;
    }
    assert_eq!(pool.list_active(), Vec::new(), "no live run is listed");
    assert_eq!(pool.len(), 0);
}

// ---------------------------------------------------------------------------
// The concurrency bound
// ---------------------------------------------------------------------------

// ports team_runner.py's pool bound (max active runtimes) and its queued wait
#[tokio::test]
async fn a_start_beyond_the_capacity_queues_until_a_permit_frees() {
    let pool = TeamPool::with_config(TeamPoolConfig::default().with_max_concurrent_runs(1));
    assert_eq!(pool.capacity(), 1, "the configured bound is effective");

    let (running_team, _gate) = blocked_team("cap-running");
    let run_a = pool
        .start(common::services(), request(&running_team, "cap-a"))
        .await
        .unwrap();
    assert_eq!(pool.available_permits(), 0, "the only permit is in flight");

    // `try_start` reports the bound instead of queueing, and creates no entry.
    let queued_team = quick_team("cap-queued");
    let error = pool
        .try_start(common::services(), request(&queued_team, "cap-b"))
        .expect_err("the pool is full");
    assert_eq!(error.kind(), ErrorKind::Config);
    assert_eq!(
        error.to_string(),
        "config: team pool is at capacity (1 concurrent runs)"
    );
    assert_eq!(
        error.message(),
        PoolError::AtCapacity { capacity: 1 }.to_string(),
        "the rejection is the `AtCapacity` variant with the effective bound"
    );
    assert_eq!(pool.len(), 1, "a rejected `try_start` leaves no entry");

    // `start` waits for a permit instead.
    let task_pool = pool.clone();
    let task_team = Arc::clone(&queued_team);
    let queued = tokio::spawn(async move {
        task_pool
            .start(common::services(), request(&task_team, "cap-b"))
            .await
            .expect("the queued admission is admitted")
    });

    // The queued entry is observable while it waits: still queued, no permit.
    until(|| {
        pool.list_active()
            .iter()
            .any(|info| info.session_id == "cap-b")
    })
    .await;
    let mut active = pool.list_active();
    assert_eq!(active.len(), 2, "the queue is listed next to the live run");
    let waiting = active
        .iter()
        .find(|info| info.session_id == "cap-b")
        .expect("the queued run is listed");
    assert_eq!(waiting.state, RunPhase::Queued);
    assert!(!waiting.running, "a queued run holds no permit");
    active.retain(|info| info.session_id == "cap-a");
    assert_eq!(active[0].run_id, run_a.run_id());
    assert!(active[0].running, "the live run holds the permit");

    // Cancelling the live run releases the permit and admits the queued run.
    run_a.cancel();
    let run_b = queued.await.expect("the admission task does not panic");
    let output = run_b.wait().await.unwrap();
    assert_eq!(
        output["payload"]["output"],
        json!(HANDOFF_SECOND_ANSWER),
        "the queued run then executes normally: {output}"
    );
    assert_eq!(run_b.state(), RunPhase::Completed);
    assert_eq!(run_b.session_id(), "cap-b");
    assert_eq!(pool.available_permits(), 1, "the permit came back");
    assert_eq!(run_a.state(), RunPhase::Cancelled);

    let _ = pool.stop_all().await;
}

// ---------------------------------------------------------------------------
// pause / resume refusals
// ---------------------------------------------------------------------------

// ports team_runner.py::_TeamRunnerMixin.pause_agent_team (a typed refusal)
#[tokio::test]
async fn pause_and_resume_refuse_a_live_key_and_reject_unknown_ones() {
    let pool = TeamPool::new();
    let (team, _gate) = blocked_team("pause-crew");
    let run = pool
        .start(common::services(), request(&team, "pause-session"))
        .await
        .unwrap();
    let key = run.key().clone();
    let mut frames = run.subscribe();
    until(|| run.state() == RunPhase::Running).await;

    for (action, result) in [("pause", pool.pause(&key)), ("resume", pool.resume(&key))] {
        let error = result.expect_err("a live pooled run cannot be suspended");
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        let message = error.to_string();
        assert!(
            message.contains(&format!("`{action}`"))
                && message.contains(run.run_id())
                && message.contains("InteractGate")
                && message.contains("cancel"),
            "the refusal names the call, the run, the missing seam, and the              alternative: {message}"
        );
    }

    // The refusal changed nothing: same phase, open gate, permit and entry kept.
    let live = pool.get(&key).expect("the entry stays live");
    assert_eq!(live.state, RunPhase::Running);
    assert!(!live.gate_closed, "the refusal left the gate open");
    assert!(live.running, "the refusal released no permit");
    assert_eq!(run.state(), RunPhase::Running);
    assert_eq!(pool.list_active().len(), 1);

    // Nothing is published for a refused pause/resume.
    let published: Vec<PoolFrame> = std::iter::from_fn(|| frames.try_recv().ok()).collect();
    assert!(
        !published
            .iter()
            .any(|frame| matches!(frame, PoolFrame::Output { .. })),
        "the blocked team published no frame yet: {published:?}"
    );

    // An unknown key is a not-found error on both verbs.
    let ghost = PoolKey::new("ghost", "nowhere");
    let error = pool.pause(&ghost).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert_eq!(
        error.to_string(),
        "not_found: no active pooled team run for 'ghost' / session 'nowhere'"
    );
    assert_eq!(error.message(), PoolError::not_found(&ghost).to_string());
    assert_eq!(pool.resume(&ghost).unwrap_err().kind(), ErrorKind::NotFound);

    // A refused pause leaves the run stoppable: the pool still cancels it.
    assert_eq!(pool.stop_all().await.stopped, 1);
    assert_eq!(run.state(), RunPhase::Cancelled);
}

// ports team_runner.py::_TeamRunnerMixin.pause_agent_team (the leased surface)
#[tokio::test]
async fn a_lease_refusal_keeps_the_run_and_its_lease_intact() {
    let pool = TeamPool::new();
    let (team, _gate) = blocked_team("lease-pause-crew");
    let lease = pool
        .acquire(common::services(), request(&team, "lease-pause-session"))
        .await
        .unwrap();
    until(|| lease.state() == RunPhase::Running).await;

    for error in [
        lease.pause().expect_err("a live lease cannot be paused"),
        lease.resume().expect_err("a live lease cannot be resumed"),
    ] {
        assert_eq!(error.kind(), ErrorKind::Unsupported);
    }
    assert_eq!(lease.state(), RunPhase::Running);
    let live = pool.get(lease.key()).expect("the entry is retained");
    assert!(!live.gate_closed, "the refusal left the gate open");
    assert_eq!(live.leases, 1, "the lease is still held");
    assert!(!lease.is_released());

    // The leased run is still stoppable through the pool.
    assert!(pool.stop(lease.key()).await.unwrap());
    assert_eq!(lease.state(), RunPhase::Cancelled);
    assert_eq!(
        pool.stop_all().await.stopped,
        0,
        "the run was already swept"
    );
}
