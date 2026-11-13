//! The team runtime pool's happy path: frames, `wait`, the active listing, the
//! busy rejection, the monitor walk, and the configuration wire shape.
//!
//! Every case drives only the public `openjiuwen_runner::pool` surface over a
//! real `BaseTeam`/handoff team (`common::handoff_team_fixture`), so the pool is
//! proven against the multiagent seam rather than a stub.
//!
//! Ports `team_runner.py::TeamRuntimeManager.activate` / `finalize` (the run
//! cycle and `_TEAM_REJECT_KINDS`' `REJECT_RUNNING`), `_TeamRunnerMixin.list_active_teams`,
//! and `TeamRuntimePool` (`config.py`'s pool bounds).
//!
//! **Why `#[tokio::test]`'s current-thread runtime matters here.** `start`
//! returns after `launch` spawns the worker but *before* the worker's first
//! poll, so subscribing to a run's frames right after `start` deterministically
//! observes `Started` instead of racing it. A multi-thread runtime would drop
//! the frames published before the subscription (the channel is best effort).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::sync::Arc;
use std::time::Duration;

use openjiuwen_agent::Agent;
use openjiuwen_common::error::ErrorKind;
use openjiuwen_multiagent::{handoff_team, BaseTeam, HandoffTeamConfig, Team};
use openjiuwen_runner::config::{RunnerConfig, TeamPoolConfig};
use openjiuwen_runner::monitor::{RunEvent, RunMonitor, RunPhase};
use openjiuwen_runner::pool::{
    ActiveTeamInfo, PoolError, PoolFrame, PoolKey, PoolRequest, TeamPool,
};
use openjiuwen_runner::run::TeamRef;
use serde_json::json;
use tokio::sync::broadcast;
use tokio::sync::Notify;

use common::{
    agent_card, handoff_team_fixture, provider, team_calls, team_card, ScriptedAgent,
    HANDOFF_SECOND_ANSWER,
};

/// Upper bound on a deterministic in-memory wait (a deadline, never an ordering
/// device).
const BOUND: Duration = Duration::from_secs(10);

/// Collects frames until the run's terminal one arrives, bounded by [`BOUND`].
async fn frames_until_terminal(receiver: &mut broadcast::Receiver<PoolFrame>) -> Vec<PoolFrame> {
    let mut frames = Vec::new();
    loop {
        let frame = tokio::time::timeout(BOUND, receiver.recv())
            .await
            .expect("the run publishes its frames before the deadline")
            .expect("the subscriber keeps up with the run");
        let terminal = matches!(
            frame,
            PoolFrame::Completed { .. } | PoolFrame::Failed { .. } | PoolFrame::Cancelled { .. }
        );
        frames.push(frame);
        if terminal {
            return frames;
        }
    }
}

/// Builds the `(team, calls)` fixture under `team_id`.
fn fixture(team_id: &str) -> (Arc<BaseTeam>, Arc<common::TeamCalls>) {
    let calls = team_calls();
    let team = handoff_team_fixture(team_id, &calls);
    (team, calls)
}

/// The request a suite sends for `team` in session `session`.
fn request(team: &Arc<BaseTeam>, session: &str) -> PoolRequest {
    PoolRequest::new(
        TeamRef::of(Arc::clone(team) as Arc<dyn Team>),
        json!({"query": "hi", "conversation_id": session}),
    )
}

/// A one-member handoff team whose only member blocks on the returned gate.
///
/// Nobody opens the gate in this suite: the point is a run that stays live (and
/// therefore keeps its key and its permit) until the pool cancels it. The member
/// is a `common::ScriptedAgent`, so the fixture stays offline and in-process.
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

/// The phases the monitor published for `run_id`, oldest first.
fn published_phases(receiver: &mut broadcast::Receiver<RunEvent>, run_id: &str) -> Vec<RunPhase> {
    std::iter::from_fn(|| receiver.try_recv().ok())
        .filter(|event| event.run_id == run_id)
        .map(|event| event.phase)
        .collect()
}

/// The single active snapshot of `pool`, for the one-run cases.
fn only_active(pool: &TeamPool) -> ActiveTeamInfo {
    let mut active = pool.list_active();
    assert_eq!(active.len(), 1, "exactly one live run: {active:?}");
    active.remove(0)
}

// ---------------------------------------------------------------------------
// The run cycle
// ---------------------------------------------------------------------------

// ports team_runner.py::TeamRuntimeManager.activate / finalize
#[tokio::test]
async fn a_pooled_run_publishes_started_output_and_completed() {
    let (team, calls) = fixture("pooled-crew");
    let pool = TeamPool::new();
    let run = pool
        .start(common::services(), request(&team, "pool-session"))
        .await
        .unwrap();

    // The key is the team's card id plus the resolved session, and the handle
    // reports it back so a host can address the run without re-deriving it.
    assert_eq!(run.key(), &PoolKey::new("pooled-crew", "pool-session"));
    assert_eq!(run.team_name(), "pooled-crew");
    assert_eq!(run.session_id(), "pool-session");
    assert!(!run.is_finished(), "the worker has not been polled yet");

    let mut frames = run.subscribe();
    let output = run.wait().await.unwrap();

    let frames = frames_until_terminal(&mut frames).await;
    assert!(
        matches!(&frames[0], PoolFrame::Started { run_id, team_name, session_id }
            if run_id == run.run_id() && team_name == "pooled-crew" && session_id == "pool-session"),
        "the first frame announces the run: {frames:?}"
    );
    assert_eq!(
        frames
            .iter()
            .filter(|frame| matches!(frame, PoolFrame::Output { .. }))
            .count(),
        2,
        "one `Output` frame per completed handoff hop: {frames:?}"
    );
    assert_eq!(
        frames.last(),
        Some(&PoolFrame::Completed {
            output: output.clone()
        }),
        "the terminal frame carries the same output `wait` returned"
    );
    assert!(
        frames
            .iter()
            .all(|frame| !matches!(frame, PoolFrame::Failed { .. })),
        "a working team never fails: {frames:?}"
    );

    // `wait` yields the last raw `stream_team` frame: the transport envelope,
    // not `invoke_team`'s unwrapped answer.
    assert_eq!(output["type"], json!("message"), "frame: {output}");
    assert_eq!(
        output["payload"]["output"],
        json!(HANDOFF_SECOND_ANSWER),
        "the last hop's answer is the envelope payload"
    );
    assert_eq!(run.state(), RunPhase::Completed);
    assert_eq!(run.output(), Some(output.clone()));
    assert!(run.is_finished());
    assert_eq!(calls.inputs().len(), 2, "both members ran once");

    // The permit is back, and the terminal entry is retained for late observers.
    assert_eq!(pool.available_permits(), pool.capacity());
    let retained = pool
        .get(&run.key().clone())
        .expect("the terminal entry stays");
    assert_eq!(retained.state, RunPhase::Completed);
    assert!(!retained.running, "a finished run holds no permit");
    assert_eq!(retained.run_id, run.run_id());

    // `stop_all` sweeps every *retained* entry, so this finished one is removed
    // from the pool; it was already terminal when the sweep began, so it is not
    // counted as a stopped run (`stopped` reports live runs stopped).
    let shutdown = pool.stop_all().await;
    assert!(shutdown.is_ok(), "no sweep failure on a finished run");
    assert_eq!(shutdown.stopped, 0, "a finished entry is not a stopped run");
    assert_eq!(pool.len(), 0, "the sweep emptied the pool");
}

// ports team_runner.py::_TeamRunnerMixin.list_active_teams
#[tokio::test]
async fn list_active_reports_the_live_run_and_not_the_finished_one() {
    let (team, _calls) = fixture("listed-crew");
    let pool = TeamPool::new();
    let run = pool
        .start(common::services(), request(&team, "listed-session"))
        .await
        .unwrap();

    let live = only_active(&pool);
    assert_eq!(live.team_name, "listed-crew");
    assert_eq!(live.session_id, "listed-session");
    assert_eq!(live.run_id, run.run_id());
    assert_eq!(
        live.label, "listed-crew",
        "the label defaults to the team name"
    );
    assert!(live.running, "the admission took a permit before returning");
    assert_eq!(live.leases, 0, "`start` takes no lease");
    assert_eq!(live.queued, 0, "no human payload was buffered");
    assert_eq!(
        live.state,
        RunPhase::Queued,
        "the entry is created queued and the worker runs after this point"
    );

    run.wait().await.unwrap();

    assert_eq!(
        pool.list_active(),
        Vec::new(),
        "a terminal run is not active"
    );
    let retained = pool.get(&run.key().clone()).expect("the entry is retained");
    assert_eq!(retained.state, RunPhase::Completed);
    assert_eq!(pool.len(), 1, "one retained entry, no live one");
    // `stop_all` closes the pool, so it runs after the assertions above.
    let _ = pool.stop_all().await;
}

// ports team_runner.py::_TEAM_REJECT_KINDS (REJECT_RUNNING)
#[tokio::test]
async fn a_second_admission_on_a_live_key_is_busy() {
    // The member blocks on a gate nobody opens, so the key stays live for the
    // whole case.
    let (team, _gate) = blocked_team("busy-crew");
    let pool = TeamPool::new();
    let run = pool
        .start(common::services(), request(&team, "busy-session"))
        .await
        .unwrap();
    let key = run.key().clone();

    // `start` rejects: a live key admits no second run (Python's
    // REJECT_RUNNING), while `acquire` is the joining admission.
    let error = pool
        .start(common::services(), request(&team, "busy-session"))
        .await
        .expect_err("a live key is busy");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(
        error.to_string(),
        "validation: team run for 'busy-crew' / session 'busy-session' is already active"
    );
    assert_eq!(
        error.message(),
        PoolError::busy(&key).to_string(),
        "the rejection is the `Busy` variant, not a bare validation error"
    );

    let error = pool
        .try_start(common::services(), request(&team, "busy-session"))
        .expect_err("a live key is busy");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("already active"), "{error}");

    // The rejection left the live run untouched and created no second entry.
    assert_eq!(pool.len(), 1);
    assert_eq!(pool.list_active().len(), 1);
    assert_eq!(only_active(&pool).run_id, run.run_id());
    assert!(!run.is_finished(), "the live run is still live");
    assert!(pool.get(&key).is_some());

    assert_eq!(
        pool.stop_all().await.stopped,
        1,
        "the blocked run was stopped"
    );
}

// ---------------------------------------------------------------------------
// The monitor walk
// ---------------------------------------------------------------------------

// ports team_runner.py's per-run status record plus the pool's own admission
#[tokio::test]
async fn the_pool_walks_the_monitor_queued_running_completed() {
    let (team, _calls) = fixture("observed-crew");
    let monitor = Arc::new(RunMonitor::new());
    let services = common::services().with_monitor(Arc::clone(&monitor));
    let mut events = monitor.subscribe();

    let pool = TeamPool::new();
    let run = pool
        .start(services, request(&team, "observed-session"))
        .await
        .unwrap();
    let run_id = run.run_id().to_owned();

    // The entry is admitted on the monitor while still queued, so a dashboard
    // sees the run before its first frame.
    let active = monitor.list_active();
    assert_eq!(active.len(), 1, "the pooled run is listed: {active:?}");
    assert_eq!(active[0].run_id, run_id);
    assert_eq!(active[0].phase, RunPhase::Queued);
    assert_eq!(active[0].label, "observed-crew");
    assert_eq!(active[0].session_id.as_deref(), Some("observed-session"));

    run.wait().await.unwrap();

    assert_eq!(
        published_phases(&mut events, &run_id),
        vec![RunPhase::Queued, RunPhase::Running, RunPhase::Completed],
        "the pool drives the run's own id through the whole walk"
    );
    assert_eq!(
        monitor.list_active(),
        Vec::new(),
        "finalisation removes the run from the active table"
    );
    let status = monitor.snapshot(&run_id).expect("the status is retained");
    assert_eq!(status.phase, RunPhase::Completed);
    assert!(
        status.finished_at.is_some(),
        "the terminal event is stamped"
    );

    let _ = pool.stop_all().await;
}

// ---------------------------------------------------------------------------
// Configuration wire shape
// ---------------------------------------------------------------------------

// ports config.py's TeamRuntimeManager pool bounds (max active runtimes, orphan
// reclaim, per-run gate capacity)
#[test]
fn the_team_pool_config_round_trips_as_snake_case_json() {
    let pool_config = TeamPoolConfig {
        max_concurrent_runs: 3,
        idle_timeout_secs: 12.5,
        run_timeout_secs: Some(7.5),
        payload_capacity: 9,
        retain_finished: 4,
        frame_capacity: 5,
        reaper_interval_secs: 1.5,
        stop_timeout_secs: 0.5,
    };
    let config = RunnerConfig::default().with_team_pool(pool_config.clone());
    let value = serde_json::to_value(&config).expect("serializable");

    // The block is nested under `team_pool` and every field keeps its wire name.
    let block = value["team_pool"].as_object().expect("a team_pool object");
    let mut keys: Vec<&str> = block.keys().map(String::as_str).collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        vec![
            "frame_capacity",
            "idle_timeout_secs",
            "max_concurrent_runs",
            "payload_capacity",
            "reaper_interval_secs",
            "retain_finished",
            "run_timeout_secs",
            "stop_timeout_secs",
        ]
    );
    assert_eq!(block["max_concurrent_runs"], json!(3));
    assert_eq!(block["idle_timeout_secs"], json!(12.5));
    assert_eq!(block["run_timeout_secs"], json!(7.5));

    let back: RunnerConfig = serde_json::from_value(value).expect("round trip");
    assert_eq!(back.team_pool, pool_config);
    assert_eq!(back.team_pool, config.team_pool);
}

// ports config.py: the pool bounds are module constants, so a host that names
// only one of them still gets the rest.
#[test]
fn a_partial_team_pool_object_fills_the_rest_from_defaults() {
    let config: RunnerConfig = serde_json::from_value(json!({
        "team_pool": {"max_concurrent_runs": 2}
    }))
    .expect("a partial block deserializes");
    let defaults = TeamPoolConfig::default();
    assert_eq!(config.team_pool.max_concurrent_runs, 2);
    assert_eq!(
        config.team_pool.idle_timeout_secs,
        defaults.idle_timeout_secs
    );
    assert_eq!(config.team_pool.payload_capacity, defaults.payload_capacity);
    assert_eq!(config.team_pool.frame_capacity, defaults.frame_capacity);
    assert_eq!(config.team_pool.retain_finished, defaults.retain_finished);
    assert_eq!(
        config.team_pool.stop_timeout_secs,
        defaults.stop_timeout_secs
    );
    assert_eq!(config.team_pool.run_timeout_secs, None);

    // `from_runner_config` is the documented bridge from the block to a pool.
    let pool = TeamPool::from_runner_config(&config);
    assert_eq!(pool.capacity(), 2);
    assert_eq!(pool.config(), config.team_pool);
    assert!(matches!(
        TeamPool::from_runner_config(&RunnerConfig::default()).capacity(),
        8
    ));
    // A value the wire omitted keeps the block tolerant of unknown fields too.
    let tolerant: RunnerConfig =
        serde_json::from_value(json!({"team_pool": {"stop_timeout_secs": 0.25, "x": [1]}}))
            .expect("unknown fields are tolerated");
    assert!((tolerant.team_pool.stop_timeout_secs - 0.25).abs() < f64::EPSILON);
}
