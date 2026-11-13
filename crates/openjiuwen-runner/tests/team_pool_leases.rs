//! The team runtime pool's leases and idle eviction.
//!
//! A lease is the pool's "somebody is still using this run" marker: while one
//! exists the reaper skips the entry, and once the last lease is released the
//! entry goes idle, after which a finished entry is evicted and a still-live
//! orphan is cancelled (Python's orphan reclaim).
//!
//! Ports `team_runner.py::_TeamRunnerMixin._maybe_release_team_session` and the
//! `TeamRuntimeManager` orphan sweep; the pool bounds are `config.py`'s
//! `TeamPoolConfig`.
//!
//! **Time is real, not paused, in this suite.** The eviction deadlines are
//! driven by `tokio::time`, but this crate's tokio does not enable `test-util`,
//! so `tokio::time::pause`/`advance` are unavailable here. Every case therefore
//! uses a very short idle timeout (50 ms) and a bounded `tokio::time::timeout`
//! poll: the reaper's *outcome* is deterministic, only the number of scan ticks
//! it takes to get there is not.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::sync::Arc;
use std::time::Duration;

use openjiuwen_agent::Agent;
use openjiuwen_common::error::ErrorKind;
use openjiuwen_multiagent::{handoff_team, BaseTeam, HandoffTeamConfig, Team};
use openjiuwen_runner::config::TeamPoolConfig;
use openjiuwen_runner::monitor::RunPhase;
use openjiuwen_runner::pool::{PoolFrame, PoolKey, PoolRequest, PoolRun, TeamPool};
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

/// The idle deadline the eviction cases configure.
const IDLE: Duration = Duration::from_millis(50);

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

/// A pool whose entries go idle after [`IDLE`] and are scanned every 10 ms.
fn evicting_pool() -> TeamPool {
    TeamPool::with_config(
        TeamPoolConfig::default()
            .with_idle_timeout_secs(IDLE.as_secs_f64())
            .with_reaper_interval_secs(0.01),
    )
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

/// Blocking-sleeps past several idle deadlines, to prove an entry is *not*
/// evicted (the only way to observe the absence of a reaper action).
async fn wait_past_the_idle_deadline() {
    tokio::time::sleep(IDLE * 5).await;
}

// ---------------------------------------------------------------------------
// Eviction
// ---------------------------------------------------------------------------

// ports team_runner.py::TeamRuntimeManager.finalize (a finished run is reclaimed)
#[tokio::test]
async fn a_finished_run_is_evicted_after_the_idle_timeout() {
    let pool = evicting_pool();
    let team = quick_team("evict-finished");
    let run = pool
        .start(common::services(), request(&team, "evict-finished-session"))
        .await
        .unwrap();
    let key = run.key().clone();
    let output = run.wait().await.unwrap();
    assert_eq!(output["payload"]["output"], json!(HANDOFF_SECOND_ANSWER));

    // Completion stamps the entry idle (it holds no lease), so the reaper drops
    // it once the deadline passes; until then `get` still answers.
    assert!(pool.get(&key).is_some(), "the terminal entry is retained");
    until(|| pool.get(&key).is_none()).await;
    assert_eq!(pool.len(), 0, "the evicted entry left the pool");
    assert_eq!(pool.list_active(), Vec::new(), "no live run is listed");
    assert!(pool.stop_all().await.is_ok());
}

// ports team_runner.py::_TeamRunnerMixin._maybe_release_team_session (orphan reclaim)
#[tokio::test]
async fn the_reaper_cancels_a_live_run_whose_last_lease_was_released() {
    let pool = evicting_pool();
    let (team, _gate) = blocked_team("evict-orphan");
    let lease = pool
        .acquire(common::services(), request(&team, "evict-orphan-session"))
        .await
        .unwrap();
    let key = lease.key().clone();
    let run: PoolRun = lease.run().clone();
    until(|| run.state() == RunPhase::Running).await;
    let mut frames = run.subscribe();

    // Releasing the lease leaves the run live but unattended: an orphan.
    lease.release();
    assert_eq!(pool.get(&key).expect("the entry stays").leases, 0);
    assert_eq!(
        run.state(),
        RunPhase::Running,
        "releasing is not cancelling"
    );

    until(|| run.state() == RunPhase::Cancelled).await;
    assert!(run.is_finished());
    assert_eq!(
        run.wait().await.unwrap_err().kind(),
        ErrorKind::Cancelled,
        "an evicted orphan reports cancellation"
    );
    let published: Vec<PoolFrame> = std::iter::from_fn(|| frames.try_recv().ok()).collect();
    assert_eq!(
        published.last(),
        Some(&PoolFrame::Cancelled {
            reason: "idle timeout".to_owned()
        }),
        "the reaper records its own reason: {published:?}"
    );

    until(|| pool.get(&key).is_none()).await;
    assert_eq!(pool.list_active(), Vec::new(), "no live run is listed");
    assert_eq!(pool.len(), 0);
    assert_eq!(pool.available_permits(), pool.capacity());
    assert!(pool.stop_all().await.is_ok());
}

// ports team_runner.py's pool entry pinning (a leased runtime is never reclaimed)
#[tokio::test]
async fn a_leased_run_is_never_evicted() {
    let pool = evicting_pool();
    let (team, _gate) = blocked_team("evict-leased");
    let lease = pool
        .acquire(common::services(), request(&team, "evict-leased-session"))
        .await
        .unwrap();
    let key = lease.key().clone();
    let run = lease.run().clone();
    until(|| run.state() == RunPhase::Running).await;
    assert!(!lease.is_released());
    assert_eq!(pool.get(&key).expect("live").leases, 1);

    // Several idle deadlines pass with the lease held: nothing may be evicted.
    wait_past_the_idle_deadline().await;
    let held = pool.get(&key).expect("a leased entry survives the reaper");
    assert_eq!(held.leases, 1);
    assert_eq!(held.state, RunPhase::Running);
    assert_eq!(run.state(), RunPhase::Running);
    assert_eq!(pool.len(), 1);

    // Dropping an unreleased lease releases it, and then eviction applies.
    drop(lease);
    assert_eq!(pool.get(&key).expect("still retained").leases, 0);
    until(|| run.state() == RunPhase::Cancelled).await;
    until(|| pool.get(&key).is_none()).await;
    assert_eq!(pool.len(), 0);
    assert!(pool.stop_all().await.is_ok());
}

// ---------------------------------------------------------------------------
// Lease refcounting
// ---------------------------------------------------------------------------

// ports team_runner.py's "the runtimes are shared, the callers are not" contract
#[tokio::test]
async fn two_acquisitions_join_one_entry_and_refcount_the_lease() {
    let pool = evicting_pool();
    let (team, _gate) = blocked_team("refcount-crew");
    let first = pool
        .acquire(common::services(), request(&team, "refcount-session"))
        .await
        .unwrap();
    let second = pool
        .acquire(common::services(), request(&team, "refcount-session"))
        .await
        .unwrap();

    // The second `acquire` joins the live run instead of starting a new one.
    assert_eq!(first.run_id(), second.run_id(), "one entry, two callers");
    let key: PoolKey = first.key().clone();
    assert_eq!(second.key(), &key);
    assert_eq!(pool.len(), 1, "the join created no second entry");
    let joined = pool.get(&key).expect("the entry is live");
    assert_eq!(joined.leases, 2);
    assert!(joined.running);
    assert_eq!(pool.list_active().len(), 1);

    let run = first.run().clone();
    until(|| run.state() == RunPhase::Running).await;

    // One lease released: the entry is still pinned, so nothing is evicted.
    first.release();
    assert_eq!(pool.get(&key).expect("pinned").leases, 1);
    wait_past_the_idle_deadline().await;
    assert!(
        pool.get(&key).is_some(),
        "the remaining lease keeps the entry out of eviction"
    );
    assert_eq!(pool.get(&key).unwrap().state, RunPhase::Running);

    // The last lease released makes the entry idle, and then evictable.
    second.release();
    assert_eq!(pool.get(&key).expect("still retained").leases, 0);
    until(|| pool.get(&key).is_none()).await;
    assert_eq!(run.state(), RunPhase::Cancelled);
    assert_eq!(pool.len(), 0);
    assert!(pool.stop_all().await.is_ok());
}
