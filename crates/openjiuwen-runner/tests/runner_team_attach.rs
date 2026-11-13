//! The `Runner` facade's team-attach primitives: refuse-instead-of-queue
//! admission (`try_start_agent_team`), lease-based joining (`acquire_agent_team`),
//! and the pooled-run HITL payload drain (`take_team_payloads`).
//!
//! These are the pool primitives `TeamPool::{try_start, acquire, take_payloads}`
//! reached through the facade, so a host that must join or refuse a live
//! `(team name, session id)` run never rebuilds a `RunServices` by hand.
//!
//! Ports `team_runner.py`'s pooled `start_agent_team` admission and
//! `_maybe_release_team_session` lease handling; the joining lease is the
//! in-process form of the cross-process attach Python defers.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::sync::Arc;
use std::time::Duration;

use openjiuwen_agent::Agent;
use openjiuwen_common::error::ErrorKind;
use openjiuwen_multiagent::{handoff_team, BaseTeam, HandoffTeamConfig, Team};
use openjiuwen_runner::config::{RunnerConfig, TeamPoolConfig};
use openjiuwen_runner::hitl::{DeliverResult, InteractKind, InteractPayload};
use openjiuwen_runner::monitor::RunPhase;
use openjiuwen_runner::pool::PoolKey;
use openjiuwen_runner::run::RunOptions;
use openjiuwen_runner::Runner;
use serde_json::json;
use tokio::sync::Notify;

use common::{agent_card, provider, team_card, ScriptedAgent};

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

/// A runner whose team pool admits `capacity` runs at once.
fn runner_with_capacity(capacity: usize) -> Arc<Runner> {
    Runner::with_config(
        RunnerConfig::default()
            .with_team_pool(TeamPoolConfig::default().with_max_concurrent_runs(capacity)),
    )
}

/// A one-member handoff team whose only member blocks on the returned gate, so
/// the run stays live until the gate opens or the run is stopped.
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

/// The inputs a pooled run of `session` is started with.
fn inputs(session: &str) -> serde_json::Value {
    json!({"query": "hi", "conversation_id": session})
}

// ports team_runner.py's pool admission: `try_start` refuses, `start` queues
#[tokio::test]
async fn a_full_pool_refuses_try_start_but_queues_start() {
    let runner = runner_with_capacity(1);

    // The first admission takes the only permit and runs.
    let (team_a, _gate_a) = blocked_team("cap-a");
    let first = runner
        .try_start_agent_team(team_a, inputs("cap-a-session"), RunOptions::default())
        .await
        .expect("an idle pool admits without queueing");
    until(|| first.state() == RunPhase::Running).await;

    // No permit is free: `try_start` reports the bound and creates nothing.
    let (team_b, _gate_b) = blocked_team("cap-b");
    let Err(refused) = runner
        .try_start_agent_team(team_b, inputs("cap-b-session"), RunOptions::default())
        .await
    else {
        panic!("a full pool must refuse");
    };
    assert_eq!(refused.kind(), ErrorKind::Config);
    assert!(refused.to_string().contains("at capacity"), "{refused}");
    assert_eq!(
        runner.list_active_teams().len(),
        1,
        "a refused admission creates no entry"
    );

    // `start` queues instead: the entry is observable as queued while it waits.
    let runner_for_queue = Arc::clone(&runner);
    let (team_c, _gate_c) = blocked_team("cap-c");
    let queued = tokio::spawn(async move {
        runner_for_queue
            .start_agent_team(team_c, inputs("cap-c-session"), RunOptions::default())
            .await
    });
    until(|| {
        runner
            .list_active_teams()
            .iter()
            .any(|team| team.team_name == "cap-c" && team.state == RunPhase::Queued)
    })
    .await;
    assert!(
        !queued.is_finished(),
        "the queued admission waits for the permit"
    );

    // Stopping the running one frees the permit, and the queued run is admitted.
    assert!(runner
        .stop_agent_team("cap-a", "cap-a-session")
        .await
        .unwrap());
    let second = tokio::time::timeout(BOUND, queued)
        .await
        .expect("the queued admission completes once a permit frees up")
        .expect("the admission task did not panic")
        .expect("the queueing admission succeeds");
    until(|| second.state() == RunPhase::Running).await;
    assert_eq!(runner.list_active_teams().len(), 1);
    assert_eq!(second.key(), &PoolKey::new("cap-c", "cap-c-session"));

    runner.stop().await.unwrap();
}

// ports team_runner.py's shared-runtime contract: a join reuses the live run
#[tokio::test]
async fn acquire_joins_the_live_run_and_ignores_the_joining_request() {
    let runner = Runner::new();
    let (team, _gate) = blocked_team("lease-crew");

    let first = runner
        .acquire_agent_team(
            Arc::clone(&team),
            inputs("lease-session"),
            RunOptions::default(),
        )
        .await
        .expect("the first acquisition starts the run");
    until(|| first.run().state() == RunPhase::Running).await;

    // The second host joins: same entry, same run id, one more lease. Its inputs
    // and options are the pool's documented ignored ones.
    let second = runner
        .acquire_agent_team(
            Arc::clone(&team),
            json!({"query": "joined", "conversation_id": "lease-session"}),
            RunOptions::default(),
        )
        .await
        .expect("a live key is joined, not refused");
    assert_eq!(first.run_id(), second.run_id(), "one entry, two callers");
    assert_eq!(second.key(), first.key());
    assert_eq!(first.key(), &PoolKey::new("lease-crew", "lease-session"));
    let active = runner.list_active_teams();
    assert_eq!(active.len(), 1, "the join created no second entry");
    assert_eq!(active[0].leases, 2, "both callers hold a lease");

    // Releasing one lease keeps the run alive for the other.
    second.release();
    assert_eq!(runner.list_active_teams()[0].leases, 1);
    assert_eq!(first.run().state(), RunPhase::Running);
    first.release();
    assert_eq!(runner.list_active_teams()[0].leases, 0);

    runner.stop().await.unwrap();
}

// ports team_runner.py::interact_agent_team (the drain without the PoolRun)
#[tokio::test]
async fn take_team_payloads_drains_a_delivered_payload_and_answers_unknown_keys() {
    let runner = Runner::new();
    assert_eq!(
        runner.take_team_payloads("no-such-team", "no-such-session"),
        Vec::new(),
        "an unknown key is an empty drain, not an error"
    );

    let (team, _gate) = blocked_team("payload-crew");
    let run = runner
        .start_agent_team(team, inputs("payload-session"), RunOptions::default())
        .await
        .unwrap();
    until(|| run.state() == RunPhase::Running).await;

    assert_eq!(
        runner.interact_agent_team(
            "payload-crew",
            "payload-session",
            InteractPayload::text("hello"),
        ),
        DeliverResult::Accepted
    );

    // The run's drain task moves the payload into its bounded buffer, which the
    // facade drains without the `PoolRun` handle.
    let mut drained = Vec::new();
    tokio::time::timeout(BOUND, async {
        loop {
            drained = runner.take_team_payloads("payload-crew", "payload-session");
            if !drained.is_empty() {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("the payload is buffered before the deadline");
    assert_eq!(drained.len(), 1, "one payload was buffered: {drained:?}");
    assert_eq!(drained[0].kind, InteractKind::Text);
    assert_eq!(drained[0].text.as_deref(), Some("hello"));
    assert!(
        runner
            .take_team_payloads("payload-crew", "payload-session")
            .is_empty(),
        "the drain is destructive"
    );
    assert!(
        runner
            .take_team_payloads("payload-crew", "other-session")
            .is_empty(),
        "a sibling session of the same team has nothing buffered"
    );

    runner.stop().await.unwrap();
}
