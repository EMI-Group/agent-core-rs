//! The team runtime pool's human-in-the-loop binding: inbound payload delivery,
//! the payload buffer and its frame, the drain, and the unbinding on
//! finalisation.
//!
//! A pooled run registers itself on the runner's `InteractionHub` (the pool owns
//! the observation seam, so the inner `stream_team` does not register a second
//! time) and buffers every inbound human payload for the host to replay. The
//! live `InteractGate` wake-up is deferred, so a payload never suspends the team.
//!
//! Ports `team_runner.py::interact_agent_team` (delivery plus `DeliverResult`),
//! `register_human_agent_inbound` (the inbound channel), and
//! `_TeamRunnerMixin.stop_agent_team`'s unregistration.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::sync::Arc;
use std::time::Duration;

use openjiuwen_agent::Agent;
use openjiuwen_common::error::ErrorKind;
use openjiuwen_multiagent::{handoff_team, BaseTeam, HandoffTeamConfig, Team};
use openjiuwen_runner::hitl::{
    DeliverResult, InteractKind, InteractPayload, InteractionHub, RunTarget,
};
use openjiuwen_runner::monitor::{RunMonitor, RunPhase};
use openjiuwen_runner::pool::{PoolFrame, PoolRequest, TeamPool};
use openjiuwen_runner::run::TeamRef;
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

/// The request a suite sends for `team` in session `session`.
fn request(team: &Arc<BaseTeam>, session: &str) -> PoolRequest {
    PoolRequest::new(
        TeamRef::of(Arc::clone(team) as Arc<dyn Team>),
        json!({"query": "hi", "conversation_id": session}),
    )
}

// ports team_runner.py::interact_agent_team (deliver + the inbound buffer)
#[tokio::test]
async fn an_inbound_payload_is_buffered_framed_and_drained() {
    let hub = Arc::new(InteractionHub::new());
    let monitor = Arc::new(RunMonitor::new());
    let services = common::services()
        .with_hitl(Arc::clone(&hub))
        .with_monitor(Arc::clone(&monitor));
    let (team, _gate) = blocked_team("hitl-crew");

    let pool = TeamPool::new();
    let run = pool
        .start(services, request(&team, "hitl-session"))
        .await
        .unwrap();
    let key = run.key().clone();
    let run_id = run.run_id().to_owned();
    let mut frames = run.subscribe();
    until(|| run.state() == RunPhase::Running).await;

    // The pool registered exactly one hub entry for the run, addressing the team
    // and the session, so a host can target it either way.
    let registered = hub.list_active();
    assert_eq!(registered.len(), 1, "one registration per pooled run");
    assert_eq!(registered[0].run_id, run_id);
    assert_eq!(registered[0].team_name.as_deref(), Some("hitl-crew"));
    assert_eq!(registered[0].session_id.as_deref(), Some("hitl-session"));

    let delivered = hub.deliver(
        RunTarget::for_team("hitl-crew"),
        InteractPayload::text("hello"),
    );
    assert_eq!(delivered, DeliverResult::Accepted);
    assert!(delivered.success());

    // The run's drain task moves the payload into its bounded buffer.
    until(|| !run.payloads().is_empty()).await;
    let buffered = run.payloads();
    assert_eq!(buffered.len(), 1, "one payload was buffered: {buffered:?}");
    assert_eq!(buffered[0].kind, InteractKind::Text);
    assert_eq!(buffered[0].text.as_deref(), Some("hello"));
    assert_eq!(
        pool.get(&key).expect("live").queued,
        1,
        "the snapshot counts it"
    );

    // The buffer is published as a frame, and the team keeps running: a payload
    // is a message for the host to replay, never a suspension.
    let published: Vec<PoolFrame> = std::iter::from_fn(|| frames.try_recv().ok()).collect();
    assert!(
        published.iter().any(|frame| matches!(
            frame,
            PoolFrame::Payload { payload } if payload.text.as_deref() == Some("hello")
        )),
        "the payload frame was published: {published:?}"
    );
    assert_eq!(
        run.state(),
        RunPhase::Running,
        "delivery does not interrupt"
    );

    // Draining is a take: the buffer empties and a second drain is empty.
    let drained = pool.take_payloads(&key);
    assert_eq!(drained.len(), 1);
    assert_eq!(drained[0].text.as_deref(), Some("hello"));
    assert!(
        pool.take_payloads(&key).is_empty(),
        "the drain is destructive"
    );
    assert_eq!(
        run.payloads(),
        Vec::new(),
        "the buffer is empty after a drain"
    );
    assert_eq!(pool.get(&key).expect("live").queued, 0);

    // Finalising unbinds the run; the monitor keeps its record.
    assert!(pool.stop(&key).await.unwrap());
    until(|| hub.is_empty()).await;
    assert_eq!(hub.len(), 0, "the hub entry is gone");
    assert_eq!(hub.list_active(), Vec::new(), "the hub lists no run");
    assert_eq!(
        monitor.list_active(),
        Vec::new(),
        "the monitor lists no live run"
    );
    let status = monitor
        .snapshot(&run_id)
        .expect("the monitor recorded the run");
    assert_eq!(status.phase, RunPhase::Cancelled);
    assert!(status.finished_at.is_some());
    assert_eq!(run.state(), RunPhase::Cancelled);
    assert!(pool.stop_all().await.is_ok());
}

// ports team_runner.py::interact_agent_team (DeliverResult::MissingTarget)
#[tokio::test]
async fn a_payload_for_no_live_run_is_not_delivered() {
    let hub = Arc::new(InteractionHub::new());
    let pool = TeamPool::new();

    // Nothing is registered yet: the target matches nothing.
    assert_eq!(
        hub.deliver(
            RunTarget::for_team("hitl-ghost"),
            InteractPayload::text("hi")
        ),
        DeliverResult::MissingTarget
    );
    assert_eq!(
        hub.deliver(RunTarget::new(), InteractPayload::text("hi")),
        DeliverResult::MissingTarget,
        "an empty target never broadcasts"
    );

    // Once the run is finalised the target matches nothing again.
    let services = common::services().with_hitl(Arc::clone(&hub));
    let (team, _gate) = blocked_team("hitl-ghost");
    let run = pool
        .start(services, request(&team, "hitl-ghost-session"))
        .await
        .unwrap();
    let key = run.key().clone();
    assert_eq!(
        hub.deliver(
            RunTarget::for_team("hitl-ghost"),
            InteractPayload::text("hi")
        ),
        DeliverResult::Accepted
    );
    until(|| !run.payloads().is_empty()).await;

    assert!(pool.stop(&key).await.unwrap());
    assert_eq!(hub.len(), 0, "the hub registered no run");
    assert_eq!(
        hub.deliver(
            RunTarget::for_team("hitl-ghost"),
            InteractPayload::text("again")
        ),
        DeliverResult::MissingTarget
    );
    assert_eq!(
        run.wait().await.unwrap_err().kind(),
        ErrorKind::Cancelled,
        "the run is terminal, not merely unbound"
    );
    assert!(pool.stop_all().await.is_ok());
}
