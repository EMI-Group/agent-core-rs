//! The [`Runner`] facade's observation and team-runtime surface: the shared run
//! monitor and HITL hub every run reports into, the pool-backed team entry
//! points, and the lifecycle that rebuilds the pool.
//!
//! Every case goes through the facade (`Runner::run_agent`,
//! `Runner::run_agent_team`, `Runner::start_agent_team`, ...) over the offline
//! fixtures: in-process `ScriptedAgent`s, the real `BaseTeam`/handoff team, and
//! no network, clock, or filesystem dependency.
//!
//! Ports `team_runner.py`'s `_TeamRunnerMixin` surface (`list_active_teams`,
//! `interact_agent_team`, `pause_agent_team`, `stop_agent_team`) and the
//! per-run status bookkeeping the Python runner kept for its active teams.
//!
//! **Why `#[tokio::test]`'s current-thread runtime matters.** `start_agent_team`
//! returns after the pool spawns the worker but *before* its first poll, so a
//! listing taken right after the call deterministically observes the queued run
//! instead of racing its completion.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::assert_is_empty
)]

mod common;

use std::sync::Arc;
use std::time::Duration;

use openjiuwen_agent::Agent;
use openjiuwen_common::error::ErrorKind;
use openjiuwen_multiagent::{handoff_team, BaseTeam, HandoffTeamConfig, Team};
use openjiuwen_runner::config::TeamPoolConfig;
use openjiuwen_runner::hitl::{DeliverResult, InteractPayload, InteractionHub};
use openjiuwen_runner::monitor::{RunEvent, RunKind, RunMonitor, RunPhase};
use openjiuwen_runner::pool::PoolKey;
use openjiuwen_runner::run::{AgentRef, RunOptions, TeamRef};
use openjiuwen_runner::{InMemoryMessageQueue, MessageQueue, Runner, RunnerConfig};
use openjiuwen_session::interaction::InteractionResponse;
use serde_json::json;
use tokio::sync::broadcast;
use tokio::sync::Notify;

use common::{
    agent_card, handoff_team_fixture, provider, scripted_agent, team_calls, team_card,
    ScriptedAgent, TeamCalls, HANDOFF_SECOND_ANSWER,
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

/// The monitor events published so far, oldest first.
fn published(receiver: &mut broadcast::Receiver<RunEvent>) -> Vec<RunEvent> {
    std::iter::from_fn(|| receiver.try_recv().ok()).collect()
}

/// Builds the `(team, calls)` fixture under `team_id`.
fn fixture(team_id: &str) -> (Arc<BaseTeam>, Arc<TeamCalls>) {
    let calls = team_calls();
    let team = handoff_team_fixture(team_id, &calls);
    (team, calls)
}

/// A one-member handoff team whose only member blocks until the returned gate
/// opens, so the run stays live (and addressable) for the assertions.
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

/// The registry-minted agent reference for a scripted agent.
fn agent_ref(agent: Arc<ScriptedAgent>) -> AgentRef {
    AgentRef::of(agent as Arc<dyn Agent>)
}

/// The registry-free team reference for a live team.
fn team_ref(team: Arc<BaseTeam>) -> TeamRef {
    TeamRef::of(team as Arc<dyn Team>)
}

// ---------------------------------------------------------------------------
// Observation seams
// ---------------------------------------------------------------------------

// ports team_runner.py's per-run status record (label + session id + status)
#[tokio::test]
async fn the_accessors_expose_the_seams_the_run_paths_report_into() {
    let runner = Runner::new();
    let monitor = runner.monitor();
    let hub = runner.hitl();

    // The hub mirrors its transitions into the runner's own monitor, so a host
    // renders one consistent view of both seams.
    assert!(
        Arc::ptr_eq(
            &hub.monitor().expect("the hub mirrors into a monitor"),
            &monitor
        ),
        "the hub and the run paths share the runner's monitor"
    );
    assert!(hub.is_empty());
    assert!(monitor.list_active().is_empty());

    let mut events = monitor.subscribe();
    let output = runner
        .run_agent(
            agent_ref(scripted_agent("assistant", "hello")),
            json!({"query": "hi"}),
            RunOptions::default(),
        )
        .await
        .unwrap();
    assert_eq!(output.output, "hello");

    let emitted = published(&mut events);
    let run_id = emitted.first().expect("the run is admitted").run_id.clone();
    assert_eq!(emitted.first().unwrap().kind, RunKind::Agent);
    assert_eq!(emitted.first().unwrap().label, "assistant");

    let status = monitor.snapshot(&run_id).expect("the run is retained");
    assert_eq!(status.phase, RunPhase::Completed);
    assert!(
        status.finished_at.is_some(),
        "the terminal transition stamps the end: {status:?}"
    );
    assert_eq!(
        emitted
            .iter()
            .map(|event| event.phase)
            .collect::<Vec<RunPhase>>(),
        vec![RunPhase::Queued, RunPhase::Running, RunPhase::Completed]
    );
    assert!(
        monitor.list_active().is_empty(),
        "a finished run is not live"
    );
    assert!(
        hub.is_empty(),
        "the hub registration is removed on completion"
    );
    assert!(Arc::ptr_eq(&runner.monitor(), &monitor));
}

// ports team_runner.py::run_agent_team (the kind the status record carries)
#[tokio::test]
async fn agent_and_team_runs_both_mirror_into_the_monitor() {
    let runner = Runner::new();
    let mut events = runner.monitor().subscribe();

    runner
        .run_agent(
            agent_ref(scripted_agent("assistant", "done")),
            json!({"query": "hi"}),
            RunOptions::default(),
        )
        .await
        .unwrap();
    let (team, _calls) = fixture("direct-crew");
    let result = runner
        .run_agent_team(
            team_ref(team),
            json!({"query": "hi", "conversation_id": "direct-session"}),
            RunOptions::default(),
        )
        .await
        .unwrap();
    assert_eq!(result["output"], json!(HANDOFF_SECOND_ANSWER));

    let emitted = published(&mut events);
    let mut kinds: Vec<RunKind> = emitted.iter().map(|event| event.kind).collect();
    kinds.dedup();
    assert_eq!(
        kinds,
        vec![RunKind::Agent, RunKind::Team],
        "one run of each kind, in call order"
    );
    let terminal: Vec<(RunKind, RunPhase)> = emitted
        .iter()
        .filter(|event| event.phase.is_terminal())
        .map(|event| (event.kind, event.phase))
        .collect();
    assert_eq!(
        terminal,
        vec![
            (RunKind::Agent, RunPhase::Completed),
            (RunKind::Team, RunPhase::Completed)
        ]
    );
    assert!(
        runner.hitl().is_empty(),
        "neither run leaves a live hub registration"
    );
}

// ports team_runner.py::list_active_teams (only live runs are listed)
#[tokio::test]
async fn start_agent_team_lists_awaits_and_then_drops_the_run() {
    let runner = Runner::new();
    let (team, calls) = fixture("pooled-crew");
    let run = runner
        .start_agent_team(
            team_ref(team),
            json!({"query": "hi", "conversation_id": "pool-session"}),
            RunOptions::default(),
        )
        .await
        .unwrap();

    let key = PoolKey::new("pooled-crew", "pool-session");
    assert_eq!(run.key(), &key);
    assert_eq!(run.session_id(), "pool-session");

    // The runner's pool is the pool the handle came from, and the run is live.
    let pool = runner.team_pool();
    assert_eq!(pool.get(&key).expect("live").run_id, run.run_id());
    let listed = runner.list_active_teams();
    assert_eq!(listed.len(), 1, "one live run: {listed:?}");
    assert_eq!(listed[0].run_id, run.run_id());
    assert_eq!(listed[0].team_name, "pooled-crew");
    assert_eq!(listed[0].session_id, "pool-session");
    assert_eq!(
        runner.hitl().list_active().len(),
        1,
        "registered on the hub"
    );

    // A second start on the same (team, session) is the pool's busy rejection.
    let busy = runner
        .start_agent_team(
            team_ref(fixture("pooled-crew").0),
            json!({"query": "again", "conversation_id": "pool-session"}),
            RunOptions::default(),
        )
        .await
        .unwrap_err();
    assert_eq!(busy.kind(), ErrorKind::Validation);
    assert!(
        busy.to_string().contains("already active"),
        "unexpected message: {busy}"
    );

    let output = run.wait().await.unwrap();
    assert_eq!(
        output["payload"]["output"],
        json!(HANDOFF_SECOND_ANSWER),
        "the pooled run streams the team's own envelope: {output}"
    );
    assert_eq!(run.state(), RunPhase::Completed);
    assert_eq!(calls.inputs().len(), 2, "both members ran once");
    assert!(
        runner.list_active_teams().is_empty(),
        "a finished run leaves the active listing"
    );
    assert_eq!(
        runner.team_pool().available_permits(),
        runner.team_pool().capacity(),
        "the concurrency permit is back"
    );
}

// ---------------------------------------------------------------------------
// HITL facade
// ---------------------------------------------------------------------------

// ports team_runner.py::interact_agent_team / pause_agent_team / stop_agent_team
#[tokio::test]
async fn interact_pause_and_stop_address_one_pooled_run() {
    let runner = Runner::new();
    let (team, _gate) = blocked_team("hitl-crew");
    let run = runner
        .start_agent_team(
            team_ref(team),
            json!({"query": "hi", "conversation_id": "hitl-session"}),
            RunOptions::default(),
        )
        .await
        .unwrap();
    let key = PoolKey::new("hitl-crew", "hitl-session");
    until(|| run.state() == RunPhase::Running).await;

    // A payload addressed to the live pair is accepted and buffered for replay.
    assert_eq!(
        runner.interact_agent_team("hitl-crew", "hitl-session", InteractPayload::text("hello")),
        DeliverResult::Accepted
    );
    until(|| !run.payloads().is_empty()).await;
    assert_eq!(run.payloads()[0].text.as_deref(), Some("hello"));

    // A target that matches no live run is reported, never broadcast.
    assert_eq!(
        runner.interact_agent_team("other-crew", "hitl-session", InteractPayload::text("x")),
        DeliverResult::MissingTarget
    );
    assert_eq!(
        runner.interact_agent_team("hitl-crew", "other-session", InteractPayload::text("x")),
        DeliverResult::MissingTarget
    );

    // The status snapshot correlates the key with the run.
    let status = runner
        .agent_team_status("hitl-crew", "hitl-session")
        .expect("the live run is known");
    assert_eq!(status.run_id, run.run_id());
    assert_eq!(status.kind, RunKind::Team);
    assert_eq!(status.phase, RunPhase::Running);
    assert_eq!(status.session_id.as_deref(), Some("hitl-session"));
    assert!(runner.agent_team_status("nobody", "s-1").is_none());

    // Pause/resume are refused: nothing can suspend the in-flight invocation.
    let pause = runner
        .pause_agent_team("hitl-crew", "hitl-session")
        .await
        .expect_err("a live pooled run cannot be paused");
    assert_eq!(pause.kind(), ErrorKind::Unsupported);
    assert!(
        pause.to_string().contains("InteractGate") && pause.to_string().contains("cancel"),
        "the refusal names the missing seam and the alternative: {pause}"
    );
    let resume = runner
        .resume_agent_team("hitl-crew", "hitl-session")
        .await
        .expect_err("a live pooled run cannot be resumed");
    assert_eq!(resume.kind(), ErrorKind::Unsupported);
    let live = runner.team_pool().get(&key).expect("live");
    assert_eq!(
        live.state,
        RunPhase::Running,
        "the refusal changed no phase"
    );
    assert!(!live.gate_closed, "the refusal left the gate open");
    let missing = runner.pause_agent_team("nobody", "s-1").await.unwrap_err();
    assert_eq!(missing.kind(), ErrorKind::NotFound);
    let missing = runner.resume_agent_team("nobody", "s-1").await.unwrap_err();
    assert_eq!(missing.kind(), ErrorKind::NotFound);

    // Stopping cancels the blocked run; a second stop is a no-op `false`.
    assert!(
        runner
            .stop_agent_team("hitl-crew", "hitl-session")
            .await
            .unwrap(),
        "the run was live"
    );
    assert!(run.is_finished());
    assert_eq!(run.state(), RunPhase::Cancelled);
    assert_eq!(
        run.wait().await.unwrap_err().kind(),
        ErrorKind::Cancelled,
        "a cancelled run reports cancellation from `wait`"
    );
    assert!(!runner
        .stop_agent_team("hitl-crew", "hitl-session")
        .await
        .unwrap());
    assert!(runner.list_active_teams().is_empty());
    assert!(runner.hitl().is_empty(), "the hub entry is gone");
}

#[tokio::test]
async fn a_direct_team_run_is_status_addressable_while_it_is_live() {
    let runner = Runner::new();
    let (team, gate) = blocked_team("direct-crew");
    let task = {
        let runner = Arc::clone(&runner);
        tokio::spawn(async move {
            runner
                .run_agent_team(
                    team_ref(team),
                    json!({"query": "hi", "conversation_id": "direct-session"}),
                    RunOptions::default(),
                )
                .await
        })
    };

    until(|| runner.hitl().list_active().len() == 1).await;
    let status = runner
        .agent_team_status("direct-crew", "direct-session")
        .expect("the live hub registration is the correlation");
    assert_eq!(status.kind, RunKind::Team);
    assert!(
        matches!(status.phase, RunPhase::Queued | RunPhase::Running),
        "the direct run is live: {status:?}"
    );
    assert_eq!(status.session_id.as_deref(), Some("direct-session"));

    // The correlation disappears with the run: the hub entry and the live
    // status are both gone once the member answers.
    gate.notify_one();
    let result = tokio::time::timeout(BOUND, task)
        .await
        .expect("the run finishes before the deadline")
        .expect("the run task does not panic")
        .unwrap();
    assert_eq!(result["output"], json!("released"));
    assert!(runner.hitl().is_empty(), "the hub entry is gone");
    assert!(runner.monitor().list_active().is_empty());
    assert!(runner
        .agent_team_status("direct-crew", "direct-session")
        .is_none());
}

#[tokio::test]
async fn answer_interaction_for_an_unknown_request_is_not_found() {
    let runner = Runner::new();
    let error = runner
        .answer_interaction(InteractionResponse::answer("missing", json!(1)))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert_eq!(
        error.to_string(),
        "not_found: interaction 'missing' is not pending on any active run"
    );
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stop_stops_every_pooled_run_and_leaves_a_fresh_pool() {
    let runner = Runner::new();
    let (team_a, _gate_a) = blocked_team("sweep-a");
    let (team_b, _gate_b) = blocked_team("sweep-b");
    let run_a = runner
        .start_agent_team(
            team_ref(team_a),
            json!({"query": "hi", "conversation_id": "sweep-session-a"}),
            RunOptions::default(),
        )
        .await
        .unwrap();
    let run_b = runner
        .start_agent_team(
            team_ref(team_b),
            json!({"query": "hi", "conversation_id": "sweep-session-b"}),
            RunOptions::default(),
        )
        .await
        .unwrap();
    until(|| run_a.state() == RunPhase::Running && run_b.state() == RunPhase::Running).await;
    assert_eq!(runner.list_active_teams().len(), 2);

    // A runner that never started still owns the pooled runs `stop` must sweep.
    assert!(!runner.is_started());
    runner.stop().await.unwrap();

    for run in [&run_a, &run_b] {
        assert!(run.is_finished(), "the sweep finished the run: {run:?}");
        assert_eq!(run.state(), RunPhase::Cancelled);
    }
    let pool = runner.team_pool();
    assert!(pool.is_empty(), "the swept pool is replaced, not emptied");
    assert!(!pool.is_closed(), "the replacement pool admits runs");
    assert!(runner.hitl().is_empty());
    assert!(runner.monitor().list_active().is_empty());

    // The replacement pool is usable without a `start`.
    let (team, _calls) = fixture("after-stop-crew");
    let after = runner
        .start_agent_team(
            team_ref(team),
            json!({"query": "hi", "conversation_id": "after-stop"}),
            RunOptions::default(),
        )
        .await
        .unwrap();
    assert_eq!(
        after.wait().await.unwrap()["payload"]["output"],
        json!(HANDOFF_SECOND_ANSWER)
    );
}

#[tokio::test]
async fn set_config_takes_effect_on_the_next_restart() {
    let runner = Runner::with_config(
        RunnerConfig::default()
            .with_team_pool(TeamPoolConfig::default().with_max_concurrent_runs(2)),
    );
    assert_eq!(runner.team_pool().capacity(), 2);
    assert_eq!(runner.team_pool().available_permits(), 2);

    runner.set_config(
        RunnerConfig::default()
            .with_team_pool(TeamPoolConfig::default().with_max_concurrent_runs(5)),
    );
    assert_eq!(
        runner.team_pool().capacity(),
        2,
        "set_config takes effect on the next start"
    );

    runner.start().await.unwrap();
    assert_eq!(runner.team_pool().capacity(), 5, "start rebuilds the pool");
    assert_eq!(runner.team_pool().available_permits(), 5);
    assert!(!runner.team_pool().is_closed());

    runner.stop().await.unwrap();
    assert_eq!(
        runner.team_pool().capacity(),
        5,
        "stop installs a fresh pool from the live configuration"
    );
    assert!(!runner.team_pool().is_closed());
    assert!(!runner.is_started());

    runner.start().await.unwrap();
    assert_eq!(runner.team_pool().capacity(), 5, "a restart keeps it");
    runner.stop().await.unwrap();
}

#[tokio::test]
async fn a_restart_stops_the_pooled_runs_of_the_retired_pool() {
    let runner = Runner::new();
    runner.start().await.unwrap();
    let (team, _gate) = blocked_team("restart-crew");
    let run = runner
        .start_agent_team(
            team_ref(team),
            json!({"query": "hi", "conversation_id": "restart-session"}),
            RunOptions::default(),
        )
        .await
        .unwrap();
    until(|| run.state() == RunPhase::Running).await;

    // `start` on a started runner is a logged no-op: the run keeps running.
    runner.start().await.unwrap();
    assert_eq!(run.state(), RunPhase::Running);

    runner.stop().await.unwrap();
    assert!(run.is_finished(), "the restart boundary stopped the run");
    assert_eq!(run.state(), RunPhase::Cancelled);
    runner.start().await.unwrap();
    assert!(runner.team_pool().is_empty());
    assert!(!runner.team_pool().is_closed());
    runner.stop().await.unwrap();
}

#[tokio::test]
async fn the_debug_view_reports_counts_without_payloads() {
    let runner = Runner::new();
    let (team, _gate) = blocked_team("debug-crew");
    let run = runner
        .start_agent_team(
            team_ref(team),
            json!({"query": "hi", "conversation_id": "debug-session"}),
            RunOptions::default(),
        )
        .await
        .unwrap();
    until(|| run.state() == RunPhase::Running).await;
    let _ = runner.interact_agent_team(
        "debug-crew",
        "debug-session",
        InteractPayload::text("secret"),
    );

    let debug = format!("{runner:?}");
    assert!(debug.contains("pooled_teams: 1"), "{debug}");
    assert!(debug.contains("active_runs: 1"), "{debug}");
    assert!(
        !debug.contains("secret"),
        "the debug view never leaks a payload: {debug}"
    );

    runner.stop().await.unwrap();
    assert!(run.is_finished());
}

// ---------------------------------------------------------------------------
// Replaceable seams
// ---------------------------------------------------------------------------

// The monitor is installable after construction: every later run reports into
// the installed instance, and the hub's mirroring follows the swap.
#[tokio::test]
async fn set_monitor_retargets_subsequent_agent_and_team_runs() {
    let runner = Runner::new();
    let retired = runner.monitor();
    let mut retired_events = retired.subscribe();

    let dashboard = Arc::new(RunMonitor::new());
    let mut events = dashboard.subscribe();
    let previous = runner.set_monitor(Arc::clone(&dashboard));
    assert!(
        Arc::ptr_eq(&previous, &retired),
        "the setter returns the previously installed monitor"
    );
    assert!(Arc::ptr_eq(&runner.monitor(), &dashboard));

    // The installed hub is re-pointed, so its transitions mirror into the new
    // monitor too.
    let hub_monitor = runner.hitl().monitor().expect("the hub mirrors a monitor");
    assert!(
        Arc::ptr_eq(&hub_monitor, &dashboard),
        "the hub follows the installed monitor"
    );

    runner
        .run_agent(
            agent_ref(scripted_agent("assistant", "done")),
            json!({"query": "hi"}),
            RunOptions::default(),
        )
        .await
        .unwrap();
    let (team, _calls) = fixture("direct-seam-crew");
    runner
        .run_agent_team(
            team_ref(team),
            json!({"query": "hi", "conversation_id": "direct-seam"}),
            RunOptions::default(),
        )
        .await
        .unwrap();
    let (pooled, _calls) = fixture("pooled-seam-crew");
    runner
        .start_agent_team(
            team_ref(pooled),
            json!({"query": "hi", "conversation_id": "pooled-seam"}),
            RunOptions::default(),
        )
        .await
        .unwrap()
        .wait()
        .await
        .unwrap();

    let emitted = published(&mut events);
    let mut kinds: Vec<RunKind> = emitted.iter().map(|event| event.kind).collect();
    kinds.dedup();
    assert_eq!(
        kinds,
        vec![RunKind::Agent, RunKind::Team],
        "the direct agent, the direct team, and the pooled team all report here"
    );
    assert_eq!(
        emitted
            .iter()
            .filter(|event| event.phase.is_terminal())
            .count(),
        3,
        "three runs finished on the installed monitor: {emitted:?}"
    );
    assert!(
        published(&mut retired_events).is_empty(),
        "the retired monitor sees no new transition"
    );
    assert!(retired.list_active().is_empty());
    assert!(dashboard.list_active().is_empty(), "every run finished");
}

// The hub is installable after construction and used as given: the new
// registrations land on it, the retired hub keeps its own.
#[tokio::test]
async fn set_hitl_installs_a_host_owned_hub() {
    let runner = Runner::new();
    let previous = runner.hitl();
    let host_hub = Arc::new(InteractionHub::new());

    let retired = runner.set_hitl(Arc::clone(&host_hub));
    assert!(Arc::ptr_eq(&retired, &previous));
    assert!(Arc::ptr_eq(&runner.hitl(), &host_hub));
    assert!(
        host_hub.monitor().is_none(),
        "the setter does not attach the runner's monitor implicitly"
    );

    let (team, gate) = blocked_team("swapped-crew");
    let run = runner
        .start_agent_team(
            team_ref(team),
            json!({"query": "hi", "conversation_id": "swapped-session"}),
            RunOptions::default(),
        )
        .await
        .unwrap();
    until(|| host_hub.list_active().len() == 1).await;
    assert_eq!(host_hub.list_active()[0].run_id, run.run_id());
    assert!(
        previous.is_empty(),
        "live registrations are not migrated to the installed hub"
    );

    // The facade's HITL shortcuts address the installed hub.
    assert_eq!(
        runner.interact_agent_team(
            "swapped-crew",
            "swapped-session",
            InteractPayload::text("x")
        ),
        DeliverResult::Accepted
    );

    gate.notify_one();
    run.wait().await.unwrap();
    assert!(
        host_hub.is_empty(),
        "the registration is removed on completion"
    );
}

// The queue is installable after construction; `start` still owns the slot.
#[tokio::test]
async fn set_message_queue_installs_a_host_owned_queue() {
    let runner = Runner::new();
    let previous = runner.message_queue();
    let broker: Arc<dyn MessageQueue> = InMemoryMessageQueue::with_capacity(8);

    let retired = runner.set_message_queue(Arc::clone(&broker));
    assert!(Arc::ptr_eq(&retired, &previous));
    assert!(Arc::ptr_eq(&runner.message_queue(), &broker));
    assert!(!runner.is_started());

    // `start` rebuilds the queue from the configuration, replacing the installed
    // one with the configured backend.
    runner.start().await.unwrap();
    assert!(
        !Arc::ptr_eq(&runner.message_queue(), &broker),
        "start installs the configured backend"
    );
    runner.stop().await.unwrap();
}
