//! Agent-team execution behaviour: in-process runs, reference resolution,
//! session ownership, teardown, cancellation, deadlines, and streaming.
//!
//! Drives only the public `openjiuwen_runner::run` surface (`invoke_team`,
//! `stream_team`, `TeamRef`, `RunOptions`, `RunServices`) over a real
//! `BaseTeam`/handoff team, so the runner's in-process path is proven against
//! the multiagent seam rather than a stub.
//! Ports `tests/unit_tests/core/runner/test_team_runner.py` (`_run_base_team`,
//! `run_agent_team`, `run_agent_team_streaming`).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use openjiuwen_agent::Agent;
use openjiuwen_common::error::ErrorKind;
use openjiuwen_multiagent::{handoff_team, make_team_session, BaseTeam, HandoffTeamConfig, Team};
use openjiuwen_runner::hitl::InteractionHub;
use openjiuwen_runner::monitor::{RunKind, RunMonitor, RunPhase};
use openjiuwen_runner::run::{invoke_team, stream_team, RunOptions, TeamRef};
use openjiuwen_session::{Session, StreamEvent};
use serde_json::json;
use tokio_util::sync::CancellationToken;

use common::{
    agent_card, handoff_team_fixture, provider, team_calls, team_card, ScriptedAgent,
    HANDOFF_FIRST_ANSWER, HANDOFF_MESSAGE, HANDOFF_SECOND_ANSWER,
};

/// The dispatch deadline the deadline fixture configures on its team.
///
/// Short so the member dispatch a timed-out invocation leaves behind expires at
/// once; it is never a wait the suite depends on for ordering.
const DISPATCH_TIMEOUT: Duration = Duration::from_millis(100);

/// Builds the fixture team plus its call log.
fn fixture(team_id: &str) -> (Arc<BaseTeam>, Arc<common::TeamCalls>) {
    let calls = team_calls();
    let team = handoff_team_fixture(team_id, &calls);
    (team, calls)
}

/// A one-member handoff team whose only member sleeps `delay` before answering.
///
/// Used to prove the run deadline without a real long wait: the sleep future is
/// dropped as soon as the deadline fires. The team's own dispatch deadline is
/// shortened to [`DISPATCH_TIMEOUT`] so the member dispatch the dropped invocation
/// leaves behind expires immediately instead of at the 30 s team default.
fn delayed_team(team_id: &str, delay: Duration) -> Arc<BaseTeam> {
    let member: Arc<dyn Agent> = Arc::new(ScriptedAgent::new("slow").with_delay(delay));
    let team = handoff_team(
        team_card(team_id),
        HandoffTeamConfig::default().configure_timeout(DISPATCH_TIMEOUT.as_secs_f64()),
    );
    team.add_agent(agent_card("slow"), provider(member))
        .expect("the only member registers");
    Arc::new(team)
}

// ---------------------------------------------------------------------------
// Plain runs
// ---------------------------------------------------------------------------

// ports tests/unit_tests/core/runner/test_team_runner.py::test_run_agent_team
#[tokio::test]
async fn a_team_run_returns_the_final_answer() {
    let (team, calls) = fixture("handoff-team");
    let result = invoke_team(
        common::services(),
        TeamRef::of(Arc::clone(&team) as Arc<dyn Team>),
        json!({"query": "hi", "conversation_id": "conv-team"}),
        RunOptions::default(),
    )
    .await
    .unwrap();

    assert_eq!(result["output"], json!(HANDOFF_SECOND_ANSWER));
    assert_eq!(result["result_type"], json!("answer"));
    assert_eq!(
        calls.parent_session_ids(),
        vec![Some("conv-team".to_owned()), Some("conv-team".to_owned())],
        "the runner started the transport and leased one team session for both hops"
    );
    assert_eq!(
        calls.inputs().len(),
        2,
        "the first member hands off and the second answers"
    );
    assert_eq!(
        calls.inputs()[1]["query"],
        json!(HANDOFF_MESSAGE),
        "the handoff message becomes the next hop's input"
    );
}

// ports tests/unit_tests/core/runner/test_team_runner.py::test_run_agent_team_with_id
#[tokio::test]
async fn a_team_id_is_resolved_through_the_registry() {
    let (team, _calls) = fixture("registered-team");
    let services = common::services();
    services
        .resources
        .register_team(
            team_card("registered-team"),
            Arc::new(move || Arc::clone(&team) as Arc<dyn Team>),
        )
        .unwrap();

    let result = invoke_team(
        services,
        TeamRef::id("registered-team"),
        json!({"query": "hi", "conversation_id": "conv-registered"}),
        RunOptions::default(),
    )
    .await
    .unwrap();

    assert_eq!(result["output"], json!(HANDOFF_SECOND_ANSWER));
}

// ports tests/unit_tests/core/runner/test_team_runner.py::test_run_agent_team_not_found
#[tokio::test]
async fn an_unregistered_team_id_is_not_found() {
    let error = invoke_team(
        common::services(),
        TeamRef::id("missing"),
        json!({"query": "hi"}),
        RunOptions::default(),
    )
    .await
    .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert_eq!(
        error.to_string(),
        "not_found: team 'missing' is not registered"
    );
}

// ---------------------------------------------------------------------------
// Session ownership
// ---------------------------------------------------------------------------

// ports tests/unit_tests/core/runner/test_team_runner.py (owned session teardown)
#[tokio::test]
async fn an_owned_session_is_created_from_the_conversation_id_and_torn_down() {
    let (team, calls) = fixture("handoff-owned");
    invoke_team(
        common::services(),
        TeamRef::of(Arc::clone(&team) as Arc<dyn Team>),
        json!({"query": "hi", "conversation_id": "conv-owned"}),
        RunOptions::default(),
    )
    .await
    .unwrap();

    assert_eq!(
        calls.parent_session_ids(),
        vec![Some("conv-owned".to_owned()), Some("conv-owned".to_owned())],
        "the created session is named from `inputs.conversation_id`"
    );
    assert!(
        team.runtime().team_session("conv-owned").is_none(),
        "an owned team session is unbound during teardown"
    );
}

// ports tests/unit_tests/core/runner/test_team_runner.py (caller-provided session)
#[tokio::test]
async fn a_caller_session_is_reused_and_left_bound() {
    let (team, calls) = fixture("handoff-borrowed");
    let caller = make_team_session(
        &team_card("handoff-borrowed"),
        &json!({"conversation_id": "caller-team"}),
    );
    team.runtime().bind_team_session(Arc::clone(&caller));

    let result = invoke_team(
        common::services(),
        TeamRef::of(Arc::clone(&team) as Arc<dyn Team>),
        json!({"query": "hi", "conversation_id": "conv-ignored"}),
        RunOptions::default().with_team_session(Arc::clone(&caller)),
    )
    .await
    .unwrap();

    assert_eq!(result["output"], json!(HANDOFF_SECOND_ANSWER));
    assert_eq!(
        calls.parent_session_ids(),
        vec![
            Some("caller-team".to_owned()),
            Some("caller-team".to_owned())
        ],
        "the caller's session is the one the members ran in"
    );
    assert!(
        team.runtime().team_session("caller-team").is_some(),
        "the runner leaves a caller-provided team session bound"
    );
    assert!(
        matches!(caller.subscribe().next().await, Some(Ok(StreamEvent::End))),
        "the runner only ends a caller-provided session's stream"
    );
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

// ports tests/unit_tests/core/runner/test_team_runner.py::test_run_agent_team_streaming
#[tokio::test]
async fn streaming_forwards_the_strategy_frames_in_order() {
    let (team, _calls) = fixture("handoff-stream");
    let frames = stream_team(
        common::services(),
        TeamRef::of(Arc::clone(&team) as Arc<dyn Team>),
        json!({"query": "hi", "conversation_id": "conv-team-stream"}),
        RunOptions::default(),
    )
    .collect::<Vec<_>>()
    .await;

    let frames = frames
        .into_iter()
        .map(|frame| frame.unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        frames.len(),
        2,
        "one `message` frame per completed hop: {frames:?}"
    );
    assert!(
        frames.iter().all(|frame| frame["type"] == json!("message")),
        "strategy frames are forwarded verbatim: {frames:?}"
    );
    assert_eq!(frames[0]["payload"]["output"], json!(HANDOFF_FIRST_ANSWER));
    assert_eq!(frames[1]["payload"]["output"], json!(HANDOFF_SECOND_ANSWER));
    assert_eq!(frames[0]["index"], json!(0));
    assert_eq!(frames[1]["index"], json!(0));
}

#[tokio::test]
async fn streaming_tears_down_an_owned_session() {
    let (team, calls) = fixture("handoff-stream-teardown");
    let frames = stream_team(
        common::services(),
        TeamRef::of(Arc::clone(&team) as Arc<dyn Team>),
        json!({"query": "hi", "conversation_id": "conv-stream-teardown"}),
        RunOptions::default(),
    )
    .collect::<Vec<_>>()
    .await;

    assert!(frames.iter().all(Result::is_ok), "no frame failed");
    assert_eq!(
        calls.parent_session_ids(),
        vec![
            Some("conv-stream-teardown".to_owned()),
            Some("conv-stream-teardown".to_owned())
        ]
    );
    assert!(
        team.runtime()
            .team_session("conv-stream-teardown")
            .is_none(),
        "the lease is finished after the last frame"
    );
}

#[tokio::test]
async fn streaming_reports_a_missing_id_as_its_only_item() {
    let frames = stream_team(
        common::services(),
        TeamRef::id("missing"),
        json!({"query": "hi"}),
        RunOptions::default(),
    )
    .collect::<Vec<_>>()
    .await;

    assert_eq!(frames.len(), 1);
    assert_eq!(
        frames[0].as_ref().unwrap_err().to_string(),
        "not_found: team 'missing' is not registered"
    );
}

// ---------------------------------------------------------------------------
// Cancellation and deadlines
// ---------------------------------------------------------------------------

// ports tests/unit_tests/core/runner/test_team_runner.py (cancelled team run)
#[tokio::test]
async fn a_cancelled_token_cancels_the_team_invocation() {
    let token = CancellationToken::new();
    token.cancel();
    let (team, calls) = fixture("cancelled-team");

    let error = invoke_team(
        common::services(),
        TeamRef::of(Arc::clone(&team) as Arc<dyn Team>),
        json!({"query": "hi", "conversation_id": "conv-cancelled"}),
        RunOptions::default().with_cancellation(token),
    )
    .await
    .unwrap_err();

    assert!(error.is_cancelled(), "unexpected: {error}");
    assert_eq!(error.kind(), ErrorKind::Cancelled);
    assert_eq!(error.to_string(), "cancelled");
    assert!(
        calls.inputs().is_empty(),
        "no member ran: the token was already tripped before the invocation"
    );
    assert!(
        team.runtime().team_session("conv-cancelled").is_none(),
        "the lease is finished on the cancelled path too"
    );
}

// ports tests/unit_tests/core/runner/test_team_runner.py (team run deadline)
#[tokio::test]
async fn an_expired_deadline_times_out_the_team_invocation() {
    // The member sleeps far past the deadline, but its sleep future is dropped the
    // moment the deadline fires, so nothing waits on the wall clock.
    let team = delayed_team("deadline-team", Duration::from_secs(60));

    let error = invoke_team(
        common::services(),
        TeamRef::of(team as Arc<dyn Team>),
        json!({"query": "hi", "conversation_id": "conv-deadline"}),
        RunOptions::default().with_timeout(Duration::from_millis(10)),
    )
    .await
    .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Timeout);
    assert_eq!(
        error.to_string(),
        "timeout: run_agent_team timed out after 10ms"
    );
}

#[tokio::test]
async fn a_cancelled_team_run_is_mirrored_as_cancelled() {
    let monitor = Arc::new(RunMonitor::new());
    let hub = Arc::new(InteractionHub::new());
    let mut transitions = monitor.subscribe();
    let services = common::services()
        .with_monitor(Arc::clone(&monitor))
        .with_hitl(Arc::clone(&hub));
    let token = CancellationToken::new();
    token.cancel();
    let (team, _calls) = fixture("cancelled-mirror");

    let error = invoke_team(
        services,
        TeamRef::of(Arc::clone(&team) as Arc<dyn Team>),
        json!({"query": "hi", "conversation_id": "conv-cancelled-mirror"}),
        RunOptions::default().with_cancellation(token),
    )
    .await
    .unwrap_err();
    assert!(error.is_cancelled(), "unexpected: {error}");

    let published: Vec<_> = std::iter::from_fn(|| transitions.try_recv().ok()).collect();
    assert_eq!(
        published
            .iter()
            .map(|event| event.phase)
            .collect::<Vec<_>>(),
        vec![RunPhase::Queued, RunPhase::Running, RunPhase::Cancelled],
        "a cancelled team run finishes its mirror as cancelled, not failed"
    );
    let run_id = published
        .first()
        .expect("the run was admitted")
        .run_id
        .clone();
    let status = monitor
        .snapshot(&run_id)
        .expect("the finished run is retained");
    assert_eq!(status.kind, RunKind::Team);
    assert_eq!(status.phase, RunPhase::Cancelled);
    assert!(status.finished_at.is_some());
    assert_eq!(monitor.list_active(), Vec::new(), "no run stays active");
    assert!(
        hub.is_empty(),
        "the cancelled run's hub binding was unregistered"
    );
}
