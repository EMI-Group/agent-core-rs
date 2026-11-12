//! End-to-end team runs through the public API: activate → invoke/stream → settle.
//!
//! The suite pins what a caller (CLI, MCP, runner) observes when it drives a live team:
//!
//! 1. a run reaches the scripted leader and returns its answer, having sent the caller's query;
//! 2. a run with no leader query settles without a turn (activation never consumes the inputs);
//! 3. the stream ends with exactly one `team.completed` marker, last, after the answer frame;
//! 4. plain text and a nested `{"input":{"query":…}}` object both become the leader query;
//! 5. an open task keeps the run open until the board settles;
//! 6. `interact` routes to a live team, reports `not_active`/`gate_closed`, and stops routing once
//!    the team is stopped;
//! 7. `TeamAgent::interact` queues while a run is open and the handle is idle after it settles;
//! 8. `snapshot()` reports the sorted roster, through the handle and the manager;
//! 9. `stop_team` removes the team and a second stop reports `false`.
//!
//! Everything is offline and deterministic: the `support` fakes answer every member input with one
//! scripted observation triple, and every wait is bounded by [`support::BOUND`].
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod support;

use std::time::Duration;

use futures::stream::BoxStream;
use futures::StreamExt;
use openjiuwen_agent::AgentChunk;
use openjiuwen_agent_teams::interaction::{InteractPayload, OperatorMessage};
use openjiuwen_agent_teams::runtime::parse_initial_input;
use openjiuwen_agent_teams::schema::{
    is_team_event_marker, HarnessState, MemberStatus, TaskGraphSpec, TaskStatus, TeamRole,
    TeamSnapshot, TEAM_COMPLETED,
};
use openjiuwen_common::error::Result;
use serde_json::{json, Value};
use tokio::time::timeout;

use support::{collect, marker_count, wait_until, Call, Fixture, BOUND};

/// How long a run that must stay open is observed before it is allowed to settle.
///
/// Generous on purpose: the assertion is "the stream did not end", not a timing budget.
const OBSERVATION_WINDOW: Duration = Duration::from_millis(175);

/// Builds an operator (external-user) payload addressed to the team, not to one member.
fn operator(body: &str) -> InteractPayload {
    InteractPayload::Operator(OperatorMessage {
        target: None,
        body: body.to_owned(),
        sender: None,
    })
}

/// Returns the snapshot's member names, in the order the snapshot reports them.
fn member_names(snapshot: &TeamSnapshot) -> Vec<String> {
    snapshot
        .members
        .iter()
        .map(|member| member.member_name.clone())
        .collect()
}

/// Drains a stream that is still open, panicking when it does not end within [`BOUND`].
async fn drain_open_stream(
    stream: &mut BoxStream<'static, Result<AgentChunk>>,
) -> Vec<Result<AgentChunk>> {
    timeout(BOUND, async {
        let mut items = Vec::new();
        while let Some(item) = stream.next().await {
            items.push(item);
        }
        items
    })
    .await
    .expect("the run did not end within the bound")
}

#[tokio::test]
async fn invoke_settles_with_the_scripted_leader_answer() {
    let fixture = Fixture::new();
    fixture
        .factory
        .answers_with("team_leader", "the leader answered once");
    let spec = support::sample_spec();
    let agent = fixture.activate(&spec, "session-a").await;

    assert!(
        fixture
            .harness("team_leader")
            .calls()
            .contains(&Call::Start),
        "activation starts the leader cycle: {:?}",
        fixture.harness("team_leader").calls()
    );

    let output = agent.invoke(json!("ship the release"), None).await.unwrap();

    assert_eq!(output.output, "the leader answered once");
    assert_eq!(
        fixture.sends("team_leader"),
        vec!["ship the release".to_owned()],
        "exactly one leader turn, carrying the caller's query"
    );
}

#[tokio::test]
async fn a_null_input_settles_without_a_leader_turn() {
    let fixture = Fixture::new();
    let spec = support::sample_spec();
    let agent = fixture.activate(&spec, "session-a").await;

    // Activation never consumes the request inputs (`TeamRuntimeManager::activate` only logs
    // them), and a `Null` request carries no query, so the run settles on an empty answer.
    let output = agent.invoke(Value::Null, None).await.unwrap();

    assert_eq!(output.output, "");
    assert!(
        fixture.sends("team_leader").is_empty(),
        "no query means no turn: {:?}",
        fixture.sends("team_leader")
    );
    assert_eq!(agent.state(), HarnessState::Idle);
}

#[tokio::test]
async fn the_stream_ends_with_exactly_one_completed_marker() {
    let fixture = Fixture::new();
    fixture
        .factory
        .answers_with("team_leader", "streamed answer");
    let spec = support::sample_spec();
    let agent = fixture.activate(&spec, "session-a").await;

    let frames = collect(agent.stream(json!("stream please"), None)).await;
    assert!(frames.iter().all(Result::is_ok), "{frames:?}");
    assert!(
        frames.len() >= 2,
        "at least the answer frame and the marker: {frames:?}"
    );

    let last = frames.last().unwrap().as_ref().unwrap();
    assert_eq!(last.kind, TEAM_COMPLETED);
    assert!(is_team_event_marker(&last.payload), "{:?}", last.payload);
    assert_eq!(
        last.payload["chunk"]["payload"]["event_type"],
        json!(TEAM_COMPLETED),
        "the marker frame carries the team-completed event type"
    );
    assert_eq!(marker_count(&frames), 1, "exactly one marker: {frames:?}");

    let earlier = &frames[..frames.len() - 1];
    assert_eq!(
        marker_count(earlier),
        0,
        "no earlier frame is a team marker: {frames:?}"
    );

    let answer_frame = earlier.iter().any(|item| {
        let payload = &item.as_ref().expect("the frame is an answer").payload;
        payload["source_member"] == json!("team_leader")
            && payload["chunk"]["type"] == json!("llm_output")
            && payload["chunk"]["payload"]["content"] == json!("streamed answer")
    });
    assert!(
        answer_frame,
        "one frame carries the leader answer text: {frames:?}"
    );
    assert!(agent.gate().is_closed(), "the run closes the gate");
}

#[tokio::test]
async fn plain_text_and_nested_objects_become_the_leader_query() {
    let roster: Vec<String> = Vec::new();
    let plain = parse_initial_input(&json!("deploy the release"), &roster);
    assert_eq!(plain.leader_query.as_deref(), Some("deploy the release"));
    assert_eq!(plain.deliveries.len(), 0, "plain text routes nothing");

    let nested = parse_initial_input(&json!({"input": {"query": "nested deploy"}}), &roster);
    assert_eq!(nested.leader_query.as_deref(), Some("nested deploy"));
    assert_eq!(nested.deliveries.len(), 0, "a nested query routes nothing");

    let fixture = Fixture::new();
    let spec = support::sample_spec();
    let agent = fixture.activate(&spec, "session-a").await;

    agent
        .invoke(json!("deploy the release"), None)
        .await
        .unwrap();
    agent
        .invoke(json!({"input": {"query": "nested deploy"}}), None)
        .await
        .unwrap();

    assert_eq!(
        fixture.sends("team_leader"),
        vec!["deploy the release".to_owned(), "nested deploy".to_owned()],
        "each request reaches the leader verbatim, in order"
    );
}

#[tokio::test]
async fn an_open_task_keeps_the_stream_open_until_the_board_settles() {
    let fixture = Fixture::new();
    fixture
        .factory
        .answers_with("team_leader", "waiting on the board");
    fixture.factory.hanging_for("team_leader");
    let spec = support::sample_spec();
    let agent = fixture.activate(&spec, "session-a").await;

    let mut stream = agent.stream(json!("build the thing"), None);
    wait_until("the leader's first turn", || {
        fixture
            .sends("team_leader")
            .iter()
            .any(|text| text == "build the thing")
    })
    .await;

    let created = agent
        .create_task(TaskGraphSpec {
            title: "ship it".to_owned(),
            content: "do the thing".to_owned(),
            ..TaskGraphSpec::default()
        })
        .await
        .unwrap();
    let task_id = created.task_id.clone();
    // The creation events steer the leader; wait for that batch so no late send can flip the
    // harness back to `Running` after the board settles.
    wait_until("the leader's task steering batch", || {
        fixture
            .sends("team_leader")
            .iter()
            .any(|text| text.contains("task "))
    })
    .await;

    let mut frames: Vec<Result<AgentChunk>> = Vec::new();
    let mut ended = false;
    let deadline = tokio::time::Instant::now() + OBSERVATION_WINDOW;
    loop {
        let left = deadline.saturating_duration_since(tokio::time::Instant::now());
        if left.is_zero() {
            break;
        }
        match timeout(left, stream.next()).await {
            Ok(Some(item)) => frames.push(item),
            Ok(None) => {
                ended = true;
                break;
            }
            Err(_) => break,
        }
    }
    assert!(!ended, "the stream ended while a task was still open");
    assert_eq!(
        marker_count(&frames),
        0,
        "the run completed with an open task: {frames:?}"
    );
    assert!(
        !agent.gate().is_closed(),
        "the gate stays open with the board"
    );

    let claimed = agent
        .backend()
        .tasks()
        .claim(&task_id, "team_leader")
        .await
        .unwrap();
    assert!(claimed.ok, "{claimed:?}");
    let completed = agent
        .backend()
        .tasks()
        .complete(&task_id, "team_leader", "shipped")
        .await
        .unwrap();
    assert_eq!(completed.task.status, TaskStatus::Completed);
    assert!(
        agent.backend().tasks().is_settled().await.unwrap(),
        "the board settled"
    );

    fixture.harness("team_leader").set_state(HarnessState::Idle);
    frames.extend(drain_open_stream(&mut stream).await);

    let last = frames.last().unwrap().as_ref().unwrap();
    assert_eq!(last.kind, TEAM_COMPLETED, "{frames:?}");
    assert!(is_team_event_marker(&last.payload));
    assert_eq!(marker_count(&frames), 1, "exactly one marker: {frames:?}");
}

#[tokio::test]
async fn interact_is_queued_on_a_live_team_and_rejected_after_stop() {
    let fixture = Fixture::new();
    let spec = support::sample_spec();
    let agent = fixture.activate(&spec, "session-a").await;

    let queued = fixture
        .manager
        .interact("alpha", operator("what is the status?"))
        .await;
    assert!(queued.ok, "{queued:?}");
    let mailbox = agent.backend().unread("team_leader").await.unwrap();
    assert_eq!(mailbox.len(), 1, "the payload reached the leader mailbox");
    assert_eq!(mailbox[0].content, "what is the status?");

    let unknown = fixture.manager.interact("ghost", operator("hello")).await;
    assert!(!unknown.ok, "{unknown:?}");
    assert_eq!(unknown.reason.as_deref(), Some("not_active"));

    // The documented pair from the manager's own tests: a closed gate is `gate_closed` while the
    // team is live, and a stopped team is `not_active`.
    agent.gate().close();
    let closed = fixture
        .manager
        .interact("alpha", operator("hello again"))
        .await;
    assert!(!closed.ok, "{closed:?}");
    assert_eq!(closed.reason.as_deref(), Some("gate_closed"));

    assert!(fixture
        .manager
        .stop_team("alpha", "session-a")
        .await
        .unwrap());
    let stopped = fixture
        .manager
        .interact("alpha", operator("are you there?"))
        .await;
    assert!(!stopped.ok, "{stopped:?}");
    assert_eq!(
        stopped.reason.as_deref(),
        Some("not_active"),
        "the stopped team is no longer live"
    );
}

#[tokio::test]
async fn handle_interact_queues_while_the_run_is_open_and_the_team_settles_idle() {
    let fixture = Fixture::new();
    fixture.factory.answers_with("team_leader", "held open");
    fixture.factory.hanging_for("team_leader");
    let spec = support::sample_spec();
    let agent = fixture.activate(&spec, "session-a").await;

    let mut stream = agent.stream(json!("start"), None);
    wait_until("the leader's first turn", || {
        fixture
            .sends("team_leader")
            .iter()
            .any(|text| text == "start")
    })
    .await;

    let queued = agent.interact(operator("steer the team")).await;
    assert!(queued.ok, "{queued:?}");
    assert_eq!(agent.gate().len(), 1, "the payload is queued on the gate");
    assert!(!agent.gate().is_closed(), "the run is still open");

    fixture.harness("team_leader").set_state(HarnessState::Idle);
    let rest = drain_open_stream(&mut stream).await;
    let last = rest.last().unwrap().as_ref().unwrap();
    assert_eq!(last.kind, TEAM_COMPLETED, "{rest:?}");
    assert_eq!(marker_count(&rest), 1, "exactly one marker: {rest:?}");
    assert_eq!(agent.state(), HarnessState::Idle, "the team settled idle");
    assert!(agent.gate().is_closed(), "the run closed the gate");
}

#[tokio::test]
async fn snapshot_lists_the_sorted_roster() {
    let fixture = Fixture::new();
    let spec = support::sample_spec();
    let agent = fixture.activate(&spec, "session-a").await;

    let snapshot = agent.snapshot().await;
    assert_eq!(snapshot.team_name, "alpha");
    assert_eq!(snapshot.state, HarnessState::Idle);
    assert_eq!(
        member_names(&snapshot),
        vec![
            "dev-1".to_owned(),
            "team_leader".to_owned(),
            "teammate".to_owned()
        ],
        "the roster is sorted by member name"
    );
    assert!(snapshot.tasks.is_empty(), "the board starts empty");

    let statuses: Vec<(&str, MemberStatus)> = snapshot
        .members
        .iter()
        .map(|member| (member.member_name.as_str(), member.status))
        .collect();
    assert_eq!(
        statuses,
        vec![
            ("dev-1", MemberStatus::Unstarted),
            ("team_leader", MemberStatus::Ready),
            ("teammate", MemberStatus::Unstarted),
        ],
        "activation starts the leader only"
    );
    let leader = snapshot
        .members
        .iter()
        .find(|member| member.member_name == "team_leader")
        .expect("the leader is on the roster");
    assert_eq!(leader.role, TeamRole::Leader);

    let managed = fixture
        .manager
        .snapshot("alpha")
        .await
        .expect("alpha is live");
    assert_eq!(member_names(&managed), member_names(&snapshot));
    assert!(fixture.manager.snapshot("ghost").await.is_none());
}

#[tokio::test]
async fn stop_team_removes_the_team_and_a_second_stop_reports_false() {
    let fixture = Fixture::new();
    let spec = support::sample_spec();
    let agent = fixture.activate(&spec, "session-a").await;

    assert!(
        !fixture
            .manager
            .stop_team("alpha", "session-b")
            .await
            .unwrap(),
        "another session cannot stop the owner's team"
    );
    assert!(fixture.manager.live("alpha").is_some());

    assert!(fixture
        .manager
        .stop_team("alpha", "session-a")
        .await
        .unwrap());
    assert!(fixture.manager.live("alpha").is_none());
    assert!(fixture.manager.live_state("alpha").is_none());
    assert!(!agent.is_started(), "the handle was torn down");
    assert!(
        !fixture
            .manager
            .stop_team("alpha", "session-a")
            .await
            .unwrap(),
        "stopping an already stopped team is a no-op success"
    );
}
