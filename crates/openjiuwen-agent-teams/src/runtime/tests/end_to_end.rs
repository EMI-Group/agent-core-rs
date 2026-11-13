//! End-to-end seams: manager → adapters → leader loop → store, plus the global registry.
//!
//! These cases drive a team the way a caller (CLI, MCP, runner) does: ask the
//! [`TeamRuntimeManager`] for a team, then invoke or stream through one of the two foreign agent
//! seams. They assert the observable contract only — the scripted leader answer, exactly one
//! `team.completed` marker, pre-routed deliveries that never reach the leader, a run held open by
//! an open task, and two sessions that stay isolated under one manager.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::Arc;

use futures::StreamExt;
use openjiuwen_agent::{Agent, AgentChunk};
use openjiuwen_common::error::Result;
use openjiuwen_multiagent::Team;
use serde_json::{json, Value};
use tokio::time::timeout;

use crate::runtime::manager::{
    clear_global_registry, global_registry, set_global_registry, try_global_registry,
};
use crate::schema::{
    is_team_event_marker, EventPayload, HarnessState, TaskGraphSpec, TeamTopic, TEAM_COMPLETED,
};
use crate::storage::TeamStore;

use super::support::{blueprint, drain, marker_count, subscribe_observer, wait_until, Fixture};

/// A registry guard that clears the process-global slot when the test ends, panics included.
struct RegistryGuard;

impl Drop for RegistryGuard {
    fn drop(&mut self) {
        clear_global_registry();
    }
}

#[tokio::test]
async fn activate_then_both_agent_seams_reach_the_scripted_leader() {
    let fixture = Fixture::new();
    let team = fixture
        .activate(&blueprint("alpha", "boss", &["dev-1"]), "session-a")
        .await;

    assert_eq!(fixture.manager.pool().len(), 1);
    let live = fixture.manager.live("alpha").expect("alpha is live");
    assert!(Arc::ptr_eq(&live, &team), "the pool holds the built handle");
    let state = fixture
        .manager
        .live_state("alpha")
        .expect("alpha has state");
    assert_eq!(state.team_name, "alpha");
    assert_eq!(state.session_id, "session-a");
    assert_eq!(state.state, HarnessState::Idle);

    let agent: &dyn Agent = team.as_ref();
    let output = agent
        .invoke(json!({"query": "ship it"}), None)
        .await
        .unwrap();
    assert_eq!(output.output, "done: boss");
    assert!(
        fixture.sends("boss").iter().any(|text| text == "ship it"),
        "{:?}",
        fixture.sends("boss")
    );

    let beta = fixture
        .activate(&blueprint("beta", "beta-lead", &[]), "session-b")
        .await;
    let team_seam: &dyn Team = beta.as_ref();
    assert_eq!(team_seam.team_id(), "beta");
    assert_eq!(team_seam.kind(), "agent_teams");
    let value = team_seam
        .invoke(json!({"query": "plan"}), None)
        .await
        .unwrap();
    assert_eq!(value["output"], json!("done: beta-lead"));
    assert!(
        fixture.sends("beta-lead").iter().any(|text| text == "plan"),
        "{:?}",
        fixture.sends("beta-lead")
    );
}

#[tokio::test]
async fn the_stream_ends_with_exactly_one_completed_marker() {
    let fixture = Fixture::new();
    let team = fixture
        .activate(&blueprint("alpha", "boss", &[]), "session-a")
        .await;

    let frames: Vec<Result<AgentChunk>> = drain(team.stream(json!("hello"), None)).await;
    assert!(frames.iter().all(Result::is_ok), "{frames:?}");
    assert!(frames.len() >= 2, "an answer and the marker: {frames:?}");

    let last = frames.last().unwrap().as_ref().unwrap();
    assert_eq!(last.kind, TEAM_COMPLETED);
    assert!(is_team_event_marker(&last.payload));
    assert_eq!(marker_count(&frames), 1, "{frames:?}");

    let earlier = &frames[..frames.len() - 1];
    assert_eq!(
        marker_count(earlier),
        0,
        "no earlier frame is a marker: {frames:?}"
    );
    let first = frames.first().unwrap().as_ref().unwrap();
    assert_eq!(first.payload["source_member"], "boss");
    assert_eq!(first.payload["role"], "leader");
    assert!(team.gate().is_closed(), "the run closes the gate");
}

#[tokio::test]
async fn a_pinned_parity_delivery_never_reaches_the_leader() {
    let fixture = Fixture::new();
    let mut spec = blueprint("counter", "boss", &["ai-counter-1"]);
    spec.enable_hitt = true;
    let team = fixture.activate(&spec, "session-a").await;
    assert!(team.inbox().is_enabled(), "the blueprint enabled HITL");

    let output = team
        .invoke(
            json!({"query": "$human-counter-1 @ai-counter-1 body"}),
            None,
        )
        .await
        .unwrap();
    assert_eq!(
        output.output, "",
        "a routed delivery is not a leader request"
    );

    let queued = team.inbox().drain().await;
    assert_eq!(queued.len(), 1, "the human inbox received the payload");
    assert_eq!(queued[0].1.human_name, "human-counter-1");
    assert_eq!(queued[0].1.target.as_deref(), Some("ai-counter-1"));
    assert_eq!(queued[0].1.body, "body");

    let mailbox = team.backend().unread("ai-counter-1").await.unwrap();
    assert_eq!(
        mailbox.len(),
        1,
        "the delivery was persisted for its target"
    );
    assert_eq!(mailbox[0].from_member_name, "human-counter-1");
    assert_eq!(mailbox[0].content, "body");
    assert!(
        team.backend().unread("boss").await.unwrap().is_empty(),
        "the leader mailbox never sees the routed body"
    );

    let output = team
        .invoke(json!({"query": "please plan"}), None)
        .await
        .unwrap();
    assert_eq!(output.output, "done: boss");
    let sends = fixture.sends("boss");
    assert!(sends.iter().any(|text| text == "please plan"), "{sends:?}");
    assert!(
        !sends.iter().any(|text| text.contains("body")),
        "the delivered body never reached the leader runtime: {sends:?}"
    );
}

#[tokio::test]
async fn an_open_task_keeps_the_run_alive_until_it_is_completed() {
    let fixture = Fixture::new();
    let team = fixture
        .activate(&blueprint("alpha", "boss", &["dev-1"]), "session-a")
        .await;
    let board = Arc::clone(team.backend().tasks());

    let mut stream = team.stream(json!("hello"), None);
    let created = board
        .create_task(TaskGraphSpec {
            title: "ship it".to_owned(),
            content: "do the thing".to_owned(),
            ..TaskGraphSpec::default()
        })
        .await
        .unwrap();
    let task_id = created.task_id().to_owned();
    team.backend().publish_mutation(&created).await.unwrap();
    assert!(board.claim(&task_id, "dev-1").await.unwrap().ok);
    assert!(!board.is_settled().await.unwrap(), "the board is open");

    let mut frames: Vec<Result<AgentChunk>> = Vec::new();
    let deadline = tokio::time::Instant::now() + super::support::WINDOW;
    loop {
        let left = deadline.saturating_duration_since(tokio::time::Instant::now());
        if left.is_zero() {
            break;
        }
        match timeout(left, stream.next()).await {
            Ok(Some(item)) => frames.push(item),
            Ok(None) | Err(_) => break,
        }
    }
    assert_eq!(
        marker_count(&frames),
        0,
        "the run ended while a task was still open: {frames:?}"
    );
    assert!(
        !team.gate().is_closed(),
        "the gate stays open with the board"
    );

    let completed = board.complete(&task_id, "dev-1", "shipped").await.unwrap();
    team.backend().publish_mutation(&completed).await.unwrap();
    assert!(board.is_settled().await.unwrap(), "the board settled");

    let rest = timeout(super::support::BOUND, async {
        let mut rest = Vec::new();
        while let Some(item) = stream.next().await {
            rest.push(item);
        }
        rest
    })
    .await
    .expect("the run did not end after the board settled");
    frames.extend(rest);

    let last = frames.last().unwrap().as_ref().unwrap();
    assert_eq!(last.kind, TEAM_COMPLETED, "{frames:?}");
    assert!(is_team_event_marker(&last.payload));
    assert_eq!(marker_count(&frames), 1, "exactly one marker: {frames:?}");
}

#[tokio::test]
async fn two_teams_in_two_sessions_share_one_manager_without_crosstalk() {
    let fixture = Fixture::new();
    let alpha = fixture
        .activate(
            &blueprint("alpha", "alpha-lead", &["alpha-dev"]),
            "session-a",
        )
        .await;
    let beta = fixture
        .activate(&blueprint("beta", "beta-lead", &["beta-dev"]), "session-b")
        .await;

    assert_eq!(fixture.manager.pool().len(), 2);
    assert!(!Arc::ptr_eq(&alpha, &beta), "two independent runtimes");
    assert_eq!(
        fixture
            .manager
            .live_state("alpha")
            .expect("alpha is live")
            .session_id,
        "session-a"
    );
    assert_eq!(
        fixture
            .manager
            .live_state("beta")
            .expect("beta is live")
            .session_id,
        "session-b"
    );

    let output = alpha
        .invoke(json!({"query": "ship alpha"}), None)
        .await
        .unwrap();
    assert_eq!(output.output, "done: alpha-lead");
    let output = beta
        .invoke(json!({"query": "ship beta"}), None)
        .await
        .unwrap();
    assert_eq!(output.output, "done: beta-lead");

    assert_eq!(fixture.sends("alpha-lead"), vec!["ship alpha".to_owned()]);
    assert_eq!(fixture.sends("beta-lead"), vec!["ship beta".to_owned()]);

    let store = Arc::clone(&fixture.store) as Arc<dyn TeamStore>;
    assert!(store.team("alpha").await.unwrap().is_some());
    assert!(store.team("beta").await.unwrap().is_some());
    assert!(store.member("alpha", "alpha-dev").await.unwrap().is_some());
    assert!(store.member("beta", "beta-dev").await.unwrap().is_some());
    assert!(
        store.member("alpha", "beta-dev").await.unwrap().is_none(),
        "roster rows are team-scoped"
    );
}

#[tokio::test]
async fn the_global_registry_activates_and_runs_a_team() {
    clear_global_registry();
    let _guard = RegistryGuard;
    let fixture = Fixture::new();
    set_global_registry(Arc::clone(&fixture.manager));

    let installed = try_global_registry().expect("the registry is installed");
    assert!(Arc::ptr_eq(&installed, &fixture.manager));
    assert!(Arc::ptr_eq(&global_registry(), &fixture.manager));

    let team = installed
        .activate(&blueprint("alpha", "boss", &[]), "session-a", &Value::Null)
        .await
        .unwrap();
    let output = team.invoke(json!({"query": "hello"}), None).await.unwrap();
    assert_eq!(output.output, "done: boss");
    assert!(installed.live("alpha").is_some());

    clear_global_registry();
    assert!(try_global_registry().is_none(), "the slot is cleared again");
}

#[tokio::test]
async fn a_member_status_change_reaches_the_team_topic() {
    let fixture = Fixture::new();
    let team = fixture
        .activate(&blueprint("alpha", "boss", &["dev-1"]), "session-a")
        .await;
    let recorder = subscribe_observer(team.backend().messager(), &[TeamTopic::Team]);

    team.spawn_member("dev-1").await.unwrap();
    wait_until("the member status event", || {
        super::support::recorded(&recorder)
            .iter()
            .any(|event| event.event_type == "member_status_changed")
    })
    .await;

    let events = super::support::recorded(&recorder);
    let started = events
        .iter()
        .find(|event| event.event_type == "member_status_changed")
        .expect("the status change was published");
    assert_eq!(started.sender_id, "alpha");
    assert_eq!(started.payload["member_name"], "dev-1");
    assert_eq!(started.payload["status"], "ready");
}

#[tokio::test]
async fn a_settled_run_publishes_one_team_completed_event_on_the_bus() {
    let fixture = Fixture::new();
    let team = fixture
        .activate(&blueprint("alpha", "boss", &[]), "session-a")
        .await;
    let recorder = subscribe_observer(team.backend().messager(), &[TeamTopic::Team]);

    let frames = drain(team.stream(json!("hello"), None)).await;
    assert!(frames.iter().all(Result::is_ok), "{frames:?}");

    let last = frames.last().unwrap().as_ref().unwrap();
    assert_eq!(last.kind, TEAM_COMPLETED);
    assert!(is_team_event_marker(&last.payload));
    assert_eq!(marker_count(&frames), 1, "exactly one marker: {frames:?}");

    // The bus carries member/board events too, so count only the settlement.
    let completed: Vec<_> = super::support::recorded(&recorder)
        .into_iter()
        .filter(|event| event.event_type == "team_completed")
        .collect();
    assert_eq!(completed.len(), 1, "one settlement event: {completed:?}");
    assert_eq!(completed[0].sender_id, "alpha");
    match EventPayload::from_message(&completed[0]).unwrap() {
        EventPayload::TeamCompleted(payload) => {
            assert_eq!(payload.team_name, "alpha");
            assert_eq!(payload.reason.as_deref(), Some("settled"));
        }
        other => panic!("expected team_completed, got {other:?}"),
    }
}
