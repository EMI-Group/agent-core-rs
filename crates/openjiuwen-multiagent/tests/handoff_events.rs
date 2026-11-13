#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Integration tests for the events a handoff chain reports on the team bus.
//!
//! A chain emits one `handoff_started` per dispatched hop, one
//! `handoff_completed` when a hop resolves it, an `error` event with the
//! `handoff` operation when a member fails, and `interrupted: true` when the hop
//! forwarded a team interrupt. Nothing reaches a bus whose runtime was dropped.
//!
//! Every case is offline and deterministic: members are in-process closures and
//! no timer runs.

mod common;

use std::sync::Arc;

use openjiuwen_agent::{Agent, AgentOutput};
use openjiuwen_common::error::{Error, ErrorKind};
use openjiuwen_multiagent::events::{TeamEvent, TeamEventFilter, TeamEventKind};
use openjiuwen_multiagent::{
    handoff_team, ContainerAgent, HandoffConfig, HandoffOrchestrator, HandoffRegistry,
    HandoffRequest, HandoffTeamConfig, RuntimeConfig, Team, TeamCard, TeamRuntime,
};
use openjiuwen_session::{AgentTeamSession, AgentTeamSessionOptions, Session};
use parking_lot::Mutex;
use serde_json::json;

use common::{agent_card, handoff_output, provider, FakeAgent};

/// A handoff team with a unique card id, so no two cases share bus topics.
fn team(name: &str) -> openjiuwen_multiagent::BaseTeam {
    handoff_team(TeamCard::new(name), HandoffTeamConfig::default())
}

/// Registers a member that always asks to hand off to `target`.
fn handoff_member(team: &openjiuwen_multiagent::BaseTeam, name: &str, target: &str) {
    let target = target.to_owned();
    let agent = FakeAgent::new(agent_card(name), move |_inputs, _session| {
        Ok(handoff_output("done", &target, None, Some("route")))
    });
    team.add_agent(agent_card(name), provider(agent)).unwrap();
}

/// Registers a member that answers verbatim.
fn answering_member(team: &openjiuwen_multiagent::BaseTeam, name: &str, answer: &str) {
    let answer = answer.to_owned();
    let agent = FakeAgent::new(agent_card(name), move |_inputs, _session| {
        Ok(AgentOutput::answer(answer.clone()))
    });
    team.add_agent(agent_card(name), provider(agent)).unwrap();
}

#[tokio::test]
async fn a_two_hop_chain_reports_each_hop_and_one_completion() {
    let team = team("handoff-events-two-hop");
    handoff_member(&team, "a", "b");
    answering_member(&team, "b", "b is done");
    let events = team.runtime().events();

    let result = team
        .invoke(
            json!({"query": "hi", "conversation_id": "events-two-hop"}),
            None,
        )
        .await
        .unwrap();
    assert_eq!(result["output"], "b is done");

    let started = events.history_of(TeamEventKind::HandoffStarted);
    assert_eq!(started.len(), 2, "one started event per dispatched hop");
    assert_eq!(started[0].peer_id.as_deref(), Some("a"));
    assert_eq!(started[1].peer_id.as_deref(), Some("b"));
    assert_ne!(
        started[0].peer_id, started[1].peer_id,
        "each hop names the member it runs"
    );
    assert_eq!(started[0].agent_id.as_deref(), Some("a"));
    assert_eq!(
        started[1].agent_id.as_deref(),
        Some("a"),
        "the previous member is the source of the second hop"
    );
    assert_eq!(started[0].detail["hop"], 1);
    assert_eq!(started[1].detail["hop"], 2);
    for event in &started {
        assert_eq!(event.session_id.as_deref(), Some("events-two-hop"));
        assert_eq!(event.team_id, team.runtime().team_id());
    }

    let completed = events.history_of(TeamEventKind::HandoffCompleted);
    assert_eq!(completed.len(), 1, "the chain completes exactly once");
    assert_eq!(completed[0].agent_id.as_deref(), Some("b"));
    assert_eq!(completed[0].peer_id.as_deref(), Some("b"));
    assert_eq!(completed[0].detail["handoffs"], 1);
    assert_eq!(completed[0].session_id.as_deref(), Some("events-two-hop"));
    assert!(completed[0].detail.get("interrupted").is_none());
    assert_eq!(events.history_of(TeamEventKind::Error).len(), 0);

    assert!(
        started[0].sequence < started[1].sequence,
        "hops are reported in order"
    );
    assert!(
        started[1].sequence < completed[0].sequence,
        "the completion follows the last hop"
    );
}

#[tokio::test]
async fn a_failing_member_reports_a_handoff_error_and_the_chain_still_resolves() {
    let team = team("handoff-events-error");
    let agent = FakeAgent::new(agent_card("a"), |_inputs, _session| {
        Err(Error::validation("target exploded"))
    });
    team.add_agent(agent_card("a"), provider(agent)).unwrap();

    let error = team
        .invoke(
            json!({"query": "hi", "conversation_id": "events-error"}),
            None,
        )
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("target exploded"), "{error}");

    let events = team.runtime().events();
    let errors = events.history_of(TeamEventKind::Error);
    assert_eq!(errors.len(), 1, "the failure is reported once");
    assert_eq!(errors[0].agent_id.as_deref(), Some("a"));
    assert_eq!(errors[0].session_id.as_deref(), Some("events-error"));
    assert_eq!(errors[0].detail["operation"], "handoff");
    assert!(
        errors[0].detail["error"]
            .as_str()
            .unwrap_or_default()
            .contains("target exploded"),
        "the event carries the failure message: {}",
        errors[0].detail
    );
    assert_eq!(events.history_of(TeamEventKind::HandoffStarted).len(), 1);
    assert_eq!(
        events.history_of(TeamEventKind::HandoffCompleted).len(),
        0,
        "a failed hop does not report a completion"
    );
}

#[tokio::test]
async fn an_interrupted_hop_marks_the_completion() {
    let team = team("handoff-events-interrupt");
    let agent = FakeAgent::new(agent_card("a"), |_inputs, _session| {
        Ok(AgentOutput::interrupt(Vec::new(), Vec::new()))
    });
    team.add_agent(agent_card("a"), provider(agent)).unwrap();

    let result = team
        .invoke(
            json!({"query": "hi", "conversation_id": "events-interrupt"}),
            None,
        )
        .await
        .unwrap();
    assert_eq!(result["result_type"], "interrupt");

    let events = team.runtime().events();
    let completed = events.history_of(TeamEventKind::HandoffCompleted);
    assert_eq!(completed.len(), 1);
    assert_eq!(completed[0].detail["interrupted"], true);
    assert_eq!(completed[0].detail["handoffs"], 0);
    assert_eq!(completed[0].agent_id.as_deref(), Some("a"));
}

#[tokio::test]
async fn a_started_filter_sees_only_started_events() {
    let team = team("handoff-events-filter");
    handoff_member(&team, "a", "b");
    answering_member(&team, "b", "b is done");

    let seen: Arc<Mutex<Vec<TeamEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&seen);
    team.runtime().events().subscribe(
        TeamEventFilter::new().kind(TeamEventKind::HandoffStarted),
        Arc::new(move |event: &TeamEvent| sink.lock().push(event.clone())),
    );

    team.invoke(
        json!({"query": "hi", "conversation_id": "events-filter"}),
        None,
    )
    .await
    .unwrap();

    let seen = seen.lock().clone();
    assert_eq!(seen.len(), 2, "the chain dispatched two hops");
    assert!(seen
        .iter()
        .all(|event| event.kind == TeamEventKind::HandoffStarted));
}

#[tokio::test]
async fn a_dropped_runtime_emits_nothing() {
    let runtime = TeamRuntime::new(RuntimeConfig::new().with_team_id("events-dropped"));
    let events = runtime.events();
    let team_session = Arc::new(AgentTeamSession::with_options(AgentTeamSessionOptions {
        session_id: Some("dropped-session".to_owned()),
        team_id: Some("events-dropped".to_owned()),
        ..AgentTeamSessionOptions::default()
    }));
    runtime.bind_team_session(Arc::clone(&team_session));
    let members = vec!["a".to_owned()];
    let coordinator = Arc::new(HandoffOrchestrator::new(
        "a",
        &members,
        &HandoffConfig::default(),
    ));
    let registry = Arc::new(HandoffRegistry::new());
    registry.insert("dropped-session", Arc::clone(&coordinator));
    let container = ContainerAgent::new(
        agent_card("a"),
        provider(FakeAgent::echoing("a")),
        Vec::new(),
        "events-dropped".to_owned(),
        "__handoff_ep_events-dropped_a".to_owned(),
        Arc::downgrade(&runtime),
        Arc::clone(&registry),
    );
    let child: Arc<dyn Session> = Arc::new(team_session.create_agent_session("endpoint"));
    drop(runtime);

    let request = HandoffRequest::new(json!({"query": "hi"}), Vec::new()).to_value();
    let error = container.invoke(request, Some(child)).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::External);
    assert_eq!(
        events.history().len(),
        0,
        "a dropped runtime reports no events: {:?}",
        events.history()
    );
    assert!(
        coordinator.complete(json!({"late": true})).await,
        "the hop never reached the coordinator"
    );
}
