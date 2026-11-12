#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Integration tests for the handoff strategy.
//!
//! The suite exercises the public handoff surface from the crate root: the route
//! graph and coordinator semantics, save/restore, the three signal locators, the
//! `transfer_to_<id>` tool, the request wire form, and full chains driven through
//! `Team::invoke`/`Team::stream` with fake members.
//!
//! Every case is offline and deterministic: members are in-process closures,
//! sessions are local, and the only timers are the explicit short deadlines of
//! the timeout cases.
mod common;

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use futures::{FutureExt, StreamExt};
use openjiuwen_agent::{Agent, AgentOutput, Observation};
use openjiuwen_common::error::{Error, ErrorKind};
use openjiuwen_foundation::tool::{Tool, ToolContext};
use openjiuwen_multiagent::{
    extract_handoff_signal, extract_handoff_signal_from_observations,
    extract_handoff_signal_from_session, extract_interrupt_signal, flush_team_session,
    handoff_team, make_team_session, HandoffConfig, HandoffHistoryEntry, HandoffOrchestrator,
    HandoffRequest, HandoffRoute, HandoffTeamConfig, HandoffTool, Team, TeamCard, TeamConfig,
    TerminationCondition, ToolHost, COORDINATOR_STATE_KEY, HANDOFF_MESSAGE_KEY, HANDOFF_REASON_KEY,
    HANDOFF_TARGET_KEY, HISTORY_STATE_KEY,
};
use openjiuwen_session::{AgentSession, AgentTeamSession, Session, StreamEvent};
use parking_lot::Mutex;
use serde_json::{json, Value};

use common::{
    agent_card, handoff_output, host_provider, provider, AsyncFakeAgent, FakeAgent,
    RecordingToolHost,
};

/// The member set every coordinator case uses.
fn members() -> Vec<String> {
    vec!["a".to_owned(), "b".to_owned(), "c".to_owned()]
}

/// Builds a handoff config with `max_handoffs` and default everything else.
fn capped(max_handoffs: usize) -> HandoffConfig {
    HandoffConfig {
        max_handoffs,
        ..HandoffConfig::default()
    }
}

/// Collects a set into a sorted `Vec`.
fn sorted(set: &BTreeSet<String>) -> Vec<String> {
    set.iter().cloned().collect()
}

// ---------------------------------------------------------------------------
// Route graph
// ---------------------------------------------------------------------------

#[test]
fn route_graph_defaults_to_a_full_mesh_minus_self() {
    let graph = HandoffOrchestrator::build_route_graph(&members(), &[]);

    assert_eq!(graph.len(), 3, "every member is a source");
    assert_eq!(sorted(&graph["a"]), ["b", "c"]);
    assert_eq!(sorted(&graph["b"]), ["a", "c"]);
    assert_eq!(sorted(&graph["c"]), ["a", "b"]);
}

#[test]
fn route_graph_honours_explicit_routes_only() {
    let routes = vec![HandoffRoute::new("a", "b"), HandoffRoute::new("a", "ghost")];
    let graph = HandoffOrchestrator::build_route_graph(&members(), &routes);

    assert_eq!(
        sorted(&graph["a"]),
        ["b", "ghost"],
        "a target need not be a member"
    );
    assert!(graph["b"].is_empty(), "no implicit mesh once routes exist");
    assert!(graph["c"].is_empty());
}

#[test]
fn a_source_without_routes_has_an_empty_target_set() {
    let routes = vec![HandoffRoute::new("c", "a")];
    let graph = HandoffOrchestrator::build_route_graph(&members(), &routes);

    assert!(graph["a"].is_empty());
    assert!(graph["b"].is_empty());
    assert_eq!(sorted(&graph["c"]), ["a"]);
}

// ---------------------------------------------------------------------------
// Coordinator semantics
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_zero_hop_cap_blocks_the_first_handoff() {
    let orchestrator = HandoffOrchestrator::new("a", &members(), &capped(0));

    assert!(
        !orchestrator.request_handoff("b", None).await.unwrap(),
        "the cap is checked before the count moves"
    );
    assert_eq!(orchestrator.handoff_count(), 0);
    assert_eq!(orchestrator.current_agent_id(), "a");
}

#[tokio::test]
async fn a_ready_termination_condition_blocks_the_handoff() {
    let condition: TerminationCondition =
        Arc::new(|snapshot| async move { Ok(snapshot.handoff_count == 0) }.boxed());
    let config = HandoffConfig {
        termination_condition: Some(condition),
        ..HandoffConfig::default()
    };
    let orchestrator = HandoffOrchestrator::new("a", &members(), &config);

    assert!(!orchestrator.request_handoff("b", None).await.unwrap());
    assert_eq!(orchestrator.handoff_count(), 0);
    assert_eq!(orchestrator.current_agent_id(), "a");
}

#[tokio::test]
async fn a_false_termination_condition_allows_the_handoff() {
    let condition: TerminationCondition =
        Arc::new(|_snapshot| futures::future::ready(Ok(false)).boxed());
    let config = HandoffConfig {
        termination_condition: Some(condition),
        ..HandoffConfig::default()
    };
    let orchestrator = HandoffOrchestrator::new("a", &members(), &config);

    assert!(orchestrator.request_handoff("b", None).await.unwrap());
    assert_eq!(orchestrator.handoff_count(), 1);
    assert_eq!(orchestrator.current_agent_id(), "b");
}

#[tokio::test]
async fn an_unknown_target_is_blocked() {
    let orchestrator = HandoffOrchestrator::new("a", &members(), &HandoffConfig::default());

    assert!(
        !orchestrator.request_handoff("ghost", None).await.unwrap(),
        "a target outside the route graph is rejected"
    );
    assert_eq!(orchestrator.handoff_count(), 0);
    assert_eq!(orchestrator.current_agent_id(), "a");
}

#[tokio::test]
async fn completion_is_first_wins() {
    let orchestrator = HandoffOrchestrator::new("a", &members(), &HandoffConfig::default());

    assert!(orchestrator.complete(json!({"answer": 7})).await);
    assert!(!orchestrator.complete(json!({"answer": 8})).await);
    assert!(!orchestrator.error(Error::validation("late")).await);
    assert_eq!(
        orchestrator
            .wait(Some(Duration::from_millis(100)))
            .await
            .unwrap(),
        json!({"answer": 7})
    );
}

#[tokio::test]
async fn an_error_resolution_wins_and_surfaces_through_wait() {
    let orchestrator = HandoffOrchestrator::new("a", &members(), &HandoffConfig::default());

    assert!(orchestrator.error(Error::validation("boom")).await);
    assert!(!orchestrator.complete(json!({"answer": 1})).await);

    let error = orchestrator.wait(None).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(error.message(), "boom");
}

#[tokio::test]
async fn wait_times_out_without_a_resolution() {
    let orchestrator = HandoffOrchestrator::new("a", &members(), &HandoffConfig::default());

    let error = orchestrator
        .wait(Some(Duration::from_millis(10)))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Timeout);
}

// ---------------------------------------------------------------------------
// Save and restore
// ---------------------------------------------------------------------------

#[tokio::test]
async fn coordinator_state_round_trips_through_the_team_session() {
    let orchestrator = HandoffOrchestrator::new("a", &members(), &HandoffConfig::default());
    assert!(orchestrator
        .request_handoff("b", Some("needs billing"))
        .await
        .unwrap());

    let session = Arc::new(AgentTeamSession::new("team-save"));
    orchestrator.save_to_session(&session).await.unwrap();

    let snapshot = session.get_state(COORDINATOR_STATE_KEY).await.unwrap();
    assert_eq!(
        snapshot,
        Some(json!({"current_agent_id": "b", "handoff_count": 1}))
    );

    let restored = HandoffOrchestrator::restore_from_session(
        snapshot.as_ref(),
        "a",
        &members(),
        &HandoffConfig::default(),
    )
    .unwrap();
    assert_eq!(restored.current_agent_id(), "b");
    assert_eq!(restored.handoff_count(), 1);
}

#[tokio::test]
async fn an_absent_or_partial_snapshot_starts_from_the_start_agent() {
    let absent =
        HandoffOrchestrator::restore_from_session(None, "a", &members(), &HandoffConfig::default())
            .unwrap();
    assert_eq!(absent.current_agent_id(), "a");
    assert_eq!(absent.handoff_count(), 0);

    let partial = json!({"current_agent_id": "b"});
    let restored = HandoffOrchestrator::restore_from_session(
        Some(&partial),
        "a",
        &members(),
        &HandoffConfig::default(),
    )
    .unwrap();
    assert_eq!(
        restored.current_agent_id(),
        "a",
        "a partial snapshot is discarded entirely"
    );
    assert_eq!(restored.handoff_count(), 0);
}

// ---------------------------------------------------------------------------
// Signal extraction
// ---------------------------------------------------------------------------

/// Builds an observation whose content is `payload`.
fn observation(payload: &Value) -> Observation {
    Observation {
        content: payload.to_string(),
        ..Observation::default()
    }
}

#[test]
fn the_top_level_target_key_wins() {
    let result = json!({
        HANDOFF_TARGET_KEY: "b",
        HANDOFF_MESSAGE_KEY: "context",
        HANDOFF_REASON_KEY: "busy",
        "output": {HANDOFF_TARGET_KEY: "nested"}
    });

    let signal = extract_handoff_signal(&result).unwrap();
    assert_eq!(signal.target, "b");
    assert_eq!(signal.message.as_deref(), Some("context"));
    assert_eq!(signal.reason.as_deref(), Some("busy"));
}

#[test]
fn nested_probes_are_tried_output_then_result_then_content() {
    let all = json!({
        "output": {HANDOFF_TARGET_KEY: "from_output"},
        "result": {HANDOFF_TARGET_KEY: "from_result"},
        "content": {HANDOFF_TARGET_KEY: "from_content"}
    });
    assert_eq!(extract_handoff_signal(&all).unwrap().target, "from_output");

    let rest = json!({
        "result": {HANDOFF_TARGET_KEY: "from_result"},
        "content": {HANDOFF_TARGET_KEY: "from_content"}
    });
    assert_eq!(extract_handoff_signal(&rest).unwrap().target, "from_result");

    let content = json!({"content": {HANDOFF_TARGET_KEY: "from_content"}});
    assert_eq!(
        extract_handoff_signal(&content).unwrap().target,
        "from_content"
    );
}

#[test]
fn observations_are_scanned_newest_first() {
    let observations = vec![
        observation(&json!({HANDOFF_TARGET_KEY: "old"})),
        observation(&json!({"unrelated": true})),
        observation(&json!({
            HANDOFF_TARGET_KEY: "new",
            HANDOFF_REASON_KEY: "second"
        })),
    ];

    let signal = extract_handoff_signal_from_observations(&observations).unwrap();
    assert_eq!(signal.target, "new");
    assert_eq!(signal.reason.as_deref(), Some("second"));
    assert!(extract_handoff_signal_from_observations(&[]).is_none());
}

#[test]
fn empty_target_message_and_reason_are_dropped() {
    let no_target = json!({HANDOFF_TARGET_KEY: ""});
    assert!(extract_handoff_signal(&no_target).is_none());

    let empties = json!({
        HANDOFF_TARGET_KEY: "b",
        HANDOFF_MESSAGE_KEY: "",
        HANDOFF_REASON_KEY: 3
    });
    let signal = extract_handoff_signal(&empties).unwrap();
    assert_eq!(signal.message, None, "an empty message becomes `None`");
    assert_eq!(signal.reason, None, "a non-string reason becomes `None`");
}

#[tokio::test]
async fn session_context_tool_messages_are_the_last_resort() {
    let session: Arc<dyn Session> = Arc::new(AgentSession::new("signal-session"));
    session
        .update_state(json!({"context": {"default_context_id": {"messages": [
            {"role": "user", "content": "hi"},
            {"role": "tool", "content": "not json"},
            {"role": "tool", "content": "{\"__handoff_to__\": \"old\"}"},
            {"role": "tool", "content": "{\"__handoff_to__\": \"new\", \"__handoff_reason__\": \"done\"}"}
        ]}}}))
        .await
        .unwrap();

    let signal = extract_handoff_signal_from_session(&json!({"output": "plain"}), &session)
        .await
        .unwrap();
    assert_eq!(signal.target, "new");
    assert_eq!(signal.reason.as_deref(), Some("done"));

    let direct = json!({HANDOFF_TARGET_KEY: "from_result"});
    assert_eq!(
        extract_handoff_signal_from_session(&direct, &session)
            .await
            .unwrap()
            .target,
        "from_result",
        "the result-level directive precedes the session"
    );
}

// ---------------------------------------------------------------------------
// Handoff tool
// ---------------------------------------------------------------------------

#[tokio::test]
async fn the_tool_card_matches_the_python_shape() {
    let tool = HandoffTool::new("b", "");

    assert_eq!(tool.name(), "transfer_to_b");
    let card = tool.card();
    assert_eq!(card.id, "transfer_to_b");
    assert_eq!(card.name, "transfer_to_b");
    assert_eq!(
        card.description,
        "Transfer the current task to b for processing."
    );
    assert!(card.parallel_safe);
    assert_eq!(card.input_params["required"], json!(["reason"]));
    assert!(card.input_params["properties"].get("reason").is_some());
    assert!(card.input_params["properties"].get("message").is_some());
}

#[tokio::test]
async fn the_tool_description_appends_the_target_description() {
    let plain = HandoffTool::new("b", "");
    assert!(!plain.card().description.contains("Handles billing"));

    let described = HandoffTool::new("b", "Handles billing questions.");
    assert_eq!(
        described.card().description,
        "Transfer the current task to b for processing. Handles billing questions."
    );
}

#[tokio::test]
async fn the_tool_never_raises_and_normalizes_its_arguments() {
    let tool = HandoffTool::new("b", "");
    let context = ToolContext::new();

    for junk in [Value::Null, json!([1, 2]), json!(7)] {
        let value = tool.invoke(junk, &context).await.unwrap();
        assert_eq!(
            value,
            json!({
                HANDOFF_TARGET_KEY: "b",
                HANDOFF_MESSAGE_KEY: "",
                HANDOFF_REASON_KEY: ""
            })
        );
    }

    let parsed = tool
        .invoke(json!("{\"reason\": \"r\", \"message\": \"m\"}"), &context)
        .await
        .unwrap();
    assert_eq!(parsed[HANDOFF_REASON_KEY], "r");
    assert_eq!(parsed[HANDOFF_MESSAGE_KEY], "m");

    let plain = tool
        .invoke(json!("the user needs billing"), &context)
        .await
        .unwrap();
    assert_eq!(plain[HANDOFF_REASON_KEY], "the user needs billing");
    assert_eq!(plain[HANDOFF_MESSAGE_KEY], "");
    assert_eq!(plain[HANDOFF_TARGET_KEY], "b");
}

// ---------------------------------------------------------------------------
// Handoff request
// ---------------------------------------------------------------------------

#[test]
fn the_request_round_trips_and_tolerates_an_absent_history() {
    let request = HandoffRequest::new(
        json!({"query": "hi"}),
        vec![HandoffHistoryEntry {
            agent: "a".to_owned(),
            output: json!({"output": "a is done"}),
        }],
    );
    let value = request.to_value();
    assert_eq!(
        value,
        json!({
            "input_message": {"query": "hi"},
            "history": [{"agent": "a", "output": {"output": "a is done"}}]
        })
    );
    assert_eq!(HandoffRequest::from_value(&value).unwrap(), request);

    let absent = HandoffRequest::from_value(&json!({"input_message": "x"})).unwrap();
    assert_eq!(absent.input_message, json!("x"));
    assert_eq!(absent.history, [] as [HandoffHistoryEntry; 0]);
}

#[test]
fn a_history_entry_keeps_its_python_field_names() {
    let entry = HandoffHistoryEntry {
        agent: "a".to_owned(),
        output: json!({"answer": 1}),
    };
    assert_eq!(
        serde_json::to_value(&entry).unwrap(),
        json!({"agent": "a", "output": {"answer": 1}})
    );
    let parsed: HandoffHistoryEntry =
        serde_json::from_value(json!({"agent": "b", "output": {"answer": 2}})).unwrap();
    assert_eq!(parsed.agent, "b");
    assert_eq!(parsed.output, json!({"answer": 2}));
}

// ---------------------------------------------------------------------------
// End-to-end chains
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_direct_handoff_forwards_the_history_and_completes_with_the_target() {
    let team = handoff_team(
        TeamCard::new("handoff-direct"),
        HandoffTeamConfig::default(),
    );
    let seen: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));

    let target_seen = Arc::clone(&seen);
    let b = FakeAgent::new(agent_card("b"), move |inputs, _session| {
        target_seen.lock().push(inputs);
        Ok(AgentOutput::answer("b is done"))
    });
    let a = FakeAgent::new(agent_card("a"), |_inputs, _session| {
        Ok(handoff_output(
            "a is done",
            "b",
            Some("context for b"),
            Some("route"),
        ))
    });

    team.add_agent(agent_card("a"), provider(a)).unwrap();
    team.add_agent(agent_card("b"), provider(b)).unwrap();

    let session = make_team_session(&team.card(), &json!({"conversation_id": "direct-1"}));
    team.runtime().bind_team_session(Arc::clone(&session));

    let result = team
        .invoke(
            json!({"query": "hi", "conversation_id": "direct-1"}),
            Some(Arc::clone(&session)),
        )
        .await
        .unwrap();

    assert_eq!(result["output"], "b is done");
    assert_eq!(result["result_type"], "answer");

    let inputs = seen.lock().first().cloned().unwrap();
    assert_eq!(
        inputs["query"], "context for b",
        "the signal message becomes the next hop's input"
    );
    let history = inputs["handoff_history"].as_array().unwrap();
    assert_eq!(history.len(), 1, "the first hop is recorded");
    assert_eq!(history[0]["agent"], "a");
    assert_eq!(history[0]["output"]["output"], "a is done");

    assert!(
        team.runtime().team_session("direct-1").is_some(),
        "a caller-provided session stays bound"
    );
    let mut events = session.subscribe();
    assert!(
        matches!(events.next().await, Some(Ok(StreamEvent::End))),
        "a caller-provided session is only end-streamed"
    );
}

#[tokio::test]
async fn the_hop_cap_completes_with_the_current_member_result() {
    let config = HandoffTeamConfig {
        handoff: capped(1),
        ..HandoffTeamConfig::default()
    };
    let team = handoff_team(TeamCard::new("handoff-capped"), config);
    let c_calls = Arc::new(AtomicUsize::new(0));

    let b = FakeAgent::new(agent_card("b"), |_inputs, _session| {
        Ok(handoff_output("b takes over", "c", None, None))
    });
    let calls = Arc::clone(&c_calls);
    let c = FakeAgent::new(agent_card("c"), move |_inputs, _session| {
        calls.fetch_add(1, Ordering::SeqCst);
        Ok(AgentOutput::answer("c is done"))
    });
    let a = FakeAgent::new(agent_card("a"), |_inputs, _session| {
        Ok(handoff_output("a is done", "b", None, None))
    });

    team.add_agent(agent_card("a"), provider(a)).unwrap();
    team.add_agent(agent_card("b"), provider(b)).unwrap();
    team.add_agent(agent_card("c"), provider(c)).unwrap();

    let result = team
        .invoke(json!({"query": "hi", "conversation_id": "capped-1"}), None)
        .await
        .unwrap();

    assert_eq!(
        result["output"], "b takes over",
        "the cap blocks the second hop and finishes with the current result"
    );
    assert_eq!(c_calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn a_signal_to_a_non_member_is_blocked_without_an_error() {
    let team = handoff_team(TeamCard::new("handoff-ghost"), HandoffTeamConfig::default());
    let a = FakeAgent::new(agent_card("a"), |_inputs, _session| {
        Ok(handoff_output("a keeps it", "ghost", None, None))
    });
    let b = FakeAgent::new(agent_card("b"), |_inputs, _session| {
        Ok(AgentOutput::answer("b is done"))
    });

    team.add_agent(agent_card("a"), provider(a)).unwrap();
    team.add_agent(agent_card("b"), provider(b)).unwrap();

    let result = team
        .invoke(json!({"query": "hi", "conversation_id": "ghost-1"}), None)
        .await
        .unwrap();
    assert_eq!(result["output"], "a keeps it");
}

#[tokio::test]
async fn a_short_message_timeout_fails_the_chain() {
    let config = HandoffTeamConfig {
        team: TeamConfig::new().configure_timeout(0.02),
        ..HandoffTeamConfig::default()
    };
    let team = handoff_team(
        TeamCard::new("handoff-timeout").with_id("handoff-timeout-id"),
        config,
    );
    let slow = AsyncFakeAgent::delayed("a", Duration::from_millis(200), "too late");
    let agent: Arc<dyn Agent> = slow;

    team.add_agent(agent_card("a"), provider(agent)).unwrap();

    let error = team
        .invoke(json!({"query": "hi", "conversation_id": "timeout-1"}), None)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Timeout);
    assert!(
        error.message().contains("handoff chain timeout after"),
        "unexpected message: {}",
        error.message()
    );
    assert!(
        error.message().contains("team='handoff-timeout-id'"),
        "the timeout names the team: {}",
        error.message()
    );
}

#[tokio::test]
async fn an_interrupt_completes_the_chain_and_persists_the_history() {
    let team = handoff_team(
        TeamCard::new("handoff-interrupt"),
        HandoffTeamConfig::default(),
    );
    let a = FakeAgent::new(agent_card("a"), |_inputs, _session| {
        Ok(AgentOutput::interrupt(Vec::new(), Vec::new()))
    });
    team.add_agent(agent_card("a"), provider(a)).unwrap();

    let session = make_team_session(&team.card(), &json!({"conversation_id": "interrupt-1"}));
    team.runtime().bind_team_session(Arc::clone(&session));

    let result = team
        .invoke(
            json!({"query": "hi", "conversation_id": "interrupt-1"}),
            Some(Arc::clone(&session)),
        )
        .await
        .unwrap();

    assert_eq!(result["result_type"], "interrupt");
    let signal = extract_interrupt_signal(&result).unwrap();
    assert_eq!(signal.message, None);
    assert_eq!(signal.result, result);

    let history = session.get_state(HISTORY_STATE_KEY).await.unwrap().unwrap();
    let entries = history.as_array().unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0]["agent"], "a");
    assert_eq!(entries[0]["output"]["result_type"], "interrupt");

    // The flush after an interrupt is best effort and must never raise.
    flush_team_session(Some(&session)).await;
    flush_team_session(None).await;
}

#[tokio::test]
async fn transfer_tools_are_mounted_once_per_endpoint() {
    let team = handoff_team(TeamCard::new("handoff-tools"), HandoffTeamConfig::default());
    let recording = RecordingToolHost::new();
    let host: Arc<dyn ToolHost> = recording.clone();

    let a: Arc<dyn Agent> = FakeAgent::new(agent_card("a"), |_inputs, _session| {
        Ok(handoff_output("a is done", "b", None, None))
    });
    let b: Arc<dyn Agent> = FakeAgent::new(agent_card("b"), |_inputs, _session| {
        Ok(AgentOutput::answer("b is done"))
    });
    let c: Arc<dyn Agent> = FakeAgent::new(agent_card("c"), |_inputs, _session| {
        Ok(AgentOutput::answer("c is done"))
    });

    team.add_agent(agent_card("a"), host_provider(a, host))
        .unwrap();
    team.add_agent(agent_card("b"), provider(b)).unwrap();
    team.add_agent(agent_card("c"), provider(c)).unwrap();

    team.invoke(json!({"query": "one", "conversation_id": "tools-1"}), None)
        .await
        .unwrap();
    assert_eq!(
        recording.mounts(),
        ["transfer_to_b", "transfer_to_c"],
        "one sorted transfer tool per allowed target"
    );
    let after_first = recording.card_count();

    team.invoke(json!({"query": "two", "conversation_id": "tools-2"}), None)
        .await
        .unwrap();
    assert_eq!(
        recording.card_count(),
        after_first,
        "a second chain on the same endpoint mounts nothing new"
    );
    assert_eq!(recording.mounts(), ["transfer_to_b", "transfer_to_c"]);
}

#[tokio::test]
async fn the_stream_yields_the_hop_frames_in_order() {
    let team = handoff_team(
        TeamCard::new("handoff-stream"),
        HandoffTeamConfig::default(),
    );
    let a = FakeAgent::new(agent_card("a"), |_inputs, _session| {
        Ok(handoff_output("a is done", "b", None, None))
    });
    let b = FakeAgent::new(agent_card("b"), |_inputs, _session| {
        Ok(AgentOutput::answer("b is done"))
    });

    team.add_agent(agent_card("a"), provider(a)).unwrap();
    team.add_agent(agent_card("b"), provider(b)).unwrap();

    let frames: Vec<_> = team
        .stream(json!({"query": "hi", "conversation_id": "stream-1"}), None)
        .collect()
        .await;

    assert_eq!(frames.len(), 2, "one frame per hop");
    let first = frames[0].as_ref().unwrap();
    assert_eq!(first["type"], "message");
    assert_eq!(first["payload"]["output"], "a is done");
    let second = frames[1].as_ref().unwrap();
    assert_eq!(second["type"], "message");
    assert_eq!(second["payload"]["output"], "b is done");
}

#[tokio::test]
async fn a_resumed_chain_starts_at_the_saved_member_and_drops_the_interrupted_hop() {
    let team = handoff_team(
        TeamCard::new("handoff-resume"),
        HandoffTeamConfig::default(),
    );
    let a_calls = Arc::new(AtomicUsize::new(0));
    let seen: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));

    let calls = Arc::clone(&a_calls);
    let a = FakeAgent::new(agent_card("a"), move |_inputs, _session| {
        calls.fetch_add(1, Ordering::SeqCst);
        Ok(handoff_output("a is done", "b", None, None))
    });
    let target_seen = Arc::clone(&seen);
    let b = FakeAgent::new(agent_card("b"), move |inputs, _session| {
        target_seen.lock().push(inputs);
        Ok(AgentOutput::answer("b resumed"))
    });

    team.add_agent(agent_card("a"), provider(a)).unwrap();
    team.add_agent(agent_card("b"), provider(b)).unwrap();

    let session = make_team_session(&team.card(), &json!({"conversation_id": "resume-1"}));
    session
        .update_state(json!({
            COORDINATOR_STATE_KEY: {"current_agent_id": "b", "handoff_count": 1},
            HISTORY_STATE_KEY: [
                {"agent": "interrupted", "output": {"result_type": "interrupt"}},
                {"agent": "kept", "output": {"output": "kept", "result_type": "answer"}}
            ]
        }))
        .await
        .unwrap();
    team.runtime().bind_team_session(Arc::clone(&session));

    let result = team
        .invoke(
            json!({"query": "hi", "conversation_id": "resume-1"}),
            Some(Arc::clone(&session)),
        )
        .await
        .unwrap();

    assert_eq!(result["output"], "b resumed");
    assert_eq!(
        a_calls.load(Ordering::SeqCst),
        0,
        "the resumed chain starts at the saved member"
    );

    let inputs = seen.lock().first().cloned().unwrap();
    let history = inputs["handoff_history"].as_array().unwrap();
    assert_eq!(history.len(), 1, "the interrupted hop was dropped");
    assert_eq!(history[0]["agent"], "kept");
}
