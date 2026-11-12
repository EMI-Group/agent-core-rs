#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Integration tests for the hierarchical strategies and the supervisor factory.
//!
//! Two topologies share one public surface at the crate root:
//!
//! - `hierarchical_msgbus_team` dispatches every invocation as one P2P request to
//!   a supervisor registered in the team runtime;
//! - `hierarchical_tools_team` mounts the sub-agents on the root agent as tools,
//!   pins the wired instance, and streams the root in-process.
//!
//! `SupervisorAgent::create` builds the default supervisor (a `ReActAgent` whose
//! model calls one agent tool per sub-agent), so the suite covers the factory
//! separately, including one end-to-end run over the bus.
//!
//! Every case is offline and deterministic: members are in-process closures,
//! models are `openjiuwen_test_support::ScriptedModel` scripts, and the only
//! timer is the explicit short deadline of the timeout case.

mod common;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use openjiuwen_agent::{Agent, AgentCard, AgentChunk, AgentOutput};
use openjiuwen_common::error::{ErrorKind, Result};
use openjiuwen_foundation::llm::Model;
use openjiuwen_foundation::tool::ToolContext;
use openjiuwen_multiagent::{
    hierarchical_msgbus_team, hierarchical_tools_team, HierarchicalMsgbusConfig,
    HierarchicalToolsConfig, MemberOptions, RuntimeConfig, SupervisorAgent, SupervisorConfig, Team,
    TeamAgent, TeamCard, TeamRuntime,
};
use openjiuwen_test_support::{MockCompletion, ScriptedModel, ScriptedTurn};
use parking_lot::Mutex;
use serde_json::{json, Value};

use crate::common::{
    agent_card, counting_provider, host_provider, provider, AsyncFakeAgent, FakeAgent,
    RecordingToolHost, StreamAgent, StreamFrame,
};

/// The supervisor's recorded `(inputs, parent session id)` pairs.
type SupervisorCalls = Arc<Mutex<Vec<(Value, Option<String>)>>>;

/// Builds a config whose entry agent is `id` with an empty (blank) id.
fn blank_card() -> AgentCard {
    AgentCard::new("").with_id("")
}

/// Builds the `(parent, child)` member options of an agents-as-tools child.
fn child_options(parent: &str) -> MemberOptions {
    MemberOptions {
        parent_agent_id: Some(parent.to_owned()),
    }
}

// ---------------------------------------------------------------------------
// configs
// ---------------------------------------------------------------------------

#[test]
fn default_strategy_configs_generate_their_entry_agent_id() {
    // `AgentCard::default()` (what both configs use) mints a 32-character uuid
    // hex, so a default config is *not* "unconfigured": only an explicitly blank
    // id reaches the `No supervisor/root agent configured` errors asserted below.
    assert_ne!(HierarchicalMsgbusConfig::default().supervisor_agent.id, "");
    assert_ne!(HierarchicalToolsConfig::default().root_agent.id, "");
}

// ---------------------------------------------------------------------------
// msgbus: supervisor over the message bus
// ---------------------------------------------------------------------------

#[tokio::test]
async fn msgbus_rejects_an_unconfigured_supervisor() {
    let config = HierarchicalMsgbusConfig::new(blank_card());
    let team = hierarchical_msgbus_team(TeamCard::new("hier-msg-nosup"), config);
    team.add_agent(agent_card("worker"), provider(FakeAgent::echoing("worker")))
        .unwrap();

    let error = team.invoke(json!({"query": "hi"}), None).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Config);
    assert_eq!(
        error.message(),
        "No supervisor configured in HierarchicalTeamConfig."
    );
}

#[tokio::test]
async fn msgbus_rejects_an_unregistered_supervisor() {
    let config = HierarchicalMsgbusConfig::new(agent_card("boss"));
    let team = hierarchical_msgbus_team(TeamCard::new("hier-msg-noboss"), config);
    team.add_agent(agent_card("worker"), provider(FakeAgent::echoing("worker")))
        .unwrap();

    let error = team.invoke(json!({"query": "hi"}), None).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Config);
    assert_eq!(
        error.message(),
        "Supervisor 'boss' is not registered in runtime. Call \
         add_agent(supervisor_card, supervisor_provider) before invoke()/stream()."
    );
}

#[tokio::test]
async fn msgbus_returns_the_supervisor_output_over_one_p2p_hop() {
    let input = json!({"conversation_id": "hier-msg-1", "query": "hi"});
    let seen: SupervisorCalls = Arc::new(Mutex::new(Vec::new()));
    let recorded = Arc::clone(&seen);
    let supervisor: Arc<dyn Agent> =
        FakeAgent::new(agent_card("boss-hop"), move |inputs, session| {
            let parent = session
                .as_ref()
                .and_then(|session| session.parent_id().map(str::to_owned));
            recorded.lock().push((inputs, parent));
            Ok(AgentOutput::answer("supervised"))
        });

    let config = HierarchicalMsgbusConfig::new(agent_card("boss-hop"));
    let team = hierarchical_msgbus_team(TeamCard::new("hier-msg-hop"), config);
    team.add_agent(agent_card("boss-hop"), provider(supervisor))
        .unwrap();

    let result = team.invoke(input.clone(), None).await.unwrap();
    assert_eq!(result, AgentOutput::answer("supervised").to_value());

    let calls = seen.lock().clone();
    assert_eq!(calls.len(), 1, "one invocation is one P2P hop");
    assert_eq!(
        calls[0].0, input,
        "the input reaches the supervisor verbatim"
    );
    assert_eq!(
        calls[0].1.as_deref(),
        Some("hier-msg-1"),
        "the supervisor runs in a child session of the conversation's team session"
    );
}

#[tokio::test]
async fn msgbus_add_agent_adopts_the_configured_deadline() {
    let config = HierarchicalMsgbusConfig {
        timeout: Some(12.5),
        ..HierarchicalMsgbusConfig::new(agent_card("boss-deadline"))
    };
    let team = hierarchical_msgbus_team(TeamCard::new("hier-msg-deadline"), config);
    team.add_agent(
        agent_card("boss-deadline"),
        provider(FakeAgent::echoing("boss-deadline")),
    )
    .unwrap();

    assert_eq!(
        team.runtime().p2p_timeout(),
        Duration::from_secs_f64(12.5),
        "registering the supervisor adopts the configured dispatch deadline"
    );
}

#[tokio::test]
async fn msgbus_add_agent_without_a_deadline_keeps_the_runtime_timeout() {
    let config = HierarchicalMsgbusConfig {
        timeout: None,
        ..HierarchicalMsgbusConfig::new(agent_card("boss-default"))
    };
    let team = hierarchical_msgbus_team(TeamCard::new("hier-msg-nodeadline"), config);
    let before = team.runtime().p2p_timeout();
    assert_eq!(before, Duration::from_secs_f64(1800.0));

    team.add_agent(
        agent_card("boss-default"),
        provider(FakeAgent::echoing("boss-default")),
    )
    .unwrap();
    assert_eq!(team.runtime().p2p_timeout(), before);
}

#[tokio::test]
async fn msgbus_stream_yields_one_output_frame() {
    let supervisor = FakeAgent::new(agent_card("boss-stream"), |_inputs, _session| {
        Ok(AgentOutput::answer("streamed"))
    });
    let config = HierarchicalMsgbusConfig::new(agent_card("boss-stream"));
    let team = hierarchical_msgbus_team(TeamCard::new("hier-msg-stream"), config);
    team.add_agent(agent_card("boss-stream"), provider(supervisor))
        .unwrap();

    let mut frames = team.stream(
        json!({"conversation_id": "hier-msg-stream-1", "query": "hi"}),
        None,
    );
    let frame = frames.next().await.expect("one frame").unwrap();
    assert_eq!(
        frame,
        json!({
            "type": "message",
            "index": 0,
            "payload": {"output": {"output": "streamed", "result_type": "answer"}},
        })
    );
    assert!(
        frames.next().await.is_none(),
        "the supervisor stream is exactly one frame"
    );
}

#[tokio::test]
async fn msgbus_invoke_honours_the_per_invoke_deadline() {
    // 20 ms against a 200 ms supervisor: the deadline wins by an order of
    // magnitude, so the test never depends on scheduler timing.
    let config = HierarchicalMsgbusConfig {
        timeout: Some(0.02),
        ..HierarchicalMsgbusConfig::new(agent_card("slow-boss"))
    };
    let team = hierarchical_msgbus_team(TeamCard::new("hier-msg-timeout"), config);
    let slow: Arc<dyn Agent> =
        AsyncFakeAgent::delayed("slow-boss", Duration::from_millis(200), "too late");
    team.add_agent(agent_card("slow-boss"), provider(slow))
        .unwrap();

    let error = team
        .invoke(
            json!({"conversation_id": "hier-msg-timeout-1", "query": "hi"}),
            None,
        )
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Timeout);
    assert!(error.message().contains("timed out after"), "{error}");
}

#[tokio::test]
async fn stream_reports_an_unregistered_supervisor_as_the_only_error() {
    let config = HierarchicalMsgbusConfig::new(agent_card("boss"));
    let team = hierarchical_msgbus_team(TeamCard::new("hier-msg-stream-noboss"), config);
    team.add_agent(agent_card("worker"), provider(FakeAgent::echoing("worker")))
        .unwrap();

    let mut frames = team.stream(
        json!({"conversation_id": "hier-msg-stream-noboss-1", "query": "hi"}),
        None,
    );
    let error = frames.next().await.expect("one error item").unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Config);
    assert_eq!(
        error.message(),
        "Supervisor 'boss' is not registered in runtime. Call \
         add_agent(supervisor_card, supervisor_provider) before invoke()/stream()."
    );
    assert!(
        frames.next().await.is_none(),
        "the readiness failure is the stream's only item"
    );
}

#[tokio::test]
async fn stream_reports_an_empty_supervisor_id_as_the_only_error() {
    let config = HierarchicalMsgbusConfig::new(blank_card());
    let team = hierarchical_msgbus_team(TeamCard::new("hier-msg-stream-nosup"), config);

    let mut frames = team.stream(
        json!({"conversation_id": "hier-msg-stream-nosup-1", "query": "hi"}),
        None,
    );
    let error = frames.next().await.expect("one error item").unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Config);
    assert_eq!(
        error.message(),
        "No supervisor configured in HierarchicalTeamConfig."
    );
    assert!(
        frames.next().await.is_none(),
        "the readiness failure is the stream's only item"
    );
}

// ---------------------------------------------------------------------------
// tools: agents as tools
// ---------------------------------------------------------------------------

#[tokio::test]
async fn tools_mounts_the_child_tools_on_the_root_and_forwards_the_input() {
    let host = RecordingToolHost::new();
    let seen: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));
    let recorded = Arc::clone(&seen);
    let root: Arc<dyn Agent> = FakeAgent::new(agent_card("root"), move |inputs, _session| {
        recorded.lock().push(inputs);
        Ok(AgentOutput::answer("root answer"))
    });

    let team = hierarchical_tools_team(
        TeamCard::new("hier-tools-mount"),
        HierarchicalToolsConfig::new(agent_card("root")),
    );
    team.add_agent(agent_card("root"), host_provider(root, host.clone()))
        .unwrap();
    for child in ["alpha", "beta"] {
        team.add_member(
            agent_card(child),
            provider(FakeAgent::echoing(child)),
            child_options("root"),
        )
        .unwrap();
    }

    let input = json!({"conversation_id": "hier-tools-mount-1", "query": "hi"});
    let result = team.invoke(input.clone(), None).await.unwrap();
    assert_eq!(result, AgentOutput::answer("root answer").to_value());

    assert_eq!(
        host.mounts(),
        vec!["alpha".to_owned(), "beta".to_owned()],
        "one tool per child, in member order"
    );
    assert_eq!(
        seen.lock().clone(),
        vec![input],
        "the root receives the input verbatim"
    );
}

#[tokio::test]
async fn tools_pins_the_wired_root_and_resolves_it_once() {
    let calls = Arc::new(AtomicUsize::new(0));
    let host = RecordingToolHost::new();
    let root: Arc<dyn Agent> = FakeAgent::new(agent_card("root"), |_inputs, _session| {
        Ok(AgentOutput::answer("root answer"))
    });
    let member = TeamAgent::with_tool_host(Arc::clone(&root), host.clone());
    let root_provider = counting_provider(member, Arc::clone(&calls));

    let team = hierarchical_tools_team(
        TeamCard::new("hier-tools-pin"),
        HierarchicalToolsConfig::new(agent_card("root")),
    );
    team.add_agent(agent_card("root"), root_provider).unwrap();
    team.add_member(
        agent_card("child"),
        provider(FakeAgent::echoing("child")),
        child_options("root"),
    )
    .unwrap();

    team.invoke(json!({"conversation_id": "hier-tools-pin-1"}), None)
        .await
        .unwrap();
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "setup resolves the root once"
    );
    let mounts = host.mounts();
    assert_eq!(mounts, vec!["child".to_owned()]);

    // The second run reaches the pinned instance: no fresh provider resolution
    // and no re-mount.
    team.invoke(json!({"conversation_id": "hier-tools-pin-2"}), None)
        .await
        .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(host.mounts(), mounts);
    assert_eq!(host.card_count(), 1);
}

#[tokio::test]
async fn tools_mounted_child_tool_dispatches_through_the_runtime() {
    let host = RecordingToolHost::new();
    let root: Arc<dyn Agent> = FakeAgent::new(agent_card("root"), |_inputs, _session| {
        Ok(AgentOutput::answer("root answer"))
    });
    let child = FakeAgent::new(agent_card("child"), |_inputs, _session| {
        Ok(AgentOutput::answer("child answer"))
    });

    let team = hierarchical_tools_team(
        TeamCard::new("hier-tools-dispatch"),
        HierarchicalToolsConfig::new(agent_card("root")),
    );
    team.add_agent(agent_card("root"), host_provider(root, host.clone()))
        .unwrap();
    team.add_member(agent_card("child"), provider(child), child_options("root"))
        .unwrap();
    team.invoke(json!({"conversation_id": "hier-tools-dispatch-1"}), None)
        .await
        .unwrap();

    let tool = host.tool("child").expect("the child tool is mounted");
    let ctx = ToolContext::new()
        .with_session("hier-tools-dispatch-1")
        .with_metadata_entry("tool_call_id", json!("call-1"));
    let output = tool.invoke(json!({"query": "ping"}), &ctx).await.unwrap();
    assert_eq!(output, AgentOutput::answer("child answer").to_value());
}

#[tokio::test]
async fn tools_stream_forwards_agent_frames_and_shapes_the_input() {
    let root = StreamAgent::new(
        agent_card("root"),
        vec![
            StreamFrame::Chunk(AgentChunk::new("llm_output", 0, json!({"content": "hi"}))),
            StreamFrame::Chunk(AgentChunk::new("answer", 1, json!({"output": "hi"}))),
        ],
    );
    let card = TeamCard::new("hier-tools-stream");
    let team = hierarchical_tools_team(
        card.clone(),
        HierarchicalToolsConfig::new(agent_card("root")),
    );
    team.add_agent(agent_card("root"), provider(root.clone()))
        .unwrap();

    let frames: Vec<Value> = team
        .stream(
            json!({"conversation_id": "hier-tools-1", "query": "go"}),
            None,
        )
        .map(|frame| frame.unwrap())
        .collect()
        .await;
    assert_eq!(
        frames,
        vec![
            json!({"type": "llm_output", "index": 0, "payload": {"content": "hi"}}),
            json!({"type": "answer", "index": 1, "payload": {"output": "hi"}}),
        ]
    );

    let seen = root.seen();
    assert_eq!(seen.len(), 1, "the root is streamed once");
    assert_eq!(seen[0]["query"], json!("go"));
    assert_eq!(seen[0]["conversation_id"], json!("hier-tools-1"));
    assert_eq!(
        seen[0]["sender"],
        json!(card.id),
        "the team card id is the sender"
    );
}

#[tokio::test]
async fn tools_stream_wraps_a_scalar_input_under_query() {
    let root = StreamAgent::streaming(
        "root",
        vec![StreamFrame::Chunk(AgentChunk::new(
            "answer",
            0,
            json!({"output": "hi"}),
        ))],
    );
    let card = TeamCard::new("hier-tools-scalar");
    let team = hierarchical_tools_team(
        card.clone(),
        HierarchicalToolsConfig::new(agent_card("root")),
    );
    team.add_agent(agent_card("root"), provider(root.clone()))
        .unwrap();

    let frames: Vec<Value> = team
        .stream(json!("plain"), None)
        .map(|frame| frame.unwrap())
        .collect()
        .await;
    assert_eq!(frames.len(), 1);

    let seen = root.seen();
    let shaped = seen[0].as_object().expect("a shaped object input");
    assert_eq!(
        shaped.len(),
        3,
        "query, conversation_id, sender: {shaped:?}"
    );
    assert_eq!(seen[0]["query"], json!("plain"));
    assert_eq!(seen[0]["sender"], json!(card.id));
    assert!(seen[0]["conversation_id"]
        .as_str()
        .is_some_and(|id| !id.is_empty()));
}

#[tokio::test]
async fn tools_rejects_an_unconfigured_root() {
    let team = hierarchical_tools_team(
        TeamCard::new("hier-tools-noroot"),
        HierarchicalToolsConfig::new(blank_card()),
    );

    let error = team.invoke(json!({"query": "hi"}), None).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Config);
    assert_eq!(
        error.message(),
        "No root agent configured in HierarchicalTeamConfig."
    );
}

#[tokio::test]
async fn tools_rejects_an_unregistered_root() {
    let team = hierarchical_tools_team(
        TeamCard::new("hier-tools-noreg"),
        HierarchicalToolsConfig::new(agent_card("root")),
    );

    let error = team.invoke(json!({"query": "hi"}), None).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Config);
    assert_eq!(
        error.message(),
        "Root agent 'root' is not registered in runtime. Call \
         add_agent(root_card, root_provider) before invoke()/stream()."
    );
}

#[tokio::test]
async fn tools_stream_turns_a_failure_into_one_error_frame() {
    let root = StreamAgent::new(
        agent_card("root"),
        vec![
            StreamFrame::Chunk(AgentChunk::new("llm_output", 0, json!({"content": "hi"}))),
            StreamFrame::Fail("boom".to_owned()),
        ],
    );
    let team = hierarchical_tools_team(
        TeamCard::new("hier-tools-fail"),
        HierarchicalToolsConfig::new(agent_card("root")),
    );
    team.add_agent(agent_card("root"), provider(root)).unwrap();

    let frames: Vec<Result<Value>> = team
        .stream(
            json!({"conversation_id": "hier-tools-fail-1", "query": "go"}),
            None,
        )
        .collect()
        .await;
    assert!(
        frames.iter().all(std::result::Result::is_ok),
        "the agent failure is data, not a stream error: {frames:?}"
    );
    assert_eq!(frames.len(), 2);
    assert_eq!(
        frames[0].as_ref().unwrap(),
        &json!({"type": "llm_output", "index": 0, "payload": {"content": "hi"}})
    );
    assert_eq!(
        frames[1].as_ref().unwrap(),
        &json!({
            "type": "message",
            "index": 0,
            "payload": {"output": "boom", "result_type": "error"},
        })
    );
}

// ---------------------------------------------------------------------------
// supervisor factory
// ---------------------------------------------------------------------------

/// Builds a supervisor config around `model` with the Python defaults.
fn supervisor_config(model: Arc<dyn Model>) -> SupervisorConfig {
    SupervisorConfig::new(agent_card("boss"), model, "you supervise")
}

#[test]
fn supervisor_create_rejects_an_empty_agent_list() {
    let error = SupervisorAgent::create(
        vec![],
        supervisor_config(Arc::new(ScriptedModel::text("ok"))),
    )
    .err()
    .expect("a validation error");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(
        error.message(),
        "[SupervisorAgent.create] agents list must not be empty"
    );
}

#[test]
fn supervisor_create_returns_the_configured_card() {
    let config = supervisor_config(Arc::new(ScriptedModel::text("ok")));
    let (card, _provider) =
        SupervisorAgent::create(vec![agent_card("worker")], config.clone()).expect("a supervisor");

    assert_eq!(card, config.card);
    let wire = serde_json::to_value(&card).unwrap();
    let reparsed: AgentCard = serde_json::from_value(wire).unwrap();
    assert_eq!(reparsed, card);
}

#[tokio::test]
async fn supervisor_dispatches_a_sub_agent_end_to_end() {
    let child_seen: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));
    let recorded = Arc::clone(&child_seen);
    let child = FakeAgent::new(agent_card("worker"), move |inputs, _session| {
        recorded.lock().push(inputs);
        Ok(AgentOutput::answer("sub result"))
    });

    let model = ScriptedModel::new([
        ScriptedTurn::Response(
            MockCompletion::tool_call("worker", json!({"query": "sub"})).to_chat_response(""),
        ),
        ScriptedTurn::text("final answer"),
    ]);
    let (supervisor_card, supervisor_provider) = SupervisorAgent::create(
        vec![agent_card("worker")],
        supervisor_config(Arc::new(model) as Arc<dyn Model>),
    )
    .expect("a supervisor");

    let team = hierarchical_msgbus_team(
        TeamCard::new("hier-sup-e2e"),
        HierarchicalMsgbusConfig::new(supervisor_card.clone()),
    );
    team.add_agent(supervisor_card, supervisor_provider)
        .unwrap();
    team.add_agent(agent_card("worker"), provider(child))
        .unwrap();

    let result = team
        .invoke(
            json!({"conversation_id": "hier-sup-1", "query": "hello"}),
            None,
        )
        .await
        .unwrap();

    assert_eq!(
        child_seen.lock().clone(),
        vec![json!({"query": "sub"})],
        "the model's tool arguments reach the sub-agent"
    );
    assert_eq!(result["output"], json!("final answer"));
}

#[tokio::test]
async fn supervisor_exposes_the_sub_agent_card_as_a_tool() {
    let described = AgentCard::new("researcher")
        .with_id("researcher-id")
        .with_description("finds things")
        .with_input_params(json!({"type": "object", "properties": {"topic": {"type": "string"}}}));
    let bare = AgentCard::new("writer").with_id("writer-id");
    let model = Arc::new(ScriptedModel::text("done"));
    let (card, supervisor_provider) = SupervisorAgent::create(
        vec![described.clone(), bare.clone()],
        supervisor_config(Arc::clone(&model) as Arc<dyn Model>),
    )
    .expect("a supervisor");

    let runtime = TeamRuntime::new(RuntimeConfig::new().with_team_id("hier-sup-tools"));
    runtime.pin_agent(&card.id, supervisor_provider());
    runtime
        .invoke_agent(&card.id, json!({"query": "hi"}), None)
        .await
        .unwrap();

    let calls = model.calls();
    let tools = calls
        .first()
        .and_then(|call| call.request.tools.clone())
        .expect("the supervisor sends tool definitions");
    let functions = tools.as_array().expect("an array of tools");
    assert_eq!(functions.len(), 2, "one tool per sub-agent card");

    let researcher = functions
        .iter()
        .find(|tool| tool["function"]["name"] == json!(described.name))
        .expect("the researcher tool");
    assert_eq!(
        researcher["function"]["description"],
        json!(described.description)
    );
    assert_eq!(
        researcher["function"]["parameters"],
        described.input_params.clone().unwrap()
    );

    let writer = functions
        .iter()
        .find(|tool| tool["function"]["name"] == json!(bare.name))
        .expect("the writer tool");
    assert_eq!(
        writer["function"]["parameters"],
        json!({"type": "object", "properties": {}}),
        "a card without a schema falls back to an empty object schema"
    );
}

#[tokio::test]
async fn supervisor_dispatch_works_without_a_parallel_budget() {
    // `max_parallel_sub_agents: 0` clamps to one permit; the mounted tools must
    // still dispatch. A true concurrency assertion is not reachable from the
    // public API (the limiter is private and the ReAct loop runs tool calls
    // sequentially), so this pins the clamp's observable effect: dispatch works.
    for limit in [0, 1] {
        let dispatched = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&dispatched);
        let child = FakeAgent::new(agent_card("worker"), move |_inputs, _session| {
            counter.fetch_add(1, Ordering::SeqCst);
            Ok(AgentOutput::answer("sub result"))
        });
        let model = ScriptedModel::new([
            ScriptedTurn::Response(
                MockCompletion::tool_call("worker", json!({"query": "sub"})).to_chat_response(""),
            ),
            ScriptedTurn::text("final answer"),
        ]);
        let mut config = supervisor_config(Arc::new(model) as Arc<dyn Model>);
        config.max_parallel_sub_agents = limit;
        let (card, supervisor_provider) =
            SupervisorAgent::create(vec![agent_card("worker")], config).expect("a supervisor");

        let runtime =
            TeamRuntime::new(RuntimeConfig::new().with_team_id(format!("hier-sup-limit-{limit}")));
        runtime.register_agent(card.clone(), supervisor_provider);
        runtime.register_agent(agent_card("worker"), provider(child));

        let output = runtime
            .invoke_agent(&card.id, json!({"query": "hi"}), None)
            .await
            .unwrap();
        assert_eq!(
            output.output, "final answer",
            "the supervisor answers for limit {limit}"
        );
        assert_eq!(
            dispatched.load(Ordering::SeqCst),
            1,
            "the sub-agent is dispatched for limit {limit}"
        );
    }
}
