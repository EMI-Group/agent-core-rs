#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Integration tests for the seams a [`TeamRuntime`] now exposes: the team
//! event bus, the resource-manager seam, member-scoped checkpoints, the
//! outbound transport, and the task events of [`AgentTool`].
//!
//! Every case is offline and deterministic: members are in-process fakes (or a
//! locally defined stateful agent), the checkpoint store is an in-memory
//! `MemoryCheckpointer`, and no timer longer than the test itself runs.

mod common;

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::StreamExt;
use openjiuwen_agent::{Agent, AgentCard, AgentChunk, AgentOutput};
use openjiuwen_common::error::{Error, ErrorKind, Result};
use openjiuwen_foundation::tool::{Tool, ToolContext};
use openjiuwen_multiagent::events::{TeamEvent, TeamEventFilter, TeamEventKind};
use openjiuwen_multiagent::member_checkpoint::MemberCheckpointer;
use openjiuwen_multiagent::resources::LocalResourceMgr;
use openjiuwen_multiagent::transport::{InProcessTransport, TeamTransport, TransportKind};
use openjiuwen_multiagent::{
    AgentProvider, AgentTool, MessageBusConfig, MessageEnvelope, RuntimeConfig, TeamAgent,
    TeamRuntime,
};
use openjiuwen_session::{
    AgentTeamSession, AgentTeamSessionOptions, Checkpointer, MemoryCheckpointer, Session,
};
use parking_lot::Mutex;
use serde_json::{json, Value};

use crate::common::{agent_card, FakeAgent, RecordingHandler};

/// Events recorded by one listener, in emission order.
type Seen = Arc<Mutex<Vec<TeamEvent>>>;

/// Subscribes a recording listener with `filter` and returns its event log.
fn record(runtime: &Arc<TeamRuntime>, filter: TeamEventFilter) -> Seen {
    let seen: Seen = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&seen);
    runtime.events().subscribe(
        filter,
        Arc::new(move |event: &TeamEvent| {
            sink.lock().push(event.clone());
        }),
    );
    seen
}

/// Returns the recorded events of `kind`.
fn of_kind(seen: &Seen, kind: TeamEventKind) -> Vec<TeamEvent> {
    seen.lock()
        .iter()
        .filter(|event| event.kind == kind)
        .cloned()
        .collect()
}

/// Builds a provider that hands out a fresh echoing fake per resolution.
fn provider_echo(name: &str) -> AgentProvider {
    let name = name.to_owned();
    Arc::new(move || TeamAgent::new(FakeAgent::echoing(&name)))
}

/// Builds a provider whose member always fails.
fn provider_failing(name: &str) -> AgentProvider {
    let name = name.to_owned();
    Arc::new(move || {
        TeamAgent::new(FakeAgent::new(agent_card(&name), |_inputs, _session| {
            Err(Error::validation("boom"))
        }))
    })
}

#[tokio::test]
async fn membership_events_track_join_replace_leave_and_pin() {
    let runtime = TeamRuntime::new(RuntimeConfig::new().with_team_id("alpha"));
    let seen = record(&runtime, TeamEventFilter::new());

    runtime.register_agent(agent_card("worker"), provider_echo("worker"));
    runtime.register_agent(agent_card("worker"), provider_echo("worker"));
    assert_eq!(
        runtime.unregister_agent("worker").map(|card| card.id),
        Some("worker".to_owned())
    );
    assert!(runtime.unregister_agent("worker").is_none());
    runtime.pin_agent("pinned", TeamAgent::new(FakeAgent::echoing("pinned")));

    let joined = of_kind(&seen, TeamEventKind::MemberJoined);
    assert_eq!(joined.len(), 3);
    assert_eq!(joined[0].team_id, "alpha");
    assert_eq!(joined[0].agent_id.as_deref(), Some("worker"));
    assert_eq!(joined[0].detail, json!({}));
    assert_eq!(joined[1].detail["replaced"], json!(true));
    assert_eq!(joined[2].agent_id.as_deref(), Some("pinned"));

    let left = of_kind(&seen, TeamEventKind::MemberLeft);
    assert_eq!(left.len(), 1);
    assert_eq!(left[0].agent_id.as_deref(), Some("worker"));
    assert_eq!(left[0].team_id, "alpha");

    // Sequence numbers strictly increase and every event belongs to the team.
    let events = seen.lock().clone();
    let sequences: Vec<u64> = events.iter().map(|event| event.sequence).collect();
    let mut sorted = sequences.clone();
    sorted.sort_unstable();
    assert_eq!(sequences, sorted);
    assert!(events.iter().all(|event| event.team_id == "alpha"));
}

#[tokio::test]
async fn send_and_publish_emit_message_sent_and_receive_events() {
    let runtime = TeamRuntime::new(RuntimeConfig::new().with_team_id("messaging"));
    let seen = record(&runtime, TeamEventFilter::new());
    runtime.register_agent(agent_card("worker"), provider_echo("worker"));
    runtime.subscribe("worker", "news").await.unwrap();

    let reply = runtime
        .send(json!({"query": "hi"}), "worker", "boss", Some("s-1"), None)
        .await
        .unwrap();
    assert_eq!(reply["output"], "worker:hi");
    runtime
        .publish(json!({"query": "hey"}), "news", "boss", Some("s-1"))
        .await
        .unwrap();

    let sent = of_kind(&seen, TeamEventKind::MessageSent);
    assert_eq!(sent.len(), 2);
    assert_eq!(sent[0].agent_id.as_deref(), Some("boss"));
    assert_eq!(sent[0].peer_id.as_deref(), Some("worker"));
    assert_eq!(sent[0].session_id.as_deref(), Some("s-1"));
    assert!(sent[0].message_id.is_some());
    assert!(sent[0].topic_id.is_none());
    assert_eq!(sent[1].topic_id.as_deref(), Some("news"));
    assert_eq!(sent[1].session_id.as_deref(), Some("s-1"));

    let received = of_kind(&seen, TeamEventKind::MessageReceived);
    assert_eq!(received.len(), 2);
    assert_eq!(received[0].agent_id.as_deref(), Some("worker"));
    assert_eq!(received[0].peer_id.as_deref(), Some("boss"));
    assert_eq!(received[0].session_id.as_deref(), Some("s-1"));
    assert!(received[0].topic_id.is_none());
    assert_eq!(received[0].message_id, sent[0].message_id);
    assert_eq!(received[1].topic_id.as_deref(), Some("news"));
    assert_eq!(received[1].agent_id.as_deref(), Some("worker"));
    assert_eq!(received[1].message_id, sent[1].message_id);
}

#[tokio::test]
async fn a_filtered_listener_only_sees_its_kind() {
    let runtime = TeamRuntime::new(RuntimeConfig::new());
    let joined = record(
        &runtime,
        TeamEventFilter::new().kind(TeamEventKind::MemberJoined),
    );
    let everything = record(&runtime, TeamEventFilter::new());

    runtime.register_agent(agent_card("a"), provider_echo("a"));
    runtime
        .send(json!({"query": "hi"}), "a", "boss", None, None)
        .await
        .unwrap();
    let _ = runtime.unregister_agent("a");

    let joined_events = joined.lock().clone();
    assert_eq!(joined_events.len(), 1);
    assert!(
        joined_events
            .iter()
            .all(|event| event.kind == TeamEventKind::MemberJoined),
        "{joined_events:?}"
    );
    assert!(everything.lock().len() > joined_events.len());
}

#[tokio::test]
async fn error_events_cover_inbound_and_outbound_dispatch() {
    let runtime = TeamRuntime::new(RuntimeConfig::new());
    let seen = record(&runtime, TeamEventFilter::new());
    runtime.register_agent(agent_card("bad"), provider_failing("bad"));

    let error = runtime
        .send(json!({}), "bad", "boss", None, None)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(error.message(), "boom");

    // The inbound dispatch failed inside the bus worker, and the caller's send
    // saw the same error propagated back.
    let errors = of_kind(&seen, TeamEventKind::Error);
    let operations: Vec<&str> = errors
        .iter()
        .filter_map(|event| event.detail["operation"].as_str())
        .collect();
    assert!(operations.contains(&"p2p_dispatch"), "{operations:?}");
    assert!(operations.contains(&"send"), "{operations:?}");
    assert!(errors
        .iter()
        .all(|event| event.agent_id.as_deref() == Some("bad")));
    assert!(errors
        .iter()
        .any(|event| event.detail["error"] == json!("validation: boom")));
    // The receive event still precedes the failure, matching Python's ordering.
    assert_eq!(of_kind(&seen, TeamEventKind::MessageReceived).len(), 1);
}

#[tokio::test]
async fn a_failing_transport_surfaces_errors_and_keeps_the_error_events() {
    /// A transport that always fails, to observe the outbound error events.
    #[derive(Debug)]
    struct FailingTransport;

    #[async_trait]
    impl TeamTransport for FailingTransport {
        fn kind(&self) -> TransportKind {
            TransportKind::InProcess
        }

        fn team_id(&self) -> &'static str {
            "broken"
        }

        async fn start(&self) -> Result<()> {
            Ok(())
        }

        async fn stop(&self) -> Result<()> {
            Ok(())
        }

        async fn send(
            &self,
            _envelope: MessageEnvelope,
            _timeout: Option<Duration>,
        ) -> Result<Value> {
            Err(Error::transport("transport down"))
        }

        async fn publish(&self, _envelope: MessageEnvelope) -> Result<()> {
            Err(Error::transport("transport down"))
        }

        fn add_subscription(&self, _agent_id: &str, _topic: &str) {}

        fn remove_subscription(&self, _agent_id: &str, _topic: &str) {}

        fn subscribers(&self, _topic_id: &str) -> Vec<String> {
            Vec::new()
        }
    }

    let runtime = TeamRuntime::new(RuntimeConfig::new().with_team_id("broken"));
    let seen = record(&runtime, TeamEventFilter::new());
    runtime.register_agent(agent_card("worker"), provider_echo("worker"));
    runtime.set_transport(Arc::new(FailingTransport)).unwrap();

    let error = runtime
        .send(json!({}), "worker", "boss", None, None)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Transport);
    let error = runtime
        .publish(json!({}), "news", "boss", None)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Transport);

    let errors = of_kind(&seen, TeamEventKind::Error);
    let operations: Vec<&str> = errors
        .iter()
        .filter_map(|event| event.detail["operation"].as_str())
        .collect();
    assert_eq!(operations, ["send", "publish"]);
    assert_eq!(of_kind(&seen, TeamEventKind::MessageSent).len(), 0);
}

#[tokio::test]
async fn resource_manager_supplies_members_the_team_never_registered() {
    let mgr = LocalResourceMgr::new();
    assert!(mgr.register_agent(agent_card("remote"), provider_echo("remote")));
    let runtime = TeamRuntime::new_with_resource_mgr(RuntimeConfig::new(), mgr.clone());

    assert!(!runtime.has_agent("remote"));
    assert_eq!(
        runtime.resolve_agent_card("remote").map(|card| card.id),
        Some("remote".to_owned())
    );
    assert!(runtime.resolve_agent_provider("remote").is_some());

    let output = runtime
        .invoke_agent("remote", json!({"query": "hi"}), None)
        .await
        .unwrap();
    assert_eq!(output.output, "remote:hi");

    // Tools, models, and team sessions resolve through the same seam.
    assert!(runtime.resolve_tool("missing").is_none());
    assert!(runtime.resolve_tool_by_name("missing").is_none());
    assert!(runtime.resolve_model("missing").is_none());
    let session = Arc::new(AgentTeamSession::with_options(AgentTeamSessionOptions {
        session_id: Some("remote-session".to_owned()),
        ..AgentTeamSessionOptions::default()
    }));
    assert!(mgr.register_team_session(Arc::clone(&session)));
    assert!(runtime.resolve_team_session("remote-session").is_some());
}

#[tokio::test]
async fn local_registry_wins_and_registration_forwards_to_the_manager() {
    let mgr = LocalResourceMgr::new();
    assert!(mgr.register_agent(
        AgentCard::new("worker")
            .with_id("worker")
            .with_name("manager-copy"),
        provider_echo("manager"),
    ));
    let runtime = TeamRuntime::new_with_resource_mgr(RuntimeConfig::new(), mgr.clone());

    let local_card = agent_card("worker").with_name("local");
    runtime.register_agent(local_card.clone(), provider_echo("local"));
    // The local registration shadows the manager's card and provider.
    assert_eq!(
        runtime.resolve_agent_card("worker").map(|card| card.name),
        Some("local".to_owned())
    );
    let output = runtime
        .invoke_agent("worker", json!({"query": "hi"}), None)
        .await
        .unwrap();
    assert_eq!(output.output, "local:hi");
    // The manager rejected the duplicate and kept its first entry.
    assert_eq!(
        mgr.agent_card("worker").map(|card| card.name),
        Some("manager-copy".to_owned())
    );

    // A duplicate local registration is still graceful.
    runtime.register_agent(local_card, provider_echo("local"));
    assert_eq!(mgr.agent_count(), 1);

    // A local team session beats the manager's session of the same id.
    let local_session = Arc::new(AgentTeamSession::with_options(AgentTeamSessionOptions {
        session_id: Some("s".to_owned()),
        ..AgentTeamSessionOptions::default()
    }));
    runtime.bind_team_session(Arc::clone(&local_session));
    assert!(
        mgr.register_team_session(Arc::new(AgentTeamSession::with_options(
            AgentTeamSessionOptions {
                session_id: Some("s".to_owned()),
                ..AgentTeamSessionOptions::default()
            }
        )))
    );
    let resolved = runtime.resolve_team_session("s").expect("session resolves");
    assert!(Arc::ptr_eq(&resolved, &local_session));
}

#[tokio::test]
async fn set_transport_rejects_another_team_and_keeps_delivering() {
    let runtime = TeamRuntime::new(RuntimeConfig::new().with_team_id("alpha"));
    runtime.register_agent(agent_card("worker"), provider_echo("worker"));

    let foreign = InProcessTransport::new("beta", MessageBusConfig::new(), RecordingHandler::new());
    let error = runtime.set_transport(foreign).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("beta"), "{error}");
    assert!(error.message().contains("alpha"), "{error}");
    assert_eq!(runtime.transport().team_id(), "alpha");

    let transport = InProcessTransport::from_bus(runtime.message_bus());
    runtime.set_transport(transport).unwrap();
    assert_eq!(runtime.transport().kind(), TransportKind::InProcess);
    assert_eq!(runtime.transport().team_id(), "alpha");

    let reply = runtime
        .send(json!({"query": "hi"}), "worker", "boss", None, None)
        .await
        .unwrap();
    assert_eq!(reply["output"], "worker:hi");
}

/// A member that increments a `counter` in its own session state.
///
/// The observed pre-invocation value is recorded, which is how a suite proves
/// that the runtime restored the member's own checkpoint before running it.
struct StatefulAgent {
    card: AgentCard,
    seen: Arc<Mutex<Vec<Option<Value>>>>,
    fail: bool,
}

#[async_trait]
impl Agent for StatefulAgent {
    fn card(&self) -> &AgentCard {
        &self.card
    }

    async fn invoke(
        &self,
        _inputs: Value,
        session: Option<Arc<dyn Session>>,
    ) -> Result<AgentOutput> {
        let session = session.expect("the runtime passes a member session");
        let current = session.get_state("counter").await?;
        let next = current.as_ref().and_then(Value::as_u64).unwrap_or(0) + 1;
        self.seen.lock().push(current);
        session.update_state(json!({"counter": next})).await?;
        if self.fail {
            return Err(Error::validation("boom"));
        }
        Ok(AgentOutput::answer(next.to_string()))
    }

    fn stream(
        &self,
        _inputs: Value,
        _session: Option<Arc<dyn Session>>,
    ) -> BoxStream<'static, Result<AgentChunk>> {
        futures::stream::empty().boxed()
    }
}

/// Registers one stateful member and returns the observed counters.
fn register_stateful(
    runtime: &Arc<TeamRuntime>,
    name: &str,
    fail: bool,
) -> Arc<Mutex<Vec<Option<Value>>>> {
    let seen: Arc<Mutex<Vec<Option<Value>>>> = Arc::new(Mutex::new(Vec::new()));
    let agent: Arc<dyn Agent> = Arc::new(StatefulAgent {
        card: agent_card(name),
        seen: Arc::clone(&seen),
        fail,
    });
    let provider: AgentProvider = Arc::new(move || TeamAgent::new(Arc::clone(&agent)));
    runtime.register_agent(agent_card(name), provider);
    seen
}

/// Binds a team session named `team-s` and returns it.
fn bind_team_session(runtime: &Arc<TeamRuntime>) -> Arc<AgentTeamSession> {
    let session = Arc::new(AgentTeamSession::with_options(AgentTeamSessionOptions {
        session_id: Some("team-s".to_owned()),
        ..AgentTeamSessionOptions::default()
    }));
    runtime.bind_team_session(Arc::clone(&session));
    session
}

#[tokio::test]
async fn member_checkpoints_restore_and_save_per_member() {
    let runtime = TeamRuntime::new(RuntimeConfig::new());
    let team_session = bind_team_session(&runtime);
    let alpha_seen = register_stateful(&runtime, "alpha", false);
    let beta_seen = register_stateful(&runtime, "beta", false);

    let checkpointer: Arc<dyn Checkpointer> = Arc::new(MemoryCheckpointer::new());
    runtime.set_member_checkpointer(Arc::new(MemberCheckpointer::new(checkpointer)));

    for name in ["alpha", "beta", "alpha", "beta"] {
        let session: Arc<dyn Session> = Arc::new(team_session.create_agent_session(name));
        runtime
            .invoke_agent(name, json!({}), Some(session))
            .await
            .unwrap();
    }

    // Each member restored only its own state: the second run of a member sees
    // the value its first run saved, and never the other member's value.
    assert_eq!(*alpha_seen.lock(), [None, Some(json!(1))]);
    assert_eq!(*beta_seen.lock(), [None, Some(json!(1))]);
    let saved = runtime
        .member_checkpointer()
        .unwrap()
        .saved_members("team-s")
        .await
        .unwrap();
    assert_eq!(saved, ["alpha".to_owned(), "beta".to_owned()]);
}

#[tokio::test]
async fn a_failed_member_is_not_checkpointed() {
    let runtime = TeamRuntime::new(RuntimeConfig::new());
    let team_session = bind_team_session(&runtime);
    let seen = register_stateful(&runtime, "alpha", true);

    let checkpointer: Arc<dyn Checkpointer> = Arc::new(MemoryCheckpointer::new());
    let member_checkpointer = Arc::new(MemberCheckpointer::new(checkpointer));
    runtime.set_member_checkpointer(Arc::clone(&member_checkpointer));

    let session: Arc<dyn Session> = Arc::new(team_session.create_agent_session("alpha"));
    let error = runtime
        .invoke_agent("alpha", json!({}), Some(session))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(error.message(), "boom");
    assert_eq!(*seen.lock(), [None]);
    assert_eq!(
        member_checkpointer.saved_members("team-s").await.unwrap(),
        Vec::<String>::new()
    );
}

#[tokio::test]
async fn agent_tool_emits_task_events() {
    let runtime = TeamRuntime::new(RuntimeConfig::new().with_team_id("tool-team"));
    let seen = record(&runtime, TeamEventFilter::new());
    runtime.register_agent(agent_card("worker"), provider_echo("worker"));

    let tool = AgentTool::new(&agent_card("worker"), Arc::downgrade(&runtime));
    let ctx = ToolContext::new()
        .with_session("parent")
        .with_metadata_entry("tool_call_id", json!("call-1"));
    let result = tool.invoke(json!({"query": "hi"}), &ctx).await.unwrap();
    assert_eq!(result["output"], "worker:hi");

    let assigned = of_kind(&seen, TeamEventKind::TaskAssigned);
    assert_eq!(assigned.len(), 1);
    assert_eq!(assigned[0].team_id, "tool-team");
    assert_eq!(assigned[0].agent_id.as_deref(), Some("worker"));
    assert_eq!(assigned[0].session_id.as_deref(), Some("parent:call-1"));
    assert_eq!(assigned[0].detail["task"], json!("worker"));
    assert_eq!(assigned[0].detail["call_id"], json!("parent:call-1"));

    let completed = of_kind(&seen, TeamEventKind::TaskCompleted);
    assert_eq!(completed.len(), 1);
    assert_eq!(completed[0].detail["call_id"], json!("parent:call-1"));

    // A failing member gets an error event instead of a completion event.
    runtime.register_agent(agent_card("bad"), provider_failing("bad"));
    let tool = AgentTool::new(&agent_card("bad"), Arc::downgrade(&runtime));
    let error = tool
        .invoke(json!({}), &ToolContext::new())
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Tool);
    assert_eq!(of_kind(&seen, TeamEventKind::TaskCompleted).len(), 1);
    let errors = of_kind(&seen, TeamEventKind::Error);
    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0].agent_id.as_deref(), Some("bad"));
    assert_eq!(errors[0].detail["operation"], json!("agent_tool"));
    assert!(errors[0].detail["error"].as_str().unwrap().contains("boom"));
}
