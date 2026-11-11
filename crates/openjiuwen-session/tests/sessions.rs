//! Behavior tests for the [`Session`] contract and the four concrete sessions.
//!
//! Pins the `Session` seam every layer above the kernel receives as
//! `Arc<dyn Session>` — global/component state scopes, streaming, human-in-the-
//! loop, tracing and checkpoints — and the kind-specific behavior of
//! [`WorkflowSession`], [`NodeSession`], [`AgentSession`], and
//! [`AgentTeamSession`].
//!
//! Ports `openjiuwen/core/session/internal/` (v0.1.20): `workflow.py`
//! (`WorkflowSession`, `NodeSession`), `agent.py` (`AgentSession`), and
//! `agent_team.py` (`AgentTeamSession`), whose Python tests live in
//! `tests/unit_tests/core/session/test_session.py` (`test_basic`,
//! `test_agent_session`, `test_node_session`). Where Rust deliberately adds
//! behavior (HITL timeouts, JSON checkpoints, a fake-friendly default surface)
//! the test names the contract it pins instead.
//!
//! Every wait is bounded by a timeout guard, so a regression fails the test
//! instead of hanging it. The suite is offline and deterministic.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::StreamExt;
use openjiuwen_common::error::{Error, ErrorKind, Result};
use openjiuwen_session::checkpoint::{Checkpointer, MemoryCheckpointer};
use openjiuwen_session::constants::DEFAULT_WORKFLOW_ID;
use openjiuwen_session::interaction::{
    InteractionKind, InteractionOutput, InteractionRequest, InteractionResponse,
};
use openjiuwen_session::session::{Session, SessionSnapshot, SessionType};
use openjiuwen_session::sessions::agent_team::DEFAULT_TEAM_ID;
use openjiuwen_session::sessions::{
    create_workflow_session, AgentSession, AgentSessionOptions, AgentTeamSession,
    AgentTeamSessionOptions, WorkflowSession, WorkflowSessionOptions,
};
use openjiuwen_session::state::{
    paths, COMP_STATE_KEY, GLOBAL_STATE_KEY, IO_STATE_KEY, TRACE_STATE_KEY, WORKFLOW_STATE_KEY,
};
use openjiuwen_session::stream::{StreamEvent, StreamItem, StreamMode};
use openjiuwen_session::tracer::{InvokeType, MemoryTraceSink, TracePosition, TraceSink, Tracer};
use parking_lot::Mutex;
use serde_json::{json, Value};

/// Upper bound on how long a deterministic in-memory wait may take.
const RECV_TIMEOUT: Duration = Duration::from_secs(1);

/// Receive one event, failing the test instead of hanging when none arrives.
async fn recv(stream: &mut BoxStream<'static, Result<StreamEvent>>) -> StreamEvent {
    tokio::time::timeout(RECV_TIMEOUT, stream.next())
        .await
        .expect("a stream event arrives before the timeout")
        .expect("the stream is still open")
        .expect("the stream yields Ok events")
}

/// Assert the stream is exhausted, failing the test instead of hanging.
async fn expect_terminated(stream: &mut BoxStream<'static, Result<StreamEvent>>) {
    let next = tokio::time::timeout(RECV_TIMEOUT, stream.next())
        .await
        .expect("the stream terminates before the timeout");
    assert!(
        next.is_none(),
        "the stream yields nothing after the end marker"
    );
}

/// Join a spawned request, failing the test when it hangs.
async fn join_request(
    handle: tokio::task::JoinHandle<Result<InteractionResponse>>,
) -> Result<InteractionResponse> {
    tokio::time::timeout(RECV_TIMEOUT, handle)
        .await
        .expect("the request resolves before the timeout")
        .expect("the request task does not panic")
}

// --- a minimal `Session` implementation (required methods only) ---

/// A session implementing only the required [`Session`] methods.
///
/// Everything else (`metadata`, `snapshot`, `restore`, `save_checkpoint`,
/// `resume`, `subscribe`, `end_stream`, `close`, `resolve_interaction`) comes
/// from the trait's provided defaults, which is exactly what the tests below
/// pin. `components` is a flat test store: the fake is not the place to
/// re-test nested component paths.
#[derive(Debug)]
struct FakeSession {
    id: String,
    parent_id: Option<String>,
    kind: SessionType,
    global: Mutex<Value>,
    components: Mutex<BTreeMap<String, Value>>,
    sink: Arc<MemoryTraceSink>,
    tracer: Tracer,
    checkpointer: Option<Arc<dyn Checkpointer>>,
}

impl FakeSession {
    /// Build a session with no checkpointer and a fresh recording sink.
    fn new(id: impl Into<String>) -> Self {
        Self::with_checkpointer(id, SessionType::Workflow, None)
    }

    /// Build a session of `kind`, optionally backed by `checkpointer`.
    fn with_checkpointer(
        id: impl Into<String>,
        kind: SessionType,
        checkpointer: Option<Arc<dyn Checkpointer>>,
    ) -> Self {
        let id = id.into();
        let sink = Arc::new(MemoryTraceSink::new());
        let tracer = Tracer::with_sink(id.clone(), format!("trace-{id}"), sink.clone());
        Self {
            id,
            parent_id: None,
            kind,
            global: Mutex::new(json!({})),
            components: Mutex::new(BTreeMap::new()),
            sink,
            tracer,
            checkpointer,
        }
    }

    /// The recording sink behind `trace()`, for frame-count assertions.
    fn recorded(&self) -> &Arc<MemoryTraceSink> {
        &self.sink
    }
}

#[async_trait]
impl Session for FakeSession {
    fn id(&self) -> &str {
        &self.id
    }

    fn parent_id(&self) -> Option<&str> {
        self.parent_id.as_deref()
    }

    fn session_type(&self) -> SessionType {
        self.kind
    }

    async fn get_state(&self, key: &str) -> Result<Option<Value>> {
        let global = self.global.lock();
        if key.is_empty() {
            return Ok(Some(global.clone()));
        }
        Ok(paths::get_value(&global, key).cloned())
    }

    async fn update_state(&self, patch: Value) -> Result<()> {
        let mut global = self.global.lock();
        paths::apply_patch(&mut global, &patch)
    }

    async fn get_component_state(&self, component_id: &str) -> Result<Option<Value>> {
        Ok(self.components.lock().get(component_id).cloned())
    }

    async fn update_component_state(&self, component_id: &str, patch: Value) -> Result<()> {
        if component_id.is_empty() {
            return Err(Error::validation("component id must not be empty"));
        }
        self.components
            .lock()
            .insert(component_id.to_string(), patch);
        Ok(())
    }

    async fn write_stream(&self, _item: StreamItem) -> Result<()> {
        Ok(())
    }

    async fn request_interaction(
        &self,
        _request: InteractionRequest,
    ) -> Result<InteractionResponse> {
        Err(Error::unsupported("not supported"))
    }

    fn trace(&self) -> &dyn TraceSink {
        self.sink.as_ref()
    }

    fn tracer(&self) -> Option<&Tracer> {
        Some(&self.tracer)
    }

    fn checkpointer(&self) -> Option<Arc<dyn Checkpointer>> {
        self.checkpointer.clone()
    }
}

// --- workflow session ---

#[tokio::test]
async fn workflow_state_merges_nested_objects_and_deletes_nulls() {
    // ports test_session.py::TestSession::test_basic / test_clean_non_value
    let session = WorkflowSession::new("wf-state");
    assert_eq!(session.get_state("").await.unwrap(), Some(json!({})));

    session
        .update_state(json!({ "a": { "b": 1 } }))
        .await
        .unwrap();
    session
        .update_state(json!({ "a": { "c": 2 } }))
        .await
        .unwrap();
    assert_eq!(
        session.get_state("a").await.unwrap(),
        Some(json!({ "b": 1, "c": 2 })),
        "object-vs-object patches recurse instead of replacing"
    );
    assert_eq!(session.get_state("a.b").await.unwrap(), Some(json!(1)));

    session
        .update_state(json!({ "a": { "b": null } }))
        .await
        .unwrap();
    assert_eq!(session.get_state("a.b").await.unwrap(), None);
    assert_eq!(
        session.get_state("a").await.unwrap(),
        Some(json!({ "c": 2 }))
    );
}

#[tokio::test]
async fn workflow_component_state_nests_by_dotted_id() {
    // ports the executable_id nesting of internal/workflow.py NodeSession state
    let session = WorkflowSession::new("wf-comp");
    session
        .update_component_state("a.b", json!({ "x": 1 }))
        .await
        .unwrap();

    assert_eq!(
        session.get_component_state("a.b").await.unwrap(),
        Some(json!({ "x": 1 }))
    );
    assert_eq!(
        session.get_component_state("a").await.unwrap(),
        Some(json!({ "b": { "x": 1 } })),
        "a dotted component id is a nested path, not a literal key"
    );
}

#[tokio::test]
async fn workflow_dump_state_has_five_lanes_and_commit_user_inputs() {
    // ports test_node_session (dump_state lane names) / commit_user_inputs
    let session = WorkflowSession::new("wf-dump");
    session
        .commit_user_inputs(json!({ "input": "value" }))
        .unwrap();

    let dump = session.dump_state();
    for key in [
        IO_STATE_KEY,
        GLOBAL_STATE_KEY,
        COMP_STATE_KEY,
        WORKFLOW_STATE_KEY,
        TRACE_STATE_KEY,
    ] {
        assert!(dump.get(key).is_some(), "missing lane `{key}`: {dump}");
    }
    assert_eq!(
        dump.as_object().unwrap().len(),
        5,
        "the committed dump is exactly the five lanes: {dump}"
    );
    assert_eq!(dump[IO_STATE_KEY], json!({ "input": "value" }));
    assert_eq!(dump[GLOBAL_STATE_KEY], json!({ "input": "value" }));
    assert_eq!(dump[COMP_STATE_KEY], json!({}));
    assert_eq!(dump[WORKFLOW_STATE_KEY], json!({}));
    assert_eq!(dump[TRACE_STATE_KEY], json!({}));
}

#[tokio::test]
async fn workflow_session_identity_metadata_and_factories() {
    let default = WorkflowSession::new("wf-default");
    assert_eq!(default.workflow_id(), DEFAULT_WORKFLOW_ID);
    assert_eq!(default.workflow_id(), "workflow");

    let named = WorkflowSession::with_options(WorkflowSessionOptions {
        session_id: Some("wf-named".to_string()),
        workflow_id: Some("flow".to_string()),
        ..WorkflowSessionOptions::default()
    });
    assert_eq!(named.workflow_id(), "flow");

    let parent: Arc<dyn Session> = Arc::new(WorkflowSession::new("parent-session"));
    let child = WorkflowSession::with_options(WorkflowSessionOptions {
        session_id: Some("child-session".to_string()),
        workflow_id: Some("flow".to_string()),
        parent: Some(parent),
        ..WorkflowSessionOptions::default()
    });
    assert_eq!(child.parent_id(), Some("parent-session"));
    assert_eq!(child.session_type(), SessionType::Workflow);
    let metadata = child.metadata();
    assert_eq!(metadata.session_id, "child-session");
    assert_eq!(
        metadata.parent_session_id.as_deref(),
        Some("parent-session")
    );
    assert_eq!(metadata.entity_id.as_deref(), Some("flow"));
    assert_eq!(metadata.entity_name.as_deref(), Some("flow"));

    let generated = WorkflowSession::with_options(WorkflowSessionOptions::default());
    assert!(
        !generated.id().is_empty(),
        "a parentless session without an id gets a generated one"
    );

    let factory = create_workflow_session(WorkflowSessionOptions {
        session_id: Some("factory-session".to_string()),
        ..WorkflowSessionOptions::default()
    });
    factory
        .update_state(json!({ "ready": true }))
        .await
        .unwrap();
    assert_eq!(factory.id(), "factory-session");
    assert_eq!(factory.get_state("ready").await.unwrap(), Some(json!(true)));
}

// --- node session ---

#[tokio::test]
async fn node_global_state_is_a_passthrough_to_the_parent() {
    let workflow = Arc::new(WorkflowSession::new("wf-node-state"));
    let node = workflow.create_node_session("node1");

    node.update_state(json!({ "shared": 1 })).await.unwrap();
    assert_eq!(workflow.get_state("shared").await.unwrap(), Some(json!(1)));
    assert_eq!(node.get_state("shared").await.unwrap(), Some(json!(1)));
    assert_eq!(node.id(), "wf-node-state");
    assert_eq!(node.session_type(), SessionType::Node);
}

#[tokio::test]
async fn node_component_state_nests_under_the_executable_id() {
    // ports test_session.py::TestSession::test_basic (node state nesting)
    let workflow = Arc::new(WorkflowSession::new("wf-node-comp"));
    let node = workflow.create_node_session("node1");

    node.update_component_state("", json!({ "x": 1 }))
        .await
        .unwrap();
    assert_eq!(node.executable_id(), "workflow.node1");
    assert_eq!(
        node.get_component_state("").await.unwrap(),
        Some(json!({ "x": 1 })),
        "an empty component id addresses the node's own subtree"
    );
    assert_eq!(
        workflow
            .get_component_state(node.executable_id())
            .await
            .unwrap(),
        Some(json!({ "x": 1 })),
        "a component-scoped write is visible from the parent"
    );
    assert_eq!(
        workflow.get_component_state("workflow").await.unwrap(),
        Some(json!({ "node1": { "x": 1 } }))
    );
}

#[tokio::test]
async fn node_executable_ids_chain_through_nested_nodes() {
    let workflow = Arc::new(WorkflowSession::with_options(WorkflowSessionOptions {
        session_id: Some("wf-chain".to_string()),
        workflow_id: Some("wf".to_string()),
        ..WorkflowSessionOptions::default()
    }));
    let first = Arc::new(workflow.create_node_session("a"));
    let second = first.create_node_session("b");

    assert_eq!(first.executable_id(), "wf.a");
    assert_eq!(second.executable_id(), "wf.a.b");
    assert_eq!(second.metadata().entity_id.as_deref(), Some("wf.a.b"));
    assert_eq!(second.parent_id(), Some("wf-chain"));
    assert_eq!(second.session_type(), SessionType::Node);
}

#[tokio::test]
async fn node_end_stream_is_a_noop_for_the_parent_stream() {
    let workflow = Arc::new(WorkflowSession::new("wf-node-stream"));
    let mut events = workflow.subscribe();
    let node = workflow.create_node_session("node1");

    node.end_stream().await.unwrap();
    assert!(
        !workflow.writers().unwrap().is_closed(),
        "a node never closes the shared hub"
    );

    workflow
        .write_stream(StreamItem::output(
            "workflow_final",
            None,
            json!({ "ok": true }),
        ))
        .await
        .unwrap();
    match recv(&mut events).await {
        StreamEvent::Item(item) => {
            assert_eq!(item.item_type, "workflow_final");
            assert_eq!(item.payload, json!({ "ok": true }));
        }
        StreamEvent::End => panic!("a node's end_stream must not close the parent stream"),
    }
}

#[tokio::test]
async fn node_interaction_frame_carries_the_executable_id() {
    let workflow = Arc::new(WorkflowSession::new("wf-node-intr"));
    let node = workflow.create_node_session("node1");
    let executable_id = node.executable_id().to_string();
    let mut events = workflow.subscribe();

    let request = InteractionRequest::new(InteractionKind::Confirm, json!({ "q": "ok?" }));
    let request_id = request.id.clone();
    let waiter_node = node.clone();
    let waiter = tokio::spawn(async move { waiter_node.request_interaction(request).await });

    let StreamEvent::Item(item) = recv(&mut events).await else {
        panic!("expected an interaction frame");
    };
    assert_eq!(item.item_type, "__interaction__");
    let output: InteractionOutput = serde_json::from_value(item.payload).unwrap();
    assert_eq!(output.id, executable_id);
    assert_eq!(output.value, json!({ "q": "ok?" }));

    node.resolve_interaction(InteractionResponse::approve(request_id.clone()))
        .await
        .unwrap();
    let response = join_request(waiter).await.unwrap();
    assert_eq!(response.request_id, request_id);
    assert!(response.is_approved());
}

// --- workflow child of an agent ---

#[tokio::test]
async fn agent_workflow_child_shares_the_hub_but_not_the_state() {
    let agent = Arc::new(AgentSession::new("agent-1"));
    let mut events = agent.subscribe();
    let child = agent.create_workflow_session();

    assert_eq!(child.id(), agent.id());
    assert_eq!(child.parent_id(), Some("agent-1"));
    assert_eq!(child.session_type(), SessionType::Workflow);

    child
        .write_stream(StreamItem::output("llm_output", None, json!({ "x": 1 })))
        .await
        .unwrap();
    match recv(&mut events).await {
        StreamEvent::Item(item) => {
            assert_eq!(item.item_type, "llm_output");
            assert_eq!(item.payload, json!({ "x": 1 }));
        }
        StreamEvent::End => panic!("the agent stream stays open"),
    }

    child
        .update_state(json!({ "only_child": true }))
        .await
        .unwrap();
    assert_eq!(
        child.get_state("only_child").await.unwrap(),
        Some(json!(true))
    );
    assert_eq!(
        agent.get_state("only_child").await.unwrap(),
        None,
        "the workflow child owns its own state store"
    );
}

// --- agent session ---

#[tokio::test]
async fn agent_session_env_and_identity_come_from_the_options() {
    // ports test_session.py::TestSession::test_agent_session (identity block)
    let mut envs = BTreeMap::new();
    envs.insert("API_KEY".to_string(), json!("k"));
    envs.insert("MODEL_NAME".to_string(), json!("m"));
    let agent = AgentSession::with_options(AgentSessionOptions {
        session_id: Some("agent-1".to_string()),
        agent_id: Some("researcher".to_string()),
        agent_name: Some("researcher".to_string()),
        description: Some("looks things up".to_string()),
        envs,
        ..AgentSessionOptions::default()
    });

    assert_eq!(agent.env("API_KEY"), Some(json!("k")));
    assert_eq!(agent.env("MODEL_NAME"), Some(json!("m")));
    assert_eq!(agent.env("MISSING"), None);
    let envs = agent.envs();
    assert_eq!(envs.get("API_KEY"), Some(&json!("k")));
    assert_eq!(envs.get("MODEL_NAME"), Some(&json!("m")));
    assert_eq!(agent.agent_id(), "researcher");
    assert_eq!(agent.agent_name(), Some("researcher"));
    assert_eq!(agent.description(), Some("looks things up"));

    let generated = AgentSession::new("");
    assert!(!generated.id().is_empty(), "an empty id generates a uuid");
}

#[tokio::test]
async fn agent_dump_state_has_only_the_two_state_lanes() {
    // ports test_session.py::TestSession::test_agent_session (dump_state), minus
    // Python's trace lane: the Rust agent dump exposes global + agent state only.
    let agent = AgentSession::new("agent-1");
    agent.update_state(json!({ "step": 1 })).await.unwrap();
    assert_eq!(agent.get_state("step").await.unwrap(), Some(json!(1)));

    let dump = agent.dump_state();
    assert_eq!(
        dump,
        json!({ "global_state": { "step": 1 }, "agent_state": {} })
    );
    assert_eq!(dump.as_object().unwrap().len(), 2);
    assert!(dump.get("trace_state").is_none());
}

#[tokio::test]
async fn agent_commit_without_checkpointer_is_none_and_post_run_closes_once() {
    let agent = AgentSession::new("agent-1");
    assert_eq!(agent.commit().await.unwrap(), None);

    let mut events = agent.subscribe();
    assert_eq!(agent.post_run().await.unwrap(), None);
    assert!(agent.writers().unwrap().is_closed());
    assert_eq!(recv(&mut events).await, StreamEvent::End);
    expect_terminated(&mut events).await;

    assert_eq!(
        agent.post_run().await.unwrap(),
        None,
        "post_run is idempotent"
    );
}

// --- team session ---

#[tokio::test]
async fn team_session_defaults_to_the_agent_team_id() {
    // ports internal/agent_team.py DEFAULT_TEAM_ID
    let default = AgentTeamSession::new("");
    assert_eq!(DEFAULT_TEAM_ID, "agent_team");
    assert_eq!(default.team_id(), DEFAULT_TEAM_ID);
    assert_ne!(default.id(), "", "an empty id generates a session id");
    assert_eq!(default.session_type(), SessionType::AgentTeam);

    let options = AgentTeamSession::with_options(AgentTeamSessionOptions::default());
    assert_eq!(options.team_id(), DEFAULT_TEAM_ID);

    let explicit = AgentTeamSession::with_options(AgentTeamSessionOptions {
        team_id: Some("crew".to_string()),
        ..AgentTeamSessionOptions::default()
    });
    assert_eq!(explicit.team_id(), "crew");
}

#[tokio::test]
async fn team_child_agent_carries_source_metadata_and_parent_id() {
    let team = Arc::new(AgentTeamSession::new("crew"));
    let child = team.create_agent_session("worker");
    let metadata = child.metadata();

    assert_eq!(
        metadata.metadata.get("source_agent_id"),
        Some(&json!("worker"))
    );
    assert_eq!(
        metadata.metadata.get("source_team_id"),
        Some(&json!("crew"))
    );
    assert_eq!(child.parent_id(), Some(team.id()));
    assert_eq!(metadata.parent_session_id.as_deref(), Some(team.id()));
    assert_eq!(child.agent_id(), "worker");
    assert_ne!(child.id(), team.id(), "the child gets its own session id");

    let silent = Arc::new(AgentTeamSession::with_options(AgentTeamSessionOptions {
        team_id: Some("crew".to_string()),
        source_metadata_enabled: false,
        ..AgentTeamSessionOptions::default()
    }));
    let child = silent.create_agent_session("worker");
    let metadata = child.metadata();
    assert!(!metadata.metadata.contains_key("source_agent_id"));
    assert!(!metadata.metadata.contains_key("source_team_id"));
}

#[tokio::test]
async fn team_child_agent_shares_the_team_writer_hub() {
    let team = Arc::new(AgentTeamSession::new("crew"));
    let mut events = team.subscribe();
    let child = team.create_agent_session("worker");

    child
        .write_stream(StreamItem::output("llm_output", Some(0), json!({ "x": 1 })))
        .await
        .unwrap();
    match recv(&mut events).await {
        StreamEvent::Item(item) => {
            assert_eq!(item.item_type, "llm_output");
            assert_eq!(item.index, Some(0));
            assert_eq!(item.payload, json!({ "x": 1 }));
        }
        StreamEvent::End => panic!("the child must not close the team stream"),
    }
    assert!(Arc::ptr_eq(
        &team.writers().unwrap(),
        &child.writers().unwrap()
    ));
}

// --- checkpoints ---

#[tokio::test]
async fn workflow_checkpoint_round_trip_restores_state() {
    let checkpointer: Arc<dyn Checkpointer> = Arc::new(MemoryCheckpointer::new());
    let session = WorkflowSession::with_options(WorkflowSessionOptions {
        session_id: Some("wf-1".to_string()),
        checkpointer: Some(Arc::clone(&checkpointer)),
        ..WorkflowSessionOptions::default()
    });
    session.update_state(json!({ "step": 1 })).await.unwrap();
    session
        .update_component_state("node", json!({ "value": 2 }))
        .await
        .unwrap();

    assert_eq!(
        session.save_checkpoint(Some("cp-wf")).await.unwrap(),
        "cp-wf",
        "a custom checkpoint id is preserved"
    );

    let restored = WorkflowSession::with_options(WorkflowSessionOptions {
        session_id: Some("wf-1".to_string()),
        checkpointer: Some(Arc::clone(&checkpointer)),
        ..WorkflowSessionOptions::default()
    });
    assert!(restored.resume("cp-wf").await.unwrap());
    assert_eq!(
        restored.get_state("").await.unwrap(),
        Some(json!({ "step": 1 }))
    );
    assert_eq!(
        restored.get_component_state("node").await.unwrap(),
        Some(json!({ "value": 2 }))
    );
}

#[tokio::test]
async fn agent_checkpoint_round_trip_restores_state() {
    let checkpointer: Arc<dyn Checkpointer> = Arc::new(MemoryCheckpointer::new());
    let session = AgentSession::with_options(AgentSessionOptions {
        session_id: Some("agent-1".to_string()),
        checkpointer: Some(Arc::clone(&checkpointer)),
        ..AgentSessionOptions::default()
    });
    session.update_state(json!({ "step": 1 })).await.unwrap();
    session
        .update_component_state("memory", json!({ "hits": 2 }))
        .await
        .unwrap();
    assert_eq!(
        session.save_checkpoint(Some("cp-agent")).await.unwrap(),
        "cp-agent"
    );

    let restored = AgentSession::with_options(AgentSessionOptions {
        session_id: Some("agent-1".to_string()),
        checkpointer: Some(Arc::clone(&checkpointer)),
        ..AgentSessionOptions::default()
    });
    assert!(restored.resume("cp-agent").await.unwrap());
    assert_eq!(
        restored.get_state("step").await.unwrap(),
        Some(json!(1)),
        "global state is restored into the global lane"
    );
    assert_eq!(
        restored.get_component_state("memory").await.unwrap(),
        Some(json!({ "hits": 2 }))
    );
    assert!(
        !restored.resume("does-not-exist").await.unwrap(),
        "a missing checkpoint resumes to false, not an error"
    );
}

#[tokio::test]
async fn save_checkpoint_without_a_checkpointer_is_a_session_error() {
    let session = WorkflowSession::new("wf-no-cp");
    let error = session.save_checkpoint(None).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Session);
    assert!(matches!(error, Error::Session(_)));
}

#[tokio::test]
async fn trace_position_round_trips_through_a_checkpoint() {
    let checkpointer: Arc<dyn Checkpointer> = Arc::new(MemoryCheckpointer::new());
    let session = FakeSession::with_checkpointer(
        "fake-trace",
        SessionType::Agent,
        Some(checkpointer.clone()),
    );

    let tracer = session.tracer().expect("the fake owns a tracer");
    let trace_id = tracer.trace_id().to_string();
    let span = tracer.start_span("answer", InvokeType::Llm);
    span.finish(json!({ "text": "hi" }));
    assert_eq!(session.recorded().len(), 1);

    let position = session
        .trace()
        .position()
        .expect("a finished frame yields a trace position");
    assert_eq!(position.trace_id, trace_id);
    assert_eq!(position.frame_count, 1);
    assert_eq!(position.last_invoke_id, Some(span.id()));

    assert_eq!(
        session.save_checkpoint(Some("cp-trace")).await.unwrap(),
        "cp-trace"
    );

    let restored =
        FakeSession::with_checkpointer("fake-trace", SessionType::Agent, Some(checkpointer));
    assert!(restored.resume("cp-trace").await.unwrap());
    assert_eq!(
        restored.trace().position(),
        Some(position),
        "resume installs the checkpointed trace position"
    );
}

// --- the `Session` trait through `Arc<dyn Session>` ---

#[tokio::test]
async fn fake_session_relies_on_the_provided_trait_defaults() {
    let fake = FakeSession::new("fake-1");

    assert_eq!(fake.id(), "fake-1");
    assert_eq!(fake.parent_id(), None);
    assert_eq!(fake.session_type(), SessionType::Workflow);
    let metadata = fake.metadata();
    assert_eq!(metadata.session_id, "fake-1");
    assert_eq!(metadata.kind, SessionType::Workflow);
    assert!(fake.checkpointer().is_none());

    let error = fake
        .resolve_interaction(InteractionResponse::approve("intr_1"))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);

    let error = fake
        .request_interaction(InteractionRequest::new(
            InteractionKind::AskUser,
            json!({ "q": 1 }),
        ))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);

    fake.end_stream().await.unwrap();
    fake.close().await.unwrap();
    assert!(
        !fake.resume("cp-missing").await.unwrap(),
        "resume without a checkpointer resumes to false"
    );
}

#[tokio::test]
async fn fake_session_uses_the_default_snapshot_and_restore() {
    let fake = FakeSession::new("fake-snapshot");
    fake.update_state(json!({ "a": 1 })).await.unwrap();

    let snapshot = fake.snapshot().await.unwrap();
    assert_eq!(snapshot.session_id, "fake-snapshot");
    assert_eq!(snapshot.kind, SessionType::Workflow);
    assert_eq!(snapshot.global, json!({ "a": 1 }));
    assert!(snapshot.components.is_empty());
    assert!(snapshot.trace.is_none());
    assert!(snapshot.metadata.is_empty());

    fake.update_state(json!({ "a": 2 })).await.unwrap();
    fake.restore(snapshot).await.unwrap();
    assert_eq!(fake.get_state("a").await.unwrap(), Some(json!(1)));
    assert_eq!(fake.get_state("").await.unwrap(), Some(json!({ "a": 1 })));

    let position = TracePosition {
        trace_id: "trace-manual".to_string(),
        last_invoke_id: Some("inv-1".to_string()),
        frame_count: 4,
    };
    let snapshot = SessionSnapshot {
        session_id: "fake-snapshot".to_string(),
        kind: SessionType::Workflow,
        global: json!({}),
        components: BTreeMap::from([("comp".to_string(), json!({ "x": 1 }))]),
        trace: Some(position.clone()),
        metadata: BTreeMap::new(),
    };
    fake.restore(snapshot).await.unwrap();
    assert_eq!(
        fake.get_component_state("comp").await.unwrap(),
        Some(json!({ "x": 1 })),
        "restore applies each components entry through update_component_state"
    );
    assert_eq!(
        fake.trace().position(),
        Some(position),
        "restore installs the snapshot's trace position"
    );

    let error = fake
        .restore(SessionSnapshot {
            session_id: "fake-snapshot".to_string(),
            kind: SessionType::Workflow,
            global: json!({}),
            components: BTreeMap::from([(String::new(), json!({}))]),
            trace: None,
            metadata: BTreeMap::new(),
        })
        .await
        .unwrap_err();
    assert_eq!(
        error.kind(),
        ErrorKind::Validation,
        "an empty component id is rejected on the restore path"
    );
}

#[tokio::test]
async fn fake_session_default_subscribe_is_immediately_empty() {
    let fake = FakeSession::new("fake-stream");
    let mut events = fake.subscribe();
    expect_terminated(&mut events).await;
}

#[tokio::test]
async fn concurrent_updates_from_eight_tasks_keep_every_key() {
    let session: Arc<dyn Session> = Arc::new(WorkflowSession::new("wf-concurrent"));
    let mut handles = Vec::new();
    for task in 0..8_u64 {
        let session = Arc::clone(&session);
        handles.push(tokio::spawn(async move {
            for index in 0..50_u64 {
                let key = format!("task-{task}-key-{index}");
                session
                    .update_state(json!({ key.clone(): index }))
                    .await
                    .unwrap();
            }
        }));
    }
    for handle in handles {
        handle.await.unwrap();
    }

    let global = session.get_state("").await.unwrap().unwrap();
    assert_eq!(
        global.as_object().unwrap().len(),
        400,
        "every update from every task is committed"
    );
    for task in 0..8_u64 {
        for index in 0..50_u64 {
            assert_eq!(
                session
                    .get_state(&format!("task-{task}-key-{index}"))
                    .await
                    .unwrap(),
                Some(json!(index))
            );
        }
    }
}

#[tokio::test]
async fn streaming_through_the_trait_preserves_the_frame_shape() {
    let session: Arc<dyn Session> = Arc::new(WorkflowSession::new("wf-stream"));
    let mut events = session.subscribe();

    session
        .write_stream(StreamItem::output(
            "workflow_final",
            Some(3),
            json!({ "ok": true }),
        ))
        .await
        .unwrap();
    session
        .write_stream(StreamItem::output(
            "workflow_final",
            Some(4),
            json!({ "ok": false }),
        ))
        .await
        .unwrap();

    match recv(&mut events).await {
        StreamEvent::Item(item) => {
            assert_eq!(item.mode, StreamMode::Output);
            assert_eq!(item.item_type, "workflow_final");
            assert_eq!(item.index, Some(3));
            assert_eq!(item.payload, json!({ "ok": true }));
            assert_eq!(
                item.to_wire(),
                json!({ "type": "workflow_final", "index": 3, "payload": { "ok": true } })
            );
        }
        StreamEvent::End => panic!("the first frame must arrive before the end marker"),
    }
    match recv(&mut events).await {
        StreamEvent::Item(item) => {
            assert_eq!(item.index, Some(4), "frames keep their delivery order");
            assert_eq!(item.payload, json!({ "ok": false }));
        }
        StreamEvent::End => panic!("the second frame must arrive before the end marker"),
    }

    session.end_stream().await.unwrap();
    assert_eq!(recv(&mut events).await, StreamEvent::End);
    expect_terminated(&mut events).await;

    let error = session
        .write_stream(StreamItem::output("workflow_final", None, Value::Null))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(matches!(error, Error::Validation(_)));
}

#[tokio::test]
async fn interaction_without_a_resolver_times_out() {
    let session: Arc<dyn Session> = Arc::new(WorkflowSession::new("wf-hitl-timeout"));
    let request = InteractionRequest::new(InteractionKind::Confirm, json!({ "q": "ok?" }))
        .with_timeout(Duration::from_millis(20));

    let result = tokio::time::timeout(RECV_TIMEOUT, session.request_interaction(request))
        .await
        .expect("the timed request returns before the guard");
    let error = result.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Timeout);
    assert!(matches!(error, Error::Timeout(_)));
}
