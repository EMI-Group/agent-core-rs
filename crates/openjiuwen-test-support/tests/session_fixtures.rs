//! Integration tests for the session fixtures.
//!
//! Python parity: the session helpers replace the
//! `create_workflow_session(session_id=...)` / `create_agent_session(session_id=...)`
//! calls every Python test repeated (`openjiuwen/core/session/workflow.py` and
//! `agent.py`). The unit-level behavior of the sessions themselves belongs to
//! `openjiuwen-session`; this file proves the fixtures return the requested
//! session kind through the `Arc<dyn Session>` seam, that state is per-session
//! and usable, and that the in-memory checkpointer helpers round-trip a
//! checkpoint.
//!
//! Everything here is offline, in-process, and deterministic.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::BTreeMap;

use openjiuwen_session::{AgentSessionOptions, SessionType, WorkflowSessionOptions};
use openjiuwen_test_support::{
    agent_session, agent_session_with, agent_session_with_checkpointer, memory_checkpointer,
    workflow_session, workflow_session_with, workflow_session_with_checkpointer,
};
use serde_json::json;

/// Python parity: `create_workflow_session(session_id="wf-...")` produced a
/// workflow-kind session whose state store starts empty and is writable.
#[tokio::test]
async fn workflow_session_fixture_has_expected_id_kind_and_state() {
    let session = workflow_session("wf-fixture-1");

    assert_eq!(session.id(), "wf-fixture-1");
    assert_eq!(session.session_type(), SessionType::Workflow);
    assert!(session.parent_id().is_none());
    let metadata = session.metadata();
    assert_eq!(metadata.session_id, "wf-fixture-1");
    assert_eq!(metadata.kind, SessionType::Workflow);
    assert_eq!(
        metadata.entity_id.as_deref(),
        Some(openjiuwen_session::DEFAULT_WORKFLOW_ID),
        "the default workflow id is the entity id"
    );
    assert!(
        session.checkpointer().is_none(),
        "the plain fixture has no checkpointer"
    );

    assert_eq!(session.get_state("").await.unwrap(), Some(json!({})));

    session
        .update_state(json!({"step": 1, "draft": {"ok": true}}))
        .await
        .unwrap();
    assert_eq!(session.get_state("step").await.unwrap(), Some(json!(1)));
    assert_eq!(
        session.get_state("draft.ok").await.unwrap(),
        Some(json!(true))
    );

    session
        .update_component_state("node-a", json!({"value": 2}))
        .await
        .unwrap();
    assert_eq!(
        session.get_component_state("node-a").await.unwrap(),
        Some(json!({"value": 2}))
    );
}

/// Python parity: `create_agent_session(session_id=...)` produced an
/// agent-kind session whose agent id defaults to the session id.
#[tokio::test]
async fn agent_session_fixture_has_expected_id_kind_and_state() {
    let session = agent_session("agent-fixture-1");

    assert_eq!(session.id(), "agent-fixture-1");
    assert_eq!(session.session_type(), SessionType::Agent);
    let metadata = session.metadata();
    assert_eq!(metadata.session_id, "agent-fixture-1");
    assert_eq!(metadata.kind, SessionType::Agent);
    assert_eq!(
        metadata.entity_id.as_deref(),
        Some("agent-fixture-1"),
        "the agent id falls back to the session id"
    );
    assert!(
        session.checkpointer().is_none(),
        "the plain fixture has no checkpointer"
    );

    session
        .update_state(json!({"turn": 4, "memory": {"notes": []}}))
        .await
        .unwrap();
    assert_eq!(session.get_state("turn").await.unwrap(), Some(json!(4)));
    assert_eq!(
        session.get_state("memory.notes").await.unwrap(),
        Some(json!([]))
    );

    session
        .update_component_state("tool", json!({"calls": 1}))
        .await
        .unwrap();
    assert_eq!(
        session.get_component_state("tool").await.unwrap(),
        Some(json!({"calls": 1}))
    );
}

/// The `*_with` variants pass options through instead of imposing defaults.
#[tokio::test]
async fn session_with_options_pass_through_workflow_and_agent_fields() {
    let workflow = workflow_session_with(WorkflowSessionOptions {
        session_id: Some("wf-options".to_string()),
        workflow_id: Some("orders".to_string()),
        metadata: BTreeMap::from([("case".to_string(), json!("checkout"))]),
        ..WorkflowSessionOptions::default()
    });
    assert_eq!(workflow.id(), "wf-options");
    assert_eq!(workflow.session_type(), SessionType::Workflow);
    let metadata = workflow.metadata();
    assert_eq!(metadata.entity_id.as_deref(), Some("orders"));
    assert_eq!(metadata.entity_name.as_deref(), Some("orders"));
    assert_eq!(metadata.metadata["workflow_id"], json!("orders"));
    assert_eq!(metadata.metadata["case"], json!("checkout"));

    let agent = agent_session_with(AgentSessionOptions {
        session_id: Some("agent-options".to_string()),
        agent_id: Some("planner".to_string()),
        agent_name: Some("Planner".to_string()),
        ..AgentSessionOptions::default()
    });
    assert_eq!(agent.id(), "agent-options");
    assert_eq!(agent.session_type(), SessionType::Agent);
    let metadata = agent.metadata();
    assert_eq!(metadata.entity_id.as_deref(), Some("planner"));
    assert_eq!(metadata.entity_name.as_deref(), Some("Planner"));
    assert_eq!(metadata.metadata["agent_id"], json!("planner"));
    assert_eq!(metadata.metadata["agent_name"], json!("Planner"));
}

/// The checkpointer helpers hand out a working store: a checkpoint saved by one
/// session is visible to a second session built from the same `Arc`.
///
/// This is why the fixture exposes `*_with` as well: a session only accepts a
/// checkpointer at construction (`Session::checkpointer` is read-only).
#[tokio::test]
async fn in_memory_checkpointer_round_trips_a_workflow_checkpoint() {
    let session = workflow_session_with_checkpointer("wf-cp");
    assert!(
        session.checkpointer().is_some(),
        "the checkpointer fixture exposes its store"
    );
    session.update_state(json!({"step": 3})).await.unwrap();
    let checkpoint_id = session.save_checkpoint(Some("cp-1")).await.unwrap();
    assert_eq!(checkpoint_id, "cp-1");

    let restored = workflow_session_with(WorkflowSessionOptions {
        session_id: Some("wf-cp".to_string()),
        checkpointer: session.checkpointer(),
        ..WorkflowSessionOptions::default()
    });
    restored.update_state(json!({"step": 99})).await.unwrap();
    assert!(restored.resume("cp-1").await.unwrap());
    assert_eq!(
        restored.get_state("step").await.unwrap(),
        Some(json!(3)),
        "resume replaced the live value with the checkpointed one"
    );
    assert!(
        !restored.resume("missing").await.unwrap(),
        "an unknown checkpoint resumes nothing"
    );
}

/// The agent checkpointer helper behaves the same, with the agent namespace.
#[tokio::test]
async fn in_memory_checkpointer_round_trips_an_agent_checkpoint() {
    let session = agent_session_with_checkpointer("agent-cp");
    assert!(session.checkpointer().is_some());
    session.update_state(json!({"turn": 7})).await.unwrap();
    assert_eq!(session.save_checkpoint(Some("cp-a")).await.unwrap(), "cp-a");

    let restored = agent_session_with(AgentSessionOptions {
        session_id: Some("agent-cp".to_string()),
        checkpointer: session.checkpointer(),
        ..AgentSessionOptions::default()
    });
    assert!(restored.resume("cp-a").await.unwrap());
    assert_eq!(restored.get_state("turn").await.unwrap(), Some(json!(7)));
}

/// Each `memory_checkpointer()` call is independent, so one test's checkpoints
/// cannot leak into another's store.
#[tokio::test]
async fn memory_checkpointer_is_isolated_per_call() {
    let first = memory_checkpointer();
    let second = memory_checkpointer();
    assert!(
        !std::sync::Arc::ptr_eq(&first, &second),
        "every helper call allocates a fresh store"
    );

    let session_a = workflow_session_with(WorkflowSessionOptions {
        session_id: Some("shared-id".to_string()),
        checkpointer: Some(first),
        ..WorkflowSessionOptions::default()
    });
    session_a.update_state(json!({"step": 1})).await.unwrap();
    session_a.save_checkpoint(Some("cp")).await.unwrap();

    let session_b = workflow_session_with(WorkflowSessionOptions {
        session_id: Some("shared-id".to_string()),
        checkpointer: Some(second),
        ..WorkflowSessionOptions::default()
    });
    assert!(
        !session_b.resume("cp").await.unwrap(),
        "the other store holds no checkpoint"
    );
}
