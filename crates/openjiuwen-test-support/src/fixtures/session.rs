//! Ready-made workflow and agent sessions for tests.
//!
//! Python parity: `create_workflow_session(session_id=...)` and
//! `create_agent_session(session_id=...)` (`openjiuwen/core/session/workflow.py`
//! and `agent.py`) were the two construction helpers every Python test used.
//! These fixtures wrap the Rust factories
//! ([`openjiuwen_session::create_workflow_session`] and
//! [`openjiuwen_session::create_agent_session`]) and hand back the kernel's
//! [`Session`] seam, so a test never repeats an options struct for the common
//! case.
//!
//! # Checkpointer attachment
//!
//! A session receives its checkpointer at construction
//! ([`WorkflowSessionOptions::checkpointer`] /
//! [`AgentSessionOptions::checkpointer`]) and exposes it read-only through
//! [`Session::checkpointer`], so there is no post-construction attach step.
//! [`memory_checkpointer`] builds a fresh in-memory store for sharing between
//! sessions, and [`workflow_session_with_checkpointer`] /
//! [`agent_session_with_checkpointer`] wire one into the default session.
//! To resume, share the store: build both sessions with `*_with` and the same
//! `Arc<dyn Checkpointer>` and the same session id.

use std::sync::Arc;

use openjiuwen_session::{
    create_agent_session, create_workflow_session, AgentSessionOptions, Checkpointer,
    MemoryCheckpointer, Session, WorkflowSessionOptions,
};

/// Creates a workflow session with `id` and otherwise default options.
///
/// The workflow id stays at [`openjiuwen_session::DEFAULT_WORKFLOW_ID`] unless a
/// caller uses [`workflow_session_with`].
#[must_use]
pub fn workflow_session(id: &str) -> Arc<dyn Session> {
    workflow_session_with(WorkflowSessionOptions {
        session_id: Some(id.to_string()),
        ..WorkflowSessionOptions::default()
    })
}

/// Creates a workflow session from `options`.
///
/// This is the escape hatch for metadata, a parent session, a custom workflow
/// id, or a checkpointer; the session id is generated when `options.session_id`
/// is absent or empty.
#[must_use]
pub fn workflow_session_with(options: WorkflowSessionOptions) -> Arc<dyn Session> {
    Arc::new(create_workflow_session(options))
}

/// Creates an agent session with `id` and otherwise default options.
///
/// `id` is both the session id and (by default) the agent id, matching
/// [`AgentSessionOptions`].
#[must_use]
pub fn agent_session(id: &str) -> Arc<dyn Session> {
    agent_session_with(AgentSessionOptions {
        session_id: Some(id.to_string()),
        ..AgentSessionOptions::default()
    })
}

/// Creates an agent session from `options`.
///
/// This is the escape hatch for an explicit agent id/name/description, a parent
/// session id, or a checkpointer; the session id is generated when
/// `options.session_id` is absent or empty.
#[must_use]
pub fn agent_session_with(options: AgentSessionOptions) -> Arc<dyn Session> {
    Arc::new(create_agent_session(options))
}

/// Creates a fresh in-memory checkpointer.
///
/// Each call returns an independent store (`MemoryCheckpointer::new`), so tests
/// stay isolated; share the returned `Arc` between sessions when one should
/// resume another's checkpoint.
#[must_use]
pub fn memory_checkpointer() -> Arc<dyn Checkpointer> {
    Arc::new(MemoryCheckpointer::new())
}

/// Creates a workflow session with `id` and a fresh in-memory checkpointer.
///
/// The checkpointer is reachable through [`Session::checkpointer`], so
/// [`Session::save_checkpoint`] works without any further setup.
#[must_use]
pub fn workflow_session_with_checkpointer(id: &str) -> Arc<dyn Session> {
    workflow_session_with(WorkflowSessionOptions {
        session_id: Some(id.to_string()),
        checkpointer: Some(memory_checkpointer()),
        ..WorkflowSessionOptions::default()
    })
}

/// Creates an agent session with `id` and a fresh in-memory checkpointer.
///
/// The checkpointer is reachable through [`Session::checkpointer`], so
/// [`Session::save_checkpoint`] works without any further setup.
#[must_use]
pub fn agent_session_with_checkpointer(id: &str) -> Arc<dyn Session> {
    agent_session_with(AgentSessionOptions {
        session_id: Some(id.to_string()),
        checkpointer: Some(memory_checkpointer()),
        ..AgentSessionOptions::default()
    })
}
