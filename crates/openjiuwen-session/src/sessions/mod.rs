//! Concrete session implementations.
//!
//! Ports `openjiuwen/core/session/internal/` (v0.1.20). All four concrete
//! sessions live here: the workflow-side [`WorkflowSession`] and
//! [`NodeSession`], and the agent-side [`AgentSession`] and
//! [`AgentTeamSession`]. They share the crate-private [`shared`] plumbing, so a
//! child session reuses its parent's config, stream hub, and trace id instead
//! of opening a second stream.
//!
//! The options structs ([`WorkflowSessionOptions`], [`AgentSessionOptions`],
//! [`AgentTeamSessionOptions`]) are process-local construction parameters: they
//! hold trait objects such as `Arc<dyn Session>` and `Arc<dyn Checkpointer>`, so
//! they are deliberately not `serde`-serializable.
pub mod agent;
pub mod agent_team;
pub(crate) mod shared;
pub mod workflow;

pub use agent::{AgentSession, AgentSessionOptions};
pub use agent_team::{AgentTeamSession, AgentTeamSessionOptions, DEFAULT_TEAM_ID};
pub use workflow::{NodeSession, WorkflowSession, WorkflowSessionOptions};

/// Creates a [`WorkflowSession`] from `options`.
#[must_use]
pub fn create_workflow_session(options: WorkflowSessionOptions) -> WorkflowSession {
    WorkflowSession::with_options(options)
}

/// Creates an [`AgentSession`] from `options`.
#[must_use]
pub fn create_agent_session(options: AgentSessionOptions) -> AgentSession {
    AgentSession::with_options(options)
}

/// Creates an [`AgentTeamSession`] from `options`.
#[must_use]
pub fn create_agent_team_session(options: AgentTeamSessionOptions) -> AgentTeamSession {
    AgentTeamSession::with_options(options)
}
