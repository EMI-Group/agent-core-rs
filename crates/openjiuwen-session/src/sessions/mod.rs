//! Concrete session implementations.
//!
//! Ports `openjiuwen/core/session/internal/` (v0.1.20). The workflow-side
//! sessions ([`WorkflowSession`] and [`NodeSession`]) are implemented here; the
//! agent and team sessions land in this same module tree, sharing the
//! crate-private [`shared`] plumbing.
//!
//! The options structs ([`WorkflowSessionOptions`] and their agent/team
//! counterparts) are process-local construction parameters: they hold trait
//! objects such as `Arc<dyn Session>` and `Arc<dyn Checkpointer>`, so they are
//! deliberately not `serde`-serializable.
pub(crate) mod shared;
pub mod workflow;

pub use workflow::{NodeSession, WorkflowSession, WorkflowSessionOptions};

/// Creates a [`WorkflowSession`] from `options`.
#[must_use]
pub fn create_workflow_session(options: WorkflowSessionOptions) -> WorkflowSession {
    WorkflowSession::with_options(options)
}
