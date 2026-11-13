//! The runtime's resource, checkpoint, and transport seams.
//!
//! Kept apart from the registry and dispatch core so [`TeamRuntime`] stays a
//! small surface: these accessors only read or replace the process-local
//! handles a team consults (a [`TeamResourceMgr`], a [`MemberCheckpointer`], and
//! a `TeamTransport`), and every one of them takes a short lock scope so no
//! lock is ever held across an `.await`.
use std::sync::Arc;

use openjiuwen_agent::AgentCard;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::Model;
use openjiuwen_foundation::tool::Tool;
use openjiuwen_session::AgentTeamSession;

use super::TeamRuntime;
use crate::member::AgentProvider;
use crate::member_checkpoint::MemberCheckpointer;
use crate::resources::TeamResourceMgr;
use crate::transport::TeamTransport;

impl TeamRuntime {
    /// Returns the configured resource manager, if any.
    #[must_use]
    pub fn resource_mgr(&self) -> Option<Arc<dyn TeamResourceMgr>> {
        self.resource_mgr.read().clone()
    }

    /// Installs `resource_mgr` as the process-level resource seam.
    ///
    /// Members already registered locally stay resolvable; a later
    /// [`register_agent`](Self::register_agent) forwards to the new manager.
    pub fn set_resource_mgr(&self, resource_mgr: Arc<dyn TeamResourceMgr>) {
        *self.resource_mgr.write() = Some(resource_mgr);
    }

    /// Resolves a tool: the resource manager first, then the local registry
    /// (which holds no tools, so `None` without a manager).
    #[must_use]
    pub fn resolve_tool(&self, tool_id: &str) -> Option<Arc<dyn Tool>> {
        self.resource_mgr().and_then(|mgr| mgr.tool(tool_id))
    }

    /// Resolves a tool by its card name, manager first.
    #[must_use]
    pub fn resolve_tool_by_name(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.resource_mgr().and_then(|mgr| mgr.tool_by_name(name))
    }

    /// Resolves a model: the resource manager first, then the local registry
    /// (which holds no models, so `None` without a manager).
    #[must_use]
    pub fn resolve_model(&self, model_id: &str) -> Option<Arc<dyn Model>> {
        self.resource_mgr().and_then(|mgr| mgr.model(model_id))
    }

    /// Resolves a team session: the bound local session first, then the
    /// resource manager.
    #[must_use]
    pub fn resolve_team_session(&self, session_id: &str) -> Option<Arc<AgentTeamSession>> {
        self.team_session(session_id).or_else(|| {
            self.resource_mgr()
                .and_then(|mgr| mgr.team_session(session_id))
        })
    }

    /// Resolves a member card: the local registry first, then the resource
    /// manager, so a runner-supplied card never shadows a locally registered
    /// one.
    #[must_use]
    pub fn resolve_agent_card(&self, agent_id: &str) -> Option<AgentCard> {
        self.agent_card(agent_id)
            .or_else(|| self.resource_mgr().and_then(|mgr| mgr.agent_card(agent_id)))
    }

    /// Resolves a member provider: the local registry first, then the resource
    /// manager.
    #[must_use]
    pub fn resolve_agent_provider(&self, agent_id: &str) -> Option<AgentProvider> {
        self.agent_provider(agent_id).or_else(|| {
            self.resource_mgr()
                .and_then(|mgr| mgr.agent_provider(agent_id))
        })
    }

    /// Installs the member-scoped checkpointer used around member invocations.
    ///
    /// See [`TeamRuntime::invoke_agent`] for the restore-before/save-after
    /// contract.
    pub fn set_member_checkpointer(&self, checkpointer: Arc<MemberCheckpointer>) {
        *self.member_checkpointer.write() = Some(checkpointer);
    }

    /// Returns the configured member-scoped checkpointer, if any.
    #[must_use]
    pub fn member_checkpointer(&self) -> Option<Arc<MemberCheckpointer>> {
        self.member_checkpointer.read().clone()
    }

    /// Returns the transport outbound deliveries route through.
    ///
    /// The default is an `InProcessTransport` over
    /// [`TeamRuntime::message_bus`], so a runtime without an explicit transport
    /// behaves exactly like the bus itself.
    #[must_use]
    pub fn transport(&self) -> Arc<dyn TeamTransport> {
        Arc::clone(&self.transport.read())
    }

    /// Replaces the outbound transport.
    ///
    /// A transport must serve the same team as the runtime, otherwise topic and
    /// key names would cross teams.
    /// Inbound dispatch remains bus-driven for the in-process transport; a
    /// backend such as the Redis transport supplies its own inbound workers.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] when `transport` serves a different team id than
    /// [`TeamRuntime::team_id`].
    pub fn set_transport(&self, transport: Arc<dyn TeamTransport>) -> Result<()> {
        let transport_team = transport.team_id();
        if transport_team != self.team_id() {
            return Err(Error::validation(format!(
                "transport team id '{transport_team}' does not match runtime team id '{}'",
                self.team_id()
            )));
        }
        *self.transport.write() = transport;
        Ok(())
    }

    /// Starts the bus and the outbound transport. Idempotent.
    ///
    /// Both are the same in-process bus by default, so the second start is a
    /// no-op.
    ///
    /// # Errors
    ///
    /// Reserved for future transports; the in-process bus never fails.
    pub async fn start(&self) -> Result<()> {
        self.bus.start().await?;
        self.transport().start().await
    }

    /// Stops the bus and the outbound transport, joining every worker.
    /// Idempotent.
    ///
    /// Both are the same in-process bus by default, so the second stop is a
    /// no-op.
    ///
    /// # Errors
    ///
    /// Reserved for future transports; the in-process bus never fails.
    pub async fn stop(&self) -> Result<()> {
        self.bus.stop().await?;
        self.transport().stop().await
    }
}
