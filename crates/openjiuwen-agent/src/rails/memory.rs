//! The long-term-memory rail: recall before the model call, write-back after
//! the invocation.
//!
//! Python parity: the memory callback of `openjiuwen/core/memory`.
//!
//! Intended behaviour (hooked in a later round):
//!
//! - `BeforeModelCall` recalls against [`LongTermMemory`] for
//!   [`MemoryRail::scope`], renders the hits with
//!   [`hits_to_system_message`](openjiuwen_memory::hits_to_system_message), and
//!   injects the resulting system message into `ctx.messages`.
//! - `AfterInvoke` extracts and stores the turn's messages with
//!   `add_messages`, honoring [`MemoryRail::agent_config`].
//!
//! Both directions are best-effort: a memory failure is logged with
//! `tracing::warn!` and never aborts the run.
use super::{AgentRail, RailContext};
use async_trait::async_trait;
use openjiuwen_common::error::Result;
use openjiuwen_memory::{AgentMemoryConfig, LongTermMemory, MemoryScope};
use std::sync::Arc;
/// Recalls long-term memory before a model call and writes the turn back after.
pub struct MemoryRail {
    memory: Arc<LongTermMemory>,
    scope: MemoryScope,
    agent_config: AgentMemoryConfig,
    name: String,
}
impl MemoryRail {
    /// Builds a rail over `memory` with the default scope and agent config.
    #[must_use]
    pub fn new(memory: Arc<LongTermMemory>) -> Self {
        Self {
            memory,
            scope: MemoryScope::default(),
            agent_config: AgentMemoryConfig::default(),
            name: "memory".to_owned(),
        }
    }
    /// Binds the rail to `scope`.
    #[must_use]
    pub fn with_scope(mut self, scope: MemoryScope) -> Self {
        self.scope = scope;
        self
    }
    /// Sets the extraction/recall configuration used for this agent.
    #[must_use]
    pub fn with_agent_config(mut self, agent_config: AgentMemoryConfig) -> Self {
        self.agent_config = agent_config;
        self
    }
    /// Returns the memory store the rail reads and writes.
    #[must_use]
    pub fn memory(&self) -> &Arc<LongTermMemory> {
        &self.memory
    }
    /// Returns the scope recall and writes are bound to.
    #[must_use]
    pub fn scope(&self) -> &MemoryScope {
        &self.scope
    }
    /// Returns the agent-level memory configuration.
    #[must_use]
    pub fn agent_config(&self) -> &AgentMemoryConfig {
        &self.agent_config
    }
}
#[async_trait]
impl AgentRail for MemoryRail {
    fn name(&self) -> &str {
        &self.name
    }
    /// TODO: recall for `self.scope`, render the hits with
    /// `hits_to_system_message`, and prepend the system message to
    /// `ctx.messages`; log failures with `tracing::warn!` and continue.
    async fn before_model_call(&self, _ctx: &mut RailContext) -> Result<()> {
        Ok(())
    }
    /// TODO: persist the turn's messages with `add_messages`; log failures with
    /// `tracing::warn!` and continue.
    async fn after_invoke(&self, _ctx: &mut RailContext) -> Result<()> {
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::rails::RailEvent;
    use openjiuwen_memory::MemoryEngineConfig;

    fn memory() -> Arc<LongTermMemory> {
        LongTermMemory::builder(MemoryEngineConfig::default())
            .build()
            .unwrap()
    }

    #[test]
    fn name_and_defaults() {
        let rail = MemoryRail::new(memory());
        assert_eq!(rail.name(), "memory");
        assert_eq!(rail.scope(), &MemoryScope::default());
        assert_eq!(rail.agent_config(), &AgentMemoryConfig::default());
    }
    #[test]
    fn scope_and_agent_config_are_builders() {
        let scope = MemoryScope::user("alice");
        let config = AgentMemoryConfig::default();
        let rail = MemoryRail::new(memory())
            .with_scope(scope.clone())
            .with_agent_config(config.clone());
        assert_eq!(rail.scope(), &scope);
        assert_eq!(rail.agent_config(), &config);
    }
    #[tokio::test]
    async fn hooks_are_placeholders_that_allow_the_run() {
        let rail = MemoryRail::new(memory());
        let mut ctx = RailContext::new(RailEvent::BeforeModelCall);
        ctx.messages
            .push(openjiuwen_foundation::BaseMessage::user("hi"));
        rail.before_model_call(&mut ctx).await.unwrap();
        assert_eq!(ctx.messages.len(), 1);
        let mut ctx = RailContext::new(RailEvent::AfterInvoke);
        rail.after_invoke(&mut ctx).await.unwrap();
    }
}
