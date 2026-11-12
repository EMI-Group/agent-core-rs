//! The long-term-memory rail: recall before the model call, write-back after
//! the invocation.
//!
//! Python parity: the memory callback of `openjiuwen/core/memory`.
//!
//! - `BeforeModelCall` recalls against [`LongTermMemory`] for
//!   [`MemoryRail::scope`] using the last user message as the query, renders the
//!   hits with
//!   [`hits_to_system_message`] under
//!   a fixed token budget, and inserts the resulting system message directly
//!   after the leading system prompt, so the agent's own identity section always
//!   comes first. Recall runs at most once per invocation, tracked by the
//!   `openjiuwen.rails.memory.recalled` flag in [`RailContext::extra`].
//! - `AfterInvoke` stores the turn's user and assistant messages with
//!   `add_messages`, honoring [`MemoryRail::agent_config`].
//!
//! Both directions are best-effort: a memory failure is logged with
//! `tracing::warn!` and never aborts the run. Both are skipped when
//! [`AgentMemoryConfig::enable_long_term_mem`] is disabled.
use super::{AgentRail, RailContext};
use async_trait::async_trait;
use openjiuwen_common::error::Result;
use openjiuwen_foundation::{BaseMessage, Role};
use openjiuwen_memory::{
    hits_to_system_message, AddOptions, AgentMemoryConfig, LongTermMemory, MemoryScope,
    SearchOptions,
};
use serde_json::Value;
use std::sync::Arc;
/// Default number of recall hits injected into the prompt.
const DEFAULT_TOP_K: usize = 5;
/// Token budget of the rendered recall block; the memory store has no
/// counterpart setting to inherit here.
const DEFAULT_RECALL_TOKEN_BUDGET: usize = 512;
/// `ctx.extra` key marking that this invocation already recalled.
const RECALLED_KEY: &str = "openjiuwen.rails.memory.recalled";
/// Recalls long-term memory before a model call and writes the turn back after.
pub struct MemoryRail {
    memory: Arc<LongTermMemory>,
    scope: MemoryScope,
    agent_config: AgentMemoryConfig,
    top_k: usize,
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
            top_k: DEFAULT_TOP_K,
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
    /// Sets how many recall hits are requested and injected.
    #[must_use]
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
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
    /// Returns the number of recall hits the rail requests.
    #[must_use]
    pub fn top_k(&self) -> usize {
        self.top_k
    }
}
/// Returns the text of the last user message, if any.
fn last_user_text(messages: &[BaseMessage]) -> Option<String> {
    messages
        .iter()
        .rev()
        .find(|message| message.role == Role::User)
        .map(|message| message.text().into_owned())
}
#[async_trait]
impl AgentRail for MemoryRail {
    fn name(&self) -> &str {
        &self.name
    }
    /// Recalls for `self.scope` and inserts the rendered hits after the leading
    /// system prompt; failures are logged and the run continues.
    async fn before_model_call(&self, ctx: &mut RailContext) -> Result<()> {
        if !self.agent_config.enable_long_term_mem || ctx.extra.contains_key(RECALLED_KEY) {
            return Ok(());
        }
        let Some(query) = last_user_text(&ctx.messages) else {
            return Ok(());
        };
        if query.trim().is_empty() {
            return Ok(());
        }
        // Mark before searching so a failing recall is not retried on every
        // iteration, and an injected block can never be prepended twice.
        ctx.extra.insert(RECALLED_KEY.to_owned(), Value::Bool(true));
        let hits = match self
            .memory
            .search(
                &self.scope,
                &query,
                &SearchOptions::default().with_top_k(self.top_k),
            )
            .await
        {
            Ok(hits) => hits,
            Err(error) => {
                tracing::warn!(
                    target: "openjiuwen_agent::rails::memory",
                    error = %error,
                    "memory recall failed"
                );
                return Ok(());
            }
        };
        if let Some(message) = hits_to_system_message(&hits, DEFAULT_RECALL_TOKEN_BUDGET) {
            let index = ctx
                .messages
                .iter()
                .take_while(|message| message.role == Role::System)
                .count();
            ctx.messages.insert(index, message);
        }
        Ok(())
    }
    /// Persists the turn's user and assistant messages with `add_messages`;
    /// failures are logged and the run continues.
    async fn after_invoke(&self, ctx: &mut RailContext) -> Result<()> {
        if !self.agent_config.enable_long_term_mem {
            return Ok(());
        }
        let Some(output) = ctx.output.as_ref() else {
            return Ok(());
        };
        let Some(query) = ctx.input.as_ref().and_then(crate::AgentInput::query_text) else {
            return Ok(());
        };
        if query.trim().is_empty() || output.output.trim().is_empty() {
            return Ok(());
        }
        let messages = vec![
            BaseMessage::user(query),
            BaseMessage::assistant(output.output.clone()),
        ];
        if let Err(error) = self
            .memory
            .add_messages(
                &self.scope,
                &messages,
                &self.agent_config,
                &AddOptions::default(),
            )
            .await
        {
            tracing::warn!(
                target: "openjiuwen_agent::rails::memory",
                error = %error,
                "memory write-back failed"
            );
        }
        Ok(())
    }
}
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::rails::RailEvent;
    use crate::{AgentInput, AgentOutput};
    use openjiuwen_memory::{MemoryEngineConfig, NewMemory};
    use serde_json::json;

    fn memory() -> Arc<LongTermMemory> {
        LongTermMemory::builder(MemoryEngineConfig::default())
            .build()
            .unwrap()
    }

    fn scope() -> MemoryScope {
        MemoryScope::user("alice")
    }

    fn rail(memory: Arc<LongTermMemory>) -> MemoryRail {
        MemoryRail::new(memory).with_scope(scope())
    }

    fn disabled_rail(memory: Arc<LongTermMemory>) -> MemoryRail {
        rail(memory).with_agent_config(AgentMemoryConfig {
            enable_long_term_mem: false,
            ..AgentMemoryConfig::default()
        })
    }

    fn context(event: RailEvent, messages: Vec<BaseMessage>) -> RailContext {
        RailContext::new(event).with_messages(messages)
    }

    async fn seed(memory: &LongTermMemory, content: &str) {
        memory
            .add(
                &scope(),
                NewMemory::semantic(content),
                &AddOptions::default(),
            )
            .await
            .unwrap();
    }

    #[test]
    fn name_and_defaults() {
        let rail = MemoryRail::new(memory());
        assert_eq!(rail.name(), "memory");
        assert_eq!(rail.scope(), &MemoryScope::default());
        assert_eq!(rail.agent_config(), &AgentMemoryConfig::default());
        assert_eq!(rail.top_k(), DEFAULT_TOP_K);
        assert_eq!(rail.top_k(), 5);
    }

    #[test]
    fn scope_agent_config_and_top_k_are_builders() {
        let scope = MemoryScope::user("alice");
        let config = AgentMemoryConfig::default();
        let rail = MemoryRail::new(memory())
            .with_scope(scope.clone())
            .with_agent_config(config.clone())
            .with_top_k(3);
        assert_eq!(rail.scope(), &scope);
        assert_eq!(rail.agent_config(), &config);
        assert_eq!(rail.top_k(), 3);
    }

    #[tokio::test]
    async fn recall_injects_a_system_message_after_the_prompt() {
        let store = memory();
        seed(&store, "the user's favourite editor is helix").await;
        let rail = rail(Arc::clone(&store));
        let mut ctx = context(
            RailEvent::BeforeModelCall,
            vec![
                BaseMessage::system("sys"),
                BaseMessage::user("which editor does the user favour? favourite editor helix"),
            ],
        );
        rail.before_model_call(&mut ctx).await.unwrap();
        assert_eq!(ctx.messages.len(), 3);
        assert_eq!(ctx.messages[0].role, Role::System);
        assert_eq!(ctx.messages[0].text().as_ref(), "sys");
        assert_eq!(ctx.messages[1].role, Role::System);
        assert!(ctx.messages[1].text().contains("Long-term memory"));
        assert!(ctx.messages[1].text().contains("helix"));
        assert_eq!(ctx.messages[2].role, Role::User);
        assert!(ctx.extra.contains_key(RECALLED_KEY));
        // A second iteration of the same invocation must not inject again.
        rail.before_model_call(&mut ctx).await.unwrap();
        assert_eq!(ctx.messages.len(), 3);
    }

    #[tokio::test]
    async fn recall_on_an_empty_store_changes_nothing() {
        let rail = rail(memory());
        let mut ctx = context(
            RailEvent::BeforeModelCall,
            vec![
                BaseMessage::system("sys"),
                BaseMessage::user("nothing is stored yet"),
            ],
        );
        rail.before_model_call(&mut ctx).await.unwrap();
        assert_eq!(ctx.messages.len(), 2);
        assert_eq!(ctx.messages[0].text().as_ref(), "sys");
        assert_eq!(ctx.messages[1].text().as_ref(), "nothing is stored yet");
    }

    #[tokio::test]
    async fn recall_without_a_user_message_changes_nothing() {
        let store = memory();
        seed(&store, "the user's favourite editor is helix").await;
        let rail = rail(Arc::clone(&store));
        let mut ctx = context(RailEvent::BeforeModelCall, vec![BaseMessage::system("sys")]);
        rail.before_model_call(&mut ctx).await.unwrap();
        assert_eq!(ctx.messages.len(), 1);
        assert_eq!(ctx.messages[0].text().as_ref(), "sys");
    }

    #[tokio::test]
    async fn disabled_long_term_memory_skips_recall() {
        let store = memory();
        seed(&store, "the user's favourite editor is helix").await;
        let rail = disabled_rail(Arc::clone(&store));
        let mut ctx = context(
            RailEvent::BeforeModelCall,
            vec![
                BaseMessage::system("sys"),
                BaseMessage::user("favourite editor helix"),
            ],
        );
        rail.before_model_call(&mut ctx).await.unwrap();
        assert_eq!(ctx.messages.len(), 2);
        assert!(!ctx.extra.contains_key(RECALLED_KEY));
    }

    #[tokio::test]
    async fn after_invoke_writes_the_turn_back() {
        let store = memory();
        let rail = rail(Arc::clone(&store));
        let mut ctx = context(RailEvent::AfterInvoke, Vec::new())
            .with_input(AgentInput::parse(&json!({"query": "hi"})).unwrap());
        ctx.output = Some(AgentOutput::answer("hello"));
        rail.after_invoke(&mut ctx).await.unwrap();
        let messages = store.get_recent_messages(&scope(), 10).await.unwrap();
        assert_eq!(messages.len(), 2);
        // Both rows share one timestamp, so the store's tie order is not fixed;
        // assert presence rather than position.
        assert!(messages
            .iter()
            .any(|message| message.role == Role::User && message.text().as_ref() == "hi"));
        assert!(messages
            .iter()
            .any(|message| message.role == Role::Assistant && message.text().as_ref() == "hello"));
    }

    #[tokio::test]
    async fn after_invoke_without_output_or_query_is_a_noop() {
        let store = memory();
        let rail = rail(Arc::clone(&store));
        // No output.
        let mut ctx = context(RailEvent::AfterInvoke, Vec::new())
            .with_input(AgentInput::parse(&json!({"query": "hi"})).unwrap());
        rail.after_invoke(&mut ctx).await.unwrap();
        // No input.
        let mut ctx = context(RailEvent::AfterInvoke, Vec::new());
        ctx.output = Some(AgentOutput::answer("hello"));
        rail.after_invoke(&mut ctx).await.unwrap();
        let messages = store.get_recent_messages(&scope(), 10).await.unwrap();
        assert_eq!(messages.len(), 0);
    }

    #[tokio::test]
    async fn disabled_long_term_memory_skips_write_back() {
        let store = memory();
        let rail = disabled_rail(Arc::clone(&store));
        let mut ctx = context(RailEvent::AfterInvoke, Vec::new())
            .with_input(AgentInput::parse(&json!({"query": "hi"})).unwrap());
        ctx.output = Some(AgentOutput::answer("hello"));
        rail.after_invoke(&mut ctx).await.unwrap();
        let messages = store.get_recent_messages(&scope(), 10).await.unwrap();
        assert_eq!(messages.len(), 0);
    }
}
