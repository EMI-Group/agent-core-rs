//! The long-term memory rail and its recall/write-back seams.
//!
//! Ports the hook bodies of `openjiuwen/harness/rails/memory/memory_rail.py`: one
//! rail injects the long-term memories relevant to the current turn before the
//! model sees the prompt, and hands the finished turn back to the memory layer
//! after the invocation completes.
//!
//! The rail owns no memory store. It depends on two object-safe seams —
//! [`MemoryRecall`] and [`MemoryWriteBack`] — so this module stays free of the
//! memory runtime and upper layers decide how recall and write-back are wired.
//! [`LongTermMemoryAdapter`] is the ready-made implementation over
//! [`openjiuwen_memory::LongTermMemory`].
//!
//! Per-invocation state (the recalled text and the turn query) lives in
//! `ctx.extra` under [`MEMORY_CONTEXT_EXTRA_KEY`] / [`MEMORY_QUERY_EXTRA_KEY`],
//! never in the rail: one rail instance serves every concurrent invocation, so
//! fields would race. Recall and write-back failures are logged and swallowed —
//! memory is an enhancement, never a reason to abort a turn.
use std::fmt;
use std::sync::Arc;

use openjiuwen_agent::InputQuery;
use openjiuwen_common::error::Result;
use openjiuwen_foundation::{BaseMessage, Role};
use openjiuwen_memory::{
    AddOptions, LongTermMemory, MemoryScope, MemoryType, NewMemory, SearchOptions,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::prompts;

use super::{
    last_user_text, publish_prompt_section, rail_priority, HarnessRail, HarnessRailContext,
    MEMORY_CONTEXT_EXTRA_KEY, MEMORY_QUERY_EXTRA_KEY,
};

/// One memory returned by a [`MemoryRecall`] implementation.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct RecalledMemory {
    /// The memory text injected into the prompt.
    pub content: String,
    /// Relevance score in `0.0..=1.0`, when the source ranks hits.
    pub score: f32,
    /// Provenance of the memory (`"long_term_memory"`, a file path, …).
    pub source: Option<String>,
}

impl RecalledMemory {
    /// Builds a memory with `content` and no score or source.
    #[must_use]
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            ..Self::default()
        }
    }

    /// Returns this memory with `score` as its relevance.
    #[must_use]
    pub fn with_score(mut self, score: f32) -> Self {
        self.score = score;
        self
    }

    /// Returns this memory with `source` as its provenance.
    #[must_use]
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }
}

/// The recall half of the memory seam.
///
/// Implemented by `crate::agent` (or an upper layer); a recall failure is
/// logged by [`MemoryRail`] and the turn proceeds without memory context.
#[async_trait::async_trait]
pub trait MemoryRecall: Send + Sync {
    /// Returns at most `limit` memories relevant to `query`, best first.
    ///
    /// # Errors
    ///
    /// The store's failure; [`MemoryRail`] logs and swallows it.
    async fn recall(&self, query: &str, limit: usize) -> Result<Vec<RecalledMemory>>;
}

/// One completed turn handed to the memory layer.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TurnRecord {
    /// The user query that opened the turn.
    pub query: String,
    /// The final answer, when the run produced one.
    pub response: Option<String>,
    /// Names of every tool the model requested during the turn.
    pub tool_calls: Vec<String>,
}

/// The write-back half of the memory seam.
///
/// A write-back failure is logged by [`MemoryRail`] and swallowed; a memory
/// layer that is briefly unavailable must not fail the turn.
#[async_trait::async_trait]
pub trait MemoryWriteBack: Send + Sync {
    /// Persists whatever the memory design wants to keep from `turn`.
    ///
    /// # Errors
    ///
    /// The store's failure; [`MemoryRail`] logs and swallows it.
    async fn write_back(&self, turn: &TurnRecord) -> Result<()>;
}

/// Tunables of [`MemoryRail`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct MemoryRailConfig {
    /// Maximum number of memories recalled for one turn.
    pub recall_limit: usize,
    /// Soft cap on the character count of the injected `<memory-context>` block.
    pub max_context_chars: usize,
    /// Whether the static memory guidance section is published.
    pub publish_prompt_section: bool,
    /// Whether the finished turn is handed to [`MemoryWriteBack`].
    pub write_back: bool,
}

impl Default for MemoryRailConfig {
    /// Five memories, a 2000-character block, both features enabled.
    fn default() -> Self {
        Self {
            recall_limit: 5,
            max_context_chars: 2000,
            publish_prompt_section: true,
            write_back: true,
        }
    }
}

impl MemoryRailConfig {
    /// Returns this config with `limit` as the recall count.
    #[must_use]
    pub const fn with_recall_limit(mut self, limit: usize) -> Self {
        self.recall_limit = limit;
        self
    }

    /// Returns this config with `max_chars` as the context-block cap.
    #[must_use]
    pub const fn with_max_context_chars(mut self, max_chars: usize) -> Self {
        self.max_context_chars = max_chars;
        self
    }

    /// Returns this config with prompt-section publishing toggled.
    #[must_use]
    pub const fn with_publish_prompt_section(mut self, enabled: bool) -> Self {
        self.publish_prompt_section = enabled;
        self
    }

    /// Returns this config with turn write-back toggled.
    #[must_use]
    pub const fn with_write_back(mut self, enabled: bool) -> Self {
        self.write_back = enabled;
        self
    }
}

/// Recalls turn-relevant memories and records finished turns.
///
/// Dispatch priority is [`rail_priority::MEMORY`] (80), the slot of the Python
/// memory rail.
pub struct MemoryRail {
    recall: Option<Arc<dyn MemoryRecall>>,
    write_back: Option<Arc<dyn MemoryWriteBack>>,
    config: MemoryRailConfig,
}

impl MemoryRail {
    /// Builds a rail with no seams and the default config.
    #[must_use]
    pub fn new() -> Self {
        Self {
            recall: None,
            write_back: None,
            config: MemoryRailConfig::default(),
        }
    }

    /// Returns this rail with `recall` as its recall seam.
    #[must_use]
    pub fn with_recall(mut self, recall: Arc<dyn MemoryRecall>) -> Self {
        self.recall = Some(recall);
        self
    }

    /// Returns this rail with `write_back` as its write-back seam.
    #[must_use]
    pub fn with_write_back(mut self, write_back: Arc<dyn MemoryWriteBack>) -> Self {
        self.write_back = Some(write_back);
        self
    }

    /// Returns this rail with `config` as its tunables.
    #[must_use]
    pub fn with_config(mut self, config: MemoryRailConfig) -> Self {
        self.config = config;
        self
    }

    /// The tunables of this rail.
    #[must_use]
    pub fn config(&self) -> MemoryRailConfig {
        self.config
    }

    /// The recall seam, when one is wired.
    #[must_use]
    pub fn recall(&self) -> Option<&Arc<dyn MemoryRecall>> {
        self.recall.as_ref()
    }

    /// The write-back seam, when one is wired.
    #[must_use]
    pub fn write_back(&self) -> Option<&Arc<dyn MemoryWriteBack>> {
        self.write_back.as_ref()
    }

    /// Resolves the text of the current turn: the last user message, falling
    /// back to the invocation input when no message carries it yet.
    fn current_query(ctx: &HarnessRailContext) -> Option<String> {
        if let Some(text) = last_user_text(&ctx.base.messages) {
            return Some(text);
        }
        ctx.base
            .input
            .as_ref()
            .and_then(|input| match &input.query {
                InputQuery::Text(text) if !text.trim().is_empty() => Some(text.trim().to_string()),
                _ => None,
            })
    }

    /// Formats the recalled memories as one `<memory-context>` block, dropping
    /// whole entries past `max_chars` but always keeping at least one.
    fn format_block(memories: &[RecalledMemory], max_chars: usize) -> String {
        let mut lines: Vec<String> = Vec::new();
        let mut used = 0;
        for memory in memories {
            let content = memory.content.trim();
            if content.is_empty() {
                continue;
            }
            let line = format!("- {content}");
            let cost = line.chars().count() + usize::from(!lines.is_empty());
            if used + cost > max_chars && !lines.is_empty() {
                break;
            }
            used += cost;
            lines.push(line);
        }
        if lines.is_empty() {
            return String::new();
        }
        format!("<memory-context>\n{}\n</memory-context>", lines.join("\n"))
    }

    /// Injects `block` as one system message after the leading system messages.
    fn inject(ctx: &mut HarnessRailContext, block: String) {
        let position = ctx
            .base
            .messages
            .iter()
            .take_while(|message| message.role == Role::System)
            .count();
        ctx.base
            .messages
            .insert(position, BaseMessage::system(block));
    }
}

impl Default for MemoryRail {
    /// Equivalent to [`MemoryRail::new`].
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for MemoryRail {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MemoryRail")
            .field("recall", &self.recall.is_some())
            .field("write_back", &self.write_back.is_some())
            .field("config", &self.config)
            .finish()
    }
}

#[async_trait::async_trait]
impl HarnessRail for MemoryRail {
    fn name(&self) -> &'static str {
        "memory"
    }

    fn priority(&self) -> i32 {
        rail_priority::MEMORY
    }

    async fn before_model_call(&self, ctx: &mut HarnessRailContext) -> Result<()> {
        if self.config.publish_prompt_section {
            publish_prompt_section(ctx, &prompts::memory_section());
        }

        let Some(query) = Self::current_query(ctx) else {
            return Ok(());
        };
        ctx.extra_mut().insert(
            MEMORY_QUERY_EXTRA_KEY.to_string(),
            Value::String(query.clone()),
        );

        let Some(recall) = &self.recall else {
            return Ok(());
        };
        let memories = match recall.recall(&query, self.config.recall_limit).await {
            Ok(memories) => memories,
            Err(error) => {
                tracing::warn!(
                    error = %error,
                    "memory recall failed; continuing without memory context"
                );
                return Ok(());
            }
        };
        let block = Self::format_block(&memories, self.config.max_context_chars);
        if block.is_empty() {
            return Ok(());
        }
        ctx.extra_mut().insert(
            MEMORY_CONTEXT_EXTRA_KEY.to_string(),
            Value::String(block.clone()),
        );
        Self::inject(ctx, block);
        Ok(())
    }

    async fn after_invoke(&self, ctx: &mut HarnessRailContext) -> Result<()> {
        if !self.config.write_back {
            return Ok(());
        }
        let Some(writer) = &self.write_back else {
            return Ok(());
        };
        let Some(query) = ctx
            .extra()
            .get(MEMORY_QUERY_EXTRA_KEY)
            .and_then(Value::as_str)
            .map(str::to_string)
            .filter(|query| !query.trim().is_empty())
        else {
            return Ok(());
        };
        let response = ctx.base.output.as_ref().map(|output| output.output.clone());
        let tool_calls: Vec<String> = ctx
            .base
            .messages
            .iter()
            .filter(|message| message.role == Role::Assistant)
            .flat_map(|message| message.tool_calls.iter())
            .map(|call| call.name.clone())
            .collect();
        let turn = TurnRecord {
            query,
            response,
            tool_calls,
        };
        if let Err(error) = writer.write_back(&turn).await {
            tracing::warn!(
                error = %error,
                "memory write-back failed; ignoring"
            );
        }
        Ok(())
    }
}

/// A [`MemoryRecall`] + [`MemoryWriteBack`] implementation over the kernel
/// long-term memory facade.
///
/// Recall searches `scope` for the turn query; write-back appends one memory
/// per turn in the adapter's [`MemoryType`] with the adapter's source marker.
#[derive(Clone)]
pub struct LongTermMemoryAdapter {
    memory: Arc<LongTermMemory>,
    scope: MemoryScope,
    kind: MemoryType,
    source: String,
}

impl LongTermMemoryAdapter {
    /// Builds an adapter over `memory` bound to `scope`.
    ///
    /// The default kind is [`MemoryType::SemanticMemory`] and the default source
    /// marker [`DEFAULT_MEMORY_SOURCE`].
    #[must_use]
    pub fn new(memory: Arc<LongTermMemory>, scope: MemoryScope) -> Self {
        Self {
            memory,
            scope,
            kind: MemoryType::SemanticMemory,
            source: DEFAULT_MEMORY_SOURCE.to_string(),
        }
    }

    /// Returns this adapter writing `kind` memories.
    #[must_use]
    pub fn with_kind(mut self, kind: MemoryType) -> Self {
        self.kind = kind;
        self
    }

    /// Returns this adapter writing the provenance marker `source`.
    #[must_use]
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = source.into();
        self
    }

    /// The wrapped long-term memory facade.
    #[must_use]
    pub fn memory(&self) -> &Arc<LongTermMemory> {
        &self.memory
    }

    /// The scope recall and write-back operate on.
    #[must_use]
    pub fn scope(&self) -> &MemoryScope {
        &self.scope
    }
}

/// Default provenance marker of [`LongTermMemoryAdapter`] writes.
pub const DEFAULT_MEMORY_SOURCE: &str = "deepagent";

impl fmt::Debug for LongTermMemoryAdapter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LongTermMemoryAdapter")
            .field("scope", &self.scope)
            .field("kind", &self.kind)
            .field("source", &self.source)
            .finish_non_exhaustive()
    }
}

#[async_trait::async_trait]
impl MemoryRecall for LongTermMemoryAdapter {
    async fn recall(&self, query: &str, limit: usize) -> Result<Vec<RecalledMemory>> {
        let options = SearchOptions::default().with_top_k(limit);
        let hits = self.memory.search(&self.scope, query, &options).await?;
        Ok(hits
            .into_iter()
            .map(|hit| RecalledMemory {
                content: hit.record.content,
                score: hit.score,
                source: Some("long_term_memory".to_string()),
            })
            .collect())
    }
}

#[async_trait::async_trait]
impl MemoryWriteBack for LongTermMemoryAdapter {
    async fn write_back(&self, turn: &TurnRecord) -> Result<()> {
        let query = turn.query.trim();
        if query.is_empty() {
            return Ok(());
        }
        let content = match turn
            .response
            .as_deref()
            .map(str::trim)
            .filter(|response| !response.is_empty())
        {
            Some(response) => format!("User: {query}\nAssistant: {response}"),
            None => format!("User: {query}"),
        };
        let memory = NewMemory::new(content, self.kind).with_source(self.source.clone());
        self.memory
            .add(&self.scope, memory, &AddOptions::default())
            .await?;
        Ok(())
    }
}

/// A guard against the two trait objects drifting apart: callers hold the
/// adapter as `Arc<dyn MemoryRecall>` and `Arc<dyn MemoryWriteBack>`.
#[allow(dead_code)]
fn adapter_is_object_safe(adapter: Arc<LongTermMemoryAdapter>) {
    let _: Arc<dyn MemoryRecall> = adapter.clone();
    let _: Arc<dyn MemoryWriteBack> = adapter;
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]

    use std::sync::Mutex;

    use openjiuwen_agent::{AgentOutput, RailEvent};
    use openjiuwen_foundation::ToolCall;
    use openjiuwen_memory::MemoryEngineConfig;
    use serde_json::json;

    use super::*;
    use crate::rails::prompt_sections;

    /// A recall seam returning a scripted result.
    struct FakeRecall {
        result: std::result::Result<Vec<RecalledMemory>, String>,
        calls: Mutex<Vec<(String, usize)>>,
    }

    impl FakeRecall {
        fn returning(memories: Vec<RecalledMemory>) -> Arc<Self> {
            Arc::new(Self {
                result: Ok(memories),
                calls: Mutex::new(Vec::new()),
            })
        }

        fn failing() -> Arc<Self> {
            Arc::new(Self {
                result: Err("recall down".to_string()),
                calls: Mutex::new(Vec::new()),
            })
        }

        fn calls(&self) -> Vec<(String, usize)> {
            self.calls.lock().unwrap().clone()
        }
    }

    #[async_trait::async_trait]
    impl MemoryRecall for FakeRecall {
        async fn recall(&self, query: &str, limit: usize) -> Result<Vec<RecalledMemory>> {
            self.calls.lock().unwrap().push((query.to_string(), limit));
            match &self.result {
                Ok(memories) => Ok(memories.clone()),
                Err(error) => Err(openjiuwen_common::error::Error::memory(error.clone())),
            }
        }
    }

    /// A write-back seam recording the turns it receives.
    struct FakeWriteBack {
        fail: bool,
        turns: Mutex<Vec<TurnRecord>>,
    }

    impl FakeWriteBack {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                fail: false,
                turns: Mutex::new(Vec::new()),
            })
        }

        fn failing() -> Arc<Self> {
            Arc::new(Self {
                fail: true,
                turns: Mutex::new(Vec::new()),
            })
        }

        fn turns(&self) -> Vec<TurnRecord> {
            self.turns.lock().unwrap().clone()
        }
    }

    #[async_trait::async_trait]
    impl MemoryWriteBack for FakeWriteBack {
        async fn write_back(&self, turn: &TurnRecord) -> Result<()> {
            self.turns.lock().unwrap().push(turn.clone());
            if self.fail {
                return Err(openjiuwen_common::error::Error::memory("write down"));
            }
            Ok(())
        }
    }

    #[test]
    fn name_priority_and_defaults() {
        let rail = MemoryRail::new();
        assert_eq!(rail.name(), "memory");
        assert_eq!(rail.priority(), rail_priority::MEMORY);
        assert_eq!(rail.priority(), 80);
        assert_eq!(rail.config(), MemoryRailConfig::default());
        assert!(rail.recall().is_none());
        assert!(rail.write_back().is_none());
        assert_eq!(MemoryRailConfig::default().recall_limit, 5);
        assert_eq!(MemoryRailConfig::default().max_context_chars, 2000);
    }

    #[test]
    fn config_builders_override_defaults() {
        let config = MemoryRailConfig::default()
            .with_recall_limit(2)
            .with_max_context_chars(50)
            .with_publish_prompt_section(false)
            .with_write_back(false);
        assert_eq!(config.recall_limit, 2);
        assert_eq!(config.max_context_chars, 50);
        assert!(!config.publish_prompt_section);
        assert!(!config.write_back);
    }

    #[test]
    fn recalled_memory_builders() {
        let memory = RecalledMemory::new("likes rust")
            .with_score(0.9)
            .with_source("test");
        assert_eq!(memory.content, "likes rust");
        assert_eq!(memory.score, 0.9);
        assert_eq!(memory.source.as_deref(), Some("test"));
    }

    #[tokio::test]
    async fn publishes_the_prompt_section_without_any_seam() {
        let rail = MemoryRail::new();
        let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeModelCall);
        rail.before_model_call(&mut ctx).await.unwrap();

        let sections = prompt_sections(&ctx);
        assert_eq!(sections.len(), 1);
        assert_eq!(sections[0].name, crate::prompts::SectionName::MEMORY);
        assert!(ctx.extra().get(MEMORY_CONTEXT_EXTRA_KEY).is_none());
        assert!(ctx.extra().get(MEMORY_QUERY_EXTRA_KEY).is_none());
    }

    #[tokio::test]
    async fn injects_one_memory_message_after_the_leading_system_messages() {
        let recall = FakeRecall::returning(vec![
            RecalledMemory::new("prefers tabs").with_score(0.8),
            RecalledMemory::new("uses vim").with_score(0.4),
        ]);
        let rail = MemoryRail::new().with_recall(recall.clone());
        let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeModelCall);
        ctx.base.messages = vec![
            BaseMessage::system("identity"),
            BaseMessage::system("workspace"),
            BaseMessage::user("how do I indent?"),
        ];

        rail.before_model_call(&mut ctx).await.unwrap();

        assert_eq!(recall.calls(), vec![("how do I indent?".to_string(), 5)]);
        assert_eq!(ctx.base.messages.len(), 4);
        assert_eq!(ctx.base.messages[0].role, Role::System);
        assert_eq!(ctx.base.messages[1].role, Role::System);
        assert_eq!(ctx.base.messages[2].role, Role::System);
        let injected = ctx.base.messages[2].content.to_plain_text();
        assert!(injected.starts_with("<memory-context>"));
        assert!(injected.ends_with("</memory-context>"));
        assert!(injected.contains("- prefers tabs"));
        assert!(injected.contains("- uses vim"));
        assert_eq!(ctx.base.messages[3].role, Role::User);

        assert_eq!(
            ctx.extra()
                .get(MEMORY_CONTEXT_EXTRA_KEY)
                .and_then(Value::as_str),
            Some(injected.as_str())
        );
        assert_eq!(
            ctx.extra()
                .get(MEMORY_QUERY_EXTRA_KEY)
                .and_then(Value::as_str),
            Some("how do I indent?")
        );
    }

    #[tokio::test]
    async fn query_falls_back_to_the_invocation_input() {
        let recall = FakeRecall::returning(vec![RecalledMemory::new("a memory")]);
        let rail = MemoryRail::new().with_recall(recall.clone());
        let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeModelCall);
        ctx.base.input = Some(openjiuwen_agent::AgentInput {
            query: InputQuery::Text("from input".to_string()),
            ..openjiuwen_agent::AgentInput::default()
        });

        rail.before_model_call(&mut ctx).await.unwrap();
        assert_eq!(recall.calls(), vec![("from input".to_string(), 5)]);
        assert_eq!(ctx.base.messages.len(), 1);
        assert_eq!(ctx.base.messages[0].role, Role::System);
    }

    #[tokio::test]
    async fn empty_recall_injects_nothing() {
        let recall = FakeRecall::returning(Vec::new());
        let rail = MemoryRail::new().with_recall(recall);
        let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeModelCall);
        ctx.base.messages = vec![BaseMessage::user("hi")];

        rail.before_model_call(&mut ctx).await.unwrap();
        assert_eq!(ctx.base.messages.len(), 1);
        assert!(ctx.extra().get(MEMORY_CONTEXT_EXTRA_KEY).is_none());
    }

    #[tokio::test]
    async fn recall_error_is_swallowed() {
        let rail = MemoryRail::new().with_recall(FakeRecall::failing());
        let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeModelCall);
        ctx.base.messages = vec![BaseMessage::user("hi")];

        rail.before_model_call(&mut ctx).await.unwrap();
        assert_eq!(ctx.base.messages.len(), 1);
        assert!(ctx.extra().get(MEMORY_CONTEXT_EXTRA_KEY).is_none());
        assert_eq!(
            ctx.extra()
                .get(MEMORY_QUERY_EXTRA_KEY)
                .and_then(Value::as_str),
            Some("hi")
        );
    }

    #[tokio::test]
    async fn context_block_respects_the_character_cap_but_keeps_one_entry() {
        let recall = FakeRecall::returning(vec![
            RecalledMemory::new("aaaa"),
            RecalledMemory::new("bbbb"),
            RecalledMemory::new("cccc"),
        ]);
        let rail = MemoryRail::new()
            .with_recall(recall)
            .with_config(MemoryRailConfig::default().with_max_context_chars(8));
        let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeModelCall);
        ctx.base.messages = vec![BaseMessage::user("hi")];

        rail.before_model_call(&mut ctx).await.unwrap();
        let block = ctx.base.messages[0].content.to_plain_text();
        assert!(block.contains("- aaaa"));
        assert!(!block.contains("- bbbb"));

        let recall = FakeRecall::returning(vec![RecalledMemory::new("too long to fit")]);
        let rail = MemoryRail::new()
            .with_recall(recall)
            .with_config(MemoryRailConfig::default().with_max_context_chars(1));
        let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeModelCall);
        ctx.base.messages = vec![BaseMessage::user("hi")];
        rail.before_model_call(&mut ctx).await.unwrap();
        assert!(ctx.base.messages[0]
            .content
            .to_plain_text()
            .contains("- too long to fit"));
    }

    #[tokio::test]
    async fn write_back_receives_query_response_and_tool_names() {
        let writer = FakeWriteBack::new();
        let rail = MemoryRail::new().with_write_back(writer.clone());
        let mut ctx = HarnessRailContext::for_event(RailEvent::AfterInvoke);
        ctx.extra_mut().insert(
            MEMORY_QUERY_EXTRA_KEY.to_string(),
            json!("what time is it?"),
        );
        ctx.base.output = Some(AgentOutput::answer("noon"));
        ctx.base.messages = vec![
            BaseMessage::user("what time is it?"),
            BaseMessage::assistant("")
                .with_tool_calls(vec![ToolCall::new("bash"), ToolCall::new("read_file")]),
            BaseMessage::tool("call-1", "12:00"),
        ];

        rail.after_invoke(&mut ctx).await.unwrap();
        let turns = writer.turns();
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].query, "what time is it?");
        assert_eq!(turns[0].response.as_deref(), Some("noon"));
        assert_eq!(turns[0].tool_calls, vec!["bash", "read_file"]);
    }

    #[tokio::test]
    async fn write_back_is_skipped_without_a_query_or_seam() {
        let writer = FakeWriteBack::new();
        let rail = MemoryRail::new().with_write_back(writer.clone());
        let mut ctx = HarnessRailContext::for_event(RailEvent::AfterInvoke);
        ctx.base.output = Some(AgentOutput::answer("noon"));
        rail.after_invoke(&mut ctx).await.unwrap();
        assert!(writer.turns().is_empty());

        let rail = MemoryRail::new();
        let mut ctx = HarnessRailContext::for_event(RailEvent::AfterInvoke);
        ctx.extra_mut()
            .insert(MEMORY_QUERY_EXTRA_KEY.to_string(), json!("q"));
        rail.after_invoke(&mut ctx).await.unwrap();
    }

    #[tokio::test]
    async fn write_back_error_is_swallowed() {
        let rail = MemoryRail::new().with_write_back(FakeWriteBack::failing());
        let mut ctx = HarnessRailContext::for_event(RailEvent::AfterInvoke);
        ctx.extra_mut()
            .insert(MEMORY_QUERY_EXTRA_KEY.to_string(), json!("q"));
        rail.after_invoke(&mut ctx).await.unwrap();
    }

    #[tokio::test]
    async fn writes_can_be_disabled() {
        let writer = FakeWriteBack::new();
        let rail = MemoryRail::new()
            .with_write_back(writer.clone())
            .with_config(MemoryRailConfig::default().with_write_back(false));
        let mut ctx = HarnessRailContext::for_event(RailEvent::AfterInvoke);
        ctx.extra_mut()
            .insert(MEMORY_QUERY_EXTRA_KEY.to_string(), json!("q"));
        rail.after_invoke(&mut ctx).await.unwrap();
        assert!(writer.turns().is_empty());
    }

    #[tokio::test]
    async fn long_term_memory_adapter_round_trips_a_turn() {
        let memory = LongTermMemory::in_memory(MemoryEngineConfig::default())
            .await
            .unwrap();
        let adapter =
            LongTermMemoryAdapter::new(memory, MemoryScope::default()).with_source("test-adapter");

        adapter
            .write_back(&TurnRecord {
                query: "favorite color".to_string(),
                response: Some("blue".to_string()),
                tool_calls: Vec::new(),
            })
            .await
            .unwrap();

        let recalled = adapter.recall("favorite color", 5).await.unwrap();
        assert!(!recalled.is_empty());
        assert!(recalled
            .iter()
            .any(|memory| memory.content.contains("User: favorite color")));
        assert!(recalled
            .iter()
            .any(|memory| memory.content.contains("Assistant: blue")));
        assert_eq!(recalled[0].source.as_deref(), Some("long_term_memory"));
    }

    #[tokio::test]
    async fn long_term_memory_adapter_skips_blank_turns() {
        let memory = LongTermMemory::in_memory(MemoryEngineConfig::default())
            .await
            .unwrap();
        let adapter = LongTermMemoryAdapter::new(memory, MemoryScope::default());
        adapter
            .write_back(&TurnRecord {
                query: "   ".to_string(),
                response: Some("ignored".to_string()),
                tool_calls: Vec::new(),
            })
            .await
            .unwrap();
        assert!(adapter.recall("ignored", 5).await.unwrap().is_empty());
    }

    #[test]
    fn adapter_debug_redacts_nothing_sensitive() {
        // No facade is needed for the Debug shape; this only asserts the type is
        // printable without exposing a memory store handle.
        fn assert_debug<T: std::fmt::Debug>() {}
        assert_debug::<LongTermMemoryAdapter>();
        assert_debug::<MemoryRail>();
    }
}
