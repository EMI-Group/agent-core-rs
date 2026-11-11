//! Conversation contexts: the mutable history a model call is assembled from.
//!
//! [`ModelContext`] is the trait an agent holds; [`ConversationContext`] is the
//! built-in implementation. A context owns its messages, the token counter used
//! to size them, the processor pipeline snapshot taken at creation time, and the
//! offload store its processors write to — everything [`prepare`] needs to build
//! a request payload without touching engine state.
//!
//! ```
//! use std::sync::Arc;
//!
//! use openjiuwen_context::{
//!     ContextConfig, ContextEngine, ModelContext, PrepareOptions,
//! };
//! use openjiuwen_foundation::prompt::BaseMessage;
//!
//! # async fn example() -> openjiuwen_common::error::Result<()> {
//! let engine = ContextEngine::new(ContextConfig::default());
//! let context = engine.create_context("main", None).await?;
//! context.add_message(BaseMessage::user("hi")).await?;
//! let prepared = engine.prepare(&context, &PrepareOptions::default()).await?;
//! assert_eq!(prepared.messages.len(), 1);
//! # Ok(())
//! # }
//! ```
//!
//! [`prepare`]: ConversationContext::prepare
//! [`ConversationContext`]: crate::context::ConversationContext
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::Model;
use openjiuwen_foundation::prompt::{BaseMessage, MessageContent, Role, TokenCounter};
use openjiuwen_session::Session;
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, RwLock};
use tokio_util::sync::CancellationToken;

use crate::config::ContextConfig;
use crate::engine::PrepareOptions;
use crate::events::ContextEvent;
use crate::offload::{OffloadScope, OffloadStore};
use crate::processor::{ContextPipeline, ContextProcessor, ProcessorContext};
use crate::stats::{compute_stats, ContextStats, PreparedContext};
use crate::tokens::count_messages;
use crate::window::{assemble_window, drop_orphan_tool_results, WindowOptions};

/// The conversation history an agent talks to.
///
/// Every method is asynchronous because implementations may persist or read
/// remote state; [`ConversationContext`] itself only guards an in-process
/// vector.
#[async_trait]
pub trait ModelContext: Send + Sync {
    /// The context identifier as the caller spelled it (dots preserved).
    fn context_id(&self) -> &str;

    /// The session this context belongs to, or the default session id.
    fn session_id(&self) -> &str;

    /// A snapshot of the stored messages, oldest first.
    async fn messages(&self) -> Vec<BaseMessage>;

    /// Replaces the whole history.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the implementation rejects a message.
    async fn set_messages(&self, messages: Vec<BaseMessage>) -> Result<()>;

    /// Appends one message.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the message is malformed.
    async fn add_message(&self, message: BaseMessage) -> Result<()>;

    /// Removes and returns the newest `count` messages.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `count` is zero.
    async fn pop_messages(&self, count: usize) -> Result<Vec<BaseMessage>>;

    /// Removes every message.
    ///
    /// # Errors
    ///
    /// Returns an error when the removal cannot be persisted.
    async fn clear_messages(&self) -> Result<()>;

    /// The counter this context sizes messages with.
    fn token_counter(&self) -> Arc<dyn TokenCounter>;

    /// Statistics of the stored history, including compression counters.
    async fn statistics(&self) -> ContextStats;

    /// The revision number, incremented on every mutation.
    fn revision(&self) -> u64;
}

/// The persistable state of a [`ConversationContext`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ContextState {
    /// The stored messages, oldest first.
    pub messages: Vec<BaseMessage>,
    /// The revision the messages were saved at.
    pub revision: u64,
}

/// Whether a message carries no payload at all.
///
/// Only [`MessageContent::Text`] that is empty and a [`MessageContent::Parts`]
/// list with no entries count as empty; a part list holding an image is *not*
/// empty, so multimodal messages are never rejected.
fn is_empty_content(content: &MessageContent) -> bool {
    match content {
        MessageContent::Text(text) => text.is_empty(),
        MessageContent::Parts(parts) => parts.is_empty(),
    }
}

/// The built-in [`ModelContext`]: messages behind an `RwLock`, a revision
/// counter, and the pipeline snapshot taken when the context was created.
pub struct ConversationContext {
    /// Context identifier as the caller spelled it.
    context_id: String,
    /// Owning session id (or the engine's default).
    session_id: String,
    /// Engine configuration snapshot.
    config: Arc<ContextConfig>,
    /// The stored history.
    messages: RwLock<Vec<BaseMessage>>,
    /// Mutation counter, used as the statistics cache key.
    revision: AtomicU64,
    /// Counter shared with processors.
    counter: Arc<dyn TokenCounter + Send + Sync>,
    /// Processor pipeline, fixed for the context's lifetime.
    pipeline: ContextPipeline,
    /// Store processors write offloaded content to.
    offload: Arc<dyn OffloadStore>,
    /// Scope offload records are keyed under.
    scope: OffloadScope,
    /// Model available to summarisation processors.
    model: Option<Arc<dyn Model>>,
    /// Cancellation signalled to processors.
    cancellation: CancellationToken,
    /// Held for the duration of one [`ContextEngine::compress`] pass.
    ///
    /// [`ContextEngine::compress`]: crate::engine::ContextEngine::compress
    compression: Mutex<()>,
    /// `(revision, statistics)` cache, invalidated on every mutation.
    stats_cache: Mutex<Option<(u64, ContextStats)>>,
    /// Accumulated `(compressions, tokens_before, tokens_after)`.
    compression_totals: Mutex<(usize, usize, usize)>,
    /// Session handle this context was created with, when one was given.
    session: Option<Arc<dyn Session>>,
}

impl std::fmt::Debug for ConversationContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConversationContext")
            .field("context_id", &self.context_id)
            .field("session_id", &self.session_id)
            .field("revision", &self.revision())
            .field("processors", &self.pipeline)
            .finish_non_exhaustive()
    }
}

impl ConversationContext {
    /// Builds a context; called by the engine only.
    ///
    /// The offload scope is keyed by the raw `(session_id, context_id)` pair so
    /// handles stay unique per context, while the pool identity is normalized by
    /// the engine.
    pub(crate) fn new(
        context_id: impl Into<String>,
        session_id: impl Into<String>,
        config: Arc<ContextConfig>,
        counter: Arc<dyn TokenCounter + Send + Sync>,
        pipeline: ContextPipeline,
        offload: Arc<dyn OffloadStore>,
        model: Option<Arc<dyn Model>>,
        session: Option<Arc<dyn Session>>,
    ) -> Self {
        let context_id = context_id.into();
        let session_id = session_id.into();
        let scope = OffloadScope::new(session_id.clone(), context_id.clone());
        Self {
            context_id,
            session_id,
            config,
            messages: RwLock::new(Vec::new()),
            revision: AtomicU64::new(0),
            counter,
            pipeline,
            offload,
            scope,
            model,
            cancellation: CancellationToken::new(),
            compression: Mutex::new(()),
            stats_cache: Mutex::new(None),
            compression_totals: Mutex::new((0, 0, 0)),
            session,
        }
    }

    /// The engine configuration this context was created with.
    #[must_use]
    pub fn config(&self) -> &ContextConfig {
        &self.config
    }

    /// The model available to summarisation processors.
    #[must_use]
    pub fn model(&self) -> Option<Arc<dyn Model>> {
        self.model.clone()
    }

    /// The processor pipeline, in execution order.
    #[must_use]
    pub fn processors(&self) -> &[Arc<dyn ContextProcessor>] {
        self.pipeline.processors()
    }

    /// The store offloaded content is written to.
    #[must_use]
    pub fn offload_store(&self) -> &Arc<dyn OffloadStore> {
        &self.offload
    }

    /// The offload scope of this context.
    #[must_use]
    pub fn scope(&self) -> OffloadScope {
        self.scope.clone()
    }

    /// The session handle this context was created with, when one was supplied.
    ///
    /// A context restored by [`ContextEngine::load_contexts`] has none, because
    /// that method only borrows the session.
    ///
    /// [`ContextEngine::load_contexts`]: crate::engine::ContextEngine::load_contexts
    #[must_use]
    pub fn session(&self) -> Option<&Arc<dyn Session>> {
        self.session.as_ref()
    }

    /// Cancels every future processor run of this context.
    pub fn cancel(&self) {
        self.cancellation.cancel();
    }

    /// A handle to this context's cancellation token.
    #[must_use]
    pub fn cancellation(&self) -> CancellationToken {
        self.cancellation.clone()
    }

    /// The compression lock, `try_lock`ed by the engine's `compress`.
    pub(crate) fn compression_lock(&self) -> &Mutex<()> {
        &self.compression
    }

    /// Runs the processor pipeline over `messages`.
    ///
    /// `manual` bypasses the threshold check, which is what an explicit
    /// compression asks for.
    pub(crate) async fn run_pipeline(
        &self,
        messages: &mut Vec<BaseMessage>,
        manual: bool,
    ) -> Result<Vec<ContextEvent>> {
        let cx = ProcessorContext {
            config: &self.config,
            counter: self.counter.as_ref(),
            model: self.model.as_ref(),
            offload: self.offload.as_ref(),
            scope: &self.scope,
            budget: self.config.effective_max_tokens(),
            trigger_tokens: self.config.compression_trigger_tokens(),
            manual,
            cancellation: &self.cancellation,
        };
        self.pipeline.run(messages, &cx).await
    }

    /// Serializes the history for persistence.
    pub async fn save_state(&self) -> ContextState {
        ContextState {
            messages: self.messages().await,
            revision: self.revision(),
        }
    }

    /// Replaces the history with `state`.
    ///
    /// The revision stored in `state` becomes the context's revision, so a
    /// restored context continues the sequence the saved one used.
    ///
    /// # Errors
    ///
    /// Currently infallible; the signature keeps the door open for
    /// implementations that persist eagerly.
    pub async fn load_state(&self, state: ContextState) -> Result<()> {
        *self.messages.write().await = state.messages;
        self.revision.store(state.revision, Ordering::SeqCst);
        self.invalidate_stats().await;
        Ok(())
    }

    /// Builds the payload to send to the model.
    ///
    /// The pipeline runs on a copy of the history, so the stored messages are
    /// never mutated: [`ContextEngine::prepare`] is a read-only operation. The
    /// window is assembled after the processors ran, and orphan tool results left
    /// behind by round selection are dropped last.
    ///
    /// When [`PrepareOptions::timeout`] is set, the whole body is wrapped in
    /// [`tokio::time::timeout`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Cancelled`] when the caller's token or the context's own
    /// token is set, [`Error::Timeout`] when the deadline expires, and the
    /// pipeline's error (for example [`Error::Llm`]) when a processor fails.
    ///
    /// [`ContextEngine::prepare`]: crate::engine::ContextEngine::prepare
    pub async fn prepare(&self, options: &PrepareOptions) -> Result<PreparedContext> {
        match options.timeout {
            Some(timeout) => tokio::time::timeout(timeout, self.prepare_inner(options))
                .await
                .map_err(|_| {
                    Error::Timeout(format!("context preparation timed out after {timeout:?}"))
                })?,
            None => self.prepare_inner(options).await,
        }
    }

    /// The body of [`ConversationContext::prepare`], without the deadline.
    async fn prepare_inner(&self, options: &PrepareOptions) -> Result<PreparedContext> {
        if options.cancellation.is_cancelled() || self.cancellation.is_cancelled() {
            return Err(Error::Cancelled);
        }
        let mut working = self.messages().await;
        let events = self.run_pipeline(&mut working, false).await?;
        let mut messages = assemble_window(
            &working,
            &options.system_messages,
            &WindowOptions {
                window_size: options.window_size,
                dialogue_rounds: options.dialogue_rounds,
            },
            &self.config,
        );
        drop_orphan_tool_results(&mut messages);
        let stats = compute_stats(&messages, self.counter.as_ref());
        let token_count = count_messages(self.counter.as_ref(), &messages);
        Ok(PreparedContext {
            messages,
            tools: options.tools.clone(),
            stats,
            token_count,
            overflow: token_count > self.config.effective_max_tokens(),
            window_size: options.window_size.or(self.config.default_window_messages),
            events,
        })
    }

    /// Records one compression pass in the statistics.
    pub(crate) async fn record_compression(&self, tokens_before: usize, tokens_after: usize) {
        let mut totals = self.compression_totals.lock().await;
        totals.0 = totals.0.saturating_add(1);
        totals.1 = totals.1.saturating_add(tokens_before);
        totals.2 = totals.2.saturating_add(tokens_after);
    }

    /// Drops the cached statistics so the next read recomputes them.
    async fn invalidate_stats(&self) {
        *self.stats_cache.lock().await = None;
    }

    /// Bumps the revision and invalidates the statistics cache.
    ///
    /// The message guard must already be released: the cache and the message
    /// vector are locked in that order.
    async fn bump_revision(&self) {
        self.revision.fetch_add(1, Ordering::SeqCst);
        self.invalidate_stats().await;
    }

    /// Rejects the messages this context cannot store.
    ///
    /// The rule is deliberately small: a user or tool message must carry a
    /// payload, because those roles are meaningless when empty. Assistant
    /// messages may be empty (a tool-call turn has no text) and system messages
    /// are left to the caller, matching Python's tolerant context.
    fn validate_message(message: &BaseMessage) -> Result<()> {
        match message.role {
            Role::User | Role::Tool if is_empty_content(&message.content) => {
                Err(Error::Validation(format!(
                    "{} message content must not be empty",
                    message.role
                )))
            }
            _ => Ok(()),
        }
    }
}

#[async_trait]
impl ModelContext for ConversationContext {
    fn context_id(&self) -> &str {
        &self.context_id
    }

    fn session_id(&self) -> &str {
        &self.session_id
    }

    async fn messages(&self) -> Vec<BaseMessage> {
        self.messages.read().await.clone()
    }

    async fn set_messages(&self, messages: Vec<BaseMessage>) -> Result<()> {
        *self.messages.write().await = messages;
        self.bump_revision().await;
        Ok(())
    }

    async fn add_message(&self, message: BaseMessage) -> Result<()> {
        Self::validate_message(&message)?;
        self.messages.write().await.push(message);
        self.bump_revision().await;
        Ok(())
    }

    async fn pop_messages(&self, count: usize) -> Result<Vec<BaseMessage>> {
        if count == 0 {
            return Err(Error::Validation(
                "pop size should be larger than 0".to_owned(),
            ));
        }
        let mut messages = self.messages.write().await;
        let keep = messages.len().saturating_sub(count);
        let popped = messages.split_off(keep);
        drop(messages);
        if !popped.is_empty() {
            self.bump_revision().await;
        }
        Ok(popped)
    }

    async fn clear_messages(&self) -> Result<()> {
        self.messages.write().await.clear();
        self.bump_revision().await;
        Ok(())
    }

    fn token_counter(&self) -> Arc<dyn TokenCounter> {
        self.counter.clone()
    }

    async fn statistics(&self) -> ContextStats {
        let revision = self.revision();
        let base = {
            let mut cache = self.stats_cache.lock().await;
            match cache.as_ref() {
                Some((cached, stats)) if *cached == revision => stats.clone(),
                _ => {
                    let messages = self.messages.read().await.clone();
                    let stats = compute_stats(&messages, self.counter.as_ref());
                    *cache = Some((revision, stats.clone()));
                    stats
                }
            }
        };
        let totals = *self.compression_totals.lock().await;
        ContextStats {
            compressions: totals.0,
            tokens_before_compression: totals.1,
            tokens_after_compression: totals.2,
            ..base
        }
    }

    fn revision(&self) -> u64 {
        self.revision.load(Ordering::SeqCst)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use openjiuwen_foundation::prompt::ApproxTokenCounter;

    use crate::offload::InMemoryOffloadStore;
    use crate::processor::{ProcessorConfig, TruncationConfig};

    fn counter() -> Arc<dyn TokenCounter + Send + Sync> {
        Arc::new(ApproxTokenCounter::with_chars_per_token(1))
    }

    fn config() -> Arc<ContextConfig> {
        Arc::new(ContextConfig::default())
    }

    /// A config whose budget is small enough to overflow with a few messages.
    fn small_config() -> Arc<ContextConfig> {
        Arc::new(ContextConfig {
            max_context_tokens: Some(10),
            ..ContextConfig::default()
        })
    }

    fn build(config: Arc<ContextConfig>, processors: Vec<ProcessorConfig>) -> ConversationContext {
        let store: Arc<dyn OffloadStore> = Arc::new(InMemoryOffloadStore::new());
        let pipeline = ContextPipeline::from_configs(&processors, None, Arc::clone(&store))
            .unwrap_or_else(|error| panic!("pipeline should build: {error}"));
        ConversationContext::new(
            "main",
            "session",
            config,
            counter(),
            pipeline,
            store,
            None,
            None,
        )
    }

    fn truncating(max_tokens: usize) -> ConversationContext {
        build(
            small_config(),
            vec![ProcessorConfig::Truncation(TruncationConfig {
                max_tokens: Some(max_tokens),
                keep_recent_messages: 2,
                keep_system: true,
            })],
        )
    }

    /// A processor that sleeps so a deadline can expire mid-run.
    struct Sleeper;

    #[async_trait]
    impl ContextProcessor for Sleeper {
        fn name(&self) -> &'static str {
            "sleeper"
        }

        async fn process(
            &self,
            _messages: &mut Vec<BaseMessage>,
            _cx: &ProcessorContext<'_>,
        ) -> Result<Option<ContextEvent>> {
            tokio::time::sleep(Duration::from_millis(200)).await;
            Ok(None)
        }
    }

    #[tokio::test]
    async fn prepare_does_not_mutate_the_stored_history() {
        let context = truncating(8);
        let messages = vec![
            BaseMessage::user("first question"),
            BaseMessage::assistant("first answer"),
            BaseMessage::user("second question"),
            BaseMessage::assistant("second answer"),
        ];
        context.set_messages(messages.clone()).await.unwrap();

        let prepared = context.prepare(&PrepareOptions::default()).await.unwrap();
        assert!(prepared.messages.len() < messages.len());
        assert!(prepared.overflow);
        assert_eq!(context.messages().await, messages);
    }

    #[tokio::test]
    async fn prepare_puts_system_messages_first_and_honours_the_window() {
        let context = build(config(), Vec::new());
        context
            .set_messages(vec![
                BaseMessage::user("q1"),
                BaseMessage::assistant("a1"),
                BaseMessage::user("q2"),
                BaseMessage::assistant("a2"),
            ])
            .await
            .unwrap();

        let options = PrepareOptions {
            system_messages: vec![BaseMessage::system("persona")],
            dialogue_rounds: Some(2),
            window_size: Some(5),
            tools: vec![serde_json::json!({"name": "read_file"})],
            ..PrepareOptions::default()
        };
        let prepared = context.prepare(&options).await.unwrap();
        assert_eq!(prepared.messages.len(), 5);
        assert_eq!(prepared.messages[0].role, Role::System);
        assert_eq!(prepared.messages[0].text(), "persona");
        assert_eq!(prepared.messages[1].text(), "q1");
        assert_eq!(prepared.tools.len(), 1);
        assert_eq!(prepared.window_size, Some(5));
        assert!(!prepared.overflow);
        assert_eq!(prepared.stats.total_messages, 5);

        // A round window keeps only the newest round.
        let one_round = PrepareOptions {
            dialogue_rounds: Some(1),
            ..PrepareOptions::default()
        };
        let prepared = context.prepare(&one_round).await.unwrap();
        assert_eq!(prepared.messages.len(), 2);
        assert_eq!(prepared.messages[0].text(), "q2");
    }

    #[tokio::test]
    async fn prepare_maps_an_expired_deadline_to_timeout() {
        let store: Arc<dyn OffloadStore> = Arc::new(InMemoryOffloadStore::new());
        let context = ConversationContext::new(
            "main",
            "session",
            config(),
            counter(),
            ContextPipeline::new(vec![Arc::new(Sleeper)]),
            store,
            None,
            None,
        );
        let options = PrepareOptions {
            timeout: Some(Duration::from_millis(1)),
            ..PrepareOptions::default()
        };
        let error = context.prepare(&options).await.unwrap_err();
        assert!(matches!(error, Error::Timeout(_)));
    }

    #[tokio::test]
    async fn prepare_refuses_a_cancelled_request_or_context() {
        let context = build(config(), Vec::new());
        let cancelled = PrepareOptions {
            cancellation: CancellationToken::new(),
            ..PrepareOptions::default()
        };
        cancelled.cancellation.cancel();
        assert!(matches!(
            context.prepare(&cancelled).await.unwrap_err(),
            Error::Cancelled
        ));

        context.cancel();
        assert!(context.cancellation().is_cancelled());
        assert!(matches!(
            context
                .prepare(&PrepareOptions::default())
                .await
                .unwrap_err(),
            Error::Cancelled
        ));
    }

    #[tokio::test]
    async fn history_mutations_bump_the_revision_and_keep_stats_fresh() {
        let context = build(config(), Vec::new());
        assert_eq!(context.revision(), 0);

        context
            .add_message(BaseMessage::user("abcd"))
            .await
            .unwrap();
        assert_eq!(context.revision(), 1);
        let stats = context.statistics().await;
        assert_eq!(stats.total_messages, 1);
        assert_eq!(stats.user_messages, 1);

        let cached = context.statistics().await;
        assert_eq!(cached, stats);

        context
            .add_message(BaseMessage::assistant("ef"))
            .await
            .unwrap();
        assert_eq!(context.revision(), 2);
        let stats = context.statistics().await;
        assert_eq!(stats.total_messages, 2);
        assert_eq!(stats.assistant_messages, 1);
        assert_eq!(stats.total_dialogues, 1);

        let popped = context.pop_messages(2).await.unwrap();
        assert_eq!(popped.len(), 2);
        assert_eq!(context.revision(), 3);
        assert_eq!(context.messages().await.len(), 0);

        context.add_message(BaseMessage::system("s")).await.unwrap();
        context.clear_messages().await.unwrap();
        assert_eq!(context.revision(), 5);
        assert_eq!(context.statistics().await.total_messages, 0);
        assert_eq!(context.messages().await.len(), 0);
    }

    #[tokio::test]
    async fn pop_rejects_zero_and_clamps_to_the_history() {
        let context = build(config(), Vec::new());
        let error = context.pop_messages(0).await.unwrap_err();
        assert!(
            matches!(error, Error::Validation(message) if message == "pop size should be larger than 0")
        );

        context
            .set_messages(vec![BaseMessage::user("a"), BaseMessage::user("b")])
            .await
            .unwrap();
        let popped = context.pop_messages(9).await.unwrap();
        assert_eq!(popped.len(), 2);
        assert_eq!(popped[0].text(), "a");
        assert_eq!(context.messages().await.len(), 0);
        assert_eq!(context.pop_messages(1).await.unwrap().len(), 0);
    }

    #[tokio::test]
    async fn empty_user_and_tool_messages_are_rejected_but_empty_assistants_are_not() {
        let context = build(config(), Vec::new());
        let error = context
            .add_message(BaseMessage::user(""))
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Validation(_)));
        assert!(context
            .add_message(BaseMessage::tool("call_1", ""))
            .await
            .is_err());
        context
            .add_message(BaseMessage::assistant(""))
            .await
            .unwrap();
        assert_eq!(context.messages().await.len(), 1);
    }

    #[tokio::test]
    async fn state_round_trips_and_restores_the_revision() {
        let context = build(config(), Vec::new());
        context.add_message(BaseMessage::user("hi")).await.unwrap();
        let state = context.save_state().await;
        assert_eq!(state.revision, 1);
        let json = serde_json::to_string(&state).unwrap();
        let restored: ContextState = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, state);

        let other = build(config(), Vec::new());
        other.load_state(restored).await.unwrap();
        assert_eq!(other.messages().await, state.messages);
        assert_eq!(other.revision(), 1);
        assert_eq!(other.statistics().await.total_messages, 1);
    }

    #[tokio::test]
    async fn compression_totals_are_merged_into_statistics() {
        let context = build(config(), Vec::new());
        context
            .set_messages(vec![BaseMessage::user("abcd")])
            .await
            .unwrap();
        context.record_compression(40, 10).await;
        context.record_compression(10, 8).await;

        let stats = context.statistics().await;
        assert_eq!(stats.compressions, 2);
        assert_eq!(stats.tokens_before_compression, 50);
        assert_eq!(stats.tokens_after_compression, 18);
        assert_eq!(stats.total_messages, 1);
    }

    #[test]
    fn context_identity_matches_the_constructor_arguments() {
        let context = build(config(), Vec::new());
        assert_eq!(context.context_id(), "main");
        assert_eq!(context.session_id(), "session");
        assert_eq!(context.scope().session_id, "session");
        assert_eq!(context.scope().context_id, "main");
        assert!(context.model().is_none());
        assert!(context.session().is_none());
        assert!(context.processors().is_empty());
        assert_eq!(
            context.offload_store().kind(),
            crate::offload::OffloadKind::InMemory
        );
        assert!(format!("{context:?}").contains("ConversationContext"));
    }
}
