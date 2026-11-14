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
//! The context is split by responsibility: `history` holds the [`ModelContext`]
//! implementation, `state` persists messages and processor checkpoints,
//! `window_tracking` keeps and diffs the last two LLM-bound windows, `usage`
//! builds the `context.usage` block, and `prepare` assembles the request
//! payload.
//!
//! [`prepare`]: ConversationContext::prepare
//! [`ConversationContext`]: crate::context::ConversationContext
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Instant;

use async_trait::async_trait;
use openjiuwen_common::error::Result;
use openjiuwen_foundation::llm::Model;
use openjiuwen_foundation::prompt::{BaseMessage, TokenCounter};
use openjiuwen_session::Session;
use tokio::sync::{Mutex, RwLock};
use tokio_util::sync::CancellationToken;

use crate::config::ContextConfig;
use crate::offload::{OffloadScope, OffloadStore};
use crate::policy::ContextWindowSnapshot;
use crate::processor::{ContextPipeline, ContextProcessor};
use crate::recall::RecallArchiveStore;
use crate::session_memory::SessionMemoryStore;
use crate::stats::ContextStats;
use crate::telemetry::ContextProcessorStateRecorder;
use crate::usage::ContextUsageAnalyzer;

mod history;
mod prepare;
mod state;
mod usage;
mod window_tracking;

pub use state::ContextState;

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
    /// Returns [`Error::Validation`](openjiuwen_common::error::Error::Validation)
    /// when the implementation rejects a message.
    async fn set_messages(&self, messages: Vec<BaseMessage>) -> Result<()>;

    /// Appends one message.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](openjiuwen_common::error::Error::Validation)
    /// when the message is malformed.
    async fn add_message(&self, message: BaseMessage) -> Result<()>;

    /// Removes and returns the newest `count` messages.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](openjiuwen_common::error::Error::Validation)
    /// when `count` is zero.
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

/// The last two LLM-bound windows and when the current one was prepared.
///
/// [`ConversationContext::detect_window_change`] diffs `previous` against
/// `current`; keeping only the two most recent windows makes the KV-cache diff
/// O(messages) and never touches the stored history.
#[derive(Debug, Default)]
struct WindowTracking {
    /// The window prepared before the current one, when there is one.
    previous: Option<ContextWindowSnapshot>,
    /// The most recent window `prepare` assembled.
    current: Option<ContextWindowSnapshot>,
    /// When the current window was prepared.
    accessed: Option<Instant>,
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
    /// Archive store compressors keep replaced history in, when configured.
    recall: Option<Arc<dyn RecallArchiveStore>>,
    /// Session-memory notes full compaction reinjects, when configured.
    session_memory: Option<Arc<dyn SessionMemoryStore>>,
    /// Recorder every processor pass of this context reports to, when set.
    recorder: Option<Arc<ContextProcessorStateRecorder>>,
    /// Analyzer that fills the prepared `context.usage` block, when configured.
    usage_analyzer: Option<Arc<ContextUsageAnalyzer>>,
    /// The last LLM-bound windows and their access time, for KV-cache diffing.
    windows: StdMutex<WindowTracking>,
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
    /// the engine. `recall` and `session_memory` are the stores the pipeline
    /// hands to processors; both are optional.
    pub(crate) fn new(
        context_id: impl Into<String>,
        session_id: impl Into<String>,
        config: Arc<ContextConfig>,
        counter: Arc<dyn TokenCounter + Send + Sync>,
        pipeline: ContextPipeline,
        offload: Arc<dyn OffloadStore>,
        model: Option<Arc<dyn Model>>,
        session: Option<Arc<dyn Session>>,
        recall: Option<Arc<dyn RecallArchiveStore>>,
        session_memory: Option<Arc<dyn SessionMemoryStore>>,
        recorder: Option<Arc<ContextProcessorStateRecorder>>,
        usage_analyzer: Option<Arc<ContextUsageAnalyzer>>,
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
            recall,
            session_memory,
            recorder,
            usage_analyzer,
            windows: StdMutex::new(WindowTracking::default()),
        }
    }
}

impl ConversationContext {
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

    /// The recall archive store compressors write replaced history to.
    ///
    /// `None` when compression recall is disabled and no store was configured
    /// on the engine or through [`ContextOptions`](crate::engine::ContextOptions).
    #[must_use]
    pub fn recall_store(&self) -> Option<&Arc<dyn RecallArchiveStore>> {
        self.recall.as_ref()
    }

    /// The session-memory store full compaction reinjects notes from.
    ///
    /// `None` when neither the engine nor [`ContextOptions`](crate::engine::ContextOptions)
    /// configured a store and the processor pipeline has no full compaction or
    /// summary-offload processor.
    #[must_use]
    pub fn session_memory_store(&self) -> Option<&Arc<dyn SessionMemoryStore>> {
        self.session_memory.as_ref()
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

    /// The recorder this context reports every processor pass to, when set.
    #[must_use]
    pub fn recorder(&self) -> Option<&Arc<ContextProcessorStateRecorder>> {
        self.recorder.as_ref()
    }

    /// Whether an instrumented pipeline pass is available.
    pub(crate) fn has_recorder(&self) -> bool {
        self.pipeline.recorder().is_some()
    }

    /// The compression lock, `try_lock`ed by the engine's `compress`.
    pub(crate) fn compression_lock(&self) -> &Mutex<()> {
        &self.compression
    }
}

#[cfg(test)]
mod tests;
