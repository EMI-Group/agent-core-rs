//! The context engine: a per-session pool of contexts plus compression control.
//!
//! [`ContextEngine`] owns the configuration, the offload store, and the pool of
//! [`ConversationContext`]s keyed by `(session_id, context_id)`. It is the seam
//! agents use:
//!
//! ```
//! use openjiuwen_context::{ContextConfig, ContextEngine, PrepareOptions};
//! use openjiuwen_foundation::prompt::BaseMessage;
//!
//! # async fn example() -> openjiuwen_common::error::Result<()> {
//! let engine = ContextEngine::new(ContextConfig::default());
//! let context = engine.create_context("main", None).await?;
//! engine.observe(&context, BaseMessage::user("hi")).await?;
//! let prepared = engine.prepare(&context, &PrepareOptions::default()).await?;
//! assert_eq!(prepared.messages.len(), 1);
//! # Ok(())
//! # }
//! ```
//!
//! `prepare` is read-only, `observe` is the append hook, and `compress` is the
//! only operation that persists a rewritten history.
//!
//! The entry point is split by responsibility: `pool` creates and looks up
//! contexts, `observe` appends messages, `compression` runs compression passes
//! and provider-overflow recovery, `usage` measures the prepared window and
//! rolls up provider KV-cache usage, and `persistence` saves and restores the
//! pool through session state.
//!
//! Every slice adds inherent methods to [`ContextEngine`], so every method keeps
//! its original public path.
use std::collections::HashMap;
use std::sync::{Arc, Mutex as StdMutex};

use openjiuwen_common::error::{Error, Result};
use tokio::sync::RwLock;

use crate::config::ContextConfig;
use crate::context::ConversationContext;
use crate::offload::{InMemoryOffloadStore, OffloadStore};
use crate::recall::RecallArchiveStore;
use crate::session_memory::SessionMemoryStore;
use crate::stats::PreparedContext;
use crate::telemetry::ContextProcessorStateRecorder;
use crate::usage::{ContextUsageAnalyzer, SessionKVCacheAggregator};

pub use crate::engine_options::{ContextOptions, PrepareOptions, DEFAULT_SESSION_ID};

mod compression;
mod observe;
mod persistence;
mod pool;
mod usage;

/// Session key the engine persists every context under.
const STATE_KEY: &str = "context";

/// The context-service entry point.
///
/// One engine per process (or per agent) owns the configuration and the pool;
/// contexts are `Arc`s, so an agent and its subagents share a context safely.
pub struct ContextEngine {
    /// Engine configuration, shared with every context it creates.
    config: Arc<ContextConfig>,
    /// Store processors write offloaded content to.
    store: Arc<dyn OffloadStore>,
    /// Default recall archive store handed to new contexts.
    recall: Option<Arc<dyn RecallArchiveStore>>,
    /// Default session-memory store handed to new contexts.
    session_memory: Option<Arc<dyn SessionMemoryStore>>,
    /// Recorder new contexts report processor passes to.
    recorder: Option<Arc<ContextProcessorStateRecorder>>,
    /// Analyzer new contexts measure the prepared `context.usage` block with.
    usage_analyzer: Option<Arc<ContextUsageAnalyzer>>,
    /// Provider KV-cache rollups, one bucket per aggregation key.
    kv_cache: StdMutex<SessionKVCacheAggregator>,
    /// `(session_id, normalized context_id)` → context.
    contexts: RwLock<HashMap<(String, String), Arc<ConversationContext>>>,
}

impl std::fmt::Debug for ContextEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContextEngine")
            .field("config", &self.config)
            .field("store", &self.store.kind())
            .field("has_recall", &self.recall.is_some())
            .field("has_session_memory", &self.session_memory.is_some())
            .field("has_recorder", &self.recorder.is_some())
            .field("has_usage_analyzer", &self.usage_analyzer.is_some())
            .field(
                "kv_cache_scopes",
                &self.kv_cache.lock().map_or(0, |cache| cache.scope_count()),
            )
            .finish_non_exhaustive()
    }
}

impl ContextEngine {
    /// Creates an engine with an in-memory offload store.
    #[must_use]
    pub fn new(config: ContextConfig) -> Self {
        Self::with_store(config, Arc::new(InMemoryOffloadStore::new()))
    }

    /// Creates an engine that offloads into `store`.
    ///
    /// No recall or session-memory store is configured, so contexts fall back to
    /// the configuration defaults: an in-memory recall archive when
    /// [`CompressionRecallConfig::enabled`](crate::config::CompressionRecallConfig::enabled)
    /// is set, and an in-memory session-memory store when the pipeline holds a
    /// full-compaction or summary-offload processor.
    #[must_use]
    pub fn with_store(config: ContextConfig, store: Arc<dyn OffloadStore>) -> Self {
        Self::with_stores(config, store, None, None)
    }

    /// Creates an engine with explicit default recall and session-memory stores.
    ///
    /// Contexts created afterwards use these stores unless
    /// [`ContextOptions::recall`] or [`ContextOptions::session_memory`] override
    /// them, which keeps distinct contexts of one engine consistent.
    #[must_use]
    pub fn with_stores(
        config: ContextConfig,
        store: Arc<dyn OffloadStore>,
        recall: Option<Arc<dyn RecallArchiveStore>>,
        session_memory: Option<Arc<dyn SessionMemoryStore>>,
    ) -> Self {
        Self {
            config: Arc::new(config),
            store,
            recall,
            session_memory,
            recorder: None,
            usage_analyzer: None,
            kv_cache: StdMutex::new(SessionKVCacheAggregator::new()),
            contexts: RwLock::new(HashMap::new()),
        }
    }

    /// Attaches the recorder new contexts report every processor pass to.
    ///
    /// A caller configures telemetry once, here, instead of repeating it in
    /// every [`ContextOptions`]; [`ContextOptions::recorder`] still overrides
    /// this default per context. Contexts created before this call keep the
    /// recorder they were built with.
    #[must_use]
    pub fn with_recorder(mut self, recorder: Option<Arc<ContextProcessorStateRecorder>>) -> Self {
        self.recorder = recorder;
        self
    }

    /// Replaces the default recorder new contexts use.
    ///
    /// Contexts already in the pool are unaffected: their pipeline snapshot was
    /// taken at creation time, matching Python's attachment-order semantics.
    pub fn set_recorder(&mut self, recorder: Option<Arc<ContextProcessorStateRecorder>>) {
        self.recorder = recorder;
    }

    /// Attaches the analyzer new contexts fill the prepared usage block with.
    ///
    /// [`ContextOptions::usage_analyzer`] overrides this default per context.
    #[must_use]
    pub fn with_usage_analyzer(mut self, analyzer: Option<Arc<ContextUsageAnalyzer>>) -> Self {
        self.usage_analyzer = analyzer;
        self
    }

    /// Replaces the default usage analyzer new contexts use.
    pub fn set_usage_analyzer(&mut self, analyzer: Option<Arc<ContextUsageAnalyzer>>) {
        self.usage_analyzer = analyzer;
    }

    /// The configuration new contexts are created with.
    #[must_use]
    pub fn config(&self) -> &ContextConfig {
        &self.config
    }
}

impl ContextEngine {
    /// Builds the payload to send to the model.
    ///
    /// Delegates to [`ConversationContext::prepare`] after honouring the
    /// caller's cancellation, so the deadline and the pipeline are the
    /// context's business while abandonment is checked here as well.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Cancelled`] when [`PrepareOptions::cancellation`] is
    /// already set, plus every error of [`ConversationContext::prepare`].
    pub async fn prepare(
        &self,
        context: &Arc<ConversationContext>,
        options: &PrepareOptions,
    ) -> Result<PreparedContext> {
        if options.cancellation.is_cancelled() {
            return Err(Error::Cancelled);
        }
        context.prepare(options).await
    }
}

#[cfg(test)]
mod tests;
