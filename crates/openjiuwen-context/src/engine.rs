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
use std::collections::HashMap;
use std::sync::{Arc, Mutex as StdMutex, PoisonError};

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::prompt::{BaseMessage, Role};
use openjiuwen_session::Session;
use serde_json::{Map, Value};
use tokio::sync::RwLock;

use crate::config::ContextConfig;
use crate::context::{ContextState, ConversationContext, ModelContext};
use crate::events::{CompressReport, CompressionStatus, ObserveOutcome, RecoveryOutcome};
use crate::offload::{InMemoryOffloadStore, OffloadStore};
use crate::policy::{ContextWindowChange, ResolvedContextWindow};
use crate::processor::{ContextPipeline, PipelineDeps, ProcessorConfig};
use crate::recall::{InMemoryRecallArchiveStore, RecallArchiveStore};
use crate::session_memory::{InMemorySessionMemoryStore, SessionMemoryStore};
use crate::stats::{ContextStats, PreparedContext};
use crate::telemetry::{CompressionStatePhase, ContextProcessorStateRecorder};
use crate::tokens::count_messages;
use crate::usage::{
    request_usage_from_metadata, AnalysisInput, CacheAggregationKey, ContextUsageAnalyzer,
    ContextUsageSnapshot, ContextWindowTokenReport, SessionKVCacheAggregator, SessionKVCacheUsage,
};

pub use crate::engine_options::{ContextOptions, PrepareOptions, DEFAULT_SESSION_ID};

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

/// Python's `context_id.replace(".", "_")`: the pool identity of a context.
fn normalize_context_id(context_id: &str) -> String {
    context_id.replace('.', "_")
}

/// The error a usage request raises when no analyzer was configured.
fn missing_analyzer() -> Error {
    Error::Config(
        "no context usage analyzer is configured; call ContextEngine::with_usage_analyzer or set \
         ContextOptions::usage_analyzer"
            .to_owned(),
    )
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

    /// Creates or returns the context `context_id` of `session`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when the configuration is invalid and the
    /// pipeline's own error (for example a summarisation processor without a
    /// model) when it cannot be built.
    pub async fn create_context(
        &self,
        context_id: impl Into<String>,
        session: Option<Arc<dyn Session>>,
    ) -> Result<Arc<ConversationContext>> {
        self.create_context_with(
            context_id,
            ContextOptions {
                session,
                ..ContextOptions::default()
            },
        )
        .await
    }

    /// Creates or returns a context with explicit options.
    ///
    /// The call is idempotent: when the pool already holds the
    /// `(session, context)` pair, the existing context is returned and
    /// `options` is ignored.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when [`ContextConfig::validate`] fails.
    pub async fn create_context_with(
        &self,
        context_id: impl Into<String>,
        options: ContextOptions,
    ) -> Result<Arc<ConversationContext>> {
        self.config.validate()?;
        let context_id = context_id.into();
        let session_id = options.session.as_ref().map_or_else(
            || DEFAULT_SESSION_ID.to_owned(),
            |session| session.id().to_owned(),
        );
        self.ensure_context(&context_id, &session_id, options).await
    }

    /// Inserts `context_id` into the pool under `session_id`, or returns the
    /// existing entry.
    async fn ensure_context(
        &self,
        context_id: &str,
        session_id: &str,
        options: ContextOptions,
    ) -> Result<Arc<ConversationContext>> {
        let key = (session_id.to_owned(), normalize_context_id(context_id));
        if let Some(existing) = self.contexts.read().await.get(&key) {
            return Ok(Arc::clone(existing));
        }

        let model = options.model;
        let counter = options
            .token_counter
            .unwrap_or_else(|| self.config.token_counter());
        let configs = options
            .processors
            .unwrap_or_else(|| self.config.processors.clone());
        // An explicit per-context store wins over the engine default, which in
        // turn wins over the configuration-derived fallback.
        let recall = options.recall.or_else(|| self.recall.clone()).or_else(|| {
            self.config
                .compression_recall
                .enabled
                .then(|| Arc::new(InMemoryRecallArchiveStore::new()) as Arc<dyn RecallArchiveStore>)
        });
        let needs_session_memory = configs.iter().any(|config| {
            matches!(
                config,
                ProcessorConfig::FullCompact(_) | ProcessorConfig::MessageSummaryOffloader(_)
            )
        });
        let session_memory = options
            .session_memory
            .or_else(|| self.session_memory.clone())
            .or_else(|| {
                needs_session_memory.then(|| {
                    Arc::new(InMemorySessionMemoryStore::new()) as Arc<dyn SessionMemoryStore>
                })
            });
        // A per-context recorder/analyzer wins over the engine default; when
        // both are unset the context stays uninstrumented.
        let recorder = options.recorder.or_else(|| self.recorder.clone());
        let usage_analyzer = options
            .usage_analyzer
            .or_else(|| self.usage_analyzer.clone());
        let pipeline = ContextPipeline::from_configs_with(
            &configs,
            &PipelineDeps::new(Arc::clone(&self.store))
                .with_model(model.clone())
                .with_recall(recall.clone())
                .with_session_memory(session_memory.clone())
                .with_recorder(recorder.clone()),
        )?;
        let context = Arc::new(ConversationContext::new(
            context_id,
            session_id,
            Arc::clone(&self.config),
            counter,
            pipeline,
            Arc::clone(&self.store),
            model,
            options.session,
            recall,
            session_memory,
            recorder,
            usage_analyzer,
        ));
        if !options.history.is_empty() {
            context.set_messages(options.history).await?;
        }

        // A concurrent creation may have won the race while the pipeline was
        // being built; the first entry wins.
        let mut pool = self.contexts.write().await;
        if let Some(existing) = pool.get(&key) {
            return Ok(Arc::clone(existing));
        }
        pool.insert(key, Arc::clone(&context));
        Ok(context)
    }

    /// Returns the context `context_id` of `session_id`, when it exists.
    pub async fn get_context(
        &self,
        session_id: &str,
        context_id: &str,
    ) -> Option<Arc<ConversationContext>> {
        let key = (session_id.to_owned(), normalize_context_id(context_id));
        self.contexts.read().await.get(&key).map(Arc::clone)
    }

    /// Removes and returns the context `context_id` of `session_id`.
    pub async fn remove_context(
        &self,
        session_id: &str,
        context_id: &str,
    ) -> Option<Arc<ConversationContext>> {
        let key = (session_id.to_owned(), normalize_context_id(context_id));
        self.contexts.write().await.remove(&key)
    }

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

    /// Appends one message and applies the rolling message limit.
    ///
    /// Eviction removes the oldest non-system messages, never the message just
    /// appended and never a system message, until the history fits
    /// [`ContextConfig::max_context_messages`].
    ///
    /// When a recorder is configured, the appended message also runs through
    /// the passive processor chain in the `add_messages` phase purely so the
    /// record describes what the processors would do to the incoming batch; the
    /// pass's result is discarded, and a telemetry failure is logged rather
    /// than propagated. Without a recorder the append path is unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when [`ConversationContext::add_message`]
    /// rejects the message.
    pub async fn observe(
        &self,
        context: &Arc<ConversationContext>,
        message: BaseMessage,
    ) -> Result<ObserveOutcome> {
        let telemetry_batch = context.has_recorder().then(|| vec![message.clone()]);
        context.add_message(message).await?;
        let mut messages = context.messages().await;
        let mut evicted = 0;
        if let Some(limit) = context.config().max_context_messages {
            while messages.len() > limit {
                // The newest message (the one just appended) is always last and
                // is never evicted.
                let candidate = messages
                    .iter()
                    .enumerate()
                    .take(messages.len().saturating_sub(1))
                    .find(|(_, candidate)| candidate.role != Role::System)
                    .map(|(index, _)| index);
                let Some(index) = candidate else {
                    break;
                };
                messages.remove(index);
                evicted += 1;
            }
            if evicted > 0 {
                context.set_messages(messages.clone()).await?;
            }
        }
        if let Some(mut batch) = telemetry_batch {
            if let Err(error) = context
                .run_pipeline(&mut batch, false, CompressionStatePhase::AddMessages)
                .await
            {
                tracing::debug!(%error, "context observe telemetry pass failed");
            }
        }
        let counter = context.token_counter();
        Ok(ObserveOutcome {
            added: 1,
            evicted,
            total_messages: messages.len(),
            total_tokens: count_messages(counter.as_ref(), &messages),
        })
    }

    /// Appends the result of the tool call `tool_call_id`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the tool result is empty.
    #[allow(clippy::similar_names)]
    pub async fn observe_tool_result(
        &self,
        context: &Arc<ConversationContext>,
        tool_call_id: &str,
        content: impl Into<String>,
    ) -> Result<ObserveOutcome> {
        self.observe(context, BaseMessage::tool(tool_call_id, content.into()))
            .await
    }

    /// Runs the pipeline over the whole history and persists the result.
    ///
    /// The report says [`CompressionStatus::Busy`] when another pass already
    /// holds the context's compression lock, [`CompressionStatus::Noop`] when no
    /// processor changed anything, and [`CompressionStatus::Compressed`] when
    /// the rewritten history was stored.
    ///
    /// # Errors
    ///
    /// Returns the pipeline's error (for example [`Error::Llm`] when
    /// summarisation fails, or [`Error::Cancelled`] when the context was
    /// cancelled).
    pub async fn compress(&self, context: &Arc<ConversationContext>) -> Result<CompressReport> {
        let counter = context.token_counter();
        let Ok(lock) = context.compression_lock().try_lock() else {
            let messages = context.messages().await;
            let tokens = count_messages(counter.as_ref(), &messages);
            return Ok(CompressReport {
                status: CompressionStatus::Busy,
                events: Vec::new(),
                tokens_before: tokens,
                tokens_after: tokens,
                messages: messages.len(),
            });
        };

        let original = context.messages().await;
        let tokens_before = count_messages(counter.as_ref(), &original);
        let mut working = original.clone();
        let events = context
            .run_pipeline(&mut working, true, CompressionStatePhase::ActiveCompress)
            .await?;
        if events.is_empty() && working == original {
            drop(lock);
            return Ok(CompressReport {
                status: CompressionStatus::Noop,
                events,
                tokens_before,
                tokens_after: tokens_before,
                messages: original.len(),
            });
        }

        context.set_messages(working.clone()).await?;
        let tokens_after = count_messages(counter.as_ref(), &working);
        context
            .record_compression(tokens_before, tokens_after)
            .await;
        drop(lock);
        Ok(CompressReport {
            status: CompressionStatus::Compressed,
            events,
            tokens_before,
            tokens_after,
            messages: working.len(),
        })
    }

    /// Recovers from a provider error that a compression pass may fix.
    ///
    /// Recovery is attempted only when the error describes an input-context
    /// overflow (see [`ContextEngine::is_context_overflow_error`]) and no output
    /// was streamed yet: retrying after a visible prefix would repeat it.
    ///
    /// # Errors
    ///
    /// Returns every error of [`ContextEngine::compress`].
    pub async fn recover_from_model_exception(
        &self,
        context: &Arc<ConversationContext>,
        error: &str,
        streaming: bool,
        chunks_emitted: usize,
    ) -> Result<RecoveryOutcome> {
        if streaming && chunks_emitted > 0 {
            tracing::warn!(
                chunks_emitted,
                "skipping context recovery after streaming output was emitted"
            );
            return Ok(RecoveryOutcome::default());
        }
        if !Self::is_context_overflow_error(error) {
            return Ok(RecoveryOutcome::default());
        }
        let report = self.compress(context).await?;
        Ok(RecoveryOutcome {
            attempted: true,
            recovered: report.is_compressed(),
            report: Some(report),
        })
    }

    /// Whether `error` describes a model input that exceeded the context window.
    ///
    /// Matching is case-insensitive after `_` and `-` are folded to spaces, so
    /// provider codes such as `context_length_exceeded` match too; generic quota,
    /// authentication, and network errors stay unrecognized.
    #[must_use]
    pub fn is_context_overflow_error(error: &str) -> bool {
        /// Provider phrases that always mean an input-context overflow.
        const PHRASES: [&str; 9] = [
            "context length",
            "context window",
            "maximum context",
            "context limit",
            "prompt is too long",
            "prompt too long",
            "input is too long",
            "input too long",
            "range of input length",
        ];

        let message = error.to_lowercase().replace(['_', '-'], " ");
        if PHRASES.iter().any(|phrase| message.contains(phrase)) {
            return true;
        }
        let mentions_input = ["context", "prompt", "input"]
            .iter()
            .any(|term| message.contains(term));
        // An output/completion/response token limit is a generation budget, not
        // an input overflow, and must not trigger compression.
        let mentions_output = ["output", "completion", "response"]
            .iter()
            .any(|term| message.contains(term));
        if message.contains("token limit") && !mentions_output {
            return true;
        }
        if message.contains("too long") && mentions_input {
            return true;
        }
        (message.contains("exceed") || message.contains("over limit")) && mentions_input
    }

    /// The context window the engine resolves for its configuration.
    ///
    /// An explicit [`ContextConfig::max_context_tokens`] wins; otherwise a
    /// configured [`ContextWindowPolicy`](crate::policy::ContextWindowPolicy)
    /// resolves the window for the configured model/provider, and a completely
    /// default policy keeps [`DEFAULT_CONTEXT_TOKENS`](crate::DEFAULT_CONTEXT_TOKENS).
    #[must_use]
    pub fn resolved_window(&self) -> ResolvedContextWindow {
        self.config.resolved_window()
    }

    /// The KV-cache diff of `context`'s most recent LLM-bound window.
    ///
    /// The convenience wrapper over [`ConversationContext::detect_window_change`];
    /// `prepare` keeps the last two windows, so no state is mutated here either.
    #[must_use]
    pub fn detect_window_change(&self, context: &ConversationContext) -> ContextWindowChange {
        context.detect_window_change()
    }

    /// Builds the `ContextWindowTokenReport` of the context's current window.
    ///
    /// The window is the most recently prepared one
    /// ([`ConversationContext::prepare`]), or an empty-system window over the
    /// stored history when the context has not been prepared yet. When
    /// `provider_input_tokens` is supplied the report's window and category
    /// parts carry the reconciled snapshot values, exactly like
    /// [`ContextEngine::usage_snapshot`]; without it the report is the frozen
    /// local measurement.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when neither the engine nor the context was
    /// given a [`ContextUsageAnalyzer`].
    pub async fn usage_report(
        &self,
        context: &ConversationContext,
        provider_input_tokens: Option<u64>,
    ) -> Result<ContextWindowTokenReport> {
        let analyzer = context
            .usage_analyzer()
            .map(Arc::as_ref)
            .ok_or_else(|| missing_analyzer())?;
        let (system, messages, tools, stats) = self.usage_window(context).await;
        let mut input = AnalysisInput::new("", context.session_id());
        input.context_id = Some(context.context_id());
        input.system_messages = Some(&system);
        input.context_messages = &messages;
        input.tools = &tools;
        input.context_stats = Some(&stats);
        input.provider_input_tokens = provider_input_tokens;
        let mut report = context.build_window_report(analyzer, &input);
        if provider_input_tokens.is_some() {
            let snapshot = context.analyze_window(analyzer, &input);
            report.context_window = snapshot.context_window;
            report.parts = snapshot.parts;
            report.measurement = snapshot.measurement;
            report.measurement_cache = snapshot.measurement_cache;
        }
        Ok(report)
    }

    /// Builds the `context.usage` snapshot of the context's current window.
    ///
    /// The snapshot is the emitted [`ContextUsageSnapshot`]; a supplied
    /// `provider_input_tokens` total is authoritative and the local categories
    /// are reconciled to it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when neither the engine nor the context was
    /// given a [`ContextUsageAnalyzer`].
    pub async fn usage_snapshot(
        &self,
        context: &ConversationContext,
        provider_input_tokens: Option<u64>,
    ) -> Result<ContextUsageSnapshot> {
        let analyzer = context
            .usage_analyzer()
            .map(Arc::as_ref)
            .ok_or_else(|| missing_analyzer())?;
        let (system, messages, tools, stats) = self.usage_window(context).await;
        let mut input = AnalysisInput::new("", context.session_id());
        input.context_id = Some(context.context_id());
        input.system_messages = Some(&system);
        input.context_messages = &messages;
        input.tools = &tools;
        input.context_stats = Some(&stats);
        input.provider_input_tokens = provider_input_tokens;
        Ok(context.analyze_window(analyzer, &input))
    }

    /// The window and statistics a usage report or snapshot measures.
    async fn usage_window(
        &self,
        context: &ConversationContext,
    ) -> (Vec<BaseMessage>, Vec<BaseMessage>, Vec<Value>, ContextStats) {
        let (system, messages, tools) = match context.last_window_snapshot() {
            Some(window) => (
                window.system_messages,
                window.context_messages,
                window.tools,
            ),
            None => (Vec::new(), context.messages().await, Vec::new()),
        };
        (system, messages, tools, context.statistics().await)
    }

    /// Records one provider usage report in the engine's KV-cache aggregator.
    ///
    /// `usage` is the provider's `usage_metadata` object parsed by
    /// [`request_usage_from_metadata`]; a call the parser flags as inconsistent
    /// is counted as partial and excluded from the hit/miss totals rather than
    /// recorded as a fake miss. Recording the same `request_id` twice for one
    /// scope is idempotent.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `request_id` is empty or blank: an
    /// empty id would be deduplicated against every other empty id.
    pub fn record_provider_usage(
        &self,
        scope: CacheAggregationKey,
        request_id: &str,
        usage: &Value,
    ) -> Result<SessionKVCacheUsage> {
        if request_id.trim().is_empty() {
            return Err(Error::Validation("request_id must not be empty".to_owned()));
        }
        let parsed = request_usage_from_metadata(usage);
        let mut aggregator = self.kv_cache.lock().unwrap_or_else(PoisonError::into_inner);
        Ok(aggregator.record(request_id, &scope, &parsed))
    }

    /// The session-to-date KV-cache rollup of `scope`, when it has calls.
    #[must_use]
    pub fn kv_cache_usage(&self, scope: &CacheAggregationKey) -> Option<SessionKVCacheUsage> {
        let aggregator = self.kv_cache.lock().unwrap_or_else(PoisonError::into_inner);
        let usage = aggregator.snapshot(scope);
        (usage.calls_total > 0).then_some(usage)
    }

    /// Persists every context of `session` under the session state key `context`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Session`] when a state cannot be serialized or the
    /// session rejects the write.
    pub async fn save_contexts(&self, session: &dyn Session) -> Result<usize> {
        let session_id = session.id();
        let contexts: Vec<Arc<ConversationContext>> = {
            let pool = self.contexts.read().await;
            pool.values()
                .filter(|context| context.session_id() == session_id)
                .map(Arc::clone)
                .collect()
        };
        let mut states = Map::new();
        for context in &contexts {
            let state = context.save_state().await;
            let value = serde_json::to_value(state).map_err(|error| {
                Error::Session(format!(
                    "failed to serialize context '{}': {error}",
                    context.context_id()
                ))
            })?;
            states.insert(context.context_id().to_owned(), value);
        }
        let count = states.len();
        session
            .update_state(serde_json::json!({ (STATE_KEY): Value::Object(states) }))
            .await?;
        Ok(count)
    }

    /// Restores every context stored under the session state key `context`.
    ///
    /// Entries that do not decode are logged and skipped so one corrupt context
    /// cannot block the rest of a session. Restored contexts do not carry a
    /// session handle, because this method only borrows the session.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Session`] when the session state cannot be read, and
    /// [`Error::Config`] when a context cannot be created.
    pub async fn load_contexts(&self, session: &dyn Session) -> Result<usize> {
        let Some(value) = session.get_state(STATE_KEY).await? else {
            return Ok(0);
        };
        let Some(entries) = value.as_object() else {
            tracing::warn!("session context state is not an object; skipping it");
            return Ok(0);
        };
        let entries: Vec<(String, Value)> = entries
            .iter()
            .map(|(context_id, state)| (context_id.clone(), state.clone()))
            .collect();

        self.config.validate()?;
        let mut loaded = 0;
        for (context_id, value) in entries {
            let state: ContextState = match serde_json::from_value(value) {
                Ok(state) => state,
                Err(error) => {
                    tracing::warn!(%context_id, %error, "skipping corrupt context state");
                    continue;
                }
            };
            let context = self
                .ensure_context(&context_id, session.id(), ContextOptions::default())
                .await?;
            context.load_state(state).await?;
            loaded += 1;
        }
        Ok(loaded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openjiuwen_foundation::prompt::ApproxTokenCounter;
    use openjiuwen_session::AgentSession;

    use crate::processor::TruncationConfig;

    fn options() -> ContextOptions {
        ContextOptions {
            token_counter: Some(Arc::new(ApproxTokenCounter::with_chars_per_token(1))),
            ..ContextOptions::default()
        }
    }

    fn truncating_engine() -> ContextEngine {
        ContextEngine::new(ContextConfig {
            max_context_tokens: Some(8),
            processors: vec![ProcessorConfig::Truncation(TruncationConfig {
                max_tokens: Some(8),
                keep_recent_messages: 2,
                keep_system: true,
            })],
            ..ContextConfig::default()
        })
    }

    #[tokio::test]
    async fn create_context_is_idempotent_and_normalizes_dots() {
        let engine = ContextEngine::new(ContextConfig::default());
        let first = engine.create_context("main.thread", None).await.unwrap();
        let second = engine.create_context("main.thread", None).await.unwrap();
        assert!(Arc::ptr_eq(&first, &second));

        let dotted = engine.create_context("main_thread", None).await.unwrap();
        assert!(Arc::ptr_eq(&first, &dotted));
        assert_eq!(first.context_id(), "main.thread");
        assert_eq!(first.session_id(), DEFAULT_SESSION_ID);

        let found = engine.get_context(DEFAULT_SESSION_ID, "main.thread").await;
        assert!(found.is_some());
        let removed = engine
            .remove_context(DEFAULT_SESSION_ID, "main.thread")
            .await;
        assert!(removed.is_some());
        assert!(engine
            .get_context(DEFAULT_SESSION_ID, "main.thread")
            .await
            .is_none());
        assert!(engine
            .remove_context(DEFAULT_SESSION_ID, "missing")
            .await
            .is_none());
    }

    #[tokio::test]
    async fn sessions_isolate_contexts_with_the_same_id() {
        let engine = ContextEngine::new(ContextConfig::default());
        let alice: Arc<dyn Session> = Arc::new(AgentSession::new("alice"));
        let bob: Arc<dyn Session> = Arc::new(AgentSession::new("bob"));
        let alice_context = engine
            .create_context("main", Some(Arc::clone(&alice)))
            .await
            .unwrap();
        let bob_context = engine
            .create_context("main", Some(Arc::clone(&bob)))
            .await
            .unwrap();
        assert!(!Arc::ptr_eq(&alice_context, &bob_context));
        assert_eq!(alice_context.session_id(), "alice");
        assert_eq!(bob_context.session_id(), "bob");
        assert!(alice_context.session().is_some());
    }

    #[tokio::test]
    async fn create_context_with_applies_history_and_options() {
        let engine = ContextEngine::new(ContextConfig::default());
        let context = engine
            .create_context_with(
                "main",
                ContextOptions {
                    history: vec![BaseMessage::user("q"), BaseMessage::assistant("a")],
                    processors: Some(Vec::new()),
                    ..options()
                },
            )
            .await
            .unwrap();
        assert_eq!(context.messages().await.len(), 2);
        assert!(context.processors().is_empty());
        assert_eq!(context.token_counter().count_text("abcd"), 4);

        let invalid = ContextEngine::new(ContextConfig {
            compression_threshold: 0.0,
            ..ContextConfig::default()
        });
        assert!(matches!(
            invalid.create_context("main", None).await.unwrap_err(),
            Error::Config(_)
        ));
    }

    #[tokio::test]
    async fn prepare_delegates_and_honours_cancellation() {
        let engine = ContextEngine::new(ContextConfig::default());
        let context = engine.create_context("main", None).await.unwrap();
        engine
            .observe(&context, BaseMessage::user("hi"))
            .await
            .unwrap();

        let prepared = engine
            .prepare(&context, &PrepareOptions::default())
            .await
            .unwrap();
        assert_eq!(prepared.messages.len(), 1);

        let cancelled = PrepareOptions::default();
        cancelled.cancellation.cancel();
        assert!(matches!(
            engine.prepare(&context, &cancelled).await.unwrap_err(),
            Error::Cancelled
        ));
    }

    #[tokio::test]
    async fn observe_counts_messages_and_tokens() {
        let engine = ContextEngine::new(ContextConfig::default());
        let context = engine.create_context_with("main", options()).await.unwrap();

        let outcome = engine
            .observe(&context, BaseMessage::user("abcd"))
            .await
            .unwrap();
        assert_eq!(outcome.added, 1);
        assert_eq!(outcome.evicted, 0);
        assert_eq!(outcome.total_messages, 1);
        assert_eq!(outcome.total_tokens, 8);

        let outcome = engine
            .observe_tool_result(&context, "call_1", "xy")
            .await
            .unwrap();
        assert_eq!(outcome.added, 1);
        assert_eq!(outcome.total_messages, 2);
        assert_eq!(context.messages().await[1].role, Role::Tool);
        assert_eq!(
            context.messages().await[1].tool_call_id.as_deref(),
            Some("call_1")
        );
    }

    #[tokio::test]
    async fn observe_evicts_oldest_non_system_messages_but_keeps_the_newest() {
        let engine = ContextEngine::new(ContextConfig {
            max_context_messages: Some(3),
            ..ContextConfig::default()
        });
        let context = engine
            .create_context_with(
                "main",
                ContextOptions {
                    history: vec![
                        BaseMessage::system("persona"),
                        BaseMessage::user("q1"),
                        BaseMessage::assistant("a1"),
                    ],
                    ..options()
                },
            )
            .await
            .unwrap();

        let outcome = engine
            .observe(&context, BaseMessage::user("q2"))
            .await
            .unwrap();
        assert_eq!(outcome.evicted, 1);
        assert_eq!(outcome.total_messages, 3);
        let messages = context.messages().await;
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].role, Role::System);
        assert_eq!(messages[1].text(), "a1");
        assert_eq!(messages[2].text(), "q2");

        // A history made only of system messages plus the new one cannot shrink
        // below the limit without dropping what must stay.
        let engine = ContextEngine::new(ContextConfig {
            max_context_messages: Some(1),
            ..ContextConfig::default()
        });
        let context = engine
            .create_context_with(
                "main",
                ContextOptions {
                    history: vec![BaseMessage::system("a"), BaseMessage::system("b")],
                    ..options()
                },
            )
            .await
            .unwrap();
        let outcome = engine
            .observe(&context, BaseMessage::user("q"))
            .await
            .unwrap();
        assert_eq!(outcome.evicted, 0);
        assert_eq!(outcome.total_messages, 3);
    }

    #[tokio::test]
    async fn compress_reports_compressed_then_noop_and_records_stats() {
        let engine = truncating_engine();
        let context = engine.create_context_with("main", options()).await.unwrap();
        context
            .set_messages(vec![
                BaseMessage::user("first question"),
                BaseMessage::assistant("first answer"),
                BaseMessage::user("second question"),
                BaseMessage::assistant("second answer"),
            ])
            .await
            .unwrap();

        let report = engine.compress(&context).await.unwrap();
        assert_eq!(report.status, CompressionStatus::Compressed);
        assert_eq!(report.events.len(), 1);
        assert!(report.tokens_after < report.tokens_before);
        assert_eq!(report.messages, context.messages().await.len());

        let stats = context.statistics().await;
        assert_eq!(stats.compressions, 1);
        assert_eq!(stats.tokens_before_compression, report.tokens_before);
        assert_eq!(stats.tokens_after_compression, report.tokens_after);

        let report = engine.compress(&context).await.unwrap();
        assert_eq!(report.status, CompressionStatus::Noop);
        assert_eq!(report.tokens_before, report.tokens_after);
        assert_eq!(context.statistics().await.compressions, 1);
    }

    #[tokio::test]
    async fn compress_holds_the_lock_so_a_second_pass_reports_busy() {
        let engine = ContextEngine::new(ContextConfig::default());
        let context = engine.create_context("main", None).await.unwrap();
        // Hold the lock the engine's `compress` would take.
        let guard = context.compression_lock().try_lock();
        assert!(guard.is_ok());
        let report = engine.compress(&context).await.unwrap();
        assert_eq!(report.status, CompressionStatus::Busy);
        assert_eq!(report.messages, 0);
        drop(guard);

        // With the lock released the pass runs again.
        let report = engine.compress(&context).await.unwrap();
        assert_eq!(report.status, CompressionStatus::Noop);
    }

    #[tokio::test]
    async fn recover_refuses_when_chunks_were_streamed_or_the_error_is_unrelated() {
        let engine = truncating_engine();
        let context = engine.create_context_with("main", options()).await.unwrap();
        context
            .set_messages(vec![
                BaseMessage::user("first question"),
                BaseMessage::assistant("first answer"),
                BaseMessage::user("second question"),
                BaseMessage::assistant("second answer"),
            ])
            .await
            .unwrap();

        let outcome = engine
            .recover_from_model_exception(&context, "rate limit exceeded", false, 0)
            .await
            .unwrap();
        assert!(!outcome.attempted);
        assert!(!outcome.recovered);
        assert!(outcome.report.is_none());
        assert_eq!(context.messages().await.len(), 4);

        let outcome = engine
            .recover_from_model_exception(&context, "context length exceeded", true, 2)
            .await
            .unwrap();
        assert!(!outcome.attempted);
        assert!(outcome.report.is_none());
        assert_eq!(context.messages().await.len(), 4);

        let outcome = engine
            .recover_from_model_exception(&context, "context length exceeded", false, 0)
            .await
            .unwrap();
        assert!(outcome.attempted);
        assert!(outcome.recovered);
        let report = outcome.report.unwrap();
        assert_eq!(report.status, CompressionStatus::Compressed);
        assert!(context.messages().await.len() < 4);
    }

    #[tokio::test]
    async fn recover_reports_an_ineffective_compression_as_not_recovered() {
        let engine = ContextEngine::new(ContextConfig::default());
        let context = engine.create_context("main", None).await.unwrap();
        engine
            .observe(&context, BaseMessage::user("hi"))
            .await
            .unwrap();

        let outcome = engine
            .recover_from_model_exception(&context, "maximum context length is 8192", false, 0)
            .await
            .unwrap();
        assert!(outcome.attempted);
        assert!(!outcome.recovered);
        assert_eq!(outcome.report.unwrap().status, CompressionStatus::Noop);
    }

    #[tokio::test]
    async fn contexts_round_trip_through_session_state() {
        let engine = ContextEngine::new(ContextConfig::default());
        let session = AgentSession::new("s");
        let handle: Arc<dyn Session> = Arc::new(AgentSession::new("s"));
        let context = engine
            .create_context("main.thread", Some(handle))
            .await
            .unwrap();
        engine
            .observe(&context, BaseMessage::user("hi"))
            .await
            .unwrap();
        engine
            .observe_tool_result(&context, "call_1", "42")
            .await
            .unwrap();

        assert_eq!(engine.save_contexts(&session).await.unwrap(), 1);
        let stored = session.get_state(STATE_KEY).await.unwrap().unwrap();
        assert!(stored.get("main.thread").is_some());

        let restored_engine = ContextEngine::new(ContextConfig::default());
        assert_eq!(restored_engine.load_contexts(&session).await.unwrap(), 1);
        let restored = restored_engine
            .get_context("s", "main.thread")
            .await
            .unwrap();
        assert_eq!(restored.messages().await, context.messages().await);
        assert_eq!(restored.revision(), context.revision());

        // A session with no contexts saves an empty map and restores nothing.
        let other = AgentSession::new("other");
        assert_eq!(engine.save_contexts(&other).await.unwrap(), 0);
        assert_eq!(restored_engine.load_contexts(&other).await.unwrap(), 0);
    }

    #[tokio::test]
    async fn load_contexts_skips_corrupt_entries_and_missing_state() {
        let engine = ContextEngine::new(ContextConfig::default());
        let session = AgentSession::new("s");
        session
            .update_state(serde_json::json!({
                "context": {
                    "good": {"messages": [], "revision": 3},
                    "bad": {"messages": "not a list"}
                }
            }))
            .await
            .unwrap();

        assert_eq!(engine.load_contexts(&session).await.unwrap(), 1);
        let restored = engine.get_context("s", "good").await.unwrap();
        assert_eq!(restored.revision(), 3);

        let empty = AgentSession::new("empty");
        assert_eq!(engine.load_contexts(&empty).await.unwrap(), 0);

        let not_an_object = AgentSession::new("broken");
        not_an_object
            .update_state(serde_json::json!({"context": 7}))
            .await
            .unwrap();
        assert_eq!(engine.load_contexts(&not_an_object).await.unwrap(), 0);
    }

    #[test]
    fn overflow_phrases_are_recognized_and_unrelated_errors_are_not() {
        let overflow = [
            "This model's maximum context length is 8192 tokens",
            "the request exceeded the CONTEXT WINDOW",
            "maximum context reached",
            "context limit reached",
            "prompt is too long: 210000 tokens",
            "prompt too long",
            "input is too long for the model",
            "input too long",
            "Range of input length should be [1, 1000000]",
            "token limit reached",
            "context_length_exceeded",
            "the prompt exceeded the limit",
        ];
        for phrase in overflow {
            assert!(ContextEngine::is_context_overflow_error(phrase), "{phrase}");
        }

        let unrelated = [
            "",
            "invalid api key",
            "rate limit exceeded",
            "connection reset by peer",
            "output token limit reached",
            "completion token limit reached",
            "response token limit reached",
            "too long",
        ];
        for phrase in unrelated {
            assert!(
                !ContextEngine::is_context_overflow_error(phrase),
                "{phrase}"
            );
        }
    }

    #[test]
    fn engine_accessors_and_option_defaults() {
        let engine = ContextEngine::with_store(
            ContextConfig::default(),
            Arc::new(InMemoryOffloadStore::new()),
        );
        assert_eq!(engine.config().compression_threshold, 0.85);
        assert!(format!("{engine:?}").contains("ContextEngine"));
        assert!(format!("{:?}", options()).contains("ContextOptions"));

        let options = PrepareOptions::default();
        assert_eq!(options.system_messages.len(), 0);
        assert_eq!(options.tools.len(), 0);
        assert!(options.timeout.is_none());
        assert!(!options.cancellation.is_cancelled());
        assert_eq!(options.window_size, None);
    }
}
