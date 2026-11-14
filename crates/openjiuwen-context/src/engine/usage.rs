//! Usage measurement, KV-cache rollups, and the read-only window accessors.
//!
//! Everything here reports on the most recently prepared window — or on an
//! empty-system window over the stored history before the first preparation —
//! and none of it mutates the stored history or the window tracking.
use std::sync::{Arc, PoisonError};

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::prompt::BaseMessage;
use serde_json::Value;

use crate::context::{ConversationContext, ModelContext};
use crate::policy::{ContextWindowChange, ResolvedContextWindow};
use crate::stats::ContextStats;
use crate::usage::{
    request_usage_from_metadata, AnalysisInput, CacheAggregationKey, ContextUsageSnapshot,
    ContextWindowTokenReport, SessionKVCacheUsage,
};

use super::ContextEngine;

/// The error a usage request raises when no analyzer was configured.
fn missing_analyzer() -> Error {
    Error::Config(
        "no context usage analyzer is configured; call ContextEngine::with_usage_analyzer or set \
         ContextOptions::usage_analyzer"
            .to_owned(),
    )
}

impl ContextEngine {
    /// The context window the engine resolves for its configuration.
    ///
    /// An explicit
    /// [`ContextConfig::max_context_tokens`](crate::config::ContextConfig::max_context_tokens)
    /// wins; otherwise a configured
    /// [`ContextWindowPolicy`](crate::policy::ContextWindowPolicy)
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
    /// given a [`ContextUsageAnalyzer`](crate::usage::ContextUsageAnalyzer).
    pub async fn usage_report(
        &self,
        context: &ConversationContext,
        provider_input_tokens: Option<u64>,
    ) -> Result<ContextWindowTokenReport> {
        let analyzer = context
            .usage_analyzer()
            .map(Arc::as_ref)
            .ok_or_else(missing_analyzer)?;
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
    /// given a [`ContextUsageAnalyzer`](crate::usage::ContextUsageAnalyzer).
    pub async fn usage_snapshot(
        &self,
        context: &ConversationContext,
        provider_input_tokens: Option<u64>,
    ) -> Result<ContextUsageSnapshot> {
        let analyzer = context
            .usage_analyzer()
            .map(Arc::as_ref)
            .ok_or_else(missing_analyzer)?;
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
}
