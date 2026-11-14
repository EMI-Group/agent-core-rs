//! Usage telemetry seam between the `ReAct` loop and `openjiuwen-context`.
//!
//! `openjiuwen-context` owns every measurement: the
//! [`ContextUsageAnalyzer`] that explains a prepared window, the
//! [`ContextProcessorStateRecorder`](openjiuwen_context::ContextProcessorStateRecorder)
//! that records compression states, the `context.usage` snapshot shape, and the
//! session KV-cache aggregator. This module only connects the agent to that
//! surface:
//!
//! - [`default_usage_analyzer`] is the analyzer a host gets without configuring
//!   anything — foundation's approximate counter with its own identity, no model
//!   and no window limit (the engine substitutes the window it resolves);
//! - [`cache_scope_key`], [`request_id`] and [`provider_usage_value`] build the
//!   identity and payload of one provider usage report, which the loop feeds to
//!   `ContextEngine::record_provider_usage` for the KV-cache rollup;
//! - [`UsageTelemetry`] buffers the post-call snapshots the loop produced, which
//!   is how an `invoke` caller (no publisher, so no frames) reads the same
//!   telemetry a `stream` caller receives as `context.usage` frames.
//!
//! Nothing here measures tokens itself, talks to a provider, or decides
//! compression: the seam is deliberately a pass-through so the agent and the
//! context engine cannot disagree about token accounting.
//!
//! [`ContextUsageAnalyzer`]: openjiuwen_context::ContextUsageAnalyzer

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use openjiuwen_context::{
    CacheAggregationKey, ContextUsageAnalyzer, ContextUsageSnapshot, CounterIdentity,
};
use openjiuwen_foundation::prompt::ApproxTokenCounter;
use openjiuwen_foundation::{ModelRequestConfig, Usage};
use parking_lot::RwLock;
use serde_json::Value;

/// The context id the `ReAct` loop creates its conversation window under.
///
/// One run owns exactly one context; a host reading telemetry after an
/// invocation (`ReActAgent::usage_report`) must use the same id.
pub const MAIN_CONTEXT_ID: &str = "main";

/// The number of undrained snapshots one [`UsageTelemetry`] keeps.
///
/// The buffer is a convenience for callers that poll instead of streaming, so
/// it is capped: the oldest snapshot is dropped once the cap is reached, and
/// the dropped events stay counted in [`UsageTelemetry::recorded_total`].
pub const MAX_TELEMETRY_EVENTS: usize = 128;

/// Builds the analyzer a host gets without configuring anything.
///
/// This is [`ContextUsageAnalyzer::default`]: foundation's
/// [`ApproxTokenCounter`] with [`CounterIdentity::approx`], no model name, and
/// no window limit, so the engine substitutes the window it resolves for the
/// context. Install anything more specific (a model-aware counter, an explicit
/// window) with `ReActAgent::set_usage_analyzer` instead.
#[must_use]
pub fn default_usage_analyzer() -> ContextUsageAnalyzer {
    ContextUsageAnalyzer::new(
        Arc::new(ApproxTokenCounter::new()),
        CounterIdentity::approx(),
    )
}

/// Builds the KV-cache aggregation scope of one run.
///
/// The scope is `(session, provider, model)`: the session id comes from the run,
/// the provider from the model that served the call, and the model name from the
/// request the call was made with. Recording the same `(scope, request_id)` pair
/// twice is idempotent in the engine's aggregator, so a retried model call
/// updates the entry instead of double counting it.
#[must_use]
pub fn cache_scope_key(
    session_id: &str,
    provider: &str,
    request: &ModelRequestConfig,
) -> CacheAggregationKey {
    CacheAggregationKey::new(session_id, provider, request.model_name.clone())
}

/// The request id one model call is recorded under.
///
/// The id is deterministic — `"<session id>#<iteration>"` — so a resumed turn
/// never records a call twice and a test can name the entry it expects; the
/// engine rejects only blank ids.
#[must_use]
pub fn request_id(session_id: &str, iteration: usize) -> String {
    format!("{session_id}#{iteration}")
}

/// Serializes a provider usage report for
/// `ContextEngine::record_provider_usage`.
///
/// The engine's parser reads the provider's `usage_metadata` object, which is
/// exactly the wire shape of [`Usage`]; an unserializable usage (impossible for
/// this type, whose `extra` map is JSON) degrades to `null`, which the parser
/// treats as "not reported".
#[must_use]
pub fn provider_usage_value(usage: &Usage) -> Value {
    serde_json::to_value(usage).unwrap_or(Value::Null)
}

/// A cloneable buffer of the post-call `context.usage` snapshots of an agent.
///
/// Every clone shares one buffer, so a host can take a handle with
/// `ReActAgent::usage_telemetry` and keep draining while invocations run. The
/// loop pushes one snapshot per successful model call when an analyzer is
/// installed; without an analyzer it pushes nothing, because the engine cannot
/// measure a window.
///
/// A buffered snapshot is stamped by the loop, not by the engine: its `phase` is
/// `post_call` and its `request_id` is the [`request_id`] of the call it belongs
/// to, so a drained snapshot is self-describing.
#[derive(Debug, Clone, Default)]
pub struct UsageTelemetry {
    inner: Arc<TelemetryInner>,
}

/// The shared state behind [`UsageTelemetry`].
#[derive(Debug, Default)]
struct TelemetryInner {
    /// Snapshots not drained yet, oldest first, capped at
    /// [`MAX_TELEMETRY_EVENTS`].
    events: RwLock<VecDeque<ContextUsageSnapshot>>,
    /// Every snapshot ever recorded, including the ones the cap dropped.
    recorded: AtomicU64,
}

impl UsageTelemetry {
    /// Creates an empty buffer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends `snapshot`, dropping the oldest one once the buffer is full.
    pub fn record(&self, snapshot: ContextUsageSnapshot) {
        {
            let mut events = self.inner.events.write();
            while events.len() >= MAX_TELEMETRY_EVENTS {
                events.pop_front();
            }
            events.push_back(snapshot);
        }
        self.inner.recorded.fetch_add(1, Ordering::Relaxed);
    }

    /// Removes and returns every buffered snapshot, oldest first.
    #[must_use]
    pub fn drain(&self) -> Vec<ContextUsageSnapshot> {
        self.inner.events.write().drain(..).collect()
    }

    /// Returns the most recently recorded snapshot, without draining it.
    #[must_use]
    pub fn last(&self) -> Option<ContextUsageSnapshot> {
        self.inner.events.read().back().cloned()
    }

    /// Returns the number of buffered snapshots.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.events.read().len()
    }

    /// Returns whether no snapshot is buffered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.events.read().is_empty()
    }

    /// Returns how many snapshots were ever recorded, drained or not.
    #[must_use]
    pub fn recorded_total(&self) -> u64 {
        self.inner.recorded.load(Ordering::Relaxed)
    }

    /// Drops every buffered snapshot without returning it.
    ///
    /// The recorded total keeps counting, so `recorded_total` stays a
    /// monotonic lifetime counter.
    pub fn clear(&self) {
        self.inner.events.write().clear();
    }
}

#[cfg(test)]
mod tests;
