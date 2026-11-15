//! The `context.compression_state` wire contract.
//!
//! One [`ContextCompressionState`] describes a single context-processor pass:
//! which operation ran, in which phase, what the history looked like before and
//! after, what it saved, which messages a processor rewrote in place, and how
//! many model tokens the pass spent. It is the payload of the session stream
//! frame typed [`CONTEXT_COMPRESSION_STATE_TYPE`], so the field names, the
//! `type` tag and the status/phase spellings follow Python's pydantic models
//! exactly.
//!
//! Every struct is `serde`-serializable with `snake_case` field names, accepts
//! unknown fields, and defaults every field, so a record persisted (or streamed)
//! by an older writer still loads. `None` is serialized as JSON `null` instead of
//! being skipped, matching `model_dump(mode="json")`.
use std::collections::BTreeMap;

use openjiuwen_foundation::prompt::Usage;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::stats::ContextStats;

/// Frame type and `type` tag of a compression-state record.
pub const CONTEXT_COMPRESSION_STATE_TYPE: &str = "context.compression_state";

/// Lifecycle of one processor pass.
///
/// This is the record's own status vocabulary; [`crate::events::CompressionStatus`]
/// is the coarser outcome of `ContextEngine::compress` and stays separate.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompressionStateStatus {
    /// The pass began.
    #[default]
    Started,
    /// The pass rewrote the history.
    Completed,
    /// The pass ran and changed nothing.
    Noop,
    /// The pass was not attempted, or had nothing to work on.
    Skipped,
    /// The pass raised an error.
    Failed,
    /// The pass stopped because the run is pausing.
    Paused,
    /// The pass stopped because the run was cancelled.
    Cancelled,
}

impl CompressionStateStatus {
    /// The wire spelling (`started`, `completed`, `noop`, …).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Started => "started",
            Self::Completed => "completed",
            Self::Noop => "noop",
            Self::Skipped => "skipped",
            Self::Failed => "failed",
            Self::Paused => "paused",
            Self::Cancelled => "cancelled",
        }
    }

    /// Whether the pass is over; only [`CompressionStateStatus::Started`] is open.
    #[must_use]
    pub const fn is_terminal(self) -> bool {
        !matches!(self, Self::Started)
    }

    /// Whether the pass finished without an error.
    #[must_use]
    pub const fn is_success(self) -> bool {
        matches!(self, Self::Completed | Self::Noop | Self::Skipped)
    }
}

/// The context-engine entry point a record was produced by.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompressionStatePhase {
    /// `add_messages` — the passive processor chain on the incoming batch.
    #[default]
    AddMessages,
    /// `get_context_window` — window assembly and its passive processors.
    GetContextWindow,
    /// `active_compress` — an explicit compression pass.
    ActiveCompress,
}

impl CompressionStatePhase {
    /// The wire spelling (`add_messages`, `get_context_window`, `active_compress`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::AddMessages => "add_messages",
            Self::GetContextWindow => "get_context_window",
            Self::ActiveCompress => "active_compress",
        }
    }
}

/// Size of the history at one instant of a pass.
///
/// `estimated`/`tokens_source` are **additive** to Python's pydantic model: a
/// record written by Python still loads here (both fields default), and a record
/// written here stays valid JSON for any tolerant reader, because a tolerant
/// reader ignores keys it does not know.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ContextCompressionMetric {
    /// ISO-8601 UTC timestamp of the observation, when one was recorded.
    pub time: Option<String>,
    /// Number of messages in the history.
    pub messages: usize,
    /// Tokens in the history.
    pub tokens: usize,
    /// `tokens` as a rounded percentage of the context window, clamped to 0–100.
    pub context_percent: Option<u32>,
    /// Whether [`ContextCompressionMetric::tokens`] is an estimate.
    ///
    /// `true` says the number was produced by an approximating strategy — the
    /// `ceil(characters / 3)` string-length fallback a recorder without a token
    /// counter uses, an approximate or `tiktoken`-family counter, or a counter
    /// that cannot describe itself — so it must never be read as exact.
    /// Python's record carries no such flag, so a legacy record loads as `false`.
    #[serde(default)]
    pub estimated: bool,
    /// How [`ContextCompressionMetric::tokens`] was produced, when it can be
    /// named: a tokenizer source such as `native_tokenizer`, `tiktoken` or
    /// `string_length_fallback`, the recorder's `string_length_estimate`
    /// fallback, or a counter name such as `approx_token_counter`.
    ///
    /// Additive and absent rather than `null` when nothing can be named, which
    /// is the one place this record departs from Python's `null`-preserving
    /// dump: an unknown source has no spelling.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens_source: Option<String>,
}

/// What a pass saved, computed as `before - after`.
///
/// Both deltas are signed: a pass that grows the history reports negative
/// savings rather than saturating at zero.
/// `estimated` is an **additive** field: it is `true` when either side of the
/// subtraction is an estimate, so a saving computed from estimates is never
/// published as an exact one. A record written by Python (which has no such
/// field) still loads, and a tolerant reader ignores the extra key.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ContextCompressionSaved {
    /// Messages removed by the pass.
    pub messages: i64,
    /// Tokens removed by the pass.
    pub tokens: i64,
    /// [`ContextCompressionSaved::tokens`] as a percentage of the before size.
    pub percent: f64,
    /// Whether [`ContextCompressionSaved::tokens`] inherits an estimate from
    /// either side; an estimate is not an exact saving.
    #[serde(default)]
    pub estimated: bool,
}

/// The model tokens one or more processor calls consumed.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ContextCompressionUsage {
    /// Number of model calls the pass made.
    pub calls: usize,
    /// Prompt tokens.
    pub input_tokens: u64,
    /// Completion tokens.
    pub output_tokens: u64,
    /// Provider-reported total, or the input/output sum.
    pub total_tokens: u64,
    /// Prompt tokens served from the provider's cache.
    pub cache_tokens: u64,
    /// Cost of the prompt tokens, in the provider's currency.
    pub input_cost: f64,
    /// Cost of the completion tokens.
    pub output_cost: f64,
    /// [`ContextCompressionUsage::input_cost`] plus the output cost.
    pub total_cost: f64,
    /// Model the calls were made against.
    pub model_name: String,
    /// Free-form per-call records.
    pub details: Vec<BTreeMap<String, Value>>,
}

impl ContextCompressionUsage {
    /// Builds a usage for one call against `model_name`.
    #[must_use]
    pub fn from_usage(usage: &Usage, model_name: impl Into<String>) -> Self {
        let mut value = Self {
            model_name: model_name.into(),
            ..Self::default()
        };
        value.record(usage);
        value
    }

    /// Counts one more call and adds its counters.
    pub fn record(&mut self, usage: &Usage) {
        self.calls = self.calls.saturating_add(1);
        self.input_tokens = self.input_tokens.saturating_add(usage.input_tokens);
        self.output_tokens = self.output_tokens.saturating_add(usage.output_tokens);
        self.total_tokens = self.total_tokens.saturating_add(usage.total_tokens);
        self.cache_tokens = self.cache_tokens.saturating_add(usage.cache_tokens);
    }

    /// Merges `other` into this usage; `details` are appended.
    pub fn add(&mut self, other: &Self) {
        self.calls = self.calls.saturating_add(other.calls);
        self.input_tokens = self.input_tokens.saturating_add(other.input_tokens);
        self.output_tokens = self.output_tokens.saturating_add(other.output_tokens);
        self.total_tokens = self.total_tokens.saturating_add(other.total_tokens);
        self.cache_tokens = self.cache_tokens.saturating_add(other.cache_tokens);
        self.input_cost += other.input_cost;
        self.output_cost += other.output_cost;
        self.total_cost += other.total_cost;
        if self.model_name.is_empty() {
            self.model_name.clone_from(&other.model_name);
        }
        self.details.extend(other.details.iter().cloned());
    }

    /// Sets the prompt and completion costs; the total is their sum.
    #[must_use]
    pub fn with_costs(mut self, input_cost: f64, output_cost: f64) -> Self {
        self.input_cost = input_cost;
        self.output_cost = output_cost;
        self.total_cost = input_cost + output_cost;
        self
    }

    /// Overrides [`ContextCompressionUsage::model_name`].
    #[must_use]
    pub fn with_model_name(mut self, model_name: impl Into<String>) -> Self {
        self.model_name = model_name.into();
        self
    }

    /// Appends one free-form per-call record.
    #[must_use]
    pub fn with_detail(mut self, detail: BTreeMap<String, Value>) -> Self {
        self.details.push(detail);
        self
    }
}

/// A message a processor rewrote in place, keeping its identity.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ContextCompressionModifiedMessage {
    /// The message's `context_message_id`, the id trajectory windows use for it.
    pub message_id: String,
    /// The message's role after the rewrite.
    pub role: String,
    /// The call id of a rewritten tool result.
    pub tool_call_id: Option<String>,
    /// Handle that restores the original content; `None` when nothing was offloaded.
    pub offload_handle: Option<String>,
    /// Offload kind (`in_memory` / `filesystem`); `None` when nothing was offloaded.
    pub offload_type: Option<String>,
}

/// One processor pass, as recorded and streamed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ContextCompressionState {
    /// Record tag; always [`CONTEXT_COMPRESSION_STATE_TYPE`] for records this
    /// crate writes, and preserved verbatim when reading.
    #[serde(rename = "type", default = "state_type_default")]
    pub state_type: String,
    /// Id correlating the pass with the model calls it made.
    pub operation_id: String,
    /// Where in the lifecycle the pass is.
    pub status: CompressionStateStatus,
    /// Which context-engine entry point produced the record.
    pub phase: CompressionStatePhase,
    /// Name of the processor that ran.
    pub processor: String,
    /// Model the processor was configured with, when it has one.
    pub model: String,
    /// Size of the history the processor received.
    pub before: ContextCompressionMetric,
    /// Size of the history the processor produced; `None` until the pass ends.
    pub after: Option<ContextCompressionMetric>,
    /// Per-role counts of the history the record describes.
    pub statistic: ContextStats,
    /// What the pass saved; `None` until the pass produces a history.
    pub saved: Option<ContextCompressionSaved>,
    /// Messages present before and after whose content or offload state changed.
    pub modified_messages: Vec<ContextCompressionModifiedMessage>,
    /// Model tokens the pass consumed, when it called a model.
    pub compression_usage: Option<ContextCompressionUsage>,
    /// Wall-clock duration of the pass, in milliseconds.
    pub duration_ms: Option<u64>,
    /// The context window size the percentages were measured against.
    pub context_max: Option<u32>,
    /// Human-readable one-line summary of what happened.
    pub summary: String,
    /// Short summary a processor produced, when it produced one.
    pub compact_summary: String,
    /// Error text of a failed pass.
    pub error: Option<String>,
    /// Why this state was reached, for example `user_pause` or `user_cancel`.
    pub reason: String,
}

fn state_type_default() -> String {
    CONTEXT_COMPRESSION_STATE_TYPE.to_owned()
}

impl Default for ContextCompressionState {
    fn default() -> Self {
        Self {
            state_type: state_type_default(),
            operation_id: String::new(),
            status: CompressionStateStatus::default(),
            phase: CompressionStatePhase::default(),
            processor: String::new(),
            model: String::new(),
            before: ContextCompressionMetric::default(),
            after: None,
            statistic: ContextStats::default(),
            saved: None,
            modified_messages: Vec::new(),
            compression_usage: None,
            duration_ms: None,
            context_max: None,
            summary: String::new(),
            compact_summary: String::new(),
            error: None,
            reason: String::new(),
        }
    }
}

impl ContextCompressionState {
    /// Creates a record for `operation_id` in `status`/`phase`.
    #[must_use]
    pub fn new(
        operation_id: impl Into<String>,
        status: CompressionStateStatus,
        phase: CompressionStatePhase,
    ) -> Self {
        Self {
            operation_id: operation_id.into(),
            status,
            phase,
            ..Self::default()
        }
    }

    /// Creates a [`CompressionStateStatus::Started`] record.
    #[must_use]
    pub fn started(operation_id: impl Into<String>, phase: CompressionStatePhase) -> Self {
        Self::new(operation_id, CompressionStateStatus::Started, phase)
    }

    /// Creates a [`CompressionStateStatus::Completed`] record.
    #[must_use]
    pub fn completed(operation_id: impl Into<String>, phase: CompressionStatePhase) -> Self {
        Self::new(operation_id, CompressionStateStatus::Completed, phase)
    }

    /// Creates a [`CompressionStateStatus::Noop`] record.
    #[must_use]
    pub fn noop(operation_id: impl Into<String>, phase: CompressionStatePhase) -> Self {
        Self::new(operation_id, CompressionStateStatus::Noop, phase)
    }

    /// Creates a [`CompressionStateStatus::Failed`] record carrying `error`.
    #[must_use]
    pub fn failed(
        operation_id: impl Into<String>,
        phase: CompressionStatePhase,
        error: impl Into<String>,
    ) -> Self {
        let mut state = Self::new(operation_id, CompressionStateStatus::Failed, phase);
        state.error = Some(error.into());
        state
    }

    /// Whether the lifecycle is over; see [`CompressionStateStatus::is_terminal`].
    #[must_use]
    pub const fn is_terminal(&self) -> bool {
        self.status.is_terminal()
    }

    /// Overrides [`ContextCompressionState::processor`].
    #[must_use]
    pub fn with_processor(mut self, processor: impl Into<String>) -> Self {
        self.processor = processor.into();
        self
    }

    /// Overrides [`ContextCompressionState::model`].
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Sets the before metric.
    #[must_use]
    pub fn with_before(mut self, before: ContextCompressionMetric) -> Self {
        self.before = before;
        self
    }

    /// Sets the after metric.
    #[must_use]
    pub fn with_after(mut self, after: ContextCompressionMetric) -> Self {
        self.after = Some(after);
        self
    }

    /// Sets the per-role statistic.
    #[must_use]
    pub fn with_statistic(mut self, statistic: ContextStats) -> Self {
        self.statistic = statistic;
        self
    }

    /// Sets the saved delta.
    #[must_use]
    pub fn with_saved(mut self, saved: ContextCompressionSaved) -> Self {
        self.saved = Some(saved);
        self
    }

    /// Replaces the rewritten-message reports.
    #[must_use]
    pub fn with_modified_messages(
        mut self,
        modified_messages: Vec<ContextCompressionModifiedMessage>,
    ) -> Self {
        self.modified_messages = modified_messages;
        self
    }

    /// Sets the model usage of the pass.
    #[must_use]
    pub fn with_compression_usage(mut self, usage: ContextCompressionUsage) -> Self {
        self.compression_usage = Some(usage);
        self
    }

    /// Sets the pass duration.
    #[must_use]
    pub fn with_duration_ms(mut self, duration_ms: u64) -> Self {
        self.duration_ms = Some(duration_ms);
        self
    }

    /// Sets the context window the percentages refer to.
    #[must_use]
    pub fn with_context_max(mut self, context_max: u32) -> Self {
        self.context_max = Some(context_max);
        self
    }

    /// Overrides the generated summary.
    #[must_use]
    pub fn with_summary(mut self, summary: impl Into<String>) -> Self {
        self.summary = summary.into();
        self
    }

    /// Overrides the reason.
    #[must_use]
    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = reason.into();
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn statuses_and_phases_use_the_python_spellings() {
        assert_eq!(CompressionStateStatus::Started.as_str(), "started");
        assert_eq!(
            serde_json::to_string(&CompressionStateStatus::Cancelled).unwrap(),
            "\"cancelled\""
        );
        assert_eq!(
            serde_json::to_string(&CompressionStatePhase::ActiveCompress).unwrap(),
            "\"active_compress\""
        );
        assert!(!CompressionStateStatus::Started.is_terminal());
        assert!(CompressionStateStatus::Paused.is_terminal());
        assert!(CompressionStateStatus::Noop.is_success());
        assert!(!CompressionStateStatus::Failed.is_success());
    }

    #[test]
    fn a_state_round_trips_with_its_type_tag() {
        let state =
            ContextCompressionState::completed("op-1", CompressionStatePhase::ActiveCompress)
                .with_processor("dialogue");
        let value = serde_json::to_value(&state).unwrap();
        assert_eq!(value["type"], CONTEXT_COMPRESSION_STATE_TYPE);
        assert_eq!(value["status"], "completed");
        assert_eq!(value["phase"], "active_compress");
        assert_eq!(
            serde_json::from_value::<ContextCompressionState>(value).unwrap(),
            state
        );

        let empty: ContextCompressionState = serde_json::from_str("{}").unwrap();
        assert_eq!(empty, ContextCompressionState::default());
        assert_eq!(empty.state_type, CONTEXT_COMPRESSION_STATE_TYPE);
    }

    #[test]
    fn usage_records_calls_and_costs() {
        let mut usage = ContextCompressionUsage::from_usage(&Usage::new(10, 5), "m");
        usage.record(&Usage::new(1, 2));
        assert_eq!(usage.calls, 2);
        assert_eq!(usage.input_tokens, 11);
        assert_eq!(usage.output_tokens, 7);
        assert_eq!(usage.total_tokens, 18);

        let priced = usage.with_costs(0.5, 0.25);
        assert!((priced.total_cost - 0.75).abs() < 1e-9);
    }
}
