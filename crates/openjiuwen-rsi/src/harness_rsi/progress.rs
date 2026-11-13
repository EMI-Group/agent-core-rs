//! Optimisation progress events shared by the harness-side loops.
//!
//! Ported from Python `openjiuwen.rsi.harness_rsi.progress`. An
//! [`OptimizationProgressEvent`] is one snapshot of a running optimisation: its
//! phase, epoch, batch and stage, the latest score and improvement flag, a
//! human-readable message, plus free-form metrics and artifacts.
//!
//! Python's callback is a plain synchronous callable
//! (`Callable[[OptimizationProgressEvent], None]`), so [`ProgressCallback`] is a
//! synchronous [`Fn`] too. [`emit_progress`] never fails and never swallows a
//! panic-suppressing wrapper: the callback runs inline, and implementations are
//! expected to hand the event to a queue or channel rather than block the
//! optimisation loop.
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

/// A single progress event emitted by the optimisation orchestrator.
///
/// Serialises with `snake_case` keys and defaults for every field, matching the
/// Python dataclass, and tolerates unknown fields so a newer producer can add
/// them without breaking older consumers.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct OptimizationProgressEvent {
    /// Optimisation phase that produced the event.
    pub phase: String,
    /// Epoch (outer optimisation round) index.
    pub epoch: u32,
    /// Batch index inside the epoch.
    pub batch_index: u32,
    /// Stage name inside the batch.
    pub stage: String,
    /// Latest score, when the stage produced one.
    pub score: Option<f64>,
    /// Whether the latest candidate improved on the incumbent.
    pub improved: Option<bool>,
    /// Human-readable progress message; empty when absent.
    pub message: String,
    /// Free-form numeric or structured metrics.
    pub metrics: BTreeMap<String, Value>,
    /// Free-form artifacts produced by the stage.
    pub artifacts: BTreeMap<String, Value>,
}

impl OptimizationProgressEvent {
    /// Creates an event for `phase` with every other field defaulted.
    #[must_use]
    pub fn new(phase: impl Into<String>) -> Self {
        Self {
            phase: phase.into(),
            ..Self::default()
        }
    }

    /// Attaches a score.
    #[must_use]
    pub fn with_score(mut self, score: f64) -> Self {
        self.score = Some(score);
        self
    }

    /// Sets the improvement flag.
    #[must_use]
    pub fn with_improved(mut self, improved: bool) -> Self {
        self.improved = Some(improved);
        self
    }

    /// Sets the progress message.
    #[must_use]
    pub fn with_message(mut self, message: impl Into<String>) -> Self {
        self.message = message.into();
        self
    }
}

/// A synchronous progress sink.
///
/// Python's `ProgressCallback` is a plain callable, so the Rust port keeps the
/// same shape instead of an async trait. The callback must not block: a slow
/// consumer belongs behind a channel.
pub type ProgressCallback = Arc<dyn Fn(&OptimizationProgressEvent) + Send + Sync>;

/// Invokes `callback` with `event`, or does nothing when no callback is set.
///
/// This is Python's `emit`: an injected sink is called once per event, and the
/// absent-sink case is a no-op rather than an error.
pub fn emit_progress(callback: Option<&ProgressCallback>, event: &OptimizationProgressEvent) {
    if let Some(callback) = callback {
        callback(event);
    }
}

/// Decision about whether a seed optimisation result is accepted.
///
/// Python's `SeedOptimizationDecisionCallback` returns a `bool` (or an awaitable
/// of one); the Rust port names the two outcomes so call sites read clearly and
/// conversion back to the Python wire value is explicit.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SeedOptimizationDecision {
    /// The seed is accepted as the new incumbent.
    #[default]
    Accept,
    /// The seed is rejected and the previous incumbent is kept.
    Reject,
}

impl SeedOptimizationDecision {
    /// Converts the Python callback's boolean into a decision.
    #[must_use]
    pub const fn from_bool(accepted: bool) -> Self {
        if accepted {
            Self::Accept
        } else {
            Self::Reject
        }
    }

    /// Returns the decision as the Python callback's boolean.
    #[must_use]
    pub const fn as_bool(self) -> bool {
        matches!(self, Self::Accept)
    }

    /// Returns the `snake_case` wire name of the decision.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Accept => "accept",
            Self::Reject => "reject",
        }
    }
}

impl From<bool> for SeedOptimizationDecision {
    fn from(accepted: bool) -> Self {
        Self::from_bool(accepted)
    }
}

impl fmt::Display for SeedOptimizationDecision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn events_round_trip_with_defaults_and_unknown_fields() {
        let event = OptimizationProgressEvent::new("optimize")
            .with_score(0.75)
            .with_improved(true)
            .with_message("step 2");
        let value = serde_json::to_value(&event).unwrap();
        assert_eq!(value["phase"], "optimize");
        assert_eq!(value["score"], 0.75);
        assert_eq!(value["improved"], true);
        let round_trip: OptimizationProgressEvent = serde_json::from_value(value).unwrap();
        assert_eq!(round_trip, event);

        let partial: OptimizationProgressEvent =
            serde_json::from_value(json!({"phase": "plan", "unknown": 1})).unwrap();
        assert_eq!(partial.phase, "plan");
        assert_eq!(partial.epoch, 0);
        assert!(partial.metrics.is_empty());
    }

    #[test]
    fn decisions_convert_to_and_from_booleans() {
        assert_eq!(
            SeedOptimizationDecision::from_bool(true),
            SeedOptimizationDecision::Accept
        );
        assert_eq!(
            SeedOptimizationDecision::from_bool(false),
            SeedOptimizationDecision::Reject
        );
        assert_eq!(
            SeedOptimizationDecision::from(false),
            SeedOptimizationDecision::Reject
        );
        assert!(SeedOptimizationDecision::Accept.as_bool());
        assert!(!SeedOptimizationDecision::Reject.as_bool());
        assert_eq!(SeedOptimizationDecision::Accept.as_str(), "accept");
        assert_eq!(SeedOptimizationDecision::Reject.to_string(), "reject");
        assert_eq!(
            SeedOptimizationDecision::default(),
            SeedOptimizationDecision::Accept
        );
    }
}
