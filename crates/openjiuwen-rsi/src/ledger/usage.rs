//! Cumulative usage totals and ledger-wide metrics.
//!
//! Python's observer keeps a running `RsiUsage` (`RsiUsageTokens` plus a call
//! count) while it replays and records calls. [`UsageTotals`] is that projection
//! for a ledger, and [`aggregate`] additionally buckets it by model and by
//! engine stage so a caller can answer "what did the planning stage cost?".
//!
//! Aggregation is pure: it reads [`RunEvent`]s and never mutates the ledger.
//! Only [`RunEvent::Usage`] events contribute; stage transitions carry no
//! billable information and are skipped.
use crate::ledger::record::{ModelCallStatus, RunEvent, UsageRecord, UsageTokens};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Stage bucket used when a usage record carries no `stage_ref`.
///
/// Mirrors Python's "call made outside a decorated stage" case, which it
/// attributes to the node but not to any engine stage.
pub const UNATTRIBUTED_STAGE: &str = "unattributed";

/// Cumulative usage counters for one ledger, model, or stage.
///
/// A counter the provider never reported contributes zero (the same rule as
/// Python's `_accumulate`, which must not erase counters accumulated from
/// earlier calls). `calls` counts every provider call, using the record's
/// `call_count` with a floor of one.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct UsageTotals {
    /// Number of provider calls.
    pub calls: u64,
    /// Cumulative input tokens.
    pub input_tokens: u64,
    /// Cumulative output tokens.
    pub output_tokens: u64,
    /// Cumulative cache-hit input tokens (a subset of `input_tokens`).
    pub cache_hit_tokens: u64,
    /// Cumulative caller-reported cost in US dollars.
    pub cost_usd: f64,
}

impl UsageTotals {
    /// Adds one ledger event to these totals; stage events are ignored.
    pub fn observe(&mut self, event: &RunEvent) {
        if let RunEvent::Usage(record) = event {
            self.observe_usage(record);
        }
    }

    /// Adds one usage record to these totals.
    pub fn observe_usage(&mut self, record: &UsageRecord) {
        let calls = record.model_call.call_count.max(1);
        self.calls = self.calls.saturating_add(calls);
        self.input_tokens = self
            .input_tokens
            .saturating_add(record.model_call.tokens.input.unwrap_or(0));
        self.output_tokens = self
            .output_tokens
            .saturating_add(record.model_call.tokens.output.unwrap_or(0));
        self.cache_hit_tokens = self
            .cache_hit_tokens
            .saturating_add(record.model_call.tokens.cache_hit.unwrap_or(0));
        self.cost_usd += record.cost_usd;
    }

    /// Merges `other` into these totals.
    pub fn merge(&mut self, other: &Self) {
        self.calls = self.calls.saturating_add(other.calls);
        self.input_tokens = self.input_tokens.saturating_add(other.input_tokens);
        self.output_tokens = self.output_tokens.saturating_add(other.output_tokens);
        self.cache_hit_tokens = self.cache_hit_tokens.saturating_add(other.cache_hit_tokens);
        self.cost_usd += other.cost_usd;
    }

    /// Reports whether nothing has been observed yet.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.calls == 0
            && self.input_tokens == 0
            && self.output_tokens == 0
            && self.cache_hit_tokens == 0
            && self.cost_usd == 0.0
    }

    /// Projects these totals onto a [`UsageTokens`] triple.
    #[must_use]
    pub const fn tokens(&self) -> UsageTokens {
        UsageTokens::new(
            Some(self.input_tokens),
            Some(self.output_tokens),
            Some(self.cache_hit_tokens),
        )
    }
}

impl std::ops::AddAssign<&Self> for UsageTotals {
    fn add_assign(&mut self, other: &Self) {
        self.merge(other);
    }
}

/// Ledger-wide usage plus failure counters and per-model / per-stage buckets.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct LedgerMetrics {
    /// Totals across every usage record.
    pub totals: UsageTotals,
    /// Calls that finished with [`ModelCallStatus::Failed`].
    pub failed_calls: u64,
    /// Calls that finished with [`ModelCallStatus::Incomplete`].
    pub incomplete_calls: u64,
    /// Per-model totals, keyed by the model identifier.
    pub by_model: BTreeMap<String, UsageTotals>,
    /// Per-stage totals, keyed by `stage_ref` or [`UNATTRIBUTED_STAGE`].
    pub by_stage: BTreeMap<String, UsageTotals>,
}

/// Aggregates a ledger's events into [`LedgerMetrics`].
///
/// Every usage record contributes its tokens, cost, and `call_count` (floored at
/// one) to the totals and to its model and stage bucket. Failed and incomplete
/// calls are counted separately. Stage transitions are ignored.
#[must_use]
pub fn aggregate(records: &[RunEvent]) -> LedgerMetrics {
    let mut metrics = LedgerMetrics::default();
    for event in records {
        let RunEvent::Usage(record) = event else {
            continue;
        };
        metrics.totals.observe_usage(record);
        let calls = record.model_call.call_count.max(1);
        match record.model_call.status {
            ModelCallStatus::Succeeded => {}
            ModelCallStatus::Failed => {
                metrics.failed_calls = metrics.failed_calls.saturating_add(calls);
            }
            ModelCallStatus::Incomplete => {
                metrics.incomplete_calls = metrics.incomplete_calls.saturating_add(calls);
            }
        }
        metrics
            .by_model
            .entry(record.model_call.model.clone())
            .or_default()
            .observe_usage(record);
        metrics
            .by_stage
            .entry(stage_bucket(record).to_owned())
            .or_default()
            .observe_usage(record);
    }
    metrics
}

/// Returns the stage bucket key of a usage record.
fn stage_bucket(record: &UsageRecord) -> &str {
    record
        .stage_ref
        .as_deref()
        .filter(|stage| !stage.is_empty())
        .unwrap_or(UNATTRIBUTED_STAGE)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ledger::record::{ModelCall, StageRecord};

    fn record(call_id: &str, tokens: UsageTokens) -> UsageRecord {
        UsageRecord::new("task", call_id, ModelCall::new("model").with_tokens(tokens))
    }

    #[test]
    fn totals_count_calls_with_a_floor_of_one() {
        let mut totals = UsageTotals::default();
        let mut unreported = record("c1", UsageTokens::default());
        unreported.model_call.call_count = 0;
        totals.observe_usage(&unreported);
        assert_eq!(totals.calls, 1);
        assert_eq!(totals.input_tokens, 0);

        let mut repeated = record("c2", UsageTokens::new(Some(4), Some(2), Some(1)));
        repeated.model_call.call_count = 3;
        repeated.cost_usd = 0.25;
        totals.observe_usage(&repeated);
        assert_eq!(totals.calls, 4);
        assert_eq!(totals.input_tokens, 4);
        assert_eq!(totals.output_tokens, 2);
        assert_eq!(totals.cache_hit_tokens, 1);
        assert!((totals.cost_usd - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn stage_events_contribute_nothing() {
        let mut totals = UsageTotals::default();
        totals.observe(&RunEvent::Stage(StageRecord::new(
            "h0",
            serde_json::Value::Null,
        )));
        assert!(totals.is_empty());
    }
}
