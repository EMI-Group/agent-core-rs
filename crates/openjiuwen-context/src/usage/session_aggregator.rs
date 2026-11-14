//! Session-level provider KV-cache aggregation.
//!
//! Ports `usage/session_aggregator.py`: [`CacheAggregationKey`] identifies one
//! cache owner, and [`SessionKVCacheAggregator`] keeps token-weighted rollups
//! per key with request-id idempotency.
//!
//! Accumulation is order-independent; the only ordering effect is that
//! [`SessionKVCacheAggregator::snapshot`] returns the state at call time, so
//! every event carries "session to date" at its own emission point.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use super::models::{
    RequestKVCacheUsage, SessionKVCacheUsage, STATUS_FULL_HIT, STATUS_INVALID, STATUS_MISS,
    STATUS_NOT_REPORTED, STATUS_PARTIAL_HIT,
};

/// The identity of one cache aggregation scope.
///
/// Distinct deployments (`api_base`) or cache identities therefore land in
/// distinct buckets, and a teammate or child agent that shares a product
/// session is not merged with its parent. The type is `Ord` and hashable so it
/// can key a map.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(default)]
pub struct CacheAggregationKey {
    /// The session id the calls belong to.
    pub session_id: String,
    /// The provider id.
    pub provider: String,
    /// The model id.
    pub model: String,
    /// The deployment (`api_base`-level) id; empty when unknown.
    pub deployment: String,
    /// The provider cache mode; `provider` by default.
    pub cache_mode: String,
    /// The cache scope; `session` by default.
    pub cache_scope: String,
    /// The product-level session id.
    pub product_session_id: Option<String>,
    /// The execution-level session id.
    pub execution_session_id: Option<String>,
    /// The provider cache identity of this owner.
    pub cache_identity: Option<String>,
    /// The cache identity of the parent owner.
    pub parent_cache_identity: Option<String>,
    /// The team id.
    pub team_id: Option<String>,
    /// The agent id.
    pub agent_id: Option<String>,
    /// The team member name.
    pub member_name: Option<String>,
}

impl CacheAggregationKey {
    /// Builds a key from the three mandatory identity fields.
    #[must_use]
    pub fn new(
        session_id: impl Into<String>,
        provider: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            session_id: session_id.into(),
            provider: provider.into(),
            model: model.into(),
            ..Self::default()
        }
    }
}

impl Default for CacheAggregationKey {
    fn default() -> Self {
        Self {
            session_id: String::new(),
            provider: String::new(),
            model: String::new(),
            deployment: String::new(),
            cache_mode: "provider".to_owned(),
            cache_scope: "session".to_owned(),
            product_session_id: None,
            execution_session_id: None,
            cache_identity: None,
            parent_cache_identity: None,
            team_id: None,
            agent_id: None,
            member_name: None,
        }
    }
}

/// The merged data quality of the observed calls of one scope.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum DataQuality {
    /// No call has been classified yet.
    #[default]
    NotReported,
    /// The provider reported a complete, consistent cache report.
    Authoritative,
    /// The miss counter had to be derived from input and read.
    Derived,
    /// At least one call was internally inconsistent.
    Invalid,
    /// At least one call reported only part of the cache counters.
    Partial,
    /// Calls disagreed about their quality.
    Mixed,
}

impl DataQuality {
    /// The wire name of the quality.
    fn as_str(self) -> &'static str {
        match self {
            Self::NotReported => STATUS_NOT_REPORTED,
            Self::Authoritative => "authoritative",
            Self::Derived => "derived",
            Self::Invalid => STATUS_INVALID,
            Self::Partial => "partial",
            Self::Mixed => "mixed",
        }
    }
}

/// The classification of one request report.
enum Classification {
    /// The report is inconsistent and is excluded from the totals.
    Invalid,
    /// The report is incomplete and is excluded from the totals.
    Partial,
    /// The report carries no usage at all.
    NotReported,
    /// The report enters the totals.
    Observed {
        /// Observed input tokens.
        input: u64,
        /// Observed cache-read tokens.
        read: u64,
        /// Observed or derived cache-miss tokens.
        miss: u64,
        /// Whether the miss counter was derived.
        derived: bool,
        /// The reported cache-write tokens, when any.
        write: Option<u64>,
    },
}

/// The running totals of one aggregation scope.
#[derive(Debug, Clone, Default)]
struct Accumulator {
    /// Calls recorded, repeated request ids counted once.
    calls_total: u64,
    /// Calls that entered the totals.
    calls_observed: u64,
    /// Calls excluded as incomplete or inconsistent.
    calls_partial: u64,
    /// Calls that reported nothing.
    calls_not_reported: u64,
    /// Observed input tokens.
    input_tokens_total: u64,
    /// Observed cache-read tokens.
    cache_read_tokens_total: u64,
    /// Observed cache-miss tokens.
    cache_miss_tokens_total: u64,
    /// Observed cache-write tokens; meaningful only with `cache_write_calls`.
    cache_write_tokens_total: u64,
    /// Observed calls that reported a write counter.
    cache_write_calls: u64,
    /// Request ids already recorded.
    request_ids: BTreeSet<String>,
    /// The merged quality state.
    data_quality: DataQuality,
}

impl Accumulator {
    /// Merges one call's quality, turning the first disagreement into `mixed`.
    fn merge_quality(&mut self, quality: DataQuality) {
        if self.data_quality == DataQuality::NotReported {
            self.data_quality = quality;
        } else if self.data_quality != quality {
            self.data_quality = DataQuality::Mixed;
        }
    }
}

/// Aggregates cache usage with request-id idempotency and weighted averages.
#[derive(Debug, Clone, Default)]
pub struct SessionKVCacheAggregator {
    /// The per-scope accumulators.
    values: BTreeMap<CacheAggregationKey, Accumulator>,
}

impl SessionKVCacheAggregator {
    /// Creates an empty aggregator.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records one request report and returns the scope's current snapshot.
    ///
    /// Recording the same `request_id` again returns the current snapshot
    /// unchanged and does not increment `calls_total`.
    pub fn record(
        &mut self,
        request_id: &str,
        scope_key: &CacheAggregationKey,
        usage: &RequestKVCacheUsage,
    ) -> SessionKVCacheUsage {
        if self
            .values
            .get(scope_key)
            .is_some_and(|accumulator| accumulator.request_ids.contains(request_id))
        {
            return self.snapshot(scope_key);
        }
        let classification = classify(usage);
        let accumulator = self.values.entry(scope_key.clone()).or_default();
        accumulator.request_ids.insert(request_id.to_owned());
        accumulator.calls_total = accumulator.calls_total.saturating_add(1);
        match classification {
            Classification::Invalid => {
                accumulator.calls_partial = accumulator.calls_partial.saturating_add(1);
                accumulator.merge_quality(DataQuality::Invalid);
            }
            Classification::Partial => {
                accumulator.calls_partial = accumulator.calls_partial.saturating_add(1);
                accumulator.merge_quality(DataQuality::Partial);
            }
            Classification::NotReported => {
                accumulator.calls_not_reported = accumulator.calls_not_reported.saturating_add(1);
            }
            Classification::Observed {
                input,
                read,
                miss,
                derived,
                write,
            } => {
                accumulator.calls_observed = accumulator.calls_observed.saturating_add(1);
                accumulator.input_tokens_total =
                    accumulator.input_tokens_total.saturating_add(input);
                accumulator.cache_read_tokens_total =
                    accumulator.cache_read_tokens_total.saturating_add(read);
                accumulator.cache_miss_tokens_total =
                    accumulator.cache_miss_tokens_total.saturating_add(miss);
                accumulator.merge_quality(if derived {
                    DataQuality::Derived
                } else {
                    DataQuality::Authoritative
                });
                if let Some(write) = write {
                    accumulator.cache_write_calls = accumulator.cache_write_calls.saturating_add(1);
                    accumulator.cache_write_tokens_total =
                        accumulator.cache_write_tokens_total.saturating_add(write);
                }
            }
        }
        self.snapshot(scope_key)
    }

    /// The session-to-date rollup of one scope.
    ///
    /// An unknown scope returns an empty rollup carrying `scope_key`, so a
    /// caller can emit a "nothing observed yet" block.
    #[must_use]
    pub fn snapshot(&self, scope_key: &CacheAggregationKey) -> SessionKVCacheUsage {
        let Some(accumulator) = self.values.get(scope_key) else {
            return SessionKVCacheUsage {
                aggregation_key: scope_key.clone(),
                ..SessionKVCacheUsage::default()
            };
        };
        let observed = accumulator.calls_observed;
        let input_total = accumulator.input_tokens_total;
        let write_total = if observed > 0 && accumulator.cache_write_calls == observed {
            Some(accumulator.cache_write_tokens_total)
        } else {
            None
        };
        SessionKVCacheUsage {
            scope: "session".to_owned(),
            aggregation_method: "token_weighted".to_owned(),
            aggregation_key: scope_key.clone(),
            calls_total: accumulator.calls_total,
            calls_observed: observed,
            calls_partial: accumulator.calls_partial,
            calls_not_reported: accumulator.calls_not_reported,
            input_tokens_total: input_total,
            cache_read_tokens_total: accumulator.cache_read_tokens_total,
            cache_miss_tokens_total: accumulator.cache_miss_tokens_total,
            cache_write_tokens_total: write_total,
            cache_write_coverage: write_coverage(accumulator).to_owned(),
            average_input_per_call: ratio(input_total, observed),
            weighted_hit_rate: ratio(accumulator.cache_read_tokens_total, input_total),
            average_cache_read_per_call: ratio(accumulator.cache_read_tokens_total, observed),
            average_cache_miss_per_call: ratio(accumulator.cache_miss_tokens_total, observed),
            status: status(accumulator).to_owned(),
            data_quality: accumulator.data_quality.as_str().to_owned(),
        }
    }

    /// The number of scopes with recorded calls.
    #[must_use]
    pub fn scope_count(&self) -> usize {
        self.values.len()
    }

    /// Whether no call has been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// Classifies one request report exactly like Python's `record` ladder.
fn classify(usage: &RequestKVCacheUsage) -> Classification {
    if usage.invalid_reason.is_some() {
        return Classification::Invalid;
    }
    let (Some(input), Some(read)) = (usage.input_tokens, usage.cache_read_tokens) else {
        let reported = usage.input_tokens.is_some()
            || usage.cache_read_tokens.is_some()
            || usage.cache_miss_tokens.is_some()
            || usage.cache_write_tokens.is_some();
        return if reported {
            Classification::Partial
        } else {
            Classification::NotReported
        };
    };
    if read > input {
        return Classification::Invalid;
    }
    let expected_miss = input.saturating_sub(read);
    let derived = usage.cache_miss_tokens.is_none() || usage.miss_tokens_derived;
    let miss = usage.cache_miss_tokens.unwrap_or(expected_miss);
    if miss > expected_miss {
        return Classification::Invalid;
    }
    if let Some(write) = usage.cache_write_tokens {
        if write > miss {
            return Classification::Invalid;
        }
    }
    Classification::Observed {
        input,
        read,
        miss,
        derived,
        write: usage.cache_write_tokens,
    }
}

/// `numerator / denominator`, or `None` when the denominator is zero.
fn ratio(numerator: u64, denominator: u64) -> Option<f64> {
    if denominator == 0 {
        None
    } else {
        Some(numerator as f64 / denominator as f64)
    }
}

/// The cache-write coverage of one scope.
fn write_coverage(accumulator: &Accumulator) -> &'static str {
    if accumulator.cache_write_calls == 0 {
        "none"
    } else if accumulator.cache_write_calls == accumulator.calls_observed {
        "complete"
    } else {
        "partial"
    }
}

/// The rollup status of one scope.
fn status(accumulator: &Accumulator) -> &'static str {
    if accumulator.calls_observed == 0 {
        return if accumulator.data_quality == DataQuality::Invalid {
            STATUS_INVALID
        } else {
            STATUS_NOT_REPORTED
        };
    }
    if accumulator.input_tokens_total == 0 {
        return STATUS_NOT_REPORTED;
    }
    if accumulator.cache_read_tokens_total == 0 {
        return STATUS_MISS;
    }
    if accumulator.cache_miss_tokens_total == 0 {
        return STATUS_FULL_HIT;
    }
    STATUS_PARTIAL_HIT
}
