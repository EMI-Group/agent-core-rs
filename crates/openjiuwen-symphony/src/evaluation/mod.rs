//! Fail-soft capability evaluation: static checks, trace metrics, and window aggregation.
//!
//! Ports `openjiuwen.symphony.evaluation` (v0.1.20): the `Evaluator`/`Metric` seam, the
//! nine built-in evaluators, `latency_statistics`, and the `EvaluationSuite`
//! orchestration with event-time windowing, per-metric aggregation, and
//! `cache_signature()`.
//!
//! # Layers
//!
//! - This module owns the shared contract: [`Metric`](crate::evaluation::Metric),
//!   [`MetricScore`](crate::evaluation::MetricScore),
//!   [`MetricStatus`](crate::evaluation::MetricStatus),
//!   [`MetricScope`](crate::evaluation::MetricScope),
//!   [`EvaluationInput`](crate::evaluation::EvaluationInput),
//!   [`EvaluationReport`](crate::evaluation::EvaluationReport),
//!   [`QualityResult`](crate::evaluation::QualityResult),
//!   [`QualityConfidence`](crate::evaluation::QualityConfidence),
//!   [`aggregate_metric`](crate::evaluation::aggregate_metric), and
//!   [`EvaluationSuite`](crate::evaluation::EvaluationSuite).
//! - [`metrics`](crate::evaluation::metrics) owns the static card metrics.
//! - [`trace_metrics`](crate::evaluation::trace_metrics) owns the trace metrics, and
//!   [`suite`](crate::evaluation::suite) the metric list, aggregation, and cache
//!   signature.
//! - [`trace`](crate::evaluation::trace) owns the caller-supplied evidence: cases,
//!   capability calls, the event-time
//!   [`EvaluationWindow`](crate::evaluation::EvaluationWindow), and
//!   [`latency_statistics`](crate::evaluation::latency_statistics).
//!
//! Every public item of the pre-split `evaluation` module is re-exported here, so
//! `evaluation::CompletenessMetric`, `evaluation::EvaluationSuite`,
//! `evaluation::LlmJudgedMetric`, and friends keep their paths.
//!
//! # Scope
//!
//! An [`EvaluationInput`](crate::evaluation::EvaluationInput) carries one
//! [`CapabilityDescriptor`](crate::model::CapabilityDescriptor), an optional judge
//! `target`, and an optional
//! [`EvaluationTrace`](crate::evaluation::EvaluationTrace) plus
//! [`EvaluationWindow`](crate::evaluation::EvaluationWindow). Static metrics inspect the
//! card; trace metrics inspect the cases the window selects. Nothing here executes or
//! installs a capability, and no composite admission decision is invented:
//! [`QualityResult::score`](crate::evaluation::QualityResult::score) is always `None`
//! because Python's suite reports per-metric facts (`aggregation = "per_metric"`).
//!
//! # Case-window rules
//!
//! Window selection is deterministic: `start` is inclusive, `end` is exclusive, and a
//! case without `event_time` is dropped as soon as either boundary is set. Cross-case
//! aggregation groups samples by metric name and then applies
//! [`aggregate_metric`](crate::evaluation::aggregate_metric), which is Python's status
//! precedence — mean over scored samples with `Pass` only when every scored sample
//! passed, then `Error`, then `Observed`, then `NotApplicable`.
//!
//! # Fail-soft contract
//!
//! A metric that fails does not abort
//! [`EvaluationSuite::evaluate`](crate::evaluation::EvaluationSuite::evaluate): the error
//! is rendered into
//! [`EvaluationReport::errors`](crate::evaluation::EvaluationReport::errors) (prefixed
//! with the metric name) and the remaining metrics still run.
//! [`EvaluationSuite::evaluate_quality`](crate::evaluation::EvaluationSuite::evaluate_quality)
//! instead records a `status = error` sample, because Python's suite materialises
//! evaluator exceptions as error results. In practice only a model-backed metric
//! configured with [`FallbackMode::Error`](crate::llm::FallbackMode) can fail here.
//!
//! # Determinism
//!
//! Static metrics are pure functions of the card, trace metrics are pure functions of
//! the supplied evidence, and no metric reads the clock: an `event_time` always comes
//! from the case. Grouping uses `BTreeMap`, so per-metric output order is sorted, and
//! [`EvaluationSuite::cache_signature`](crate::evaluation::EvaluationSuite::cache_signature)
//! is a digest over a sorted-key document.
//!
//! # The cache key
//!
//! [`SYMPHONY_EVALUATION_PROTOCOL`](crate::evaluation::SYMPHONY_EVALUATION_PROTOCOL) is
//! Python's protocol string (`"symphony-evaluation-suite-v5"`).
//! [`EvaluationSuite::cache_signature`](crate::evaluation::EvaluationSuite::cache_signature)
//! returns the lowercase hex SHA-256 of the canonical, sorted-key JSON document
//! `{"metrics":["<name>:<scope>:<key>",…],"protocol":"<protocol>"}`, where `<key>` is
//! [`Metric::cache_key`](crate::evaluation::Metric::cache_key) when a metric supplies one
//! and `builtin:<name>` otherwise. The signature is stable across process runs and
//! independent of JSON key order, so it is a safe cache key for reused quality results.
//! Unlike Python, the Rust suite always has a signature: a metric without an explicit key
//! is identified by its name, and metric order is significant (registration order is part
//! of the contract).
use std::collections::BTreeMap;

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::model::CapabilityDescriptor;

pub mod metrics;
pub mod suite;
pub mod trace;
pub mod trace_metrics;

pub use metrics::{
    ClassificationConsistencyMetric, CompletenessMetric, DescriptionQualityMetric, LlmJudgedMetric,
    SchemaValidityMetric, StructureConformanceMetric,
};
pub use suite::{aggregate_metric, EvaluationSuite};
pub use trace::{
    latency_statistics, percentile, select_cases, CapabilityCall, CaseOutcome, EvaluationCase,
    EvaluationTrace, EvaluationWindow, LatencyStatistics,
};
pub use trace_metrics::{
    AccuracyMetric, CapabilitySelectionMetric, CompositionEffectivenessMetric, LatencyMetric,
    SuccessRateMetric,
};

/// Python's evaluation-cache protocol string (`_EVALUATION_CACHE_PROTOCOL`).
///
/// Bump-only: it is part of [`EvaluationSuite::cache_signature`], so changing the
/// aggregation or metric semantics must bump this constant to invalidate cached results.
pub const SYMPHONY_EVALUATION_PROTOCOL: &str = "symphony-evaluation-suite-v5";

/// The [`QualityResult::aggregation`] value: one aggregated result per metric, no
/// composite score.
pub const PER_METRIC_AGGREGATION: &str = "per_metric";

/// The outcome of one metric sample.
///
/// Python's `MetricStatus`, unchanged: `pass`, `fail`, `not_applicable`, `error`,
/// `observed`. Only `Pass` and `Fail` carry a score; the other three are scoreless.
///
/// [`MetricStatus::default`] is `Pass`, the neutral "this metric produced a verdict"
/// status, so a legacy payload without a `status` key still deserializes as a scored
/// sample. A metric that actually computes a verdict always sets `Pass` or `Fail`
/// explicitly (see [`MetricScore::scored`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MetricStatus {
    /// The metric scored the input and met its threshold.
    #[default]
    Pass,
    /// The metric scored the input and missed its threshold.
    Fail,
    /// The metric had no applicable input; it must not carry a score.
    NotApplicable,
    /// The metric could not produce a verdict; it must not carry a score.
    Error,
    /// The metric recorded a raw observation without applying a threshold.
    Observed,
}

impl MetricStatus {
    /// Whether the status carries a score (`Pass` or `Fail`).
    #[must_use]
    pub fn is_scored(self) -> bool {
        matches!(self, Self::Pass | Self::Fail)
    }

    /// The `snake_case` wire name of the status.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pass => "pass",
            Self::Fail => "fail",
            Self::NotApplicable => "not_applicable",
            Self::Error => "error",
            Self::Observed => "observed",
        }
    }
}

/// Whether a metric reads static card metadata or caller-supplied trace evidence.
///
/// Python's `EvaluationScope`. The suite uses it to split
/// [`EvaluationSuite::evaluate_static`] from [`EvaluationSuite::evaluate_case`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MetricScope {
    /// The metric inspects the capability card only.
    #[default]
    Static,
    /// The metric inspects one or more [`EvaluationCase`]s.
    Trace,
}

impl MetricScope {
    /// The `snake_case` wire name of the scope.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Static => "static",
            Self::Trace => "trace",
        }
    }
}

/// Confidence assigned to an aggregated quality result.
///
/// Python's `QualityConfidence`, derived from the sample count: `0 → None`, `1 → Low`,
/// `2..=9 → Normal`, `10+ → High`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum QualityConfidence {
    /// No sample was available.
    #[default]
    None,
    /// Exactly one sample.
    Low,
    /// Two to nine samples.
    Normal,
    /// Ten or more samples.
    High,
}

impl QualityConfidence {
    /// Maps a sample count onto the confidence tier, using Python's thresholds.
    #[must_use]
    pub fn from_sample_count(sample_count: usize) -> Self {
        match sample_count {
            0 => Self::None,
            1 => Self::Low,
            2..=9 => Self::Normal,
            _ => Self::High,
        }
    }

    /// The `snake_case` wire name of the tier.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Low => "low",
            Self::Normal => "normal",
            Self::High => "high",
        }
    }
}

/// The subject of one metric evaluation.
///
/// `trace` and `window` are optional: a static-only evaluation leaves both unset, and
/// every pre-existing constructor (`EvaluationInput::default`, struct literals with the
/// two original fields via [`Default`], [`EvaluationInput::new`]) keeps working.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct EvaluationInput {
    /// The capability card under evaluation.
    pub capability: CapabilityDescriptor,
    /// Optional judge target, for example the model a deferred metric would call.
    pub target: Option<String>,
    /// Caller-supplied evidence for the trace metrics.
    pub trace: Option<EvaluationTrace>,
    /// Optional event-time window applied before a trace metric runs.
    pub window: Option<EvaluationWindow>,
}

impl EvaluationInput {
    /// Builds a static evaluation of `capability`, with no trace and no window.
    #[must_use]
    pub fn new(capability: CapabilityDescriptor) -> Self {
        Self {
            capability,
            ..Self::default()
        }
    }

    /// Sets the optional judge target.
    #[must_use]
    pub fn with_target(mut self, target: impl Into<String>) -> Self {
        self.target = Some(target.into());
        self
    }

    /// Attaches caller-supplied trace evidence.
    #[must_use]
    pub fn with_trace(mut self, trace: EvaluationTrace) -> Self {
        self.trace = Some(trace);
        self
    }

    /// Attaches an event-time window.
    #[must_use]
    pub fn with_window(mut self, window: EvaluationWindow) -> Self {
        self.window = Some(window);
        self
    }

    /// The cases a metric may inspect after applying the window.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::Error::Validation`] for an invalid window or an
    /// unparseable case `event_time`; see
    /// [`select_cases`].
    pub fn selected_cases(&self) -> Result<Vec<&EvaluationCase>> {
        let Some(trace) = self.trace.as_ref() else {
            return Ok(Vec::new());
        };
        select_cases(&trace.cases, self.window.as_ref())
    }

    /// The number of cases selected by the window.
    ///
    /// # Errors
    ///
    /// Propagates the [`Self::selected_cases`] validation errors.
    pub fn sample_count(&self) -> Result<usize> {
        self.selected_cases().map(|cases| cases.len())
    }
}

/// One metric's verdict on a capability, or one trace case.
///
/// `score` is normalized to `[0.0, 1.0]` and is `None` for the scoreless statuses
/// (`NotApplicable`, `Error`, `Observed`). `passed` is the metric's own threshold
/// decision, and `detail` explains a partial or failing result (it is `None` when the
/// metric is fully satisfied).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct MetricScore {
    /// The metric name, matching [`Metric::name`].
    pub metric: String,
    /// The normalized score in `[0.0, 1.0]`; `None` when the status is scoreless.
    pub score: Option<f64>,
    /// Whether the score meets the metric's pass threshold.
    pub passed: bool,
    /// Human-readable explanation of missing pieces or offending names.
    pub detail: Option<String>,
    /// The metric status; defaults to `pass` for legacy payloads without the field.
    pub status: MetricStatus,
}

impl MetricScore {
    /// Builds a scored verdict, with `status` derived from `passed`.
    #[must_use]
    pub fn scored(metric: impl Into<String>, score: f64, passed: bool) -> Self {
        Self {
            metric: metric.into(),
            score: Some(score),
            passed,
            detail: None,
            status: if passed {
                MetricStatus::Pass
            } else {
                MetricStatus::Fail
            },
        }
    }

    /// Builds a `Fail` verdict with score `0.0`, used by the deterministic guards.
    #[must_use]
    pub fn failed(metric: impl Into<String>, detail: impl Into<String>) -> Self {
        Self::scored(metric, 0.0, false).with_detail(detail)
    }

    /// Builds a scoreless `NotApplicable` verdict.
    #[must_use]
    pub fn not_applicable(metric: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            metric: metric.into(),
            score: None,
            passed: false,
            detail: Some(detail.into()),
            status: MetricStatus::NotApplicable,
        }
    }

    /// Builds a scoreless `Error` verdict.
    #[must_use]
    pub fn error(metric: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            metric: metric.into(),
            score: None,
            passed: false,
            detail: Some(detail.into()),
            status: MetricStatus::Error,
        }
    }

    /// Builds a scoreless `Observed` verdict (raw data, no threshold).
    #[must_use]
    pub fn observed(metric: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            metric: metric.into(),
            score: None,
            passed: false,
            detail: Some(detail.into()),
            status: MetricStatus::Observed,
        }
    }

    /// Attaches a `detail` string.
    #[must_use]
    pub fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = Some(detail.into());
        self
    }

    /// Attaches `detail` when present, leaving the sample unchanged otherwise.
    #[must_use]
    pub fn with_optional_detail(mut self, detail: Option<String>) -> Self {
        if detail.is_some() {
            self.detail = detail;
        }
        self
    }

    /// The score that participates in aggregation: `Some` only for a scored status.
    #[must_use]
    pub fn effective_score(&self) -> Option<f64> {
        if self.status.is_scored() {
            self.score
        } else {
            None
        }
    }
}

/// The fail-soft result of evaluating one capability.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct EvaluationReport {
    /// The id of the evaluated capability.
    pub capability_id: String,
    /// The successful metric scores, in suite order.
    pub scores: Vec<MetricScore>,
    /// The mean of the scored samples; `0.0` when there are none.
    pub overall: f64,
    /// Metric failures, rendered as `"<metric>: <error>"`, in suite order.
    pub errors: Vec<String>,
}

/// The aggregated quality result for one capability and evidence window.
///
/// Python's `QualityResult`, reduced to the fields this port reports: the confidence
/// tier, an always-`None` composite `score`, the `"per_metric"` aggregation marker, the
/// sample count the confidence was derived from, the applied window, and one aggregated
/// [`MetricScore`] per metric name (sorted, so the map is deterministic).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct QualityResult {
    /// The confidence tier derived from [`QualityResult::sample_count`].
    pub confidence: QualityConfidence,
    /// Always `None`: Python never computes a composite admission score.
    pub score: Option<f64>,
    /// The aggregation mode, always [`PER_METRIC_AGGREGATION`].
    pub aggregation: String,
    /// The number of samples the aggregation covers.
    pub sample_count: usize,
    /// The applied event-time window, if any.
    pub window: Option<EvaluationWindow>,
    /// One aggregated sample per metric name.
    pub per_metric: BTreeMap<String, MetricScore>,
}

impl Default for QualityResult {
    fn default() -> Self {
        Self {
            confidence: QualityConfidence::None,
            score: None,
            aggregation: PER_METRIC_AGGREGATION.to_string(),
            sample_count: 0,
            window: None,
            per_metric: BTreeMap::new(),
        }
    }
}

/// One deterministic or model-backed quality check over an [`EvaluationInput`].
///
/// Implementations must be cheap to share (`Send + Sync`, kept behind an `Arc`) and
/// must not panic on malformed input: an unusable input is a low score or an
/// [`openjiuwen_common::error::Error`], never a crash.
#[async_trait]
pub trait Metric: Send + Sync {
    /// The metric name recorded in [`MetricScore::metric`].
    fn name(&self) -> &str;

    /// Whether the metric reads the card (`Static`) or trace evidence (`Trace`).
    fn scope(&self) -> MetricScope {
        MetricScope::Static
    }

    /// An optional secret-free cache key for this metric implementation.
    ///
    /// The key participates in [`EvaluationSuite::cache_signature`]; `None` means the
    /// metric is identified by its name alone.
    fn cache_key(&self) -> Option<String> {
        None
    }

    /// Scores `input`, returning the verdict or the reason it could not be computed.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error`] when the metric cannot produce a
    /// verdict, for example when a model-backed metric keeps receiving unusable replies
    /// and is configured with [`FallbackMode::Error`](crate::llm::FallbackMode), or when
    /// the input's window is invalid. The suite records the error instead of aborting.
    async fn evaluate(&self, input: &EvaluationInput) -> Result<MetricScore>;
}
