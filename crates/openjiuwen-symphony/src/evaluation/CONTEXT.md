# evaluation

## Intent

Fail-soft capability evaluation, porting Python `openjiuwen.symphony.evaluation` (v0.1.20):
`base.py`'s `Evaluator`/`MetricResult` seam, the static and trace evaluators of
`evaluators.py`, and `suite.py`'s window selection, per-metric aggregation, and
`cache_signature()`.
Static metrics inspect a `CapabilityDescriptor`; trace metrics inspect caller-supplied
cases (capability calls, latencies, answers) selected by an event-time window.
Nothing here executes, installs, or admits a capability, and no composite score is invented:
`QualityResult::score` is always `None` and `aggregation` is always `"per_metric"`.
Everything is offline and deterministic by default; judge-backed metrics run with
`FallbackMode::Deterministic`, so a suite always produces a verdict without a model.

## Module split

- `mod.rs` — shared contract: `Metric`, `MetricScore`, `MetricStatus`, `MetricScope`,
  `EvaluationInput`, `EvaluationReport`, `QualityResult`, `QualityConfidence`, the protocol
  constants, and the `pub use` re-exports that keep every pre-split path working.
- `metrics.rs` — static card metrics plus the `LlmJudgedMetric` wrapper and the shared
  judge helper `normalize_status`, and the `#[cfg(test)] complete_capability()` fixture.
- `trace_metrics.rs` — the five evidence metrics and the `evaluate_trace_cases` helper.
- `suite.rs` — `aggregate_metric`, `EvaluationSuite`, and `mean_score`.
- `trace.rs` — evidence types, window selection, `latency_statistics`/`percentile`.

## API Surface (`mod.rs`)

```rust
pub const SYMPHONY_EVALUATION_PROTOCOL: &str = "symphony-evaluation-suite-v5"; // cache-key protocol
pub const PER_METRIC_AGGREGATION: &str = "per_metric";                         // QualityResult::aggregation

pub enum MetricStatus { Pass, Fail, NotApplicable, Error, Observed }  // snake_case, Default = Pass
impl MetricStatus { pub fn is_scored(self) -> bool; pub fn as_str(self) -> &'static str; }

pub enum MetricScope { Static, Trace }        // Default = Static; as_str()
pub enum QualityConfidence { None, Low, Normal, High }  // Default = None; as_str()
impl QualityConfidence { pub fn from_sample_count(sample_count: usize) -> Self; } // 0/1/2-9/>=10

pub struct EvaluationInput { capability: CapabilityDescriptor, target: Option<String>,
                              trace: Option<EvaluationTrace>, window: Option<EvaluationWindow> }
impl EvaluationInput {
    pub fn new(capability: CapabilityDescriptor) -> Self;
    pub fn with_target(self, target: impl Into<String>) -> Self;
    pub fn with_trace(self, trace: EvaluationTrace) -> Self;
    pub fn with_window(self, window: EvaluationWindow) -> Self;
    pub fn selected_cases(&self) -> Result<Vec<&EvaluationCase>>;   // window-filtered
    pub fn sample_count(&self) -> Result<usize>;                    // selected_cases().len()
}

pub struct MetricScore { metric: String, score: Option<f64>, passed: bool,
                         detail: Option<String>, status: MetricStatus }
impl MetricScore {
    pub fn scored(metric, score: f64, passed: bool) -> Self;        // status Pass/Fail
    pub fn failed(metric, detail) -> Self;                          // score 0.0, status Fail
    pub fn not_applicable(metric, detail) -> Self;                  // score None
    pub fn error(metric, detail) -> Self;                           // score None
    pub fn observed(metric, detail) -> Self;                        // score None
    pub fn with_detail(self, detail) -> Self;
    pub fn with_optional_detail(self, detail: Option<String>) -> Self;
    pub fn effective_score(&self) -> Option<f64>;                   // Some only when scored
}

pub struct EvaluationReport { capability_id: String, scores: Vec<MetricScore>,
                              overall: f64, errors: Vec<String> }
pub struct QualityResult { confidence: QualityConfidence, score: Option<f64>,
                           aggregation: String, sample_count: usize,
                           window: Option<EvaluationWindow>,
                           per_metric: BTreeMap<String, MetricScore> }

#[async_trait]
pub trait Metric: Send + Sync {
    fn name(&self) -> &str;
    fn scope(&self) -> MetricScope { MetricScope::Static }
    fn cache_key(&self) -> Option<String> { None }
    async fn evaluate(&self, input: &EvaluationInput) -> Result<MetricScore>;
}
```

Re-exported here: `CompletenessMetric`, `SchemaValidityMetric`,
`StructureConformanceMetric`, `DescriptionQualityMetric`, `ClassificationConsistencyMetric`,
`LlmJudgedMetric` (from `metrics`); `SuccessRateMetric`, `LatencyMetric`, `AccuracyMetric`,
`CapabilitySelectionMetric`, `CompositionEffectivenessMetric` (from `trace_metrics`);
`aggregate_metric`, `EvaluationSuite` (from `suite`); `CapabilityCall`, `CaseOutcome`,
`EvaluationCase`, `EvaluationTrace`, `EvaluationWindow`, `LatencyStatistics`,
`latency_statistics`, `percentile`, `select_cases` (from `trace`).

## API Surface (`metrics`)

```rust
pub struct CompletenessMetric;                 // description, tags, described inputs, described outputs
pub struct SchemaValidityMetric;               // blank/duplicate IO names, required-plus-default
pub struct StructureConformanceMetric;         // seven-field fingerprint contract, all-or-nothing
pub struct DescriptionQualityMetric { .. }     // empty-description guard, then the judge
pub struct ClassificationConsistencyMetric { .. } // empty classification+tags guard, then the judge
pub struct LlmJudgedMetric { .. }              // wrapper over llm::LlmJudgeMetric ("llm_judged", DEFAULT_RUBRIC)
```

Each metric is a unit struct (`new()`, `Default`) except the judge-backed ones, which add
`with_model(Arc<dyn CompletionModel>)` and `has_model()`; `LlmJudgedMetric` also adds
`with_rubric`, `with_fallback`, and `rubric()`.
Every judge-backed metric names its model-free verdict with a `deterministic_fallback:`
detail prefix.

## API Surface (`trace_metrics`)

```rust
pub struct SuccessRateMetric;                  // trace; mean of the recorded call outcomes
pub struct LatencyMetric;                      // trace; Observed, score None, statistics in detail
pub struct AccuracyMetric { .. }               // exact-match short-circuit, else the judge
pub struct CapabilitySelectionMetric { .. }    // NotApplicable with no calls, else the judge
pub struct CompositionEffectivenessMetric { .. } // NotApplicable with < 2 calls, else the judge
```

The three judge-backed trace metrics add `with_model`/`has_model`; all five expose `new()`
and `Default`.

## API Surface (`suite`)

```rust
pub fn aggregate_metric(metric: &str, samples: &[MetricScore]) -> MetricScore;

pub struct EvaluationSuite;
impl EvaluationSuite {
    pub fn new(metrics: Vec<Arc<dyn Metric>>) -> Self;
    pub fn default_suite() -> Self;                     // CompletenessMetric + SchemaValidityMetric
    pub fn with_llm_judge(self) -> Self;                // appends the model-free LlmJudgedMetric
    pub fn with_llm_judge_model(self, model: Arc<dyn CompletionModel>) -> Self;
    pub fn metrics(&self) -> &[Arc<dyn Metric>];
    pub fn cache_signature(&self) -> String;            // SHA-256 hex of the protocol + metric keys
    pub async fn evaluate(&self, input: &EvaluationInput) -> EvaluationReport;
    pub async fn evaluate_quality(&self, input: &EvaluationInput) -> Result<QualityResult>;
    pub async fn evaluate_static(&self, input: &EvaluationInput) -> Vec<MetricScore>;
    pub async fn evaluate_case(&self, input: &EvaluationInput, case: &EvaluationCase) -> Vec<MetricScore>;
    pub fn aggregate(&self, samples: &[MetricScore], window: Option<&EvaluationWindow>,
                     sample_count: Option<usize>) -> Result<QualityResult>;
}
```

## Rules matched from Python

- **Structure conformance** inspects exactly seven fields (`capability_id`,
  `capability_type`, `name`, `content_hash`, `inputs`, `outputs`, `tags`) and is
  all-or-nothing: every issue is named in `detail`, the score is `1.0` or `0.0`.
- **Success rate** contributes `1.0` per `Success` case, `0.0` per `Failure` case, and a
  `not_applicable` sample when no call recorded an outcome; with no case at all the metric
  is `not_applicable` with `score = None`.
- **Latency** is `Observed` with `score = None`; the detail carries `LatencyStatistics`
  (`avg_ms`, `p50_ms`, `p95_ms`, `max_ms`, `count`), and no observation is
  `not_applicable`.
- **Percentiles** interpolate at `position = (n - 1) * f`: `p50` and `p95` are exact for
  `n = 1, 2, 4, 5, 10`, and samples are sorted before slicing.
- **Accuracy** short-circuits on an exact expected/actual match (`Pass`, no model call),
  is `not_applicable` without a query or answer, and otherwise asks the judge.
- **Capability selection** is `not_applicable` for a case without capability calls;
  **composition effectiveness** is `not_applicable` below two calls. Both delegate
  everything else to the judge.
- **Description quality / classification consistency** fail deterministically before any
  model call when the card has no description, respectively no classification and no tags.
- **Window coercion**: `start` is inclusive, `end` is exclusive, and a case without
  `event_time` is dropped as soon as either boundary is set. Unbounded windows keep every
  case; an unparsable boundary or `event_time` is `Error::Validation`.
- **Aggregation** groups samples by metric name (`BTreeMap`, sorted), reduces each group
  with `aggregate_metric`, and reports `score = None`, `aggregation = "per_metric"`, and a
  `sample_count` that defaults to the largest group.
- **Status precedence** in `aggregate_metric`: a single sample is returned unchanged; with
  several, the score is the mean over scored samples and the status is `Pass` only when
  every scored sample passed, else `Fail`; without a scored sample the status is `Error`,
  then `Observed`, then `NotApplicable`.
- **Confidence** follows Python's thresholds: `0` → `None`, `1` → `Low`, `2..=9` →
  `Normal`, `>= 10` → `High`.
- **Fail-soft**: `evaluate` records a metric error as `"<metric>: <message>"` in
  `EvaluationReport::errors`; `evaluate_quality`, `evaluate_static` and `evaluate_case`
  materialise it as a `status = error` sample instead, so every metric stays visible.
- **`cache_signature`** is the lowercase hex SHA-256 of
  `{"metrics":["<name>:<scope>:<key>",…],"protocol":"symphony-evaluation-suite-v5"}`, where
  `<key>` is `Metric::cache_key()` or `builtin:<name>`; it is stable across runs, ignores
  JSON key order, and changes with any metric change.

## Deliberate differences

- `MetricScore::score` is `Option<f64>` (Python's `MetricResult.score` is `float | None`);
  `effective_score()` is the single accessor that decides whether a sample was scored.
- `MetricStatus::default()` is `Pass` — the neutral "this sample carries a verdict" status
  required by the additive migration — whereas Python's enum has no default; every metric
  therefore sets its status explicitly (`scored`/`failed`/`not_applicable`/`error`/`observed`).
- `CaseOutcome` is derived from the per-call `success` flags of an `EvaluationCase`; Python
  reads a case-level `success` field. A case whose calls are all undecided has no outcome.
- `cache_signature()` always returns a `String`; Python returns `None` for opaque
  extension metrics. Registration order is part of the key.
- `QualityResult::per_metric` is a sorted `BTreeMap` instead of an ordered mapping, so the
  serialized output is deterministic.
- `SYMPHONY_EVALUATION_PROTOCOL` is a Rust constant name; its value is Python's protocol
  string.
- The accuracy rubric is an English condensation of Python's long Chinese rubric; its
  decision boundary (factual error versus no usable answer) is preserved.
- No clock is read anywhere: an `event_time` always comes from the caller's case, so the
  accuracy reference time Python derives from `datetime.now()` does not exist.
- `sha256_hex` is reused from `crate::flow::model`; no new crypto dependency.
- `DescriptionQualityMetric`/`ClassificationConsistencyMetric` keep Python's guard step but
  always end in a verdict (deterministic fallback), so they are never `Error::Unsupported`.

## Constraints

- Only the shared contract lives in `mod.rs`; metric implementations stay in `metrics.rs`
  (static) and `trace_metrics.rs` (trace), and suite orchestration in `suite.rs`.
- Every public item is documented (`missing_docs` + clippy `-D warnings`); no `unwrap`,
  `expect`, `println!`, or `unsafe` outside tests.
- No hash-map iteration order or wall-clock value ever reaches an output; ties resolve on
  sorted metric names.
- Metrics never execute or install a capability and never perform I/O.
- `complete_capability()` is a `#[cfg(test)] pub(super)` fixture in `metrics.rs`, shared
  with the `trace_metrics` unit tests.

## Tests

- `src/evaluation/mod.rs`, `metrics.rs`, `trace_metrics.rs`, `suite.rs`, `trace.rs` carry
  the module-level unit tests (rubric weights, structural checks, guards, trace verdicts,
  aggregation precedence, confidence tiers, signature stability, window selection, and the
  percentile interpolation).
- `crates/openjiuwen-symphony/tests/evaluation.rs` is the behavior suite: default suite on
  a complete card, reusable `Arc<dyn Metric>` statics, structure conformance on good and
  bad cards, deterministic guards, success rate over mixed cases, latency as `Observed`,
  hand-computed percentile interpolation for `n = 1, 2, 4, 5, 10`, latency statistics
  ordering, window boundaries (start inclusive, end exclusive), invalid window/event-time
  errors, `aggregate_metric` precedence, confidence thresholds at `0/1/2/9/10`,
  `evaluate_quality` means, error samples, `cache_signature` stability and sensitivity,
  judge offline/model/error/deterministic paths, report round-trips, the static/trace
  split, and serde round-trips of every new type plus legacy `MetricScore`/`EvaluationInput`
  payloads.
- Everything is offline: scripted models or no model, no network, no clock reads.
