//! Evaluation seams, reports, and the deterministic evaluation driver.
//!
//! Ports Python `openjiuwen.agent_evolving.evaluator`:
//!
//! - [`Evaluator`] — turns a `(case, prediction)` pair into an
//!   [`EvaluatedCase`]; the provided [`batch_evaluate`](Evaluator::batch_evaluate)
//!   runs it with bounded concurrency.
//! - [`CaseRunner`] — produces the prediction for a case (see `runner` for the
//!   agent and map adapters).
//! - [`EvaluationReport`] — aggregate statistics over the evaluated cases.
//! - [`evaluate_cases`] — the end-to-end driver: run, score, and report.
//!
//! Metrics and their aggregation live in `metrics`.
//!
//! # Two scoring paths
//!
//! - **Live model judge** — [`LlmJudge`] sends
//!   [`build_judge_messages`] to a [`Model`](openjiuwen_core::foundation::llm::Model)
//!   and parses the reply into a [`JudgeVerdict`]; [`LlmJudgeEvaluator`] and
//!   [`crate::evaluator::DefaultEvaluator`] wrap that with a
//!   deterministic fallback metric. Only this path performs I/O, and every
//!   fallback is stated in the evaluated case's reason.
//! - **Deterministic metrics** — [`ExactMatchMetric`] and
//!   [`LlmAsJudgeMetric`] score a prediction that is already present, so a
//!   recorded verdict replays offline.
//!
//! # Determinism
//!
//! The report carries a caller-supplied `run_id` instead of generating one, so
//! evaluation never reads a UUID source or the wall clock; the same inputs always
//! produce the same report. This is a deliberate refinement of Python, which
//! reads `run_id` from ambient configuration.
//!
//! # Example
//!
//! ```
//! use std::collections::BTreeMap;
//! use std::sync::Arc;
//!
//! use openjiuwen_agent_evolving::dataset::Case;
//! use openjiuwen_agent_evolving::evaluator::{
//!     evaluate_cases, EvaluationConfig, ExactMatchMetric, MapCaseRunner, Metric, MetricEvaluator,
//! };
//! use openjiuwen_common::error::Result;
//! use serde_json::Value;
//! use tokio_util::sync::CancellationToken;
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     let mut inputs = BTreeMap::new();
//!     inputs.insert("question".to_owned(), Value::String("2 + 2?".to_owned()));
//!     let mut label = BTreeMap::new();
//!     label.insert("answer".to_owned(), Value::String("4".to_owned()));
//!     let case = Case::new(inputs, label)?.with_case_id("case-1");
//!
//!     let mut runner = MapCaseRunner::default();
//!     runner.insert("case-1", Value::String(" 4 ".to_owned()));
//!
//!     let metric: Arc<dyn Metric> = Arc::new(ExactMatchMetric::new());
//!     let evaluator = MetricEvaluator::new(vec![metric]);
//!     let report = evaluate_cases(
//!         "run-1",
//!         &runner,
//!         &evaluator,
//!         &[case],
//!         &CancellationToken::new(),
//!         &EvaluationConfig::default(),
//!     )
//!     .await?;
//!
//!     assert_eq!(report.num_cases, 1);
//!     assert!((report.mean_score - 1.0).abs() < f64::EPSILON);
//!     Ok(())
//! }
//! ```

mod judge;
mod metrics;
mod runner;

use std::collections::BTreeMap;

use async_trait::async_trait;
use futures::{StreamExt, TryStreamExt};
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

pub use judge::{
    build_judge_messages, JudgeCriterion, JudgeRubric, JudgeVerdict, LlmJudge, LlmJudgeEvaluator,
};
pub use metrics::{
    Aggregate, DefaultEvaluator, ExactMatchMetric, LlmAsJudgeMetric, Metric, MetricEvaluator,
    MetricValue,
};
pub use runner::{AgentCaseRunner, MapCaseRunner};

use crate::dataset::{Case, EvaluatedCase};

/// The minimum accepted `num_parallel` value, mirroring Python's
/// `TuneConstant.min_parallel_num`.
const MIN_PARALLEL: usize = 1;

/// The maximum accepted `num_parallel` value, mirroring Python's
/// `TuneConstant.max_parallel_num`.
const MAX_PARALLEL: usize = 20;

/// Scores a prediction against a case.
///
/// Python: `openjiuwen.agent_evolving.evaluator.evaluator.BaseEvaluator`.
#[async_trait]
pub trait Evaluator: Send + Sync {
    /// The evaluator name, used for diagnostics.
    fn name(&self) -> &str;

    /// Scores `prediction` for `case`.
    ///
    /// # Errors
    ///
    /// Returns an error when scoring cannot proceed (for example an unreachable
    /// judge model with no usable fallback) or when the produced score is
    /// invalid.
    async fn evaluate(&self, case: &Case, prediction: &Value) -> Result<EvaluatedCase>;

    /// Evaluates `cases` and `predictions` pairwise with bounded concurrency.
    ///
    /// Pairs stay aligned by index; `buffered` preserves the input order in the
    /// result.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when the two slices differ in length or when
    /// `num_parallel` is outside `1..=20`; otherwise the first scoring error.
    async fn batch_evaluate(
        &self,
        cases: &[Case],
        predictions: &[Value],
        num_parallel: usize,
    ) -> Result<Vec<EvaluatedCase>> {
        if cases.len() != predictions.len() {
            return Err(Error::validation(format!(
                "length of cases ({}) does not equal length of predictions ({})",
                cases.len(),
                predictions.len()
            )));
        }
        if !(MIN_PARALLEL..=MAX_PARALLEL).contains(&num_parallel) {
            return Err(Error::validation(format!(
                "num_parallel must be between {MIN_PARALLEL} and {MAX_PARALLEL}, got {num_parallel}"
            )));
        }

        // Drive the stream by index rather than by zipping the two slices: the
        // boxed futures returned by `async_trait` are not general enough over
        // two independent borrow lifetimes, and an index pair keeps the
        // case/prediction alignment explicit.
        futures::stream::iter(0..cases.len())
            .map(|index| async move { self.evaluate(&cases[index], &predictions[index]).await })
            .buffered(num_parallel)
            .try_collect()
            .await
    }
}

/// Produces the prediction an [`Evaluator`] scores.
///
/// This is the agent-execution seam: Python couples evaluation to the tuning
/// pipeline, while here any source of predictions (an agent, a fixed map, a
/// recorded baseline) implements [`CaseRunner`].
#[async_trait]
pub trait CaseRunner: Send + Sync {
    /// The runner name, used for diagnostics.
    fn name(&self) -> &str;

    /// Runs `case` and returns its prediction.
    ///
    /// # Errors
    ///
    /// [`Error::cancelled`] when `cancel` is already cancelled, or whatever the
    /// underlying runner fails with.
    async fn run_case(&self, case: &Case, cancel: &CancellationToken) -> Result<Value>;
}

/// Aggregate statistics over a set of evaluated cases.
///
/// Python returns the evaluated list and leaves aggregation to callers; the
/// report bundles the same data with the mean/min/max scores and the per-metric
/// means so the trainer and the service can share one shape.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct EvaluationReport {
    /// Caller-supplied identifier for this evaluation run.
    pub run_id: String,
    /// Number of evaluated cases.
    pub num_cases: usize,
    /// Mean score across all cases; `0.0` when there are none.
    pub mean_score: f64,
    /// Lowest score; `0.0` when there are no cases.
    pub min_score: f64,
    /// Highest score; `0.0` when there are no cases.
    pub max_score: f64,
    /// Mean of every per-metric key, over the cases that carry that key.
    pub per_metric: BTreeMap<String, f64>,
    /// The evaluated cases, in input order.
    pub cases: Vec<EvaluatedCase>,
}

impl EvaluationReport {
    /// Builds a report from already-evaluated cases.
    ///
    /// Scores are aggregated over every case; a per-metric key's mean ignores
    /// cases that do not carry it. An empty input yields a zeroed report.
    #[must_use]
    pub fn from_evaluated(run_id: impl Into<String>, cases: Vec<EvaluatedCase>) -> Self {
        let num_cases = cases.len();
        let (mean_score, min_score, max_score) = if cases.is_empty() {
            (0.0, 0.0, 0.0)
        } else {
            let sum: f64 = cases.iter().map(|case| case.score).sum();
            let min = cases
                .iter()
                .map(|case| case.score)
                .fold(f64::INFINITY, f64::min);
            let max = cases
                .iter()
                .map(|case| case.score)
                .fold(f64::NEG_INFINITY, f64::max);
            (sum / num_cases as f64, min, max)
        };

        let mut totals: BTreeMap<String, (f64, usize)> = BTreeMap::new();
        for case in &cases {
            if let Some(per_metric) = &case.per_metric {
                for (name, value) in per_metric {
                    let entry = totals.entry(name.clone()).or_insert((0.0, 0));
                    entry.0 += value;
                    entry.1 += 1;
                }
            }
        }
        let per_metric = totals
            .into_iter()
            .map(|(name, (sum, count))| (name, sum / count as f64))
            .collect();

        Self {
            run_id: run_id.into(),
            num_cases,
            mean_score,
            min_score,
            max_score,
            per_metric,
            cases,
        }
    }

    /// Builds a zero-score report for cases that were not evaluated.
    #[must_use]
    pub fn from_cases(run_id: impl Into<String>, cases: Vec<Case>) -> Self {
        Self::from_evaluated(run_id, cases.into_iter().map(EvaluatedCase::from).collect())
    }
}

/// Bounds for [`evaluate_cases`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct EvaluationConfig {
    /// Number of cases evaluated concurrently; must be in `1..=20`.
    pub num_parallel: usize,
    /// Optional cap on the number of cases evaluated.
    pub max_cases: Option<usize>,
}

impl Default for EvaluationConfig {
    fn default() -> Self {
        Self {
            num_parallel: 1,
            max_cases: None,
        }
    }
}

/// Runs `runner`, scores every case with `evaluator`, and aggregates the result.
///
/// The cases are truncated to `config.max_cases` before anything runs; the
/// cancellation token is checked both before each case starts and by runners
/// themselves.
///
/// `run_id` is supplied by the caller (rather than generated) so evaluation
/// stays deterministic — no UUID source and no wall clock enter the report.
///
/// # Errors
///
/// [`Error::validation`] when `config.num_parallel` is outside `1..=20`,
/// [`Error::cancelled`] when the token is already cancelled, or the first runner
/// or evaluator error.
pub async fn evaluate_cases(
    run_id: &str,
    runner: &dyn CaseRunner,
    evaluator: &dyn Evaluator,
    cases: &[Case],
    cancel: &CancellationToken,
    config: &EvaluationConfig,
) -> Result<EvaluationReport> {
    if !(MIN_PARALLEL..=MAX_PARALLEL).contains(&config.num_parallel) {
        return Err(Error::validation(format!(
            "num_parallel must be between {MIN_PARALLEL} and {MAX_PARALLEL}, got {}",
            config.num_parallel
        )));
    }

    // The selected cases are cloned so the per-case futures own their `Case`
    // instead of borrowing the caller's slice; that keeps the returned future
    // `Send`, which axum handlers and `tokio::spawn` require. The clone is
    // bounded by `config.max_cases`.
    let selected: Vec<Case> = cases
        .iter()
        .take(config.max_cases.unwrap_or(usize::MAX))
        .cloned()
        .collect();

    let predictions: Vec<Value> = futures::stream::iter(selected.clone())
        .map(|case| async move {
            if cancel.is_cancelled() {
                Err(Error::cancelled())
            } else {
                runner.run_case(&case, cancel).await
            }
        })
        .buffered(config.num_parallel)
        .try_collect()
        .await?;

    let evaluated = evaluator
        .batch_evaluate(&selected, &predictions, config.num_parallel)
        .await?;

    Ok(EvaluationReport::from_evaluated(run_id, evaluated))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(key: &str, value: Value) -> BTreeMap<String, Value> {
        let mut map = BTreeMap::new();
        map.insert(key.to_owned(), value);
        map
    }

    fn sample_case(id: &str, label: &str) -> Case {
        Case::new(
            map("question", Value::String("q".to_owned())),
            map("answer", Value::String(label.to_owned())),
        )
        .unwrap()
        .with_case_id(id)
    }

    fn close(actual: f64, expected: f64) -> bool {
        (actual - expected).abs() < 1e-9
    }

    /// Scores 1.0 when the projected answer matches, and records an
    /// `accuracy` metric for the report tests.
    struct AnswerEvaluator;

    #[async_trait]
    impl Evaluator for AnswerEvaluator {
        fn name(&self) -> &'static str {
            "answer"
        }

        async fn evaluate(&self, case: &Case, prediction: &Value) -> Result<EvaluatedCase> {
            let expected = case.label.get("answer").cloned().unwrap_or(Value::Null);
            let predicted = match prediction {
                Value::Object(object) => object
                    .get("answer")
                    .cloned()
                    .unwrap_or_else(|| prediction.clone()),
                other => other.clone(),
            };
            let score = if expected == predicted { 1.0 } else { 0.0 };
            let mut evaluated = EvaluatedCase::new(case.clone(), None, score, "answer")?;
            let mut per_metric = BTreeMap::new();
            per_metric.insert("accuracy".to_owned(), score);
            evaluated.per_metric = Some(per_metric);
            Ok(evaluated)
        }
    }

    #[tokio::test]
    async fn batch_evaluate_keeps_index_pairing() {
        let cases = [sample_case("case-1", "a"), sample_case("case-2", "b")];
        let predictions = [
            serde_json::json!({"answer": "a"}),
            serde_json::json!({"answer": "x"}),
        ];
        let evaluated = AnswerEvaluator
            .batch_evaluate(&cases, &predictions, 2)
            .await
            .unwrap();
        assert_eq!(evaluated.len(), 2);
        assert!(close(evaluated[0].score, 1.0));
        assert!(close(evaluated[1].score, 0.0));
        assert_eq!(evaluated[0].case_id(), "case-1");
        assert_eq!(evaluated[1].case_id(), "case-2");
    }

    #[tokio::test]
    async fn batch_evaluate_validates_lengths_and_parallelism() {
        let cases = [sample_case("case-1", "a")];
        let predictions: Vec<Value> = Vec::new();

        let error = AnswerEvaluator
            .batch_evaluate(&cases, &predictions, 1)
            .await
            .unwrap_err();
        assert!(
            matches!(&error, Error::Validation(message)
                if message.contains("cases (1)") && message.contains("predictions (0)")),
            "unexpected error: {error}"
        );

        for num_parallel in [0, 21] {
            let error = AnswerEvaluator
                .batch_evaluate(&[sample_case("case-1", "a")], &[Value::Null], num_parallel)
                .await
                .unwrap_err();
            assert!(
                matches!(&error, Error::Validation(message)
                    if message.contains('1') && message.contains("20")),
                "unexpected error for {num_parallel}: {error}"
            );
        }
    }

    #[test]
    fn report_aggregates_scores_and_per_metric_means() {
        let mut first = EvaluatedCase::new(sample_case("case-1", "a"), None, 1.0, "first").unwrap();
        first.per_metric = Some(BTreeMap::from([
            ("accuracy".to_owned(), 1.0),
            ("latency".to_owned(), 0.2),
        ]));
        let mut second =
            EvaluatedCase::new(sample_case("case-2", "b"), None, 0.5, "second").unwrap();
        second.per_metric = Some(BTreeMap::from([
            ("accuracy".to_owned(), 0.5),
            ("latency".to_owned(), 0.4),
        ]));
        let third = EvaluatedCase::new(sample_case("case-3", "c"), None, 0.25, "third").unwrap();

        let report = EvaluationReport::from_evaluated("run-1", vec![first, second, third]);
        assert_eq!(report.run_id, "run-1");
        assert_eq!(report.num_cases, 3);
        assert!(close(report.mean_score, 1.75 / 3.0));
        assert!(close(report.min_score, 0.25));
        assert!(close(report.max_score, 1.0));
        assert_eq!(report.per_metric.len(), 2);
        assert!(close(report.per_metric["accuracy"], 1.5 / 2.0));
        assert!(close(report.per_metric["latency"], 0.3));
    }

    #[test]
    fn empty_report_is_zeroed() {
        let report = EvaluationReport::from_evaluated("run-1", Vec::new());
        assert_eq!(report.num_cases, 0);
        assert!(close(report.mean_score, 0.0));
        assert!(close(report.min_score, 0.0));
        assert!(close(report.max_score, 0.0));
        assert!(report.per_metric.is_empty());
        assert_eq!(report.cases.len(), 0);
    }

    #[test]
    fn from_cases_wraps_with_zero_scores() {
        let report = EvaluationReport::from_cases(
            "run-1",
            vec![sample_case("case-1", "a"), sample_case("case-2", "b")],
        );
        assert_eq!(report.num_cases, 2);
        assert!(close(report.mean_score, 0.0));
        assert!(report.per_metric.is_empty());
        assert_eq!(report.cases[0].case_id(), "case-1");
    }

    #[tokio::test]
    async fn evaluate_cases_runs_scores_and_reports() {
        let cases = [sample_case("case-1", "a"), sample_case("case-2", "b")];
        let mut runner = MapCaseRunner::default();
        runner.insert("case-1", serde_json::json!({"answer": "a"}));
        runner.insert("case-2", serde_json::json!({"answer": "b"}));

        let report = evaluate_cases(
            "run-1",
            &runner,
            &AnswerEvaluator,
            &cases,
            &CancellationToken::new(),
            &EvaluationConfig {
                num_parallel: 2,
                max_cases: None,
            },
        )
        .await
        .unwrap();

        assert_eq!(report.num_cases, 2);
        assert!(close(report.mean_score, 1.0));
        assert_eq!(report.cases[0].case_id(), "case-1");
        assert_eq!(report.cases[1].case_id(), "case-2");
    }

    /// The driver's future must be `Send` so axum handlers and `tokio::spawn`
    /// can hold it. This is a compile-time check: the future is never awaited,
    /// only passed to a `Send`-bounded helper.
    #[test]
    fn evaluate_cases_future_is_send() {
        fn assert_send<T: Send>(value: T) -> T {
            value
        }

        let cases = [sample_case("case-1", "a")];
        let runner = MapCaseRunner::default();
        let cancel = CancellationToken::new();
        let config = EvaluationConfig::default();

        let future = evaluate_cases("run-1", &runner, &AnswerEvaluator, &cases, &cancel, &config);
        let _future = assert_send(future);
    }

    /// The same bound, exercised at runtime: the whole evaluation runs inside a
    /// spawned (and therefore `Send + 'static`) task.
    #[tokio::test]
    async fn evaluate_cases_runs_inside_a_spawned_task() {
        let report = tokio::spawn(async move {
            let cases = [sample_case("case-1", "a")];
            let mut runner = MapCaseRunner::default();
            runner.insert("case-1", serde_json::json!({"answer": "a"}));
            evaluate_cases(
                "run-1",
                &runner,
                &AnswerEvaluator,
                &cases,
                &CancellationToken::new(),
                &EvaluationConfig::default(),
            )
            .await
        })
        .await
        .expect("spawned evaluation does not panic")
        .expect("spawned evaluation succeeds");

        assert_eq!(report.num_cases, 1);
        assert!(close(report.mean_score, 1.0));
    }

    #[tokio::test]
    async fn evaluate_cases_truncates_to_max_cases() {
        let cases = [
            sample_case("case-1", "a"),
            sample_case("case-2", "b"),
            sample_case("case-3", "c"),
        ];
        let mut runner = MapCaseRunner::default();
        for id in ["case-1", "case-2", "case-3"] {
            runner.insert(id, Value::Null);
        }

        let report = evaluate_cases(
            "run-1",
            &runner,
            &AnswerEvaluator,
            &cases,
            &CancellationToken::new(),
            &EvaluationConfig {
                num_parallel: 1,
                max_cases: Some(2),
            },
        )
        .await
        .unwrap();

        assert_eq!(report.num_cases, 2);
        assert_eq!(report.cases[1].case_id(), "case-2");
    }

    #[tokio::test]
    async fn evaluate_cases_honours_cancellation_and_validates_parallelism() {
        let cases = [sample_case("case-1", "a")];
        let runner = MapCaseRunner::default();

        let cancel = CancellationToken::new();
        cancel.cancel();
        let error = evaluate_cases(
            "run-1",
            &runner,
            &AnswerEvaluator,
            &cases,
            &cancel,
            &EvaluationConfig::default(),
        )
        .await
        .unwrap_err();
        assert!(error.is_cancelled(), "unexpected error: {error}");

        let error = evaluate_cases(
            "run-1",
            &runner,
            &AnswerEvaluator,
            &cases,
            &CancellationToken::new(),
            &EvaluationConfig {
                num_parallel: 21,
                max_cases: None,
            },
        )
        .await
        .unwrap_err();
        assert!(
            matches!(&error, Error::Validation(message) if message.contains("num_parallel")),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn config_serde_defaults_and_tolerates_unknown_fields() {
        let config = EvaluationConfig::default();
        assert_eq!(config.num_parallel, 1);
        assert_eq!(config.max_cases, None);

        let restored: EvaluationConfig =
            serde_json::from_value(serde_json::json!({"num_parallel": 4, "unknown": true}))
                .unwrap();
        assert_eq!(restored.num_parallel, 4);
        assert_eq!(restored.max_cases, None);
    }

    #[test]
    fn report_serde_uses_python_field_names() {
        let report = EvaluationReport::from_cases("run-1", vec![sample_case("case-1", "a")]);
        let value = serde_json::to_value(&report).unwrap();
        let object = value.as_object().unwrap();
        for key in [
            "run_id",
            "num_cases",
            "mean_score",
            "min_score",
            "max_score",
            "per_metric",
            "cases",
        ] {
            assert!(object.contains_key(key), "missing field {key}");
        }
        let restored: EvaluationReport = serde_json::from_value(value).unwrap();
        assert_eq!(restored, report);
    }
}
