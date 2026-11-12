#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Integration tests for `openjiuwen_agent_evolving::evaluator`.
//!
//! Covers the deterministic scoring surface: `ExactMatchMetric` normalization
//! and label/prediction projections, metric aggregation (`Mean`/`First`,
//! non-finite coercion), the deferred `LlmAsJudgeMetric`, the default
//! `Evaluator::batch_evaluate`, `EvaluationReport` math, `MapCaseRunner`, the
//! `evaluate_cases` driver, and `AgentCaseRunner` over a fake `Agent`.
//!
//! Everything here is offline and deterministic: no network, no environment
//! reads, no clock, and no dependence on execution order.
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::StreamExt;
use openjiuwen_agent_evolving::dataset::{Case, EvaluatedCase};
use openjiuwen_agent_evolving::evaluator::*;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_core::session::Session;
use openjiuwen_core::single_agent::{Agent, AgentCard, AgentChunk, AgentOutput, ResultType};
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

/// A one-entry map from an owned JSON value, used for labels and inputs.
fn map(key: &str, value: Value) -> BTreeMap<String, Value> {
    BTreeMap::from([(key.to_owned(), value)])
}
/// A one-entry map holding a string, the common label shape.
fn text_map(key: &str, value: &str) -> BTreeMap<String, Value> {
    map(key, text(value))
}
/// A string prediction.
fn text(value: &str) -> Value {
    Value::String(value.to_owned())
}
/// A valid case with fixed inputs and the given label.
fn case(id: &str, label: BTreeMap<String, Value>) -> Case {
    Case::new(map("question", json!("2 + 2?")), label)
        .unwrap()
        .with_case_id(id)
}
/// A valid case whose label is `{"answer": answer}`.
fn answer_case(id: &str, answer: &str) -> Case {
    case(id, text_map("answer", answer))
}
/// Compares scores without relying on exact float equality.
fn close(actual: f64, expected: f64) -> bool {
    (actual - expected).abs() < 1e-9
}
/// Whether an exact-match metric considers `prediction` a match for `label`.
fn is_match(
    metric: &ExactMatchMetric,
    prediction: &Value,
    label: &BTreeMap<String, Value>,
) -> bool {
    match metric.compute(prediction, label).unwrap() {
        MetricValue::Scalar(value) => close(value, 1.0),
        MetricValue::Named(named) => panic!("expected a scalar, got {named:?}"),
    }
}
/// Asserts an error is a validation failure mentioning `needle`.
fn assert_validation(error: &Error, needle: &str) {
    assert!(
        matches!(error, Error::Validation(message) if message.contains(needle)),
        "unexpected error: {error}"
    );
}
/// Runs a metric evaluator over one case, unwrapping the result.
async fn run(evaluator: &MetricEvaluator, case: &Case, prediction: &Value) -> EvaluatedCase {
    evaluator.evaluate(case, prediction).await.unwrap()
}
/// An evaluation config with the given bounds.
fn config(num_parallel: usize, max_cases: Option<usize>) -> EvaluationConfig {
    EvaluationConfig {
        num_parallel,
        max_cases,
    }
}

/// One table row: metric, prediction, label, expected match, description.
type Row<'a> = (
    &'a ExactMatchMetric,
    Value,
    &'a BTreeMap<String, Value>,
    bool,
    &'a str,
);

/// Asserts every `(metric, prediction, label, expected, description)` row.
fn assert_rows(rows: Vec<Row<'_>>) {
    for (metric, prediction, label, expected, description) in rows {
        assert_eq!(
            is_match(metric, &prediction, label),
            expected,
            "case: {description}, prediction: {prediction}"
        );
    }
}

/// Whitespace, case, and unicode normalization (`new()`) vs raw comparison.
#[test]
fn exact_match_normalizes_whitespace_case_and_unicode() {
    let metric = ExactMatchMetric::new();
    let raw = ExactMatchMetric::new().raw();
    assert!(metric.normalize());
    assert!(!raw.normalize());
    assert_eq!(metric.name(), "exact_match");

    let spaced = text_map("answer", "  Hello   World ");
    let exact = text_map("answer", "hello world");
    let unicode = text_map("answer", "Héllo");
    assert_rows(vec![
        (
            &metric,
            text("hello world"),
            &spaced,
            true,
            "normalized case/space",
        ),
        (
            &metric,
            text("  Hello   World "),
            &spaced,
            true,
            "already spaced",
        ),
        (&raw, text("hello world"), &spaced, false, "raw keeps case"),
        (&raw, text("  Hello   World "), &spaced, true, "raw exact"),
        (
            &metric,
            text("hello world"),
            &exact,
            true,
            "normalized exact",
        ),
        (&metric, text("héllo"), &unicode, true, "unicode lowercase"),
        (
            &raw,
            text("héllo"),
            &unicode,
            false,
            "raw keeps unicode case",
        ),
    ]);
}

/// Label projections (`answer`, single-key, full object), prediction shapes,
/// and JSON equality for non-string values.
#[test]
fn exact_match_projects_labels_predictions_and_non_strings() {
    let metric = ExactMatchMetric::new();
    let number = map("answer", json!(4));
    let answer = text_map("answer", "4");
    let single = text_map("result", "4");
    let full = BTreeMap::from([("a".to_owned(), json!(1)), ("b".to_owned(), json!(2))]);
    let null = map("answer", Value::Null);
    let array = map("answer", json!([1, 2]));
    let nested = json!({"nested": [1], "other": 2});
    let object = map("answer", nested.clone());
    assert_rows(vec![
        (&metric, json!(4), &number, true, "numeric equality"),
        (&metric, json!(2), &number, false, "numeric mismatch"),
        (&metric, text("4"), &number, false, "string vs number"),
        (&metric, text("4"), &answer, true, "plain string"),
        (
            &metric,
            json!({"answer": "4"}),
            &answer,
            true,
            "answer-key object",
        ),
        (
            &metric,
            json!({"x": "4"}),
            &answer,
            true,
            "single-entry object",
        ),
        (&metric, text("5"), &answer, false, "plain string mismatch"),
        (
            &metric,
            json!({"result": "4"}),
            &single,
            true,
            "single-key label",
        ),
        (&metric, text("4"), &single, true, "single-key label, plain"),
        (
            &metric,
            text("5"),
            &single,
            false,
            "single-key label mismatch",
        ),
        (
            &metric,
            json!({"a": 1, "b": 2}),
            &full,
            true,
            "full-object label",
        ),
        (
            &metric,
            json!({"a": 1, "b": 3}),
            &full,
            false,
            "full mismatch",
        ),
        (&metric, Value::Null, &null, true, "null equality"),
        (&metric, json!([1, 2]), &array, true, "array equality"),
        (&metric, json!([1, 3]), &array, false, "array mismatch"),
        (&metric, nested.clone(), &object, true, "object equality"),
    ]);
}

/// A metric returning a fixed value: a scalar under `name`, or named entries.
struct FixedMetric {
    name: &'static str,
    value: MetricValue,
}

impl Metric for FixedMetric {
    fn name(&self) -> &str {
        self.name
    }
    fn compute(
        &self,
        _prediction: &Value,
        _label: &BTreeMap<String, Value>,
    ) -> Result<MetricValue> {
        Ok(self.value.clone())
    }
}

/// A metric reporting `value` as a scalar under `name`.
fn scalar(name: &'static str, value: f64) -> Arc<dyn Metric> {
    Arc::new(FixedMetric {
        name,
        value: MetricValue::Scalar(value),
    })
}

/// A metric reporting named values inserted out of order (`b` before `a`);
/// `BTreeMap` re-sorts them, so aggregation sees `a` then `b`.
fn weighted() -> Arc<dyn Metric> {
    let mut named = BTreeMap::new();
    named.insert("b".to_owned(), 0.4);
    named.insert("a".to_owned(), 0.6);
    Arc::new(FixedMetric {
        name: "weighted",
        value: MetricValue::Named(named),
    })
}

/// The ordered metric list shared by the aggregation tests.
fn mixed_metrics() -> Vec<Arc<dyn Metric>> {
    vec![scalar("accuracy", 0.2), weighted()]
}

#[tokio::test]
async fn mean_and_first_aggregate_scalar_and_named_values_in_order() {
    let evaluator = MetricEvaluator::new(mixed_metrics());
    assert_eq!(evaluator.name(), "metric");
    assert_eq!(evaluator.aggregate(), Aggregate::Mean);
    assert_eq!(evaluator.metrics().len(), 2);
    let mean = run(
        &evaluator,
        &answer_case("case-1", "4"),
        &json!({"answer": "4"}),
    )
    .await;
    // Values in ordered order: 0.2 (scalar), then a=0.6, b=0.4.
    assert!(close(mean.score, 0.4), "score: {}", mean.score);
    let per_metric = mean.per_metric.clone().expect("per-metric breakdown");
    let keys: Vec<&str> = per_metric.keys().map(String::as_str).collect();
    assert_eq!(keys, ["a", "accuracy", "b"]);
    assert!(close(per_metric["a"], 0.6));
    assert!(close(per_metric["b"], 0.4));
    assert!(close(per_metric["accuracy"], 0.2));
    assert_eq!(mean.reason, "2 metric(s)");

    let first = run(
        &MetricEvaluator::new(mixed_metrics()).with_aggregate(Aggregate::First),
        &answer_case("case-1", "4"),
        &Value::Null,
    )
    .await;
    assert!(close(first.score, 0.2), "score: {}", first.score);
    assert_eq!(
        first.per_metric.expect("breakdown").len(),
        3,
        "every value is still recorded"
    );
}

#[tokio::test]
async fn non_finite_and_empty_metrics_do_not_error() {
    let coerced = run(
        &MetricEvaluator::new(vec![scalar("nan", f64::NAN), scalar("one", 1.0)]),
        &answer_case("case-1", "4"),
        &Value::Null,
    )
    .await;
    assert!(close(coerced.score, 0.5), "score: {}", coerced.score);
    let per_metric = coerced.per_metric.expect("per-metric breakdown");
    assert!(close(per_metric["nan"], 0.0));
    assert!(close(per_metric["one"], 1.0));

    let empty = run(
        &MetricEvaluator::new(Vec::new()),
        &answer_case("case-1", "4"),
        &Value::Null,
    )
    .await;
    assert!(close(empty.score, 0.0));
    assert!(empty.per_metric.is_none());
    assert_eq!(empty.reason, "0 metric(s)");
}

#[test]
fn llm_as_judge_is_deferred() {
    let metric = LlmAsJudgeMetric::new();
    assert_eq!(metric.name(), "llm_as_judge");
    assert!(metric.higher_is_better());
    let error = metric
        .compute(&text("4"), &text_map("answer", "4"))
        .unwrap_err();
    assert!(
        matches!(&error, Error::Unsupported(message) if message.contains("LlmAsJudgeMetric")),
        "unexpected error: {error}"
    );
}

/// Scores with the prediction's numeric value; `EvaluatedCase::new` clamps it.
struct ConstantEvaluator;

#[async_trait]
impl Evaluator for ConstantEvaluator {
    fn name(&self) -> &'static str {
        "constant"
    }
    async fn evaluate(&self, case: &Case, prediction: &Value) -> Result<EvaluatedCase> {
        let score = prediction.as_f64().unwrap_or(0.0);
        EvaluatedCase::new(case.clone(), None, score, "constant")
    }
}

#[tokio::test]
async fn batch_evaluate_validates_inputs_and_keeps_order() {
    let cases = [answer_case("case-1", "a")];
    let error = ConstantEvaluator
        .batch_evaluate(&cases, &[], 1)
        .await
        .unwrap_err();
    assert_validation(&error, "length of cases");

    let two = [answer_case("case-1", "a"), answer_case("case-2", "b")];
    let error = ConstantEvaluator
        .batch_evaluate(&two, &[json!(0.5)], 1)
        .await
        .unwrap_err();
    assert_validation(&error, "length of cases");

    for num_parallel in [0_usize, 21] {
        let error = ConstantEvaluator
            .batch_evaluate(&cases, &[json!(0.5)], num_parallel)
            .await
            .unwrap_err();
        assert_validation(&error, "num_parallel");
    }

    let three = [
        answer_case("case-1", "a"),
        answer_case("case-2", "b"),
        answer_case("case-3", "c"),
    ];
    let evaluated = ConstantEvaluator
        .batch_evaluate(&three, &[json!(0.25), json!(0.5), json!(1.0)], 2)
        .await
        .unwrap();
    let ids: Vec<&str> = evaluated.iter().map(EvaluatedCase::case_id).collect();
    assert_eq!(ids, ["case-1", "case-2", "case-3"]);
    for (case, expected) in evaluated.iter().zip([0.25, 0.5, 1.0]) {
        assert!(close(case.score, expected), "score: {}", case.score);
    }
}

/// An evaluated case with an explicit score and optional per-metric map.
fn evaluated_case(
    id: &str,
    score: f64,
    per_metric: Option<BTreeMap<String, f64>>,
) -> EvaluatedCase {
    let mut evaluated = EvaluatedCase::new(answer_case(id, "a"), None, score, "scored").unwrap();
    evaluated.per_metric = per_metric;
    evaluated
}

#[test]
fn report_aggregates_scores_and_per_metric_means() {
    let cases = vec![
        evaluated_case(
            "case-1",
            0.25,
            Some(BTreeMap::from([("a".to_owned(), 0.5)])),
        ),
        evaluated_case(
            "case-2",
            0.5,
            Some(BTreeMap::from([
                ("a".to_owned(), 1.0),
                ("b".to_owned(), 0.25),
            ])),
        ),
        evaluated_case("case-3", 1.0, None),
    ];
    let report = EvaluationReport::from_evaluated("run-1", cases);
    assert_eq!(report.run_id, "run-1");
    assert_eq!(report.num_cases, 3);
    assert!(close(report.mean_score, 1.75 / 3.0));
    assert!(close(report.min_score, 0.25));
    assert!(close(report.max_score, 1.0));
    // Metric "a" is averaged over the two cases that carry it.
    assert_eq!(report.per_metric.len(), 2);
    assert!(close(report.per_metric["a"], 0.75));
    assert!(close(report.per_metric["b"], 0.25));
}

#[test]
fn report_zeroes_empty_and_unevaluated_inputs_and_round_trips() {
    let empty = EvaluationReport::from_evaluated("run-1", Vec::new());
    assert_eq!(empty.num_cases, 0);
    assert!(close(empty.mean_score, 0.0));
    assert!(close(empty.min_score, 0.0));
    assert!(close(empty.max_score, 0.0));
    assert!(empty.per_metric.is_empty());
    assert_eq!(empty.cases.len(), 0);

    let report = EvaluationReport::from_cases(
        "run-2",
        vec![answer_case("case-1", "a"), answer_case("case-2", "b")],
    );
    assert_eq!(report.num_cases, 2);
    assert!(close(report.mean_score, 0.0));
    assert!(report.cases.iter().all(|case| close(case.score, 0.0)));
    assert_eq!(report.cases[0].case_id(), "case-1");

    let scored = EvaluationReport::from_evaluated(
        "run-3",
        vec![evaluated_case(
            "case-1",
            0.5,
            Some(BTreeMap::from([("a".to_owned(), 0.5)])),
        )],
    );
    let value = serde_json::to_value(&scored).unwrap();
    assert_eq!(value["run_id"], json!("run-3"));
    assert_eq!(value["cases"][0]["case"]["case_id"], json!("case-1"));
    let restored: EvaluationReport = serde_json::from_value(value).unwrap();
    assert_eq!(restored, scored);
}

#[tokio::test]
async fn map_runner_replays_predictions_by_case_id() {
    let mut runner = MapCaseRunner::default();
    runner.insert("case-1", json!({"answer": "4"}));
    assert_eq!(runner.name(), "map");
    let value = runner
        .run_case(&answer_case("case-1", "4"), &CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(value, json!({"answer": "4"}));

    let missing = answer_case("case-2", "4");
    let error = runner
        .run_case(&missing, &CancellationToken::new())
        .await
        .unwrap_err();
    assert!(
        matches!(&error, Error::NotFound(message) if message.contains("case-2")),
        "unexpected error: {error}"
    );

    let cancel = CancellationToken::new();
    cancel.cancel();
    let error = runner
        .run_case(&answer_case("case-1", "4"), &cancel)
        .await
        .unwrap_err();
    assert!(error.is_cancelled(), "unexpected error: {error}");
}

#[tokio::test]
async fn evaluate_cases_truncates_reports_run_id_and_honours_cancellation() {
    let cases = [
        answer_case("case-1", "a"),
        answer_case("case-2", "b"),
        answer_case("case-3", "c"),
    ];
    let mut runner = MapCaseRunner::default();
    runner.insert("case-1", json!(0.25));
    runner.insert("case-2", json!(0.75));
    runner.insert("case-3", json!(1.0));

    let report = evaluate_cases(
        "run-42",
        &runner,
        &ConstantEvaluator,
        &cases,
        &CancellationToken::new(),
        &config(2, Some(2)),
    )
    .await
    .unwrap();
    assert_eq!(report.run_id, "run-42");
    assert_eq!(report.num_cases, 2);
    assert_eq!(report.cases[0].case_id(), "case-1");
    assert_eq!(report.cases[1].case_id(), "case-2");
    assert!(close(report.mean_score, 0.5));

    let cancel = CancellationToken::new();
    cancel.cancel();
    let error = evaluate_cases(
        "run-1",
        &runner,
        &ConstantEvaluator,
        &cases,
        &cancel,
        &EvaluationConfig::default(),
    )
    .await
    .unwrap_err();
    assert!(error.is_cancelled(), "unexpected error: {error}");

    let empty = evaluate_cases(
        "run-1",
        &runner,
        &ConstantEvaluator,
        &cases,
        &CancellationToken::new(),
        &config(1, Some(0)),
    )
    .await
    .unwrap();
    assert_eq!(empty.run_id, "run-1");
    assert_eq!(empty.num_cases, 0);
    assert_eq!(empty.cases.len(), 0);
    assert!(close(empty.mean_score, 0.0));
}

/// Records the inputs it receives and answers with a fixed output.
struct RecordingAgent {
    card: AgentCard,
    output: AgentOutput,
    seen_inputs: Mutex<Option<Value>>,
    invocations: AtomicUsize,
}

impl RecordingAgent {
    fn new(answer: &str) -> Self {
        Self {
            card: AgentCard::new("fake").with_id("fake-agent"),
            output: AgentOutput {
                output: answer.to_owned(),
                result_type: ResultType::Answer,
                ..AgentOutput::default()
            },
            seen_inputs: Mutex::new(None),
            invocations: AtomicUsize::new(0),
        }
    }

    fn invocations(&self) -> usize {
        self.invocations.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl Agent for RecordingAgent {
    fn card(&self) -> &AgentCard {
        &self.card
    }
    async fn invoke(
        &self,
        inputs: Value,
        _session: Option<Arc<dyn Session>>,
    ) -> Result<AgentOutput> {
        self.invocations.fetch_add(1, Ordering::SeqCst);
        *self.seen_inputs.lock().unwrap() = Some(inputs);
        Ok(self.output.clone())
    }
    fn stream(
        &self,
        _inputs: Value,
        _session: Option<Arc<dyn Session>>,
    ) -> BoxStream<'static, Result<AgentChunk>> {
        futures::stream::empty().boxed()
    }
}

#[tokio::test]
async fn agent_runner_feeds_case_inputs_and_scores_cancellation() {
    let agent = Arc::new(RecordingAgent::new("4"));
    let runner = AgentCaseRunner::new(agent.clone());
    assert_eq!(runner.name(), "agent");
    assert!(runner.session().is_none());
    assert_eq!(runner.agent().card().id, "fake-agent");

    let case = answer_case("case-1", "4");
    let prediction = runner
        .run_case(&case, &CancellationToken::new())
        .await
        .unwrap();
    // The case inputs are forwarded verbatim as a JSON object.
    let seen = agent.seen_inputs.lock().unwrap().clone().expect("invoked");
    assert_eq!(seen, json!({"question": "2 + 2?"}));
    assert_eq!(agent.invocations(), 1);
    // The serialized output carries the answer under `output`.
    assert_eq!(prediction["output"], json!("4"));
    assert_eq!(prediction["result_type"], json!("answer"));

    // The label projects to `"4"`, so the agent's answer is an exact match.
    let evaluator = MetricEvaluator::new(vec![Arc::new(ExactMatchMetric::new())]);
    let evaluated = evaluator
        .evaluate(&case, &prediction["output"])
        .await
        .unwrap();
    assert!(close(evaluated.score, 1.0), "score: {}", evaluated.score);
    let per_metric = evaluated.per_metric.expect("per-metric breakdown");
    assert!(close(per_metric["exact_match"], 1.0));

    // A pre-cancelled token short-circuits before the agent runs.
    let cancel = CancellationToken::new();
    cancel.cancel();
    let error = runner.run_case(&case, &cancel).await.unwrap_err();
    assert!(error.is_cancelled(), "unexpected error: {error}");
    assert_eq!(agent.invocations(), 1);
}

#[tokio::test]
async fn default_evaluator_is_reachable_from_the_crate_root() {
    // The crate root re-exports the primary evaluator types.
    let evaluator = openjiuwen_agent_evolving::DefaultEvaluator::new("accuracy");
    assert_eq!(evaluator.name(), "default");
    assert_eq!(evaluator.metric(), "accuracy");
    assert_eq!(
        openjiuwen_agent_evolving::CRATE_NAME,
        "openjiuwen-agent-evolving"
    );

    let error = evaluator
        .evaluate(&answer_case("case-1", "4"), &Value::Null)
        .await
        .unwrap_err();
    assert!(
        matches!(&error, Error::Unsupported(message)
            if message.contains("DefaultEvaluator") && message.contains("docs/PARITY.md")),
        "unexpected error: {error}"
    );
}
