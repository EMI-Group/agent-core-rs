//! Tests for per-case run telemetry: what a runner measures and how the seam
//! carries it.
//!
//! Every model path is scripted (`ScriptedModel` with explicit `Usage`s), so the
//! token totals are exact; latency is only ever asserted to be measured and
//! non-negative, because no test may depend on wall-clock time.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::Result;
use openjiuwen_foundation::prompt::Usage;
use openjiuwen_test_support::{MockCompletion, ScriptedModel};
use serde_json::{json, Value};

use crate::tune::base::{Case, EvaluatedCase};
use crate::tune::tool::{RunTelemetry, ToolCallRecord, ToolStubs, ToolTrace};

use super::super::{
    CaseEvaluator, DeterministicEvaluator, EvaluatedRun, ModelEvaluator, ModelPromptRunner,
    PromptRunner, RunnerEvaluator, ScriptedPromptRunner,
};
use super::{answer_map, cases, stub, tool_case};

/// A runner that only implements [`PromptRunner::run`], so `run_traced` comes
/// from the provided default.
struct AnswerOnlyRunner;

#[async_trait]
impl PromptRunner for AnswerOnlyRunner {
    async fn run(&self, _prompt: &str, _case: &Case) -> Result<BTreeMap<String, Value>> {
        Ok(answer_map("output", "Paris"))
    }
}

/// An evaluator that only implements [`CaseEvaluator::evaluate`].
struct ScoreOnly {
    /// The score every case gets.
    score: f64,
}

#[async_trait]
impl CaseEvaluator for ScoreOnly {
    async fn evaluate(&self, _prompt: &str, case: &Case) -> Result<EvaluatedCase> {
        EvaluatedCase::new(
            case.clone(),
            answer_map("output", "Paris"),
            self.score,
            "fixed",
        )
    }
}

/// An evaluator that only overrides [`CaseEvaluator::evaluate_run`].
struct RunOnly;

#[async_trait]
impl CaseEvaluator for RunOnly {
    async fn evaluate(&self, _prompt: &str, case: &Case) -> Result<EvaluatedCase> {
        EvaluatedCase::new(case.clone(), BTreeMap::new(), 0.0, "untraced")
    }

    async fn evaluate_run(&self, _prompt: &str, case: &Case) -> Result<EvaluatedRun> {
        let evaluated = EvaluatedCase::new(
            case.clone(),
            answer_map("output", "Paris"),
            1.0,
            "measured run",
        )?;
        Ok(EvaluatedRun::new(
            evaluated,
            ToolTrace::new([ToolCallRecord::success(
                "lookup",
                json!({}),
                json!({"ok": true}),
            )]),
            RunTelemetry::default()
                .with_latency_ms(1.5)
                .with_prompt_tokens(4)
                .with_completion_tokens(2),
        ))
    }
}

/// A model whose first turn calls `lookup` and whose second answers, with the
/// given usage per turn.
fn tool_then_answer_model(first: Usage, second: Usage) -> ScriptedModel {
    ScriptedModel::builder()
        .completion(MockCompletion::tool_call("lookup", json!({"q": "paris"})).with_usage(first))
        .completion(MockCompletion::text("4").with_usage(second))
        .build()
}

#[tokio::test]
async fn model_runner_measures_latency_and_sums_token_usage() {
    let model = tool_then_answer_model(Usage::new(7, 2), Usage::new(5, 3));
    let recorder = model.clone();
    let runner = ModelPromptRunner::new(Arc::new(model))
        .with_stubs(ToolStubs::new().with_stub(stub("lookup")));

    let outcome = runner
        .run_traced("prompt", &tool_case("c1", &["lookup"]))
        .await
        .unwrap();

    assert_eq!(recorder.call_count(), 2, "one tool round plus the answer");
    let latency = outcome
        .telemetry
        .latency_ms
        .expect("the run measured latency");
    assert!(latency.is_finite() && latency >= 0.0, "{latency}");
    assert_eq!(
        outcome.telemetry.prompt_tokens,
        Some(12),
        "7 + 5 prompt tokens over both calls"
    );
    assert_eq!(
        outcome.telemetry.completion_tokens,
        Some(5),
        "2 + 3 completion tokens over both calls"
    );
    assert_eq!(outcome.telemetry.total_tokens(), Some(17));
    assert_eq!(
        outcome.telemetry.cost, None,
        "cost is never inferred from tokens"
    );
    assert!(!outcome.telemetry.is_empty());
    assert_eq!(outcome.trace.names(), vec!["lookup"]);
    assert_eq!(outcome.answer, answer_map("output", "4"));
}

#[tokio::test]
async fn model_runner_measures_the_truncated_tool_loop_too() {
    let model = ScriptedModel::builder()
        .completion(MockCompletion::tool_call("lookup", json!({})).with_usage(Usage::new(2, 1)))
        .completion(MockCompletion::tool_call("lookup", json!({})).with_usage(Usage::new(3, 1)))
        .build();
    let recorder = model.clone();
    let runner = ModelPromptRunner::new(Arc::new(model))
        .with_stubs(ToolStubs::new().with_stub(stub("lookup")))
        .with_max_tool_rounds(1);

    let outcome = runner
        .run_traced("prompt", &tool_case("c1", &["lookup"]))
        .await
        .unwrap();

    assert_eq!(
        recorder.call_count(),
        2,
        "the budget stops the second round"
    );
    assert_eq!(outcome.trace.len(), 1);
    assert!(outcome.telemetry.latency_ms.is_some());
    assert_eq!(outcome.telemetry.prompt_tokens, Some(5));
    assert_eq!(outcome.telemetry.completion_tokens, Some(2));
}

#[tokio::test]
async fn model_runner_reports_zero_tokens_when_the_model_reports_no_usage() {
    // A scripted turn carries an all-zero `Usage`, which is still a measurement:
    // the counters are reported as zero rather than left absent.
    let runner = ModelPromptRunner::new(Arc::new(ScriptedModel::text("Paris")));
    let outcome = runner
        .run_traced("prompt", &cases().remove(0))
        .await
        .unwrap();
    assert_eq!(outcome.telemetry.prompt_tokens, Some(0));
    assert_eq!(outcome.telemetry.completion_tokens, Some(0));
    assert_eq!(outcome.telemetry.total_tokens(), Some(0));
    assert!(outcome.telemetry.latency_ms.is_some());
    assert!(outcome.telemetry.cost.is_none());
    assert!(!outcome.telemetry.is_empty());
}

#[tokio::test]
async fn scripted_and_default_runners_report_absent_telemetry() {
    let scripted = ScriptedPromptRunner::new().with_text_answer("c1", "Paris");
    let outcome = scripted.run_traced("p", &cases().remove(0)).await.unwrap();
    assert_eq!(outcome.telemetry, RunTelemetry::default());
    assert!(outcome.telemetry.is_empty());

    let plain = AnswerOnlyRunner;
    let outcome = plain.run_traced("p", &cases().remove(0)).await.unwrap();
    assert!(
        outcome.telemetry.is_empty(),
        "the provided default measures nothing"
    );
    assert_eq!(outcome.telemetry.latency_ms, None);
    assert_eq!(outcome.telemetry.total_tokens(), None);
}

#[tokio::test]
async fn evaluate_run_defaults_to_an_empty_trace_and_absent_telemetry() {
    let evaluator = ScoreOnly { score: 0.5 };
    let run = evaluator
        .evaluate_run("prompt", &cases().remove(0))
        .await
        .unwrap();
    assert!((run.score() - 0.5).abs() < 1e-9);
    assert_eq!(run.answer(), &answer_map("output", "Paris"));
    assert!(run.trace.is_empty());
    assert_eq!(run.used_tools(), Vec::<String>::new());
    assert!(run.is_empty_telemetry());
    assert_eq!(run.telemetry, RunTelemetry::default());
}

#[tokio::test]
async fn only_evaluate_run_is_honoured_through_evaluate_traced() {
    let (evaluated, trace) = RunOnly
        .evaluate_traced("prompt", &cases().remove(0))
        .await
        .unwrap();
    assert!(
        (evaluated.score - 1.0).abs() < 1e-9,
        "the run override answered, not `evaluate`: {evaluated:?}"
    );
    assert_eq!(trace.names(), vec!["lookup"]);

    let (mean, results) = RunOnly
        .score_cases_traced("prompt", &cases())
        .await
        .unwrap();
    assert!((mean - 1.0).abs() < 1e-9);
    assert_eq!(results.len(), 2);
    for (evaluated, trace) in &results {
        assert!((evaluated.score - 1.0).abs() < 1e-9);
        assert_eq!(trace.names(), vec!["lookup"]);
    }
}

#[tokio::test]
async fn runner_evaluator_surfaces_the_runner_telemetry() {
    let model = tool_then_answer_model(Usage::new(7, 2), Usage::new(5, 3));
    let recorder = model.clone();
    let runner: Arc<dyn PromptRunner> = Arc::new(
        ModelPromptRunner::new(Arc::new(model))
            .with_stubs(ToolStubs::new().with_stub(stub("lookup"))),
    );
    let evaluator = RunnerEvaluator::new(runner);

    let run = evaluator
        .evaluate_run("prompt", &tool_case("c1", &["lookup"]))
        .await
        .unwrap();
    assert_eq!(run.used_tools(), vec!["lookup"]);
    assert_eq!(run.telemetry.prompt_tokens, Some(12));
    assert_eq!(run.telemetry.completion_tokens, Some(5));
    assert!(
        run.telemetry
            .latency_ms
            .is_some_and(|latency| latency >= 0.0),
        "{:?}",
        run.telemetry
    );
    assert!(!run.is_empty_telemetry());
    assert_eq!(recorder.call_count(), 2);

    // `evaluate` reaches the same single run, so it adds no model call beyond
    // the scripted fallback turn.
    let before = recorder.call_count();
    let evaluated = evaluator
        .evaluate("prompt", &tool_case("c2", &["lookup"]))
        .await
        .unwrap();
    assert!((0.0..=1.0).contains(&evaluated.score));
    assert_eq!(recorder.call_count(), before + 1);
}

#[tokio::test]
async fn model_evaluator_forwards_the_measured_run() {
    let model = ScriptedModel::builder()
        .completion(MockCompletion::text("Paris").with_usage(Usage::new(3, 1)))
        .build();
    let recorder = model.clone();
    let evaluator = ModelEvaluator::new(Arc::new(model));

    let run = evaluator
        .evaluate_run("prompt", &cases().remove(0))
        .await
        .unwrap();
    assert!((run.score() - 1.0).abs() < 1e-9, "{:?}", run.evaluated);
    assert!(run.trace.is_empty());
    assert_eq!(run.telemetry.prompt_tokens, Some(3));
    assert_eq!(run.telemetry.completion_tokens, Some(1));
    assert!(run.telemetry.latency_ms.is_some());
    assert_eq!(recorder.call_count(), 1);
}

#[tokio::test]
async fn deterministic_evaluator_reports_absent_telemetry() {
    let evaluator = DeterministicEvaluator::new("Answer {{question}}.");
    let run = evaluator
        .evaluate_run(
            "Answer {{question}} step by step in the requested output format.",
            &cases().remove(0),
        )
        .await
        .unwrap();
    assert!(run.score() > 0.0, "{:?}", run.evaluated);
    assert!(run.trace.is_empty());
    assert!(run.is_empty_telemetry());
}

#[tokio::test]
async fn score_runs_means_the_runs_and_matches_the_untraced_views() {
    let runner = Arc::new(
        ScriptedPromptRunner::new()
            .with_text_answer("c1", "Paris")
            .with_text_answer("c2", "5"),
    );
    let evaluator = RunnerEvaluator::new(runner);
    let cases = cases();

    let (mean, runs) = evaluator.score_runs("prompt", &cases).await.unwrap();
    assert_eq!(runs.len(), 2);
    assert!(
        (mean - 0.5).abs() < 1e-9,
        "one of two cases correct: {mean}"
    );
    for run in &runs {
        assert!(run.is_empty_telemetry());
        assert!(run.trace.is_empty());
    }

    let (untraced_mean, untraced) = evaluator.score_cases("prompt", &cases).await.unwrap();
    assert!((untraced_mean - mean).abs() < 1e-12);
    let from_runs: Vec<EvaluatedCase> = runs.iter().map(|run| run.evaluated.clone()).collect();
    assert_eq!(from_runs, untraced);

    let (traced_mean, traced) = evaluator
        .score_cases_traced("prompt", &cases)
        .await
        .unwrap();
    assert!((traced_mean - mean).abs() < 1e-12);
    assert_eq!(traced.len(), runs.len());
    for ((evaluated, trace), run) in traced.iter().zip(&runs) {
        assert_eq!(evaluated, &run.evaluated);
        assert!(trace.is_empty());
    }

    let (empty_mean, empty_runs) = evaluator.score_runs("prompt", &[]).await.unwrap();
    assert!(empty_mean.abs() < 1e-9);
    assert_eq!(empty_runs, Vec::<EvaluatedRun>::new());
}

#[test]
fn evaluated_run_round_trips_and_defaults_every_field() {
    let evaluated = EvaluatedCase::new(
        cases().remove(0),
        answer_map("output", "Paris"),
        1.0,
        "exact",
    )
    .unwrap();
    let run = EvaluatedRun::new(
        evaluated,
        ToolTrace::new([ToolCallRecord::success(
            "lookup",
            json!({}),
            json!({"ok": true}),
        )]),
        RunTelemetry::default()
            .with_latency_ms(2.0)
            .with_prompt_tokens(4)
            .with_completion_tokens(1),
    );

    let value = serde_json::to_value(&run).unwrap();
    for key in ["evaluated", "trace", "telemetry"] {
        assert!(value.get(key).is_some(), "missing {key}");
    }
    let restored: EvaluatedRun = serde_json::from_value(value).unwrap();
    assert_eq!(restored, run);
    assert_eq!(restored.used_tools(), vec!["lookup"]);
    assert!(format!("{run:?}").contains("EvaluatedRun"));

    let empty: EvaluatedRun = serde_json::from_str("{}").unwrap();
    assert_eq!(empty, EvaluatedRun::default());
    assert!(empty.is_empty_telemetry());
    assert!(empty.trace.is_empty());
    assert!(empty.score().abs() < 1e-9);
    assert!(empty.answer().is_empty());
    assert_eq!(empty.used_tools(), Vec::<String>::new());

    let unknown: EvaluatedRun = serde_json::from_str(r#"{"extra": true}"#).unwrap();
    assert_eq!(unknown, EvaluatedRun::default());

    let partial: EvaluatedRun =
        serde_json::from_str(r#"{"telemetry": {"prompt_tokens": 3}}"#).unwrap();
    assert_eq!(partial.telemetry.prompt_tokens, Some(3));
    assert_eq!(partial.telemetry.latency_ms, None);
    assert_eq!(partial.telemetry.completion_tokens, None);
}
