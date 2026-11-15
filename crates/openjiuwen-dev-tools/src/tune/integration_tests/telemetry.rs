//! Scenario 4: measured telemetry reaching the reward components.
//!
//! A custom evaluator that reports a fixed `RunTelemetry` drives the latency,
//! token-usage, and cost components on real numbers, while a telemetry-less
//! execution keeps their documented constant.

use std::collections::BTreeMap;

use async_trait::async_trait;
use openjiuwen_common::error::Result;
use serde_json::json;

use super::support::*;
use crate::tune::*;

/// An evaluator that scores every case the same and reports a fixed telemetry.
struct TelemetryEvaluator {
    /// The score every case gets.
    score: f64,
    /// What every run reports it cost.
    telemetry: RunTelemetry,
}

#[async_trait]
impl CaseEvaluator for TelemetryEvaluator {
    async fn evaluate(&self, _prompt: &str, entry: &Case) -> Result<EvaluatedCase> {
        EvaluatedCase::new(
            entry.clone(),
            BTreeMap::from([("output".to_owned(), json!("answer"))]),
            self.score,
            "scripted score",
        )
    }

    async fn evaluate_run(&self, prompt: &str, entry: &Case) -> Result<EvaluatedRun> {
        let evaluated = self.evaluate(prompt, entry).await?;
        Ok(EvaluatedRun::new(
            evaluated,
            ToolTrace::default(),
            self.telemetry,
        ))
    }
}

#[tokio::test]
async fn a_telemetry_carrying_evaluator_feeds_the_reward_components() {
    let telemetry = RunTelemetry::default()
        .with_latency_ms(1000.0)
        .with_prompt_tokens(600)
        .with_completion_tokens(400)
        .with_cost(0.5);
    let evaluator = TelemetryEvaluator {
        score: 0.75,
        telemetry,
    };
    let cases = nine_cases();

    // The seam keeps what the run reported, per case and in aggregate.
    let (mean, runs) = evaluator
        .score_runs("prompt", &cases[..2])
        .await
        .expect("a scripted evaluation");
    assert_close(mean, 0.75);
    assert_eq!(runs.len(), 2);
    for run in &runs {
        assert_eq!(run.telemetry, telemetry, "score_runs keeps the telemetry");
        assert_close(run.score(), 0.75);
        assert!(!run.is_empty_telemetry());
    }
    let single = evaluator
        .evaluate_run("prompt", &cases[0])
        .await
        .expect("a scripted evaluation");
    assert_eq!(single.telemetry, telemetry);
    // The telemetry-free view of the same call drops it.
    let (unused_telemetry, trace) = evaluator
        .evaluate_traced("prompt", &cases[0])
        .await
        .expect("a scripted evaluation");
    assert_eq!(unused_telemetry, runs[0].evaluated);
    assert!(trace.is_empty());

    // The measured values are what the reward components score on.
    let task = PromptTaskSpec::new("Answer the question.");
    let mut execution = Execution::new(PromptCandidate::new("prompt", "seed"));
    execution.telemetry = telemetry;
    assert_close(execution.measured_latency_s(), 1.0);
    assert_eq!(execution.measured_total_tokens(), 1000);
    assert_eq!(execution.measured_cost(), Some(0.5));
    // 1 s against a 1 s half-life, 1000 tokens against a 1000-token half-life,
    // and 0.5 currency against a 0.5 half-life are all exactly half credit.
    assert_close(LatencyReward::new(1.0).score(&execution, &task), 0.5);
    assert_close(TokenUsageReward::new(1000.0).score(&execution, &task), 0.5);
    assert_close(CostReward::new(0.0, 0.5).score(&execution, &task), 0.5);
    // A bigger latency scores lower and a shorter one higher.
    assert!(
        LatencyReward::new(1.0).score(&execution, &task)
            < LatencyReward::new(4.0).score(&execution, &task)
    );

    // Nothing reported keeps every component at its documented constant.
    let bare = Execution::new(PromptCandidate::new("prompt", "seed"));
    assert!(bare.telemetry.is_empty());
    assert_close(LatencyReward::new(1.0).score(&bare, &task), 1.0);
    assert_close(TokenUsageReward::new(1000.0).score(&bare, &task), 1.0);
    assert_close(CostReward::new(3.0, 0.5).score(&bare, &task), 1.0);
}
