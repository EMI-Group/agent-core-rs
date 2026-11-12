//! `12_self_improvement` — the self-improvement surfaces, driven entirely by fakes.
//!
//! Two seams, both offline and instant: **auto-harness orchestration**
//! (`openjiuwen::rsi::auto_harness`), where a [`HarnessRunner`] is the only seam
//! to `openjiuwen-harness`, and **the evolving trainer's real loop**
//! (`openjiuwen::agent_evolving`), where `Trainer::train` runs the baseline, the
//! per-epoch forward pass and the candidate evaluations over a fake agent, a
//! scripted optimizer and a keyword evaluator. Python counterparts: the evolving
//! trainer mirrors `examples/agent_evolving/react_agent_evolving.py`; auto-harness
//! has no Python example, so part one mirrors the Rust integration tests. Live mode
//! stays scripted.
//!
//! The fake agent has no model: it answers from a script keyed by the applied
//! candidate, so the first candidate misses the keyword and the second hits it and
//! early-stops the run after two epochs of a tiny two-train/one-validation case
//! setup. `Trainer::new` installs a disabled checkpoint store, so the run writes
//! nothing and every `EpochReport::checkpoint_saved` is false.

#![allow(
    // Every fake names itself with a literal, as its Python counterpart does.
    clippy::unnecessary_literal_bound
)]

mod common;

use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use anyhow::{Context, Result};
use async_trait::async_trait;
use openjiuwen::agent_evolving::dataset::{Case as TrainingCase, EvaluatedCase};
use openjiuwen::agent_evolving::evaluator::Evaluator as TrainerEvaluator;
use openjiuwen::agent_evolving::optimizer::{Candidate, Optimizer};
use openjiuwen::agent_evolving::trainer::{
    StopReason, TrainableTarget, Trainer, TrainerConfig, TrainerReport,
};
use openjiuwen::rsi::auto_harness::{
    AutoHarnessConfig, AutoHarnessOrchestrator, HarnessRunOutcome, HarnessRunRequest, HarnessRunner,
};
use openjiuwen::rsi::{Case, CaseResult, Error, EvaluationReport, Result as SdkResult};
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

/// Upper bound on one session, so a misconfiguration cannot hang the demo.
const RUN_TIMEOUT: Duration = Duration::from_secs(20);
const COST_BUDGET_USD: f64 = 1.0;
const CASE_ID: &str = "case-1";
const CANDIDATES: [(&str, f64); 2] = [("temperature-0.80", 0.8), ("temperature-0.30", 0.3)];
/// The keyword a validation hit requires.
const EXPECTED_KEYWORD: &str = "Paris";
/// What the fake agent answers before any candidate is applied, and after each.
const BASELINE_ANSWER: &str = "Lyon is in France.";
const SCRIPTED_ANSWERS: [(&str, &str); 2] = [
    ("temperature-0.80", "Lyon is in France."),
    ("temperature-0.30", "Paris is the capital of France."),
];
const TRAIN_CASE_IDS: [&str; 2] = ["train-1", "train-2"];
const VAL_CASE_ID: &str = "val-1";

/// A harness runner that answers from a script and records every request.
/// A production session puts a real `DeepAgent` run behind [`HarnessRunner`].
struct ScriptedRunner {
    outcomes: Mutex<VecDeque<HarnessRunOutcome>>,
    requests: Mutex<Vec<HarnessRunRequest>>,
}

impl ScriptedRunner {
    /// Queues `outcomes` in call order.
    fn new(outcomes: Vec<HarnessRunOutcome>) -> Arc<Self> {
        Arc::new(Self {
            outcomes: Mutex::new(outcomes.into()),
            requests: Mutex::new(Vec::new()),
        })
    }

    /// Every request the orchestrator issued, in call order. A poisoned lock
    /// still yields a usable log: the fake runs on a single task.
    fn requests(&self) -> Vec<HarnessRunRequest> {
        self.requests
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}

#[async_trait]
impl HarnessRunner for ScriptedRunner {
    fn name(&self) -> &str {
        "scripted"
    }

    async fn run(&self, request: HarnessRunRequest) -> SdkResult<HarnessRunOutcome> {
        // Same poisoned-lock reasoning as `requests`: the queue stays usable.
        self.requests
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(request);
        // The guard is dropped here, so no lock is held while the outcome is built.
        let next = {
            let mut outcomes = self.outcomes.lock().unwrap_or_else(PoisonError::into_inner);
            outcomes.pop_front()
        };
        next.map_or_else(
            || {
                Err(Error::external(std::io::Error::other(
                    "scripted runner exhausted",
                )))
            },
            Ok,
        )
    }
}

/// One successful harness run: a graded one-case report plus its diagnostics.
fn outcome(score: f64, cost: f64) -> HarnessRunOutcome {
    let result = CaseResult {
        case_id: CASE_ID.to_owned(),
        passed: true,
        score: Some(score),
        ..CaseResult::default()
    };
    let mut report = EvaluationReport::from_results(vec![result]);
    report.cost_usd = cost;
    HarnessRunOutcome {
        report,
        messages: vec![format!("scripted run scored {score:.2}")],
        ..HarnessRunOutcome::default()
    }
}

/// Part 1: one auto-harness session graded by a scripted [`HarnessRunner`].
async fn auto_harness_section() -> Result<()> {
    common::banner::step("part 1 — auto-harness: a scripted HarnessRunner drives one session");

    // `validate()` rejects an empty `data_dir`; the default checkpoint store is
    // in memory, so nothing is ever written under this temporary directory.
    let data_dir = common::tempdir()?;
    let config = AutoHarnessConfig {
        data_dir: data_dir.path().to_path_buf(),
        max_iterations: 2,
        session_budget_secs: 60.0,
        cost_limit_usd: COST_BUDGET_USD,
        ..AutoHarnessConfig::default()
    };
    let target = Case {
        id: CASE_ID.to_owned(),
        name: Some("scripted case".to_owned()),
        input: json!({"task": "explain the pending change"}),
        ..Case::default()
    };

    let runner = ScriptedRunner::new(vec![outcome(0.4, 0.002), outcome(0.7, 0.003)]);
    let runner_seam: Arc<dyn HarnessRunner> = runner.clone();
    let orchestrator = AutoHarnessOrchestrator::builder(config)
        .with_runner(runner_seam)
        .with_cases(vec![target])
        .with_session_id("examples-12-session")
        .build()?;
    let summary = tokio::time::timeout(RUN_TIMEOUT, Arc::new(orchestrator).run())
        .await
        .context("the auto-harness run timed out")??;

    let requests = runner.requests();
    let stopped = summary
        .stop_reason
        .map_or("none".to_owned(), |r| format!("{r:?}"));
    let best = summary
        .best_score
        .map_or("none".to_owned(), |s| format!("{s:.2}"));
    let best_by = summary
        .best_candidate
        .as_ref()
        .map_or("none".to_owned(), ToString::to_string);
    common::banner::note(&format!(
        "workload: {} iteration(s), 1 proposal each, {} case(s) per run, ${COST_BUDGET_USD:.2} cap, stopped on {stopped}",
        summary.iterations,
        requests.first().map_or(0, |request| request.cases.len())
    ));
    common::banner::note(&format!(
        "evaluated {} candidate(s), {} accepted, best {best} from `{best_by}`",
        summary.evaluated, summary.accepted
    ));
    common::banner::note(&format!(
        "spent ${:.4}; the scripted runner saw {} request(s)",
        summary.spent_usd,
        requests.len()
    ));
    Ok(())
}

/// Proposes queued candidates and tracks the best score it observed.
struct ScriptedOptimizer {
    pending: VecDeque<Candidate>,
    observed: Vec<(Candidate, f64)>,
}

impl ScriptedOptimizer {
    fn new(candidates: impl IntoIterator<Item = Candidate>) -> Self {
        Self {
            pending: candidates.into_iter().collect(),
            observed: Vec::new(),
        }
    }
}

#[async_trait]
impl Optimizer for ScriptedOptimizer {
    fn name(&self) -> &str {
        "scripted"
    }

    async fn propose(&mut self, limit: usize) -> SdkResult<Vec<Candidate>> {
        let take = limit.min(self.pending.len());
        Ok(self.pending.drain(..take).collect())
    }

    fn observe(&mut self, candidate: &Candidate, score: f64) {
        let mut seen = candidate.clone();
        seen.score = Some(score);
        self.observed.push((seen, score));
    }

    fn best(&self) -> Option<&Candidate> {
        self.observed
            .iter()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(candidate, _)| candidate)
    }
}

/// Scores a prediction `1.0` when it contains [`EXPECTED_KEYWORD`], else `0.0`.
struct KeywordEvaluator;

#[async_trait]
impl TrainerEvaluator for KeywordEvaluator {
    fn name(&self) -> &str {
        "keyword"
    }

    async fn evaluate(&self, case: &TrainingCase, prediction: &Value) -> SdkResult<EvaluatedCase> {
        let answer = prediction
            .get("answer")
            .and_then(Value::as_str)
            .unwrap_or("");
        let hit = answer.contains(EXPECTED_KEYWORD);
        let reason = if hit { "keyword hit" } else { "keyword miss" };
        EvaluatedCase::new(case.clone(), None, f64::from(u8::from(hit)), reason)
    }
}

/// A fake agent with no model: it answers from a script keyed by the applied
/// candidate, which is why no model is needed. A production target puts a
/// `ReActAgent` behind the same [`TrainableTarget`] seam.
struct FakeAgent {
    applied: Mutex<Option<String>>,
    baseline: &'static str,
    script: &'static [(&'static str, &'static str)],
}

impl FakeAgent {
    /// Starts unconfigured, so `run_case` answers with `baseline`.
    fn new(baseline: &'static str, script: &'static [(&'static str, &'static str)]) -> Self {
        Self {
            applied: Mutex::new(None),
            baseline,
            script,
        }
    }

    /// The answer scripted for the applied candidate, else the baseline one.
    fn answer(&self) -> &'static str {
        let applied = self.applied.lock().unwrap_or_else(PoisonError::into_inner);
        let Some(id) = applied.as_deref() else {
            return self.baseline;
        };
        self.script
            .iter()
            .find(|(candidate, _)| *candidate == id)
            .map_or(self.baseline, |(_, answer)| *answer)
    }
}

#[async_trait]
impl TrainableTarget for FakeAgent {
    fn id(&self) -> &str {
        "fake-agent"
    }

    async fn run_case(&self, _case: &TrainingCase, cancel: &CancellationToken) -> SdkResult<Value> {
        if cancel.is_cancelled() {
            return Err(Error::cancelled());
        }
        Ok(json!({ "answer": self.answer() }))
    }

    async fn apply(&self, candidate: &Candidate) -> SdkResult<()> {
        *self.applied.lock().unwrap_or_else(PoisonError::into_inner) = Some(candidate.id.clone());
        Ok(())
    }
}

/// Why the trainer stopped, in the lowercase spelling the notes use.
fn stop_reason(reason: StopReason) -> &'static str {
    match reason {
        StopReason::Completed => "completed",
        StopReason::EarlyStop => "early stop",
        StopReason::Cancelled => "cancelled",
        StopReason::BudgetExhausted => "budget exhausted",
        StopReason::NoCandidates => "no candidates",
    }
}

/// Part 2: `Trainer::train` over the fake agent, optimizer, and evaluator.
async fn evolving_section() -> Result<()> {
    common::banner::step("part 2 — agent-evolving: the Trainer::train loop over a fake agent");

    let candidates = CANDIDATES.iter().map(|(id, temperature)| {
        let parameters = BTreeMap::from([("temperature".to_owned(), json!(temperature))]);
        Candidate::new(*id, parameters)
    });
    let mut optimizer = ScriptedOptimizer::new(candidates);
    let evaluator = KeywordEvaluator;
    let target = FakeAgent::new(BASELINE_ANSWER, &SCRIPTED_ANSWERS);
    let trainer = Trainer::new(TrainerConfig::default());

    let inputs = BTreeMap::from([("question".to_owned(), json!("Capital of France?"))]);
    let label = BTreeMap::from([("answer".to_owned(), json!(EXPECTED_KEYWORD))]);
    let train_cases = vec![
        TrainingCase::new(inputs.clone(), label.clone())?.with_case_id(TRAIN_CASE_IDS[0]),
        TrainingCase::new(inputs.clone(), label.clone())?.with_case_id(TRAIN_CASE_IDS[1]),
    ];
    let val_cases = vec![TrainingCase::new(inputs, label)?.with_case_id(VAL_CASE_ID)];
    common::banner::note(&format!(
        "workload: {} train case(s), {} validation case(s), up to {} epoch(s) of {} \
         candidate(s), early stop at {:.2}",
        train_cases.len(),
        val_cases.len(),
        trainer.config().num_iterations,
        trainer.config().candidates_per_epoch,
        trainer.config().early_stop_score
    ));

    let report = tokio::time::timeout(
        RUN_TIMEOUT,
        trainer.train(
            &target,
            &mut optimizer,
            &evaluator,
            &train_cases,
            &val_cases,
            &CancellationToken::new(),
        ),
    )
    .await
    .context("the evolving run timed out")??;

    for step in &report.steps {
        common::banner::note(&format!(
            "epoch {}: score {:.2}, best {:.2}, {} candidate(s), {} evaluation(s), {}, {}",
            step.epoch,
            step.score,
            step.best_score,
            step.candidates,
            step.evaluations,
            if step.improved {
                "improved"
            } else {
                "no improvement"
            },
            if step.checkpoint_saved {
                "checkpoint saved"
            } else {
                "checkpoint not saved"
            }
        ));
    }
    print_summary(&report);
    Ok(())
}

/// Prints the run summary and the candidate the trainer adopted.
fn print_summary(report: &TrainerReport) {
    common::banner::note(&format!(
        "stopped on {}, {} epoch(s), best {:.2} over {} evaluation(s)",
        stop_reason(report.stopped_reason),
        report.epochs_run,
        report.best_score,
        report.evaluations
    ));
    let winner = report.best_candidate.as_ref().map_or_else(
        || "none".to_owned(),
        |candidate| {
            let temperature = candidate
                .parameters
                .get("temperature")
                .map_or_else(|| "none".to_owned(), ToString::to_string);
            format!("`{}` at temperature {temperature}", candidate.id)
        },
    );
    common::banner::note(&format!("best candidate: {winner}"));
    // `Trainer::new` disables the checkpoint store, so the run wrote nothing.
    common::banner::note(
        "the trainer's checkpoint store is disabled by default (`Trainer::new`), so this run \
         wrote no file",
    );
}

#[tokio::main]
async fn main() -> Result<()> {
    let _mode = common::setup()?;
    common::banner::title("12 self improvement");

    auto_harness_section().await?;
    evolving_section().await?;

    common::banner::step("done");
    Ok(())
}
