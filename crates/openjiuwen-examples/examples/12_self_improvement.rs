//! `12_self_improvement` — the self-improvement surfaces, driven entirely by fakes.
//!
//! Two seams, both offline and instant: **auto-harness orchestration**
//! (`openjiuwen::rsi::auto_harness`), where a [`HarnessRunner`] is the only seam
//! to `openjiuwen-harness`, and **the evolving trainer's collaborators**
//! (`openjiuwen::agent_evolving`), whose `Optimizer` and `Evaluator` run over
//! fixed data. Python counterparts: the evolving trainer mirrors
//! `examples/agent_evolving/react_agent_evolving.py`; auto-harness has no Python
//! example, so part one mirrors the Rust integration tests. Live mode stays scripted.

#![allow(
    // The two fakes name themselves with a literal, as their Python counterparts do.
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
use openjiuwen::agent_evolving::trainer::{Trainer, TrainerConfig};
use openjiuwen::rsi::auto_harness::{
    AutoHarnessConfig, AutoHarnessOrchestrator, HarnessRunOutcome, HarnessRunRequest, HarnessRunner,
};
use openjiuwen::rsi::{Case, CaseResult, Error, EvaluationReport, Result as SdkResult};
use serde_json::{json, Value};

/// Upper bound on one session, so a misconfiguration cannot hang the demo.
const RUN_TIMEOUT: Duration = Duration::from_secs(20);
const COST_BUDGET_USD: f64 = 1.0;
const CASE_ID: &str = "case-1";
const CANDIDATES: [(&str, f64); 2] = [("temperature-0.80", 0.8), ("temperature-0.30", 0.3)];
/// The keyword a hit requires, and the two predictions the evaluator scores.
const EXPECTED_KEYWORD: &str = "Paris";
const PREDICTIONS: [&str; 2] = ["Paris is the capital of France.", "Lyon is in France."];

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

/// Part 2: the optimizer and evaluator seams the evolving trainer drives.
async fn evolving_section() -> Result<()> {
    common::banner::step("part 2 — agent-evolving: propose, score, and observe a candidate");

    let candidates = CANDIDATES.iter().map(|(id, temperature)| {
        let parameters = BTreeMap::from([("temperature".to_owned(), json!(temperature))]);
        Candidate::new(*id, parameters)
    });
    let mut optimizer = ScriptedOptimizer::new(candidates);
    let evaluator = KeywordEvaluator;
    let inputs = BTreeMap::from([("question".to_owned(), json!("Capital of France?"))]);
    let label = BTreeMap::from([("answer".to_owned(), json!(EXPECTED_KEYWORD))]);
    let validation_case = TrainingCase::new(inputs, label)?.with_case_id("val-1");
    for answer in PREDICTIONS {
        for candidate in optimizer.propose(1).await? {
            let prediction = json!({ "answer": answer });
            let evaluated = evaluator.evaluate(&validation_case, &prediction).await?;
            common::banner::note(&format!(
                "candidate `{}` scored {:.2} on `{}` ({})",
                candidate.id, evaluated.score, evaluated.case.case_id, evaluated.reason
            ));
            optimizer.observe(&candidate, evaluated.score);
        }
    }
    let best = optimizer.best().map_or_else(
        || "none".to_owned(),
        |best| format!("`{}` at {:.2}", best.id, best.score.unwrap_or_default()),
    );
    common::banner::note(&format!(
        "observed {} candidate(s); optimizer.best() = {best}",
        optimizer.observed.len()
    ));

    // The trainer loop itself cannot run here: `Trainer::train` needs a
    // `TrainableTarget`, whose `run_case(&self, case, cancel: &CancellationToken)`
    // must name `tokio_util::sync::CancellationToken` in this file, and this crate
    // has no `tokio-util` dependency (the SDK does not re-export the type).
    let trainer = Trainer::new(TrainerConfig {
        num_iterations: 1,
        candidates_per_epoch: 1,
        ..TrainerConfig::default()
    });
    common::banner::note(&format!(
        "a Trainer is ready: {} epoch(s), {} candidate(s) per epoch, early stop at {:.1}",
        trainer.config().num_iterations,
        trainer.config().candidates_per_epoch,
        trainer.config().early_stop_score
    ));
    common::banner::note(
        "`Trainer::train` would report `TrainerReport { best_score, steps, stopped_reason, \
         evaluations }` over the two fakes above; add `tokio-util.workspace = true` under \
         [dev-dependencies] in crates/openjiuwen-examples/Cargo.toml to name the token and run it",
    );
    Ok(())
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
