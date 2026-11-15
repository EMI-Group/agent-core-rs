//! [`PromptSearchOptimizer`]: Python's bounded, multi-round prompt search.
//!
//! Ports the optimization loop of
//! `openjiuwen/dev_tools/tune/optimizer/prompt_search/optimizer.py` onto the
//! domain layer of [`crate::tune`]'s `prompt_search` module: the task/execution
//! vocabulary, the composite reward with its anti-hacking guards, the
//! convergence detector, the drift judge, the prompt memory, and the history
//! compressor.
//!
//! One run is a bounded loop over rounds. Each round proposes a candidate pool
//! — the round seed, the warm-start prompts (round 1 only), every deterministic
//! mutation of the round seed, and one model rewrite when a model is attached —
//! executes each candidate against every case, scores it through the composite
//! reward, records the round, and feeds the compressed lessons back into the
//! next round's rewrite request.
//!
//! Adaptation notes (the crate has no `LLMPromptPolicy` and no `LLMEnvironment`):
//!
//! - The *policy* is deterministic: the mutation bank plus one optional model
//!   rewrite. Elitism carries the best prompt forward as the next round's seed,
//!   so the result is never worse than the starting prompt.
//! - The *environment* is the [`CaseEvaluator`]
//!   ([`crate::tune::CaseEvaluator`]) seam, so the same loop runs offline
//!   against the deterministic evaluator or through a model. Each case runs
//!   through [`CaseEvaluator::evaluate_run`], so the run's measured
//!   [`RunTelemetry`](crate::tune::RunTelemetry) is kept per
//!   [`CaseResult`](crate::tune::CaseResult) and merged into the
//!   [`Execution`](crate::tune::Execution): the execution's latency and token
//!   totals are the sum over its cases, and its legacy `latency_ms`/
//!   `total_tokens` fields mirror them only when they were measured. An
//!   evaluator that measures nothing leaves both at zero, so the latency,
//!   token-usage, and cost reward components stay constant for it, while a
//!   measuring evaluator feeds them. [`OptimizationResult::token_usage`] stays
//!   empty either way because the result reports no run-level counters.
//! - The round budget is the smaller of [`PromptSearchConfig::max_iterations`]
//!   and the shared [`OptimizerConfig::iterations`], so the Python RLAF budget
//!   is capped by the tuning budget; with the defaults that is
//!   `min(6, 3) = 3` rounds.
//! - Ids, timestamps, and the run id are derived deterministically (see
//!   [`PromptSearchOptimizer::run_id`]), never from a clock, a uuid, or the
//!   environment.
//! - Logging is optional and side-effect free: a [`RunLog`] attached with
//!   [`PromptSearchOptimizer::with_run_log`] receives exactly one
//!   [`RunLogRecord`](crate::tune::RunLogRecord) per completed round (after that
//!   round's convergence check), an [`AdoptPolicy`] attached with
//!   [`PromptSearchOptimizer::with_adopt_policy`] adds the round's adoption
//!   verdict to it, and the caller's
//!   [`with_timestamp`](PromptSearchOptimizer::with_timestamp) stamps every
//!   record. Without a log nothing is built, a failed write never fails the run,
//!   and the returned [`OptimizationResult`] is identical either way.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::{Arc, Mutex};

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::Model;

use crate::tune::base::Case;
use crate::tune::evaluator::CaseEvaluator;
use crate::tune::optimizer::adopt::{AdoptDecision, AdoptPolicy};
use crate::tune::optimizer::config::OptimizerConfig;
use crate::tune::optimizer::search_config::PromptSearchConfig;
use crate::tune::optimizer::suggestion::{
    default_evaluator, describe_model, ensure_placeholders, validate_inputs, worst_cases, Mutator,
};
use crate::tune::optimizer::OptimizationResult;
use crate::tune::prompt_search::{
    candidate_id, CompositeReward, ConvergenceDetector, ConvergenceState, DriftJudge, Evaluation,
    HistoryCompressor, HistoryEntry, IterationRecord, LexicalDriftJudge, NullDriftJudge,
    NullPromptMemory, OptimizationHistory, PromptCandidate, PromptMemory, PromptRecord,
    PromptTaskCase, PromptTaskSpec, RewardBreakdown, RunLog,
};

/// How many prior records a run warm-starts from (Python's `top_k=3`).
const WARM_START_LIMIT: usize = 3;

/// How many worst cases the model rewrite is shown (the other optimizers' cap).
const WORST_CASE_LIMIT: usize = 5;

/// Offset basis of the 32-bit FNV-1a hash used by [`PromptSearchOptimizer::run_id`].
const FNV32_OFFSET: u32 = 0x811c_9dc5;

/// Prime of the 32-bit FNV-1a hash used by [`PromptSearchOptimizer::run_id`].
const FNV32_PRIME: u32 = 0x0100_0193;

/// The 32-bit FNV-1a hash of the concatenation of `parts`.
fn fnv1a32(parts: &[&[u8]]) -> u32 {
    let mut hash = FNV32_OFFSET;
    for part in parts {
        for byte in *part {
            hash ^= u32::from(*byte);
            hash = hash.wrapping_mul(FNV32_PRIME);
        }
    }
    hash
}

/// Runs Python's bounded, multi-round prompt search and returns the best prompt.
///
/// Python: `openjiuwen.dev_tools.tune.optimizer.prompt_search.PromptSearchOptimizer`.
pub struct PromptSearchOptimizer {
    /// The shared search budget (round cap, parallelism, early stop).
    config: OptimizerConfig,
    /// The RLAF-P tunables (rounds, candidates, convergence, drift, gate).
    search: PromptSearchConfig,
    /// The model consulted for one rewrite per round, when one is attached.
    model: Option<Arc<dyn Model>>,
    /// The evaluator, or the deterministic default when none was injected.
    evaluator: Option<Arc<dyn CaseEvaluator>>,
    /// The composite reward, or one built from the search config.
    reward: Option<CompositeReward>,
    /// The drift judge, or the config-derived default.
    drift_judge: Option<Arc<dyn DriftJudge>>,
    /// The prompt memory, when the caller wants warm starts and persistence.
    memory: Option<Arc<dyn PromptMemory>>,
    /// The run log, when the caller wants one record per round.
    run_log: Option<Arc<Mutex<dyn RunLog>>>,
    /// The timestamp stamped on every run-log record (never a clock read).
    timestamp: f64,
    /// The adoption policy, when the caller wants a per-round verdict.
    adopt_policy: Option<AdoptPolicy>,
    /// The seed of the deterministic run identifier.
    run_id_seed: u64,
}

impl Default for PromptSearchOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl PromptSearchOptimizer {
    /// Creates an optimizer with the default configuration, no model, and the
    /// deterministic evaluator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: OptimizerConfig::default(),
            search: PromptSearchConfig::default(),
            model: None,
            evaluator: None,
            reward: None,
            drift_judge: None,
            memory: None,
            run_log: None,
            timestamp: 0.0,
            adopt_policy: None,
            run_id_seed: 0,
        }
    }

    /// Replaces the shared search budget.
    #[must_use]
    pub fn with_config(mut self, config: OptimizerConfig) -> Self {
        self.config = config;
        self
    }

    /// Replaces the prompt-search (RLAF-P) tunables.
    #[must_use]
    pub fn with_search_config(mut self, search: PromptSearchConfig) -> Self {
        self.search = search;
        self
    }

    /// Attaches a model that rewrites the round seed once per round.
    #[must_use]
    pub fn with_model(mut self, model: Arc<dyn Model>) -> Self {
        self.model = Some(model);
        self
    }

    /// Replaces the evaluator.
    #[must_use]
    pub fn with_evaluator(mut self, evaluator: Arc<dyn CaseEvaluator>) -> Self {
        self.evaluator = Some(evaluator);
        self
    }

    /// Replaces the composite reward model.
    #[must_use]
    pub fn with_reward(mut self, reward: CompositeReward) -> Self {
        self.reward = Some(reward);
        self
    }

    /// Replaces the drift judge.
    #[must_use]
    pub fn with_drift_judge(mut self, drift_judge: Arc<dyn DriftJudge>) -> Self {
        self.drift_judge = Some(drift_judge);
        self
    }

    /// Replaces the prompt memory (enables warm starts and persistence).
    #[must_use]
    pub fn with_memory(mut self, memory: Arc<dyn PromptMemory>) -> Self {
        self.memory = Some(memory);
        self
    }

    /// Attaches a run log that receives one record per completed round.
    ///
    /// The log is shared through a [`Mutex`], so a caller keeps its own handle
    /// and reads back what the run wrote (an
    /// [`InMemoryRunLog`](crate::tune::InMemoryRunLog) is the offline choice).
    /// A poisoned mutex or a sink error is logged and never fails the run, and
    /// without a log nothing is built, so the search result is byte-identical
    /// either way.
    #[must_use]
    pub fn with_run_log(mut self, log: Arc<Mutex<dyn RunLog>>) -> Self {
        self.run_log = Some(log);
        self
    }

    /// Sets the timestamp stamped on every run-log record.
    ///
    /// The caller supplies it (nothing here reads a clock); the default is
    /// `0.0`.
    #[must_use]
    pub fn with_timestamp(mut self, timestamp: f64) -> Self {
        self.timestamp = timestamp;
        self
    }

    /// Attaches an adoption policy that judges every round's winner.
    ///
    /// The decision is recorded on that round's run-log record and never
    /// influences the search itself: without a run log nothing is emitted,
    /// and the elite is still chosen by reward alone.
    #[must_use]
    pub fn with_adopt_policy(mut self, policy: AdoptPolicy) -> Self {
        self.adopt_policy = Some(policy);
        self
    }

    /// Replaces the seed of the deterministic run identifier.
    #[must_use]
    pub fn with_run_id_seed(mut self, run_id_seed: u64) -> Self {
        self.run_id_seed = run_id_seed;
        self
    }

    /// The shared search budget.
    #[must_use]
    pub fn config(&self) -> &OptimizerConfig {
        &self.config
    }

    /// The prompt-search tunables.
    #[must_use]
    pub fn search_config(&self) -> &PromptSearchConfig {
        &self.search
    }

    /// Whether a model is attached.
    #[must_use]
    pub fn has_model(&self) -> bool {
        self.model.is_some()
    }

    /// Whether a run log is attached.
    #[must_use]
    pub fn has_run_log(&self) -> bool {
        self.run_log.is_some()
    }

    /// Whether an adoption policy is attached.
    #[must_use]
    pub fn has_adopt_policy(&self) -> bool {
        self.adopt_policy.is_some()
    }

    /// The timestamp stamped on every run-log record (`0.0` by default).
    #[must_use]
    pub fn timestamp(&self) -> f64 {
        self.timestamp
    }

    /// The deterministic run identifier a run of `task` reports with `seed`.
    ///
    /// `{seed:016x}-{hash:08x}`, where the suffix is the 32-bit FNV-1a hash of
    /// the objective, the base prompt, and the case count: two runs of the same
    /// task share an id, different tasks do not, and nothing reads a clock or a
    /// uuid. [`Self::with_run_id_seed`] replaces the seed.
    #[must_use]
    pub fn run_id(seed: u64, task: &PromptTaskSpec) -> String {
        let case_count = task.cases.len().to_string();
        let hash = fnv1a32(&[
            task.objective.as_bytes(),
            b"\x1f",
            task.base_prompt.as_bytes(),
            b"\x1f",
            case_count.as_bytes(),
        ]);
        format!("{seed:016x}-{hash:08x}")
    }

    /// Searches for a better prompt than `prompt` for `cases`.
    ///
    /// The seed prompt doubles as the task objective, so the drift judge
    /// measures every candidate against the intent the prompt started with.
    /// All cases are treated as visible; use [`Self::optimize_task`] to declare
    /// held-out cases, constraints, or metadata.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] for a blank prompt, an empty case list, or an
    /// out-of-range configuration; otherwise the evaluator's own error.
    pub async fn optimize(&self, prompt: &str, cases: &[Case]) -> Result<OptimizationResult> {
        validate_inputs(prompt, cases)?;
        let task = PromptTaskSpec::new(prompt)
            .with_base_prompt(prompt)
            .with_cases(
                cases
                    .iter()
                    .map(|case| PromptTaskCase::from_case(case, false))
                    .collect(),
            );
        self.run(&task, cases).await
    }

    /// Runs the full search for `task` and returns the best prompt found.
    ///
    /// Unlike [`Self::optimize`] this keeps the task's objective, constraints,
    /// base prompt, hidden cases, and metadata. A blank
    /// [`PromptTaskSpec::base_prompt`] falls back to the objective; the
    /// objective itself must not be blank.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] for a blank objective, an empty case list, a case
    /// with blank input text, or an out-of-range configuration.
    pub async fn optimize_task(&self, task: &PromptTaskSpec) -> Result<OptimizationResult> {
        if task.objective.trim().is_empty() {
            return Err(Error::validation("objective must not be blank"));
        }
        if task.cases.is_empty() {
            return Err(Error::validation("cases must not be empty"));
        }
        let mut cases = Vec::with_capacity(task.cases.len());
        for (index, case) in task.cases.iter().enumerate() {
            let converted = case.to_case().map_err(|error| {
                Error::validation(format!(
                    "PromptTaskSpec.cases[{index}] cannot be evaluated: {error}"
                ))
            })?;
            cases.push(converted);
        }
        self.run(task, &cases).await
    }

    /// The shared loop: rounds of propose → execute → reward → converge.
    async fn run(&self, task: &PromptTaskSpec, cases: &[Case]) -> Result<OptimizationResult> {
        let setup = self.prepare(task, cases)?;
        let reward = self.reward.as_ref().unwrap_or(&setup.default_reward);
        let mutator = Mutator::new(self.model.clone());
        let collaborators = Collaborators {
            evaluator: &*setup.evaluator,
            reward,
            drift_judge: &*setup.drift_judge,
            mutator: &mutator,
        };

        let mut history = OptimizationHistory::default();
        let mut tracker = Tracker::new(&setup.base);
        let mut records: Vec<IterationRecord> = Vec::new();
        let mut round_best_scores: Vec<f64> = Vec::new();
        let mut converged = false;
        let mut convergence_reason: Option<String> = None;
        let mut counter = 0usize;
        let mut context = policy_context(&history.compressed, &setup.warm);

        for index in 1..=setup.budget {
            // Elitism: later rounds start from the best prompt so far.
            let first = index == 1;
            let round_seed = if first {
                setup.base.clone()
            } else {
                tracker.prompt.clone()
            };
            let input = RoundInput {
                task,
                cases,
                hidden: &setup.hidden,
                warm: &setup.warm,
                round_seed: round_seed.clone(),
                first,
                context: context.as_deref(),
            };
            let evaluations = self.run_round(input, &collaborators, &mut counter).await;
            let scores: Vec<f64> = evaluations.iter().map(Evaluation::score).collect();
            let Some(round_best_index) = best_index(&scores) else {
                break;
            };
            let round_best = &evaluations[round_best_index];
            let round_observations = observations(round_best.candidate(), &round_best.reward);
            // The adoption history is the rounds before this one, so a round is
            // never compared against itself.
            let adopt = adopt_decision(self.adopt_policy.as_ref(), round_best, &round_best_scores);
            let stop = self.close_round(
                ScoredRound {
                    index,
                    first,
                    round_seed: &round_seed,
                    evaluations: &evaluations,
                    best_index: round_best_index,
                    scores: &scores,
                    observations: &round_observations,
                    adopt,
                },
                &mut history,
                &mut records,
                &mut round_best_scores,
                &mut tracker,
                &setup,
            );
            context = policy_context(&history.compressed, &setup.warm);
            if let Some(reason) = stop {
                converged = true;
                convergence_reason = Some(reason);
                break;
            }
        }

        tracker.persist(
            self.memory.is_some(),
            &*setup.memory,
            &setup.run_id,
            task,
            setup.baseline,
            records.len(),
        );

        let score = tracker.finite_score();
        Ok(OptimizationResult {
            best_prompt: tracker.prompt,
            score,
            iterations: records.len(),
            converged,
            convergence_reason,
            token_usage: BTreeMap::new(),
            run_id: setup.run_id,
            records,
            ..OptimizationResult::default()
        })
    }

    /// Absorbs one scored round into the run.
    ///
    /// The round is recorded in the iteration records and the compressed
    /// history, its winner's adoption verdict (when a policy is attached) is
    /// taken against the preceding rounds, the elite moves only on a strictly
    /// better score, and the convergence detector plus the shared early-stop
    /// score decide whether the run ends after this round. The run-log record is
    /// written last, so it carries this round's stopping verdict. Returns why the
    /// run should stop, if it should.
    fn close_round(
        &self,
        round: ScoredRound<'_>,
        history: &mut OptimizationHistory,
        records: &mut Vec<IterationRecord>,
        round_best_scores: &mut Vec<f64>,
        tracker: &mut Tracker,
        setup: &Setup,
    ) -> Option<String> {
        let round_best = &round.evaluations[round.best_index];
        record_round(
            records,
            history,
            round.index,
            round.evaluations,
            round_best,
            round.observations,
        );
        let previous = tracker.score;
        tracker.absorb(
            round.first,
            round.round_seed,
            round.scores,
            round.evaluations,
            round_best,
            round.observations.to_vec(),
        );
        let selected = tracker.score > previous;
        round_best_scores.push(round_best.score());
        let state = setup.detector.check(round_best_scores);
        let stop = stopping_reason(state.as_ref(), tracker.score, self.config.early_stop_score);
        self.log_round(RoundRecord {
            iteration: round.index,
            evaluations: round.evaluations,
            best_index: round.best_index,
            // The elite score the round started from: round 1 starts from no
            // elite at all (`-inf`), which the record reports as `None`.
            prior_best_score: previous.is_finite().then_some(previous),
            best_score: tracker.finite_score(),
            selected,
            convergence: state.as_ref(),
            stop_reason: stop.as_deref(),
            adopt: round.adopt,
            timestamp: self.timestamp,
            run_id: &setup.run_id,
            observations: round.observations,
        });
        stop
    }

    /// Validates the run and builds the collaborators and derived inputs once.
    fn prepare(&self, task: &PromptTaskSpec, cases: &[Case]) -> Result<Setup> {
        self.config.validate()?;
        self.search.validate()?;
        if task.objective.trim().is_empty() {
            return Err(Error::validation("objective must not be blank"));
        }
        if cases.is_empty() {
            return Err(Error::validation("cases must not be empty"));
        }

        // The seed prompt is the fixed intent drift is measured against.
        let base = base_prompt(task);
        let hidden: Vec<bool> = (0..cases.len())
            .map(|index| task.cases.get(index).is_some_and(|case| case.hidden))
            .collect();
        let evaluator = self
            .evaluator
            .clone()
            .unwrap_or_else(|| default_evaluator(&base));
        let drift_judge: Arc<dyn DriftJudge> = match &self.drift_judge {
            Some(judge) => judge.clone(),
            None if self.search.drift_penalty == 0.0 => Arc::new(NullDriftJudge::new()),
            None => Arc::new(LexicalDriftJudge::new()),
        };
        let memory: Arc<dyn PromptMemory> = self
            .memory
            .clone()
            .unwrap_or_else(|| Arc::new(NullPromptMemory::new()));
        let detector = ConvergenceDetector::new(
            self.search.convergence_threshold,
            self.search.convergence_window,
            self.search.target_score,
        )?;
        let baseline = memory
            .best_for_objective(&task.objective)
            .map(|record| record.reward);
        Ok(Setup {
            base,
            budget: self.search.max_iterations.min(self.config.iterations),
            run_id: Self::run_id(self.run_id_seed, task),
            hidden,
            evaluator,
            default_reward: CompositeReward::default()
                .with_min_correctness(self.search.min_correctness)
                .with_drift_penalty(self.search.drift_penalty),
            drift_judge,
            warm: warm_start(&*memory, task),
            memory,
            detector,
            baseline,
        })
    }

    /// One round: execute the seed, propose the pool, and score every candidate.
    async fn run_round(
        &self,
        input: RoundInput<'_>,
        collaborators: &Collaborators<'_>,
        counter: &mut usize,
    ) -> Vec<Evaluation> {
        let rationale = if input.first {
            "user base prompt"
        } else {
            "carried-over best prompt (elitism)"
        };
        let seed = PromptCandidate::new(&input.round_seed, rationale)
            .with_id(candidate_id(&input.round_seed, *counter));
        *counter += 1;
        // The seed runs first: it is both an elitist candidate and the evidence
        // the round's rewrite is shown.
        let seed_outcome =
            execute_candidate(collaborators.evaluator, seed, input.cases, input.hidden).await;
        let evidence = worst_cases(&seed_outcome.evaluated, WORST_CASE_LIMIT);

        let mut proposals: Vec<(String, String)> = Vec::new();
        if input.first {
            for record in input.warm {
                if record.prompt != input.round_seed {
                    proposals.push((
                        record.prompt.clone(),
                        "warm start from prompt memory".to_owned(),
                    ));
                }
            }
        }
        for mutation in Mutator::deterministic(&input.round_seed) {
            proposals.push((mutation, "deterministic mutation".to_owned()));
        }
        if let Some(rewrite) = collaborators
            .mutator
            .model_rewrite_with(
                &input.round_seed,
                &evidence,
                input.cases,
                Some(self.search.policy_temperature as f32),
                input.context,
            )
            .await
        {
            proposals.push((rewrite, "model rewrite".to_owned()));
        }
        let candidates = self.round_candidates(
            seed_outcome.execution.candidate.clone(),
            &input.round_seed,
            proposals,
            counter,
        );
        let rest = execute_candidates(
            collaborators.evaluator,
            &candidates[1..],
            input.cases,
            input.hidden,
            self.config.num_parallel,
        )
        .await;

        let mut executions = Vec::with_capacity(candidates.len());
        executions.push(seed_outcome.execution);
        executions.extend(rest.into_iter().map(|outcome| outcome.execution));
        let drift = drift_scores(
            collaborators.drift_judge,
            &executions,
            input.task,
            self.search.drift_penalty,
        )
        .await;
        let breakdowns = collaborators
            .reward
            .evaluate(&executions, input.task, &drift);
        executions
            .into_iter()
            .zip(breakdowns)
            .map(|(execution, reward)| Evaluation { execution, reward })
            .collect()
    }

    /// Normalizes, deduplicates, and truncates a round's proposals.
    ///
    /// The seed always leads, `{{placeholder}}`s are repaired against it, the
    /// first occurrence of a prompt wins, and at most
    /// [`PromptSearchConfig::candidate_prompts`] candidates are kept. Ids come
    /// from the run-wide `counter`, so they stay unique and reproducible.
    fn round_candidates(
        &self,
        seed: PromptCandidate,
        round_seed: &str,
        proposals: Vec<(String, String)>,
        counter: &mut usize,
    ) -> Vec<PromptCandidate> {
        let mut visited: BTreeSet<String> = BTreeSet::new();
        visited.insert(round_seed.to_owned());
        let mut candidates = vec![seed];
        for (prompt, rationale) in proposals {
            let prompt = ensure_placeholders(round_seed, &prompt);
            if !visited.insert(prompt.clone()) {
                continue;
            }
            if candidates.len() >= self.search.candidate_prompts {
                break;
            }
            let candidate =
                PromptCandidate::new(&prompt, rationale).with_id(candidate_id(&prompt, *counter));
            *counter += 1;
            candidates.push(candidate);
        }
        candidates
    }

    /// Writes one round to the attached run log, or does nothing without one.
    fn log_round(&self, round: RoundRecord<'_>) {
        if self.run_log.is_none() {
            return;
        }
        let record = round.build();
        emit(self.run_log.as_deref(), &record);
    }
}

impl fmt::Debug for PromptSearchOptimizer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PromptSearchOptimizer")
            .field("config", &self.config)
            .field("search_config", &self.search)
            .field("model", &describe_model(self.model.as_ref()))
            .field("custom_evaluator", &self.evaluator.is_some())
            .field("custom_reward", &self.reward.is_some())
            .field("custom_drift_judge", &self.drift_judge.is_some())
            .field("has_memory", &self.memory.is_some())
            .field("has_run_log", &self.run_log.is_some())
            .field("has_adopt_policy", &self.adopt_policy.is_some())
            .field("timestamp", &self.timestamp)
            .field("run_id_seed", &self.run_id_seed)
            .finish()
    }
}

/// The collaborators one run composes once and reuses in every round.
struct Collaborators<'a> {
    /// The evaluator every candidate is executed through.
    evaluator: &'a dyn CaseEvaluator,
    /// The composite reward every execution is scored by.
    reward: &'a CompositeReward,
    /// The drift judge, unless the drift penalty disables it.
    drift_judge: &'a dyn DriftJudge,
    /// The deterministic bank plus the optional model rewrite.
    mutator: &'a Mutator,
}

/// One scored round, ready to be absorbed into the run.
struct ScoredRound<'a> {
    /// The 1-based round number.
    index: usize,
    /// Whether this is round 1 (which sets the elite's floor).
    first: bool,
    /// The prompt the round started from.
    round_seed: &'a str,
    /// Every evaluated candidate of the round, in evaluation order.
    evaluations: &'a [Evaluation],
    /// The index of the round's winning candidate.
    best_index: usize,
    /// The candidates' scores, in the same order as `evaluations`.
    scores: &'a [f64],
    /// The winner's observations.
    observations: &'a [String],
    /// The winner's adoption verdict, when a policy is attached.
    adopt: Option<AdoptDecision>,
}

/// Everything one round needs from the run around it.
struct RoundInput<'a> {
    /// The task being optimized.
    task: &'a PromptTaskSpec,
    /// The evaluator-facing cases, in task order.
    cases: &'a [Case],
    /// Whether each case is held out.
    hidden: &'a [bool],
    /// The warm-start records (used by round 1 only).
    warm: &'a [PromptRecord],
    /// The round seed: the base prompt, or the elite of the previous rounds.
    round_seed: String,
    /// Whether this is round 1.
    first: bool,
    /// The round's policy context for the model rewrite.
    context: Option<&'a str>,
}

/// Records the round in `records` and in the compressed history.
fn record_round(
    records: &mut Vec<IterationRecord>,
    history: &mut OptimizationHistory,
    index: usize,
    evaluations: &[Evaluation],
    best: &Evaluation,
    best_observations: &[String],
) {
    let mut record = IterationRecord::new(index);
    for evaluation in evaluations {
        record.push(evaluation.candidate().clone(), evaluation.score());
    }
    record.best_score = best.score();
    records.push(record);
    history.add(
        HistoryEntry::new(index, best.score(), best.candidate().prompt())
            .with_observations(best_observations.to_vec()),
    );
    history.compressed = HistoryCompressor::default().compress(history);
}

/// Why the search should stop after a round, if it should: the already-computed
/// convergence verdict first (target reward, plateau, variance), then the shared
/// early-stop score.
fn stopping_reason(
    state: Option<&ConvergenceState>,
    best_score: f64,
    early_stop_score: f64,
) -> Option<String> {
    if let Some(state) = state {
        return Some(state.reason.clone());
    }
    if best_score >= early_stop_score {
        return Some(format!("early stop score {early_stop_score:.3} reached"));
    }
    None
}

/// The collaborators and derived inputs one run builds once.
struct Setup {
    /// The seed prompt drift is measured against.
    base: String,
    /// The round budget: the RLAF budget capped by the shared one.
    budget: usize,
    /// The deterministic run id.
    run_id: String,
    /// Whether each case is held out.
    hidden: Vec<bool>,
    /// The evaluator every candidate is executed through.
    evaluator: Arc<dyn CaseEvaluator>,
    /// The default reward, used when none was injected.
    default_reward: CompositeReward,
    /// The drift judge, unless the drift penalty disables it.
    drift_judge: Arc<dyn DriftJudge>,
    /// The prompt memory (null unless one was injected).
    memory: Arc<dyn PromptMemory>,
    /// The convergence detector.
    detector: ConvergenceDetector,
    /// The warm-start records, highest reward first.
    warm: Vec<PromptRecord>,
    /// The best prior reward for this objective, when memory held one.
    baseline: Option<f64>,
}

/// The running elite the loop carries between rounds.
struct Tracker {
    /// The best prompt seen so far (the seed before round 1).
    prompt: String,
    /// The best reward seen so far (`-inf` before round 1).
    score: f64,
    /// The best round's reward breakdown.
    reward: Option<RewardBreakdown>,
    /// The best round's observations.
    observations: Vec<String>,
}

impl Tracker {
    /// Starts from the seed prompt with no score yet.
    fn new(base: &str) -> Self {
        Self {
            prompt: base.to_owned(),
            score: f64::NEG_INFINITY,
            reward: None,
            observations: Vec::new(),
        }
    }

    /// Absorbs a round: round 1 sets the floor from the seed, then the round
    /// best replaces the elite only when it is strictly better.
    fn absorb(
        &mut self,
        first: bool,
        round_seed: &str,
        scores: &[f64],
        evaluations: &[Evaluation],
        round_best: &Evaluation,
        round_observations: Vec<String>,
    ) {
        if first {
            self.score = scores[0];
            round_seed.clone_into(&mut self.prompt);
            self.reward = Some(evaluations[0].reward.clone());
            self.observations = observations(evaluations[0].candidate(), &evaluations[0].reward);
        }
        if round_best.score() > self.score {
            self.score = round_best.score();
            round_best.candidate().prompt().clone_into(&mut self.prompt);
            self.reward = Some(round_best.reward.clone());
            self.observations = round_observations;
        }
    }

    /// The reported score: `0.0` when no round ever produced a finite one.
    fn finite_score(&self) -> f64 {
        if self.score.is_finite() {
            self.score
        } else {
            0.0
        }
    }

    /// Stores the elite in a caller-supplied memory.
    ///
    /// A `store` failure is logged and never fails an otherwise successful run.
    fn persist(
        &self,
        store: bool,
        memory: &dyn PromptMemory,
        run_id: &str,
        task: &PromptTaskSpec,
        baseline: Option<f64>,
        iterations: usize,
    ) {
        if !store {
            return;
        }
        let Some(breakdown) = &self.reward else {
            return;
        };
        let mut stored = PromptRecord::new(
            format!("{run_id}-best"),
            &self.prompt,
            self.score.max(0.0),
            task.objective.clone(),
        )
        .with_characteristics(task.characteristics())
        .with_metrics(metrics_of(&breakdown.raw_components))
        .with_observations(self.observations.clone())
        .with_metadata(record_metadata(task, iterations));
        if let Some(baseline) = baseline {
            stored = stored.with_baseline_reward(baseline);
        }
        if let Err(error) = memory.store(stored) {
            tracing::warn!(
                %error,
                "prompt memory store failed; the search result is unaffected"
            );
        }
    }
}

mod execute;
mod run_log_record;

use execute::{
    base_prompt, best_index, drift_scores, execute_candidate, execute_candidates, metrics_of,
    observations, policy_context, record_metadata, warm_start,
};
use run_log_record::{adopt_decision, emit, RoundRecord};

#[cfg(test)]
mod tests;

#[cfg(test)]
#[path = "run_log_tests.rs"]
mod run_log_tests;
