//! The bounded paper-optimisation loop.
//!
//! Python parity: the loop mirrors the node lifecycle of
//! `auto_research/tree_provider/orchestrator.py` (v0.1.20) — propose from the
//! frontier, score the candidate, adopt it only if it strictly beats the
//! frontier, and remember why a pruned attempt failed — reduced to the portable
//! core:
//!
//! 1. ensure the root exists;
//! 2. ask the [`PaperProposer`] for `beam_width` candidates derived from the
//!    frontier body;
//! 3. score each candidate through the [`PaperScorer`] (a scoring failure is
//!    recorded as a pruned `scoring_error` node, never fatal);
//! 4. rank the scored candidates with the shared
//!    [`PuctTree`] — the same
//!    candidate-ranking engine the program search uses;
//! 5. apply the [`adoption_gate`] against the frontier score and append either an
//!    adopted reporting node or a pruned node.
//!
//! The six `DeepAgent` stage agents, the LaTeX/pyright/Pillow toolchain and the
//! model environment plumbing stay deferred: a caller wires a real pipeline
//! behind [`PaperProposer`] and a real judge behind [`PaperScorer`]. Everything
//! in this module is offline and deterministic; ids and scores never read a
//! clock or a random source.
use super::config::PaperOptConfig;
use super::proposal::{PaperCandidate, PaperProposalRequest, PaperProposer, TemplatePaperProposer};
use super::scoring::{aggregate, ScoreBreakdown};
use super::tree::{adoption_gate, AdoptionOutcome, PaperNodeType, PaperTree};
use crate::artifact_rsi::program_opt::{Program, ProgramOptConfig, PuctTree};
use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::utils::id::uuid_v4;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;
/// Lifecycle of one paper-optimisation task.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PaperStatus {
    /// The run may still perform iterations.
    #[default]
    Running,
    /// The run stopped because its iteration budget, a target score, or an empty
    /// proposal set ended it.
    Completed,
    /// The run stopped because the proposer or a tree operation failed.
    Failed,
    /// The run stopped because [`PaperOptimizer::stop`] was called.
    Cancelled,
}
/// Observable state of one paper-optimisation task.
///
/// Python parity: `tree_provider.schemas.PaperTaskState`, restricted to the
/// portable fields. `node_count` excludes the root (Python's rule);
/// `best_node_id` is only ever a reporting node, never the root.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct PaperTaskState {
    /// Task identifier (`uuid_v4` unless the caller sets one).
    pub task_id: String,
    /// Lifecycle state.
    pub status: PaperStatus,
    /// Configured iteration budget (`PaperOptConfig::max_iterations`).
    pub max_iterations: u32,
    /// Path of the compiled paper artifact, when the caller sets one.
    pub artifact_path: String,
    /// Id of the best adopted reporting node.
    pub best_node_id: Option<String>,
    /// Composite score of `best_node_id`.
    pub score: Option<f64>,
    /// Baseline composite the task is compared against.
    pub baseline: Option<f64>,
    /// Id of the node the next round would revise (root or last adopted node).
    pub frontier_node_id: Option<String>,
    /// Number of candidate nodes created, the root excluded.
    pub node_count: usize,
    /// Consecutive failed/rejected attempts against the current frontier.
    pub attempts_since_last_adoption: u32,
    /// Most recent failure class (empty when the last attempt was adopted).
    pub last_reason: String,
    /// Machine-readable error code of a failed run.
    pub error_code: Option<String>,
    /// Human-readable error message of a failed run.
    pub error_message: Option<String>,
}
/// Everything one finished run produced.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct PaperRunOutcome {
    /// Final task state.
    pub state: PaperTaskState,
    /// The full paper tree.
    pub tree: PaperTree,
    /// Ids of the adopted nodes, in adoption order.
    pub adopted: Vec<String>,
    /// Ids of the pruned/rejected nodes, in creation order.
    pub rejected: Vec<String>,
}
/// Grades one candidate paper.
///
/// Python parity: `tree_provider/judge.py::score_paper`, inverted into a trait so
/// the loop never constructs a model. [`RubricPaperScorer`] is the offline
/// implementation; a real judge drives the Python `PaperScorer` behind the same
/// trait.
#[async_trait]
pub trait PaperScorer: Send + Sync {
    /// Returns the scorer's human-readable name.
    fn name(&self) -> &str;
    /// Scores `candidate`, or fails if the paper cannot be graded.
    ///
    /// # Errors
    ///
    /// Implementations surface a missing model, a failed judge call, or an
    /// unparsable scoresheet here. The orchestrator turns a failure into a
    /// pruned `scoring_error` node instead of aborting the run.
    async fn score(&self, candidate: &PaperCandidate) -> Result<ScoreBreakdown>;
}
/// Aggregates a candidate's rubric samples with the default weights.
///
/// This is the deterministic half of Python's judge: no model call, only
/// [`aggregate`] over the samples the proposer attached.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RubricPaperScorer;
#[async_trait]
impl PaperScorer for RubricPaperScorer {
    // The Python trait returns `str`; `&str` here keeps model-driven judges
    // free to return a borrowed name.
    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        "rubric"
    }
    async fn score(&self, candidate: &PaperCandidate) -> Result<ScoreBreakdown> {
        aggregate(&candidate.samples)
    }
}
/// Drives the greedy-frontier paper tree.
///
/// Construct with [`PaperOptimizer::new`] and replace the collaborator seams with
/// the builder-style methods when a model-driven proposer or judge is available.
/// One iteration is observable through [`PaperOptimizer::run_iteration`]; the
/// whole run is [`PaperOptimizer::run`].
pub struct PaperOptimizer {
    config: PaperOptConfig,
    proposer: Arc<dyn PaperProposer>,
    scorer: Arc<dyn PaperScorer>,
    tree: PaperTree,
    state: PaperTaskState,
    cancelled: CancellationToken,
    seed_body: String,
    frontier_body: String,
    iterations_run: u32,
    adopted: Vec<String>,
    rejected: Vec<String>,
}
impl PaperOptimizer {
    /// Builds an optimizer with the template proposer and the rubric scorer.
    #[must_use]
    pub fn new(config: PaperOptConfig) -> Self {
        let max_iterations = config.max_iterations;
        Self {
            config,
            proposer: Arc::new(TemplatePaperProposer::new()),
            scorer: Arc::new(RubricPaperScorer),
            tree: PaperTree::new(),
            state: PaperTaskState {
                task_id: uuid_v4(),
                status: PaperStatus::Running,
                max_iterations,
                ..PaperTaskState::default()
            },
            cancelled: CancellationToken::new(),
            seed_body: String::new(),
            frontier_body: String::new(),
            iterations_run: 0,
            adopted: Vec::new(),
            rejected: Vec::new(),
        }
    }
    /// Replaces the proposer (a model-driven pipeline behind the same trait).
    #[must_use]
    pub fn with_proposer(mut self, proposer: Arc<dyn PaperProposer>) -> Self {
        self.proposer = proposer;
        self
    }
    /// Replaces the scorer (a real judge behind the same trait).
    #[must_use]
    pub fn with_scorer(mut self, scorer: Arc<dyn PaperScorer>) -> Self {
        self.scorer = scorer;
        self
    }
    /// Sets the task id (default a fresh `uuid_v4`).
    #[must_use]
    pub fn with_task_id(mut self, task_id: impl Into<String>) -> Self {
        self.state.task_id = task_id.into();
        self
    }
    /// Sets the body of the initial paper (the first frontier).
    #[must_use]
    pub fn with_seed_body(mut self, body: impl Into<String>) -> Self {
        let body = body.into();
        self.frontier_body.clone_from(&body);
        self.seed_body = body;
        self
    }
    /// Records the path of the compiled paper artifact.
    #[must_use]
    pub fn with_artifact_path(mut self, artifact_path: impl Into<String>) -> Self {
        self.state.artifact_path = artifact_path.into();
        self
    }
    /// Returns the configuration this optimizer runs with.
    #[must_use]
    pub fn config(&self) -> &PaperOptConfig {
        &self.config
    }
    /// Returns the current task state.
    #[must_use]
    pub fn state(&self) -> &PaperTaskState {
        &self.state
    }
    /// Returns the current paper tree.
    #[must_use]
    pub fn tree(&self) -> &PaperTree {
        &self.tree
    }
    /// Requests cancellation; the next await point ends the run as
    /// [`PaperStatus::Cancelled`].
    pub fn stop(&self) {
        self.cancelled.cancel();
    }
    /// Runs the whole task and returns the final state and tree.
    ///
    /// The loop performs iterations until the run reaches its iteration budget, a
    /// configured target composite, an empty proposal set, or cancellation.
    ///
    /// # Errors
    ///
    /// Returns the [`PaperProposer`] or tree failure that ended the run and
    /// records it in the state (`status = failed`, `error_code`,
    /// `error_message`). A scoring failure of one candidate is not an error.
    pub async fn run(&mut self) -> Result<PaperRunOutcome> {
        loop {
            match self.run_iteration().await {
                Ok(Some(_)) => {}
                Ok(None) => break,
                Err(error) => {
                    self.record_failure(&error);
                    return Err(error);
                }
            }
        }
        Ok(PaperRunOutcome {
            state: self.state.clone(),
            tree: self.tree.clone(),
            adopted: self.adopted.clone(),
            rejected: self.rejected.clone(),
        })
    }
    /// Performs exactly one iteration.
    ///
    /// Returns `Some(node_id)` when the iteration appended the round's winning
    /// node (adopted or pruned) and `None` when the run has already stopped
    /// (finished, failed, or cancelled).
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] for an invalid configuration, the proposer's
    /// error when it cannot propose, and [`Error::validation`] when a tree
    /// operation is given an unknown node. A single candidate's scoring failure is
    /// recorded as a pruned node, not returned.
    pub async fn run_iteration(&mut self) -> Result<Option<String>> {
        if let Err(error) = self.config.validate() {
            self.record_failure(&error);
            return Err(error);
        }
        if self.state.status != PaperStatus::Running {
            return Ok(None);
        }
        if self.cancelled.is_cancelled() {
            self.mark_cancelled();
            return Ok(None);
        }
        if self.iterations_run >= self.config.max_iterations {
            self.complete("max_iterations");
            return Ok(None);
        }
        let frontier = self.ensure_frontier();
        let parent_score = self.tree.node(&frontier).and_then(|node| node.score);
        let request = self.proposal_request(&frontier);
        let candidates = match self.proposer.propose(request).await {
            Ok(candidates) => candidates,
            Err(error) => {
                self.record_failure(&error);
                return Err(error);
            }
        };
        if self.cancelled.is_cancelled() {
            self.mark_cancelled();
            return Ok(None);
        }
        if candidates.is_empty() {
            self.complete("no_candidates");
            return Ok(None);
        }
        let mut scored: Vec<(PaperCandidate, ScoreBreakdown)> = Vec::new();
        let mut last_node: Option<String> = None;
        for candidate in candidates {
            match self.scorer.score(&candidate).await {
                Ok(score) => scored.push((candidate, score)),
                Err(error) => {
                    last_node = Some(self.record_scoring_error(&candidate, &frontier, &error)?);
                }
            }
            if self.cancelled.is_cancelled() {
                self.mark_cancelled();
                self.refresh_counts();
                return Ok(None);
            }
        }
        self.iterations_run = self.iterations_run.saturating_add(1);
        if scored.is_empty() {
            self.refresh_counts();
            if last_node.is_none() {
                self.complete("no_candidates");
                return Ok(None);
            }
            if self.iterations_run >= self.config.max_iterations {
                self.complete("max_iterations");
            }
            return Ok(last_node);
        }
        let winner = self.rank_winner(&scored)?;
        let (candidate, score) = scored.swap_remove(winner);
        self.apply_candidate(&candidate, score, parent_score, &frontier)
            .map(Some)
    }
    /// Ensures the root exists and returns the current frontier node id.
    fn ensure_frontier(&mut self) -> String {
        if self.tree.is_empty() {
            let root = self.tree.with_root("initial paper");
            self.state.frontier_node_id = Some(root.clone());
            root
        } else {
            self.state
                .frontier_node_id
                .clone()
                .unwrap_or_else(|| "n000".to_owned())
        }
    }
    /// Builds the proposal request for one round.
    fn proposal_request(&self, frontier: &str) -> PaperProposalRequest {
        PaperProposalRequest {
            iteration: self.iterations_run.saturating_add(1),
            frontier: Some(frontier.to_owned()),
            parent_body: self.frontier_body.clone(),
            count: self.config.beam_width,
            existing_ids: self
                .tree
                .nodes()
                .iter()
                .map(|node| node.node_id.clone())
                .collect(),
        }
    }
    /// Records a candidate whose scoring failed as a pruned `scoring_error` node.
    fn record_scoring_error(
        &mut self,
        candidate: &PaperCandidate,
        parent_id: &str,
        error: &Error,
    ) -> Result<String> {
        tracing::warn!(
            target: "openjiuwen_rsi::paper_opt::orchestrator",
            task_id = %self.state.task_id,
            candidate = %candidate.candidate_id,
            %error,
            "paper candidate could not be scored; pruning it"
        );
        let node = self.tree.add_candidate(
            Some(parent_id),
            candidate.summary.clone(),
            candidate.changes.clone(),
        )?;
        self.tree.set_outcome(
            &node,
            false,
            PaperNodeType::Pruned,
            Some("scoring_error".to_owned()),
        )?;
        self.rejected.push(node.clone());
        self.state.attempts_since_last_adoption =
            self.state.attempts_since_last_adoption.saturating_add(1);
        "scoring_error".clone_into(&mut self.state.last_reason);
        Ok(node)
    }
    /// Appends the round winner, applies the adoption gate, and checks the stop
    /// conditions.
    fn apply_candidate(
        &mut self,
        candidate: &PaperCandidate,
        score: ScoreBreakdown,
        parent_score: Option<ScoreBreakdown>,
        parent_id: &str,
    ) -> Result<String> {
        let outcome = adoption_gate(
            parent_score.as_ref(),
            &score,
            candidate.required_artifact_present || !self.config.require_artifact,
            None,
            &candidate.pipeline_status,
        );
        let node_id = self.tree.add_candidate(
            Some(parent_id),
            candidate.summary.clone(),
            candidate.changes.clone(),
        )?;
        self.tree.record_score(&node_id, score)?;
        match outcome {
            AdoptionOutcome::Adopted => {
                self.tree
                    .set_outcome(&node_id, true, PaperNodeType::Reporting, None)?;
                if self.state.baseline.is_none() {
                    self.state.baseline =
                        Some(parent_score.map_or(score.composite, |parent| parent.composite));
                }
                self.state.best_node_id = Some(node_id.clone());
                self.state.score = Some(score.composite);
                self.state.frontier_node_id = Some(node_id.clone());
                self.frontier_body.clone_from(&candidate.body);
                self.state.attempts_since_last_adoption = 0;
                self.state.last_reason.clear();
                self.adopted.push(node_id.clone());
            }
            other => {
                let failure = other.failure_class().unwrap_or("unknown").to_owned();
                self.tree.set_outcome(
                    &node_id,
                    false,
                    PaperNodeType::Pruned,
                    Some(failure.clone()),
                )?;
                self.state.attempts_since_last_adoption =
                    self.state.attempts_since_last_adoption.saturating_add(1);
                self.state.last_reason = failure;
                self.rejected.push(node_id.clone());
            }
        }
        self.refresh_counts();
        if let Some(target) = self.config.min_composite {
            if self.state.score.is_some_and(|score| score >= target) {
                self.complete("min_composite_reached");
                return Ok(node_id);
            }
        }
        if self.iterations_run >= self.config.max_iterations {
            self.complete("max_iterations");
        }
        Ok(node_id)
    }
    /// Ranks the scored candidates with a per-round `PUCT` tree and returns the
    /// index of the winner in `scored`.
    ///
    /// The frontier is added as an *unscored* root so it anchors ancestry and
    /// iteration labels without competing: scoring it would let the incumbent win
    /// a round in which every candidate regressed, and the gate would then never
    /// report `rejected_by_score`. Candidate order is preserved, and
    /// [`PuctTree::best`](crate::artifact_rsi::program_opt::PuctTree::best) breaks
    /// a tie towards the first proposal.
    fn rank_winner(&self, scored: &[(PaperCandidate, ScoreBreakdown)]) -> Result<usize> {
        let config = ProgramOptConfig {
            max_expansions: self.config.beam_width,
            ..ProgramOptConfig::default()
        };
        let mut ranking = PuctTree::new(&config);
        let root_program = Program {
            program_id: self
                .state
                .frontier_node_id
                .clone()
                .unwrap_or_else(|| "root".to_owned()),
            code: self.frontier_body.clone(),
            ..Program::default()
        };
        let root = ranking.add_node(None, root_program)?;
        let mut winner = 0_usize;
        let mut winner_score = f64::NEG_INFINITY;
        for (position, (candidate, score)) in scored.iter().enumerate() {
            let program = Program {
                program_id: candidate.candidate_id.clone(),
                parent_id: Some(self.state.frontier_node_id.clone().unwrap_or_default()),
                code: candidate.body.clone(),
                change_summary: candidate.summary.clone(),
                metrics: BTreeMap::from([("composite".to_owned(), score.composite)]),
                valid: true,
                error: None,
                ..Program::default()
            };
            let index = ranking.add_node(Some(root), program)?;
            ranking.record_score(index, Some(score.composite))?;
            if score.composite > winner_score {
                winner = position;
                winner_score = score.composite;
            }
        }
        Ok(winner)
    }
    /// Copies derived counters onto the observable state.
    fn refresh_counts(&mut self) {
        self.state.node_count = self.tree.len().saturating_sub(1);
        self.state.max_iterations = self.config.max_iterations;
    }
    /// Marks the run completed, keeping a more specific reason when one is set.
    fn complete(&mut self, reason: &str) {
        self.state.status = PaperStatus::Completed;
        if self.state.last_reason.is_empty() {
            reason.clone_into(&mut self.state.last_reason);
        }
    }
    /// Marks the run cancelled.
    fn mark_cancelled(&mut self) {
        self.state.status = PaperStatus::Cancelled;
        if self.state.last_reason.is_empty() {
            "cancelled".clone_into(&mut self.state.last_reason);
        }
    }
    /// Marks the run failed and records the error.
    fn record_failure(&mut self, error: &Error) {
        self.state.status = PaperStatus::Failed;
        self.state.error_code = Some(error.kind().as_str().to_owned());
        self.state.error_message = Some(error.to_string());
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn state_round_trips_through_json() {
        let state = PaperTaskState {
            task_id: "task".to_owned(),
            status: PaperStatus::Completed,
            max_iterations: 3,
            artifact_path: "paper/main.pdf".to_owned(),
            best_node_id: Some("n002".to_owned()),
            score: Some(7.5),
            baseline: Some(6.0),
            frontier_node_id: Some("n002".to_owned()),
            node_count: 2,
            attempts_since_last_adoption: 1,
            last_reason: "rejected_by_score".to_owned(),
            error_code: None,
            error_message: None,
        };
        let encoded = serde_json::to_value(&state).expect("serialize");
        let decoded: PaperTaskState = serde_json::from_value(encoded).expect("deserialize");
        assert_eq!(decoded, state);
        assert_eq!(decoded.status, PaperStatus::Completed);
    }
    #[test]
    fn new_records_the_iteration_budget_and_a_task_id() {
        let optimizer = PaperOptimizer::new(PaperOptConfig {
            max_iterations: 3,
            ..PaperOptConfig::default()
        });
        assert_eq!(optimizer.state().max_iterations, 3);
        assert_eq!(optimizer.state().status, PaperStatus::Running);
        assert_ne!(optimizer.state().task_id, "");
        assert!(optimizer.tree().is_empty());
    }
}
