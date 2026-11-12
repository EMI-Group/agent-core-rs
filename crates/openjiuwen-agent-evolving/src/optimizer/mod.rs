//! Candidate generation, optimizer seam, deterministic grid baseline, best-so-far
//! tracking, and the LLM retry policy.
//!
//! This module ports the portable parts of Python `openjiuwen.agent_evolving.optimizer`
//! (the optimizer seam and its candidate records) and keeps the ML prompt optimisers
//! out of scope: [`GridSearchOptimizer`] provides a deterministic baseline that never
//! calls a model, so the trainer can run offline.
//!
//! Checkpoint persistence lives in `checkpoint` and is re-exported here.
//!
//! # Example
//!
//! A grid search over two parameters (the last dimension varies fastest):
//!
//! ```
//! use openjiuwen_agent_evolving::optimizer::{ParameterSpace, select_best};
//!
//! let space = ParameterSpace::new()
//!     .with_dimension("temperature", vec![0.2.into(), 0.7.into()])
//!     .with_dimension("top_k", vec![10.into(), 50.into()]);
//! assert_eq!(space.len(), 4);
//!
//! let mut candidates = space.grid(4);
//! assert_eq!(candidates.len(), 4);
//! assert_eq!(candidates[0].id, "cand-0");
//! assert_eq!(candidates[1].parameters["top_k"].as_i64(), Some(50));
//!
//! for (index, candidate) in candidates.iter_mut().enumerate() {
//!     candidate.score = Some(index as f64 / 10.0);
//! }
//! assert_eq!(select_best(&candidates).unwrap().score, Some(0.3));
//! ```

mod checkpoint;

pub use checkpoint::{
    CheckpointBest, CheckpointManager, CheckpointSnapshot, CheckpointStep, CheckpointStore,
    DefaultCheckpointManager, EvolveCheckpoint, FileCheckpointStore, MemoryCheckpointStore,
    RestoredState,
};

use std::collections::BTreeMap;

use async_trait::async_trait;
use openjiuwen_common::error::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Upper bound on the candidates a [`GridSearchOptimizer`] materializes up front.
///
/// A grid can describe an enormous product (e.g. five dimensions with ten values
/// each); materializing it eagerly would exhaust memory. Candidate `propose` calls
/// therefore see at most this many points, and the cap is documented rather than
/// hidden inside the optimizer.
pub const MAX_MATERIALIZED_CANDIDATES: usize = 100_000;

/// A single candidate configuration proposed by an [`Optimizer`].
///
/// Field names mirror Python's candidate record. `parameters` is the operator's
/// tunable state, [`Candidate::state`] projects it to the JSON object expected by a
/// state-restoring target.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Candidate {
    /// Stable identifier, `cand-{index}` for grid-generated candidates.
    pub id: String,
    /// Proposed parameter values keyed by parameter name.
    pub parameters: BTreeMap<String, Value>,
    /// Observed score; `None` until the candidate has been evaluated.
    #[serde(default)]
    pub score: Option<f64>,
    /// Optimizer-defined generation counter (grid candidates are generation `0`).
    #[serde(default)]
    pub generation: u64,
}

impl Candidate {
    /// Creates an unscored generation-`0` candidate.
    #[must_use]
    pub fn new(id: impl Into<String>, parameters: BTreeMap<String, Value>) -> Self {
        Self {
            id: id.into(),
            parameters,
            score: None,
            generation: 0,
        }
    }

    /// Projects the parameter map to the JSON object used by state-restoring targets.
    #[must_use]
    pub fn state(&self) -> Value {
        Value::Object(self.parameters.clone().into_iter().collect())
    }
}

/// The candidate-generation seam.
///
/// Implementations propose at most `limit` candidates per call and may track their
/// own best-so-far through [`Optimizer::observe`]. The default `observe`/`best`
/// implementations make a pure generator valid without any bookkeeping.
#[async_trait]
pub trait Optimizer: Send + Sync {
    /// Human-readable optimizer name (Python class-name analogue).
    fn name(&self) -> &str;

    /// Proposes up to `limit` candidates; an empty vector means exhaustion.
    ///
    /// # Errors
    ///
    /// Returns an error when candidate generation itself fails (for example a
    /// model-backed optimizer); the grid baseline never fails.
    async fn propose(&mut self, limit: usize) -> Result<Vec<Candidate>>;

    /// Records the observed score for `candidate`. The default is a no-op.
    fn observe(&mut self, candidate: &Candidate, score: f64) {
        let _ = (candidate, score);
    }

    /// Best candidate observed so far, if the optimizer tracks one.
    fn best(&self) -> Option<&Candidate> {
        None
    }
}

/// An ordered Cartesian parameter space.
///
/// Dimensions are visited in [`BTreeMap`] (lexicographic) key order and the last
/// dimension varies fastest, matching a nested loop written from first to last.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ParameterSpace {
    /// Candidate values per dimension.
    #[serde(default)]
    pub dimensions: BTreeMap<String, Vec<Value>>,
}

impl ParameterSpace {
    /// Creates an empty space.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds (or replaces) one dimension.
    #[must_use]
    pub fn with_dimension(mut self, name: impl Into<String>, values: Vec<Value>) -> Self {
        self.dimensions.insert(name.into(), values);
        self
    }

    /// The dimension map.
    #[must_use]
    pub fn dimensions(&self) -> &BTreeMap<String, Vec<Value>> {
        &self.dimensions
    }

    /// Whether the space is degenerate: no dimensions, or at least one empty list.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.dimensions.is_empty() || self.dimensions.values().any(Vec::is_empty)
    }

    /// Cartesian-product size, saturating at [`usize::MAX`]; `0` when empty.
    #[must_use]
    pub fn len(&self) -> usize {
        if self.is_empty() {
            return 0;
        }
        self.dimensions
            .values()
            .fold(1_usize, |acc, values| acc.saturating_mul(values.len()))
    }

    /// Materializes the first `limit` grid points deterministically.
    ///
    /// Ids are `cand-{index}` (zero-based), `generation` is `0`, and `score` is
    /// `None`. `limit == 0` and an empty space both yield an empty vector.
    #[must_use]
    pub fn grid(&self, limit: usize) -> Vec<Candidate> {
        if limit == 0 || self.is_empty() {
            return Vec::new();
        }
        let dimensions: Vec<(&String, &Vec<Value>)> = self.dimensions.iter().collect();
        let count = self.len().min(limit);
        let mut candidates = Vec::with_capacity(count);
        for index in 0..count {
            let mut remainder = index;
            let mut parameters = BTreeMap::new();
            // Last dimension fastest: take the least-significant digit first.
            for (name, values) in dimensions.iter().rev() {
                let position = remainder % values.len();
                remainder /= values.len();
                parameters.insert((*name).clone(), values[position].clone());
            }
            candidates.push(Candidate::new(format!("cand-{index}"), parameters));
        }
        candidates
    }
}

/// Deterministic grid-search baseline implementing [`Optimizer`].
///
/// The full grid is materialized at construction (up to
/// [`MAX_MATERIALIZED_CANDIDATES`]), then handed out in chunks; the optimizer also
/// tracks its own best observed candidate so the trainer needs no ML dependency.
#[derive(Debug, Clone, Default)]
pub struct GridSearchOptimizer {
    candidates: Vec<Candidate>,
    cursor: usize,
    best: Option<Candidate>,
}

impl GridSearchOptimizer {
    /// Materializes `space` (capped at [`MAX_MATERIALIZED_CANDIDATES`]) as candidates.
    #[must_use]
    pub fn new(space: ParameterSpace) -> Self {
        let limit = space.len().min(MAX_MATERIALIZED_CANDIDATES);
        Self::from_points(space.grid(limit))
    }

    /// Builds an optimizer from an explicit candidate list.
    #[must_use]
    pub fn from_points(points: Vec<Candidate>) -> Self {
        Self {
            candidates: points,
            cursor: 0,
            best: None,
        }
    }

    /// Number of candidates not yet proposed.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.candidates.len().saturating_sub(self.cursor)
    }

    /// The materialized candidate list.
    #[must_use]
    pub fn candidates(&self) -> &[Candidate] {
        &self.candidates
    }
}

#[async_trait]
impl Optimizer for GridSearchOptimizer {
    fn name(&self) -> &'static str {
        "grid_search"
    }

    async fn propose(&mut self, limit: usize) -> Result<Vec<Candidate>> {
        if limit == 0 || self.cursor >= self.candidates.len() {
            return Ok(Vec::new());
        }
        let end = self.cursor.saturating_add(limit).min(self.candidates.len());
        let chunk = self.candidates[self.cursor..end].to_vec();
        self.cursor = end;
        Ok(chunk)
    }

    fn observe(&mut self, candidate: &Candidate, score: f64) {
        if !score.is_finite() {
            return;
        }
        let improved = self
            .best
            .as_ref()
            .is_none_or(|best| best.score.is_some_and(|best_score| score > best_score));
        if improved {
            let mut scored = candidate.clone();
            scored.score = Some(score);
            self.best = Some(scored);
        }
    }

    fn best(&self) -> Option<&Candidate> {
        self.best.as_ref()
    }
}

/// Best-so-far tracking used by the trainer and optimizer.
///
/// Only strictly greater scores replace the incumbent, so the first of equal scores
/// wins (deterministic tie-breaking).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct BestSoFar {
    /// Best score observed, if any.
    #[serde(default)]
    pub score: Option<f64>,
    /// Id of the candidate that produced [`BestSoFar::score`].
    #[serde(default)]
    pub candidate_id: Option<String>,
}

impl BestSoFar {
    /// Creates empty tracking state.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a scored candidate; unscored candidates are ignored.
    pub fn observe(&mut self, candidate: &Candidate) {
        if let Some(score) = candidate.score {
            self.consider(candidate.id.clone(), score);
        }
    }

    /// Records `score` for candidate `id` when it improves the incumbent.
    pub fn consider(&mut self, id: impl Into<String>, score: f64) {
        if self.is_improved(score) {
            self.score = Some(score);
            self.candidate_id = Some(id.into());
        }
    }

    /// Whether `score` would replace the incumbent (no incumbent, or strictly greater).
    #[must_use]
    pub fn is_improved(&self, score: f64) -> bool {
        self.score.is_none_or(|best| score > best)
    }

    /// Best score observed so far.
    #[must_use]
    pub fn score(&self) -> Option<f64> {
        self.score
    }

    /// Id of the candidate holding the best score.
    #[must_use]
    pub fn candidate_id(&self) -> Option<&str> {
        self.candidate_id.as_deref()
    }
}

/// Selects the candidate with the highest finite score; ties keep the first.
///
/// Returns `None` when no candidate has a finite score.
#[must_use]
pub fn select_best(candidates: &[Candidate]) -> Option<&Candidate> {
    let mut best: Option<(&Candidate, f64)> = None;
    for candidate in candidates {
        let Some(score) = candidate.score else {
            continue;
        };
        if !score.is_finite() {
            continue;
        }
        if best.is_none_or(|(_, best_score)| score > best_score) {
            best = Some((candidate, score));
        }
    }
    best.map(|(candidate, _)| candidate)
}

/// Pure-data retry policy for LLM invocations, with the Python defaults.
///
/// Python's `LlmInvokePolicy` defaults are: per-attempt timeout `60.0` seconds,
/// total budget `600.0` seconds, `2` attempts, backoff base `1.0` second, and
/// retrying empty responses enabled. The policy carries no timers or state; the
/// caller applies it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct LlmInvokePolicy {
    /// Timeout for a single attempt, in seconds.
    pub attempt_timeout_secs: f64,
    /// Total time budget across all attempts, in seconds.
    pub total_budget_secs: f64,
    /// Maximum number of attempts (including the first).
    pub max_attempts: u32,
    /// Base backoff delay between attempts, in seconds.
    pub backoff_base_secs: f64,
    /// Whether an empty response counts as a retryable failure.
    pub retry_empty_response: bool,
}

impl Default for LlmInvokePolicy {
    fn default() -> Self {
        Self {
            attempt_timeout_secs: 60.0,
            total_budget_secs: 600.0,
            max_attempts: 2,
            backoff_base_secs: 1.0,
            retry_empty_response: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn candidate(id: &str, score: Option<f64>) -> Candidate {
        Candidate {
            id: id.to_owned(),
            parameters: BTreeMap::new(),
            score,
            generation: 0,
        }
    }

    #[test]
    fn candidate_state_projects_parameters() {
        let mut parameters = BTreeMap::new();
        parameters.insert("temperature".to_owned(), json!(0.5));
        let candidate = Candidate::new("c1", parameters);
        assert_eq!(candidate.state(), json!({"temperature": 0.5}));
        assert_eq!(candidate.score, None);
        assert_eq!(candidate.generation, 0);
    }

    #[test]
    fn space_shape_and_grid_order() {
        let space = ParameterSpace::new()
            .with_dimension("a", vec![json!(1), json!(2)])
            .with_dimension("b", vec![json!("x"), json!("y"), json!("z")]);
        assert_eq!(space.len(), 6);
        let grid = space.grid(3);
        assert_eq!(grid.len(), 3);
        assert_eq!(grid[0].parameters["a"], json!(1));
        assert_eq!(grid[0].parameters["b"], json!("x"));
        assert_eq!(grid[1].parameters["b"], json!("y"));
        assert_eq!(grid[2].parameters["b"], json!("z"));
        assert_eq!(grid[0].id, "cand-0");
    }

    #[test]
    fn degenerate_space_is_empty() {
        assert!(ParameterSpace::new().is_empty());
        assert_eq!(ParameterSpace::new().len(), 0);
        let empty_dimension = ParameterSpace::new().with_dimension("a", Vec::<Value>::new());
        assert!(empty_dimension.is_empty());
        assert_eq!(empty_dimension.len(), 0);
        assert_eq!(empty_dimension.grid(5).len(), 0);
    }

    #[test]
    fn grid_respects_zero_limit() {
        let space = ParameterSpace::new().with_dimension("a", vec![json!(1)]);
        assert_eq!(space.grid(0).len(), 0);
    }

    #[test]
    fn space_length_saturates() {
        let space = ParameterSpace::new()
            .with_dimension("a", vec![json!(1); 8])
            .with_dimension("b", vec![json!(1); 8]);
        assert_eq!(space.len(), 64);
    }

    #[test]
    fn best_so_far_keeps_strict_improvement() {
        let mut best = BestSoFar::new();
        assert!(best.is_improved(0.5));
        best.consider("a", 0.5);
        assert!(!best.is_improved(0.5));
        best.consider("b", 0.5);
        assert_eq!(best.candidate_id(), Some("a"));
        best.consider("b", 0.7);
        assert_eq!(best.score(), Some(0.7));
        assert_eq!(best.candidate_id(), Some("b"));
        best.observe(&candidate("c", Some(0.6)));
        assert_eq!(best.candidate_id(), Some("b"));
        best.observe(&candidate("d", None));
        assert_eq!(best.candidate_id(), Some("b"));
    }

    #[test]
    fn select_best_skips_unscored_and_keeps_first_tie() {
        let candidates = vec![
            candidate("a", None),
            candidate("b", Some(0.4)),
            candidate("c", Some(0.9)),
            candidate("d", Some(0.9)),
        ];
        assert_eq!(select_best(&candidates).map(|c| c.id.as_str()), Some("c"));
        assert!(select_best(&candidates[..1]).is_none());
    }

    #[test]
    fn policy_defaults_match_python() {
        let policy = LlmInvokePolicy::default();
        assert_eq!(policy.attempt_timeout_secs, 60.0);
        assert_eq!(policy.total_budget_secs, 600.0);
        assert_eq!(policy.max_attempts, 2);
        assert_eq!(policy.backoff_base_secs, 1.0);
        assert!(policy.retry_empty_response);
    }

    #[tokio::test]
    async fn grid_optimizer_proposes_chunks_and_tracks_best() {
        let space = ParameterSpace::new()
            .with_dimension("a", vec![json!(1), json!(2)])
            .with_dimension("b", vec![json!(1), json!(2)]);
        let mut optimizer = GridSearchOptimizer::new(space);
        assert_eq!(optimizer.name(), "grid_search");
        assert_eq!(optimizer.remaining(), 4);

        let first = optimizer.propose(2).await.expect("propose");
        assert_eq!(first.len(), 2);
        optimizer.observe(&first[0], 0.3);
        optimizer.observe(&first[1], 0.3);
        assert_eq!(optimizer.best().map(|c| c.id.as_str()), Some("cand-0"));
        optimizer.observe(&first[1], 0.9);
        assert_eq!(optimizer.best().map(|c| c.id.as_str()), Some("cand-1"));

        let second = optimizer.propose(2).await.expect("propose");
        assert_eq!(second.len(), 2);
        assert_eq!(optimizer.remaining(), 0);
        assert_eq!(optimizer.propose(2).await.expect("propose").len(), 0);
    }

    #[tokio::test]
    async fn grid_optimizer_from_points_and_zero_limit() {
        let points = vec![candidate("x", Some(0.2)), candidate("y", Some(0.1))];
        let mut optimizer = GridSearchOptimizer::from_points(points);
        assert_eq!(optimizer.propose(0).await.expect("propose").len(), 0);
        assert_eq!(optimizer.propose(5).await.expect("propose").len(), 2);
        assert!(optimizer.best().is_none());
    }
}
