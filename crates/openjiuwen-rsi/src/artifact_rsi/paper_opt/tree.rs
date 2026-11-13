//! The tree of paper attempts and the strict adoption gate.
//!
//! Python parity: [`PaperNode`] is `tree_provider.schemas.RsiTreeNode`
//! (`type`, `adopted`, `score`, `summary`, `failure_class`, `changes`,
//! `extra`) and [`adoption_gate`] is the decision block of
//! `tree_provider/orchestrator.py::_build_node` (v0.1.20): a candidate must come
//! from a `complete` pipeline, carry its compiled paper, have been scored, and
//! strictly beat the frontier composite.
//!
//! Two Python fields are not ported here: the run-directory/artifact-id
//! bookkeeping (`snapshot_artifact_id`, `run_dir`) belongs to the host toolchain,
//! and the public display labels in `schemas._FAILURE_CLASS_LABELS` are text, not
//! logic — the failure classes themselves are preserved verbatim.
//!
//! The tree is an append-only list of nodes linked by `parent_id`. Ids are
//! sequential (`n000`, `n001`, …) and iteration labels start at `1` for the first
//! candidate; the root carries iteration `0`. Nothing here reads a clock or a
//! random source, so a run over the same proposals yields the same tree.
use super::scoring::ScoreBreakdown;
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
/// What a node represents in the paper tree.
///
/// Python parity: the `type` literal of `RsiTreeNode` (`root`, `reporting`,
/// `pruned`). A rejected candidate is recorded as [`PaperNodeType::Pruned`] with
/// `failure_class = "rejected_by_score"`, exactly as Python does.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PaperNodeType {
    /// The seed node; never scored and never adopted.
    #[default]
    Root,
    /// A candidate whose paper was generated and scored.
    Reporting,
    /// A candidate that produced no adoptable paper.
    Pruned,
}
/// One logical change a candidate made to the paper.
///
/// Python parity: `RsiChange` (`operation`, `function`, `target`, `summary`)
/// plus a `group` label the Rust tree uses to keep paper edits distinct from
/// future non-paper edits.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct PaperChange {
    /// Group the change belongs to (the template proposer writes `paper`).
    pub group: String,
    /// Operation performed (`revise`, `generate`, …).
    pub operation: String,
    /// Logical function or stage that made the change.
    pub function: String,
    /// Path or section the change applies to.
    pub target: String,
    /// One-line human-readable summary.
    pub summary: String,
}
/// One attempt in the paper tree.
///
/// Python parity: `RsiTreeNode`; `score` is the full [`ScoreBreakdown`] rather
/// than just the composite because the gate and the reports need both family
/// scores.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct PaperNode {
    /// Sequential node id (`n000`, `n001`, …).
    pub node_id: String,
    /// Iteration that produced the node; `0` for the root.
    pub iteration: u32,
    /// Id of the parent node, `None` for the root.
    pub parent_id: Option<String>,
    /// What the node represents.
    pub node_type: PaperNodeType,
    /// Whether the node became the frontier (never true for a pruned node).
    pub adopted: bool,
    /// The candidate's aggregate rubric score, when it was scored.
    pub score: Option<ScoreBreakdown>,
    /// Human-readable description of the attempt.
    pub summary: String,
    /// Why a pruned or rejected node did not advance the frontier.
    pub failure_class: Option<String>,
    /// Changes the candidate claims to have made.
    pub changes: Vec<PaperChange>,
    /// Caller annotations, never interpreted here.
    pub metadata: BTreeMap<String, Value>,
}
/// An append-only tree of paper attempts.
///
/// The root is created by [`PaperTree::with_root`]; every candidate is appended
/// by [`PaperTree::add_candidate`] and later annotated with
/// [`PaperTree::record_score`] and [`PaperTree::set_outcome`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct PaperTree {
    /// Nodes in insertion order; the root, once created, is `nodes[0]`.
    pub nodes: Vec<PaperNode>,
    /// Iteration label the next candidate will carry.
    pub next_iteration: u32,
}
impl PaperTree {
    /// Creates an empty tree.
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            next_iteration: 1,
        }
    }
    /// Appends the root node and returns its id.
    ///
    /// The root carries iteration `0`, is never adopted, and has no score. This
    /// method does not check for an existing root; callers that need one root
    /// check [`PaperTree::is_empty`] first (the orchestrator does).
    pub fn with_root(&mut self, summary: impl Into<String>) -> String {
        let node_id = self.next_node_id();
        self.nodes.push(PaperNode {
            node_id: node_id.clone(),
            iteration: 0,
            parent_id: None,
            node_type: PaperNodeType::Root,
            adopted: false,
            score: None,
            summary: summary.into(),
            failure_class: None,
            changes: Vec::new(),
            metadata: BTreeMap::new(),
        });
        node_id
    }
    /// Appends a candidate node under `parent_id` and returns its id.
    ///
    /// The node is appended unadopted and unscored with the next sequential id
    /// and iteration label; the caller fills those in with
    /// [`PaperTree::record_score`] and [`PaperTree::set_outcome`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] when `parent_id` names a node this tree does
    /// not hold.
    pub fn add_candidate(
        &mut self,
        parent_id: Option<&str>,
        summary: impl Into<String>,
        changes: Vec<PaperChange>,
    ) -> Result<String> {
        if let Some(parent) = parent_id {
            if self.node(parent).is_none() {
                return Err(Error::validation(format!(
                    "paper_opt: parent node {parent} does not exist, tree holds {} nodes",
                    self.nodes.len()
                )));
            }
        }
        let node_id = self.next_node_id();
        let iteration = self.next_iteration;
        self.next_iteration = self.next_iteration.saturating_add(1);
        self.nodes.push(PaperNode {
            node_id: node_id.clone(),
            iteration,
            parent_id: parent_id.map(str::to_owned),
            node_type: PaperNodeType::Reporting,
            adopted: false,
            score: None,
            summary: summary.into(),
            failure_class: None,
            changes,
            metadata: BTreeMap::new(),
        });
        Ok(node_id)
    }
    /// Records the aggregate score of `node_id`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] when `node_id` is not a node of this tree.
    pub fn record_score(&mut self, node_id: &str, score: ScoreBreakdown) -> Result<()> {
        let node = self.node_mut(node_id)?;
        node.score = Some(score);
        Ok(())
    }
    /// Records whether `node_id` was adopted, its node type and its failure class.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] when `node_id` is not a node of this tree.
    pub fn set_outcome(
        &mut self,
        node_id: &str,
        adopted: bool,
        node_type: PaperNodeType,
        failure_class: Option<String>,
    ) -> Result<()> {
        let node = self.node_mut(node_id)?;
        node.adopted = adopted;
        node.node_type = node_type;
        node.failure_class = failure_class;
        Ok(())
    }
    /// Returns the node with `node_id`, if present.
    #[must_use]
    pub fn node(&self, node_id: &str) -> Option<&PaperNode> {
        self.nodes.iter().find(|node| node.node_id == node_id)
    }
    /// Returns every node in insertion order.
    #[must_use]
    pub fn nodes(&self) -> &[PaperNode] {
        &self.nodes
    }
    /// Returns the number of nodes (the root included).
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }
    /// Reports whether the tree has no nodes at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
    /// Returns the highest-scoring node; ties go to the earliest node.
    ///
    /// Only scored nodes compete — the root and any unscored candidate never
    /// win. Returns `None` when nothing has been scored.
    #[must_use]
    pub fn best(&self) -> Option<&PaperNode> {
        let mut best: Option<&PaperNode> = None;
        for node in &self.nodes {
            let Some(score) = node.score else {
                continue;
            };
            match best {
                None => best = Some(node),
                Some(current) => {
                    if score.composite > current.score.map_or(f64::NEG_INFINITY, |s| s.composite) {
                        best = Some(node);
                    }
                }
            }
        }
        best
    }
    /// Returns the most recently adopted reporting node — the current frontier.
    ///
    /// The root is never a frontier: it carries [`PaperNodeType::Root`], so a
    /// fresh tree yields `None`.
    #[must_use]
    pub fn frontier(&self) -> Option<&PaperNode> {
        self.nodes
            .iter()
            .rev()
            .find(|node| node.adopted && node.node_type == PaperNodeType::Reporting)
    }
    /// Returns the root-to-node chain, root first.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] when `node_id` is unknown or the parent
    /// chain contains a cycle.
    pub fn path_to(&self, node_id: &str) -> Result<Vec<&PaperNode>> {
        let mut path = Vec::new();
        let mut seen: Vec<&str> = Vec::new();
        let mut cursor = Some(node_id);
        while let Some(current) = cursor {
            if seen.contains(&current) {
                return Err(Error::validation(format!(
                    "paper_opt: parent cycle at node {current}"
                )));
            }
            seen.push(current);
            let node = self.node(current).ok_or_else(|| {
                Error::validation(format!(
                    "paper_opt: node {current} does not exist, tree holds {} nodes",
                    self.nodes.len()
                ))
            })?;
            path.push(node);
            cursor = node.parent_id.as_deref();
        }
        path.reverse();
        Ok(path)
    }
    /// Iterates the pruned nodes in insertion order.
    pub fn pruned(&self) -> impl Iterator<Item = &PaperNode> + '_ {
        self.nodes
            .iter()
            .filter(|node| node.node_type == PaperNodeType::Pruned)
    }
    /// Counts the pruned nodes.
    #[must_use]
    pub fn pruned_count(&self) -> usize {
        self.pruned().count()
    }
    /// Returns the iteration label the next candidate will carry.
    #[must_use]
    pub fn next_iteration(&self) -> u32 {
        self.next_iteration
    }
    /// Formats the id of the node at position `index`.
    fn next_node_id(&self) -> String {
        format!("n{:03}", self.nodes.len())
    }
    /// Returns the mutable node with `node_id`.
    fn node_mut(&mut self, node_id: &str) -> Result<&mut PaperNode> {
        let count = self.nodes.len();
        self.nodes
            .iter_mut()
            .find(|node| node.node_id == node_id)
            .ok_or_else(|| {
                Error::validation(format!(
                    "paper_opt: node {node_id} does not exist, tree holds {count} nodes"
                ))
            })
    }
}
/// The decision the adoption gate returns for one candidate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "outcome")]
pub enum AdoptionOutcome {
    /// The candidate becomes the new frontier.
    Adopted,
    /// The candidate produced no adoptable paper.
    Pruned {
        /// Why the attempt was pruned (`pipeline_*`, `paper_generation`, `scoring_error`).
        failure_class: String,
    },
    /// The candidate was scored but did not strictly beat the frontier.
    Rejected {
        /// Always `rejected_by_score`.
        failure_class: String,
    },
}
impl AdoptionOutcome {
    /// Reports whether the candidate was adopted.
    #[must_use]
    pub fn is_adopted(&self) -> bool {
        matches!(self, Self::Adopted)
    }
    /// Returns the failure class of a non-adopted outcome.
    #[must_use]
    pub fn failure_class(&self) -> Option<&str> {
        match self {
            Self::Adopted => None,
            Self::Pruned { failure_class } | Self::Rejected { failure_class } => {
                Some(failure_class.as_str())
            }
        }
    }
}
/// Decides whether a scored candidate becomes the new frontier.
///
/// The check order is Python's `_build_node`, and it matters: a run that failed
/// is pruned before its (possibly empty) artifact is inspected, and an unscorable
/// candidate is pruned before any score comparison happens. Each failure class:
///
/// - `pipeline_<status>` — the paper pipeline did not reach `complete`
///   (`pipeline_failed`, `pipeline_blocked`, `pipeline_incomplete`, …).
/// - `paper_generation` — the pipeline completed but the compiled paper is
///   missing.
/// - `scoring_error` — the scorer failed; an unvetted paper must never win.
/// - `rejected_by_score` — the candidate was scored and did not strictly beat
///   the frontier (`>`; a tie is not evidence of improvement).
///
/// The first candidate of a task has no parent score and is always adopted once
/// it passes the three pruning checks. `required_artifact_present` is only
/// consulted when the caller enforces it; the orchestrator passes
/// `candidate.required_artifact_present || !config.require_artifact`.
#[must_use]
pub fn adoption_gate(
    parent_score: Option<&ScoreBreakdown>,
    candidate: &ScoreBreakdown,
    required_artifact_present: bool,
    scoring_error: Option<&str>,
    pipeline_status: &str,
) -> AdoptionOutcome {
    if pipeline_status != "complete" {
        return AdoptionOutcome::Pruned {
            failure_class: format!("pipeline_{pipeline_status}"),
        };
    }
    if !required_artifact_present {
        return AdoptionOutcome::Pruned {
            failure_class: "paper_generation".to_owned(),
        };
    }
    if scoring_error.is_some() {
        return AdoptionOutcome::Pruned {
            failure_class: "scoring_error".to_owned(),
        };
    }
    match parent_score {
        None => AdoptionOutcome::Adopted,
        Some(parent) if candidate.is_improvement_over(parent) => AdoptionOutcome::Adopted,
        Some(_) => AdoptionOutcome::Rejected {
            failure_class: "rejected_by_score".to_owned(),
        },
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn breakdown(composite: f64) -> ScoreBreakdown {
        ScoreBreakdown {
            general: composite,
            experiment: composite,
            composite,
            sample_count: 1,
        }
    }
    #[test]
    fn ids_and_iterations_are_sequential() {
        let mut tree = PaperTree::new();
        assert!(tree.is_empty());
        let root = tree.with_root("seed");
        assert_eq!(root, "n000");
        let first = tree
            .add_candidate(Some(&root), "first", Vec::new())
            .expect("add");
        let second = tree
            .add_candidate(Some(&first), "second", Vec::new())
            .expect("add");
        assert_eq!((first.as_str(), second.as_str()), ("n001", "n002"));
        assert_eq!(tree.node(&root).expect("root").iteration, 0);
        assert_eq!(tree.node(&first).expect("first").iteration, 1);
        assert_eq!(tree.node(&second).expect("second").iteration, 2);
        assert_eq!(tree.next_iteration(), 3);
        assert!(tree.best().is_none());
        assert!(tree.frontier().is_none());
    }
    #[test]
    fn unknown_parent_and_node_are_rejected() {
        let mut tree = PaperTree::new();
        tree.with_root("seed");
        let error = tree
            .add_candidate(Some("missing"), "x", Vec::new())
            .expect_err("unknown parent");
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
        let error = tree
            .record_score("missing", breakdown(1.0))
            .expect_err("unknown node");
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
    }
    #[test]
    fn best_breaks_ties_by_earliest_node_and_frontier_is_the_last_adopted() {
        let mut tree = PaperTree::new();
        let root = tree.with_root("seed");
        let first = tree
            .add_candidate(Some(&root), "first", Vec::new())
            .expect("add");
        let second = tree
            .add_candidate(Some(&root), "second", Vec::new())
            .expect("add");
        tree.record_score(&first, breakdown(7.0)).expect("score");
        tree.record_score(&second, breakdown(7.0)).expect("score");
        assert_eq!(tree.best().expect("best").node_id, first);
        tree.set_outcome(&first, true, PaperNodeType::Reporting, None)
            .expect("outcome");
        assert_eq!(tree.frontier().expect("frontier").node_id, first);
        tree.set_outcome(&second, true, PaperNodeType::Reporting, None)
            .expect("outcome");
        assert_eq!(tree.frontier().expect("frontier").node_id, second);
    }
    #[test]
    fn path_and_pruned_count() {
        let mut tree = PaperTree::new();
        let root = tree.with_root("seed");
        let first = tree
            .add_candidate(Some(&root), "first", Vec::new())
            .expect("add");
        let second = tree
            .add_candidate(Some(&first), "second", Vec::new())
            .expect("add");
        tree.set_outcome(
            &first,
            false,
            PaperNodeType::Pruned,
            Some("scoring_error".to_owned()),
        )
        .expect("outcome");
        let path: Vec<&str> = tree
            .path_to(&second)
            .expect("path")
            .into_iter()
            .map(|node| node.node_id.as_str())
            .collect();
        assert_eq!(path, vec![root.as_str(), first.as_str(), second.as_str()]);
        assert_eq!(tree.pruned_count(), 1);
        assert!(tree.pruned().any(|node| node.node_id == first));
    }
}
