//! The `PUCT` node list: a flat tree of candidate programs.
//!
//! Python parity: [`Node`] and [`PuctTree`] are `tree.Node` and `tree.PuctTree`
//! in `openjiuwen/rsi/artifact_rsi/program_opt/tree.py` (v0.1.20, `Node` at line
//! 53, `PuctTree` at line 89); the selection rule itself is `FlatPuct` in
//! `agentdescent/selection.py` (line 479), which that module delegates to.
//!
//! Three properties of the Python tree are deliberate and are preserved here:
//!
//! 1. **Flat.** Every node is a selection candidate, always. There is no descent
//!    from the root, so a good node at depth 1 competes with one at depth 9.
//! 2. **Ranked, not scored.** The exploitation term is the node's *rank* among
//!    all nodes, normalised to `[0, 1]`, which makes `c_puct` independent of the
//!    metric's units.
//! 3. **The visit is reserved at selection.** `select` increments the chosen
//!    node and every ancestor before the expansion runs — the parallel virtual
//!    loss. With one proposal in flight this is upstream's serial loop.
//!
//! Deviations from Python (intentional, documented):
//!
//! - [`PuctTree::add_node`] reports a missing parent or a reached
//!   `candidate_limit` as [`Error::Validation`]; Python silently reparents an
//!   unknown parent to the root and does not enforce the limit inside
//!   `add_node`.
//! - The tree does not take a lock. Each caller owns its tree, and Rust's
//!   borrow checker replaces `PuctTree._lock`.
//! - A node's score is recorded after the expansion ([`PuctTree::record_score`])
//!   rather than passed to `add_node`; a failed candidate is still a node, so
//!   `None` (Python `-inf`) is a value and never an omission.
//!
//! ```
//! use openjiuwen_rsi::artifact_rsi::program_opt::{Program, ProgramOptConfig, PuctTree};
//!
//! let config = ProgramOptConfig::default();
//! let mut tree = PuctTree::new(&config);
//! let root = tree
//!     .add_node(None, Program { program_id: "seed".to_owned(), ..Program::default() })
//!     .expect("empty tree accepts a root");
//! assert_eq!(root, 0);
//! tree.record_score(root, Some(0.4)).expect("root exists");
//! assert_eq!(tree.select(), Some(0));
//! assert_eq!(tree.root().expect("root").num_visits, 1);
//! ```
use crate::artifact_rsi::program_opt::config::ProgramOptConfig;
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
/// One node's payload: the candidate program plus where it came from and how it
/// did.
///
/// Python parity: `Program` in `program.py` line 196, with `error` as
/// `Option<String>` (Python uses `""`) so "no error" and "an empty error" are
/// not the same value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct Program {
    /// Stable identity of the candidate, normally a content hash of `code`.
    pub program_id: String,
    /// Iteration label the proposal carried (Python `Program.iteration`).
    pub iteration: u32,
    /// `program_id` of the node this candidate was mutated from.
    pub parent_id: Option<String>,
    /// The candidate source (or serialised file tree).
    pub code: String,
    /// One-line description of what changed against the parent.
    pub change_summary: String,
    /// Measured metrics, keyed by name (Python `Dict[str, Any]` narrowed to
    /// numbers because the portable half only ranks and reports them).
    pub metrics: BTreeMap<String, f64>,
    /// Whether the candidate passed the pre-evaluation gate; a candidate that
    /// did not is still a node and scores `-inf`-equivalent (`None`).
    pub valid: bool,
    /// Why the candidate is invalid, when it is.
    pub error: Option<String>,
}
impl Default for Program {
    fn default() -> Self {
        Self {
            program_id: String::new(),
            iteration: 0,
            parent_id: None,
            code: String::new(),
            change_summary: String::new(),
            metrics: BTreeMap::new(),
            valid: true,
            error: None,
        }
    }
}
/// One entry in the flat `PUCT` tree.
///
/// Python parity: `tree.Node` (line 53). `score` is `Option<f64>` where Python
/// uses `-inf` for a failed candidate: the distinction is kept because serde has
/// no `-inf`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct Node {
    /// Position in [`PuctTree::nodes`]; assigned by [`PuctTree::add_node`].
    pub index: usize,
    /// Index of the node this one was expanded from; `None` for the root.
    pub parent_index: Option<usize>,
    /// The candidate this node holds.
    pub program: Program,
    /// Evaluation score; `None` means not scored, which ranks last.
    pub score: Option<f64>,
    /// Subtree visit count: selections of this node plus selections of every
    /// descendant (Python `num_visits`).
    pub num_visits: u64,
    /// The model's own rating of how far this direction could go after tuning —
    /// the prior `P(s, a)` when `prior_exponent > 0`; deliberately not the score.
    pub promise: Option<f64>,
}
/// A flat `PUCT` tree of candidate programs, ranked by rank score plus
/// visit-count exploration.
///
/// Python parity: `tree.PuctTree` (line 89). The formula, verbatim from
/// `FlatPuct` (`selection.py` lines 489-509, implemented at 526-599):
///
/// ```text
/// rank_score_i = rank_i / (N - 1)          # 0.5 for every node when N <= 1
/// puct_i       = rank_score_i + c_puct * P_i * sqrt(sum_visits) / (1 + visits_i)
/// ```
///
/// where `rank_i` is the position of node `i` in an ascending, stable sort of
/// all node scores (`None` sorts as `-inf`), `N` is the number of nodes,
/// `visits_i` is [`Node::num_visits`], `sum_visits` is their total, and `P_i` is
/// the prior: `1/N` uniformly when `prior_exponent <= 0` or no node has a usable
/// promise, otherwise `promise_i^prior_exponent` renormalised to sum to one,
/// with unrated nodes taking the mean of the rated ones.
///
/// Scores are never `NaN` for a valid tree: `c_puct` and `prior_exponent` are
/// validated by [`ProgramOptConfig::validate`], visits are finite counts, and an
/// unusable prior total falls back to the uniform prior. An all-equal tree is
/// therefore a reached-by-ties case, and [`PuctTree::select`] breaks ties by
/// taking the first maximum — the lowest index, matching Python's stable
/// `argmax`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct PuctTree {
    /// Exploration constant of the `PUCT` formula.
    pub c_puct: f64,
    /// Exponent applied to a node's promise when building the prior.
    pub prior_exponent: f64,
    /// Maximum number of nodes; `None` is unbounded.
    pub candidate_limit: Option<usize>,
    /// Nodes in insertion order; `nodes[0]` is the root once one exists.
    pub nodes: Vec<Node>,
    /// Label the next added node will carry.
    pub next_iteration: u32,
}
impl Default for PuctTree {
    fn default() -> Self {
        Self {
            c_puct: 1.0,
            prior_exponent: 0.0,
            candidate_limit: None,
            nodes: Vec::new(),
            next_iteration: 1,
        }
    }
}
impl PuctTree {
    /// Builds an empty tree carrying the search knobs of `config`.
    ///
    /// Python parity: `PuctTree(c_puct=…, prior_exponent=…, candidate_limit=…)`.
    /// Only `c_puct`, `prior_exponent` and `candidate_limit` are copied; budgets,
    /// workers, the entrypoint and the reply format stay with the run.
    ///
    /// This constructor does not validate: it cannot fail, and a caller that
    /// wants the configuration checked should use [`ProgramOptConfig::to_tree`].
    #[must_use]
    pub fn new(config: &ProgramOptConfig) -> Self {
        Self {
            c_puct: config.c_puct,
            prior_exponent: config.prior_exponent,
            candidate_limit: config.candidate_limit,
            nodes: Vec::new(),
            next_iteration: 1,
        }
    }
    /// Builds a tree whose root is `program`, using the default search knobs
    /// (`c_puct = 1.0`, `prior_exponent = 0.0`, no candidate limit).
    ///
    /// Python parity: `PuctTree.seed` appends the root at index 0 with no
    /// parent and no score, and — as there — the root does not consume an
    /// iteration label: [`PuctTree::next_iteration`] stays `1`, so the first
    /// expansion is labelled 1. For non-default knobs, build with
    /// [`PuctTree::new`] and add the root with `add_node(None, program)`.
    #[must_use]
    pub fn with_root(program: Program) -> Self {
        let mut tree = Self::default();
        tree.nodes.push(Node {
            index: 0,
            parent_index: None,
            program,
            score: None,
            num_visits: 0,
            promise: None,
        });
        tree
    }
    /// Rebuilds a tree from previously serialised nodes.
    ///
    /// The restored [`PuctTree::next_iteration`] is one past the highest
    /// [`Program::iteration`] among `nodes` (and `1` for an empty list), so a
    /// resumed search never reuses a label — the property Python's
    /// `resume_at`/`_next_iteration` exists to preserve.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the knobs are not finite
    /// non-negative numbers, when `nodes` are not stored at their own
    /// `index` starting from 0, when a `parent_index` does not point at an
    /// earlier node, or when `candidate_limit` is smaller than the node count.
    pub fn from_nodes(
        c_puct: f64,
        prior_exponent: f64,
        candidate_limit: Option<usize>,
        nodes: Vec<Node>,
    ) -> Result<Self> {
        if !c_puct.is_finite() || c_puct < 0.0 {
            return Err(Error::validation(format!(
                "program_opt: c_puct must be a finite non-negative number, got {c_puct}"
            )));
        }
        if !prior_exponent.is_finite() || prior_exponent < 0.0 {
            return Err(Error::validation(format!(
                "program_opt: prior_exponent must be a finite non-negative number, got \
                 {prior_exponent}"
            )));
        }
        for (position, node) in nodes.iter().enumerate() {
            if node.index != position {
                return Err(Error::validation(format!(
                    "program_opt: node at position {position} has index {}, expected {position}",
                    node.index
                )));
            }
            if let Some(parent) = node.parent_index {
                if parent >= position {
                    return Err(Error::validation(format!(
                        "program_opt: node {position} points at parent {parent}, which is not an \
                         earlier node"
                    )));
                }
            }
        }
        if let Some(limit) = candidate_limit {
            if limit < nodes.len() {
                return Err(Error::validation(format!(
                    "program_opt: candidate_limit {limit} is smaller than the restored node count \
                     {}",
                    nodes.len()
                )));
            }
        }
        let next_iteration = nodes
            .iter()
            .map(|node| node.program.iteration)
            .max()
            .map_or(1, |highest| highest.saturating_add(1));
        Ok(Self {
            c_puct,
            prior_exponent,
            candidate_limit,
            nodes,
            next_iteration,
        })
    }
    /// The nodes, in insertion order.
    #[must_use]
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }
    /// Number of nodes in the tree.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }
    /// Whether the tree holds no node at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
    /// The root node, when the tree has been seeded.
    #[must_use]
    pub fn root(&self) -> Option<&Node> {
        self.nodes.first()
    }
    /// The label the next added node will carry.
    #[must_use]
    pub const fn next_iteration(&self) -> u32 {
        self.next_iteration
    }
    /// Whether the tree has reached its `candidate_limit`.
    ///
    /// A reached limit is compared against the node count (Python's limit is
    /// compared against the iteration label instead; the Rust tree counts what
    /// it actually holds).
    #[must_use]
    pub fn candidate_limit_reached(&self) -> bool {
        self.candidate_limit
            .is_some_and(|limit| self.nodes.len() >= limit)
    }
    /// Appends an expansion and returns its index.
    ///
    /// The node gets `index = nodes.len()`, `program.iteration =
    /// next_iteration` (which is then incremented), no score and no visits. When
    /// `parent_index` resolves and `program.parent_id` is `None`, the parent's
    /// `program_id` is recorded so the payload and the tree agree on ancestry.
    ///
    /// `parent_index = None` on an empty tree creates the root. On a non-empty
    /// tree it attaches to the root, matching Python's `add_node`, which coerces
    /// a `None` parent to index 0 (`tree.py` line 174).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the candidate limit is already reached
    /// (checked first, so a full tree reports the limit even when the parent is
    /// also unknown) or when `parent_index` names a node this tree does not
    /// hold.
    pub fn add_node(&mut self, parent_index: Option<usize>, mut program: Program) -> Result<usize> {
        if self.candidate_limit_reached() {
            return Err(Error::validation(format!(
                "program_opt: candidate limit {} reached, refusing to add node {}",
                self.candidate_limit.unwrap_or(0),
                self.nodes.len()
            )));
        }
        let parent = match parent_index {
            None if self.nodes.is_empty() => None,
            None => Some(0),
            Some(index) => {
                if index >= self.nodes.len() {
                    return Err(Error::validation(format!(
                        "program_opt: parent node {index} does not exist, tree holds {} nodes",
                        self.nodes.len()
                    )));
                }
                Some(index)
            }
        };
        if let Some(parent) = parent {
            if program.parent_id.is_none() {
                program.parent_id = Some(self.nodes[parent].program.program_id.clone());
            }
        }
        let index = self.nodes.len();
        program.iteration = self.next_iteration;
        self.next_iteration = self.next_iteration.saturating_add(1);
        self.nodes.push(Node {
            index,
            parent_index: parent,
            program,
            score: None,
            num_visits: 0,
            promise: None,
        });
        Ok(index)
    }
    /// Records the evaluation score of `index`; `None` marks it unscored.
    ///
    /// Python passes the score to `add_node`; splitting the two keeps a failed
    /// evaluation ([`Program::valid`] `false`, score `None`) a node with a
    /// recorded reason.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `index` is not a node of this tree.
    pub fn record_score(&mut self, index: usize, score: Option<f64>) -> Result<()> {
        let count = self.nodes.len();
        let node = self.nodes.get_mut(index).ok_or_else(|| {
            Error::validation(format!(
                "program_opt: node {index} does not exist, tree holds {count} nodes"
            ))
        })?;
        node.score = score;
        Ok(())
    }
    /// Records one additional visit on `index` and every ancestor.
    ///
    /// This is upstream's `futs.backpropagate_visit` (`tree.py` line 140): one
    /// visit for the node, one for its parent, and so on to the root. `index`
    /// outside the tree is a no-op, and a parent chain that somehow loops stops
    /// at the first repeated node instead of spinning.
    pub fn record_visits(&mut self, index: usize) {
        let mut seen: Vec<usize> = Vec::new();
        let mut cursor = Some(index);
        while let Some(current) = cursor {
            if current >= self.nodes.len() || seen.contains(&current) {
                break;
            }
            seen.push(current);
            let node = &mut self.nodes[current];
            node.num_visits = node.num_visits.saturating_add(1);
            cursor = node.parent_index;
        }
    }
    /// Picks the next node to expand, reserving the visit on it and its
    /// ancestors, and returns its index.
    ///
    /// Python parity: `PuctTree.select_parent` (line 146) over
    /// `FlatPuct.select` (line 572). The chosen index is the first maximum of
    /// [`PuctTree::puct_scores`]; ties therefore go to the lowest index, and an
    /// empty tree yields `None`.
    pub fn select(&mut self) -> Option<usize> {
        let scores = self.puct_scores();
        if scores.is_empty() {
            return None;
        }
        let mut best_index = 0_usize;
        let mut best_score = f64::NEG_INFINITY;
        for (index, score) in scores.iter().enumerate() {
            if *score > best_score {
                best_index = index;
                best_score = *score;
            }
        }
        self.record_visits(best_index);
        Some(best_index)
    }
    /// The `PUCT` score of every node, in node order.
    ///
    /// `puct_i = rank_score_i + c_puct * P_i * sqrt(sum_visits) / (1 + visits_i)`.
    /// A single node scores `0.5` with a prior of `1.0` (Python's `select`
    /// short-circuits at `N <= 1`, but the exposed score keeps the formula's
    /// documented values so callers can inspect it).
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // visit counts are exact integers in f64 practice
    pub fn puct_scores(&self) -> Vec<f64> {
        if self.nodes.is_empty() {
            return Vec::new();
        }
        let ranks = self.rank_scores();
        let priors = self.priors();
        let visits: Vec<f64> = self
            .nodes
            .iter()
            .map(|node| node.num_visits as f64)
            .collect();
        let total_visits: f64 = visits.iter().sum();
        let exploration_scale = total_visits.sqrt();
        ranks
            .iter()
            .zip(priors.iter())
            .zip(visits.iter())
            .map(|((rank, prior), visit)| {
                rank + self.c_puct * prior * exploration_scale / (1.0 + visit)
            })
            .collect()
    }
    /// The prior `P_i` of every node, in node order.
    ///
    /// Uniform `1/N` when `prior_exponent <= 0` (the upstream default) or when no
    /// node carries a usable promise. Otherwise every promise is raised to
    /// `prior_exponent` and normalised to sum to one; an unrated node takes the
    /// mean promise of the rated ones, so a missing number never bars a direction
    /// from being explored. A promise counts as usable when it is `Some`, finite
    /// and positive — the same filter as `FlatPuct._priors`
    /// (`selection.py` line 546).
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // node counts are exact integers in f64 practice
    pub fn priors(&self) -> Vec<f64> {
        let count = self.nodes.len();
        if count == 0 {
            return Vec::new();
        }
        let uniform = 1.0 / count as f64;
        if self.prior_exponent <= 0.0 {
            return vec![uniform; count];
        }
        let rated: Vec<f64> = self
            .nodes
            .iter()
            .filter_map(|node| usable_promise(node.promise))
            .collect();
        if rated.is_empty() {
            return vec![uniform; count];
        }
        let fallback = rated.iter().sum::<f64>() / rated.len() as f64;
        let raw: Vec<f64> = self
            .nodes
            .iter()
            .map(|node| {
                usable_promise(node.promise)
                    .unwrap_or(fallback)
                    .powf(self.prior_exponent)
            })
            .collect();
        let total: f64 = raw.iter().sum();
        if !total.is_finite() || total <= 0.0 {
            return vec![uniform; count];
        }
        raw.into_iter().map(|value| value / total).collect()
    }
    /// The rank score `rank_i / (N - 1)` of every node, in node order.
    ///
    /// Ranks come from an ascending, stable sort of the node scores, so a tie
    /// gives the lower index the lower rank. `None` sorts as `-inf` (Python ranks
    /// an unscored candidate last rather than first); a `NaN` score sorts after
    /// every real score, deterministically. `N <= 1` gives every node `0.5`,
    /// which is upstream's single-node case.
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // node counts are exact integers in f64 practice
    pub fn rank_scores(&self) -> Vec<f64> {
        let count = self.nodes.len();
        if count == 0 {
            return Vec::new();
        }
        if count == 1 {
            return vec![0.5];
        }
        let mut order: Vec<usize> = (0..count).collect();
        order.sort_by(|&left, &right| {
            ranking_key(self.nodes[left].score).total_cmp(&ranking_key(self.nodes[right].score))
        });
        let denominator = (count - 1) as f64;
        let mut ranks = vec![0.0_f64; count];
        for (rank, &index) in order.iter().enumerate() {
            ranks[index] = rank as f64 / denominator;
        }
        ranks
    }
    /// The best node: highest score, smallest index on ties.
    ///
    /// Unscored nodes are treated as `-inf` and can never win against a scored
    /// one. A tree in which nothing has been scored yet returns the root — the
    /// incumbent every later candidate is compared against — and an empty tree
    /// returns `None`.
    #[must_use]
    pub fn best(&self) -> Option<&Node> {
        let mut best: Option<&Node> = None;
        for node in &self.nodes {
            let Some(score) = node.score else {
                continue;
            };
            match best {
                None => best = Some(node),
                Some(current) => {
                    if score > current.score.unwrap_or(f64::NEG_INFINITY) {
                        best = Some(node);
                    }
                }
            }
        }
        best.or_else(|| self.root())
    }
}
/// The score a node is ranked by: unscored is `-inf`.
fn ranking_key(score: Option<f64>) -> f64 {
    score.unwrap_or(f64::NEG_INFINITY)
}
/// A promise the prior may use: present, finite and positive.
fn usable_promise(promise: Option<f64>) -> Option<f64> {
    promise.filter(|value| value.is_finite() && *value > 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;

    fn program(id: &str) -> Program {
        Program {
            program_id: id.to_owned(),
            ..Program::default()
        }
    }

    fn node(
        index: usize,
        parent_index: Option<usize>,
        score: Option<f64>,
        num_visits: u64,
        promise: Option<f64>,
    ) -> Node {
        Node {
            index,
            parent_index,
            program: Program {
                program_id: format!("p{index}"),
                iteration: u32::try_from(index).expect("small index"),
                ..Program::default()
            },
            score,
            num_visits,
            promise,
        }
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-12,
            "expected {expected}, got {actual}"
        );
    }

    /// Python parity: `FlatPuct._rank_scores` + `select` (`agentdescent/selection.py`
    /// lines 561-599), `c_puct = 1.0`, uniform priors.
    #[test]
    fn hand_computed_puct_scores() {
        // Scores 0.5 / 0.2 / 0.9 → ascending stable ranks 1 / 0 / 2 out of N = 3,
        // so rank scores 0.5 / 0.0 / 1.0. Visits 2 / 1 / 0 sum to 3.
        let tree = PuctTree::from_nodes(
            1.0,
            0.0,
            None,
            vec![
                node(0, None, Some(0.5), 2, None),
                node(1, Some(0), Some(0.2), 1, None),
                node(2, Some(0), Some(0.9), 0, None),
            ],
        )
        .expect("valid tree");

        assert_eq!(tree.rank_scores(), vec![0.5, 0.0, 1.0]);
        assert_close(tree.priors()[0], 1.0 / 3.0);
        assert_close(tree.priors()[1], 1.0 / 3.0);
        assert_close(tree.priors()[2], 1.0 / 3.0);

        let root_of_three = 3.0_f64.sqrt();
        let scores = tree.puct_scores();
        // 0.5 + 1.0 * (1/3) * sqrt(3) / (1 + 2)
        assert_close(scores[0], 0.5 + root_of_three / 3.0 / 3.0);
        // 0.0 + 1.0 * (1/3) * sqrt(3) / (1 + 1)
        assert_close(scores[1], root_of_three / 3.0 / 2.0);
        // 1.0 + 1.0 * (1/3) * sqrt(3) / (1 + 0)
        assert_close(scores[2], 1.0 + root_of_three / 3.0);
    }

    /// Python parity: the single-node `0.5` rank (`selection.py` lines 561-570) and
    /// the exposed score formula (`0.5 + c * (1/1) * sqrt(0) / (1 + 0)`).
    #[test]
    fn rank_score_edges() {
        let mut single =
            PuctTree::from_nodes(1.0, 0.0, None, vec![node(0, None, Some(0.3), 0, None)])
                .expect("valid tree");
        assert_eq!(single.rank_scores(), vec![0.5]);
        assert_close(single.priors()[0], 1.0);
        assert_close(single.puct_scores()[0], 0.5);
        assert_eq!(single.select(), Some(0));

        // An unscored node ranks last: None is -inf, so it takes rank 0 of 2.
        let mixed = PuctTree::from_nodes(
            1.0,
            0.0,
            None,
            vec![
                node(0, None, Some(0.5), 0, None),
                node(1, Some(0), None, 0, None),
            ],
        )
        .expect("valid tree");
        assert_eq!(mixed.rank_scores(), vec![1.0, 0.0]);
    }

    #[test]
    fn select_empty_tree_returns_none() {
        let mut tree = PuctTree::default();
        assert_eq!(tree.select(), None);
        assert_eq!(tree.best(), None);
        assert_eq!(tree.puct_scores(), Vec::<f64>::new());
        assert_eq!(tree.priors(), Vec::<f64>::new());
        assert_eq!(tree.rank_scores(), Vec::<f64>::new());
    }

    /// Python parity: `_backpropagate_locked` (`tree.py` line 140) — the chosen node
    /// and every ancestor.
    #[test]
    fn select_increments_chosen_node_and_ancestors() {
        let mut tree = PuctTree::from_nodes(
            1.0,
            0.0,
            None,
            vec![
                node(0, None, Some(1.0), 0, None),
                node(1, Some(0), Some(1.0), 0, None),
                node(2, Some(1), Some(1.0), 0, None),
            ],
        )
        .expect("valid tree");
        // Equal scores rank by index (0.0 / 0.5 / 1.0) and total visits are 0, so the
        // exploration term vanishes and index 2 wins.
        assert_eq!(tree.select(), Some(2));
        let visits: Vec<u64> = tree.nodes().iter().map(|entry| entry.num_visits).collect();
        assert_eq!(visits, vec![1, 1, 1]);
        // A second selection still favours the same branch; the whole chain is +1.
        assert_eq!(tree.select(), Some(2));
        let visits: Vec<u64> = tree.nodes().iter().map(|entry| entry.num_visits).collect();
        assert_eq!(visits, vec![2, 2, 2]);
        // Selecting an unknown index is a no-op rather than a panic.
        tree.record_visits(99);
        let visits: Vec<u64> = tree.nodes().iter().map(|entry| entry.num_visits).collect();
        assert_eq!(visits, vec![2, 2, 2]);
    }

    #[test]
    fn add_node_rejects_unknown_parent() {
        let mut tree = PuctTree::with_root(program("seed"));
        let error = tree
            .add_node(Some(7), program("child"))
            .expect_err("unknown parent");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("parent node 7"));
        assert_eq!(tree.len(), 1);
    }

    /// Python `candidate_limit` is the run's cap on expansions; here the tree
    /// refuses to grow past it with `Error::Validation`.
    #[test]
    fn add_node_rejects_reached_candidate_limit() {
        let config = ProgramOptConfig {
            candidate_limit: Some(2),
            ..ProgramOptConfig::default()
        };
        let mut tree = PuctTree::new(&config);
        assert!(!tree.candidate_limit_reached());
        assert_eq!(tree.add_node(None, program("seed")).expect("root"), 0);
        assert_eq!(tree.add_node(Some(0), program("child")).expect("child"), 1);
        assert!(tree.candidate_limit_reached());
        let error = tree
            .add_node(Some(0), program("rejected"))
            .expect_err("limit reached");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("candidate limit 2"));
        assert_eq!(tree.len(), 2);
        // The limit is checked before the parent, so a full tree reports the limit
        // even when the parent is also unknown.
        let error = tree
            .add_node(Some(99), program("rejected"))
            .expect_err("limit reached");
        assert!(error.to_string().contains("candidate limit 2"));

        let unbounded = PuctTree::with_root(program("seed"));
        assert!(!unbounded.candidate_limit_reached());
    }

    #[test]
    fn add_node_assigns_index_iteration_and_parent_id() {
        let mut tree = PuctTree::with_root(program("seed"));
        assert_eq!(tree.next_iteration(), 1);
        assert_eq!(tree.add_node(Some(0), program("child")).expect("child"), 1);
        assert_eq!(
            tree.add_node(Some(1), program("grandchild"))
                .expect("grandchild"),
            2
        );
        assert_eq!(tree.next_iteration(), 3);

        let first = &tree.nodes()[1];
        assert_eq!(first.index, 1);
        assert_eq!(first.parent_index, Some(0));
        assert_eq!(first.program.iteration, 1);
        assert_eq!(first.program.parent_id.as_deref(), Some("seed"));
        assert_eq!(first.num_visits, 0);
        assert_eq!(first.score, None);

        let second = &tree.nodes()[2];
        assert_eq!(second.program.iteration, 2);
        assert_eq!(second.program.parent_id.as_deref(), Some("child"));

        // A caller-supplied parent_id is preserved.
        let mut kept = program("kept");
        kept.parent_id = Some("explicit".to_owned());
        let index = tree.add_node(Some(0), kept).expect("child");
        assert_eq!(
            tree.nodes()[index].program.parent_id.as_deref(),
            Some("explicit")
        );
        // None attaches to the root on a non-empty tree (Python coerces it to 0).
        let index = tree.add_node(None, program("attached")).expect("attached");
        assert_eq!(tree.nodes()[index].parent_index, Some(0));
    }

    #[test]
    fn best_prefers_highest_score_then_lowest_index() {
        let tree = PuctTree::from_nodes(
            1.0,
            0.0,
            None,
            vec![
                node(0, None, Some(0.5), 0, None),
                node(1, Some(0), Some(0.9), 0, None),
                node(2, Some(0), Some(0.9), 0, None),
                node(3, Some(0), None, 0, None),
            ],
        )
        .expect("valid tree");
        assert_eq!(tree.best().map(|entry| entry.index), Some(1));

        let unscored = PuctTree::from_nodes(
            1.0,
            0.0,
            None,
            vec![
                node(0, None, None, 0, None),
                node(1, Some(0), Some(0.2), 0, None),
            ],
        )
        .expect("valid tree");
        assert_eq!(unscored.best().map(|entry| entry.index), Some(1));

        let seedless = PuctTree::with_root(program("seed"));
        assert_eq!(seedless.best().map(|entry| entry.index), Some(0));
    }

    #[test]
    fn from_nodes_rejects_bad_structure() {
        let cases: Vec<(f64, f64, Option<usize>, Vec<Node>)> = vec![
            (-1.0, 0.0, None, vec![]),
            (1.0, f64::NAN, None, vec![]),
            (1.0, 0.0, None, vec![node(1, None, None, 0, None)]),
            (1.0, 0.0, None, vec![node(0, Some(0), None, 0, None)]),
            (
                1.0,
                0.0,
                None,
                vec![
                    node(0, None, None, 0, None),
                    node(1, Some(5), None, 0, None),
                ],
            ),
            (
                1.0,
                0.0,
                Some(1),
                vec![
                    node(0, None, None, 0, None),
                    node(1, Some(0), None, 0, None),
                ],
            ),
        ];
        for (c_puct, prior_exponent, limit, nodes) in cases {
            let error = PuctTree::from_nodes(c_puct, prior_exponent, limit, nodes)
                .expect_err("rejected structure");
            assert_eq!(error.kind(), ErrorKind::Validation);
        }
    }

    #[test]
    fn from_nodes_restores_next_iteration() {
        let mut high = node(0, None, None, 0, None);
        high.program.iteration = 7;
        let tree = PuctTree::from_nodes(1.0, 0.0, Some(5), vec![high]).expect("valid tree");
        assert_eq!(tree.next_iteration(), 8);
        assert_eq!(tree.candidate_limit, Some(5));
        assert!(PuctTree::from_nodes(1.0, 0.0, None, vec![])
            .expect("valid empty tree")
            .is_empty());
        assert_eq!(
            PuctTree::from_nodes(1.0, 0.0, None, vec![])
                .expect("valid empty tree")
                .next_iteration(),
            1
        );
    }

    #[test]
    fn record_score_rejects_unknown_index() {
        let mut tree = PuctTree::with_root(program("seed"));
        tree.record_score(0, Some(0.7)).expect("root exists");
        assert_eq!(tree.best().and_then(|entry| entry.score), Some(0.7));
        let error = tree.record_score(3, Some(0.1)).expect_err("unknown index");
        assert_eq!(error.kind(), ErrorKind::Validation);
    }

    #[test]
    fn prior_exponent_weights_promises_and_falls_back_to_uniform() {
        let nodes = vec![
            node(0, None, None, 0, Some(2.0)),
            node(1, Some(0), None, 0, None),
            node(2, Some(0), None, 0, Some(4.0)),
        ];
        let weighted = PuctTree::from_nodes(1.0, 2.0, None, nodes.clone()).expect("valid tree");
        // Rated 2.0 and 4.0: unrated takes the mean 3.0; raw 4 / 9 / 16, total 29.
        let priors = weighted.priors();
        assert_close(priors[0], 4.0 / 29.0);
        assert_close(priors[1], 9.0 / 29.0);
        assert_close(priors[2], 16.0 / 29.0);
        assert_close(priors.iter().sum::<f64>(), 1.0);

        // prior_exponent <= 0 ignores promises entirely.
        let uniform = PuctTree::from_nodes(1.0, 0.0, None, nodes.clone()).expect("valid tree");
        assert_eq!(uniform.priors(), vec![1.0 / 3.0; 3]);

        // No usable promise at all is the uniform case too, whatever the exponent.
        let unrated = PuctTree::from_nodes(
            1.0,
            2.0,
            None,
            vec![
                node(0, None, None, 0, None),
                node(1, Some(0), None, 0, Some(0.0)),
            ],
        )
        .expect("valid tree");
        assert_eq!(unrated.priors(), vec![0.5, 0.5]);
    }

    #[test]
    fn with_root_keeps_the_program_and_starts_unscored() {
        let seed = Program {
            program_id: "seed".to_owned(),
            iteration: 4,
            code: "print(1)".to_owned(),
            ..Program::default()
        };
        let tree = PuctTree::with_root(seed.clone());
        assert_eq!(tree.len(), 1);
        let root = tree.root().expect("root");
        assert_eq!(root.index, 0);
        assert_eq!(root.parent_index, None);
        assert_eq!(root.program, seed);
        assert_eq!(root.score, None);
        assert_eq!(root.num_visits, 0);
        assert_eq!(tree.next_iteration(), 1);
    }

    #[test]
    fn serde_round_trips() {
        let tree = PuctTree::from_nodes(
            1.5,
            2.0,
            Some(9),
            vec![
                Node {
                    promise: Some(3.0),
                    score: Some(0.25),
                    ..node(0, None, None, 2, None)
                },
                Node {
                    promise: None,
                    ..node(1, Some(0), Some(0.75), 1, None)
                },
            ],
        )
        .expect("valid tree");
        let encoded = serde_json::to_value(&tree).expect("serialize");
        let decoded: PuctTree = serde_json::from_value(encoded).expect("deserialize");
        assert_eq!(decoded, tree);
        assert_eq!(decoded.nodes()[0].promise, Some(3.0));

        // Unknown fields are tolerated at every level; missing fields default.
        let document = json!({
            "c_puct": 1.25,
            "nodes": [{"program": {"program_id": "seed"}, "unknown": 1}],
            "unknown": "drop me",
        });
        let decoded: PuctTree = serde_json::from_value(document).expect("tolerant");
        assert!((decoded.c_puct - 1.25).abs() < f64::EPSILON);
        assert!(decoded.prior_exponent.abs() < f64::EPSILON);
        assert_eq!(decoded.candidate_limit, None);
        assert_eq!(decoded.next_iteration, 1);
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded.nodes()[0].program.program_id, "seed");
        assert!(decoded.nodes()[0].program.valid);
        assert_eq!(decoded.nodes()[0].program.iteration, 0);
        assert_eq!(decoded.nodes()[0].score, None);
        assert_eq!(decoded.nodes()[0].num_visits, 0);

        let plain: Program = serde_json::from_value(json!({})).expect("default program");
        assert_eq!(plain, Program::default());
        assert!(plain.valid);

        let invalid: Program = serde_json::from_value(json!({"valid": false, "error": "boom"}))
            .expect("explicit invalid program");
        assert!(!invalid.valid);
        assert_eq!(invalid.error.as_deref(), Some("boom"));
        assert_eq!(
            serde_json::to_value(&invalid).expect("serialize")["error"],
            json!("boom")
        );
    }
}
