//! Credit assignment: backward propagation of a run's reward to its causes.
//!
//! A run's outcome is observable at its *sink* steps (the final reply, the failing
//! tool call). Attribution answers "which earlier steps caused that?" by walking the
//! dependency graph **outward from the rewarded nodes to their ancestors** and
//! handing each cause a decaying share of the reward.
//!
//! Python leaves this to the deferred attribution deep agent (an LLM with ten
//! tools); the crate keeps the deterministic arithmetic so rollups are testable and
//! reproducible, and the LLM diagnosis stays deferred.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};

use super::graph::AttributionGraph;

/// Exact zero test by bit pattern, so no float equality comparison is needed.
///
/// Returns `true` for both `0.0` and `-0.0` (and `false` for `NaN`).
pub(super) fn is_zero(value: f64) -> bool {
    value.to_bits() << 1 == 0
}

/// The default [`CreditAssigner::decay`]: each hop keeps half the credit.
const DEFAULT_DECAY: f64 = 0.5;
/// The default [`CreditAssigner::max_depth`]: eight hops of ancestry.
const DEFAULT_MAX_DEPTH: usize = 8;

/// How one rewarded node's score spreads across the graph.
///
/// # Example
///
/// ```
/// use std::collections::BTreeMap;
///
/// use openjiuwen_agent_evolving::attribution::{
///     AttributionEdge, AttributionGraph, AttributionNode, CreditAssigner,
/// };
///
/// let graph = AttributionGraph::new(
///     vec![
///         AttributionNode::new("root", "plan", "message"),
///         AttributionNode::new("leaf", "reply", "message"),
///     ],
///     vec![AttributionEdge::new("root", "leaf")],
/// )
/// .expect("valid graph");
/// let assigner = CreditAssigner::new().with_normalise(false);
/// let report = assigner
///     .assign(&graph, &BTreeMap::from([("leaf".to_owned(), 1.0)]))
///     .expect("known node");
/// assert!((report.credit("leaf") - 1.0).abs() < 1e-9);
/// assert!((report.credit("root") - 0.5).abs() < 1e-9);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct CreditAssigner {
    /// The per-hop retention factor, in `0.0..=1.0`.
    pub decay: f64,
    /// Whether the propagated credits are rescaled to sum to the input reward.
    pub normalise: bool,
    /// The maximum number of hops a share travels (a direct parent is depth 1).
    pub max_depth: usize,
}

impl Default for CreditAssigner {
    fn default() -> Self {
        Self::new()
    }
}

impl CreditAssigner {
    /// Creates an assigner with `decay` `0.5`, `normalise` `true`, and
    /// `max_depth` `8`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            decay: DEFAULT_DECAY,
            normalise: true,
            max_depth: DEFAULT_MAX_DEPTH,
        }
    }

    /// Sets the per-hop retention factor.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when `decay` is not finite or outside `0.0..=1.0`.
    pub fn with_decay(mut self, decay: f64) -> Result<Self> {
        if !decay.is_finite() || !(0.0..=1.0).contains(&decay) {
            return Err(Error::validation(format!(
                "credit decay must be finite and within 0.0..=1.0, got {decay}"
            )));
        }
        self.decay = decay;
        Ok(self)
    }

    /// Sets whether credits are rescaled to sum to the input reward.
    #[must_use]
    pub fn with_normalise(mut self, normalise: bool) -> Self {
        self.normalise = normalise;
        self
    }

    /// Sets the maximum propagation depth (`0` credits only the rewarded nodes).
    #[must_use]
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// The per-hop retention factor.
    #[must_use]
    pub fn decay(&self) -> f64 {
        self.decay
    }

    /// Whether credits are rescaled to sum to the input reward.
    #[must_use]
    pub fn normalise(&self) -> bool {
        self.normalise
    }

    /// The maximum propagation depth.
    #[must_use]
    pub fn max_depth(&self) -> usize {
        self.max_depth
    }

    /// Spreads `rewards` from their nodes to their ancestors.
    ///
    /// Each rewarded node keeps its reward; then, per rewarded node, credit flows
    /// breadth-first to its direct parents (`reward * edge weight * decay`), their
    /// parents (`* decay` again) and so on, stopping after [`Self::max_depth`]
    /// hops. A node is reached at most once per rewarded node, by its shortest path,
    /// so diamond-shaped graphs never double-count one cause within one propagation
    /// and cycles terminate. Credit accumulates across rewarded nodes and across
    /// propagation paths. An edge with weight `0.0` contributes nothing and
    /// therefore blocks credit from passing through it.
    ///
    /// When [`Self::normalise`] is set and the summed input reward is non-zero, the
    /// per-node credits are scaled so they sum to that total, which keeps every
    /// `share` comparable across runs.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when a rewarded node id is not in `graph`, or a reward
    /// is not finite.
    pub fn assign(
        &self,
        graph: &AttributionGraph,
        rewards: &BTreeMap<String, f64>,
    ) -> Result<CreditReport> {
        for (node_id, reward) in rewards {
            if !graph.contains(node_id) {
                return Err(Error::validation(format!(
                    "reward references unknown attribution node `{node_id}`"
                )));
            }
            if !reward.is_finite() {
                return Err(Error::validation(format!(
                    "reward for attribution node `{node_id}` must be finite, got {reward}"
                )));
            }
        }
        let mut credits: BTreeMap<String, f64> = rewards.clone();
        for (node_id, reward) in rewards {
            self.propagate(graph, node_id, *reward, &mut credits);
        }
        let total: f64 = rewards.values().sum();
        let mut normalised = false;
        if self.normalise && !is_zero(total) {
            let sum: f64 = credits.values().sum();
            if !is_zero(sum) && sum.is_finite() {
                let factor = total / sum;
                for credit in credits.values_mut() {
                    *credit *= factor;
                }
                normalised = true;
            }
        }
        let ranking = build_ranking(&credits, total);
        Ok(CreditReport {
            per_node: credits,
            ranking,
            total,
            normalised,
        })
    }

    /// Propagates one rewarded node's credit to its ancestors, breadth-first.
    fn propagate(
        &self,
        graph: &AttributionGraph,
        seed_id: &str,
        seed_value: f64,
        credits: &mut BTreeMap<String, f64>,
    ) {
        let mut visited: BTreeSet<String> = BTreeSet::new();
        visited.insert(seed_id.to_owned());
        // `(node, hops from the seed, decay^hops)`.
        let mut queue: VecDeque<(String, usize, f64)> = VecDeque::new();
        queue.push_back((seed_id.to_owned(), 0, 1.0));
        while let Some((node_id, depth, multiplier)) = queue.pop_front() {
            if depth >= self.max_depth {
                continue;
            }
            for edge in &graph.edges {
                if edge.to != node_id || !visited.insert(edge.from.clone()) {
                    continue;
                }
                let contribution = seed_value * edge.weight * multiplier * self.decay;
                if !is_zero(contribution) {
                    *credits.entry(edge.from.clone()).or_insert(0.0) += contribution;
                }
                queue.push_back((edge.from.clone(), depth + 1, multiplier * self.decay));
            }
        }
    }
}

/// One row of the credit ranking.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct CreditRank {
    /// The attributed node id.
    #[serde(default)]
    pub node_id: String,
    /// The credit the node received.
    #[serde(default)]
    pub credit: f64,
    /// `credit / total`, or `0.0` when the total input reward is zero.
    #[serde(default)]
    pub share: f64,
}

/// The outcome of [`CreditAssigner::assign`].
///
/// [`per_node`](Self::per_node) holds every rewarded node plus every ancestor that
/// received a non-zero propagated share, so a node that merely appears in the graph
/// is absent rather than a zero row. [`ranking`](Self::ranking) sorts the same rows
/// by credit descending, then node id ascending.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct CreditReport {
    /// The credit per node, keyed and ordered by node id.
    #[serde(default)]
    pub per_node: BTreeMap<String, f64>,
    /// The descending credit ranking.
    #[serde(default)]
    pub ranking: Vec<CreditRank>,
    /// The summed input reward that was distributed.
    #[serde(default)]
    pub total: f64,
    /// Whether the credits were actually rescaled (normalisation was requested and
    /// the summed input reward was non-zero).
    #[serde(default)]
    pub normalised: bool,
}

impl CreditReport {
    /// The credit of one node; `0.0` when the node received none.
    #[must_use]
    pub fn credit(&self, node_id: &str) -> f64 {
        self.per_node.get(node_id).copied().unwrap_or_default()
    }

    /// The first `n` rows of the ranking (all of them when `n` exceeds its length).
    #[must_use]
    pub fn top(&self, n: usize) -> &[CreditRank] {
        &self.ranking[..n.min(self.ranking.len())]
    }
}

/// Builds the descending ranking with shares against `total`.
fn build_ranking(credits: &BTreeMap<String, f64>, total: f64) -> Vec<CreditRank> {
    let mut ranking: Vec<CreditRank> = credits
        .iter()
        .map(|(node_id, credit)| CreditRank {
            node_id: node_id.clone(),
            credit: *credit,
            share: if is_zero(total) { 0.0 } else { credit / total },
        })
        .collect();
    ranking.sort_by(|left, right| {
        right
            .credit
            .total_cmp(&left.credit)
            .then_with(|| left.node_id.cmp(&right.node_id))
    });
    ranking
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attribution::graph::{AttributionEdge, AttributionNode};

    fn chain() -> AttributionGraph {
        AttributionGraph::new(
            vec![
                AttributionNode::new("root", "plan", "message"),
                AttributionNode::new("mid", "work", "tool"),
                AttributionNode::new("leaf", "reply", "message"),
            ],
            vec![
                AttributionEdge::new("root", "mid"),
                AttributionEdge::new("mid", "leaf"),
            ],
        )
        .unwrap()
    }

    #[test]
    fn chain_propagates_with_decay() {
        let report = CreditAssigner::new()
            .with_normalise(false)
            .assign(&chain(), &BTreeMap::from([("leaf".to_owned(), 1.0)]))
            .unwrap();
        assert!((report.credit("leaf") - 1.0).abs() < 1e-9);
        assert!((report.credit("mid") - 0.5).abs() < 1e-9);
        assert!((report.credit("root") - 0.25).abs() < 1e-9);
        assert!((report.total - 1.0).abs() < 1e-9);
        assert!(!report.normalised);
        assert_eq!(
            report
                .ranking
                .iter()
                .map(|rank| rank.node_id.as_str())
                .collect::<Vec<_>>(),
            ["leaf", "mid", "root"]
        );
        assert!((report.top(1)[0].share - 1.0).abs() < 1e-9);
    }

    #[test]
    fn depth_cutoff_and_validation() {
        let report = CreditAssigner::new()
            .with_normalise(false)
            .with_max_depth(1)
            .assign(&chain(), &BTreeMap::from([("leaf".to_owned(), 1.0)]))
            .unwrap();
        assert!((report.credit("mid") - 0.5).abs() < 1e-9);
        assert!(is_zero(report.credit("root")));

        assert!(CreditAssigner::new().with_decay(1.5).is_err());
        assert!(CreditAssigner::new().with_decay(f64::NAN).is_err());
        let assigner = CreditAssigner::new();
        assert!(assigner
            .assign(&chain(), &BTreeMap::from([("nope".to_owned(), 1.0)]))
            .is_err());
        assert!(assigner
            .assign(&chain(), &BTreeMap::from([("leaf".to_owned(), f64::NAN)]))
            .is_err());
        assert_eq!(CreditAssigner::default(), CreditAssigner::new());
    }
}
