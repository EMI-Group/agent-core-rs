//! The reward stage: a [`RewardSummary`] as diagnostic signals attached to the graph.
//!
//! Ports the signal half of Python `openjiuwen.agent_evolving.attribution.reward_stage`:
//! a reward total becomes one [`DiagnosticSignal`] per sink step, and signals that
//! carry a score can be folded back into per-node rewards for
//! [`CreditAssigner`](crate::attribution::CreditAssigner). The reward deep agent, the
//! payload schema and the browser evidence tools are deferred; `crate::reward` owns
//! the scoring.

use std::collections::BTreeMap;

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::credit::{is_zero, CreditAssigner};
use super::graph::AttributionGraph;
use super::model::{DiagnosticSignal, SignalSource};
use crate::reward::RewardSummary;

/// Bridges a [`RewardSummary`] and the attribution graph.
///
/// # Example
///
/// ```
/// use openjiuwen_agent_evolving::attribution::{
///     AttributionEdge, AttributionGraph, AttributionNode, CreditAssigner,
///     RewardStageAdapter,
/// };
/// use openjiuwen_agent_evolving::reward::{RewardPack, RewardSignal};
///
/// let graph = AttributionGraph::new(
///     vec![
///         AttributionNode::new("root", "plan", "message"),
///         AttributionNode::new("leaf", "reply", "message"),
///     ],
///     vec![AttributionEdge::new("root", "leaf")],
/// )
/// .expect("valid graph");
/// let summary = RewardPack::default_pack()
///     .aggregate(&[RewardSignal::new("correctness", 1.0)])
///     .expect("valid pack");
///
/// let adapter = RewardStageAdapter::new(CreditAssigner::new().with_normalise(false));
/// let signals = adapter
///     .signals_from_summary(&graph, &summary)
///     .expect("finite total");
/// let rewards = adapter.rewards_from_signals(&signals).expect("valid signals");
/// let report = adapter
///     .assigner()
///     .assign(&graph, &rewards)
///     .expect("known nodes");
/// assert!((report.credit("leaf") - 1.0).abs() < 1e-9);
/// assert!((report.credit("root") - 0.5).abs() < 1e-9);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct RewardStageAdapter {
    /// The assigner hosts should use for the rewards this adapter produces.
    pub assigner: CreditAssigner,
    /// The tier stamped on every reward signal; defaults to `"uncertain"`.
    #[serde(default = "default_tier")]
    pub tier: String,
}

/// The default reward-signal tier: the reward stage is evidence, not proof.
fn default_tier() -> String {
    "uncertain".to_owned()
}

impl RewardStageAdapter {
    /// Creates an adapter with the `"uncertain"` tier.
    #[must_use]
    pub fn new(assigner: CreditAssigner) -> Self {
        Self {
            assigner,
            tier: default_tier(),
        }
    }

    /// Sets the tier stamped on produced signals.
    #[must_use]
    pub fn with_tier(mut self, tier: impl Into<String>) -> Self {
        self.tier = tier.into();
        self
    }

    /// The assigner carried for convenience.
    #[must_use]
    pub fn assigner(&self) -> &CreditAssigner {
        &self.assigner
    }

    /// The tier stamped on produced signals.
    #[must_use]
    pub fn tier(&self) -> &str {
        &self.tier
    }

    /// Turns a reward total into one signal per sink step of `graph`.
    ///
    /// The total is spread equally over [`AttributionGraph::leaves`] — the steps
    /// whose outcome was observed — falling back to every node when the graph has no
    /// leaves. Signals keep node order, carry the id `reward-<node_id>`, source
    /// [`SignalSource::Reward`], the adapter tier, a confidence of `1.0`, and the
    /// metadata `{"total", "num_nodes"}` so a host can show the split. An empty graph
    /// produces no signals.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when the summary total is not finite.
    pub fn signals_from_summary(
        &self,
        graph: &AttributionGraph,
        summary: &RewardSummary,
    ) -> Result<Vec<DiagnosticSignal>> {
        let total = summary.total();
        if !total.is_finite() {
            return Err(Error::validation(format!(
                "reward total must be finite, got {total}"
            )));
        }
        let mut targets = graph.leaves();
        if targets.is_empty() {
            targets = graph.node_ids().collect();
        }
        let num_nodes = targets.len();
        if num_nodes == 0 {
            return Ok(Vec::new());
        }
        let share = total / num_nodes as f64;
        let mut metadata = BTreeMap::new();
        metadata.insert("total".to_owned(), Value::from(total));
        metadata.insert("num_nodes".to_owned(), Value::from(num_nodes));
        let mut signals = Vec::with_capacity(num_nodes);
        for node_id in targets {
            signals.push(
                DiagnosticSignal::new(
                    SignalSource::Reward,
                    format!("reward total {total:.3} distributed to node {node_id}"),
                )?
                .with_signal_id(format!("reward-{node_id}"))
                .with_tier(self.tier.clone())
                .with_node_ids(vec![node_id.to_owned()])
                .with_score(share)
                .with_confidence(1.0)
                .with_metadata(metadata.clone()),
            );
        }
        Ok(signals)
    }

    /// Folds scored signals back into per-node rewards.
    ///
    /// Every signal with a score contributes to every id in its
    /// [`node_ids`](DiagnosticSignal::node_ids); a node's reward is the
    /// confidence-weighted mean of the scores it received (confidence defaults to
    /// `1.0`, and a node whose confidences sum to zero takes the first score).
    /// Signals without a score or without node ids are skipped.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when a signal carries a non-finite score.
    pub fn rewards_from_signals(
        &self,
        signals: &[DiagnosticSignal],
    ) -> Result<BTreeMap<String, f64>> {
        let mut totals: BTreeMap<String, (f64, f64, f64)> = BTreeMap::new();
        for signal in signals {
            let Some(score) = signal.score else {
                continue;
            };
            if !score.is_finite() {
                return Err(Error::validation(format!(
                    "diagnostic signal `{}` carries a non-finite reward score",
                    signal.signal_id
                )));
            }
            if signal.node_ids.is_empty() {
                continue;
            }
            let confidence = signal
                .confidence
                .filter(|value| value.is_finite())
                .unwrap_or(1.0);
            for node_id in &signal.node_ids {
                let entry = totals.entry(node_id.clone()).or_insert((0.0, 0.0, score));
                entry.0 = score.mul_add(confidence, entry.0);
                entry.1 += confidence;
            }
        }
        let rewards = totals
            .into_iter()
            .map(|(node_id, (weighted, weight, first))| {
                let reward = if is_zero(weight) {
                    first
                } else {
                    weighted / weight
                };
                (node_id, reward)
            })
            .collect();
        Ok(rewards)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attribution::graph::{AttributionEdge, AttributionNode};

    fn two_leaves() -> AttributionGraph {
        AttributionGraph::new(
            vec![
                AttributionNode::new("root", "plan", "message"),
                AttributionNode::new("left", "tool-a", "tool"),
                AttributionNode::new("right", "tool-b", "tool"),
            ],
            vec![
                AttributionEdge::new("root", "left"),
                AttributionEdge::new("root", "right"),
            ],
        )
        .unwrap()
    }

    fn summary(total: f64) -> RewardSummary {
        RewardSummary {
            total,
            ..RewardSummary::zero()
        }
    }

    #[test]
    fn signals_split_the_total_over_leaves() {
        let adapter = RewardStageAdapter::new(CreditAssigner::new());
        let signals = adapter
            .signals_from_summary(&two_leaves(), &summary(0.9))
            .unwrap();
        let ids: Vec<&str> = signals
            .iter()
            .map(|signal| signal.signal_id.as_str())
            .collect();
        assert_eq!(ids, ["reward-left", "reward-right"]);
        for signal in &signals {
            assert_eq!(signal.source, SignalSource::Reward);
            assert_eq!(signal.tier, "uncertain");
            assert!((signal.score.unwrap() - 0.45).abs() < 1e-9);
            assert_eq!(signal.metadata["num_nodes"], Value::from(2));
        }
    }

    #[test]
    fn round_trip_returns_the_total() {
        let adapter = RewardStageAdapter::new(CreditAssigner::new());
        let signals = adapter
            .signals_from_summary(&two_leaves(), &summary(0.9))
            .unwrap();
        let rewards = adapter.rewards_from_signals(&signals).unwrap();
        assert_eq!(rewards.len(), 2);
        let sum: f64 = rewards.values().sum();
        assert!((sum - 0.9).abs() < 1e-9);

        let empty = AttributionGraph::default();
        assert_eq!(
            adapter
                .signals_from_summary(&empty, &summary(0.5))
                .unwrap()
                .len(),
            0
        );
        assert!(adapter
            .signals_from_summary(&two_leaves(), &summary(f64::NAN))
            .is_err());
    }
}
