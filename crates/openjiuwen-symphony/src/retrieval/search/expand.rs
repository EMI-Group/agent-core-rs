//! The default target expander: selected targets become leaves and child cursors.
//!
//! Ports Python `openjiuwen.symphony.retrieval.search.runtime.expand` (v0.1.20).
//!
//! A selected target is one of two things:
//!
//! - a **terminal** resolution carrying an [`RetrieverItem`] — its item becomes a ranked
//!   leaf [`RetrieverCandidate`], emitted in the selection's declared order and ranked
//!   `1` (the reduce stage re-ranks nothing; rank is positional metadata);
//! - a **branch** resolution naming a node — it becomes a
//!   [`ChildSearchCursor`] one level deeper, carrying the resolution's branch path and a
//!   per-branch budget so one branch cannot claim the whole `top_k`.
//!
//! # The per-branch budget
//!
//! Python's rule is `min(top_k, ceil(top_k / branches) + branch_candidate_slack)`, with
//! `branches = max(1, selected.len())` and a floor of `1`. The crate keeps that formula
//! and adds its `top_k == 0` sentinel: an unlimited request yields an unlimited branch
//! budget (`0`), instead of Python's `1`.
//!
//! # Rust-only bounds
//!
//! - [`ProgressiveSearchConfig::max_branch_choices`] (`> 0`) keeps only the first N child
//!   cursors in declared order — a deterministic cap, where Python expands every branch.
//! - [`ProgressiveSearchConfig::max_depth`] (`> 0`) stops at `cursor.depth >= max_depth`:
//!   no child cursors are created, so the children's leaves are unreachable by design.
//!   Nothing is logged per call — the bound is configuration, not an anomaly — and leaf
//!   candidates selected at the deepest allowed node are still emitted.
//!
//! [`RetrieverItem`]: crate::retrieval::types::RetrieverItem
//! [`RetrieverCandidate`]: crate::retrieval::types::RetrieverCandidate

use crate::retrieval::search::config::ProgressiveSearchConfig;
use crate::retrieval::search::contracts::TargetExpander;
use crate::retrieval::search::flow::{
    ChildSearchCursor, ExpansionPlan, SearchCursor, SelectableTarget,
};
use crate::retrieval::types::RetrieverCandidate;

/// The default [`TargetExpander`].
///
/// Mirrors Python `DefaultTargetExpander`: it holds the traversal config and nothing else,
/// so it is cheap to clone and safe to share.
#[derive(Debug, Clone, Default)]
pub struct DefaultTargetExpander {
    /// The traversal config (the branch slack and the Rust-only bounds).
    config: ProgressiveSearchConfig,
}

impl DefaultTargetExpander {
    /// Builds the expander over `config`.
    #[must_use]
    pub fn new(config: ProgressiveSearchConfig) -> Self {
        Self { config }
    }

    /// The `top_k` one child branch may contribute.
    ///
    /// `min(top_k, ceil(top_k / max(1, branch_count)) + branch_candidate_slack)`, floored
    /// at `1`, except that `top_k == 0` (unlimited) yields `0` (unlimited).
    #[must_use]
    pub fn branch_top_k(&self, top_k: usize, branch_count: usize) -> usize {
        if top_k == 0 {
            return 0;
        }
        let branches = branch_count.max(1);
        let budget = top_k.div_ceil(branches) + self.config.branch_candidate_slack;
        top_k.min(budget.max(1))
    }

    /// Whether this node may still spawn child cursors.
    ///
    /// `false` when the Rust-only `max_depth` bound is reached.
    fn may_descend(&self, cursor: &SearchCursor) -> bool {
        self.config.max_depth == 0 || cursor.depth < self.config.max_depth
    }
}

impl TargetExpander for DefaultTargetExpander {
    /// Expands `selected` relative to `cursor`, preserving the declared order.
    fn expand_selected_targets(
        &self,
        cursor: &SearchCursor,
        selected: &[SelectableTarget],
    ) -> ExpansionPlan {
        let branch_top_k = self.branch_top_k(cursor.top_k, selected.len());
        let descend = self.may_descend(cursor);
        let mut leaf_results: Vec<RetrieverCandidate> = Vec::new();
        let mut child_cursors: Vec<ChildSearchCursor> = Vec::new();
        for target in selected {
            let resolution = &target.resolution;
            if resolution.is_terminal {
                if let Some(item) = resolution.item.as_ref() {
                    leaf_results.push(RetrieverCandidate {
                        rank: 1,
                        item_id: item.item_id.clone(),
                        payload: item.payload.clone(),
                        branch_path: resolution.branch_path.clone(),
                        label: item.label.clone(),
                        description: item.description.clone(),
                    });
                }
                continue;
            }
            if !descend {
                continue;
            }
            let Some(node_id) = resolution.node_id.as_ref().filter(|id| !id.is_empty()) else {
                continue;
            };
            child_cursors.push(ChildSearchCursor {
                cursor: SearchCursor::at(
                    node_id.clone(),
                    cursor.depth + 1,
                    resolution.branch_path.clone(),
                    branch_top_k,
                ),
                target: target.clone(),
            });
        }
        if self.config.max_branch_choices > 0 {
            child_cursors.truncate(self.config.max_branch_choices);
        }
        ExpansionPlan {
            leaf_results,
            child_cursors,
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use super::*;
    use crate::retrieval::search::config::ProgressiveSearchConfig;
    use crate::retrieval::search::flow::SelectableResolution;
    use crate::retrieval::types::{RetrieverItem, ROOT_NODE_ID};
    use serde_json::json;

    /// A terminal resolution carrying `item_id`/`payload` at `path`.
    fn terminal(code: &str, item_id: &str, payload: &str) -> SelectableTarget {
        SelectableTarget::new(SelectableResolution {
            code: code.to_owned(),
            canonical_id: format!("item::{payload}"),
            display_name: item_id.to_owned(),
            label: format!("{item_id}-label"),
            description: format!("{item_id}-description"),
            is_terminal: true,
            branch_path: vec![ROOT_NODE_ID.to_owned(), "Skills".to_owned()],
            node_id: None,
            item: Some(
                RetrieverItem::new(item_id, payload)
                    .with_label(format!("{item_id}-label"))
                    .with_description(format!("{item_id}-description")),
            ),
        })
    }

    /// A branch resolution pointing at `node_id` under `branch_path`.
    fn branch(code: &str, node_id: &str, branch_path: Vec<String>) -> SelectableTarget {
        SelectableTarget::new(SelectableResolution {
            code: code.to_owned(),
            canonical_id: format!("node::{node_id}"),
            display_name: node_id.to_owned(),
            label: node_id.to_owned(),
            description: String::new(),
            is_terminal: false,
            branch_path,
            node_id: Some(node_id.to_owned()),
            item: None,
        })
    }

    /// A cursor at the root with `top_k`.
    fn cursor(top_k: usize) -> SearchCursor {
        SearchCursor::at(ROOT_NODE_ID, 0, vec![ROOT_NODE_ID.to_owned()], top_k)
    }

    /// The default expander.
    fn expander() -> DefaultTargetExpander {
        DefaultTargetExpander::new(ProgressiveSearchConfig::default())
    }

    #[test]
    fn branch_budget_is_the_sum_share_plus_slack() {
        let expander = expander();
        // ceil(10 / 3) = 4, plus the default slack of 1.
        assert_eq!(expander.branch_top_k(10, 3), 5);
        // ceil(2 / 3) = 1, plus 1.
        assert_eq!(expander.branch_top_k(2, 3), 2);
        // The floor of 1 wins over a zero slack.
        assert_eq!(expander.branch_top_k(1, 5), 1);
        // The clamp to top_k wins over a large slack.
        let generous = DefaultTargetExpander::new(ProgressiveSearchConfig {
            branch_candidate_slack: 99,
            ..ProgressiveSearchConfig::default()
        });
        assert_eq!(generous.branch_top_k(4, 2), 4);
        // top_k == 0 is the unlimited sentinel.
        assert_eq!(expander.branch_top_k(0, 3), 0);
        // A zero branch count behaves like one branch.
        assert_eq!(expander.branch_top_k(9, 0), 9);
    }

    #[test]
    fn the_budget_uses_a_zero_slack_config() {
        let tight = DefaultTargetExpander::new(ProgressiveSearchConfig {
            branch_candidate_slack: 0,
            ..ProgressiveSearchConfig::default()
        });
        assert_eq!(tight.branch_top_k(9, 3), 3);
        assert_eq!(tight.branch_top_k(10, 3), 4);
    }

    #[test]
    fn terminal_targets_become_ranked_leaf_candidates() {
        let plan = expander().expand_selected_targets(
            &cursor(5),
            &[
                terminal("1", "alpha", "Skills.alpha.one"),
                terminal("2", "beta", "Skills.beta.two"),
            ],
        );
        assert!(plan.child_cursors.is_empty());
        assert_eq!(plan.leaf_results.len(), 2);
        let first = &plan.leaf_results[0];
        assert_eq!(first.rank, 1);
        assert_eq!(first.item_id, "alpha");
        assert_eq!(first.payload, "Skills.alpha.one");
        assert_eq!(first.label, "alpha-label");
        assert_eq!(first.description, "alpha-description");
        assert_eq!(
            first.branch_path,
            vec![ROOT_NODE_ID.to_owned(), "Skills".to_owned()]
        );
        assert_eq!(plan.leaf_results[1].item_id, "beta");
    }

    #[test]
    fn a_terminal_without_an_item_is_skipped() {
        let mut target = terminal("1", "alpha", "Skills.alpha.one");
        target.resolution.item = None;
        let plan = expander().expand_selected_targets(&cursor(5), &[target]);
        assert!(plan.leaf_results.is_empty());
        assert!(plan.child_cursors.is_empty());
    }

    #[test]
    fn branch_targets_become_deeper_child_cursors() {
        let selected = vec![
            branch(
                "1",
                "Skills.alpha",
                vec![
                    ROOT_NODE_ID.to_owned(),
                    "Skills".to_owned(),
                    "Skills.alpha".to_owned(),
                ],
            ),
            branch(
                "2",
                "Skills.beta",
                vec![
                    ROOT_NODE_ID.to_owned(),
                    "Skills".to_owned(),
                    "Skills.beta".to_owned(),
                ],
            ),
        ];
        let plan = expander().expand_selected_targets(&cursor(10), &selected);
        assert!(plan.leaf_results.is_empty());
        assert_eq!(plan.child_cursors.len(), 2);
        let child = &plan.child_cursors[0];
        assert_eq!(child.cursor.node_id, "Skills.alpha");
        assert_eq!(child.cursor.depth, 1);
        assert_eq!(
            child.cursor.branch_path,
            vec![
                ROOT_NODE_ID.to_owned(),
                "Skills".to_owned(),
                "Skills.alpha".to_owned()
            ]
        );
        // ceil(10 / 2) = 5 plus the default slack of 1.
        assert_eq!(child.cursor.top_k, 6);
        assert_eq!(child.target, selected[0]);
        assert_eq!(plan.child_cursors[1].cursor.top_k, 6);
    }

    #[test]
    fn a_zero_top_k_stays_unlimited_on_the_children() {
        let plan = expander().expand_selected_targets(
            &cursor(0),
            &[branch("1", "Skills.alpha", vec![ROOT_NODE_ID.to_owned()])],
        );
        assert_eq!(plan.child_cursors.len(), 1);
        assert_eq!(plan.child_cursors[0].cursor.top_k, 0);
    }

    #[test]
    fn a_branch_without_a_node_id_is_skipped() {
        let mut target = branch("1", "Skills.alpha", vec![ROOT_NODE_ID.to_owned()]);
        target.resolution.node_id = None;
        let plan = expander().expand_selected_targets(&cursor(5), &[target.clone()]);
        assert!(plan.child_cursors.is_empty());
        target.resolution.node_id = Some(String::new());
        let plan = expander().expand_selected_targets(&cursor(5), &[target]);
        assert!(plan.child_cursors.is_empty());
    }

    #[test]
    fn max_branch_choices_keeps_the_first_declared_branches() {
        let capped = DefaultTargetExpander::new(ProgressiveSearchConfig {
            max_branch_choices: 2,
            ..ProgressiveSearchConfig::default()
        });
        let selected: Vec<SelectableTarget> = (0..4)
            .map(|index| {
                branch(
                    &index.to_string(),
                    &format!("branch{index}"),
                    vec![ROOT_NODE_ID.to_owned()],
                )
            })
            .collect();
        let plan = capped.expand_selected_targets(&cursor(5), &selected);
        let ids: Vec<&str> = plan
            .child_cursors
            .iter()
            .map(|child| child.cursor.node_id.as_str())
            .collect();
        assert_eq!(ids, vec!["branch0", "branch1"]);
    }

    #[test]
    fn max_depth_stops_child_cursors_but_keeps_leaves() {
        let bounded = DefaultTargetExpander::new(ProgressiveSearchConfig {
            max_depth: 2,
            ..ProgressiveSearchConfig::default()
        });
        let at_depth_two = SearchCursor::at(
            "Skills.alpha",
            2,
            vec![ROOT_NODE_ID.to_owned(), "Skills.alpha".to_owned()],
            5,
        );
        let plan = bounded.expand_selected_targets(
            &at_depth_two,
            &[
                branch("1", "Skills.alpha.one", vec!["Skills.alpha".to_owned()]),
                terminal("2", "leaf", "Skills.alpha.leaf"),
            ],
        );
        assert!(plan.child_cursors.is_empty(), "the depth bound is reached");
        assert_eq!(plan.leaf_results.len(), 1);
        assert_eq!(plan.leaf_results[0].item_id, "leaf");

        let at_depth_one = SearchCursor::at(
            "Skills",
            1,
            vec![ROOT_NODE_ID.to_owned(), "Skills".to_owned()],
            5,
        );
        let plan = bounded.expand_selected_targets(
            &at_depth_one,
            &[branch("1", "Skills.alpha", vec!["Skills".to_owned()])],
        );
        assert_eq!(plan.child_cursors.len(), 1);
        assert_eq!(plan.child_cursors[0].cursor.depth, 2);
    }

    #[test]
    fn an_empty_selection_yields_an_empty_plan() {
        let plan = expander().expand_selected_targets(&cursor(5), &[]);
        assert_eq!(plan, ExpansionPlan::default());
        assert_eq!(
            serde_json::to_value(&plan).unwrap(),
            json!({"leaf_results": [], "child_cursors": []})
        );
    }
}
