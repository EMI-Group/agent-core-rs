//! The default branch reducer: one candidate list out of many branches.
//!
//! Ports Python `openjiuwen.symphony.retrieval.search.runtime.reduce` (v0.1.20)
//! verbatim in behaviour.
//!
//! The branch list is `[local_leaves, child_results…]` and the candidates are interleaved
//! **round-robin** (one candidate per branch, then the next index across every branch), so
//! one deep or fruitful branch cannot starve the others' best hits. A sequential
//! append pass over the concatenated branches follows when the round-robin stalls, and
//! every pass dedupes by `payload` (falling back to `item_id` when the payload is blank).
//!
//! [`ProgressiveSearchConfig::round_robin_branch_reduce`] `= false` switches to
//! concatenate-then-dedupe, which is what Python's legacy path does.
//!
//! # The `top_k == 0` sentinel
//!
//! Python truncates by slicing (`[:top_k]`), so `top_k == 0` keeps nothing. This crate
//! treats `0` as "unlimited" everywhere (search requests, semantic indexes, budgets), so
//! the reducer consumes every branch instead of dropping everything.

use std::collections::BTreeSet;

use crate::retrieval::search::config::ProgressiveSearchConfig;
use crate::retrieval::search::contracts::BranchReducer;
use crate::retrieval::search::flow::{NodeSearchResult, SearchCursor};
use crate::retrieval::types::RetrieverCandidate;

/// The default [`BranchReducer`].
///
/// Mirrors Python `DefaultBranchReducer`: it holds the traversal config and nothing else.
#[derive(Debug, Clone, Default)]
pub struct DefaultBranchReducer {
    /// The traversal config (only `round_robin_branch_reduce` matters here).
    config: ProgressiveSearchConfig,
}

impl DefaultBranchReducer {
    /// Builds the reducer over `config`.
    #[must_use]
    pub fn new(config: ProgressiveSearchConfig) -> Self {
        Self { config }
    }

    /// Merges `branches` into one deduped list of at most `top_k` candidates.
    ///
    /// `top_k == 0` means unlimited. The merge is the round-robin interleave described at
    /// the module level, or the concatenate-then-dedupe order when
    /// [`ProgressiveSearchConfig::round_robin_branch_reduce`] is off.
    #[must_use]
    pub fn merge_branch_candidates(
        &self,
        branches: &[Vec<RetrieverCandidate>],
        top_k: usize,
    ) -> Vec<RetrieverCandidate> {
        let limit = if top_k == 0 { usize::MAX } else { top_k };
        if !self.config.round_robin_branch_reduce {
            return dedupe(branches).into_iter().take(limit).collect();
        }

        let mut reduced: Vec<RetrieverCandidate> = Vec::new();
        let mut seen: BTreeSet<String> = BTreeSet::new();
        let mut index = 0;
        while reduced.len() < limit {
            let mut added = false;
            for branch in branches {
                let Some(candidate) = branch.get(index) else {
                    continue;
                };
                if !seen.insert(dedupe_key(candidate)) {
                    continue;
                }
                reduced.push(candidate.clone());
                added = true;
                if reduced.len() >= limit {
                    break;
                }
            }
            if !added {
                break;
            }
            index += 1;
        }
        if reduced.len() >= limit {
            return reduced;
        }

        for candidate in branches.iter().flatten() {
            if !seen.insert(dedupe_key(candidate)) {
                continue;
            }
            reduced.push(candidate.clone());
            if reduced.len() >= limit {
                break;
            }
        }
        reduced
    }
}

impl BranchReducer for DefaultBranchReducer {
    /// Reduces `local_leaves` and `child_results` into one candidate list for `cursor`.
    fn reduce_branch_results(
        &self,
        cursor: &SearchCursor,
        local_leaves: &[RetrieverCandidate],
        child_results: &[NodeSearchResult],
    ) -> NodeSearchResult {
        let mut branches: Vec<Vec<RetrieverCandidate>> =
            Vec::with_capacity(child_results.len() + 1);
        branches.push(local_leaves.to_vec());
        for result in child_results {
            branches.push(result.candidates.clone());
        }
        NodeSearchResult {
            candidates: self.merge_branch_candidates(&branches, cursor.top_k),
        }
    }
}

/// The dedupe identity of one candidate: its payload, or its `item_id` when the payload
/// is blank.
fn dedupe_key(candidate: &RetrieverCandidate) -> String {
    if candidate.payload.is_empty() {
        candidate.item_id.clone()
    } else {
        candidate.payload.clone()
    }
}

/// Concatenates `branches` in order, keeping the first candidate per dedupe key.
fn dedupe(branches: &[Vec<RetrieverCandidate>]) -> Vec<RetrieverCandidate> {
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut kept: Vec<RetrieverCandidate> = Vec::new();
    for candidate in branches.iter().flatten() {
        if seen.insert(dedupe_key(candidate)) {
            kept.push(candidate.clone());
        }
    }
    kept
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use super::*;
    use crate::retrieval::search::config::ProgressiveSearchConfig;
    use crate::retrieval::types::ROOT_NODE_ID;

    /// A candidate with `item_id` and `payload`.
    fn candidate(item_id: &str, payload: &str) -> RetrieverCandidate {
        RetrieverCandidate::new(1, item_id, payload)
    }

    /// A cursor with `top_k`.
    fn cursor(top_k: usize) -> SearchCursor {
        SearchCursor::at(ROOT_NODE_ID, 0, vec![ROOT_NODE_ID.to_owned()], top_k)
    }

    /// The reducer over the config carrying `round_robin`.
    fn reducer(round_robin: bool) -> DefaultBranchReducer {
        DefaultBranchReducer::new(ProgressiveSearchConfig {
            round_robin_branch_reduce: round_robin,
            ..ProgressiveSearchConfig::default()
        })
    }

    /// Three branches with one overlapping payload (`a`) between the first two.
    fn branches() -> Vec<Vec<RetrieverCandidate>> {
        vec![
            vec![
                candidate("A", "a"),
                candidate("B", "b"),
                candidate("C", "c"),
            ],
            vec![
                candidate("A2", "a"),
                candidate("D", "d"),
                candidate("E", "e"),
            ],
            vec![candidate("F", "f"), candidate("G", "g")],
        ]
    }

    /// The item ids of `candidates`, in order.
    fn ids(candidates: &[RetrieverCandidate]) -> Vec<&str> {
        candidates
            .iter()
            .map(|candidate| candidate.item_id.as_str())
            .collect()
    }

    #[test]
    fn round_robin_interleaves_and_dedupes_by_payload() {
        let merged = reducer(true).merge_branch_candidates(&branches(), 0);
        // Index 0: A (b0), A2 is a duplicate of payload "a", F (b2).
        // Index 1: B, D, G. Index 2: C, E.
        assert_eq!(ids(&merged), vec!["A", "F", "B", "D", "G", "C", "E"]);
    }

    #[test]
    fn round_robin_clamps_to_top_k() {
        let merged = reducer(true).merge_branch_candidates(&branches(), 3);
        assert_eq!(ids(&merged), vec!["A", "F", "B"]);
    }

    #[test]
    fn a_stall_falls_back_to_a_sequential_append_pass() {
        // Index 0 yields `A` and `B`; index 1 yields only duplicates, so the round-robin
        // pass stalls with `T` (branch 0, index 2) still unvisited. The sequential append
        // pass is what makes `T` reachable at all.
        let stalled = vec![
            vec![
                candidate("A", "a"),
                candidate("A2", "a"),
                candidate("T", "t"),
            ],
            vec![candidate("B", "b"), candidate("B2", "b")],
        ];
        let merged = reducer(true).merge_branch_candidates(&stalled, 0);
        assert_eq!(ids(&merged), vec!["A", "B", "T"]);
    }

    #[test]
    fn dedupe_falls_back_to_the_item_id_for_a_blank_payload() {
        let branches = vec![
            vec![candidate("shared", ""), candidate("first", "")],
            vec![candidate("shared", ""), candidate("second", "")],
        ];
        let merged = reducer(true).merge_branch_candidates(&branches, 0);
        assert_eq!(ids(&merged), vec!["shared", "first", "second"]);
    }

    #[test]
    fn concatenate_then_dedupe_keeps_the_declared_branch_order() {
        let merged = reducer(false).merge_branch_candidates(&branches(), 0);
        assert_eq!(ids(&merged), vec!["A", "B", "C", "D", "E", "F", "G"]);
        let clamped = reducer(false).merge_branch_candidates(&branches(), 2);
        assert_eq!(ids(&clamped), vec!["A", "B"]);
    }

    #[test]
    fn a_zero_top_k_is_unlimited() {
        let merged = reducer(true).merge_branch_candidates(&branches(), 0);
        assert_eq!(merged.len(), 7);
    }

    #[test]
    fn reduce_branch_results_puts_the_local_leaves_first_in_the_branch_list() {
        let local = vec![candidate("local", "local")];
        let children = vec![
            NodeSearchResult {
                candidates: vec![candidate("child-a", "child-a")],
            },
            NodeSearchResult {
                candidates: vec![candidate("child-b", "child-b")],
            },
        ];
        let result = reducer(true).reduce_branch_results(&cursor(5), &local, &children);
        assert_eq!(ids(&result.candidates), vec!["local", "child-a", "child-b"]);
    }

    #[test]
    fn reduce_branch_results_honours_the_cursor_top_k() {
        let local = vec![candidate("a", "a"), candidate("b", "b")];
        let children = vec![NodeSearchResult {
            candidates: vec![candidate("c", "c")],
        }];
        let result = reducer(true).reduce_branch_results(&cursor(2), &local, &children);
        assert_eq!(ids(&result.candidates), vec!["a", "c"]);
    }

    #[test]
    fn empty_inputs_reduce_to_an_empty_result() {
        let result = reducer(true).reduce_branch_results(&cursor(0), &[], &[]);
        assert!(result.candidates.is_empty());
        assert!(reducer(true)
            .merge_branch_candidates(&[Vec::new(), Vec::new()], 5)
            .is_empty());
    }
}
