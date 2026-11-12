//! Selection strategies that keep the elite of an evaluated candidate set.
//!
//! Selection is pure and deterministic: it never re-evaluates, never touches
//! the clock, and breaks every tie by candidate id. The helpers [`ranked`] and
//! [`rank_score`] also feed the `PUCT` rank prior in
//! [`crate::artifact_rsi::program_opt`], so the ordering here is part of the
//! public behaviour of both RSI halves.

use std::cmp::Ordering;
use std::collections::BTreeSet;

use super::candidate::{Candidate, CandidateId};

/// The name reported by [`ElitistSelection`].
const SELECTION_NAME: &str = "elitist";

/// Chooses which candidates survive into the next generation.
pub trait SelectionStrategy: Send + Sync {
    /// Returns the strategy's human-readable name.
    fn name(&self) -> &str;

    /// Selects at most `count` candidate ids from `candidates`.
    ///
    /// Selection is deterministic: the same input always yields the same ids in
    /// the same order.
    fn select(&self, candidates: &[Candidate], count: usize) -> Vec<CandidateId>;
}

/// Keeps the highest-scoring candidates.
///
/// Ordered by score descending; unscored (missing report or invalid) candidates
/// come last; ties break by candidate id ascending. The result is capped at
/// both `count` and [`Self::elite_size`], and candidates that share an id are
/// selected only once (the duplicate is skipped).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ElitistSelection {
    /// The maximum elite size (`1` by default).
    pub elite_size: usize,
}

impl Default for ElitistSelection {
    /// Returns Python's default elite size of one.
    fn default() -> Self {
        Self { elite_size: 1 }
    }
}

impl ElitistSelection {
    /// Creates an elitist strategy keeping at most `elite_size` candidates.
    #[must_use]
    pub fn new(elite_size: usize) -> Self {
        Self { elite_size }
    }

    /// Returns the configured elite size.
    #[must_use]
    pub fn elite_size(&self) -> usize {
        self.elite_size
    }
}

impl SelectionStrategy for ElitistSelection {
    fn name(&self) -> &str {
        SELECTION_NAME
    }

    fn select(&self, candidates: &[Candidate], count: usize) -> Vec<CandidateId> {
        let limit = self.elite_size.min(count);
        if limit == 0 {
            return Vec::new();
        }

        let mut sorted: Vec<&Candidate> = candidates.iter().collect();
        sorted.sort_by(|left, right| compare_desc(left, right));

        let mut selected = Vec::new();
        let mut seen: BTreeSet<&str> = BTreeSet::new();
        for candidate in sorted {
            if selected.len() >= limit {
                break;
            }
            if seen.insert(candidate.id.as_str()) {
                selected.push(candidate.id.clone());
            }
        }
        selected
    }
}

/// Returns the normalised rank score of `rank` among `total` candidates.
///
/// Python's `artifact_rsi` rank prior: `rank / (total - 1)`, with a degenerate
/// `total <= 1` scoring `0.5` (there is no spread to normalise over).
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn rank_score(rank: usize, total: usize) -> f64 {
    if total <= 1 {
        return 0.5;
    }
    rank as f64 / (total - 1) as f64
}

/// Returns the candidates sorted ascending by score, unscored last.
///
/// The sort is stable, so equal scores keep their input order; a `None` score
/// (or an invalid candidate) sorts as negative infinity.
#[must_use]
pub fn ranked(candidates: &[Candidate]) -> Vec<&Candidate> {
    let mut sorted: Vec<&Candidate> = candidates.iter().collect();
    sorted.sort_by(|left, right| compare_asc(left, right));
    sorted
}

/// Compares by score ascending, with `None` last; equal scores stay in input
/// order.
fn compare_asc(left: &Candidate, right: &Candidate) -> Ordering {
    match (left.score(), right.score()) {
        (Some(left_score), Some(right_score)) => left_score
            .partial_cmp(&right_score)
            .unwrap_or(Ordering::Equal),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

/// Compares by score descending, then id ascending; `None` last.
fn compare_desc(left: &Candidate, right: &Candidate) -> Ordering {
    match (left.score(), right.score()) {
        (Some(left_score), Some(right_score)) => {
            let order = right_score
                .partial_cmp(&left_score)
                .unwrap_or(Ordering::Equal);
            if order == Ordering::Equal {
                left.id.as_str().cmp(right.id.as_str())
            } else {
                order
            }
        }
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => left.id.as_str().cmp(right.id.as_str()),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use super::*;
    use crate::evolution::evaluation::{CaseResult, EvaluationReport};
    use crate::evolution::CandidateStatus;

    fn scored(id: &str, score: Option<f64>) -> Candidate {
        let mut candidate = Candidate::new(id, "body");
        if let Some(score) = score {
            candidate.status = CandidateStatus::Evaluated;
            candidate.evaluation = Some(EvaluationReport::from_results(vec![CaseResult {
                case_id: "case".to_owned(),
                passed: true,
                score: Some(score),
                weight: 1.0,
                error: None,
                details: Value::Null,
            }]));
        }
        candidate
    }

    fn approx(actual: f64, expected: f64) -> bool {
        (actual - expected).abs() < 1e-9
    }

    #[test]
    fn rank_score_handles_degenerate_totals() {
        assert!(approx(rank_score(0, 0), 0.5));
        assert!(approx(rank_score(7, 1), 0.5));
        assert!(approx(rank_score(0, 3), 0.0));
        assert!(approx(rank_score(1, 3), 0.5));
        assert!(approx(rank_score(2, 3), 1.0));
    }

    #[test]
    fn ranked_sorts_ascending_with_unscored_last() {
        let candidates = vec![
            scored("c", Some(0.5)),
            scored("none", None),
            scored("a", Some(0.1)),
            scored("b", Some(0.9)),
        ];
        let ids: Vec<&str> = ranked(&candidates).iter().map(|c| c.id.as_str()).collect();
        assert_eq!(ids, ["a", "c", "b", "none"]);
    }

    #[test]
    fn ranked_ties_preserve_input_order() {
        let candidates = vec![
            scored("third", Some(0.5)),
            scored("first", Some(0.5)),
            scored("second", Some(0.5)),
        ];
        let order: Vec<&str> = ranked(&candidates).iter().map(|c| c.id.as_str()).collect();
        assert_eq!(order, ["third", "first", "second"]);
    }

    #[test]
    fn elitist_keeps_the_highest_score() {
        let selection = ElitistSelection::new(2);
        let candidates = vec![
            scored("low", Some(0.1)),
            scored("high", Some(0.9)),
            scored("mid", Some(0.5)),
        ];
        assert_eq!(selection.name(), "elitist");
        let ids: Vec<String> = selection
            .select(&candidates, 2)
            .into_iter()
            .map(|id| id.as_str().to_owned())
            .collect();
        assert_eq!(ids, ["high", "mid"]);
    }

    #[test]
    fn elitist_breaks_ties_by_id_ascending() {
        let selection = ElitistSelection::new(3);
        let candidates = vec![
            scored("b", Some(0.5)),
            scored("a", Some(0.5)),
            scored("c", Some(0.5)),
        ];
        let ids: Vec<String> = selection
            .select(&candidates, 3)
            .into_iter()
            .map(|id| id.as_str().to_owned())
            .collect();
        assert_eq!(ids, ["a", "b", "c"]);
    }

    #[test]
    fn elitist_caps_at_count_and_elite_size() {
        let candidates = vec![scored("a", Some(0.9)), scored("b", Some(0.5))];
        let selection = ElitistSelection::new(2);
        assert_eq!(selection.select(&candidates, 1).len(), 1);
        assert_eq!(selection.select(&candidates, 10).len(), 2);

        let single = ElitistSelection::default();
        assert_eq!(single.elite_size(), 1);
        assert_eq!(single.select(&candidates, 10).len(), 1);
        assert_eq!(ElitistSelection::new(0).select(&candidates, 10), Vec::new());
    }

    #[test]
    fn elitist_skips_unscored_when_scored_candidates_exist() {
        let selection = ElitistSelection::new(1);
        let candidates = vec![scored("none", None), scored("some", Some(0.0))];
        let ids: Vec<String> = selection
            .select(&candidates, 1)
            .into_iter()
            .map(|id| id.as_str().to_owned())
            .collect();
        assert_eq!(ids, ["some"]);
    }

    #[test]
    fn elitist_skips_duplicate_ids() {
        let selection = ElitistSelection::new(3);
        let candidates = vec![
            scored("same", Some(0.9)),
            scored("same", Some(0.8)),
            scored("other", Some(0.1)),
        ];
        let ids: Vec<String> = selection
            .select(&candidates, 3)
            .into_iter()
            .map(|id| id.as_str().to_owned())
            .collect();
        assert_eq!(ids, ["same", "other"]);
    }
}
