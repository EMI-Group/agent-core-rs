//! Review-round tallying: the pure decision behind the review gate.
//!
//! Python parity source: `TeamTaskManager.get_review_tally` in
//! `openjiuwen/agent_teams/tools/task_manager.py` (the binary `verifier` / `challenger` pool of a
//! round) and the `review_vote` table of `tools/database/task_dao.py`.
//!
//! Only the binary pool is ported: an `inspector` reviewer (a scored `0..=1` vote, averaged rather
//! than counted) is part of the deferred structured-reviewer model, so this crate tallies
//! [`Verdict::Pass`] / [`Verdict::Fail`] votes only.
//!
//! The tally is a pure function of the votes of one round plus the task's reviewer set: a member of
//! the reviewer set that has not voted yet simply has no vote, which is what makes "still waiting"
//! observable ([`ReviewTally::decided`]) without any extra state.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::schema::{ReviewVote, Verdict};

/// The vote state of one review round.
///
/// `reviewer_count` is the size of the task's reviewer set — the number of votes the round needs —
/// not the number of votes cast, so a caller can distinguish "1 of 3 voted" from "the only reviewer
/// voted". `passed` is only `true` once every reviewer has voted (all votes are `pass` or `fail`, so
/// `pass_count + fail_count` is the number of voters) and the passes outnumber the fails: a majority
/// of a partial round never settles the gate.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ReviewTally {
    /// Review round the tally describes; `0` when there are no votes.
    pub round: u32,
    /// Votes of this round that passed the task.
    pub pass_count: u32,
    /// Votes of this round that sent it back for rework.
    pub fail_count: u32,
    /// Number of reviewers the round waits for.
    pub reviewer_count: u32,
    /// Whether the round has settled in favour of the author.
    pub passed: bool,
}

impl ReviewTally {
    /// Returns the number of votes cast so far (`pass_count + fail_count`).
    #[must_use]
    pub const fn voted(&self) -> u32 {
        self.pass_count + self.fail_count
    }

    /// Returns `true` once every expected reviewer has voted, so the round can be settled.
    #[must_use]
    pub const fn decided(&self) -> bool {
        self.reviewer_count > 0 && self.voted() >= self.reviewer_count
    }

    /// Returns `true` when nothing has been voted yet.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.voted() == 0
    }
}

/// Tallies `votes` without an expected reviewer set: every distinct reviewer counts.
///
/// `reviewer_count` becomes the number of members that actually voted, so `passed` reduces to
/// "more passes than fails". Use [`tally_with`] when the task's reviewer set is known — that is what
/// the board does, because a round is only decided once *every* reviewer voted.
#[must_use]
pub fn tally(votes: &[ReviewVote]) -> ReviewTally {
    let counted = count(votes, None);
    tally_from(counted)
}

/// Tallies the `votes` of one round against the task's `reviewers`.
///
/// Votes from members that are no longer reviewers are ignored (the reviewer set can change between
/// rounds), duplicate votes of one reviewer keep the last one, and `reviewer_count` is the size of
/// the deduplicated reviewer set.
#[must_use]
pub fn tally_with(votes: &[ReviewVote], reviewers: &[String]) -> ReviewTally {
    let allowed: Vec<&str> = reviewers.iter().map(String::as_str).collect();
    let counted = count(votes, Some(&allowed));
    let reviewer_count = distinct_len(&allowed);
    ReviewTally {
        reviewer_count,
        passed: reviewer_count > 0
            && counted.voted() >= reviewer_count
            && counted.pass > counted.fail,
        ..tally_from(counted)
    }
}

/// One round's counted votes before the reviewer set is folded in.
#[derive(Debug, Clone, Copy, Default)]
struct Counted {
    round: u32,
    pass: u32,
    fail: u32,
}

impl Counted {
    const fn voted(self) -> u32 {
        self.pass + self.fail
    }
}

/// Counts the latest vote per reviewer, optionally restricted to `allowed` reviewers.
fn count(votes: &[ReviewVote], allowed: Option<&[&str]>) -> Counted {
    // Latest vote wins, matching the DAO's replace-on-same-(round, reviewer) behaviour.
    let mut latest: BTreeMap<&str, Verdict> = BTreeMap::new();
    let mut round = 0;
    for vote in votes {
        if let Some(allowed) = allowed {
            if !allowed.contains(&vote.reviewer.as_str()) {
                continue;
            }
        }
        round = round.max(vote.round);
        latest.insert(vote.reviewer.as_str(), vote.verdict);
    }
    let mut counted = Counted {
        round,
        ..Counted::default()
    };
    for verdict in latest.into_values() {
        match verdict {
            Verdict::Pass => counted.pass += 1,
            Verdict::Fail => counted.fail += 1,
        }
    }
    counted
}

/// Projects [`Counted`] into a tally whose reviewer count equals the votes cast.
fn tally_from(counted: Counted) -> ReviewTally {
    ReviewTally {
        round: counted.round,
        pass_count: counted.pass,
        fail_count: counted.fail,
        reviewer_count: counted.voted(),
        passed: counted.pass > counted.fail,
    }
}

/// Returns the number of distinct names in `names`.
fn distinct_len(names: &[&str]) -> u32 {
    let distinct: std::collections::BTreeSet<&str> = names.iter().copied().collect();
    u32::try_from(distinct.len()).unwrap_or(u32::MAX)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn vote(round: u32, reviewer: &str, verdict: Verdict) -> ReviewVote {
        ReviewVote {
            team_name: "team".to_owned(),
            task_id: "task-1".to_owned(),
            round,
            reviewer: reviewer.to_owned(),
            verdict,
        }
    }

    fn reviewers(names: &[&str]) -> Vec<String> {
        names.iter().map(|name| (*name).to_owned()).collect()
    }

    #[test]
    fn an_empty_round_is_not_decided() {
        let empty = tally(&[]);
        assert_eq!(empty.round, 0);
        assert_eq!(empty.voted(), 0);
        assert!(empty.is_empty());
        assert!(!empty.decided());
        assert!(!empty.passed);
    }

    #[test]
    fn a_partial_round_is_not_decided_even_with_a_majority() {
        let votes = [vote(1, "alice", Verdict::Pass)];
        let partial = tally_with(&votes, &reviewers(&["alice", "bob", "carol"]));
        assert_eq!(partial.pass_count, 1);
        assert_eq!(partial.fail_count, 0);
        assert_eq!(partial.reviewer_count, 3);
        assert_eq!(partial.voted(), 1);
        assert!(!partial.decided());
        assert!(!partial.passed);
        assert_eq!(partial.round, 1);
    }

    #[test]
    fn a_full_round_passes_on_a_majority() {
        let votes = [
            vote(2, "alice", Verdict::Pass),
            vote(2, "bob", Verdict::Pass),
            vote(2, "carol", Verdict::Fail),
        ];
        let full = tally_with(&votes, &reviewers(&["alice", "bob", "carol"]));
        assert_eq!(full.pass_count, 2);
        assert_eq!(full.fail_count, 1);
        assert_eq!(full.reviewer_count, 3);
        assert!(full.decided());
        assert!(full.passed);
        assert_eq!(full.round, 2);
    }

    #[test]
    fn a_full_round_fails_on_a_majority() {
        let votes = [
            vote(1, "alice", Verdict::Fail),
            vote(1, "bob", Verdict::Fail),
            vote(1, "carol", Verdict::Pass),
        ];
        let full = tally_with(&votes, &reviewers(&["alice", "bob", "carol"]));
        assert!(full.decided());
        assert!(!full.passed);
        assert_eq!(full.fail_count, 2);
    }

    #[test]
    fn a_tie_does_not_pass() {
        let votes = [
            vote(1, "alice", Verdict::Pass),
            vote(1, "bob", Verdict::Fail),
        ];
        let tied = tally_with(&votes, &reviewers(&["alice", "bob"]));
        assert!(tied.decided());
        assert!(!tied.passed);
    }

    #[test]
    fn a_vote_from_a_former_reviewer_is_ignored() {
        let votes = [
            vote(1, "alice", Verdict::Pass),
            vote(1, "bob", Verdict::Pass),
        ];
        let tally = tally_with(&votes, &reviewers(&["alice"]));
        assert_eq!(tally.reviewer_count, 1);
        assert_eq!(tally.pass_count, 1);
        assert!(tally.passed);
    }

    #[test]
    fn a_repeated_vote_keeps_the_latest_verdict() {
        let votes = [
            vote(1, "alice", Verdict::Fail),
            vote(1, "alice", Verdict::Pass),
        ];
        let tally = tally_with(&votes, &reviewers(&["alice"]));
        assert_eq!(tally.pass_count, 1);
        assert_eq!(tally.fail_count, 0);
        assert!(tally.passed);
    }

    #[test]
    fn duplicate_reviewer_entries_are_counted_once() {
        let votes = [vote(1, "alice", Verdict::Pass)];
        let tally = tally_with(&votes, &reviewers(&["alice", "alice"]));
        assert_eq!(tally.reviewer_count, 1);
        assert!(tally.passed);
    }

    #[test]
    fn a_task_without_reviewers_never_passes() {
        let votes = [vote(1, "alice", Verdict::Pass)];
        let tally = tally_with(&votes, &[]);
        assert_eq!(tally.reviewer_count, 0);
        assert!(!tally.passed);
        assert!(!tally.decided());
    }

    #[test]
    fn tally_without_a_reviewer_set_counts_the_voters() {
        let votes = [
            vote(3, "alice", Verdict::Pass),
            vote(3, "bob", Verdict::Fail),
        ];
        let tally = tally(&votes);
        assert_eq!(tally.round, 3);
        assert_eq!(tally.reviewer_count, 2);
        assert_eq!(tally.pass_count, 1);
        assert_eq!(tally.fail_count, 1);
        assert!(!tally.passed);
    }

    #[test]
    fn the_tally_round_survives_a_json_round_trip() {
        let tally = ReviewTally {
            round: 2,
            pass_count: 1,
            fail_count: 0,
            reviewer_count: 3,
            passed: false,
        };
        let encoded = serde_json::to_string(&tally).unwrap();
        let decoded: ReviewTally = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, tally);
        assert_eq!(
            serde_json::from_str::<ReviewTally>("{}").unwrap(),
            ReviewTally::default()
        );
    }
}
