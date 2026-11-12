//! Candidate mutation strategies.
//!
//! A [`MutationStrategy`] turns selected parents into the next generation of
//! [`Candidate`]s. The trait is the seam: an LLM-driven mutator plugs in
//! without the orchestrator noticing, while [`DeterministicMutator`] is the
//! offline, ML-free implementation used by tests and by the default
//! configuration.
//!
//! Determinism is a hard requirement: given the same request, a strategy must
//! produce the same ids, ordering, and bodies. [`DeterministicMutator`] gets
//! this by appending a fixed marker line and deriving ids from the parent id,
//! the generation, and the variant index.

use std::collections::{BTreeMap, BTreeSet};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use openjiuwen_common::error::{Error, Result};

use super::candidate::Candidate;

/// The name reported by [`DeterministicMutator`].
const MUTATOR_NAME: &str = "deterministic";

/// One mutation request: the parents to derive from and how many proposals the
/// caller wants.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct MutationRequest {
    /// The parent candidates, in the order the caller ranked them.
    pub parents: Vec<Candidate>,
    /// The generation the proposals belong to.
    pub generation: u32,
    /// The maximum number of proposals to return; must be greater than zero.
    pub count: usize,
    /// Caller-specific annotations; never interpreted here.
    pub metadata: BTreeMap<String, Value>,
}

/// Proposes the next generation of candidates from a set of parents.
#[async_trait]
pub trait MutationStrategy: Send + Sync {
    /// Returns the strategy's human-readable name.
    fn name(&self) -> &str;

    /// Proposes new candidates for `request`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the request itself is unusable (an
    /// empty proposal count, a zero variant count, or ids that would collide);
    /// a strategy that calls a model may also return provider errors.
    async fn propose(&self, request: MutationRequest) -> Result<Vec<Candidate>>;
}

/// ML-free mutation: appends a marker line to each parent body.
///
/// For every parent and every variant `n` in `0..variants_per_parent`, the
/// proposal body is the parent body, a newline, and
/// `<comment_prefix>rsi variant g<generation>-<n>`. The proposal id is
/// `<parent-id>/g<generation>/v<n>`, which is deterministic and unique per
/// parent, generation, and variant.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct DeterministicMutator {
    /// The comment marker prefixed to the appended marker line (`"# "`).
    pub comment_prefix: String,
    /// How many variants to derive from each parent (`1`).
    pub variants_per_parent: usize,
}

impl Default for DeterministicMutator {
    /// Returns Python's defaults: a `"# "` comment prefix and one variant.
    fn default() -> Self {
        Self {
            comment_prefix: "# ".to_owned(),
            variants_per_parent: 1,
        }
    }
}

impl DeterministicMutator {
    /// Creates a mutator with the default prefix and variant count.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the comment marker prefixed to the appended marker line.
    #[must_use]
    pub fn with_comment_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.comment_prefix = prefix.into();
        self
    }

    /// Sets how many variants to derive from each parent.
    #[must_use]
    pub fn with_variants_per_parent(mut self, variants: usize) -> Self {
        self.variants_per_parent = variants;
        self
    }

    /// Returns the configured comment prefix.
    #[must_use]
    pub fn comment_prefix(&self) -> &str {
        &self.comment_prefix
    }

    /// Returns the configured variant count.
    #[must_use]
    pub fn variants_per_parent(&self) -> usize {
        self.variants_per_parent
    }
}

#[async_trait]
impl MutationStrategy for DeterministicMutator {
    fn name(&self) -> &str {
        MUTATOR_NAME
    }

    async fn propose(&self, request: MutationRequest) -> Result<Vec<Candidate>> {
        if request.count == 0 {
            return Err(Error::validation(
                "mutation request count must be greater than zero",
            ));
        }
        if self.variants_per_parent == 0 {
            return Err(Error::validation(
                "deterministic mutator variants_per_parent must be greater than zero",
            ));
        }

        let mut proposals = Vec::new();
        let mut seen: BTreeSet<String> = BTreeSet::new();

        for parent in &request.parents {
            for variant in 0..self.variants_per_parent {
                let id = format!(
                    "{}/g{}/v{}",
                    parent.id.as_str(),
                    request.generation,
                    variant
                );
                if !seen.insert(id.clone()) {
                    return Err(Error::validation(format!(
                        "duplicate candidate id `{id}`: parents must have unique ids"
                    )));
                }

                let body = format!(
                    "{}\n{}rsi variant g{}-{}",
                    parent.body, self.comment_prefix, request.generation, variant
                );

                let mut candidate = Candidate::new(id, body);
                candidate.parent = Some(parent.id.clone());
                candidate.generation = request.generation;
                candidate.change_summary = format!(
                    "deterministic variant {variant} of `{}` at generation {}",
                    parent.id, request.generation
                );
                proposals.push(candidate);
            }
        }

        proposals.truncate(request.count);
        Ok(proposals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evolution::CandidateId;

    fn parent(id: &str, body: &str) -> Candidate {
        Candidate::new(id, body)
    }

    fn request(parents: Vec<Candidate>, generation: u32, count: usize) -> MutationRequest {
        MutationRequest {
            parents,
            generation,
            count,
            metadata: BTreeMap::new(),
        }
    }

    #[tokio::test]
    async fn default_mutator_appends_one_marker_per_parent() {
        let mutator = DeterministicMutator::new();
        let proposals = mutator
            .propose(request(vec![parent("seed", "model: demo")], 1, 1))
            .await
            .expect("mutation succeeds");

        assert_eq!(proposals.len(), 1);
        let proposal = &proposals[0];
        assert_eq!(proposal.id.as_str(), "seed/g1/v0");
        assert_eq!(
            proposal.parent.as_ref().map(CandidateId::as_str),
            Some("seed")
        );
        assert_eq!(proposal.generation, 1);
        assert_eq!(proposal.body, "model: demo\n# rsi variant g1-0");
        assert!(proposal.change_summary.contains("variant 0"));
        assert_eq!(proposal.status, crate::evolution::CandidateStatus::Pending);
        assert!(proposal.evaluation.is_none());
    }

    #[tokio::test]
    async fn output_order_is_deterministic() {
        let mutator = DeterministicMutator::new().with_variants_per_parent(2);
        let proposals = mutator
            .propose(request(
                vec![parent("a", "body-a"), parent("b", "body-b")],
                3,
                4,
            ))
            .await
            .expect("mutation succeeds");

        let ids: Vec<&str> = proposals.iter().map(|c| c.id.as_str()).collect();
        assert_eq!(ids, ["a/g3/v0", "a/g3/v1", "b/g3/v0", "b/g3/v1"]);
        assert_eq!(proposals[0].body, "body-a\n# rsi variant g3-0");
        assert_eq!(proposals[1].body, "body-a\n# rsi variant g3-1");
        assert_eq!(proposals[2].body, "body-b\n# rsi variant g3-0");
    }

    #[tokio::test]
    async fn count_caps_the_number_of_proposals() {
        let mutator = DeterministicMutator::new().with_variants_per_parent(2);
        let proposals = mutator
            .propose(request(
                vec![parent("a", "body-a"), parent("b", "body-b")],
                1,
                3,
            ))
            .await
            .expect("mutation succeeds");
        let ids: Vec<&str> = proposals.iter().map(|c| c.id.as_str()).collect();
        assert_eq!(ids, ["a/g1/v0", "a/g1/v1", "b/g1/v0"]);
    }

    #[tokio::test]
    async fn custom_prefix_is_applied() {
        let mutator = DeterministicMutator::new().with_comment_prefix("// ");
        let proposals = mutator
            .propose(request(vec![parent("p", "fn main() {}")], 2, 1))
            .await
            .expect("mutation succeeds");
        assert_eq!(proposals[0].body, "fn main() {}\n// rsi variant g2-0");
    }

    #[tokio::test]
    async fn zero_count_is_rejected() {
        let error = DeterministicMutator::new()
            .propose(request(vec![parent("a", "x")], 1, 0))
            .await
            .expect_err("zero count is rejected");
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
    }

    #[tokio::test]
    async fn zero_variants_is_rejected() {
        let mutator = DeterministicMutator::new().with_variants_per_parent(0);
        let error = mutator
            .propose(request(vec![parent("a", "x")], 1, 1))
            .await
            .expect_err("zero variants is rejected");
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
    }

    #[tokio::test]
    async fn colliding_parent_ids_are_rejected() {
        let error = DeterministicMutator::new()
            .propose(request(
                vec![parent("same", "x"), parent("same", "y")],
                1,
                2,
            ))
            .await
            .expect_err("duplicate parent ids are rejected");
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
        assert!(error.to_string().contains("duplicate candidate id"));
    }

    #[tokio::test]
    async fn empty_parents_yield_no_proposals() {
        let proposals = DeterministicMutator::new()
            .propose(request(Vec::new(), 1, 1))
            .await
            .expect("empty parents are not an error");
        assert_eq!(proposals, Vec::new());
    }

    #[test]
    fn mutator_defaults_and_accessors() {
        let mutator = DeterministicMutator::default();
        assert_eq!(mutator.comment_prefix(), "# ");
        assert_eq!(mutator.variants_per_parent(), 1);
        assert_eq!(DeterministicMutator::new(), mutator);
        assert_eq!(
            DeterministicMutator::new()
                .with_variants_per_parent(3)
                .variants_per_parent(),
            3
        );
    }
}
