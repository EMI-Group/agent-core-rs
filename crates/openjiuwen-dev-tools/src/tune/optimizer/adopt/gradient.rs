//! [`GradientPolicy`]: the seam that turns failing cases into a "textual
//! gradient".
//!
//! Python's `PartialOptimizer` asks a model twice per failing case (generate,
//! then reduce) and stores the resulting Chinese prose on a `TextualParameter`
//! as its gradient. That requires a model, a prompt bank, and a `TextualParameter`
//! to store into; none of those exist on this side of the port, and the
//! deterministic path must work offline.
//!
//! So the gradient is behind a seam: [`GradientPolicy`] takes the task, the
//! candidate prompt, and the failing cases, and returns the gradient text.
//! [`DeterministicGradientPolicy`] is the default — it renders
//! [`describe_gradient_context`] under a fixed header, so a caller gets a real,
//! reproducible artifact without a model. A caller that wants a model-backed
//! gradient implements the trait (a few lines over
//! `Arc<dyn openjiuwen_foundation::llm::Model>` plus
//! [`crate::util::extract_tag`]) and passes it in; this module deliberately
//! stays model-free.

use async_trait::async_trait;
use openjiuwen_common::error::Result;

use crate::tune::{EvaluatedCase, PromptCandidate, PromptTaskSpec};

use super::describe_gradient_context;

/// Produces the textual gradient of one candidate prompt.
///
/// Implementations are `async` because a model-backed one calls a provider; the
/// deterministic default never awaits anything external.
#[async_trait]
pub trait GradientPolicy: Send + Sync {
    /// The gradient text for `candidate` on `task`, derived from the failing
    /// `bad` cases.
    ///
    /// # Errors
    ///
    /// A model or transport failure. The deterministic default never fails.
    async fn gradient(
        &self,
        task: &PromptTaskSpec,
        candidate: &PromptCandidate,
        bad: &[EvaluatedCase],
    ) -> Result<String>;
}

/// The default gradient: [`render_gradient`], with no model involved.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DeterministicGradientPolicy;

impl DeterministicGradientPolicy {
    /// Creates the policy.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl GradientPolicy for DeterministicGradientPolicy {
    async fn gradient(
        &self,
        task: &PromptTaskSpec,
        candidate: &PromptCandidate,
        bad: &[EvaluatedCase],
    ) -> Result<String> {
        Ok(render_gradient(task, candidate, bad))
    }
}

/// Renders the deterministic gradient block for `candidate`.
///
/// Three header lines name the intent, the candidate, and the block, followed by
/// one [`describe_gradient_context`] entry per failing case:
///
/// ```text
/// objective: answer arithmetic questions
/// candidate: adopt-1a2b3c4d
/// gradient:
/// case 0: c1
///   score: 0.000
///   ...
/// ```
///
/// The objective line is [`PromptTaskSpec::characteristics`] (the objective plus
/// any constraints) and the candidate line is the candidate's own identifier, or
/// `(unnamed candidate)` when it carries none — both deterministic, so the same
/// inputs always render the same block.
#[must_use]
pub fn render_gradient(
    task: &PromptTaskSpec,
    candidate: &PromptCandidate,
    bad: &[EvaluatedCase],
) -> String {
    let name = if candidate.candidate_id.trim().is_empty() {
        "(unnamed candidate)"
    } else {
        candidate.candidate_id.as_str()
    };
    format!(
        "objective: {}\ncandidate: {name}\ngradient:\n{}",
        task.characteristics(),
        describe_gradient_context(bad)
    )
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use serde_json::json;

    use crate::tune::Case;

    use super::*;

    fn failing(id: &str, reason: &str) -> EvaluatedCase {
        let inputs = BTreeMap::from([("query".to_owned(), json!(format!("question {id}")))]);
        let label = BTreeMap::from([("output".to_owned(), json!("4"))]);
        let case = Case::new(inputs, label).unwrap().with_case_id(id);
        let answer = BTreeMap::from([("output".to_owned(), json!("5"))]);
        EvaluatedCase::new(case, answer, 0.0, reason).unwrap()
    }

    #[tokio::test]
    async fn the_default_policy_renders_the_context_under_a_header() {
        let task = PromptTaskSpec::new("answer arithmetic questions")
            .with_constraints(vec!["be brief".to_owned()]);
        let candidate = PromptCandidate::new("Solve {{question}}.", "seed").with_id("cand-1");
        let bad = [failing("c1", "wrong result")];
        let gradient = DeterministicGradientPolicy::new()
            .gradient(&task, &candidate, &bad)
            .await
            .unwrap();
        assert_eq!(
            gradient,
            "objective: answer arithmetic questions\nconstraints: be brief\n\
             candidate: cand-1\ngradient:\n\
             case 0: c1\n  score: 0.000\n  input: {\"query\":\"question c1\"}\n  \
             answer: {\"output\":\"5\"}\n  expected: {\"output\":\"4\"}\n  \
             feedback: wrong result"
        );
        assert_eq!(gradient, render_gradient(&task, &candidate, &bad));
    }

    #[tokio::test]
    async fn an_unnamed_candidate_and_empty_cases_still_render() {
        let task = PromptTaskSpec::new("answer questions");
        let gradient = DeterministicGradientPolicy
            .gradient(&task, &PromptCandidate::default(), &[])
            .await
            .unwrap();
        assert_eq!(
            gradient,
            "objective: answer questions\ncandidate: (unnamed candidate)\ngradient:\nno failing cases"
        );
        assert_eq!(
            DeterministicGradientPolicy,
            DeterministicGradientPolicy::new()
        );
    }

    #[test]
    fn rendering_is_deterministic_and_ignores_case_order_insensitive_content() {
        let task = PromptTaskSpec::new("objective");
        let candidate = PromptCandidate::new("prompt", "seed").with_id("c");
        let bad = [failing("a", "first"), failing("b", "second")];
        let first = render_gradient(&task, &candidate, &bad);
        assert_eq!(first, render_gradient(&task, &candidate, &bad));
        // The rendered case order follows the slice, so the caller's selection
        // order is what the artifact reports.
        let reversed = render_gradient(&task, &candidate, &[bad[1].clone(), bad[0].clone()]);
        assert_ne!(first, reversed);
        assert!(reversed.contains("case 0: b"), "{reversed}");
    }
}
