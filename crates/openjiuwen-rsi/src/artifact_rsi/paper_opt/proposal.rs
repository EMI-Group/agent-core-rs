//! The proposal seam: one round's candidate papers.
//!
//! Python's tree provider asks a `ManagerRuntime` pipeline (six `DeepAgent`
//! stage agents) to rewrite the paper, then scores the compiled result. That
//! model-driven step is out of scope for the offline port; this module keeps the
//! *shape* of it:
//!
//! - [`PaperProposer`] is the seam a model-driven proposer implements — it
//!   receives the frontier body and returns whole candidate papers;
//! - [`TemplatePaperProposer`] is a deterministic offline implementation used by
//!   [`PaperOptimizer`](super::orchestrator::PaperOptimizer)'s default
//!   constructor and by tests.
//!
//! A candidate carries its own paper body (the LaTeX source), the rubric samples
//! its reviewer produced, and the pipeline/artifact status the adoption gate
//! needs. The orchestrator never mutates a candidate.
use super::scoring::RubricSample;
use super::tree::PaperChange;
use async_trait::async_trait;
use openjiuwen_common::error::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fmt::Write as _;
/// Sections the template proposer revises when the caller injects none.
pub const DEFAULT_SECTIONS: [&str; 6] = [
    "abstract",
    "introduction",
    "method",
    "experiments",
    "results",
    "discussion",
];
/// Status a completed paper pipeline reports (Python `TerminalReport.status`).
pub const PIPELINE_COMPLETE: &str = "complete";
/// One proposed paper version and everything the round needs to judge it.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct PaperCandidate {
    /// Proposer-assigned id, normally `<parent>/p<iteration>/v<n>`.
    pub candidate_id: String,
    /// Id of the tree node this candidate revises, if any.
    pub parent_id: Option<String>,
    /// One-line description of the revision.
    pub summary: String,
    /// The paper source (LaTeX) this candidate would produce.
    pub body: String,
    /// Logical changes the candidate claims to have made.
    pub changes: Vec<PaperChange>,
    /// Rubric samples the reviewer produced for this candidate.
    pub samples: Vec<RubricSample>,
    /// Whether the compiled paper artifact exists.
    pub required_artifact_present: bool,
    /// Terminal status of the paper pipeline (`complete`, `failed`, …).
    pub pipeline_status: String,
    /// Caller annotations, forwarded verbatim.
    pub metadata: BTreeMap<String, Value>,
}
/// Everything one proposal round tells a proposer.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct PaperProposalRequest {
    /// 1-based optimisation iteration.
    pub iteration: u32,
    /// Id of the frontier node the candidates should revise, if any.
    pub frontier: Option<String>,
    /// Body of the frontier paper (empty for a fresh task).
    pub parent_body: String,
    /// Number of candidates requested (`PaperOptConfig::beam_width`).
    pub count: usize,
    /// Node ids already present in the tree, so a proposer can avoid a clash.
    pub existing_ids: Vec<String>,
}
/// The model-driven seam: proposes one round of candidate papers.
///
/// A real proposer drives the Python six-stage pipeline (or an equivalent agent
/// flow) and returns the generated paper plus its rubric samples. Tests and
/// offline runs use [`TemplatePaperProposer`] instead.
#[async_trait]
pub trait PaperProposer: Send + Sync {
    /// Returns the proposer's human-readable name.
    fn name(&self) -> &str;
    /// Proposes up to `request.count` candidates for one iteration.
    ///
    /// # Errors
    ///
    /// Implementations surface their backend failures here; the orchestrator
    /// treats a proposal error as fatal for the run.
    async fn propose(&self, request: PaperProposalRequest) -> Result<Vec<PaperCandidate>>;
}
/// Deterministic offline proposer.
///
/// Each variant appends one section-revision block to the parent body and
/// reports a single `revise` change. Ids are `<parent>/p<iteration>/v<n>`, where
/// `<parent>` is the frontier id (or `root` when there is none) and `n` counts
/// from zero. Rubric samples are taken cyclically from the injected list — the
/// variant at index `n` gets `samples[n % samples.len()]` — so variants are
/// distinguishable without a model. This is explicitly **not** a paper writer: it
/// exists to exercise the loop, and a model-driven [`PaperProposer`] replaces it
/// unchanged.
#[derive(Debug, Clone, PartialEq)]
pub struct TemplatePaperProposer {
    sections: Vec<String>,
    samples: Vec<RubricSample>,
    pipeline_status: String,
    required_artifact_present: bool,
}
impl Default for TemplatePaperProposer {
    fn default() -> Self {
        Self::new()
    }
}
impl TemplatePaperProposer {
    /// Builds a proposer with the default sections and one uniform rubric sample.
    #[must_use]
    pub fn new() -> Self {
        Self {
            sections: DEFAULT_SECTIONS
                .iter()
                .map(|section| (*section).to_owned())
                .collect(),
            samples: vec![RubricSample::uniform(7.0)],
            pipeline_status: PIPELINE_COMPLETE.to_owned(),
            required_artifact_present: true,
        }
    }
    /// Replaces the section templates variants cycle through.
    ///
    /// An empty list is ignored, so the proposer always has a section to revise.
    #[must_use]
    pub fn with_sections(mut self, sections: Vec<String>) -> Self {
        if !sections.is_empty() {
            self.sections = sections;
        }
        self
    }
    /// Replaces the rubric samples variants cycle through.
    ///
    /// An empty list is ignored, so every candidate keeps a scorable sample set.
    #[must_use]
    pub fn with_samples(mut self, samples: Vec<RubricSample>) -> Self {
        if !samples.is_empty() {
            self.samples = samples;
        }
        self
    }
    /// Sets the terminal pipeline status every candidate reports.
    ///
    /// Any value other than [`PIPELINE_COMPLETE`] makes the adoption gate prune
    /// the candidate with `pipeline_<status>`.
    #[must_use]
    pub fn with_pipeline_status(mut self, status: impl Into<String>) -> Self {
        self.pipeline_status = status.into();
        self
    }
    /// Reports a missing compiled paper, so the gate prunes with
    /// `paper_generation`.
    #[must_use]
    pub fn without_artifact(mut self) -> Self {
        self.required_artifact_present = false;
        self
    }
}
#[async_trait]
impl PaperProposer for TemplatePaperProposer {
    // The Python trait returns `str`; `&str` here keeps model-driven
    // implementations free to return a borrowed name.
    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        "template"
    }
    async fn propose(&self, request: PaperProposalRequest) -> Result<Vec<PaperCandidate>> {
        if request.count == 0 {
            return Ok(Vec::new());
        }
        let parent_label = request
            .frontier
            .clone()
            .unwrap_or_else(|| "root".to_owned());
        let mut candidates = Vec::new();
        let mut variant = 0_usize;
        let limit = request
            .count
            .saturating_add(request.existing_ids.len())
            .max(1);
        while candidates.len() < request.count && variant < limit {
            let candidate_id = format!("{parent_label}/p{}/v{variant}", request.iteration);
            let section = &self.sections[variant % self.sections.len()];
            variant += 1;
            if request.existing_ids.iter().any(|id| id == &candidate_id) {
                continue;
            }
            let mut body = request.parent_body.clone();
            let _ = writeln!(
                body,
                "% --- paper revision p{}/v{} ({section}) ---",
                request.iteration,
                variant - 1
            );
            let change = PaperChange {
                group: "paper".to_owned(),
                operation: "revise".to_owned(),
                function: "reporting".to_owned(),
                target: section.clone(),
                summary: format!("Appended a {section} revision block."),
            };
            let sample = self.samples[(variant - 1) % self.samples.len()];
            let mut metadata = BTreeMap::new();
            metadata.insert("variant".to_owned(), Value::from(variant - 1));
            metadata.insert("section".to_owned(), Value::from(section.clone()));
            candidates.push(PaperCandidate {
                candidate_id,
                parent_id: request.frontier.clone(),
                summary: format!("Revision of {section}"),
                body,
                changes: vec![change],
                samples: vec![sample],
                required_artifact_present: self.required_artifact_present,
                pipeline_status: self.pipeline_status.clone(),
                metadata,
            });
        }
        Ok(candidates)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn template_proposals_are_deterministic_and_unique() {
        let proposer = TemplatePaperProposer::new();
        let request = PaperProposalRequest {
            iteration: 2,
            frontier: Some("n001".to_owned()),
            parent_body: "seed body".to_owned(),
            count: 2,
            existing_ids: vec!["n001/p2/v0".to_owned()],
        };
        let first = proposer.propose(request.clone()).await.expect("propose");
        let second = proposer.propose(request).await.expect("propose again");
        assert_eq!(first, second);
        assert_eq!(first.len(), 2);
        assert_eq!(first[0].candidate_id, "n001/p2/v1");
        assert_eq!(first[1].candidate_id, "n001/p2/v2");
        assert!(first[0].body.starts_with("seed body"));
        assert_eq!(first[0].changes.len(), 1);
        assert_eq!(first[0].changes[0].operation, "revise");
        assert_eq!(first[0].pipeline_status, PIPELINE_COMPLETE);
        assert!(first[0].required_artifact_present);
    }
    #[tokio::test]
    async fn samples_cycle_and_status_overrides_apply() {
        let proposer = TemplatePaperProposer::new()
            .with_samples(vec![RubricSample::uniform(4.0), RubricSample::uniform(9.0)])
            .with_pipeline_status("failed")
            .without_artifact();
        let request = PaperProposalRequest {
            iteration: 1,
            frontier: None,
            parent_body: String::new(),
            count: 3,
            existing_ids: Vec::new(),
        };
        let candidates = proposer.propose(request).await.expect("propose");
        assert_eq!(candidates[0].samples[0].general.overall, 4.0);
        assert_eq!(candidates[1].samples[0].general.overall, 9.0);
        assert_eq!(candidates[2].samples[0].general.overall, 4.0);
        assert_eq!(candidates[0].candidate_id, "root/p1/v0");
        assert_eq!(candidates[0].pipeline_status, "failed");
        assert!(!candidates[0].required_artifact_present);
    }
}
