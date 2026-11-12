//! The candidate unit shared by harness optimisation and program search.
//!
//! A [`Candidate`] body is plain text, never a Python object: a YAML
//! configuration document for [`crate::harness_rsi::auto_harness`], or program
//! source for [`crate::artifact_rsi::program_opt`]. Keeping the body textual is
//! what lets both halves share one mutation / evaluation / selection protocol
//! and what makes a checkpoint a JSON document rather than a pickled graph.

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::evaluation::EvaluationReport;

/// Stable identifier of a [`Candidate`].
///
/// Ids are deterministic text (for example `seed/g1/v0`), never UUIDs, so the
/// mutation output and every ordering derived from a candidate set are
/// reproducible.
#[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CandidateId(String);

impl CandidateId {
    /// Creates an id from anything convertible into a `String`.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Returns the id as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consumes the id, returning the inner `String`.
    #[must_use]
    pub fn into_string(self) -> String {
        self.0
    }
}

impl fmt::Display for CandidateId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<&str> for CandidateId {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

impl From<String> for CandidateId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl AsRef<str> for CandidateId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

/// Lifecycle state of a [`Candidate`].
///
/// Python keeps the same vocabulary (`pending`, `evaluated`, `invalid`,
/// `selected`); the wire names are `snake_case`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateStatus {
    /// Proposed, not evaluated yet.
    #[default]
    Pending,
    /// Evaluated successfully; [`Candidate::evaluation`] holds the report.
    Evaluated,
    /// Evaluation failed or the report was invalid.
    Invalid,
    /// Chosen by a [`crate::evolution::SelectionStrategy`].
    Selected,
}

/// One candidate configuration or program.
///
/// `body` is opaque text (see the module docs). `metadata` carries
/// caller-specific annotations and is never interpreted by the protocol.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct Candidate {
    /// The candidate's deterministic id.
    pub id: CandidateId,
    /// The candidate this one was mutated from, if any.
    pub parent: Option<CandidateId>,
    /// The search generation that produced this candidate; `0` for a seed.
    pub generation: u32,
    /// The candidate body: configuration YAML or program source.
    pub body: String,
    /// Human-readable summary of what the mutation changed.
    pub change_summary: String,
    /// The candidate's lifecycle state.
    pub status: CandidateStatus,
    /// The evaluation report, once the candidate has been graded.
    pub evaluation: Option<EvaluationReport>,
    /// Caller-specific annotations; never interpreted here.
    pub metadata: BTreeMap<String, Value>,
}

impl Candidate {
    /// Creates a pending, generation-0 candidate with an empty summary.
    #[must_use]
    pub fn new(id: impl Into<CandidateId>, body: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            parent: None,
            generation: 0,
            body: body.into(),
            change_summary: String::new(),
            status: CandidateStatus::Pending,
            evaluation: None,
            metadata: BTreeMap::new(),
        }
    }

    /// Returns the candidate's evaluation score.
    ///
    /// An invalid candidate never scores, whatever report it carries; a valid
    /// candidate without a report (or with an entirely ungraded report) returns
    /// `None`.
    #[must_use]
    pub fn score(&self) -> Option<f64> {
        if self.status == CandidateStatus::Invalid {
            return None;
        }
        self.evaluation.as_ref().and_then(EvaluationReport::score)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn report(score: f64) -> EvaluationReport {
        EvaluationReport::from_results(vec![super::super::evaluation::CaseResult {
            case_id: "case".to_owned(),
            passed: true,
            score: Some(score),
            weight: 1.0,
            error: None,
            details: Value::Null,
        }])
    }

    #[test]
    fn candidate_id_round_trips_and_displays() {
        let id = CandidateId::new("seed/g1/v0");
        assert_eq!(id.as_str(), "seed/g1/v0");
        assert_eq!(id.to_string(), "seed/g1/v0");
        assert_eq!(CandidateId::from("seed"), CandidateId::new("seed"));
        assert_eq!(
            CandidateId::from(String::from("seed")),
            CandidateId::new("seed")
        );
        let encoded = serde_json::to_string(&id).expect("candidate id serializes");
        assert_eq!(encoded, "\"seed/g1/v0\"");
        let decoded: CandidateId = serde_json::from_str(&encoded).expect("candidate id parses");
        assert_eq!(decoded, id);
    }

    #[test]
    fn new_candidate_is_pending_generation_zero() {
        let candidate = Candidate::new("seed", "model: demo");
        assert_eq!(candidate.id.as_str(), "seed");
        assert_eq!(candidate.generation, 0);
        assert_eq!(candidate.status, CandidateStatus::Pending);
        assert!(candidate.parent.is_none());
        assert!(candidate.evaluation.is_none());
        assert!(candidate.metadata.is_empty());
        assert!(candidate.score().is_none());
    }

    #[test]
    fn invalid_candidates_never_score() {
        let mut candidate = Candidate::new("seed", "body");
        candidate.status = CandidateStatus::Invalid;
        candidate.evaluation = Some(report(0.9));
        assert!(candidate.score().is_none());
    }

    #[test]
    fn evaluated_candidates_expose_the_report_score() {
        let mut candidate = Candidate::new("seed", "body");
        candidate.status = CandidateStatus::Evaluated;
        candidate.evaluation = Some(report(0.75));
        assert_eq!(candidate.score(), Some(0.75));
    }

    #[test]
    fn candidate_deserializes_with_unknown_and_missing_fields() {
        let candidate: Candidate = serde_json::from_value(json!({
            "id": "seed",
            "body": "model: demo",
            "unknown_field": {"nested": true}
        }))
        .expect("candidate tolerates unknown fields");
        assert_eq!(candidate.id.as_str(), "seed");
        assert_eq!(candidate.status, CandidateStatus::Pending);
        assert_eq!(candidate.generation, 0);
        assert!(candidate.metadata.is_empty());
    }

    #[test]
    fn candidate_round_trips_through_json() {
        let mut candidate = Candidate::new("seed/g2/v1", "model: demo");
        candidate.parent = Some(CandidateId::new("seed"));
        candidate.generation = 2;
        candidate.change_summary = "deterministic variant".to_owned();
        candidate.status = CandidateStatus::Selected;
        candidate.evaluation = Some(report(0.5));
        candidate
            .metadata
            .insert("origin".to_owned(), json!("unit-test"));

        let encoded = serde_json::to_value(&candidate).expect("candidate serializes");
        let decoded: Candidate = serde_json::from_value(encoded).expect("candidate parses");
        assert_eq!(decoded, candidate);
    }

    #[test]
    fn candidate_status_wire_names_are_snake_case() {
        assert_eq!(
            serde_json::to_value(CandidateStatus::Selected).expect("status serializes"),
            json!("selected")
        );
        assert_eq!(CandidateStatus::default(), CandidateStatus::Pending);
    }
}
