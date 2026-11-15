//! The matcher request/response protocol: the adapter between a model and the
//! deterministic matcher.
//!
//! Ports Python `openjiuwen.symphony.orchestration.graph.matcher.protocol` — the pure
//! functions that build the compact document a matcher model answers and that expand and
//! validate its reply. Nothing here calls a model: the model-calling side is
//! [`OntologyMatcher`](crate::llm::ontology::OntologyMatcher), and this module is the
//! contract between the two.
//!
//! - [`protocol`] builds the request: one entry per
//!   [`RelationCandidate`](crate::orchestration::RelationCandidate), numbered `c1..cN`, with
//!   the skill briefs, the truncated port descriptions and the derived directional
//!   evidence.
//! - [`validate`] expands a compact reply into full matches, validates every endpoint,
//!   confidence, relation and supporting field against the candidate evidence, and applies
//!   the per-relation thresholds.
//!
//! ```no_run
//! use openjiuwen_symphony::llm::ontology::OntologyMatcher;
//! use openjiuwen_symphony::model::CapabilityDescriptor;
//! use openjiuwen_symphony::orchestration::matcher::{
//!     build_match_request, parse_match_response_with_evidence, MatchThresholds,
//! };
//! use openjiuwen_symphony::orchestration::CandidateGenerator;
//! use serde_json::Value;
//!
//! # fn main() -> openjiuwen_symphony::error::Result<()> {
//! let capabilities: Vec<CapabilityDescriptor> = Vec::new(); // the catalog under construction
//! let matcher = OntologyMatcher::new(capabilities.clone());
//! let candidates = CandidateGenerator::default().generate(&capabilities);
//!
//! // 1. The compact document the model is asked to judge.
//! let request = build_match_request(&capabilities, &candidates, false);
//!
//! // 2. The model's reply, produced outside this crate; `matcher_signature()` names the
//! //    exact model/prompt pair `matcher.match_candidates(&candidates).await` asks with.
//! let payload: Value = serde_json::json!({"matches": [{"id": "c1", "confidence": 0.9}]});
//!
//! // 3. The validated verdicts, judged against the evidence the model actually saw.
//! let outcome = parse_match_response_with_evidence(
//!     &payload,
//!     &candidates,
//!     &request.evidence(),
//!     &MatchThresholds::default(),
//! );
//! assert!(outcome.matches.is_empty());
//! assert!(!matcher.matcher_signature().is_empty());
//! # Ok(())
//! # }
//! ```
//!
//! The pipeline is fail-soft: every malformed item becomes a
//! [`validate::GraphDiagnostic`](crate::orchestration::matcher::GraphDiagnostic) instead of
//! an error, a rejected match stays in the outcome with its errors, and the model never
//! chooses the relation (it is always
//! [`CAN_FEED`](crate::graph::CAN_FEED)) nor the ports (they must be a subset of the
//! evidence that was shown).

pub mod protocol;
pub mod validate;

pub use protocol::{
    build_match_request, candidate_key, directional_evidence, truncate_chars, CandidateContext,
    CandidateEvidence, DirectionEvidence, IoBrief, MatchRequest, PortBrief, SkillBrief,
    DIRECTION_FORWARD, DIRECTION_REVERSE, MAX_DERIVED_PORTS, MAX_PORT_DESCRIPTION_LENGTH,
    MAX_REASON_LENGTH, MAX_SKILL_DESCRIPTION_LENGTH,
};
pub use validate::{
    diagnostics_by_candidate, matches_by_candidate, parse_match_response,
    parse_match_response_with, parse_match_response_with_evidence, GraphDiagnostic, MatchOutcome,
    MatchThresholds, PortMappingPair, SupportingFields, ValidatedMatch, DEFAULT_CAN_FEED_THRESHOLD,
    DIAGNOSTIC_DUPLICATE_CANDIDATE_ID, DIAGNOSTIC_INVALID_CANDIDATE_DIRECTION,
    DIAGNOSTIC_INVALID_MATCHES_PAYLOAD, DIAGNOSTIC_INVALID_MATCH_ITEM,
    DIAGNOSTIC_LOW_CONFIDENCE_LLM_MATCH, DIAGNOSTIC_REJECTED_LLM_MATCH,
    DIAGNOSTIC_UNKNOWN_CANDIDATE_ID, DIAGNOSTIC_UNSUPPORTED_CANDIDATE_DIRECTION,
};
