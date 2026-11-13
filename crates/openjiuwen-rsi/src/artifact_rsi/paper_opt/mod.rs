//! Paper optimisation: rubric scoring, the greedy frontier tree, proposal, and
//! the deterministic orchestration loop.
//!
//! This is the portable core of Python `openjiuwen.rsi.artifact_rsi.paper_opt`
//! (v0.1.20). The six `DeepAgent` stage agents, the `ManagerRuntime` pipeline,
//! the `LaTeX`/pyright/Pillow toolchain and the model-environment plumbing are
//! **not** ported: they live behind the two injected seams
//! [`PaperProposer`] and
//! [`PaperScorer`], so a real pipeline plugs in
//! without changing the loop and the loop stays testable offline.
//!
//! | Module | Contents |
//! |---|---|
//! | [`config`] | rubric/`composite` weights and [`PaperOptConfig`] |
//! | [`scoring`] | [`RubricSample`], aggregation, dispersion, compact scoresheet |
//! | [`tree`] | [`PaperNode`], [`PaperTree`], [`adoption_gate`] |
//! | [`proposal`] | [`PaperProposer`] and the deterministic template proposer |
//! | [`orchestrator`] | [`PaperOptimizer`], run state, and the loop |
//!
//! ```
//! use openjiuwen_rsi::artifact_rsi::paper_opt::{
//!     adoption_gate, AdoptionOutcome, PaperOptConfig, ScoreBreakdown,
//! };
//!
//! let config = PaperOptConfig::default();
//! assert_eq!(config.max_iterations, 5);
//! let candidate = ScoreBreakdown { general: 7.0, experiment: 7.0, composite: 7.0, sample_count: 1 };
//! assert_eq!(
//!     adoption_gate(None, &candidate, true, None, "complete"),
//!     AdoptionOutcome::Adopted
//! );
//! ```
pub mod config;
pub mod orchestrator;
pub mod proposal;
pub mod scoring;
pub mod tree;
pub use config::{
    ExperimentWeights, GeneralWeights, PaperOptConfig, PaperWeights, WEIGHT_TOLERANCE,
};
pub use orchestrator::{
    PaperOptimizer, PaperRunOutcome, PaperScorer, PaperStatus, PaperTaskState, RubricPaperScorer,
};
pub use proposal::{
    PaperCandidate, PaperProposalRequest, PaperProposer, TemplatePaperProposer, DEFAULT_SECTIONS,
    PIPELINE_COMPLETE,
};
pub use scoring::{
    aggregate, aggregate_with_weights, compact_scoresheet, composite_score, dispersion,
    experiment_score, flags_dispersion, general_score, round_half_even, ExperimentScores,
    GeneralScores, RubricDispersion, RubricSample, ScoreBreakdown, DEFAULT_MAD_THRESHOLD,
    DEFAULT_RANGE_THRESHOLD,
};
pub use tree::{adoption_gate, AdoptionOutcome, PaperChange, PaperNode, PaperNodeType, PaperTree};
