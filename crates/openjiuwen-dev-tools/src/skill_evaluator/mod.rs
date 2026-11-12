//! Offline, deterministic evaluation of skills against test cases.
//!
//! Ports the evaluation artifacts of `openjiuwen/dev_tools/skill_evaluator`
//! (case model, design/functional/safety reports, aggregation formula) onto a
//! typed Rust harness that runs without a model unless one is supplied.
//!
//! # Pipeline
//!
//! 1. **Design review** ([`score_design`]) applies a seven-dimension structural
//!    rubric to a [`SkillDefinition`]: knowledge volume, section/numbered-step
//!    craft, `SKILL.md` spec compliance and round-trip fidelity, resource
//!    references, heading structure, practical lists/code, and output
//!    specification. The dimension maxima sum to 100 points and the design score
//!    is their sum.
//! 2. **Functional evaluation** runs each [`EvalCase`] through a [`CaseRunner`]
//!    (scripted or model-backed) and checks its parsed [`Assertion`]s against the
//!    answer, producing a [`FunctionalReport`] whose summary carries the pass
//!    rate ([`SkillEvaluator`] drives the whole pipeline).
//! 3. **Safety gate** ([`scan_safety`]) scans the rendered document with
//!    caller-supplied [`SafetyRule`] regexes. Each match costs 25 points and
//!    any `critical` finding blocks unconditionally.
//! 4. **Aggregation** ([`aggregate`]) weights design at `0.40` and functional at
//!    `0.60`, yielding the [`FinalScore`]: `COMPLETED` when both gates pass,
//!    otherwise `BLOCKED_AT_E1_LLM_JUDGE` or `BLOCKED_AT_SAFETY`.
//!
//! # Determinism and deviations
//!
//! Every step is a pure function of its inputs: no network, no environment
//! reads, no wall clock. Python graded design by prompting a model with a prose
//! rubric and ran a shell harness for cases; the Rust port uses deterministic
//! structure checks plus expectation assertions, keeping the report field names
//! and the weighting formula. A model-supplied design/safety judgement can be
//! layered on later through [`score_design`]'s inputs and the [`Assertion`]
//! traits without changing the report types.
//!
//! [`SkillDefinition`]: crate::skill_creator::SkillDefinition
pub mod assertion;
pub mod case;
pub mod evaluator;
pub mod report;
pub mod runner;
pub mod scoring;

pub use assertion::{Assertion, AssertionOutcome};
pub use case::{CaseStatus, EvalCase};
pub use evaluator::{EvaluatorConfig, SkillEvaluator};
pub use report::{
    grade_for, CaseResult, CaseSummary, DesignReport, EvaluationReport, EvaluatorScore,
    EvaluatorScores, FinalScore, FunctionalReport, SafetyFinding, SafetyReport, ScoreDimension,
};
pub use runner::{CaseRunner, ModelCaseRunner, ScriptedCaseRunner};
pub use scoring::{aggregate, scan_safety, score_design, SafetyRule};
