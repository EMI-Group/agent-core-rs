//! Offline-first evaluation of skills against test cases.
//!
//! Ports the evaluation artifacts of `openjiuwen/dev_tools/skill_evaluator`
//! (case model, design/functional/safety reports, aggregation formula) plus the
//! `skills/skill_tester/scripts/run_eval_query.py` single-query harness onto a
//! typed Rust toolkit that runs without a model unless one is supplied.
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
//! # Optional LLM judging
//!
//! [`judge_scoring`] adds the Python model-judged path: [`design_rubric`] scores
//! the seven design dimensions and [`safety_rubric`] scores four risk
//! dimensions, both through the shared [`crate::judge`] seam. [`SkillJudge`]
//! degrades to the deterministic scorers on any model or parse failure and marks
//! the verdict source, while [`SkillEvaluator::with_judge`] plus
//! [`SkillEvaluator::evaluate_with_judge`] record that provenance in
//! [`JudgeRecord`]. A matching [`SafetyRule`] always blocks, even when the model
//! verdict would pass.
//!
//! # Evaluation harness
//!
//! [`eval`] loads cases from JSON or JSONL ([`load_cases_json`],
//! [`load_cases_file`]), runs them through an [`EvalAgent`] ([`ModelAgent`] or a
//! [`CaseRunnerAgent`]), and emits an [`EvalRun`] as JSON, JSONL, or files.
//! [`EvalHarness::run_query`] ports `run_eval_query.py`: one agent call, an
//! answer file, and a JSON report beside it, with caller-supplied timestamps.
//!
//! # Determinism and deviations
//!
//! The deterministic path ([`SkillEvaluator::evaluate`]) is a pure function of
//! its inputs: no network, no environment reads, and a report timestamp supplied
//! by the caller or read from the wall clock only at the documented entry point.
//! The judge and harness paths touch a model only when the caller supplies one,
//! and both keep the Python report field names and the `0.40/0.60` weighting.
//!
//! [`SkillDefinition`]: crate::skill_creator::SkillDefinition
pub mod assertion;
pub mod case;
pub mod eval;
pub mod evaluator;
pub mod judge_scoring;
pub mod report;
pub mod runner;
pub mod scoring;

pub use assertion::{Assertion, AssertionOutcome};
pub use case::{CaseStatus, EvalCase};
pub use eval::{
    load_cases_file, load_cases_json, resolve_output_file, CaseRunnerAgent, EvalAgent, EvalHarness,
    EvalRun, ModelAgent, QueryOutcome, QueryRequest,
};
pub use evaluator::{EvaluatorConfig, SkillEvaluator};
pub use judge_scoring::{
    design_report_from_verdict, design_rubric, safety_rubric, DesignJudgement, SafetyJudgement,
    SkillJudge,
};
pub use report::{
    grade_for, CaseResult, CaseSummary, DesignReport, EvaluationReport, EvaluatorScore,
    EvaluatorScores, FinalScore, FunctionalReport, JudgeRecord, SafetyFinding, SafetyReport,
    ScoreDimension,
};
pub use runner::{CaseRunner, ModelCaseRunner, ScriptedCaseRunner};
pub use scoring::{aggregate, scan_safety, score_design, SafetyRule};
