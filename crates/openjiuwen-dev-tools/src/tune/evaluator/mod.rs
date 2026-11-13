//! The evaluator seam: run a candidate prompt, score the answer, keep the case.
//!
//! Ports the role of Python `openjiuwen/dev_tools/tune/evaluator/evaluator.py`
//! but splits it into three small seams so every pipeline can be driven offline:
//!
//! - [`PromptRunner`] turns `(candidate prompt, case)` into an answer map.
//!   `ModelPromptRunner` calls a `Model` with the candidate as the system
//!   message and the rendered case inputs as the user message;
//!   [`ScriptedPromptRunner`] replays canned answers for offline tests.
//! - [`CaseScorer`] turns `(case, answer)` into a score plus a reason.
//!   [`LabelScorer`] is the deterministic default: it compares the answer with
//!   the case label key by key.
//! - [`CaseEvaluator`] composes the two into an [`EvaluatedCase`].
//!   [`RunnerEvaluator`] evaluates through any [`PromptRunner`],
//!   [`ModelEvaluator`] is the model-backed preset, and
//!   [`DeterministicEvaluator`] evaluates without any model so prompt search has
//!   a real, repeatable landscape to climb.
//!
//! Each seam also has a traced form, so a run's tool use stays observable:
//! `ModelPromptRunner` executes the tools a case declares through [`ToolStubs`]
//! and records every call — failures included — in a [`ToolTrace`] bounded by
//! [`DEFAULT_MAX_TOOL_ROUNDS`]; [`CaseEvaluator::evaluate_traced`] and
//! [`CaseEvaluator::score_cases_traced`] return those traces, [`tool_usage`]
//! aggregates them, and [`ToolUsageScorer`] requires that the declared tools
//! were actually used.
//!
//! [`ToolStubs`]: crate::tune::tool::ToolStubs
//! [`ToolTrace`]: crate::tune::tool::ToolTrace
//!
//! Python's `DefaultEvaluator` asks a model for a `{"result": bool, "reason": …}`
//! verdict and gives everything the same all-or-nothing score. The port keeps
//! the seams and the `0.0..=1.0` range but makes the default scorer
//! deterministic and graded, so a search can tell a near miss from a wrong
//! answer. A model can still judge cases: [`ModelEvaluator`] plus a custom
//! [`CaseScorer`] (or [`RunnerEvaluator`] with a model-runner and a custom
//! scorer) is the equivalent hook.
//!
//! # Scoring bands of [`LabelScorer`]
//!
//! For every key of the case label the answer value is compared with the
//! expected value, after trimming:
//!
//! | Condition                              | Score |
//! | -------------------------------------- | ----- |
//! | exact match                            | 1.00  |
//! | case-insensitive match                 | 0.90  |
//! | either value contains the other        | 0.75  |
//! | token-set Jaccard in `0.5..=1.0`       | 0.30 + 0.30·(J−0.5)/0.5 |
//! | otherwise                              | 0.00  |
//!
//! The case score is the mean over the label keys. When the answer carries
//! exactly one entry (for example the `output` key a plain-text model answer
//! produces) that entry stands in for any missing label key, so a model that
//! answers in prose is still scored instead of always getting `0.0`.
mod deterministic;
mod runner;
mod scorer;
mod seam;

pub use self::deterministic::{DeterministicEvaluator, DeterministicScore};
pub use self::runner::{
    render_case_inputs, ModelPromptRunner, PromptCall, PromptRunner, ScriptedPromptRunner,
    DEFAULT_MAX_TOOL_ROUNDS,
};
pub use self::scorer::{CaseScorer, LabelScorer, ToolUsageScorer};
pub use self::seam::{tool_usage, CaseEvaluator, ModelEvaluator, RunnerEvaluator};

#[cfg(test)]
mod tests;
