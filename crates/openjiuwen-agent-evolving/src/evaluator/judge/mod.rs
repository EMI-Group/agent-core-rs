//! Model-backed LLM-as-judge scoring.
//!
//! Ports the model-judge part of Python `openjiuwen.agent_evolving.evaluator`
//! (`evaluator.DefaultEvaluator`, `metrics.llm_as_judge.LLMAsJudgeMetric`, and
//! `templates.LLM_METRIC_TEMPLATE`) onto this crate's own
//! [`Metric`](super::metrics::Metric) and [`Evaluator`](super::Evaluator) seams rather than transliterating the Python prompt-string
//! plumbing.
//!
//! The module keeps three layers apart:
//!
//! - [`JudgeRubric`] / [`JudgeCriterion`] — the criteria and score scale, as
//!   pure data.
//! - [`build_judge_messages`] and [`JudgeVerdict::parse`] — a deterministic
//!   prompt and a tolerant, deterministic response parser. Both are pure
//!   functions, so they are fully testable offline.
//! - [`LlmJudge`] / [`LlmJudgeEvaluator`] — the only code that talks to a model.
//!
//! # Determinism
//!
//! Nothing here reads the clock, a random source, or the environment: the prompt
//! is a pure function of `(case, prediction, rubric)` and parsing is a pure
//! function of the response text. Model calls are the only non-determinism, and
//! the unit tests substitute a scripted model.
//!
//! # Example
//!
//! ```
//! use std::collections::BTreeMap;
//!
//! use openjiuwen_agent_evolving::dataset::Case;
//! use openjiuwen_agent_evolving::evaluator::{build_judge_messages, JudgeRubric, JudgeVerdict};
//! use openjiuwen_core::foundation::prompt::Role;
//! use serde_json::json;
//!
//! # fn main() -> openjiuwen_common::error::Result<()> {
//! let case = Case::new(
//!     BTreeMap::from([("question".to_owned(), json!("2 + 2?"))]),
//!     BTreeMap::from([("answer".to_owned(), json!("4"))]),
//! )?;
//! let rubric = JudgeRubric::default_rubric();
//!
//! let messages = build_judge_messages(&case, &json!("4"), &rubric, 0);
//! assert_eq!(messages.len(), 2);
//! assert_eq!(messages[0].role, Role::System);
//!
//! let verdict = JudgeVerdict::parse(r#"{"score": 8, "reasoning": "close enough"}"#, &rubric)?;
//! assert!((verdict.score - 0.8).abs() < 1e-9);
//! assert!(verdict.passed);
//! # Ok(())
//! # }
//! ```

mod live;
mod prompt;
mod rubric;
mod verdict;

pub use live::{LlmJudge, LlmJudgeEvaluator};
pub use prompt::build_judge_messages;
pub use rubric::{JudgeCriterion, JudgeRubric};
pub use verdict::JudgeVerdict;

pub(crate) use live::{fallback_evaluated, judge_evaluated};

/// The score scale used when a rubric does not state one (Python's `0..10`).
const DEFAULT_MAX_SCORE: f64 = 10.0;

/// The score scale a rubric implies, falling back to 10.
fn scale_of(rubric: &JudgeRubric) -> f64 {
    if rubric.max_score.is_finite() && rubric.max_score > 0.0 {
        rubric.max_score
    } else {
        DEFAULT_MAX_SCORE
    }
}

#[cfg(test)]
mod test_support {
    use std::collections::BTreeMap;

    use serde_json::json;

    use crate::dataset::Case;

    pub(super) fn close(actual: f64, expected: f64) -> bool {
        (actual - expected).abs() < 1e-9
    }

    pub(super) fn case() -> Case {
        Case::new(
            BTreeMap::from([("question".to_owned(), json!("2 + 2?"))]),
            BTreeMap::from([("answer".to_owned(), json!("4"))]),
        )
        .unwrap()
    }
}
