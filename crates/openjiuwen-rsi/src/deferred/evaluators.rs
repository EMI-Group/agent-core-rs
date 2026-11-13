//! Deferred LLM-as-judge evaluator.
//!
//! The benchmark runtimes are **implemented**: [`SweBenchEvaluator`] and
//! [`TerminalBenchEvaluator`] live in
//! [`crate::harness_rsi::evaluator`] and drive the
//! official harnesses through the crate's command seam. Only the judge model is
//! left here, because building a guarded judge agent and its prompts belongs to
//! the harness layer.
//!
//! [`SweBenchEvaluator`]: crate::harness_rsi::evaluator::SweBenchEvaluator
//! [`TerminalBenchEvaluator`]: crate::harness_rsi::evaluator::TerminalBenchEvaluator

use serde_json::Value;

use openjiuwen_common::error::Result;

use super::not_implemented;

/// The surface name used by [`LlmJudgeEvaluator`]'s errors.
const SURFACE_LLM_JUDGE: &str = "LLM judge evaluator";

/// LLM-as-judge grader — **deferred**.
///
/// Mirrors Python's `harness_rsi.evaluator.judger`, which builds a guarded judge
/// `DeepAgent` and parses its verdict. Judge-model plumbing and prompt
/// construction belong to the harness layer and are not portable here.
///
/// The portable half of that package is not deferred: verdict parsing, the
/// weighted behaviour/penalty contract, and grading a case outcome live in
/// [`crate::harness_rsi::evaluator`], so a caller
/// that supplies the judge text itself is fully served.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LlmJudgeEvaluator;

impl LlmJudgeEvaluator {
    /// Mirrors the judge's constructor; always reports that it is deferred.
    ///
    /// # Errors
    ///
    /// Always returns [`openjiuwen_common::error::Error::Unsupported`] (`LLM judge evaluator`).
    pub fn new() -> Result<Self> {
        Err(not_implemented(
            SURFACE_LLM_JUDGE,
            "needs guarded judge-model plumbing and prompt construction in the harness layer",
        ))
    }

    /// Mirrors `LlmJudgeEvaluator.evaluate`; always reports that it is deferred.
    ///
    /// # Errors
    ///
    /// Always returns [`openjiuwen_common::error::Error::Unsupported`] (`LLM judge evaluator`).
    pub fn evaluate(_request: Value) -> Result<Value> {
        Err(not_implemented(
            SURFACE_LLM_JUDGE,
            "needs guarded judge-model plumbing and prompt construction in the harness layer",
        ))
    }
}
