//! Deferred benchmark and judge evaluators.

use serde_json::Value;

use openjiuwen_common::error::Result;

use super::not_implemented;

/// The surface name used by [`SweBenchEvaluator`]'s errors.
const SURFACE_SWEBENCH: &str = "SWE-bench evaluator";
/// The surface name used by [`TerminalBenchEvaluator`]'s errors.
const SURFACE_TERMINAL_BENCH: &str = "terminal-bench evaluator";
/// The surface name used by [`LlmJudgeEvaluator`]'s errors.
const SURFACE_LLM_JUDGE: &str = "LLM judge evaluator";

/// `SWE-bench` instance runner — **deferred**.
///
/// Mirrors Python's `harness_rsi.evaluator.swebench_runtime`, which provisions
/// a per-instance container, applies the candidate patch, and runs the
/// benchmark's tests. That needs the `SWE-bench` docker/`WSL` runtimes and
/// network access, so it is an API-shaped stub.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SweBenchEvaluator;

impl SweBenchEvaluator {
    /// Mirrors the runtime's constructor; always reports that it is deferred.
    ///
    /// # Errors
    ///
    /// Always returns [`openjiuwen_common::error::Error::Unsupported`] (`SWE-bench evaluator`).
    pub fn new() -> Result<Self> {
        Err(not_implemented(
            SURFACE_SWEBENCH,
            "needs the SWE-bench docker/WSL instance runtimes and benchmark data",
        ))
    }

    /// Mirrors `SweBenchEvaluator.evaluate`; always reports that it is
    /// deferred.
    ///
    /// # Errors
    ///
    /// Always returns [`openjiuwen_common::error::Error::Unsupported`] (`SWE-bench evaluator`).
    pub fn evaluate(_request: Value) -> Result<Value> {
        Err(not_implemented(
            SURFACE_SWEBENCH,
            "needs the SWE-bench docker/WSL instance runtimes and benchmark data",
        ))
    }
}

/// Terminal-bench task runner — **deferred**.
///
/// Mirrors Python's `harness_rsi.evaluator.terminal_bench_runtime`, which drives
/// the terminal-bench harness against the candidate agent. That needs the
/// external runtime and network access.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TerminalBenchEvaluator;

impl TerminalBenchEvaluator {
    /// Mirrors the runtime's constructor; always reports that it is deferred.
    ///
    /// # Errors
    ///
    /// Always returns [`openjiuwen_common::error::Error::Unsupported`] (`terminal-bench evaluator`).
    pub fn new() -> Result<Self> {
        Err(not_implemented(
            SURFACE_TERMINAL_BENCH,
            "needs the external terminal-bench runtime and task data",
        ))
    }

    /// Mirrors `TerminalBenchEvaluator.evaluate`; always reports that it is
    /// deferred.
    ///
    /// # Errors
    ///
    /// Always returns [`openjiuwen_common::error::Error::Unsupported`] (`terminal-bench evaluator`).
    pub fn evaluate(_request: Value) -> Result<Value> {
        Err(not_implemented(
            SURFACE_TERMINAL_BENCH,
            "needs the external terminal-bench runtime and task data",
        ))
    }
}

/// LLM-as-judge grader — **deferred**.
///
/// Mirrors Python's `harness_rsi.evaluator.judger`, which builds a guarded judge
/// `DeepAgent` and parses its verdict. Judge-model plumbing and prompt
/// construction belong to the harness layer and are not portable here.
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
