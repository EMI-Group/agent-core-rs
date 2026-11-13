//! Harness evaluation: case records, scoring contracts, benchmark evaluators
//! (SWE-bench, terminal-bench), and metrics collection.
//!
//! Maps to Python `openjiuwen.rsi.harness_rsi.evaluator` (v0.1.20). The port
//! owns the **portable** half of that package — everything that is data or
//! arithmetic — and leaves the model/Docker half behind seams:
//!
//! | Module | Contents | Python source |
//! |---|---|---|
//! | [`case`] | [`EvaluationCase`], [`CaseReference`], [`SweBenchCase`], [`Dataset`] (JSON/JSONL) | `case_backend.py`, `team_evaluator.py`, `data_loader.grading_contract` |
//! | [`scoring`] | verdict parsing, weighted behaviours, forbidden penalties, failure classification | `judger/scoring.py`, `transport.py`, `requirement_results.py` |
//! | [`runner`] | [`CaseExecutionBackend`], [`CaseRunner`], command backend, judge-grade bridge | `case_backend.py`, `case_runner.py` |
//! | [`swebench`] | official harness argv, report classification, failure literals | `swebench_runtime.py` |
//! | [`terminal_bench`] | harness argv, results parsing, command logs | `terminal_bench_runtime.py` |
//! | [`metrics`] | `summary.json` aggregation | `metrics_collector.py` |
//!
//! Deliberately **out of scope** (behind seams, never silently degraded): the
//! `DeepAgent` solver backend (`SingleHarnessExecutionBackend`, `TeamEvaluator`),
//! the LLM judge model (`judger/*`), and the SWE-bench Docker/WSL instance
//! runtimes. Those need harness construction, model plumbing, or container
//! topologies; callers implement [`runner::CaseExecutionBackend`] and hand the
//! result to [`runner::CaseRunner`], which grades exactly like Python.
//!
//! Everything here is offline and deterministic: the benchmark evaluators take a
//! `CommandRunner`, so tests inject the crate's `FakeCommandRunner` and write
//! the report/results files themselves.

pub mod case;
pub mod metrics;
pub mod runner;
pub mod scoring;
pub mod swebench;
pub mod terminal_bench;

pub use case::{
    AnswerRole, BehaviorSpec, CaseReference, Dataset, EvaluationCase, ForbiddenBehaviorSpec,
    PenaltyMode, SweBenchCase, DEFAULT_DATASET_ID,
};
pub use metrics::{
    collect_metrics, summarize_reports, summary_from_results, write_summary, CaseResultRecord,
    MetricsSummary,
};
pub use runner::{
    is_transient_error, CandidateCaseRunner, CaseCommandBuilder, CaseExecutionBackend,
    CaseExecutionRequest, CaseExecutionResult, CaseRunner, CaseTrace, CommandCaseBackend,
    ExecutionStatus, CANDIDATE_BODY_KEY, TRANSIENT_ERROR_MARKERS,
};
pub use scoring::{
    classify_judge_failure, parse_judge_verdict, requirement_results, score_verdict,
    verdict_from_case, BehaviorVerdict, ForbiddenHit, JudgeFailure, JudgeResult, JudgeVerdict,
    RequirementResult, ScoringContract, DEFAULT_FORBIDDEN_PENALTY, DEFAULT_SUCCESS_THRESHOLD,
    METHOD_JUDGE, METHOD_RESPONSE_SCORE, VERDICT_COMPLETED,
};
pub use swebench::{
    classification_from_report, failure_reason, infrastructure_failure_reason,
    per_instance_failure_reason, SweBenchClassification, SweBenchConfig, SweBenchDatasetEvaluator,
    SweBenchEvaluator, SweBenchOutcome, AGGREGATE_REPORT_MISSING,
    DEFAULT_INFRASTRUCTURE_FAILURE_PATTERNS, HARNESS_EXIT_REASON_PREFIX,
    INFRASTRUCTURE_PATTERN_PREFIX, PER_INSTANCE_REPORT_MISSING, REPORT_CLASSIFIED_ERROR,
    REPORT_CLASSIFIED_INCOMPLETE,
};
pub use terminal_bench::{
    parse_results, TerminalBenchCommandLog, TerminalBenchConfig, TerminalBenchDatasetEvaluator,
    TerminalBenchEvaluator, TerminalBenchOutcome, COMMAND_EXCERPT_LIMIT, RESULTS_FILE_MISSING,
    TASK_ID_KEY,
};

/// Reduces an id to the characters that are safe in a file name.
///
/// Lowercases, keeps ASCII alphanumerics, maps everything else to `-`, trims
/// leading/trailing `-`, and truncates to 48 characters (Python's `_safe_id`).
#[must_use]
pub fn sanitize_id(value: &str) -> String {
    let lowered = value.to_lowercase();
    let mapped: String = lowered
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '-' })
        .collect();
    let trimmed = mapped.trim_matches('-');
    if trimmed.is_empty() {
        return "case".to_owned();
    }
    trimmed.chars().take(48).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_id_keeps_file_names_portable() {
        assert_eq!(sanitize_id("django__django-11099"), "django--django-11099");
        assert_eq!(sanitize_id("  "), "case");
        assert_eq!(sanitize_id("/a/b/"), "a-b");
        assert!(sanitize_id(&"x".repeat(80)).chars().count() <= 48);
    }
}
