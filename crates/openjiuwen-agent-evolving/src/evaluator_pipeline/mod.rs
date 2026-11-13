//! The evaluator pipeline: task-in-container evaluation over Docker + git.
//!
//! Ports Python `openjiuwen.agent_evolving.evaluator.evaluator_pipeline`: the
//! records (`models.py`), configuration (`config.py`), the Docker environment
//! lifecycle (`docker_env.py`), git workspace preparation and the pipeline driver.
//!
//! Every external side effect goes through the [`CommandRunner`] seam, so the
//! pipeline is fully testable offline with a fake runner; the real runner shells
//! out to `docker`/`git` through `tokio::process`.
//!
//! - [`CommandRunner`] / [`ProcessCommandRunner`] / [`FakeCommandRunner`] —
//!   the seam, the real implementation, and a scripted implementation for tests.
//! - [`ExecResult`] / [`CommandSpec`] — what a command produced and what it was.
//! - [`DockerEnvironment`] — build/start/exec/copy/stop with typed lifecycle
//!   errors.
//! - [`WorkspacePreparer`] / [`GitWorkspaceSpec`] / [`PreparedWorkspace`] —
//!   clone, fetch, checkout, submodules and patch application.
//! - [`ScoreSpec`] / [`parse_score`] / [`ScoredRun`] — exit-code, JSON, and
//!   test-summary scoring, plus the [`parse_last_json`] / [`parse_test_counts`]
//!   helpers.
//! - [`EvaluationTask`] / [`TaskReport`] — the evaluated task and its report.
//! - [`EvaluationPipeline`] / [`PipelineConfig`] / [`PipelineSummary`] — the
//!   driver, its configuration, and batch aggregation.
//!
//! The Python package's agent adapters (`base.py`), LLM evolution loop, skill
//! manager, and mirror-rewriting image builds are not ported: they need live
//! agents, a model gateway, and network access, so they cannot be exercised
//! offline.
//!
//! When `docker` is not installed the pipeline reports
//! [`openjiuwen_common::error::Error::Unsupported`] naming the missing binary, so
//! the failure is explicit instead of a silent no-op.
mod command;
mod docker;
mod git;
mod models;
mod pipeline;
mod scoring;

pub use command::{find_program, CommandRunner, FakeCommandRunner, ProcessCommandRunner};
pub use docker::{container_name, DockerEnvironment, CONTAINER_NAME_PREFIX, DEFAULT_TIMEOUT_MS};
pub use git::{PreparedWorkspace, WorkspacePreparer, PATCH_FILE_NAME};
pub use models::{
    sanitize_task_id, CommandSpec, EvaluationTask, ExecResult, GitWorkspaceSpec, TaskReport,
    MAX_TASK_ID_SLUG_CHARS,
};
pub use pipeline::{
    EvaluationPipeline, PipelineConfig, PipelineSummary, MAX_ATTEMPTS, MIN_ATTEMPTS,
};
pub use scoring::{
    parse_last_json, parse_score, parse_test_counts, ScoreSpec, ScoredRun, TestCounts,
};
