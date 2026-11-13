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
//! - [`CommandRunner`] / [`ProcessCommandRunner`] / [`ExecResult`] — the seam.
//! - [`DockerEnvironment`] — build/start/exec/stop with typed lifecycle errors.
//! - [`EvaluationPipeline`] — workspace setup, test execution, score parsing.
//!
//! When `docker` is not installed the pipeline reports
//! [`openjiuwen_common::error::Error::Unsupported`] naming the missing binary, so
//! the failure is explicit instead of a silent no-op.
