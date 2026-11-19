//! Process-level infrastructure: the command runner, git operations, CI gates,
//! the Python toolchain, and subprocess program evaluation.
//!
//! This module is the real replacement for the `subprocess`/`git` halves of the
//! Python RSI infrastructure (`harness_rsi/auto_harness/infra`,
//! `harness_rsi/auto_harness/stages/verify.py`,
//! `artifact_rsi/program_opt/{runtime,execution}.py`). It exposes these layers:
//!
//! - [`command`] — a typed [`CommandSpec`] / [`CommandOutcome`] seam with a real
//!   process runner ([`ProcessCommandRunner`]) and a scripted test double
//!   ([`FakeCommandRunner`]).
//! - [`git`] — [`GitOperations`], which runs `git` through that seam and parses
//!   branch, status, diff, worktree, push, and cherry-pick results as data.
//! - [`ci`] — [`CiGateRunner`]: gate configuration, command normalisation,
//!   changed-line scoping, and failure-output sanitising.
//! - [`pytest`] — pure parsers for pytest's JSON and `JUnit` reports.
//! - [`python`] — [`PythonToolchain`]: virtualenv creation, `pip`, and pytest
//!   invocation.
//! - [`subprocess`] — [`SubprocessEvaluator`]: stage a candidate program, run
//!   it, and parse its reply.
//!
//! Exit codes are data, never errors; only a missing executable, an I/O
//! failure, or cancellation is returned as an [`openjiuwen_common::error::Error`].
//! Every layer builds on the command seam rather than spawning processes itself.

pub mod ci;
pub mod command;
pub mod git;
pub mod pytest;
pub mod python;
pub mod subprocess;

pub use ci::{
    changed_line_ranges, default_gates_yaml, extract_commits, filter_findings, gate_outcome,
    normalize_command, parse_gates, sanitize_failure_output, shell_command, CiGateReport,
    CiGateRunner, CiGateSpec, GateKind, GateOutcome, FAIL_MARKER, OUTPUT_LIMIT,
};
pub use command::{
    decode_output, CommandOutcome, CommandRunner, CommandSpec, FakeCommandRunner, FakeReply,
    ProcessCommandRunner, DEFAULT_COMMAND_TIMEOUT, MAX_OUTPUT_BYTES,
};
pub use git::{
    build_auth_env, scrub_credentials, slugify_topic, GitBranchResult, GitCherryPick,
    GitOperations, GitStatus, GitStepResult,
};
pub use pytest::{parse_json_report, parse_junit_xml, parse_report, PytestReport, TestFailure};
pub use python::{
    discover_python, pytest_args, PipInstallOutcome, PytestOutcome, PythonToolchain, ReportFormat,
    ToolchainSetup,
};
pub use subprocess::{
    parse_files_reply, parse_tagged_reply, SubprocessConfig, SubprocessEvaluator, SubprocessResult,
};
