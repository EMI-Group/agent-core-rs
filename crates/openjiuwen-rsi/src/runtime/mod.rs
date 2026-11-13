//! Process-level infrastructure: the command runner and git operations.
//!
//! This module is the real replacement for the `subprocess`/`git` halves of the
//! Python RSI infrastructure (`harness_rsi/auto_harness/infra`). It exposes two
//! layers:
//!
//! - [`command`] — a typed [`CommandSpec`] / [`CommandOutcome`] seam with a real
//!   process runner ([`ProcessCommandRunner`]) and a scripted test double
//!   ([`FakeCommandRunner`]).
//! - [`git`] — [`GitOperations`], which runs `git` through that seam and parses
//!   branch, status, diff, worktree, push, and cherry-pick results as data.
//!
//! Exit codes are data, never errors; only a missing executable, an I/O
//! failure, or cancellation is returned as an [`openjiuwen_common::error::Error`].
//! Higher layers (CI gates, pytest, evaluators) are expected to build on this
//! command seam rather than spawning processes themselves.

pub mod command;
pub mod git;

pub use command::{
    decode_output, CommandOutcome, CommandRunner, CommandSpec, FakeCommandRunner, FakeReply,
    ProcessCommandRunner,
};
pub use git::{
    build_auth_env, scrub_credentials, slugify_topic, GitBranchResult, GitCherryPick,
    GitOperations, GitStatus, GitStepResult,
};
