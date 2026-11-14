//! `run`/`chat` wiring for the binary's own flags.
//!
//! The harness owns `run` and `chat`; this module adds only the two things a
//! terminal user needs that the harness grammar has no words for:
//!
//! - `--worktree <NAME>`: create (or reuse) a git worktree of the repository
//!   that contains the resolved workspace and run the prompt or REPL inside it.
//!   The worktree comes from [`openjiuwen_harness::worktree::WorktreeManager`],
//!   the same manager the agent's `worktree`/`enter_worktree` tools use, so the
//!   registry, slugging and lock policy are one implementation. The run's
//!   workspace is redirected to the worktree path; everything else — option
//!   resolution, permissions, sessions, the REPL — is untouched.
//! - `--watch`: start the workspace watcher and report reloads on stderr while
//!   the REPL runs (see [`crate::watch`]).
//!
//! Both default to today's behaviour: with neither flag the calls here are a
//! straight pass-through to [`openjiuwen_harness::cli::run_cli`].
use openjiuwen_common::error::Result;

use crate::cli::{ChatCliArgs, RunCliArgs};

/// Runs `openjiuwen run`, honouring `--worktree`.
///
/// # Errors
///
/// Returns the harness run's errors, plus a typed error when the workspace is
/// not inside a git repository or git refuses the worktree.
pub(crate) async fn run_run(args: &RunCliArgs, base: &openjiuwen_harness::cli::Cli) -> Result<()> {
    let _ = (args, base);
    Err(openjiuwen_common::error::Error::unsupported(
        "run --worktree is not wired yet",
    ))
}

/// Runs `openjiuwen chat`, honouring `--worktree` and `--watch`.
///
/// # Errors
///
/// Returns the harness REPL's errors, plus typed errors from the worktree or the
/// watcher as documented on [`run_run`].
pub(crate) async fn run_chat(
    args: &ChatCliArgs,
    base: &openjiuwen_harness::cli::Cli,
) -> Result<()> {
    let _ = (args, base);
    Err(openjiuwen_common::error::Error::unsupported(
        "chat --worktree/--watch is not wired yet",
    ))
}
