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
use std::time::Duration;

use openjiuwen_common::error::Result;
use openjiuwen_harness::cli::Cli;
use openjiuwen_harness::worktree::{find_git_root, ProcessGitRunner, WorktreeManager};

use crate::cli::{ChatCliArgs, RunCliArgs};
use crate::watch::{self, WatchSpec};

/// Debounce window of the `chat --watch` notices.
const CHAT_WATCH_DEBOUNCE: Duration = Duration::from_millis(250);

/// Creates (or recovers) the worktree `name` of the workspace's repository.
///
/// The effective workspace is resolved exactly as the harness resolves it, then
/// made relative-path safe and created; the repository is discovered from it,
/// and the returned command line points `-w` at the worktree path. The created
/// location is announced on stderr so stdout stays the machine-readable result
/// stream.
///
/// # Errors
///
/// Returns a typed `Error::config` when the workspace is outside a git
/// repository or git is not installed, and the worktree manager's errors when
/// git refuses the worktree (occupied path, failed branch, …).
async fn prepare_worktree(name: &str, base: &Cli) -> Result<Cli> {
    let options = openjiuwen_harness::cli::CliOptions::resolve(base)?;
    options.ensure_workspace()?;
    let runner = ProcessGitRunner::new();
    let repo = find_git_root(&runner, &options.workspace).await?;
    let manager = WorktreeManager::new(repo);
    let created = manager.create(name, None, None).await?;
    eprintln!(
        "worktree: {} -> {} (branch {})",
        created.slug,
        created.path.display(),
        created.branch
    );
    let mut cli = base.clone();
    cli.workspace = Some(created.path);
    Ok(cli)
}

/// Runs `openjiuwen run`, honouring `--worktree`.
///
/// # Errors
///
/// Returns the harness run's errors, plus a typed error when the workspace is
/// not inside a git repository or git refuses the worktree.
pub(crate) async fn run_run(args: &RunCliArgs, base: &Cli) -> Result<()> {
    let cli = match &args.worktree {
        Some(name) => prepare_worktree(name, base).await?,
        None => base.clone(),
    };
    openjiuwen_harness::cli::run_cli(cli).await
}

/// Runs `openjiuwen chat`, honouring `--worktree` and `--watch`.
///
/// # Errors
///
/// Returns the harness REPL's errors, plus typed errors from the worktree or the
/// watcher as documented on [`run_run`].
pub(crate) async fn run_chat(args: &ChatCliArgs, base: &Cli) -> Result<()> {
    let cli = match &args.worktree {
        Some(name) => prepare_worktree(name, base).await?,
        None => base.clone(),
    };
    if !args.watch {
        return openjiuwen_harness::cli::run_cli(cli).await;
    }

    let session = watch::start(&cli, WatchSpec::chat_notices(CHAT_WATCH_DEBOUNCE)).await?;
    let run = openjiuwen_harness::cli::run_cli(cli).await;
    let stopped = session.stop().await;
    // The REPL's own failure wins: a stop failure is a secondary diagnostic and
    // must not hide why the conversation ended.
    match run {
        Err(error) => Err(error),
        Ok(()) => stopped,
    }
}
