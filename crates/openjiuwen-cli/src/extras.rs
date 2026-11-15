//! `run`/`chat` wiring for the binary's own flags.
//!
//! The harness owns `run` and `chat`; this module adds only the things a
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
//! - `--resume <ID>` / `--continue`: continue a persisted session instead of
//!   starting a fresh one. The id is resolved by the catalogue in
//!   [`crate::sessions`], which knows the CLI session records, the
//!   persistent-agent session records and the execution journals. A one-line
//!   note naming what was resumed is printed on **stderr**, so stdout keeps the
//!   run's output format (`text`, `json`, `stream-json`) and the machine-readable
//!   stream is untouched.
//!
//! Both commands continue the resolved session id. `run --resume` hands the
//! restored turn to [`openjiuwen_harness::cli::run_with_model_resumed`], whose
//! `ResumedTurn` seeds the history (and the persisted `deepagent` state) into a
//! kernel session with that id and continues that id's CLI session record.
//! `chat --resume` adopts the same id with
//! [`Repl::with_session_id`](openjiuwen_harness::cli::Repl::with_session_id)
//! before seeding the transcript, so every following REPL turn appends to the
//! persisted `<home>/sessions/<id>.json` record instead of minting a new
//! `cli-*` session; the note on stderr is the catalogue summary alone.
//!
//! Everything else defaults to today's behaviour: without these flags the calls
//! here are a straight pass-through to
//! [`openjiuwen_harness::cli::run_cli`].
use std::time::Duration;

use openjiuwen_common::error::Result;
use openjiuwen_harness::cli::{
    build_agent, onboarding, paths, run_with_model_resumed, stdout_is_tty, Cli, CliIo, CliOptions,
    ProcessEnv, Repl, ResumedTurn,
};
use openjiuwen_harness::worktree::{find_git_root, ProcessGitRunner, WorktreeManager};

use crate::cli::{ChatCliArgs, RunCliArgs};
use crate::sessions::catalog::{Catalog, Resolved, ResumeRequest};
use crate::watch::{self, WatchSpec};

/// Debounce window of the `chat --watch` notices.
const CHAT_WATCH_DEBOUNCE: Duration = Duration::from_millis(250);

/// Builds the resume request the two flags stand for.
///
/// `None` means "start a fresh session": neither flag was given.
fn resume_request(resume: Option<&str>, continue_: bool) -> Option<ResumeRequest<'_>> {
    match resume {
        Some(id) => Some(ResumeRequest::Id(id)),
        None if continue_ => Some(ResumeRequest::Continue),
        None => None,
    }
}

/// Resolves the resume request against the catalogue of `options`.
///
/// The workspace is materialised first, because the catalogue reads the
/// workspace-scoped persistent-agent layout and the caller is about to run.
async fn resolve_session(options: &CliOptions, request: ResumeRequest<'_>) -> Result<Resolved> {
    options.ensure_workspace()?;
    Catalog::for_options(options, None).resolve(request).await
}

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

/// Runs `openjiuwen run`, honouring `--worktree`, `--resume` and `--continue`.
///
/// # Errors
///
/// Returns the harness run's errors, a typed error when the workspace is not
/// inside a git repository or git refuses the worktree, and the catalogue's
/// errors (an unknown id, no resumable session, an unreadable record).
pub(crate) async fn run_run(args: &RunCliArgs, base: &Cli) -> Result<()> {
    let cli = match &args.worktree {
        Some(name) => prepare_worktree(name, base).await?,
        None => base.clone(),
    };
    let Some(request) = resume_request(args.resume.as_deref(), args.continue_) else {
        return openjiuwen_harness::cli::run_cli(cli).await;
    };
    let options = CliOptions::resolve(&cli)?;
    let resolved = resolve_session(&options, request).await?;
    let Resolved {
        id,
        summary,
        history,
        state,
        ..
    } = resolved;
    eprintln!("{summary}");
    let mut resumed = ResumedTurn::new(id).with_history(history);
    if let Some(state) = state {
        resumed = resumed.with_state(state);
    }
    // The one-shot run stays non-interactive and uncoloured, exactly as
    // `run_cli`'s `Dispatch::Run` branch: permission prompts fail closed. The
    // model is built here so the turn goes through the harness's resumed-turn
    // seam, which resolves the prompt once and seeds the restored history/state
    // before streaming.
    let io = CliIo::stdio_non_interactive();
    let model = options.build_model().await?;
    run_with_model_resumed(&options, &args.base, &io, model, resumed)
        .await
        .map(|_| ())
}

/// Runs `openjiuwen chat`, honouring `--worktree`, `--watch` and `--resume`.
///
/// # Errors
///
/// Returns the harness REPL's errors, plus typed errors from the worktree, the
/// watcher or the session catalogue as documented on [`run_run`].
pub(crate) async fn run_chat(args: &ChatCliArgs, base: &Cli) -> Result<()> {
    let cli = match &args.worktree {
        Some(name) => prepare_worktree(name, base).await?,
        None => base.clone(),
    };
    if !args.watch {
        return chat_entry(args, &cli).await;
    }

    let session = watch::start(&cli, WatchSpec::chat_notices(CHAT_WATCH_DEBOUNCE)).await?;
    let run = chat_entry(args, &cli).await;
    let stopped = session.stop().await;
    // The REPL's own failure wins: a stop failure is a secondary diagnostic and
    // must not hide why the conversation ended.
    match run {
        Err(error) => Err(error),
        Ok(()) => stopped,
    }
}

/// Runs the REPL, seeded from a persisted session when the flags ask for it.
async fn chat_entry(args: &ChatCliArgs, cli: &Cli) -> Result<()> {
    match resume_request(args.resume.as_deref(), args.continue_) {
        Some(request) => run_chat_resumed(request, cli).await,
        None => openjiuwen_harness::cli::run_cli(cli.clone()).await,
    }
}

/// Mirrors `run_cli`'s chat branch, continuing a persisted session.
///
/// The REPL adopts the resolved session id before the transcript is seeded, so
/// the restored history lands in that kernel session and every following turn
/// appends to the same `<home>/sessions/<id>.json` record. Only the catalogue
/// summary is announced on stderr: there is no second, freshly minted id.
async fn run_chat_resumed(request: ResumeRequest<'_>, cli: &Cli) -> Result<()> {
    let options = CliOptions::resolve(cli)?;
    let resolved = resolve_session(&options, request).await?;
    let Resolved {
        id,
        summary,
        history,
        ..
    } = resolved;
    let color = paths::color_enabled(cli.no_color, &ProcessEnv, stdout_is_tty());
    let io = CliIo::stdio().with_color(color);
    onboarding::maybe_run(&io, &options.home, &ProcessEnv).await?;
    let model = options.build_model().await?;
    let agent = build_agent(model, &options, &io).await?;
    // The setter rebuilds the kernel session, so it must precede the seed: an
    // earlier seed would be dropped with the session it was written into.
    let mut repl = Repl::with_io_and_signals(agent, options, io).with_session_id(id);
    repl.seed_history(history).await?;
    eprintln!("{summary}");
    repl.run().await
}
