//! `auto-harness` command group of the `openjiuwen` binary.
//!
//! Ported from Python's `openjiuwen auto-harness` Click group
//! (`openjiuwen/harness/cli/cli.py`): `run`, `experience search`, `experience
//! list`, `verify-ext`, `gap-analyze` and `history`. The group itself lives in
//! this crate because the harness grammar deliberately has no auto-harness
//! command; the flag names, defaults and output formats follow the Python
//! implementation, while the internal machinery is ordinary Rust.
//!
//! Module layout:
//!
//! - `mod.rs` (this file) — [`AutoHarnessArgs`], [`AutoHarnessCommand`] and the
//!   dispatch into the three implementation modules;
//! - `run` — `auto-harness run`: config loading, overrides, task resolution and
//!   the `--dry-run` task list;
//! - `experience` — the read-only experience store behind `experience search`,
//!   `experience list` and `history`;
//! - `deferred` — `verify-ext` and `gap-analyze`, both honest about what the
//!   Rust port cannot do yet.
//!
//! One intentional addition over Python: `--json`, a global flag of this group
//! that makes every text command print machine-readable JSON instead. Python
//! only ever prints text here, so no script depending on Python output breaks.
use std::path::PathBuf;

use openjiuwen_common::error::Result;
use openjiuwen_harness::cli::{Cli as HarnessCli, CliOptions};

mod deferred;
mod experience;
mod run;

/// Arguments of the `auto-harness` command group.
#[derive(Debug, Clone, clap::Args)]
pub(crate) struct AutoHarnessArgs {
    /// Print machine-readable JSON instead of text.
    ///
    /// Rust addition: Python's group is text-only, and every command here keeps
    /// its text form as the default.
    #[arg(long, global = true)]
    pub(crate) json: bool,
    /// Auto-harness command to run.
    #[command(subcommand)]
    pub(crate) command: AutoHarnessCommand,
}

/// Subcommands of `auto-harness`.
#[derive(Debug, Clone, clap::Subcommand)]
pub(crate) enum AutoHarnessCommand {
    /// Run an optimisation cycle.
    ///
    /// Session execution itself is deferred; `--dry-run` resolves and prints the
    /// task list.
    Run(run::RunCommand),
    /// Inspect the experience store.
    #[command(subcommand)]
    Experience(ExperienceCommand),
    /// Validate an extension package (structure + lint).
    ///
    /// Deferred: the runtime extension loader needs Python's import machinery.
    VerifyExt(deferred::VerifyExtCommand),
    /// Analyse capability gaps against a competing implementation.
    ///
    /// Placeholder in Python and in the Rust port: no model is wired in.
    GapAnalyze(deferred::GapAnalyzeCommand),
    /// Show the optimisation history.
    History(HistoryCommand),
}

/// Subcommands of `auto-harness experience`.
#[derive(Debug, Clone, clap::Subcommand)]
pub(crate) enum ExperienceCommand {
    /// Search the experience store by keyword.
    Search(SearchCommand),
    /// List recent experience records.
    List(ListCommand),
}

/// Options of `auto-harness experience search`.
#[derive(Debug, Clone, clap::Args)]
pub(crate) struct SearchCommand {
    /// Query text; words shorter than two characters are ignored.
    pub(crate) query: String,
}

/// Options of `auto-harness experience list`.
#[derive(Debug, Clone, clap::Args)]
pub(crate) struct ListCommand {
    /// Filter by type (for example `optimization`, `failure`, `insight`).
    #[arg(long = "type")]
    pub(crate) mem_type: Option<String>,
    /// Number of records to return.
    #[arg(long, default_value_t = 10)]
    pub(crate) limit: usize,
}

/// Options of `auto-harness history`.
#[derive(Debug, Clone, clap::Args)]
pub(crate) struct HistoryCommand {
    /// Number of records to return.
    #[arg(long, default_value_t = 20)]
    pub(crate) limit: usize,
}

/// Executes one `auto-harness` invocation.
///
/// `global` is the converted harness command line; it supplies the globals the
/// Python group reads from its `CLIOptions` object (workspace, home, model).
///
/// # Errors
///
/// Returns [`openjiuwen_common::error::Error::Config`] or
/// [`openjiuwen_common::error::Error::Validation`] when the configuration or a
/// task file cannot be used, and [`openjiuwen_common::error::Error::Unsupported`]
/// from the commands whose Python machinery has no Rust port yet.
pub(crate) async fn run(args: &AutoHarnessArgs, global: &HarnessCli) -> Result<()> {
    match &args.command {
        AutoHarnessCommand::Run(command) => run::run(command, global).await,
        AutoHarnessCommand::Experience(ExperienceCommand::Search(search)) => {
            let workspace = workspace_of(global)?;
            experience::search(&workspace, &search.query, args.json).await
        }
        AutoHarnessCommand::Experience(ExperienceCommand::List(list)) => {
            let workspace = workspace_of(global)?;
            experience::list(&workspace, list.mem_type.as_deref(), list.limit, args.json).await
        }
        AutoHarnessCommand::History(history) => {
            let workspace = workspace_of(global)?;
            experience::list(&workspace, None, history.limit, args.json).await
        }
        AutoHarnessCommand::VerifyExt(command) => deferred::verify_ext(command),
        AutoHarnessCommand::GapAnalyze(command) => deferred::gap_analyze(command, args.json),
    }
}

/// Resolves the effective workspace directory for the store commands.
///
/// Python uses `opts.workspace or os.getcwd()`; [`CliOptions::resolve`] already
/// produces exactly that (the `-w` flag, then `OPENJIUWEN_WORKSPACE`, then the
/// current directory), so the resolved path is used unchanged.
fn workspace_of(global: &HarnessCli) -> Result<PathBuf> {
    Ok(CliOptions::resolve(global)?.workspace)
}

/// Directory holding the auto-harness state: `<base>/auto_harness`.
///
/// Python's asymmetry is preserved: `run` uses the resolved `-w` workspace when
/// the flag was given and the home directory (`~/.openjiuwen/auto_harness`)
/// otherwise, even though [`CliOptions::workspace`] always falls back to the
/// current directory.
pub(crate) fn data_dir(global: &HarnessCli, options: &CliOptions) -> PathBuf {
    if global.workspace.is_some() {
        options.workspace.join("auto_harness")
    } else {
        options.home.join("auto_harness")
    }
}
