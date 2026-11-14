//! `openjiuwen cron` — the workspace cron schedule, driven from the terminal.
//!
//! The harness owns the schedule parser, the persisted job list and the bounded
//! run history ([`openjiuwen_harness::tools::cron`]); this module only maps a
//! command line onto a [`CronScheduler`] call and renders the result. Jobs live
//! where the agent's `cron` tool keeps them: `<workspace>/cron`.
//!
//! The scheduler takes an injected clock, so every command passes
//! `chrono::Utc::now()` exactly once, at the call site.
use openjiuwen_common::error::Result;

/// `openjiuwen cron <COMMAND>`.
#[derive(Debug, Clone, clap::Args)]
pub(crate) struct CronCli {
    /// Print machine-readable JSON instead of text.
    #[arg(long, global = true)]
    pub(crate) json: bool,
    /// The cron operation to run.
    #[command(subcommand)]
    pub(crate) command: CronCommand,
}

/// The cron operations a terminal user needs.
#[derive(Debug, Clone, clap::Subcommand)]
pub(crate) enum CronCommand {
    /// List the stored jobs.
    List,
    /// Report job counts, the due count and the earliest pending run.
    Status,
    /// Show one job.
    Get {
        /// Job id.
        job_id: String,
    },
    /// Store a new job from a cron expression.
    Create {
        /// Cron expression: classic five fields, or six/seven Quartz fields.
        cron: String,
        /// Prompt to run when the job fires.
        #[arg(long)]
        prompt: String,
        /// Human-readable description.
        #[arg(long)]
        description: Option<String>,
    },
    /// Flip a job between enabled and disabled.
    Toggle {
        /// Job id.
        job_id: String,
    },
    /// Show the next occurrences of a job without mutating it.
    Preview {
        /// Job id.
        job_id: String,
        /// How many occurrences to show (1-20).
        #[arg(long, default_value_t = 5)]
        count: usize,
    },
    /// Record a manual run of a job and advance its next run.
    Run {
        /// Job id.
        job_id: String,
    },
    /// Show the bounded run history, newest first.
    Runs {
        /// Restrict to one job.
        #[arg(long)]
        job_id: Option<String>,
        /// How many records to show (0 = all).
        #[arg(long, default_value_t = 10)]
        limit: usize,
    },
    /// Delete a job and its run records.
    Cancel {
        /// Job id.
        job_id: String,
    },
}

/// Runs `openjiuwen cron`.
///
/// # Errors
///
/// Returns the scheduler's typed errors (unknown job, invalid expression, store
/// failures) and option-resolution errors from the harness.
pub(crate) async fn run(args: &CronCli, base: &openjiuwen_harness::cli::Cli) -> Result<()> {
    let _ = (args, base);
    Err(openjiuwen_common::error::Error::unsupported(
        "cron command is not wired yet",
    ))
}
