//! `openjiuwen cron` — the workspace cron schedule, driven from the terminal.
//!
//! The harness owns the schedule parser, the persisted job list and the bounded
//! run history ([`openjiuwen_harness::tools::cron`]); this module only maps a
//! command line onto a [`CronScheduler`] call and renders the result. Jobs live
//! where the agent's `cron` tool keeps them: `<workspace>/cron`.
//!
//! The scheduler takes an injected clock, so every command passes
//! `chrono::Utc::now()` exactly once, at the call site. The `-w` workspace is
//! created first, so a workspace that was never materialised still gets its
//! `cron` directory instead of a store error.
//!
//! # Output
//!
//! `--json` (group-global) prints exactly one pretty JSON document on stdout:
//! the job array of `list`, the `CronStatus` object of `status`, one `CronJob`
//! for `get`/`create`/`toggle`/`cancel`, one `CronRun` for `run`, the
//! occurrences array of `preview`, or the run array of `runs`.
//!
//! Text mode prints one item per line:
//!
//! - `list`/`get`: `<id>\t<cron>\tnext <next-run>\t<enabled|disabled>`, or
//!   `no cron jobs` when the store is empty;
//! - `status`: `N cron job(s): N enabled, N disabled, N due; N run record(s)`
//!   plus `next run at <ts>` when an enabled job is pending;
//! - `create`: `created ` followed by the job line; `toggle`: the state word
//!   (`enabled`/`disabled`) followed by the job line;
//! - `preview`: one occurrence per line, or `no upcoming occurrences`;
//! - `run`: `ran <id>; next run at <ts>`;
//! - `runs`: `<recorded>\t<job-id>\t<manual|scheduled>\toccurrence <ts>`, or
//!   `no cron runs` when the history is empty;
//! - `cancel`: `cancelled <id>`.
//!
//! Timestamps use `to_rfc3339_opts(SecondsFormat::AutoSi, true)`, matching the
//! `cron` tool's message style.
use chrono::{DateTime, SecondsFormat, Utc};
use openjiuwen_common::error::{Error, Result};
use openjiuwen_harness::cli::CliOptions;
use openjiuwen_harness::tools::cron::{CronJob, CronRun, CronSchedule, CronScheduler};
use openjiuwen_harness::Workspace;

/// Largest preview the store produces; `--count` is clamped to this.
const MAX_PREVIEW: usize = 20;

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
    let options = CliOptions::resolve(base)?;
    tokio::fs::create_dir_all(&options.workspace).await?;
    let workspace = Workspace::new(options.workspace);
    let scheduler = CronScheduler::for_workspace(&workspace);
    let now = Utc::now();
    match &args.command {
        CronCommand::List => list(&scheduler, args.json).await,
        CronCommand::Status => status(&scheduler, args.json, now).await,
        CronCommand::Get { job_id } => get(&scheduler, job_id, args.json).await,
        CronCommand::Create {
            cron,
            prompt,
            description,
        } => {
            create(
                &scheduler,
                cron,
                prompt,
                description.as_deref(),
                args.json,
                now,
            )
            .await
        }
        CronCommand::Toggle { job_id } => toggle(&scheduler, job_id, args.json, now).await,
        CronCommand::Preview { job_id, count } => {
            preview(&scheduler, job_id, *count, args.json, now).await
        }
        CronCommand::Run { job_id } => run_job(&scheduler, job_id, args.json, now).await,
        CronCommand::Runs { job_id, limit } => {
            runs(&scheduler, job_id.as_deref(), *limit, args.json).await
        }
        CronCommand::Cancel { job_id } => cancel(&scheduler, job_id, args.json).await,
    }
}

/// Prints the stored jobs.
async fn list(scheduler: &CronScheduler, json: bool) -> Result<()> {
    let jobs = scheduler.list().await?;
    if json {
        return print_json(&jobs);
    }
    if jobs.is_empty() {
        println!("no cron jobs");
    } else {
        for job in &jobs {
            println!("{}", job_line(job));
        }
    }
    Ok(())
}

/// Prints the aggregate status at `now`.
async fn status(scheduler: &CronScheduler, json: bool, now: DateTime<Utc>) -> Result<()> {
    let status = scheduler.status(now).await?;
    if json {
        return print_json(&status);
    }
    println!(
        "{} cron job(s): {} enabled, {} disabled, {} due; {} run record(s)",
        status.total, status.enabled, status.disabled, status.due, status.runs
    );
    if let Some(next_run_at) = status.next_run_at {
        println!("next run at {}", timestamp(next_run_at));
    }
    Ok(())
}

/// Prints one job.
async fn get(scheduler: &CronScheduler, job_id: &str, json: bool) -> Result<()> {
    let job = scheduler
        .get(job_id)
        .await?
        .ok_or_else(|| not_found(job_id))?;
    if json {
        return print_json(&job);
    }
    println!("{}", job_line(&job));
    Ok(())
}

/// Stores a new job and prints it.
async fn create(
    scheduler: &CronScheduler,
    cron: &str,
    prompt: &str,
    description: Option<&str>,
    json: bool,
    now: DateTime<Utc>,
) -> Result<()> {
    let schedule = CronSchedule::parse(cron)?;
    let job = match description {
        Some(description) => {
            scheduler
                .create_with_description(&schedule, prompt, description, now)
                .await?
        }
        None => scheduler.create(&schedule, prompt, now).await?,
    };
    if json {
        return print_json(&job);
    }
    println!("created {}", job_line(&job));
    Ok(())
}

/// Flips the enabled flag of one job and prints the result.
async fn toggle(
    scheduler: &CronScheduler,
    job_id: &str,
    json: bool,
    now: DateTime<Utc>,
) -> Result<()> {
    let job = scheduler
        .toggle(job_id, now)
        .await?
        .ok_or_else(|| not_found(job_id))?;
    if json {
        return print_json(&job);
    }
    let state = if job.enabled { "enabled" } else { "disabled" };
    println!("{state} {}", job_line(&job));
    Ok(())
}

/// Prints the upcoming occurrences of one job.
async fn preview(
    scheduler: &CronScheduler,
    job_id: &str,
    count: usize,
    json: bool,
    now: DateTime<Utc>,
) -> Result<()> {
    let occurrences = scheduler
        .preview(job_id, count.clamp(1, MAX_PREVIEW), now)
        .await?
        .ok_or_else(|| not_found(job_id))?;
    if json {
        let stamps: Vec<String> = occurrences.into_iter().map(timestamp).collect();
        return print_json(&stamps);
    }
    if occurrences.is_empty() {
        println!("no upcoming occurrences");
    } else {
        for occurrence in occurrences {
            println!("{}", timestamp(occurrence));
        }
    }
    Ok(())
}

/// Records a manual run of one job and prints where the job went next.
async fn run_job(
    scheduler: &CronScheduler,
    job_id: &str,
    json: bool,
    now: DateTime<Utc>,
) -> Result<()> {
    let run = scheduler
        .run_now(job_id, now)
        .await?
        .ok_or_else(|| not_found(job_id))?;
    if json {
        return print_json(&run);
    }
    println!(
        "ran {}; next run at {}",
        run.job_id,
        timestamp(run.next_run_at)
    );
    Ok(())
}

/// Prints the run history, newest first.
async fn runs(
    scheduler: &CronScheduler,
    job_id: Option<&str>,
    limit: usize,
    json: bool,
) -> Result<()> {
    let runs = scheduler.runs(job_id, limit).await?;
    if json {
        return print_json(&runs);
    }
    if runs.is_empty() {
        println!("no cron runs");
    } else {
        for run in &runs {
            println!("{}", run_line(run));
        }
    }
    Ok(())
}

/// Deletes one job and prints its identifier.
async fn cancel(scheduler: &CronScheduler, job_id: &str, json: bool) -> Result<()> {
    let job = scheduler
        .cancel(job_id)
        .await?
        .ok_or_else(|| not_found(job_id))?;
    if json {
        return print_json(&job);
    }
    println!("cancelled {}", job.id);
    Ok(())
}

/// Builds the shared "no such job" error of every id-taking command.
fn not_found(job_id: &str) -> Error {
    Error::not_found(format!("cron job not found: {job_id}"))
}

/// Renders one job as `<id>\t<cron>\tnext <ts>\t<enabled|disabled>`.
fn job_line(job: &CronJob) -> String {
    let state = if job.enabled { "enabled" } else { "disabled" };
    format!(
        "{}\t{}\tnext {}\t{state}",
        job.id,
        job.cron,
        timestamp(job.next_run_at)
    )
}

/// Renders one run record as `<recorded>\t<job-id>\t<trigger>\toccurrence <ts>`.
fn run_line(run: &CronRun) -> String {
    format!(
        "{}\t{}\t{}\toccurrence {}",
        timestamp(run.recorded_at),
        run.job_id,
        run.trigger.as_str(),
        timestamp(run.occurrence)
    )
}

/// Formats `at` the way the `cron` tool's messages do.
fn timestamp(at: DateTime<Utc>) -> String {
    at.to_rfc3339_opts(SecondsFormat::AutoSi, true)
}

/// Prints `value` as one pretty JSON document on stdout.
fn print_json(value: &impl serde::Serialize) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}
