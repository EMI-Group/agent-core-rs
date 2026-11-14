//! The action surface of the `cron` tool: the action enum, its arguments and its handlers.
//!
//! One tool entry point, ten actions:
//!
//! | action | effect |
//! | --- | --- |
//! | `create` | store a new job from a cron expression plus a prompt |
//! | `list` | every stored job, enabled or not |
//! | `cancel` | remove a job and its run history |
//! | `get_job` | one job, or a not-found failure |
//! | `update` | patch `cron` / `prompt` / `description` / `enabled`, recomputing the next run |
//! | `toggle` | flip the enabled flag, or force it with `enabled` |
//! | `preview` | the next `count` occurrences after the injected clock, without mutating |
//! | `run` | record a manual execution now and advance the schedule |
//! | `runs` | the bounded run history, newest first, for one job or all |
//! | `status` | counts plus the earliest pending occurrence |
//!
//! Every handler returns the six-key [`ToolPayload`] envelope: an expected failure (missing
//! id, rejected expression, storage error) is a `success: false` payload with the error text,
//! never an [`Err`](openjiuwen_common::error::Error).

use chrono::{DateTime, SecondsFormat, Utc};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

use openjiuwen_common::error::Error;

use super::schedule::PREVIEW_LIMIT;
use super::store::{CronJob, CronScheduler, CronStatus, CronUpdate};
use crate::tools::support::ToolPayload;

/// The action names in card order.
pub(super) const ACTION_NAMES: [&str; 10] = [
    "create", "list", "cancel", "get_job", "update", "toggle", "preview", "run", "runs", "status",
];

/// The action list as it is spelled in validation messages.
pub(super) const ACTION_LIST: &str =
    "create, list, cancel, get_job, update, toggle, preview, run, runs or status";

/// Default number of occurrences a `preview` reports.
pub(super) const DEFAULT_PREVIEW_COUNT: u32 = 5;

/// Largest number of occurrences a `preview` reports.
///
/// Kept in step with `schedule::PREVIEW_LIMIT` by a unit test, so the model-facing clamp and
/// the engine agree on one number.
pub(super) const MAX_PREVIEW_COUNT: u32 = 20;

/// Default number of history entries `runs` returns.
pub(super) const DEFAULT_RUN_LIMIT: u32 = 10;

/// Largest number of history entries `runs` returns.
pub(super) const MAX_RUN_LIMIT: u32 = 50;

/// The action of one `cron` call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CronAction {
    /// Create and persist a new job.
    Create,
    /// List the stored jobs.
    List,
    /// Remove a stored job.
    Cancel,
    /// Read one stored job.
    GetJob,
    /// Patch one stored job.
    Update,
    /// Flip or set the enabled flag of one job.
    Toggle,
    /// Report the upcoming occurrences of one job without changing anything.
    Preview,
    /// Record a manual execution of one job now.
    Run,
    /// Read the bounded run history.
    Runs,
    /// Summarize the job list and the run history.
    Status,
}

impl CronAction {
    /// Every action, in card order.
    pub const ALL: [Self; 10] = [
        Self::Create,
        Self::List,
        Self::Cancel,
        Self::GetJob,
        Self::Update,
        Self::Toggle,
        Self::Preview,
        Self::Run,
        Self::Runs,
        Self::Status,
    ];

    /// The action name as the model writes it.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Create => "create",
            Self::List => "list",
            Self::Cancel => "cancel",
            Self::GetJob => "get_job",
            Self::Update => "update",
            Self::Toggle => "toggle",
            Self::Preview => "preview",
            Self::Run => "run",
            Self::Runs => "runs",
            Self::Status => "status",
        }
    }

    /// Parses an action name, accepting surrounding whitespace only.
    #[must_use]
    pub fn parse(action: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|candidate| candidate.as_str() == action)
    }
}

/// Arguments of the `cron` tool.
///
/// One flat object for all ten actions, so the model never has to guess a nested shape:
/// `cron`/`prompt`/`description` belong to `create` and patch `update`, `count` belongs to
/// `preview`, `limit` to `runs`, `enabled` to `toggle`/`update`, and `job_id` names the target
/// of every action except `create`, `list`, `runs` and `status`. Every field is optional at the
/// serde level so the tool can report its own validation text instead of a deserialization
/// error; `action` is the only required field in the card schema.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct CronArgs {
    /// The action to perform, one of [`ACTION_NAMES`](self::ACTION_NAMES).
    #[serde(default)]
    pub action: String,
    /// A cron expression; `create` requires it, `update` replaces the stored one with it.
    #[serde(default)]
    pub cron: String,
    /// The prompt handed to the agent when the job fires; `create` only (and the fallback
    /// prompt is `description`).
    #[serde(default)]
    pub prompt: String,
    /// A human description of the job; `create` only, and used as the prompt when `prompt` is
    /// absent.
    #[serde(default)]
    pub description: String,
    /// The identifier of the target job.
    #[serde(default)]
    pub job_id: String,
    /// How many upcoming occurrences `preview` reports (clamped to 1..=20, default 5).
    #[serde(default)]
    pub count: u32,
    /// How many run-history entries `runs` returns (clamped to 1..=50, default 10).
    #[serde(default)]
    pub limit: u32,
    /// `toggle`: force the enabled flag instead of flipping it; `update`: the new flag.
    #[serde(default)]
    pub enabled: Option<bool>,
}

/// Runs one action against the injected clock value `now`.
pub(super) async fn dispatch(
    scheduler: &CronScheduler,
    action: CronAction,
    args: &CronArgs,
    now: DateTime<Utc>,
) -> ToolPayload {
    match action {
        CronAction::Create => create(scheduler, args, now).await,
        CronAction::List => list(scheduler).await,
        CronAction::Cancel => cancel(scheduler, args).await,
        CronAction::GetJob => get_job(scheduler, args).await,
        CronAction::Update => update(scheduler, args, now).await,
        CronAction::Toggle => toggle(scheduler, args, now).await,
        CronAction::Preview => preview(scheduler, args, now).await,
        CronAction::Run => run(scheduler, args, now).await,
        CronAction::Runs => runs(scheduler, args).await,
        CronAction::Status => status(scheduler, now).await,
    }
}

/// Handles `action=create`.
async fn create(scheduler: &CronScheduler, args: &CronArgs, now: DateTime<Utc>) -> ToolPayload {
    if args.cron.trim().is_empty() {
        return ToolPayload::fail("cron is required for action=create");
    }
    let prompt = if args.prompt.trim().is_empty() {
        args.description.trim()
    } else {
        args.prompt.trim()
    };
    if prompt.is_empty() {
        return ToolPayload::fail("prompt is required for action=create");
    }
    let schedule = match super::schedule::CronSchedule::parse(&args.cron) {
        Ok(schedule) => schedule,
        Err(error) => return failure(&error),
    };
    match scheduler
        .create_with_description(&schedule, prompt, args.description.trim(), now)
        .await
    {
        Ok(job) => {
            let message = format!(
                "Created cron job {} ({}) next run at {}",
                job.id,
                job.cron,
                stamp(job.next_run_at)
            );
            ToolPayload::ok(json!({"job": job, "message": message}))
        }
        Err(error) => failure(&error),
    }
}

/// Handles `action=list`.
async fn list(scheduler: &CronScheduler) -> ToolPayload {
    match scheduler.list().await {
        Ok(jobs) => ToolPayload::ok(json!({"jobs": jobs})),
        Err(error) => failure(&error),
    }
}

/// Handles `action=cancel`.
async fn cancel(scheduler: &CronScheduler, args: &CronArgs) -> ToolPayload {
    let job_id = match job_id(args, CronAction::Cancel) {
        Ok(job_id) => job_id,
        Err(payload) => return payload,
    };
    match scheduler.cancel(job_id).await {
        Ok(Some(job)) => ToolPayload::ok(json!({
            "job_id": job.id,
            "message": format!("Cancelled cron job {}", job.id),
        })),
        Ok(None) => not_found(job_id),
        Err(error) => failure(&error),
    }
}

/// Handles `action=get_job`.
async fn get_job(scheduler: &CronScheduler, args: &CronArgs) -> ToolPayload {
    let job_id = match job_id(args, CronAction::GetJob) {
        Ok(job_id) => job_id,
        Err(payload) => return payload,
    };
    match scheduler.get(job_id).await {
        Ok(Some(job)) => {
            let message = state_message(&job);
            ToolPayload::ok(json!({"job": job, "message": message}))
        }
        Ok(None) => not_found(job_id),
        Err(error) => failure(&error),
    }
}

/// Handles `action=update`.
async fn update(scheduler: &CronScheduler, args: &CronArgs, now: DateTime<Utc>) -> ToolPayload {
    let job_id = match job_id(args, CronAction::Update) {
        Ok(job_id) => job_id,
        Err(payload) => return payload,
    };
    let patch = CronUpdate {
        cron: non_empty(&args.cron),
        prompt: non_empty(&args.prompt),
        description: non_empty(&args.description),
        enabled: args.enabled,
    };
    if patch.is_empty() {
        return ToolPayload::fail(
            "nothing to update for action=update (provide cron, prompt, description or enabled)",
        );
    }
    match scheduler.update(job_id, &patch, now).await {
        Ok(Some(job)) => {
            let message = format!(
                "Updated cron job {} ({}) next run at {}",
                job.id,
                job.cron,
                stamp(job.next_run_at)
            );
            ToolPayload::ok(json!({"job": job, "message": message}))
        }
        Ok(None) => not_found(job_id),
        Err(error) => failure(&error),
    }
}

/// Handles `action=toggle`.
async fn toggle(scheduler: &CronScheduler, args: &CronArgs, now: DateTime<Utc>) -> ToolPayload {
    let job_id = match job_id(args, CronAction::Toggle) {
        Ok(job_id) => job_id,
        Err(payload) => return payload,
    };
    let updated = match args.enabled {
        Some(enabled) => scheduler.set_enabled(job_id, enabled, now).await,
        None => scheduler.toggle(job_id, now).await,
    };
    match updated {
        Ok(Some(job)) => {
            let state = if job.enabled { "Enabled" } else { "Disabled" };
            let message = format!(
                "{state} cron job {} next run at {}",
                job.id,
                stamp(job.next_run_at)
            );
            ToolPayload::ok(json!({"job": job, "message": message}))
        }
        Ok(None) => not_found(job_id),
        Err(error) => failure(&error),
    }
}

/// Handles `action=preview`.
async fn preview(scheduler: &CronScheduler, args: &CronArgs, now: DateTime<Utc>) -> ToolPayload {
    let job_id = match job_id(args, CronAction::Preview) {
        Ok(job_id) => job_id,
        Err(payload) => return payload,
    };
    let count = preview_count(args.count);
    match scheduler.preview(job_id, count, now).await {
        Ok(Some(occurrences)) => {
            let rendered: Vec<String> = occurrences.iter().copied().map(stamp).collect();
            let message = format!(
                "Next {} occurrence(s) of cron job {job_id} after {}: {}",
                rendered.len(),
                stamp(now),
                rendered.join(", ")
            );
            ToolPayload::ok(json!({
                "job_id": job_id,
                "after": stamp(now),
                "count": rendered.len(),
                "occurrences": rendered,
                "message": message,
            }))
        }
        Ok(None) => not_found(job_id),
        Err(error) => failure(&error),
    }
}

/// Handles `action=run`.
async fn run(scheduler: &CronScheduler, args: &CronArgs, now: DateTime<Utc>) -> ToolPayload {
    let job_id = match job_id(args, CronAction::Run) {
        Ok(job_id) => job_id,
        Err(payload) => return payload,
    };
    match scheduler.run_now(job_id, now).await {
        Ok(Some(run)) => {
            let message = format!(
                "Ran cron job {} manually; next run at {}",
                run.job_id,
                stamp(run.next_run_at)
            );
            ToolPayload::ok(json!({"run": run, "message": message}))
        }
        Ok(None) => not_found(job_id),
        Err(error) => failure(&error),
    }
}

/// Handles `action=runs`.
async fn runs(scheduler: &CronScheduler, args: &CronArgs) -> ToolPayload {
    let filter = non_empty(&args.job_id);
    let limit = run_limit(args.limit);
    match scheduler.runs(filter.as_deref(), limit).await {
        Ok(runs) => {
            let count = runs.len();
            ToolPayload::ok(json!({"runs": runs, "count": count}))
        }
        Err(error) => failure(&error),
    }
}

/// Handles `action=status`.
async fn status(scheduler: &CronScheduler, now: DateTime<Utc>) -> ToolPayload {
    match scheduler.status(now).await {
        Ok(status) => {
            let message = status_message(&status);
            ToolPayload::ok(json!({"status": status, "message": message}))
        }
        Err(error) => failure(&error),
    }
}

/// Reads the required `job_id` of an action, or builds the validation failure.
fn job_id<'args>(args: &'args CronArgs, action: CronAction) -> Result<&'args str, ToolPayload> {
    let job_id = args.job_id.trim();
    if job_id.is_empty() {
        return Err(ToolPayload::fail(format!(
            "job_id is required for action={}",
            action.as_str()
        )));
    }
    Ok(job_id)
}

/// Clamps the `preview` count to `1..=20`, defaulting to 5.
fn preview_count(raw: u32) -> usize {
    let count = if raw == 0 {
        DEFAULT_PREVIEW_COUNT
    } else {
        raw.clamp(1, MAX_PREVIEW_COUNT)
    };
    usize::try_from(count).unwrap_or(PREVIEW_LIMIT)
}

/// Clamps the `runs` limit to `1..=50`, defaulting to 10.
fn run_limit(raw: u32) -> usize {
    let limit = if raw == 0 {
        DEFAULT_RUN_LIMIT
    } else {
        raw.clamp(1, MAX_RUN_LIMIT)
    };
    usize::try_from(limit).unwrap_or(DEFAULT_RUN_LIMIT as usize)
}

/// Trims `text`, mapping the empty string to `None` (the "leave unchanged" marker of a patch).
fn non_empty(text: &str) -> Option<String> {
    let trimmed = text.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_owned())
}

/// The one-line state description of a job, used by `get_job`.
fn state_message(job: &CronJob) -> String {
    let state = if job.enabled { "enabled" } else { "disabled" };
    format!(
        "Cron job {} ({}) is {state}, next run at {}",
        job.id,
        job.cron,
        stamp(job.next_run_at)
    )
}

/// Renders an instant the way the payloads do: RFC 3339 in UTC with a `Z` suffix.
fn stamp(instant: DateTime<Utc>) -> String {
    instant.to_rfc3339_opts(SecondsFormat::AutoSi, true)
}

/// The one-line summary of a status report.
fn status_message(status: &CronStatus) -> String {
    format!(
        "{} cron job(s): {} enabled, {} disabled, {} due; {} run record(s)",
        status.total, status.enabled, status.disabled, status.due, status.runs
    )
}

/// Builds the not-found failure shared by every id-carrying action.
fn not_found(job_id: &str) -> ToolPayload {
    ToolPayload::fail(format!("cron job not found: {job_id}"))
}

/// Builds a failure payload from a storage or validation error.
fn failure(error: &Error) -> ToolPayload {
    ToolPayload::fail(error.message().into_owned())
}
