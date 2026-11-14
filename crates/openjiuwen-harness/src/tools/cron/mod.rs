//! Cron tools: expression parsing, next-occurrence computation, a persisted job list and its
//! bounded run history.
//!
//! Ports `openjiuwen/harness/tools/cron.py` (the model-facing `cron` tool with its single
//! dispatch entry point) and the scheduling half of the Python cron backend.
//! There is no background daemon thread: the host polls `CronScheduler::due(now)`.
//!
//! # Required public API (registry contract)
//!
//! - `CronSchedule` — a parsed expression in either the classic 5-field form (`m h dom mon dow`)
//!   or the 6/7-field Quartz form (`sec m h dom mon dow [year]`), with `*`, `*/n`, ranges,
//!   wrap-around ranges, lists, month/day names, `?`, `L`, `L-n`, `nL`, `d#n` and the
//!   `@hourly`/`@daily`/`@weekly`/`@monthly` shorthands;
//!   `CronSchedule::parse(&str) -> Result<Self>`, `next_after(now: DateTime<Utc>) ->
//!   Option<DateTime<Utc>>` and `next_occurrences(now, count) -> Vec<DateTime<Utc>>`.
//! - `CronScheduler::new(dir: PathBuf) -> Arc<Self>`, `CronScheduler::for_workspace(&Workspace)
//!   -> Arc<Self>`, `create`/`list`/`get`/`update`/`set_enabled`/`toggle`/`preview`/`run_now`/
//!   `runs`/`status`/`cancel`/`due`.
//! - `CronTool::new(scheduler: Arc<CronScheduler>, language: impl Into<String>) -> Self` and
//!   `CronTool::card_for(language: &str) -> ToolCard`; model-facing name `cron`.
//! - All "now" values are injected, never read from the wall clock inside the scheduler.
//!
//! # Layout
//!
//! `schedule` owns both expression grammars and the next-occurrence scan, `store` owns the
//! persisted [`CronJob`] list, its mutations and the scheduler lock, `history` owns the bounded
//! [`CronRun`] log, `actions` owns the ten-action surface and its payload contract, `tool` owns
//! the model-facing `cron` tool and its card, and `tests` drives them all end to end over a
//! temporary directory.
//!
//! # Deviations from Python
//!
//! Python's cron backend is a 7-field Quartz engine driven by host-provided `schedule.at`/
//! `schedule.every` forms, `payload` and `delivery` routing, a `wake_offset_seconds`
//! compatibility field and a run history. This port keeps a smaller, self-contained surface:
//!
//! - the expression is either the classic 5-field form (day-of-month and day-of-week combined
//!   with Vixie OR semantics) or the 6/7-field Quartz form (exactly one day field is `?`);
//!   see the `schedule` module for the implemented extensions and the rejected tokens;
//! - a job is a flat `{id, cron, next_run_at, created_at, enabled, prompt, description}` record
//!   persisted as one JSON array at `<dir>/jobs.json`, and its runs are a flat
//!   `{id, job_id, trigger, recorded_at, occurrence, next_run_at}` array at `<dir>/runs.json`
//!   capped at 20 records per job and 200 in total;
//! - there is no daemon: the host calls [`CronScheduler::due`] with its own clock;
//! - the tool exposes `create`, `list`, `cancel`, `get_job`, `update`, `toggle`, `preview`,
//!   `run`, `runs` and `status`. Python's `wake` action, the `schedule.at` / `schedule.every`
//!   forms, `payload`/`delivery` routing and the legacy per-action tool set are deferred;
//! - expected failures are `success: false` [`ToolPayload`](crate::tools::support::ToolPayload)
//!   envelopes rather than the exceptions Python raises;
//! - an invalid expression is rejected when the job is created or updated, so a stored job that
//!   no longer parses only appears after a hand-edited file; `due` then fires it once and
//!   disables it.

mod actions;
mod history;
mod schedule;
mod store;
#[cfg(test)]
mod tests;
mod tool;

pub use self::actions::{CronAction, CronArgs};
pub use self::history::{CronRun, CronRunTrigger};
pub use self::schedule::CronSchedule;
pub use self::store::{CronJob, CronScheduler, CronStatus, CronUpdate};
pub use self::tool::CronTool;
