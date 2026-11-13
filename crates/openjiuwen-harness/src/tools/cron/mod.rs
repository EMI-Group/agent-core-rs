//! Cron tools: expression parsing, next-occurrence computation and a persisted job list.
//!
//! Ports `openjiuwen/harness/tools/cron.py` (the model-facing `cron` tool with
//! create/list/cancel actions) and the scheduling half of the Python cron backend.
//! There is no background daemon thread: the host polls `CronScheduler::due(now)`.
//!
//! # Required public API (registry contract)
//!
//! - `CronSchedule` — a parsed 5-field expression (`m h dom mon dow`, `*`, `*/n`, ranges,
//!   lists) plus the `@hourly`/`@daily`/`@weekly`/`@monthly` shorthands;
//!   `CronSchedule::parse(&str) -> Result<Self>` and `next_after(now: DateTime<Utc>) ->
//!   Option<DateTime<Utc>>`.
//! - `CronScheduler::new(dir: PathBuf) -> Arc<Self>`, `CronScheduler::for_workspace(&Workspace)
//!   -> Arc<Self>`, `create/list/cancel/due`.
//! - `CronTool::new(scheduler: Arc<CronScheduler>, language: impl Into<String>) -> Self` and
//!   `CronTool::card_for(language: &str) -> ToolCard`; model-facing name `cron`.
//! - All "now" values are injected, never read from the wall clock inside the scheduler.
//!
//! # Layout
//!
//! `schedule` owns the expression grammar and the next-occurrence scan, `store` owns the
//! persisted [`CronJob`] list and its lock, `tool` owns the model-facing `cron` tool, and
//! `tests` drives the three of them end to end over a temporary directory.
//!
//! # Deviations from Python
//!
//! Python's cron backend is a 7-field Quartz engine (`second minute hour day month dow year`,
//! with `?` and a one-shot year) driven by host-provided `schedule.at`/`schedule.every` forms,
//! `payload` and `delivery` routing, a `wake_offset_seconds` compatibility field and a run
//! history. This port is deliberately smaller:
//!
//! - the expression is the classic 5-field form, with day-of-month and day-of-week combined
//!   with standard cron OR semantics (see [`CronSchedule::next_after`] and the `schedule` module);
//! - a job is a flat `{id, cron, next_run_at, created_at, enabled, prompt, description}`
//!   record persisted as one JSON array at `<dir>/jobs.json`;
//! - there is no daemon: the host calls [`CronScheduler::due`] with its own clock;
//! - the tool exposes only `create`, `list` and `cancel`. Python's `status`, `update`,
//!   `get_job`, `toggle_job`, `preview_job`, `run`, `runs`, `wake` actions, the `schedule.at`
//!   and `schedule.every` forms, and `payload`/`delivery` routing are deferred (see the `tool` module);
//! - expected failures are `success: false` [`ToolPayload`](crate::tools::support::ToolPayload)
//!   envelopes rather than the exceptions Python raises;
//! - an invalid expression is rejected when the job is created, so a stored job that no longer
//!   parses only appears after a hand-edited file; `due` then fires it once and disables it.

mod schedule;
mod store;
#[cfg(test)]
mod tests;
mod tool;

pub use self::schedule::CronSchedule;
pub use self::store::{CronJob, CronScheduler};
pub use self::tool::{CronAction, CronArgs, CronTool};
