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
