# cron

## Intent

The `cron` tool and its scheduling half: a 5-field cron expression engine, a persisted job list, and one model-facing tool with `create`/`list`/`cancel`.
Ports `openjiuwen/harness/tools/cron.py` and the scheduling part of the Python cron backend, deliberately smaller than Python's 7-field Quartz engine (see `docs/PARITY.md`/module docs).
There is no daemon thread: the host polls `CronScheduler::due(now)` with its own clock, and every `now` is an injected `DateTime<Utc>`.

## API Surface

- `CronSchedule` — `parse(&str) -> Result<Self>`, `next_after(DateTime<Utc>) -> Option<DateTime<Utc>>`, `expression() -> &str`.
  - Grammar: `*`, `*/n`, `a`, `a-b`, `a-b/n`, `a/n` (read as `a-max/n`), comma lists, day-of-week `SUN..SAT` (case-insensitive) and `0..=7` with `7` = Sunday.
  - Shorthands: `@hourly`, `@daily`, `@midnight`, `@weekly`, `@monthly`, `@yearly`, `@annually`.
  - Validation errors name the field and echo its text: `invalid minute field: "75"`; an empty expression is `cron expression cannot be empty`; a wrong field count is `invalid cron expression: expected 5 fields (...), got N: "<text>"`.
  - Day matching uses standard Vixie OR semantics: both day fields restricted → either matches; one unrestricted (`*`, `*/1`, `0-7`) → it alone decides.
  - `next_after` is strictly after `now` at minute granularity and bounded by a five-year horizon plus a 200 000-step cap, so `0 0 31 2 *` yields `None` instead of hanging.
- `CronJob` — `{ id, cron, next_run_at, created_at, enabled, prompt, description }`, serde snake_case, persisted as one JSON array.
- `CronScheduler` — `new(dir: impl Into<PathBuf>) -> Arc<Self>`, `for_workspace(&Workspace) -> Arc<Self>`, `dir() -> &Path`, `path() -> PathBuf`, `list() -> Result<Vec<CronJob>>`, `create(&CronSchedule, prompt, DateTime<Utc>) -> Result<CronJob>`, `create_with_description(&CronSchedule, prompt, description, DateTime<Utc>) -> Result<CronJob>`, `cancel(&str) -> Result<Option<CronJob>>`, `due(DateTime<Utc>) -> Result<Vec<CronJob>>`.
- `CronAction` — `Create`/`List`/`Cancel` with `as_str()` and `parse(&str)`.
- `CronArgs` — `action`, `cron`, `prompt`, `description`, `job_id` (all strings, all serde-defaulted so the tool reports its own validation text).
- `CronTool` — `new(Arc<CronScheduler>, language: impl Into<String>) -> Self`, `unconfigured(language) -> Self`, `card_for(&str) -> ToolCard`, `scheduler() -> Option<&Arc<CronScheduler>>`, `invoke_at(&Value, DateTime<Utc>) -> ToolPayload`.
  - Model-facing name `cron`; card required list is `["action"]` with the three-member `action` enum; per-property hints are English (`en`) or Chinese (anything else).

## Payload Contract

All results are the six-key `ToolPayload` envelope; expected failures are `success=false` with `data: null`, never `Err` (only cancellation is an `Err`).
- `create` → `data = {job: {...}, message: "Created cron job <id> (<cron>) next run at <rfc3339>"}`.
- `list` → `data = {jobs: [...]}` (enabled and disabled jobs, creation order).
- `cancel` → `data = {job_id, message: "Cancelled cron job <id>"}`; an unknown id is `success=false` with `cron job not found: <id>`.
- Validation messages: `action is required (expected create, list or cancel)`, `invalid action: "<x>" (expected create, list or cancel)`, `cron is required for action=create`, `prompt is required for action=create`, `job_id is required for action=cancel`, plus the parser text and `cron expression never matches a future time: "<expr>"`.
- No scheduler wired → `cron is not available: no cron scheduler is configured`.
- `render_for_llm` returns the success `message`, else compact JSON of `data`, else the error text.

## Store Semantics

Jobs live at `<dir>/jobs.json`; `for_workspace` uses the workspace `cron` node when the schema declares one, else `<workspace root>/cron/jobs.json`.
Writes go through a temporary file plus rename (Windows retries once), and one `tokio::sync::Mutex` per scheduler serializes load-modify-save; there is no process-global state.
`create` rejects an expression that never matches in the horizon, so the only way to store an invalid job is a hand-edited file.
`due(now)` returns a snapshot of each job with `next_run_at <= now` (the occurrence that fired) while the persisted job advances to `next_after(now)`; a job whose stored expression no longer parses, or can never match again, is returned once, disabled with a `tracing` warning, and never fires again.
`list` returns jobs in file order and never mutates; disabled jobs are listed but never due.

## Constraints

- Depends only on `crate::workspace`, `crate::tools::support`, `crate::tools::descriptions`, `openjiuwen_foundation::tool` and `openjiuwen_common::error`; no dependency on `crate::agent`, `crate::rails`, `crate::task_loop`, `crate::subagents` or `crate::prompts`.
- Clocks are injected: `Utc::now()` appears only in `Tool::invoke` (one call at the call site, documented); the parser, scheduler and `invoke_at` never read the wall clock.
- No panics on user input, no `unwrap`/`expect` outside tests, no `unsafe`, no `println!`/`eprintln!`; every public item is documented.
- Files stay under 1000 lines; shared helpers live in the owning module.

## Deferred Python Actions

Python's tool also exposes `status`, `update`, `get_job`, `toggle_job`, `preview_job`, `run`, `runs` and `wake`, plus `schedule.at`/`schedule.every` job forms, `payload`/`delivery` routing and `wake_offset_seconds`.
None of that is ported: this module has three actions and the flat job record above.
Also deferred: the 7-field Quartz grammar (seconds/year/`?`), month names, wrap-around ranges (`FRI-MON`), run history, per-job `enabled` toggling through the tool, and any background scheduler loop.

## Routing Table

- `./mod.rs` → module docs (contract, deviations), declarations and flat re-exports
- `./schedule.rs` → `CronSchedule`, field grammar, shorthands, day matching, `next_after`
- `./store.rs` → `CronJob`, `CronScheduler`, atomic persistence, injected-clock `due`
- `./tool.rs` → `CronTool`, `CronArgs`, `CronAction`, card schema and payload handling
- `./tests.rs` → cross-module end-to-end suite (tool + scheduler + parser over a temp dir)

## Test Strategy

- `schedule`: field parsing, lists/ranges/steps, shorthands, day-of-week names and `7`, dedup/restricted flags, exact validation messages, strictly-after boundaries, seconds ignored, hour/day/month/year rollovers, `MON-FRI`, Sunday by name and number, leap day, dom/dow OR semantics, impossible expressions, far-future overflow, case-insensitive names, source-spelling equality.
- `store`: missing file, create/list round-trip, description default, reopen over the same directory, never-matching rejection, due only at/after the occurrence, catch-up, disabled jobs skipped, broken/never-matching jobs disabled after firing once, cancel semantics, corrupt file, workspace path, persisted Python keys and RFC 3339 timestamps.
- `tool`: create/list/cancel round-trip, description-as-prompt fallback, invalid expression, missing cron/prompt/action/job id, unknown action, never-matching schedule, non-object arguments, unconfigured tool, card shape (name, enum, `required`, no `$ref`), language hints, wall-clock `invoke`, `render_for_llm` shapes.
- `tests.rs`: create → list → `due` → advanced `due` → cancel → empty list, persistence across two `CronScheduler` instances, invalid-cron failure payloads, workspace-backed persistence, `Tool::invoke` payload shape, disabled jobs listed but never due.
- Definition of done: `cargo fmt -p openjiuwen-harness`, `cargo test -p openjiuwen-harness --offline`, `cargo test -p openjiuwen-harness --no-default-features --offline`, `cargo clippy -p openjiuwen-harness --all-targets --offline -- -D warnings`.
