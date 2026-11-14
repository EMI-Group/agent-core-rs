# cron

## Intent

The `cron` tool and its scheduling half: a cron expression engine covering the classic 5-field form and the 6/7-field Quartz form, a persisted job list, a bounded persisted run history, and one model-facing tool with ten actions (`create`, `list`, `cancel`, `get_job`, `update`, `toggle`, `preview`, `run`, `runs`, `status`).
Ports `openjiuwen/harness/tools/cron.py` and the scheduling part of the Python cron backend; the host-driven `schedule.at`/`schedule.every` job forms, `payload`/`delivery` routing, `wake` and the legacy per-action tool set stay deferred (see Deferred Python Actions).
There is no daemon thread: the host polls `CronScheduler::due(now)` with its own clock, and every `now` other than `Tool::invoke`'s single `Utc::now()` is injected.

## API Surface

- `CronSchedule` — `parse(&str) -> Result<Self>`, `next_after(DateTime<Utc>) -> Option<DateTime<Utc>>`, `next_occurrences(DateTime<Utc>, count) -> Vec<DateTime<Utc>>`, `expression() -> &str`, `has_seconds() -> bool`.
  - Parsing internals (all private to `schedule.rs`): `FieldTexts::split` names the field slots and rejects a bad field count, `require_single_day_field` enforces the Quartz `?` rule, `parse_values`/`parse_optional_values` return sorted vectors, and `ParsedField::into_parts` returns the owned `FieldParts { values, last_day_offsets, last_weekdays, nth_weekdays }` the schedule stores.
  - Accepted forms: 5-field classic `minute hour day-of-month month day-of-week`; 6-field Quartz `second minute hour day-of-month month day-of-week`; 7-field Quartz with a trailing `year`.
  - Shared value grammar: `*`, `*/n`, `a`, `a-b`, `a-b/n`, `a/n` (read as `a-max/n`), comma lists, month names `JAN..DEC`, day names `SUN..SAT` (case-insensitive), day-of-week `0..=7` with `7` = Sunday, and wrap-around ranges (`22-2` in hours, `FRI-MON` in weekdays).
  - Shorthands: `@hourly`, `@daily`, `@midnight`, `@weekly`, `@monthly`, `@yearly`, `@annually` (expanded into the 5-field form).
  - Quartz day rule: exactly one of day-of-month / day-of-week is `?` (the other decides); two `?` fields or none are typed parse errors.
  - Quartz extensions implemented: `L` and `L-n` (`0..=30`) in day-of-month, `nL` in day-of-week (a bare `L` reads as `6L`), and `d#n` (`n` in `1..=5`) in day-of-week; `W` is rejected with `unsupported cron token: "15W": "W" (nearest weekday) is not supported`.
  - 5-field day matching keeps Vixie OR semantics: both day fields restricted → either matches; one unrestricted (`*`, `*/1`, `0-7`) → it alone decides.
  - Validation errors name the field and echo its text: `invalid minute field: "75"`; an empty expression is `cron expression cannot be empty`; a field count outside 5/6/7 is `invalid cron expression: expected 5 fields (minute hour day-of-month month day-of-week) or 6/7 Quartz fields (second minute hour day-of-month month day-of-week [year]), got N: "<text>"`.
  - A Quartz-only token in the 5-field form is a plain field error (`invalid day-of-week field: "FRI-MON"`), because the classic form keeps its contract; a misplaced `?` is `unsupported cron token: "<field>": "?" (must be the whole field) is not supported`.
  - `next_after` is strictly after `now`, at second granularity when a seconds field is present and at minute granularity otherwise; the scan is bounded by a five-year horizon (extended to the end of a named final year) plus a 200 000-step cap, so `0 0 31 2 *` yields `None` instead of hanging.
  - `next_occurrences(now, count)` collects at most `count` strictly-after instants, capped in-crate by `PREVIEW_LIMIT` (20).
- `CronJob` — `{ id, cron, next_run_at, created_at, enabled, prompt, description }`, serde snake_case, persisted as one JSON array.
- `CronUpdate` — `{ cron, prompt, description, enabled }` patch with `CronUpdate::enabled(bool)` and `is_empty()`; `None` means "leave unchanged".
- `CronStatus` — `{ total, enabled, disabled, due, next_run_at, runs }` reported by `status`.
- `CronRun` / `CronRunTrigger` — `{ id, job_id, trigger, recorded_at, occurrence, next_run_at }` with trigger `manual`/`scheduled`, `CronRunTrigger::as_str()` and `CronRun::new(...)`.
- `CronScheduler` — `new(dir: impl Into<PathBuf>) -> Arc<Self>`, `for_workspace(&Workspace) -> Arc<Self>`, `dir()`, `path()`, `history_path()`, `list()`, `get(&id)`, `create(&CronSchedule, prompt, now)`, `create_with_description(&CronSchedule, prompt, description, now)`, `update(&id, &CronUpdate, now)`, `set_enabled(&id, bool, now)`, `toggle(&id, now)`, `preview(&id, count, now)`, `run_now(&id, now)`, `runs(Option<&str>, limit)`, `status(now)`, `cancel(&id)`, `due(now)`.
- `CronAction` — the ten actions with `ALL: [Self; 10]`, `as_str()` and `parse(&str)`.
- `CronArgs` — `action`, `cron`, `prompt`, `description`, `job_id`, `count` (u32), `limit` (u32), `enabled` (`Option<bool>`); every field serde-defaulted so the tool reports its own validation text.
- `CronTool` — `new(Arc<CronScheduler>, language: impl Into<String>) -> Self`, `unconfigured(language) -> Self`, `card_for(&str) -> ToolCard`, `scheduler() -> Option<&Arc<CronScheduler>>`, `invoke_at(&Value, DateTime<Utc>) -> ToolPayload`.
  - Model-facing name `cron`; `required = ["action"]`; exactly eight properties (`action`, `cron`, `prompt`, `description`, `job_id`, `count`, `limit`, `enabled`) with no `$ref`.
  - The `action` enum lists the ten names in `ACTION_NAMES` order; `count` carries `minimum 1 / maximum 20 / default 5`, `limit` `minimum 1 / maximum 50 / default 10`; `MAX_PREVIEW_COUNT` is kept equal to `schedule::PREVIEW_LIMIT` by a unit test.
  - Per-property hints are English (`en`) or Chinese (anything else); the card description comes from `crate::tools::descriptions` (the shared table text).

## Payload Contract

All results are the six-key `ToolPayload` envelope; expected failures are `success=false` with `data: null`, never `Err` (only cancellation is an `Err`).
Timestamps inside `data` are serde RFC 3339 with a `Z` suffix, and rendered message timestamps use `to_rfc3339_opts(SecondsFormat::AutoSi, true)` so both agree.
- `create` → `{job, message: "Created cron job <id> (<cron>) next run at <ts>"}`.
- `list` → `{jobs: [...]}` (enabled and disabled jobs, creation order).
- `cancel` → `{job_id, message: "Cancelled cron job <id>"}`; an unknown id is `cron job not found: <id>`.
- `get_job` → `{job, message: "Cron job <id> (<cron>) is enabled|disabled, next run at <ts>"}`.
- `update` → `{job, message: "Updated cron job <id> (<cron>) next run at <ts>"}`; a patch with nothing set is `nothing to update for action=update (provide cron, prompt, description or enabled)`.
- `toggle` → `{job, message: "Enabled|Disabled cron job <id> next run at <ts>"}`.
- `preview` → `{job_id, after, count, occurrences: [<ts>...], message: "Next N occurrence(s) of cron job <id> after <ts>: ..."}`; never mutates the stored job.
- `run` → `{run, message: "Ran cron job <id> manually; next run at <ts>"}`.
- `runs` → `{runs: [...], count}` newest first, filtered by `job_id` when supplied.
- `status` → `{status: CronStatus, message: "N cron job(s): N enabled, N disabled, N due; N run record(s)"}`.
- Validation messages: `action is required (expected create, list, cancel, get_job, update, toggle, preview, run, runs or status)`, `invalid action: "<x>" (expected …)`, `job_id is required for action=<action>`, `cron is required for action=create`, `prompt is required for action=create`, plus the parser text and `cron expression never matches a future time: "<expr>"`.
- No scheduler wired → `cron is not available: no cron scheduler is configured`.
- `render_for_llm` returns the success `message`, else compact JSON of `data`, else the error text.

## Store Semantics

Jobs live at `<dir>/jobs.json` and the run history at `<dir>/runs.json`; `for_workspace` uses the workspace `cron` node when the schema declares one, else `<workspace root>/cron/`.
Writes go through a temporary file plus rename (Windows retries once), and one `tokio::sync::Mutex` per scheduler serializes every load-modify-save sequence, history writes included; there is no process-global state.
`create` and `update` reject an expression that does not parse or never matches in the horizon, so the only way to store an invalid job is a hand-edited file; a rejected patch leaves the stored job untouched.
`update` recomputes `next_run_at` from `now` whenever the expression or the enabled flag changes; `set_enabled`/`toggle` recompute it when turning a job **on** and keep the stored value when turning it **off**, so re-enabling always lands on a fresh future occurrence.
`preview` reads the stored expression and returns up to `count` occurrences strictly after `now` without writing anything, and it previews disabled jobs too.
`run_now` records a `manual` run at `now`, advances `next_run_at` to the first occurrence strictly after `now`, and leaves the enabled flag alone (a manual run of a disabled job is allowed); the schedule is persisted before the run record, so a history failure leaves an advanced job without a record.
`due(now)` returns a snapshot of each enabled job with `next_run_at <= now` (the occurrence that fired), advances the persisted job to `next_after(now)` and appends a `scheduled` run record for each fired job; a job whose stored expression no longer parses, or can never match again, is returned once, disabled with a `tracing` warning, and never fires again.
`cancel` removes the job and trims its run records so `runs` never reports entries of a job that no longer exists.
`runs(job_id, limit)` loads the history file, filters by job when asked, reverses to newest first, then truncates to `limit` (`0` = no limit, the tool passes 10 by default and clamps to 1..=50).
The history file is capped at `MAX_RUNS_PER_JOB` (20) records per job and `MAX_RUNS_TOTAL` (200) records in total, rotating oldest-first, so a long-running schedule cannot grow the file without bound.
`list` returns jobs in file order and never mutates; disabled jobs are listed, never due, and still reported by `status` in the `disabled` count.

## Constraints

- Depends only on `crate::workspace`, `crate::tools::support`, `crate::tools::descriptions`, `openjiuwen_foundation::tool` and `openjiuwen_common::error`; no dependency on `crate::agent`, `crate::rails`, `crate::task_loop`, `crate::subagents` or `crate::prompts`.
- Clocks are injected: `Utc::now()` appears only in `Tool::invoke` (one call at the call site, documented); the parser, scheduler, history and `invoke_at` never read the wall clock.
- No panics on user input, no `unwrap`/`expect` outside tests, no `unsafe`, no `println!`/`eprintln!`; every public item is documented.
- Files stay under 1000 lines; shared helpers live in the owning module (`stamp` in `actions.rs`, `write_atomic` in `store.rs`).
- Test modules live in separate `*_tests.rs` files wired with `#[path = "..."] mod tests;` so no implementation file carries the suite.

## Deferred Python Actions

Python's tool also exposes `add`, `remove`, `run`, `runs`, `wake` and the legacy per-action tool set, plus `schedule.at`/`schedule.every` job forms, `payload`/`delivery` routing and `wake_offset_seconds`.
Still deferred here: the `wake` action and `wake_offset_seconds`, the `schedule.at`/`schedule.every` job forms, `payload`/`delivery` routing, job tags/priority, and the legacy per-action tool set (one dispatching `cron` tool instead).
Also deferred: the 7-field Quartz token `W` (nearest weekday, including `LW`), `?` inside a list, Quartz cron-expression `year` values outside `1970..=2099`, run output/exit-code capture (a run record is a receipt only), and any background scheduler loop or daemon.

## Routing Table

- `./mod.rs` → module docs (both grammars, layout, deviations), declarations and flat re-exports
- `./schedule.rs` → `CronSchedule`, field grammar (classic + Quartz), shorthands, day rules, extensions, `FieldTexts`/`ParsedField`/`FieldParts` field split, bounded `next_after`/`next_occurrences`
- `./store.rs` → `CronJob`, `CronUpdate`, `CronStatus`, `CronScheduler`, atomic persistence, injected-clock `update`/`toggle`/`preview`/`run_now`/`status`/`due`
- `./history.rs` → `CronRun`, `CronRunTrigger`, the bounded `runs.json` history and its rotation caps
- `./actions.rs` → `CronAction`, `CronArgs`, the ten action handlers, message/`stamp`/validation helpers
- `./tool.rs` → `CronTool`, card schema (action enum, `count`/`limit`/`enabled`, en/cn hints), `Tool` impl and `render_for_llm`
- `./schedule_tests.rs`, `./store_tests.rs`, `./history` inline tests, `./tool_tests.rs` → per-unit suites
- `./tests.rs` → cross-module end-to-end suite (tool + scheduler + history + parser over a temp dir)

## Test Strategy

- `schedule`: classic field parsing, lists/ranges/steps, shorthands, day names and `7`, dedup/restricted flags, exact validation messages, strictly-after boundaries, hour/day/month/year rollovers, `MON-FRI`, leap day, dom/dow OR semantics, impossible expressions, far-future overflow, case-insensitive names, source-spelling equality; Quartz 6/7-field parsing, second granularity and minute-boundary rollover, `?` selection and both-`?`/no-`?` errors, `?` inside a list, `L`/`L-n`/`nL`/`L` alone, `d#n` bounds, wrap-around ranges (`22-2`, `FRI-MON`, `SAT-SUN`), month/day names, year bounds and horizon truncation, `W` rejection, and Quartz-only tokens rejected in the classic form.
- `store`: missing file, create/list/get round-trip, description default, reopen over the same directory, never-matching rejection, due only at/after the occurrence, catch-up, disabled jobs skipped, broken/never-matching jobs disabled after firing once, cancel semantics (job and history), corrupt files, workspace path, persisted Python keys, Quartz jobs round-tripping, `update` patch semantics and rejection, `set_enabled`/`toggle` recompute-vs-hold, `preview` non-mutation, `run_now` advance, `runs` filtering/limit, `status` counts and earliest pending occurrence.
- `history`: record/load round-trip, newest-first ordering, per-job cap (20) and total cap (200) rotation, missing file, corrupt file, trigger round-trip.
- `tool`: create/list/cancel/get_job/update/toggle/preview/run/runs/status happy paths and every failure path, description-as-prompt fallback, invalid expression, never-matching schedule, missing cron/prompt/action/job id, unknown action, non-object arguments, unconfigured tool, card shape (name, ten-member enum, `required`, eight properties, no `$ref`, integer bounds), bilingual hints, wall-clock `invoke`, `render_for_llm` shapes, and the description-table identity check.
- `tests.rs`: create → list → `due` → advanced `due` → cancel → empty list, the `get_job`/`update`/`toggle`/`preview`/`run`/`status` flow, a Quartz job firing at second granularity, persistence across two `CronScheduler` instances, invalid-cron failure payloads, workspace-backed persistence, `Tool::invoke` payload shape, disabled jobs listed but never due.
- Definition of done: `cargo fmt -p openjiuwen-harness`, `cargo test -p openjiuwen-harness --offline`, `cargo test -p openjiuwen-harness --no-default-features --offline`, `cargo clippy -p openjiuwen-harness --all-targets --offline -- -D warnings`, `cargo doc -p openjiuwen-harness --no-deps`.
