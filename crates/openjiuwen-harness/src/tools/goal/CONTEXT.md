# goal
## Intent
The persisted session goal and the two model-facing goal tools, porting `openjiuwen/harness/goal/{schema,store,manager}.py` and the model-facing half of `openjiuwen/harness/tools/goal.py`.
Python's split into session-state store + `GoalManager` + per-round `GoalReportSink` is reduced to a file-backed `GoalStore` plus tools that commit the assessment directly; the loop-side manager role belongs to `../rails/goal.rs`.
## API Surface
- `schema.rs` — `GoalStatus` (`active|paused|completed|blocked`), `GoalAssessmentStatus` (`continue|complete|blocked`), `GoalAssessment` (`to_dict`/`from_dict`/`new`), `TokenUsage` (`accumulate`, `to_dict`, `from_dict`), plus `now_iso(DateTime<Utc>)`, `parse_timestamp(&str)` and `DEFAULT_SESSION_ID`.
- `record.rs` — `GoalRecord`: Python JSON keys, `new`/`with_session_id`/`with_limits`/`touch`/`bump_revision`/`start_timing`/`settle_active_time`/`record_progress`/`set_status`/`is_active`/`progress_summary`/`to_dict`/`from_dict`, plus the persisted-record integer/timestamp validators.
- `store.rs` — `GoalStore::new(path)`, `for_session(path, session_id)`, `for_workspace(&Workspace)` (`<goal node>/goal.json`, falling back to `<root>/goal/goal.json`), `path()`, `session_id()`, `lock()`, `load()`, `save(&record)`, `start(objective, now)`, `update(|record| …, now)`, `clear()`.
- `tools.rs` — `GetCurrentGoalTool` and `SubmitGoalReportTool` with `new(Arc<GoalStore>, language)`, `unavailable(language)`, `card_for(language)`, `store()`, `language()`; `GetCurrentGoalArgs` (`{}`), `SubmitGoalReportArgs { status, evidence, remaining_work, next_instruction }`.
## Constraints
- Persisted JSON keys are Python's (`goal_id`, `session_id`, `objective`, `status`, `revision`, `attempt_count`, `token_usage`, `max_attempts`, `token_budget`, `last_assessment`, `last_stop_reason`, `time_used_seconds`, `active_started_at`, `created_at`, `updated_at`); timestamps are RFC3339 UTC strings.
- No helper reads the wall clock: every `GoalRecord` mutation and every store mutation receives `now`; only the tool boundary calls `Utc::now()`.
- `status`/`evidence` are required on the card but tolerated when missing on the wire (Python defaults to `continue`/`""`); an unknown assessment status normalizes to `continue`.
- Card descriptions come from `crate::tools::descriptions`; nothing in this module inlines description text.
- Expected failures (no store, corrupt record) are `success=false` payloads, never `Err`; a malformed `goal.json` is `Error::Validation` naming the path.
- Every save is a temporary file renamed over `goal.json`; the store lock is never held across a caller's await.
## Design decisions
- Unknown persisted `status` degrades to `active` with a warning instead of clearing the goal (Python raises and the store drops the record).
- `get_current_goal` returns `{goal: <record>|null, message?}` inside the six-key envelope and renders Python's flat `goal_id/objective/status/attempt_count/last_assessment` lines for the model.
- `submit_goal_report` returns `{result: "report_accepted", status}` and adds `goal_id`/`goal_status` when a record was updated, or `message` when none exists (nothing is created).
- `record_progress` never bumps the revision (Python parity); settling statuses close the clock and set `last_stop_reason`.
## Routing Table
- `./mod.rs` → module docs, declarations, flat re-exports
- `./schema.rs` → statuses, assessment, token usage, timestamp/value helpers, serde
- `./record.rs` → `GoalRecord`, its Python validation and mutations
- `./tests.rs` → `#[cfg(test)]` parity suite for the domain model
- `./store.rs` → `GoalStore` persistence, atomic save, lock
- `./tools.rs` → `get_current_goal`, `submit_goal_report`, their cards and rendering
## Test Strategy
- `tests.rs` unit tests: status parsing/fallback, round-trip through `to_dict`/`from_dict`/serde, missing-field tolerance, every `Error::Validation` message, assessment normalization, clock settle/reopen, progress summary, token usage.
- `store.rs` unit tests over temp dirs: missing file, start/load/persist across two instances, empty objective, replace, update stamping, update without a record, clear, malformed JSON and invalid record messages, two concurrent updates, workspace path.
- `tools.rs` unit tests: card names/schemas/`required`, no-store failures, no-goal messages (en/cn), record snapshot and rendering, corrupt record, submit complete/continue/blocked, unknown-status normalization, protocol failure.
