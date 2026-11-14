# src/dreaming

## Intent

Dreaming is the deferred memory-consolidation job queue: compaction work (dedupe, summarize, forget, graph extraction) is enqueued as a durable `DreamJob` so it can be claimed, retried, observed, and purged after the requesting call returns.
It is a memory-kernel service only: it depends on `crate::record` (scopes, `CallOptions`), `openjiuwen_common::error`, `chrono`, `serde`, and (behind the `sqlite` feature) `sqlx`.
It never touches `openjiuwen-context`, `openjiuwen-session`, or an LLM, and it never reads a clock of its own — `now` is always a parameter.
The queue is a design of this crate; the Python `openjiuwen.core.memory.dreaming` package only contributes the idle-scheduler primitive and is not transliterated.

## API Surface

`model.rs`:
- `DreamState { Pending, Running, Succeeded, Failed, Skipped, Cancelled }` with `ALL`, `as_str`, `is_terminal`, `parse` (tolerant, case-insensitive), `Display`; serde is strict `snake_case`.
- `DreamKind { Consolidation (default), Dedupe, Summarize, Forget, GraphExtraction, Custom(String) }` with alias-tolerant `parse`, custom-string serde in both directions.
- `DreamTrigger { Manual (default), Schedule, MessageThreshold, Idle }` with `as_str`, `parse`, `Display`.
- `DreamMetrics { records_scanned, records_merged, records_deleted, records_created, tokens_saved, duration_ms }` with `is_empty`, saturating `accumulate`, `with_duration_ms`.
- `DreamJob { id, scope, kind, state, trigger, priority, created_at, updated_at, started_at, finished_at, attempts, max_attempts, last_error, metrics, payload, metadata }` with `Default` (UUID v4, `DEFAULT_MAX_ATTEMPTS = 3`), `new`, `with_*` builders, `is_terminal`, and `validate`.
- `DEFAULT_MAX_ATTEMPTS = 3`.

`outcome.rs`:
- `DreamOutcome { job_id, state, metrics, notes }` with `new`, `succeeded`, `skipped`, `cancelled`, `failed`, `with_metrics`, `with_note`, and `validate` (non-empty id, terminal state).
- `DreamRunSummary { scanned, succeeded, failed, skipped, cancelled, outcomes }` with `new`, `is_empty`, `record` (bumps `scanned` plus the state counter), and `merge`.

`config.rs`:
- `DreamConfig { enabled = true, min_interval_secs = 60, initial_delay_secs = 120, batch_size = 10, max_attempts = 3, retention_secs = 604_800, message_threshold = 100, idle_secs = 300, dedupe_similarity = 0.92 }` with builder setters, `retention()`, and `validate`.

`repository.rs`:
- `DreamFilter { ids, states, kinds, triggers, user_id, scope_id, session_id, agent_id, created_after, created_before, limit }` with `new`, `for_scope`, `with_state`, `with_kind`, `with_trigger`, `with_limit`, and `matches` (predicates only).
- `#[async_trait] DreamRepository` with `enqueue`, `claim_next(now)`, `complete`, `fail`, `get`, `list`, `count_by_state`, `delete_finished`, and a default `purge(now, retention)` that delegates to `delete_finished`.
- `enqueue` validates the job, inserts verbatim, and rejects a duplicate id with `Error::Validation`.
- `claim_next(now)` is the only place attempt counting happens: one atomic `Pending` → `Running` transition ordered `priority DESC, created_at ASC, id ASC`, setting `attempts += 1`, `started_at = now`, `updated_at = now`; eligibility is `Pending && attempts < max_attempts && (started_at.is_none() || started_at < now)`, so a job requeued at clock `now` is not re-claimable by the same run.
- `complete(job_id, outcome)` validates the outcome, requires `Running`, requires `outcome.job_id == job_id`, then stores state + metrics, sets `finished_at = updated_at = now`, and clears `last_error`; an unknown id is `Error::NotFound`, any other state is `Error::Validation`.
- `fail(job_id, error)` requires `Running`; it records `last_error`, preserves `started_at` as the retry marker, and lands in terminal `Failed` (with `finished_at`) only when `attempts >= max_attempts`, otherwise `Pending` with `finished_at = None`.
- `list` applies `DreamFilter::matches`, the shared `(priority DESC, created_at ASC, id ASC)` ordering, then `limit`; `count_by_state` reports all six states including zeros; `delete_finished` removes terminal rows whose finish time (falling back to `updated_at`) is strictly earlier than the cutoff.
- `InMemoryDreamRepository` (`BTreeMap` + `parking_lot::RwLock`, `Debug + Default`, `new`/`len`/`is_empty`) is the default-build backend.

`sqlite.rs` (feature `sqlite`):
- `SqliteDreamRepository { pool }` (`Debug + Clone`, clones share the pool) with `connect(url)`, `from_pool(pool)`, and `pool()`; both constructors apply the schema idempotently.

`planner.rs`:
- `#[async_trait] DreamConsolidator` with `consolidate(&self, job, ctx) -> Result<DreamOutcome>`; the consolidator owns the memory mutation and its own cooperative cancellation, returns a terminal outcome or an error, and never touches the repository.
- `NoopConsolidator` (`Debug + Clone + Copy + Default`) returns `DreamOutcome::skipped(id, "no consolidator configured")`.
- `DreamPlanner { repository: Arc<dyn DreamRepository>, consolidator: Arc<dyn DreamConsolidator>, config: DreamConfig }` with manual `Debug`, `new` (validates the config), `with_in_memory`, `repository`, `config`, `run(now, limit, ctx)`, `run_once(now, limit)` (default `CallOptions`), and `run_due(now, ctx)` (uses `config.batch_size`).
- `DreamingEngine` is a documented alias of `DreamPlanner`.
`lease.rs`:
- Lease data: `DreamLease { job_id, owner, acquired_at, expires_at }` (`is_expired(now)`), `LeaseRequest { owner, ttl }`, `LeasedJob { job, lease }`, `LeaseLostReason { Missing, Expired, OwnerMismatch, NotRunning }`, `LeaseStatus { Renewed(DreamLease), Lost(reason) }`, and `DreamLeaseOptions { owner, ttl, heartbeat }` (`heartbeat < ttl`).
- `DreamLeaseGuard` (`new(repository, lease)`, `lease`, `job_id`, `owner`, `is_lost`, `lost_reason`, `ensure_valid(now)`, `renew(now, ttl)`, `release`) latches a loss so a worker that lost its job cannot resume.
- The `DreamRepository` lease methods `claim_next_leased`, `renew_lease`, `release_lease`, `lease`, and `delete_job` default to typed `Error::Unsupported`; `InMemoryDreamRepository` and `SqliteDreamRepository` override them, and takeover applies only to a `Running` job whose lease expired.

`planner/lease.rs` (child module of `planner`; inherent methods are still `DreamPlanner` API):
- `run_leased(now, limit, ctx, options)` validates `DreamLeaseOptions`, applies the same early exits and cancellation check as `run`, claims through `claim_next_leased`, wraps the lease in a `DreamLeaseGuard`, consolidates through the shared `attempt` steps, renews the lease before recording the transition, and releases the lease on every path.
- A `LeaseStatus::Lost` at renewal records nothing (the job may already belong to the new owner), logs the job id and reason, and stops the sweep; the summary counts recorded outcomes only, so such a job adds nothing.
- `run_due_leased(now, ctx, options)` is the `batch_size` counterpart of `run_due`; `DreamLeaseOptions::heartbeat` is advisory because one reconciliation renews once per job and spawns no background task.

## SQL Schema

Table `dream_jobs` (this crate's table; it does not share a name with `kv_store`, `documents`, or the teams tables):
`id TEXT PRIMARY KEY NOT NULL`, `user_id TEXT NOT NULL`, `scope_id TEXT NOT NULL`, `session_id TEXT`, `agent_id TEXT`, `kind TEXT NOT NULL`, `state TEXT NOT NULL`, `trigger TEXT NOT NULL`, `priority INTEGER NOT NULL DEFAULT 0`, `attempts INTEGER NOT NULL DEFAULT 0`, `max_attempts INTEGER NOT NULL DEFAULT 3`, `created_at TEXT NOT NULL`, `updated_at TEXT NOT NULL`, `started_at TEXT`, `finished_at TEXT`, `last_error TEXT`, `metrics_json TEXT NOT NULL DEFAULT '{}'`, `payload_json TEXT NOT NULL DEFAULT '{}'`, `metadata_json TEXT NOT NULL DEFAULT '{}'`.
Indexes: `dream_jobs_claim_idx (state, priority DESC, created_at ASC, id ASC)` for the claim head and list ordering, and `dream_jobs_scope_idx (user_id, scope_id)` for scope-scoped listings.
Timestamps are stored as RFC 3339 UTC TEXT in one canonical format — `to_rfc3339_opts(SecondsFormat::Micros, true)`, e.g. `2023-11-14T22:13:20.000000Z` — so fixed-width lexicographic ordering equals chronological ordering and retry/retention cutoffs are plain string comparisons.
Reads parse with `DateTime::parse_from_rfc3339(..).with_timezone(&Utc)` and map failures to `Error::External` naming the column; `kind`/`state`/`trigger` decode tolerantly (`DreamKind::parse`, `DreamState::parse` with unknown → `Pending` plus a warning, `DreamTrigger::parse` with unknown → `Manual` plus a warning) and the three JSON columns round-trip through serde.
Every statement is parameter-bound with explicit `?N` placeholders; the only interpolated text is the module-level column-list/schema consts, never user data.
The claim is one atomic `UPDATE ... WHERE id = (SELECT ... LIMIT 1) AND state = 'pending' RETURNING <columns>` statement, so racing workers cannot both win: exactly one `Some` comes back and the loser matches no row.
`list` narrows the four tenant ids and the two time bounds in SQL with `(?N IS NULL OR col = ?N)` predicates, then applies the residual `DreamFilter::matches` predicates and the limit in Rust so both backends rank and truncate identically.
`complete` and `fail` update with `WHERE id = ?1 AND state = 'running'` inside a transaction and re-read the row; a zero-row update rolls back and reports `NotFound` or `Validation` with the observed state.

## Planner Semantics

`run(now, limit, ctx)` is one sweep: it claims at most `limit` jobs and stops earlier when the queue has nothing eligible.
A disabled `DreamConfig::enabled` or `limit == 0` returns an empty `DreamRunSummary` without claiming anything.
`ctx.check()` runs before every claim, so a cancelled sweep never claims a job just to abandon it; a sweep cancelled while a consolidator runs completes that claimed job as `Cancelled` and breaks the loop.
The consolidator call is wrapped in `ctx.run(...)`, which turns a triggered token into `Error::Cancelled` and an elapsed deadline into `Error::Timeout`.
A terminal `Ok(outcome)` is stored through `complete`; the planner forces `outcome.job_id` to the claimed id (warning if the consolidator reported a different or empty one) and routes a non-terminal outcome through `fail` instead of `complete`.
`Error::Cancelled` from either layer completes the job as `Cancelled` and stops the sweep; every other error, `Error::Timeout` included, goes through `fail` and is recorded as a `Failed` outcome whose note says whether it was requeued (`attempt n/max`) or exhausted the budget.
`DreamRunSummary::scanned` counts claimed jobs and `failed` counts failed attempts, including attempts that were rescheduled; because retry eligibility keys on `started_at`, the same `now` never re-claims a requeued job and retries need a later clock.
Repository errors from `claim_next`/`complete`/`fail` abort the sweep with `Err`; transitions that already committed stay committed.

## Constraints

- No `unwrap`/`expect`/`panic` outside `#[cfg(test)]`; `tracing` only (no `println!`/`eprintln!`), targets `openjiuwen_memory::dreaming::{config,model,outcome,repository,planner,sqlite}`.
- Errors are `openjiuwen_common::error::{Error, Result}`: `Validation` for bad input and illegal transitions, `NotFound` for an unknown job id, `External` for storage and decoding failures, cancellation via `Error::Cancelled`.
- Every public item is documented (clippy runs with `-D warnings` and pedantic enabled), every public data type is `serde`-serializable `snake_case` and tolerant of missing/unknown fields.
- The SQL backend is feature-gated behind `sqlite`; the default build compiles and tests fully offline with the in-memory backend only.
- No blocking I/O on async tasks: `sqlx` uses the tokio runtime, `parking_lot` guards are never held across `.await`.
- Files stay around ≤1000 lines and are split by responsibility (`model`/`outcome`, `repository`/`sqlite`, `planner`, `config`).

## Deferred

- The scheduled background orchestrator and its busy-backoff/idle loop (`DreamingOrchestrator`): callers own the timer and call `run_due`.
- The LLM-driven consolidation pipeline: `DreamConsolidator` is the injection point, and only the deterministic dedupe rules in `crate::recall` exist in the crate today.
- Background lease heartbeats and lease brokers: leases are rows and a sweep renews once per job, but nothing renews a long-running consolidation in the background.
- Daemon-side retention enforcement: nothing calls `purge` on a timer, so a caller must invoke it.
- ML-based salience/importance scoring for choosing which memories are worth consolidating.

## Test Strategy

`model`, `outcome`, and `config` tests cover serde compatibility, defaults, builder methods, name parsing/aliases, metrics saturation, and every validation branch.
`repository` tests cover the in-memory backend end to end: claim ordering, retry eligibility via `started_at`, requeue-then-terminal failure, complete/fail error kinds, filtering, counting, deletion, and the default `purge`.
`sqlite` tests (feature `sqlite`) use `tempfile::tempdir()` plus `sqlite:{path}` URLs and fixed timestamps from `DateTime::from_timestamp(1_700_000_000, 0)`: enqueue→claim→complete→fail across a reopen of the same file, claim ordering by priority/creation/id, two concurrent `claim_next` calls via `tokio::join!` yielding exactly one winner plus the same-clock retry-marker check, retention deletion and `purge`, exact JSON column round-trips, all six state counts, duplicate-enqueue/unknown-complete/pending-complete errors, and filtered+limited listing.
`planner` tests use a scripted consolidator (a `Step` queue behind a `parking_lot::Mutex`, with `tokio::sync::Notify` for the blocking case) to cover success with metrics, retry-until-`max_attempts` with an advancing clock, pre-cancelled and mid-consolidation cancellation, a timeout treated as a failed attempt, non-terminal and mismatched outcome ids, `NoopConsolidator` skip semantics, disabled config, zero limit, priority ordering plus limit, `run_due` using `batch_size`, config validation, and `Debug`.
Gates: `cargo fmt -p openjiuwen-memory`, `cargo check -p openjiuwen-memory` with and without `--features sqlite`, `cargo test -p openjiuwen-memory --lib dreaming` with and without `--features sqlite`, `cargo clippy -p openjiuwen-memory --all-targets -D warnings` with and without `--features sqlite`, and `cargo doc -p openjiuwen-memory --no-deps` warning-free.
