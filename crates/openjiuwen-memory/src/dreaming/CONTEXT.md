# src/dreaming

## Intent

Dreaming is the durable memory-consolidation queue plus the workers that drain it: compaction work (dedupe, summarize, forget, graph extraction) is enqueued as a `DreamJob`, claimed through a lease, handed to a `DreamConsolidator`, recorded, and purged after the retention window.
It is a memory-kernel service: it depends on `crate::record` (scopes, `CallOptions`), `crate::recall` (jaccard, recency), `openjiuwen_common::error`, `chrono`, `serde`, `tokio`/`tokio-util`, and (behind `sqlite`) `sqlx`.
Only `pipeline` calls a model, and it receives `openjiuwen_foundation::llm::Model` by injection; nothing here touches `openjiuwen-context` or `openjiuwen-session`.
No operation reads a clock of its own: every queue, planner, retention, salience, and orchestrator entry point takes `now` (the sole exception is `DreamOrchestrator::run_until_cancelled`, the caller-owned loop, which reads `Utc::now()` between sleeps).
The module spawns no task, starts no process, and keeps no global state; the queue and its design are this crate's, not a transliteration of the Python `openjiuwen.core.memory.dreaming` idle scheduler.

## API Surface

`model.rs`:
- `DreamState { Pending, Running, Succeeded, Failed, Skipped, Cancelled }` with `ALL`, `as_str`, `is_terminal`, `parse` (tolerant, case-insensitive), `Display`; serde is strict `snake_case`.
- `DreamKind { Consolidation (default), Dedupe, Summarize, Forget, GraphExtraction, Custom(String) }` with alias-tolerant `parse`, custom-string serde in both directions.
- `DreamTrigger { Manual (default), Schedule, MessageThreshold, Idle }` with `as_str`, `parse`, `Display`.
- `DreamMetrics { records_scanned, records_merged, records_deleted, records_created, tokens_saved, duration_ms }` with `is_empty`, saturating `accumulate`, `with_duration_ms`.
- `DreamJob { id, scope, kind, state, trigger, priority, created_at, updated_at, started_at, finished_at, attempts, max_attempts, last_error, metrics, payload, metadata }` with `Default` (UUID v4, `DEFAULT_MAX_ATTEMPTS = 3`), `new`, `with_*` builders, `is_terminal`, and `validate`.

`outcome.rs`:
- `DreamOutcome { job_id, state, metrics, notes }` with `new`, `succeeded`, `skipped`, `cancelled`, `failed`, `with_metrics`, `with_note`, `validate` (non-empty id, terminal state).
- `DreamRunSummary { scanned, succeeded, failed, skipped, cancelled, outcomes }` with `new`, `is_empty`, `record`, `merge`.

`config.rs`:
- `DreamConfig { enabled = true, min_interval_secs = 60, initial_delay_secs = 120, batch_size = 10, max_attempts = 3, retention_secs = 604_800, message_threshold = 100, idle_secs = 300, dedupe_similarity = 0.92 }` with builder setters, `retention()`, `validate`.

`repository.rs`:
- `DreamFilter { ids, states, kinds, triggers, user_id, scope_id, session_id, agent_id, created_after, created_before, limit }` with `new`, `for_scope`, `with_state`, `with_kind`, `with_trigger`, `with_limit`, `matches`.
- `#[async_trait] DreamRepository` with `enqueue`, `claim_next(now)`, `complete`, `fail`, `get`, `list`, `count_by_state`, `delete_finished`, `purge(now, retention)`, and the lease pair `claim_next_leased(now, &LeaseRequest)`, `renew_lease(job_id, owner, now, ttl)`, `release_lease(job_id, owner)`, `lease(job_id)`, plus `delete_job(job_id)`.
- The five lease methods and `delete_job` default to typed `Error::Unsupported`, so a third-party repository compiles without leasing and says so instead of degrading.
- `claim_next(now)` is the only place attempt counting happens: one atomic `Pending` → `Running` transition ordered `priority DESC, created_at ASC, id ASC`, setting `attempts += 1`, `started_at = now`, `updated_at = now`; eligibility is `Pending && attempts < max_attempts && (started_at.is_none() || started_at < now)`, so a job requeued at clock `now` is not re-claimable by the same run.
- `complete` requires `Running` and `outcome.job_id == job_id`, stores state + metrics, sets `finished_at = updated_at = now`, clears `last_error`; unknown id is `NotFound`, any other state `Validation`.
- `fail` requires `Running`, records `last_error`, keeps `started_at` as the retry marker, and lands in terminal `Failed` only when `attempts >= max_attempts`, otherwise `Pending`.
- `list` applies `DreamFilter::matches`, the shared `(priority DESC, created_at ASC, id ASC)` ordering, then `limit`; `count_by_state` reports all six states; `delete_finished(cutoff)` removes terminal rows whose finish time (falling back to `updated_at`) is strictly before the cutoff.
- `InMemoryDreamRepository` (`BTreeMap` + `parking_lot::RwLock`, plus a `BTreeMap` of leases) is the default-build backend; it claims the pending head first and, failing that, takes over the earliest-queued `Running` job whose lease is expired.

`sqlite.rs` and `sqlite_lease.rs` (feature `sqlite`):
- `SqliteDreamRepository { pool }` (`Debug + Clone`, clones share the pool) with `connect(url)`, `from_pool(pool)`, `pool()`; constructors apply the job and lease schema idempotently.
- The lease queries live in `sqlite_lease.rs`: takeover is one `UPDATE ... WHERE id = (SELECT ... JOIN lease ... WHERE l.expires_at <= ?1 LIMIT 1) AND state = 'running' RETURNING ...`, so expiry is inclusive at the boundary and exactly one racer wins.
- `delete_finished` deletes expired jobs and their leases in one transaction; `delete_job` deletes a job and its lease.

`lease.rs`:
- `DreamLease { job_id, owner, acquired_at, expires_at }` with `new(job_id, owner, now, ttl)`, `is_expired(now)` (`now >= expires_at`), `validate`.
- `LeaseRequest { owner, ttl }`, `LeasedJob { job, lease }`, `LeaseLostReason { Missing, Expired, OwnerMismatch, NotRunning }` (`as_str`, `Display`).
- `LeaseStatus { Renewed(DreamLease), Lost(reason) }` with `is_renewed`, `is_lost`, `lease`, `reason`.
- `DreamLeaseOptions { owner, ttl, heartbeat }` (`new`, `request`, `validate` requiring `0 < heartbeat < ttl`); JSON carries `ttl`/`heartbeat` as whole milliseconds and defaults to a 60 s TTL / 15 s heartbeat with an empty owner (which `validate` rejects).
- `DreamLeaseGuard` (`new(repository, lease)`, `lease`, `job_id`, `owner`, `is_lost`, `lost_reason`, `ensure_valid(now)`, `async renew(now, ttl)`, `async release`) latches a loss in an `AtomicBool` + `Mutex<Option<LeaseLostReason>>`, never holds a lock across `.await`, and keeps reporting the first loss reason.

`planner.rs`:
- `#[async_trait] DreamConsolidator` with `consolidate(&self, job, ctx) -> Result<DreamOutcome>`; the consolidator owns the memory mutation and its cancellation and never touches the repository.
- `NoopConsolidator` returns `Skipped("no consolidator configured")`.
- `DreamPlanner { repository, consolidator, config }` with `new`, `with_in_memory`, `repository`, `config`, `run(now, limit, ctx)`, `run_once`, `run_due` (uses `config.batch_size`).
- `DreamingEngine` is a documented alias of `DreamPlanner`.

`planner/lease.rs` (inherent methods, still `DreamPlanner` API):
- `run_leased(now, limit, ctx, &DreamLeaseOptions)` and `run_due_leased(now, ctx, &DreamLeaseOptions)`: validate the options, apply the same early exits and `ctx.check()`, claim through `claim_next_leased`, wrap the lease in a `DreamLeaseGuard`, consolidate through the shared attempt steps, renew the lease **before** recording, release on every path.
- A `LeaseStatus::Lost` at renewal records nothing (the job may belong to the new owner), warns with the job id and reason, and stops the sweep with the summary collected so far.
- `heartbeat` is advisory: one renewal happens per job and nothing spawns a background heartbeat, so a consolidator that can exceed its TTL must renew through its own guard.

`orchestrator.rs`:
- `DreamOrchestratorConfig { enabled = true, poll_interval_ms = 60_000, initial_delay_ms = 120_000, max_interval_ms = 600_000, backoff_multiplier = 2.0, max_backoff_steps = 4, run_retention_on_tick = true, lease: Option<DreamLeaseOptions> = None, retention: Option<DreamRetention> = None }` with `with_*` setters, `poll_interval()`, `initial_delay()`, `max_interval()`, `interval_for(empty_sweeps)`, `validate`.
- `DreamOrchestratorStatus { ticks, consecutive_empty_sweeps, next_interval_ms, last_tick_at, last_summary, last_retention, last_error }` (`Debug + Clone + Default + PartialEq`, serde).
- `DreamOrchestrator { planner, repository, config, state }` with `new(planner, repository, config)` (validates), `config`, `planner`, `repository`, `status` (clone), `interval_for_empty_sweeps`, `tick(now)`, `tick_with(now, ctx)`, `run_until_cancelled(cancel) -> Result<DreamRunSummary>`.
- The orchestrator uses its repository handle for retention and as the lease store only; callers pass the planner's repository so both halves see one queue.

`retention.rs`:
- `DreamRetention { retention_secs = 604_800, max_per_state: Option<usize> = None }` with `new`, `disabled`, `from_config`, `with_retention_secs`, `with_max_per_state`, `retention()`, `is_enabled`, `validate` (rejects `Some(0)`).
- `RetentionReport { deleted_expired, deleted_over_cap }` with `total`, `is_empty`, `merge`.
- `async purge_expired(&self, repository: &dyn DreamRepository, now) -> Result<RetentionReport>`: age purge first (`delete_finished(now - retention_secs)`), then the per-state cap over `Succeeded`/`Failed`/`Skipped`/`Cancelled` keeping the newest and deleting the rest through `delete_job`; idempotent, so a second call at the same `now` deletes nothing.

`salience.rs`:
- `SalienceBackend { Heuristic (default), Ml }`, `SalienceWeights` (four weights, default `1.0`), `KindWeights` (`user_profile 1.0`, `semantic_memory 0.9`, `summary 0.8`, `coding 0.85`, `episodic_memory 0.7`, `variable 0.6`, `unknown`), `SalienceConfig { backend, half_life_secs = 2_592_000, access_saturation = 10.0, weights, kind_weights }`.
- `SalienceScore { total, importance, recency, access, kind }`, `salience_score(&MemoryRecord, now, &SalienceConfig) -> Result<SalienceScore>`, `rank_by_salience(&[MemoryRecord], now, &SalienceConfig) -> Result<Vec<(String, SalienceScore)>>` (descending `total`).
- Formula: `importance` clamped to `0.0..=1.0`; `recency = 0.5 ^ (max(0, now - timestamp) / half_life_secs)`; `access = count / (count + access_saturation)`; `kind = kind_weights.weight(kind)`; `total = Σ wᵢ·termᵢ / Σ wᵢ` (all weights zero → `0.0`, never `NaN`).
- `SalienceBackend::Ml` returns typed `Error::Unsupported` naming ML salience as deferred; it never silently falls back.

`pipeline.rs` + `pipeline/plan.rs` + `pipeline/tests.rs`:
- Model-facing half: `DreamCandidate` (JSON `text`/`type` aliases), `DreamCandidateSource` (`StaticCandidateSource`, `PayloadCandidateSource` reading the job payload key `candidates`), `ConsolidationSink`/`RecordOnlySink`, `LlmConsolidatorConfig { enabled = true, model_retries = 1, max_candidates = 50, max_content_chars = 500, fallback_similarity = 0.75 }`, `LlmConsolidator` (`new(model: Option<Arc<dyn Model>>, source, config)`, `payload_driven`, `with_sink`, `config`, `plan`, `impl DreamConsolidator`).
- Pure decision half: `DreamConsolidationAction { Merge | Keep | Summarize | Forget }` (internally tagged `action`), `DreamConsolidationOrigin { Rules, Llm }`, `DreamConsolidationPlan { job_id, origin, actions, degraded, metrics }` (`is_degraded`, `is_empty`, `notes`), `parse_consolidation_response`, `deterministic_plan`.
- `prompts.rs`: `CONSOLIDATION_SYSTEM_PROMPT`, `CONSOLIDATION_RESPONSE_SCHEMA`, `CONSOLIDATION_ACTION_NAMES`. There is no Python consolidation prompt; the request shape and schema are this crate's design (documented in the module header): a system message with the rules + schema, and a JSON user message `{job_id, kind, scope, max_actions, candidates:[{id, text, kind, importance, timestamp, access_count}]}`.
- The model is attempted `model_retries + 1` times; provider errors, empty answers, unparseable JSON, invented-only ids, and non-terminal answers all count as failed attempts. Exhaustion yields `deterministic_plan` (token Jaccard ≥ `fallback_similarity` or equal normalized text → `Merge`, everything else `Keep`) with `degraded = Some(first failure)`. `{"actions":[]}` is a valid empty plan (`Skipped` outcome). Cancellation is propagated, never degraded.
- Plan metrics: `records_scanned` = candidates, `records_merged` = merge sources, `records_created` = summaries, `records_deleted` = forgotten ids, `tokens_saved` = reclaimed characters / 4.

## SQL Schema

Table `dream_jobs` (this crate's table; it does not share a name with `kv_store`, `documents`, or the teams tables):
`id TEXT PRIMARY KEY NOT NULL`, `user_id TEXT NOT NULL`, `scope_id TEXT NOT NULL`, `session_id TEXT`, `agent_id TEXT`, `kind TEXT NOT NULL`, `state TEXT NOT NULL`, `trigger TEXT NOT NULL`, `priority INTEGER NOT NULL DEFAULT 0`, `attempts INTEGER NOT NULL DEFAULT 0`, `max_attempts INTEGER NOT NULL DEFAULT 3`, `created_at TEXT NOT NULL`, `updated_at TEXT NOT NULL`, `started_at TEXT`, `finished_at TEXT`, `last_error TEXT`, `metrics_json TEXT NOT NULL DEFAULT '{}'`, `payload_json TEXT NOT NULL DEFAULT '{}'`, `metadata_json TEXT NOT NULL DEFAULT '{}'`.
Indexes: `dream_jobs_claim_idx (state, priority DESC, created_at ASC, id ASC)` and `dream_jobs_scope_idx (user_id, scope_id)`.

Table `dream_job_leases` (one row per leased job; `job_id` is the job id and the row disappears on release or job deletion):
`job_id TEXT PRIMARY KEY NOT NULL`, `owner TEXT NOT NULL`, `acquired_at TEXT NOT NULL`, `expires_at TEXT NOT NULL`, index `dream_job_leases_expiry_idx (expires_at)`.
An unleased `Running` job is never stolen: takeover joins the lease table, so only a lease that exists and expired can be reclaimed.

Timestamps are RFC 3339 UTC TEXT in one canonical format — `to_rfc3339_opts(SecondsFormat::Micros, true)` — so lexicographic ordering is chronological and cutoffs are plain string comparisons.
Reads parse with `DateTime::parse_from_rfc3339(..).with_timezone(&Utc)` and map failures to `Error::External` naming the column; `kind`/`state`/`trigger` decode tolerantly; JSON columns round-trip through serde.
Every statement is parameter-bound with explicit `?N` placeholders; only module-level column/schema consts are interpolated.
The unleased claim is one `UPDATE ... RETURNING` statement; completion and failure update `WHERE id = ?N AND state = 'running'` inside a transaction and re-read the row, so a zero-row update reports `NotFound` or the observed state.
`list` narrows the four tenant ids and the two time bounds in SQL, then applies residual `DreamFilter` predicates and the limit in Rust so both backends rank and truncate identically.

## Worker Semantics

`run(now, limit, ctx)` is one sweep: claim at most `limit` jobs, stop when the queue has nothing eligible, `ctx.check()` before every claim.
A disabled config or `limit == 0` returns an empty summary without claiming.
The consolidator call is wrapped in `ctx.run(...)`: a triggered token becomes `Error::Cancelled` (the job is completed as `Cancelled` and the sweep stops) and an elapsed deadline becomes `Error::Timeout` (a failed attempt, requeued or terminal by the attempt budget).
A terminal `Ok(outcome)` is stored through `complete` with the job id forced to the claimed one; a non-terminal outcome goes through `fail`.
Other errors go through `fail` and produce a `Failed` outcome whose note says whether the attempt was requeued (`attempt n/max`) or exhausted the budget.
`DreamRunSummary::scanned` counts claimed jobs and `failed` counts failed attempts, including rescheduled ones; retry eligibility keys on `started_at`, so the same `now` never re-claims a requeued job.
Repository errors from claim/complete/fail abort the sweep with `Err`; transitions already committed stay committed.
`run_leased` adds the lease arm: claim and lease atomically, renew before recording, release on every path, and stop without recording when the lease was lost.

## Orchestrator Schedule

`run_until_cancelled(cancel)` waits `initial_delay` (120 s default, Python parity), then ticks and waits `interval_for(consecutive_empty_sweeps)`:
`steps = min(consecutive_empty_sweeps, max_backoff_steps)`, `interval = min(poll_interval_ms * backoff_multiplier^steps, max_interval_ms)`, computed in integer milliseconds with saturating conversion.
A sweep that claimed nothing increments `consecutive_empty_sweeps`; any sweep that claimed a job resets it to `0`, so the interval returns to `poll_interval_ms` immediately after real work.
The loop selects between `cancel.cancelled()` and `tokio::time::sleep(delay)`; a cancelled wait returns `Err(Error::Cancelled)` promptly, and a cancellation during a tick reaches the planner through `CallOptions`, which completes the claimed job as `Cancelled` (no job is left `Running`).
A non-cancellation tick error is logged, stored in `status.last_error`, and the loop continues; only cancellation ends it.
`tick`/`tick_with` are the injectable-clock entry points: they run exactly one leased or unleased sweep of `batch_size` jobs, optionally run retention (a retention error warns and sets `last_error` but does not fail the tick), advance `ticks`/`last_tick_at`/`consecutive_empty_sweeps`/`next_interval_ms`/`last_summary`, and never sleep or spawn.
A disabled orchestrator ticks nothing, purges nothing, and leaves its status untouched.

## Constraints

- No `unwrap`/`expect`/`panic` outside `#[cfg(test)]`; `tracing` only (no `println!`/`eprintln!`), targets `openjiuwen_memory::dreaming::{config,model,outcome,repository,planner,pipeline,lease,retention,salience,sqlite,orchestrator}`.
- Errors are `openjiuwen_common::error::{Error, Result}`: `Validation` for bad input and illegal transitions, `NotFound` for unknown ids, `Unsupported` for a repository without leasing and for ML salience, `External` for storage/decoding failures, cancellation via `Error::Cancelled`.
- Every public data type is `serde`-serializable `snake_case` and tolerant of missing/unknown fields (including `DreamLeaseOptions`, whose durations are milliseconds and whose `Default` is deliberately not yet valid).
- The SQL backends are feature-gated behind `sqlite`; the default build compiles and tests fully offline with the in-memory backend and `RecordOnlySink`.
- No blocking I/O on async tasks; `sqlx` uses the tokio runtime; a `parking_lot` guard is never held across `.await` (status/lost-reason locks are taken, cloned, and dropped).
- Files stay around ≤1000 lines and are split by responsibility (`model`/`outcome`/`config`, `repository`/`sqlite`/`sqlite_lease`, `planner`/`planner/lease`, `pipeline`/`pipeline/plan`, `lease`, `retention`, `salience`, `orchestrator`).
- No new dependencies for the orchestration work: timers use `tokio::time`, cancellation uses `tokio_util::sync::CancellationToken`, and leases use the existing repositories rather than a broker.

## Design Decisions

- **A durable queue, not Python's idle scheduler.** Python's `dreaming` package contributes a busy-checker + interval loop; this crate makes the job itself durable so a pass survives the request that asked for it, can be retried, and can be inspected.
- **The orchestrator is caller-owned.** `DreamOrchestrator::run_until_cancelled` is a future an application drives (and cancels) itself; the crate never spawns it, so tests and embeddings control lifetime and there is no hidden daemon.
- **Two scheduling knobs, one clock.** `DreamConfig` keeps the Python-facing interval/settle/trigger knobs while `DreamOrchestratorConfig` carries the loop's millisecond schedule, the exponential backoff, the lease options, and the retention policy; both are plain values so a caller can persist them.
- **Leases are repository rows, not files.** The SQLite lease table and the in-memory map are claimed in the same statement/transaction as the job, so there is no window between claim and lease; renew-before-record means a worker that lost its lease cannot overwrite the new owner's row, and the guard latches the first loss so the worker stops.
- **Deterministic fallback is a first-class result.** Every LLM failure produces the rule plan plus a `degraded` reason that travels into `DreamOutcome::notes`; there is no silent degradation and no partial model output applied.
- **The deterministic salience score replaces the deferred ML model.** Importance × recency × access × kind with renormalised weights is inspectable (all four components are returned) and needs no inference runtime; `SalienceBackend::Ml` stays as a typed seam.
- **Retention is a policy object, not a timer.** `DreamRetention` is a pure value; the orchestrator's tick or a caller can run `purge_expired(now)`, which is idempotent by construction (age first, then newest-first caps).
- **`RecordOnlySink` is the default sink.** Deciding and writing are separated so the pipeline can be exercised offline; a store-backed sink is the application's injection point.

## Deferred

- ML-based salience (`SalienceBackend::Ml` is typed `Unsupported`): no inference runtime and no training data pipeline.
- A plan sink backed by `LongTermMemory` (`RecordOnlySink` records counters and mutates nothing), provider-specific prompt tuning, and cross-job learning.
- Background lease heartbeats and lease brokers: a sweep renews once per job; a consolidation longer than its TTL must renew through its own guard, and there is no broker/fencing-token service.
- Daemon-side retention: nothing purges automatically unless the orchestrator is driven (or a caller invokes `purge_expired`), and no cross-process lease supervises retention.
- The Python dreaming package's idle-layer extras (busy-checker hooks, trigger wiring into session/message events) and any graph-extraction consolidator implementation.

## Test Strategy

`model`, `outcome`, `config`, `lease`, `retention`, and `salience` unit tests cover serde compatibility (including `DreamLeaseOptions` millisecond round-trip), defaults, builders, name parsing, metrics saturation, lease loss latching, expiry boundaries, weight validation, the formula components/ordering, idempotent purges, and every validation branch.
`repository` tests cover the in-memory backend end to end: claim ordering, retry eligibility via `started_at`, requeue-then-terminal failure, complete/fail error kinds, filtering, counting, deletion, `purge`, lease claim/renew/release, takeover, and `delete_job`.
`sqlite` tests (feature `sqlite`) use `tempfile::tempdir()` plus `sqlite:{path}` URLs and fixed `DateTime::from_timestamp(1_700_000_000, 0)` timestamps: cross-reopen durability, claim ordering, concurrent claims via `tokio::join!`, expiry-inclusive takeover, lease-row cleanup, retention cutoffs computed from `Utc::now()`, JSON column round-trips, all six state counts, and the complete/fail error contract.
`planner` tests use a scripted consolidator (a `Step` queue behind `parking_lot::Mutex`, `tokio::sync::Notify` for blocking) to cover success with metrics, retry-until-`max_attempts` with an advancing clock, pre-cancelled and mid-consolidation cancellation, timeout as a failed attempt, non-terminal and mismatched outcome ids, skip semantics, disabled config, zero limit, ordering plus limit, and `run_due` batch size.
`planner/lease.rs` adds leased-sweep coverage: completion releases the lease, racing sweeps claim one job once, an expired lease is taken over and the old owner records nothing, cancellation releases, terminal/requeued failures release, a non-leasing repository reports `Unsupported`, and invalid options are rejected before claiming.
`pipeline/tests.rs` covers the LLM happy path, retry-then-fallback for prose and provider errors, no-model/disabled/no-candidate degradation, invented ids, the empty plan, sink success/failure, cancellation, config bounds, prompt truncation, both candidate sources, candidate serde, and the pure decision layer — all offline through `openjiuwen_test_support::ScriptedModel`.
`orchestrator/tests.rs` covers a tick claiming and completing a due job, the full backoff schedule up to the cap and its reset, config validation, disabled behaviour, retention on tick (idempotent, absent policy, error tolerated), leased ticks and the non-leasing error, two orchestrators on one repository (one winner), expiry takeover, `run_until_cancelled` cancellation and mid-consolidation cancellation leaving no `Running` job, and `Debug`/serde of the status and config.
Gates: `cargo fmt -p openjiuwen-memory -- --check`, `cargo test -p openjiuwen-memory --offline` and `--all-features`, `cargo clippy -p openjiuwen-memory --all-targets --all-features --offline -- -D warnings`, and `cargo doc -p openjiuwen-memory --no-deps` warning-free.

## Routing Table

- `./model.rs` → `DreamState`, `DreamKind`, `DreamTrigger`, `DreamMetrics`, `DreamJob`, `DEFAULT_MAX_ATTEMPTS`
- `./outcome.rs` → `DreamOutcome`, `DreamRunSummary`
- `./config.rs` → `DreamConfig` and `retention()`
- `./repository.rs` → `DreamFilter`, `DreamRepository` (job + lease methods), `InMemoryDreamRepository`
- `./sqlite.rs` → `SqliteDreamRepository` (job schema/CRUD, feature `sqlite`)
- `./sqlite_lease.rs` → lease schema, takeover/claim/renew/release queries (feature `sqlite`)
- `./lease.rs` → `DreamLease`, `LeaseRequest`, `LeasedJob`, `LeaseStatus`, `LeaseLostReason`, `DreamLeaseOptions`, `DreamLeaseGuard`
- `./planner.rs` → `DreamConsolidator`, `NoopConsolidator`, `DreamPlanner`/`DreamingEngine` unleased sweeps
- `./planner/lease.rs` → `DreamPlanner::run_leased`/`run_due_leased` and the leased-sweep tests
- `./orchestrator.rs` → `DreamOrchestratorConfig`, `DreamOrchestratorStatus`, `DreamOrchestrator` (tick, `run_until_cancelled`)
- `./orchestrator/tests.rs` → the orchestrator unit suite
- `./retention.rs` → `DreamRetention`, `RetentionReport`, `purge_expired`
- `./salience.rs` → `SalienceBackend`, `SalienceWeights`, `KindWeights`, `SalienceConfig`, `SalienceScore`, `salience_score`, `rank_by_salience`
- `./pipeline.rs` → candidate model, candidate sources, sink seam, `LlmConsolidatorConfig`, `LlmConsolidator`, request rendering
- `./pipeline/` → pure decision layer (`plan.rs`) and the pipeline unit suite (`tests.rs`; see `pipeline/CONTEXT.md`)
- `./prompts.rs` → consolidation system prompt, response schema, action names
- `./mod.rs` → module docs, declarations, public re-exports
