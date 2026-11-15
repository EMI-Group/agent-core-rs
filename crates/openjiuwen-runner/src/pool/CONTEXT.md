# openjiuwen-runner — `src/pool/`

## Intent

The in-process team runtime pool: a bounded set of live team runs keyed by `(team_name, session_id)`, each with a monitor entry, a HITL binding, an inbound-payload buffer, a cancellation token, and a broadcast frame stream.
Ports the in-process part of `openjiuwen/core/runner/team_runner.py`'s `TeamRuntimeManager` pool: admission/reject, the active-team listing, stop, the typed pause/resume refusals, orphan reclaim, and single-run-per-key.
Every run in the pool is driven by `crate::run::stream_team`; the pool never re-implements the execution path, it only owns the *observation* seams (monitor + HITL) and the lifecycle around it.

## API Surface

- `TeamPool` (`src/pool/mod.rs`) — cheap-clone `Arc` handle: `new`, `with_config`, `from_runner_config`, `config`, `capacity`, `available_permits`, `start`, `try_start`, `acquire`, `get`, `list_active`, `stop`, `stop_all`, `pause`, `resume`, `take_payloads`, `len`, `is_empty`, `is_closed`. `Default` = `new()`.
- `PoolKey` (`src/pool/key.rs`) — `{ team_name, session_id }`, `new`, `for_team`, `Display` as `team@session`.
- `PoolRequest` (`src/pool/request.rs`) — runtime-only request (`team`, `inputs`, `session_id`, `label`, `options`) with `new`, `with_session_id`, `with_label`, `with_options`, `with_cancellation`, `with_timeout`, and a `key(&RunnerConfig)` derivation.
- `PoolRun` — `run_id`, `key`, `team_name`, `session_id`, `state`, `subscribe`, `wait`, `output`, `cancel`, `payloads`, `is_finished`.
- `PoolLease` — `key`, `run`, `wait`, `pause`, `resume` (both refuse), `release`, `is_released`, `run_id`, `state`; `Drop` releases an unreleased lease.
- `PoolFrame` (`src/pool/entry.rs`) — `Started`, `Payload`, `Output`, `Completed`, `Failed`, `Cancelled`, serde-tagged by `kind` (no pause/resume frame: nothing suspends a run).
- `ActiveTeamInfo` — read-only snapshot: `run_id`, `team_name`, `session_id`, `label`, `state`, `gate_closed`, `running`, `leases`, `queued`.
- `PoolError` (`src/pool/error.rs`) — `AtCapacity`, `Busy`, `NotFound`, `Closed`, `Cancelled`, with `kind()`, `into_error()`, `From<PoolError> for Error`.
- `PoolShutdown` — `{ stopped, first_error }` with `is_ok`, `into_result`; `stopped` counts only the live runs the sweep stopped, and retained finished entries are removed from the pool but not counted.
- `TeamPoolConfig` and the `DEFAULT_TEAM_POOL_*` constants live in `crate::config`; the pool reads them through `TeamPoolConfig`'s accessors (`max_runs`, `idle_timeout`, `run_timeout`, `payload_capacity`, `frame_capacity`, `reaper_interval`, `stop_timeout`).

## Constraints

- `parking_lot::Mutex` guards the entry map and each entry's state; no guard is ever held across an `.await` (values are cloned or `Arc`s are taken out first).
- Only `tokio::time` (`Instant`, `sleep`) is used for deadlines and idle stamps, never `std::time::Instant`.
- Single live run per key: `start`/`try_start` reject a live key with `PoolError::Busy`, `acquire` joins it (the joining request's inputs/options are ignored) and takes another lease.
- Bounded concurrency: one `tokio::sync::Semaphore` created with `max_runs()` permits; a queued admission awaits a permit in a `tokio::select!` biased to the entry's cancellation token.
- Finalisation happens exactly once (`AtomicBool` swap) and never panics on user input; a panicking team is caught with `catch_unwind` and finalised as `Failed("run task panicked")`.
- Every public item is documented; logs go to `tracing` target `openjiuwen_runner::pool`; no `println!`.

## Design Decisions

- **The pool owns the observation seams.** A pooled run is given a `RunServices` copy with `monitor` and `hitl` cleared (`pooled_services`), so `stream_team` neither creates a second monitor entry nor a second HITL binding; the pool registers one of each and finishes them in a fixed order (monitor first, then unbind the hub) so `InteractionHub::unregister` cannot overwrite the pool's terminal phase.
- **One worker task per admitted run**, holding an `OwnedSemaphorePermit` (released at finalisation) and the entry's `CancellationToken`; `hub.cancel` trips the same token, so a host can stop a pooled run through either surface.
- **Cancellation and deadlines are pool-scoped.** The worker selects between the entry token, the caller's `RunOptions::cancellation`, and `next_stream_step`; the effective deadline is `request.options.timeout` else `config.run_timeout()`.
- **`acquire` joins, `start` rejects.** Python's `REJECT_RUNNING` is a rejection for a *new* run; a caller that wants "the run for this key, whoever owns it" uses `acquire` and gets a lease that keeps the entry out of idle eviction.
- **Finalisation order**: state written → monitor finished → terminal frame emitted → hub binding dropped → permit released → `wait()` woken → retention pruned.
- **Retention is best-effort**: terminal entries stay (so `get`/`wait` keep working) until `retain_finished` newer terminal entries exist, and an entry with a live lease is never pruned.
- **Pause/resume refuse instead of pretending.** `TeamPool::pause`/`resume` keep their signatures and negative answers (unknown key or finished entry → `PoolError::NotFound`) but a live run answers `Error::Unsupported` from `crate::error::no_suspension_seam`: an in-process strategy owns no mid-flight suspension seam, so the pool cannot suspend or restart the invocation. The refusal changes nothing (no phase, gate, frame, or monitor transition); `stop`/`PoolRun::cancel` trip the token for a host that must stop the run.
- **Payloads, not gates**: the pool registers the run on the hub and drains its inbox into a bounded `VecDeque` (`payload_capacity`, oldest dropped on overflow), so a host re-invokes the strategy with the drained payloads. No session is attached to the hub entry, so `InteractionHub::answer` cannot resolve in-flight interaction requests of a pooled run.
- **Idle eviction is one lazy task per pool** (`src/pool/reaper.rs`), started by the first admission when `idle_timeout()` is `Some`; it uses only `tokio::time` so an embedding process that enables tokio's `test-util` can pause and advance it deterministically. An idle *live* orphan is cancelled and removed (Python's orphan reclaim), never a leased entry.
- **Errors are `openjiuwen_common::error::Error`**; `PoolError` only exists to name the pool's failure vocabulary and map it onto `ErrorKind` (`AtCapacity`/`Closed` → `Config`, `Busy` → `Validation`, `NotFound` → `NotFound`, `Cancelled` → `Cancelled`). Because `Error` is not `Clone`, an entry stores a `PoolFailure { kind, message }` and rebuilds the error on demand.

## Test Strategy

- Unit tests beside each module (`#[cfg(test)] mod tests`): config defaults/serde/clamping, `PoolKey` derivation and `Display`, `PoolError` kind/message mapping, `ActiveTeamInfo`/`PoolFrame` serde shape, finalisation idempotence and lease accounting, payload bounding, retention pruning, idle-deadline arithmetic (`due_at`/`is_due`), reaper install-at-most-once, admission guards (closed pool, no runtime, out of permits), and the `stop_all` counting rule (a finished retained entry is swept but not counted).
- No unit test builds a real team; the end-to-end behaviour (start/join/stop/stop_all, frames, cancellation, deadlines, payload drain, hub unbinding) belongs to the `tests/team_pool*.rs` suite.
- Eviction deadlines are computed with `tokio::time::Instant` and `sleep`, so a paused clock drives them.
- This crate's test build cannot enable tokio's `test-util` feature: doing so needs a `tokio` dev-dependency feature change in `Cargo.toml`, which is owned elsewhere.
- The integration suite therefore uses short real idle timeouts and polls their effect with bounded `tokio::time::timeout` waits instead of `advance()`, so those cases are not clock-deterministic and the timeouts stay generous.

## Deferrals

- Distributed transports (Pulsar/ZeroMQ) and cross-process team attachment — `RunnerConfig::distributed` stays data only.
- OS-process and external-CLI team members, `member=True` spawn routing.
- The live `InteractGate` wake-up: a pooled run does not suspend on an interaction and resume it in place; payloads buffer for the host to replay.
- Python's team product surface beyond the pool: `delete_agent_team`, `TeamStreamLogger`, monitoring/TUI, `register_human_agent_inbound`, team pools' HITL/monitor product endpoints.

## Routing Table

- `./mod.rs` → `TeamPool` facade, `TeamPoolInner` (entries, semaphore, closed flag, sequence, reaper slot), retention pruning
- `./key.rs` → `PoolKey` and team-name derivation
- `./error.rs` → `PoolError`, `PoolShutdown`, kind mapping, `rebuild`
- `./entry.rs` → `Entry`, `EntryState`, `ActiveTeamInfo`, `PoolFrame`, finalisation, leases, payload buffer
- `./request.rs` → `PoolRequest`: the runtime-only request, its builders, session-key derivation, effective options
- `./run.rs` → `PoolRun`, `PoolLease`, admission (`start`/`try_start`/`acquire`), the worker and payload-drain tasks, `pooled_services`
- `./reaper.rs` → idle-eviction task, `due_at`/`is_due`, `ensure_started`, `reap`
