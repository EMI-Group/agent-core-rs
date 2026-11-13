# monitor/ — run status and event monitoring
## Intent
Keeps one status snapshot per run the runner is executing and publishes one event per state transition on a tokio broadcast channel, so a host can render live progress and the final outcome without polling the session.
In-memory only: the monitor is a view of what the runner is doing, not a durable record (see Notes & Deferrals).
A spawned child is tracked like any other run: `Spawn` is the kind `spawn::MonitoredSpawnedAgent` admits, updates on health probes, and finishes on shutdown, kill, or drop, so spawn lifecycle events travel through the same bounded per-run history and broadcast channel as agent, workflow, and team runs.
Python parity: Python had no standalone monitor type; the fields are the union of `team_runner.py`'s active-team records (component label, session id, status) and the callback `_framework:*` event payloads (`run_id`, component, session id).
## API Surface
```rust
// types (status.rs) — serde snake_case, tolerant of unknown/missing fields
pub enum RunKind { Agent, Workflow, Team, Spawn }           // Copy+Eq+Hash, Default = Agent
pub enum RunPhase { Queued, Running, Paused, Completed, Failed, Cancelled } // Default = Queued
impl RunPhase { pub fn is_terminal(&self) -> bool; }        // Completed | Failed | Cancelled
pub struct RunStart { pub kind: RunKind, pub label: String, pub session_id: Option<String>, pub detail: Value }
impl RunStart { pub fn new(kind: RunKind, label: impl Into<String>) -> Self; with_session; with_detail } // + Default
pub struct RunOutcome { pub phase: RunPhase, pub error: Option<String>, pub detail: Value }
impl RunOutcome { pub fn completed() -> Self; pub fn failed(message: impl Into<String>) -> Self;
                  pub fn cancelled() -> Self; pub fn with_detail(detail: Value) -> Self }            // + Default = completed
pub struct RunStatus { pub run_id: String, pub kind: RunKind, pub label: String, pub phase: RunPhase,
                       pub session_id: Option<String>, pub started_at: i64, pub finished_at: Option<i64>,
                       pub error: Option<String>, pub detail: Value }
impl RunStatus { pub fn is_terminal(&self) -> bool; }       // + Default (empty Queued status)
// events (event.rs)
pub struct RunEvent { pub run_id: String, pub kind: RunKind, pub label: String, pub phase: RunPhase,
                      pub session_id: Option<String>, pub at_ms: i64, pub detail: Value }          // + Default
// bounds (mod.rs)
pub const EVENT_HISTORY_LIMIT: usize = 128;                 // per-run history, oldest dropped
pub const LIVE_RUN_LIMIT: usize = 64;                       // finished runs retained
pub const EVENT_CHANNEL_CAPACITY: usize = 256;              // broadcast buffer per subscriber
// facade (mod.rs) — Clone is cheap (one Arc)
pub struct RunMonitor { /* Arc<Inner> */ }
impl RunMonitor {
    pub fn new() -> Self;                                   // default limits
    pub fn with_capacity(event_capacity: usize) -> Self;    // capacity < 1 is raised to 1
    pub fn begin(&self, start: RunStart) -> RunStatus;      // mints a UUID v4 id, phase Queued
    pub fn track(&self, run_id: impl Into<String>, start: RunStart) -> RunStatus; // adopts an existing id unchanged
    pub fn update(&self, run_id: &str, phase: RunPhase, detail: Value) -> Result<RunStatus>;
    pub fn finish(&self, run_id: &str, outcome: RunOutcome) -> Result<RunStatus>;
    pub fn snapshot(&self, run_id: &str) -> Result<RunStatus>;
    pub fn list_active(&self) -> Vec<RunStatus>;            // non-terminal, sorted by (started_at, run_id)
    pub fn history(&self, run_id: &str) -> Result<Vec<RunEvent>>; // oldest first
    pub fn subscribe(&self) -> tokio::sync::broadcast::Receiver<RunEvent>;
    pub fn clear(&self);                                    // drops runs/history/retention, publishes nothing
}
impl Default for RunMonitor /* == new() */
// crate-internal seam (guard.rs) — NOT public; re-exported as `pub(crate) use guard::RunStatusGuard`
pub(crate) struct RunStatusGuard { monitor: Option<Arc<RunMonitor>>, run_id: String, finished: bool }
impl RunStatusGuard {
    pub(crate) fn begin(monitor: Option<Arc<RunMonitor>>, run_id: &str, start: RunStart) -> Self; // None = inert
    pub(crate) fn running(&mut self, detail: Value);
    pub(crate) fn ok(&mut self, detail: Value);
    pub(crate) fn fail(&mut self, error: &openjiuwen_common::error::Error, detail: Value); // Cancelled -> Cancelled, else Failed
    pub(crate) fn cancelled(&mut self, detail: Value);
}
impl Drop for RunStatusGuard // unfinished -> finish as Cancelled with detail {"reason": "run dropped"}
```
## Semantics
- One event per transition: `begin`/`track` publish the `Queued` event, `update` publishes the phase it was given, `finish` publishes the outcome phase; the per-run history append and the channel publish happen in the same lock step.
- `track` on an already-tracked id publishes nothing and returns the existing status unchanged; on an unknown id it behaves like `begin` under that id.
- `update` on an unknown id is `Error::NotFound("run '<id>' is not tracked")`; on a run whose current phase is terminal, and on a request for a terminal phase, it is `Error::Validation` — terminal phases go through `finish` so `finished_at` is stamped and the run is retained.
- `finish` on an unknown id uses the same `NotFound` wording; a second `finish` is the same `Validation` wording; the outcome's phase is recorded as given.
- History retains the newest `EVENT_HISTORY_LIMIT` events per run; a retained finished run keeps at least its `Queued` and terminal events.
- Retention: every live run stays; finishing one more than `LIVE_RUN_LIMIT` finished runs evicts the oldest finished run along with its status and history, so `snapshot`/`history` of an evicted id are `NotFound`.
- `Spawn` runs are ordinary runs with label == the spawned handle's id: `spawn::MonitoredSpawnedAgent` admits them in `Running` (after a `Queued` track), reports each `health` probe as a `Running` update whose detail is `{backend, alive, healthy, pid}` (or `{healthy: false, error}`), and finishes them as `Completed` (`{"graceful": …}` / `{"reason": "handle_dropped"}`), `Cancelled` (`{"reason": "force_kill"}`), or `Failed`; nothing about the retention bounds or the event contract differs.
- `list_active` excludes terminal runs and sorts by `(started_at, run_id)`, so the order is deterministic when two runs share a millisecond; `snapshot` and `history` are unaffected by ordering.
- `subscribe` returns a receiver for events published from then on; a slow subscriber gets `Err(RecvError::Lagged(n))` (frames dropped for that subscriber only, it can keep receiving) and `RecvError::Closed` only once every `RunMonitor` clone is gone.
- Timestamps are epoch milliseconds (`SystemTime` since `UNIX_EPOCH`), saturated at `i64::MAX` and `0` for a pre-epoch clock; no call panics on the clock.
- Guard methods are synchronous and best-effort: a mirror error is logged with `tracing::warn!` (target `openjiuwen_runner::monitor`) and never returned, and `Drop` needs no runtime and never panics.
## Constraints
- Synchronous and `await`-free: one `parking_lot::Mutex` guards the statuses, histories, and retention queue for a short critical section per transition, always dropped before returning; a monitor call is safe from a run path, a `Drop`, or a plain test.
- `Arc<Inner>` holds the mutex and the `broadcast::Sender<RunEvent>`; cloning a `RunMonitor` shares state, so a clone observes every later transition.
- No new dependencies: `uuid` (via `openjiuwen_common::utils::id::uuid_v4`), `tokio::sync::broadcast`, `serde`/`serde_json`, `parking_lot`, `tracing`.
- Every public item and field is documented; workspace lints include `missing_docs`, `unreachable_pub`, clippy `all`+`pedantic`; `#[cfg(test)]` modules allow `unwrap_used`/`expect_used`/`assert_is_empty` like the rest of the crate.
- Files stay small: `mod.rs` (facade + façade tests) ≈ 510 lines, the submodules are ≤ 380 lines.
## Routing Table
- `./mod.rs` → `RunMonitor` facade, the three public constants, module-level contract docs, facade unit tests
- `./status.rs` → `RunKind`, `RunPhase`, `RunStart`, `RunOutcome`, `RunStatus` (+ serde shape tests)
- `./event.rs` → `RunEvent`
- `./store.rs` → `Inner` (mutex + broadcast sender), the transition logic, retention/eviction, `now_ms`, bounds tests
- `./guard.rs` → `RunStatusGuard` lifecycle mirroring and its tests
- `../run/` → the consumers: run paths build a guard at admission and finish it explicitly or rely on `Drop`
## Notes & Deferrals
- In-memory only: nothing is persisted, no events are shipped over the message queue, a websocket, or a file, and there is no metrics/aggregation layer.
- No wall-clock ordering guarantee across runs: `at_ms` is informational and two runs can share a millisecond; only `list_active` sorts deterministically.
- `RunStatusGuard` is unused inside the crate until the run paths adopt it, hence the documented `#[allow(dead_code)]` on `mod guard` and `#[allow(unused_imports)]` on its `pub(crate)` re-export.
- Backpressure is per subscriber: the channel never blocks a transition, so a slow host loses frames (with a `Lagged` signal) rather than stalling or slowing the run.
- Deferred: event persistence/replay, per-run filtering subscriptions, aggregate metrics, and any transport (MQ/HTTP/websocket) for events.
