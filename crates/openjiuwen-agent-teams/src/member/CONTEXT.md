# openjiuwen-agent-teams/src/member

## Intent

The member seam of the crate: the runtime a member's "brain" implements, the status machine over one roster row, the leader-side activity projection, and the in-process spawn handle.
Python parity sources: `openjiuwen/agent_teams/agent/member.py` (`TeamMember.update_status` / `update_execution_status`), `agent/member_runtime.py` (`MemberRuntime` Protocol), `agent/member_activity.py` (`MemberActivityRegistry`, `IdleSignal`) and `spawn/inprocess_spawn.py` + `spawn/inprocess_handle.py`.
This node owns *member-level lifecycle* only: it writes the roster row through the `TeamStore` CAS seam and reports transitions.
Task assignment, mailbox routing, dispatch, event publication (topic + payload) and the team-idle marker belong to `../runtime/`; building a harness belongs to `../harness/`.

## API Surface

```rust
// runtime.rs — the brain seam
pub trait MemberObserver: Send + Sync {
    fn on_state(&self, state: HarnessState);
    fn on_round(&self, outcome: RoundOutcome);
    fn on_chunk(&self, chunk: Value);
}
#[async_trait] pub trait MemberRuntime: Send + Sync {
    fn session_id(&self) -> &str;
    fn state(&self) -> HarnessState;
    async fn start(&self, team_session: Option<Arc<dyn Session>>) -> Result<()>;
    async fn stop(&self) -> Result<()>;
    async fn send(&self, content: String, immediate: bool) -> Result<()>;
    async fn abort(&self, immediate: bool) -> Result<()>;
    async fn pause(&self) -> Result<()>;
    async fn resume(&self, query: Option<String>) -> Result<()>;
    fn subscribe(&self, observer: Arc<dyn MemberObserver>);
}
pub enum ObserverEvent { State(HarnessState), Round(RoundOutcome), Chunk(Value) }
pub struct RecordingObserver;   // ::new(), take() drains, len(), is_empty()
pub struct ObserverSet;         // ::new(), add(), state()/round()/chunk(), len(), is_empty()

// lifecycle.rs — the status machine
pub struct TeamMember;
impl TeamMember {
    pub fn new(store: Arc<dyn TeamStore>, record: MemberRecord) -> Self;
    pub fn record(&self) -> MemberRecord;
    pub fn team_name(&self) -> &str;   pub fn member_name(&self) -> &str;
    pub fn role(&self) -> TeamRole;    pub fn status(&self) -> MemberStatus;
    pub fn execution(&self) -> ExecutionStatus;
    pub fn runtime(&self) -> Option<Arc<dyn MemberRuntime>>;
    pub fn attach_runtime(&self, runtime: Arc<dyn MemberRuntime>);
    pub fn observe(&self, observer: Arc<dyn MemberObserver>);
    pub async fn refresh(&self) -> Result<MemberRecord>;
    pub async fn start(&self, team_session: Option<Arc<dyn Session>>) -> Result<()>;
    pub async fn mark(&self, to: MemberStatus) -> Result<bool>;
    pub async fn mark_execution(&self, to: ExecutionStatus) -> Result<()>;
    pub async fn note_round(&self, outcome: RoundOutcome) -> Result<()>;
    pub async fn shutdown(&self, lifetime: MemberLifetime, force: bool) -> Result<MemberStatus>;
    pub async fn error(&self, message: &str) -> Result<()>;
}

// activity.rs — leader-side projection and the pure debounce predicate
pub const DEFAULT_IDLE_DEBOUNCE: Duration;   // 2 s, Python _TEAM_IDLE_DEBOUNCE_SECONDS
pub struct MemberActivityRegistry;
impl MemberActivityRegistry {
    pub fn new() -> Self;
    pub fn register(&self, member: Arc<TeamMember>);
    pub fn mark_busy(&self, member_name: &str);   pub fn mark_idle(&self, member_name: &str);
    pub fn is_moving(&self) -> bool;
    pub fn idle_since(&self, member_name: &str) -> Option<Instant>;
    pub fn last_activity(&self) -> Option<Instant>;
    pub fn settled_members(&self) -> Vec<String>;   // sorted
    pub fn snapshot(&self) -> Vec<MemberSnapshot>;  // sorted by member_name
    pub fn remove(&self, member_name: &str) -> Option<Arc<TeamMember>>;
    pub fn len(&self) -> usize;   pub fn is_empty(&self) -> bool;
}
pub struct IdleDebounce { pub window: Duration }
impl IdleDebounce { pub const fn new(window: Duration) -> Self;
                    pub fn should_emit_idle(&self, now: Instant, last_activity: Option<Instant>) -> bool; }

// spawn.rs — in-process task ownership
pub struct MemberSpawnRequest { pub team_name, pub member_name, pub session_id: String, pub workspace_root: PathBuf }
#[async_trait] pub trait MemberSpawner: Send + Sync {
    async fn spawn(&self, request: MemberSpawnRequest) -> Result<Arc<dyn MemberRuntime>>;
    async fn despawn(&self, member_name: &str) -> Result<()>;
}
pub struct InProcessSpawnHandle;   // new(), with_cancellation(), member_name(), cancellation_token(),
                                   // is_finished(), cancel(), abort(), async join() -> Result<()>
pub struct SpawnRegistry;          // insert(handle) -> Option<handle>, contains, remove, take,
                                   // cancel_all() -> Vec<handle>, len, is_empty, names (sorted)
```

`mod.rs` re-exports the surface flat, mirroring `agent_teams/__init__.py`'s member exports.
`openjiuwen_harness_protocol::HarnessState` is re-exported by `../schema/`; this node imports it and never defines a second state enum.

## Lifecycle rules

| Operation | From | To | Notes |
| --- | --- | --- | --- |
| `start` | `unstarted` | `starting` → `ready` | CAS; non-`unstarted` is `Error::validation("member '<name>' is not in unstarted state")` |
| `start` (runtime error) | `starting` | `unstarted` | rollback via CAS; a failed rollback is logged, the original error is returned unchanged |
| `mark` (same status) | any | same | idempotent `Ok(true)`, no store write |
| `mark` (illegal edge) | any | rejected | `Ok(false)`, store untouched |
| `mark` (lost CAS) | any | rejected | `Ok(false)` after refreshing the cached row |
| `note_round(Started)` | `idle` | `starting → running` | execution walks the `round_transitions` table in order |
| `note_round(Finished)` | `running` | `completing → completed → idle` | |
| `note_round(Aborted)` | `running` | `cancel_requested → cancelling → cancelled → idle` | |
| `note_round(Failed)` | `running` | `failed → idle` | |
| `note_round` (member mirror) | walk ends `running` | `busy` | applied only when the member table allows the edge |
| `note_round` (member mirror) | walk ends `idle` | `ready` | same "where legal" guard; a retiring member keeps `shutdown_requested` |
| `shutdown(lifetime, force)` | live status | `shutdown_requested` → `retire_target` | direct to target when `shutdown_requested` is unreachable (e.g. `unstarted → shut_down`) |
| `shutdown` (dormant, unstarted) | `unstarted` | rejected | `Error::validation("member '<name>' cannot shut down from status 'unstarted'")` |
| `shutdown` (already at target) | target | target | idempotent `Ok(target)` |
| `error(message)` | recoverable | `error` | best effort; logs the message; a departed member keeps its status and the call still returns `Ok(())` |

`retire_target(MemberLifetime::Shutdown | Dormant, force)`: `dormant` only for `Dormant` with `force == false`, `shut_down` otherwise.
The status machine never stops the runtime: `shutdown` only settles the row, and the runtime manager owns teardown order.

## Constraints

- Only files under `src/member/` are edited; `../lib.rs`, `../schema/`, `../storage/` and sibling modules belong to other nodes.
- `openjiuwen_common::error::{Error, Result}` only; no panic on user input; no `unwrap`/`expect` outside `#[cfg(test)]`.
- Async-first: `#[async_trait]` for the object-safe seams; `parking_lot` locks are held for one map read/write only and never across `.await`; cancellation is `tokio_util::sync::CancellationToken`.
- Config errors for missing build dependencies (no runtime attached), validation errors for bad names/states, not-found for missing rows, unsupported for deferred spawn modes.
- `tracing` only; no `println!`/`eprintln!`.
- No `unsafe`; no global mutable state — every registry is a local instance owned by the caller.
- Files ≤1000 lines; the status-machine suite lives in `lifecycle_tests.rs` so the implementation stays readable.
- `--no-default-features` must build: this node has no feature gates.

## Routing Table

- `./mod.rs` → module docs, flat re-exports, `#[cfg(test)] mod lifecycle_tests`
- `./runtime.rs` → `MemberRuntime`, `MemberObserver`, `RecordingObserver`, `ObserverEvent`, `ObserverSet`
- `./lifecycle.rs` → `TeamMember`: cached row, `mark`/`mark_execution`, `note_round`, `start`+rollback, `shutdown`, `error`
- `./lifecycle_tests.rs` → status-machine suite with `FakeMemberRuntime` (`#[cfg(test)]`)
- `./activity.rs` → `MemberActivityRegistry`, `IdleDebounce`, `DEFAULT_IDLE_DEBOUNCE`
- `./spawn.rs` → `MemberSpawner`, `MemberSpawnRequest`, `InProcessSpawnHandle`, `SpawnRegistry`

## Test Strategy

Unit tests live beside each file; `cargo test -p openjiuwen-agent-teams --lib` runs them offline and deterministically (313 crate tests, 55 of them in this node).
The status-machine suite is `lifecycle_tests.rs`: start happy path (`starting` then `ready`, runtime started once), start failure rollback with the original error propagated, start without a runtime and start on a non-`unstarted` member, idempotent re-assert, illegal edge untouched, lost CAS refreshes the cache, execution validation and persistence, all four `note_round` walks with the `busy`/`ready` mirror, repeated round notes and the duplicate-note-does-not-revive guard, retire targets (`shutdown`/`dormant`/forced/direct-to-`shut_down`/unreachable dormant/idempotent), `error()`, observer queuing and forwarding, `refresh` after a peer write, and the "status machine never stops the runtime" invariant.
`runtime.rs` covers recording order/drain and observer-set fan-out; `activity.rs` covers `is_moving`, `settled_members`, `idle_since`/`last_activity`, re-registration, removal, sorted snapshots and the `IdleDebounce` boundary (inclusive window, `None`, backwards clock, zero window); `spawn.rs` covers cancel+join, abort→`Cancelled`, task error propagation, panic→`External`, `is_finished`, registry insert-replace/remove/take/cancel-all bookkeeping, request serde and spawner object safety.
Every file has a doctest; all four pass. The crate still reports two pre-existing doctest failures (`lib.rs`, `constants.rs`) that reference crate-root re-exports not yet declared — those belong to the crate root node, not here.
Definition of done for this node: `rustfmt --check` clean on every `src/member/*.rs`, `cargo test -p openjiuwen-agent-teams --lib` and `--no-default-features --lib` green, and no clippy diagnostic pointing at `src/member/`.
Crate-wide `cargo clippy -- -D warnings` still reports `missing_docs` on the five still-empty sibling modules (`harness`, `mcp`, `runtime`, `skill`, `taskboard`), declared in `../lib.rs`; those belong to the crate root node.

## Notes for Agents

- Python keeps the `TeamMember` handle, the member table and the event stream in sync; the Rust port keeps the row authoritative behind `TeamStore::transition_member` (CAS) and mirrors it into a cached projection for the synchronous getters. Call `refresh()` when a peer may have written.
- Event publication is deliberately absent: `TeamMember` has no messager. The runtime (`TeamBackend`) owns the topic and publishes `MemberStatusChangedEvent` / `MemberExecutionChangedEvent` after a successful `mark`.
- `mark` short-circuits an equal status to `Ok(true)` (Python's no-op path), so recovery code can assert `ready` unconditionally without polluting a status stream.
- `mark_execution` validates locally because the store's `update_member` patch path is verbatim; it is an error (not `Ok(false)`) to ask for an illegal execution edge, because only `note_round` is expected to tolerate that.
- `note_round` skips execution steps the current status cannot reach, which makes late or duplicated observer callbacks harmless; the final step is what the row ends on.
- A round walk that changed nothing mirrors no lifecycle status unless the member is still `busy`/`starting`, so a duplicate terminal note cannot revive a `paused` or `stopped` member.
- `shutdown` mirrors the member-side settle (`TeamAgent.shutdown_self`) rather than the leader-side request (`shutdown_member` writes `shutdown_requested` and stops there): it may write the terminal target directly when `shutdown_requested` is unreachable, so a never-started member still retires to `shut_down`.
- `MemberLifetime` is a plain parameter (Python reads it from the member row); the caller resolves it from the member record before asking.
- The activity registry is a deliberate Rust reshape of Python's `IdleSignal`-returning registry: Python folds movement into a "marker owed" enum, Rust exposes the questions (`is_moving`, `settled_members`, `idle_since`, `last_activity`) and leaves the marker timer to `TeamBackend`. `IdleDebounce` is the only timing concept and it is a pure predicate.
- `settled_members` uses `MemberStatus::is_settled` **and** "not moving", so a `paused` member finalising a round is not reported settled; `unstarted` is quiescent but never settled.
- `mark_busy` / `mark_idle` only affect registered members; `mark_idle` keeps the first idle timestamp so repeated status events cannot postpone the idle marker.
- Dropping a `tokio::task::JoinHandle` detaches the task, so every handle that leaves `SpawnRegistry` must be joined or aborted: `take`/`cancel_all` cancel the token and hand the handles back, and `insert` returns the replaced handle. `TeamBackend` is the documented owner.
- `InProcessSpawnHandle::new` mints its own token; a spawner whose task must observe cancellation uses `with_cancellation` (the task holds a clone from `cancellation_token()`).
- `MemberSpawner` builds but never starts a runtime: `MemberRuntime::start` is called by `TeamMember::start` so a failed start can roll the row back.
- Deferred here and surfaced as `Error::Unsupported` by the spawner implementation: OS-process members (`spawn_mode="process"`), external-CLI members, SSH transport, fork/compact injection and swarmflow sharing (`inprocess_spawn.py`'s context injection has no Rust counterpart yet).
- Python's rail/memory/background-controller helpers on `MemberRuntime` (`find_rails`, `register_member_tools`, `inject_member_memory`, `set_background_task_controller`, `workspace`, `sys_operation`) are not part of the Rust seam; the harness adapter reaches its own agent directly.
- `TeamMember` implements `Debug` non-exhaustively (identity, cached statuses, whether a runtime is attached); `ObserverSet` prints its observer count.
