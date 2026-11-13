# openjiuwen-agent-teams/src/monitor

## Intent

Plain-text, greppable status stream for a live team: one record per line, no TUI, no colour, no cursor control.
Ports the idea of `openjiuwen/agent_teams/monitor/{models.py,team_monitor.py,stream_logger.py}` (Python v0.1.20) — a renderable event model plus a streaming logger — without the database, the Textual app or the chunk aggregation.
The node renders what the runtime hands it: no store query, no `TeamStore` access, no chunk buffering.
Three layers: `event` (renderable event model), `render` (deterministic one-line renderer), `stream` (bounded async emitter with backpressure).

## API Surface

```rust
// event.rs
pub enum MonitorEvent { TeamStatus{state: HarnessState}, TeamStopped, MemberSpawned{member_name: String, role: TeamRole},
                        MemberStatus{member_name: String, status: MemberStatus, task_id: Option<String>},
                        MemberRound{member_name: String, round: u32, outcome: RoundOutcome},
                        MemberStopped{member_name: String, reason: Option<String>},
                        TaskCreated{task_id: String, title: String},
                        TaskStatus{task_id: String, status: TaskStatus, assignee: Option<String>} }
impl MonitorEvent {
    pub fn from_member_status(event: &MemberStatusChangedEvent, task_id: Option<String>) -> Self;
    pub fn from_task_created(event: &TaskCreatedEvent) -> Self;
    pub fn from_task_status(event: &TaskStatusChangedEvent) -> Self;
    pub const fn kind(&self) -> &'static str;         // "team_status" … "task_status"
    pub fn subject(&self) -> &str;                    // member name / task id / ""
}
pub struct StampedEvent { pub timestamp: u64 /* millis since epoch */, pub event: MonitorEvent }
impl StampedEvent { pub const fn new(timestamp: u64, event: MonitorEvent) -> Self; pub fn now(event: MonitorEvent) -> Self; }
pub fn snapshot_events(snapshot: &TeamSnapshot) -> Vec<StampedEvent>;

// render.rs
pub enum TimestampPolicy { Hidden, Seconds /* default */, Millis }
pub struct TeamMonitorConfig { pub timestamp: TimestampPolicy, pub utc_offset_seconds: i32, pub include_header: bool }
pub struct TeamMonitor;
impl TeamMonitor {
    pub fn new() -> Self;
    pub fn with_config(config: TeamMonitorConfig) -> Self;
    pub const fn config(&self) -> &TeamMonitorConfig;
    pub fn render_event(&self, event: &StampedEvent) -> String;     // one line, no trailing newline
    pub fn render_snapshot(&self, snapshot: &TeamSnapshot) -> String; // lines joined by '\n', no trailing newline
}

// stream.rs
pub enum BackpressurePolicy { Block, #[default] DropOldest }
pub enum EmitOutcome { Sent, DroppedOldest, DroppedNew, Closed }
impl EmitOutcome { pub const fn as_str(self) -> &'static str; }   // "sent" | "dropped_oldest" | "dropped_new" | "closed"
pub struct StreamSummary { pub written: u64, pub dropped: u64 }
pub struct MonitorEmitter;
impl MonitorEmitter {
    pub fn try_emit(&self, event: MonitorEvent) -> EmitOutcome;
    pub async fn emit(&self, event: MonitorEvent) -> EmitOutcome;
    pub fn emit_at(&self, timestamp: u64, event: MonitorEvent) -> EmitOutcome;
    pub fn emitted(&self) -> u64;   // events that entered the buffer (sent + drop-oldest evictions)
    pub fn dropped(&self) -> u64;   // refusals + events evicted by drop-oldest
    pub fn is_closed(&self) -> bool;
    pub fn policy(&self) -> BackpressurePolicy;
    pub fn capacity(&self) -> usize;   // clamped to at least 1
}
impl Clone for MonitorEmitter;  impl Drop for MonitorEmitter;   // last clone closes the queue
pub struct MonitorReceiver;
impl MonitorReceiver {
    pub async fn recv(&mut self) -> Option<StampedEvent>;   // None once closed AND drained
    pub fn try_recv(&mut self) -> Option<StampedEvent>;
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
}
pub struct StatusStreamHandle;
impl StatusStreamHandle {
    pub async fn shutdown(self) -> openjiuwen_common::error::Result<StreamSummary>;
    pub fn abort(&self);
}
pub struct StatusStream;
impl StatusStream {
    pub fn channel(capacity: usize, policy: BackpressurePolicy) -> (MonitorEmitter, MonitorReceiver);
    pub fn spawn<W>(writer: W, monitor: TeamMonitor, capacity: usize, policy: BackpressurePolicy)
        -> (MonitorEmitter, StatusStreamHandle) where W: tokio::io::AsyncWrite + Unpin + Send + 'static;
}

// mod.rs flat re-exports
pub use event::{snapshot_events, MonitorEvent, StampedEvent};
pub use render::{TeamMonitor, TeamMonitorConfig, TimestampPolicy};
pub use stream::{BackpressurePolicy, EmitOutcome, MonitorEmitter, MonitorReceiver, StatusStream, StatusStreamHandle, StreamSummary};
```

## Rendered-line contract

Prefix (optional, from `StampedEvent::timestamp` shifted by `TeamMonitorConfig::utc_offset_seconds`, UTC-based): `Hidden` → none, `Seconds` → `[HH:MM:SS] `, `Millis` → `[HH:MM:SS.mmm] ` — brackets are literal and a single space separates the prefix from the body.
Bodies, exactly: `team {state}` · `team stopped` · `member {name} spawned role={role}` · `member {name} {status}` (` task={id}` appended when known) · `round {name} round={n} outcome={outcome}` · `member {name} stopped` (` reason={reason}` appended when known) · `task {id} created title={title}` · `task {id} {status}` (` assignee={name}` appended when known).
`render_event` returns one line with no trailing newline; the writer task appends exactly one `\n` per rendered line.
`render_snapshot` joins with `'\n'`: header `team {name} {state} members={n} tasks={m}` (when `include_header`), then members sorted by name, then tasks sorted by id — both stamped with `TeamSnapshot::updated_at`; no trailing newline; header off plus an empty snapshot renders the empty string.
`state_str` maps `HarnessState` to `idle|running|pausing|paused|terminated`, and `round_outcome_str` maps `RoundOutcome` to `started|finished|aborted|failed`; both live in `render.rs` because `openjiuwen_harness_protocol::HarnessState` has no `as_str`/`Display` and `schema::RoundOutcome` has no `as_str`.
Time of day is computed in `i128` and reduced with `rem_euclid`, so any `u64` timestamp and any `i32` offset wraps modulo 24 hours instead of panicking (a negative offset crossing midnight included).

## Constraints

- Edit scope: only `src/monitor/{mod.rs,event.rs,render.rs,stream.rs,CONTEXT.md}`; the node never imports `crate::observability`.
- Dependencies: `serde`, `serde_json`, `tokio` (rt/rt-multi-thread/macros/sync/time/fs/io-util), `parking_lot`, `tracing`, `tempfile`, `futures` (`poll!` in tests) and `openjiuwen_common::error::{Error, Result}`; no new crates.
- Errors: `openjiuwen_common::error::Error` only (`Error::external` for sink and join failures); user input never panics.
- Logging: `tracing` only — no `println!`/`eprintln!` in library code.
- Async: no lock held across `.await` (the `items` guard is dropped before every await), no blocking call in an async task, `tokio::spawn` only inside `StatusStream::spawn`.
- No `unsafe`; workspace lints `all` + `pedantic` + `unwrap_used`/`expect_used` with `-D warnings`; files ≤ 1000 lines.

## Design Decisions

- Event model over DB models: `MonitorEvent` carries only the fields a line renders, unlike Python's flattened Pydantic `MonitorEvent`, so history/query changes cannot break the renderer and `MemberStatusChangedEvent::previous` is simply not carried.
- `HarnessState` string map lives here (the protocol crate exposes no `as_str`/`Display`), pinned to the serde wire names by `state_str_matches_serde_wire_values`; `RoundOutcome` gets the same treatment in `round_outcome_str_matches_serde_wire_values`.
- UTC+offset rendering instead of local time: deterministic golden tests, no `chrono`/tz dependency, and modular arithmetic so no timestamp or offset range can panic.
- `DropOldest` is the default backpressure because a monitoring sink must never stall the team it observes; `Block` exists for sinks that must not lose a record and is the only policy whose `emit()` waits.
- The receiver is `MonitorReceiver`, not `tokio::sync::mpsc::Receiver`: drop-oldest eviction is impossible from the sender side of an mpsc channel (a full channel cannot have its oldest element removed by the sender), and the contract is "when full, discard the oldest buffered event and enqueue the new one".
- Counters: `emitted` counts events that entered the buffer (sent + drop-oldest evictions), `dropped` counts refusals plus evictions, and `StreamSummary.written` counts lines fully written (both `write_all` calls) — so `written + dropped` exposes any loss a caller should know about.
- Shutdown is close → drain in FIFO order → single `flush` → join, returning `StreamSummary`; dropping the last emitter closes the queue too, so a writer task can never hang, and a later `try_emit` is `Closed`.
- A writer I/O error closes the queue and surfaces as `Err(Error::external(..))` rather than panicking on a broken pipe or a closed sink.
- Capacity `0` clamps to `1` (a zero-length buffer can hold nothing, which no policy can express), which is why `channel` returns no `Result`.
- No TUI: the contract is plain text lines, and presentation beyond them belongs to the CLI front-end.
- Eviction is a documented two-step (pop front, then push the new event), so a competing producer can take the freed slot and turn the outcome into `DroppedNew`; accepted because monitoring is observational and the loss is counted.
- Cancelling `emit()` before it completes drops the event it carried and leaves no counter trace; monitoring is best-effort and must never keep a task alive.
- Dropping `MonitorReceiver` while a `Block` emitter waits leaves that emitter waiting until the queue is closed — callers own the pairing of receiver lifetime and close.

## Emission points

| Host hook | Emitted event |
| --- | --- |
| `TeamAgent::start` / team runtime state change | `TeamStatus { state }` |
| `TeamAgent::stop` / team teardown | `TeamStopped` |
| `spawn_member` / member row creation | `MemberSpawned { member_name, role }` |
| `MemberStatusChangedEvent` (plus the task the member is working on) | `MemberStatus { member_name, status, task_id }` |
| `MemberObserver::on_round(RoundOutcome)` | `MemberRound { member_name, round, outcome }` |
| member shutdown/retire (with `MemberRecord`'s last error/reason when present) | `MemberStopped { member_name, reason }` |
| `TaskCreatedEvent` | `TaskCreated { task_id, title }` |
| `TaskStatusChangedEvent` | `TaskStatus { task_id, status, assignee }` |
| `TeamRuntimeManager::snapshot` initial paint | `snapshot_events` into the stream, or `render_snapshot` for the header view |

## Test Strategy

Inline `#[cfg(test)] mod tests` in each file, offline and deterministic: explicit timestamps, `tempfile` directories, 1 s `tokio::time::timeout` bounds, `futures::poll!` instead of sleeps, `#[allow(clippy::unwrap_used, clippy::expect_used)]` on tests only.
`event.rs` — 13 tests: every variant's wire `kind`/`subject`, the three `from_*` conversions (including `previous`/`assignee` being dropped), snapshot projection order (members by name, then tasks by id), empty snapshot, determinism and non-mutation.
`render.rs` — 16 tests: all eight bodies with the seconds prefix, suffix omission, the three prefix policies, header/no-header snapshots, empty snapshot, UTC-offset shifting and backwards midnight wrap, extreme timestamps/offsets, `state_str`/`round_outcome_str` wire pins, config serde defaults and unknown fields.
`stream.rs` — 17 tests: `EmitOutcome::as_str` + serde names, `BackpressurePolicy` names/default, `StreamSummary` round-trip/default/unknown fields, drop-oldest keeping the newest event, capacity-0 clamping, block-policy refusal then `Pending`-until-drain then `Sent` with FIFO order, `emit_at` timestamps and FIFO through `spawn`, shutdown closing the stream, last-emitter drop closing the queue, clone sharing one queue, temp-file write with `shutdown()` before dropping the emitter, post-shutdown emit leaving the file unchanged, in-memory sink bytes with exactly one flush, write error surfaced as `Error::External`, receiver `len`/`is_empty`/`try_recv` FIFO, `abort()` closing without flushing, and hidden timestamps rendering without a prefix.

## Routing Table

- `./mod.rs` → module doc, the three-layer map, and the flat re-exports of `event`, `render` and `stream`.
- `./event.rs` → `MonitorEvent`, `StampedEvent`, the payload conversions and `snapshot_events`.
- `./render.rs` → `TeamMonitor`, `TeamMonitorConfig`, `TimestampPolicy` and the exact line formats.
- `./stream.rs` → bounded shared queue, `MonitorEmitter`/`MonitorReceiver`, backpressure, writer task and `StatusStreamHandle`.

## Deferred

- Interactive TUI / `/team monitor` CLI rendering — the text stream is the contract and the CLI front-end owns presentation (sibling node).
- DB-backed readers (`TeamInfo`/`MemberInfo`/`TaskInfo`/`MessageInfo` and Python's `TeamMonitor::get_members/get_tasks/get_messages`) — this port queries `TeamStore`, not SQLModel tables, and readers belong to the runtime/CLI nodes.
- Python's `TeamStreamLogger` chunk aggregation (per-`(member, role)` buffering, `answer`/`llm_output` dedup, level routing, 2000-char caps) — needs the `TeamOutputSchema` chunk vocabulary and a file target, deferred with it.
- `hide_dm` / message events — mailbox traffic is not a status line.
- OTel / observability export — owned by the `observability` node.
- Monitor rebinding after a runtime rebuild — the CLI resolves a live client per command instead.
