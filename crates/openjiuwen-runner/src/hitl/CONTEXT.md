# hitl/ — runner-level human-in-the-loop hub

## Intent
Brokers a host UI against the runs a runner is executing: the live-run registry, a bounded inbox of inbound human payloads per run, answering the interaction requests a run's session published, pause/resume/cancel coordination signals, and best-effort mirroring into a `RunMonitor`.
Python correspondence: `team_runner.py`'s `interact_agent_team` (payload delivery to an active team runtime, returning a `DeliverResult`), `register_human_agent_inbound` (the team→user channel the `InteractKind` variants model), and `pause_agent_team` / `stop_agent_team`.
`InteractPayload` is the Rust union of `GodViewMessage` / `OperatorMessage` / `HumanAgentMessage` / `GroupChatMessage`, plus `InteractKind::Text` for Python's bare-string shorthand.
Pause and resume are status/coordination signals: they flip the hub's state and gate and mirror the monitor, but they do not suspend an agent or team loop.
The hub is in-process only: no MQ/websocket/Pulsar transport, no reply-topic routing, no authentication of who may answer.

## API Surface
```rust
// payload.rs — serde snake_case, tolerant of unknown/missing fields
pub enum InteractKind { GodView, Operator, HumanAgent, GroupChat, Text }  // Copy+Eq+Hash, Default = Text
pub struct InteractPayload { pub kind: InteractKind, pub body: Value, pub text: Option<String>,
                             pub sender: Option<String>, pub correlation_id: Option<String> } // + Default
impl InteractPayload { pub fn new(kind: InteractKind) -> Self; pub fn text(text: impl Into<String>) -> Self;
                       pub fn body(kind: InteractKind, body: Value) -> Self;
                       pub fn with_sender(..) -> Self; pub fn with_correlation(..) -> Self; pub fn with_text(..) -> Self }

// target.rs
pub struct RunTarget { pub team_name: Option<String>, pub session_id: Option<String> }  // + Default
impl RunTarget { pub fn new() -> Self; pub fn for_team(name: impl Into<String>) -> Self;
                 pub fn for_session(id: impl Into<String>) -> Self; pub fn matches(&self, info: &ActiveRunInfo) -> bool }
pub enum DeliverResult { Accepted, MissingTarget, NoInbox, Closed }   // Copy+Eq+Debug
impl DeliverResult { pub fn success(&self) -> bool; }                 // only Accepted

// bridge.rs — public run spec/info/handle
pub struct ActiveRunSpec { pub run_id: Option<String>, pub label: String, pub team_name: Option<String>,
                           pub session_id: Option<String>, pub cancellation: Option<CancellationToken>,
                           pub detail: Value }
impl ActiveRunSpec { pub fn new(label: impl Into<String>) -> Self;
                     pub(crate) fn for_execution(run_id: &str, label: &str, team_name: Option<&str>,
                                                 session_id: Option<&str>, cancellation: &CancellationToken,
                                                 detail: Value) -> Self }
pub struct ActiveRunInfo { pub run_id: String, pub label: String, pub team_name: Option<String>,
                           pub session_id: Option<String>, pub state: RunPhase, pub gate_closed: bool,
                           pub queued: usize }                            // + Default, serde snake_case
pub struct RunHandle { .. }                                              // Clone+Debug
impl RunHandle { pub fn run_id(&self) -> &str; pub fn cancellation(&self) -> CancellationToken; }

// hub.rs — Clone is cheap (one Arc table) and copies the monitor setting; Default == new()
pub const INBOX_CAPACITY: usize = 64;
pub struct InteractionHub { .. }
impl InteractionHub {
    pub fn new() -> Self;
    pub fn attach_monitor(&self, monitor: Option<Arc<RunMonitor>>) -> Option<Arc<RunMonitor>>;
    pub fn with_monitor(self, monitor: Arc<RunMonitor>) -> Self;
    pub fn monitor(&self) -> Option<Arc<RunMonitor>>;
    pub fn register(&self, spec: ActiveRunSpec) -> RunHandle;
    pub fn unregister(&self, run_id: &str) -> Result<()>;
    pub fn attach_session(&self, run_id: &str, session: Arc<dyn Session>) -> Result<()>;
    pub fn deliver(&self, target: RunTarget, payload: InteractPayload) -> DeliverResult;
    pub fn take_inbox(&self, run_id: &str) -> Option<tokio::sync::mpsc::Receiver<InteractPayload>>;
    pub async fn answer(&self, response: InteractionResponse) -> Result<()>;
    pub fn pending(&self, run_id: &str) -> Vec<String>;
    pub fn pending_all(&self) -> Vec<(String, Vec<String>)>;
    pub fn pause(&self, run_id: &str) -> Result<()>;
    pub fn resume(&self, run_id: &str) -> Result<()>;
    pub fn cancel(&self, run_id: &str) -> Result<()>;                 // synchronous transition
    pub fn list_active(&self) -> Vec<ActiveRunInfo>;                  // sorted by run_id
    pub async fn close_interactions(&self, reason: impl Into<String>) -> Result<()>;
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
}

// crate-internal bridge the run paths adopt next (not public)
pub(crate) struct RunBinding { .. }                                   // Drop unregisters
impl RunBinding { pub(crate) fn run_id(&self) -> &str; pub(crate) fn unregister(&self); }  // idempotent, never panics
pub(crate) fn bind_run(hub: Option<&Arc<InteractionHub>>, spec: ActiveRunSpec,
                       session: Option<Arc<dyn Session>>) -> RunBinding;
pub(crate) use bridge::{bind_run, RunBinding};                        // re-exported at `openjiuwen_runner::hitl`
```
`bind_run(None, ..)` yields an inert binding whose `run_id()` is the spec's id or a fresh UUID v4.
The crate root re-exports these items from `openjiuwen_runner::hitl` (added by the crate facade, not by this module).

## Semantics
- `register` mints a UUID v4 when the spec's id is absent or empty, stores label/team/session/detail, starts in `RunPhase::Queued` with an open gate, keeps the spec's token (or a fresh one), and creates a `mpsc::channel(INBOX_CAPACITY)` inbox.
- Registration is not deduplicated: a second `register` under a live id replaces the entry and cancels the previous watcher, so a re-run under the same id starts clean.
- With a monitor attached, `register` calls `monitor.track` with `RunKind::Team` when the spec names a team and `RunKind::Agent` otherwise; `track` is idempotent, so a run path that already began the status is not duplicated.
- `unregister` on an unknown id is `Error::NotFound("run '<id>' is not registered")`; otherwise it drops the inbox, session, and pending set, cancels the watcher token, removes the entry, and finalises a still non-terminal monitor run as `RunOutcome::completed()`.
- `attach_session` on an unknown id is the same `NotFound`; it subscribes to the session stream before spawning the watcher (so an immediately published frame is buffered), and replacing a session cancels the previous watcher.
- Outside a tokio runtime `attach_session` attaches the session without a watcher and logs a warning instead of panicking.
- The watcher records the request id of every `__interaction__` `StreamEvent::Item` (payload `id`, falling back to `value.id`, skipping null/empty), ends on its token, on end of stream, or on `StreamEvent::End`, logs stream errors at debug, and holds only a `Weak` registry reference.
- `deliver` visits runs in sorted run-id order; both target fields unset (or no match) is `MissingTarget`, a run whose sender is gone is `NoInbox`, and a closed or full inbox is `Closed` (a closed inbox also drops the dead sender, so later deliveries report `NoInbox`).
- `take_inbox` hands out the receiver once per run; a second call and an unknown run return `None`.
- `answer` resolves an indexed pending id through its run's session (propagating the session's error verbatim and leaving the id pending), otherwise tries every attached session in sorted run-id order and returns `Error::NotFound("interaction '<id>' is not pending on any active run")` when none accepts it.
- `pause`: `Paused` is idempotent, `Cancelled` is `Error::Validation("run '<id>' has already been cancelled")`, and any live state becomes `Paused` with a closed gate.
- `resume`: `Running` is idempotent, `Cancelled` is the same validation, and `Paused`/`Queued` become `Running` with an open gate.
- `cancel`: already `Cancelled` is idempotent; otherwise it trips the run's token, resolves every pending interaction with `InteractionResponse::cancel` carrying `"run cancelled"`, clears the pending set, sets `Cancelled` with a closed gate, and mirrors `RunPhase::Cancelled`.
- `cancel` is synchronous: the cancellation responses are delivered on a detached task, so a waiter observes its `Cancel` decision asynchronously (outside a runtime nothing is delivered and a warning is logged).
- `close_interactions` resolves every pending id with `InteractionResponse::cancel` carrying the given reason, clears the pending sets first, logs per-session failures, and returns `Ok(())` (also for an empty hub).
- `pending` returns sorted ids (unknown run: empty) and `pending_all` returns every registered run's sorted ids sorted by run id, including runs with no pending ids.
- `list_active` returns every registered run sorted by run id; a cancelled run stays listed until it is unregistered.
- `ActiveRunInfo::queued` counts the pending interaction ids the hub observed, not the inbox depth.
- Monitor mirroring is best-effort: a rejected transition (a run already finished or evicted) is logged with `tracing::warn!` (target `openjiuwen_runner::hitl`) and never fails a hub operation.
- `attach_monitor` swaps the mirrored monitor at runtime and returns the previous one; it is what `Runner::set_monitor` uses to re-point the installed hub, and `with_monitor` is its builder form.
- The monitor sits behind a `parking_lot::RwLock` read for the length of one mirror call; a clone copies the setting it was made with, so re-pointing the original never retroacts onto an earlier clone.

## Constraints
- In-process only: one `Arc<parking_lot::Mutex<BTreeMap<run_id, Entry>>>` per hub, shared by clones; nothing is persisted or transported.
- No lock is held across `.await`: `answer` and `close_interactions` clone the `Arc<dyn Session>` and the ids, drop the guard, and only then await; `register`/`deliver`/`take_inbox`/`pending`/`list_active`/`pause`/`resume`/`cancel` are fully synchronous.
- The watcher is owned by its run: cancelled on unregister and on session replacement, and it never keeps the hub alive (it holds a `Weak`).
- The inbox is bounded (`INBOX_CAPACITY`); a full inbox is reported as `Closed`, never silently dropped.
- Public data types are serde (`snake_case`, `default`, tolerant of unknown fields); `ActiveRunSpec` is not serde because it holds a `CancellationToken`.
- Every public item and field is documented; workspace lints include `missing_docs`, `unreachable_pub`, `unsafe_code = deny`, clippy `all`+`pedantic`+`unwrap_used`+`expect_used`+`needless_pass_by_value`.
- The agent, workflow, and team run paths and the team pool call `bind_run`, build specs with `ActiveRunSpec::for_execution`, and hold the returned `RunBinding`; the item-level `#[allow(dead_code)]` attributes stay only because `RunBinding::run_id` is unreferenced outside this module's tests.
- Files stay around ≤ 600 lines: `mod.rs` (docs, re-exports, hub-wide constants, error wording, tests), `hub.rs` (registry, inbox, answering), `bridge.rs` (spec/info/handle/binding plus session watcher and cancellation delivery), `payload.rs`, `target.rs`.

## Routing Table
- `./mod.rs` → module docs, re-exports (public items and the crate-internal bridge), `TARGET`/`CANCEL_REASON`, hub error wording (`not_registered`, `already_cancelled`, `already_finished`), and the hub/payload/target/bridge unit tests
- `./hub.rs` → `InteractionHub`, `INBOX_CAPACITY`, the internal `Entry`/`Inner` table, delivery, answering, pause/resume/cancel, listings
- `./bridge.rs` → `ActiveRunSpec`, `ActiveRunInfo`, `RunHandle`, `RunBinding`, `bind_run`, `watch_pending`, `spawn_cancellations`
- `./payload.rs` → `InteractKind`, `InteractPayload` and their builders
- `./target.rs` → `RunTarget`, `DeliverResult`
- `../run/` → the future consumers: run paths build `ActiveRunSpec::for_execution`, call `bind_run`, take the inbox, and observe the run's `CancellationToken`
- `../monitor/` → `RunMonitor`, `RunPhase`, `RunStart`, `RunOutcome`, `RunKind` mirrored by the hub
- `../runner.rs` → the facade the hub hangs off once the run paths are wired

## Notes & Deferrals
- No MQ/websocket/Pulsar transport between processes, no reply-topic routing, no `RemoteAgent` detection, and no A2A adapter for interactions.
- No `InteractGate` enforcement inside the agent/team loop: pause/resume only flip hub state and the gate flag.
- No OS-process members, no team-member spawning, no worktree or SSH transport, and no team pools/monitors over the hub.
- No authentication or ACL on who may answer an interaction or deliver a payload.
- Hub state is in memory and lost when the hub is dropped; nothing survives a process restart.
- A synchronous `cancel` cannot await the pending sessions, so cancellation responses are delivered on a detached task rather than before `cancel` returns.
- Python quirks deliberately not reproduced: `interact_agent_team` requiring both `team_name` and `session_id` (a Rust target may set either), exceptions for an unknown team/session (Rust reports `DeliverResult::MissingTarget`), and `register_human_agent_inbound` raising `KeyError` for an unknown member.
