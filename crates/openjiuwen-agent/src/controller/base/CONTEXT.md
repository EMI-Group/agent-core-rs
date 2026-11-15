# openjiuwen-agent controller base

## Intent

The controller turn loop — bounded event queue, intent recognition, planning, synchronous dispatch, aggregation, and session-persisted task state — plus the lifecycle, task control, reasoner application, and group routing that sit beside it.
Ports the driver half of Python `openjiuwen/core/controller/base.py` (`invoke`, `stream`, `bind_session`, `add_task_executor`, `set_event_handler`, `_ensure_started`) onto one synchronous turn with an additive background scheduler.
The loop also enforces four `ControllerConfig` knobs: the per-session conversation history (`history_limit`), reasoning exposure (`expose_reasoning`), the per-event deadline (`event_timeout_ms`), and the stream's first-frame deadline (`stream_first_frame_timeout_ms`).

## API Surface

- `Controller` (async trait) — `card()`, `config()`, `async invoke(inputs, Option<Arc<dyn Session>>) -> Result<ControllerOutput>`, `stream(inputs, Option<Arc<dyn Session>>) -> BoxStream<'static, Result<ControllerOutputChunk>>`.
- `BaseController` (`Clone`, `Arc<Inner>`) — `new(card, config)`, `with_recognizer(card, config, recognizer)`, `card()`, `config()`, `set_config(config)`.
- Event queue — `push_event(Event) -> bool` (`false` when full; `event_queue_size == 0` = unlimited), `drain_events() -> Vec<Event>`, `pending_events() -> usize`.
- Seams — `set_intent_recognizer`/`intent_recognizer`, `set_event_handler`/`clear_event_handler`, `set_router`/`clear_router`, `set_planner(Planner)`.
- Executors — `add_executor(task_type, executor) -> &Self` (chainable, re-keyed), `add_default_executor(executor) -> &Self`, `remove_executor`, `executor`, `task_types()` (sorted), `registry()` (snapshot), `scheduler()` (snapshot `TaskScheduler` with `max_tasks` and `task_timeout_ms`).
- Task store — `task_manager() -> TaskManager` (shared handle), `async tasks() -> Vec<Task>` (every task in the shared store, all sessions, creation order).
- Lifecycle — `start()`, `stop()`, `async shutdown()`, `is_running()`, `scheduler_loop() -> Option<Arc<TaskSchedulerLoop>>`, `scheduler_state() -> TaskSchedulerState`, `wake_scheduler() -> bool`, `running_tasks() -> Vec<String>`.
- `TaskSchedulerState` — `Stopped`/`Running`/`Stopping` (serde snake_case, `as_str`/`Display`), derived from `is_running()` plus `scheduler_loop()`; `is_running()` keeps its own meaning (the flag).
- Task control — `async pause_task(&str) -> bool`, `async cancel_task(&str) -> bool`, plus `impl TaskControl` (`pause`, `cancel`, `resume`).
- Sessions — `async bind_session(Arc<dyn Session>) -> bool`, `async unbind_session(&str) -> Option<Arc<dyn Session>>`, `bound_sessions() -> Vec<String>`.
- Conversation history — `history(session_id) -> Vec<HistoryEntry>` (the session's log, oldest first, trimmed to the newest `history_limit` messages) and `clear_history(session_id)` (drops that session's log); the log lives per session in `Inner` and is empty for an unknown session.
- Reasoner — `reasoner() -> Reasoner` (the controller installed as `TaskControl`), `async apply_task_ops(&[TaskOp], session_id, Option<&Arc<dyn Session>>) -> Vec<TaskOpOutcome>`.
- Group routing — `set_group(Arc<dyn GroupRouter>)`, `clear_group()`, `group()`, `group_router()` (alias of `group()`), `with_group(router)` (builder-style `set_group`), `async send_to_agent(agent_id, &Event, Option<Arc<dyn Session>>) -> Result<Value>`, `async publish(&Event, Option<Arc<dyn Session>>) -> Result<Vec<Value>>`.
- `EventHandler` (async trait) — `name()`, `async handle(&Event, &TaskContext<'_>) -> Result<EventHandlerOutcome>`.
- `EventHandlerOutcome` — `Intents(Vec<Intent>)`, `Output(Box<ControllerOutput>)`, `Ignored`.
- `TASKS_STATE_KEY` — `"__controller_tasks__"`; the persisted value is a `TaskManagerState` (`{"tasks": [...], "next_seq": n}`) under that one session key.

## Enforced Configuration

Four `ControllerConfig` fields change the loop; the remaining fields keep their Python parity meaning.

- **History recording.** After a turn computed its output, `record_history` appends one `HistoryEntry::user` carrying the turn's first non-empty event text (none when every event text is empty) and one `HistoryEntry::assistant` carrying the non-empty `output.output`, attributed with `.with_task_id` when the output names a task, then trims with `history::trim_history(log, history_limit, turn_start)`.
- **Current-turn layering.** Append-time trimming never drops an entry of the turn being recorded (`turn_start`), while every read inside a turn — `history(session_id)`, the recognizer's `recognize_with_history`, and both dispatched `TaskContext`s — sees the previous turns only, because recording is the last step. Read-time `history::window` re-trims on every read, so a `set_config` with a smaller limit applies to the next read.
- **`history_limit`** counts messages; `0` retains nothing (the append-time trim clears the log and the read-time window is empty).
- **Reasoning exposure.** `dispatch_runs` reads a run's reasoning only when `expose_reasoning` is true: each run with a non-empty reasoning text gets one extra `ControllerOutputChunk::new(index, ControllerOutputPayload::text(OutputKind::Reasoning, text))` directly after its own frame (not `last_chunk`, index from the same counter) and a `"reasoning": text` key in its `data.tasks[]` entry. With the flag off nothing reads reasoning, so no `reasoning` frame and no `reasoning` key reach `frames`, `data`, or the serialized output.
- **`event_timeout_ms`** is **milliseconds**; each drained event's `process_event` runs under `tokio::time::timeout`, and expiry returns `Error::timeout("controller event handling timed out after {ms} ms")` (`ErrorKind::Timeout`), which `Controller::invoke` propagates and `stream_turn` emits as its single `Err` frame. Dropping the timed-out future is the cancellation, so no orphan task and no `tokio::spawn`. `0` means no deadline; outside a tokio runtime the deadline is skipped because the legacy wrapper drives a turn inline with no timer.
- **`stream_first_frame_timeout_ms`** is **milliseconds**; `stream_turn` sends the `processing` acknowledgment first (a sentinel that never consumes the deadline), then awaits the pinned turn future under the deadline. Expiry sends exactly one `Error::timeout("controller stream first frame timed out after {ms} ms")` and returns, dropping (cancelling) the turn and ending the stream. `0` means no deadline, and the setting never bounds `invoke`.

## Constraints

- `invoke` resolves the session through `crate::run::resolve_session` (borrowed when supplied, otherwise a fresh `AgentSession` finished at the end), calls `start()`, binds the session, loads state when `enable_task_persistence`, pushes the inbound event, drains every queued event, runs the turn, saves state, checks the completion sentinel, unbinds the session, and finishes an owned session.
- A turn never returns `Err` for a missing executor, a failing executor, a timeout, or an unknown task target: those become `Failed`/`InputRequired` outputs and task statuses.
- `stream` emits a `processing` chunk first, the turn frames with `last_chunk` cleared and re-indexed, then exactly one terminal `last_chunk` chunk (unless `suppress_completion_signal`); a failed turn emits one `Err` item after the `processing` chunk and no terminal chunk.
- `Inner`/`TurnState` fields stay `pub(super)`; nothing leaks outside `base`.
- `save_state` writes per session and overwrites the snapshot's `next_seq` with `Planner::next_seq()`; `load_state` treats missing, unreadable, or unrestorable state as `None` (warn, clear that session's slice, sequence back to 1).
- Nothing in `base` panics outside tests; the runtime assumptions are `stream`'s spawn and `start`'s loop spawn, and `start` degrades to a logged no-op outside a runtime.
- The history `RwLock` guard is never held across an `.await`: reads clone the window, and `record_history` is a synchronous helper.

## Notes for Agents

### Turn pipeline

- Events are processed in FIFO order; an `EventHandler`, when installed, sees every drained event and can supply intents, short-circuit with an output, or ignore the event (an `Err` from it fails the turn).
- Without a handler, `Input`/`FollowUp` events go through the recognizer (or one default intent when `enable_intent_recognition = false`); the signal kinds (`task_interaction`, `task_completion`, `task_failed`) become a signal output directly.
- Recognized intents are filtered by `ControllerConfig::accepts` (`intent_type_list`), truncated to `max_intents`, and an empty set for an input event falls back to the default intent.
- `UnknownTask` short-circuits the event with a clarification output (status `InputRequired`, `data.clarification_prompt`), discarding any work accumulated for that event.
- Planned intents are routed once per turn through the `TaskRouter` (only with at least two registered executors) and planned by `Planner` into `Submitted` tasks with ids `{session_id}_task_{seq}`.
- `PauseTask`/`CancelTask` intents only flip the stored task's status (`Paused`/`Canceled`) and do not stop a task already running in the background loop; `ResumeTask` sets `Submitted`, claims the task, and re-dispatches it synchronously in the same turn.
- Dispatch is the value-type `TaskScheduler` (sequential, `max_tasks`, `task_timeout_ms`); a task enters the `TaskManager` only after its batch ran, so the background loop never sees a turn's submitted tasks.
- Aggregation precedence is `InputRequired` > `Completed` > `Failed`; completed texts join with `\n\n`, `default_response.text` fills an empty answer, and `data.tasks` lists `{task_id, status, output}` (plus `reasoning` when the config exposes it).
- The last event's produced output wins; a turn with no output falls back to the default response with status `Completed`.
- A turn records its history after it computed its output, so `process_event` (recognizer and event-handler context) and `dispatch_runs` (executor context) observe the previous turns only; a turn that fails — including an event deadline — records nothing.
- The event deadline is skipped when no tokio runtime is present, so the legacy wrapper's inline turn keeps working without a timer; `history_limit = 0` still records nothing and never fails a turn.

### Lifecycle and task control

- `start` flips `running`, reuses a live loop (idempotent) or builds and starts a fresh one from the shared manager, registry, config, card id, and a `SessionProvider` over the bound sessions; outside a runtime it logs and leaves the flag set.
- `stop` clears the flag and fire-and-forgets `TaskSchedulerLoop::stop`; `shutdown` awaits it; `is_running` reports the flag, not the loop.
- Every `invoke` calls `start()`, so a controller that ran once owns a running loop, and every turn ends with `unbind_session`, even for a session the caller bound.
- With a started loop, `pause_task`/`cancel_task` delegate to the loop (executor hooks, token cancellation, awaited driver) and `TaskControl::resume` flips a `Paused` task to `Submitted` and wakes the loop.
- Without a loop, `pause_task` flips `Submitted`/`Working` to `Paused` and refuses anything else; `cancel_task` cancels a `Submitted` task, returns `true` unchanged for a terminal one, and refuses a `Working` one; `resume` behaves as above but has no loop to wake.
- `reasoner()` builds a `Reasoner` from the current config with the controller as its `TaskControl`; `apply_task_ops` takes the planner out of its lock for the batch, applies the ops, writes it back, and wakes the loop when one exists.
- Group routing resolves the owner as the card name (or id when the name is empty) and fails with `Error::config("{owner}: Cannot {method}. Agent is not part of a group with a controller.")`; `send_to_agent` and `publish` forward the event and session unchanged.
- `scheduler_state` reports `Stopped` when the flag is false, `Running` when the flag is set and the loop is live, and `Stopping` when the flag is set without a live loop (a `start` outside a runtime, or a loop winding down); `wake_scheduler` returns `false` without a running loop and `running_tasks` returns the loop's in-flight ids (empty otherwise).
- The per-conversation `TaskQueue` is not owned here: turns dispatch synchronously and track no per-conversation turn, so the queue belongs to the legacy wrapper only.
- The completion sentinel is not produced here: the turn only calls `TaskSchedulerLoop::ensure_completion_signal` when a loop exists.

## Divergences from Python

- Single-turn, in-process loop with a `VecDeque` drained by the turn that enqueued into it; no asyncio queue or pub/sub.
- Tasks are persisted only after they ran, and the turn's `max_tasks` budget is enforced by the scheduler, not by the manager.
- Control intents are manager-only status flips and bypass the loop's executor hooks.
- `tasks()` is unscoped (whole shared store), unlike Python's session-scoped views.
- Persistence is one session key holding the manager snapshot plus the planner sequence; the ordering key `created_seq` is never on the wire.
- `stream` is the only path that writes turn frames to the session stream; `invoke` returns them on the output.

## Test Strategy

- 42 offline tests: `tests.rs` 32, `state.rs` 5, `mod.rs` 5.
- `tests.rs` covers router selection and the single-executor shortcut, unregistered task types with the default response, stream shape, the three handler outcomes, signal events, `max_intents`, the started loop running a manager-inserted task, start/stop idempotency, shutdown of an in-flight run, manager-only pause/cancel fallbacks, loop-delegated pause/cancel/resume, group routing with typed errors, the `with_group`/`group_router` round-trip, reasoner create → loop execution, and the sentinel written once per idle stretch.
- The enforcement tests cover history recording shape (user then assistant, task id on the assistant entry), the append-time window, recognizer/executor history parity with the current turn excluded, `history_limit == 0` retaining nothing, `clear_history`, read-time re-trim after `set_config`, the reasoning frame and `data.tasks[].reasoning` (and their absence with the flag off), the event deadline cancelling a stalled run and surfacing as the stream's single `Err` frame (`0` completing), and the first-frame deadline ending a slow stream while `invoke` stays unbounded (`0` streaming the slow turn).
- `state.rs` covers persistence across turns and instances, pause/cancel status persistence, corrupt-state recovery, disabling persistence, and the exact snapshot wire shape.
- `mod.rs` covers the state key, lifecycle/queue behaviour (binding, executor registry, queue capacity and FIFO drain), the `TaskSchedulerState` wire names, the `Stopped → Running → Stopped` transitions around `start`/`stop`, `Stopping` outside a runtime, and `wake_scheduler`/`running_tasks` with and without a loop.
- `fixtures.rs` supplies `StubExecutor`, `CooperativeExecutor` (pause/cancel aware, token-driven), `ContextExecutor` (records `TaskContext::history` and `expose_reasoning`, optional reasoning), `StallingExecutor` (`started`/`finished` counters that prove cancellation), `StubHandler`, `CountingRecognizer`, `RecordingRecognizer` (records each `recognize_with_history` slice), `RecordingSession`, the `bounded` 10 s await guard, and the session/controller/control-turn helpers; `ScriptedModel` is used only by the router test.

## Routing Table

- `./mod.rs` → `Controller`, `BaseController`, `EventHandler`, accessors, lifecycle, queue tests
- `./turn.rs` → `impl Controller`, event processing, intent routing, dispatch, aggregation, streaming, sentinel call
- `./state.rs` → `save_state`/`load_state`/`tasks`, the `__controller_tasks__` codec
- `./tests.rs` → routing, persistence, streaming, handler, lifecycle, control, reasoner, group, sentinel tests
- `./fixtures.rs` → shared test doubles
