# openjiuwen-agent controller

## Intent

Rust counterpart of the Python package `openjiuwen/core/controller/`: an event feeds intent recognition, intents become tasks in a shared task store, and each task is dispatched to the executor registered for its task type.
The synchronous turn (`BaseController::invoke`) drains the bounded event queue, recognizes, plans, dispatches and aggregates one `ControllerOutput`.
The background half (`executors::background::TaskSchedulerLoop`) runs the tasks no turn dispatches: it scans the store, admits them under a concurrency gate, bounds them with a timeout, honours pause/cancel, and reports completion with a sentinel chunk.
The task-operation layer (`reasoner`) turns one intent into `TaskOp`s (create/pause/resume/cancel/continue/supplement/modify) and applies them to the store behind a confidence gate.
`group::GroupRouter` is the agent-group seam behind point-to-point sends and publishes; `queue::TaskQueue` tracks at most one running task per conversation for the legacy wrapper's preemption.
Two Python generations live here without sharing types: the current event-queue pipeline (`schema`, `config`, `intent`, `executors`, `task_manager`, `base`, `reasoner`, `group`, `queue`) and the superseded string-routed lineage (`legacy`), whose types carry a `Legacy` prefix.
The turn also keeps a per-session conversation history (`history.rs`) bounded by `ControllerConfig::history_limit` and hands it to the intent recognizer and the dispatched executors.
The four behavioural knobs `history_limit`, `expose_reasoning`, `event_timeout_ms` and `stream_first_frame_timeout_ms` are enforced by the turn loop and the background scheduler; see *Config Enforcement*.

## API Surface

### Turn loop — `base/`

- `Controller` — `card`, `invoke(inputs, session) -> Result<ControllerOutput>`, `stream(inputs, session) -> BoxStream<'static, Result<ControllerOutputChunk>>`.
- `BaseController` (holds an `Arc<Inner>`, is `Clone`) — `new(card, config)`, `with_recognizer`, `card`/`config`/`set_config`, `push_event`/`drain_events`/`pending_events`, `set_intent_recognizer`/`intent_recognizer`, `set_event_handler`/`clear_event_handler`, `set_router`/`clear_router`, `set_planner`, `add_executor`/`add_default_executor`/`remove_executor`/`executor`/`task_types`/`registry`/`scheduler`, `task_manager`, `start`/`stop`/`shutdown`/`is_running`/`scheduler_loop`, `scheduler_state`/`wake_scheduler`/`running_tasks`, `pause_task`/`cancel_task`, `bind_session`/`unbind_session`/`bound_sessions`, `save_state`/`load_state`, `reasoner`/`apply_task_ops`, `set_group`/`clear_group`/`group`/`group_router`/`with_group`/`send_to_agent`/`publish`, plus `impl TaskControl` (a resume flips `Paused` → `Submitted` and wakes the loop).
- `TaskSchedulerState` — the host-facing scheduler view: `Stopped` (the running flag is false), `Running` (flag set and a live `TaskSchedulerLoop` backs it), `Stopping` (flag set, no live loop: started outside a runtime or winding down); `as_str()`/`Display` yield the snake_case wire names and the type is serde-serializable.
- `EventHandler` / `EventHandlerOutcome` — `Intents(Vec<Intent>)`, `Output(Box<ControllerOutput>)`, `Ignored`; a handler replaces the built-in recognizer for an event and is what the `Reasoner`-driven path installs on.
- `TASKS_STATE_KEY` — `"__controller_tasks__"`; the session state key holds `{"tasks": [...], "next_seq": n}` (a `TaskManagerState`).
- The turn pipeline: drain the queue → handler/recognizer → `Planner` → registry lookup (task type from the intent, else the router's choice, else `default_task_type`) → `TaskScheduler` (sequential value-type dispatch) → `TaskManager::add_tasks` for the tasks that actually ran → aggregate.
- Scheduler facade — `scheduler_state()` derives `Stopped`/`Running`/`Stopping` from `is_running()` plus `scheduler_loop()`, `wake_scheduler()` wakes a live loop and returns `false` without one, and `running_tasks()` lists the loop's in-flight task ids (empty without a loop).
- The scheduling knobs are set through `ControllerConfig` builders: `with_max_concurrent_tasks`, `with_schedule_interval`, `with_task_timeout`; dependency gating has no config field because the loop reads `metadata["depends_on"]` per task.
- The per-conversation `TaskQueue` is legacy-scoped: `BaseController` dispatches each turn synchronously and tracks no per-conversation turn, so the queue (and `has_running`/`running_conversations`/`interrupt`) lives only on `IntentDetectionController`.
- Global aggregate precedence is `InputRequired` > `Completed` > `Failed`; completed texts are joined by `\n\n`, `default_response.text` fills an empty aggregate, and `data` is `{"tasks": [{"task_id", "status", "output"}...]}`.
- `stream` emits one `processing` chunk, then the turn's frames, then one terminal `last_chunk` chunk unless `suppress_completion_signal` is set; a preempted/aborted turn is `Error::Cancelled`.
- `history(session_id) -> Vec<HistoryEntry>` returns the session's stored log trimmed with the current `history_limit`, and `clear_history(session_id)` drops it; the store is a per-session `RwLock<BTreeMap<String, Vec<HistoryEntry>>>` on `Inner`.
- `process_event` calls `IntentRecognizer::recognize_with_history(event, &history)` and builds both turn `TaskContext`s with `.with_history(..)`/`.with_expose_reasoning(..)`, so the recognizer, the event handler and every executor of the turn see the same trimmed window.
- `dispatch_runs` adds one `OutputKind::Reasoning` frame per run whose `TaskRun::reasoning()` is non-empty, plus a `reasoning` key in that run's `data.tasks[]` entry, and does both only when `expose_reasoning` is on.

### Conversation history — `history.rs`

- `HistoryRole { User, Assistant }` (wire `"user"`/`"assistant"`, `as_str`, `Display`) and `HistoryEntry { role, text, task_id: Option<String> }` (serde snake_case, unknown fields tolerated) with `HistoryEntry::user(text)`, `assistant(text)` and `with_task_id(..)`.
- `trim_history(&mut Vec<HistoryEntry>, limit, turn_start)` runs on append: it removes only the oldest prefix and never an entry at index `>= turn_start`, so a turn that alone produces more messages than the limit keeps its own entries; `limit == 0` clears the log.
- `window(&[HistoryEntry], limit) -> &[HistoryEntry]` is the read-time trim: the newest `limit` entries (`0` → empty), so lowering `history_limit` takes effect on the very next read.
- The limit counts **messages**, not turns, matching Python's message-indexed `chat_history[-2 * max_turns:]` slice.
- Recording is one `user` entry per turn (its first non-empty event text) plus one `assistant` entry for a non-empty aggregated output, written after the turn's dispatch; the current turn is therefore never part of what its own recognizer or executors observe.

### Task store — `task_manager/`

- `TaskManager` (a synchronous, thread-safe handle over `Arc<Mutex<..>>`; every getter returns owned deep copies) — `new`/`with_config`/`config`/`set_config`, `set_on_task_submitted`/`clear_on_task_submitted` (the scheduler's wake hook), `add_task`/`add_tasks`/`get_task`/`task`/`tasks`/`session_tasks`, `update_task`/`update_tasks`, `remove_task`/`pop_task`, `get_child_task`, `claim_task`, `update_task_status(id, status, with_children, recursive, error_message)`, `set_priority`, `len`/`is_empty`/`has_active_tasks`/`session_has_active_tasks`/`next_created_seq`/`clear`, `snapshot_session`/`restore_session`.
- `TaskFilter` — `by_id`/`by_ids`/`by_session`/`by_user`/`by_status`/`by_priority`/`highest_priority`/`roots`/`include_children`, `validate()` requiring one of task id/session/user/priority/status/is_root (`"At least one filter parameter (task_id, session_id, user_id, priority, status, or is_root) must be provided"`).
- `TaskIdSelector::{One, Many}`; `PrioritySelector::{Level(i32), Highest}` (`From<i32>`, `level`, `is_highest`, numeric strings accepted); `TaskManagerState::new(tasks, next_seq)`.
- Resolution: primary filters (id/session/priority/root) union, then `status`/`user_id` (against `metadata["user_id"]`) narrow, then `with_children` expands transitively to descendants.
- Mutations: `add_task` rejects a duplicate id (`Error::validation`) and assigns `created_seq` when it is `0`; `update_task`/`update_task_status` on an unknown id return `false`/`()`; `claim_task` is one atomic `Submitted` → `Working` flip (the scheduler's double-execution guard); `remove_task`/`pop_task` drop the matched subtree and promote surviving orphans to roots; `update_task_status(.., recursive = true)` walks the subtree; `Failed` defaults `error_message` to `"Task execution failed"`.
- Persistence: `snapshot_session(session_id)` / `restore_session(...)` round-trip one session's tasks plus `next_seq`; `BaseController::save_state` writes `{TASKS_STATE_KEY: TaskManagerState}` for the bound session and `load_state` restores tolerantly (missing or corrupt state clears that session and never errors).

### Background scheduler — `executors/background.rs`

- `TaskSchedulerLoop::new(manager, Arc<RwLock<TaskExecutorRegistry>>, Arc<RwLock<ControllerConfig>>, controller_id, SessionProvider)`; `start(self: &Arc<Self>) -> bool` (idempotent), `is_running`, `stop`, `wake`, `in_flight`, `scan_once`, `wait_until_idle`, `pause_task`, `cancel_task`, `ensure_completion_signal`.
- `SessionProvider = Arc<dyn Fn(&str) -> Option<Arc<dyn Session>> + Send + Sync>` resolves a driving session per task session id; `TaskContext::cancellation` hands the executor a `CancellationToken`.
- Scan: dependency gate first (`metadata["depends_on"]` — a `Waiting` task whose dependencies are all `Completed` is promoted to `Submitted`), then `Submitted` tasks in priority-descending order with `created_seq`/`task_id` tiebreaks; already-running tasks are skipped; admission respects `max_concurrent_tasks` (with a value of `0` meaning unlimited).
- Driving one task: `claim_task`, run the executor under `task_timeout_ms` (timeout → `Failed` with `"Task timeout after {t} seconds"`), swallow cancellation, write `Completed`/`InputRequired`/`Failed`, untrack, `wake()` the next scan.
- The driver context carries the controller's `expose_reasoning` flag (its history stays empty: a scheduled run has no dispatching turn); after the status write, `record_outputs` appends one `OutputKind::Reasoning` chunk before the `task_completion`/`task_interaction`/`task_failed` chunk when the flag is on and the executor reported reasoning, and appends nothing else.
- Waiting: `tokio::sync::Notify` woken on submit and on every completion, capped by `schedule_interval_ms` so a lost wakeup cannot stall the loop; loop errors are logged and followed by a short sleep, never killing the loop.
- Completion sentinel: once a session has no `Submitted`/`Working` task, `ALL_TASKS_PROCESSED` (`"all_tasks_processed"`) carries `frames_text("All tasks have been successfully processed")` with `last_chunk: true`, written once per idle stretch and suppressed by `ControllerConfig::suppress_completion_signal`.
- Two-phase pause/cancel: a `Submitted` cancel short-circuits to `Canceled`; terminal states are idempotent `true`; otherwise the executor is asked outside the lock, the running entry is re-checked, the driver token is cancelled and the driver awaited, and only then is the status written (`Paused`/`Canceled`). A resume flips `Paused` → `Submitted` so the next scan picks it up.

### Executors — `executors/mod.rs`, `executors/builtin.rs`

- `TaskExecutor` — `task_type`, `description`, `async execute(&Task, &TaskContext<'_>) -> Result<TaskResult>`, plus four defaulted hooks: `can_pause() -> (bool, String)` (`(false, "task executor does not support pausing")`), `pause() -> bool` (`false`), `can_cancel() -> (bool, String)` (`(true, "")`), `cancel() -> bool` (`false`).
- `TaskExecutorRegistry`, `TaskScheduler` (sequential value-type dispatch with `max_tasks` and an optional timeout), `TaskRun { status, text, frames }` (`reasoning()` reads the successful result's `TaskResult::reasoning`), `TaskContext { session, controller_id, cancellation, history, expose_reasoning }` (`with_cancellation`, `with_history`, `with_expose_reasoning`; a fresh context carries an empty history and `expose_reasoning == false`, which is what background runs get), `TaskRouter::select(query, intents)`.
- Built-ins: `ControllerAgent` (task type `agent`), `WorkflowController` (task type `workflow`, `map_output`), `ExecutorInfo`, `LLMController` (the router; one registered executor is chosen without a model call), constants `AGENT_TASK_TYPE`/`WORKFLOW_TASK_TYPE`/`LLM_TASK_TYPE`.
- `ControllerAgent` invokes the wrapped agent when `TaskContext::expose_reasoning` is off and streams it when it is on, folding every `llm_reasoning` chunk's `content` into `TaskResult::reasoning` and taking the run's output from the final `answer` chunk through `AgentOutput::from_value`; interrupts still map onto `input_required` because that chunk carries the whole interrupt output. `WorkflowController` and `LLMController` do not capture reasoning.

### Reasoner — `reasoner/`

- `TaskOp` — `Create { intent, task_type }`, `Pause { task_id }`, `Resume { task_id }`, `Cancel { task_id }`, `Continue { depend_task_id, event, metadata, description }`, `Supplement { task_id, info }`, `Modify { task_id, details }`, `Unknown { prompt }`; `kind()` and `task_id()` describe it.
- `TaskOpOutcome { kind, task_id, status, applied, message, clarification }`; `TaskControl` (`async pause/cancel/resume(&str) -> bool`) is the seam the controller implements so the reasoner can drive the scheduler.
- `Reasoner` — `new(config, Option<Arc<dyn TaskControl>>)`, `config`/`set_config`, `control`/`set_control`, `plan_ops(&Intent, session_id) -> Vec<TaskOp>`, `reason(intents, session, ..)`, `apply_task_ops(ops, session_id, session)`.
- `plan_ops` maps Python's task-op intents onto ops: create (task type from the intent, else `default_task_type`), pause/cancel/resume by target, a switch-task intent into a cancel plus a create, and a control intent with an unknown target into `Unknown`; an op at or below `intent_confidence_threshold` degrades to `Unknown`.
- `reason` records a clarification prompt in the session (`CLARIFICATION_STATE_KEY = "__controller_clarification__"`, default `"Please clarify your request."`) and emits a `TaskInteraction` frame; `apply_task_ops` is non-aborting (one outcome per op, `applied: false` + message on failure).
- Create mints the task through the `Planner` with `task_type`, `priority = 1`, `status = Submitted`, `context_id = "<session>_<task_id>"`, the intent's events as inputs and its metadata; pause/cancel/resume go through `TaskControl`; continue gathers the dependency task's inputs and context; supplement pauses and appends `"\n\n任务补充信息:\n{info}"` to the description, then resubmits; modify cancels, replaces and resubmits.

### Group routing — `group.rs`

- `GroupRouter` — `group_id`, `async send_to_agent(agent_id, &Event, Option<Arc<dyn Session>>) -> Result<Value>`, `async publish(&Event, Option<Arc<dyn Session>>) -> Result<Vec<Value>>`.
- `GroupMethod::{SendToAgent(&str), Publish}` (`method_name`, `agent_id`, `Display`); `NOT_IN_GROUP = "Agent is not part of a group with a controller."`.
- `BaseController::send_to_agent`/`publish` return `group::send_without_group`/`publish_without_group` (`Error::Config` naming the method and, for a send, the agent) when no router is installed; `require_group` is the shared guard.
- `BaseController::group_router()` reads the same injected router as `group()`, and `with_group(router)` is the builder-style `set_group` so a controller can be wired in one expression.

### Per-conversation queue — `queue.rs`

- `TaskQueue` — `register(conversation_id, Task, JoinHandle<()>, CancellationToken, target_id) -> RunningTask`, `async cancel_running(conversation_id) -> bool`, `unregister`, `find`, `has_running`, `running_conversations`, `len`, `is_empty`.
- `RunningTask` is a detachable snapshot (`Clone` deliberately does not clone the join handle) carrying `conversation_id`, `task`, `target_id`, `started_at` (monotonic), `token`, `is_finished`.
- `cancel_running` cancels the token and awaits the handle, mapping a cancelled `JoinError` to success and leaving the entry for its owner to unregister.
- The queue is legacy-scoped: only `IntentDetectionController::task_queue()` (plus `has_running`/`running_conversations`/`interrupt`) exposes it, because `BaseController` dispatches synchronously and drives no per-conversation turn; `BaseController::running_tasks()` reports the background loop's in-flight runs instead.

### Legacy lineage — `legacy/`

- `IntentDetectionController` wraps `BaseController`: `intent_detection`, `add_executor`, `set_router`, `set_event_handler`, the inherited controller surface, plus `task_queue`/`has_running`/`running_conversations`/`interrupt` and `set_group`/`clear_group`/`group`/`send_to_agent`/`publish`.
- Preemption: the conversation id comes from `conversation_id`/`conversation` in the payload, then the session id, then `"default"`; a new message aborts the previous turn's handler (fire-and-forget), cancels the tracked run, then enqueues itself.
- `IntentDetector` / `LegacyPlanner` — tolerant `{"result": n}` category classification and the one-undefined-task planner stub; `legacy::reasoner` holds their helpers.
- `legacy::schema` — `LegacyTask`, `LegacyTaskStatus`, `TaskInput`, `LegacyTaskResult`, `TaskDependency`, `DependencyType`, `LegacyEvent` (with `receiver_id`/`custom_event_type`, both skipped when unset, plus `with_receiver`/`with_custom_event_type`/`handoff_to_agent`/`routed_to`), `LegacyEventType`, `EventPriority`, `SourceType`, `EventSource`, `EventContent`, `EventContext`.
- `IntentDetectionController` falls back to inline known-intent handling when no tokio runtime is available (no preemption handles to abort).

### Schema and re-exports

- `schema` — `DataFrame` family (`TextDataFrame`, `FileDataFrame`, `JsonDataFrame`), `frames_text`, `Event`/`EventType`, `Intent`/`IntentType`, `Task`/`TaskStatus`/`TaskResult` (`TaskResult::reasoning` + `with_reasoning`), `ControllerOutputPayload`/`ControllerOutputChunk`/`ControllerOutput`/`OutputKind` (`task_completion`, `task_interaction`, `task_failed`, `reasoning`, `processing`, `all_tasks_processed`), constants `ALL_TASKS_PROCESSED`/`CONTROLLER_OUTPUT_TYPE`/`TASK_PROCESSING`.
- `mod.rs` re-exports the controller-level names of every module; `schema` and `legacy::schema` keep their own namespaces.

## Config Enforcement

Each enforced knob and its unit; every default stays Python-shaped (`history_limit` 50, `expose_reasoning` `false`, `event_timeout_ms` 300_000, `stream_first_frame_timeout_ms` 30_000).

- `history_limit` (messages, default 50, `0` = retain nothing): one `user` entry per turn (its first non-empty event text) and one `assistant` entry for a non-empty aggregated output (carrying the output's `task_id` when it names one) are recorded after the dispatch; the append trim drops only the oldest prefix beyond the limit and never the turn being recorded, reads apply `window` with the current limit, and the intent recognizer plus every dispatched `TaskContext` see that trimmed window with the current turn excluded.
- `expose_reasoning` (bool, default `false`, Rust extension): when on, `dispatch_runs` emits one `OutputKind::Reasoning` frame per run that reports reasoning plus a `reasoning` key in `data.tasks[]`, the background scheduler appends the same chunk to the stored task, and `ControllerAgent` captures the wrapped agent's `llm_reasoning` stream into `TaskResult::reasoning`; when off, reasoning is dropped from frames and `data` everywhere and every executor sees `TaskContext::expose_reasoning == false`.
- `event_timeout_ms` (milliseconds, default 300_000, `0` = no deadline): each drained event's handling in `invoke_turn` runs under `tokio::time::timeout`; expiry returns `Error::timeout("controller event handling timed out after <n> ms")` (`ErrorKind::Timeout`) to `invoke` and as the `Err` frame of `stream`, and dropping the future cancels the stalled work. No deadline is applied when no tokio runtime is available (the legacy inline fallback).
- `stream_first_frame_timeout_ms` (milliseconds, default 30_000, `0` = no deadline): `stream_turn` sends the local `processing` acknowledgment first (a sentinel, not a turn frame, so it never consumes the deadline), then bounds the wait for the turn's first output frame; expiry sends exactly one `Err(Error::timeout("controller stream first frame timed out after <n> ms"))`, drops the turn (cancelling it) and ends the stream, so no frame follows and the next poll is `None`. It never bounds `invoke`, and `legacy::IntentDetectionController::stream` inherits it because it delegates to `BaseController::stream`.

## Constraints

- Every public type is serde-serializable with Python wire names (`mimeType`, `input-required`, `controller_output`, `user_input`, `single_agent`) and tolerates unknown fields; serde defaults match the runtime defaults.
- Errors are `openjiuwen_common::error::{Error, Result}`; a missing executor, an executor failure, a timeout, or a task-control intent with an unknown target becomes a failed task/output or an `applied: false` outcome, never `Err`.
- `invoke` is single-turn and in-process; the background loop is additive and only runs what the manager holds after a turn (or what a `Reasoner`/host added to it).
- No panics, `unwrap`, or `expect` outside tests; `tracing` for diagnostics; no new dependencies beyond the crate's existing set.
- No `parking_lot` guard is held across an `.await`: the scheduler clones what it needs, drops the guard, then awaits.
- Files stay around ≤1000 lines; `intent.rs` (1112, recognizer + planner plus two test modules), `base/mod.rs` (964), `base/tests.rs` (966), `executors/background/tests.rs` (1035), `legacy/schema.rs` (900) and `legacy/tests.rs` (586) are data- and test-heavy with documented roles.
- Only this node's files are edited; `src/lib.rs`, `src/CONTEXT.md`, the crate `CONTEXT.md`, and `docs/` belong to other nodes.

## Design Decisions

- **Two generations, one module tree, no shared types**: Python's new pipeline and its legacy lineage are ported side by side; legacy types are prefixed `Legacy` where Python reused a name (`Task`, `TaskStatus`, `Event`, `EventType`, `TaskResult`).
- **`TaskStatus` follows the current Python pipeline** (`submitted`, `working`, `paused`, `input-required`, `completed`, `canceled`, `failed`, `waiting`, `unknown`); the legacy `Pending`/`Running`/`Success`/`Cancelled`/`Interrupted` enum is `legacy::schema::LegacyTaskStatus`. An interrupted task is `input-required` in the new pipeline.
- **Events are one flattened record** instead of Python's `InputEvent`/`TaskInteractionEvent`/`TaskCompletionEvent`/`TaskFailedEvent`/`FollowUpEvent` classes; `EventType` selects the payload field (`input_data`, `task_interaction`, `task_result`) and `Event::text()`/`frames()` read it.
- **Intent carries Python's fields plus two Rust extensions** (`name`, `task_type`, both skipped when unset) so routing and dedupe need no side table.
- **`ControllerOutput` is a batch record** (`output`, `task_id`, `status`, `data`, `frames`), whereas Python's batch `ControllerOutput` is a `type`/`data`/`input_event_id` envelope; the streaming form is `ControllerOutputChunk` with the `controller_output` frame type.
- **The LLM recognizer uses a JSON-array contract**, not Python's `IntentToolkits` tool-call loop; malformed answers degrade to the default intent.
- **The router is a seam** (`TaskRouter`), not a `Runner.resource_mgr` model lookup: `BaseController` consults it once per turn and only when at least two executors are registered; one executor is chosen without a model call.
- **One task store, two dispatchers**: the synchronous turn dispatches its batch directly and then records the tasks in the manager, while the loop dispatches from the manager; `claim_task` guarantees a task is executed once even if both paths see it.
- **The manager owns hierarchy and indexes**: `created_seq`, priority levels, parent/child links and the root set live in the store so filtering and scheduling need no side tables; Python's `TaskManagerState` index serialization is replaced by one flat task list plus `next_seq`.
- **Pause/cancel is two-phase** (ask the executor outside the lock, re-check membership, cancel the driver, then write the status) so a task that finished in the meantime can never be resurrected into a non-terminal state.
- **`max_concurrent_tasks == 0` means unlimited** — a deliberate divergence: Python compares `>= 0` and therefore admits nothing.
- **Reasoner ops are a deterministic layer over the manager**, not a second control flow: the controller loop can call `Reasoner::apply_task_ops`, and its results are plain outcomes a host can inspect.

## Divergences from Python

- Admission order is deterministic (priority descending, then `created_seq`, then `task_id`); Python iterates an unordered selection.
- `metadata["depends_on"]` dependency gating is a Rust extension: a `Waiting` task with all dependencies `Completed` is promoted to `Submitted` at the top of a scan, and `TaskManager` keeps no dependency graph beyond that key.
- `max_concurrent_tasks = 0` is unlimited (see above).
- There is no asyncio event queue or pub/sub: the bounded FIFO is drained by the turn that enqueued into it, a dispatched task's result is recorded on the task, and `publish`/`send_to_agent` exist only through the `GroupRouter` seam.
- Group routing is a trait seam: Python's `Controller` publishes to a group object it owns; here the host installs any `GroupRouter`.
- The LLM intent toolkits (`IntentToolkits` schemas, tool-result feedback), the intent-detector prompt history (`chat_history`, `enable_history`, `category_info`, `example_content`), the context-engine/ability-manager injection into request handlers, and telemetry spans are not ported; the LLM recognizer replays the trailing conversation history as prior turns instead (`HISTORY_REPLAY_LIMIT` messages).
- `history_limit` counts **messages**, matching Python's message-indexed `chat_history[-2 * max_turns:]` slice, and the turn being recorded is protected from the append trim; Python keeps chat history in `ReasonerUtils.get_chat_history` (`agent_context.get_messages()`) rather than in `ControllerConfig`, and its newer `IntentRecognizer` asks the context for the newest 50 messages (current turn included).
- `expose_reasoning` has no Python counterpart: Python always streams reasoning as `llm_reasoning` frames, while this port gates reasoning behind the flag and drops it entirely when off.
- `event_timeout_ms` is in milliseconds (`0` = no deadline, Python's `None`) and also bounds the turn's synchronous value-type dispatch, because this port dispatches in-turn while Python's task execution is asynchronous (`task_timeout` covers that).
- `stream_first_frame_timeout_ms` bounds the first output frame **after** the `processing` acknowledgment (Python bounds one `__anext__` on the session stream, which that acknowledgment already is) and does not bound `invoke`.
- `ReasonerConfig`/`PlannerConfig`/`ProactiveIdentifierConfig`/`ReflectorConfig` are not ported; `agent_reasoner.py` is an empty artifact in Python and is covered by the `reasoner` task-op layer instead.
- The `Message*` legacy aliases are not ported; `WorkflowController`'s LLM workflow selection lives in `crate::workflow_agent`.
- `enable_task_persistence` defaults to `true` (Python's default is `false`) because the turn loop always saves; `false` makes `save_state` a no-op while in-memory state still accumulates.

## Notes for Agents

- `Event::from_user_inputs` accepts a string, a string array (one text frame per item), `{"query": ...}` (string or JSON payload), an interactive object (`user_inputs`/`interrupt_answers`), and any other object as a JSON frame; `Null`/scalars are `Error::validation`.
- `BaseController::start` spawns the loop only inside a tokio runtime (`Handle::try_current()`); outside one it stays a no-op and every pause/cancel falls back to the manager's status writes.
- `stop` is fire-and-forget (it requests the loop's cancellation and drops the handle), so a later `start` brings up a fresh loop; `shutdown` awaits the cancellation.
- Pause/cancel without a started loop still write statuses on the manager: cancel of an unknown task is `false`, of a `Submitted` task `Canceled` + `true`, of a terminal task `true`; pause of an `Submitted`/`Working` task `Paused` + `true`.
- The turn writes tasks to the manager only after the synchronous dispatch, so a loop started mid-turn never re-runs a task the turn just executed; `claim_task` is the hard guard.
- `UnknownTask` intents produce a clarification `ControllerOutput` with status `InputRequired` and `data.clarification_prompt`, and create no task.
- `TaskRun::frames` marks its single chunk `last_chunk`; the controller's `stream` clears intermediate flags so only the terminal chunk is last.
- `Planner::plan` ids are deterministic (`<session>_task_<seq>`), which the state and streaming tests rely on; `set_next_seq` rehydrates the counter from persisted state, and `save_state` overwrites the stored `next_seq` with the planner's.
- `enable_intent_recognition = false` bypasses the recognizer and yields one `default` create-task intent per input event.
- `Task::created_seq` is `#[serde(skip)]`; ordering after a restore comes from the stored `next_seq` and the task ids.
- Legacy types must never be passed into `schema::Task`/`schema::Event` (or vice versa): the two families are unrelated and there is no converter.
- `LLMController::select` propagates model transport errors but never errors on a malformed answer (first executor wins); the `TaskRouter` impl delegates to it and wraps it in `Some`.
- `legacy::schema::SourceType::Agent` keeps Python's `"single_agent"` wire value; `EventPriority` maps to Python's integer 1–4 via `EventPriority::value`.
- A legacy handoff sets `to_agent_id`; `handoff_to_agent()` translates it into `receiver_id`, which `routed_to(agent_id)` matches.
- `TaskQueue` deliberately keeps a cancelled entry until its owner unregisters it, so the next message can still see `has_running` while the previous driver winds down.
- The conversation history is in-memory only (never written to the session state), so `load_state` does not restore it and a fresh controller instance starts empty; `set_config` neither clears nor prunes it, the next append or read trims it.
- A turn that alone exceeds `history_limit` leaves the stored log longer than the limit (the append trim protects the turn being recorded) while `history()` still windows the read to the newest `limit` entries.
- `event_timeout_ms` needs a tokio runtime: without one the deadline is skipped rather than panicking, which keeps the legacy inline fallback working.
- A stream under `stream_first_frame_timeout_ms` emits the turn's own error (an event timeout, a recognizer failure) as the single `Err` frame after the `processing` ack; only a turn that produces no frame at all is replaced by the first-frame timeout error.
- Reasoning frames are `OutputKind::Reasoning` chunks that are never `last_chunk`, and `TaskRun::frames` does not produce them: the turns and the background `record_outputs` prepend them.

## Test Strategy

- Inline `#[cfg(test)]` modules in every file, offline and deterministic; the controller suite is 250 unit tests and runs in a couple of seconds.
- `task_manager` (46) covers filters and their validation matrix, union/narrowing/`with_children` resolution, deep-copy isolation, duplicate ids, claim atomicity, status updates with children, priority ordering, child promotion, and the snapshot/restore round-trip plus tolerance to corrupt state.
- `executors` (42) covers the registry, sequential dispatch, unknown types, `max_tasks`, timeouts, executor errors, the built-ins' answer/error/interrupt mapping, and `LLMController` selection — plus the `background` loop: admission order, the concurrency gate (unlimited and serial), dependency promotion, timeout text, two-phase pause/cancel and their idempotency, `stop`, `in_flight`, and the completion sentinel including its suppression.
- `reasoner` (19) covers `plan_ops` for every op kind, the confidence gate, the clarification state/frame, and each `apply_task_ops` mutation against a real `TaskManager` with fake executors.
- `base` (24) runs the loop end to end: routing (two executors through `LLMController`, the single-executor shortcut), unregistered task types, persistence across turns and a fresh restore, streaming shape, lifecycle and queue bounds, the three handler outcomes, task interaction, pause/cancel intents, and the started-loop scheduler path.
- `legacy` (42) covers wire round-trips, query extraction, detector parsing, the planner stub, delegation, per-conversation preemption, group delegation, and handoff translation.
- `queue` (6) and `group` (6) cover preemption bookkeeping and the typed not-in-group errors plus forwarding.
- `schema` (35), `intent` (21) and `config` (5) cover the wire shapes, tolerant LLM parsing, planner budgeting, and the config defaults.
- Timing-sensitive scheduler tests hold a load-bearing sleep ≥10× the step they wait for (e.g. `SLOW_EXECUTOR_HOLD` 500 ms against a 10 ms timeout) or poll with a bound, so they stay green under build load; every await that could hang is wrapped in `tokio::time::timeout`.
- Tests use `openjiuwen_test_support::ScriptedModel`, stub executors/handlers, and `AgentSession`; no network and no wall-clock ordering assertions.

## Routing Table

- `./schema/` → payload frames, events, intents, tasks, controller output (`schema/CONTEXT.md`)
- `./config.rs` → `ControllerConfig`, `DefaultResponse`, and the builders of the enforced knobs (`with_history_limit`, `with_event_timeout`, `with_stream_first_frame_timeout`, `with_expose_reasoning`)
- `./history.rs` → `HistoryEntry`/`HistoryRole`, the append trim (`trim_history`) and the read window (`window`) behind `history_limit`
- `./intent.rs` → `IntentRecognizer` (incl. the defaulted `recognize_with_history`), `IntentRule`, `RuleIntentRecognizer`, `LlmIntentRecognizer` (history replay), `Planner`
- `./task_manager/` → `TaskManager`, `TaskFilter`, indexes, hierarchy, persistence (`task_manager/CONTEXT.md`)
- `./executors/` → executor seam/registry, value-type `TaskScheduler`, built-in executors, and the background `TaskSchedulerLoop` (`executors/CONTEXT.md`)
- `./reasoner/` → `TaskOp`/`TaskOpOutcome`/`TaskControl` and the intent → task-op state machine (`reasoner/CONTEXT.md`)
- `./queue.rs` → per-conversation `TaskQueue`/`RunningTask`
- `./group.rs` → `GroupRouter` seam, `GroupMethod`, typed not-in-group errors
- `./base/` → `Controller`, `BaseController`, `EventHandler`, the turn loop, task control, persistence (`base/CONTEXT.md`)
- `./legacy/` → legacy value objects, `IntentDetector`, `LegacyPlanner`, `IntentDetectionController` preemption and group delegation (`legacy/CONTEXT.md`)
- `./mod.rs` → module declarations, controller-level re-exports, and the module-level architecture notes
