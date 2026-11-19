# openjiuwen-agent controller executors

## Intent

The task-executor seam and its built-ins, the synchronous batch scheduler used by the controller turn, and the background scheduling loop that drains the shared task store.
Ports Python `openjiuwen/core/controller/modules/task_scheduler.py` (executor contract, registry, scheduling loop) and the concrete executors of the application lineage (`ControllerAgent`, workflow and LLM controllers).
`TaskScheduler` serves one turn (caller-supplied tasks, sequential); `TaskSchedulerLoop` serves work that no turn dispatches (manager-backed, concurrent, pause/cancel-aware).
A driven run releases its in-flight slot through an RAII guard, and a panicking executor is contained into a failed task instead of unwinding the driver.

## API Surface

### `mod.rs` — seam, registry, router, batch scheduler

- `TaskExecutor` (async trait) — required `task_type()` and `execute(&Task, &TaskContext<'_>) -> Result<TaskResult>`; defaulted `description()` (`""`), `can_pause` (`(false, "task executor does not support pausing")`), `pause` (`false`), `can_cancel` (`(true, "")`), `cancel` (`false`).
- `TaskContext<'a>` (`Clone + Default`) — `session: Option<Arc<dyn Session>>`, `controller_id: &'a str`, `cancellation: CancellationToken`, `history: Vec<HistoryEntry>` (the dispatching turn's conversation, oldest first and already trimmed to `ControllerConfig::history_limit`; empty for a background-scheduler run), `expose_reasoning: bool` (whether to capture reasoning into `TaskResult::reasoning`; the default captures nothing); `new(session, controller_id)` starts with an empty history and `false`, then `with_cancellation(token)`, `with_history(history)`, `with_expose_reasoning(expose)`, plus the readers `session_id()` and `require_session()` (`Error::config("<owner> requires a session")`, owner is `controller_id` or `"task execution"`).
- `TaskExecutorRegistry` — `BTreeMap<String, Arc<dyn TaskExecutor>>`: `new`, `add` (keyed by `task_type()`), `insert(task_type, executor)`, `remove`, `get`, `contains`, `task_types` (sorted), `len`, `is_empty`.
- `TaskRouter` (async trait) — `name()`, `select(query: &str, intents: &[Intent]) -> Result<Option<String>>`; `Ok(None)` means "no opinion" and lets the controller fall back to `ControllerConfig::default_task_type`.
- `TaskRun` — `task: Task`, `result: Result<TaskResult>`; `status()`, `text()`, `reasoning()` (the executor's `TaskResult::reasoning` when the result is `Ok`, always `None` for a scheduler-recorded `Err`), `frames(start_index)`.
- `TaskScheduler` (value type, `Clone + Default`) — `new(registry, max_tasks)`, `with_timeout(Option<Duration>)`, `with_max_tasks`, `registry`, `registry_mut`, `max_tasks` (`0` = unlimited), `timeout`, `async dispatch(Vec<Task>, &TaskContext<'_>) -> Vec<TaskRun>`.
- Constants — `AGENT_TASK_TYPE = "agent"`, `WORKFLOW_TASK_TYPE = "workflow"`, `LLM_TASK_TYPE = "llm"`.

### `background.rs` — `TaskSchedulerLoop`

- `SessionProvider = Arc<dyn Fn(&str) -> Option<Arc<dyn Session>> + Send + Sync>`; the loop never owns sessions and asks the provider per admitted task.
- `TaskSchedulerLoop` (cheap `Clone`, custom non-exhaustive `Debug`) — `new(TaskManager, Arc<RwLock<TaskExecutorRegistry>>, Arc<RwLock<ControllerConfig>>, controller_id, SessionProvider)`, `manager()`, `config()`, `registry()` (the latter two clone the caller's handles), `start(self: &Arc<Self>) -> bool`, `is_running()`, `async stop()`, `wake()`, `in_flight() -> Vec<String>`, `async scan_once()`, `async wait_until_idle()`, `async pause_task(&str) -> bool`, `async cancel_task(&str) -> bool`, `async ensure_completion_signal(session_id)`.
- `record_outputs` (private) — appends the run's frames to the stored task as output chunks: the `task_completion`/`task_interaction`/`task_failed` chunk always last, preceded by one `reasoning` chunk when the run was dispatched with `expose_reasoning` and the executor reported non-empty reasoning.
- `execute_guarded` (private) — runs one executor under `task_timeout_ms` inside `AssertUnwindSafe(..).catch_unwind()`, mapping a panic payload to `"task executor panicked: <payload>"` so the driver never unwinds.
- `RunGuard` (private, `background/guard.rs`) — RAII cleanup of one driven run: `Drop` untracks the task, wakes the scan loop, and (when the run was not disarmed by `finish`) marks a still non-terminal task `Failed` with `DRIVER_PANIC_MESSAGE`; it holds only `Weak` handles to the in-flight map and the wake-up, plus a `TaskManager` clone.
- Constants — `COMPLETION_TEXT = "All tasks have been successfully processed"`, `DEPENDS_ON_KEY = "depends_on"`, `DRIVER_PANIC_MESSAGE = "task driver panicked"`.

### `builtin.rs` — built-in executors and the router

- `ControllerAgent` (type `agent`, description `"runs a controller agent"`) — `new(AgentCard, Arc<dyn Agent>)`, `card()`, `agent()`; `execute` calls `agent.invoke({"query": task.input_text()}, ctx.session)` and maps `Answer` → text, `Error` → failed, `Interrupt` → `input_required` (one JSON frame per agent state frame plus a text frame when the answer is non-empty). With `ctx.expose_reasoning` on, `execute` streams the agent instead (`agent.stream`, same inputs), folds every `llm_reasoning` chunk's string `content` into the result's reasoning, takes the last `answer` chunk's payload as the `AgentOutput`, and attaches the reasoning through `TaskResult::with_reasoning`; an `Err` item, an unparsable answer payload, and a stream without an `answer` chunk (`Error::llm("agent stream ended without an answer frame")`) surface as task failure. Reasoning capture is the only reason the streaming path is taken, and interrupts still map to `input_required` because the `answer` chunk carries the whole interrupt `AgentOutput`.
- `WorkflowController` (type `workflow`, description `"runs a workflow"`) — `new()`, `add_workflow` / `add_workflows` / `remove_workflow` / `workflows` / `get` keyed by workflow card id, `map_output(workflow_id, WorkflowOutput) -> TaskResult`, and `execute` selecting the workflow from `metadata["workflow_id"]`, then `task.description`, then the single registered workflow and running it in a child `WorkflowSession` named `"{parent}::{workflow_id}"`.
- `ExecutorInfo` — serde record `{task_type, name, description}` with `new`, used as the LLM router's model-visible catalogue.
- `LLMController` (`TaskRouter`, `name() == "llm"`) — `new(model, Vec<ExecutorInfo>)`, `with_default_task_type`, `with_request`, `with_prompt`, `executors()`, `default_task_type()`, `add_executor` (consuming), `async select(query, intents) -> Result<String>`.

## Constraints

- `TaskScheduler::dispatch` runs sequentially in caller order, truncates at `max_tasks` (`0` = unlimited), and never returns `Err`: a missing executor, an executor error, and a timeout become failed tasks carrying the reason.
- Terminal tasks pass through unchanged; a successful result copies its status and `error_message` onto the task only for `Completed`/`Failed`/`InputRequired`/`Paused`/`Canceled` (other statuses leave the task as it was).
- `TaskRun::frames` always yields exactly one `ControllerOutputChunk::last`: `task_completion` for `Completed`, `task_interaction` for `InputRequired`, `task_failed` for everything else, with the result's frames or one text frame.
- The batch timeout message is `"task {task_id} timed out after {seconds}s"`; the background loop's is Python's `"Task timeout after {seconds} seconds"`.
- Reasoning capture is opt-in per dispatch: `TaskContext::expose_reasoning` is the only switch, `TaskResult::reasoning` is the only carrier, and `record_outputs` drops reasoning when the flag is off, so a host that never sets it observes exactly the previous output chunks.
- `ControllerAgent` and `WorkflowController` inherit the default pause/cancel hooks; only a host-supplied executor overrides them, and a hook's return value is informational because the scheduler records `Paused`/`Canceled` itself.
- The workflow executor requires a session (`require_session`) and reports `Error::not_found("no workflow registered for task {task_id}")` when selection fails.
- `LLMController::select` returns the default task type with no executors, the sole executor's type without a model call, and otherwise asks the model once with a deterministic catalogue; transport errors propagate while a malformed answer falls back to the first executor.
- Public paths stay `controller::executors::*`; `background`, `background::guard` and `builtin` are additive submodules.
- Every admitted run releases its slot: the RAII guard owns untrack/wake/failure marking, and `finish` only disarms the guard's failure marking before untracking, waking and signalling, so the happy-path cleanup is unchanged.
- A panicking executor never escapes the driver (`catch_unwind` turns it into a failed task with `"task executor panicked: <payload>"`), and the guard's own `Drop` body is wrapped in `catch_unwind` because a panic while unwinding aborts the process.
- Files stay around ≤1000 lines; each suite lives beside its module rather than inside it (`builtin/tests.rs`, `background/tests/mod.rs` — 1040 lines of behaviour suites — plus `background/tests/panic.rs` and the `guard.rs` unit module).

## Notes for Agents

### Background loop semantics

- `start` flips `running`, installs the manager's submitted callback (a `Weak` handle, replacing any previous listener), spawns `run_loop`, and returns `false` when already running; it panics outside a Tokio runtime.
- `run_loop` scans, then waits on cancellation, a `wake`, or `schedule_interval_ms.max(1)`; a wake during a scan is not lost because `Notify` keeps a permit.
- `scan_once` yields once, promotes ready dependents, fetches `Submitted` tasks, sorts by priority descending then `created_seq` then `task_id`, admits until `max_concurrent_tasks` (`0` = unlimited), skips ids already in flight, skips an unknown session with a warning (the task stays `Submitted`), claims the task through `TaskManager::claim_task`, and spawns one driver per admission.
- Dependency gating reads `metadata["depends_on"]` as an array of task ids: every id must exist and be `Completed`, an empty array promotes immediately, and a task with a missing or unfinished dependency stays `Waiting`.
- The driver creates its `RunGuard` before any await; a panic anywhere after that point still untracks the run and wakes the loop, and a task that never reached a terminal status is marked `Failed` with `"task driver panicked"`.
- The driver builds `TaskContext` with the run's cancellation token and the live `expose_reasoning` flag (both read from the shared config before any await, next to `task_timeout_ms`), and leaves its history empty because a scheduled run has no dispatching turn; it then records the executor so a later pause/cancel can reach its hooks, applies `task_timeout_ms`, leaves the status to the caller when the token was cancelled during execution, maps an executor `Err` to `Failed`, maps `Failed`/`InputRequired` through and every other result status to `Completed`, appends the run's output chunks, then untracks itself, wakes the loop, and checks the completion signal.
- The appended chunks are the `task_completion`/`task_interaction`/`task_failed` chunk (always `last_chunk`, indices following the reasoning chunk) preceded by one `reasoning` chunk (indices `0`, `last_chunk` only when it is the sole chunk) when the run was dispatched with `expose_reasoning` on and the executor reported non-empty reasoning; reasoning is otherwise dropped, and a run with neither data nor exposed reasoning appends nothing.
- Pause and cancel are two-phase: look up the task, snapshot the in-flight run, ask `can_pause`/`can_cancel` (a refusal warns and returns `false` without calling the hook), take the run under the lock, cancel its token, await the driver, untrack it, then set `Paused`/`Canceled`.
- `cancel_task` short-circuits a `Submitted` task to `Canceled`, returns `true` for an already-terminal task, and `false` for a task that is neither in flight nor resumable.
- `stop` is the only teardown and is idempotent: it cancels the loop token, joins the loop, then cancels and joins every in-flight driver; the loop token stays cancelled, so a stopped scheduler is rebuilt rather than restarted.
- The completion sentinel writes one `all_tasks_processed` frame (`COMPLETION_TEXT`, `last_chunk`) through `session.write_stream`, at most once per idle stretch per session; the marker resets when the session has active tasks or a new run starts, and `suppress_completion_signal` disables it.
- Nothing escapes as `Err`: scan failures, unknown sessions, missing executors and driver failures are logged and swallowed, and `wait_until_idle` polls every 5 ms up to a 5 s deadline instead of hanging.

### Ordering and admission

- Deterministic ordering is a Rust choice: Python iterates an unspecified selection and blocks every admission when `max_concurrent_tasks == 0`.
- Synchronous turn dispatch is kept; the loop only picks up tasks left `Submitted` in the store.
- The loop's knobs are set on `ControllerConfig` through the `with_max_concurrent_tasks`, `with_schedule_interval`, `with_task_timeout`, `with_history_limit` and `with_expose_reasoning` builders and read live from the shared config handle; dependency gating has no config field (it reads `metadata["depends_on"]` per task).
- `TaskContext::history` is filled by whoever dispatches: the controller turn passes its trimmed conversation, a direct `TaskScheduler::dispatch` caller passes what it wants, and the background scheduler never passes one.
- Reasoning reaches a host through two channels and never through `TaskRun::frames`: `TaskResult::reasoning`/`TaskRun::reasoning` for the turn's batch runs, and the appended `reasoning` output chunk for a background run.

## Design Decisions

- **The loop is a seam over `TaskManager` state, not an event consumer**: results are recorded on the task (status, `error_message`, appended output chunks), replacing Python's `TaskCompletionEvent`/`TaskInteractionEvent`/`TaskFailedEvent` publication.
- **Admission is guarded twice**: the in-flight map prevents local double-runs and `claim_task`'s atomic `Submitted -> Working` prevents one across a concurrent status change.
- **The loop is single-use**: `stop` cancels the loop's own token permanently, so `BaseController` builds a fresh loop after a stop.
- **`TaskRun::frames` reuses the output schema** so the turn and the loop report results in the same wire form.
- **Reasoning is captured by streaming, never by a second agent call**: `ControllerAgent` streams only when `expose_reasoning` is on, so a host that never opts in keeps the single `invoke` call and its exact output.
- **Driver cleanup is RAII, not a tail call**: `RunGuard` owns untrack/wake/panic marking so the invariant survives an unwind, while `finish` disarms the marking and keeps the happy path's untrack→wake→signal order.
- **Panic containment is per run, not per scheduler**: `execute_guarded` catches the executor's panic into a `Failed` task with its payload in `error_message`, so the loop keeps running and the capacity slot is reusable.
- **Reasoning precedes the terminal chunk in the task log**: a `reasoning` chunk is appended before the completion/interaction/failure chunk so the `last_chunk` marker always stays on the chunk that reports the outcome.

## Test Strategy

- 56 offline tests in this node: `mod.rs` 12, `builtin/tests.rs` 16, `background/tests/mod.rs` 22, `background/guard.rs` 5, `background/tests/panic.rs` 1; all deterministic, bounded and network-free.
- `mod.rs` covers the `TaskContext` session requirement, cancellation plumbing, the empty-history/`false` defaults and both new builders, the registry add/replace/remove and sorted types, sequential order, unknown-task failure, `max_tasks` truncation, executor error capture, timeout text, terminal pass-through, `TaskRun::reasoning` (successful result, absent reasoning, scheduler-recorded `Err`), and the `TaskRun` frame kinds.
- `builtin.rs` covers agent answer/error/interrupt mapping, `map_output` for every workflow state, workflow registry/selection/session errors, the LLM router's zero/one/many-executor paths, prompt and request builders, `TaskRouter` use, and `ExecutorInfo` round-trips — plus `controller_agent_captures_reasoning_only_when_exposed` (a `ScriptedModel` built from `MockCompletion::text("the answer").with_reasoning("the reasoning")`: flag on yields `reasoning == Some("the reasoning")` with the unchanged answer text, flag off takes the `invoke` path and carries no reasoning), `controller_agent_streams_interrupts_into_input_required` (an `answer` chunk carrying an interrupt `AgentOutput` still maps to `input_required` with its frames), and `controller_agent_reports_stream_failures` (missing answer frame, propagated stream `Err`, and the flag-off path staying on `invoke`).
- `background/tests/mod.rs` covers the concurrency gate (1 vs unlimited), the claim guard, the exact timeout text, cooperative pause/cancel plus refusals, resubmission after pause, missing-executor containment, dependency promotion, outcome-to-status mapping with output chunks, the completion sentinel (once per idle stretch, blocked by active tasks, suppressed by config), unknown sessions, in-flight listing, start/stop idempotency with wake-based scans, and cancellation-token plumbing — plus `exposed_reasoning_is_appended_before_the_terminal_chunk` (two chunks, `reasoning` first and not last, `task_completion` last), `reasoning_is_dropped_when_not_exposed` (exactly today's single terminal chunk, no `reasoning` payload), and `reasoning_without_frames_is_recorded_alone_when_exposed` (one `reasoning` chunk when exposed, nothing when not).
- `background/guard.rs` covers the guard directly: a dropped guard untracks, wakes the loop and fails a `Working` task; a guard dropped during an actual unwind does the same; a disarmed guard and an armed guard over a terminal task leave the recorded status alone; and `panic_text` reads `&'static str`, formatted `String` and non-text payloads.
- `background/tests/panic.rs` is the capacity-leak regression: with `max_concurrent_tasks = 2` two panicking runs are both admitted, both end `Failed` with `"task executor panicked: boom"`, both are untracked, a healthy task is still admitted and completes afterwards, and `all_tasks_processed` fires once per idle stretch (the panic stretch and the healthy run).
- Timing margins: waits poll every 5 ms with a 5 s bound, cooperative holds are 20–300 ms and timeout/pause holds 2–5 s (always interrupted), and two fixed settle sleeps (20 ms, 50 ms) cover suppression and post-stop behaviour; assertions read store state and recorded frames, never elapsed time.

## Routing Table

- `./mod.rs` → `TaskContext`, `TaskExecutor`, `TaskExecutorRegistry`, `TaskRouter`, `TaskRun`, `TaskScheduler`, task-type constants
- `./builtin.rs` → `ControllerAgent` (including its reasoning-capture streaming path), `WorkflowController`, `ExecutorInfo`, `LLMController`
- `./builtin/tests.rs` → the concrete executors' behaviour suite
- `./background.rs` → `TaskSchedulerLoop`, `SessionProvider`, scan/admission/driver/pause/cancel/completion sentinel, output-chunk append order
- `./background/guard.rs` → `RunGuard`, `panic_text`, `DRIVER_PANIC_MESSAGE` and their unit suite
- `./background/tests/mod.rs` → the loop's behaviour suite (fixtures, `TestExecutor`, `Outcome`, `Fixture`)
- `./background/tests/panic.rs` → the panicking-executor capacity-leak regression
