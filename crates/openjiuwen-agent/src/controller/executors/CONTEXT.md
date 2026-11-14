# openjiuwen-agent controller executors

## Intent

The task-executor seam and its built-ins, the synchronous batch scheduler used by the controller turn, and the background scheduling loop that drains the shared task store.
Ports Python `openjiuwen/core/controller/modules/task_scheduler.py` (executor contract, registry, scheduling loop) and the concrete executors of the application lineage (`ControllerAgent`, workflow and LLM controllers).
`TaskScheduler` serves one turn (caller-supplied tasks, sequential); `TaskSchedulerLoop` serves work that no turn dispatches (manager-backed, concurrent, pause/cancel-aware).

## API Surface

### `mod.rs` — seam, registry, router, batch scheduler

- `TaskExecutor` (async trait) — required `task_type()` and `execute(&Task, &TaskContext<'_>) -> Result<TaskResult>`; defaulted `description()` (`""`), `can_pause` (`(false, "task executor does not support pausing")`), `pause` (`false`), `can_cancel` (`(true, "")`), `cancel` (`false`).
- `TaskContext<'a>` — `session: Option<Arc<dyn Session>>`, `controller_id: &'a str`, `cancellation: CancellationToken`; `new(session, controller_id)`, `with_cancellation(token)`, `session_id()`, `require_session()` (`Error::config("<owner> requires a session")`, owner is `controller_id` or `"task execution"`).
- `TaskExecutorRegistry` — `BTreeMap<String, Arc<dyn TaskExecutor>>`: `new`, `add` (keyed by `task_type()`), `insert(task_type, executor)`, `remove`, `get`, `contains`, `task_types` (sorted), `len`, `is_empty`.
- `TaskRouter` (async trait) — `name()`, `select(query: &str, intents: &[Intent]) -> Result<Option<String>>`; `Ok(None)` means "no opinion" and lets the controller fall back to `ControllerConfig::default_task_type`.
- `TaskRun` — `task: Task`, `result: Result<TaskResult>`; `status()`, `text()`, `frames(start_index)`.
- `TaskScheduler` (value type, `Clone + Default`) — `new(registry, max_tasks)`, `with_timeout(Option<Duration>)`, `with_max_tasks`, `registry`, `registry_mut`, `max_tasks` (`0` = unlimited), `timeout`, `async dispatch(Vec<Task>, &TaskContext<'_>) -> Vec<TaskRun>`.
- Constants — `AGENT_TASK_TYPE = "agent"`, `WORKFLOW_TASK_TYPE = "workflow"`, `LLM_TASK_TYPE = "llm"`.

### `background.rs` — `TaskSchedulerLoop`

- `SessionProvider = Arc<dyn Fn(&str) -> Option<Arc<dyn Session>> + Send + Sync>`; the loop never owns sessions and asks the provider per admitted task.
- `TaskSchedulerLoop` (cheap `Clone`, custom non-exhaustive `Debug`) — `new(TaskManager, Arc<RwLock<TaskExecutorRegistry>>, Arc<RwLock<ControllerConfig>>, controller_id, SessionProvider)`, `manager()`, `config()`, `registry()` (the latter two clone the caller's handles), `start(self: &Arc<Self>) -> bool`, `is_running()`, `async stop()`, `wake()`, `in_flight() -> Vec<String>`, `async scan_once()`, `async wait_until_idle()`, `async pause_task(&str) -> bool`, `async cancel_task(&str) -> bool`, `async ensure_completion_signal(session_id)`.
- Constants — `COMPLETION_TEXT = "All tasks have been successfully processed"`, `DEPENDS_ON_KEY = "depends_on"`.

### `builtin.rs` — built-in executors and the router

- `ControllerAgent` (type `agent`, description `"runs a controller agent"`) — `new(AgentCard, Arc<dyn Agent>)`, `card()`, `agent()`; `execute` calls `agent.invoke({"query": task.input_text()}, ctx.session)` and maps `Answer` → text, `Error` → failed, `Interrupt` → `input_required` (one JSON frame per agent state frame plus a text frame when the answer is non-empty).
- `WorkflowController` (type `workflow`, description `"runs a workflow"`) — `new()`, `add_workflow` / `add_workflows` / `remove_workflow` / `workflows` / `get` keyed by workflow card id, `map_output(workflow_id, WorkflowOutput) -> TaskResult`, and `execute` selecting the workflow from `metadata["workflow_id"]`, then `task.description`, then the single registered workflow and running it in a child `WorkflowSession` named `"{parent}::{workflow_id}"`.
- `ExecutorInfo` — serde record `{task_type, name, description}` with `new`, used as the LLM router's model-visible catalogue.
- `LLMController` (`TaskRouter`, `name() == "llm"`) — `new(model, Vec<ExecutorInfo>)`, `with_default_task_type`, `with_request`, `with_prompt`, `executors()`, `default_task_type()`, `add_executor` (consuming), `async select(query, intents) -> Result<String>`.

## Constraints

- `TaskScheduler::dispatch` runs sequentially in caller order, truncates at `max_tasks` (`0` = unlimited), and never returns `Err`: a missing executor, an executor error, and a timeout become failed tasks carrying the reason.
- Terminal tasks pass through unchanged; a successful result copies its status and `error_message` onto the task only for `Completed`/`Failed`/`InputRequired`/`Paused`/`Canceled` (other statuses leave the task as it was).
- `TaskRun::frames` always yields exactly one `ControllerOutputChunk::last`: `task_completion` for `Completed`, `task_interaction` for `InputRequired`, `task_failed` for everything else, with the result's frames or one text frame.
- The batch timeout message is `"task {task_id} timed out after {seconds}s"`; the background loop's is Python's `"Task timeout after {seconds} seconds"`.
- `ControllerAgent` and `WorkflowController` inherit the default pause/cancel hooks; only a host-supplied executor overrides them, and a hook's return value is informational because the scheduler records `Paused`/`Canceled` itself.
- The workflow executor requires a session (`require_session`) and reports `Error::not_found("no workflow registered for task {task_id}")` when selection fails.
- `LLMController::select` returns the default task type with no executors, the sole executor's type without a model call, and otherwise asks the model once with a deterministic catalogue; transport errors propagate while a malformed answer falls back to the first executor.
- Public paths stay `controller::executors::*`; `background` and `builtin` are additive submodules.

## Notes for Agents

### Background loop semantics

- `start` flips `running`, installs the manager's submitted callback (a `Weak` handle, replacing any previous listener), spawns `run_loop`, and returns `false` when already running; it panics outside a Tokio runtime.
- `run_loop` scans, then waits on cancellation, a `wake`, or `schedule_interval_ms.max(1)`; a wake during a scan is not lost because `Notify` keeps a permit.
- `scan_once` yields once, promotes ready dependents, fetches `Submitted` tasks, sorts by priority descending then `created_seq` then `task_id`, admits until `max_concurrent_tasks` (`0` = unlimited), skips ids already in flight, skips an unknown session with a warning (the task stays `Submitted`), claims the task through `TaskManager::claim_task`, and spawns one driver per admission.
- Dependency gating reads `metadata["depends_on"]` as an array of task ids: every id must exist and be `Completed`, an empty array promotes immediately, and a task with a missing or unfinished dependency stays `Waiting`.
- The driver builds `TaskContext` with the run's cancellation token, records the executor so a later pause/cancel can reach its hooks, applies `task_timeout_ms`, leaves the status to the caller when the token was cancelled during execution, maps an executor `Err` to `Failed`, maps `Failed`/`InputRequired` through and every other result status to `Completed`, appends one output chunk to the stored task when `result.data` is non-empty, then untracks itself, wakes the loop, and checks the completion signal.
- Pause and cancel are two-phase: look up the task, snapshot the in-flight run, ask `can_pause`/`can_cancel` (a refusal warns and returns `false` without calling the hook), take the run under the lock, cancel its token, await the driver, untrack it, then set `Paused`/`Canceled`.
- `cancel_task` short-circuits a `Submitted` task to `Canceled`, returns `true` for an already-terminal task, and `false` for a task that is neither in flight nor resumable.
- `stop` is the only teardown and is idempotent: it cancels the loop token, joins the loop, then cancels and joins every in-flight driver; the loop token stays cancelled, so a stopped scheduler is rebuilt rather than restarted.
- The completion sentinel writes one `all_tasks_processed` frame (`COMPLETION_TEXT`, `last_chunk`) through `session.write_stream`, at most once per idle stretch per session; the marker resets when the session has active tasks or a new run starts, and `suppress_completion_signal` disables it.
- Nothing escapes as `Err`: scan failures, unknown sessions, missing executors and driver failures are logged and swallowed, and `wait_until_idle` polls every 5 ms up to a 5 s deadline instead of hanging.

### Ordering and admission

- Deterministic ordering is a Rust choice: Python iterates an unspecified selection and blocks every admission when `max_concurrent_tasks == 0`.
- Synchronous turn dispatch is kept; the loop only picks up tasks left `Submitted` in the store.

## Design Decisions

- **The loop is a seam over `TaskManager` state, not an event consumer**: results are recorded on the task (status, `error_message`, appended output chunks), replacing Python's `TaskCompletionEvent`/`TaskInteractionEvent`/`TaskFailedEvent` publication.
- **Admission is guarded twice**: the in-flight map prevents local double-runs and `claim_task`'s atomic `Submitted -> Working` prevents one across a concurrent status change.
- **The loop is single-use**: `stop` cancels the loop's own token permanently, so `BaseController` builds a fresh loop after a stop.
- **`TaskRun::frames` reuses the output schema** so the turn and the loop report results in the same wire form.

## Test Strategy

- 42 offline tests in this node: `mod.rs` 10, `builtin.rs` 13, `background/tests.rs` 19; all deterministic, bounded and network-free.
- `mod.rs` covers the `TaskContext` session requirement and cancellation plumbing, registry add/replace/remove/sorted types, sequential order, unknown-task failure, `max_tasks` truncation, executor error capture, timeout text, terminal pass-through, and the `TaskRun` frame kinds.
- `builtin.rs` covers agent answer/error/interrupt mapping, `map_output` for every workflow state, workflow registry/selection/session errors, the LLM router's zero/one/many-executor paths, prompt and request builders, `TaskRouter` use, and `ExecutorInfo` round-trips.
- `background/tests.rs` covers the concurrency gate (1 vs unlimited), the claim guard, the exact timeout text, cooperative pause/cancel plus refusals, resubmission after pause, missing-executor containment, dependency promotion, outcome-to-status mapping with output chunks, the completion sentinel (once per idle stretch, blocked by active tasks, suppressed by config), unknown sessions, in-flight listing, start/stop idempotency with wake-based scans, and cancellation-token plumbing.
- Timing margins: waits poll every 5 ms with a 5 s bound, cooperative holds are 20–300 ms and timeout/pause holds 2–5 s (always interrupted), and two fixed settle sleeps (20 ms, 50 ms) cover suppression and post-stop behaviour; assertions read store state and recorded frames, never elapsed time.

## Routing Table

- `./mod.rs` → `TaskContext`, `TaskExecutor`, `TaskExecutorRegistry`, `TaskRouter`, `TaskRun`, `TaskScheduler`, task-type constants
- `./builtin.rs` → `ControllerAgent`, `WorkflowController`, `ExecutorInfo`, `LLMController`
- `./background.rs` → `TaskSchedulerLoop`, `SessionProvider`, scan/admission/driver/pause/cancel/completion sentinel
- `./background/tests.rs` → the loop's behaviour suite
