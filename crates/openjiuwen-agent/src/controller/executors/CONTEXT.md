# openjiuwen-agent controller executors

## Intent

The task-executor seam and its built-in implementations, plus the sequential task scheduler.
Ports Python `openjiuwen/core/controller/modules/task_scheduler.py` (executor contract and registry) and the concrete executors of the application lineage (`ControllerAgent`, workflow and LLM controllers).

## API Surface

- `TaskExecutor` — `task_type`, default `description`, `async execute(&Task, &TaskContext<'_>) -> Result<TaskResult>`.
- `TaskContext<'a>` — `session: Option<Arc<dyn Session>>`, `controller_id: &'a str`; `session_id`, `require_session`.
- `TaskExecutorRegistry` — `new`, `add`, `insert`, `remove`, `get`, `contains`, `task_types`, `len`, `is_empty` (keyed by task type, sorted).
- `TaskRouter` — `name`, `async select(query, intents) -> Result<Option<String>>`.
- `TaskScheduler` — `new(registry, max_tasks)`, `with_timeout`, `with_max_tasks`, `registry`/`registry_mut`, `max_tasks`, `timeout`, `async dispatch(tasks, &TaskContext<'_>) -> Vec<TaskRun>`.
- `TaskRun` — `task`, `result`, `status`, `text`, `frames(start_index)`.
- `ControllerAgent` (task type `agent`) — wraps `Arc<dyn Agent>` and `AgentCard`.
- `WorkflowController` (task type `workflow`) — `add_workflow(s)`, `remove_workflow`, `workflows`, `get`, `map_output`.
- `ExecutorInfo` / `LLMController` (the `TaskRouter` impl) — `select`, `executors`, `default_task_type`, `with_request`, `with_prompt`.
- Constants `AGENT_TASK_TYPE`, `WORKFLOW_TASK_TYPE`, `LLM_TASK_TYPE`.

## Constraints

- `dispatch` is sequential in caller order, truncates at `max_tasks` (`0` = unlimited), and never returns `Err`: missing executor, executor error, and timeout become failed tasks carrying the reason.
- Terminal tasks pass through unchanged; a successful result copies its status and `error_message` onto the task.
- The workflow executor builds a child `WorkflowSession` whose `parent` is the task session; `Workflow::invoke` receives `{"query": <task input text>}`.
- `ControllerAgent` maps `Answer` → completed text, `Error` → failed, `Interrupt` → `InputRequired` with the agent's interaction frames.
- Public paths stay `controller::executors::*`; `builtin` is an additive submodule.

## Notes for Agents

- `LLMController::select` returns the sole executor without a model call, otherwise asks the model with a deterministic executor list; a malformed answer falls back to the first executor, while model transport errors propagate.
- `WorkflowController::map_output` is public so hosts and tests can map a `WorkflowOutput` without running a workflow; `Completed` keeps the raw `result` value, `InputRequired` turns the frame array into JSON data frames, `Error` becomes a failed result.
- The scheduler has no background loop, priority ordering, or pause/cancel; those belong to the deferred Python `TaskScheduler`.
- `TaskContext` is a borrowed view: executors must clone the session out of it before spawning work.

## Routing Table

- `./mod.rs` → `TaskContext`, `TaskExecutor`, `TaskExecutorRegistry`, `TaskRouter`, `TaskRun`, `TaskScheduler`, task-type constants, scheduler tests
- `./builtin.rs` → `ControllerAgent`, `WorkflowController`, `ExecutorInfo`, `LLMController`, mapping/scheduling tests
