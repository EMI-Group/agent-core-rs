# openjiuwen-agent controller

## Intent

Rust counterpart of the Python package `openjiuwen/core/controller/`: an event feeds intent recognition, intents are planned into tasks, and each task is dispatched to the executor registered for its task type.
The turn loop owns the bounded event queue, the recognizer, the optional LLM task router, the planner, and the executor registry, and persists its task state in the session so a later turn resumes with the same tasks.
Two Python generations live here without sharing types: the current event-queue pipeline (`schema`, `config`, `intent`, `executors`, `base`) and the superseded string-routed lineage (`legacy`), whose types carry a `Legacy` prefix.

## API Surface

- `base::Controller` — `card`, `config`, `invoke(inputs, session) -> Result<ControllerOutput>`, `stream(inputs, session) -> BoxStream<'static, Result<ControllerOutputChunk>>`.
- `base::BaseController` — the loop: `new(card, config)`, `with_recognizer`, `push_event`/`drain_events`/`pending_events`, `set_intent_recognizer`, `set_event_handler`/`clear_event_handler`, `set_router`/`clear_router`, `set_planner`, `add_executor`/`add_default_executor`/`remove_executor`/`executor`/`task_types`/`registry`/`scheduler`, `start`/`stop`/`is_running`, `bind_session`/`unbind_session`/`bound_sessions`, `save_state`/`load_state`/`tasks`, and `impl Controller`.
- `base::EventHandler` / `EventHandlerOutcome` — `Intents(Vec<Intent>)`, `Output(Box<ControllerOutput>)`, `Ignored`; a handler replaces the built-in recognizer for an event.
- `base::TASKS_STATE_KEY` — `"__controller_tasks__"`, the session state key holding `{"tasks": [<Task>...], "next_seq": <u64>}`.
- `config::ControllerConfig` — `max_intents`, `max_tasks`, `history_limit`, `default_response`, `expose_reasoning`, `default_task_type`, plus Python knobs (`event_queue_size`, `max_concurrent_tasks`, `schedule_interval_ms`, `task_timeout_ms`, `default_task_priority`, `enable_task_persistence`, `event_timeout_ms`, `enable_intent_recognition`, `intent_confidence_threshold`, `intent_type_list`, `suppress_completion_signal`, `stream_first_frame_timeout_ms`) and `accepts(IntentType)`.
- `config::DefaultResponse` — optional fallback answer text.
- `intent::IntentRecognizer` — `name`, `async recognize(&Event) -> Result<Vec<Intent>>`.
- `intent::RuleIntentRecognizer` / `IntentRule` — keyword rules; no match yields one `default` create-task intent.
- `intent::LlmIntentRecognizer` — asks a `Model` for a JSON intent array (tolerant parse, default intent on any failure).
- `intent::Planner` — `plan(session_id, intents)` / `plan_with_task_type(session_id, intents, Option<&str>)`; validates, dedupes by label, budgets, and mints `<session>_task_<seq>` ids.
- `executors::TaskExecutor` — `task_type`, `description`, `async execute(&Task, &TaskContext<'_>) -> Result<TaskResult>`.
- `executors::TaskExecutorRegistry`, `TaskScheduler` (sequential, `max_tasks`, optional timeout), `TaskRun` (`status`, `text`, `frames`), `TaskContext` (`session`, `controller_id`).
- `executors::TaskRouter` — `async select(query, intents) -> Result<Option<String>>`.
- `executors::ControllerAgent` (task type `agent`), `WorkflowController` (task type `workflow`, plus `map_output`), `ExecutorInfo`, `LLMController` (the router; single-executor shortcut without a model call).
- `legacy::IntentDetectionController` — `BaseController` wrapper preserving Python's intent-detection entry points (`intent_detection`, `add_executor`, `set_router`, `set_event_handler`).
- `legacy::IntentDetector` / `LegacyPlanner` — tolerant `{"result": n}` category classification and the one-undefined-task planner stub.
- `legacy::schema` — `LegacyTask`, `LegacyTaskStatus`, `TaskInput`, `LegacyTaskResult`, `TaskDependency`, `DependencyType`, `LegacyEvent`, `LegacyEventType`, `EventPriority`, `SourceType`, `EventSource`, `EventContent`, `EventContext`.
- `schema` — `DataFrame` family, `Event`/`EventType`, `Intent`/`IntentType`, `Task`/`TaskStatus`/`TaskResult`, `ControllerOutputPayload`/`ControllerOutputChunk`/`ControllerOutput`/`OutputKind`.
- `mod.rs` re-exports the controller-level names above; `schema` and `legacy::schema` keep their own namespaces.

## Constraints

- Every public type is serde-serializable with Python wire names (`mimeType`, `input-required`, `controller_output`, `user_input`, `single_agent`) and tolerates unknown fields; serde defaults match `Default`.
- Errors are `openjiuwen_common::error::{Error, Result}`; a missing executor, an executor failure, a timeout, or a task-control intent with an unknown target becomes a failed task/output, never `Err`.
- `invoke` is single-turn and in-process: events are drained FIFO, intents are processed in order, tasks are dispatched sequentially, and results aggregate in intent order.
- No panics, `unwrap`, or `expect` outside tests; `tracing` for diagnostics; no new dependencies beyond the crate's existing set.
- Files stay around ≤700 lines (largest: `legacy/schema.rs` 777, `intent.rs` 918 — both are data-heavy/test-heavy with documented roles).
- Only this node's files are edited; `src/lib.rs`, `src/CONTEXT.md`, the crate `CONTEXT.md`, and `docs/` belong to other nodes.

## Design Decisions

- **Two generations, one module tree, no shared types**: Python's new pipeline and its legacy lineage are ported side by side; legacy types are prefixed `Legacy` where Python reused a name (`Task`, `TaskStatus`, `Event`, `EventType`, `TaskResult`).
- **`TaskStatus` follows the current Python pipeline** (`submitted`, `working`, `paused`, `input-required`, `completed`, `canceled`, `failed`, `waiting`, `unknown`); the legacy `Pending`/`Running`/`Success`/`Cancelled`/`Interrupted` enum is `legacy::schema::LegacyTaskStatus`. An interrupted task is `input-required` in the new pipeline.
- **Events are one flattened record** instead of Python's `InputEvent`/`TaskInteractionEvent`/`TaskCompletionEvent`/`TaskFailedEvent`/`FollowUpEvent` classes; `EventType` selects the payload field (`input_data`, `task_interaction`, `task_result`) and `Event::text()`/`frames()` read it.
- **Intent carries Python's fields plus two Rust extensions** (`name`, `task_type`, both skipped when unset) so routing and dedupe need no side table.
- **`ControllerOutput` is a batch record** (`output`, `task_id`, `status`, `data`, `frames`), whereas Python's batch `ControllerOutput` is a `type`/`data`/`input_event_id` envelope; the streaming form is `ControllerOutputChunk` with the `controller_output` frame type.
- **The LLM recognizer uses a JSON-array contract**, not Python's `IntentToolkits` tool-call loop; malformed answers degrade to the default intent.
- **The router is a seam** (`TaskRouter`), not a `Runner.resource_mgr` model lookup: `BaseController` consults it once per turn and only when at least two executors are registered; one executor is chosen without a model call.
- **Persistence is one session key** (`__controller_tasks__`) holding tasks plus the planner sequence, replacing Python's `TaskManagerState` indexes; `Task::created_seq` is `#[serde(skip)]`, so ordering after a restore comes from the stored `next_seq` and task ids.

## Notes for Agents

- `Event::from_user_inputs` accepts a string, a string array (one text frame per item), `{"query": ...}` (string or JSON payload), an interactive object (`user_inputs`/`interrupt_answers`), and any other object as a JSON frame; `Null`/scalars are `Error::validation`.
- `BaseController` holds an `Arc<Inner>` and is `Clone`; `stream` clones the handle into a spawned turn, emits a `processing` chunk first, then the turn frames, then one terminal `last_chunk` chunk unless `suppress_completion_signal` is set.
- Aggregation precedence is `InputRequired` > `Completed` > `Failed`; completed task texts are joined with `\n\n`; when no text remains the `default_response.text` is used (empty when unset); `data` is `{"tasks": [{"task_id", "status", "output"}...]}`.
- `UnknownTask` intents produce a clarification `ControllerOutput` with status `InputRequired` and `data.clarification_prompt`, and create no task.
- Pause/Cancel/Resume only touch persisted tasks: missing target → `Failed` output "task <id> not found"; resume re-dispatches the task.
- `TaskRun::frames` marks its single chunk `last_chunk`; the controller's `stream` clears intermediate flags so only the terminal chunk is last.
- `Planner::plan` ids are deterministic (`<session>_task_<seq>`), which the state and streaming tests rely on; `set_next_seq` rehydrates the counter from persisted state.
- `enable_task_persistence` defaults to `true` (Python's default is `false`) because the turn loop always saves; setting it to `false` makes `save_state` a no-op while in-memory state still accumulates.
- `enable_intent_recognition = false` bypasses the recognizer and yields one `default` create-task intent per input event.
- `max_intents`, `max_tasks`, `event_queue_size`, `task_timeout_ms`, `default_task_type`, `intent_type_list`, `default_response`, `suppress_completion_signal`, and `enable_task_persistence` are live; `max_concurrent_tasks`, `schedule_interval_ms`, `event_timeout_ms`, `history_limit`, `expose_reasoning`, `intent_confidence_threshold`, and `stream_first_frame_timeout_ms` are carried for parity but unused (no background scheduler, no context engine, no stream host).
- `legacy::schema::SourceType::Agent` keeps Python's `"single_agent"` wire value; `EventPriority` maps to Python's integer 1–4 via `EventPriority::value`.
- Legacy types must never be passed into `schema::Task`/`schema::Event` (or vice versa): the two families are unrelated and there is no converter.
- `LLMController::select` propagates model transport errors but never errors on a malformed answer (first executor wins); the `TaskRouter` impl delegates to it and wraps it in `Some`.

## Deferred Python Behavior

- The background scheduler: no scheduling loop, no concurrency (`max_concurrent_tasks`), no polling (`schedule_interval_ms`), no event timeout, no pause/cancel of a running executor (`can_pause`/`can_cancel`/`pause`/`cancel`), no `asyncio` task registry.
- `TaskManager`'s filters, priority index, parent/child index promotion, `pop_task`, dependency graph (`dependencies`/`dependents` live only on the legacy `LegacyTask`), and `TaskManagerState` migration.
- The event queue's pub/sub topics, `MessageQueueInMemory` back-pressure, and `publish_event`/`publish_event_async`; the port models the queue as a bounded `VecDeque` drained by the same turn.
- The LLM intent toolkits (`IntentToolkits` tool schemas, confidence thresholding, tool-result feedback) and the intent-detector prompt history (`chat_history`, `enable_history`, `category_info`, `example_content`).
- The `AbilityManager`/`ContextEngine` injection into request handlers, session stream writes from `invoke` (frames are returned on the output and emitted by `stream`), and telemetry/tracing spans.
- Group routing (`set_group`, `send_to_agent`, `publish`), per-conversation `TaskQueue` with in-flight cancellation, and the `Message*` legacy aliases.
- `agent_reasoner.py` (an empty artifact in Python), `ReasonerConfig`/`PlannerConfig`/`ProactiveIdentifierConfig`/`ReflectorConfig`, and `WorkflowController`'s LLM workflow selection (that lives in `crate::workflow_agent`).

## Test Strategy

- Inline `#[cfg(test)]` modules in every file, offline and deterministic: 125 controller tests (`schema` 35, `intent` 21, `executors` 23, `base` 13, `legacy` 28, `config` 5).
- `base` tests run the loop end to end: two executors routed by `LLMController` (model called once), the single-executor shortcut (no model call), an unregistered task type (`Failed` + `default_response`), persistence across two turns plus a fresh controller restore, streaming shape, lifecycle/queue bounds, the three `EventHandler` outcomes, `TaskInteraction`, and pause/cancel intents.
- `executors` tests cover the registry, sequential dispatch order, unknown task type, `max_tasks`, timeouts, executor errors, `ControllerAgent` answer/error/interrupt mapping, `WorkflowController::map_output` plus one real workflow run, and `LLMController` selection/fallback.
- `intent` tests cover rule matching/dedupe, the default intent, tolerant LLM parsing (fences, single object, garbage, unknown kind, truncation, model failure), and planner ids/metadata/budgets/overrides.
- `legacy` tests cover wire-shape round-trips, query extraction, detector parsing/classification, the planner stub, and `IntentDetectionController` invoke/stream delegation.
- Tests use `openjiuwen_test_support::ScriptedModel`, stub executors/handlers, and `AgentSession`; no network, no wall-clock assertions.

## Routing Table

- `./schema/` → payload frames, events, intents, tasks, controller output (`schema/CONTEXT.md`)
- `./config.rs` → `ControllerConfig`, `DefaultResponse`
- `./intent.rs` → `IntentRecognizer`, `IntentRule`, `RuleIntentRecognizer`, `LlmIntentRecognizer`, `Planner`
- `./executors/` → executor seam/registry/scheduler and the built-in agent, workflow, and LLM executors (`executors/CONTEXT.md`)
- `./base/` → `Controller`, `BaseController`, `EventHandler`, persisted task state, the turn loop (`base/CONTEXT.md`)
- `./legacy/` → legacy value objects, `IntentDetector`, `LegacyPlanner`, `IntentDetectionController` (`legacy/CONTEXT.md`)
- `./mod.rs` → module declarations, crate-level re-exports, and deviation notes
