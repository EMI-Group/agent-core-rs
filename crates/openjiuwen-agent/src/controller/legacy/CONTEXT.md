# openjiuwen-agent controller legacy

## Intent

Python's pre-event-queue controller lineage: legacy value objects, an LLM category detector, the placeholder planner, and the `IntentDetectionController` wrapper.
Ports `openjiuwen/core/controller/legacy/` (`event/event.py`, `task/task.py`, `reasoner/*`, `intent_detection_controller.py`) and adds the per-conversation preemption, group routing, and handoff translation that Python kept around the legacy controller.
The current pipeline ([`crate::controller::base`]) supersedes this lineage; the module keeps a Rust counterpart for Python-shaped consumers.

## API Surface

- `IntentDetectionController` (`Clone`) — `new(card, config)`, `with_recognizer(card, config, recognizer)`, `controller() -> &BaseController`, `add_executor`/`set_router`/`set_event_handler` (each `-> &Self`), `set_group`/`clear_group`/`group`, `async send_to_agent`/`async publish`, `task_queue() -> TaskQueue`, `has_running(&str) -> bool`, `running_conversations() -> Vec<String>`, `async interrupt(&str) -> bool`, `async intent_detection(&Event) -> Result<Vec<Intent>>`, plus `impl Controller`.
- `schema::LegacyTaskStatus` — `Pending`, `Running`, `Success`, `Failed`, `Cancelled`, `Interrupted`, with `as_str` and `is_terminal` (`Interrupted` is not terminal).
- `schema::LegacyTask` — `agent_id`, `task_id`, `task_type`, `description`, `status`, `metadata`, `input: TaskInput`, `result: Option<LegacyTaskResult>`, `dependencies`, `dependents`, `parent_task_id`, `child_task_ids`, `group_id`, `level`, plus `UNDEFINED_TASK_TYPE = "undefined"`, `new`, `with_description`, `with_input`, `with_metadata`, `set_agent_id`, `is_terminal`.
- `schema::TaskInput`, `TaskDependency`, `DependencyType` (`sequential`/`parallel`/`conditional`/`data`), `LegacyTaskResult` (`success`/`failed`).
- `schema::LegacyEventType` — `user_input`, `agent_response`, `agent_handoff`, `task_completed`, `task_interrupted`, `error`, `info`.
- `schema::EventPriority` (1–4, `value()`), `SourceType` (`Agent` = `"single_agent"`), `EventSource`, `EventContent` (`get_query`, `extension`), `EventContext`.
- `schema::LegacyEvent` — `id`, `event_type`, `priority`, `source`, `content`, `context`, `metadata`, `receiver_id`, `custom_event_type`; predicates (`is_from_user`, `is_from_agent`, `is_task_related`, `is_workflow_related`), `set_correlation`, `set_conversation`, `get_query`, `with_receiver`, `with_custom_event_type`, `handoff_to_agent`, `routed_to`, `to_dict`.
- `reasoner::IntentDetector` — `new(model, categories)`, `with_default_class`, `with_prompt`, `categories`, `default_class`, `parse`, `classify`, `async detect`; `DEFAULT_CLASS = "unknown"` and `INTENT_DETECTION_PROMPT`.
- `reasoner::LegacyPlanner` — `new()`, `with_task_type`, `task_type`, `plan(&LegacyEvent) -> LegacyTask` (one pending, id-less task with `task_source = "planner"`).

## Constraints

- Legacy types never mix with `schema::` types and there is no converter between the families.
- Wire names follow Python: `snake_case` statuses/events, `single_agent` for `SourceType::Agent`, integer priorities 1–4 via `EventPriority::value`, and legacy statuses as strings (Python uses `auto()` ints with no stable wire form).
- `receiver_id` and `custom_event_type` are omitted from the wire form while unset and round-trip when set.
- `IntentDetector` never errors on a malformed model answer or a model failure: it warns and falls back to `default_class`.
- `IntentDetectionController` delegates all turn work to `BaseController` and adds no second loop.
- Streaming turns are never tracked: the returned `BoxStream` owns the turn, so no `AbortHandle` exists for a later turn to abort.
- No panics outside tests; a missing runtime only downgrades `invoke` (inline fallback).

## Notes for Agents

### Per-conversation preemption

- A conversation id comes from `inputs["conversation_id"]`, then `inputs["conversation"]` (first non-empty string), then the session id, then the literal `"default"`.
- `invoke` preempts the previous turn of the same conversation before doing anything else, then spawns the inner `Controller::invoke` under `catch_unwind` and registers the turn in the `TaskQueue` plus its `AbortHandle` in the handler map.
- Preemption aborts the registered handle, cancels the queue entry's token, awaits its `JoinHandle`, and unregisters the entry; `interrupt(conversation_id)` is exactly that and reports whether a live turn was stopped.
- A preempted `invoke` returns `Error::Cancelled`, or `Error::session("controller turn for conversation '<id>' panicked")` when the turn panicked; a normal turn cleans its own entries with id and start-time guards so a preempting turn's registration is not clobbered.
- `stream` preempts without awaiting (abort plus token cancel) and then returns `BaseController::stream` directly.
- Without a Tokio runtime, `invoke` logs and runs the inner turn inline, untracked and unpreemptable; `stream` has no such fallback because the base controller spawns.
- `task_queue` / `has_running` / `running_conversations` expose the preemption bookkeeping; the tracked `RunningTask::target_id` carries the configured default task type.

### Group routing and handoff

- Group methods delegate verbatim to `BaseController`; the router lives there and the wrapper keeps no group field.
- A missing group is `Error::config` with `"{owner}: Cannot send_to_agent('<id>'). Agent is not part of a group with a controller."` or `"{owner}: Cannot publish(). ..."`, the owner being the card name (or id when the name is empty).
- Handoff translation is a value-object API only: `handoff_to_agent` prefers `receiver_id` and otherwise reads the string `to_agent_id` extension from `content`, and `routed_to(agent_id)` compares that result.
- `with_receiver`/`with_custom_event_type` set their field; `custom_event_type` is metadata for a group implementation, not a routing selector here.
- Neither the group router nor the queue inspects `receiver_id`/`to_agent_id` today; `send_to_agent` takes the target id explicitly and forwards the event unchanged.

## Deferred Python Behavior

- `IntentDetector` category→task generation and workflow selection by category description; the detector reports only a category name, and workflow selection lives in `crate::workflow_agent`.
- `IntentDetectionConfig`'s chat history, `category_info`, `example_content`, and `enable_input` knobs.
- `ReasonerConfig`, `PlannerConfig`, `ProactiveIdentifierConfig`, `ReflectorConfig`, and the proactive identifier/reflector.
- The `Message*` backward-compatibility aliases of `event.py`.
- `legacy/reasoner/agent_reasoner.py`, an empty and unreferenced Python artifact.

## Test Strategy

- 42 offline tests: `mod.rs` 15, `schema.rs` 18, `reasoner.rs` 9.
- `mod.rs` covers delegation (invoke/stream/intent detection), chainable builders, group forwarding and typed errors with `clear_group`, the value-object re-exports, conversation-id resolution, the inline no-runtime path, same-conversation preemption, cross-conversation isolation, `interrupt` on idle and running conversations, residue-free failed turns, and stream preemption of an in-flight invoke.
- `schema.rs` covers wire round-trips for tasks and events, defaults and unknown fields, status strings and terminality, dependency/input defaults, priority integers, predicates and context setters, `get_query` precedence, extensions, source helpers, routing-field omission/round-trip, and handoff resolution precedence.
- `reasoner.rs` covers tolerant parsing (plain, fenced, numeric strings, out-of-range, malformed), classification fallback, configurable categories and prompt, model-driven detection, failure fallback, and the planner stub; it uses `ScriptedModel`.
- Doubles: `StubExecutor`, `SlowExecutor` (100 ms hold), `StubRecognizer`, `FailingRecognizer`, `StubRouter`, `RecordingGroup`, `StubHandler`; waits poll every 1 ms with a 5 s bound, and the only timings are the slow executor's hold and that bound.

## Routing Table

- `./mod.rs` → module docs, re-exports, `IntentDetectionController` and its preemption/group surface
- `./schema.rs` → legacy task/event value objects and handoff translation
- `./reasoner.rs` → `IntentDetector`, `LegacyPlanner`
