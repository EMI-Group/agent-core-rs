# openjiuwen-agent controller legacy

## Intent

Python's pre-event-queue controller lineage: legacy value objects, an LLM category detector, the placeholder planner, and `IntentDetectionController`.
Ports `openjiuwen/core/controller/legacy/` (`event/event.py`, `task/task.py`, `reasoner/*`, `intent_detection_controller.py`).
The current pipeline ([`crate::controller::base`]) supersedes this lineage; the module is kept so Python-shaped consumers keep a Rust counterpart.

## API Surface

- `schema::LegacyTaskStatus` — `Pending`, `Running`, `Success`, `Failed`, `Cancelled`, `Interrupted` (+ `as_str`, `is_terminal`).
- `schema::LegacyTask` — dependency-graph task (`input: TaskInput`, `result: Option<LegacyTaskResult>`, `dependencies`, `dependents`, `parent_task_id`, `child_task_ids`, `group_id`, `level`), `UNDEFINED_TASK_TYPE`, builders, `set_agent_id`.
- `schema::TaskInput`, `TaskDependency`, `DependencyType`, `LegacyTaskResult` (`success`/`failed`).
- `schema::LegacyEventType` — `user_input`, `agent_response`, `agent_handoff`, `task_completed`, `task_interrupted`, `error`, `info`.
- `schema::EventPriority` (1–4), `SourceType` (`Agent` = `single_agent`), `EventSource`, `EventContent` (`get_query`, `extension`), `EventContext`, `LegacyEvent` (predicates, `set_correlation`, `set_conversation`, `to_dict`).
- `reasoner::IntentDetector` — `parse` (`{"result": n}`, fenced, 1-based → 0-based), `classify`, `async detect`, categories/default class/prompt.
- `reasoner::LegacyPlanner` — `plan(&LegacyEvent) -> LegacyTask` (one undefined pending task).
- `IntentDetectionController` — `BaseController` wrapper: `intent_detection`, `add_executor`, `set_router`, `set_event_handler`, `controller`, and `impl Controller`.

## Constraints

- Legacy types never mix with `schema::` types and there is no converter between the families.
- Wire names follow Python: `snake_case` statuses/events, `single_agent` for `SourceType::Agent`, integer priorities 1–4 via `EventPriority::value`.
- Legacy statuses are string-valued here (Python uses `auto()` ints, which have no stable wire form).
- `IntentDetector` never errors on a malformed model answer or a model failure: it falls back to `default_class` and logs.
- `IntentDetectionController` delegates everything to `crate::controller::base::BaseController`; it adds no second loop.

## Notes for Agents

- `IntentDetectionController::intent_detection` runs only the recognizer; task planning and dispatch happen through `Controller::invoke` on the same inner controller.
- Python's `IntentDetector` also generated tasks and matched workflows by description; that selection lives in `crate::workflow_agent`, so the port exposes classification only.
- The per-conversation `TaskQueue`, in-flight cancellation, group routing, config dataclasses (`IntentDetectionConfig`, `ReasonerConfig`, `PlannerConfig`, …), and the empty `agent_reasoner.py` artifact are not ported.

## Routing Table

- `./mod.rs` → module docs, re-exports, `IntentDetectionController`
- `./schema.rs` → legacy task/event value objects
- `./reasoner.rs` → `IntentDetector`, `LegacyPlanner`
