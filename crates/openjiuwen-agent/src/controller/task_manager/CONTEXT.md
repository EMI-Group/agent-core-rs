# openjiuwen-agent task_manager

## Intent

Shared, indexed task store: one `TaskManager` holds the tasks of every session plus the priority and parent/child indexes that keep selection, removal, status updates, and priority changes cheap.
Ports `openjiuwen/core/controller/modules/task_manager.py` onto `crate::controller::schema::Task` as a synchronous, lock-guarded store.
It is a leaf: the turn loop, the background scheduler, and the `Reasoner` drive it, and it never calls back into `base`, `executors`, `intent`, or `reasoner`.

## API Surface

- `TaskManager` (`mod.rs`) — `#[derive(Clone, Default)]` handle over `Arc<Mutex<Indexes>>` plus `Arc<RwLock<ControllerConfig>>` and `Arc<RwLock<Option<SubmittedCallback>>>`; clones share the store, the config, and the submitted listener, and `new()` is `Default::default()`.
- Construction and configuration: `new`, `with_config`, `config`, `set_config`, `set_on_task_submitted(Arc<dyn Fn() + Send + Sync>)`, `clear_on_task_submitted`.
- Write path: `add_task(Task) -> Result<()>`, `add_tasks(Vec<Task>) -> Result<()>`, `update_task(Task) -> bool`, `update_tasks(Vec<Task>) -> bool`, `remove_task(Option<&TaskFilter>) -> Result<Vec<Task>>`, `pop_task(Option<&TaskFilter>) -> Result<Vec<Task>>`, `clear`.
- Read path: `task(&str) -> Option<Task>`, `tasks() -> Vec<Task>`, `session_tasks(&str) -> Vec<Task>`, `get_task(Option<&TaskFilter>) -> Result<Vec<Task>>`, `get_child_task(&str) -> Vec<Task>`, `len`, `is_empty`, `next_created_seq() -> u64` (never `0`).
- Execution path: `claim_task(&str) -> bool` (the atomic `Submitted -> Working` guard), `update_task_status(task_id, status, with_children, recursive, error_message)` and `set_priority(task_id, priority, with_children, recursive)`; the last two return `()` and are silent no-ops for an unknown id.
- Liveness: `has_active_tasks()` and `session_has_active_tasks(&str)` — active is `TaskStatus::is_active` (`Submitted` or `Working`; `Paused`, `Waiting`, `InputRequired` and terminal statuses are not).
- Persistence: `snapshot_session(&str) -> TaskManagerState` and `restore_session(&str, Option<&TaskManagerState>) -> Result<()>`, where `None` clears the session's slice.
- `TaskFilter` (`filter.rs`) — selectors `task_id: Option<TaskIdSelector>`, `session_id`, `user_id`, `priority: Option<PrioritySelector>`, `status: Option<TaskStatus>`, plus `with_children: bool` and `is_root: bool`; builders `new`, `by_id`, `by_ids`, `by_session`, `by_user`, `by_status`, `by_priority`, `highest_priority`, `roots`, `with_children`, `include_children`, `roots_only`; checks `validate`, `has_selector`, and the crate-private `has_primary_selector`.
- `TaskFilter` serde — `#[serde(rename_all = "snake_case", default)]`; the selector `Option`s are skipped when `None`, while `with_children`/`is_root` are always emitted, so `by_id("t1")` serializes as `{"task_id":"t1","with_children":false,"is_root":false}`.
- `TaskIdSelector` — `#[serde(untagged)]` `One(String)`/`Many(Vec<String>)` with `From<&str>`, `From<String>`, `From<Vec<String>>`, `From<[&str; N]>`, `to_vec`, `matches`, `len`, `is_empty`; only `Many(vec![])` is empty (an empty `One` counts as one id).
- `PrioritySelector` — `Level(i32)` or `Highest`, with `From<i32>`, `level()`, `is_highest()`; hand-written serde accepts an integer, a numeric string (`"3"`), or `"highest"` case-insensitively and rejects floats and out-of-range integers.
- `TaskManagerState` (`state.rs`) — public `tasks: Vec<Task>` in stored order and `next_seq: u64`, plus `new`, `len`, `is_empty`, `task_ids`.
- `Indexes` (`index.rs`, `pub(super)`) — the `task_id -> Task` map, the `BTreeMap<i32, Vec<String>>` priority index, both parent/child directions, the `BTreeSet` roots, and the sequence counter; `DEFAULT_FAILURE_MESSAGE = "Task execution failed"` lives beside it.

## Constraints

- Every method is synchronous and holds the lock only for its own duration; no `await` holds it, and the submitted listener fires after release.
- Returned tasks are owned deep copies, so mutating them never affects the store.
- `add_task`/`add_tasks` reject a duplicate `task_id` even across sessions with `"{task_id} already exists!"`; a batch is validated first, then inserted, so a duplicate stores nothing.
- `restore_session` rejects a same-session id already owned by another session with `"Task ID already belongs to another session: {task_id}"` before mutating anything.
- `TaskFilter::validate` is the pydantic validator's analogue and must be called explicitly: `get_task`, `remove_task`, and `pop_task` call it, construction does not; an empty filter is `Error::validation("At least one filter parameter (task_id, session_id, user_id, priority, status, or is_root) must be provided")`.
- `remove_task(None)`/`pop_task(None)` are `Error::validation("task_filter cannot be None in {method}")`; `get_task(None)` returns every task.
- `"highest"` is a `pop_task`-only marker: `get_task` reports `"Priority 'highest' is not supported in get_task, use pop_task instead"` and `remove_task` its own message.
- Errors are `openjiuwen_common::error::{Error, Result}` and every one of them is `Error::validation`; no panics, `unwrap`, or `expect` outside tests.
- Public types are `Serialize + Deserialize` with `#[serde(rename_all = "snake_case", default)]` and tolerate unknown fields.
- Files stay under ~1000 lines (`index.rs` is the largest at 815) and add no dependency beyond the crate's serde/serde_json/parking_lot.

## Design Decisions

- **Selection is a union of primary selectors, then narrowing**: `task_id`, `session_id`, the priority level, and `roots` union into one candidate set; `status`/`user_id` narrow it (`user_id` reads `metadata["user_id"]`, so a task without it never matches); a status/user-only filter scans the whole store.
- **`with_children` expands after narrowing**: every transitive descendant of each match is appended without re-checking `status`/`user_id`, so `by_status(Working).include_children()` can return non-`Working` children.
- **Ordering is Rust-defined**: results sort by the Rust-only `Task::created_seq` ascending with `task_id` as the tiebreak, replacing Python's set iteration; `add_task` assigns `next_created_seq()` when a task carries `0`, and any explicit sequence also advances the counter.
- **`update_task` keeps the stored sequence** when the incoming `created_seq` is `0` (a JSON round-trip drops it), moves the priority and hierarchy entries with the task, and never fires the submitted listener.
- **The submitted listener fires** on `add_task`, once per `add_tasks` batch that contains a `Submitted` task, and from `update_task_status` only for a transition to `Submitted` that changed at least one task.
- **`update_task_status`** stores `error_message` or `"Task execution failed"` only for `Failed`, leaves `error_message` untouched for other statuses, and treats `with_children` and `recursive` as independent knobs (`recursive` is ignored without `with_children`).
- **The priority index holds only live levels**: removing or moving a task drops an emptied level, so `pop_task`'s `"highest"` never resolves to a stale empty level (Python keeps the empty list).
- **Removal promotes orphans**: after a matched task and its requested children are dropped, every surviving child becomes a root with `parent_task_id` cleared; grandchildren keep their surviving parent.
- **`clear` keeps the counter monotonic**, so ordering never repeats.
- **Persistence is per session and atomic**: `TaskManagerState` serializes as exactly `{"tasks": [...], "next_seq": n}`; restore filters by `session_id`, rejects a collision, reindexes, and reassigns `created_seq` in stored order, so restored tasks sort after the other sessions' existing tasks.
- **On the session wire `next_seq` is the planner's id sequence**: `base::save_state` overwrites the snapshot's `next_seq` with `Planner::next_seq()` before writing `__controller_tasks__`, so a restore rehydrates the planner, not just the manager counter.
- **`config`/`set_config` are inert storage**: the manager exposes the controller configuration but applies no scheduling defaults; the planner and the scheduler decide priorities and admission.

## Test Strategy

- 46 offline unit tests beside the code, all deterministic with no clock or network.
- `filter.rs` (7) covers the validation message, every single-selector filter, all wire shapes (`{"task_id": "t1"}`, `{"task_id": ["t1","t2"]}`, `{"priority": 2}`, `{"priority": "highest"}`, `{"is_root": true}`), unknown-field tolerance, round-trips, and the selector conversions.
- `state.rs` (4) pins the two-key persisted shape, the `{}` defaults, unknown-field tolerance with stored order preserved, and the JSON round-trip.
- `index.rs` (12) covers sequence assignment and index building, update moving metadata, sequence preservation on a wire copy, orphan promotion on removal, empty-level cleanup, one-shot claiming, status/priority bookkeeping, deterministic selection with child expansion, monotonic `clear`, per-session snapshot/restore (restored tasks re-sequenced after existing ones), collision rejection leaving the store untouched, and the active-task helpers.
- `tests.rs` (23) drives the public surface: creation-order queries, duplicate rejection across sessions, all-or-nothing batches, every submitted-listener path, update reporting and index moves, deep-copy isolation, filter unions and narrowing, transitive children and roots, empty and `None` filters, `highest` as pop-only, `get_child_task`, claim atomicity, status/priority propagation, JSON snapshot→restore, session-scoped restore with `None` clearing, collision errors, active helpers, handle sharing, and the config round-trip.
- Orphan promotion and empty-level cleanup are covered by `index.rs`, not by `tests.rs`.

## Routing Table

- `./mod.rs` → `TaskManager` public surface, submitted listener, removal path
- `./filter.rs` → `TaskFilter`, `TaskIdSelector`, `PrioritySelector`
- `./state.rs` → `TaskManagerState` wire record
- `./index.rs` → `Indexes`: map, priority/hierarchy indexes, selection, restore
- `./tests.rs` → behaviour tests for the manager surface
