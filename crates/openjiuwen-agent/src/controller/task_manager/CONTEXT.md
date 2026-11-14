# openjiuwen-agent task_manager

## Intent

Shared task store for the controller stack: one `TaskManager` holds every session's tasks plus the priority and parent/child indexes that make selection, removal, status updates, and priority changes cheap.
Ports `openjiuwen/core/controller/modules/task_manager.py` onto `crate::controller::schema::Task` as an idiomatic, synchronous, lock-guarded store.
It is a leaf: the planner, the scheduler, and the turn loop call it, and it never calls back into `base`, `executors`, or `intent`.

## API Surface

- `TaskManager` (`mod.rs`) — cloneable handle over `Arc<parking_lot::Mutex<Indexes>>`; clones share one store.
- Construction and configuration: `new`, `with_config`, `config`, `set_config`, `set_on_task_submitted`, `clear_on_task_submitted`.
- Write path: `add_task`, `add_tasks`, `update_task`, `update_tasks`, `remove_task`, `pop_task`, `clear`.
- Read path: `task`, `tasks`, `session_tasks`, `get_task`, `get_child_task`, `len`, `is_empty`, `next_created_seq`.
- Execution path: `claim_task` (atomic `Submitted -> Working`), `update_task_status`, `set_priority`.
- Liveness: `has_active_tasks`, `session_has_active_tasks`.
- Persistence: `snapshot_session` produces a `TaskManagerState`; `restore_session` replaces one session's slice (`None` clears it).
- `TaskFilter` (`filter.rs`) — selectors `task_id`, `session_id`, `user_id`, `priority`, `status`, plus `with_children` and `is_root`; builders `new`, `by_id`, `by_ids`, `by_session`, `by_user`, `by_status`, `by_priority`, `highest_priority`, `roots`, `with_children`, `include_children`, `roots_only`; checks `validate`, `has_selector`.
- `TaskIdSelector` (`filter.rs`) — `#[serde(untagged)]` `One(String)`/`Many(Vec<String>)` with `From<&str>`/`From<String>`/`From<Vec<String>>`/`From<[&str; N]>` and `to_vec`/`matches`/`len`/`is_empty`.
- `PrioritySelector` (`filter.rs`) — `Level(i32)` or `Highest`, hand-written serde (integer or `"highest"`), `level`, `is_highest`.
- `TaskManagerState` (`state.rs`) — public `tasks: Vec<Task>` (creation order) and `next_seq: u64`, plus `new`, `len`, `is_empty`, `task_ids`.
- `Indexes` (`index.rs`, crate-private) — the `task_id -> Task` map, the priority index, both parent/child directions, the root set, and the sequence counter.

## Constraints

- Every method is synchronous and takes the lock only for the duration of the call; no `await` holds it, and the submitted listener fires after release.
- Returned tasks are owned deep copies, so mutating them never affects the store.
- No panics, `unwrap`, or `expect` outside tests: a missing task is `None`/`false`/`Ok(vec![])`, never an error.
- Errors are `openjiuwen_common::error::{Error, Result}`; malformed filters, unknown duplicates, `None` filters, and session-id collisions are `Error::validation` with Python's messages.
- Public types are `Serialize + Deserialize` with `#[serde(rename_all = "snake_case", default)]` and tolerant of unknown fields.
- Files stay under ~1000 lines; no dependency beyond the crate's existing serde/serde_json/parking_lot.

## Design Decisions

- **Ordering is Rust-defined**: results sort by the Rust-only `Task::created_seq` with `task_id` as the tiebreak, replacing Python's set iteration; `add_task` assigns the sequence when a task carries `0`.
- **The priority index holds only live levels**: removing or moving a task drops an emptied level, so `pop_task`'s `"highest"` never resolves to a stale empty level (Python keeps the empty list).
- **`add_tasks` is all-or-nothing**: the batch is validated first, then inserted, so a duplicate stores nothing where Python inserts until the duplicate raises.
- **`update_task` keeps the stored sequence** when the incoming task's `created_seq` is `0` (a JSON round-trip drops it) and moves the priority/hierarchy entries with the task; it never fires the submitted listener, matching Python.
- **`update_task_status` fires the submitted listener** only when the new status is `Submitted` and at least one task existed; a `Failed` transition stores `error_message` or `"Task execution failed"` and leaves `error_message` alone for other statuses.
- **Validation lives in the manager**: Python's pydantic validator is reproduced by `TaskFilter::validate`, called by `get_task`, `remove_task`, and `pop_task`.
- **Resolution order**: primary selectors (`task_id`, session, priority level, roots) union into one candidate set, `status`/`user_id` narrow it (`user_id` reads `metadata["user_id"]`, so a task without it never matches), `with_children` appends every transitive descendant, and only a status/user-only filter scans the whole store.
- **`"highest"` is a pop-only marker**: `get_task` reports `Priority 'highest' is not supported in get_task, use pop_task instead`, `remove_task` reports its own message, and `pop_task` resolves the largest live level (an empty store pops nothing).
- **Removal promotes orphans**: after a matched task and its requested children are dropped, every surviving child becomes a root with `parent_task_id` cleared.
- **Persistence is per session and atomic**: `TaskManagerState` is the controller's existing wire shape (`{"tasks": [...], "next_seq": n}`); restore filters the state by `session_id`, rejects a collision before mutating, reindexes, and reassigns `created_seq` in stored order, then sets the counter to the maximum of the current, stored, and assigned sequences.
- **`clear` keeps the counter monotonic**, so ordering never repeats.
- **`config`/`set_config` are inert storage**: the manager exposes the controller configuration but does not apply scheduling defaults; the planner decides task priorities.

## Test Strategy

- 46 offline unit tests beside the code, all deterministic with no clock or network.
- `filter.rs` (7) covers validation messages, every single-selector filter, all wire shapes (`{"task_id": "t1"}`, `{"task_id": ["t1","t2"]}`, `{"priority": 2}`, `{"priority": "highest"}`, `{"is_root": true}`), unknown fields, round-trips, and the selector conversions.
- `state.rs` (4) pins the persisted shape, defaults, unknown-field tolerance, and ordering through `serde_json`.
- `index.rs` (12) covers sequence assignment, priority/hierarchy index maintenance, sequence preservation on update, orphan promotion, empty-level cleanup, one-shot claiming, status/priority bookkeeping, deterministic selection, monotonic `clear`, per-session snapshot/restore, and collision rejection.
- `tests.rs` (23) drives the manager surface: add/duplicate/atomic batches, the submitted listener, update reporting, deep-copy isolation, filter unions and narrowing, child expansion, root filters, filter validation, `highest` in `get_task`/`remove_task`/`pop_task`, `None` filters, child promotion on remove/pop, `get_child_task`, claim atomicity, status/priority propagation, JSON snapshot round-trip, per-session restore, collision error, active-task helpers, handle sharing, config round-trip, and monotonic clearing.

## Routing Table

- `./mod.rs` → `TaskManager` public surface, submitted listener, removal path
- `./filter.rs` → `TaskFilter`, `TaskIdSelector`, `PrioritySelector`
- `./state.rs` → `TaskManagerState` wire record
- `./index.rs` → `Indexes`: map, priority/hierarchy indexes, selection, restore
- `./tests.rs` → behaviour tests for the manager surface
