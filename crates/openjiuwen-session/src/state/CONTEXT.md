# state

## Intent

State stores and JSON path arithmetic for sessions.
Ports `openjiuwen/core/session/state/` (v0.1.20): `CommitState` lanes with per-node staging, the agent/team `StateCollection`, and `state/utils.py` nested-path helpers.

## API Surface

- `StateStore` trait — the storage seam every session uses: `get_global(key)`, `update_global(patch)`, `get_component(id)`, `update_component(id, patch)`, `components()`, `dump()`, `restore(global, components)`, `trace_state()`, `update_trace(patch)`.
- `WorkflowState` — Python `CommitState`: lanes `io_state`, `global_state`, `comp_state`, `workflow_state`, `trace_state`; staging per node id (`update`, `update_global`, `update_workflow`, `update_component`), `commit(Option<&str>)`, `rollback(node_id)`, `updates()`, `set_updates()`, `inputs()`, `outputs(node_id)`, `set_outputs`, `commit_user_inputs(inputs)`, `get_by_schema(schema)`, `node(node_id, parent_id)` (shares lanes), `dump()`, `workflow_only()`.
- `AgentState` — Python agent/team `StateCollection`: `global_state` + `agent_state` lanes, no staging; `dump()` = `{"global_state": …, "agent_state": …}`.
- `paths` — `get_value`, `set_value`, `remove_value`, `get_by_prefix`, `apply_patch` (Python `update_dict`: object-vs-object recurse, `null` deletes after writes, everything else replaces), `is_ref_path`, `extract_origin_key`.
- Lane constants: `IO_STATE_KEY`, `IO_STATE_UPDATES_KEY`, `GLOBAL_STATE_KEY`, `GLOBAL_STATE_UPDATES_KEY`, `COMP_STATE_KEY`, `COMP_STATE_UPDATES_KEY`, `WORKFLOW_STATE_KEY`, `WORKFLOW_STATE_UPDATES_KEY`, `TRACE_STATE_KEY`, `AGENT_STATE_KEY`, `DEFAULT_NODE_ID`, `DEFAULT_WORKFLOW_ID`.

## Constraints

- Lanes and staging are written under one `parking_lot::RwLock` per store; no lock is held across `.await` (all methods are synchronous).
- Component ids are nested paths (`a.b` → `{"a": {"b": …}}`), matching Python `executable_id` nesting.
- Patches must be JSON objects; anything else is `Error::Validation`. Arrays are replaced, never element-merged.
- `dump()` emits the Python lane keys (`io_state`, `global_state`, `comp_state`, `workflow_state`, `trace_state`); `global_state` is omitted from `get_state()` only for workflow-only stores, never from `dump()`.
- Child/node stores share lane objects by reference (Python `create_node_state` is O(1)); the trace lane is a plain JSON value.
- No `unsafe`, no stdout logging, files ≤ ~1000 lines.

## Routing Table

- `./mod.rs` → `StateStore` trait, lane constants, re-exports
- `./paths.rs` → nested-path get/set/delete, prefix reads, `apply_patch`, `${ref}` helpers
- `./workflow.rs` → `WorkflowState` (four lanes, staging, commit/rollback, inputs/outputs, schema projection)
- `./agent.rs` → `AgentState` (global + agent lanes)
