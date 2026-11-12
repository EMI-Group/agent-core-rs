# todo

## Intent

The session-scoped todo store and the four model-facing tools (`todo_create`, `todo_list`, `todo_get`, `todo_modify`), split by concern: ports `openjiuwen/harness/tools/todo.py` with Python names, argument shapes, message text and payload envelopes.
`TodoStore` persists one JSON array of `crate::schema::TodoItem` per session at `<todo dir>/<session_id>/todo.json`.

## API Surface

- `mod.rs` — the `//!` port/deviations docs, `todo_tool!` (generates each tool's `Tool` glue), module declarations, and the re-exports consumed by `crate::tools::mod`: `session_key`, `TodoStore`, `TodoCreateArgs`/`TodoCreateTool`, `TodoListTool`, `TodoGetArgs`/`TodoGetTool`, `TodoAction`, `TodoData`, `TodoModifyArgs`/`TodoModifyTool`.
- `store.rs` — `TODO_FILE`/`DEFAULT_SESSION`, `is_safe_session`, `normalize_session`, `session_key`, `TodoStore` with per-session async locks and atomic save.
- `schema.rs` — `card_schema` and the `$defs` inlining / status-enum rewriting helpers that build the four tool cards (Python's public shapes, per-tool `required` lists).
- `shared.rs` — `Outcome<T>`, item validation/parsing helpers (`required_create_field`, `task_id`, `parse_status`, `validate_todo_item(s)`, …), `render_message`, and the shared item/status constants.
- `create.rs` → `TodoCreateArgs`, `TodoCreateTool`; `list.rs` → `TodoListTool`; `get.rs` → `TodoGetArgs`, `TodoGetTool`; `modify.rs` → `TodoAction`, `TodoData`, `TodoModifyArgs`, `TodoModifyTool` and the action helpers.
- `tests.rs` — the create→list→get→modify parity table moved verbatim.

## Constraints

- Persisted JSON keys and the Python message text (including validation errors) are unchanged.
- Every create/modify action keeps the per-session lock from load to save; a save renames a temporary file over `todo.json`.
- Expected failures return a `success=false` `ToolPayload`, never `Err`; session id falls back to `default`.
- Files ≤1000 lines; the `todo_tool!` macro is the single place that generates the `Tool` impls.

## Routing Table

- `./mod.rs` → docs, `todo_tool!`, declarations, re-exports
- `./store.rs` → `TodoStore`, session keys, persistence and locks
- `./schema.rs` → card JSON-schema inlining and Python shape rewrites
- `./shared.rs` → `Outcome`, item validation/parsing, shared render/constants
- `./create.rs` → `todo_create`
- `./list.rs` → `todo_list`
- `./get.rs` → `todo_get`
- `./modify.rs` → `todo_modify` and its action handlers
- `./tests.rs` → `#[cfg(test)] mod tests`
