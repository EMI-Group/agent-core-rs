# checkpoint

## Intent

Checkpoint persistence: the `Checkpointer` contract, the checkpoint data model, Python-compatible key layout, configuration/factory, and memory/file/sqlite backends.
Ports `openjiuwen/core/session/checkpointer/` (v0.1.20): `base.py`, `factory.py`, `inmemory.py`, `persistence.py` (sqlite/shelve). Redis and other external backends remain out of scope (trait seam only).

## API Surface

- `Checkpointer` trait — `save(Checkpoint) -> Result<String>`, `load(session_id, checkpoint_id)`, `latest(session_id)` (provided), `list(session_id) -> Vec<CheckpointMeta>`, `delete`, `release(session_id)`, `session_exists(session_id)`.
- `Checkpoint` — `session_id`, `checkpoint_id`, `namespace`, `entity_id`, `created_at`, `global_state`, `component_state`, `updates`, `trace: Option<TracePosition>`, `metadata`; builders `new`, `with_checkpoint_id`, `with_global_state`, `with_component_state`, `with_updates`, `with_trace`, `with_metadata`.
- `CheckpointMeta` — `session_id`, `checkpoint_id`, `namespace`, `entity_id`, `created_at`.
- `CheckpointNamespace` — `Agent`, `AgentTeam`, `Workflow`, `WorkflowGraph` with wire values `agent`, `agent-team`, `workflow`, `workflow-graph`; constants `SESSION_NAMESPACE_AGENT`, `SESSION_NAMESPACE_AGENT_TEAM`, `SESSION_NAMESPACE_WORKFLOW`, `WORKFLOW_NAMESPACE_GRAPH`.
- `build_key(parts)` (`:`-joined) and `build_key_with_namespace(session_id, namespace, entity_id, suffixes)`.
- Backends: `MemoryCheckpointer` (`new()`, `shared()`), `FileCheckpointer` (`new(root)`, JSON per checkpoint, atomic write/rename), `SqliteCheckpointer` (`open(path).await`, feature `sqlite`, table `session_checkpoints`).
- `CheckpointerConfig { kind, path, extra }`, `CheckpointerKind { InMemory, File, Sqlite }`, `create_checkpointer(&config) -> Result<Arc<dyn Checkpointer>>`.

## Constraints

- Checkpoint ids are generated with `openjiuwen_common::utils::id::uuid_hex()` when empty; `latest` orders by `created_at` then `checkpoint_id` descending.
- `session_exists` is prefix-scoped semantics per backend: memory/sqlite exactly per session id, file per session directory; `release` deletes every checkpoint of a session.
- Session ids and checkpoint ids are validated before being used in keys or paths: no `/`, `\`, NUL, or `..`; violations are `Error::Validation`.
- File and sqlite writes are atomic/crash-safe (temp file + rename; single SQL statement per save); no blocking I/O on async tasks (use `tokio::fs` / `sqlx`).
- Serialized checkpoints are JSON with snake_case field names and tolerate unknown fields; pickle compatibility is out of scope.
- Errors map to `Error::Session` (I/O, storage) and `Error::Validation` (ids/config); no panics on user input.
- No `unsafe`, no stdout logging, files ≤ ~1000 lines; backend code is split one file per backend.

## Routing Table

- `./mod.rs` → `Checkpointer` trait, `Checkpoint`, `CheckpointMeta`, namespaces, key builders, `CheckpointerConfig`, `create_checkpointer`
- `./memory.rs` → `MemoryCheckpointer`
- `./file.rs` → `FileCheckpointer` (filesystem/JSON backend, Python `shelve` analogue)
- `./sqlite.rs` → `SqliteCheckpointer` behind the `sqlite` feature
