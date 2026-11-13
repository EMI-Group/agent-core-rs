# checkpoint

## Intent

Checkpoint persistence: the `Checkpointer` contract, the checkpoint data model, Python-compatible key layout, configuration/factory, and the memory/file/sqlite/redis backends.
Ports `openjiuwen/core/session/checkpointer/` (v0.1.20): `base.py`, `factory.py`, `inmemory.py`, `persistence.py` (sqlite/shelve), and the Redis backend, which Python drives through the `redis` client library.
The Redis backend is self-contained and offline-buildable: it speaks RESP2 over `tokio::net::TcpStream` through the crate's own `resp` client, so no Redis crate is a dependency.
Checkpoint payloads are JSON, so Python's pickle files remain out of scope.

## API Surface

- `Checkpointer` trait — `save(Checkpoint) -> Result<String>`, `load(session_id, checkpoint_id)`, `latest(session_id)` (provided), `list(session_id) -> Vec<CheckpointMeta>`, `delete`, `release(session_id)`, `session_exists(session_id)`.
- `Checkpoint` — `session_id`, `checkpoint_id`, `namespace`, `entity_id`, `created_at`, `global_state`, `component_state`, `updates`, `trace: Option<TracePosition>`, `metadata`; builders `new`, `with_checkpoint_id`, `with_global_state`, `with_component_state`, `with_updates`, `with_trace`, `with_metadata`.
- `CheckpointMeta` — `session_id`, `checkpoint_id`, `namespace`, `entity_id`, `created_at`.
- `CheckpointNamespace` — `Agent`, `AgentTeam`, `Workflow`, `WorkflowGraph` with wire values `agent`, `agent-team`, `workflow`, `workflow-graph`; constants `SESSION_NAMESPACE_AGENT`, `SESSION_NAMESPACE_AGENT_TEAM`, `SESSION_NAMESPACE_WORKFLOW`, `WORKFLOW_NAMESPACE_GRAPH`.
- `build_key(parts)` (`:`-joined) and `build_key_with_namespace(session_id, namespace, entity_id, suffixes)`.
- Backends: `MemoryCheckpointer` (`new()`, `shared()` process-wide singleton, `len(session_id)`, `is_empty()`, `clear()`), `FileCheckpointer` (`new(root)`, JSON per checkpoint, atomic write/rename), `SqliteCheckpointer` (feature `sqlite`; `open(path).await` creating the parent directory, `in_memory().await` for a single-connection in-memory database; table `session_checkpoints`, one upsert per save), `RedisCheckpointer` (always compiled).
- `RedisCheckpointer::connect(url)` / `connect_with_options(RedisOptions)` dials eagerly (bad URL, `rediss://`, or an unreachable server fails at construction) and exposes `options()`; `RedisOptions { url, prefix, ttl: Option<Duration>, database: Option<u32> }` is serde-serializable, defaults `prefix` to `DEFAULT_REDIS_KEY_PREFIX = "openjiuwen:session"`, sends `SELECT` for `database`, and its `Debug` output masks the URL password.
- Redis key layout: one JSON document per checkpoint at `{prefix}:session:{session_id}:checkpoint:{checkpoint_id}` (the document shape `FileCheckpointer` writes), a per-session id set at `{prefix}:session:{session_id}:checkpoints`, and a global session set at `{prefix}:sessions`, so checkpointers with different prefixes on one server never see each other's data.
- Redis behaviour: `save` writes the document, applies `EXPIRE <seconds>` when `ttl` is set (a sub-second TTL is rounded up to one second) and then updates both index sets; `list` reads the id set, decodes each document, and prunes members whose document expired or was deleted out of band; `delete` removes the document plus its index member; `release` deletes every document of the session and both index entries; `session_exists` checks the session's id set.
- Redis connection handling: one `RespConnection` serialized behind a `tokio::sync::Mutex`; a transport failure marks it broken and the command is retried once on a fresh connection, while a server error reply stays `Error::Session` without reconnecting.
- `resp` — the dependency-free RESP2 client: `RespValue { Simple, Error, Integer, Bulk(Option<Vec<u8>>), Array(Option<Vec<RespValue>>) }` with `as_str`/`as_bytes`/`as_i64`/`as_array`/`is_nil`/`into_result`, `encode_command(&[&str]) -> Vec<u8>`, and `read_value(&mut impl AsyncBufRead)` (iterative nesting, 512 MiB bulk and 16M-element array limits); `RespConnection::connect(url)` parses `redis://[user[:password]@]host[:port][/db]` (port defaults to 6379, bracketed IPv6 hosts, password-only credentials as `redis://password@host`) and runs `AUTH` (with the username when present) plus `SELECT` for `/db`, then serves `command`, `command_bytes`, `select_database`, and `is_broken`.
- RESP URL errors: `rediss://` (TLS) is `Error::Unsupported`; a missing host, an invalid port, or a non-numeric `/db` is `Error::Validation`.
- `list` order is backend-defined: memory, file, and redis ascending by `(created_at, checkpoint_id)`, sqlite newest first.
- `CheckpointerConfig { kind, path, extra }`, `CheckpointerKind { InMemory, File, Sqlite, Redis }`, `create_checkpointer(&config).await -> Result<Arc<dyn Checkpointer>>`; `file` and `sqlite` require a `path` (`Error::Validation` otherwise), `sqlite` also requires the `sqlite` feature (`Error::Unsupported` without it), and `redis` requires a URL in `extra["url"]` (falling back to the rendered `path`; `Error::Validation` otherwise) while reading the optional `extra` keys `prefix` (string), `ttl_seconds` (unsigned integer), and `database` (unsigned integer).

## Constraints

- Checkpoint ids are generated with `openjiuwen_common::utils::id::uuid_hex()` when empty; `latest` sorts by `(created_at, checkpoint_id)` descending itself, so it is independent of the backend's `list` order.
- `session_exists` is prefix-scoped semantics per backend: memory/sqlite/redis exactly per session id (the Redis id set), file per session directory; `release` deletes every checkpoint of a session.
- Session ids and checkpoint ids are validated before being used in keys or paths: no `/`, `\`, NUL, or `..`; violations are `Error::Validation`.
- File and sqlite writes are atomic/crash-safe (temporary `.<uuid>.tmp` file + rename; single SQL statement per save); no blocking I/O on async tasks (use `tokio::fs` / `sqlx` / `tokio::net`).
- Undecodable stored data is `Error::Session`, never a panic: the file backend reports a corrupt JSON document, the sqlite backend a corrupt payload column, the Redis backend an undecodable document or an unexpected reply shape.
- Serialized checkpoints are JSON with snake_case field names and tolerate unknown fields; pickle compatibility is out of scope.
- Errors map to `Error::Session` (I/O, storage, Redis transport/server errors) and `Error::Validation` (ids/config/URLs); no panics on user input.
- Redis has no cluster, Sentinel, or TLS support and no pipelining: commands are serialized over one connection per checkpointer.
- No `unsafe`, no stdout logging, files ≤ ~1000 lines; backend code is split one file per backend.

## Routing Table

- `./mod.rs` → `Checkpointer` trait, `Checkpoint`, `CheckpointMeta`, namespaces, key builders, `CheckpointerConfig`, `create_checkpointer`
- `./memory.rs` → `MemoryCheckpointer`
- `./file.rs` → `FileCheckpointer` (filesystem/JSON backend, Python `shelve` analogue)
- `./sqlite.rs` → `SqliteCheckpointer` behind the `sqlite` feature
- `./redis.rs` → `RedisCheckpointer`, `RedisOptions`, key/index layout, TTL, single-connection reconnect
- `./resp.rs` → the dependency-free RESP2 codec and connection (`RespValue`, `encode_command`, `read_value`, `RespConnection`)
