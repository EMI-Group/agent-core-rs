# checkpoint

## Intent

Checkpoint persistence: the `Checkpointer` contract, the checkpoint data model, Python-compatible key layout, configuration/factory, the memory/file/sqlite/redis backends, the RESP2 transport those last two share, and the pickle codec used to interop with Python-written payloads.
Ports `openjiuwen/core/session/checkpointer/` (v0.1.20): `base.py`, `factory.py`, `inmemory.py`, `persistence.py` (sqlite/shelve), and the Redis backend, which Python drives through the `redis` client library.
The Redis backend is self-contained and offline-buildable: it speaks RESP2 over a plaintext `tokio::net::TcpStream` or a `tokio-rustls` stream through the crate's own `resp` client, so no Redis crate is a dependency.
Checkpoint payloads are JSON, so the crate never guesses a storage format; `pickle` is an explicit interop codec for Python streams, not a backend format.

## API Surface

- `Checkpointer` trait — `save(Checkpoint) -> Result<String>`, `load(session_id, checkpoint_id)`, `latest(session_id)` (provided), `list(session_id) -> Vec<CheckpointMeta>`, `delete`, `release(session_id)`, `session_exists(session_id)`.
- `Checkpoint` — `session_id`, `checkpoint_id`, `namespace`, `entity_id`, `created_at`, `global_state`, `component_state`, `updates`, `trace: Option<TracePosition>`, `metadata`; builders `new`, `with_checkpoint_id`, `with_global_state`, `with_component_state`, `with_updates`, `with_trace`, `with_metadata`.
- `CheckpointMeta` — `session_id`, `checkpoint_id`, `namespace`, `entity_id`, `created_at`.
- `CheckpointNamespace` — `Agent`, `AgentTeam`, `Workflow`, `WorkflowGraph` with wire values `agent`, `agent-team`, `workflow`, `workflow-graph`; constants `SESSION_NAMESPACE_AGENT`, `SESSION_NAMESPACE_AGENT_TEAM`, `SESSION_NAMESPACE_WORKFLOW`, `WORKFLOW_NAMESPACE_GRAPH`.
- `build_key(parts)` (`:`-joined) and `build_key_with_namespace(session_id, namespace, entity_id, suffixes)`.
- Backends: `MemoryCheckpointer` (`new()`, `shared()` process-wide singleton, `len(session_id)`, `is_empty()`, `clear()`), `FileCheckpointer` (`new(root)`, JSON per checkpoint, atomic write/rename), `SqliteCheckpointer` (feature `sqlite`; `open(path).await` creating the parent directory, `in_memory().await` for a single-connection in-memory database; table `session_checkpoints`, one upsert per save), `RedisCheckpointer` (always compiled).
- `RedisCheckpointer::connect(url)` / `connect_with_options(RedisOptions)` dials eagerly (a bad URL, an unreachable server, or a failed TLS handshake fails at construction) and exposes `options()`; `RedisOptions { url, prefix, ttl: Option<Duration>, database: Option<u32>, tls_ca_file: Option<String>, io_timeout: Option<Duration> }` is serde-serializable, defaults `prefix` to `DEFAULT_REDIS_KEY_PREFIX = "openjiuwen:session"`, sends `SELECT` for `database`, and its `Debug` output masks the URL password.
- `RedisOptions::effective_io_timeout()` returns `io_timeout` or `DEFAULT_REDIS_IO_TIMEOUT` = 60 s (the workspace HTTP request default); `RespConnection::connect_with_ca_file_and_timeout` bounds dialing plus the TLS handshake with it and `command_bytes` bounds each write-and-reply exchange, an elapsed deadline is `Error::Timeout` with the connection marked broken, and `RedisCheckpointer::exchange` discards (never re-pools) a timed-out connection while taking the pool mutex only to take/return the handle.
- `connect` picks the transport from the scheme: `redis://` is plaintext, `rediss://` is TLS, and TLS is never skipped in favour of a clear-text fallback.
- `RedisOptions::from_config(&CheckpointerConfig)` is the declarative path used by `create_checkpointer`: it reads `extra["url"]` (with the rendered `path` as fallback) then `prefix`, `ttl_seconds`, `database`, `tls_ca_file` (accepting `tls-ca-file` as an alias and preferring the snake_case key) and `io_timeout_ms`; malformed values, a zero `io_timeout_ms`, and a CA file without a `rediss://` URL are `Error::Validation`.
- `tls_ca_file` (a path to a PEM bundle or a single DER certificate) is *added* to the compiled-in `webpki_roots` anchors, so public Redis deployments need no file and private CAs need only that one.
- TLS failures are `Error::Session` naming the host (untrusted or mismatched certificate, a server that does not speak TLS, an unreadable or unusable anchor); a structurally broken CA file (bad base64, missing END line, empty block or empty file) is `Error::Validation`.
- `resp_tls` owns the TLS half behind crate-private helpers: `client_config(host, ca_file)` builds the `rustls` client configuration over `webpki_roots::TLS_SERVER_ROOTS` plus the CA file, `server_name(host)` maps an IP literal to an IP address and anything else to a DNS name, and `handshake(socket, host, config)` runs the `tokio-rustls` connection; the crypto provider is `ring` and no system certificate store is consulted.
- `resp_tls` also holds the hand-rolled PEM/base64 decoder (`read_ca_certificates`, `decode_pem_bundle`, `base64_decode`), because the crate takes no PEM or base64 dependency.
- Redis key layout: one JSON document per checkpoint at `{prefix}:session:{session_id}:checkpoint:{checkpoint_id}` (the document shape `FileCheckpointer` writes), a per-session id set at `{prefix}:session:{session_id}:checkpoints`, and a global session set at `{prefix}:sessions`, so checkpointers with different prefixes on one server never see each other's data.
- Redis behaviour: `save` writes the document, applies `EXPIRE <seconds>` when `ttl` is set (a sub-second TTL is rounded up to one second) and then updates both index sets; `list` reads the id set, decodes each document, and prunes members whose document expired or was deleted out of band; `delete` removes the document plus its index member; `release` deletes every document of the session and both index entries; `session_exists` checks the session's id set.
- Redis connection handling: one `RespConnection` serialized behind a `tokio::sync::Mutex`; a transport failure marks it broken and the command is retried once on a fresh connection, while a server error reply stays `Error::Session` without reconnecting.
- `resp` — the dependency-free RESP2 client: `RespValue { Simple, Error, Integer, Bulk(Option<Vec<u8>>), Array(Option<Vec<RespValue>>) }` with `as_str`/`as_bytes`/`as_i64`/`as_array`/`is_nil`/`into_result`, `encode_command(&[&str]) -> Vec<u8>`, and `read_value(&mut impl AsyncBufRead)` (iterative nesting, 512 MiB bulk and 16M-element array limits).
- `RespConnection::connect(url)` forwards to `connect_with_ca_file(url, None)`; the parser handles `redis://`/`rediss://[user[:password]@]host[:port][/db]` (port defaults to 6379, bracketed IPv6 hosts, password-only credentials as `redis://password@host`), records the TLS decision in `ConnectionTarget`, and runs `AUTH` (with the username when present) plus `SELECT` for `/db`.
- `RespConnection` then serves `command`, `command_bytes`, `select_database`, `is_broken`, `is_tls`, `uses_tls(url)` (scheme only, no dial) and stores a `BufReader` over one boxed `AsyncReadWrite`, so the plaintext socket and the TLS stream share the same codec and command path.
- RESP URL errors: a scheme other than `redis://`/`rediss://` (case-sensitive), a missing host, an invalid port, a non-numeric `/db`, or a CA file paired with a `redis://` URL is `Error::Validation`.
- `list` order is backend-defined: memory, file, and redis ascending by `(created_at, checkpoint_id)`, sqlite newest first.
- `CheckpointerConfig { kind, path, extra }`, `CheckpointerKind { InMemory, File, Sqlite, Redis }`, `create_checkpointer(&config).await -> Result<Arc<dyn Checkpointer>>`; `file` and `sqlite` require a `path` (`Error::Validation` otherwise), `sqlite` also requires the `sqlite` feature (`Error::Unsupported` without it), and `redis` delegates to `RedisOptions::from_config` (so a missing URL is `Error::Validation`).
- `pickle` — dependency-free codec: `loads(&[u8]) -> Result<Value>` and `dumps(&Value) -> Result<Vec<u8>>` over `serde_json::Value`, covering the opcodes Python emits for plain data across protocols 0–5 (`PROTO`, `FRAME`, `STOP`, marks, memo, integers, `BINFLOAT`, text/binary strings, `BINUNICODE`, `SHORT_BINBYTES`/`BINBYTES`, list/dict/tuple opcodes, `LONG1`/`LONG4`).
- `pickle` type mapping and its lossiness: dict keys must be strings, tuples become JSON arrays (and are written back as lists), `bytes` use the reserved `{"__bytes__": "<lowercase hex>"}` shape (`BYTES_KEY`), integers must fit `i64`, floats must be finite, nesting deeper than `MAX_DEPTH = 512` and length prefixes above `MAX_LENGTH = 512 MiB` are rejected, and globals/`REDUCE`/`BUILD`/buffers/unknown opcodes are `Error::Validation` naming the opcode and offset.
- `dumps` emits protocol 4 (`PROTO 4` plus one `FRAME` covering the body and `STOP`, exactly as `CPython` frames it) with memoized strings and containers, so `CPython` 3.4+ can read the result.
- `checkpoint_from_pickle(&[u8]) -> Result<Checkpoint>` and `checkpoint_to_pickle(&Checkpoint) -> Result<Vec<u8>>` map the Python checkpoint dict (the checkpoint's snake_case JSON shape) to and from the model; `session_id` is required, a missing `checkpoint_id` is generated, and a bad payload is `Error::Validation`.

## Constraints

- Checkpoint ids are generated with `openjiuwen_common::utils::id::uuid_hex()` when empty; `latest` sorts by `(created_at, checkpoint_id)` descending itself, so it is independent of the backend's `list` order.
- `session_exists` is prefix-scoped semantics per backend: memory/sqlite/redis exactly per session id (the Redis id set), file per session directory; `release` deletes every checkpoint of a session.
- Session ids and checkpoint ids are validated before being used in keys or paths: no `/`, `\`, NUL, or `..`; violations are `Error::Validation`.
- File and sqlite writes are atomic/crash-safe (temporary `.<uuid>.tmp` file + rename; single SQL statement per save); no blocking I/O on async tasks (use `tokio::fs` / `sqlx` / `tokio::net`).
- Undecodable stored data is `Error::Session`, never a panic: the file backend reports a corrupt JSON document, the sqlite backend a corrupt payload column, the Redis backend an undecodable document or an unexpected reply shape.
- Serialized checkpoints are JSON with snake_case field names and tolerate unknown fields; pickle is only produced or consumed through the explicit `pickle` helpers.
- Errors map to `Error::Session` (I/O, storage, Redis transport/server, TLS handshake) and `Error::Validation` (ids/config/URLs/CA files/pickle payloads); no panics on user input.
- Redis has no cluster or Sentinel support and no pipelining: commands are serialized over one connection per checkpointer, and a cluster URL is not detected (the host list parses as one host).
- The TLS stack is pinned inline in `Cargo.toml` (`rustls 0.23` with `ring`, `tokio-rustls 0.26`, `webpki-roots 1`); no PEM or base64 crate is used.
- No `unsafe`, no stdout logging; backend code is split one file per backend, and `pickle.rs` (~1450 lines) stays whole because it is one codec plus its checkpoint mapping and inline opcode tests.
- `resp_tls`'s pure helpers are unit-tested inline; the end-to-end TLS behaviour is covered by `tests/checkpoint_redis_tls.rs` against the static fixtures and the loopback server's TLS mode.

## Routing Table

- `./mod.rs` → `Checkpointer` trait, `Checkpoint`, `CheckpointMeta`, namespaces, key builders, `CheckpointerConfig`, `create_checkpointer`, pickle helper re-exports
- `./memory.rs` → `MemoryCheckpointer`
- `./file.rs` → `FileCheckpointer` (filesystem/JSON backend, Python `shelve` analogue)
- `./sqlite.rs` → `SqliteCheckpointer` behind the `sqlite` feature
- `./redis.rs` → `RedisCheckpointer`, `RedisOptions` (incl. `tls_ca_file`, `io_timeout`, `from_config`), key/index layout, TTL, bounded single-connection reconnect
- `./resp.rs` → the dependency-free RESP2 codec and connection (`RespValue`, `encode_command`, `read_value`, `RespConnection`, `ConnectionTarget`)
- `./resp_tls.rs` → the `rediss://` transport: `client_config`, `server_name`, `handshake`, the PEM/base64 decoder
- `./pickle.rs` → `loads`/`dumps`, `BYTES_KEY`, `checkpoint_from_pickle`/`checkpoint_to_pickle`
