# openjiuwen-session

## Intent

Per-run execution context for the kernel: sessions and their scoped state, streaming frames, human-in-the-loop interactions, checkpoint/resume, and lightweight tracing.
Ports the Python package `openjiuwen.core.session` (v0.1.20): `session.py`, `internal/` (`WorkflowSession`, `NodeSession`, `AgentSession`, `AgentTeamSession`, `SubWorkflowSession`, and the `wrapper.py` `RouterSession`), `state/` (`CommitState` lanes and agent state), `stream/` (`StreamMode`, schemas, emitter, writer manager), `interaction/` (`InteractiveInput`, `InteractionOutput`, workflow/agent interrupts), `checkpointer/` (memory, file, SQLite, and the Redis/RESP backends over clear text or TLS, plus the pickle interop codec for Python-written payloads), `session_controller/` (the disk-backed storage subset and the session link graph), `vcs/` (the local version-history subset with refs, rewinds and forks), `tracer/`, the session-scoped trace-sink hook that replaces Python's `set_tracer`, and `config/base.py`.
The concrete session inventory is complete: the four execution sessions plus the `SubWorkflowSession`/`RouterSession` wrappers, with no session type deferred to a later phase.
Every graph node, workflow component, agent, team, and runner receives `Arc<dyn Session>` (`docs/ARCHITECTURE.md` §2.4); this crate is the stable seam for state, streaming, HITL, tracing, and resume, and higher layers must not re-implement it.

## API Surface

### `session` — the contract (`src/session.rs`)

```rust
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionType { #[default] Workflow, Agent, AgentTeam, Node }

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SessionMetadata {
    pub session_id: String, pub parent_session_id: Option<String>, pub kind: SessionType,
    pub entity_id: Option<String>, pub entity_name: Option<String>,
    pub trace_id: Option<String>, pub created_at: Option<String>,
    pub metadata: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SessionSnapshot {
    pub session_id: String, pub kind: SessionType,
    pub global: Value, pub components: BTreeMap<String, Value>,
    pub trace: Option<TracePosition>, pub metadata: BTreeMap<String, Value>,
}

#[async_trait]
pub trait Session: Send + Sync {
    fn id(&self) -> &str;
    fn parent_id(&self) -> Option<&str>;
    fn session_type(&self) -> SessionType;
    async fn get_state(&self, key: &str) -> Result<Option<Value>>;
    async fn update_state(&self, patch: Value) -> Result<()>;
    async fn get_component_state(&self, component_id: &str) -> Result<Option<Value>>;
    async fn update_component_state(&self, component_id: &str, patch: Value) -> Result<()>;
    async fn write_stream(&self, item: StreamItem) -> Result<()>;
    async fn request_interaction(&self, request: InteractionRequest) -> Result<InteractionResponse>;
    fn trace(&self) -> &dyn TraceSink;

    // Provided methods (defaults keep fakes and test doubles cheap)
    fn install_trace_sink(&self, sink: Arc<dyn TraceSink>) -> Result<()>;  // default: Unsupported
    fn metadata(&self) -> SessionMetadata;                       // derived from id/parent/kind
    fn checkpointer(&self) -> Option<Arc<dyn Checkpointer>> { None }
    fn tracer(&self) -> Option<&Tracer> { None }
    fn config(&self) -> Option<Arc<RwLock<SessionConfig>>> { None }      // parking_lot::RwLock
    fn writers(&self) -> Option<Arc<StreamWriterManager>> { None }
    fn subscribe(&self) -> BoxStream<'static, Result<StreamEvent>>;   // empty by default
    async fn resolve_interaction(&self, response: InteractionResponse) -> Result<()>;
    async fn end_stream(&self) -> Result<()> { Ok(()) }
    async fn close(&self) -> Result<()> { Ok(()) }
    async fn snapshot(&self) -> Result<SessionSnapshot>;
    async fn restore(&self, snapshot: SessionSnapshot) -> Result<()>;
    async fn save_checkpoint(&self, checkpoint_id: Option<&str>) -> Result<String>;
    async fn resume(&self, checkpoint_id: &str) -> Result<bool>;
}
```

State semantics (contract for every implementor):
- `get_state(key)` reads the session's global lane; `key` is a dot/bracket path (`a.b[0].c`); an empty key returns the whole global object.
- `update_state(patch)` deep-merges an object patch into the global lane under one write lock: object-vs-object recurses, a `null` value deletes the path (deletes applied after writes), any other value replaces wholesale (arrays are never element-merged).
- `get_component_state(id)` / `update_component_state(id, patch)` address the component lane by component id; ids are treated as nested paths, so component `a.b` is stored at `{"a": {"b": ...}}` exactly like Python's `executable_id` nesting. An empty component id is `Error::Validation`.
- `write_stream` publishes a [`StreamItem`] to the session's hub; a `Null` payload is `Error::Validation` (Python error 111132). `end_stream` publishes the completion marker (`StreamEvent::End`), is idempotent, and never fails on an already-closed stream.
- `request_interaction` publishes a `__interaction__` output frame and awaits a [`InteractionResponse`]; a broker seeded via `InteractionBroker::seed`/`seed_interrupt` returns the queued answer immediately without re-emitting the frame, but the `Session` trait exposes no seeding accessor (a session's broker stays private; `openjiuwen-workflow` seeds its own `InteractionState` instead). Timeout resolves to `Error::Timeout`, cancellation to `Error::Cancelled`.
- `trace()` returns the session's sink; `tracer()` returns the span factory when the session owns one.
- `install_trace_sink(sink)` is the session-scoped trace hook: the provided default reports `Error::Unsupported`, and sessions that own a swappable sink reroute every span started afterwards without changing the `trace()` reference (and therefore the checkpointed trace position).
- `snapshot`/`restore` are the checkpointing seam: default implementations cover global state + trace position; concrete sessions add component state.
- `save_checkpoint` returns the checkpoint id (generated when `None`); `resume` loads and restores, returning `Ok(false)` when the checkpoint does not exist.

### `sessions` — concrete sessions (`src/sessions/`)

- `WorkflowSession::new(session_id)` / `with_options(WorkflowSessionOptions)`, `create_workflow_session(WorkflowSessionOptions)`, `workflow_id()`, `state() -> Arc<WorkflowState>`, `dump_state()`, `commit_user_inputs(inputs)`, `create_node_session(component_id) -> NodeSession`, `save_checkpoint`, `resume`; a parentless session without an id generates a `uuid_hex()`, a parented one defaults to the parent id.
- `NodeSession` — a component-scoped child view: its executable id follows Python's `create_executable_id` (the bare `component_id` when the parent is not itself a `NodeSession`, `a.b.c` dot-joined when the parent is a `NodeSession`), and `metadata().entity_id` is that executable id; global reads pass through, component reads/writes nest under its executable id, streaming/tracing/interaction/snapshot delegate, `end_stream`/`close` are no-ops; its checkpoint entity id is the executable id. `new(parent, component_id)` and `new_with_type(parent, component_id, node_type)` build it; inherent accessors are `executable_id()`, `component_id()`, `node_type() -> Option<&str>`, `workflow_id()`, `main_workflow_id()`, `workflow_nesting_depth() -> u32`, `parent()`, and `create_node_session(component_id)`.
- Workflow placement metadata: `WorkflowSession::with_options` stamps `workflow_id`, `main_workflow_id` (equal to `workflow_id` for a top-level workflow), and `workflow_nesting_depth` (0) into the session metadata unless the caller supplied them, and `NodeSession` reads them back from its parent's metadata — `workflow_nesting_depth` falls back to 0, `workflow_id` to `DEFAULT_WORKFLOW_ID`, and `main_workflow_id` to `workflow_id`.
- `AgentSession::new` / `create_agent_session(AgentSessionOptions)` — agent state (`global_state` + `agent_state` lanes), agent id/name/description metadata, `env(key) -> Option<Value>` / `envs()`, `create_workflow_session()` sharing the agent session id, config, hub and trace id but owning an independent workflow state, `commit() -> Result<Option<String>>`, idempotent `post_run()` (`end_stream` when `close_stream_on_post_run`, then commit), `resume`.
- `AgentTeamSession::new` / `create_agent_team_session(AgentTeamSessionOptions)` — team id (default `DEFAULT_TEAM_ID = "agent_team"`), `global_state`/`agent_state` lanes, `create_agent_session(agent_id)` child with `parent_session_id` = team session id, own session id, `source_agent_id`/`source_team_id` metadata when `source_metadata_enabled`, and the team's hub/config/checkpointer/trace id, `commit()`/`flush_checkpoint()`, idempotent `post_run()`.
- `SubWorkflowSession::new(&host_node, workflow_id)` — the session a nested workflow runs with: it rebuilds the host component id and node type over `host.parent()` (re-parenting) and therefore keeps the host node's executable id, while overriding the workflow placement (`workflow_id`; an empty argument keeps the host's, never empty), `main_workflow_id` (inherited from the host), and `workflow_nesting_depth` (host depth + 1). It adds an end-output slot shared across clones (`set_end_output`, `end_output`, `take_end_output` — consuming, and `propagate_end_output`, which merges `{"output": value}` into the host node's component lane idempotently without consuming) and delegates state, streams, HITL, tracing, snapshots, and the trace hook to the inner node view; `session_type()`/`metadata().kind` are `Node`, and `end_stream`/`close` stay no-ops so a nested workflow never closes the parent hub.
- `RouterSession::new(inner: Arc<dyn Session>)` — the read-only view handed to conditional-edge routers: `id`, `parent_id`, `get_state`, `get_component_state`, `trace`, and `snapshot` read through (with `kind` forced to `Node`), while `update_state`, `update_component_state`, `write_stream`, `restore`, and `resolve_interaction` return `Error::Unsupported` with a per-method reason pointing at `inner()`; `end_stream`/`close` stay deliberate `Ok(())` lifecycle no-ops, `request_interaction` and `install_trace_sink` are `Error::Unsupported`, `config`/`writers`/`tracer`/`checkpointer` are `None`, `subscribe` is empty, `session_type()`/`metadata().kind` are `Node`, and `inner()`/`session_id()`/`executable_id()` expose the wrapped session.
- Every session shares `Arc<StreamHub>` + `Arc<StreamWriterManager>` down to children; config and tracer are shared by reference; each session tracks its own interaction broker.

### `state` — state stores (`src/state/`)

- `StateStore` trait (used by every session): `get_global`, `update_global`, `write_global`, `get_component`, `update_component`, `components`, `dump`, `restore`, `trace_state`, `update_trace`; `write_global` is the committed-lane write sessions use (the default forwards to `update_global`, which may stage).
- `WorkflowState` — Python `CommitState`: lanes `io_state`, `global_state`, `comp_state`, `workflow_state`, `trace_state`; per-node staging (`update`/`update_global`/`update_workflow`/`update_component`), `commit(node_id: Option<&str>)`, `rollback(node_id)`, `updates()`, `set_updates()`, `inputs()`, `outputs(node_id)`, `set_outputs`, `commit_user_inputs`, `get_by_schema`, `node(node_id, parent_id)` sharing lanes, `dump()`.
- `AgentState` — Python agent/team `StateCollection`: `global_state` + `agent_state` lanes, no staging, `dump()` = `{"global_state":…, "agent_state":…}`.
- `paths` helpers: `get_value`, `set_value`, `remove_value`, `get_by_prefix`, `apply_patch` (Python `update_dict`), `is_ref_path`/`extract_origin_key` for `${node.field}`.
- Lane key constants: `IO_STATE_KEY`, `IO_STATE_UPDATES_KEY`, `GLOBAL_STATE_KEY`, `GLOBAL_STATE_UPDATES_KEY`, `COMP_STATE_KEY`, `COMP_STATE_UPDATES_KEY`, `WORKFLOW_STATE_KEY`, `WORKFLOW_STATE_UPDATES_KEY`, `TRACE_STATE_KEY`, `AGENT_STATE_KEY`.

### `stream` — frames and delivery (`src/stream.rs`)

```rust
pub enum StreamMode { Output, Trace, Custom }        // wire "output"/"trace"/"custom"
pub const END_FRAME: &str = "all streaming outputs finish";
pub struct StreamItem { pub mode: StreamMode, pub item_type: String /* "type" */,
                        pub index: Option<i64>, pub payload: Value }
pub enum StreamEvent { Item(StreamItem), End }
pub enum StreamSendOutcome { Delivered { subscribers: usize }, NoSubscribers,
                             Closed, ModeDisabled }
pub struct StreamHub;                                // bounded broadcast transport
pub struct StreamWriterManager;                      // mode routing over one hub
```

- `StreamItem::output(item_type, index, payload)`, `::output_at(item_type, index: i64, payload)`, `::trace(item_type, payload)`, `::custom(payload)` (frame type `"custom"`), `with_mode(mode)`, `is_end()`, `to_wire() -> Value` (`{"type","index","payload"}`, mode dropped for wire parity, `index` omitted when `None`).
- `StreamMode::as_str() -> &'static str` returns `"output"`/`"trace"`/`"custom"`; `StreamMode` is `Copy` and `Default = Output`.
- `StreamHub::new()` (capacity `DEFAULT_STREAM_CAPACITY = 1024`), `with_capacity`, `send_now(StreamItem) -> Result<StreamSendOutcome>`, `send` (the same, `async`), `subscribe() -> BoxStream<'static, Result<StreamEvent>>`, `close`, `is_closed`, `capacity`.
- `send_now` is the canonical publish path: a synchronous `tokio::sync::broadcast` send with no I/O and no await, usable from synchronous trace sinks and from any thread; `send` is an `async` wrapper that never awaits internally (kept for API symmetry with Python). It reports `Delivered { subscribers }` when at least one subscriber received the frame, `NoSubscribers` when the open hub has none, and `Closed` after `close()`; only a `Null` payload is an error.
- `StreamWriterManager::new(hub)` (all modes), `with_modes(hub, modes)`, `write_now`, `write`, `subscribe`, `close`, `is_closed`, `modes`, `enables`, `hub`.
- `write_now` is the canonical write path (mode check, then `StreamHub::send_now`) and returns the outcome too, reporting `ModeDisabled` for a mode the manager does not route; `write` is its `async` wrapper, also without an internal await.
- Ordered delivery per subscriber, multiple subscribers, bounded buffering (a lagging subscriber logs a warning and skips frames rather than blocking producers), and a terminal `End` event visible to late subscribers.
- `StreamSendOutcome` is `Copy`, serde `snake_case` (`{"delivered":{"subscribers":n}}`, `"no_subscribers"`, `"closed"`, `"mode_disabled"`), and deliberately not `#[must_use]`, so fire-and-forget producers ignore it while `StreamTraceSink` acts on it: the trace position advances only on `Delivered`, so an unobserved session checkpoints `trace: None`.
- Frame `type` vocabulary is open: `__interaction__`, `tracer_agent`, `tracer_workflow`, `workflow_final`, `llm_output`, …

### `interaction` — HITL (`src/interaction.rs`)

```rust
pub const INTERACTION_FRAME_TYPE: &str = "__interaction__";
pub const INTERACTIVE_INPUT_KEY: &str = "__interactive_input__";

pub enum InteractionKind { Permission, Confirm, AskUser, SkillAuthorization,
                           SkillEvolutionApproval, TeamToolApproval, ClientTool, Custom }
pub enum InteractionDecision { Approve /* allow */, ApproveAlways /* allow_always */,
                               Reject /* deny */, Cancel, Answer }
pub struct InteractionRequest { pub id: String, pub kind: InteractionKind,
    pub component_id: Option<String>, pub question: Value,
    pub options: Vec<InteractionOption>, pub timeout_ms: Option<u64>,
    pub metadata: BTreeMap<String, Value>, /* #[serde(skip)] cancellation */ }
pub struct InteractionResponse { pub request_id: String, pub decision: InteractionDecision,
                                 pub value: Value, pub reason: Option<String> }
pub struct InteractiveInput { pub user_inputs: BTreeMap<String, Value>,
                              pub raw_inputs: Option<Value>,
                              pub interrupt_answers: BTreeMap<String, Value> }
pub struct InteractionOutput { pub id: String, pub value: Value }
pub struct InteractionBroker;   // pending oneshots, seeded answers, default timeout
```

- `InteractionRequest::new(kind, question)` mints `intr_<uuid hex>`; helpers `permission`, `confirm`, `ask_user`, `with_component`, `with_timeout(Duration)`, `to_stream_item(index)`.
- `InteractionResponse::approve/approve_always/reject/answer/cancel`, `is_approved()`.
- `InteractionBroker::request(&self, request, &StreamWriterManager)` publishes the `__interaction__` frame with payload `InteractionOutput { id: component_id.unwrap_or(request.id), value: question }` and awaits; `resolve` wakes a pending request (`Error::NotFound` when unknown); `seed(InteractiveInput)` queues resume answers (FIFO per interrupt id); `cancel_all`.
- Default timeout comes from the request, else the broker (`set_default_timeout`), else wait forever (Python behavior).

### `checkpoint` — persistence (`src/checkpoint/`)

```rust
pub const SESSION_NAMESPACE_AGENT: &str = "agent";
pub const SESSION_NAMESPACE_AGENT_TEAM: &str = "agent-team";
pub const SESSION_NAMESPACE_WORKFLOW: &str = "workflow";
pub const WORKFLOW_NAMESPACE_GRAPH: &str = "workflow-graph";

pub enum CheckpointNamespace { Agent, AgentTeam, Workflow, WorkflowGraph }
pub struct Checkpoint { pub session_id: String, pub checkpoint_id: String,
    pub namespace: CheckpointNamespace, pub entity_id: String, pub created_at: String,
    pub global_state: Value, pub component_state: Value, pub updates: Value,
    pub trace: Option<TracePosition>, pub metadata: BTreeMap<String, Value> }
pub struct CheckpointMeta { session_id, checkpoint_id, namespace, entity_id, created_at }

#[async_trait]
pub trait Checkpointer: Send + Sync + std::fmt::Debug {
    async fn save(&self, checkpoint: Checkpoint) -> Result<String>;
    async fn load(&self, session_id: &str, checkpoint_id: &str) -> Result<Option<Checkpoint>>;
    async fn latest(&self, session_id: &str) -> Result<Option<Checkpoint>>;   // provided
    async fn list(&self, session_id: &str) -> Result<Vec<CheckpointMeta>>;
    async fn delete(&self, session_id: &str, checkpoint_id: &str) -> Result<()>;
    async fn release(&self, session_id: &str) -> Result<()>;
    async fn session_exists(&self, session_id: &str) -> Result<bool>;
}
```

- `build_key(parts: &[&str]) -> String` (`:`-joined) and `build_key_with_namespace(session_id, namespace, entity_id, suffixes)` preserve Python's key layout.
- Backends: `MemoryCheckpointer` (`new()`, `shared()` process-wide singleton, `len(session_id)`, `is_empty()`, `clear()`; clones share storage), `FileCheckpointer` (`new(root)`, `<root>/<session_id>/<checkpoint_id>.json`, atomic write through a `.<uuid>.tmp` file plus rename, corrupt JSON is `Error::Session`), `SqliteCheckpointer` (feature `sqlite`, `open(path).await` / `in_memory().await`, one `session_checkpoints` table, a single upsert statement per save, corrupt payload is `Error::Session`), and `RedisCheckpointer` (always compiled; below).
- `RedisCheckpointer::connect(url)` / `connect_with_options(RedisOptions)` implements the same [`Checkpointer`] contract over RESP2 — clear text for `redis://`, TLS for `rediss://` — and exposes `options()`; `RedisOptions { url, prefix, ttl: Option<Duration>, database: Option<u32>, tls_ca_file: Option<String> }` is serde-serializable, defaults the prefix to `checkpoint::redis::DEFAULT_REDIS_KEY_PREFIX = "openjiuwen:session"`, and `Debug` output redacts the URL password.
- `RedisOptions::from_config(&CheckpointerConfig)` reads the declarative configuration (`extra["url"]` with the rendered `path` as fallback, then `prefix`, `ttl_seconds`, `database`, and `tls_ca_file` with the hyphenated `tls-ca-file` alias), so `create_checkpointer` never enumerates those keys itself.
- `tls_ca_file` names extra trust anchors for a `rediss://` URL (a PEM bundle of `CERTIFICATE` blocks or a single DER certificate); the anchors are *added* to the compiled-in `webpki_roots`, and pairing the option with a `redis://` URL is `Error::Validation` rather than a silent no-op.
- TLS failures are `Error::Session` naming the host (an untrusted or mismatched certificate, a server not speaking TLS, an unreadable CA file, a certificate rustls cannot use as an anchor); a structurally broken CA file (bad base64, missing END line, empty block or file) is `Error::Validation`.
- `checkpoint::resp_tls` builds the `rustls` client configuration over `webpki_roots::TLS_SERVER_ROOTS` plus the CA file, derives the server name from the URL host (an IP literal stays an IP, anything else a DNS name) and performs the handshake; the crypto provider is `ring`, no system certificate store is consulted, and PEM/base64 decoding is hand-rolled in that module.
- Redis key layout: one JSON document per checkpoint at `{prefix}:session:{session_id}:checkpoint:{checkpoint_id}` (the same document shape the file backend writes), a per-session id set at `{prefix}:session:{session_id}:checkpoints`, and a global session set at `{prefix}:sessions`, so checkpointers with different prefixes on one server never see each other's data.
- Redis behaviour: `save` writes the document, applies `EXPIRE <seconds>` when `ttl` is set (a sub-second TTL is rounded up to one second), then updates both index sets; `list` reads the id set, decodes each document, and prunes members whose document expired or was deleted out of band; `delete` removes the document and its index member; `release` deletes every document of the session plus both index entries; `session_exists` checks the session's id set.
- Redis connection handling: `connect` dials eagerly (a bad URL, an unreachable server, or a failed TLS handshake fails at construction), one `RespConnection` is serialized behind a `tokio::sync::Mutex`, a transport failure is retried once on a fresh connection, and server error replies stay `Error::Session` without reconnecting.
- `resp` — the dependency-free RESP2 client: `RespValue { Simple, Error, Integer, Bulk(Option<Vec<u8>>), Array(Option<Vec<RespValue>>) }` with `as_str`/`as_bytes`/`as_i64`/`as_array`/`is_nil`/`into_result` (a server error reply becomes `Error::Session`), `encode_command(&[&str]) -> Vec<u8>` (one array of bulk strings), and `read_value(&mut impl AsyncBufRead)` with iterative nesting (512 MiB bulk and 16M-element array limits); `RespConnection::connect(url)` parses `redis://`/`rediss://[user[:password]@]host[:port][/db]` (port defaults to 6379, bracketed IPv6 hosts, password-only credentials as `redis://password@host`) and runs `AUTH` (with the username when present) plus `SELECT` for `/db`, then serves `command`, `command_bytes`, `select_database`, and `is_broken`.
- `RespConnection::connect_with_ca_file(url, Option<&Path>)` is the TLS-aware constructor (`connect` forwards to it with no CA file), `uses_tls(url)` reports the scheme without dialing, and `is_tls()` reports the connected transport; the connection stores a `BufReader` over one boxed `AsyncReadWrite`, so the plaintext socket and the `tokio_rustls` stream share the same codec and command path.
- RESP URL errors: a scheme other than `redis://`/`rediss://`, a missing host, an invalid port, or a non-numeric `/db` is `Error::Validation`, and a CA file paired with a `redis://` URL is `Error::Validation`.
- `CheckpointerConfig { kind, path, extra }`, `CheckpointerKind { InMemory, File, Sqlite, Redis }`, `create_checkpointer(&config).await -> Result<Arc<dyn Checkpointer>>`; `file` and `sqlite` require a `path` (`Error::Validation` otherwise), `redis` reads its options through `RedisOptions::from_config` (`url`, `prefix`, `ttl_seconds`, `database`, `tls_ca_file`), and `sqlite` additionally requires the `sqlite` feature (`Error::Unsupported` without it).
- `Checkpoint` derives serde with snake_case field names; checkpoint ids are generated (`uuid_hex()`) when empty; `latest` sorts `(created_at, checkpoint_id)` descending itself, so it does not depend on backend list order; session and checkpoint ids are validated (no path separators, no `..`).
- `list` order is backend-defined: `MemoryCheckpointer`, `FileCheckpointer`, and `RedisCheckpointer` return `(created_at, checkpoint_id)` ascending, `SqliteCheckpointer` newest first; callers that need a specific order must sort the result.
- Resume restores global state, component state, updates, and the trace position.
- `pickle` — the dependency-free Python-pickle codec: `pickle::loads(&[u8]) -> Value` and `pickle::dumps(&Value) -> Vec<u8>` cover the opcode subset Python emits for plain data across protocols 0–5 (framing, memo, scalars, text/binary strings, bytes, dict/list/tuple, `BINFLOAT`, `LONG1`/`LONG4`).
- Pickle restrictions: dict keys must be strings, tuples decode as JSON arrays (and encode back as lists), `bytes` use the reserved `{"__bytes__": "<lowercase hex>"}` shape (`BYTES_KEY`), integers must fit `i64`, floats must be finite, nesting deeper than 512 levels and length prefixes above 512 MiB are rejected, and globals/`REDUCE`/`BUILD`/buffers/unknown opcodes are `Error::Validation` naming the opcode and offset.
- `dumps` emits protocol 4 (`PROTO 4` + one `FRAME` covering the body and `STOP`, exactly as `CPython` frames it) with memoized strings and containers, so the output is readable by `CPython` 3.4+.
- `checkpoint_from_pickle(&[u8]) -> Checkpoint` and `checkpoint_to_pickle(&Checkpoint) -> Vec<u8>` map the Python checkpoint dict (the checkpoint's `snake_case` JSON shape) to and from the crate's model; `session_id` is required, a missing `checkpoint_id` is generated, and bad payloads are `Error::Validation`.
- Pickle is an explicit interop codec, not a backend format: JSON stays native, and no backend auto-detects or writes pickle.
- Redis tests are offline: `tests/support/resp_server.rs` provides `FakeRedis`, a loopback RESP2 server with in-memory strings and sets, a command log (`commands`, `count`, `command`), a `hang_up_after_each_reply` switch, Redis's empty-set-disappears rule, and a TLS mode (`start_tls`, `url_tls`) serving the DER fixtures in `tests/fixtures/tls/`.

### `session_controller` — disk-backed product sessions (`src/session_controller/`)

- Scope types (all `Display` + `FromStr` + serde-through-string, malformed values are `Error::Validation`): `Scope { Main /* "main" */, Custom(String) }`, `Subject { Direct { user_id } /* direct:<user> */, Group { group_id } /* group:<group> */, GroupUser { group_id, user_id } /* group:<group>:user:<user>; ids containing a colon use the percent-escaped group-user-v2: form */ }`, `SessionScope { scope, subject }` (`"main"`, `"main:direct:u1"`, `"main:group:g1"`, `"main:group:g1:user:u1"`, or a custom scope name) with constructors `main`/`direct`/`group`/`group_user`/`custom`, and `SessionScopeKey { agent_id, scope }` (`agent:<agent_id>:<scope>`).
- Schema (`sessions.json` and `state.data`): `SessionMeta { session_id, created_at, updated_at /* UTC epoch seconds, f64 */, version, is_active, data_container_type, title, metadata }` with `new` (active, version 1) and `touch` (strictly monotonic `updated_at`, version bump), `DEFAULT_DATA_CONTAINER_TYPE = "agent"`, `ScopeSessionsMeta { session_scope_key, active_session, sessions }` (`get_session`, `upsert` which deactivates siblings, `remove`, `active`, `sort_sessions` newest-first), `SessionsIndex { agent_id, updated_at, scopes }`, and `SessionRecord { meta, snapshot }` serialized as `{"meta": …, "data": <SessionSnapshot>}`.
- `DiskSessionStore::new(base)` — on-disk layout `<base>/<agent_id>/sessions/sessions.json` (the index), `<base>/<agent_id>/sessions/<session_id>/state.data` (one record), `<base>/<agent_id>/sessions/<session_id>/history/` (version history), and `<base>/<agent_id>/links.json` (the session link graph); path builders `base`, `agent_dir`, `sessions_dir`, `sessions_index_path`, `session_dir`, `state_path`, `history_dir`, `links_path`, and `history() -> SessionHistory`.
- `DiskSessionStore` methods: `save(agent_id, scope, record) -> SessionMeta`, `save_snapshot`, `save_from_session`, `load`, `load_snapshot`, `load_into_session` (one `Session::restore`), `exists`, `delete` (returns `false` when absent and also drops the session's link entry), `list(agent_id, Option<&SessionScope>)`, `list_all`, `list_agents`, `read_index`, `write_index`.
- Link-graph wrappers on the store: `read_links`/`write_links` (a missing document is an empty graph, writes are atomic and serialized by a store-level lock shared across clones), `link_parent(child, parent)`, `link_fork(child, origin)`, `parent_of`, `children_of`, `ancestors`, `descendants`, `chain` (each takes `agent_id` first).
- Write semantics: every document is pretty JSON written to a sibling `.<name>.<uuid>.tmp` and renamed, so readers never observe a partial file and a failed write leaves no temp file; corrupt documents are `Error::Session`, and agent/session ids go through `checkpoint::validate_id` (`Error::Validation`).
- `save` semantics: a new session starts at version 1 and becomes the scope's active session (deactivating siblings); an existing record keeps `created_at`, takes the new `title`/`metadata`/`data_container_type`, and bumps `version`; `updated_at` is strictly monotonic per agent, so newest-first listings are deterministic; the stored `data` is the snapshot itself, so a record loads directly with `Session::restore`.
- `SessionController::new(store, agent_id)` — the per-agent cache in front of the store (an `Arc<parking_lot::Mutex<BTreeMap<…>>>`): `create` (empty `SessionType::Agent` snapshot), `upsert`/`flush` (persist in the scope the index already records), `upsert_in`/`save_in` (explicit scope, needed for a session the index does not know), `save`, `get` (cache first, populated on a store read), `delete`, `list` (a scope-less listing merges cache-only records), plus `agent_id` and `store`.
- Scope resolution is strict: `scope_of` returns `Option` — an unparseable scope key in `sessions.json` is `Error::Session` naming the key (never a warn-and-skip) and an unknown session makes `upsert`/`flush` return `Error::Validation` instead of silently filing it under `main()`.
- `SessionLinks` (persisted as `LinksIndex`) is the session link graph of `LinkKind::{Parent, Fork}` entries (`SessionLink { session_id, parent, forked_from, children, created_at }`) with cycle-safe traversal (`chain`/`ancestors`/`descendants`/`depth`/`roots`/`leaves`/`forks_of`); a self or cycle-creating link is `Error::Validation`, a stored cycle is `Error::Session`, and forks are downstream but never children.
- Deferred from Python's `session_controller/`: data-container/permission policies and the process-global `GlobalSessionController` singleton; Python's `ChainSession` session type is deliberately not ported, because state fan-out needs a shadow store and belongs to the crates that own a run loop (`session_controller/CONTEXT.md`).

### `vcs` — local session history (`src/vcs/`)

- `VersionId(u64)` — `Display` renders `v000001` (six zero-padded digits), `FromStr` accepts `v000001` or `1` and rejects `0` and non-numeric input, plus `as_u64` and `From<u64>`.
- `Version { id, label, created_at /* RFC 3339 */, parent: Option<u64>, snapshot }` and `VersionMeta { id, label, created_at, parent }` with `Version::meta()`.
- `SessionHistory::new(DiskSessionStore)` (or `store.history()`) — layout `<session_dir>/history/versions/v000001.json` plus an optional `<history>/index.json` metadata cache that is rescanned whenever it is absent or corrupt, and `<history>/refs.json` (the local `Refs` document); ids are sequential and allocated under one store-level `tokio::sync::Mutex`, so concurrent commits — even through cloned stores — never reuse an id; a new version's `parent` is the previous latest, every commit advances `HEAD` and `main`, and every write is atomic (temp file + rename).
- `SessionHistory` methods: `commit(agent_id, session_id, &SessionSnapshot, label) -> VersionMeta`, `commit_from_session`, `list` (ascending id), `load`, `latest`, `delete` (returns `false` when missing), `clear` (removes the history directory), `diff(left, right) -> Vec<JsonChange>` (`Error::Session` when either version is missing), and `restore(agent_id, session_id, id, &dyn Session) -> bool` (applies an old snapshot without committing); path helpers `history_dir`, `versions_dir`, `version_path`, `index_path`, `refs_path`.
- `Refs` — local refs over the versions: `RefName` (validated: non-empty, at most `MAX_REF_NAME_LEN = 128` characters, no whitespace, control characters, path separators or `..`), `RefEntry { name, target, created_at, updated_at }`, `HEAD_REF = "HEAD"` (the mutable tip) and `MAIN_REF = "main"` (the latest commit); `SessionHistory::{refs, set_ref, delete_ref, resolve_ref, list_refs, head}` manage them, `set_ref` rejects a version that does not exist, and `head()` falls back to `latest()` when the document or its target is missing.
- `rewind(agent_id, session_id, Option<VersionId>, &dyn Session)` restores the live session and moves `HEAD` back while keeping the newer versions (`RewindOptions::SOFT`); `rewind_hard` (`RewindOptions::HARD`) also deletes every newer version and re-points `main`; `rewind_with` takes the options explicitly, a missing target is `Error::Session`, and the restore happens before anything is deleted.
- `fork(agent_id, source_session_id, Option<VersionId>, new_session_id, title)` writes the chosen version's snapshot as a new session record in the source session's scope and commits it as the fork's first version, labelled `fork of <source>@v000003` with no `parent`; an unknown source session or an existing target id is `Error::Validation`, and the source history and refs stay untouched.
- Pure diffing: `ChangeKind { Added, Removed, Changed }` and `JsonChange { path, kind, before, after }`; `diff_values` walks objects by sorted key and arrays by ascending index (an appended tail is one `Added` per index, a removed tail one `Removed` per index) and reports any other inequality as `Changed` at that path; paths use the crate's nested-path syntax (`a.b[0].c`, root `""`) and the output is deterministic.
- `diff_snapshots` diffs the persisted snapshot shape, so a difference surfaces as `kind`, `global.<path>`, `components.<id>.<path>`, `trace.<field>`, or `metadata.<key>`.
- Deferred from Python's `vcs/`: git and other external backends, event/WAL (`e<N>`) references, codecs, and context-engine adapters; the local history, refs, rewinds and forks are self-contained and offline.

### `tracer` — spans and frames (`src/tracer.rs`)

```rust
pub enum InvokeType { Prompt, Llm, Plugin, Workflow, Chain, Retriever, Evaluator }
pub enum SpanStatus { Start, Running, Finish, Interrupted, Error }
pub enum TraceSpanKind { Agent, Workflow, Component }
pub enum TraceFrameType { Agent /* tracer_agent */, Workflow /* tracer_workflow */ }
pub struct TraceSpan { /* camelCase wire keys: traceId, sessionId, invokeId, parentInvokeId,
    childInvokes, name, spanKind, invokeType, status, startTime, endTime, elapsedTime,
    inputs, outputs, error, metaData, componentId, componentName, componentType, workflowId */
    pub kind: TraceSpanKind }
pub struct TraceFrame { pub item_type: String /* tracer_agent | tracer_workflow */, pub payload: TraceSpan }
pub struct TracePosition { pub trace_id: String, pub last_invoke_id: Option<String>, pub frame_count: u64 }

pub trait TraceSink: Send + Sync + std::fmt::Debug {
    fn emit(&self, frame: TraceFrame);                        // synchronous; never awaits
    fn position(&self) -> Option<TracePosition> { None }
    fn set_position(&self, position: TracePosition) {}
}
pub struct Tracer;     // span factory bound to a session id + trace id + sink
pub struct SpanHandle; // id / trace_id / set_inputs / set_outputs / set_metadata / set_status /
                       // finish / fail / interrupted / snapshot — all take &self
```

- Sinks: `NoopTraceSink` (the default), `MemoryTraceSink` (records frames; `frames()`, `take()`, `len()`, `is_empty()`, `clear()`), `StreamTraceSink` (publishes through an `Arc<StreamWriterManager>` with `write_now` and only logs at debug level, dropping the frame, when the write or the JSON serialization fails; it also tracks the published frame position, implementing `position`/`set_position` over shared `Arc<parking_lot::Mutex<…>>` state guarded by a baseline plus an emitted counter so a rebased position continues correctly, advancing only on `StreamSendOutcome::Delivered` and therefore reporting no position until a frame is delivered or one is installed).
- `Tracer::new(session_id, trace_id)`, `with_sink`, `set_sink`, `start_span(name, invoke_type)` (an agent span), `start_workflow_span(component_id, name)`, `start_child(&parent, name, invoke_type)`, `sink() -> Arc<dyn TraceSink>` (a cloned handle, because the sink slot lives behind an `RwLock`), `trace_id()`, `session_id()`.
- `start_workflow_span` mints a `Component` span with `component_id` set, `workflow_id == component_id` and `invoke_type = Workflow`; `start_child` inherits the parent's `TraceSpanKind`, records `parentInvokeId`, and appends the child id to the parent's `childInvokes`.
- `SpanHandle` takes `&self` everywhere (interior mutability through `Arc<Mutex<SpanState>>`); clones share one span, `finish`/`fail`/`interrupted` emit exactly one frame and later terminal transitions are no-ops, and the lock is released before the sink runs.
- Agent spans emit as `tracer_agent` frames; workflow and component spans emit as `tracer_workflow`; `TraceFrame::agent(span)` / `TraceFrame::workflow(span)` take only the span because `session_id`/`trace_id` live on `TraceSpan`, and `to_stream_item()` projects a frame onto a trace-mode `StreamItem`.
- Empty span names default to the kind name (`agent`, `workflow`, `component`); an empty trace id is replaced by a generated uuid; `TracePosition` stays snake_case because it is the checkpoint JSON shape.
- `MemoryTraceSink::position` derives `(trace_id, last_invoke_id, frame_count)` from the emitted frames; `set_position` installs the frame-count baseline and resets the emitted counter, and `clear()` resets both the frames and the derived position.

### Trace hook and `otel` feature (`src/trace_hook.rs`, `src/otel.rs`)

- `SwappableTraceSink::new(inner)` / `default()` (a `NoopTraceSink`), `current() -> Arc<dyn TraceSink>`, and `install(sink)`; its `TraceSink` impl (`emit`, `position`, `set_position`) clones the current sink out of the `RwLock` before using it, so an installed sink that installs recursively cannot deadlock.
- `Session::install_trace_sink(sink)` is the provided session hook (default `Error::Unsupported`, `"session does not support installing a trace sink"`); `WorkflowSession`, `NodeSession`, `AgentSession`, `AgentTeamSession`, and `SubWorkflowSession` reroute the session's `Arc<SwappableTraceSink>` — shared with children and clones — so spans started afterwards reach the new sink while `Session::trace` keeps returning the same reference. `RouterSession` rejects installation (`Error::Unsupported`, `"router sessions are read-only"`).
- Feature `otel = []` (a marker with no dependency) gates `otel::try_install_trace_sink(&dyn Session, Arc<dyn TraceSink>) -> Result<TraceSinkInstall>`: with the feature it forwards to `Session::install_trace_sink` and reports `TraceSinkInstall::Installed`; without it, it logs at warn level naming the disabled feature and returns `TraceSinkInstall::FeatureDisabled` without touching the session — an observable outcome, never `Error::Unsupported` and never a silent success.
- `otel::is_enabled()` is `cfg!(feature = "otel")` as a `const fn`, and `otel::install_trace_sink` is the compatibility wrapper with the same signature in both builds: it discards the outcome, so a feature-off build keeps returning `Ok(())` (with the feature it still propagates the session's error).

### `config`, `constants`

- `SessionConfig` (Python `config/base.py`): `envs`, `workflow_configs`, `agent_config`, plus defaults `comp_stream_call_timeout_ms = -1`, `stream_input_generator_timeout_ms = -1`, `end_comp_template_branch_render_timeout_ms = 5`, `end_comp_template_render_position_timeout_ms = 5`, `execute_timeout_ms = 60_000`, `stream_frame_timeout_ms = -1`, `stream_first_frame_timeout_ms = -1`, `loop_number_max_limit = 1000`, `force_del_workflow_state = false`; `new()`/`from_env()`, `set_env`, `get_env`, `get_envs`, `set_agent_config`/`agent_config`, `add_workflow_config`/`workflow_config`.
- Env keys read by `from_env`: `WORKFLOW_EXECUTE_TIMEOUT`, `WORKFLOW_STREAM_FRAME_TIMEOUT`, `WORKFLOW_STREAM_FIRST_FRAME_TIMEOUT`, `COMP_STREAM_CALL_TIMEOUT`, `STREAM_INPUT_GEN_TIMEOUT`, `LOOP_NUMBER_MAX_LIMIT`, `FORCE_DEL_WORKFLOW_STATE` (`"true"`/`"false"` only).
- `constants`: env key names, `DEFAULT_NODE_ID = "default"`, `DEFAULT_WORKFLOW_ID = "workflow"`, `SINGLE_COMPONENT_EXECUTION_KEY = "_single_component_execution"`, `DEFAULT_STREAM_CAPACITY`, `END_FRAME`.

## Constraints

- Follows `docs/ARCHITECTURE.md` §2.4 exactly for the `Session` trait seam; additions there are provided (defaulted) methods only, so dependent crates keep compiling with small fakes.
- Errors: `openjiuwen_common::error::{Error, Result}`; use `Error::Session` for storage/checkpoint failures, `Error::Validation` for bad patches/ids, `Error::Timeout`/`Error::Cancelled` for HITL; never panic on user input.
- Async-first tokio; locks are `parking_lot` or `tokio::sync` and are never held across `.await`; public streams are `Send + 'static`; cancellation via `CancellationToken`; no `std::sync` lock held across await.
- Every public data type is serde-serializable, snake_case (except trace spans, which keep Python's camelCase wire aliases), tolerant of unknown fields, with `#[serde(default)]`.
- No `unsafe`; `tracing` only (targets `openjiuwen_session::<module>`); no `println!`/`eprintln!`.
- Crate features: `sqlite` (SQLite checkpointing) and `otel` (a marker that enables the real trace-sink installation path, with no dependency); the Redis/RESP backend (clear text and TLS), the disk store, the wrapper sessions, and the VCS history compile unconditionally and need no network at build time.
- The new areas follow the same rules as the rest of the crate: offline and deterministic, no blocking I/O on async tasks (`tokio::fs`), atomic file writes (temp + rename), `tracing` only, and typed errors (`Error::{Validation, Session, Unsupported}`) instead of silent no-ops — except `RouterSession`'s documented `end_stream`/`close` lifecycle no-ops and the `otel` helper's observed `FeatureDisabled` outcome.
- The TLS stack is pinned inline in `Cargo.toml` (`rustls 0.23` with `ring`, `tokio-rustls 0.26`, `webpki-roots 1`) because the workspace manifest has no entry for it; no PEM or base64 crate is used (the decoder lives in `checkpoint/resp_tls.rs`).
- Files ≤ ~1000 lines; `openjiuwen`-specific docs per subdirectory live in that subdirectory's `CONTEXT.md`.
- No dependency on `openjiuwen-harness-protocol` (layer-1 rule in `docs/ARCHITECTURE.md` §1); interaction↔harness-protocol conversion is documented below and left to the bridging crate.

## Design Decisions

- **One `Session` trait, two state lanes.** Python's `State` has io/global/comp/workflow lanes plus staging; §2.4 exposes only global + component. `WorkflowState` keeps all four lanes and staging for the graph/workflow engines; sessions expose global + component through the trait.
- **Component ids are nested paths.** Python nests committed component state by `executable_id` (`a.b` → `{"a":{"b":…}}`); we preserve that so JSON snapshots and `${node.field}` resolution stay compatible.
- **Streaming is a bounded broadcast hub**, not Python's single-consumer unbounded asyncio queue: multiple subscribers are allowed, ordering is per-subscriber, lagging subscribers skip frames with a warning, completion is an explicit `StreamEvent::End` (Python's `END_FRAME` sentinel is filtered out of `stream_output` and never reaches consumers), and every publish reports a `StreamSendOutcome` so an unobserved frame is observable instead of a silent drop.
- **HITL adds timeout and cancellation** (Python has none) because the Rust runtime is cancellation-first; the wire frame and `InteractiveInput` shapes are unchanged.
- **Checkpoints are JSON**, not pickles (`docs/PARITY.md` intentional deviation); namespaced key layout (`agent`, `agent-team`, `workflow`, `workflow-graph`) is preserved, and `Checkpoint` field names are the JSON contract.
- **Pickle is an explicit codec, not a storage format.** `checkpoint::pickle` maps Python streams onto `serde_json::Value` and the checkpoint dict, so Python interop is opt-in and no backend has to guess a payload format; the type mapping's lossiness (tuples, `bytes`, string-only keys) is documented on the module.
- **TLS rides the hand-rolled RESP2 client.** `resp_tls` owns the `rustls` configuration and handshake and `RespConnection` boxes the transport, so `rediss://` and `redis://` share the codec, verify against `webpki_roots` plus an optional `tls_ca_file`, and never fall back to clear text; a failed handshake is a typed `Error::Session` naming the host.
- **One `TraceSpan`** with a `kind` discriminator replaces Python's `TraceAgentSpan`/`TraceWorkflowSpan` class pair; wire keys stay camelCase, and the two frame types (`tracer_agent`, `tracer_workflow`) are preserved.
- **No process-global checkpointer singleton.** Python's `CheckpointerFactory` default singleton is replaced by explicit `Arc<dyn Checkpointer>` injection at construction; facades above own the instance, which keeps tests isolated and avoids undocumented global mutable state.
- **No ambient session ContextVar.** Python's `get_current_session`/`with_session` decorators exist to smuggle a session into components; Rust passes `Arc<dyn Session>` explicitly, so the ambient mechanism is not ported.
- **Session state writes bypass staging.** `Session::update_state` must be immediately visible to `Session::get_state`, so sessions call `StateStore::write_global` (a committed-lane merge) rather than `update_global`, which `WorkflowState` stages for the graph/workflow commit cycle.
- **Workflow sessions expose a visible global lane.** `WorkflowSession` builds `WorkflowState::with_global_lane(false)` (not the workflow-only `WorkflowState::new()`), so the documented `Session` semantics — an empty `get_state("")` key returns the whole global object — hold for a parentless workflow too.
- **A team's child agent session owns its own session id** and records `parent_session_id`; Python instead reuses the team's session id for every member. Sharing one id would make member checkpoints collide, so the Rust port separates them while keeping the shared hub/config/trace id.
- **`InteractionDecision` wire values follow Python's approval vocabulary** (`allow`, `allow_always`, `deny`) rather than the harness protocol's (`allow_for_session`, `abort`); the mapping table below is the only coupling.
- **A hand-written RESP2 client instead of a Redis crate.** The workspace builds offline and pins no Redis dependency, so the backend speaks RESP2 over `tokio::net::TcpStream`; only the commands the checkpointer needs are implemented, one serialized connection is retried once after a transport failure, and `rediss://` wraps the same socket in a `tokio-rustls` session (webpki roots plus an optional CA file) instead of connecting in the clear.
- **Redis stores the same JSON document as the file backend.** One document per checkpoint under `{prefix}:session:{id}:checkpoint:{cid}`, two index sets for per-session and global discovery, and `EXPIRE` per document when a TTL is configured; prefixes replace database separation, so two checkpointers on one server stay isolated.
- **`session_controller` stores a `SessionSnapshot` as its `data` document.** Python's data container is opaque; persisting the crate's snapshot JSON keeps `state.data` loadable through `Session::restore` with no adapter, while the `{"meta": …, "data": …}` envelope preserves Python's metadata and index shape.
- **VCS history is one JSON file per version with sequential ids**, not git or a WAL: ids are allocated under a store-level lock, each version records its `parent`, and diffing is a pure function over JSON, so history stays offline, inspectable, and dependency-free.
- **Refs, rewinds and forks sit on the JSON history.** `HEAD`/`main` are plain refs, a soft rewind only moves `HEAD` and keeps the newer versions, a hard rewind truncates them and re-points `main`, and `fork` copies a version into a new session record with a fresh one-version history, so lineage never needs git or a WAL.
- **The session link graph is metadata, not a `ChainSession`.** `links.json` records parent/fork lineage per agent and `SessionLinks` traverses and validates it, but state fan-out needs a second shadow store and belongs to the crates that own a run loop, so a session stays an `Arc<dyn Session>` the caller owns.
- **Scope resolution never guesses.** A session whose scope the index does not record is `Error::Validation` (the caller uses `upsert_in`/`save_in`), and an unparseable index key is `Error::Session`, so a corrupt index can never silently move a session into `main()`.
- **Wrapper sessions reproduce Python's selective no-ops, typed where a no-op would be a lie.** `SubWorkflowSession` delegates everything but the workflow placement and end-output slot; `RouterSession` passes reads through, rejects state writes, stream writes, restores and interaction resolution with `Error::Unsupported` naming `inner()`, and keeps `end_stream`/`close` as silent lifecycle no-ops because a nested router must never close the run's stream.
- **The trace sink is swapped behind a stable reference.** `Session::trace()` must keep returning the same reference because snapshots and checkpoints serialize its position, so the session and its tracer share one `Arc<SwappableTraceSink>` and installation replaces the inner sink rather than the field.

## Python Mapping

| Python | Rust |
|---|---|
| `session.BaseSession`, `ProxySession` | `session::Session` trait; children hold `Arc<dyn Session>` |
| `internal/workflow.py` `WorkflowSession`, `NodeSession`, `SubWorkflowSession` | `sessions::{WorkflowSession, NodeSession, SubWorkflowSession}` (`SubWorkflowSession` re-parents a host `NodeSession`, keeping its executable id and bumping the nesting depth) |
| `internal/wrapper.py` `StateSession`, `RouterSession` | `sessions::RouterSession` (read-through reads, typed `Unsupported` write/stream/restore/interaction-resolve rejections, `end_stream`/`close` lifecycle no-ops, `Unsupported` interaction and sink installation) |
| `internal/agent.py` `AgentSession` | `sessions::AgentSession` |
| `internal/agent_team.py` `AgentTeamSession` | `sessions::AgentTeamSession` |
| `workflow.Session`, `agent.Session`, `agent_team.Session`, `node.Session` facades | concrete session types + options structs; component-facing conveniences stay in `openjiuwen-workflow`/`openjiuwen-agent` |
| `state/commit_state.py` `CommitState` | `state::WorkflowState` |
| `state/agent_state.py` `StateCollection` | `state::AgentState` |
| `state/utils.py` nested paths, `update_dict`, `${ref}` | `state::paths` |
| `stream/schema.py`, `stream/emitter.py`, `stream/manager.py` | `stream::{StreamItem, StreamEvent, StreamSendOutcome, StreamHub, StreamWriterManager}` |
| `interaction/` `InteractiveInput`, `InteractionOutput`, `WorkflowInteraction`, `SimpleAgentInteraction` | `interaction::{InteractiveInput, InteractionOutput, InteractionRequest, InteractionResponse, InteractionBroker}` |
| `checkpointer/base.py`, `factory.py`, `inmemory.py`, `persistence.py` (pickle payloads), Redis backend | `checkpoint::{Checkpointer, CheckpointerConfig, create_checkpointer, MemoryCheckpointer, FileCheckpointer, SqliteCheckpointer, RedisCheckpointer, RedisOptions}` plus `checkpoint::pickle::{loads, dumps, checkpoint_from_pickle, checkpoint_to_pickle}` for the payload interop |
| Python's `redis` client library (used by the Redis checkpointer) | `checkpoint::resp` — the dependency-free RESP2 codec and connection (`RespValue`, `encode_command`, `read_value`, `RespConnection`), with `checkpoint::resp_tls` supplying `rediss://` TLS |
| `session_controller/` (`scope.py`, `schema.py`, storage, `downstreams/` links) | `session_controller::{Scope, Subject, SessionScope, SessionScopeKey, SessionMeta, SessionRecord, SessionsIndex, DiskSessionStore, SessionController, SessionLinks, SessionLink, LinkKind, LinksIndex}` (storage subset plus the metadata link graph) |
| `vcs/` (local JSON version history, `VersioningManager` restore/rewind, forks) | `vcs::{SessionHistory, Version, VersionId, VersionMeta, Refs, RefName, RefEntry, RewindOptions, diff_values, diff_snapshots}` (local subset with refs, rewinds and forks) |
| `tracer/span.py`, `tracer/tracer.py`, `tracer/span_manager.py` | `tracer::{TraceSpan, TraceFrame, Tracer, SpanHandle}` |
| `tracer/handler.py` registry (`TraceExtAgentHandler`, `TraceExtWorkflowHandler`) | `tracer::TraceSink` (one synchronous sink trait instead of two async handler families) |
| `BaseSession.set_tracer` + handler registration | `Session::install_trace_sink` over `trace_hook::SwappableTraceSink`; `otel::install_trace_sink` is the feature-gated entry point (the OTel SDK sink lives in `openjiuwen-extensions`) |
| `config/base.py` `Config`, `workflow_session_vars` | `config::SessionConfig` (env vars only; no ContextVar override) |

### Interaction ↔ `openjiuwen-harness-protocol` mapping (documented, not coded here)

| `session` | harness protocol |
|---|---|
| `InteractionKind::Permission` + `InteractionDecision::Approve/ApproveAlways/Reject/Cancel` | `ToolApprovalRequest` + `ToolApprovalDecision` (`allow`/`allow_for_session`/`deny`/`abort`) |
| `InteractionKind::AskUser` + `InteractionDecision::Answer/Reject/Cancel` | `UserInputRequest` + `UserInputResponse` (`InteractionResponseStatus`) |
| `InteractionKind::ClientTool` | `DynamicToolCallRequest`/`Response` |
| `InteractionKind::Permission` with an MCP server in metadata | `McpElicitationRequest`/`Response` |

## Routing Table

- `./src/session.rs` → `Session` trait, `SessionType`, `SessionMetadata`, `SessionSnapshot`
- `./src/sessions/` → `WorkflowSession`, `NodeSession`, `AgentSession`, `AgentTeamSession`, options structs, factories (`sessions/CONTEXT.md`)
- `./src/sessions/sub_workflow.rs` → `SubWorkflowSession`: host-node re-parenting, workflow placement overrides, end-output slot
- `./src/sessions/router.rs` → `RouterSession`: the read-only router view, its typed write rejections and its lifecycle no-ops
- `./src/sessions/shared.rs` → crate-private `SessionInner`/`resolve_shared` plumbing (hub, config, tracer, checkpointer, trace sink)
- `./src/state/` → `StateStore`, `WorkflowState` lanes + staging, `AgentState`, nested-path helpers (`state/CONTEXT.md`)
- `./src/stream.rs` → `StreamMode`, `StreamItem`, `StreamEvent`, `StreamSendOutcome`, `StreamHub`, `StreamWriterManager`
- `./src/interaction.rs` → HITL requests/responses, `InteractiveInput`, `InteractionBroker`
- `./src/checkpoint/` → `Checkpointer` trait, checkpoint model, keys, config, memory/file/sqlite/redis backends, pickle interop (`checkpoint/CONTEXT.md`)
- `./src/checkpoint/resp.rs` → the dependency-free RESP2 codec and connection (`RespValue`, `encode_command`, `read_value`, `RespConnection`)
- `./src/checkpoint/resp_tls.rs` → the `rediss://` transport: rustls client config over webpki roots plus a CA file, server-name derivation, handshake, hand-rolled PEM/base64 decoding
- `./src/checkpoint/pickle.rs` → the Python-pickle codec (`loads`/`dumps`, `BYTES_KEY`) and the `Checkpoint` payload mapping
- `./src/checkpoint/redis.rs` → the Redis backend (`RedisCheckpointer`, `RedisOptions` incl. `tls_ca_file`/`from_config`, key/index layout, TTL, single-connection reconnect)
- `./src/session_controller/` → `Scope`/`Subject`/`SessionScope`/`SessionScopeKey`, `SessionMeta`/`SessionRecord`/`SessionsIndex`, `DiskSessionStore`, `SessionController`, the `links.json` link graph (`session_controller/CONTEXT.md`)
- `./src/vcs/` → `VersionId`/`Version`/`VersionMeta`, `SessionHistory` (refs, rewinds, forks), `diff_values`/`diff_snapshots` (`vcs/CONTEXT.md`)
- `./src/trace_hook.rs` → `SwappableTraceSink`, the stable sink handle sessions swap behind
- `./src/otel.rs` → the feature-gated `install_trace_sink`/`try_install_trace_sink`/`TraceSinkInstall` helper (an observed `FeatureDisabled` outcome without the feature)
- `./src/tracer.rs` → `TraceSpan`, `TraceFrame`, `TraceSink`, `Tracer`, spans, sinks
- `./src/config.rs` → `SessionConfig`, env resolution, defaults
- `./src/constants.rs` → env keys, default ids, frame types, stream capacity
- `./tests/` → crate-level behavior tests: state scopes, streaming and send outcomes, HITL, checkpoint/resume (memory/file/sqlite/redis, clear text and TLS, pickle interop), disk sessions and their link graph, VCS history/refs/rewinds/forks, wrappers and the trace hook, tracer, session factories (`tests/CONTEXT.md`)
- `./tests/support/` → the shared backend contract and the loopback `FakeRedis` (clear text and TLS); `./tests/fixtures/tls/` → the static CA/server certificates those TLS tests load

## Test Strategy

- Unit tests beside modules for pure helpers (`state::paths`, config env resolution, checkpoint key building, tracer times); integration tests under `./tests/` for cross-module behavior.
- Integration files: `tests/state.rs`, `tests/stream.rs`, `tests/interaction.rs`, `tests/checkpoint.rs`, `tests/checkpoint_sqlite.rs` (feature-gated), `tests/checkpoint_redis.rs`, `tests/checkpoint_redis_tls.rs`, `tests/checkpoint_pickle.rs`, `tests/session_controller.rs`, `tests/session_controller_links.rs`, `tests/vcs.rs`, `tests/session_wrappers.rs`, `tests/tracer.rs`, `tests/config.rs`, and `tests/sessions.rs`, with the shared backend contract (`run_backend_contract(backend, Arc<dyn Checkpointer>, ListOrder::{OldestFirst, NewestFirst})`), the fake RESP server (clear text and TLS) and the static TLS certificates under `tests/support/` and `tests/fixtures/tls/`.
- `tests/checkpoint_redis.rs` runs the shared contract against `ListOrder::OldestFirst` on the loopback `FakeRedis` and adds the Redis specifics: documented key layout, `EXPIRE`/TTL, prefix isolation, cross-session resume on one server, factory wiring from `extra`, missing-URL validation, AUTH/SELECT from the URL, reconnect after the server hangs up, connection failures as `Error::Session`, and pruning of expired index members.
- `tests/checkpoint_redis_tls.rs` runs the shared contract again over the loopback server's TLS mode against the static fixtures: trusting the fixture CA through `tls_ca_file`, key layout and TTL parity with clear text, `uses_tls` selecting by scheme, a TLS client against a clear-text server failing as `Error::Session` naming the host before any command is sent, an untrusted certificate (PEM or DER) as `Error::Session`, a malformed CA file as `Error::Validation`, the factory reading `tls_ca_file` from `extra`, and a CA file paired with a `redis://` URL as `Error::Validation`.
- `tests/checkpoint_pickle.rs` covers the codec: scalar/text/binary round-trips, memo and shared references, `bytes` through `BYTES_KEY`, dict/list/tuple handling, `BINFLOAT`, long integers, protocol-4 framing that `CPython` can read, `checkpoint_from_pickle`/`checkpoint_to_pickle` round-trips, and every typed rejection (non-string keys, oversized integers, deep nesting, unknown opcodes, bad framing).
- `tests/session_controller.rs` covers the disk layout (`sessions.json` + `state.data`), workflow/agent snapshot round-trips, scope isolation and the active-session marker, newest-first listings, `list_agents`, delete, invalid ids, atomic writes leaving no temporary files, corrupt documents as `Error::Session`, and the controller cache/flush/save paths, including the strict scope-resolution errors and the `upsert_in`/`save_in` explicit-scope paths.
- `tests/session_controller_links.rs` covers `links.json`: parent and fork links, `parent_of`/`children_of`/`forks_of`, `roots`/`leaves`, `chain`/`ancestors`/`descendants`/`depth`, cycle-safe traversal, self and cycle-creating links as `Error::Validation`, a stored cycle as `Error::Session`, link removal on session delete, concurrent writes through cloned stores, and the atomic layout.
- `tests/vcs.rs` covers sequential commits and parent links, nested/array diff paths, restore without committing, delete/clear, concurrent commits receiving unique ids, `VersionId` round-trips, invalid ids, refs (`set_ref`/`resolve_ref`/`delete_ref`/`list_refs`, the `HEAD`/`main` advance on commit, the `head()` fallback to `latest()`), soft and hard rewinds, and `fork` (scope rules, unknown source, existing target).
- `tests/session_wrappers.rs` covers sub-workflow identity/placement/state/end-output, the router's read-through plus every typed rejection (state writes, stream writes, restore, interaction resolution, interaction requests, sink installation) and its `end_stream`/`close` no-ops, trace-hook installation (including shared sinks and checkpoint positions), the `otel` helper in both feature states through `try_install_trace_sink`/`is_enabled`/`TraceSinkInstall` plus the `Session` default `Unsupported`, and the snake_case wire names of the install outcome.
- Unit tests inside the new modules cover the RESP codec/URL parsing (`checkpoint/resp.rs`), the TLS client config, PEM marker/base64 decoding and server-name derivation (`checkpoint/resp_tls.rs`), the pickle opcode decoding and its rejections (`checkpoint/pickle.rs`), Redis options/validation/redaction (`checkpoint/redis.rs`), the wrapper sessions (`sessions/sub_workflow.rs`, `sessions/router.rs`), the swappable sink (`trace_hook.rs`), scope/subject string forms (`session_controller/scope.rs`), the session schema (`session_controller/schema.rs`), path builders (`session_controller/store.rs`), link-graph linking and traversal (`session_controller/links.rs`), ref names and the refs document (`vcs/refs.rs`), and version ids plus diffing (`vcs/version.rs`, `vcs/diff.rs`).
- `tests/sessions.rs` pins the contract through `Arc<dyn Session>` with a minimal inline fake (so the provided `metadata`/`snapshot`/`restore`/`save_checkpoint`/`resume`/`subscribe` defaults are what runs) and covers all four concrete sessions, the workflow child of an agent, team source metadata, checkpoint round-trips, an end-to-end trace-position save/resume through a fresh `WorkflowSession`, and concurrent updates.
- Covered suites: state paths/merge/`null` deletion/dotted paths, nested component ids, staging commit/rollback, concurrent updates, stream ordering + multiple subscribers + end marker + late subscribers + mode filtering + every `StreamSendOutcome` + the `StreamTraceSink` position only advancing on delivery, interaction approve/reject/answer/timeout/cancellation/seeded answers, checkpoint save/load/list/latest/delete/release/exists round-trips on memory/file/sqlite/redis (clear text and TLS), the pickle codec and checkpoint interop, the backend contract (ordering, overwrite, invalid ids), disk session store/controller round-trips and atomicity plus the link graph and strict scope errors, VCS commit/diff/restore/concurrency plus refs, rewinds and forks, wrapper-session read-through/no-op/end-output and router rejections, the trace-sink hook and the `otel` install outcomes, tracer span lifecycle, child linking, kind/frame mapping, sink recording and position, config env resolution and defaults.
- Deterministic and offline: no network, no sleeps except short `tokio::time::timeout` guards, `tempfile` for file/sqlite/disk-session/VCS roots, the in-process `FakeRedis` (clear text or TLS) instead of a Redis server, static certificate fixtures instead of live TLS endpoints, and monotonic assertions instead of timing assertions.
- Behavior ports cite the Python test they mirror in a comment (e.g. `// ports tests/unit_tests/core/session/state/test_...`).
- Definition of done: `cargo fmt -p openjiuwen-session`, `cargo test -p openjiuwen-session --offline`, `cargo test -p openjiuwen-session --all-features --offline`, `cargo clippy -p openjiuwen-session --all-targets --all-features -- -D warnings` (and the default feature set), and `cargo doc -p openjiuwen-session --no-deps`, all clean; `--all-features` covers `sqlite` and `otel` together, so the per-feature runs are subsumed.
- `tests/fixtures/tls/README.md` records where the static CA/server certificates came from and how to regenerate them; nothing there is secret (host-only certificates valid to 2040, with the server key used solely by the loopback test server), and deliberately malformed CA files are written into temp directories instead.

## Notes for Agents

- `openjiuwen::core::session` is this crate's root (`openjiuwen_core/src/lib.rs` does `pub use openjiuwen_session as session;`), so every `lib.rs` crate-root `pub use` is reachable as `openjiuwen::core::session::X`; the inner `session` module still exists at `openjiuwen::core::session::session::{Session, SessionType, SessionMetadata, SessionSnapshot}`.
- Checkpointer types are reachable both as `openjiuwen::core::session::{Checkpointer, Checkpoint, CheckpointMeta, CheckpointerConfig, CheckpointerKind, create_checkpointer, MemoryCheckpointer, FileCheckpointer, RedisCheckpointer, RedisOptions, build_key, build_key_with_namespace, checkpoint_from_pickle, checkpoint_to_pickle}` (crate-root re-exports in `src/lib.rs`) and nested at `openjiuwen::core::session::checkpoint::*`.
- The wrappers, the trace hook, the disk store, the link graph and the VCS are crate-root re-exports too: `RouterSession`, `SubWorkflowSession`, `SwappableTraceSink`, `install_trace_sink`/`try_install_trace_sink`/`TraceSinkInstall`/`is_enabled` (the `otel` helpers), `DiskSessionStore`, `SessionController`, `SessionMeta`, `SessionRecord`, `SessionScope`, `SessionScopeKey`, `Scope`, `Subject`, `SessionLinks`, `SessionLink`, `LinkKind`, `LinksIndex`, `SessionHistory`, `Version`, `VersionId`, `VersionMeta`, `Refs`, `RefName`, `RefEntry`, `RewindOptions`, `HEAD_REF`, `MAIN_REF`, `MAX_REF_NAME_LEN`, `diff_values`, `diff_snapshots`, `ChangeKind`, `JsonChange`; the pickle codec is re-exported as `loads`, `dumps`, `BYTES_KEY` (reach the rest through `checkpoint::pickle`).
- `graph`/`workflow` should take `Arc<WorkflowState>` from `WorkflowSession::state()` for the io/workflow lanes and staging; the `Session` trait intentionally exposes only global + component state.
- `AgentSession` is a state/stream/trace container only: it stores no conversation history or message list, and this crate has no `AgentStreamMode`, `StreamData`, or agent-chunk type — agent crates push chunks as `StreamItem::output`/`output_at`/`custom` frames.
- `AgentSessionOptions` has no `config` field: a fresh session gets `SessionConfig::new()` (env-seeded, then `options.envs` applied) or inherits the parent's config; afterwards it is mutable only through `Session::config()` (`Arc<parking_lot::RwLock<SessionConfig>>`).
- Sessions write the committed global lane through `StateStore::write_global`; `update_global` keeps Python's staging behavior for graph/workflow commit cycles — do not swap the two.
- Use `StreamWriterManager::write_now` (and `StreamHub::send_now`) from synchronous code such as trace sinks and callbacks; use `write`/`send` in async code. Both are non-blocking.
- `request_interaction` only completes when another task calls `resolve_interaction` (normally the host transport reading `__interaction__` frames); seeded `InteractiveInput::interrupt_answers` short-circuit it on resume.
- Facades above own the `Arc<dyn Checkpointer>` and inject it through the options structs; there is no process-global default checkpointer.
- `save_checkpoint`/`resume`/`get_state`/`update_state` are `Session` trait methods, so calling them on an `AgentSession` requires importing the trait (`use openjiuwen::core::session::{AgentSession, Session};`); only `new`/`with_options`/`state`/`commit`/`post_run` are inherent.
- `FileCheckpointer::new(root)` is infallible and creates the directory lazily on the first save; a memory and a file backend passed to two sessions must be the *same* instance (or same root) to resume across sessions.
- `Session::trace()` on a concrete session returns its `SwappableTraceSink`, which forwards to a `StreamTraceSink` over the writer manager by default — that is the checkpointed trace position; `session.install_trace_sink(sink)` reroutes the spans started afterwards for the session and its children without replacing the reference `trace()` hands out, whereas `Tracer::set_sink` swaps only the tracer's own view (so `trace()` stops observing those spans).
- Interaction vocabulary maps to `openjiuwen-harness-protocol` by table only (see the mapping above); this crate must not depend on the protocol crate (layer rule, `docs/ARCHITECTURE.md` §1).
- `src/tracer.rs` is ~1150 lines because it holds the span model, frame type, three sinks, the tracer, and their inline tests; split it when another sink or span family lands.
- `src/checkpoint/pickle.rs` is ~1450 lines and stays one file on purpose: it is a single codec (unpickler, pickler, the `bytes`/tuple/`__bytes__` type mapping, the checkpoint dict mapping) whose opcode tests sit beside the arms they exercise; split it only along the read/write seam.
- The Redis backend is always compiled and has no optional dependency, but `RedisCheckpointer::connect`/`connect_with_options` dial eagerly, so constructing one needs a reachable server; offline tests use `tests/support/resp_server::FakeRedis` (`start`, `start_tls`, `url`, `url_tls`, `url_with`, `commands`, `count`, `command`, `hang_up_after_each_reply`).
- A TLS checkpointer is `RedisCheckpointer::connect("rediss://host:port")` for a publicly trusted server, or `connect_with_options(RedisOptions { url, tls_ca_file: Some("<pem or der path>"), .. })` to add a private CA to the compiled-in webpki roots; `tls_ca_file` with a `redis://` URL is `Error::Validation`, so the scheme decides the transport.
- Encoding a Python-written checkpoint is `checkpoint_from_pickle(bytes)` (JSON payloads via `pickle::loads`); producing one for Python is `checkpoint_to_pickle(&checkpoint)`. The codec is lossy by design: tuples become arrays, `bytes` become `{"__bytes__": "<hex>"}`, dict keys must be strings, and anything outside the opcode subset is `Error::Validation` — no backend writes or auto-detects pickle.
- The session link graph lives at `<base>/<agent_id>/links.json` and is reached through the store (`link_parent`, `link_fork`, `parent_of`, `children_of`, `ancestors`, `descendants`, `chain`) or by loading `read_links() -> SessionLinks` and using the traversal helpers; a self or cycle-creating link is `Error::Validation`, a cycle already on disk is `Error::Session`, and deleting a session drops its entry.
- `SessionHistory` keeps `HEAD`/`main` refs in `<history>/refs.json`: `head()` falls back to `latest()`, `commit` advances both refs, and `resolve_ref`/`set_ref` refuse a target version that does not exist.
- `rewind`/`rewind_hard` are the two recipes over `rewind_with(RewindOptions)`: the soft rewind restores the version and moves `HEAD` but keeps newer versions, the hard one also deletes them and re-points `main`; both restore first and fail with `Error::Session` when the target version is missing.
- `fork` writes the chosen version's snapshot as a new session record in the *source* session's scope, so pass a `new_session_id` that is unique for that scope; an unknown source session or a taken target id is `Error::Validation`.
- `StreamSendOutcome` tells a producer whether anyone saw the frame: `Delivered { subscribers }`, `NoSubscribers`, `Closed`, `ModeDisabled`; it is not `must_use`, so ignore it for fire-and-forget writes, but note that `StreamTraceSink` advances the trace position only on delivery (an unobserved session checkpoints `trace: None`).
- `RedisOptions` for a declarative factory config: `CheckpointerConfig { kind: CheckpointerKind::Redis, extra: {"url", "prefix", "ttl_seconds", "database", "tls_ca_file"} }` (`tls-ca-file` is an accepted alias); `path` is only a fallback for `url`, and a missing URL is `Error::Validation`.
- `SubWorkflowSession::new(&host_node, workflow_id)` re-parents to the host node's parent, so a nested workflow reads and writes the parent workflow's lanes under the host's executable id (the host's checkpoint stays readable); an empty `workflow_id` keeps the host's workflow id.
- `SubWorkflowSession::propagate_end_output` merges `{"output": …}` into the host lane and leaves the slot intact (idempotent, no consumption), while `take_end_output` consumes it and `end_output` only reads it.
- `RouterSession` rejects `update_state`/`update_component_state`/`write_stream`/`restore`/`resolve_interaction` with `Error::Unsupported` whose reason names `inner()`, and hides `config`/`writers`/`tracer`/`checkpointer` (`None`); use `inner()` when code outside a router callback needs to write or to reach the wrapped session, and expect `end_stream`/`close` to succeed as no-ops.
- Install a real exporter with `otel::install_trace_sink(&session, sink)` (returns `Ok(())` in both feature states); use `try_install_trace_sink` when the caller must distinguish `TraceSinkInstall::Installed` from `FeatureDisabled`, `is_enabled()` to branch ahead of time, or `Session::install_trace_sink` directly to observe the trait default and the wrapper's `Unsupported`.
- `DiskSessionStore` persists `SessionSnapshot` JSON, so a record written by any session kind loads back through `load_into_session`/`Session::restore`; `SessionHistory` is created from a store (`store.history()`) and writes under that session's `history/` directory.

## Known Gaps and Deferred Work

- No `store.py` (unused in Python; `MemoryStore` is not public API) and no ambient-session ContextVar: sessions are passed explicitly.
- `session_controller` still defers data-container/permission policies and the process-global `GlobalSessionController` singleton; the link graph is metadata only, so Python's `ChainSession` state fan-out is not ported (sessions stay `Arc<dyn Session>`, see the design decisions).
- `vcs` still defers git and other external backends, event/WAL (`e<N>`) references, codecs, and context-engine adapters; refs, rewinds and forks are local JSON.
- Redis has no cluster or Sentinel support and no pipelining: commands are serialized over one connection per checkpointer.
- A cluster/Sentinel URL is not detected: `ConnectionTarget::parse` (`checkpoint/resp.rs`) takes a comma-separated host list as one host, so `connect` fails as a transport `Error::Session` instead of reporting `Unsupported`.
- The pickle codec covers the opcode subset Python emits for plain data, not arbitrary pickles: objects built with `REDUCE`/`BUILD`, persistent ids, buffers and out-of-band data are `Error::Validation`, and the tuple/`bytes`/string-key restrictions mean a round-trip through the codec can lose type information.
- Tracer extensions: the per-session sink hook exists, but the real OTel SDK sink lives in `openjiuwen-extensions`; there is no process-global handler registry — sinks are injected per session.
- Python's `SubWorkflowSession` also inherits the host node's `node_name`; the Rust port has no node-name concept, so `entity_name` stays the component id (the executable id matches Python's: `a`, or `a.b` under a nested node).
- Concrete sessions checkpoint only the trace position the installed sink observes since it was created or rebased, so a resumed session's `frame_count` continues from the checkpointed baseline and a session whose frames were never delivered checkpoints `trace: None`.
- `AgentSession::dump_state`/`AgentTeamSession::dump_state` expose exactly `global_state` + `agent_state`; Python's agent dump also includes `trace_state`. The trace lane is reachable through `state().trace_state()`.
- Python's team-session state sharing (members writing into the team's global lane) is not reproduced: every team child session owns an independent state store, while a `SubWorkflowSession` does share the parent workflow's store.
