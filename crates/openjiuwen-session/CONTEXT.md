# openjiuwen-session

## Intent

Per-run execution context for the kernel: sessions and their scoped state, streaming frames, human-in-the-loop interactions, checkpoint/resume, and lightweight tracing.
Ports the Python package `openjiuwen.core.session` (v0.1.20): `session.py`, `internal/` (`WorkflowSession`, `NodeSession`, `AgentSession`, `AgentTeamSession`), `state/` (`CommitState` lanes and agent state), `stream/` (`StreamMode`, schemas, emitter, writer manager), `interaction/` (`InteractiveInput`, `InteractionOutput`, workflow/agent interrupts), `checkpointer/`, `tracer/`, and `config/base.py`.
Every graph node, workflow component, agent, team, and runner receives `Arc<dyn Session>` (`docs/ARCHITECTURE.md` §2.4); this crate is the stable seam for state, streaming, HITL, tracing, and resume, and higher layers must not re-implement it.

## API Surface

### `session` — the contract (`src/session.rs`)

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionType { Agent, AgentTeam, Workflow, Node }

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
    fn metadata(&self) -> SessionMetadata;                       // derived from id/parent/kind
    fn checkpointer(&self) -> Option<Arc<dyn Checkpointer>> { None }
    fn tracer(&self) -> Option<&Tracer> { None }
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
- `request_interaction` publishes a `__interaction__` output frame and awaits a [`InteractionResponse`]; a seeded answer (`InteractiveInput::interrupt_answers`) returns immediately. Timeout resolves to `Error::Timeout`, cancellation to `Error::Cancelled`.
- `trace()` returns the session's sink; `tracer()` returns the span factory when the session owns one.
- `snapshot`/`restore` are the checkpointing seam: default implementations cover global state + trace position; concrete sessions add component state.
- `save_checkpoint` returns the checkpoint id (generated when `None`); `resume` loads and restores, returning `Ok(false)` when the checkpoint does not exist.

### `sessions` — concrete sessions (`src/sessions/`)

- `WorkflowSession::new(session_id)` / `with_config` / `with_parent`, `create_workflow_session(WorkflowSessionOptions)`, `state() -> Arc<WorkflowState>`, `dump_state()`, `create_node_session(component_id) -> NodeSession`, `save_checkpoint`, `resume`.
- `NodeSession` — a component-scoped child view: executable id `parent.executable + "." + component_id`; global reads pass through, component reads/writes nest under its executable id, streaming/tracing delegate, `end_stream`/`close` are no-ops.
- `AgentSession::new` / `create_agent_session(AgentSessionOptions)` — agent/team state (`global` + `agent` lanes), agent id/name metadata, env access, `create_workflow_session()` sharing the agent session id and hub, `commit()`, idempotent `post_run()`, `resume`.
- `AgentTeamSession::new` / `create_agent_team_session(AgentTeamSessionOptions)` — team id (default `DEFAULT_TEAM_ID = "agent_team"`), global lane, `create_agent_session(agent_id)` child carrying `source_agent_id`/`source_team_id`, `commit()`/`flush_checkpoint()`, idempotent `post_run()`.
- Every session shares `Arc<StreamHub>` + `Arc<StreamWriterManager>` down to children; config and tracer are shared by reference; each session tracks its own interaction broker.

### `state` — state stores (`src/state/`)

- `StateStore` trait (used by every session): `get_global`, `update_global`, `get_component`, `update_component`, `components`, `dump`, `restore`, `trace_state`, `update_trace`.
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
pub struct StreamHub;                                // bounded broadcast transport
pub struct StreamWriterManager;                      // mode routing over one hub
```

- `StreamItem::output(item_type, index, payload)`, `::trace(item_type, payload)`, `::custom(payload)`, `is_end()`, `to_wire() -> Value` (`{"type","index","payload"}`, mode dropped for wire parity).
- `StreamHub::new()` (capacity `DEFAULT_STREAM_CAPACITY = 1024`), `with_capacity`, `send_now`, `send`, `subscribe() -> BoxStream<'static, Result<StreamEvent>>`, `close`, `is_closed`, `capacity`.
- `send_now` is the canonical publish path: a synchronous `tokio::sync::broadcast` send with no I/O and no await, usable from synchronous trace sinks and from any thread; `send` is an `async` wrapper that never awaits internally (kept for API symmetry with Python).
- `StreamWriterManager::new(hub)` (all modes), `with_modes(hub, modes)`, `write_now`, `write`, `subscribe`, `close`, `is_closed`, `modes`, `enables`, `hub`.
- `write_now` is the canonical write path (mode check, then `StreamHub::send_now`); `write` is its `async` wrapper, also without an internal await.
- Ordered delivery per subscriber, multiple subscribers, bounded buffering (a lagging subscriber logs a warning and skips frames rather than blocking producers), and a terminal `End` event visible to late subscribers.
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
- Backends: `MemoryCheckpointer` (`new()`, `shared()` process-wide singleton, `len()`, `is_empty()`, `clear()`), `FileCheckpointer` (`new(root)`, `<root>/<session_id>/<checkpoint_id>.json`, atomic write through a `.<uuid>.tmp` file plus rename, corrupt JSON is `Error::Session`), `SqliteCheckpointer` (feature `sqlite`, `open(path).await` / `in_memory().await`, one `session_checkpoints` table, a single upsert statement per save, corrupt payload is `Error::Session`).
- `CheckpointerConfig { kind, path, extra }`, `CheckpointerKind { InMemory, File, Sqlite }`, `create_checkpointer(&config).await -> Result<Arc<dyn Checkpointer>>`; `file` and `sqlite` require a `path` (`Error::Validation` otherwise), and `sqlite` additionally requires the `sqlite` feature (`Error::Unsupported` without it).
- `Checkpoint` derives serde with snake_case field names; checkpoint ids are generated (`uuid_hex()`) when empty; `latest` sorts `(created_at, checkpoint_id)` descending itself, so it does not depend on backend list order; session and checkpoint ids are validated (no path separators, no `..`).
- `list` order is backend-defined: `MemoryCheckpointer` and `FileCheckpointer` return `(created_at, checkpoint_id)` ascending, `SqliteCheckpointer` newest first; callers that need a specific order must sort the result.
- Resume restores global state, component state, updates, and the trace position.

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

- Sinks: `NoopTraceSink` (the default), `MemoryTraceSink` (records frames; `frames()`, `take()`, `len()`, `is_empty()`, `clear()`), `StreamTraceSink` (publishes through an `Arc<StreamWriterManager>` with `write_now` and only logs at debug level, dropping the frame, when the write or the JSON serialization fails).
- `Tracer::new(session_id, trace_id)`, `with_sink`, `set_sink`, `start_span(name, invoke_type)` (an agent span), `start_workflow_span(component_id, name)`, `start_child(&parent, name, invoke_type)`, `sink() -> Arc<dyn TraceSink>` (a cloned handle, because the sink slot lives behind an `RwLock`), `trace_id()`, `session_id()`.
- `start_workflow_span` mints a `Component` span with `component_id` set, `workflow_id == component_id` and `invoke_type = Workflow`; `start_child` inherits the parent's `TraceSpanKind`, records `parentInvokeId`, and appends the child id to the parent's `childInvokes`.
- `SpanHandle` takes `&self` everywhere (interior mutability through `Arc<Mutex<SpanState>>`); clones share one span, `finish`/`fail`/`interrupted` emit exactly one frame and later terminal transitions are no-ops, and the lock is released before the sink runs.
- Agent spans emit as `tracer_agent` frames; workflow and component spans emit as `tracer_workflow`; `TraceFrame::agent(span)` / `TraceFrame::workflow(span)` take only the span because `session_id`/`trace_id` live on `TraceSpan`, and `to_stream_item()` projects a frame onto a trace-mode `StreamItem`.
- Empty span names default to the kind name (`agent`, `workflow`, `component`); an empty trace id is replaced by a generated uuid; `TracePosition` stays snake_case because it is the checkpoint JSON shape.
- `MemoryTraceSink::position` derives `(trace_id, last_invoke_id, frame_count)` from the emitted frames; `set_position` installs the frame-count baseline and resets the emitted counter, and `clear()` resets both the frames and the derived position.

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
- Files ≤ ~1000 lines; `openjiuwen`-specific docs per subdirectory live in that subdirectory's `CONTEXT.md`.
- No dependency on `openjiuwen-harness-protocol` (layer-1 rule in `docs/ARCHITECTURE.md` §1); interaction↔harness-protocol conversion is documented below and left to the bridging crate.

## Design Decisions

- **One `Session` trait, two state lanes.** Python's `State` has io/global/comp/workflow lanes plus staging; §2.4 exposes only global + component. `WorkflowState` keeps all four lanes and staging for the graph/workflow engines; sessions expose global + component through the trait.
- **Component ids are nested paths.** Python nests committed component state by `executable_id` (`a.b` → `{"a":{"b":…}}`); we preserve that so JSON snapshots and `${node.field}` resolution stay compatible.
- **Streaming is a bounded broadcast hub**, not Python's single-consumer unbounded asyncio queue: multiple subscribers are allowed, ordering is per-subscriber, lagging subscribers skip frames with a warning, and completion is an explicit `StreamEvent::End` (Python's `END_FRAME` sentinel is filtered out of `stream_output` and never reaches consumers).
- **HITL adds timeout and cancellation** (Python has none) because the Rust runtime is cancellation-first; the wire frame and `InteractiveInput` shapes are unchanged.
- **Checkpoints are JSON**, not pickles (`docs/PARITY.md` intentional deviation); namespaced key layout (`agent`, `agent-team`, `workflow`, `workflow-graph`) is preserved, and `Checkpoint` field names are the JSON contract.
- **One `TraceSpan`** with a `kind` discriminator replaces Python's `TraceAgentSpan`/`TraceWorkflowSpan` class pair; wire keys stay camelCase, and the two frame types (`tracer_agent`, `tracer_workflow`) are preserved.
- **No process-global checkpointer singleton.** Python's `CheckpointerFactory` default singleton is replaced by explicit `Arc<dyn Checkpointer>` injection at construction; facades above own the instance, which keeps tests isolated and avoids undocumented global mutable state.
- **No ambient session ContextVar.** Python's `get_current_session`/`with_session` decorators exist to smuggle a session into components; Rust passes `Arc<dyn Session>` explicitly, so the ambient mechanism is not ported.
- **`InteractionDecision` wire values follow Python's approval vocabulary** (`allow`, `allow_always`, `deny`) rather than the harness protocol's (`allow_for_session`, `abort`); the mapping table below is the only coupling.

## Python Mapping

| Python | Rust |
|---|---|
| `session.BaseSession`, `ProxySession` | `session::Session` trait; children hold `Arc<dyn Session>` |
| `internal/workflow.py` `WorkflowSession`, `NodeSession`, `SubWorkflowSession` | `sessions::WorkflowSession`, `sessions::NodeSession` (nested workflows use a child `WorkflowSession`) |
| `internal/agent.py` `AgentSession` | `sessions::AgentSession` |
| `internal/agent_team.py` `AgentTeamSession` | `sessions::AgentTeamSession` |
| `workflow.Session`, `agent.Session`, `agent_team.Session`, `node.Session` facades | concrete session types + options structs; component-facing conveniences stay in `openjiuwen-workflow`/`openjiuwen-agent` |
| `state/commit_state.py` `CommitState` | `state::WorkflowState` |
| `state/agent_state.py` `StateCollection` | `state::AgentState` |
| `state/utils.py` nested paths, `update_dict`, `${ref}` | `state::paths` |
| `stream/schema.py`, `stream/emitter.py`, `stream/manager.py` | `stream::{StreamItem, StreamEvent, StreamHub, StreamWriterManager}` |
| `interaction/` `InteractiveInput`, `InteractionOutput`, `WorkflowInteraction`, `SimpleAgentInteraction` | `interaction::{InteractiveInput, InteractionOutput, InteractionRequest, InteractionResponse, InteractionBroker}` |
| `checkpointer/base.py`, `factory.py`, `inmemory.py`, `persistence.py` | `checkpoint::{Checkpointer, CheckpointerConfig, create_checkpointer, MemoryCheckpointer, FileCheckpointer, SqliteCheckpointer}` |
| `tracer/span.py`, `tracer/tracer.py`, `tracer/span_manager.py` | `tracer::{TraceSpan, TraceFrame, Tracer, SpanHandle}` |
| `tracer/handler.py` registry (`TraceExtAgentHandler`, `TraceExtWorkflowHandler`) | `tracer::TraceSink` (one synchronous sink trait instead of two async handler families) |
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
- `./src/state/` → `StateStore`, `WorkflowState` lanes + staging, `AgentState`, nested-path helpers (`state/CONTEXT.md`)
- `./src/stream.rs` → `StreamMode`, `StreamItem`, `StreamEvent`, `StreamHub`, `StreamWriterManager`
- `./src/interaction.rs` → HITL requests/responses, `InteractiveInput`, `InteractionBroker`
- `./src/checkpoint/` → `Checkpointer` trait, checkpoint model, keys, config, memory/file/sqlite backends (`checkpoint/CONTEXT.md`)
- `./src/tracer.rs` → `TraceSpan`, `TraceFrame`, `TraceSink`, `Tracer`, spans, sinks
- `./src/config.rs` → `SessionConfig`, env resolution, defaults
- `./src/constants.rs` → env keys, default ids, frame types, stream capacity
- `./tests/` → crate-level behavior tests: state scopes, streaming, HITL, checkpoint/resume, tracer, session factories (`tests/CONTEXT.md`)

## Test Strategy

- Unit tests beside modules for pure helpers (`state::paths`, config env resolution, checkpoint key building, tracer times); integration tests under `./tests/` for cross-module behavior.
- Integration files: `tests/state.rs`, `tests/stream.rs`, `tests/interaction.rs`, `tests/checkpoint.rs`, `tests/tracer.rs`, `tests/config.rs`, and `tests/checkpoint_sqlite.rs` (feature-gated), with the shared backend contract in `tests/support/mod.rs`.
- Covered suites: state paths/merge/`null` deletion/dotted paths, nested component ids, staging commit/rollback, concurrent updates, stream ordering + multiple subscribers + end marker + late subscribers + mode filtering, interaction approve/reject/answer/timeout/cancellation/seeded answers, checkpoint save/load/list/latest/delete/release/exists round-trips on memory/file/sqlite, the backend contract (ordering, overwrite, invalid ids), tracer span lifecycle, child linking, kind/frame mapping, sink recording and position, config env resolution and defaults.
- Deterministic and offline: no network, no sleeps except short `tokio::time::timeout` guards, `tempfile` for file/sqlite backends, monotonic assertions instead of timing assertions.
- Behavior ports cite the Python test they mirror in a comment (e.g. `// ports tests/unit_tests/core/session/state/test_...`).
- Gates: `cargo fmt -p openjiuwen-session`, `cargo test -p openjiuwen-session`, `cargo test -p openjiuwen-session --features sqlite`, `cargo clippy -p openjiuwen-session --all-targets -- -D warnings`, all clean.

## Known Gaps and Deferred Work

- No Redis/remote checkpointer (trait seam only), no `session_controller/` disk-backed product sessions, no `vcs/` version history, no `store.py` (unused in Python; `MemoryStore` is not public API), no ambient-session ContextVar.
- Python's checkpoint payloads are pickles; only JSON-compatible state is preserved, so Python-produced pickles cannot be read.
- Tracer extensions: no OTel (that lives in `openjiuwen-extensions`), no process-global handler registry — sinks are injected per session.
- `SubWorkflowSession`'s id-flattening quirk and `RouterSession`'s selective no-ops are not ported; nested workflows use a plain child `WorkflowSession`.
