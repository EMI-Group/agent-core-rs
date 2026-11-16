# sessions

## Intent

Concrete `Session` implementations, the wrapper sessions, and their factories.
Ports `openjiuwen/core/session/internal/` (v0.1.20): `WorkflowSession`, `NodeSession`, `AgentSession`, `AgentTeamSession` from `workflow.py`/`agent.py`/`agent_team.py`, `SubWorkflowSession` from `workflow.py`, and `RouterSession` from `wrapper.py`, plus the `workflow.Session` / `agent.Session` / `agent_team.Session` construction parameters as options structs.

## API Surface

- `shared::SessionInner` (crate-private) — id, parent id, kind, entity id/name, shared `Arc<RwLock<SessionConfig>>`, shared `Arc<StreamHub>` + `Arc<StreamWriterManager>`, `Arc<InteractionBroker>`, `Tracer`, `Arc<dyn StateStore>`, optional `Arc<dyn Checkpointer>`, optional parent `Arc<dyn Session>`, metadata, creation timestamp.
- `SessionInner` owns one `Arc<SwappableTraceSink>` (from `../trace_hook.rs`) that both `Session::trace` and the `Tracer` hold, so `SessionInner::install_trace_sink(sink)` reroutes spans started afterwards without replacing the reference already handed out; `resolve_shared` inherits the parent's config, writers, trace id, and checkpointer when a parent exists.
- `WorkflowSession` — `new(session_id)`, `with_options(WorkflowSessionOptions)`, `workflow_id()`, `state() -> Arc<WorkflowState>`, `dump_state()`, `commit_user_inputs(inputs)`, `create_node_session(component_id) -> NodeSession`, plus the trait's `save_checkpoint`/`resume`.
  There is no `with_config`/`with_parent`: configuration, parent, checkpointer, trace id, and env overrides are `WorkflowSessionOptions` fields passed to `with_options` (or through the `create_workflow_session` factory).
- `WorkflowSession::with_options` stamps the workflow placement into the session metadata: `workflow_id`, `main_workflow_id` (equal to `workflow_id` for a top-level workflow), and `workflow_nesting_depth` (0), each only when the caller did not supply that key.
- `NodeSession` — component-scoped view over a parent session: executable id, nested component state, delegated streaming/tracing/interaction; `end_stream`/`close` are no-ops that never close the parent, `save_checkpoint` uses the executable id as its entity id.
  The executable id follows Python's `create_executable_id`: a node of a workflow/agent session is the bare component id (`a`), and nesting node sessions dot-joins them (`a.b`, `a.b.c`).
  `WorkflowSessionOptions::workflow_id` is entity metadata only; node executable ids follow `create_executable_id`.
- `NodeSession` constructors and accessors: `new(parent, component_id)`, `new_with_type(parent, component_id, node_type)`, `executable_id()`, `component_id()`, `node_type() -> Option<&str>`, `workflow_id()`, `main_workflow_id()`, `workflow_nesting_depth() -> u32`, `parent()`, `create_node_session(component_id)`; `metadata()` reports `SessionType::Node`, the executable id as `entity_id`, the component id as `entity_name`, and the extra metadata keys `executable_id`, `component_id`, and (when supplied) `node_type`.
- `NodeSession` reads its placement from the parent's metadata — `workflow_id` (falling back to `DEFAULT_WORKFLOW_ID`), `main_workflow_id` (falling back to `workflow_id`), and `workflow_nesting_depth` (falling back to 0).
- `SubWorkflowSession::new(&host_node, workflow_id)` — the session a nested workflow runs with: it rebuilds the host component id and node type over `host.parent()` (re-parenting), so it keeps the host node's executable id while overriding the workflow placement (`workflow_id`; an empty argument keeps the host's, never empty), `main_workflow_id` (inherited from the host), and `workflow_nesting_depth` (host depth + 1).
  It adds an end-output slot shared across clones (`set_end_output`, `end_output`, `take_end_output` — consuming, and async `propagate_end_output`, which merges `{"output": value}` into the host node's component lane idempotently without consuming) and delegates state, streams, HITL, tracing, snapshots, and sink installation to the inner node view; `session_type()`/`metadata().kind` are `Node`, and `end_stream`/`close` stay no-ops so a nested workflow never closes the parent hub.
- `RouterSession::new(inner: Arc<dyn Session>)` — the read-only view handed to conditional-edge routers: `id`, `parent_id`, `get_state`, `get_component_state`, `trace`, and `snapshot` read through (with `kind` forced to `Node`), while `update_state`, `update_component_state`, `write_stream`, `restore`, and `resolve_interaction` return `Error::Unsupported` with a per-method reason naming `inner()`; `end_stream`/`close` stay deliberate `Ok(())` lifecycle no-ops, `request_interaction` and `install_trace_sink` are `Error::Unsupported`, `config`/`writers`/`tracer`/`checkpointer` are `None`, `subscribe` is empty, `session_type()`/`metadata().kind` are `Node`, and `inner()`/`session_id()`/`executable_id()` expose the wrapped session.
- `AgentSession` — agent state (`global_state` + `agent_state` lanes), `agent_id`/`agent_name`/`description` metadata, `env(key) -> Option<Value>`, `envs()`, `state()`, `dump_state()`, `create_workflow_session()` sharing the agent hub/config/trace id but owning an independent workflow state, `commit() -> Result<Option<String>>`, idempotent `post_run()`, and the trait's `save_checkpoint`/`resume`.
- `AgentTeamSession` — `team_id` (default `DEFAULT_TEAM_ID = "agent_team"`), `global_state`/`agent_state` lanes, `create_agent_session(agent_id)` child tagged with `source_agent_id`/`source_team_id` and sharing the team hub/config/trace id, `commit()`, `flush_checkpoint()`, idempotent `post_run()`.
- Factories and options: `create_workflow_session(WorkflowSessionOptions)`, `create_agent_session(AgentSessionOptions)`, `create_agent_team_session(AgentTeamSessionOptions)`; the options structs hold trait objects, so they are deliberately not serde-serializable.

## Constraints

- Children share the parent's hub, writer manager, and config by `Arc`; a child never owns a second stream.
- `Session` trait semantics (state merge, `null` deletion, component nesting, `__interaction__` frames, timeout/cancellation) are defined in `../CONTEXT.md` and must not be re-derived here.
- Checkpoint save/resume builds a `Checkpoint` from `SessionSnapshot` (global + components + trace position) and restores through `StateStore::restore`; the namespace follows the session kind and the entity id is the metadata's `entity_id` (the executable id for a node) or the session id.
- Session ids: caller-supplied or `uuid_hex()` (a parentless workflow generates one, a parented workflow defaults to the parent id), matching Python id shapes.
- `save_checkpoint` without a configured checkpointer is `Error::Session`; `resume` with a missing checkpoint returns `Ok(false)`.
- State is in-process and per session object: constructing a second session with the same `session_id` starts with empty lanes, and construction never auto-restores — a caller resumes only by calling `Session::resume(checkpoint_id)` on a session built with a checkpointer over the same storage (and the same session id, since `resume` loads by `self.id()`); `Checkpointer::latest(session_id)` discovers the newest id.
- `Clone` on a concrete session shares the same `SessionInner`/state store, so reusing the clone (rather than constructing a new session) sees prior state; `post_run` closes that shared hub by default (`close_stream_on_post_run`), so a reused session's next run emits frames into a closed stream unless the option is disabled.
- No `unsafe`, no stdout logging, files ≤ ~1000 lines.

## Routing Table

- `./mod.rs` → module wiring, re-exports, factories, options structs
- `./shared.rs` → `SessionInner` and shared plumbing (state store, hub, broker, tracer, stable trace-sink handle, checkpoint snapshot/restore helpers)
- `./workflow.rs` → `WorkflowSession`, `NodeSession`
- `./sub_workflow.rs` → `SubWorkflowSession` (host re-parenting, placement overrides, end-output slot)
- `./router.rs` → `RouterSession` (read-through reads, typed `Unsupported` write/stream/restore/interaction-resolve rejections, `end_stream`/`close` lifecycle no-ops, `Unsupported` interaction and sink installation)
- `./agent.rs` → `AgentSession`
- `./agent_team.rs` → `AgentTeamSession`
