# sessions

## Intent

Concrete `Session` implementations and their factories.
Ports `openjiuwen/core/session/internal/` (v0.1.20): `WorkflowSession`, `NodeSession`, `AgentSession`, `AgentTeamSession`, plus the `workflow.Session` / `agent.Session` / `agent_team.Session` construction parameters as options structs.

## API Surface

- `shared::SessionInner` (crate-private) — id, parent id, kind, entity id/name, shared `Arc<RwLock<SessionConfig>>`, shared `Arc<StreamHub>` + `Arc<StreamWriterManager>`, `Arc<InteractionBroker>`, `Tracer`, `Arc<dyn StateStore>`, optional `Arc<dyn Checkpointer>`, optional parent `Arc<dyn Session>`, metadata, creation timestamp.
- `WorkflowSession` — `new(session_id)`, `with_config`, `with_parent`, `state() -> Arc<WorkflowState>`, `create_node_session(component_id) -> NodeSession`, `dump_state()`, `save_checkpoint`, `resume`.
- `NodeSession` — component-scoped view over a parent session: executable id, nested component state, delegated streaming/tracing/interaction; `end_stream`/`close` are no-ops that never close the parent.
- `AgentSession` — agent and team state, `agent_id`/`agent_name`/`description` metadata, `env(key, default)`, `envs()`, `create_workflow_session()`, `commit(checkpoint_id)`, `resume(checkpoint_id)`.
- `AgentTeamSession` — `team_id`, global state, `create_agent_session(agent_id)` child tagged with `source_agent_id`/`source_team_id`.
- Factories and options: `create_workflow_session(WorkflowSessionOptions)`, `create_agent_session(AgentSessionOptions)`, `create_agent_team_session(AgentTeamSessionOptions)`.

## Constraints

- Children share the parent's hub, writer manager, and config by `Arc`; a child never owns a second stream.
- `Session` trait semantics (state merge, `null` deletion, component nesting, `__interaction__` frames, timeout/cancellation) are defined in `../CONTEXT.md` and must not be re-derived here.
- Checkpoint save/resume builds a `Checkpoint` from `SessionSnapshot` (global + components + trace position) and restores through `StateStore::restore`.
- Session ids: caller-supplied or `uuid_hex()` (parentless workflow) / `uuid_v4()` (agent/team facades), matching Python id shapes.
- `save_checkpoint` without a configured checkpointer is `Error::Session`; `resume` with a missing checkpoint returns `Ok(false)`.
- No `unsafe`, no stdout logging, files ≤ ~1000 lines.

## Routing Table

- `./mod.rs` → module wiring, re-exports, factories, options structs
- `./shared.rs` → `SessionInner` and shared plumbing (state store, hub, broker, tracer, checkpoint snapshot/restore helpers)
- `./workflow.rs` → `WorkflowSession`, `NodeSession`
- `./agent.rs` → `AgentSession`
- `./agent_team.rs` → `AgentTeamSession`
