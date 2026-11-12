# agent

## Intent

The `DeepAgent` facade: configuration, builder/factory assembly, single-round and task-loop invocation, streaming, follow-up/steering/abort, pause/resume, mode switching, and session-persisted state.
Ports `openjiuwen/harness/deep_agent.py` and `factory.py`.

## API Surface

- `DeepAgent::new(AgentCard)`, `configure(DeepAgentConfig)`, `ensure_initialized()`, `invoke(inputs, session)` / `stream(inputs, session)` (impl `openjiuwen_agent::Agent`), `follow_up(msg, task_id)`, `steer(msg)`, `abort()`, `start()/stop()`, `send_input(SendInputRequest)`, `phase()`, `switch_mode(mode)`, `load_state()/save_state()/clear_state()`.
- `DeepAgentBuilder`: `new(card)`, `config(DeepAgentConfig)`, `model(Arc<dyn Model>)`, `sys_operation(SysOperation)`, `workspace(Workspace)`, `tool(Arc<dyn Tool>)`/`tools(...)`, `rail(Arc<dyn HarnessRail>)`, `permission_host(Arc<dyn PermissionHost>)`, `memory(Arc<...>)`, `build() -> Result<Arc<DeepAgent>>`.
- `create_deep_agent(...) -> Result<Arc<DeepAgent>>` — async factory (Python keyword surface: model, card, tools, rails, workspace, skills, subagents, enable_task_loop, max_iterations, language, prompt_mode, restrict_to_work_dir, permissions, …).
- `DeepAgentParts { config, rails, tools }` + `resolve_deep_agent_parts(...)` (pure assembly), `apply_deep_agent_parts(agent, parts)`.
- `state_store`: `load_state(session) -> DeepAgentState`, `save_state(session, &state)`, `clear_state(session)` under session state key `deepagent`.
- Default rails added by the factory: `SecurityRail` (when `enable_security_rail`), `SkillsRail` (skills or discovery), `SubagentRail` (subagents), `MemoryRail` (when a memory is configured); `PermissionInterruptRail` when permissions are enabled.
- Input normalization: accepts a string, `{"query": ...}`, or an interactive/resume payload; reserved keys consumed, others become prompt variables; invalid input → `Error::validation` (Python `DEEPAGENT_INPUT_PARAM_ERROR`).

## Constraints

- Composition: `DeepAgent` owns a `ReActAgent` (kernel) and mounts harness rails through `HarnessRailAdapter`; the harness never re-implements the model/tool loop.
- `DeepAgentConfig` is serde data; live handles only through the builder. Missing model → `Error::config`.
- One invocation at a time per session; a concurrent `invoke` on the same session returns `Error::session`/state error.
- Task loop runs only when `enable_task_loop` and no hard iteration limit; otherwise exactly one ReAct turn.
- State is saved after every task-loop round and after a single round; resume payloads never append a duplicate user message.
- Streaming: `BoxStream<'static, Result<AgentChunk>>`; interaction frames (`__interaction__`) surface to the caller.
- Cancellation: `abort()` is synchronous and idempotent; in-flight streaming tasks are cancelled and joined.
- No `unsafe`, no globals; `Arc<DeepAgent>` is shareable.

## Routing Table

- `./mod.rs` → `DeepAgent` struct, `Agent` impl, follow-up/steer/abort, mode/state entry points
- `./builder.rs` → `DeepAgentBuilder` and wiring of tools/rails/permissions
- `./factory.rs` → `create_deep_agent`, `DeepAgentParts`, `resolve_deep_agent_parts`, default rails
- `./round.rs` → single-round execution, task-loop driving, steering drain, output mapping
- `./state_store.rs` → session-persisted `DeepAgentState`

## Test Strategy

- Integration tests (`tests/deep_agent_loop.rs`, `tests/state_persistence.rs`): no-tool answer, one tool round, max-iteration error, follow-up queue, abort, interrupt/resume, state round-trip, mode switch, prompt sections include workspace/tools.
- Parity references: `tests/unit_tests/harness/test_deep_agent.py`, `test_deep_agent_session_state.py`, `test_deep_agent_interaction.py`, `test_deep_agent_workspace.py`.
