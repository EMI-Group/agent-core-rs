# agent
## Intent
The `DeepAgent` facade: configuration, builder/factory assembly, single-round and task-loop invocation, streaming, follow-up/steering/abort, pause/resume, mode switching, and session-persisted state.
Ports `openjiuwen/harness/deep_agent.py` and `factory.py`; `DeepAgent` composes the kernel `ReActAgent` and never re-implements the model/tool loop.
## API Surface
- `DeepAgent::new(AgentCard)`, `configure(DeepAgentConfig)`, `ensure_initialized()`, `invoke(inputs, session)` / `stream(inputs, session)` (impl `openjiuwen_agent::Agent`), `follow_up(msg, task_id)`, `steer(msg)`, `abort()`, `start()/stop()`, `send_input(SendInputRequest)`, `phase()`, `switch_mode(mode)`, `load_state()`/`save_state()`/`clear_state()`, `add_rail(Arc<dyn HarnessRail>)` / `register_rail(Arc<dyn AgentRail>)`, `rails()`.
- `DeepAgentBuilder`: `new(card)`, `config(DeepAgentConfig)`, `model(Arc<dyn Model>)`, `sys_operation(SysOperation)`, `workspace(Workspace)`, `tool(Arc<dyn Tool>)`/`tools(...)`, `rail(Arc<dyn HarnessRail>)`, `permission_host(Arc<dyn PermissionHost>)`, `user_asker(Arc<dyn crate::tools::UserAsker>)`, `memory(Arc<dyn MemoryRecall>, Arc<dyn MemoryWriteBack>)`, `subagent_factory(Arc<dyn SubAgentFactory>)`, `build() -> Result<Arc<DeepAgent>>`.
- Model: when no model is supplied, `ensure_initialized` builds one lazily from `DeepAgentConfig` (`model_client_config` + `model_config`, falling back to the flat `model_name`/`model_provider`/`api_key`/`api_base`/`custom_headers` fields) through `openjiuwen_foundation::llm::init_model`; impossible configuration → `Error::config`.
- `create_deep_agent(...) -> Result<Arc<DeepAgent>>` — async factory (Python keyword surface: model, card, config, tools, rails, workspace, skills, subagents, enable_task_loop, max_iterations, language, prompt_mode, restrict_to_work_dir, permissions, permission_host).
- `DeepAgentParts { config, rails, tools }` + `resolve_deep_agent_parts(...)` (pure assembly), `apply_deep_agent_parts(agent, parts)`.
- `state_store`: `load_state(&dyn Session)`, `save_state(&dyn Session, &DeepAgentState)`, `clear_state(&dyn Session)` under session state key `deepagent`.
- Input normalization: string, `{"query": ...}`, or an interactive/resume payload; reserved keys consumed, others become prompt variables; a non-string/non-object or a missing `query` → `Error::validation` (Python `DEEPAGENT_INPUT_PARAM_ERROR`).
## Constraints
- Composition: `DeepAgent` owns an `Arc<ReActAgent>` (kernel) and mounts harness rails through `HarnessRailAdapter`; tools are mounted on the inner agent from one `ToolRegistry`.
- `DeepAgentConfig` is serde data; live handles only through the builder. Missing/unbuildable model → `Error::config`.
- One invocation at a time per `DeepAgent`: a second concurrent invocation returns `Error::config`/`Error::validation` ("invocation already running"), never interleaves session state.
- Task loop runs only when `config.enable_task_loop`; otherwise exactly one ReAct turn. `max_iterations_hard_limit` still lowers the inner budget.
- State is saved after every task-loop round and after a single round; resume payloads never append a duplicate user message.
- Streaming: `BoxStream<'static, Result<AgentChunk>>`; interaction frames (`__interaction__`) surface to the caller; the terminal frame is the inner agent's `answer` frame.
- Cancellation: `abort()` is synchronous, idempotent (`CancellationToken::cancel` + `TaskLoopController::request_abort`); the in-flight round observes it and yields `RoundOutcome::abort()`.
- No `unsafe`, no process globals; `Arc<DeepAgent>` is shareable; locks (`parking_lot`) are never held across `.await`.
- Errors: `openjiuwen_common::error::{Error, Result}`; Python `build_error(code, msg)` text is preserved in the message.
- Secrets never logged (`api_key` is `SecretString`); `tracing` only, CLI owns stdout.
## Design decisions
- **Prompt assembly**: `DeepAgent` builds a `HarnessPromptBuilder` from the config (identity from `system_prompt` else the card description, safety, tools from the mounted cards, workspace, task loop when enabled, mode instructions, todo when a plan exists, subagent section when subagents exist) and the inner `ReActAgent` receives that static text as its single `system` `prompt_template` message.
  Per-invocation sections published by rails (`skills`, `memory`, `subagent`) are rendered by `PromptRail` (a `HarnessRail`, priority 5, i.e. last) in `before_model_call`: it replaces the leading system message with `builder.build()` plus published sections that have no static counterpart, deduplicated by section name.
  This keeps the prompt pure data assembly (`crate::prompts`) and gives rails the documented `PROMPT_SECTIONS_EXTRA_KEY` channel.
- **Steering**: `PromptRail` also drains `LoopQueues::steering` in `before_model_call` and appends one `user` message per steering input (`User steering: <content>`) to the model window, so steering lands on the next model call without restarting the round; `before_steering_drain` fires from `round.rs` before the inner invoke.
  Steering text is not persisted to the session message history (documented deviation: the kernel has no mid-run user-message API).
- **Task loop ownership**: `DeepAgent` stores `controller: parking_lot::Mutex<Option<TaskLoopController>>`; `invoke` takes the controller out, awaits `TaskLoopController::run_until_idle(&RoundExecutorHandle)`, and puts it back, so no lock is held across `.await`.
  `RoundExecutorHandle` is a separate `Send + Sync` value cloned from `Arc`s (inner agent, prompt pieces, queues, config, cancellation, token counter) — it never borrows `DeepAgent`.
- **Round granularity**: one task-loop round = one `ReActAgent::invoke` call. `run_round` fires `on_user_message` + `before_steering_drain`, runs the inner invoke, maps `AgentOutput` (`error` → `RoundOutcome::failed`, `interrupt` → successful round with `next_work = None` so the loop ends and the caller sees the interrupt), and fire `after_task_iteration`/`before_task_iteration` around task-loop rounds through `HarnessRailRegistry::fire_task_iteration`.
- **Follow-ups**: `follow_up` pushes onto the shared `LoopQueues`; a running loop consumes one follow-up per round (`is_follow_up = true`); an idle agent queues it for the next `invoke`, and `DeepAgentState::pending_follow_ups` persists the queue between sessions.
- **State**: `state_store` uses `Session::get_state("deepagent")` / `Session::update_state({"deepagent": ...})` and `DeepAgentState::to_session_dict`/`from_session_dict`; a missing key or non-object payload degrades to default with a debug log, while a stored-but-corrupt object surfaces `Error::session`.
  The controller's `StateSink` writes the state after each round; `save_state`/`load_state`/`clear_state` are the explicit entry points.
- **Subagents**: `SubAgentRegistry::with_builtin_presets(language)` + `SubAgentRuntime` with a `DeepAgentSubAgentFactory` that builds a nested `DeepAgent` from `SubAgentConfig` (depth 1: nested agents never mount subagent tools again), re-rooted at `<workspace>/sub-agents/<sub_session_id>` and inheriting the parent's model and language.
  `TaskTool` + the six `subagent_*` tools are mounted only when `subagents` is non-empty or `add_general_purpose_agent` is set.
- **Permissions**: `config.permissions` builds one `Arc<PermissionEngine>` shared by `ToolRegistry::with_permissions` and `PermissionInterruptRail`, which is registered on the inner agent (priority 90) when permissions are enabled; the `PermissionHost` defaults to `NoopPermissionHost`.
- **Phases**: `InteractionPhase` mirrors the controller (`Idle`/`Running`/`Terminated`); `start`/`stop` drive the controller token, `send_input` routes `FollowUp`/`Steer` (mode `None` → follow-up when idle, steer while running) through `LoopQueues`.
- **Read-only agent access**: `rails()` returns the harness registry; `register_rail` forwards kernel rails to the inner `ReActAgent`; `add_rail` registers a harness rail in both the harness registry and (via `to_agent_rails`) the inner agent.
## Routing Table
- `./mod.rs` → `DeepAgent` struct, `Agent` impl, follow-up/steer/abort, mode/state entry points, re-exports
- `./builder.rs` → `DeepAgentBuilder`, tool/rail/permission/subagent wiring, lazy model resolution
- `./factory.rs` → `create_deep_agent`, `DeepAgentParts`, `resolve_deep_agent_parts`, default rails
- `./round.rs` → input normalization, single-round execution, `RoundExecutorHandle`, phase/abort mapping, `stream`
- `./prompt_rail.rs` → `PromptRail` (system-prompt assembly from builder + published sections, steering injection)
- `./subagent_factory.rs` → `DeepAgentSubAgentFactory` (nested `DeepAgent` construction for the runtime)
- `./state_store.rs` → session-persisted `DeepAgentState` (load/save/clear)
- `./tests.rs` → node unit/integration suite with `ScriptedModel` and temp workspaces
## Test Strategy
- Offline, deterministic tests in `./tests.rs` with `openjiuwen_test_support::ScriptedModel` and `tempfile` workspaces: configure + defaults, prompt includes identity/tools/workspace, single-round answer, tool-call round (fake echo tool), max-iteration error mapping, follow-up queue + state persistence, abort outcome, permission denial skips the tool, mode switch, `send_input`/phase transitions, steering injection, prompt rail section merge.
- Kernel-only fakes: a `FakePermissionHost`/`UserAsker`, no network, no real model provider (lazy model construction is unit-tested only where it can fail fast).
- Parity references: `tests/unit_tests/harness/test_deep_agent.py`, `test_deep_agent_session_state.py`, `test_deep_agent_interaction.py`, `test_deep_agent_workspace.py`.
