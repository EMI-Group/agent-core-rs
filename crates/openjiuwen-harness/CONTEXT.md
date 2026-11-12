# openjiuwen-harness (DeepAgent)

## Intent

The flagship coding-agent framework ("DeepAgent"), ported from the Python package `openjiuwen.harness` (v0.1.20, ~184k lines).
It composes the kernel (`openjiuwen-core`: agent, session, sysop, security, memory, context) into a task-loop agent with a workspace, a permission engine, coding tools, rails, prompts, skills and subagents.
It also supplies the native implementation of the provider-neutral SPI in `openjiuwen-harness-protocol` and the agent logic the `openjiuwen-cli` binary calls.
Fidelity target: public Python entry points, persisted JSON (session state, todo, skill library), tool names/JSON schemas, permission outcomes, and loop semantics — internals are idiomatic Rust.

## API Surface

Entry points (`openjiuwen_harness::agent`):
- `DeepAgent` — `new(AgentCard)`, `configure(DeepAgentConfig)`, `invoke`/`stream` (impl `openjiuwen_agent::Agent`), `follow_up`, `steer`, `abort`, `start`/`stop`/`send_input`, `load_state`/`save_state`, `switch_mode`, `add_rail`/`register_rail`.
- `DeepAgentConfig`, `DeepAgentBuilder`, `create_deep_agent(...)` (async factory), `AgentMode`, `DeepAgentState`.
Task loop (`openjiuwen_harness::task_loop`): `TaskLoopController`, `TaskLoopEventHandler`, `LoopCoordinator`, `LoopQueues`, `DeepLoopEvent`.
Schema (`openjiuwen_harness::schema`): config/state/task/stop-condition/interaction/loop-event types.
Workspace (`openjiuwen_harness::workspace`): `Workspace`, `WorkspaceNode`, `DirectoryNode`, `DirectoryBuilder`.
Permissions (`openjiuwen_harness::permission`): `PermissionEngine`, `PermissionLevel`, `PermissionResult`, `PermissionConfirmResponse`, `PermissionsSection`, `PermissionInterruptRail`.
Tools (`openjiuwen_harness::tools`): the model-facing tool set — `bash`, `read_file`, `write_file`, `edit_file`, `glob`, `list_files`, `grep`, `todo_create`/`todo_list`/`todo_get`/`todo_modify`, `ask_user`, `memory_search`/`memory_get`/`write_memory`/`read_memory`/`edit_memory`, `coding_memory_read`/`coding_memory_write`/`coding_memory_edit` — plus `BuiltinTool`/`ToolRegistry`; `task_tool` and the `subagent_*` tools are mounted by `subagents`.
Rails (`openjiuwen_harness::rails`): `HarnessRail`, `HarnessRailContext`, `HarnessRailRegistry`, `SecurityRail`, `MemoryRail`, `SkillsRail`, `SubagentRail`, `EvolutionRail`.
Prompts (`openjiuwen_harness::prompts`): `HarnessPromptBuilder`, `PromptSection`, `SectionName`, `PromptMode`, `PromptReport`.
Subagents (`openjiuwen_harness::subagents`): `SubAgentConfig`, `SubAgentRegistry`, `SubAgentPreset`, `SubAgentRuntime`, `TaskTool`.
Skills (`openjiuwen_harness::skills`): `Skill`, `SkillLibrary`, `SkillRootMount`, `SkillResourceValidator`.
Manifest (`openjiuwen_harness::manifest`): `ElementKind`, `HarnessElementDescriptor`, `Catalog`.
Protocol (`openjiuwen_harness::protocol`): `NativeHarness` (impl `HarnessProtocol` + `HarnessModelControl`), `NativeHarnessProvider` (impl `HarnessProvider` + `ModelConfigProvider`).
Observability (`openjiuwen_harness::observability`): span/event name helpers and the `TelemetrySink` seam.
CLI (`openjiuwen_harness::cli`): `Cli`, `run`, `Repl`.
Every public item is also re-exported flat from the crate root.

## Constraints

- Depends only on `openjiuwen-core` (kernel umbrella) and `openjiuwen-harness-protocol`; never on `openjiuwen-agent-teams`, `openjiuwen-rsi`, `openjiuwen-agent-evolving`, `openjiuwen-symphony`, `openjiuwen-harness-providers`.
- Upper-layer seams are traits: `SubAgentSpawner` (agent-teams/providers), `EvolutionRail` (deferred evolution), `TelemetrySink` (extensions/OTel).
- Errors: `openjiuwen_common::error::{Error, Result}` everywhere; Python `build_error(code, msg)` maps to `Error::config`/`validation`/`security`/`tool`/`unsupported` with the Python message text. No panics on user input.
- Async-first on tokio; cancellation via `tokio_util::sync::CancellationToken`; locks never held across `.await`.
- Serde: snake_case, tolerant of unknown fields; persisted session JSON keeps Python keys (`deepagent`, `iteration`, `task_plan`, `pending_follow_ups`, `plan_mode`, `todos`, `skills_state`).
- Secrets are never logged; `api_key` is `SecretString`; `Debug` of configs redacts credentials.
- Logging via `tracing` only; no stdout in library code (CLI owns stdout).
- Files ≤1000 lines; shared helpers live in the lowest owning module. `permission/model.rs` and `permission/paths.rs` exceed it only through inline test modules and carry a top-of-file size note.
- Tests: integration tests in `./tests/`, unit tests beside modules; `ScriptedModel` from `openjiuwen-test-support`; offline, deterministic.
- Features: `native` (default), `browser`, `cua`, `mobile` (deferred preset stubs), `notify` (workspace watcher), `live-tests` (opt-in network tests); the crate must also build with `--no-default-features`.

## Module dependency contract

`schema` ← every module (no harness dependencies).
`workspace`, `skills`, `manifest` ← `schema`.
`permission` ← `schema`, `openjiuwen-security`, session interactions.
`prompts` ← `schema`, foundation `ToolCard` only (no dependency on `tools`).
`rails` ← `schema`, `prompts`, `skills`, `permission`, kernel memory/security.
`subagents` ← `schema`, `prompts`, agent runtime.
`tools` ← `schema`, `workspace`, `permission`, sysop/foundation, kernel memory.
`task_loop` ← `schema`, agent.
`agent` ← all of the above (facade).
`protocol` ← `agent`, `schema`, `openjiuwen-harness-protocol`.
`cli` ← `agent`, `protocol`, `permission`.
`observability` ← tracing only.

## Design decisions

- Python `DeepAgentConfig` mixes live objects (`Model`, rails) with data; the Rust `DeepAgentConfig` is serde data only, and live handles (`Arc<dyn Model>`, tools, rails, `SysOperation`) are supplied through `DeepAgentBuilder`. Documented deviation.
- Python's 13 `AgentCallbackEvent` hooks are split: the nine kernel hooks are reused through `openjiuwen_agent::AgentRail`; harness rails additionally implement task-loop hooks. `HarnessRail` owns all hooks and a `HarnessRailAdapter` bridges the kernel subset into the inner `ReActAgent` rail registry.
- The task loop is explicit controller + event handler (not an actor framework): rounds are driven by `TaskLoopController::run_round`, follow-ups/steering live in `LoopQueues`, and state is persisted after each round.
- Protocol capability gating: `NativeHarness` declares steer, graceful/force abort, pause_resume, persistent_session, checkpoint, native_tools, mcp_tools, model_selection, model_discovery; anything not declared returns `HarnessError::UnsupportedCapability`.
- Permission "no opinion" is `None`, folded with `strictest` (deny < ask < allow); absence of a matching rule yields ASK, never ALLOW.
- Tools never panic and never return `Err` for expected failures: they return a `ToolOutput`-shaped JSON payload with `success=false` and `error`.

## Deferred Python areas

- `subagents/{browser,cua,mobile_gui}_agent.py`, `tools/{browser_move,cua,mobile_gui,cua}` — feature-gated stubs (`browser`, `cua`, `mobile`), return `Error::Unsupported`.
- `lsp/`, `lsp_tool/` — no Rust LSP client; deferred.
- `goal/`, `heartbeat_rail`, `personal_context`, `moa/`, `kv_cache/`, `persistent_agent/` — deferred; rail seams noted but not implemented.
- `observability/` OTel exporter — seam only (`TelemetrySink`); exporter belongs to `openjiuwen-extensions`.
- `resources/`, `extension_binder.py`, plugin/agent-template loading (`DeepAgentSpec`) — only the manifest catalog is ported.
- `subagent_runtime/` full quota/LRU/activity/transcript streams — ported as in-process spawn/wait/list/send_input/close with bounded concurrency; persistence buckets simplified.
- Progressive tool discovery, web search/fetch tools, MCP resource tools, worktree, cron, image/audio multimodal tools — deferred.
- `agent_evolving` / RSI evolution rail bodies — trait seam only.

## Routing Table

- `./src/schema/` → config, state, task/todo, stop conditions, interaction, loop events, agent mode
- `./src/workspace/` → `Workspace`, named directory nodes, default schema, `DirectoryBuilder`
- `./src/permission/` → permission model, engine, file/shell guards, session decisions, HITL interrupt rail
- `./src/tools/` → model-facing coding tools, tool registry, shared tool helpers
- `./src/rails/` → harness rail trait/registry plus security, memory, skills, subagent, evolution rails
- `./src/prompts/` → system prompt builder, sections, tool descriptions, prompt report, sanitization
- `./src/subagents/` → `SubAgentConfig`, registry, presets, in-process runtime, task tool
- `./src/skills/` → skill discovery/library state, mounting, `SKILL.md` validation
- `./src/task_loop/` → loop coordinator, queues, event manager, task-loop controller/handler
- `./src/agent/` → `DeepAgent`, builder, factory, session state persistence, round execution
- `./src/protocol/` → native harness protocol/provider implementation, event bridge, cursors
- `./src/cli/` → clap `run` command and interactive REPL with permission prompts
- `./src/manifest.rs` → harness element catalog (`@harness_element` equivalent)
- `./src/observability.rs` → tracing spans/events and telemetry seam
- `./src/lib.rs` → crate docs, module declarations, flat re-exports, feature flags
- `./tests/` → offline integration suites (loop, workspace safety, permissions, state, subagents, protocol, CLI) (`tests/CONTEXT.md`)

## Offline recipe (DeepAgent + native harness)

- Minimal agent: `DeepAgentBuilder::new(AgentCard::new(id)).config(DeepAgentConfig::default().with_language("en").with_workspace(Workspace::for_language(tmp.path(), "en"))).model(Arc::new(ScriptedModel::text("hi"))).build()` → `Result<Arc<DeepAgent>>`, then `agent.invoke(json!({"query": "hi"}), None).await` (`Agent` trait) for one turn; canonical example `./tests/deep_agent_loop.rs:112-131`.
- One-call factory: `create_deep_agent(model, card, config, tools, rails, workspace, skills, subagents, enable_task_loop, max_iterations, language, prompt_mode, restrict_to_work_dir, permissions, permission_host)` at `./src/agent/factory.rs:249`; it always sets `enable_task_loop`/`restrict_to_work_dir` from its arguments.
- Native harness: `NativeHarness::new(Arc<DeepAgent>, NativeHarnessOptions::new())` then `start(HarnessContext::new(name, agent_id, host_session_id, prompt))`, `events()` (single consumer), `send(HarnessInput::text(..), DeliveryMode::Auto)`, `stop()`; driving example `./src/protocol/tests/turns.rs:5-51`.
- Builder path registers only `PromptRail` (+ memory/subagent/caller rails); the security and skills rails come from `create_deep_agent`/`resolve_deep_agent_parts`. No default path opens a network connection; the local sysop backend and the builtin tools touch the filesystem only when a tool is invoked.

## Test Strategy

- Integration tests drive `DeepAgent` with `openjiuwen_test_support::ScriptedModel` and temp workspaces: no-tool answer, tool round, permission denial, interrupt/resume, max-iteration, follow-up queue, state round-trip.
- Workspace/path tests prove traversal rejection (`..`, absolute, drive letters) and default schema seeding.
- Permission tests cover allow/deny/ask precedence, session-scoped approvals, and file-guard path folds.
- Protocol tests drive `NativeHarness` end to end (start/send/events/stop/checkpoint) and assert capability gating.
- Tool tests assert exact tool names/JSON schemas and `success=false` error payloads.
- Definition of done: `cargo fmt -p openjiuwen-harness`, `cargo test -p openjiuwen-harness`, `cargo test -p openjiuwen-harness --no-default-features`, `cargo clippy -p openjiuwen-harness --all-targets -- -D warnings` all clean.
