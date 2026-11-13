# openjiuwen-harness (DeepAgent)

## Intent

The flagship coding-agent framework ("DeepAgent"), ported from the Python package `openjiuwen.harness` (v0.1.20).
It composes the kernel (`openjiuwen-core`: agent, session, sysop, security, memory, context) into a task-loop agent with a workspace, a permission engine, coding tools, rails, prompts, skills and subagents.
It also supplies the native implementation of the provider-neutral SPI in `openjiuwen-harness-protocol`, the language-server and git-worktree helpers, the file-based `DeepAgentSpec` loader, and the agent logic the `openjiuwen-cli` binary calls.
Fidelity target: public Python entry points, persisted JSON (session state, todo, skill library, goal, cron jobs, worktree index, evolution snapshot), tool names/JSON schemas, permission outcomes, and loop semantics — internals are idiomatic Rust.

## API Surface

Entry points (`openjiuwen_harness::agent`):
- `DeepAgent` — `new(AgentCard)`, `configure(DeepAgentConfig)`, `invoke`/`stream` (impl `openjiuwen_agent::Agent`), `follow_up`, `steer`, `abort`, `start`/`stop`/`send_input`, `load_state`/`save_state`, `switch_mode`, `add_rail`/`register_rail`.
- `DeepAgentConfig`, `DeepAgentBuilder`, `create_deep_agent(...)` (async factory), `AgentMode`, `DeepAgentState`, `resolve_deep_agent_parts`, `apply_deep_agent_parts`.

Task loop (`openjiuwen_harness::task_loop`): `TaskLoopController`, `TaskLoopEventHandler`, `LoopCoordinator`, `LoopQueues`, `DeepLoopEvent`.
Schema (`openjiuwen_harness::schema`): config/state/task/stop-condition/interaction/loop-event types.
Workspace (`openjiuwen_harness::workspace`): `Workspace`, `WorkspaceNode`, `DirectoryNode`, `DirectoryBuilder`.
Permissions (`openjiuwen_harness::permission`): `PermissionEngine`, `PermissionLevel`, `PermissionResult`, `PermissionConfirmResponse`, `PermissionsSection`, `PermissionInterruptRail`.

Tools (`openjiuwen_harness::tools`): `ToolRegistry`, `BuiltinTool`, `tool_names()`, `extended_tool_names()`.
- `BuiltinTool::DEFAULT` (twenty core names, the registry default) — `bash`, `read_file`, `write_file`, `edit_file`, `glob`, `list_files`, `grep`, `todo_create`/`todo_list`/`todo_get`/`todo_modify`, `ask_user`, `memory_search`/`memory_get`/`write_memory`/`read_memory`/`edit_memory`, `coding_memory_read`/`coding_memory_write`/`coding_memory_edit`.
- `BuiltinTool::EXTENDED` (twelve opt-in names; `ALL` is core plus extended) — `fetch_webpage`, `web_search`, `image_metadata`, `audio_metadata`, `cron`, `list_mcp_resources`, `read_mcp_resource`, `get_current_goal`, `submit_goal_report`, `tool_search`, `lsp`, `worktree`.
- Registry seams: `HttpFetcher`/`ReqwestFetcher`, `SearchBackend`/`FakeSearchBackend`, `McpResourceProvider`/`FakeMcpResourceProvider`, `ToolIndex`, `CronScheduler`, `GoalStore`, plus `with_lsp_config`, `with_worktree_manager`, `with_worktree_root`, `with_deferred_tools`/`discover`/`reveal`/`deferred_cards` (progressive discovery).
- `task_tool` and the seven `subagent_*` tools are mounted by `subagents` through `subagent_tools_with_transcript`.

Rails (`openjiuwen_harness::rails`): `HarnessRail`, `HarnessRailContext`, `HarnessRailRegistry`, `SecurityRail`, `MemoryRail`, `SkillsRail`, `SubagentRail`, `EvolutionRail`.
Evolution bodies: `EvolutionMetrics`, `EvolutionSnapshot`, `EvolutionProposal`, `IterationRecorder`, `MetricsClock`, `MetricsEvolutionRail`, `MetricsEvolutionRailConfig`, `ProposalLog`, `evolution_adapter`/`EvolutionRailAdapter`, `NoopEvolutionRail`, `EVOLUTION_SNAPSHOT_VERSION`.
Goal rail: `tools::goal::GoalRail` + `GoalRailConfig` + `GOAL_EVENT_EXTRA_KEY` (a `HarnessRail`; reachable only under `tools` because `rails` must not depend on `tools`).

Prompts (`openjiuwen_harness::prompts`): `HarnessPromptBuilder`, `PromptSection`, `SectionName`, `PromptMode`, `PromptReport`.
Subagents (`openjiuwen_harness::subagents`): `SubAgentConfig`, `SubAgentRegistry`, `SubAgentPreset`, `SubAgentRuntime`, `TaskTool`; runtime surface `SubAgentQuota`, `SpawnBudget`, `SubagentMetadata`, `SubagentActivity`, `select_lru_candidate`, `TranscriptBuffer`/`TranscriptEntry`/`TranscriptKind`/`TranscriptSummary`/`TranscriptSubscription`, `PersistedSubAgentBuckets`/`PersistedSubAgentRecord`, `save_bucket_file`/`load_bucket_file`, `subagent_tools_with_transcript`/`SubagentTranscriptTool`, `Clock`/`SystemClock`/`ManualClock`, `LoadReport`, `ResumeReport`.
Skills (`openjiuwen_harness::skills`): `Skill`, `SkillLibrary`, `SkillRootMount`, `SkillResourceValidator`.
Manifest (`openjiuwen_harness::manifest`): `ElementKind`, `HarnessElementDescriptor`, `Catalog`.
Protocol (`openjiuwen_harness::protocol`): `NativeHarness` (impl `HarnessProtocol` + `HarnessModelControl`), `NativeHarnessProvider` (impl `HarnessProvider` + `ModelConfigProvider`).
Observability (`openjiuwen_harness::observability`): span/event name helpers and the `TelemetrySink` seam.
LSP (`openjiuwen_harness::lsp`): `LspClient`, `LspTool`, `LspToolConfig`, `LspServerConfig`, `LspTransport`/`StdioTransport`/`ChannelTransport`, normalised wire types (`Position`, `Range`, `Location`, `Hover`, `CompletionItem`, `DocumentSymbol`).
Spec (`openjiuwen_harness::spec`): `DeepAgentSpec`, `SpecRegistry`, `SpecEntry`, `SpecSource`, `ResolvedSpec`, `resolve`, `SpecBuildOptions`, `PluginManifest`, `PluginRegistry`.
Worktree (`openjiuwen_harness::worktree`): `WorktreeManager`, `WorktreeTool`, `WorktreeInfo`, `WorktreeRecord`, `WorktreeEvent`, `GitRunner`/`ProcessGitRunner`, `slugify`.
CLI (`openjiuwen_harness::cli`): `Cli`, `run`, `run_cli`, `CliOptions`, `Repl`/`ReplCommand`, `CliIo` (`LineReader`/`Writer`), `SessionStore`, `InstructionFiles`, `McpConfig`, `AllowRuleStore`, `History`, `Onboarding`, `Style`, `Usage`.
Parser seam: the crate re-exports the exact `clap` version `Cli` derives from (`openjiuwen_harness::clap`), so hosts and `openjiuwen-e2e` can call `Cli::try_parse_from`/`Cli::command()` and assert grammar, `clap::error::ErrorKind` and `Error::exit_code()` without adding a clap dependency; it is a build/test seam, not SDK domain API.
Every public item is also re-exported flat from the crate root.

## Constraints

- Depends only on `openjiuwen-core` (kernel umbrella), `openjiuwen-harness-protocol`, and the kernel crates the umbrella re-exports; never on `openjiuwen-agent-teams`, `openjiuwen-rsi`, `openjiuwen-agent-evolving`, `openjiuwen-symphony`, `openjiuwen-harness-providers`.
- Upper-layer seams are traits: `SubAgentSpawner` (agent-teams/providers), `EvolutionRail` (evolution), `TelemetrySink` (extensions/OTel), `LspTransport`, `SearchBackend`, `HttpFetcher`, `McpResourceProvider`, `GitRunner`.
- Errors: `openjiuwen_common::error::{Error, Result}` everywhere; Python `build_error(code, msg)` maps to `Error::config`/`validation`/`security`/`tool`/`unsupported` with the Python message text. No panics on user input.
- Async-first on tokio; cancellation via `tokio_util::sync::CancellationToken`; locks never held across `.await`.
- Serde: snake_case, tolerant of unknown fields; persisted session JSON keeps Python keys (`deepagent`, `iteration`, `task_plan`, `pending_follow_ups`, `plan_mode`, `todos`, `skills_state`); goal/cron/worktree/evolution files keep documented keys.
- Secrets are never logged; `api_key` is `SecretString`; `Debug` of configs redacts credentials (LSP env values included).
- Logging via `tracing` only; no stdout in library code (CLI owns stdout through its injected `Writer`).
- Files ≤1000 lines; shared helpers live in the lowest owning module. Overruns come only from inline test modules and carry a top-of-file size note.
- Tests: integration tests in `./tests/`, unit tests beside modules; `ScriptedModel` from `openjiuwen-test-support`; offline, deterministic (injected clocks, injected IO, temp dirs, loopback sockets, scripted `GitRunner`/`HttpFetcher`/transport).
- Features: `native` (default), `browser`, `cua`, `mobile` (deferred preset stubs), `notify` (workspace watcher), `live-tests` (opt-in network tests); the crate must also build with `--no-default-features`.
- Dependencies stay offline-resolvable: the only HTTP client is the workspace `reqwest` (`default-features = false`, rustls); no crates are added for HTML parsing, media headers, cron, BM25, LSP framing or YAML — those are written in-crate.

## Module dependency contract

`schema` ← every module (no harness dependencies).
`workspace`, `skills`, `manifest` ← `schema`.
`permission` ← `schema`, `openjiuwen-security`, session interactions.
`prompts` ← `schema`, foundation `ToolCard` only (no dependency on `tools`).
`rails` ← `schema`, `prompts`, `skills`, `permission`, kernel memory/security; `rails` never depends on `tools`, which is why the goal rail lives at `tools::goal::GoalRail`.
`tools` ← `schema`, `workspace`, `permission`, `lsp`, `worktree`, sysop/foundation, kernel memory.
`subagents` ← `schema`, `prompts`, agent runtime.
`task_loop` ← `schema`, agent.
`agent` ← all of the above (facade).
`protocol` ← `agent`, `schema`, `openjiuwen-harness-protocol`.
`spec` ← `agent`, `schema`, `tools`, `skills`, `subagents`.
`lsp`, `worktree` ← `schema`, `permission`, sysop; no harness module above them.
`cli` ← `agent`, `protocol`, `permission`, `tools`, `skills`.
`observability` ← tracing only.

## Design decisions

- Python `DeepAgentConfig` mixes live objects (`Model`, rails) with data; the Rust `DeepAgentConfig` is serde data only, and live handles (`Arc<dyn Model>`, tools, rails, `SysOperation`) are supplied through `DeepAgentBuilder`. Documented deviation.
- Python's 13 `AgentCallbackEvent` hooks are split: the nine kernel hooks are reused through `openjiuwen_agent::AgentRail`; harness rails additionally implement task-loop hooks. `HarnessRail` owns all hooks and a `HarnessRailAdapter` bridges the kernel subset into the inner `ReActAgent` rail registry.
- The task loop is explicit controller + event handler (not an actor framework): rounds are driven by `TaskLoopController::run_round`, follow-ups/steering live in `LoopQueues`, and state is persisted after each round.
- Protocol capability gating: `NativeHarness` declares steer, graceful/force abort, pause_resume, persistent_session, checkpoint, native_tools, mcp_tools, model_selection, model_discovery; anything not declared returns `HarnessError::UnsupportedCapability`.
- Permission "no opinion" is `None`, folded with `strictest` (deny < ask < allow); absence of a matching rule yields ASK, never ALLOW.
- Tools never panic and never return `Err` for expected failures: they return a `ToolOutput`-shaped JSON payload with `success=false` and `error`.
- `BuiltinTool::DEFAULT` keeps the twenty Python core tools as the registry/builder default, so prompt size and permission behaviour stay stable; extended tools are opt-in per registry (`with_tools`), and the CLI opts into eleven of them (`CLI_EXTENDED_TOOLS`, everything but `lsp`). An extended tool that needs a backend (search, MCP resources, LSP server) reports a typed `success=false` reason instead of disappearing.
- Progressive tool discovery is a registry concern: deferred tools are carded only after `tool_search` matches and `reveal` mounts them.
- The goal rail is registered automatically by `DeepAgentBuilder::build` / `resolve_deep_agent_parts` when a goal tool is mounted, and skipped when a caller rail named `goal` exists.
- Subagent runtime policy is explicit: quota checks run depth → per-parent → cumulative → live, and LRU eviction never closes a running instance.
- `DeepAgentSpec` templates are pure data: no dylib loading, no process spawning, no template evaluation. Live handles stay on `DeepAgentBuilder`.
- The worktree helper shells out to `git` through an injectable `GitRunner`; layout is `<repo>/.worktrees/<slug>` with the registry at `<workspace>/.worktree/index.json`.
- The LSP client owns one I/O task per server, correlates ids through oneshot channels, and takes a `Box<dyn LspTransport>` so tests run an in-process fake server over `tokio::io::duplex` with no language server installed.
- Evolution work stays harness-side: the crate cannot depend on `openjiuwen-rsi`/`openjiuwen-agent-evolving` (they depend on it), so `MetricsEvolutionRail` records metrics/proposals and writes a snapshot JSON, while the LLM/RSI orchestrators implement the same trait from those crates. All evolution hooks are additive and defaulted.
- CLI behaviour is testable by construction: IO, environment, home directory, shell runner and model are injected, so every command, prompt and wizard path is exercised without a terminal or network.

## Known Issues

- Goal events queued under `harness.goal_events` have no consumer: `agent`/`task_loop` build and drop rail contexts, so a host must drain the key to observe `goal.updated` frames. A drain seam would be new API.
- The default `PermissionsSection` leaves every tool at ASK, and a one-shot `openjiuwen run` handle is non-interactive, so tool calls are refused with `HOST_PERMISSION_FAILED` unless `permissions.yaml` (or the caller) allows them.
- `lsp` has no CLI mounting: the CLI does not resolve a per-workspace server config yet.
- `web_search`, `list_mcp_resources` and `read_mcp_resource` need a backend; with none wired they answer with typed failure payloads.

## Deferred Python areas

- `subagents/{browser,cua,mobile_gui}_agent.py`, `tools/{browser_move,cua,mobile_gui,cua}` — feature-gated stubs (`browser`, `cua`, `mobile`), return `Error::Unsupported`.
- `heartbeat_rail`, `personal_context`, `moa/`, `kv_cache/`, `persistent_agent/` — deferred.
- `observability/` OTel exporter — seam only (`TelemetrySink`); exporter belongs to `openjiuwen-extensions`.
- Model-backed multimodal tools beyond header metadata (vision question answering, captioning, transcription), paid/vendor search backends, JS rendering, and the LLM quality/assessment pass over fetched pages.
- LSP diagnostics (push registry and pull requests), call hierarchy/implementation, per-extension server discovery, crash retry/backoff; cron update/toggle/preview/run actions and the 7-field Quartz grammar; MCP resource subscriptions and the live-client adapter (the foundation client exposes tools only).
- Worktree cross-process locking, sparse checkout, enter/exit-worktree agent tools, worktree rails and change summaries.
- Subagent activity streams (throttled queue), background eviction scheduler, lifecycle resource hooks; Python's dynamic plugin execution model (`import`, dylib, template evaluation).
- `agent_evolving` / RSI orchestrators and LLM-driven evolution bodies live in the layer-6 crates; only the harness-side rail and metrics bodies are here.

## Routing Table

- `./src/schema/` → config, state, task/todo, stop conditions, interaction, loop events, agent mode
- `./src/workspace/` → `Workspace`, named directory nodes, default schema, `DirectoryBuilder`
- `./src/permission/` → permission model, engine, file/shell/net guards, session decisions, HITL interrupt rail
- `./src/tools/` → model-facing tools (`registry.rs`, core and extended sets), web fetch/search, media metadata, cron, MCP resources, goal, progressive discovery, descriptions, shared helpers
- `./src/rails/` → rail trait/registry, security/memory/skills/subagent rails, evolution metrics/proposals/snapshot rail
- `./src/prompts/` → system prompt builder, sections, tool descriptions, prompt report, sanitization
- `./src/subagents/` → `SubAgentConfig`, registry, presets, quota/LRU/transcript/persistence runtime, task and subagent tools
- `./src/skills/` → skill discovery/library state, mounting, `SKILL.md` validation
- `./src/task_loop/` → loop coordinator, queues, event manager, task-loop controller/handler
- `./src/agent/` → `DeepAgent`, builder, factory, goal-rail assembly, subagent wiring, session state persistence, round execution
- `./src/protocol/` → native harness protocol/provider implementation, event bridge, cursors
- `./src/lsp/` → JSON-RPC framing/transport, LSP client, rendering, `lsp` tool
- `./src/spec/` → `DeepAgentSpec` templates, registry/precedence, plugins, embedded `resources/`, resolution into builder parts
- `./src/worktree/` → git runner, worktree manager/layout/registry, events, `worktree` tool
- `./src/cli/` → clap `run`/`chat` grammar, option resolution, one-shot runner, REPL and slash commands, history, styling, instruction files, MCP config, allow-rule write-back, onboarding
- `./src/manifest.rs` → harness element catalog (`@harness_element` equivalent)
- `./src/observability.rs` → tracing spans/events and telemetry seam
- `./src/lib.rs` → crate docs, module declarations, flat re-exports, feature flags
- `./tests/` → offline integration suites (loop, workspace safety, permissions, state, subagents/transcripts, protocol, CLI) (`tests/CONTEXT.md`)

## Offline recipe (DeepAgent + native harness)

- Minimal agent: `DeepAgentBuilder::new(AgentCard::new(id)).config(DeepAgentConfig::default().with_language("en").with_workspace(Workspace::for_language(tmp.path(), "en"))).model(Arc::new(ScriptedModel::text("hi"))).build()` → `Result<Arc<DeepAgent>>`, then `agent.invoke(json!({"query": "hi"}), None).await` (`Agent` trait) for one turn; canonical example `./tests/deep_agent_loop.rs`.
- One-call factory: `create_deep_agent(...)` at `./src/agent/factory.rs`; it always sets `enable_task_loop`/`restrict_to_work_dir` from its arguments and registers `security`, `skills`, `subagent` and (when a goal tool is mounted) `goal`.
- Extended tools: `ToolRegistry::new(workspace).with_tools(BuiltinTool::EXTENDED)` for everything, or a chosen subset; `with_lsp_config(..)`/`with_worktree_root(..)` supply the two configured tools, and absent backends yield `success=false` payloads.
- Spec-driven agent: `SpecRegistry::discover(workspace, &[home])?.resolve("name")?` then `ResolvedSpec::build_agent(model, ...)`; templates live in `<workspace>/.openjiuwen/agents/*.{yaml,yml,json,md}`, plugins in `<workspace>/.openjiuwen/plugins/<name>/plugin.yaml`.
- Native harness: `NativeHarness::new(Arc<DeepAgent>, NativeHarnessOptions::new())` then `start(HarnessContext::new(name, agent_id, host_session_id, prompt))`, `events()` (single consumer), `send(HarnessInput::text(..), DeliveryMode::Auto)`, `stop()`.
- LSP offline: `LspTool::new(LspToolConfig::default()).with_transport(Box::new(ChannelTransport::new(buffer)))` plus an in-process fake server task; no language server binary required.
- Worktree offline: `WorktreeManager::new(repo_root).with_runner(Arc::new(scripted_runner))` or a `tempfile` git repo when `git` is installed.
- No default path opens a network connection; web fetch/search, MCP resources and the LSP server are only reached when a tool is invoked with a configured backend.

## Test Strategy

- Integration tests drive `DeepAgent` with `openjiuwen_test_support::ScriptedModel` and temp workspaces: no-tool answer, tool round, permission denial, interrupt/resume, max-iteration, follow-up queue, state round-trip.
- Workspace/path tests prove traversal rejection (`..`, absolute, drive letters) and default schema seeding.
- Permission tests cover allow/deny/ask precedence, session-scoped approvals, file/net guard folds.
- Protocol tests drive `NativeHarness` end to end (start/send/events/stop/checkpoint) and assert capability gating.
- Tool tests assert exact names/JSON schemas/description tables and `success=false` error payloads; web/media/cron/goal/spec/worktree suites use injected fetchers, fixture bytes, scripted git output and frozen clocks.
- Subagent tests cover quota refusals and frees, per-parent/depth limits, LRU eviction with transcript digests, and persistence round-trips.
- CLI tests inject reader/writer/env/home/shell/model and assert slash commands, history round-trip, colour opt-out byte output, instruction precedence, MCP config errors, allow-rule YAML write-back, `!cmd`, skill slash and the onboarding paths.
- LSP tests cover framing edge cases, transport lifecycles, timeout/cancel, and a fake in-process server.
- Definition of done: `cargo fmt -p openjiuwen-harness`, `cargo test -p openjiuwen-harness`, `cargo test -p openjiuwen-harness --no-default-features`, `cargo clippy -p openjiuwen-harness --all-targets -- -D warnings`, `cargo doc -p openjiuwen-harness --no-deps` all clean and offline.
