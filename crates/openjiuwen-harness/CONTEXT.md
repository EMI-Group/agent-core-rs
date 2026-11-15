# openjiuwen-harness (DeepAgent)

## Intent

The flagship coding-agent framework ("DeepAgent"), ported from the Python package `openjiuwen.harness` (v0.1.20).
It composes the kernel (`openjiuwen-core`: agent, session, sysop, security, memory, context) into a task-loop agent with a workspace, a permission engine, coding tools, rails, prompts, skills and subagents.
It also supplies the native implementation of the provider-neutral SPI in `openjiuwen-harness-protocol`, the language-server and git-worktree helpers, the file-based `DeepAgentSpec` loader, and the agent logic the `openjiuwen-cli` binary calls.
Fidelity target: public Python entry points, persisted JSON (session state, todo, skill library, goal, cron jobs, worktree index, evolution snapshot), tool names/JSON schemas, permission outcomes, and loop semantics — internals are idiomatic Rust.

## API Surface

Entry points (`openjiuwen_harness::agent`):
- `DeepAgent` — `new(AgentCard)`, `configure(DeepAgentConfig) -> Result<()>` (validates first and installs nothing on rejection), `invoke`/`stream` (impl `openjiuwen_agent::Agent`), `follow_up`, `steer`, `abort`, `start`/`stop`/`send_input`, `load_state`/`save_state`, `switch_mode`, `tool_cards`, `add_rail`/`register_rail`.
- `DeepAgentConfig`, `DeepAgentBuilder`, `create_deep_agent(...)` (async factory), `AgentMode`, `DeepAgentState`, `resolve_deep_agent_parts`, `apply_deep_agent_parts`.
- Rail-event drain: `drain_rail_events`, `RailEventBus`, `RailEventDrain`, `RAIL_EVENTS_EXTRA_PREFIX`, `DEFAULT_RAIL_EVENT_CAPACITY`; `DeepAgent::drain_rail_events()` collects retained events, and the invocation stream carries rail-queued frames as interaction chunks.

Task loop (`openjiuwen_harness::task_loop`): `TaskLoopController`, `TaskLoopEventHandler`, `LoopCoordinator`, `LoopQueues`, `DeepLoopEvent`.
Schema (`openjiuwen_harness::schema`): config/state/task/stop-condition/interaction/loop-event types.
Workspace (`openjiuwen_harness::workspace`): `Workspace`, `WorkspaceNode`, `DirectoryNode`, `DirectoryBuilder`.
Permissions (`openjiuwen_harness::permission`): `PermissionEngine`, `PermissionLevel`, `PermissionResult`, `PermissionConfirmResponse`, `PermissionsSection`, `PermissionInterruptRail`.

Tools (`openjiuwen_harness::tools`): `ToolRegistry`, `BuiltinTool`, `tool_names()`, `extended_tool_names()`.
- `BuiltinTool::DEFAULT` (twenty core names, the registry default) — `bash`, `read_file`, `write_file`, `edit_file`, `glob`, `list_files`, `grep`, `todo_create`/`todo_list`/`todo_get`/`todo_modify`, `ask_user`, `memory_search`/`memory_get`/`write_memory`/`read_memory`/`edit_memory`, `coding_memory_read`/`coding_memory_write`/`coding_memory_edit`.
- `BuiltinTool::EXTENDED` (fifteen opt-in names; `ALL` is the thirty-five-tool union) — `fetch_webpage`, `web_search`, `image_metadata`, `audio_metadata`, `cron`, `list_mcp_resources`, `read_mcp_resource`, `get_current_goal`, `submit_goal_report`, `tool_search`, `lsp`, `worktree`, `mcp_resource_subscription`, `enter_worktree`, `exit_worktree`.
- Registry seams: `HttpFetcher`/`ReqwestFetcher`, `SearchBackend`/`FakeSearchBackend` (`HttpSearchBackend` with `OPENJIUWEN_WEB_SEARCH_ENDPOINT`), `McpResourceProvider`/`FakeMcpResourceProvider` and `McpClientResourceProvider` (live foundation client), `McpResourceSubscriptionProvider`/`McpResourceSubscriptions`, `ToolIndex`, `CronScheduler`, `GoalStore`, `WorktreeSessionStore`, plus `with_lsp_config`, `with_worktree_manager`, `with_worktree_root`, `with_worktree_session_store`, `with_mcp_subscriptions`, `with_project_root`, `with_deferred_tools`/`discover`/`reveal`/`deferred_cards` (progressive discovery).
- MCP resource notifications: `McpResourceSubscriptions::ingest(method, &Value)` / `ingest_notification(&McpNotification)` / `ingest_notifications(&[McpNotification])` publish `notifications/resources/updated` (and the bare `resources/updated` spelling) into the subscribed URI and report a `NotificationOutcome` (`Published`, `ListChanged`, `IgnoredMethod`, `NotSubscribed`, `NoUri`) without ever creating a subscription; `McpClientResourceProvider::drain_notifications(&McpResourceSubscriptions)` feeds them from `McpClient::take_notifications`; `RESOURCE_UPDATED_METHOD(_SHORT)`, `RESOURCE_LIST_CHANGED_METHOD(_SHORT)`, `LIVE_CLIENT_SUBSCRIBE_UNSUPPORTED` and `LIVE_CLIENT_UNSUBSCRIBE_UNSUPPORTED` name the methods and the live adapter's refusals.
- Memory backend: `WorkspaceMemoryBackend` (`new(Workspace)`, `workspace()`) is the production `MemoryToolBackend` — file-backed, Python scope/path mapping with line windows and coding-memory front matter, and a deterministic offline keyword search (`provider: "local"`, `search_mode: "keyword_only"`, `embedding_available: false`); `ToolRegistry::with_workspace_memory` installs it, `with_memory` substitutes a host backend, and `DeepAgentBuilder::build` mounts it by default.
- Cron: `CronSchedule` accepts 5-field classic and 6/7-field Quartz (`?`, `L`, `nL`, `d#n`, month/day names, wrap-around ranges) with `CronTool` actions `create`/`list`/`cancel`/`get_job`/`update`/`toggle`/`preview`/`run`/`runs`/`status`, `CronUpdate`, `CronStatus`, `CronRun`/`CronRunTrigger` and a bounded run history.
- `task_tool` and the seven `subagent_*` tools are mounted by `subagents` through `subagent_tools_with_transcript`.

Rails (`openjiuwen_harness::rails`): `HarnessRail`, `HarnessRailContext`, `HarnessRailRegistry`, `SecurityRail`, `MemoryRail`, `SkillsRail`, `SubagentRail`, `EvolutionRail`.
Additional rails: `HeartbeatRail`, `KvCacheLifecycleRail` (+ `KvCacheSubagentLifecycle`, `KvCacheProviderRegistry`), `PersonalContextRail` (+ `PersonalContextStore`, `SourceMetadataStore`, `SourceLinkBook`, `LocalFilesProvider`, `SourceProviderRegistry`).
Evolution bodies: `EvolutionMetrics`, `EvolutionSnapshot`, `EvolutionProposal`, `IterationRecorder`, `MetricsClock`, `MetricsEvolutionRail`, `MetricsEvolutionRailConfig`, `ProposalLog`, `evolution_adapter`/`EvolutionRailAdapter`, `NoopEvolutionRail`, `EVOLUTION_SNAPSHOT_VERSION`.
Goal rail: `tools::goal::GoalRail` + `GoalRailConfig` + `GOAL_EVENT_EXTRA_KEY` (a `HarnessRail`; reachable only under `tools` because `rails` must not depend on `tools`).

Prompts (`openjiuwen_harness::prompts`): `HarnessPromptBuilder`, `PromptSection`, `SectionName`, `PromptMode`, `PromptReport`.
Subagents (`openjiuwen_harness::subagents`): `SubAgentConfig`, `SubAgentRegistry`, `SubAgentPreset`, `SubAgentRuntime`, `TaskTool`, `RuntimeSubAgentSpawner` (`new`/`with_limit`/`limit`/`close`, the production impl of the rail's `SubAgentSpawner`, reachable from a built agent through `DeepAgent::subagent_spawner()`); runtime surface `SubAgentQuota`, `SpawnBudget`, `SubagentMetadata`, `SubagentActivity`, `select_lru_candidate`, `TranscriptBuffer`/`TranscriptEntry`/`TranscriptKind`/`TranscriptSummary`/`TranscriptSubscription`, activity streams (`ActivityStream`, `ActivitySubscription`, `SubagentActivityFrame`, `ActivityThrottle`), the eviction scheduler (`EvictionSchedulerConfig`, `EvictionSchedulerHandle`, `SweepReport`) and lifecycle hooks (`SubagentLifecycleHook`, `LifecycleHooks`, `CallbackLifecycleHook`), `PersistedSubAgentBuckets`/`PersistedSubAgentRecord`, `save_bucket_file`/`load_bucket_file`, `subagent_tools_with_transcript`/`SubagentTranscriptTool`, `Clock`/`SystemClock`/`ManualClock`, `LoadReport`, `ResumeReport`.
MoA (`openjiuwen_harness::moa`): `MoaConfig`/`MoaMode`, `consult`/`consult_problem`, `ProposalStore`/`ProposalBatch`, `rank_proposals`/`rank_with_config` (`StructuralVerifier`, `LlmVerifier`, `ProgressVerifier`), `Aggregator`/`AggregationStrategy`, `MoaRail`, `ConsultationHistory`, `ConsultationSkillMount`.
Persistent agent (`openjiuwen_harness::persistent_agent`): `ExecutionLog`/`ExecutionEntry`/`LogDigest`/`Replay`, `PersistentSessionRecord`/`SessionStateStore`/`FileSessionStore`/`SessionSummary`, `WindowAssembler`/`WindowManager`/`Capacity`/`WindowPolicy` (`assemble`/`assemble_with_budget` stay infallible, `try_assemble`/`try_assemble_with_budget` return the summariser error), `WindowSummarizer` with `ExtractiveSummarizer`, `UnsupportedModelSummarizer` and `CompletionSummarizer` over the `CompletionFn = dyn Fn(&str) -> Result<String> + Send + Sync` seam, `WindowAssembly::summary_error`, `Continuation`/`resume`/`ResumePlan`, `hand_off`/`LocalHandoff`/`HandoffTransport`, `Control`, `Emitter`/`EventSink`, `ShareFile`, `ExecutionRecordRail`/`ModelCallRail`.
Recording: `recording::{ExecutionRecorder, ActiveExecution, PersistentRunConfig}` (`begin`/`finish`/`save_session` around a live turn), driven by `DeepAgentConfig::persistent_agent`/`with_persistent_agent`.
Host discovery: `ExecutionCatalog`/`ExecutionSummary` (enumerate a log root), `FileSessionStore::list`/`SessionSummary`, `PersistentAgentLayout` (`logs_root`/`sessions_root`/`links_root`), `percent_decode_identifier`, and `root()` on `RootLogSource`/`FileLinkStore`/`FileSessionStore`; `ResumePlan` is `Serialize`/`Deserialize` with `seed_messages()`/`seed_history()` for the rebuilt window.
Watcher (`openjiuwen_harness::watcher`, feature `notify`): `WorkspaceWatcher`/`WatcherConfig`/`WatcherHandle`, `WatcherConfig: FromStr + Display` (`debounce=…,event_capacity=…,intake_capacity=…,dispatch_capacity=…`, duration-only shorthand, round-trippable), `WatchEvent`/`WatchUpdate`/`WatchTargetKind`/`ChangeKind`, `ReloadHub`/`ReloadHandler`, `SkillLibraryReloader`/`InstructionReloader`/`McpConfigReloader`; without the feature the API still compiles and `start()` returns `Error::Unsupported`.
Skills (`openjiuwen_harness::skills`): `Skill`, `SkillLibrary`, `SkillRootMount`, `SkillResourceValidator`.
Manifest (`openjiuwen_harness::manifest`): `ElementKind`, `HarnessElementDescriptor`, `Catalog`.
Protocol (`openjiuwen_harness::protocol`): `NativeHarness` (impl `HarnessProtocol` + `HarnessModelControl`), `NativeHarnessProvider` (impl `HarnessProvider` + `ModelConfigProvider`).
Observability (`openjiuwen_harness::observability`): span/event name helpers and the `TelemetrySink` seam.
LSP (`openjiuwen_harness::lsp`): `LspClient` (implementation, call hierarchy, document diagnostic, notification handlers), `LspTool`, `LspToolConfig` (`new` + `FromStr`/`Display` over a `key=value` spec), `LspAction`/`LspArgs` (`FromStr`/`Display`, `LspArgs::new(action)` plus `with_path`/`with_position`/`with_query`/`with_context_lines`), `LspServerConfig`, `LspTransport`/`StdioTransport`/`ChannelTransport`, normalised wire types (`Position`, `Range`, `Location`, `Hover`, `CompletionItem`, `DocumentSymbol`, `Diagnostic`, `CallHierarchyItem`), `DiagnosticRegistry`, `DiscoveryIndex`/`ResolvedServer`/`ServerKey`, `ServerSupervisor`/`RestartPolicy`/`BackoffTimer`.
Spec (`openjiuwen_harness::spec`): `DeepAgentSpec`, `SpecRegistry`, `SpecEntry`, `SpecSource`, `ResolvedSpec`, `resolve`, `SpecBuildOptions`, `PluginManifest`, `PluginRegistry`.
Worktree (`openjiuwen_harness::worktree`): `WorktreeManager`, `WorktreeTool`, `EnterWorktreeTool`/`ExitWorktreeTool`, `WorktreeInfo`, `WorktreeRecord`, `WorktreeEvent`, `GitRunner`/`ProcessGitRunner`, `slugify`, `WorktreeLock`/`WorktreeLockConfig`, `SparseConfig`/`SparseMode`, `WorktreeChangeSummary`/`change_summary`, `CleanupPolicy`/`CleanupReport`, `WorktreeRail`, `WorktreeSessionStore`/`MemoryWorktreeSessionStore`.
CLI (`openjiuwen_harness::cli`): `Cli`, `run`, `run_cli`, `CliOptions`, `Repl`/`ReplCommand`, `CliIo` (`LineReader`/`Writer`), `SessionStore`, `InstructionFiles`, `McpConfiguration`, `AllowRules`, `History`, `OnboardingWizard`/`maybe_run`, `Style`, `UsageTotals`.
CLI resume: `ResumedTurn` (`new`/`with_history`/`with_state`), `run_once_resumed`, `run_with_model_resumed` (both callable from the crate root), and `Repl::seed_history` / `Repl::with_session_id`, which adopt an existing `<home>/sessions/<id>.json` (seeding its transcript, keeping its id) instead of truncating a fresh one.
Seeding (`openjiuwen_harness::agent`): `DeepAgent::seed_history(&self, &Arc<dyn Session>, Vec<BaseMessage>)` fills the kernel `"main"` context (create-or-set, before the first `invoke`/`stream`), and `seed_state(&dyn Session, Value)` replaces the persisted `deepagent` state.
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
- Features: `native` (default, empty marker — no source reads it), `browser`/`cua`/`mobile` (gate only the deferred `SubAgentPreset` stub variants), `notify` (the only feature that gates real code: `watcher::backend::start`), `live-tests` (empty marker, no source reads it); the crate must also build with `--no-default-features`.
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
`agent` ← all of the above (facade); it also drives `persistent_agent::recording` when `DeepAgentConfig::persistent_agent` is on.
`protocol` ← `agent`, `schema`, `openjiuwen-harness-protocol`.
`spec` ← `agent`, `schema`, `tools`, `skills`, `subagents`.
`lsp`, `worktree` ← `schema`, `permission`, sysop; no harness module above them.
`moa` ← `schema`, `rails` (`HarnessRail`), `prompts`, `skills`, `subagents` (clock), foundation `Model`.
`persistent_agent` ← `rails`, `subagents` (clock), `schema`, sysop; no `agent`/`task_loop` dependency.
`watcher` ← `skills`, `cli` reloaders, `workspace`; nothing above it depends on it.
`cli` ← `agent`, `protocol`, `permission`, `tools`, `skills`.
`observability` ← tracing only.

## Design decisions

- Python `DeepAgentConfig` mixes live objects (`Model`, rails) with data; the Rust `DeepAgentConfig` is serde data only, and live handles (`Arc<dyn Model>`, tools, rails, `SysOperation`) are supplied through `DeepAgentBuilder`. Documented deviation.
- Python's 13 `AgentCallbackEvent` hooks are split: the nine kernel hooks are reused through `openjiuwen_agent::AgentRail`; harness rails additionally implement task-loop hooks. `HarnessRail` owns all hooks and a `HarnessRailAdapter` bridges the kernel subset into the inner `ReActAgent` rail registry.
- The task loop is explicit controller + event handler (not an actor framework): rounds are driven by `TaskLoopController::run_round`, follow-ups/steering live in `LoopQueues`, and state is persisted after each round.
- Protocol capability gating: `NativeHarness` declares steer, graceful/force abort, pause_resume, persistent_session, checkpoint, native_tools, mcp_tools, model_selection, model_discovery; anything not declared returns `HarnessError::UnsupportedCapability`.
- Permission "no opinion" is `None`, folded with `strictest` (deny < ask < allow); absence of a matching rule yields ASK, never ALLOW.
- Tools never panic and never return `Err` for expected failures: they return a `ToolOutput`-shaped JSON payload with `success=false` and `error`.
- `BuiltinTool::DEFAULT` keeps the twenty Python core tools as the registry/builder default, so prompt size and permission behaviour stay stable; extended tools are opt-in per registry (`with_tools`), and the CLI opts into thirteen of them (`CLI_EXTENDED_TOOLS`, everything but `lsp` and `mcp_resource_subscription`). An extended tool that needs a backend (search, MCP resources, LSP server, worktree manager) reports a typed `success=false` reason instead of disappearing.
- Progressive tool discovery is a registry concern: deferred tools are carded only after `tool_search` matches and `reveal` mounts them.
- The goal rail is registered automatically by `DeepAgentBuilder::build` / `resolve_deep_agent_parts` when a goal tool is mounted, and skipped when a caller rail named `goal` exists.
- Subagent runtime policy is explicit: quota checks run depth → per-parent → cumulative → live, and LRU eviction never closes a running instance.
- `DeepAgentSpec` templates are pure data: no dylib loading, no process spawning, no template evaluation. Live handles stay on `DeepAgentBuilder`.
- The worktree helper shells out to `git` through an injectable `GitRunner`; layout is `<repo>/.worktrees/<slug>` with the registry at `<workspace>/.worktree/index.json`.
- The LSP client owns one I/O task per server, correlates ids through oneshot channels, and takes a `Box<dyn LspTransport>` so tests run an in-process fake server over `tokio::io::duplex` with no language server installed.
- Evolution work stays harness-side: the crate cannot depend on `openjiuwen-rsi`/`openjiuwen-agent-evolving` (they depend on it), so `MetricsEvolutionRail` records metrics/proposals and writes a snapshot JSON, while the LLM/RSI orchestrators implement the same trait from those crates. All evolution hooks are additive and defaulted.
- CLI behaviour is testable by construction: IO, environment, home directory, shell runner and model are injected, so every command, prompt and wizard path is exercised without a terminal or network.
- Rail-queued events drain through `RailEventDrain`, a `HarnessRail` at `i32::MIN` priority registered last on both the harness and kernel registries; it moves `harness.goal_events` and any `harness.events.*` payload out of `ctx.extra` into the invocation stream, and non-streaming hosts collect the retained copy with `DeepAgent::drain_rail_events()`.
- The watcher is a typed event pipeline, not a callback tangle: OS notifications feed a per-path trailing debounce, the resulting `WatchEvent`s publish through bounded subscribers, and `ReloadHub` dispatches named reload handlers (skills, instruction files, MCP config) with reload failures reported as `WatchUpdate::ReloadFailed` instead of aborting the watcher.
- Nothing in the crate starts a watcher: `DeepAgent`, the builder and the CLI hold no watcher handle, so a host (or a test) drives `ReloadHub::dispatch(&WatchEvent)` itself; `ReloadHub`, the reloaders and the plan types compile without the `notify` feature, and only `WorkspaceWatcher::start` reports the missing backend.
- MoA rails never call a model: `MoaRail` publishes the consultation prompt section and state, while `consult`/`verify_progress` are explicit calls the host drives. `DeepAgentConfig` has no `moa` field, so a host either registers a `MoaRail` through `DeepAgentBuilder::rail`/`DeepAgent::add_rail` or drives the module standalone.
- The persistent-agent journal is JSONL plus atomic metadata, so a crashed run is resumable by reopening the directory; window assembly is deterministic and reports exactly what it elided.
- Resuming a persisted run seeds the kernel `"main"` context (`DeepAgent::seed_history` → `get_context`/`set_messages`, else `create_context_with(ContextOptions::history)`), not a rail: the harness rail hook fires after the system prompt is assembled, runs once per ReAct iteration and its mutations are never persisted, while the context survives every later turn of the same session. Seeding must follow the last `configure` (which replaces the engine and drops pooled contexts) and the pool key is `(session.id(), "main")`.
- The CLI stays free of a `persistent_agent` edge: a host reads a `ResumePlan`, passes plain `Vec<BaseMessage>`/JSON through `ResumedTurn`, and `cli`/`agent` only ever see kernel types; `run_with_model` keeps its frozen signature and shares the extracted `stream_run` tail with `run_with_model_resumed`.
- Worktree mutations are serialized by a `flock`-backed `WorktreeLock` (bounded wait, pid/timestamp sidecar, stale recovery); `enter_worktree`/`exit_worktree` report the target and original directory instead of mutating the process cwd.
- Subagent activity is published as throttled, bounded, sequenced frames with explicit truncation/lag accounting; background eviction is opt-in, skips running instances and fires lifecycle hooks after the state lock is dropped.
- The workspace memory backend is file-backed: the eight memory tools map Python's scope/path arguments onto the workspace nodes (line windows included) and search is a deterministic offline keyword scan (`provider: "local"`, `search_mode: "keyword_only"`, `embedding_available: false`) rather than Python's hybrid index; `DeepAgentBuilder::build` mounts it so `memory_*`/`coding_memory_*` work without host wiring, and `with_memory` still replaces it.
- The subagent rail's default spawner is `RuntimeSubAgentSpawner`: a bounded `tokio::sync::Semaphore` reserves capacity at `spawn` (an over-limit call fails with `Error::Validation` instead of queueing), a background watcher releases the permit at a terminal status, and `close` cancels the instance's own `CancellationToken` before delegating to the runtime.
- With `DeepAgentConfig::persistent_agent` on, `DeepAgent` journals each turn: `begin_recording` opens the execution and writes the query, `finish_recording` appends the answer plus a `PersistentSessionRecord`, `ExecutionLog::append_entry` fsyncs every entry, and a recording failure is warn-only so journaling never costs a turn its answer.
- `DeepAgent::configure` validates before it mutates: an invalid `DeepAgentConfig` is rejected with the Python `__post_init__` message and the previously installed config, prompt, kernel settings, permission policy and tool set stay in place.
- `enable_task_planning` (default `false`, the Python default) gates the four `todo_*` tools and the TODO prompt section at build time; `ToolRegistry`'s `BuiltinTool::DEFAULT` set is unchanged and the CLI sets the flag to `true`.
- `enable_async_subagent`/`enable_subagent_runtime` select the sub-agent tool set through `agent::subagent_factory::SubagentMode` in the order `Runtime` > `AsyncSession` > `SyncTask`; `task_tool` is always mounted, `AsyncSession` adds `subagent_spawn`/`wait`/`list`/`close`, and `Runtime` adds all seven `subagent_*` tools.
- `enable_tool_resilience_rail` (default `true`) drives the kernel `openjiuwen_agent::policy::ToolResilience` rather than a rail object: 3 attempts with a 100 ms initial backoff for idempotent tools, a single attempt for a non-idempotent card, applied through `inner.set_tool_resilience(..)` in the builder, `configure` and `apply_extra_tools`, because the kernel never fires `RailEvent::OnToolException`.
- `project_root` is an access boundary: `ToolEnv::with_access_root` rejects an absolute path outside it (`Access denied: '<path>' is outside the project root.`) and `PermissionEngineConfig::with_project_root` denies outside absolute path arguments (matched-rule suffix `|project_root`), while relative paths stay workspace-sandboxed and `None` keeps the cwd/workspace behaviour.
- `invocation_log_dir` writes one bounded JSONL file per `(agent, session)` per turn from `Driver::run_resolved` (agent/session ids, role, query, answer, status, tool-call and iteration counts, start/finish times, `truncated`), bounded by 1024-char text, 4096-char entries, 128 KiB per file with oldest-entry trimming and 256-char identifiers; IO failures warn and never fail a turn, and `None` logs nothing.
- Diagnostics report a missing capability instead of an empty list: a server that does not advertise `textDocument/diagnostic` or answers `-32601` yields a typed `success=false` failure naming `textDocument/diagnostic`, an unresolved server yields informational `success=true` with `source: "none"`, and only a full report produces `source: "pull"` with an empty list for a clean file.

## Known Issues

- The default `PermissionsSection` leaves every tool at ASK, and a one-shot `openjiuwen run` handle is non-interactive, so an `ask` is refused without printing and the tool is skipped with `[PERMISSION_DENIED] <reason> (Hosted permission request failed)` unless `permissions.yaml` (or the caller) allows it; there is no `HOST_PERMISSION_FAILED` identifier in the source.
- `lsp` has no CLI mounting: the CLI does not resolve a per-workspace server config yet, so the CLI excludes it from `CLI_EXTENDED_TOOLS`.
- `mcp_resource_subscription` has no CLI mounting: the CLI wires no `McpResourceSubscriptions` registry.
- `web_search` answers with a typed not-configured reason until an endpoint is configured (`OPENJIUWEN_WEB_SEARCH_ENDPOINT` or `ToolRegistry::with_search_backend`); `list_mcp_resources`/`read_mcp_resource` need an `McpResourceProvider`, and resource subscriptions additionally need a provider that supports `resources/subscribe`.
- LSP server discovery uses the builtin per-extension table plus host-supplied configs; servers that are not in `PATH` or not covered by a config are reported as typed launch failures, and `.gitignore` filtering is not applied to workspace scans.
- A resumed window is lossy by construction: entry text is truncated to `SUMMARY_LIMIT`/`DETAIL_LIMIT`, lifecycle markers and elided groups are dropped, and a record resumed from a paused execution only restarts cleanly when the host also seeds the interruption keys alongside the history.
- `run_once_resumed` is signature-pinned but has no behavioural test: it resolves a real provider through `CliOptions::build_model`, so every offline suite drives `run_with_model_resumed`, which differs only by that call.
- `CronSchedule` rejects `W` with a substring test, so the weekday name `WED` is rejected in both grammars and inside lists/ranges.
- The `lsp` tool is read-only and never sends `textDocument/formatting`, `rename`, `codeAction` or `workspace/applyEdit`, and has no `Error::Unsupported` seam at all; a missing server is a typed `success=false` launch failure, a diagnostics pull against a server that does not advertise `textDocument/diagnostic` (or answers `-32601`) is a typed capability failure, and every other `-32601` maps to `Operation not supported by this language server: <method>`.
- `LspArgs::from_str` splits on whitespace, so a path containing spaces must go through `with_path`; a path whose last `:` segment is numeric is read as a position.

- `permission::rules::OS_CONTROL_TOOLS` names twelve cua tools that `BuiltinTool` never mounts, so that policy list is inert.
- MCP resource subscriptions are wired on the push side only: the foundation `McpClient` exposes `list_resources`/`read_resource` and `take_notifications` but no `resources/subscribe` request, so the live adapter reports `supports_subscriptions() == false` and its `subscribe`/`unsubscribe` answer `LIVE_CLIENT_SUBSCRIBE_UNSUPPORTED`/`LIVE_CLIENT_UNSUBSCRIBE_UNSUPPORTED`; rmcp 0.8.5 itself carries `resources/subscribe`, so the gap is in the foundation trait, not the transport.
- Two concurrent subscribes of the same new URI can leave the provider holding a second sink (a `debug` log records it), and `notifications/resources/list_changed` neither re-lists nor reaps: it is a counted no-op.
- The crate has no `#[ignore]`d test and no `todo!`/`unimplemented!`; the single panic path is the internal `memory_kind` `unreachable!` (`src/tools/registry.rs:45`).

## Deferred Python areas

- `subagents/{browser,cua,mobile_gui}_agent.py`, `tools/{browser_move,cua,mobile_gui,cua}` — feature-gated stubs (`browser`, `cua`, `mobile`), return `Error::Unsupported`.
- `observability/` OTel exporter — seam only (`TelemetrySink`); exporter belongs to `openjiuwen-extensions`.
- Model-backed multimodal tools beyond header metadata (vision question answering, captioning, transcription), paid/vendor search backends and their keys, JS/reader-proxy rendering, and the LLM quality/assessment pass over fetched pages.
- LSP: `.gitignore` filtering, `PATH`/manifest server auto-discovery beyond the builtin table, diagnostics subscriptions and workspace-wide pulls, formatter/rename edits (the tool is read-only), workspace-symbol caching.
- Cron: `wake`, the `schedule.at`/`schedule.every` job forms, `payload`/`delivery` routing, legacy per-action tools, `W` (nearest weekday), run output capture and any background daemon (the host polls `due(now)`).
- MCP: the `resources/subscribe` request and resource pagination (server-pushed updates already reach `McpResourceSubscriptions::ingest`), and live-transport MCP tools beyond the foundation client.
- Worktree: Python lifecycle policies (`ephemeral`/`durable`/`auto`) with shutdown auto-cleanup timers, symlink directories/include-pattern copying/hooks-path/`core.longpaths`, the dead `WorktreeLifecycleRail`/`AutoSetupRail` chain, backend/vcs hooks, and process-global cwd mutation (the tools report the paths instead).
- Subagents: the stream-chunk milestone projector and parent-session activity-queue drain (typed transition frames stand in), `enable_transcript_stream`, persisted activity frames, cross-process sweepers.
- MoA: `jev_routed` needs an external System One client (`Error::Unsupported`), automatic per-iteration verifier routing is host-driven, and the `consult_reference_models` tool registration plus Python's ability-manager integration are not ported.
- Persistent agent: remote/distributed handoff transports are `Error::Unsupported`, and the crate ships the `CompletionFn` summarisation seam but no model client behind it (a host injects one); group-chat participation, the budget service, archive storage and the `read_execution_record`/`search_session_history`/`new_context` tools are out of module scope.
- Personal context: the seven vendor fetchers (GitHub, GitCode, Feishu, browser bookmarks, Zhihu, Toutiao, RSS) return `Error::Unsupported` through `SourceProvider`; `context_pipeline.py`, `agent_support.py`, `file_tools.py`, the context-graph APIs and Markdown link rewriting are not ported.
- KV cache: Redis/distributed backends and provider prefetch/offload are `UnsupportedKvCacheBackend`; lineage is a seam and identity bookkeeping belongs to the host tracker.
- Watcher: no target add/remove after `start`, no cross-process watch locking, no automatic re-registration after an OS watch limit (reported as `WatcherFailed`), best-effort rename pairing.
- `agent_evolving` / RSI orchestrators and LLM-driven evolution bodies live in the layer-6 crates; only the harness-side rail and metrics bodies are here.

## Routing Table

- `./src/schema/` → config, state, task/todo, stop conditions, interaction, loop events, agent mode
- `./src/workspace/` → `Workspace`, named directory nodes, default schema, `DirectoryBuilder`
- `./src/permission/` → permission model, engine, file/shell/net guards, session decisions, HITL interrupt rail
- `./src/tools/` → model-facing tools (`registry.rs`, core and extended sets), the file-backed memory backend (`memory/workspace_backend.rs`), web fetch/search, media metadata, cron (classic + Quartz grammar, ten actions, run history), MCP resources and subscriptions, worktree enter/exit, goal, progressive discovery, descriptions, shared helpers
- `./src/rails/` → rail trait/registry, security/memory/skills/subagent/evolution rails, heartbeat rail, KV-cache lifecycle rail, personal-context rail (`personal_context/` store/ingest/search)
- `./src/moa/` → mixture-of-agents: config/validation, concurrent consultation, verifier/ranking, aggregation, state/history, `MoaRail`, embedded consult skill
- `./src/persistent_agent/` → execution journal/replay, execution catalogue and layout roots, persisted session state plus session listing, window assembly, continuation/resume (`ResumePlan` JSON view + window seed), live-turn recording (`recording.rs`: `ExecutionRecorder`/`ActiveExecution`/`PersistentRunConfig`), handoff, control/emitter/share-file seams, record rails
- `./src/prompts/` → system prompt builder, sections, tool descriptions, prompt report, sanitization
- `./src/subagents/` → `SubAgentConfig`, registry, presets, the rail's production `RuntimeSubAgentSpawner` (`spawner.rs`), quota/LRU/transcript/activity/eviction/lifecycle runtime, task and subagent tools
- `./src/skills/` → skill discovery/library state, mounting, `SKILL.md` validation
- `./src/task_loop/` → loop coordinator, queues, event manager, task-loop controller/handler
- `./src/agent/` → `DeepAgent`, builder, factory, goal-rail assembly, rail-event drain, subagent wiring (`subagent_factory.rs`, the three sub-agent tool modes), session state persistence, round execution, history/state seeding (`seed.rs`, `state_store.rs`), the tool retry budget (`resilience.rs`) and the invocation journal (`invocation_log.rs`)
- `./src/protocol/` → native harness protocol/provider implementation, event bridge, cursors
- `./src/lsp/` → JSON-RPC framing/transport, LSP client, diagnostics registry, server discovery/supervisor with retry, rendering, `lsp` tool, config/args/action text forms (`parse.rs`)
- `./src/watcher/` → debounced workspace watcher (`notify`), change classification, bounded subscribers, reload hub and skills/instruction-file/MCP reloaders, `WatcherConfig` text form
- `./src/spec/` → `DeepAgentSpec` templates, registry/precedence, plugins, embedded `resources/`, resolution into builder parts
- `./src/worktree/` → git runner, worktree manager/layout/registry, locking, sparse checkout, change summaries/cleanup, enter/exit tools, worktree rail, `worktree` tool
- `./src/cli/` → clap `run`/`chat` grammar, option resolution, one-shot runner over a stable agent card id, resumed one-shot runs (`resume.rs`), REPL and slash commands with session-id adoption (`Repl::with_session_id`), history, styling, instruction files, MCP config, allow-rule write-back, onboarding
- `./src/manifest.rs` → harness element catalog (`@harness_element` equivalent)
- `./src/observability.rs` → tracing spans/events and telemetry seam
- `./src/lib.rs` → crate docs, module declarations, flat re-exports, feature flags
- `./tests/` → offline integration suites (loop, workspace safety, permissions, state, subagents/transcripts, protocol, CLI, public surface, MoA, persistent-agent resume, goal rail events, watcher targets) (`tests/CONTEXT.md`)

## Offline recipe (DeepAgent + native harness)

- Minimal agent: `DeepAgentBuilder::new(AgentCard::new(id)).config(DeepAgentConfig::default().with_language("en").with_workspace(Workspace::for_language(tmp.path(), "en"))).model(Arc::new(ScriptedModel::text("hi"))).build()` → `Result<Arc<DeepAgent>>`, then `agent.invoke(json!({"query": "hi"}), None).await` (`Agent` trait) for one turn; canonical example `./tests/deep_agent_loop.rs`.
- One-call factory: `create_deep_agent(...)` at `./src/agent/factory.rs`; it always sets `enable_task_loop`/`restrict_to_work_dir` from its arguments and registers `security`, `skills`, `subagent` and (when a goal tool is mounted) `goal`.
- Extended tools: `ToolRegistry::new(workspace).with_tools(BuiltinTool::EXTENDED)` for everything, or a chosen subset; `with_lsp_config(..)`/`with_worktree_root(..)`/`with_mcp_subscriptions(..)`/`with_search_backend(..)` supply the configured tools, and absent backends yield `success=false` payloads.
- Spec-driven agent: `SpecRegistry::discover(workspace, &[home])?.resolve("name")?` then `ResolvedSpec::build_agent(model, ...)`; templates live in `<workspace>/.openjiuwen/agents/*.{yaml,yml,json,md}`, plugins in `<workspace>/.openjiuwen/plugins/<name>/plugin.yaml`.
- Native harness: `NativeHarness::new(Arc<DeepAgent>, NativeHarnessOptions::new())` then `start(HarnessContext::new(name, agent_id, host_session_id, prompt))`, `events()` (single consumer), `send(HarnessInput::text(..), DeliveryMode::Auto)`, `stop()`.
- LSP offline: `LspTool::new(LspToolConfig::default()).with_transport(Box::new(ChannelTransport::new(buffer)))` plus an in-process fake server task; no language server binary required.
- Worktree offline: `WorktreeManager::new(repo_root).with_runner(Arc::new(scripted_runner))` or a `tempfile` git repo when `git` is installed.
- Watcher offline: `WorkspaceWatcher::for_workspace(&workspace).start()` (feature `notify`) with a temp workspace, then mutate `skills/`, `IDENTITY.md` or the MCP config and read `Subscription::recv`; without the feature `start()` is a typed `Unsupported`.
- MoA offline: `MoaConfig` with two `ScriptedModel` proposers → `consult` → `rank_with_config` → `Aggregator`, all without a network.
- Persistent agent offline: `ExecutionLog::open(dir, identity, capacity)` (or `open_with_clock`) appends journal entries; `FileSessionStore` + `FileLinkStore` + `RootLogSource` feed `resume(&store, &logs, &links, &binding, request_id, &continuation, &WindowAssembler::new(Capacity::default()))`, which returns a `ResumePlan`; `LocalHandoff`/`ShareFile` stay on disk.
- Host resume offline: `PersistentAgentLayout::for_workspace(root)` gives the three roots; `ExecutionCatalog::new(layout.logs_root(), LogCapacity::default()).list()` and `FileSessionStore::list()` enumerate what is on disk; `resume(..)` builds the plan; then `run_with_model_resumed(&options, &args, &io, Arc::new(ScriptedModel::…), ResumedTurn::new(plan.record.session_id.clone()).with_history(plan.seed_history()).with_state(state))` continues the CLI session record, and `Repl::seed_history(plan.seed_history())` does the same for chat; canonical example `./tests/persistent_agent_resume.rs`.
- No default path opens a network connection; web fetch/search, MCP resources and the LSP server are only reached when a tool is invoked with a configured backend.

## Test Strategy

- Integration tests drive `DeepAgent` with `openjiuwen_test_support::ScriptedModel` and temp workspaces: no-tool answer, tool round, permission denial, interrupt/resume, max-iteration, follow-up queue, state round-trip.
- Workspace/path tests prove traversal rejection (`..`, absolute, drive letters) and default schema seeding.
- Permission tests cover allow/deny/ask precedence, session-scoped approvals, file/net guard folds.
- Protocol tests drive `NativeHarness` end to end (start/send/events/stop/checkpoint) and assert capability gating.
- Tool tests assert exact names/JSON schemas/description tables and `success=false` error payloads; web/media/cron/goal/spec/worktree suites use injected fetchers, fixture bytes, scripted git output and frozen clocks.
- `tests/memory_tools.rs` drives the eight memory tools (never the backend) over a temp workspace: Python path mapping, line windows, payload shapes, edit failures, keyword-search results, typed traversal failures, and that `DeepAgentBuilder` mounts the backend by default.
- Subagent tests cover quota refusals and frees, per-parent/depth limits, LRU eviction with transcript digests, and persistence round-trips.
- `tests/config_flags.rs` pins the configuration knobs end to end over a fake failing `Tool` and `ScriptedModel`: `configure` rejection keeps the previous config, the retry budget for an idempotent, a non-idempotent and an always-failing tool plus the switch-off case, task-planning tool gating, the invocation journal (a recorded turn, an unset directory, an unusable directory), the project-root boundary with and without a root, and the three sub-agent tool modes.
- `tests/public_surface.rs` also guards the flat re-exports of the completion seam (`CompletionFn`, `CompletionSummarizer`) and the MCP notification surface (`NotificationOutcome`, the four method constants, both live-client refusals).
- `tests/subagent_spawner.rs` proves a built `DeepAgent` exposes a working spawner over its own runtime, a caller-supplied `subagent` rail still wins, an agent without sub-agents has none, and the concurrency ledger rejects the extra live spawn and frees the slot at a terminal instance.
- CLI tests inject reader/writer/env/home/shell/model and assert slash commands, history round-trip, colour opt-out byte output, instruction precedence, MCP config errors, allow-rule YAML write-back, `!cmd`, skill slash, the onboarding paths, resumed-run output parity and that seeding precedes the new query in the model request.
- LSP tests cover framing edge cases, transport lifecycles, timeout/cancel/retry, diagnostics push+pull, discovery, a fake in-process server, and every text form (`LspToolConfig` spec, `LspAction` spellings, `LspArgs` grammar) with its error paths and `Display` round-trips.
- Cron tests cover both grammars, every action's happy and failure path, run-history rotation and cross-instance persistence with frozen clocks.
- MoA tests use scripted proposers for fan-out, partial/all-fail, ranking, aggregation, rail activation and history round-trips.
- Persistent-agent tests append/rotate/reopen a journal, resume after a simulated crash, assemble windows under tight budgets, assert share-bundle/`Unsupported` behaviour, enumerate a log root and a session store, and round-trip a `ResumePlan` through JSON.
- `tests/persistent_deep_agent.rs` journals two real turns with `persistent_agent` on, re-derives execution/catalogue/journal/session from disk alone, resumes into a continued transcript, and asserts recording is off by default.
- Host resume tests run turn one through the CLI path, drop every handle, rediscover the execution and session from disk only, then continue the CLI record with `run_with_model_resumed`, asserting the rebuilt window precedes the new query and the transcript is continued rather than truncated.
- Watcher tests (feature `notify`) drive OS notifications over a temp workspace: create/modify/remove delivery, burst coalescing, reload handlers, error storms and idempotent shutdown; the feature-off build asserts the typed `Unsupported` reason; `WatcherConfig` has its own feature-independent `FromStr`/`Display` suite.
- Rails tests cover heartbeat cadence with an injected clock, KV-cache affinity/release, personal-context path safety, ingestion and BM25F recall.
- Definition of done: `cargo fmt -p openjiuwen-harness`, `cargo test -p openjiuwen-harness`, `cargo test -p openjiuwen-harness --no-default-features`, `cargo clippy -p openjiuwen-harness --all-targets -- -D warnings`, `cargo doc -p openjiuwen-harness --no-deps` all clean and offline.
