# rails

## Intent

The harness hook layer: one rail trait covering the nine kernel lifecycle events plus the four outer task-loop events, a priority-ordered copy-on-write registry, and the built-in rails (security, memory, skills, subagent, evolution, heartbeat, KV-cache lifecycle, personal context).
Ports `openjiuwen/harness/rails/` and the hook model of `openjiuwen/core/single_agent/rail/base.py` (`AgentRail`).
The evolution area is both a seam for the layer-6 orchestrators and a complete harness-side body (`MetricsEvolutionRail`) that records metrics, proposals and a run snapshot on its own.
The heartbeat rail (`heartbeat.rs`) injects the heartbeat instructions on heartbeat or due runs and queues a bounded event log its host can drain.
The KV-cache area (`kv_cache/`) is the subagent cache-affinity policy, the provider seam it drives, and the rail that translates subagent tool payloads into lifecycle transitions.
The personal-context area (`personal_context/`) is the portable offline subset of Python's package: status codes, path safety, source metadata, the source-link book, local ingestion, BM25F recall, the persisted store and the prompt rail.
Rails never call the model and never execute tools: they mutate the context handed to them and return control signals.
The agent owns the firing points; this module owns the contract.
Deferred: the LLM-driven evolution bodies (RSI search, `agent_evolving` trainers) stay in their own crates and are not reimplemented here.

## API Surface

`HarnessRail` (`#[async_trait]`, `Send + Sync`, object-safe; crate root re-exports the same names):
- `name() -> &str`, `priority() -> i32` (default `rail_priority::DEFAULT` = 50).
- Kernel hooks (mirror `openjiuwen_agent::AgentRail`): `before_invoke`, `after_invoke`, `before_model_call`, `after_model_call`, `on_model_exception`, `before_tool_call`, `after_tool_call`, `on_tool_exception`, `after_react_iteration`.
- Outer-loop hooks: `on_user_message`, `before_steering_drain`, `before_task_iteration`, `after_task_iteration`.
- Every hook is a no-op `Ok(())` default; a returned `Err` aborts the run for that firing point.

`HarnessRailContext { pub base: RailContext, pub task: Option<TaskIterationInfo> }` — `base` is the kernel context (messages, tools, tool call/result, output, `extra`, control flags); `task` is set only while a task-iteration hook fires.
- Builders: `new(base)`, `for_event(RailEvent)`, `with_task(TaskIterationInfo)`; accessors `event()`, `base()`, `base_mut()`, `task_info()`.
- `extra()` / `extra_mut()` expose the per-invocation scratch map shared by all rails.
- Control signals delegate to `base`: `request_skip_tool(msg)`, `request_interrupt(InterruptRequest)`, `request_force_finish(AgentOutput)`, `request_retry(Option<Duration>)`, `request_model_continue()`, `is_interrupted()`.

`TaskIterationInfo { round_id: u64, iteration: usize, is_follow_up: bool, loop_event: Option<DeepLoopEventType> }` with `new`/`with_follow_up`/`with_loop_event`.
`TaskIterationPhase { Before, After }` selects the task-iteration hook a registry call fires.

`HarnessRailRegistry` (cloneable handle over copy-on-write snapshots; firing never holds a lock):
- `register(Arc<dyn HarnessRail>)` replaces a rail of the same name and re-sorts (priority descending, insertion order for ties); `unregister(name) -> bool`; `rails() -> Vec<Arc<dyn HarnessRail>>` / `snapshot() -> Arc<Vec<...>>` (cheap clone); `names()`, `len()`, `is_empty()`.
- `fire(&mut HarnessRailContext)` dispatches on `ctx.base.event`; `fire_task_iteration(ctx, info, phase)` sets `ctx.task` first; `fire_user_message(ctx)`; `fire_steering_drain(ctx)`. Dispatch stops at the first `Err`.
- `to_agent_rails() -> Vec<Arc<dyn AgentRail>>` wraps every rail in a `HarnessRailAdapter`.

`HarnessRailAdapter` implements `openjiuwen_agent::AgentRail` for an `Arc<dyn HarnessRail>`: it `mem::replace`s the incoming `RailContext` into a `HarnessRailContext`, awaits the harness hook, then copies the context (and therefore every control flag) back before returning the hook result — flags survive a hook error.

`rail_priority` constants mirror the Python ladder: 100 `SYS_OPERATION`; 95 `MCP`/`SKILLS`/`SUBAGENT`; 90 `PLANNING`/`INTERRUPT`/`SECURITY`; 85 `AGENT_MODE`/`CONTEXT`/`SAFETY_PROMPT`; 80 `MEMORY`/`HEARTBEAT`; 70 `RESILIENCE`; 60 `EVOLUTION`/`LSP`; 50 `DEFAULT`; 10 `COMPLETION`/`MOA`.

Prompt-section channel (the seam to `crate::agent`'s prompt assembly):
- `publish_prompt_section(ctx, &PromptSection)` merges a section into `ctx.extra[PROMPT_SECTIONS_EXTRA_KEY]` keyed by section name (JSON: `{"content": {"cn": …, "en": …}, "priority": 40, "carrier": "system_message", "category": "skills"}`, `category` from `PromptCategory::as_str`).
- `prompt_sections(ctx) -> Vec<PromptSection>` decodes the channel, sorted by `(priority, name)`; unknown fields are ignored.
- `last_user_text(&[BaseMessage]) -> Option<String>` is the shared "current turn text" helper (last `Role::User` message, flattened, non-blank).

Built-in rails:
- `SecurityRail` (priority `SAFETY_PROMPT` = 85, Python `SafetyPromptRail`): wraps `Arc<openjiuwen_security::GuardrailPipeline>` plus `ScanTargets`. Scans `before_invoke` (user input), `before_model_call` (prompt messages), `after_model_call` (reply text), `before_tool_call` (tool name + parsed arguments), `after_tool_call` (tool output value). A blocked user-input/prompt/model-output scan returns `Error::security`; a blocked tool input calls `request_skip_tool` and returns `Ok`; a blocked tool output returns `Error::security`. Pipeline failures are fail-closed by default (the pipeline's default `FailurePolicy::FailClosed` turns a failing guardrail into a blocking finding, so the turn aborts); only an explicitly configured `FailurePolicy::FailOpen` logs them and lets the run continue. Constructors: `new(GuardrailPipeline)`, `from_arc`, `from_config(&SecurityConfig)` (propagates detector-config validation), `with_targets`; `Default` is an empty pipeline.
- `MemoryRail` (priority `MEMORY` = 80): on `before_model_call` publishes `prompts::memory_section()` and, when a `MemoryRecall` seam is wired, recalls for the current turn, injects one `<memory-context>` system message after the leading system messages, and records the text under `MEMORY_CONTEXT_EXTRA_KEY` (the query under `MEMORY_QUERY_EXTRA_KEY`, so state never lives in the shared rail). On `after_invoke` it hands a `TurnRecord` (query, final output, requested tool names) to the `MemoryWriteBack` seam. Recall/write-back errors are logged and swallowed. `MemoryRailConfig { recall_limit: 5, max_context_chars: 2000, publish_prompt_section: true, write_back: true }` with `with_*` builder setters (a `const fn` each).
- Memory seams: `MemoryRecall::recall(query, limit) -> Result<Vec<RecalledMemory>>`, `MemoryWriteBack::write_back(&TurnRecord) -> Result<()>`, data types `RecalledMemory { content, score, source }` and `TurnRecord { query, response, tool_calls }`. `LongTermMemoryAdapter` implements both over `openjiuwen_memory::LongTermMemory` + a `MemoryScope` (search for recall, one `NewMemory` per turn for write-back); its default kind is `SemanticMemory` and its default write-back source is `DEFAULT_MEMORY_SOURCE` (`"deepagent"`), overridable with `with_kind`/`with_source`.
- `SkillsRail` (priority `SKILLS` = 95): owns a `crate::skills::SkillLibrary` plus `SkillsRailConfig { budget: SkillBudget, language: Option<String> }`. `before_model_call` refreshes the discovery/selection cache when the invocation marked it stale, then publishes `prompts::skills_section(&selected)`. `after_invoke` marks the cache stale so the next invocation re-discovers; `library_mut()` also re-stales. A discovery error is logged and keeps the previous selection, so `refresh()` never fails today. Accessors `skills()`, `library()`, `library_mut()`, `budget()`, `config()`, `refresh()` (explicit async refresh).
- `SubagentRail` (priority `SUBAGENT` = 95): publishes `prompts::subagent_section()` on `before_model_call` and exposes an optional `Arc<dyn SubAgentSpawner>`.
- Subagent seam: `spawn(SubAgentSpawner::spawn(SpawnRequest)) -> Result<SubAgentHandle>`, `wait(id, Option<Duration>) -> Result<SubAgentStatus>`, `list() -> Result<Vec<SubAgentStatus>>`, `send_input(id, String) -> Result<()>`, `close(id) -> Result<()>`; data types `SpawnRequest { prompt, preset, agent_id, parent_task_id }`, `SubAgentHandle { id, preset }`, `SubAgentStatus { id, state, output, error }`, `SubAgentState { Running, Completed, Failed, Closed }`. The runtime lives in `crate::subagents`; this module only defines the seam.
- `HeartbeatRail` (name `HEARTBEAT_SECTION` = `"heartbeat"`, priority `rail_priority::HEARTBEAT` = 80): `new(HeartbeatRailConfig)`, `with_clock(HeartbeatClock)`, `config()`, `is_due(agent_id, session_id)`, `observe(agent_id, session_id, run_kind) -> Option<HeartbeatEvent>`, `events()`, `drain_events()`, `last_fired_ms(agent_id, session_id)`, `reset_session`, `reset_all`; its only hook is `before_model_call`.
- Heartbeat data: `HeartbeatRailConfig { interval_ms: 60_000, language: "cn", event_capacity: 64, session_capacity: 1024, force_on_heartbeat_run: true, publish_prompt_section: true }` (serde snake_case, `default`, one `with_*` setter per field), `HeartbeatEvent { agent_id, session_id, sequence, fired_at_ms, reason }`, `HeartbeatReason { Scheduled, RunKind }`, and `HeartbeatClock = Arc<dyn Fn() -> u64 + Send + Sync>` with `default_heartbeat_clock()` (monotonic milliseconds).
- Heartbeat helpers and constants: `build_heartbeat_section(language)`, `heartbeat_prompt(language)` (English for `"en"`, Chinese for everything else), `HEARTBEAT_SECTION_PRIORITY` = 78, `RUN_KIND_EXTRA_KEY` = `"run_kind"`, `HEARTBEAT_RUN_KIND` = `"heartbeat"`, `DEFAULT_HEARTBEAT_INTERVAL_MS` = 60_000, `DEFAULT_HEARTBEAT_EVENT_CAPACITY` = 64, `DEFAULT_HEARTBEAT_SESSION_CAPACITY` = 1024.
- Heartbeat firing: a run whose `extra[RUN_KIND_EXTRA_KEY]` is `HEARTBEAT_RUN_KIND` fires when `force_on_heartbeat_run`; otherwise a `(agent_id, session_id)` pair fires once `interval_ms` have elapsed since its baseline, and the first observation of a pair only records that baseline. `interval_ms == 0` disables the periodic path, `publish_prompt_section == false` still queues events, and both capacities of `0` disable their bound (`event_capacity == 0` keeps every event, so the host must drain; `session_capacity == 0` keeps every pair).
- Heartbeat pair bound: every observation stamps its pair as most recently seen, and a map over `session_capacity` evicts the least-recently-seen pairs first (ties on the map key order, the observed pair is never evicted).
- `KvCacheLifecycleRail` (name `"kv_cache_lifecycle"`, priority `rail_priority::DEFAULT` = 50): `new(Arc<KvCacheSubagentLifecycle>)`, `lifecycle()`; it drives `after_tool_call` only.
- `KvCacheSubagentLifecycle`: `new(Arc<dyn KvCacheProviderRegistry>)` (affinity off), `with_config(KvCacheAffinityConfig)`, `with_lineage(Arc<dyn KvCacheLineage>)`, `affinity_enabled()`, `registry()`, `sessions()`, `get(id)`, `prepare_subagent(KvCacheSubagent)`, `finish_subagent(id, succeeded)`, `evict_subagent(id)`, `release_all()`.
- KV-cache data: `KvCacheSubagent { subagent_id, subagent_type, sub_session_id, parent_cache_id }`, `KvCacheSubagentSession` (the same identity plus `state` and the ordered `actions` history), `KvCacheAction { Prepare, Suspend, Release }`, `KvCacheSubagentState { Active, Suspended, Released }` — all serde snake_case.
- KV-cache seams and policy: `KvCacheProviderRegistry` (`backend`, `prepare`, `suspend`, `release`), `UnsupportedKvCacheBackend::new(backend)`, `KvCacheAffinityConfig { enable_kv_cache_affinity }`, `affinity_enabled`, `is_sticky_subagent_type`, `KvCacheLineage`/`RuntimeSessionLineage`, `resolve_subagent_parent_cache_id`, `scope_sub_session_id`, `resolve_sub_session_id`, `AFFINITY_PARENT_SESSION_ID_ENV`, `STICKY_SUBAGENT_TYPE` (`"verification_agent"`).
- `PersonalContextRail` (name `PERSONAL_CONTEXT_SECTION` = `"personal_context"`, priority `PERSONAL_CONTEXT_PRIORITY` = 40): `new(PersonalContextRailConfig)`, `with_store`, `store()`, `config()`, `context_root()`, `description_path()`, `agent_use_enabled()`, `refresh_switch()`, `search(query)`, `description()`; hooks `before_invoke`, `before_model_call`, `after_model_call`, `after_invoke`.
- `PersonalContextRailConfig`: `new(home)`, `with_home`, `with_config_filename`, `with_language`, `with_max_description_chars`, `with_section_priority`, `with_section_name`, `config_path()`; defaults are the switch `<home>/personal_context.yaml`, language `cn`, 3000 description characters and the `personal_context` section name.
- Personal-context store and ingestion: `PersonalContextStore` (`new(home)`, `home`, `context_root`, `description_path`, `index_path`, `source_root`, `link_book_path`, `metadata_store`, `entries`, `entry`, `search`, `ingest_local`, `link_book`, `save_link_book`, `write_description`), `ContextEntry`, and the `ingest` module's `ingest_local_directory`, `LocalDirectoryConfig`, `IngestedDocument`, `IngestReport`.
- Personal-context recall: the `search` free function plus `SearchDocument`, `SearchHit`, `snippet`, `tokenize` (BM25F over title/path/headings/body), and `messages_are_contiguous` + `resolve_context_relative_path` as the rail's guards. The token and heading patterns are `OnceLock<Option<Regex>>` constants resolved through `resolve_pattern`, which reports a broken constant once as a typed `Error::config` and leaves `tokenize`/`page_heading_fields` degrading to an empty token stream instead of panicking.
- Personal-context sources and providers: `SourceMetadata`, `SourceMetadataStore`, `SourceLink`, `SourceLinkBook`, `SourceKind`, `SourceProvider`, `SourceProviderRegistry`, `LocalFilesProvider`, `UnsupportedSourceProvider`, `FetchRequest`, `FetchedSource`, `PersonalContextStatus`, `build_error`, `normalize_source_locator`, `source_id_for_locator`.
- The full personal-context surface is reachable under `rails::personal_context::*`; `rails` re-exports the commonly wired items flat (rail, config, store, providers, search, source metadata/link book, status).
- `EvolutionRail` (priority `EVOLUTION` = 60) + `NoopEvolutionRail`: synchronous metadata hooks `on_before_invoke`, `on_after_model_call`, `on_after_tool_call`, `on_after_task_iteration`, `on_after_invoke` plus `name()`/`priority()`. Every hook is a no-op `Ok(())` default, so an upper-layer implementation overrides only what it needs and the layer-6 orchestrators keep compiling.
- Evolution hook context: `on_before_invoke` sees the run identity (`base.agent_id`, `base.session`) and an empty `base.extra`; `on_after_model_call` sees `base.response` (message with tool calls, `usage`, `finish_reason`) and `base.iteration`; `on_after_tool_call` sees `base.tool_call` and/or `base.tool_result` (either may be absent, so read defensively); `on_after_task_iteration` sees `ctx.task` (`TaskIterationInfo`) — `base.event` is *not* a reliable phase discriminator there; `on_after_invoke` sees the terminal `base.output` (`termination_reason`/`finish_reason`) and the whole run's `extra`.
- `EvolutionRailAdapter` (built by `evolution_adapter`) implements `HarnessRail`, forwards exactly those five hooks, and logs and swallows every hook `Err` — an evolution failure never aborts a run; the other eight harness hooks stay no-ops.
- `MetricsEvolutionRail` (`name()` = `"metrics-evolution"`, priority `EVOLUTION`): the harness-side body that needs no upper layer — mints a fresh `run_id` and clears both buffers on `on_before_invoke`, counts model calls/usage/finish reason, counts tool calls by name and failures, closes one `EvolutionMetrics` row per task iteration, and writes the snapshot on `on_after_invoke`.
- `MetricsEvolutionRailConfig { metrics_capacity: 256, proposal_capacity: 64, snapshot_path: None, proposal_path: None }` with `with_*` setters; `0` capacity means unbounded, a missing path means "in memory only".
- `EvolutionMetrics` (serde + `JsonSchema` + `Default`): `iteration`, `elapsed_ms`, `model_calls`, `tool_calls`, `failed_tool_calls`, `tool_calls_by_name: BTreeMap<String, u64>`, optional `input_tokens`/`output_tokens`/`total_tokens` (left `None` unless a reported sum is `> 0`, via `with_usage`), `stop_reason`; helper `record_tool_call(name, failed)`.
- `IterationRecorder`: bounded ring buffer (`capacity == 0` unbounded, oldest evicted first) over `EvolutionMetrics` with `push`, `list`, `clear`, `capacity`, `len`, `is_empty`, `now_ms`, and `totals()` (folded counters, merged per-tool map, summed tokens, last non-`None` stop reason, `iteration` = folded row count).
- `MetricsClock = Arc<dyn Fn() -> u64 + Send + Sync>` with `default_metrics_clock()` (monotonic milliseconds since the first reading) and `IterationRecorder::with_clock` / `MetricsEvolutionRail::with_clock` for deterministic tests.
- `ProposalLog` (bounded, oldest first) + `EvolutionProposal { id, kind, summary, created_at_ms }`: `new(capacity)`, `with_path`, `path`, `push`, `entries`, `clear`, `save`, `load`. `save` is atomic (`<path>.tmp` + rename) and returns `Error::config` without a configured path; `load` returns `Error::validation` on a malformed document and an empty log when the file is absent. The document is `{"version": 1, "proposals": [...]}`; a bare array is accepted too. `MetricsEvolutionRail::push_proposal` stamps an unstamped (`created_at_ms == 0`) proposal with the rail clock.
- `EvolutionSnapshot` (`EVOLUTION_SNAPSHOT_VERSION` = 1; serde + `JsonSchema`): `version`, `run_id`, `agent_id`, `session_id`, `started_at_ms`, `finished_at_ms`, `duration_ms`, `iteration_count`, `model_calls`, `tool_calls`, `failed_tool_calls`, `stop_reason`, `iterations`, `proposals`; `save(&Path)` / `load(&Path)` use the same atomic writer, tolerate unknown fields, and default a missing `version` to 1.
- The rail's fallible surface stays public — `snapshot()` (in memory, and complete even mid-run because unflushed counters are folded in) and `write_snapshot()` — while the hooks only `tracing::warn!` a failure and swallow it.

## Priorities

Dispatch order is the `rail_priority` ladder above, descending, with registration order kept for ties.
`HeartbeatRail` uses `rail_priority::HEARTBEAT` (80) and publishes its section at `HEARTBEAT_SECTION_PRIORITY` (78), the Python slot and render priority of the heartbeat rail.
`PersonalContextRail` and its published section both use 40, i.e. below the ladder's `DEFAULT` (50), mirroring Python's class priority because the description is a runtime attachment rather than base system-prompt content.
`KvCacheLifecycleRail` uses `rail_priority::DEFAULT` (50) because affinity is a best-effort optimisation rather than a prompt-shaping or safety rail.

## Constraints

- Rails mutate only the context; control signals are flags on `RailContext`; a rail `Err` aborts the run (the only exceptions are the documented fail-open rails: security infrastructure failures under `FailurePolicy::FailOpen`, memory seams, evolution hooks, and the heartbeat/KV-cache/personal-context rails, which never return `Err`).
- Registry snapshots are `Arc` swaps: `fire` clones the snapshot once, so `register`/`unregister` during a run is safe, lock-free while hooks execute, and takes effect on the next firing point.
- Never log prompt text, tool arguments/outputs, memory content, description text, or secrets: log guardrail/pipeline names, counts and risk types only.
- Only this module imports `openjiuwen_security` / `openjiuwen_memory` kernel APIs, and it must not import `crate::tools`, `crate::agent`, or `crate::subagents`.
- No panics on user input, no `unsafe`, `tracing` only (no stdout), files ≤1000 lines, every public item documented.
- Per-invocation state lives in `ctx.extra` or in the context, never in rail fields (rails are shared across invocations); `MetricsEvolutionRail`, `HeartbeatRail` and `KvCacheSubagentLifecycle` are the documented exceptions because an invocation (or a delegation) is their unit of work.
- State is per rail instance: the heartbeat cadence and event queue, the KV-cache session map and the personal-context switch flag live on their own object; there are no process-global registries.
- No lock is held across `.await`: heartbeat and KV-cache critical sections are short `parking_lot` scopes around reads/records, and provider calls happen outside them.
- The evolution module imports no evolution crate, reads time only through `MetricsClock`, and writes files only through the `evolution/store.rs` atomic writer (`<path>.tmp` + rename); no new crates.io dependencies.
- Evolution failures are isolated: IO and serialization errors from `MetricsEvolutionRail` are `tracing::warn!`-ed and swallowed, and the rail never logs proposal summaries or prompt text.
- The personal-context subtree reads time only through injected `observed_at_ms` arguments and home only through its configured path, and every filesystem access is byte-capped.

## Design decisions

- The Python rail protocol is identity-based override detection over 13 `AgentCallbackEvent` hooks; Rust has no such introspection, so every `HarnessRail` hook is an explicit no-op default and the registry always fires all of them.
- Python's `_skip_tool`/`tool_result` convention and "model-event reject → `request_force_finish`" become: `skip_tool` for tool inputs, `Error::security` for prompt/user/model-output and tool-output blocks. Documented deviation: an aborting error is unambiguous in Rust and already means "abort the run".
- Live services arrive as seam traits (`MemoryRecall`, `MemoryWriteBack`, `SubAgentSpawner`, `EvolutionRail`, `KvCacheProviderRegistry`, `SourceProvider`) instead of builder-wired objects, so this module stays free of runtime dependencies and upper layers keep the wiring.
- The prompt-section channel is a JSON contract in `ctx.extra`, not a rail-owned builder handle: the agent renders sections when it assembles the prompt, and rails stay pure context mutators.
- The evolution trait is deliberately additive: its three `on_after_*` hooks have no-op defaults, so the layer-6 orchestrators (`openjiuwen-agent-evolving`, `openjiuwen-rsi`) compile unchanged and decide which firing points they want.
- Metrics are sampled at the harness instead of being reconstructed by the orchestrator: the rail sees `AfterModelCall`/`AfterToolCall`/`AfterTaskIteration`, which no upper layer observes directly, and the persisted snapshot is the artefact they consume.
- The heartbeat periodic cadence is additive: Python has only the explicit `run_kind` path, while a host that drives the agent itself has no scheduler to mark a run, so the rail also fires once per `(agent, session)` interval and records the event.
- The first observation of a `(agent, session)` pair only records a baseline, so a fresh run never receives heartbeat text on its first model call; `reset_session` restarts one pair and `reset_all` restarts every pair.
- A heartbeat section is static prompt content, so the rail publishes it on `before_model_call` and never clears it; the personal-context section is a temporary runtime attachment and is cleared before publishing, after the model call and after the invocation.
- The KV-cache rail observes subagent tool payloads (`subagent_spawn`/`task_tool`/`task` prepare, `subagent_wait` finish, `subagent_close` evict) because the runtime lives in `crate::subagents`, which rails must not import.
- A host with direct runtime access calls the lifecycle methods itself, notably `evict_subagent`/`release_all` for LRU eviction, which no tool payload reports.
- Vendored personal-context fetchers are typed `Error::Unsupported` through `UnsupportedSourceProvider` and `SourceProviderRegistry::with_unsupported_vendors`, never silent no-ops.
- Source metadata is one JSON document per source with Python's field names and `src_<32 hex>` ids, which keeps ids wire-compatible while removing Python's Markdown heading parser; markdown link rewriting stays deferred.
- SHA-256 for identifier digests is `openjiuwen_sysop::util::sha256_hex` (the workspace's `sha2`-based helper); `rails/sha256.rs` only adds the `sha256_prefix` truncation wrapper, and the digest is an identifier, never a security primitive.

## Known Issues

- The crate-root `src/lib.rs` flat export list is owned by the integration layer and does not yet re-export the heartbeat, KV-cache or personal-context names; reach them through `rails::…` until it does.
- Personal-context vendor transports, Markdown link rewriting (`normalize_source_links`, `rewrite_markdown_prose`) and `pcs_search` tool mounting are deferred; `PersonalContextRail::search` is the capability a host wraps in a tool.
- The KV-cache rail cannot observe LRU eviction: the runtime's eviction path must call `KvCacheSubagentLifecycle::evict_subagent` directly.
- `HeartbeatRail` names itself with the string `"heartbeat"` because `SectionName::HEARTBEAT` lives in `src/prompts`, outside this node.

## Routing Table

- `./mod.rs` → crate docs, module declarations, `HarnessRail`, `rail_priority`, re-exports, and the cross-module composition tests
- `./context.rs` → `HarnessRailContext`, `TaskIterationInfo`, `TaskIterationPhase`, prompt-section channel, `last_user_text`
- `./registry.rs` → `HarnessRailRegistry` (copy-on-write dispatch) and `HarnessRailAdapter` (kernel bridge)
- `./security.rs` → `SecurityRail` (guardrail scan of user input, prompts, model output, tool I/O)
- `./memory.rs` → `MemoryRail`, `MemoryRecall`/`MemoryWriteBack` seams, `LongTermMemoryAdapter`
- `./skills.rs` → `SkillsRail` (discovery cache, budget, skills prompt section)
- `./subagent.rs` → `SubagentRail` + `SubAgentSpawner` seam and request/status data
- `./heartbeat.rs` → `HeartbeatRail`, heartbeat/interval cadence, bounded event queue and bounded pair map, prompt texts and section builder
- `./kv_cache/` → affinity policy helpers, provider seam, subagent lifecycle and `KvCacheLifecycleRail` (`kv_cache/CONTEXT.md`)
- `./personal_context/` → status codes, path safety, source metadata/link book, ingestion, BM25F search, store, provider seams and the rail (`personal_context/CONTEXT.md`)
- `./sha256.rs` → `sha256_prefix` truncation wrapper over `openjiuwen_sysop::util::sha256_hex` for stable source, link-book and KV-cache identifiers
- `./evolution/` → the self-evolution rail seam and its harness-side bodies
- `./evolution/mod.rs` → `EvolutionRail` (five hooks), `NoopEvolutionRail`, `EvolutionRailAdapter`/`evolution_adapter`, module re-exports
- `./evolution/metrics.rs` → `EvolutionMetrics`, `IterationRecorder`, `MetricsClock`, `default_metrics_clock`
- `./evolution/proposal.rs` → `EvolutionProposal`, `ProposalLog` (bounded, atomic save/load)
- `./evolution/snapshot.rs` → `EvolutionSnapshot` (+ `EVOLUTION_SNAPSHOT_VERSION`), save/load
- `./evolution/rail.rs` → `MetricsEvolutionRail`, `MetricsEvolutionRailConfig`, run state and hook bodies
- `./evolution/store.rs` → shared atomic JSON writer/reader (`<path>.tmp` + rename)

## Test Strategy

- Unit tests live beside each module (`#[cfg(test)] mod tests`) and never touch the network or the real model: `ScriptedModel`/hand-built `BaseMessage`s, `tempfile` roots and injected clocks only.
- `context.rs`: builders, `extra_mut`, control-signal delegation, prompt-section publish/collect round-trip (category, carrier, priority, overwrite by name), `last_user_text`.
- `registry.rs`: ordering (priority desc, stable ties), replace-by-name, unregister, per-event dispatch of all nine kernel hooks, error aborts later rails, task-iteration hooks see `ctx.task` per phase, registration during `fire` is not observed by the running snapshot.
- `registry.rs` adapter: forwards the nine kernel events exactly once, copies every control flag back, and keeps flags even when the hook returns `Err`.
- `security.rs`: fake `Guardrail` (pass/block/fail) per stage — prompt/user/model-output/tool-output blocks return `Error::security`, tool-input block sets `skip_tool` and returns `Ok`, disabled scan targets skip the pipeline, a failing guardrail blocks the turn under the default `FailClosed` policy (message names `guardrail_error`) and is swallowed only under an explicit `FailOpen` policy.
- `memory.rs`: fake recaller/writer — injection position and `<memory-context>` wrapper, no memories → no message, recall/write-back errors swallowed, turn content handed to the writer, `LongTermMemoryAdapter` round-trip over an in-memory facade.
- `skills.rs`: temp roots with `SKILL.md` — refresh selects and publishes the section, budget/disabled filtering, stale-once-per-invocation refresh, missing roots degrade to an empty selection.
- `subagent.rs`: section publication, spawner accessor, a fake spawner exercises the whole seam.
- `heartbeat.rs`: section text and priority per language with the Chinese fallback, an injected clock driving the cadence (baseline-only first observation, due firing, `interval_ms == 0` disabling the periodic path, marked-run precedence), pair isolation, bounded and unbounded queues, least-recently-seen pair eviction, `events` as a snapshot vs `drain_events`, `reset_session`, and config serde.
- `kv_cache/`: policy helpers (off-by-default affinity, trimmed stickiness with `browser_agent` non-sticky, scope digests, lineage degradation), the typed unsupported backend, and a recording provider fake covering OFF inertness, sticky prepare→suspend, release on failure, failing prepare, idempotent evict, `release_all`, and the rail's spawn/wait/close/ignored-frame/swallowed-error paths.
- `personal_context/`: status codes and `ErrorKind` mapping, the path-safety rejection matrix, source ids/metadata, link book, ingest caps and classification, BM25F ranking and snippets, provider `Unsupported` messages, store round-trip, and the rail's switch truth table, description bounds, section publish/clear isolation, contiguity matrix, search guards and an ingest → description → section → search recall test.
- `mod.rs` integration tests: heartbeat, KV-cache and personal-context rails registered in one `HarnessRailRegistry` — section ordering and both texts on one `BeforeModelCall`, the queued `HeartbeatReason::RunKind` event, clearance of the temporary section on `AfterModelCall`, the spawn → wait → close provider sequence, affinity-off inertness, and a disabled switch publishing no personal-context section.
- `evolution/mod.rs`: adapter forwards all five hooks exactly once, swallows an `Err` from every hook, leaves the other harness hooks as no-ops, and `NoopEvolutionRail` mutates nothing (`name`/`priority` asserted).
- `evolution/metrics.rs`: usage stays `None` unless reported, per-tool/failure counting, bounded eviction order, unbounded `capacity == 0`, `clear`, injectable clock, folded totals incl. an empty recorder.
- `evolution/proposal.rs`: uuid + stamping, insertion order, cap eviction, atomic save/load round-trip, bare-array documents, missing file → empty, save without path → `Error::config`, malformed JSON → `Error::validation`.
- `evolution/snapshot.rs`: version default, save/load round-trip, unknown-field tolerance, missing file → `Error::config`, malformed → `Error::validation`, unwritable destination → `Error::config`.
- `evolution/rail.rs`: one metrics row per iteration (counts, failures, per-tool map, tokens, stop reason), counter reset between iterations, unflushed counters still in the totals, proposal stamping/ordering/cap and a clean buffer per run, snapshot round-trip through a temp file, fresh `run_id` per run, proposal-log persistence, a failing write swallowed by the hook but visible on `write_snapshot()`, no path → no write.
- `evolution/store.rs`: round-trip, missing file, malformed document, unwritable destination, temporary-path suffix.
- Parity references: `tests/unit_tests/harness/test_deep_agent_rail_event_routing.py`, `rails/test_typed_interrupts.py`, `rails/test_base_security_rail.py`, `rails/evolution/test_evolution_rail.py`, `test_skill_rail.py`, `rails/test_memory_rail.py`, `rails/test_personal_context_rail.py`.
- Definition of done: `cargo fmt -p openjiuwen-harness`, `cargo test -p openjiuwen-harness`, `cargo test -p openjiuwen-harness --no-default-features`, `cargo clippy -p openjiuwen-harness --all-targets -- -D warnings` clean.
