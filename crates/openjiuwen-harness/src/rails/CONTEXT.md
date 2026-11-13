# rails

## Intent

The harness hook layer: one rail trait covering the nine kernel lifecycle events plus the four outer task-loop events, a priority-ordered copy-on-write registry, and the built-in rails (security, memory, skills, subagent, evolution).
Ports `openjiuwen/harness/rails/` and the hook model of `openjiuwen/core/single_agent/rail/base.py` (`AgentRail`).
The evolution area is both a seam for the layer-6 orchestrators and a complete harness-side body (`MetricsEvolutionRail`) that records metrics, proposals and a run snapshot on its own.
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
- `SecurityRail` (priority `SAFETY_PROMPT` = 85, Python `SafetyPromptRail`): wraps `Arc<openjiuwen_security::GuardrailPipeline>` plus `ScanTargets`. Scans `before_invoke` (user input), `before_model_call` (prompt messages), `after_model_call` (reply text), `before_tool_call` (tool name + parsed arguments), `after_tool_call` (tool output value). A blocked user-input/prompt/model-output scan returns `Error::security`; a blocked tool input calls `request_skip_tool` and returns `Ok`; a blocked tool output returns `Error::security`. Pipeline failures are fail-open (logged, run continues) unless the policy says `FailClosed`. Constructors: `new(GuardrailPipeline)`, `from_arc`, `from_config(&SecurityConfig)` (propagates detector-config validation), `with_targets`; `Default` is an empty pipeline.
- `MemoryRail` (priority `MEMORY` = 80): on `before_model_call` publishes `prompts::memory_section()` and, when a `MemoryRecall` seam is wired, recalls for the current turn, injects one `<memory-context>` system message after the leading system messages, and records the text under `MEMORY_CONTEXT_EXTRA_KEY` (the query under `MEMORY_QUERY_EXTRA_KEY`, so state never lives in the shared rail). On `after_invoke` it hands a `TurnRecord` (query, final output, requested tool names) to the `MemoryWriteBack` seam. Recall/write-back errors are logged and swallowed. `MemoryRailConfig { recall_limit: 5, max_context_chars: 2000, publish_prompt_section: true, write_back: true }` with `with_*` builder setters (a `const fn` each).
- Memory seams: `MemoryRecall::recall(query, limit) -> Result<Vec<RecalledMemory>>`, `MemoryWriteBack::write_back(&TurnRecord) -> Result<()>`, data types `RecalledMemory { content, score, source }` and `TurnRecord { query, response, tool_calls }`. `LongTermMemoryAdapter` implements both over `openjiuwen_memory::LongTermMemory` + a `MemoryScope` (search for recall, one `NewMemory` per turn for write-back); its default kind is `SemanticMemory` and its default write-back source is `DEFAULT_MEMORY_SOURCE` (`"deepagent"`), overridable with `with_kind`/`with_source`.
- `SkillsRail` (priority `SKILLS` = 95): owns a `crate::skills::SkillLibrary` plus `SkillsRailConfig { budget: SkillBudget, language: Option<String> }`. `before_model_call` refreshes the discovery/selection cache when the invocation marked it stale, then publishes `prompts::skills_section(&selected)`. `after_invoke` marks the cache stale so the next invocation re-discovers; `library_mut()` also re-stales. A discovery error is logged and keeps the previous selection, so `refresh()` never fails today. Accessors `skills()`, `library()`, `library_mut()`, `budget()`, `config()`, `refresh()` (explicit async refresh).
- `SubagentRail` (priority `SUBAGENT` = 95): publishes `prompts::subagent_section()` on `before_model_call` and exposes an optional `Arc<dyn SubAgentSpawner>`.
- Subagent seam: `spawn(SubAgentSpawner::spawn(SpawnRequest)) -> Result<SubAgentHandle>`, `wait(id, Option<Duration>) -> Result<SubAgentStatus>`, `list() -> Result<Vec<SubAgentStatus>>`, `send_input(id, String) -> Result<()>`, `close(id) -> Result<()>`; data types `SpawnRequest { prompt, preset, agent_id, parent_task_id }`, `SubAgentHandle { id, preset }`, `SubAgentStatus { id, state, output, error }`, `SubAgentState { Running, Completed, Failed, Closed }`. The runtime lives in `crate::subagents`; this module only defines the seam.
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

## Constraints

- Rails mutate only the context; control signals are flags on `RailContext`; a rail `Err` aborts the run (the only exceptions are the documented fail-open rails: security infrastructure failures under `FailurePolicy::FailOpen`, memory seams, and evolution hooks).
- Registry snapshots are `Arc` swaps: `fire` clones the snapshot once, so `register`/`unregister` during a run is safe, lock-free while hooks execute, and takes effect on the next firing point.
- Never log prompt text, tool arguments/outputs, memory content, or secrets: log guardrail/pipeline names, counts and risk types only.
- Only this module imports `openjiuwen_security` / `openjiuwen_memory` kernel APIs, and it must not import `crate::tools`, `crate::agent`, or `crate::subagents`.
- No panics on user input, no `unsafe`, `tracing` only (no stdout), files ≤1000 lines, every public item documented.
- Per-invocation state lives in `ctx.extra` or in the context, never in rail fields (rails are shared across invocations); `MetricsEvolutionRail` is the documented exception — it keeps its counters in a `parking_lot::Mutex` because an invocation is its unit of work.
- The evolution module imports no evolution crate, reads time only through `MetricsClock`, and writes files only through the `evolution/store.rs` atomic writer (`<path>.tmp` + rename); no new crates.io dependencies.
- Evolution failures are isolated: IO and serialization errors from `MetricsEvolutionRail` are `tracing::warn!`-ed and swallowed, and the rail never logs proposal summaries or prompt text.

## Design decisions

- The Python rail protocol is identity-based override detection over 13 `AgentCallbackEvent` hooks; Rust has no such introspection, so every `HarnessRail` hook is an explicit no-op default and the registry always fires all of them.
- Python's `_skip_tool`/`tool_result` convention and "model-event reject → `request_force_finish`" become: `skip_tool` for tool inputs, `Error::security` for prompt/user/model-output and tool-output blocks. Documented deviation: an aborting error is unambiguous in Rust and already means "abort the run".
- Live services arrive as seam traits (`MemoryRecall`, `MemoryWriteBack`, `SubAgentSpawner`, `EvolutionRail`) instead of builder-wired objects, so this module stays free of runtime dependencies and upper layers keep the wiring.
- The prompt-section channel is a JSON contract in `ctx.extra`, not a rail-owned builder handle: the agent renders sections when it assembles the prompt, and rails stay pure context mutators.
- The evolution trait is deliberately additive: the three `on_after_*` hooks arrived with no-op defaults so the layer-6 orchestrators (`openjiuwen-agent-evolving`, `openjiuwen-rsi`) keep compiling unchanged and decide which firing points they want.
- Metrics are sampled at the harness instead of being reconstructed by the orchestrator: the rail sees `AfterModelCall`/`AfterToolCall`/`AfterTaskIteration`, which no upper layer observes directly, and the persisted snapshot is the artefact they consume.

## Routing Table

- `./mod.rs` → crate docs, module declarations, `HarnessRail`, `rail_priority`, re-exports
- `./context.rs` → `HarnessRailContext`, `TaskIterationInfo`, `TaskIterationPhase`, prompt-section channel, `last_user_text`
- `./registry.rs` → `HarnessRailRegistry` (copy-on-write dispatch) and `HarnessRailAdapter` (kernel bridge)
- `./security.rs` → `SecurityRail` (guardrail scan of user input, prompts, model output, tool I/O)
- `./memory.rs` → `MemoryRail`, `MemoryRecall`/`MemoryWriteBack` seams, `LongTermMemoryAdapter`
- `./skills.rs` → `SkillsRail` (discovery cache, budget, skills prompt section)
- `./subagent.rs` → `SubagentRail` + `SubAgentSpawner` seam and request/status data
- `./evolution/` → the self-evolution rail seam and its harness-side bodies
- `./evolution/mod.rs` → `EvolutionRail` (five hooks), `NoopEvolutionRail`, `EvolutionRailAdapter`/`evolution_adapter`, module re-exports
- `./evolution/metrics.rs` → `EvolutionMetrics`, `IterationRecorder`, `MetricsClock`, `default_metrics_clock`
- `./evolution/proposal.rs` → `EvolutionProposal`, `ProposalLog` (bounded, atomic save/load)
- `./evolution/snapshot.rs` → `EvolutionSnapshot` (+ `EVOLUTION_SNAPSHOT_VERSION`), save/load
- `./evolution/rail.rs` → `MetricsEvolutionRail`, `MetricsEvolutionRailConfig`, run state and hook bodies
- `./evolution/store.rs` → shared atomic JSON writer/reader (`<path>.tmp` + rename)

## Test Strategy

- Unit tests live beside each module (`#[cfg(test)] mod tests`) and never touch the network or the real model: `ScriptedModel`/hand-built `BaseMessage`s and `tempfile` skill roots only.
- `context.rs`: builders, `extra_mut`, control-signal delegation, prompt-section publish/collect round-trip (category, carrier, priority, overwrite by name), `last_user_text`.
- `registry.rs`: ordering (priority desc, stable ties), replace-by-name, unregister, per-event dispatch of all nine kernel hooks, error aborts later rails, task-iteration hooks see `ctx.task` per phase, registration during `fire` is not observed by the running snapshot.
- `registry.rs` adapter: forwards the nine kernel events exactly once, copies every control flag back, and keeps flags even when the hook returns `Err`.
- `security.rs`: fake `Guardrail` (pass/block/fail) per stage — prompt/user/model-output/tool-output blocks return `Error::security`, tool-input block sets `skip_tool` and returns `Ok`, disabled scan targets skip the pipeline, a failing guardrail under `FailOpen` is swallowed.
- `memory.rs`: fake recaller/writer — injection position and `<memory-context>` wrapper, no memories → no message, recall/write-back errors swallowed, turn content handed to the writer, `LongTermMemoryAdapter` round-trip over an in-memory facade.
- `skills.rs`: temp roots with `SKILL.md` — refresh selects and publishes the section, budget/disabled filtering, stale-once-per-invocation refresh, missing roots degrade to an empty selection.
- `subagent.rs`: section publication, spawner accessor, a fake spawner exercises the whole seam.
- `evolution/mod.rs`: adapter forwards all five hooks exactly once, swallows an `Err` from every hook, leaves the other harness hooks as no-ops, and `NoopEvolutionRail` mutates nothing (`name`/`priority` asserted).
- `evolution/metrics.rs`: usage stays `None` unless reported, per-tool/failure counting, bounded eviction order, unbounded `capacity == 0`, `clear`, injectable clock, folded totals incl. an empty recorder.
- `evolution/proposal.rs`: uuid + stamping, insertion order, cap eviction, atomic save/load round-trip, bare-array documents, missing file → empty, save without path → `Error::config`, malformed JSON → `Error::validation`.
- `evolution/snapshot.rs`: version default, save/load round-trip, unknown-field tolerance, missing file → `Error::config`, malformed → `Error::validation`, unwritable destination → `Error::config`.
- `evolution/rail.rs`: one metrics row per iteration (counts, failures, per-tool map, tokens, stop reason), counter reset between iterations, unflushed counters still in the totals, proposal stamping/ordering/cap and a clean buffer per run, snapshot round-trip through a temp file, fresh `run_id` per run, proposal-log persistence, a failing write swallowed by the hook but visible on `write_snapshot()`, no path → no write.
- `evolution/store.rs`: round-trip, missing file, malformed document, unwritable destination, temporary-path suffix.
- Parity references: `tests/unit_tests/harness/test_deep_agent_rail_event_routing.py`, `rails/test_typed_interrupts.py`, `rails/test_base_security_rail.py`, `rails/evolution/test_evolution_rail.py`, `test_skill_rail.py`, `rails/test_memory_rail.py`.
- Definition of done: `cargo fmt -p openjiuwen-harness`, `cargo test -p openjiuwen-harness`, `cargo test -p openjiuwen-harness --no-default-features`, `cargo clippy -p openjiuwen-harness --all-targets -- -D warnings` clean.
