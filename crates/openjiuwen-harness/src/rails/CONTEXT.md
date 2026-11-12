# rails

## Intent

The harness hook layer: one rail trait covering the nine kernel lifecycle events plus the four outer task-loop events, a priority-ordered copy-on-write registry, and the built-in rails (security, memory, skills, subagent, evolution seam).
Ports `openjiuwen/harness/rails/` and the hook model of `openjiuwen/core/single_agent/rail/base.py` (`AgentRail`).
Rails never call the model and never execute tools: they mutate the context handed to them and return control signals.
The agent owns the firing points; this module owns the contract.

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
- `EvolutionRail` (priority `EVOLUTION` = 60) + `NoopEvolutionRail`: `on_before_invoke` / `on_after_task_iteration` plus `name()`/`priority()`. `EvolutionRailAdapter` (built by `evolution_adapter`) implements `HarnessRail` and is the only entry into the registry; it logs and swallows hook errors, so an evolution failure never aborts a run. No dependency on `openjiuwen-agent-evolving`.

## Constraints

- Rails mutate only the context; control signals are flags on `RailContext`; a rail `Err` aborts the run (the only exceptions are the documented fail-open rails: security infrastructure failures under `FailurePolicy::FailOpen`, memory seams, and evolution hooks).
- Registry snapshots are `Arc` swaps: `fire` clones the snapshot once, so `register`/`unregister` during a run is safe, lock-free while hooks execute, and takes effect on the next firing point.
- Never log prompt text, tool arguments/outputs, memory content, or secrets: log guardrail/pipeline names, counts and risk types only.
- Only this module imports `openjiuwen_security` / `openjiuwen_memory` kernel APIs, and it must not import `crate::tools`, `crate::agent`, or `crate::subagents`.
- No panics on user input, no `unsafe`, `tracing` only (no stdout), files ≤1000 lines, every public item documented.
- Per-invocation state lives in `ctx.extra` or in the context, never in rail fields (rails are shared across invocations).

## Design decisions

- The Python rail protocol is identity-based override detection over 13 `AgentCallbackEvent` hooks; Rust has no such introspection, so every `HarnessRail` hook is an explicit no-op default and the registry always fires all of them.
- Python's `_skip_tool`/`tool_result` convention and "model-event reject → `request_force_finish`" become: `skip_tool` for tool inputs, `Error::security` for prompt/user/model-output and tool-output blocks. Documented deviation: an aborting error is unambiguous in Rust and already means "abort the run".
- Live services arrive as seam traits (`MemoryRecall`, `MemoryWriteBack`, `SubAgentSpawner`, `EvolutionRail`) instead of builder-wired objects, so this module stays free of runtime dependencies and upper layers keep the wiring.
- The prompt-section channel is a JSON contract in `ctx.extra`, not a rail-owned builder handle: the agent renders sections when it assembles the prompt, and rails stay pure context mutators.

## Routing Table

- `./mod.rs` → crate docs, module declarations, `HarnessRail`, `rail_priority`, re-exports
- `./context.rs` → `HarnessRailContext`, `TaskIterationInfo`, `TaskIterationPhase`, prompt-section channel, `last_user_text`
- `./registry.rs` → `HarnessRailRegistry` (copy-on-write dispatch) and `HarnessRailAdapter` (kernel bridge)
- `./security.rs` → `SecurityRail` (guardrail scan of user input, prompts, model output, tool I/O)
- `./memory.rs` → `MemoryRail`, `MemoryRecall`/`MemoryWriteBack` seams, `LongTermMemoryAdapter`
- `./skills.rs` → `SkillsRail` (discovery cache, budget, skills prompt section)
- `./subagent.rs` → `SubagentRail` + `SubAgentSpawner` seam and request/status data
- `./evolution.rs` → `EvolutionRail`, `NoopEvolutionRail`, `EvolutionRailAdapter` (failure-isolating bridge)

## Test Strategy

- Unit tests live beside each module (`#[cfg(test)] mod tests`) and never touch the network or the real model: `ScriptedModel`/hand-built `BaseMessage`s and `tempfile` skill roots only.
- `context.rs`: builders, `extra_mut`, control-signal delegation, prompt-section publish/collect round-trip (category, carrier, priority, overwrite by name), `last_user_text`.
- `registry.rs`: ordering (priority desc, stable ties), replace-by-name, unregister, per-event dispatch of all nine kernel hooks, error aborts later rails, task-iteration hooks see `ctx.task` per phase, registration during `fire` is not observed by the running snapshot.
- `registry.rs` adapter: forwards the nine kernel events exactly once, copies every control flag back, and keeps flags even when the hook returns `Err`.
- `security.rs`: fake `Guardrail` (pass/block/fail) per stage — prompt/user/model-output/tool-output blocks return `Error::security`, tool-input block sets `skip_tool` and returns `Ok`, disabled scan targets skip the pipeline, a failing guardrail under `FailOpen` is swallowed.
- `memory.rs`: fake recaller/writer — injection position and `<memory-context>` wrapper, no memories → no message, recall/write-back errors swallowed, turn content handed to the writer, `LongTermMemoryAdapter` round-trip over an in-memory facade.
- `skills.rs`: temp roots with `SKILL.md` — refresh selects and publishes the section, budget/disabled filtering, stale-once-per-invocation refresh, missing roots degrade to an empty selection.
- `subagent.rs`: section publication, spawner accessor, a fake spawner exercises the whole seam.
- `evolution.rs`: adapter forwards `before_invoke`/`after_task_iteration`, swallows failures, `NoopEvolutionRail` name/priority.
- Parity references: `tests/unit_tests/harness/test_deep_agent_rail_event_routing.py`, `rails/test_typed_interrupts.py`, `rails/test_base_security_rail.py`, `rails/evolution/test_evolution_rail.py`, `test_skill_rail.py`, `rails/test_memory_rail.py`.
- Definition of done: `cargo fmt -p openjiuwen-harness`, `cargo test -p openjiuwen-harness`, `cargo test -p openjiuwen-harness --no-default-features`, `cargo clippy -p openjiuwen-harness --all-targets -- -D warnings` clean.
