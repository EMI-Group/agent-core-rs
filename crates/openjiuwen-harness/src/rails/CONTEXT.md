# rails

## Intent

The harness hook layer: one rail trait covering the kernel lifecycle events plus the outer task-loop hooks, a priority-ordered registry, and the built-in rails (security, memory, skills, subagent, evolution seam).
Ports `openjiuwen/harness/rails/` and bridges `openjiuwen/core/single_agent/rail/base.py` (`AgentRail`).

## API Surface

- `HarnessRail` (`#[async_trait]`, `Send + Sync`): `name()`, `priority()` (default 50, descending order, stable ties), and no-op hooks `before_invoke`, `after_invoke`, `on_user_message`, `before_steering_drain`, `before_model_call`, `after_model_call`, `on_model_exception`, `before_tool_call`, `after_tool_call`, `on_tool_exception`, `after_react_iteration`, `before_task_iteration`, `after_task_iteration`.
- `HarnessRailContext { base: RailContext, task: Option<TaskIterationInfo> }` — `base` is the kernel context (messages, tools, tool call/result, output, `extra`, control flags); `task` carries `{round_id, iteration, is_follow_up, loop_event}`. Builders: `for_event`, `with_task`, `extra_mut`, `request_skip_tool`, `request_interrupt`.
- `HarnessRailRegistry`: `register` (replace by name, re-sort), `unregister`, `rails()`, `fire(&mut HarnessRailContext)`, `fire_task_iteration(...)`, `fire_user_message(...)`, `to_agent_rails() -> Vec<Arc<dyn AgentRail>>` (bridged).
- `HarnessRailAdapter` — implements `openjiuwen_agent::AgentRail` by forwarding the shared subset into a `HarnessRail` and copying control flags back.
- `SecurityRail` (priority 85): scans prompts/tool inputs/tool outputs with `openjiuwen_security::GuardrailPipeline`; a blocking outcome rejects the invocation/tool with `Error::security` (or `skip_tool` for tool calls).
- `MemoryRail` (priority 80): recall relevant memories before the model call (inject into `messages`/`extra`) and write back the turn's facts after the invocation via `openjiuwen_memory`.
- `SkillsRail` (priority 95): discover/select skills, expose the skills prompt section, and surface skill tools; honours `SkillBudget`.
- `SubagentRail` (priority 95): inject the subagent prompt section and expose subagent tools through a `SubAgentSpawner` seam.
- `EvolutionRail` (priority 60): trait seam for `agent_evolving`; `NoopEvolutionRail` default; no dependency on `openjiuwen-agent-evolving`.
- `rail_priority` constants mirroring the Python ladder.

## Constraints

- Rail return values are control signals through `RailContext` (`skip_tool`, `force_finish`, `request_retry`, `model_continue`, interrupts); a rail `Err` aborts the run.
- Registry order is descending priority with stable insertion for ties; registration while a run holds a snapshot must not block (copy-on-write snapshot semantics).
- Rails never call the model or execute tools directly; they only mutate the context.
- `SecurityRail` and `MemoryRail` are fail-open on infrastructure errors (log + continue) and fail-closed on explicit guardrail blocks.
- Secrets and raw memory content are never logged.
- Only this module may import `openjiuwen_security`/`openjiuwen_memory` kernel APIs.

## Routing Table

- `./mod.rs` → `HarnessRail`, `HarnessRailContext`, `HarnessRailRegistry`, adapter, priorities
- `./security.rs` → `SecurityRail` (guardrail scan of prompts/tool I/O)
- `./memory.rs` → `MemoryRail` (recall + write-back)
- `./skills.rs` → `SkillsRail` (skill discovery/section/tools)
- `./subagent.rs` → `SubagentRail` + `SubAgentSpawner` seam
- `./evolution.rs` → `EvolutionRail` trait + no-op default

## Test Strategy

- Unit tests: registry ordering (priority desc, stable ties), replace-by-name, adapter forwards the nine kernel events exactly once, task hooks fire only on loop rounds, security rail blocks/skips, memory rail injects and writes back behind a fake store, evolution rail failure isolation.
- Parity references: `tests/unit_tests/harness/test_deep_agent_rail_event_routing.py`, `rails/test_typed_interrupts.py`, `rails/test_base_security_rail.py`, `rails/evolution/test_evolution_rail.py`.
