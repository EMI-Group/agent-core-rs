# subagents

## Intent

Subagent definitions, presets, an in-process runtime, and the model-facing task/subagent tools.
Ports `openjiuwen/harness/subagents/` (presets), `subagent_runtime/` (control/instance/registry subset), `subagent_lifecycle.py`, and `tools/subagent/{task_tool,subagent_tools}.py`.
`SubAgentConfig` is defined in `crate::schema` and re-exported here.

## API Surface

- `SubAgentConfig` (re-export) + `SubAgentSpec { config: SubAgentConfig, preset: SubAgentPreset }`.
- `SubAgentPreset` enum: `Code`, `Explore`, `Plan`, `Research`, `Verification`, plus feature-gated `Browser`, `Cua`, `Mobile` stubs returning `Error::Unsupported`.
- `SubAgentRegistry`: `register(spec)` (replace by `agent_card.name`), `get(name)`, `list()`, `presets()` (built-in code/explore/plan/research/verification), `general_purpose()`.
- `SubAgentRuntime` (`SubAgentSpawner` impl): `new(registry, parent_session, workspace, max_subagents=10, max_concurrent_running=5)`, `spawn(SubAgentSpawnRequest) -> SubAgentId`, `wait(ids, timeout) -> Vec<SubAgentOutcome>`, `list()`, `send_input(id, query, interrupt)`, `resume(id)`, `close(id)`, `cancel_all()`.
- `SubagentStatus { PendingInit, Running, Completed, Interrupted, Errored, Closed, NotFound }` and status transitions; `SubagentInstance` view with `turns`, `output`.
- `TaskTool` (`foundation::Tool`, name `task_tool`): inputs `subagent_type`, `task_description`, `resume_task_id`; output `{output, agent_id, subagent_type}`; default timeout 720 s.
- `SubagentTools`: `subagent_spawn`, `subagent_wait`, `subagent_list`, `subagent_send_input`, `subagent_close`, `subagent_resume` with Python schemas.
- Child session id `<parent>_sub_<type>[_<8hex>]`; workspace re-rooted at `sub-agents/<subsession_id>`.

## Constraints

- Spawning is in-process: child `DeepAgent`s are created lazily through a `SubAgentFactory` trait (`create(config) -> Arc<dyn Agent>`), so no dependency on `crate::agent`.
- Capacity is reserved before creation (rollback on failure); concurrency capped by a semaphore; status is queryable and emits serializable events.
- Subagent errors are contained: a failed subagent yields an `Errored` outcome, never panics the parent.
- Preset prompts are inline bilingual strings matching Python; presets set max_iterations 50 (verification/research/code/explore/plan).
- Deferred: browser/CUA/mobile presets (feature-gated stubs), LRU eviction/activity/transcript streams.

## Routing Table

- `./mod.rs` → re-exports, `SubAgentSpec`, registry
- `./presets.rs` → preset configs and prompts (code/explore/plan/research/verification + stubs)
- `./runtime.rs` → `SubAgentRuntime`, instances, status, events, spawner trait
- `./task_tool.rs` → `TaskTool`
- `./subagent_tools.rs` → spawn/wait/list/send_input/close/resume tools

## Test Strategy

- Unit tests: registry replace/list, preset defaults (max_iterations, restrict_to_work_dir), spawn returns a runnable id through a fake factory, wait timeout, close/resume transitions, capacity rollback, `task_tool` schema and error payload for an unknown subagent.
- Parity references: `tests/unit_tests/harness/test_plan_agent.py`, `test_explore_agent.py`, `test_deep_agent.py` (research preset + dispatch), `subagent_runtime/test_control.py`, `tools/subagent/test_task_tool.py`.
