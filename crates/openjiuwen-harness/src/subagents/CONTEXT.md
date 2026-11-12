# subagents

## Intent

Subagent definitions, presets, an in-process runtime, and the model-facing task/subagent tools.
Ports `openjiuwen/harness/subagents/` (presets), `subagent_runtime/` (control/instance/registry subset), `subagent_lifecycle.py`, and `tools/subagent/{task_tool,subagent_tools}.py`.
`SubAgentConfig` is defined in `crate::schema` and re-exported here.
Everything is in-process and in-memory: no child processes, no network, no persisted runtime bucket.
A child agent is built lazily by a [`SubAgentFactory`] so this module never depends on `crate::agent`.

## API Surface

Re-exports (`mod.rs`): `SubAgentConfig` (from `crate::schema`), `SubAgentSpec`, `SubAgentRegistry`, `SubAgentPreset`, `build_preset`, `general_purpose_config`, `SubAgentRuntime`, `SubAgentFactory`, `SubAgentId`, `SubAgentSpawnRequest`, `SubAgentOutcome`, `SubagentMetadata`, `SubagentStatus`, `TaskTool`, and the six subagent tools.

- `SubAgentSpec { config: SubAgentConfig, preset: Option<SubAgentPreset> }`.
- `SubAgentPreset` enum: `Code`, `Explore`, `Plan`, `Research`, `Verification`, plus feature-gated `Browser`, `Cua`, `Mobile`.
  `as_str()` returns the Python type name (`code_agent`, `explore_agent`, `plan_agent`, `research_agent`, `verification_agent`, `browser_agent`, `cua_agent`, `mobile_gui_agent`).
- `build_preset(preset: SubAgentPreset, language: &str) -> Result<SubAgentConfig>`; feature-gated presets return `Error::Unsupported`.
- `general_purpose_config() -> SubAgentConfig` — card name `general-purpose`, empty prompt, `restrict_to_work_dir = false`, CN description (Python default language).
- `SubAgentRegistry`: `new()`, `with_builtin_presets(language) -> Self`, `register(SubAgentSpec)` (replace by `agent_card.name`), `get(name) -> Option<&SubAgentConfig>`, `list() -> Vec<SubAgentConfig>`, `names() -> Vec<String>`, `general_purpose() -> SubAgentConfig` (registers on first use).
- `SubagentStatus` (`snake_case` serde): `PendingInit`, `Running`, `Completed`, `Interrupted`, `Errored`, `Closed`, `NotFound`; `is_final()` is true unless `PendingInit`/`Running`/`Interrupted`. `as_str()` exposes the Python wire string.
- `SubAgentId = String`; id format `<parent_session>_sub_<type>[_<8hex>]`, `<parent_session>_sub_<type>` (no hex) for the sticky type `verification_agent`.
- `SubAgentSpawnRequest { subagent_type, task_description, display_name: Option<String>, role: Option<String>, subagent_id: Option<String> }`.
- `SubAgentOutcome { subagent_id, sub_session_id, subagent_type, agent_id, status, output: Option<String>, error: Option<String> }` with `is_final()`.
- `SubagentMetadata { subagent_id, sub_session_id, subagent_type, display_name, role, task_description, status, created_at, updated_at }` (serde, RFC 3339 timestamps).
- `SubAgentFactory` (`#[async_trait]`, `Send + Sync`): `async fn create(&self, config: &SubAgentConfig, session_id: &str) -> Result<Arc<dyn Agent>>`.
- `SubAgentRuntime`: `new(registry: Arc<SubAgentRegistry>, factory: Arc<dyn SubAgentFactory>, parent_session_id: impl Into<String>)`, `with_limits(max_subagents, max_concurrent_running)`, `with_turn_timeout(Duration)`, `spawn(SubAgentSpawnRequest) -> Result<SubAgentId>`, `wait(ids: &[SubAgentId], timeout: Duration) -> Vec<SubAgentOutcome>`, `get_status(id) -> SubagentStatus`, `get_outcome(id) -> Option<SubAgentOutcome>`, `list() -> Vec<SubagentMetadata>`, `capacity() -> (usize, usize)` (`used`, `max`), `send_input(id, query, interrupt: bool) -> Result<()>`, `resume(id) -> Result<ResumeReport>`, `close(id) -> Result<SubagentStatus>`, `cancel_all() -> Vec<SubAgentId>`, `describe_list() -> Value` (Python `describe_list` shape).
- `TaskTool` (`foundation::Tool`, name `task_tool`): inputs `subagent_type` (required), `task_description` (required), `resume_task_id` (optional); output `{output, agent_id, subagent_type}`; default timeout 720 s (overridable via `OPENJIUWEN_SUBAGENT_TASK_TIMEOUT_S`).
- `SubagentTools` (`foundation::Tool` set over one runtime):
  - `subagent_spawn`: `subagent_type`, `task_description`, `display_name`, `role` (all required) → `{subagent_id, sub_session_id, task_id, status}`.
  - `subagent_wait`: `subagent_ids` (required, non-empty array), `timeout_ms` (default 2_400_000, clamped to 10_000..=3_600_000) → `{statuses, results, output_files, timed_out}`.
  - `subagent_list` (no inputs) → `describe_list()` including `capacity`, `subagents`, `summary`.
  - `subagent_send_input`: `subagent_id`, `query` (required), `interrupt` (bool, default false) → `{subagent_id, task_id, status}`.
  - `subagent_close`: `subagent_id` → `{subagent_id, previous_status}`.
  - `subagent_resume`: `subagent_id` → `{subagent_id, status, restored}`.

## Constraints

- `send_input` on a closed/unknown instance returns `Error::validation("subagent closed or not found: <id>; call subagent_resume first")`; `close` on `Running` returns `Error::validation("cannot close running subagent: <id>")`; `spawn` on a live duplicate returns `Error::validation("subagent already live: <id>; use subagent_wait to collect its result")`; unknown type returns `Error::validation`.
- Capacity is reserved synchronously before the child is created and rolled back on any creation/spawn failure; `max_subagents` counts live instances, `max_concurrent_running` bounds turns via `tokio::sync::Semaphore`.
- Child turns run in background `tokio` tasks: `spawn` returns once the instance is registered, `wait` blocks until every id is final or the timeout elapses (input order preserved, unknown ids map to `NotFound`).
- Presets set `max_iterations = 50`, `enable_task_loop = false`, `enable_plan_mode = false`, `parallel_tool_calls = true`; `explore`/`plan` set `restrict_to_work_dir = false`; `code`/`research` set `factory_name` (`code_agent`, `research_agent`).
- Feature-gated presets (`browser`, `cua`, `mobile`) exist only with the matching cargo feature and return `Error::Unsupported` from `build_preset`; the agents themselves are deferred.
- No dependency on `crate::agent`, `crate::rails`, `crate::tools` internals beyond the shared `crate::tools::support` helpers (`tool_card`, `schema_for`, `parse_args`, `ToolPayload`).
- Tools never panic and never return `Err` for expected failures: they return a `ToolPayload` JSON with `success=false` and a Python-derived message.
- Locks are `parking_lot::Mutex` and never held across `.await`; status waits use `tokio::sync::Notify`.
- Files ≤ 1000 lines; unit tests in a `#[cfg(test)] mod tests` (or `tests.rs`) beside the modules using a `FakeFactory`.

## Design decisions

- No `SubAgentSpawner` trait is defined here: `crate::rails` (parallel work) owns that seam and can `impl` it / wrap [`SubAgentRuntime`]. The inherent runtime methods above are the stable partner API.
- The workspace is not a runtime field: `SubAgentConfig` carries the subagent's own settings and the factory re-roots it at `sub-agents/<sub_session_id>` (Python `create_subagent` behavior).
- Python's `SubagentControl` persisted snapshot/activity/transcript streams and LRU eviction are deferred; the runtime keeps live instances only, with in-memory closed tombstones (`Closed`) so `resume`/`close` stay idempotent.
- Thread/task interrupt uses `JoinHandle::abort`, the only safe cancellation of a `dyn Agent::invoke` future in Rust; the aborted turn yields `Interrupted` before the queued query starts.
- `TaskTool` awaits `spawn` + `wait` inline (Python sync delegation path) and reports `agent_id` from `Agent::card().id`.

## Deferred Python areas

- `subagents/{browser,cua,mobile_gui}_agent.py` — feature-gated stubs returning `Error::Unsupported`.
- `subagent_runtime/{persistence,activity,transcript,eviction}.py` — no persisted buckets, activity/transcript streams, or LRU eviction.
- `subagent_lifecycle.py` resource hooks — the factory owns child resource setup/teardown.
- `browser_capabilities`, `resume_task_id` orchestration for browser/CUA, KV-affinity sub-session rewriting.

## Routing Table

- `./mod.rs` → re-exports, `SubAgentSpec`, `SubAgentRegistry`
- `./presets.rs` → preset matrix, bilingual prompts/descriptions, `build_preset`, `general_purpose_config`
- `./runtime.rs` → `SubAgentRuntime`, statuses, ids, outcomes, `SubAgentFactory`
- `./task_tool.rs` → `TaskTool`
- `./subagent_tools.rs` → spawn/wait/list/send_input/close/resume tools

## Test Strategy

- Unit tests with a `FakeFactory` (returns a fake `Agent` answering with its session id and recording the config) and `#[tokio::test]`.
- Registry: replace by name, builtin preset names, `general_purpose` insertion.
- Presets: max_iterations 50, `restrict_to_work_dir` per preset, `factory_name` for code/research, verdict contract text in the verification prompt, feature-gated presets return `Error::Unsupported`.
- Runtime: spawn → PendingInit/Running → Completed with output; wait timeout leaves non-final; close refuses running; close/resume on completed instance; capacity rejection at `max_subagents`; rollback leaves capacity free after a factory failure; duplicate live spawn rejected; child session id format and sticky verification id.
- Tools: exact names and JSON schema keys/required lists; `task_tool` success payload shape and unknown-type failure; `subagent_wait` timeout validation; every tool returns `success=false` (never `Err`) for expected failures.
- Parity references: `tests/unit_tests/harness/test_plan_agent.py`, `test_explore_agent.py`, `test_deep_agent.py`, `subagent_runtime/test_control.py`, `tools/subagent/test_task_tool.py`.
