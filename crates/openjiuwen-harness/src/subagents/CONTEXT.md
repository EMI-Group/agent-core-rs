# subagents

## Intent

Sub-agent definitions, presets, an in-process runtime, and the model-facing task/subagent tools.
Ports `openjiuwen/harness/subagents/` (presets), `subagent_runtime/` (control/instance/registry subset), `subagent_lifecycle.py`, and `tools/subagent/{task_tool,subagent_tools}.py`.
`SubAgentConfig` is defined in `crate::schema` and re-exported here.
Everything is in-process and in-memory: no child processes, no network, no persisted runtime bucket.
A child agent is built lazily by a [`SubAgentFactory`] so this module never depends on `crate::agent`.

## API Surface

Re-exports (`mod.rs`): `SubAgentConfig` (from `crate::schema`), `SubAgentSpec`, `SubAgentRegistry`, `SubAgentPreset`, `build_preset`, `general_purpose_config`, `GENERAL_PURPOSE_NAME`, `SubAgentRuntime`, `SubAgentFactory`, `SubAgentId`, `SubAgentSpawnRequest`, `SubAgentOutcome`, `SubagentMetadata`, `SubagentStatus`, `ResumeReport`, `TaskTool`, and the six subagent tools plus `subagent_tools(runtime)`.

- `SubAgentSpec { config: SubAgentConfig, preset: Option<SubAgentPreset> }` with `new(config)`, `with_preset(config, preset)`, `name()` (`agent_card.name`).
- `SubAgentPreset` enum: `Code`, `Explore`, `Plan`, `Research`, `Verification`, plus feature-gated `Browser`, `Cua`, `Mobile`; `Default = Code`.
  `as_str()` returns the Python type name (`code_agent`, `explore_agent`, `plan_agent`, `research_agent`, `verification_agent`, `browser_agent`, `cua_agent`, `mobile_gui_agent`); `builtin()` returns the five always-available presets in Python order.
- `build_preset(preset: SubAgentPreset, language: &str) -> Result<SubAgentConfig>`; feature-gated presets return `Error::Unsupported`.
- `general_purpose_config() -> SubAgentConfig` — card name `general-purpose`, no system prompt, `restrict_to_work_dir = false`, `max_iterations = None`, CN description (Python default language), `subagent_type = Some("general-purpose")`.
- `SubAgentRegistry`: `new()`, `with_builtin_presets(language) -> Self`, `register(SubAgentSpec)` (replace by `agent_card.name`), `get(name) -> Option<SubAgentConfig>` (cloned), `get_spec(name) -> Option<SubAgentSpec>`, `contains(name)`, `list()`, `list_specs()`, `names()` (BTreeMap order = name order), `len()`, `is_empty()`, `general_purpose() -> SubAgentConfig` (registers on first use).
  The registry is `Clone` and every clone shares one `Arc<parking_lot::RwLock<BTreeMap<String, SubAgentSpec>>>`.
- `SubagentStatus` (`snake_case` serde): `PendingInit`, `Running`, `Completed`, `Interrupted`, `Errored`, `Closed`, `NotFound`; `is_final()` is true for `Completed`/`Errored`/`Closed`/`NotFound`; `as_str()` exposes the Python wire string (`pending_init`, `running`, `completed`, `interrupted`, `errored`, `closed`, `not_found`).
- `SubAgentId = String`; id format `<parent_session>_sub_<type>[_<8hex>]`, `<parent_session>_sub_<type>` (no hex) for the sticky type `verification_agent`; `sub_session_id == id`.
- `SubAgentSpawnRequest { subagent_type, task_description, display_name: Option<String>, role: Option<String>, subagent_id: Option<String> }` with `new(type, description)`.
- `SubAgentOutcome { subagent_id, sub_session_id, subagent_type, agent_id, status, output: Option<String>, error: Option<String> }` with `is_final()`.
- `SubagentMetadata { subagent_id, sub_session_id, subagent_type, display_name, role, task_description, status, created_at, updated_at }` (serde, RFC 3339 timestamps).
- `ResumeReport { status, restored, message: Option<String> }`.
- `SubAgentFactory` (`#[async_trait]`, `Send + Sync`): `async fn create(&self, config: &SubAgentConfig, session_id: &str) -> Result<Arc<dyn Agent>>`.
- `SubAgentRuntime`: `new(registry: Arc<SubAgentRegistry>, factory: Arc<dyn SubAgentFactory>, parent_session_id: impl Into<String>)`, `with_limits(max_subagents, max_concurrent_running)`, `with_turn_timeout(Duration)`, `spawn(SubAgentSpawnRequest) -> Result<SubAgentId>`, `wait(&[SubAgentId], timeout: Duration) -> Vec<SubAgentOutcome>`, `get_status(id) -> SubagentStatus`, `get_outcome(id) -> Option<SubAgentOutcome>`, `list() -> Vec<SubagentMetadata>` (live instances only), `capacity() -> (usize, usize)`, `send_input(id, query, interrupt: bool) -> Result<()>`, `resume(id) -> Result<ResumeReport>`, `close(id) -> Result<SubagentStatus>`, `cancel_all() -> Vec<SubAgentId>`, `describe_list() -> Value`, `parent_session_id()`, `registry() -> &Arc<SubAgentRegistry>`.
  Defaults: 10 live instances, 5 concurrent turns, 2400 s per turn; a turn over budget ends `Errored` with `subagent turn timed out after <n>s`.
- `TaskTool` (`foundation::Tool`, name `task_tool`): inputs `subagent_type` (required), `task_description` (required), `resume_task_id` (optional); output `{output, agent_id, subagent_type}`; `new(runtime)`, `with_timeout(Duration)`, `card_for(language)`, `timeout()`; default timeout 720 s, overridable via `OPENJIUWEN_SUBAGENT_TASK_TIMEOUT_S` (invalid/non-positive keeps 720 s).
- Six tools, each `new(runtime: Arc<SubAgentRuntime>)`, `with_language(&str)`, `card_for(language)`; `new` builds the card in `crate::prompts::DEFAULT_LANGUAGE` (`cn`):
  - `subagent_spawn`: `subagent_type`, `task_description`, `display_name`, `role` (all required) → `{subagent_id, sub_session_id, task_id, status}`.
  - `subagent_wait`: `subagent_ids` (required, non-empty array, singular `subagent_id` accepted as fallback), `timeout_ms` (integer, default 2_400_000, clamped to 10_000..=3_600_000) → `{statuses, results, output_files, timed_out}`.
  - `subagent_list` (no inputs) → `describe_list()` including `capacity`, `subagents`, `live_subagents`, `closed_subagents`, `summary`.
  - `subagent_send_input`: `subagent_id`, `query` (required), `interrupt` (bool, default false) → `{subagent_id, task_id, status}`.
  - `subagent_close`: `subagent_id` → `{subagent_id, previous_status}`.
  - `subagent_resume`: `subagent_id` → `{subagent_id, status, restored}` plus `message` when present.
  `subagent_tools(runtime) -> Vec<Arc<dyn Tool>>` returns spawn, wait, list, send_input, close, resume in that order.

## Constraints

- `send_input` on a closed/unknown instance returns `Error::validation("subagent closed or not found: <id>; call subagent_resume first")`; `close` on `Running` returns `Error::validation("cannot close running subagent: <id>")`; `close`/`resume` on an unknown id returns `Error::validation("subagent not found: <id>")`; `spawn` on a live duplicate returns `Error::validation("subagent already live: <id>; use subagent_wait to collect its result")`; an unknown type returns `Error::validation("subagent type not found: <t>")`.
- Capacity is `Error::config("subagent capacity exceeded: used=<used>, limit=<limit>")`; it is reserved synchronously (the `PendingInit` instance is inserted before the factory is awaited) and rolled back on any creation failure.
- `max_subagents` counts live instances; `max_concurrent_running` bounds turns through `tokio::sync::Semaphore`.
- Child turns run in background `tokio` tasks: `spawn` returns once the instance is registered, `wait` blocks until every id is final or the timeout elapses (input order preserved, duplicates collapsed, unknown ids map to `NotFound`).
- Presets set `max_iterations = 50`, `enable_task_loop = false`, `enable_plan_mode = false`, `parallel_tool_calls = true`; `explore`/`plan` set `restrict_to_work_dir = false`; `code`/`research` set `factory_name` (`code_agent`, `research_agent`).
- Feature-gated presets (`browser`, `cua`, `mobile`) exist only with the matching cargo feature and return `Error::Unsupported` from `build_preset`; the agents themselves are deferred.
- No dependency on `crate::agent`, `crate::rails`, or `crate::tools` internals beyond the shared `crate::tools` helpers (`tool_card`, `parse_args`, `ToolPayload`, `render_fields`, `descriptions::description`).
- Tools never panic and never return `Err` for expected failures: they return a `ToolPayload` JSON with `success=false`, `data=null` and the Python message in `error`.
- Locks are `parking_lot::Mutex`/`RwLock` and never held across `.await`; status waits use `tokio::sync::Notify`; interrupted turns use `JoinHandle::abort`.
- Files ≤ 1000 lines (tests live in `runtime/tests.rs` and `subagent_tools/tests.rs`); unit tests beside the modules; `subagent_tools.rs` uses one scoped `macro_rules!` to declare the six tools' shared `new`/`card_for`/`with_language` surface.

## Design decisions

- No `SubAgentSpawner` trait is defined here: `crate::rails` (parallel work) owns that seam and can `impl` it / wrap [`SubAgentRuntime`]. The inherent runtime methods above are the stable partner API.
- The workspace is not a runtime field: `SubAgentConfig` carries the subagent's own settings and the factory re-roots it at `sub-agents/<sub_session_id>` (Python `create_subagent` behavior).
- Python's `SubagentControl` persisted snapshot/activity/transcript streams and LRU eviction are deferred; the runtime keeps live instances only, with in-memory closed tombstones (`Closed`) so `resume`/`close` stay idempotent.
- Runtime internals: one `Arc<Shared>` (`registry`, `factory`, `parent_session_id`, `Notify`, `Mutex<RuntimeState>`) plus per-runtime `max_subagents`, `turn_timeout` and `Semaphore`. Each instance keeps a spawn-sequence generation so an aborted or closed turn can never resurrect a tombstoned instance.
- Queued `send_input` queries are FIFO; a turn that finds queued work transitions through `PendingInit` instead of exposing a final status, and the last turn applies the terminal status.
- Thread/task interrupt uses `JoinHandle::abort`, the only safe cancellation of a `dyn Agent::invoke` future in Rust; the interrupting caller sets `Interrupted` itself, because an aborted task runs no code.
- `TaskTool` awaits `spawn` + `wait` inline (Python sync delegation path) and reports `agent_id` from the child `Agent::card().id`.
- Task ids are a tool-layer concept: `TaskTool`/`subagent_*` payloads carry a fresh uuid hex `task_id` generated by the tool, because the Rust runtime models work as instance turns, not as a task queue.
- Tool descriptions resolve through `description_for` in `task_tool.rs`: `crate::tools::descriptions::description` first, then the `crate::prompts::description` table (which owns the `task_tool`/`subagent_*` rows today).
- `subagent_tools` is a directory module: the parent file keeps the constants, the `subagent_tool!` macro, `subagent_tools()` and the spawn/wait/list tools, while `control.rs` holds the send_input/close/resume tools and `util.rs` the helpers both share. `mod control;` must stay below the `subagent_tool!` definition, because a `macro_rules!` macro is only in scope for child modules declared after it.

## Known Issues

- `crate::tools::descriptions` has no rows for `task_tool` or the six `subagent_*` names, so their descriptions fall back to the `crate::prompts::tool_descriptions` table; adding the rows belongs to `crate::tools` (sibling module).

## Deferred Python areas

- `subagents/{browser,cua,mobile_gui}_agent.py` — feature-gated stubs returning `Error::Unsupported`.
- `subagent_runtime/{persistence,activity,transcript,eviction}.py` — no persisted buckets, activity/transcript streams, or LRU eviction.
- `subagent_lifecycle.py` resource hooks — the factory owns child resource setup/teardown.
- `browser_capabilities`, `resume_task_id` orchestration for browser/CUA, KV-affinity sub-session rewriting, and `TaskTool`'s streaming/observability path.

## Routing Table

- `./mod.rs` → re-exports, `SubAgentSpec`, `SubAgentRegistry`
- `./presets.rs` → preset matrix, bilingual prompts/descriptions, `build_preset`, `general_purpose_config`
- `./runtime.rs` → `SubAgentRuntime`, statuses, ids, outcomes, metadata, `ResumeReport`, `SubAgentFactory`
- `./runtime/tests.rs` → runtime unit tests (fake factory/agent)
- `./task_tool.rs` → `TaskTool`, timeout resolution, shared tool description lookup
- `./subagent_tools.rs` → module docs, constants, the `subagent_tool!` macro, `subagent_tools()`, and the spawn/wait/list tools; re-exports the control tools
- `./subagent_tools/control.rs` → `subagent_send_input`, `subagent_close`, `subagent_resume` tools
- `./subagent_tools/util.rs` → shared parsing, task-id and payload-rendering helpers (`pub(crate)`)
- `./subagent_tools/tests.rs` → unit tests for the six tools
- `./test_support.rs` → `#[cfg(test)]` fake agent/factory and runtime fixture shared by the tool tests

## Test Strategy

- Unit tests with fakes: `test_support.rs` provides an echo `FakeAgent`, an error-result variant, a `Notify`-gated blocked variant, and a `FakeFactory` recording `(config, session_id)`; runtime tests keep their own local fakes in `runtime/tests.rs`.
- `test_support` is `#[cfg(test)] mod` and every item is `pub(crate)`, so integration tests under `../../tests/` must define their own `Agent` + `SubAgentFactory` fakes.
- `TaskTool::new` reads `OPENJIUWEN_SUBAGENT_TASK_TIMEOUT_S` from the process environment at construction, so tests needing the 720 s default must not set that variable; use `.with_timeout(..)` instead.
- `subagent_wait` clamps `timeout_ms` to 10_000..=3_600_000, so a `timed_out: true` assertion waits at least 10 s; use `SubAgentRuntime::wait` with a short `Duration` for short-timeout tests.
- Registry: replace by name, builtin preset names, `general_purpose` insertion, shared map across clones, name ordering.
- Presets: max_iterations 50, `restrict_to_work_dir` per preset, `factory_name` for code/research, bilingual selection (`en` vs anything else → `cn`), verdict contract text in the verification prompt, feature-gated presets return `Error::Unsupported`.
- Runtime: spawn → PendingInit/Running → Completed with output; wait timeout leaves non-final; close refuses running; close/resume on completed instance; capacity rejection at `max_subagents`; rollback leaves capacity free after a factory failure; duplicate live spawn rejected; child session id format and sticky verification id; FIFO queued inputs; interrupt; `cancel_all`; `describe_list` shape.
- Tools: exact names, exact `required` lists and property keys; `task_tool` success payload shape, unknown-type failure, env-var timeout resolution via a pure helper; `subagent_wait` validation and clamp helper; every expected failure returns `success=false` (never `Err`).
- Current suite: 55 unit tests under `subagents` (crate total 1088 unit + 80 integration + 8 doc tests), green with default, `--no-default-features` and `--all-features`; `cargo clippy -p openjiuwen-harness --all-targets -- -D warnings` clean.
- Parity references: `tests/unit_tests/harness/test_plan_agent.py`, `test_explore_agent.py`, `test_deep_agent.py`, `subagent_runtime/test_control.py`, `tools/subagent/test_task_tool.py`.
