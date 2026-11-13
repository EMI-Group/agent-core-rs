# subagents

## Intent

Sub-agent definitions, presets, an in-process runtime, and the model-facing task/subagent tools.
Ports `openjiuwen/harness/subagents/` (presets), `subagent_runtime/` (control/instance/registry subset), `subagent_lifecycle.py`, and `tools/subagent/{task_tool,subagent_tools}.py`.
`SubAgentConfig` is defined in `crate::schema` and re-exported here.
Everything is in-process and in-memory: no child processes, no network, no persisted runtime bucket.
A child agent is built lazily by a [`SubAgentFactory`] so this module never depends on `crate::agent`.

## API Surface

Re-exports (`mod.rs`): `SubAgentConfig` (from `crate::schema`), `SubAgentSpec`, `SubAgentRegistry`, `SubAgentPreset`, `build_preset`, `general_purpose_config`, `GENERAL_PURPOSE_NAME`, `SubAgentRuntime`, `SubAgentFactory`, `SubAgentId`, `SubAgentSpawnRequest`, `SubAgentOutcome`, `SubagentMetadata`, `SubagentStatus`, `ResumeReport`, `TaskTool`, the six subagent tools plus `subagent_tools(runtime)`.
Runtime support re-exports (`mod.rs`): `runtime::{clock, quota, transcript}` modules, `Clock`, `SystemClock`, `ManualClock`, `rfc3339_from_ms`, `SubAgentQuota`, `SpawnBudget`, `SubagentActivity`, `LruCandidate`, `select_lru_candidate`, and the transcript types.

- `SubAgentSpec { config: SubAgentConfig, preset: Option<SubAgentPreset> }` with `new(config)`, `with_preset(config, preset)`, `name()` (`agent_card.name`).
- `SubAgentPreset` enum: `Code`, `Explore`, `Plan`, `Research`, `Verification`, plus feature-gated `Browser`, `Cua`, `Mobile`; `Default = Code`.
  `as_str()` returns the Python type name (`code_agent`, `explore_agent`, `plan_agent`, `research_agent`, `verification_agent`, `browser_agent`, `cua_agent`, `mobile_gui_agent`); `builtin()` returns the five always-available presets in Python order.
- `build_preset(preset: SubAgentPreset, language: &str) -> Result<SubAgentConfig>`; feature-gated presets return `Error::Unsupported`.
- `general_purpose_config() -> SubAgentConfig` — card name `general-purpose`, no system prompt, `restrict_to_work_dir = false`, `max_iterations = None`, CN description (Python default language), `subagent_type = Some("general-purpose")`.
- `SubAgentRegistry`: `new()`, `with_builtin_presets(language) -> Self`, `register(SubAgentSpec)` (replace by `agent_card.name`), `get(name) -> Option<SubAgentConfig>` (cloned), `get_spec(name) -> Option<SubAgentSpec>`, `contains(name)`, `list()`, `list_specs()`, `names()` (BTreeMap order = name order), `len()`, `is_empty()`, `general_purpose() -> SubAgentConfig` (registers on first use).
  The registry is `Clone` and every clone shares one `Arc<parking_lot::RwLock<BTreeMap<String, SubAgentSpec>>>`.
- `SubagentStatus` (`snake_case` serde): `PendingInit`, `Running`, `Completed`, `Interrupted`, `Errored`, `Closed`, `NotFound`; `is_final()` is true for `Completed`/`Errored`/`Closed`/`NotFound`; `as_str()` exposes the Python wire string (`pending_init`, `running`, `completed`, `interrupted`, `errored`, `closed`, `not_found`).
- `SubAgentId = String`; id format `<parent_session>_sub_<type>[_<8hex>]`, `<parent_session>_sub_<type>` (no hex) for the sticky type `verification_agent`; `sub_session_id == id`.
- `SubAgentSpawnRequest { subagent_type, task_description, display_name: Option<String>, role: Option<String>, subagent_id: Option<String>, parent_subagent_id: Option<String> }` with `new(type, description)`; `parent_subagent_id` names the live instance a nested spawn belongs to (`None` = top-level).
- `SubAgentOutcome { subagent_id, sub_session_id, subagent_type, agent_id, status, output: Option<String>, error: Option<String> }` with `is_final()`.
- `SubagentMetadata { subagent_id, sub_session_id, subagent_type, display_name, role, task_description, status, created_at, updated_at, activity, depth, parent_subagent_id, last_used_at, closed_reason }` (serde, snake_case, RFC 3339 timestamps; the five trailing fields are `#[serde(default)]`).
- `ResumeReport { status, restored, message: Option<String> }`.
- `SubAgentFactory` (`#[async_trait]`, `Send + Sync`): `async fn create(&self, config: &SubAgentConfig, session_id: &str) -> Result<Arc<dyn Agent>>`.
- `SubAgentRuntime`: `new(registry: Arc<SubAgentRegistry>, factory: Arc<dyn SubAgentFactory>, parent_session_id: impl Into<String>)`, `with_quota(SubAgentQuota)`, `with_clock(Arc<dyn Clock>)`, `with_limits(max_subagents, max_concurrent_running)`, `with_turn_timeout(Duration)`, `quota() -> &SubAgentQuota`, `spawn(SubAgentSpawnRequest) -> Result<SubAgentId>`, `wait(&[SubAgentId], timeout: Duration) -> Vec<SubAgentOutcome>`, `get_status(id) -> SubagentStatus`, `get_outcome(id) -> Option<SubAgentOutcome>`, `get_metadata(id) -> Option<SubagentMetadata>` (closed tombstones included), `list() -> Vec<SubagentMetadata>` (live instances only), `capacity() -> (usize, usize)`, `spawned_total() -> usize`, `send_input(id, query, interrupt: bool) -> Result<()>`, `resume(id) -> Result<ResumeReport>`, `close(id) -> Result<SubagentStatus>`, `cancel_all() -> Vec<SubAgentId>`, `evict(id) -> Result<SubagentStatus>`, `evict_lru() -> Result<Option<SubAgentId>>`, `eviction_candidates() -> Vec<SubAgentId>` (evictable ids in LRU order), `describe_list() -> Value`, `parent_session_id()`, `registry() -> &Arc<SubAgentRegistry>`.
  Defaults: `SubAgentQuota::default()` (10 live instances, 5 concurrent turns, no cumulative/per-parent/depth cap, LRU eviction policy enabled), `SystemClock`, 2400 s per turn; a turn over budget ends `Errored` with `subagent turn timed out after <n>s`.
  `with_limits` is the pre-eviction convenience constructor: it sets both caps and forces `enable_lru_eviction = false`, so the live cap raises `Error::Config` instead of recycling an idle instance.
- `SubAgentQuota { max_subagents, max_concurrent_running, max_total_spawned: Option<usize>, max_children_per_parent: Option<usize>, max_depth: Option<usize>, enable_lru_eviction, transcript_capacity, transcript_channel_capacity }` (`Copy`, serde `snake_case` with `default`): `Default`, `#[must_use]` fluent `with_*` setters, `check_spawn(&SpawnBudget, parent: &str) -> Result<()>`.
- `SpawnBudget { live, spawned, parent_live, depth }` is the spawn demand the runtime counts; `SubagentActivity` phases (`Queued`, `Running`, `Idle`, `Closed`) expose `as_str()`/`is_evictable()` (true for `Idle`/`Queued`).
- `LruCandidate { subagent_id, activity, last_activity_ms, spawn_seq }` and `select_lru_candidate(&[LruCandidate]) -> Option<&LruCandidate>` (skips `Running`/`Closed`, prefers `Idle` over `Queued`, then oldest `last_activity_ms`, then smallest id); `SubAgentRuntime::{evict, evict_lru, eviction_candidates}` are the wired eviction surface and `EVICTION_REASON` (`"evicted"`) is the recorded `closed_reason`.
- Clock: `Clock { now_ms(), now_rfc3339() }` with `SystemClock` and `ManualClock::new/advance/set`; every runtime timestamp is read through it.
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
- Capacity is `Error::config("subagent capacity exceeded: used=<used>, limit=<limit>")`; it is reserved synchronously (the `Queued` instance is inserted before the factory is awaited) and rolled back on any creation failure (the rollback also decrements the cumulative spawn ledger).
- Quota refusals are `Error::Config` and checked in a fixed order — depth (`subagent depth limit exceeded: depth=<d>, limit=<l>`), per-parent (`subagent child limit exceeded: parent=<p>, used=<u>, limit=<l>`), cumulative (`subagent spawn budget exhausted: spawned=<s>, limit=<l>`), then live (`subagent capacity exceeded: used=<u>, limit=<l>`).
- A nested spawn (`parent_subagent_id: Some(p)`) inherits `depth(p) + 1` and returns `Error::validation("subagent parent not found: <p>")` when `p` is unknown or `Closed`; top-level instances have `depth = 1` and are counted in the root bucket labelled by the parent session id.
- `state.spawned` is the cumulative ledger (`spawned_total()`): closing instances does not lower it; only a factory-failure rollback does.
- `max_subagents` counts live instances; `max_concurrent_running` bounds turns through `tokio::sync::Semaphore`.
- Activity transitions are the only writers of the activity phase: reservation -> `Queued`, turn start -> `Running`, last turn of a worker -> `Idle` (whatever its final status), `close`/`cancel_all` -> `Closed`, `resume` -> `Idle`; each also stamps `updated_at`/`last_activity_at_ms`/`last_used_at_ms` from the clock.
- `closed_reason` is `manual` for `close`, `parent_ended` for `cancel_all` and `evicted` (`EVICTION_REASON`) for every eviction (Python names), cleared by `resume`.
- Eviction: `reserve` closes LRU evictable instances (`evict::make_room`) before the quota check while `live >= max_subagents && enable_lru_eviction`; the requesting parent is never recycled; a `Running` instance is never evicted, so a runtime saturated with running turns keeps the legacy `subagent capacity exceeded` error. `evict(id)` errors are `subagent not found: <id>`, `cannot evict running subagent: <id>` and `subagent already closed: <id>`; eviction happens synchronously under the state lock and notifies waiters after it is released.
- `describe_list().summary` carries `live_count`, `closed_count`, `evicted_count` and `eviction` (`{"enabled": bool, "candidates": <evictable live instances>}`).
- Child turns run in background `tokio` tasks: `spawn` returns once the instance is registered, `wait` blocks until every id is final or the timeout elapses (input order preserved, duplicates collapsed, unknown ids map to `NotFound`).
- Presets set `max_iterations = 50`, `enable_task_loop = false`, `enable_plan_mode = false`, `parallel_tool_calls = true`; `explore`/`plan` set `restrict_to_work_dir = false`; `code`/`research` set `factory_name` (`code_agent`, `research_agent`).
- Feature-gated presets (`browser`, `cua`, `mobile`) exist only with the matching cargo feature and return `Error::Unsupported` from `build_preset`; the agents themselves are deferred.
- No dependency on `crate::agent`, `crate::rails`, or `crate::tools` internals beyond the shared `crate::tools` helpers (`tool_card`, `parse_args`, `ToolPayload`, `render_fields`, `descriptions::description`).
- Tools never panic and never return `Err` for expected failures: they return a `ToolPayload` JSON with `success=false`, `data=null` and the Python message in `error`.
- Locks are `parking_lot::Mutex`/`RwLock` and never held across `.await`; status waits use `tokio::sync::Notify`; interrupted turns use `JoinHandle::abort`.
- Files ≤ 1000 lines (tests live in `runtime/tests.rs`, `runtime/tests/*.rs`, `runtime/quota.rs` and `subagent_tools/tests.rs`); unit tests beside the modules; `subagent_tools.rs` uses one scoped `macro_rules!` to declare the six tools' shared `new`/`card_for`/`with_language` surface.

## Design decisions

- No `SubAgentSpawner` trait is defined here: `crate::rails` (parallel work) owns that seam and can `impl` it / wrap [`SubAgentRuntime`]. The inherent runtime methods above are the stable partner API.
- The workspace is not a runtime field: `SubAgentConfig` carries the subagent's own settings and the factory re-roots it at `sub-agents/<sub_session_id>` (Python `create_subagent` behavior).
- Python's `SubagentControl` persisted snapshots and the activity/transcript *streams* are deferred; the runtime keeps live instances only, with in-memory closed tombstones (`Closed`) so `resume`/`close` stay idempotent.
- LRU eviction is wired, not scheduled: `evict::make_room` runs inside `reserve` (Python `SubagentControl._acquire_slot`) and evicts one idle instance per iteration, and `evict`/`evict_lru` are explicit caller-driven operations. No background eviction pass exists; `candidates` sorts with `quota::compare_lru`, the same order `select_lru_candidate` picks from, so the reported list and the pick agree.
- Runtime internals: one `Arc<Shared>` (`registry`, `factory`, `parent_session_id`, `Notify`, `Mutex<RuntimeState>`) plus per-runtime `quota`, `clock`, `turn_timeout` and `Semaphore`. Each instance keeps a spawn-sequence generation so an aborted or closed turn can never resurrect a tombstoned instance.
- `RuntimeState.spawned` is the quota ledger behind `spawned_total()`; the runtime reserves a `Queued` instance, the `TurnRunner` marks it `Running`/`Idle`, and `close`/`cancel_all` mark it `Closed`.
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
- `subagent_runtime/{persistence,activity,transcript,eviction}.py` — no persisted buckets and no activity/transcript streams; the eviction *policy* is ported (LRU pick plus the reserve-time `_acquire_slot` pass), the Python background eviction scheduler is not.
- `config.py` fields without a Rust counterpart: `turn_timeout_s` lives separately as `SubAgentRuntime::with_turn_timeout`, and the activity-stream knobs (`enable_activity_stream`, `activity_queue_size`, `activity_text_max_len`, `activity_throttle_ms`, `enable_transcript_stream`) are covered by `transcript_capacity`/`transcript_channel_capacity` only.
- `subagent_lifecycle.py` resource hooks — the factory owns child resource setup/teardown.
- `browser_capabilities`, `resume_task_id` orchestration for browser/CUA, KV-affinity sub-session rewriting, and `TaskTool`'s streaming/observability path.

## Routing Table

- `./mod.rs` → re-exports, `SubAgentSpec`, `SubAgentRegistry`
- `./presets.rs` → preset matrix, bilingual prompts/descriptions, `build_preset`, `general_purpose_config`
- `./runtime.rs` → `SubAgentRuntime`, statuses, ids, outcomes, metadata, `ResumeReport`, `SubAgentFactory`, quota-aware reservation with the reserve-time eviction pass, `describe_list`
- `./runtime/instance.rs` → `Instance` state (identity, depth/parent, activity phase, timestamps, pending queue) and its `outcome`/`metadata`/`touch`/`close` transitions
- `./runtime/quota.rs` → `SubAgentQuota` (Python `SubagentRuntimeConfig`), `SpawnBudget`, `SubagentActivity`, `LruCandidate`, `select_lru_candidate`, `compare_lru`, quota unit tests
- `./runtime/evict.rs` → `EVICTION_REASON`, `SubAgentRuntime::{evict, evict_lru, eviction_candidates}`, `live_count`/`candidates`/`evict_locked`/`make_room` helpers
- `./runtime/turn.rs` → `TurnRunner` background worker, `begin_turn`, `finish_turn`, `map_turn_result`
- `./runtime/clock.rs` → `Clock` trait, `SystemClock`, `ManualClock`, `rfc3339_from_ms`
- `./runtime/transcript.rs` → transcript ring buffer and broadcast channel (merged, not yet wired into the runtime)
- `./runtime/tests.rs` → runtime unit tests (fake factory/agent)
- `./runtime/tests/quota.rs` → quota, injected-clock and activity-phase tests over the same fakes
- `./runtime/tests/eviction.rs` → LRU eviction tests (cap reclaim, running guard, Idle-before-Queued, age/id order, `evict`/`evict_lru`/`eviction_candidates`, budget interplay)
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
- Quota/clock/activity: cumulative budget (exact `spawned=2, limit=2` message, unchanged by `close`, rolled back on factory failure), per-parent buckets (root label = parent session id vs nested parent id), depth `Some(2)` refuses depth 3, unknown/closed parent validation error, permissive defaults, `ManualClock` timestamps (`created_at`/`last_used_at`), `Queued → Running → Idle → Closed` observed through `get_metadata`/`describe_list`, `closed_reason` (`manual`/`parent_ended`), and `with_limits` disabling LRU while keeping the legacy capacity message.
- Eviction: a full runtime with one `Idle` instance reclaims it for the new spawn (`capacity()` stays at the cap, `closed_reason == "evicted"`, `summary.evicted_count == 1`); an all-`Running` runtime keeps the legacy capacity error and closes nothing; `Idle` beats an older `Queued` (built by closing and resuming a semaphore-queued instance), age then id order, `evict_lru` `Some`→`None`, `evict` validation messages, the cumulative budget surviving eviction, and a nested spawn never evicting its own parent.
- Every new test is deterministic: `#[tokio::test]` (current-thread) plus the `watch`-gated blocked fake and `ManualClock`; no sleeps.
- Tools: exact names, exact `required` lists and property keys; `task_tool` success payload shape, unknown-type failure, env-var timeout resolution via a pure helper; `subagent_wait` validation and clamp helper; every expected failure returns `success=false` (never `Err`).
- Current suite: 113 unit tests under `subagents`, of which 10 are the `runtime/tests/eviction.rs` cases, 11 the `runtime/tests/quota.rs` cases and 14 the `runtime/quota.rs` cases (crate total 1146 unit + 80 integration + 8 doc tests), green with default and `--no-default-features`; `cargo clippy -p openjiuwen-harness --all-targets -- -D warnings` clean.
- Parity references: `tests/unit_tests/harness/test_plan_agent.py`, `test_explore_agent.py`, `test_deep_agent.py`, `subagent_runtime/test_control.py`, `tools/subagent/test_task_tool.py`.
