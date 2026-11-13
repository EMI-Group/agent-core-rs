# subagents

## Intent

Sub-agent definitions, presets, an in-process runtime with configurable quotas, LRU eviction, bounded transcripts and persisted buckets, and the model-facing task/subagent tools.
Ports `openjiuwen/harness/subagents/` (presets), `subagent_runtime/` (control/instance/registry/config/transcript/persistence subset), `subagent_lifecycle.py`, and `tools/subagent/{task_tool,subagent_tools}.py`.
`SubAgentConfig` is defined in `crate::schema` and re-exported here.
Everything is in-process: no child processes, no network, no background scheduler; the runtime restores only what a caller loads into it.
A child agent is built lazily by a [`SubAgentFactory`] so this module never depends on `crate::agent`.

## API Surface

Re-exports (`mod.rs`): `SubAgentConfig` (from `crate::schema`), `SubAgentSpec`, `SubAgentRegistry`, `SubAgentPreset`, `build_preset`, `general_purpose_config`, `GENERAL_PURPOSE_NAME`, `SubAgentRuntime`, `SubAgentFactory`, `SubAgentId`, `SubAgentSpawnRequest`, `SubAgentOutcome`, `SubagentMetadata`, `SubagentStatus`, `ResumeReport`, `LoadReport`, `EVICTION_REASON`, `TaskTool`, the model-facing tools plus `subagent_tools(runtime)` and `subagent_tools_with_transcript(runtime)`.
Runtime support re-exports (`mod.rs`): the `runtime::{clock, persistence, quota, transcript}` modules, `Clock`, `SystemClock`, `ManualClock`, `rfc3339_from_ms`, `SubAgentQuota`, `SpawnBudget`, `SubagentActivity`, `LruCandidate`, `select_lru_candidate`, the transcript types, `PersistedSubAgentBuckets`, `PersistedSubAgentRecord`, `SUBAGENTS_KEY`, `max_persisted_records`, `load_bucket_file`, `save_bucket_file`.

- `SubAgentSpec { config: SubAgentConfig, preset: Option<SubAgentPreset> }` with `new(config)`, `with_preset(config, preset)`, `name()` (`agent_card.name`).
- `SubAgentPreset` enum: `Code`, `Explore`, `Plan`, `Research`, `Verification`, plus feature-gated `Browser`, `Cua`, `Mobile`; `Default = Code`.
  `as_str()` returns the Python type name (`code_agent`, `explore_agent`, `plan_agent`, `research_agent`, `verification_agent`, `browser_agent`, `cua_agent`, `mobile_gui_agent`); `builtin()` returns the five always-available presets in Python order.
- `build_preset(preset, language) -> Result<SubAgentConfig>`; feature-gated presets return `Error::Unsupported`.
- `general_purpose_config()` — card name `general-purpose`, no system prompt, `restrict_to_work_dir = false`, `max_iterations = None`, CN description, `subagent_type = Some("general-purpose")`.
- `SubAgentRegistry`: `new()`, `with_builtin_presets(language)`, `register(SubAgentSpec)` (replace by `agent_card.name`), `get(name)`, `get_spec(name)`, `contains`, `list`, `list_specs`, `names()`, `len`, `is_empty`, `general_purpose()` (registers on first use).
  Every clone shares one `Arc<parking_lot::RwLock<BTreeMap<String, SubAgentSpec>>>`.
- `SubagentStatus` (`snake_case` serde): `PendingInit`, `Running`, `Completed`, `Interrupted`, `Errored`, `Closed`, `NotFound`; `is_final()` is true for `Completed`/`Errored`/`Closed`/`NotFound`; `as_str()` exposes the Python wire string.
- `SubagentActivity` (derived phase, `snake_case` serde): `Queued`, `Running`, `Idle`, `Closed`; `as_str()`, `is_evictable()` (true for `Idle`/`Queued`).
- `SubAgentId = String`; id format `<parent_session>_sub_<type>[_<8hex>]`, without the hex suffix for the sticky type `verification_agent`; `sub_session_id == id`.
- `SubAgentSpawnRequest { subagent_type, task_description, display_name, role, subagent_id, parent_subagent_id }` with `new(type, description)`; `parent_subagent_id` names the live instance a nested spawn belongs to (`None` = top-level).
- `SubAgentOutcome { subagent_id, sub_session_id, subagent_type, agent_id, status, output, error, transcript: TranscriptSummary }` with `is_final()`.
- `SubagentMetadata { subagent_id, sub_session_id, subagent_type, display_name, role, task_description, status, created_at, updated_at, activity, depth, parent_subagent_id, last_used_at, closed_reason, transcript: TranscriptSummary }`.
- `ResumeReport { status, restored, message }`; `LoadReport { records, restored, skipped }`.
- `SubAgentFactory` (`#[async_trait]`, `Send + Sync`): `async fn create(&self, config: &SubAgentConfig, session_id: &str) -> Result<Arc<dyn Agent>>`.
- `SubAgentRuntime::new(registry, factory, parent_session_id)`; builders `with_quota(SubAgentQuota)`, `with_clock(Arc<dyn Clock>)`, `with_limits(max_subagents, max_concurrent_running)`, `with_turn_timeout(Duration)`; accessors `quota()`, `parent_session_id()`, `registry()`.
  Lifecycle: `spawn(SubAgentSpawnRequest) -> Result<SubAgentId>`, `wait(&[SubAgentId], timeout) -> Vec<SubAgentOutcome>`, `get_status`, `get_outcome`, `get_metadata` (tombstones included), `list()` (live only), `capacity() -> (used, max)`, `spawned_total()`, `send_input(id, query, interrupt) -> Result<()>`, `resume(id) -> Result<ResumeReport>`, `close(id) -> Result<SubagentStatus>`, `cancel_all() -> Vec<SubAgentId>`, `describe_list() -> Value`.
  Eviction (`runtime/evict.rs`): `evict(id) -> Result<SubagentStatus>`, `evict_lru() -> Result<Option<SubAgentId>>`, `eviction_candidates() -> Vec<SubAgentId>` (evictable ids in LRU order), `EVICTION_REASON` (`"evicted"`).
  Transcript (`runtime/transcript_api.rs`): `transcript(id)`, `transcript_after(id, cursor, limit)` (`0` = no limit), `transcript_summary(id)`, `subscribe_transcript(id)`, `record_tool_call(id, tool_name, detail, success) -> Result<u64>`, `record_transcript(id, entry) -> Result<u64>`, `clear_transcript(id) -> Option<usize>`.
  Persistence (`runtime/persist_api.rs`): `snapshot() -> PersistedSubAgentBuckets`, `save() -> PersistedSubAgentBuckets` (trimmed), `save_to_path(&Path) -> Result<PersistedSubAgentBuckets>`, `load(&PersistedSubAgentBuckets) -> Result<LoadReport>`, `load_from_path(&Path) -> Result<LoadReport>`.
  Defaults: `SubAgentQuota::default()` (10 live, 5 concurrent turns, no cumulative/per-parent/depth cap, LRU eviction enabled), `SystemClock`, 2400 s per turn.
- `SubAgentQuota { max_subagents, max_concurrent_running, max_total_spawned, max_children_per_parent, max_depth, enable_lru_eviction, transcript_capacity, transcript_channel_capacity }` (`Copy`, serde with `default`): `Default`, `#[must_use]` fluent `with_*` setters, `check_spawn(&SpawnBudget, parent: &str) -> Result<()>`.
- `SpawnBudget { live, spawned, parent_live, depth }`; `LruCandidate { subagent_id, activity, last_activity_ms, spawn_seq }`; `select_lru_candidate(&[LruCandidate]) -> Option<&LruCandidate>`.
- Clock: `Clock { now_ms(), now_rfc3339() }` with `SystemClock` and `ManualClock::new/advance/set`; every runtime timestamp is read through it.
- Transcript types (`runtime/transcript.rs`): `TranscriptKind` (`spawn`/`input`/`assistant_turn`/`tool_call`/`completion`/`error`/`status`/`closed`/`evicted`), `TranscriptEntry { seq, subagent_id, kind, role, content, tool_name, success, at_ms }` with per-kind constructors and `summary()`, `TranscriptBuffer` (ring with `push`/`entries`/`recent`/`get`/`after`/`summary`/`dropped`/`clear`, cap `1..=4096`), `TranscriptSummary { count, dropped, last_kind, last_content }` (`from_buffer`, `display`), `TranscriptChannel` (`publish` that is a no-op without subscribers, `subscribe`, `receiver_count`), `TranscriptSubscription = tokio::sync::broadcast::Receiver<TranscriptEntry>`.
- Persistence types (`runtime/persistence.rs`): `SUBAGENTS_KEY` (`"subagents"`), `max_persisted_records(max_subagents) = max(n * 4, 40)`, `PersistedSubAgentRecord` (identity, depth/parent, `status`, `activity`, `spawn_seq`, `agent_id`, output/error, epoch-ms timestamps, `closed_at_ms`, `closed_reason`, `config: SubAgentConfig`, `transcript`), `PersistedSubAgentBuckets { revision, records }` with `get`/`insert`/`remove`/`merge`/`trim`/`to_json`/`from_json`/`to_json_string`/`from_json_string`, and `load_bucket_file`/`save_bucket_file`.
- `TaskTool` (`foundation::Tool`, name `task_tool`): inputs `subagent_type` (required), `task_description` (required), `resume_task_id` (optional); output `{output, agent_id, subagent_type}`; `new(runtime)`, `with_timeout(Duration)`, `card_for(language)`, `timeout()`; default timeout 720 s, overridable via `OPENJIUWEN_SUBAGENT_TASK_TIMEOUT_S` (invalid/non-positive keeps 720 s).
- Six parity tools, each `new(runtime: Arc<SubAgentRuntime>)`, `with_language(&str)`, `card_for(language)`; `new` builds the card in `crate::prompts::DEFAULT_LANGUAGE` (`cn`):
  - `subagent_spawn`: `subagent_type`, `task_description`, `display_name`, `role` (all required) -> `{subagent_id, sub_session_id, task_id, status}`.
  - `subagent_wait`: `subagent_ids` (required, non-empty; singular `subagent_id` accepted as fallback), `timeout_ms` (default 2_400_000, clamped to 10_000..=3_600_000) -> `{statuses, results, output_files, timed_out, transcripts}`.
  - `subagent_list` (no inputs) -> `describe_list()` (`capacity`, `subagents`, `live_subagents`, `closed_subagents`, `summary`).
  - `subagent_send_input`: `subagent_id`, `query` (required), `interrupt` (bool, default false) -> `{subagent_id, task_id, status}`.
  - `subagent_close`: `subagent_id` -> `{subagent_id, previous_status}`.
  - `subagent_resume`: `subagent_id` -> `{subagent_id, status, restored}` plus `message` when present.
- `subagent_transcript` (`SubagentTranscriptTool`, Rust-only extension): `subagent_id` (required), `limit` (integer, default 20, clamped `1..=200`), `after_seq` (integer, default 0) -> `{subagent_id, status, activity, entries, count, total, dropped, last_kind, has_more}`.
- `subagent_tools(runtime)` returns the six parity tools (spawn, wait, list, send_input, close, resume) — a crate-root test asserts that exact list; `subagent_tools_with_transcript(runtime)` returns seven in order spawn, wait, list, transcript, send_input, close, resume.

## Constraints

- `send_input` on a closed/unknown instance returns `Error::Validation("subagent closed or not found: <id>; call subagent_resume first")`, and on a live instance whose child agent was never built (a restored one) `Error::Validation("subagent not resumed: <id>; call subagent_resume first")`; `PendingInit` keeps queueing because its first turn drains the queue.
- Other validation messages: `subagent not found: <id>`, `subagent already live: <id>; use subagent_wait to collect its result`, `subagent type not found: <t>`, `subagent parent not found: <p>`, `cannot close running subagent: <id>`, `cannot evict running subagent: <id>`, `subagent already closed: <id>`.
- Quota refusals are `Error::Config`, checked depth -> per-parent -> cumulative -> live: `subagent depth limit exceeded: depth=<d>, limit=<l>`, `subagent child limit exceeded: parent=<p>, used=<u>, limit=<l>`, `subagent spawn budget exhausted: spawned=<s>, limit=<l>`, `subagent capacity exceeded: used=<u>, limit=<l>` (the live-cap text is the legacy one).
- A nested spawn inherits `depth(parent) + 1` and is refused when the parent is unknown or `Closed`; top-level instances have `depth = 1` and are counted in the root bucket labelled by the parent session id.
- `state.spawned` is the cumulative ledger behind `spawned_total()`: closing never lowers it, only a factory-failure rollback does. Capacity is reserved synchronously (the `Queued` instance exists before the factory is awaited) and rolled back on failure.
- `max_subagents` counts live instances; `max_concurrent_running` bounds turns through `tokio::sync::Semaphore`. `with_limits` is the pre-eviction convenience constructor: it sets both caps and forces `enable_lru_eviction = false`.
- Activity transitions are the only writers of the phase: reservation -> `Queued`, turn start -> `Running`, last turn of a worker -> `Idle` (whatever its final status), `close`/`cancel_all` -> `Closed`, `resume` -> `Idle`; each also stamps `updated_at`/`last_activity_at_ms`/`last_used_at_ms` from the clock.
- Eviction policy: `reserve` runs `evict::make_room` before the quota check while `live >= max_subagents && enable_lru_eviction`; it never evicts a `Running` instance, never recycles the requesting parent, prefers `Idle` over `Queued`, then the oldest `last_activity_ms`, then the smallest id, and stops when no evictable candidate remains (an all-running runtime keeps the legacy capacity error).
- `closed_reason` is `manual` for `close`, `parent_ended` for `cancel_all`, `evicted` (`EVICTION_REASON`) for evictions, and is cleared by `resume`.
- `resume` rebuilds an instance without a child agent through `factory.create(&config, &sub_session_id)` outside the state lock, propagating a factory error and leaving the instance untouched; a live instance with its child keeps the `restored: false` "already live" report.
- Persistence restore rules (`load`): visit records in `spawn_seq` then id order, skip ids already present (idempotent) and everything once `quota.max_subagents` live instances are held, downgrade `Running`/`PendingInit` to `Interrupted` and `Running`/`Queued` activity to `Idle`, keep `closed_reason` only on `Closed`, assign fresh `seq`, set `agent = None`/`running = false`, set `state.spawned = max(spawned, records.len())`, notify waiters once, and reject the whole bucket with `Error::Validation("invalid subagent record: <id> has status not_found")` before restoring anything.
- Persisted JSON shape: `{"revision": u64, "records": {"<subagent_id>": {<record keys>}}}` with snake_case record keys (`subagent_id`, `sub_session_id`, `subagent_type`, `display_name`, `role`, `task_description`, `status`, `activity`, `depth`, `parent_subagent_id`, `spawn_seq`, `agent_id`, `output`, `error`, `created_at_ms`, `updated_at_ms`, `last_used_at_ms`, `last_activity_at_ms`, `closed_at_ms`, `closed_reason`, `config`, `transcript`); `trim` drops `Closed` records first (oldest `sort_key_ms`, then id), then the oldest remaining ones.
- Persistence errors are typed: encode `Error::Validation("failed to encode subagent bucket: ...")`, decode `Error::Validation("invalid subagent bucket: ...")`, file IO `Error::Session("failed to {read,write} subagent bucket <path>: ...")`; a missing bucket file loads as an empty bucket.
- Transcript: each `Instance` owns one `TranscriptBuffer` (cap `quota.transcript_capacity`) and one `TranscriptChannel` (cap `quota.transcript_channel_capacity`); events are pushed and published under the state lock (`publish` is a non-blocking broadcast fan-out, so no `.await` runs inside the lock).
- Transcript recording is explicit at each transition: `Instance::new` records `spawn(task_description)`, `send_input` records `input(query)` plus `status("interrupted")` when it aborted a turn, `begin_turn` records `status("running")`, `finish_turn` records `input(follow_up)` or `completion(output)`/`error(message)`/`status("interrupted")`, `close` records `closed("manual")`, `cancel_all` `closed("parent_ended")`, eviction `evicted("evicted")`, `resume` `status("idle")`.
- The runtime only sees whole `Agent::invoke` turns, so child-internal tool calls reach the transcript through `record_tool_call`/`record_transcript` (factory or child rails layer); `record_transcript` overwrites the caller's `subagent_id`/`seq`/`at_ms`.
- `describe_list().capacity` carries `used`, `max`, `spawned`, `max_spawned`, `running`, `max_running`, `transcript_entries`; `summary` carries `live_count`, `closed_count`, `evicted_count`, `eviction {enabled, candidates}`; every row carries `activity`, `depth`, `parent_subagent_id`, `last_used_at`, `closed_reason`, `transcript`.
- Child turns run in background `tokio` tasks: `spawn` returns once the instance is registered, `wait` blocks until every id is final or the timeout elapses (input order preserved, duplicates collapsed, unknown ids map to `NotFound`).
- Presets set `max_iterations = 50`, `enable_task_loop = false`, `enable_plan_mode = false`, `parallel_tool_calls = true`; `explore`/`plan` set `restrict_to_work_dir = false`; `code`/`research` set `factory_name`.
- Feature-gated presets (`browser`, `cua`, `mobile`) exist only with the matching cargo feature and return `Error::Unsupported` from `build_preset`; the agents themselves are deferred.
- No dependency on `crate::agent`, `crate::rails`, or `crate::tools` internals beyond the shared `crate::tools` helpers (`tool_card`, `parse_args`, `ToolPayload`, `render_fields`, `descriptions::description`).
- Tools never panic and never return `Err` for expected failures: they return a `ToolPayload` JSON with `success=false`, `data=null` and the Python message in `error`.
- Locks are `parking_lot::Mutex`/`RwLock` and never held across `.await`; status waits use `tokio::sync::Notify`; interrupted turns use `JoinHandle::abort`.
- Files ≤ 1000 lines (`runtime.rs` is the largest at ~940; tests live in `runtime/tests.rs`, `runtime/tests/*.rs`, `runtime/quota.rs`, `subagent_tools/tests.rs`, `subagent_tools/tests/transcript.rs`).

## Design decisions

- Status and activity are separate axes: `SubagentStatus` keeps the Python wire strings so `wait`/`is_final` semantics stay intact, while `SubagentActivity` is the derived `Queued -> Running -> Idle -> Closed` phase that drives eviction.
- No `SubAgentSpawner` trait is defined here: `crate::rails` owns that seam and can `impl` it / wrap [`SubAgentRuntime`]. The inherent runtime methods are the stable partner API.
- The workspace is not a runtime field: `SubAgentConfig` carries the subagent's own settings and the factory re-roots it at `sub-agents/<sub_session_id>` (Python `create_subagent` behavior).
- LRU eviction is wired, not scheduled: `evict::make_room` runs inside `reserve` (Python `SubagentControl._acquire_slot`) and `evict`/`evict_lru` are caller-driven. `eviction_candidates` sorts with `quota::compare_lru`, the same order `select_lru_candidate` picks from, so the reported list and the pick agree.
- The runtime holds no persistence state of its own: `snapshot`/`save` project the live state, `load` restores what a caller provides, closed tombstones stay in memory so `resume`/`close` remain idempotent, and nothing schedules a background save.
- `Instance.config` keeps the patched configuration the child was built from, so `Instance::persisted()` writes a restorable record and `Instance::restore()` rebuilds one without a child agent (which `resume` then creates).
- Runtime internals: one `Arc<Shared>` (`registry`, `factory`, `parent_session_id`, `clock`, `notify`, `Mutex<RuntimeState>`) plus per-runtime `quota`, `turn_timeout` and `Semaphore`; each instance keeps a spawn-sequence generation so an aborted or closed turn can never resurrect a tombstoned instance.
- Queued `send_input` queries are FIFO; a turn that finds queued work transitions through `PendingInit` instead of exposing a final status, and the last turn applies the terminal status.
- Thread/task interrupt uses `JoinHandle::abort`, the only safe cancellation of a `dyn Agent::invoke` future in Rust; the interrupting caller sets `Interrupted` itself because an aborted task runs no code.
- `TaskTool` awaits `spawn` + `wait` inline (Python sync delegation path) and reports `agent_id` from the child `Agent::card().id`.
- Task ids are a tool-layer concept: `TaskTool`/`subagent_*` payloads carry a fresh uuid hex `task_id` generated by the tool, because the runtime models work as instance turns, not as a task queue.
- Tool descriptions resolve through `description_for` in `task_tool.rs`: `crate::tools::descriptions` first, then the `crate::prompts` table, then `EXTENSION_TOOL_DESCRIPTIONS` (which holds the Rust-only `subagent_transcript` row); unknown names yield `""`.
- `subagent_tools` is a directory module: the parent file keeps the constants, the `subagent_tool!` macro, `subagent_tools()`/`subagent_tools_with_transcript()` and the spawn/wait/list tools, `control.rs` holds send_input/close/resume, `transcript.rs` the transcript reader and `util.rs` the shared helpers. `mod` declarations must stay below the `subagent_tool!` definition, because a `macro_rules!` macro is only in scope for child modules declared after it.
- `subagent_tools()` stays at six tools for Python parity while `subagent_tools_with_transcript()` mounts the additive seventh; wiring that set into the DeepAgent factory belongs to `crate::agent` (outside this node).

## Known Issues

- `crate::tools::descriptions` has no rows for `task_tool` or the `subagent_*` names (including `subagent_transcript`), so their descriptions fall back to the `crate::prompts` table and `EXTENSION_TOOL_DESCRIPTIONS`; adding the rows belongs to `crate::tools` (sibling module).
- The crate-root integration suite `tests/subagents_skills.rs` covers the pre-quota surface only; the quota, LRU, transcript and persistence surfaces are covered by the node's unit suites listed under Test Strategy, because `tests/` sits outside this node.
- `crates/openjiuwen-harness/src/lib.rs` has no flat crate-root re-exports of the new public items (`Clock`, `SubAgentQuota`, `TranscriptEntry`, `PersistedSubAgentBuckets`, …); they are reachable through `openjiuwen_harness::subagents::{...}` and the flat `subagents` re-exports.

## Deferred Python areas

- `subagents/{browser,cua,mobile_gui}_agent.py` — feature-gated stubs returning `Error::Unsupported`.
- `subagent_runtime/activity.py` — no activity stream (the transcript ring and its broadcast feed are ported instead); the Python stream-chunk projector is not ported, and `activity_queue_size`/`activity_text_max_len`/`activity_throttle_ms`/`enable_transcript_stream` map only onto `transcript_capacity`/`transcript_channel_capacity`.
- `subagent_runtime` background eviction scheduler — eviction is caller-driven (reserve-time `_acquire_slot` pass plus the explicit `evict`/`evict_lru` API).
- `config.py`'s `turn_timeout_s` lives separately as `SubAgentRuntime::with_turn_timeout`.
- `subagent_lifecycle.py` resource hooks — the factory owns child resource setup/teardown.
- `browser_capabilities`, `resume_task_id` orchestration for browser/CUA, KV-affinity sub-session rewriting, and `TaskTool`'s streaming/observability path.

## Routing Table

- `./mod.rs` → re-exports, `SubAgentSpec`, `SubAgentRegistry`
- `./presets.rs` → preset matrix, bilingual prompts/descriptions, `build_preset`, `general_purpose_config`
- `./runtime.rs` → `SubAgentRuntime`, statuses, ids, outcomes, metadata, `ResumeReport`, `SubAgentFactory`, quota-aware reservation with the reserve-time eviction pass, `describe_list`
- `./runtime/instance.rs` → `Instance` state (identity, depth/parent, activity phase, timestamps, pending queue, transcript ring + feed, `config`) and its `persisted`/`restore`/`record`/`outcome`/`metadata`/`touch`/`close` transitions
- `./runtime/quota.rs` → `SubAgentQuota` (Python `SubagentRuntimeConfig`), `SpawnBudget`, `SubagentActivity`, `LruCandidate`, `select_lru_candidate`, `compare_lru`, quota unit tests
- `./runtime/evict.rs` → `EVICTION_REASON`, `SubAgentRuntime::{evict, evict_lru, eviction_candidates}`, `live_count`/`candidates`/`evict_locked`/`make_room` helpers
- `./runtime/turn.rs` → `TurnRunner` background worker, `begin_turn`, `finish_turn`, `map_turn_result`
- `./runtime/clock.rs` → `Clock` trait, `SystemClock`, `ManualClock`, `rfc3339_from_ms`
- `./runtime/transcript.rs` → `TranscriptKind`/`TranscriptEntry`/`TranscriptBuffer`/`TranscriptSummary`/`TranscriptChannel` (ring + broadcast feed)
- `./runtime/transcript_api.rs` → `SubAgentRuntime` transcript accessors: read, paginate, digest, subscribe, record tool calls/caller entries, clear
- `./runtime/persistence.rs` → persisted bucket types, JSON/file round-trip, deterministic trimming
- `./runtime/persist_api.rs` → `LoadReport`, snapshot/save/load accessors, file round-trip, and `resume`'s child-agent rebuild
- `./runtime/tests.rs` → runtime unit tests (fake factory/agent)
- `./runtime/tests/quota.rs` → quota, injected-clock and activity-phase tests
- `./runtime/tests/eviction.rs` → LRU eviction tests (cap reclaim, running guard, Idle-before-Queued, age/id order, `evict`/`evict_lru`/`eviction_candidates`, budget interplay)
- `./runtime/tests/transcript.rs` → transcript tests (lifecycle order and clock stamps, ring cap, `record_*` seams, pagination, subscription, digests)
- `./runtime/tests/persistence_runtime.rs` → bucket round-trip, restore rules, resume rebuild and file IO tests
- `./task_tool.rs` → `TaskTool`, timeout resolution, shared tool description lookup
- `./subagent_tools.rs` → module docs, constants, the `subagent_tool!` macro, `subagent_tools()`, `subagent_tools_with_transcript()`, and the spawn/wait/list tools
- `./subagent_tools/control.rs` → `subagent_send_input`, `subagent_close`, `subagent_resume` tools
- `./subagent_tools/transcript.rs` → `SubagentTranscriptTool` (`subagent_transcript`) and its rendering
- `./subagent_tools/util.rs` → shared parsing, task-id and payload/row/transcript rendering helpers (`pub(crate)`)
- `./subagent_tools/tests.rs` → unit tests for the tools and the additive tool set
- `./subagent_tools/tests/transcript.rs` → `subagent_transcript` card/payload/pagination/rendering tests
- `./test_support.rs` → `#[cfg(test)]` fake agent/factory and runtime fixture shared by the tool tests

## Test Strategy

- Unit tests with fakes: `test_support.rs` provides echo/error-result/failing agents and factories plus `Notify` gates; runtime tests keep their own local fakes in `runtime/tests.rs`.
- `test_support` is `#[cfg(test)] mod` with `pub(crate)` items, so integration tests under `../../tests/` must define their own `Agent` + `SubAgentFactory` fakes.
- Everything is deterministic: `ManualClock` freezes or advances timestamps, `Notify`/`watch` gates replace sleeps, `#[tokio::test]` runs current-thread, and subscription assertions are bounded by a ≤5 s `tokio::time::timeout`.
- Registry/presets: replace by name, builtin names, `general_purpose` insertion, shared map across clones, name ordering, `max_iterations` 50, per-preset `restrict_to_work_dir`, `factory_name`, bilingual selection (`en` vs anything else -> `cn`), feature-gated presets returning `Error::Unsupported`.
- Runtime: spawn -> `PendingInit`/`Running` -> `Completed` with output; wait timeout leaves non-final; close refuses running; close/resume on a completed instance; capacity rejection; rollback frees the slot and the spawn ledger; duplicate live spawn rejected; child session id format and sticky verification id; FIFO queued inputs; interrupt; `cancel_all`; `describe_list` shape.
- Quota/clock/activity: cumulative budget (exact message, unchanged by `close`, rolled back on factory failure), per-parent buckets (root label = parent session id vs nested parent id), depth refusal, unknown/closed parent validation, permissive defaults, exact `ManualClock` timestamps, `Queued -> Running -> Idle -> Closed` through `get_metadata`/`describe_list`, `closed_reason` values, and `with_limits` disabling LRU while keeping the legacy message.
- Eviction: a full runtime with an `Idle` instance reclaims it (capacity stays at the cap, `closed_reason == "evicted"`, `summary.evicted_count == 1`); an all-`Running` runtime keeps the capacity error and closes nothing; `Idle` beats an older `Queued`; age then id order; `evict`/`evict_lru`/`eviction_candidates`; the cumulative budget survives eviction; a nested spawn never evicts its own parent.
- Transcript: exact `(kind, content)` sequences with frozen clock stamps, monotone `seq`, ring-cap `dropped`, unknown-id validation, `record_tool_call`/`record_transcript` stamping, pagination and `clear_transcript`, live subscription delivery, and digests on metadata/outcome/`describe_list`.
- Persistence: save -> reconstruct (status, output, `agent_id`, transcript, config, `closed_reason`), `resume` rebuild and `send_input` before it, mid-turn downgrade, live cap, double `load` idempotence, retention trim, file round-trip/missing/broken file, `not_found` record rejection, restored spawn budget, snapshot revision.
- Tools: exact names, `required` lists and property keys; `task_tool` payload shape and env-var timeout resolution; `subagent_wait` validation, clamp and `transcripts` map; `subagent_transcript` payload/pagination/clamping/rendering; `subagent_list` capacity/eviction rendering; every expected failure returns `success=false` (never `Err`).
- Current suite: 162 unit tests under `subagents` (crate total 1192 unit + 88 integration + 8 doc tests), green with default and `--no-default-features`; `cargo clippy -p openjiuwen-harness --all-targets -- -D warnings` clean.
- Parity references: `tests/unit_tests/harness/test_plan_agent.py`, `test_explore_agent.py`, `test_deep_agent.py`, `subagent_runtime/test_control.py`, `tools/subagent/test_task_tool.py`.
