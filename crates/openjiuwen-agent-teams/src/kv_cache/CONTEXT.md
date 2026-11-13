# kv_cache (openjiuwen-agent-teams)

## Intent

Team-side KV-cache policy: bind every member of a team to a stable product-level cache identity, derive forked child identities from a parent member, and release the cache before the owning runtime is disposed.
Ports `openjiuwen/agent_teams/kv_cache/` (Python v0.1.20): `kv_cache_cleanup.py` becomes `cleanup.rs`, and the member-affinity half of the module — which Python splits across `Session.bind_parent_session_id` / `release_kvc` plus the harness lifecycle hook — becomes the registry in `affinity.rs`.
It owns no cache I/O: recording, lookup and eviction are delegated to `openjiuwen_agent::kv_cache` through `KVCacheRuntimeProtocol`, and the model-call hook is the agent crate's `KVCacheModelCallHook`.
Affinity is opt-in and best-effort: a disabled config registers nothing, builds no runtime and makes every accessor return `None`.

## API Surface

```rust
// affinity.rs
pub const MEMBER_FORK_SEGMENT: &str = "fork";
pub struct MemberCacheKey { pub team_name: String, pub team_session_id: String, pub member_name: String }
impl MemberCacheKey { pub fn new(team_name, team_session_id, member_name) -> Self; pub fn cache_id(&self) -> String; }
pub struct MemberCacheEntry { pub key: MemberCacheKey, pub identity: KVCacheIdentity, pub parent_member: Option<String> }
impl MemberCacheEntry { pub fn new(key, identity) -> Self; pub fn with_parent_member(self, parent) -> Self; }
pub struct TeamKvCacheAffinity;   // Debug (manual), Send + Sync
impl TeamKvCacheAffinity {
    pub fn new(config: KVCacheAffinityConfig) -> Self;
    pub fn disabled() -> Self;
    pub fn with_runtime(self, runtime: Arc<dyn KVCacheRuntimeProtocol>) -> Self;
    pub fn is_enabled(&self) -> bool;
    pub fn config(&self) -> &KVCacheAffinityConfig;
    pub fn runtime(&self) -> Option<Arc<dyn KVCacheRuntimeProtocol>>;
    pub fn register_member(&self, key: MemberCacheKey, model: &str) -> KVCacheIdentity;
    pub fn register_child(&self, parent_member: &str, key: MemberCacheKey, model: &str) -> Result<KVCacheIdentity>;
    pub fn identity(&self, member_name: &str) -> Option<KVCacheIdentity>;
    pub fn affinity_session_id(&self, member_name: &str) -> Option<String>;
    pub fn entry(&self, member_name: &str) -> Option<MemberCacheEntry>;
    pub fn members(&self) -> Vec<String>;                       // sorted
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn env(&self, member_name: &str) -> Option<BTreeMap<String, String>>;
    pub fn child_session_kwargs(&self, member_name: &str) -> Option<serde_json::Value>;
    pub fn model_call_hook(&self) -> Option<KVCacheModelCallHook>;
    pub async fn release(&self, member_name: &str) -> usize;
    pub async fn release_team(&self, team_session_id: &str) -> usize;
    pub fn clear(&self);
}

// cleanup.rs
pub struct CleanupReport { pub released: bool, pub disposed: bool, pub timed_out: bool, pub release_error: Option<String>, pub dispose_error: Option<String> }
pub fn terminal_cleanup_timeout() -> std::time::Duration;   // 5 s
pub async fn cancellation_safe_release_then_dispose(
    release: Option<BoxFuture<'static, Result<bool>>>,
    dispose: BoxFuture<'static, Result<()>>,
    owner_id: &str,
    timeout: Duration,
) -> CleanupReport;
```

`mod.rs` re-exports the two submodules flat (`affinity`, `cleanup`); the crate root adds the flat re-exports later, not this node.

## Reused agent-crate types (never redefined here)

- `openjiuwen_agent::kv_cache::KVCacheIdentity` — the member identity returned by the registry; no local identity type exists.
- `openjiuwen_agent::kv_cache::team_member_cache_id(team_session_id, team_id, member_id)` — persisted cache id `team:{session}:team:{team}:member:{member}` used by `MemberCacheKey::cache_id`.
- `openjiuwen_agent::kv_cache::team_member_cache_identity(team_session_id, team_id, member_id, model)` — root member identity (parent = team session, empty prefix hash, zero counts).
- `openjiuwen_agent::kv_cache::KVCacheEntry` — only in tests, to pre-record prefixes before a release.
- `openjiuwen_agent::kv_cache::KVCacheRuntimeProtocol` — `record`/`lookup`/`evict_session`/`stats`; the only cache I/O seam.
- `openjiuwen_agent::kv_cache::InProcessKVCacheRuntime` — the default runtime, built at `KVCacheAffinityConfig::max_entries`.
- `openjiuwen_agent::kv_cache::KVCacheAffinityConfig` — enabled flag, `max_entries`, and `effective_terminal_cleanup_timeout()` for the dispose budget.
- `openjiuwen_agent::kv_cache::KVCacheModelCallHook` — `new(runtime, config)` for `model_call_hook()`, `build_child_session_kwargs(parent)` for `child_session_kwargs()`.
- `openjiuwen_agent::kv_cache::KV_CACHE_AFFINITY_SESSION_ID_ENV` (`kv_cache_affinity_session_id`) and `KV_CACHE_AFFINITY_PARENT_SESSION_ID_ENV` (`kv_cache_affinity_parent_session_id`) — the two env keys in `env()`.
- `openjiuwen_agent::kv_cache::KVC_TERMINAL_CLEANUP_TIMEOUT_SECONDS` (`5.0`) — the default cleanup budget.

## Constraints

- Only this directory is edited; `src/lib.rs` already declares `pub mod kv_cache;` and the crate root adds flat re-exports.
- No new dependencies: serde, serde_json, futures, tokio, parking_lot, tracing plus `openjiuwen_agent`/`openjiuwen_common` cover everything.
- Errors are `openjiuwen_common::error::{Error, Result}` only: `Error::not_found` for unknown/disabled fork parents, `Error::validation` for a blank fork member name; `release` and the cleanup path never return an error.
- No panics on user input, no `unsafe`, `tracing` only (no `println!`), every public item documented, pure getters `#[must_use]`.
- The registry mutex (`parking_lot::Mutex<BTreeMap<String, MemberCacheEntry>>`) is never held across an `await`: `release` collects targets under the lock, drops it, then awaits eviction.
- Public data types are `Debug + Clone`, serde `snake_case` + `default` + unknown-field tolerant, so a registry row or report can cross a wire format safely.
- Files stay well under 1000 lines (`affinity.rs` ~840 with tests, `cleanup.rs` ~440); no global state — one registry instance per team run.

## Design Decisions

### Identity and fork derivation

- A root member identity is exactly `team_member_cache_identity(team_session_id, team_name, member_name, model)`: persisted cache id, empty prefix hash, zero counts, parent = the team session id; the model-call hook fills the concrete hash and counts when it records a real request.
- `MemberCacheKey::team_name` doubles as the `team_id` argument of the persisted cache id, because `TeamAgentSpec` has no separate team id field; the mapping is documented rather than inventing a second id.
- A forked child session id is `format!("{parent_cache_id}:fork:{child_member}")` with `parent_session_id = parent cache id`, the same model, an empty prefix hash and `MemberCacheEntry::parent_member = Some(parent)`.
- The `:fork:` seam is local to this crate: `openjiuwen_agent::kv_cache` exposes no team-member fork constructor (only `context_compressor_cache_identity` and `team_member_cache_identity`), so the child shape is defined here and pinned by tests; `MEMBER_FORK_SEGMENT` names the literal.
- Registration is idempotent by `MemberCacheKey`: re-registering the same key returns the stored identity unchanged, so a re-registration cannot flatten a forked lineage back to a root identity.
- The registry is keyed by member name (one instance per team run); registering the same member name under a different team replaces the row, while `release_team` filters rows by `team_session_id` so a shared registry can still drop one team.
- `register_child` fails with `Error::not_found` when affinity is disabled or the parent is not registered, and with `Error::validation` for a blank child member name — a fork always needs a real parent to branch from.

### Disabled path

- `TeamKvCacheAffinity::new` builds an `InProcessKVCacheRuntime` only when `KVCacheAffinityConfig::is_enabled`; `disabled()` is the default config.
- With a disabled config nothing is stored (not even when a runtime is injected with `with_runtime`), `identity`/`env`/`child_session_kwargs`/`model_call_hook` return `None`, `register_member` still returns the deterministic identity (so a caller can keep one code path), and `release`/`release_team` return 0.
- Enabling is a property of the configuration, never of the injected runtime, so a host cannot half-enable affinity by accident.

### Release semantics

- `release(member)` drops the registry rows of the member and of every member forked from it (walking `parent_member`, so grandchildren are covered), then asks the runtime to evict each cache id; the runtime's own `evict_session` also cascades to anything parented at an evicted id, which is why a second pass over the intermediate ids counts 0 instead of double counting.
- The returned count is the sum of the runtime's evicted entries; a runtime error is logged and counts as 0, because release is best effort and a teardown must not fail.
- `release_team(team_session_id)` releases every registered member of that team only, leaving other teams sharing the registry untouched.
- `clear()` is the sync teardown: it drops the rows without evicting, for a host that is about to drop the whole runtime; use `release`/`release_team` when provider-side cache must actually be evicted.

### Cleanup ordering, shield and timeout

- Release runs before dispose inside one `tokio::spawn`ed task, because eviction must happen while the runtime that owns the cache handle still exists.
- Rust has no `asyncio.shield`: dropping a future is cancellation, so the work lives in a spawned task instead — dropping the future returned by `cancellation_safe_release_then_dispose` leaves that task running (still bounded by its own timeout), which is the shield equivalent and is pinned by `dropping_the_returned_future_does_not_abandon_the_cleanup`.
- A release error, and a panic in either step, is caught (`futures::FutureExt::catch_unwind` over `AssertUnwindSafe`), logged through `tracing`, recorded in `release_error`/`dispose_error`, and never skips `dispose`.
- The whole sequence is bounded by the `timeout` argument (default `terminal_cleanup_timeout()` = 5 s); on expiry the task is aborted, `timed_out = true`, and the partial progress already written to the shared report is kept, so "release done, dispose cut off" is visible.
- The function never panics and never returns an error type: failures live in `CleanupReport`, because a session teardown must not fail on cache work.
- `CleanupReport::released` is the release future's own report: `Ok(true)` means something was evicted, `Ok(false)` means the step ran with nothing to release, and no release future at all also leaves it `false`; `release_error` distinguishes a failed step.

## Test Strategy

38 inline `#[cfg(test)]` tests, offline and deterministic (no network, no wall-clock ordering, bounded waits ≤ 500 ms, panic-path tests use `catch_unwind`).

`affinity.rs` (24): `cache_key_serializes_with_snake_case_and_tolerates_unknown_fields`, `cache_id_keeps_the_persisted_team_member_format`, `entry_round_trips_through_json`, `registering_a_member_is_idempotent`, `member_identity_starts_empty_and_parents_at_the_team_session`, `register_child_derives_from_the_parent_cache_id`, `register_child_of_an_unknown_parent_is_not_found`, `register_child_rejects_a_blank_member_name`, `register_child_is_idempotent`, `env_carries_the_two_persisted_keys`, `child_session_kwargs_render_the_python_shape`, `affinity_session_id_and_entry_expose_the_registration`, `disabled_registry_stores_nothing_and_evicts_nothing`, `disabled_registry_ignores_an_injected_runtime`, `registration_builds_an_in_process_runtime_at_max_entries`, `release_evicts_the_member_and_its_forks`, `release_keeps_unrelated_members`, `release_of_an_unknown_member_is_a_noop`, `release_of_a_grandchild_keeps_the_parent`, `release_team_only_drops_that_team`, `clear_drops_rows_without_touching_the_runtime`, `model_call_hook_records_and_reuses_a_member_prefix`, `concurrent_registration_and_release_leave_no_entries` (32 spawned registrations, then 32 concurrent releases → registry and runtime empty), `debug_lists_registered_members_without_exposing_the_runtime`.

`cleanup.rs` (14): `report_defaults_to_nothing_done`, `report_serializes_every_field`, `terminal_cleanup_timeout_matches_the_python_constant`, `release_runs_before_dispose` (ordering through a shared log), `without_release_only_dispose_runs`, `release_failure_still_disposes`, `release_reporting_nothing_released_is_not_an_error`, `dispose_failure_is_reported`, `release_panic_still_disposes`, `dispose_panic_is_reported`, `timeout_cancels_the_slow_dispose_and_keeps_partial_progress`, `timeout_without_release_reports_no_progress`, `dropping_the_returned_future_does_not_abandon_the_cleanup`, `cleanup_runs_once_per_call`.

Tests carry `#[allow(clippy::unwrap_used, clippy::expect_used)]`; the module doc of `mod.rs` carries the single runnable doctest.

Definition of done: `cargo fmt -p openjiuwen-agent-teams`, `cargo test -p openjiuwen-agent-teams --offline --lib` (with `--no-default-features` and `--features otel`), `cargo clippy -p openjiuwen-agent-teams --all-targets --offline -- -D warnings` (also with `otel`), `cargo doc -p openjiuwen-agent-teams --no-deps` warning-free.

## Where the runtime should wire this

- Register: when the runtime creates or admits a member and knows `team_session_id` + the member's model, call `TeamKvCacheAffinity::register_member(MemberCacheKey::new(team_name, team_session_id, member_name), model)` and hand `env(member_name)` (or `affinity_session_id`) to the member session; for a spawned sub-member of an existing member use `register_child(parent, key, model)`.
- Release: a one-shot worker releases at the end of its only turn (`release(member_name)`); a multi-turn member releases at `stop`/`shutdown`; the team releases at teardown (`release_team(team_session_id)`) and drops the registry afterwards (`clear()` only if the runtime is being dropped too).
- Dispose: when the same component also owns a runtime, wrap the terminal path in `cancellation_safe_release_then_dispose(Some(Box::pin(async { Ok(affinity.release(name).await > 0) })), dispose, session_id, affinity.config().effective_terminal_cleanup_timeout())` so a cancelled caller still evicts.
- The registry belongs to the team run (`TeamBackend`/`TeamRuntimeManager` scope): one instance per team, shared as `Arc<TeamKvCacheAffinity>` with the member runtimes, never a process global.

## Routing Table

- `affinity.rs` → `MemberCacheKey`, `MemberCacheEntry`, `TeamKvCacheAffinity` (registration, forked children, env/kwargs, hook, release, `release_team`, `clear`) and its 24 tests.
- `cleanup.rs` → `CleanupReport`, `terminal_cleanup_timeout`, `cancellation_safe_release_then_dispose` and its 14 tests.
- `mod.rs` → module documentation, `affinity`/`cleanup` declarations, flat re-exports, runnable doctest.

## Deferred (with reasons)

- Provider-side prefetch/offload: `KVCacheCallCapabilities::offload` is always `false` in `openjiuwen_agent::kv_cache` and the in-process runtime only stores hints — there is no provider SDK action to call, so the team layer schedules nothing.
- `Redis`/`distributed` runtimes: `KVCacheBackend::Redis`/`Distributed` return `Error::Unsupported` in the agent crate; `with_runtime` accepts any protocol implementation, so a future backend plugs in without changing this module.
- Harness DeepAgent session-lifecycle hooks (Python `kv_cache_harness_session_lifecycle_hook.py`: `configure_harness_session_hooks`, `on_harness_session_created`, `after_harness_session_finished`, `clear_harness_session_hooks`, `is_harness_affinity_enabled`): they read `harness.deep_config.kv_cache_affinity_config` and keep a weak per-harness registry, which the Rust harness adapter and `DeepAgentConfig` do not expose; the identity/registry primitives they would call are implemented here.
- Cross-process / distributed member affinity: the team runtime is in-process only, and a second process would need its own registry plus a shared store or transport for eviction; `release` is therefore process-local and `live-tests` covers nothing here.
- Group-chat, org-layer and swarmflow-specific cache policy (per-round eviction in Python's `team_harness.start`, worker-vs-avatar lifetimes in the swarmflow backends): the org layer and the swarmflow runner do not consult this registry, so the module exposes only the primitive such a policy would need.
- Cache observability (eviction counters per member, OTel spans around release): the `monitor`/`observability` nodes own reporting; `CleanupReport` is the only structured output this module produces.
