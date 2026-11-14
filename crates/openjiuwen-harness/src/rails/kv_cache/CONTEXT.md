# kv_cache

## Intent

The harness-side bridge between DeepAgent subagent lifecycle events and provider-level KV-cache affinity, ported from `openjiuwen/harness/kv_cache/` (policy: `kv_cache_subagent_lifecycle.py`).
It owns policy and identity bookkeeping only: whether affinity is on, which subagent types are sticky, which cache identity a child delegation gets, and which provider action each lifecycle transition drives.
Cache mechanics (prefetch/offload/evict, bindings, action tails) belong to the provider seam and to the kernel session, not here.
Lifecycle mapping (spec S_16): start + sticky → prepare; success + sticky → suspend (resumable); success + one-shot → release; failure/cancel/any type → release; explicit eviction → release.
Affinity is an optional optimisation: when off, nothing is stored, no child cache identity is minted and no provider call is made, and every lifecycle error is logged and swallowed so bookkeeping can never abort a run.

## API Surface

`policy` (pure helpers, no state, no IO):
- `AFFINITY_PARENT_SESSION_ID_ENV` = `"kv_cache_affinity_parent_session_id"` (mirrors `KV_CACHE_AFFINITY_PARENT_SESSION_ID_ENV`).
- `STICKY_SUBAGENT_TYPE` = `"verification_agent"`; `browser_agent` is deliberately non-sticky because browser process/profile reuse belongs to the browser registry.
- `KvCacheAffinityConfig { enable_kv_cache_affinity: bool }` (serde, snake_case, default-tolerant).
- `affinity_enabled(Option<&KvCacheAffinityConfig>) -> bool` — true only for the literal `true`; missing/default config means off.
- `is_sticky_subagent_type(&str) -> bool` — trimmed equality with `STICKY_SUBAGENT_TYPE`.
- `KvCacheLineage` (`resolve_parent_cache_id(&self, runtime_session_id: &str) -> Result<String>`) and the default `RuntimeSessionLineage` (identity).
- `resolve_subagent_parent_cache_id(Option<&dyn KvCacheLineage>, &str) -> String` — lineage result, degrading to the runtime id with a warning on error or a blank result.
- `scope_sub_session_id(sub_session_id, runtime_parent_session_id, parent_cache_id) -> String` — unchanged when the ids match, else `{sub_session_id}_scope_{sha256(parent_cache_id)[..12]}`.
- `resolve_sub_session_id(task_id, parent_session_id, metadata_sub_session_id) -> String` — metadata wins, else `{parent_session_id}_sub_{task_id or "unknown"}`.

`provider` (the seam the lifecycle drives):
- `KvCacheProviderRegistry` (`#[async_trait]`, object-safe): `backend() -> &str`, `prepare(cache_id)`, `suspend(cache_id)`, `release(cache_id)`.
- `UnsupportedKvCacheBackend::new(backend)` + `backend_name()`; every operation returns `ErrorKind::Unsupported` with the fixed message ``kv_cache: `<backend>` provider backend is not implemented; KV-cache prefetch/offload and distributed (Redis/cluster) backends are deferred``.

`rail` (state + the harness adapter):
- `KvCacheAction { Prepare, Suspend, Release }`, `KvCacheSubagentState { Active, Suspended, Released }`, `KvCacheSubagent { subagent_id, subagent_type, sub_session_id, parent_cache_id }`, `KvCacheSubagentSession` (the same identity plus `state` and the ordered `actions` history) — all serde snake_case.
- `KvCacheSubagentLifecycle::new(Arc<dyn KvCacheProviderRegistry>)` (affinity off), `with_config`, `with_lineage`, `affinity_enabled`, `registry`, `sessions`, `get`, `prepare_subagent`, `finish_subagent`, `evict_subagent`, `release_all`.
- `KvCacheLifecycleRail::new(Arc<KvCacheSubagentLifecycle>)` + `lifecycle()`; `impl HarnessRail` with name `"kv_cache_lifecycle"`, priority `rail_priority::DEFAULT` (50), driving `after_tool_call` only.

## Constraints

- `openjiuwen_common::error::{Error, Result}` everywhere; `tracing` only; no stdout; no `unwrap`/`expect` outside tests; no panics on user input.
- No lock is held across `.await`: every lifecycle method takes a short `parking_lot::Mutex` section before the provider call and another to record the result.
- `src/rails` must not import `crate::tools`, `crate::agent` or `crate::subagents`; this module reads tool payloads as opaque JSON instead.
- Digests come from `crate::rails::sha256::{sha256_hex, sha256_prefix}`; `sha2` is not a dependency of the crate and must not be added.
- State is per `KvCacheSubagentLifecycle` instance; no module-level singletons or caches.
- Unknown subagent ids and provider failures never surface as panics: direct calls return `Err`/`Ok(None)`, the rail logs and swallows them.
- The module carries `#![allow(unreachable_pub, dead_code, unused_imports)]` because `rails/mod.rs` (not editable from this subtree) still has to add the flat `pub use` re-exports; remove the allow when that lands.

## Routing Table

- `./mod.rs` → module docs, submodule declarations and the flat re-exports.
- `./policy.rs` → affinity config, sticky types, lineage seam, parent cache id resolution and sub-session id scoping/naming.
- `./provider.rs` → `KvCacheProviderRegistry` and the typed `UnsupportedKvCacheBackend`.
- `./rail.rs` → action/state/session types, `KvCacheSubagentLifecycle` (state + the prepare/finish/evict/release-all mapping) and the `KvCacheLifecycleRail` adapter.

## Design Decisions

- Python resolves the parent cache identity from a live `Session` through `core.kv_cache.kv_cache_metadata`; the Rust port cannot reach that kernel metadata store from `rails`, so lineage is a `KvCacheLineage` seam with an identity default, and the rail resolves it from `ctx.base.session.id()`.
- Python's `create_subagent_session` (sharing the parent `kv_cache_runtime`) is not ported: the Rust kernel session has no `prepare_kvc/suspend_kvc/release_kvc` or runtime object, so the lifecycle records the child identity (`sub_session_id`, `parent_cache_id`) instead and calls the provider seam directly.
- Concurrency: Python's session is per turn; the Rust tracker is shared, so each transition is read → provider call → record, and `evict_subagent` always calls `release` again (the provider seam is required to be idempotent).
- The rail observes `subagent_spawn`/`task_tool`/`task` (prepare), `subagent_wait` (finish per terminal status) and `subagent_close` (evict), because the runtime lives in `crate::subagents`; the payload shapes are read tolerantly (data envelope, `subagent_id` or `agent_id`, type from the payload or the call arguments).
- Terminal status mapping trusts the runtime: `completed` → succeeded; `failed`/`errored`/`closed`/`cancelled`/`interrupted` → not succeeded; `running`/`pending_init`/`not_found`/unknown are left for a later wait.
- A failing `prepare` leaves no state behind, so a broken provider cannot strand a phantom `Active` session.

## Test Strategy

- Inline `#[cfg(test)] mod tests` per file with `#![allow(clippy::unwrap_used, clippy::expect_used)]`; offline and deterministic.
- `policy.rs`: off-by-default affinity, trim-only stickiness (`browser_agent` false), scope digest length/stability/distinctness, metadata-vs-`"unknown"` sub-session ids, lineage success/blank/failure degradation, config serde with unknown fields.
- `provider.rs`: the unsupported backend names itself and every operation returns the exact documented `Unsupported` message.
- `rail.rs`: a recording provider fake (a `Mutex<Vec<String>>` of `prepare:/suspend:/release:<cache_id>` and a failure switch) covers OFF inertness, sticky prepare→suspend vs release-on-failure, non-sticky release, failing prepare, idempotent evict, unknown ids, `release_all` skipping released sessions, action/state serde, and the rail's spawn/wait/close/ignored-frame/swallowed-error/inert paths.
- Run with `cargo test -p openjiuwen-harness --offline --lib rails::kv_cache`.

## Deferrals

- Redis/cluster/distributed KV-cache backends and real provider prefetch/offload are deferred: `UnsupportedKvCacheBackend` reports them with a typed `Error::Unsupported`.
- The kernel session has no `prepare_kvc/suspend_kvc/release_kvc` (nor a `get_kv_cache_runtime` object), so `create_subagent_session` and the per-session binding live outside this node.
- `openjiuwen/core/kv_cache/` (identity/lineage metadata, `resolve_session_lineage`) is not reimplemented here; a host wires `KvCacheLineage` to it.
