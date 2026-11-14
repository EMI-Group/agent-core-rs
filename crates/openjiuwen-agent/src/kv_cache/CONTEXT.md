# kv_cache (openjiuwen-agent)

## Intent

Provider-facing prefix-cache (`KV`-cache) affinity primitives for the agent kernel: stable request identities, session lineage, reuse-range metadata, an in-process runtime that retains the prefixes an agent has already sent, the model-call hook that reports how much of the next request is reusable, and the provider-action executor that runs prefetch/offload/evict around a session's inference.
Ports `openjiuwen/core/kv_cache/` (`kv_cache_types.py`, `kv_cache_config.py`, `kv_cache_metadata.py`, `kv_cache_runtime_config.py`, `kv_cache_runtime.py`, `kv_cache_model_hook.py`) plus `openjiuwen/core/single_agent/kv_cache/` (`kv_cache_react_model_call_hook.py`, `kv_cache_child_session.py`) onto `openjiuwen-foundation` messages/tools.
Affinity is opt-in and best-effort: a disabled config performs no runtime work, and every recorded prefix is an optimization hint, never a correctness requirement.
Provider actions are explicit trait objects (`KvCacheActionProvider`) instead of Python's `model.*_kvc` methods, and they never block or fail an inference: every provider error, timeout, or stall fails open.

## API Surface

### Prefix bookkeeping and affinity

- `KVCacheIdentity` — `{session_id, parent_session_id, prefix_hash, model, message_count, tool_count}`, `Debug + Clone + Default + PartialEq + Eq + Hash + Serialize + Deserialize`, `snake_case` wire names, unknown fields tolerated.
  Builders: `new(session_id, prefix_hash, model)`, `with_parent`, `with_counts`; accessors `parent`, `is_root`.
- `KVCacheEntry` — `{identity, prefix_hash, prefix_len, token_estimate, last_used_seq, hits}`; `new`, `with_token_estimate`, `with_hits`, `bumped`.
- `KVCacheLookup` — `{hit, entry, reused_prefix_len}`; `miss()`, `hit(entry, reused_prefix_len)`, `is_hit()`.
- `KVCacheStats` — `{entries, hits, misses, evictions}`, `lookups()`.
- `KVCacheRuntimeProtocol` — async `record(entry)`, `lookup(&identity)`, `evict_session(session_id) -> usize`, `stats()`.
- `KVCacheBackend` — `InProcess | Redis | Distributed`; `as_str`, `Display`, and `runtime(&KVCacheRuntimeConfig) -> Result<Arc<dyn KVCacheRuntimeProtocol>>`.
- `KVCacheAffinityConfig` — `enabled`, `max_entries`, `max_attempts`, `action_timeout`, `evict_timeout`, `offload_prefetch_timeout`, `terminal_cleanup_timeout` (durations as whole milliseconds on the wire), `enabled()`/`is_enabled()`, `with_*` builders, `effective_*_timeout()`.
- Constants — `KVC_MANAGEMENT_MAX_ATTEMPTS` (`u32 = 1`), `KVC_RANGE_ACTION_TIMEOUT_SECONDS` (`1.5`), `KVC_SESSION_OFFLOAD_PREFETCH_TIMEOUT_SECONDS` (`2.0`), `KVC_SESSION_EVICT_TIMEOUT_SECONDS` (`3.0`), `KVC_TERMINAL_CLEANUP_TIMEOUT_SECONDS` (`5.0`), `DEFAULT_KV_CACHE_CAPACITY` (`256`).
- `resolve_kvc_action_timeout(&KVCacheAffinityConfig) -> Option<Duration>` — `None` when affinity is off.
- `KVCacheRuntimeConfig` — `{backend, capacity, ttl, affinity}`, `Default`, `with_*` builders, `build_runtime()`.
- `InProcessKVCacheRuntime` — `new(capacity)`, `with_ttl(capacity, ttl)`, `capacity()`, `len()`, `is_empty()`, `entries()` (most recently used first), `clear()`, plus the protocol impl.
- `stable_prefix_hash(messages, tools, model) -> String` — 16 lowercase hex chars.
- `resolve_session_lineage(session_id, parent, env) -> (Option<String>, Option<String>)`, `first_changed_index(previous, current) -> usize`, `message_range_kwargs(from_index)`, `tools_range_kwargs(tool_names)`, `self_parent_kwargs(cache_id)`.
- `context_compressor_cache_identity(session_id, parent, message_count, tool_count, model)`, `team_member_cache_id(team_session_id, team_id, member_id)`, `team_member_cache_identity(team_session_id, team_id, member_id, model)`.
- `KV_CACHE_AFFINITY_SESSION_ID_ENV` (`kv_cache_affinity_session_id`), `KV_CACHE_AFFINITY_PARENT_SESSION_ID_ENV` (`kv_cache_affinity_parent_session_id`), `CONTEXT_COMPRESSOR_CACHE_SUFFIX` (`compressor`).
- `KVCacheModelCallHook` — `new(runtime, config)`, `with_action_runtime(Arc<KvCacheActionRuntime>)`, `action_runtime()`, `runtime()`, `config()`, `capabilities()`, `identity(...)`, `affinity(...)`, `note_usage(&identity, tokens)`, `stats()`, `build_child_session_kwargs(parent_session_id)`, and the lifecycle wrappers `prepare`, `suspend`, `release`, `begin_inference`, `end_inference`.
- `KVCacheCallCapabilities` — `{affinity, offload, evict}`; `offload` is `true` only when an action runtime is attached and `config.enabled`.

### Provider actions (`actions.rs`, `backend.rs`, `action_runtime/`)

- `KvCacheAction` — `Prefetch | Offload | Evict`; wire names `"prefetch"|"offload"|"evict"`, `as_str()`, `Display`.
- `KvCacheActionRequest` — `{cache_id, parent_cache_id, target, model}`, `Default` with `target = "session"`; `new(cache_id, parent_cache_id)`, `with_target`, `with_model`.
- `KvCacheActionProvider` (`#[async_trait]`, `Send + Sync`) — `prefetch/offload/evict(&KvCacheActionRequest) -> Result<bool>`, `supports() -> bool` (default `true`), and the provided `perform(action, &request)` dispatcher.
- `KvCacheActionRecord` — `{action, request}`, `new`, `cache_id()`.
- `KvCacheActionStats` — `{prefetch_succeeded, offload_succeeded, evict_succeeded, failures, skipped}`, `succeeded(action)`; the runtime bumps the counters.
- `InProcessKvCacheActionProvider` — `new()`/`Default` (every action succeeds), `with_result(action, bool)`, `with_error(action, message)` (that action returns `Error::transport(message)`), `without_support()` (actions answer `Ok(false)`); recording `history()`, `actions()`, `len()`, `is_empty()`, `calls(action)`, `clear_history()`, and `subscribe() -> broadcast::Receiver<KvCacheActionRecord>` (capacity 64; a send with no receivers is ignored).
- `KvCacheActionRuntime` — `new(Arc<dyn KvCacheActionProvider>, KVCacheAffinityConfig)`, `provider()`, `config()`, `stats()`, `binding(&identity) -> Option<KvCacheBinding>`, `admission(&identity) -> Option<KvCacheAdmission>`, `fail_open(&identity) -> bool`, `active_inference(&identity) -> usize`, `register_binding(&identity) -> Option<KvCacheBinding>`, `prepare`, `suspend`, `release`, `begin_inference(&identity) -> Option<KvCacheInferenceLease>`, `end_inference(&lease, succeeded)`.
- `KvCacheAdmission` — `Open | Blocked | Terminal`; `KvCacheResidency` — `Resident | Offloaded | Unknown` (default `Unknown`).
- `KvCacheBinding` — `{cache_id, parent_cache_id, model, revision, residency}` snapshot.
- `KvCacheInferenceLease` — `Debug`, `is_released()`; fields private, released once.
- `KV_CACHE_ACTION_TARGET_SESSION` (`"session"`).

## Semantics

- Identity mapping: the cache id is `identity.session_id.trim()`; the parent cache id is the trimmed parent, falling back to the cache id when absent or blank.
  A blank cache id makes every method inert (`false`/`None`).
- Every lifecycle method (`prepare`, `suspend`, `release`, `begin_inference`) registers the identity it is called with, so a host may call them standalone; `register_binding` additionally bumps `revision`, and a lineage move bumps it too.
  `begin_inference` registers as well, so `prepare`/`end_inference` revision guards see the newest registration.
- Bindings live in `bindings: HashMap<cache_id, BindingState{parent, model, revision, residency}>`, indexed by `root_index: parent -> {cache_id}`; per-scope state is keyed by `Binding(cache_id)` or `Root(parent_cache_id)`.
- `prepare` opens a scope: for a child identity it prepares `Binding(cache)` plus the lineage root when that root is `Blocked` (root first); a root identity prepares `Root(cache)`.
  A pending offload whose provider call has not started is cancelled (no provider call, one `skipped`); a started offload is left to finish and the prefetch is ordered behind it.
  A prefetch is enqueued only when the scope is blocked, a related scope is blocked, or a covered binding is not `Resident`; an identical prefetch already in flight yields `false`.
  `prepare` returns once the provider call starts (bounded by `effective_offload_prefetch_timeout`), not when it finishes.
- `suspend` sets admission `Blocked` and enqueues an offload; a `fail_open` scope is reopened and skipped, and a `Blocked` scope with no pending action whose bindings are all `Offloaded` is skipped (both count one `skipped`).
- `release` sets admission `Terminal`, enqueues an evict, awaits the enqueued outcomes bounded by `effective_offload_prefetch_timeout + effective_evict_timeout`, returns whether any reported success, then forgets the binding (the whole lineage for a root) and drops its ordering lock.
- `begin_inference` returns `None` when affinity is disabled, the identity is blank, or the scope is `Terminal`; otherwise it waits out blocked scopes over two rounds and, if preparation failed, fails those scopes open before admitting.
- `end_inference` releases the lease once (`AtomicBool`), decrements both counters with saturation, sets the binding's residency to `Resident` on success and `Unknown` on failure, and notifies idle waiters.
- Actions run in a spawned task that takes the **per-lineage ordering lock** (one `tokio::sync::Mutex<()>` per parent cache id, shared by root and binding actions), waits for `active_inference == 0` for offload/evict (bounded by the offload or evict budget), snapshots the covered bindings, marks the provider call started and reopens a prefetched scope, then calls `provider.perform` under the action budget.
  The first snapshot supplies the request (root actions use `cache_id = parent_cache_id = parent` and the first binding's model; an empty model becomes `None`).
- Results apply only to bindings whose `revision` still matches the snapshot: `Offloaded` for a successful offload, `Resident` for a successful prefetch, `Unknown` for a successful evict; failures never change residency.
  A failure sets the scope's `fail_open` and bumps `failures`; a success bumps the matching per-action counter.
- Fail-open is the invariant that inference never waits on cache management: provider errors, timeouts, an empty binding snapshot, and an unreachable idle state are logged with `tracing::warn!`, counted, and never returned.
- A disabled `KVCacheAffinityConfig` makes `prepare`/`suspend`/`release` return `false` and `begin_inference` return `None` without registering or spawning anything; `register_binding` (bookkeeping only) and the introspection accessors stay available.
- The hook's wrappers return `false`/`None` when no action runtime is attached; `end_inference` is then a no-op (a lease can only come from an attached runtime).

## Design Decisions

- The prefix runtime is keyed by the full `KVCacheIdentity` (exact-match `HashMap`), so a prefix is reusable only when its lineage, model, counts, and hash all match.
- `affinity` finds a reusable prefix by probing message-prefix lengths longest-first (`messages.len()` down to `1`), hashing an append-only conversation's previous request with the same hash function.
  `prefix_hash_states` computes every prefix hash from one canonical stream, so a probe pass costs one message projection per message rather than one per candidate.
- `record` is an upsert: it stamps `last_used_seq` from the runtime sequence, keeps the greater `hits` and `token_estimate`, and evicts least-recently-used entries until the entry count fits `capacity`.
- `capacity == 0` disables retention (every record evicts immediately); `ttl` drops entries idle for at least the configured duration.
- `evict_session` removes entries owned by the session and entries whose `parent_session_id` is the session, so a root eviction cascades to team members and compressors.
- Statistics count runtime lookups, not agent requests: a hit on the second turn of a two-message conversation sees two misses (full prefix, then shorter) followed by one hit on the next turn.
- The prefix hash is `FNV-1a` (64-bit) over a hand-built canonical JSON stream of `model`, messages (role, serialized content, name, tool calls, tool-call id), and tools (name, description, input schema); `std`'s randomized `Hasher` and external hashing crates are deliberately avoided so hashes are stable across runs and platforms.
- Provider-only message metadata and reasoning text are excluded from the hash; multimodal content is hashed through its serialized form, so image/file parts change the hash.
- Derived identities (`context_compressor_cache_identity`, `team_member_cache_identity`) are lineage-only with an empty `prefix_hash`; the hook fills the concrete hash when it records.
- Duration fields serialize as whole milliseconds through a local `serde` helper (no new dependency); sub-millisecond precision is intentionally dropped.
- `resolve_kvc_action_timeout` returns `None` for a disabled config rather than a default budget, so a disabled host never schedules cache management.
- Action coordination uses `watch` channels (started/outcome), a shared `tokio::sync::Notify` for idle transitions, and one async `Mutex` per lineage for ordering; there are no control domains and no fallback binding provider, so two sessions that share a provider but differ in model/provider namespace share ordering and stats.
- The action executor is a separate seam from `KVCacheRuntimeProtocol`: prefix bookkeeping answers "how much is reusable", the action runtime answers "where does the provider hold it".

## Constraints

- Only this module is edited; `src/lib.rs` already declares `pub mod kv_cache;` and the rest of the crate is read-only.
- No `parking_lot` lock is held across an `await`: state is read or mutated inside a short critical section, the guard is dropped, and only then does the code await or spawn.
- No panics on input: blank identities are inert, counters saturate, and an aborted action is simply dropped.
- No new dependencies; hashing, duration serde, the in-process runtimes, and the action state machine are implemented in-crate (`tokio`, `futures`, `serde`, `parking_lot`, `tracing`, `async-trait` only).
- Deferred capability paths are typed: `KVCacheBackend::runtime` for `Redis`/`Distributed` returns `Error::Unsupported("kv-cache backend '<name>' is not implemented; use the in-process runtime")`.

## Notes for Agents

- The hook's prefix methods are stateless: all prefix state lives in the shared runtime, so a child session that inherits the parent's runtime sees the parent's prefixes.
- `build_child_session_kwargs` returns only `{parent_session_id, envs:{kv_cache_affinity_parent_session_id}}`; the runtime handle is inherited as an `Arc` host-side and has no JSON form (Python's `kv_cache_runtime` key).
- `note_usage` records the caller's identity with `prefix_len = identity.message_count`; call it with the identity produced by `KVCacheModelCallHook::identity`, not with a shorter matched-prefix identity.
- `KVCacheLookup.entry` is present only on a hit; `reused_prefix_len` comes from the matched entry, so it equals the previous request's message count.
- `KVCacheAffinityConfig::max_entries` is the capacity the host should pass when building an in-process runtime; it is not enforced by the hook or the action runtime.
- `KVCacheBackend::runtime` ignores `config.backend` and uses the receiver, so call `config.backend.runtime(&config)` (or `KVCacheRuntimeConfig::build_runtime`) to honour the configured backend.
- `prepare` succeeding means "the scope is ready to admit inference", not "the provider finished prefetching"; use the counters (`stats()`) or `binding(...).residency` to observe completion.
- `admission`/`fail_open`/`active_inference` report the identity's primary scope: `Root(cache_id)` for a root identity, `Binding(cache_id)` for a child, so a child's root-scope state is not visible through them.
- A root action (`Root(parent)`) covers every binding indexed under that parent, ordered by cache id, so a root offload marks the whole lineage `Offloaded` and a root eviction forgets it all.
- Lease results are revision-guarded: re-registering a binding (`register_binding`, and therefore `begin_inference`) while an action runs discards that action's residency result.
- The in-process action provider is the test/host default; a real provider implements `KvCacheActionProvider` (its HTTP transport stays deferred, as does the `Redis`/`Distributed` prefix bookkeeping).

## Test Strategy

- Inline `#[cfg(test)]` modules only; offline and deterministic (no network, no wall-clock ordering; `TTL` expiry is exercised with `Duration::ZERO`, action ordering via channels and bounded `yield_now` polls).
- `types.rs` pins identity JSON round-trip/unknown-field tolerance, entry/lookup helpers, stats arithmetic, backend wire names, and the exact deferred-backend `Error::Unsupported` message.
- `config.rs` pins defaults, builder chaining, timeout resolution, millisecond duration round-trip, and runtime-config deserialization defaults.
- `metadata.rs` pins env-override lineage precedence, blank-value handling, Python `first_changed_index` semantics, range-dict shapes, and the compressor/team-member id contracts.
- `runtime.rs` pins hash stability and sensitivity (messages, tool calls, tools, model), prefix-hash consistency across prefix lengths, hit/miss recording, `LRU` eviction, capacity zero, upsert merge, per-session cascade eviction, `TTL` expiry, `clear`, and concurrent lookups from two tasks.
- `hook.rs` pins the disabled/empty bypass (zero runtime calls through a counting fake), two-turn prefix reuse agreeing with `first_changed_index`, identical-request hits, message/tool/model edits missing, lineage isolation, `note_usage` persistence, capabilities with and without an attached action runtime, the lifecycle wrappers (delegating and no-runtime no-ops), and child-session kwargs.
- `actions.rs` pins action/request/stats/record wire shapes, unknown-field tolerance, and the per-action counters.
- `backend.rs` pins the recorded call order, configured outcomes and `Error::transport` mapping, the unsupported-provider answer, broadcast delivery, and that a send without receivers is not an error.
- `action_runtime/tests.rs` drives the state machine with a gated/delaying/failing test provider: a prefetch that opens the scope before it completes, an offload that waits for idle, the already-offloaded skip, cancellation of a queued offload, fail-open after a provider error, the revision guard against a re-registered binding, an action timeout becoming a failure, the in-process prefetch/offload/evict sequence plus the broadcast subscriber, the root cascade, lease idempotence, a disabled config being inert, and identity trimming/lineage moves.

## Routing Table

Route by file at this level; `action_runtime/` is the one child directory.

- `types.rs` → identity/entry/lookup/stats types, runtime protocol, backend factory.
- `config.rs` → affinity/runtime config, timeout constants, millisecond duration serde.
- `metadata.rs` → env keys, lineage resolution, range helpers, derived identities.
- `runtime.rs` → in-process prefix runtime and the deterministic prefix hash.
- `hook.rs` → model-call affinity hook, capabilities, lifecycle wrappers, child-session kwargs.
- `actions.rs` → action enum, request, provider trait, record, counters.
- `backend.rs` → deterministic in-process action provider with recorded/broadcast history.
- `action_runtime/mod.rs` → public admission/residency/binding/lease types and the `KvCacheActionRuntime` lifecycle API.
- `action_runtime/engine.rs` → binding/lineage index, per-scope admission state, action queue, per-lineage ordering, provider task, revision-guarded result application.
- `action_runtime/tests.rs` → `#[cfg(test)]` unit tests of the runtime (gated/delaying/failing provider).
- `mod.rs` → module documentation and the public re-exports.
