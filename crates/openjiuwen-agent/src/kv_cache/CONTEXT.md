# kv_cache (openjiuwen-agent)

## Intent

Provider-facing prefix-cache (`KV`-cache) affinity primitives for the agent kernel: stable request identities, session lineage, reuse-range metadata, an in-process runtime that retains the prefixes an agent has already sent, and the model-call hook that reports how much of the next request is reusable.
Ports `openjiuwen/core/kv_cache/` (`kv_cache_types.py`, `kv_cache_config.py`, `kv_cache_metadata.py`, `kv_cache_runtime_config.py`, `kv_cache_runtime.py`, `kv_cache_model_hook.py`) plus `openjiuwen/core/single_agent/kv_cache/` (`kv_cache_react_model_call_hook.py`, `kv_cache_child_session.py`) onto `openjiuwen-foundation` messages/tools.
Affinity is opt-in and best-effort: a disabled config performs no runtime work, and every recorded prefix is an optimization hint, never a correctness requirement.

## API Surface

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
- `KVCacheModelCallHook` — `new(runtime, config)`, `runtime()`, `config()`, `capabilities()`, `identity(...)`, `affinity(...)`, `note_usage(&identity, tokens)`, `stats()`, `build_child_session_kwargs(parent_session_id)`.
- `KVCacheCallCapabilities` — `{affinity, offload, evict}`; `offload` is always `false` because provider offload is deferred.

## Design Decisions

- The runtime is keyed by the full `KVCacheIdentity` (exact-match `HashMap`), so a prefix is reusable only when its lineage, model, counts, and hash all match.
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

## Constraints

- Only this module is edited; `src/lib.rs` already declares `pub mod kv_cache;` and the rest of the crate is read-only.
- No lock is held across an `await`: the in-process runtime locks `parking_lot::Mutex<State>` inside each protocol method and returns before yielding.
- No panics on input: empty session ids and empty message lists return a miss; hash serialization uses infallible `Value` display.
- No new dependencies; hashing, duration serde, and the runtime are implemented in-crate.
- Deferred capability paths are typed: `KVCacheBackend::runtime` for `Redis`/`Distributed` returns `Error::Unsupported("kv-cache backend '<name>' is not implemented; use the in-process runtime")`.

## Notes for Agents

- The hook is stateless: all prefix state lives in the shared runtime, so a child session that inherits the parent's runtime sees the parent's prefixes.
- `build_child_session_kwargs` returns only `{parent_session_id, envs:{kv_cache_affinity_parent_session_id}}`; the runtime handle is inherited as an `Arc` host-side and has no JSON form (Python's `kv_cache_runtime` key).
- `note_usage` records the caller's identity with `prefix_len = identity.message_count`; call it with the identity produced by `KVCacheModelCallHook::identity`, not with a shorter matched-prefix identity.
- `KVCacheLookup.entry` is present only on a hit; `reused_prefix_len` comes from the matched entry, so it equals the previous request's message count.
- `KVCacheAffinityConfig::max_entries` is the capacity the host should pass when building an in-process runtime; it is not enforced by the hook.
- `KVCacheBackend::runtime` ignores `config.backend` and uses the receiver, so call `config.backend.runtime(&config)` (or `KVCacheRuntimeConfig::build_runtime`) to honour the configured backend.

## Test Strategy

- Inline `#[cfg(test)]` modules only; offline and deterministic (no network, no wall-clock ordering; `TTL` expiry is exercised with `Duration::ZERO`).
- `types.rs` pins identity JSON round-trip/unknown-field tolerance, entry/lookup helpers, stats arithmetic, backend wire names, and the exact deferred-backend `Error::Unsupported` message.
- `config.rs` pins defaults, builder chaining, timeout resolution, millisecond duration round-trip, and runtime-config deserialization defaults.
- `metadata.rs` pins env-override lineage precedence, blank-value handling, Python `first_changed_index` semantics, range-dict shapes, and the compressor/team-member id contracts.
- `runtime.rs` pins hash stability and sensitivity (messages, tool calls, tools, model), prefix-hash consistency across prefix lengths, hit/miss recording, `LRU` eviction, capacity zero, upsert merge, per-session cascade eviction, `TTL` expiry, `clear`, and concurrent lookups from two tasks.
- `hook.rs` pins the disabled/empty bypass (zero runtime calls through a counting fake), two-turn prefix reuse agreeing with `first_changed_index`, identical-request hits, message/tool/model edits missing, lineage isolation, `note_usage` persistence, capabilities, and child-session kwargs.

## Routing Table

This directory is flat: no child directories; route by file.

- `types.rs` → identity/entry/lookup/stats types, runtime protocol, backend factory.
- `config.rs` → affinity/runtime config, timeout constants, millisecond duration serde.
- `metadata.rs` → env keys, lineage resolution, range helpers, derived identities.
- `runtime.rs` → in-process runtime and the deterministic prefix hash.
- `hook.rs` → model-call affinity hook, capabilities, child-session kwargs.
- `mod.rs` → module documentation and the public re-exports.
