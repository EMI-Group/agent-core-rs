# policy

## Intent

Owns provider context-window resolution and KV-cache diffing for `openjiuwen-context`.
Ports `openjiuwen/core/context_engine/context/context_utils.py` (`resolve_context_max`, `MODEL_DEFAULT_CONTEXT_WINDOW_TOKENS`, the OpenRouter cache and fetch), `base.py` (`ContextWindowChange`) and `context/context.py` (`detect_context_window_change`), plus Python's `model_revision`/`position` diffing.
Resolution answers "how large is the window this model accepts, and who said so?"; diffing answers "how much of the previous prompt is still cache-warm?".
The engine does not consume this module yet; a later pass wires `ContextWindowPolicy` and the diff primitive into `ContextEngine`.

## API Surface

`window.rs` — resolution:
- `ContextWindowPolicy` — serde-defaulted config: `context_window_tokens`, `model_context_window_tokens_override`, `model_context_window_tokens`, `enable_openrouter_model_context_window_tokens` (default `false`), `openrouter_request_timeout` (default 3 s), `openrouter_endpoint`, `model_name`, `model_provider`, `openrouter_cache: Option<Arc<OpenRouterWindowCache>>`.
- Builders `new`/`with_global_tokens`/`with_model_override`/`with_model_window`/`with_openrouter_enabled`/`with_openrouter_timeout`/`with_openrouter_endpoint`/`with_model`/`with_provider`/`with_openrouter_cache`; accessors `cache`, `endpoint`, `openrouter_enabled`, `effective_model`, `effective_provider`; `validate` (zero request timeout → `Error::Config`); `resolve(model, provider) -> ResolvedContextWindow`.
- `ResolvedContextWindow` — `tokens`, `source`, `new`, `is_default`.
- `ContextWindowSource` — `Global` | `Override` | `ConfiguredModel` | `BuiltIn` | `OpenRouterCache` | `Default`, with `as_str`.
- `MODEL_DEFAULT_CONTEXT_WINDOW_TOKENS: &[(&str, usize)]` — Python's hand-maintained table, verbatim, grouped by family; `lookup_model_default_window(model) -> Option<usize>` (exact then case-insensitive, trimming, empty never matches).

`change.rs` — diffing:
- `ContextWindowSnapshot` — `system_messages`, `context_messages`, `tools`, `model_revision`, `position`, `tools_signature`, `message_fingerprints`; `new`, `from_messages`, `with_model_revision`, `with_position`, `set_tools`, `set_context_messages`, `set_system_messages`, `recompute_derived`, `messages`, `message_count`, `system_message_count`.
- `MessageFingerprint` — `digest` (sha256 over role, content text, `name`, `tool_call_id`, and each tool call's id/name/arguments), `of`, `as_str`.
- `tools_signature(tools) -> String` — sha256 over canonical JSON, array order significant.
- `ContextWindowChange` — `Unchanged { messages }` | `Appended { prefix_messages }` | `SystemChanged { index }` | `ToolsChanged { requires_suffix_invalidation }` | `Rewritten { first_divergence }` | `ModelChanged { from_revision, to_revision }` | `PositionChanged { from, to }`; `kind`, `reusable_prefix_len`, `invalidates_cache`, `prompt_cache_impact`.
- `PromptCacheImpact` — `reused_messages`, `invalidated_messages: Option<usize>`, `reason`.
- `detect_context_window_change(previous, current) -> ContextWindowChange`.
- `PositionChange` — `ModelRevision { from, to }` | `Position { from, to, delta }`, with `invalidates_everything`; `diff_positions(previous, current) -> Vec<PositionChange>`.

`openrouter.rs` — cache and fetch:
- `OpenRouterWindowCache` — `new`, `with_ttl`, `insert`, `get`, `contains`, `clear`, `len`, `is_empty`, `set_ttl`, `ttl`, `snapshot`; lowercase keys, per-entry TTL, zero TTL means immediately stale.
- `global_openrouter_window_cache()` — the single process-global cache.
- `OpenRouterModelTable` — `windows` (id and unambiguous provider-stripped alias) and `models` (accepted record count); `parse_openrouter_models(&Value) -> Result<OpenRouterModelTable>`.
- `ContextWindowPolicy::fetch(&[model_id]) -> Result<usize>` and `::prefetch(&[model_id], &HttpClient) -> Result<usize>` (inherent impls living in this file); `openrouter_unavailable_error`, `openrouter_invalid_response_error`, `openrouter_disabled_error`.
- Constants — `OPENROUTER_MODELS_URL`, `DEFAULT_OPENROUTER_CACHE_TTL` (6 h), `DEFAULT_OPENROUTER_REQUEST_TIMEOUT` (3 s), `OPENROUTER_UNAVAILABLE_CODE`, `OPENROUTER_INVALID_RESPONSE_CODE`, `OPENROUTER_DISABLED_CODE`.
- `mod.rs` re-exports everything above; callers use `openjiuwen_context::policy::…`.

## Constraints

- `resolve` is synchronous and offline: it reads the injected or global cache and never issues a request, so context creation can never wait on the network.
- `fetch`/`prefetch` return `Ok(0)` without touching the network while `enable_openrouter_model_context_window_tokens` is `false`; the flag is the only network gate.
- Fetch deadlines come from `openrouter_request_timeout` on both the client and the request; a transport failure is `Error::Transport` with `openrouter_unavailable`, a deadline is `Error::Timeout` with the same code, and an unusable payload is `Error::Validation` with `openrouter_invalid_response`.
- A zero/`None` window override is ignored, never clamped; only a zero request timeout is a config error.
- Hashing uses `sha2`; canonical JSON comes from `crate::usage::context_tokens::canonical_json` (the workspace enables `preserve_order`), so object key order never changes a signature.
- Fingerprints cover only wire-relevant message fields; local metadata such as `context_message_id` never changes a digest.
- Tool definitions serialize immediately after the system prompt, so any tool change reports `ToolsChanged { requires_suffix_invalidation: true }` even when the change only appends; a tool change outranks a message append in the diff.
- `Unchanged` carries the window size; `invalidated_messages` is `None` whenever the diff does not know the old window length, so "definitely zero" is never confused with "unknown".
- `prompt_cache_impact().reason` is human-readable with a stable prefix (`unchanged`, `appended`, `system_changed`, `tools_changed`, `rewritten`, `model_changed`, `position_changed`).
- The process-wide cache is the one documented global mutable state here; its `RwLock` is never held across an `await`, and tests should inject a cache instead of mutating it.
- No `unsafe`, no `println!`, no `unwrap`/`expect` in library code, `tracing` only, every public item documented, every public type serde-able with `#[serde(default)]` and unknown fields tolerated.

## Design Decisions

- The resolution ladder is global → selected-model override → configured map → built-in table → OpenRouter cache → `DEFAULT_CONTEXT_TOKENS` (200 000); provider prefixes only participate in cache lookups (`provider/model` first, then the bare id).
- Case-insensitive matching uses `eq_ignore_ascii_case` (model ids are ASCII), which reproduces Python's `lower()` comparison for real ids without allocating per entry.
- `openrouter_request_timeout` serializes as a number of seconds and also accepts the explicit `{secs, nanos}` shape, so hand-written config and a round-tripped value both load.
- The OpenRouter catalogue is cached in full, not just for the requested ids; `fetch`/`prefetch` return how many *requested* ids the cache covers afterwards, and an empty id list is a no-op.
- Provider-less aliases are added exactly like Python's parser (last write wins, ambiguous aliases dropped) because a model configured without its vendor prefix should still resolve.
- `prefetch` skips the request when every requested id is already fresh; `fetch` always asks, which is what an explicit warm-up wants.
- Diff precedence is model revision → message divergence (system vs rewritten) → tool signature → append → position → unchanged, matching the severity of cache invalidation.
- A position-only change leaves the prompt byte-identical, so it does not invalidate the cache; the label exists for telemetry and for `diff_positions` deltas.
- `reusable_prefix_len` reports the divergence index for `SystemChanged` and `Rewritten`, the old length for `Appended`, and `0` for tool/model changes (the suffix is invalid and the diff does not know the system-message count).

## Routing Table

- `./mod.rs` → module docs, public module tree, crate-root re-exports
- `./window.rs` → `ContextWindowPolicy`, `ResolvedContextWindow`, `ContextWindowSource`, the built-in window table and its lookup, duration serde support
- `./change.rs` → `ContextWindowSnapshot`, `MessageFingerprint`, `tools_signature`, `ContextWindowChange`, `PromptCacheImpact`, `detect_context_window_change`, `PositionChange`, `diff_positions`
- `./openrouter.rs` → `OpenRouterWindowCache`, the process-wide cache, catalogue parsing, `fetch`/`prefetch` and the compact error codes
- `../../tests/policy.rs` → public-API suite: precedence matrix, table lookups, validation, fingerprints, diff matrix, cache TTL, wiremock fetch/prefetch, serde round-trips
