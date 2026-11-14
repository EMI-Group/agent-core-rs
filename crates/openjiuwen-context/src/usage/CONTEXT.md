# usage

## Intent

Owns the accounting seam that explains where a request's prompt tokens went: four prompt categories (`system_prompt`, `tools`, `skills`, `messages`), the provider-total reconciliation, the provider's KV-cache `usage_metadata` normalization, and the session-to-date cache rollup.
Ports the public behavior of Python's `openjiuwen/core/context_engine/usage/` (`models.py`, `analyzer.py`, `provider_usage.py`, `session_aggregator.py`, `context_tokens.py`) to serde value objects plus a pure analyzer.
The analyzer only explains the window the processors already produced: it never decides compression, truncation, or routing, and it never calls a model.
The engine does not call it yet; a later pass wires `ContextUsageAnalyzer`/`SessionKVCacheAggregator` into `ContextEngine`.

## API Surface

`models.rs` — the `context.usage` wire contract (snake_case, `#[serde(default)]`, unknown fields tolerated):

- `ContextCategory` — `SystemPrompt` | `Tools` | `Skills` | `Messages`; legacy `memory` deserializes to `SystemPrompt`, as does anything unknown; `ALL`, `as_str`, `from_name`, `accepts_attachment` (only `system_prompt`/`skills` accept `_context_usage_category`/`_context_usage_fragments`).
- `ContextPartUsage` — `tokens`, `percentage_of_window`, `percentage_of_input`, `estimated`, `tokenizer`, `carrier`, `source`, `measurement_source`, `fallback_reason`, `fallback_tokenizer_model`, `order`, `stable_id`, plus `is_reported`.
- `ContextWindowUsage` — `tokens` (the authoritative total), `tokens_source`, `limit_tokens`, `occupancy_rate`, `local_estimated_input_tokens`, `reconciliation_delta_tokens` (diagnostic only), `context_snapshot_id`, `parts`.
- `RequestKVCacheUsage` — `input_tokens`, `cache_read_tokens`, `cache_miss_tokens`, `cache_write_tokens`, `hit_rate`, `status`, `source`, `authoritative`, `miss_tokens_derived`, `invalid_reason`, `cache_mode`, `cache_scope`, `cache_identity`, `parent_cache_identity`, plus `reports_cache_fields`.
- `SessionKVCacheUsage` — the token-weighted rollup: `scope`, `aggregation_method`, `aggregation_key`, `calls_total`/`calls_observed`/`calls_partial`/`calls_not_reported`, the three token totals, `cache_write_tokens_total` + `cache_write_coverage`, the average/weighted rates, `status`, `data_quality`.
- `ContextUsageMeasurement` — `category_source`, `total_source`, `tokenizer`, `estimated`, `authoritative_total`, `measurement_version`, `context_snapshot_id`, `reconciliation`.
- `ContextUsageSnapshot` — the emitted payload: serde `type` (alias `event_type`) = `context.usage`, `schema_version`, `phase`, request/session/context/agent ids, `sequence`, `provider`, `model`, `deployment`, `timestamp`, `context_snapshot_id`, `context_changes`, `measurement_cache`, `context_window`, `parts`, `kv_cache`, `session_kv_cache_hit_rate`, `measurement`, `context_stats`, plus the attribution fields and `set_attribution`.
- `ContextWindowTokenReport` — the immutable-by-convention per-request report: `context_snapshot_id`, `context_id`, session ids, attribution, the message/prompt/skills/attachment revisions, `tools_signature`, `tokenizer_identity`, `measurement_version`, `component_signatures`, `context_window`, `parts`, `context_stats`, `measurement`, `context_changes`, `measurement_cache`, plus `attribution()`.
- `MeasurementCacheHit` (`miss`|`partial`|`full`), `MeasurementCacheInfo` (`from_parts` sorts/dedups and derives the hit), `KvCacheUsage` (`request`, `session`, `new`, `session_hit_rate`), `Attribution` (`from_report`).
- `CounterIdentity` — the counting strategy (`class`, `tokenizer`, `model`, `fallback_reason`, `fallback_tokenizer_model`, `source`): `new`, `with_tokenizer`, `with_fallback_reason`, `with_fallback_tokenizer_model`, `approx`, `model(&ModelTokenCounter)`, `selected(&SelectedTokenizer)`, `source_name`, `is_fallback`, `identity_string`.
- Constants — `CONTEXT_USAGE_EVENT_TYPE`, `CONTEXT_USAGE_SCHEMA_VERSION`, `PHASE_PRE_CALL`/`PHASE_POST_CALL`/`PHASE_POST_COMPACT`, `NOT_REPORTED`, `MIXED`, `MEASUREMENT_VERSION`, the four `SOURCE_PROVIDER_USAGE*`, the five `STATUS_*`, plus `source_name`/`is_fallback_source`.

`context_tokens.rs` — the shared sizing policy:

- `count_context_window_tokens(messages, tools, counter)`, `count_messages_tokens(messages, counter, usage_aware)`, `count_tools_tokens(tools, counter)`, `count_usage_tokens_with_tail(messages)`, `estimate_content_tokens`, `estimate_message_tokens`, `estimate_text_tokens`, `usage_total_tokens`, `has_valid_usage_metadata`, `canonical_json`.
- Constants `USAGE_METADATA_KEY` (`usage_metadata`), `USAGE_STALE_METADATA_KEY`, `CHARACTERS_PER_ESTIMATED_TOKEN` (3).

`analyzer.rs` — measurement and reconciliation:

- `ContextUsageAnalyzer` — `new(Arc<dyn TokenCounter + Send + Sync>, CounterIdentity)`, `with_model`, `with_context_window_limit`, `with_measurement_cache_capacity`, `model`, `context_window_limit`, `counter_identity`, `measurement_cache_len`, `clear_measurement_cache`, `analyze(&AnalysisInput) -> ContextUsageSnapshot`, `build_report(&AnalysisInput) -> ContextWindowTokenReport`, `analyze_from_report(&ContextWindowTokenReport, &SnapshotOptions) -> ContextUsageSnapshot`.
- `AnalysisInput` — `request_id`, `session_id`, `context_id`, `phase`, `provider`, `agent_id`, `sequence`, `provider_input_tokens`, `system_messages`, `context_messages`, `tools`, `system_prompt_sections`, `kv_cache`, `deployment`, `context_snapshot_id`, `context_changes`, `attribution`, `signatures`, `context_stats`.
- `SnapshotOptions` — `request_id`, `session_id`, `phase`, `provider`, `agent_id`, `sequence`, `provider_input_tokens`, `deployment`, `context_snapshot_id`, `attribution`, `context_stats`, `context_changes`, `kv_cache`; `new(request_id, session_id)` sets the defaults.
- `PromptSection` (`new(category, text)`, `with_carrier`), `WindowSignatures` (revisions, `tools_signature`, `component_signatures`), `MeasurementCache` (`new(capacity)`, `capacity`, `len`, `is_empty`, `get`, `insert`, `clear`), `DEFAULT_MEASUREMENT_CACHE_CAPACITY` (512).
- Attachment metadata keys: `ATTACHMENT_CATEGORY_METADATA_KEY` (`_context_usage_category`), `ATTACHMENT_CARRIER_METADATA_KEY` (`_context_usage_carrier`), `ATTACHMENT_FRAGMENTS_METADATA_KEY` (`_context_usage_fragments`).

`provider_usage.rs`:

- `request_usage_from_metadata(&serde_json::Value) -> RequestKVCacheUsage` plus `LEGACY_CACHE_TOKENS_KEY`, `CACHE_STATUS_KEY`, `CACHE_SOURCE_KEY`, `CACHE_AUTHORITATIVE_KEY`.

`session_aggregator.rs`:

- `CacheAggregationKey` — `session_id`, `provider`, `model`, `deployment`, `cache_mode`, `cache_scope`, `product_session_id`, `execution_session_id`, `cache_identity`, `parent_cache_identity`, `team_id`, `agent_id`, `member_name`; `new(session_id, provider, model)` plus `Default` (`cache_mode = "provider"`, `cache_scope = "session"`); `Ord`/`Hash` so it keys a map.
- `SessionKVCacheAggregator` — `new`, `record(request_id, &CacheAggregationKey, &RequestKVCacheUsage) -> SessionKVCacheUsage`, `snapshot(&CacheAggregationKey)`, `scope_count`, `is_empty`.

## Constraints

- Pure CPU: no I/O, no network, no filesystem, no model call, no clock other than the snapshot timestamp; the analyzer is synchronous and allocation-bound.
- No new dependencies: hashing uses `sha2`, canonical JSON uses `serde_json` with hand-sorted keys (the workspace enables `preserve_order`), and all rounding is integer (`u128`) arithmetic.
- No `unsafe`, no `println!`, no `unwrap`/`expect` in library code; `tracing` only.
- Every public item and field is documented (`missing_docs`), every data type is `serde`-serializable and tolerant of unknown fields, and every numeric field that can be absent is an `Option` rather than a sentinel.
- Percentages and rates are `Option<f64>` and are `None` when the denominator is zero or missing, never `0.0`-as-unknown.
- `RequestKVCacheUsage` never clamps: an inconsistent provider report keeps its raw values, sets `invalid_reason`, `status = invalid`, and `hit_rate = None`.
- Errors are not raised from this module: malformed metadata degrades into a reported `invalid`/`not_reported` usage, and a counter that fails to serialize a message only logs at debug and uses `Null` in the cache key.
- Files stay well under 1000 lines; the analyzer is split into measurement, reconciliation, and percentage helpers.

## Design Decisions

- A measurement is one measurement of one logical input: `system_prompt`/`tools`/`messages` are measured once and reused, logical sections and attachment fragments are measured as whole carriers, and their tokens are split by largest-remainder weight (the remainder always lands on the last fragment).
- `system_prompt_sections` replaces the `system_messages` measurement entirely; `skills` can only come from sections or attachments, never from history. A section or fragment whose category cannot accept an attachment folds into `system_prompt`, so the four categories always account for the whole window.
- Attachment routing is metadata-driven: `_context_usage_category` moves the whole message, `_context_usage_fragments` splits the measured carrier across fragments, and a non-object/absent fragment list falls back to the whole message.
- A `carrier` is `"system_message"`, `"tools"`, `"context_message"`, an explicit section/fragment carrier, or `MIXED` (the `mixed` string) when one category received more than one carrier; a not-reported category reports its carrier but keeps `source = not_reported` and `tokens = 0`.
- Provider reconciliation branches exactly as Python does: a fallback counter scales every non-zero category proportionally (no part is skipped except a zero-token/zero-allocation one), a native counter keeps the non-message categories fixed and puts `provider_total - fixed` on `messages` (floored at 0), and a fallback counter with a zero local total puts the whole provider total on `messages` as `provider_usage_unattributed`.
- `provider_input_tokens` (when set) is the authoritative window total and the percentage denominator; otherwise the local total is both, so `parts` always sums to `context_window.tokens`.
- `reconciliation_delta_tokens` is `provider_total - local_total` and is diagnostic only; nothing downstream uses it to correct a count.
- The measurement cache key is `sha256(class:tokenizer:model:<measurement-vN>:part:canonical_json)`; the cache holds only `TokenMeasurement` values, so processors and window assembly still run on every call and only the counting is memoized.
- The cache is FIFO and bounded per analyzer (`512` by default, `0` disables it); a cache smaller than the number of parts can evict an entry before it is reused, so a repeat pass is only guaranteed to be a full hit when the cache fits the window.
- `analyze_from_report` reuses the report's category values and KV-cache blocks without re-measuring, re-applies the provider reconciliation with the report's fallback category source, and copies the report's measurement-cache info, revisions, `context_changes` and attribution; it does not carry the measured `context_stats` into the snapshot.
- Provider normalization accepts ints, integral finite floats, and digit-only strings; booleans, negatives, fractional values and non-digit strings are rejected with a reason naming the field rather than being coerced.
- A derived miss (`input - read`) is marked `miss_tokens_derived`; a provider verdict of `miss`, `partial_hit` or `full_hit` is trusted as reported, `observed` is never trusted, and otherwise the status is recomputed from what was reported.
- The legacy `cache_tokens` field is used as `cache_read_tokens` only when the modern field is absent/null and the legacy value is non-zero, which keeps "cache not used" from being reported as a zero hit.
- The aggregator is idempotent per `(scope, request id)`: a repeated id returns the current snapshot unchanged and never increments `calls_total`, so a retry cannot double-count a call.
- Only observed calls with consistent numbers enter the totals; incomplete and inconsistent calls are counted in `calls_partial`, calls with no cache fields in `calls_not_reported`, and `data_quality` is `mixed` as soon as two qualities coexist.
- `weighted_hit_rate = read_total / input_total` (token weighted, not a mean of ratios); per-call averages divide by `calls_observed` only.
- `cache_write_tokens_total` is only reported when every observed call reported a write (`cache_write_coverage = complete`); one missing write downgrades the coverage to `partial` and hides the total rather than reporting a misleading sum.
- The session status ladder is `not_reported` (no observed call), `invalid` (only excluded calls), `miss` (no reads), `full_hit` (no misses), `partial_hit` otherwise.

## Routing Table

- `./models.rs` → `ContextCategory`, `ContextPartUsage`, `ContextWindowUsage`, `RequestKVCacheUsage`, `SessionKVCacheUsage`, `ContextUsageMeasurement`, `ContextUsageSnapshot`, `ContextWindowTokenReport`, `KvCacheUsage`, `Attribution`, `CounterIdentity`, wire constants
- `./parts.rs` → `PartMeasurement` (internal), category merging, carrier/source merging, attachment and fragment splitting
- `./context_tokens.rs` → usage baseline policy, category sizing estimators, canonical JSON for cache keys
- `./analyzer.rs` → `ContextUsageAnalyzer`, measurement cache, category assembly, provider reconciliation, percentage application
- `./analyzer_options.rs` → `AnalysisInput`, `SnapshotOptions`, `WindowSignatures`, `PromptSection`
- `./provider_usage.rs` → `request_usage_from_metadata`: strict field parsing, semantic validation, legacy fallback, status derivation
- `./session_aggregator.rs` → `CacheAggregationKey`, `SessionKVCacheAggregator`: classification, token-weighted rollup, write coverage, status ladder
- `../../tests/usage.rs` → the integration suite for all of the above
