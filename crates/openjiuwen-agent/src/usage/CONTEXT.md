# usage/
## Intent
The agent's seam onto `openjiuwen-context`'s usage/telemetry surface: analyzer and recorder ownership, the identity helpers of one provider usage report, and the drained snapshot buffer a non-streaming caller reads.
Measurement itself stays in `openjiuwen-context`; this module never counts tokens, talks to a provider, or decides compression.
Ports the agent-side wiring of Python's `context_usage`/KV-cache reporting, not the analyzer.
## API Surface
- `MAIN_CONTEXT_ID` (`"main"`) — the context id the `ReAct` loop creates its conversation window under; `usage_report` reads the same id.
- `MAX_TELEMETRY_EVENTS` (128) — the cap of the undrained snapshot buffer.
- `default_usage_analyzer() -> ContextUsageAnalyzer` — foundation's `ApproxTokenCounter` with `CounterIdentity::approx`, no model and no window limit (the engine substitutes the window it resolves).
- `cache_scope_key(session_id, provider, request: &ModelRequestConfig) -> CacheAggregationKey` — the `(session, provider, model)` KV-cache scope; the model name comes from the request, so the scope matches what the provider was asked for.
- `request_id(session_id, iteration) -> String` — the deterministic `"<session>#<iteration>"` id one model call is recorded under (the engine rejects blank ids only).
- `provider_usage_value(&Usage) -> Value` — the `usage_metadata` object `ContextEngine::record_provider_usage` parses; an unserializable usage degrades to `null` ("not reported").
- `UsageTelemetry` — cloneable, `Arc`-shared buffer: `new`, `record(snapshot)`, `drain() -> Vec<ContextUsageSnapshot>`, `last() -> Option<ContextUsageSnapshot>`, `len`, `is_empty`, `recorded_total() -> u64`, `clear()`.
## Semantics
- `UsageTelemetry::clone` shares one buffer with one recorded counter; `drain` empties the buffer but `recorded_total` keeps counting for the lifetime of the buffer.
- `record` drops the oldest snapshot once `MAX_TELEMETRY_EVENTS` is reached, so a long-lived agent cannot grow it without bound; the dropped events stay counted.
- `drain` returns oldest first; `last` never drains, so a caller can poll and still drain later.
- The loop pushes one snapshot per successful model call when an analyzer is installed and nothing otherwise, because an engine without an analyzer cannot measure a window.
## Constraints
- No token counting, no network, no provider knowledge: every value is either from `openjiuwen-context` or a pure function of the run's own identifiers.
- `ApproxTokenCounter` comes from `openjiuwen_foundation::prompt`; the analyzer types come from `openjiuwen_context`.
- The buffer is per agent, never global; it is held as `Arc<UsageTelemetry>` so a spawned streaming loop can record into it.
- Files stay small; unit tests live in `tests.rs` beside the module.
## Test Strategy
- `tests.rs` covers the identity helpers (`cache_scope_key` defaults, `request_id` determinism, `provider_usage_value` round-trip through the engine's parser), the analyzer defaults, and the buffer (record/last/drain, clone sharing, the cap, `clear`, and the serialized snapshot shape).
- The loop-level behaviour (frames, buffered snapshots, KV-cache rollup, `usage_report`) is pinned by `tests/usage.rs`.
## Routing Table
- `./mod.rs` → the seam: constants, identity helpers, `UsageTelemetry`.
- `./tests.rs` → unit tests of the helpers and the buffer.
