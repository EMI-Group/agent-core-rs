# openjiuwen-context/src/telemetry — compression telemetry

## Intent

Owns the observability side of the compression pipeline: the `context.compression_state` record, the per-processor recorder that builds and fans it out, and the task-local request-purpose correlation scope for processor model calls.
Ports `schema/context_state.py`, `context/processor_state_recorder.py` and `context/compression_scope.py`.
Every token number the record publishes says whether it is a count or an estimate and, when it can, which strategy produced it, so a reader never has to guess.
Nothing here changes a history, calls a model, or blocks; `emit` is a side channel that can fail only into a log line.
`processor::ContextPipeline` owns an optional recorder and brackets every processor pass with it (`run_recorded`), passing `stats_before`/`stats_after` computed with the processor's own `cx.counter` and flagged with the recorder's declared provenance.

## API Surface

- `state` — the wire contract.
  - `CONTEXT_COMPRESSION_STATE_TYPE` (`"context.compression_state"`), the frame type and the record's `type` tag.
  - `ContextCompressionState` — `state_type` (serde `type`), `operation_id`, `status`, `phase`, `processor`, `model`, `before`, `after`, `statistic`, `saved`, `modified_messages`, `compression_usage`, `duration_ms`, `context_max`, `summary`, `compact_summary`, `error`, `reason`.
  - `CompressionStateStatus` — `started`/`completed`/`noop`/`skipped`/`failed`/`paused`/`cancelled`, with `as_str`, `is_terminal`, `is_success`; it is **not** `events::CompressionStatus`.
  - `CompressionStatePhase` — `add_messages`/`get_context_window`/`active_compress`.
  - `ContextCompressionMetric` (`time`, `messages`, `tokens`, `context_percent`, `estimated`, `tokens_source`), `ContextCompressionSaved` (signed `messages`/`tokens`, `percent`, `estimated`).
  - `ContextCompressionUsage` — `calls`, token counters, three costs, `model_name`, free-form `details`; `from_usage`, `record`, `add`, `with_costs`, `with_model_name`, `with_detail`.
  - `ContextCompressionModifiedMessage` — `message_id`, `role`, `tool_call_id`, `offload_handle`, `offload_type`.
  - Convenience constructors `ContextCompressionState::{new, started, completed, noop, failed}` plus `with_*` setters for every non-derived field.
- `recorder` — `ContextProcessorStateRecorder`, its inputs, and the provenance vocabulary.
  - `ContextProcessorStateInput` — `operation_id`, `status`, `phase`, `processor`, `model`, `context_max`, `before`, `after`, `stats_before`, `stats_after`, `usage`, `error`, `reason`, `summary`, `compact_summary`, `started_at`, `ended_at`; `new` plus `with_after`/`with_model`/`with_context_max`/`with_stats`/`with_usage`/`with_timing`/`with_summary`/`with_compact_summary`/`with_reason`/`with_error`.
  - `TokenProvenance` — what the recorder knows about a number: `counted(tokenizer)`, `estimated(source, note)`, `from_source(TokenizerSource)`, `of_selected(&SelectedTokenizer)`, `string_length_estimate()`, `unverified(note)`, `is_estimated`, `source`, `note`, plus a `Default` that is an unnamed estimate; re-exported from `telemetry` and the crate root.
  - `MeasuredTokens` — `tokens` + `provenance`, the pair `metric()` and `statistic()` are built from; re-exported from `telemetry` and the crate root.
  - `ContextProcessorStateRecorder` — `new`, `with_history_limit`, `with_sink`, `with_session`, `with_token_counter`, `with_token_counter_provenance`, `with_token_provenance`, `set_sink`/`set_session`/`set_token_counter`/`set_token_counter_with_provenance`/`set_token_provenance`, `token_provenance`, `history`, `last`, `clear_history`, `load_history`, `build_state`, `emit`, `session_id`, `context_id`, `history_limit`.
  - `CompressionStateSink` (async trait, one `on_state` method) and the in-memory `RecordingSink` (`new`/`failing`/`states`/`statuses`/`last`/`len`/`is_empty`).
  - `modified_messages(before, after)` — the public diffing rule; `format_timestamp(SystemTime)` — the ISO-8601 UTC formatter; `DEFAULT_HISTORY_LIMIT` (100).
- `scope` — compaction correlation.
  - `CompressionOperation` (`id`, `purpose`) with `new`/`with_purpose`/`is_empty`/`request_purpose`/`context_operation_id`/`model_kwargs`/`apply_to`/`clear_from`.
  - `with_operation`, `with_operation_scope`, `current_compression_operation`, `current_compression_operation_id`, `current_request_purpose`, `stamp_current_operation`, `apply_operation`.
  - `COMPACTION_PURPOSE` (`"compaction"`), `REQUEST_PURPOSE_KEY`, `CONTEXT_OPERATION_ID_KEY`.

## Wire Contract

- Every field name is Python's; `None` serializes as JSON `null` rather than being skipped, matching `model_dump(mode="json")`.
- The one exception is `ContextCompressionMetric::tokens_source`, which is absent rather than `null` when nothing can be named, because an unknown source has no spelling.
- `type` is written as `context.compression_state` and preserved verbatim when read.
- Every struct is `#[serde(default)]` and tolerates unknown fields, so a record written by an older or newer writer still loads; a missing `type` defaults to the constant.
- `estimated` and `tokens_source` on a metric, and `estimated` on `saved`, are additive to Python's pydantic models: a Python-written record loads with them defaulted, and a Rust-written record stays valid JSON for any tolerant reader.
- The stream frame is `StreamItem::output(CONTEXT_COMPRESSION_STATE_TYPE, Some(0), serde_json::to_value(&state))` — Python's `index=0`.
- `statistic` is `crate::stats::ContextStats` (the crate's per-role counters, including its additive `estimated` flag), not Python's `ContextStats` class name-for-name.
- `time` is always UTC with a `Z` suffix (`2024-01-02T03:04:05.006Z`); Python formats in the local timezone, so cross-host records compare as strings here.

## Design Decisions

- `modified_messages` diffs by `context_message_id`: a message present on both sides whose content changed (even at the same length) or whose offload marker is new is a rewrite; removed, new, and id-less messages are never reported.
- The offload state is read from the `[[OFFLOAD: …]]` marker inside the message text (this crate has no `OffloadMixin` field), so `offload_handle`/`offload_type` come from `OffloadMarker::parse` and only when the handle differs from the before side.
- The recorder keeps history in a `std::sync::RwLock<Vec<_>>`, bounded to `history_limit` (oldest dropped) with `load_history` keeping the newest entries; no lock is ever held across an `await`.
- Sink, session, and token counter sit behind `RwLock<Option<Arc<…>>>` because the active model's counter and the session writer are usually known only after the context exists.
- `emit` builds the record, records it, logs one `tracing::info!` line (naming the before/after counts and their provenance), calls the sink, then writes the stream frame, swallowing every failure with a `warn!` (and a `debug!` when no session writer is attached) — telemetry never fails a processor pass.
- A record's `before`/`after` metrics are measured through `TokenCounter::count_messages`, so the workspace's per-message framing and any tool calls are part of `tokens`; `estimated` and `tokens_source` come from the counter's declared `TokenProvenance`.
- Without a counter the recorder measures with `ApproxTokenCounter` at three characters per token — Python's `ceil(characters / 3)` rule plus the workspace framing — and publishes `estimated = true` with `tokens_source = "string_length_estimate"`, warning once per recorder instance rather than once per pass.
- `ContextEngine` installs its own counter on the recorder together with the provenance it can name (`approx_token_counter`, `characters_token_counter`, `tiktoken` for a `tiktoken` configuration, an unnamed estimate for `TokenCounterKind::Model` and for a per-context supplied counter), so `before`/`after` in a record and the pipeline's `statistic` agree by construction.
- A counter a caller set on the recorder wins over the engine's installation and is never replaced; `set_token_counter(None)` returns to the string-length fallback and lets a later engine installation take over.
- `statistic` prefers the per-role counts a caller supplied (`stats_after`, then `stats_before`), because the pipeline measures them with the context's own counter and owns their provenance; otherwise it computes them here and flags them with the recorder's `estimated`.
- The pipeline computes `stats_before`/`stats_after` with `cx.counter` and stamps `with_estimated(recorder.token_provenance().is_estimated())`, so a per-role count is never published as exact when the counter behind it only estimates.
- `PreparedContext::stats` derives its flag the same way: the recorder's declared provenance first, then the usage analyzer's `CounterIdentity`, else an unnamed estimate.
- `ContextCompressionMetric::estimated` is the broad fact (a `tiktoken` vocabulary is an estimate too), while `tokens_source` names the strategy; `saved.estimated` is `before.estimated || after.estimated`, so a saving computed from estimates is never published as exact.
- `metric()` measures the whole message list through the counter, so a non-text payload counts as whatever the counter charges it rather than as zero characters.
- `saved` is signed (`before - after`), so a pass that grows the history reports negative savings; `percent` is `saved_tokens / before_tokens * 100` rounded to one decimal.
- `context_percent` is `round(tokens / context_max * 100)` clamped to 0–100, and `None` when `context_max` is absent or zero.
- `summary` is generated from Python's `_build_summary` ladder (`Compressing …` / `Compressed …` / `Context unchanged …` / `Context processor skipped: …` / paused / cancelled / failed) unless the input supplies one; counts are compacted (`1.5m`, `12.3k`).
- `duration_ms` comes from `ended_at - started_at` on `SystemTime`; a reversed clock yields `None` rather than a panic.
- The correlation scope is a `tokio` task-local, not a process-global mutex, so nested scopes restore the outer id on drop, concurrent tasks see their own id, and nothing leaks outside a scope.
- `apply_to` merges `request_purpose`/`context_operation_id` into `ModelRequestConfig::extra`, the request's only free-form field, preserving existing keys and leaving an empty operation untouched; Python's keys are callback-only kwargs, while a Rust provider merges `extra` into the body last, so the helper is opt-in.
- An `extra` that is neither a JSON object nor `null` cannot carry the keys: `apply_to` drops them, warns once naming the JSON shape, the operation id and the purpose, and the model call still runs uncorrelated rather than failing.
- `clear_from`/`apply_operation(request, None)` remove the two keys, so a caller can reuse a request outside a compaction pass.

## Constraints

- Errors use `openjiuwen_common::error::{Error, Result}`; the sink trait returns `Result<()>` only so an implementation can report a delivery failure, which the recorder logs and swallows.
- No `unsafe`, no `println!`/`eprintln!`, no `unwrap`/`expect` in library code, `tracing` only, no network, no new crates, no blocking I/O on async paths.
- `emit` and `build_state` take a borrowed input, so a caller's message slices stay owned by the caller.
- `ContextProcessorStateRecorder` is not `Clone`; share it as `Arc`. Its `Debug` impl reports presence flags for the three trait-object fields (plus the token provenance) instead of the fields themselves.
- Everything is deterministic and offline: no model, no clock read unless a caller passes a `SystemTime`, no filesystem.

## Routing Table

- `./state.rs` → the `context.compression_state` value objects (including the provenance fields), status/phase vocabularies, constructors and setters
- `./recorder/` → `mod.rs` (`ContextProcessorStateRecorder`, `ContextProcessorStateInput`, `TokenProvenance`, `MeasuredTokens`, the counter binding, provenance-aware measurement and statistics, rewrite diffing, sinks), `metrics.rs` (timestamp formatter, metric/saved math, compact summaries), `tests.rs`
- `./scope.rs` → `CompressionOperation`, the task-local scope, the request stamping helpers and their shape warning

## Test Strategy

- `cargo test -p openjiuwen-context --offline` runs 374 library unit tests, 34 integration binaries and 16 doc tests — 1024 tests, green offline.
- Unit tests beside the modules: timestamp/civil-date formatting, compact numbers and summary texts, the rewrite diffing (unchanged, rewritten at equal length, offloaded, new, removed, id-less), and the scope helpers.
- `crates/openjiuwen-context/tests/telemetry.rs` (24 tests) pins the JSON shape and field names, every status/phase spelling, round-trip and unknown-field tolerance, the convenience constructors, diffing (including offload handle/type propagation), metric/saved/duration/statistic/summary values, the history cap and `load_history`, sink ordering and swallowed sink failure, session-stream emission against a real `AgentSession`, and scope correlation (visible inside, absent outside, nested restore, concurrent tasks, custom purpose, request stamping).
- `crates/openjiuwen-context/tests/telemetry_provenance.rs` (8 tests) pins the three rungs a recorder can be on — the `ceil(characters / 3)` fallback, a declared approximate counter, and a native tokenizer that counts exactly — the equality of a record's `before`/`after`/`saved` numbers with its `statistic`, serde tolerance of the additive keys in both directions, and the correlation warning of a non-object request `extra`.
- `crates/openjiuwen-context/tests/pipeline_telemetry.rs` (13 tests) covers the per-entry-point phases, the bracketing records and the propagation of the provenance flag into `ContextStats`.
- Offline and deterministic: the one-character-per-token counter or a tiny in-memory BPE artifact, no model, in-process session, bounded 5 s stream await, and a local tracing subscriber instead of a log file.
