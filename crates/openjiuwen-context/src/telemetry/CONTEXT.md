# openjiuwen-context/src/telemetry — compression telemetry

## Intent

Owns the observability side of the compression pipeline: the `context.compression_state` record, the per-processor recorder that builds and fans it out, and the task-local request-purpose correlation scope for processor model calls.
Ports `schema/context_state.py`, `context/processor_state_recorder.py` and `context/compression_scope.py`.
Nothing here changes a history, calls a model, or blocks; `emit` is a side channel that can fail only into a log line.
A later pass wires the recorder into `processor::ContextPipeline`; this node is the seam plus the recorder itself.

## API Surface

- `state` — the wire contract.
  - `CONTEXT_COMPRESSION_STATE_TYPE` (`"context.compression_state"`), the frame type and the record's `type` tag.
  - `ContextCompressionState` — `state_type` (serde `type`), `operation_id`, `status`, `phase`, `processor`, `model`, `before`, `after`, `statistic`, `saved`, `modified_messages`, `compression_usage`, `duration_ms`, `context_max`, `summary`, `compact_summary`, `error`, `reason`.
  - `CompressionStateStatus` — `started`/`completed`/`noop`/`skipped`/`failed`/`paused`/`cancelled`, with `as_str`, `is_terminal`, `is_success`; it is **not** `events::CompressionStatus`.
  - `CompressionStatePhase` — `add_messages`/`get_context_window`/`active_compress`.
  - `ContextCompressionMetric` (`time`, `messages`, `tokens`, `context_percent`), `ContextCompressionSaved` (signed `messages`/`tokens` plus `percent`).
  - `ContextCompressionUsage` — `calls`, token counters, three costs, `model_name`, free-form `details`; `from_usage`, `record`, `add`, `with_costs`, `with_model_name`, `with_detail`.
  - `ContextCompressionModifiedMessage` — `message_id`, `role`, `tool_call_id`, `offload_handle`, `offload_type`.
  - Convenience constructors `ContextCompressionState::{new, started, completed, noop, failed}` plus `with_*` setters for every non-derived field.
- `recorder` — `ContextProcessorStateRecorder` and its inputs.
  - `ContextProcessorStateInput` — `operation_id`, `status`, `phase`, `processor`, `model`, `context_max`, `before`, `after`, `stats_before`, `stats_after`, `usage`, `error`, `reason`, `summary`, `compact_summary`, `started_at`, `ended_at`; `new` plus `with_after`/`with_model`/`with_context_max`/`with_stats`/`with_usage`/`with_timing`/`with_summary`/`with_compact_summary`/`with_reason`/`with_error`.
  - `ContextProcessorStateRecorder` — `new`, `with_history_limit`, `with_sink`, `with_session`, `with_token_counter`, `set_sink`/`set_session`/`set_token_counter`, `history`, `last`, `clear_history`, `load_history`, `build_state`, `emit`, `session_id`, `context_id`, `history_limit`.
  - `CompressionStateSink` (async trait, one `on_state` method) and the in-memory `RecordingSink` (`new`/`failing`/`states`/`statuses`/`last`/`len`/`is_empty`).
  - `modified_messages(before, after)` — the public diffing rule; `format_timestamp(SystemTime)` — the ISO-8601 UTC formatter; `DEFAULT_HISTORY_LIMIT` (100).
- `scope` — compaction correlation.
  - `CompressionOperation` (`id`, `purpose`) with `new`/`with_purpose`/`is_empty`/`request_purpose`/`context_operation_id`/`model_kwargs`/`apply_to`/`clear_from`.
  - `with_operation`, `with_operation_scope`, `current_compression_operation`, `current_compression_operation_id`, `current_request_purpose`, `stamp_current_operation`, `apply_operation`.
  - `COMPACTION_PURPOSE` (`"compaction"`), `REQUEST_PURPOSE_KEY`, `CONTEXT_OPERATION_ID_KEY`.

## Wire Contract

- Every field name is Python's; `None` serializes as JSON `null` rather than being skipped, matching `model_dump(mode="json")`.
- `type` is written as `context.compression_state` and preserved verbatim when read.
- Every struct is `#[serde(default)]` and tolerates unknown fields, so a record written by an older or newer writer still loads; a missing `type` defaults to the constant.
- The stream frame is `StreamItem::output(CONTEXT_COMPRESSION_STATE_TYPE, Some(0), serde_json::to_value(&state))` — Python's `index=0`.
- `statistic` is `crate::stats::ContextStats` (the crate's per-role counters), not Python's `ContextStats` class name-for-name.
- `time` is always UTC with a `Z` suffix (`2024-01-02T03:04:05.006Z`); Python formats in the local timezone, so cross-host records compare as strings here.

## Design Decisions

- `modified_messages` diffs by `context_message_id`: a message present on both sides whose content changed (even at the same length) or whose offload marker is new is a rewrite; removed, new, and id-less messages are never reported.
- The offload state is read from the `[[OFFLOAD: …]]` marker inside the message text (this crate has no `OffloadMixin` field), so `offload_handle`/`offload_type` come from `OffloadMarker::parse` and only when the handle differs from the before side.
- The recorder keeps history in a `std::sync::RwLock<Vec<_>>`, bounded to `history_limit` (oldest dropped) with `load_history` keeping the newest entries; no lock is ever held across an `await`.
- Sink, session, and token counter sit behind `RwLock<Option<Arc<…>>>` because the active model's counter and the session writer are usually known only after the context exists.
- `emit` builds the record, records it, logs one `tracing::info!` line, calls the sink, then writes the stream frame, swallowing every failure with a `warn!` (and a `debug!` when no session writer is attached) — telemetry never fails a processor pass.
- Token measurement uses the configured `TokenCounter` and falls back to Python's `ceil(characters / 3)`; statistics fall back to `compute_stats` with a 3-characters-per-token counter, so `statistic` always carries token counts (Python leaves them zero without a counter).
- `saved` is signed (`before - after`), so a pass that grows the history reports negative savings; `percent` is `saved_tokens / before_tokens * 100` rounded to one decimal.
- `context_percent` is `round(tokens / context_max * 100)` clamped to 0–100, and `None` when `context_max` is absent or zero.
- `summary` is generated from Python's `_build_summary` ladder (`Compressing …` / `Compressed …` / `Context unchanged …` / `Context processor skipped: …` / paused / cancelled / failed) unless the input supplies one; counts are compacted (`1.5m`, `12.3k`).
- `duration_ms` comes from `ended_at - started_at` on `SystemTime`; a reversed clock yields `None` rather than a panic.
- The correlation scope is a `tokio` task-local, not a process-global mutex, so nested scopes restore the outer id on drop, concurrent tasks see their own id, and nothing leaks outside a scope.
- `apply_to` merges `request_purpose`/`context_operation_id` into `ModelRequestConfig::extra`, the request's only free-form field, preserving existing keys and leaving an empty operation untouched; Python's keys are callback-only kwargs, while a Rust provider merges `extra` into the body last, so the helper is opt-in.
- `clear_from`/`apply_operation(request, None)` remove the two keys, so a caller can reuse a request outside a compaction pass.

## Constraints

- Errors use `openjiuwen_common::error::{Error, Result}`; the sink trait returns `Result<()>` only so an implementation can report a delivery failure, which the recorder logs and swallows.
- No `unsafe`, no `println!`/`eprintln!`, no `unwrap`/`expect` in library code, `tracing` only, no network, no new crates, no blocking I/O on async paths.
- `emit` and `build_state` take a borrowed input, so a caller's message slices stay owned by the caller.
- `ContextProcessorStateRecorder` is not `Clone`; share it as `Arc`. Its `Debug` impl reports presence flags for the three trait-object fields instead of the fields themselves.
- Everything is deterministic and offline: no model, no clock read unless a caller passes a `SystemTime`, no filesystem.

## Routing Table

- `./state.rs` → the `context.compression_state` value objects, status/phase vocabularies, constructors and setters
- `./recorder/` → `mod.rs` (`ContextProcessorStateRecorder`, `ContextProcessorStateInput`, rewrite diffing, sinks), `metrics.rs` (timestamp formatter, metric/saved math, compact summaries), `tests.rs`
- `./scope.rs` → `CompressionOperation`, the task-local scope, the request stamping helpers

## Test Strategy

- Unit tests beside the modules: timestamp/civil-date formatting, compact numbers and summary texts, and the rewrite diffing (unchanged, rewritten at equal length, offloaded, new, removed, id-less).
- `crates/openjiuwen-context/tests/telemetry.rs` (24 tests) pins the JSON shape and field names, every status/phase spelling, round-trip and unknown-field tolerance, the convenience constructors, diffing (including offload handle/type propagation), metric/saved/duration/statistic/summary values, the history cap and `load_history`, sink ordering and swallowed sink failure, session-stream emission against a real `AgentSession`, and scope correlation (visible inside, absent outside, nested restore, concurrent tasks, custom purpose, request stamping).
- Offline and deterministic: the one-character-per-token counter, no model, in-process session, bounded 5 s stream await.
