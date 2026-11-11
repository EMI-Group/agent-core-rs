# openjiuwen-context

## Intent

Owns the seam between the messages an agent accumulates and the payload sent to a model: conversation history, window assembly, token accounting, and the compression/offload pipeline.
Ports a faithful core subset of `openjiuwen/core/context_engine/` (`ModelContext`, `ContextEngine`, processor pipeline, token counters, offload storage) to idiomatic Rust.
The engine is a reusable kernel service for `openjiuwen-agent`, `openjiuwen-workflow`, and `openjiuwen-harness`; it never calls a provider unless a summarisation processor was explicitly configured with a `Model`.

## API Surface

Crate root (`openjiuwen_context`) re-exports every public item below.
Execution seams used by agents and the harness:

```rust
// Engine
let engine = ContextEngine::new(ContextConfig::default());
let ctx = engine.create_context("main", session).await?;          // session: Option<Arc<dyn Session>>
ctx.add_message(BaseMessage::user("hi")).await?;                 // also via ModelContext trait
let prepared: PreparedContext = engine.prepare(&ctx, &PrepareOptions::default()).await?;
// send prepared.messages (system first) to the model
engine.observe(&ctx, BaseMessage::assistant("hello")).await?;
engine.observe_tool_result(&ctx, "call_1", "42").await?;
let report: CompressReport = engine.compress(&ctx).await?;       // status: noop | compressed | busy
let recovered: RecoveryOutcome = engine
    .recover_from_model_exception(&ctx, &error_text, streaming, chunks_emitted).await?;
```

- `ContextEngine` — config, per-session context pool, `create_context`/`get_context`/`remove_context`, `prepare`, `observe`, `observe_tool_result`, `compress`, `recover_from_model_exception`, `is_context_overflow_error`, `save_contexts`/`load_contexts`.
- `ContextConfig` — `max_context_tokens`, `reserve_tokens`, `compression_threshold` (0.85 default), `recent_messages` (6), `max_context_messages`, `default_window_messages`, `default_window_rounds`, `enable_reload`, `token_counter: TokenCounterConfig`, `compression_recall: CompressionRecallConfig`, `processors: Vec<ProcessorConfig>`, `model_name`; `effective_max_tokens()`, `compression_trigger_tokens()`, `token_counter()`, `validate()`.
- `TokenCounterConfig` — `Approx` (foundation 4 chars/token), `Characters { chars_per_token }` (Python `len // 3` fallback), `Tiktoken` (foundation BPE with heuristic fallback).
- `ModelContext` trait — `context_id`, `session_id`, `messages`, `set_messages`, `add_message`, `pop_messages`, `clear_messages`, `token_counter`, `statistics`, `revision`.
- `ConversationContext` — the built-in `ModelContext`: append/pop/clear, rolling `max_context_messages` eviction, stats cache, cancellation token, `save_state`/`load_state`.
- `PrepareOptions` — `system_messages`, `window_size`, `dialogue_rounds`, `tools: Vec<Value>` (pass-through), `cancellation: CancellationToken`, `timeout: Option<Duration>`.
- `PreparedContext` — `messages` (system first, then selected context), `tools`, `stats: ContextStats`, `token_count`, `overflow`, `window_size`, `events`.
- `ContextStats` — total/per-role message counts and token counts, `total_dialogues`, `tools`, `compressions`, `tokens_before_compression`, `tokens_after_compression`.
- `ContextEvent` / `ContextEventKind` (`truncated`, `summarized`, `offloaded`, `tool_result_trimmed`), `CompressionStatus`, `CompressReport`, `RecoveryOutcome`, `ObserveOutcome`.
- Processors (`processor` module): `ContextProcessor` trait (`name`, `process`, `save_state`, `load_state`), `ProcessorContext`, `ContextPipeline`, `ProcessorConfig::{Truncation, ToolResult, Summarization}` with their configs.
- Offload (`offload` module): `OffloadStore` trait (`kind`, `put`, `get`, `remove`), `InMemoryOffloadStore`, `KvOffloadStore` (foundation `KvStore`), `SessionOffloadStore` (`openjiuwen_session::Session`), `OffloadScope`, `OffloadRecord`, `OffloadMarker`/`OffloadKind`, `format_marker`, `parse_marker`, `rehydrate`.
- `window` module: `dialogue_round_ranges`, `assemble_window`, `strip_leading_tool_messages`, `drop_orphan_tool_results`, `truncate_keeping_rounds`, `validate_tool_pairs`.
- `prompts` module: summary prompt template constants (`SUMMARY_SYSTEM_PROMPT`, `SUMMARY_USER_TEMPLATE`).

## Constraints

- Errors use `openjiuwen_common::error::{Error, Result}`; invalid config is `Error::Config`, bad message/window input `Error::Validation`, missing context `Error::NotFound`, summarisation failures `Error::Llm`, cancellation `Error::Cancelled`.
- Async-first on tokio; public processors are `Send + Sync` via `async_trait`; cancellation uses `tokio_util::sync::CancellationToken`; optional per-`prepare` timeouts use `tokio::time::timeout`.
- No blocking I/O on async tasks; offload stores are async traits; locks are never held across a model call.
- Every public data type is `serde`-serializable, `snake_case`, tolerant of unknown fields (`#[serde(default)]`), and config round-trips through JSON.
- Token counting always goes through `openjiuwen_foundation::prompt::TokenCounter`; the deterministic default is `ApproxTokenCounter`.
- Summarisation is the only path that invokes a `Model`; it uses `Model::invoke` with `ModelRequestConfig` and never streams.
- System messages are never compressed or evicted; tool-call/tool-result pairs are never split by truncation or summarisation.
- No `unsafe`; `tracing` only; files stay around ≤1000 lines.

## Design Decisions

- The engine owns the default processor pipeline; a context carries an immutable pipeline snapshot created at `create_context`, matching Python's attachment-order semantics.
- `prepare` runs the pipeline on a working copy of the history and never mutates stored messages; explicit `compress` persists the changed history and reports `compressed` exactly once.
- `observe` is the add-hook seam: it appends one message, applies the rolling `max_context_messages` limit, and recomputes stats; passive processors may inspect new messages inside `prepare`.
- Compression triggers on `token_count > compression_threshold * effective_max_tokens`; `max_tokens = effective_max_tokens - reserve_tokens` (saturating).
- Offload handles are opaque UUIDs scoped by `OffloadScope { session_id, context_id }`, so concurrent sessions cannot read each other's payloads; markers keep Python's `[[OFFLOAD: handle=..., type=in_memory|filesystem]]` shape.
- Tool-result processors rewrite message content only (never remove messages), which preserves every `tool_call_id` pairing by construction.
- Summarisation replaces complete oldest dialogue rounds with one `[CONTEXT_SUMMARY]` user message, keeps every system message and the newest `recent_messages`, and never touches the most recent round.
- Window assembly applies round truncation first, then message-count truncation where system messages consume the budget, then strips leading `tool` messages (Python parity).
- The context pool key is `(session_id, context_id)` with dots in `context_id` normalized to `_`, matching Python's pool identity.
- Context state persistence is a serde `ContextState { messages, revision }` written under session key `context.<context_id>`; the engine wraps a map of context id → state under `context`.

## Routing Table

- `./src/lib.rs` → crate docs, module declaration, public re-exports
- `./src/config.rs` → `ContextConfig`, `TokenCounterConfig`, `CompressionRecallConfig`, defaults and validation
- `./src/tokens.rs` → token-counter construction, per-message/per-request counting, overflow detection
- `./src/stats.rs` → `ContextStats`, `PreparedContext`, `compute_stats`, per-role accounting
- `./src/events.rs` → `ContextEvent`, `ContextEventKind`, `CompressionStatus`, `CompressReport`, `RecoveryOutcome`, `ObserveOutcome`
- `./src/window.rs` → dialogue rounds, window assembly, tool-pair helpers, round-aware truncation
- `./src/context.rs` → `ModelContext` trait, `ConversationContext`, `ContextState` save/load
- `./src/engine.rs` → `ContextEngine`, context pool, `prepare`/`observe`/`compress`/recovery, session persistence
- `./src/offload.rs` → `OffloadStore` implementations, `OffloadScope`/`OffloadRecord`/markers, rehydration
- `./src/prompts.rs` → summarisation system/user prompt templates
- `./src/processor/mod.rs` → `ContextProcessor` trait, `ProcessorContext`, `ProcessorConfig`, `ContextPipeline`
- `./src/processor/truncate.rs` → `TruncationProcessor` + config (round-aware eviction)
- `./src/processor/summarize.rs` → `SummarizationProcessor` + config (LLM summarisation via `Model`)
- `./src/processor/tool_result.rs` → `ToolResultProcessor` + config (per-result truncation and offload)
- `./tests/` → crate-level integration tests (engine, processors, offload, serde)

## Test Strategy

- Unit tests live beside each module; integration tests live in `crates/openjiuwen-context/tests/`.
- Deterministic offline tests use `ApproxTokenCounter::with_chars_per_token(1)` (or a local fixed counter) and `openjiuwen_test_support::ScriptedModel` as the `Model`.
- Covered behaviors: window assembly with round/message budgets, leading-tool stripping, per-role stats, overflow detection, every processor, summary replacement with system/recent preservation, tool-pair preservation, offload/rehydrate round-trip, session isolation, `noop`/`compressed`/`busy`, overflow recovery refusal after streamed chunks, config serde round-trip, cancellation.
- `cargo test -p openjiuwen-context` must pass offline; `cargo clippy -p openjiuwen-context --all-targets -- -D warnings` and `cargo fmt -p openjiuwen-context` must be clean.

## Deferred (Python `context_engine` features not ported)

- Tokenizer registry/selector/artifact manager, native `tokenizer.json` loading, download logic, and model-specific BPE parity (`Tiktoken` uses `cl100k_base` with the documented heuristic fallback).
- Usage/KV-cache telemetry (`ContextUsageAnalyzer`, `SessionKVCacheAggregator`, `context.usage` snapshots) beyond `ContextStats` and compression before/after counts.
- Provider-specific context policies, `ContextWindowChange`/KV-cache eviction detection, OpenRouter context-window cache, and `model_revision`/`position` diffing.
- The Python compressor family not represented here: `DialogueCompressor`, `CurrentRoundCompressor`, `RoundLevelCompressor`, `FullCompactProcessor`, `MicroCompactProcessor`, `MessageOffloader`, `MessageSummaryOffloader`, `ToolResultBudgetProcessor`, `ToolResultWindowProcessor`; their essential behaviors are covered by the three Rust processors.
- `processor/forked/` registry monkey-patching, session-memory agent, `[SESSION_MEMORY_BOUNDARY]` reinjection, skill/tool reinjection, and compression observability hooks beyond `ContextEvent`/`tracing`.
- Compression-recall chunking, `enable_reload` filesystem rehydration prompt injection, and per-processor `save_state`/`load_state` checkpointing.
