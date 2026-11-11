# openjiuwen-context

## Intent

Owns the seam between the messages an agent accumulates and the payload sent to a model: conversation history, window assembly, token accounting, and the compression/offload pipeline.
Ports a faithful core subset of `openjiuwen/core/context_engine/` (`ModelContext`, `ContextEngine`, processor pipeline, token counters, offload storage) to idiomatic Rust.
The engine is a reusable kernel service for `openjiuwen-agent`, `openjiuwen-workflow`, and `openjiuwen-harness`; it never calls a provider unless a summarisation processor was explicitly configured with a `Model`.

## API Surface

Crate root (`openjiuwen_context`) re-exports every public item below; `CRATE_NAME`/`CRATE_VERSION` are also exported.
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

- `ContextEngine` — `new`, `with_store`, `config`, `create_context`, `create_context_with`, `get_context`, `remove_context`, `prepare`, `observe`, `observe_tool_result`, `compress`, `recover_from_model_exception`, `is_context_overflow_error`, `save_contexts`, `load_contexts`.
- `DEFAULT_SESSION_ID` — session id used when `create_context` receives no session.
- `ContextOptions` — `session`, `history`, `processors`, `token_counter: Option<Arc<dyn TokenCounter + Send + Sync>>`, `model`.
- `PrepareOptions` — `system_messages`, `window_size`, `dialogue_rounds`, `tools: Vec<Value>` (pass-through), `cancellation: CancellationToken`, `timeout: Option<Duration>`.
- `ContextConfig` — `max_context_tokens`, `reserve_tokens`, `compression_threshold` (0.85 default), `recent_messages` (6), `max_context_messages`, `default_window_messages`, `default_window_rounds`, `enable_reload`, `token_counter: TokenCounterConfig`, `compression_recall: CompressionRecallConfig`, `processors: Vec<ProcessorConfig>` (default `[ToolResult, Truncation]`), `model_name`; `effective_max_tokens()`, `compression_trigger_tokens()`, `token_counter() -> Arc<dyn TokenCounter + Send + Sync>`, `validate()`.
- `TokenCounterConfig` — `kind: TokenCounterKind`, `chars_per_token` (4 default); `TokenCounterKind` is `Approx` | `Characters` | `Tiktoken`.
- `CompressionRecallConfig` — `enabled`, `chunk_size_tokens`, `chunk_overlap_tokens`.
- `DEFAULT_CONTEXT_TOKENS` — 200 000, used when `max_context_tokens` is `None`.
- `ModelContext` trait — `context_id`, `session_id`, `messages`, `set_messages`, `add_message`, `pop_messages`, `clear_messages`, `token_counter`, `statistics`, `revision`.
- `ConversationContext` — the built-in `ModelContext`: config/model/processors/offload-store accessors, `scope()`, `session()`, `cancel()`, `cancellation()`, `save_state`, `load_state`, `prepare`; the constructor is engine-only.
- `PreparedContext` — `messages` (system first, then selected context), `tools`, `stats: ContextStats`, `token_count`, `overflow`, `window_size`, `events`.
- `ContextStats` — total/per-role message counts and token counts, `total_dialogues`, `compressions`, `tokens_before_compression`, `tokens_after_compression`.
- `ContextEvent` / `ContextEventKind` (`truncated`, `summarized`, `offloaded`, `tool_result_trimmed`), `CompressionStatus`, `CompressReport`, `RecoveryOutcome`, `ObserveOutcome`.
- `ContextState` — serde `{ messages, revision }` snapshot used by session persistence.
- Processors (`processor` module): `ContextProcessor` trait (`name`, `process`, `save_state`, `load_state`), `ProcessorContext`, `ContextPipeline::{new, from_configs, processors, run}`, `ProcessorConfig::{Truncation, ToolResult, Summarization}` with `TruncationConfig`, `ToolResultConfig`, `SummarizationConfig`; constants `TRUNCATED_MARKER`, `PERSISTED_OUTPUT_TAG`, `CONTEXT_SUMMARY_PREFIX`.
- Offload (`offload` module): `OffloadStore` trait (`kind`, `put`, `get`, `remove`), `InMemoryOffloadStore`, `KvOffloadStore` (foundation `KvStore`, `new`/`with_prefix`), `SessionOffloadStore` (`openjiuwen_session::Session`), `OffloadScope`, `OffloadRecord`, `OffloadKind`, `OffloadMarker::{to_marker_string, parse}`, `OFFLOAD_MARKER_PREFIX`, `sanitize_key`, `rehydrate`.
- `window` module: `dialogue_round_ranges`, `find_last_n_rounds_start`, `WindowOptions`, `assemble_window`, `strip_leading_tool_messages`, `drop_orphan_tool_results`, `validate_tool_pairs`, `truncate_keeping_rounds`.
- `tokens` module: `count_message`, `count_messages`, `TokenBudget`, `check_budget`.
- `stats` module: `compute_stats`.
- `prompts` module: `SUMMARY_SYSTEM_PROMPT`, `SUMMARY_USER_TEMPLATE`, `render_summary_prompt`.

## Constraints

- Errors use `openjiuwen_common::error::{Error, Result}`; invalid config is `Error::Config`, bad message/window input `Error::Validation`, missing context `Error::NotFound`, summarisation failures and malformed offload records `Error::Llm`/`Error::Validation`, prepare timeouts `Error::Timeout`, cancellation `Error::Cancelled`.
- Async-first on tokio; public processors are `Send + Sync` via `async_trait`; cancellation uses `tokio_util::sync::CancellationToken`; optional per-`prepare` timeouts use `tokio::time::timeout`.
- No blocking I/O on async tasks; offload stores are async traits; locks are never held across a model call.
- Every public data type is `serde`-serializable, `snake_case`, tolerant of unknown fields (`#[serde(default)]`), and config round-trips through JSON.
- Token counting always goes through `openjiuwen_foundation::prompt::TokenCounter`; the deterministic default is `ApproxTokenCounter`.
- Summarisation is the only path that invokes a `Model`; it uses `Model::invoke` with `ModelRequestConfig` and never streams.
- System messages are never compressed or evicted; tool-call/tool-result pairs are never split by truncation or summarisation.
- No `unsafe`; `tracing` only; files stay at or below 1000 lines (`engine.rs` sits at the limit — split by responsibility before adding to it).

## Design Decisions

- The engine owns the default processor pipeline; a context carries an immutable pipeline snapshot created at `create_context`, matching Python's attachment-order semantics.
- `prepare` runs the pipeline on a working copy of the history and never mutates stored messages; explicit `compress` persists the changed history and reports `compressed` exactly once.
- `observe` is the add-hook seam: it appends one message, applies the rolling `max_context_messages` limit, and recomputes stats; passive processors may inspect new messages inside `prepare`.
- Compression triggers on `token_count > compression_threshold * effective_max_tokens`; `max_tokens = effective_max_tokens - reserve_tokens` (saturating).
- Offload handles are opaque UUIDs scoped by `OffloadScope { session_id, context_id }`, so concurrent sessions cannot read each other's payloads; markers keep Python's `[[OFFLOAD: handle=..., type=in_memory|filesystem]]` shape.
- Token counters are stored as `Arc<dyn TokenCounter + Send + Sync>` (foundation's `TokenCounter` has no auto-trait supertraits) so processor futures stay `Send`; `ProcessorContext::counter` carries the same bounds and coerces to `&dyn TokenCounter` at counting call sites.
- Tool-result processors rewrite message content only (never remove messages), which preserves every `tool_call_id` pairing by construction.
- Summarisation replaces complete oldest dialogue rounds with one `[CONTEXT_SUMMARY]` user message, keeps every system message and the newest `recent_messages`, and never touches the most recent round.
- Window assembly applies round truncation first, then message-count truncation where system messages consume the budget, then strips leading `tool` messages (Python parity).
- A `TruncationConfig::max_tokens` below the current size acts regardless of the trigger threshold; the threshold only gates the case where no processor budget is set.
- `TruncationConfig::keep_system = false` drops the oldest system messages first, which is the only way that field is not inert.
- The context pool key is `(session_id, context_id)` with dots in `context_id` normalized to `_`, matching Python's pool identity; the context keeps its original id.
- Context state persistence is a serde `ContextState { messages, revision }` written under session key `context.<context_id>`; the engine wraps a map of context id → state under `context`.

## Routing Table

- `./src/lib.rs` → crate docs, module declaration, public re-exports, lint attributes
- `./src/config.rs` → `ContextConfig`, `TokenCounterConfig`, `CompressionRecallConfig`, defaults and validation
- `./src/tokens.rs` → token-counter accounting helpers, budget check
- `./src/stats.rs` → `ContextStats`, `PreparedContext`, `compute_stats`, per-role accounting
- `./src/events.rs` → `ContextEvent`, `ContextEventKind`, `CompressionStatus`, `CompressReport`, `RecoveryOutcome`, `ObserveOutcome`
- `./src/window.rs` → dialogue rounds, window assembly, tool-pair helpers, round-aware truncation
- `./src/context.rs` → `ModelContext` trait, `ConversationContext`, `ContextState` save/load, prepare runner
- `./src/engine.rs` → `ContextEngine`, context pool, `observe`/`compress`/recovery, session persistence
- `./src/offload.rs` → `OffloadStore` implementations, `OffloadScope`/`OffloadRecord`/markers, rehydration
- `./src/prompts.rs` → summarisation system/user prompt templates
- `./src/processor/mod.rs` → `ContextProcessor` trait, `ProcessorContext`, `ProcessorConfig`, `ContextPipeline`
- `./src/processor/truncate.rs` → `TruncationProcessor` + config (round-aware eviction)
- `./src/processor/summarize.rs` → `SummarizationProcessor` + config (LLM summarisation via `Model`)
- `./src/processor/tool_result.rs` → `ToolResultProcessor` + config (per-result truncation and offload)
- `./tests/engine.rs` → engine end-to-end suite (prepare/observe/compress/recovery/persistence)
- `./tests/processors.rs` → pipeline ordering, truncation, tool-result offload, summarisation, event serde
- `./tests/offload.rs` → marker wire format, key isolation across all three stores, rehydration round-trips

## Test Strategy

- Unit tests live beside each module (80 tests); integration suites live in `crates/openjiuwen-context/tests/` (56 tests); doc tests cover the crate-level example paths (6).
- Deterministic offline tests use `ApproxTokenCounter::with_chars_per_token(1)` (via `TokenCounterKind::Characters`) and `openjiuwen_test_support::ScriptedModel` as the `Model`; sessions are real `openjiuwen_session::AgentSession` instances.
- Covered behaviors: window assembly with round/message budgets, leading-tool stripping, per-role stats, overflow detection, every processor, summary replacement with system/recent preservation, tool-pair preservation, offload/rehydrate round-trip, session isolation, `noop`/`compressed`/`busy`, overflow recovery refusal after streamed chunks, config serde round-trip, cancellation, `save_contexts`/`load_contexts` round-trip, corrupt-state tolerance.
- `cargo test -p openjiuwen-context --offline` must pass; `cargo clippy -p openjiuwen-context --all-targets --offline -- -D warnings` and `cargo fmt -p openjiuwen-context -- --check` must be clean.

## Deferred (Python `context_engine` features not ported)

- Tokenizer registry/selector/artifact manager, native `tokenizer.json` loading, download logic, and model-specific BPE parity (`Tiktoken` uses `cl100k_base` with the documented heuristic fallback).
- Usage/KV-cache telemetry (`ContextUsageAnalyzer`, `SessionKVCacheAggregator`, `context.usage` snapshots) beyond `ContextStats` and compression before/after counts.
- Provider-specific context policies, `ContextWindowChange`/KV-cache eviction detection, OpenRouter context-window cache, and `model_revision`/`position` diffing.
- The Python compressor family not represented here: `DialogueCompressor`, `CurrentRoundCompressor`, `RoundLevelCompressor`, `FullCompactProcessor`, `MicroCompactProcessor`, `MessageOffloader`, `MessageSummaryOffloader`, `ToolResultBudgetProcessor`, `ToolResultWindowProcessor`; their essential behaviors are covered by the three Rust processors.
- `processor/forked/` registry monkey-patching, session-memory agent, `[SESSION_MEMORY_BOUNDARY]` reinjection, skill/tool reinjection, and compression observability hooks beyond `ContextEvent`/`tracing`.
- Compression-recall chunking, `enable_reload` filesystem rehydration prompt injection, and per-processor `save_state`/`load_state` checkpointing (the trait hooks exist but the engine does not persist processor state).
