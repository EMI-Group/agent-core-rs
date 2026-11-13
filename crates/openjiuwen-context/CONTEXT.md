# openjiuwen-context

## Intent

Owns the seam between the messages an agent accumulates and the payload sent to a model: conversation history, window assembly, token accounting, and the compression/offload pipeline.
Ports the public behavior of `openjiuwen/core/context_engine/` (`ModelContext`, `ContextEngine`, the processor family, tokenizer registry, token counters, offload storage, compression recall, session-memory boundary, reload hint) to idiomatic Rust.
The engine is a reusable kernel service for `openjiuwen-agent`, `openjiuwen-workflow`, and `openjiuwen-harness`.
Only callers that configure a `Model` cause provider calls; every LLM-backed path has a deterministic offline fallback, so compression works with no network and no credentials.

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

- `ContextEngine` — `new`, `with_store`, `with_stores(config, offload, recall, session_memory)`, `config`, `create_context`, `create_context_with`, `get_context`, `remove_context`, `prepare`, `observe`, `observe_tool_result`, `compress`, `recover_from_model_exception`, `is_context_overflow_error`, `save_contexts`, `load_contexts`.
- `DEFAULT_SESSION_ID` — session id used when `create_context` receives no session.
- There is no `get_context_window`: `ContextEngine::prepare` + `PreparedContext` is the window/messages getter, and `ModelContext::messages()` is the raw stored-history reader.
- `ContextOptions` — `session`, `history`, `processors`, `token_counter: Option<Arc<dyn TokenCounter + Send + Sync>>`, `model`, `recall`, `session_memory` (defined in `engine_options`, re-exported from `engine`).
- `PrepareOptions` — `system_messages`, `window_size`, `dialogue_rounds`, `tools: Vec<Value>` (pass-through), `cancellation: CancellationToken`, `timeout: Option<Duration>` (defined in `engine_options`, re-exported from `engine`).
- `ContextConfig` — `max_context_tokens`, `reserve_tokens`, `compression_threshold` (0.85 default), `recent_messages` (6), `max_context_messages`, `default_window_messages`, `default_window_rounds`, `enable_reload`, `token_counter: TokenCounterConfig`, `tokenizer_provider`, `tokenizer_registry: Vec<TokenizerSpec>`, `compression_recall: CompressionRecallConfig`, `processors: Vec<ProcessorConfig>` (default `[ToolResult, Truncation]`), `model_name`; `effective_max_tokens()`, `compression_trigger_tokens()`, `token_counter() -> Arc<dyn TokenCounter + Send + Sync>`, `validate()`.
- `TokenCounterConfig` — `kind: TokenCounterKind`, `chars_per_token` (4 default); `TokenCounterKind` is `Approx` | `Characters` | `Tiktoken` | `Model`.
- `CompressionRecallConfig` — `enabled`, `chunk_size_tokens` (3 000), `chunk_overlap_tokens` (300).
- `DEFAULT_CONTEXT_TOKENS` — 200 000, used when `max_context_tokens` is `None`.
- `ModelContext` trait — `context_id`, `session_id`, `messages`, `set_messages`, `add_message`, `pop_messages`, `clear_messages`, `token_counter`, `statistics`, `revision`.
- `ConversationContext` has no inherent history accessor: `messages`/`set_messages`/`add_message`/`pop_messages`/`clear_messages`/`statistics`/`revision` are `ModelContext` trait methods (re-exported at crate root), so a caller must import `ModelContext` to read, replace, or clear stored history.
- `ContextEngine` exposes no history reader and no pool enumeration: a caller reaches a context through `get_context(session_id, context_id)` (or `create_context`, which is idempotent and ignores options for an existing pair), and `remove_context` only drops the pool entry without touching session state.
- `ConversationContext::save_state`/`load_state` are public inherent methods, so a `ContextState` round-trip also replaces stored messages and the revision from outside.
- `ConversationContext` — the built-in `ModelContext`: config/model/processors/offload-store accessors, `recall_store()`, `session_memory_store()`, `scope()`, `session()`, `cancel()`, `cancellation()`, `save_state`, `load_state` (history + processor checkpoints), `prepare`; the constructor is engine-only.
- `PreparedContext` — `messages` (system first, then selected context, plus the reload hint when injected), `tools`, `stats: ContextStats`, `token_count`, `overflow`, `window_size`, `events`.
- `ContextStats` — total/per-role message counts and token counts, `total_dialogues`, `compressions`, `tokens_before_compression`, `tokens_after_compression`.
- `ContextEvent` / `ContextEventKind` (`truncated`, `summarized`, `offloaded`, `tool_result_trimmed`), `CompressionStatus`, `CompressReport`, `RecoveryOutcome`, `ObserveOutcome`.
- `ContextState` — serde `{ messages, revision, processors }` snapshot used by session persistence; `processors` is a `BTreeMap<String, serde_json::Value>` keyed by processor name and defaults to empty, so a state written before processors kept counters still loads.
- `ProcessorState` — the `Send + Sync` JSON bag a processor keeps its cross-run counters in: `snapshot`, `restore`, `bump`, `counter`, `set_u64`, `record_tokens`, and a `Clone` that copies the snapshot.
- `ContextProcessor` trait — `name`, `process(messages, &ProcessorContext)`, `save_state`, `load_state`.
- `ProcessorContext` — `config`, `counter`, `model`, `offload`, `scope`, `recall: Option<&dyn RecallArchiveStore>`, `session_memory: Option<&dyn SessionMemoryStore>`, `budget`, `trigger_tokens`, `manual`, `cancellation`.
- `ContextPipeline` — `new`, `from_configs(configs, model, offload)`, `from_configs_with(configs, &PipelineDeps)`, `processors`, `save_state`/`load_state`, `run`.
- `PipelineDeps` — the stores a pipeline build needs: `offload`, `model`, `recall`, `session_memory`; `new(offload)` plus `with_model`/`with_recall`/`with_session_memory` builders.
- `ProcessorConfig` — `Truncation`, `ToolResult`, `Summarization`, `Dialogue`, `CurrentRound`, `RoundLevel`, `FullCompact`, `MicroCompact`, `MessageOffloader`, `ToolResultBudget`, `ToolResultWindow`, `MessageSummaryOffloader` (wire tags `truncation`, `tool_result`, `summarization`, `dialogue`, `current_round`, `round_level`, `full_compact`, `micro_compact`, `message_offloader`, `tool_result_budget`, `tool_result_window`, `message_summary_offloader`).
- Truncation family (`processor::truncate`, `processor::tool_result`, `processor::summarize`): `TruncationProcessor`/`TruncationConfig` (round-aware eviction), `ToolResultProcessor`/`ToolResultConfig` (per-result trim and offload), `SummarizationProcessor`/`SummarizationConfig` (LLM round summarisation); constants `TRUNCATED_MARKER`, `PERSISTED_OUTPUT_TAG`, `CONTEXT_SUMMARY_PREFIX`.
- Compressor family (`processor::compressor`): `DialogueCompressor`, `CurrentRoundCompressor`, `RoundLevelCompressor`, `FullCompactProcessor`, `MicroCompactProcessor` with their `*Config`; shared helpers `MemoryBlockKind` (`Dialogue`/`CurrentRound`/`RoundLevel`), `DIALOGUE_MEMORY_BLOCK`, `CURRENT_ROUND_MEMORY_BLOCK`, `ROUND_LEVEL_MEMORY_BLOCK`, `group_completed_rounds`, `find_last_completed_round_end`, `iter_summary_merge_ranges`, `is_memory_block`, `summarize`, `summarize_offline`, `archive_replaced_messages`, `rollback_recall_archive`, `RecallArchiveMarker`, `split_recall_markers`, `append_recall_markers`.
- Offloader family (`processor::offloader`, `processor::summary_offloader`, `processor::tool_result_window`): `MessageOffloader` + `MessageOffloaderConfig`, `MessageOffloadRole`, `ToolResultBudgetProcessor` + config, `ToolResultWindowProcessor` + config (re-exported from `offloader`), `MessageSummaryOffloader` (`new(config, model, store)`) + config; helpers `wildcard_match`, `OMIT_STRING`, `WINDOWED_OUTPUT_TAG`, `WINDOW_MANAGED_METADATA_KEY`, `summarize_text_offline`, `OFFLOADED_INFO_TAG`.
- Stateful processors expose `pub state: ProcessorState` and count their work: `dialogue` (`compressions`, `rounds_dropped`), `current_round` (`compressions`, `merges`), `round_level` (`compressions`, `blocks`, `merges`, `truncations`), `full_compact` (`compactions`, `session_memory_compactions`), `micro_compact` (`cleared`, `groups`), the offloaders (`offloaded`, `trimmed`), `message_summary_offloader` (`offloaded`, `summaries`), plus `last_tokens_before`/`last_tokens_after` wherever tokens are reported.
- Tokenizers (`tokenizers` module): `TokenizerSpec`, `TokenizerEngine`, `TokenizerSource`, `TokenizerRegistry` (`register`, `resolve` → `TokenizerMatch { spec, kind }`), `TokenizerSelector` (`select` → `SelectedTokenizer`), `ModelEncoding` (`from_model`), `ModelTokenCounter`, `TokenMeasurement`; `load_native_tokenizer` returns `Error::Unsupported`.
- Offload (`offload` module): `OffloadStore` trait (`kind`, `put`, `get`, `remove`), `InMemoryOffloadStore`, `KvOffloadStore` (foundation `KvStore`, `new`/`with_prefix`), `SessionOffloadStore` (`openjiuwen_session::Session`), `OffloadScope`, `OffloadRecord`, `OffloadKind`, `OffloadMarker::{to_marker_string, parse}`, `OFFLOAD_MARKER_PREFIX`, `sanitize_key`, `rehydrate`.
- Compression recall (`recall` module): `RecallArchive`, `RecallChunk`, `RecallTurn`, `TurnRecord`, `RecallArchiveStore` with `InMemoryRecallArchiveStore` and `KvRecallArchiveStore`, `archive_from_messages`, `try_archive_from_messages`, `group_turns`, `split_text`, `Bm25Index`, `recall_compressed_context`, `RecallOptions`, `RecallResult`, `RecallChunkMatch`, `RecallTurnMatch`, `recall_marker`, `parse_recall_marker`, `COMPRESSION_RECALL_MARKER_PREFIX`.
- Session memory (`session_memory` module): `SessionMemoryNotes`, `SessionMemoryStore` with `InMemorySessionMemoryStore`/`KvSessionMemoryStore`, `boundary_message`, `summary_message`, `is_boundary_message`, `reinject_session_memory`, `stamp_message_id`, `SESSION_MEMORY_BOUNDARY`, `SESSION_MEMORY_SUMMARY_INTRO`, `RECENT_MESSAGES_NOTICE`, `CONTEXT_MESSAGE_ID_KEY`.
- Reload (`reload` module): `ReloadLanguage`, `RELOAD_HINT_EN`, `RELOAD_HINT_CN`, `reload_hint`, `has_offload_markers`, `reload_system_message`, `inject_reload_hint`.
- `window` module: `dialogue_round_ranges`, `find_last_n_rounds_start`, `WindowOptions`, `assemble_window`, `strip_leading_tool_messages`, `drop_orphan_tool_results`, `validate_tool_pairs`, `truncate_keeping_rounds`.
- `tokens` module: `count_message`, `count_messages`, `TokenBudget`, `check_budget`; `stats` module: `compute_stats`; `prompts` module: `SUMMARY_SYSTEM_PROMPT`, `SUMMARY_USER_TEMPLATE`, `render_summary_prompt`.

## Constraints

- Errors use `openjiuwen_common::error::{Error, Result}`; invalid config is `Error::Config`, bad message/window input `Error::Validation`, missing context `Error::NotFound`, malformed offload/recall records `Error::Validation`, prepare timeouts `Error::Timeout`, cancellation `Error::Cancelled`, and unavailable native tokenizer artifacts `Error::Unsupported`.
- Async-first on tokio; public processors are `Send + Sync` via `async_trait`; cancellation uses `tokio_util::sync::CancellationToken`; optional per-`prepare` timeouts use `tokio::time::timeout`.
- No blocking I/O on async tasks; offload/recall/session-memory stores are async traits; locks are never held across a model call.
- Every public data type is `serde`-serializable, `snake_case`, tolerant of unknown fields (`#[serde(default)]`), and config round-trips through JSON.
- Token counting always goes through `openjiuwen_foundation::prompt::TokenCounter`; the deterministic default is `ApproxTokenCounter`, and `TokenCounterKind::Model` resolves through the tokenizer registry and falls back to an approximate counter.
- Model calls happen only in summarisation/compression steps and `MessageSummaryOffloader`; they use `Model::invoke` with `ModelRequestConfig` and never stream, and every such step falls back to a deterministic offline summary on a missing model, transport error, or empty reply.
- System messages are never compressed or evicted; tool-call/tool-result pairs are never split by truncation, summarisation, or recall archiving.
- Offloaders rewrite only plain text content: an assistant message that carries `tool_calls`, non-text content, and any text that already contains `[[OFFLOAD:` are left untouched, and role/`name`/`metadata`/`tool_call_id` survive every rewrite.
- No `unsafe`; `tracing` only; files stay around 1000 lines (`engine.rs` sits at the limit, `tokenizers.rs` is 1 053 lines because it owns the model→encoding and registry tables — split it before adding unrelated logic).

## Design Decisions

- The engine owns the default processor pipeline; a context carries an immutable pipeline snapshot created at `create_context`, matching Python's attachment-order semantics.
- `prepare` runs the pipeline on a working copy of the history and never mutates stored messages; explicit `compress` persists the changed history and reports `compressed` exactly once.
- `observe` is the add-hook seam: it appends one message, applies the rolling `max_context_messages` limit, and recomputes stats; passive processors inspect the history inside `prepare`.
- Compression triggers on `token_count > compression_threshold * effective_max_tokens`; `max_tokens = effective_max_tokens - reserve_tokens` (saturating).
- Offload handles are opaque UUIDs scoped by `OffloadScope { session_id, context_id }`, so concurrent sessions cannot read each other's payloads; markers keep Python's `[[OFFLOAD: handle=..., type=in_memory|filesystem]]` shape, and the Rust stores write no filesystem `path` (so markers carry no `path=` field).
- Token counters are stored as `Arc<dyn TokenCounter + Send + Sync>` (foundation's `TokenCounter` has no auto-trait supertraits) so processor futures stay `Send`; `ProcessorContext::counter` carries the same bounds and coerces to `&dyn TokenCounter` at counting call sites.
- Tokenizer selection is explicit spec → registry (`exact`, then unique `family` with `-_ :.` boundaries) → model encoding (`gpt-4*`/`gpt-3.5*`/embeddings → `cl100k_base`, `gpt-4o*`/`o1-`/`o3-`/`gpt-4.1-`/`gpt-5-` → `o200k_base`) → tiktoken `get_bpe_from_model` → 3-characters-per-token estimate; ambiguous family matches are refused rather than guessed.
- BPE vocabularies come from `tiktoken-rs`'s embedded data cached in `OnceLock`s, so model-aware counting is offline; a BPE that cannot be built degrades to the 4-characters-per-token approximation and the measurement reports `tiktoken_unavailable`.
- `TokenCounterKind::Model` uses `ContextConfig::model_name` + `tokenizer_provider` + `tokenizer_registry`; an empty configuration still yields a working approximate counter rather than an error.
- Tool-result processors rewrite message content only (never remove messages), which preserves every `tool_call_id` pairing by construction.
- Summarisation replaces complete oldest dialogue rounds with one `[CONTEXT_SUMMARY]` user message, keeps every system message and the newest `recent_messages`, and never touches the most recent round.
- The compressor family shares one round model: a completed round starts at the first of a contiguous user run, includes assistant `tool_calls` and their paired tool results, and closes on an assistant message with no pending calls; system messages never belong to a round.
- `DialogueCompressor` compresses completed rounds before the keep boundary and keeps the anchor user message raw; `CurrentRoundCompressor` compresses the increment after the last user turn, clamped to a completed API round, then merges contiguous memory blocks; both are benefit-gated and never touch the newest round.
- `RoundLevelCompressor` runs a fixed ladder — per-round blocks, contiguous-block merge, aggressive pass over the protected tail, then deterministic head/tail hard truncation into one `truncated_full_context` block — every rung benefit-gated, so the history is always strictly smaller.
- `FullCompactProcessor` anchors on the last `[FULL_COMPACT_BOUNDARY]`/`[SESSION_MEMORY_BOUNDARY]` system message: the session-memory path reinjects stored notes and keeps the tail after the anchor, the LLM path summarises the active slice (deterministic `Summary:\n[i] role: ...` fallback), and the kept tail extends backwards to include a kept tool result's owning assistant message.
- `MicroCompactProcessor` is fully deterministic: it groups tool results by resolved tool name and replaces the text of all but the newest `keep_recent_per_tool` with `[Old tool result content cleared]`, never removing a message.
- Memory-block envelopes (`[DIALOGUE_MEMORY_BLOCK]`, `[CURRENT_ROUND_MEMORY_BLOCK]`, `[ROUND_LEVEL_MEMORY_BLOCK]`) keep Python's marker line, authority/instruction-status/conflict-priority lines, and `Summary:` body so a resumed Python session can read Rust-written blocks.
- Compressors and `FullCompactProcessor` archive the messages they replace through `archive_replaced_messages` only when `compression_recall.enabled` and `ProcessorContext::recall` is set, then append `[[COMPRESSION_RECALL: id=...]]` as the block's last line; a replacement rejected by a benefit gate calls `rollback_recall_archive`.
- Merge stages strip the individual blocks' recall markers from joined bodies and re-append them deduplicated, so a merged block keeps every handle exactly once.
- Recall chunking is byte-window based (`chunk_size_tokens * 4` bytes with bounded overlap) instead of tokenizer based, so it stays offline; retrieval is two-stage BM25 (`k1 = 1.5`, `b = 0.75`, ASCII/`_`-part/CJK-run+bigram tokens) with per-archive score normalisation and a skip-not-truncate token budget.
- `MessageOffloader` protects indices at or after `min(len - messages_to_keep, newest assistant without tool calls)` when `keep_last_round` is set, and resolves a tool result's name through the owning assistant's `tool_calls` (`name` or `name:glob` protection, `*`/`?` over string arguments).
- `ToolResultBudgetProcessor` re-measures the round after every offload and takes the largest eligible result (oldest on a tie) until the round fits or no candidate remains; `tool_name_allowlist = None` protects nothing.
- `ToolResultWindowProcessor` keeps one global window across all listed tools (suffix match, falling back to the message `name`); results below `min_offload_chars` are compacted in place without a store write and marked with `tool_result_window_managed`.
- `MessageSummaryOffloader` prefers its own model, then the context model, and falls back to `summarize_text_offline`; `from_configs` never fails for a missing model, and the replacement keeps `[offloaded_info]` on its own line.
- Store resolution per context is `ContextOptions` override → engine-level store (`with_stores`) → configuration default: an in-memory recall archive when `compression_recall.enabled`, and an in-memory session-memory store when the pipeline holds `FullCompact` or `MessageSummaryOffloader`.
- `enable_reload` is implemented as a prepared-window hint: `prepare` injects the English reload section after the leading system messages, only when the assembled window carries an offload marker; injection is idempotent and the hint never becomes part of stored history.
- Processor checkpoints ride the existing session seam: `ConversationContext::save_state` fills `ContextState::processors` from `ContextPipeline::save_state`, and `load_state` calls `ContextPipeline::load_state` after replacing the messages. Keys are positional in pipeline order (`name`, `name#2`, `name#3`), `Null` states are omitted, and an unknown key is ignored with a debug log.
- Counters are observability only: a processor bumps them solely when its pass really changed the working history, so a no-op run leaves the checkpoint untouched and no transform ever reads a counter.
- Window assembly applies round truncation first, then message-count truncation where system messages consume the budget, then strips leading `tool` messages (Python parity).
- A `TruncationConfig::max_tokens` below the current size acts regardless of the trigger threshold; the threshold only gates the case where no processor budget is set.
- `TruncationConfig::keep_system = false` drops the oldest system messages first, which is the only way that field is not inert.
- The context pool key is `(session_id, context_id)` with dots in `context_id` normalized to `_`, matching Python's pool identity; the context keeps its original id.
- The processor extension surface is exactly: a config struct (`serde(default)`, snake_case), a `ProcessorConfig` variant (internally tagged `{"type": ...}`), an arm in `ContextPipeline::from_configs_with`, a re-export in `lib.rs`, and a `ContextEventKind` reuse or variant when it reports a change; `default_processors` lists only the two defaulted processors.
- `ProcessorContext` carries no session handle, so a processor persists only through `offload`/`recall`/`session_memory`/`scope` and its own `ProcessorState`, which the pipeline saves and loads around the session checkpoint.
- Context state persistence is a serde `ContextState { messages, revision, processors }` written under session key `context.<context_id>`; the engine wraps a map of context id → state under `context`.
- Every offloader reports `ContextEventKind::Offloaded` with the pre-change indices, the last handle it wrote, and whole-history before/after token counts, and returns `None` when it changed nothing; `cx.manual` bypasses the thresholds, never the candidate rules.

## Routing Table

- `./src/lib.rs` → crate docs, module declaration, public re-exports, lint attributes
- `./src/config.rs` → `ContextConfig`, `TokenCounterConfig`, `TokenCounterKind`, `CompressionRecallConfig`, tokenizer registry config, defaults and validation
- `./src/tokenizers.rs` → model→encoding table, `TokenizerSpec`/`TokenizerRegistry`, `TokenizerSelector`, `ModelTokenCounter`, `SelectedTokenizer`, native-loading deferral
- `./src/tokens.rs` → token-counter accounting helpers, budget check
- `./src/stats.rs` → `ContextStats`, `PreparedContext`, `compute_stats`, per-role accounting
- `./src/events.rs` → `ContextEvent`, `ContextEventKind`, `CompressionStatus`, `CompressReport`, `RecoveryOutcome`, `ObserveOutcome`
- `./src/window.rs` → dialogue rounds, window assembly, tool-pair helpers, round-aware truncation
- `./src/context.rs` → `ModelContext` trait, `ConversationContext`, `ContextState` (messages + processor checkpoints), prepare runner and reload injection
- `./src/engine.rs` → `ContextEngine`, context pool, `observe`/`compress`/recovery, session persistence, store resolution
- `./src/engine_options.rs` → `PrepareOptions`, `ContextOptions`, `DEFAULT_SESSION_ID` (re-exported from `engine`)
- `./src/offload.rs` → `OffloadStore` implementations, `OffloadScope`/`OffloadRecord`/markers, rehydration
- `./src/recall/mod.rs` → recall archive types, store implementations, cross-archive retrieval, markers
- `./src/recall/chunk.rs` → turn detection, section transcript rendering, byte-window chunking
- `./src/recall/bm25.rs` → dependency-free BM25 index and tokenizer
- `./src/session_memory.rs` → notes store, boundary/summary messages, prefix reinjection
- `./src/reload.rs` → reload hint texts, marker detection, idempotent injection
- `./src/prompts.rs` → summarisation system/user prompt templates
- `./src/processor/mod.rs` → `ContextProcessor` trait, `ProcessorContext`, `ProcessorState`, `ProcessorConfig`, `PipelineDeps`, `ContextPipeline` (run + checkpoint save/load)
- `./src/processor/truncate.rs` → `TruncationProcessor` + config (round-aware eviction)
- `./src/processor/summarize.rs` → `SummarizationProcessor` + config (LLM summarisation via `Model`)
- `./src/processor/tool_result.rs` → `ToolResultProcessor` + config (per-result truncation and offload)
- `./src/processor/offloader.rs` → `MessageOffloader`, `ToolResultBudgetProcessor`, shared offloader helpers (protection patterns, placeholder, marker/event builders)
- `./src/processor/tool_result_window.rs` → `ToolResultWindowProcessor` + config (global per-tool window, in-place compaction), re-exported from `offloader`
- `./src/processor/summary_offloader.rs` → `MessageSummaryOffloader` + config, offline head/tail summary
- `./src/processor/compressor/mod.rs` → shared compressor helpers (memory-block envelope, completed-round grouping, merge ranges, offline/LLM summarisation, recall archiving and marker merge)
- `./src/processor/compressor/dialogue.rs` → `DialogueCompressor` (whole-dialogue compaction)
- `./src/processor/compressor/current_round.rs` → `CurrentRoundCompressor` (in-flight round compaction + block merge)
- `./src/processor/compressor/round_level.rs` → `RoundLevelCompressor` (per-round blocks, recursive merge, aggressive pass, hard truncation)
- `./src/processor/compressor/compact.rs` → `FullCompactProcessor` (boundary detection, kept tail, session-memory or LLM full compaction)
- `./src/processor/compressor/micro_compact.rs` → `MicroCompactProcessor` (deterministic per-tool tool-result clearing)
- `./tests/engine.rs` → engine end-to-end suite (prepare/observe/compress/recovery/persistence)
- `./tests/processors.rs` → pipeline ordering, truncation, tool-result offload, summarisation, event serde
- `./tests/tokenizers.rs` → encoding mapping, registry matching, selector ladder, model-counter counts, config integration, native deferral
- `./tests/offload.rs` → marker wire format, key isolation across all three stores, rehydration round-trips
- `./tests/offloaders.rs` → the four offloader processors, placeholder/flag/marker formats, store contents, `ProcessorConfig` tags, pipeline wiring
- `./tests/compressors_dialogue.rs` → dialogue compressor: round grouping, anchor/newest-round/system preservation, pair safety, benefit gate, events
- `./tests/compressors_current_round.rs` → current-round compressor: boundary span, API-round clamp, min-selected gate, merge stage
- `./tests/compressors_round_level.rs` → round-level ladder: blocks per round, pair safety, merge, aggressive pass, hard truncation, events
- `./tests/compressors_compact.rs` → full-compact boundary/kept-tail/session-memory paths and micro-compact clearing, `ProcessorConfig` tags
- `./tests/recall.rs` → turn grouping, byte-window chunking, BM25 ranking, archive stores and scope isolation, retrieval budgets
- `./tests/recall_integration.rs` → compressor recall: archives per compressor, marker placement, merge preservation, hard-truncation prefix, full-compaction paths, rollback, disabled/absent store
- `./tests/session_memory.rs` → notes stores, boundary/summary shapes, prefix reinjection with tool-pair safety, id stamping
- `./tests/reload.rs` → exact reload texts, marker detection, injection placement and idempotence
- `./tests/integration.rs` → engine/context plumbing: store resolution and precedence, `from_configs_with` wiring, prepare reload injection, `ProcessorContext` store fields
- `./tests/processor_state.rs` → `ProcessorState` semantics, pipeline key scheme, one counter test per stateful processor, session save/load round trip, backward-compatible and corrupt state

## Test Strategy

- Unit tests live beside each module (123 tests); integration suites live in `crates/openjiuwen-context/tests/` (247 tests across the files above); doc tests cover the crate-level example paths (10).
- Deterministic offline tests use `ApproxTokenCounter::with_chars_per_token(1)` (via `TokenCounterKind::Characters`) and `openjiuwen_test_support::ScriptedModel` as the `Model`; sessions are real `openjiuwen_session::AgentSession` instances.
- Covered behaviors: window assembly with round/message budgets, leading-tool stripping, per-role stats, overflow detection, every processor, summary replacement with system/recent preservation, tool-pair preservation, offload/rehydrate round-trip, session isolation, `noop`/`compressed`/`busy`, overflow recovery refusal after streamed chunks, config serde round-trip, cancellation, `save_contexts`/`load_contexts` round-trip, corrupt-state tolerance.
- Family-specific coverage: tokenizer mapping/registry/selector ladders and real BPE counts; compressor span math, envelopes, benefit gates, merge and truncation rungs, tool-pair validity, offline fallbacks and `ScriptedModel` paths; offloader trim/placeholder wire formats, keep boundaries, protection patterns, per-round budget order, tool window and managed-flag idempotence; recall chunk boundaries (CJK), BM25 ranking, archive isolation and budget skipping, marker preservation and rollback; session-memory reinjection and reload injection placement.
- `cargo test -p openjiuwen-context --offline` must pass; `cargo clippy -p openjiuwen-context --all-targets --offline -- -D warnings`, `cargo fmt -p openjiuwen-context -- --check`, and `cargo doc -p openjiuwen-context --no-deps --offline` must be clean.

## Deferred (Python `context_engine` features not ported)

- Tokenizer artifact manager and downloads (`huggingface`/`modelscope` sources, cache dirs, locks, sha256 validation), native `tokenizer.json`/`tiktoken.model` loading, and `provider_official` adapters; `load_native_tokenizer` returns `Error::Unsupported` (needs the HuggingFace `tokenizers` runtime and network access).
- Usage/KV-cache telemetry (`ContextUsageAnalyzer`, `SessionKVCacheAggregator`, `context.usage` snapshots) beyond `ContextStats` and compression before/after counts.
- Provider-specific context policies, `ContextWindowChange`/KV-cache eviction detection, the OpenRouter context-window cache, and `model_revision`/`position` diffing.
- Python's adaptive strategy selection inside the compressor family (model-chosen passes, task extraction, JSON-schema memory blocks): the Rust compressors run a fixed deterministic ladder with an optional single summary call.
- Python's adaptive summarisation pipeline inside `MessageSummaryOffloader` (task extraction, JSON-schema output, `offload_data_explanation`, `enable_precise_step`, strategy selection): the Rust processor makes one summary call with a fixed system prompt and falls back to a deterministic head/tail summary.
- `processor/forked/` registry monkey-patching, the LLM-driven session-memory agent (`agent_edit`/`direct_replace` notes extraction, pending-file commit), and the Python `PrefixCompactProcessor`/`SessionMemoryCompressor` class names: Rust keeps the behavior under its own processor names and takes session-memory notes from `SessionMemoryStore`.
- `FullCompactProcessor` state-snapshot reinjection (skills, task status, plan mode, team policy) and the `enable_reload`-style `state_snapshot_max_chars` trimming: the Rust processor compacts history only, because the host state those builders read lives outside this crate.
- Rule-based offloader compression (`processor/forked/offloader/rule_compression/*`: HTML/JSON/log/diff/search-result compressors, query-term routing) and file-based archive dumps (`compression_dump`, `raw_messages.json`, `turns.jsonl`); recall archives are JSON records in the archive store.
- Compression observability hooks beyond `ContextEvent`/`tracing`: `ContextCompressionState` stream frames, `ContextProcessorStateRecorder` history, and the `compaction` request-purpose correlation scope.
