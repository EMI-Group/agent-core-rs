# processor::compressor

## Intent

Owns the compressor family: the processors that replace a span of conversation with one protocolised *memory block* instead of dropping it.
Ports Python's `processor/compressor/` and `processor/forked/compressor/` (`dialogue_compressor.py`, `current_round_compressor.py`, `round_level_compressor.py`, `session_memory_compressor.py`, `reinjection/`) to a fixed deterministic ladder with at most one bounded summary call per pass.
The shared round model, the memory-block envelope and recall archiving live in `mod.rs`; each compressor owns its span maths and its own config.
Adaptive strategy selection (`adaptive/`) lets a model choose the pass, and state reinjection (`reinjection/`) restores host state after a compaction; both are off by default and both degrade to deterministic behaviour.
Full compaction's session-memory branch runs a real note extractor — `SessionMemoryAgent` from the crate's `session_memory_agent` module — over the shared `Model` seam, with a deterministic offline fallback, so the branch can write the notes it reinjects instead of only reading them.
Every path works offline and never fails for a missing model, so a context with no credentials still compresses.

## API Surface

`mod.rs` — shared support and the family re-exports:

- `MemoryBlockKind` (`Dialogue` | `CurrentRound` | `RoundLevel`) with `marker`, `default_scope`, `render`, `render_with_scope`, `summary_body`; constants `DIALOGUE_MEMORY_BLOCK`, `CURRENT_ROUND_MEMORY_BLOCK`, `ROUND_LEVEL_MEMORY_BLOCK`, `TRUNCATED_MARKER` (`...[TRUNCATED]...`), `SUMMARY_JOIN_SEPARATOR` (`\n---\n`).
- `group_completed_rounds(messages) -> Vec<Range<usize>>`, `find_last_completed_round_end`, `iter_summary_merge_ranges`, `is_memory_block`.
- `summarize(model, system_prompt, user_prompt, max_tokens, fallback)` (one bounded, non-streaming model call) and `summarize_offline(messages, max_chars)`.
- `archive_replaced_messages(cx, replaced) -> Option<RecallArchiveMarker>`, `rollback_recall_archive`, `RecallArchiveMarker` (`marker`, `append`), `split_recall_markers`, `append_recall_markers`.

Processors (each a `ContextProcessor`, re-exported at the crate root):

- `DialogueCompressor`/`DialogueCompressorConfig` — `messages_threshold`, `tokens_threshold`, `messages_to_keep`, `keep_last_round`, `compression_target_tokens`, `custom_compression_prompt`, `adaptive`; `name()` is `dialogue`, `ProcessorConfig` tag is `dialogue_compressor`.
- `CurrentRoundCompressor`/`CurrentRoundCompressorConfig` — `tokens_threshold`, `messages_to_keep`, `min_selected_tokens_for_compression`, `compression_target_tokens`, `summary_merge_min_blocks`, `prior_context_window_size`, `custom_compression_prompt`, `adaptive`; `name()` is `current_round`, tag `current_round_compressor`.
- `RoundLevelCompressor`/`RoundLevelCompressorConfig` — `trigger_context_ratio`, `target_total_tokens`, `keep_recent_messages`, `compression_call_max_tokens`, `first_pass_target_tokens`, `second_pass_target_tokens`, `third_pass_target_tokens`, `truncate_head_ratio`, `truncated_marker`, `compression_marker`; `name()` is `round_level`, tag `round_level`; `TRUNCATED_FULL_CONTEXT_SCOPE`.
- `FullCompactProcessor`/`FullCompactProcessorConfig` — `trigger_total_tokens`, `compression_call_max_tokens`, `messages_to_keep`, `session_memory_enabled`, `keep_tool_message_pairs`, `reinject_state`, `state_snapshot_max_chars`, `reinject_sections`, `state_marker`; `name()` is `full_compact`, tag `full_compact`; constants `FULL_COMPACT_BOUNDARY`, `FULL_COMPACT_BOUNDARY_TEXT`, `FULL_COMPACT_SUMMARY_INTRO`; fields `model`, `session_memory`, `session_memory_agent`, `reinjection`, `state`; `new`, `with_session_memory`, `with_session_memory_agent`, `with_reinjection`, and the internal `effective_agent`/`refuse_session_memory`/`session_memory_candidate`/`summary_candidate`.
- `MicroCompactProcessor`/`MicroCompactProcessorConfig` — `trigger_threshold`, `compactable_tool_names`, `keep_recent_per_tool`, `cleared_marker`; `name()` is `micro_compact`, tag `micro_compact`; `MICRO_COMPACT_CLEARED_MARKER`.
- Counters and state keys: `dialogue` (`compressions`, `rounds_dropped`), `current_round` (`compressions`, `merges`), `round_level` (`compressions`, `blocks`, `merges`, `truncations`), `full_compact` (`compactions`, `session_memory_compactions`, `session_memory_extractions`, `session_memory_directives`, the `session_memory_refusals` counter and the text keys `session_memory_source`/`session_memory_refusal`), `micro_compact` (`cleared`, `groups`), plus `last_tokens_before`/`last_tokens_after`.
- `adaptive/` — `CompressionStrategy` (`dialogue_summary`/`current_round`/`round_level`/`full_compact`/`micro_compact`/`noop`), `AdaptiveConfig`, `AdaptiveStrategyConfig`, `AdaptiveStrategy` (`new`, `with_model`, `config`, `has_model`, `select`, `deterministic`), `AdaptiveContext`, `StrategyInput`, `StrategyDecision`, `StrategySource`, `StrategyParse`, `MemoryBlockPlan`, `render_memory_block_from_plan`, `render_memory_block_from_plan_with_marker`.
- `reinjection/` — `ContextSnapshotProvider`, `SnapshotRegistry`, `SnapshotContext`, `SnapshotSection`, `RenderedSection`, `is_reinjected_state_message`, `strip_state_header`, `truncate_state_text`, and the marker/instruction constants (`DEFAULT_STATE_MARKER`, `STATE_REINJECTION_MARKER`, `RECOVERED_CONTEXT_TAG`, `RECOVERED_CONTEXT_INSTRUCTION`, `REINJECTED_SECTIONS_METADATA_KEY`, `DEFAULT_STATE_SNAPSHOT_MAX_CHARS`).

Seams the family reads from the processor layer and from the note extractor:

- `ProcessorContext::session_memory_agent` — the per-request `Option<Arc<SessionMemoryAgent>>` override; a caller that sets it wins over the processor's own agent for that one pass.
- `PipelineDeps::session_memory_agent` with the `with_session_memory_agent(Option<Arc<SessionMemoryAgent>>)` builder — the pipeline-level agent the built `FullCompactProcessor` receives.
- `ProcessorState::set_text`/`text` — the string channel a processor uses to report an outcome that is not a change (here: `session_memory_refusal`, `session_memory_source`); `bump`/`counter`/`set_u64` stay the numeric channel.
- `SessionMemoryAgent` (in `../../session_memory_agent.rs`) and what this node uses of it: `new`, `config`, `render_prompt`, `update(messages, current_notes, model) -> SessionMemoryOutcome`; its `SessionMemoryAgentConfig`, `SessionMemoryUpdateMode` (`directives` default, `direct_replace`), `NoteDirective` (`add`/`replace`/`remove`/`direct_replace`), `NoteSource` (`model`/`offline`), `SessionMemoryOutcome` (`written`/`unchanged`/`model_failed`/`skipped`), `NoteEdit`, `extract_offline_notes`, `apply_note_directives`, `render_session_memory_prompt`.

## Constraints

- Errors use `openjiuwen_common::error::{Error, Result}`; invalid adaptive config is `Error::Config`, a malformed model plan is `Error::Validation`, and cancellation is the pipeline's `Error::Cancelled` (checked after a note extraction too).
- Processors are `Send + Sync` async traits (`async_trait`); the pipeline checks cancellation between processors; no lock is held across a model call.
- A pass runs at most one model call per path and never streams; a missing model, transport error, or empty reply falls back to `summarize_offline` rather than an error.
- A session-memory branch that cannot run never fails the pass: it reports its reason and the model/offline summary path takes over.
- The notes a compaction reinjects come from the store or from an extraction this pass committed to it; there is no third producer.
- System messages are never part of a replaced span, and a tool-call/tool-result pair is never split: every span is clamped to the shared completed-round boundaries.
- Memory-block envelopes are wire contract: marker line, `processor`/`type`/`scope`, authority, instruction-status and conflict-priority lines, then `Summary:`; a resumed Python session must be able to read a Rust-written block.
- Recall archiving happens only when `compression_recall.enabled` and `ProcessorContext::recall` is set, and a rejected replacement rolls its archive entry back.
- Adaptive selection is off by default (`AdaptiveConfig::enabled = false`), so an existing pipeline is byte-identical to the pre-adaptive one.
- Reinjection is off by default (`reinject_state = false`) and has no effect without a wired `SnapshotRegistry`.
- No `unsafe`, no `println!`, `tracing` only, no new dependencies; files stay under 1000 lines (state reinjection lives in its own directory rather than in `compact.rs`).

## Design Decisions

- A completed round starts at the first message of a contiguous user run, includes the assistant's `tool_calls` and their paired tool results, and closes on an assistant message with no pending calls; an unclosed trailing round is not a candidate, and system messages never belong to a round.
- `find_last_completed_round_end` clamps any requested span back to a completed boundary, so a compressor can never orphan an in-flight round's tool pair.
- The benefit gate is local and strict: a replacement is only applied when the resulting history is strictly smaller; a pass that would grow or keep the size returns `None` and rolls back its recall archive.
- `summarize_offline` renders `Summary:` plus one `[i] role: text` line per message and keeps a head/tail around `TRUNCATED_MARKER`; it is the fallback for every model path and never fails.
- `DialogueCompressor` replaces every eligible completed span with its own `[DIALOGUE_MEMORY_BLOCK]` (newest span first, so earlier indices stay valid), keeps the anchor user message raw, keeps every system message, and never touches the newest round.
- `CurrentRoundCompressor` compresses only the increment after the last user turn, clamped to a completed API round; it applies `min_selected_tokens_for_compression` before rewriting and then merges contiguous memory blocks once at least `summary_merge_min_blocks` of them exist.
- `RoundLevelCompressor` is a strictly decreasing ladder: per-round blocks, contiguous-block merge, an aggressive pass over the completed rounds the keep window protected, then deterministic head/tail truncation into one `truncated_full_context` block marked with `truncated_marker`; each rung is benefit-gated.
- `FullCompactProcessor` anchors on the last `[FULL_COMPACT_BOUNDARY]` or `[SESSION_MEMORY_BOUNDARY]` system message; the kept tail extends backwards to include a kept tool result's owning assistant message when `keep_tool_message_pairs` is set, and a candidate is accepted only when the history really shrinks.
- The session-memory branch runs first and reinjects the notes over the prefix they cover; the model path summarises the active slice (deterministic `Summary:\n[i] role: ...` fallback) and only runs when the session-memory branch could not apply.
- Stored notes are usable only when they are non-empty *and* `notes_anchor_index` still finds their anchor in the active history; notes anchored on a message that has left the history are stale, not authoritative.
- A branch with no store, no agent, no extractable notes, or no replaceable prefix calls `refuse_session_memory`, which emits one `tracing::warn!` and records the reason in `session_memory_refusal` plus one `session_memory_refusals` bump, because a processor that returns no event has no other reporting channel.
- A session-memory candidate that still exceeds `trigger_total_tokens` is rejected with a `tracing::debug!` and no refusal, since the branch did run and the summary path simply takes over.
- When the store holds no usable notes, `effective_agent` resolves the request agent first and then the processor's own agent; the agent's `update` runs with `cx.model` before the processor model, stamps the anchor message id, and its `Written` notes are `store.put` before the reinjection.
- A failed `store.put` costs durability, not the pass: the notes in hand are still the ones this compaction uses, and the failure is logged with a warning.
- The extraction reports which producer wrote the notes: `session_memory_source` is `store`, `model` or `offline`, `session_memory_extractions` counts the extractions, and `session_memory_directives` records how many directives changed the notes.
- The built pipeline resolves the agent as an explicit `PipelineDeps::session_memory_agent`, else a default `SessionMemoryAgent` whenever a session-memory store is mounted and `session_memory_enabled` is on, else none — in which case a compaction reports why its branch could not run.
- `MicroCompactProcessor` is fully deterministic: it groups tool results by resolved tool name and replaces the text of all but the newest `keep_recent_per_tool` with `cleared_marker`, never removing a message.
- Merges strip the individual blocks' `[[COMPRESSION_RECALL: ...]]` markers from the joined bodies and re-append them deduplicated, so a merged block keeps every handle exactly once and recall can still reach every replaced span.
- Adaptive selection runs after the compressor's own trigger gate: `Noop` leaves history and counters untouched and reports nothing, the compressor's own strategy proceeds with the normal span logic but builds the body from the `MemoryBlockPlan`, and any other strategy returns a `ContextEvent` with empty `messages_changed` naming the recommendation so the processor it names can run instead.
- The deterministic ladder is total and first-match-wins, with every span rung gated by `min_savings_ratio`: no measurable history → `noop`, stale tool results holding the ratio → `micro_compact`, an in-flight round holding it → `current_round`, at least three completed rounds holding it → `round_level`, at least one completed round holding it → `dialogue_summary`, over budget → `full_compact`, else `fallback_strategy`.
- A model-chosen strategy is refused outside `strategy_allowlist`, its reply is bounded by `max_decision_tokens` and never streamed, and a disabled flag, absent model, provider error, empty reply, unparsable reply or unknown strategy name all land on the deterministic ladder with `StrategySource::DeterministicFallback`.
- `MemoryBlockPlan` (`summary`, `tasks`, `decisions`, `constraints`, `files`, `open_questions`) is the durable task state a block carries: `from_model_reply` parses it, `from_messages` extracts it deterministically when `task_extraction` is on, `fill_missing` completes a partial model plan, and `json_schema` is the shape the decision prompt asks for.
- State reinjection runs only after a compaction has been accepted by its benefit gate: `FullCompactProcessor` renders the registry's sections into one `<recovered_context>` user message placed after the summary, and a message already carrying the reinjection marker is never injected twice.
- Each section is trimmed to `state_snapshot_max_chars` and the merged message to `SnapshotRegistry::total_max_chars`, and a trimmed section is marked lossy so a reader can tell it was cut rather than short.
- A `ContextSnapshotProvider` is a pure reader called inline: a provider that cannot render returns `None` and is skipped, and the registry does not `catch_unwind` because a panicking builder would hide a host bug behind a missing section.

## Routing Table

- `./mod.rs` → shared envelope, round grouping, merge ranges, offline/LLM summarisation, recall archiving and marker merging
- `./dialogue.rs` → `DialogueCompressor` + config, adaptive hook
- `./current_round.rs` → `CurrentRoundCompressor` + config, API-round clamp, block merge, adaptive hook
- `./round_level.rs` → `RoundLevelCompressor` + config, the four-rung ladder, hard truncation
- `./compact.rs` → `FullCompactProcessor` + config, boundary detection, kept tail, the session-memory branch (store, note extraction, refusal reporting, reinjection), the model summary path, state-reinjection integration
- `./micro_compact.rs` → `MicroCompactProcessor` + config, per-tool clearing
- `./adaptive/` → `mod.rs` (module docs and re-exports), `strategy.rs` (`CompressionStrategy`, `AdaptiveStrategy`, ladder, selection, outcome), `config.rs` (`AdaptiveConfig`), `render.rs` (block and recommendation rendering), `plan.rs` (`MemoryBlockPlan`, task extraction, JSON schema), `prompt.rs` (decision prompts), `tests.rs` (parsing and ladder tests)
- `./reinjection/` → `mod.rs` (`ContextSnapshotProvider`, `SnapshotRegistry`, `<recovered_context>` rendering and trimming), `tests.rs` (provider fixtures)
- `../../session_memory_agent.rs` → the note extractor this node runs (`SessionMemoryAgent`, the directive protocol, the offline extractor), outside this node
- `../../tests/compressors_dialogue.rs`, `../../tests/compressors_current_round.rs`, `../../tests/compressors_round_level.rs`, `../../tests/compressors_compact.rs` → per-compressor suites
- `../../tests/session_memory_agent.rs` → the extractor and the full-compaction session-memory branch (extraction, refusal reasons, per-request agent precedence, the pipeline default)
- `../../tests/adaptive.rs`, `../../tests/reinjection.rs`, `../../tests/recall_integration.rs` → adaptive selection, state reinjection, and compression-recall integration

## Test Strategy

- Suites: `compressors_dialogue` (12 tests), `compressors_current_round` (11), `compressors_round_level` (10), `compressors_compact` (17), `session_memory_agent` (16), `adaptive` (23), `reinjection` (23), plus `recall_integration` (12) for the archiving paths.
- Unit tests beside `mod.rs` (round grouping), `compact.rs` (config round-trip, API-round grouping, tool-pair extension), `dialogue.rs`, `current_round.rs`, `round_level.rs`, `micro_compact.rs`, `adaptive/tests.rs`, `adaptive/plan.rs` and `reinjection/mod.rs` cover the round model, the plan parsing, the adaptive ladder and section trimming.
- Deterministic offline: a one-character-per-token counter (`ApproxTokenCounter::with_chars_per_token(1)`), `openjiuwen_test_support::ScriptedModel` when a model path is exercised, `InMemorySessionMemoryStore` for the notes, no network and no wall-clock ordering.
- Every suite asserts tool-pair safety and system-message preservation, and the benefit-gate cases assert that a rejected candidate leaves the history and the recall archive unchanged.
- The session-memory coverage proves the reachable branch: a model reply that writes notes, a `direct_replace` reply, the offline fallback on a failed or missing model, an unparsable reply that writes nothing, per-request agent precedence, the refusal reasons (no store, no agent, rejected extraction), the default agent a pipeline mounts, and the extract-then-reinject flow end to end.

## Deferred

- The extractor is a directive protocol, not a tool loop: Python's `edit_file` tool runtime, its `max_iterations` bound and its pending-file commit are not ported, so a note edit is whatever one reply's `ADD`/`REPLACE`/`REMOVE`/`DIRECT_REPLACE` lines say.
- The offline extractor is a cue/path heuristic over the transcript (task, decision and file cues, bounded entries) rather than an LLM, so it cannot derive a note the cues do not point at.
