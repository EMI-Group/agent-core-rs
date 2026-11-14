# openjiuwen-harness::persistent_agent — long-lived, resumable execution records

## Intent

Persistent-agent support for the `DeepAgent` harness: a durable execution journal, a session-state record, deterministic window assembly, continuation/resume and handoff seams.
Ports the useful, self-contained subset of Python `openjiuwen/harness/persistent_agent/execution/` (v0.1.20) without the host-owned ports, group-chat participation, budget service and vendor transports.
The module owns *state that survives an execution*: appended execution entries, the persisted session record, the reconstructed model window, and the transfer records a successor or a share bundle consumes.
It does not run the agent loop, does not call a model, and never owns host queueing, delivery or authorization.
All persistence is file-based under caller-provided directories; every clock is injected, so the module is deterministic and offline in tests.

## API Surface

Foundation (`util`, `identity`, `entry`):
- `util`: `canonical_json`, `digest_hex`, `sha256_hex_str`, `percent_encode_identifier`, `is_valid_identifier`, `atomic_write`, `write_json_atomic`, `read_json`, `truncate_chars`.
- `identity`: `ExecutionIdentity { namespace_id, agent_id, session_id, request_id, execution_id, source_ref }`, `SessionBinding { agent_id, session_id }`, `LinkRelation { first, confirmation_resume, continuation }`, `IDENTIFIER_MAX_CHARS`, `SESSION_KEY_PREFIX`.
- `SessionBinding` keys: `key_prefix()` (`pa:{encoded agent}:{encoded session}:`), `state_key()`, `archive_key(window_seq)`, `archive_index_key()`.
- `ExecutionIdentity`: `new`, `validate`, `session_binding`, `event_id(local_seq)`, `directory_name()`.
- `entry`: `ExecutionEntry { seq, occurred_at, #[serde(flatten)] payload }`, `EntryPayload` (tagged `kind`: `execution_started`, `model_turn`, `tool_intent`, `tool_completed`, `reply_delta`, `execution_paused`, `execution_final`, `window_switched`, `note`), `EntryKind`.
- Enums: `ToolStatus { succeeded, failed, unknown }`, `PauseReason { awaiting_confirmation, stopped, budget_denied, context_unfit, error }`, `FinalStatus { completed, ignored, cancelled, failed }`, `UsageStatus { known, unknown }`.
- `ModelUsage { status, input_tokens, output_tokens, reason }` with `known`, `unknown`, `total_tokens`.
- Entry helpers: `redact_arguments`, `args_summary`, `result_summary`, `operation_id`, `params_digest`, `ENTRY_SCHEMA_VERSION`, `SUMMARY_LIMIT`, `DETAIL_LIMIT`.

Execution journal (`log`):
- `LogCapacity { max_entries, max_bytes, keep_segments }` (serde, validated; defaults `4096`, `4 MiB`, `4`).
- `ExecutionLog::open(dir, identity, capacity)` (uses `SystemClock`) / `open_with_clock(dir, identity, capacity, clock)` / `for_workspace(workspace, identity, capacity)` (both async; `<persistent_agent node | root/persistent_agent>/logs/<identity.directory_name()>`), plus `identity`, `dir`, `capacity`, and the async `next_seq`, `cursor` (they read the write mutex).
- Writes: `append(EntryPayload) -> ExecutionEntry`, `append_many`, `rotate_now`.
- Reads: `read() -> LogSnapshot`, `replay() -> Replay`, `digest() -> LogDigest`.
- `LogSnapshot { identity, entries, skipped_invalid, duplicates_removed, conflicting_seqs, missing_ranges, retained_out, unusable }` with `is_usable`, `is_truncated`, `digest`, `replay` (`unusable` is `empty` / `no_usable_entry` / `None`).
- `LogDigest { entries, first_seq, last_seq, counts, tool_outcomes, input_tokens, output_tokens, unknown_usage, missing_ranges, skipped_invalid, duplicates_removed }` with `summary()`; `tool_outcomes` is keyed by the snake-case status spelling (`succeeded`/`failed`/`unknown`) because `ToolStatus` is not `Ord`.
- `Replay { cursor, entries, digest, truncated }`, `LogCursor { next_seq, last_seq, generation }`.
- `LOG_ENCODING = "pa-jsonl/1"`, `LOG_ENTRY_FILE = "entries.jsonl"`, `LOG_META_FILE = "meta.json"`.

Session state (`session`):
- `PersistentSessionRecord` persists Python's keys: `schema_version`, `agent_id`, `session_id`, `messages`, `interruption`, `auto_confirm`, `deepagent_state`, `checkpoint`, `last_group_message_id`, `current_window_id`, `window_seq`, `last_rollover`, `ending`.
- Rust additions documented as deviations: `capabilities` (capability flags), `participation`, `continuation`, and a `#[serde(flatten)] extra` map that preserves unknown fields across a load/save round-trip.
- Nested records: `RecordCheckpoint { request_id, execution_id, iteration_seq, kind, last_group_message_id }`, `CheckpointKind { react_iteration, request_start, rollover }`, `RecordEnding { outcome, request_id, execution_id, spec_fingerprint, reply_target, output_refs }`, `RecordRollover { old_window_id, old_window_seq, new_window_id, new_window_seq, archive_ref, trigger_kind, token_usage }`, `RolloverTrigger { agent, capacity_fallback, request_entry }`.
- `WaitingScene { request_id, reply_target, spec_fingerprint }`, `ParticipationState`, `ParticipationDecision { ignore, respond, intervene }`, `ContinuationCursor { recovery_ref, predecessor_id, relation, resume_seq, last_group_message_id }`.
- `PersistentSessionRecord` methods: `new`, `validate`, `binding`, `is_waiting`, `waiting_scene`, `patch_stopped_calls`, `clear_stale_scene`.
- `SessionStateStore` trait (`load`, `save`, `delete`) plus `FileSessionStore` (`new`, `for_workspace`, `path_for`) writing `<root>/<encoded agent>/<encoded session>/state.json` atomically.
- Constants: `SESSION_RECORD_VERSION = 1`, `PLACEHOLDER_TEXT = "结果未取得，实际效果需核实"`, `HANDOFF_TEXT_PREFIX`, `RECORD_INVALID_PREFIX`.

Window assembly (`window`):
- `Capacity { context_window, output_reserve, input_limit, safety_margin }` with `validate`, `available_input`, `rollover_threshold` (`context_window * 4 / 5`), `new_window_limit` (`context_window * 2 / 5`).
- `WindowPolicy { max_recent_groups, summary_max_chars, min_recent_groups, include_lifecycle_markers }`.
- `WindowAssembler::new(capacity)`, `with_counter`, `with_policy`, `with_summarizer`, `assemble(entries)`, `assemble_with_budget(entries, budget)`.
- `WindowAssembly { messages, tokens, budget, summary, elided, retained_seqs, over_budget }` with `elided_count`.
- `ElidedEntry { seq, kind, reason }`, `ElisionReason { over_budget, counted_only, lifecycle_marker, retention_limit }`.
- `WindowSummarizer` trait plus `ExtractiveSummarizer` (default) and `UnsupportedModelSummarizer` (typed `Unsupported`).
- `WindowManager { assembler, capacity }` with `assemble`, `should_rollover`, `new_window_limit`, `plan`; `WindowPlan { assembly, rollover }`.

Continuation and handoff (`continuation`, `handoff`):
- `Continuation { recovery_ref, instruction, chain_limit }` with `new`, `continued_query` (`"{text}\n\n【继续指示】{instruction}"`), `CONTINUATION_RULES`.
- `ExecutionLink` (identity ids + `predecessor_id` + `relation`) with `validate`.
- `ExecutionLinkSource` trait + `FileLinkStore` (`<root>/<encoded execution>/link.json`).
- `ExecutionLogSource` trait + `RootLogSource`.
- `resume(store, logs, links, binding, continuation, assembler) -> ResumePlan`; `ResumePlan { record, cursor, window, chain, background, digest }`.
- `continuation_background(chain, snapshots, continuation)` renders the deterministic background document.
- `HandoffTransfer { from, to, record, cursor, window, next_goal, shared_files }`, `HandoffReceipt { successor_session, accepted_entries, transferred_at, successor_execution_id }`.
- `HandoffTarget` trait (`transport`, `accept`), `HandoffTransport { in_process, remote, distributed }`, `LocalHandoff` (`new`, `take`), `RemoteHandoff`, `DistributedHandoff`.
- `hand_off(target, transfer)` validates and delegates; non-in-process transports return typed `Unsupported`.

Seams (`control`, `emitter`, `share_file`):
- `Control`: async-free flags via `parking_lot::Mutex`; `pause`, `resume`, `request_abort`, `lock_outcome`, `mark_fatal`, `is_paused`, `abort_reason`, `outcome_locked`, `fatal`, `snapshot`.
- `ControlSnapshot { paused, pause_reason, abort_reason, outcome_locked, fatal, resumes }`; `AbortReason { cancel, stop }`.
- `PersistentEvent` (tagged `event_type`) mirrors the Python lifecycle frames (`execution_started`, `progress_updated`, `reply_delta`, `model_completed`, `tool_intent`, `tool_completed`, `window_switched`, `execution_paused`, `execution_final`), `EventType`, `InterceptedCall`.
- `EmittedEvent { identity, local_seq, occurred_at, event, event_id }`; `EventSink` trait; `InMemorySink`; `Emitter` (`new`, `emit`, `emit_ending`, `last_seq`) with monotonic sequence and `event_id`.
- `ShareFile` export/import service: `SharedFileEntry { path, share_mode, caption }`, `ShareMode { upload, attachment }`, `ShareBundle { schema_version, bundle_id, exported_at, binding, files }`, `ImportedShareBundle { bundle, files }`.
- `ShareFile::export(bundle, sources) -> PathBuf` writes `bundle.json` atomically and copies files under `files/`; `import(path)` verifies sizes and digests.
- `is_logical_path`, `valid_caption`, `SHARE_FILE_MAX_BYTES` (32 MiB), `MAX_SHARED_FILES` (64), `SHARE_MODES`.

Rails (`rails`):
- `ExecutionRecordRail` (`HarnessRail`, priority 5): `before_tool_call` records `tool_intent`, `after_tool_call` records `tool_completed`, `after_model_call` records `model_turn`.
- `ModelCallRail` (`HarnessRail`, priority 10) backed by `ModelCallGate` (`admit`, `report`) and `ModelAdmission { admitted, denied }`; records `model_turn` with usage after the call.
- `ToolRecordPolicy { record_read_only, summarize_limit }`; `READ_ONLY_TOOLS`, `is_read_only_tool`, `MODEL_CALL_ID_EXTRA_KEY`, `deterministic_model_call_id`.

## Constraints

- Only file-based, offline persistence: caller-provided root directories, temp dirs in tests, no network, no external database or broker.
- Every timestamp comes from an injected `Clock` (re-exported from `crate::subagents`) and is rendered RFC 3339 UTC with millisecond precision.
- Async-first on tokio: file IO through `tokio::fs`; `parking_lot::Mutex` guards never span an `.await`.
- Errors are `openjiuwen_common::error::{Error, Result}`; no panics or `unwrap` outside tests; deferred capabilities return `Error::Unsupported` with a precise reason.
- Serde is snake_case and unknown-field tolerant; persisted records keep Python key names where Python has one, and `extra` preserves unknown keys.
- Canonical JSON (sorted keys, compact separators) backs every digest and id; sha256 comes from `openjiuwen_sysop::util::sha256_hex`.
- Logs are append-only JSONL with atomically written metadata; rotation retains whole segments and never rewrites an existing segment.
- Rails never call the model or execute tools; they append records and publish typed extension points (`ModelCallGate`, `ToolRecordPolicy`).
- Files stay around or below 1000 lines; shared helpers live in `util`.

## Routing Table

- `./mod.rs` → module docs, module declarations, flat re-exports, cross-module constants
- `./util.rs` → canonical JSON, digests, percent encoding, identifier checks, atomic file helpers
- `./identity.rs` → `ExecutionIdentity`, `SessionBinding`, session key scheme, `LinkRelation`
- `./entry.rs` → execution-entry types, payloads, redaction/argument helpers, digests
- `./log.rs` → execution journal: append, rotation/retention, snapshot, replay, digest
- `./session.rs` → persisted session record, validation, waiting scene, session store
- `./window.rs` → capacity, token-budgeted window assembly, elision reporting, rollover planning
- `./continuation.rs` → continuation record, execution links, resume and background preview
- `./handoff.rs` → handoff transfer/receipt, in-process target, unsupported transports
- `./control.rs` → pause/resume/abort flags and outcome locking
- `./emitter.rs` → lifecycle event types, sequence/event ids, event sinks
- `./share_file.rs` → shareable state bundle export/import and path/content validation
- `./rails.rs` → `HarnessRail` implementations recording model turns and tool intents/outcomes
- `./tests.rs` → cross-module offline tests (crash/resume, handoff, share bundle) (`#[cfg(test)]` only)

## Design Decisions

- The execution journal is the durability boundary: a resumed session replays the JSONL instead of trusting an in-memory transcript, so a crash between appends loses at most the last partial line.
- One `tokio::sync::Mutex` guards `next_seq`/`lines`/`bytes`/`generation`, and reads take the same mutex, so a rotation can never race an append or a read.
- Rotation renames the active segment to the next free `entries.<generation>.jsonl` (skipping taken names instead of overwriting), rewrites `meta.json` atomically, then prunes the lowest generations beyond `keep_segments`; opening recomputes counters by scanning the segments and rewrites the metadata, so reopening is idempotent.
- A read skips empty, unparseable and invalid lines (zero `seq` or a non-RFC 3339 timestamp) into `skipped_invalid`, collapses exact duplicates by `digest()`, drops a whole sequence group whose members disagree, and reports gaps from `1` before retention trimming.
- `LogSnapshot::replay()` derives its cursor from the entries with `generation = 0` (a snapshot does not carry the segment counter); `ExecutionLog::cursor()` and `ExecutionLog::replay()` report the live values.
- `ExecutionEntry` flattens an internally tagged `EntryPayload`, giving one self-describing JSON object per line (`{"seq":…,"occurred_at":…,"kind":"tool_intent",…}`).
- Session records keep Python's persisted key names; capability flags, participation and continuation cursor are additive Rust fields, not renames.
- Window assembly is synchronous and pure: entries → message groups → newest-first budget fit, with an extractive summary and a per-entry elision report; an LLM summariser is an injected `WindowSummarizer` seam.
- Assistant tool-call groups are indivisible: tool intents and their results are elided together, never split by message count.
- `Capacity` uses Python's integer-floor thresholds (`80%` rollover, `40%` new-window limit) without floating-point arithmetic.
- Continuation chains are host-owned links; `FileLinkStore` is the in-process implementation, and an unreadable link stops tracing without failing the resume.
- Handoff transports other than in-process are deliberately unimplemented and return `Error::Unsupported` naming the transport.
- The emitter advances `local_seq` before the sink call, so a failed ordinary sink write leaves a gap but never a duplicate; critical frames (`tool_intent`, `execution_paused`, `execution_final`) propagate the sink error.
- Model call ids are deterministic (`mc_` + digest prefix of identity + iteration) where Python uses `uuid4`; the id stays opaque and stable across a retry.
- Tool intents are recorded only for non-read-only tools; read-only calls still produce a `tool_completed` receipt without an `operation_id`.
- Share bundles copy file bytes under `files/` and record `sha256` + size; import verifies both, so a tampered bundle is rejected rather than silently accepted.

## Deferred areas (typed `Unsupported`)

- Remote and distributed handoff transports: `Error::unsupported("persistent agent handoff over a remote transport is not implemented; use an in-process HandoffTarget")` and the distributed counterpart.
- Model-backed window summarisation: `UnsupportedModelSummarizer` returns `Error::unsupported("persistent window summarisation requires an injected WindowSummarizer; the model-backed summariser is not implemented")`.
- Host KV / Redis / database session stores and distributed execution-log backends: only `FileSessionStore` and the filesystem log exist; hosts implement `SessionStateStore` / `ExecutionLogSource`.
- Group-chat participation, budget admission service, permission/workspace rails, group-history readers and the `share_file` agent tool: host-owned in Python and not ported here.
- Window archive KV layout and archive-index rebuild: the window manager plans rollovers; archival storage stays with the host `SessionStateStore`.
- `read_execution_record` paging tool, `search_session_history`, `new_context`: not mounted here.

## Test Strategy

- Inline unit tests per module: canonical JSON/digest/percent-encoding, entry serde round-trips and redaction, log append/rotation/replay/digest, session record round-trip with unknown fields, validation rules, window assembly under tight/loose budgets with elision reports, control flags, emitter sequencing and failure policy, share bundle round-trip and tamper rejection, handoff transfer, continuation links and resume.
- Cross-module tests in `tests.rs`: append entries, simulate a crash by dropping and reopening the same directory, resume and assert cursor/window/digest; hand a state record through `LocalHandoff`; export/import a share bundle.
- All tests use `tempfile::TempDir`, `ManualClock`, `ScriptedModel`-free offline fakes and no sleeps.
- Definition of done: `cargo test -p openjiuwen-harness --offline` (default, `--no-default-features`, `--all-features`), `cargo fmt -p openjiuwen-harness`, `cargo clippy -p openjiuwen-harness --all-targets --all-features --offline -- -D warnings`, `cargo doc -p openjiuwen-harness --no-deps`.
