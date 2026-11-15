# openjiuwen-harness::persistent_agent — long-lived, resumable execution records

## Intent

Persistent-agent support for the `DeepAgent` harness: a durable execution journal, a session-state record, deterministic window assembly, continuation/resume and handoff seams.
Ports the useful, self-contained subset of Python `openjiuwen/harness/persistent_agent/execution/` (v0.1.20) without the host-owned ports, group-chat participation, budget service, archive storage and vendor transports.
The module owns *state that survives an execution*: appended execution entries, the persisted session record, the reconstructed model window, the transfer records a successor or a share bundle consumes.
It does not run the agent loop, does not call a model, and never owns host queueing, delivery or authorization.
All persistence is file-based under caller-provided directories; every clock is injected, so the module is deterministic and offline in tests.

## API Surface

Foundation (`util`, `identity`, `entry`):
- `util`: `canonical_json`, `digest_hex`, `sha256_hex_str`, `percent_encode_identifier`, `percent_decode_identifier`, `is_valid_identifier`, `atomic_write`, `write_json_atomic`, `read_json`, `truncate_chars`, `IDENTIFIER_MAX_CHARS`.
- `identity`: `ExecutionIdentity { namespace_id, agent_id, session_id, request_id, execution_id, source_ref }`, `SessionBinding { agent_id, session_id }`, `LinkRelation { first, confirmation_resume, continuation }`, `SESSION_KEY_PREFIX` (`"pa:"`).
- `SessionBinding`: `new`, `key_prefix()` (`pa:{encoded agent}:{encoded session}:`), `state_key()`, `archive_key(window_seq)`, `archive_index_key()`.
- `ExecutionIdentity`: `new`, `validate`, `session_binding`, `event_id(local_seq)`, `directory_name()`.
- `entry`: `ExecutionEntry { seq, occurred_at, #[serde(flatten)] payload }`, `EntryPayload` (tagged `kind`: `execution_started`, `model_turn`, `tool_intent`, `tool_completed`, `reply_delta`, `execution_paused`, `execution_final`, `window_switched`, `note`), `EntryKind`.
- Entry payload builders: `EntryPayload::{execution_started, model_turn, tool_intent, tool_completed, reply_delta, execution_paused, execution_final, window_switched, note}`, `with_target_ref`, `with_operation_id`, `kind()`.
- Enums: `ToolStatus { succeeded, failed, unknown }`, `PauseReason { awaiting_confirmation, stopped, budget_denied, context_unfit, error }`, `FinalStatus { completed, ignored, cancelled, failed }`, `UsageStatus { known, unknown }`.
- `ModelUsage { status, input_tokens, output_tokens, reason }` with `known`, `unknown`, `total_tokens`.
- Entry helpers: `redact_arguments`, `args_summary`, `result_summary`, `operation_id`, `params_digest`, `ENTRY_SCHEMA_VERSION`, `SUMMARY_LIMIT` (500), `DETAIL_LIMIT` (4096).

Execution journal (`log`):
- `LogCapacity { max_entries, max_bytes, keep_segments }` (serde, `validate`; defaults `4096`, `4 MiB`, `4`).
- `LogMeta { schema_version, encoding, identity, created_at, generation }` — the persisted `meta.json` record.
- `ExecutionLog::open(dir, identity, capacity)` (system clock) / `open_with_clock(dir, identity, capacity, clock)` / `for_workspace(workspace, identity, capacity)` (all async; `<persistent_agent node | root/persistent_agent>/logs/<identity.directory_name()>`), plus `identity`, `dir`, `capacity`, and the async `next_seq`, `cursor` (they take the write mutex).
- Writes: `append(EntryPayload) -> ExecutionEntry`, `append_many`, `rotate_now`.
- Reads: `read() -> LogSnapshot`, `replay() -> Replay`, `digest() -> LogDigest`.
- `LogSnapshot { identity, entries, skipped_invalid, duplicates_removed, conflicting_seqs, missing_ranges, retained_out, unusable }` with `is_usable`, `is_truncated`, `digest`, `replay` (`unusable` is `empty` / `no_usable_entry` / `None`).
- `LogDigest { entries, first_seq, last_seq, counts, tool_outcomes, input_tokens, output_tokens, unknown_usage, missing_ranges, skipped_invalid, duplicates_removed }` with `summary()`; `counts` is keyed by `EntryKind` and `tool_outcomes` by the status's snake-case spelling.
- `Replay { cursor, entries, digest, truncated }`, `LogCursor { next_seq, last_seq, generation }`.
- `LOG_ENCODING = "pa-jsonl/1"`, `LOG_ENTRY_FILE = "entries.jsonl"`, `LOG_META_FILE = "meta.json"`.

Session state (`session`):
- `PersistentSessionRecord` persists Python's keys: `schema_version`, `agent_id`, `session_id`, `messages`, `interruption`, `auto_confirm`, `deepagent_state`, `checkpoint`, `last_group_message_id`, `current_window_id`, `window_seq`, `last_rollover`, `ending`.
- Rust additions documented as deviations: `capabilities` (capability flags), `participation`, `continuation`, and a `#[serde(flatten)] extra` map that preserves unknown fields across a load/save round-trip.
- Nested records: `RecordCheckpoint { request_id, execution_id, iteration_seq, kind, last_group_message_id }`, `CheckpointKind { react_iteration, request_start, rollover }`, `RecordEnding { outcome, request_id, execution_id, spec_fingerprint, reply_target, output_refs }`, `RecordRollover { old_window_id, old_window_seq, new_window_id, new_window_seq, archive_ref, trigger_kind, token_usage }`, `RolloverTrigger { agent, capacity_fallback, request_entry }`.
- `WaitingScene { request_id, reply_target, spec_fingerprint }`, `ParticipationState { decision, reason, event_id, decided_at }`, `ParticipationDecision { ignore, respond, intervene }`, `ContinuationCursor { recovery_ref, predecessor_id, relation, resume_seq, last_group_message_id }`.
- `PersistentSessionRecord` methods: `new`, `validate`, `binding`, `is_waiting`, `waiting_scene`, `with_checkpoint`, `with_ending`, `patch_stopped_calls`, `clear_stale_scene`.
- `SessionStateStore` trait (`load`, `save`, `delete`) plus `FileSessionStore` (`new`, `for_workspace`, `root`, `path_for`, `list`) writing `<root>/<encoded agent>/<encoded session>/state.json` atomically.
- `SessionSummary { binding, waiting, current_window_id, window_seq, has_checkpoint, has_ending, state_path }` (serde) is what `FileSessionStore::list` reports: identity plus boundary flags plus the `state.json` path, never the message window.
- `FileSessionStore::list()` scans `<root>/<encoded agent>/<encoded session>/state.json`, decodes the directory names (falling back to the record's own ids), skips unreadable/invalid records with `tracing::warn!`, and orders by agent id then session id; a missing root yields an empty list.
- `patch_stopped_calls(&[BaseMessage]) -> Vec<BaseMessage>` is also a free function, so a host can settle a window without a record.
- Constants: `SESSION_RECORD_VERSION = 1`, `PLACEHOLDER_TEXT = "结果未取得，实际效果需核实"`, `HANDOFF_TEXT_PREFIX = "【实际交出的最终回答】\n"`, `RECORD_INVALID_PREFIX = "session state record is invalid: "`.

Window assembly (`window`):
- `Capacity { context_window, output_reserve, input_limit, safety_margin }` with `validate`, `available_input`, `rollover_threshold` (`context_window * 4 / 5`), `new_window_limit` (`context_window * 2 / 5`); the default is a `128_000`-token window.
- `WindowPolicy { max_recent_groups, summary_max_chars, min_recent_groups, include_lifecycle_markers }` with `validate`.
- `WindowAssembler::new(capacity)`, `with_counter`, `with_policy`, `with_summarizer`, `capacity()`, `policy()`, `assemble(entries)`, `assemble_with_budget(entries, budget)`.
- `WindowAssembly { messages, tokens, budget, summary, elided, retained_seqs, over_budget }` with `elided_count`.
- `ElidedEntry { seq, kind, reason }`, `ElisionReason { over_budget, counted_only, lifecycle_marker, retention_limit }`.
- `WindowSummarizer` trait plus `ExtractiveSummarizer::new(max_chars)` (default) and `UnsupportedModelSummarizer` (typed `Unsupported`).
- `WindowManager { assembler, capacity }` with `new`, `assemble`, `should_rollover`, `new_window_limit`, `plan`; `WindowPlan { assembly, rollover }`.

Continuation and handoff (`continuation`, `handoff`):
- `Continuation { recovery_ref, instruction, chain_limit }` with `new`, `continued_query` (`"{text}\n\n【继续指示】{instruction}"`), `CONTINUATION_RULES` (literal `&str`).
- `ExecutionLink` (identity ids + `predecessor_id` + `relation`) with `from_identity`, `validate`, `matches`.
- `ExecutionLinkSource` trait + `FileLinkStore` (`new`, `root`, `link_path`; `<root>/<encoded execution>/link.json`).
- `ExecutionLogSource` trait + `RootLogSource` (`new(root, capacity)`, `with_clock`, `root`, `dir_for(execution_id)`; opens `<root>/<encoded execution>/` from its `meta.json`).
- `LogPreview::preview(&LogSnapshot)` renders the deterministic per-execution preview block.
- `resume(store, logs, links, binding, request_id, continuation, assembler) -> ResumePlan`; `ResumePlan { record, cursor, window, chain, background, digest }`.
- `ResumePlan` is `Serialize + Deserialize + PartialEq`: the JSON view is stable (snake_case, unknown-field tolerant, no live handles) and may only change additively, so a host can persist a plan and resume from it later.
- `ResumePlan::seed_messages() -> &[BaseMessage]` and `seed_history() -> Vec<BaseMessage>` return `window.messages` — the reconstructed model window a resumed turn starts from.
- `continuation_background(continuation, successor_execution_id, chain, snapshots, stop)` renders the deterministic background document.
- `HandoffTransfer { from, to, record, cursor, window, next_goal, shared_files }` with `validate`, `HandoffReceipt { successor_session, accepted_entries, transferred_at, successor_execution_id }`.
- `HandoffTarget` trait (`transport`, `accept`), `HandoffTransport { in_process, remote, distributed }` with `as_str`, `reason`, `LocalHandoff::new(clock)`/`take`, `RemoteHandoff`, `DistributedHandoff`.
- `hand_off(target, transfer)` validates and delegates; non-in-process transports return typed `Unsupported`.

Seams (`control`, `emitter`, `share_file`):
- `Control`: synchronous flags behind `parking_lot::Mutex`; `new`, `pause`, `resume`, `request_abort`, `lock_outcome`, `mark_fatal`, `is_paused`, `pause_reason`, `abort_reason`, `outcome_locked`, `fatal`, `snapshot`.
- `ControlSnapshot { paused, pause_reason, abort_reason, outcome_locked, fatal, resumes }`; `AbortReason { cancel, stop }`.
- `PersistentEvent` (tagged `event_type`) mirrors the Python lifecycle frames (`execution_started`, `progress_updated`, `reply_delta`, `model_completed`, `tool_intent`, `tool_completed`, `window_switched`, `execution_paused`, `execution_final`), `EventType` with `as_str`/`is_critical`, `InterceptedCall`.
- `EmittedEvent { identity, local_seq, occurred_at, event, event_id }`; `EventSink` trait; `InMemorySink` (`new`, `events`, `len`, `is_empty`, `clear`); `Emitter` (`new`, `emit`, `emit_ending`, `last_seq`, `identity`, `sink`) with monotonic sequence and `event_id`.
- `ShareFile` export/import service (`new`, `with_clock`, `with_max_bytes`, `root`, `bundle_dir`, `bundle_path`, `export`, `import`, `import_bundle`): `SharedFileEntry { path, share_mode, caption, sha256, size }` with `validate_input`/`validate_persisted`, `ShareMode { upload, attachment }`, `ShareBundle { schema_version, bundle_id, exported_at, binding, files, extra }` with `validate`, `ImportedShareBundle { bundle, files }`.
- `ShareFile::export(bundle_id, binding, files, sources) -> ShareBundle` writes `bundle.json` atomically and copies files under `files/<index>.bin`; `import` verifies sizes and digests.
- `is_logical_path`, `valid_caption`, `BUNDLE_FILE` (`"bundle.json"`), `SHARE_BUNDLE_SCHEMA_VERSION`, `SHARE_FILE_MAX_BYTES` (32 MiB), `MAX_SHARED_FILES` (64), `FILE_PATH_MAX_CHARS`, `CAPTION_MAX_CHARS`, `SHARE_MODES`.

Catalogue and layout (`catalog`, `layout`):
- `ExecutionSummary { execution_id, namespace_id, agent_id, session_id, request_id, source_ref, created_at, generation, entries, last_seq, usable }` (serde, `#[serde(default)]`, unknown fields tolerated) describes one persisted execution without carrying its entries.
- `ExecutionCatalog`: `new(root, capacity)`, `with_clock`, `root()`, `path_for(execution_id)`, async `list() -> Result<Vec<ExecutionSummary>>`, async `open(execution_id) -> Result<ExecutionLog>`; it delegates to an internal `RootLogSource`, so the catalogue and the resume path share one layout.
- `ExecutionCatalog::list()` is best effort per directory: a missing root is `Ok(Vec::new())`, an unreadable entry or a missing/malformed `meta.json` is skipped with `tracing::warn!`, a journal that cannot be opened/read is still listed with `usable = false`, and the result is sorted by `created_at` then `execution_id`.
- `PersistentAgentLayout`: `new(base)`, `for_workspace(workspace)`, `base()`, `logs_root()`, `sessions_root()`, `links_root()`; the single definition of the standard roots (`logs/`, `sessions/`, `links/`) used by `ExecutionLog::for_workspace`, `FileSessionStore::for_workspace` and the resume recipe.
- Layout constants: `PERSISTENT_AGENT_NODE` (`"persistent_agent"`), `LOGS_DIR` (`"logs"`), `SESSIONS_DIR` (`"sessions"`), `LINKS_DIR` (`"links"`).
Rails (`rails`):
- `ExecutionRecordRail` (`HarnessRail`, priority 5): `new`, `with_policy`; `before_tool_call` records `tool_intent`, `after_tool_call` records `tool_completed`, `after_model_call` records `model_turn`.
- `ModelCallRail` (`HarnessRail`, priority 10) backed by `ModelCallGate` (`admit`, `report`) and `ModelAdmission { admitted { reservation_ref }, denied { reason } }`.
- `ToolRecordPolicy { record_read_only, summarize_limit }` with `new`; `READ_ONLY_TOOLS`, `is_read_only_tool`, `MODEL_CALL_ID_EXTRA_KEY`, `deterministic_model_call_id`.

Re-exports:
- Every module item above is also re-exported flat from `persistent_agent` (see `mod.rs`), alongside `rfc3339_from_ms`, `Clock`, `ManualClock` and `SystemClock` from `crate::subagents`.

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

## Consumer contract (outside the crate)

- Every item in `mod.rs` is `pub` and flat re-exported from `persistent_agent`; the crate-root list in `src/lib.rs` names the pre-existing surface (`resume`, `ResumePlan`, `FileSessionStore`, …) but not yet the discovery additions (`ExecutionCatalog`, `ExecutionSummary`, `SessionSummary`, `PersistentAgentLayout`, `percent_decode_identifier`, `PERSISTENT_AGENT_NODE`/`LOGS_DIR`/`SESSIONS_DIR`/`LINKS_DIR`), so those are reachable today only as `openjiuwen_harness::persistent_agent::…`.
- No other part of the crate consumes this module: `src/cli/`, `src/agent/` and `src/protocol/` contain zero references to `persistent_agent`, `ResumePlan`, `PersistentSessionRecord` or `WindowAssembly`, so nothing feeds a `ResumePlan` or its `WindowAssembly::messages` back into `run_once`, `Repl`, `DeepAgent::invoke`/`stream` or `NativeHarness`.
- Re-seeding a model window is kernel-shaped, not module-shaped: `ContextOptions::history` / `ContextEngine::load_contexts` (`crates/openjiuwen-context/src/engine_options.rs`, `engine/persistence.rs:60`) or a custom `HarnessRail::before_model_call` mutating `RailContext::messages`; `ResumePlan::seed_messages`/`seed_history` are the plan-side source those paths can be fed from.
- Discovery is module-side: `ExecutionCatalog::list` enumerates executions under a log root, `FileSessionStore::list` enumerates session records, and `PersistentAgentLayout` names the three standard roots, so a host no longer has to `read_dir` the roots itself or guess the subdirectory names.
- `ResumePlan` is serde data end to end (`Serialize + Deserialize + PartialEq`), so a plan can be cached as JSON and replayed; `seed_messages` is the reconstructed window it carries.
- Identifier rule: `is_valid_identifier` accepts 1..=512 Unicode scalar values with no whitespace and no control characters (`util.rs`); every constructor that takes an id enforces it, and `percent_decode_identifier` is the inverse of the directory-name encoding (invalid escapes or non-UTF-8 yield `None`).

## Routing Table

- `./mod.rs` → module docs, module declarations, flat re-exports, `#[cfg(test)] mod tests`
- `./layout.rs` → `PersistentAgentLayout` (base + `logs_root`/`sessions_root`/`links_root`) and the workspace-node rule the stores share
- `./catalog.rs` → `ExecutionCatalog`/`ExecutionSummary` enumeration over a log root (best-effort list, deterministic order, open by id)
- `./util.rs` → canonical JSON, digests, percent encoding/decoding, identifier checks, atomic file helpers
- `./identity.rs` → `ExecutionIdentity`, `SessionBinding`, session key scheme, `LinkRelation`
- `./entry.rs` → execution-entry types, payloads, redaction/argument helpers, digests
- `./log.rs` → execution journal: append, rotation/retention, snapshot, replay, digest
- `./session.rs` → persisted session record, validation, waiting scene, session store, store enumeration
- `./window.rs` → capacity, token-budgeted window assembly, elision reporting, rollover planning
- `./continuation.rs` → continuation record, execution links, log source, resume, background preview and the serializable `ResumePlan`
- `./handoff.rs` → handoff transfer/receipt, in-process target, unsupported transports
- `./control.rs` → pause/resume/abort flags and outcome locking
- `./emitter.rs` → lifecycle event types, sequence/event ids, event sinks
- `./share_file.rs` → shareable state bundle export/import and path/content validation
- `./rails.rs` → `HarnessRail` implementations recording model turns and tool intents/outcomes
- `./tests.rs` → cross-module offline tests (crash/resume, handoff, share bundle, emitter vs journal, typed refusals) (`#[cfg(test)]` only)

## Design Decisions

- The execution journal is the durability boundary: a resumed session replays the JSONL instead of trusting an in-memory transcript, so a crash between appends loses at most the last partial line.
- One `parking_lot::Mutex` guards `next_seq`/`lines`/`bytes`/`generation` and is only held for the in-memory reads and mutations; a second `parking_lot` gate serializes the file operations, and both are only ever taken on the blocking pool, so no async task holds a lock across an `.await` and no disk IO runs on the async runtime.
- Every journal file operation (open scan, append, rotate, read, replay, metadata write) runs through `tokio::task::spawn_blocking`; the operation gate serializes appends and rotations there, so concurrent appends still produce contiguous, unique sequence numbers.
- `util::truncate_chars` treats `max_chars` as a hard budget: the ellipsis replaces the last kept character, so a truncated result is exactly `max_chars` characters, `0` yields an empty string and `1` yields `…` alone.
- `ExecutionEntry::digest` falls back to a digest over `seq`, `kind` and `occurred_at` when serialization fails, so a failure can never collapse distinct entries onto `digest(null)`.
- `LogPreview::preview` skips a line it cannot serialize (reporting it with `tracing::warn`) instead of emitting an empty line that would read back as a corrupt record.
- A redacted tool payload that cannot be rendered becomes the literal `<unserializable>` marker, never an empty string.
- Rotation renames the active segment to the next free `entries.<generation>.jsonl` (skipping taken names instead of overwriting), rewrites `meta.json` atomically, then prunes the lowest generations beyond `keep_segments`.
- Opening recomputes the counters by scanning the segments and rewrites `meta.json`, so reopening is idempotent and a stale metadata generation never loses entries.
- A read skips empty, unparseable, zero-`seq` and non-RFC-3339 lines into `skipped_invalid`, collapses exact duplicates by `digest()`, drops a whole sequence group whose members disagree, and reports gaps from `1` before retention trimming.
- A torn trailing line stays in place rather than being truncated: the next append opens the file in append mode, and the invalid line is skipped by every later read.
- `LogDigest::tool_outcomes` is keyed by the status's snake-case string because `ToolStatus` is not `Ord`, which keeps the map serde-serializable and its rendering deterministic.
- `LogSnapshot::replay()` reports `generation = 0` because a snapshot does not carry the segment counter; `ExecutionLog::cursor()` and `ExecutionLog::replay()` report the live value.
- `ExecutionLog::for_workspace`, `next_seq` and `cursor` are async because each reads the same write mutex an append takes.
- `ExecutionEntry` flattens an internally tagged `EntryPayload`, giving one self-describing JSON object per line (`{"seq":…,"occurred_at":…,"kind":"tool_intent",…}`).
- Session records keep Python's persisted key names; capability flags, participation and continuation cursor are additive Rust fields, not renames.
- The record validator allows a draft (neither checkpoint nor ending), because a new request persists its window before it reaches a boundary.
- Unknown or empty `outcome` strings are preserved for forward compatibility and only the exact `awaiting_confirmation` marks a waiting scene; a waiter must carry a 64-character lowercase hex `spec_fingerprint`.
- A tool call without an id cannot be answered, so `patch_stopped_calls` leaves it alone; the placeholder is appended after the contiguous result group so the group stays attached to its call.
- `clear_stale_scene` drops only the interruption and the ending; the cursor, window and remembered confirmations stay, so a continuation can still resume the journal.
- Window assembly is synchronous and pure: entries → message groups → newest-first budget fit, with an extractive count summary and a per-entry elision report; an LLM summariser is an injected `WindowSummarizer` seam.
- The default summary is an extractive count summary (per-kind counts, first/last sequence, one hint); a summariser error falls back to a count-only sentence and logs the failure instead of failing the assembly.
- The summary is rendered as a real first message and counted in `WindowAssembly::tokens`, so the budget covers what the model actually receives.
- `min_recent_groups`/`max_recent_groups` are normalised defensively (`max(1)`/`clamp`) so an unvalidated policy cannot make the budget pass drop everything.
- Assistant tool-call groups are indivisible: tool intents and their results are elided together, never split by message count.
- `Capacity` uses Python's integer-floor thresholds (`80%` rollover, `40%` new-window limit) without floating-point arithmetic.
- Continuation chains are host-owned links stored newest-first (`chain[0]` is the recovery execution); `FileLinkStore` is the in-process implementation, and a missing journal or link stops tracing with a stop note instead of failing the resume.
- `CONTINUATION_RULES` is a literal `&str` rather than a template, so the rule block cannot drift from the text a resumed execution is given.
- `resume` passes no successor execution id to `continuation_background`, because the successor's id is assigned by the host after the plan is built.
- Handoff transports other than in-process are deliberately unimplemented and return `Error::Unsupported` naming the transport; `HandoffTransport::InProcess.reason()` is the empty string.
- `LocalHandoff` parks one transfer and lets the successor take it exactly once; a newer accepted transfer replaces a parked one that was never taken.
- `HandoffTransfer::validate` checks the parts belong together (source/target binding, record ownership, cursor reference, share listings) but does not re-run the record validator, which the store already applied on save.
- The emitter advances `local_seq` before the sink call, so a failed ordinary sink write leaves a gap but never a duplicate; critical frames (`tool_intent`, `execution_paused`, `execution_final`) propagate the sink error.
- `emit_ending` retries the same numbered event once, so a host that saw the failed attempt deduplicates by `event_id`.
- Event spellings are snake_case (`execution_started`), where Python's lifecycle frames are dotted (`execution.started`).
- Model call ids are deterministic (`mc_` + 32 hex characters of a digest over identity + iteration) where Python uses `uuid4`; the id stays opaque and stable across a retry.
- `ModelCallRail` answers a denial with `Error::Config` naming the gate's reason; the task loop owns the pause, so the rail owns only the decision's record.
- A `ModelCallGate::report` failure is logged and never aborts the run, and a missing tool-call id gets a synthetic stable key (`name:index`) so an intent and its outcome still match.
- Tool intents are recorded only for non-read-only tools; read-only calls still produce a `tool_completed` receipt without an `operation_id`, and a failed call that had an intent is recorded as `unknown`, not `failed`.
- Share bundles copy file bytes under `files/` and record `sha256` + size; import verifies both, so a tampered bundle is rejected rather than silently accepted.
- The catalogue is a view, never a second store: `ExecutionCatalog` owns a `RootLogSource`, so listing, opening and `resume` cannot disagree about where a journal lives or how it is opened.
- Listing is best effort and deterministic: a corrupt `meta.json` is skipped, a journal that exists but cannot be read is reported as `usable = false`, and the fixed sort (`created_at`, then `execution_id`) makes a listing reproducible across runs.
- `PersistentAgentLayout` is the one place the standard roots are spelled out; `ExecutionLog::for_workspace` and `FileSessionStore::for_workspace` derive their paths from it, so the directory names can never drift apart.
- Session `list()` decodes the directory names it walked (the store's own encoding is the inverse) and falls back to the record's ids only when a name does not decode, so a summary describes the path the file was actually found at.
- The plan's JSON view is a contract: `ResumePlan` is data only, so serializing it and parsing it back is lossless, and a host resumes by feeding `seed_messages` into its model window.

## Test Strategy

- Inline unit tests per module: canonical JSON/digest/percent-encoding, entry serde round-trips and redaction, log append/rotation/replay/digest, hard-budget truncation at every boundary, digest and redaction serialization-failure fallbacks, log preview skipping unrenderable lines, 32 concurrent appends staying contiguous and replayable, session record round-trip with unknown fields, validation rules, window assembly under tight/loose budgets with elision reports, control flags, emitter sequencing and failure policy, share bundle round-trip and tamper rejection, handoff transfer, continuation links and resume.
- Cross-module flows live in `src/persistent_agent/tests.rs` (an integration file under `crates/openjiuwen-harness/tests/` would also work but is outside this module's write scope): crash-and-reopen resume, in-process handoff, share-bundle round trip with tamper rejection, emitter-vs-journal sequence independence, the exact `Unsupported` messages, and the discovery recipe (layout roots → journal + record + link on disk → catalogue/session listing → `resume` → plan JSON round trip carrying the run's text).
- All tests use `tempfile::TempDir`, `ManualClock`, file-backed stores and no sleeps; nothing opens a socket or reads the wall clock.
- Definition of done: `cargo test -p openjiuwen-harness --offline` (default, `--no-default-features`, `--all-features`), `cargo fmt -p openjiuwen-harness`, `cargo clippy -p openjiuwen-harness --all-targets --all-features --offline -- -D warnings`, `cargo doc -p openjiuwen-harness --no-deps`.

## Deferred areas (typed `Unsupported`)

- Remote handoff transport: `Error::unsupported("persistent agent handoff over a remote transport is not implemented; use an in-process HandoffTarget")`.
- Distributed handoff transport: `Error::unsupported("persistent agent handoff over a distributed transport is not implemented; use an in-process HandoffTarget")`.
- Model-backed window summarisation: `UnsupportedModelSummarizer` returns `Error::unsupported("persistent window summarisation requires an injected WindowSummarizer; the model-backed summariser is not implemented")`.
- Host KV / Redis / database session stores and distributed execution-log backends: only `FileSessionStore` and the filesystem/Root log source exist; hosts implement `SessionStateStore` / `ExecutionLogSource` / `ExecutionLinkSource`.
- Group-chat participation, the budget admission service, permission/workspace rails, group-history readers and the `share_file` agent tool: host-owned in Python and not ported here.
- Window archive KV layout and archive-index rebuild: the window manager plans rollovers; archival storage stays with the host `SessionStateStore`.
- The `read_execution_record` paging tool, `search_session_history` and `new_context`: referenced by the continuation rules and the read-only tool list, but not mounted here.
