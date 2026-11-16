# `src/context/` — the per-component context's conversation seam

## Intent

This directory holds `history.rs`, the session-backed conversation store behind `ComponentContext::history()`.
It ports Python's `ModelContext.get_messages()` / `add_messages()` (`openjiuwen/core/context_engine/base.py`) onto the workflow session's `__messages__` global lane.
`ComponentContext` itself lives in `../context.rs`; only the history module lives here.
The lane is shared with `components::intent_detection`, so both components see one conversation on one session.
The store never fails and never panics: read and write problems are logged with `tracing::warn!` and recorded as a `HistoryCorruption` that callers can inspect or raise.

## API Surface

`pub const MESSAGES_STATE_KEY: &str = "__messages__"` — the session global-lane key, holding a JSON array of `{"role", "content"}` objects.

`pub trait MessageHistory` (`Send + Sync`, `#[async_trait]`) — `messages() -> Vec<BaseMessage>`, `append(Vec<BaseMessage>)`, `clear()`, plus the defaulted `corruption() -> Option<HistoryCorruption>` that returns `None`; the three data methods keep their existing signatures, so external implementations stay source-compatible.

`pub struct SessionMessageHistory` — `Clone` (clones share the session handle and the corruption record), `new(Arc<dyn Session>)`, `session() -> &Arc<dyn Session>`, a manual `Debug` reporting the lane key, the session id and whether a corruption is recorded, and the `MessageHistory` impl over the lane.

`pub enum HistoryCorruptionReason` — `NotAnArray`, `Unreadable`, `MalformedEntries`, `WriteFailed`; `Copy`, serde `snake_case`, `as_str()` and `Display` (`"not_an_array"`, `"unreadable"`, `"malformed_entries"`, `"write_failed"`).

`pub struct HistoryCorruption { pub reason: HistoryCorruptionReason, pub detail: String }` — serde snake_case, `new(reason, detail)`, `Display` as `"<reason>: <detail>"`.

`pub const WORKFLOW_HISTORY_CORRUPT: u32 = 100_060` and `pub fn history_corrupt_error(&HistoryCorruption) -> Error` — a coded `Error::Workflow` of the form `[100060] message history is corrupt, reason=<reason>, detail='<detail>'`.

Re-exports: `crate::context` re-exports `HistoryCorruption`, `HistoryCorruptionReason`, `MessageHistory`, `SessionMessageHistory` and `MESSAGES_STATE_KEY`; `crate::context::history` is the canonical module path.

## Behavior

A missing lane is a clean, empty history.
A non-array lane value yields empty history plus a `NotAnArray` record whose detail names the key and the JSON type.
An unreadable lane yields empty history plus an `Unreadable` record whose detail names the key and the session error.
An entry that is not an object, whose `role` is missing/unknown/`Other`, or whose `content` is not a string is skipped, and the read records `MalformedEntries` whose detail aggregates `skipped <n> of <total>`.
A failed lane write records `WriteFailed`; a successful write (through `append` or `clear`) clears any record, because the lane then holds exactly what this store wrote.
`messages()` only records; it never clears an older record, so `corruption()` always describes the most recent read or write.

## Constraints

- Offline and deterministic: the store touches only the session state lanes; no clock, network or ordering assumptions.
- No new dependencies: `Arc<parking_lot::Mutex<Option<HistoryCorruption>>>` is the shared interior state.
- No `unsafe`, no panics on user data, no stdout; logging goes through `tracing::warn!`.
- `openjiuwen_common::error::{Error, Result}` only; the `100060` code lives here because `crate::error` is the shared Python code table and `100060` is Rust-only.
- Method signatures of `MessageHistory` are frozen: corruption reporting is additive through the defaulted method.

## Notes for Agents

- `ComponentContext` keeps one `SessionMessageHistory` per context, so every session-backed handle returned by `ComponentContext::history()` shares the record and `ComponentContext::history_corruption()` observes what a component's store reported.
- An overridden store (`ComponentContext::with_history`) reports its own record; a store that keeps the trait default always reports `None`.
- Callers that want to fail hard convert the record with `history_corrupt_error` or `ComponentContext::history_corrupt_error`.
- `100060` sits in the unused gap between `WORKFLOW_COMPONENT_EXECUTION_ERROR` (`100054`) and `WORKFLOW_COMPILE_ERROR` (`100100`), so it cannot collide with a ported Python code.
- Tests: inline unit tests here cover the decode/aggregation helper, every corruption reason (including the failing-session paths), the JSON round-trip, and the error builder; `../../tests/history_corruption.rs` covers the same behavior through the public API and `ComponentContext`.
