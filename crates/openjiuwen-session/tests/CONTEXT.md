# tests

## Intent

Crate-level behavior tests for `openjiuwen-session`, exercising cross-module behavior that unit tests inside `src/` cannot cover: state scopes through the `Session` trait, streaming delivery, HITL flows, checkpoint/resume round-trips, and tracer output.

## API Surface

One integration test crate per concern:
- `state.rs` — global/component state isolation, merge and `null` deletion, dotted paths, child-session scoping, concurrent updates.
- `stream.rs` — ordering, multiple subscribers, completion marker, late subscribers, mode filtering, bounded-capacity lag behavior.
- `interaction.rs` — approve/reject/answer flows, timeout, cancellation, seeded resume answers, `__interaction__` frames.
- `checkpoint.rs` — memory and file backends: save/load/list/latest/delete/release/exists, resume restores global + component state + trace position.
- `checkpoint_sqlite.rs` — the same contract under `--features sqlite` (file guarded with `#![cfg(feature = "sqlite")]`).
- `tracer.rs` — span lifecycle frames, recording sink, stream sink, trace position.
- `sessions.rs` — factories, metadata, parent delegation, agent/team/workflow specifics, config env resolution.

## Constraints

- Offline and deterministic: no network, no real sleeps (only short `tokio::time::timeout` guards), `tempfile` for file/sqlite roots.
- Start each crate with `#![allow(clippy::unwrap_used, clippy::expect_used)]`; library code stays panic-free.
- Assert exact JSON field names where a wire/snapshot contract exists (`io_state`, `global_state`, `comp_state`, `trace_state`, `type`, `index`, `payload`, `session_id`, `checkpoint_id`, camelCase trace keys).
- Behavior ports name the Python test they mirror in a comment.
- Every test compiles under `cargo test -p openjiuwen-session` with default features; sqlite-only tests are feature-gated.

## Routing Table

No child directories; files are listed in `../CONTEXT.md` under Test Strategy.
