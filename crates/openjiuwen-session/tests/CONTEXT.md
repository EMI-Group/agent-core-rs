# tests

## Intent

Crate-level behavior tests for `openjiuwen-session`, exercising cross-module behavior that unit tests inside `src/` cannot cover: state scopes through the `Session` trait, streaming delivery, HITL flows, checkpoint/resume round-trips, disk-backed sessions and history, wrapper sessions, the trace-sink hook, and tracer output.

## API Surface

One integration test crate per concern (plus the `support/` module each file that needs it declares):
- `state.rs` — global/component state isolation, merge and `null` deletion, dotted paths, child-session scoping, concurrent updates.
- `stream.rs` — ordering, multiple subscribers, completion marker, late subscribers, mode filtering, bounded-capacity lag behavior.
- `interaction.rs` — approve/reject/answer flows, timeout, cancellation, seeded resume answers, `__interaction__` frames.
- `checkpoint.rs` — memory and file backends: save/load/list/latest/delete/release/exists, resume restores global + component state + trace position.
- `checkpoint_sqlite.rs` — the same contract under `--features sqlite` (file guarded with `#![cfg(feature = "sqlite")]`).
- `checkpoint_redis.rs` — the shared backend contract against the loopback `FakeRedis`, plus Redis specifics: documented key layout, `EXPIRE`/TTL, prefix isolation, cross-session resume on one server, factory wiring from `extra`, missing-URL validation, `AUTH`/`SELECT` from the URL, reconnect after the server hangs up, connection failures as `Error::Session`, and pruning of expired index members.
- `session_controller.rs` — the disk layout (`sessions.json` + `state.data`), workflow/agent snapshot round-trips, scope isolation and the active-session marker, newest-first listings, `list_agents`, delete, invalid ids, atomic writes leaving no temporary files, corrupt documents as `Error::Session`, and the controller cache/flush/save paths.
- `vcs.rs` — sequential commits and parent links, nested/array diff paths, restore without committing, delete/clear, concurrent commits receiving unique ids, and `VersionId` round-trips.
- `session_wrappers.rs` — `SubWorkflowSession` identity/placement/state/end-output, the `RouterSession` read-through and no-ops, trace-sink installation (including shared sinks and checkpoint positions), and the `otel` helper in both feature states plus the `Session` default `Unsupported`.
- `tracer.rs` — span lifecycle frames, recording sink, stream sink, trace position.
- `sessions.rs` — the `Session` contract through `Arc<dyn Session>` (state merge/`null` deletion, stream frames, HITL timeout, provided `snapshot`/`restore`/`subscribe`/`resume` defaults exercised through a minimal fake) and the four concrete sessions (workflow lanes and factories, node executable-id nesting and delegation, agent env/metadata/`post_run`, team ids and child source metadata), plus workflow/agent checkpoint round-trips, a trace-position round-trip, and an end-to-end trace-position save/resume through a fresh `WorkflowSession`.
- `config.rs` — `SessionConfig` defaults and environment resolution through public paths only, with a file-level `ENV_LOCK`/`EnvGuard` so process-global env mutations cannot leak between tests.
- `support/mod.rs` — the shared backend contract (`run_backend_contract(backend, Arc<dyn Checkpointer>, ListOrder::{OldestFirst, NewestFirst})`) used by `checkpoint.rs`, `checkpoint_sqlite.rs`, and `checkpoint_redis.rs`.
- `support/resp_server.rs` — `FakeRedis`, the loopback RESP2 server the Redis tests run against (`start`, `url`, `url_with`, `commands`, `count`, `command`, `hang_up_after_each_reply`), with in-memory strings/sets and Redis's empty-set-disappears rule.

## Constraints

- Offline and deterministic: no network, no real sleeps (only short `tokio::time::timeout` guards), `tempfile` for file/sqlite/disk-session/VCS roots, and the in-process `FakeRedis` instead of a Redis server.
- Start each crate with `#![allow(clippy::unwrap_used, clippy::expect_used)]`; library code stays panic-free.
- Assert exact JSON field names where a wire/snapshot contract exists (`io_state`, `global_state`, `comp_state`, `trace_state`, `type`, `index`, `payload`, `session_id`, `checkpoint_id`, camelCase trace keys).
- Behavior ports name the Python test they mirror in a comment.
- Every test compiles under `cargo test -p openjiuwen-session` with default features; sqlite-only tests are feature-gated, and the `otel` helper tests cover both feature states.

## Routing Table

- `./support/` → shared test helpers (backend contract, `FakeRedis`); it has no `CONTEXT.md` of its own and no child directories
