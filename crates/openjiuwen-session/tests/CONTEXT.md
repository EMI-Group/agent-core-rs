# tests

## Intent

Crate-level behavior tests for `openjiuwen-session`, exercising cross-module behavior that unit tests inside `src/` cannot cover: state scopes through the `Session` trait, streaming delivery and send outcomes, HITL flows, checkpoint/resume round-trips over every backend (clear text, TLS, and the pickle interop codec), disk-backed sessions with their link graph, local history with refs/rewinds/forks, wrapper sessions, the trace-sink hook and its `otel` outcomes, and tracer output.

## API Surface

One integration test crate per concern (plus the `support/` module each file that needs it declares):
- `state.rs` — global/component state isolation, merge and `null` deletion, dotted paths, child-session scoping, concurrent updates.
- `stream.rs` — ordering, multiple subscribers, completion marker, late subscribers, mode filtering, bounded-capacity lag behavior, every `StreamSendOutcome` (delivered/no-subscriber/closed/mode-disabled), and the `StreamTraceSink` position advancing only on delivery.
- `interaction.rs` — approve/reject/answer flows, timeout, cancellation, seeded resume answers, `__interaction__` frames.
- `checkpoint.rs` — memory and file backends: save/load/list/latest/delete/release/exists, resume restores global + component state + trace position.
- `checkpoint_sqlite.rs` — the same contract under `--features sqlite` (file guarded with `#![cfg(feature = "sqlite")]`).
- `checkpoint_redis.rs` — the shared backend contract against the loopback `FakeRedis`, plus Redis specifics: documented key layout, `EXPIRE`/TTL, prefix isolation, cross-session resume on one server, factory wiring from `extra`, missing-URL validation, `AUTH`/`SELECT` from the URL, reconnect after the server hangs up, connection failures as `Error::Session`, pruning of expired index members, and the deadline behavior against a `Blackhole` stub listener (typed `Error::Timeout` on `save`/`load`/`list` within the bound, a concurrent second call that dials instead of waiting behind the hung exchange, and a timed-out connection that is dropped instead of reused).
- `checkpoint_redis_tls.rs` — the same contract over the server's TLS mode, trusting the fixture CA through `tls_ca_file`, plus `uses_tls` scheme selection, a TLS client against a clear-text server, an untrusted certificate (PEM and DER), a malformed CA file, the factory carrying `tls_ca_file`, and a CA file without a TLS URL.
- `checkpoint_pickle.rs` — the codec: protocol 0/2/4 streams, every supported scalar and container, memo references, protocol-4 encoding pinned byte for byte, `dumps`/`loads` round-trips, every typed rejection (unknown opcode, truncation, non-string key, unresolvable memo, unrepresentable values), and the `checkpoint_from_pickle`/`checkpoint_to_pickle` mapping including a missing id and a bad payload.
- `session_controller.rs` — the disk layout (`sessions.json` + `state.data`), workflow/agent snapshot round-trips, scope isolation and the active-session marker, newest-first listings, `list_agents`, delete, invalid ids, atomic writes leaving no temporary files, corrupt documents as `Error::Session`, the controller cache/flush/save paths, the strict scope-resolution errors, the `upsert_in`/`save_in` explicit-scope paths, and concurrent saves of distinct sessions keeping every index entry.
- `session_controller_links.rs` — the persisted link graph: parent/fork links, traversal (`parent_of`/`children_of`/`forks_of`/`roots`/`leaves`/`chain`/`ancestors`/`descendants`/`depth`), a stored cycle as `Error::Session`, self and cycle-creating links as `Error::Validation`, delete dropping exactly one entry, corrupt documents, the `LinksIndex` document round-trip, concurrent writers through cloned stores, and the strict scope errors.
- `vcs.rs` — sequential commits and parent links, nested/array diff paths, restore without committing, delete/clear, concurrent commits receiving unique ids, `VersionId` round-trips, invalid ids, refs (`set_ref`/`resolve_ref`/`delete_ref`/`list_refs`, the `HEAD`/`main` advance on commit, the `head()` fallback to `latest()`), soft and hard rewinds, and `fork`.
- `session_wrappers.rs` — `SubWorkflowSession` identity/placement/state/end-output, the `RouterSession` read-through plus every typed rejection and its `end_stream`/`close` no-ops, trace-sink installation (including shared sinks and checkpoint positions), and the `otel` helpers in both feature states (`try_install_trace_sink`, `TraceSinkInstall`, `is_enabled`, the compatibility `install_trace_sink`, the outcome's snake_case wire names) plus the `Session` default `Unsupported`.
- `tracer.rs` — span lifecycle frames, recording sink, stream sink, trace position.
- `sessions.rs` — the `Session` contract through `Arc<dyn Session>` (state merge/`null` deletion, stream frames, HITL timeout, provided `snapshot`/`restore`/`subscribe`/`resume` defaults exercised through a minimal fake) and the four concrete sessions (workflow lanes and factories, node executable-id nesting and delegation, agent env/metadata/`post_run`, team ids and child source metadata), plus workflow/agent checkpoint round-trips, a trace-position round-trip, and an end-to-end trace-position save/resume through a fresh `WorkflowSession`.
- `config.rs` — `SessionConfig` defaults and environment resolution through public paths only, with a file-level `ENV_LOCK`/`EnvGuard` so process-global env mutations cannot leak between tests.
- `support/mod.rs` — the shared backend contract (`run_backend_contract(backend, Arc<dyn Checkpointer>, ListOrder::{OldestFirst, NewestFirst})`) used by `checkpoint.rs`, `checkpoint_sqlite.rs`, `checkpoint_redis.rs`, and `checkpoint_redis_tls.rs`.
- `support/resp_server.rs` — `FakeRedis`, the loopback RESP2 server the Redis tests run against (`start`, `start_tls`, `url`, `url_tls`, `url_with`, `commands`, `count`, `command`, `hang_up_after_each_reply`, plus `fixture(name)` for a path under the TLS fixture directory), with in-memory strings/sets, Redis's empty-set-disappears rule, and a TLS mode built from `server.der` + `server.key.der`.

## Constraints

- Offline and deterministic: no network, no real sleeps (only short `tokio::time::timeout` guards), `tempfile` for file/sqlite/disk-session/VCS roots, the in-process `FakeRedis` (clear text or TLS) instead of a Redis server, and static certificate fixtures instead of live TLS endpoints.
- Start each crate with `#![allow(clippy::unwrap_used, clippy::expect_used)]`; library code stays panic-free.
- Assert exact JSON field names where a wire/snapshot contract exists (`io_state`, `global_state`, `comp_state`, `trace_state`, `type`, `index`, `payload`, `session_id`, `checkpoint_id`, camelCase trace keys, snake_case `StreamSendOutcome`/`TraceSinkInstall` names).
- Behavior ports name the Python test they mirror in a comment.
- Every test compiles under `cargo test -p openjiuwen-session --offline` with default features; sqlite-only tests are feature-gated, the `otel` helper tests cover both feature states, and `--all-features` runs everything together.
- TLS tests never depend on a public endpoint: they load `tests/fixtures/tls/ca.pem` (or `.der`) and a deliberately wrong CA, and write malformed CA files into a `tempfile` directory.
- Pickle tests use hand-assembled byte streams (hex helpers) so the expected encodings stay readable and independent of the codec under test; no test shells out to `CPython`, so "readable by CPython" is only byte-pinned, never executed.
- No suite opens the SQLite database directly or asserts its `session_checkpoints` DDL, columns or payload column shape; the SQLite format is covered only through the `Checkpointer` contract (`checkpoint_sqlite.rs`).
- The SQLite suite is compiled out under default features and `make test`/`make test-all` never enable `-p openjiuwen-session --features sqlite`; it runs only via `--all-features` or an explicit `--features sqlite`.
- `tests/fixtures/tls/` holds the static certificates (`ca`, `other-ca`, `server` in PEM and DER, the server key) with a `README.md` that documents provenance and regeneration; nothing there is secret and the certificates only cover `localhost`/`127.0.0.1`.

## Routing Table

- `./support/` → shared test helpers (backend contract, `FakeRedis`); no `CONTEXT.md` of its own and no child directories
- `./fixtures/tls/` → static TLS certificates and the regeneration README used by the TLS tests
