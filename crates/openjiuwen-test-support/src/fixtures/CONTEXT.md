# fixtures — temp directories, environment isolation, and ready-made sessions

## Intent

Pytest-equivalent fixtures for Rust tests: `tmp_path` becomes `TempWorkspace`, `monkeypatch.setenv("OPENJIUWEN_HOME", ...)` becomes `TempHomeGuard`, `monkeypatch` env cleanup becomes `EnvGuard`, and Python's `create_workflow_session`/`create_agent_session` test helpers become the `session.rs` session and checkpointer fixtures.
The filesystem and environment helpers are drop-scoped (temporary directory removed, environment restored) and never leak state between tests.
The session fixtures build on the implemented `openjiuwen-session` factories, so a test gets a real `Arc<dyn Session>` without repeating an options struct.

## API Surface

- `TempWorkspace` — `new`, `root`, `path`, `home`, `workspace`, `logs`, `create_dir`, `write`, `read`, `write_json`, `read_json`, `exists`, `files`.
- `EnvGuard` — `new`, `hold`, `set`, `remove`, `get`; `ISOLATED_ENV_VARS`.
- `EnvGuard` deliberately has no `Debug` because it holds saved environment values that may be secrets; `TempHomeGuard` does implement it.
- `TempHomeGuard` — `new`, `isolated`, `root`, `home`, `set_var`; it keeps its own resolved `home` path so `home()` can return a borrow instead of recomputing the join.
- Session fixtures (`session.rs`) — `workflow_session`, `workflow_session_with`, `agent_session`, `agent_session_with`, `memory_checkpointer`, `workflow_session_with_checkpointer`, `agent_session_with_checkpointer`; the plain helpers build a default `WorkflowSessionOptions`/`AgentSessionOptions` around the given id and return `Arc<dyn openjiuwen_session::Session>`.
- `memory_checkpointer` returns a fresh `Arc<dyn Checkpointer>` wrapping `MemoryCheckpointer::new`, so each call is an isolated store.
- `mod.rs` re-exports the three guard types, the session helpers, and the constant, and nothing else.

## Constraints

- `TempWorkspace` never touches the process environment; it only owns a `tempfile::TempDir`.
- `TempWorkspace::new` propagates a failure to create the initial `home`/`workspace`/`logs` layout, because that layout is part of the contract.
- The `home()`/`workspace()`/`logs()` getters ignore a creation failure because they cannot fail; a caller that needs the directory to exist uses `create_dir` and handles the error.
- `files()` skips unreadable directory entries instead of reporting them, since it is a diagnostics helper for assertions.
- `EnvGuard` serializes on one process-global lock, reentrant on the owning thread so nested guards do not deadlock; the lock protects save/clear/restore and must be held for the guard's lifetime.
- `set`/`remove` on an `EnvGuard` are only restored on drop for the names the guard was constructed with; names mutated without being listed keep their final value, so a test must list every variable it touches.
- Strictly one env-mutating guard family per test is the documented pattern; parallel tests serialize automatically, never spin or poll for the lock.
- Env guards are `!Send` (held lock) so they stay on the test thread; document that they must not be moved into spawned tasks.
- `TempHomeGuard` is `!Send` as well because it owns an `EnvGuard`, so a test holding one cannot use `#[tokio::test(flavor = "multi_thread")]` and must not hand it to `tokio::spawn`; the default current-thread `#[tokio::test]` is fine.
- `ENV_LOCK` serializes only guards inside one test binary: each `cargo test` target is a separate process, and the hand-rolled `EnvGuard` copies in other crates (e.g. `openjiuwen-common`, `openjiuwen-session` tests) use their own locks, so mutual exclusion is never guaranteed across binaries or foreign guards.
- `EnvGuard::drop` restores the saved names in construction order (`saved.drain(..)`), although the `saved` field comment in `env.rs` says "reverse order"; only the set of restored names matters today.
- `OPENJIUWEN_HOME` is set to `<temp root>/home` and the directory is created eagerly.
- `ISOLATED_ENV_VARS` uses the constants from `openjiuwen_common::config::env` where available (never re-typed string literals for those names).
- `write_json`/`read_json` return `openjiuwen_common::error::Result` (`Error::sys_op` for I/O, `Error::validation` for serialization, both naming the path); plain file helpers return `io::Result`.
- A session takes its checkpointer at construction only: `WorkflowSessionOptions::checkpointer` / `AgentSessionOptions::checkpointer` are the attach point and `Session::checkpointer` is a read-only getter, so there is no post-construction `attach`.
- To resume, callers build both sessions through `*_with` with the same session id and the same `Arc<dyn Checkpointer>`; the convenience `*_with_checkpointer` helpers create a fresh isolated `MemoryCheckpointer` per call.
- Session fixtures never touch the process environment and never use the process-wide `MemoryCheckpointer::shared` store, so parallel tests stay isolated.

## Test Strategy

Unit tests in `workspace.rs` (layout creation, nested relative paths, missing-file errors, JSON round-trip) and in `env.rs` (set/restore on drop, clear on construction, nested guards on one thread, and cross-thread serialization).
The cross-thread `env.rs` test proves mutual exclusion with a channel handshake rather than a bare sleep: the child signals that it is about to construct its guard, the parent asserts no acquisition arrives while its own guard is alive, and only after the parent drops the guard does the child report acquiring it.
`../../tests/fixtures.rs` verifies drop cleanup and `OPENJIUWEN_HOME` resolution through `OpenJiuwenHome::resolve`.
`../../tests/session_fixtures.rs` verifies the session helpers return the requested id/kind through `Arc<dyn Session>`, that state is readable and writable, that `*_with` passes options through, and that the in-memory checkpointer saves and resumes checkpoints (including the per-call store isolation).
