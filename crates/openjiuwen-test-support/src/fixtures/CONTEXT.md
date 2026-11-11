# fixtures — temp directories and environment isolation

## Intent

Pytest-equivalent fixtures for Rust tests: `tmp_path` becomes `TempWorkspace`, `monkeypatch.setenv("OPENJIUWEN_HOME", ...)` becomes `TempHomeGuard`, and `monkeypatch` env cleanup becomes `EnvGuard`.
All helpers are drop-scoped (temporary directory removed, environment restored) and never leak state between tests.

## API Surface

- `TempWorkspace` — `new`, `root`, `path`, `home`, `workspace`, `logs`, `create_dir`, `write`, `read`, `write_json`, `read_json`, `exists`, `files`.
- `EnvGuard` — `new`, `hold`, `set`, `remove`, `get`; `ISOLATED_ENV_VARS`.
- `TempHomeGuard` — `new`, `isolated`, `root`, `home`, `set_var`.
- `mod.rs` re-exports the three types and the constant, and nothing else.

## Constraints

- `TempWorkspace` never touches the process environment; it only owns a `tempfile::TempDir`.
- `EnvGuard` serializes on one process-global lock, reentrant on the owning thread so nested guards do not deadlock; the lock protects save/clear/restore and must be held for the guard's lifetime.
- Strictly one env-mutating guard family per test is the documented pattern; parallel tests serialize automatically, never spin or poll for the lock.
- Env guards are `!Send` (held lock) so they stay on the test thread; document that they must not be moved into spawned tasks.
- `OPENJIUWEN_HOME` is set to `<temp root>/home` and the directory is created eagerly.
- `ISOLATED_ENV_VARS` uses the constants from `openjiuwen_common::config::env` where available (never re-typed string literals for those names).
- `write_json`/`read_json` return `openjiuwen_common::error::Result` (`Error::sys_op` for I/O, `Error::validation` for serialization, both naming the path); plain file helpers return `io::Result`.
- Sessions are deliberately not constructed here: `openjiuwen-session` has no `Session` implementation yet; document the gap in the crate `CONTEXT.md`, not with a stub module.

## Test Strategy

Unit tests in `workspace.rs` (layout creation, nested relative paths, missing-file errors, JSON round-trip) and in `env.rs` (set/restore on drop, clear on construction, nested guards on one thread, parallel guards in two threads staying isolated).
`../../tests/fixtures.rs` verifies drop cleanup and `OPENJIUWEN_HOME` resolution through `OpenJiuwenHome::resolve`.
