# openjiuwen-e2e

## Intent

Cross-crate integration scenarios.

## API Surface

`openjiuwen-e2e` exposes the `openjiuwen_e2e` module root (`crate_name()` only); real content lives in integration-test binaries under `tests/`.
`tests/` currently holds only `.gitkeep` — the crate has no test targets yet.
Every dependency is declared under `[dependencies]`; there is no `[dev-dependencies]` section and no feature beyond `default = []`.
Available to test binaries: `openjiuwen` (umbrella, default features incl. `native`), `openjiuwen-test-support`, `tokio` (workspace features: rt, rt-multi-thread, macros, sync, time, fs, io-util, process, net, signal), `serde_json`, `anyhow`, `tempfile`, `futures`, `async-trait`.
`openjiuwen-runner` is **not** a direct dependency: reach it as `openjiuwen::core::runner` (kernel umbrella re-export) or add `openjiuwen-runner.workspace = true` for the `openjiuwen_runner::` spelling.
`openjiuwen-test-support` supplies the offline pieces: `MockLlmServer`, `ScriptedModel`, `ScriptedResponse`, `TempWorkspace`, `TempHomeGuard`, `EnvGuard`, `agent_session*`/`workflow_session*`, `memory_checkpointer`, `eventually*`, `assert_json_*`.
Shared helper modules (`tests/common/mod.rs`, declared as `mod common;`) exist in sibling crates (`openjiuwen-runner`, `openjiuwen-agent`, `openjiuwen-graph`, `openjiuwen-workflow`, `openjiuwen-multiagent`) but are private to their own test binaries and cannot be imported here.

## Constraints

Follows the shared contracts in `docs/ARCHITECTURE.md`: async-first tokio I/O, `openjiuwen_common::error` results, serde-serializable public types, no `unsafe`, and `tracing` logging.
Tests must run offline and deterministically: fake/scripted models, no network, temp dirs; timeouts only prove a deadline, never synchronize.
Use `Runner::new()` in tests, never `Runner::global()`; `Runner::reset_global()` exists for a test that must touch the singleton.
`EnvGuard`/`TempHomeGuard` are `!Send` and serialize on one process-global environment lock; `serial_test` is not a workspace dependency, so hold the guard for the test body instead of using `#[serial]`.
`openjiuwen-e2e` is not in the workspace `default-members`, so root `cargo test` skips it — use `cargo test -p openjiuwen-e2e` or `cargo test --workspace`.

## Routing Table

- `./tests/` → cross-crate integration scenarios (currently empty apart from `.gitkeep`)
- `./src/` → crate root `lib.rs` (`crate_name()` only)
