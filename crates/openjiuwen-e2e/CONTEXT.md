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
Not dependencies here — do not import them without adding the workspace dep: `parking_lot` (use `std::sync::Mutex`), `tokio-util` (no `CancellationToken`), `serial_test`, any `openjiuwen-<layer>` crate.
Each `tests/*.rs` file is auto-discovered as its own test binary; there are no `[[test]]`/`required-features` entries, and a `tests/common/mod.rs` helper would need `#![allow(dead_code, clippy::unwrap_used, clippy::expect_used)]` because each binary uses only part of it.

## Constraints

Follows the shared contracts in `docs/ARCHITECTURE.md`: async-first tokio I/O, `openjiuwen_common::error` results, serde-serializable public types, no `unsafe`, and `tracing` logging.
Tests must run offline and deterministically: fake/scripted models, no network, temp dirs; timeouts only prove a deadline, never synchronize.
Use `Runner::new()` in tests, never `Runner::global()`; `Runner::reset_global()` exists for a test that must touch the singleton.
`EnvGuard`/`TempHomeGuard` are `!Send` and serialize on one process-global environment lock; `serial_test` is not a workspace dependency, so hold the guard for the test body instead of using `#[serial]`.
`openjiuwen-e2e` is not in the workspace `default-members`, so root `cargo test` skips it — use `cargo test -p openjiuwen-e2e` or `cargo test --workspace`.

## Routing Table

- `./tests/` → cross-crate integration scenarios (currently empty apart from `.gitkeep`)
- `./src/` → crate root `lib.rs` (`crate_name()` only)

## Reference Patterns

Copy the shape of sibling suites rather than inventing fixtures: `../openjiuwen-agent/tests/react_agent.rs` (`tool_round_feeds_the_tool_message_back_to_the_model` — scripted tool round) and `streaming.rs` (`collect_frames`/`content_of` — frame assertions).
`../openjiuwen-workflow/tests/linear.rs` (`linear_workflow` — `start → llm → end` with `${start.query}` references) and `checkpoint_resume.rs` (`a_resumed_run_does_not_re_execute_completed_nodes` — resume via `MemoryCheckpointer`).
`../openjiuwen-runner/tests/runner_lifecycle.rs` (`run_agent_through_the_facade_without_start` — local `Runner::new()` + `register_agent` + `run_agent`) is the facade template; workflow output is `WorkflowOutput.result` (`{"output": {...}}`) streamed end frames become a JSON array.
