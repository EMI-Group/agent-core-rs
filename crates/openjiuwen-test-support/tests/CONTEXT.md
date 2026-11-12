# tests — crate-level integration suites

## Intent

The four integration binaries that prove the fakes behave like the Python ones end to end.
`mock_server.rs` drives the real `OpenAiCompatibleClient` over a bound loopback socket; `scripted_model.rs` exercises the fake through `Arc<dyn Model>` and `ModelExt`; `fixtures.rs` proves the temp-directory and environment guards are drop-scoped; `session_fixtures.rs` proves the session helpers return real `Arc<dyn Session>` workflow/agent sessions with working state and checkpoints.
Unit tests stay beside the modules in `../src/`; only behavior that needs a real client, a real socket, or process-global environment mutation lives here.

## Constraints

- Every binary starts with `#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]`: these are integration-test targets linted with `-D warnings`, and panicking assertions are the point.
- Offline only: loopback sockets for the mock server, no other network, no external services, no wall-clock dependence.
- Deterministic and bounded: every wait and timeout is ≤2 s so a regression fails fast instead of hanging; waits use short real durations because tokio's `test-util` feature (`tokio::time::pause`) is not enabled.
- Each test owns its own server/script/workspace and must pass in any order and when the file is run in parallel with the other binaries.
- Tests that touch the process environment hold an `EnvGuard`/`TempHomeGuard` for their whole body, never call bare `set_var`/`remove_var`, and assert only inside the guard.
- Do not weaken an assertion to make a test pass; a real defect belongs in the owning `../src/` module.
- Ported tests cite the Python source they mirror in a comment (for example `tests/unit_tests/fixtures/mock_llm.py`).

## Routing Table

- `./mock_server.rs` → end-to-end client behavior: non-streaming/streaming text, tool calls, reasoning, error status mapping, fallback and deterministic ids, request recording, route alias, wrong-method 404, waiter wake-up, graceful shutdown, builder options
- `./scripted_model.rs` → `ScriptedModel` through the public trait: `Arc<dyn Model>` dispatch, `ModelExt::complete`, mixed turn order, failure propagation, call recording, fallback, `wait_for_calls`, builder overrides, `ScriptedError` kinds
- `./fixtures.rs` → `TempWorkspace` drop cleanup and I/O, `TempHomeGuard` + `OpenJiuwenHome::resolve`, `EnvGuard` restore/clear semantics, `ISOLATED_ENV_VARS` contents, cross-thread guard visibility
- `./session_fixtures.rs` → `workflow_session`/`agent_session` id and kind through `Arc<dyn Session>`, global/component state, `*_with` option pass-through, in-memory checkpointer save/resume and store isolation
