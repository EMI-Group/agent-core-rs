# verification

## Matrix rows
| Python path | Rust crate | Status | Notes |
|---|---|---|---|
| repository `tests/` (cross-package) | `openjiuwen-e2e` | Implemented | 35 cross-crate scenario tests over 8 binaries (`kernel_flow` 6, `cli_binary` 6, `provider_native` 5, `deep_agent` 4, `a2a_roundtrip` 4, `self_improvement` 4, `team_flow` 3, `symphony_plan` 3); 30 run with default features, 34 with `--features a2a` because `a2a_roundtrip` is gated, and 1 is `#[ignore]`d. Every scenario runs offline against `ScriptedModel`, a loopback `MockLlmServer`, or loopback sockets, with a 10 s per-await `BOUND`. |
| repository `examples/` | `openjiuwen-examples` | Implemented | 12 explicit `[[example]]` binaries (`autoexamples = false`), numbered `01_hello_agent` … `12_self_improvement`; example 11 declares `required-features = ["a2a"]` and `mcp` is an empty marker feature. There is no `tests/` directory: the only tests are 15 library unit tests in `src/support/` (`mode` 10, `banner` 4, `logging` 1), and the examples are validated by compilation plus manual smoke runs. Every example runs offline by default and switches to a real provider only when `OPENJIUWEN_LIVE` plus credentials are present. |
| — (Rust-only) | `openjiuwen-test-support` | Implemented | Dev-only (`publish = false`, never a library dependency): loopback axum `MockLlmServer` speaking OpenAI-compatible JSON and SSE, `ScriptedModel`/`MockCompletion`, `TempWorkspace`/`EnvGuard`/`TempHomeGuard` fixtures, workflow/agent session and in-memory checkpointer fixtures, JSON assertion helpers with diff paths, `eventually` polling helpers, and a `wiremock` re-export. 149 tests: 100 unit tests beside the modules plus 49 in 4 integration binaries (`mock_server` 21, `scripted_model` 13, `fixtures` 9, `session_fixtures` 6). |

## Verification commands
- `cargo test --workspace` (Makefile `test`) — the default offline gate over all 30 crates, including `openjiuwen-test-support`, `openjiuwen-e2e` with default features, and the `openjiuwen-examples` library unit tests.
- `cargo test --workspace --offline` — the same gate in the form `docs/PARITY.md` §Verification reports.
- `cargo test` (bare) — uses the 27 `default-members` and therefore skips `openjiuwen-e2e`, `openjiuwen-examples` and `openjiuwen-test-support`.
- `cargo test -p openjiuwen-e2e` — 30 passing plus 1 ignored with default features; `a2a_roundtrip` is `#![cfg(feature = "a2a")]` and compiles empty.
- `cargo test -p openjiuwen-e2e --features a2a` — 34 passing plus 1 ignored; this is the `openjiuwen-e2e` line of `make test-all`.
- `cargo test -p openjiuwen-test-support` — 149 tests, offline, loopback-only, with waits bounded at ≤2 s.
- `cargo test -p openjiuwen-examples` — the 15 `src/support/` unit tests (run-mode detection and banner formatting); no integration binaries exist.
- `cargo build -p openjiuwen-examples --examples` and `cargo build -p openjiuwen-examples --examples --features a2a,mcp` — compile all 12 examples (the second form is the only one that builds example 11).
- `cargo test -p openjiuwen-harness --no-default-features`, `cargo test -p openjiuwen-agent-teams --no-default-features`, `cargo test -p openjiuwen-harness-providers --features vendor`, `cargo test -p openjiuwen-symphony --features sqlite`, `cargo test -p openjiuwen-sysop --features sandbox`, `cargo test -p openjiuwen-extensions --features a2a,otel` — the six feature suites of `make test-all`.
- `make verify` — runs `fmt-check`, `clippy`, `test`, `test-all` and both example builds in that order.
- `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo doc --workspace --no-deps` — style, lint and rustdoc gates.
- `cargo test -p openjiuwen-sysop --features sandbox -- --ignored` — runs the single network-dependent test (live sandbox gateway probe).

## Ignored tests
- 3 total across the workspace; each names its prerequisite, and the prerequisites group as: (a) network — `openjiuwen-sysop/src/sandbox/http.rs::probe_against_live_gateway` needs a live sandbox gateway and is reachable only with `--features sandbox -- --ignored`; (b) build graph — `openjiuwen-e2e/tests/cli_binary.rs::binary_version_over_the_process_boundary` cannot use `CARGO_BIN_EXE_openjiuwen` because that variable exists only for the package owning the binary (the process-level smoke lives in `openjiuwen-cli/tests/cli_smoke.rs`); (c) harness role — `openjiuwen-runner/tests/spawn_subprocess.rs::fake_child` is not a standalone test but the stdio protocol child that the sibling subprocess tests re-execute through a libtest filter, selected by `FAKE_CHILD_MODE`.
- No `#[ignore]`d test needs credentials or a wall clock, and no test is gated on the `live-tests` feature: 7 manifests declare `live-tests`, and no `.rs` file under `crates/` references it.

## Structural facts
- 8 e2e test binaries: `a2a_roundtrip`, `cli_binary`, `deep_agent`, `kernel_flow`, `provider_native`, `self_improvement`, `symphony_plan`, `team_flow`, plus the helper `tests/support/mod.rs` (`BOUND`, `within`, `collect_bounded`); no `[[test]]` entries in the manifest.
- 12 examples: `01_hello_agent`, `02_react_with_tools`, `03_sessions_and_resume`, `04_workflow_pipeline`, `05_context_and_memory`, `06_retrieval`, `07_mcp_tool`, `08_deep_agent`, `09_harness_provider`, `10_agent_team`, `11_a2a_server_client`, `12_self_improvement`, plus the example-only helper module `examples/common/mod.rs`; the shared library helpers live in `crates/openjiuwen-examples/src/support/` (`setup`, `mode`, `logging`, `banner`).
- 30 workspace crates; `default-members` lists 27 of them, omitting `openjiuwen-e2e`, `openjiuwen-examples` and `openjiuwen-test-support`.
- 290 top-level `crates/*/tests/*.rs` binaries exist (282 outside `openjiuwen-e2e`), and 384 `.rs` files sit under `crates/*/tests/` once the helper modules in the `tests/support/` and `tests/common/` directories are counted.
- `openjiuwen-test-support` has 6 public modules (`mock_llm`, `fixtures`, `responses`, `scripted_model`, `assert_json`, `eventually`), all re-exported at the crate root alongside `CRATE_NAME`, `CRATE_VERSION`, `TEST_API_KEY` and `pub use wiremock`.

## Corrections to other PARITY.md sections
- §Verification: current text says "8164 passed, 0 failed, 5 ignored"; should say "3 ignored", because exactly three `#[ignore]` attributes exist in the tree (sysop live-gateway probe, e2e process-boundary CLI test, runner `fake_child`).
- §Verification: current text says "A bare `cargo test` uses the workspace `default-members` and skips `openjiuwen-e2e` and `openjiuwen-examples`"; should say it also skips `openjiuwen-test-support` — the three omitted members out of 30.
- §Verification: current text says the feature suites include "`cargo test -p openjiuwen-extensions --features a2a`, `--features otel`, `--features sqlite`"; should say `make test-all` runs `cargo test -p openjiuwen-extensions --features a2a,otel` and no documented gate runs the extensions `sqlite` suite.
- §Package Matrix (`repository tests/` row): current text says "35 cross-crate scenario tests over 8 binaries (1 documented `#[ignore]`); `a2a_roundtrip` is feature-gated"; should add the per-binary split and the 30-default / 34-with-`a2a` pass counts.
- §Package Matrix (`— (Rust-only)` `openjiuwen-test-support` row): current text says "Dev-only `ScriptedModel`, loopback `MockLlmServer`, workspace/session/checkpointer fixtures"; should also name the JSON assertion helpers, the `eventually` helpers, the `wiremock` re-export, and the `publish = false` / 149-test figure.
- crates/CONTEXT.md §Test Conventions: current text says "~176 `tests/*.rs` files today"; should say "290 top-level `tests/*.rs` binaries (384 `.rs` files under `tests/` including helpers)".
- crates/CONTEXT.md §Test Conventions: current text says "Network/vendor tests are opt-in behind the `live-tests` feature and `OPENJIUWEN_LIVE`"; should say no test is gated on `live-tests` today — the one network test is `#[ignore]`d behind `--features sandbox`, and `OPENJIUWEN_LIVE` only switches the examples from offline to live.

## ARCHITECTURE.md corrections
- §6 Testing infrastructure: current text says `openjiuwen-test-support` provides "JSON snapshots for Cards"; should say it provides JSON assertion helpers (`assert_json_eq`, `assert_json_subset`, `assert_json_pointer_eq`, `assert_json_roundtrip`, `first_difference`), `eventually` polling, and a `wiremock` re-export, with no card snapshot store.
- §6 Testing infrastructure: current text lists the `openjiuwen-e2e` scenarios as "ReAct loop with tool use, workflow graph execution + streaming, session checkpoint/resume, memory recall, MCP stdio round-trip, and CLI smoke tests"; should say the 8 binaries cover kernel flow (ReAct tool round, checkpoint round trip, workflow `start → llm → end`, local `Runner`), DeepAgent rounds and permissions, native provider lifecycle, agent teams, A2A round trip (feature `a2a`), symphony planning/execution, RSI + trainer, and the harness CLI layer — with no long-term-memory recall, no MCP stdio round trip, and no process-level CLI smoke (that lives in `openjiuwen-cli`; the e2e variant is ignored).
- §6 Testing infrastructure: current text says "any test needing network is `#[ignore]`d and documented"; should say only 1 of the 3 ignored tests needs the network, the other two being a re-executed stdio child and a Cargo build-graph limitation.
- §5 Feature flags: current text presents `live-tests` as an opt-in network/vendor test gate; should say 7 manifests declare `live-tests` and no test in the workspace is gated on it.
