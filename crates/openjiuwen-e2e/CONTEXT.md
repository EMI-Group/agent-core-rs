# openjiuwen-e2e

## Intent

The cross-crate scenario suite of the Rust rewrite: the only place where several public crates are driven together through their real seams.
It answers "does the composed system work?" — kernel flow, DeepAgent harness, provider SPI, agent teams, A2A, symphony, self-improvement, and the CLI — while every crate's own `tests/` answer "does this module work?".
Every scenario is offline, deterministic and bounded: scripted models or a loopback `MockLlmServer`, temp directories, loopback sockets only, no credentials, no wall-clock ordering.
The crate is dev-only (`publish = false`) and never a dependency of a library crate.

## API Surface

- `src/lib.rs` — marker module (`crate_name()`); the crate has no runtime API.
- `tests/support/mod.rs` — the cross-scenario helper contract: `BOUND` (10 s per-await deadline), `within(what, future)`, `collect_bounded(stream, max)`. Scenario-specific fixtures stay in `openjiuwen-test-support` or beside the scenario.
- Test binaries (each `tests/*.rs` is its own binary; no `[[test]]` entries):
  `kernel_flow`, `deep_agent`, `provider_native`, `team_flow`, `a2a_roundtrip`, `symphony_plan`, `self_improvement`, `cli_binary`.
- Features: `default = []`; `a2a` (enables `openjiuwen/a2a` + `openjiuwen-extensions/a2a`, gates `a2a_roundtrip.rs`); `otel` (declared for parity, no scenario).
- Dependencies are declared under `[dependencies]` so the `a2a`/`otel` features can enable features on the same crates: the umbrella `openjiuwen` (module aliases `core`, `harness`, `harness_protocol`, `harness_providers`, `agent_teams`, `extensions`, `symphony`, `rsi`, `agent_evolving`), the concrete crates for their module spellings, `openjiuwen-test-support`, `tokio`, `tokio-util`, `futures`, `async-trait`, `anyhow`, `reqwest`, `serde_json`, `tempfile`.

## Scenario Matrix

| Binary | Scenario | Crates / seams exercised |
|---|---|---|
| `kernel_flow.rs` | ReAct tool round against `ScriptedModel`; tool message fed back to the model | `openjiuwen-test-support` (`ScriptedModel`), `openjiuwen-agent` (`ReActAgent`, `FunctionTool`), `openjiuwen-foundation` (`Model`) |
| `kernel_flow.rs` | Session state + checkpoint round trip through a shared `MemoryCheckpointer` | `openjiuwen-session` (`Session`, `Checkpointer`), test-support session fixtures |
| `kernel_flow.rs` | Workflow `start → llm → end` with `${node.field}` references | `openjiuwen-workflow` (builder, `StartComponent`/`LlmComponent`/`EndComponent`), `openjiuwen-graph` |
| `kernel_flow.rs` | Local `Runner` runs an agent and a workflow, plus agent streaming | `openjiuwen-runner` (`Runner::new`, `ResourceMgr`, `RunOptions`, `run_agent_streaming`) |
| `deep_agent.rs` | DeepAgent tool round over a temp workspace (read → write → answer) | `openjiuwen-harness` (`DeepAgentBuilder`, `ToolRegistry`, `BuiltinTool`), `openjiuwen-agent::Agent` |
| `deep_agent.rs` | Same round over HTTP through `MockLlmServer` | `openjiuwen-foundation` `init_model`, test-support `MockLlmServer` |
| `deep_agent.rs` | Permission denial (`Deny` rule) surfaces `[PERMISSION_DENIED]` to the model, not an `Err` | `openjiuwen-harness` `permission` (`PermissionsSection`, `PermissionLevel`) |
| `deep_agent.rs` | `Ask` rule interrupts; resume with `interrupt_answers` on the same session | `openjiuwen-harness` `PermissionInterruptRail`, `openjiuwen-agent` `ResultType::Interrupt`, `openjiuwen-session` interactions |
| `provider_native.rs` | Provider registry answers `available_providers`/`describe_provider`; unknown → `NotFound`, deferred vendor → `Unsupported` | `openjiuwen-harness-providers` factory + registry |
| `provider_native.rs` | Native harness `start` → `send` → event drain (`TurnLifecycle`, `Output`) → `stop` | `openjiuwen-harness-protocol` (`HarnessProtocol`, `HarnessEventCursor`), `openjiuwen-harness` (`NativeHarness`) |
| `provider_native.rs` | `create_harness("native", json)` over loopback `MockLlmServer` | `openjiuwen-harness-providers::create_harness` factory path |
| `provider_native.rs` | `HarnessIOAdapter` projects harness events onto `AgentChunk`s (output, tool call/result, turn lifecycle) | `openjiuwen-harness-providers` `io_adapter`, fake `HarnessProtocol` |
| `team_flow.rs` | `TeamAgentSpec` → `build_with` a scripted `MemberHarnessFactory` | `openjiuwen-agent-teams` (`TeamAgentSpec`, `SpecBuild`, `TeamBuildDeps`) |
| `team_flow.rs` | Leader/teammate dispatch; exactly one `team.completed` marker, member frames tagged | `openjiuwen-agent-teams` runtime + stream tagging |
| `team_flow.rs` | Task board create → claim → complete settles the board | `openjiuwen-agent-teams` `TaskBoard` |
| `a2a_roundtrip.rs` (feature `a2a`) | Agent card fetch, `A2aClient::invoke`, `stream`, `cancel_task` against a loopback `A2aServer` on an ephemeral port | `openjiuwen-extensions::a2a` (`A2aServer`, `A2aClient`), `openjiuwen-harness-protocol` executors |
| `symphony_plan.rs` | Register capabilities → `plan` → `compile` to a `CompiledGraph` → execute with a recording runner | `openjiuwen-symphony` (`SymphonyRuntime`, `CapabilityRegistry`, `PlanExecutor`), `openjiuwen-graph` |
| `symphony_plan.rs` | Graph snapshot published to a file store and reloaded by a second runtime | `openjiuwen-symphony` artifact/asset stores |
| `self_improvement.rs` | RSI orchestrator with a fake `HarnessRunner` converges in two iterations | `openjiuwen-rsi` (`AutoHarnessOrchestrator`, `HarnessRunner`, `SessionSummary`) |
| `self_improvement.rs` | RSI checkpoint resume continues at the persisted iteration | `openjiuwen-rsi` `CheckpointStore`/`SessionState` |
| `self_improvement.rs` | Trainer improves a fake `TrainableTarget` and resumes from a checkpoint | `openjiuwen-agent-evolving` (`Trainer`, `Optimizable`/`TrainableTarget`, checkpoints) |
| `cli_binary.rs` | `run_with_model` one-shot with injected model: `text`, `json`, `stream-json` output | `openjiuwen-harness` `cli` (`run_with_model`, `CliIo`, `OutputFormat`) |
| `cli_binary.rs` | Usage error and `--help`/`--version` resolved by clap (`try_parse_from`), exit code 2/0 semantics | `openjiuwen-harness` `cli::Cli` grammar |

## Constraints

- Offline and deterministic: `ScriptedModel`/`MockLlmServer`/loopback sockets only; no credentials, no `#[ignore]`d network tests, no `Runner::global()` (always a local `Runner::new()`), no `MemoryCheckpointer::shared()`.
- No sleeps for ordering: a scenario waits on the event, frame, or state transition it asserts; timeouts exist only so a hang fails fast ([`support::BOUND`](tests/support/mod.rs)).
- Never assert non-deterministic fields: timestamps, `duration_ms`, generated ids other than the documented counters (`turn-1`, `message-1`, `call_<tool>`), uuid plan/run ids.
- Every test file starts with `#![allow(clippy::unwrap_used, clippy::expect_used)]` (workspace lints are warnings and the gate is `-D warnings`); shared helper modules additionally allow `dead_code`, `unreachable_pub`.
- Each scenario owns its temp directory (`tempfile::TempDir` / `TempWorkspace`) and keeps it alive for the test body; nothing writes outside it.
- A scenario that needs a sibling API change is `#[ignore = "<reason>"]`d with the gap recorded under **Known Gaps**, never fixed by editing the sibling crate.
- `openjiuwen-e2e` is not in the workspace `default-members`: use `cargo test -p openjiuwen-e2e` (and `--features a2a`).

## Routing Table

- `./tests/kernel_flow.rs` → model → tool → session → workflow → runner flows (kernel crates)
- `./tests/deep_agent.rs` → DeepAgent task loop, workspace tools, permissions, interrupt/resume
- `./tests/provider_native.rs` → provider-neutral harness SPI: factory, native harness lifecycle, IO adapter
- `./tests/team_flow.rs` → agent teams: spec build, leader/teammate runtime, task board
- `./tests/a2a_roundtrip.rs` → A2A client/server round trip (feature `a2a`)
- `./tests/symphony_plan.rs` → symphony capability registry, planning, graph execution, artifact reload
- `./tests/self_improvement.rs` → RSI orchestrator and agent-evolving trainer with fake runners
- `./tests/cli_binary.rs` → harness CLI layer (one-shot output formats, grammar, version/help)
- `./tests/support/mod.rs` → cross-scenario deadline and stream helpers (`BOUND`, `within`, `collect_bounded`)
- `./Cargo.toml` → manifest, test-only dependency set, `a2a`/`otel` features

## Test Strategy

- One binary per scenario area so a failure names the seam and the suite parallelises safely (each binary has its own temp dirs, its own model script, its own loopback port).
- Assert on recorded evidence, not timing: `ScriptedModel::calls()`/`call_count()`, harness event sequences, collected frames, store contents, file contents.
- Fake only what a scenario cannot own: harness protocols, member factories, symphony capability runners, RSI harness runners, evolving targets. The fakes are copied into the scenario that needs them (they are test-binary-local), keeping each binary's contract explicit.
- Every loopback scenario binds an ephemeral port (`127.0.0.1:0`) and reads the bound address back; no fixed ports, so suites can run in parallel.
- Gates: `cargo fmt -p openjiuwen-e2e`, `cargo test -p openjiuwen-e2e`, `cargo test -p openjiuwen-e2e --features a2a`, `cargo clippy -p openjiuwen-e2e --all-targets --no-deps -- -D warnings`.

## Design Decisions

- Shared helpers are limited to the timing/stream contract; fixtures live in `openjiuwen-test-support` (models, workspaces, sessions, checkpoints) so library and scenario suites use the same fakes.
- Dependencies live under `[dependencies]` rather than `[dev-dependencies]`: the `a2a`/`otel` features must enable features of the same crates, and a feature may not reliably forward to a dev-dependency.
- `axum`/`tower` are intentionally absent: the A2A scenario uses a real loopback socket (`A2aServer::serve_in_background` + `A2aClient`) instead of an in-process `tower::ServiceExt::oneshot` harness, so the transport, the JSON-RPC envelope and the SSE framing are all exercised.
- `reqwest` is a direct dependency only for the agent-card endpoint (the client itself lives in `openjiuwen-extensions`).
- `wiremock` is reached through `openjiuwen_test_support::wiremock` when a scenario needs a non-LLM HTTP fake; no separate dependency.

## Known Gaps

- `cli_binary.rs` cannot use `env!("CARGO_BIN_EXE_openjiuwen")`: Cargo defines that variable only for tests of the package that owns the binary. The process-level smoke test lives in `crates/openjiuwen-cli/tests/cli_smoke.rs`; the e2e suite asserts the same verdicts through the clap grammar and `run_with_model`, and the binary-spawn case is `#[ignore]`d with that reason.
- Vendor harness providers (`claudecode`, `codex`, `dsh`) and every live-provider path (`live-tests`) need credentials/network and are out of scope; `provider_native.rs` asserts only that they report `Error::Unsupported`/`NotFound`.
- Deferred ML/LLM-backed areas (symphony `deferred::*`, `evaluation::LlmJudgedMetric`; agent-evolving RL/judge stubs; RSI `deferred::*`) return `Error::Unsupported` and are not exercised.
- `otel` has no scenario: the seam is a no-op without the feature and its live path needs a collector; behaviour is covered by `openjiuwen-extensions` unit tests. The feature is declared so the gate can be run once a collector-free scenario exists.

## Reference Patterns

Copy the shape of the sibling suites instead of inventing fixtures: `../openjiuwen-agent/tests/react_agent.rs` (tool round), `../openjiuwen-workflow/tests/linear.rs` + `references.rs` (`${a.b}` flows), `../openjiuwen-runner/tests/run_agent.rs` (facade), `../openjiuwen-harness/tests/deep_agent_loop.rs` + `permissions.rs` (harness rounds), `../openjiuwen-harness-providers/tests/io_adapter.rs` (event → chunk projection), `../openjiuwen-agent-teams/tests/team_run.rs` + `task_flow.rs`, `../openjiuwen-symphony/tests/{executor,runtime}.rs`, `../openjiuwen-rsi/tests/{orchestrator,checkpoint}.rs`, `../openjiuwen-agent-evolving/tests/trainer.rs`, `../openjiuwen-extensions/tests/a2a_*.rs`, `../openjiuwen-harness/tests/cli.rs`, `../openjiuwen-cli/tests/cli_smoke.rs`.
