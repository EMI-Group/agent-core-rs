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
- Dependencies are declared under `[dependencies]` so the `a2a`/`otel` features can enable features on the same crates: the umbrella `openjiuwen` (module aliases `core`, `harness`, `harness_protocol`, `harness_providers`, `agent_teams`, `extensions`, `symphony`, `rsi`, `agent_evolving`), the concrete harness-layer crates `openjiuwen-harness`, `openjiuwen-harness-protocol`, `openjiuwen-harness-providers`, `openjiuwen-agent-teams`, `openjiuwen-extensions`, `openjiuwen-symphony`, `openjiuwen-rsi`, `openjiuwen-agent-evolving`, plus `openjiuwen-test-support`, `tokio`, `tokio-util`, `futures`, `async-trait`, `anyhow`, `reqwest`, `serde_json`, `tempfile`.
- No kernel crate (`openjiuwen-session`, `-workflow`, `-graph`, `-agent`, `-foundation`, `-runner`, `-common`) is a direct dependency: kernel types must be imported through the umbrella as `openjiuwen::core::<module>::…` (e.g. `openjiuwen::core::workflow::{Workflow, WorkflowCard, …}`, `openjiuwen::core::session::AgentSessionOptions`, `openjiuwen::core::runner::{Runner, RunOptions}`), which requires no feature.
- Model/session fixtures come from `openjiuwen_test_support`, never from `tests/support/mod.rs`: `agent_session`, `agent_session_with`, `agent_session_with_checkpointer`, `workflow_session`, `workflow_session_with`, `workflow_session_with_checkpointer`, `memory_checkpointer`, `ScriptedModel`, `MockCompletion`, `MockLlmServer`, `ScriptedResponse`, `TEST_API_KEY`.
- `a2a_roundtrip.rs` needs `#![allow(missing_docs)]` **before** its `#![cfg(feature = "a2a")]`: with the feature off the whole crate body (including its `//!` docs) is stripped, so a later `allow` is stripped too and the denied `missing_docs` lint fires on the empty crate root.

## Scenario Matrix

40 tests across 9 binaries: 39 pass with `--features a2a` (35 with default features, since `a2a_roundtrip` is gated), 1 documented `#[ignore]`.

| Binary | Scenario | Crates / seams exercised |
|---|---|---|
| `kernel_flow.rs` (6) | ReAct tool round against `ScriptedModel`; tool message fed back to the model | `openjiuwen-test-support` (`ScriptedModel`), `openjiuwen-agent` (`ReActAgent`, `FunctionTool`), `openjiuwen-foundation` (`Model`) |
| `kernel_flow.rs` | Session global + component state round-tripped through a shared `MemoryCheckpointer` | `openjiuwen-session` (`Session`, `Checkpointer`), test-support session fixtures |
| `kernel_flow.rs` | Workflow `start → llm → end` with `${start.query}` / `${llm.result}` references | `openjiuwen-workflow` (builder, `StartComponent`/`LlmComponent`/`EndComponent`), `openjiuwen-graph` |
| `kernel_flow.rs` | Local `Runner` runs an agent and a workflow; agent streaming frame contract | `openjiuwen-runner` (`Runner::new`, `ResourceMgr`, `RunOptions`, `run_agent_streaming`) |
| `deep_agent.rs` (4) | DeepAgent tool round in a temp workspace (read seeded file → write new file → answer) | `openjiuwen-harness` (`DeepAgentBuilder`, `ToolRegistry`, `BuiltinTool`), `openjiuwen-agent::Agent` |
| `deep_agent.rs` | Same round over HTTP through `MockLlmServer` (also needs `with_model_name`/`with_model_config`, see Known Behaviours) | `openjiuwen-foundation` `init_model`, test-support `MockLlmServer` |
| `deep_agent.rs` | Permission `Deny` surfaces `[PERMISSION_DENIED]` to the model; the loop continues with `ResultType::Answer` | `openjiuwen-harness` `permission` (`PermissionsSection`, `PermissionLevel`) |
| `deep_agent.rs` | `Ask` rule interrupts; resume with `interrupt_answers` on the same session | `PermissionInterruptRail`, `openjiuwen-agent` `ResultType::Interrupt`, session interactions |
| `provider_native.rs` (5) | Provider registry lists/describes `native`; unknown → `NotFound`, vendor → `Unsupported` | `openjiuwen-harness-providers` factory + registry |
| `provider_native.rs` | Native harness `start` → `send` → event drain (`StateChanged`/`TurnLifecycle`/`Output`) → `stop` | `openjiuwen-harness-protocol` (`HarnessProtocol`, `HarnessEventCursor`), `openjiuwen-harness` (`NativeHarness`) |
| `provider_native.rs` | `create_harness("native", json)` runs a real turn over loopback `MockLlmServer` | `openjiuwen-harness-providers::create_harness` factory path |
| `provider_native.rs` | `HarnessIOAdapter` projects events onto `AgentChunk`s (output, tool call/result, optional turn lifecycle) | `openjiuwen-harness-providers` `io_adapter`, hand-built `HarnessProtocol` double |
| `team_flow.rs` (3) | `TeamAgentSpec` → `build_with` a scripted `MemberHarnessFactory`; roster and member cards | `openjiuwen-agent-teams` (`TeamAgentSpec`, `SpecBuild`, `TeamBuildDeps`) |
| `team_flow.rs` | Leader/teammate dispatch; exactly one `team.completed` marker, member frames tagged | `openjiuwen-agent-teams` runtime + stream tagging |
| `team_flow.rs` | Task board create → claim → complete settles the board | `openjiuwen-agent-teams` `TaskBoard` |
| `a2a_roundtrip.rs` (4, feature `a2a`) | Card endpoint over HTTP + `A2aClient::invoke` against a loopback `A2aServer` on an ephemeral port | `openjiuwen-extensions::a2a` (`A2aServer`, `A2aClient`), `reqwest` for the card |
| `a2a_roundtrip.rs` | Streaming send yields the deterministic frame sequence and one terminal frame | A2A SSE framing |
| `a2a_roundtrip.rs` | Paused turn (`INPUT_REQUIRED`) → `cancel_task` → `CANCELED` | A2A task store + cancel verb |
| `a2a_roundtrip.rs` | Non-object inputs rejected with `Error::Validation` before any I/O | `A2aClient` input mapping |
| `symphony_plan.rs` (3) | Register capabilities → `plan` → `compile` to a `CompiledGraph` → execute with a recording runner | `openjiuwen-symphony` (`SymphonyRuntime`, `CapabilityRegistry`, `PlanExecutor`), `openjiuwen-graph` |
| `symphony_plan.rs` | Plan determinism, `NeedsInput`/`NoPlan` outcomes | symphony planner |
| `symphony_plan.rs` | Graph snapshot published to file artifacts and reloaded by a fresh runtime | symphony artifact/asset stores |
| `self_improvement.rs` (4) | RSI orchestrator with a fake `HarnessRunner` converges after `patience` iterations | `openjiuwen-rsi` (`AutoHarnessOrchestrator`, `HarnessRunner`, `SessionSummary`) |
| `self_improvement.rs` | RSI checkpoint resume continues at the persisted iteration | `openjiuwen-rsi` `CheckpointStore`/`SessionState` |
| `self_improvement.rs` | Trainer improves a fake `TrainableTarget` and saves checkpoints | `openjiuwen-agent-evolving` (`Trainer`, `TrainableTarget`, `GridSearchOptimizer`, `FileCheckpointStore`) |
| `self_improvement.rs` | Trainer resume restores best score and re-runs the stored epoch | `openjiuwen-agent-evolving` checkpoints |
| `cli_binary.rs` (5 + 1 ignored) | `run_with_model` one-shot: `text`, `json`, `stream-json` output contracts | `openjiuwen-harness` `cli` (`run_with_model`, `CliIo`, `OutputFormat`) |
| `cli_binary.rs` | Pre-run validation (option resolution, blank prompt) rejects before any model call | `CliOptions::resolve_with`, `Cli::dispatch` |
| `cli_binary.rs` | Process-boundary `--version`/`--help` — `#[ignore]`d, see Known Gaps | `openjiuwen-cli` binary, resolved at runtime |
| `runner_subprocess.rs` (4 + 1 inert child) | `Runner::set_spawner` + `spawn_agent` launches this test binary as the NDJSON child; one `send` round-trips an `AgentOutput` | `openjiuwen-runner` `spawn` (`SubprocessSpawner`, `SpawnSpec`, `SpawnMessage`) |
| `runner_subprocess.rs` | The `AgentSpawner` seam returns a working `Arc<dyn SpawnedAgent>`; two sends on one child answer their own payloads | `AgentSpawner::spawn`, `SpawnedAgent::send` |
| `runner_subprocess.rs` | Explicit `health()` reports healthy, `shutdown(BOUND)` acknowledges and reaps the child | `SpawnedProcess` (`health`, `is_healthy`, `shutdown`, `is_alive`) |
| `runner_subprocess.rs` | An `error` frame surfaces `ErrorKind::Transport`; non-frame JSON surfaces `ErrorKind::Validation` — neither hangs | `SubprocessConfig::with_request_timeout`, frame parsing |

## Constraints

- Offline and deterministic: `ScriptedModel`/`MockLlmServer`/loopback sockets only; no credentials, no `Runner::global()` (always a local `Runner::new()`), no `MemoryCheckpointer::shared()`.
- No sleeps for ordering: a scenario waits on the event, frame, or state transition it asserts; timeouts exist only so a hang fails fast (`support::BOUND`).
- Never assert non-deterministic fields: timestamps, `duration_ms`, `plan_id`/`run_id` uuids, `GraphRevision::created_at`, or ids you did not read back from the producer.
- Every test file starts with `#![allow(clippy::unwrap_used, clippy::expect_used)]` (workspace lints are warnings and the gate is `-D warnings`); shared helper modules additionally allow `dead_code`, `unreachable_pub`.
- Each scenario owns its temp directory (`tempfile::TempDir` / `TempWorkspace`) and keeps it alive for the test body; nothing writes outside it.
- Loopback services bind `127.0.0.1:0` and read the address back; every scenario shuts its server down (`ServerHandle::shutdown` + `join`).
- A scenario that needs a sibling API change is `#[ignore = "<reason>"]`d with the gap recorded under **Known Gaps**, never fixed by editing the sibling crate.
- `openjiuwen-e2e` is not in the workspace `default-members`: use `cargo test -p openjiuwen-e2e` (and `--features a2a`).
- `openjiuwen-agent-teams` is taken with its default features only (`crates/openjiuwen-e2e/Cargo.toml:20`); the umbrella's `sqlite` feature forwards to extensions/symphony but not to agent-teams (`crates/openjiuwen/Cargo.toml:42`), so a SQLite team store needs a crate feature forwarding `openjiuwen-agent-teams/sqlite`.

## Routing Table

- `./tests/kernel_flow.rs` → model → tool → session → workflow → runner flows (kernel crates)
- `./tests/deep_agent.rs` → DeepAgent task loop, workspace tools, permissions, interrupt/resume
- `./tests/provider_native.rs` → provider-neutral harness SPI: factory, native harness lifecycle, IO adapter
- `./tests/team_flow.rs` → agent teams: spec build, leader/teammate runtime, task board (umbrella `openjiuwen::agent_teams` paths only, scripted `MemberHarnessFactory` + `InMemoryTeamStore`)
- `./tests/a2a_roundtrip.rs` → A2A client/server round trip (feature `a2a`)
- `./tests/symphony_plan.rs` → symphony capability registry, planning, graph execution, artifact reload
- `./tests/self_improvement.rs` → RSI orchestrator and agent-evolving trainer with fake runners
- `./tests/cli_binary.rs` → harness CLI layer (one-shot output formats, pre-run validation, process boundary)
- `./tests/runner_subprocess.rs` → runner subprocess spawn seam: self-executing NDJSON child, request/reply, health, graceful shutdown, typed transport/validation errors
- `./tests/support/mod.rs` → cross-scenario deadline and stream helpers (`BOUND`, `within`, `collect_bounded`)
- `./Cargo.toml` → manifest, test-only dependency set, `a2a`/`otel` features

## Test Strategy

- One binary per scenario area so a failure names the seam and the suite parallelises safely (each binary has its own temp dirs, its own model script, its own loopback port).
- Assert on recorded evidence, not timing: `ScriptedModel::calls()`/`call_count()`/`remaining()`, harness event sequences, collected frames, task board state, store and file contents, `MockLlmServer::request_count`.
- Fake only what a scenario cannot own: harness protocols, member factories, symphony capability runners, RSI harness runners, evolving targets. Each fake is copied into the scenario that needs it (test binaries cannot import each other), keeping every binary's contract explicit.
- Gates: `cargo fmt -p openjiuwen-e2e -- --check`, `cargo test -p openjiuwen-e2e`, `cargo test -p openjiuwen-e2e --features a2a`, `cargo clippy -p openjiuwen-e2e --all-targets [--features a2a] --no-deps -- -D warnings`. The whole suite runs in well under a second of test time and is stable across repeated runs.

## Design Decisions

- `runner_subprocess.rs` re-executes **its own test binary** as the fake agent child instead of shipping a second binary: `std::env::current_exe()` plus the libtest filter `fake_agent_child --exact --test-threads=1 --nocapture`.
- The child entry point is a normal `#[test]` that returns immediately unless `E2E_FAKE_AGENT_CHILD` is set (the parent sets it), so it is never `#[ignore]`d and the default suite runs it as a silent pass; `--nocapture` is mandatory or libtest swallows the protocol frames.
- `E2E_FAKE_AGENT_MODE` selects `echo`/`error_frame`/`bad_frame`; the background health task is disabled (`with_health_check(Duration::ZERO, …)`) so no probe races an exchange.

- Shared helpers are limited to the timing/stream contract; fixtures live in `openjiuwen-test-support` (models, workspaces, sessions, checkpoints) so library and scenario suites use the same fakes.
- Dependencies live under `[dependencies]` rather than `[dev-dependencies]`: the `a2a`/`otel` features must enable features of the same crates, and a feature may not reliably forward to a dev-dependency.
- `axum`/`tower` are intentionally absent: the A2A scenario uses a real loopback socket (`A2aServer::serve_in_background` + `A2aClient`) instead of an in-process `tower::ServiceExt::oneshot` harness, so the transport, the JSON-RPC envelope and the SSE framing are all exercised.
- `reqwest` is a direct dependency for the agent-card endpoint (the A2A client itself lives in `openjiuwen-extensions`).
- `wiremock` is reached through `openjiuwen_test_support::wiremock` when a scenario needs a non-LLM HTTP fake; no separate dependency.

## Known Behaviours

Deterministic, surprising-but-correct outcomes the scenarios assert; do not "fix" them.
- `create_harness("native", …)` builds a model-less `DeepAgent`: the factory cannot take an `Arc<dyn Model>`, so the turn is resolved lazily from the flat `model`/`provider`/`api_base`/`api_key` keys (a loopback `MockLlmServer` works) or the caller drives `NativeHarness::new(agent, …)` with an injected model.
- A DeepAgent HTTP round needs model identity: an injected model alone fails with `Error::Config("model_name must not be empty…")` unless `DeepAgentConfig` also sets `model_name` (and the matching request config).
- Permission denial is not an `Err`: `result_type` stays `Answer`, the observation starts with `DENIED_PREFIX`, and the next model call carries the same marker in a `Role::Tool` message.
- `TeamAgentSpec::default()` does not validate; `agents["leader"]` must exist, and member rows start `Unstarted` until `spawn_member`.
- `TaskBoard::create_task` publishes both `task_created` and an initial `task_status_changed`; `spawn_member` steers a "member ready" notice to the leader after the caller's query.
- `PlanExecution::steps` is a `BTreeMap`, so its keys read back lexicographically — assert plan ordering on `OrchestrationPlan::steps`/`stages()`, not on the map.
- A2A: the server auto-completes any sink left open, so `INPUT_REQUIRED` is the only durable non-terminal state reachable through a real `SendMessage` and is what `CancelTask` acts on; `message/stream` is always routed regardless of the card's `capabilities.streaming`.
- RSI: the first graded candidate is adopted as incumbent (`accepted == true`, `improvement ≈ 0`); convergence is reported on the following flat iteration.

## Known Gaps

- `cli_binary.rs` cannot use `env!("CARGO_BIN_EXE_openjiuwen")`: Cargo defines that variable only for tests of the package that owns the binary. The process-level smoke test lives in `crates/openjiuwen-cli/tests/cli_smoke.rs`; the e2e suite asserts the reachable layers (`run_with_model` output formats, `CliOptions::resolve_with` validation, `Cli::dispatch`) and keeps one runtime-resolution process test `#[ignore]`d with that reason.
- The clap grammar itself is not reachable from this crate: `clap` is not a dependency and `openjiuwen-harness` does not re-export `Parser`, so `--help`/`--version`/exit-code-2 verdicts cannot be asserted here. A `pub use clap;` (or a `Cli::parse_from` helper) in `openjiuwen_harness::cli` would close the gap.
- Vendor harness providers (`claudecode`, `codex`, `dsh`) and every live-provider path (`live-tests`) need credentials/network and are out of scope; `provider_native.rs` asserts only that they report `Error::Unsupported`/`NotFound`.
- Deferred ML/LLM-backed areas (symphony `deferred::*`, `evaluation::LlmJudgedMetric`; agent-evolving RL/judge stubs; RSI `deferred::*`) return `Error::Unsupported` and are not exercised.
- `otel` has no scenario: the seam is a no-op without the feature and its live path needs a collector; behaviour is covered by `openjiuwen-extensions` unit tests. The feature is declared so the gate can be run once a collector-free scenario exists.

## Reference Patterns

Copy the shape of the sibling suites instead of inventing fixtures: `../openjiuwen-agent/tests/react_agent.rs` (tool round), `../openjiuwen-workflow/tests/linear.rs` + `references.rs` (`${a.b}` flows), `../openjiuwen-runner/tests/run_agent.rs` (facade), `../openjiuwen-harness/tests/deep_agent_loop.rs` + `permissions.rs` (harness rounds), `../openjiuwen-harness-providers/tests/{native,io_adapter}.rs` (harness lifecycle, event → chunk projection), `../openjiuwen-agent-teams/tests/team_run.rs` + `task_flow.rs`, `../openjiuwen-symphony/tests/{executor,runtime}.rs`, `../openjiuwen-rsi/tests/{orchestrator,checkpoint}.rs`, `../openjiuwen-agent-evolving/tests/trainer.rs`, `../openjiuwen-extensions/tests/a2a_*.rs`, `../openjiuwen-harness/tests/cli.rs`, `../openjiuwen-cli/tests/cli_smoke.rs`.
