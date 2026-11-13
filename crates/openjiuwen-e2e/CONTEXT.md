# openjiuwen-e2e

## Intent

The cross-crate scenario suite of the Rust rewrite: the only place where several public crates are driven together through their real seams.
It answers "does the composed system work?" — kernel flow, workflow components, context compression, retrieval, memory, DeepAgent harness, harness tools/CLI, provider SPI, agent teams (including storage/attach), A2A, symphony, self-improvement (evaluator, ledger, trainer), subprocess spawning, and the CLI — while every crate's own `tests/` answer "does this module work?".
Every scenario is offline, deterministic and bounded: scripted models or a loopback `MockLlmServer`/`wiremock` server, temp directories, loopback sockets and Unix sockets only, no credentials, no wall-clock ordering.
The crate is dev-only (`publish = false`) and never a dependency of a library crate.

## API Surface

- `src/lib.rs` — marker module (`crate_name()`); the crate has no runtime API.
- `tests/support/mod.rs` — the cross-scenario helper contract: `BOUND` (10 s per-await deadline), `within(what, future)`, `collect_bounded(stream, max)`. Scenario-specific fixtures stay in `openjiuwen-test-support` or beside the scenario.
- Test binaries (each `tests/*.rs` is its own binary; no `[[test]]` entries):
  `kernel_flow`, `workflow_components`, `context_engine`, `retrieval_pipeline`, `memory_recall`, `deep_agent`, `harness_cli_tools`, `provider_native`, `team_flow`, `team_backend`, `a2a_roundtrip`, `symphony_plan`, `self_improvement`, `evaluation_ledger`, `runner_subprocess`, `cli_binary`.
- Features: `default = []`; `a2a` (enables `openjiuwen/a2a` + `openjiuwen-extensions/a2a`, gates `a2a_roundtrip.rs`); `otel` (declared for parity, no scenario); `sqlite` (enables `openjiuwen-agent-teams/sqlite`, gates only the SQLite test inside `team_backend.rs`).
- Dependencies are declared under `[dependencies]` so the `a2a`/`otel`/`sqlite` features can enable features on the same crates: the umbrella `openjiuwen` (module aliases `core`, `harness`, `harness_protocol`, `harness_providers`, `agent_teams`, `extensions`, `symphony`, `rsi`, `agent_evolving`), the concrete harness-layer crates `openjiuwen-harness`, `openjiuwen-harness-protocol`, `openjiuwen-harness-providers`, `openjiuwen-agent-teams`, `openjiuwen-extensions`, `openjiuwen-symphony`, `openjiuwen-rsi`, `openjiuwen-agent-evolving`, plus `openjiuwen-test-support`, `tokio`, `tokio-util`, `futures`, `async-trait`, `anyhow`, `reqwest`, `serde_json`, `tempfile`.
- No kernel crate (`openjiuwen-session`, `-workflow`, `-graph`, `-agent`, `-foundation`, `-runner`, `-context`, `-memory`, `-retrieval`, `-common`) is a direct dependency: kernel types are imported through the umbrella as `openjiuwen::core::<module>::…` (e.g. `openjiuwen::core::workflow::{Workflow, WorkflowCard, …}`, `openjiuwen::core::context_engine::{ContextEngine, ContextConfig, …}`, `openjiuwen::core::retrieval::{AutoLoader, Bm25Index, …}`, `openjiuwen::core::memory::{LongTermMemory, NewMemory, …}`, `openjiuwen::core::runner::{Runner, WorkflowRef, SubprocessSpawner, …}`), which requires no feature.
- Model/session fixtures come from `openjiuwen_test_support`, never from `tests/support/mod.rs`: `agent_session`, `agent_session_with`, `agent_session_with_checkpointer`, `workflow_session`, `workflow_session_with`, `workflow_session_with_checkpointer`, `memory_checkpointer`, `ScriptedModel`, `MockCompletion`, `MockLlmServer`, `ScriptedResponse`, `TEST_API_KEY`, plus the re-exported `wiremock` (`openjiuwen_test_support::wiremock`).
- `a2a_roundtrip.rs` needs `#![allow(missing_docs)]` **before** its `#![cfg(feature = "a2a")]`: with the feature off the whole crate body (including its `//!` docs) is stripped, so a later `allow` is stripped too and the denied `missing_docs` lint fires on the empty crate root.

## Scenario Matrix

79 test functions across 15 scenario binaries: 78 pass with `--all-features` (73 with default features), and exactly 1 documented `#[ignore]`.

| Binary | Scenario | Crates / seams exercised |
|---|---|---|
| `kernel_flow.rs` (6) | ReAct tool round against `ScriptedModel`; tool message fed back to the model | `openjiuwen-test-support` (`ScriptedModel`), `openjiuwen-agent` (`ReActAgent`, `FunctionTool`), `openjiuwen-foundation` (`Model`) |
| `kernel_flow.rs` | Session global + component state round-tripped through a shared `MemoryCheckpointer` | `openjiuwen-session` (`Session`, `Checkpointer`), test-support session fixtures |
| `kernel_flow.rs` | Workflow `start → llm → end` with `${start.query}` / `${llm.result}` references | `openjiuwen-workflow` (builder, `StartComponent`/`LlmComponent`/`EndComponent`), `openjiuwen-graph` |
| `kernel_flow.rs` | Local `Runner` runs an agent and a workflow; agent streaming frame contract | `openjiuwen-runner` (`Runner::new`, `ResourceMgr`, `RunOptions`, `run_agent_streaming`) |
| `workflow_components.rs` (5) | `start → http → end` against a loopback `wiremock` server; `${http.body.<field>}` and `${http.statusCode}` resolve | `openjiuwen-workflow` `HttpRequestComponent` + `HttpComponentConfig` (deserialized from JSON), `wiremock` |
| `workflow_components.rs` | A non-2xx response is data, not an error (`${http.ok}`, status and body reach the end node) | `HttpRequestComponent` response projection |
| `workflow_components.rs` | Parent workflow whose `sub` node is a `SubWorkflowComponent`; `${sub.output.answer}` reaches the parent end | `SubWorkflowComponent::new(&child)`, child `Workflow` + `LlmComponent` |
| `workflow_components.rs` | `QuestionerComponent` pauses (`InputRequired` + `__interaction__` wire), `resume(InteractiveInput, session)` completes | `QuestionerComponent`, `InteractiveInput`, `MemoryCheckpointer` |
| `workflow_components.rs` | Same HTTP topology driven through `Runner::run_workflow(WorkflowRef::of(Arc<Workflow>), …)` | `openjiuwen-runner` facade over the workflow engine |
| `context_engine.rs` (4) | 40-message transcript → `compress` reports `Compressed`, tokens and stored messages shrink, a second pass is `Noop` | `openjiuwen-context` (`ContextEngine`, `ContextConfig`, `ProcessorConfig::Truncation`, character token counting) |
| `context_engine.rs` | The turn after compression still reaches the model: `prepare` yields a bounded non-overflowing window fed to `ScriptedModel` | `ModelContext::messages`, `PrepareOptions`, `openjiuwen-foundation` `Model` |
| `context_engine.rs` | `prepare` previews on a copy while only `compress` persists the shrunken history | context pipeline persist/no-persist contract |
| `context_engine.rs` | A `ReActAgent` with an injected tiny `ContextConfig` runs two turns and prepares a smaller window than its stored history | `ReActAgentConfig::configure_context_engine`, `agent.context_engine()` |
| `retrieval_pipeline.rs` (6) | DOCX fixture → `AutoLoader::load` → paragraphs, tables and metadata | `openjiuwen-retrieval` `loaders::DocxLoader`, `parsing::crc32` |
| `retrieval_pipeline.rs` | XLSX fixture → deterministic row/column documents and ids | `loaders::XlsxLoader` |
| `retrieval_pipeline.rs` | PDF fixture → extracted text, `page_count`, `has_text` | `loaders::PdfLoader` |
| `retrieval_pipeline.rs` | Fixtures chunked with `CharChunker` → `Bm25Index` → `Bm25Query` ranks the alice chunk first with `doc_id`/`chunk_id` metadata | `openjiuwen-retrieval` `Bm25Index`, `Bm25Config`, `Bm25Query`, `stats`/`idf` |
| `retrieval_pipeline.rs` | `SparseRetriever` serves BM25, honours top-k/threshold, rejects non-sparse modes with `Error::Validation` | `retriever::SparseRetriever`, `RetrieveMode` |
| `retrieval_pipeline.rs` | `SimpleKnowledgeBase` with `index_type = "bm25"` + `HashEmbedding` + `InMemoryVectorStore`: `parse_files` then `retrieve("alice")` | `SimpleKnowledgeBase`, `KnowledgeBaseConfig`, knowledge-base statistics |
| `memory_recall.rs` (6) | Remembered facts recalled through the keyword (BM25) path; evidence fields and ranking asserted | `openjiuwen-memory` `LongTermMemory::in_memory`, `NewMemory`, `SearchMode::Keyword` |
| `memory_recall.rs` | Exact tokens outrank partial tokens; a no-overlap query returns nothing above the default threshold | memory BM25 keyword scoring, `SearchOptions::with_threshold` |
| `memory_recall.rs` | Semantic recall with `HashEmbedding` returns the vocabulary match first (`similarity > 0`) | `LongTermMemory::builder().embedding(…)`, `SearchMode::Semantic`/`Hybrid` |
| `memory_recall.rs` | Without an embedding, `Semantic` is `Error::Unsupported` while `Hybrid` degrades to keyword-only hits (`similarity == 0.0`) | memory semantic-fallback contract |
| `memory_recall.rs` | Retrieval → memory bridge: BM25 top chunk remembered and recallable by keyword and by semantic mode | `Bm25Index` + `NewMemory` + one shared `HashEmbedding` |
| `memory_recall.rs` | `remember(ExtractionInput::from_text(…))` yields a `RememberReport` whose record is recallable and tagged | rule-based analyzer, `RememberReport` |
| `deep_agent.rs` (4) | DeepAgent tool round in a temp workspace (read seeded file → write new file → answer) | `openjiuwen-harness` (`DeepAgentBuilder`, `ToolRegistry`, `BuiltinTool`), `openjiuwen-agent::Agent` |
| `deep_agent.rs` | Same round over HTTP through `MockLlmServer` (also needs `with_model_name`/`with_model_config`, see Known Behaviours) | `openjiuwen-foundation` `init_model`, test-support `MockLlmServer` |
| `deep_agent.rs` | Permission `Deny` surfaces `[PERMISSION_DENIED]` to the model; the loop continues with `ResultType::Answer` | `openjiuwen-harness` `permission` (`PermissionsSection`, `PermissionLevel`) |
| `deep_agent.rs` | `Ask` rule interrupts; resume with `interrupt_answers` on the same session | `PermissionInterruptRail`, `openjiuwen-agent` `ResultType::Interrupt`, session interactions |
| `harness_cli_tools.rs` (7) | `fetch_webpage` against a loopback `wiremock` HTML page: title, extracted text, status and envelope | `openjiuwen-harness` `ToolRegistry` + `BuiltinTool::FetchWebpage`, `wiremock` |
| `harness_cli_tools.rs` | `fetch_webpage` on a JSON endpoint pretty-prints the body and reports no title | fetch response projection |
| `harness_cli_tools.rs` | A 500 response becomes a `success == false` payload with the status in `error` (never `Err`) | fetch error surface |
| `harness_cli_tools.rs` | Cron create → list → `due(next_run_at)` fires once and advances → cancel; a second scheduler over the same dir sees the job | `openjiuwen-harness` `CronScheduler`, `CronTool`, `CronJob` (time injected, no `chrono` dependency) |
| `harness_cli_tools.rs` | `--no-color` parses before/after a subcommand and `paths::color_enabled` + `Style` bytes gate colour | `openjiuwen-harness` `cli::Cli`, `cli::paths`, `cli::style::Style`, `MapEnv` |
| `harness_cli_tools.rs` | `Repl::handle_input` maps every slash command and prompt form | CLI REPL parser (`ReplCommand`) |
| `harness_cli_tools.rs` | One scripted chat turn through `Repl::run()` over a `CliIo` double is escape-free without colour | CLI REPL run loop, scripted model |
| `provider_native.rs` (5) | Provider registry lists/describes `native`; unknown → `NotFound`, vendor → `Unsupported` | `openjiuwen-harness-providers` factory + registry |
| `provider_native.rs` | Native harness `start` → `send` → event drain (`StateChanged`/`TurnLifecycle`/`Output`) → `stop` | `openjiuwen-harness-protocol` (`HarnessProtocol`, `HarnessEventCursor`), `openjiuwen-harness` (`NativeHarness`) |
| `provider_native.rs` | `create_harness("native", json)` runs a real turn over loopback `MockLlmServer` | `openjiuwen-harness-providers::create_harness` factory path |
| `provider_native.rs` | `HarnessIOAdapter` projects events onto `AgentChunk`s (output, tool call/result, optional turn lifecycle) | `openjiuwen-harness-providers` `io_adapter`, hand-built `HarnessProtocol` double |
| `team_flow.rs` (3) | `TeamAgentSpec` → `build_with` a scripted `MemberHarnessFactory`; roster and member cards | `openjiuwen-agent-teams` (`TeamAgentSpec`, `SpecBuild`, `TeamBuildDeps`) |
| `team_flow.rs` | Leader/teammate dispatch; exactly one `team.completed` marker, member frames tagged | `openjiuwen-agent-teams` runtime + stream tagging |
| `team_flow.rs` | Task board create → claim → complete settles the board | `openjiuwen-agent-teams` `TaskBoard` |
| `team_backend.rs` (4, 5 with `sqlite`) | Event bus subscription records `member_status_changed`, mailbox and task publications in order, plus the settled run's single `team_completed` publication | `TeamBackend::messager`, `Messager::subscribe`, `TeamTopic`, `EventMessage` |
| `team_backend.rs` | Mailbox send / `unread` / `mark_read` and the task board read back through the backend | `TeamBackend::send_message`, `TaskBoard` |
| `team_backend.rs` | Per-member checkpoint payload round-trips through `HarnessCheckpoint` → `NativeCheckpoint` and validates its session/card | `export_checkpoint`, `NativeCheckpoint::validate_compatible` |
| `team_backend.rs` | SQLite team store (`StorageSpec { kind: "sqlite" }`) persists board rows and mail across store instances | `open_store`, `SqliteTeamStore` (feature `sqlite`) |
| `team_backend.rs` | Team attach over a Unix socket: `list_members`, `snapshot`, `send_member`, `stop_team`, then transport error after teardown | `AttachClient`, attach host started from `spec.metadata["attach"]` |
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
| `evaluation_ledger.rs` (6) | Evaluator pipeline over `FakeCommandRunner`: failing test summary scores `2/3`, one attempt, container removed | `agent_evolving::evaluator_pipeline` (`EvaluationPipeline`, `PipelineConfig`, `ScoreSpec`, `ExecResult`) |
| `evaluation_ledger.rs` | `run_all` + `summary` over a passing and a failing task (`tasks`/`passed`/`failed`/`mean_score`) | evaluator pipeline batch summary |
| `evaluation_ledger.rs` | A missing `docker` binary is `Error::Unsupported` before any command runs | `FakeCommandRunner::with_program_missing` |
| `evaluation_ledger.rs` | `RunLedger` deduplicates call ids, aggregates totals/metrics/snapshot, orders store events, gates delivery on pending calls | `rsi::RunLedger`, `InMemoryLedgerStore`, `StageRecord` |
| `evaluation_ledger.rs` | `emit_progress` invokes an injected callback once and is a no-op without one | `rsi::harness_rsi::progress` |
| `evaluation_ledger.rs` | Orchestrator event stream is ordered and ends with the summary; a ledger kept beside it matches evaluated candidates | `AutoHarnessEvent` stream + scripted `HarnessRunner` |
| `runner_subprocess.rs` (5, 1 is the inert child) | `Runner::set_spawner` + `spawn_agent` launches this test binary as the NDJSON child; one `send` round-trips an `AgentOutput` | `openjiuwen-runner` `spawn` (`SubprocessSpawner`, `SpawnSpec`, `SpawnMessage`) |
| `runner_subprocess.rs` | The `AgentSpawner` seam returns a working `Arc<dyn SpawnedAgent>`; two sends answer their own payloads | `AgentSpawner::spawn`, `SpawnedAgent::send` |
| `runner_subprocess.rs` | Explicit `health()` reports healthy, `shutdown(BOUND)` acknowledges and reaps the child | `SpawnedProcess` (`health`, `is_healthy`, `shutdown`, `is_alive`) |
| `runner_subprocess.rs` | An `error` frame surfaces `ErrorKind::Transport`; non-frame JSON surfaces `ErrorKind::Validation` — neither hangs | `SubprocessConfig::with_request_timeout`, frame parsing |
| `cli_binary.rs` (5 + 1 ignored) | `run_with_model` one-shot: `text`, `json`, `stream-json` output contracts | `openjiuwen-harness` `cli` (`run_with_model`, `CliIo`, `OutputFormat`) |
| `cli_binary.rs` | Pre-run validation (option resolution, blank prompt) rejects before any model call | `CliOptions::resolve_with`, `Cli::dispatch` |
| `cli_binary.rs` | Process-boundary `--version`/`--help` — `#[ignore]`d, see Known Gaps | `openjiuwen-cli` binary, resolved at runtime |

## Constraints

- Offline and deterministic: `ScriptedModel`/`MockLlmServer`/`wiremock`/loopback and Unix sockets only; no credentials, no `Runner::global()` (always a local `Runner::new()`), no `MemoryCheckpointer::shared()`.
- No sleeps for ordering: a scenario waits on the event, frame, or state transition it asserts; timeouts exist only so a hang fails fast (`support::BOUND`).
- Never assert non-deterministic fields: timestamps, `duration_ms`, `plan_id`/`run_id` uuids, `GraphRevision::created_at`, generated memory/chunk/doc ids, or ids you did not read back from the producer.
- Cron is tested by time injection, never by waiting: create through the tool, deserialize `payload["data"]["job"]`, then pass `job.next_run_at` to `CronScheduler::due`. `chrono` is deliberately not a dependency.
- Every test file starts with `#![allow(clippy::unwrap_used, clippy::expect_used)]` (workspace lints are warnings and the gate is `-D warnings`); shared helper modules additionally allow `dead_code`, `unreachable_pub`.
- Each scenario owns its temp directory (`tempfile::TempDir` / `TempWorkspace`) and keeps it alive for the test body; nothing writes outside it.
- Loopback services bind `127.0.0.1:0` and read the address back; Unix-socket paths stay inside the test's `TempDir` and use short prefixes (a long `tempfile` prefix overflows `sun_path`).
- A scenario that needs a sibling API change is recorded under **Known Gaps**, never fixed by editing the sibling crate; the suite adds no new `#[ignore]`d tests.
- `openjiuwen-e2e` is not in the workspace `default-members`: use `cargo test -p openjiuwen-e2e` (plus `--features a2a,otel` and `--features sqlite`).
- The umbrella's `sqlite` feature does not forward to `openjiuwen-agent-teams`, so this crate declares `sqlite = ["openjiuwen-agent-teams/sqlite"]` and gates only the SQLite test inside `team_backend.rs`; the default build stays `sqlx`-free.
- No new crates.io dependency may be added: `wiremock` is reached through `openjiuwen_test_support::wiremock`, `clap` through `openjiuwen_harness::clap`, and time types are never named directly.

## Routing Table

- `./tests/kernel_flow.rs` → model → tool → session → workflow → runner flows (kernel crates)
- `./tests/workflow_components.rs` → `SubWorkflowComponent`, `HttpRequestComponent` (loopback), `QuestionerComponent` HITL resume, runner façade
- `./tests/context_engine.rs` → context window assembly, truncation/compress persist-vs-preview, `ReActAgent` context wiring
- `./tests/retrieval_pipeline.rs` → DOCX/XLSX/PDF fixtures → loaders → chunking → BM25 → `SparseRetriever` → knowledge base
- `./tests/memory_recall.rs` → remember/recall, keyword vs semantic vs hybrid fallback, retrieval→memory bridge
- `./tests/deep_agent.rs` → DeepAgent task loop, workspace tools, permissions, interrupt/resume
- `./tests/harness_cli_tools.rs` → `fetch_webpage` against a loopback server, cron scheduling, `--no-color`, slash commands
- `./tests/provider_native.rs` → provider-neutral harness SPI: factory, native harness lifecycle, IO adapter
- `./tests/team_flow.rs` → agent teams: spec build, leader/teammate runtime, task board (umbrella `openjiuwen::agent_teams` paths only, scripted `MemberHarnessFactory` + `InMemoryTeamStore`)
- `./tests/team_backend.rs` → team event bus, mailbox, per-member checkpoint payloads, SQLite store (feature `sqlite`), Unix-socket attach
- `./tests/a2a_roundtrip.rs` → A2A client/server round trip (feature `a2a`)
- `./tests/symphony_plan.rs` → symphony capability registry, planning, graph execution, artifact reload
- `./tests/self_improvement.rs` → RSI orchestrator and agent-evolving trainer with fake runners
- `./tests/evaluation_ledger.rs` → evaluator pipeline with a fake command runner, RSI ledger, progress events, event stream
- `./tests/runner_subprocess.rs` → runner subprocess spawn seam: self-executing NDJSON child, request/reply, health, graceful shutdown, typed transport/validation errors
- `./tests/cli_binary.rs` → harness CLI layer (one-shot output formats, pre-run validation, process boundary)
- `./tests/support/mod.rs` → cross-scenario deadline and stream helpers (`BOUND`, `within`, `collect_bounded`)
- `./Cargo.toml` → manifest, test-only dependency set, `a2a`/`otel`/`sqlite` features

## Test Strategy

- One binary per scenario area so a failure names the seam and the suite parallelises safely (each binary has its own temp dirs, its own model script, its own loopback port, its own Unix socket).
- Assert on recorded evidence, not timing: `ScriptedModel::calls()`/`call_count()`/`remaining()`, harness event sequences, collected frames, task board and mailbox state, store and file contents, `MockLlmServer::request_count`, `FakeCommandRunner::invocations()`, ledger `totals()`/`events()`.
- Fake only what a scenario cannot own: harness protocols, member factories, symphony capability runners, RSI harness runners, evolving targets, Docker (`FakeCommandRunner`), the subprocess child (the test binary itself). Each fake is copied into the scenario that needs it (test binaries cannot import each other), keeping every binary's contract explicit.
- Fixtures that the library keeps private are rebuilt from public primitives inside the scenario (`build_zip` for the DOCX/XLSX fixtures uses `parsing::crc32`), never by widening library visibility.
- Gates: `cargo fmt -p openjiuwen-e2e -- --check`; `cargo test -p openjiuwen-e2e --offline`; `--features a2a,otel`; `--features sqlite`; `cargo clippy -p openjiuwen-e2e --all-targets [--all-features] --offline -- -D warnings`. The whole suite runs in well under a second of test time (the subprocess and attach scenarios included) and is stable across repeated runs.

## Design Decisions

- `runner_subprocess.rs` re-executes **its own test binary** as the fake agent child instead of shipping a second binary: `std::env::current_exe()` plus the libtest filter `fake_agent_child --exact --test-threads=1 --nocapture`.
- The child entry point is a normal `#[test]` that returns immediately unless `E2E_FAKE_AGENT_CHILD` is set (the parent sets it), so it is never `#[ignore]`d and the default suite runs it as a silent pass; `--nocapture` is mandatory or libtest swallows the protocol frames.
- `E2E_FAKE_AGENT_MODE` selects `echo`/`error_frame`/`bad_frame`; the background health task is disabled (`with_health_check(Duration::ZERO, …)`) so no probe races an exchange.
- Shared helpers are limited to the timing/stream contract; fixtures live in `openjiuwen-test-support` (models, workspaces, sessions, checkpoints) so library and scenario suites use the same fakes.
- Dependencies live under `[dependencies]` rather than `[dev-dependencies]`: features must enable features of the same crates, and a feature may not reliably forward to a dev-dependency.
- `axum`/`tower` are intentionally absent: the A2A scenario uses a real loopback socket (`A2aServer::serve_in_background` + `A2aClient`) instead of an in-process `tower::ServiceExt::oneshot` harness, so the transport, the JSON-RPC envelope and the SSE framing are all exercised.
- Non-LLM HTTP fakes use `openjiuwen_test_support::wiremock`; set bodies with `set_body_raw(body, mime)` because `set_body_string` forces `text/plain`.
- `reqwest` is a direct dependency for the agent-card endpoint (the A2A client itself lives in `openjiuwen-extensions`).

## Known Behaviours

Deterministic, surprising-but-correct outcomes the scenarios assert; do not "fix" them.
- `create_harness("native", …)` builds a model-less `DeepAgent`: the factory cannot take an `Arc<dyn Model>`, so the turn is resolved lazily from the flat `model`/`provider`/`api_base`/`api_key` keys (a loopback `MockLlmServer` works) or the caller drives `NativeHarness::new(agent, …)` with an injected model.
- A DeepAgent HTTP round needs model identity: an injected model alone fails with `Error::Config("model_name must not be empty…")` unless `DeepAgentConfig` also sets `model_name` (and the matching request config).
- Permission denial is not an `Err`: `result_type` stays `Answer`, the observation starts with `DENIED_PREFIX`, and the next model call carries the same marker in a `Role::Tool` message.
- `HttpRequestComponent` builds its own HTTP client per invoke; a non-2xx status is ordinary data (`ok == false`) while a transport failure is a coded `Error::Workflow`.
- A `QuestionerComponent` interaction id is the question object's `id` when present, otherwise the component id; resuming requires a session with a checkpointer.
- `SubWorkflowComponent` returns the child's end output unchanged (`${sub.output.<key>}`) and relays a child `INPUT_REQUIRED` onto the parent, so a nested resume restarts the child.
- `ContextEngine::prepare` runs processors on a copy and never persists; only `compress` persists and only reports `Compressed` when a processor actually changed something.
- `ReActAgentConfig::configure_context_engine` is the only context-engine wiring and `configure()` rebuilds the engine, dropping existing contexts.
- Memory: `SearchMode::Semantic` without an embedder is `Error::Unsupported`; `Hybrid` without one degrades to keyword-only with `similarity == 0.0` (a sentinel, not `None`); the default threshold is `0.3`.
- Retrieval: chunk and DOCX/PDF document ids are uuid4, so they are read back from the producer; XLSX ids are deterministic (`people_s0_r2`). `SimpleKnowledgeBase::parse_files` parses but does not ingest.
- `TeamAgentSpec::default()` does not validate; `agents["leader"]` must exist, and member rows start `Unstarted` until `spawn_member`.
- `TaskBoard::create_task` publishes both `task_created` and an initial `task_status_changed`; `spawn_member` steers a "member ready" notice to the leader after the caller's query.
- A settled run publishes both halves of completion: the `team.completed` result-stream marker (the last frame, exactly one) and one `EventPayload::TeamCompleted` bus event on `TeamTopic::Team` with `reason: Some("settled")`, published between the pre-run publications and any task the caller creates afterwards.
- `build_with` consumes `TeamBuildDeps.store` and ignores `TeamAgentSpec::storage`; a SQLite store must be injected through the deps.
- `PlanExecution::steps` is a `BTreeMap`, so its keys read back lexicographically — assert plan ordering on `OrchestrationPlan::steps`/`stages()`, not on the map.
- A2A: the server auto-completes any sink left open, so `INPUT_REQUIRED` is the only durable non-terminal state reachable through a real `SendMessage` and is what `CancelTask` acts on; `message/stream` is always routed regardless of the card's `capabilities.streaming`.
- RSI: the first graded candidate is adopted as incumbent (`accepted == true`, `improvement ≈ 0`); convergence is reported on the following flat iteration.
- `EvaluationPipeline` requires `docker run` to print a non-empty container id; a non-zero `docker exec` exit is data, not an error.
- Cron: `due(now)` fires strictly due jobs and advances `next_run_at`; re-running at the same instant yields nothing, and a cancelled job is never due.

## Known Gaps

- `cli_binary.rs` cannot use `env!("CARGO_BIN_EXE_openjiuwen")`: Cargo defines that variable only for tests of the package that owns the binary. The process-level smoke test lives in `crates/openjiuwen-cli/tests/cli_smoke.rs`; the e2e suite asserts the reachable layers (`run_with_model` output formats, `CliOptions::resolve_with` validation, `Cli::dispatch`) and keeps one runtime-resolution process test `#[ignore]`d with that reason.
That ignored test resolves `<target>/<profile>/openjiuwen` then `<profile>/deps/openjiuwen` from `std::env::current_exe()` and returns early (prints a skip line) instead of failing when neither exists — so it is not a reliable gate.
It also spawns the binary with no `.env()`/`.env_clear()`/`.current_dir()`, so the child inherits the ambient environment (`OPENJIUWEN_HOME`, `API_KEY`, …); a new subprocess scenario must clear or pin the env explicitly.
`openjiuwen-cli` is not a dependency of this crate, so the `openjiuwen` binary is not guaranteed in the `-p openjiuwen-e2e` build graph.
- `--help`/`--version`/exit-code-2 verdicts are asserted in `crates/openjiuwen-cli/tests/cli_smoke.rs`, not here: flag parsing is reachable through `openjiuwen_harness::clap::Parser` but process exit behaviour belongs to the binary's own package.
- Per-member checkpoints are asserted at the payload level only: `openjiuwen-agent-teams` never constructs a `NativeCheckpoint` in its runtime (`HarnessContext.checkpoint` is always `None`), so `team_backend.rs` scripts `export_checkpoint` and validates the payload; wiring it into `HarnessMemberRuntime` would close the gap.
- `MemberHarnessContext` carries no card id and its `session_id` is the shared team host session, so the checkpoint payload uses the `<host>:<member>` pair and a scripted card id.
- The DOCX/XLSX ZIP fixture builder in `openjiuwen-retrieval` (`loaders::fixtures`) is `pub(crate)`, so `retrieval_pipeline.rs` keeps its own `build_zip` over the public `parsing::crc32`; loaders are path-only (no public byte-level `load_bytes`).
- `AutoHarnessOrchestrator` exposes no ledger hook, so `evaluation_ledger.rs` maintains its own `RunLedger` from `candidates_evaluated` events and asserts it against `SessionSummary::evaluated`; an orchestrator-owned ledger would close the gap.
- `SpawnedAgent` (the trait seam) does not expose `health()`/`is_healthy()`; only the concrete `SpawnedProcess` does, so the health scenario drives `SubprocessSpawner::spawn_process` (the same backend `set_spawner` installs).
- Vendor harness providers (`claudecode`, `codex`, `dsh`) spawn real vendor CLIs behind the `vendor` feature, so they need those binaries and credentials; `provider_native.rs` asserts the typed binary-resolution error instead.
- Deferred ML/LLM-backed areas (symphony `deferred::*`, `evaluation::LlmJudgedMetric`; agent-evolving RL/judge stubs; RSI `deferred::*`) return `Error::Unsupported` and are not exercised.
- `otel` has no scenario: the seam is a no-op without the feature and its live path needs a collector; behaviour is covered by `openjiuwen-extensions` unit tests. The feature is declared so the gate can be run once a collector-free scenario exists.

## Reference Patterns

Copy the shape of the sibling suites instead of inventing fixtures: `../openjiuwen-workflow/tests/{sub_workflow,http_component,questioner_component}.rs`, `../openjiuwen-context/tests/engine.rs`, `../openjiuwen-retrieval/tests/{loaders,sparse}.rs`, `../openjiuwen-memory/tests/{recall,facade_smoke,analyzer_wiring}.rs`, `../openjiuwen-agent/tests/react_agent.rs` (tool round), `../openjiuwen-workflow/tests/linear.rs` + `references.rs` (`${a.b}` flows), `../openjiuwen-runner/tests/{run_agent,spawn_subprocess}.rs`, `../openjiuwen-harness/tests/{deep_agent_loop,permissions,cli}.rs` and `../openjiuwen-harness/src/tools/web/tests.rs` + `src/tools/cron/tests.rs`, `../openjiuwen-harness-providers/tests/{native,io_adapter}.rs`, `../openjiuwen-agent-teams/tests/{team_run,task_flow,attach_e2e}.rs` and `../openjiuwen-agent-teams/tests/full_team_sqlite.rs`, `../openjiuwen-symphony/tests/{executor,runtime}.rs`, `../openjiuwen-rsi/tests/{orchestrator,checkpoint,ledger,progress}.rs`, `../openjiuwen-agent-evolving/tests/{pipeline,evaluator,trainer}.rs`, `../openjiuwen-extensions/tests/a2a_*.rs`, `../openjiuwen-cli/tests/cli_smoke.rs`.
