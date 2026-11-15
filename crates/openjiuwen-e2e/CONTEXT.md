# openjiuwen-e2e

## Intent

The cross-crate scenario suite of the Rust rewrite: the only place where several public crates are driven together through their real seams.
It answers "does the composed system work?" — kernel flow, workflow components, context compression and tokenizers, retrieval, memory (recall, dreaming, team memory), DeepAgent harness, harness tools/CLI, provider SPI, agent teams (including storage/attach), runner hosting and federation transports, A2A, symphony planning and object artifacts, self-improvement (evaluator, ledger, trainer), subprocess spawning, and the CLI — while every crate's own `tests/` answer "does this module work?".
Every scenario is offline, deterministic and bounded: scripted models or a loopback `MockLlmServer`/`wiremock` server, hand-written loopback protocol fakes, temp directories, loopback TCP/Unix sockets only, no credentials, no wall-clock ordering.
The crate is dev-only (`publish = false`) and never a dependency of a library crate.

## API Surface

- `src/lib.rs` — marker module (`crate_name()`); the crate has no runtime API.
- `tests/support/mod.rs` — the cross-scenario helper contract: `BOUND` (10 s per-await deadline), `within(what, future)`, `collect_bounded(stream, max)`. Scenario-specific fixtures stay in `openjiuwen-test-support` or beside the scenario.
- Test binaries (each `tests/*.rs` is its own binary; no `[[test]]` entries), 25 of them:
  `kernel_flow`, `workflow_components`, `context_engine`, `context_tokenizers`, `retrieval_pipeline`, `memory_recall`, `memory_dreaming`, `team_memory`, `deep_agent`, `harness_cli_tools`, `provider_native`, `team_flow`, `team_backend`, `runner_subprocess`, `runner_hosting`, `federation_transports`, `a2a_roundtrip`, `symphony_plan`, `symphony_artifacts`, `self_improvement`, `evaluation_ledger`, `plugin_observability`, `persistent_agent_journal`, `cli_binary`.
- Features: `default = []`;
  `a2a` (enables `openjiuwen/a2a` + `openjiuwen-extensions/a2a`, gates `a2a_roundtrip.rs`);
  `otel` (enables `openjiuwen/otel` + `openjiuwen-extensions/otel`, gates `plugin_observability.rs`);
  `sqlite` (enables `openjiuwen/sqlite` — which forwards the kernel foundation/session/memory backends — plus `openjiuwen-agent-teams/sqlite`; gates the SQLite tests inside `memory_dreaming.rs` and `team_backend.rs`).
- Dependencies are declared under `[dependencies]` so the features can enable features on the same crates: the umbrella `openjiuwen` (module aliases `core`, `harness`, `harness_protocol`, `harness_providers`, `agent_teams`, `extensions`, `symphony`, `rsi`, `agent_evolving`), the concrete harness-layer crates `openjiuwen-harness`, `openjiuwen-harness-protocol`, `openjiuwen-harness-providers`, `openjiuwen-agent-teams`, `openjiuwen-extensions`, `openjiuwen-symphony`, `openjiuwen-rsi`, `openjiuwen-agent-evolving`, plus `openjiuwen-test-support`, `tokio`, `tokio-util`, `futures`, `async-trait`, `anyhow`, `reqwest`, `serde_json`, `tempfile`.
- No kernel crate (`openjiuwen-session`, `-workflow`, `-graph`, `-agent`, `-foundation`, `-runner`, `-context`, `-memory`, `-retrieval`, `-multiagent`, `-common`) is a direct dependency: kernel types are imported through the umbrella as `openjiuwen::core::<module>::…` (e.g. `openjiuwen::core::workflow::{Workflow, WorkflowCard, …}`, `openjiuwen::core::context_engine::{ContextEngine, ContextConfig, …}`, `openjiuwen::core::context_engine::tokenizers::artifacts::…`, `openjiuwen::core::retrieval::{AutoLoader, Bm25Index, …}`, `openjiuwen::core::memory::{LongTermMemory, NewMemory, …}`, `openjiuwen::core::memory::team::{TeamMemoryHooks, MemberMemoryToolkit, …}`, `openjiuwen::core::memory::dreaming::{DreamOrchestrator, …}`, `openjiuwen::core::runner::{Runner, SpawnerRegistry, SpawnBackend, SubprocessSpawner, …}`, `openjiuwen::core::multi_agent::transport::{build_transport, RedisTransport, …}`, `openjiuwen::core::foundation::store::external::redis::codec::{Decoder, RespValue}`), which requires no feature.
- Harness-layer paths used directly: `openjiuwen::agent_teams::{TaskBoard, InMemoryTeamStore, TeamTopic, …}`, `openjiuwen::symphony::object_artifact::{ObjectGraphArtifactStore, ArtifactTransport, GcsConfig, AzureBlobConfig, AzureBlobAuth}`, `openjiuwen::extensions::otel::…` (feature `otel`), `openjiuwen_harness_providers`, `openjiuwen_rsi`, `openjiuwen_agent_evolving`.
- Model/session fixtures come from `openjiuwen_test_support`, never from `tests/support/mod.rs`: `agent_session`, `agent_session_with`, `agent_session_with_checkpointer`, `workflow_session`, `workflow_session_with`, `workflow_session_with_checkpointer`, `memory_checkpointer`, `ScriptedModel`, `MockCompletion`, `MockLlmServer`, `ScriptedResponse`, `TEST_API_KEY`, plus the re-exported `wiremock` (`openjiuwen_test_support::wiremock`).
- `a2a_roundtrip.rs` and `plugin_observability.rs` need `#![allow(missing_docs)]` (+ the usual unwrap/expect allow) **before** `#![cfg(feature = "…")]`: with the feature off the whole crate body (including its `//!` docs) is stripped, so a later `allow` is stripped too and the denied `missing_docs` lint fires on the empty crate root.

## Scenario Matrix

116 test functions across 25 scenario binaries: 115 pass with `--features sqlite,a2a,otel` (106 with default features), and exactly 1 documented `#[ignore]`.

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
| `context_tokenizers.rs` (4) | A hand-written byte-level `tokenizer.json` in a temp dir, selected as the native tokenizer, drives `prepare`/`compress` — native counts (`"abc"` → 3, a 3-message window → 42), no fallback, `Truncated` to 14 tokens | `openjiuwen-context` `TokenCounterKind::Model`, `TokenizerSource::NativeTokenizer`, `load_native_tokenizer`, `ContextEngine::prepare`/`compress` |
| `context_tokenizers.rs` | `TokenizerArtifactManager` downloads once over loopback `wiremock`, validates the declared sha256, and serves the second `resolve`/`resolve_cached` from the cache with exactly one HTTP request | `tokenizers::artifacts::{TokenizerArtifactManager, …}`, `load_native_tokenizer` |
| `context_tokenizers.rs` | A tampered cache file is not an artifact (`tokenizer_sha256_mismatch`), a `.openjiuwen-partial.tmp` sibling is ignored, and `download` refetches; a 404 revision is `NotFound` (`tokenizer_revision_unavailable`) | artifact cache + validation seam |
| `context_tokenizers.rs` | `ArtifactLock` semantics (`LOCK_WAIT_LIMIT` timeout, stale-marker replacement via `STALE_LOCK_AFTER`) and a default manager reporting `remote_download_disabled` with zero requests | `artifacts::{ArtifactLock, STALE_LOCK_AFTER, LOCK_WAIT_LIMIT}` |
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
| `memory_dreaming.rs` (5, 4 by default) | One tick over the in-memory planner with an injected `now` decays both memories to their recomputed salience, keeps them recallable, and reports `ticks == 1` / no error / the job row `Succeeded` | `openjiuwen-memory` `dreaming::{DreamOrchestrator, DreamPlanner, DreamRetention}`, `salience::rank_by_salience`, `InMemoryDreamRepository` |
| `memory_dreaming.rs` | An aged low-salience record is forgotten while a fresh one survives; the salience formula is pinned (`0.26125` vs `0.7125`) and the survivor is still recallable | salience scoring + `rank_by_salience` threshold |
| `memory_dreaming.rs` | A tick with nothing due does no work while the schedule backs off (`30/60/120/240 s`, capped) and a job queued at the same instant is picked up | `DreamOrchestratorStatus`, `interval_for` |
| `memory_dreaming.rs` | The configured retention policy purges finished job rows only (`max_per_state`), idempotently | `DreamRetention`, job-row purge |
| `memory_dreaming.rs` | SQLite durability: consolidation and job rows survive a reopen of both the memory and dream files (`#[cfg(feature = "sqlite")]`) | `LongTermMemory::sqlite`, `dreaming_sqlite`, `SqliteDreamRepository` |
| `team_memory.rs` (4) | Member A's buffered turns flush into a decision record; member B reads it back through the shared facade and a toolkit search | `openjiuwen-memory` `team::{TeamMemoryHooks, MemberMemoryToolkit, TeamMemory}` |
| `team_memory.rs` | Two member toolkits share team-visible records but never `private` ones; `read_only()` refuses writes with `Unsupported` + `error_envelope` | `MemberMemoryContext`, toolkit schemas and envelopes |
| `team_memory.rs` | `TEAM_MEMORY.md` export → parse round trip → import into a fresh facade without duplicates (front matter, section and metadata invariants pinned) | `TeamMemoryDocument` export/parse/import |
| `team_memory.rs` | Cross-crate composition: a completed `TaskBoard` task becomes a flushed `Task`-kind memory record that is recallable | `openjiuwen::agent_teams::{TaskBoard, InMemoryTeamStore}`, `TaskTopic` publications |
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
| `team_backend.rs` (5 with `sqlite`) | Event bus subscription records `member_status_changed`, mailbox and task publications in order, plus the settled run's single `team_completed` publication | `TeamBackend::messager`, `Messager::subscribe`, `TeamTopic`, `EventMessage` |
| `team_backend.rs` | Mailbox send / `unread` / `mark_read` and the task board read back through the backend | `TeamBackend::send_message`, `TaskBoard` |
| `team_backend.rs` | Per-member checkpoint payload round-trips through `HarnessCheckpoint` → `NativeCheckpoint` and validates its session/card | `export_checkpoint`, `NativeCheckpoint::validate_compatible` |
| `team_backend.rs` | SQLite team store (`StorageSpec { kind: "sqlite" }`) persists board rows and mail across store instances | `open_store`, `SqliteTeamStore` (feature `sqlite`) |
| `team_backend.rs` | Team attach over a Unix socket: `list_members`, `snapshot`, `send_member`, `stop_team`, then transport error after teardown | `AttachClient`, attach host started from `spec.metadata["attach"]` |
| `runner_hosting.rs` (5) | Registry routing by spec hint, explicit default, sole spawner and ambiguity; `backends()` order, `set_default` rejecting an unregistered backend | `openjiuwen::core::runner` `SpawnerRegistry`, `SpawnBackend`, `SpawnSpec`, `DeferredSpawner` |
| `runner_hosting.rs` | A `Runner` hosts the registry and spawns through the routed in-process backend; `RemoteMq` and a missing subprocess registration are typed `Unsupported` with no active entry | `Runner::set_spawner_registry`, `spawn_agent_with`, `spawn_backends` |
| `runner_hosting.rs` | `active_spawns` tracks each hosted child (`RunKind::Spawn`, `detail.backend`/`pid`) until its own handle ends; shutdown and drop retain the run with their own detail | `Runner::active_spawns`, monitor retention |
| `runner_hosting.rs` | A hosted spawn reports health (`health`/`is_healthy`/`health_snapshot`, `pid == None` in-process) and an unresolvable external command fails `NotFound` before launch | `SpawnedAgent::health`, external-command resolution |
| `runner_hosting.rs` | A runner hosts a dream driver beside a live spawn: an injected-clock tick runs to completion (job `Skipped`, `status().ticks == 1`) and the spawn stays healthy | `Runner` + `openjiuwen::core::memory::dreaming::DreamOrchestrator` |
| `federation_transports.rs` (4) | Redis Streams P2P round trip over a hand-written RESP server: `XADD envelope/reply`, handler reply, `XACK` clears the pending entry, no `RPUSH`/`RPOP` | `openjiuwen::core::multi_agent::transport::{RedisTransport, build_transport, TransportConfig}`, foundation redis `codec::{Decoder, RespValue}` |
| `federation_transports.rs` | Broadcast and directed deliveries follow a `RoutingTable` snapshot (pattern routes, per-subscriber `XADD` copies) | multiagent routing table + pubsub routing |
| `federation_transports.rs` | `InboxFilter` gates what reaches a member; a rejected P2P surfaces a typed `Transport` error at the sender and rejected pubsub entries are still `XACK`ed | multiagent `InboxFilter` |
| `federation_transports.rs` | Registry write-through failures are typed `External`, an unreachable server fails `redis connect …`, and `RedisTopology::Cluster` is `Unsupported` | transport registry, `RedisTopology` |
| `a2a_roundtrip.rs` (4, feature `a2a`) | Card endpoint over HTTP + `A2aClient::invoke` against a loopback `A2aServer` on an ephemeral port | `openjiuwen-extensions::a2a` (`A2aServer`, `A2aClient`), `reqwest` for the card |
| `a2a_roundtrip.rs` | Streaming send yields the deterministic frame sequence and one terminal frame | A2A SSE framing |
| `a2a_roundtrip.rs` | Paused turn (`INPUT_REQUIRED`) → `cancel_task` → `CANCELED` | A2A task store + cancel verb |
| `a2a_roundtrip.rs` | Non-object inputs rejected with `Error::Validation` before any I/O | `A2aClient` input mapping |
| `symphony_plan.rs` (3) | Register capabilities → `plan` → `compile` to a `CompiledGraph` → execute with a recording runner | `openjiuwen-symphony` (`SymphonyRuntime`, `CapabilityRegistry`, `PlanExecutor`), `openjiuwen-graph` |
| `symphony_plan.rs` | Plan determinism, `NeedsInput`/`NoPlan` outcomes | symphony planner |
| `symphony_plan.rs` | Graph snapshot published to file artifacts and reloaded by a fresh runtime | symphony artifact/asset stores |
| `symphony_artifacts.rs` (4) | GCS object-artifact round trip over a loopback JSON-API emulator: revision-then-pointer uploads, `Authorization: Bearer`, percent-encoded object names, `alt=media` reads | `openjiuwen::symphony::object_artifact::{ObjectGraphArtifactStore, ArtifactTransport::Gcs, GcsConfig}` |
| `symphony_artifacts.rs` | Azure Blob with SharedKey: container create `restype=container`, `x-ms-blob-type: BlockBlob`, `x-ms-version`/`x-ms-date`, `Authorization: SharedKey account:<sig>` | `AzureBlobConfig::with_shared_key`, `AzureBlobAuth` |
| `symphony_artifacts.rs` | Azure Blob with a SAS token: the token travels in the query string and no `Authorization` header is sent | `AzureBlobAuth::sas` |
| `symphony_artifacts.rs` | Rejected requests are typed `External` with the status in the message; malformed revision names are `Validation` before the wire | artifact error surface |
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
| `agent_usage_events.rs` (4) | Two streamed `ReActAgent` turns each emit exactly one `context.usage` frame, mirrored byte-for-byte into the session trace | `openjiuwen::core::single_agent` streaming frames (`CONTEXT_USAGE`), `Session::subscribe` |
| `agent_usage_events.rs` | A usage-carrying model response adds one `llm_usage` frame in the deterministic order `context.usage, llm_output, llm_usage, answer`, and the payload normalizes through `request_usage_from_metadata` | `AgentChunk::llm_usage`, `context_engine::usage::request_usage_from_metadata` |
| `agent_usage_events.rs` | Legacy `cache_tokens` metadata normalizes to read/miss counters (`partial` hit, `derived` quality) and one recorded request yields a `SessionKVCacheUsage` with its aggregation key | `Usage`, `SessionKVCacheUsage`, `CacheAggregationKey` |
| `agent_usage_events.rs` | Two requests in one scope accumulate (and re-recording a turn is idempotent) while per-request reports stay non-cumulative | `SessionKVCacheAggregator` |
| `plugin_observability.rs` (3, feature `otel`) | Team-memory hook events observed by a recording observer are projected onto spans and exported OTLP/HTTP JSON to a loopback collector: path, content type, resource attributes, scope name, span names/order/kind, ids and team attributes | `openjiuwen::core::memory::team` hooks + `extensions::otel::{OtelTraceSink, TracerHandle}`, `wiremock` |
| `plugin_observability.rs` | A 500 collector yields exactly one POST per batch, no retry, and a clean bounded shutdown | OTLP exporter failure surface |
| `plugin_observability.rs` | Two flushes export two ordered batches sharing the session trace id; `OtlpEncoding` fixes the path and content type | `OtlpEncoding`, `traces_url` |
| `persistent_agent_journal.rs` (4) | A completed journal (five entries, one rotation) reopens from disk: segment line counts, metadata encoding/generation, `replay.cursor` (`next_seq`/`last_seq`/`generation`), digest sequence span, per-kind counts and token totals all read back | `openjiuwen::harness::persistent_agent` (`ExecutionLog`, `LogCapacity`, `EntryPayload`, `ModelUsage`, `EntryKind`, `Replay`, `LOG_ENTRY_FILE`/`LOG_META_FILE`/`LOG_ENCODING`), `ManualClock` |
| `persistent_agent_journal.rs` | `resume` over two linked executions rebuilds `cursor` (`resume_seq`/`recovery_ref`/`predecessor_id`/relation), a non-empty window containing the journalled goal, the traced chain, the background document and the record's new continuation without rewriting the stored record | `resume`, `Continuation`, `RootLogSource`, `FileLinkStore`, `FileSessionStore`, `PersistentSessionRecord`/`RecordEnding`, `ExecutionLink`/`LinkRelation`, `WindowAssembler`/`Capacity` |
| `persistent_agent_journal.rs` | A hand-damaged `entries.jsonl` (retried append, conflicting rewrite, torn trailing line) reports `skipped_invalid`/`duplicates_removed`/`conflicting_seqs`, stays usable, keeps the valid entries, and keeps accepting appends past the damage (`missing_ranges` appears once the writer continues) | `ExecutionEntry` serialization + the tolerant read (`LogSnapshot`, `LogDigest::summary`) |
| `persistent_agent_journal.rs` | The remote and distributed handoff transports answer `Error::Unsupported` with their documented `reason()`; the in-process transport has none | `hand_off`, `RemoteHandoff`, `DistributedHandoff`, `HandoffTransfer`, `HandoffTransport` |
| `cli_binary.rs` (5 + 1 ignored) | `run_with_model` one-shot: `text`, `json`, `stream-json` output contracts | `openjiuwen-harness` `cli` (`run_with_model`, `CliIo`, `OutputFormat`) |
| `cli_binary.rs` | Pre-run validation (option resolution, blank prompt) rejects before any model call | `CliOptions::resolve_with`, `Cli::dispatch` |
| `cli_binary.rs` | Process-boundary `--version`/`--help` — `#[ignore]`d, see Known Gaps | `openjiuwen-cli` binary, resolved at runtime |

## Constraints

- Offline and deterministic: `ScriptedModel`/`MockLlmServer`/`wiremock`/loopback TCP and Unix sockets only; no credentials, no `Runner::global()` (always a local `Runner::new()`), no `MemoryCheckpointer::shared()`.
- Protocol fakes are hand-written and loopback-only: the RESP Streams server (`federation_transports.rs`), the GCS JSON-API and Azure Blob emulators (`symphony_artifacts.rs`), and the OTLP collector (`plugin_observability.rs`, `wiremock`). No new crates.io dependency and no import of another crate's test helper — test binaries cannot import each other, so each fake lives in the single scenario that needs it.
- No sleeps for ordering: a scenario waits on the event, frame, or state transition it asserts; timeouts exist only so a hang fails fast (`support::BOUND`).
- Never assert non-deterministic fields: timestamps, `duration_ms`, `plan_id`/`run_id` uuids, `GraphRevision::created_at`, generated memory/chunk/doc ids, or ids you did not read back from the producer.
- Time types are never named directly (`chrono` is not a dependency): read the clock through `openjiuwen::core::common::utils::id::now_utc()`, and build a specific instant through a public helper that returns the API's own timestamp type.
- Cron is tested by time injection, never by waiting: create through the tool, deserialize `payload["data"]["job"]`, then pass `job.next_run_at` to `CronScheduler::due`.
- Dreaming is driven by an injected `now` per tick; retention assertions use the clock-free `max_per_state` cap because the repository stamps completion with the process clock.
- Every test file starts with `#![allow(clippy::unwrap_used, clippy::expect_used)]` (workspace lints are warnings and the gate is `-D warnings`); shared helper modules additionally allow `dead_code`, `unreachable_pub`; a long scenario body may take a local `#[allow(clippy::too_many_lines)]`.
- Each scenario owns its temp directory (`tempfile::TempDir` / `TempWorkspace`) and keeps it alive for the test body; nothing writes outside it.
- Loopback services bind `127.0.0.1:0` and read the address back; Unix-socket paths stay inside the test's `TempDir` and use short prefixes (a long `tempfile` prefix overflows `sun_path`).
- A scenario that needs a sibling API change is recorded under **Known Gaps**, never fixed by editing the sibling crate; the suite adds no new `#[ignore]`d tests.
- `openjiuwen-e2e` is not in the workspace `default-members`: use `cargo test -p openjiuwen-e2e` (plus `--features sqlite,a2a,otel`).
- The `sqlite` feature enables the kernel backends through `openjiuwen/sqlite` (the umbrella) and `openjiuwen-agent-teams/sqlite` directly; the default build stays `sqlx`-free.
- No new crates.io dependency may be added: `wiremock` is reached through `openjiuwen_test_support::wiremock`, `clap` through `openjiuwen_harness::clap`, and time types are never named directly.

## Routing Table

- `./tests/kernel_flow.rs` → model → tool → session → workflow → runner flows (kernel crates)
- `./tests/workflow_components.rs` → `SubWorkflowComponent`, `HttpRequestComponent` (loopback), `QuestionerComponent` HITL resume, runner façade
- `./tests/context_engine.rs` → context window assembly, truncation/compress persist-vs-preview, `ReActAgent` context wiring
- `./tests/context_tokenizers.rs` → native `tokenizer.json` counting through the context engine, artifact manager download/cache/validation/lock
- `./tests/retrieval_pipeline.rs` → DOCX/XLSX/PDF fixtures → loaders → chunking → BM25 → `SparseRetriever` → knowledge base
- `./tests/memory_recall.rs` → remember/recall, keyword vs semantic vs hybrid fallback, retrieval→memory bridge
- `./tests/memory_dreaming.rs` → salience decay, forgetting, scheduling backoff, retention purge, SQLite durability
- `./tests/team_memory.rs` → team-memory hooks flush, member toolkits and scopes, `TEAM_MEMORY.md` export/import, task-board composition
- `./tests/deep_agent.rs` → DeepAgent task loop, workspace tools, permissions, interrupt/resume
- `./tests/harness_cli_tools.rs` → `fetch_webpage` against a loopback server, cron scheduling, `--no-color`, slash commands
- `./tests/provider_native.rs` → provider-neutral harness SPI: factory, native harness lifecycle, IO adapter
- `./tests/team_flow.rs` → agent teams: spec build, leader/teammate runtime, task board (umbrella `openjiuwen::agent_teams` paths only, scripted `MemberHarnessFactory` + `InMemoryTeamStore`)
- `./tests/team_backend.rs` → team event bus, mailbox, per-member checkpoint payloads, SQLite store (feature `sqlite`), Unix-socket attach
- `./tests/runner_hosting.rs` → `SpawnerRegistry` routing, runner-hosted spawns, `active_spawns`, health, a dream driver beside a spawn
- `./tests/federation_transports.rs` → Redis Streams transport over a hand-written RESP server, routing table, inbox filters, registry failures
- `./tests/a2a_roundtrip.rs` → A2A client/server round trip (feature `a2a`)
- `./tests/symphony_plan.rs` → symphony capability registry, planning, graph execution, artifact reload
- `./tests/symphony_artifacts.rs` → GCS and Azure Blob object-artifact transports over loopback emulators (auth headers, layouts, typed errors)
- `./tests/self_improvement.rs` → RSI orchestrator and agent-evolving trainer with fake runners
- `./tests/evaluation_ledger.rs` → evaluator pipeline with a fake command runner, RSI ledger, progress events, event stream
- `./tests/runner_subprocess.rs` → runner subprocess spawn seam: self-executing NDJSON child, request/reply, health, graceful shutdown, typed transport/validation errors
- `./tests/agent_usage_events.rs` → `context.usage` / `llm_usage` frames, provider-usage normalization, session cache aggregation
- `./tests/plugin_observability.rs` → team-memory events projected to OTLP/HTTP spans against a loopback collector (feature `otel`)
- `./tests/persistent_agent_journal.rs` → persistent-agent journal durability: crash/reopen replay, `resume` cursor/window/chain, damaged-journal degradation, typed deferred handoffs
- `./tests/cli_binary.rs` → harness CLI layer (one-shot output formats, pre-run validation, process boundary)
- `./tests/support/mod.rs` → cross-scenario deadline and stream helpers (`BOUND`, `within`, `collect_bounded`)
- `./Cargo.toml` → manifest, test-only dependency set, `a2a`/`otel`/`sqlite` features

## Test Strategy

- One binary per scenario area so a failure names the seam and the suite parallelises safely (each binary has its own temp dirs, its own model script, its own loopback port, its own Unix socket).
- Assert on recorded evidence, not timing: `ScriptedModel::calls()`/`call_count()`/`remaining()`, harness event sequences, collected frames, task board and mailbox state, store and file contents, `MockLlmServer::request_count`, `FakeCommandRunner::invocations()`, ledger `totals()`/`events()`, the fake servers' recorded commands and requests, exported OTLP payloads.
- Fake only what a scenario cannot own: harness protocols, member factories, symphony capability runners, RSI harness runners, evolving targets, Docker (`FakeCommandRunner`), the subprocess child (the test binary itself), the Redis server, the object-storage APIs, the OTLP collector, and the dream consolidator (a per-scenario salience consolidator). Each fake is copied into the scenario that needs it, keeping every binary's contract explicit.
- Fixtures that the library keeps private are rebuilt from public primitives inside the scenario (`build_zip` for the DOCX/XLSX fixtures uses `parsing::crc32`; the tokenizer fixture is a hand-written byte-level `tokenizer.json`), never by widening library visibility.
- Gates: `cargo fmt -p openjiuwen-e2e -- --check`; `cargo test -p openjiuwen-e2e --offline`; `--features sqlite,a2a,otel`; `cargo clippy -p openjiuwen-e2e --all-targets [--all-features] --offline -- -D warnings`. Default features run 106 scenarios + 1 ignored, the feature set runs 115 + 1 ignored, and the whole suite finishes in seconds of test time (the subprocess, attach, federation and artifact scenarios included) and is stable across repeated runs.

## Design Decisions

- `runner_subprocess.rs` re-executes **its own test binary** as the fake agent child instead of shipping a second binary: `std::env::current_exe()` plus the libtest filter `fake_agent_child --exact --test-threads=1 --nocapture`.
- The child entry point is a normal `#[test]` that returns immediately unless `E2E_FAKE_AGENT_CHILD` is set (the parent sets it), so it is never `#[ignore]`d and the default suite runs it as a silent pass; `--nocapture` is mandatory or libtest swallows the protocol frames.
- `E2E_FAKE_AGENT_MODE` selects `echo`/`error_frame`/`bad_frame`; the background health task is disabled (`with_health_check(Duration::ZERO, …)`) so no probe races an exchange.
- `federation_transports.rs` is deliberately one file: the fake RESP server and the scenarios share one state machine (entries, group cursors, pending lists, SET registry), so splitting them would duplicate the server or force it into `tests/support/` where every other suite would compile it unused.
- `symphony_artifacts.rs` mounts both emulators (GCS JSON API, Blob REST) behind one `wiremock` catch-all responder with a shared `BTreeMap` store, so the object layout and the auth headers are asserted against the real transport code rather than a mock of it.
- `memory_dreaming.rs` owns a small salience consolidator and re-wraps the facade's planner queue (`DreamPlanner::new(Arc::clone(planner.repository()), …)`) because the facade hard-codes `NoopConsolidator`.
- `plugin_observability.rs` owns the event→span projection because no team-memory→OTel adapter exists in the workspace; the sink, exporter and wire format are the production ones.
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
- `TokenCounterKind::Model` + a local `tokenizer.json` selects `TokenizerSource::NativeTokenizer` and reports `estimated() == false`; the byte-level fixture counts one token per UTF-8 byte plus four framing tokens per message.
- The artifact manager validates a cached file only against a declared `sha256`; a mismatching cache entry reports `tokenizer_sha256_mismatch` and `download` refetches, a `.openjiuwen-partial.tmp` sibling is never an artifact, and a manager with downloads disabled reports `remote_download_disabled` with no request.
- Memory: `SearchMode::Semantic` without an embedder is `Error::Unsupported`; `Hybrid` without one degrades to keyword-only with `similarity == 0.0` (a sentinel, not `None`); the default threshold is `0.3`; a blended-recall score is positive even without query overlap, so assertions are containment checks.
- Dreaming: salience is `(importance + recency + usage + base) / 4` (an aged `0.02` record scores `0.26125`, a fresh `0.95` one `0.7125`); an in-place importance rewrite is reported as `records_merged` and a forgotten record as `records_deleted`; an empty tick backs the interval off `30 → 60 → 120 → 240 s` (capped) and a job queued at the same instant is still picked up; `DreamRetention::new(0).with_max_per_state(n)` purges settled job rows and is idempotent.
- Team memory: `on_member_join` writes a `MemberProfile` tagged `member,presence`, a decision turn flushes a `Decision` tagged `decision`, a handoff a `HandoffNote` tagged `lesson`, and hook record ids are prefixed `hook-`; toolkit schema ids are `team_memory.<team>.<member>.<tool>`; a `read_only()` context answers `Unsupported` for writes with the `{ok:false,…}` envelope; `TEAM_MEMORY.md` carries `team_memory: openjiuwen-team-memory/v1` front matter and section-per-kind headings, and import is an upsert (no duplicates).
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
- Spawner registry: `backends()` follows enum order `ExternalCommand, InProcess, RemoteMq`; a spec hint outranks the default, and an unregistered `set_default` is `Validation`; an unhinted spec on an ambiguous registry is `Unknown` ("ambiguous spawn backend"), on an empty one "no default spawn backend".
- Runner-hosted spawns: `active_spawns` detail is `{backend, pid}` (`pid` is null in-process), a graceful `shutdown` retains the run with `detail.graceful == true`, and a dropped handle retains it with `detail.reason == "handle_dropped"`; an unresolvable external command is `NotFound` before any launch.
- An in-process dream job whose consolidator is the facade's `NoopConsolidator` finishes `Skipped` with the note "no consolidator configured".
- Usage: one `context.usage` frame per streamed turn (mirrored into the session trace), a usage-carrying response adds an `llm_usage` frame, and legacy `cache_tokens` metadata normalizes to a cache-read count with `partial` hit rate and `derived` data quality.
- Redis federation: the P2P queue is `<team>:<team>:p2p:<agent>` and the consumer group `<team>:<team>:p2p:cg`; a request is one `XADD` with `envelope`/`reply` fields, the reply stream is `DEL`eted after consumption, and rejected inbox entries are still `XACK`ed so they are never redelivered.
- Symphony object artifacts: `publish` probes the revision key before creating the bucket/container and writes the revision object before the pointer; GCS object names are one percent-encoded segment (`alt=media` reads), Azure blob names keep `/`, `versions()` re-reads each revision so object-GET counts are not one per published version.

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
- `SpawnedAgent` (the trait seam) does expose `health()`/`is_healthy()`/`health_snapshot()` with defaults, which is what `runner_hosting.rs` drives; the concrete `SpawnedProcess` adds `is_alive()`.
- **`openjiuwen-agent-teams` has no team-memory integration point**: no `TeamMemory`/`TeamMemoryHooks` handle on `TeamAgent`/`TeamBuildDeps`/`TeamBackend`, no session/member lifecycle bridge (`start`/`spawn_member`/`settle` never call the hooks), no task-board → hook turn/note bridge (`TeamMemoryHooks` has no public way to buffer a task note at all — `HookState::task_notes` is only ever written empty), and a host must build `MemberMemoryContext` + toolkit itself. `team_memory.rs` therefore composes the two crates at the caller level (`TaskBoard` events → `TeamTurn` → hooks flush); a runtime-owned bridge would close the gap.
- **No runner-owned dream/background driver**: `openjiuwen-runner` has no dream entry point, so `runner_hosting.rs` composes the tick on a spawned task beside a live spawn.
- **No team-memory → OTel span bridge**: `openjiuwen-memory`'s team observer seam hands events to the caller and nothing in `openjiuwen-extensions` subscribes, so `plugin_observability.rs` owns the observer and the event→span projection; `InvokeType` has no `Memory` variant, so the spans are named `chain.team_memory.<op>`.
- The OTLP export failure is not reachable as a typed error: `SpanExporter::export` returns an SDK error type that `openjiuwen-extensions` does not re-export and `TracerHandle::{force_flush, shutdown}` log and return `()`, so the failure scenario asserts the outward contract (one POST per batch, no retry, pipeline still usable) instead.
- `TokenizerArtifactManager::{lock_path, cache_key}` are private, so the stale-lock scenario drives the public `ArtifactLock`/`STALE_LOCK_AFTER` seam; cached-artifact validation is digest-only (a truncated `tokenizer.json` with no declared sha256 is returned and only fails later in `load_native_tokenizer`).
- No nameable clock type without `chrono`: `memory_dreaming.rs` derives its injected instants through the public `external::rest_json::parse_timestamp` helper (and reads "now" through `common::utils::id::now_utc()`); a `from_unix_secs`/timestamp alias on the memory or common surface would remove the indirection.
- `DreamRepository::{complete, fail}` take no `now` and stamp the process clock, so an age-based `DreamRetention` purge is not assertable deterministically and the retention scenario asserts the clock-free `max_per_state` cap; the dreaming facade also hard-codes `NoopConsolidator` and the metric vocabulary has no "updated" counter.
- `build_transport` returns `Arc<dyn TeamTransport>`, while `watch_agent` and the key accessors are inherent to `RedisTransport`, so a factory-built transport can publish but cannot register an inbox worker — `federation_transports.rs` builds the receiver with `RedisTransport::new` and only the publisher through the factory.
- `RoutingTable`'s only assembler (`routing::assemble_routing_table`) is `pub(crate)`; its fields are public, so the scenario builds a literal, and binding a Redis transport to a `TeamRuntime` is not reachable (`Arc::new_cyclic` creates the handler internally).
- `federation_transports.rs` never drives `TeamRuntime`: no `set_transport`, runtime `start`/`stop`/`send`/`publish`/`subscribe`, `routing_table()` or `set_inbox_filter` call — the routing table is a hand-built literal and the `InboxFilter` is applied by the scenario's own `MessageHandler`, so the runtime-owned dispatch path stays out of this suite.
- A Redis connect failure surfaces as `ErrorKind::External` (the foundation client wraps the I/O error), not `Transport`; the scenario accepts either and asserts the `redis connect …` text.
- Vendor harness providers (`claudecode`, `codex`, `dsh`) spawn real vendor CLIs behind the `vendor` feature, so they need those binaries and credentials; `provider_native.rs` asserts the typed binary-resolution error instead.
- Deferred ML/LLM-backed areas (symphony `deferred::*`, `evaluation::LlmJudgedMetric`; agent-evolving RL/judge stubs; RSI `deferred::*`) return `Error::Unsupported` and are not exercised.

## Reference Patterns

Copy the shape of the sibling suites instead of inventing fixtures: `../openjiuwen-workflow/tests/{sub_workflow,http_component,questioner_component}.rs`, `../openjiuwen-context/tests/{engine,usage,engine_usage,tokenizers_native,tokenizers_artifacts}.rs`, `../openjiuwen-retrieval/tests/{loaders,sparse}.rs`, `../openjiuwen-memory/tests/{recall,facade_smoke,analyzer_wiring,dreaming_orchestrator,team_memory,team_toolkit}.rs`, `../openjiuwen-agent/tests/react_agent.rs` (tool round), `../openjiuwen-workflow/tests/linear.rs` + `references.rs` (`${a.b}` flows), `../openjiuwen-runner/tests/{run_agent,spawn_subprocess,spawn_routing}.rs`, `../openjiuwen-multiagent/tests/{transport_redis,api_surface,resources}.rs` (the full fake RESP server lives there), `../openjiuwen-harness/tests/{deep_agent_loop,permissions,cli}.rs` and `../openjiuwen-harness/src/tools/web/tests.rs` + `src/tools/cron/tests.rs`, `../openjiuwen-harness-providers/tests/{native,io_adapter}.rs`, `../openjiuwen-agent-teams/tests/{team_run,task_flow,attach_e2e}.rs` and `../openjiuwen-agent-teams/tests/full_team_sqlite.rs`, `../openjiuwen-symphony/tests/{executor,runtime,artifact_gcs,artifact_azure,object_artifact}.rs`, `../openjiuwen-rsi/tests/{orchestrator,checkpoint,ledger,progress}.rs`, `../openjiuwen-agent-evolving/tests/{pipeline,evaluator,trainer}.rs`, `../openjiuwen-extensions/tests/{a2a_*,otel_otlp,otel_seam,otel_integrations}.rs`, `../openjiuwen-cli/tests/cli_smoke.rs`.
