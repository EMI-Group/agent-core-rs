# Parity Matrix

Reference: openJiuwen Python SDK v0.1.20 (read-only foreign repo `/home/bill/Source/agent-core`).
The parity target is the Python package's public surface: module paths, wire formats, persisted JSON, tool names/schemas, permission outcomes, and observable loop behavior.
Rust internals are idiomatic (tokio, serde, traits, thiserror) rather than a transliteration, so internals are not compared field by field.

## Status Legend

- **Implemented** — the targeted Python public surface is ported, exported, and covered by tests.
- **Subset** — a coherent core is ported and usable while a named remainder is deferred (see §Deferred Features).
- **Stub** — typed entry points exist and return `Error::Unsupported`; it applies to areas inside a crate, never to a crate as a whole.
- **Deferred** — not ported at all, with no Rust crate and no empty seam in its place; like `Stub` it applies only to areas.

## Package Matrix

| Python path | Rust crate | Status | Notes |
|---|---|---|---|
| `core.common` (schema, exception, logging, clients, task_manager, security, utils, constants, run_context) | `openjiuwen-common` | Implemented | `Error`/`ErrorKind`, `schemars` cards, `tracing` logging, env resolvers, `SecretString`, async HTTP client with retry, task manager; connector pools, AES-GCM crypto, SSRF checks and log rotation are unported. |
| `core.foundation.prompt` | `openjiuwen-foundation` | Implemented | `Role`, one `BaseMessage` struct, `MessageContent`/`ContentPart`, tool-call types and accumulator, `PromptTemplate` with assembler and YAML loading, three token counters. |
| `core.foundation.llm` | `openjiuwen-foundation` | Subset | OpenAI-compatible and Anthropic transports, reasoning profiles, IntelliRouter routing/failover, OpenAI OAuth and DashScope media; the OpenAI Responses transport, `output_parsers` and TPM/RPM enforcement are unported. |
| `core.foundation.tool` | `openjiuwen-foundation` | Implemented | `Tool` trait with `stream`/`render_for_llm`, flat `ToolCard`, `FunctionTool`/builder/`ToolOutput`, `RestfulApiTool`, `#[tool]` macro; `ClientTool` lives in `openjiuwen-agent`, the auth-strategy registry is unported. |
| `core.foundation.tool.mcp` | `openjiuwen-foundation` (feature `mcp`, default) | Subset | stdio, SSE, streamable-HTTP and OpenAPI transports over `rmcp`/`reqwest`, server GET stream, resources, reconnect, session replay; the Playwright transport and legacy-SSE `Last-Event-ID` replay are missing. |
| `core.foundation.store` | `openjiuwen-foundation` (feature `sqlite`) | Subset | kv/document/vector/graph/object traits with in-memory, local-file, SQLite and network backends behind `StoreRegistry`; pgvector/postgres/gauss, external graph stores and Redis TLS are missing. |
| `core.session` | `openjiuwen-session` (feature `sqlite`) | Implemented | Four execution sessions plus `SubWorkflowSession`/`RouterSession`, scoped state, HITL, memory/file/SQLite/Redis checkpointers, `DiskSessionStore`, `SessionHistory`; pickles, `rediss://` TLS and git VCS are missing. |
| `core.graph` | `openjiuwen-graph` | Implemented | Pregel/BSP supersteps, CNF barrier channels, conditional routing, halt, retries, timeouts, cancellation, JSON snapshots, Mermaid output; pickle internals, `stream_actor`, DOT/PNG/SVG and any-of/count triggers are missing. |
| `core.workflow` | `openjiuwen-workflow` | Subset | Builder, `${node.field}` resolution, 14 built-ins, streaming, checkpointing, first-arrival fan-in, `default_outputs`, `enable_history`, `response_format`, SVG draw; live producer interleaving, PNG, per-run mutation and sub-workflow HITL resume remain. |
| `core.context_engine` | `openjiuwen-context` | Subset | Truncation/tool-result/summary processors, five compressor and four offloader families, recall archives, memory reinjection, reload hint, tokenizer registry, per-processor checkpoints; native tokenizer artifacts, usage telemetry and adaptive passes remain. |
| `core.memory` | `openjiuwen-memory` (feature `sqlite`) | Subset | Scoped CRUD, hybrid recall, deterministic dedupe, entity/graph memory, rule-first analyzer with LLM fallback, provider SPI, SQLite migrations, lite Markdown+FTS5+lock+watcher, dreaming, team memory; vendor SDKs, AES/distributed lock and the Milvus graph stack remain. |
| `core.retrieval` | `openjiuwen-retrieval` | Subset | Loaders including DOCX/XLSX/PDF, chunkers, embeddings, in-memory store, BM25, dense/sparse/hybrid/graph/agentic retrievers, query rewriting, triple extraction, multimodal documents; external vector DBs, LLM captioner/extractor, OCR and encrypted PDFs remain. |
| `core.sys_operation` | `openjiuwen-sysop` (feature `sandbox`) | Subset | Local fs/shell/code with path policy, allowlists, cross-process `flock`, PTY/`stdbuf`, process-group kill, CJK decoding and sandbox launcher lifecycle; only remote sandbox transports and the streaming wire return `Unsupported`. |
| `core.security` | `openjiuwen-security` | Subset | Ordered pipeline, critical-only blocking, prompt-injection/PII/content-rule/tool-argument detectors, fail-open/closed policy, HTTP moderation transport, OpenAI/Azure/Aliyun adapters, injected LLM judge; local inference and further vendors remain. |
| `core.single_agent` | `openjiuwen-agent` | Implemented | `Agent`/`BaseAgent`/`ReActAgent`, sub-agent/workflow/MCP abilities, exposure, callbacks, steering, client tools, resilience, final-request check, multimodal results, evolve, YAML prompt packs; controller background scheduling is deferred. |
| `core.application` | `openjiuwen-agent` | Implemented | `LLMAgent` composes `ReActAgent` via `LlmAgent::new`/`set_prompt_template`; `WorkflowAgent` runs or selects workflows with `default_response`; both are exposed through `openjiuwen_core::application`. |
| `core.multi_agent` | `openjiuwen-multiagent` | Subset | `Team`/`BaseTeam`, cards/configs, in-process bus, handoff and both hierarchical strategies, `AgentTool`, typed events, auto-subscription, Redis transport, member checkpoints; Pulsar and drunner/`RemoteAgent` remain deferred. |
| `core.runner` | `openjiuwen-runner` | Subset | `Runner` plus `ResourceMgr` with five managers, callback filters/chains/decorators, in-memory queue, subprocess/HTTP-remote spawners, team pool, HITL hub, monitor; broker transports and distributed mode remain deferred. |
| `core.controller` | `openjiuwen-agent::controller` | Subset | `BaseController` event→intent→task loop, schemas, `ControllerAgent`/`LLMController`/`WorkflowController`, intent recognition and legacy `IntentDetectionController`; background scheduling, task-manager indexes and executor pause/cancel remain deferred. |
| `core.kv_cache` | `openjiuwen-agent::kv_cache` | Subset | `KVCacheIdentity`/entry/lookup/stats, `InProcessKVCacheRuntime` (LRU, session eviction, prefix hashing), affinity config and model-call hook; Redis/distributed backends and provider prefetch/offload remain deferred. |
| `core.operator` | `openjiuwen-agent::operator` (`::evolve`) | Subset | `Operator`/`PreviewableOperator` plus LLM, tool, memory, skill and context-evolve operators, and `ReActAgentEvolve` in `::evolve`; the legacy `LLMCall` wrapper and ML-backed Metis learner remain deferred. |
| `core` umbrella | `openjiuwen-core` | Implemented | Pure re-export facade with Python-shaped module paths for every kernel crate plus `single_agent`, `application`, `controller`, `kv_cache`, `operator`, `multi_agent` and `runner`. |
| `harness` (DeepAgent) | `openjiuwen-harness` (feature `native` default; `browser`/`cua`/`mobile`/`notify`) | Subset | DeepAgent/loop, workspace, permissions, 32 tools (lsp, worktree, web, cron, MCP resources, goal, discovery), rails, subagents with quota/LRU/transcripts, spec/plugins, native protocol, CLI; browser/cua/mobile and the OTel exporter remain. |
| `harness.cli` | `openjiuwen-harness` (`cli`) + `openjiuwen-cli` binary | Subset | `run`/`chat`, three output formats, slash commands, skill slash, `!cmd`, history with replay, opt-in ANSI colour, instruction files, `mcp.json` merge, allow-rule write-back, onboarding; auto-harness subcommands and `--remote` remain. |
| `harness_protocol` | `openjiuwen-harness-protocol` | Implemented | Provider-neutral SPI traits, events/results/streams, checkpoints, hooks, interactions, tool gateway, model control and the `"1.0"` wire codec; zero workspace dependencies. |
| `harness_providers` | `openjiuwen-harness-providers` (feature `native` default; `vendor`) | Subset | `native` provider, registry/factory, descriptors, `build_harness_context`, `HarnessIOAdapter`, skill-bundle installer; `native_v2` (hosted by `agent_teams`) and `claudecode`/`codex`/`dsh` stay declared stubs returning `Error::Unsupported`. |
| `agent_teams` | `openjiuwen-agent-teams` (feature `mcp` default; `sqlite`/`otel`/`distributed` optional) | Subset | Team runtime, SQLite store, process/external-CLI/SSH members, git-worktree workspaces, org layer, model routing, KV affinity, monitor stream and Unix-socket attach; pyzmq/Pulsar, PostgreSQL/MySQL, remote workspace dispatch and OTLP export are deferred. |
| `extensions.a2a` | `openjiuwen-extensions` (feature `a2a`) | Subset | A2A client/server over JSON-RPC and SSE plus the REST `HTTP+JSON` binding, agent-card adapter and harness bridge; only the gRPC binding is deferred. |
| `extensions.tracer_otel` | `openjiuwen-extensions` (feature `otel`) | Subset | SDK provider, `TraceSink`, agent/workflow handlers, OTLP/HTTP JSON exporter, Grafana Cloud and Langfuse presets; OTLP/gRPC, `http/protobuf` and the local Grafana Agent push are deferred. |
| `extensions.checkpointer`/`store`/`message_queue`/`sys_operation`/`context_evolver`/`vendor_specific` | `openjiuwen-extensions` (features `sqlite`/`redis`/`pulsar`) | Subset | Factories over kernel seams with live Redis KV+checkpointer, Elasticsearch vectors, AIO/JiuwenBox/YuanRong sandboxes, DashScope rerank and deterministic+LLM evolvers; Pulsar, Redis TLS/cluster and Postgres/Gauss document stores are deferred. |
| `dev_tools` | `openjiuwen-dev-tools` | Subset | `judge`, `prompt_builder`, `skill_creator`/`omni`, staged `agent_builder` with Mermaid, `skill_evaluator` and `EvalHarness`; zip/gzip unpacking, browser rendering, vision judging and pipeline extras are deferred. |
| `dev_tools.tune` | `openjiuwen-dev-tools::tune` | Implemented | `Case`/`CaseLoader`, the evaluator seam, the four real searches (`Instruction`/`Example`/`Joint`/`PromptSearch`), `Trainer` and `optimize_prompt`; dataset samplers and `chat_agent` adapters are not ported. |
| `symphony` | `openjiuwen-symphony` (feature `sqlite`) | Subset | Cards, registry, BM25+semantic search, `can_feed` graph, artifact/object stores, `SKILL.md` catalog, experience bank, flow distillation/SkillPack, planner/executor, fallback-backed LLM stages; faiss, vLLM logits and GCS/Azure transports are deferred. |
| `rsi` | `openjiuwen-rsi` | Subset | Auto-harness and single-harness loops, SWE-bench/terminal-bench runtimes, member/paper optimizers, PUCT `program_opt`, `runtime` command/git/CI/pytest, JSONL `RunLedger`; agentdescent, guarded judge and `sys.modules` loader are deferred. |
| `agent_evolving` | `openjiuwen-agent-evolving` | Subset | Datasets, evaluation, LLM-as-judge, grid/research optimizers, trainer/checkpoints, rails, portable RL loop, signal/reward/attribution/trajectory/experience/sharing/TTSE, Docker+git pipeline; GPU RL engines and the online Redis/vLLM state machine are deferred. |
| `auto_harness` | `openjiuwen-auto-harness` | Implemented | Pure re-export facade over `openjiuwen_rsi::auto_harness` (same types by `TypeId`); it adds no behaviour and inherits the `openjiuwen-rsi` deferrals unchanged. |
| `openjiuwen` umbrella | `openjiuwen` (feature `native` default; `otel`, `a2a`, `sqlite`, `live-tests`) | Implemented | Eleven module aliases (including `auto_harness`) plus a ~40-item `prelude`; `core::{controller, kv_cache, operator}` and every nested crate tree are reachable, and `harness_providers` is ungated. |
| `openjiuwen` console script | `openjiuwen-cli` | Implemented | Binary `openjiuwen`: `.env` loading, stderr logging, clap parse, `run`/`chat` dispatch and exit codes 0/1/2. |
| repository `examples/` | `openjiuwen-examples` | Implemented | 12 explicit `[[example]]` binaries (`autoexamples = false`), numbered `01_hello_agent` … `12_self_improvement`; example 11 requires `a2a`, and the 15 `src/support/` unit tests are the only tests. |
| repository `tests/` (cross-package) | `openjiuwen-e2e` | Implemented | 35 cross-crate scenarios over 8 binaries; 30 run with default features, 34 with `--features a2a` because `a2a_roundtrip` is gated, and 1 is `#[ignore]`d; every scenario is offline with a 10 s `BOUND`. |
| — (Rust-only) | `openjiuwen-macros` | Implemented | `#[tool]` generates a `FunctionTool` wrapper plus `{fn}_tool()` with doc-comment descriptions and trybuild coverage; `Args:`-block descriptions, flat keyword parameters and sync/generator handlers have no equivalent. |
| — (Rust-only) | `openjiuwen-test-support` | Implemented | Dev-only (`publish = false`): loopback axum `MockLlmServer`, `ScriptedModel`, workspace/session/checkpointer fixtures, JSON assertion and `eventually` helpers, `wiremock` re-export; 149 tests. |

No row is `Stub` or `Deferred`: 17 rows are `Implemented` and 25 are `Subset`; `Stub` survives only inside crates (`symphony` faiss/vLLM stubs, `rsi::deferred`, the agent-evolving RL optimizers, and the vendor harness providers).

## Wire/Behavior Compatibility

### Models and streaming

- OpenAI-compatible vendors (`openai`, `pangu`, `qwen`, `deepseek`, and aliases) all resolve to one chat-completions client selected by `ModelClientConfig::provider`/`api_base`.
- `anthropic` is the distinct Messages transport; the three automatic `cache_control` anchors and hoisted-or-inline mid-conversation `system` turns are implemented, while per-message caller breakpoints and selectable caching modes are not.
- Requests preserve the OpenAI wire format for messages, tool calls, `tool_choice`, `temperature`, `top_p`, `max_tokens`, `stop`, and streaming `stream=true`.
- Responses expose `ChatResponse`/`Choice`/`Usage` and a `StreamChunk` stream (`Delta`, `Usage`, `Done`); `StreamChunk::Delta` additionally carries `finish_reason` (additive field).
- Parallel tool calls and argument deltas are accumulated by `ToolCallAccumulator`; a mid-stream provider failure surfaces as the stream's `Err` item.
- Retries and timeouts live in the shared HTTP client; a non-2xx response is returned normally unless the caller requests `error_for_status()`.

### Cards, tools, and schemas

- Every `*Card` is a serde struct with Python field names and `#[serde(default)]`; unknown fields are tolerated and `None` fields are skipped on serialize.
- Cards implement `openjiuwen_common::card::Card` and derive `schemars::JsonSchema`, so JSON Schema is generated rather than hand-written.
- `ToolCard` is flat (`id`, `name`, `description`, `input_params`, policy flags); `input_params` is the JSON Schema the model sees, generated from typed handlers where available.
- `FunctionTool` handlers are async-only; Python sync handlers and generator-based tool streams are not reproduced (a blocking handler must use `spawn_blocking`).
- MCP tool names and JSON schemas are preserved through `McpTool`; streamable HTTP replays `Mcp-Session-Id` and holds the server-initiated GET stream, and resource listing/reading is implemented for transports that support it.

### Sessions and checkpoints

- State is JSON, keyed by `session_id`, with a global lane and component-namespaced lanes; dumps keep Python field names.
- `AgentSession` is the in-memory session implementation; there is no separate `InMemorySession` type in the Rust API.
- Checkpointing is the `Checkpointer` trait: in-memory, filesystem/JSON, SQLite (`sqlite` feature), and a Redis RESP2 checkpointer that is always compiled; disk-backed product sessions are `DiskSessionStore` records with local `SessionHistory` versions.
- Python pickle checkpoint payloads cannot be read, and a resumed trace continues from the checkpointed baseline.
- Human-in-the-loop pauses are `AgentOutput { result_type: Interrupt, ... }` values plus persisted `InterruptionState`, and resume is a normal invoke with an `InteractiveInput` in the query slot.
- Sessions write streaming frames as `StreamItem` values; agent `AgentChunk` frames map onto them and remain the public streaming contract.

### Workflow data flow

- `${node.field}` references are resolved in the engine against workflow state; a missing path resolves to `null` instead of failing, and a non-numeric field segment applied to an array of objects projects over its elements (a documented Rust extension; Python returns `null`).
- `outputs_schema` resolves against workflow state plus the node id and an `output` key (`${<id>.field}` / `${output.field}`); there is no `self` shorthand or bare `${field}`.
- `wait_for_all=false` with at least two inbound sources compiles to first-arrival dynamic edges (the node re-fires and unproduced keys resolve to `null`), while CNF barriers remain for single-source nodes, declared conditional edges only, and `Collect`/`Transform` nodes.
- Builder methods never return `Err`; problems are recorded and reported by `build`/`invoke`/`stream`/`to_mermaid`.

### Environment variables

- `OPENJIUWEN_HOME`, `OPENJIUWEN_API_KEY`, `OPENJIUWEN_API_BASE`, `OPENJIUWEN_SERVER_URL`, `OPENJIUWEN_LOG_LEVEL`, `OPENJIUWEN_LOG_FORMAT`, `OPENJIUWEN_BASH_STRICT`, `OPENJIUWEN_MCP_RECONNECT_ATTEMPTS`, `OPENJIUWEN_MCP_SERVER_CMD`, `OPENJIUWEN_OTEL_GLOBAL_INSTRUMENT_ENABLE`.
- CLI/harness variables: `OPENJIUWEN_HISTORY`, `OPENJIUWEN_HISTORY_REPLAY`, `OPENJIUWEN_MCP_CONFIG`, `OPENJIUWEN_NO_ONBOARDING`, `NO_COLOR`.
- Agent/CLI variables: `OPENJIUWEN_MODEL`, `OPENJIUWEN_MODEL_NAME`, `OPENJIUWEN_MODEL_PROVIDER`, `OPENJIUWEN_PROVIDER`, `OPENJIUWEN_WORKSPACE`, `OPENJIUWEN_MAX_ITERATIONS`, `OPENJIUWEN_MAX_TOKENS`, `OPENJIUWEN_SUBAGENT_TASK_TIMEOUT_S`, `OPENJIUWEN_TEAM_JOIN`, `OPENJIUWEN_LIVE`.
- OpenAI/legacy variables: `API_KEY`, `API_BASE`, `MODEL_PROVIDER`, `MODEL_NAME`, `OPENAI_API_KEY`, `OPENAI_BASE_URL`.
- `.env` loading is opt-in at binary/example entry points only (`dotenvy`), never in library constructors.

### CLI commands

- `openjiuwen run [PROMPT] [-f text|json|stream-json]` runs one prompt; the prompt is read from stdin when omitted or `-`.
- `openjiuwen chat` runs the interactive REPL with `/new`, `/clear`, `/help`, `/status`, `/cost`, `/compact`, `/sessions` and `/mode`, plus `!cmd` shell passthrough and skill slash commands; no subcommand chooses `chat` on a TTY and `run` otherwise.
- Global options: `-m/--model`, `--provider`, `--api-key`, `--api-base`, `-w/--workspace`, `--no-color`, `-v/--verbose`.
- Exit codes: `0` success and `--help`/`--version`, `2` clap usage errors, `1` runtime failure printed as `error: <cause>` on stderr.
- `team-member` (skill CLI) and `openjiuwen-team-mcp` (stdio MCP server, `mcp` feature) ship from `openjiuwen-agent-teams`.

### Documented deviations

- `AgentSession` replaces Python's `InMemorySession`; `create_agent_session`/`AgentSession::with_options` are the constructors.
- `LlmAgent::new(LlmAgentConfig)` plus `set_prompt_template` are the configuration entry points where Python exposes `LLMAgent.configure`.
- A denied tool call is not an error: the observation starts with `[PERMISSION_DENIED]` (`DENIED_PREFIX`), `result_type` stays `Answer`, and the loop continues.
- CLI JSON shapes are `{answer, result_type, usage, session_id}` and a `stream-json` line per chunk plus a terminal `{"type":"result", ...}` line; Python emits a different `{result, chunks, model}` object and no terminal line.
- The CLI prints plain text by default and enables ANSI colour only on a TTY without `--no-color`/`NO_COLOR`; a persistent `~/.openjiuwen_history` with optional replay replaces Python's history handling, while the rich/prompt_toolkit renderer and its completion remain unported.
- `TeamAgentSpec`'s `SpecBuild` is `#[async_trait]` and `build_with(&self)` does not mutate the spec (Python's `build()` fills `language` in place).
- Teammate worktree names use a 10-hex FNV-1a digest instead of `sha256(...)[:10]`, and member state lives under `<base>/<team>/sessions/<sid>/members/<member>`.
- `openjiuwen-harness-providers::create_harness` cannot receive an injected `Model`; offline tests construct `NativeHarness::new(agent, options)` directly.
- Errors are one `Error` enum plus `ErrorKind`, with `"<kind>: <message>"` display; Python's numeric `StatusCode` and exception class tree are not reproduced.
- Token counting resolves a tokenizer spec/registry, then a model→encoding table (`cl100k_base`/`o200k_base`), then tiktoken's model lookup, and finally a character-ratio approximation, so counts can differ from Python for uncatalogued models.
- Ordering is deterministic where Python used sets or dict iteration (sorted subscribers, ordered tool batches, sorted registry listings).

## Deferred Features (not implemented)

Each entry below is either absent or an API-shaped stub returning `Error::Unsupported`; the reason is categorized as vendor SDK/external service, ML/research stack, platform automation, or incompatible format/design.

### Kernel

- `openjiuwen-common` — Python's `StatusCode` enum and exception class tree (replaced by `ErrorKind` and one `Error`), the `clients/` connector pools/session managers/`ClientRegistry`, AES-GCM crypto, SSRF checks, SSL helpers, log rotation and invocation JSONL sinks (format/design).
- `openjiuwen-macros` — `Args:`-block parameter descriptions scraped from doc comments, flat keyword-parameter expansion, and sync/generator tool handlers (format/design).
- `openjiuwen-foundation` — `pgvector`/`postgres`/`gauss` vector backends, every external graph backend, Redis TLS/cluster, the Playwright MCP transport, the OpenAI Responses transport, DashScope streaming media and result-URL downloads, Anthropic thinking blocks in non-streaming responses, and `output_parsers`/`system_one` (vendor SDK / external service / platform automation).
- `openjiuwen-session` — Python pickle checkpoint payloads, `rediss://` TLS and Redis cluster/Sentinel, the `session_controller` downstream link graph/policies/`GlobalSessionController`, git VCS backends with `rewind`/`fork`, and Python's team-session shared global lane (format/design).
- `openjiuwen-graph` — pickle-persisted Pregel internals, the `stream_actor` graph-streaming machinery, DOT/PNG/SVG rendering, and any-of/count/timeout edge trigger modes (format/design).
- `openjiuwen-workflow` — live interleaving of concurrent streaming producers (Python's actor network), PNG rendering with `WORKFLOW_DRAWABLE` gating, per-run graph mutation, and sub-workflow HITL resume with forwarded seeded answers (format/design).
- `openjiuwen-context` — tokenizer artifact manager/downloads, native `tokenizer.json`/`tiktoken.model` loading, `provider_official` adapters, usage/KV-cache telemetry, provider-specific context policies, adaptive compressor/summariser selection, `processor/forked/` monkey-patching, rule-based offloader compression, and compression state stream frames (ML/research stack, ecosystem).
- `openjiuwen-memory` — vendor transports/SDKs for AgentArts/Jiuwen/LakeBase/mem0/OpenViking, per-backend migrators with backup/restore keyspace, AES at-rest codec, distributed KV lock and `scope_user_mapping`, the Milvus graph stack (LLM dedupe, time validity, relation rewiring), lite `sqlite_vec` embeddings and `MEMORY.md` per-fact split, the dreaming idle loop/LLM consolidation, and team-memory `TEAM_MEMORY.md` (vendor SDK, ML/research stack, format/design).
- `openjiuwen-retrieval` — Milvus/Chroma/PGVector/Elasticsearch vector stores, DashScope embeddings and rerankers, the LLM image captioner and triple extractor (seams only), multimodal embeddings, OCR for scanned PDFs, encrypted PDFs/object streams, and cross-reference-table page ordering (external service, ML/research stack, format/design).
- `openjiuwen-security` — `LocalModelBackend` inference (`torch`/`transformers`), moderation transports beyond OpenAI/Azure/Aliyun (DashScope/Google/Bedrock), batched or streaming moderation, and the process-global `register_guardrail` helper owned by `openjiuwen-runner` (ML/research stack, vendor SDK, operational).
- `openjiuwen-agent` — controller background scheduling/concurrency, task-manager indexes/filters/dependency scheduling, executor pause/cancel, per-conversation `TaskQueue` cancellation, group routing and `agent_reasoner.py`; KV-cache provider prefetch/offload and Redis/distributed backends; the operator `LLMCall` wrapper, `MetisContextEvolveOperator::learn` and `SkillExperienceOperator` staging (design/format).
- `openjiuwen-multiagent` — the Pulsar transport, poll-based and per-instance Redis delivery, Redis cluster/Sentinel/Streams consumer groups, drunner factories/reply-topic routing/`RemoteAgent`, and an out-of-process team event bus (external service/operational).
- `openjiuwen-runner` — Pulsar/ZeroMQ/A2A transports and reply-topic routing, Python's `python -m` child bootstrap and process pool, OS-process/external-CLI/SSH members, cross-process attach, the org layer, the monitor TUI, in-loop `InteractGate`, HITL auth/ACL, monitor persistence/export, callback signature introspection and resource-manager extras (operational/external service/platform).
- `openjiuwen-sysop` — `aio`/`jiuwenbox`/`yuanrong` sandbox transports, endpoint discovery beyond `LaunchedSandbox`, auth-header/query injection, the streaming wire format, provider/container caches, Big5-HKSCS/CP932 extensions, and the macOS `script` wrapper/Windows command classification (vendor SDK/platform/format).

### Harness, teams, and providers

- `openjiuwen-harness` — `browser`/`cua`/`mobile` subagent presets and tools (feature-gated stubs), `heartbeat_rail`, `personal_context`, `moa`, the harness `kv_cache` rail, `persistent_agent`, the OTel exporter (only the `TelemetrySink` seam exists), model-backed multimodal beyond header metadata, vendor/paid search backends, JS rendering and the LLM page-quality pass, LSP diagnostics/call hierarchy/implementation/crash retry, cron `update`/`toggle`/`preview`/`run` and the 7-field Quartz grammar, MCP resource subscriptions and the live-client adapter, worktree cross-process locking/sparse checkout/enter-exit tools, subagent activity streams/background eviction/lifecycle hooks, and Python's dynamic plugin execution model (platform automation / format design).
- `openjiuwen-harness-providers` — `native_v2` (hosted by `openjiuwen-agent-teams`) and the `claudecode`/`codex`/`dsh` vendor harnesses, whose config types ship behind `vendor` but whose `create` needs vendor SDKs and external CLIs (vendor SDK / external service).
- `openjiuwen-agent-teams` — pyzmq/hybrid-WebSocket and Pulsar transports, PostgreSQL/MySQL team stores, the distributed remote-workspace transport (`RemoteDispatch` seam only), OTLP span export (the `otel` feature is an in-memory seam), the monitor TUI and DB-backed monitor readers, attach authentication/TLS/named pipes, cross-process organisation federation, and LLM-driven fork/compact summaries with `IntelliRouter` selection (external service, platform automation).
- `openjiuwen-extensions` — the Pulsar message-queue transport, Redis `rediss://` TLS/cluster slot routing and its document/vector surfaces, the Elasticsearch and Gauss/PostgreSQL document stores, OTLP/gRPC and `http/protobuf` export, the local Grafana Agent push, the A2A gRPC binding, Milvus-backed context-evolver scoring, and remote-container sandbox teardown (external service, ML stack).

### Self-improvement and tooling

- `openjiuwen-dev-tools` — zip/gzip archive unpacking (format/design); Playwright/JS rendering with stealth/cookie handling and the Pillow/multimodal image judge (platform automation); agent-builder pipeline extras (intention detector, multi-turn clarifier, three-pass designer, DL refine mode, resource retrieval, streaming) and `tune` dataset samplers/`chat_agent` adapters (ML/research stack).
- `openjiuwen-symphony` — the faiss + sentence-transformers experience index and `transformers`/vLLM logit-based branch selection (ML/research stack); GCS and Azure Blob artifact transports (external service).
- `openjiuwen-rsi` — `agentdescent` 0.4.6 evolve/ledger/aggregator internals (ML/research stack); guarded judge-model plumbing and its prompts (`deferred::LlmJudgeEvaluator`, ML/research stack); synthetic `sys.modules` hot-loading (`deferred::RuntimeExtensionLoader`, format/design); live LLM stage/DeepAgent orchestration left to the caller's `HarnessRunner`; `infra::{github_cli,skill_source_manager}` shelling out to `gh`/the network (platform automation).
- `openjiuwen-agent-evolving` — the RL trainer on `torch`/`transformers`/`verl`/`ray` (ML/research stack); the Redis/vLLM online RL state machine and the `agent_rl` proxy/judge/verification gateway (external service); the remote sharing hub client (external service); vendor browser/PIL reward checks (platform automation); LLM experience drafting and TTSE embedding/dreaming (ML/research stack).

### CLI and development crates

- `openjiuwen-cli` / `openjiuwen-harness::cli` — auto-harness subcommands (`openjiuwen auto-harness …`), `--remote` agent-server mode, the rich/prompt_toolkit renderer with completion, and CLI mounting of the `lsp` tool (format design).

### `Error::Unsupported` surface

- `openjiuwen-foundation` — `store::external::registry` (`pgvector`/`postgres`/`gauss`/`gaussvector` and every non-local graph backend), `tool::mcp::connect_mcp_server` (`McpClientType::Playwright`) and `list_resources`/`read_resource` on transports without resources, `RestfulApiTool::stream`, Redis `rediss://`, and an unregistered LLM provider.
- `openjiuwen-session` — `checkpoint::{resp,redis}` (`rediss://` TLS), `checkpoint::create_checkpointer` (`sqlite` without the feature), the `Session::install_trace_sink` trait default, and read-only `RouterSession` views.
- `openjiuwen-workflow` — `Component::{invoke,stream,collect}` defaults for an undeclared ability, and `components::http_request::{prepare_body,apply_authentication}` for multipart bodies and digest/AWS signing.
- `openjiuwen-context` — `tokenizers::load_native_tokenizer`, which needs the native `tokenizers` runtime and network downloads.
- `openjiuwen-retrieval` — `Loaders`/`AutoLoader` for unknown extensions and image-only PDFs, `parsing::{zip,zlib}` for ZIP64/split/encrypted archives and non-stored methods, `parsing::pdf` for `/Encrypt` and compressed object streams, `VectorStoreRegistry` for any provider other than `memory`/`in_memory`, and `ChunkerRegistry` for unknown chunkers.
- `openjiuwen-memory` — `external::vendor` (the five vendor providers), `external::registry` for an unregistered factory, semantic search without an embedding model, and `lite::lock::FileLock` on non-unix targets.
- `openjiuwen-security` — `backends::local::LocalModelBackend::analyze` (no inference runtime) and `backends::api::ApiModelBackend` without an injected `ModerationTransport`.
- `openjiuwen-agent` — `kv_cache::types::KVCacheBackend::runtime` (Redis/distributed) and `operator::context_evolve::MetisContextEvolveOperator::learn`.
- `openjiuwen-multiagent` — `transport::build_transport` for `TransportKind::Pulsar` and `BaseTeam` without a strategy.
- `openjiuwen-runner` — `spawn::UnsupportedSpawner`, `spawn::remote::RemoteProtocol::{Mq,A2a}`, and `mq::create_message_queue` for Pulsar/ZeroMQ.
- `openjiuwen-sysop` — `sandbox::http::HttpSandboxGateway::invoke` (provider streaming wire) and `sandbox::{provider,launcher}` dispatch for an unregistered type or a disabled `sandbox` feature.
- `openjiuwen-harness` — `subagents::presets::build_preset` (`browser`/`cua`/`mobile`) and `workspace::DirectoryBuilder::build` when the sys-operation has no filesystem backend.
- `openjiuwen-harness-providers` — `factory::{create_harness,resolve_provider}` and `providers::{claudecode,codex,dsh}::provider` for `native_v2` and the vendor harnesses.
- `openjiuwen-agent-teams` — `messager::{create_messager,distributed}` (pyzmq/hybrid/Pulsar), `storage::open_store` (PostgreSQL/MySQL/unknown), `workspace::worktree_remote` for distributed workspaces, non-Unix `attach`, and vendor CLI harness translation through the generic stdio child.
- `openjiuwen-extensions` — `message_queue::pulsar`, backend/operation pairs in `store` (`build_kv_store`/`build_document_store`/`build_vector_store`), `checkpointer::redis` TLS/cluster, `otel::grafana` and `otel::env::parse_protocol`, a gRPC A2A `protocol_binding`, unknown `sys_operation` sandbox types, and `context_evolver::UnsupportedTaskMemoryService`.
- `openjiuwen-dev-tools` — `skill_creator::omni::ArchiveUnpacker` for `ArchiveFormat::{Zip,Gzip}`.
- `openjiuwen-symphony` — `deferred::{FaissExperienceIndex,VllmLogitSelector}`, `object_artifact::ArtifactTransport::{Gcs,AzureBlob}`, `orchestration::UnsupportedRunner`, and `evaluation::LlmJudgedMetric`.
- `openjiuwen-rsi` — `deferred::{EvolutionEngine,LlmJudgeEvaluator,RuntimeExtensionLoader}`.
- `openjiuwen-agent-evolving` — `rl::OfflineRlOptimizer::{train,start_training}`, `rl::OnlineRlOptimizer::{start_training,train_on_batch}` (the axum RL routes answer `501`), and `evaluator_pipeline` when `docker` or a command binary is missing.

Every other crate — `openjiuwen-common`, `openjiuwen-macros`, `openjiuwen-graph`, `openjiuwen-core`, `openjiuwen-harness-protocol`, `openjiuwen-cli`, `openjiuwen-auto-harness`, the `openjiuwen` umbrella, `openjiuwen-e2e`, `openjiuwen-examples` and `openjiuwen-test-support` — constructs no `Error::Unsupported`; capability gaps in `openjiuwen-harness-protocol` surface as `HarnessError::UnsupportedCapability` instead.

## Intentionally Different

- Typed traits and structs replace Python duck typing, decorators, and pydantic validation; cards carry `schemars` schemas instead of runtime validators.
- Streaming is `BoxStream<'static, Result<T>>` built with `async_stream`, not asyncio queues; cancellation is `tokio_util::sync::CancellationToken`.
- Synchronous Python entry points become `async`; there is no `asyncio.run`/`block_on` bridging in library APIs (`dev_tools` builders, `init_model`, `SpecBuild`).
- `Error` is one typed enum with `ErrorKind`; error categories preserve Python names but numeric status codes are gone.
- Registries are typed and instance-scoped (`ModelProviderRegistry`, `ResourceMgr`, `ToolRegistry`, harness provider registry) instead of module-level dicts; `Runner::global()` is the one documented process singleton.
- Services are injected as `Arc` values (`LongTermMemory`, `Runner`, sessions, checkpointers); Python's process-global memory singleton and implicit `Runner` wiring are gone.
- serde uses `snake_case` by default and `camelCase` only where an external protocol requires it (A2A, OTel attribute keys); unknown fields are tolerated everywhere.
- One `BaseMessage` struct replaces four role subclasses, and `ToolCard` is flat instead of embedding `CardBase`.
- Structured concurrency replaces asyncio task groups: the task manager registers before polling, tracks abort handles, catches panics, and aborts leftovers on drop.
- `AgentOutput.output` is a string with structured payloads in `data`/`workflow_output` (Python nested dicts are not reproduced).
- Determinism is a design rule: ids are explicit or UUID-based, subscriber/tool/registry ordering is sorted or insertion-ordered, and tests never assert wall-clock values.

## Verification

- `cargo test --workspace --offline` — the full offline gate over all 30 crates; per-runner pass/fail/ignore totals are recorded by CI and every runner is green.
- `cargo test -p openjiuwen-e2e --offline` — 30 passing plus 1 ignored with default features; `a2a_roundtrip` is `#![cfg(feature = "a2a")]` and compiles empty.
- `cargo test -p openjiuwen-e2e --features a2a --offline` — 34 passing plus 1 ignored; this is the `openjiuwen-e2e` line of `make test-all`.
- `cargo build -p openjiuwen-examples --examples` and `cargo build -p openjiuwen-examples --examples --features a2a,mcp` — compile all 12 examples (the second form is the only one that builds example 11).
- `cargo test -p openjiuwen-examples` — the 15 `src/support/` unit tests (run-mode detection and banner formatting); no integration binaries exist.
- `cargo test -p openjiuwen-test-support` — 149 tests, offline and loopback-only, with waits bounded at ≤2 s.
- Feature suites: `cargo test -p openjiuwen-harness --no-default-features`, `cargo test -p openjiuwen-agent-teams --no-default-features`, `cargo test -p openjiuwen-harness-providers --features vendor`, `cargo test -p openjiuwen-symphony --features sqlite`, `cargo test -p openjiuwen-sysop --features sandbox`, `cargo test -p openjiuwen-extensions --features a2a,otel` — the six feature suites of `make test-all`.
- `make verify` — runs `fmt-check`, `clippy`, `test`, `test-all` and both example builds in that order.
- `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo doc --workspace --no-deps` — style, lint and rustdoc gates.
- A bare `cargo test` uses the 27 `default-members` and therefore skips `openjiuwen-e2e`, `openjiuwen-examples` and `openjiuwen-test-support`; use `--workspace` or name those packages explicitly.
- 3 tests are `#[ignore]`d: `openjiuwen-sysop/src/sandbox/http.rs::probe_against_live_gateway` needs a live sandbox gateway (`--features sandbox -- --ignored`); `openjiuwen-e2e/tests/cli_binary.rs::binary_version_over_the_process_boundary` cannot use `CARGO_BIN_EXE_openjiuwen` because that variable exists only for the package owning the binary (the process-level smoke lives in `openjiuwen-cli/tests/cli_smoke.rs`); `openjiuwen-runner/tests/spawn_subprocess.rs::fake_child` is the stdio protocol child that sibling tests re-execute through a libtest filter selected by `FAKE_CHILD_MODE`.
- No test requires the network, credentials, or a wall clock: 7 manifests declare `live-tests`, no `.rs` file references it, and `OPENJIUWEN_LIVE` only switches the examples from offline to live.
