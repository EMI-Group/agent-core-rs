# Parity Matrix

Reference: openJiuwen Python SDK v0.1.20 (read-only foreign repo `/home/bill/Source/agent-core`).
The parity target is the Python package's public surface: module paths, wire formats, persisted JSON, tool names/schemas, permission outcomes, and observable loop behavior.
Rust internals are idiomatic (tokio, serde, traits, thiserror) rather than a transliteration, so internals are not compared field by field.

## Status Legend

- **Implemented** — the targeted Python public surface is ported, exported, and covered by tests.
- **Subset** — a coherent core is ported and usable; named peripheral paths are deferred (see §Deferred Features).
- **Stub** — the entry points exist but every operation returns `Error::Unsupported`.
- **Deferred** — not ported at all; no Rust crate or an explicitly empty seam stands in its place.

## Package Matrix

| Python path | Rust crate | Status | Notes |
|---|---|---|---|
| `core.common` (schema, exception, logging, clients, task_manager, security, utils, constants, run_context) | `openjiuwen-common` | Implemented | One `Error` enum plus `ErrorKind`, `Card`/`CardBase`/`CardId` with `schemars`, `tracing` logging, typed env resolvers, `SecretString`, shared async HTTP client, structured task manager. |
| `core.foundation.prompt` | `openjiuwen-foundation` | Implemented | `Role`, one `BaseMessage` struct, `MessageContent`/`ContentPart`, `ToolCall`/`ToolCallDelta`/`ToolCallAccumulator`, `PromptTemplate`, token counters. |
| `core.foundation.llm` | `openjiuwen-foundation` | Subset | OpenAI-compatible providers (`openai`/`pangu`/`qwen`/`deepseek` aliases) share one client and `anthropic` is the distinct Messages transport; reasoning profiles, IntelliRouter routing, OpenAI OAuth, and DashScope media generation are deferred. |
| `core.foundation.tool` | `openjiuwen-foundation` | Implemented | `Tool` trait, flat `ToolCard`, `FunctionTool`/`FunctionToolBuilder`, `RestfulApiTool`, JSON Schema via `schemars`, `#[tool]` macro in `openjiuwen-macros`. |
| `core.foundation.tool.mcp` | `openjiuwen-foundation` (feature `mcp`, default) | Subset | stdio, streamable-HTTP, and SSE clients over `rmcp`/`reqwest` with reconnect and session replay; no server-initiated GET stream, and OpenAPI/Playwright MCP transports are deferred. |
| `core.foundation.store` | `openjiuwen-foundation` (feature `sqlite`) | Subset | `KvStore`/`DocumentStore`/`VectorStore`/`GraphStore`/`ObjectStore` with in-memory, local-file, and optional SQLite backends; Chroma/Milvus/Gauss/Elasticsearch/pgvector/S3/Redis backends and the plugin factory registry are deferred. |
| `core.session` | `openjiuwen-session` (feature `sqlite`) | Implemented | `WorkflowSession`, `AgentSession`, `AgentTeamSession`, `NodeSession`; scoped state, stream frames, HITL interactions, checkpoint/resume, trace sinks; JSON state only and no remote checkpointer. |
| `core.graph` | `openjiuwen-graph` | Implemented | Pregel/BSP superstep executor, CNF channels, conditional routing, halt, retries, timeouts, cancellation, checkpoint store, Mermaid rendering. |
| `core.workflow` | `openjiuwen-workflow` | Subset | Builder API, `${node.field}` resolution, Start/End, LLM/tool/branch/loop/retrieval/memory components, streaming and checkpointing; several Python components and the streaming actor network are deferred. |
| `core.context_engine` | `openjiuwen-context` | Subset | `ContextEngine`, window assembly, token accounting, truncation/tool-result/summarisation processors, offload stores, overflow recovery; tokenizer registry and the Python compressor family are deferred. |
| `core.memory` | `openjiuwen-memory` (feature `sqlite`) | Subset | Scoped long-term memory CRUD, blended semantic/keyword recall, deterministic dedupe, basic entity/graph memory, lite coding facts, Python-compatible key layout; vendor providers, migrations, and the full Milvus graph stack are deferred. |
| `core.retrieval` | `openjiuwen-retrieval` | Subset | Text/JSON/Markdown/CSV/HTML/URL loaders, character/recursive/token chunking, OpenAI-compatible embeddings plus `HashEmbedding`, in-memory vector store, score/LLM reranker seams, `SimpleKnowledgeBase` pipeline; external vector DBs and PDF/DOCX/XLSX parsing are deferred. |
| `core.sys_operation` | `openjiuwen-sysop` (feature `sandbox`) | Subset | Local filesystem/shell/code backends with allowlists, path policy, timeouts, output bounds, streaming and cancellation; the remote sandbox provider registry is implemented but the AIO launcher wire protocol returns `Error::Unsupported`. |
| `core.security` | `openjiuwen-security` | Subset | Ordered guardrail pipeline, critical-only blocking, prompt-injection/sensitive-data/content-rule/tool-argument detectors, fail-open/closed policy; vendor moderation transports and local model inference are deferred. |
| `core.single_agent` | `openjiuwen-agent` | Implemented | `Agent` trait, `BaseAgent`, `ReActAgent`, abilities, rails, interrupts/HITL, streaming, `LLMAgent`, `WorkflowAgent`; legacy controllers, steering queues, and agent callbacks are deferred. |
| `core.application` | `openjiuwen-agent` | Implemented | `LLMAgent` is a composition over `ReActAgent`, `WorkflowAgent` selects and runs workflows; both are exposed through the `openjiuwen-core` `application` shim. |
| `core.multi_agent` | `openjiuwen-multiagent` | Implemented | `BaseTeam`/`Team`, `TeamCard`/`TeamConfig`, in-process message bus, handoff and hierarchical strategies, `AgentTool`, supervisor agents; distributed transports and callbacks are deferred. |
| `core.runner` | `openjiuwen-runner` | Subset | `Runner` (global plus local instances), `ResourceMgr`, callback framework, guardrail binding, in-process message queue, in-process team runs; Pulsar/ZeroMQ, remote agents, subprocess spawn, and team pools are deferred. |
| `core.controller` | — | Deferred | The legacy controller stack (`ControllerAgent`, `LLMController`, `WorkflowController`, intent controllers) is deliberately not ported. |
| `core.kv_cache` | — | Deferred | KV-cache runtime/config/metadata types and cache-affinity hooks have no Rust counterpart. |
| `core.operator` | — | Deferred | The tunable operator abstraction (`LLMCallOperator`, `ToolCallOperator`, `MemoryCallOperator`, context-evolve operators) belongs to the deferred tuning/evolution stack. |
| `core` umbrella | `openjiuwen-core` | Implemented | Pure re-export facade with Python-shaped module paths (`context_engine`, `sys_operation`, `multi_agent`, `single_agent`, `application`, `runner`). |
| `harness` (DeepAgent) | `openjiuwen-harness` (feature `native` default; `browser`/`cua`/`mobile`/`notify`) | Subset | DeepAgent/loop/builder, workspace, permissions, 20 model-facing tools, rail registry, prompt builder, subagents, skills, task loop, native harness protocol, CLI logic, manifest, observability seam. |
| `harness.cli` | `openjiuwen-harness` (`cli`) + `openjiuwen-cli` binary | Subset | `run`/`chat`, option precedence, permission prompts, session records, text/json/stream-json output; rich rendering, remote mode, auto-harness subcommands, and history restore are deferred. |
| `harness_protocol` | `openjiuwen-harness-protocol` | Implemented | Provider-neutral SPI: traits, events/results/streams, checkpoints, hooks, interactions, tool gateway, model control, wire codec (`PROTOCOL_VERSION` "1.0"). |
| `harness_providers` | `openjiuwen-harness-providers` (feature `native` default; `vendor`) | Subset | `native` provider implemented, registry/factory (`create_harness`, `resolve_provider`, descriptors), `HarnessIOAdapter`, skill-bundle installer; `native_v2` and `claudecode`/`codex`/`dsh` are declared but return `Error::Unsupported`. |
| `agent_teams` | `openjiuwen-agent-teams` (feature `mcp` default; `distributed` marker) | Subset | Team spec/build, leader/teammate runtime, task board, mailbox, HITL, workspace isolation, harness-protocol adapter, `team-member` CLI, `openjiuwen-team-mcp` server. |
| `extensions.a2a` | `openjiuwen-extensions` (feature `a2a`) | Subset | A2A client/server, agent-card adapter, JSON-RPC and SSE framing, harness bridge; gRPC/REST variants are deferred. |
| `extensions.tracer_otel` | `openjiuwen-extensions` (feature `otel`) | Subset | `TraceSink` implementation, agent/workflow handlers, semantic conventions, redaction, `tracing` layer; OTLP export and Grafana/Langfuse integrations are deferred. |
| `extensions.checkpointer`/`store`/`message_queue`/`sys_operation`/`context_evolver`/`vendor_specific` | `openjiuwen-extensions` (features `sqlite`/`redis`/`pulsar`) | Subset | Factories over kernel seams, JSON queue bridge, sandbox provider registration, offline context evolver, vendor reranker wrapper; Redis/Elasticsearch/Postgres/Gauss/Pulsar/AIO/JiuwenBox/YuanRong are deferred. |
| `dev_tools` | `openjiuwen-dev-tools` | Implemented | `prompt_builder`, `skill_creator`, `agent_builder`, and `skill_evaluator` with deterministic cores, bundle I/O, and agent DSL/config bridging. |
| `dev_tools.tune` | `openjiuwen-dev-tools::tune` | Stub | `Case`/`EvaluatedCase`/`CaseLoader` are ported; optimizers, `Trainer`, and `optimize_prompt` return `Error::Unsupported`. |
| `symphony` | `openjiuwen-symphony` (feature `sqlite`) | Subset | Capability cards/fingerprints, registry, lexical + semantic search, JGF `can_feed` graph, artifact store, deterministic planner, graph executor, runtime, static evaluation; LLM-guided stages are deferred. |
| `rsi` | `openjiuwen-rsi` | Subset | `harness_rsi::auto_harness` orchestrator/budget/stop/checkpoints, shared `evolution` protocol, deterministic mutator/selection, PUCT `program_opt`, `EvolutionPolicy`; agentdescent, paper optimization, and evaluator runtimes are deferred. |
| `agent_evolving` | `openjiuwen-agent-evolving` | Subset | Dataset/evaluator/optimizer/checkpoint/trainer/rails/service skeleton, `GridSearchOptimizer`, exact-match metrics, axum RL service with documented 501s; RL training and LLM judges are deferred. |
| `auto_harness` | `openjiuwen-auto-harness` | Implemented | Pure re-export facade over `openjiuwen_rsi::auto_harness` (same types by `TypeId`). |
| `openjiuwen` umbrella | `openjiuwen` (feature `native` default; `otel`, `a2a`, `sqlite`, `live-tests`) | Implemented | Module aliases for every package tree plus a ~35-item `prelude`; provider registry reachable as `openjiuwen::harness_providers`. |
| `openjiuwen` console script | `openjiuwen-cli` | Implemented | Binary `openjiuwen`: `.env` loading, logging, clap parse, `run`/`chat` dispatch and exit codes 0/1/2. |
| repository `examples/` | `openjiuwen-examples` | Implemented | Twelve runnable examples covering agent, tools, sessions, workflow, context/memory, retrieval, MCP, DeepAgent, native provider, teams, A2A, self-improvement. |
| repository `tests/` (cross-package) | `openjiuwen-e2e` | Implemented | 35 cross-crate scenario tests over 8 binaries (1 documented `#[ignore]`); `a2a_roundtrip` is feature-gated. |
| — (Rust-only) | `openjiuwen-macros` | Implemented | `#[tool]` attribute macro with trybuild pass/fail coverage. |
| — (Rust-only) | `openjiuwen-test-support` | Implemented | Dev-only `ScriptedModel`, loopback `MockLlmServer`, workspace/session/checkpointer fixtures. |

## Wire/Behavior Compatibility

### Models and streaming

- OpenAI-compatible vendors (`openai`, `pangu`, `qwen`, `deepseek`, and aliases) all resolve to one chat-completions client selected by `ModelClientConfig::provider`/`api_base`.
- `anthropic` is the only distinct transport and speaks the Messages API; `cache_control` breakpoints and mid-conversation `system` turns are deferred.
- Requests preserve the OpenAI wire format for messages, tool calls, `tool_choice`, `temperature`, `top_p`, `max_tokens`, `stop`, and streaming `stream=true`.
- Responses expose `ChatResponse`/`Choice`/`Usage` and a `StreamChunk` stream (`Delta`, `Usage`, `Done`); `StreamChunk::Delta` additionally carries `finish_reason` (additive field).
- Parallel tool calls and argument deltas are accumulated by `ToolCallAccumulator`; a mid-stream provider failure surfaces as the stream's `Err` item.
- Retries and timeouts live in the shared HTTP client; a non-2xx response is returned normally unless the caller requests `error_for_status()`.

### Cards, tools, and schemas

- Every `*Card` is a serde struct with Python field names and `#[serde(default)]`; unknown fields are tolerated and `None` fields are skipped on serialize.
- Cards implement `openjiuwen_common::card::Card` and derive `schemars::JsonSchema`, so JSON Schema is generated rather than hand-written.
- `ToolCard` is flat (`id`, `name`, `description`, `input_params`, policy flags); `input_params` is the JSON Schema the model sees, generated from typed handlers where available.
- `FunctionTool` handlers are async-only; Python sync handlers and generator-based tool streams are not reproduced (a blocking handler must use `spawn_blocking`).
- MCP tool names and JSON schemas are preserved through `McpTool`; streamable HTTP replays `Mcp-Session-Id`.

### Sessions and checkpoints

- State is JSON, keyed by `session_id`, with a global lane and component-namespaced lanes; dumps keep Python field names.
- `AgentSession` is the in-memory session implementation; there is no separate `InMemorySession` type in the Rust API.
- Checkpointing is the `Checkpointer` trait: in-memory, filesystem/JSON, and optional SQLite; two sessions sharing one checkpointer and id resume each other.
- Python pickle checkpoint payloads cannot be read, and a resumed trace continues from the checkpointed baseline.
- Human-in-the-loop pauses are `AgentOutput { result_type: Interrupt, ... }` values plus persisted `InterruptionState`, and resume is a normal invoke with an `InteractiveInput` in the query slot.
- Sessions write streaming frames as `StreamItem` values; agent `AgentChunk` frames map onto them and remain the public streaming contract.

### Workflow data flow

- `${node.field}` references are resolved in the engine against workflow state; a missing path resolves to `null` instead of failing.
- `outputs_schema` resolves against workflow state plus the node id and an `output` key (`${<id>.field}` / `${output.field}`); there is no `self` shorthand or bare `${field}`.
- Fan-in is CNF: a node with edges from several predecessors waits for all of them, so alternative branch targets must not merge into one node (Python's `wait_for_all=false` first-arrival path is not ported).
- Builder methods never return `Err`; problems are recorded and reported by `build`/`invoke`/`stream`/`to_mermaid`.

### Environment variables

- `OPENJIUWEN_HOME`, `OPENJIUWEN_API_KEY`, `OPENJIUWEN_API_BASE`, `OPENJIUWEN_SERVER_URL`, `OPENJIUWEN_LOG_LEVEL`, `OPENJIUWEN_LOG_FORMAT`, `OPENJIUWEN_BASH_STRICT`, `OPENJIUWEN_MCP_RECONNECT_ATTEMPTS`, `OPENJIUWEN_MCP_SERVER_CMD`, `OPENJIUWEN_OTEL_GLOBAL_INSTRUMENT_ENABLE`.
- Agent/CLI variables: `OPENJIUWEN_MODEL`, `OPENJIUWEN_MODEL_NAME`, `OPENJIUWEN_MODEL_PROVIDER`, `OPENJIUWEN_PROVIDER`, `OPENJIUWEN_WORKSPACE`, `OPENJIUWEN_MAX_ITERATIONS`, `OPENJIUWEN_MAX_TOKENS`, `OPENJIUWEN_SUBAGENT_TASK_TIMEOUT_S`, `OPENJIUWEN_TEAM_JOIN`, `OPENJIUWEN_LIVE`.
- OpenAI/legacy variables: `API_KEY`, `API_BASE`, `MODEL_PROVIDER`, `MODEL_NAME`, `OPENAI_API_KEY`, `OPENAI_BASE_URL`.
- `.env` loading is opt-in at binary/example entry points only (`dotenvy`), never in library constructors.

### CLI commands

- `openjiuwen run [PROMPT] [-f text|json|stream-json]` runs one prompt; the prompt is read from stdin when omitted or `-`.
- `openjiuwen chat` runs the interactive REPL; no subcommand chooses `chat` on a TTY and `run` otherwise.
- Global options: `-m/--model`, `--provider`, `--api-key`, `--api-base`, `-w/--workspace`, `-v/--verbose`.
- Exit codes: `0` success and `--help`/`--version`, `2` clap usage errors, `1` runtime failure printed as `error: <cause>` on stderr.
- `team-member` (skill CLI) and `openjiuwen-team-mcp` (stdio MCP server, `mcp` feature) ship from `openjiuwen-agent-teams`.

### Documented deviations

- `AgentSession` replaces Python's `InMemorySession`; `create_agent_session`/`AgentSession::with_options` are the constructors.
- A denied tool call is not an error: the observation starts with `[PERMISSION_DENIED]` (`DENIED_PREFIX`), `result_type` stays `Answer`, and the loop continues.
- CLI JSON shapes are `{answer, result_type, usage, session_id}` and a `stream-json` line per chunk plus a terminal `{"type":"result", ...}` line; Python emits a different `{result, chunks, model}` object and no terminal line.
- The CLI has no rich/prompt_toolkit renderer, ANSI colouring, slash-completion, or `~/.openjiuwen_history`; text output is plain and streamed.
- `TeamAgentSpec`'s `SpecBuild` is `#[async_trait]` and `build_with(&self)` does not mutate the spec (Python's `build()` fills `language` in place).
- Teammate worktree names use a 10-hex FNV-1a digest instead of `sha256(...)[:10]`, and member state lives under `<base>/<team>/sessions/<sid>/members/<member>`.
- `openjiuwen-harness-providers::create_harness` cannot receive an injected `Model`; offline tests construct `NativeHarness::new(agent, options)` directly.
- Errors are one `Error` enum plus `ErrorKind`, with `"<kind>: <message>"` display; Python's numeric `StatusCode` and exception class tree are not reproduced.
- Token counting uses `tiktoken-rs` (`cl100k_base`) with an approximate fallback, so counts can differ from Python for non-OpenAI models.
- Ordering is deterministic where Python used sets or dict iteration (sorted subscribers, ordered tool batches, sorted registry listings).

## Deferred Features (not implemented)

Each entry below is either absent or an API-shaped stub returning `Error::Unsupported`; the reason is categorized as ecosystem/vendor, ML/research, or operational/distributed.

### Kernel

- `openjiuwen-runner` — Pulsar/ZeroMQ distributed mode, reply-topic routing/`RemoteAgent`, real subprocess spawning (`UnsupportedSpawner` is the built-in), team pools/HITL/monitoring, tag/MCP/sysop/model/prompt managers, and callback decorators/filters/chains (operational/distributed).
- `openjiuwen-workflow` — `SubWorkflowComponent`, `QuestionerComponent`, `IntentDetectionComponent`, `ReActAgentComp`, `HTTPRequestComponent`, the streaming actor network, `wait_for_all=false` fan-in, PNG/SVG `draw()`, per-run graph mutation, `response_format` instruction injection, `enable_history` insertion, and `ExceptionConfig::default_outputs` recovery (ecosystem/peripheral).
- `openjiuwen-context` — tokenizer registry/native `tokenizer.json` loading, usage/KV-cache telemetry, provider-specific context policies, the Python compressor family (`DialogueCompressor`, `CurrentRoundCompressor`, `RoundLevelCompressor`, `FullCompactProcessor`, `MicroCompactProcessor`, offloaders, budget/window processors), forked processor registry, session-memory boundary reinjection, compression-recall chunking, `enable_reload` prompt injection, and per-processor checkpoint state (ML/ecosystem).
- `openjiuwen-memory` — external memory-provider SPI and vendor adapters (AgentArts/Jiuwen/LakeBase/Mem0/OpenViking), migrations/legacy prefixes/AES at-rest codec/distributed lock, LLM analyzer-extractor and conflict classification, the full Milvus graph stack (LLM dedupe, time validity, graph retrieval), the full lite stack (Markdown source of truth, FTS5/sqlite_vec, watchers, cross-process lock), and dreaming/team memory/SQL message tables (vendor/ML/ecosystem).
- `openjiuwen-retrieval` — Milvus/Chroma/PGVector/Elasticsearch vector stores, PDF/DOCX/XLSX parsing, triple/graph extraction, multimodal documents, query rewriting, sparse/BM25 retrievers, and agentic retrievers; unsupported formats and providers return `Error::Unsupported` (external services/parsers).
- `openjiuwen-sysop` — remote sandbox launcher lifecycle, endpoint discovery and `sandbox_id` assignment, isolation-key-to-container mapping, provider payload schemas, streaming wire format, and auth-header/query injection (sandbox calls return `Error::Unsupported`, surfaced as `OpResult` 199007); legacy CJK encodings, cross-process file locks, PTY/`stdbuf` streaming, and process-group kill are also deferred (external service/platform).
- `openjiuwen-security` — vendor moderation adapters (OpenAI/Aliyun/Azure), the HTTP transport for `ApiModelBackend` (needs an injected `ModerationTransport`), local model inference for `LocalModelBackend` (`torch`/`transformers`), and a real LLM prompt-injection judge; the built-in detectors themselves are implemented (vendor/ML).
- `openjiuwen-agent` — the legacy controller stack, steering queues, KV-cache affinity, skill utilities, the agent callback framework, sub-agent/workflow/MCP abilities, tool exposure policy and allowlists, client tools, per-tool resilience timeouts, `FinalRequestCheck`, multimodal tool results, `ReActAgentEvolve` operators, `context.usage`/`llm_route` frames, and YAML prompt packs (legacy/peripheral).
- `openjiuwen-multiagent` — Pulsar/Redis transports, event callbacks, card-driven auto-subscription, the global `Runner.resource_mgr` path, per-member checkpoint identity, Python-repr fallback, and `P2PAbilityManager` subclassing (operational).
- `openjiuwen-session` — Redis/remote checkpointer, disk-backed product sessions (`session_controller`), VCS history, Python pickle checkpoints, per-session OTel (lives in `openjiuwen-extensions`), and the `SubWorkflowSession`/`RouterSession` quirks (storage/format).
- `openjiuwen-graph` — Python's pickle-persisted Pregel internals are not reproduced; the observable builder/superstep/routing/resume/Mermaid contract is implemented (format).
- `openjiuwen-common` — Python's 400-value `StatusCode` enum and exception class tree are replaced by `ErrorKind` and one `Error` enum (design).

### Harness, teams, and providers

- `openjiuwen-harness` — `browser`/`cua`/`mobile` subagents and tools (feature-gated stubs returning `Error::Unsupported`), LSP client/tool, goal/heartbeat/personal_context/moa/kv_cache/persistent_agent, the OTel exporter (telemetry seam only), `DeepAgentSpec`/plugin/agent-template loading and `resources/`, full subagent quota/LRU/transcript streams, progressive tool discovery, web search/fetch tools, MCP resource tools, worktree, cron, image/audio multimodal tools, and RSI/evolving rail bodies (ecosystem/vendor/platform).
- `openjiuwen-harness-providers` — `native_v2` (hosted by `agent_teams`, returns `Error::Unsupported`) and the `claudecode`/`codex`/`dsh` vendor providers (config types exist behind `vendor`, but `create` returns `Error::Unsupported`; no vendor SDKs or external CLIs are spawned).
- `openjiuwen-agent-teams` — ZeroMQ/Pulsar transports, SQLite/PostgreSQL/MySQL stores, process/external-CLI/SSH members, git-worktree and remote workspace backends, the organisation layer, swarmflow, fork/compact, KV-cache affinity, OTel, monitor/TUI, model-router allocation, and cross-process attach (distributed/vendor).
- `openjiuwen-extensions` — Redis checkpointer/store, Elasticsearch/Postgres/Gauss stores, Pulsar message queue, ZeroMQ transports, OTLP span export, DashScope reranker HTTP transport, AIO/JiuwenBox/YuanRong sandbox providers, LLM-driven context-evolver algorithms, A2A gRPC/REST variants, and Grafana/Langfuse integrations (external services/vendor).

### Self-improvement and tooling

- `openjiuwen-dev-tools` — `skill_omni_creation` scraping/download (Playwright, bs4, image filtering), LLM agent-builder pipeline internals (clarifier, workflow designer, DL generator/reflector, Mermaid rendering), prompt-tuning optimizers (`InstructionOptimizer`, `ExampleOptimizer`, `JointOptimizer`, `Trainer`, `PromptSearchOptimizer`, which return `Error::Unsupported`), LLM-judge scoring, and the `run_eval_query.py` harness (ML/research).
- `openjiuwen-symphony` — LLM progressive tree retrieval, LLM ontology matching, LLM semantic planning scores, flow distillation/SkillPack, the experience bank, `SKILL.md` scanning and LLM fingerprint extraction, LLM-judged metrics, S3/OBS storage, and the `SkillFS` BM25 discovery layer (ML/LLM/vendor).
- `openjiuwen-rsi` — agentdescent 0.4.6 internals, subprocess/pip/pytest/git/CI infrastructure, evaluator runtimes (SWE-bench, terminal-bench, judge plumbing), paper optimization, member optimization, `single_harness` bodies, and the run ledger/progress records; each returns `Error::Unsupported` (ML/subprocess/vendor).
- `openjiuwen-agent-evolving` — the RL trainer (`torch`/`transformers`/`verl`/`ray`), the online RL state machine (Redis/vLLM), the LLM-as-judge default evaluator, research optimizers (instruction/tool/memory/skill/Metis/TTSE), the Docker+git evaluator pipeline, reward/attribution/experience/sharing/signal/trajectory/ttse subsystems, the proxy/judge/verification services, and the full RL config ecosystem (ML stack).
- `openjiuwen-auto-harness` — nothing beyond the `openjiuwen-rsi` deferrals; the facade re-exports exactly the canonical surface.

### CLI and development crates

- `openjiuwen-cli` / `openjiuwen-harness::cli` — auto-harness subcommands, remote mode, rich/prompt_toolkit rendering, ANSI colouring, the onboarding wizard, `/clear`/`/cost`/`/compact`/`/sessions`, skill-slash and `!cmd` passthrough, MCP config loading, web tools, `IDENTITY.md`/`OPENJIUWEN.md` overrides, plugin hot-loading, permanent allow-rule YAML write-back, and history restore.
- `openjiuwen-e2e` — no OTel scenario (the seam is a no-op without the feature and needs a collector); the process-boundary clap smoke test lives in `openjiuwen-cli`.
- `openjiuwen-examples` — `mobile_gui`, `lsp`, the RL training pipelines, vendor-backed stores, notebooks, and the research drops (`MANGO`, `PerStream`) are not ported.

### `Error::Unsupported` surface

- `openjiuwen-harness-providers`: `native_v2`, and vendor providers `claudecode`/`codex`/`dsh` through `create`/`provider()`.
- `openjiuwen-agent-teams`: distributed transports, external stores, process/CLI/SSH members, and non-local workspace backends.
- `openjiuwen-extensions`: Redis/Elasticsearch/Postgres/Gauss/Pulsar/AIO/JiuwenBox/YuanRong backends, the LLM context evolver, and the DashScope reranker transport.
- `openjiuwen-dev-tools::tune` optimizers and `Trainer` return `Error::Unsupported` naming the missing pipeline (they are not silent no-ops).
- `openjiuwen-symphony::deferred`, `openjiuwen-rsi::deferred`, `openjiuwen-agent-evolving::deferred`/`rl`, and `openjiuwen-security` model backends expose typed stubs that return `Error::Unsupported` with a reason.
- `openjiuwen-retrieval` unknown formats/providers, `openjiuwen-sysop` sandbox dispatch, and `openjiuwen-harness` feature-gated presets use the same error kind.

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

- `cargo test --workspace --offline` — 8164 passed, 0 failed, 5 ignored across 252 unit/integration/doc runners (exit 0).
- `cargo test -p openjiuwen-e2e --features a2a --offline` — 34 passed, 0 failed, 1 ignored across 10 runners (exit 0); the default-feature e2e run contributes 30 passed and 1 ignored.
- `cargo fmt --all -- --check` — clean.
- `cargo clippy --workspace --all-targets --no-deps --offline -- -D warnings` — clean, zero warnings (exit 0); `--no-deps` avoids pre-existing sibling `must_use_candidate` noise on `crate_name`.
- Feature suites: `cargo test -p openjiuwen-harness-providers --features vendor`, `cargo test -p openjiuwen-harness --no-default-features`, `cargo test -p openjiuwen-agent-teams --no-default-features`, `cargo test -p openjiuwen-extensions --features a2a`, `--features otel`, `--features sqlite`, `cargo test -p openjiuwen-sysop --features sandbox`, and `cargo test -p openjiuwen-symphony --features sqlite`.
- Examples: `cargo build -p openjiuwen-examples --examples` and `--features a2a,mcp` compile all 12 examples; `cargo test -p openjiuwen-examples` runs the shared-helper unit tests.
- A bare `cargo test` uses the workspace `default-members` and skips `openjiuwen-e2e` and `openjiuwen-examples`; use `--workspace` or name those packages explicitly.
- No test requires the network, credentials, or a wall clock; `#[ignore]`d tests name their prerequisite.
