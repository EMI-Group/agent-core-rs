# Workspace crates

## Intent

Every crate in the Rust rewrite of openJiuwen — **30 crates**, all of them implemented to some documented depth; no crate is an empty placeholder.
Each crate maps to a Python package (or focused kernel module) and owns its own `CONTEXT.md`.
Dependencies point downward through the layers below (details in `../docs/ARCHITECTURE.md` §1–§2).
Python package `openjiuwen.<pkg>` maps to crate `openjiuwen-<pkg>`; `openjiuwen-core` and `openjiuwen` are umbrellas preserving the module paths.

## Workspace Layout

| Crate | Layer | Role | Status |
| --- | --- | --- | --- |
| `openjiuwen-common` | 0 | typed errors, Cards/Config schema, logging, env/config, secrets, async HTTP, task manager, utilities | Implemented |
| `openjiuwen-macros` | 0 | `#[tool]` proc macro (async fn + arg struct → `ToolCard` + `Tool` impl) | Implemented |
| `openjiuwen-harness-protocol` | 0 | provider-neutral harness SPI: traits, events, turn results, checkpoints, streams, tool gateway, wire helpers | Implemented |
| `openjiuwen-foundation` | 1 | messages/prompts, LLM clients and `Model` (OpenAI/Anthropic/DashScope, OAuth), tools + MCP (stdio/SSE/HTTP/OpenAPI), storage backends (memory/file/SQLite/Redis/S3/Chroma/Milvus/Elasticsearch) | Implemented |
| `openjiuwen-session` | 1 | sessions, scoped state, streaming frames, HITL interactions, memory/file/SQLite/Redis checkpointers, disk store, VCS history, tracer | Subset |
| `openjiuwen-graph` | 1 | Pregel/BSP graph execution engine (builder, supersteps, routing, resume, Mermaid) | Implemented |
| `openjiuwen-retrieval` | 1 | documents, DOCX/XLSX/PDF parsing, chunking, embeddings, in-memory store, BM25 plus dense/sparse/hybrid/graph/agentic retrievers, query rewriting, triple/graph extraction, knowledge base | Subset |
| `openjiuwen-sysop` | 1 | local filesystem/shell/code operations (flock, PTY, CJK decoding), sandbox launcher lifecycle and provider registry | Subset |
| `openjiuwen-workflow` | 2 | workflow builder, control-flow built-ins plus sub-workflow, HTTP-request, questioner, intent-detection, ReAct, retrieval, memory and tool components, `${node.field}` data flow, run engine | Subset |
| `openjiuwen-context` | 2 | context windows, token accounting, compressor and offload processor families, recall archives, reload hints | Subset |
| `openjiuwen-memory` | 2 | long-term, entity/graph, lite coding memory (Markdown + FTS5), dreaming, team memory, migrations | Subset |
| `openjiuwen-security` | 2 | guardrails, detection pipeline, content rules, PII redaction, tool-arg validation, OpenAI/Azure/Aliyun moderation adapters | Subset |
| `openjiuwen-agent` | 3 | `Agent` seam, `ReActAgent`, abilities, rails, interrupts, controller stack, KV-cache runtime, operators and evolve, `LLMAgent`, `WorkflowAgent` | Subset |
| `openjiuwen-multiagent` | 3 | `BaseTeam`, team runtime, in-process + Redis message bus, handoff/supervisor/agents-as-tools strategies, member checkpoints | Implemented |
| `openjiuwen-runner` | 4 | `Runner`, `ResourceMgr` with five managers, callback framework, in-memory message queue, subprocess/HTTP-remote spawners, team pool, HITL hub, monitor | Implemented |
| `openjiuwen-core` | 4 | kernel umbrella: Python-shaped module path table over the concrete kernel crates | Implemented |
| `openjiuwen-harness` | 5 | DeepAgent task loop, workspace, permissions, model-facing tools (LSP, worktree, web, cron, MCP resources, goal, discovery), rails, prompts, subagents, skills, spec/plugins, native protocol impl, CLI logic | Subset |
| `openjiuwen-extensions` | 5 | A2A, OTel tracer (OTLP/HTTP, Grafana, Langfuse), checkpointer/store/MQ/sysop adapters, DashScope reranker, `context_evolver` | Subset |
| `openjiuwen-dev-tools` | 5 | prompt builder, skill creator/evaluator, agent builder, tune optimizers/trainer | Subset |
| `openjiuwen-symphony` | 5 | capability-asset engine: registry, BM25/semantic search, capability graph, `SkillFs`, flow distillation, planner, executor, runtime | Subset |
| `openjiuwen-harness-providers` | 6 | provider registry/factory, `native` and `native_v2` providers, `HarnessIOAdapter`, skill installer, `claudecode`/`codex`/`dsh` vendor CLIs behind `vendor` | Subset |
| `openjiuwen-rsi` | 6 | `auto_harness` self-optimisation loop, `program_opt` PUCT search, evolution protocol, `runtime` command/git/CI infrastructure | Subset |
| `openjiuwen-agent-evolving` | 6 | datasets, evaluators and evaluator pipeline, LLM-as-judge, optimizers/checkpoints, bounded trainer, evolution rails, portable RL loop, HTTP service | Subset |
| `openjiuwen-auto-harness` | 6 | pure facade over `openjiuwen-rsi::auto_harness` | Implemented |
| `openjiuwen-agent-teams` | 6 | leader/teammate runtime, task board, mailbox, HITL, SQLite team store, process/external-CLI/SSH members, org layer, Unix-socket attach, skill CLI + MCP server | Subset |
| `openjiuwen` | 7 | top-level umbrella: module aliases for every public tree plus a curated `prelude` | Implemented |
| `openjiuwen-cli` | 7 | binary `openjiuwen` (`run`, `chat`) — thin process boundary over `openjiuwen_harness::cli` | Implemented |
| `openjiuwen-test-support` | dev | loopback mock LLM server, scripted model, fixtures, JSON/eventual helpers | Implemented |
| `openjiuwen-e2e` | dev | cross-crate scenario suite (16 test binaries, 79 scenarios) | Implemented |
| `openjiuwen-examples` | dev | 20 runnable examples plus the offline smoke test that executes them | Implemented |

## Status Legend

- **Implemented** — the Python area the crate targets is covered end to end; remaining differences are internals deliberately not transliterated, or scope documented as out-of-scope elsewhere.
- **Subset** — a documented portion of the Python area is ported and working; the crate marks the unported remainder as deferred (trait seam, `Error::Unsupported`, or a documented gap section).
- **Stub** — API-shaped surface only, entry points return `Error::Unsupported`; applies to *areas* inside crates (`symphony`'s faiss/vLLM paths, `rsi::deferred`, the agent-evolving RL optimizers, the security local model backend, dev-tools' zip/gzip unpacking), never to a whole crate.
- **Deferred** — not implemented at all and documented as such (browser/CUA/mobile harness agents, distributed transports, external vector databases, OTLP/gRPC export); usually reached only through a trait seam.

Counts at crate level: **14 Implemented, 16 Subset, 0 Stub, 0 Deferred**.
`openjiuwen-foundation`, `openjiuwen-multiagent` and `openjiuwen-runner` keep their crate-level `Implemented` status while documenting backend, transport and parity gaps in the table below.

Deferred areas per crate (the 16 Subset crates, plus the documented gaps inside the otherwise Implemented `openjiuwen-foundation`, `openjiuwen-multiagent` and `openjiuwen-runner`):

| Crate | Deferred areas |
| --- | --- |
| `openjiuwen-session` | Python pickle checkpoint interop; `session_controller`'s downstream link graph/`ChainSession`, data-container and permission policies, global singleton; `vcs` git/external backends, WAL event references, `rewind`/`fork`, codecs; Redis cluster/Sentinel/TLS (`rediss://`) and pipelining; the process-global trace-handler registry; Python's `node_name` on `SubWorkflowSession`; team-session state sharing |
| `openjiuwen-retrieval` | Milvus/Chroma/PGVector/Elasticsearch stores, DashScope embeddings and rerankers, the LLM image captioner and LLM triple extractor (seams only), multimodal embedding models, OCR for scanned PDFs, encrypted PDFs and object streams, xref-table page ordering |
| `openjiuwen-sysop` | `aio`/`jiuwenbox`/`yuanrong` transports (they live in `openjiuwen-extensions`), the streaming wire (`invoke` → 199007), container config hashes/metadata/provider cache, endpoint discovery beyond `LaunchedSandbox`, auth-header/query injection, a background idle sweeper; CJK lacks Big5-HKSCS and CP932 extensions |
| `openjiuwen-workflow` | streaming-actor live interleaving, PNG rendering and `WORKFLOW_DRAWABLE` gating, per-run graph mutation, sub-workflow HITL resume (the child re-runs), HTTP multipart/digest/AWS auth |
| `openjiuwen-context` | tokenizer artifact manager/downloads and native tokenizer loading (`load_native_tokenizer` is `Unsupported`), usage/KV-cache telemetry, provider-specific window policies, adaptive compressor strategy selection and adaptive summarisation, the forked-processor registry and LLM session-memory agent, full-compact state reinjection, rule-based offloader compression and archive dumps, compression observability frames |
| `openjiuwen-memory` | vendor memory providers (record-only SPI), Python's per-backend migrators/backup-restore/AES/distributed KV lock, analyzer model overrides, the Milvus-backed graph stack and LLM dedupe/merge, lite semantic search (`sqlite_vec`; `SearchMode::Semantic` is `Unsupported`), the LLM dreaming pipeline, team-memory tool hooks and non-SQLite team backends, SQL message cursor parity |
| `openjiuwen-security` | `LocalModelBackend::analyze` (no inference runtime), moderation transports beyond OpenAI/Azure/Aliyun, batching and streaming moderation, the runner-owned global registration helper |
| `openjiuwen-agent` | the `before_steering_drain` rail hook, the process-global callback framework, KV-cache provider prefetch/offload and Redis/distributed backends, controller background scheduling/concurrency, task-manager indexes and filters, executor pause/cancel, group routing, `agent_reasoner.py`, the legacy operator `LLMCall` wrapper, the ML-backed Metis learner, memory-variable prompt keys, Python's prompt-file layout |
| `openjiuwen-harness` | `browser`/`cua`/`mobile` presets and agents, `heartbeat_rail`/`personal_context`/`moa`/`kv_cache`/`persistent_agent`, the OTel exporter (seam only), model-backed multimodal tools and vendor search backends, LSP diagnostics/call hierarchy/per-extension discovery/crash retry and CLI mounting, cron update/toggle/preview and the 7-field Quartz grammar, MCP resource subscriptions and the live-client adapter, worktree cross-process locking/sparse checkout/enter-exit tools, subagent activity streams and the background eviction scheduler, Python's dynamic plugin execution model |
| `openjiuwen-extensions` | OTLP/gRPC and `http/protobuf`, the local Grafana Agent push, `rediss://` TLS and Redis cluster/vector/document surfaces, the Elasticsearch document store, the Gauss/PostgreSQL document store, Pulsar, ZeroMQ, remote-container teardown and conditional sandbox writes, the Milvus-backed `context_evolver` scoring |
| `openjiuwen-dev-tools` | zip/gzip unpacking in `skill_omni_creation`, Playwright/JS rendering and vision judging, agent-builder pipeline extras (intention detector, clarifier, three-pass designer, DL refine, reflection loop, platform retrieval, streaming), `tune` dataset samplers/`chat_agent` adapters/`optimizer/adopt` services, `skill_evaluator`'s `create_subagent` ReAct orchestration |
| `openjiuwen-symphony` | the faiss/sentence-transformers experience index, transformers/vLLM logit branch selection, GCS and Azure Blob artifact transports |
| `openjiuwen-harness-providers` | `claudecode` lifecycle receipts and OTLP model-request observation, `codex` rollout-trace tailing and dynamic host tools, `dsh` plugin overlays and interrupt, vendor fork modes, `native_v2` checkpoint export/MCP tools/model selection, manifest and agent-template loading, entry-point discovery |
| `openjiuwen-rsi` | the agentdescent `program_opt` internals, `paper_opt` auto-research stages, evaluator runtime adapters and judger plumbing, Docker/WSL swe-bench and terminal-bench container internals, the LLM attributor/planner/executor seams, `github_cli`/`skill_source_manager`, `sys.modules` extension loading, live stage orchestration (the caller's `HarnessRunner`) |
| `openjiuwen-agent-evolving` | PPO/verl/ray training and the online RL service state machine (`Error::Unsupported`; HTTP routes answer `501`), the agent_rl proxy/judge gateway, the RL YAML ecosystem, vendor reward checks, the remote sharing-hub client, the LLM experience-draft pipeline, TTSE embedding retrieval and LLM dreaming |
| `openjiuwen-agent-teams` | ZeroMQ/WebSocket/Pulsar transports, PostgreSQL/MySQL stores and the distributed workspace transport, OTLP span export and the `opentelemetry` SDK, the monitor TUI and DB-backed readers, attach security beyond a `0600` socket, vendor CLI harness translation, cross-process org federation, LLM-driven context compaction and `IntelliRouter` |
| `openjiuwen-foundation` | pgvector/postgres/gauss vector backends and every external graph backend (typed `Unsupported`), Redis TLS/cluster and one-connection serialisation, Milvus/Chroma/Elasticsearch behavioural limits, the MCP Playwright transport and legacy-SSE replay, OpenAPI beyond local `$ref`, Anthropic non-streaming thinking blocks, the `Responses` transport (body encoding only), TPM/RPM enforcement, OAuth `/models`/CLI/file lock, DashScope streaming media |
| `openjiuwen-multiagent` | Pulsar and drunner/`RemoteAgent` transports |
| `openjiuwen-runner` | distributed mode (Pulsar/ZeroMQ/MQ envelopes/`RemoteAgent`, `RemoteProtocol::A2a`), Python's `python -m` child bootstrap and the process pool, team-pool extras (OS-process/external-CLI/SSH members, cross-process attach, monitor TUI, org layer), in-loop HITL gate enforcement and answering auth, monitor persistence/exports, resource-manager and callback-framework extras, `ReadWriteLockManager` |

## Dependency Layers

- **Layer 0 — independent foundations**: `openjiuwen-common`, `openjiuwen-macros`, `openjiuwen-harness-protocol`. No workspace dependencies (macros: compile-time only).
- **Layer 1 — primitives**: `openjiuwen-foundation` (depends on `common`, `macros`), `openjiuwen-session`, `openjiuwen-graph`, `openjiuwen-retrieval`, `openjiuwen-sysop`.
- **Layer 2 — composition**: `openjiuwen-workflow`, `openjiuwen-context`, `openjiuwen-memory`, `openjiuwen-security`.
- **Layer 3 — agents**: `openjiuwen-agent`, `openjiuwen-multiagent`.
- **Layer 4 — orchestration**: `openjiuwen-runner`, `openjiuwen-core` (umbrella re-exporting layers 0–4).
- **Layer 5 — product frameworks**: `openjiuwen-harness` (also consumes the `openjiuwen-core` umbrella), `openjiuwen-extensions`, `openjiuwen-dev-tools`, `openjiuwen-symphony`.
- **Layer 6 — providers and self-improvement**: `openjiuwen-harness-providers`, `openjiuwen-rsi`, `openjiuwen-agent-evolving`, `openjiuwen-auto-harness`, `openjiuwen-agent-teams`.
- **Layer 7 — top-level**: `openjiuwen` (umbrella over every public tree), `openjiuwen-cli` (binary).
- **Development crates** (never dependencies of library crates): `openjiuwen-test-support`, `openjiuwen-e2e`, `openjiuwen-examples`.

## Constraints

- A crate may depend only on crates in lower layers (see above); no cycles. `openjiuwen-runner` is driven by upper layers through their public seams only.
- `openjiuwen-core` and `openjiuwen` are umbrella crates; concrete crates must not depend on them except the harness layer and above.
- Shared types belong in `openjiuwen-common` or `openjiuwen-foundation`, never duplicated across crates.
- Crates are `publish = false` until the API stabilizes.
- Test-only crates are never dependencies of library crates; `openjiuwen-examples` uses `openjiuwen` (umbrella) plus dev-dependency crates only.
- Two binaries live outside `openjiuwen-cli`: `team-member` and `openjiuwen-team-mcp` (`required-features = ["mcp"]`) in `openjiuwen-agent-teams`.

## Feature Flags

Default builds need no network and no external service; every feature except the ones marked *default* is additive and opt-in.
`openjiuwen-agent`, `openjiuwen-agent-evolving`, `openjiuwen-auto-harness`, `openjiuwen-common`, `openjiuwen-context`, `openjiuwen-dev-tools`, `openjiuwen-graph`, `openjiuwen-harness-protocol`, `openjiuwen-macros`, `openjiuwen-multiagent`, `openjiuwen-retrieval`, `openjiuwen-rsi`, `openjiuwen-runner`, `openjiuwen-security`, `openjiuwen-test-support`, `openjiuwen-workflow` declare `default = []` and no other feature (16 crates).

| Crate | Feature | Default | Effect |
| --- | --- | --- | --- |
| `openjiuwen-foundation` | `mcp` | yes | MCP client/server tooling (pulls `rmcp`) |
| `openjiuwen-foundation` | `tiktoken` | yes | `cl100k_base` token counting |
| `openjiuwen-foundation` | `sqlite` | no | SQLite storage backends |
| `openjiuwen-foundation` | `live-tests` | no | empty opt-in test gate (no test reads it) |
| `openjiuwen-session` | `sqlite` | no | SQLite checkpointer |
| `openjiuwen-session` | `otel` | no | empty marker that enables the real trace-sink installation path |
| `openjiuwen-memory` | `sqlite` | no | SQLite memory persistence |
| `openjiuwen-sysop` | `sandbox` | no | sandbox provider transport (`reqwest`) |
| `openjiuwen-harness` | `native` | yes | native in-process DeepAgent harness |
| `openjiuwen-harness` | `browser`, `cua`, `mobile` | no | documented `Unsupported` preset stubs — agents deferred |
| `openjiuwen-harness` | `notify` | no | pulls the optional `notify` dependency; no code path is gated on it |
| `openjiuwen-harness` | `live-tests` | no | empty opt-in test gate (no test reads it) |
| `openjiuwen-harness-providers` | `native` | yes | `native` provider (in-process DeepAgent harness) |
| `openjiuwen-harness-providers` | `vendor` | no | registers `claudecode`/`codex`/`dsh`, which drive an operator-installed CLI through `vendor_cli` |
| `openjiuwen-harness-providers` | `native-v2` | no | adds the `native_v2` provider hosted by `openjiuwen-agent-teams` |
| `openjiuwen-harness-providers` | `live-tests` | no | empty opt-in test gate (no test reads it) |
| `openjiuwen-agent-teams` | `mcp` | yes | the `openjiuwen-team-mcp` binary and stdio MCP server |
| `openjiuwen-agent-teams` | `distributed` | no | distributed `Messager` seams (reject at construction) |
| `openjiuwen-agent-teams` | `sqlite` | no | `SqliteTeamStore` (`sqlx`) |
| `openjiuwen-agent-teams` | `otel` | no | in-memory/tracing OTel seam (the exporter is in `openjiuwen-extensions`) |
| `openjiuwen-agent-teams` | `live-tests` | no | empty opt-in test gate (no test reads it) |
| `openjiuwen-extensions` | `a2a` | no | A2A client/server, wire types, harness bridge |
| `openjiuwen-extensions` | `otel` | no | real OTel SDK provider, sink, handlers, tracing layer |
| `openjiuwen-extensions` | `mcp` | no | MCP adapter surface (`rmcp`) |
| `openjiuwen-extensions` | `sqlite` | no | SQLite checkpointer/store backends |
| `openjiuwen-extensions` | `redis` | no | Redis KV/checkpointer adapters (`rediss://` TLS and cluster stay `Unsupported`) |
| `openjiuwen-extensions` | `pulsar` | no | Pulsar queue surface — construction returns `Error::Unsupported` |
| `openjiuwen-extensions` | `live-tests` | no | empty opt-in test gate (no test reads it) |
| `openjiuwen-symphony` | `sqlite` | no | SQLite capability-asset store (`store::KvAssetStore::sqlite`) |
| `openjiuwen-core` | `otel` | no | forwards `openjiuwen-session/otel`, so `core::session::install_trace_sink` really installs the supplied sink |
| `openjiuwen` | `native` | yes | forwards `harness/native` + `harness-providers/native` |
| `openjiuwen` | `otel`, `a2a`, `sqlite` | no | forwards the matching `openjiuwen-extensions` (plus symphony `sqlite`, kernel `otel`) features |
| `openjiuwen` | `live-tests` | no | forwards the harness/providers/teams/extensions live gates |
| `openjiuwen-cli` | `live-tests` | no | empty opt-in test gate (no test reads it) |
| `openjiuwen-e2e` | `a2a` | no | enables `tests/a2a_roundtrip.rs` |
| `openjiuwen-e2e` | `otel` | no | parity forwarding only — no cross-crate scenario |
| `openjiuwen-e2e` | `sqlite` | no | enables `openjiuwen-agent-teams/sqlite` for `tests/team_backend.rs` |
| `openjiuwen-examples` | `a2a` | no | example 11 (`a2a_server_client`) |
| `openjiuwen-examples` | `mcp` | no | marker only — MCP is always compiled through `openjiuwen-foundation`'s default |

Note: `openjiuwen-harness` and `openjiuwen-harness-providers` are non-optional umbrella dependencies taken with default features, so their `native` default always applies; nothing in the workspace enables `vendor`, and a consumer wanting vendor providers must depend on `openjiuwen-harness-providers` directly.
`openjiuwen-harness-providers`' optional `native-v2` feature depends on `openjiuwen-agent-teams` (same layer 6), the only same-layer edge in the workspace; it is off by default and enabled by nothing here.

## Test Conventions

- Unit tests live beside the module (`#[cfg(test)] mod tests`) and are documented in that module's `CONTEXT.md`.
- Integration tests live in `crates/<crate>/tests/` as one binary per file (308 `tests/*.rs` files today; the umbrellas keep only `tests/module_paths.rs`, and `openjiuwen-examples/tests/examples_offline.rs` runs every example binary offline).
- Cross-crate scenarios live only in `openjiuwen-e2e/tests/`: 16 scenario binaries (`kernel_flow`, `workflow_components`, `context_engine`, `retrieval_pipeline`, `memory_recall`, `deep_agent`, `harness_cli_tools`, `provider_native`, `team_flow`, `team_backend`, `a2a_roundtrip`, `symphony_plan`, `self_improvement`, `evaluation_ledger`, `runner_subprocess`, `cli_binary`) holding 79 scenarios, with bounded helpers in `tests/support/mod.rs`.
- 78 e2e scenarios pass with `--all-features` and 73 with default features; exactly one is a documented `#[ignore]`.
- `openjiuwen-test-support` supplies the shared fakes: loopback `MockLlmServer`, `ScriptedModel`, temp-workspace/env fixtures, JSON assertions, eventual-consistency helpers.
- Everything is offline and deterministic by default: scripted models or the loopback server, temp directories, loopback sockets, no credentials, no wall-clock ordering, bounded awaits (10 s `BOUND` in `openjiuwen-e2e`).
- The `live-tests` features (`foundation`, `harness`, `harness-providers`, `agent-teams`, `extensions`, `cli`, `openjiuwen`) are declared empty gates that no Rust source reads; suites that need an optional capability are instead gated by `#![cfg(feature = "…")]` (`a2a`, `otel`, `sqlite`, `redis`, `pulsar`, `vendor`, `native-v2`) or by `#![cfg(unix)]`, and the single live-network test is `#[ignore]`d.
- `cargo test --workspace` must pass offline; each crate additionally documents its own definition of done (`cargo fmt`, `cargo clippy -- -D warnings`).
- Deferred areas are proven, not just documented: dedicated suites assert `Error::Unsupported` and its message (for example `openjiuwen-dev-tools/tests/deferred_surface.rs`, `openjiuwen-symphony/tests/deferred.rs`, `openjiuwen-agent-evolving/tests/rl.rs`, `openjiuwen-sysop/tests/sandbox_stub.rs`).
- `cargo doc --workspace --no-deps` (and `--all-features`) must stay warning-free; link private or feature-gated items as plain code spans instead of widening visibility, and never add crate-level `rustdoc::` allows.
- Module-level (`//!`) docs resolve intra-doc links in the parent module's scope, so links there usually need an explicit `crate::…` path.
- `crates/openjiuwen-cli`'s `[[bin]] name = "openjiuwen"` sets `doc = false`: its rustdoc output name would otherwise collide with the `openjiuwen` umbrella crate's `target/doc/openjiuwen/`.

## Routing Table

### Kernel core (layers 0–4)

- `./openjiuwen-common/` → Card/Config schema, typed errors, logging, env/config, HTTP helpers, task manager, utilities
- `./openjiuwen-macros/` → the `#[tool]` attribute macro (`async fn` + arg struct → `ToolCard` + `Tool` impl)
- `./openjiuwen-foundation/` → LLM clients and `Model`, prompts/messages, tools + MCP, storage backends
- `./openjiuwen-session/` → sessions, state scopes, streaming, interactions, checkpointers, disk store, VCS history, tracer
- `./openjiuwen-graph/` → Pregel/BSP graph execution engine
- `./openjiuwen-retrieval/` → documents, parsing, chunking, embeddings, vector stores, BM25/retrievers, rerankers, query rewriting, knowledge base
- `./openjiuwen-sysop/` → local/sandbox filesystem, shell (PTY, flock), and code operations
- `./openjiuwen-workflow/` → workflow builder, built-in components, `${node.field}` data flow
- `./openjiuwen-context/` → context windows, token accounting, compressor/offload processors
- `./openjiuwen-memory/` → long-term, entity/graph, lite, team memory, dreaming
- `./openjiuwen-security/` → guardrails, moderation backends and content-safety pipeline
- `./openjiuwen-agent/` → `Agent` seam, `ReActAgent`, abilities, rails, interrupts, controller, KV cache, operators/evolve, `LLMAgent`, `WorkflowAgent`
- `./openjiuwen-multiagent/` → `BaseTeam`, team runtime, message bus (in-process + Redis), strategies
- `./openjiuwen-runner/` → `Runner`, `ResourceMgr`, callbacks, in-memory MQ, subprocess spawner, pool, HITL, monitor
- `./openjiuwen-core/` → kernel umbrella module path table over the crate above

### Harness

- `./openjiuwen-harness-protocol/` → provider-neutral harness SPI (traits, events, results, streams)
- `./openjiuwen-harness/` → DeepAgent task loop, rails, workspace, permissions, tools, LSP, skills, subagents, prompts, native protocol, CLI logic
- `./openjiuwen-harness-providers/` → provider registry/factory, `native` and `native_v2` providers, `HarnessIOAdapter`, vendor CLI adapters
- `./openjiuwen/` → top-level umbrella re-exporting every public module tree and the `prelude`

### Teams and self-improvement

- `./openjiuwen-agent-teams/` → leader/teammate runtime, worktrees, SQLite store, org layer, attach, HITL, skill CLI/MCP server
- `./openjiuwen-rsi/` → recursive self-improvement: `auto_harness` loop, `program_opt`, evolution protocol, `runtime` infrastructure
- `./openjiuwen-auto-harness/` → re-export shim over `rsi::auto_harness`
- `./openjiuwen-agent-evolving/` → evaluation/judging/optimisation/trainer skeleton with the RL engines deferred

### Integrations

- `./openjiuwen-extensions/` → A2A, OTel tracer (OTLP/HTTP, Grafana, Langfuse), checkpointer/store/MQ/sysop adapters (feature-gated)
- `./openjiuwen-symphony/` → offline capability-asset engine (registry/search/graph/`SkillFs`/flow/planner/executor/runtime)

### Tooling and CLI

- `./openjiuwen-cli/` → binary `openjiuwen` (`run`, `chat`)
- `./openjiuwen-dev-tools/` → prompt builder, skill creator/evaluator, agent builder, tune optimizers

### Test infrastructure

- `./openjiuwen-test-support/` → mock LLM server, fakes, fixtures
- `./openjiuwen-e2e/` → cross-crate integration scenarios

### Examples

- `./openjiuwen-examples/` → 20 runnable examples mirroring the Python `examples/` tree
