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
| `openjiuwen-foundation` | 1 | messages/prompts, LLM clients and `Model` (OpenAI-compatible/Anthropic/Responses/DashScope, OAuth, the external-provider registry, `purpose`/`metadata` hints), tools + MCP (stdio/SSE/HTTP/OpenAPI) with `ToolInfo` flat accessors, storage backends (memory/file/SQLite/Redis/S3/Chroma/Milvus/Elasticsearch) | Implemented |
| `openjiuwen-session` | 1 | sessions, scoped state, streaming frames, HITL interactions, memory/file/SQLite/Redis checkpointers, disk store, VCS history, tracer | Subset |
| `openjiuwen-graph` | 1 | Pregel/BSP graph execution engine (builder, supersteps, routing, resume, Mermaid) | Implemented |
| `openjiuwen-retrieval` | 1 | documents, DOCX/XLSX/PDF/PPTX/ODT parsing from paths or in-memory bytes, chunking, embeddings, in-memory store, BM25 plus dense/sparse/hybrid/graph/agentic retrievers, query rewriting, triple/graph extraction, knowledge base | Subset |
| `openjiuwen-sysop` | 1 | local filesystem/shell/code operations (flock, PTY, CJK decoding), sandbox launcher lifecycle and provider registry | Subset |
| `openjiuwen-workflow` | 2 | workflow builder, control-flow built-ins plus sub-workflow, HTTP-request, questioner, intent-detection, ReAct, retrieval, memory and tool components, `${node.field}` data flow, run engine | Subset |
| `openjiuwen-context` | 2 | context windows, token accounting and usage telemetry, native tokenizer loading with the artefact manager, provider window policies, compressor/offload processor families plus rule compression, adaptive ladders, recall archives, reload hints and full-compact reinjection | Subset |
| `openjiuwen-memory` | 2 | long-term/entity-graph memory, external REST provider transports, migrations/backup/KV locks, a staged analyzer with variables, lite coding memory (facts, embeddings, sessions, daily rollups, sync), dreaming orchestration with leases/retention/salience, team hooks and toolkit | Subset |
| `openjiuwen-security` | 2 | guardrails, detection pipeline, content rules, PII redaction, tool-arg validation, OpenAI/Azure/Aliyun moderation adapters | Subset |
| `openjiuwen-agent` | 3 | `Agent` seam, `ReActAgent`, abilities, rails, interrupts, controller with background scheduler/task manager/group router/reasoner, KV-cache action runtime, memory-provider bridge, usage telemetry, operators (incl. legacy `LLMCall`)/experience staging/prompt packs, `LLMAgent`, `WorkflowAgent` | Subset |
| `openjiuwen-multiagent` | 3 | `BaseTeam`, team runtime, in-process + Redis Streams transports with a subscription registry, routing/inbox filters, handoff/supervisor/agents-as-tools strategies, member checkpoints | Implemented |
| `openjiuwen-runner` | 4 | `Runner`, `ResourceMgr` with five managers, callback framework, message queue with `QueueStats`, spawn seam/registry and monitored handles, team pool, HITL hub, monitor, dreaming service | Implemented |
| `openjiuwen-core` | 4 | kernel umbrella: Python-shaped module path table over the concrete kernel crates | Implemented |
| `openjiuwen-harness` | 5 | DeepAgent task loop, workspace, permissions, 35 model-facing tools (LSP, worktree, web/search, cron, MCP resources/subscriptions, goal, discovery), rails incl. MoA/personal-context/heartbeat/KV-cache, prompts, subagents with activity/eviction/lifecycle, skills, persistent agent and resume APIs, the `notify` watcher, spec/plugins, native protocol impl, CLI logic | Subset |
| `openjiuwen-extensions` | 5 | A2A, OTel tracer (OTLP/HTTP JSON + protobuf, Grafana Cloud/local push, Langfuse), the `observability` utilities package, checkpointer/store/MQ/sysop adapters (Redis KV/document/vector, Elasticsearch vector/document, sandbox teardown and conditional writes), DashScope reranker, `context_evolver` | Subset |
| `openjiuwen-dev-tools` | 5 | prompt builder, skill creator/evaluator, agent builder, tune optimizers/trainer | Subset |
| `openjiuwen-symphony` | 5 | capability-asset engine: registry, BM25/semantic search, capability graph, `SkillFs` discovery toolkits, flow distillation/codegen/render, planner/executor, evaluation and trace metrics, experience index, mutation journal, observation revisions, matcher/normalization, capability-tree build and retrieval runtime | Subset |
| `openjiuwen-harness-providers` | 6 | provider registry/factory, `native` and `native_v2` providers, `HarnessIOAdapter`, skill installer, `claudecode`/`codex`/`dsh` vendor CLIs behind `vendor` | Subset |
| `openjiuwen-rsi` | 6 | `auto_harness` self-optimisation loop, `program_opt` PUCT search, evolution protocol, `runtime` command/git/CI infrastructure | Subset |
| `openjiuwen-agent-evolving` | 6 | datasets, evaluators and evaluator pipeline, LLM-as-judge, optimizers/checkpoints, bounded trainer, evolution rails/signals/rewards/attribution/trajectory/experience, experience drafting, sharing hub, TTSE retrieval, portable RL loop with a Redis-backed online state machine, HTTP service | Subset |
| `openjiuwen-auto-harness` | 6 | pure facade over `openjiuwen-rsi::auto_harness` | Implemented |
| `openjiuwen-agent-teams` | 6 | leader/teammate runtime, task board, mailbox, HITL, SQLite team store, process/external-CLI/SSH members with checkpoint env vars, org layer, Unix-socket attach, team-memory wiring and OTLP memory observer, skill CLI + MCP server | Subset |
| `openjiuwen` | 7 | top-level umbrella: module aliases for every public tree plus a curated `prelude` | Implemented |
| `openjiuwen-cli` | 7 | binary `openjiuwen` (`run`, `chat`, `lsp`, `cron`, `team`, `watch`, `sessions`, `auto-harness`, with `run`/`chat` `--resume`/`--continue`) — thin process boundary over `openjiuwen-harness` | Implemented |
| `openjiuwen-test-support` | dev | loopback mock LLM server, scripted model, fixtures, JSON/eventual helpers | Implemented |
| `openjiuwen-e2e` | dev | cross-crate scenario suite (26 test binaries, 125 scenarios) | Implemented |
| `openjiuwen-examples` | dev | 26 runnable examples plus the offline smoke test that executes them | Implemented |

## Status Legend

- **Implemented** — the Python area the crate targets is covered end to end; remaining differences are internals deliberately not transliterated, or scope documented as out-of-scope elsewhere.
- **Subset** — a documented portion of the Python area is ported and working; the crate marks the unported remainder as deferred (trait seam, `Error::Unsupported`, or a documented gap section).
- **Stub** — API-shaped surface only, entry points return `Error::Unsupported`; applies to *areas* inside crates (`symphony`'s faiss/vLLM paths, `rsi::deferred`, the agent-evolving RL optimizers, the security local model backend, dev-tools' zip/gzip unpacking, extensions' `observability::deferred`), never to a whole crate.
- **Deferred** — not implemented at all and documented as such (browser/CUA/mobile harness agents, distributed transports, external vector databases, OTLP/gRPC export); usually reached only through a trait seam.

Counts at crate level: **14 Implemented, 16 Subset, 0 Stub, 0 Deferred**.
`openjiuwen-foundation`, `openjiuwen-multiagent`, `openjiuwen-runner`, `openjiuwen-cli` and `openjiuwen-macros` keep their crate-level `Implemented` status while documenting backend, transport and parity gaps in the table below.

Deferred areas per crate, condensed from each crate's own `CONTEXT.md` gap section (Implemented crates appear here only for the documented gaps they keep; every item is genuinely unimplemented and carries its reason in the crate document):

| Crate | Deferred areas |
| --- | --- |
| `openjiuwen-session` | Python pickle checkpoint interop; `session_controller`'s downstream link graph/`ChainSession`, data-container and permission policies, the global singleton; `vcs` git/external backends, WAL event references, `rewind`/`fork` and codecs; Redis cluster/Sentinel/TLS (`rediss://`) and pipelining; a process-global trace-handler registry; Python's `node_name` on `SubWorkflowSession`; team-session state sharing and `trace_state` in `dump_state` |
| `openjiuwen-retrieval` | Milvus/Chroma/PGVector/Elasticsearch stores, DashScope embeddings and rerankers, the LLM image captioner and LLM triple extractor (seams only), multimodal embedding models, OCR for scanned PDFs, encrypted PDFs and object streams, xref-table page ordering |
| `openjiuwen-sysop` | `aio`/`jiuwenbox`/`yuanrong` transports (they live in `openjiuwen-extensions`), the streaming wire (`invoke` → 199007), container config hashes/metadata/provider cache, endpoint discovery beyond `LaunchedSandbox`, auth-header/query injection, a background idle sweeper; Big5-HKSCS and CP932 CJK extensions |
| `openjiuwen-workflow` | streaming-actor live interleaving, Python's loop-coordination components (`LoopGroup`/`LoopBreakComponent`/`LoopSetVariableComponent`/`AdvancedLoopComponent`), PNG rendering and Python's `WORKFLOW_DRAWABLE` gate, per-run graph mutation, Python's per-component config objects and `WorkflowCard::tool_info()`, HTTP multipart/digest/AWS auth, build-time loop-parameter checks and Python's split build/run validation |
| `openjiuwen-context` | provider-official tokenizer adapters, regex-pattern tokenizer nodes and the NFC/NFD/NFKC/NFKD normalizer families (`regex` and `unicode-normalization` are already locked transitively but are not direct dependencies, so these are offline-implementable), full HuggingFace Unigram parity, host-specific reinjection builders (the `ContextSnapshotProvider` seam is here; the readers live in `openjiuwen-harness`), the LLM session-memory agent, rule-compression gaps (JSON repair, readability ladder, Unicode sniffing, workspace dumps), Python's JSONL debug/compression dumps and structured compression-error taxonomy, a `ModelRequestConfig` purpose field |
| `openjiuwen-memory` | external providers' SDK transports/session-mapping/circuit breakers (record-only SPI), Alembic-style migrations and dialect handling, the AES at-rest codec and multi-node KV locks, provider-specific analyzer tuning and the LLM redundancy checker, the Milvus-backed graph stack with LLM dedupe/merge, `sqlite_vec` ANN and Python's lite schema, the ML salience backend, background dream lease heartbeats/fencing and timer-driven retention, team tool registration/prompt injection and embedding-based team recall, SQL message pagination parity |
| `openjiuwen-security` | `LocalModelBackend::analyze` (no inference runtime), moderation transports beyond OpenAI/Azure/Aliyun, batching and streaming moderation, the runner-owned global registration helper |
| `openjiuwen-agent` | the `before_steering_drain` rail hook, the process-global callback framework, KV-cache Redis/distributed backends and vendor prefetch/offload transports, memory-provider backends (bridged only), tokenizer artifacts and window policies (owned by `openjiuwen-context`), controller timeout/history limits and Python's reasoner/proactive/reflector config, the ML-backed Metis learner and subset approval, Python's manifest/skill packages as prompt sources |
| `openjiuwen-harness` | browser/CUA/mobile presets and agents, the OTel exporter (seam only), model-backed multimodal tools and vendor search backends, LSP `.gitignore` filtering/server auto-discovery/formatter-rename, cron `wake`/`at`/`every` forms/payload delivery/`W`/run-output capture, MCP server-pushed subscriptions and pagination, Python's worktree lifecycle policies and symlink handling, subagent stream-milestone projection and cross-process sweepers, MoA `jev_routed`, persistent-agent remote handoff/budget service/archive tools, personal-context vendor fetchers, KV-cache Redis backends, watcher target add/remove and cross-process locking, Python's dynamic plugin execution model |
| `openjiuwen-extensions` | OTLP/gRPC framing, `rediss://` TLS and Redis cluster slot routing, the Gauss/PostgreSQL document store, the Pulsar and ZeroMQ transports, AIO container teardown, cross-process conditional-write atomicity, Redis vector ranking (brute force, no RediSearch), Milvus-backed `context_evolver` scoring, Elasticsearch credentials on the vector adapter, and the `observability` SDK-adapter layer (`setup`/`demand`/`runtime`, span-record/contextvar fan-out, monkey-patched instrumentation, the compression→trajectory bridge) |
| `openjiuwen-dev-tools` | zip/gzip unpacking beyond stored/DEFLATE, Playwright/JS rendering and vision judging, agent-builder pipeline extras (intention detector, clarifier, three-pass designer, DL refine, reflection loop, platform retrieval, streaming), `tune` dataset samplers/`chat_agent` adapters/`optimizer/adopt` services, `skill_evaluator`'s `create_subagent` orchestration, human-readable eval reports |
| `openjiuwen-symphony` | the faiss/sentence-transformers experience index, transformers/vLLM logit branch selection, GCS service-account signing and Azure AD client credentials, LLM/faiss grouping and tree-build paths, Python's matcher file lock and pending-merge batching, the retrieval search-term families and the agent-facing retrieval toolkit, and discovery's filesystem caches/index states |
| `openjiuwen-harness-providers` | `claudecode` lifecycle receipts and OTLP model-request observation, `codex` rollout-trace tailing and dynamic host tools/model control, `dsh` plugin overlays and interrupt, vendor fork modes, `native_v2` checkpoint export/MCP tools/model selection, manifest and agent-template loading, entry-point discovery |
| `openjiuwen-rsi` | the agentdescent `program_opt` internals, `paper_opt` auto-research stages, evaluator runtime adapters and judger plumbing, Docker/WSL swe-bench and terminal-bench container internals, the LLM attributor/planner/executor seams, `github_cli`/`skill_source_manager`, `sys.modules` extension loading, live stage orchestration (the caller's `HarnessRunner`) |
| `openjiuwen-agent-evolving` | PPO/verl/ray training (GPU stack), the agent_rl proxy/judge gateway, the RL YAML ecosystem, vendor browser/PIL reward checks, the evaluator pipeline's agent adapters and evolution loop, and a wholesale port of Python's unit suite |
| `openjiuwen-agent-teams` | ZeroMQ/WebSocket/Pulsar transports, PostgreSQL/MySQL stores and the distributed workspace transport, OTLP span export beyond the team-memory bridge (member/round/task spans stay in-memory), the monitor TUI/DB-backed readers, attach security beyond a `0600` socket, vendor CLI harness translation, cross-process org federation, LLM-driven context compaction and `IntelliRouter`, the task-board→memory bridge |
| `openjiuwen-foundation` | pgvector/postgres/gauss vector backends and every external graph backend (typed `Unsupported`), Redis TLS/cluster and one-connection serialisation, Milvus/Chroma/Elasticsearch behavioural limits, the MCP Playwright transport and legacy-SSE replay, OpenAPI beyond local `$ref`, Anthropic non-streaming thinking blocks, TPM/RPM enforcement, OAuth login/status/logout and cross-process file lock, DashScope streaming media, and the Responses gaps (`previous_response_id`, background mode, hosted tools, encrypted-reasoning replay) |
| `openjiuwen-multiagent` | the Pulsar transport, Redis cluster/Sentinel topologies, stream housekeeping (`XTRIM`)/`XPENDING` fallback, Python's `drunner`/`RemoteAgent` factories and reply-topic routing, and the distributed team event bus |
| `openjiuwen-runner` | distributed mode (Pulsar/ZeroMQ/MQ envelopes/`RemoteAgent`, `RemoteProtocol::A2a`), Python's `python -m` child bootstrap and the process pool, team-pool extras (OS-process/external-CLI/SSH members, cross-process attach, monitor TUI, org layer), in-loop HITL gate enforcement and answering auth, monitor persistence/exports, resource-manager and callback-framework extras, `ReadWriteLockManager`, and spawner introspection/live deadline re-tuning |
| `openjiuwen-macros` | sync and generator tool handlers and `Args:` doc-block parameter descriptions (handlers are async-only; `schemars` owns descriptions) |
| `openjiuwen-cli` | non-dry `auto-harness run` and `verify-ext` (no RSI evaluation runner or extension loader), `team`'s `invoke_tool` op, MCP/OTel-backed harness surfaces, and execution-journal resume for the interactive REPL (it seeds history but cannot adopt the persisted session id) |

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

Notable cross-layer edges, each justified in the consuming crate's `CONTEXT.md`:

- `openjiuwen-memory` (layer 2) → `openjiuwen-sysop` (layer 1): advisory file locks for local coding memory.
- `openjiuwen-runner` (layer 4) → `openjiuwen-memory` (layer 2): the runner owns the one dream-loop task so the memory crate stays spawn-free.
- `openjiuwen-agent-teams` (layer 6) → `openjiuwen-memory` (layer 2), and optionally → `openjiuwen-extensions` (layer 5) behind its `otlp` feature for the team-memory → OTel bridge.
- `openjiuwen-cli` (layer 7) → `openjiuwen-harness` with `notify`: the CLI's `watch` command and `chat --watch`.
- `openjiuwen-harness-providers` (layer 6) → `openjiuwen-agent-teams` (same layer 6) behind `native-v2` is the only same-layer edge; it is off by default and enabled by nothing here.

## Constraints

- A crate may depend only on crates in lower layers (see above); no cycles. `openjiuwen-runner` is driven by upper layers through their public seams only.
- `openjiuwen-core` and `openjiuwen` are umbrella crates; concrete crates must not depend on them except the harness layer and above.
- Shared types belong in `openjiuwen-common` or `openjiuwen-foundation`, never duplicated across crates.
- Crates are `publish = false` until the API stabilizes.
- Test-only crates are never dependencies of library crates; `openjiuwen-examples` uses `openjiuwen` (umbrella) plus dev-dependency crates only.
- Two binaries live outside `openjiuwen-cli`: `team-member` and `openjiuwen-team-mcp` (`required-features = ["mcp"]`) in `openjiuwen-agent-teams`.
- Request signing is hand-rolled over `sha2` in exactly three places — `openjiuwen-foundation::store::external::s3::sigv4` (AWS SigV4), `openjiuwen-security::backends::vendors::aliyun::signing` (ACS3-HMAC-SHA256) and `openjiuwen-symphony::object_artifact::http::hmac_sha256` (Azure Blob SharedKey); no workspace crate depends on the RustCrypto `hmac` crate (`hmac 0.12.1` is only a transitive lock entry via `sqlx-postgres`), and GCS service-account signing plus Azure AD client credentials remain typed `Error::Unsupported` (Azure SharedKey, SAS and anonymous access work).
- The only real XML parser is the in-crate tolerant one at `openjiuwen-retrieval::parsing::xml` (re-exported as `parsing::parse_xml` and `openjiuwen_retrieval::parse_xml`); no XML crate (`quick-xml`, `roxmltree`, `xml-rs`, …) is a dependency or a `Cargo.lock` entry, and `openjiuwen-rsi`'s JUnit reader plus `openjiuwen-dev-tools`' tag helpers are hand-rolled scanners.
- `openjiuwen-foundation` and `openjiuwen-retrieval` are both layer 1, so foundation cannot reuse that XML parser; upper layers reach it through the `openjiuwen_core::retrieval` umbrella alias.
- `openjiuwen-harness` and `openjiuwen-harness-providers` are non-optional umbrella dependencies taken with default features, so their `native` default always applies.
- Several `Unsupported` reason strings still claim a dependency the workspace does not ship while the dependency is in fact already locked (`rustls`/`tokio-rustls` for `rediss://` in foundation/extensions/session/agent-evolving/agent kv-cache; `regex` and `unicode-normalization` for tokenizer nodes in context), so those gaps are offline-implementable rather than blocked.
- Capability deferrals are individually enumerated in each crate's `CONTEXT.md`; the crate-level table above is a summary, not the source of truth.

## Feature Flags

Default builds need no network and no external service; every feature except the ones marked *default* is additive and opt-in.
`openjiuwen-agent`, `openjiuwen-auto-harness`, `openjiuwen-common`, `openjiuwen-context`, `openjiuwen-dev-tools`, `openjiuwen-graph`, `openjiuwen-harness-protocol`, `openjiuwen-macros`, `openjiuwen-multiagent`, `openjiuwen-retrieval`, `openjiuwen-rsi`, `openjiuwen-runner`, `openjiuwen-security`, `openjiuwen-test-support` and `openjiuwen-workflow` declare `default = []` and no other feature (15 crates).

| Crate | Feature | Default | Effect |
| --- | --- | --- | --- |
| `openjiuwen-foundation` | `mcp` | yes | MCP client/server tooling (pulls `rmcp`) |
| `openjiuwen-foundation` | `tiktoken` | yes | `cl100k_base` token counting |
| `openjiuwen-foundation` | `sqlite` | no | SQLite storage backends |
| `openjiuwen-foundation` | `live-tests` | no | empty opt-in test gate (no test reads it) |
| `openjiuwen-session` | `sqlite` | no | SQLite checkpointer |
| `openjiuwen-session` | `otel` | no | empty marker that enables the real trace-sink installation path |
| `openjiuwen-memory` | `sqlite` | no | SQLite memory persistence (and FTS5 coding index; enables `openjiuwen-foundation/sqlite`) |
| `openjiuwen-sysop` | `sandbox` | no | sandbox provider transport (`reqwest`) |
| `openjiuwen-harness` | `native` | yes | native in-process DeepAgent harness |
| `openjiuwen-harness` | `browser`, `cua`, `mobile` | no | documented `Unsupported` preset stubs — agents deferred |
| `openjiuwen-harness` | `notify` | no | compiles the `notify`-backed `watcher::WorkspaceWatcher` intake; without it the watcher API still compiles and `start()` returns `Error::Unsupported` |
| `openjiuwen-harness` | `live-tests` | no | empty opt-in test gate (no test reads it) |
| `openjiuwen-harness-providers` | `native` | yes | `native` provider (in-process DeepAgent harness) |
| `openjiuwen-harness-providers` | `vendor` | no | registers `claudecode`/`codex`/`dsh`, which drive an operator-installed CLI through `vendor_cli` |
| `openjiuwen-harness-providers` | `native-v2` | no | adds the `native_v2` provider hosted by `openjiuwen-agent-teams` |
| `openjiuwen-harness-providers` | `live-tests` | no | empty opt-in test gate (no test reads it) |
| `openjiuwen-agent-teams` | `mcp` | yes | the `openjiuwen-team-mcp` binary and stdio MCP server |
| `openjiuwen-agent-teams` | `distributed` | no | distributed `Messager` seams (reject at construction) |
| `openjiuwen-agent-teams` | `sqlite` | no | `SqliteTeamStore` (`sqlx`) |
| `openjiuwen-agent-teams` | `otel` | no | in-memory/tracing OTel seam (the exporter is in `openjiuwen-extensions`) |
| `openjiuwen-agent-teams` | `otlp` | no | the team-memory → OTel bridge (`observability::{OtlpMemoryObserver, otlp_memory_observer}`), pulling `openjiuwen-extensions` with its `otel` sink |
| `openjiuwen-agent-teams` | `live-tests` | no | empty opt-in test gate (no test reads it) |
| `openjiuwen-extensions` | `a2a` | no | A2A client/server, wire types, REST binding, harness bridge |
| `openjiuwen-extensions` | `otel` | no | real OTel SDK provider, sink, handlers, tracing layer, OTLP/HTTP export in JSON and protobuf, Grafana/Langfuse presets, the file exporter |
| `openjiuwen-extensions` | `mcp` | no | MCP adapter surface (`rmcp`) |
| `openjiuwen-extensions` | `sqlite` | no | SQLite checkpointer/store backends |
| `openjiuwen-extensions` | `redis` | no | Redis KV/checkpointer adapters plus the document and vector stores (`rediss://` TLS and cluster stay `Unsupported`) |
| `openjiuwen-extensions` | `pulsar` | no | Pulsar queue surface — construction returns `Error::Unsupported` |
| `openjiuwen-extensions` | `live-tests` | no | empty opt-in test gate (no test reads it) |
| `openjiuwen-symphony` | `sqlite` | no | SQLite capability-asset store (`store::KvAssetStore::sqlite`) |
| `openjiuwen-agent-evolving` | `redis` | no | `online::RedisOnlineStore` over `openjiuwen-extensions`' RESP client (the RL online state machine) |
| `openjiuwen-core` | `otel` | no | forwards `openjiuwen-session/otel`, so `core::session::install_trace_sink` really installs the supplied sink |
| `openjiuwen-core` | `sqlite` | no | forwards the foundation/session/memory SQLite backends |
| `openjiuwen-core` | `sandbox` | no | forwards `openjiuwen-sysop/sandbox` |
| `openjiuwen-core` | `live-tests` | no | re-forwards `openjiuwen-foundation/live-tests` |
| `openjiuwen` | `native` | yes | forwards `harness/native` + `harness-providers/native` |
| `openjiuwen` | `native-v2` | no | forwards `harness-providers/native-v2` |
| `openjiuwen` | `vendor` | no | forwards `harness-providers/vendor` (registers the three vendor CLIs) |
| `openjiuwen` | `browser`, `cua`, `mobile` | no | forwards the matching `harness` preset stubs |
| `openjiuwen` | `notify` | no | forwards `harness/notify` (the workspace watcher) |
| `openjiuwen` | `sqlite` | no | forwards the foundation/session/memory/extensions/symphony/agent-teams SQLite backends |
| `openjiuwen` | `a2a` | no | forwards `openjiuwen-extensions/a2a` |
| `openjiuwen` | `mcp` | no | forwards `openjiuwen-extensions/mcp` |
| `openjiuwen` | `redis` | no | forwards `openjiuwen-extensions/redis` + `openjiuwen-agent-evolving/redis` |
| `openjiuwen` | `pulsar` | no | forwards `openjiuwen-extensions/pulsar` |
| `openjiuwen` | `distributed` | no | forwards `openjiuwen-agent-teams/distributed` |
| `openjiuwen` | `otel` | no | forwards extensions/kernel/agent-teams `otel` |
| `openjiuwen` | `otlp` | no | forwards `openjiuwen-agent-teams/otlp` (independent of `otel`, mirroring that crate's split) |
| `openjiuwen` | `sandbox` | no | forwards `openjiuwen-sysop/sandbox` |
| `openjiuwen` | `live-tests` | no | forwards the foundation/core/harness/providers/teams/extensions live gates |
| `openjiuwen-cli` | `live-tests` | no | empty opt-in test gate (no test reads it) |
| `openjiuwen-e2e` | `a2a` | no | enables `tests/a2a_roundtrip.rs` |
| `openjiuwen-e2e` | `otel` | no | enables the extensions/teams OTLP path and `tests/plugin_observability.rs` |
| `openjiuwen-e2e` | `sqlite` | no | enables the umbrella and agent-teams SQLite backends for `tests/memory_dreaming.rs` and `tests/team_backend.rs` |
| `openjiuwen-examples` | `a2a` | no | example 11 (`a2a_server_client`) |
| `openjiuwen-examples` | `mcp` | no | marker only — MCP is always compiled through `openjiuwen-foundation`'s default |

Note: nothing in the workspace enables `vendor`, so a consumer wanting vendor providers must depend on `openjiuwen-harness-providers` directly.
`openjiuwen-cli` enables `openjiuwen-harness/notify` unconditionally, so the binary's `watch` and `chat --watch` commands are always compiled.
`openjiuwen-agent-teams` keeps `otel` (in-memory seam) and `otlp` (extensions bridge) as separate features; the umbrella mirrors that split.
`openjiuwen-core`'s four forwarders cover the entire gated kernel surface, and the umbrella forwards them plus the product-layer features.

## Test Conventions

- Unit tests live beside the module (`#[cfg(test)] mod tests`) and are documented in that module's `CONTEXT.md`.
- Integration tests live in `crates/<crate>/tests/` as one binary per file (394 `tests/*.rs` files today; `openjiuwen` keeps only `tests/module_paths.rs`, `openjiuwen-core` adds `tests/feature_forwarding.rs`, and `openjiuwen-examples/tests/examples_offline.rs` runs every example binary offline).
- Every crate has a `tests/` directory (30 of 30), with scenario-specific helpers in a nested `tests/support/mod.rs` that is not itself a suite.
- Cross-crate scenarios live only in `openjiuwen-e2e/tests/`: 26 scenario binaries holding 125 test functions, with bounded helpers in `tests/support/mod.rs`.
- 124 e2e scenarios pass with `--features sqlite,a2a,otel` and 114 with default features; exactly one is a documented `#[ignore]` (`cli_binary.rs`'s process-level binary-resolution test).
- Workspace-wide there are five `#[ignore]` attributes: that e2e case, three hermetic protocol-child helpers in `openjiuwen-runner` (re-executed by their own suites, so the attribute is load-bearing), and one live-sandbox-gateway probe in `openjiuwen-sysop`.
- `openjiuwen-test-support` supplies the shared fakes: loopback `MockLlmServer`, `ScriptedModel`, temp-workspace/env fixtures, JSON assertions, eventual-consistency helpers.
- Everything is offline and deterministic by default: scripted models or the loopback server, temp directories, loopback TCP/Unix sockets, no credentials, no wall-clock ordering, bounded awaits (10 s `BOUND` in `openjiuwen-e2e`).
- The `live-tests` features (`foundation`, `harness`, `harness-providers`, `agent-teams`, `extensions`, `cli`, `openjiuwen`, `core`) are declared empty gates that no test or code path reads (only `openjiuwen-core/tests/feature_forwarding.rs` compile-checks their forwarding); suites that need an optional capability are gated by `#![cfg(feature = "…")]` (`a2a`, `otel`, `otlp`, `sqlite`, `redis`, `pulsar`, `vendor`, `native-v2`, `notify`) or by `#![cfg(unix)]`, and the single live-network test is `#[ignore]`d.
- `cargo test --workspace` must pass offline; each crate additionally documents its own definition of done (`cargo fmt`, `cargo clippy -- -D warnings`).
- Deferred areas are proven, not just documented: dedicated suites assert `Error::Unsupported` and its message (`openjiuwen-dev-tools/tests/deferred_surface.rs`, `openjiuwen-symphony/tests/deferred.rs`, `openjiuwen-rsi/tests/deferred.rs`, `openjiuwen-agent-evolving/tests/rl.rs`, `openjiuwen-sysop/tests/sandbox_stub.rs`).
- `cargo doc --workspace --no-deps` (and `--all-features`) must stay warning-free; link private or feature-gated items as plain code spans instead of widening visibility, and never add crate-level `rustdoc::` allows.
- Module-level (`//!`) docs resolve intra-doc links in the parent module's scope, so links there usually need an explicit `crate::…` path.
- `crates/openjiuwen-cli`'s `[[bin]] name = "openjiuwen"` sets `doc = false`: its rustdoc output name would otherwise collide with the `openjiuwen` umbrella crate's `target/doc/openjiuwen/`.
- Feature matrixes are verified by enabling each gate offline: `--all-features` and `--no-default-features` builds are part of each crate's definition of done where it declares features.

## Routing Table

### Kernel core (layers 0–4)

- `./openjiuwen-common/` → Card/Config schema, typed errors, logging, env/config, HTTP helpers, task manager, utilities
- `./openjiuwen-macros/` → the `#[tool]` attribute macro (`async fn` + arg struct → `ToolCard` + `Tool` impl)
- `./openjiuwen-foundation/` → LLM clients (OpenAI-compatible/Anthropic/Responses/DashScope, OAuth, external providers), prompts/messages, tools + MCP, storage backends
- `./openjiuwen-session/` → sessions, state scopes, streaming, interactions, checkpointers, disk store, VCS history, tracer
- `./openjiuwen-graph/` → Pregel/BSP graph execution engine
- `./openjiuwen-retrieval/` → documents, parsing (incl. PPTX/ODT and byte loading), chunking, embeddings, vector stores, BM25/retrievers, rerankers, query rewriting, knowledge base
- `./openjiuwen-sysop/` → local/sandbox filesystem, shell (PTY, flock), and code operations
- `./openjiuwen-workflow/` → workflow builder, built-in components, `${node.field}` data flow
- `./openjiuwen-context/` → context windows, token accounting/usage telemetry, native tokenizers and artefacts, compressor/offload processors, rule compression, adaptive ladders, reinjection
- `./openjiuwen-memory/` → long-term/entity-graph memory, external providers, migrations/backup/locks, analyzer, lite coding memory, dreaming, team memory
- `./openjiuwen-security/` → guardrails, moderation backends and content-safety pipeline
- `./openjiuwen-agent/` → `Agent` seam, `ReActAgent`, abilities, rails, interrupts, controller, KV cache, memory provider, usage, operators/evolve, prompt packs, `LLMAgent`, `WorkflowAgent`
- `./openjiuwen-multiagent/` → `BaseTeam`, team runtime, message bus (in-process + Redis Streams), routing/inbox, strategies
- `./openjiuwen-runner/` → `Runner`, `ResourceMgr`, callbacks, MQ stats, spawn seam/registry, pool, HITL, monitor, dream service
- `./openjiuwen-core/` → kernel umbrella module path table over the crate above

### Harness

- `./openjiuwen-harness-protocol/` → provider-neutral harness SPI (traits, events, results, streams)
- `./openjiuwen-harness/` → DeepAgent task loop, rails, workspace, permissions, tools, LSP, worktree, cron, watcher, subagents, skills, persistent agent, prompts, native protocol, CLI logic
- `./openjiuwen-harness-providers/` → provider registry/factory, `native` and `native_v2` providers, `HarnessIOAdapter`, vendor CLI adapters
- `./openjiuwen/` → top-level umbrella re-exporting every public module tree and the `prelude`

### Teams and self-improvement

- `./openjiuwen-agent-teams/` → leader/teammate runtime, worktrees, team-memory wiring/OTLP observer, SQLite store, org layer, attach, HITL, skill CLI/MCP server
- `./openjiuwen-rsi/` → recursive self-improvement: `auto_harness` loop, `program_opt`, evolution protocol, `runtime` infrastructure
- `./openjiuwen-auto-harness/` → re-export shim over `rsi::auto_harness`
- `./openjiuwen-agent-evolving/` → evaluation/judging/optimisation/trainer, evolution subsystems and the Redis-backed online RL service, with the GPU engines deferred

### Integrations

- `./openjiuwen-extensions/` → A2A, OTel tracer (OTLP/HTTP JSON + protobuf, Grafana, Langfuse), observability utilities, checkpointer/store/MQ/sysop adapters (feature-gated)
- `./openjiuwen-symphony/` → offline capability-asset engine (registry/search/graph/`SkillFs`/flow codegen/planner/executor/evaluation/experience/observation/retrieval)

### Tooling and CLI

- `./openjiuwen-cli/` → binary `openjiuwen` (`run`, `chat`, `lsp`, `cron`, `team`, `watch`, `sessions`, `auto-harness`, with `run`/`chat` `--resume`/`--continue`)
- `./openjiuwen-dev-tools/` → prompt builder, skill creator/evaluator, agent builder, tune optimizers

### Test infrastructure

- `./openjiuwen-test-support/` → mock LLM server, fakes, fixtures
- `./openjiuwen-e2e/` → cross-crate integration scenarios (26 binaries, 125 scenarios)

### Examples

- `./openjiuwen-examples/` → 26 runnable examples mirroring the Python `examples/` tree
