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
| `openjiuwen-harness-protocol` | 0 | provider-neutral harness SPI: traits, events, turn results, checkpoints, streams, wire helpers | Implemented |
| `openjiuwen-foundation` | 1 | messages/prompts, LLM clients and `Model`, tools + MCP, storage backends | Implemented |
| `openjiuwen-session` | 1 | sessions, scoped state, streaming frames, HITL interactions, checkpointers, tracer | Subset |
| `openjiuwen-graph` | 1 | Pregel/BSP graph execution engine (builder, supersteps, routing, resume, Mermaid) | Implemented |
| `openjiuwen-retrieval` | 1 | documents, chunking, embeddings, in-memory vector store, rerank seams, knowledge base | Subset |
| `openjiuwen-sysop` | 1 | local filesystem/shell/code operations, sandbox provider registry | Subset |
| `openjiuwen-workflow` | 2 | workflow builder, components, `${node.field}` data flow, run engine | Subset |
| `openjiuwen-context` | 2 | context windows, token accounting, compression/offload processors | Subset |
| `openjiuwen-memory` | 2 | long-term, entity/graph, and lite coding memory | Subset |
| `openjiuwen-security` | 2 | guardrails, detection pipeline, content rules, PII redaction, tool-arg validation | Subset |
| `openjiuwen-agent` | 3 | `Agent` seam, `ReActAgent`, abilities, rails, interrupts, `LLMAgent`, `WorkflowAgent` | Subset |
| `openjiuwen-multiagent` | 3 | `BaseTeam`, team runtime, message bus, handoff/supervisor/agents-as-tools strategies | Implemented |
| `openjiuwen-runner` | 4 | `Runner`, `ResourceMgr`, callback framework, in-memory message queue, spawn/team seams | Implemented |
| `openjiuwen-core` | 4 | kernel umbrella: Python-shaped module path table over the concrete kernel crates | Implemented |
| `openjiuwen-harness` | 5 | DeepAgent task loop, workspace, permissions, coding tools, rails, prompts, subagents, skills, native protocol impl, CLI logic | Subset |
| `openjiuwen-extensions` | 5 | A2A, OTel tracer, checkpointer/store/MQ/sysop adapters, `context_evolver` | Subset |
| `openjiuwen-dev-tools` | 5 | prompt builder, skill creator/evaluator, agent builder, tuning helpers | Subset |
| `openjiuwen-symphony` | 5 | capability-asset engine: registry, search, capability graph, planner, executor, runtime | Subset |
| `openjiuwen-harness-providers` | 6 | provider registry/factory, `native` provider, `HarnessIOAdapter`, skill installer, vendor adapters | Subset |
| `openjiuwen-rsi` | 6 | `auto_harness` self-optimisation loop, `program_opt` PUCT search, evolution protocol | Subset |
| `openjiuwen-agent-evolving` | 6 | eval/scoring, optimizer/checkpoints, bounded trainer, evolution rails, HTTP service | Subset |
| `openjiuwen-auto-harness` | 6 | pure facade over `openjiuwen-rsi::auto_harness` | Implemented |
| `openjiuwen-agent-teams` | 6 | in-process leader/teammate runtime, task board, mailbox, HITL, skill CLI + MCP server | Subset |
| `openjiuwen` | 7 | top-level umbrella: module aliases for every public tree plus a curated `prelude` | Implemented |
| `openjiuwen-cli` | 7 | binary `openjiuwen` (`run`, `chat`) — thin process boundary over `openjiuwen_harness::cli` | Implemented |
| `openjiuwen-test-support` | dev | loopback mock LLM server, scripted model, fixtures, JSON/eventual helpers | Implemented |
| `openjiuwen-e2e` | dev | cross-crate scenario suite (8 test binaries) | Implemented |
| `openjiuwen-examples` | dev | 12 runnable examples mirroring the Python `examples/` tree | Implemented |

## Status Legend

- **Implemented** — the Python area the crate targets is covered end to end; remaining differences are internals deliberately not transliterated, or scope documented as out-of-scope elsewhere.
- **Subset** — a documented portion of the Python area is ported and working; the crate marks the unported remainder as deferred (trait seam, `Error::Unsupported`, or a documented gap section).
- **Stub** — API-shaped surface only, entry points return `Error::Unsupported`; applies to *areas* inside crates (tune optimizers, RL optimizers, `symphony::deferred`, vendor harness providers, agent-evolving judge/RL), never to a whole crate.
- **Deferred** — not implemented at all and documented as such (browser/CUA/mobile harness agents, LSP, distributed transports, external vector databases, OTLP export); usually reached only through a trait seam.

Counts at crate level: **14 Implemented, 16 Subset, 0 Stub, 0 Deferred**.
Some Implemented crates still document out-of-scope integrations in their own `CONTEXT.md` (for example `openjiuwen-foundation` ships no vendor vector/database stores and no DashScope media client, and `openjiuwen-multiagent` is in-process only); those are backend integrations rather than missing public surface, which is why the crate-level status stays Implemented.

Deferred areas per crate (the 16 Subset crates, plus the seam-level deferrals inside the otherwise Implemented `openjiuwen-runner`):

| Crate | Deferred areas |
| --- | --- |
| `openjiuwen-session` | Redis/remote checkpointer, disk-backed `session_controller`, `vcs/`, pickle checkpoint interop, OTel (lives in `openjiuwen-extensions`), Python's sub-workflow/team-session state sharing |
| `openjiuwen-retrieval` | Milvus/Chroma/PGVector/Elasticsearch stores, PDF/DOCX/XLSX parsing, triple/graph extraction, multimodal documents, query rewriting, BM25/sparse and agentic retrievers |
| `openjiuwen-sysop` | Sandbox launcher lifecycle (`create/pause/resume/delete`), endpoint discovery, `aio`/`jiuwenbox`/`yuanrong` provider transports (registry seam only) |
| `openjiuwen-workflow` | `SubWorkflowComponent`, `Questioner`/`IntentDetection`/`ReActAgentComp`/`HTTPRequest` components, streaming-actor live interleaving, `draw()` PNG/SVG |
| `openjiuwen-context` | Tokenizer registry/model-specific BPE parity, KV-cache telemetry, provider-specific window policies, most of the Python compressor family, forked-processor registry and session-memory reinjection |
| `openjiuwen-memory` | LLM conflict checker, Markdown/FTS5/`sqlite_vec` coding index, migrations, dreaming, external providers, scope-level model config |
| `openjiuwen-security` | Vendor/cloud moderation transports, real LLM prompt-injection judge (Python's rule-backend stub semantics kept), local model runtime |
| `openjiuwen-agent` | Legacy controller machinery (`ControllerAgent`, `LLMController`, `WorkflowController`), intent-detection controllers, multimodal tool results |
| `openjiuwen-harness` | `browser`/`cua`/`mobile` presets and agents, LSP client, `goal`/`heartbeat_rail`/`personal_context`/`moa`/`kv_cache`/`persistent_agent`, progressive tool discovery, web search/fetch, MCP resource tools, worktree/cron, OTel exporter |
| `openjiuwen-extensions` | Redis checkpointer/store, Elasticsearch vector store, Gauss/PostgreSQL document store, Pulsar queue, ZeroMQ transports, OTLP span export, DashScope reranker transport, AIO/JiuwenBox/YuanRong sandbox providers, LLM-driven `context_evolver` pipelines |
| `openjiuwen-dev-tools` | `tune` optimizers/`Trainer`/`optimize_prompt`, agent-builder LLM pipeline internals (clarifier, workflow designer, DL generator/reflector), Mermaid rendering |
| `openjiuwen-symphony` | LLM-guided retrieval, LLM ontology matching, LLM-judged evaluation, ML-backed experience bank, S3 backend, host-side skill filesystem + BM25 content search |
| `openjiuwen-harness-providers` | `native_v2` (wiring lives in `openjiuwen-agent-teams`), vendor `claudecode`/`codex`/`dsh` (registered but `Unsupported` behind `vendor`), manifest/agent-template loading |
| `openjiuwen-rsi` | Agent-descent/ML search, vendor harness SDKs, git/subprocess execution, distributed evolution — `deferred::*` returning `Error::Unsupported` |
| `openjiuwen-agent-evolving` | Offline/online RL optimizers (train/rollout/export), LLM-as-judge metric, HTTP service RL endpoints (stable 501 contract) |
| `openjiuwen-agent-teams` | ZeroMQ/WebSocket/Pulsar transports, sqlite/postgres/mysql team stores, OS-process and external CLI members, SSH transport, git-worktree isolation, cross-process attach, org layer, OTel observability, monitor/TUI |
| `openjiuwen-runner` | Distributed mode (Pulsar/ZeroMQ), reply-topic routing/`RemoteAgent`, real subprocess spawning (`UnsupportedSpawner`), team pools/HITL/monitoring, tags-as-registry-index |

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
`openjiuwen-agent`, `openjiuwen-agent-evolving`, `openjiuwen-auto-harness`, `openjiuwen-common`, `openjiuwen-context`, `openjiuwen-core`, `openjiuwen-dev-tools`, `openjiuwen-graph`, `openjiuwen-harness-protocol`, `openjiuwen-macros`, `openjiuwen-multiagent`, `openjiuwen-retrieval`, `openjiuwen-rsi`, `openjiuwen-runner`, `openjiuwen-security`, `openjiuwen-test-support`, `openjiuwen-workflow` declare `default = []` and no other feature (17 crates).

| Crate | Feature | Default | Effect |
| --- | --- | --- | --- |
| `openjiuwen-foundation` | `mcp` | yes | MCP client/server tooling (pulls `rmcp`) |
| `openjiuwen-foundation` | `tiktoken` | yes | `cl100k_base` token counting |
| `openjiuwen-foundation` | `sqlite` | no | SQLite storage backends |
| `openjiuwen-foundation` | `live-tests` | no | opt-in network tests |
| `openjiuwen-session` | `sqlite` | no | SQLite checkpointer |
| `openjiuwen-memory` | `sqlite` | no | SQLite memory persistence |
| `openjiuwen-sysop` | `sandbox` | no | sandbox provider transport (`reqwest`) |
| `openjiuwen-harness` | `native` | yes | native in-process DeepAgent harness |
| `openjiuwen-harness` | `browser`, `cua`, `mobile` | no | documented `Unsupported` preset stubs — agents deferred |
| `openjiuwen-harness` | `notify` | no | workspace file watcher |
| `openjiuwen-harness` | `live-tests` | no | opt-in network/vendor tests |
| `openjiuwen-harness-providers` | `native` | yes | `native` provider (in-process DeepAgent harness) |
| `openjiuwen-harness-providers` | `vendor` | no | registers `claudecode`/`codex`/`dsh`, each still `Error::Unsupported` |
| `openjiuwen-harness-providers` | `live-tests` | no | opt-in vendor CLI tests |
| `openjiuwen-agent-teams` | `mcp` | yes | the `openjiuwen-team-mcp` binary and stdio MCP server |
| `openjiuwen-agent-teams` | `distributed` | no | distributed `Messager` seams (reject at construction) |
| `openjiuwen-agent-teams` | `live-tests` | no | opt-in live team tests |
| `openjiuwen-extensions` | `a2a` | no | A2A client/server, wire types, harness bridge |
| `openjiuwen-extensions` | `otel` | no | real OTel SDK provider, sink, handlers, tracing layer |
| `openjiuwen-extensions` | `mcp` | no | MCP adapter surface (`rmcp`) |
| `openjiuwen-extensions` | `sqlite` | no | SQLite checkpointer/store backends |
| `openjiuwen-extensions` | `redis` | no | Redis backend surface — construction returns `Error::Unsupported` |
| `openjiuwen-extensions` | `pulsar` | no | Pulsar queue surface — construction returns `Error::Unsupported` |
| `openjiuwen-extensions` | `live-tests` | no | opt-in live integration tests |
| `openjiuwen-symphony` | `sqlite` | no | SQLite capability-asset store (`store::KvAssetStore::sqlite`) |
| `openjiuwen` | `native` | yes | forwards `harness/native` + `harness-providers/native` |
| `openjiuwen` | `otel`, `a2a`, `sqlite` | no | forwards the matching `openjiuwen-extensions` (and symphony `sqlite`) features |
| `openjiuwen` | `live-tests` | no | forwards the harness/providers/teams/extensions live gates |
| `openjiuwen-cli` | `live-tests` | no | opt-in live CLI tests |
| `openjiuwen-e2e` | `a2a`, `otel` | no | enables the A2A round-trip scenario / parity forwarding |
| `openjiuwen-examples` | `a2a` | no | example 11 (`a2a_server_client`) |
| `openjiuwen-examples` | `mcp` | no | marker only — MCP is always compiled through `openjiuwen-foundation`'s default |

Note: `openjiuwen-harness` and `openjiuwen-harness-providers` are non-optional umbrella dependencies taken with default features, so their `native` default always applies; no workspace crate enables `vendor`, and a consumer wanting vendor providers must depend on `openjiuwen-harness-providers` directly.

## Test Conventions

- Unit tests live beside the module (`#[cfg(test)] mod tests`) and are documented in that module's `CONTEXT.md`.
- Integration tests live in `crates/<crate>/tests/` as one binary per file (~176 `tests/*.rs` files today; the umbrellas keep only `tests/module_paths.rs`, `openjiuwen-examples` has none and is validated by compilation plus smoke runs).
- Cross-crate scenarios live only in `openjiuwen-e2e/tests/` (`kernel_flow`, `deep_agent`, `provider_native`, `team_flow`, `a2a_roundtrip`, `symphony_plan`, `self_improvement`, `cli_binary`) with shared bounded helpers in `tests/support/mod.rs`.
- `openjiuwen-test-support` supplies the shared fakes: loopback `MockLlmServer`, `ScriptedModel`, temp-workspace/env fixtures, JSON assertions, eventual-consistency helpers.
- Everything is offline and deterministic by default: scripted models or the loopback server, temp directories, loopback sockets, no credentials, no wall-clock ordering, bounded awaits (10 s `BOUND` in `openjiuwen-e2e`).
- Network/vendor tests are opt-in behind the `live-tests` feature and `OPENJIUWEN_LIVE`.
- `cargo test --workspace` must pass offline; each crate additionally documents its own definition of done (`cargo fmt`, `cargo clippy -- -D warnings`).
- Deferred areas are proven, not just documented: dedicated suites assert `Error::Unsupported` and its message (for example `openjiuwen-dev-tools/tests/deferred_surface.rs`, `openjiuwen-symphony/tests/deferred.rs`, agent-evolving RL stubs, `openjiuwen-sysop/tests/sandbox_stub.rs`).
- `cargo doc --workspace --no-deps` (and `--all-features`) must stay warning-free; link private or feature-gated items as plain code spans instead of widening visibility, and never add crate-level `rustdoc::` allows.
- Module-level (`//!`) docs resolve intra-doc links in the parent module's scope, so links there usually need an explicit `crate::…` path.
- `crates/openjiuwen-cli`'s `[[bin]] name = "openjiuwen"` sets `doc = false`: its rustdoc output name would otherwise collide with the `openjiuwen` umbrella crate's `target/doc/openjiuwen/`.

## Routing Table

### Kernel core (layers 0–4)

- `./openjiuwen-common/` → Card/Config schema, typed errors, logging, env/config, HTTP helpers, task manager, utilities
- `./openjiuwen-macros/` → the `#[tool]` attribute macro (`async fn` + arg struct → `ToolCard` + `Tool` impl)
- `./openjiuwen-foundation/` → LLM clients and `Model`, prompts/messages, tools + MCP, storage backends
- `./openjiuwen-session/` → sessions, state scopes, streaming, interactions, checkpointers, tracer
- `./openjiuwen-graph/` → Pregel/BSP graph execution engine
- `./openjiuwen-retrieval/` → documents, chunking, embeddings, vector stores, rerankers, knowledge base
- `./openjiuwen-sysop/` → local/sandbox filesystem, shell, and code operations
- `./openjiuwen-workflow/` → workflow builder, components, `${node.field}` data flow
- `./openjiuwen-context/` → context windows, token accounting, compression/offload
- `./openjiuwen-memory/` → long-term, entity/graph, and coding memory
- `./openjiuwen-security/` → guardrails and content-safety pipeline
- `./openjiuwen-agent/` → `Agent` seam, `ReActAgent`, abilities, rails, interrupts, `LLMAgent`, `WorkflowAgent`
- `./openjiuwen-multiagent/` → `BaseTeam`, team runtime, message bus, strategies
- `./openjiuwen-runner/` → `Runner`, `ResourceMgr`, callbacks, in-memory MQ, spawn/team seams
- `./openjiuwen-core/` → kernel umbrella module path table over the crate above

### Harness

- `./openjiuwen-harness-protocol/` → provider-neutral harness SPI (traits, events, results, streams)
- `./openjiuwen-harness/` → DeepAgent task loop, rails, workspace, permissions, tools, skills, subagents, prompts, native protocol, CLI logic
- `./openjiuwen-harness-providers/` → provider registry/factory, `native` provider, `HarnessIOAdapter`, vendor `Unsupported` adapters
- `./openjiuwen/` → top-level umbrella re-exporting every public module tree and the `prelude`

### Teams and self-improvement

- `./openjiuwen-agent-teams/` → leader/teammate runtime, worktrees, HITL, skill CLI/MCP server
- `./openjiuwen-rsi/` → recursive self-improvement: `auto_harness` loop, `program_opt`, evolution protocol
- `./openjiuwen-auto-harness/` → re-export shim over `rsi::auto_harness`
- `./openjiuwen-agent-evolving/` → evaluation/optimisation/trainer skeleton with RL/ML stubs

### Integrations

- `./openjiuwen-extensions/` → A2A, OTel tracer, checkpointer/store/MQ/sysop adapters (feature-gated)
- `./openjiuwen-symphony/` → offline capability-asset engine (registry/search/graph/planner/executor/runtime)

### Tooling and CLI

- `./openjiuwen-cli/` → binary `openjiuwen` (`run`, `chat`)
- `./openjiuwen-dev-tools/` → prompt builder, skill creator/evaluator, agent builder, tuning helpers

### Test infrastructure

- `./openjiuwen-test-support/` → mock LLM server, fakes, fixtures
- `./openjiuwen-e2e/` → cross-crate integration scenarios

### Examples

- `./openjiuwen-examples/` → 12 runnable examples mirroring the Python `examples/` tree
