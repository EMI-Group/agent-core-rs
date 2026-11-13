# teams-extensions

## Matrix rows
| Python path | Rust crate | Status | Notes |
|---|---|---|---|
| `agent_teams` | `openjiuwen-agent-teams` (feature `mcp` default; `sqlite`/`otel`/`distributed` optional) | Subset | Team runtime, SQLite store, process/external-CLI/SSH members, git-worktree workspaces, org layer, model routing, KV affinity, monitor stream and Unix-socket attach; pyzmq/Pulsar transports, PostgreSQL/MySQL stores, remote workspace dispatch and OTLP export are deferred. |
| `extensions.a2a` | `openjiuwen-extensions` (feature `a2a`) | Subset | A2A client/server over JSON-RPC and SSE plus the REST `HTTP+JSON` binding, agent-card adapter and harness bridge; only the gRPC binding is deferred. |
| `extensions.tracer_otel` | `openjiuwen-extensions` (feature `otel`) | Subset | SDK provider, `TraceSink`, agent/workflow handlers, OTLP/HTTP JSON exporter, Grafana Cloud and Langfuse presets; OTLP/gRPC, `http/protobuf` and the local Grafana Agent push are deferred. |
| `extensions.checkpointer`/`store`/`message_queue`/`sys_operation`/`context_evolver`/`vendor_specific` | `openjiuwen-extensions` (features `sqlite`/`redis`/`pulsar`) | Subset | Factories over kernel seams with live Redis KV+checkpointer, Elasticsearch vectors, AIO/JiuwenBox/YuanRong sandboxes, DashScope rerank and deterministic+LLM evolvers; Pulsar, Redis TLS/cluster, Postgres/Gauss document stores are deferred. |

## Deferred (still unimplemented)
- `openjiuwen-agent-teams` — ZeroMQ/pyzmq, hybrid-WebSocket and Pulsar transports; PostgreSQL/MySQL team stores; the distributed remote-workspace transport (`RemoteDispatch` seam only); OTLP span export (the `otel` feature is an in-memory seam); the monitor TUI and DB-backed monitor readers; attach authentication/TLS/named pipes; cross-process organisation federation; LLM-driven fork/compact summaries and `IntelliRouter` selection (external service, platform automation).
- `openjiuwen-extensions` — the Pulsar message-queue transport; Redis `rediss://` TLS, cluster slot routing and the Redis document/vector surfaces; the Elasticsearch and Gauss/PostgreSQL document stores; OTLP/gRPC and `http/protobuf` export; the local Grafana Agent push; the A2A gRPC binding; Milvus-backed context-evolver scoring; remote-container teardown and conditional writes in the sandbox providers (external service, ML stack).

## Error::Unsupported surface
- `openjiuwen-agent-teams`: `messager::create_messager`/`messager::distributed` — `pyzmq`/hybrid rejects at construction and the Pulsar/ZeroMQ paths name the deferred transport.
- `openjiuwen-agent-teams`: `storage::open_store` — `postgresql`/`mysql` and unknown kinds report the unimplemented `TeamStore`.
- `openjiuwen-agent-teams`: `workspace::worktree_remote::{UnavailableRemoteDispatch, RemoteWorkspaceAdapter::unavailable}` and `UnsupportedWorktreeBackend` — no distributed workspace transport ships, and a host may refuse worktrees outright.
- `openjiuwen-agent-teams`: `attach` (`AttachServer::bind`/`AttachClient::connect` and typed wrappers) — non-Unix targets, because the wire is Unix domain sockets only.
- `openjiuwen-agent-teams`: vendor CLI harness translation — `claudecode`/`codex`/`dsh` stay unsupported in `openjiuwen-harness-providers`; the crate ships only the generic stdio child.
- `openjiuwen-extensions`: `message_queue::pulsar` — every entry point returns `PULSAR_DEFERRED_REASON`; no protobuf-framed broker transport.
- `openjiuwen-extensions`: `store` (`build_kv_store`/`build_document_store`/`build_vector_store`) — backend/operation pairs with no surface (Elasticsearch KV or document, Redis document or vector, SQLite vector) and a disabled `sqlite`/`redis` feature.
- `openjiuwen-extensions`: `checkpointer::redis` — `rediss://` TLS and cluster slot routing need a vendored TLS stack and MOVED/ASK following.
- `openjiuwen-extensions`: `otel::grafana` (`GrafanaAgentConfig`) — the local Agent/Alloy push requires an external collector.
- `openjiuwen-extensions`: `otel::env::parse_protocol`/`otel::handle` — OTLP `grpc` and `http/protobuf` need `tonic`/`prost`, which are not available offline.
- `openjiuwen-extensions`: `a2a::server::service` — a gRPC `protocol_binding` has no transport.
- `openjiuwen-extensions`: `sys_operation` — unknown sandbox types (`remote_provider_set`/`register_remote`) and `SandboxProviderSet::deferred` name the supported alternatives.
- `openjiuwen-extensions`: `context_evolver::UnsupportedTaskMemoryService` — the documented deferred `TaskMemoryService` selection.

## Corrections to other PARITY.md sections
- Deferred Features → "Harness, teams, and providers" (`openjiuwen-agent-teams`): current text says "ZeroMQ/Pulsar transports, SQLite/PostgreSQL/MySQL stores, process/external-CLI/SSH members, git-worktree and remote workspace backends, the organisation layer, swarmflow, fork/compact, KV-cache affinity, OTel, monitor/TUI, model-router allocation, and cross-process attach"; should say "pyzmq/hybrid/Pulsar transports, PostgreSQL/MySQL stores, the distributed remote-workspace transport, OTLP span export, the monitor TUI, attach authentication/TLS, cross-process organisation federation, LLM-driven fork/compact summaries, and `IntelliRouter` selection".
- Deferred Features → "Harness, teams, and providers" (`openjiuwen-extensions`): current text says "Redis checkpointer/store, Elasticsearch/Postgres/Gauss stores, Pulsar message queue, ZeroMQ transports, OTLP span export, DashScope reranker HTTP transport, AIO/JiuwenBox/YuanRong sandbox providers, LLM-driven context-evolver algorithms, A2A gRPC/REST variants, and Grafana/Langfuse integrations"; should say "the Pulsar message queue, Redis TLS/cluster routing and its document/vector surfaces, the Elasticsearch and Gauss/PostgreSQL document stores, OTLP/gRPC and `http/protobuf` export, the local Grafana Agent push, the A2A gRPC binding, Milvus-backed context-evolver scoring, and sandbox remote-container teardown".
- `Error::Unsupported` surface (`openjiuwen-agent-teams`): current text says "distributed transports, external stores, process/CLI/SSH members, and non-local workspace backends"; should say "pyzmq/hybrid/Pulsar transports, PostgreSQL/MySQL stores, the remote workspace transport, non-Unix attach, and vendor CLI harness translation".
- `Error::Unsupported` surface (`openjiuwen-extensions`): current text says "Redis/Elasticsearch/Postgres/Gauss/Pulsar/AIO/JiuwenBox/YuanRong backends, the LLM context evolver, and the DashScope reranker transport"; should say "the Pulsar queue, Redis TLS/cluster routing and its document/vector surfaces, the Elasticsearch and Gauss/PostgreSQL document stores, OTLP/gRPC export, the local Grafana Agent push, deferred sandbox types, a gRPC A2A binding, and the opt-out context-evolver service".

## ARCHITECTURE.md corrections
- §5 Feature flags (`openjiuwen-agent-teams`): current text says "`mcp` (default), `distributed` (empty marker), `live-tests`"; should say "`mcp` (default), `sqlite` (team store), `otel` (observability hooks over the in-memory seam), `distributed` (transport stubs), `live-tests`".
- §5 Feature flags (`openjiuwen-extensions`): current text says "`a2a`, `otel`, `sqlite`, `redis`, `pulsar`, `mcp`, `live-tests` (all optional)"; should say "`a2a`, `otel`, `sqlite`, `redis`, `pulsar`, `mcp`, `live-tests` (all optional), with live Elasticsearch vector and AIO/JiuwenBox/YuanRong sandbox transports always compiled in".
