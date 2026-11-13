# openjiuwen-extensions

## Intent

Optional integrations and backend adapters that live outside the kernel: the A2A protocol client/server (JSON-RPC, SSE and REST bindings), OpenTelemetry tracing over the session `TraceSink` SPI with OTLP/HTTP export, and config-driven adapters over the kernel's checkpointer, store, message-queue, and sys-operation seams.
Maps to the Python package `openjiuwen.extensions` (v0.1.20): `a2a/`, `tracer_otel/`, `checkpointer/`, `store/`, `message_queue/`, `sys_operation/`, `context_evolver/`, `vendor_specific/`.
Nothing in the kernel depends on this crate; every heavy dependency and external service sits behind a Cargo feature (or lazily activates only when a caller uses it), so the default build needs no network, no broker and no collector.

## API Surface

Every module is reachable from the crate root; feature-gated modules disappear entirely when their feature is off (no silent stubs), except `otel`, which keeps a documented no-op seam.

- `a2a` (`feature = "a2a"`) — `A2aClient`, `A2aRemoteClient`, `A2aServer`, `A2aServerAdapter`, wire types (`AgentCard`, `Message`, `Part`, `Task`, `TaskState`, …), JSON-RPC envelopes, `A2aTransformer`, `AgentCardAdapter`, `harness_bridge` (A2A ⇄ `openjiuwen_harness_protocol`), and the REST `HTTP+JSON` binding (`A2aTransport::{JsonRpc, Rest, Both}`, `rest_path`/`rest_prefix`, REST routes under `/a2a/rest`, `{"error":{code,status,message}}` envelope).
- `otel` (always compiled) — `OtelTracerConfig` (plus `from_env`/`apply_env`), `semconv` constants, `truncate`/`hash_value`/`Redactor`, span-name builders, and `init_otel_tracer` → `TracerHandle`; with `feature = "otel"` the handle is a real SDK provider with batch export plus `OtelTraceSink`, `OtelAgentHandler`, `OtelWorkflowHandler`, and a `tracing-opentelemetry` layer.
- `otel::otlp` (`feature = "otel"`) — `OtlpHttpExporter` (OTLP/HTTP JSON `POST {endpoint}/v1/traces` on an exporter-owned thread, fail-open, idempotent `shutdown_with_timeout`) and the `OPENJIUWEN_OTEL_*` / standard `OTEL_*` endpoint-header-protocol resolvers.
- `otel::grafana` (`feature = "otel"`) — `GrafanaCloudConfig` OTLP gateway preset (`Authorization: Basic`); the local Agent/Alloy push is a documented `Error::Unsupported`.
- `otel::langfuse` (`feature = "otel"`) — the OTLP preset (`{host}/api/public/otel` + Basic auth) and `LangfuseClient` REST ingestion (`POST /api/public/ingestion`, typed `trace-create`/`span-create`/`generation-create` events, batched).
- `checkpointer` — `build_checkpointer(&CheckpointerConfig)` dispatching to the kernel backends, and with `feature = "redis"` the live `RedisCheckpointer` (`RedisCheckpointerConfig`, `RedisConnectionConfig`, `RedisTTLConfig`, `RedisCheckpointer::{new, connect, create}`) over the in-crate RESP client; TLS and cluster slot routing stay `Error::Unsupported`.
- `redis` (`feature = "redis"`) — the in-crate RESP2/RESP3 client (`Client`, `ConnectionOptions`, `Reply`, `codec::{encode_command, parse_reply}`, `testsupport` fakes) over `tokio::net::TcpStream`.
- `store` — `StoreBackend` + `build_kv_store`/`build_document_store`/`build_vector_store` over `openjiuwen_foundation::store`: live single-node `RedisStore` (feature `redis`, one-byte value kind tag) and live REST `ElasticsearchVectorStore` (k-NN search, `_meta` schema round-trip); Gauss/PostgreSQL document store and Redis document/vector surfaces are not present.
- `message_queue` — `QueueConfig` + `create_queue` over `openjiuwen_runner::mq`, the JSON `QueueBridge`, and with `feature = "pulsar"` the `PulsarConfig`/`PulsarMessageQueue` surface, which validates and then returns `Error::Unsupported` naming the missing protobuf-framed transport.
- `sys_operation` — `SandboxProviderSet` registration against `openjiuwen_sysop::sandbox::provider::SandboxProviderRegistry` (`register`, `register_gateway_provider`, `register_deferred` compatibility no-op) plus the three real remote providers in `sys_operation::remote`: `aio` (kernel gateway envelope), `jiuwenbox` (native `/api/v1` REST) and `yuanrong` (ADX), installed with `register_remote_providers`/`register_remote` and described by `REMOTE_SANDBOX_TYPES`.
- `context_evolver` — schemas and records (`Algo`, `normalize_algo_name`, `AddMemoryRequest`, `MemoryRecord`), per-family prompts and reply parsers, `MemoryStore` persistence (`InMemory`, `JsonFile`), `MemoryVectorStore`, and the `TaskMemoryService` trait with two services: `DeterministicTaskMemoryService` (offline lexical + embedding scoring) and `LlmTaskMemoryService` (model-driven over an injected `MemoryCompletion`, with `OpenAiChatCompletion`), built through `create_task_memory_service`, plus the documented `UnsupportedTaskMemoryService` opt-out.
- `vendor_specific` — deprecated `AliyunReranker` alias with the live DashScope HTTP rerank transport (`from_config`: `POST {api_base}/services/rerank/text-rerank/text-rerank`, `Bearer` auth, score mapping) and an adapter over an injected `openjiuwen_retrieval::Reranker` (`new`).

## Feature Matrix

| Feature | Default | Enables | External requirement | Behavior when off |
|---|---|---|---|---|
| — (default) | yes | `otel` seam (no-op), `checkpointer`, `store` (memory/file, live Elasticsearch), `message_queue` (in-process), `sys_operation` (live `aio`/`jiuwenbox`/`yuanrong` transports), `context_evolver`, `vendor_specific` | none until a remote backend/collector is actually used | n/a |
| `a2a` | no | `axum`, `async-stream`, `bytes`; whole `a2a` module | none (client/server tests use loopback) | module absent |
| `otel` | no | `opentelemetry`, `opentelemetry_sdk`, `tracing-opentelemetry`, `tracing-subscriber`, `chrono`; real provider, OTLP/HTTP JSON exporter, Grafana/Langfuse presets, sink, handlers, tracing layer | an OTLP/HTTP (JSON) collector only for `ExporterType::Otlp`; Memory/Console/None need nothing | `init_otel_tracer` returns an inactive `TracerHandle`; `TracerHandle::is_active() == false` |
| `sqlite` | no | `sqlx`, `openjiuwen-foundation/sqlite`, `openjiuwen-session/sqlite` | a local file/in-memory database | `sqlite` backends answer `Error::Unsupported` |
| `redis` | no | the `redis` client module, `checkpointer::redis`, `store::redis` | a Redis server for live use (tests use an in-process RESP fake) | module absent |
| `pulsar` | no | `message_queue::pulsar` surface | a Pulsar broker (not implemented) | variant absent |
| `mcp` | no | `rmcp` and `tokio-stream` dependencies only; MCP support itself is `openjiuwen_foundation::tool::mcp` (enabled by default in the kernel) | none | no module in this crate is gated on it |
| `live-tests` | no | network/service-gated tests | live services | those tests do not exist |

Remaining gaps (`Error::Unsupported` or documented behavioural limits, never silent degradation): OTLP/gRPC and `http/protobuf` (no `tonic`/`prost` offline), the local Grafana Agent push, `rediss://` TLS and Redis cluster slot routing, Redis document/vector surfaces, the Elasticsearch document store, the Gauss/PostgreSQL document store, the Pulsar transport and ZeroMQ transports, remote-container teardown and conditional writes in the sandbox providers, and the Milvus/vector-store-backed `context_evolver` scoring.

## Constraints

- Only this crate and its `Cargo.toml` change from here; kernel crates are consumed as re-exported through `openjiuwen-core` (`session`, `runner`, `sys_operation`, `retrieval`, `foundation`).
- Async-first: tokio I/O, no blocking calls on async tasks, public streams are `BoxStream<'static, Result<T>>`, cancellation via `tokio_util::sync::CancellationToken`.
- The only sanctioned blocking seam is the OTLP exporter thread: `BatchSpanProcessor` drives `export()` on a plain std thread, so the exporter owns a current-thread runtime plus its job channel and never creates or drops a runtime on the caller's thread.
- Errors: `openjiuwen_common::error::{Error, Result}`; adapters map kernel errors, never panic on user input; unsupported backends return `Error::Unsupported` with the backend name and the supported alternative.
- Serde: public wire types are `#[serde(rename_all = "camelCase")]` where the external protocol is camelCase (A2A, agent cards, OTEL attribute names, Langfuse bodies) and `snake_case` elsewhere; unknown fields are tolerated.
- Logging: `tracing` only (targets `openjiuwen_extensions::*`); no stdout/stderr writes in library code; credentials, payload contents and memories never logged above `debug`.
- Safety: no `unsafe`; no global mutable state beyond the documented Python-compatible process globals (the sandbox container cache keyed by base URL + isolation key).
- Size: files ≤ ~1000 lines; one file per concern; shared helpers (status mapping, SSE framing, redaction, HTTP error mapping) live once at the module root.
- Intentionally large: `src/otel/workflow_handler/tests.rs` and `src/otel/agent_handler/tests.rs` are exhaustive callback/attribute matrices that lose value if split further; `tests/sandbox_providers.rs` is one binary covering three providers × three operation families.
- Feature tests must pass offline: `cargo test -p openjiuwen-extensions` plus `--features a2a`, `--features otel`, `--features sqlite`, `--features redis`, `--features pulsar`, and `--all-features`.

## Test Strategy

- Unit tests live beside the code that owns the behavior: `src/a2a/client/tests.rs`, `src/a2a/transform/tests.rs`, `src/a2a/harness_bridge/tests.rs`, `src/a2a/server/{tests,rest_tests}.rs` plus inline tests in `types.rs`, `jsonrpc.rs`, and `card.rs`.
- OTel unit tests: `src/otel/sink/tests.rs`, `src/otel/sink/tests/lazy_open.rs`, `src/otel/agent_handler/tests.rs`, `src/otel/workflow_handler/tests.rs`, `src/otel/otlp/tests.rs`, `src/otel/grafana/tests.rs`, the `src/otel/langfuse/*` suites, plus inline tests in `redaction.rs`, `span_names.rs`, `config.rs`, `env.rs`, `handle.rs`, `attrs.rs`, and `llm_attrs.rs`.
- Adapter unit tests live inline or in `tests.rs` siblings: `src/checkpointer/{mod,redis}.rs` (+ `redis/tests.rs`), `src/store/{mod,redis}.rs` (+ `redis/tests.rs`), `src/store/elasticsearch/{mod,client,tests}.rs`, `src/redis/{client,codec}`, `src/message_queue/{mod,bridge,pulsar}.rs`, `src/sys_operation/{mod,remote/**}`, `src/context_evolver/{mod,record,store,completion,memory_store}.rs` + `service/` and `prompt/` modules, and `src/vendor_specific/mod.rs`.
- `src/a2a/e2e_tests.rs` keeps the in-crate public-path client/server suites; `./tests/a2a_client.rs` and `./tests/a2a_server.rs` mirror them as crate-boundary targets for external consumers.
- Module doctests in `src/lib.rs` and `src/otel/mod.rs` compile as external crates and carry the public seam/handler assertions; the OTel handler doctests are `#[cfg(feature = "otel")]`-guarded so the default build stays green.
- Crate-boundary suites under `./tests/` use only `openjiuwen_extensions::...` public paths: `a2a_client.rs`, `a2a_server.rs`, `a2a_rest.rs` (`#![cfg(feature = "a2a")]`), `otel.rs`, `otel_otlp.rs`, `otel_integrations.rs` (`#![cfg(feature = "otel")]`), `redis.rs` (`#![cfg(feature = "redis")]`), `otel_seam.rs` and `adapters.rs` (compiled in every build, with per-test feature gates), plus the always-compiled `elasticsearch.rs`, `sandbox_providers.rs`, `vendor_reranker.rs`, and `context_evolver.rs`.
- A2A: `wiremock` for the client (send, streaming SSE, cancel, error mapping) and `axum`'s `Router::oneshot` (via `tower::ServiceExt`) for the server routes; no real sockets unless a test binds `127.0.0.1:0`.
- OTel: the in-memory `CaptureExporter` records exported `SpanData`, so span names, attributes, statuses, dedup, and parenting are asserted without a collector; every capture test calls `force_flush()` first. The OTLP exporter, Grafana/Langfuse presets and Langfuse ingestion are driven against `wiremock` peers.
- Redis speaks RESP to an in-process fake server; Elasticsearch, the sandbox gateways and DashScope use `wiremock`; the evolver's model path uses `wiremock` chat completions with a deterministic fallback suite that needs no peer.
- Deferred capabilities assert `ErrorKind::Unsupported` plus the message naming the missing piece and the supported alternative.
- Each ported behavior test names its Python counterpart (e.g. `tests/unit_tests/extensions/tracer_otel/test_redaction.py`) in a comment.

## Routing Table

- `./src/a2a/` → A2A wire types, JSON-RPC, REST `HTTP+JSON` binding, SSE framing, transformer, agent-card adapter, client, server, harness bridge, in-crate e2e suites (`a2a` feature)
- `./src/otel/` → OTel config/env, semantic conventions, redaction, span names, tracer handle, SDK provider, OTLP/HTTP JSON exporter (`./otlp/`), Grafana and Langfuse presets (`./grafana/`, `./langfuse/`), `TraceSink` sink, agent/workflow handlers (`otel` feature for the live parts)
- `./src/redis/` → in-crate RESP2/RESP3 client, codec, and the in-test fake server
- `./src/checkpointer/` → checkpointer factory over the kernel plus the live Redis backend (`redis` feature)
- `./src/store/` → key-value/document/vector store factories over the kernel, live Redis key-value backend and live Elasticsearch vector backend
- `./src/message_queue/` → queue factory over `openjiuwen_runner::mq`, JSON bridge, deferred Pulsar backend (`pulsar` feature)
- `./src/sys_operation/` → sandbox provider registration facade plus the real `aio`/`jiuwenbox`/`yuanrong` remote transports (`./remote/`)
- `./src/context_evolver/` → algorithm names, memory records, prompts/parsers, persistence, in-memory cosine vector store, and the deterministic + model-driven `TaskMemoryService` layer
- `./src/vendor_specific/` → deprecated vendor alias with the live DashScope rerank transport
- `./tests/` → crate-boundary public-API suites (`a2a_client`, `a2a_server`, `a2a_rest`, `otel`, `otel_otlp`, `otel_integrations`, `otel_seam`, `redis`, `elasticsearch`, `sandbox_providers`, `vendor_reranker`, `context_evolver`, `adapters`)
