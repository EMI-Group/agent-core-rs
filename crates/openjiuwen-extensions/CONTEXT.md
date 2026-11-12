# openjiuwen-extensions

## Intent

Optional integrations and backend adapters that live outside the kernel: the A2A protocol client/server, OpenTelemetry tracing over the session `TraceSink` SPI, and config-driven adapters over the kernel's checkpointer, store, message-queue, and sys-operation seams.
Maps to the Python package `openjiuwen.extensions` (v0.1.20): `a2a/`, `tracer_otel/`, `checkpointer/`, `store/`, `message_queue/`, `sys_operation/`, `context_evolver/`, `vendor_specific/`.
Nothing in the kernel depends on this crate; every heavy dependency and external service sits behind a Cargo feature, so the default build needs no network and no broker.

## API Surface

Every module is reachable from the crate root; feature-gated modules disappear entirely when their feature is off (no silent stubs), except `otel`, which keeps a documented no-op seam.

- `a2a` (`feature = "a2a"`) — `A2aClient`, `A2aRemoteClient`, `A2aServer`, `A2aServerAdapter`, wire types (`AgentCard`, `Message`, `Part`, `Task`, `TaskState`, …), JSON-RPC envelopes, `A2aTransformer`, `AgentCardAdapter`, and `harness_bridge` (A2A ⇄ `openjiuwen_harness_protocol`).
- `otel` (always compiled) — `OtelTracerConfig`, `semconv` constants, `truncate`/`hash_value`/`Redactor`, span-name builders, and `init_otel_tracer` → `TracerHandle`; with `feature = "otel"` the handle is a real SDK provider plus `OtelTraceSink`, `OtelAgentHandler`, `OtelWorkflowHandler`, and a `tracing-opentelemetry` layer.
- `checkpointer` — `build_checkpointer(&CheckpointerConfig)` dispatching to the kernel backends, and with `feature = "redis"` the `RedisCheckpointer` surface (`RedisConnectionConfig`, `RedisTTLConfig`, deferred to `Error::Unsupported`).
- `store` — `StoreBackend` + `build_kv_store`/`build_document_store`/`build_vector_store` over `openjiuwen_foundation::store`; Redis and Elasticsearch variants deferred.
- `message_queue` — `QueueConfig` + `create_queue` over `openjiuwen_runner::mq`, the JSON `QueueBridge`, and with `feature = "pulsar"` the `PulsarConfig`/`PulsarMessageQueue` surface (deferred).
- `sys_operation` — `SandboxProviderSet` registration against `openjiuwen_sysop::sandbox::provider::SandboxProviderRegistry`, plus documented deferred providers for `aio`, `jiuwenbox`, `yuanrong`.
- `context_evolver` — pure schema/vector/normalisation logic (`Algo`, `normalize_algo_name`, `AddMemoryRequest`, `MemoryVectorStore`) and the `TaskMemoryService` trait; the LLM-driven algorithms are a documented `Error::Unsupported` implementation.
- `vendor_specific` — deprecated `AliyunReranker` adapter over an injected `openjiuwen_retrieval::Reranker` (DashScope transport deferred).

## Feature Matrix

| Feature | Default | Enables | External requirement | Behavior when off |
|---|---|---|---|---|
| — (default) | yes | `otel` seam (no-op), `checkpointer`, `store` (memory/file), `message_queue` (in-process), `sys_operation`, `context_evolver`, `vendor_specific` | none | n/a |
| `a2a` | no | `axum`, `async-stream`, `bytes`; whole `a2a` module | none (client/server tests use loopback) | module absent |
| `otel` | no | `opentelemetry`, `opentelemetry_sdk`, `tracing-opentelemetry`, `tracing-subscriber`, `chrono`; real provider, sink, handlers, tracing layer | an OTLP collector only for OTLP export (deferred) | `init_otel_tracer` returns an inactive `TracerHandle`; `TracerHandle::is_active() == false` |
| `sqlite` | no | `sqlx`, `openjiuwen-foundation/sqlite`, `openjiuwen-session/sqlite` | a local file/in-memory database | `sqlite` backends answer `Error::Unsupported` |
| `redis` | no | `checkpointer::redis`, `store::redis` surfaces | a Redis server (not implemented) | variants absent |
| `pulsar` | no | `message_queue::pulsar` surface | a Pulsar broker (not implemented) | variant absent |
| `mcp` | no | `rmcp` re-export seam for hosts | an MCP server | absent |
| `live-tests` | no | network/service-gated tests | live services | those tests do not exist |

Deferred integrations (API-shaped, `Error::Unsupported`, documented in the module `CONTEXT.md`): Redis checkpoint/store backends, Elasticsearch vector store, Gauss/PostgreSQL document store, Pulsar message queue, ZeroMQ transports, OTLP span export, DashScope reranker transport, AIO/JiuwenBox/YuanRong sandbox providers, and the LLM-driven `context_evolver` algorithms.

## Constraints

- Only this crate and its `Cargo.toml` change from here; kernel crates are consumed as re-exported through `openjiuwen-core` (`session`, `runner`, `sys_operation`, `retrieval`, `foundation`).
- Async-first: tokio I/O, no blocking calls on async tasks, public streams are `BoxStream<'static, Result<T>>`, cancellation via `tokio_util::sync::CancellationToken`.
- Errors: `openjiuwen_common::error::{Error, Result}`; adapters map kernel errors, never panic on user input; deferred backends return `Error::Unsupported` with the backend name and the supported alternative.
- Serde: public wire types are `#[serde(rename_all = "camelCase")]` where the external protocol is camelCase (A2A, agent cards, OTEL attribute names) and `snake_case` elsewhere; unknown fields are tolerated.
- Logging: `tracing` only (targets `openjiuwen_extensions::*`); no stdout/stderr writes in library code; payload contents never logged above `debug`.
- Safety: no `unsafe`; no global mutable state beyond the documented Python-compatible process globals (none today).
- Size: files ≤ ~1000 lines; one file per concern; shared helpers (status mapping, SSE framing, redaction) live once at the module root.
- Feature tests must pass offline: `cargo test -p openjiuwen-extensions`, `--features a2a`, `--features otel`, plus `--features sqlite` where the environment allows.

## Test Strategy

- Unit tests live beside the code that owns the behavior (redaction, span names, status mapping, card/description embedding, normalisation, bridge conversion).
- Integration suites live in `./tests/` and start with `#![cfg(feature = "...")]`, so the default `cargo test` compiles them to nothing.
- A2A: `wiremock` for the client (send, streaming SSE, cancel, error mapping) and `axum`'s `Router::oneshot` (via `tower::ServiceExt`) for the server routes; no real sockets unless a test binds `127.0.0.1:0`.
- OTEL: an in-memory `SpanExporter` captures `SpanData` so attribute conventions, span names, statuses, and parenting are asserted without a collector.
- Adapters: kernel contract tests are reused through the factory (memory/file/sqlite), and deferred backends assert `ErrorKind::Unsupported` plus the message naming the backend.
- Each ported behavior test names its Python counterpart (e.g. `tests/unit_tests/extensions/tracer_otel/test_redaction.py`) in a comment.

## Routing Table

- `./src/a2a/` → A2A wire types, JSON-RPC, transformer, agent-card adapter, client, server, harness bridge (`a2a` feature)
- `./src/otel/` → OTel config, semantic conventions, redaction, span names, tracer handle, SDK provider, `TraceSink` sink, agent/workflow handlers (`otel` feature for the live parts)
- `./src/checkpointer/` → checkpointer factory over the kernel plus the deferred Redis backend (`redis` feature)
- `./src/store/` → key-value/document/vector store factories over the kernel plus deferred Redis/Elasticsearch backends
- `./src/message_queue/` → queue factory over `openjiuwen_runner::mq`, JSON bridge, deferred Pulsar backend (`pulsar` feature)
- `./src/sys_operation/` → sandbox provider registration facade and deferred provider sets
- `./src/context_evolver/` → algorithm names, memory schema, in-memory cosine vector store, `TaskMemoryService` seam
- `./src/vendor_specific/` → deprecated vendor aliases (`AliyunReranker`)
- `./tests/` → feature-gated integration suites (`a2a_client`, `a2a_server`, `otel`, `adapters`)
