# openjiuwen-extensions/tests

## Intent

Crate-boundary integration suites for `openjiuwen-extensions`: every assertion enters through a public `openjiuwen_extensions::…` path, so a broken re-export, factory signature, or deferral contract fails here before any consumer notices.
In-module unit tests (`src/**/tests.rs`) cover the same behavior at module scope; these files pin what an external host sees.
Each file is one test binary and starts with a `#![allow(clippy::unwrap_used, clippy::expect_used)]` (plus `clippy::panic` in `otel.rs`) because the workspace lints warn on both.
Every behavior test names its Python counterpart in a comment above it (e.g. `// Port of tests/unit_tests/extensions/checkpointer/test_agent_storage.py`).

## API Surface

| File | Feature gate | Scope |
| --- | --- | --- |
| `adapters.rs` | none (per-test `pulsar` gate) | checkpointer/store/message-queue/context-evolver factories: in-memory round-trips, backend selection, the Pulsar deferral contract |
| `context_evolver.rs` | none | the evolver end to end: deterministic retrieve/summarize over the JSON file store, the model-driven flows against a `wiremock` peer, `ScriptedCompletion`, algorithm normalization, kernel-`VectorStore`-backed scoring parity/durability, and the `Unsupported` opt-out |
| `redis.rs` | `#![cfg(feature = "redis")]` | RESP codec vectors, `RedisStore` KV/TTL/pipeline/prefix contracts, the document and vector stores (collections, query windows, schema round-trip, search ranking), `RedisCheckpointer` lifecycle over the in-test RESP2 fake (`redis/fake.rs`), factory wiring, TLS/cluster deferrals |
| `elasticsearch.rs` | none | every `VectorStore` operation of the REST backend (create/exists/schema/delete/list, NDJSON bulk, k-NN search, delete-by-query, 400/404/409/503 mapping) and every `DocumentStore` operation of the factory-wired kernel backend (`_bulk` upsert, get, query with sort+paging, count, `_delete_by_query`, list, drop-missing, Basic/ApiKey auth) against a `wiremock` cluster |
| `otel_seam.rs` | none (per-test `otel` gates) | `init_otel_tracer` for every `ExporterType` with and without the feature, including the `Otlp`/`Http` and `Otlp`/`HttpProtobuf` paths building a real exporter |
| `otel_otlp.rs` | `#![cfg(feature = "otel")]` | the OTLP/HTTP exporter against `wiremock`: JSON wire body shape, protocol/encoding serde, headers, fail-open behaviour, graceful shutdown |
| `otel_otlp_protobuf.rs` | `#![cfg(feature = "otel")]` | the protobuf encoding against `wiremock`: `Content-Type: application/x-protobuf`, byte-level decode of ids/name/kind/timestamps/attributes/status, plus a JSON-vs-protobuf equivalence case |
| `otel_integrations.rs` | `#![cfg(feature = "otel")]` | the Grafana Cloud OTLP preset, the Grafana local Agent/Alloy preset, and the Langfuse OTLP preset plus `LangfuseClient` REST ingestion |
| `otel.rs` | `#![cfg(feature = "otel")]` | `CaptureExporter` span names/attributes/statuses through the session `Tracer`/`TraceSink` and `OtelAgentHandler` |
| `a2a_client.rs` | `#![cfg(feature = "a2a")]` | `A2aClient` against a `wiremock` peer; folded `AgentResult` plus the received JSON-RPC envelope |
| `a2a_server.rs` | `#![cfg(feature = "a2a")]` | `A2aServer::router()` driven by `Router::oneshot`; card route, send/get/cancel, SSE frame order |
| `a2a_rest.rs` | `#![cfg(feature = "a2a")]` | the REST (`HTTP+JSON`) binding: card parity per transport, `message:send`/`messages:send`, raw SSE stream, `tasks/{id}`, both cancel and both subscribe spellings, REST error envelopes (400/404/405/409), transport config serde |
| `sandbox_providers.rs` | none | the three real remote sandbox providers (`aio` gateway envelope, `jiuwenbox` REST, `yuanrong` ADX) against loopback `wiremock` gateways: registration, wire payloads, auth injection, error mapping, streams, plus the teardown and conditional-write modules (`sandbox_providers/teardown.rs`, `sandbox_providers/conditional_write.rs`) |
| `vendor_reranker.rs` | none | the live DashScope rerank transport behind `AliyunReranker::from_config`: request shape, score/order mapping, `top_n` precedence, error mapping, `new` delegation |

## Constraints and Test Idioms

- Always-compiled files (`adapters.rs`, `otel_seam.rs`) gate optional backends per test or per branch, never at file level: `#[cfg(feature = "redis")]` above `#[tokio::test]`, and `if cfg!(feature = "otel") { … }` inside a test body when one test must cover both builds.
- No shared helper module: each suite declares its own small private helpers (`expect_error`, `rpc_url`, `client`, `received_body`, `server`, `post`, `sse_frames`) at the top of the file; `redis/fake.rs` is the only shared fake (an in-process RESP server), and `sandbox_providers/{teardown,conditional_write}.rs` are `#[path]`-mounted modules of the one `sandbox_providers` binary.
- Deferred backends are asserted through the error API, not pattern matching: `assert_eq!(error.kind(), ErrorKind::Unsupported)` plus `assert!(error.message().contains("redis"), "unexpected: {error}")`, where `error` comes from the local `fn expect_error<T>(result: Result<T>) -> Error` (usable because several factory `Ok` types are trait objects without `Debug`, so `unwrap_err` does not compile).
- `Error::Unsupported` is a tuple variant (`Error::Unsupported(String)`); `error.message()` returns `Cow<'_, str>` and `error.kind() -> ErrorKind`.
- Async style: plain `#[tokio::test]` (multi-thread-capable workspace tokio) and `#[test]` for pure functions; the crate takes no `test-util` feature, so `#[tokio::test(start_paused = true)]` and `tokio::time::pause` are unavailable.
- Bounding is done through the API under test (e.g. `request_json("math", json!(21), Some(Duration::from_secs(5)))`) or short real `tokio::time::timeout` calls in the in-src unit tests — no test harness timeout.
- `tempfile` is a dev-dependency used by in-src unit tests (`src/checkpointer/mod.rs` uses `tempfile::tempdir()`, `src/store/tests.rs` uses `tempfile::TempDir::new()`) and by `context_evolver.rs`, which roots its `JsonFileMemoryStore` in a `tempfile::tempdir()`.
- HTTP peers: `wiremock::MockServer::start()` with `server.uri()` for the client side (`a2a_client.rs`, `sandbox_providers.rs`, `vendor_reranker.rs`, `elasticsearch.rs`, and the OTLP suites; `redis.rs` uses a hand-written RESP server, and `context_evolver.rs` drives the real OpenAI-compatible adapter); `server.router().oneshot(...)` via `tower::ServiceExt` for the server side; `http_body_util::BodyExt::collect()` decodes bodies.
- The sandbox and DashScope suites bind an ephemeral loopback socket (`MockServer::start()` on `127.0.0.1:0`); the A2A suites never bind one.
- Provider suites never touch `sandbox::provider::global()`: each test builds a local `SandboxProviderRegistry` and gives every endpoint a unique `isolation_key` (or a pinned sandbox/environment id via `launcher_config.extra_params`), so the process-global container caches cannot leak between tests. Whole-cache teardown drains go through one crate-internal guard or a child process, because the parallel test binaries share that cache.
- `sandbox_providers.rs` is intentionally one large file: it is a single test binary covering three providers × three operation families, with the teardown and conditional-write cases in `#[path]`-mounted module files rather than a second binary.
- `elasticsearch.rs`, `redis.rs` and `context_evolver.rs` are intentionally one binary per backend family; their shared fixtures are small and local.
- OTel assertions always call `handle.force_flush()` before reading the in-memory `CaptureExporter`; the last re-recorded attribute value wins, hence the reversed `attribute()` helper. The protobuf suite decodes the exported bytes with a test-local decoder and one test encodes the same batch both ways.
- Everything is offline and deterministic; no `live-tests` gated file exists in this directory.

## Test Gaps

- No `sqlite`-gated test at the crate boundary: the SQLite checkpointer/store backends are exercised only by `src/checkpointer/mod.rs` and `src/store/tests.rs` unit tests.
- No test pairs `A2aClient` with `A2aServer` over a real loopback socket here; the only real-socket path is `src/a2a/e2e_tests.rs`.
- No crate-boundary case drives `KernelVectorIndex` over a vendor (`Elasticsearch`/`Redis`) vector store; the parity suite uses the kernel in-memory/file stores, and the vendor vector stores have their own suites.
- No `a2a_rest.rs` case exercises the REST `rest_prefix = ""` (v1-native) URL set or a custom `rest_path`; both are covered only by `src/a2a/server/rest_tests.rs`.
- The sandbox suites cover success and failure wire paths but never a live container, so container lifecycle against a real JiuwenBox/ADX deployment stays an operator-side check.
- `JIUWENBOX_API_TOKEN` resolution is only covered in-src (mutating the process environment from a parallel test binary would be racy); the crate-boundary suite supplies the bearer through `auth_headers` instead.
- No crate-boundary case runs the Elasticsearch or Redis backends against a real server; `wiremock` and the in-process RESP fake cover the wire contracts only.
