# openjiuwen-extensions/tests

## Intent

Crate-boundary integration suites for `openjiuwen-extensions`: every assertion enters through a public `openjiuwen_extensions::…` path, so a broken re-export, factory signature, or deferral contract fails here before any consumer notices.
In-module unit tests (`src/**/tests.rs`) cover the same behavior at module scope; these files pin what an external host sees.
Each file is one test binary and starts with a `#![allow(clippy::unwrap_used, clippy::expect_used)]` (plus `clippy::panic` in `otel.rs`) because the workspace lints warn on both.
Every behavior test names its Python counterpart in a comment above it (e.g. `// Port of tests/unit_tests/extensions/checkpointer/test_agent_storage.py`).

## API Surface

| File | Feature gate | Scope |
| --- | --- | --- |
| `adapters.rs` | none (per-test `redis`/`pulsar` gates) | checkpointer/store/message-queue/sys-op/context-evolver factories, deferred backends, in-memory round-trips |
| `otel_seam.rs` | none (per-test `otel` gates) | `init_otel_tracer` for every `ExporterType` with and without the feature; OTLP deferred |
| `otel.rs` | `#![cfg(feature = "otel")]` | `CaptureExporter` span names/attributes/statuses through the session `Tracer`/`TraceSink` and `OtelAgentHandler` |
| `a2a_client.rs` | `#![cfg(feature = "a2a")]` | `A2aClient` against a `wiremock` peer; folded `AgentResult` plus the received JSON-RPC envelope |
| `a2a_server.rs` | `#![cfg(feature = "a2a")]` | `A2aServer::router()` driven by `Router::oneshot`; card route, send/get/cancel, SSE frame order |

## Constraints and Test Idioms

- Always-compiled files (`adapters.rs`, `otel_seam.rs`) gate optional backends per test or per branch, never at file level: `#[cfg(feature = "redis")]` above `#[tokio::test]`, and `if cfg!(feature = "otel") { … }` inside a test body when one test must cover both builds.
- No shared helper module: there is no `tests/support/` and no `mod` file — each suite declares its own small private helpers (`expect_error`, `rpc_url`, `client`, `received_body`, `server`, `post`, `sse_frames`) at the top of the file.
- Deferred backends are asserted through the error API, not pattern matching: `assert_eq!(error.kind(), ErrorKind::Unsupported)` plus `assert!(error.message().contains("redis"), "unexpected: {error}")`, where `error` comes from the local `fn expect_error<T>(result: Result<T>) -> Error` (usable because several factory `Ok` types are trait objects without `Debug`, so `unwrap_err` does not compile).
- `Error::Unsupported` is a tuple variant (`Error::Unsupported(String)`); `error.message()` returns `Cow<'_, str>` and `error.kind() -> ErrorKind`.
- Async style: plain `#[tokio::test]` (multi-thread-capable workspace tokio) and `#[test]` for pure functions; the crate takes no `test-util` feature, so `#[tokio::test(start_paused = true)]` and `tokio::time::pause` are unavailable.
- Bounding is done through the API under test (e.g. `request_json("math", json!(21), Some(Duration::from_secs(5)))`) or short real `tokio::time::timeout` calls in the in-src unit tests — no test harness timeout.
- `tempfile` is a dev-dependency of the crate but currently used only by in-src unit tests (`src/checkpointer/mod.rs` uses `tempfile::tempdir()`, `src/store/mod.rs` uses `tempfile::TempDir::new()`); no file-boundary test in this directory uses a temp dir yet.
- HTTP peers: `wiremock::MockServer::start()` with `server.uri()` for the client side; `server.router().oneshot(...)` via `tower::ServiceExt` for the server side; `http_body_util::BodyExt::collect()` decodes bodies. No suite binds a real socket here.
- OTel assertions always call `handle.force_flush()` before reading the in-memory `CaptureExporter`; the last re-recorded attribute value wins, hence the reversed `attribute()` helper.
- Everything is offline and deterministic; no `live-tests` gated file exists in this directory.

## Test Gaps

- No `sqlite`-gated test at the crate boundary: the SQLite checkpointer/store backends are exercised only by `src/checkpointer/mod.rs` and `src/store/mod.rs` unit tests.
- No test pairs `A2aClient` with `A2aServer` over a real loopback socket here; the only real-socket path is `src/a2a/e2e_tests.rs`.
- `context_evolver::MemoryVectorStore` and `sandbox` provider registration have no direct crate-boundary suite beyond `adapters.rs`.
