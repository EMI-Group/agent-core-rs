# server/ — the A2A server (`feature = "a2a"`)

## Intent

The axum-backed A2A server: both wire bindings (JSON-RPC 2.0 and REST `HTTP+JSON`), the executor SPI, the event sink, and the task store.
Port of `openjiuwen/extensions/a2a/a2a_server.py`, `a2a_agent_executor.py`, and `a2a_server_adapter.py` (v0.1.20); the REST routes mirror the `a2a-sdk` REST app that Python mounts at `/a2a/rest`.
The module-level documentation (route tables, method table, error mapping, event order, store semantics, SSE framing, colon-suffix rationale, documented deviations) lives in `mod.rs`.

## API Surface

Everything is re-exported at `crate::a2a::server::`; the child modules are private.

- `A2aServerConfig`, `A2aTransport`, `DEFAULT_RPC_PATH`, `DEFAULT_CARD_PATH`, `DEFAULT_REST_PATH`, `DEFAULT_REST_PREFIX`, `DEFAULT_PROTOCOL_VERSION`, `DEFAULT_HOST`, `DEFAULT_PORT`, `DEFAULT_MAX_TASKS`, `DEFAULT_TERMINAL_TASK_TTL_MS`.
- `RetentionPolicy`, the bounds an `InMemoryTaskStore` applies (`max_tasks` `0` = unbounded, `terminal_ttl` `None` = keep terminal tasks forever).
- `A2aServerConfig` derives `Serialize`/`Deserialize` with container-level `#[serde(default)]`: `{}` deserializes to the defaults, unknown fields are tolerated, and every field has a default.
- Transport fields: `transport` (`A2aTransport::{JsonRpc, Rest, Both}`, default `JsonRpc`), `rest_path` (`/a2a/rest`), `rest_prefix` (`/v1`, `""` selects the v1-native URL set), `rest_interface_url` (derived when `None`).
- `A2aTransport` serializes as the wire binding strings (`"JSONRPC"`, `"HTTP+JSON"`, `"BOTH"`) and accepts `json_rpc`/`jsonrpc`, `rest`/`http+json`/`http_json`, and `both` as aliases; `mounts_json_rpc()`/`mounts_rest()` report the mounted routes.
- Builders: `new(card)`, `with_interface_url`, `with_bind(host, port)`, `with_rpc_path`, `with_transport`, `with_rest_path`, `with_rest_prefix`, `with_rest_interface_url`, `with_max_tasks`, `with_terminal_task_ttl_ms`.
- `max_tasks` (default `DEFAULT_MAX_TASKS`) and `terminal_task_ttl_ms` (default `DEFAULT_TERMINAL_TASK_TTL_MS`) bound the store `A2aServer::new` builds; `retention_policy()` projects them onto a `RetentionPolicy`, and a store supplied through `A2aServer::with_store` owns its own policy.
- `A2aServerConfig::rest_base_path()` returns the normalized `{rest_path}{rest_prefix}` (one `/` between the parts, no trailing slash); there is no `with_card_path`, set the field directly.
- `A2aServer::agent_card()` returns the card served at every card route; it is derived at construction (`derive_card`) so the interfaces advertise exactly the mounted bindings (JSON-RPC mode keeps the configured card unchanged).
- Only `serve_in_background` (async, binds before returning) exposes the bound address, so a `port = 0` caller must use it or `A2aServerAdapter::local_addr`; `serve()` consumes `self` and reports the address only through `tracing::info!`.
- `A2aExecutionContext`, `A2aExecutor`, `EventSink` (`EventSink::new` is `pub(crate)` and returns an `EventStream`), `TaskStore`, `InMemoryTaskStore`, `RetentionPolicy`, `A2aServer`, `ServerHandle`, `A2aServerAdapter`.

## Routes and Behavior

- The card route (`config.card_path`) is mounted in every transport mode; it serves `A2aServer::agent_card()`.
- `A2aTransport::JsonRpc` mounts only `POST config.rpc_path` (default `/a2a/jsonrpc/`, literal trailing slash).
- `A2aTransport::Rest` mounts `GET {base}/card`, `POST {base}/message:send`, `POST {base}/messages:send`, `POST {base}/message:stream`, `GET {base}/tasks/{id}`, `POST {base}/tasks/{id}/cancel` and `{base}/tasks/{id}:cancel`, and `GET|POST {base}/tasks/{id}/subscribe` and `{base}/tasks/{id}:subscribe`, where `{base}` is `rest_base_path()`.
- `A2aTransport::Both` mounts both; routing is exclusive per binding, matching Python's `a2a_server.py`.
- The REST request model is the JSON-RPC params object (`{"message": …, "task_id": …, "sessionId": …, …metadata}`); both bindings call `ops::plan_send`.
- REST send routes answer `200` with the stored `Task`; streaming answers `text/event-stream` with bare `data: <json>` frames; subscribe answers the task snapshot then its `TaskStatusUpdateEvent`, then closes.
- REST failures use `{"error": {"code": <i32>, "status": "<GRPC-ish>", "message": "<msg>"}}` with the HTTP status mapped from the A2A code (400/405/404/409/501/415/500).
- A known REST path with the wrong HTTP method answers `405` with that JSON envelope; `Router::method_not_allowed_fallback` is installed on the REST sub-router after its routes, so the JSON-RPC route keeps its default behavior.
- The colon spellings are served by one `{base}/tasks/{segment}` dispatcher that splits the segment on its last `:`; `matchit` 0.8 cannot express a dynamic segment followed by a literal suffix, and task ids are 32-hex UUIDs so the split is unambiguous.
- Construction validation: `card_path`, `rpc_path`, and `rest_path` must start with `/`, the REST base must be non-empty and absolute, and a REST base colliding with `card_path`/`rpc_path` is `Error::Config`; gRPC interfaces are `Error::Unsupported`.
- Event order, store semantics, and the terminal-status rules are shared by both bindings through `ops` (see `mod.rs`).
- Retention runs on every store write and lazily on reads: expired terminal tasks are dropped, then terminal tasks are evicted oldest-terminal-first until `max_tasks` holds; running and awaited (`INPUT_REQUIRED`) tasks are never evicted, so a burst of in-flight work can exceed the cap instead of losing tasks.
- Each stream uses a bounded channel (`EVENT_BUFFER_CAPACITY` = 256). A subscriber that falls behind is dropped without blocking the publisher: a `warn` is logged, `EventSink::subscriber_overflowed()` and `EventStream::overflowed()` report it, and the stream ends with a terminal `INTERNAL` error frame (JSON-RPC envelope or REST `{"error": …}` value). A subscriber that keeps up receives every event in order.

## Extending the Routes

- `A2aServer::router(&self) -> Router` returns the fully state-injected router; it adds the card route, the JSON-RPC route when `mounts_json_rpc()`, and merges `rest::router(&inner)` when `mounts_rest()`.
- State is the private `struct Inner { config, executor, store, card }` in `mod.rs`; `ops.rs`, `rest.rs`, and `service.rs` share it, while an outside compositor adds routes by merging onto `server.router()`.
- Binding-neutral operations live in `ops.rs` (`plan_send`, `send_task`, `start_stream`, `task_snapshot`, `stored_task`, `cancel_stored_task`, `context_from_task`, `task_id_param`, the JSON/plain-text response builders); extend that module instead of duplicating logic in a binding.
- SSE framing lives in `sse.rs`: `frame`/`response`/`responses`/`receiver_response` wrap JSON-RPC responses, `value_frame`/`value_stream`/`value_responses`/`value_receiver_response` stream bare values for REST.
- `EventSink::new` is `pub(crate)` and returns `(EventSink, EventStream)`; a new route can build a sink exactly like `plan_send` does, keeping the store coherent through the same path.

## Constraints

- Public paths must stay at `crate::a2a::server::`; child modules are declared `mod x;` and re-exported with `pub use`.
- Behavior is a port of the Python executor: `Task{SUBMITTED}` → `WORKING` → artifacts → closing artifact updates → terminal status, identical for both bindings.
- JSON-RPC errors surface as JSON-RPC codes at HTTP 200 (`-32600`, `-32601`, `-32602`, `-32001`, `-32002`, `-32603`); REST errors surface as the mapped HTTP status plus the error envelope; no panics on user input.
- SSE frames end with `\n\n` for both bindings (`sse_starlette` uses `\r\n\r\n`; documented intentional difference); payload contents never logged above `debug`.
- Files stay well under ~1000 lines; no `unsafe`, no blocking I/O, `tracing` only.
- Event emits never block: the bounded channel uses `try_send` and drops a slow subscriber rather than stalling the executor.

## Test Strategy

- `./tests.rs` covers routing, JSON-RPC dispatch, event order, store retention (cap, terminal-first eviction, TTL, running/awaited retention), stream backpressure (dropped slow subscriber, in-order healthy subscriber), lifecycle, and the JSON-RPC config surface, plus a compile-time check of the public paths.
- `./rest_tests.rs` covers config defaults/serde/`rest_base_path`, transport serde and builders, card derivation in all three modes, construction errors, derived interface URLs, and the REST routes through `A2aServer::router()` + `tower::ServiceExt::oneshot` (card, send singular/plural, stream, get/404, cancel both spellings/409, subscribe both spellings, unknown action, 405 envelope, parse/params errors, and exclusive mounting).
- Both suites reuse the `tests.rs` fixtures (`card`, `stored_task`, `EchoExecutor`).
- Python counterparts are named in the tests where a behavior is ported.

## Routing Table

- `./mod.rs` → module documentation, submodule declarations, `pub use` re-exports, `Inner`, `derive_card`, the shared id/message helpers, the test declarations
- `./config.rs` → `A2aTransport`, `A2aServerConfig` and its public default consts/builders, `A2aExecutionContext`, `A2aExecutor`
- `./ops.rs` → binding-neutral send/stream/snapshot/cancel operations and shared response builders
- `./sse.rs` → SSE framing for JSON-RPC envelopes and bare REST values
- `./rest.rs` → the REST (`HTTP+JSON`) sub-router, colon-action dispatcher, and error envelope/status mapping
- `./store.rs` → `TaskStore` and `InMemoryTaskStore`
- `./event_sink.rs` → `EventSink`, which publishes events and keeps the store coherent
- `./service.rs` → `A2aServer`, `ServerHandle`, `A2aServerAdapter`, transport-aware router construction, path validation, and the thin JSON-RPC handlers
- `./tests.rs` → JSON-RPC/config/lifecycle unit tests plus the shared fixtures
- `./rest_tests.rs` → REST binding and transport-configuration unit tests
