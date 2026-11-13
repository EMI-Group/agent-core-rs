# server/ — the A2A server (`feature = "a2a"`)

## Intent

The axum-backed A2A server: routes, JSON-RPC dispatch, the executor SPI, the event sink, and the task store.
Port of `openjiuwen/extensions/a2a/a2a_server.py`, `a2a_agent_executor.py`, and `a2a_server_adapter.py` (v0.1.20).
The module-level documentation (routes, method table, event order, store semantics, documented deviations) lives in `mod.rs`.

## API Surface

Everything is re-exported at `crate::a2a::server::`; the child modules are private.

- `A2aServerConfig`, `DEFAULT_RPC_PATH`, `DEFAULT_CARD_PATH`, `DEFAULT_PROTOCOL_VERSION`, `DEFAULT_HOST`, `DEFAULT_PORT`
- `A2aServerConfig` implements `Default` and offers `new(card)`, `with_interface_url`, `with_bind(host, port)`, `with_rpc_path` — there is no `with_card_path`; set the field directly.
- Only `serve_in_background` (async, binds before returning) exposes the bound address, so a `port = 0` caller must use it or `A2aServerAdapter::local_addr`; `serve()` consumes `self` and reports the address only through `tracing::info!`.
- `A2aExecutionContext`, `A2aExecutor`
- `EventSink` (`EventSink::new` is `pub(crate)`)
- `TaskStore`, `InMemoryTaskStore`
- `A2aServer`, `ServerHandle`, `A2aServerAdapter`

## Extending the Routes (e.g. a REST binding)

- `A2aServer::router(&self) -> Router` is public and returns the fully state-injected router: `Router::new().route(&config.card_path, get(card_handler)).route(&config.rpc_path, post(rpc_handler)).with_state(Arc::clone(&self.inner))`.
- State is the private `struct Inner { config: A2aServerConfig, executor: Arc<dyn A2aExecutor>, store: Arc<dyn TaskStore> }`; because `Inner` and the handlers are private, extra routes are added by `.merge(..)`-ing an independently-built router onto `server.router()`, or by adding handlers inside `service.rs` — never by constructing `Inner` elsewhere.
- `A2aServer::config()`, `store()`, and `Clone` give an outside compositor the card, the task store, and a cheap handle; `A2aServerAdapter` only wraps `serve`/`serve_in_background` lifecycle.
- A card advertising `protocol_binding = "HTTP+JSON"` is tolerated: `reject_unsupported_transports` only rejects `gRPC`, and no REST route is mounted today.
- The SSE framing helpers (`sse_frame`, `sse_response`, `sse_responses`, `sse_receiver_response`, `json_result`, `json_error`, `task_id_param`) are private to `service.rs`, so a new REST/SSE route must live in that module to reuse them or rebuild `Body::from_stream` plus the `text/event-stream`/`no-cache` headers itself.
- `EventSink::new` is `pub(crate)`; a new route inside `service.rs` can therefore build a sink exactly like `plan_send` does, keeping the store coherent through the same path.

## Constraints

- Public paths must stay at `crate::a2a::server::`; child modules are declared `mod x;` and re-exported with `pub use`.
- Behavior is a port of the Python executor: `Task{SUBMITTED}` → `WORKING` → artifacts → closing artifact updates → terminal status.
- Errors surface as JSON-RPC codes (`-32600`, `-32601`, `-32602`, `-32001`, `-32002`, `-32603`); no panics on user input.
- Files stay well under ~1000 lines; no `unsafe`, no blocking I/O.

## Routing Table

- `./mod.rs` → module documentation, submodule declarations, `pub use` re-exports, the shared id/message helpers, the `tests` declaration
- `./config.rs` → `A2aServerConfig` and its public default consts, `A2aExecutionContext`, `A2aExecutor`
- `./store.rs` → `TaskStore` and `InMemoryTaskStore`
- `./event_sink.rs` → `EventSink`, which publishes events and keeps the store coherent
- `./service.rs` → `A2aServer`, `ServerHandle`, `A2aServerAdapter`, the axum handlers, JSON-RPC dispatch, and SSE framing
- `./tests.rs` → unit tests (routing, dispatch, event order, store, lifecycle) plus a compile-time check of the public paths
