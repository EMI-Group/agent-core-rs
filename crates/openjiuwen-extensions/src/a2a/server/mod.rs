//! The A2A server: axum routes for both wire bindings, the executor SPI, and
//! the task store.
//!
//! Port of `openjiuwen/extensions/a2a/a2a_server.py`, `a2a_agent_executor.py`,
//! and `a2a_server_adapter.py` (v0.1.20). Python hands the transport to
//! `FastAPI` plus the `a2a-sdk` `DefaultRequestHandler`; this module owns the
//! same surface over axum: the JSON-RPC 2.0 endpoint, the `a2a-sdk` REST
//! (`HTTP+JSON`) endpoint, the [`A2aExecutor`] SPI, an [`EventSink`] that
//! mirrors Python's `TaskUpdater`, and a [`TaskStore`] kept coherent by that
//! sink.
//!
//! # Bindings and routes
//!
//! The card's `supportedInterfaces[*].protocolBinding` selects what is mounted
//! ([`config::A2aTransport`]): `JSONRPC` mounts only the JSON-RPC route,
//! `HTTP+JSON` mounts only the REST routes, and `BOTH` mounts both. Every mode
//! serves the well-known card route; the served card is derived at construction
//! so it advertises exactly the bindings that are mounted
//! ([`A2aServer::agent_card`]).
//!
//! | Binding | Route | Method | Body |
//! |---|---|---|---|
//! | card | [`A2aServerConfig::card_path`] (default `/.well-known/agent-card.json`) | `GET`, `HEAD` | the served [`types::AgentCard`] as `application/json` |
//! | JSON-RPC | [`A2aServerConfig::rpc_path`] (default `/a2a/jsonrpc/`) | `POST` | one JSON-RPC envelope, or `text/event-stream` for streaming methods |
//! | REST | `{rest_base}/card` | `GET` | the served [`types::AgentCard`] |
//! | REST | `{rest_base}/message:send`, `{rest_base}/messages:send` | `POST` | a [`types::Task`] as JSON |
//! | REST | `{rest_base}/message:stream` | `POST` | `text/event-stream`, one `data:` frame per published event |
//! | REST | `{rest_base}/tasks/{id}` | `GET` | a [`types::Task`], or a `404` envelope |
//! | REST | `{rest_base}/tasks/{id}/cancel`, `{rest_base}/tasks/{id}:cancel` | `POST` | the cancelled [`types::Task`] |
//! | REST | `{rest_base}/tasks/{id}/subscribe`, `{rest_base}/tasks/{id}:subscribe` | `GET`, `POST` | `text/event-stream`: the task snapshot, then its current status |
//!
//! `{rest_base}` is [`A2aServerConfig::rest_base_path`]
//! (`{rest_path}{rest_prefix}`, default `/a2a/rest/v1`). The `rpc_path` is
//! mounted **literally**, trailing slash included, because `a2a-sdk` clients
//! and cards publish `.../a2a/jsonrpc/` and streaming clients often do not
//! follow the `307` a missing slash would produce.
//!
//! # Request model (shared by both bindings)
//!
//! Both bindings accept the same openJiuwen payload object the JSON-RPC
//! binding carries: `{"message": …, "task_id": …, "sessionId": …, …metadata}`.
//! The REST send routes therefore need no second request model — they hand the
//! body to the same [`crate::a2a::transform::A2aTransformer`] mapping and the same
//! `plan_send` path.
//!
//! # JSON-RPC methods
//!
//! | Canonical (v1) | v0.3 alias | Result |
//! |---|---|---|
//! | `SendMessage` | `message/send` | the stored [`types::Task`] |
//! | `SendStreamingMessage` | `message/stream` | `text/event-stream`, one `data:` frame per event |
//! | `GetTask` | `tasks/get` | the stored [`types::Task`] |
//! | `CancelTask` | `tasks/cancel` | the stored [`types::Task`] in `CANCELED` |
//! | `SubscribeToTask` | `tasks/resubscribe` | a task snapshot frame plus its current status frame |
//!
//! Both spellings resolve through [`crate::a2a::jsonrpc::method::canonical`].
//! Every non-streaming JSON-RPC reply is HTTP 200 carrying a JSON-RPC
//! envelope: `-32600`/`-32700` for a malformed request, `-32601` for an
//! unknown method, `-32602` for unusable params, `-32001` for an unknown task,
//! `-32002` for a task that is already terminal, and `-32603` when the
//! executor fails.
//!
//! # REST errors
//!
//! Unlike JSON-RPC, REST failures carry the HTTP status, with the envelope
//! `{"error": {"code": <int>, "status": "<GRPC-ish>", "message": "<msg>"}}`
//! (`a2a/utils/error_handlers.py`):
//!
//! | code | HTTP | `status` |
//! |---|---|---|
//! | `-32700`/`-32600`/`-32602` | 400 | `INVALID_ARGUMENT` |
//! | `-32601` (HTTP method mismatch) | 405 | `METHOD_NOT_ALLOWED` |
//! | `-32001` | 404 | `NOT_FOUND` |
//! | `-32002` | 409 | `FAILED_PRECONDITION` |
//! | `-32004` | 501 | `UNIMPLEMENTED` |
//! | `-32005` | 415 | `UNSUPPORTED_MEDIA_TYPE` |
//! | `-32603`/unknown | 500 | `INTERNAL` |
//!
//! Starlette answers a known path with the wrong method with a plain-text
//! `405`; this server emits the JSON envelope instead (documented deviation).
//!
//! # Event order (Python `A2AAgentExecutor`)
//!
//! Every send — streaming or not, either binding — publishes, in this exact
//! order:
//!
//! 1. a [`types::Task`] snapshot with `state = SUBMITTED` and
//!    `history = [request message]`;
//! 2. a `WORKING` [`types::TaskStatusUpdateEvent`] carrying the agent message
//!    `Processing your request...`;
//! 3. zero or more artifact updates produced by the executor (`last_chunk` is
//!    set only when the executor marks the artifact terminal);
//! 4. for each artifact still open, a closing
//!    [`types::TaskArtifactUpdateEvent`] with empty `parts`, `append = true`,
//!    and `last_chunk = true`, in ascending artifact-id order;
//! 5. the terminal status: `FAILED`, `CANCELED`, `INPUT_REQUIRED`, otherwise
//!    `COMPLETED`.
//!
//! The snapshot has to come first: `a2a-sdk` consumers reject a status update
//! that arrives before the task exists — their error is "Agent should enqueue
//! `Task` before `TaskStatusUpdateEvent`". Executor errors publish `FAILED` and,
//! for a non-streaming JSON-RPC call, are also reported as a `-32603` error
//! envelope.
//!
//! # SSE framing
//!
//! One `data: <json>` frame per event, terminated by `\n\n`; there is no
//! `event:` name for normal frames (the REST SDK uses bare data frames too).
//! `sse_starlette` terminates frames with `\r\n\r\n`; `\n\n` is kept here as a
//! documented intentional difference shared by both bindings.
//!
//! # Colon-suffix routes
//!
//! The REST binding spells task actions as a colon suffix (`/tasks/{id}:cancel`,
//! `/tasks/{id}:subscribe`) as well as with a sub-path, because both forms are
//! in use (`a2a-sdk`'s v1 routes use the colon form). The pinned router
//! (`matchit` 0.8) does not allow a dynamic segment followed by a literal
//! suffix, so those spellings are served by one dispatcher route
//! `/tasks/{segment}` that splits `segment` on its **last** `:`. Task ids are
//! 32-hex UUIDs, so the split is unambiguous.
//!
//! # Store semantics
//!
//! Every [`EventSink`] emit also updates the stored [`types::Task`], which is
//! what `GetTask`, `CancelTask`, the non-streaming send reply, and the REST
//! routes read: the first snapshot inserts the task, status updates replace its
//! status, and an artifact update replaces the artifact with the same id or
//! appends it. Closing an open artifact does **not** touch the store — the
//! accumulated parts stay intact, matching Python's "close prior chunks without
//! replacing their accumulated contents". A dropped stream receiver is not an
//! error: the store is still updated, which is exactly what a non-streaming
//! call needs.
//!
//! # Documented deviations from the `CONTEXT.md` contract
//!
//! 1. [`A2aServer::serve_in_background`] is `async`: a socket must be bound
//!    before its local address is known (Python's `port = 0` case picks the port
//!    at bind time), so the caller awaits the bind and receives a
//!    [`ServerHandle`] that is already listening.
//! 2. [`A2aExecutor::cancel`]'s default body returns `Ok(())` and the server
//!    publishes the `CANCELED` status and event itself. The observable behavior
//!    matches Python's "default: emit Canceled", without forcing every executor
//!    to drive the sink.
//! 3. `SubscribeToTask` (`tasks/resubscribe`) — and the REST subscribe routes —
//!    answer with a stored task snapshot plus its current status and then end
//!    the stream; they are not live subscriptions to a running task.
//! 4. The gRPC transport is rejected at construction time
//!    ([`openjiuwen_common::error::Error::Unsupported`]).
//!
//! # Stream backpressure
//!
//! Each streaming send writes into a **bounded** channel
//! (`EVENT_BUFFER_CAPACITY` events). The publisher never blocks: when a
//! subscriber reads slower than the executor emits and the buffer fills, the
//! server logs the overflow at `warn`, drops that subscriber's channel, and the
//! stream ends with a terminal `INTERNAL` error frame. The task keeps running
//! and its store updates are unaffected, so `GetTask` and a later
//! `SubscribeToTask` still observe the final state. A client that keeps up
//! receives every event in order.
//!
//! # Documented limitations
//!
//! Tasks live in memory by default ([`InMemoryTaskStore`]) and are bounded by
//! [`A2aServerConfig::max_tasks`] and
//! [`A2aServerConfig::terminal_task_ttl_ms`]: terminal tasks are pruned first
//! (oldest terminal state first) and a running or awaited task is never
//! evicted. Supply another [`TaskStore`] through [`A2aServer::with_store`] to
//! persist them and own the retention policy.

mod config;
mod event_sink;
mod ops;
mod rest;
mod service;
mod sse;
mod store;

#[cfg(test)]
mod rest_tests;
#[cfg(test)]
mod tests;

pub use config::{
    A2aExecutionContext, A2aExecutor, A2aServerConfig, A2aTransport, DEFAULT_CARD_PATH,
    DEFAULT_HOST, DEFAULT_PORT, DEFAULT_PROTOCOL_VERSION, DEFAULT_REST_PATH, DEFAULT_REST_PREFIX,
    DEFAULT_RPC_PATH,
};
pub use event_sink::EventSink;
pub use service::{A2aServer, A2aServerAdapter, ServerHandle};
pub use store::{
    InMemoryTaskStore, RetentionPolicy, TaskStore, DEFAULT_MAX_TASKS, DEFAULT_TERMINAL_TASK_TTL_MS,
};

use std::sync::Arc;

use super::types;

/// The `protocolBinding` string of the JSON-RPC binding.
const JSONRPC_BINDING: &str = "JSONRPC";

/// The `protocolBinding` string of the REST binding.
const REST_BINDING: &str = "HTTP+JSON";

/// The shared server state behind every route of both bindings.
pub(super) struct Inner {
    /// The configuration this server was built with.
    pub(super) config: config::A2aServerConfig,
    /// The application hook that produces tasks.
    pub(super) executor: Arc<dyn config::A2aExecutor>,
    /// The tasks produced so far.
    pub(super) store: Arc<dyn store::TaskStore>,
    /// The card served at every card route (derived from `config`).
    pub(super) card: types::AgentCard,
}

/// Derives the served card from the configured one for the mounted transport.
///
/// The transport interfaces are rewritten so the card advertises exactly one
/// endpoint per binding the server mounts; every non-transport interface is
/// preserved in order, and every other card field is passed through unchanged.
/// `A2aTransport::JsonRpc` keeps the configured card untouched, which is what
/// existing JSON-RPC-only deployments publish.
pub(super) fn derive_card(config: &config::A2aServerConfig) -> types::AgentCard {
    let mut card = config.agent_card.clone();
    if config.transport == config::A2aTransport::JsonRpc {
        return card;
    }
    let rest_url = config.rest_interface_url.clone().unwrap_or_else(|| {
        format!(
            "http://{}:{}{}",
            config.host,
            config.port,
            config.rest_base_path()
        )
    });
    let rpc_url = config
        .interface_url
        .clone()
        .unwrap_or_else(|| format!("http://{}:{}{}", config.host, config.port, config.rpc_path));
    let others = || {
        config
            .agent_card
            .supported_interfaces
            .iter()
            .filter(|interface| {
                !interface
                    .protocol_binding
                    .eq_ignore_ascii_case(JSONRPC_BINDING)
                    && !interface
                        .protocol_binding
                        .eq_ignore_ascii_case(REST_BINDING)
            })
            .cloned()
    };
    let rest_interface = types::AgentInterface {
        url: rest_url,
        protocol_binding: REST_BINDING.to_owned(),
        protocol_version: config.protocol_version.clone(),
        tenant: None,
    };
    card.supported_interfaces = if config.transport.mounts_json_rpc() {
        let jsonrpc_interface = types::AgentInterface {
            url: rpc_url,
            protocol_binding: JSONRPC_BINDING.to_owned(),
            protocol_version: config.protocol_version.clone(),
            tenant: None,
        };
        std::iter::once(rest_interface)
            .chain(std::iter::once(jsonrpc_interface))
            .chain(others())
            .collect()
    } else {
        std::iter::once(rest_interface).chain(others()).collect()
    };
    card
}

/// A fresh 32-lowercase-hex id, shaped like Python's `uuid4().hex`.
fn fresh_id() -> String {
    uuid::Uuid::new_v4().simple().to_string()
}

/// An agent-authored message for `state`, stamped with the task and context ids.
fn agent_message(text: impl Into<String>, task_id: &str, context_id: &str) -> types::Message {
    types::Message {
        message_id: fresh_id(),
        context_id: Some(context_id.to_owned()),
        task_id: Some(task_id.to_owned()),
        role: types::Role::Agent,
        parts: vec![types::Part::text(text)],
        ..types::Message::default()
    }
}
