//! The A2A server: axum routes, `JSON-RPC` dispatch, the executor SPI, and the
//! task store.
//!
//! Port of `openjiuwen/extensions/a2a/a2a_server.py`, `a2a_agent_executor.py`,
//! and `a2a_server_adapter.py` (v0.1.20). Python hands the transport to
//! `FastAPI` plus the `a2a-sdk` `DefaultRequestHandler`; this module owns the
//! same surface over axum: two routes, five `JSON-RPC` 2.0 methods, the
//! [`A2aExecutor`] SPI, an [`EventSink`] that mirrors Python's `TaskUpdater`,
//! and a [`TaskStore`] kept coherent by that sink.
//!
//! # Routes
//!
//! | Route | Method | Body |
//! |---|---|---|
//! | [`A2aServerConfig::card_path`] (default `/.well-known/agent-card.json`) | `GET`, `HEAD` | the serialized [`types::AgentCard`] as `application/json` |
//! | [`A2aServerConfig::rpc_path`] (default `/a2a/jsonrpc/`) | `POST` | one `JSON-RPC` envelope, or `text/event-stream` for streaming methods |
//!
//! The `rpc_path` is mounted **literally**, trailing slash included, because
//! `a2a-sdk` clients and cards publish `.../a2a/jsonrpc/` and streaming clients
//! often do not follow the `307` a missing slash would produce. Clients must
//! `POST` to exactly that path.
//!
//! # `JSON-RPC` methods
//!
//! | Canonical (v1) | v0.3 alias | Result |
//! |---|---|---|
//! | `SendMessage` | `message/send` | the stored [`types::Task`] |
//! | `SendStreamingMessage` | `message/stream` | `text/event-stream`, one `data:` frame per event |
//! | `GetTask` | `tasks/get` | the stored [`types::Task`] |
//! | `CancelTask` | `tasks/cancel` | the stored [`types::Task`] in `CANCELED` |
//! | `SubscribeToTask` | `tasks/resubscribe` | a task snapshot frame plus its current status frame |
//!
//! Both spellings resolve through
//! [`crate::a2a::jsonrpc::method::canonical`]. Every non-streaming reply is HTTP
//! 200 carrying a `JSON-RPC` envelope, as the `JSON-RPC` convention requires:
//! `-32600`/`-32700` for a malformed request, `-32601` for an unknown method,
//! `-32602` for unusable params, `-32001` for an unknown task, `-32002` for a
//! task that is already terminal, and `-32603` when the executor fails.
//!
//! # Event order (Python `A2AAgentExecutor`)
//!
//! Every `SendMessage` — streaming or not — publishes, in this exact order:
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
//! for a non-streaming call, are also reported as a `-32603` error envelope.
//!
//! # Store semantics
//!
//! Every [`EventSink`] emit also updates the stored [`types::Task`], which is
//! what `GetTask`, `CancelTask`, and the non-streaming `SendMessage` reply read:
//! the first snapshot inserts the task, status updates replace its status, and
//! an artifact update replaces the artifact with the same id or appends it.
//! Closing an open artifact does **not** touch the store — the accumulated
//! parts stay intact, matching Python's "close prior chunks without replacing
//! their accumulated contents". A dropped stream receiver is not an error: the
//! store is still updated, which is exactly what a non-streaming call needs.
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
//! 3. `SubscribeToTask` (`tasks/resubscribe`) answers with a stored task
//!    snapshot plus its current status and then ends the stream; it is not a
//!    live subscription to a running task.
//!
//! # Documented limitations
//!
//! - The gRPC transport is rejected at construction time
//!   ([`openjiuwen_common::error::Error::Unsupported`]). A REST (`HTTP+JSON`)
//!   interface is tolerated on the card, but no REST route is mounted, exactly
//!   like the Python wrapper when only the JSON-RPC transport is configured.
//! - Tasks live in memory by default ([`InMemoryTaskStore`]); supply another
//!   [`TaskStore`] through [`A2aServer::with_store`] to persist them.

mod config;
mod event_sink;
mod service;
mod store;

#[cfg(test)]
mod tests;

pub use config::{
    A2aExecutionContext, A2aExecutor, A2aServerConfig, DEFAULT_CARD_PATH, DEFAULT_HOST,
    DEFAULT_PORT, DEFAULT_PROTOCOL_VERSION, DEFAULT_RPC_PATH,
};
pub use event_sink::EventSink;
pub use service::{A2aServer, A2aServerAdapter, ServerHandle};
pub use store::{InMemoryTaskStore, TaskStore};

use super::types;

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
