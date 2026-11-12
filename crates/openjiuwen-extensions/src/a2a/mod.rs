//! The Agent2Agent (`A2A`) protocol client, server, and wire types.
//!
//! Port of the Python package `openjiuwen.extensions.a2a` (v0.1.20). The Python
//! implementation is a thin adapter over `a2a-sdk`; here the wire layer is owned
//! directly (JSON-RPC 2.0 over HTTP, server-sent events for streaming) so the
//! method names, card JSON, status values, and roles match what `a2a-sdk` 1.0
//! emits on the wire.
//!
//! Layout:
//!
//! - [`types`] — wire models (`AgentCard`, `Message`, `Task`, events, requests).
//! - [`jsonrpc`] — JSON-RPC 2.0 envelopes, error codes, method names and aliases.
//! - [`transform`] — the openJiuwen-side `AgentResult` model and request/response
//!   mapping in both directions.
//! - [`card`] — conversion between the kernel agent card and the A2A agent card.
//! - [`client`] — [`client::A2aClient`] and [`client::A2aRemoteClient`].
//! - [`server`] — the axum router, executor SPI, event sink, and task store.
//! - [`harness_bridge`] — conversions to and from `openjiuwen-harness-protocol`.
//!
//! The whole module only exists with the Cargo feature `a2a`; nothing in this
//! module is compiled into the default build.

pub mod card;
pub mod client;
pub mod harness_bridge;
pub mod jsonrpc;
pub mod server;
pub mod transform;
pub mod types;
