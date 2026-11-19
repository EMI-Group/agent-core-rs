//! Cross-process team attach over Unix domain sockets.
//!
//! A team hosted by one `openjiuwen` process can be driven by another process — the `team-member`
//! skill CLI or the `openjiuwen-team-mcp` server — through a small request/response protocol on a
//! Unix socket.
//! This directory owns the platform-independent half of that bridge:
//! the wire contract in [`protocol`], which the socket server and client are built on top of, and
//! [`AttachClient`], the client that drives a hosted team over one connection.
//!
//! # Framing
//!
//! One UTF-8 JSON object per line, terminated by `\n`.
//! A request is `{"v":1,"id":7,"op":"list_members","args":{}}`; the server answers with exactly one
//! response echoing `id`, either `{"v":1,"id":7,"ok":true,"data":...}` or
//! `{"v":1,"id":7,"ok":false,"error":{"kind":"not_found","message":"..."}}`.
//! Frames are versioned ([`PROTOCOL_VERSION`]), bounded ([`MAX_FRAME_BYTES`]), and ignore unknown
//! JSON fields for forward compatibility.
//! A frame-level failure closes the connection; an op-level failure is returned in-band and keeps
//! it open.
//!
//! # Client
//!
//! [`AttachClient::connect`] opens the socket (no handshake — connecting is the availability
//! contract), [`AttachClient::call`] performs one request/response exchange with a bounded deadline,
//! and the typed wrappers (`list_members`, `task_detail`, `snapshot`, …) decode the op payloads.
//! One client holds one connection and at most one in-flight request.
//!
//! # Server
//!
//! [`AttachServer::bind`] creates the socket file (parent directories, stale-socket reclamation and
//! mode `0600` included), [`AttachServer::serve`] answers every connection concurrently through an
//! [`AttachHandler`] under the [`AttachLimits`] bounds (a concurrency ceiling plus a per-connection
//! idle timeout), and [`AttachServer::shutdown`] stops the accept loop and removes the file.
//! Dropping a server also removes the socket file, and only ever a path that is still a socket.
//!
//! # Security
//!
//! The protocol carries no authentication or encryption: trust comes from the filesystem.
//! The server binds its socket inside a directory only the owning user can enter (`0700`), so only
//! same-user processes can connect.
//! Handlers must still validate every name, id and tool argument they receive, because the socket
//! constrains *who* may connect, not *what* they may ask for.
//!
//! # Platform support
//!
//! Cross-process attach uses Unix domain sockets only.
//! The protocol layer here is platform independent and compiles everywhere; [`AttachClient`] and
//! [`AttachServer`] keep the same public surface on every target, but their transport methods return
//! `Error::Unsupported` with the message "cross-process attach requires Unix domain sockets" on
//! non-Unix targets (and [`AttachClient::is_available`] is always `false` there).
pub mod client;
pub mod protocol;
pub mod server;

pub use client::AttachClient;
pub use protocol::{
    AttachError, AttachHandler, AttachOp, AttachResult, RequestFrame, ResponseFrame,
    MAX_FRAME_BYTES, PROTOCOL_VERSION,
};
pub use server::{AttachLimits, AttachServer};

#[cfg(test)]
#[path = "client_tests.rs"]
mod client_tests;

#[cfg(test)]
#[path = "protocol_tests.rs"]
mod protocol_tests;

#[cfg(test)]
#[path = "server_tests.rs"]
mod server_tests;
