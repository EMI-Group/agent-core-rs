//! Cross-process team attach over Unix domain sockets.
//!
//! A team hosted by one `openjiuwen` process can be driven by another process — the `team-member`
//! skill CLI or the `openjiuwen-team-mcp` server — through a small request/response protocol on a
//! Unix socket.
//! This directory owns the platform-independent half of that bridge:
//! the wire contract in [`protocol`], which the socket server and client are built on top of.
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
//! The protocol layer here is platform independent and compiles everywhere; the socket server and
//! client constructors built on top of it return `Error::Unsupported` with the message
//! "cross-process attach requires Unix domain sockets" on non-Unix targets.
pub mod protocol;

pub use protocol::{
    AttachError, AttachHandler, AttachOp, AttachResult, RequestFrame, ResponseFrame,
    MAX_FRAME_BYTES, PROTOCOL_VERSION,
};

#[cfg(test)]
#[path = "protocol_tests.rs"]
mod protocol_tests;
