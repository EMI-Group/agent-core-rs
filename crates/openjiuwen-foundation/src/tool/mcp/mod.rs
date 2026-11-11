//! MCP client support (feature `mcp`).
//!
//! [`client`] owns the provider-neutral [`client::McpClient`] trait and the
//! `rmcp`-backed stdio / SSE / streamable-HTTP transports with reconnect;
//! [`tool`] wraps a remote tool as an [`crate::tool::traits::Tool`]; [`result`]
//! normalizes MCP call results.
//!
//! Only stdio and streamable-HTTP/SSE transports are in scope; the Python
//! `openapi`/`playwright` transports are deliberately not ported.

pub mod client;
pub mod result;
pub mod tool;
