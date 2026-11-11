//! MCP client support (feature `mcp`).
//!
//! [`client`] owns the provider-neutral [`client::McpClient`] trait and the
//! `rmcp`-backed stdio / SSE / streamable-HTTP transports with reconnect;
//! [`tool`] wraps a remote tool as an [`crate::tool::traits::Tool`]; [`result`]
//! normalizes MCP call results.
//!
//! Only stdio and streamable-HTTP/SSE transports are in scope; the Python
//! `openapi`/`playwright` transports are deliberately not ported.
//!
//! The HTTP transports are implemented directly on `reqwest` and implement
//! [`rmcp::transport::Transport`] themselves: the workspace enables `rmcp`
//! without its `transport-*-reqwest` features, so `rmcp`'s own HTTP clients are
//! not compiled in. Everything above the transport (JSON-RPC framing, the
//! initialize handshake, request ids, cancellation) still comes from `rmcp`
//! through [`client::RmcpClient`].
mod http;
pub mod client;
pub mod result;
pub mod tool;

#[cfg(test)]
mod tests;

pub use client::{
    connect_mcp_server, McpClient, McpClientType, McpRemoteTool, McpServerConfig, RmcpClient,
};
pub use result::{
    extract_content, normalize_call_tool_result, McpToolResult, DEFAULT_MCP_ERROR_TEXT,
};
pub use tool::{load_mcp_tools, McpTool, McpToolOptions};

#[cfg(test)]
pub use client::FakeMcpClient;
