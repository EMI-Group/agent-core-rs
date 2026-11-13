//! MCP client support (feature `mcp`).
//!
//! [`client`](crate::tool::mcp::client) owns the provider-neutral [`McpClient`]
//! trait and the `rmcp`-backed stdio transport plus the SSE / streamable-HTTP
//! transports with reconnect; [`tool`](crate::tool::mcp::tool) wraps a remote
//! tool in a [`Tool`]; [`result`](crate::tool::mcp::result) normalizes MCP call
//! results.
//!
//! The streamable-HTTP transport also opens the server-initiated `GET` event
//! stream once the session id is known, so server notifications and
//! server-initiated requests arrive: notifications are buffered and drained
//! through [`McpClient::take_notifications`], and
//! [`McpClient::supports_server_stream`] reports the negotiated capability.
//!
//! Only stdio and streamable-HTTP/SSE transports are in scope; the Python
//! `openapi`/`playwright` transports are deliberately not ported.
//!
//! The HTTP transports are implemented directly on `reqwest` and implement
//! [`rmcp::transport::Transport`] themselves: the workspace enables `rmcp`
//! without its `transport-*-reqwest` features, so `rmcp`'s own HTTP clients are
//! not compiled in. Everything above the transport (JSON-RPC framing, the
//! initialize handshake, request ids, cancellation) still comes from `rmcp`
//! through [`RmcpClient`].
pub mod client;
mod http;
pub mod result;
pub mod tool;

#[cfg(test)]
mod stream_tests;
#[cfg(test)]
mod tests;

pub use client::{
    connect_mcp_server, McpClient, McpClientType, McpNotification, McpRemoteTool, McpServerConfig,
    RmcpClient,
};
pub use result::{
    extract_content, normalize_call_tool_result, McpToolResult, DEFAULT_MCP_ERROR_TEXT,
};
pub use tool::{load_mcp_tools, McpTool, McpToolOptions};

#[cfg(test)]
pub use client::FakeMcpClient;
