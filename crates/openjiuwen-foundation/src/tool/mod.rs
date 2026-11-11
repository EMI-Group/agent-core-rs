//! Tools: cards, the [`traits::Tool`] contract, function tools, REST tools, and MCP.
//!
//! Ports `openjiuwen/core/foundation/tool/`. A tool is data ([`card::ToolCard`],
//! the LLM-facing projection) plus behavior (the [`traits::Tool`] impl, which may
//! hold transport details that never serialize).
//!
//! - [`card`] — `ToolCard`, `ToolInfo`, exposure/execution policy enums.
//! - [`context`] — `ToolContext` (invocation id, cancellation, session, deadline).
//! - [`traits`] — the `Tool` trait plus `ToolOutput` and the render helpers.
//! - [`function`] — `FunctionTool` and its builder (closure and typed handlers).
//! - [`schema`] — JSON-Schema derivation, default filling, and validation.
//! - [`restful`] — `RestfulApiTool` (OpenAPI/REST invocation).
//! - `mcp` — MCP client transports and `McpTool` (feature `mcp`).

/// Tool cards and the OpenAI tool projection.
pub mod card;
/// The invocation context handed to every tool call.
pub mod context;
/// Function tools: closure and typed handlers plus their builder.
pub mod function;
/// MCP client transports and tools (feature `mcp`).
#[cfg(feature = "mcp")]
pub mod mcp;
/// REST/OpenAPI tools and their parameter mapping.
pub mod restful;
/// JSON-Schema derivation, default filling, and input validation.
pub mod schema;
/// The `Tool` contract, `ToolOutput`, and the render helpers.
pub mod traits;

pub use card::{ToolCard, ToolExecution, ToolExposure, ToolFunctionInfo, ToolInfo};
pub use context::ToolContext;
pub use function::{FunctionTool, FunctionToolBuilder, ToolHandler};
#[cfg(feature = "mcp")]
pub use mcp::{
    connect_mcp_server, load_mcp_tools, McpClient, McpClientType, McpRemoteTool, McpServerConfig,
    McpTool, McpToolOptions, McpToolResult, RmcpClient,
};
pub use restful::{
    ApiParamMapper, HttpMethod, ParamLocation, ParamMap, RestfulApiConfig, RestfulApiTool,
    DEFAULT_MAX_RESPONSE_BYTES, DEFAULT_TIMEOUT,
};
pub use schema::{apply_defaults, ensure_object_schema, openai_tools, schema_for, validate_inputs};
pub use traits::{
    render_payload_text, render_tool_output, Tool, ToolOutput, EMPTY_FAILURE_TEXT,
    EMPTY_SUCCESS_TEXT,
};
