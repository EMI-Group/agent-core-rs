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
//! - [`mcp`] — MCP client transports and `McpTool` (feature `mcp`).

pub mod card;
pub mod context;
pub mod function;
#[cfg(feature = "mcp")]
pub mod mcp;
pub mod restful;
pub mod schema;
pub mod traits;
