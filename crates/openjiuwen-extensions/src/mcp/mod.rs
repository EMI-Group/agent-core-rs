//! MCP tools served to an external harness through the `ToolGateway` seam (feature `mcp`).
//!
//! MCP client/server tooling itself lives in `openjiuwen_foundation::tool::mcp`: the stdio, SSE,
//! streamable-HTTP and OpenAPI transports, the JSON-RPC session, and the `McpTool` projection of a
//! remote tool onto the kernel `Tool` contract. Enabling this crate's `mcp` feature only turns
//! that kernel feature on (`openjiuwen-foundation/mcp`), and the foundation default already
//! enables it, so no umbrella feature can turn MCP off.
//!
//! This module is therefore **not** a transport and adds no JSON-RPC of its own. It is the bridge
//! between the two surfaces that already exist: a set of kernel tools ([`openjiuwen_core::foundation::tool::Tool`]
//! objects, typically the `McpTool`s returned by
//! [`load_mcp_tools`](openjiuwen_core::foundation::tool::mcp::load_mcp_tools)) and the
//! provider-neutral [`ToolGateway`](openjiuwen_harness_protocol::ToolGateway) that an external
//! harness provider calls to enumerate and invoke native tools.
//!
//! Python parity: `openjiuwen.extensions` (v0.1.20) ships no MCP module, so there is no Python
//! counterpart to port; the counterpart of the *gateway* is the host service in
//! `openjiuwen.harness_protocol.tools`. The tables below are the contract.
//!
//! # Projection (`ToolCard` → `ToolDefinition`)
//!
//! | [`openjiuwen_core::foundation::tool::ToolCard`] | [`ToolDefinition`](openjiuwen_harness_protocol::ToolDefinition) |
//! |---|---|
//! | `name` | `name` |
//! | `description` | `description` |
//! | `input_params` when it is a JSON object | `input_schema` |
//! | `input_params` that is not a JSON object | `input_schema = {}` (accepts any arguments; the harness wire type only carries object schemas) |
//!
//! `ToolDefinition` has no metadata field, so the MCP provenance the kernel card carries (`mcp:
//! true`, `server_name`, `server_id`) plus the card id and the `parallel_safe` flag live on
//! [`McpToolEntry`], which [`McpToolGateway::entries`] hands out alongside the definition.
//! [`McpToolGateway::definitions`] is deterministic: tools are sorted by name (ascending) at
//! construction, so the same set of tools always produces the same list — unlike the kernel's
//! `Vec<McpTool>`, whose order follows the server's `tools/list` reply.
//!
//! # Invocation outcome (`openjiuwen_common::error::Error` → harness result)
//!
//! [`McpToolGateway::invoke`](openjiuwen_harness_protocol::ToolGateway::invoke) resolves the tool
//! by name and runs it with a fresh [`ToolContext`](openjiuwen_core::foundation::tool::ToolContext)
//! (the harness `ToolInvocation` carries no session and no deadline; the call id rides in the
//! context metadata under [`HARNESS_CALL_ID_METADATA_KEY`]).
//!
//! | Kernel outcome | Harness outcome |
//! |---|---|
//! | `Ok(value)` that does not read as a kernel `ToolOutput` | success: `content = {"text": <rendered>, "structured": <value>}` |
//! | `Ok(ToolOutput)` with `success: true` | success: `content = {"text": <rendered>, "structured": <ToolOutput.data>}` |
//! | `Ok(ToolOutput)` with `success: false` | error result (`is_error: true`), `content = {"error": "MCP tool \`<name>\` failed: <ToolOutput.error>"}` |
//! | `Err(Error::Validation)` | [`HarnessError::validation`](openjiuwen_harness_protocol::HarnessError::validation) naming the tool |
//! | `Err(Error::Unsupported)` | [`HarnessError::unsupported_capability`](openjiuwen_harness_protocol::HarnessError::unsupported_capability) naming the tool |
//! | `Err(Error::Config)` | [`HarnessError::protocol`](openjiuwen_harness_protocol::HarnessError::protocol) — the host cannot address the tool it advertised |
//! | `Err(Error::Cancelled)` | [`HarnessError::state`](openjiuwen_harness_protocol::HarnessError::state) — the call was cancelled before it produced a result |
//! | any other `Err` (`Tool`, `Mcp`, `Transport`, `Timeout`, `NotFound`, …) | error result (`is_error: true`), `content = {"error": "MCP tool \`<name>\` failed: <detail>"}` |
//!
//! A tool-level failure travels in the result channel (`is_error`) so the harness can show it to
//! the model and continue; only caller mistakes and unresolvable capability/config/state
//! mismatches become [`HarnessError`](openjiuwen_harness_protocol::HarnessError)s. Error text is
//! the kernel's own message, truncated to 512 characters; invocation arguments are never echoed.
//! An unknown tool name is a typed validation error.
//!
//! # Permissions
//!
//! `ToolGateway::invoke` is documented as executing a tool "under the host's permission policy".
//! This gateway owns no policy of its own: the optional [`McpToolGateway::with_permission`] filter
//! is the modelled seam (a tool the filter rejects is hidden from `definitions` and rejected with
//! a typed [`HarnessError`](openjiuwen_harness_protocol::HarnessError) if invoked anyway), and
//! without a filter every served tool is visible and runnable. Approval gates, workspace sandbox
//! rules and per-agent allow-lists stay with the harness rails that call the gateway.
//!
//! ```
//! use openjiuwen_extensions::mcp::McpToolGateway;
//!
//! // An empty gateway is valid: it advertises no tools and rejects every invocation.
//! let gateway = McpToolGateway::new(std::iter::empty()).expect("no tools is a valid gateway");
//! assert!(gateway.servers().is_empty());
//! ```
mod gateway;
#[cfg(test)]
mod tests;

pub use gateway::{
    McpToolEntry, McpToolGateway, PermissionFilter, HARNESS_CALL_ID_METADATA_KEY,
    MCP_SERVED_METADATA_KEY, MCP_SERVER_ID_METADATA_KEY, MCP_SERVER_NAME_METADATA_KEY,
};
