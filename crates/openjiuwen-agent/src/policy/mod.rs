//! Tool policy: exposure policy, MCP allowlists, client tools, resilience, and
//! the pre-final-request check.
//!
//! These are the runtime policies Python keeps next to `AbilityManager`
//! (`set_tool_exposure_policy`, `set_mcp_tool_allowlist`, client tools,
//! per-tool resilience timeouts, and `FinalRequestCheck`). They are data and
//! decision helpers: the `ReAct` loop and the host own the wiring.
//!
//! - [`exposure`] — which tools the model may see, plus MCP tool allowlists.
//! - [`client_tools`] — handlers for tool calls the connected client executes.
//! - [`resilience`] — per-tool timeouts, attempt budgets, and backoff.
//! - [`final_check`] — the hook run before the final model request of a round.
pub mod client_tools;
pub mod exposure;
pub mod final_check;
pub mod resilience;
pub use client_tools::{ClientToolHandler, ClientToolRegistry, FnClientToolHandler};
pub use exposure::{McpToolAllowlist, ToolExposureMode, ToolExposurePolicy, ALL_TOOLS};
pub use final_check::{
    DefaultFinalRequestCheck, FinalRequestCheck, FinalRequestContext, FinalRequestDecision,
};
pub use resilience::{ToolResilience, ToolResiliencePolicy};
