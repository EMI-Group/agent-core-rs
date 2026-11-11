//! LLM clients and `Model`, prompts/messages, tools and MCP, and storage backends.
//!
//! Maps to the Python package `core.foundation` in the openJiuwen reference implementation.

/// Returns the crate name.
pub fn crate_name() -> &'static str {
    "openjiuwen-foundation"
}
