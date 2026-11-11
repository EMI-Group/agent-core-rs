//! Leader/teammate runtime, worktrees, HITL, and the skill CLI/MCP server.
//!
//! Maps to the Python package `agent_teams` in the openJiuwen reference implementation.

/// Returns the crate name.
pub fn crate_name() -> &'static str {
    "openjiuwen-agent-teams"
}
