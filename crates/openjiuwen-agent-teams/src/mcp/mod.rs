//! The `openjiuwen-team-mcp` stdio MCP server.
//!
//! Python parity source: `openjiuwen/agent_teams/mcp/server.py`. The server projects the team
//! collaboration surface to an agent that lives outside the team process, in one of two scenarios
//! selected by the join descriptor's scope:
//!
//! * `member` — an external CLI acting as a first-class teammate, driven by the real teammate tool
//!   set the connected client advertises (inbound messages are pushed by the parent process, so the
//!   operator-only pull tool is absent and no server-level instructions are shipped).
//! * `operator` (default) — an outside team-control surface over the task board, messaging and
//!   roster, with [`OPERATOR_INSTRUCTIONS`] describing the control workflow.
//!
//! This module owns no team logic: it re-exposes [`crate::skill::TeamClient`] operations and the
//! [`crate::skill::operator_tools`] catalogue. Every result is one plain-text block; failures are
//! ordinary text, never MCP protocol errors.
//!
//! Module map:
//!
//! | Module | Responsibility |
//! | --- | --- |
//! | [`dispatch`] | operator/member dispatch over [`TeamClient`], argument extraction, error text |
//! | [`server`] | `TeamServerHandler`, the `rmcp` server adapter (feature `mcp`) |
//! | [`stdio`] | `serve_stdio` and the environment-driven entry point (feature `mcp`) |
//! | `tests` | offline `FakeClient` suite (runs with and without the `mcp` feature) |
//!
//! The join descriptor and its `OPENJIUWEN_TEAM_JOIN` environment variable are owned by
//! [`crate::skill`]; the process wiring is the `openjiuwen-team-mcp` binary in this crate's
//! `src/bin/`.
pub use crate::skill::{ClientScope, TeamClient, TeamToolDef};
pub use dispatch::{dispatch_member, dispatch_operator};
use serde_json::Value;
use std::sync::Arc;
pub mod dispatch;
#[cfg(feature = "mcp")]
pub mod server;
#[cfg(feature = "mcp")]
pub mod stdio;
#[cfg(test)]
mod tests;
/// The MCP `serverInfo` name this server advertises.
///
/// The logical registration name CLIs use (`openjiuwen-team`) belongs to the spawn layer.
pub const SERVER_NAME: &str = "openjiuwen-team-member";
/// The operator-only inbox pull tool. Members receive their messages by push, so this tool is
/// deliberately absent from the member scope.
pub const READ_INBOX: &str = "read_inbox";
/// Server-level instructions for the `operator` scope — the control workflow for a non-member
/// interface driving the team.
///
/// The `member` scope ships none: the team system prompt is injected into the CLI at spawn time.
pub const OPERATOR_INSTRUCTIONS: &str =
    "You operate an OpenJiuWen agent team from outside it (you are not a team
member). Drive the team through these tools: read_inbox to see the mailbox and
task board, create_task to hand out work, update_task to edit / assign /
cancel, send_message to direct members (use \"*\" to broadcast), list_members to
see the roster, and the task-board tools. Refer to members by name.
";
/// Scope-dependent facade over one connected [`TeamClient`].
///
/// Cheap to clone and `Send + Sync`: the scope is resolved per request from the client, never
/// frozen at construction time (Python re-binds the client context on every tool call).
#[derive(Clone)]
pub struct TeamMcpServer {
    client: Arc<dyn TeamClient>,
}
impl TeamMcpServer {
    /// Wraps an already connected client.
    #[must_use]
    pub fn new(client: Arc<dyn TeamClient>) -> Self {
        Self { client }
    }
    /// Returns the connected client.
    #[must_use]
    pub fn client(&self) -> &Arc<dyn TeamClient> {
        &self.client
    }
    /// Returns the connected client's scope.
    #[must_use]
    pub fn scope(&self) -> ClientScope {
        self.client.scope()
    }
    /// Returns the tool definitions this server advertises.
    ///
    /// Member scope returns the client's own teammate tool set in client order; operator scope
    /// returns [`crate::skill::operator_tools`]. `read_inbox` is never injected for members.
    // Kept `async` by contract: the tool catalogue may become client-async as the skill node
    // completes, and every caller is already in an async context.
    #[allow(clippy::unused_async, clippy::unused_async_trait_impl)]
    pub async fn list_tools(&self) -> Vec<TeamToolDef> {
        match self.scope() {
            ClientScope::Operator => crate::skill::operator_tools(),
            ClientScope::Member => self.client.tools(),
        }
    }
    /// Runs one tool call and returns its single plain-text result.
    ///
    /// Failures are ordinary text (`Internal error: …`, `Unknown tool: …`), matching the Python
    /// server: the caller never sees an MCP protocol error.
    pub async fn call_tool(&self, name: &str, arguments: Value) -> String {
        let scope = self.scope();
        if name == READ_INBOX && !matches!(scope, ClientScope::Operator) {
            return format!("Unknown tool: {name}");
        }
        match scope {
            ClientScope::Operator => {
                dispatch_operator(self.client.as_ref(), name, &arguments).await
            }
            ClientScope::Member => dispatch_member(self.client.as_ref(), name, &arguments).await,
        }
    }
    /// Returns the server-level instructions for this scope.
    #[must_use]
    pub fn instructions(&self) -> Option<&'static str> {
        match self.scope() {
            ClientScope::Operator => Some(OPERATOR_INSTRUCTIONS),
            ClientScope::Member => None,
        }
    }
}
/// Builds the MCP server over `client`.
#[must_use]
pub fn build_mcp_server(client: Arc<dyn TeamClient>) -> TeamMcpServer {
    TeamMcpServer::new(client)
}
