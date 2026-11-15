//! stdio transport and process entry points for the team MCP server.
//!
//! One platform-independent implementation serves over `rmcp::transport::stdio()`: the process's
//! `tokio::io::Stdin`/`Stdout` pair (the `io-std` feature) converts into an `rmcp` transport through
//! the `transport-async-rw` feature, on every target this crate builds for.
//!
//! The `openjiuwen-team-mcp` console entry point lives in this crate (`src/bin/`); this module
//! provides the entry points it calls.
use crate::mcp::server::TeamServerHandler;
use crate::mcp::{ClientScope, TeamClient, TeamMcpServer, SERVER_NAME};
use crate::skill::TeamJoinDescriptor;
use openjiuwen_common::error::{Error, Result};
use rmcp::ServiceExt;
use std::sync::Arc;
/// Serves the team MCP server over the process's stdio until the client disconnects.
///
/// # Errors
///
/// Returns [`Error::config`] when the MCP session fails to initialize or terminates abnormally.
pub async fn serve_stdio(server: TeamMcpServer) -> Result<()> {
    serve_impl(server).await
}
async fn serve_impl(server: TeamMcpServer) -> Result<()> {
    let handler = TeamServerHandler::new(server);
    let service = handler
        .serve(rmcp::transport::stdio())
        .await
        .map_err(|error| Error::config(format!("MCP server initialization failed: {error}")))?;
    service
        .waiting()
        .await
        .map_err(|error| Error::config(format!("MCP server task failed: {error}")))?;
    Ok(())
}
/// Resolves the join descriptor from `OPENJIUWEN_TEAM_JOIN` and serves over stdio.
///
/// The server attaches to a team hosted in *this* process through the process-global runtime
/// registry, or to one hosted by *another* process over that host's Unix-socket attach bridge
/// (`crate::attach`); `resolve_from_registry` goes through
/// [`crate::skill::connect_live_client`], the single resolution point both routes share. A team
/// neither route reaches is [`Error::config`] and exits `1`.
///
/// Returns the process exit code: `0` after a clean session, `1` when the descriptor or the client
/// cannot be resolved or the session fails.
pub async fn run_mcp_server_from_env() -> i32 {
    let descriptor = match TeamJoinDescriptor::from_env() {
        Ok(descriptor) => descriptor,
        Err(error) => {
            tracing::error!(
                error = %error,
                "openjiuwen-team-mcp: invalid team join descriptor"
            );
            return 1;
        }
    };
    run_mcp_server_with_descriptor(descriptor, resolve_from_registry).await
}
/// Resolves the join descriptor from the environment and serves the client `resolver` returns.
///
/// `resolver` is the seam used by the CLI and by tests to inject a pre-built client. The default
/// `resolve_from_registry` is [`crate::skill::connect_live_client`]: a team hosted in this process
/// or one reachable over a published attach socket.
///
/// Returns the process exit code: `0` on success, `1` on a descriptor/resolver/session failure.
pub async fn run_mcp_server_with<F>(resolver: F) -> i32
where
    F: FnOnce(&TeamJoinDescriptor) -> Result<Arc<dyn TeamClient>>,
{
    let descriptor = match TeamJoinDescriptor::from_env() {
        Ok(descriptor) => descriptor,
        Err(error) => {
            tracing::error!(
                error = %error,
                "openjiuwen-team-mcp: invalid team join descriptor"
            );
            return 1;
        }
    };
    run_mcp_server_with_descriptor(descriptor, resolver).await
}
/// Serves the client `resolver` returns for an already loaded `descriptor`.
///
/// The environment-free seam behind [`run_mcp_server_from_env`] and [`run_mcp_server_with`], so the
/// resolver-`Err` exit path is testable without mutating the process environment.
///
/// Returns the process exit code: `0` on success, `1` on a resolver or session failure.
pub(crate) async fn run_mcp_server_with_descriptor<F>(
    descriptor: TeamJoinDescriptor,
    resolver: F,
) -> i32
where
    F: FnOnce(&TeamJoinDescriptor) -> Result<Arc<dyn TeamClient>>,
{
    let client = match resolver(&descriptor) {
        Ok(client) => client,
        Err(error) => {
            tracing::error!(
                error = %error,
                "openjiuwen-team-mcp: cannot resolve the team client"
            );
            return 1;
        }
    };
    let server = TeamMcpServer::new(client);
    tracing::info!(
        server = SERVER_NAME,
        scope = scope_label(server.scope()),
        "openjiuwen-team-mcp: serving over stdio"
    );
    match serve_stdio(server).await {
        Ok(()) => 0,
        Err(error) => {
            tracing::error!(error = %error, "openjiuwen-team-mcp: stdio server failed");
            1
        }
    }
}
/// Default client resolver: the team `descriptor` names, when this process hosts it.
pub(crate) fn resolve_from_registry(
    descriptor: &TeamJoinDescriptor,
) -> Result<Arc<dyn TeamClient>> {
    crate::skill::connect_live_client(descriptor).ok_or_else(|| {
        Error::config(format!(
            "no live team '{}' hosted in this process",
            descriptor.team_name
        ))
    })
}
/// Returns the descriptor wire name of `scope`, for logs.
fn scope_label(scope: ClientScope) -> &'static str {
    match scope {
        ClientScope::Operator => "operator",
        ClientScope::Member => "member",
    }
}
