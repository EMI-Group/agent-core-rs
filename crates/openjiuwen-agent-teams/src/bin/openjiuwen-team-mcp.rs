//! `openjiuwen-team-mcp`: the stdio entry point of the team MCP server.
//!
//! The server speaks the Model Context Protocol over this process's stdin/stdout and identifies
//! itself as `openjiuwen-team-member` ([`openjiuwen_agent_teams::mcp::SERVER_NAME`]). It attaches
//! only to a team hosted in *this* process: the join descriptor comes from `OPENJIUWEN_TEAM_JOIN`
//! and is resolved through the process-global runtime registry, so a team this process does not
//! host is exit `1`. Cross-process attach (a shared store or a distributed transport) is deferred.
//!
//! The exit code is the library's: `0` after a clean session, `1` when the descriptor, the client
//! or the session fails. Every branch is owned by
//! [`openjiuwen_agent_teams::mcp::stdio::run_mcp_server_from_env`]; this `main` only installs the
//! log subscriber, builds the async runtime around it and forwards the code.
//!
//! The default subscriber writes the human-readable log stream to **stderr**, because stdout is the
//! MCP wire: a descriptor, client or session failure therefore now produces a stderr diagnostic
//! while stdout stays clean. [`openjiuwen_common::logging::init_with`] is idempotent, and a
//! logging setup failure never prevents serving.
use openjiuwen_common::logging::{LogWriter, LoggingConfig};

#[tokio::main]
async fn main() {
    let _log_init = openjiuwen_common::logging::init_with(
        &LoggingConfig::from_env().with_writer(LogWriter::Stderr),
    );
    let code = openjiuwen_agent_teams::mcp::stdio::run_mcp_server_from_env().await;
    std::process::exit(code);
}
