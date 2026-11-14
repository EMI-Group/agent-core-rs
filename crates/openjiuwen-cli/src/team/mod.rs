//! `openjiuwen team` — operate a team hosted by another process.
//!
//! Cross-process teams meet over the `openjiuwen-agent-teams` attach protocol
//! (one JSON request/response per line on a Unix domain socket). This module is
//! a thin client: it resolves the socket, dials it with
//! [`openjiuwen_agent_teams::AttachClient`], maps each subcommand onto one
//! attach operation, and renders the answer. The team itself — its members, task
//! board, mailbox and leader — runs in the hosting process.
//!
//! The socket comes from `--socket`, else `OPENJIUWEN_TEAM_SOCKET`. The command
//! vocabulary follows the `team-member` binary's operator scope (`send`,
//! `broadcast`, `create-task`, `task list|claimable|get`, `claim`, `complete`,
//! `update`, `members`, `inbox`) and adds the two operations only the attach
//! protocol has: `snapshot` and `stop`.
use std::path::PathBuf;

use openjiuwen_common::error::Result;

/// `openjiuwen team [--socket PATH] [--json] <COMMAND>`.
#[derive(Debug, Clone, clap::Args)]
pub(crate) struct TeamCli {
    /// Attach socket of the hosting team.
    ///
    /// Falls back to `OPENJIUWEN_TEAM_SOCKET` when absent.
    #[arg(long, global = true, value_name = "PATH")]
    pub(crate) socket: Option<PathBuf>,
    /// Print machine-readable JSON instead of text.
    #[arg(long, global = true)]
    pub(crate) json: bool,
    /// The team operation to run.
    #[command(subcommand)]
    pub(crate) command: TeamCommand,
}

/// Attach operations a terminal user needs.
#[derive(Debug, Clone, clap::Subcommand)]
pub(crate) enum TeamCommand {
    /// List the team's members.
    Members,
    /// Show one member.
    Member {
        /// Member name.
        name: String,
    },
    /// Read and drain the leader's unread inbox.
    Inbox,
    /// Send a message to one member.
    Send {
        /// Recipient member name.
        to: String,
        /// Message text.
        content: String,
    },
    /// Send a message to every member.
    Broadcast {
        /// Message text.
        content: String,
    },
    /// Create a task on the team's board.
    CreateTask {
        /// Task title.
        title: String,
        /// Task content.
        content: String,
        /// Explicit task id; the team assigns one when omitted.
        #[arg(long = "task-id")]
        task_id: Option<String>,
        /// Id of a task this one is blocked by; repeatable.
        #[arg(long = "dep")]
        dependencies: Vec<String>,
    },
    /// Task-board reads.
    Task {
        /// The read to perform.
        #[command(subcommand)]
        command: TeamTaskCommand,
    },
    /// Claim a task for this caller.
    Claim {
        /// Task id.
        task_id: String,
    },
    /// Mark a claimed task completed.
    Complete {
        /// Task id.
        task_id: String,
    },
    /// Patch a task's title or content.
    Update {
        /// Task id.
        task_id: String,
        /// New title.
        #[arg(long)]
        title: Option<String>,
        /// New content.
        #[arg(long)]
        content: Option<String>,
    },
    /// Print the whole team snapshot (members, tasks, state).
    Snapshot,
    /// Ask the hosting team to stop.
    Stop,
}

/// Task-board reads.
#[derive(Debug, Clone, clap::Subcommand)]
pub(crate) enum TeamTaskCommand {
    /// List tasks, optionally filtered by status.
    List {
        /// Status filter: `pending`, `blocked`, `planning`, `in_progress`,
        /// `in_review`, `completed` or `cancelled`.
        #[arg(long)]
        status: Option<String>,
    },
    /// List the tasks this caller may claim.
    Claimable,
    /// Show one task in full.
    Get {
        /// Task id.
        task_id: String,
    },
}

/// Runs `openjiuwen team`.
///
/// # Errors
///
/// Returns a validation error when no socket is configured, the attach
/// transport's typed errors, and remote failures with their wire kind (for
/// example `not_found` for an unknown member or task).
pub(crate) async fn run(args: &TeamCli) -> Result<()> {
    let _ = args;
    Err(openjiuwen_common::error::Error::unsupported(
        "team command is not wired yet",
    ))
}
