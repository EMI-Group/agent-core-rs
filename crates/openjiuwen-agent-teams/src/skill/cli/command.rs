//! The clap command tree of `team-member`: one root parser per client scope.
//!
//! Python parity source: `_build_member_parser` and `_build_operator_parser` in
//! `openjiuwen/agent_teams/skill/cli.py`. The two roots carry the same connection flags, marked
//! `global`, so `--descriptor-file`/`--descriptor-json` are accepted both before and after the
//! subcommand exactly as Python's parent parser accepts them.
//!
//! The scope is chosen by the join descriptor, which is read *before* this module runs (see
//! [`super::SkillCli::parse_descriptor_args`]), so the caller picks the root: [`parse_member`] or
//! [`parse_operator`].

use clap::{Parser, Subcommand, ValueEnum};

use openjiuwen_common::error::{Error, Result};

use super::DescriptorArgs;
use crate::skill::descriptor::ClientScope;

/// Program name every parser and usage line carries.
pub(crate) const PROG: &str = "team-member";

/// The board view a member-scope `view_task` requests.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, ValueEnum)]
pub enum ViewAction {
    /// List the board (`list`, the default).
    #[default]
    List,
    /// Show one task in detail (`get`).
    Get,
    /// List the pending tasks a member may claim (`claimable`).
    Claimable,
}

impl ViewAction {
    /// Returns the tool argument spelling of this action.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::List => "list",
            Self::Get => "get",
            Self::Claimable => "claimable",
        }
    }
}

/// The status a member-scope `claim_task` drives a task to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ClaimStatus {
    /// Claim a pending task (`claimed`).
    Claimed,
    /// Complete a task the member holds (`completed`).
    Completed,
}

impl ClaimStatus {
    /// Returns the tool argument spelling of this status.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Claimed => "claimed",
            Self::Completed => "completed",
        }
    }
}

/// Member-scope command words (the teammate tool set plus `inbox`).
#[derive(Debug, Clone, PartialEq, Eq, Subcommand)]
pub enum MemberCommand {
    /// Show unread messages and the task board.
    #[command(about = "Show unread messages and the task board.")]
    Inbox {
        /// Accepted for parity; the read is one-shot (streaming is deferred).
        #[arg(long)]
        watch: bool,
    },
    /// View tasks (list / get / claimable).
    #[command(name = "view_task", about = "View tasks (list / get / claimable).")]
    ViewTask {
        /// Which view to request.
        #[arg(long, value_enum, default_value = "list")]
        action: ViewAction,
        /// Task id (required for `action=get`).
        #[arg(long = "task-id")]
        task_id: Option<String>,
        /// Status filter for `action=list`.
        #[arg(long)]
        status: Option<String>,
    },
    /// Claim a task, or complete one the member holds.
    #[command(name = "claim_task", about = "Claim a task, or complete one you hold.")]
    ClaimTask {
        /// Task identifier.
        task_id: String,
        /// Target status.
        status: ClaimStatus,
    },
    /// Send a message to a member, or broadcast with `*`.
    #[command(name = "send_message", about = "Send a message (\"*\" to broadcast).")]
    SendMessage {
        /// Recipient member name, or `*` to broadcast.
        to: String,
        /// Message body.
        content: String,
    },
}

/// Operator-scope task subcommands.
#[derive(Debug, Clone, PartialEq, Eq, Subcommand)]
pub enum TaskCommand {
    /// List tasks, optionally filtered by status.
    #[command(about = "List tasks.")]
    List {
        /// Filter by task status.
        #[arg(long)]
        status: Option<String>,
    },
    /// List the pending tasks available to claim.
    #[command(about = "List pending tasks available to claim.")]
    Claimable,
    /// Show one task in detail.
    #[command(about = "Show one task in detail.")]
    Get {
        /// Task identifier.
        task_id: String,
    },
}

/// Operator-scope command words (the outside team-control surface).
#[derive(Debug, Clone, PartialEq, Eq, Subcommand)]
pub enum OperatorCommand {
    /// Show unread messages and the task board.
    #[command(about = "Show unread messages and the task board.")]
    Inbox {
        /// Accepted for parity; the read is one-shot (streaming is deferred).
        #[arg(long)]
        watch: bool,
    },
    /// Send a direct message to a member.
    #[command(about = "Send a direct message to a member.")]
    Send {
        /// Recipient member name.
        to: String,
        /// Message body.
        content: String,
    },
    /// Broadcast a message to the whole team.
    #[command(about = "Broadcast a message to the whole team.")]
    Broadcast {
        /// Message body.
        content: String,
    },
    /// Create a team task.
    #[command(name = "create_task", about = "Create a team task.")]
    CreateTask {
        /// Task title.
        title: String,
        /// Task content / acceptance criteria.
        content: String,
        /// Optional pinned task id.
        #[arg(long = "task-id")]
        task_id: Option<String>,
    },
    /// Inspect the task board.
    #[command(about = "Inspect the task board.")]
    Task {
        /// Which board view to run.
        #[command(subcommand)]
        action: TaskCommand,
    },
    /// Claim a pending task.
    #[command(about = "Claim a pending task.")]
    Claim {
        /// Task identifier.
        task_id: String,
    },
    /// Complete a claimed task.
    #[command(about = "Complete a claimed task.")]
    Complete {
        /// Task identifier.
        task_id: String,
    },
    /// Edit a task's title and/or content.
    #[command(about = "Edit a task's title and/or content.")]
    Update {
        /// Task identifier.
        task_id: String,
        /// New title.
        #[arg(long)]
        title: Option<String>,
        /// New content.
        #[arg(long)]
        content: Option<String>,
    },
    /// List team members.
    #[command(about = "List team members.")]
    Members,
}

/// The member-scope parser: connection flags plus the member command words.
#[derive(Debug, Parser)]
#[command(
    name = PROG,
    about = "Participate in an agent team as a member.",
    disable_help_subcommand = true
)]
struct MemberCli {
    /// Path to a JSON team join descriptor.
    #[arg(long = "descriptor-file", global = true, value_name = "PATH")]
    descriptor_file: Option<String>,
    /// Inline JSON team join descriptor.
    #[arg(long = "descriptor-json", global = true, value_name = "JSON")]
    descriptor_json: Option<String>,
    /// The operation to perform.
    #[command(subcommand)]
    command: MemberCommand,
}

/// The operator-scope parser: connection flags plus the operator command words.
#[derive(Debug, Parser)]
#[command(
    name = PROG,
    about = "Operate an agent team from outside it.",
    disable_help_subcommand = true
)]
struct OperatorCli {
    /// Path to a JSON team join descriptor.
    #[arg(long = "descriptor-file", global = true, value_name = "PATH")]
    descriptor_file: Option<String>,
    /// Inline JSON team join descriptor.
    #[arg(long = "descriptor-json", global = true, value_name = "JSON")]
    descriptor_json: Option<String>,
    /// The operation to perform.
    #[command(subcommand)]
    command: OperatorCommand,
}

/// Parses one member-scope invocation, returning its connection flags and its command.
///
/// # Errors
///
/// Returns [`Error::Validation`] carrying clap's message for an unknown subcommand, an unknown flag
/// or a missing required argument.
pub(crate) fn parse_member(argv: Vec<String>) -> Result<(DescriptorArgs, MemberCommand)> {
    let parsed = MemberCli::try_parse_from(argv).map_err(|err| clap_error(&err))?;
    Ok((
        DescriptorArgs {
            descriptor_file: parsed.descriptor_file,
            descriptor_json: parsed.descriptor_json,
        },
        parsed.command,
    ))
}

/// Parses one operator-scope invocation, returning its connection flags and its command.
///
/// # Errors
///
/// Returns [`Error::Validation`] carrying clap's message for an unknown subcommand, an unknown flag
/// or a missing required argument.
pub(crate) fn parse_operator(argv: Vec<String>) -> Result<(DescriptorArgs, OperatorCommand)> {
    let parsed = OperatorCli::try_parse_from(argv).map_err(|err| clap_error(&err))?;
    Ok((
        DescriptorArgs {
            descriptor_file: parsed.descriptor_file,
            descriptor_json: parsed.descriptor_json,
        },
        parsed.command,
    ))
}

/// Maps a clap parse failure onto the crate error, keeping clap's message (which names the offending
/// token).
fn clap_error(err: &clap::Error) -> Error {
    Error::validation(err.to_string())
}

/// Renders the help text `argv` asks for, or `None` when it asks for none.
///
/// `--help`/`-h` is answered by clap itself, which reports it as a [`clap::error::ErrorKind`]
/// display error rather than a parse success. Rendering it separately from
/// [`parse_member`]/[`parse_operator`] lets the caller print help on stdout and exit `0` (Python
/// argparse parity) before it needs a join descriptor, while every other failure keeps travelling
/// the [`clap_error`] path.
/// Only an explicit `--help`/`-h` counts: `DisplayHelpOnMissingArgumentOrSubcommand` (a bare
/// invocation with no subcommand) is *not* help, because Python argparse reports a missing
/// subcommand as a usage error on stderr with exit `2`.
#[must_use]
pub(crate) fn display_text(argv: Vec<String>, scope: ClientScope) -> Option<String> {
    let err = match scope {
        ClientScope::Member => MemberCli::try_parse_from(argv).err()?,
        ClientScope::Operator => OperatorCli::try_parse_from(argv).err()?,
    };
    match err.kind() {
        clap::error::ErrorKind::DisplayHelp | clap::error::ErrorKind::DisplayVersion => {
            Some(err.to_string())
        }
        _ => None,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// Builds a process-style argument vector: program name first, then `args`.
    fn argv(args: &[&str]) -> Vec<String> {
        std::iter::once(PROG.to_string())
            .chain(args.iter().map(|arg| (*arg).to_string()))
            .collect()
    }

    fn member(args: &[&str]) -> (DescriptorArgs, MemberCommand) {
        parse_member(argv(args)).unwrap()
    }

    fn operator(args: &[&str]) -> (DescriptorArgs, OperatorCommand) {
        parse_operator(argv(args)).unwrap()
    }

    #[test]
    fn member_subcommands_parse() {
        assert_eq!(member(&["inbox"]).1, MemberCommand::Inbox { watch: false });
        assert_eq!(
            member(&["inbox", "--watch"]).1,
            MemberCommand::Inbox { watch: true }
        );
        assert_eq!(
            member(&["view_task"]).1,
            MemberCommand::ViewTask {
                action: ViewAction::List,
                task_id: None,
                status: None,
            }
        );
        assert_eq!(
            member(&[
                "view_task",
                "--action",
                "get",
                "--task-id",
                "t-1",
                "--status",
                "pending",
            ])
            .1,
            MemberCommand::ViewTask {
                action: ViewAction::Get,
                task_id: Some("t-1".to_string()),
                status: Some("pending".to_string()),
            }
        );
        assert_eq!(
            member(&["view_task", "--action", "claimable"]).1,
            MemberCommand::ViewTask {
                action: ViewAction::Claimable,
                task_id: None,
                status: None,
            }
        );
        assert_eq!(
            member(&["claim_task", "t-1", "claimed"]).1,
            MemberCommand::ClaimTask {
                task_id: "t-1".to_string(),
                status: ClaimStatus::Claimed,
            }
        );
        assert_eq!(
            member(&["claim_task", "t-1", "completed"]).1,
            MemberCommand::ClaimTask {
                task_id: "t-1".to_string(),
                status: ClaimStatus::Completed,
            }
        );
        assert_eq!(
            member(&["send_message", "team_leader", "done"]).1,
            MemberCommand::SendMessage {
                to: "team_leader".to_string(),
                content: "done".to_string(),
            }
        );
    }

    #[test]
    fn operator_subcommands_parse() {
        assert_eq!(
            operator(&["inbox", "--watch"]).1,
            OperatorCommand::Inbox { watch: true }
        );
        assert_eq!(
            operator(&["send", "dev-1", "hi"]).1,
            OperatorCommand::Send {
                to: "dev-1".to_string(),
                content: "hi".to_string(),
            }
        );
        assert_eq!(
            operator(&["broadcast", "all hands"]).1,
            OperatorCommand::Broadcast {
                content: "all hands".to_string(),
            }
        );
        assert_eq!(
            operator(&["create_task", "title", "body"]).1,
            OperatorCommand::CreateTask {
                title: "title".to_string(),
                content: "body".to_string(),
                task_id: None,
            }
        );
        assert_eq!(
            operator(&["create_task", "title", "body", "--task-id", "t-9"]).1,
            OperatorCommand::CreateTask {
                title: "title".to_string(),
                content: "body".to_string(),
                task_id: Some("t-9".to_string()),
            }
        );
        assert_eq!(
            operator(&["task", "list"]).1,
            OperatorCommand::Task {
                action: TaskCommand::List { status: None },
            }
        );
        assert_eq!(
            operator(&["task", "list", "--status", "pending"]).1,
            OperatorCommand::Task {
                action: TaskCommand::List {
                    status: Some("pending".to_string()),
                },
            }
        );
        assert_eq!(
            operator(&["task", "claimable"]).1,
            OperatorCommand::Task {
                action: TaskCommand::Claimable,
            }
        );
        assert_eq!(
            operator(&["task", "get", "t-1"]).1,
            OperatorCommand::Task {
                action: TaskCommand::Get {
                    task_id: "t-1".to_string(),
                },
            }
        );
        assert_eq!(
            operator(&["claim", "t-1"]).1,
            OperatorCommand::Claim {
                task_id: "t-1".to_string(),
            }
        );
        assert_eq!(
            operator(&["complete", "t-1"]).1,
            OperatorCommand::Complete {
                task_id: "t-1".to_string(),
            }
        );
        assert_eq!(
            operator(&["update", "t-1", "--title", "new", "--content", "body"]).1,
            OperatorCommand::Update {
                task_id: "t-1".to_string(),
                title: Some("new".to_string()),
                content: Some("body".to_string()),
            }
        );
        assert_eq!(operator(&["members"]).1, OperatorCommand::Members);
    }

    #[test]
    fn connection_flags_parse_before_and_after_the_subcommand() {
        let before = operator(&["--descriptor-json", "{\"team_name\":\"alpha\"}", "members"]);
        assert_eq!(
            before.0.descriptor_json.as_deref(),
            Some("{\"team_name\":\"alpha\"}")
        );
        assert_eq!(before.1, OperatorCommand::Members);

        let after = operator(&["members", "--descriptor-json", "{\"team_name\":\"alpha\"}"]);
        assert_eq!(
            after.0.descriptor_json.as_deref(),
            Some("{\"team_name\":\"alpha\"}")
        );
        assert_eq!(after.1, OperatorCommand::Members);

        let member_after = member(&["inbox", "--descriptor-file", "join.json"]);
        assert_eq!(member_after.0.descriptor_file.as_deref(), Some("join.json"));
        assert_eq!(member_after.1, MemberCommand::Inbox { watch: false });
    }

    #[test]
    fn unknown_subcommand_names_the_token() {
        let err = parse_operator(argv(&["explode"])).unwrap_err();
        assert!(
            err.to_string().contains("explode"),
            "unexpected message: {err}"
        );
        assert!(err.to_string().starts_with("validation: "));

        let member_err = parse_member(argv(&["send", "dev-1", "hi"])).unwrap_err();
        assert!(
            member_err.to_string().contains("send"),
            "unexpected message: {member_err}"
        );
    }

    #[test]
    fn missing_required_argument_is_a_validation_error() {
        let err = parse_member(argv(&["claim_task"])).unwrap_err();
        assert!(err.to_string().starts_with("validation: "));
        assert!(
            err.to_string().contains("TASK_ID"),
            "unexpected message: {err}"
        );
    }
}
