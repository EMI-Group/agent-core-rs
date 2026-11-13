//! The `team-member` skill CLI: join a team from an external process and drive it.
//!
//! Python parity source: `openjiuwen/agent_teams/skill/cli.py` plus
//! `openjiuwen/agent_teams/external/` (descriptor and client). The module is split so each part can
//! be tested without a running team:
//!
//! | Module | Responsibility |
//! | --- | --- |
//! | [`descriptor`] | [`TeamJoinDescriptor`]: identity, [`ClientScope`], team settings and the attach socket |
//! | [`client`] | [`TeamClient`] seam, outcome types and the tool catalogue per scope |
//! | [`store_client`] | [`StoreTeamClient`]: the store/task-board/messager implementation |
//! | [`socket_client`] | [`SocketTeamClient`]: the same seam over the attach socket of a team hosted elsewhere |
//! | [`live`] | live-team resolution: this process's registry first, then a published attach socket |
//! | `cli` | the `team-member` command tree, its two-stage parse and the process entry point |

pub mod cli;
pub mod client;
pub mod descriptor;
pub mod live;
pub mod socket_client;
pub mod store_client;

pub use cli::{
    run_skill_cli, ClaimStatus, CliCommand, DescriptorArgs, MemberCommand, OperatorCommand,
    SkillCli, TaskCommand, ViewAction,
};
pub use client::{
    member_tools_for, operator_tools, CreateOutcome, MemberSummary, OpOutcome, SendOutcome,
    TeamClient, TeamToolDef,
};
pub use descriptor::{ClientScope, TeamJoinDescriptor};
pub use live::{connect_live_client, connect_socket_client};
pub use socket_client::SocketTeamClient;
pub use store_client::StoreTeamClient;

#[cfg(test)]
#[path = "socket_client_tests.rs"]
mod socket_client_tests;

#[cfg(test)]
#[path = "live_socket_tests.rs"]
mod live_socket_tests;
