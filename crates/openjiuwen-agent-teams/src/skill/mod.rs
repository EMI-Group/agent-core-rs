//! The `team-member` skill CLI: join a team from an external process and drive it.
//!
//! Python parity source: `openjiuwen/agent_teams/skill/cli.py` plus
//! `openjiuwen/agent_teams/external/` (descriptor and client). The module is split so each part can
//! be tested without a running team:
//!
//! | Module | Responsibility |
//! | --- | --- |
//! | [`descriptor`] | [`TeamJoinDescriptor`]: identity, [`ClientScope`] and team settings |
//! | [`client`] | [`TeamClient`] seam, outcome types and the tool catalogue per scope |
//! | [`store_client`] | [`StoreTeamClient`]: the store/task-board/messager implementation |
//! | [`live`] | live-team resolution through the process-global runtime registry |
//! | `cli` | the `team-member` command tree, its two-stage parse and the process entry point |

pub mod cli;
pub mod client;
pub mod descriptor;
pub mod live;
pub mod store_client;

pub use cli::{run_skill_cli, DescriptorArgs, SkillCli};
pub use client::{
    member_tools_for, operator_tools, CreateOutcome, MemberSummary, OpOutcome, SendOutcome,
    TeamClient, TeamToolDef,
};
pub use descriptor::{ClientScope, TeamJoinDescriptor};
pub use live::connect_live_client;
pub use store_client::StoreTeamClient;
