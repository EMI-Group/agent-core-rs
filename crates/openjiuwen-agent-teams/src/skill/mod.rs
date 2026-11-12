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
//!
//! The clap command tree (`cli`) is added in a later stage; this module currently exposes what it
//! builds upon.

pub mod client;
pub mod descriptor;
pub mod store_client;

pub use client::{
    member_tools_for, operator_tools, CreateOutcome, MemberSummary, OpOutcome, SendOutcome,
    TeamClient, TeamToolDef,
};
pub use descriptor::{ClientScope, TeamJoinDescriptor};
pub use store_client::StoreTeamClient;
