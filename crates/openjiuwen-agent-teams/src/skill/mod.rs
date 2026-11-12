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
//!
//! The store-backed client (`store_client`) and the clap command tree (`cli`) are added in later
//! stages; this module currently exposes only what those stages build upon.

pub mod client;
pub mod descriptor;

pub use client::{
    member_tools_for, operator_tools, CreateOutcome, MemberSummary, OpOutcome, SendOutcome,
    TeamClient, TeamToolDef,
};
pub use descriptor::{ClientScope, TeamJoinDescriptor};
