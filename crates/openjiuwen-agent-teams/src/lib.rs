//! Leader/teammate multi-agent runtime.
//!
//! Maps to the Python package `openjiuwen.agent_teams` in the openJiuwen reference implementation.
//!
//! The crate is layered bottom-up; each module owns its node of the Context Tree and documents its
//! own API surface:
//!
//! | Module | Responsibility |
//! | --- | --- |
//! | [`constants`] | reserved member names, defaults, mention parsing |
//! | [`schema`] | specs, team runtime types, statuses, task/event/stream records |
//! | [`experts`] | `ExpertCatalog` seam and agent-group packages |
//! | [`storage`] | `TeamStore` seam and the in-memory store |
//! | [`taskboard`] | task tree, dependency graph, claims, reviews |
//! | [`messager`] | `Messager` seam and the in-process transport |
//! | [`workspace`] | member workspace allocation (local directories) |
//! | [`member`] | member runtime seam, status machine, spawn handles |
//! | [`interaction`] | HITL payloads, routing grammar, interact gate |
//! | [`runtime`] | `TeamAgent` leader/teammate loop, dispatch, runtime manager |
//! | [`harness`] | harness-protocol adapter and native member factory |
//! | [`skill`] | the `team-member` skill CLI and join descriptor |
//! | [`mcp`] | the `openjiuwen-team-mcp` MCP server |
//!
//! Everything is in-process and tokio-native; distributed transports, OS-process members and
//! external CLI members are trait seams that return [`openjiuwen_common::error::Error::Unsupported`].
//!
//! ```
//! use openjiuwen_agent_teams::DEFAULT_LEADER_MEMBER_NAME;
//! assert_eq!(DEFAULT_LEADER_MEMBER_NAME, "team_leader");
//! ```
#![forbid(unsafe_code)]
// Python package names (`agent_teams`, `team_agent`) repeat deliberately in module names.
#![allow(clippy::module_name_repetitions)]

pub mod constants;
pub mod experts;
pub mod harness;
pub mod interaction;
pub mod mcp;
pub mod member;
pub mod messager;
pub mod runtime;
pub mod schema;
pub mod skill;
pub mod storage;
pub mod taskboard;
pub mod workspace;

/// Returns the crate name.
#[must_use]
pub fn crate_name() -> &'static str {
    "openjiuwen-agent-teams"
}
