//! Teams of agents: team cards and configuration, an in-process message bus, a
//! team runtime, and the `TeamStrategy` seam (handoff and hierarchical
//! strategies build on it).
//!
//! Maps to the Python package `core.multi_agent` in the openJiuwen reference
//! implementation: a `Team` owns a card, a [`TeamRuntime`], a [`MessageBus`]
//! transport and a strategy object that decides how members collaborate.
#![forbid(unsafe_code)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]
#![allow(clippy::module_name_repetitions)]

mod agent_tool;
mod card;
mod config;
mod envelope;
mod error;
mod handoff;
mod member;
mod message_bus;
mod runtime;
mod session;
mod strategy;
mod subscription;
mod team;

pub use agent_tool::AgentTool;
pub use card::{EventDrivenTeamCard, TeamCard};
pub use config::TeamConfig;
pub use envelope::MessageEnvelope;
pub use error::TeamError;
pub use member::{AgentProvider, BindHook, TeamAgent, ToolHost};
pub use message_bus::{MessageBus, MessageBusConfig, MessageHandler};
pub use runtime::{RuntimeConfig, TeamRuntime};
pub use session::{make_team_session, team_stream, SessionLease};
pub use strategy::{MemberOptions, StrategyContext, TeamStrategy};
pub use subscription::{topic_matches, SubscriptionManager};
pub use team::{BaseTeam, Team};
