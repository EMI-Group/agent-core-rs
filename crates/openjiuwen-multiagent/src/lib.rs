//! Teams of agents: team cards and configuration, an in-process message bus, a
//! team runtime, and the strategy seam (`TeamStrategy`).
//!
//! Maps to the Python package `core.multi_agent` in the openJiuwen reference implementation.
#![forbid(unsafe_code)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]
#![allow(clippy::module_name_repetitions)]
mod card;
mod config;
mod envelope;
mod error;
mod member;
mod message_bus;
mod runtime;
mod strategy;
mod subscription;
pub use card::{EventDrivenTeamCard, TeamCard};
pub use config::TeamConfig;
pub use envelope::MessageEnvelope;
pub use error::TeamError;
pub use member::{AgentProvider, BindHook, TeamAgent, ToolHost};
pub use message_bus::{MessageBus, MessageBusConfig, MessageHandler};
pub use runtime::{RuntimeConfig, TeamRuntime};
pub use strategy::{MemberOptions, StrategyContext, TeamStrategy};
pub use subscription::{topic_matches, SubscriptionManager};

mod agent_tool;
mod team;

pub use agent_tool::AgentTool;
pub use team::{BaseTeam, Team};
