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
mod message_bus;
mod subscription;

pub use card::{EventDrivenTeamCard, TeamCard};
pub use config::TeamConfig;
pub use envelope::MessageEnvelope;
pub use error::TeamError;
pub use message_bus::{MessageBus, MessageBusConfig, MessageHandler};
pub use subscription::{topic_matches, SubscriptionManager};
