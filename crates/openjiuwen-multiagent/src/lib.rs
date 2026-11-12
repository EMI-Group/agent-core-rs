//! Teams of agents: team cards and configuration, an in-process message bus, a
//! team runtime, and the strategy seam (`TeamStrategy`).
//!
//! Maps to the Python package `core.multi_agent` in the openJiuwen reference implementation.

#![forbid(unsafe_code)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]
// Card and config types are named `TeamCard`/`TeamConfig`, so the module name
// repeats in the item name; the qualified paths stay clearer than renaming them.
#![allow(clippy::module_name_repetitions)]

mod card;
mod config;
mod error;

pub use card::{EventDrivenTeamCard, TeamCard};
pub use config::TeamConfig;
pub use error::TeamError;
