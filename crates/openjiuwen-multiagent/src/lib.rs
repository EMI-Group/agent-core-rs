//! `BaseTeam`, team runtime, message bus, and strategies.
//!
//! Maps to the Python package `core.multi_agent` in the openJiuwen reference implementation.

/// Returns the crate name.
pub fn crate_name() -> &'static str {
    "openjiuwen-multiagent"
}

mod envelope;
mod subscription;

pub use envelope::MessageEnvelope;
pub use subscription::{topic_matches, SubscriptionManager};
