//! Top-level umbrella crate: the Rust counterpart of the Python package root `openjiuwen`.
//!
//! This crate is a pure facade. Every item it exposes is a re-export of a concrete
//! workspace crate, so code can keep the Python-shaped dotted paths it uses in the
//! reference SDK (`openjiuwen::harness::DeepAgent`,
//! `openjiuwen::core::single_agent::AgentCard`, …) without depending on each crate
//! individually. The concrete crates remain canonical: they own the documentation,
//! behavior, semver and tests.
//!
//! # Python → Rust module aliases
//!
//! | Python | Rust | Concrete crate |
//! | --- | --- | --- |
//! | `openjiuwen.core` | [`core`] | `openjiuwen-core` (kernel umbrella) |
//! | `openjiuwen.harness` | [`harness`] | `openjiuwen-harness` |
//! | `openjiuwen.harness_protocol` | [`harness_protocol`] | `openjiuwen-harness-protocol` |
//! | `openjiuwen.harness_providers` | [`harness_providers`] | `openjiuwen-harness-providers` |
//! | `openjiuwen.agent_teams` | [`agent_teams`] | `openjiuwen-agent-teams` |
//! | `openjiuwen.extensions` | [`extensions`] | `openjiuwen-extensions` |
//! | `openjiuwen.dev_tools` | [`dev_tools`] | `openjiuwen-dev-tools` |
//! | `openjiuwen.symphony` | [`symphony`] | `openjiuwen-symphony` |
//! | `openjiuwen.rsi` | [`rsi`] | `openjiuwen-rsi` |
//! | `openjiuwen.agent_evolving` | [`agent_evolving`] | `openjiuwen-agent-evolving` |
//! | `openjiuwen.auto_harness` | [`auto_harness`] | `openjiuwen-auto-harness` (shim over `rsi::auto_harness`) |
//!
//! # Feature matrix
//!
//! | Feature | Default | Enables | Effect |
//! | --- | --- | --- | --- |
//! | `native` | yes | `openjiuwen-harness/native`, `openjiuwen-harness-providers/native` | in-process native `DeepAgent` harness and provider |
//! | `otel` | no | `openjiuwen-extensions/otel` | live OpenTelemetry exporters and sink |
//! | `a2a` | no | `openjiuwen-extensions/a2a` | Agent-to-Agent client/server modules |
//! | `sqlite` | no | `openjiuwen-extensions/sqlite`, `openjiuwen-symphony/sqlite` | optional SQLite checkpointer, stores, and asset store |
//! | `live-tests` | no | `openjiuwen-harness/live-tests`, `openjiuwen-harness-providers/live-tests`, `openjiuwen-agent-teams/live-tests`, `openjiuwen-extensions/live-tests` | opt-in network/vendor test gates |
//!
//! The default build needs no network and no external service: every feature above is
//! additive and off by default except `native`.
//!
//! # Getting started
//!
//! ```
//! use openjiuwen::prelude::*;
//!
//! let card = AgentCard::new("my-agent");
//! assert_eq!(card.name, "my-agent");
//!
//! let message = BaseMessage::user("hello");
//! assert_eq!(message.role, openjiuwen::core::foundation::Role::User);
//!
//! let session = AgentSession::new("s-1");
//! assert_eq!(session.id(), "s-1");
//! ```
#![forbid(unsafe_code)]
// The aliases below keep the Python package spelling, which repeats the crate
// suffix (`harness_protocol`, `agent_teams`, `sys_operation`, …).
#![allow(clippy::module_name_repetitions)]

// --- module aliases (alphabetical) ------------------------------------------
pub use openjiuwen_agent_evolving as agent_evolving;
pub use openjiuwen_agent_teams as agent_teams;
pub use openjiuwen_auto_harness as auto_harness;
pub use openjiuwen_core as core;
pub use openjiuwen_dev_tools as dev_tools;
pub use openjiuwen_extensions as extensions;
pub use openjiuwen_harness as harness;
pub use openjiuwen_harness_protocol as harness_protocol;
pub use openjiuwen_harness_providers as harness_providers;
pub use openjiuwen_rsi as rsi;
pub use openjiuwen_symphony as symphony;

pub mod prelude;

/// Returns the crate name.
#[must_use]
pub const fn crate_name() -> &'static str {
    "openjiuwen"
}
