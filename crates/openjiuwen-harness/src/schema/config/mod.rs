//! Configuration of the `DeepAgent` harness, ported from
//! `openjiuwen/harness/schema/config.py`.
//!
//! The types here are serde data only: Python's `DeepAgentConfig` mixes live
//! handles (`Model`, rails, `SysOperation`, `permission_host`) with data, and the
//! Rust port keeps the data while the live handles are supplied through
//! `DeepAgentBuilder`.
//!
//! - [`DeepAgentConfig`] — the full agent configuration plus
//!   [`DeepAgentConfig::validate`] and [`resolve_inner_react_max_iterations`].
//! - [`SubAgentConfig`] — one sub-agent specification.
//! - [`TaskLoopNoProgressGuardConfig`] — the repeated-empty-answer guard.
//! - [`OnMaxIterations`] — re-exported exhaustion policy.
//!
//! The public paths are stable (`crate::schema::<item>`), independent of
//! the file split.

mod agent;
mod subagent;

pub use agent::{resolve_inner_react_max_iterations, DeepAgentConfig};
pub use subagent::{OnMaxIterations, SubAgentConfig, TaskLoopNoProgressGuardConfig};

#[cfg(test)]
mod tests;
