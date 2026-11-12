//! Permission model, engine, file/shell guards, session decisions, and HITL interrupt rail.
//!
//! A DeepAgent tool call passes one tiered policy pipeline plus two targeted guards
//! before it runs:
//!
//! 1. [`tiered`] — builtin command rules, user rules, approval overrides, tool
//!    baselines and `defaults["*"]`, always yielding a decision.
//! 2. [`file_guard`] — path read/write/exec policy for file and shell tools.
//! 3. [`net_guard`] — URL allow/deny policy for fetch tools.
//!
//! [`engine::PermissionEngine`] folds those three with [`engine::strictest`]
//! (`deny < ask < allow`) and renders the Python-compatible `matched_rule` and
//! `reason` strings. [`interrupt::PermissionInterruptRail`] turns an `ask` into a
//! human-in-the-loop confirmation on the kernel's [`openjiuwen_agent::AgentRail`]
//! seam, with session-scoped and permanent "remember" persistence routed through
//! [`session::PermissionHost`].
//!
//! The module contract, deviations from the Python reference and the test strategy
//! are documented in `CONTEXT.md`.

pub mod builtin;
pub mod engine;
pub mod file_guard;
pub mod interrupt;
pub mod model;
pub mod net_guard;
pub mod paths;
pub mod patterns;
pub mod presentation;
pub mod rules;
pub mod session;
pub mod shell;
pub mod tiered;

#[cfg(test)]
mod tests;
