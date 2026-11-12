//! DeepAgent harness of the openJiuwen SDK — a port of the Python `openjiuwen.harness` package.
//!
//! This crate composes the `openjiuwen_core` kernel (agent runtime, sessions, sysop, security,
//! memory, context engine) into the DeepAgent task-loop framework with its workspace, permission
//! engine, coding tools, rails, prompts, skills and subagents.
//! It also implements the native provider for the provider-neutral SPI defined by
//! `openjiuwen_harness_protocol`.
//! The module tree and per-module responsibilities are documented in `CONTEXT.md`.

#![forbid(unsafe_code)]

pub mod agent;
pub mod cli;
pub mod manifest;
pub mod observability;
pub mod permission;
pub mod prompts;
pub mod protocol;
pub mod rails;
pub mod schema;
pub mod skills;
pub mod subagents;
pub mod task_loop;
pub mod tools;
pub mod workspace;

/// Returns the crate name.
pub fn crate_name() -> &'static str {
    "openjiuwen-harness"
}
