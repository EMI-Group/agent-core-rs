//! `Runner`, `ResourceMgr`, callbacks, and the message-queue abstraction — the
//! process-level orchestrator of the openJiuwen kernel.
//!
//! Maps to the Python package `openjiuwen.core.runner` (v0.1.20). The crate owns:
//!
//! - [`Runner`] — configuration, lifecycle, and the six execution entry points
//!   (`run_agent`, `run_workflow`, `run_agent_team` plus streaming variants);
//! - [`ResourceMgr`] — the typed registries for tools, agents, workflows, and
//!   teams, with providers invoked per lookup;
//! - [`CallbackFramework`] — priority-ordered lifecycle events with explicit
//!   error isolation, and the guardrail binding ([`bind_guardrails`]);
//! - [`MessageQueue`] — an in-process, per-topic FIFO queue ([`InMemoryMessageQueue`])
//!   and the trait seam a broker backend (`openjiuwen-extensions`) implements;
//! - the subprocess spawn seam ([`AgentSpawner`], [`SpawnSpec`], [`SpawnMessage`]),
//!   which is a documented contract without a process implementation.
//!
//! ```no_run
//! use openjiuwen_runner::run::RunOptions;
//! use openjiuwen_runner::runner::Runner;
//! use serde_json::json;
//!
//! # async fn example() -> openjiuwen_common::error::Result<()> {
//! let runner = Runner::new();
//! runner.start().await?;
//! let output = runner
//!     .run_agent("assistant", json!({"query": "hi"}), RunOptions::default())
//!     .await?;
//! println!("{}", output.output);
//! runner.stop().await?;
//! # Ok(())
//! # }
//! ```
//!
//! Design boundaries and the ported behaviours are documented in `CONTEXT.md`
//! and in each module directory's `CONTEXT.md`.

pub mod callback;
pub mod config;
pub mod error;
pub mod mq;
pub mod resources;
pub mod run;
pub mod runner;
pub mod spawn;

// The flat crate-root re-exports (`openjiuwen_runner::Runner`) are added by the
// integration pass, once every module defines its public surface.
