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
//! Every public type is re-exported flat at the crate root; the module paths
//! (`openjiuwen_runner::callback::CallbackEvent`, `openjiuwen_runner::run::RunOptions`,
//! ...) remain available for hosts that prefer them.
//!
//! ```no_run
//! use openjiuwen_runner::{RunOptions, Runner};
//! use serde_json::json;
//!
//! # async fn example() -> openjiuwen_common::error::Result<()> {
//! let runner = Runner::new();
//! runner.start().await?;
//! let output = runner
//!     .run_agent("assistant", json!({"query": "hi"}), RunOptions::default())
//!     .await?;
//! assert!(!output.output.is_empty());
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
pub mod hitl;
pub mod monitor;
pub mod mq;
pub mod pool;
pub mod resources;
pub mod run;
pub mod runner;
pub mod spawn;

// ------------------------------------------------------- flat re-exports --

pub use callback::{
    bind_guardrails, event_for_stage, scan_stage_for, AsyncHandler, CallbackContext, CallbackEvent,
    CallbackFailure, CallbackFramework, CallbackHandler, CallbackId, CallbackInfo,
    CallbackRegistration, CallbackReport, CallbackResult, CallbackVerdict, EventDomain,
    GuardrailHandler, SyncHandler,
};
pub use config::{
    DistributedConfig, MessageQueueConfig, MessageQueueKind, RunnerConfig, CONVERSATION_ID_KEY,
    DEFAULT_REQUEST_TIMEOUT_SECS, DEFAULT_RUNNER_ID, DEFAULT_SESSION_ID, DEFAULT_TOPIC_CAPACITY,
};
pub use mq::{
    create_message_queue, InMemoryMessageQueue, MessageHandler, MessageQueue, QueueMessage, SUCCESS,
};
pub use resources::{Provider, RegisterPolicy, Registration, Registry, ResourceCard, ResourceMgr};
pub use run::{
    invoke_agent, invoke_team, invoke_workflow, stream_agent, stream_team, stream_workflow,
    AgentRef, RunOptions, RunServices, TeamRef, WorkflowRef,
};
pub use runner::Runner;
pub use spawn::{
    AgentSpawner, SpawnAgentKind, SpawnMessage, SpawnMessageKind, SpawnSpec, SpawnedAgent,
    UnsupportedSpawner,
};
