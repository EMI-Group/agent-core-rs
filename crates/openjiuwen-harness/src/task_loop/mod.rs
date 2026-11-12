//! Loop coordinator, queues, event manager, and the task-loop controller and handler.
//!
//! The outer task loop wraps single `ReAct` turns: it owns the round budget and
//! stop-condition chain ([`LoopCoordinator`]), the follow-up and steering queues
//! ([`LoopQueues`]), the optional priority event queue ([`EventManager`]), the
//! hook seam ([`TaskLoopEventHandler`]) and the driver that sequences them
//! ([`TaskLoopController`]).
//!
//! The controller never owns the agent: one round of work is executed through the
//! object-safe [`RoundExecutor`] seam supplied by `crate::agent`, so this module
//! stays pure loop machinery. Python parity: `openjiuwen/harness/task_loop/*` and
//! the `_run_task_loop` driver in `deep_agent.py`.
//!
//! Every public item is re-exported flat from this module; the submodules are
//! private. See `CONTEXT.md` for the full contract.

mod controller;
mod coordinator;
mod events;
mod handler;
mod queues;

pub use controller::{
    LoopReport, RoundExecutor, RoundId, StateSink, TaskLoopConfig, TaskLoopController,
};
pub use coordinator::{default_evaluators, LoopCoordinator, DEFAULT_MAX_OUTER_ROUNDS};
pub use events::{EventManager, DEFAULT_EVENT_CAPACITY};
pub use handler::{DefaultTaskLoopEventHandler, TaskLoopEventHandler};
pub use queues::{LoopQueues, QueuedInput};
