//! Controller stack: routing events to agents and workflows.
//!
//! Ports `openjiuwen/core/controller/`: an event queue feeds intent recognition,
//! the recognizer produces an intent, the task scheduler picks a registered
//! task executor (a controller agent, an LLM router, or a workflow), and the
//! controller returns the executor's output.
//!
//! # Modules
//!
//! The port is landing in phases. [`config`] holds the behaviour knobs,
//! [`schema`] the data model, [`executors`] the task-executor seam, [`intent`]
//! recognition and planning, and [`base`] the controller loop itself. A later
//! phase adds `legacy` (the string-routed controller that predates the event
//! queue).
pub mod base;
pub mod config;
pub mod executors;
pub mod intent;
pub mod schema;
