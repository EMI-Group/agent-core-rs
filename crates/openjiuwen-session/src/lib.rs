//! Sessions, state scopes, streaming, and human-in-the-loop interactions.
//!
//! Maps to the Python package `openjiuwen.core.session` in the openJiuwen
//! reference implementation. The crate is the kernel's stable seam for
//! per-run execution context: state lanes, stream frames, and interaction
//! requests/responses. Checkpointing, tracing, and the concrete session types
//! live in sibling modules introduced by later phases.
//!
//! Modules:
//!
//! - [`config`] — [`config::SessionConfig`], built-in defaults and environment overrides.
//! - [`constants`] — environment key names and default ids/frame constants.
//! - [`state`] — state stores ([`state::WorkflowState`], [`state::AgentState`])
//!   and nested JSON path helpers.
//! - [`stream`] — stream frames ([`stream::StreamItem`]) and delivery
//!   ([`stream::StreamHub`], [`stream::StreamWriterManager`]).
//! - [`interaction`] — human-in-the-loop requests, responses, and the
//!   [`interaction::InteractionBroker`].
#![forbid(unsafe_code)]

pub mod config;
pub mod constants;
pub mod interaction;
pub mod state;
pub mod stream;
