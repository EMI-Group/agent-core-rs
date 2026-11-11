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
//!
//! The types most consumers need are re-exported at the crate root:
//! [`SessionConfig`], the default ids and stream constants ([`DEFAULT_NODE_ID`],
//! [`DEFAULT_STREAM_CAPACITY`], [`DEFAULT_WORKFLOW_ID`], [`END_FRAME`]), the
//! state stores and lane keys ([`WorkflowState`], [`AgentState`], [`StateStore`],
//! [`StateLane`], and the `*_KEY` constants), the interaction vocabulary
//! ([`InteractionBroker`], [`InteractionRequest`], [`InteractionResponse`],
//! [`InteractionKind`], [`InteractionDecision`], [`InteractionOption`],
//! [`InteractionOutput`], [`InteractiveInput`]), and the stream frames
//! ([`StreamItem`], [`StreamEvent`], [`StreamMode`], [`StreamHub`],
//! [`StreamWriterManager`]).
#![forbid(unsafe_code)]

pub mod config;
pub mod constants;
pub mod interaction;
pub mod state;
pub mod stream;

pub use config::SessionConfig;
pub use constants::{DEFAULT_NODE_ID, DEFAULT_STREAM_CAPACITY, DEFAULT_WORKFLOW_ID, END_FRAME};
pub use interaction::{
    InteractionBroker, InteractionDecision, InteractionKind, InteractionOption, InteractionOutput,
    InteractionRequest, InteractionResponse, InteractiveInput,
};
pub use state::{AgentState, StateLane, StateStore, WorkflowState};
pub use state::{
    AGENT_STATE_KEY, COMP_STATE_KEY, COMP_STATE_UPDATES_KEY, GLOBAL_STATE_KEY,
    GLOBAL_STATE_UPDATES_KEY, IO_STATE_KEY, IO_STATE_UPDATES_KEY, TRACE_STATE_KEY,
    WORKFLOW_STATE_KEY, WORKFLOW_STATE_UPDATES_KEY,
};
pub use stream::{StreamEvent, StreamHub, StreamItem, StreamMode, StreamWriterManager};
