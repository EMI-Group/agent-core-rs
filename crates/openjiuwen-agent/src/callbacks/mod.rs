//! Agent callback framework: lifecycle events, typed payloads, ordered handlers.
//!
//! Ports `openjiuwen/core/single_agent/agent_callback_manager.py`: a per-agent
//! registry of callbacks that fire on invocation start, `ReAct` step start and
//! end, tool-call start and end, and invocation end.
//!
//! Unlike Python, which registers every agent's callbacks in the process-global
//! `Runner.callback_framework`, [`CallbackFramework`] is an ordinary value the
//! owner of an agent holds: there is no global mutable state, and an empty
//! registry costs nothing on the invoke path.
//!
//! - [`event`] — [`AgentCallbackEvent`], the six lifecycle events.
//! - [`payload`] — the typed payload of each event and the [`CallbackPayload`]
//!   union that carries them.
//! - [`framework`] — the [`AgentCallback`] trait and the [`CallbackFramework`]
//!   registry that dispatches by event.
//!
//! ```
//! use openjiuwen_agent::callbacks::{AgentCallbackEvent, CallbackFramework};
//!
//! let framework = CallbackFramework::new();
//! assert!(!framework.has_hooks(AgentCallbackEvent::InvokeStart));
//! ```
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]
pub mod event;
pub mod framework;
pub mod payload;
pub use event::AgentCallbackEvent;
pub use framework::{AgentCallback, CallbackFramework, DEFAULT_CALLBACK_PRIORITY};
pub use payload::{
    CallbackPayload, InvokeEndPayload, InvokeStartPayload, StepEndPayload, StepStartPayload,
    ToolCallEndPayload, ToolCallPayload,
};
