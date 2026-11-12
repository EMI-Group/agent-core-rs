//! The runner's callback framework and guardrail binding.
//!
//! [`CallbackEvent`] is the typed wire vocabulary (`"_framework:<event>"`),
//! [`CallbackFramework`] the priority-ordered dispatcher with explicit error
//! isolation, and [`bind_guardrails`] the adapter that runs an
//! [`openjiuwen_security::GuardrailPipeline`] at the model/tool invoke events.
//!
//! ```no_run
//! use openjiuwen_runner::callback::{CallbackContext, CallbackEvent, CallbackFramework, CallbackVerdict};
//!
//! # async fn example() -> openjiuwen_common::error::Result<()> {
//! let framework = CallbackFramework::new();
//! framework.register_sync(CallbackEvent::AgentInvokeOutput, 10, |ctx: &CallbackContext| {
//!     let _ = ctx.session_id();
//!     Ok(CallbackVerdict::Continue)
//! });
//!
//! let report = framework
//!     .trigger(CallbackEvent::AgentInvokeOutput, &CallbackContext::new(CallbackEvent::AgentInvokeOutput))
//!     .await?;
//! assert_eq!(report.fired, 1);
//! # Ok(())
//! # }
//! ```
//!
//! Dispatch contract: handlers run by descending priority with registration
//! order as the tiebreak; a handler `Err` is recorded in
//! [`CallbackReport::failures`] and dispatch continues;
//! [`CallbackVerdict::Abort`] stops dispatch and surfaces as the `trigger`
//! error; [`CallbackVerdict::Stop`] stops dispatch successfully.

pub mod context;
pub mod event;
pub mod framework;
pub mod handler;
pub mod security;

pub use context::{CallbackContext, CallbackFailure, CallbackReport, CallbackVerdict};
pub use event::{CallbackEvent, EventDomain, DEFAULT_SCOPE, SCOPE_PREFIX};
pub use framework::{CallbackFramework, CallbackId, CallbackInfo, CallbackRegistration};
pub use handler::{AsyncHandler, CallbackHandler, CallbackResult, SyncHandler};
pub use security::{
    bind_guardrails, event_for_stage, scan_stage_for, GuardrailHandler, GUARDRAIL_NAMESPACE,
    GUARDRAIL_PRIORITY, GUARDRAIL_STAGES, GUARDRAIL_TAG,
};

/// `tracing` target every callback log uses.
pub(crate) const LOG_TARGET: &str = "openjiuwen_runner::callback";
