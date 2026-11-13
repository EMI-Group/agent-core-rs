//! The runner's callback framework, filters, chains, and handler combinators.
//!
//! [`CallbackEvent`] is the typed wire vocabulary (`"_framework:<event>"`),
//! [`CallbackFramework`] the priority-ordered dispatcher with explicit error
//! isolation, and [`bind_guardrails`] the adapter that runs an
//! [`openjiuwen_security::GuardrailPipeline`] at the model/tool invoke events.
//!
//! Three layers sit on top of that dispatch core:
//!
//! * [`EventFilter`] gates a handler before it runs
//!   ([`NameFilter`], [`KindFilter`], [`SessionFilter`], [`RateLimitFilter`],
//!   [`CircuitBreakerFilter`], [`ValidationFilter`], [`LoggingFilter`],
//!   [`AuthFilter`], [`ParamModifyFilter`], [`ConditionalFilter`], composed with
//!   [`FilterChain`]/[`OrFilter`]/[`NotFilter`]).
//! * [`CallbackChain`] runs an ordered group of handlers with per-entry filters,
//!   priorities, enable flags, and fatal-vs-isolated error policy.
//! * [`on`]/[`emit_before`]/[`emit_after`]/[`emit_around`]/[`transform_io`]/
//!   [`wrap`] build handlers and call wrappers, the Rust equivalents of Python's
//!   callback decorators.
//!
//! ```no_run
//! use openjiuwen_runner::callback::{
//!     on, CallbackChain, CallbackContext, CallbackEvent, CallbackFramework, CallbackVerdict,
//!     FilterChain, NameFilter,
//! };
//!
//! # async fn example() -> openjiuwen_common::error::Result<()> {
//! let framework = CallbackFramework::new();
//! on(CallbackEvent::AgentInvokeOutput)
//!     .named("recorder")
//!     .with_priority(10)
//!     .when(FilterChain::new().with(NameFilter::new().with_prefix("record")))
//!     .sync(|ctx: &CallbackContext| {
//!         let _ = ctx.session_id();
//!         Ok(CallbackVerdict::Continue)
//!     })
//!     .register(&framework);
//!
//! let chain = CallbackChain::new("audit");
//! on(CallbackEvent::AgentInvokeOutput)
//!     .named("audit")
//!     .sync(|_: &CallbackContext| Ok(CallbackVerdict::Continue))
//!     .add_to(&chain);
//!
//! let report = framework
//!     .trigger(CallbackEvent::AgentInvokeOutput, &CallbackContext::new(CallbackEvent::AgentInvokeOutput))
//!     .await?;
//! assert_eq!(report.fired, 1);
//! chain
//!     .trigger(CallbackEvent::AgentInvokeOutput, &CallbackContext::new(CallbackEvent::AgentInvokeOutput))
//!     .await?;
//! # Ok(())
//! # }
//! ```
//!
//! Dispatch contract: handlers run by descending priority with registration
//! order as the tiebreak; a handler `Err` is recorded in
//! [`CallbackReport::failures`] and dispatch continues;
//! [`CallbackVerdict::Abort`] stops dispatch and surfaces as the `trigger`
//! error; [`CallbackVerdict::Stop`] stops dispatch successfully.

pub mod chain;
pub mod context;
pub mod decorator;
pub mod event;
pub mod filters;
pub mod framework;
pub mod handler;
pub mod security;

pub use chain::{
    CallbackChain, ChainEntry, ChainEntryId, ChainEntryInfo, ChainReport, EntryOutcome, EntryRecord,
};
pub use context::{CallbackContext, CallbackFailure, CallbackReport, CallbackVerdict};
pub use decorator::{
    emit_after, emit_around, emit_before, input_as, on, on_wrap, output_as, payload_call,
    require_input, require_output, set_input, set_output, transform_io, wrap, wrap_handler,
    BuiltHandler, CallReport, ClosureWrapHandler, EmitHandler, HandlerBuilder, NextCall,
    PayloadCall, PayloadTransform, TransformHandler, WrapHandler, WrapId, WrapRegistration,
    WrapRegistry, WrappedCall,
};
pub use event::{CallbackEvent, EventDomain, DEFAULT_SCOPE, SCOPE_PREFIX};
pub use filters::{
    and, lenient, not, or, record_failure, record_success, run_filters, AuthFilter,
    CircuitBreakerFilter, Clock, ClosureFilter, ConditionalFilter, EventFilter, EventFilterExt,
    FilterChain, FilterDecision, FilterFeedback, FilterInput, FilterRunOutcome, KindFilter,
    LenientFilter, LoggingFilter, ManualClock, NameFilter, NotFilter, OrFilter, ParamModifyFilter,
    PredicateFilter, RateLimitFilter, SessionFilter, ValidationFilter,
};
pub use framework::{CallbackFramework, CallbackId, CallbackInfo, CallbackRegistration};
pub use handler::{AsyncHandler, CallbackHandler, CallbackResult, SyncHandler};
pub use security::{
    bind_guardrails, event_for_stage, scan_stage_for, GuardrailHandler, GUARDRAIL_NAMESPACE,
    GUARDRAIL_PRIORITY, GUARDRAIL_STAGES, GUARDRAIL_TAG,
};

/// `tracing` target every callback log uses.
pub(crate) const LOG_TARGET: &str = "openjiuwen_runner::callback";
