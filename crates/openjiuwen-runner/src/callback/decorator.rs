//! Handler combinators: the Rust equivalent of Python's callback decorators.
//!
//! Rust has no function decorators, so `openjiuwen/core/runner/callback/decorator.py`
//! is ported as constructors that build [`CallbackHandler`]s (for
//! [`CallbackFramework`] registration and [`CallbackChain`] entries) plus the
//! call-wrapping layer that decorators really provided:
//!
//! * [`on`] returns a [`HandlerBuilder`] — Python's `framework.on(...)`: name,
//!   event, priority, namespace, tags, `once`, filters ([`HandlerBuilder::when`]),
//!   and the fatal-vs-isolated error policy, ending in a [`BuiltHandler`] that
//!   knows how to register itself.
//! * [`emit_before`]/[`emit_after`]/[`emit_around`] produce handlers that fire an
//!   auxiliary event with the source payload, turning an auxiliary abort into
//!   [`CallbackVerdict::Abort`] (Python's propagated `AbortError`) and an
//!   auxiliary stop into [`CallbackVerdict::Stop`].
//! * [`transform_io`] rewrites the input payload before downstream work and the
//!   output payload afterwards, propagating transform errors. It works both as a
//!   [`CallbackHandler`] and as an [`EventFilter`] (the in-place form whose
//!   rewrites later entries observe).
//! * [`wrap`] builds a [`WrappedCall`]: an async call surrounded by
//!   [`WrapHandler`] layers, each of which receives `call_next`, may rewrite
//!   arguments and results, and may short-circuit by never calling it
//!   (Python's `create_wrap_decorator`). [`on_wrap`] registers a layer on a
//!   [`WrapRegistry`] so [`WrappedCall::from_registry`] resolves layers per call
//!   (Python's `create_on_wrap_decorator` / `create_wrap_by_event_decorator`).
//!
//! Payloads stay `serde_json::Value` on [`CallbackContext`]; the
//! [`input_as`]/[`output_as`]/[`set_input`]/[`set_output`] helpers give typed
//! access, and unknown events stay expressible through [`CallbackEvent::Custom`]
//! (their payload rides the same `input`/`output` fields).
#![allow(clippy::module_name_repetitions)]
// A combinator computes its name at construction time, so it cannot return `&'static str`.
#![allow(clippy::unnecessary_literal_bound)]
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::unnecessary_wraps
    )
)]

use crate::callback::chain::{CallbackChain, ChainEntry, ChainEntryId};
use crate::callback::context::{CallbackContext, CallbackVerdict};
use crate::callback::event::CallbackEvent;
use crate::callback::filters::{
    record_failure, record_success, run_filters, EventFilter, FilterRunOutcome,
};
use crate::callback::framework::{CallbackFramework, CallbackId, CallbackRegistration};
use crate::callback::handler::{AsyncHandler, CallbackHandler, CallbackResult, SyncHandler};
use crate::callback::LOG_TARGET;
use async_trait::async_trait;
use futures::future::BoxFuture;
use openjiuwen_common::error::{Error, Result};
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::collections::BTreeSet;
use std::fmt;
use std::sync::Arc;
pub mod emit;
pub mod transform;
pub mod wrap;
pub use emit::{emit_after, emit_around, emit_before, EmitHandler};
pub use transform::{transform_io, PayloadTransform, TransformHandler};
pub use wrap::{
    on_wrap, payload_call, wrap, wrap_handler, CallReport, ClosureWrapHandler, NextCall,
    PayloadCall, WrapHandler, WrapId, WrapRegistration, WrapRegistry, WrappedCall,
};

// ---------------------------------------------------------------------------
// Typed payload access
// ---------------------------------------------------------------------------

/// `ctx.input` deserialized as `T`, or `None` when the event carries no input.
///
/// # Errors
///
/// Returns `Error::Validation` when the payload does not deserialize as `T`.
pub fn input_as<T: DeserializeOwned>(ctx: &CallbackContext) -> Result<Option<T>> {
    ctx.input
        .as_ref()
        .map(|value| serde_json::from_value(value.clone()).map_err(Error::from))
        .transpose()
}

/// `ctx.output` deserialized as `T`, or `None` when the event carries no output.
///
/// # Errors
///
/// Returns `Error::Validation` when the payload does not deserialize as `T`.
pub fn output_as<T: DeserializeOwned>(ctx: &CallbackContext) -> Result<Option<T>> {
    ctx.output
        .as_ref()
        .map(|value| serde_json::from_value(value.clone()).map_err(Error::from))
        .transpose()
}

/// `ctx.input` deserialized as `T`, failing when the event carries no input.
///
/// # Errors
///
/// Returns `Error::Validation` when the payload is absent or does not
/// deserialize as `T`.
pub fn require_input<T: DeserializeOwned>(ctx: &CallbackContext) -> Result<T> {
    input_as(ctx)?.ok_or_else(|| Error::validation("callback context carries no input payload"))
}

/// `ctx.output` deserialized as `T`, failing when the event carries no output.
///
/// # Errors
///
/// Returns `Error::Validation` when the payload is absent or does not
/// deserialize as `T`.
pub fn require_output<T: DeserializeOwned>(ctx: &CallbackContext) -> Result<T> {
    output_as(ctx)?.ok_or_else(|| Error::validation("callback context carries no output payload"))
}

/// Serializes `value` into `ctx.input`.
///
/// # Errors
///
/// Returns `Error::Validation` when `value` does not serialize.
pub fn set_input<T: Serialize>(ctx: &mut CallbackContext, value: &T) -> Result<()> {
    ctx.input = Some(serde_json::to_value(value)?);
    Ok(())
}

/// Serializes `value` into `ctx.output`.
///
/// # Errors
///
/// Returns `Error::Validation` when `value` does not serialize.
pub fn set_output<T: Serialize>(ctx: &mut CallbackContext, value: &T) -> Result<()> {
    ctx.output = Some(serde_json::to_value(value)?);
    Ok(())
}
// ---------------------------------------------------------------------------
// on(...) builder
// ---------------------------------------------------------------------------

/// The handler a [`HandlerBuilder`] produced.
///
/// Registration-ready: [`Self::register`] subscribes it to a
/// [`CallbackFramework`], [`Self::chain_entry`] turns it into a
/// [`ChainEntry`], and [`Self::handler`] hands out the name-normalizing,
/// filter-applying handler itself.
pub struct BuiltHandler {
    event: CallbackEvent,
    name: String,
    priority: i32,
    namespace: String,
    tags: BTreeSet<String>,
    once: bool,
    fatal: bool,
    filters: Vec<Arc<dyn EventFilter>>,
    inner: Arc<dyn CallbackHandler>,
    wrapped: Arc<dyn CallbackHandler>,
}

impl BuiltHandler {
    /// The handler name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The event the builder targets.
    #[must_use]
    pub fn event(&self) -> &CallbackEvent {
        &self.event
    }

    /// The dispatch priority.
    #[must_use]
    pub const fn priority(&self) -> i32 {
        self.priority
    }

    /// The namespace.
    #[must_use]
    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    /// Whether a handler error aborts the dispatch.
    #[must_use]
    pub const fn is_fatal(&self) -> bool {
        self.fatal
    }

    /// The filters, in evaluation order.
    #[must_use]
    pub fn filters(&self) -> &[Arc<dyn EventFilter>] {
        &self.filters
    }

    /// The wrapper: normalizes the name, applies the filters, and enforces the
    /// error policy. Use it when a single [`CallbackHandler`] is wanted.
    #[must_use]
    pub fn handler(&self) -> Arc<dyn CallbackHandler> {
        Arc::clone(&self.wrapped)
    }

    /// The raw handler, without the name/filter/policy wrapper.
    #[must_use]
    pub fn inner_handler(&self) -> Arc<dyn CallbackHandler> {
        Arc::clone(&self.inner)
    }

    /// The framework registration this builder describes.
    #[must_use]
    pub fn registration(&self) -> CallbackRegistration {
        let mut registration = CallbackRegistration::new(self.event.clone(), self.handler())
            .with_priority(self.priority)
            .with_namespace(self.namespace.clone());
        for tag in &self.tags {
            registration = registration.with_tag(tag.clone());
        }
        if self.once {
            registration = registration.once();
        }
        registration
    }

    /// Registers the handler on `framework`; returns its removal handle.
    pub fn register(&self, framework: &CallbackFramework) -> CallbackId {
        framework.register(self.registration())
    }

    /// A chain entry carrying the builder's name, priority, namespace, filters,
    /// and error policy.
    #[must_use]
    pub fn chain_entry(&self) -> ChainEntry {
        let mut entry = ChainEntry::new(self.name.clone(), Arc::clone(&self.inner))
            .with_priority(self.priority)
            .in_namespace(self.namespace.clone());
        for filter in &self.filters {
            entry = entry.with_filter(Arc::clone(filter));
        }
        if self.fatal {
            entry = entry.fatal();
        }
        entry
    }

    /// Appends the builder's [`ChainEntry`] to `chain`.
    pub fn add_to(&self, chain: &CallbackChain) -> ChainEntryId {
        chain.add(self.chain_entry())
    }
}

impl fmt::Debug for BuiltHandler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BuiltHandler")
            .field("event", &self.event)
            .field("name", &self.name)
            .field("priority", &self.priority)
            .field("namespace", &self.namespace)
            .field("fatal", &self.fatal)
            .field(
                "filters",
                &self
                    .filters
                    .iter()
                    .map(EventFilter::name)
                    .collect::<Vec<_>>(),
            )
            .finish_non_exhaustive()
    }
}

/// The handler that carries a builder's name, filters, and error policy.
struct ComposedHandler {
    name: String,
    namespace: String,
    filters: Vec<Arc<dyn EventFilter>>,
    fatal: bool,
    inner: Arc<dyn CallbackHandler>,
}

#[async_trait]
impl CallbackHandler for ComposedHandler {
    fn name(&self) -> &str {
        &self.name
    }

    async fn handle(&self, ctx: &CallbackContext) -> CallbackResult {
        let mut working = ctx.clone();
        match run_filters(
            &self.filters,
            self.inner.name(),
            &self.namespace,
            &mut working,
        ) {
            Ok(FilterRunOutcome::Allow) => {}
            Ok(FilterRunOutcome::Skip { filter, reason }) => {
                tracing::debug!(
                    target: LOG_TARGET,
                    handler = %self.name,
                    filter = %filter,
                    reason = %reason,
                    "handler skipped by a filter"
                );
                return Ok(CallbackVerdict::Continue);
            }
            // A rejected filter and a filter that cannot decide both abort: a
            // filter must not silently allow the handler it guards.
            Ok(FilterRunOutcome::Reject(error)) | Err(error) => {
                return Ok(CallbackVerdict::Abort(error));
            }
        }
        match self.inner.handle(&working).await {
            Ok(verdict) => {
                record_success(&self.filters, &working, self.inner.name());
                Ok(verdict)
            }
            Err(error) => {
                record_failure(&self.filters, &working, self.inner.name());
                if self.fatal {
                    Ok(CallbackVerdict::Abort(error))
                } else {
                    Err(error)
                }
            }
        }
    }
}

/// Builds one handler for one event (Python's `framework.on(event)`).
#[must_use]
pub fn on(event: CallbackEvent) -> HandlerBuilder {
    HandlerBuilder::new(event)
}

/// Configures a handler before it is built.
pub struct HandlerBuilder {
    event: CallbackEvent,
    name: Option<String>,
    priority: i32,
    namespace: String,
    tags: BTreeSet<String>,
    once: bool,
    fatal: bool,
    filters: Vec<Arc<dyn EventFilter>>,
}

impl HandlerBuilder {
    /// A builder for `event` at priority `0` in namespace `"default"`.
    #[must_use]
    pub fn new(event: CallbackEvent) -> Self {
        Self {
            event,
            name: None,
            priority: 0,
            namespace: "default".to_owned(),
            tags: BTreeSet::new(),
            once: false,
            fatal: false,
            filters: Vec::new(),
        }
    }

    /// Overrides the handler name (defaults to the closure's adapter name).
    #[must_use]
    pub fn named(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Sets the dispatch priority; higher runs first.
    #[must_use]
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Sets the namespace.
    #[must_use]
    pub fn in_namespace(mut self, namespace: impl Into<String>) -> Self {
        self.namespace = namespace.into();
        self
    }

    /// Adds one tag (used by `CallbackFramework::unregister_tag`).
    #[must_use]
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.insert(tag.into());
        self
    }

    /// Removes the handler after its first successful run.
    #[must_use]
    pub fn once(mut self) -> Self {
        self.once = true;
        self
    }

    /// A handler error becomes an abort instead of an isolated failure.
    #[must_use]
    pub fn fatal(mut self) -> Self {
        self.fatal = true;
        self
    }

    /// Adds a filter that runs before the handler.
    #[must_use]
    pub fn when(mut self, filter: impl EventFilter + 'static) -> Self {
        self.filters.push(Arc::new(filter));
        self
    }

    /// Adds an already shared filter.
    #[must_use]
    pub fn when_arc(mut self, filter: Arc<dyn EventFilter>) -> Self {
        self.filters.push(filter);
        self
    }

    /// Builds from a synchronous closure.
    #[must_use]
    pub fn sync<F>(self, handler: F) -> BuiltHandler
    where
        F: Fn(&CallbackContext) -> CallbackResult + Send + Sync + 'static,
    {
        let default_name = "sync_handler".to_owned();
        let inner: Arc<dyn CallbackHandler> = Arc::new(SyncHandler::new(handler));
        self.finish(inner, default_name)
    }

    /// Builds from an async closure returning a boxed future.
    #[must_use]
    pub fn async_fn<F>(self, handler: F) -> BuiltHandler
    where
        F: Fn(CallbackContext) -> BoxFuture<'static, CallbackResult> + Send + Sync + 'static,
    {
        let default_name = "async_handler".to_owned();
        let inner: Arc<dyn CallbackHandler> = Arc::new(AsyncHandler::new(handler));
        self.finish(inner, default_name)
    }

    /// Builds from a handler value (including [`EmitHandler`], [`TransformHandler`],
    /// or an `Arc<CallbackChain>`).
    #[must_use]
    pub fn handler<H>(self, handler: H) -> BuiltHandler
    where
        H: CallbackHandler + 'static,
    {
        let default_name = handler.name().to_owned();
        let inner: Arc<dyn CallbackHandler> = Arc::new(handler);
        self.finish(inner, default_name)
    }

    fn finish(self, inner: Arc<dyn CallbackHandler>, default_name: String) -> BuiltHandler {
        let name = self.name.unwrap_or(default_name);
        let composed = ComposedHandler {
            name: name.clone(),
            namespace: self.namespace.clone(),
            filters: self.filters.clone(),
            fatal: self.fatal,
            inner: Arc::clone(&inner),
        };
        BuiltHandler {
            event: self.event,
            name,
            priority: self.priority,
            namespace: self.namespace,
            tags: self.tags,
            once: self.once,
            fatal: self.fatal,
            filters: self.filters,
            inner,
            wrapped: Arc::new(composed),
        }
    }
}

impl fmt::Debug for HandlerBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HandlerBuilder")
            .field("event", &self.event)
            .field("name", &self.name)
            .field("priority", &self.priority)
            .field("namespace", &self.namespace)
            .field("fatal", &self.fatal)
            .field("filters", &self.filters.len())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::callback::filters::{FilterDecision, FilterInput, NameFilter};
    use serde_json::{json, Value};
    #[test]
    fn typed_payload_helpers_round_trip() {
        let mut ctx = CallbackContext::new(CallbackEvent::ToolInvokeInput);
        set_input(&mut ctx, &json!({"q": "rust"})).unwrap();
        set_output(&mut ctx, &vec![1, 2, 3]).unwrap();
        assert_eq!(
            input_as::<serde_json::Map<String, Value>>(&ctx).unwrap(),
            Some(json!({"q": "rust"}).as_object().cloned().unwrap())
        );
        assert_eq!(output_as::<Vec<u8>>(&ctx).unwrap(), Some(vec![1, 2, 3]));
        assert_eq!(require_input::<Value>(&ctx).unwrap(), json!({"q": "rust"}));
        let empty = CallbackContext::new(CallbackEvent::Custom("host".to_owned()));
        assert_eq!(input_as::<Value>(&empty).unwrap(), None);
        let error = require_output::<Value>(&empty).unwrap_err();
        assert!(error.to_string().contains("no output"));
    }

    #[tokio::test]
    async fn an_empty_builder_normalizes_the_name() {
        let built = on(CallbackEvent::AgentStarted)
            .named("recorder")
            .sync(|_: &CallbackContext| Ok(CallbackVerdict::Continue));
        assert_eq!(built.name(), "recorder");
        assert_eq!(built.handler().name(), "recorder");
        assert!(!built.is_fatal());
        let ctx = CallbackContext::new(CallbackEvent::AgentStarted);
        assert!(matches!(
            built.handler().handle(&ctx).await.unwrap(),
            CallbackVerdict::Continue
        ));
        assert!(format!("{built:?}").contains("recorder"));
        assert!(format!("{:?}", on(CallbackEvent::AgentStarted)).contains("on"));
    }

    #[tokio::test]
    async fn a_builder_filter_can_skip_or_reject() {
        let skipped = on(CallbackEvent::AgentStarted)
            .when(NameFilter::new().with_name("someone_else"))
            .sync(|_: &CallbackContext| panic!("the handler must not run when the filter skips"));
        let ctx = CallbackContext::new(CallbackEvent::AgentStarted);
        assert!(matches!(
            skipped.handler().handle(&ctx).await.unwrap(),
            CallbackVerdict::Continue
        ));

        let rejecting = on(CallbackEvent::AgentStarted)
            .when(|_: FilterInput<'_>| Ok(FilterDecision::reject(Error::security("nope"))))
            .sync(|_: &CallbackContext| Ok(CallbackVerdict::Continue));
        match rejecting.handler().handle(&ctx).await.unwrap() {
            CallbackVerdict::Abort(error) => {
                assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::Security);
            }
            other => panic!("expected an abort, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn a_fatal_entry_turns_an_error_into_an_abort() {
        let isolated = on(CallbackEvent::AgentStarted)
            .sync(|_: &CallbackContext| Err(Error::validation("boom")));
        let ctx = CallbackContext::new(CallbackEvent::AgentStarted);
        assert!(isolated.handler().handle(&ctx).await.is_err());

        let fatal = on(CallbackEvent::AgentStarted)
            .fatal()
            .sync(|_: &CallbackContext| Err(Error::validation("boom")));
        assert!(matches!(
            fatal.handler().handle(&ctx).await.unwrap(),
            CallbackVerdict::Abort(_)
        ));
    }
}
