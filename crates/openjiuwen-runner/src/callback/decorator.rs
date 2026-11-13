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

use async_trait::async_trait;
use futures::future::BoxFuture;
use parking_lot::{Mutex, RwLock};
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::Value;
use std::collections::{BTreeSet, HashMap};
use std::fmt;
use std::future::Future;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};

use super::chain::{CallbackChain, ChainEntry, ChainEntryId};
use super::context::{CallbackContext, CallbackVerdict};
use super::event::CallbackEvent;
use super::filters::{record_failure, record_success, run_filters, EventFilter, FilterRunOutcome};
use super::framework::{CallbackFramework, CallbackId, CallbackRegistration};
use super::handler::{AsyncHandler, CallbackHandler, CallbackResult, SyncHandler};
use crate::callback::LOG_TARGET;

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
// The wrap layer
// ---------------------------------------------------------------------------

/// An async call over a JSON payload.
pub type PayloadCall = Arc<dyn Fn(Value) -> BoxFuture<'static, Result<Value>> + Send + Sync>;

/// Wraps an async closure as a [`PayloadCall`].
pub fn payload_call<F, Fut>(call: F) -> PayloadCall
where
    F: Fn(Value) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<Value>> + Send + 'static,
{
    Arc::new(move |input| Box::pin(call(input)))
}

/// The rest of a wrap chain, handed to a [`WrapHandler`].
///
/// Calling [`NextCall::call`] forwards to the next layer (and ultimately the
/// wrapped call); a handler that never calls it short-circuits the chain, which
/// [`NextCall::called`] makes observable.
#[derive(Clone)]
pub struct NextCall {
    next: PayloadCall,
    called: Arc<AtomicBool>,
}

impl NextCall {
    /// Whether the rest of the chain has been invoked.
    #[must_use]
    pub fn called(&self) -> bool {
        self.called.load(Ordering::Relaxed)
    }

    /// Invokes the rest of the chain with `input`.
    ///
    /// The input may differ from the handler's own input: that is how a wrap layer
    /// rewrites arguments before downstream work (Python's `call_next(*args)`).
    ///
    /// # Errors
    ///
    /// Returns whatever the next layer or the wrapped call returns.
    pub async fn call(&self, input: Value) -> Result<Value> {
        self.called.store(true, Ordering::Relaxed);
        (self.next)(input).await
    }
}

/// One layer around a [`PayloadCall`].
///
/// The layer sees the [`CallbackContext`] of the call (its `input` is the layer's
/// own input) and decides when, and whether, to invoke [`NextCall::call`].
#[async_trait]
pub trait WrapHandler: Send + Sync {
    /// The layer's name, used in logs and reports.
    fn name(&self) -> &str {
        "wrap_handler"
    }

    /// Runs the layer around `call_next`.
    ///
    /// # Errors
    ///
    /// Propagates its own failures and those of the wrapped call.
    async fn handle(&self, ctx: &CallbackContext, call_next: NextCall) -> Result<Value>;
}

/// Adapts a boxed-future closure to [`WrapHandler`].
pub struct ClosureWrapHandler<F> {
    name: String,
    handler: F,
}

#[async_trait]
impl<F> WrapHandler for ClosureWrapHandler<F>
where
    F: Fn(CallbackContext, NextCall) -> BoxFuture<'static, Result<Value>> + Send + Sync,
{
    fn name(&self) -> &str {
        &self.name
    }

    async fn handle(&self, ctx: &CallbackContext, call_next: NextCall) -> Result<Value> {
        (self.handler)(ctx.clone(), call_next).await
    }
}

/// A closure layer built with [`wrap_handler`].
#[must_use]
pub fn wrap_handler<F>(name: impl Into<String>, handler: F) -> Arc<dyn WrapHandler>
where
    F: Fn(CallbackContext, NextCall) -> BoxFuture<'static, Result<Value>> + Send + Sync + 'static,
{
    Arc::new(ClosureWrapHandler {
        name: name.into(),
        handler,
    })
}

/// The outcome of one [`WrappedCall`] invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallReport {
    /// The value the outermost layer returned.
    pub value: Value,
    /// Whether the wrapped call itself ran (false when a layer short-circuited).
    pub inner_ran: bool,
}

/// An async call surrounded by [`WrapHandler`] layers.
///
/// Layers are applied outermost first: `handlers[0]` runs first and wraps the
/// rest, matching `create_wrap_decorator`'s argument order. With
/// [`WrappedCall::from_registry`] the layers are resolved from a
/// [`WrapRegistry`] on every invocation, so registering a layer later affects
/// the next call (Python's dynamic wrap chain).
pub struct WrappedCall {
    event: CallbackEvent,
    handlers: Vec<Arc<dyn WrapHandler>>,
    inner: PayloadCall,
    registry: Option<Arc<WrapRegistry>>,
}

impl WrappedCall {
    /// Wraps `inner` with a fixed, initially empty layer list.
    #[must_use]
    pub fn new(event: CallbackEvent, inner: PayloadCall) -> Self {
        Self {
            event,
            handlers: Vec::new(),
            inner,
            registry: None,
        }
    }

    /// Wraps `inner`, resolving layers from `registry` on every call.
    #[must_use]
    pub fn from_registry(
        registry: Arc<WrapRegistry>,
        event: CallbackEvent,
        inner: PayloadCall,
    ) -> Self {
        Self {
            event,
            handlers: Vec::new(),
            inner,
            registry: Some(registry),
        }
    }

    /// Appends one layer, inside the layers added before it.
    #[must_use]
    pub fn with_handler(mut self, handler: Arc<dyn WrapHandler>) -> Self {
        self.handlers.push(handler);
        self
    }

    /// Appends one layer, inside the layers added before it.
    #[must_use]
    pub fn with_layer<W: WrapHandler + 'static>(self, handler: W) -> Self {
        self.with_handler(Arc::new(handler))
    }

    /// The event the layers and the wrapped call are associated with.
    #[must_use]
    pub fn event(&self) -> &CallbackEvent {
        &self.event
    }

    /// The fixed layers' names, outermost first.
    #[must_use]
    pub fn layers(&self) -> Vec<String> {
        self.handlers
            .iter()
            .map(|handler| handler.name().to_owned())
            .collect()
    }

    /// Runs the layers around the wrapped call.
    ///
    /// # Errors
    ///
    /// Returns the error of the layer or wrapped call that failed.
    pub async fn call(&self, input: Value) -> Result<Value> {
        self.call_with_report(input)
            .await
            .map(|report| report.value)
    }

    /// Runs the layers around the wrapped call and reports whether it ran.
    ///
    /// # Errors
    ///
    /// Returns the error of the layer or wrapped call that failed.
    pub async fn call_with_report(&self, input: Value) -> Result<CallReport> {
        let handlers = self.registry.as_ref().map_or_else(
            || self.handlers.clone(),
            |registry| registry.handlers(&self.event),
        );
        let ctx = CallbackContext::new(self.event.clone()).with_input(input.clone());
        let ran = Arc::new(AtomicBool::new(false));
        let outer = build_layers(&handlers, Arc::clone(&self.inner), &ctx, Arc::clone(&ran));
        let value = outer(input).await?;
        Ok(CallReport {
            value,
            inner_ran: ran.load(Ordering::Relaxed),
        })
    }
}

impl fmt::Debug for WrappedCall {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WrappedCall")
            .field("event", &self.event)
            .field("layers", &self.layers())
            .field("registry", &self.registry.is_some())
            .finish_non_exhaustive()
    }
}

/// Surrounds `inner` with a fixed layer list (Python's `create_wrap_decorator`).
#[must_use]
pub fn wrap(event: CallbackEvent, inner: PayloadCall) -> WrappedCall {
    WrappedCall::new(event, inner)
}

/// Folds the layers, innermost last, into one callable.
fn build_layers(
    handlers: &[Arc<dyn WrapHandler>],
    inner: PayloadCall,
    ctx: &CallbackContext,
    ran: Arc<AtomicBool>,
) -> PayloadCall {
    let mut current: PayloadCall = Arc::new(move |input| {
        let ran = Arc::clone(&ran);
        let inner = Arc::clone(&inner);
        Box::pin(async move {
            ran.store(true, Ordering::Relaxed);
            inner(input).await
        })
    });
    for handler in handlers.iter().rev() {
        let handler = Arc::clone(handler);
        let next = current;
        let ctx = ctx.clone();
        current = Arc::new(move |input| {
            let handler = Arc::clone(&handler);
            let next = Arc::clone(&next);
            let mut layer_ctx = ctx.clone();
            layer_ctx.input = Some(input.clone());
            Box::pin(async move {
                let call_next = NextCall {
                    next,
                    called: Arc::new(AtomicBool::new(false)),
                };
                handler.handle(&layer_ctx, call_next).await
            })
        });
    }
    current
}

/// The handle [`WrapRegistration::register`] returns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct WrapId(u64);

impl WrapId {
    /// The raw numeric value.
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl fmt::Display for WrapId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "wrap-{}", self.0)
    }
}

/// One wrap layer to register.
pub struct WrapRegistration {
    event: CallbackEvent,
    handler: Arc<dyn WrapHandler>,
    priority: i32,
}

impl WrapRegistration {
    /// A layer at priority `0`.
    #[must_use]
    pub fn new(event: CallbackEvent, handler: Arc<dyn WrapHandler>) -> Self {
        Self {
            event,
            handler,
            priority: 0,
        }
    }

    /// Sets the layer priority; higher runs further out.
    #[must_use]
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Registers the layer; returns its removal handle.
    pub fn register(self, registry: &WrapRegistry) -> WrapId {
        registry.register(self)
    }
}

/// Builds a [`WrapRegistration`] (Python's `on_wrap`/`create_on_wrap_decorator`).
#[must_use]
pub fn on_wrap(event: CallbackEvent, handler: Arc<dyn WrapHandler>) -> WrapRegistration {
    WrapRegistration::new(event, handler)
}

/// The shared store of [`WrapHandler`] layers per event.
///
/// Python keeps wrap handlers in the framework's callback registry under a
/// `"__wrap__:<event>"` key; [`CallbackFramework`] exposes only registration
/// metadata per event and holds the handler privately, so the layers live in
/// this dedicated, equally priority-ordered registry instead.
#[derive(Default)]
pub struct WrapRegistry {
    state: RwLock<HashMap<CallbackEvent, Vec<StoredWrap>>>,
    next_id: AtomicU64,
    next_sequence: AtomicU64,
}

#[derive(Clone)]
struct StoredWrap {
    id: WrapId,
    sequence: u64,
    priority: i32,
    enabled: bool,
    handler: Arc<dyn WrapHandler>,
}

impl WrapRegistry {
    /// An empty registry.
    #[must_use]
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Registers one layer; returns its removal handle.
    pub fn register(&self, registration: WrapRegistration) -> WrapId {
        let id = WrapId(self.next_id.fetch_add(1, Ordering::Relaxed));
        let sequence = self.next_sequence.fetch_add(1, Ordering::Relaxed);
        self.state
            .write()
            .entry(registration.event)
            .or_default()
            .push(StoredWrap {
                id,
                sequence,
                priority: registration.priority,
                enabled: true,
                handler: registration.handler,
            });
        id
    }

    /// Removes one layer; returns whether it existed.
    pub fn unregister(&self, id: WrapId) -> bool {
        let mut state = self.state.write();
        let mut removed = false;
        for layers in state.values_mut() {
            let before = layers.len();
            layers.retain(|layer| layer.id != id);
            removed |= layers.len() != before;
        }
        state.retain(|_, layers| !layers.is_empty());
        removed
    }

    /// Removes every layer for `event`; returns how many were removed.
    pub fn unregister_event(&self, event: &CallbackEvent) -> usize {
        self.state
            .write()
            .remove(event)
            .map_or(0, |layers| layers.len())
    }

    /// Enables or disables one layer; returns whether the handle exists.
    pub fn set_enabled(&self, id: WrapId, enabled: bool) -> bool {
        for layers in self.state.write().values_mut() {
            for layer in layers {
                if layer.id == id {
                    layer.enabled = enabled;
                    return true;
                }
            }
        }
        false
    }

    /// The enabled layers for `event`, outermost first (priority descending).
    #[must_use]
    pub fn handlers(&self, event: &CallbackEvent) -> Vec<Arc<dyn WrapHandler>> {
        let state = self.state.read();
        let Some(layers) = state.get(event) else {
            return Vec::new();
        };
        let mut ordered: Vec<&StoredWrap> = layers.iter().filter(|layer| layer.enabled).collect();
        ordered.sort_by(|left, right| {
            right
                .priority
                .cmp(&left.priority)
                .then(left.sequence.cmp(&right.sequence))
        });
        ordered
            .into_iter()
            .map(|layer| Arc::clone(&layer.handler))
            .collect()
    }

    /// The events with at least one registered layer.
    #[must_use]
    pub fn events(&self) -> Vec<CallbackEvent> {
        let mut events: Vec<CallbackEvent> = self.state.read().keys().cloned().collect();
        events.sort_unstable();
        events
    }

    /// The total number of registered layers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.state.read().values().map(Vec::len).sum()
    }

    /// Whether no layer is registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl fmt::Debug for WrapRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WrapRegistry")
            .field("events", &self.events())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// emit_before / emit_after / emit_around
// ---------------------------------------------------------------------------

enum EmitKind {
    Before(CallbackEvent),
    After(CallbackEvent),
    Around {
        before: CallbackEvent,
        after: CallbackEvent,
    },
}

/// Fires auxiliary events around the event it is registered for.
///
/// Python decorated a function with an event trigger; here the combinator is
/// itself a handler, so it fires its auxiliary event while the source event is
/// being dispatched. An auxiliary abort becomes [`CallbackVerdict::Abort`] on the
/// source event (Python's propagated `AbortError`), an auxiliary stop becomes
/// [`CallbackVerdict::Stop`], and a clean auxiliary dispatch lets the source
/// dispatch continue.
pub struct EmitHandler {
    name: String,
    framework: Arc<CallbackFramework>,
    kind: EmitKind,
}

impl EmitHandler {
    /// Overrides the name reported to the framework.
    #[must_use]
    pub fn named(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Fires one auxiliary event with the source payload.
    async fn fire(
        &self,
        source: &CallbackContext,
        event: CallbackEvent,
        with_output: bool,
    ) -> CallbackResult {
        let aux = aux_context(source, event.clone(), with_output);
        match self.framework.trigger(event, &aux).await {
            Ok(report) if report.aborted => Ok(CallbackVerdict::Stop),
            Ok(_) => Ok(CallbackVerdict::Continue),
            Err(error) => Ok(CallbackVerdict::Abort(error)),
        }
    }
}

/// Builds the context an auxiliary event sees: the source's session, component,
/// input, index, metadata, and guardrail content, plus its output when asked.
fn aux_context(
    source: &CallbackContext,
    event: CallbackEvent,
    with_output: bool,
) -> CallbackContext {
    let mut aux = CallbackContext::new(event)
        .with_component(source.component.clone())
        .with_metadata(
            "source_event",
            Value::String(source.event.as_str().to_owned()),
        );
    if let Some(session) = &source.session {
        aux = aux.with_session(Arc::clone(session));
    }
    if let Some(input) = &source.input {
        aux = aux.with_input(input.clone());
    }
    if with_output {
        if let Some(output) = &source.output {
            aux = aux.with_output(output.clone());
        }
    }
    aux.index = source.index;
    aux.error.clone_from(&source.error);
    aux.metadata.extend(
        source
            .metadata
            .iter()
            .map(|(key, value)| (key.clone(), value.clone())),
    );
    aux.guardrail.clone_from(&source.guardrail);
    aux
}

#[async_trait]
impl CallbackHandler for EmitHandler {
    fn name(&self) -> &str {
        &self.name
    }

    async fn handle(&self, ctx: &CallbackContext) -> CallbackResult {
        match &self.kind {
            EmitKind::Before(event) => self.fire(ctx, event.clone(), false).await,
            EmitKind::After(event) => self.fire(ctx, event.clone(), true).await,
            EmitKind::Around { before, after } => {
                match self.fire(ctx, before.clone(), false).await? {
                    CallbackVerdict::Continue => {}
                    other => return Ok(other),
                }
                self.fire(ctx, after.clone(), true).await
            }
        }
    }
}

impl fmt::Debug for EmitHandler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EmitHandler")
            .field("name", &self.name)
            .field(
                "kind",
                &match &self.kind {
                    EmitKind::Before(event) => format!("before {}", event.as_str()),
                    EmitKind::After(event) => format!("after {}", event.as_str()),
                    EmitKind::Around { before, after } => {
                        format!("around {} / {}", before.as_str(), after.as_str())
                    }
                },
            )
            .finish_non_exhaustive()
    }
}

/// Fires `event` before the source dispatch continues (Python's `emit_before`).
#[must_use]
pub fn emit_before(framework: Arc<CallbackFramework>, event: CallbackEvent) -> EmitHandler {
    EmitHandler {
        name: format!("emit_before({})", event.as_str()),
        framework,
        kind: EmitKind::Before(event),
    }
}

/// Fires `event` with the source's output payload (Python's `emit_after`).
#[must_use]
pub fn emit_after(framework: Arc<CallbackFramework>, event: CallbackEvent) -> EmitHandler {
    EmitHandler {
        name: format!("emit_after({})", event.as_str()),
        framework,
        kind: EmitKind::After(event),
    }
}

/// Fires `before` then `after` inside one handler.
///
/// Python's `emit_around` wrapped a function call; a handler cannot interpose on
/// the caller's call, so the two auxiliary dispatches happen back to back. Use
/// [`wrap`] when the "after" event must observe work done between the two.
#[must_use]
pub fn emit_around(
    framework: Arc<CallbackFramework>,
    before: CallbackEvent,
    after: CallbackEvent,
) -> EmitHandler {
    EmitHandler {
        name: format!("emit_around({} / {})", before.as_str(), after.as_str()),
        framework,
        kind: EmitKind::Around { before, after },
    }
}

// ---------------------------------------------------------------------------
// transform_io
// ---------------------------------------------------------------------------

/// One payload rewrite.
pub type PayloadTransform = Arc<dyn Fn(&Value) -> Result<Value> + Send + Sync>;

#[derive(Clone, Default)]
struct TransformState {
    input: Option<Value>,
    output: Option<Value>,
}

/// Rewrites the input payload before downstream work and the output afterwards.
///
/// Ports `create_transform_io_decorator`. Two forms, because
/// [`CallbackHandler::handle`] receives `&CallbackContext`:
///
/// * As an [`EventFilter`] (and therefore via [`HandlerBuilder::when`] or a
///   [`ChainEntry`]) it rewrites the working context in place, so every later
///   entry observes the transformed payloads.
/// * As a [`CallbackHandler`] it rewrites its own clone of the context and
///   publishes the result through [`Self::last_input`]/[`Self::last_output`].
/// * [`Self::apply`] transforms a direct async call, which is the closest to
///   Python's decorator.
///
/// A transform error is propagated, never swallowed.
pub struct TransformHandler {
    name: String,
    input: Option<PayloadTransform>,
    output: Option<PayloadTransform>,
    state: Arc<Mutex<TransformState>>,
}

impl TransformHandler {
    /// A transform with no rewrites yet.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            input: None,
            output: None,
            state: Arc::new(Mutex::new(TransformState::default())),
        }
    }

    /// Sets the input rewrite.
    #[must_use]
    pub fn with_input(
        mut self,
        transform: impl Fn(&Value) -> Result<Value> + Send + Sync + 'static,
    ) -> Self {
        self.input = Some(Arc::new(transform));
        self
    }

    /// Sets the output rewrite.
    #[must_use]
    pub fn with_output(
        mut self,
        transform: impl Fn(&Value) -> Result<Value> + Send + Sync + 'static,
    ) -> Self {
        self.output = Some(Arc::new(transform));
        self
    }

    /// The last rewritten input this handler produced.
    #[must_use]
    pub fn last_input(&self) -> Option<Value> {
        self.state.lock().input.clone()
    }

    /// The last rewritten output this handler produced.
    #[must_use]
    pub fn last_output(&self) -> Option<Value> {
        self.state.lock().output.clone()
    }

    /// Rewrites `ctx` in place (the filter form).
    ///
    /// # Errors
    ///
    /// Propagates the transform's error.
    pub fn rewrite(&self, ctx: &mut CallbackContext) -> Result<()> {
        if let (Some(transform), Some(input)) = (&self.input, &ctx.input) {
            let rewritten = transform(input)?;
            self.state.lock().input = Some(rewritten.clone());
            ctx.input = Some(rewritten);
        }
        if let (Some(transform), Some(output)) = (&self.output, &ctx.output) {
            let rewritten = transform(output)?;
            self.state.lock().output = Some(rewritten.clone());
            ctx.output = Some(rewritten);
        }
        Ok(())
    }

    /// Runs `call` between the input and output rewrites.
    ///
    /// # Errors
    ///
    /// Propagates a transform error or the call's error.
    pub async fn apply<F, Fut>(&self, input: Value, call: F) -> Result<Value>
    where
        F: FnOnce(Value) -> Fut,
        Fut: Future<Output = Result<Value>>,
    {
        let input = match &self.input {
            Some(transform) => transform(&input)?,
            None => input,
        };
        self.state.lock().input = Some(input.clone());
        let output = call(input).await?;
        let output = match &self.output {
            Some(transform) => transform(&output)?,
            None => output,
        };
        self.state.lock().output = Some(output.clone());
        Ok(output)
    }
}

impl fmt::Debug for TransformHandler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TransformHandler")
            .field("name", &self.name)
            .field("input", &self.input.is_some())
            .field("output", &self.output.is_some())
            .finish_non_exhaustive()
    }
}

/// Builds a [`TransformHandler`] (Python's `transform_io`).
#[must_use]
pub fn transform_io(name: impl Into<String>) -> TransformHandler {
    TransformHandler::new(name)
}

#[async_trait]
impl CallbackHandler for TransformHandler {
    fn name(&self) -> &str {
        &self.name
    }

    async fn handle(&self, ctx: &CallbackContext) -> CallbackResult {
        let mut working = ctx.clone();
        self.rewrite(&mut working)?;
        Ok(CallbackVerdict::Continue)
    }
}

impl EventFilter for TransformHandler {
    fn name(&self) -> &str {
        &self.name
    }

    fn allows(
        &self,
        input: super::filters::FilterInput<'_>,
    ) -> Result<super::filters::FilterDecision> {
        self.rewrite(input.ctx)?;
        Ok(super::filters::FilterDecision::Allow)
    }
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
    use crate::callback::filters::{FilterDecision, FilterInput, KindFilter, NameFilter};
    use serde_json::json;

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

    #[tokio::test]
    async fn wrap_layers_run_outside_in_and_can_short_circuit() {
        let order = Arc::new(Mutex::new(Vec::new()));
        let inner_ran = Arc::new(AtomicBool::new(false));

        let inner_flag = Arc::clone(&inner_ran);
        let call = payload_call(move |input: Value| {
            let flag = Arc::clone(&inner_flag);
            async move {
                flag.store(true, Ordering::Relaxed);
                Ok(json!({"echo": input}))
            }
        });

        let outer_order = Arc::clone(&order);
        let layer = wrap_handler("layer", move |_ctx, next: NextCall| {
            let order = Arc::clone(&outer_order);
            Box::pin(async move {
                order.lock().push("before".to_owned());
                let value = next.call(json!("rewritten")).await?;
                order.lock().push("after".to_owned());
                Ok(value)
            })
        });

        let wrapped = wrap(CallbackEvent::ToolInvokeInput, call).with_handler(layer);
        assert_eq!(wrapped.layers(), ["layer"]);
        let report = wrapped.call_with_report(json!("original")).await.unwrap();
        assert_eq!(report.value, json!({"echo": "rewritten"}));
        assert!(report.inner_ran);
        assert_eq!(*order.lock(), ["before", "after"]);

        let short = wrap(
            CallbackEvent::ToolInvokeInput,
            payload_call(|_: Value| async move { panic!("the wrapped call must not run") }),
        )
        .with_layer(ShortCircuit);
        let report = short.call_with_report(json!(null)).await.unwrap();
        assert_eq!(report.value, json!("short"));
        assert!(!report.inner_ran);
        assert!(format!("{short:?}").contains("short_circuit"));
    }

    struct ShortCircuit;

    #[async_trait]
    impl WrapHandler for ShortCircuit {
        fn name(&self) -> &str {
            "short_circuit"
        }

        async fn handle(&self, _ctx: &CallbackContext, call_next: NextCall) -> Result<Value> {
            assert!(!call_next.called());
            Ok(json!("short"))
        }
    }

    #[tokio::test]
    async fn the_wrap_registry_resolves_layers_per_call() {
        let registry = WrapRegistry::new();
        assert!(registry.is_empty());
        let id = on_wrap(
            CallbackEvent::AgentStarted,
            wrap_handler("upper", |_ctx, next: NextCall| {
                Box::pin(async move {
                    let value = next.call(json!(null)).await?;
                    Ok(json!(value.as_str().unwrap_or_default().to_uppercase()))
                })
            }),
        )
        .register(&registry);

        let wrapped = WrappedCall::from_registry(
            Arc::clone(&registry),
            CallbackEvent::AgentStarted,
            payload_call(|_: Value| async move { Ok(json!("abc")) }),
        );
        assert_eq!(wrapped.call(json!(null)).await.unwrap(), json!("ABC"));
        assert_eq!(registry.events(), [CallbackEvent::AgentStarted]);
        assert_eq!(registry.len(), 1);
        assert!(registry.set_enabled(id, false));
        assert_eq!(wrapped.call(json!(null)).await.unwrap(), json!("abc"));
        assert!(registry.unregister(id));
        assert_eq!(registry.unregister_event(&CallbackEvent::AgentStarted), 0);
        assert!(format!("{registry:?}").contains("WrapRegistry"));
        assert_eq!(
            on_wrap(CallbackEvent::AgentStarted, Arc::new(ShortCircuit))
                .with_priority(3)
                .register(&registry)
                .to_string()
                .split('-')
                .next(),
            Some("wrap")
        );
    }

    #[tokio::test]
    async fn emit_before_and_after_carry_the_payload() {
        let framework = CallbackFramework::new();
        let seen = Arc::new(Mutex::new(Vec::new()));

        let before_seen = Arc::clone(&seen);
        framework.register_sync(
            CallbackEvent::LlmCallStarted,
            0,
            move |ctx: &CallbackContext| {
                before_seen.lock().push(format!(
                    "before:{}",
                    ctx.input.clone().unwrap_or(Value::Null)
                ));
                Ok(CallbackVerdict::Continue)
            },
        );
        let after_seen = Arc::clone(&seen);
        framework.register_sync(
            CallbackEvent::LlmResponseReceived,
            0,
            move |ctx: &CallbackContext| {
                after_seen.lock().push(format!(
                    "after:{}",
                    ctx.output.clone().unwrap_or(Value::Null)
                ));
                Ok(CallbackVerdict::Continue)
            },
        );

        let around = emit_around(
            Arc::clone(&framework),
            CallbackEvent::LlmCallStarted,
            CallbackEvent::LlmResponseReceived,
        );
        let ctx = CallbackContext::new(CallbackEvent::LlmInvokeInput)
            .with_component("assistant")
            .with_input(json!("hi"))
            .with_output(json!("hello"));
        assert!(matches!(
            around.handle(&ctx).await.unwrap(),
            CallbackVerdict::Continue
        ));
        assert_eq!(*seen.lock(), ["before:\"hi\"", "after:\"hello\""]);
        assert!(format!("{around:?}").contains("llm_call_started"));
    }

    #[tokio::test]
    async fn emit_propagates_an_auxiliary_abort() {
        let framework = CallbackFramework::new();
        framework.register_sync(CallbackEvent::ToolInvokeInput, 0, |_: &CallbackContext| {
            Ok(CallbackVerdict::Abort(Error::security("blocked")))
        });
        let before =
            emit_before(Arc::clone(&framework), CallbackEvent::ToolInvokeInput).named("gate");
        assert_eq!(before.name(), "gate");
        let ctx = CallbackContext::new(CallbackEvent::ToolInvokeOutput).with_output(json!("ok"));
        match before.handle(&ctx).await.unwrap() {
            CallbackVerdict::Abort(error) => assert_eq!(error.to_string(), "security: blocked"),
            other => panic!("expected an abort, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn transform_io_rewrites_both_payloads_and_propagates_errors() {
        let transform = transform_io("shout")
            .with_input(|value| Ok(json!(value.as_str().unwrap_or_default().to_uppercase())))
            .with_output(|value| Ok(json!(format!("{value}!"))));
        let output = transform
            .apply(json!("hi"), |input| async move {
                Ok(json!(format!(
                    "{}+call",
                    input.as_str().unwrap_or_default()
                )))
            })
            .await
            .unwrap();
        assert_eq!(output, json!("\"HI+call\"!"));
        assert_eq!(transform.last_input(), Some(json!("HI")));
        assert_eq!(transform.last_output(), Some(json!("\"HI+call\"!")));

        let failing = transform_io("failing").with_input(|_| Err(Error::validation("bad input")));
        let error = failing
            .apply(json!("hi"), |input| async move { Ok(input) })
            .await
            .unwrap_err();
        assert_eq!(error.to_string(), "validation: bad input");

        let ctx = CallbackContext::new(CallbackEvent::LlmInvokeInput).with_input(json!("hi"));
        assert!(matches!(
            EventFilter::allows(
                &transform,
                FilterInput {
                    handler: "h",
                    namespace: "default",
                    ctx: &mut ctx.clone(),
                }
            )
            .unwrap(),
            FilterDecision::Allow
        ));
        assert!(KindFilter::domain(crate::callback::EventDomain::Llm)
            .allows(FilterInput {
                handler: "h",
                namespace: "default",
                ctx: &mut CallbackContext::new(CallbackEvent::LlmInvokeInput),
            })
            .unwrap()
            .is_allow());
    }
}
