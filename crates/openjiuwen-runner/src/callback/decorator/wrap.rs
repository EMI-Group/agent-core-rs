//! The call-wrapping layer: `wrap`, `on_wrap`, `WrappedCall`, `WrapRegistry`.
//!
//! Ports Python's `create_wrap_decorator` / `create_on_wrap_decorator` /
//! `create_wrap_by_event_decorator`. A wrap layer is a [`WrapHandler`]: it
//! receives the pending call as a [`NextCall`], may rewrite the argument, may
//! rewrite the result, and may short-circuit by never calling it.
//!
//! [`WrapRegistry`] is the `__wrap__:<event>` channel of Python: a separate
//! registry because the framework hides handler values from callers, so a
//! wrapped call cannot recover its layers from a `CallbackFramework`.
#![allow(clippy::module_name_repetitions)]
// A wrap layer computes its name at construction time, so the trait cannot return `&'static str`.
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
use crate::callback::context::CallbackContext;
use crate::callback::event::CallbackEvent;
use async_trait::async_trait;
use futures::future::BoxFuture;
use openjiuwen_common::error::Result;
use parking_lot::RwLock;
use serde_json::Value;
use std::collections::HashMap;
use std::fmt;
use std::future::Future;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

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

#[cfg(test)]
mod tests {
    use super::*;
    use parking_lot::Mutex;
    use serde_json::json;
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
}
