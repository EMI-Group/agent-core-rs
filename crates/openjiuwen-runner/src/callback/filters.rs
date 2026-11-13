//! Event filters: composable predicates that gate a handler before it runs.
//!
//! Ports `openjiuwen/core/runner/callback/filters.py` together with
//! `FilterAction`/`FilterResult` from `enums.py`/`models.py`, translated to the
//! crate's error-propagating style:
//!
//! * Python's `FilterAction::CONTINUE` is `FilterDecision::Allow`.
//! * `FilterAction::SKIP` is `FilterDecision::Skip`, carrying the reason for
//!   telemetry. Dispatch continues with the next handler.
//! * `FilterAction::STOP`-as-failure is `FilterDecision::Reject`, carrying the
//!   `Error` that ends dispatch.
//! * `FilterAction::MODIFY` needs no variant: `FilterInput::ctx` is a mutable
//!   working context, so a filter rewrites the payload in place and every later
//!   handler (and later filter) sees the mutation.
//!
//! This module owns the vocabulary (decisions, the trait, the dispatch helpers,
//! the injectable clock) and the composition combinators (`FilterChain`,
//! `OrFilter`, `NotFilter`, `LenientFilter`, the `and`/`or`/`not` free
//! functions, and the `EventFilterExt` sugar).
//! The concrete filters live in `builtin` and are re-exported here, so
//! `callback::filters::RateLimitFilter` and `callback::filters::builtin::RateLimitFilter`
//! name the same type.
//!
//! Dispatch note: a filter that returns an error fails closed (it becomes a
//! rejection); wrap it in `lenient` for Python's skip-on-error behaviour.
#![allow(clippy::module_name_repetitions)]
// A filter computes its name at construction time, so the trait cannot return `&'static str`.
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
use super::context::CallbackContext;
use openjiuwen_common::error::{Error, Result};
use parking_lot::Mutex;
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};
pub mod builtin;
pub use builtin::{
    AuthFilter, CircuitBreakerFilter, ConditionalFilter, ContextModifier, KindFilter,
    LoggingFilter, NameFilter, ParamModifyFilter, PredicateFilter, RateLimitFilter, SessionFilter,
    ValidationFilter,
};

use crate::callback::LOG_TARGET;
/// What a filter decided about the handler it guards.
#[derive(Debug)]
pub enum FilterDecision {
    /// Run the handler.
    Allow,
    /// Do not run this handler; the caller moves on to the next one.
    Skip {
        /// Why the handler was skipped, for logs and reports.
        reason: String,
    },
    /// End dispatch and surface this error to the caller.
    Reject(Error),
}

impl FilterDecision {
    /// A soft skip carrying `reason`.
    #[must_use]
    pub fn skip(reason: impl Into<String>) -> Self {
        Self::Skip {
            reason: reason.into(),
        }
    }

    /// A hard rejection carrying `error`.
    #[must_use]
    pub fn reject(error: Error) -> Self {
        Self::Reject(error)
    }

    /// Whether the handler may run.
    #[must_use]
    pub const fn is_allow(&self) -> bool {
        matches!(self, Self::Allow)
    }

    /// Whether the handler is skipped without failing dispatch.
    #[must_use]
    pub const fn is_skip(&self) -> bool {
        matches!(self, Self::Skip { .. })
    }

    /// Whether the decision ends dispatch with an error.
    #[must_use]
    pub const fn is_reject(&self) -> bool {
        matches!(self, Self::Reject(_))
    }

    /// The skip reason, when the decision is a skip.
    #[must_use]
    pub fn reason(&self) -> Option<&str> {
        match self {
            Self::Skip { reason } => Some(reason),
            Self::Allow | Self::Reject(_) => None,
        }
    }
}

/// Everything a filter may inspect, including the handler it guards.
///
/// `ctx` is a mutable working copy: a filter that rewrites
/// [`CallbackContext::input`]/[`CallbackContext::output`] changes what every
/// later handler observes (Python's `FilterAction::MODIFY`, without a separate
/// variant).
#[derive(Debug)]
pub struct FilterInput<'a> {
    /// The name of the handler being filtered.
    pub handler: &'a str,
    /// The namespace of the handler being filtered (`"default"` unless set).
    pub namespace: &'a str,
    /// The mutable working context of the dispatch.
    pub ctx: &'a mut CallbackContext,
}

/// Gates one handler per dispatch.
pub trait EventFilter: Send + Sync {
    /// The filter's name, used in reports, logs, and [`FilterChain`] reasons.
    fn name(&self) -> &str {
        "filter"
    }

    /// Decides whether the guarded handler runs.
    ///
    /// # Errors
    ///
    /// Returning `Err` is a filter failure. Callers treat it as fatal (fail
    /// closed): a filter that cannot decide must not silently allow the handler.
    /// Use [`LenientFilter`] to turn filter errors into [`FilterDecision::Skip`].
    fn allows(&self, input: FilterInput<'_>) -> Result<FilterDecision>;

    /// An optional feedback sink, called by the dispatch loop with the guarded
    /// handler's outcome.
    ///
    /// Only stateful filters (currently [`CircuitBreakerFilter`]) implement it.
    fn feedback(&self) -> Option<&dyn FilterFeedback> {
        None
    }
}

/// Receives the guarded handler's outcome, so a filter can track health.
pub trait FilterFeedback: Send + Sync {
    /// The guarded handler ran successfully.
    fn record_success(&self, ctx: &CallbackContext, handler: &str) {
        let _ = (ctx, handler);
    }

    /// The guarded handler failed.
    fn record_failure(&self, ctx: &CallbackContext, handler: &str) {
        let _ = (ctx, handler);
    }
}

impl EventFilter for Arc<dyn EventFilter> {
    fn name(&self) -> &str {
        (**self).name()
    }

    fn allows(&self, input: FilterInput<'_>) -> Result<FilterDecision> {
        (**self).allows(input)
    }

    fn feedback(&self) -> Option<&dyn FilterFeedback> {
        (**self).feedback()
    }
}

impl<F> EventFilter for F
where
    F: Fn(FilterInput<'_>) -> Result<FilterDecision> + Send + Sync,
{
    fn allows(&self, input: FilterInput<'_>) -> Result<FilterDecision> {
        self(input)
    }
}

/// A named closure filter, for hosts that want more than `"filter"` in reports.
pub struct ClosureFilter<F> {
    name: String,
    predicate: F,
}

impl<F> ClosureFilter<F>
where
    F: Fn(FilterInput<'_>) -> Result<FilterDecision> + Send + Sync,
{
    /// Wraps `predicate`, reporting `name`.
    #[must_use]
    pub fn new(name: impl Into<String>, predicate: F) -> Self {
        Self {
            name: name.into(),
            predicate,
        }
    }
}

impl<F> EventFilter for ClosureFilter<F>
where
    F: Fn(FilterInput<'_>) -> Result<FilterDecision> + Send + Sync,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn allows(&self, input: FilterInput<'_>) -> Result<FilterDecision> {
        (self.predicate)(input)
    }
}

/// The outcome of running a whole filter list against one handler.
#[derive(Debug)]
pub enum FilterRunOutcome {
    /// Every filter allowed the handler.
    Allow,
    /// A filter skipped the handler; later filters did not run.
    Skip {
        /// The skipping filter's name.
        filter: String,
        /// Why it skipped.
        reason: String,
    },
    /// A filter rejected the handler; dispatch ends with this error.
    Reject(Error),
}

/// Runs `filters` in order against one handler.
///
/// The first [`FilterDecision::Skip`] or [`FilterDecision::Reject`] short
/// circuits; mutations a filter made to `ctx` are kept for later filters and
/// for the handler.
///
/// # Errors
///
/// Returns the [`Error`] of the first filter that failed (fail-closed).
pub fn run_filters(
    filters: &[Arc<dyn EventFilter>],
    handler: &str,
    namespace: &str,
    ctx: &mut CallbackContext,
) -> Result<FilterRunOutcome> {
    for filter in filters {
        let decision = filter.allows(FilterInput {
            handler,
            namespace,
            ctx,
        })?;
        match decision {
            FilterDecision::Allow => {}
            FilterDecision::Skip { reason } => {
                return Ok(FilterRunOutcome::Skip {
                    filter: filter.name().to_owned(),
                    reason,
                })
            }
            FilterDecision::Reject(error) => return Ok(FilterRunOutcome::Reject(error)),
        }
    }
    Ok(FilterRunOutcome::Allow)
}

/// Tells every feedback-capable filter that `handler` succeeded.
pub fn record_success(filters: &[Arc<dyn EventFilter>], ctx: &CallbackContext, handler: &str) {
    for filter in filters {
        if let Some(sink) = filter.feedback() {
            sink.record_success(ctx, handler);
        }
    }
}

/// Tells every feedback-capable filter that `handler` failed.
pub fn record_failure(filters: &[Arc<dyn EventFilter>], ctx: &CallbackContext, handler: &str) {
    for filter in filters {
        if let Some(sink) = filter.feedback() {
            sink.record_failure(ctx, handler);
        }
    }
}

/// An injectable monotonic clock.
///
/// The default is [`Clock::system`] (`std::time::Instant::now`); tests use
/// [`ManualClock::clock`] so rate limiting and circuit breaking are deterministic
/// without sleeping.
#[derive(Clone)]
pub struct Clock {
    now: Arc<dyn Fn() -> Instant + Send + Sync>,
}

impl Clock {
    /// A clock reading `std::time::Instant::now`.
    #[must_use]
    pub fn system() -> Self {
        Self {
            now: Arc::new(Instant::now),
        }
    }

    /// A clock driven by `now`.
    #[must_use]
    pub fn from_fn(now: impl Fn() -> Instant + Send + Sync + 'static) -> Self {
        Self { now: Arc::new(now) }
    }

    /// The current reading.
    #[must_use]
    pub fn now(&self) -> Instant {
        (self.now)()
    }
}

impl Default for Clock {
    fn default() -> Self {
        Self::system()
    }
}

impl fmt::Debug for Clock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Clock")
    }
}

/// A hand-advanced clock for deterministic tests.
#[derive(Clone, Debug)]
pub struct ManualClock {
    now: Arc<Mutex<Instant>>,
}

impl ManualClock {
    /// A clock reading the current instant.
    #[must_use]
    pub fn new() -> Self {
        Self::at(Instant::now())
    }

    /// A clock pinned to `at`.
    #[must_use]
    pub fn at(at: Instant) -> Self {
        Self {
            now: Arc::new(Mutex::new(at)),
        }
    }

    /// Moves the clock forward by `by`.
    pub fn advance(&self, by: Duration) {
        *self.now.lock() += by;
    }

    /// Pins the clock to `at`.
    pub fn set(&self, at: Instant) {
        *self.now.lock() = at;
    }

    /// The current reading.
    #[must_use]
    pub fn now(&self) -> Instant {
        *self.now.lock()
    }

    /// A [`Clock`] backed by this manual clock.
    #[must_use]
    pub fn clock(&self) -> Clock {
        let now = Arc::clone(&self.now);
        Clock::from_fn(move || *now.lock())
    }
}

impl Default for ManualClock {
    fn default() -> Self {
        Self::new()
    }
}
/// Runs several filters in order; the first Skip/Reject wins (logical AND).
#[derive(Default)]
pub struct FilterChain {
    filters: Vec<Arc<dyn EventFilter>>,
}

impl FilterChain {
    /// An empty chain, which allows everything.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends `filter`.
    #[must_use]
    pub fn with(mut self, filter: impl EventFilter + 'static) -> Self {
        self.filters.push(Arc::new(filter));
        self
    }

    /// Appends an already shared `filter`.
    #[must_use]
    pub fn with_arc(mut self, filter: Arc<dyn EventFilter>) -> Self {
        self.filters.push(filter);
        self
    }

    /// Appends `filter` in place.
    pub fn push(&mut self, filter: impl EventFilter + 'static) {
        self.filters.push(Arc::new(filter));
    }

    /// The filters, in evaluation order.
    #[must_use]
    pub fn filters(&self) -> &[Arc<dyn EventFilter>] {
        &self.filters
    }

    /// How many filters the chain holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.filters.len()
    }

    /// Whether the chain holds no filters.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.filters.is_empty()
    }
}

impl fmt::Debug for FilterChain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FilterChain")
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

impl EventFilter for FilterChain {
    fn name(&self) -> &str {
        "filter_chain"
    }

    fn allows(&self, input: FilterInput<'_>) -> Result<FilterDecision> {
        for filter in &self.filters {
            match filter.allows(FilterInput {
                handler: input.handler,
                namespace: input.namespace,
                ctx: input.ctx,
            })? {
                FilterDecision::Allow => {}
                other => return Ok(other),
            }
        }
        Ok(FilterDecision::Allow)
    }

    fn feedback(&self) -> Option<&dyn FilterFeedback> {
        None
    }
}

/// Allows when either filter allows (logical OR).
///
/// A rejection from either side wins over a skip, so a hard guard cannot be
/// hidden by a permissive sibling.
pub struct OrFilter {
    left: Arc<dyn EventFilter>,
    right: Arc<dyn EventFilter>,
}

impl OrFilter {
    /// Combines `left` and `right`.
    #[must_use]
    pub fn new(left: impl EventFilter + 'static, right: impl EventFilter + 'static) -> Self {
        Self {
            left: Arc::new(left),
            right: Arc::new(right),
        }
    }
}

impl EventFilter for OrFilter {
    fn name(&self) -> &str {
        "or"
    }

    fn allows(&self, input: FilterInput<'_>) -> Result<FilterDecision> {
        let left = match self.left.allows(FilterInput {
            handler: input.handler,
            namespace: input.namespace,
            ctx: input.ctx,
        })? {
            FilterDecision::Allow => return Ok(FilterDecision::Allow),
            FilterDecision::Reject(error) => return Ok(FilterDecision::Reject(error)),
            FilterDecision::Skip { reason } => reason,
        };
        match self.right.allows(FilterInput {
            handler: input.handler,
            namespace: input.namespace,
            ctx: input.ctx,
        })? {
            FilterDecision::Allow => Ok(FilterDecision::Allow),
            FilterDecision::Reject(error) => Ok(FilterDecision::Reject(error)),
            FilterDecision::Skip { reason } => Ok(FilterDecision::skip(format!(
                "neither filter allowed the handler: {left}; {reason}"
            ))),
        }
    }
}

/// Inverts a filter: an allow becomes a skip and a skip becomes an allow.
///
/// Rejections are not inverted — a filter that faults still faults.
pub struct NotFilter {
    inner: Arc<dyn EventFilter>,
}

impl NotFilter {
    /// Inverts `inner`.
    #[must_use]
    pub fn new(inner: impl EventFilter + 'static) -> Self {
        Self {
            inner: Arc::new(inner),
        }
    }
}

impl EventFilter for NotFilter {
    fn name(&self) -> &str {
        "not"
    }

    fn allows(&self, input: FilterInput<'_>) -> Result<FilterDecision> {
        let inner = Arc::clone(&self.inner);
        let allowed = inner.allows(FilterInput {
            handler: input.handler,
            namespace: input.namespace,
            ctx: input.ctx,
        })?;
        Ok(match allowed {
            FilterDecision::Allow => FilterDecision::skip("negated filter allowed the handler"),
            FilterDecision::Skip { .. } => FilterDecision::Allow,
            FilterDecision::Reject(error) => FilterDecision::Reject(error),
        })
    }
}

/// Turns a filter error into a soft [`FilterDecision::Skip`].
///
/// The Python family skipped on internal errors; this wrapper restores that
/// behaviour where fail-closed is not wanted.
pub struct LenientFilter {
    inner: Arc<dyn EventFilter>,
}

impl LenientFilter {
    /// Wraps `inner` so its errors skip instead of aborting.
    #[must_use]
    pub fn new(inner: impl EventFilter + 'static) -> Self {
        Self {
            inner: Arc::new(inner),
        }
    }
}

impl EventFilter for LenientFilter {
    fn name(&self) -> &str {
        "lenient"
    }

    fn allows(&self, input: FilterInput<'_>) -> Result<FilterDecision> {
        let inner = Arc::clone(&self.inner);
        match inner.allows(FilterInput {
            handler: input.handler,
            namespace: input.namespace,
            ctx: input.ctx,
        }) {
            Ok(decision) => Ok(decision),
            Err(error) => {
                tracing::warn!(
                    target: LOG_TARGET,
                    filter = %inner.name(),
                    handler = %input.handler,
                    error = %error,
                    "filter failed; skipping the handler"
                );
                Ok(FilterDecision::skip(format!(
                    "{} failed: {error}",
                    inner.name()
                )))
            }
        }
    }
}

/// Filters composed as AND (delegates to [`FilterChain`]).
#[must_use]
pub fn and(left: impl EventFilter + 'static, right: impl EventFilter + 'static) -> FilterChain {
    FilterChain::new().with(left).with(right)
}

/// Filters composed as OR.
#[must_use]
pub fn or(left: impl EventFilter + 'static, right: impl EventFilter + 'static) -> OrFilter {
    OrFilter::new(left, right)
}

/// The negation of a filter.
#[must_use]
pub fn not(inner: impl EventFilter + 'static) -> NotFilter {
    NotFilter::new(inner)
}

/// A filter whose errors skip instead of aborting.
#[must_use]
pub fn lenient(inner: impl EventFilter + 'static) -> LenientFilter {
    LenientFilter::new(inner)
}

/// Fluent combinators for any filter, including `Arc<dyn EventFilter>`.
pub trait EventFilterExt: EventFilter + Sized + 'static {
    /// Requires both `self` and `other` to allow.
    #[must_use]
    fn and<F>(self, other: F) -> FilterChain
    where
        F: EventFilter + 'static,
    {
        FilterChain::new().with(self).with(other)
    }

    /// Requires either `self` or `other` to allow.
    #[must_use]
    fn or<F>(self, other: F) -> OrFilter
    where
        F: EventFilter + 'static,
    {
        OrFilter::new(self, other)
    }

    /// Inverts `self`.
    #[must_use]
    fn negate(self) -> NotFilter {
        NotFilter::new(self)
    }

    /// Turns `self`'s errors into skips.
    #[must_use]
    fn lenient(self) -> LenientFilter {
        LenientFilter::new(self)
    }

    /// Erases `self` into the shared filter handle the dispatch loops store.
    #[must_use]
    fn into_filter_arc(self) -> Arc<dyn EventFilter> {
        Arc::new(self)
    }
}

impl<T> EventFilterExt for T where T: EventFilter + Sized + 'static {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::callback::CallbackEvent;
    use serde_json::json;
    fn context(event: CallbackEvent) -> CallbackContext {
        CallbackContext::new(event)
    }
    #[test]
    fn a_closure_is_a_filter() {
        let filter = |input: FilterInput<'_>| {
            Ok(if input.handler == "allowed" {
                FilterDecision::Allow
            } else {
                FilterDecision::skip("not allowed")
            })
        };
        let mut ctx = context(CallbackEvent::AgentInvokeInput);
        assert!(filter
            .allows(FilterInput {
                handler: "allowed",
                namespace: "default",
                ctx: &mut ctx,
            })
            .expect("allow")
            .is_allow());
        assert!(filter
            .allows(FilterInput {
                handler: "other",
                namespace: "default",
                ctx: &mut ctx,
            })
            .expect("skip")
            .is_skip());
    }
    #[test]
    fn an_empty_chain_allows_and_the_combinators_compose() {
        let empty = FilterChain::new();
        let mut ctx = context(CallbackEvent::ToolInvokeInput);
        assert!(empty
            .allows(FilterInput {
                handler: "tool",
                namespace: "default",
                ctx: &mut ctx,
            })
            .expect("empty chain allows")
            .is_allow());
        let chained = and(
            NameFilter::named(["tool"]),
            not(NameFilter::named(["other"])),
        );
        assert_eq!(chained.len(), 2);
        assert!(chained
            .allows(FilterInput {
                handler: "tool",
                namespace: "default",
                ctx: &mut ctx,
            })
            .expect("composed allow")
            .is_allow());
        let negated = not(NameFilter::named(["tool"]));
        assert!(negated
            .allows(FilterInput {
                handler: "tool",
                namespace: "default",
                ctx: &mut ctx,
            })
            .expect("negated skip")
            .is_skip());
        let allowed = or(NameFilter::named(["nope"]), NameFilter::named(["tool"]));
        assert!(allowed
            .allows(FilterInput {
                handler: "tool",
                namespace: "default",
                ctx: &mut ctx,
            })
            .expect("or allow")
            .is_allow());
    }
    #[test]
    fn a_filter_error_aborts_unless_lenient() {
        let failing = ClosureFilter::new("failing", |_: FilterInput<'_>| {
            Err(Error::validation("boom"))
        });
        let mut ctx = context(CallbackEvent::AgentInvokeInput);
        let error = failing
            .allows(FilterInput {
                handler: "agent",
                namespace: "default",
                ctx: &mut ctx,
            })
            .expect_err("fails closed");
        assert!(error.to_string().contains("boom"));
        let tolerated = lenient(failing);
        assert!(tolerated
            .allows(FilterInput {
                handler: "agent",
                namespace: "default",
                ctx: &mut ctx,
            })
            .expect("lenient skips")
            .is_skip());
    }
    #[test]
    fn run_filters_reports_the_first_skip_and_keeps_mutations() {
        let filters: Vec<Arc<dyn EventFilter>> = vec![
            Arc::new(ParamModifyFilter::new(
                "seed",
                |ctx: &mut CallbackContext| {
                    ctx.input = Some(json!({ "seen": true }));
                    Ok(())
                },
            )),
            Arc::new(NameFilter::named(["wanted"])),
        ];
        let mut ctx = context(CallbackEvent::AgentInvokeInput);
        let outcome = run_filters(&filters, "wanted", "default", &mut ctx).expect("runs");
        assert!(matches!(outcome, FilterRunOutcome::Allow));
        assert_eq!(ctx.input, Some(json!({ "seen": true })));

        let mut ctx = context(CallbackEvent::AgentInvokeInput);
        let outcome = run_filters(&filters, "other", "default", &mut ctx).expect("runs");
        match outcome {
            FilterRunOutcome::Skip { filter, reason } => {
                assert_eq!(filter, "name");
                assert!(reason.contains("wanted"));
            }
            other => panic!("expected a skip, got {other:?}"),
        }
    }
}
