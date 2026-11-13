//! Event filters: composable predicates that gate a handler before it runs.
//!
//! Ports `openjiuwen/core/runner/callback/filters.py` together with
//! `FilterAction`/`FilterResult` from `enums.py`/`models.py`, translated to the
//! crate's error-propagating style:
//!
//! * Python's `FilterAction::CONTINUE` is [`FilterDecision::Allow`].
//! * `FilterAction::SKIP` is [`FilterDecision::Skip`], carrying the reason for
//!   telemetry. Dispatch continues with the next handler.
//! * `FilterAction::STOP`-as-failure is [`FilterDecision::Reject`], carrying the
//!   [`Error`] that ends dispatch.
//! * `FilterAction::MODIFY` needs no variant: [`FilterInput::ctx`] is a mutable
//!   working context, so a filter rewrites the payload in place and every later
//!   handler (and later filter) sees the mutation.
//!
//! Filters see the handler's identity ([`FilterInput::handler`]) and namespace
//! ([`FilterInput::namespace`]) in addition to the event context, which is what
//! makes the built-in [`NameFilter`] and [`FilterChain`] meaningful.
//!
//! Time-dependent filters ([`RateLimitFilter`], [`CircuitBreakerFilter`]) take an
//! injectable [`Clock`], so tests advance a [`ManualClock`] instead of sleeping.
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

use parking_lot::Mutex;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};

use openjiuwen_common::error::{Error, Result};
use serde_json::Value;

use super::context::CallbackContext;
use super::event::{CallbackEvent, EventDomain};
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

/// Matches handlers by name, name prefix, and namespace.
///
/// The name filter of the Python family, widened to cover namespace routing: a
/// dimension that was never configured does not constrain the match, so
/// `NameFilter::new()` allows everything.
#[derive(Debug, Clone, Default)]
pub struct NameFilter {
    names: BTreeSet<String>,
    prefixes: Vec<String>,
    namespaces: BTreeSet<String>,
}

impl NameFilter {
    /// A filter that matches every handler.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Restricts the match to the exact handler `name` (repeatable).
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.names.insert(name.into());
        self
    }

    /// Restricts the match to handlers whose name starts with `prefix`.
    #[must_use]
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefixes.push(prefix.into());
        self
    }

    /// Restricts the match to one namespace (repeatable).
    #[must_use]
    pub fn with_namespace(mut self, namespace: impl Into<String>) -> Self {
        self.namespaces.insert(namespace.into());
        self
    }

    /// A filter matching any of `names`.
    #[must_use]
    pub fn named<I, S>(names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut filter = Self::new();
        for name in names {
            filter.names.insert(name.into());
        }
        filter
    }
}

impl EventFilter for NameFilter {
    fn name(&self) -> &str {
        "name"
    }

    fn allows(&self, input: FilterInput<'_>) -> Result<FilterDecision> {
        if !self.names.is_empty() && !self.names.contains(input.handler) {
            return Ok(FilterDecision::skip(format!(
                "handler '{}' is not in the name set {:?}",
                input.handler, self.names
            )));
        }
        if !self.prefixes.is_empty()
            && !self
                .prefixes
                .iter()
                .any(|prefix| input.handler.starts_with(prefix.as_str()))
        {
            return Ok(FilterDecision::skip(format!(
                "handler '{}' matches no configured prefix",
                input.handler
            )));
        }
        if !self.namespaces.is_empty() && !self.namespaces.contains(input.namespace) {
            return Ok(FilterDecision::skip(format!(
                "namespace '{}' is not in the namespace set",
                input.namespace
            )));
        }
        Ok(FilterDecision::Allow)
    }
}

/// Matches events by exact variant or by [`EventDomain`].
#[derive(Debug, Clone, Default)]
pub struct KindFilter {
    events: BTreeSet<CallbackEvent>,
    domains: BTreeSet<EventDomain>,
}

impl KindFilter {
    /// A filter that matches every event.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Restricts the match to `event`.
    #[must_use]
    pub fn event(event: CallbackEvent) -> Self {
        Self::new().with_event(event)
    }

    /// Restricts the match to `domain`.
    #[must_use]
    pub fn domain(domain: EventDomain) -> Self {
        Self::new().with_domain(domain)
    }

    /// Adds one exact event to the allowed set.
    #[must_use]
    pub fn with_event(mut self, event: CallbackEvent) -> Self {
        self.events.insert(event);
        self
    }

    /// Adds one domain to the allowed set.
    #[must_use]
    pub fn with_domain(mut self, domain: EventDomain) -> Self {
        self.domains.insert(domain);
        self
    }

    /// The exact events this filter allows.
    #[must_use]
    pub fn events(&self) -> &BTreeSet<CallbackEvent> {
        &self.events
    }

    /// The domains this filter allows.
    #[must_use]
    pub fn domains(&self) -> &BTreeSet<EventDomain> {
        &self.domains
    }
}

impl EventFilter for KindFilter {
    fn name(&self) -> &str {
        "kind"
    }

    fn allows(&self, input: FilterInput<'_>) -> Result<FilterDecision> {
        if self.events.is_empty() && self.domains.is_empty() {
            return Ok(FilterDecision::Allow);
        }
        let event = &input.ctx.event;
        if self.events.contains(event) || self.domains.contains(&event.domain()) {
            return Ok(FilterDecision::Allow);
        }
        Ok(FilterDecision::skip(format!(
            "event '{}' is not in the kind set",
            event.as_str()
        )))
    }
}

/// Matches the context's session id, with explicit "no session" semantics.
#[derive(Debug, Clone, Default)]
pub struct SessionFilter {
    ids: BTreeSet<String>,
    allow_absent: bool,
}

impl SessionFilter {
    /// A filter matching session `id` only.
    #[must_use]
    pub fn only(id: impl Into<String>) -> Self {
        Self::default().with_id(id)
    }

    /// A filter matching any of `ids`.
    #[must_use]
    pub fn any_of<I, S>(ids: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut filter = Self::default();
        for id in ids {
            filter.ids.insert(id.into());
        }
        filter
    }

    /// Adds one accepted session id.
    #[must_use]
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.ids.insert(id.into());
        self
    }

    /// Also allows events that carry no session at all (default: skipped).
    #[must_use]
    pub fn allow_absent(mut self) -> Self {
        self.allow_absent = true;
        self
    }
}

impl EventFilter for SessionFilter {
    fn name(&self) -> &str {
        "session"
    }

    fn allows(&self, input: FilterInput<'_>) -> Result<FilterDecision> {
        match input.ctx.session_id() {
            Some(id) if self.ids.is_empty() || self.ids.contains(id) => Ok(FilterDecision::Allow),
            Some(id) => Ok(FilterDecision::skip(format!(
                "session '{id}' is not in the session set"
            ))),
            None if self.allow_absent => Ok(FilterDecision::Allow),
            None => Ok(FilterDecision::skip("event carries no session")),
        }
    }
}

/// Allows at most `max_calls` executions per sliding `window`, per event/handler.
///
/// Ports `RateLimitFilter`. The window is pruned on every call using the injected
/// [`Clock`], so no timer task is involved.
pub struct RateLimitFilter {
    max_calls: usize,
    window: Duration,
    clock: Clock,
    calls: Mutex<BTreeMap<String, VecDeque<Instant>>>,
}

impl RateLimitFilter {
    /// At most `max_calls` executions per `window`.
    #[must_use]
    pub fn new(max_calls: usize, window: Duration) -> Self {
        Self {
            max_calls,
            window,
            clock: Clock::system(),
            calls: Mutex::new(BTreeMap::new()),
        }
    }

    /// Replaces the clock (tests use [`ManualClock::clock`]).
    #[must_use]
    pub fn with_clock(mut self, clock: Clock) -> Self {
        self.clock = clock;
        self
    }

    /// The configured call budget.
    #[must_use]
    pub const fn max_calls(&self) -> usize {
        self.max_calls
    }

    /// The configured window.
    #[must_use]
    pub const fn window(&self) -> Duration {
        self.window
    }

    /// How many calls are currently counted inside the window.
    #[must_use]
    pub fn calls(&self, ctx: &CallbackContext, handler: &str) -> usize {
        let now = self.clock.now();
        let calls = self.calls.lock();
        calls.get(&key(ctx, handler)).map_or(0, |window| {
            window
                .iter()
                .filter(|at| now.saturating_duration_since(**at) <= self.window)
                .count()
        })
    }
}

impl EventFilter for RateLimitFilter {
    fn name(&self) -> &str {
        "rate_limit"
    }

    fn allows(&self, input: FilterInput<'_>) -> Result<FilterDecision> {
        let now = self.clock.now();
        let key = key(input.ctx, input.handler);
        let mut calls = self.calls.lock();
        let window = calls.entry(key).or_default();
        while let Some(front) = window.front() {
            if now.saturating_duration_since(*front) > self.window {
                window.pop_front();
            } else {
                break;
            }
        }
        if window.len() >= self.max_calls {
            return Ok(FilterDecision::skip(format!(
                "rate limit exceeded: {} calls per {} ms",
                self.max_calls,
                self.window.as_millis()
            )));
        }
        window.push_back(now);
        Ok(FilterDecision::Allow)
    }
}

#[derive(Debug, Clone, Default)]
struct BreakerState {
    failures: u32,
    opened_at: Option<Instant>,
}

struct BreakerInner {
    threshold: u32,
    cooldown: Duration,
    clock: Mutex<Clock>,
    state: Mutex<BTreeMap<String, BreakerState>>,
}

impl BreakerInner {
    fn now(&self) -> Instant {
        self.clock.lock().now()
    }
}

/// Skips a handler whose circuit is open after repeated failures.
///
/// Ports `CircuitBreakerFilter`: after `failure_threshold` recorded failures the
/// circuit opens for `cooldown`, then closes again on the first call that finds
/// the cooldown elapsed. The filter reports itself as a [`FilterFeedback`] sink,
/// so a dispatch loop that honours [`EventFilter::feedback`] feeds it
/// automatically; a host may also call [`CircuitBreakerFilter::record_failure`]
/// directly.
#[derive(Clone)]
pub struct CircuitBreakerFilter {
    inner: Arc<BreakerInner>,
}

impl CircuitBreakerFilter {
    /// Opens the circuit after `failure_threshold` failures for `cooldown`.
    #[must_use]
    pub fn new(failure_threshold: u32, cooldown: Duration) -> Self {
        Self {
            inner: Arc::new(BreakerInner {
                threshold: failure_threshold.max(1),
                cooldown,
                clock: Mutex::new(Clock::system()),
                state: Mutex::new(BTreeMap::new()),
            }),
        }
    }

    /// Replaces the clock (tests use [`ManualClock::clock`]).
    ///
    /// The clock lives behind the shared breaker state, so this works on a clone
    /// too and keeps any failures already recorded.
    #[must_use]
    pub fn with_clock(self, clock: Clock) -> Self {
        *self.inner.clock.lock() = clock;
        self
    }

    /// The configured failure threshold.
    #[must_use]
    pub fn failure_threshold(&self) -> u32 {
        self.inner.threshold
    }

    /// The configured cooldown.
    #[must_use]
    pub fn cooldown(&self) -> Duration {
        self.inner.cooldown
    }

    /// Records a successful run, resetting the failure count.
    pub fn record_success(&self, ctx: &CallbackContext, handler: &str) {
        let mut state = self.inner.state.lock();
        state.remove(&key(ctx, handler));
    }

    /// Records a failed run, opening the circuit at the threshold.
    pub fn record_failure(&self, ctx: &CallbackContext, handler: &str) {
        let now = self.inner.now();
        let key = key(ctx, handler);
        let mut state = self.inner.state.lock();
        let breaker = state.entry(key.clone()).or_default();
        breaker.failures = breaker.failures.saturating_add(1);
        if breaker.failures >= self.inner.threshold {
            breaker.opened_at = Some(now);
            tracing::warn!(
                target: LOG_TARGET,
                handler = %handler,
                event = %ctx.event.as_str(),
                failures = breaker.failures,
                key = %key,
                "circuit breaker opened"
            );
        }
    }

    /// Whether the circuit is currently open (ignoring cooldown expiry).
    #[must_use]
    pub fn is_open(&self, ctx: &CallbackContext, handler: &str) -> bool {
        self.inner
            .state
            .lock()
            .get(&key(ctx, handler))
            .is_some_and(|breaker| breaker.opened_at.is_some())
    }

    /// The recorded failure count (zero once the circuit closes).
    #[must_use]
    pub fn failures(&self, ctx: &CallbackContext, handler: &str) -> u32 {
        self.inner
            .state
            .lock()
            .get(&key(ctx, handler))
            .map_or(0, |breaker| breaker.failures)
    }
}

impl EventFilter for CircuitBreakerFilter {
    fn name(&self) -> &str {
        "circuit_breaker"
    }

    fn allows(&self, input: FilterInput<'_>) -> Result<FilterDecision> {
        let now = self.inner.now();
        let key = key(input.ctx, input.handler);
        let mut state = self.inner.state.lock();
        let Some(breaker) = state.get_mut(&key) else {
            return Ok(FilterDecision::Allow);
        };
        let Some(opened_at) = breaker.opened_at else {
            return Ok(FilterDecision::Allow);
        };
        if now.saturating_duration_since(opened_at) > self.inner.cooldown {
            state.remove(&key);
            tracing::debug!(
                target: LOG_TARGET,
                handler = %input.handler,
                key = %key,
                "circuit breaker closed after cooldown"
            );
            return Ok(FilterDecision::Allow);
        }
        Ok(FilterDecision::skip(format!(
            "circuit breaker open, retry after {} ms",
            self.inner.cooldown.as_millis()
        )))
    }

    fn feedback(&self) -> Option<&dyn FilterFeedback> {
        Some(self)
    }
}

impl FilterFeedback for CircuitBreakerFilter {
    fn record_success(&self, ctx: &CallbackContext, handler: &str) {
        Self::record_success(self, ctx, handler);
    }

    fn record_failure(&self, ctx: &CallbackContext, handler: &str) {
        Self::record_failure(self, ctx, handler);
    }
}

/// Rejects a handler whose typed payload fails `check`.
///
/// The rejection is an [`openjiuwen_common::error::Error::Validation`], so a
/// caller can match [`openjiuwen_common::error::ErrorKind::Validation`] rather
/// than parse a message.
pub struct ValidationFilter {
    name: String,
    check: Arc<dyn Fn(&CallbackContext) -> bool + Send + Sync>,
    message: String,
}

impl ValidationFilter {
    /// Validates with `check`, reporting `name`.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        check: impl Fn(&CallbackContext) -> bool + Send + Sync + 'static,
    ) -> Self {
        let name = name.into();
        let message = format!("argument validation failed for {name}");
        Self {
            name,
            check: Arc::new(check),
            message,
        }
    }

    /// Overrides the rejection message.
    #[must_use]
    pub fn with_message(mut self, message: impl Into<String>) -> Self {
        self.message = message.into();
        self
    }

    /// The rejection message.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl EventFilter for ValidationFilter {
    fn name(&self) -> &str {
        &self.name
    }

    fn allows(&self, input: FilterInput<'_>) -> Result<FilterDecision> {
        let check = Arc::clone(&self.check);
        if check(input.ctx) {
            return Ok(FilterDecision::Allow);
        }
        Ok(FilterDecision::reject(Error::validation(
            self.message.clone(),
        )))
    }
}

/// Logs every dispatch decision at `debug` and always allows.
///
/// Ports `LoggingFilter`. It never logs payload values (they may hold prompts).
#[derive(Debug, Clone)]
pub struct LoggingFilter {
    name: String,
}

impl LoggingFilter {
    /// A logging filter named `"logging"`.
    #[must_use]
    pub fn new() -> Self {
        Self::named("logging")
    }

    /// A logging filter with a custom name.
    #[must_use]
    pub fn named(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

impl Default for LoggingFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl EventFilter for LoggingFilter {
    fn name(&self) -> &str {
        &self.name
    }

    fn allows(&self, input: FilterInput<'_>) -> Result<FilterDecision> {
        tracing::debug!(
            target: LOG_TARGET,
            event = %input.ctx.event.as_str(),
            handler = %input.handler,
            namespace = %input.namespace,
            component = %input.ctx.component,
            "callback filter inspected a handler"
        );
        Ok(FilterDecision::Allow)
    }
}

/// Which check an [`AuthFilter`] performs.
enum AuthCheck {
    Role(String),
    Token(String),
    Predicate(String, Arc<dyn Fn(&CallbackContext) -> bool + Send + Sync>),
}

/// Authorises a handler from the context's metadata.
///
/// Ports `AuthFilter`: Python reads `user_role` from the callback kwargs and
/// defaults to `"guest"`; here the role lives in
/// `ctx.metadata["user_role"]` (the Rust context has no `**kwargs`), and the
/// token variant reads `ctx.metadata["auth_token"]`.
/// A denial is a soft [`FilterDecision::Skip`] by default (Python parity); call
/// [`AuthFilter::rejecting`] to fail closed with `Error::Security` instead.
pub struct AuthFilter {
    name: String,
    check: AuthCheck,
    rejecting: bool,
}

impl AuthFilter {
    /// Requires `ctx.metadata["user_role"] == role`.
    #[must_use]
    pub fn role(role: impl Into<String>) -> Self {
        Self {
            name: "auth".to_owned(),
            check: AuthCheck::Role(role.into()),
            rejecting: false,
        }
    }

    /// Requires `ctx.metadata["auth_token"] == token`.
    #[must_use]
    pub fn token(token: impl Into<String>) -> Self {
        Self {
            name: "auth".to_owned(),
            check: AuthCheck::Token(token.into()),
            rejecting: false,
        }
    }

    /// Requires `check` to accept the context.
    #[must_use]
    pub fn predicate(
        name: impl Into<String>,
        check: impl Fn(&CallbackContext) -> bool + Send + Sync + 'static,
    ) -> Self {
        let name = name.into();
        Self {
            name: name.clone(),
            check: AuthCheck::Predicate(name, Arc::new(check)),
            rejecting: false,
        }
    }

    /// Turns a denial into `Error::Security` instead of a soft skip.
    #[must_use]
    pub fn rejecting(mut self) -> Self {
        self.rejecting = true;
        self
    }

    fn denied(&self, reason: String) -> FilterDecision {
        if self.rejecting {
            FilterDecision::reject(Error::security(reason))
        } else {
            FilterDecision::skip(reason)
        }
    }
}

impl EventFilter for AuthFilter {
    fn name(&self) -> &str {
        &self.name
    }

    fn allows(&self, input: FilterInput<'_>) -> Result<FilterDecision> {
        match &self.check {
            AuthCheck::Role(required) => {
                let actual = input
                    .ctx
                    .metadata
                    .get("user_role")
                    .and_then(Value::as_str)
                    .unwrap_or("guest");
                if actual == required {
                    Ok(FilterDecision::Allow)
                } else {
                    Ok(self.denied(format!("unauthorized: requires {required}, got {actual}")))
                }
            }
            AuthCheck::Token(required) => {
                let actual = input
                    .ctx
                    .metadata
                    .get("auth_token")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                if actual == required {
                    Ok(FilterDecision::Allow)
                } else {
                    Ok(self.denied("unauthorized: missing or invalid auth_token".to_owned()))
                }
            }
            AuthCheck::Predicate(name, check) => {
                let check = Arc::clone(check);
                if check(input.ctx) {
                    Ok(FilterDecision::Allow)
                } else {
                    Ok(self.denied(format!("unauthorized: {name} check failed")))
                }
            }
        }
    }
}

/// One context rewrite, used by [`ParamModifyFilter`].
pub type ContextModifier = Arc<dyn Fn(&mut CallbackContext) -> Result<()> + Send + Sync>;

/// Rewrites the working context before the guarded handler runs.
///
/// The Rust form of `ParamModifyFilter`/`FilterAction::MODIFY`: the modifier
/// mutates [`FilterInput::ctx`] in place, so later filters and every later handler
/// observe the change. A modifier that fails aborts dispatch with its error
/// (Python skipped instead); wrap it in [`lenient`] for the Python behaviour.
pub struct ParamModifyFilter {
    name: String,
    modifier: ContextModifier,
}

impl ParamModifyFilter {
    /// Rewrites the context with `modifier`.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        modifier: impl Fn(&mut CallbackContext) -> Result<()> + Send + Sync + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            modifier: Arc::new(modifier),
        }
    }

    /// Rewrites [`CallbackContext::input`] with `map`, when it is present.
    #[must_use]
    pub fn input(map: impl Fn(&Value) -> Result<Value> + Send + Sync + 'static) -> Self {
        Self::new("param_modify", move |ctx| {
            if let Some(input) = &ctx.input {
                ctx.input = Some(map(input)?);
            }
            Ok(())
        })
    }
}

impl EventFilter for ParamModifyFilter {
    fn name(&self) -> &str {
        &self.name
    }

    fn allows(&self, input: FilterInput<'_>) -> Result<FilterDecision> {
        (self.modifier)(input.ctx)?;
        Ok(FilterDecision::Allow)
    }
}

/// Applies `inner` only when `condition` holds.
///
/// Ports `ConditionalFilter`. When the condition is false the default action is
/// [`FilterDecision::Skip`]; [`ConditionalFilter::allow_on_false`] switches it to
/// [`FilterDecision::Allow`] (useful to exempt a subset of handlers from a guard).
pub struct ConditionalFilter {
    name: String,
    condition: Arc<dyn Fn(&CallbackContext) -> bool + Send + Sync>,
    inner: Arc<dyn EventFilter>,
    allow_on_false: bool,
}

impl ConditionalFilter {
    /// Guards `inner` with `condition`.
    #[must_use]
    pub fn new(
        condition: impl Fn(&CallbackContext) -> bool + Send + Sync + 'static,
        inner: impl EventFilter + 'static,
    ) -> Self {
        let inner: Arc<dyn EventFilter> = Arc::new(inner);
        Self {
            name: format!("conditional({})", inner.name()),
            condition: Arc::new(condition),
            inner,
            allow_on_false: false,
        }
    }

    /// Allows the handler when the condition is false instead of skipping it.
    #[must_use]
    pub fn allow_on_false(mut self) -> Self {
        self.allow_on_false = true;
        self
    }
}

impl EventFilter for ConditionalFilter {
    fn name(&self) -> &str {
        &self.name
    }

    fn allows(&self, input: FilterInput<'_>) -> Result<FilterDecision> {
        let condition = Arc::clone(&self.condition);
        let inner = Arc::clone(&self.inner);
        if !condition(input.ctx) {
            return Ok(if self.allow_on_false {
                FilterDecision::Allow
            } else {
                FilterDecision::skip("condition not satisfied")
            });
        }
        inner.allows(input)
    }
}

/// Allows a handler whose context satisfies `predicate`, skipping it otherwise.
pub struct PredicateFilter {
    name: String,
    predicate: Arc<dyn Fn(&CallbackContext) -> bool + Send + Sync>,
    reason: String,
}

impl PredicateFilter {
    /// Allows when `predicate` accepts the context.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        predicate: impl Fn(&CallbackContext) -> bool + Send + Sync + 'static,
    ) -> Self {
        let name = name.into();
        let reason = format!("{name} predicate rejected the handler");
        Self {
            name,
            predicate: Arc::new(predicate),
            reason,
        }
    }

    /// Overrides the skip reason.
    #[must_use]
    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = reason.into();
        self
    }
}

impl EventFilter for PredicateFilter {
    fn name(&self) -> &str {
        &self.name
    }

    fn allows(&self, input: FilterInput<'_>) -> Result<FilterDecision> {
        let predicate = Arc::clone(&self.predicate);
        if predicate(input.ctx) {
            Ok(FilterDecision::Allow)
        } else {
            Ok(FilterDecision::skip(self.reason.clone()))
        }
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

/// The rate-limit / circuit-breaker map key: event plus handler identity.
fn key(ctx: &CallbackContext, handler: &str) -> String {
    format!("{}:{handler}", ctx.event.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;
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
                FilterDecision::skip("no")
            })
        };
        let mut ctx = context(CallbackEvent::AgentStarted);
        assert!(filter
            .allows(FilterInput {
                handler: "allowed",
                namespace: "default",
                ctx: &mut ctx,
            })
            .unwrap()
            .is_allow());
        assert!(filter
            .allows(FilterInput {
                handler: "other",
                namespace: "default",
                ctx: &mut ctx,
            })
            .unwrap()
            .is_skip());
    }

    #[test]
    fn an_empty_chain_allows_and_and_composes() {
        let mut ctx = context(CallbackEvent::AgentStarted);
        assert!(FilterChain::new()
            .allows(FilterInput {
                handler: "h",
                namespace: "default",
                ctx: &mut ctx,
            })
            .unwrap()
            .is_allow());
        let composed = NameFilter::new()
            .with_name("a")
            .and(NameFilter::new().with_prefix("b"));
        assert_eq!(composed.len(), 2);
        assert!(!composed.is_empty());
        assert_eq!(composed.name(), "filter_chain");
        assert_eq!(
            composed
                .filters()
                .iter()
                .map(EventFilter::name)
                .collect::<Vec<_>>(),
            ["name", "name"]
        );
    }

    #[test]
    fn a_filter_error_aborts_unless_lenient() {
        let failing = ClosureFilter::new("boom", |_: FilterInput<'_>| {
            Err(Error::validation("filter exploded"))
        });
        let mut ctx = context(CallbackEvent::AgentStarted);
        let error = failing
            .allows(FilterInput {
                handler: "h",
                namespace: "default",
                ctx: &mut ctx,
            })
            .unwrap_err();
        assert_eq!(error.to_string(), "validation: filter exploded");

        let decision = lenient(failing)
            .allows(FilterInput {
                handler: "h",
                namespace: "default",
                ctx: &mut ctx,
            })
            .unwrap();
        assert!(decision.is_skip());
        assert!(decision.reason().unwrap().contains("filter exploded"));
    }

    #[test]
    fn run_filters_reports_the_first_skip_and_keeps_mutations() {
        let filters: Vec<Arc<dyn EventFilter>> = vec![
            Arc::new(ParamModifyFilter::input(|value| {
                Ok(json!(format!("{}-rewritten", value.as_str().unwrap_or(""))))
            })),
            Arc::new(NameFilter::new().with_name("someone_else")),
        ];
        let mut ctx = context(CallbackEvent::ToolInvokeInput).with_input(json!("q"));
        let outcome = run_filters(&filters, "handler", "default", &mut ctx).unwrap();
        match outcome {
            FilterRunOutcome::Skip { filter, reason } => {
                assert_eq!(filter, "name");
                assert!(reason.contains("someone_else"));
            }
            other => panic!("expected a skip, got {other:?}"),
        }
        assert_eq!(ctx.input, Some(json!("q-rewritten")));
    }

    #[test]
    fn a_reject_surfaces_validation_for_the_caller() {
        let filter = ValidationFilter::new("payload", |ctx: &CallbackContext| ctx.input.is_some());
        let mut ctx = context(CallbackEvent::ToolInvokeInput);
        let decision = filter
            .allows(FilterInput {
                handler: "h",
                namespace: "default",
                ctx: &mut ctx,
            })
            .unwrap();
        match decision {
            FilterDecision::Reject(error) => {
                assert_eq!(
                    error.kind(),
                    openjiuwen_common::error::ErrorKind::Validation
                );
                assert!(error.to_string().contains("payload"));
            }
            other => panic!("expected a rejection, got {other:?}"),
        }
    }

    #[test]
    fn the_manual_clock_drives_the_rate_limit() {
        let clock = ManualClock::new();
        let filter = RateLimitFilter::new(2, Duration::from_secs(1)).with_clock(clock.clock());
        let ctx = context(CallbackEvent::ToolInvokeInput);
        let check = |filter: &RateLimitFilter, ctx: &CallbackContext| {
            let mut working = ctx.clone();
            filter
                .allows(FilterInput {
                    handler: "h",
                    namespace: "default",
                    ctx: &mut working,
                })
                .unwrap()
        };
        assert!(check(&filter, &ctx).is_allow());
        assert!(check(&filter, &ctx).is_allow());
        assert!(check(&filter, &ctx).is_skip());
        assert_eq!(filter.calls(&ctx, "h"), 2);
        clock.advance(Duration::from_secs(2));
        assert!(check(&filter, &ctx).is_allow());
    }

    #[test]
    fn the_circuit_breaker_opens_at_the_threshold_and_recovers() {
        let clock = ManualClock::new();
        let filter = CircuitBreakerFilter::new(2, Duration::from_secs(5)).with_clock(clock.clock());
        let ctx = context(CallbackEvent::AgentStarted);
        let check = |filter: &CircuitBreakerFilter| {
            let mut working = ctx.clone();
            filter
                .allows(FilterInput {
                    handler: "h",
                    namespace: "default",
                    ctx: &mut working,
                })
                .unwrap()
        };
        assert!(check(&filter).is_allow());
        filter.record_failure(&ctx, "h");
        assert!(check(&filter).is_allow());
        filter.record_failure(&ctx, "h");
        assert!(filter.is_open(&ctx, "h"));
        assert!(check(&filter).is_skip());
        clock.advance(Duration::from_secs(6));
        assert!(check(&filter).is_allow());
        assert!(!filter.is_open(&ctx, "h"));
        assert_eq!(filter.failures(&ctx, "h"), 0);
        assert_eq!(filter.failure_threshold(), 2);
        assert_eq!(filter.cooldown(), Duration::from_secs(5));
        assert!(filter.feedback().is_some());
    }
}
