//! The built-in filter family: name, kind, session, rate limit, circuit
//! breaker, validation, logging, auth, param modify, conditional, predicate.
//!
//! Ports the `EventFilter` implementations of `callback/filters.py`. Each one is
//! a small `Send + Sync` value that inspects the handler identity in
//! `FilterInput` and/or the typed payload in `CallbackContext`, returning
//! `FilterDecision::Allow`, a soft skip, or a rejection error.
//!
//! Time-dependent filters take an injectable clock so tests advance a
//! `ManualClock` instead of sleeping; the default is the system clock.
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
use super::super::context::CallbackContext;
use super::super::event::{CallbackEvent, EventDomain};
use super::super::LOG_TARGET;
use super::{Clock, EventFilter, FilterDecision, FilterFeedback, FilterInput};
use openjiuwen_common::error::{Error, Result};
use parking_lot::Mutex;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

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

    /// Replaces the clock (tests use
    /// [`ManualClock::clock`](crate::callback::filters::ManualClock::clock)).
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

    /// Replaces the clock (tests use
    /// [`ManualClock::clock`](crate::callback::filters::ManualClock::clock)).
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
/// (Python skipped instead); wrap it in
/// [`lenient()`](crate::callback::filters::lenient) for the Python behaviour.
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

/// The rate-limit / circuit-breaker map key: event plus handler identity.
fn key(ctx: &CallbackContext, handler: &str) -> String {
    format!("{}:{handler}", ctx.event.as_str())
}

#[cfg(test)]
mod tests {
    use super::super::ManualClock;
    use super::*;
    fn context(event: CallbackEvent) -> CallbackContext {
        CallbackContext::new(event)
    }
    #[test]
    fn the_name_filter_matches_names_prefixes_and_namespaces() {
        let filter = NameFilter::named(["worker"]);
        let mut ctx = context(CallbackEvent::AgentInvokeInput);
        let allow = filter
            .allows(FilterInput {
                handler: "worker",
                namespace: "default",
                ctx: &mut ctx,
            })
            .expect("allow");
        assert!(allow.is_allow());
        let skip = filter
            .allows(FilterInput {
                handler: "other",
                namespace: "default",
                ctx: &mut ctx,
            })
            .expect("skip");
        match skip {
            FilterDecision::Skip { reason } => assert!(reason.contains("worker")),
            other => panic!("expected a skip, got {other:?}"),
        }
    }
    #[test]
    fn the_manual_clock_drives_the_rate_limit() {
        let clock = ManualClock::new();
        let filter = RateLimitFilter::new(2, Duration::from_secs(1)).with_clock(clock.clock());
        let mut ctx = context(CallbackEvent::ToolInvokeInput);
        let check = |ctx: &mut CallbackContext| {
            filter
                .allows(FilterInput {
                    handler: "tool",
                    namespace: "default",
                    ctx,
                })
                .expect("checked")
        };
        assert!(check(&mut ctx).is_allow());
        assert!(check(&mut ctx).is_allow());
        assert!(check(&mut ctx).is_skip());
        clock.advance(Duration::from_secs(2));
        assert!(check(&mut ctx).is_allow());
    }
    #[test]
    fn the_circuit_breaker_opens_at_the_threshold_and_recovers() {
        let clock = ManualClock::new();
        let breaker =
            CircuitBreakerFilter::new(2, Duration::from_secs(5)).with_clock(clock.clock());
        let mut ctx = context(CallbackEvent::ToolInvokeInput);
        assert!(breaker
            .allows(FilterInput {
                handler: "tool",
                namespace: "default",
                ctx: &mut ctx,
            })
            .expect("closed")
            .is_allow());
        breaker.record_failure(&ctx, "tool");
        breaker.record_failure(&ctx, "tool");
        assert!(breaker.is_open(&ctx, "tool"));
        assert!(breaker
            .allows(FilterInput {
                handler: "tool",
                namespace: "default",
                ctx: &mut ctx,
            })
            .expect("open")
            .is_skip());
        clock.advance(Duration::from_secs(6));
        assert!(breaker
            .allows(FilterInput {
                handler: "tool",
                namespace: "default",
                ctx: &mut ctx,
            })
            .expect("half open")
            .is_allow());
        breaker.record_success(&ctx, "tool");
        assert!(!breaker.is_open(&ctx, "tool"));
    }
    #[test]
    fn the_validation_filter_rejects_with_its_message() {
        let filter = ValidationFilter::new("payload", |ctx: &CallbackContext| {
            ctx.input.as_ref().is_some_and(Value::is_object)
        })
        .with_message("payload must be an object");
        let mut ctx = context(CallbackEvent::LlmInvokeInput);
        ctx.input = Some(Value::from(3));
        let decision = filter
            .allows(FilterInput {
                handler: "llm",
                namespace: "default",
                ctx: &mut ctx,
            })
            .expect("decides");
        match decision {
            FilterDecision::Reject(error) => {
                assert!(error.to_string().contains("payload must be an object"));
            }
            other => panic!("expected a rejection, got {other:?}"),
        }
    }
}
