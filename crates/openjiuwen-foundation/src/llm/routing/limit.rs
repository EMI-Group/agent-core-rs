//! Per-key TPM/RPM throttling for the routing model.
//!
//! A resolved route may carry a `tpm` (tokens per minute) and an `rpm` (requests
//! per minute) hint. This module enforces them: every route of a
//! [`RoutingModel`](super::model::RoutingModel) reserves budget before it calls its
//! provider and reconciles the reservation against the usage the provider reported.
//!
//! The rules, precisely:
//!
//! - **Window** — both budgets are measured over one rolling
//!   [`RATE_LIMIT_WINDOW`] (60 s) per key. A request is admitted when one more
//!   request still fits under `rpm` *and* the prompt-token estimate still fits
//!   under `tpm`; otherwise the limiter sleeps until enough of the window has
//!   expired.
//! - **Key** — a [`LimitKey`] is the resolved route's provider plus model name, so
//!   routes that resolved to the same provider/model share one provider-side
//!   budget and different providers never share one. Keys are never global.
//! - **Estimate then reconcile** — the prompt tokens are reserved from an estimate
//!   before the request ([`RateLimiter::reserve`]) and replaced by the provider's
//!   reported total once it is known ([`RateLimiter::reconcile`]). A failed request
//!   keeps its estimate, which is the conservative choice.
//! - **Bounded wait** — a caller passes the route/group `timeout_seconds` as
//!   `max_wait`; when the window would not free up in time the limiter returns
//!   [`Error::timeout`] naming TPM/RPM instead of waiting. An absent bound means
//!   "wait as long as the window needs".
//! - **No limits** — a [`Limits`] with both fields `None` never waits and never
//!   records state.
//!
//! State is per [`RateLimiter`] instance, hence per
//! [`RoutingModel`](super::model::RoutingModel) instance: nothing here is
//! process-global, and two routing models do not share a budget. Waiting uses the
//! injected [`Clock`], so a test installs a virtual clock instead of sleeping.
use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::future::BoxFuture;
use futures::FutureExt;
use openjiuwen_common::error::{Error, Result};
use parking_lot::Mutex;
use tracing::debug;

/// Tracing target of the routing layer.
const TRACING_TARGET: &str = "openjiuwen_foundation::llm::routing";

/// Length of the rolling window every budget is measured over: 60 seconds.
pub const RATE_LIMIT_WINDOW: Duration = Duration::from_secs(60);

/// A source of time and of sleeps, injected so throttling stays testable offline.
///
/// The routing model never calls [`Instant::now`] or tokio's timer directly: it
/// asks its clock. [`SystemClock`] is the default;
/// [`RoutingModel::with_clock`](super::model::RoutingModel::with_clock) installs
/// another one. A virtual clock only has to override [`Clock::sleep`], so that a
/// waiting budget resolves when the clock is advanced instead of when wall-clock
/// time passes.
pub trait Clock: Send + Sync {
    /// The clock's current instant.
    fn now(&self) -> Instant;

    /// Completes once `duration` has elapsed on this clock.
    ///
    /// The default implementation sleeps on the tokio timer. A virtual clock
    /// overrides it; because the returned future is `'static`, an implementation
    /// that needs shared state must hold it in an [`Arc`].
    fn sleep(&self, duration: Duration) -> BoxFuture<'static, ()> {
        tokio::time::sleep(duration).boxed()
    }
}

/// The real clock: [`Instant::now`] plus the tokio timer.
#[derive(Debug, Clone, Copy, Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> Instant {
        Instant::now()
    }
}

/// The identity a budget belongs to: one provider/model pair.
///
/// Built from the resolved route (provider plus model name), so two routes that
/// resolved to the same provider model share the provider-side budget they
/// describe, and routes of different providers never share one.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LimitKey {
    provider: String,
    model: String,
}

impl LimitKey {
    /// Builds a key from a provider name and a model name.
    #[must_use]
    pub fn new(provider: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            provider: provider.into(),
            model: model.into(),
        }
    }

    /// The provider half of the key.
    #[must_use]
    pub fn provider(&self) -> &str {
        &self.provider
    }

    /// The model half of the key.
    #[must_use]
    pub fn model(&self) -> &str {
        &self.model
    }
}

impl fmt::Display for LimitKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.provider, self.model)
    }
}

/// The budgets one [`LimitKey`] is throttled by; `None` means "no throttle".
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Limits {
    /// Requests per [`RATE_LIMIT_WINDOW`].
    pub rpm: Option<f64>,
    /// Tokens per [`RATE_LIMIT_WINDOW`].
    pub tpm: Option<f64>,
}

impl Limits {
    /// No throttle at all.
    pub const UNLIMITED: Self = Self {
        rpm: None,
        tpm: None,
    };

    /// Builds a budget from the two optional limits.
    #[must_use]
    pub const fn new(rpm: Option<f64>, tpm: Option<f64>) -> Self {
        Self { rpm, tpm }
    }

    /// Whether neither limit is set, in which case nothing is ever recorded.
    #[must_use]
    pub const fn is_unlimited(&self) -> bool {
        self.rpm.is_none() && self.tpm.is_none()
    }

    /// Rejects a limit that cannot describe a budget.
    ///
    /// A non-finite or non-positive `rpm`/`tpm` can never be satisfied, so it is a
    /// configuration error rather than a silent no-op.
    pub(crate) fn validate(&self, key: &LimitKey) -> Result<()> {
        for (name, value) in [("rpm", self.rpm), ("tpm", self.tpm)] {
            if let Some(value) = value {
                if !value.is_finite() || value <= 0.0 {
                    return Err(Error::validation(format!(
                        "route {key} declares {name}={value}, but a {name} rate limit must be a \
                         positive finite number"
                    )));
                }
            }
        }
        Ok(())
    }
}

/// What one key currently holds in its rolling window.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WindowUsage {
    /// Requests still inside the window.
    pub requests: u64,
    /// Tokens still inside the window.
    pub tokens: u64,
}

/// A budget reservation held by one in-flight request.
///
/// Returned by [`RateLimiter::reserve`] and handed back to
/// [`RateLimiter::reconcile`] once the provider reported its usage.
#[derive(Debug, Clone)]
pub struct Reservation {
    key: LimitKey,
    id: u64,
    reserved_tokens: u64,
}

impl Reservation {
    /// The key this reservation belongs to.
    #[must_use]
    pub fn key(&self) -> &LimitKey {
        &self.key
    }

    /// The token estimate that was reserved before the request.
    #[must_use]
    pub fn reserved_tokens(&self) -> u64 {
        self.reserved_tokens
    }

    /// The id of this reservation inside the window (`0` for an unlimited key,
    /// which records no state).
    #[must_use]
    pub fn id(&self) -> u64 {
        self.id
    }
}

/// One admitted request inside the rolling window.
#[derive(Debug, Clone)]
struct WindowEntry {
    /// When the request was admitted.
    at: Instant,
    /// Tokens charged to this request: the estimate until it is reconciled.
    tokens: u64,
    /// Reservation id, so a reconciliation can find its own entry.
    id: u64,
}

/// A rolling-window rate limiter with one budget per [`LimitKey`].
///
/// The state belongs to *this* instance (and therefore to one
/// [`RoutingModel`](super::model::RoutingModel)): there is no process-global
/// limiter and no shared registry. Cloning the limiter clones the handle (`Arc`
/// inside the routing model), so a routing model's budgets follow it.
pub struct RateLimiter {
    clock: Arc<dyn Clock>,
    state: Mutex<HashMap<LimitKey, VecDeque<WindowEntry>>>,
    next_id: AtomicU64,
}

impl RateLimiter {
    /// Builds an empty limiter that reads time from `clock`.
    #[must_use]
    pub fn new(clock: Arc<dyn Clock>) -> Self {
        Self {
            clock,
            state: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(1),
        }
    }

    /// Builds an empty limiter driven by the real clock.
    #[must_use]
    pub fn system() -> Self {
        Self::new(Arc::new(SystemClock))
    }

    /// The clock this limiter waits on.
    #[must_use]
    pub fn clock(&self) -> &Arc<dyn Clock> {
        &self.clock
    }

    /// Reserves one request plus `estimated_tokens` tokens for `key`.
    ///
    /// The call returns as soon as both budgets can admit the request, sleeping
    /// (on the injected clock) while the window has to expire first. `max_wait`
    /// bounds the total waiting time — the routing model passes the route/group
    /// `timeout_seconds` — and an exhausted bound is
    /// [`Error::timeout`] naming TPM/RPM. `None` means "wait as long as needed".
    ///
    /// An unlimited key returns a reservation without recording anything and a
    /// request whose estimate alone exceeds `tpm` is admitted once the window is
    /// empty (it could never fit, so waiting would never help).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when a limit is not a positive finite number,
    /// and [`Error::timeout`] when the required wait exceeds `max_wait`.
    pub async fn reserve(
        &self,
        key: &LimitKey,
        limits: Limits,
        estimated_tokens: u64,
        max_wait: Option<Duration>,
    ) -> Result<Reservation> {
        limits.validate(key)?;
        if limits.is_unlimited() {
            return Ok(Reservation {
                key: key.clone(),
                id: 0,
                reserved_tokens: estimated_tokens,
            });
        }
        let deadline = max_wait.and_then(|bound| self.clock.now().checked_add(bound));
        loop {
            let now = self.clock.now();
            let wait = {
                // The lock is never held across an `.await`: the wait is computed,
                // the reservation recorded, and the guard dropped inside this block.
                let mut state = self.state.lock();
                let entries = state.entry(key.clone()).or_default();
                prune_expired(entries, now);
                let wait = required_wait(entries, limits, estimated_tokens, now);
                if wait.is_zero() {
                    let id = self.next_id.fetch_add(1, Ordering::Relaxed);
                    entries.push_back(WindowEntry {
                        at: now,
                        tokens: estimated_tokens,
                        id,
                    });
                    return Ok(Reservation {
                        key: key.clone(),
                        id,
                        reserved_tokens: estimated_tokens,
                    });
                }
                wait
            };
            if let Some(deadline) = deadline {
                let remaining = deadline.saturating_duration_since(now);
                if wait > remaining {
                    return Err(Error::timeout(format!(
                        "TPM/RPM rate limit for {key} (rpm={}, tpm={}) would wait {wait:?}, \
                         beyond the {remaining:?} left of the routing timeout",
                        describe_limit(limits.rpm),
                        describe_limit(limits.tpm)
                    )));
                }
            }
            debug!(
                target: TRACING_TARGET,
                key = %key,
                wait_ms = u64::try_from(wait.as_millis()).unwrap_or(u64::MAX),
                "rate limit budget exhausted; waiting for the window"
            );
            self.clock.sleep(wait).await;
        }
    }

    /// Charges a finished request its actual token usage instead of the estimate.
    ///
    /// `actual_tokens` is the provider-reported total (prompt plus completion);
    /// `None` — or a reservation for an unlimited key, or one whose window entry
    /// already expired — leaves the state untouched.
    pub fn reconcile(&self, reservation: &Reservation, actual_tokens: Option<u64>) {
        let Some(actual) = actual_tokens else {
            return;
        };
        let mut state = self.state.lock();
        let Some(entries) = state.get_mut(reservation.key()) else {
            return;
        };
        if let Some(entry) = entries.iter_mut().find(|entry| entry.id == reservation.id) {
            entry.tokens = actual;
        }
    }

    /// What `key` currently holds in its rolling window.
    #[must_use]
    pub fn usage(&self, key: &LimitKey) -> WindowUsage {
        let now = self.clock.now();
        let mut state = self.state.lock();
        let Some(entries) = state.get_mut(key) else {
            return WindowUsage::default();
        };
        prune_expired(entries, now);
        WindowUsage {
            requests: u64::try_from(entries.len()).unwrap_or(u64::MAX),
            tokens: entries.iter().map(|entry| entry.tokens).sum(),
        }
    }
}

impl fmt::Debug for RateLimiter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RateLimiter")
            .field("keys", &self.state.lock().len())
            .finish_non_exhaustive()
    }
}

/// Drops every entry that left the rolling window.
fn prune_expired(entries: &mut VecDeque<WindowEntry>, now: Instant) {
    while let Some(front) = entries.front() {
        if !is_expired(front, now) {
            break;
        }
        entries.pop_front();
    }
}

/// Whether one entry has left the rolling window at `now`.
fn is_expired(entry: &WindowEntry, now: Instant) -> bool {
    entry
        .at
        .checked_add(RATE_LIMIT_WINDOW)
        .is_none_or(|expiry| expiry <= now)
}

/// How long until one entry leaves the rolling window.
fn time_until_expiry(entry: &WindowEntry, now: Instant) -> Duration {
    entry
        .at
        .checked_add(RATE_LIMIT_WINDOW)
        .map_or(Duration::ZERO, |expiry| {
            expiry.saturating_duration_since(now)
        })
}

/// How long the window has to shrink before `estimated_tokens` more can be admitted.
///
/// The two budgets are independent, so the result is the later of what `rpm` and
/// what `tpm` require. Zero means "admit now".
fn required_wait(
    entries: &VecDeque<WindowEntry>,
    limits: Limits,
    estimated_tokens: u64,
    now: Instant,
) -> Duration {
    let mut wait = Duration::ZERO;
    if let Some(rpm) = limits.rpm {
        let count = entries.len();
        // One more request must still fit under `rpm`; otherwise the oldest
        // entries have to expire first.
        if (count as f64) + 1.0 > rpm {
            let excess = ((count as f64) + 1.0 - rpm).ceil();
            let expire = usize::try_from(excess as u64)
                .unwrap_or(usize::MAX)
                .min(entries.len());
            if let Some(entry) = entries.get(expire.saturating_sub(1)) {
                wait = wait.max(time_until_expiry(entry, now));
            }
        }
    }
    if let Some(tpm) = limits.tpm {
        let used: u64 = entries.iter().map(|entry| entry.tokens).sum();
        let projected = used.saturating_add(estimated_tokens);
        // A request that cannot fit even in an empty window is admitted once the
        // window is empty: waiting longer would never satisfy it.
        if used > 0 && projected as f64 > tpm {
            let needed = ((projected as f64) - tpm).ceil() as u64;
            let mut freed = 0_u64;
            for entry in entries {
                freed = freed.saturating_add(entry.tokens);
                wait = wait.max(time_until_expiry(entry, now));
                if freed >= needed {
                    break;
                }
            }
        }
    }
    wait
}

/// Renders one optional limit for a diagnostic message.
fn describe_limit(limit: Option<f64>) -> String {
    limit.map_or_else(|| "none".to_owned(), |value| value.to_string())
}

/// A virtual clock for tests: time only moves when [`ManualClock::advance`] runs.
///
/// The limiter's `sleep` therefore never blocks the test: a waiting request stays
/// pending until the test advances the clock past its deadline, so a 60 s window
/// costs no wall-clock time.
#[cfg(test)]
#[derive(Debug, Clone, Default)]
pub(crate) struct ManualClock {
    inner: Arc<ManualClockInner>,
}

/// Shared state of a [`ManualClock`], so `sleep` can return a `'static` future.
#[cfg(test)]
#[derive(Debug)]
struct ManualClockInner {
    base: Instant,
    offset: Mutex<Duration>,
    notify: tokio::sync::Notify,
}

#[cfg(test)]
impl Default for ManualClockInner {
    fn default() -> Self {
        Self {
            base: Instant::now(),
            offset: Mutex::new(Duration::ZERO),
            notify: tokio::sync::Notify::new(),
        }
    }
}

#[cfg(test)]
impl ManualClock {
    /// A clock starting at "now" that does not move on its own.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Moves the clock forward and wakes every sleeping request.
    pub(crate) fn advance(&self, by: Duration) {
        *self.inner.offset.lock() += by;
        self.inner.notify.notify_waiters();
    }
}

#[cfg(test)]
impl Clock for ManualClock {
    fn now(&self) -> Instant {
        self.inner.base + *self.inner.offset.lock()
    }

    fn sleep(&self, duration: Duration) -> BoxFuture<'static, ()> {
        let inner = Arc::clone(&self.inner);
        Box::pin(async move {
            let deadline = inner.base + *inner.offset.lock() + duration;
            loop {
                if inner.base + *inner.offset.lock() >= deadline {
                    return;
                }
                // `enable` registers the waiter before the second check, so an
                // advance that happens in between cannot be lost.
                let notified = inner.notify.notified();
                tokio::pin!(notified);
                notified.as_mut().enable();
                if inner.base + *inner.offset.lock() >= deadline {
                    return;
                }
                notified.await;
            }
        })
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests of the limiter with a virtual clock (no wall-clock waits).

    use super::*;

    /// A limiter over a manual clock, plus the clock itself.
    fn limiter() -> (RateLimiter, ManualClock) {
        let clock = ManualClock::new();
        (RateLimiter::new(Arc::new(clock.clone())), clock)
    }

    /// The key the tests share.
    fn key() -> LimitKey {
        LimitKey::new("openai", "gpt-4o-mini")
    }

    #[tokio::test]
    async fn unlimited_limits_never_wait_and_record_nothing() {
        let (limiter, _clock) = limiter();
        for _ in 0..100 {
            limiter
                .reserve(&key(), Limits::UNLIMITED, 10_000, None)
                .await
                .expect("an unlimited key always admits");
        }
        assert_eq!(limiter.usage(&key()), WindowUsage::default());
    }

    #[tokio::test]
    async fn rpm_admits_the_budget_then_waits_for_the_window() {
        let (limiter, clock) = limiter();
        let limits = Limits::new(Some(2.0), None);
        limiter.reserve(&key(), limits, 1, None).await.unwrap();
        limiter.reserve(&key(), limits, 1, None).await.unwrap();
        assert_eq!(limiter.usage(&key()).requests, 2);

        // The third request has to wait for the window; it stays pending until the
        // clock advances past the 60 s marker. `join!` polls the reservation first,
        // so the sleep is registered before the clock moves.
        let pending_key = key();
        let pending = limiter.reserve(&pending_key, limits, 1, None);
        let (reservation, ()) = tokio::join!(pending, async {
            clock.advance(RATE_LIMIT_WINDOW);
        });
        reservation.expect("the window freed up");
        assert_eq!(
            limiter.usage(&key()).requests,
            1,
            "the two expired requests left the window"
        );
    }

    #[tokio::test]
    async fn rpm_bound_reports_a_typed_timeout_naming_rpm() {
        let (limiter, _clock) = limiter();
        let limits = Limits::new(Some(2.0), None);
        limiter.reserve(&key(), limits, 1, None).await.unwrap();
        limiter.reserve(&key(), limits, 1, None).await.unwrap();

        let error = limiter
            .reserve(&key(), limits, 1, Some(Duration::from_secs(5)))
            .await
            .expect_err("the window cannot free up within the bound");
        assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::Timeout);
        let message = error.message().to_string();
        assert!(message.contains("TPM/RPM"), "{message}");
        assert!(message.contains("rpm=2"), "{message}");
    }

    #[tokio::test]
    async fn tpm_blocks_on_the_reserved_estimate_and_reconciles_the_actual_usage() {
        let (limiter, _clock) = limiter();
        let limits = Limits::new(None, Some(10.0));

        // Reserve 6 tokens, then report that only 2 were used.
        let first = limiter.reserve(&key(), limits, 6, None).await.unwrap();
        assert_eq!(limiter.usage(&key()).tokens, 6);
        limiter.reconcile(&first, Some(2));
        assert_eq!(limiter.usage(&key()).tokens, 2);

        // A second request fits thanks to the reconciliation: 2 + 6 <= 10.
        let second = limiter.reserve(&key(), limits, 6, None).await.unwrap();
        limiter.reconcile(&second, Some(2));

        // Without reconciliation the window would hold 6 + 6 = 12 > 10, so a third
        // request would have to wait; it does not.
        let third = limiter.reserve(&key(), limits, 6, None).await;
        assert!(third.is_ok(), "the window holds 4 tokens, not 12");
    }

    #[tokio::test]
    async fn tpm_reservation_blocks_a_request_that_would_exceed_the_window() {
        let (limiter, clock) = limiter();
        let limits = Limits::new(None, Some(10.0));
        let first = limiter.reserve(&key(), limits, 6, None).await.unwrap();
        limiter.reconcile(&first, Some(10));
        assert_eq!(limiter.usage(&key()).tokens, 10);

        let error = limiter
            .reserve(&key(), limits, 6, Some(Duration::from_secs(1)))
            .await
            .expect_err("10 + 6 > 10 for the whole window");
        assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::Timeout);
        let message = error.message().to_string();
        assert!(message.contains("tpm=10"), "{message}");

        // After the window expires the same reservation is admitted.
        clock.advance(RATE_LIMIT_WINDOW);
        limiter
            .reserve(&key(), limits, 6, Some(Duration::from_secs(1)))
            .await
            .expect("the window is empty again");
    }

    #[tokio::test]
    async fn budgets_are_isolated_per_key() {
        let (limiter, _clock) = limiter();
        let limits = Limits::new(Some(1.0), Some(10.0));
        let first = LimitKey::new("openai", "gpt-4o-mini");
        let second = LimitKey::new("anthropic", "claude-3-5-sonnet");

        limiter.reserve(&first, limits, 5, None).await.unwrap();
        // The other provider has its own (still untouched) budget.
        limiter.reserve(&second, limits, 5, None).await.unwrap();
        assert_eq!(limiter.usage(&first).requests, 1);
        assert_eq!(limiter.usage(&second).requests, 1);

        let error = limiter
            .reserve(&first, limits, 5, Some(Duration::from_secs(1)))
            .await
            .expect_err("the first key's single-request budget is spent");
        assert!(error.message().contains(&first.to_string()), "{error}");
    }

    #[tokio::test]
    async fn a_request_larger_than_the_whole_budget_is_admitted_once_the_window_is_empty() {
        let (limiter, _clock) = limiter();
        let limits = Limits::new(None, Some(4.0));
        // 10 > 4 can never fit; waiting for the window would never help.
        limiter.reserve(&key(), limits, 10, None).await.unwrap();
        assert_eq!(limiter.usage(&key()).tokens, 10);
    }

    #[tokio::test]
    async fn invalid_limits_are_a_validation_error() {
        let (limiter, _clock) = limiter();
        for limits in [
            Limits::new(Some(0.0), None),
            Limits::new(None, Some(-1.0)),
            Limits::new(None, Some(f64::NAN)),
            Limits::new(Some(f64::INFINITY), None),
        ] {
            let error = limiter
                .reserve(&key(), limits, 1, None)
                .await
                .expect_err("a non-positive or non-finite budget is rejected");
            assert_eq!(
                error.kind(),
                openjiuwen_common::error::ErrorKind::Validation,
                "{error}"
            );
            assert!(error.message().contains(&key().to_string()), "{error}");
        }
    }

    #[tokio::test]
    async fn fractional_rpm_waits_for_two_entries_to_expire() {
        let (limiter, clock) = limiter();
        let limits = Limits::new(Some(2.5), None);
        for _ in 0..2 {
            limiter.reserve(&key(), limits, 1, None).await.unwrap();
        }
        assert!(
            limiter
                .reserve(&key(), limits, 1, Some(Duration::ZERO))
                .await
                .is_err(),
            "3 > 2.5 requests needs the window to shrink"
        );
        clock.advance(RATE_LIMIT_WINDOW);
        limiter
            .reserve(&key(), limits, 1, None)
            .await
            .expect("the window expired");
    }
}
