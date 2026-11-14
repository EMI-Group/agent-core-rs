//! [`DreamOrchestrator`]: the idle/backoff loop that drives dream sweeps.
//!
//! [`DreamPlanner`] executes one sweep; somebody has to decide *when* a sweep happens.
//! In Python that is the `DreamingOrchestrator` (settle delay, minimum interval, then a
//! background task). Here it is an explicit, injectable value:
//!
//! - [`crate::dreaming::DreamOrchestratorConfig`] holds the schedule as plain integers
//!   (milliseconds), so it round-trips through JSON exactly like the rest of the crate and
//!   a caller can persist it;
//! - [`crate::dreaming::DreamOrchestrator`] owns one planner plus a status snapshot, and
//!   exposes [`crate::dreaming::DreamOrchestrator::tick`] —
//!   one iteration with an injected clock — for callers that own their own timer;
//! - [`crate::dreaming::DreamOrchestrator::run_until_cancelled`] is the loop itself, driven
//!   by a [`CancellationToken`] the caller passes in.
//!
//! Nothing here is a process global and nothing here starts a task: the crate never
//! spawns, never registers, and never sleeps on the caller's behalf. A test, a CLI, or an
//! embedding application decides whether a loop exists at all, and passes the clock in.
//!
//! ## The schedule
//!
//! The first sweep waits `initial_delay_ms` (Python parity: 120 s, so storage can finish
//! opening). After a tick the loop waits `next_interval_ms`, which is
//! [`crate::dreaming::DreamOrchestratorConfig::interval_for`] over the number of
//! *consecutive empty sweeps*: a sweep that claims nothing multiplies the base interval by
//! `backoff_multiplier`, one step per empty sweep, up to `max_backoff_steps` steps and
//! never beyond `max_interval_ms`. A sweep that claims at least one job resets the count,
//! so steady work runs at `poll_interval_ms` and an idle queue settles at the cap.
//!
//! The backoff is deliberately small and bounded: it is a poll schedule, not a distributed
//! scheduler, and waking up at the cap forever costs almost nothing.
//!
//! ## Failure policy
//!
//! One tick is one sweep plus, optionally, one retention purge. A sweep error propagates to
//! the caller (and is recorded in [`crate::dreaming::DreamOrchestratorStatus::last_error`]),
//! while a *retention* error never fails the tick: retention is housekeeping, so it is
//! logged and recorded and the sweep's result still stands. The loop turns a non-cancellation
//! tick error into a log line and keeps looping — the same thing Python's orchestrator does
//! when its background task raises — and only cancellation ends it.
//!
//! ## Leases
//!
//! When `lease` is set, every sweep goes through the leased planner path, so two processes
//! sharing one repository cannot consolidate the same job: the claim writes a lease, the
//! sweep renews it before recording the transition, and it is released afterwards. The
//! orchestrator uses its own `repository` handle only for retention and as the lease store;
//! callers should pass the planner's repository so both halves see the same queue.
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use openjiuwen_common::error::{Error, Result};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::record::CallOptions;

use super::lease::DreamLeaseOptions;
use super::outcome::DreamRunSummary;
use super::planner::DreamPlanner;
use super::repository::DreamRepository;
use super::retention::{DreamRetention, RetentionReport};

#[cfg(test)]
mod tests;

/// The tracing target for every event this module emits.
const TARGET: &str = "openjiuwen_memory::dreaming::orchestrator";

/// The largest `max_backoff_steps` a configuration may ask for.
///
/// Four steps already reach `16x` the base interval; the cap keeps a typo from turning the
/// delay into a number of milliseconds that cannot be represented.
const MAX_BACKOFF_STEPS: u32 = 16;

/// The schedule of a [`DreamOrchestrator`], as serializable integers.
///
/// Durations are milliseconds rather than [`std::time::Duration`] so the configuration is
/// plain JSON: `{"poll_interval_ms": 60000, ...}` serializes, deserializes, and survives a
/// round trip through a config file. Every field is public and has a documented default;
/// [`DreamOrchestratorConfig::validate`] rejects a schedule that would busy-spin, never
/// fire, or overflow.
///
/// The defaults follow Python's `DreamingOrchestrator`: enable dreaming, wake every 60 s,
/// wait 120 s before the first sweep, and settle at a 10-minute cap when the queue is
/// empty.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct DreamOrchestratorConfig {
    /// Whether the orchestrator does anything at all; default `true`.
    ///
    /// A disabled orchestrator claims nothing, purges nothing, and leaves
    /// [`DreamOrchestratorStatus`] untouched, so jobs stay [`crate::dreaming::DreamState::Pending`]
    /// until somebody enables it.
    pub enabled: bool,
    /// The delay between sweeps while the queue has work; default `60_000` (60 s).
    ///
    /// This is the base of the backoff and the floor of every interval: an empty sweep can
    /// only make the wait longer. Python's `min_interval` floor of 60 s is this value.
    pub poll_interval_ms: u64,
    /// The delay before the *first* sweep; default `120_000` (120 s).
    ///
    /// Matches Python's settle delay, which gives an application time to open its storage
    /// before consolidation competes with it. Tests set it to `0`.
    pub initial_delay_ms: u64,
    /// The largest interval the backoff may reach; default `600_000` (10 min).
    ///
    /// Also the ceiling for [`DreamOrchestratorConfig::initial_delay_ms`]: a settle delay
    /// longer than the cap would be the only part of the schedule that ignores it.
    pub max_interval_ms: u64,
    /// The factor applied per consecutive empty sweep; default `2.0`.
    ///
    /// Must be finite and at least `1.0`: a multiplier below `1.0` would shrink the
    /// interval on an idle queue, which inverts the point of the backoff.
    pub backoff_multiplier: f32,
    /// How many backoff steps are applied at most; default `4`.
    ///
    /// With the default multiplier that caps the growth at `16x` the base interval. The
    /// value is capped by validation (see [`DreamOrchestratorConfig::validate`]).
    pub max_backoff_steps: u32,
    /// Whether each tick also applies [`DreamOrchestratorConfig::retention`]; default
    /// `true`.
    ///
    /// Retention is skipped entirely when this is `false` or when `retention` is `None`.
    pub run_retention_on_tick: bool,
    /// The lease every sweep claims through; default `None` (no leasing).
    ///
    /// `Some` routes each sweep through the leased planner path, which makes cross-process
    /// workers safe on one repository. The orchestrator never renews a lease in the
    /// background: the planner renews once per job and this loop does not shorten a TTL.
    ///
    /// This is the one field JSON does not carry: [`DreamLeaseOptions`] holds a
    /// [`chrono::Duration`] and a [`std::time::Duration`], and the lease module keeps it a
    /// plain value rather than a wire type, so the field is skipped by serde and defaults
    /// to `None` on deserialize. Set it in code with
    /// [`DreamOrchestratorConfig::with_lease`].
    #[serde(skip)]
    pub lease: Option<DreamLeaseOptions>,
    /// The retention policy applied on each tick; default `None` (no purge).
    ///
    /// A policy that is present but disabled
    /// ([`DreamRetention::is_enabled`] is `false`) is treated like `None`: nothing is
    /// purged and the status keeps its previous report.
    pub retention: Option<DreamRetention>,
}

impl Default for DreamOrchestratorConfig {
    /// The documented defaults: enabled, 60 s interval, 120 s settle delay, 10-minute cap,
    /// `2.0x` backoff over at most four steps, retention on tick, and neither a lease nor
    /// a retention policy configured.
    fn default() -> Self {
        Self {
            enabled: true,
            poll_interval_ms: 60_000,
            initial_delay_ms: 120_000,
            max_interval_ms: 600_000,
            backoff_multiplier: 2.0,
            max_backoff_steps: 4,
            run_retention_on_tick: true,
            lease: None,
            retention: None,
        }
    }
}

impl DreamOrchestratorConfig {
    /// Returns the configuration with `enabled` as its enable switch.
    #[must_use]
    pub const fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Returns the configuration with `poll_interval_ms` as its base interval.
    #[must_use]
    pub const fn with_poll_interval_ms(mut self, poll_interval_ms: u64) -> Self {
        self.poll_interval_ms = poll_interval_ms;
        self
    }

    /// Returns the configuration with `initial_delay_ms` as its settle delay.
    #[must_use]
    pub const fn with_initial_delay_ms(mut self, initial_delay_ms: u64) -> Self {
        self.initial_delay_ms = initial_delay_ms;
        self
    }

    /// Returns the configuration with `max_interval_ms` as its interval ceiling.
    #[must_use]
    pub const fn with_max_interval_ms(mut self, max_interval_ms: u64) -> Self {
        self.max_interval_ms = max_interval_ms;
        self
    }

    /// Returns the configuration with `backoff_multiplier` as its growth factor.
    #[must_use]
    pub const fn with_backoff_multiplier(mut self, backoff_multiplier: f32) -> Self {
        self.backoff_multiplier = backoff_multiplier;
        self
    }

    /// Returns the configuration with `max_backoff_steps` as its step cap.
    #[must_use]
    pub const fn with_max_backoff_steps(mut self, max_backoff_steps: u32) -> Self {
        self.max_backoff_steps = max_backoff_steps;
        self
    }

    /// Returns the configuration with `run_retention_on_tick` as its retention switch.
    #[must_use]
    pub const fn with_run_retention_on_tick(mut self, run_retention_on_tick: bool) -> Self {
        self.run_retention_on_tick = run_retention_on_tick;
        self
    }

    /// Returns the configuration claiming through `lease`.
    #[must_use]
    pub fn with_lease(mut self, lease: DreamLeaseOptions) -> Self {
        self.lease = Some(lease);
        self
    }

    /// Returns the configuration purging through `retention` on every tick.
    #[must_use]
    pub fn with_retention(mut self, retention: DreamRetention) -> Self {
        self.retention = Some(retention);
        self
    }

    /// Returns [`DreamOrchestratorConfig::poll_interval_ms`] as a [`Duration`].
    #[must_use]
    pub const fn poll_interval(&self) -> Duration {
        Duration::from_millis(self.poll_interval_ms)
    }

    /// Returns [`DreamOrchestratorConfig::initial_delay_ms`] as a [`Duration`].
    #[must_use]
    pub const fn initial_delay(&self) -> Duration {
        Duration::from_millis(self.initial_delay_ms)
    }

    /// Returns [`DreamOrchestratorConfig::max_interval_ms`] as a [`Duration`].
    #[must_use]
    pub const fn max_interval(&self) -> Duration {
        Duration::from_millis(self.max_interval_ms)
    }

    /// The interval to wait after a sweep, given how many *consecutive* empty sweeps have
    /// happened.
    ///
    /// The documented formula, computed in integer milliseconds:
    ///
    /// ```text
    /// steps    = min(empty_sweeps, max_backoff_steps)
    /// interval = min(poll_interval_ms * backoff_multiplier^steps, max_interval_ms)
    /// ```
    ///
    /// The product is rounded to the nearest millisecond and clamped to
    /// [`DreamOrchestratorConfig::max_interval_ms`], so `interval_for(0)` is exactly
    /// [`DreamOrchestratorConfig::poll_interval`] and a large `empty_sweeps` value can
    /// neither overflow nor escape the cap. With the defaults the sequence is 60 s, 120 s,
    /// 240 s, 480 s, then 600 s forever.
    #[must_use]
    pub fn interval_for(&self, empty_sweeps: u32) -> Duration {
        let steps = empty_sweeps.min(self.max_backoff_steps);
        let exponent = i32::try_from(steps).unwrap_or(i32::MAX);
        let factor = f64::from(self.backoff_multiplier).powi(exponent);
        let base = self.poll_interval_ms as f64;
        let scaled = base * factor;
        // A non-finite product (a hand-built config with an infinite multiplier or a huge
        // `poll_interval_ms`) and anything past the cap both land on the cap; the cast
        // saturates, so no intermediate value can wrap.
        let millis = if scaled.is_finite() && scaled < self.max_interval_ms as f64 {
            scaled.round() as u64
        } else {
            self.max_interval_ms
        };
        Duration::from_millis(millis.min(self.max_interval_ms))
    }

    /// Validates the schedule, naming the offending field in every error.
    ///
    /// The rules:
    ///
    /// - [`DreamOrchestratorConfig::poll_interval_ms`] is greater than zero, so the loop
    ///   cannot busy-spin;
    /// - [`DreamOrchestratorConfig::max_interval_ms`] is at least
    ///   [`DreamOrchestratorConfig::poll_interval_ms`] (the cap can never shorten the base
    ///   interval);
    /// - [`DreamOrchestratorConfig::initial_delay_ms`] does not exceed
    ///   [`DreamOrchestratorConfig::max_interval_ms`] (the settle delay obeys the cap);
    /// - [`DreamOrchestratorConfig::backoff_multiplier`] is finite and at least `1.0`;
    /// - [`DreamOrchestratorConfig::max_backoff_steps`] is at most `16`;
    /// - a present [`DreamOrchestratorConfig::lease`] passes
    ///   [`DreamLeaseOptions::validate`], and a present
    ///   [`DreamOrchestratorConfig::retention`] passes [`DreamRetention::validate`].
    pub fn validate(&self) -> Result<()> {
        if self.poll_interval_ms == 0 {
            return Err(Error::validation(
                "dream orchestrator `poll_interval_ms` must be greater than 0",
            ));
        }
        if self.max_interval_ms < self.poll_interval_ms {
            return Err(Error::validation(format!(
                "dream orchestrator `max_interval_ms` ({}) must not be shorter than \
                 `poll_interval_ms` ({})",
                self.max_interval_ms, self.poll_interval_ms
            )));
        }
        if self.initial_delay_ms > self.max_interval_ms {
            return Err(Error::validation(format!(
                "dream orchestrator `initial_delay_ms` ({}) must not exceed \
                 `max_interval_ms` ({})",
                self.initial_delay_ms, self.max_interval_ms
            )));
        }
        if !self.backoff_multiplier.is_finite() || self.backoff_multiplier < 1.0 {
            return Err(Error::validation(format!(
                "dream orchestrator `backoff_multiplier` ({}) must be finite and at least 1.0",
                self.backoff_multiplier
            )));
        }
        if self.max_backoff_steps > MAX_BACKOFF_STEPS {
            return Err(Error::validation(format!(
                "dream orchestrator `max_backoff_steps` ({}) must be at most {MAX_BACKOFF_STEPS}",
                self.max_backoff_steps
            )));
        }
        if let Some(lease) = &self.lease {
            lease.validate()?;
        }
        if let Some(retention) = &self.retention {
            retention.validate()?;
        }
        Ok(())
    }
}

/// What the orchestrator has done so far, and what it will wait next.
///
/// This is an observable snapshot, not control state: [`DreamOrchestrator::status`] returns
/// a clone and callers never write it. JSON is `snake_case`, every field is optional on
/// input (missing fields default), and unknown fields are ignored, so a status can be
/// logged or shipped as-is.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct DreamOrchestratorStatus {
    /// Ticks that reached a sweep (or that failed while sweeping); default `0`.
    ///
    /// A tick on a disabled orchestrator does not count.
    pub ticks: u64,
    /// Sweeps in a row that claimed nothing; default `0`; resets on any recorded attempt.
    ///
    /// This is the backoff input: [`DreamOrchestratorConfig::interval_for`] is applied to
    /// it to produce [`DreamOrchestratorStatus::next_interval_ms`].
    pub consecutive_empty_sweeps: u32,
    /// How long the loop waits *after* the last tick before the next one.
    ///
    /// It is `0` before the first tick (nothing has been scheduled yet); the loop's actual
    /// pre-first-sweep wait is [`DreamOrchestratorConfig::initial_delay`], which is not
    /// reflected here.
    pub next_interval_ms: u64,
    /// The clock of the last tick; default `None`.
    pub last_tick_at: Option<DateTime<Utc>>,
    /// The last sweep's summary; default `None`.
    pub last_summary: Option<DreamRunSummary>,
    /// The last retention report, when a tick actually purged something; default `None`.
    ///
    /// Retention that is switched off, unconfigured, or disabled as a policy leaves this
    /// untouched; a retention *failure* clears it, so a stale report never reads as a
    /// fresh one.
    pub last_retention: Option<RetentionReport>,
    /// The last error text, from a failed sweep or a failed retention pass; default `None`.
    ///
    /// A successful retention pass clears it; a failed retention pass sets it without
    /// failing the tick. Because this is a `String` and not an [`Error`], the status stays
    /// serializable.
    pub last_error: Option<String>,
}

/// One tick's retention outcome, before it is folded into the status.
enum RetentionStep {
    /// Retention did not run (switched off, unconfigured, or a disabled policy).
    Skipped,
    /// Retention ran and removed what the report says.
    Purged(RetentionReport),
    /// Retention failed; the message goes to
    /// [`DreamOrchestratorStatus::last_error`] and the tick still succeeds.
    Failed(String),
}

/// The caller-driven idle/backoff loop over one [`DreamPlanner`].
///
/// The orchestrator is a value, not a service: it owns a planner, a repository handle, a
/// schedule, and a status snapshot, and it never spawns a task or installs anything
/// globally. A caller that wants background consolidation owns the loop — usually by
/// awaiting [`DreamOrchestrator::run_until_cancelled`] on a task it created itself — and a
/// caller that already has a timer calls [`DreamOrchestrator::tick`] instead.
///
/// `repository` should be the planner's repository (the same `Arc`), because the
/// orchestrator uses that handle for retention and as the lease store; the planner keeps
/// its own handle, which is the queue the sweeps claim from, and this type never assumes
/// the two differ.
pub struct DreamOrchestrator {
    /// The sweep implementation.
    planner: Arc<DreamPlanner>,
    /// The queue used for retention and leasing.
    repository: Arc<dyn DreamRepository>,
    /// The schedule and its switches.
    config: DreamOrchestratorConfig,
    /// The observable snapshot, updated once per tick.
    state: Mutex<DreamOrchestratorStatus>,
}

impl fmt::Debug for DreamOrchestrator {
    /// Prints the schedule and the current status.
    ///
    /// The planner and repository are `Arc`s of trait objects and concrete types without a
    /// shared printable form, so they are omitted (the struct is finished
    /// non-exhaustively) rather than printed as pointer addresses.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DreamOrchestrator")
            .field("config", &self.config)
            .field("status", &*self.state.lock())
            .finish_non_exhaustive()
    }
}

impl DreamOrchestrator {
    /// Builds an orchestrator over `planner`, `repository`, and `config`.
    ///
    /// The configuration is validated here ([`DreamOrchestratorConfig::validate`]), so an
    /// unusable schedule fails at construction instead of mid-loop. `repository` should be
    /// the planner's repository: it is used for retention and as the lease store, while the
    /// planner keeps the queue handle its sweeps claim from.
    pub fn new(
        planner: Arc<DreamPlanner>,
        repository: Arc<dyn DreamRepository>,
        config: DreamOrchestratorConfig,
    ) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            planner,
            repository,
            config,
            state: Mutex::new(DreamOrchestratorStatus::default()),
        })
    }

    /// The schedule this orchestrator follows.
    #[must_use]
    pub fn config(&self) -> &DreamOrchestratorConfig {
        &self.config
    }

    /// The planner every tick sweeps through.
    #[must_use]
    pub fn planner(&self) -> &Arc<DreamPlanner> {
        &self.planner
    }

    /// The repository used for retention and as the lease store.
    #[must_use]
    pub fn repository(&self) -> &Arc<dyn DreamRepository> {
        &self.repository
    }

    /// A snapshot of the status, copied out so no lock is held by the caller.
    #[must_use]
    pub fn status(&self) -> DreamOrchestratorStatus {
        self.state.lock().clone()
    }

    /// The interval the loop would wait after `empty_sweeps` consecutive empty sweeps.
    ///
    /// Delegates to [`DreamOrchestratorConfig::interval_for`]; exposed so a caller with its
    /// own timer can reuse the schedule without reading the configuration.
    #[must_use]
    pub fn interval_for_empty_sweeps(&self, empty_sweeps: u32) -> Duration {
        self.config.interval_for(empty_sweeps)
    }

    /// Runs one tick at the caller's clock, without cancellation.
    ///
    /// Equivalent to [`DreamOrchestrator::tick_with`] with a default [`CallOptions`].
    pub async fn tick(&self, now: DateTime<Utc>) -> Result<DreamRunSummary> {
        self.tick_with(now, &CallOptions::new()).await
    }

    /// Runs one tick at the caller's clock: one sweep, then optionally one retention pass.
    ///
    /// The behaviour, in order:
    ///
    /// - A disabled [`DreamOrchestratorConfig::enabled`] returns an empty
    ///   [`DreamRunSummary`] immediately: nothing is claimed, nothing is purged, and the
    ///   status is not touched (so a later enable starts from a clean snapshot).
    /// - The sweep is [`DreamPlanner::run_due`] or, when
    ///   [`DreamOrchestratorConfig::lease`] is set, [`DreamPlanner::run_due_leased`]; either
    ///   way it claims at most [`crate::dreaming::DreamConfig::batch_size`] jobs. A sweep
    ///   error is recorded in [`DreamOrchestratorStatus::last_error`] with the tick counted
    ///   and `last_tick_at` stamped, and then returned to the caller.
    /// - Retention runs when [`DreamOrchestratorConfig::run_retention_on_tick`] is set, a
    ///   policy is configured, and the policy reports itself enabled. It calls
    ///   [`DreamRetention::purge_expired`] with the same `now`. A retention *error* never
    ///   fails the tick: it is logged, stored in `last_error`, and leaves `last_retention`
    ///   at `None`; success stores the report and clears `last_error`.
    /// - Afterwards the status advances: `ticks` grows by one, `last_tick_at` becomes
    ///   `now`, `consecutive_empty_sweeps` grows when the sweep's
    ///   [`DreamRunSummary::scanned`] is zero (and resets otherwise), `next_interval_ms`
    ///   becomes [`DreamOrchestratorConfig::interval_for`] of that count, and
    ///   `last_summary` is replaced.
    ///
    /// The tick never spawns a task and never sleeps; a caller that wants the schedule
    /// applies [`DreamOrchestrator::interval_for_empty_sweeps`] itself (or uses
    /// [`DreamOrchestrator::run_until_cancelled`]).
    pub async fn tick_with(
        &self,
        now: DateTime<Utc>,
        ctx: &CallOptions,
    ) -> Result<DreamRunSummary> {
        if !self.config.enabled {
            tracing::debug!(
                target: TARGET,
                "the dream orchestrator is disabled; the tick does nothing"
            );
            return Ok(DreamRunSummary::new());
        }
        let sweep = match self.sweep(now, ctx).await {
            Ok(summary) => summary,
            Err(error) => {
                tracing::warn!(
                    target: TARGET,
                    error = %error,
                    "the dream orchestrator sweep failed; the tick reports it"
                );
                self.record_failed_tick(now, &error);
                return Err(error);
            }
        };
        let retention = self.retention_step(now).await;
        self.record_tick(now, &sweep, retention);
        Ok(sweep)
    }

    /// Runs the loop until `cancel` fires, then returns the merged summaries collected so
    /// far.
    ///
    /// This is the whole scheduler; nothing spawns it for you. The loop:
    ///
    /// - returns `Ok` with an empty summary immediately when
    ///   [`DreamOrchestratorConfig::enabled`] is `false`;
    /// - waits [`DreamOrchestratorConfig::initial_delay`] before the first sweep, then
    ///   [`DreamOrchestrator::interval_for_empty_sweeps`] of the status'
    ///   `consecutive_empty_sweeps` between sweeps, by awaiting a timer in
    ///   `tokio::select!` against [`CancellationToken::cancelled`] — the orchestrator
    ///   itself does the waiting, no helper task exists;
    /// - ticks with [`Utc::now`] as the clock. The injectable clock belongs to
    ///   [`DreamOrchestrator::tick`]/[`DreamOrchestrator::tick_with`]; a running loop is
    ///   real time by definition;
    /// - stops on cancellation. If the token fires while waiting, the loop returns
    ///   [`Error::Cancelled`] promptly. If it fires during a tick, the [`CallOptions`]
    ///   carrying the same token makes the planner complete the claimed job as
    ///   [`crate::dreaming::DreamState::Cancelled`], so no job is left `Running`, and the
    ///   loop returns that `Error::Cancelled` when the next wait observes it;
    /// - swallows a *non-cancellation* tick error: it is already recorded in
    ///   [`DreamOrchestratorStatus::last_error`] (and logged again here), and the loop keeps
    ///   going — Python's orchestrator also survives exceptions in its background task. Only
    ///   cancellation ends the loop;
    /// - merges every sweep into one [`DreamRunSummary`], which is only returned if the loop
    ///   ends by something other than cancellation (today that cannot happen, so the
    ///   summary is essentially a diagnostic of the accumulated work).
    pub async fn run_until_cancelled(&self, cancel: CancellationToken) -> Result<DreamRunSummary> {
        let mut total = DreamRunSummary::new();
        if !self.config.enabled {
            tracing::debug!(
                target: TARGET,
                "the dream orchestrator is disabled; the loop exits immediately"
            );
            return Ok(total);
        }
        let mut delay = self.config.initial_delay();
        loop {
            if wait_for(&cancel, delay).await {
                tracing::debug!(
                    target: TARGET,
                    ticks = self.status().ticks,
                    "the dream orchestrator loop stopped by cancellation"
                );
                return Err(Error::cancelled());
            }
            let ctx = CallOptions::new().with_cancellation(cancel.clone());
            match self.tick_with(Utc::now(), &ctx).await {
                Ok(sweep) => total.merge(sweep),
                Err(error) if error.is_cancelled() => {
                    tracing::debug!(
                        target: TARGET,
                        "the dream orchestrator loop stopped by cancellation during a tick"
                    );
                    return Err(error);
                }
                Err(error) => tracing::warn!(
                    target: TARGET,
                    error = %error,
                    "the dream orchestrator tick failed; the loop continues"
                ),
            }
            delay = self
                .config
                .interval_for(self.status().consecutive_empty_sweeps);
        }
    }

    /// Runs one sweep, through the leased planner path when a lease is configured.
    async fn sweep(&self, now: DateTime<Utc>, ctx: &CallOptions) -> Result<DreamRunSummary> {
        if let Some(lease) = &self.config.lease {
            self.planner.run_due_leased(now, ctx, lease).await
        } else {
            self.planner.run_due(now, ctx).await
        }
    }

    /// Applies the retention policy for this tick, if there is one to apply.
    async fn retention_step(&self, now: DateTime<Utc>) -> RetentionStep {
        if !self.config.run_retention_on_tick {
            return RetentionStep::Skipped;
        }
        let Some(policy) = &self.config.retention else {
            return RetentionStep::Skipped;
        };
        if !policy.is_enabled() {
            tracing::debug!(
                target: TARGET,
                "the dream retention policy is disabled; the tick purges nothing"
            );
            return RetentionStep::Skipped;
        }
        match policy.purge_expired(self.repository.as_ref(), now).await {
            Ok(report) => {
                tracing::debug!(
                    target: TARGET,
                    expired = report.deleted_expired,
                    over_cap = report.deleted_over_cap,
                    "the dream retention pass finished"
                );
                RetentionStep::Purged(report)
            }
            Err(error) => {
                tracing::warn!(
                    target: TARGET,
                    error = %error,
                    "the dream retention pass failed; the tick still succeeds"
                );
                RetentionStep::Failed(error.to_string())
            }
        }
    }

    /// Records a tick whose sweep failed: the tick counts and the error is kept.
    fn record_failed_tick(&self, now: DateTime<Utc>, error: &Error) {
        let mut status = self.state.lock();
        status.ticks = status.ticks.saturating_add(1);
        status.last_tick_at = Some(now);
        status.last_error = Some(error.to_string());
    }

    /// Records a successful tick's sweep and retention outcome.
    fn record_tick(&self, now: DateTime<Utc>, sweep: &DreamRunSummary, retention: RetentionStep) {
        let mut status = self.state.lock();
        status.ticks = status.ticks.saturating_add(1);
        status.last_tick_at = Some(now);
        status.consecutive_empty_sweeps = if sweep.scanned == 0 {
            status.consecutive_empty_sweeps.saturating_add(1)
        } else {
            0
        };
        status.next_interval_ms = self
            .config
            .interval_for(status.consecutive_empty_sweeps)
            .as_millis() as u64;
        status.last_summary = Some(sweep.clone());
        match retention {
            RetentionStep::Skipped => {}
            RetentionStep::Purged(report) => {
                status.last_retention = Some(report);
                status.last_error = None;
            }
            RetentionStep::Failed(message) => {
                status.last_retention = None;
                status.last_error = Some(message);
            }
        }
    }
}

/// Awaits `delay`, returning `true` when `cancel` fired first.
///
/// A zero delay does not race the timer: both branches of a `select!` would be ready and
/// the winner would be unspecified, so the token is checked directly instead. That keeps
/// `initial_delay_ms = 0` meaning "tick now unless already cancelled".
async fn wait_for(cancel: &CancellationToken, delay: Duration) -> bool {
    if delay.is_zero() {
        return cancel.is_cancelled();
    }
    tokio::select! {
        () = cancel.cancelled() => true,
        () = tokio::time::sleep(delay) => false,
    }
}
