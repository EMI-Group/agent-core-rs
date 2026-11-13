//! Requests, live handles, leases, and the one worker task per pooled run.
//!
//! A [`PoolRequest`] is what a host wants to run (a [`TeamRef`], its inputs, a
//! session, and [`RunOptions`]); a [`PoolRun`] is the cheap handle the pool
//! hands back; a [`PoolLease`] is a run kept alive by a holder. Admission is
//! three steps: claim the key (creating the entry in
//! [`RunPhase::Queued`](crate::monitor::RunPhase::Queued)), register it on the
//! observation seams, then wait for one of the pool's concurrency permits before
//! spawning the worker.
//!
//! The worker drives `crate::run::stream_team` with the pool-owned observation
//! seams cleared, forwards every frame as a [`PoolFrame::Output`], and finalises
//! the entry exactly once — as completed, failed, cancelled, or (for a panicking
//! team) failed with `"run task panicked"`.

use std::fmt;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use std::time::Duration;

use futures::FutureExt;
use openjiuwen_common::error::{Error, Result};
use serde_json::{json, Value};
use tokio::sync::broadcast;
use tokio::sync::mpsc;
use tokio::sync::OwnedSemaphorePermit;
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};

use crate::config::{RunnerConfig, TeamPoolConfig};
use crate::hitl::{ActiveRunSpec, InteractPayload};
use crate::monitor::RunPhase;
use crate::run::{
    next_stream_step, resolve_session_id, stream_team, RunOptions, RunServices, StreamStep, TeamRef,
};

use super::entry::{Entry, EntryInit, FinalOutcome, PoolFailure, PoolFrame, CANCELLED_REASON};
use super::error::PoolError;
use super::key::PoolKey;
use super::reaper;
use super::{TeamPool, TeamPoolInner, LOG_TARGET};

/// The reason a run stopped by [`TeamPool::stop`] records.
const STOPPED_REASON: &str = "stopped";

/// What a host asks the pool to run.
///
/// Not serde-serializable on purpose: it carries a live [`TeamRef`] (which may
/// hold an `Arc<dyn Team>`) and [`RunOptions`] (which carries sessions and a
/// cancellation token).
///
/// **Session precedence.** [`PoolRequest::key`] derives the entry's session id
/// through `crate::run::resolve_session_id`:
/// `inputs["conversation_id"]` first, then this request's
/// [`PoolRequest::session_id`], then [`RunOptions::session_id`], then
/// `RunnerConfig::default_session_id`. The derived id always wins, and it is
/// written back into the [`RunOptions`] handed to the run, so the run and the
/// pool entry can never disagree.
#[derive(Clone)]
pub struct PoolRequest {
    /// The team to run: a registry id or a live instance.
    pub team: TeamRef,
    /// The inputs the team's strategy receives.
    pub inputs: Value,
    /// The session id this request asks for, when it asks for one.
    pub session_id: Option<String>,
    /// The label reported on the monitor and the snapshot; defaults to the team
    /// name.
    pub label: Option<String>,
    /// Per-run options (cancellation, deadline, session overrides, metadata).
    pub options: RunOptions,
}

impl PoolRequest {
    /// Builds a request for `team` with `inputs` and default options.
    #[must_use]
    pub fn new(team: impl Into<TeamRef>, inputs: Value) -> Self {
        Self {
            team: team.into(),
            inputs,
            session_id: None,
            label: None,
            options: RunOptions::default(),
        }
    }

    /// Requests `session_id` for the run.
    #[must_use]
    pub fn with_session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    /// Sets the label reported on the monitor and the snapshot.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Replaces the per-run options.
    #[must_use]
    pub fn with_options(mut self, options: RunOptions) -> Self {
        self.options = options;
        self
    }

    /// Cancels the run when `cancellation` is triggered.
    #[must_use]
    pub fn with_cancellation(mut self, cancellation: CancellationToken) -> Self {
        self.options.cancellation = cancellation;
        self
    }

    /// Bounds the run with `timeout` (overriding the pool's `run_timeout`).
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.options.timeout = Some(timeout);
        self
    }

    /// Derives the key this request is pooled under.
    ///
    /// The team name comes from [`PoolKey::for_team`] (an id reference verbatim,
    /// a live team's card id); the session id comes from
    /// `crate::run::resolve_session_id`.
    #[must_use]
    pub fn key(&self, config: &RunnerConfig) -> PoolKey {
        let session_id = resolve_session_id(&self.inputs, &self.options_for_key(), config);
        PoolKey::for_team(&self.team, session_id)
    }

    /// The label this request reports under `team_name`.
    #[must_use]
    pub fn label_or(&self, team_name: &str) -> String {
        self.label.clone().unwrap_or_else(|| team_name.to_owned())
    }

    /// The deadline this request runs under, given the pool configuration.
    #[must_use]
    pub fn deadline(&self, config: &TeamPoolConfig) -> Option<Duration> {
        self.options.timeout.or_else(|| config.run_timeout())
    }

    /// The effective options handed to the run, with the key's session id.
    #[must_use]
    pub fn effective_options(&self, session_id: &str) -> RunOptions {
        let mut options = self.options.clone();
        options.session_id = Some(session_id.to_owned());
        options
    }

    /// The options the key derivation sees: this request's session id, when it
    /// names one, takes precedence over the options' own.
    fn options_for_key(&self) -> RunOptions {
        let mut options = self.options.clone();
        if self.session_id.is_some() {
            options.session_id.clone_from(&self.session_id);
        }
        options
    }
}

impl fmt::Debug for PoolRequest {
    /// Reports the reference, inputs, labels, and options.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PoolRequest")
            .field("team", &self.team)
            .field("inputs", &self.inputs)
            .field("session_id", &self.session_id)
            .field("label", &self.label)
            .field("options", &self.options)
            .finish()
    }
}

/// A cheap, cloneable handle to one pooled run.
///
/// Every clone observes the same entry: the same state, the same frame channel,
/// and the same `wait()`. The handle keeps the entry alive (and therefore out of
/// eviction) but does **not** hold a lease; a run is kept out of eviction by a
/// [`PoolLease`] or by an active concurrency permit.
#[derive(Clone)]
pub struct PoolRun {
    /// The entry the handle points at.
    entry: Arc<Entry>,
}

impl PoolRun {
    /// Wraps one entry (crate-internal: the pool mints the entry).
    pub(crate) fn new(entry: Arc<Entry>) -> Self {
        Self { entry }
    }

    /// The run id the pool minted.
    #[must_use]
    pub fn run_id(&self) -> &str {
        self.entry.run_id()
    }

    /// The key the run is pooled under.
    #[must_use]
    pub fn key(&self) -> &PoolKey {
        self.entry.key()
    }

    /// The team name (the key's team name).
    #[must_use]
    pub fn team_name(&self) -> &str {
        self.entry.team_name()
    }

    /// The session the run is bound to (the key's session id).
    #[must_use]
    pub fn session_id(&self) -> &str {
        self.entry.session_id()
    }

    /// The run's phase.
    #[must_use]
    pub fn state(&self) -> RunPhase {
        self.entry.phase()
    }

    /// Subscribes to the run's frames.
    ///
    /// Best effort: frames published before the call are not replayed, and a
    /// subscriber that falls behind observes `Lagged` instead of missed frames.
    #[must_use]
    pub fn subscribe(&self) -> broadcast::Receiver<PoolFrame> {
        self.entry.subscribe()
    }

    /// Waits for the run to finish and returns its output.
    ///
    /// # Errors
    ///
    /// The run's own failure (kind preserved), and
    /// [`openjiuwen_common::error::Error::Cancelled`] for a cancelled run.
    pub async fn wait(&self) -> Result<Value> {
        self.entry.wait().await
    }

    /// The run's output, once it completed.
    #[must_use]
    pub fn output(&self) -> Option<Value> {
        self.entry.output()
    }

    /// Cancels the run; idempotent, and a finished run is untouched.
    pub fn cancel(&self) {
        self.entry.cancel();
    }

    /// Returns a snapshot of the buffered inbound human payloads.
    #[must_use]
    pub fn payloads(&self) -> Vec<InteractPayload> {
        self.entry.payloads()
    }

    /// Whether the run reached a terminal phase.
    #[must_use]
    pub fn is_finished(&self) -> bool {
        self.entry.is_terminal()
    }
}

impl fmt::Debug for PoolRun {
    /// Reports the run id, key, and phase.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PoolRun")
            .field("run_id", &self.run_id())
            .field("key", &self.key())
            .field("state", &self.state())
            .finish()
    }
}

/// A run kept alive by the holder of the lease.
///
/// A lease is the pool's "somebody is still using this run" marker: while one
/// exists, idle eviction skips the entry. Releasing the last lease stamps the
/// entry idle, after which a finished entry is evicted once
/// `TeamPoolConfig::idle_timeout` passes (and a still-running orphan is stopped
/// at the same deadline). Dropping a lease releases it.
pub struct PoolLease {
    /// The pool the run belongs to (for `pause`/`resume`).
    pool: TeamPool,
    /// The run the lease holds.
    run: PoolRun,
    /// Whether the lease was already released.
    released: bool,
}

impl PoolLease {
    /// Wraps a run whose lease was already counted.
    pub(crate) fn new(pool: TeamPool, run: PoolRun) -> Self {
        Self {
            pool,
            run,
            released: false,
        }
    }

    /// The key of the leased run.
    #[must_use]
    pub fn key(&self) -> &PoolKey {
        self.run.key()
    }

    /// The leased run.
    #[must_use]
    pub fn run(&self) -> &PoolRun {
        &self.run
    }

    /// Waits for the leased run to finish and returns its output.
    ///
    /// # Errors
    ///
    /// Same as [`PoolRun::wait`].
    pub async fn wait(&self) -> Result<Value> {
        self.run.wait().await
    }

    /// Pauses the leased run (a coordination signal; the team keeps running).
    ///
    /// # Errors
    ///
    /// [`PoolError::NotFound`] when the run left the pool.
    pub fn pause(&self) -> Result<()> {
        self.pool.pause(self.run.key())
    }

    /// Resumes the leased run (a coordination signal).
    ///
    /// # Errors
    ///
    /// [`PoolError::NotFound`] when the run left the pool.
    pub fn resume(&self) -> Result<()> {
        self.pool.resume(self.run.key())
    }

    /// Releases the lease (idempotent; `Drop` would do it anyway).
    pub fn release(mut self) {
        self.release_inner();
    }

    /// Whether the lease was already released.
    #[must_use]
    pub fn is_released(&self) -> bool {
        self.released
    }

    /// The run id of the leased run.
    #[must_use]
    pub fn run_id(&self) -> &str {
        self.run.run_id()
    }

    /// The phase of the leased run.
    #[must_use]
    pub fn state(&self) -> RunPhase {
        self.run.state()
    }

    /// Decrements the entry's lease count, at most once.
    fn release_inner(&mut self) {
        if self.released {
            return;
        }
        self.released = true;
        self.run.entry.release_lease();
    }
}

impl Drop for PoolLease {
    /// Releases the lease unless it was released explicitly.
    fn drop(&mut self) {
        self.release_inner();
    }
}

impl fmt::Debug for PoolLease {
    /// Reports the run id, the key, and whether the lease was released.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PoolLease")
            .field("run_id", &self.run_id())
            .field("key", &self.key())
            .field("released", &self.released)
            .finish_non_exhaustive()
    }
}

/// One admission slot: an entry that was just created, or a live one to join.
enum Slot {
    /// The key already had a live run, which this admission joins.
    Joined(Arc<Entry>),
    /// A fresh entry was created and inserted.
    Fresh(Arc<Entry>),
}

/// The body of one pooled run, driven by its worker task.
struct StreamDrive {
    /// The entry the run finalises.
    entry: Arc<Entry>,
    /// The inner services, without the observation seams.
    services: RunServices,
    /// The team to run.
    team: TeamRef,
    /// The team's inputs.
    inputs: Value,
    /// The effective options (the key's session id written back).
    options: RunOptions,
    /// The entry's own token (tripped by `stop`, `cancel`, and `hub.cancel`).
    entry_token: CancellationToken,
    /// The caller's token, observed by `next_stream_step`.
    caller_token: CancellationToken,
    /// The configured deadline, for the timeout message.
    limit: Option<Duration>,
    /// The absolute deadline, so `tokio::time::pause()` drives it.
    deadline_at: Option<(tokio::time::Instant, Duration)>,
}

impl StreamDrive {
    /// Drives the team stream and returns the terminal verdict.
    async fn run(self) -> FinalOutcome {
        let mut stream = stream_team(self.services, self.team, self.inputs, self.options);
        let mut last = Value::Null;
        loop {
            // Biased to the entry's own token, then to `next_stream_step`, which
            // itself is biased to the caller's token, then to the deadline, then
            // to the frame: neither cancellation can be masked by a ready frame.
            let step = tokio::select! {
                biased;
                () = self.entry_token.cancelled() => StreamStep::Cancelled,
                step = next_stream_step(&mut stream, &self.caller_token, self.deadline_at) => step,
            };
            match step {
                StreamStep::Frame(Some(Ok(value))) => {
                    self.entry.emit(PoolFrame::Output {
                        value: value.clone(),
                    });
                    last = value;
                }
                StreamStep::Frame(Some(Err(error))) => {
                    return FinalOutcome::Failed(PoolFailure::from_error(&error))
                }
                StreamStep::Frame(None) => return FinalOutcome::Completed(last),
                StreamStep::Cancelled => {
                    return FinalOutcome::Cancelled(CANCELLED_REASON.to_owned())
                }
                StreamStep::TimedOut => {
                    let limit = self.limit.unwrap_or_default();
                    let error = crate::error::run_timeout("run_agent_team", limit);
                    return FinalOutcome::Failed(PoolFailure::from_error(&error));
                }
            }
        }
    }
}

impl TeamPool {
    /// Starts a new run, waiting for a concurrency permit when the pool is full.
    ///
    /// The entry is created in [`RunPhase::Queued`]
    /// and registered on the monitor and the hub before the wait, so a queued run
    /// is observable. A run whose entry token is tripped while queued is
    /// finalised as cancelled and never starts.
    ///
    /// # Errors
    ///
    /// [`PoolError::Busy`] when the key already has a live run,
    /// [`PoolError::Closed`] after [`TeamPool::stop_all`],
    /// [`openjiuwen_common::error::Error::Cancelled`] when the run was cancelled
    /// while queued, and [`openjiuwen_common::error::ErrorKind::Config`] outside
    /// a tokio runtime.
    #[allow(clippy::needless_pass_by_value)]
    pub async fn start(&self, services: RunServices, request: PoolRequest) -> Result<PoolRun> {
        admit(&self.inner, services, request, false).await
    }

    /// Starts a new run only when a permit is available right now.
    ///
    /// Unlike [`TeamPool::start`], a full pool is [`PoolError::AtCapacity`] and
    /// **no** entry is created.
    ///
    /// # Errors
    ///
    /// [`PoolError::AtCapacity`], [`PoolError::Busy`], [`PoolError::Closed`], and
    /// [`openjiuwen_common::error::ErrorKind::Config`] outside a tokio runtime.
    #[allow(clippy::needless_pass_by_value)]
    pub fn try_start(&self, services: RunServices, request: PoolRequest) -> Result<PoolRun> {
        try_admit(&self.inner, services, request)
    }

    /// Starts a run and takes its first lease, or joins a live one.
    ///
    /// Joining a live run increments its lease count and **ignores** the joining
    /// request's inputs, label, and options: the run keeps the inputs and options
    /// of the request that created it.
    ///
    /// # Errors
    ///
    /// [`PoolError::Closed`], the cancellation error of a run cancelled while
    /// queued, and [`openjiuwen_common::error::ErrorKind::Config`] outside a
    /// tokio runtime.
    #[allow(clippy::needless_pass_by_value)]
    pub async fn acquire(&self, services: RunServices, request: PoolRequest) -> Result<PoolLease> {
        let run = admit(&self.inner, services, request, true).await?;
        Ok(PoolLease::new(self.clone(), run))
    }

    /// Stops the run pooled under `key`.
    ///
    /// Cancels the entry, finalises it as
    /// [`RunPhase::Cancelled`], removes it
    /// from the pool, and awaits the worker for at most
    /// `TeamPoolConfig::stop_timeout` (aborting it on expiry). An unknown key is
    /// `Ok(false)`, not an error, so a host can call it unconditionally.
    ///
    /// # Errors
    ///
    /// None today: every failure path inside is logged and the sweep continues.
    /// The `Result` is kept so a future bounded teardown can report one.
    pub async fn stop(&self, key: &PoolKey) -> Result<bool> {
        let Some(entry) = self.inner.remove(key) else {
            return Ok(false);
        };
        entry.cancel_and_finalise(self.inner.next_seq(), STOPPED_REASON);
        entry.abort_drain();
        if let Some(mut worker) = entry.take_worker() {
            let timeout = self.inner.config().stop_timeout();
            if tokio::time::timeout(timeout, &mut worker).await.is_err() {
                warn!(
                    target: LOG_TARGET,
                    key = %key,
                    "pooled team run did not stop within the stop timeout"
                );
                worker.abort();
            }
        }
        debug!(target: LOG_TARGET, key = %key, "stopped a pooled team run");
        Ok(true)
    }

    /// Closes the entry's gate and reports it as paused.
    ///
    /// A coordination signal only: it does not suspend the team loop (Python's
    /// live `InteractGate` wake-up is deferred).
    ///
    /// # Errors
    ///
    /// [`PoolError::NotFound`] when no live run is pooled under `key`.
    pub fn pause(&self, key: &PoolKey) -> Result<()> {
        let entry = self
            .inner
            .entry(key)
            .ok_or_else(|| PoolError::not_found(key).into_error())?;
        entry.pause()
    }

    /// Opens the entry's gate and reports it as resumed.
    ///
    /// # Errors
    ///
    /// [`PoolError::NotFound`] when no live run is pooled under `key`.
    pub fn resume(&self, key: &PoolKey) -> Result<()> {
        let entry = self
            .inner
            .entry(key)
            .ok_or_else(|| PoolError::not_found(key).into_error())?;
        entry.resume()
    }

    /// Drains the buffered inbound human payloads of the run under `key`.
    ///
    /// An unknown or already-dropped key yields an empty vector.
    pub fn take_payloads(&self, key: &PoolKey) -> Vec<InteractPayload> {
        self.inner
            .entry(key)
            .map_or_else(Vec::new, |entry| entry.take_payloads())
    }
}

/// Admits a request: claim the key, observe the run, wait for a permit, spawn.
async fn admit(
    inner: &Arc<TeamPoolInner>,
    services: RunServices,
    request: PoolRequest,
    lease: bool,
) -> Result<PoolRun> {
    if inner.is_closed() {
        return Err(PoolError::Closed.into_error());
    }
    require_runtime()?;
    let key = request.key(&services.config);
    let entry = match claim(inner, &services, &request, &key, lease)? {
        Slot::Joined(entry) => {
            entry.add_lease();
            debug!(target: LOG_TARGET, key = %key, "joined a live pooled team run");
            return Ok(PoolRun::new(entry));
        }
        Slot::Fresh(entry) => entry,
    };
    observe(&services, &entry);
    reaper::ensure_started(inner);

    let token = entry.token();
    let permit = tokio::select! {
        biased;
        () = token.cancelled() => {
            entry.finalise(
                &FinalOutcome::Cancelled(CANCELLED_REASON.to_owned()),
                inner.next_seq(),
            );
            inner.prune_finished();
            return Err(Error::cancelled());
        }
        permit = inner.semaphore().acquire_owned() => match permit {
            Ok(permit) => permit,
            Err(_) => return Err(PoolError::Closed.into_error()),
        },
    };
    Ok(launch(inner, services, request, entry, permit, lease))
}

/// Admits a request without waiting for a permit.
fn try_admit(
    inner: &Arc<TeamPoolInner>,
    services: RunServices,
    request: PoolRequest,
) -> Result<PoolRun> {
    if inner.is_closed() {
        return Err(PoolError::Closed.into_error());
    }
    require_runtime()?;
    let key = request.key(&services.config);
    if inner
        .entry(&key)
        .is_some_and(|existing| !existing.is_terminal())
    {
        return Err(PoolError::busy(&key).into_error());
    }
    let permit = match inner.semaphore().try_acquire_owned() {
        Ok(permit) => permit,
        Err(tokio::sync::TryAcquireError::NoPermits) => {
            return Err(PoolError::AtCapacity {
                capacity: inner.config().max_runs(),
            }
            .into_error())
        }
        Err(tokio::sync::TryAcquireError::Closed) => return Err(PoolError::Closed.into_error()),
    };
    let Slot::Fresh(entry) = claim(inner, &services, &request, &key, false)? else {
        return Err(PoolError::busy(&key).into_error());
    };
    observe(&services, &entry);
    reaper::ensure_started(inner);
    Ok(launch(inner, services, request, entry, permit, false))
}

/// Claims `key` for a new entry, or hands back the live one to join.
///
/// The check and the insertion happen under one map lock, so two competing
/// admissions cannot both create an entry for the same key.
fn claim(
    inner: &Arc<TeamPoolInner>,
    services: &RunServices,
    request: &PoolRequest,
    key: &PoolKey,
    lease: bool,
) -> Result<Slot> {
    let mut entries = inner.entries();
    if let Some(existing) = entries.get(key) {
        if !existing.is_terminal() {
            if lease {
                return Ok(Slot::Joined(Arc::clone(existing)));
            }
            return Err(PoolError::busy(key).into_error());
        }
    }
    let entry = Arc::new(Entry::new(EntryInit {
        run_id: openjiuwen_common::utils::id::uuid_v4(),
        key: key.clone(),
        label: request.label_or(&key.team_name),
        deadline: request.deadline(inner.config()),
        payload_capacity: inner.config().payload_capacity(),
        frame_capacity: inner.config().frame_capacity(),
        hub: services.hitl.clone(),
        monitor: services.monitor.clone(),
    }));
    entries.insert(key.clone(), Arc::clone(&entry));
    Ok(Slot::Fresh(entry))
}

/// Registers the entry on the hub (with a payload drain) and starts its worker.
fn launch(
    inner: &Arc<TeamPoolInner>,
    services: RunServices,
    request: PoolRequest,
    entry: Arc<Entry>,
    permit: OwnedSemaphorePermit,
    lease: bool,
) -> PoolRun {
    entry.install_permit(permit);
    if lease {
        entry.add_lease();
    }
    let worker = tokio::spawn(drive(
        Arc::downgrade(inner),
        Arc::clone(&entry),
        services,
        request,
    ));
    entry.install_worker(worker);
    PoolRun::new(entry)
}

/// Registers the run on the hub and starts its payload drain.
fn observe(services: &RunServices, entry: &Arc<Entry>) {
    let detail = json!({"team": entry.team_name()});
    let binding = crate::hitl::bind_run(
        services.hitl.as_ref(),
        ActiveRunSpec::for_execution(
            entry.run_id(),
            entry.label(),
            Some(entry.team_name()),
            Some(entry.session_id()),
            &entry.token(),
            detail,
        ),
        None,
    );
    entry.install_binding(binding);

    let Some(hub) = services.hitl.as_ref() else {
        return;
    };
    let Some(receiver) = hub.take_inbox(entry.run_id()) else {
        return;
    };
    entry.install_drain(tokio::spawn(drain_payloads(
        Arc::downgrade(entry),
        receiver,
    )));
}

/// Moves every inbound payload of the run into its bounded buffer.
///
/// Ends when the hub closes the inbox or the entry's token fires. Holds only a
/// `Weak` entry, so the drain can never keep a run alive.
async fn drain_payloads(
    entry: std::sync::Weak<Entry>,
    mut receiver: mpsc::Receiver<InteractPayload>,
) {
    let Some(strong) = entry.upgrade() else {
        return;
    };
    let token = strong.token();
    drop(strong);
    loop {
        tokio::select! {
            biased;
            () = token.cancelled() => break,
            inbound = receiver.recv() => match inbound {
                Some(payload) => match entry.upgrade() {
                    Some(entry) => entry.push_payload(payload),
                    None => break,
                },
                None => break,
            },
        }
    }
}

/// The worker of one pooled run: announce it, drive it, finalise it once.
async fn drive(
    pool: std::sync::Weak<TeamPoolInner>,
    entry: Arc<Entry>,
    services: RunServices,
    request: PoolRequest,
) {
    let run_id = entry.run_id().to_owned();
    let session_id = entry.session_id().to_owned();

    entry.mark_running();
    if let Some(hub) = services.hitl.as_ref() {
        if let Err(error) = hub.resume(&run_id) {
            debug!(target: LOG_TARGET, run_id, %error, "hub resume failed");
        }
    }
    entry.emit(PoolFrame::Started {
        run_id: run_id.clone(),
        team_name: entry.team_name().to_owned(),
        session_id: session_id.clone(),
    });

    let options = request.effective_options(&session_id);
    let caller_token = options.cancellation.clone();
    let entry_token = entry.token();
    let limit = entry.deadline();
    let deadline_at = limit.map(|limit| (tokio::time::Instant::now() + limit, limit));
    let drive = StreamDrive {
        entry: Arc::clone(&entry),
        services: pooled_services(services),
        team: request.team,
        inputs: request.inputs,
        options,
        entry_token,
        caller_token,
        limit,
        deadline_at,
    };

    let outcome = AssertUnwindSafe(drive.run()).catch_unwind().await;
    let outcome = outcome.unwrap_or_else(|_| {
        warn!(target: LOG_TARGET, run_id, "pooled team run panicked");
        FinalOutcome::Failed(PoolFailure::panicked())
    });

    let pool = pool.upgrade();
    let seq = pool.as_ref().map_or(0, |inner| inner.next_seq());
    entry.finalise(&outcome, seq);
    if let Some(inner) = pool.as_ref() {
        inner.prune_finished();
    }
}

/// A `RunServices` snapshot with the pool-owned observation seams cleared.
///
/// The pool owns the monitor and the hub for a pooled run, so the inner
/// execution path must not create a second monitor entry or a second hub
/// registration (which would fight over the run's phase).
pub(crate) fn pooled_services(services: RunServices) -> RunServices {
    let mut inner = services;
    inner.monitor = None;
    inner.hitl = None;
    inner
}

/// Rejects an admission outside a tokio runtime instead of panicking in
/// `tokio::spawn`.
fn require_runtime() -> Result<()> {
    tokio::runtime::Handle::try_current()
        .map(|_| ())
        .map_err(|_| Error::config("the team pool requires a tokio runtime"))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::callback::CallbackFramework;
    use crate::resources::ResourceMgr;

    fn services(default_session: &str) -> RunServices {
        RunServices::new(
            ResourceMgr::new(),
            CallbackFramework::new(),
            RunnerConfig::default().with_default_session_id(default_session),
            None,
        )
    }

    fn request(session: Option<&str>) -> PoolRequest {
        let mut request = PoolRequest::new("crew", json!({"query": "hi"}));
        if let Some(session) = session {
            request = request.with_session_id(session);
        }
        request
    }

    #[test]
    fn a_request_defaults_to_empty_options() {
        let request = PoolRequest::new("crew", json!({"q": 1}));
        assert!(matches!(request.team, TeamRef::Id(ref id) if id == "crew"));
        assert_eq!(request.inputs, json!({"q": 1}));
        assert!(request.session_id.is_none());
        assert!(request.label.is_none());
        assert!(request.options.session_id.is_none());
        let debug = format!("{request:?}");
        assert!(debug.contains("PoolRequest"), "{debug}");
        assert!(debug.contains("crew"), "{debug}");
    }

    #[test]
    fn builders_set_the_label_session_cancellation_and_timeout() {
        let token = CancellationToken::new();
        let request = PoolRequest::new("crew", json!({}))
            .with_session_id("s-9")
            .with_label("nightly")
            .with_options(RunOptions::default().with_session_id("s-other"))
            .with_cancellation(token.clone())
            .with_timeout(Duration::from_millis(250));
        assert_eq!(request.session_id.as_deref(), Some("s-9"));
        assert_eq!(request.label.as_deref(), Some("nightly"));
        assert_eq!(request.options.timeout, Some(Duration::from_millis(250)));
        token.cancel();
        assert!(request.options.cancellation.is_cancelled());
        // The explicit request session id wins over the options'.
        let config = RunnerConfig::default();
        assert_eq!(request.key(&config).session_id, "s-9");
    }

    #[test]
    fn the_key_prefers_the_conversation_then_the_request_then_the_options() {
        let config = RunnerConfig::default().with_default_session_id("runner-default");
        assert_eq!(
            request(None).key(&config),
            PoolKey::new("crew", "runner-default")
        );
        assert_eq!(
            request(Some("s-9")).key(&config),
            PoolKey::new("crew", "s-9")
        );

        let options_request =
            request(None).with_options(RunOptions::default().with_session_id("s-opt"));
        assert_eq!(options_request.key(&config), PoolKey::new("crew", "s-opt"));

        let mut with_conversation = request(Some("s-9"));
        with_conversation.inputs = json!({"conversation_id": "conv-1"});
        assert_eq!(
            with_conversation.key(&config),
            PoolKey::new("crew", "conv-1")
        );
    }

    #[test]
    fn a_label_falls_back_to_the_team_name_and_a_deadline_to_the_pool() {
        assert_eq!(request(None).label_or("crew"), "crew");
        assert_eq!(request(None).with_label("n").label_or("crew"), "n");

        let pool = TeamPoolConfig::default();
        assert!(request(None).deadline(&pool).is_none());
        let pool = pool.with_run_timeout_secs(Some(2.0));
        assert_eq!(request(None).deadline(&pool), Some(Duration::from_secs(2)));
        assert_eq!(
            request(None)
                .with_timeout(Duration::from_millis(5))
                .deadline(&pool),
            Some(Duration::from_millis(5))
        );
    }

    #[test]
    fn the_effective_options_carry_the_key_session() {
        let options = request(Some("s-1")).effective_options("s-2");
        assert_eq!(options.session_id.as_deref(), Some("s-2"));
    }

    #[test]
    fn pooled_services_clear_the_observation_seams_only() {
        let monitor = Arc::new(crate::monitor::RunMonitor::new());
        let hub = Arc::new(crate::hitl::InteractionHub::new());
        let services = services("s-1")
            .with_monitor(Arc::clone(&monitor))
            .with_hitl(Arc::clone(&hub));
        let resources = Arc::clone(&services.resources);
        let pooled = pooled_services(services);
        assert!(pooled.monitor.is_none());
        assert!(pooled.hitl.is_none());
        assert!(Arc::ptr_eq(&pooled.resources, &resources));
    }

    #[tokio::test]
    async fn a_closed_pool_refuses_every_admission() {
        let pool = TeamPool::new();
        let _ = pool.stop_all().await;
        assert!(pool.is_closed());

        let start = pool
            .start(services("s-1"), request(None))
            .await
            .unwrap_err();
        assert_eq!(start.kind(), openjiuwen_common::error::ErrorKind::Config);
        assert_eq!(start.to_string(), "config: team pool is closed");

        let try_start = pool.try_start(services("s-1"), request(None)).unwrap_err();
        assert_eq!(
            try_start.kind(),
            openjiuwen_common::error::ErrorKind::Config
        );
        let acquire = pool
            .acquire(services("s-1"), request(None))
            .await
            .unwrap_err();
        assert_eq!(acquire.kind(), openjiuwen_common::error::ErrorKind::Config);
    }

    #[test]
    fn an_admission_outside_a_runtime_is_a_config_error() {
        let pool = TeamPool::new();
        let error = pool
            .try_start(services("s-1"), request(None))
            .expect_err("no runtime");
        assert_eq!(
            error.to_string(),
            "config: the team pool requires a tokio runtime"
        );
        assert!(pool.is_empty(), "nothing was created");
    }

    #[tokio::test]
    async fn stopping_an_unknown_key_is_false_and_pausing_one_is_not_found() {
        let pool = TeamPool::new();
        let key = PoolKey::new("crew", "s-1");
        assert!(!pool.stop(&key).await.unwrap());
        assert_eq!(pool.take_payloads(&key), Vec::new());

        let error = pool.pause(&key).unwrap_err();
        assert_eq!(
            error.to_string(),
            "not_found: no active pooled team run for 'crew' / session 's-1'"
        );
        let error = pool.resume(&key).unwrap_err();
        assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::NotFound);
    }

    #[tokio::test]
    async fn a_run_without_a_permit_reports_the_capacity() {
        let pool = TeamPool::with_config(TeamPoolConfig::default().with_max_concurrent_runs(1));
        // A key is busy only once an entry exists; capacity is checked next, so a
        // pool whose only permit is gone reports the bound instead of an entry.
        let permits = pool.inner.semaphore();
        let held = permits.try_acquire_owned().expect("one permit");
        let error = pool
            .try_start(services("s-1"), request(None))
            .expect_err("at capacity");
        assert_eq!(
            error.to_string(),
            "config: team pool is at capacity (1 concurrent runs)"
        );
        drop(held);
    }
}
