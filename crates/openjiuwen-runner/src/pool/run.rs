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

use crate::hitl::{ActiveRunSpec, InteractPayload};
use crate::monitor::RunPhase;
use crate::run::{next_stream_step, stream_team, RunOptions, RunServices, StreamStep, TeamRef};

use super::entry::{Entry, EntryInit, FinalOutcome, PoolFailure, PoolFrame, CANCELLED_REASON};
use super::error::PoolError;
use super::key::PoolKey;
use super::reaper;
use super::request::PoolRequest;
use super::{TeamPool, TeamPoolInner, LOG_TARGET};

/// The reason a run stopped by [`TeamPool::stop`] records.
const STOPPED_REASON: &str = "stopped";

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
    /// The pool the run belongs to (for `pause`/`resume`/`stop`).
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

    /// Refuses to pause the leased run; see [`TeamPool::pause`].
    ///
    /// # Errors
    ///
    /// [`PoolError::NotFound`] when the run left the pool or already finished,
    /// and [`Error::Unsupported`] for a live run (nothing can suspend it).
    pub fn pause(&self) -> Result<()> {
        self.pool.pause(self.run.key())
    }

    /// Refuses to resume the leased run; see [`TeamPool::pause`].
    ///
    /// # Errors
    ///
    /// Same as [`PoolLease::pause`].
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
    /// A distributed-configured [`RunServices`] is refused **before** any of
    /// that: [`RunServices::require_local`] answers `Error::Unsupported` naming
    /// the deferred transports, so no permit is consumed, no entry is created, and
    /// no monitor or hub status exists.
    ///
    /// # Errors
    ///
    /// [`PoolError::Busy`] when the key already has a live run,
    /// [`PoolError::Closed`] after [`TeamPool::stop_all`],
    /// [`openjiuwen_common::error::Error::Cancelled`] when the run was cancelled
    /// while queued,
    /// [`openjiuwen_common::error::ErrorKind::Unsupported`] for a
    /// distributed-configured runner, and
    /// [`openjiuwen_common::error::ErrorKind::Config`] outside a tokio runtime.
    #[allow(clippy::needless_pass_by_value)]
    pub async fn start(&self, services: RunServices, request: PoolRequest) -> Result<PoolRun> {
        admit(&self.inner, services, request, false).await
    }

    /// Starts a new run only when a permit is available right now.
    ///
    /// Unlike [`TeamPool::start`], a full pool is [`PoolError::AtCapacity`] and
    /// **no** entry is created. A distributed-configured [`RunServices`] is
    /// refused first ([`RunServices::require_local`]), ahead of every other
    /// admission outcome.
    ///
    /// # Errors
    ///
    /// [`PoolError::AtCapacity`], [`PoolError::Busy`], [`PoolError::Closed`],
    /// [`openjiuwen_common::error::ErrorKind::Unsupported`] for a
    /// distributed-configured runner, and
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
    /// A distributed-configured [`RunServices`] is refused first
    /// ([`RunServices::require_local`]): nothing is created or joined, so a
    /// distributed runner can never attach to an in-process run under a
    /// distributed name.
    ///
    /// # Errors
    ///
    /// [`PoolError::Closed`], the cancellation error of a run cancelled while
    /// queued, [`openjiuwen_common::error::ErrorKind::Unsupported`] for a
    /// distributed-configured runner, and
    /// [`openjiuwen_common::error::ErrorKind::Config`] outside a tokio runtime.
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

    /// Refuses to pause the run pooled under `key`.
    ///
    /// A live run answers [`Error::Unsupported`]: the in-process team strategy
    /// owns no mid-flight suspension seam, so nothing can suspend the invocation
    /// (Python's live `InteractGate` wake-up is deferred), and reporting
    /// [`RunPhase::Paused`] would describe a run that keeps running. The refusal
    /// changes nothing — no phase, no gate, no frame, no monitor transition — and
    /// a host that must stop the run calls [`TeamPool::stop`] or
    /// [`PoolRun::cancel`], which trip the run's cancellation token.
    ///
    /// # Errors
    ///
    /// [`PoolError::NotFound`] when no live run is pooled under `key` (unknown
    /// key, or an entry that already finished), and [`Error::Unsupported`] for a
    /// live one.
    pub fn pause(&self, key: &PoolKey) -> Result<()> {
        self.refuse_suspension("pause", key)
    }

    /// Refuses to resume the run pooled under `key`; see [`TeamPool::pause`].
    ///
    /// # Errors
    ///
    /// Same as [`TeamPool::pause`].
    pub fn resume(&self, key: &PoolKey) -> Result<()> {
        self.refuse_suspension("resume", key)
    }

    /// The typed refusal a pause/resume of `key` reports.
    ///
    /// Shared so the two verbs cannot drift: the negative answers (unknown key,
    /// finished entry) keep their [`PoolError::NotFound`] wording, and a live
    /// entry names the missing suspension seam.
    fn refuse_suspension(&self, action: &str, key: &PoolKey) -> Result<()> {
        let entry = self
            .inner
            .entry(key)
            .ok_or_else(|| PoolError::not_found(key).into_error())?;
        if entry.is_terminal() {
            return Err(PoolError::not_found(key).into_error());
        }
        Err(crate::error::no_suspension_seam(action, entry.run_id()))
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
    // Mode enforcement comes before any admission work: a distributed-configured
    // runner admits nothing, so no permit is consumed, no entry is created, no
    // monitor status exists, and the mode refusal takes precedence over `Busy`.
    services.require_local()?;
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
    // Same ordering as `admit`: the mode refusal precedes every other admission
    // outcome (no permit attempted, no entry created, no monitor status).
    services.require_local()?;
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
    use crate::config::{RunnerConfig, TeamPoolConfig};
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
