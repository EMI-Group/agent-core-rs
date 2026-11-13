//! The live-run registry, the per-run inbox, and the interaction answering path.
//!
//! [`InteractionHub`] is one `parking_lot::Mutex` over a `BTreeMap<run_id, Entry>`
//! behind an `Arc`, so cloning it is cheap and every clone sees the same runs.
//! The `BTreeMap` is what makes delivery, `list_active`, and `pending_all`
//! deterministic: they always visit runs in sorted run-id order.
//!
//! **What an entry owns.** The label/team/session from
//! [`ActiveRunSpec`], the hub's coordination state and gate flag, the run's
//! cancellation token, a bounded inbox (`INBOX_CAPACITY` payloads), the optional
//! [`Session`] the run is bound to, the set of interaction request ids the hub
//! has observed but not answered, and the token of the run's watcher task.
//!
//! **The watcher.** [`InteractionHub::attach_session`] subscribes to the
//! session's stream and spawns one task (in the `bridge` submodule) that records
//! every `__interaction__` frame's request id in the run's pending set. The task
//! holds only a `Weak` reference to the registry and is cancelled by the run's
//! watcher token, so a removed (or dropped) hub cannot be kept alive by it.
//!
//! **No lock across `.await`.** Every method that awaits (`answer`,
//! `close_interactions`) clones the `Arc<dyn Session>` and the ids, drops the
//! guard, and only then awaits. The synchronous methods never hold the guard
//! longer than their own body.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::utils::id::uuid_v4;
use openjiuwen_session::interaction::InteractionResponse;
use openjiuwen_session::Session;
use parking_lot::Mutex;
use serde_json::Value;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};

use crate::monitor::{RunKind, RunMonitor, RunOutcome, RunPhase, RunStart};

use super::bridge::{ActiveRunInfo, ActiveRunSpec, RunHandle};
use super::payload::InteractPayload;
use super::target::{DeliverResult, RunTarget};
use super::TARGET;

/// How many inbound payloads a run's inbox buffers before `deliver` reports
/// [`DeliverResult::Closed`].
///
/// The inbox is per run and bounded so a host that floods one run cannot exhaust
/// memory; a full inbox is reported, never silently dropped.
pub const INBOX_CAPACITY: usize = 64;

/// One registered run.
pub(super) struct Entry {
    /// Human-readable label reported in [`ActiveRunInfo`].
    label: String,
    /// Team name, when the run is a team run.
    team_name: Option<String>,
    /// Session id, when the run is bound to a session.
    session_id: Option<String>,
    /// Caller detail forwarded to the monitor's initial event.
    #[allow(dead_code)] // retained for the execution paths' host detail
    detail: Value,
    /// The hub's coordination state.
    state: RunPhase,
    /// Whether the interaction gate is closed (paused or cancelled).
    gate_closed: bool,
    /// The run's cancellation token.
    cancellation: CancellationToken,
    /// The inbox sender; `None` once the receiver was dropped.
    inbox: Option<mpsc::Sender<InteractPayload>>,
    /// The inbox receiver, until [`InteractionHub::take_inbox`] takes it.
    inbox_rx: Option<mpsc::Receiver<InteractPayload>>,
    /// Whether [`InteractionHub::take_inbox`] already handed out the receiver.
    inbox_taken: bool,
    /// The session the run is bound to, once attached.
    session: Option<Arc<dyn Session>>,
    /// Interaction request ids observed on the session stream, awaiting an answer.
    pub(super) pending: BTreeSet<String>,
    /// Cancels this run's watcher task (a new token per attached session).
    watcher: CancellationToken,
}

impl Entry {
    /// Renders the host-facing snapshot of this entry registered under `run_id`.
    fn info(&self, run_id: &str) -> ActiveRunInfo {
        ActiveRunInfo {
            run_id: run_id.to_string(),
            label: self.label.clone(),
            team_name: self.team_name.clone(),
            session_id: self.session_id.clone(),
            state: self.state,
            gate_closed: self.gate_closed,
            queued: self.pending.len(),
        }
    }
}

/// The shared run table.
#[derive(Default)]
pub(super) struct Inner {
    /// The registered runs; `BTreeMap` keeps every listing sorted by run id.
    pub(super) runs: BTreeMap<String, Entry>,
}

/// The runner-level human-in-the-loop hub for active runs.
///
/// See the [crate-level docs](crate::hitl) for the hub's Python correspondence
/// and in-process boundaries, and this module's docs for what a hub owns.
/// Cloning shares one table.
#[derive(Clone, Default)]
pub struct InteractionHub {
    /// The run table.
    inner: Arc<Mutex<Inner>>,
    /// The optional monitor mirrored with the hub's transitions.
    monitor: Option<Arc<RunMonitor>>,
}

impl InteractionHub {
    /// Creates an empty hub with no monitor.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Attaches a monitor the hub mirrors its state transitions into.
    ///
    /// Only this hub (and clones made afterwards) observe the monitor; a clone
    /// made earlier keeps its own setting.
    #[must_use]
    pub fn with_monitor(mut self, monitor: Arc<RunMonitor>) -> Self {
        self.monitor = Some(monitor);
        self
    }

    /// Returns the attached monitor, if any.
    #[must_use]
    pub fn monitor(&self) -> Option<Arc<RunMonitor>> {
        self.monitor.clone()
    }

    /// Registers a run and returns its handle.
    ///
    /// Mints a UUID v4 run id when `spec.run_id` is absent or empty. Registers
    /// are not deduplicated: a second register under a live id replaces the entry
    /// (its inbox, session, pending set, and watcher are dropped), so a re-run
    /// under the same id starts clean. With a monitor attached the run is tracked
    /// as [`RunPhase::Queued`]; the kind is [`RunKind::Team`] when the spec names
    /// a team and [`RunKind::Agent`] otherwise. `track` is idempotent, so a run
    /// path that already began the status is not duplicated, and the hub's own
    /// state stays authoritative.
    pub fn register(&self, spec: ActiveRunSpec) -> RunHandle {
        let ActiveRunSpec {
            run_id,
            label,
            team_name,
            session_id,
            cancellation,
            detail,
        } = spec;
        let run_id = run_id.filter(|id| !id.is_empty()).unwrap_or_else(uuid_v4);
        let cancellation = cancellation.unwrap_or_default();
        let (inbox, inbox_rx) = mpsc::channel(INBOX_CAPACITY);

        let replaced = {
            let mut inner = self.inner.lock();
            inner.runs.insert(
                run_id.clone(),
                Entry {
                    label: label.clone(),
                    team_name: team_name.clone(),
                    session_id: session_id.clone(),
                    detail: detail.clone(),
                    state: RunPhase::Queued,
                    gate_closed: false,
                    cancellation: cancellation.clone(),
                    inbox: Some(inbox),
                    inbox_rx: Some(inbox_rx),
                    inbox_taken: false,
                    session: None,
                    pending: BTreeSet::new(),
                    watcher: CancellationToken::new(),
                },
            )
        };
        if let Some(previous) = replaced {
            previous.watcher.cancel();
        }

        if let Some(monitor) = self.monitor.as_ref() {
            let kind = if team_name.is_some() {
                RunKind::Team
            } else {
                RunKind::Agent
            };
            let start = RunStart {
                kind,
                label,
                session_id,
                detail,
            };
            let _ = monitor.track(run_id.clone(), start);
        }

        RunHandle::new(run_id, cancellation)
    }

    /// Removes a registered run.
    ///
    /// Drops the inbox, the attached session, and the pending set, closes the
    /// gate (the entry is gone, so no delivery or pause can reach it), and cancels
    /// the run's watcher token so the task ends. When a monitor is attached and
    /// the run is still non-terminal (`Queued`/`Running`/`Paused`), it is
    /// finalised as [`RunOutcome::completed`]; a monitor that no longer knows the
    /// run is logged at debug. The watcher task is never waited on.
    ///
    /// # Errors
    ///
    /// [`Error::NotFound`] (`"run '<id>' is not registered"`) when the id is
    /// unknown.
    pub fn unregister(&self, run_id: &str) -> Result<()> {
        let removed = self.inner.lock().runs.remove(run_id);
        let Some(entry) = removed else {
            return Err(super::not_registered(run_id));
        };
        entry.watcher.cancel();
        drop(entry);

        if let Some(monitor) = self.monitor.as_ref() {
            match monitor.snapshot(run_id) {
                Ok(status) if !status.is_terminal() => {
                    if let Err(error) = monitor.finish(run_id, RunOutcome::completed()) {
                        warn!(
                            target: TARGET,
                            run_id,
                            %error,
                            "run status finish failed"
                        );
                    }
                }
                Ok(_) => {}
                Err(error) => {
                    debug!(target: TARGET, run_id, %error, "monitor did not track the run");
                }
            }
        }
        Ok(())
    }

    /// Binds `session` to a run and starts its interaction watcher.
    ///
    /// The stream is subscribed on the calling thread (before the watcher task
    /// exists), so a frame published immediately after this call is buffered and
    /// still observed. Replacing a session cancels the previous watcher. Outside a
    /// tokio runtime the session is attached without a watcher and a warning is
    /// logged instead of panicking.
    ///
    /// # Errors
    ///
    /// [`Error::NotFound`] when the run id is unknown.
    pub fn attach_session(&self, run_id: &str, session: Arc<dyn Session>) -> Result<()> {
        let (stream, watcher) = {
            let mut inner = self.inner.lock();
            let entry = inner
                .runs
                .get_mut(run_id)
                .ok_or_else(|| super::not_registered(run_id))?;
            entry.watcher.cancel();
            entry.watcher = CancellationToken::new();
            let stream = session.subscribe();
            let watcher = entry.watcher.clone();
            entry.session = Some(session);
            (stream, watcher)
        };

        let inner = Arc::downgrade(&self.inner);
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => {
                handle.spawn(super::bridge::watch_pending(
                    inner,
                    run_id.to_string(),
                    stream,
                    watcher,
                ));
            }
            Err(_) => {
                warn!(
                    target: TARGET,
                    run_id,
                    "no tokio runtime: attached the session without an interaction watcher"
                );
            }
        }
        Ok(())
    }

    /// Delivers `payload` to the single live run `target` addresses.
    ///
    /// Runs are visited in sorted run-id order, so the choice is deterministic
    /// when two runs share a team name. A full inbox and a closed inbox both
    /// report [`DeliverResult::Closed`]; a closed inbox also drops the dead
    /// sender, so later deliveries report [`DeliverResult::NoInbox`].
    #[must_use]
    // Signature fixed by the hub API: a target is a value object a caller builds
    // per delivery.
    #[allow(clippy::needless_pass_by_value)]
    pub fn deliver(&self, target: RunTarget, payload: InteractPayload) -> DeliverResult {
        let mut inner = self.inner.lock();
        let Some(run_id) = inner
            .runs
            .iter()
            .find(|(run_id, entry)| target.matches(&entry.info(run_id)))
            .map(|(run_id, _)| run_id.clone())
        else {
            return DeliverResult::MissingTarget;
        };
        let Some(entry) = inner.runs.get_mut(&run_id) else {
            return DeliverResult::MissingTarget;
        };
        let outcome = match entry.inbox.as_ref() {
            Some(sender) => sender.try_send(payload),
            None => return DeliverResult::NoInbox,
        };
        match outcome {
            Ok(()) => DeliverResult::Accepted,
            Err(mpsc::error::TrySendError::Closed(_)) => {
                entry.inbox = None;
                DeliverResult::Closed
            }
            Err(mpsc::error::TrySendError::Full(_)) => DeliverResult::Closed,
        }
    }

    /// Takes the run's inbox receiver, exactly once.
    ///
    /// Returns `None` for an unknown run and for a second call; the receiver is
    /// handed out only to the one consumer that drains the run's payloads.
    #[must_use]
    pub fn take_inbox(&self, run_id: &str) -> Option<mpsc::Receiver<InteractPayload>> {
        let mut inner = self.inner.lock();
        let entry = inner.runs.get_mut(run_id)?;
        if entry.inbox_taken {
            return None;
        }
        let receiver = entry.inbox_rx.take()?;
        entry.inbox_taken = true;
        Some(receiver)
    }

    /// Answers a pending interaction request.
    ///
    /// The request id is looked up in the hub's pending index first; the run's
    /// session resolves it, and the id is removed from the pending set on success
    /// (a session error is propagated verbatim and leaves the id pending). When
    /// stream observation has not caught up yet, every attached session is tried
    /// in sorted run-id order and the first acceptance wins; sessions that reject
    /// the id (including sessions that do not accept interaction responses) are
    /// ignored.
    ///
    /// # Errors
    ///
    /// The session's own error for an indexed request, and
    /// [`Error::NotFound`] (`"interaction '<id>' is not pending on any active
    /// run"`) when no run accepts the id.
    pub async fn answer(&self, response: InteractionResponse) -> Result<()> {
        let request_id = response.request_id.clone();
        let indexed = {
            let inner = self.inner.lock();
            inner.runs.iter().find_map(|(run_id, entry)| {
                (entry.pending.contains(&request_id))
                    .then(|| {
                        entry
                            .session
                            .clone()
                            .map(|session| (run_id.clone(), session))
                    })
                    .flatten()
            })
        };

        if let Some((run_id, session)) = indexed {
            session.resolve_interaction(response).await?;
            self.clear_pending(&run_id, &request_id);
            return Ok(());
        }

        let sessions: Vec<(String, Arc<dyn Session>)> = {
            let inner = self.inner.lock();
            inner
                .runs
                .iter()
                .filter_map(|(run_id, entry)| {
                    entry
                        .session
                        .clone()
                        .map(|session| (run_id.clone(), session))
                })
                .collect()
        };
        for (run_id, session) in sessions {
            if session.resolve_interaction(response.clone()).await.is_ok() {
                self.clear_pending(&run_id, &request_id);
                return Ok(());
            }
        }
        Err(Error::not_found(format!(
            "interaction '{request_id}' is not pending on any active run"
        )))
    }

    /// Returns the sorted interaction request ids pending on `run_id`.
    #[must_use]
    pub fn pending(&self, run_id: &str) -> Vec<String> {
        self.inner
            .lock()
            .runs
            .get(run_id)
            .map(|entry| entry.pending.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Returns every registered run's sorted pending ids, sorted by run id.
    #[must_use]
    pub fn pending_all(&self) -> Vec<(String, Vec<String>)> {
        self.inner
            .lock()
            .runs
            .iter()
            .map(|(run_id, entry)| (run_id.clone(), entry.pending.iter().cloned().collect()))
            .collect()
    }

    /// Marks `run_id` paused and closes its interaction gate.
    ///
    /// A status/coordination signal only: nothing suspends the agent or team loop
    /// (in-loop gate enforcement is Python's `InteractGate`, deferred). Idempotent
    /// while already paused.
    ///
    /// # Errors
    ///
    /// [`Error::NotFound`] for an unknown id, and [`Error::Validation`]
    /// (`"run '<id>' has already been cancelled"`) after [`InteractionHub::cancel`].
    pub fn pause(&self, run_id: &str) -> Result<()> {
        {
            let mut inner = self.inner.lock();
            let entry = inner
                .runs
                .get_mut(run_id)
                .ok_or_else(|| super::not_registered(run_id))?;
            match entry.state {
                RunPhase::Paused => return Ok(()),
                RunPhase::Cancelled => return Err(super::already_cancelled(run_id)),
                RunPhase::Completed | RunPhase::Failed => {
                    return Err(super::already_finished(run_id))
                }
                RunPhase::Queued | RunPhase::Running => {}
            }
            entry.state = RunPhase::Paused;
            entry.gate_closed = true;
        }
        self.mirror(run_id, RunPhase::Paused);
        Ok(())
    }

    /// Marks `run_id` running and opens its interaction gate.
    ///
    /// A status/coordination signal only, like [`InteractionHub::pause`].
    /// Idempotent while already running.
    ///
    /// # Errors
    ///
    /// [`Error::NotFound`] for an unknown id, and [`Error::Validation`]
    /// (`"run '<id>' has already been cancelled"`) after [`InteractionHub::cancel`].
    pub fn resume(&self, run_id: &str) -> Result<()> {
        {
            let mut inner = self.inner.lock();
            let entry = inner
                .runs
                .get_mut(run_id)
                .ok_or_else(|| super::not_registered(run_id))?;
            match entry.state {
                RunPhase::Running => return Ok(()),
                RunPhase::Cancelled => return Err(super::already_cancelled(run_id)),
                RunPhase::Completed | RunPhase::Failed => {
                    return Err(super::already_finished(run_id))
                }
                RunPhase::Queued | RunPhase::Paused => {}
            }
            entry.state = RunPhase::Running;
            entry.gate_closed = false;
        }
        self.mirror(run_id, RunPhase::Running);
        Ok(())
    }

    /// Cancels a run: trips its token, settles its pending interactions, and
    /// marks it [`RunPhase::Cancelled`].
    ///
    /// The token is the run's own (`RunOptions::cancellation`), so the run path's
    /// `tokio::select!` observes it and returns [`Error::Cancelled`](openjiuwen_common::error::Error::Cancelled).
    /// Every pending interaction of the attached session is resolved with
    /// [`InteractionResponse::cancel`] carrying the reason `"run cancelled"`; the
    /// hub clears its pending set before returning. The responses are delivered on
    /// a detached task (this method has no await point), so a waiter observes its
    /// `Cancel` decision asynchronously; outside a runtime nothing is delivered
    /// and a warning is logged. Idempotent once cancelled.
    ///
    /// # Errors
    ///
    /// [`Error::NotFound`] for an unknown id.
    pub fn cancel(&self, run_id: &str) -> Result<()> {
        let deliveries = {
            let mut inner = self.inner.lock();
            let entry = inner
                .runs
                .get_mut(run_id)
                .ok_or_else(|| super::not_registered(run_id))?;
            if entry.state == RunPhase::Cancelled {
                return Ok(());
            }
            entry.cancellation.cancel();
            entry.state = RunPhase::Cancelled;
            entry.gate_closed = true;
            let ids: Vec<String> = entry.pending.iter().cloned().collect();
            entry.pending.clear();
            entry.session.clone().map(|session| (session, ids))
        };
        self.mirror_cancelled(run_id);

        if let Some((session, ids)) = deliveries {
            if !ids.is_empty() {
                super::bridge::spawn_cancellations(run_id.to_string(), session, ids);
            }
        }
        Ok(())
    }

    /// Lists every registered run, sorted by run id.
    ///
    /// A cancelled run stays listed (with [`RunPhase::Cancelled`] and a closed
    /// gate) until it is unregistered.
    #[must_use]
    pub fn list_active(&self) -> Vec<ActiveRunInfo> {
        self.inner
            .lock()
            .runs
            .iter()
            .map(|(run_id, entry)| entry.info(run_id))
            .collect()
    }

    /// Cancels every pending interaction on every attached session.
    ///
    /// Used for a hub-wide shutdown: each pending id is resolved with
    /// [`InteractionResponse::cancel`] carrying `reason`, the pending sets are
    /// cleared first, and a per-session failure (a session that already dropped
    /// its waiter) is logged and does not fail the call.
    ///
    /// # Errors
    ///
    /// Never returns an error today; the `Result` keeps the shutdown path
    /// uniform.
    pub async fn close_interactions(&self, reason: impl Into<String>) -> Result<()> {
        let reason = reason.into();
        let targets: Vec<(String, Arc<dyn Session>, Vec<String>)> = {
            let mut inner = self.inner.lock();
            inner
                .runs
                .iter_mut()
                .filter_map(|(run_id, entry)| {
                    let session = entry.session.clone()?;
                    if entry.pending.is_empty() {
                        return None;
                    }
                    let ids: Vec<String> = entry.pending.iter().cloned().collect();
                    entry.pending.clear();
                    Some((run_id.clone(), session, ids))
                })
                .collect()
        };

        for (run_id, session, ids) in targets {
            for id in ids {
                let response = InteractionResponse {
                    reason: Some(reason.clone()),
                    ..InteractionResponse::cancel(id)
                };
                if let Err(error) = session.resolve_interaction(response).await {
                    warn!(
                        target: TARGET,
                        run_id = %run_id,
                        %error,
                        "failed to cancel a pending interaction"
                    );
                }
            }
        }
        Ok(())
    }

    /// Returns the number of registered runs.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.lock().runs.len()
    }

    /// Whether no run is registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.lock().runs.is_empty()
    }

    /// Removes `request_id` from `run_id`'s pending set, when still present.
    fn clear_pending(&self, run_id: &str, request_id: &str) {
        if let Some(entry) = self.inner.lock().runs.get_mut(run_id) {
            entry.pending.remove(request_id);
        }
    }

    /// Mirrors `phase` into the monitor, logging a rejected transition.
    fn mirror(&self, run_id: &str, phase: RunPhase) {
        let Some(monitor) = self.monitor.as_ref() else {
            return;
        };
        if let Err(error) = monitor.update(run_id, phase, Value::Null) {
            warn!(
                target: TARGET,
                run_id,
                phase = ?phase,
                %error,
                "run status mirror failed"
            );
        }
    }
    /// Finalises a cancelled run on the monitor with `finish`.
    ///
    /// A terminal phase never goes through `update`, so the mirrored run gets its
    /// `finished_at` stamped and enters the retention queue. A monitor that does
    /// not track the run, or that already finished it, is logged and ignored.
    fn mirror_cancelled(&self, run_id: &str) {
        let Some(monitor) = self.monitor.as_ref() else {
            return;
        };
        match monitor.snapshot(run_id) {
            Ok(status) if !status.is_terminal() => {
                if let Err(error) = monitor.finish(run_id, RunOutcome::cancelled()) {
                    warn!(
                        target: TARGET,
                        run_id,
                        %error,
                        "run status finish failed"
                    );
                }
            }
            Ok(_) => {}
            Err(error) => {
                debug!(target: TARGET, run_id, %error, "monitor did not track the run");
            }
        }
    }
}
