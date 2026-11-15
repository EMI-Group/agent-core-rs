//! One pooled team run: its live state, the snapshot a host sees, the frames it
//! publishes, and its exactly-once finalisation.
//!
//! The entry is the single owner of a run's mutable facts. Every knob that
//! decides *what* runs (`run_id`, [`PoolKey`], label, deadline, capacities) is
//! immutable, so the only locked state is the small [`EntryState`] — phase,
//! gate, output, failure, idle stamp, and the monitor guard. The atomics
//! (`leases`, `finalised`) and the small `Mutex`es (permit, worker handle,
//! drain handle, hub binding, payload buffer) are separate so no critical
//! section has to span them.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use openjiuwen_common::error::{Error, ErrorKind, Result};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::broadcast;
use tokio::sync::Notify;
use tokio::sync::OwnedSemaphorePermit;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::debug;

use crate::hitl::{InteractPayload, InteractionHub, RunBinding};
use crate::monitor::{RunKind, RunMonitor, RunPhase, RunStart, RunStatusGuard};

use super::error::rebuild;
use super::key::PoolKey;
use super::LOG_TARGET;

/// The reason a cancelled pooled run records when nothing more specific
/// applies (the entry's token fired, Python's "run cancelled").
pub(crate) const CANCELLED_REASON: &str = "run cancelled";

/// A read-only snapshot of one pooled run.
///
/// Produced by [`TeamPool::get`](crate::pool::TeamPool::get) and
/// [`TeamPool::list_active`](crate::pool::TeamPool::list_active). It carries no
/// live [`PoolRun`](crate::pool::PoolRun), no cancellation token, and no permit,
/// so a host UI cannot mutate or keep a run alive through the snapshot.
///
/// Serialized with `snake_case` field names, every field defaulted, and unknown
/// fields tolerated; `state` uses the monitor's `snake_case` phase strings.
/// Python correspondence: `agent_teams/runtime/pool.py::ActiveTeamInfo`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ActiveTeamInfo {
    /// The run id the pool minted for this run.
    pub run_id: String,
    /// The team the run belongs to (the key's team name).
    pub team_name: String,
    /// The session the run is bound to (the key's session id).
    pub session_id: String,
    /// The human-readable label (`PoolRequest::label`, else the team name).
    pub label: String,
    /// The run's phase; terminal phases describe a retained finished run.
    pub state: RunPhase,
    /// Whether the entry's interaction gate is closed (finalised).
    pub gate_closed: bool,
    /// Whether the run holds a concurrency permit and is in flight.
    pub running: bool,
    /// How many live leases the entry has.
    pub leases: usize,
    /// How many inbound human payloads are buffered but not yet taken.
    pub queued: usize,
}

/// One frame a pooled run publishes.
///
/// Internally tagged by `kind` and `snake_case` throughout. The channel is
/// **best effort**: a subscriber that falls behind sees
/// `broadcast::error::RecvError::Lagged` and loses the missed frames, because
/// frames are never replayed. [`PoolFrame::Started`] is emitted once when the
/// run is admitted, [`PoolFrame::Payload`] once per buffered inbound payload,
/// [`PoolFrame::Output`] once per stream frame, and the
/// terminal [`PoolFrame::Completed`] / [`PoolFrame::Failed`] /
/// [`PoolFrame::Cancelled`] exactly once.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PoolFrame {
    /// The run left the queue and started executing.
    Started {
        /// The run id.
        run_id: String,
        /// The team the run belongs to.
        team_name: String,
        /// The session the run is bound to.
        session_id: String,
    },
    /// One inbound human payload reached the run's buffer.
    Payload {
        /// The payload that arrived.
        payload: InteractPayload,
    },
    /// One frame of the team's own stream, forwarded verbatim.
    Output {
        /// The value the strategy produced.
        value: Value,
    },
    /// The run finished successfully with the last streamed value.
    Completed {
        /// The run's output.
        output: Value,
    },
    /// The run finished with an error.
    Failed {
        /// The failure message (`Error::message()`, without the kind prefix).
        error: String,
    },
    /// The run was cancelled.
    Cancelled {
        /// Why it was cancelled.
        reason: String,
    },
}

/// A run failure reduced to a clonable, printable pair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PoolFailure {
    /// The crate-wide category of the originating error.
    pub(crate) kind: ErrorKind,
    /// The error payload (without the kind prefix).
    pub(crate) message: String,
}

impl PoolFailure {
    /// Reduces `error` to its kind and message.
    pub(crate) fn from_error(error: &Error) -> Self {
        Self {
            kind: error.kind(),
            message: error.message().into_owned(),
        }
    }

    /// The failure of a worker task that panicked.
    pub(crate) fn panicked() -> Self {
        Self {
            kind: ErrorKind::Validation,
            message: "run task panicked".to_owned(),
        }
    }

    /// Rebuilds the crate-wide error, preserving the kind.
    pub(crate) fn to_error(&self) -> Error {
        rebuild(self.kind, self.message.clone())
    }
}

/// The terminal verdict one pooled run finishes with.
#[derive(Debug, Clone)]
pub(crate) enum FinalOutcome {
    /// The stream ended with this value.
    Completed(Value),
    /// The stream reported an error, or the worker panicked.
    Failed(PoolFailure),
    /// The run was cancelled, with the recorded reason.
    Cancelled(String),
}

impl FinalOutcome {
    /// The phase this outcome finalises the entry as.
    fn phase(&self) -> RunPhase {
        match self {
            Self::Completed(_) => RunPhase::Completed,
            Self::Failed(_) => RunPhase::Failed,
            Self::Cancelled(_) => RunPhase::Cancelled,
        }
    }
}

/// Everything [`Entry::new`] needs; built by the pool when it creates a run.
pub(crate) struct EntryInit {
    /// The run id the pool minted.
    pub(crate) run_id: String,
    /// The key the run is pooled under.
    pub(crate) key: PoolKey,
    /// The human-readable label.
    pub(crate) label: String,
    /// The run deadline, when one applies.
    pub(crate) deadline: Option<Duration>,
    /// The inbound payload buffer capacity.
    pub(crate) payload_capacity: usize,
    /// The frame channel capacity.
    pub(crate) frame_capacity: usize,
    /// The hub the run is registered on, when the runner has one.
    pub(crate) hub: Option<Arc<InteractionHub>>,
    /// The monitor the run mirrors into, when the runner has one.
    pub(crate) monitor: Option<Arc<RunMonitor>>,
}

/// The mutable facts of one pooled run, guarded by a single short lock.
pub(crate) struct EntryState {
    /// The run's phase.
    phase: RunPhase,
    /// Whether the interaction gate is closed (paused).
    gate_closed: bool,
    /// The output of a completed run.
    output: Option<Value>,
    /// The failure of a failed run.
    failure: Option<PoolFailure>,
    /// When the entry became idle (no leases, finished or orphaned).
    idle_since: Option<tokio::time::Instant>,
    /// The finalisation sequence number used for pruning order.
    finished_seq: Option<u64>,
    /// The monitor guard, owned by the entry until finalisation.
    status: Option<RunStatusGuard>,
}

/// One pooled team run.
pub(crate) struct Entry {
    /// The run id.
    run_id: String,
    /// The key the run is pooled under.
    key: PoolKey,
    /// The human-readable label.
    label: String,
    /// The entry's own cancellation token (`hub.cancel` trips it).
    token: CancellationToken,
    /// The deadline applied to the run, when any.
    deadline: Option<Duration>,
    /// The inbound payload buffer capacity.
    payload_capacity: usize,
    /// The live state.
    state: Mutex<EntryState>,
    /// The frame channel.
    frames: broadcast::Sender<PoolFrame>,
    /// Wakes every `wait()` caller at finalisation.
    wake: Notify,
    /// How many live leases the entry has.
    leases: AtomicUsize,
    /// The concurrency permit, released at finalisation.
    permit: Mutex<Option<OwnedSemaphorePermit>>,
    /// The worker task handle, awaited by `stop`.
    worker: Mutex<Option<JoinHandle<()>>>,
    /// The payload drain task handle.
    drain: Mutex<Option<JoinHandle<()>>>,
    /// The hub registration; dropping it unregisters the run.
    binding: Mutex<Option<RunBinding>>,
    /// Whether finalisation already happened.
    finalised: AtomicBool,
    /// The hub the run is registered on, when any (`cancel`).
    hub: Option<Arc<InteractionHub>>,
    /// The buffered inbound payloads.
    payloads: Mutex<VecDeque<InteractPayload>>,
}

impl Entry {
    /// Builds an entry in [`RunPhase::Queued`] and admits it on the monitor.
    ///
    /// The monitor guard is begun here — before the hub registration — so the
    /// hub's own `track` adopts the pool's team-shaped status instead of
    /// publishing an agent-shaped one.
    pub(crate) fn new(init: EntryInit) -> Self {
        let detail = json!({"team": init.key.team_name});
        let status = RunStatusGuard::begin(
            init.monitor,
            &init.run_id,
            RunStart::new(RunKind::Team, init.label.clone())
                .with_session(init.key.session_id.clone())
                .with_detail(detail),
        );
        let (frames, _) = broadcast::channel(init.frame_capacity.max(1));
        Self {
            run_id: init.run_id,
            key: init.key,
            label: init.label,
            token: CancellationToken::new(),
            deadline: init.deadline,
            payload_capacity: init.payload_capacity.max(1),
            state: Mutex::new(EntryState {
                phase: RunPhase::Queued,
                gate_closed: false,
                output: None,
                failure: None,
                idle_since: None,
                finished_seq: None,
                status: Some(status),
            }),
            frames,
            wake: Notify::new(),
            leases: AtomicUsize::new(0),
            permit: Mutex::new(None),
            worker: Mutex::new(None),
            drain: Mutex::new(None),
            binding: Mutex::new(None),
            finalised: AtomicBool::new(false),
            hub: init.hub,
            payloads: Mutex::new(VecDeque::new()),
        }
    }

    /// The run id.
    pub(crate) fn run_id(&self) -> &str {
        &self.run_id
    }

    /// The key the run is pooled under.
    pub(crate) fn key(&self) -> &PoolKey {
        &self.key
    }

    /// The team name (the key's team name).
    pub(crate) fn team_name(&self) -> &str {
        &self.key.team_name
    }

    /// The session id (the key's session id).
    pub(crate) fn session_id(&self) -> &str {
        &self.key.session_id
    }

    /// The human-readable label.
    pub(crate) fn label(&self) -> &str {
        &self.label
    }

    /// The entry's cancellation token.
    pub(crate) fn token(&self) -> CancellationToken {
        self.token.clone()
    }

    /// The deadline applied to the run, when any.
    pub(crate) fn deadline(&self) -> Option<Duration> {
        self.deadline
    }

    /// The run's phase.
    pub(crate) fn phase(&self) -> RunPhase {
        self.state.lock().phase
    }

    /// Whether the run reached a terminal phase.
    pub(crate) fn is_terminal(&self) -> bool {
        self.phase().is_terminal()
    }

    /// The run's output, once it completed.
    pub(crate) fn output(&self) -> Option<Value> {
        self.state.lock().output.clone()
    }

    /// When the entry became idle, if it did.
    pub(crate) fn idle_since(&self) -> Option<tokio::time::Instant> {
        self.state.lock().idle_since
    }

    /// The finalisation sequence number, once finalised.
    pub(crate) fn finished_seq(&self) -> Option<u64> {
        self.state.lock().finished_seq
    }

    /// The read-only snapshot of this run.
    pub(crate) fn info(&self) -> ActiveTeamInfo {
        let state = self.state.lock();
        ActiveTeamInfo {
            run_id: self.run_id.clone(),
            team_name: self.key.team_name.clone(),
            session_id: self.key.session_id.clone(),
            label: self.label.clone(),
            state: state.phase,
            gate_closed: state.gate_closed,
            running: self.permit.lock().is_some(),
            leases: self.leases.load(Ordering::SeqCst),
            queued: self.payloads.lock().len(),
        }
    }

    /// Subscribes to the run's frames (best effort: earlier frames are lost).
    pub(crate) fn subscribe(&self) -> broadcast::Receiver<PoolFrame> {
        self.frames.subscribe()
    }

    /// Publishes one frame; a channel without subscribers is not an error.
    pub(crate) fn emit(&self, frame: PoolFrame) {
        let _ = self.frames.send(frame);
    }

    /// Installs the concurrency permit the run holds until finalisation.
    pub(crate) fn install_permit(&self, permit: OwnedSemaphorePermit) {
        *self.permit.lock() = Some(permit);
    }

    /// Installs the worker task handle.
    pub(crate) fn install_worker(&self, worker: JoinHandle<()>) {
        *self.worker.lock() = Some(worker);
    }

    /// Takes the worker task handle.
    pub(crate) fn take_worker(&self) -> Option<JoinHandle<()>> {
        self.worker.lock().take()
    }

    /// Installs the payload drain task handle.
    pub(crate) fn install_drain(&self, drain: JoinHandle<()>) {
        *self.drain.lock() = Some(drain);
    }

    /// Takes (and aborts) the payload drain task handle.
    pub(crate) fn abort_drain(&self) {
        if let Some(drain) = self.drain.lock().take() {
            drain.abort();
        }
    }

    /// Installs the hub registration.
    pub(crate) fn install_binding(&self, binding: RunBinding) {
        *self.binding.lock() = Some(binding);
    }

    /// How many live leases the entry has.
    pub(crate) fn leases(&self) -> usize {
        self.leases.load(Ordering::SeqCst)
    }

    /// Adds one live lease.
    pub(crate) fn add_lease(&self) {
        self.leases.fetch_add(1, Ordering::SeqCst);
    }

    /// Releases one live lease.
    ///
    /// Reaching zero stamps the entry idle, which is what lets the reaper evict
    /// a finished run (or stop an orphaned one) after `idle_timeout`.
    pub(crate) fn release_lease(&self) {
        let previous = self.leases.fetch_sub(1, Ordering::SeqCst);
        if previous <= 1 {
            self.leases.store(0, Ordering::SeqCst);
            self.state.lock().idle_since = Some(tokio::time::Instant::now());
        }
    }

    /// Marks the run [`RunPhase::Running`] on the entry and the monitor.
    pub(crate) fn mark_running(&self) {
        let detail = json!({"team": self.key.team_name});
        let mut state = self.state.lock();
        if let Some(status) = state.status.as_mut() {
            status.running(detail);
        }
        state.phase = RunPhase::Running;
        state.gate_closed = false;
    }

    /// Trips the run's token so the worker finishes it as cancelled.
    pub(crate) fn cancel(&self) {
        if self.is_terminal() {
            return;
        }
        self.token.cancel();
        self.cancel_on_hub();
    }

    /// Pushes one inbound payload, dropping the oldest on overflow.
    pub(crate) fn push_payload(&self, payload: InteractPayload) {
        {
            let mut payloads = self.payloads.lock();
            if payloads.len() >= self.payload_capacity {
                payloads.pop_front();
                debug!(
                    target: LOG_TARGET,
                    run_id = %self.run_id,
                    capacity = self.payload_capacity,
                    "dropped the oldest inbound payload"
                );
            }
            payloads.push_back(payload.clone());
        }
        self.emit(PoolFrame::Payload { payload });
    }

    /// Returns a snapshot of the buffered inbound payloads.
    pub(crate) fn payloads(&self) -> Vec<InteractPayload> {
        self.payloads.lock().iter().cloned().collect()
    }

    /// Drains the buffered inbound payloads.
    pub(crate) fn take_payloads(&self) -> Vec<InteractPayload> {
        self.payloads.lock().drain(..).collect()
    }

    /// Waits for the run to finish and returns its output, or its failure.
    ///
    /// # Errors
    ///
    /// The rebuilt failure of a [`FinalOutcome::Failed`] run, and
    /// [`Error::Cancelled`] for a cancelled one.
    pub(crate) async fn wait(&self) -> Result<Value> {
        loop {
            let notified = self.wake.notified();
            tokio::pin!(notified);
            // Registered before the state check, so a finalisation that lands in
            // between cannot be lost.
            notified.as_mut().enable();
            if let Some(result) = self.result() {
                return result;
            }
            notified.await;
        }
    }

    /// The run's result, once it finished.
    fn result(&self) -> Option<Result<Value>> {
        let state = self.state.lock();
        match state.phase {
            RunPhase::Completed => Some(Ok(state.output.clone().unwrap_or(Value::Null))),
            RunPhase::Failed => Some(Err(state
                .failure
                .as_ref()
                .map_or_else(|| Error::validation("run failed"), PoolFailure::to_error))),
            RunPhase::Cancelled => Some(Err(Error::cancelled())),
            RunPhase::Queued | RunPhase::Running | RunPhase::Paused => None,
        }
    }

    /// Cancels the entry and finalises it as [`RunPhase::Cancelled`].
    ///
    /// Idempotent: a run that already finalised is left untouched.
    pub(crate) fn cancel_and_finalise(&self, seq: u64, reason: &str) {
        self.token.cancel();
        self.cancel_on_hub();
        self.finalise(&FinalOutcome::Cancelled(reason.to_owned()), seq);
    }

    /// Finalises the entry, exactly once.
    ///
    /// The order is deliberate: the state is written first, then the monitor is
    /// finished, then the terminal frame is published, then the hub binding is
    /// dropped (so `InteractionHub::unregister` sees an already-terminal run and
    /// cannot overwrite the pool's phase), then the permit is released, then
    /// every `wait()` caller is woken. Returns `false` when the entry had
    /// already finalised.
    pub(crate) fn finalise(&self, outcome: &FinalOutcome, seq: u64) -> bool {
        if self.finalised.swap(true, Ordering::SeqCst) {
            return false;
        }
        let status = {
            let mut state = self.state.lock();
            state.phase = outcome.phase();
            match outcome {
                FinalOutcome::Completed(value) => state.output = Some(value.clone()),
                FinalOutcome::Failed(failure) => state.failure = Some(failure.clone()),
                FinalOutcome::Cancelled(_) => {}
            }
            state.gate_closed = true;
            state.finished_seq = Some(seq);
            if self.leases.load(Ordering::SeqCst) == 0 {
                state.idle_since = Some(tokio::time::Instant::now());
            }
            state.status.take()
        };

        let detail = json!({"team": self.key.team_name});
        if let Some(mut status) = status {
            match outcome {
                FinalOutcome::Completed(_) => {
                    status.ok(detail);
                }
                FinalOutcome::Failed(failure) => status.fail(&failure.to_error(), detail),
                FinalOutcome::Cancelled(reason) => {
                    status.cancelled(json!({"reason": reason, "team": self.key.team_name}));
                }
            }
        }

        match outcome {
            FinalOutcome::Completed(value) => {
                self.emit(PoolFrame::Completed {
                    output: value.clone(),
                });
            }
            FinalOutcome::Failed(failure) => {
                self.emit(PoolFrame::Failed {
                    error: failure.message.clone(),
                });
            }
            FinalOutcome::Cancelled(reason) => {
                self.emit(PoolFrame::Cancelled {
                    reason: reason.clone(),
                });
            }
        }

        drop(self.binding.lock().take());
        *self.permit.lock() = None;
        self.wake.notify_waiters();
        true
    }

    /// Mirrors a cancellation into the hub, best effort.
    fn cancel_on_hub(&self) {
        let Some(hub) = self.hub.as_ref() else {
            return;
        };
        if let Err(error) = hub.cancel(&self.run_id) {
            debug!(target: LOG_TARGET, run_id = %self.run_id, %error, "hub cancel failed");
        }
    }
}

impl Drop for Entry {
    /// Cancels the entry's token and aborts the payload drain.
    ///
    /// Dropping a [`JoinHandle`] never aborts its task, so the drain is aborted
    /// explicitly; the worker needs no abort because it holds its own `Arc` of
    /// the entry and therefore cannot be dropped while it runs.
    fn drop(&mut self) {
        self.token.cancel();
        if let Some(drain) = self.drain.lock().take() {
            drain.abort();
        }
    }
}

/// The keys to evict so at most `retain` finished entries stay pooled.
///
/// `finished` holds `(finalisation sequence, key)` pairs of terminal entries
/// **with no live lease**; the oldest (smallest sequence) go first. A bound that
/// cannot be met (every candidate is leased) is reported by returning fewer keys,
/// never by evicting a live one.
pub(crate) fn select_pruned(finished: &[(u64, PoolKey)], retain: usize) -> Vec<PoolKey> {
    if finished.len() <= retain {
        return Vec::new();
    }
    let mut ordered: Vec<&(u64, PoolKey)> = finished.iter().collect();
    ordered.sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));
    ordered
        .iter()
        .take(finished.len() - retain)
        .map(|(_, key)| key.clone())
        .collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::hitl::InteractionHub;

    fn entry(run_id: &str) -> Entry {
        Entry::new(EntryInit {
            run_id: run_id.to_owned(),
            key: PoolKey::new("crew", "s-1"),
            label: "crew".to_owned(),
            deadline: None,
            payload_capacity: 2,
            frame_capacity: 4,
            hub: None,
            monitor: None,
        })
    }

    #[test]
    fn a_new_entry_is_queued_and_unleased() {
        let entry = entry("run-1");
        assert_eq!(entry.phase(), RunPhase::Queued);
        assert!(!entry.is_terminal());
        assert_eq!(entry.leases(), 0);
        assert!(entry.output().is_none());
        assert!(entry.idle_since().is_none());
        assert_eq!(entry.info().run_id, "run-1");
        assert_eq!(entry.info().team_name, "crew");
        assert_eq!(entry.info().session_id, "s-1");
        assert!(!entry.info().running);
        assert_eq!(entry.info().queued, 0);
    }

    #[test]
    fn finalisation_is_exactly_once_and_notifies_the_state() {
        let entry = entry("run-1");
        assert!(entry.finalise(&FinalOutcome::Completed(json!({"ok": true})), 7));
        assert!(!entry.finalise(&FinalOutcome::Cancelled("late".to_owned()), 8));

        let info = entry.info();
        assert_eq!(info.state, RunPhase::Completed);
        assert!(info.gate_closed);
        assert_eq!(entry.output(), Some(json!({"ok": true})));
        assert_eq!(entry.finished_seq(), Some(7));
        assert!(entry.idle_since().is_some(), "no lease means idle at once");
    }

    #[test]
    fn a_failure_keeps_its_kind() {
        let entry = entry("run-1");
        let error = Error::timeout("run_agent_team timed out after 5ms");
        entry.finalise(&FinalOutcome::Failed(PoolFailure::from_error(&error)), 1);
        let info = entry.info();
        assert_eq!(info.state, RunPhase::Failed);
        let rebuilt = PoolFailure {
            kind: ErrorKind::Timeout,
            message: "run_agent_team timed out after 5ms".to_owned(),
        }
        .to_error();
        assert_eq!(rebuilt.kind(), ErrorKind::Timeout);
        assert_eq!(
            rebuilt.to_string(),
            "timeout: run_agent_team timed out after 5ms"
        );
    }

    #[tokio::test]
    async fn wait_resolves_on_finalisation() {
        let entry = Arc::new(entry("run-1"));
        let waiter = {
            let entry = Arc::clone(&entry);
            tokio::spawn(async move { entry.wait().await })
        };
        tokio::task::yield_now().await;
        entry.finalise(&FinalOutcome::Completed(json!("done")), 1);
        assert_eq!(
            waiter.await.expect("the waiter does not panic").unwrap(),
            json!("done")
        );
    }

    #[tokio::test]
    async fn wait_reports_a_cancellation() {
        let entry = entry("run-1");
        entry.cancel_and_finalise(1, CANCELLED_REASON);
        let error = entry.wait().await.unwrap_err();
        assert!(error.is_cancelled());
        assert_eq!(entry.info().state, RunPhase::Cancelled);
    }

    #[test]
    fn leases_are_counted_and_idle_on_the_last_release() {
        let entry = entry("run-1");
        entry.add_lease();
        entry.add_lease();
        assert_eq!(entry.info().leases, 2);
        entry.release_lease();
        assert!(entry.idle_since().is_none());
        entry.release_lease();
        assert_eq!(entry.info().leases, 0);
        assert!(entry.idle_since().is_some());
        // A defensive extra release never underflows.
        entry.release_lease();
        assert_eq!(entry.leases(), 0);
    }

    #[tokio::test]
    async fn payloads_are_bounded_dropping_the_oldest() {
        let entry = entry("run-1");
        let mut frames = entry.subscribe();
        for text in ["one", "two", "three"] {
            entry.push_payload(InteractPayload::text(text));
        }
        assert_eq!(entry.info().queued, 2, "capacity is two");
        let buffered = entry.payloads();
        assert_eq!(
            buffered
                .iter()
                .map(|payload| payload.text.clone().unwrap())
                .collect::<Vec<_>>(),
            vec!["two".to_owned(), "three".to_owned()]
        );
        let frame = frames.try_recv().expect("the first push is buffered");
        assert!(matches!(frame, PoolFrame::Payload { .. }));
        assert_eq!(entry.take_payloads().len(), 2);
        assert_eq!(entry.info().queued, 0);
        assert_eq!(entry.payloads(), Vec::new());
    }

    #[test]
    fn a_session_is_registered_on_the_hub_and_removed_at_finalisation() {
        let hub = Arc::new(InteractionHub::new());
        let entry = Entry::new(EntryInit {
            run_id: "run-1".to_owned(),
            key: PoolKey::new("crew", "s-1"),
            label: "crew".to_owned(),
            deadline: Some(Duration::from_secs(1)),
            payload_capacity: 4,
            frame_capacity: 4,
            hub: Some(Arc::clone(&hub)),
            monitor: None,
        });
        let detail = json!({"team": "crew"});
        let binding = crate::hitl::bind_run(
            Some(&hub),
            crate::hitl::ActiveRunSpec::for_execution(
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
        assert_eq!(hub.len(), 1);
        entry.finalise(&FinalOutcome::Completed(Value::Null), 1);
        assert!(hub.is_empty(), "finalisation unregistered the run");
    }

    #[test]
    fn pruning_keeps_the_newest_entries() {
        let finished = vec![
            (3, PoolKey::new("a", "s")),
            (1, PoolKey::new("b", "s")),
            (2, PoolKey::new("c", "s")),
        ];
        assert_eq!(select_pruned(&finished, 3), Vec::new());
        assert_eq!(
            select_pruned(&finished, 1),
            vec![PoolKey::new("b", "s"), PoolKey::new("c", "s")]
        );
        assert_eq!(select_pruned(&finished, 0).len(), 3);
        assert_eq!(select_pruned(&[], 0), Vec::new());
    }

    #[test]
    fn info_serializes_with_the_documented_shape() {
        let info = ActiveTeamInfo {
            run_id: "run-1".to_owned(),
            team_name: "crew".to_owned(),
            session_id: "s-1".to_owned(),
            label: "crew".to_owned(),
            state: RunPhase::Running,
            gate_closed: false,
            running: true,
            leases: 1,
            queued: 2,
        };
        let value = serde_json::to_value(&info).unwrap();
        assert_eq!(
            value,
            json!({
                "run_id": "run-1",
                "team_name": "crew",
                "session_id": "s-1",
                "label": "crew",
                "state": "running",
                "gate_closed": false,
                "running": true,
                "leases": 1,
                "queued": 2,
            })
        );
        assert_eq!(
            serde_json::from_value::<ActiveTeamInfo>(value).unwrap(),
            info
        );
        let default = ActiveTeamInfo::default();
        assert_eq!(default.state, RunPhase::Queued);
        assert_eq!(default.run_id, "");
        let partial: ActiveTeamInfo =
            serde_json::from_value(json!({"team_name": "crew", "future": 1})).unwrap();
        assert_eq!(partial.team_name, "crew");
        assert_eq!(partial.state, RunPhase::Queued);
    }

    #[test]
    fn frames_serialize_with_a_kind_tag() {
        let cases = [
            (
                PoolFrame::Started {
                    run_id: "run-1".to_owned(),
                    team_name: "crew".to_owned(),
                    session_id: "s-1".to_owned(),
                },
                "started",
            ),
            (PoolFrame::Output { value: json!(1) }, "output"),
            (
                PoolFrame::Completed {
                    output: json!({"ok": true}),
                },
                "completed",
            ),
            (
                PoolFrame::Failed {
                    error: "boom".to_owned(),
                },
                "failed",
            ),
            (
                PoolFrame::Cancelled {
                    reason: "stopped".to_owned(),
                },
                "cancelled",
            ),
        ];
        for (frame, kind) in cases {
            let value = serde_json::to_value(&frame).unwrap();
            assert_eq!(value["kind"], json!(kind));
            assert_eq!(serde_json::from_value::<PoolFrame>(value).unwrap(), frame);
        }

        let payload = PoolFrame::Payload {
            payload: InteractPayload::text("hi"),
        };
        let value = serde_json::to_value(&payload).unwrap();
        assert_eq!(value["kind"], json!("payload"));
        assert_eq!(value["payload"]["text"], json!("hi"));
        assert_eq!(serde_json::from_value::<PoolFrame>(value).unwrap(), payload);
    }
}
