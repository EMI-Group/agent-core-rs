//! One place that fans lifecycle facts out to observability hooks and the monitor stream.
//!
//! [`TeamSignals`] is built once per team and shared by the agent, the backend and every member
//! observer. Every method is infallible and best-effort:
//!
//! * observability hooks are plain synchronous callbacks and may not fail a run;
//! * monitor events are pushed with the non-blocking [`MonitorEmitter`](crate::monitor::MonitorEmitter)'s
//!   `try_emit` from the synchronous lifecycle methods, so a slow or absent reader never stalls the
//!   leader loop;
//! * with no emitter installed and no hooks, every call is a no-op.
//!
//! # Which emission sites support blocking
//!
//! [`BackpressurePolicy::Block`](crate::monitor::BackpressurePolicy::Block) is a *production* policy:
//! it asks for an event to survive a full queue. Blocking is only possible where the caller is an
//! `async fn` that can afford to suspend, so this module offers both modes and documents each site:
//!
//! | site | mode | what a full `Block` queue does |
//! | --- | --- | --- |
//! | [`TeamSignals::emit`] (every lifecycle hook below) | non-blocking | refuses the event, counts it in [`TeamSignals::monitor_dropped`] and logs a `warn!` |
//! | [`TeamSignals::emit_waiting`] | awaits capacity | suspends until a slot frees, counted only if the stream closed |
//! | [`TeamSignals::emit_waiting_within`] | awaits capacity, bounded | the budget expiring is reported like a refusal |
//!
//! The lifecycle hooks (`team_started`, `member_status`, `round_finished`, `task_status`, …) are
//! synchronous and are called from contexts that must not suspend — a member's status may change
//! inside a `Drop`, and the monitor must never be able to stall the team it observes. They therefore
//! use the non-blocking path, and under [`BackpressurePolicy::Block`](crate::monitor::BackpressurePolicy::Block)
//! a refusal is a *policy violation* that is counted and logged instead of being discarded: the
//! event is visibly lost, never silently. A caller that needs the lossless guarantee calls
//! [`TeamSignals::emit_waiting`] (or [`TeamSignals::emit_waiting_within`]) from its own async
//! context, where suspending is allowed and can be bounded by its turn or cancellation token.
//!
//! [`TeamSignals::monitor_dropped`] is the accounting channel: every event the fan-out could not
//! hand to the sink — a `Block` refusal, a drop-oldest eviction, a closed queue, an expired budget —
//! increments it, mirroring [`MonitorEmitter::dropped`](crate::monitor::MonitorEmitter::dropped) for
//! the emitters this team owns.
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use parking_lot::RwLock;
use tracing::{debug, warn};

use crate::monitor::{
    BackpressurePolicy, EmitOutcome, MonitorEmitter, MonitorEvent, MonitorReceiver, StatusStream,
};
use crate::observability::{
    DelegationEvent, MemberObservableEvent, ObservabilityHooks, RoundObservableEvent,
    TaskTransitionEvent, TeamObservability, TurnMetrics,
};
use crate::schema::{MemberStatus, RoundOutcome, TaskStatus, TeamRole};
use openjiuwen_harness_protocol::HarnessState;

/// Team-wide lifecycle signals: observability hooks plus the optional monitor sink.
#[derive(Clone)]
pub struct TeamSignals {
    team_name: String,
    observability: Arc<TeamObservability>,
    emitter: Arc<RwLock<Option<MonitorEmitter>>>,
    /// Monitor events the fan-out could not hand to the sink (the accounting channel).
    dropped: Arc<AtomicU64>,
}
impl TeamSignals {
    /// Builds the signals for `team_name` over `hooks` (empty means all no-ops).
    #[must_use]
    pub fn new(team_name: impl Into<String>, hooks: Vec<Arc<dyn ObservabilityHooks>>) -> Self {
        let team_name = team_name.into();
        let observability = Arc::new(TeamObservability::with_hooks(team_name.clone(), hooks));
        Self {
            team_name,
            observability,
            emitter: Arc::new(RwLock::new(None)),
            dropped: Arc::new(AtomicU64::new(0)),
        }
    }
    /// Returns the observability facade.
    #[must_use]
    pub fn observability(&self) -> Arc<TeamObservability> {
        Arc::clone(&self.observability)
    }
    /// Returns the team name the signals belong to.
    #[must_use]
    pub fn team_name(&self) -> &str {
        &self.team_name
    }
    /// Installs (or replaces) the monitor emitter.
    pub fn install_monitor(&self, emitter: MonitorEmitter) {
        *self.emitter.write() = Some(emitter);
    }
    /// Returns the installed emitter, if any.
    #[must_use]
    pub fn monitor_emitter(&self) -> Option<MonitorEmitter> {
        self.emitter.read().clone()
    }
    /// Creates a monitor queue, installs its emitter and returns the receiving end.
    ///
    /// A host attaches a [`StatusStream`] writer to the receiver; a later call replaces the
    /// emitter, so events go to the newest reader only (documented monitor policy).
    #[must_use]
    pub fn monitor_stream(&self, capacity: usize, policy: BackpressurePolicy) -> MonitorReceiver {
        let (emitter, receiver) = StatusStream::channel(capacity, policy);
        self.install_monitor(emitter);
        receiver
    }
    /// Emits one monitor event without waiting for capacity.
    ///
    /// This is the path every lifecycle hook uses: a monitoring sink must never be able to stall
    /// the team it observes, so the call never suspends.
    /// An absent emitter means nobody is watching and nothing is lost.
    /// A closed queue, a drop-oldest eviction and — under
    /// [`BackpressurePolicy::Block`](crate::monitor::BackpressurePolicy::Block) — a refusal all
    /// count in [`TeamSignals::monitor_dropped`]; the refusal is additionally logged at `warn!`
    /// level, because under a `Block` policy it is a policy violation rather than the documented
    /// lossy default.
    pub fn emit(&self, event: MonitorEvent) {
        let emitter = self.emitter.read().clone();
        if let Some(emitter) = emitter {
            let outcome = emitter.try_emit(event);
            self.record(outcome, &emitter);
        }
    }
    /// Emits one monitor event, waiting for capacity under a blocking policy.
    ///
    /// Under [`BackpressurePolicy::Block`](crate::monitor::BackpressurePolicy::Block) this
    /// suspends until the writer drains a slot or the stream closes, which is what makes the policy
    /// *real* where the caller can afford it; under
    /// [`BackpressurePolicy::DropOldest`](crate::monitor::BackpressurePolicy::DropOldest) it is
    /// [`TeamSignals::emit`] with an `.await`.
    ///
    /// The caller bounds the wait: cancel the future (or wrap it in [`TeamSignals::emit_waiting_within`])
    /// and the event is dropped by the cancelled future rather than blocking a turn forever.
    /// An absent emitter is reported as [`EmitOutcome::Closed`] and counts nothing: there is no sink
    /// to lose the event to.
    pub async fn emit_waiting(&self, event: MonitorEvent) -> EmitOutcome {
        let Some(emitter) = self.monitor_emitter() else {
            return EmitOutcome::Closed;
        };
        let outcome = emitter.emit(event).await;
        self.record(outcome, &emitter);
        outcome
    }
    /// Emits one monitor event, waiting at most `budget` for capacity.
    ///
    /// Same contract as [`TeamSignals::emit_waiting`], with the caller's turn expressed as a
    /// deadline instead of a cancellation: an expired budget is reported as
    /// [`EmitOutcome::DroppedNew`], counted in [`TeamSignals::monitor_dropped`] and logged at
    /// `warn!` level, so a sink that never drains cannot silently swallow events either.
    pub async fn emit_waiting_within(&self, event: MonitorEvent, budget: Duration) -> EmitOutcome {
        let Some(emitter) = self.monitor_emitter() else {
            return EmitOutcome::Closed;
        };
        match tokio::time::timeout(budget, emitter.emit(event)).await {
            Ok(outcome) => {
                self.record(outcome, &emitter);
                outcome
            }
            Err(_elapsed) => {
                let dropped = self.dropped.fetch_add(1, Ordering::AcqRel) + 1;
                warn!(
                    team = %self.team_name,
                    budget_ms = u64::try_from(budget.as_millis()).unwrap_or(u64::MAX),
                    dropped,
                    "monitor emission exceeded its budget on a full sink; the event was not buffered"
                );
                EmitOutcome::DroppedNew
            }
        }
    }
    /// Events this team's monitor fan-out could not hand to its sink.
    ///
    /// The documented overflow channel: an event that is not [`EmitOutcome::Sent`] increments this
    /// counter, so `emitted + dropped` still accounts for every emission attempt.
    #[must_use]
    pub fn monitor_dropped(&self) -> u64 {
        self.dropped.load(Ordering::Acquire)
    }
    /// Records one emission outcome: counts every loss and logs a refusal where the policy forbids
    /// losing events.
    fn record(&self, outcome: EmitOutcome, emitter: &MonitorEmitter) {
        match outcome {
            EmitOutcome::Sent => {}
            EmitOutcome::DroppedOldest => {
                let dropped = self.dropped.fetch_add(1, Ordering::AcqRel) + 1;
                debug!(
                    team = %self.team_name,
                    dropped,
                    "monitor queue evicted the oldest event"
                );
            }
            EmitOutcome::DroppedNew => {
                let dropped = self.dropped.fetch_add(1, Ordering::AcqRel) + 1;
                if emitter.policy() == BackpressurePolicy::Block {
                    warn!(
                        team = %self.team_name,
                        dropped,
                        capacity = emitter.capacity(),
                        "monitor emission refused by a full blocking sink; the Block policy could \
                         not be honoured on the non-blocking path"
                    );
                } else {
                    debug!(
                        team = %self.team_name,
                        dropped,
                        "monitor emission refused by a full queue"
                    );
                }
            }
            EmitOutcome::Closed => {
                let dropped = self.dropped.fetch_add(1, Ordering::AcqRel) + 1;
                debug!(
                    team = %self.team_name,
                    dropped,
                    "monitor emission dropped on a closed stream"
                );
            }
        }
    }
    /// Opens the team span (idempotent: an existing root span is reused).
    pub fn team_started(&self, session_id: Option<&str>) {
        if self.observability.root_span().is_none() {
            let _ = self.observability.start_team(session_id);
        }
        self.emit(MonitorEvent::TeamStatus {
            state: HarnessState::Running,
        });
    }
    /// Closes the team span and signals the stop.
    pub fn team_finished(&self) {
        self.observability.finish_team();
        self.emit(MonitorEvent::TeamStopped);
    }
    /// Records that `member_name` was materialised.
    pub fn member_spawned(&self, member_name: &str, role: TeamRole) {
        let span = self.observability.member_span(member_name, role);
        self.observability.on_member_spawned(
            &span,
            &MemberObservableEvent {
                team_name: self.team_name.clone(),
                member_name: member_name.to_owned(),
                role,
                status: MemberStatus::Starting,
            },
        );
        self.emit(MonitorEvent::MemberSpawned {
            member_name: member_name.to_owned(),
            role,
        });
    }
    /// Records a member lifecycle status (also the monitor `member_status` event).
    pub fn member_status(&self, member_name: &str, role: TeamRole, status: MemberStatus) {
        let span = self.observability.member_span(member_name, role);
        self.observability.on_member_spawned(
            &span,
            &MemberObservableEvent {
                team_name: self.team_name.clone(),
                member_name: member_name.to_owned(),
                role,
                status,
            },
        );
        self.emit(MonitorEvent::MemberStatus {
            member_name: member_name.to_owned(),
            status,
            task_id: None,
        });
    }
    /// Records a member stop with the human-readable reason.
    pub fn member_stopped(&self, member_name: &str, role: TeamRole, reason: &str) {
        let span = self.observability.member_span(member_name, role);
        self.observability.on_member_stopped(
            &span,
            &MemberObservableEvent {
                team_name: self.team_name.clone(),
                member_name: member_name.to_owned(),
                role,
                status: MemberStatus::Stopped,
            },
        );
        self.emit(MonitorEvent::MemberStopped {
            member_name: member_name.to_owned(),
            reason: Some(reason.to_owned()),
        });
    }
    /// Records the start of a round.
    pub fn round_started(&self, member_name: &str, round: u32) {
        let span = self.observability.round_span(member_name, round);
        self.observability.on_round_started(
            &span,
            &RoundObservableEvent {
                team_name: self.team_name.clone(),
                member_name: member_name.to_owned(),
                round,
                outcome: RoundOutcome::Started,
            },
        );
    }
    /// Records a finished round plus its (zero-token) turn metrics.
    pub fn round_finished(
        &self,
        member_name: &str,
        round: u32,
        outcome: RoundOutcome,
        duration_ms: u64,
    ) {
        let span = self.observability.round_span(member_name, round);
        self.observability.on_round_finished(
            &span,
            &RoundObservableEvent {
                team_name: self.team_name.clone(),
                member_name: member_name.to_owned(),
                round,
                outcome,
            },
        );
        self.observability.on_turn_metrics(
            &span,
            &TurnMetrics {
                team_name: self.team_name.clone(),
                member_name: member_name.to_owned(),
                turn: round,
                input_tokens: 0,
                output_tokens: 0,
                tool_calls: 0,
                duration_ms,
            },
        );
        self.emit(MonitorEvent::MemberRound {
            member_name: member_name.to_owned(),
            round,
            outcome,
        });
    }
    /// Records a task creation on both sinks.
    pub fn task_created(&self, event: &crate::schema::TaskCreatedEvent) {
        self.emit(MonitorEvent::from_task_created(event));
    }
    /// Records a task status change on both sinks.
    pub fn task_status(&self, event: &crate::schema::TaskStatusChangedEvent, actor: &str) {
        let span = self.observability.task_span(&event.task_id);
        let from = TaskStatus::default();
        self.observability.on_task_transition(
            &span,
            &TaskTransitionEvent {
                team_name: self.team_name.clone(),
                task_id: event.task_id.clone(),
                from,
                to: event.status,
                actor: Some(actor.to_owned()),
            },
        );
        self.emit(MonitorEvent::from_task_status(event));
    }
    /// Records a delegation decision (allowed or refused) for observability.
    pub fn delegation(&self, from_member: &str, to_member: &str, task_id: &str, reason: &str) {
        let span = self
            .observability
            .member_span(from_member, TeamRole::Leader);
        self.observability.on_delegation(
            &span,
            &DelegationEvent {
                team_name: self.team_name.clone(),
                from_member: from_member.to_owned(),
                to_member: to_member.to_owned(),
                task_id: Some(task_id.to_owned()),
                reason: Some(reason.to_owned()),
            },
        );
    }
}
impl std::fmt::Debug for TeamSignals {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TeamSignals")
            .field("team_name", &self.team_name)
            .field("hooks", &self.observability.hooks().len())
            .field("monitor", &self.monitor_emitter().is_some())
            .field("monitor_dropped", &self.monitor_dropped())
            .finish_non_exhaustive()
    }
}
