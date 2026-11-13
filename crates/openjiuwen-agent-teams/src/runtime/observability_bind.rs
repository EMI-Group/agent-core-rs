//! One place that fans lifecycle facts out to observability hooks and the monitor stream.
//!
//! [`TeamSignals`] is built once per team and shared by the agent, the backend and every member
//! observer. Every method is infallible and best-effort:
//!
//! * observability hooks are plain synchronous callbacks and may not fail a run;
//! * monitor events are pushed with the non-blocking [`MonitorEmitter`](crate::monitor::MonitorEmitter)'s `try_emit`, so a slow or
//!   absent reader never stalls the leader loop;
//! * with no emitter installed and no hooks, every call is a no-op.
//!
//! The wrapper exists so wiring code never has to know which signal goes where; the mapping is
//! pinned here (and covered by `wiring_tests`).
use std::sync::Arc;

use parking_lot::RwLock;

use crate::monitor::{
    BackpressurePolicy, MonitorEmitter, MonitorEvent, MonitorReceiver, StatusStream,
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
    /// Emits one monitor event best-effort; an absent or closed queue is silently ignored.
    pub fn emit(&self, event: MonitorEvent) {
        let emitter = self.emitter.read().clone();
        if let Some(emitter) = emitter {
            let _ = emitter.try_emit(event);
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
            .finish_non_exhaustive()
    }
}
