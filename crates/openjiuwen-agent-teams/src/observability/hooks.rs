//! Observation hooks and the per-team observability facade.
//!
//! Python parity source: `openjiuwen/agent_teams/observability/rail.py` (the rail that decorates a
//! member turn with team/member/task attributes) and the team-trace entry points of
//! `span_context.py` (`get_or_create_team_span`, `finalize_trace`). The Python version drives an
//! OpenTelemetry `Tracer`; this port keeps the *seam* and drops the SDK: a hook receives the span
//! and the typed event, and decides what to do with them.
//!
//! # Three kinds of "event" names
//!
//! - [`ObservableEvent::kind`] is the **payload** kind: `"member"`, `"round"`, `"task"`,
//!   `"delegation"`, `"turn"` — the serde tag of the flattened payload.
//! - [`RecordedEvent::kind`] is the **callback** kind recorded by
//!   [`RecordingHooks`]: `"member_spawned"`,
//!   `"member_stopped"`, `"round_started"`, `"round_finished"`, `"task_transition"`,
//!   `"delegation"`, `"turn_metrics"`. One callback fires per recorded event, so the callback kind
//!   is the exact one; the payload kind is coarser (a round event is either a start or a finish).
//! - [`TeamObservability`] exposes one method per callback and fans it out to every registered
//!   hook in registration order.
//!
//! # Fan-out contract
//!
//! [`ObservabilityHooks`] is infallible: no callback returns `Result`, so nothing is swallowed and
//! nothing needs to be retried. [`TeamObservability`] clones its `Arc` list before dispatching and
//! never holds a lock while a hook runs, so a hook may call back into the facade. An empty hook
//! list is a silent no-op — [`TeamObservability::new`] therefore always produces a usable facade.
//!
//! Everything here is infallible: no function returns `Result`.
//!
//! # Serde
//!
//! Every payload type is `snake_case`, tolerates unknown fields and skips absent optionals.
//! [`ObservableEvent`] is an internally tagged enum (`"kind": "member" | …`), while
//! [`RecordedEvent`] is a **diagnostic** type used by tests and in-memory inspection: it derives
//! `Serialize` but not `Deserialize`, because its `kind` is a `&'static str` that could not be
//! deserialized without leaking.

use std::sync::Arc;

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::span::{
    TeamSpan, AT_MEMBER_NAME, AT_MEMBER_ROLE, AT_TASK_ID, AT_TEAM_NAME, GEN_AI_CONVERSATION_ID,
};
use crate::schema::{MemberStatus, RoundOutcome, TaskStatus, TeamRole};

/// A member lifecycle observation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct MemberObservableEvent {
    /// Name of the team the member belongs to.
    pub team_name: String,
    /// Name of the member.
    pub member_name: String,
    /// Role the member plays.
    pub role: TeamRole,
    /// Lifecycle status the observation sees.
    pub status: MemberStatus,
}

/// A member round observation, delivered on round start and on round finish.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct RoundObservableEvent {
    /// Name of the team the member belongs to.
    pub team_name: String,
    /// Name of the member that ran the round.
    pub member_name: String,
    /// One-based round counter within the member's run.
    pub round: u32,
    /// How the round ended (or [`RoundOutcome::Started`] for a start observation).
    pub outcome: RoundOutcome,
}

/// A task-board status transition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct TaskTransitionEvent {
    /// Name of the team that owns the task.
    pub team_name: String,
    /// Task-board task id.
    pub task_id: String,
    /// Status before the transition.
    pub from: TaskStatus,
    /// Status after the transition.
    pub to: TaskStatus,
    /// Member that caused the transition, when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub actor: Option<String>,
}

/// A delegation of work from one member to another.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct DelegationEvent {
    /// Name of the team the delegation happens in.
    pub team_name: String,
    /// Member that handed the work off.
    pub from_member: String,
    /// Member that received the work.
    pub to_member: String,
    /// Task the delegation refers to, when it is tied to a board entry.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_id: Option<String>,
    /// Free-form reason the leader attached, when any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// Token, tool-call and duration metrics of one member turn.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct TurnMetrics {
    /// Name of the team the member belongs to.
    pub team_name: String,
    /// Name of the member that produced the turn.
    pub member_name: String,
    /// One-based turn counter within the member's run.
    pub turn: u32,
    /// Prompt tokens reported by the model provider.
    pub input_tokens: u64,
    /// Completion tokens reported by the model provider.
    pub output_tokens: u64,
    /// Number of tool calls the turn made.
    pub tool_calls: u32,
    /// Wall-clock duration of the turn in milliseconds.
    pub duration_ms: u64,
}

/// Callbacks the team runtime fires for observable state changes.
///
/// Every callback has a no-op default body, so an implementation overrides only what it needs.
/// The trait is infallible: a hook reports through `tracing` or its own sink, never through a
/// `Result`, and it must not block the runtime for long — callbacks are invoked synchronously, in
/// registration order, on the thread that observed the change.
pub trait ObservabilityHooks: Send + Sync {
    /// Fired after a member was spawned and reached the roster.
    fn on_member_spawned(&self, _span: &TeamSpan, _event: &MemberObservableEvent) {}

    /// Fired after a member was shut down or retired.
    fn on_member_stopped(&self, _span: &TeamSpan, _event: &MemberObservableEvent) {}

    /// Fired when a member round starts.
    fn on_round_started(&self, _span: &TeamSpan, _event: &RoundObservableEvent) {}

    /// Fired when a member round finishes, aborts or fails.
    fn on_round_finished(&self, _span: &TeamSpan, _event: &RoundObservableEvent) {}

    /// Fired after a task moved from one status to another.
    fn on_task_transition(&self, _span: &TeamSpan, _event: &TaskTransitionEvent) {}

    /// Fired when work is delegated from one member to another.
    fn on_delegation(&self, _span: &TeamSpan, _event: &DelegationEvent) {}

    /// Fired when a member turn reports its token/tool metrics.
    fn on_turn_metrics(&self, _span: &TeamSpan, _metrics: &TurnMetrics) {}
}

/// A hook that ignores every callback.
///
/// Useful as a placeholder in [`TeamObservability::with_hooks`] when a caller wants the facade but
/// no sink, and as the value a runtime stores when observability is disabled.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoopHooks;

impl ObservabilityHooks for NoopHooks {}

/// A typed observation payload, as recorded by [`RecordingHooks`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ObservableEvent {
    /// A member lifecycle observation ([`MemberObservableEvent`]).
    Member(MemberObservableEvent),
    /// A member round observation ([`RoundObservableEvent`]).
    Round(RoundObservableEvent),
    /// A task-board transition ([`TaskTransitionEvent`]).
    Task(TaskTransitionEvent),
    /// A delegation ([`DelegationEvent`]).
    Delegation(DelegationEvent),
    /// Turn metrics ([`TurnMetrics`]).
    Turn(TurnMetrics),
}

impl ObservableEvent {
    /// Returns the payload kind of this observation.
    ///
    /// This is the coarse payload tag (`"member"`, `"round"`, `"task"`, `"delegation"`, `"turn"`)
    /// and differs from the callback-specific [`RecordedEvent::kind`].
    #[must_use]
    pub const fn kind(&self) -> &'static str {
        match self {
            Self::Member(_) => "member",
            Self::Round(_) => "round",
            Self::Task(_) => "task",
            Self::Delegation(_) => "delegation",
            Self::Turn(_) => "turn",
        }
    }
}

/// One callback invocation captured by [`RecordingHooks`].
///
/// `kind` is the callback-specific name (`"member_spawned"`, `"round_finished"`, …) — not the
/// payload kind of [`ObservableEvent::kind`]. The type is diagnostic, so it serializes for
/// assertions and logs but does not deserialize (`kind` is a `&'static str`).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RecordedEvent {
    /// Callback-specific kind.
    pub kind: &'static str,
    /// Span the callback received, cloned at record time.
    pub span: TeamSpan,
    /// Typed payload the callback received.
    pub event: ObservableEvent,
}

/// The in-memory hook: records every callback so tests can assert what the runtime emitted.
///
/// Recording is `O(1)` per call and preserves call order across all hooks sharing one recorder.
#[derive(Debug, Default)]
pub struct RecordingHooks {
    events: Mutex<Vec<RecordedEvent>>,
}

impl RecordingHooks {
    /// Creates an empty recorder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns a snapshot of everything recorded so far, in call order.
    #[must_use]
    pub fn events(&self) -> Vec<RecordedEvent> {
        self.events.lock().clone()
    }

    /// Returns the callback kinds recorded so far, in call order.
    #[must_use]
    pub fn kinds(&self) -> Vec<&'static str> {
        self.events.lock().iter().map(|event| event.kind).collect()
    }

    /// Returns how many callbacks were recorded.
    #[must_use]
    pub fn len(&self) -> usize {
        self.events.lock().len()
    }

    /// Returns `true` when nothing was recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.events.lock().is_empty()
    }

    /// Drops everything recorded so far.
    pub fn clear(&self) {
        self.events.lock().clear();
    }

    /// Appends one record; the only mutation path, so call order is the record order.
    fn record(&self, kind: &'static str, span: &TeamSpan, event: ObservableEvent) {
        self.events.lock().push(RecordedEvent {
            kind,
            span: span.clone(),
            event,
        });
    }
}

impl ObservabilityHooks for RecordingHooks {
    fn on_member_spawned(&self, span: &TeamSpan, event: &MemberObservableEvent) {
        self.record(
            "member_spawned",
            span,
            ObservableEvent::Member(event.clone()),
        );
    }

    fn on_member_stopped(&self, span: &TeamSpan, event: &MemberObservableEvent) {
        self.record(
            "member_stopped",
            span,
            ObservableEvent::Member(event.clone()),
        );
    }

    fn on_round_started(&self, span: &TeamSpan, event: &RoundObservableEvent) {
        self.record("round_started", span, ObservableEvent::Round(event.clone()));
    }

    fn on_round_finished(&self, span: &TeamSpan, event: &RoundObservableEvent) {
        self.record(
            "round_finished",
            span,
            ObservableEvent::Round(event.clone()),
        );
    }

    fn on_task_transition(&self, span: &TeamSpan, event: &TaskTransitionEvent) {
        self.record(
            "task_transition",
            span,
            ObservableEvent::Task(event.clone()),
        );
    }

    fn on_delegation(&self, span: &TeamSpan, event: &DelegationEvent) {
        self.record(
            "delegation",
            span,
            ObservableEvent::Delegation(event.clone()),
        );
    }

    fn on_turn_metrics(&self, span: &TeamSpan, metrics: &TurnMetrics) {
        self.record("turn_metrics", span, ObservableEvent::Turn(metrics.clone()));
    }
}

/// The per-team observability facade: holds the root span and fans callbacks out to hooks.
///
/// The facade is the only observability object the runtime touches. It owns no global state, so
/// two teams observed at once never share a trace, and tests can build one per case.
pub struct TeamObservability {
    team_name: String,
    hooks: Vec<Arc<dyn ObservabilityHooks>>,
    root: Mutex<Option<TeamSpan>>,
}

impl TeamObservability {
    /// Creates a facade for `team_name` with no hooks.
    ///
    /// Every callback is then a silent no-op, and nothing panics — a runtime can always construct
    /// the facade and call it unconditionally.
    #[must_use]
    pub fn new(team_name: impl Into<String>) -> Self {
        Self::with_hooks(team_name, Vec::new())
    }

    /// Creates a facade for `team_name` with an initial hook list, in registration order.
    #[must_use]
    pub fn with_hooks(
        team_name: impl Into<String>,
        hooks: Vec<Arc<dyn ObservabilityHooks>>,
    ) -> Self {
        Self {
            team_name: team_name.into(),
            hooks,
            root: Mutex::new(None),
        }
    }

    /// Appends a hook at the end of the fan-out order.
    pub fn add_hook(&mut self, hook: Arc<dyn ObservabilityHooks>) {
        self.hooks.push(hook);
    }

    /// Returns the registered hooks in registration order.
    #[must_use]
    pub fn hooks(&self) -> &[Arc<dyn ObservabilityHooks>] {
        &self.hooks
    }

    /// Returns the team this facade observes.
    #[must_use]
    pub fn team_name(&self) -> &str {
        &self.team_name
    }

    /// Starts the team trace and stores its root span.
    ///
    /// The root is `team.{team_name}` ([`TeamSpan::root_for`]); when `session_id` is `Some` and
    /// non-empty it is recorded as [`GEN_AI_CONVERSATION_ID`]. Starting twice replaces the stored
    /// root, so callers start once per run.
    pub fn start_team(&self, session_id: Option<&str>) -> TeamSpan {
        let mut span = TeamSpan::root_for(self.team_name.clone());
        if let Some(session_id) = session_id.filter(|id| !id.is_empty()) {
            span =
                span.with_attribute(GEN_AI_CONVERSATION_ID, Value::String(session_id.to_owned()));
        }
        *self.root.lock() = Some(span.clone());
        span
    }

    /// Returns a clone of the stored root span, or `None` before [`start_team`](Self::start_team)
    /// and after [`finish_team`](Self::finish_team).
    #[must_use]
    pub fn root_span(&self) -> Option<TeamSpan> {
        self.root.lock().clone()
    }

    /// Ends the team trace by dropping the stored root span; a no-op when none was started.
    pub fn finish_team(&self) {
        *self.root.lock() = None;
    }

    /// Builds the span of one member: `member.{member_name}`.
    ///
    /// A child of the stored root when the team trace is live, otherwise a standalone root span
    /// whose [`AT_TEAM_NAME`] is overridden with this facade's team name (only when non-empty).
    /// Either way the span carries [`AT_MEMBER_NAME`] and [`AT_MEMBER_ROLE`].
    #[must_use]
    pub fn member_span(&self, member_name: &str, role: TeamRole) -> TeamSpan {
        self.span_for(format!("member.{member_name}"))
            .with_attribute(AT_MEMBER_NAME, Value::String(member_name.to_owned()))
            .with_attribute(AT_MEMBER_ROLE, Value::String(role.as_str().to_owned()))
    }

    /// Builds the span of one member round: `round.{member_name}.{round}`.
    ///
    /// Same parenting rules as [`member_span`](Self::member_span); always carries
    /// [`AT_MEMBER_NAME`].
    #[must_use]
    pub fn round_span(&self, member_name: &str, round: u32) -> TeamSpan {
        self.span_for(format!("round.{member_name}.{round}"))
            .with_attribute(AT_MEMBER_NAME, Value::String(member_name.to_owned()))
    }

    /// Builds the span of one task-board entry: `task.{task_id}`.
    ///
    /// Same parenting rules as [`member_span`](Self::member_span); always carries [`AT_TASK_ID`].
    #[must_use]
    pub fn task_span(&self, task_id: &str) -> TeamSpan {
        self.span_for(format!("task.{task_id}"))
            .with_attribute(AT_TASK_ID, Value::String(task_id.to_owned()))
    }

    /// Fires [`ObservabilityHooks::on_member_spawned`] on every hook, in registration order.
    pub fn on_member_spawned(&self, span: &TeamSpan, event: &MemberObservableEvent) {
        self.fan_out(|hook| hook.on_member_spawned(span, event));
    }

    /// Fires [`ObservabilityHooks::on_member_stopped`] on every hook, in registration order.
    pub fn on_member_stopped(&self, span: &TeamSpan, event: &MemberObservableEvent) {
        self.fan_out(|hook| hook.on_member_stopped(span, event));
    }

    /// Fires [`ObservabilityHooks::on_round_started`] on every hook, in registration order.
    pub fn on_round_started(&self, span: &TeamSpan, event: &RoundObservableEvent) {
        self.fan_out(|hook| hook.on_round_started(span, event));
    }

    /// Fires [`ObservabilityHooks::on_round_finished`] on every hook, in registration order.
    pub fn on_round_finished(&self, span: &TeamSpan, event: &RoundObservableEvent) {
        self.fan_out(|hook| hook.on_round_finished(span, event));
    }

    /// Fires [`ObservabilityHooks::on_task_transition`] on every hook, in registration order.
    pub fn on_task_transition(&self, span: &TeamSpan, event: &TaskTransitionEvent) {
        self.fan_out(|hook| hook.on_task_transition(span, event));
    }

    /// Fires [`ObservabilityHooks::on_delegation`] on every hook, in registration order.
    pub fn on_delegation(&self, span: &TeamSpan, event: &DelegationEvent) {
        self.fan_out(|hook| hook.on_delegation(span, event));
    }

    /// Fires [`ObservabilityHooks::on_turn_metrics`] on every hook, in registration order.
    pub fn on_turn_metrics(&self, span: &TeamSpan, metrics: &TurnMetrics) {
        self.fan_out(|hook| hook.on_turn_metrics(span, metrics));
    }

    /// Builds `name` as a child of the stored root, or as a standalone root when absent.
    fn span_for(&self, name: String) -> TeamSpan {
        let root = self.root.lock().clone();
        if let Some(root) = root {
            return root.child(name);
        }
        let span = TeamSpan::root(name);
        if self.team_name.is_empty() {
            span
        } else {
            span.with_attribute(AT_TEAM_NAME, Value::String(self.team_name.clone()))
        }
    }

    /// Dispatches one callback to every hook, in registration order.
    ///
    /// The `Arc` list is cloned *before* dispatch and the root lock is never held here, so a hook
    /// may call back into the facade (including [`start_team`](Self::start_team)) without
    /// deadlocking. An empty list does nothing.
    fn fan_out(&self, call: impl Fn(&dyn ObservabilityHooks)) {
        let hooks = self.hooks.clone();
        for hook in hooks {
            call(hook.as_ref());
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::schema::MemberStatus;
    use serde_json::json;

    fn member_event() -> MemberObservableEvent {
        MemberObservableEvent {
            team_name: "alpha".to_owned(),
            member_name: "dev".to_owned(),
            role: TeamRole::Teammate,
            status: MemberStatus::Ready,
        }
    }

    fn round_event(outcome: RoundOutcome) -> RoundObservableEvent {
        RoundObservableEvent {
            team_name: "alpha".to_owned(),
            member_name: "dev".to_owned(),
            round: 2,
            outcome,
        }
    }

    fn task_event() -> TaskTransitionEvent {
        TaskTransitionEvent {
            team_name: "alpha".to_owned(),
            task_id: "t-1".to_owned(),
            from: TaskStatus::InProgress,
            to: TaskStatus::InReview,
            actor: Some("dev".to_owned()),
        }
    }

    fn delegation_event() -> DelegationEvent {
        DelegationEvent {
            team_name: "alpha".to_owned(),
            from_member: "team_leader".to_owned(),
            to_member: "dev".to_owned(),
            task_id: Some("t-1".to_owned()),
            reason: None,
        }
    }

    fn metrics() -> TurnMetrics {
        TurnMetrics {
            team_name: "alpha".to_owned(),
            member_name: "dev".to_owned(),
            turn: 1,
            input_tokens: 10,
            output_tokens: 20,
            tool_calls: 1,
            duration_ms: 120,
        }
    }

    #[test]
    fn recording_hooks_preserve_call_order_and_counts() {
        let hooks = RecordingHooks::new();
        assert!(hooks.is_empty());
        assert_eq!(hooks.len(), 0);

        let span = TeamSpan::root_for("alpha");
        let member = member_event();
        hooks.on_member_spawned(&span, &member);
        hooks.on_round_started(&span, &round_event(RoundOutcome::Started));
        hooks.on_round_finished(&span, &round_event(RoundOutcome::Finished));
        hooks.on_task_transition(&span, &task_event());
        hooks.on_delegation(&span, &delegation_event());
        hooks.on_turn_metrics(&span, &metrics());
        hooks.on_member_stopped(&span, &member);

        assert_eq!(
            hooks.kinds(),
            vec![
                "member_spawned",
                "round_started",
                "round_finished",
                "task_transition",
                "delegation",
                "turn_metrics",
                "member_stopped",
            ]
        );
        assert_eq!(hooks.len(), 7);
        assert!(!hooks.is_empty());
        assert_eq!(hooks.events().len(), 7);
        assert_eq!(hooks.events()[0].span, span);

        hooks.clear();
        assert!(hooks.is_empty());
        assert_eq!(hooks.kinds(), Vec::<&'static str>::new());
    }

    #[test]
    fn observable_event_kinds_are_payload_level() {
        let span = TeamSpan::root("demo");
        assert_eq!(ObservableEvent::Member(member_event()).kind(), "member");
        assert_eq!(
            ObservableEvent::Round(round_event(RoundOutcome::Started)).kind(),
            "round"
        );
        assert_eq!(ObservableEvent::Task(task_event()).kind(), "task");
        assert_eq!(
            ObservableEvent::Delegation(delegation_event()).kind(),
            "delegation"
        );
        assert_eq!(ObservableEvent::Turn(metrics()).kind(), "turn");

        let hooks = RecordingHooks::new();
        hooks.on_round_started(&span, &round_event(RoundOutcome::Started));
        let recorded = hooks.events();
        assert_eq!(recorded[0].kind, "round_started");
        assert_eq!(recorded[0].event.kind(), "round");
    }

    #[test]
    fn two_recording_hooks_see_the_same_sequence() {
        let first = Arc::new(RecordingHooks::new());
        let second = Arc::new(RecordingHooks::new());
        let observability =
            TeamObservability::with_hooks("alpha", vec![first.clone(), second.clone()]);
        let span = observability.start_team(Some("s-1"));
        let member = member_event();

        observability.on_member_spawned(&span, &member);
        observability.on_task_transition(&span, &task_event());
        observability.on_turn_metrics(&span, &metrics());

        assert_eq!(first.kinds(), second.kinds());
        assert_eq!(first.events(), second.events());
        assert_eq!(first.len(), 3);
    }

    #[test]
    fn noop_hooks_are_silent() {
        let hooks = NoopHooks;
        let span = TeamSpan::root("demo");
        hooks.on_member_spawned(&span, &member_event());
        hooks.on_member_stopped(&span, &member_event());
        hooks.on_round_started(&span, &round_event(RoundOutcome::Started));
        hooks.on_round_finished(&span, &round_event(RoundOutcome::Failed));
        hooks.on_task_transition(&span, &task_event());
        hooks.on_delegation(&span, &delegation_event());
        hooks.on_turn_metrics(&span, &metrics());
    }

    #[test]
    fn facade_without_hooks_never_panics() {
        let observability = TeamObservability::new("teamless");
        assert!(observability.hooks().is_empty());
        assert_eq!(observability.team_name(), "teamless");
        assert!(observability.root_span().is_none());

        let span = observability.member_span("dev", TeamRole::Teammate);
        observability.on_member_spawned(&span, &member_event());
        observability.on_member_stopped(&span, &member_event());
        observability.on_round_started(&span, &round_event(RoundOutcome::Started));
        observability.on_round_finished(&span, &round_event(RoundOutcome::Aborted));
        observability.on_task_transition(&span, &task_event());
        observability.on_delegation(&span, &delegation_event());
        observability.on_turn_metrics(&span, &metrics());
        observability.finish_team();

        assert!(observability.root_span().is_none());
    }

    #[test]
    fn start_team_stores_root_and_records_session_id() {
        let observability = TeamObservability::new("alpha");
        let span = observability.start_team(Some("s-1"));
        assert_eq!(span.name, "team.alpha");
        assert_eq!(
            span.attribute(GEN_AI_CONVERSATION_ID),
            Some(&Value::String("s-1".to_owned()))
        );
        assert_eq!(observability.root_span(), Some(span));

        let blank = observability.start_team(Some(""));
        assert!(blank.attribute(GEN_AI_CONVERSATION_ID).is_none());
        assert!(observability
            .start_team(None)
            .attribute(GEN_AI_CONVERSATION_ID)
            .is_none());
    }

    #[test]
    fn finish_team_clears_the_root() {
        let observability = TeamObservability::new("alpha");
        assert!(observability.root_span().is_none());
        observability.start_team(None);
        assert!(observability.root_span().is_some());
        observability.finish_team();
        assert!(observability.root_span().is_none());
        observability.finish_team();
    }

    #[test]
    fn member_span_is_a_child_of_a_live_root() {
        let observability = TeamObservability::new("alpha");
        let root = observability.start_team(Some("s-1"));
        let span = observability.member_span("dev", TeamRole::Teammate);

        assert_eq!(span.name, "member.dev");
        assert_eq!(span.trace_id(), root.trace_id());
        assert_eq!(span.parent, Some(root.context.clone()));
        assert_eq!(
            span.attribute(AT_MEMBER_NAME),
            Some(&Value::String("dev".to_owned()))
        );
        assert_eq!(
            span.attribute(AT_MEMBER_ROLE),
            Some(&Value::String("teammate".to_owned()))
        );
        assert_eq!(
            span.attribute(AT_TEAM_NAME),
            Some(&Value::String("alpha".to_owned()))
        );
    }

    #[test]
    fn member_span_without_root_is_standalone_with_team_name() {
        let observability = TeamObservability::new("alpha");
        let span = observability.member_span("dev", TeamRole::Leader);
        assert!(span.parent.is_none());
        assert_eq!(span.name, "member.dev");
        assert_eq!(
            span.attribute(AT_TEAM_NAME),
            Some(&Value::String("alpha".to_owned()))
        );
        assert_eq!(
            span.attribute(AT_MEMBER_ROLE),
            Some(&Value::String("leader".to_owned()))
        );

        let unnamed = TeamObservability::new("");
        let span = unnamed.member_span("dev", TeamRole::Leader);
        // An empty team name cannot override the root default, which records the span name.
        assert_eq!(
            span.attribute(AT_TEAM_NAME),
            Some(&Value::String("member.dev".to_owned()))
        );
    }

    #[test]
    fn round_and_task_spans_carry_names_and_attributes() {
        let observability = TeamObservability::new("alpha");
        let root = observability.start_team(None);

        let round = observability.round_span("dev", 3);
        assert_eq!(round.name, "round.dev.3");
        assert_eq!(round.parent, Some(root.context.clone()));
        assert_eq!(
            round.attribute(AT_MEMBER_NAME),
            Some(&Value::String("dev".to_owned()))
        );

        let task = observability.task_span("t-9");
        assert_eq!(task.name, "task.t-9");
        assert_eq!(task.parent, Some(root.context.clone()));
        assert_eq!(
            task.attribute(AT_TASK_ID),
            Some(&Value::String("t-9".to_owned()))
        );

        let standalone_round = TeamObservability::new("beta").round_span("dev", 1);
        assert!(standalone_round.parent.is_none());
        assert_eq!(
            standalone_round.attribute(AT_TEAM_NAME),
            Some(&Value::String("beta".to_owned()))
        );

        let standalone_task = TeamObservability::new("beta").task_span("t-1");
        assert!(standalone_task.parent.is_none());
        assert_eq!(
            standalone_task.attribute(AT_TASK_ID),
            Some(&Value::String("t-1".to_owned()))
        );
    }

    #[test]
    fn add_hook_appends_in_registration_order() {
        let mut observability = TeamObservability::new("alpha");
        assert!(observability.hooks().is_empty());

        let first = Arc::new(RecordingHooks::new());
        let second = Arc::new(RecordingHooks::new());
        observability.add_hook(first.clone());
        observability.add_hook(second.clone());
        assert_eq!(observability.hooks().len(), 2);

        let span = observability.start_team(None);
        observability.on_round_finished(&span, &round_event(RoundOutcome::Finished));
        assert_eq!(first.kinds(), vec!["round_finished"]);
        assert_eq!(second.kinds(), vec!["round_finished"]);
    }

    #[test]
    fn payloads_round_trip_through_json() {
        let member = member_event();
        let back: MemberObservableEvent =
            serde_json::from_str(&serde_json::to_string(&member).unwrap()).unwrap();
        assert_eq!(back, member);

        let task = task_event();
        let back: TaskTransitionEvent =
            serde_json::from_str(&serde_json::to_string(&task).unwrap()).unwrap();
        assert_eq!(back, task);

        let delegation = delegation_event();
        let back: DelegationEvent =
            serde_json::from_str(&serde_json::to_string(&delegation).unwrap()).unwrap();
        assert_eq!(back, delegation);

        let round = round_event(RoundOutcome::Failed);
        let value = serde_json::to_value(&round).unwrap();
        assert_eq!(value["outcome"], json!("failed"));
        assert_eq!(
            serde_json::from_value::<RoundObservableEvent>(value).unwrap(),
            round
        );

        let metrics = metrics();
        let back: TurnMetrics =
            serde_json::from_str(&serde_json::to_string(&metrics).unwrap()).unwrap();
        assert_eq!(back, metrics);

        let optional = serde_json::to_value(&delegation).unwrap();
        assert!(
            optional.get("reason").is_none(),
            "absent optional is skipped"
        );
        let with_reason = DelegationEvent {
            reason: Some("handoff".to_owned()),
            ..delegation
        };
        let back: DelegationEvent =
            serde_json::from_str(&serde_json::to_string(&with_reason).unwrap()).unwrap();
        assert_eq!(back, with_reason);
    }

    #[test]
    fn observable_event_round_trips_with_internal_tag() {
        let event = ObservableEvent::Task(task_event());
        let value = serde_json::to_value(&event).unwrap();
        assert_eq!(value["kind"], json!("task"));
        assert_eq!(
            serde_json::from_value::<ObservableEvent>(value).unwrap(),
            event
        );

        let recorded = RecordedEvent {
            kind: "task_transition",
            span: TeamSpan::root_for("alpha"),
            event,
        };
        let value = serde_json::to_value(&recorded).unwrap();
        assert_eq!(value["kind"], json!("task_transition"));
        assert_eq!(value["span"]["name"], json!("team.alpha"));
    }
}
