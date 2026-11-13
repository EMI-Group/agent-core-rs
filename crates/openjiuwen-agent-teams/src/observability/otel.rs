//! Feature-gated OTel-style sink: structured `tracing` events plus a bounded in-memory span buffer.
//!
//! Python parity source: `openjiuwen/agent_teams/observability/` (v0.1.20) — the span records of
//! `monitor_handler.py` and the turn decoration of `rail.py`, without the OpenTelemetry SDK. This
//! module is compiled only with the `otel` feature ([`crate::observability`] re-exports it then).
//!
//! # What the feature does
//!
//! Enabling `otel` adds exactly two things on top of the hook seam:
//!
//! - [`OtelHooks`] turns every [`ObservabilityHooks`] callback into **one structured
//!   `tracing::info!` event** carrying the trace/span identity and the event's identity fields
//!   (team, member, task, status, …).
//! - the same callback appends one [`SpanRecord`] to its [`OtelSpanSink`], a bounded in-memory
//!   ring so a test or an in-process consumer can inspect what the runtime emitted without an SDK.
//!
//! That is all: [`OTLP_EXPORT_AVAILABLE`](crate::observability::OTLP_EXPORT_AVAILABLE) stays
//! `false`. Real OTLP export, the `opentelemetry` SDK and exporter configuration are deferred to
//! `openjiuwen-extensions`, which owns the workspace's `OTel` integration and sits in the same
//! dependency layer as this crate (so it cannot be depended on from here).
//!
//! # Sink semantics
//!
//! The sink is process-local: it lives on the `Arc` a caller shares with [`OtelHooks`], never in
//! global state, so two teams (or two tests) never see each other's spans. Retention is a ring
//! bound: a positive capacity keeps the newest `capacity` records and drops the oldest first,
//! counting every drop in [`OtelSpanSink::evicted`]. A capacity of `0` disables retention
//! entirely — nothing is stored, and every [`OtelSpanSink::record`] increments `evicted`.
//! [`OtelSpanSink::clear`] empties the buffer but leaves the cumulative counter untouched.
//!
//! Everything here is infallible: no function returns `Result`.

use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::hooks::{
    DelegationEvent, MemberObservableEvent, ObservabilityHooks, RoundObservableEvent,
    TaskTransitionEvent, TurnMetrics,
};
use super::span::TeamSpan;
use crate::schema::RoundOutcome;

/// Returns the Python wire string of a round outcome (`RoundOutcome` has no `as_str` in `schema`).
const fn round_outcome_str(outcome: RoundOutcome) -> &'static str {
    match outcome {
        RoundOutcome::Started => "started",
        RoundOutcome::Finished => "finished",
        RoundOutcome::Aborted => "aborted",
        RoundOutcome::Failed => "failed",
    }
}

/// One recorded span, flattened for an in-memory sink or a downstream exporter.
///
/// The record is a plain snapshot: identity ids are copied out of the [`TeamSpan`]'s
/// [`SpanContext`](crate::observability::SpanContext), `parent_span_id` is the parent's span id
/// when the span had a parent, `attributes` is the span's attribute map verbatim, and `timestamp`
/// is the span's *start* time in milliseconds since the Unix epoch (not the record time).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SpanRecord {
    /// Span name (`member.spawned`, `task.transition`, … — the callback that recorded it).
    pub name: String,
    /// Trace identity shared by every span of one team run.
    pub trace_id: String,
    /// Identity of the span that produced this record.
    pub span_id: String,
    /// Span id of the parent span, absent for a root span.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_span_id: Option<String>,
    /// Attributes in effect on the span, copied at record time.
    pub attributes: BTreeMap<String, Value>,
    /// Span start time in milliseconds since the Unix epoch.
    pub timestamp: u64,
}

/// Mutable state behind the sink's lock.
#[derive(Debug, Default)]
struct SinkState {
    /// Retained records, oldest first.
    records: VecDeque<SpanRecord>,
    /// Cumulative count of records dropped by the bound (or refused by capacity `0`).
    evicted: u64,
}

/// A bounded, process-local in-memory buffer of [`SpanRecord`]s.
///
/// See the module docs for the ring/`capacity == 0` rules. The sink is `Send + Sync` and safe to
/// share through an [`Arc`] between a hook and the code that inspects it.
#[derive(Debug)]
pub struct OtelSpanSink {
    capacity: usize,
    state: Mutex<SinkState>,
}

impl OtelSpanSink {
    /// Creates a sink retaining at most `capacity` records.
    ///
    /// A capacity of `0` disables retention: [`record`](Self::record) stores nothing and counts
    /// every call as evicted.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            state: Mutex::new(SinkState::default()),
        }
    }

    /// Records `span` under `name`, dropping the oldest record when the bound is exceeded.
    ///
    /// The record copies the span's trace/span identity, its parent span id, its attributes and its
    /// start time; see [`SpanRecord`]. With capacity `0` the record is dropped immediately and
    /// [`evicted`](Self::evicted) still advances, so a disabled sink stays observable.
    pub fn record(&self, span: &TeamSpan, name: impl Into<String>) {
        let record = SpanRecord {
            name: name.into(),
            trace_id: span.context.trace_id.clone(),
            span_id: span.context.span_id.clone(),
            parent_span_id: span.parent.as_ref().map(|parent| parent.span_id.clone()),
            attributes: span.attributes.clone(),
            timestamp: span.started_at,
        };
        let mut state = self.state.lock();
        if self.capacity == 0 {
            state.evicted = state.evicted.saturating_add(1);
            return;
        }
        state.records.push_back(record);
        while state.records.len() > self.capacity {
            state.records.pop_front();
            state.evicted = state.evicted.saturating_add(1);
        }
    }

    /// Returns a snapshot of the retained records, oldest first.
    #[must_use]
    pub fn records(&self) -> Vec<SpanRecord> {
        self.state.lock().records.iter().cloned().collect()
    }

    /// Returns how many records are retained right now.
    #[must_use]
    pub fn len(&self) -> usize {
        self.state.lock().records.len()
    }

    /// Returns `true` when no record is retained (always `true` at capacity `0`).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.state.lock().records.is_empty()
    }

    /// Drops every retained record.
    ///
    /// The cumulative [`evicted`](Self::evicted) counter is *not* reset: it counts drops, not
    /// buffer contents, so it keeps its value across a `clear`.
    pub fn clear(&self) {
        self.state.lock().records.clear();
    }

    /// Returns the cumulative number of records dropped by the bound (or refused at capacity `0`).
    #[must_use]
    pub fn evicted(&self) -> u64 {
        self.state.lock().evicted
    }
}

/// The `otel` hook: one structured `tracing::info!` plus one [`SpanRecord`] per callback.
///
/// Cloneable and cheap to share; every clone writes to the same [`OtelSpanSink`] because the sink
/// lives behind the shared `Arc` returned by [`sink`](Self::sink).
#[derive(Debug, Clone)]
pub struct OtelHooks {
    sink: Arc<OtelSpanSink>,
}

impl OtelHooks {
    /// Creates a hook writing into `sink`.
    #[must_use]
    pub fn new(sink: Arc<OtelSpanSink>) -> Self {
        Self { sink }
    }

    /// Returns the sink this hook records into.
    #[must_use]
    pub fn sink(&self) -> &Arc<OtelSpanSink> {
        &self.sink
    }
}

impl ObservabilityHooks for OtelHooks {
    fn on_member_spawned(&self, span: &TeamSpan, event: &MemberObservableEvent) {
        tracing::info!(
            trace_id = span.trace_id(),
            span_id = span.span_id(),
            team = %event.team_name,
            member = %event.member_name,
            role = event.role.as_str(),
            status = event.status.as_str(),
            "otel member.spawned"
        );
        self.sink.record(span, "member.spawned");
    }

    fn on_member_stopped(&self, span: &TeamSpan, event: &MemberObservableEvent) {
        tracing::info!(
            trace_id = span.trace_id(),
            span_id = span.span_id(),
            team = %event.team_name,
            member = %event.member_name,
            role = event.role.as_str(),
            status = event.status.as_str(),
            "otel member.stopped"
        );
        self.sink.record(span, "member.stopped");
    }

    fn on_round_started(&self, span: &TeamSpan, event: &RoundObservableEvent) {
        tracing::info!(
            trace_id = span.trace_id(),
            span_id = span.span_id(),
            team = %event.team_name,
            member = %event.member_name,
            round = event.round,
            outcome = round_outcome_str(event.outcome),
            "otel round.started"
        );
        self.sink.record(span, "round.started");
    }

    fn on_round_finished(&self, span: &TeamSpan, event: &RoundObservableEvent) {
        tracing::info!(
            trace_id = span.trace_id(),
            span_id = span.span_id(),
            team = %event.team_name,
            member = %event.member_name,
            round = event.round,
            outcome = round_outcome_str(event.outcome),
            "otel round.finished"
        );
        self.sink.record(span, "round.finished");
    }

    fn on_task_transition(&self, span: &TeamSpan, event: &TaskTransitionEvent) {
        tracing::info!(
            trace_id = span.trace_id(),
            span_id = span.span_id(),
            team = %event.team_name,
            task = %event.task_id,
            from = event.from.as_str(),
            to = event.to.as_str(),
            actor = ?event.actor,
            "otel task.transition"
        );
        self.sink.record(span, "task.transition");
    }

    fn on_delegation(&self, span: &TeamSpan, event: &DelegationEvent) {
        tracing::info!(
            trace_id = span.trace_id(),
            span_id = span.span_id(),
            team = %event.team_name,
            from = %event.from_member,
            to = %event.to_member,
            task = ?event.task_id,
            "otel delegation"
        );
        self.sink.record(span, "delegation");
    }

    fn on_turn_metrics(&self, span: &TeamSpan, metrics: &TurnMetrics) {
        tracing::info!(
            trace_id = span.trace_id(),
            span_id = span.span_id(),
            team = %metrics.team_name,
            member = %metrics.member_name,
            turn = metrics.turn,
            input_tokens = metrics.input_tokens,
            output_tokens = metrics.output_tokens,
            tool_calls = metrics.tool_calls,
            duration_ms = metrics.duration_ms,
            "otel turn.metrics"
        );
        self.sink.record(span, "turn.metrics");
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::observability::span::{AT_MEMBER_NAME, AT_TEAM_NAME};
    use crate::observability::TeamObservability;
    use crate::schema::{MemberStatus, RoundOutcome, TaskStatus, TeamRole};
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
            from: TaskStatus::Pending,
            to: TaskStatus::InProgress,
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
            turn: 3,
            input_tokens: 120,
            output_tokens: 42,
            tool_calls: 2,
            duration_ms: 900,
        }
    }

    fn named_span(name: &str) -> TeamSpan {
        TeamSpan::root(name)
    }

    #[test]
    fn sink_retains_oldest_first_up_to_capacity() {
        let sink = OtelSpanSink::new(2);
        assert!(sink.is_empty());
        assert_eq!(sink.len(), 0);

        sink.record(&named_span("a"), "first");
        assert_eq!(sink.len(), 1);
        assert!(!sink.is_empty());

        sink.record(&named_span("b"), "second");
        sink.record(&named_span("c"), "third");

        let names: Vec<String> = sink
            .records()
            .into_iter()
            .map(|record| record.name)
            .collect();
        assert_eq!(names, vec!["second".to_owned(), "third".to_owned()]);
        assert_eq!(sink.len(), 2);
        assert_eq!(sink.evicted(), 1);
    }

    #[test]
    fn capacity_zero_retains_nothing_and_counts_every_record() {
        let sink = OtelSpanSink::new(0);
        sink.record(&named_span("a"), "first");
        sink.record(&named_span("b"), "second");

        assert_eq!(
            sink.records(),
            Vec::<SpanRecord>::new(),
            "no record is retained at capacity 0"
        );
        assert!(sink.is_empty());
        assert_eq!(sink.len(), 0);
        assert_eq!(sink.evicted(), 2);
    }

    #[test]
    fn record_copies_identity_attributes_and_start_time() {
        let sink = OtelSpanSink::new(4);
        let root = TeamSpan::root_for("alpha").with_attribute("custom", json!(1));
        let child = root
            .child("member.dev")
            .with_attribute(AT_MEMBER_NAME, json!("dev"))
            .with_attribute(AT_TEAM_NAME, json!("beta"));

        sink.record(&root, "root");
        sink.record(&child, "Member Spawned");

        let records = sink.records();
        let root_record = &records[0];
        assert_eq!(root_record.name, "root");
        assert_eq!(root_record.trace_id, root.context.trace_id);
        assert_eq!(root_record.span_id, root.context.span_id);
        assert_eq!(root_record.parent_span_id, None);
        assert_eq!(root_record.attributes, root.attributes);
        assert_eq!(root_record.timestamp, root.started_at);

        let child_record = &records[1];
        assert_eq!(child_record.trace_id, root.trace_id());
        assert_eq!(child_record.span_id, child.span_id());
        assert_eq!(
            child_record.parent_span_id,
            Some(root.context.span_id.clone())
        );
        assert_eq!(
            child_record.attributes.get("custom"),
            Some(&json!(1)),
            "inherited keys survive"
        );
        assert_eq!(
            child_record.attributes.get(AT_MEMBER_NAME),
            Some(&json!("dev"))
        );
        assert_eq!(
            child_record.attributes.get(AT_TEAM_NAME),
            Some(&json!("beta")),
            "a child override wins over the inherited key"
        );
        assert_eq!(child_record.timestamp, child.started_at);
    }

    #[test]
    fn clear_drops_records_but_keeps_the_evicted_counter() {
        let sink = OtelSpanSink::new(1);
        sink.record(&named_span("a"), "first");
        sink.record(&named_span("b"), "second");
        assert_eq!(sink.len(), 1);
        assert_eq!(sink.evicted(), 1);

        sink.clear();
        assert_eq!(
            sink.records(),
            Vec::<SpanRecord>::new(),
            "no record is retained at capacity 0"
        );
        assert!(sink.is_empty());
        assert_eq!(sink.evicted(), 1);

        sink.record(&named_span("c"), "third");
        assert_eq!(sink.len(), 1);
        assert_eq!(sink.evicted(), 1);
    }

    #[test]
    fn span_record_round_trips_and_tolerates_unknown_fields() {
        let record = SpanRecord {
            name: "task.transition".to_owned(),
            trace_id: "trace".to_owned(),
            span_id: "span".to_owned(),
            parent_span_id: Some("parent".to_owned()),
            attributes: BTreeMap::from([("task.id".to_owned(), json!("t-1"))]),
            timestamp: 42,
        };
        let text = serde_json::to_string(&record).unwrap();
        let back: SpanRecord = serde_json::from_str(&text).unwrap();
        assert_eq!(back, record);

        let back: SpanRecord =
            serde_json::from_str(r#"{"name":"n","extra":{"nested":true}}"#).unwrap();
        assert_eq!(back.name, "n");
        assert_eq!(back.trace_id, "");
        assert_eq!(back.parent_span_id, None);
        assert!(back.attributes.is_empty());
        assert_eq!(back.timestamp, 0);
    }

    #[test]
    fn span_record_omits_absent_parent_and_keeps_a_present_one() {
        let root = SpanRecord {
            name: "root".to_owned(),
            ..SpanRecord::default()
        };
        let value = serde_json::to_value(&root).unwrap();
        assert!(
            value.get("parent_span_id").is_none(),
            "an absent parent is skipped"
        );

        let child = SpanRecord {
            parent_span_id: Some("parent".to_owned()),
            ..root
        };
        let value = serde_json::to_value(&child).unwrap();
        assert_eq!(value["parent_span_id"], json!("parent"));
    }

    #[test]
    fn round_outcome_str_matches_the_python_wire_values() {
        for (outcome, wire) in [
            (RoundOutcome::Started, "started"),
            (RoundOutcome::Finished, "finished"),
            (RoundOutcome::Aborted, "aborted"),
            (RoundOutcome::Failed, "failed"),
        ] {
            assert_eq!(round_outcome_str(outcome), wire);
            assert_eq!(json!(wire), serde_json::to_value(outcome).unwrap());
        }
    }

    #[test]
    fn hooks_record_one_span_per_callback() {
        let sink = Arc::new(OtelSpanSink::new(16));
        let hooks = OtelHooks::new(sink.clone());
        assert!(
            Arc::ptr_eq(hooks.sink(), &sink),
            "sink() returns the shared Arc"
        );

        let team_span = TeamSpan::root_for("alpha");
        let member_span = team_span.child("member.dev");
        let round_span = member_span.child("round.dev.2");
        let task_span = team_span.child("task.t-1");

        hooks.on_member_spawned(&member_span, &member_event());
        hooks.on_member_stopped(&member_span, &member_event());
        hooks.on_round_started(&round_span, &round_event(RoundOutcome::Started));
        hooks.on_round_finished(&round_span, &round_event(RoundOutcome::Finished));
        hooks.on_task_transition(&task_span, &task_event());
        hooks.on_delegation(&team_span, &delegation_event());
        hooks.on_turn_metrics(&member_span, &metrics());

        let names: Vec<String> = sink
            .records()
            .into_iter()
            .map(|record| record.name)
            .collect();
        assert_eq!(
            names,
            vec![
                "member.spawned".to_owned(),
                "member.stopped".to_owned(),
                "round.started".to_owned(),
                "round.finished".to_owned(),
                "task.transition".to_owned(),
                "delegation".to_owned(),
                "turn.metrics".to_owned(),
            ]
        );
        assert_eq!(sink.len(), 7);
        assert_eq!(sink.evicted(), 0);
    }

    #[test]
    fn hooks_wired_through_the_facade_record_spans() {
        let sink = Arc::new(OtelSpanSink::new(8));
        let hooks = OtelHooks::new(sink.clone());
        let hook: Arc<dyn ObservabilityHooks> = Arc::new(hooks);
        let observability = TeamObservability::with_hooks("alpha", vec![hook]);
        let root = observability.start_team(Some("s-1"));

        let member_span = observability.member_span("dev", TeamRole::Teammate);
        observability.on_member_spawned(&member_span, &member_event());

        let task_span = observability.task_span("t-1");
        observability.on_task_transition(&task_span, &task_event());

        let round_span = observability.round_span("dev", 2);
        observability.on_turn_metrics(&round_span, &metrics());

        let records = sink.records();
        let names: Vec<&str> = records.iter().map(|record| record.name.as_str()).collect();
        assert_eq!(
            names,
            vec!["member.spawned", "task.transition", "turn.metrics"]
        );
        assert_eq!(records[0].trace_id, root.trace_id());
        assert_eq!(
            records[0].parent_span_id,
            Some(root.span_id().to_owned()),
            "facade spans are children of the team root"
        );
        assert_eq!(records[1].span_id, task_span.span_id());
        assert_eq!(records[2].span_id, round_span.span_id());
    }
}
