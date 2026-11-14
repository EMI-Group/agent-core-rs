//! Bridge from the team-memory observer seam onto the team span tree.
//!
//! The team memory facade ([`openjiuwen_memory::team::TeamMemory`]) emits one
//! [`TeamMemoryEvent`] per observable operation through its
//! [`TeamMemoryObserver`] seam. This module turns every such event into one child span of the
//! team's root span, so team memory shows up next to the `member.*`, `round.*` and `task.*` spans
//! of the same trace instead of living in a separate log stream.
//!
//! ```text
//! team.{team_name}
//! ├── member.{member_name}
//! ├── task.{task_id}
//! └── memory.{operation}           one per observed team-memory operation
//! ```
//!
//! # What one observation produces
//!
//! [`MemorySpanObserver::observe`] builds `memory.{operation}` with
//! [`TeamSpan::child`](crate::observability::TeamSpan::child) — so it keeps the root's trace id,
//! records the root as its parent and inherits the root's attributes at creation time — and adds
//! the event's identity on top: [`AT_MEMORY_TEAM_ID`], [`AT_MEMORY_OPERATION`],
//! [`AT_MEMORY_OUTCOME`], [`AT_MEMORY_COUNT`], plus [`AT_MEMORY_MEMBER_ID`], [`AT_MEMORY_KIND`]
//! and [`AT_MEMORY_ERROR`] only when the event carries them (an absent *or* empty value never
//! becomes an attribute). The same fields are emitted as one structured `tracing` event at target
//! `openjiuwen_agent_teams::memory`: `debug!` on [`TeamMemoryOutcome::Ok`], `warn!` on
//! [`TeamMemoryOutcome::Failed`].
//!
//! # Feature split
//!
//! The baseline build records the span as a `tracing` event only. With the `otel` feature the
//! observer also carries an [`OtelSpanSink`](crate::observability::otel::OtelSpanSink) —
//! [`MemorySpanObserver::with_sink`] — and appends one
//! [`SpanRecord`](crate::observability::otel::SpanRecord) per event, so an in-process consumer can
//! inspect what the team memory did without an exporter.
//!
//! Real OTLP export and the `opentelemetry` SDK are deliberately *not* part of this crate:
//! [`OTLP_EXPORT_AVAILABLE`](crate::observability::OTLP_EXPORT_AVAILABLE) stays `false` because
//! `openjiuwen-extensions`, which owns the workspace's `OTel` integration, sits in the same
//! dependency layer as this crate and therefore cannot be depended on from here. A host that wants
//! OTLP forwards these `tracing` events/spans through the extensions layer it already runs; this
//! observer is the seam such an adapter subscribes to.
//!
//! # Cost
//!
//! The observer is `Send + Sync`, infallible (`observe` returns `()`) and non-blocking: no
//! `.await`, and the only lock is the short one inside the sink (which never spans other work).
//! Attaching is purely additive — the memory facade's results, ordering and cancellation
//! behaviour are unchanged.

use std::sync::Arc;

use openjiuwen_memory::team::{
    TeamMemoryEvent, TeamMemoryKind, TeamMemoryObserver, TeamMemoryOutcome,
};
use serde_json::Value;
use tracing::{debug, warn};

#[cfg(feature = "otel")]
use super::otel::OtelSpanSink;
use super::span::TeamSpan;

/// The tracing target of every log line this module emits.
const LOG_TARGET: &str = "openjiuwen_agent_teams::memory";

/// Attribute key holding the memory scope's team id (the event's own `team_id` field).
pub const AT_MEMORY_TEAM_ID: &str = "team_id";
/// Attribute key holding the observed team-memory operation (`search`, `write`, …).
pub const AT_MEMORY_OPERATION: &str = "operation";
/// Attribute key holding how the operation ended (`ok` or `failed`).
pub const AT_MEMORY_OUTCOME: &str = "outcome";
/// Attribute key holding how many records the operation touched.
pub const AT_MEMORY_COUNT: &str = "count";
/// Attribute key holding the member the operation was scoped to, when it was member-scoped.
pub const AT_MEMORY_MEMBER_ID: &str = "member_id";
/// Attribute key holding the kind of the record involved, when the operation has a single kind.
pub const AT_MEMORY_KIND: &str = "kind";
/// Attribute key holding the error text of a failed operation.
pub const AT_MEMORY_ERROR: &str = "error";

/// Turns every team-memory event into one `memory.{operation}` child of a team span.
///
/// One observer observes one team run: the root span is built once from the team name
/// ([`TeamSpan::root_for`](crate::observability::TeamSpan::root_for)) and every event derives a
/// child from it, so a trace never leaks between teams. Build it with [`MemorySpanObserver::new`]
/// (or the [`memory_observer`] shortcut), optionally attach a sink with
/// [`MemorySpanObserver::with_sink`] on the `otel` build, and hand it to the facade with
/// `TeamMemory::with_observer(Arc::new(observer))`.
#[derive(Debug, Clone)]
pub struct MemorySpanObserver {
    /// The team root every memory span is a child of.
    root: TeamSpan,
    /// The optional in-memory span sink (only on the `otel` build).
    #[cfg(feature = "otel")]
    sink: Option<Arc<OtelSpanSink>>,
}

impl MemorySpanObserver {
    /// Builds an observer for `team_name`, rooted at `team.{team_name}`.
    ///
    /// Each call mints a fresh trace, so two observers never share identity and no global
    /// registry is involved.
    #[must_use]
    pub fn new(team_name: impl Into<String>) -> Self {
        Self {
            root: TeamSpan::root_for(team_name),
            #[cfg(feature = "otel")]
            sink: None,
        }
    }

    /// Attaches `sink` to receive one [`SpanRecord`](crate::observability::otel::SpanRecord) per
    /// observed event.
    ///
    /// Replacing the sink drops the previous one; the records already written to it stay where
    /// they are.
    #[cfg(feature = "otel")]
    #[must_use]
    pub fn with_sink(mut self, sink: Arc<OtelSpanSink>) -> Self {
        self.sink = Some(sink);
        self
    }

    /// Returns the team root span this observer derives every memory span from.
    #[must_use]
    pub fn root(&self) -> &TeamSpan {
        &self.root
    }

    /// Returns the trace id shared by the root and every memory span of this observer.
    #[must_use]
    pub fn trace_id(&self) -> &str {
        self.root.trace_id()
    }

    /// Builds the `memory.{operation}` child span of `event`.
    ///
    /// The child inherits the root's attributes ([`TeamSpan::child`]), then stamps the event's
    /// identity on top; absent optionals are left out rather than written as empty strings.
    fn span_for(&self, event: &TeamMemoryEvent) -> TeamSpan {
        let mut span = self
            .root
            .child(format!("memory.{}", event.operation.as_str()))
            .with_attribute(AT_MEMORY_TEAM_ID, Value::from(event.team_id.as_str()))
            .with_attribute(AT_MEMORY_OPERATION, Value::from(event.operation.as_str()))
            .with_attribute(AT_MEMORY_OUTCOME, Value::from(event.outcome.as_str()))
            .with_attribute(AT_MEMORY_COUNT, Value::from(event.count));
        if let Some(member_id) = event.member_id.as_deref().filter(|id| !id.is_empty()) {
            span = span.with_attribute(AT_MEMORY_MEMBER_ID, Value::from(member_id));
        }
        if let Some(kind) = event.kind.as_ref().map(TeamMemoryKind::as_str) {
            if !kind.is_empty() {
                span = span.with_attribute(AT_MEMORY_KIND, Value::from(kind));
            }
        }
        if let Some(error) = event.error.as_deref().filter(|text| !text.is_empty()) {
            span = span.with_attribute(AT_MEMORY_ERROR, Value::from(error));
        }
        span
    }
}

impl TeamMemoryObserver for MemorySpanObserver {
    fn observe(&self, event: &TeamMemoryEvent) {
        let span = self.span_for(event);
        let member = span
            .attribute(AT_MEMORY_MEMBER_ID)
            .and_then(Value::as_str)
            .unwrap_or("");
        let kind = span
            .attribute(AT_MEMORY_KIND)
            .and_then(Value::as_str)
            .unwrap_or("");
        match event.outcome {
            TeamMemoryOutcome::Ok => debug!(
                target: LOG_TARGET,
                trace_id = span.trace_id(),
                span_id = span.span_id(),
                team = %event.team_id,
                operation = event.operation.as_str(),
                outcome = event.outcome.as_str(),
                count = event.count,
                member_id = member,
                kind,
                "team memory operation"
            ),
            TeamMemoryOutcome::Failed => {
                let error = span
                    .attribute(AT_MEMORY_ERROR)
                    .and_then(Value::as_str)
                    .unwrap_or("");
                warn!(
                    target: LOG_TARGET,
                    trace_id = span.trace_id(),
                    span_id = span.span_id(),
                    team = %event.team_id,
                    operation = event.operation.as_str(),
                    outcome = event.outcome.as_str(),
                    count = event.count,
                    member_id = member,
                    kind,
                    error,
                    "team memory operation failed"
                );
            }
        }
        #[cfg(feature = "otel")]
        if let Some(sink) = &self.sink {
            sink.record(&span, span.name.clone());
        }
    }
}

/// Builds an observer for `team_name`; the sink form is attached afterwards with
/// [`MemorySpanObserver::with_sink`] on the `otel` build.
#[must_use]
pub fn memory_observer(team_name: impl Into<String>) -> MemorySpanObserver {
    MemorySpanObserver::new(team_name)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::observability::span::{AT_TEAM_ID, AT_TEAM_NAME, OJ_AGENT_MODE};
    use openjiuwen_memory::record::CallOptions;
    use openjiuwen_memory::team::{
        NewTeamMemory, TeamMemory, TeamMemoryOperation, TeamMemoryScope, TeamSearchOptions,
    };
    use parking_lot::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn text(value: &str) -> Value {
        Value::String(value.to_owned())
    }

    #[test]
    fn an_ok_event_builds_one_child_span_with_the_documented_attributes() {
        let observer = memory_observer("alpha");
        let event = TeamMemoryEvent::new(TeamMemoryOperation::Write, "team-1").with_count(2);
        let span = observer.span_for(&event);

        assert_eq!(span.name, "memory.write");
        assert_eq!(span.trace_id(), observer.trace_id());
        assert_eq!(
            span.parent,
            Some(observer.root().context.clone()),
            "the memory span is a child of the team root"
        );
        assert_ne!(span.span_id(), observer.root().span_id());
        // Inherited from the root at creation time.
        assert_eq!(span.attribute(AT_TEAM_NAME), Some(&text("alpha")));
        assert_eq!(span.attribute(AT_TEAM_ID), Some(&text("alpha")));
        assert_eq!(span.attribute(OJ_AGENT_MODE), Some(&text("team")));
        // Stamped from the event.
        assert_eq!(span.attribute(AT_MEMORY_TEAM_ID), Some(&text("team-1")));
        assert_eq!(span.attribute(AT_MEMORY_OPERATION), Some(&text("write")));
        assert_eq!(span.attribute(AT_MEMORY_OUTCOME), Some(&text("ok")));
        assert_eq!(span.attribute(AT_MEMORY_COUNT), Some(&Value::from(2_u64)));
        // Absent identity fields stay absent.
        assert!(span.attribute(AT_MEMORY_MEMBER_ID).is_none());
        assert!(span.attribute(AT_MEMORY_KIND).is_none());
        assert!(span.attribute(AT_MEMORY_ERROR).is_none());
    }

    #[test]
    fn span_names_follow_memory_dot_operation() {
        let observer = MemorySpanObserver::new("alpha");
        for operation in [
            TeamMemoryOperation::Search,
            TeamMemoryOperation::Write,
            TeamMemoryOperation::Forget,
            TeamMemoryOperation::Clear,
            TeamMemoryOperation::Prune,
            TeamMemoryOperation::Extract,
        ] {
            let event = TeamMemoryEvent::new(operation, "team-1");
            let span = observer.span_for(&event);
            assert_eq!(span.name, format!("memory.{}", operation.as_str()));
            assert_eq!(
                span.attribute(AT_MEMORY_OPERATION),
                Some(&text(operation.as_str()))
            );
        }
    }

    #[test]
    fn a_failed_event_carries_the_failed_outcome_and_the_error() {
        let observer = MemorySpanObserver::new("alpha");
        let event = TeamMemoryEvent::new(TeamMemoryOperation::Search, "team-1")
            .with_member("alice")
            .failed("store unavailable");
        let span = observer.span_for(&event);
        assert_eq!(span.name, "memory.search");
        assert_eq!(span.attribute(AT_MEMORY_OUTCOME), Some(&text("failed")));
        assert_eq!(
            span.attribute(AT_MEMORY_ERROR),
            Some(&text("store unavailable"))
        );
        assert_eq!(span.attribute(AT_MEMORY_MEMBER_ID), Some(&text("alice")));
    }

    #[test]
    fn optional_identity_attributes_are_present_only_when_set_and_non_empty() {
        let observer = MemorySpanObserver::new("alpha");
        let event = TeamMemoryEvent::new(TeamMemoryOperation::Write, "team-1")
            .with_member("alice")
            .with_kind(TeamMemoryKind::Decision)
            .with_count(3);
        let span = observer.span_for(&event);
        assert_eq!(span.attribute(AT_MEMORY_MEMBER_ID), Some(&text("alice")));
        assert_eq!(span.attribute(AT_MEMORY_KIND), Some(&text("decision")));
        assert_eq!(span.attribute(AT_MEMORY_COUNT), Some(&Value::from(3_u64)));

        // Empty strings never become attributes.
        let blank = TeamMemoryEvent::new(TeamMemoryOperation::Forget, "team-1")
            .with_member("")
            .failed("");
        let span = observer.span_for(&blank);
        assert!(span.attribute(AT_MEMORY_MEMBER_ID).is_none());
        assert!(span.attribute(AT_MEMORY_ERROR).is_none());
        assert_eq!(span.attribute(AT_MEMORY_OUTCOME), Some(&text("failed")));
    }

    #[test]
    fn observing_two_teams_keeps_two_distinct_traces() {
        let alpha = memory_observer("alpha");
        let beta = memory_observer("beta");
        assert_ne!(
            alpha.trace_id(),
            beta.trace_id(),
            "each observer mints its own trace"
        );

        let event = TeamMemoryEvent::new(TeamMemoryOperation::Search, "alpha-scope");
        let from_alpha = alpha.span_for(&event);
        let from_beta = beta.span_for(&event);
        assert_eq!(from_alpha.trace_id(), alpha.trace_id());
        assert_eq!(from_beta.trace_id(), beta.trace_id());
        assert_ne!(from_alpha.trace_id(), from_beta.trace_id());
        assert_eq!(from_alpha.attribute(AT_TEAM_NAME), Some(&text("alpha")));
        assert_eq!(from_beta.attribute(AT_TEAM_NAME), Some(&text("beta")));

        // Two observers of the same team name are still independent traces.
        let second = memory_observer("alpha");
        assert_ne!(second.trace_id(), alpha.trace_id());
    }

    #[test]
    fn the_observer_is_send_and_sync_and_usable_as_a_trait_object() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MemorySpanObserver>();
        let shared: Arc<dyn TeamMemoryObserver> = Arc::new(memory_observer("alpha"));
        shared.observe(&TeamMemoryEvent::new(TeamMemoryOperation::Prune, "team-1"));
    }

    /// Counts the events the facade hands over while delegating to the bridge under test.
    #[derive(Debug)]
    struct CountingObserver {
        inner: MemorySpanObserver,
        seen: AtomicUsize,
        last: Mutex<Option<TeamMemoryEvent>>,
    }

    impl CountingObserver {
        fn new(inner: MemorySpanObserver) -> Self {
            Self {
                inner,
                seen: AtomicUsize::new(0),
                last: Mutex::new(None),
            }
        }
    }

    impl TeamMemoryObserver for CountingObserver {
        fn observe(&self, event: &TeamMemoryEvent) {
            self.seen.fetch_add(1, Ordering::SeqCst);
            *self.last.lock() = Some(event.clone());
            self.inner.observe(event);
        }
    }

    #[tokio::test]
    async fn real_facade_events_reach_the_bridge() {
        let observer = Arc::new(CountingObserver::new(memory_observer("alpha")));
        let memory = TeamMemory::in_memory().with_observer(observer.clone());
        memory
            .remember(
                NewTeamMemory::decision("team-1", "use sqlite for the shared store")
                    .with_member("alice"),
                &CallOptions::new(),
            )
            .await
            .expect("remember succeeds");
        memory
            .search(
                &TeamMemoryScope::member("team-1", "alice"),
                "sqlite",
                &TeamSearchOptions::default(),
            )
            .await
            .expect("search succeeds");

        assert_eq!(observer.seen.load(Ordering::SeqCst), 2);
        let last = observer.last.lock().clone().expect("an observed event");
        assert_eq!(last.operation, TeamMemoryOperation::Search);
        assert_eq!(last.outcome, TeamMemoryOutcome::Ok);
    }

    #[cfg(feature = "otel")]
    mod otel_bridge {
        use super::*;

        #[tokio::test]
        async fn the_sink_holds_one_record_per_event_with_the_shared_trace() {
            let sink = Arc::new(OtelSpanSink::new(8));
            let observer = Arc::new(memory_observer("alpha").with_sink(sink.clone()));
            let trace_id = observer.trace_id().to_owned();
            let memory = TeamMemory::in_memory().with_observer(observer);
            memory
                .remember(
                    NewTeamMemory::decision("team-1", "ship behind a flag").with_member("alice"),
                    &CallOptions::new(),
                )
                .await
                .expect("remember succeeds");
            memory
                .forget("team-1", "missing-id")
                .await
                .expect("forget is idempotent");

            let records = sink.records();
            assert_eq!(records.len(), 2, "{records:?}");
            assert_eq!(records[0].name, "memory.write");
            assert_eq!(records[1].name, "memory.forget");
            for record in &records {
                assert_eq!(record.trace_id, trace_id);
                assert_eq!(
                    record.attributes.get(AT_MEMORY_TEAM_ID),
                    Some(&text("team-1"))
                );
            }
            assert_eq!(sink.evicted(), 0);
        }

        #[test]
        fn a_capacity_zero_sink_stores_nothing() {
            let sink = Arc::new(OtelSpanSink::new(0));
            let observer = memory_observer("alpha").with_sink(sink.clone());
            observer.observe(&TeamMemoryEvent::new(TeamMemoryOperation::Search, "team-1"));
            observer.observe(&TeamMemoryEvent::new(TeamMemoryOperation::Write, "team-1"));
            assert!(sink.is_empty());
            assert_eq!(sink.len(), 0);
            assert_eq!(sink.evicted(), 2);
        }

        #[test]
        fn the_sink_records_the_failed_outcome_and_the_error() {
            let sink = Arc::new(OtelSpanSink::new(4));
            let observer = memory_observer("alpha").with_sink(sink.clone());
            observer.observe(
                &TeamMemoryEvent::new(TeamMemoryOperation::Prune, "team-1").failed("boom"),
            );
            let records = sink.records();
            assert_eq!(records.len(), 1);
            assert_eq!(records[0].name, "memory.prune");
            assert_eq!(
                records[0].attributes.get(AT_MEMORY_OUTCOME),
                Some(&text("failed"))
            );
            assert_eq!(
                records[0].attributes.get(AT_MEMORY_ERROR),
                Some(&text("boom"))
            );
            assert_eq!(
                records[0].parent_span_id,
                Some(observer.root().span_id().to_owned())
            );
        }
    }
}
