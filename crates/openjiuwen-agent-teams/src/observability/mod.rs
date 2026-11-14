//! OTel-style span and hook seam for the team runtime.
//!
//! Python parity source: `openjiuwen/agent_teams/observability/` (v0.1.20) — the span identity
//! helpers and team-trace entry points of `span_context.py`, the turn decoration of `rail.py` and
//! the setup/acquire lifecycle of `setup.py`. This module keeps the *observability seam* and drops
//! the SDK: teams emit a small in-process span tree plus typed hook events, and a caller decides
//! where they go.
//!
//! # The span tree
//!
//! One team run owns one trace, rooted at `team.{team_name}`
//! ([`TeamSpan::root_for`](crate::observability::TeamSpan::root_for)). The runtime derives four
//! kinds of children from it:
//!
//! ```text
//! team.{team_name}
//! ├── member.{member_name}          one per spawned member
//! │   └── round.{member}.{round}    one per member round
//! ├── task.{task_id}                one per task-board entry
//! └── memory.{operation}            one per observed team-memory operation
//! ```
//!
//! [`TeamObservability`] holds the root of the team it
//! observes, so two teams running at once never share a trace and a hooked test can build the
//! facade per case. The `memory.*` branch comes from
//! [`MemorySpanObserver`], the bridge onto the team-memory observer seam
//! ([`openjiuwen_memory::team::TeamMemoryObserver`]).
//!
//! # Attribute inheritance
//!
//! Propagation is *creation-time inheritance*: a child span copies its parent's attribute map when
//! it is created ([`TeamSpan::child`]). A later
//! `with_attribute` on the child overrides the inherited key on the child only; there is no global
//! parent chain and no lazy walk, which is what keeps spans cheap, `Send` and safe to hand to a
//! hook on another task.
//!
//! # Hook fan-out
//!
//! [`ObservabilityHooks`] is infallible and every
//! callback has a no-op default. The facade clones its `Arc` list, then calls the hooks
//! synchronously in registration order — never holding the root lock, so a hook may call back into
//! the facade. An empty hook list is a silent no-op, which is why
//! [`TeamObservability::new`] is always usable. [`NoopHooks`] is the explicit "record
//! nothing" hook, and [`RecordingHooks`] is the in-memory recorder that tests
//! assert against.
//!
//! Nothing in this module is fallible: no function returns `Result`.
//!
//! # Feature split
//!
//! The baseline build ships the span/hook seam only: events are structured values, and anything
//! that wants them subscribes with a hook. The `otel` feature adds an in-memory span sink — spans
//! are also emitted as structured `tracing` events so a subscriber can consume them without an
//! SDK. [`MemorySpanObserver`] follows the same split: it always emits one `tracing` event per
//! observed team-memory operation, and only with `otel` does it also carry a sink
//! (`MemorySpanObserver::with_sink`).
//!
//! The `otlp` feature adds a *second* team-memory bridge,
//! [`OtlpMemoryObserver`], which forwards every
//! event into the real `OTel` pipeline through `openjiuwen-extensions`'
//! `OtelTraceSink` (the feature pulls that crate in as an optional dependency).
//!
//! # Deferrals
//!
//! - Real OTLP export is wired for the team-memory bridge only: with `otlp`,
//!   [`OtlpMemoryObserver`] drives the `openjiuwen-extensions` exporter
//!   (`OTLP/HTTP`, memory, console, …). Everything else in this crate — the member/round/task
//!   spans, the hook fan-out and [`MemorySpanObserver`] — stays on the in-memory/`tracing` seam,
//!   which is why [`OTLP_EXPORT_AVAILABLE`] describes the *baseline* build and remains `false`.
//!   The memory bridge's spans form their own trace in the exporter, because the in-process
//!   [`TeamSpan`] tree is plain data rather than an `OTel` span.
//! - Python's `monitor_handler.py` (a `LoggingSpanProcessor` that renders spans as log lines),
//!   `setup.py` (process-global tracer acquisition/release) and the tool/LLM span stack built by
//!   `rail.py` are not ported; `rail.py`'s turn decoration maps onto
//!   [`TurnMetrics`] plus the member/round/task spans above.
//! - There is no process-global tracer or span registry: every [`TeamObservability`] value is
//!   local to the team that owns it, and the extensions sink owns its private provider.

pub mod hooks;
pub mod memory;
#[cfg(feature = "otel")]
pub mod otel;
#[cfg(feature = "otlp")]
pub mod otlp;
pub mod span;

pub use hooks::{
    DelegationEvent, MemberObservableEvent, NoopHooks, ObservabilityHooks, ObservableEvent,
    RecordedEvent, RecordingHooks, RoundObservableEvent, TaskTransitionEvent, TeamObservability,
    TurnMetrics,
};
pub use memory::{
    memory_observer, MemorySpanObserver, AT_MEMORY_COUNT, AT_MEMORY_ERROR, AT_MEMORY_KIND,
    AT_MEMORY_MEMBER_ID, AT_MEMORY_OPERATION, AT_MEMORY_OUTCOME, AT_MEMORY_TEAM_ID,
};
#[cfg(feature = "otel")]
pub use otel::{OtelHooks, OtelSpanSink, SpanRecord};
#[cfg(feature = "otlp")]
pub use otlp::{otlp_memory_observer, OtlpMemoryObserver};
pub use span::{
    SpanContext, TeamSpan, AT_MEMBER_NAME, AT_MEMBER_ROLE, AT_TASK_ID, AT_TEAM_ID, AT_TEAM_NAME,
    GEN_AI_CONVERSATION_ID, OJ_AGENT_MODE,
};

/// Whether the baseline build of this crate exports spans over OTLP. Always `false` in every
/// build.
///
/// It describes the crate-wide promise: without a feature the `otel` one only enables structured
/// `tracing` events plus an in-memory sink, and the same seam carries the runtime's
/// member/round/task spans. The opt-in `otlp` feature upgrades the team-memory bridge alone — it
/// adds `OtlpMemoryObserver`, which drives the real `OTel` exporter owned by
/// `openjiuwen-extensions` — so this constant stays `false` even when that bridge is compiled in.
pub const OTLP_EXPORT_AVAILABLE: bool = false;

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    #[cfg(not(feature = "otel"))]
    use crate::schema::{MemberStatus, TeamRole};
    #[cfg(not(feature = "otel"))]
    use std::sync::Arc;

    #[test]
    fn otlp_export_is_never_available() {
        let export_available = OTLP_EXPORT_AVAILABLE;
        assert!(!export_available, "the baseline build never exports OTLP");
    }

    #[cfg(feature = "otlp")]
    #[test]
    fn the_otlp_feature_upgrades_only_the_memory_bridge() {
        assert!(
            !OTLP_EXPORT_AVAILABLE,
            "the crate-wide promise is still no exporter"
        );
        let observer = otlp_memory_observer("s-1").expect("the memory provider builds");
        assert!(observer.sink().is_recording());
        assert_eq!(observer.session_id(), Some("s-1"));
    }

    #[cfg(not(feature = "otel"))]
    #[test]
    fn facade_records_events_without_the_otel_feature() {
        let export_available = OTLP_EXPORT_AVAILABLE;
        assert!(
            !export_available,
            "the `otel` feature only adds an in-memory sink"
        );
        let recorder = Arc::new(RecordingHooks::new());
        let observability = TeamObservability::with_hooks("alpha", vec![recorder.clone()]);
        let root = observability.start_team(Some("s-1"));
        assert_eq!(observability.root_span(), Some(root.clone()));

        let event = MemberObservableEvent {
            team_name: "alpha".to_owned(),
            member_name: "dev".to_owned(),
            role: TeamRole::Teammate,
            status: MemberStatus::Ready,
        };
        observability.on_member_spawned(&observability.member_span("dev", event.role), &event);
        assert_eq!(recorder.kinds(), vec!["member_spawned"]);
        assert!(recorder.events()[0]
            .span
            .attribute(AT_MEMBER_NAME)
            .is_some());

        let round = observability.round_span("dev", 1);
        observability.on_round_started(
            &round,
            &RoundObservableEvent {
                team_name: "alpha".to_owned(),
                member_name: "dev".to_owned(),
                round: 1,
                outcome: crate::schema::RoundOutcome::Started,
            },
        );
        assert_eq!(recorder.kinds(), vec!["member_spawned", "round_started"]);

        observability.finish_team();
        assert!(observability.root_span().is_none());
    }
}
