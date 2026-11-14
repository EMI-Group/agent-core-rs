//! OTLP forwarding bridge from the team-memory observer seam.
//!
//! [`MemorySpanObserver`](crate::observability::MemorySpanObserver) keeps team memory inside the
//! in-process span tree plus `tracing`; this module hands the same events to a real `OTel`
//! pipeline. It implements the engine's [`TeamMemoryObserver`] seam a second time and forwards
//! every [`TeamMemoryEvent`] into an [`OtelTraceSink`] — the session [`TraceSink`] that
//! `openjiuwen-extensions` owns and that maps a trace frame onto one `OTel` span, exporting it
//! through the provider the sink was built over (`OTLP/HTTP`, memory, console, …).
//!
//! ```text
//! TeamMemory ──observe(TeamMemoryEvent)──> OtlpMemoryObserver ──emit(TraceFrame)──> OtelTraceSink ──> OTLP
//! ```
//!
//! # Frame shape
//!
//! Every event becomes one **component** frame — [`TraceSpanKind::Component`] wrapped by
//! [`TraceFrame::workflow`] (`tracer_workflow` on the wire):
//!
//! | Field | Value |
//! | --- | --- |
//! | `name` | `memory.{operation}` (`memory.search`, `memory.write`, …) |
//! | `kind` | [`TraceSpanKind::Component`] |
//! | `invoke_id` | `memory.{operation}.{16 hex digits}` — a pure function of the event identity |
//! | `trace_id` | 32 lowercase hex characters, one trace per team scope |
//! | `session_id` | the observer's session id, else empty (the sink's own fallback fills it) |
//! | `status` | [`SpanStatus::Finish`] for [`TeamMemoryOutcome::Ok`], [`SpanStatus::Error`] otherwise |
//! | `error` | the event's error text, only when the event carries a non-empty one |
//! | `component_id` / `component_name` / `component_type` | the invoke id / `memory.{operation}` / `"MemoryOperation"` |
//! | `meta_data` | the attribute bag below |
//!
//! The attribute bag in `meta_data` is exactly what the in-memory bridge stamps onto its
//! `memory.{operation}` span, reusing that module's constants: [`AT_MEMORY_TEAM_ID`],
//! [`AT_MEMORY_OPERATION`], [`AT_MEMORY_OUTCOME`], [`AT_MEMORY_COUNT`], plus
//! [`AT_MEMORY_MEMBER_ID`], [`AT_MEMORY_KIND`] and [`AT_MEMORY_ERROR`] only when the event carries
//! them (an absent *or* empty value never becomes an attribute). The sink serializes the map as
//! the `openjiuwen.meta_data` attribute, so an OTLP consumer sees the same keys the in-process
//! span carries.
//!
//! ## Why the component/workflow route
//!
//! [`OtelTraceSink::emit`] dispatches on the span kind, not on the frame type: agent frames take
//! the agent path and workflow/component frames take the workflow handler. A team-memory
//! operation is a leaf unit of work with no LLM, tool or retrieval semantics, which is exactly the
//! workflow handler's generic component branch: it writes the component identity, the trace and
//! session identity, the metadata blob and a plain finish/error status, and invents no `gen_ai.*`
//! attribute. The agent path would have to claim an invoke type (prompt/chain/…) that a memory
//! operation is not.
//!
//! Two consequences are worth knowing:
//!
//! - The extensions component handler names the exported `OTel` span `component.{invoke_id}`
//!   (Python `on_call_start`'s generic component branch) — the frame's own `name` travels as
//!   `openjiuwen.workflow.component.name` and inside `openjiuwen.meta_data`. `invoke_id` therefore
//!   keeps `memory.{operation}` as its readable prefix so the exported name still says what the
//!   span is.
//! - The observer stamps `component_type = "MemoryOperation"`, which matches none of the handler's
//!   LLM/tool/retrieval component classifiers, so no `gen_ai.*` attribute is invented.
//!
//! ## The bridge's own trace
//!
//! The frame has no `parent_invoke_id`: the team trace of the in-process bridge
//! ([`TeamSpan`](crate::observability::TeamSpan)) is plain data, not an `OTel` span, so there is no
//! parent span context to attach to, and the extensions sink starts every parentless frame as its
//! own `OTel` trace. Each memory span is therefore an independent trace root in the exporter, and
//! the derived team trace id is *not* installed as the `OTel` trace id; it travels as the
//! `openjiuwen.trace.id` attribute, which lets a backend group the memory spans of one team (and
//! correlate them with the other spans of the same frame trace). Linking them into one `OTel`
//! trace would need a live parent span, which this seam does not have.
//!
//! # Feature split
//!
//! The module exists only with the crate's `otlp` feature, which pulls `openjiuwen-extensions`
//! (with its `otel` feature) in as an optional dependency. `openjiuwen-extensions` is the
//! workspace's `OTel` owner and sits one dependency layer *below* this crate, so the edge is legal
//! and no cycle can exist. [`OTLP_EXPORT_AVAILABLE`](crate::observability::OTLP_EXPORT_AVAILABLE)
//! still reports `false`: it describes the baseline promise of the whole crate (no exporter), and
//! the `otlp` feature upgrades only this bridge — the runtime's member/round/task spans keep the
//! in-memory/`tracing` seam.
//!
//! # Cost
//!
//! [`OtlpMemoryObserver::observe`] stays infallible and non-blocking: no `.await`, no lock of its
//! own, and no error path (the sink logs and drops what it cannot map). A sink that is not
//! recording — a shut-down handle, or a build without the extensions `otel` feature — makes the
//! observer a silent no-op apart from one `debug!` line.

use std::collections::BTreeMap;
use std::sync::Arc;

use openjiuwen_common::error::Result;
use openjiuwen_extensions::otel::OtelTraceSink;
use openjiuwen_memory::team::{
    TeamMemoryEvent, TeamMemoryKind, TeamMemoryObserver, TeamMemoryOutcome,
};
use openjiuwen_session::tracer::{SpanStatus, TraceFrame, TraceSink, TraceSpan, TraceSpanKind};
use serde_json::Value;
use tracing::debug;

use super::memory::{
    AT_MEMORY_COUNT, AT_MEMORY_ERROR, AT_MEMORY_KIND, AT_MEMORY_MEMBER_ID, AT_MEMORY_OPERATION,
    AT_MEMORY_OUTCOME, AT_MEMORY_TEAM_ID,
};

/// The tracing target shared with the in-memory bridge (`../memory.rs`).
const LOG_TARGET: &str = "openjiuwen_agent_teams::memory";

/// The `openjiuwen.workflow.component.type` value stamped on every forwarded span.
///
/// Deliberately free of the extensions component classifiers' substrings (`LLM`, `Tool`,
/// `Retrieval`), so the workflow handler treats the span as a generic component.
const MEMORY_COMPONENT_TYPE: &str = "MemoryOperation";

/// FNV-1a offset basis: the seed of every derived id.
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
/// Mixing constant applied between hash parts, so part boundaries stay distinct.
const PART_SEPARATOR: u64 = 0x9e37_79b9_7f4a_7c15;

/// Forwards every team-memory event to an [`OtelTraceSink`] as one `OTel` component span.
///
/// Build it with [`OtlpMemoryObserver::new`] over a sink the host already owns, with
/// [`OtlpMemoryObserver::with_sink`] (the alias that mirrors
/// [`MemorySpanObserver::with_sink`](crate::observability::MemorySpanObserver::with_sink)), or
/// with [`otlp_memory_observer`] over a private in-memory provider that never touches the network.
/// Hand it to the facade with `TeamMemory::with_observer(Arc::new(observer))`.
///
/// One observer serves one team run: the trace id is derived from the event's `team_id`, so two
/// observers of two teams never share a trace even though both may feed one sink.
#[derive(Debug, Clone)]
pub struct OtlpMemoryObserver {
    /// The extensions sink that maps a frame onto an `OTel` span.
    sink: Arc<OtelTraceSink>,
    /// Session id stamped onto every frame; `None` leaves it empty for the sink's fallback.
    session_id: Option<String>,
}

impl OtlpMemoryObserver {
    /// Builds an observer over an existing sink.
    ///
    /// The frame's session id stays empty, so whatever session id the sink itself was built with
    /// (`OtelTraceSink::with_session`) applies as the fallback.
    #[must_use]
    pub fn new(sink: Arc<OtelTraceSink>) -> Self {
        Self {
            sink,
            session_id: None,
        }
    }

    /// Builds an observer over an existing sink — the same behaviour as [`Self::new`].
    ///
    /// The name mirrors
    /// [`MemorySpanObserver::with_sink`](crate::observability::MemorySpanObserver::with_sink), so a
    /// wiring site picks the bridge and the sink in one expression either way.
    #[must_use]
    pub fn with_sink(sink: Arc<OtelTraceSink>) -> Self {
        Self::new(sink)
    }

    /// Returns the sink this observer forwards to.
    #[must_use]
    pub fn sink(&self) -> &Arc<OtelTraceSink> {
        &self.sink
    }

    /// Returns the session id stamped onto every frame, when one was configured.
    #[must_use]
    pub fn session_id(&self) -> Option<&str> {
        self.session_id.as_deref()
    }

    /// Builds the `memory.{operation}` component frame of `event`; see [`Self::observe`].
    fn frame_for(&self, event: &TeamMemoryEvent) -> TraceFrame {
        TraceFrame::workflow(span_for(event, self.session_id.as_deref()))
    }
}

impl TeamMemoryObserver for OtlpMemoryObserver {
    /// Forwards one event as one component frame.
    ///
    /// Nothing here can fail: a sink that is not recording (a shut-down handle, or a build without
    /// the extensions `otel` feature behind the sink) drops the frame, and the sink itself logs
    /// and ignores anything it cannot map. A no-op costs one `debug!` line.
    fn observe(&self, event: &TeamMemoryEvent) {
        if !self.sink.is_recording() {
            debug!(
                target: LOG_TARGET,
                team = %event.team_id,
                operation = event.operation.as_str(),
                "otlp memory observer: the sink is not recording, dropping the span"
            );
            return;
        }
        self.sink.emit(self.frame_for(event));
    }
}

/// Builds an observer over a private in-memory `OTel` provider.
///
/// Delegates to `OtelTraceSink::with_session`, so the observer owns an adapter that records spans
/// in memory and touches no network and no collector: useful for tests and for a host that only
/// wants the frames turned into spans. A host that really exports over OTLP builds its own
/// `TracerHandle` (`ExporterType::Otlp`) and passes `OtelTraceSink::new(&handle)` to
/// [`OtlpMemoryObserver::new`].
///
/// # Errors
///
/// Returns the error of `OtelTraceSink::with_session` — `Error::validation` when the default
/// tracer configuration is rejected. Building the private memory provider cannot fail today, so
/// the result is `Ok` for every input in practice.
pub fn otlp_memory_observer(session_id: impl Into<String>) -> Result<OtlpMemoryObserver> {
    let session_id = session_id.into();
    let sink = OtelTraceSink::with_session(session_id.clone())?;
    Ok(OtlpMemoryObserver {
        sink,
        session_id: Some(session_id),
    })
}

/// Builds the component span of `event`.
///
/// The span is terminal (`finish`/`error`): the memory facade reports no timing, so the span covers
/// a single instant — `end_time` repeats `start_time` and the elapsed time is `0`.
fn span_for(event: &TeamMemoryEvent, session_id: Option<&str>) -> TraceSpan {
    let operation = event.operation.as_str();
    let invoke_id = invoke_id_for(event);
    let name = format!("memory.{operation}");
    let mut span = TraceSpan::new(
        trace_id_for(&event.team_id),
        session_id.unwrap_or_default(),
        invoke_id.clone(),
        name.clone(),
        TraceSpanKind::Component,
    );
    span.status = status_for(event.outcome);
    span.end_time = span.start_time;
    span.elapsed_time = Some(0);
    span.component_id = Some(invoke_id);
    span.component_name = Some(name);
    span.component_type = Some(MEMORY_COMPONENT_TYPE.to_owned());
    span.meta_data = attributes_for(event);
    if let Some(error) = event.error.as_deref().filter(|text| !text.is_empty()) {
        span.error = Some(error.to_owned());
    }
    span
}

/// Maps an outcome onto the frame status: `Ok` → `finish`, `Failed` → `error`.
const fn status_for(outcome: TeamMemoryOutcome) -> SpanStatus {
    match outcome {
        TeamMemoryOutcome::Ok => SpanStatus::Finish,
        TeamMemoryOutcome::Failed => SpanStatus::Error,
    }
}

/// Builds the attribute bag the in-memory bridge stamps onto its span.
///
/// The keys are that module's `AT_MEMORY_*` constants, and the presence rules match: an absent or
/// empty optional value is left out instead of being written as an empty string.
fn attributes_for(event: &TeamMemoryEvent) -> BTreeMap<String, Value> {
    let mut attributes = BTreeMap::new();
    attributes.insert(
        AT_MEMORY_TEAM_ID.to_owned(),
        Value::from(event.team_id.as_str()),
    );
    attributes.insert(
        AT_MEMORY_OPERATION.to_owned(),
        Value::from(event.operation.as_str()),
    );
    attributes.insert(
        AT_MEMORY_OUTCOME.to_owned(),
        Value::from(event.outcome.as_str()),
    );
    attributes.insert(AT_MEMORY_COUNT.to_owned(), Value::from(event.count));
    if let Some(member_id) = event.member_id.as_deref().filter(|id| !id.is_empty()) {
        attributes.insert(AT_MEMORY_MEMBER_ID.to_owned(), Value::from(member_id));
    }
    if let Some(kind) = event
        .kind
        .as_ref()
        .map(TeamMemoryKind::as_str)
        .filter(|kind| !kind.is_empty())
    {
        attributes.insert(AT_MEMORY_KIND.to_owned(), Value::from(kind));
    }
    if let Some(error) = event.error.as_deref().filter(|text| !text.is_empty()) {
        attributes.insert(AT_MEMORY_ERROR.to_owned(), Value::from(error));
    }
    attributes
}

/// Derives the invoke id of `event`: `memory.{operation}.{16 hex digits}`.
///
/// A pure function of the event identity (`team_id`, `operation`, `member_id`), so a retried
/// operation reuses the span id while a different operation, member or team gets its own. The sink
/// opens and closes each terminal frame independently, so two identical events of one run still
/// export two spans under the same invoke id.
fn invoke_id_for(event: &TeamMemoryEvent) -> String {
    let digest = digest(&[
        event.team_id.as_str(),
        event.operation.as_str(),
        event.member_id.as_deref().unwrap_or(""),
    ]);
    format!("memory.{}.{digest:016x}", event.operation.as_str())
}

/// Derives the trace id of a team scope: 32 lowercase hex characters (the `OTel` shape).
///
/// Two hashes with distinct domain prefixes keep the two 64-bit halves independent while staying a
/// pure function of `team_id`, so every memory span of one team shares one trace and two teams
/// never collide in practice.
fn trace_id_for(team_id: &str) -> String {
    let high = digest(&["trace", team_id]);
    let low = digest(&["span", team_id]);
    format!("{high:016x}{low:016x}")
}

/// FNV-1a over the parts of one identity.
///
/// Parts are hashed in order with a mixing round between them, so `["ab", "c"]` and `["a", "bc"]`
/// produce different digests. No `Hasher` is involved: the digest must stay stable across
/// processes and Rust versions, which `std`'s `RandomState` explicitly does not guarantee.
fn digest(parts: &[&str]) -> u64 {
    parts.iter().fold(FNV_OFFSET_BASIS, |hash, part| {
        let mixed = part.as_bytes().iter().fold(hash, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(FNV_PRIME)
        });
        mixed.wrapping_mul(FNV_PRIME) ^ PART_SEPARATOR
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use openjiuwen_extensions::otel::semconv::{
        OJ_META_DATA, OJ_STATUS, OJ_TRACE_ID, OJ_WORKFLOW_COMPONENT_NAME,
        OJ_WORKFLOW_COMPONENT_TYPE, SPAN_STATUS_ERROR, SPAN_STATUS_FINISH,
    };
    use openjiuwen_extensions::otel::{
        init_otel_tracer, CaptureExporter, ExporterType, OtelTracerConfig, TracerHandle,
    };
    use openjiuwen_memory::record::CallOptions;
    use openjiuwen_memory::team::{
        NewTeamMemory, TeamMemory, TeamMemoryOperation, TeamMemoryScope, TeamSearchOptions,
    };

    /// A private in-memory provider plus the sink it backs and the exported-span buffer.
    struct Fixture {
        /// Keeps the provider (and its batch processor) alive while the test runs.
        handle: TracerHandle,
        /// Spans the provider exported.
        capture: Arc<CaptureExporter>,
        /// Sink the observer forwards to.
        sink: Arc<OtelTraceSink>,
    }

    impl Fixture {
        /// Builds a provider, sink and capture trio over the memory exporter.
        fn build() -> Self {
            let config = OtelTracerConfig {
                // Readable attribute values: the assertions read them verbatim.
                redaction_enabled: false,
                ..OtelTracerConfig::default()
            }
            .with_exporter_type(ExporterType::Memory);
            let handle = init_otel_tracer(config).expect("the memory exporter initializes");
            let capture = handle
                .capture()
                .expect("the memory exporter captures spans");
            let sink = OtelTraceSink::new(&handle).expect("a live handle builds a sink");
            Self {
                handle,
                capture,
                sink,
            }
        }

        /// An observer over this fixture's sink, plus the fixture itself.
        fn observer(self) -> (Self, OtlpMemoryObserver) {
            let observer = OtlpMemoryObserver::new(self.sink.clone());
            (self, observer)
        }

        /// Flushes the batch processor so the exporter has seen every finished span.
        fn flush(&self) {
            self.handle.force_flush();
        }

        /// The number of exported spans.
        fn len(&self) -> usize {
            self.flush();
            self.capture.len()
        }

        /// The exported name of span `index`.
        fn name(&self, index: usize) -> String {
            self.flush();
            let spans = self.capture.spans();
            spans[index].name.to_string()
        }

        /// The last value recorded under `key` on span `index`, rendered as text.
        fn attribute(&self, index: usize, key: &str) -> Option<String> {
            self.flush();
            let spans = self.capture.spans();
            spans[index]
                .attributes
                .iter()
                .rev()
                .find(|pair| pair.key.as_str() == key)
                .map(|pair| pair.value.as_str().into_owned())
        }

        /// The exported span names, after flushing the batch processor.
        fn names(&self) -> Vec<String> {
            self.handle.force_flush();
            self.capture.names()
        }
    }

    /// A successful `write` of two decision records scoped to `alice`.
    fn write_event() -> TeamMemoryEvent {
        TeamMemoryEvent::new(TeamMemoryOperation::Write, "team-1")
            .with_member("alice")
            .with_kind(TeamMemoryKind::Decision)
            .with_count(2)
    }

    #[test]
    fn an_ok_event_builds_a_finished_component_frame_with_the_documented_attributes() {
        let event = write_event();
        let frame = TraceFrame::workflow(span_for(&event, Some("s-1")));
        let span = frame.payload;

        assert_eq!(frame.item_type, "tracer_workflow");
        assert_eq!(span.kind, TraceSpanKind::Component);
        assert_eq!(span.name, "memory.write");
        assert_eq!(span.status, SpanStatus::Finish);
        assert_eq!(span.session_id, "s-1");
        assert_eq!(span.component_name.as_deref(), Some("memory.write"));
        assert_eq!(span.component_type.as_deref(), Some("MemoryOperation"));
        assert_eq!(span.component_id.as_deref(), Some(span.invoke_id.as_str()));
        assert_eq!(span.invoke_id, invoke_id_for(&event));
        assert!(span.invoke_id.starts_with("memory.write."));
        // One instant: the memory facade reports no duration.
        assert_eq!(span.start_time, span.end_time);
        assert_eq!(span.elapsed_time, Some(0));
        // Nothing to parent onto: the in-process team trace is not an OTel span.
        assert!(span.parent_invoke_id.is_none());

        let meta = &span.meta_data;
        assert_eq!(meta.get(AT_MEMORY_TEAM_ID), Some(&Value::from("team-1")));
        assert_eq!(meta.get(AT_MEMORY_OPERATION), Some(&Value::from("write")));
        assert_eq!(meta.get(AT_MEMORY_OUTCOME), Some(&Value::from("ok")));
        assert_eq!(meta.get(AT_MEMORY_COUNT), Some(&Value::from(2_u64)));
        assert_eq!(meta.get(AT_MEMORY_MEMBER_ID), Some(&Value::from("alice")));
        assert_eq!(meta.get(AT_MEMORY_KIND), Some(&Value::from("decision")));
        assert!(meta.get(AT_MEMORY_ERROR).is_none());
        assert!(span.error.is_none());
    }

    #[test]
    fn span_names_follow_memory_dot_operation() {
        for operation in [
            TeamMemoryOperation::Search,
            TeamMemoryOperation::Write,
            TeamMemoryOperation::Forget,
            TeamMemoryOperation::Clear,
            TeamMemoryOperation::Prune,
            TeamMemoryOperation::Extract,
        ] {
            let span = span_for(&TeamMemoryEvent::new(operation, "team-1"), None);
            assert_eq!(span.name, format!("memory.{}", operation.as_str()));
            assert_eq!(
                span.meta_data.get(AT_MEMORY_OPERATION),
                Some(&Value::from(operation.as_str()))
            );
        }
    }

    #[test]
    fn a_failed_event_carries_the_error_text_and_the_failed_status() {
        let event =
            TeamMemoryEvent::new(TeamMemoryOperation::Prune, "team-1").failed("store unavailable");
        let span = span_for(&event, None);

        assert_eq!(span.name, "memory.prune");
        assert_eq!(span.status, SpanStatus::Error);
        assert_eq!(span.error.as_deref(), Some("store unavailable"));
        assert_eq!(
            span.meta_data.get(AT_MEMORY_OUTCOME),
            Some(&Value::from("failed"))
        );
        assert_eq!(
            span.meta_data.get(AT_MEMORY_ERROR),
            Some(&Value::from("store unavailable"))
        );
    }

    #[test]
    fn optional_identity_attributes_are_present_only_when_set_and_non_empty() {
        let blank = TeamMemoryEvent::new(TeamMemoryOperation::Forget, "team-1")
            .with_member("")
            .failed("");
        let span = span_for(&blank, None);

        assert!(span.meta_data.get(AT_MEMORY_MEMBER_ID).is_none());
        assert!(span.meta_data.get(AT_MEMORY_KIND).is_none());
        assert!(span.meta_data.get(AT_MEMORY_ERROR).is_none());
        assert_eq!(
            span.meta_data.get(AT_MEMORY_OUTCOME),
            Some(&Value::from("failed"))
        );
        // A failed event without text still exports a failed span, just without a message.
        assert_eq!(span.status, SpanStatus::Error);
        assert!(span.error.is_none());
    }

    #[test]
    fn the_invoke_id_is_stable_across_a_retry_and_differs_per_operation_member_and_team() {
        let base = write_event();
        assert_eq!(invoke_id_for(&base), invoke_id_for(&write_event()));

        let other_operation =
            TeamMemoryEvent::new(TeamMemoryOperation::Search, "team-1").with_member("alice");
        assert_ne!(invoke_id_for(&base), invoke_id_for(&other_operation));

        let other_member = write_event().with_member("bob");
        assert_ne!(invoke_id_for(&base), invoke_id_for(&other_member));

        let unscoped = TeamMemoryEvent::new(TeamMemoryOperation::Write, "team-1");
        assert_ne!(invoke_id_for(&base), invoke_id_for(&unscoped));

        let other_team = write_event_for_team("team-2");
        assert_ne!(invoke_id_for(&base), invoke_id_for(&other_team));

        // A payload field that is not identity keeps the id, so a retry reuses its span id.
        let recounted = write_event().with_count(9);
        assert_eq!(invoke_id_for(&base), invoke_id_for(&recounted));
    }

    /// The write event of `team`, same identity fields as [`write_event`].
    fn write_event_for_team(team: &str) -> TeamMemoryEvent {
        TeamMemoryEvent::new(TeamMemoryOperation::Write, team)
            .with_member("alice")
            .with_kind(TeamMemoryKind::Decision)
            .with_count(2)
    }

    #[test]
    fn the_trace_id_is_32_lowercase_hex_and_shared_by_one_team() {
        let trace = trace_id_for("team-1");
        assert_eq!(trace.len(), 32, "{trace}");
        assert!(
            trace
                .chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
            "{trace}"
        );
        assert_eq!(trace, trace_id_for("team-1"));
        assert_ne!(trace, trace_id_for("team-2"));

        let write = span_for(
            &TeamMemoryEvent::new(TeamMemoryOperation::Write, "team-1"),
            None,
        );
        let search = span_for(
            &TeamMemoryEvent::new(TeamMemoryOperation::Search, "team-1"),
            None,
        );
        assert_eq!(write.trace_id, search.trace_id, "one team scope, one trace");
        assert_eq!(write.trace_id, trace);
    }

    #[test]
    fn a_frame_built_without_a_session_leaves_the_sink_fallback_in_charge() {
        let span = span_for(
            &TeamMemoryEvent::new(TeamMemoryOperation::Search, "team-1"),
            None,
        );
        assert!(span.session_id.is_empty());
    }

    #[test]
    fn otlp_memory_observer_builds_over_a_private_memory_provider() {
        let observer = otlp_memory_observer("session-1").expect("the memory provider builds");
        assert_eq!(observer.session_id(), Some("session-1"));
        assert!(observer.sink().is_recording());
        let span = observer
            .frame_for(&TeamMemoryEvent::new(
                TeamMemoryOperation::Extract,
                "team-1",
            ))
            .payload;
        assert_eq!(span.session_id, "session-1");
        assert_eq!(span.name, "memory.extract");
    }

    #[test]
    fn with_sink_is_the_same_constructor_as_new() {
        let fixture = Fixture::build();
        let observer = OtlpMemoryObserver::with_sink(fixture.sink.clone());
        assert_eq!(observer.session_id(), None);
        assert_eq!(
            observer.frame_for(&write_event()).payload.session_id,
            OtlpMemoryObserver::new(fixture.sink.clone())
                .frame_for(&write_event())
                .payload
                .session_id
        );
    }

    #[test]
    fn observing_on_a_not_recording_sink_is_a_silent_no_op() {
        let config = OtelTracerConfig {
            redaction_enabled: false,
            ..OtelTracerConfig::default()
        }
        .with_exporter_type(ExporterType::Memory);
        let handle = init_otel_tracer(config).expect("the memory exporter initializes");
        let capture = handle
            .capture()
            .expect("the memory exporter captures spans");
        handle.shutdown();
        let sink = OtelTraceSink::new(&handle).expect("a shut-down handle still builds a sink");
        assert!(!sink.is_recording(), "a shut-down handle never records");

        let observer = OtlpMemoryObserver::new(sink);
        observer.observe(&write_event());
        observer.observe(&TeamMemoryEvent::new(TeamMemoryOperation::Clear, "team-1"));
        handle.force_flush();
        assert!(capture.is_empty(), "nothing reaches the exporter");
    }

    #[test]
    fn the_observer_is_send_and_sync_and_usable_as_a_trait_object() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<OtlpMemoryObserver>();
        let fixture = Fixture::build();
        let shared: Arc<dyn TeamMemoryObserver> =
            Arc::new(OtlpMemoryObserver::with_sink(fixture.sink.clone()));
        shared.observe(&TeamMemoryEvent::new(TeamMemoryOperation::Prune, "team-1"));
        assert_eq!(fixture.names().len(), 1);
    }

    #[test]
    fn the_extensions_sink_exports_one_finished_component_span_per_event() {
        let (fixture, observer) = Fixture::build().observer();
        let event = write_event();
        let expected_name = format!("component.{}", invoke_id_for(&event));
        observer.observe(&event);

        assert_eq!(fixture.len(), 1, "{:?}", fixture.names());
        assert_eq!(fixture.name(0), expected_name);
        assert_eq!(
            fixture.attribute(0, OJ_STATUS).as_deref(),
            Some(SPAN_STATUS_FINISH)
        );
        assert_eq!(
            fixture.attribute(0, OJ_WORKFLOW_COMPONENT_TYPE).as_deref(),
            Some(MEMORY_COMPONENT_TYPE)
        );
        assert_eq!(
            fixture.attribute(0, OJ_WORKFLOW_COMPONENT_NAME).as_deref(),
            Some("memory.write")
        );
        let meta = fixture
            .attribute(0, OJ_META_DATA)
            .expect("the metadata blob is exported");
        assert!(meta.contains("\"team_id\":\"team-1\""), "{meta}");
        assert!(meta.contains("\"operation\":\"write\""), "{meta}");
        assert!(meta.contains("\"outcome\":\"ok\""), "{meta}");
        assert!(meta.contains("\"count\":2"), "{meta}");
        assert!(meta.contains("\"member_id\":\"alice\""), "{meta}");
    }

    #[test]
    fn a_failed_event_exports_one_error_status_span() {
        let (fixture, observer) = Fixture::build().observer();
        observer.observe(
            &TeamMemoryEvent::new(TeamMemoryOperation::Search, "team-1")
                .with_member("alice")
                .failed("store unavailable"),
        );

        assert_eq!(fixture.len(), 1, "{:?}", fixture.names());
        assert_eq!(
            fixture.attribute(0, OJ_STATUS).as_deref(),
            Some(SPAN_STATUS_ERROR)
        );
        let meta = fixture
            .attribute(0, OJ_META_DATA)
            .expect("the metadata blob is exported");
        assert!(meta.contains("\"outcome\":\"failed\""), "{meta}");
        assert!(meta.contains("\"error\":\"store unavailable\""), "{meta}");
    }

    #[test]
    fn two_events_of_one_team_carry_one_derived_trace_id() {
        let (fixture, observer) = Fixture::build().observer();
        observer.observe(&write_event());
        observer.observe(&TeamMemoryEvent::new(TeamMemoryOperation::Search, "team-1"));

        assert_eq!(fixture.len(), 2, "{:?}", fixture.names());
        // The sink starts each parentless frame as its own OTel trace, so the OTel trace ids
        // differ; the derived team trace id travels as `openjiuwen.trace.id` on both spans.
        for index in 0..2 {
            assert_eq!(
                fixture.attribute(index, OJ_TRACE_ID).as_deref(),
                Some(trace_id_for("team-1").as_str()),
                "the memory spans of one team share the derived trace id"
            );
        }
    }

    #[tokio::test]
    async fn a_real_facade_event_reaches_the_exporter() {
        let (fixture, observer) = Fixture::build().observer();
        let memory = TeamMemory::in_memory().with_observer(Arc::new(observer));
        memory
            .remember(
                NewTeamMemory::decision("team-1", "ship behind a flag").with_member("alice"),
                &CallOptions::new(),
            )
            .await
            .expect("remember succeeds");
        memory
            .search(
                &TeamMemoryScope::member("team-1", "alice"),
                "ship",
                &TeamSearchOptions::default(),
            )
            .await
            .expect("search succeeds");

        assert_eq!(fixture.len(), 2, "{:?}", fixture.names());
        let names = fixture.names();
        assert!(names[0].starts_with("component.memory.write."), "{names:?}");
        assert!(
            names[1].starts_with("component.memory.search."),
            "{names:?}"
        );
        for index in 0..2 {
            assert_eq!(
                fixture.attribute(index, OJ_STATUS).as_deref(),
                Some(SPAN_STATUS_FINISH)
            );
            let meta = fixture
                .attribute(index, OJ_META_DATA)
                .expect("the metadata blob is exported");
            assert!(meta.contains("\"team_id\":\"team-1\""), "{meta}");
        }
    }
}
