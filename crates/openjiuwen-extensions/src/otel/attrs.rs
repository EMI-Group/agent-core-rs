//! Shared attribute-writing helpers for sinks and handlers.
//!
//! Ports the attribute-writing sites of
//! `openjiuwen/extensions/tracer_otel/handler.py` (v0.1.20) that the agent and
//! workflow handlers share: the span base fields ([`apply_start_attributes`]),
//! the end-time fields ([`apply_end_attributes`]), the failure fields
//! ([`apply_error_attributes`]), and the interruption field
//! ([`apply_interrupt_attributes`]), plus the string helpers Python keeps inline
//! (`_serialize`, the elapsed-time formatting, and the local-time stamp).
//!
//! Every helper takes the concrete [`Span`] because the mutation methods of
//! `opentelemetry::trace::Span` need `&mut self`; nothing here clones, ends, or
//! records a span, so the callers stay in charge of the span lifecycle. No
//! helper panics: a non-recording span silently ignores every write, exactly
//! like Python's `set_attribute` on a dropped span.
//!
//! # Deviations from Python
//!
//! - Python serializes payloads with `json.dumps`, whose default separators add
//!   a space after `,` and `:`; [`serialize_value`] and [`error_payload`] emit
//!   compact JSON so the attribute round-trips through `serde_json` unchanged.
//!   Non-ASCII text is preserved either way (`ensure_ascii=False`).
//! - Python's `_serialize` renders scalars with `str()`, so `True` and `None`
//!   become `"True"` / `"None"`; here every non-string value uses its JSON text
//!   (`"true"`, `"null"`), matching `super::redaction`'s `text_of`.
//! - [`OJ_PARENT_INVOKE_ID`] is written only when the span has a parent;
//!   Python writes an empty string for root spans.
//! - Python records the failure of a non-`BaseError` exception with
//!   `OJStatusCode.WORKFLOW_EXECUTION_ERROR.code` (`100102`); the Rust helpers
//!   take the code as an argument so callers can pass
//!   [`openjiuwen_common::error::ErrorKind::as_str`] through
//!   [`error_payload_for`], with [`DEFAULT_ERROR_CODE`] as the fallback for a
//!   [`TraceSpan`], which carries no code.

// The agent and workflow handlers that call these helpers are ported in
// follow-up commits; until they land no crate code calls them, so the module
// opts out of the dead-code lint instead of hiding every helper behind an
// `#[allow]`. The unit tests below exercise each helper.
#![allow(dead_code)]

use std::borrow::Cow;

use opentelemetry::trace::{Span as _, Status};
use opentelemetry::KeyValue;
use opentelemetry_sdk::trace::Span;
use serde_json::{json, Value};

use openjiuwen_common::error::Error;
use openjiuwen_core::session::tracer::{SpanStatus, TraceSpan};

use super::semconv::{
    ERROR_TYPE, GEN_AI_PROVIDER_NAME, GEN_AI_PROVIDER_OPENJIUWEN, OJ_AGENT_ERROR_MESSAGE,
    OJ_CHILD_INVOKE_IDS, OJ_ELAPSED_TIME, OJ_END_TIME, OJ_ERROR, OJ_INVOKE_ID, OJ_PARENT_INVOKE_ID,
    OJ_SESSION_ID, OJ_START_TIME, OJ_STATUS, OJ_TRACE_ID, OJ_WORKFLOW_ERROR_MESSAGE,
    SPAN_STATUS_ERROR, SPAN_STATUS_FINISH, SPAN_STATUS_INTERRUPTED, SPAN_STATUS_RUNNING,
    SPAN_STATUS_START,
};

/// `strftime` pattern of the `openjiuwen.*_time` attributes.
///
/// Python counterpart: `datetime.now(tz=_CACHED_TZLOCAL)`, whose `str()` form is
/// `%Y-%m-%d %H:%M:%S.%f` — local time with microseconds.
const TIMESTAMP_FORMAT: &str = "%Y-%m-%d %H:%M:%S%.6f";

/// Milliseconds in one second: the boundary between the two elapsed formats.
const MILLIS_PER_SECOND: u64 = 1_000;

/// Error code used when a span carries error text but no structured code.
///
/// Python counterpart: `OJStatusCode.WORKFLOW_EXECUTION_ERROR.code` (`100102`),
/// which `handler.py` uses for exceptions that are not a `BaseError`.
pub(crate) const DEFAULT_ERROR_CODE: &str = "100102";

/// Name of the span event that mirrors `record_exception`.
const EXCEPTION_EVENT: &str = "exception";

/// Attribute key of the message carried by [`EXCEPTION_EVENT`].
const EXCEPTION_MESSAGE_KEY: &str = "exception.message";

/// Which handler family sets a span's error attributes.
///
/// Python keeps two near-identical error paths (`OtelAgentHandler.
/// _mark_error_and_end` and the error branch of `OtelWorkflowHandler.on_invoke`);
/// the only difference on the wire is the attribute key carrying the error text,
/// so the scope selects that key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum AttributeScope {
    /// Agent spans: error text lands on `openjiuwen.agent.error_message`.
    Agent,
    /// Workflow and component spans: error text lands on
    /// `openjiuwen.workflow.error_message`.
    Workflow,
}

impl AttributeScope {
    /// The attribute key that carries the error text of this scope.
    pub(crate) const fn error_message_key(self) -> &'static str {
        match self {
            Self::Agent => OJ_AGENT_ERROR_MESSAGE,
            Self::Workflow => OJ_WORKFLOW_ERROR_MESSAGE,
        }
    }
}

/// The local wall-clock stamp used by `openjiuwen.start_time` / `end_time`.
///
/// Python counterpart: `str(datetime.now())` — local time, seconds resolution
/// plus six fractional digits.
pub(crate) fn local_timestamp() -> String {
    chrono::Local::now().format(TIMESTAMP_FORMAT).to_string()
}

/// Formats a duration the way both handlers do.
///
/// Python counterpart: `f"{elapsed_ms:.0f}ms" if elapsed_ms < 1000 else
/// f"{elapsed_ms / 1000:.2f}s"`, so a duration below one second keeps the
/// millisecond unit and anything longer switches to seconds with two decimals.
pub(crate) fn elapsed_string(millis: u64) -> String {
    if millis < MILLIS_PER_SECOND {
        return format!("{millis}ms");
    }
    // A duration in milliseconds fits an `f64` with far more precision than the
    // two decimals the format keeps.
    #[allow(clippy::cast_precision_loss)]
    let seconds = millis as f64 / MILLIS_PER_SECOND as f64;
    format!("{seconds:.2}s")
}

/// Renders a JSON value as attribute text.
///
/// Python counterpart: `_serialize` in `tracer_otel/handler.py`. Strings are
/// used verbatim; every other value is rendered as compact JSON, so objects and
/// arrays stay machine-readable and non-ASCII text is preserved. Scalars use
/// their JSON text (`true`, `null`, `12`) rather than Python's `str()` spelling.
pub(crate) fn serialize_value(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

/// Writes a JSON value as a string attribute.
pub(crate) fn set_json_attribute(span: &mut Span, key: &'static str, value: &Value) {
    let text = serialize_value(value);
    set_text_attribute(span, key, &text);
}

/// Writes a JSON value as a string attribute, skipping the write for `None`.
pub(crate) fn set_json_attribute_opt(span: &mut Span, key: &'static str, value: Option<&Value>) {
    if let Some(value) = value {
        set_json_attribute(span, key, value);
    }
}

/// Writes one string attribute.
///
/// Duplicate handling and ordering are the SDK's; a non-recording span ignores
/// the write, so callers never need a `recorded` check here.
pub(crate) fn set_text_attribute(span: &mut Span, key: &'static str, value: &str) {
    span.set_attribute(KeyValue::new(key, value.to_owned()));
}

/// Maps a session [`SpanStatus`] onto the `openjiuwen.status` value vocabulary.
///
/// The wire names are [`SpanStatus::as_str`]'s, mirrored by the
/// [`SPAN_STATUS_START`] family of `semconv`; keeping the single mapping here
/// stops a call site from re-typing a literal.
pub(crate) fn span_status_name(status: SpanStatus) -> &'static str {
    match status {
        SpanStatus::Start => SPAN_STATUS_START,
        SpanStatus::Running => SPAN_STATUS_RUNNING,
        SpanStatus::Finish => SPAN_STATUS_FINISH,
        SpanStatus::Interrupted => SPAN_STATUS_INTERRUPTED,
        SpanStatus::Error => SPAN_STATUS_ERROR,
    }
}

/// The status a span carries once it is closed.
///
/// A span still marked `start` or `running` when it ends is reported as
/// `finish` (Python: `span.status or NodeStatus.FINISH.value`, whose default for
/// a completed invocation is `finish`); terminal statuses are kept as they are.
fn ended_status_name(status: SpanStatus) -> &'static str {
    match status {
        SpanStatus::Start | SpanStatus::Running => SPAN_STATUS_FINISH,
        terminal => span_status_name(terminal),
    }
}

/// Builds the `openjiuwen.error` payload for an error code and message.
///
/// Python counterpart: `_serialize({"error_code": …, "message": …})`. The object
/// keeps the Python key order (`error_code`, then `message`).
pub(crate) fn error_payload(error_code: &str, message: &str) -> String {
    json!({ "error_code": error_code, "message": message }).to_string()
}

/// Builds the `openjiuwen.error` payload for a library error.
///
/// The code is the [`openjiuwen_common::error::ErrorKind`] name (`workflow`,
/// `tool`, …) and the message is [`openjiuwen_common::error::Error::message`].
pub(crate) fn error_payload_for(error: &Error) -> String {
    let message = error.message();
    error_payload(error.kind().as_str(), &message)
}

/// Writes the span base fields recorded when a span starts.
///
/// Python counterpart: `_start_and_push` in the agent handler and the base
/// attributes of `on_call_start` in the workflow handler — the trace id, the
/// provider name, the session id (only when the span has one), the invoke id,
/// the parent invoke id (only when the span has a parent), and the local start
/// time.
pub(crate) fn apply_start_attributes(span: &mut Span, trace_span: &TraceSpan) {
    set_text_attribute(span, OJ_TRACE_ID, &trace_span.trace_id);
    // Every span declares the provider; the non-standard `gen_ai.system` is
    // intentionally unused on this stack (see `semconv`).
    set_text_attribute(span, GEN_AI_PROVIDER_NAME, GEN_AI_PROVIDER_OPENJIUWEN);
    if !trace_span.session_id.is_empty() {
        set_text_attribute(span, OJ_SESSION_ID, &trace_span.session_id);
    }
    set_text_attribute(span, OJ_INVOKE_ID, &trace_span.invoke_id);
    if let Some(parent) = trace_span.parent_invoke_id.as_deref() {
        set_text_attribute(span, OJ_PARENT_INVOKE_ID, parent);
    }
    let start_time = local_timestamp();
    set_text_attribute(span, OJ_START_TIME, &start_time);
}

/// Writes the closing attributes and marks the span successful.
///
/// Python counterpart: `_set_end_attrs` plus the `Status(StatusCode.OK)` of
/// `_end_and_pop` — the end time, the elapsed time, the child invoke ids (only
/// when the span recorded some), and the span status.
pub(crate) fn apply_end_attributes(span: &mut Span, trace_span: &TraceSpan) {
    span.set_status(Status::Ok);
    let end_time = local_timestamp();
    set_text_attribute(span, OJ_END_TIME, &end_time);
    let elapsed = elapsed_string(trace_span.elapsed().unwrap_or(0));
    set_text_attribute(span, OJ_ELAPSED_TIME, &elapsed);
    if !trace_span.child_invokes.is_empty() {
        let children = json!(trace_span.child_invokes);
        set_json_attribute(span, OJ_CHILD_INVOKE_IDS, &children);
    }
    let status = ended_status_name(trace_span.status);
    set_text_attribute(span, OJ_STATUS, status);
}

/// Writes the failure attributes of a span and marks it failed.
///
/// The message is the span's own [`TraceSpan::error`]; a span that carries no
/// text still gets an empty message, so a failed span never looks successful.
/// The `openjiuwen.error` code falls back to [`DEFAULT_ERROR_CODE`]; callers
/// holding a structured error use [`apply_error_attributes_with`] with
/// [`error_payload_for`]'s code instead.
pub(crate) fn apply_error_attributes(
    span: &mut Span,
    trace_span: &TraceSpan,
    scope: AttributeScope,
) {
    let message = trace_span.error.clone().unwrap_or_default();
    apply_error_attributes_with(span, DEFAULT_ERROR_CODE, &message, scope);
}

/// Writes the failure attributes of a span from an explicit code and message.
///
/// Python counterpart: the error branch of `on_invoke` and `_mark_error_and_end`
/// — an `ERROR` status with the message as description, `openjiuwen.status =
/// "error"`, the scope's error-message attribute, the `openjiuwen.error` JSON
/// payload, `error.type`, and an `exception` event carrying `exception.message`
/// (the `record_exception` equivalent).
pub(crate) fn apply_error_attributes_with(
    span: &mut Span,
    error_code: &str,
    message: &str,
    scope: AttributeScope,
) {
    span.set_status(Status::Error {
        description: Cow::Owned(message.to_owned()),
    });
    set_text_attribute(span, OJ_STATUS, SPAN_STATUS_ERROR);
    set_text_attribute(span, scope.error_message_key(), message);
    let payload = error_payload(error_code, message);
    set_text_attribute(span, OJ_ERROR, &payload);
    set_text_attribute(span, ERROR_TYPE, error_code);
    span.add_event(
        EXCEPTION_EVENT,
        vec![KeyValue::new(EXCEPTION_MESSAGE_KEY, message.to_owned())],
    );
}

/// Writes the attributes of an interrupted span.
///
/// Python counterpart: the `GraphInterrupt` branch of `on_invoke`. An interrupt
/// is control flow, not a failure, so the span keeps a non-error status and
/// gains no `openjiuwen.error`; the interrupt payload is preserved on the
/// workflow scope's error-message attribute for debuggability.
pub(crate) fn apply_interrupt_attributes(
    span: &mut Span,
    trace_span: &TraceSpan,
    scope: AttributeScope,
) {
    set_text_attribute(span, OJ_STATUS, SPAN_STATUS_INTERRUPTED);
    if scope != AttributeScope::Workflow {
        return;
    }
    if let Some(text) = trace_span.error.as_deref().filter(|text| !text.is_empty()) {
        set_text_attribute(span, OJ_WORKFLOW_ERROR_MESSAGE, text);
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::super::semconv::{
        GEN_AI_REQUEST_MODEL, OJ_AGENT_INPUTS, OJ_AGENT_NAME, OJ_AGENT_OUTPUTS,
    };
    use super::*;
    use crate::otel::config::{ExporterType, OtelTracerConfig};
    use crate::otel::handle::{init_otel_tracer, TracerHandle};
    use openjiuwen_core::session::tracer::TraceSpanKind;
    use opentelemetry::trace::Tracer as _;
    use opentelemetry::Context;
    use opentelemetry_sdk::trace::SpanData;

    // port of tests/unit_tests/extensions/tracer_otel/test_handler.py attribute
    // assertions (`_set_end_attrs`, `_mark_error_and_end`, the interrupt branch).

    /// A tracer over a private in-memory provider.
    fn tracer_handle() -> TracerHandle {
        let config = OtelTracerConfig::default().with_exporter_type(ExporterType::Memory);
        init_otel_tracer(config).expect("the memory exporter always initializes")
    }

    /// Starts a recording span with no parent.
    fn start_span(handle: &TracerHandle, name: &'static str) -> Span {
        handle
            .tracer()
            .expect("the memory handle is active")
            .start_with_context(name, &Context::new())
    }

    /// The exported snapshot of a span that has not ended yet.
    fn snapshot(span: &Span) -> SpanData {
        span.exported_data().expect("in-memory spans record")
    }

    /// The string value of one attribute, when present.
    fn attribute(data: &SpanData, key: &str) -> Option<String> {
        data.attributes
            .iter()
            .find(|pair| pair.key.as_str() == key)
            .map(|pair| pair.value.as_str().into_owned())
    }

    /// A span with every base field set.
    fn agent_span() -> TraceSpan {
        let mut span = TraceSpan::new(
            "trace-1",
            "session-1",
            "invoke-1",
            "chat Agent",
            TraceSpanKind::Agent,
        );
        span.parent_invoke_id = Some("parent-1".to_owned());
        span.start_time = Some(1_000);
        span.end_time = Some(1_250);
        span
    }

    #[test]
    fn elapsed_string_switches_unit_at_one_second() {
        // test_handler.py pins the same two shapes (`"12ms"`, `"1.50s"`).
        assert_eq!(elapsed_string(0), "0ms");
        assert_eq!(elapsed_string(999), "999ms");
        assert_eq!(elapsed_string(1_000), "1.00s");
        assert_eq!(elapsed_string(1_500), "1.50s");
        assert_eq!(elapsed_string(61_000), "61.00s");
    }

    #[test]
    fn local_timestamp_has_the_python_datetime_shape() {
        let stamp = local_timestamp();
        assert_eq!(stamp.len(), 26, "unexpected stamp: {stamp}");
        let (date, rest) = stamp.split_once(' ').expect("date and time separated");
        assert_eq!(date.len(), 10, "unexpected date: {date}");
        assert_eq!(rest.len(), 15, "unexpected time: {rest}");
        let (clock, fraction) = rest.split_once('.').expect("a fraction is present");
        assert_eq!(fraction.len(), 6, "six fractional digits expected: {rest}");
        for (index, byte) in clock.bytes().enumerate() {
            if index == 2 || index == 5 {
                assert_eq!(byte, b':', "unexpected clock {clock}");
            } else {
                assert!(byte.is_ascii_digit(), "unexpected clock {clock}");
            }
        }
        assert!(fraction.bytes().all(|byte| byte.is_ascii_digit()));
        // Two consecutive stamps never go backwards (no wall-clock equality).
        let later = local_timestamp();
        assert!(later >= stamp, "{later} precedes {stamp}");
    }

    #[test]
    fn serialize_value_keeps_strings_verbatim_and_json_elsewhere() {
        assert_eq!(serialize_value(&json!("plain text")), "plain text");
        assert_eq!(
            serialize_value(&json!({"a": 1, "b": [true, null]})),
            r#"{"a":1,"b":[true,null]}"#
        );
        assert_eq!(serialize_value(&json!([1, "two"])), r#"[1,"two"]"#);
        assert_eq!(serialize_value(&json!(12)), "12");
        assert_eq!(serialize_value(&json!(true)), "true");
        assert_eq!(serialize_value(&json!(null)), "null");
        assert_eq!(
            serialize_value(&json!({"city": "北京"})),
            r#"{"city":"北京"}"#
        );
    }

    #[test]
    fn error_payload_carries_code_and_message() {
        assert_eq!(
            error_payload("100102", "boom"),
            r#"{"error_code":"100102","message":"boom"}"#
        );
        assert_eq!(
            error_payload_for(&Error::workflow("bad node")),
            r#"{"error_code":"workflow","message":"bad node"}"#
        );
    }

    #[test]
    fn start_attributes_match_the_python_base_fields() {
        let handle = tracer_handle();
        let mut span = start_span(&handle, "chat Agent");
        apply_start_attributes(&mut span, &agent_span());
        let data = snapshot(&span);
        assert_eq!(attribute(&data, OJ_TRACE_ID).as_deref(), Some("trace-1"));
        assert_eq!(
            attribute(&data, GEN_AI_PROVIDER_NAME).as_deref(),
            Some(GEN_AI_PROVIDER_OPENJIUWEN)
        );
        assert_eq!(
            attribute(&data, OJ_SESSION_ID).as_deref(),
            Some("session-1")
        );
        assert_eq!(attribute(&data, OJ_INVOKE_ID).as_deref(), Some("invoke-1"));
        assert_eq!(
            attribute(&data, OJ_PARENT_INVOKE_ID).as_deref(),
            Some("parent-1")
        );
        let stamp = attribute(&data, OJ_START_TIME).expect("start_time is written");
        assert_eq!(stamp.len(), 26, "unexpected stamp: {stamp}");
        span.end();
    }

    #[test]
    fn start_attributes_skip_an_empty_session_and_an_absent_parent() {
        let handle = tracer_handle();
        let mut span = start_span(&handle, "chat Agent");
        let mut trace_span = agent_span();
        trace_span.session_id = String::new();
        trace_span.parent_invoke_id = None;
        apply_start_attributes(&mut span, &trace_span);
        let data = snapshot(&span);
        assert!(attribute(&data, OJ_SESSION_ID).is_none());
        assert!(attribute(&data, OJ_PARENT_INVOKE_ID).is_none());
        assert!(attribute(&data, OJ_TRACE_ID).is_some());
        span.end();
    }

    #[test]
    fn end_attributes_mark_the_span_ok_and_carry_the_duration() {
        let handle = tracer_handle();
        let mut span = start_span(&handle, "chat Agent");
        let mut trace_span = agent_span();
        trace_span.child_invokes = vec!["child-1".to_owned(), "child-2".to_owned()];
        apply_end_attributes(&mut span, &trace_span);
        let data = snapshot(&span);
        assert_eq!(data.status, Status::Ok);
        assert_eq!(attribute(&data, OJ_ELAPSED_TIME).as_deref(), Some("250ms"));
        assert_eq!(
            attribute(&data, OJ_CHILD_INVOKE_IDS).as_deref(),
            Some(r#"["child-1","child-2"]"#)
        );
        assert_eq!(
            attribute(&data, OJ_STATUS).as_deref(),
            Some(SPAN_STATUS_FINISH)
        );
        let end_time = attribute(&data, OJ_END_TIME).expect("end_time is written");
        assert_eq!(end_time.len(), 26, "unexpected stamp: {end_time}");
        span.end();
    }

    #[test]
    fn end_attributes_omit_absent_child_ids_and_report_terminal_status() {
        let handle = tracer_handle();
        let mut span = start_span(&handle, "chat Agent");
        let mut trace_span = agent_span();
        trace_span.child_invokes = Vec::new();
        trace_span.status = SpanStatus::Interrupted;
        apply_end_attributes(&mut span, &trace_span);
        let data = snapshot(&span);
        assert!(attribute(&data, OJ_CHILD_INVOKE_IDS).is_none());
        assert_eq!(
            attribute(&data, OJ_STATUS).as_deref(),
            Some(SPAN_STATUS_INTERRUPTED)
        );
        span.end();
    }

    #[test]
    fn end_attributes_treat_start_and_running_as_finish() {
        let handle = tracer_handle();
        for status in [SpanStatus::Start, SpanStatus::Running] {
            let mut span = start_span(&handle, "chat Agent");
            let mut trace_span = agent_span();
            trace_span.status = status;
            apply_end_attributes(&mut span, &trace_span);
            assert_eq!(
                attribute(&snapshot(&span), OJ_STATUS).as_deref(),
                Some(SPAN_STATUS_FINISH),
                "{status:?} must close as finish"
            );
            span.end();
        }
    }

    #[test]
    fn error_attributes_use_the_span_error_and_the_default_code() {
        let handle = tracer_handle();
        let mut span = start_span(&handle, "chat Agent");
        let mut trace_span = agent_span();
        trace_span.status = SpanStatus::Error;
        trace_span.error = Some("model exploded".to_owned());
        apply_error_attributes(&mut span, &trace_span, AttributeScope::Agent);
        let data = snapshot(&span);
        assert_eq!(
            data.status,
            Status::Error {
                description: "model exploded".into()
            }
        );
        assert_eq!(
            attribute(&data, OJ_STATUS).as_deref(),
            Some(SPAN_STATUS_ERROR)
        );
        assert_eq!(
            attribute(&data, OJ_AGENT_ERROR_MESSAGE).as_deref(),
            Some("model exploded")
        );
        assert!(attribute(&data, OJ_WORKFLOW_ERROR_MESSAGE).is_none());
        assert_eq!(
            attribute(&data, OJ_ERROR).as_deref(),
            Some(r#"{"error_code":"100102","message":"model exploded"}"#)
        );
        assert_eq!(attribute(&data, ERROR_TYPE).as_deref(), Some("100102"));
        let event = data
            .events
            .iter()
            .find(|event| event.name == EXCEPTION_EVENT)
            .expect("an exception event is recorded");
        assert_eq!(
            event
                .attributes
                .iter()
                .find(|pair| pair.key.as_str() == EXCEPTION_MESSAGE_KEY)
                .map(|pair| pair.value.as_str().into_owned()),
            Some("model exploded".to_owned())
        );
        span.end();
    }

    #[test]
    fn error_attributes_reach_the_workflow_message_key() {
        let handle = tracer_handle();
        let mut span = start_span(&handle, "invoke-1");
        apply_error_attributes_with(
            &mut span,
            "workflow",
            "node failed",
            AttributeScope::Workflow,
        );
        let data = snapshot(&span);
        assert_eq!(
            attribute(&data, OJ_WORKFLOW_ERROR_MESSAGE).as_deref(),
            Some("node failed")
        );
        assert!(attribute(&data, OJ_AGENT_ERROR_MESSAGE).is_none());
        assert_eq!(
            attribute(&data, OJ_ERROR).as_deref(),
            Some(r#"{"error_code":"workflow","message":"node failed"}"#)
        );
        assert_eq!(attribute(&data, ERROR_TYPE).as_deref(), Some("workflow"));
        span.end();
    }

    #[test]
    fn error_attributes_of_a_missing_message_are_still_error_typed() {
        let handle = tracer_handle();
        let mut span = start_span(&handle, "chat Agent");
        apply_error_attributes(&mut span, &agent_span(), AttributeScope::Agent);
        let data = snapshot(&span);
        assert!(matches!(data.status, Status::Error { .. }));
        assert_eq!(
            attribute(&data, OJ_AGENT_ERROR_MESSAGE).as_deref(),
            Some("")
        );
        assert_eq!(
            attribute(&data, OJ_STATUS).as_deref(),
            Some(SPAN_STATUS_ERROR)
        );
        span.end();
    }

    #[test]
    fn interrupt_attributes_never_look_like_a_failure() {
        let handle = tracer_handle();
        for scope in [AttributeScope::Agent, AttributeScope::Workflow] {
            let mut span = start_span(&handle, "component.c1");
            let mut trace_span = agent_span();
            trace_span.status = SpanStatus::Interrupted;
            trace_span.error = Some("awaiting input".to_owned());
            apply_interrupt_attributes(&mut span, &trace_span, scope);
            let data = snapshot(&span);
            assert_eq!(
                attribute(&data, OJ_STATUS).as_deref(),
                Some(SPAN_STATUS_INTERRUPTED)
            );
            assert!(attribute(&data, OJ_ERROR).is_none());
            assert!(attribute(&data, ERROR_TYPE).is_none());
            assert!(data.events.is_empty(), "no exception event on an interrupt");
            assert!(
                !matches!(data.status, Status::Error { .. }),
                "an interrupt must not set the error status"
            );
            let workflow_text = attribute(&data, OJ_WORKFLOW_ERROR_MESSAGE);
            if scope == AttributeScope::Workflow {
                assert_eq!(workflow_text.as_deref(), Some("awaiting input"));
            } else {
                assert!(workflow_text.is_none());
            }
            span.end();
        }
    }

    #[test]
    fn interrupt_attributes_skip_an_empty_payload() {
        let handle = tracer_handle();
        let mut span = start_span(&handle, "component.c1");
        apply_interrupt_attributes(&mut span, &agent_span(), AttributeScope::Workflow);
        let data = snapshot(&span);
        assert!(attribute(&data, OJ_WORKFLOW_ERROR_MESSAGE).is_none());
        span.end();
    }

    #[test]
    fn json_attribute_helpers_skip_absent_values() {
        let handle = tracer_handle();
        let mut span = start_span(&handle, "chat Agent");
        set_json_attribute_opt(&mut span, OJ_AGENT_INPUTS, None);
        set_json_attribute(&mut span, OJ_AGENT_OUTPUTS, &json!({"answer": 42}));
        let data = snapshot(&span);
        assert!(attribute(&data, OJ_AGENT_INPUTS).is_none());
        assert_eq!(
            attribute(&data, OJ_AGENT_OUTPUTS).as_deref(),
            Some(r#"{"answer":42}"#)
        );
        // A JSON string attribute carries the raw text, not quoted JSON.
        set_json_attribute_opt(&mut span, OJ_AGENT_NAME, Some(&json!("天气助手")));
        assert_eq!(
            attribute(&snapshot(&span), OJ_AGENT_NAME).as_deref(),
            Some("天气助手")
        );
        span.end();
    }

    #[test]
    fn set_text_attribute_keeps_non_ascii_and_empty_values() {
        let handle = tracer_handle();
        let mut span = start_span(&handle, "chat Agent");
        set_text_attribute(&mut span, OJ_AGENT_NAME, "天气助手");
        set_text_attribute(&mut span, GEN_AI_REQUEST_MODEL, "");
        let data = snapshot(&span);
        assert_eq!(attribute(&data, OJ_AGENT_NAME).as_deref(), Some("天气助手"));
        assert_eq!(attribute(&data, GEN_AI_REQUEST_MODEL).as_deref(), Some(""));
        span.end();
    }

    #[test]
    fn span_status_name_covers_every_session_status() {
        assert_eq!(span_status_name(SpanStatus::Start), SPAN_STATUS_START);
        assert_eq!(span_status_name(SpanStatus::Running), SPAN_STATUS_RUNNING);
        assert_eq!(span_status_name(SpanStatus::Finish), SPAN_STATUS_FINISH);
        assert_eq!(
            span_status_name(SpanStatus::Interrupted),
            SPAN_STATUS_INTERRUPTED
        );
        assert_eq!(span_status_name(SpanStatus::Error), SPAN_STATUS_ERROR);
        // The wire names mirror the session enum exactly.
        for status in [
            SpanStatus::Start,
            SpanStatus::Running,
            SpanStatus::Finish,
            SpanStatus::Interrupted,
            SpanStatus::Error,
        ] {
            assert_eq!(span_status_name(status), status.as_str());
        }
    }

    #[test]
    fn writes_to_a_non_recording_span_are_ignored() {
        let handle = tracer_handle();
        let mut span = start_span(&handle, "chat Agent");
        span.end();
        // Post-end writes must not panic and must not resurrect the span.
        apply_start_attributes(&mut span, &agent_span());
        apply_end_attributes(&mut span, &agent_span());
        apply_error_attributes(&mut span, &agent_span(), AttributeScope::Agent);
        apply_interrupt_attributes(&mut span, &agent_span(), AttributeScope::Workflow);
        assert!(
            span.exported_data().is_none(),
            "an ended span keeps no data"
        );
    }
}
