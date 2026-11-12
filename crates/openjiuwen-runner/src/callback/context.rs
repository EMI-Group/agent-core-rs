//! The payload passed to every callback handler, and the verdicts they return.
//!
//! Python narrows `*args`/`**kwargs` per handler by signature inspection; here
//! every handler receives one [`CallbackContext`] and reads the fields it needs.
//! The context is deliberately cheap to clone (payloads are `serde_json::Value`
//! and the session is an `Arc`), because an async handler adapter takes it by
//! value.
//!
//! Error isolation lives in [`CallbackReport`]: a handler that returns `Err` is
//! recorded and dispatch continues, exactly like Python's swallowed callback
//! exception. A caller that wants to stop the call returns
//! [`CallbackVerdict::Abort`].
#![allow(clippy::module_name_repetitions)]
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::unnecessary_wraps
    )
)]

use openjiuwen_common::error::{Error, ErrorKind};
use openjiuwen_security::GuardrailContext;
use openjiuwen_session::Session;
use serde_json::Value;
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use super::event::CallbackEvent;
use super::framework::CallbackId;

/// Everything a handler may inspect for one event.
///
/// [`Debug`] is implemented manually: it reports payload *shapes* and metadata
/// keys, never the values (they may contain prompts or tool arguments).
#[derive(Clone)]
pub struct CallbackContext {
    /// The event being dispatched.
    pub event: CallbackEvent,
    /// The session the event belongs to, when the runner has one.
    pub session: Option<Arc<dyn Session>>,
    /// The agent, workflow, team, or component id that produced the event.
    pub component: String,
    /// Input payload, when the event has one.
    pub input: Option<Value>,
    /// Output payload, when the event has one.
    pub output: Option<Value>,
    /// Failure description for error-shaped events.
    pub error: Option<String>,
    /// Frame index for streaming events.
    pub index: Option<i64>,
    /// Extra facts about the event (small, non-payload metadata).
    pub metadata: BTreeMap<String, Value>,
    /// The content a guardrail should scan, when the caller already built it.
    ///
    /// A [`super::GuardrailHandler`] prefers this over deriving one from
    /// [`Self::input`]/[`Self::output`].
    pub guardrail: Option<GuardrailContext>,
}

impl CallbackContext {
    /// An empty context for `event`: no session, empty component, no payloads.
    #[must_use]
    pub fn new(event: CallbackEvent) -> Self {
        Self {
            event,
            session: None,
            component: String::new(),
            input: None,
            output: None,
            error: None,
            index: None,
            metadata: BTreeMap::new(),
            guardrail: None,
        }
    }

    /// Attaches the session the event belongs to.
    #[must_use]
    pub fn with_session(mut self, session: Arc<dyn Session>) -> Self {
        self.session = Some(session);
        self
    }

    /// Sets the producing component id.
    #[must_use]
    pub fn with_component(mut self, component: impl Into<String>) -> Self {
        self.component = component.into();
        self
    }

    /// Sets the input payload.
    #[must_use]
    pub fn with_input(mut self, input: Value) -> Self {
        self.input = Some(input);
        self
    }

    /// Sets the output payload.
    #[must_use]
    pub fn with_output(mut self, output: Value) -> Self {
        self.output = Some(output);
        self
    }

    /// Sets the failure description.
    #[must_use]
    pub fn with_error(mut self, error: impl Into<String>) -> Self {
        self.error = Some(error.into());
        self
    }

    /// Sets the streaming frame index.
    #[must_use]
    pub fn with_index(mut self, index: i64) -> Self {
        self.index = Some(index);
        self
    }

    /// Inserts one metadata entry; a later insert replaces the previous value.
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Sets the content a guardrail should scan.
    #[must_use]
    pub fn with_guardrail(mut self, ctx: GuardrailContext) -> Self {
        self.guardrail = Some(ctx);
        self
    }

    /// The attached session's id, when there is a session.
    #[must_use]
    pub fn session_id(&self) -> Option<&str> {
        self.session.as_deref().map(Session::id)
    }

    /// The JSON kind of a payload, used by [`Debug`] to avoid dumping values.
    fn payload_shape(payload: Option<&Value>) -> Option<&'static str> {
        payload.map(|value| match value {
            Value::Null => "null",
            Value::Bool(_) => "bool",
            Value::Number(_) => "number",
            Value::String(_) => "string",
            Value::Array(_) => "array",
            Value::Object(_) => "object",
        })
    }
}

impl fmt::Debug for CallbackContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CallbackContext")
            .field("event", &self.event)
            .field("session", &self.session_id())
            .field("component", &self.component)
            .field("input", &Self::payload_shape(self.input.as_ref()))
            .field("output", &Self::payload_shape(self.output.as_ref()))
            .field("error", &self.error)
            .field("index", &self.index)
            .field("metadata", &self.metadata.keys().collect::<Vec<_>>())
            .field(
                "guardrail",
                &self.guardrail.as_ref().map(|ctx| ctx.event.as_str()),
            )
            .finish()
    }
}

/// What a handler wants the framework to do next.
///
/// Mirrors Python's three outcomes: returning `None` (here
/// [`Continue`](Self::Continue)), raising `AbortError(cause=...)` (here
/// [`Abort`](Self::Abort) with the propagated error), and
/// `AbortError(is_error=False)` for expected control flow (here
/// [`Stop`](Self::Stop), which ends dispatch without an error).
#[derive(Debug)]
pub enum CallbackVerdict {
    /// Run the next handler.
    Continue,
    /// Stop dispatch; [`super::CallbackFramework::trigger`] returns this error.
    Abort(Error),
    /// Stop dispatch without an error (an interrupt, not a failure).
    Stop,
}

/// One handler that failed during a dispatch.
///
/// Recorded in [`CallbackReport::failures`]; dispatch continued after it. The
/// error is kept verbatim so callers can inspect the kind and message.
#[derive(Debug)]
pub struct CallbackFailure {
    /// The registration the failure came from.
    pub id: CallbackId,
    /// The handler's name.
    pub handler: String,
    /// The error it returned.
    pub error: Error,
}

impl CallbackFailure {
    /// Whether the failure was a security decision (a guardrail abort).
    #[must_use]
    pub fn is_security(&self) -> bool {
        self.error.kind() == ErrorKind::Security
    }
}

/// The outcome of one [`super::CallbackFramework::trigger`] call.
#[derive(Debug)]
pub struct CallbackReport {
    /// The event that was dispatched.
    pub event: CallbackEvent,
    /// How many handlers ran successfully (`Continue` or `Stop`).
    pub fired: usize,
    /// Handlers that returned `Err`, in dispatch order.
    pub failures: Vec<CallbackFailure>,
    /// Whether a handler returned `Abort` or `Stop`.
    pub aborted: bool,
}

impl CallbackReport {
    /// An empty report for `event`.
    #[must_use]
    pub fn new(event: CallbackEvent) -> Self {
        Self {
            event,
            fired: 0,
            failures: Vec::new(),
            aborted: false,
        }
    }

    /// Whether no handler failed and dispatch was not aborted.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.failures.is_empty() && !self.aborted
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn builders_fill_the_context() {
        let context = CallbackContext::new(CallbackEvent::ToolInvokeInput)
            .with_component("search")
            .with_input(json!({"q": "rust"}))
            .with_output(json!(["hit"]))
            .with_error("boom")
            .with_index(3)
            .with_metadata("tool_name", json!("search"))
            .with_metadata("attempt", json!(1));

        assert_eq!(context.event, CallbackEvent::ToolInvokeInput);
        assert_eq!(context.component, "search");
        assert_eq!(context.input, Some(json!({"q": "rust"})));
        assert_eq!(context.output, Some(json!(["hit"])));
        assert_eq!(context.error.as_deref(), Some("boom"));
        assert_eq!(context.index, Some(3));
        assert_eq!(context.metadata["tool_name"], json!("search"));
        assert_eq!(context.metadata["attempt"], json!(1));
        assert!(context.session_id().is_none());
        assert!(context.guardrail.is_none());
    }

    #[test]
    fn a_later_metadata_insert_replaces_the_previous_value() {
        let context = CallbackContext::new(CallbackEvent::AgentStarted)
            .with_metadata("k", json!(1))
            .with_metadata("k", json!(2));
        assert_eq!(context.metadata["k"], json!(2));
        assert_eq!(context.metadata.len(), 1);
    }

    #[test]
    fn guardrail_context_is_attached() {
        let guardrail =
            GuardrailContext::text("check me", openjiuwen_security::ScanStage::LlmInvokeInput);
        let context =
            CallbackContext::new(CallbackEvent::LlmInvokeInput).with_guardrail(guardrail.clone());
        assert_eq!(
            context.guardrail.as_ref().map(|ctx| ctx.content.clone()),
            Some(json!("check me"))
        );
    }

    #[test]
    fn debug_reports_shapes_not_payloads() {
        let context = CallbackContext::new(CallbackEvent::LlmInvokeInput)
            .with_component("assistant")
            .with_input(json!({"secret": "do not log me"}))
            .with_output(json!("also hidden"))
            .with_metadata("conversation", json!("c-1"));

        let debug = format!("{context:?}");
        assert!(debug.contains("CallbackContext"));
        assert!(debug.contains("LlmInvokeInput"));
        assert!(debug.contains("object"));
        assert!(debug.contains("string"));
        assert!(debug.contains("conversation"));
        assert!(!debug.contains("do not log me"), "{debug}");
        assert!(!debug.contains("also hidden"), "{debug}");
    }

    #[test]
    fn debug_handles_absent_payloads() {
        let debug = format!("{:?}", CallbackContext::new(CallbackEvent::AgentStarted));
        assert!(debug.contains("input: None"), "{debug}");
        assert!(debug.contains("guardrail: None"), "{debug}");
    }

    #[test]
    fn a_new_report_is_clean() {
        let report = CallbackReport::new(CallbackEvent::AgentStarted);
        assert_eq!(report.fired, 0);
        assert!(!report.aborted);
        assert!(report.is_clean());
    }

    #[test]
    fn failures_and_aborts_make_a_report_dirty() {
        let mut report = CallbackReport::new(CallbackEvent::AgentStarted);
        report.failures.push(CallbackFailure {
            id: CallbackId::from_raw(1),
            handler: "h".to_owned(),
            error: Error::validation("bad"),
        });
        assert!(!report.is_clean());

        let mut report = CallbackReport::new(CallbackEvent::AgentStarted);
        report.aborted = true;
        assert!(!report.is_clean());
    }

    #[test]
    fn security_failures_are_recognizable() {
        let failure = CallbackFailure {
            id: CallbackId::from_raw(2),
            handler: "guardrail".to_owned(),
            error: Error::Security("blocked".to_owned()),
        };
        assert!(failure.is_security());
    }
}
