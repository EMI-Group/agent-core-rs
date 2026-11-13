//! The emit combinators: fire an auxiliary event around the guarded one.
//!
//! Ports Python's `create_emit_before_decorator` / `create_emit_after_decorator`
//! / around form. Each combinator builds an [`EmitHandler`] over a
//! `CallbackFramework`; running it fires the auxiliary event with a derived
//! payload, and an auxiliary abort becomes the source event's abort while an
//! auxiliary stop becomes the source event's stop.
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
use crate::callback::context::{CallbackContext, CallbackVerdict};
use crate::callback::event::CallbackEvent;
use crate::callback::framework::CallbackFramework;
use crate::callback::handler::{CallbackHandler, CallbackResult};
use async_trait::async_trait;
use serde_json::Value;
use std::fmt;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// emit_before / emit_after / emit_around
// ---------------------------------------------------------------------------

enum EmitKind {
    Before(CallbackEvent),
    After(CallbackEvent),
    Around {
        before: CallbackEvent,
        after: CallbackEvent,
    },
}

/// Fires auxiliary events around the event it is registered for.
///
/// Python decorated a function with an event trigger; here the combinator is
/// itself a handler, so it fires its auxiliary event while the source event is
/// being dispatched. An auxiliary abort becomes [`CallbackVerdict::Abort`] on the
/// source event (Python's propagated `AbortError`), an auxiliary stop becomes
/// [`CallbackVerdict::Stop`], and a clean auxiliary dispatch lets the source
/// dispatch continue.
pub struct EmitHandler {
    name: String,
    framework: Arc<CallbackFramework>,
    kind: EmitKind,
}

impl EmitHandler {
    /// Overrides the name reported to the framework.
    #[must_use]
    pub fn named(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Fires one auxiliary event with the source payload.
    async fn fire(
        &self,
        source: &CallbackContext,
        event: CallbackEvent,
        with_output: bool,
    ) -> CallbackResult {
        let aux = aux_context(source, event.clone(), with_output);
        match self.framework.trigger(event, &aux).await {
            Ok(report) if report.aborted => Ok(CallbackVerdict::Stop),
            Ok(_) => Ok(CallbackVerdict::Continue),
            Err(error) => Ok(CallbackVerdict::Abort(error)),
        }
    }
}

/// Builds the context an auxiliary event sees: the source's session, component,
/// input, index, metadata, and guardrail content, plus its output when asked.
fn aux_context(
    source: &CallbackContext,
    event: CallbackEvent,
    with_output: bool,
) -> CallbackContext {
    let mut aux = CallbackContext::new(event)
        .with_component(source.component.clone())
        .with_metadata(
            "source_event",
            Value::String(source.event.as_str().to_owned()),
        );
    if let Some(session) = &source.session {
        aux = aux.with_session(Arc::clone(session));
    }
    if let Some(input) = &source.input {
        aux = aux.with_input(input.clone());
    }
    if with_output {
        if let Some(output) = &source.output {
            aux = aux.with_output(output.clone());
        }
    }
    aux.index = source.index;
    aux.error.clone_from(&source.error);
    aux.metadata.extend(
        source
            .metadata
            .iter()
            .map(|(key, value)| (key.clone(), value.clone())),
    );
    aux.guardrail.clone_from(&source.guardrail);
    aux
}

#[async_trait]
impl CallbackHandler for EmitHandler {
    fn name(&self) -> &str {
        &self.name
    }

    async fn handle(&self, ctx: &CallbackContext) -> CallbackResult {
        match &self.kind {
            EmitKind::Before(event) => self.fire(ctx, event.clone(), false).await,
            EmitKind::After(event) => self.fire(ctx, event.clone(), true).await,
            EmitKind::Around { before, after } => {
                match self.fire(ctx, before.clone(), false).await? {
                    CallbackVerdict::Continue => {}
                    other => return Ok(other),
                }
                self.fire(ctx, after.clone(), true).await
            }
        }
    }
}

impl fmt::Debug for EmitHandler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EmitHandler")
            .field("name", &self.name)
            .field(
                "kind",
                &match &self.kind {
                    EmitKind::Before(event) => format!("before {}", event.as_str()),
                    EmitKind::After(event) => format!("after {}", event.as_str()),
                    EmitKind::Around { before, after } => {
                        format!("around {} / {}", before.as_str(), after.as_str())
                    }
                },
            )
            .finish_non_exhaustive()
    }
}

/// Fires `event` before the source dispatch continues (Python's `emit_before`).
#[must_use]
pub fn emit_before(framework: Arc<CallbackFramework>, event: CallbackEvent) -> EmitHandler {
    EmitHandler {
        name: format!("emit_before({})", event.as_str()),
        framework,
        kind: EmitKind::Before(event),
    }
}

/// Fires `event` with the source's output payload (Python's `emit_after`).
#[must_use]
pub fn emit_after(framework: Arc<CallbackFramework>, event: CallbackEvent) -> EmitHandler {
    EmitHandler {
        name: format!("emit_after({})", event.as_str()),
        framework,
        kind: EmitKind::After(event),
    }
}

/// Fires `before` then `after` inside one handler.
///
/// Python's `emit_around` wrapped a function call; a handler cannot interpose on
/// the caller's call, so the two auxiliary dispatches happen back to back. Use
/// [`wrap`] when the "after" event must observe work done between the two.
#[must_use]
pub fn emit_around(
    framework: Arc<CallbackFramework>,
    before: CallbackEvent,
    after: CallbackEvent,
) -> EmitHandler {
    EmitHandler {
        name: format!("emit_around({} / {})", before.as_str(), after.as_str()),
        framework,
        kind: EmitKind::Around { before, after },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openjiuwen_common::error::Error;
    use parking_lot::Mutex;
    use serde_json::json;
    #[tokio::test]
    async fn emit_before_and_after_carry_the_payload() {
        let framework = CallbackFramework::new();
        let seen = Arc::new(Mutex::new(Vec::new()));

        let before_seen = Arc::clone(&seen);
        framework.register_sync(
            CallbackEvent::LlmCallStarted,
            0,
            move |ctx: &CallbackContext| {
                before_seen.lock().push(format!(
                    "before:{}",
                    ctx.input.clone().unwrap_or(Value::Null)
                ));
                Ok(CallbackVerdict::Continue)
            },
        );
        let after_seen = Arc::clone(&seen);
        framework.register_sync(
            CallbackEvent::LlmResponseReceived,
            0,
            move |ctx: &CallbackContext| {
                after_seen.lock().push(format!(
                    "after:{}",
                    ctx.output.clone().unwrap_or(Value::Null)
                ));
                Ok(CallbackVerdict::Continue)
            },
        );

        let around = emit_around(
            Arc::clone(&framework),
            CallbackEvent::LlmCallStarted,
            CallbackEvent::LlmResponseReceived,
        );
        let ctx = CallbackContext::new(CallbackEvent::LlmInvokeInput)
            .with_component("assistant")
            .with_input(json!("hi"))
            .with_output(json!("hello"));
        assert!(matches!(
            around.handle(&ctx).await.unwrap(),
            CallbackVerdict::Continue
        ));
        assert_eq!(*seen.lock(), ["before:\"hi\"", "after:\"hello\""]);
        assert!(format!("{around:?}").contains("llm_call_started"));
    }

    #[tokio::test]
    async fn emit_propagates_an_auxiliary_abort() {
        let framework = CallbackFramework::new();
        framework.register_sync(CallbackEvent::ToolInvokeInput, 0, |_: &CallbackContext| {
            Ok(CallbackVerdict::Abort(Error::security("blocked")))
        });
        let before =
            emit_before(Arc::clone(&framework), CallbackEvent::ToolInvokeInput).named("gate");
        assert_eq!(before.name(), "gate");
        let ctx = CallbackContext::new(CallbackEvent::ToolInvokeOutput).with_output(json!("ok"));
        match before.handle(&ctx).await.unwrap() {
            CallbackVerdict::Abort(error) => assert_eq!(error.to_string(), "security: blocked"),
            other => panic!("expected an abort, got {other:?}"),
        }
    }
}
