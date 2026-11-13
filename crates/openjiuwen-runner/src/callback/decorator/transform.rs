//! The `transform_io` combinator: rewrite the input before downstream work and
//! the output afterwards.
//!
//! Ports Python's `create_transform_io_decorator`. [`TransformHandler`] works
//! twice over: as a [`CallbackHandler`] (the `AgentInvokeInput`/`...Output` pair,
//! each half applying only where the transform is configured) and as an
//! [`EventFilter`] (the in-place form, so later chain entries see the rewrite).
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
use crate::callback::filters::EventFilter;
use crate::callback::handler::{CallbackHandler, CallbackResult};
use async_trait::async_trait;
use openjiuwen_common::error::Result;
use parking_lot::Mutex;
use serde_json::Value;
use std::fmt;
use std::future::Future;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// transform_io
// ---------------------------------------------------------------------------

/// One payload rewrite.
pub type PayloadTransform = Arc<dyn Fn(&Value) -> Result<Value> + Send + Sync>;

#[derive(Clone, Default)]
struct TransformState {
    input: Option<Value>,
    output: Option<Value>,
}

/// Rewrites the input payload before downstream work and the output afterwards.
///
/// Ports `create_transform_io_decorator`. Two forms, because
/// [`CallbackHandler::handle`] receives `&CallbackContext`:
///
/// * As an [`EventFilter`] (and therefore via
///   [`HandlerBuilder::when`](crate::callback::decorator::HandlerBuilder::when) or
///   a [`ChainEntry`](crate::callback::ChainEntry)) it rewrites the working
///   context in place, so every later entry observes the transformed payloads.
/// * As a [`CallbackHandler`] it rewrites its own clone of the context and
///   publishes the result through [`Self::last_input`]/[`Self::last_output`].
/// * [`Self::apply`] transforms a direct async call, which is the closest to
///   Python's decorator.
///
/// A transform error is propagated, never swallowed.
pub struct TransformHandler {
    name: String,
    input: Option<PayloadTransform>,
    output: Option<PayloadTransform>,
    state: Arc<Mutex<TransformState>>,
}

impl TransformHandler {
    /// A transform with no rewrites yet.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            input: None,
            output: None,
            state: Arc::new(Mutex::new(TransformState::default())),
        }
    }

    /// Sets the input rewrite.
    #[must_use]
    pub fn with_input(
        mut self,
        transform: impl Fn(&Value) -> Result<Value> + Send + Sync + 'static,
    ) -> Self {
        self.input = Some(Arc::new(transform));
        self
    }

    /// Sets the output rewrite.
    #[must_use]
    pub fn with_output(
        mut self,
        transform: impl Fn(&Value) -> Result<Value> + Send + Sync + 'static,
    ) -> Self {
        self.output = Some(Arc::new(transform));
        self
    }

    /// The last rewritten input this handler produced.
    #[must_use]
    pub fn last_input(&self) -> Option<Value> {
        self.state.lock().input.clone()
    }

    /// The last rewritten output this handler produced.
    #[must_use]
    pub fn last_output(&self) -> Option<Value> {
        self.state.lock().output.clone()
    }

    /// Rewrites `ctx` in place (the filter form).
    ///
    /// # Errors
    ///
    /// Propagates the transform's error.
    pub fn rewrite(&self, ctx: &mut CallbackContext) -> Result<()> {
        if let (Some(transform), Some(input)) = (&self.input, &ctx.input) {
            let rewritten = transform(input)?;
            self.state.lock().input = Some(rewritten.clone());
            ctx.input = Some(rewritten);
        }
        if let (Some(transform), Some(output)) = (&self.output, &ctx.output) {
            let rewritten = transform(output)?;
            self.state.lock().output = Some(rewritten.clone());
            ctx.output = Some(rewritten);
        }
        Ok(())
    }

    /// Runs `call` between the input and output rewrites.
    ///
    /// # Errors
    ///
    /// Propagates a transform error or the call's error.
    pub async fn apply<F, Fut>(&self, input: Value, call: F) -> Result<Value>
    where
        F: FnOnce(Value) -> Fut,
        Fut: Future<Output = Result<Value>>,
    {
        let input = match &self.input {
            Some(transform) => transform(&input)?,
            None => input,
        };
        self.state.lock().input = Some(input.clone());
        let output = call(input).await?;
        let output = match &self.output {
            Some(transform) => transform(&output)?,
            None => output,
        };
        self.state.lock().output = Some(output.clone());
        Ok(output)
    }
}

impl fmt::Debug for TransformHandler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TransformHandler")
            .field("name", &self.name)
            .field("input", &self.input.is_some())
            .field("output", &self.output.is_some())
            .finish_non_exhaustive()
    }
}

/// Builds a [`TransformHandler`] (Python's `transform_io`).
#[must_use]
pub fn transform_io(name: impl Into<String>) -> TransformHandler {
    TransformHandler::new(name)
}

#[async_trait]
impl CallbackHandler for TransformHandler {
    fn name(&self) -> &str {
        &self.name
    }

    async fn handle(&self, ctx: &CallbackContext) -> CallbackResult {
        let mut working = ctx.clone();
        self.rewrite(&mut working)?;
        Ok(CallbackVerdict::Continue)
    }
}

impl EventFilter for TransformHandler {
    fn name(&self) -> &str {
        &self.name
    }

    fn allows(
        &self,
        input: crate::callback::filters::FilterInput<'_>,
    ) -> Result<crate::callback::filters::FilterDecision> {
        self.rewrite(input.ctx)?;
        Ok(crate::callback::filters::FilterDecision::Allow)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::callback::event::CallbackEvent;
    use crate::callback::filters::{FilterDecision, FilterInput, KindFilter};
    use openjiuwen_common::error::Error;
    use serde_json::json;
    #[tokio::test]
    async fn transform_io_rewrites_both_payloads_and_propagates_errors() {
        let transform = transform_io("shout")
            .with_input(|value| Ok(json!(value.as_str().unwrap_or_default().to_uppercase())))
            .with_output(|value| Ok(json!(format!("{value}!"))));
        let output = transform
            .apply(json!("hi"), |input| async move {
                Ok(json!(format!(
                    "{}+call",
                    input.as_str().unwrap_or_default()
                )))
            })
            .await
            .unwrap();
        assert_eq!(output, json!("\"HI+call\"!"));
        assert_eq!(transform.last_input(), Some(json!("HI")));
        assert_eq!(transform.last_output(), Some(json!("\"HI+call\"!")));

        let failing = transform_io("failing").with_input(|_| Err(Error::validation("bad input")));
        let error = failing
            .apply(json!("hi"), |input| async move { Ok(input) })
            .await
            .unwrap_err();
        assert_eq!(error.to_string(), "validation: bad input");

        let ctx = CallbackContext::new(CallbackEvent::LlmInvokeInput).with_input(json!("hi"));
        assert!(matches!(
            EventFilter::allows(
                &transform,
                FilterInput {
                    handler: "h",
                    namespace: "default",
                    ctx: &mut ctx.clone(),
                }
            )
            .unwrap(),
            FilterDecision::Allow
        ));
        assert!(KindFilter::domain(crate::callback::EventDomain::Llm)
            .allows(FilterInput {
                handler: "h",
                namespace: "default",
                ctx: &mut CallbackContext::new(CallbackEvent::LlmInvokeInput),
            })
            .unwrap()
            .is_allow());
    }
}
