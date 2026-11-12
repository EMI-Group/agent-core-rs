//! Handlers and the two closure adapters.
//!
//! A handler is async by contract: guardrails scan asynchronously, and a host
//! handler may await I/O. [`SyncHandler`] adapts a plain closure, [`AsyncHandler`]
//! adapts a closure that returns a boxed future; both are registered through
//! [`super::CallbackFramework::register_sync`] / `register_async`.
#![allow(clippy::module_name_repetitions)]
// Handlers may compute their name at runtime, so the trait cannot return `&'static str`.
#![allow(clippy::unnecessary_literal_bound)]
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::unnecessary_wraps
    )
)]

use async_trait::async_trait;
use futures::future::BoxFuture;

use super::context::{CallbackContext, CallbackVerdict};
use openjiuwen_common::error::Result;

/// What a handler returns.
pub type CallbackResult = Result<CallbackVerdict>;

/// One subscriber to one event.
///
/// Returning `Err` is recorded in the report and dispatch continues (Python's
/// swallowed callback exception). Returning [`CallbackVerdict::Abort`] stops
/// dispatch and propagates the error out of `trigger`.
#[async_trait]
pub trait CallbackHandler: Send + Sync {
    /// The name shown in reports and logs.
    fn name(&self) -> &str {
        "handler"
    }

    /// Handles one event dispatch.
    async fn handle(&self, ctx: &CallbackContext) -> CallbackResult;
}

/// Adapts a synchronous closure to [`CallbackHandler`].
///
/// The closure receives the context by reference; it cannot mutate it (the
/// framework hands the same context to every handler).
pub struct SyncHandler<F> {
    handler: F,
    name: String,
}

impl<F> SyncHandler<F>
where
    F: Fn(&CallbackContext) -> CallbackResult + Send + Sync,
{
    /// Wraps `handler`, naming it `"sync_handler"`.
    #[must_use]
    pub fn new(handler: F) -> Self {
        Self {
            handler,
            name: "sync_handler".to_owned(),
        }
    }

    /// Overrides the name reported in [`super::CallbackInfo`].
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }
}

#[async_trait]
impl<F> CallbackHandler for SyncHandler<F>
where
    F: Fn(&CallbackContext) -> CallbackResult + Send + Sync,
{
    fn name(&self) -> &str {
        &self.name
    }

    async fn handle(&self, ctx: &CallbackContext) -> CallbackResult {
        (self.handler)(ctx)
    }
}

/// Adapts a closure that produces a boxed future to [`CallbackHandler`].
///
/// The context is passed by value because the returned future is `'static` and
/// may outlive the borrow; [`CallbackContext`] is cheap to clone.
pub struct AsyncHandler<F> {
    handler: F,
    name: String,
}

impl<F> AsyncHandler<F>
where
    F: Fn(CallbackContext) -> BoxFuture<'static, CallbackResult> + Send + Sync,
{
    /// Wraps `handler`, naming it `"async_handler"`.
    #[must_use]
    pub fn new(handler: F) -> Self {
        Self {
            handler,
            name: "async_handler".to_owned(),
        }
    }

    /// Overrides the name reported in [`super::CallbackInfo`].
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }
}

#[async_trait]
impl<F> CallbackHandler for AsyncHandler<F>
where
    F: Fn(CallbackContext) -> BoxFuture<'static, CallbackResult> + Send + Sync,
{
    fn name(&self) -> &str {
        &self.name
    }

    async fn handle(&self, ctx: &CallbackContext) -> CallbackResult {
        (self.handler)(ctx.clone()).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn context() -> CallbackContext {
        CallbackContext::new(super::super::CallbackEvent::AgentStarted)
    }

    #[tokio::test]
    async fn sync_handler_runs_a_plain_closure() {
        let handler = SyncHandler::new(|ctx: &CallbackContext| {
            Ok(if ctx.component == "agent" {
                CallbackVerdict::Continue
            } else {
                CallbackVerdict::Stop
            })
        });
        assert_eq!(handler.name(), "sync_handler");
        let ctx = context().with_component("agent");
        assert!(matches!(
            handler.handle(&ctx).await.unwrap(),
            CallbackVerdict::Continue
        ));
    }

    #[tokio::test]
    async fn sync_handler_can_be_named() {
        let handler = SyncHandler::new(|_: &CallbackContext| Ok(CallbackVerdict::Continue))
            .with_name("recorder");
        assert_eq!(handler.name(), "recorder");
    }

    #[tokio::test]
    async fn sync_handler_propagates_errors() {
        let handler = SyncHandler::new(|_: &CallbackContext| {
            Err(openjiuwen_common::error::Error::validation("nope"))
        });
        let error = handler.handle(&context()).await.unwrap_err();
        assert!(error.to_string().contains("nope"));
    }

    #[tokio::test]
    async fn async_handler_sees_the_context_by_value() {
        let handler = AsyncHandler::new(|ctx: CallbackContext| {
            Box::pin(async move {
                Ok(if ctx.input == Some(json!(1)) {
                    CallbackVerdict::Continue
                } else {
                    CallbackVerdict::Stop
                })
            }) as BoxFuture<'static, CallbackResult>
        });
        assert_eq!(handler.name(), "async_handler");
        let ctx = context().with_input(json!(1));
        assert!(matches!(
            handler.handle(&ctx).await.unwrap(),
            CallbackVerdict::Continue
        ));
    }

    #[tokio::test]
    async fn async_handler_can_be_named() {
        let handler = AsyncHandler::new(|_: CallbackContext| {
            Box::pin(async move { Ok(CallbackVerdict::Continue) })
                as BoxFuture<'static, CallbackResult>
        })
        .with_name("async_recorder");
        assert_eq!(handler.name(), "async_recorder");
    }
}
