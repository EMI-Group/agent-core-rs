//! Client-executed tools: the handler seam and its registry.
//!
//! Ports the client-tool half of `openjiuwen/core/single_agent/ability_manager.py`
//! (`_pause_client_tool_call` / `_finish_client_tool_call`) as a Rust seam: a
//! card with [`ToolExecution::Client`](openjiuwen_foundation::ToolExecution)
//! pauses the agent turn with a `client_tool` interrupt, and the connected
//! client's answer is handed to a [`ClientToolHandler`] to produce the tool
//! result. The registry only stores and dispatches handlers; pausing and
//! resuming is the `ReAct` loop's job.
use async_trait::async_trait;
use futures::future::BoxFuture;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::ToolCall;
use parking_lot::RwLock;
use serde_json::Value;
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;
/// Completes a paused client tool call with the answer the client sent back.
#[async_trait]
pub trait ClientToolHandler: Send + Sync {
    /// Turns the client's `answer` for `call` into the tool result value.
    ///
    /// # Errors
    ///
    /// Returns an error when the answer is unusable or the client side could
    /// not complete the call; the loop renders it as the tool's error result.
    async fn invoke(&self, call: &ToolCall, answer: &Value) -> Result<Value>;
}
/// A boxed asynchronous client-tool function.
type ClientToolFn =
    dyn Fn(ToolCall, Value) -> BoxFuture<'static, Result<Value>> + Send + Sync + 'static;
/// A [`ClientToolHandler`] built from an asynchronous closure.
///
/// The closure receives owned clones of the call and answer, so it can be
/// spawned or awaited freely.
pub struct FnClientToolHandler {
    name: String,
    function: Arc<ClientToolFn>,
}
impl FnClientToolHandler {
    /// Wraps `function` under the diagnostic `name`.
    pub fn new<F>(name: impl Into<String>, function: F) -> Self
    where
        F: Fn(ToolCall, Value) -> BoxFuture<'static, Result<Value>> + Send + Sync + 'static,
    {
        Self {
            name: name.into(),
            function: Arc::new(function),
        }
    }
    /// The name this handler was built with.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }
}
impl fmt::Debug for FnClientToolHandler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FnClientToolHandler")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}
#[async_trait]
impl ClientToolHandler for FnClientToolHandler {
    async fn invoke(&self, call: &ToolCall, answer: &Value) -> Result<Value> {
        (self.function)(call.clone(), answer.clone()).await
    }
}
/// Handlers for the client tools an agent may pause on, keyed by tool name.
///
/// Registration is by the model-facing tool name (the [`ToolCall::name`]).
#[derive(Default)]
pub struct ClientToolRegistry {
    handlers: RwLock<BTreeMap<String, Arc<dyn ClientToolHandler>>>,
}
impl ClientToolRegistry {
    /// Creates an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
    /// Registers `handler` under `name`, replacing any previous handler.
    pub fn register(&self, name: impl Into<String>, handler: Arc<dyn ClientToolHandler>) {
        self.handlers.write().insert(name.into(), handler);
    }
    /// Removes the handler registered under `name`; `true` when one existed.
    pub fn unregister(&self, name: &str) -> bool {
        self.handlers.write().remove(name).is_some()
    }
    /// Returns `true` when `name` has a registered handler.
    #[must_use]
    pub fn is_client_tool(&self, name: &str) -> bool {
        self.handlers.read().contains_key(name)
    }
    /// The handler registered under `name`, if any.
    #[must_use]
    pub fn handler(&self, name: &str) -> Option<Arc<dyn ClientToolHandler>> {
        self.handlers.read().get(name).cloned()
    }
    /// The registered tool names, sorted.
    #[must_use]
    pub fn names(&self) -> Vec<String> {
        self.handlers.read().keys().cloned().collect()
    }
    /// Dispatches `call` to its registered handler.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] when `call.name` has no registered handler;
    /// handler failures propagate unchanged.
    pub async fn invoke(&self, call: &ToolCall, answer: &Value) -> Result<Value> {
        let handler = self.handler(&call.name).ok_or_else(|| {
            Error::NotFound(format!("client tool '{}' is not registered", call.name))
        })?;
        handler.invoke(call, answer).await
    }
}
impl fmt::Debug for ClientToolRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ClientToolRegistry")
            .field("names", &self.names())
            .finish()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;
    use parking_lot::Mutex;
    use serde_json::json;
    use std::sync::Arc;
    #[derive(Debug, Default)]
    struct RecordingHandler {
        calls: Mutex<Vec<(String, Value)>>,
    }
    #[async_trait]
    impl ClientToolHandler for RecordingHandler {
        async fn invoke(&self, call: &ToolCall, answer: &Value) -> Result<Value> {
            self.calls.lock().push((call.name.clone(), answer.clone()));
            Ok(json!({"echo": answer}))
        }
    }
    fn call(name: &str) -> ToolCall {
        ToolCall::new(name)
    }
    #[tokio::test]
    async fn registry_dispatches_and_records_the_answer() {
        let handler = Arc::new(RecordingHandler::default());
        let registry = ClientToolRegistry::new();
        registry.register(
            "run_on_client",
            Arc::clone(&handler) as Arc<dyn ClientToolHandler>,
        );
        assert!(registry.is_client_tool("run_on_client"));
        assert!(!registry.is_client_tool("missing"));
        assert_eq!(registry.names(), vec!["run_on_client".to_owned()]);
        let result = registry
            .invoke(&call("run_on_client"), &json!({"stdout": "ok"}))
            .await
            .expect("dispatch");
        assert_eq!(result, json!({"echo": {"stdout": "ok"}}));
        let recorded = handler.calls.lock();
        assert_eq!(recorded.len(), 1);
        assert_eq!(recorded[0].0, "run_on_client");
        assert_eq!(recorded[0].1, json!({"stdout": "ok"}));
    }
    #[tokio::test]
    async fn unknown_tool_is_not_found() {
        let registry = ClientToolRegistry::new();
        let error = registry
            .invoke(&call("missing"), &Value::Null)
            .await
            .expect_err("unregistered client tool");
        assert_eq!(error.kind(), ErrorKind::NotFound);
        assert!(error.to_string().contains("missing"));
    }
    #[test]
    fn unregister_removes_the_handler() {
        let registry = ClientToolRegistry::new();
        registry.register("tool", Arc::new(RecordingHandler::default()));
        assert!(registry.unregister("tool"));
        assert!(!registry.unregister("tool"));
        assert!(registry.handler("tool").is_none());
    }
    #[tokio::test]
    async fn fn_handler_forwards_to_the_closure() {
        let handler = FnClientToolHandler::new("shout", |call, answer| {
            Box::pin(async move {
                let text = answer.as_str().unwrap_or_default().to_uppercase();
                Ok(json!({"tool": call.name, "text": text}))
            })
        });
        assert_eq!(handler.name(), "shout");
        let registry = ClientToolRegistry::new();
        registry.register("shout", Arc::new(handler));
        let result = registry
            .invoke(&call("shout"), &json!("hello"))
            .await
            .expect("fn handler");
        assert_eq!(result, json!({"tool": "shout", "text": "HELLO"}));
    }
    #[test]
    fn debug_lists_registered_names() {
        let registry = ClientToolRegistry::new();
        registry.register("tool", Arc::new(RecordingHandler::default()));
        let rendered = format!("{registry:?}");
        assert!(rendered.contains("tool"));
    }
}
