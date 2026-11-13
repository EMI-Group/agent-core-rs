//! The callback trait and the per-agent registry that dispatches by event.
//!
//! [`AgentCallback`] is the trait a handler implements: a stable name, a
//! priority, and one no-op default hook per lifecycle event. [`CallbackFramework`]
//! keeps the handlers of one agent grouped by event in stable descending
//! priority order; [`CallbackFramework::invoke`] fires only the hook matching
//! the payload's event and stops at the first handler that returns an error.
use super::event::AgentCallbackEvent;
use super::payload::{
    CallbackPayload, InvokeEndPayload, InvokeStartPayload, StepEndPayload, StepStartPayload,
    ToolCallEndPayload, ToolCallPayload,
};
use openjiuwen_common::error::Result;
use std::cmp::Reverse;
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;
/// Priority a callback gets when it does not override
/// [`AgentCallback::priority`].
pub const DEFAULT_CALLBACK_PRIORITY: i32 = 50;
/// A handler invoked at one or more agent lifecycle events.
///
/// Every hook has a no-op default, so a callback overrides only the events it
/// subscribes to. Returning `Err` aborts dispatch (and the agent run): the
/// error propagates to the caller unchanged.
#[async_trait::async_trait]
pub trait AgentCallback: Send + Sync {
    /// Stable name of the callback; registering two callbacks with one name for
    /// the same event replaces the first.
    fn name(&self) -> &str;
    /// Dispatch priority; higher runs first. Defaults to
    /// [`DEFAULT_CALLBACK_PRIORITY`].
    fn priority(&self) -> i32 {
        DEFAULT_CALLBACK_PRIORITY
    }
    /// A new invocation starts.
    async fn on_invoke_start(&self, _payload: &InvokeStartPayload) -> Result<()> {
        Ok(())
    }
    /// One `ReAct` step starts.
    async fn on_step_start(&self, _payload: &StepStartPayload) -> Result<()> {
        Ok(())
    }
    /// One `ReAct` step ends.
    async fn on_step_end(&self, _payload: &StepEndPayload) -> Result<()> {
        Ok(())
    }
    /// One tool call starts.
    async fn on_tool_call_start(&self, _payload: &ToolCallPayload) -> Result<()> {
        Ok(())
    }
    /// One tool call ends.
    async fn on_tool_call_end(&self, _payload: &ToolCallEndPayload) -> Result<()> {
        Ok(())
    }
    /// The invocation ends.
    async fn on_invoke_end(&self, _payload: &InvokeEndPayload) -> Result<()> {
        Ok(())
    }
}
/// The callbacks of one agent, grouped by the event they subscribe to.
///
/// Registration keeps each event's list stable-sorted by descending priority,
/// so callbacks of equal priority keep their insertion order and a re-registered
/// name keeps the position of the callback it replaces.
#[derive(Default)]
pub struct CallbackFramework {
    handlers: BTreeMap<AgentCallbackEvent, Vec<Arc<dyn AgentCallback>>>,
}
impl CallbackFramework {
    /// Creates an empty framework.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
    /// Registers `callback` for `event`, replacing a callback of the same name
    /// and re-sorting by descending priority.
    pub fn register(&mut self, event: AgentCallbackEvent, callback: Arc<dyn AgentCallback>) {
        let handlers = self.handlers.entry(event).or_default();
        match handlers
            .iter_mut()
            .find(|existing| existing.name() == callback.name())
        {
            Some(existing) => *existing = callback,
            None => handlers.push(callback),
        }
        handlers.sort_by_key(|callback| Reverse(callback.priority()));
    }
    /// Removes the callback named `name` from `event`.
    ///
    /// Returns `true` when a callback was removed.
    #[must_use]
    pub fn unregister(&mut self, event: AgentCallbackEvent, name: &str) -> bool {
        let Some(handlers) = self.handlers.get_mut(&event) else {
            return false;
        };
        let before = handlers.len();
        handlers.retain(|callback| callback.name() != name);
        handlers.len() != before
    }
    /// Removes the callbacks of `event`, or of every event when `event` is
    /// `None`.
    pub fn clear(&mut self, event: Option<AgentCallbackEvent>) {
        match event {
            Some(event) => {
                self.handlers.remove(&event);
            }
            None => self.handlers.clear(),
        }
    }
    /// Whether any callback is registered for `event`.
    #[must_use]
    pub fn has_hooks(&self, event: AgentCallbackEvent) -> bool {
        self.handlers
            .get(&event)
            .is_some_and(|handlers| !handlers.is_empty())
    }
    /// The callbacks of `event` in firing order (stable descending priority).
    #[must_use]
    pub fn handlers(&self, event: AgentCallbackEvent) -> Vec<Arc<dyn AgentCallback>> {
        self.handlers.get(&event).cloned().unwrap_or_default()
    }
    /// Fires the hook matching `payload.event()` on every callback of that
    /// event, in stored order.
    ///
    /// Callbacks registered for other events are never called. Dispatch stops
    /// at the first callback returning `Err`: later callbacks do not run and the
    /// error propagates to the caller, which aborts the run. An empty registry
    /// returns immediately without allocating.
    ///
    /// # Errors
    ///
    /// Returns the first error a callback raises.
    pub async fn invoke(&self, payload: &CallbackPayload) -> Result<()> {
        let Some(handlers) = self.handlers.get(&payload.event()) else {
            return Ok(());
        };
        for callback in handlers {
            match payload {
                CallbackPayload::InvokeStart(payload) => {
                    callback.on_invoke_start(payload).await?;
                }
                CallbackPayload::StepStart(payload) => {
                    callback.on_step_start(payload).await?;
                }
                CallbackPayload::StepEnd(payload) => {
                    callback.on_step_end(payload).await?;
                }
                CallbackPayload::ToolCallStart(payload) => {
                    callback.on_tool_call_start(payload).await?;
                }
                CallbackPayload::ToolCallEnd(payload) => {
                    callback.on_tool_call_end(payload).await?;
                }
                CallbackPayload::InvokeEnd(payload) => {
                    callback.on_invoke_end(payload).await?;
                }
            }
        }
        Ok(())
    }
}
// `dyn AgentCallback` is not `Debug`, so the registry is summarized as the
// per-event handler counts instead of printing the callbacks themselves.
impl fmt::Debug for CallbackFramework {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut debug = f.debug_struct("CallbackFramework");
        for event in AgentCallbackEvent::ALL {
            if let Some(handlers) = self.handlers.get(&event) {
                debug.field(event.as_str(), &handlers.len());
            }
        }
        debug.finish()
    }
}
#[cfg(test)]
mod tests {
    use super::{AgentCallback, CallbackFramework};
    use crate::callbacks::{
        AgentCallbackEvent, CallbackPayload, InvokeEndPayload, InvokeStartPayload,
        StepStartPayload, ToolCallPayload,
    };
    use openjiuwen_common::error::{Error, Result};
    use std::sync::{Arc, Mutex};
    /// A callback that records each `on_step_start` firing and optionally fails.
    struct Recorder {
        name: String,
        priority: i32,
        fail: bool,
        seen: Arc<Mutex<Vec<String>>>,
    }
    impl Recorder {
        fn recording(
            name: &str,
            priority: i32,
            seen: &Arc<Mutex<Vec<String>>>,
        ) -> Arc<dyn AgentCallback> {
            Arc::new(Self {
                name: name.to_owned(),
                priority,
                fail: false,
                seen: Arc::clone(seen),
            })
        }
        fn failing(
            name: &str,
            priority: i32,
            seen: &Arc<Mutex<Vec<String>>>,
        ) -> Arc<dyn AgentCallback> {
            Arc::new(Self {
                name: name.to_owned(),
                priority,
                fail: true,
                seen: Arc::clone(seen),
            })
        }
        fn record(&self, event: &str) {
            self.seen
                .lock()
                .expect("recorder lock")
                .push(format!("{}:{event}", self.name));
        }
    }
    #[async_trait::async_trait]
    impl AgentCallback for Recorder {
        fn name(&self) -> &str {
            &self.name
        }
        fn priority(&self) -> i32 {
            self.priority
        }
        async fn on_step_start(&self, payload: &StepStartPayload) -> Result<()> {
            self.record(&format!("step_start:{}", payload.iteration));
            if self.fail {
                return Err(Error::validation("boom"));
            }
            Ok(())
        }
        async fn on_invoke_start(&self, _payload: &InvokeStartPayload) -> Result<()> {
            self.record("invoke_start");
            Ok(())
        }
        async fn on_invoke_end(&self, _payload: &InvokeEndPayload) -> Result<()> {
            self.record("invoke_end");
            Ok(())
        }
        async fn on_tool_call_start(&self, _payload: &ToolCallPayload) -> Result<()> {
            self.record("tool_call_start");
            Ok(())
        }
    }
    fn step_start(iteration: usize) -> CallbackPayload {
        CallbackPayload::StepStart(StepStartPayload {
            agent_id: "agent-1".to_owned(),
            session_id: Some("session-1".to_owned()),
            iteration,
            ..StepStartPayload::default()
        })
    }
    fn tool_call_start() -> CallbackPayload {
        CallbackPayload::ToolCallStart(ToolCallPayload {
            agent_id: "agent-1".to_owned(),
            ..ToolCallPayload::default()
        })
    }
    fn names(framework: &CallbackFramework, event: AgentCallbackEvent) -> Vec<String> {
        framework
            .handlers(event)
            .iter()
            .map(|callback| callback.name().to_owned())
            .collect()
    }
    #[test]
    fn register_and_unregister_and_clear() {
        let mut framework = CallbackFramework::new();
        let seen = Arc::new(Mutex::new(Vec::new()));
        assert!(!framework.has_hooks(AgentCallbackEvent::StepStart));
        assert!(framework.handlers(AgentCallbackEvent::StepStart).is_empty());
        framework.register(
            AgentCallbackEvent::StepStart,
            Recorder::recording("a", 10, &seen),
        );
        assert!(framework.has_hooks(AgentCallbackEvent::StepStart));
        assert!(!framework.has_hooks(AgentCallbackEvent::StepEnd));
        assert!(framework.unregister(AgentCallbackEvent::StepStart, "a"));
        assert!(!framework.has_hooks(AgentCallbackEvent::StepStart));
        assert!(!framework.unregister(AgentCallbackEvent::StepStart, "a"));
        framework.register(
            AgentCallbackEvent::StepStart,
            Recorder::recording("a", 10, &seen),
        );
        framework.register(
            AgentCallbackEvent::InvokeEnd,
            Recorder::recording("b", 10, &seen),
        );
        framework.clear(Some(AgentCallbackEvent::StepStart));
        assert!(!framework.has_hooks(AgentCallbackEvent::StepStart));
        assert!(framework.has_hooks(AgentCallbackEvent::InvokeEnd));
        framework.clear(None);
        assert!(!framework.has_hooks(AgentCallbackEvent::InvokeEnd));
    }
    #[test]
    fn registering_a_name_again_replaces_it() {
        let mut framework = CallbackFramework::new();
        let seen = Arc::new(Mutex::new(Vec::new()));
        framework.register(
            AgentCallbackEvent::StepStart,
            Recorder::recording("a", 10, &seen),
        );
        framework.register(
            AgentCallbackEvent::StepStart,
            Recorder::recording("b", 20, &seen),
        );
        framework.register(
            AgentCallbackEvent::StepStart,
            Recorder::recording("a", 30, &seen),
        );
        assert_eq!(
            names(&framework, AgentCallbackEvent::StepStart),
            vec!["a".to_owned(), "b".to_owned()]
        );
        assert_eq!(framework.handlers(AgentCallbackEvent::StepStart).len(), 2);
    }
    #[test]
    fn handlers_are_priority_ordered_and_stable() {
        let mut framework = CallbackFramework::new();
        let seen = Arc::new(Mutex::new(Vec::new()));
        for (name, priority) in [("a", 10), ("b", 30), ("c", 30), ("d", 0)] {
            framework.register(
                AgentCallbackEvent::StepStart,
                Recorder::recording(name, priority, &seen),
            );
        }
        assert_eq!(
            names(&framework, AgentCallbackEvent::StepStart),
            vec![
                "b".to_owned(),
                "c".to_owned(),
                "a".to_owned(),
                "d".to_owned()
            ]
        );
    }
    #[tokio::test]
    async fn dispatch_is_priority_ordered_and_stable() {
        let mut framework = CallbackFramework::new();
        let seen = Arc::new(Mutex::new(Vec::new()));
        for (name, priority) in [("a", 10), ("b", 30), ("c", 30)] {
            framework.register(
                AgentCallbackEvent::StepStart,
                Recorder::recording(name, priority, &seen),
            );
        }
        framework
            .invoke(&step_start(1))
            .await
            .expect("dispatch succeeds");
        assert_eq!(
            *seen.lock().expect("seen lock"),
            vec![
                "b:step_start:1".to_owned(),
                "c:step_start:1".to_owned(),
                "a:step_start:1".to_owned()
            ]
        );
    }
    #[tokio::test]
    async fn a_step_start_handler_is_not_called_for_other_events() {
        let mut framework = CallbackFramework::new();
        let seen = Arc::new(Mutex::new(Vec::new()));
        framework.register(
            AgentCallbackEvent::StepStart,
            Recorder::recording("a", 10, &seen),
        );
        framework
            .invoke(&tool_call_start())
            .await
            .expect("dispatch succeeds");
        assert!(seen.lock().expect("seen lock").is_empty());
        framework
            .invoke(&CallbackPayload::InvokeStart(InvokeStartPayload {
                agent_id: "agent-1".to_owned(),
                ..InvokeStartPayload::default()
            }))
            .await
            .expect("dispatch succeeds");
        assert!(seen.lock().expect("seen lock").is_empty());
        framework
            .invoke(&step_start(2))
            .await
            .expect("dispatch succeeds");
        assert_eq!(
            *seen.lock().expect("seen lock"),
            vec!["a:step_start:2".to_owned()]
        );
    }
    #[tokio::test]
    async fn dispatch_stops_at_the_first_error() {
        let mut framework = CallbackFramework::new();
        let seen = Arc::new(Mutex::new(Vec::new()));
        framework.register(
            AgentCallbackEvent::StepStart,
            Recorder::failing("bad", 30, &seen),
        );
        framework.register(
            AgentCallbackEvent::StepStart,
            Recorder::recording("later", 10, &seen),
        );
        let error = framework
            .invoke(&step_start(3))
            .await
            .expect_err("the first error aborts dispatch");
        assert_eq!(error.to_string(), "validation: boom");
        let seen = seen.lock().expect("seen lock");
        assert_eq!(*seen, vec!["bad:step_start:3".to_owned()]);
    }
    #[tokio::test]
    async fn empty_framework_invokes_ok() {
        let framework = CallbackFramework::new();
        for event in AgentCallbackEvent::ALL {
            assert!(!framework.has_hooks(event));
        }
        let payloads = [
            step_start(1),
            tool_call_start(),
            CallbackPayload::InvokeEnd(InvokeEndPayload::default()),
        ];
        for payload in &payloads {
            framework.invoke(payload).await.expect("no-op dispatch");
        }
    }
    #[test]
    fn debug_lists_handler_counts_without_debugging_callbacks() {
        let mut framework = CallbackFramework::new();
        let seen = Arc::new(Mutex::new(Vec::new()));
        assert_eq!(format!("{framework:?}"), "CallbackFramework");
        framework.register(
            AgentCallbackEvent::StepStart,
            Recorder::recording("a", 10, &seen),
        );
        framework.register(
            AgentCallbackEvent::StepStart,
            Recorder::recording("b", 20, &seen),
        );
        framework.register(
            AgentCallbackEvent::InvokeEnd,
            Recorder::recording("c", 10, &seen),
        );
        let debug = format!("{framework:?}");
        assert!(debug.contains("step_start: 2"), "{debug}");
        assert!(debug.contains("invoke_end: 1"), "{debug}");
        assert!(!debug.contains("callback"), "{debug}");
    }
}
