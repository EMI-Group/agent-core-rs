//! Model, tool, and loop hooks around the agent runtime.
//!
//! Ports the hook model of Python
//! `openjiuwen/core/single_agent/rail/base.py`: [`AgentRail`] is the trait a
//! hook implements, [`RailContext`] carries the per-event data plus the control
//! flags a rail may set, and [`RailRegistry`] keeps the rails in
//! descending-priority order and dispatches by [`RailEvent`].
//!
//! The loop owns the firing points; a rail never mutates the agent directly, it
//! only mutates the [`RailContext`] handed to it.
pub mod guardrail;
pub mod interrupt;
pub mod memory;
pub mod model_backup;

pub use guardrail::{GuardrailRail, StageFlags};
pub use interrupt::{AskUserRail, BaseInterruptRail, ConfirmInterruptRail, ToolFilter};
pub use memory::MemoryRail;
pub use model_backup::ModelBackupRail;

use crate::abilities::ToolCallResult;
use crate::input::AgentInput;
use crate::interrupt::InterruptRequest;
use crate::output::AgentOutput;
use crate::react::model_step::ModelReply;
use openjiuwen_common::error::Result;
use openjiuwen_foundation::{BaseMessage, Model, ToolCall, ToolCard};
use openjiuwen_session::Session;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

/// Priority a rail gets when it does not override [`AgentRail::priority`].
pub const DEFAULT_RAIL_PRIORITY: i32 = 50;

/// A firing point of the agent loop.
///
/// Wire names are the Python hook names (`before_invoke`, `after_invoke`,
/// `before_model_call`, `after_model_call`, `on_model_exception`,
/// `before_tool_call`, `after_tool_call`, `on_tool_exception`,
/// `after_react_iteration`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RailEvent {
    /// A new invocation starts, before any message is built.
    #[default]
    BeforeInvoke,
    /// The invocation finished and its output is final.
    AfterInvoke,
    /// About to call the model with the prepared window.
    BeforeModelCall,
    /// The model returned a reply.
    AfterModelCall,
    /// The model call failed.
    OnModelException,
    /// About to execute one tool call.
    BeforeToolCall,
    /// One tool call returned.
    AfterToolCall,
    /// The tool call failed.
    OnToolException,
    /// One `ReAct` iteration finished (tool messages appended).
    AfterReactIteration,
}

/// A request to call the model again after an exception.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RetryRequest {
    /// Delay before the retry; `None` means "retry immediately".
    pub delay: Option<Duration>,
}

impl RetryRequest {
    /// Builds a retry request with an optional delay.
    #[must_use]
    pub fn new(delay: Option<Duration>) -> Self {
        Self { delay }
    }

    /// Builds an immediate retry request.
    #[must_use]
    pub fn immediate() -> Self {
        Self { delay: None }
    }

    /// Builds a retry request that waits `delay` first.
    #[must_use]
    pub fn delayed(delay: Duration) -> Self {
        Self { delay: Some(delay) }
    }
}

/// The data and control flags one rail hook observes.
///
/// Only the fields relevant to [`RailContext::event`] are populated. `extra`
/// lives for the whole invocation (Python `ctx.extra`) and is where the loop
/// seeds resume answers and auto-confirm flags.
pub struct RailContext {
    /// The event being fired.
    pub event: RailEvent,
    /// Session of the running invocation, when one exists.
    pub session: Option<Arc<dyn Session>>,
    /// Id of the agent the invocation belongs to.
    pub agent_id: String,
    /// Current loop iteration, `1`-based.
    pub iteration: usize,
    /// Parsed invocation input.
    pub input: Option<AgentInput>,
    /// Messages of the current model request.
    pub messages: Vec<BaseMessage>,
    /// Tools exposed to the model in this call.
    pub tools: Vec<ToolCard>,
    /// Model reply (`AfterModelCall`).
    pub response: Option<ModelReply>,
    /// The tool call being executed (`BeforeToolCall`/`AfterToolCall`).
    pub tool_call: Option<ToolCall>,
    /// The tool result (`AfterToolCall`/`OnToolException`).
    pub tool_result: Option<ToolCallResult>,
    /// Terminal output of the invocation (`AfterInvoke`).
    pub output: Option<AgentOutput>,
    /// Per-invocation scratch space shared by all rails.
    pub extra: BTreeMap<String, Value>,
    /// Output a rail wants the loop to return immediately.
    pub force_finish: Option<AgentOutput>,
    /// Whether the loop should continue instead of finishing.
    pub model_continue: bool,
    /// A pending model retry request.
    pub retry: Option<RetryRequest>,
    /// Replacement text for a tool the rail wants skipped.
    pub skip_tool: Option<String>,
    /// Interrupt requests raised during this firing point.
    pub interrupts: Vec<InterruptRequest>,
    /// Model a rail wants the loop to use instead of the configured one.
    pub model_override: Option<Arc<dyn Model>>,
}

impl RailContext {
    /// Builds an empty context for `event`.
    #[must_use]
    pub fn new(event: RailEvent) -> Self {
        Self {
            event,
            session: None,
            agent_id: String::new(),
            iteration: 0,
            input: None,
            messages: Vec::new(),
            tools: Vec::new(),
            response: None,
            tool_call: None,
            tool_result: None,
            output: None,
            extra: BTreeMap::new(),
            force_finish: None,
            model_continue: false,
            retry: None,
            skip_tool: None,
            interrupts: Vec::new(),
            model_override: None,
        }
    }

    /// Attaches the invocation session.
    #[must_use]
    pub fn with_session(mut self, session: Arc<dyn Session>) -> Self {
        self.session = Some(session);
        self
    }

    /// Sets the agent id.
    #[must_use]
    pub fn with_agent_id(mut self, agent_id: impl Into<String>) -> Self {
        self.agent_id = agent_id.into();
        self
    }

    /// Sets the loop iteration.
    #[must_use]
    pub fn with_iteration(mut self, iteration: usize) -> Self {
        self.iteration = iteration;
        self
    }

    /// Sets the parsed invocation input.
    #[must_use]
    pub fn with_input(mut self, input: AgentInput) -> Self {
        self.input = Some(input);
        self
    }

    /// Sets the messages of the model request.
    #[must_use]
    pub fn with_messages(mut self, messages: Vec<BaseMessage>) -> Self {
        self.messages = messages;
        self
    }

    /// Sets the tools exposed to the model.
    #[must_use]
    pub fn with_tools(mut self, tools: Vec<ToolCard>) -> Self {
        self.tools = tools;
        self
    }

    /// Asks the loop to finish now with `output`.
    pub fn request_force_finish(&mut self, output: AgentOutput) {
        self.force_finish = Some(output);
    }

    /// Takes the requested force-finish output, if any.
    #[must_use]
    pub fn take_force_finish(&mut self) -> Option<AgentOutput> {
        self.force_finish.take()
    }

    /// Asks the loop to continue instead of finishing.
    pub fn request_model_continue(&mut self) {
        self.model_continue = true;
    }

    /// Takes the model-continue request, clearing the flag.
    #[must_use]
    pub fn take_model_continue(&mut self) -> bool {
        std::mem::take(&mut self.model_continue)
    }

    /// Asks the loop to retry the model call, optionally after `delay`.
    pub fn request_retry(&mut self, delay: Option<Duration>) {
        self.retry = Some(RetryRequest::new(delay));
    }

    /// Takes the pending retry request, if any.
    #[must_use]
    pub fn take_retry(&mut self) -> Option<RetryRequest> {
        self.retry.take()
    }

    /// Replaces the tool result with `message` instead of executing the tool.
    pub fn skip_tool_with(&mut self, message: impl Into<String>) {
        self.skip_tool = Some(message.into());
    }

    /// Takes the skip-tool replacement text, if any.
    #[must_use]
    pub fn take_skip_tool(&mut self) -> Option<String> {
        self.skip_tool.take()
    }

    /// Records an interrupt request raised by this rail.
    pub fn request_interrupt(&mut self, request: InterruptRequest) {
        self.interrupts.push(request);
    }

    /// Whether any interrupt has been raised for this firing point.
    #[must_use]
    pub fn is_interrupted(&self) -> bool {
        !self.interrupts.is_empty()
    }

    /// Swaps the model the loop uses for the next call.
    pub fn set_model_override(&mut self, model: Arc<dyn Model>) {
        self.model_override = Some(model);
    }
}

// Trait objects (`Arc<dyn Session>`, `Arc<dyn Model>`) are not `Debug`, so the
// fields holding them are summarized instead of printed.
#[allow(clippy::missing_fields_in_debug)]
impl fmt::Debug for RailContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RailContext")
            .field("event", &self.event)
            .field("agent_id", &self.agent_id)
            .field("iteration", &self.iteration)
            .field("messages", &self.messages.len())
            .field("tools", &self.tools.len())
            .field("response", &self.response.is_some())
            .field("tool_call", &self.tool_call.is_some())
            .field("tool_result", &self.tool_result.is_some())
            .field("output", &self.output.is_some())
            .field("is_interrupted", &self.is_interrupted())
            .field("model_override", &self.model_override.is_some())
            .finish()
    }
}

/// A hook the loop fires around its model, tool, and iteration steps.
///
/// Every hook has a no-op default, so a rail overrides only what it needs. A
/// rail returns `Err` to abort the run: the registry stops dispatching and the
/// error propagates to the caller.
#[async_trait::async_trait]
pub trait AgentRail: Send + Sync {
    /// Stable name of the rail; registering two rails with one name replaces
    /// the first.
    fn name(&self) -> &str;

    /// Dispatch priority; higher runs first. Defaults to
    /// [`DEFAULT_RAIL_PRIORITY`].
    fn priority(&self) -> i32 {
        DEFAULT_RAIL_PRIORITY
    }

    /// A new invocation starts.
    async fn before_invoke(&self, _ctx: &mut RailContext) -> Result<()> {
        Ok(())
    }

    /// The invocation finished.
    async fn after_invoke(&self, _ctx: &mut RailContext) -> Result<()> {
        Ok(())
    }

    /// About to call the model.
    async fn before_model_call(&self, _ctx: &mut RailContext) -> Result<()> {
        Ok(())
    }

    /// The model returned a reply.
    async fn after_model_call(&self, _ctx: &mut RailContext) -> Result<()> {
        Ok(())
    }

    /// The model call failed.
    async fn on_model_exception(&self, _ctx: &mut RailContext) -> Result<()> {
        Ok(())
    }

    /// About to execute one tool call.
    async fn before_tool_call(&self, _ctx: &mut RailContext) -> Result<()> {
        Ok(())
    }

    /// One tool call returned.
    async fn after_tool_call(&self, _ctx: &mut RailContext) -> Result<()> {
        Ok(())
    }

    /// The tool call failed.
    async fn on_tool_exception(&self, _ctx: &mut RailContext) -> Result<()> {
        Ok(())
    }

    /// One `ReAct` iteration finished.
    async fn after_react_iteration(&self, _ctx: &mut RailContext) -> Result<()> {
        Ok(())
    }
}

/// The priority-ordered set of rails a loop fires.
///
/// Registration keeps the rails stable-sorted by descending priority, so rails
/// of equal priority keep their insertion order.
#[derive(Clone, Default)]
pub struct RailRegistry {
    rails: Vec<Arc<dyn AgentRail>>,
}

impl RailRegistry {
    /// Creates an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds `rail`, replacing a rail of the same name, and re-sorts by priority.
    pub fn register(&mut self, rail: Arc<dyn AgentRail>) {
        match self
            .rails
            .iter_mut()
            .find(|existing| existing.name() == rail.name())
        {
            Some(existing) => *existing = rail,
            None => self.rails.push(rail),
        }
        self.rails
            .sort_by_key(|rail| std::cmp::Reverse(rail.priority()));
    }

    /// Removes the rail named `name`.
    pub fn unregister(&mut self, name: &str) {
        self.rails.retain(|rail| rail.name() != name);
    }

    /// Returns the rails in firing order (descending priority).
    #[must_use]
    pub fn rails(&self) -> &[Arc<dyn AgentRail>] {
        &self.rails
    }

    /// Returns the number of registered rails.
    #[must_use]
    pub fn len(&self) -> usize {
        self.rails.len()
    }

    /// Whether no rail is registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rails.is_empty()
    }

    /// Fires the hook matching `ctx.event` on each rail, in stored order
    /// (stable descending priority, ties in insertion order).
    ///
    /// Dispatch stops at the first rail returning `Err`: later rails do not
    /// run and the error propagates to the caller, which aborts the run.
    ///
    /// # Errors
    ///
    /// Returns the first error a rail raises.
    pub async fn fire(&self, ctx: &mut RailContext) -> Result<()> {
        for rail in &self.rails {
            match ctx.event {
                RailEvent::BeforeInvoke => rail.before_invoke(ctx).await?,
                RailEvent::AfterInvoke => rail.after_invoke(ctx).await?,
                RailEvent::BeforeModelCall => rail.before_model_call(ctx).await?,
                RailEvent::AfterModelCall => rail.after_model_call(ctx).await?,
                RailEvent::OnModelException => rail.on_model_exception(ctx).await?,
                RailEvent::BeforeToolCall => rail.before_tool_call(ctx).await?,
                RailEvent::AfterToolCall => rail.after_tool_call(ctx).await?,
                RailEvent::OnToolException => rail.on_tool_exception(ctx).await?,
                RailEvent::AfterReactIteration => rail.after_react_iteration(ctx).await?,
            }
        }
        Ok(())
    }
}

impl fmt::Debug for RailRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RailRegistry")
            .field(
                "rails",
                &self
                    .rails
                    .iter()
                    .map(|rail| rail.name())
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openjiuwen_foundation::ToolCall;
    use std::sync::Mutex;

    /// A rail that only reports a name and priority.
    struct TestRail {
        name: &'static str,
        priority: i32,
    }

    #[async_trait::async_trait]
    impl AgentRail for TestRail {
        fn name(&self) -> &str {
            self.name
        }

        fn priority(&self) -> i32 {
            self.priority
        }
    }

    fn rail(name: &'static str, priority: i32) -> Arc<dyn AgentRail> {
        Arc::new(TestRail { name, priority })
    }

    /// A rail that logs the hook it was called for.
    ///
    /// Every hook records `"<name>:<wire event>"` into `log` so a test can
    /// check which rails ran and in which order. A `fail` rail returns
    /// [`openjiuwen_common::error::Error::config`] after recording.
    struct RecordingRail {
        name: &'static str,
        priority: i32,
        log: Arc<Mutex<Vec<String>>>,
        fail: bool,
    }

    impl RecordingRail {
        fn record(&self, hook: &str) -> Result<()> {
            if let Ok(mut log) = self.log.lock() {
                log.push(format!("{}:{hook}", self.name));
            }
            if self.fail {
                Err(openjiuwen_common::error::Error::config("boom"))
            } else {
                Ok(())
            }
        }
    }

    #[async_trait::async_trait]
    impl AgentRail for RecordingRail {
        fn name(&self) -> &str {
            self.name
        }

        fn priority(&self) -> i32 {
            self.priority
        }

        async fn before_invoke(&self, _ctx: &mut RailContext) -> Result<()> {
            self.record("before_invoke")
        }

        async fn after_invoke(&self, _ctx: &mut RailContext) -> Result<()> {
            self.record("after_invoke")
        }

        async fn before_model_call(&self, _ctx: &mut RailContext) -> Result<()> {
            self.record("before_model_call")
        }

        async fn after_model_call(&self, _ctx: &mut RailContext) -> Result<()> {
            self.record("after_model_call")
        }

        async fn on_model_exception(&self, _ctx: &mut RailContext) -> Result<()> {
            self.record("on_model_exception")
        }

        async fn before_tool_call(&self, _ctx: &mut RailContext) -> Result<()> {
            self.record("before_tool_call")
        }

        async fn after_tool_call(&self, _ctx: &mut RailContext) -> Result<()> {
            self.record("after_tool_call")
        }

        async fn on_tool_exception(&self, _ctx: &mut RailContext) -> Result<()> {
            self.record("on_tool_exception")
        }

        async fn after_react_iteration(&self, _ctx: &mut RailContext) -> Result<()> {
            self.record("after_react_iteration")
        }
    }

    fn recorder(
        name: &'static str,
        priority: i32,
        log: &Arc<Mutex<Vec<String>>>,
    ) -> Arc<dyn AgentRail> {
        Arc::new(RecordingRail {
            name,
            priority,
            log: Arc::clone(log),
            fail: false,
        })
    }

    fn failing_recorder(
        name: &'static str,
        priority: i32,
        log: &Arc<Mutex<Vec<String>>>,
    ) -> Arc<dyn AgentRail> {
        Arc::new(RecordingRail {
            name,
            priority,
            log: Arc::clone(log),
            fail: true,
        })
    }

    fn entries(log: &Arc<Mutex<Vec<String>>>) -> Vec<String> {
        log.lock().map(|log| log.clone()).unwrap_or_default()
    }

    /// A rail that only implements `before_tool_call`.
    struct OnlyToolRail {
        log: Arc<Mutex<Vec<String>>>,
    }

    #[async_trait::async_trait]
    impl AgentRail for OnlyToolRail {
        fn name(&self) -> &'static str {
            "only_tool"
        }

        async fn before_tool_call(&self, _ctx: &mut RailContext) -> Result<()> {
            if let Ok(mut log) = self.log.lock() {
                log.push("only_tool:before_tool_call".to_owned());
            }
            Ok(())
        }
    }

    #[test]
    fn default_priority_is_used_when_not_overridden() {
        struct Plain;
        #[async_trait::async_trait]
        impl AgentRail for Plain {
            fn name(&self) -> &'static str {
                "plain"
            }
        }
        assert_eq!(Plain.priority(), DEFAULT_RAIL_PRIORITY);
        assert_eq!(DEFAULT_RAIL_PRIORITY, 50);
    }

    #[test]
    fn register_sorts_descending_and_keeps_ties() {
        let mut registry = RailRegistry::new();
        registry.register(rail("low", 10));
        registry.register(rail("high", 90));
        registry.register(rail("tie_a", 50));
        registry.register(rail("tie_b", 50));
        let names: Vec<&str> = registry.rails().iter().map(|rail| rail.name()).collect();
        assert_eq!(names, vec!["high", "tie_a", "tie_b", "low"]);
        assert_eq!(registry.rails()[0].priority(), 90);
        assert_eq!(registry.len(), 4);
        assert!(!registry.is_empty());
    }

    #[test]
    fn same_name_registration_replaces_in_place() {
        let mut registry = RailRegistry::new();
        registry.register(rail("only", 70));
        registry.register(rail("only", 20));
        assert_eq!(registry.len(), 1);
        assert_eq!(registry.rails()[0].priority(), 20);
    }

    #[test]
    fn unregister_removes_by_name() {
        let mut registry = RailRegistry::new();
        registry.register(rail("a", 10));
        registry.register(rail("b", 20));
        registry.unregister("a");
        assert_eq!(registry.len(), 1);
        assert_eq!(registry.rails()[0].name(), "b");
        registry.unregister("missing");
        assert_eq!(registry.len(), 1);
        registry.unregister("b");
        assert!(registry.is_empty());
    }

    #[tokio::test]
    async fn fire_dispatches_in_priority_order() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut registry = RailRegistry::new();
        registry.register(recorder("low", 10, &log));
        registry.register(recorder("high", 90, &log));
        registry.register(recorder("tie_a", 50, &log));
        registry.register(recorder("tie_b", 50, &log));

        let mut ctx = RailContext::new(RailEvent::BeforeInvoke);
        registry.fire(&mut ctx).await.unwrap();
        assert_eq!(
            entries(&log),
            vec![
                "high:before_invoke",
                "tie_a:before_invoke",
                "tie_b:before_invoke",
                "low:before_invoke",
            ]
        );
        // The hooks only observe; they do not mutate the context.
        assert!(ctx.extra.is_empty());
    }

    #[tokio::test]
    async fn fire_stops_at_the_first_error() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut registry = RailRegistry::new();
        registry.register(failing_recorder("boom", 90, &log));
        registry.register(recorder("later", 10, &log));

        let mut ctx = RailContext::new(RailEvent::BeforeInvoke);
        let err = registry.fire(&mut ctx).await.unwrap_err();
        assert!(
            matches!(err, openjiuwen_common::error::Error::Config(_)),
            "unexpected error: {err:?}"
        );
        assert_eq!(entries(&log), vec!["boom:before_invoke"]);
    }

    #[tokio::test]
    async fn fire_dispatches_only_the_hook_matching_the_event() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut registry = RailRegistry::new();
        registry.register(recorder("rec", 10, &log));

        let cases = [
            (RailEvent::BeforeInvoke, "before_invoke"),
            (RailEvent::AfterInvoke, "after_invoke"),
            (RailEvent::BeforeModelCall, "before_model_call"),
            (RailEvent::AfterModelCall, "after_model_call"),
            (RailEvent::OnModelException, "on_model_exception"),
            (RailEvent::BeforeToolCall, "before_tool_call"),
            (RailEvent::AfterToolCall, "after_tool_call"),
            (RailEvent::OnToolException, "on_tool_exception"),
            (RailEvent::AfterReactIteration, "after_react_iteration"),
        ];
        for (index, (event, _wire)) in cases.iter().enumerate() {
            let mut ctx = RailContext::new(*event);
            registry.fire(&mut ctx).await.unwrap();
            assert_eq!(
                entries(&log),
                (0..=index)
                    .map(|i| format!("rec:{}", cases[i].1))
                    .collect::<Vec<_>>(),
                "unexpected log after firing {}",
                cases[index].1
            );
        }
        assert_eq!(entries(&log).len(), cases.len());
    }

    #[tokio::test]
    async fn fire_does_not_call_a_rail_for_another_event() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut registry = RailRegistry::new();
        registry.register(Arc::new(OnlyToolRail {
            log: Arc::clone(&log),
        }));

        let mut ctx = RailContext::new(RailEvent::BeforeInvoke);
        registry.fire(&mut ctx).await.unwrap();
        assert_eq!(entries(&log), Vec::<String>::new());

        let mut ctx = RailContext::new(RailEvent::BeforeToolCall);
        registry.fire(&mut ctx).await.unwrap();
        assert_eq!(entries(&log), vec!["only_tool:before_tool_call"]);
    }

    #[tokio::test]
    async fn fire_with_no_rails_is_a_no_op() {
        let registry = RailRegistry::new();
        let mut ctx = RailContext::new(RailEvent::AfterInvoke);
        registry.fire(&mut ctx).await.unwrap();
    }

    #[tokio::test]
    async fn fire_skips_an_unregistered_rail() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut registry = RailRegistry::new();
        registry.register(recorder("gone", 10, &log));
        registry.unregister("gone");

        let mut ctx = RailContext::new(RailEvent::BeforeInvoke);
        registry.fire(&mut ctx).await.unwrap();
        assert_eq!(entries(&log), Vec::<String>::new());
    }

    #[test]
    fn force_finish_round_trips() {
        let mut ctx = RailContext::new(RailEvent::AfterReactIteration);
        assert!(ctx.take_force_finish().is_none());
        ctx.request_force_finish(AgentOutput::answer("done"));
        assert_eq!(ctx.take_force_finish(), Some(AgentOutput::answer("done")));
        assert!(ctx.force_finish.is_none());
    }

    #[test]
    fn model_continue_round_trips() {
        let mut ctx = RailContext::new(RailEvent::AfterModelCall);
        assert!(!ctx.take_model_continue());
        ctx.request_model_continue();
        assert!(ctx.take_model_continue());
        assert!(!ctx.take_model_continue());
    }

    #[test]
    fn retry_round_trips() {
        let mut ctx = RailContext::new(RailEvent::OnModelException);
        assert!(ctx.take_retry().is_none());
        ctx.request_retry(Some(Duration::from_millis(5)));
        assert_eq!(
            ctx.take_retry(),
            Some(RetryRequest::delayed(Duration::from_millis(5)))
        );
        ctx.request_retry(None);
        assert_eq!(ctx.take_retry(), Some(RetryRequest::immediate()));
    }

    #[test]
    fn skip_tool_round_trips() {
        let mut ctx = RailContext::new(RailEvent::BeforeToolCall);
        assert!(ctx.take_skip_tool().is_none());
        ctx.skip_tool_with("not allowed");
        assert_eq!(ctx.take_skip_tool().as_deref(), Some("not allowed"));
        assert!(ctx.take_skip_tool().is_none());
    }

    #[test]
    fn interrupts_mark_the_context_interrupted() {
        let mut ctx = RailContext::new(RailEvent::BeforeToolCall);
        assert!(!ctx.is_interrupted());
        ctx.request_interrupt(InterruptRequest::new(
            crate::interrupt::InterruptKind::Confirm,
            "owner",
            "ok?",
        ));
        assert!(ctx.is_interrupted());
        assert_eq!(ctx.interrupts.len(), 1);
    }

    #[test]
    fn builders_fill_the_event_fields() {
        let input = AgentInput::default();
        let tool_call = ToolCall {
            name: "bash".to_owned(),
            ..ToolCall::default()
        };
        let mut ctx = RailContext::new(RailEvent::BeforeToolCall)
            .with_agent_id("agent-1")
            .with_iteration(2)
            .with_input(input.clone())
            .with_messages(vec![BaseMessage::user("hi")])
            .with_tools(vec![ToolCard::new("bash", "run", serde_json::json!({}))]);
        ctx.tool_call = Some(tool_call);
        assert_eq!(ctx.agent_id, "agent-1");
        assert_eq!(ctx.iteration, 2);
        assert_eq!(ctx.input, Some(input));
        assert_eq!(ctx.messages.len(), 1);
        assert_eq!(ctx.tools.len(), 1);
        assert!(ctx.tool_call.is_some());
        assert!(ctx.session.is_none());
        assert!(ctx.model_override.is_none());
    }

    #[test]
    fn debug_output_does_not_panic() {
        let ctx = RailContext::new(RailEvent::BeforeInvoke);
        let rendered = format!("{ctx:?}");
        assert!(rendered.contains("BeforeInvoke"));
        assert!(rendered.contains("is_interrupted"));
        let mut registry = RailRegistry::new();
        registry.register(rail("named", 10));
        assert!(format!("{registry:?}").contains("named"));
    }

    #[test]
    fn rail_event_wire_names() {
        let cases = [
            (RailEvent::BeforeInvoke, "before_invoke"),
            (RailEvent::AfterInvoke, "after_invoke"),
            (RailEvent::BeforeModelCall, "before_model_call"),
            (RailEvent::AfterModelCall, "after_model_call"),
            (RailEvent::OnModelException, "on_model_exception"),
            (RailEvent::BeforeToolCall, "before_tool_call"),
            (RailEvent::AfterToolCall, "after_tool_call"),
            (RailEvent::OnToolException, "on_tool_exception"),
            (RailEvent::AfterReactIteration, "after_react_iteration"),
        ];
        for (event, wire) in cases {
            assert_eq!(
                serde_json::to_value(event).unwrap(),
                serde_json::json!(wire)
            );
            let restored: RailEvent = serde_json::from_value(serde_json::json!(wire)).unwrap();
            assert_eq!(restored, event);
        }
        assert_eq!(RailEvent::default(), RailEvent::BeforeInvoke);
    }
}
