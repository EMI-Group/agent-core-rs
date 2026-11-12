//! The priority-ordered, copy-on-write harness rail registry and its kernel bridge.
//!
//! [`HarnessRailRegistry`] is a cheap-to-clone handle over an `Arc`-swapped
//! snapshot of rails. Registration replaces a snapshot, so a rail registered or
//! unregistered during dispatch is not observed by the running firing point and
//! takes effect on the next one; no lock is held while hooks run.
//!
//! [`HarnessRailAdapter`] presents one [`HarnessRail`] as an
//! [`openjiuwen_agent::AgentRail`], so the harness rails can also be installed
//! on the inner `ReActAgent` of the kernel.
use std::fmt;
use std::sync::Arc;

use openjiuwen_agent::{AgentRail, RailContext, RailEvent};
use openjiuwen_common::error::Result;
use parking_lot::RwLock;

use super::{HarnessRail, HarnessRailContext, TaskIterationInfo, TaskIterationPhase};

/// An immutable, shared snapshot of the registered rails.
pub(crate) type RailSnapshot = Arc<Vec<Arc<dyn HarnessRail>>>;

/// The rails a harness loop fires, stable-sorted by descending priority.
#[derive(Clone, Default)]
pub struct HarnessRailRegistry {
    rails: Arc<RwLock<RailSnapshot>>,
}

impl HarnessRailRegistry {
    /// Creates an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds `rail`, replacing a rail of the same name, and re-sorts by priority.
    ///
    /// The rails are stable-sorted by descending [`HarnessRail::priority`], so
    /// rails of equal priority keep their insertion order.
    pub fn register(&self, rail: Arc<dyn HarnessRail>) {
        let mut guard = self.rails.write();
        let current = Arc::clone(&guard);
        let mut next = Vec::with_capacity(current.len() + 1);
        let mut replaced = false;
        for existing in current.iter() {
            if !replaced && existing.name() == rail.name() {
                next.push(Arc::clone(&rail));
                replaced = true;
            } else {
                next.push(Arc::clone(existing));
            }
        }
        if !replaced {
            next.push(rail);
        }
        next.sort_by_key(|rail| std::cmp::Reverse(rail.priority()));
        *guard = Arc::new(next);
    }

    /// Removes the rail named `name`; returns whether one was removed.
    pub fn unregister(&self, name: &str) -> bool {
        let mut guard = self.rails.write();
        let current = Arc::clone(&guard);
        if !current.iter().any(|rail| rail.name() == name) {
            return false;
        }
        let next: Vec<Arc<dyn HarnessRail>> = current
            .iter()
            .filter(|rail| rail.name() != name)
            .map(Arc::clone)
            .collect();
        *guard = Arc::new(next);
        true
    }

    /// Returns the rails in firing order (descending priority).
    #[must_use]
    pub fn rails(&self) -> Vec<Arc<dyn HarnessRail>> {
        self.snapshot().as_ref().clone()
    }

    /// Returns the current snapshot; cloning it is cheap and lock-free afterwards.
    #[must_use]
    pub fn snapshot(&self) -> RailSnapshot {
        Arc::clone(&self.rails.read())
    }

    /// Returns the names of the rails in firing order.
    #[must_use]
    pub fn names(&self) -> Vec<String> {
        self.snapshot()
            .iter()
            .map(|rail| rail.name().to_string())
            .collect()
    }

    /// Returns the number of registered rails.
    #[must_use]
    pub fn len(&self) -> usize {
        self.rails.read().len()
    }

    /// Whether no rail is registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rails.read().is_empty()
    }

    /// Fires the kernel hook matching [`HarnessRailContext::event`] on every
    /// rail, in stored order (stable descending priority, ties in insertion
    /// order).
    ///
    /// # Errors
    ///
    /// Returns the first error a rail raises; later rails do not run.
    pub async fn fire(&self, ctx: &mut HarnessRailContext) -> Result<()> {
        let rails = self.snapshot();
        let event = ctx.event();
        for rail in rails.iter() {
            match event {
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

    /// Fires the task-iteration hook selected by `phase` on every rail, in
    /// stored order.
    ///
    /// `ctx.task` is set to `info` before any hook runs, so every rail observes
    /// the outer-loop information of this firing point.
    ///
    /// # Errors
    ///
    /// Returns the first error a rail raises; later rails do not run.
    pub async fn fire_task_iteration(
        &self,
        ctx: &mut HarnessRailContext,
        info: TaskIterationInfo,
        phase: TaskIterationPhase,
    ) -> Result<()> {
        ctx.task = Some(info);
        let rails = self.snapshot();
        for rail in rails.iter() {
            match phase {
                TaskIterationPhase::Before => rail.before_task_iteration(ctx).await?,
                TaskIterationPhase::After => rail.after_task_iteration(ctx).await?,
            }
        }
        Ok(())
    }

    /// Fires [`HarnessRail::on_user_message`] on every rail, in stored order.
    ///
    /// # Errors
    ///
    /// Returns the first error a rail raises; later rails do not run.
    pub async fn fire_user_message(&self, ctx: &mut HarnessRailContext) -> Result<()> {
        let rails = self.snapshot();
        for rail in rails.iter() {
            rail.on_user_message(ctx).await?;
        }
        Ok(())
    }

    /// Fires [`HarnessRail::before_steering_drain`] on every rail, in stored
    /// order.
    ///
    /// # Errors
    ///
    /// Returns the first error a rail raises; later rails do not run.
    pub async fn fire_steering_drain(&self, ctx: &mut HarnessRailContext) -> Result<()> {
        let rails = self.snapshot();
        for rail in rails.iter() {
            rail.before_steering_drain(ctx).await?;
        }
        Ok(())
    }

    /// Wraps every rail in a [`HarnessRailAdapter`] for the kernel registry.
    #[must_use]
    pub fn to_agent_rails(&self) -> Vec<Arc<dyn AgentRail>> {
        self.snapshot()
            .iter()
            .map(|rail| Arc::new(HarnessRailAdapter::new(Arc::clone(rail))) as Arc<dyn AgentRail>)
            .collect()
    }
}

#[allow(clippy::missing_fields_in_debug)]
impl fmt::Debug for HarnessRailRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HarnessRailRegistry")
            .field("rails", &self.names())
            .finish()
    }
}

/// Presents a [`HarnessRail`] as an [`AgentRail`] of the kernel.
#[derive(Clone)]
pub struct HarnessRailAdapter {
    inner: Arc<dyn HarnessRail>,
}

impl HarnessRailAdapter {
    /// Wraps `inner`.
    #[must_use]
    pub fn new(inner: Arc<dyn HarnessRail>) -> Self {
        Self { inner }
    }

    /// The wrapped rail.
    #[must_use]
    pub fn inner(&self) -> &Arc<dyn HarnessRail> {
        &self.inner
    }
}

/// Forwards one kernel hook to the wrapped rail.
///
/// The kernel context is moved into a [`HarnessRailContext`] with
/// [`std::mem::replace`] and copied back after the hook returns — before the
/// result is returned — so every control flag a rail set survives even when the
/// hook aborts the run.
///
/// The macro expands to the hook body rather than to a whole `async fn`:
/// `#[async_trait::async_trait]` is expanded before `macro_rules!`, so it cannot
/// rewrite a macro-generated `async fn` into the boxed-future signature the
/// trait expects. The receiver is passed in explicitly because a `macro_rules!`
/// body cannot refer to the `self` of its expansion site.
macro_rules! forward_hook {
    ($this:ident, $hook:ident, $ctx:ident) => {{
        let event = $ctx.event;
        let mut harness_ctx =
            HarnessRailContext::new(std::mem::replace($ctx, RailContext::new(event)));
        let result = $this.inner.$hook(&mut harness_ctx).await;
        *$ctx = harness_ctx.base;
        result
    }};
}

#[async_trait::async_trait]
impl AgentRail for HarnessRailAdapter {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn priority(&self) -> i32 {
        self.inner.priority()
    }

    async fn before_invoke(&self, ctx: &mut RailContext) -> Result<()> {
        forward_hook!(self, before_invoke, ctx)
    }

    async fn after_invoke(&self, ctx: &mut RailContext) -> Result<()> {
        forward_hook!(self, after_invoke, ctx)
    }

    async fn before_model_call(&self, ctx: &mut RailContext) -> Result<()> {
        forward_hook!(self, before_model_call, ctx)
    }

    async fn after_model_call(&self, ctx: &mut RailContext) -> Result<()> {
        forward_hook!(self, after_model_call, ctx)
    }

    async fn on_model_exception(&self, ctx: &mut RailContext) -> Result<()> {
        forward_hook!(self, on_model_exception, ctx)
    }

    async fn before_tool_call(&self, ctx: &mut RailContext) -> Result<()> {
        forward_hook!(self, before_tool_call, ctx)
    }

    async fn after_tool_call(&self, ctx: &mut RailContext) -> Result<()> {
        forward_hook!(self, after_tool_call, ctx)
    }

    async fn on_tool_exception(&self, ctx: &mut RailContext) -> Result<()> {
        forward_hook!(self, on_tool_exception, ctx)
    }

    async fn after_react_iteration(&self, ctx: &mut RailContext) -> Result<()> {
        forward_hook!(self, after_react_iteration, ctx)
    }
}

impl fmt::Debug for HarnessRailAdapter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HarnessRailAdapter")
            .field("rail", &self.inner.name())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use crate::rails::rail_priority;
    use openjiuwen_agent::{AgentOutput, InterruptKind, InterruptRequest};
    use std::sync::Mutex;
    use std::time::Duration;

    /// One rail hook invocation recorded by [`RecordingRail`].
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Hit {
        Event(RailEvent),
        TaskBefore(TaskIterationInfo),
        TaskAfter(TaskIterationInfo),
        UserMessage,
        SteeringDrain,
    }

    #[derive(Clone, Default)]
    struct Hits(Arc<Mutex<Vec<Hit>>>);

    impl Hits {
        fn record(&self, hit: Hit) {
            self.0.lock().unwrap().push(hit);
        }

        fn list(&self) -> Vec<Hit> {
            self.0.lock().unwrap().clone()
        }

        fn len(&self) -> usize {
            self.0.lock().unwrap().len()
        }
    }

    /// A rail that records every hook it observes.
    struct RecordingRail {
        name: &'static str,
        priority: i32,
        hits: Hits,
    }

    impl RecordingRail {
        fn new(name: &'static str, priority: i32) -> (Arc<Self>, Hits) {
            let hits = Hits::default();
            (
                Arc::new(Self {
                    name,
                    priority,
                    hits: hits.clone(),
                }),
                hits,
            )
        }
    }

    #[async_trait::async_trait]
    impl HarnessRail for RecordingRail {
        fn name(&self) -> &str {
            self.name
        }

        fn priority(&self) -> i32 {
            self.priority
        }

        async fn before_invoke(&self, ctx: &mut HarnessRailContext) -> Result<()> {
            self.hits.record(Hit::Event(ctx.event()));
            Ok(())
        }

        async fn after_invoke(&self, ctx: &mut HarnessRailContext) -> Result<()> {
            self.hits.record(Hit::Event(ctx.event()));
            Ok(())
        }

        async fn before_model_call(&self, ctx: &mut HarnessRailContext) -> Result<()> {
            self.hits.record(Hit::Event(ctx.event()));
            Ok(())
        }

        async fn after_model_call(&self, ctx: &mut HarnessRailContext) -> Result<()> {
            self.hits.record(Hit::Event(ctx.event()));
            Ok(())
        }

        async fn on_model_exception(&self, ctx: &mut HarnessRailContext) -> Result<()> {
            self.hits.record(Hit::Event(ctx.event()));
            Ok(())
        }

        async fn before_tool_call(&self, ctx: &mut HarnessRailContext) -> Result<()> {
            self.hits.record(Hit::Event(ctx.event()));
            Ok(())
        }

        async fn after_tool_call(&self, ctx: &mut HarnessRailContext) -> Result<()> {
            self.hits.record(Hit::Event(ctx.event()));
            Ok(())
        }

        async fn on_tool_exception(&self, ctx: &mut HarnessRailContext) -> Result<()> {
            self.hits.record(Hit::Event(ctx.event()));
            Ok(())
        }

        async fn after_react_iteration(&self, ctx: &mut HarnessRailContext) -> Result<()> {
            self.hits.record(Hit::Event(ctx.event()));
            Ok(())
        }

        async fn on_user_message(&self, _ctx: &mut HarnessRailContext) -> Result<()> {
            self.hits.record(Hit::UserMessage);
            Ok(())
        }

        async fn before_steering_drain(&self, _ctx: &mut HarnessRailContext) -> Result<()> {
            self.hits.record(Hit::SteeringDrain);
            Ok(())
        }

        async fn before_task_iteration(&self, ctx: &mut HarnessRailContext) -> Result<()> {
            self.hits.record(Hit::TaskBefore(*ctx.task_info().unwrap()));
            Ok(())
        }

        async fn after_task_iteration(&self, ctx: &mut HarnessRailContext) -> Result<()> {
            self.hits.record(Hit::TaskAfter(*ctx.task_info().unwrap()));
            Ok(())
        }
    }

    /// A rail that fails before recording anything.
    struct FailingRail {
        name: &'static str,
        priority: i32,
    }

    #[async_trait::async_trait]
    impl HarnessRail for FailingRail {
        fn name(&self) -> &str {
            self.name
        }

        fn priority(&self) -> i32 {
            self.priority
        }

        async fn before_invoke(&self, _ctx: &mut HarnessRailContext) -> Result<()> {
            Err(openjiuwen_common::error::Error::security("blocked"))
        }
    }

    /// A rail that registers another rail during its first `before_invoke`.
    struct RegisteringRail {
        registry: Arc<Mutex<Option<HarnessRailRegistry>>>,
        late: Arc<RecordingRail>,
    }

    #[async_trait::async_trait]
    impl HarnessRail for RegisteringRail {
        fn name(&self) -> &'static str {
            "registering"
        }

        fn priority(&self) -> i32 {
            rail_priority::DEFAULT
        }

        async fn before_invoke(&self, _ctx: &mut HarnessRailContext) -> Result<()> {
            if let Some(registry) = self.registry.lock().unwrap().as_ref() {
                registry.register(Arc::clone(&self.late) as Arc<dyn HarnessRail>);
            }
            Ok(())
        }
    }

    fn named(name: &'static str, priority: i32) -> Arc<dyn HarnessRail> {
        let (rail, _) = RecordingRail::new(name, priority);
        rail as Arc<dyn HarnessRail>
    }

    #[test]
    fn rails_are_ordered_by_descending_priority() {
        let registry = HarnessRailRegistry::new();
        registry.register(named("low", 10));
        registry.register(named("high", 90));
        registry.register(named("middle", 50));
        assert_eq!(registry.names(), ["high", "middle", "low"]);
    }

    #[test]
    fn equal_priorities_keep_insertion_order() {
        let registry = HarnessRailRegistry::new();
        registry.register(named("first", 80));
        registry.register(named("second", 80));
        registry.register(named("third", 80));
        assert_eq!(registry.names(), ["first", "second", "third"]);
    }

    #[test]
    fn register_replaces_a_rail_of_the_same_name() {
        let registry = HarnessRailRegistry::new();
        registry.register(named("rail", 20));
        registry.register(named("rail", 90));
        assert_eq!(registry.len(), 1);
        assert_eq!(registry.rails()[0].priority(), 90);
    }

    #[test]
    fn unregister_reports_whether_a_rail_was_removed() {
        let registry = HarnessRailRegistry::new();
        assert!(registry.is_empty());
        registry.register(named("rail", 20));
        assert_eq!(registry.len(), 1);
        assert!(!registry.is_empty());
        assert!(registry.unregister("rail"));
        assert!(!registry.unregister("rail"));
        assert!(registry.is_empty());
    }

    #[test]
    fn cloning_shares_the_registry_contents() {
        let registry = HarnessRailRegistry::new();
        let clone = registry.clone();
        registry.register(named("rail", 20));
        assert_eq!(clone.names(), ["rail"]);
    }

    #[test]
    fn debug_lists_rail_names() {
        let registry = HarnessRailRegistry::new();
        registry.register(named("rail", 20));
        assert_eq!(
            format!("{registry:?}"),
            "HarnessRailRegistry { rails: [\"rail\"] }"
        );
    }

    #[tokio::test]
    async fn all_nine_kernel_events_are_dispatched() {
        let registry = HarnessRailRegistry::new();
        let (rail, hits) = RecordingRail::new("recorder", 50);
        registry.register(rail);

        let events = [
            RailEvent::BeforeInvoke,
            RailEvent::AfterInvoke,
            RailEvent::BeforeModelCall,
            RailEvent::AfterModelCall,
            RailEvent::OnModelException,
            RailEvent::BeforeToolCall,
            RailEvent::AfterToolCall,
            RailEvent::OnToolException,
            RailEvent::AfterReactIteration,
        ];
        for event in events {
            let mut ctx = HarnessRailContext::for_event(event);
            registry.fire(&mut ctx).await.unwrap();
        }

        let recorded: Vec<Hit> = hits.list();
        assert_eq!(recorded.len(), 9);
        for (recorded, event) in recorded.iter().zip(events) {
            assert_eq!(*recorded, Hit::Event(event));
        }
    }

    /// A rail that appends its name to a shared firing order on `before_invoke`.
    struct OrderRail {
        name: &'static str,
        priority: i32,
        order: Arc<Mutex<Vec<&'static str>>>,
    }

    #[async_trait::async_trait]
    impl HarnessRail for OrderRail {
        fn name(&self) -> &str {
            self.name
        }

        fn priority(&self) -> i32 {
            self.priority
        }

        async fn before_invoke(&self, _ctx: &mut HarnessRailContext) -> Result<()> {
            self.order.lock().unwrap().push(self.name);
            Ok(())
        }
    }

    #[tokio::test]
    async fn dispatch_follows_priority_order() {
        let registry = HarnessRailRegistry::new();
        let order = Arc::new(Mutex::new(Vec::new()));
        for (name, priority) in [("low", 10), ("high", 90), ("middle", 50), ("tie", 90)] {
            registry.register(Arc::new(OrderRail {
                name,
                priority,
                order: Arc::clone(&order),
            }));
        }

        let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeInvoke);
        registry.fire(&mut ctx).await.unwrap();

        assert_eq!(
            *order.lock().unwrap(),
            ["high", "tie", "middle", "low"],
            "higher priority first, ties in insertion order"
        );
    }

    #[tokio::test]
    async fn an_error_aborts_later_rails() {
        let registry = HarnessRailRegistry::new();
        let (late, late_hits) = RecordingRail::new("late", 10);
        registry.register(Arc::new(FailingRail {
            name: "failing",
            priority: 90,
        }));
        registry.register(late);

        let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeInvoke);
        let error = registry.fire(&mut ctx).await.unwrap_err();
        assert!(error.to_string().contains("blocked"));
        assert_eq!(late_hits.len(), 0);
    }

    #[tokio::test]
    async fn task_iteration_hooks_see_the_iteration_info_per_phase() {
        let registry = HarnessRailRegistry::new();
        let (rail, hits) = RecordingRail::new("recorder", 50);
        registry.register(rail);

        let info = TaskIterationInfo::new(4, 1).with_follow_up();
        let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeInvoke);
        registry
            .fire_task_iteration(&mut ctx, info, TaskIterationPhase::Before)
            .await
            .unwrap();
        assert_eq!(ctx.task_info(), Some(&info));

        registry
            .fire_task_iteration(&mut ctx, info, TaskIterationPhase::After)
            .await
            .unwrap();

        assert_eq!(hits.list(), [Hit::TaskBefore(info), Hit::TaskAfter(info)]);
    }

    #[tokio::test]
    async fn user_message_and_steering_hooks_are_dispatched() {
        let registry = HarnessRailRegistry::new();
        let (rail, hits) = RecordingRail::new("recorder", 50);
        registry.register(rail);

        let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeInvoke);
        registry.fire_user_message(&mut ctx).await.unwrap();
        registry.fire_steering_drain(&mut ctx).await.unwrap();

        assert_eq!(hits.list(), [Hit::UserMessage, Hit::SteeringDrain]);
    }

    #[tokio::test]
    async fn a_rail_registered_during_fire_is_seen_only_on_the_next_fire() {
        let registry = Arc::new(HarnessRailRegistry::new());
        let slot = Arc::new(Mutex::new(None::<HarnessRailRegistry>));
        *slot.lock().unwrap() = Some((*registry).clone());
        let (late, late_hits) = RecordingRail::new("late", 90);

        registry.register(Arc::new(RegisteringRail {
            registry: Arc::clone(&slot),
            late: Arc::clone(&late),
        }));

        let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeInvoke);
        registry.fire(&mut ctx).await.unwrap();
        // The registration already landed in the registry, but the snapshot the
        // running dispatch iterates over did not include it.
        assert_eq!(registry.len(), 2);
        assert_eq!(late_hits.len(), 0);

        registry.fire(&mut ctx).await.unwrap();
        assert_eq!(late_hits.len(), 1);
        assert_eq!(registry.names(), ["late", "registering"]);
    }

    #[test]
    fn to_agent_rails_wraps_every_rail_in_firing_order() {
        let registry = HarnessRailRegistry::new();
        registry.register(named("low", 10));
        registry.register(named("high", 90));

        let rails = registry.to_agent_rails();
        let names: Vec<&str> = rails.iter().map(|rail| rail.name()).collect();
        assert_eq!(names, ["high", "low"]);
        assert_eq!(rails[0].priority(), 90);
    }

    /// A rail that sets every control flag it can, in `before_invoke`.
    struct FlaggingRail {
        hits: Hits,
        fail: bool,
    }

    #[async_trait::async_trait]
    impl HarnessRail for FlaggingRail {
        fn name(&self) -> &'static str {
            "flagging"
        }

        fn priority(&self) -> i32 {
            rail_priority::DEFAULT
        }

        async fn before_invoke(&self, ctx: &mut HarnessRailContext) -> Result<()> {
            self.hits.record(Hit::Event(ctx.event()));
            ctx.request_skip_tool("skipped");
            ctx.request_force_finish(AgentOutput::answer("forced"));
            ctx.request_interrupt(InterruptRequest::new(
                InterruptKind::Permission,
                "flagging",
                "allow?",
            ));
            ctx.request_model_continue();
            ctx.request_retry(Some(Duration::from_millis(5)));
            if self.fail {
                return Err(openjiuwen_common::error::Error::security("flag failure"));
            }
            Ok(())
        }
    }

    #[tokio::test]
    async fn adapter_forwards_every_kernel_event_exactly_once() {
        let registry = HarnessRailRegistry::new();
        let (rail, hits) = RecordingRail::new("recorder", 50);
        registry.register(rail);
        let adapters = registry.to_agent_rails();
        let adapter = &adapters[0];
        assert_eq!(adapter.name(), "recorder");

        let events = [
            RailEvent::BeforeInvoke,
            RailEvent::AfterInvoke,
            RailEvent::BeforeModelCall,
            RailEvent::AfterModelCall,
            RailEvent::OnModelException,
            RailEvent::BeforeToolCall,
            RailEvent::AfterToolCall,
            RailEvent::OnToolException,
            RailEvent::AfterReactIteration,
        ];
        for event in events {
            let mut kernel_ctx = RailContext::new(event);
            match event {
                RailEvent::BeforeInvoke => adapter.before_invoke(&mut kernel_ctx).await.unwrap(),
                RailEvent::AfterInvoke => adapter.after_invoke(&mut kernel_ctx).await.unwrap(),
                RailEvent::BeforeModelCall => {
                    adapter.before_model_call(&mut kernel_ctx).await.unwrap();
                }
                RailEvent::AfterModelCall => {
                    adapter.after_model_call(&mut kernel_ctx).await.unwrap();
                }
                RailEvent::OnModelException => {
                    adapter.on_model_exception(&mut kernel_ctx).await.unwrap();
                }
                RailEvent::BeforeToolCall => {
                    adapter.before_tool_call(&mut kernel_ctx).await.unwrap();
                }
                RailEvent::AfterToolCall => {
                    adapter.after_tool_call(&mut kernel_ctx).await.unwrap();
                }
                RailEvent::OnToolException => {
                    adapter.on_tool_exception(&mut kernel_ctx).await.unwrap();
                }
                RailEvent::AfterReactIteration => {
                    adapter
                        .after_react_iteration(&mut kernel_ctx)
                        .await
                        .unwrap();
                }
            }
            assert_eq!(kernel_ctx.event, event);
        }

        assert_eq!(hits.len(), 9);
    }

    #[tokio::test]
    async fn adapter_copies_every_control_flag_back() {
        let adapter = HarnessRailAdapter::new(Arc::new(FlaggingRail {
            hits: Hits::default(),
            fail: false,
        }));
        let mut ctx = RailContext::new(RailEvent::BeforeInvoke);
        adapter.before_invoke(&mut ctx).await.unwrap();

        assert_eq!(ctx.skip_tool.as_deref(), Some("skipped"));
        assert_eq!(
            ctx.force_finish.as_ref().map(|out| out.output.as_str()),
            Some("forced")
        );
        assert_eq!(ctx.interrupts.len(), 1);
        assert!(ctx.model_continue);
        assert_eq!(
            ctx.retry.as_ref().and_then(|retry| retry.delay),
            Some(Duration::from_millis(5))
        );
    }

    #[tokio::test]
    async fn adapter_keeps_flags_when_the_hook_fails() {
        let adapter = HarnessRailAdapter::new(Arc::new(FlaggingRail {
            hits: Hits::default(),
            fail: true,
        }));
        let mut ctx = RailContext::new(RailEvent::BeforeInvoke);
        let error = adapter.before_invoke(&mut ctx).await.unwrap_err();
        assert!(error.to_string().contains("flag failure"));

        assert_eq!(ctx.event, RailEvent::BeforeInvoke);
        assert_eq!(ctx.skip_tool.as_deref(), Some("skipped"));
        assert!(ctx.force_finish.is_some());
        assert_eq!(ctx.interrupts.len(), 1);
        assert!(ctx.model_continue);
        assert!(ctx.retry.is_some());
    }

    #[test]
    fn adapter_exposes_the_inner_rail() {
        let (rail, _) = RecordingRail::new("inner", 42);
        let adapter = HarnessRailAdapter::new(rail);
        assert_eq!(adapter.inner().name(), "inner");
        assert_eq!(adapter.priority(), 42);
        assert_eq!(
            format!("{adapter:?}"),
            "HarnessRailAdapter { rail: \"inner\" }"
        );
    }
}
