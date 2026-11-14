//! Lifecycle resource hooks for subagent instances.
//!
//! Ports the prepare/cleanup resource seam of `openjiuwen/harness/subagent_lifecycle.py` and the
//! per-instance status/activity callbacks of Python `subagent_runtime/session_manager.py` to the
//! in-process Rust runtime.
//! Python prepares a child session's resources before the first turn and tears them down when the
//! instance leaves, and it reports each status/activity transition to per-instance callbacks.
//! Here that seam is one owner-implemented [`SubagentLifecycleHook`] trait with four stages
//! ([`LifecycleStage::Spawn`], [`LifecycleStage::Ready`], [`LifecycleStage::BeforeEvict`],
//! [`LifecycleStage::Close`]), invoked by [`LifecycleHooks`] in registration order.
//! Every invocation receives a cheap owned [`SubagentLifecycleEvent`] snapshot that the runtime
//! builds; the runtime calls these hooks outside its state lock, so a hook may call back into the
//! runtime (or register another hook) without deadlocking, and a failing hook never fails the
//! runtime — it is logged and reported in [`LifecycleReport::failures`].
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use futures::future::BoxFuture;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use openjiuwen_common::error::Result;

use super::quota::SubagentActivity;
use super::transcript::TranscriptSummary;
use super::SubagentStatus;

/// The transition a lifecycle hook observes.
///
/// Python's `subagent_lifecycle.py` splits resource handling into preparation and cleanup, while
/// `session_manager.py` reports each status change; the four stages below cover both.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum LifecycleStage {
    /// The instance was registered and its resources must be prepared.
    #[default]
    Spawn,
    /// The instance's child agent is live and idle after its first turn.
    Ready,
    /// The instance is about to be evicted to free a slot.
    BeforeEvict,
    /// The instance is closing (manually, with its parent, or by eviction).
    Close,
}

impl LifecycleStage {
    /// Returns the Python wire string of this stage (`spawn`, `ready`, `before_evict`, `close`).
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Spawn => "spawn",
            Self::Ready => "ready",
            Self::BeforeEvict => "before_evict",
            Self::Close => "close",
        }
    }
}

/// The owned snapshot every lifecycle hook receives.
///
/// The runtime builds it outside its state lock from the instance's identity, its derived status
/// and activity, the optional reason of the transition, and the transcript digest, so a hook never
/// observes a half-updated instance and never needs the runtime lock.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SubagentLifecycleEvent {
    /// Instance id; it doubles as the child session id (`sub_session_id`).
    pub subagent_id: String,
    /// Child session id of the instance.
    pub sub_session_id: String,
    /// Registered sub-agent type (`agent_card.name`).
    pub subagent_type: String,
    /// Nesting depth of the instance (top-level = 1).
    pub depth: usize,
    /// The instance that spawned this one, when nested; `None` for a top-level instance.
    pub parent_subagent_id: Option<String>,
    /// The transition this event reports.
    pub stage: LifecycleStage,
    /// Lifecycle status of the instance at the time of the transition.
    pub status: SubagentStatus,
    /// Derived activity phase of the instance at the time of the transition.
    pub activity: SubagentActivity,
    /// Close/evict reason, or the turn error that ended the instance.
    pub reason: Option<String>,
    /// Milliseconds since the Unix epoch, taken from the runtime's injected clock.
    pub at_ms: i64,
    /// Digest of the instance's transcript at the time of the transition.
    pub transcript: TranscriptSummary,
}

/// Owner-implemented lifecycle hook for subagent instances.
///
/// Contract:
/// - hooks should return quickly and must not block the runtime for long; a hook that awaits slow
///   IO delays the transition it observes (the runtime does not hold its state lock meanwhile);
/// - hooks must not panic; a panic unwinds the caller's task (the runtime does not catch it);
/// - hooks must not assume the runtime lock is held — it is not, so a hook may call back into the
///   runtime (list instances, read transcripts, register another hook) safely;
/// - errors are contained by [`LifecycleHooks::run`]: they are logged, reported, and the remaining
///   hooks still run.
///
/// Every method is defaulted to a no-op, so an implementation only overrides the stages it cares
/// about. A host bridges an arbitrary release path (for example a KV-cache eviction) with
/// [`CallbackLifecycleHook`].
#[async_trait::async_trait]
pub trait SubagentLifecycleHook: Send + Sync {
    /// Unique hook name; registering a second hook with the same name replaces the first in place.
    fn name(&self) -> &str;

    /// Called after the instance is registered, before its first turn starts.
    async fn on_spawn(&self, _event: &SubagentLifecycleEvent) -> Result<()> {
        Ok(())
    }

    /// Called once the instance's child agent is live and idle.
    async fn on_ready(&self, _event: &SubagentLifecycleEvent) -> Result<()> {
        Ok(())
    }

    /// Called before the instance is evicted to free a capacity slot.
    async fn on_before_evict(&self, _event: &SubagentLifecycleEvent) -> Result<()> {
        Ok(())
    }

    /// Called when the instance closes (manual close, parent shutdown, or eviction).
    async fn on_close(&self, _event: &SubagentLifecycleEvent) -> Result<()> {
        Ok(())
    }
}

/// One contained hook failure.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct LifecycleFailure {
    /// Name of the hook that returned the error.
    pub hook: String,
    /// Stage the failing hook was invoked for.
    pub stage: LifecycleStage,
    /// The error message the hook returned.
    pub message: String,
}

/// Outcome of one [`LifecycleHooks::run`] pass.
///
/// A pass always runs every registered hook; failures are collected instead of aborting, so a
/// caller can log or surface them without losing the remaining hooks.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct LifecycleReport {
    /// Number of hooks the pass attempted.
    pub invoked: usize,
    /// Hooks that returned an error, in invocation order.
    pub failures: Vec<LifecycleFailure>,
}

impl LifecycleReport {
    /// Whether every attempted hook succeeded.
    #[must_use]
    pub fn is_ok(&self) -> bool {
        self.failures.is_empty()
    }
}

/// An ordered registry of lifecycle hooks, cheap to clone.
///
/// Clones share one hook list, so registering a hook through any clone makes it visible to every
/// clone; the runtime and a host therefore hold the same registry.
#[derive(Clone, Default)]
pub struct LifecycleHooks {
    /// Hooks in registration order, shared across clones.
    hooks: Arc<RwLock<Vec<Arc<dyn SubagentLifecycleHook>>>>,
}

impl LifecycleHooks {
    /// Creates an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers `hook`; a hook whose [`SubagentLifecycleHook::name`] already exists replaces the
    /// existing one in place, keeping its registration position.
    pub fn register(&self, hook: Arc<dyn SubagentLifecycleHook>) {
        let mut hooks = self.hooks.write();
        match hooks
            .iter_mut()
            .find(|existing| existing.name() == hook.name())
        {
            Some(slot) => *slot = hook,
            None => hooks.push(hook),
        }
    }

    /// Builder form of [`Self::register`].
    #[must_use]
    pub fn with_hook(self, hook: Arc<dyn SubagentLifecycleHook>) -> Self {
        self.register(hook);
        self
    }

    /// Returns a snapshot of the registered hooks in registration order.
    ///
    /// The runtime uses it to copy a caller-supplied registry into its own; the clone shares the
    /// hook objects, not the registry, so later changes to the source registry do not leak in.
    #[must_use]
    pub fn hooks(&self) -> Vec<Arc<dyn SubagentLifecycleHook>> {
        self.hooks.read().clone()
    }

    /// Returns the registered hook names in registration order.
    #[must_use]
    pub fn names(&self) -> Vec<String> {
        self.hooks
            .read()
            .iter()
            .map(|hook| hook.name().to_owned())
            .collect()
    }

    /// Returns the number of registered hooks.
    #[must_use]
    pub fn len(&self) -> usize {
        self.hooks.read().len()
    }

    /// Whether no hook is registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.hooks.read().is_empty()
    }

    /// Removes every registered hook.
    pub fn clear(&self) {
        self.hooks.write().clear();
    }

    /// Runs `stage` on every registered hook, in registration order.
    ///
    /// The hook list is snapshotted under the read lock and the lock is dropped before any hook is
    /// awaited, so a hook may call [`Self::names`], [`Self::register`] or [`Self::run`] without
    /// deadlocking.
    /// A hook that returns an error is logged with `tracing` and recorded in
    /// [`LifecycleReport::failures`]; the pass continues with the remaining hooks and never fails
    /// the caller.
    #[must_use]
    pub async fn run(
        &self,
        stage: LifecycleStage,
        event: &SubagentLifecycleEvent,
    ) -> LifecycleReport {
        let snapshot: Vec<Arc<dyn SubagentLifecycleHook>> = {
            let hooks = self.hooks.read();
            let snapshot = hooks.clone();
            drop(hooks);
            snapshot
        };
        let mut report = LifecycleReport::default();
        for hook in snapshot {
            report.invoked += 1;
            let outcome = match stage {
                LifecycleStage::Spawn => hook.on_spawn(event).await,
                LifecycleStage::Ready => hook.on_ready(event).await,
                LifecycleStage::BeforeEvict => hook.on_before_evict(event).await,
                LifecycleStage::Close => hook.on_close(event).await,
            };
            if let Err(error) = outcome {
                tracing::warn!(
                    hook = hook.name(),
                    stage = stage.as_str(),
                    subagent_id = %event.subagent_id,
                    error = %error,
                    "subagent lifecycle hook failed"
                );
                report.failures.push(LifecycleFailure {
                    hook: hook.name().to_owned(),
                    stage,
                    message: error.to_string(),
                });
            }
        }
        report
    }
}

impl fmt::Debug for LifecycleHooks {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LifecycleHooks")
            .field("names", &self.names())
            .finish()
    }
}

/// A boxed async lifecycle callback: it receives an owned event snapshot.
pub type LifecycleCallback = Arc<
    dyn Fn(SubagentLifecycleEvent) -> BoxFuture<'static, openjiuwen_common::error::Result<()>>
        + Send
        + Sync,
>;

/// A generic hook that dispatches one caller-supplied callback per stage.
///
/// A host uses it to bridge an arbitrary release path (for example
/// `KvCacheSubagentLifecycle::evict_subagent` from [`crate::rails`]) without implementing a new
/// hook type; a stage without a callback is a no-op.
pub struct CallbackLifecycleHook {
    /// Hook name reported to [`LifecycleHooks`].
    name: String,
    /// Callback per stage; absent stages are no-ops.
    callbacks: BTreeMap<LifecycleStage, LifecycleCallback>,
}

impl CallbackLifecycleHook {
    /// Creates a hook named `name` with no callbacks registered.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            callbacks: BTreeMap::new(),
        }
    }

    /// Registers `callback` for `stage`, replacing any callback previously set for that stage.
    #[must_use]
    pub fn on(mut self, stage: LifecycleStage, callback: LifecycleCallback) -> Self {
        self.callbacks.insert(stage, callback);
        self
    }

    /// Awaits the callback registered for `stage`, passing an owned clone of `event`.
    async fn invoke(&self, stage: LifecycleStage, event: &SubagentLifecycleEvent) -> Result<()> {
        match self.callbacks.get(&stage) {
            Some(callback) => callback(event.clone()).await,
            None => Ok(()),
        }
    }
}

impl fmt::Debug for CallbackLifecycleHook {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CallbackLifecycleHook")
            .field("name", &self.name)
            .field("stages", &self.callbacks.keys().collect::<Vec<_>>())
            .finish()
    }
}

#[async_trait::async_trait]
impl SubagentLifecycleHook for CallbackLifecycleHook {
    fn name(&self) -> &str {
        &self.name
    }

    async fn on_spawn(&self, event: &SubagentLifecycleEvent) -> Result<()> {
        self.invoke(LifecycleStage::Spawn, event).await
    }

    async fn on_ready(&self, event: &SubagentLifecycleEvent) -> Result<()> {
        self.invoke(LifecycleStage::Ready, event).await
    }

    async fn on_before_evict(&self, event: &SubagentLifecycleEvent) -> Result<()> {
        self.invoke(LifecycleStage::BeforeEvict, event).await
    }

    async fn on_close(&self, event: &SubagentLifecycleEvent) -> Result<()> {
        self.invoke(LifecycleStage::Close, event).await
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::time::Duration;

    use parking_lot::Mutex;

    use openjiuwen_common::error::{Error, Result};

    use super::super::transcript::TranscriptKind;
    use super::*;

    /// One recorded hook invocation: the stage the registry dispatched and the event payload.
    type Call = (LifecycleStage, SubagentLifecycleEvent);

    /// Shared invocation log of one recording hook.
    type Seen = Arc<Mutex<Vec<Call>>>;

    /// Shared cross-hook invocation log (`<hook>:<stage>`), used to prove registration order.
    type Order = Arc<Mutex<Vec<String>>>;

    /// One recorded callback call: stage, subagent id, and reason.
    type CallRecord = (LifecycleStage, String, Option<String>);

    /// A pass over `invoked` hooks where every hook succeeded.
    fn clean(invoked: usize) -> LifecycleReport {
        LifecycleReport {
            invoked,
            failures: Vec::new(),
        }
    }

    /// The dispatched stages of `seen`, in invocation order.
    fn stages(seen: &Seen) -> Vec<LifecycleStage> {
        seen.lock().iter().map(|(stage, _)| *stage).collect()
    }

    /// Hook that records the dispatched stage plus the event payload and can fail on one stage.
    struct RecordingHook {
        name: &'static str,
        seen: Seen,
        order: Option<Order>,
        fail_on: Option<LifecycleStage>,
    }

    impl RecordingHook {
        fn new(name: &'static str) -> (Self, Seen) {
            Self::build(name, None)
        }

        fn failing(name: &'static str, stage: LifecycleStage) -> (Self, Seen) {
            Self::build(name, Some(stage))
        }

        fn build(name: &'static str, fail_on: Option<LifecycleStage>) -> (Self, Seen) {
            let seen: Seen = Arc::new(Mutex::new(Vec::new()));
            (
                Self {
                    name,
                    seen: Arc::clone(&seen),
                    order: None,
                    fail_on,
                },
                seen,
            )
        }

        /// Also appends `<name>:<stage>` to the shared cross-hook `order` log.
        fn with_order(mut self, order: Order) -> Self {
            self.order = Some(order);
            self
        }

        fn record(&self, stage: LifecycleStage, event: &SubagentLifecycleEvent) -> Result<()> {
            self.seen.lock().push((stage, event.clone()));
            if let Some(order) = &self.order {
                order
                    .lock()
                    .push(format!("{}:{}", self.name, stage.as_str()));
            }
            if self.fail_on == Some(stage) {
                return Err(Error::Validation(format!(
                    "{} refused {stage:?}",
                    self.name
                )));
            }
            Ok(())
        }
    }

    #[async_trait::async_trait]
    impl SubagentLifecycleHook for RecordingHook {
        fn name(&self) -> &'static str {
            self.name
        }

        async fn on_spawn(&self, event: &SubagentLifecycleEvent) -> Result<()> {
            self.record(LifecycleStage::Spawn, event)
        }

        async fn on_ready(&self, event: &SubagentLifecycleEvent) -> Result<()> {
            self.record(LifecycleStage::Ready, event)
        }

        async fn on_before_evict(&self, event: &SubagentLifecycleEvent) -> Result<()> {
            self.record(LifecycleStage::BeforeEvict, event)
        }

        async fn on_close(&self, event: &SubagentLifecycleEvent) -> Result<()> {
            self.record(LifecycleStage::Close, event)
        }
    }

    /// Hook with no overrides: every trait method is the defaulted no-op.
    struct BareHook;

    #[async_trait::async_trait]
    impl SubagentLifecycleHook for BareHook {
        fn name(&self) -> &'static str {
            "bare"
        }
    }

    /// Hook that calls back into the registry from inside its body.
    struct ReentrantHook {
        inner: LifecycleHooks,
        seen: Arc<Mutex<Vec<LifecycleStage>>>,
        /// Report of the nested `Ready` pass, so the test can inspect its contents.
        nested: Arc<Mutex<Option<LifecycleReport>>>,
    }

    #[async_trait::async_trait]
    impl SubagentLifecycleHook for ReentrantHook {
        fn name(&self) -> &'static str {
            "reentrant"
        }

        async fn on_spawn(&self, event: &SubagentLifecycleEvent) -> Result<()> {
            self.seen.lock().push(LifecycleStage::Spawn);
            // Both calls re-enter the registry while this pass is awaiting the hook.
            let names = self.inner.names();
            assert_eq!(names.len(), self.inner.len());
            let nested = self.inner.run(LifecycleStage::Ready, event).await;
            // The nested pass invoked every registered hook for `Ready`, not just this one.
            assert_eq!(nested.invoked, names.len());
            *self.nested.lock() = Some(nested);
            Ok(())
        }
    }

    /// The canonical event payload the tests dispatch.
    fn event(stage: LifecycleStage) -> SubagentLifecycleEvent {
        SubagentLifecycleEvent {
            subagent_id: "parent_sub_code_agent_0011aabb".to_owned(),
            sub_session_id: "parent_sub_code_agent_0011aabb".to_owned(),
            subagent_type: "code_agent".to_owned(),
            depth: 2,
            parent_subagent_id: Some("parent".to_owned()),
            stage,
            status: SubagentStatus::Completed,
            activity: SubagentActivity::Idle,
            reason: Some("manual".to_owned()),
            at_ms: 1_700_000_000_000,
            transcript: TranscriptSummary {
                count: 3,
                dropped: 1,
                last_kind: Some(TranscriptKind::Completion),
                last_content: Some("done".to_owned()),
            },
        }
    }

    #[tokio::test]
    async fn hooks_run_in_registration_order() {
        let hooks = LifecycleHooks::new();
        let order: Order = Arc::new(Mutex::new(Vec::new()));
        let (first, first_seen) = RecordingHook::new("first");
        let (second, second_seen) = RecordingHook::new("second");
        hooks.register(Arc::new(first.with_order(Arc::clone(&order))));
        hooks.register(Arc::new(second.with_order(Arc::clone(&order))));

        let spawn_event = event(LifecycleStage::Spawn);
        let report = hooks.run(LifecycleStage::Spawn, &spawn_event).await;

        assert_eq!(report, clean(2));
        // The shared log proves the hooks were awaited in registration order, which the
        // per-hook logs alone cannot show.
        assert_eq!(
            *order.lock(),
            vec!["first:spawn".to_owned(), "second:spawn".to_owned()]
        );
        // Each hook received the dispatched stage and the very payload under test.
        assert_eq!(
            *first_seen.lock(),
            vec![(LifecycleStage::Spawn, spawn_event.clone())]
        );
        assert_eq!(
            *second_seen.lock(),
            vec![(LifecycleStage::Spawn, spawn_event.clone())]
        );
        assert_eq!(hooks.names(), vec!["first".to_owned(), "second".to_owned()]);
    }

    #[tokio::test]
    async fn a_failing_hook_is_reported_and_later_hooks_still_run() {
        let hooks = LifecycleHooks::new();
        let order: Order = Arc::new(Mutex::new(Vec::new()));
        let (first, first_seen) = RecordingHook::new("first");
        hooks.register(Arc::new(first.with_order(Arc::clone(&order))));
        let (failing, failing_seen) = RecordingHook::failing("broken", LifecycleStage::BeforeEvict);
        hooks.register(Arc::new(failing.with_order(Arc::clone(&order))));
        let (last, last_seen) = RecordingHook::new("last");
        hooks.register(Arc::new(last.with_order(Arc::clone(&order))));

        let evict_event = event(LifecycleStage::BeforeEvict);
        let report = hooks.run(LifecycleStage::BeforeEvict, &evict_event).await;

        assert!(!report.is_ok());
        assert_eq!(report.invoked, 3);
        // The failure names the hook, the stage and the exact error the hook returned.
        assert_eq!(
            report.failures,
            vec![LifecycleFailure {
                hook: "broken".to_owned(),
                stage: LifecycleStage::BeforeEvict,
                message: "validation: broken refused BeforeEvict".to_owned(),
            }]
        );
        // All three hooks were awaited, in registration order, even though the middle one failed.
        assert_eq!(
            *order.lock(),
            vec![
                "first:before_evict".to_owned(),
                "broken:before_evict".to_owned(),
                "last:before_evict".to_owned(),
            ]
        );
        // The failing hook did observe the event before returning its error, and the hooks before
        // and after it still received the same payload.
        assert_eq!(
            *failing_seen.lock(),
            vec![(LifecycleStage::BeforeEvict, evict_event.clone())]
        );
        assert_eq!(
            *first_seen.lock(),
            vec![(LifecycleStage::BeforeEvict, evict_event.clone())]
        );
        assert_eq!(
            *last_seen.lock(),
            vec![(LifecycleStage::BeforeEvict, evict_event.clone())]
        );
    }

    #[tokio::test]
    async fn registering_by_name_replaces_in_place_and_keeps_order() {
        let hooks = LifecycleHooks::new();
        let (first, first_seen) = RecordingHook::new("first");
        let (second, second_seen) = RecordingHook::new("second");
        hooks.register(Arc::new(first));
        hooks.register(Arc::new(second));
        let (replacement, replacement_seen) = RecordingHook::new("first");
        hooks.register(Arc::new(replacement));

        assert_eq!(hooks.len(), 2);
        assert_eq!(hooks.names(), vec!["first".to_owned(), "second".to_owned()]);

        let close_event = event(LifecycleStage::Close);
        let report = hooks.run(LifecycleStage::Close, &close_event).await;
        assert_eq!(report, clean(2));
        // The replacement ran in the evicted slot and the other hook kept its position.
        assert_eq!(
            *replacement_seen.lock(),
            vec![(LifecycleStage::Close, close_event.clone())]
        );
        assert_eq!(
            *second_seen.lock(),
            vec![(LifecycleStage::Close, close_event.clone())]
        );
        assert!(
            first_seen.lock().is_empty(),
            "the replaced hook must not run again"
        );
    }

    #[tokio::test]
    async fn with_hook_chains_and_exposes_names() {
        let (first, first_seen) = RecordingHook::new("first");
        let (second, second_seen) = RecordingHook::new("second");
        let hooks = LifecycleHooks::new()
            .with_hook(Arc::new(first))
            .with_hook(Arc::new(second));

        assert_eq!(hooks.len(), 2);
        assert!(!hooks.is_empty());
        assert_eq!(hooks.names(), vec!["first".to_owned(), "second".to_owned()]);

        let close_event = event(LifecycleStage::Close);
        assert_eq!(
            hooks.run(LifecycleStage::Close, &close_event).await,
            clean(2)
        );
        assert_eq!(stages(&first_seen), vec![LifecycleStage::Close]);
        assert_eq!(stages(&second_seen), vec![LifecycleStage::Close]);

        hooks.clear();
        assert!(hooks.is_empty());
        assert_eq!(hooks.len(), 0);
        assert_eq!(hooks.names(), Vec::<String>::new());
        // The cleared registry awaits nothing, so neither hook is invoked again.
        assert_eq!(
            hooks.run(LifecycleStage::Close, &close_event).await,
            clean(0)
        );
        assert_eq!(stages(&first_seen), vec![LifecycleStage::Close]);
        assert_eq!(stages(&second_seen), vec![LifecycleStage::Close]);
    }

    #[tokio::test]
    async fn defaulted_methods_are_no_ops_for_every_stage() {
        let hooks = LifecycleHooks::new();
        hooks.register(Arc::new(BareHook));
        let (recorder, recorder_seen) = RecordingHook::new("recorder");
        hooks.register(Arc::new(recorder));

        for stage in [
            LifecycleStage::Spawn,
            LifecycleStage::Ready,
            LifecycleStage::BeforeEvict,
            LifecycleStage::Close,
        ] {
            let payload = event(stage);
            let report = hooks.run(stage, &payload).await;
            // The bare hook contributes no failure; the recorder proves the pass really
            // dispatched this stage with this payload.
            assert_eq!(report, clean(2), "stage {}", stage.as_str());
            assert_eq!(
                *recorder_seen.lock(),
                vec![(stage, payload)],
                "stage {} reached the recorder",
                stage.as_str()
            );
            recorder_seen.lock().clear();
        }
    }

    #[tokio::test]
    async fn callback_hook_invokes_only_the_registered_stage_and_forwards_the_event() {
        let received: Arc<Mutex<Vec<CallRecord>>> = Arc::new(Mutex::new(Vec::new()));

        let sink = Arc::clone(&received);
        let record = move |event: SubagentLifecycleEvent| -> BoxFuture<'static, Result<()>> {
            let sink = Arc::clone(&sink);
            Box::pin(async move {
                sink.lock()
                    .push((event.stage, event.subagent_id, event.reason));
                Ok(())
            })
        };
        let refuse = |_: SubagentLifecycleEvent| -> BoxFuture<'static, Result<()>> {
            Box::pin(async { Err(Error::Validation("release refused".to_owned())) })
        };
        let hook = CallbackLifecycleHook::new("kv-cache")
            .on(LifecycleStage::BeforeEvict, Arc::new(record))
            .on(LifecycleStage::Ready, Arc::new(refuse));

        let hooks = LifecycleHooks::new().with_hook(Arc::new(hook));

        // A stage without a callback is a no-op, not a failure.
        let spawn_report = hooks
            .run(LifecycleStage::Spawn, &event(LifecycleStage::Spawn))
            .await;
        assert_eq!(spawn_report, clean(1));
        assert!(
            received.lock().is_empty(),
            "the spawn stage has no callback"
        );

        // A callback that fails is reported against the hook name and the dispatched stage, and
        // it never reaches the `BeforeEvict` recorder.
        let ready_report = hooks
            .run(LifecycleStage::Ready, &event(LifecycleStage::Ready))
            .await;
        assert_eq!(
            ready_report,
            LifecycleReport {
                invoked: 1,
                failures: vec![LifecycleFailure {
                    hook: "kv-cache".to_owned(),
                    stage: LifecycleStage::Ready,
                    message: "validation: release refused".to_owned(),
                }],
            }
        );
        assert!(
            received.lock().is_empty(),
            "the ready callback is the failing one"
        );

        let evict_event = event(LifecycleStage::BeforeEvict);
        let evict_report = hooks.run(LifecycleStage::BeforeEvict, &evict_event).await;
        assert_eq!(evict_report, clean(1));

        let calls = received.lock().clone();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, LifecycleStage::BeforeEvict);
        assert_eq!(calls[0].1, evict_event.subagent_id);
        assert_eq!(calls[0].2.as_deref(), Some("manual"));
    }

    #[tokio::test]
    async fn a_hook_may_reenter_the_registry_without_deadlock() {
        let hooks = LifecycleHooks::new();
        let (watcher, watcher_seen) = RecordingHook::new("watcher");
        hooks.register(Arc::new(watcher));
        let reentrant_seen: Arc<Mutex<Vec<LifecycleStage>>> = Arc::new(Mutex::new(Vec::new()));
        let nested_report: Arc<Mutex<Option<LifecycleReport>>> = Arc::new(Mutex::new(None));
        hooks.register(Arc::new(ReentrantHook {
            inner: hooks.clone(),
            seen: Arc::clone(&reentrant_seen),
            nested: Arc::clone(&nested_report),
        }));

        let spawn_event = event(LifecycleStage::Spawn);
        let report = tokio::time::timeout(
            Duration::from_secs(5),
            hooks.run(LifecycleStage::Spawn, &spawn_event),
        )
        .await
        .expect("the reentrant hook must not deadlock");

        // The outer pass ran both hooks; the nested `Ready` pass did the same.
        assert_eq!(report, clean(2));
        assert_eq!(*reentrant_seen.lock(), vec![LifecycleStage::Spawn]);
        let nested = nested_report
            .lock()
            .clone()
            .expect("the nested pass finished");
        assert_eq!(nested, clean(2));
        // The nested pass reached the recording watcher at `Ready` and forwarded the same event.
        assert_eq!(
            stages(&watcher_seen),
            vec![LifecycleStage::Spawn, LifecycleStage::Ready]
        );
        assert!(
            watcher_seen
                .lock()
                .iter()
                .all(|(_, payload)| *payload == spawn_event),
            "both passes forwarded the same payload"
        );
    }

    #[test]
    fn event_round_trips_and_tolerates_defaults() {
        let original = event(LifecycleStage::BeforeEvict);
        let encoded = serde_json::to_value(&original).expect("the event encodes");
        assert_eq!(encoded["stage"], serde_json::json!("before_evict"));
        assert_eq!(
            encoded["sub_session_id"],
            serde_json::json!(original.sub_session_id)
        );
        assert_eq!(encoded["parent_subagent_id"], serde_json::json!("parent"));
        assert_eq!(
            encoded["transcript"]["last_kind"],
            serde_json::json!("completion")
        );

        let decoded: SubagentLifecycleEvent =
            serde_json::from_value(encoded).expect("the event decodes");
        assert_eq!(decoded, original);

        let defaults: SubagentLifecycleEvent =
            serde_json::from_str("{}").expect("an empty object decodes");
        assert_eq!(defaults, SubagentLifecycleEvent::default());
        assert_eq!(defaults.stage, LifecycleStage::Spawn);
        assert_eq!(defaults.subagent_id, "");
        assert!(defaults.parent_subagent_id.is_none());
    }

    #[test]
    fn stage_wire_strings_match_python() {
        assert_eq!(LifecycleStage::Spawn.as_str(), "spawn");
        assert_eq!(LifecycleStage::Ready.as_str(), "ready");
        assert_eq!(LifecycleStage::BeforeEvict.as_str(), "before_evict");
        assert_eq!(LifecycleStage::Close.as_str(), "close");
        let decoded: LifecycleStage =
            serde_json::from_str("\"before_evict\"").expect("the stage decodes");
        assert_eq!(decoded, LifecycleStage::BeforeEvict);
    }
}
