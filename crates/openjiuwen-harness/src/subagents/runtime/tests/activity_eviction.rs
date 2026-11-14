//! One scripted-model end-to-end pass over the activity, lifecycle and eviction surfaces.
//!
//! The sibling suites prove each surface over the local fakes of [`super`]; this one joins them
//! by building a **real** child agent: [`ScriptedFactory`] answers every
//! [`SubAgentFactory::create`] with a [`DeepAgent`](crate::agent::DeepAgent) whose model is a
//! [`ScriptedModel`] over a per-session temporary workspace under one `TempDir`.
//!
//! Two tests, both offline and deterministic (a frozen [`ManualClock`], bounded
//! `tokio::time::timeout` around every await that can hang, no sleeps):
//!
//! - the recording-hook path: `spawn` -> `Running`/`Idle`/`Completed` activity frames and the
//!   live feed -> `Spawn`/`Ready` lifecycle hooks -> an explicit `evict` after the clock is
//!   advanced -> terminal `Evicted` frame, `evicted` transcript entry and
//!   `BeforeEvict`/`Close` hooks, after which the terminal stream suppresses late progress;
//! - the closure path: a [`CallbackLifecycleHook`] registered for `BeforeEvict`/`Close` observes
//!   the eviction and calls back into the runtime from inside the hook without deadlocking.
#![allow(clippy::unwrap_used, clippy::expect_used)]
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use futures::future::BoxFuture;
use openjiuwen_agent::AgentCard;
use openjiuwen_test_support::ScriptedModel;
use parking_lot::Mutex;
use tempfile::TempDir;

use super::*;
use crate::agent::DeepAgentBuilder;
use crate::schema::{DeepAgentConfig, SubAgentConfig};
use crate::subagents::{
    ActivityEvent, ActivityKind, CallbackLifecycleHook, LifecycleCallback, LifecycleHooks,
    LifecycleStage, ManualClock, SubAgentSpec, SubagentLifecycleEvent, SubagentLifecycleHook,
    TranscriptKind,
};
use crate::workspace::Workspace;

/// The scripted answer of every child turn.
const ANSWER: &str = "scripted worker report";

/// Upper bound for any await that could hang; a green run finishes in well under it.
const BOUND: Duration = Duration::from_secs(10);

/// Clock origin, so every recorded timestamp is reproducible.
const CLOCK_START_MS: i64 = 1_700_000_000_000;

/// Session id the runtime hands every instance (it doubles as the sub-agent id).
const PARENT_SESSION: &str = "parent-session";

/// Builds a real [`DeepAgent`](crate::agent::DeepAgent) per child session and records the ids.
struct ScriptedFactory {
    /// Root of the per-session child workspaces.
    root: PathBuf,
    /// Child session ids passed to [`SubAgentFactory::create`], in call order.
    built: Mutex<Vec<String>>,
}

impl ScriptedFactory {
    /// Creates a factory rooted at `root`, which it also creates.
    fn new(root: PathBuf) -> Arc<Self> {
        std::fs::create_dir_all(&root).expect("child workspace root");
        Arc::new(Self {
            root,
            built: Mutex::new(Vec::new()),
        })
    }

    /// The session ids built so far, in call order.
    fn built(&self) -> Vec<String> {
        self.built.lock().clone()
    }
}

#[async_trait::async_trait]
impl SubAgentFactory for ScriptedFactory {
    async fn create(&self, config: &SubAgentConfig, session_id: &str) -> Result<Arc<dyn Agent>> {
        self.built.lock().push(session_id.to_owned());
        // The runtime does not re-root the workspace: Python's `create_subagent` gives every
        // child its own root, so the factory does the same here.
        let root = self.root.join(session_id);
        std::fs::create_dir_all(&root).map_err(|error| {
            Error::session(format!(
                "failed to create child workspace {}: {error}",
                root.display()
            ))
        })?;
        let workspace = Workspace::for_language(root, "en");
        let agent = DeepAgentBuilder::new(config.agent_card.clone())
            .config(
                DeepAgentConfig::default()
                    .with_language("en".to_owned())
                    .with_workspace(workspace),
            )
            .model(Arc::new(ScriptedModel::text(ANSWER)))
            .build()?;
        Ok(agent)
    }
}

/// One recorded lifecycle invocation.
#[derive(Debug, Clone, PartialEq)]
struct Step {
    /// Stage the hook was invoked for.
    stage: LifecycleStage,
    /// Status snapshot carried by the event.
    status: SubagentStatus,
    /// Close/evict reason carried by the event.
    reason: Option<String>,
}

/// Log of a recording hook, shared with the test.
type Seen = Arc<Mutex<Vec<Step>>>;

/// Hook that records every stage it observes; the log guard is dropped before returning, so no
/// lock is ever held across an `.await`.
struct RecordingHook {
    /// Shared invocation log.
    seen: Seen,
}

impl RecordingHook {
    /// Creates the hook and returns it with its log.
    fn new() -> (Self, Seen) {
        let seen: Seen = Arc::new(Mutex::new(Vec::new()));
        let hook = Self {
            seen: Arc::clone(&seen),
        };
        (hook, seen)
    }

    /// Appends one invocation, keeping the guard inside the call.
    fn record(&self, event: &SubagentLifecycleEvent) {
        let mut seen = self.seen.lock();
        seen.push(Step {
            stage: event.stage,
            status: event.status,
            reason: event.reason.clone(),
        });
    }
}

#[async_trait::async_trait]
impl SubagentLifecycleHook for RecordingHook {
    fn name(&self) -> &'static str {
        "recording"
    }

    async fn on_spawn(&self, event: &SubagentLifecycleEvent) -> Result<()> {
        self.record(event);
        Ok(())
    }

    async fn on_ready(&self, event: &SubagentLifecycleEvent) -> Result<()> {
        self.record(event);
        Ok(())
    }

    async fn on_before_evict(&self, event: &SubagentLifecycleEvent) -> Result<()> {
        self.record(event);
        Ok(())
    }

    async fn on_close(&self, event: &SubagentLifecycleEvent) -> Result<()> {
        self.record(event);
        Ok(())
    }
}

/// Builds the runtime over the scripted factory: one custom spec, a manual clock and `hooks`.
fn scripted_runtime(
    temp: &TempDir,
    clock: &Arc<ManualClock>,
    hooks: LifecycleHooks,
) -> (Arc<SubAgentRuntime>, Arc<ScriptedFactory>) {
    let factory = ScriptedFactory::new(temp.path().join("children"));
    let registry = SubAgentRegistry::new();
    registry.register(SubAgentSpec::new(SubAgentConfig::new(
        AgentCard::new("worker_agent").with_description("scripted worker"),
    )));
    let runtime = SubAgentRuntime::new(
        Arc::new(registry),
        Arc::clone(&factory) as Arc<dyn SubAgentFactory>,
        PARENT_SESSION,
    )
    .with_clock(Arc::clone(clock) as Arc<dyn crate::subagents::Clock>)
    .with_lifecycle_hooks(hooks);
    (Arc::new(runtime), factory)
}

/// The `kind` of every retained activity frame of `id`, oldest first.
fn frame_kinds(runtime: &SubAgentRuntime, id: &str) -> Vec<ActivityKind> {
    runtime
        .activity(id)
        .expect("a known instance")
        .into_iter()
        .map(|frame| frame.kind)
        .collect()
}

/// The stages of a recorded log, in invocation order.
fn stages(seen: &Seen) -> Vec<LifecycleStage> {
    seen.lock().iter().map(|step| step.stage).collect()
}

/// Drains exactly `count` frames from a live feed, failing on lag.
async fn drain_frames(
    subscription: &mut crate::subagents::ActivitySubscription,
    count: usize,
) -> Vec<ActivityKind> {
    let mut kinds = Vec::with_capacity(count);
    for _ in 0..count {
        let event = tokio::time::timeout(BOUND, subscription.recv())
            .await
            .expect("a live frame arrives")
            .expect("the feed stays open");
        match event {
            ActivityEvent::Frame(frame) => kinds.push(frame.kind),
            ActivityEvent::Lagged { dropped } => {
                panic!("a small stream must not lag, lost {dropped} frames")
            }
        }
    }
    kinds
}

/// Closure-shaped release path: records the stage and reads the runtime from inside the hook.
fn release_callback(
    seen: &Seen,
    slot: &Arc<Mutex<Option<Arc<SubAgentRuntime>>>>,
) -> LifecycleCallback {
    let seen = Arc::clone(seen);
    let slot = Arc::clone(slot);
    Arc::new(
        move |event: SubagentLifecycleEvent| -> BoxFuture<'static, Result<()>> {
            let seen = Arc::clone(&seen);
            let slot = Arc::clone(&slot);
            Box::pin(async move {
                seen.lock().push(Step {
                    stage: event.stage,
                    status: event.status,
                    reason: event.reason.clone(),
                });
                // The runtime lock is not held while a hook runs, so this must not deadlock.
                let runtime = slot
                    .lock()
                    .clone()
                    .expect("the runtime handle is installed");
                assert!(runtime.activity_summary(&event.subagent_id).is_some());
                // `BeforeEvict` observes the pre-eviction status; `Close` the tombstone.
                let expected = if event.stage == LifecycleStage::Close {
                    SubagentStatus::Closed
                } else {
                    SubagentStatus::Completed
                };
                assert_eq!(
                    runtime.get_status(&event.subagent_id),
                    expected,
                    "the status the {} hook observes",
                    event.stage.as_str()
                );
                Ok(())
            })
        },
    )
}

/// Spawns `worker_agent` on `runtime` and returns its instance id.
async fn spawn_worker(runtime: &SubAgentRuntime) -> SubAgentId {
    tokio::time::timeout(
        BOUND,
        runtime.spawn(SubAgentSpawnRequest::new("worker_agent", "report status")),
    )
    .await
    .expect("the spawn returns")
    .expect("the spawn succeeds")
}

/// Waits for the single scripted turn of `id` and asserts it completed.
async fn completed_outcome(runtime: &SubAgentRuntime, id: &str) -> SubAgentOutcome {
    let outcome = tokio::time::timeout(
        BOUND,
        runtime.wait(std::slice::from_ref(&id.to_owned()), BOUND),
    )
    .await
    .expect("the wait returns")
    .pop()
    .expect("one outcome");
    assert_eq!(outcome.status, SubagentStatus::Completed);
    assert_eq!(runtime.get_status(id), SubagentStatus::Completed);
    outcome
}

/// Asserts the retained ring and digest of a completed scripted turn.
fn assert_turn_activity(runtime: &SubAgentRuntime, id: &str) {
    let frames = runtime.activity(id).expect("the instance exists");
    assert_eq!(
        frames.iter().map(|frame| frame.kind).collect::<Vec<_>>(),
        vec![
            ActivityKind::Spawn,
            ActivityKind::Running,
            ActivityKind::Idle,
            ActivityKind::Completed
        ]
    );
    assert_eq!(frames[0].summary, "report status");
    assert_eq!(
        frames[0].task_id, "",
        "the spawn frame precedes the first turn"
    );
    assert!(frames.windows(2).all(|pair| pair[0].seq < pair[1].seq));
    let last = frames.last().expect("a terminal frame");
    assert_eq!(last.kind, ActivityKind::Completed);
    assert_eq!(last.ok, Some(true));
    assert_eq!(last.summary, ANSWER);
    assert_eq!(last.at_ms, CLOCK_START_MS);
    let summary = runtime.activity_summary(id).expect("the instance exists");
    assert!(summary.terminal);
    assert_eq!(summary.last_kind, Some(ActivityKind::Completed));
    assert_eq!(summary.last_seq, last.seq);
}

/// Asserts the `Spawn` then `Ready` prefix of a lifecycle log, before any eviction.
fn assert_ready_prefix(seen: &Seen) {
    let steps = seen.lock().clone();
    assert_eq!(
        stages(seen),
        vec![LifecycleStage::Spawn, LifecycleStage::Ready]
    );
    assert_eq!(steps[0].status, SubagentStatus::PendingInit);
    assert!(steps[0].reason.is_none());
    assert_eq!(steps[1].status, SubagentStatus::Completed);
    assert!(steps[1].reason.is_none());
}

/// Advances `clock` past a TTL, evicts `id` and asserts the tombstone status.
async fn evict_after_ttl(runtime: &SubAgentRuntime, clock: &ManualClock, id: &str) {
    clock.advance(60_000);
    let previous = tokio::time::timeout(BOUND, runtime.evict(id))
        .await
        .expect("the eviction returns")
        .expect("the instance is evictable");
    assert_eq!(previous, SubagentStatus::Completed);
    assert_eq!(runtime.get_status(id), SubagentStatus::Closed);
}

/// Asserts the evicted transcript digest and the terminal activity frame of a tombstone.
fn assert_eviction_activity(runtime: &SubAgentRuntime, id: &str) {
    let digest = runtime
        .transcript_summary(id)
        .expect("the tombstone exists");
    assert_eq!(digest.last_kind, Some(TranscriptKind::Evicted));
    assert_eq!(digest.last_content.as_deref(), Some(EVICTION_REASON));

    let frames = runtime.activity(id).expect("the tombstone exists");
    let evicted = frames.last().expect("the evicted frame");
    assert_eq!(evicted.kind, ActivityKind::Evicted);
    assert_eq!(evicted.summary, EVICTION_REASON);
    assert_eq!(evicted.ok, None);
    assert!(evicted.is_terminal());
    assert_eq!(evicted.at_ms, CLOCK_START_MS + 60_000);
    assert!(runtime
        .activity_summary(id)
        .is_some_and(|digest| digest.terminal));
}

/// Asserts the `BeforeEvict` then `Close` tail of a lifecycle log.
fn assert_eviction_hooks(seen: &Seen) {
    let steps = seen.lock().clone();
    assert_eq!(
        steps.iter().map(|step| step.stage).collect::<Vec<_>>(),
        vec![
            LifecycleStage::Spawn,
            LifecycleStage::Ready,
            LifecycleStage::BeforeEvict,
            LifecycleStage::Close
        ]
    );
    assert!(steps[2].reason.is_none());
    assert_eq!(steps[2].status, SubagentStatus::Completed);
    assert_eq!(steps[3].reason.as_deref(), Some(EVICTION_REASON));
    assert_eq!(steps[3].status, SubagentStatus::Closed);
}

#[tokio::test]
async fn scripted_child_runs_spawn_ready_evict_end_to_end() {
    let temp = TempDir::new().expect("temp root");
    let clock = Arc::new(ManualClock::new(CLOCK_START_MS));
    let (hook, seen) = RecordingHook::new();
    let (runtime, factory) = scripted_runtime(
        &temp,
        &clock,
        LifecycleHooks::new().with_hook(Arc::new(hook)),
    );

    // 2. A spawn really builds a child agent for the generated session id.
    let id = spawn_worker(&runtime).await;
    assert_eq!(factory.built(), vec![id.clone()]);
    // Taken after `spawn` returned, so the feed starts at the first turn's `Running` frame.
    let mut subscription = runtime.subscribe_activity(&id).expect("a live instance");

    // 3. The one scripted turn completes with the scripted answer.
    let outcome = completed_outcome(&runtime, &id).await;
    assert_eq!(outcome.output.as_deref(), Some(ANSWER));
    assert_eq!(outcome.sub_session_id, id);

    // 4. Activity end-to-end: retained ring, ordering, digest and live feed.
    assert_turn_activity(&runtime, &id);
    assert_eq!(
        drain_frames(&mut subscription, 3).await,
        vec![
            ActivityKind::Running,
            ActivityKind::Idle,
            ActivityKind::Completed
        ]
    );

    // 5. Lifecycle end-to-end so far: exactly `Spawn` then `Ready`, nothing else.
    assert_ready_prefix(&seen);

    // 6. Eviction end-to-end after the clock advances past a TTL: the tombstone, the transcript,
    //    the terminal activity frame and the hooks must all agree.
    evict_after_ttl(&runtime, &clock, &id).await;
    assert_eq!(
        drain_frames(&mut subscription, 1).await,
        vec![ActivityKind::Evicted]
    );
    assert_eviction_activity(&runtime, &id);
    assert_eviction_hooks(&seen);

    // A terminal stream suppresses every later progress record.
    assert_eq!(
        runtime
            .record_activity(&id, ActivityKind::Thinking, "late", None, None)
            .expect("the tombstone exists"),
        None
    );
    assert_eq!(
        frame_kinds(&runtime, &id).last(),
        Some(&ActivityKind::Evicted)
    );
}

#[tokio::test]
async fn callback_hook_observes_eviction_and_calls_back_without_deadlock() {
    let temp = TempDir::new().expect("temp root");
    let clock = Arc::new(ManualClock::new(CLOCK_START_MS));
    let seen: Seen = Arc::new(Mutex::new(Vec::new()));
    // Installed once the runtime exists; the hooks need a handle to call back into it.
    let slot: Arc<Mutex<Option<Arc<SubAgentRuntime>>>> = Arc::new(Mutex::new(None));

    let callback = CallbackLifecycleHook::new("release")
        .on(LifecycleStage::BeforeEvict, release_callback(&seen, &slot))
        .on(LifecycleStage::Close, release_callback(&seen, &slot));
    let (runtime, factory) = scripted_runtime(
        &temp,
        &clock,
        LifecycleHooks::new().with_hook(Arc::new(callback)),
    );
    *slot.lock() = Some(Arc::clone(&runtime));

    let id = spawn_worker(&runtime).await;
    assert_eq!(factory.built(), vec![id.clone()]);
    let outcome = completed_outcome(&runtime, &id).await;
    assert_eq!(outcome.output.as_deref(), Some(ANSWER));

    clock.advance(60_000);
    let previous = tokio::time::timeout(BOUND, runtime.evict(&id))
        .await
        .expect("the closure path must not deadlock")
        .expect("the instance is evictable");
    assert_eq!(previous, SubagentStatus::Completed);
    assert_eq!(runtime.get_status(&id), SubagentStatus::Closed);
    assert_eq!(
        stages(&seen),
        vec![LifecycleStage::BeforeEvict, LifecycleStage::Close]
    );
    assert_eq!(
        runtime
            .transcript_summary(&id)
            .and_then(|digest| digest.last_kind),
        Some(TranscriptKind::Evicted)
    );
    assert_eq!(
        frame_kinds(&runtime, &id).last(),
        Some(&ActivityKind::Evicted)
    );
}
