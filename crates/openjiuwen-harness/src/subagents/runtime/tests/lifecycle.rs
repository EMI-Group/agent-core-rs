//! Lifecycle-hook behaviour of the in-process sub-agent runtime.
//!
//! The recording hooks below hold their log behind a `parking_lot::Mutex` and release it before
//! returning, so no lock is ever held across an `.await`. Each test drives the real runtime over
//! the fakes of [`super`]: an echoing factory reaches `Ready`, [`blocked`] keeps turns running for
//! `cancel_all`, and the capacity-pressure spawn exercises the synchronous `Close`-only path.
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use tokio::sync::Notify;

use super::*;
use crate::subagents::{ActivityKind, SubAgentQuota, SubagentLifecycleHook, TranscriptKind};

/// One recorded hook invocation.
#[derive(Debug, Clone, PartialEq)]
struct Call {
    /// Stage the hook was invoked for.
    stage: LifecycleStage,
    /// Instance the event described.
    subagent_id: String,
    /// Close/evict reason of the event.
    reason: Option<String>,
    /// Status snapshot of the event.
    status: SubagentStatus,
}

/// Shared log of the recording hooks.
type Seen = Arc<Mutex<Vec<Call>>>;

/// Hook that records every invocation into [`Seen`].
struct RecordingHook {
    /// Hook name reported to the registry.
    name: &'static str,
    /// Shared invocation log.
    seen: Seen,
}

impl RecordingHook {
    /// Creates a hook named `name` and returns it with its log.
    fn new(name: &'static str) -> (Self, Seen) {
        let seen: Seen = Arc::new(Mutex::new(Vec::new()));
        (
            Self {
                name,
                seen: Arc::clone(&seen),
            },
            seen,
        )
    }

    /// Appends one invocation to the log, dropping the guard before returning.
    fn record(&self, event: &SubagentLifecycleEvent) {
        self.seen.lock().push(Call {
            stage: event.stage,
            subagent_id: event.subagent_id.clone(),
            reason: event.reason.clone(),
            status: event.status,
        });
    }
}

#[async_trait::async_trait]
impl SubagentLifecycleHook for RecordingHook {
    fn name(&self) -> &'static str {
        self.name
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

/// Hook whose `BeforeEvict` stage fails; the runtime must contain the error.
struct FailingBeforeEvict;

#[async_trait::async_trait]
impl SubagentLifecycleHook for FailingBeforeEvict {
    fn name(&self) -> &'static str {
        "failing-before-evict"
    }

    async fn on_before_evict(&self, _event: &SubagentLifecycleEvent) -> Result<()> {
        Err(Error::validation("evict refused by hook"))
    }
}

/// Hook that calls back into the runtime from inside `BeforeEvict`.
struct ReentrantHook {
    /// Filled after the runtime is created (the hook needs a handle to it).
    runtime: Arc<Mutex<Option<Arc<SubAgentRuntime>>>>,
}

#[async_trait::async_trait]
impl SubagentLifecycleHook for ReentrantHook {
    fn name(&self) -> &'static str {
        "reentrant"
    }

    async fn on_before_evict(&self, event: &SubagentLifecycleEvent) -> Result<()> {
        let runtime = self
            .runtime
            .lock()
            .clone()
            .expect("the runtime handle is installed before the spawn");
        assert_eq!(runtime.list().len(), 1);
        assert!(runtime.activity_summary(&event.subagent_id).is_some());
        runtime.record_activity(
            &event.subagent_id,
            ActivityKind::ToolCall,
            "probe",
            None,
            None,
        )?;
        Ok(())
    }
}

/// Hook that parks `BeforeEvict` until the test releases it, so a second eviction races it.
struct GatedBeforeEvict {
    /// Signalled once the hook is inside `BeforeEvict` (the claim is held).
    entered: Arc<Notify>,
    /// Waited on by the hook before it returns.
    release: Arc<Notify>,
}

#[async_trait::async_trait]
impl SubagentLifecycleHook for GatedBeforeEvict {
    fn name(&self) -> &'static str {
        "gated-before-evict"
    }

    async fn on_before_evict(&self, _event: &SubagentLifecycleEvent) -> Result<()> {
        self.entered.notify_one();
        self.release.notified().await;
        Ok(())
    }
}

/// The calls of `seen` for `stage`, in invocation order.
fn calls_for(seen: &Seen, stage: LifecycleStage) -> Vec<Call> {
    seen.lock()
        .iter()
        .filter(|call| call.stage == stage)
        .cloned()
        .collect()
}

#[tokio::test]
async fn hooks_observe_spawn_ready_before_evict_close_in_order() {
    let runtime = build_runtime(&["worker"], FakeFactory::new(Behavior::Echo));
    let (hook, seen) = RecordingHook::new("recording");
    let runtime = runtime.with_lifecycle_hooks(LifecycleHooks::new().with_hook(Arc::new(hook)));
    let id = spawn(&runtime, "worker", "task").await;
    wait_one(&runtime, &id).await;

    let eviction = runtime
        .evict_with_hooks(&id)
        .await
        .expect("the instance is evictable");
    assert_eq!(eviction.subagent_id, id);
    assert_eq!(eviction.previous_status, SubagentStatus::Completed);
    assert_eq!(eviction.reason, EVICTION_REASON);
    assert_eq!(eviction.transcript.count, 4);

    let calls = seen.lock().clone();
    assert_eq!(
        calls.iter().map(|call| call.stage).collect::<Vec<_>>(),
        vec![
            LifecycleStage::Spawn,
            LifecycleStage::Ready,
            LifecycleStage::BeforeEvict,
            LifecycleStage::Close
        ]
    );
    assert!(calls.iter().all(|call| call.subagent_id == id));
    assert!(calls[0].reason.is_none());
    assert_eq!(calls[0].status, SubagentStatus::PendingInit);
    assert!(calls[1].reason.is_none());
    assert_eq!(calls[1].status, SubagentStatus::Completed);
    assert!(calls[2].reason.is_none());
    assert_eq!(calls[2].status, SubagentStatus::Completed);
    assert_eq!(calls[3].reason.as_deref(), Some(EVICTION_REASON));
    assert_eq!(calls[3].status, SubagentStatus::Closed);
}

#[tokio::test]
async fn a_failing_before_evict_hook_is_contained() {
    let runtime = build_runtime(&["worker"], FakeFactory::new(Behavior::Echo))
        .with_lifecycle_hooks(LifecycleHooks::new().with_hook(Arc::new(FailingBeforeEvict)));
    let id = spawn(&runtime, "worker", "task").await;
    wait_one(&runtime, &id).await;

    let previous = runtime
        .evict(&id)
        .await
        .expect("the eviction still happens");
    assert_eq!(previous, SubagentStatus::Completed);
    let metadata = runtime.get_metadata(&id).expect("the tombstone exists");
    assert_eq!(metadata.status, SubagentStatus::Closed);
    assert_eq!(metadata.closed_reason.as_deref(), Some(EVICTION_REASON));
    assert_eq!(
        runtime
            .transcript(&id)
            .and_then(|entries| entries.last().map(|entry| entry.kind)),
        Some(TranscriptKind::Evicted)
    );
    assert_eq!(
        runtime
            .activity(&id)
            .and_then(|frames| frames.last().map(|frame| frame.kind)),
        Some(ActivityKind::Evicted)
    );
}

#[tokio::test]
async fn close_fires_a_manual_close_hook() {
    let runtime = build_runtime(&["worker"], FakeFactory::new(Behavior::Echo));
    let (hook, seen) = RecordingHook::new("recording");
    let runtime = runtime.with_lifecycle_hooks(LifecycleHooks::new().with_hook(Arc::new(hook)));
    let id = spawn(&runtime, "worker", "task").await;
    wait_one(&runtime, &id).await;

    runtime.close(&id).await.expect("the instance closes");
    let closes = calls_for(&seen, LifecycleStage::Close);
    assert_eq!(closes.len(), 1);
    assert_eq!(closes[0].subagent_id, id);
    assert_eq!(closes[0].reason.as_deref(), Some("manual"));
    assert_eq!(closes[0].status, SubagentStatus::Closed);
}

#[tokio::test]
async fn cancel_all_fires_parent_ended_closes_in_spawn_order() {
    let (gate, factory) = blocked();
    let runtime = build_runtime(&["worker"], factory);
    let (hook, seen) = RecordingHook::new("recording");
    let runtime = runtime.with_lifecycle_hooks(LifecycleHooks::new().with_hook(Arc::new(hook)));
    let first = spawn(&runtime, "worker", "one").await;
    let second = spawn(&runtime, "worker", "two").await;
    wait_running(&runtime, &first).await;
    wait_running(&runtime, &second).await;

    runtime.cancel_all().await;
    let closes = calls_for(&seen, LifecycleStage::Close);
    assert_eq!(closes.len(), 2);
    assert_eq!(closes[0].subagent_id, first);
    assert_eq!(closes[1].subagent_id, second);
    assert!(closes
        .iter()
        .all(|call| call.reason.as_deref() == Some("parent_ended")));
    assert!(closes
        .iter()
        .all(|call| call.status == SubagentStatus::Closed));
    gate.send_replace(true);
}

#[tokio::test]
async fn capacity_pressure_eviction_fires_close_with_evicted_only() {
    let runtime = build_runtime(&["worker"], FakeFactory::new(Behavior::Echo));
    let (hook, seen) = RecordingHook::new("recording");
    let runtime = runtime
        .with_quota(SubAgentQuota::default().with_max_subagents(1))
        .with_lifecycle_hooks(LifecycleHooks::new().with_hook(Arc::new(hook)));
    let first = spawn(&runtime, "worker", "one").await;
    wait_one(&runtime, &first).await;

    let second = spawn(&runtime, "worker", "two").await;
    wait_one(&runtime, &second).await;
    assert_eq!(runtime.get_status(&first), SubagentStatus::Closed);

    let closes = calls_for(&seen, LifecycleStage::Close);
    assert_eq!(closes.len(), 1);
    assert_eq!(closes[0].subagent_id, first);
    assert_eq!(closes[0].reason.as_deref(), Some(EVICTION_REASON));
    assert_eq!(closes[0].status, SubagentStatus::Closed);
    // The capacity-pressure path cannot await `BeforeEvict`: it fires `Close` only.
    assert_eq!(
        calls_for(&seen, LifecycleStage::BeforeEvict),
        Vec::<Call>::new()
    );
    assert_eq!(calls_for(&seen, LifecycleStage::Spawn).len(), 2);
    assert_eq!(calls_for(&seen, LifecycleStage::Ready).len(), 2);
}

#[tokio::test]
async fn a_hook_may_call_back_into_the_runtime_without_deadlock() {
    let slot: Arc<Mutex<Option<Arc<SubAgentRuntime>>>> = Arc::new(Mutex::new(None));
    let runtime = build_runtime(&["worker"], FakeFactory::new(Behavior::Echo))
        .with_lifecycle_hooks(LifecycleHooks::new().with_hook(Arc::new(ReentrantHook {
            runtime: Arc::clone(&slot),
        })));
    let runtime = Arc::new(runtime);
    *slot.lock() = Some(Arc::clone(&runtime));

    let id = spawn(&runtime, "worker", "task").await;
    wait_one(&runtime, &id).await;
    let previous = tokio::time::timeout(Duration::from_secs(5), runtime.evict(&id))
        .await
        .expect("the reentrant hook must not deadlock")
        .expect("the eviction succeeds");
    assert_eq!(previous, SubagentStatus::Completed);
}

#[tokio::test]
async fn concurrent_evictions_of_one_id_write_exactly_one_tombstone() {
    let entered = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let runtime = build_runtime(&["worker"], FakeFactory::new(Behavior::Echo))
        .with_lifecycle_hooks(LifecycleHooks::new().with_hook(Arc::new(GatedBeforeEvict {
            entered: Arc::clone(&entered),
            release: Arc::clone(&release),
        })));
    let runtime = Arc::new(runtime);
    let id = spawn(&runtime, "worker", "task").await;
    wait_one(&runtime, &id).await;

    let winner = {
        let runtime = Arc::clone(&runtime);
        let id = id.clone();
        tokio::spawn(async move { runtime.evict(&id).await })
    };
    tokio::time::timeout(Duration::from_secs(5), entered.notified())
        .await
        .expect("the winner reaches BeforeEvict");

    let error = runtime.evict(&id).await.expect_err("the loser is refused");
    assert!(matches!(
        &error,
        Error::Validation(message) if *message == format!("subagent eviction in progress: {id}")
    ));
    release.notify_one();
    let previous = winner
        .await
        .expect("the winner task joins")
        .expect("the winner evicts");
    assert_eq!(previous, SubagentStatus::Completed);

    let transcript = runtime.transcript(&id).expect("the tombstone exists");
    assert_eq!(
        transcript
            .iter()
            .filter(|entry| entry.kind == TranscriptKind::Evicted)
            .count(),
        1
    );
    let frames = runtime.activity(&id).expect("the tombstone exists");
    assert_eq!(
        frames
            .iter()
            .filter(|frame| frame.kind == ActivityKind::Evicted)
            .count(),
        1
    );
    assert_eq!(
        runtime.get_metadata(&id).map(|m| m.closed_reason),
        Some(Some(EVICTION_REASON.to_owned()))
    );
}

#[tokio::test]
async fn with_lifecycle_hooks_is_additive_across_calls() {
    let (first, first_seen) = RecordingHook::new("first");
    let (second, second_seen) = RecordingHook::new("second");
    let first_hooks = LifecycleHooks::new().with_hook(Arc::new(first));
    let runtime = build_runtime(&["worker"], FakeFactory::new(Behavior::Echo))
        .with_lifecycle_hooks(first_hooks.clone())
        .with_lifecycle_hooks(LifecycleHooks::new().with_hook(Arc::new(second)))
        // Repeating a call is idempotent: the shared hook replaces itself in place.
        .with_lifecycle_hooks(first_hooks);
    assert_eq!(
        runtime.lifecycle_hooks().names(),
        vec!["first".to_owned(), "second".to_owned()]
    );

    let id = spawn(&runtime, "worker", "task").await;
    wait_one(&runtime, &id).await;
    assert_eq!(
        calls_for(&first_seen, LifecycleStage::Spawn).len(),
        1,
        "the first hook runs once, not once per registration"
    );
    assert_eq!(calls_for(&second_seen, LifecycleStage::Spawn).len(), 1);
}
