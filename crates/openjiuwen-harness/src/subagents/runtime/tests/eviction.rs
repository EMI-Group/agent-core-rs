//! Deterministic LRU eviction tests of the in-process sub-agent runtime.
//!
//! They reuse the fakes of [`super`]: `FakeFactory::new(Behavior::Echo)` for instances that
//! park in the `Idle` phase and [`blocked`] for turns that stay `Running` (or `Queued` behind a
//! saturated turn semaphore) until the test opens the gate. Every test is deterministic: no
//! sleeps, and the clock-sensitive ones inject a [`ManualClock`].
use super::quota::{config_error, split_request};
use super::*;
use crate::subagents::{ManualClock, SubAgentQuota, SubagentActivity, EVICTION_REASON};

/// Builds a runtime with an injected clock frozen at a fixed instant.
fn manual_runtime(factory: Arc<FakeFactory>) -> (SubAgentRuntime, Arc<ManualClock>) {
    let clock = Arc::new(ManualClock::new(1_700_000_000_000));
    let injected: Arc<dyn Clock> = clock.clone();
    (
        build_runtime(&["worker"], factory).with_clock(injected),
        clock,
    )
}

#[tokio::test]
async fn spawn_evicts_the_idle_instance_when_the_live_cap_is_hit() {
    let (runtime, _clock) = manual_runtime(FakeFactory::new(Behavior::Echo));
    let runtime = runtime.with_quota(SubAgentQuota::default().with_max_subagents(1));
    let first = spawn(&runtime, "worker", "one").await;
    wait_one(&runtime, &first).await;
    assert_eq!(runtime.capacity(), (1, 1));
    assert_eq!(runtime.eviction_candidates(), vec![first.clone()]);

    let second = spawn(&runtime, "worker", "two").await;
    assert_ne!(second, first);
    assert_eq!(runtime.capacity(), (1, 1));
    let evicted = runtime.get_metadata(&first).unwrap();
    assert_eq!(evicted.status, SubagentStatus::Closed);
    assert_eq!(evicted.activity, SubagentActivity::Closed);
    assert_eq!(evicted.closed_reason.as_deref(), Some(EVICTION_REASON));
    assert_eq!(runtime.spawned_total(), 2);

    let list = runtime.describe_list();
    assert_eq!(list["summary"]["evicted_count"], json!(1));
    assert_eq!(list["summary"]["live_count"], json!(1));
    assert_eq!(list["summary"]["closed_count"], json!(1));
    assert_eq!(list["summary"]["eviction"]["enabled"], json!(true));
    assert_eq!(
        wait_one(&runtime, &second).await.status,
        SubagentStatus::Completed
    );
}

#[tokio::test]
async fn a_running_instance_is_never_evicted() {
    let (gate, factory) = blocked();
    let runtime = build_runtime(&["worker"], factory)
        .with_quota(SubAgentQuota::default().with_max_subagents(1));
    let id = spawn(&runtime, "worker", "one").await;
    wait_running(&runtime, &id).await;
    assert_eq!(runtime.eviction_candidates(), Vec::<SubAgentId>::new());

    let error = runtime
        .spawn(SubAgentSpawnRequest::new("worker", "two"))
        .await
        .unwrap_err();
    assert_eq!(
        config_error(&error),
        "subagent capacity exceeded: used=1, limit=1"
    );
    assert_eq!(runtime.capacity(), (1, 1));
    assert_eq!(runtime.get_status(&id), SubagentStatus::Running);
    let listed = runtime.describe_list();
    assert_eq!(listed["closed_subagents"].as_array().map(Vec::len), Some(0));
    assert_eq!(listed["summary"]["evicted_count"], json!(0));
    assert_eq!(listed["summary"]["eviction"]["candidates"], json!(0));

    gate.send_replace(true);
    assert_eq!(
        wait_one(&runtime, &id).await.status,
        SubagentStatus::Completed
    );
}

#[tokio::test]
async fn an_idle_instance_is_evicted_before_an_older_queued_one() {
    let clock = Arc::new(ManualClock::new(1_000));
    let injected: Arc<dyn Clock> = clock.clone();
    let (gate, factory) = blocked();
    let runtime = build_runtime(&["worker"], factory)
        .with_quota(
            SubAgentQuota::default()
                .with_max_subagents(3)
                .with_concurrent_running(1),
        )
        .with_clock(injected);

    // The holder owns the only turn permit, so the other two turns stay `Queued`.
    let holder = spawn(&runtime, "worker", "holder").await;
    wait_running(&runtime, &holder).await;
    clock.advance(100);
    let queued = spawn(&runtime, "worker", "queued").await;
    assert_eq!(
        runtime.get_metadata(&queued).unwrap().activity,
        SubagentActivity::Queued
    );

    // Closing a queued instance and resuming it parks it in `Idle` with a newer timestamp.
    clock.advance(100);
    let victim = spawn(&runtime, "worker", "victim").await;
    assert_eq!(
        runtime.close(&victim).await.unwrap(),
        SubagentStatus::PendingInit
    );
    clock.advance(100);
    assert!(runtime.resume(&victim).await.unwrap().restored);
    assert_eq!(
        runtime.get_metadata(&victim).unwrap().activity,
        SubagentActivity::Idle
    );

    // The idle victim is newer than the queued instance but still comes first.
    assert_eq!(
        runtime.eviction_candidates(),
        vec![victim.clone(), queued.clone()]
    );
    assert_eq!(runtime.evict_lru().await.unwrap(), Some(victim.clone()));
    let evicted = runtime.get_metadata(&victim).unwrap();
    assert_eq!(evicted.status, SubagentStatus::Closed);
    assert_eq!(evicted.closed_reason.as_deref(), Some(EVICTION_REASON));
    assert_eq!(
        runtime.get_metadata(&queued).unwrap().activity,
        SubagentActivity::Queued
    );
    assert_eq!(runtime.eviction_candidates(), vec![queued]);

    gate.send_replace(true);
}

#[tokio::test]
async fn candidates_are_ordered_by_age_then_by_id() {
    let (runtime, clock) = manual_runtime(FakeFactory::new(Behavior::Echo));
    let oldest = spawn(&runtime, "worker", "oldest").await;
    wait_one(&runtime, &oldest).await;
    clock.advance(1_000);
    let newer = spawn(&runtime, "worker", "newer").await;
    wait_one(&runtime, &newer).await;

    assert_eq!(
        runtime.eviction_candidates(),
        vec![oldest.clone(), newer.clone()]
    );
    assert_eq!(runtime.evict_lru().await.unwrap(), Some(oldest.clone()));
    assert_eq!(runtime.eviction_candidates(), vec![newer]);
}

#[tokio::test]
async fn equal_timestamps_break_ties_by_the_smallest_id() {
    let (runtime, _clock) = manual_runtime(FakeFactory::new(Behavior::Echo));
    let mut first = SubAgentSpawnRequest::new("worker", "first");
    first.subagent_id = Some("b-idle".to_owned());
    let b = runtime.spawn(first).await.unwrap();
    let mut second = SubAgentSpawnRequest::new("worker", "second");
    second.subagent_id = Some("a-idle".to_owned());
    let a = runtime.spawn(second).await.unwrap();
    wait_one(&runtime, &a).await;
    wait_one(&runtime, &b).await;

    assert_eq!(
        runtime.eviction_candidates(),
        vec!["a-idle".to_owned(), "b-idle".to_owned()]
    );
    assert_eq!(runtime.evict_lru().await.unwrap(), Some(a));
}

#[tokio::test]
async fn evict_lru_returns_the_id_then_none_when_nothing_is_evictable() {
    let runtime = build_runtime(&["worker"], FakeFactory::new(Behavior::Echo));
    assert_eq!(runtime.evict_lru().await.unwrap(), None);
    assert_eq!(runtime.eviction_candidates(), Vec::<SubAgentId>::new());

    let id = spawn(&runtime, "worker", "task").await;
    wait_one(&runtime, &id).await;
    assert_eq!(runtime.eviction_candidates(), vec![id.clone()]);
    assert_eq!(runtime.evict_lru().await.unwrap(), Some(id.clone()));
    assert_eq!(runtime.get_status(&id), SubagentStatus::Closed);
    assert_eq!(runtime.capacity(), (0, 10));
    assert_eq!(runtime.spawned_total(), 1);
    assert_eq!(runtime.eviction_candidates(), Vec::<SubAgentId>::new());
    assert_eq!(runtime.evict_lru().await.unwrap(), None);
}

#[tokio::test]
async fn evict_validates_running_unknown_and_closed_ids() {
    let (gate, factory) = blocked();
    let runtime = build_runtime(&["worker"], factory);
    let error = runtime.evict("ghost").await.unwrap_err();
    assert!(matches!(&error, Error::Validation(message) if message == "subagent not found: ghost"));

    let id = spawn(&runtime, "worker", "task").await;
    wait_running(&runtime, &id).await;
    let expected = format!("cannot evict running subagent: {id}");
    let error = runtime.evict(&id).await.unwrap_err();
    assert!(matches!(&error, Error::Validation(message) if *message == expected));
    assert_eq!(runtime.get_status(&id), SubagentStatus::Running);

    gate.send_replace(true);
    wait_one(&runtime, &id).await;
    assert_eq!(runtime.evict(&id).await.unwrap(), SubagentStatus::Completed);
    let metadata = runtime.get_metadata(&id).unwrap();
    assert_eq!(metadata.status, SubagentStatus::Closed);
    assert_eq!(metadata.closed_reason.as_deref(), Some(EVICTION_REASON));

    let expected = format!("subagent already closed: {id}");
    let error = runtime.evict(&id).await.unwrap_err();
    assert!(matches!(&error, Error::Validation(message) if *message == expected));
}

#[tokio::test]
async fn eviction_does_not_reset_the_cumulative_spawn_budget() {
    let runtime = build_runtime(&["worker"], FakeFactory::new(Behavior::Echo)).with_quota(
        SubAgentQuota::default()
            .with_max_subagents(1)
            .with_total_spawned(2),
    );
    let first = spawn(&runtime, "worker", "one").await;
    wait_one(&runtime, &first).await;
    let second = spawn(&runtime, "worker", "two").await;
    assert_eq!(runtime.get_status(&first), SubagentStatus::Closed);
    wait_one(&runtime, &second).await;
    assert_eq!(runtime.spawned_total(), 2);

    let error = runtime
        .spawn(SubAgentSpawnRequest::new("worker", "three"))
        .await
        .unwrap_err();
    assert_eq!(
        config_error(&error),
        "subagent spawn budget exhausted: spawned=2, limit=2"
    );
    assert_eq!(runtime.spawned_total(), 2);
}

#[tokio::test]
async fn a_disabled_policy_reports_its_candidates_and_closes_nothing() {
    let runtime = build_runtime(&["worker"], FakeFactory::new(Behavior::Echo)).with_limits(1, 1);
    let first = spawn(&runtime, "worker", "one").await;
    wait_one(&runtime, &first).await;
    let listed = runtime.describe_list();
    assert_eq!(listed["summary"]["eviction"]["enabled"], json!(false));
    assert_eq!(listed["summary"]["eviction"]["candidates"], json!(1));

    let error = runtime
        .spawn(SubAgentSpawnRequest::new("worker", "two"))
        .await
        .unwrap_err();
    assert_eq!(
        config_error(&error),
        "subagent capacity exceeded: used=1, limit=1"
    );
    assert_eq!(runtime.get_status(&first), SubagentStatus::Completed);
    assert_eq!(
        runtime.describe_list()["summary"]["evicted_count"],
        json!(0)
    );
}

#[tokio::test]
async fn a_nested_spawn_never_evicts_its_own_parent() {
    let runtime = build_runtime(&["worker"], FakeFactory::new(Behavior::Echo))
        .with_quota(SubAgentQuota::default().with_max_subagents(1));
    let root = spawn(&runtime, "worker", "root").await;
    wait_one(&runtime, &root).await;

    let error = runtime.spawn(split_request(&root)).await.unwrap_err();
    assert_eq!(
        config_error(&error),
        "subagent capacity exceeded: used=1, limit=1"
    );
    assert_eq!(runtime.get_status(&root), SubagentStatus::Completed);
}
