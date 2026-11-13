//! Quota, clock and activity tests of the in-process sub-agent runtime.
//!
//! They reuse the fakes of [`super`]: `FakeFactory::new(Behavior::Echo)` for completed turns
//! and [`blocked`] for turns that stay `Running` until the test opens the gate. Every test is
//! deterministic: no sleeps, and the clock-injecting tests drive a [`ManualClock`].
use super::*;
use crate::subagents::{ManualClock, SubAgentQuota, SubagentActivity};

fn config_error(error: &Error) -> &str {
    match error {
        Error::Config(message) => message,
        other => panic!("expected a config error, got {other:?}"),
    }
}

fn split_request(parent: &str) -> SubAgentSpawnRequest {
    let mut request = SubAgentSpawnRequest::new("worker", "nested");
    request.parent_subagent_id = Some(parent.to_owned());
    request
}

#[tokio::test]
async fn cumulative_spawn_budget_is_not_reset_by_close() {
    let runtime = build_runtime(&["worker"], FakeFactory::new(Behavior::Echo))
        .with_quota(SubAgentQuota::default().with_total_spawned(2));
    let first = spawn(&runtime, "worker", "one").await;
    wait_one(&runtime, &first).await;
    let second = spawn(&runtime, "worker", "two").await;
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

    runtime.close(&first).await.unwrap();
    assert_eq!(runtime.capacity(), (1, 10));
    assert_eq!(runtime.spawned_total(), 2);
    let error = runtime
        .spawn(SubAgentSpawnRequest::new("worker", "three"))
        .await
        .unwrap_err();
    assert_eq!(
        config_error(&error),
        "subagent spawn budget exhausted: spawned=2, limit=2"
    );
    assert_eq!(runtime.describe_list()["capacity"]["spawned"], json!(2));
    assert_eq!(runtime.describe_list()["capacity"]["max_spawned"], json!(2));
}

#[tokio::test]
async fn spawn_budget_rolls_back_when_the_factory_fails() {
    let runtime = build_runtime(&["worker"], FakeFactory::new(Behavior::Fail))
        .with_quota(SubAgentQuota::default().with_total_spawned(1));
    let error = runtime
        .spawn(SubAgentSpawnRequest::new("worker", "boom"))
        .await
        .unwrap_err();
    assert_eq!(config_error(&error), "boom");
    assert_eq!(runtime.spawned_total(), 0);
    assert_eq!(runtime.capacity(), (0, 10));
    assert!(runtime
        .get_metadata("parent-session_sub_worker_0")
        .is_none());
    assert_eq!(runtime.list().len(), 0);
}

#[tokio::test]
async fn per_parent_limit_separates_the_root_and_nested_buckets() {
    let runtime = build_runtime(&["worker"], FakeFactory::new(Behavior::Echo))
        .with_quota(SubAgentQuota::default().with_children_per_parent(1));
    let root = spawn(&runtime, "worker", "root").await;
    wait_one(&runtime, &root).await;

    let error = runtime
        .spawn(SubAgentSpawnRequest::new("worker", "second-root"))
        .await
        .unwrap_err();
    assert_eq!(
        config_error(&error),
        "subagent child limit exceeded: parent=parent-session, used=1, limit=1"
    );

    let child = runtime.spawn(split_request(&root)).await.unwrap();
    wait_one(&runtime, &child).await;
    let error = runtime.spawn(split_request(&root)).await.unwrap_err();
    assert_eq!(
        config_error(&error),
        format!("subagent child limit exceeded: parent={root}, used=1, limit=1")
    );

    let grandchild = runtime.spawn(split_request(&child)).await.unwrap();
    wait_one(&runtime, &grandchild).await;
    assert_eq!(runtime.capacity(), (3, 10));
}

#[tokio::test]
async fn depth_limit_refuses_deeper_nesting() {
    let runtime = build_runtime(&["worker"], FakeFactory::new(Behavior::Echo))
        .with_quota(SubAgentQuota::default().with_max_depth(2));
    let root = spawn(&runtime, "worker", "root").await;
    wait_one(&runtime, &root).await;
    assert_eq!(runtime.get_metadata(&root).unwrap().depth, 1);

    let child = runtime.spawn(split_request(&root)).await.unwrap();
    wait_one(&runtime, &child).await;
    assert_eq!(runtime.get_metadata(&child).unwrap().depth, 2);

    let error = runtime.spawn(split_request(&child)).await.unwrap_err();
    assert_eq!(
        config_error(&error),
        "subagent depth limit exceeded: depth=3, limit=2"
    );
}

#[tokio::test]
async fn unknown_or_closed_parent_is_a_validation_error() {
    let runtime = build_runtime(&["worker"], FakeFactory::new(Behavior::Echo));
    let error = runtime.spawn(split_request("ghost")).await.unwrap_err();
    assert!(
        matches!(&error, Error::Validation(message) if message == "subagent parent not found: ghost")
    );

    let parent = spawn(&runtime, "worker", "parent").await;
    wait_one(&runtime, &parent).await;
    runtime.close(&parent).await.unwrap();
    let error = runtime.spawn(split_request(&parent)).await.unwrap_err();
    let expected = format!("subagent parent not found: {parent}");
    assert!(matches!(&error, Error::Validation(message) if *message == expected));
}

#[tokio::test]
async fn default_quota_never_refuses_optional_limits() {
    let runtime = build_runtime(&["worker"], FakeFactory::new(Behavior::Echo));
    let quota = *runtime.quota();
    assert_eq!(quota, SubAgentQuota::default());
    assert_eq!(quota.max_total_spawned, None);
    assert_eq!(quota.max_children_per_parent, None);
    assert_eq!(quota.max_depth, None);
    assert!(quota.enable_lru_eviction);

    let root = spawn(&runtime, "worker", "root").await;
    let child = runtime.spawn(split_request(&root)).await.unwrap();
    let grandchild = runtime.spawn(split_request(&child)).await.unwrap();
    wait_one(&runtime, &grandchild).await;
    assert_eq!(runtime.get_metadata(&grandchild).unwrap().depth, 3);
    assert_eq!(runtime.spawned_total(), 3);
}

#[tokio::test]
async fn injected_clock_drives_the_activity_timestamps() {
    let clock = Arc::new(ManualClock::new(1_700_000_000_000));
    let injected: Arc<dyn Clock> = clock.clone();
    let runtime = build_runtime(&["worker"], FakeFactory::new(Behavior::Echo)).with_clock(injected);
    let id = spawn(&runtime, "worker", "task").await;
    wait_one(&runtime, &id).await;

    let metadata = runtime.get_metadata(&id).unwrap();
    assert_eq!(metadata.created_at, "2023-11-14T22:13:20.000Z");
    assert_eq!(metadata.updated_at, "2023-11-14T22:13:20.000Z");
    assert_eq!(metadata.last_used_at, "2023-11-14T22:13:20.000Z");

    clock.advance(5_000);
    runtime.send_input(&id, "again", false).await.unwrap();
    let metadata = runtime.get_metadata(&id).unwrap();
    assert_eq!(metadata.created_at, "2023-11-14T22:13:20.000Z");
    assert_eq!(metadata.last_used_at, "2023-11-14T22:13:25.000Z");
    assert_eq!(
        wait_one(&runtime, &id).await.output.as_deref(),
        Some("echo:again")
    );
}

#[tokio::test]
async fn activity_phase_follows_the_lifecycle() {
    let (gate, factory) = blocked();
    let runtime = build_runtime(&["worker"], factory);
    let id = spawn(&runtime, "worker", "task").await;
    let queued = runtime.get_metadata(&id).unwrap();
    assert_eq!(queued.activity, SubagentActivity::Queued);

    wait_running(&runtime, &id).await;
    let running = runtime.get_metadata(&id).unwrap();
    assert_eq!(running.activity, SubagentActivity::Running);
    assert_eq!(running.last_used_at, running.updated_at);
    let listed = runtime.describe_list();
    assert_eq!(listed["subagents"][0]["activity"], json!("running"));
    assert_eq!(listed["capacity"]["running"], json!(1));
    assert_eq!(listed["capacity"]["max_running"], json!(5));

    gate.send_replace(true);
    wait_one(&runtime, &id).await;
    let idle = runtime.get_metadata(&id).unwrap();
    assert_eq!(idle.status, SubagentStatus::Completed);
    assert_eq!(idle.activity, SubagentActivity::Idle);
    assert_eq!(idle.closed_reason, None);

    assert_eq!(runtime.close(&id).await.unwrap(), SubagentStatus::Completed);
    let closed = runtime.get_metadata(&id).unwrap();
    assert_eq!(closed.activity, SubagentActivity::Closed);
    assert_eq!(closed.closed_reason.as_deref(), Some("manual"));
    let listed = runtime.describe_list();
    assert_eq!(listed["closed_subagents"][0]["activity"], json!("closed"));
    assert_eq!(
        listed["closed_subagents"][0]["closed_reason"],
        json!("manual")
    );
    assert_eq!(listed["capacity"]["running"], json!(0));

    let report = runtime.resume(&id).await.unwrap();
    assert!(report.restored);
    let restored = runtime.get_metadata(&id).unwrap();
    assert_eq!(restored.activity, SubagentActivity::Idle);
    assert_eq!(restored.closed_reason, None);
}

#[tokio::test]
async fn cancel_all_records_the_parent_ended_reason() {
    let (_gate, factory) = blocked();
    let runtime = build_runtime(&["worker"], factory);
    let id = spawn(&runtime, "worker", "task").await;
    wait_running(&runtime, &id).await;
    assert_eq!(runtime.cancel_all().await, vec![id.clone()]);
    let metadata = runtime.get_metadata(&id).unwrap();
    assert_eq!(metadata.activity, SubagentActivity::Closed);
    assert_eq!(metadata.closed_reason.as_deref(), Some("parent_ended"));
}

#[tokio::test]
async fn metadata_carries_depth_parent_and_last_use() {
    let runtime = build_runtime(&["worker"], FakeFactory::new(Behavior::Echo));
    let root = spawn(&runtime, "worker", "root").await;
    wait_one(&runtime, &root).await;
    let child = runtime.spawn(split_request(&root)).await.unwrap();
    wait_one(&runtime, &child).await;

    let root_metadata = runtime.get_metadata(&root).unwrap();
    assert_eq!(root_metadata.depth, 1);
    assert_eq!(root_metadata.parent_subagent_id, None);
    let child_metadata = runtime.get_metadata(&child).unwrap();
    assert_eq!(child_metadata.depth, 2);
    assert_eq!(
        child_metadata.parent_subagent_id.as_deref(),
        Some(root.as_str())
    );
    assert_ne!(child_metadata.last_used_at, "");
    assert!(runtime.get_metadata("ghost").is_none());

    let listed = runtime.describe_list();
    assert_eq!(listed["capacity"]["used"], json!(2));
    assert_eq!(listed["capacity"]["max"], json!(10));
    assert_eq!(listed["capacity"]["spawned"], json!(2));
    assert_eq!(listed["capacity"]["max_spawned"], Value::Null);
    assert_eq!(listed["subagents"][1]["depth"], json!(2));
    assert_eq!(listed["subagents"][1]["parent_subagent_id"], json!(root));
}

#[tokio::test]
async fn with_limits_disables_lru_eviction_and_keeps_the_legacy_message() {
    let runtime = build_runtime(&["worker"], FakeFactory::new(Behavior::Echo)).with_limits(1, 1);
    assert_eq!(runtime.quota().max_subagents, 1);
    assert_eq!(runtime.quota().max_concurrent_running, 1);
    assert!(!runtime.quota().enable_lru_eviction);
    let first = spawn(&runtime, "worker", "one").await;
    wait_one(&runtime, &first).await;
    let error = runtime
        .spawn(SubAgentSpawnRequest::new("worker", "two"))
        .await
        .unwrap_err();
    assert_eq!(
        config_error(&error),
        "subagent capacity exceeded: used=1, limit=1"
    );
}
