//! Persistence and restore tests of the in-process sub-agent runtime.
//!
//! They drive the runtime with the fakes of [`super`] and a [`ManualClock`] frozen at
//! [`T0`], so a bucket written by one runtime restores identically into another. Restores are
//! synchronous and the fakes answer without sleeping, so no test needs a timing bound.

use super::*;
use crate::subagents::{
    max_persisted_records, ManualClock, PersistedSubAgentBuckets, PersistedSubAgentRecord,
    SubAgentQuota, SubagentActivity,
};

/// The instant every manual clock starts at.
const T0: i64 = 1_700_000_000_000;

/// Builds a runtime over the `worker` type whose clock is frozen at [`T0`].
fn frozen_runtime(factory: Arc<FakeFactory>) -> SubAgentRuntime {
    build_runtime(&["worker"], factory).with_clock(Arc::new(ManualClock::new(T0)))
}

/// Builds a persisted record for `id`, in `status`/`activity` at `spawn_seq`.
fn record(
    id: &str,
    status: SubagentStatus,
    activity: SubagentActivity,
    spawn_seq: u64,
) -> PersistedSubAgentRecord {
    PersistedSubAgentRecord {
        subagent_id: id.to_owned(),
        sub_session_id: id.to_owned(),
        subagent_type: "worker".to_owned(),
        task_description: format!("task of {id}"),
        status,
        activity,
        depth: 1,
        spawn_seq,
        agent_id: format!("agent_{id}"),
        created_at_ms: T0,
        updated_at_ms: T0,
        last_used_at_ms: T0,
        last_activity_at_ms: T0,
        ..PersistedSubAgentRecord::default()
    }
}

/// Builds a bucket holding `records`.
fn bucket_with(records: Vec<PersistedSubAgentRecord>) -> PersistedSubAgentBuckets {
    let mut buckets = PersistedSubAgentBuckets::new();
    for record in records {
        buckets.insert(record);
    }
    buckets
}

#[tokio::test]
async fn save_then_load_rebuilds_a_completed_and_a_closed_instance() {
    let runtime = frozen_runtime(FakeFactory::new(Behavior::Echo));
    let done = spawn(&runtime, "worker", "hello").await;
    assert_eq!(
        wait_one(&runtime, &done).await.output.as_deref(),
        Some("echo:hello")
    );
    let closed = spawn(&runtime, "worker", "later").await;
    wait_one(&runtime, &closed).await;
    assert_eq!(
        runtime.close(&closed).await.unwrap(),
        SubagentStatus::Completed
    );

    let buckets = runtime.save();
    assert_eq!(buckets.len(), 2);
    assert!(buckets.revision >= 1);
    assert_eq!(runtime.snapshot().len(), 2);

    let restored = frozen_runtime(FakeFactory::new(Behavior::Echo));
    let report = restored.load(&buckets).unwrap();
    assert_eq!(
        report,
        LoadReport {
            records: 2,
            restored: 2,
            skipped: 0,
        }
    );

    let outcome = restored.get_outcome(&done).unwrap();
    assert_eq!(outcome.status, SubagentStatus::Completed);
    assert_eq!(outcome.output.as_deref(), Some("echo:hello"));
    assert_eq!(
        outcome.agent_id,
        runtime.get_outcome(&done).unwrap().agent_id
    );
    assert_eq!(
        outcome.transcript,
        runtime.transcript_summary(&done).unwrap()
    );
    assert_eq!(
        restored.transcript(&done).unwrap(),
        runtime.transcript(&done).unwrap()
    );
    assert_eq!(restored.capacity(), (1, 10));

    let record = restored.snapshot();
    let record = record.get(&done).unwrap();
    assert_eq!(record.config.agent_card.name, "worker");
    assert_eq!(record.config.subagent_type.as_deref(), Some("worker"));
    let tombstone = restored.get_metadata(&closed).unwrap();
    assert_eq!(tombstone.status, SubagentStatus::Closed);
    assert_eq!(tombstone.closed_reason.as_deref(), Some("manual"));
}

#[tokio::test]
async fn resume_rebuilds_the_child_of_a_restored_instance() {
    let runtime = frozen_runtime(FakeFactory::new(Behavior::Echo));
    let id = spawn(&runtime, "worker", "first").await;
    wait_one(&runtime, &id).await;
    let buckets = runtime.save();

    let factory = FakeFactory::new(Behavior::Echo);
    let restored = frozen_runtime(Arc::clone(&factory));
    assert_eq!(restored.load(&buckets).unwrap().restored, 1);
    assert!(
        factory.recorded.lock().is_empty(),
        "restoring must not build a child"
    );

    let error = restored
        .send_input(&id, "too soon", false)
        .await
        .unwrap_err();
    let expected = format!("subagent not resumed: {id}; call subagent_resume first");
    assert!(matches!(&error, Error::Validation(message) if *message == expected));

    let report = restored.resume(&id).await.unwrap();
    assert_eq!(
        report,
        ResumeReport {
            status: SubagentStatus::Completed,
            restored: true,
            message: None,
        }
    );
    let recorded = factory.recorded.lock().clone();
    assert_eq!(recorded.len(), 1);
    assert_eq!(recorded[0].1, id);
    assert_eq!(recorded[0].0.subagent_type.as_deref(), Some("worker"));

    restored.send_input(&id, "again", false).await.unwrap();
    let outcome = wait_one(&restored, &id).await;
    assert_eq!(outcome.status, SubagentStatus::Completed);
    assert_eq!(outcome.output.as_deref(), Some("echo:again"));
    assert!(matches!(
        restored.resume(&id).await.unwrap(),
        ResumeReport {
            restored: false,
            message: Some(_),
            ..
        }
    ));
}

#[tokio::test]
async fn mid_turn_records_are_downgraded_on_restore() {
    let buckets = bucket_with(vec![
        record(
            "queued",
            SubagentStatus::PendingInit,
            SubagentActivity::Queued,
            1,
        ),
        record(
            "running",
            SubagentStatus::Running,
            SubagentActivity::Running,
            2,
        ),
    ]);
    let factory = FakeFactory::new(Behavior::Echo);
    let restored = frozen_runtime(Arc::clone(&factory));
    assert_eq!(
        restored.load(&buckets).unwrap(),
        LoadReport {
            records: 2,
            restored: 2,
            skipped: 0,
        }
    );

    for id in ["queued", "running"] {
        let metadata = restored.get_metadata(id).unwrap();
        assert_eq!(metadata.status, SubagentStatus::Interrupted);
        assert_eq!(metadata.activity, SubagentActivity::Idle);
        assert!(metadata.closed_reason.is_none());
        let error = restored.send_input(id, "wake up", false).await.unwrap_err();
        assert!(
            matches!(&error, Error::Validation(message) if message.contains("subagent not resumed"))
        );
    }
    // A restored non-closed instance keeps its status when resumed.
    let report = restored.resume("running").await.unwrap();
    assert_eq!(report.status, SubagentStatus::Interrupted);
    assert!(report.restored);
    assert_eq!(factory.recorded.lock().len(), 1);
    restored
        .send_input("running", "again", false)
        .await
        .unwrap();
    assert_eq!(
        wait_one(&restored, "running").await.output.as_deref(),
        Some("echo:again")
    );
}

#[tokio::test]
async fn the_live_cap_bounds_how_many_records_restore() {
    let buckets = bucket_with(vec![
        record(
            "first",
            SubagentStatus::Completed,
            SubagentActivity::Idle,
            1,
        ),
        record(
            "second",
            SubagentStatus::Completed,
            SubagentActivity::Idle,
            2,
        ),
    ]);
    let restored = build_runtime(&["worker"], FakeFactory::new(Behavior::Echo))
        .with_quota(SubAgentQuota::default().with_max_subagents(1))
        .with_clock(Arc::new(ManualClock::new(T0)));
    assert_eq!(
        restored.load(&buckets).unwrap(),
        LoadReport {
            records: 2,
            restored: 1,
            skipped: 1,
        }
    );
    assert_eq!(restored.capacity(), (1, 1));
    assert_eq!(restored.get_status("first"), SubagentStatus::Completed);
    assert!(restored.get_metadata("second").is_none());
    assert_eq!(restored.spawned_total(), 2);
}

#[tokio::test]
async fn loading_the_same_bucket_twice_restores_nothing_new() {
    let buckets = bucket_with(vec![
        record(
            "alpha",
            SubagentStatus::Completed,
            SubagentActivity::Idle,
            1,
        ),
        record("beta", SubagentStatus::Completed, SubagentActivity::Idle, 2),
    ]);
    let restored = frozen_runtime(FakeFactory::new(Behavior::Echo));
    assert_eq!(restored.load(&buckets).unwrap().restored, 2);
    let before = restored.snapshot();
    assert_eq!(
        restored.load(&buckets).unwrap(),
        LoadReport {
            records: 2,
            restored: 0,
            skipped: 2,
        }
    );
    assert_eq!(restored.capacity(), (2, 10));
    let after = restored.snapshot();
    assert_eq!(after.ids(), before.ids());
    assert_eq!(after.revision, before.revision + 1);
}

#[tokio::test]
async fn save_trims_closed_tombstones_to_the_retention_bound() {
    let runtime = build_runtime(&["worker"], FakeFactory::new(Behavior::Echo))
        .with_quota(
            SubAgentQuota::default()
                .with_max_subagents(1)
                .with_total_spawned(64),
        )
        .with_clock(Arc::new(ManualClock::new(T0)));
    let bound = max_persisted_records(runtime.quota().max_subagents);
    assert_eq!(bound, 40);
    for index in 0..bound + 5 {
        let id = spawn(&runtime, "worker", &format!("task {index}")).await;
        wait_one(&runtime, &id).await;
        runtime.close(&id).await.unwrap();
    }
    assert_eq!(runtime.snapshot().len(), bound + 5);
    assert_eq!(runtime.capacity(), (0, 1));
    let saved = runtime.save();
    assert_eq!(saved.len(), bound);
    assert_eq!(saved.revision, runtime.snapshot().revision - 1);
}

#[tokio::test]
async fn bucket_files_round_trip_and_report_io_errors() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sessions").join("subagents.json");
    let runtime = frozen_runtime(FakeFactory::new(Behavior::Echo));
    let id = spawn(&runtime, "worker", "persist me").await;
    wait_one(&runtime, &id).await;
    let written = runtime.save_to_path(&path).unwrap();
    assert_eq!(written.len(), 1);
    assert!(path.exists());

    let fresh = frozen_runtime(FakeFactory::new(Behavior::Echo));
    assert_eq!(fresh.load_from_path(&path).unwrap().restored, 1);
    assert_eq!(fresh.get_status(&id), SubagentStatus::Completed);

    let absent = fresh
        .load_from_path(&dir.path().join("absent.json"))
        .unwrap();
    assert_eq!(absent, LoadReport::default());

    let broken = dir.path().join("broken.json");
    std::fs::write(&broken, "{ not json").unwrap();
    let error = fresh.load_from_path(&broken).unwrap_err();
    assert!(matches!(error, Error::Session(_)), "unexpected: {error}");
    assert!(
        error.to_string().contains(&broken.display().to_string()),
        "unexpected: {error}"
    );
}

#[tokio::test]
async fn a_not_found_record_rejects_the_whole_bucket() {
    let buckets = bucket_with(vec![
        record("good", SubagentStatus::Completed, SubagentActivity::Idle, 1),
        record("ghost", SubagentStatus::NotFound, SubagentActivity::Idle, 2),
    ]);
    let restored = frozen_runtime(FakeFactory::new(Behavior::Echo));
    let error = restored.load(&buckets).unwrap_err();
    assert!(
        matches!(&error, Error::Validation(message) if message == "invalid subagent record: ghost has status not_found"),
        "unexpected: {error}"
    );
    assert_eq!(restored.capacity(), (0, 10));
    assert!(restored.get_metadata("good").is_none());
    assert_eq!(restored.spawned_total(), 0);
}

#[tokio::test]
async fn the_restored_count_feeds_the_spawn_budget() {
    let buckets = bucket_with(vec![
        record(
            "alpha",
            SubagentStatus::Completed,
            SubagentActivity::Idle,
            1,
        ),
        record("beta", SubagentStatus::Completed, SubagentActivity::Idle, 2),
    ]);
    let restored = build_runtime(&["worker"], FakeFactory::new(Behavior::Echo))
        .with_quota(SubAgentQuota::default().with_total_spawned(2))
        .with_clock(Arc::new(ManualClock::new(T0)));
    assert_eq!(restored.spawned_total(), 0);
    restored.load(&buckets).unwrap();
    assert_eq!(restored.spawned_total(), 2);
    let error = restored
        .spawn(SubAgentSpawnRequest::new("worker", "third"))
        .await
        .unwrap_err();
    assert!(
        matches!(&error, Error::Config(message) if message == "subagent spawn budget exhausted: spawned=2, limit=2"),
        "unexpected: {error}"
    );
}

#[test]
fn every_snapshot_advances_the_bucket_revision() {
    let runtime = build_runtime(&["worker"], FakeFactory::new(Behavior::Echo));
    let empty = runtime.snapshot();
    assert_eq!((empty.revision, empty.len()), (1, 0));
    let empty = runtime.save();
    assert_eq!((empty.revision, empty.len()), (2, 0));
    assert_eq!(runtime.load(&empty).unwrap(), LoadReport::default());
    assert_eq!(runtime.snapshot().revision, 3);
}
