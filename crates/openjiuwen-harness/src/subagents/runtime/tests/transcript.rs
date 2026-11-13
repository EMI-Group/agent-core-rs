//! Transcript tests of the in-process sub-agent runtime.
//!
//! They drive the runtime with the fakes of [`super`] and a [`ManualClock`] frozen at a fixed
//! instant, so every recorded entry has a known sequence number and timestamp. Instances that
//! must stay `Running` use the gate of [`blocked`]; instances whose first turn must not start
//! yet use the factory gate of [`create_blocked`]. No test sleeps.

use super::*;
use crate::subagents::{
    ManualClock, SubAgentQuota, TranscriptEntry, TranscriptKind, TranscriptSummary, EVICTION_REASON,
};

/// The instant every manual clock starts at.
const T0: i64 = 1_700_000_000_000;

/// Builds a runtime whose clock is frozen at [`T0`].
fn manual_runtime(factory: Arc<FakeFactory>) -> (SubAgentRuntime, Arc<ManualClock>) {
    let clock = Arc::new(ManualClock::new(T0));
    let injected: Arc<dyn Clock> = clock.clone();
    let runtime = build_runtime(&["worker"], factory).with_clock(injected);
    (runtime, clock)
}

/// Waits until `id` is reserved (the spawn may still be blocked inside the factory).
async fn wait_reserved(runtime: &SubAgentRuntime, id: &str) {
    for _ in 0..200 {
        if runtime.get_status(id) != SubagentStatus::NotFound {
            return;
        }
        tokio::task::yield_now().await;
    }
    panic!("subagent was not reserved: {id}");
}

/// Spawns `worker` on a background task with a pinned id and returns the join handle.
fn spawn_pinned(
    runtime: &Arc<SubAgentRuntime>,
    task: &str,
) -> tokio::task::JoinHandle<Result<SubAgentId>> {
    let mut request = SubAgentSpawnRequest::new("worker", task);
    request.subagent_id = Some("sub-1".to_owned());
    let runtime = Arc::clone(runtime);
    tokio::spawn(async move { runtime.spawn(request).await })
}

/// Returns the `(kind, content)` pairs of `entries`, for compact assertions.
fn pairs(entries: &[TranscriptEntry]) -> Vec<(TranscriptKind, &str)> {
    entries
        .iter()
        .map(|entry| (entry.kind, entry.content.as_str()))
        .collect()
}

#[tokio::test]
async fn lifecycle_records_spawn_input_status_and_completion_in_order() {
    let (gate, factory) = create_blocked();
    let (runtime, clock) = manual_runtime(factory);
    let runtime = Arc::new(runtime);
    let spawner = spawn_pinned(&runtime, "explore");
    wait_reserved(&runtime, "sub-1").await;
    // The instance has no agent yet, so this input is queued: it is recorded before the first
    // turn starts.
    runtime.send_input("sub-1", "next", false).await.unwrap();
    assert_eq!(clock.now_ms(), T0);
    gate.send_replace(true);
    spawner.await.unwrap().unwrap();
    let outcome = wait_one(&runtime, "sub-1").await;
    assert_eq!(outcome.status, SubagentStatus::Completed);
    assert_eq!(outcome.output.as_deref(), Some("echo:next"));

    let entries = runtime.transcript("sub-1").unwrap();
    assert_eq!(
        pairs(&entries),
        vec![
            (TranscriptKind::Spawn, "explore"),
            (TranscriptKind::Input, "next"),
            (TranscriptKind::Status, "running"),
            (TranscriptKind::Input, "next"),
            (TranscriptKind::Status, "running"),
            (TranscriptKind::Completion, "echo:next"),
        ]
    );
    let seqs: Vec<u64> = entries.iter().map(|entry| entry.seq).collect();
    assert_eq!(seqs, vec![1, 2, 3, 4, 5, 6]);
    assert!(entries.iter().all(|entry| entry.at_ms == T0));
    assert!(entries
        .iter()
        .all(|entry| entry.subagent_id == "sub-1" && !entry.role.is_empty()));
}

#[tokio::test]
async fn an_errored_turn_is_recorded_with_its_message() {
    let (runtime, _clock) = manual_runtime(FakeFactory::new(Behavior::ErrorResult));
    let id = spawn(&runtime, "worker", "boom-task").await;
    let outcome = wait_one(&runtime, &id).await;
    assert_eq!(outcome.status, SubagentStatus::Errored);

    let entries = runtime.transcript(&id).unwrap();
    let last = entries.last().unwrap();
    assert_eq!(last.kind, TranscriptKind::Error);
    assert_eq!(last.content, "model failed");
    assert_eq!(last.success, Some(false));
    assert_eq!(last.role, "assistant");
    assert_eq!(
        pairs(&entries),
        vec![
            (TranscriptKind::Spawn, "boom-task"),
            (TranscriptKind::Status, "running"),
            (TranscriptKind::Error, "model failed"),
        ]
    );
}

#[tokio::test]
async fn an_interrupting_input_records_the_query_and_the_interrupted_status() {
    let (gate, factory) = blocked();
    let runtime = build_runtime(&["worker"], factory);
    let id = spawn(&runtime, "worker", "first").await;
    wait_running(&runtime, &id).await;
    runtime.send_input(&id, "second", true).await.unwrap();

    let entries = runtime.transcript(&id).unwrap();
    assert_eq!(
        pairs(&entries),
        vec![
            (TranscriptKind::Spawn, "first"),
            (TranscriptKind::Status, "running"),
            (TranscriptKind::Input, "second"),
            (TranscriptKind::Status, "interrupted"),
        ]
    );

    gate.send_replace(true);
    let outcome = wait_one(&runtime, &id).await;
    assert_eq!(outcome.status, SubagentStatus::Completed);
    assert_eq!(outcome.output.as_deref(), Some("echo:second"));
    let entries = runtime.transcript(&id).unwrap();
    assert_eq!(
        pairs(&entries)[4..],
        [
            (TranscriptKind::Status, "running"),
            (TranscriptKind::Completion, "echo:second"),
        ]
    );
}

#[tokio::test]
async fn close_resume_and_evict_record_their_own_events() {
    let (runtime, _clock) = manual_runtime(FakeFactory::new(Behavior::Echo));
    let id = spawn(&runtime, "worker", "task").await;
    wait_one(&runtime, &id).await;

    assert_eq!(runtime.close(&id).await.unwrap(), SubagentStatus::Completed);
    let entries = runtime.transcript(&id).unwrap();
    assert_eq!(
        pairs(&entries).last().copied(),
        Some((TranscriptKind::Closed, "manual"))
    );

    runtime.resume(&id).await.unwrap();
    assert_eq!(
        pairs(&runtime.transcript(&id).unwrap()).last().copied(),
        Some((TranscriptKind::Status, "idle"))
    );

    assert_eq!(runtime.evict(&id).await.unwrap(), SubagentStatus::Completed);
    let entries = runtime.transcript(&id).unwrap();
    assert_eq!(
        pairs(&entries).last().copied(),
        Some((TranscriptKind::Evicted, EVICTION_REASON))
    );
    assert_eq!(runtime.get_status(&id), SubagentStatus::Closed);
}

#[tokio::test]
async fn a_capacity_eviction_records_an_evicted_entry() {
    let (runtime, _clock) = manual_runtime(FakeFactory::new(Behavior::Echo));
    let runtime = runtime.with_quota(SubAgentQuota::default().with_max_subagents(1));
    let first = spawn(&runtime, "worker", "one").await;
    wait_one(&runtime, &first).await;
    let second = spawn(&runtime, "worker", "two").await;
    wait_one(&runtime, &second).await;

    let entries = runtime.transcript(&first).unwrap();
    assert_eq!(
        pairs(&entries).last().copied(),
        Some((TranscriptKind::Evicted, EVICTION_REASON))
    );
    assert_eq!(runtime.transcript_summary(&first).unwrap().count, 4);
}

#[tokio::test]
async fn the_ring_cap_drops_the_oldest_entries_and_reports_them() {
    let (runtime, _clock) = manual_runtime(FakeFactory::new(Behavior::Echo));
    let runtime = runtime.with_quota(SubAgentQuota::default().with_transcript_capacity(3));
    let id = spawn(&runtime, "worker", "task").await;
    wait_one(&runtime, &id).await;
    // `spawn`, `status` and `completion` already fill the ring; two more events evict two.
    runtime
        .record_tool_call(&id, "bash", "ls", Some(true))
        .unwrap();
    assert_eq!(
        runtime
            .record_tool_call(&id, "read_file", "/tmp/x", Some(false))
            .unwrap(),
        5
    );

    let entries = runtime.transcript(&id).unwrap();
    assert_eq!(entries.len(), 3);
    let summary = runtime.transcript_summary(&id).unwrap();
    assert_eq!(summary.count, 3);
    assert_eq!(summary.dropped, 2);
    assert_eq!(summary.last_kind, Some(TranscriptKind::ToolCall));
    assert_eq!(summary.last_content.as_deref(), Some("/tmp/x"));
    assert_eq!(summary.display(), "3 entries, last: tool_call: /tmp/x");
}

#[tokio::test]
async fn record_tool_call_and_record_transcript_stamp_the_runtime_envelope() {
    let (runtime, clock) = manual_runtime(FakeFactory::new(Behavior::Echo));
    let id = spawn(&runtime, "worker", "task").await;
    wait_one(&runtime, &id).await;

    assert_eq!(
        runtime
            .record_tool_call(&id, "bash", "echo hi", Some(true))
            .unwrap(),
        4
    );
    let entries = runtime.transcript(&id).unwrap();
    let tool_call = &entries[3];
    assert_eq!(tool_call.kind, TranscriptKind::ToolCall);
    assert_eq!(tool_call.tool_name.as_deref(), Some("bash"));
    assert_eq!(tool_call.content, "echo hi");
    assert_eq!(tool_call.success, Some(true));
    assert_eq!(tool_call.at_ms, T0);

    let mut subscription = runtime.subscribe_transcript(&id).unwrap();
    let seq = runtime
        .record_tool_call(&id, "grep", "needle", None)
        .unwrap();
    assert_eq!(seq, 5);
    let published = subscription.try_recv().unwrap();
    assert_eq!(published.seq, 5);
    assert_eq!(published.tool_name.as_deref(), Some("grep"));
    assert_eq!(published.success, None);

    clock.advance(2_500);
    let supplied = TranscriptEntry {
        seq: 99,
        subagent_id: "elsewhere".to_owned(),
        kind: TranscriptKind::AssistantTurn,
        at_ms: 7,
        content: "parsed the file".to_owned(),
        ..TranscriptEntry::default()
    };
    let seq = runtime.record_transcript(&id, supplied).unwrap();
    let last = runtime.transcript(&id).unwrap().pop().unwrap();
    assert_eq!(last.seq, seq);
    assert_ne!(last.seq, 99);
    assert_eq!(last.at_ms, T0 + 2_500);
    assert_eq!(last.subagent_id, id);
    assert_eq!(last.kind, TranscriptKind::AssistantTurn);

    let error = runtime
        .record_tool_call("ghost", "bash", "x", None)
        .unwrap_err();
    assert!(
        matches!(error, Error::Validation(ref message) if message == "subagent not found: ghost")
    );
    let error = runtime
        .record_transcript("ghost", TranscriptEntry::default())
        .unwrap_err();
    assert!(
        matches!(error, Error::Validation(ref message) if message == "subagent not found: ghost")
    );
}

#[tokio::test]
async fn transcript_after_paginates_and_clear_empties_the_ring() {
    let (runtime, _clock) = manual_runtime(FakeFactory::new(Behavior::Echo));
    let id = spawn(&runtime, "worker", "task").await;
    wait_one(&runtime, &id).await;
    for text in ["one", "two", "three"] {
        runtime
            .record_transcript(&id, TranscriptEntry::assistant_turn(&id, text, 0))
            .unwrap();
    }
    let seqs = |entries: Vec<TranscriptEntry>| -> Vec<u64> {
        entries.iter().map(|entry| entry.seq).collect()
    };

    assert_eq!(
        seqs(runtime.transcript_after(&id, 0, 0).unwrap()),
        vec![1, 2, 3, 4, 5, 6]
    );
    assert_eq!(
        seqs(runtime.transcript_after(&id, 1, 2).unwrap()),
        vec![2, 3]
    );
    assert_eq!(
        seqs(runtime.transcript_after(&id, 4, 0).unwrap()),
        vec![5, 6]
    );
    assert_eq!(
        runtime.transcript_after(&id, 6, 0).unwrap(),
        Vec::<TranscriptEntry>::new()
    );
    assert!(runtime.transcript_after("ghost", 0, 0).is_none());

    assert_eq!(runtime.clear_transcript(&id), Some(6));
    assert_eq!(
        runtime.transcript(&id).unwrap(),
        Vec::<TranscriptEntry>::new()
    );
    assert_eq!(runtime.transcript_summary(&id).unwrap().count, 0);
    assert_eq!(runtime.transcript_summary(&id).unwrap().last_kind, None);
    assert_eq!(runtime.clear_transcript("ghost"), None);

    // Clearing is not a ring eviction: the sequence keeps counting.
    assert_eq!(
        runtime
            .record_tool_call(&id, "bash", "again", None)
            .unwrap(),
        7
    );
    assert_eq!(runtime.transcript_summary(&id).unwrap().count, 1);
}

#[tokio::test]
async fn subscription_receives_the_entries_published_after_it() {
    let (gate, factory) = blocked();
    let runtime = build_runtime(&["worker"], factory);
    let id = spawn(&runtime, "worker", "first").await;
    wait_running(&runtime, &id).await;
    let mut subscription = runtime.subscribe_transcript(&id).unwrap();
    runtime.send_input(&id, "second", false).await.unwrap();
    gate.send_replace(true);

    let mut kinds = Vec::new();
    while kinds.last() != Some(&TranscriptKind::Completion) {
        let entry = tokio::time::timeout(Duration::from_secs(5), subscription.recv())
            .await
            .unwrap()
            .unwrap();
        kinds.push(entry.kind);
    }
    assert_eq!(
        kinds,
        vec![
            TranscriptKind::Input,
            TranscriptKind::Input,
            TranscriptKind::Status,
            TranscriptKind::Completion,
        ]
    );
    assert!(runtime.subscribe_transcript("ghost").is_none());
}

#[tokio::test]
async fn metadata_outcomes_and_describe_list_carry_the_digest() {
    let (runtime, _clock) = manual_runtime(FakeFactory::new(Behavior::Echo));
    let id = spawn(&runtime, "worker", "task").await;
    let outcome = wait_one(&runtime, &id).await;
    let summary = runtime.transcript_summary(&id).unwrap();
    assert_eq!(outcome.transcript, summary);
    assert_eq!(runtime.get_outcome(&id).unwrap().transcript, summary);
    assert_eq!(runtime.get_metadata(&id).unwrap().transcript, summary);

    let list = runtime.describe_list();
    assert_eq!(list["subagents"][0]["transcript"]["count"], json!(3));
    assert_eq!(
        list["subagents"][0]["transcript"]["last_kind"],
        json!("completion")
    );
    assert_eq!(list["capacity"]["transcript_entries"], json!(3));
    assert_eq!(list["capacity"]["used"], json!(1));
    assert_eq!(list["capacity"]["max"], json!(10));

    let unknown = runtime
        .wait(&["ghost".to_owned()], Duration::from_millis(10))
        .await;
    assert_eq!(unknown[0].transcript, TranscriptSummary::default());
}
