//! Activity-stream behaviour of the in-process sub-agent runtime.
//!
//! These tests drive the real runtime over the fakes of [`super`]: an echoing factory parks an
//! instance `Idle` after its turn (publishing the terminal frame), and [`blocked`] keeps a turn
//! `Running` so a test can record caller-supplied activity mid-turn. Everything is deterministic:
//! frames are read from the bounded ring, and the live feed is observed through bounded awaits.
use std::time::Duration;

use futures::StreamExt;

use super::*;
use crate::subagents::{ActivityEvent, ActivityKind, SubAgentQuota};

/// The `kind` of every retained frame of `id`, oldest first.
fn kinds(runtime: &SubAgentRuntime, id: &str) -> Vec<ActivityKind> {
    runtime
        .activity(id)
        .expect("a known instance")
        .into_iter()
        .map(|frame| frame.kind)
        .collect()
}

#[tokio::test]
async fn spawn_and_completed_turn_publish_lifecycle_frames() {
    let runtime = build_runtime(&["worker"], FakeFactory::new(Behavior::Echo));
    let id = spawn(&runtime, "worker", "hello").await;
    let outcome = wait_one(&runtime, &id).await;
    assert_eq!(outcome.status, SubagentStatus::Completed);

    let frames = runtime.activity(&id).expect("the instance exists");
    assert_eq!(
        frames.iter().map(|frame| frame.kind).collect::<Vec<_>>(),
        vec![
            ActivityKind::Spawn,
            ActivityKind::Running,
            ActivityKind::Idle,
            ActivityKind::Completed
        ]
    );
    assert_eq!(frames[0].summary, "hello");
    assert_eq!(
        frames[0].task_id, "",
        "the spawn frame precedes the first turn"
    );
    assert_eq!(frames[1].task_id, format!("{id}#1"));
    assert_eq!(frames[3].summary, "echo:hello");
    assert_eq!(frames[3].ok, Some(true));

    let summary = runtime.activity_summary(&id).expect("the instance exists");
    assert_eq!(summary.count, 4);
    assert!(summary.terminal);
    assert_eq!(summary.last_kind, Some(ActivityKind::Completed));
    assert_eq!(summary.last_seq, frames[3].seq);
    assert!(frames[3].is_terminal());
}

#[tokio::test]
async fn record_tool_call_publishes_a_tool_frame_and_keeps_the_transcript_in_step() {
    let (gate, factory) = blocked();
    let runtime = build_runtime(&["worker"], factory);
    let id = spawn(&runtime, "worker", "task").await;
    wait_running(&runtime, &id).await;

    let seq = runtime
        .record_tool_call(&id, "bash", "ls -la", Some(true))
        .expect("the instance exists");
    let frames = runtime.activity(&id).expect("the instance exists");
    let tool = frames
        .iter()
        .find(|frame| frame.kind == ActivityKind::ToolCall)
        .expect("a tool frame was published");
    assert_eq!(tool.summary, "ls -la");
    assert_eq!(tool.tool_name.as_deref(), Some("bash"));
    assert_eq!(tool.ok, Some(true));
    assert_eq!(tool.seq, seq);
    assert_eq!(
        runtime.transcript_summary(&id).map(|digest| digest.count),
        Some(3)
    );

    gate.send_replace(true);
    wait_one(&runtime, &id).await;
}

#[tokio::test]
async fn subscriptions_and_streams_deliver_frames_published_after_the_call() {
    let (gate, factory) = blocked();
    let runtime = build_runtime(&["worker"], factory);
    let id = spawn(&runtime, "worker", "task").await;
    wait_running(&runtime, &id).await;

    let mut subscription = runtime
        .subscribe_activity(&id)
        .expect("the instance exists");
    let mut events = runtime.activity_stream(&id).expect("the instance exists");
    gate.send_replace(true);
    wait_one(&runtime, &id).await;

    let mut from_subscription = Vec::new();
    for _ in 0..2 {
        let event = tokio::time::timeout(Duration::from_secs(5), subscription.recv())
            .await
            .expect("an event arrives")
            .expect("the feed stays open");
        if let ActivityEvent::Frame(frame) = event {
            from_subscription.push(frame.kind);
        }
    }
    let mut from_stream = Vec::new();
    for _ in 0..2 {
        let event = tokio::time::timeout(Duration::from_secs(5), events.next())
            .await
            .expect("an event arrives")
            .expect("the stream stays open");
        if let ActivityEvent::Frame(frame) = event {
            from_stream.push(frame.kind);
        }
    }
    let expected = vec![ActivityKind::Idle, ActivityKind::Completed];
    assert_eq!(from_subscription, expected);
    assert_eq!(from_stream, expected);
}

#[tokio::test]
async fn record_activity_reports_the_sequence_and_is_suppressed_once_terminal() {
    let (gate, factory) = blocked();
    let runtime = build_runtime(&["worker"], factory);
    let id = spawn(&runtime, "worker", "task").await;
    wait_running(&runtime, &id).await;

    let seq = runtime
        .record_activity(
            &id,
            ActivityKind::ToolResult,
            "done",
            Some("bash"),
            Some(true),
        )
        .expect("the instance exists")
        .expect("the frame was recorded");
    let frames = runtime.activity(&id).expect("the instance exists");
    let frame = frames
        .iter()
        .find(|frame| frame.kind == ActivityKind::ToolResult)
        .expect("a tool result frame");
    assert_eq!((frame.seq, frame.tool_name.as_deref()), (seq, Some("bash")));

    gate.send_replace(true);
    wait_one(&runtime, &id).await;
    assert!(runtime
        .activity_summary(&id)
        .is_some_and(|digest| digest.terminal));
    assert_eq!(
        runtime
            .record_activity(&id, ActivityKind::Error, "late", None, None)
            .unwrap(),
        None
    );
    assert!(matches!(
        runtime.record_activity("ghost", ActivityKind::Running, "x", None, None),
        Err(Error::Validation(message)) if message == "subagent not found: ghost"
    ));
}

#[tokio::test]
async fn eviction_publishes_the_terminal_evicted_frame() {
    let runtime = build_runtime(&["worker"], FakeFactory::new(Behavior::Echo));
    let id = spawn(&runtime, "worker", "task").await;
    wait_one(&runtime, &id).await;
    let previous = runtime.evict(&id).await.expect("the instance is evictable");
    assert_eq!(previous, SubagentStatus::Completed);

    let frames = runtime.activity(&id).expect("the tombstone exists");
    let last = frames.last().expect("the evicted frame");
    assert_eq!(last.kind, ActivityKind::Evicted);
    assert_eq!(last.summary, EVICTION_REASON);
    assert!(last.is_terminal());
    assert!(runtime
        .activity_summary(&id)
        .is_some_and(|digest| digest.terminal));
    assert_eq!(
        runtime
            .record_activity(&id, ActivityKind::Running, "late", None, None)
            .unwrap(),
        None
    );
}

#[tokio::test]
async fn resume_after_close_clears_the_terminal_flag_and_publishes_idle() {
    let runtime = build_runtime(&["worker"], FakeFactory::new(Behavior::Echo));
    let id = spawn(&runtime, "worker", "task").await;
    wait_one(&runtime, &id).await;
    runtime.close(&id).await.expect("the instance closes");
    let closed = runtime.activity(&id).expect("the tombstone exists");
    let closed_seq = closed.last().expect("the closed frame").seq;
    assert_eq!(
        closed.last().map(|frame| frame.kind),
        Some(ActivityKind::Closed)
    );
    assert!(runtime
        .activity_summary(&id)
        .is_some_and(|digest| digest.terminal));

    let report = runtime.resume(&id).await.expect("the instance resumes");
    assert!(report.restored);
    let frames = runtime.activity(&id).expect("the instance exists");
    let last = frames.last().expect("the idle frame");
    assert_eq!(last.kind, ActivityKind::Idle);
    assert!(last.seq > closed_seq);
    assert!(!runtime
        .activity_summary(&id)
        .is_some_and(|digest| digest.terminal));
}

#[tokio::test]
async fn disabled_activity_stream_records_nothing() {
    let runtime = build_runtime(&["worker"], FakeFactory::new(Behavior::Echo))
        .with_quota(SubAgentQuota::default().with_activity_stream(false));
    let id = spawn(&runtime, "worker", "task").await;
    wait_one(&runtime, &id).await;
    assert_eq!(runtime.activity(&id), Some(Vec::new()));
    assert_eq!(kinds(&runtime, &id), Vec::new());
    let summary = runtime.activity_summary(&id).expect("the instance exists");
    assert_eq!(
        (summary.count, summary.terminal, summary.last_seq),
        (0, false, 0)
    );
    assert_eq!(
        runtime
            .record_activity(&id, ActivityKind::ToolCall, "bash", Some("bash"), None)
            .unwrap(),
        None
    );
    // The transcript is unaffected by the activity switch.
    assert!(runtime
        .transcript_summary(&id)
        .is_some_and(|digest| digest.count > 0));
}

#[tokio::test]
async fn activity_after_paginates_from_a_cursor() {
    let runtime = build_runtime(&["worker"], FakeFactory::new(Behavior::Echo));
    let id = spawn(&runtime, "worker", "task").await;
    wait_one(&runtime, &id).await;
    let first = runtime
        .activity_after(&id, 0, 1)
        .expect("the instance exists");
    assert_eq!(first.len(), 1);
    assert_eq!(first[0].kind, ActivityKind::Spawn);
    let rest = runtime
        .activity_after(&id, first[0].seq, 0)
        .expect("the instance exists");
    assert_eq!(rest.len(), 3);
    assert_eq!(runtime.activity_after("ghost", 0, 0), None);
}
