//! Capture, turn-publication and key-layout tests of the online RL state machine.
//!
//! The shared fixtures (`registry`, `spec`, `delayed`, `TASK`, `SESSION`) come
//! from the parent test module.

use serde_json::json;

use super::{delayed, registry, spec, SESSION, TASK};
use crate::online::capture::{CaptureMeta, CaptureStatus, PublishState};
use crate::online::error::OnlineError;
use crate::online::store::{keys, OnlineStore};
use crate::online::task::FinishReason;

#[tokio::test]
async fn begin_capture_reserves_and_replays() {
    let registry = registry();
    registry.start(&spec(TASK, SESSION)).await.expect("start");
    assert_eq!(
        registry
            .begin_capture(TASK, "c1", None, "req")
            .await
            .expect("begin"),
        None
    );
    let meta = CaptureMeta::from_json(
        &registry
            .store()
            .get(&keys::capture(TASK, "c1"))
            .await
            .unwrap()
            .expect("stored"),
    )
    .expect("decode");
    assert_eq!(meta.status, CaptureStatus::Open);
    assert_eq!(meta.request_fingerprint, "req");
    assert_eq!(meta.agent_turn_id, None);
    assert_eq!(registry.open_capture_count(TASK).await.unwrap(), 1);

    // Exact replay: no error, no new state.
    assert_eq!(
        registry
            .begin_capture(TASK, "c1", None, "req")
            .await
            .expect("replay"),
        None
    );
    assert_eq!(registry.open_capture_count(TASK).await.unwrap(), 1);
}

#[tokio::test]
async fn begin_capture_rejects_other_input_and_closed_tasks() {
    let registry = registry();
    registry.start(&spec(TASK, SESSION)).await.expect("start");
    registry
        .begin_capture(TASK, "c1", None, "req")
        .await
        .expect("begin");
    let error = registry
        .begin_capture(TASK, "c1", None, "other")
        .await
        .expect_err("mismatch");
    assert_eq!(
        error.to_string(),
        "capture_id already belongs to different input: c1"
    );

    let missing = registry
        .begin_capture("nope", "c1", None, "req")
        .await
        .expect_err("missing");
    assert_eq!(missing.code(), "task_not_found");

    registry
        .discard_capture(TASK, "c1", None)
        .await
        .expect("discard");
    registry
        .finish(TASK, FinishReason::UserStopped)
        .await
        .expect("finish");
    let closed = registry
        .begin_capture(TASK, "c2", None, "req")
        .await
        .expect_err("closed");
    assert_eq!(closed.to_string(), "RL Task is not active: task-1");
}

#[tokio::test]
async fn commit_capture_stages_a_sample_once() {
    let registry = registry();
    registry.start(&spec(TASK, SESSION)).await.expect("start");
    registry
        .begin_capture(TASK, "c1", None, "req")
        .await
        .expect("begin");
    let sample = json!({"sample_id": "c1"});
    assert!(registry
        .commit_capture(TASK, "c1", None, "resp", &sample)
        .await
        .expect("commit"));

    let store = registry.store();
    let meta = CaptureMeta::from_json(
        &store
            .get(&keys::capture(TASK, "c1"))
            .await
            .unwrap()
            .expect("meta"),
    )
    .expect("decode");
    assert_eq!(meta.status, CaptureStatus::Committed);
    assert_eq!(meta.response_fingerprint.as_deref(), Some("resp"));
    assert_eq!(
        store.set_members(&keys::open_captures(TASK)).await.unwrap(),
        Vec::<String>::new()
    );
    assert_eq!(
        store
            .set_members(&keys::committed_captures(TASK))
            .await
            .unwrap(),
        vec!["c1".to_owned()]
    );
    assert_eq!(
        store
            .get(&keys::pending_sample(TASK, "c1"))
            .await
            .unwrap()
            .as_deref(),
        Some(sample.to_string().as_str())
    );
    assert_eq!(
        registry.pending_sample_ids(TASK).await.unwrap(),
        vec!["c1".to_owned()]
    );

    // Replay returns false; a different response conflicts.
    assert!(!registry
        .commit_capture(TASK, "c1", None, "resp", &sample)
        .await
        .expect("replay"));
    let error = registry
        .commit_capture(TASK, "c1", None, "other", &sample)
        .await
        .expect_err("different response");
    assert_eq!(
        error.to_string(),
        "capture_id already has a different response: c1"
    );
}

#[tokio::test]
async fn commit_capture_requires_a_reservation() {
    let registry = registry();
    registry.start(&spec(TASK, SESSION)).await.expect("start");
    let error = registry
        .commit_capture(TASK, "ghost", None, "resp", &json!({}))
        .await
        .expect_err("no reservation");
    assert_eq!(error.to_string(), "capture has no matching before: ghost");

    registry
        .begin_capture(TASK, "c1", Some("turn-1"), "req")
        .await
        .expect("begin");
    let mismatch = registry
        .commit_capture(TASK, "c1", Some("turn-2"), "resp", &json!({}))
        .await
        .expect_err("turn mismatch");
    assert_eq!(mismatch.to_string(), "capture turn mismatch: c1");

    registry
        .discard_capture(TASK, "c1", Some("turn-1"))
        .await
        .expect("discard");
    registry
        .finish(TASK, FinishReason::UserStopped)
        .await
        .expect("finish");
    let closed = registry
        .commit_capture(TASK, "c1", Some("turn-1"), "resp", &json!({}))
        .await
        .expect_err("closed task");
    assert_eq!(closed.to_string(), "capture has no matching before: c1");

    // A commit racing the finalize still sees the inactive task guard.
    let racing = CaptureMeta {
        capture_id: "c9".to_owned(),
        agent_turn_id: Some("turn-1".to_owned()),
        request_fingerprint: "req".to_owned(),
        response_fingerprint: None,
        status: CaptureStatus::Open,
    };
    registry
        .store()
        .set(
            &keys::capture(TASK, "c9"),
            &racing.to_json().expect("encode"),
        )
        .await
        .unwrap();
    let inactive = registry
        .commit_capture(TASK, "c9", Some("turn-1"), "resp", &json!({}))
        .await
        .expect_err("inactive task");
    assert_eq!(inactive.to_string(), "RL Task is not active: task-1");
}

#[tokio::test]
async fn discard_capture_releases_the_reservation() {
    let registry = registry();
    registry.start(&spec(TASK, SESSION)).await.expect("start");
    registry
        .begin_capture(TASK, "c1", None, "req")
        .await
        .expect("begin");
    registry
        .discard_capture(TASK, "c1", None)
        .await
        .expect("discard");
    assert_eq!(registry.open_capture_count(TASK).await.unwrap(), 0);
    assert_eq!(
        registry
            .store()
            .get(&keys::capture(TASK, "c1"))
            .await
            .unwrap(),
        None
    );
    assert_eq!(
        registry
            .store()
            .set_members(&keys::captures(TASK))
            .await
            .unwrap(),
        Vec::<String>::new()
    );
    // Missing capture and missing turn both stay silent.
    registry
        .discard_capture(TASK, "c1", None)
        .await
        .expect("idempotent");
    registry
        .discard_capture(TASK, "c2", None)
        .await
        .expect("missing");

    registry
        .begin_capture(TASK, "c3", None, "req")
        .await
        .expect("begin");
    registry
        .commit_capture(TASK, "c3", None, "resp", &json!({}))
        .await
        .expect("commit");
    let error = registry
        .discard_capture(TASK, "c3", None)
        .await
        .expect_err("committed");
    assert_eq!(error.to_string(), "capture is already committed: c3");

    registry
        .finish(TASK, FinishReason::UserStopped)
        .await
        .expect("finish");
    registry
        .begin_capture(TASK, "c4", None, "req")
        .await
        .expect_err("closed");
}

#[tokio::test]
async fn discard_after_abort_is_silent() {
    let registry = registry();
    registry.start(&spec(TASK, SESSION)).await.expect("start");
    registry
        .begin_capture(TASK, "c1", None, "req")
        .await
        .expect("begin");
    registry
        .abort(TASK, FinishReason::CaptureFailed)
        .await
        .expect("abort");
    registry
        .discard_capture(TASK, "c1", None)
        .await
        .expect("silent");
}

// ---------------------------------------------------------------- delayed turns

#[tokio::test]
async fn delayed_capture_opens_the_turn() {
    let registry = registry();
    registry
        .start(&delayed(TASK, SESSION))
        .await
        .expect("start");
    let transition = registry
        .begin_capture(TASK, "c1", Some("turn-1"), "req")
        .await
        .expect("begin");
    assert_eq!(transition, None);
    assert_eq!(
        registry
            .store()
            .get(&keys::current_turn(TASK))
            .await
            .unwrap()
            .as_deref(),
        Some("turn-1")
    );
}

#[tokio::test]
async fn delayed_capture_requests_a_turn_transition() {
    let registry = registry();
    registry
        .start(&delayed(TASK, SESSION))
        .await
        .expect("start");
    registry
        .begin_capture(TASK, "c1", Some("turn-1"), "req")
        .await
        .expect("begin");
    let transition = registry
        .begin_capture(TASK, "c2", Some("turn-2"), "req")
        .await
        .expect("new turn");
    let transition = transition.expect("transition");
    assert_eq!(transition.previous, "turn-1");
    assert_eq!(transition.next, "turn-2");
    assert_eq!(
        registry
            .store()
            .get(&keys::turn_transition(TASK))
            .await
            .unwrap()
            .as_deref(),
        Some("turn-2")
    );
    // The parked capture is not created.
    assert_eq!(
        registry
            .store()
            .get(&keys::capture(TASK, "c2"))
            .await
            .unwrap(),
        None
    );

    // Re-requesting the same parked turn yields the same transition.
    assert_eq!(
        registry
            .begin_capture(TASK, "c2", Some("turn-2"), "req")
            .await
            .expect("replay"),
        Some(transition)
    );
}

#[tokio::test]
async fn delayed_capture_conflicts_with_a_waiting_turn() {
    let registry = registry();
    registry
        .start(&delayed(TASK, SESSION))
        .await
        .expect("start");
    registry
        .begin_capture(TASK, "c1", Some("turn-1"), "req")
        .await
        .expect("begin");
    registry
        .begin_capture(TASK, "c2", Some("turn-2"), "req")
        .await
        .expect("park");
    let error = registry
        .begin_capture(TASK, "c3", Some("turn-3"), "req")
        .await
        .expect_err("third turn");
    assert_eq!(
        error.to_string(),
        "turn turn-2 is already waiting for turn-1"
    );
    assert_eq!(error.code(), "task_conflict");
}

#[tokio::test]
async fn closed_and_closing_turns_refuse_captures() {
    let registry = registry();
    registry
        .start(&delayed(TASK, SESSION))
        .await
        .expect("start");
    registry
        .begin_capture(TASK, "c1", Some("turn-1"), "req")
        .await
        .expect("begin");
    registry
        .commit_capture(TASK, "c1", Some("turn-1"), "resp", &json!({}))
        .await
        .expect("commit");
    assert_eq!(
        registry
            .claim_turn_publish(TASK, "turn-1")
            .await
            .expect("claim"),
        PublishState::Claimed
    );
    let closing = registry
        .begin_capture(TASK, "c2", Some("turn-1"), "req")
        .await
        .expect_err("closing");
    assert!(matches!(closing, OnlineError::TurnClosed(_)));
    assert_eq!(closing.to_string(), "turn is already closing: turn-1");
    assert_eq!(closing.code(), "turn_closed");

    registry
        .complete_turn_publish(TASK, "turn-1")
        .await
        .expect("publish");
    let closed = registry
        .begin_capture(TASK, "c3", Some("turn-1"), "req")
        .await
        .expect_err("closed");
    assert_eq!(closed.to_string(), "turn is already closed: turn-1");

    // A turn whose successor is parked is closing as well: park turn-2, advance
    // to it, open a capture there, then park turn-3 behind it.
    let transition = registry
        .begin_capture(TASK, "c4", Some("turn-2"), "req")
        .await
        .expect("park")
        .expect("transition");
    assert_eq!(transition.previous, "turn-1");
    assert_eq!(transition.next, "turn-2");
    registry
        .advance_turn(TASK, "turn-1", "turn-2")
        .await
        .expect("advance");
    registry
        .begin_capture(TASK, "c4", Some("turn-2"), "req")
        .await
        .expect("begin");
    registry
        .begin_capture(TASK, "c5", Some("turn-3"), "req")
        .await
        .expect("park");
    let stalled = registry
        .begin_capture(TASK, "c6", Some("turn-2"), "req")
        .await
        .expect_err("closing");
    assert_eq!(stalled.to_string(), "turn is closing: turn-2");
}

#[tokio::test]
async fn turn_publish_waits_for_open_captures() {
    let registry = registry();
    registry
        .start(&delayed(TASK, SESSION))
        .await
        .expect("start");
    registry
        .begin_capture(TASK, "c1", Some("turn-1"), "req")
        .await
        .expect("begin");
    assert_eq!(
        registry
            .claim_turn_publish(TASK, "turn-1")
            .await
            .expect("claim"),
        PublishState::Waiting
    );
    assert_eq!(
        registry
            .store()
            .get(&keys::turn_publish(TASK, "turn-1"))
            .await
            .unwrap(),
        None
    );
}

#[tokio::test]
async fn turn_publish_lifecycle_closes_the_turn() {
    let registry = registry();
    registry
        .start(&delayed(TASK, SESSION))
        .await
        .expect("start");
    registry
        .begin_capture(TASK, "c1", Some("turn-1"), "req")
        .await
        .expect("begin");
    registry
        .commit_capture(TASK, "c1", Some("turn-1"), "resp", &json!({"id": "c1"}))
        .await
        .expect("commit");
    assert_eq!(
        registry.pending_sample_ids(TASK).await.unwrap(),
        vec!["c1".to_owned()]
    );

    assert_eq!(
        registry
            .claim_turn_publish(TASK, "turn-1")
            .await
            .expect("claim"),
        PublishState::Claimed
    );
    assert_eq!(
        registry
            .claim_turn_publish(TASK, "turn-1")
            .await
            .expect("second claim"),
        PublishState::Publishing
    );
    registry
        .complete_turn_publish(TASK, "turn-1")
        .await
        .expect("publish");

    assert_eq!(
        registry
            .claim_turn_publish(TASK, "turn-1")
            .await
            .expect("published"),
        PublishState::Published
    );
    assert_eq!(
        registry
            .store()
            .set_members(&keys::closed_turns(TASK))
            .await
            .unwrap(),
        vec!["turn-1".to_owned()]
    );
    assert_eq!(
        registry.pending_sample_ids(TASK).await.unwrap(),
        Vec::<String>::new()
    );
    let again = registry
        .complete_turn_publish(TASK, "turn-1")
        .await
        .expect_err("already published");
    assert_eq!(again.to_string(), "turn is not publishing: turn-1");
}

#[tokio::test]
async fn turn_publish_terminal_states() {
    let registry = registry();
    registry
        .start(&delayed(TASK, SESSION))
        .await
        .expect("start");
    registry
        .fail_turn_publish(TASK, "turn-1")
        .await
        .expect("fail");
    assert_eq!(
        registry
            .claim_turn_publish(TASK, "turn-1")
            .await
            .expect("claim"),
        PublishState::Failed
    );
    let error = registry
        .complete_turn_publish(TASK, "turn-1")
        .await
        .expect_err("failed turn");
    assert_eq!(error.to_string(), "turn is not publishing: turn-1");

    // `claim` guards captures and the publish key only, exactly like Python: a
    // turn nobody captured is claimed and later fails its own validation.
    let unclaimed = registry
        .claim_turn_publish("nope", "turn-1")
        .await
        .expect("claim has no task guard");
    assert_eq!(unclaimed, PublishState::Claimed);
}

#[tokio::test]
async fn publish_state_reports_unknown_store_values() {
    let registry = registry();
    registry
        .start(&delayed(TASK, SESSION))
        .await
        .expect("start");
    registry
        .store()
        .set(&keys::turn_publish(TASK, "turn-1"), "bogus")
        .await
        .unwrap();
    let error = registry
        .claim_turn_publish(TASK, "turn-1")
        .await
        .expect_err("unknown state");
    assert_eq!(
        error.to_string(),
        "turn turn-1 has an unknown publish state: bogus"
    );
}

#[tokio::test]
async fn advance_turn_moves_the_current_turn() {
    let registry = registry();
    registry
        .start(&delayed(TASK, SESSION))
        .await
        .expect("start");
    registry
        .begin_capture(TASK, "c1", Some("turn-1"), "req")
        .await
        .expect("begin");
    registry
        .commit_capture(TASK, "c1", Some("turn-1"), "resp", &json!({}))
        .await
        .expect("commit");
    registry
        .begin_capture(TASK, "c2", Some("turn-2"), "req")
        .await
        .expect("park");

    let unpublished = registry
        .advance_turn(TASK, "turn-1", "turn-2")
        .await
        .expect_err("not published");
    assert_eq!(unpublished.to_string(), "turn is not published: turn-1");

    registry
        .claim_turn_publish(TASK, "turn-1")
        .await
        .expect("claim");
    registry
        .complete_turn_publish(TASK, "turn-1")
        .await
        .expect("publish");
    registry
        .advance_turn(TASK, "turn-1", "turn-2")
        .await
        .expect("advance");
    assert_eq!(
        registry
            .store()
            .get(&keys::current_turn(TASK))
            .await
            .unwrap()
            .as_deref(),
        Some("turn-2")
    );
    assert_eq!(
        registry
            .store()
            .get(&keys::turn_transition(TASK))
            .await
            .unwrap(),
        None
    );
    // Idempotent once the successor is current.
    registry
        .advance_turn(TASK, "turn-1", "turn-2")
        .await
        .expect("replay");
}

#[tokio::test]
async fn advance_turn_guards_the_handover() {
    let registry = registry();
    registry
        .start(&delayed(TASK, SESSION))
        .await
        .expect("start");
    registry
        .begin_capture(TASK, "c1", Some("turn-1"), "req")
        .await
        .expect("begin");
    registry
        .store()
        .set(&keys::turn_publish(TASK, "turn-9"), "published")
        .await
        .unwrap();
    let changed = registry
        .advance_turn(TASK, "turn-9", "turn-2")
        .await
        .expect_err("current turn changed");
    assert_eq!(
        changed.to_string(),
        "current turn changed during transition"
    );

    registry
        .store()
        .set(&keys::turn_publish(TASK, "turn-1"), "published")
        .await
        .unwrap();
    let unexpected = registry
        .advance_turn(TASK, "turn-1", "turn-2")
        .await
        .expect_err("no parked turn");
    assert_eq!(
        unexpected.to_string(),
        "next turn changed during transition"
    );
}

// ---------------------------------------------------------------- key layout

#[tokio::test]
async fn written_keys_are_the_python_layout() {
    let registry = registry();
    registry
        .start(&delayed(TASK, SESSION))
        .await
        .expect("start");
    registry
        .begin_capture(TASK, "c1", Some("turn-1"), "req")
        .await
        .expect("begin");
    registry
        .commit_capture(TASK, "c1", Some("turn-1"), "resp", &json!({}))
        .await
        .expect("commit");
    registry
        .claim_turn_publish(TASK, "turn-1")
        .await
        .expect("claim");
    registry
        .complete_turn_publish(TASK, "turn-1")
        .await
        .expect("publish");

    assert_eq!(
        registry.store().keys().await,
        vec![
            "rl:v1:active_session:sess-1".to_owned(),
            "rl:v1:task:task-1".to_owned(),
            "rl:v1:task:task-1:capture:c1".to_owned(),
            "rl:v1:task:task-1:captures".to_owned(),
            "rl:v1:task:task-1:captures:committed".to_owned(),
            "rl:v1:task:task-1:turn:current".to_owned(),
            "rl:v1:task:task-1:turn:turn-1:captures".to_owned(),
            "rl:v1:task:task-1:turn:turn-1:publish".to_owned(),
            "rl:v1:task:task-1:turns:closed".to_owned(),
            "rl:v1:tasks:active".to_owned(),
        ]
    );
}
