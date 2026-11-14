//! Unit tests of the online RL task state machine.
//!
//! Every test runs against [`MemoryOnlineStore`] with a [`FixedClock`], so the
//! whole suite is offline, deterministic, and one test per behaviour of the
//! transition tables documented in [`super::task`] / [`super::capture`].
//!
//! This module owns the store, spec, task lifecycle and reward tests; the
//! capture and turn tests live in the `capture` child module.

use std::sync::Arc;

use serde_json::json;

use super::capture::{CaptureStatus, PublishState};
use super::error::OnlineError;
use super::store::{keys, MemoryOnlineStore, OnlineStore};
use super::task::{
    FinishReason, FixedClock, RewardMode, TaskRecord, TaskRegistry, TaskSpec, TaskStatus,
};

mod capture;

const STAMP: &str = "2026-01-01T00:00:00+00:00";
const SESSION: &str = "sess-1";
const TASK: &str = "task-1";

fn spec(task_id: &str, session: &str) -> TaskSpec {
    TaskSpec {
        rl_task_id: task_id.to_owned(),
        agent_session_id: session.to_owned(),
        model_id: "model-a".to_owned(),
        policy_lora_name: "policy-a".to_owned(),
        reward_mode: RewardMode::Terminal,
        policy_model: None,
    }
}

fn delayed(task_id: &str, session: &str) -> TaskSpec {
    TaskSpec {
        reward_mode: RewardMode::DelayedFeedback,
        ..spec(task_id, session)
    }
}

fn registry() -> TaskRegistry<MemoryOnlineStore> {
    TaskRegistry::with_clock(MemoryOnlineStore::new(), Arc::new(FixedClock::new(STAMP)))
}

async fn finalized_with_samples(registry: &TaskRegistry<MemoryOnlineStore>, capture_ids: &[&str]) {
    registry.start(&spec(TASK, SESSION)).await.expect("start");
    for capture_id in capture_ids {
        registry
            .begin_capture(TASK, capture_id, None, "request")
            .await
            .expect("begin");
        registry
            .commit_capture(
                TASK,
                capture_id,
                None,
                "response",
                &json!({ "id": capture_id }),
            )
            .await
            .expect("commit");
    }
    registry
        .finish(TASK, FinishReason::UserStopped)
        .await
        .expect("finish");
}

// ---------------------------------------------------------------- errors

#[test]
fn error_codes_and_statuses_match_the_service_envelope() {
    let cases = [
        (
            OnlineError::TaskNotFound("x".to_owned()),
            "task_not_found",
            404,
        ),
        (
            OnlineError::TaskConflict("x".to_owned()),
            "task_conflict",
            409,
        ),
        (OnlineError::TurnClosed("x".to_owned()), "turn_closed", 409),
        (
            OnlineError::InvalidRequest("x".to_owned()),
            "invalid_request",
            400,
        ),
        (
            OnlineError::InsufficientSamples { have: 1, need: 8 },
            "insufficient_samples",
            409,
        ),
        (
            OnlineError::Store(openjiuwen_common::error::Error::workflow("x")),
            "internal_error",
            500,
        ),
    ];
    for (error, code, status) in cases {
        assert_eq!(error.code(), code);
        assert_eq!(error.http_status(), status);
    }
}

#[test]
fn error_display_is_the_python_message() {
    assert_eq!(
        OnlineError::TaskConflict("rl_task_id already exists: t".to_owned()).to_string(),
        "rl_task_id already exists: t"
    );
    assert_eq!(
        OnlineError::InsufficientSamples { have: 2, need: 8 }.to_string(),
        "insufficient samples: have 2, need 8"
    );
}

#[test]
fn errors_convert_into_kernel_errors() {
    let not_found: openjiuwen_common::error::Error =
        OnlineError::TaskNotFound("gone".to_owned()).into();
    assert!(matches!(
        not_found,
        openjiuwen_common::error::Error::NotFound(ref message) if message == "gone"
    ));
    let invalid: openjiuwen_common::error::Error =
        OnlineError::InvalidRequest("bad".to_owned()).into();
    assert!(matches!(
        invalid,
        openjiuwen_common::error::Error::Validation(ref message) if message == "bad"
    ));
    let conflict: openjiuwen_common::error::Error =
        OnlineError::TurnClosed("closed".to_owned()).into();
    assert!(matches!(
        conflict,
        openjiuwen_common::error::Error::Workflow(ref message) if message == "closed"
    ));
    let stored: openjiuwen_common::error::Error =
        OnlineError::Store(openjiuwen_common::error::Error::timeout("slow")).into();
    assert!(matches!(
        stored,
        openjiuwen_common::error::Error::Timeout(_)
    ));
}

// ---------------------------------------------------------------- store

#[tokio::test]
async fn memory_store_primitives() {
    let store = MemoryOnlineStore::new();
    assert_eq!(store.get("k").await.unwrap(), None);
    store.set("k", "v").await.unwrap();
    assert_eq!(store.get("k").await.unwrap().as_deref(), Some("v"));
    assert!(!store.set_if_absent("k", "other").await.unwrap());
    assert!(store.set_if_absent("fresh", "v").await.unwrap());
    assert_eq!(store.get("fresh").await.unwrap().as_deref(), Some("v"));

    assert!(store.set_add("s", "b").await.unwrap());
    assert!(store.set_add("s", "a").await.unwrap());
    assert!(!store.set_add("s", "a").await.unwrap());
    assert_eq!(store.set_members("s").await.unwrap(), vec!["a", "b"]);
    assert!(store.set_contains("s", "b").await.unwrap());
    assert!(!store.set_contains("s", "z").await.unwrap());
    assert!(store.set_remove("s", "b").await.unwrap());
    assert!(!store.set_remove("s", "b").await.unwrap());
    assert_eq!(store.set_members("s").await.unwrap(), vec!["a"]);

    assert!(store.del("k").await.unwrap());
    assert!(!store.del("k").await.unwrap());
    assert!(store.del("s").await.unwrap());
    assert_eq!(store.set_members("s").await.unwrap(), Vec::<String>::new());
}

#[tokio::test]
async fn memory_store_helpers_report_keys_and_values() {
    let store = MemoryOnlineStore::new();
    store.set("b", "1").await.unwrap();
    store.set("a", "2").await.unwrap();
    store.set_add("c", "member").await.unwrap();
    assert_eq!(store.keys().await, vec!["a", "b", "c"]);
    let snapshot = store.snapshot().await;
    assert_eq!(snapshot.get("a").map(String::as_str), Some("2"));
    assert_eq!(snapshot.len(), 2);
}

// ---------------------------------------------------------------- specs

#[test]
fn spec_requires_every_identity() {
    for field in [
        "rl_task_id",
        "agent_session_id",
        "model_id",
        "policy_lora_name",
    ] {
        let mut broken = spec(TASK, SESSION);
        match field {
            "rl_task_id" => broken.rl_task_id = "  ".to_owned(),
            "agent_session_id" => broken.agent_session_id = String::new(),
            "model_id" => broken.model_id = "\t".to_owned(),
            _ => broken.policy_lora_name = String::new(),
        }
        let error = broken.validate().expect_err("blank identity");
        assert_eq!(error.code(), "invalid_request");
        assert_eq!(error.to_string(), format!("{field} is required"));
    }
}

#[test]
fn spec_restricts_the_task_id_shape() {
    assert!(spec("abc_123-XY", SESSION).validate().is_ok());
    for bad in ["a b", "a/b", "", "task.1", "ünicode"] {
        let error = spec(bad, SESSION).validate().expect_err("bad id");
        assert_eq!(error.code(), "invalid_request");
    }
    let long = "a".repeat(65);
    assert!(spec(&long, SESSION).validate().is_err());
    assert!(spec(&"a".repeat(64), SESSION).validate().is_ok());
}

#[test]
fn spec_policy_model_defaults_to_the_lora_name() {
    let mut subject = spec(TASK, SESSION);
    assert_eq!(subject.policy_model(), "policy-a");
    subject.policy_model = Some("served-a".to_owned());
    assert_eq!(subject.policy_model(), "served-a");
    subject.policy_model = Some("   ".to_owned());
    assert_eq!(
        subject
            .validate()
            .expect_err("blank policy model")
            .to_string(),
        "policy_model is required"
    );
}

#[test]
fn task_record_round_trips_and_tolerates_unknown_fields() {
    let record = TaskRecord {
        rl_task_id: TASK.to_owned(),
        agent_session_id: SESSION.to_owned(),
        model_id: "model-a".to_owned(),
        policy_lora_name: "policy-a".to_owned(),
        reward_mode: RewardMode::DelayedFeedback,
        status: TaskStatus::Finalized,
        created_at: STAMP.to_owned(),
        finished_at: Some(STAMP.to_owned()),
        finish_reason: Some(FinishReason::Timeout),
        policy_model: "served-a".to_owned(),
    };
    let payload = record.to_json().expect("encode");
    assert_eq!(TaskRecord::from_json(&payload).expect("decode"), record);

    let tolerated = json!({
        "rl_task_id": TASK,
        "agent_session_id": SESSION,
        "model_id": "model-a",
        "policy_lora_name": "policy-a",
        "reward_mode": "delayed_feedback",
        "status": "active",
        "created_at": STAMP,
        "future_field": {"ignored": true},
    });
    let decoded = TaskRecord::from_json(&tolerated.to_string()).expect("tolerant decode");
    assert_eq!(decoded.policy_model, "policy-a");
    assert_eq!(decoded.finished_at, None);
    assert_eq!(decoded.finish_reason, None);
}

#[test]
fn wire_literals_match_python() {
    assert_eq!(
        serde_json::to_string(&RewardMode::DelayedFeedback).unwrap(),
        "\"delayed_feedback\""
    );
    assert_eq!(
        serde_json::to_string(&TaskStatus::Finalized).unwrap(),
        "\"finalized\""
    );
    assert_eq!(
        serde_json::to_string(&FinishReason::ServiceRestarted).unwrap(),
        "\"service_restarted\""
    );
    assert_eq!(
        serde_json::to_string(&CaptureStatus::Committed).unwrap(),
        "\"committed\""
    );
    assert_eq!(
        serde_json::to_string(&PublishState::Published).unwrap(),
        "\"published\""
    );
}

// ---------------------------------------------------------------- start

#[tokio::test]
async fn start_creates_the_task_and_its_indexes() {
    let registry = registry();
    let (record, created) = registry.start(&spec(TASK, SESSION)).await.expect("start");
    assert!(created);
    assert_eq!(record.status, TaskStatus::Active);
    assert_eq!(record.created_at, STAMP);
    assert_eq!(record.policy_model, "policy-a");
    assert_eq!(record.finished_at, None);
    assert_eq!(registry.get(TASK).await.unwrap(), record);

    let store = registry.store();
    assert_eq!(
        store
            .get(&keys::active_session(SESSION))
            .await
            .unwrap()
            .as_deref(),
        Some(TASK)
    );
    assert_eq!(
        store.set_members(&keys::active_tasks()).await.unwrap(),
        vec![TASK.to_owned()]
    );
    assert_eq!(
        registry.active_task_for_session(SESSION).await.unwrap(),
        Some(record)
    );
}

#[tokio::test]
async fn start_replays_the_exact_spec() {
    let registry = registry();
    let (first, created) = registry.start(&spec(TASK, SESSION)).await.expect("start");
    assert!(created);
    let (second, created) = registry.start(&spec(TASK, SESSION)).await.expect("replay");
    assert!(!created);
    assert_eq!(first, second);
}

#[tokio::test]
async fn start_rejects_a_different_spec_for_the_same_id() {
    let registry = registry();
    registry.start(&spec(TASK, SESSION)).await.expect("start");
    let mut other = spec(TASK, SESSION);
    other.model_id = "model-b".to_owned();
    let error = registry.start(&other).await.expect_err("conflict");
    assert!(matches!(error, OnlineError::TaskConflict(_)));
    assert_eq!(error.to_string(), "rl_task_id already exists: task-1");
    assert_eq!(registry.get(TASK).await.unwrap().model_id, "model-a");
}

#[tokio::test]
async fn start_adopts_the_sessions_active_task() {
    let registry = registry();
    registry.start(&spec(TASK, SESSION)).await.expect("start");
    let (adopted, created) = registry
        .start(&spec("task-2", SESSION))
        .await
        .expect("adopt");
    assert!(!created);
    assert_eq!(adopted.rl_task_id, TASK);
    assert!(registry.get("task-2").await.is_err());
}

#[tokio::test]
async fn start_creates_a_new_task_after_the_previous_one_ended() {
    let registry = registry();
    registry.start(&spec(TASK, SESSION)).await.expect("start");
    registry
        .finish(TASK, FinishReason::UserStopped)
        .await
        .expect("finish");
    let (record, created) = registry
        .start(&spec("task-2", SESSION))
        .await
        .expect("start");
    assert!(created);
    assert_eq!(record.rl_task_id, "task-2");
    assert_eq!(
        registry
            .store()
            .get(&keys::active_session(SESSION))
            .await
            .unwrap()
            .as_deref(),
        Some("task-2")
    );
}

#[tokio::test]
async fn start_ignores_a_dangling_active_pointer() {
    let registry = registry();
    registry
        .store()
        .set(&keys::active_session(SESSION), TASK)
        .await
        .unwrap();
    let (record, created) = registry.start(&spec(TASK, SESSION)).await.expect("start");
    assert!(created);
    assert_eq!(record.rl_task_id, TASK);
}

#[tokio::test]
async fn start_validates_the_spec() {
    let registry = registry();
    let error = registry
        .start(&spec("", SESSION))
        .await
        .expect_err("invalid");
    assert_eq!(error.code(), "invalid_request");
}

#[tokio::test]
async fn get_reports_missing_tasks() {
    let registry = registry();
    let error = registry.get("nope").await.expect_err("missing");
    assert!(matches!(error, OnlineError::TaskNotFound(_)));
    assert_eq!(error.to_string(), "unknown RL Task: nope");
    assert_eq!(error.http_status(), 404);
}

// ---------------------------------------------------------------- finish / abort

#[tokio::test]
async fn finish_closes_the_task_and_clears_the_indexes() {
    let registry = registry();
    registry.start(&spec(TASK, SESSION)).await.expect("start");
    let finished = registry
        .finish(TASK, FinishReason::UserStopped)
        .await
        .expect("finish");
    assert_eq!(finished.status, TaskStatus::Finalized);
    assert_eq!(finished.finished_at.as_deref(), Some(STAMP));
    assert_eq!(finished.finish_reason, Some(FinishReason::UserStopped));
    assert_eq!(registry.get(TASK).await.unwrap(), finished);
    let store = registry.store();
    assert_eq!(
        store.get(&keys::active_session(SESSION)).await.unwrap(),
        None
    );
    assert_eq!(
        store.set_members(&keys::active_tasks()).await.unwrap(),
        Vec::<String>::new()
    );
    assert_eq!(
        registry.active_task_for_session(SESSION).await.unwrap(),
        None
    );
}

#[tokio::test]
async fn finish_keeps_the_first_reason() {
    let registry =
        TaskRegistry::with_clock(MemoryOnlineStore::new(), Arc::new(FixedClock::new("later")));
    registry.start(&spec(TASK, SESSION)).await.expect("start");
    let first = registry
        .finish(TASK, FinishReason::UserStopped)
        .await
        .expect("finish");
    let again = registry
        .finish(TASK, FinishReason::Timeout)
        .await
        .expect("replay");
    assert_eq!(first, again);
    assert_eq!(again.finished_at.as_deref(), Some("later"));
    let aborted = registry
        .abort(TASK, FinishReason::CaptureFailed)
        .await
        .expect("abort after finish");
    assert_eq!(aborted.status, TaskStatus::Finalized);
    assert_eq!(aborted.finish_reason, Some(FinishReason::UserStopped));
}

#[tokio::test]
async fn finish_refuses_open_captures() {
    let registry = registry();
    registry.start(&spec(TASK, SESSION)).await.expect("start");
    registry
        .begin_capture(TASK, "c1", None, "request")
        .await
        .expect("begin");
    let error = registry
        .finish(TASK, FinishReason::UserStopped)
        .await
        .expect_err("open capture");
    assert!(matches!(error, OnlineError::TaskConflict(_)));
    assert_eq!(error.to_string(), "RL Task still has open captures");
    assert_eq!(registry.get(TASK).await.unwrap().status, TaskStatus::Active);
}

#[tokio::test]
async fn abort_drops_open_captures_and_unpublished_samples() {
    let registry = registry();
    registry.start(&spec(TASK, SESSION)).await.expect("start");
    registry
        .begin_capture(TASK, "c1", None, "r1")
        .await
        .expect("begin");
    registry
        .commit_capture(TASK, "c1", None, "response", &json!({"id": "c1"}))
        .await
        .expect("commit");
    registry
        .begin_capture(TASK, "c2", None, "r2")
        .await
        .expect("begin");

    let aborted = registry
        .abort(TASK, FinishReason::CaptureFailed)
        .await
        .expect("abort");
    assert_eq!(aborted.status, TaskStatus::Aborted);
    assert_eq!(aborted.finish_reason, Some(FinishReason::CaptureFailed));

    let store = registry.store();
    assert_eq!(store.get(&keys::capture(TASK, "c2")).await.unwrap(), None);
    assert_eq!(
        store.set_members(&keys::open_captures(TASK)).await.unwrap(),
        Vec::<String>::new()
    );
    assert_eq!(
        store.set_members(&keys::captures(TASK)).await.unwrap(),
        vec!["c1".to_owned()]
    );
    assert_eq!(
        store.get(&keys::pending_sample(TASK, "c1")).await.unwrap(),
        None
    );
    assert_eq!(
        store.get(&keys::active_session(SESSION)).await.unwrap(),
        None
    );
    assert_eq!(
        store.set_members(&keys::active_tasks()).await.unwrap(),
        Vec::<String>::new()
    );
    // Aborting again keeps the first outcome and the first reason.
    let replayed = registry
        .abort(TASK, FinishReason::Timeout)
        .await
        .expect("replay");
    assert_eq!(replayed, aborted);
}

// ---------------------------------------------------------------- rewards

#[tokio::test]
async fn record_reward_validates_the_range() {
    let registry = registry();
    finalized_with_samples(&registry, &["c1"]).await;
    for bogus in [-0.1, 1.1, f64::NAN, f64::INFINITY] {
        let error = registry
            .record_reward(TASK, bogus)
            .await
            .expect_err("out of range");
        assert_eq!(error.code(), "invalid_request");
    }
    assert_eq!(
        registry.pending_sample_ids(TASK).await.unwrap(),
        vec!["c1".to_owned()]
    );
}

#[tokio::test]
async fn record_reward_requires_a_finalized_terminal_task() {
    let registry = registry();
    registry.start(&spec(TASK, SESSION)).await.expect("start");
    let active = registry
        .record_reward(TASK, 0.5)
        .await
        .expect_err("active task");
    assert_eq!(
        active.to_string(),
        "terminal reward requires a finalized Task"
    );

    let delayed_task =
        TaskRegistry::with_clock(MemoryOnlineStore::new(), Arc::new(FixedClock::new(STAMP)));
    delayed_task
        .start(&delayed("d1", SESSION))
        .await
        .expect("start");
    delayed_task
        .finish("d1", FinishReason::UserStopped)
        .await
        .expect("finish");
    let wrong_mode = delayed_task
        .record_reward("d1", 0.5)
        .await
        .expect_err("mode");
    assert_eq!(
        wrong_mode.to_string(),
        "terminal reward is not valid for delayed_feedback Task"
    );
}

#[tokio::test]
async fn record_reward_needs_a_capture() {
    let registry = registry();
    registry.start(&spec(TASK, SESSION)).await.expect("start");
    registry
        .finish(TASK, FinishReason::UserStopped)
        .await
        .expect("finish");
    let error = registry
        .record_reward(TASK, 0.5)
        .await
        .expect_err("no samples");
    assert_eq!(
        error.to_string(),
        "terminal reward requires at least one capture"
    );
}

#[tokio::test]
async fn record_reward_projects_once_and_is_idempotent() {
    let registry = registry();
    finalized_with_samples(&registry, &["c1", "c2"]).await;
    assert_eq!(registry.record_reward(TASK, 0.25).await.expect("reward"), 2);

    let reward_key = keys::reward(TASK);
    let stored = registry
        .store()
        .get(&reward_key)
        .await
        .unwrap()
        .expect("record");
    assert_eq!(
        stored,
        json!({"reward": 0.25, "status": "published", "sample_count": 2}).to_string()
    );
    assert_eq!(
        registry.pending_sample_ids(TASK).await.unwrap(),
        Vec::<String>::new()
    );
    assert_eq!(
        registry
            .store()
            .get(&keys::pending_sample(TASK, "c1"))
            .await
            .unwrap(),
        None
    );
    // Replaying the same value neither errors nor re-projects.
    assert_eq!(registry.record_reward(TASK, 0.25).await.expect("replay"), 2);
}

#[tokio::test]
async fn record_reward_rejects_a_different_value() {
    let registry = registry();
    finalized_with_samples(&registry, &["c1"]).await;
    assert_eq!(registry.record_reward(TASK, 0.25).await.expect("reward"), 1);
    let error = registry
        .record_reward(TASK, 0.75)
        .await
        .expect_err("conflict");
    assert_eq!(
        error.to_string(),
        "RL Task already has a different terminal reward"
    );
    assert_eq!(error.code(), "task_conflict");
}

#[tokio::test]
async fn record_reward_resumes_a_projecting_record() {
    let registry = registry();
    finalized_with_samples(&registry, &["c1", "c2"]).await;
    registry
        .store()
        .set(
            &keys::reward(TASK),
            &json!({"reward": 0.5, "status": "projecting", "sample_count": 2}).to_string(),
        )
        .await
        .unwrap();
    assert_eq!(registry.record_reward(TASK, 0.5).await.expect("resume"), 2);
    let stored = registry
        .store()
        .get(&keys::reward(TASK))
        .await
        .unwrap()
        .expect("record");
    assert!(stored.contains("\"published\""));
    assert_eq!(
        registry.pending_sample_ids(TASK).await.unwrap(),
        Vec::<String>::new()
    );
}

#[tokio::test]
async fn record_reward_reports_missing_tasks() {
    let registry = registry();
    let error = registry
        .record_reward("nope", 0.5)
        .await
        .expect_err("missing");
    assert_eq!(error.code(), "task_not_found");
}
