//! Integration tests for the online-RL HTTP service.
//!
//! The router is driven in-process with [`tower::ServiceExt::oneshot`]: no port
//! is bound, no model is called, and no clock is read. The training runner uses
//! the deterministic local executor writing under a temporary `lora_root`, and
//! every recording stamp comes from the registry's fixed clock.
//!
//! Python parity: `openjiuwen/agent_evolving/agent_rl/online/service.py` — the
//! task, capture, training, upload and trajectory routes, plus the error
//! envelope of every clause that can reject a request.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use axum::body::{to_bytes, Body};
use axum::http::{header::CONTENT_TYPE, Request, StatusCode};
use openjiuwen_agent_evolving::online::{RlServiceState, EPOCH_STAMP};
use openjiuwen_agent_evolving::service::{build_router, build_router_with_state, RlServiceConfig};
use serde_json::{json, Value};
use tempfile::TempDir;
use tower::ServiceExt;

/// Session header of the start route.
const SESSION: &str = "session-1";
/// Policy model the gateway serves; capture requests must name it.
const POLICY_MODEL: &str = "policy-model-1";

/// A served router plus the state behind it and the artifact directory.
struct Service {
    router: axum::Router,
    state: RlServiceState,
    dir: TempDir,
}

impl Service {
    /// Builds a service with `min_samples_for_training = 1` and the defaults.
    fn new() -> Self {
        Self::with_config(RlServiceConfig::default())
    }

    /// Builds a service from `config`, rooting artifacts in a temporary directory.
    fn with_config(config: RlServiceConfig) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let config = RlServiceConfig {
            model_id: "model-1".to_owned(),
            lora_root: Some(dir.path().to_string_lossy().into_owned()),
            ..config
        };
        let state = RlServiceState::new(&config).unwrap();
        let router = build_router_with_state(config, state.clone());
        Self { router, state, dir }
    }

    /// Sends `request` and decodes the JSON body (empty bodies read as `null`).
    async fn send(&self, request: Request<Body>) -> (StatusCode, Value) {
        let response = self.router.clone().oneshot(request).await.unwrap();
        let status = response.status();
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let body = if bytes.is_empty() {
            Value::Null
        } else {
            serde_json::from_slice(&bytes).unwrap()
        };
        (status, body)
    }

    /// Starts a task with every gateway header set.
    async fn start_task(&self, task_id: &str, reward_mode: &str) -> (StatusCode, Value) {
        self.send(start_request(task_id, "lora-a", POLICY_MODEL, reward_mode))
            .await
    }
}

/// Builds a `GET` request.
fn get(uri: &str) -> Request<Body> {
    Request::builder()
        .method("GET")
        .uri(uri)
        .body(Body::empty())
        .unwrap()
}

/// Builds a `POST` request with a JSON body.
fn post_json(uri: &str, body: &Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(uri)
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_vec(body).unwrap()))
        .unwrap()
}

/// Builds a `POST` request with no body.
fn post_empty(uri: &str) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(uri)
        .body(Body::empty())
        .unwrap()
}

/// Builds a `POST /v1/rl/tasks/start` request with the gateway headers.
fn start_request(
    task_id: &str,
    policy_name: &str,
    policy_model: &str,
    mode: &str,
) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/rl/tasks/start")
        .header(CONTENT_TYPE, "application/json")
        .header("x-agent-session-id", SESSION)
        .header("x-aigw-rl-task-id", task_id)
        .header("x-aigw-rl-policy-name", policy_name)
        .header("x-aigw-rl-policy-model", policy_model)
        .body(Body::from(
            serde_json::to_vec(&json!({ "reward_mode": mode })).unwrap(),
        ))
        .unwrap()
}

/// A normalized-request-shaped chat payload naming the policy model.
fn chat_request(stream: bool) -> Value {
    let mut request = json!({
        "model": POLICY_MODEL,
        "messages": [{"role": "user", "content": "hi"}],
    });
    if stream {
        request["stream"] = json!(true);
        request["stream_options"] = json!({"chunk_include_usage": false});
    }
    request
}

/// A complete chat completion with one choice and matching usage counters.
fn chat_response(content: &str) -> Value {
    json!({
        "object": "chat.completion",
        "model": POLICY_MODEL,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    })
}

/// A sample payload the upload route accepts.
fn sample(sample_id: &str, reward: f64) -> Value {
    json!({
        "sample_id": sample_id,
        "reward": reward,
        "session_id": SESSION,
        "task_id": "task-1",
        "payload": {"policy_version": "lora-a"},
    })
}

/// Asserts the Python envelope shape and returns `(code, message)`.
fn envelope(body: &Value) -> (&str, &str) {
    let error = body["error"]
        .as_object()
        .unwrap_or_else(|| panic!("missing error envelope: {body}"));
    assert_eq!(error.len(), 2, "envelope carries only code/message: {body}");
    (
        error["code"].as_str().unwrap(),
        error["message"].as_str().unwrap(),
    )
}

#[tokio::test]
async fn task_lifecycle_start_replay_reward_and_stop() {
    let service = Service::new();

    let (status, body) = service.start_task("task-1", "terminal").await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(body["rl_task_id"], "task-1");
    assert_eq!(body["agent_session_id"], SESSION);
    assert_eq!(body["model_id"], "model-1");
    assert_eq!(body["policy_lora_name"], "lora-a");
    assert_eq!(body["policy_model"], POLICY_MODEL);
    assert_eq!(body["reward_mode"], "terminal");
    assert_eq!(body["status"], "active");
    assert_eq!(body["created_at"], EPOCH_STAMP);
    assert!(body["finished_at"].is_null());

    // An exact replay adopts the record instead of creating a second task.
    let (status, replay) = service.start_task("task-1", "terminal").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(replay["created_at"], EPOCH_STAMP);

    // A different policy under the same id conflicts.
    let (status, body) = service
        .send(start_request("task-1", "lora-b", POLICY_MODEL, "terminal"))
        .await;
    assert_eq!(status, StatusCode::CONFLICT);
    assert_eq!(envelope(&body).0, "task_conflict");

    let (status, body) = service.send(get("/v1/rl/tasks/task-1")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "active");
}

#[tokio::test]
async fn start_adopts_the_sessions_active_task() {
    let service = Service::new();
    let (status, first) = service.start_task("task-1", "terminal").await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(first["rl_task_id"], "task-1");

    let (status, adopted) = service
        .send(start_request("task-2", "lora-a", POLICY_MODEL, "terminal"))
        .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(adopted["rl_task_id"], "task-1");
}

#[tokio::test]
async fn capture_before_after_and_discard() {
    let service = Service::new();
    service.start_task("task-1", "terminal").await;

    let before = json!({
        "rl_task_id": "task-1",
        "capture_id": "capture-1",
        "request": chat_request(true),
    });
    let (status, body) = service
        .send(post_json("/internal/v1/completions:before", &before))
        .await;
    assert_eq!(status, StatusCode::OK);
    let normalized = &body["request"];
    assert_eq!(normalized["logprobs"], true);
    assert_eq!(normalized["top_logprobs"], 1);
    assert_eq!(normalized["return_token_ids"], true);
    assert_eq!(normalized["stream_options"]["include_usage"], true);
    // The pre-existing stream option survives the normalization.
    assert_eq!(normalized["stream_options"]["chunk_include_usage"], false);
    assert_eq!(normalized["messages"][0]["content"], "hi");

    // The same request replays; a different one for the same id conflicts.
    let (status, _) = service
        .send(post_json("/internal/v1/completions:before", &before))
        .await;
    assert_eq!(status, StatusCode::OK);
    let (status, body) = service
        .send(post_json(
            "/internal/v1/completions:before",
            &json!({
                "rl_task_id": "task-1",
   #[tokio::test]
async fn capture_discard_semantics() {
    let service = Service::new();
    service.start_task("task-1", "terminal").await;
    let before = json!({
        "rl_task_id": "task-1",
        "capture_id": "capture-1",
        "request": chat_request(false),
    });
    let (status, _) = service
        .send(post_json("/internal/v1/completions:before", &before))
        .await;
    assert_eq!(status, StatusCode::OK);
    let (status, _) = service
        .send(post_json(
            "/internal/v1/completions:after",
            &json!({
                "rl_task_id": "task-1",
                "capture_id": "capture-1",
                "request": chat_request(false),
                "response": chat_response("hello"),
            }),
        ))
        .await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    // A policy mismatch is a conflict of the capture, not of the request shape.
    let (status, body) = service
        .send(post_json(
            "/internal/v1/completions:before",
            &json!({
                "rl_task_id": "task-1",
                "capture_id": "capture-2",
                "request": {"model": "other-model", "messages": []},
            }),
        ))
        .await;
    assert_eq!(status, StatusCode::CONFLICT);
    assert_eq!(envelope(&body).0, "completion_conflict");
    // An abandoned capture discards idempotently, including an unknown id.
    let discard = json!({
        "rl_task_id": "task-1",
        "capture_id": "capture-2",
    });
    let (status, _) = service
        .send(post_json("/internal/v1/completions:discard", &discard))
        .await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    let (status, _) = service
        .send(post_json("/internal/v1/completions:discard", &discard))
        .await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    // A committed capture cannot be discarded.
    let (status, body) = service
        .send(post_json(
            "/internal/v1/completions:discard",
            &json!({"rl_task_id": "task-1", "capture_id": "capture-1"}),
        ))
        .await;
    assert_eq!(status, StatusCode::CONFLICT);
    assert_eq!(envelope(&body).0, "completion_conflict");
}
#[tokio::test]
async fn task_reward_requires_finalization_and_ranges() {
    let service = Service::new();
    service.start_task("task-1", "terminal").await;
    // A capture makes the terminal reward meaningful.
    let (status, _) = service
        .send(post_json(
            "/internal/v1/completions:before",
            &json!({
                "rl_task_id": "task-1",
                "capture_id": "capture-1",
                "request": chat_request(false),
            }),
        ))
        .await;
    assert_eq!(status, StatusCode::OK);
    let (status, body) = service
        .send(post_json(
            "/internal/v1/completions:after",
            &json!({
                "rl_task_id": "task-1",
                "capture_id": "capture-1",
                "request": chat_request(false),
                "response": chat_response("hello"),
            }),
        ))
        .await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    assert!(body.is_null());

    // A reward needs a finalized task first.
    let (status, body) = service
        .send(post_json(
            "/v1/rl/tasks/task-1/reward",
            &json!({"reward": 0.5}),
        ))
        .await;
    assert_eq!(status, StatusCode::CONFLICT);
    assert_eq!(envelope(&body).0, "task_conflict");

    let (status, body) = service.send(post_empty("/v1/rl/tasks/task-1/stop")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "finalized");
    assert_eq!(body["finish_reason"], "user_stopped");
    assert_eq!(body["finished_at"], EPOCH_STAMP);

    let (status, body) = service
        .send(post_json(
            "/v1/rl/tasks/task-1/reward",
            &json!({"reward": 0.5}),
        ))
        .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["sample_count"], 1);

    // Replaying the same reward is idempotent; a different one conflicts.
    let (status, body) = service
        .send(post_json(
            "/v1/rl/tasks/task-1/reward",
            &json!({"reward": 0.5}),
        ))
        .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["sample_count"], 1);
    let (status, body) = service
        .send(post_json(
            "/v1/rl/tasks/task-1/reward",
            &json!({"reward": 0.9}),
        ))
        .await;
    assert_eq!(status, StatusCode::CONFLICT);
    assert_eq!(envelope(&body).0, "task_conflict");

    // Out-of-range and non-numeric rewards are client errors.
    for reward in [json!(1.5), json!(-0.1), json!("high"), json!(null)] {
        let (status, body) = service
            .send(post_json(
                "/v1/rl/tasks/task-1/reward",
                &json!({"reward": reward}),
            ))
            .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "reward: {reward}");
        assert_eq!(envelope(&body).0, "invalid_reward");
    }
    let (status, body) = service
        .send(post_json(
            "/v1/rl/tasks/task-1/reward",
            &json!({"reward": 0.5, "extra": 1}),
        ))
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(envelope(&body).0, "invalid_reward_request");
}
#[tokio::test]
async fn delayed_feedback_captures_require_their_turn() {
    let service = Service::new();
    service.start_task("task-1", "delayed_feedback").await;

    // A delayed-feedback capture needs an agent turn.
    let (status, body) = service
        .send(post_json(
            "/internal/v1/completions:before",
            &json!({
                "rl_task_id": "task-1",
                "capture_id": "capture-1",
                "request": chat_request(false),
            }),
        ))
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(envelope(&body).0, "invalid_completion");

    let (status, _) = service
        .send(post_json(
            "/internal/v1/completions:before",
            &json!({
                "rl_task_id": "task-1",
                "capture_id": "capture-1",
                "agent_turn_id": "turn-1",
                "request": chat_request(false),
            }),
        ))
        .await;
    assert_eq!(status, StatusCode::OK);

    // A second turn may only open once the first one published, and publishing
    // waits for the open capture of turn-1.
    let (status, body) = service
        .send(post_json(
            "/internal/v1/completions:before",
            &json!({
                "rl_task_id": "task-1",
                "capture_id": "capture-2",
                "agent_turn_id": "turn-2",
                "request": chat_request(false),
            }),
        ))
        .await;
    assert_eq!(status, StatusCode::CONFLICT);
    assert_eq!(envelope(&body).0, "completion_conflict");

    // Committing the open capture clears the way for the next turn.
    let (status, _) = service
        .send(post_json(
            "/internal/v1/completions:after",
            &json!({
                "rl_task_id": "task-1",
                "capture_id": "capture-1",
                "agent_turn_id": "turn-1",
                "request": chat_request(false),
                "response": chat_response("hello"),
            }),
        ))
        .await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    let (status, _) = service
        .send(post_json(
            "/internal/v1/completions:before",
            &json!({
                "rl_task_id": "task-1",
                "capture_id": "capture-2",
                "agent_turn_id": "turn-2",
                "request": chat_request(false),
            }),
        ))
        .await;
    assert_eq!(status, StatusCode::OK);
}
#[tokio::test]
async fn capture_validation_rejects_bad_shapes() {
    let service = Service::new();
    service.start_task("task-1", "terminal").await;
    // Shape and existence errors.
    let (status, body) = service
        .send(post_json("/internal/v1/completions:before", &json!({})))
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(envelope(&body).0, "invalid_completion");
    let (status, body) = service
        .send(post_json(
            "/internal/v1/completions:before",
            &json!({
                "rl_task_id": "task-9",
                "capture_id": "capture-1",
                "request": chat_request(false),
            }),
        ))
        .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(envelope(&body).0, "task_not_found");
    // `n` must be 1 and the response usage must add up.
    let (status, body) = service
        .send(post_json(
            "/internal/v1/completions:before",
            &json!({
                "rl_task_id": "task-1",
                "capture_id": "capture-3",
                "request": {
                    "model": POLICY_MODEL,
                    "messages": [],
                    "n": 2,
                },
            }),
        ))
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(envelope(&body).0, "invalid_completion");
    let mut bad_usage = chat_response("hello");
    bad_usage["usage"] = json!({"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 9});
    let (status, body) = service
        .send(post_json(
            "/internal/v1/completions:after",
            &json!({
                "rl_task_id": "task-1",
                "capture_id": "capture-3",
                "request": chat_request(false),
                "response": bad_usage,
            }),
        ))
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(envelope(&body).0, "invalid_completion");
}
             "capture_id": "capture-1",
                "agent_turn_id": "turn-1",
                "request": chat_request(false),
                "response": chat_response("hello"),
            }),
        ))
        .await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    let (status, _) = service
        .send(post_json(
            "/internal/v1/completions:before",
            &json!({
                "rl_task_id": "task-1",
                "capture_id": "capture-2",
                "agent_turn_id": "turn-2",
                "request": chat_request(false),
            }),
        ))
        .await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn internal_finish_and_abort_validate_reasons() {
    let service = Service::new();
    service.start_task("task-1", "terminal").await;

    let (status, body) = service
        .send(post_json(
            "/internal/v1/rl/tasks/task-1:finish",
            &json!({"reason": "user_stopped"}),
        ))
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(envelope(&body).0, "invalid_finish_reason");

    let (status, body) = service
        .send(post_json(
            "/internal/v1/rl/tasks/task-1:finish",
            &json!({"reason": "service_stopped"}),
        ))
        .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "finalized");
    assert_eq!(body["finish_reason"], "service_stopped");

    // Aborting an already closed task is idempotent.
    let (status, body) = service
        .send(post_json(
            "/internal/v1/rl/tasks/task-1:abort",
            &json!({"reason": "timeout"}),
        ))
        .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "finalized");

    service.start_task("task-2", "terminal").await;
    let (status, body) = service
        .send(post_json(
            "/internal/v1/rl/tasks/task-2:abort",
            &json!({"reason": "capture_failed"}),
        ))
        .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "aborted");
    assert_eq!(body["finish_reason"], "capture_failed");

    service.start_task("task-3", "terminal").await;
    for reason in [json!("bogus"), json!(null)] {
        let (status, body) = service
            .send(post_json(
                "/internal/v1/rl/tasks/task-3:abort",
                &json!({"reason": reason}),
            ))
            .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "reason: {reason}");
        assert_eq!(envelope(&body).0, "invalid_abort_reason");
    }

    let (status, body) = service
        .send(post_json(
            "/internal/v1/rl/tasks/task-9:finish",
            &json!({"reason": "service_stopped"}),
        ))
        .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(envelope(&body).0, "task_not_found");
}

#[tokio::test]
async fn training_runs_claim_upload_and_export() {
    let service = Service::new();

    // No samples yet: the claim refuses with the shortage it needs.
    let (status, body) = service
        .send(post_json("/v1/rl/training/runs", &json!(null)))
        .await;
    assert_eq!(status, StatusCode::CONFLICT);
    assert_eq!(envelope(&body).0, "insufficient_samples");

    // An upload persists each sample once.
    let upload = json!({
        "protocol_version": "rail-v1",
        "samples": [sample("sample-1", 1.0), sample("sample-2", 0.5)],
        "unknown_field": {"tolerated": true},
    });
    let (status, body) = service
        .send(post_json("/v1/gateway/upload/batch", &upload))
        .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["ok"], true);
    assert_eq!(body["result"]["accepted"], 2);
    assert_eq!(body["result"]["rejected"], 0);
    assert_eq!(body["result"]["duplicate"], 0);
    assert_eq!(body["result"]["protocol_version"], "rail-v1");
    assert_eq!(body["result"]["items"], json!([]));

    let (status, body) = service
        .send(post_json(
            "/v1/gateway/upload/batch",
            &json!({
                "samples": [
                    sample("sample-1", 1.0),
                    sample("sample-2", 0.5),
                    sample("sample-3", 0.0),
                ],
            }),
        ))
        .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["result"]["accepted"], 1);
    assert_eq!(body["result"]["duplicate"], 2);

    // The claim drains every pending sample, capped by `max_samples_per_run`.
    let (status, run) = service
        .send(post_json("/v1/rl/training/runs", &json!({})))
        .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(run["status"], "succeeded");
    // `stage` is the recovery boundary (the last place a restart must resume
    // from), so a succeeded run keeps the stage it finished in.
    assert_eq!(run["stage"], "activating");
    assert_eq!(run["sample_count"], 3);
    assert_eq!(run["lora_name"], "model-1:v1");
    assert_eq!(run["policy_versions"]["lora-a"], 3);
    assert!(run.get("sample_ids").is_none());
    let run_id = run["training_run_id"].as_str().unwrap().to_owned();

    let (status, body) = service
        .send(get(&format!("/v1/rl/training/runs/{run_id}")))
        .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "succeeded");
    assert_eq!(body["lora_path"], run["lora_path"]);

    // A terminal run stops idempotently.
    let (status, body) = service
        .send(post_empty(&format!("/v1/rl/training/runs/{run_id}/stop")))
        .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "succeeded");

    let (status, body) = service.send(get("/v1/rl/training/runs/run-9")).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(envelope(&body).0, "training_run_not_found");

    // A queued run (claimed through the state) is visible as the active run.
    service
        .send(post_json(
            "/v1/gateway/upload/batch",
            &json!({"samples": [sample("sample-4", 0.25)]}),
        ))
        .await;
    let (queued, created, _samples) = service.state.runner().claim().await.unwrap();
    assert!(created);
    let (status, body) = service
        .send(post_json("/v1/rl/training/runs", &json!({})))
        .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["training_run_id"], queued.training_run_id.as_str());
    assert_eq!(body["status"], "pending");

    let (status, body) = service
        .send(post_empty(&format!(
            "/v1/rl/training/runs/{}/stop",
            queued.training_run_id
        )))
        .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "canceled");

    // The checkpoint export is reachable through the shared state.
    let export_dir = tempfile::tempdir().unwrap();
    let path = service
        .state
        .runner()
        .export_checkpoint(&run_id, export_dir.path())
        .await
        .unwrap();
    let contents = std::fs::read_to_string(&path).unwrap();
    assert!(path.starts_with(export_dir.path()));
    let checkpoint: Value = serde_json::from_str(&contents).unwrap();
    assert_eq!(checkpoint["training_run_id"], run_id);
    assert_eq!(checkpoint["artifact"]["lora_name"], "model-1:v1");

    // The repository shares the executor root, so the descriptor is on disk.
    let repository = service.state.lora().expect("lora_root is configured");
    assert_eq!(repository.root(), service.dir.path());
}

#[tokio::test]
async fn upload_validation_and_sample_defaults() {
    let service = Service::with_config(RlServiceConfig {
        upload_max_samples: 2,
        ..RlServiceConfig::default()
    });

    let (status, body) = service
        .send(
            Request::builder()
                .method("POST")
                .uri("/v1/gateway/upload/batch")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from("{not json"))
                .unwrap(),
        )
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(envelope(&body).0, "invalid_rail_batch");

    let (status, body) = service
        .send(post_json(
            "/v1/gateway/upload/batch",
            &json!({"samples": "nope"}),
        ))
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(envelope(&body).0, "invalid_rail_batch");

    let (status, body) = service
        .send(post_json(
            "/v1/gateway/upload/batch",
            &json!({"samples": [{"reward": 0.5}]}),
        ))
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(envelope(&body).0, "invalid_rail_batch");

    let (status, body) = service
        .send(post_json(
            "/v1/gateway/upload/batch",
            &json!({"samples": [sample("sample-1", 0.5), sample("sample-2", 0.5), sample("sample-3", 0.5)]}),
        ))
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(envelope(&body).0, "invalid_rail_batch");
    assert!(envelope(&body).1.contains("upload limit"));

    // An omitted `user_id` is taken from the service's model id.
    let (status, body) = service
        .send(post_json(
            "/v1/gateway/upload/batch",
            &json!({"samples": [{"sample_id": "sample-1", "reward": 0.5}]}),
        ))
        .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["result"]["accepted"], 1);
    let (status, body) = service.send(get("/v1/rl/trajectories/sample-1")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["user_id"], "model-1");
    assert_eq!(body["status"], "pending");
}

#[tokio::test]
async fn trajectory_read_model_lists_and_counts() {
    let service = Service::new();
    let (status, body) = service.send(get("/v1/rl/trajectories/stats")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["model_id"], "model-1");
    assert_eq!(body["user_id"], "model-1");
    assert_eq!(body["total_samples"], 0);

    service
        .send(post_json(
            "/v1/gateway/upload/batch",
            &json!({"samples": [sample("sample-1", 1.0), sample("sample-2", 0.25)]}),
        ))
        .await;

    let (status, body) = service.send(get("/v1/rl/trajectories/stats")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["pending_samples"], 2);
    assert_eq!(body["trained_samples"], 0);
    assert_eq!(body["total_samples"], 2);

    let (status, body) = service.send(get("/v1/rl/trajectories")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["count"], 2);
    assert_eq!(body["limit"], 100);
    assert!(body["status"].is_null());
    assert_eq!(body["trajectories"][0]["sample_id"], "sample-1");
    assert_eq!(body["trajectories"][1]["sample_id"], "sample-2");

    let (status, body) = service
        .send(get("/v1/rl/trajectories?status=pending&limit=1"))
        .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["count"], 1);
    assert_eq!(body["status"], "pending");

    let (status, body) = service.send(get("/v1/rl/trajectories?status=bogus")).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(envelope(&body).0, "invalid_request");

    let (status, body) = service.send(get("/v1/rl/trajectories?limit=0")).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(envelope(&body).0, "invalid_request");

    let (status, body) = service.send(get("/v1/rl/trajectories/sample-2")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["sample_id"], "sample-2");
    assert_eq!(body["reward"], 0.25);

    let (status, body) = service.send(get("/v1/rl/trajectories/sample-9")).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(envelope(&body).0, "trajectory_not_found");
}

#[tokio::test]
async fn health_reports_executor_readiness() {
    let service = Service::new();
    let (status, body) = service.send(get("/health")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "ready");
    assert_eq!(body["model_id"], "model-1");

    // A service that is not ready refuses everything `/health` reports.
    let not_ready = Service::with_config(RlServiceConfig {
        ready: false,
        ..RlServiceConfig::default()
    });
    let (status, body) = not_ready.send(get("/health")).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(envelope(&body).0, "service_not_ready");

    // The GPU executor cannot train in this build, so the probe fails.
    let ppo = Service::with_config(RlServiceConfig {
        executor: "ppo".to_owned(),
        ..RlServiceConfig::default()
    });
    let (status, body) = ppo.send(get("/health")).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    let (code, message) = envelope(&body);
    assert_eq!(code, "service_not_ready");
    assert!(message.contains("torch"), "{message}");
    assert!(message.contains("cuda:0"), "{message}");

    // A training claim on that executor still claims, and records the failure.
    ppo.send(post_json(
        "/v1/gateway/upload/batch",
        &json!({"samples": [sample("sample-1", 1.0)]}),
    ))
    .await;
    let (status, body) = ppo
        .send(post_json("/v1/rl/training/runs", &json!(null)))
        .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(body["status"], "failed");
    assert!(
        body["failure_reason"].as_str().unwrap().contains("torch"),
        "failure_reason: {}",
        body["failure_reason"]
    );
}

#[tokio::test]
async fn misconfigured_state_is_a_transport_error() {
    for config in [
        RlServiceConfig {
            executor: "tensorrt".to_owned(),
            ..RlServiceConfig::default()
        },
        RlServiceConfig {
            min_samples_for_training: 0,
            ..RlServiceConfig::default()
        },
        RlServiceConfig {
            max_samples_per_run: 1,
            min_samples_for_training: 4,
            ..RlServiceConfig::default()
        },
        RlServiceConfig {
            upload_max_samples: 0,
            ..RlServiceConfig::default()
        },
    ] {
        let router = build_router(config);
        let response = router
            .oneshot(get("/v1/rl/trajectories/stats"))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let body: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(envelope(&body).0, "service_not_ready");
    }
}

#[tokio::test]
async fn error_envelopes_carry_only_a_code_and_a_message() {
    let service = Service::new();
    for request in [
        post_empty("/v1/rl/tasks/start"),
        get("/v1/rl/tasks/task-9"),
        get("/v1/rl/training/runs/run-9"),
        get("/v1/rl/trajectories/traj-9"),
        post_empty("/internal/v1/completions:before"),
        post_empty("/internal/v1/rl/tasks/task-9"),
        post_empty("/v1/gateway/upload/batch"),
    ] {
        let (status, body) = service.send(request).await;
        assert!(status.is_client_error(), "status: {status} body: {body}");
        let (code, message) = envelope(&body);
        assert!(!code.is_empty());
        assert!(!message.is_empty());
    }
}
