//! Offline integration tests for the two stateless routes of the RL service.
//!
//! The router is driven in-process with [`tower::ServiceExt::oneshot`]: no port is
//! bound, no model is called, and no environment or clock is read. The tests pin
//! `/health` and the offline `/v1/evaluate` scorer, plus the envelope the
//! state-machine routes answer with; the RL state machines themselves are
//! exercised by `tests/online_service.rs`.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use axum::body::{to_bytes, Body};
use axum::http::{header::CONTENT_TYPE, Request, StatusCode};
use openjiuwen_agent_evolving::service::{build_router, EvaluateRequest, RlServiceConfig};
use serde_json::{json, Value};
use tower::ServiceExt;

/// Sends `request` through `router` and decodes the JSON response.
async fn send(router: axum::Router, request: Request<Body>) -> (StatusCode, Value) {
    let response = router.oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    (status, serde_json::from_slice(&bytes).unwrap())
}

/// Builds a bodyless `GET` request.
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

/// A valid case payload with an explicit `case_id` and an `answer` label.
fn case_json(case_id: &str, answer: &str) -> Value {
    json!({
        "inputs": {"question": "what?"},
        "label": {"answer": answer},
        "case_id": case_id,
    })
}

/// The default router used by tests that do not care about configuration.
fn default_router() -> axum::Router {
    build_router(RlServiceConfig::default())
}

#[tokio::test]
async fn health_reports_ready_and_echoes_model_id() {
    let config = RlServiceConfig {
        model_id: "gpt-test".to_owned(),
        ..RlServiceConfig::default()
    };
    let (status, body) = send(build_router(config), get("/health")).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "ready");
    assert_eq!(body["model_id"], "gpt-test");
}

#[tokio::test]
async fn health_reports_service_not_ready() {
    let config = RlServiceConfig {
        ready: false,
        ..RlServiceConfig::default()
    };
    let (status, body) = send(build_router(config), get("/health")).await;

    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(body["error"]["code"], "service_not_ready");
    assert!(body["error"]["message"].is_string());
}

#[tokio::test]
async fn evaluate_scores_all_correct_predictions() {
    let request = json!({
        "cases": [case_json("case-1", "4"), case_json("case-2", "Paris")],
        "predictions": ["4", "Paris"],
    });
    let (status, body) = send(default_router(), post_json("/v1/evaluate", &request)).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["num_cases"], 2);
    assert_eq!(body["mean_score"], 1.0);
    assert_eq!(body["per_metric"]["exact_match"], 1.0);
}

#[tokio::test]
async fn evaluate_scores_partial_match() {
    let request = json!({
        "cases": [case_json("case-1", "4"), case_json("case-2", "Paris")],
        "predictions": ["4", "London"],
    });
    let (status, body) = send(default_router(), post_json("/v1/evaluate", &request)).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["num_cases"], 2);
    assert_eq!(body["mean_score"], 0.5);
    assert_eq!(body["per_metric"]["exact_match"], 0.5);
}

#[tokio::test]
async fn evaluate_accepts_answers_map_with_normalized_match() {
    // No `predictions`: the map keyed by `case_id` is the prediction source.
    // "paris" matches "Paris" because exact match normalizes case and whitespace.
    let request = json!({
        "cases": [case_json("case-1", "4"), case_json("case-2", "Paris")],
        "answers": {"case-1": "4", "case-2": "  paris "},
    });
    let (status, body) = send(default_router(), post_json("/v1/evaluate", &request)).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["num_cases"], 2);
    assert_eq!(body["mean_score"], 1.0);
}

#[tokio::test]
async fn evaluate_rejects_answers_map_missing_case() {
    let request = json!({
        "cases": [case_json("case-1", "4"), case_json("case-2", "Paris")],
        "answers": {"case-1": "4"},
    });
    let (status, body) = send(default_router(), post_json("/v1/evaluate", &request)).await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["code"], "invalid_request");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("case-2"),
        "unexpected message: {body}"
    );
}

#[tokio::test]
async fn evaluate_rejects_prediction_length_mismatch() {
    let request = json!({
        "cases": [case_json("case-1", "4"), case_json("case-2", "Paris")],
        "predictions": ["4"],
    });
    let (status, body) = send(default_router(), post_json("/v1/evaluate", &request)).await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["code"], "invalid_request");
}

#[tokio::test]
async fn evaluate_rejects_unknown_metric_and_accepts_exact_match() {
    let unknown = json!({
        "cases": [case_json("case-1", "4")],
        "predictions": ["4"],
        "metric": "rouge",
    });
    let (status, body) = send(default_router(), post_json("/v1/evaluate", &unknown)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["code"], "invalid_request");
    assert!(
        body["error"]["message"].as_str().unwrap().contains("rouge"),
        "unexpected message: {body}"
    );

    let explicit = json!({
        "cases": [case_json("case-1", "4")],
        "predictions": ["4"],
        "metric": "exact_match",
    });
    let (status, body) = send(default_router(), post_json("/v1/evaluate", &explicit)).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["mean_score"], 1.0);
}

#[tokio::test]
async fn evaluate_accepts_empty_cases() {
    let request = json!({"cases": [], "predictions": []});
    let (status, body) = send(default_router(), post_json("/v1/evaluate", &request)).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["num_cases"], 0);
    assert_eq!(body["mean_score"], 0.0);
    assert_eq!(body["per_metric"], json!({}));
    assert_eq!(body["cases"], json!([]));
}

#[tokio::test]
async fn evaluate_response_has_report_shape() {
    let request = json!({
        "cases": [case_json("case-1", "4")],
        "predictions": ["4"],
    });
    let (status, body) = send(default_router(), post_json("/v1/evaluate", &request)).await;

    assert_eq!(status, StatusCode::OK);
    let object = body.as_object().unwrap();
    for key in [
        "run_id",
        "num_cases",
        "mean_score",
        "min_score",
        "max_score",
        "per_metric",
        "cases",
    ] {
        assert!(object.contains_key(key), "missing key: {key}");
    }
    assert_eq!(body["run_id"], "service");
    assert_eq!(body["min_score"], 1.0);
    assert_eq!(body["max_score"], 1.0);
}

#[test]
fn evaluate_request_tolerates_defaults_and_unknown_fields() {
    let request: EvaluateRequest = serde_json::from_value(json!({
        "cases": [case_json("case-1", "4")],
        "unknown": true,
    }))
    .unwrap();

    assert_eq!(request.cases.len(), 1);
    assert_eq!(request.cases[0].case_id, "case-1");
    assert_eq!(request.predictions.len(), 0);
    assert!(request.answers.is_none());
    assert!(request.metric.is_none());
}

#[tokio::test]
async fn rl_task_start_requires_a_session_header() {
    let (status, body) = send(
        default_router(),
        post_json("/v1/rl/tasks/start", &json!({"reward_mode": "terminal"})),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["code"], "missing_session_id");
}
#[tokio::test]
async fn rl_task_by_id_reports_not_found() {
    let (status, body) = send(default_router(), get("/v1/rl/tasks/task-1")).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["error"]["code"], "task_not_found");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("task-1"),
        "unexpected message: {body}"
    );
}
#[tokio::test]
async fn trajectory_routes_serve_the_sample_ledger() {
    // The static `stats` route must not be shadowed by the `{id}` route.
    let (status, body) = send(default_router(), get("/v1/rl/trajectories/stats")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["total_samples"], 0);
    let (status, body) = send(default_router(), get("/v1/rl/trajectories")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["count"], 0);
    let (status, body) = send(default_router(), get("/v1/rl/trajectories/traj-1")).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["error"]["code"], "trajectory_not_found");
}
#[tokio::test]
async fn internal_completion_route_validates_its_payload() {
    let (status, body) = send(
        default_router(),
        post_json("/internal/v1/completions:before", &json!({})),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["code"], "invalid_completion");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("rl_task_id"),
        "unexpected message: {body}"
    );
}
