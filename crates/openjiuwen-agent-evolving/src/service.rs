//! Offline axum skeleton of the Python RL service (`agent_rl` service layer).
//!
//! Python ships a full RL/RLOps HTTP service backed by Redis, vLLM/verl training
//! executors, and a rewards gateway. That stack is deferred (see
//! `docs/PARITY.md`), so this module keeps the *contract* instead: the health and
//! offline-evaluation endpoints work end-to-end, and every RL operation is
//! registered with a stable `501` response rather than a `404`.
//!
//! # Endpoints
//!
//! | Method | Path | Behaviour |
//! |---|---|---|
//! | `GET` | `/health` | `200 {"status":"ready","model_id":...}`, or `503` when `ready` is false |
//! | `POST` | `/v1/evaluate` | offline exact-match scoring over `cases` (or an `answers` map) |
//! | `POST` | `/v1/rl/tasks/start` | `501` |
//! | `GET` | `/v1/rl/tasks/{rl_task_id}` | `501` |
//! | `POST` | `/v1/rl/tasks/{rl_task_id}/stop` | `501` |
//! | `POST` | `/v1/rl/tasks/{rl_task_id}/reward` | `501` |
//! | `POST` | `/v1/rl/training/runs` | `501` |
//! | `GET` | `/v1/rl/training/runs/{id}` | `501` |
//! | `POST` | `/v1/rl/training/runs/{id}/stop` | `501` |
//! | `GET` | `/v1/rl/trajectories/stats` | `501` |
//! | `GET` | `/v1/rl/trajectories` | `501` |
//! | `GET` | `/v1/rl/trajectories/{id}` | `501` |
//! | `POST` | `/internal/v1/completions:before` | `501` |
//! | `POST` | `/internal/v1/completions:after` | `501` |
//! | `POST` | `/internal/v1/completions:discard` | `501` |
//! | `POST` | `/internal/v1/rl/tasks/{*operation}` | `501` (serves `.../{id}:finish` and `.../{id}:abort`) |
//!
//! # Error contract
//!
//! Every failure uses the Python envelope
//! `{"error": {"code": "...", "message": "..."}}`:
//!
//! - `invalid_request` — `400`, malformed or inconsistent request payloads.
//! - `service_not_ready` — `503`, the service is not accepting work yet.
//! - `internal_error` — `500`, an unexpected evaluation failure.
//! - `unsupported` — `501`, the deferred RL/ML stack (the message names the
//!   operation and points at `docs/PARITY.md`).
//!
//! The router is pure: building and exercising it binds no socket and needs no
//! Redis, no model, and no credentials.

use std::collections::BTreeMap;
use std::sync::Arc;

use axum::extract::State;
use axum::http::{StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

use crate::dataset::Case;
use crate::evaluator::{
    evaluate_cases, EvaluationConfig, EvaluationReport, ExactMatchMetric, MapCaseRunner, Metric,
    MetricEvaluator,
};

/// The only metric the offline service implements.
const SUPPORTED_METRIC: &str = "exact_match";

/// The caller-supplied run identifier used by the evaluation endpoint.
const SERVICE_RUN_ID: &str = "service";

/// The reason appended to every `unsupported` message.
const DEFERRED_REASON: &str =
    "is not supported: the Python RL/ML service stack is deferred; see docs/PARITY.md";

/// Configuration of the RL service skeleton.
///
/// Python: `openjiuwen.agent_evolving.agent_rl.rl_service.config.RLServiceConfig`
/// (reduced — the Redis and training-executor fields belong to deferred scope).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct RlServiceConfig {
    /// Identifier of the model the service would serve; echoed by `/health`.
    pub model_id: String,
    /// Host the listener binds to.
    pub listen_host: String,
    /// Port the listener binds to.
    pub listen_port: u16,
    /// Whether the service reports itself ready.
    pub ready: bool,
}

impl Default for RlServiceConfig {
    fn default() -> Self {
        Self {
            model_id: String::new(),
            listen_host: "127.0.0.1".to_owned(),
            listen_port: 18_081,
            ready: true,
        }
    }
}

/// Payload of `POST /v1/evaluate`.
///
/// Exactly one prediction source is used: when `answers` is present it is keyed
/// by `case_id` and takes precedence over `predictions`, which is the shape the
/// Python gateway uses for recorded baselines.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluateRequest {
    /// The cases to score.
    pub cases: Vec<Case>,
    /// One prediction per case, positionally aligned with `cases`.
    #[serde(default)]
    pub predictions: Vec<Value>,
    /// Predictions keyed by `case_id`; overrides `predictions` when present.
    #[serde(default)]
    pub answers: Option<BTreeMap<String, Value>>,
    /// Optional metric name; only `exact_match` is accepted.
    #[serde(default)]
    pub metric: Option<String>,
}

/// Response of `POST /v1/evaluate`: the standard [`EvaluationReport`].
pub type EvaluateResponse = EvaluationReport;

/// An HTTP error carrying the Python `{"error": {"code", "message"}}` envelope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiError {
    /// Stable machine-readable error code (snake case, as in Python).
    pub code: String,
    /// Human-readable explanation.
    pub message: String,
    /// HTTP status; serialized only as the response status line.
    #[serde(skip)]
    pub status: StatusCode,
}

impl ApiError {
    /// Builds a `400 invalid_request` error.
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self {
            code: "invalid_request".to_owned(),
            message: message.into(),
            status: StatusCode::BAD_REQUEST,
        }
    }

    /// Builds a `501 unsupported` error naming the deferred `operation`.
    pub fn unsupported(operation: impl Into<String>) -> Self {
        Self {
            code: "unsupported".to_owned(),
            message: format!("{} {DEFERRED_REASON}", operation.into()),
            status: StatusCode::NOT_IMPLEMENTED,
        }
    }

    /// Builds a `503 service_not_ready` error.
    pub fn service_not_ready() -> Self {
        Self {
            code: "service_not_ready".to_owned(),
            message: "the RL service is not ready to accept requests".to_owned(),
            status: StatusCode::SERVICE_UNAVAILABLE,
        }
    }

    /// Builds a `500 internal_error` error.
    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            code: "internal_error".to_owned(),
            message: message.into(),
            status: StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = json!({
            "error": {
                "code": self.code,
                "message": self.message,
            }
        });
        (self.status, Json(body)).into_response()
    }
}

/// Builds the service router with `config` captured as shared state.
///
/// The router is built once per server, and also per request in tests; it never
/// performs I/O of its own.
pub fn build_router(config: RlServiceConfig) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/evaluate", post(evaluate))
        .route("/v1/rl/tasks/start", post(unsupported_operation))
        .route("/v1/rl/tasks/{rl_task_id}", get(unsupported_operation))
        .route(
            "/v1/rl/tasks/{rl_task_id}/stop",
            post(unsupported_operation),
        )
        .route(
            "/v1/rl/tasks/{rl_task_id}/reward",
            post(unsupported_operation),
        )
        .route("/v1/rl/training/runs", post(unsupported_operation))
        .route("/v1/rl/training/runs/{id}", get(unsupported_operation))
        .route(
            "/v1/rl/training/runs/{id}/stop",
            post(unsupported_operation),
        )
        // Static route registered before the dynamic one so `stats` can never be
        // captured as a trajectory id.
        .route("/v1/rl/trajectories/stats", get(unsupported_operation))
        .route("/v1/rl/trajectories", get(unsupported_operation))
        .route("/v1/rl/trajectories/{id}", get(unsupported_operation))
        .route(
            "/internal/v1/completions:before",
            post(unsupported_operation),
        )
        .route(
            "/internal/v1/completions:after",
            post(unsupported_operation),
        )
        .route(
            "/internal/v1/completions:discard",
            post(unsupported_operation),
        )
        // axum 0.8 cannot express a parameter that only covers part of a path
        // segment (`{id}:finish`), so the internal task actions — the Python
        // `.../{id}:finish` and `.../{id}:abort` endpoints — are served by one
        // catch-all that reports the requested path.
        .route(
            "/internal/v1/rl/tasks/{*operation}",
            post(unsupported_operation),
        )
        .with_state(Arc::new(config))
}

/// Binds `config.listen_host:config.listen_port` and serves the router.
///
/// # Errors
///
/// [`Error::transport`] when the address cannot be bound or the server fails
/// while serving.
pub async fn serve(config: RlServiceConfig) -> Result<()> {
    let address = format!("{}:{}", config.listen_host, config.listen_port);
    let listener = tokio::net::TcpListener::bind(&address)
        .await
        .map_err(|error| Error::transport(format!("binding {address} failed: {error}")))?;

    tracing::info!(
        address = %address,
        model_id = %config.model_id,
        "RL service listening",
    );

    axum::serve(listener, build_router(config))
        .await
        .map_err(|error| Error::transport(format!("serving on {address} failed: {error}")))
}

/// `GET /health` — readiness probe.
async fn health(
    State(config): State<Arc<RlServiceConfig>>,
) -> std::result::Result<Json<Value>, ApiError> {
    if !config.ready {
        return Err(ApiError::service_not_ready());
    }
    Ok(Json(json!({
        "status": "ready",
        "model_id": config.model_id,
    })))
}

/// `POST /v1/evaluate` — offline exact-match scoring.
async fn evaluate(
    Json(request): Json<EvaluateRequest>,
) -> std::result::Result<Json<EvaluateResponse>, ApiError> {
    if let Some(metric) = request.metric.as_deref() {
        if metric != SUPPORTED_METRIC {
            return Err(ApiError::invalid_request(format!(
                "unsupported metric: {metric}"
            )));
        }
    }

    let predictions = resolve_predictions(&request)?;
    let runner = MapCaseRunner::new(
        request
            .cases
            .iter()
            .zip(predictions)
            .map(|(case, prediction)| (case.case_id.clone(), prediction)),
    );
    let metric: Arc<dyn Metric> = Arc::new(ExactMatchMetric::new());
    let evaluator = MetricEvaluator::new(vec![metric]);

    let report = evaluate_cases(
        SERVICE_RUN_ID,
        &runner,
        &evaluator,
        &request.cases,
        &CancellationToken::new(),
        &EvaluationConfig::default(),
    )
    .await
    .map_err(|error| match error {
        Error::Validation(message) => ApiError::invalid_request(message),
        other => ApiError::internal(other.to_string()),
    })?;

    Ok(Json(report))
}

/// Every deferred RL operation: answers `501` naming the requested path.
async fn unsupported_operation(uri: Uri) -> ApiError {
    ApiError::unsupported(uri.path().to_owned())
}

/// Resolves the per-case predictions from `answers` or `predictions`.
fn resolve_predictions(request: &EvaluateRequest) -> std::result::Result<Vec<Value>, ApiError> {
    if let Some(answers) = &request.answers {
        return request
            .cases
            .iter()
            .map(|case| {
                answers.get(&case.case_id).cloned().ok_or_else(|| {
                    ApiError::invalid_request(format!(
                        "missing answer for case_id: {}",
                        case.case_id
                    ))
                })
            })
            .collect();
    }

    if request.predictions.len() != request.cases.len() {
        return Err(ApiError::invalid_request(format!(
            "predictions length {} does not match cases length {}",
            request.predictions.len(),
            request.cases.len()
        )));
    }

    Ok(request.predictions.clone())
}

#[cfg(test)]
mod tests {
    use axum::body::{to_bytes, Body};
    use axum::http::{header::CONTENT_TYPE, Request, StatusCode};
    use serde::Serialize;
    use serde_json::Value;
    use tower::ServiceExt;

    use super::{build_router, EvaluateRequest, RlServiceConfig};

    async fn send(router: axum::Router, request: Request<Body>) -> (StatusCode, Value) {
        let response = router.oneshot(request).await.unwrap();
        let status = response.status();
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        (status, serde_json::from_slice(&body).unwrap())
    }

    fn get(uri: &str) -> Request<Body> {
        Request::builder()
            .method("GET")
            .uri(uri)
            .body(Body::empty())
            .unwrap()
    }

    fn post_json(uri: &str, body: &impl Serialize) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri(uri)
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_vec(body).unwrap()))
            .unwrap()
    }

    fn post_empty(uri: &str) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri(uri)
            .body(Body::empty())
            .unwrap()
    }

    fn case_json(case_id: &str, answer: &str) -> Value {
        serde_json::json!({
            "inputs": {"question": "what?"},
            "label": {"answer": answer},
            "case_id": case_id,
        })
    }

    #[test]
    fn config_defaults_match_python_shape() {
        let config = RlServiceConfig::default();
        assert_eq!(config.model_id, "");
        assert_eq!(config.listen_host, "127.0.0.1");
        assert_eq!(config.listen_port, 18_081);
        assert!(config.ready);

        let restored: RlServiceConfig =
            serde_json::from_value(serde_json::json!({"model_id": "m", "unknown": 1})).unwrap();
        assert_eq!(restored.model_id, "m");
        assert_eq!(restored.listen_host, "127.0.0.1");
        assert!(restored.ready);
    }

    #[tokio::test]
    async fn health_reports_ready() {
        let (status, body) = send(build_router(RlServiceConfig::default()), get("/health")).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["status"], "ready");
        assert_eq!(body["model_id"], "");
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
    async fn evaluate_scores_predictions_with_normalized_exact_match() {
        let request = serde_json::json!({
            "cases": [case_json("case-1", "4"), case_json("case-2", "paris")],
            "predictions": ["4", "Paris"],
        });
        let (status, body) = send(
            build_router(RlServiceConfig::default()),
            post_json("/v1/evaluate", &request),
        )
        .await;

        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["run_id"], "service");
        assert_eq!(body["num_cases"], 2);
        assert_eq!(body["mean_score"], 1.0);
        assert_eq!(body["per_metric"]["exact_match"], 1.0);
    }

    #[tokio::test]
    async fn evaluate_uses_answers_map() {
        let request = serde_json::json!({
            "cases": [case_json("case-1", "4"), case_json("case-2", "paris")],
            "answers": {"case-1": "4", "case-2": "  PARIS "},
        });
        let (status, body) = send(
            build_router(RlServiceConfig::default()),
            post_json("/v1/evaluate", &request),
        )
        .await;

        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["num_cases"], 2);
        assert_eq!(body["mean_score"], 1.0);
    }

    #[tokio::test]
    async fn evaluate_rejects_missing_answer() {
        let request = serde_json::json!({
            "cases": [case_json("case-1", "4"), case_json("case-2", "paris")],
            "answers": {"case-1": "4"},
        });
        let (status, body) = send(
            build_router(RlServiceConfig::default()),
            post_json("/v1/evaluate", &request),
        )
        .await;

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
        let request = serde_json::json!({
            "cases": [case_json("case-1", "4"), case_json("case-2", "paris")],
            "predictions": ["4"],
        });
        let (status, body) = send(
            build_router(RlServiceConfig::default()),
            post_json("/v1/evaluate", &request),
        )
        .await;

        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["code"], "invalid_request");
        let message = body["error"]["message"].as_str().unwrap();
        assert!(
            message.contains("predictions length 1") && message.contains("cases length 2"),
            "{message}"
        );
    }

    #[tokio::test]
    async fn evaluate_rejects_unknown_metric() {
        let request = serde_json::json!({
            "cases": [case_json("case-1", "4")],
            "predictions": ["4"],
            "metric": "llm_as_judge",
        });
        let (status, body) = send(
            build_router(RlServiceConfig::default()),
            post_json("/v1/evaluate", &request),
        )
        .await;

        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["code"], "invalid_request");
        assert_eq!(body["error"]["message"], "unsupported metric: llm_as_judge");
    }

    #[tokio::test]
    async fn evaluate_accepts_explicit_exact_match_metric_and_empty_input() {
        let request = EvaluateRequest {
            cases: Vec::new(),
            predictions: Vec::new(),
            answers: None,
            metric: Some("exact_match".to_owned()),
        };
        let (status, body) = send(
            build_router(RlServiceConfig::default()),
            post_json("/v1/evaluate", &request),
        )
        .await;

        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["num_cases"], 0);
        assert_eq!(body["mean_score"], 0.0);
    }

    #[tokio::test]
    async fn rl_task_start_returns_unsupported_envelope() {
        let (status, body) = send(
            build_router(RlServiceConfig::default()),
            post_empty("/v1/rl/tasks/start"),
        )
        .await;

        assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
        assert_eq!(body["error"]["code"], "unsupported");
        let message = body["error"]["message"].as_str().unwrap();
        assert!(message.contains("/v1/rl/tasks/start"), "{message}");
        assert!(message.contains("deferred"), "{message}");
        assert!(message.contains("docs/PARITY.md"), "{message}");
    }

    #[tokio::test]
    async fn trajectory_stats_route_does_not_collide_with_id_route() {
        let router = build_router(RlServiceConfig::default());

        let (status, body) = send(router.clone(), get("/v1/rl/trajectories/stats")).await;
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
        assert_eq!(body["error"]["code"], "unsupported");
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .starts_with("/v1/rl/trajectories/stats "),
            "unexpected message: {body}"
        );

        let (status, body) = send(router.clone(), get("/v1/rl/trajectories")).await;
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .starts_with("/v1/rl/trajectories "),
            "unexpected message: {body}"
        );

        let (status, body) = send(router, get("/v1/rl/trajectories/abc")).await;
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .starts_with("/v1/rl/trajectories/abc "),
            "unexpected message: {body}"
        );
    }

    #[tokio::test]
    async fn internal_completion_routes_return_unsupported() {
        let router = build_router(RlServiceConfig::default());
        for uri in [
            "/internal/v1/completions:before",
            "/internal/v1/completions:after",
            "/internal/v1/completions:discard",
            "/internal/v1/rl/tasks/t1:finish",
            "/internal/v1/rl/tasks/t1:abort",
        ] {
            let (status, body) = send(router.clone(), post_empty(uri)).await;
            assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "uri: {uri}");
            assert_eq!(body["error"]["code"], "unsupported", "uri: {uri}");
            assert!(
                body["error"]["message"].as_str().unwrap().contains(uri),
                "uri: {uri} message: {body}"
            );
        }
    }

    #[tokio::test]
    async fn rl_action_routes_return_unsupported() {
        let router = build_router(RlServiceConfig::default());
        for (method, uri) in [
            ("POST", "/v1/rl/tasks/t1/reward"),
            ("POST", "/v1/rl/training/runs"),
            ("GET", "/v1/rl/training/runs/r1"),
            ("POST", "/v1/rl/training/runs/r1/stop"),
            ("GET", "/v1/rl/tasks/t1"),
            ("POST", "/v1/rl/tasks/t1/stop"),
        ] {
            let request = Request::builder()
                .method(method)
                .uri(uri)
                .body(Body::empty())
                .unwrap();
            let (status, body) = send(router.clone(), request).await;
            assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "uri: {uri}");
            assert!(
                body["error"]["message"].as_str().unwrap().contains(uri),
                "uri: {uri} message: {body}"
            );
        }
    }

    #[tokio::test]
    async fn empty_cases_with_prediction_mismatch_is_rejected() {
        let request = serde_json::json!({
            "cases": [],
            "predictions": ["4"],
        });
        let (status, body) = send(
            build_router(RlServiceConfig::default()),
            post_json("/v1/evaluate", &request),
        )
        .await;

        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["code"], "invalid_request");
    }
}
