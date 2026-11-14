//! HTTP service of the Python online-RL (`agent_rl`) service layer.
//!
//! Python ships the RL Service as a FastAPI app over Redis, vLLM/verl training
//! executors and a rewards gateway. The Rust port keeps the transport contract
//! and the same state machines, backed by an in-memory store and the
//! deterministic offline training executor (see `docs/PARITY.md` for what stays
//! deferred).
//!
//! # Endpoints
//!
//! | Method | Path | Behaviour |
//! |---|---|---|
//! | `GET` | `/health` | `200 {"status":"ready","model_id":...}`, or `503 service_not_ready` when the service is not ready or its executor cannot train |
//! | `POST` | `/v1/evaluate` | offline exact-match scoring over `cases` (or an `answers` map) |
//! | `POST` | `/v1/rl/tasks/start` | `201` created / `200` adopted task record |
//! | `GET` | `/v1/rl/tasks/{rl_task_id}` | task record, `404 task_not_found` |
//! | `POST` | `/v1/rl/tasks/{rl_task_id}/stop` | `finish(UserStopped)` |
//! | `POST` | `/v1/rl/tasks/{rl_task_id}/reward` | `{"sample_count": n}` |
//! | `POST` | `/v1/rl/training/runs` | `201` new / `200` active run, `409 insufficient_samples` |
//! | `GET` | `/v1/rl/training/runs/{id}` | public training-run record |
//! | `POST` | `/v1/rl/training/runs/{id}/stop` | canceled run record |
//! | `GET` | `/v1/rl/trajectories/stats` | sample-ledger counters |
//! | `GET` | `/v1/rl/trajectories` | sample-ledger listing (`status`, `limit`) |
//! | `GET` | `/v1/rl/trajectories/{id}` | one sample record |
//! | `POST` | `/v1/gateway/upload/batch` | persist a sample batch once |
//! | `POST` | `/internal/v1/completions:before` | reserve a capture, normalize the request |
//! | `POST` | `/internal/v1/completions:after` | commit the response, `204` |
//! | `POST` | `/internal/v1/completions:discard` | release the capture, `204` |
//! | `POST` | `/internal/v1/rl/tasks/{*operation}` | serves `.../{id}:finish` and `.../{id}:abort` |
//!
//! The RL routes are implemented in [`crate::online::routes`] over the state
//! machines of [`crate::online`]; this module owns the configuration, the
//! envelope, the router assembly and the two routes that need no state machine.
//!
//! # Error contract
//!
//! Every failure uses the Python envelope
//! `{"error": {"code": "...", "message": "..."}}`:
//!
//! - `invalid_request` — `400`, malformed or inconsistent request payloads.
//! - the state-machine codes — `400`/`404`/`409`, one per rule (`task_conflict`,
//!   `turn_closed`, `completion_conflict`, `insufficient_samples`, ...).
//! - `service_not_ready` — `503`, the service is not accepting work yet, or its
//!   configured executor cannot train.
//! - `internal_error` — `500`, an unexpected evaluation or storage failure.
//! - `unsupported` — `501`, a deferred capability the message names and points
//!   at `docs/PARITY.md` for.
//!
//! The router is pure: building and exercising it binds no socket and needs no
//! Redis, no model, and no credentials.

use std::collections::BTreeMap;
use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
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
use crate::online::routes::{rl_router, RlServiceState, RouteState};
use crate::online::OnlineError;

/// The only metric the offline service implements.
const SUPPORTED_METRIC: &str = "exact_match";

/// The caller-supplied run identifier used by the evaluation endpoint.
const SERVICE_RUN_ID: &str = "service";

/// The reason appended to every `unsupported` message.
const DEFERRED_REASON: &str =
    "is not supported: the Python RL/ML service stack is deferred; see docs/PARITY.md";

/// Configuration of the RL service.
///
/// Python: `openjiuwen.agent_evolving.agent_rl.rl_service.config.RLServiceConfig`
/// (reduced — the Redis URL and the vendor gateway fields belong to deferred
/// scope; the batching, activation and executor fields are ported).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct RlServiceConfig {
    /// Identifier of the model the service serves; echoed by `/health` and
    /// stamped on every task it starts.
    pub model_id: String,
    /// Host the listener binds to.
    pub listen_host: String,
    /// Port the listener binds to.
    pub listen_port: u16,
    /// Whether the service reports itself ready.
    pub ready: bool,
    /// Pending samples a training run needs before it may claim a batch.
    pub min_samples_for_training: usize,
    /// Largest batch one training run may claim.
    pub max_samples_per_run: usize,
    /// Activate a succeeded run's artifact through the activator seam.
    pub auto_activate_lora: bool,
    /// Root of the `LoRA` artifact repository and of the local executor.
    pub lora_root: Option<String>,
    /// Training executor to drive: `"local"` or `"ppo"`.
    pub executor: String,
    /// Largest number of samples one upload batch may carry.
    pub upload_max_samples: usize,
}

impl Default for RlServiceConfig {
    fn default() -> Self {
        Self {
            model_id: String::new(),
            listen_host: "127.0.0.1".to_owned(),
            listen_port: 18_081,
            ready: true,
            min_samples_for_training: 1,
            max_samples_per_run: 8,
            auto_activate_lora: true,
            lora_root: None,
            executor: "local".to_owned(),
            upload_max_samples: 256,
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

    /// Builds an error from an explicit code, message and status.
    pub fn with_status(
        code: impl Into<String>,
        message: impl Into<String>,
        status: StatusCode,
    ) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            status,
        }
    }

    /// Builds a `503 service_not_ready` error naming the reason.
    pub fn service_not_ready_because(message: impl Into<String>) -> Self {
        Self::with_status(
            "service_not_ready",
            message,
            StatusCode::SERVICE_UNAVAILABLE,
        )
    }

    /// The envelope of an online-RL state-machine failure.
    ///
    /// The code and the status both come from [`OnlineError`], so a rule decided
    /// in the state machine is never re-mapped at the transport.
    pub fn from_online(error: OnlineError) -> Self {
        let status =
            StatusCode::from_u16(error.http_status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        Self {
            code: error.code().to_owned(),
            message: error.to_string(),
            status,
        }
    }

    /// The envelope of a kernel failure raised outside a state machine.
    ///
    /// A deferred capability keeps the [`DEFERRED_REASON`] wording and answers
    /// `501`; a validation failure is a `400`; a missing entity is a `404`;
    /// everything else is an internal error.
    pub fn from_error(error: Error) -> Self {
        match error {
            Error::Unsupported(message) => Self::unsupported(message),
            Error::Validation(message) => Self::invalid_request(message),
            Error::NotFound(message) => {
                Self::with_status("not_found", message, StatusCode::NOT_FOUND)
            }
            other => Self::internal(other.to_string()),
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

/// Builds the service router over a fresh in-memory state.
///
/// The state is built from `config`; when the configuration cannot produce one
/// (an unknown `executor`, incoherent batching limits) the same route table
/// answers `503 service_not_ready` naming the reason, so a misconfiguration is a
/// transport error rather than a panic or a silent fallback.
///
/// The router is built once per server, and also per request in tests; it never
/// performs I/O of its own.
pub fn build_router(config: RlServiceConfig) -> Router {
    match RlServiceState::new(&config) {
        Ok(state) => build_router_with_state(config, state),
        Err(error) => router(RouteState::unavailable(
            config,
            ApiError::service_not_ready_because(error.to_string()),
        )),
    }
}

/// Builds the service router over an explicitly supplied [`RlServiceState`].
///
/// Use this to share one state across routers, or to inspect the state (task
/// registry, ledger, training runner, checkpoint export) from the caller that
/// serves it.
pub fn build_router_with_state(config: RlServiceConfig, state: RlServiceState) -> Router {
    router(RouteState::ready(config, state))
}

/// Assembles the two stateless routes and the online-RL route table.
fn router(state: RouteState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/evaluate", post(evaluate))
        .merge(rl_router())
        .with_state(state)
}

/// Binds `config.listen_host:config.listen_port` and serves the router.
///
/// The state is the one [`build_router`] builds from `config`, so the default
/// configuration serves the deterministic local executor; a `ppo` executor
/// serves the same routes and answers `/health` with `503`, because it cannot
/// train in this build.
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
///
/// Ready means "this process can accept RL work": the configuration says so, the
/// state could be built, and the executor can actually train. A `ppo` service
/// therefore answers `503` with the executor's reason instead of advertising a
/// GPU stack this build does not link.
async fn health(State(state): State<RouteState>) -> std::result::Result<Json<Value>, ApiError> {
    if !state.config().ready {
        return Err(ApiError::service_not_ready());
    }
    if let Err(error) = state.online()?.check_health().await {
        return Err(ApiError::service_not_ready_because(error.to_string()));
    }
    Ok(Json(json!({
        "status": "ready",
        "model_id": state.config().model_id,
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
    use serde_json::{json, Value};
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

    /// Builds a request of any method with an optional JSON body.
    fn request_with(method: &str, uri: &str, body: Option<&Value>) -> Request<Body> {
        let builder = Request::builder().method(method).uri(uri);
        match body {
            Some(body) => builder
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_vec(body).unwrap()))
                .unwrap(),
            None => builder.body(Body::empty()).unwrap(),
        }
    }

    /// Builds a `/v1/rl/tasks/start` request carrying every gateway header.
    fn start_task_request(
        task_id: &str,
        policy_name: &str,
        policy_model: &str,
        reward_mode: &str,
    ) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri("/v1/rl/tasks/start")
            .header(CONTENT_TYPE, "application/json")
            .header("x-agent-session-id", "session-1")
            .header("x-aigw-rl-task-id", task_id)
            .header("x-aigw-rl-policy-name", policy_name)
            .header("x-aigw-rl-policy-model", policy_model)
            .body(Body::from(
                serde_json::to_vec(&json!({"reward_mode": reward_mode})).unwrap(),
            ))
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
        assert_eq!(config.min_samples_for_training, 1);
        assert_eq!(config.max_samples_per_run, 8);
        assert!(config.auto_activate_lora);
        assert!(config.lora_root.is_none());
        assert_eq!(config.executor, "local");
        assert_eq!(config.upload_max_samples, 256);

        let restored: RlServiceConfig =
            serde_json::from_value(serde_json::json!({"model_id": "m", "unknown": 1})).unwrap();
        assert_eq!(restored.model_id, "m");
        assert_eq!(restored.listen_host, "127.0.0.1");
        assert!(restored.ready);
        assert_eq!(restored.executor, "local");
        assert_eq!(restored.max_samples_per_run, 8);
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
    async fn rl_task_start_validates_its_headers_and_body() {
        let router = build_router(RlServiceConfig::default());

        // No `X-Agent-Session-Id`: the session owns the capture, so it is
        // checked before the body.
        let (status, body) = send(
            router.clone(),
            post_json("/v1/rl/tasks/start", &json!({"reward_mode": "terminal"})),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["code"], "missing_session_id");

        // The session header alone is not enough: the gateway identity is required.
        let session_only = serde_json::to_vec(&json!({"reward_mode": "terminal"})).unwrap();
        let request = Request::builder()
            .method("POST")
            .uri("/v1/rl/tasks/start")
            .header(CONTENT_TYPE, "application/json")
            .header("x-agent-session-id", "session-1")
            .body(Body::from(session_only))
            .unwrap();
        let (status, body) = send(router.clone(), request).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["code"], "invalid_task_request");

        // A body carrying more than `reward_mode` is rejected as a whole.
        let extra_field = serde_json::to_vec(&json!({"reward_mode": "terminal", "n": 2})).unwrap();
        let request = Request::builder()
            .method("POST")
            .uri("/v1/rl/tasks/start")
            .header(CONTENT_TYPE, "application/json")
            .header("x-agent-session-id", "session-1")
            .header("x-aigw-rl-task-id", "task-1")
            .header("x-aigw-rl-policy-name", "policy-a")
            .header("x-aigw-rl-policy-model", "pol-model")
            .body(Body::from(extra_field))
            .unwrap();
        let (status, body) = send(router.clone(), request).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["code"], "invalid_task_request");

        // An unknown reward mode is its own code.
        let (status, body) = send(
            router,
            start_task_request("task-1", "policy-a", "pol-model", "per_turn"),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["code"], "invalid_reward_mode");
    }

    #[tokio::test]
    async fn rl_task_and_trajectory_unknown_ids_report_not_found() {
        let router = build_router(RlServiceConfig::default());

        let (status, body) = send(router.clone(), get("/v1/rl/tasks/task-1")).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["error"]["code"], "task_not_found");

        // The static `stats` route must not be shadowed by the `{id}` route.
        let (status, body) = send(router.clone(), get("/v1/rl/trajectories/stats")).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["total_samples"], 0);

        let (status, body) = send(router, get("/v1/rl/trajectories/traj-1")).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["error"]["code"], "trajectory_not_found");
    }

    #[tokio::test]
    async fn internal_routes_validate_their_payloads() {
        let router = build_router(RlServiceConfig::default());

        let (status, body) = send(
            router.clone(),
            post_json("/internal/v1/completions:before", &json!({})),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["code"], "invalid_completion");

        // An internal operation with no `:action` suffix is not a task action.
        let (status, body) = send(router, post_empty("/internal/v1/rl/tasks/t1")).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["error"]["code"], "not_found");
    }

    #[tokio::test]
    async fn rl_action_routes_report_their_state_errors() {
        let router = build_router(RlServiceConfig::default());
        let reward = Some(json!({"reward": 0.5}));
        for (method, uri, payload, code) in [
            ("POST", "/v1/rl/tasks/t1/reward", reward, "task_not_found"),
            ("POST", "/v1/rl/training/runs", None, "invalid_request"),
            (
                "GET",
                "/v1/rl/training/runs/r1",
                None,
                "training_run_not_found",
            ),
            (
                "POST",
                "/v1/rl/training/runs/r1/stop",
                None,
                "training_run_not_found",
            ),
            ("POST", "/v1/rl/tasks/t1/stop", None, "task_not_found"),
        ] {
            let request = request_with(method, uri, payload.as_ref());
            let (status, body) = send(router.clone(), request).await;
            assert_eq!(body["error"]["code"], code, "uri: {uri} body: {body}");
            assert!(status.is_client_error(), "uri: {uri} status: {status}");
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
