//! Task lifecycle routes and the internal finish/abort actions.

use axum::body::Bytes;
use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::Json;
use serde_json::Value;

use super::super::task::{FinishReason, RewardMode, TaskRecord, TaskSpec};
use super::{
    fail, header_str, object_body, online_failure, require_only, task_failure, ApiResult,
    RouteState,
};

/// Header naming the agent session the task captures.
const SESSION_HEADER: &str = "x-agent-session-id";
/// Header carrying the gateway's RL task id.
const TASK_HEADER: &str = "x-aigw-rl-task-id";
/// Header carrying the policy (`LoRA`) name the samples are rolled out on.
const POLICY_NAME_HEADER: &str = "x-aigw-rl-policy-name";
/// Header carrying the policy model name the policy server answers with.
const POLICY_MODEL_HEADER: &str = "x-aigw-rl-policy-model";

/// `POST /v1/rl/tasks/start` — create or adopt the session's active task.
pub(super) async fn start_task(
    State(state): State<RouteState>,
    headers: HeaderMap,
    body: Bytes,
) -> ApiResult<(StatusCode, Json<TaskRecord>)> {
    let service = state.online()?;
    let Some(session_id) = header_str(&headers, SESSION_HEADER) else {
        return Err(fail(
            "missing_session_id",
            StatusCode::BAD_REQUEST,
            "X-Agent-Session-Id is required",
        ));
    };
    let fields = object_body(&body, "invalid_task_request")?;
    require_only(
        &fields,
        &["reward_mode"],
        "invalid_task_request",
        "body must contain only reward_mode",
    )?;
    let reward_mode = match fields.get("reward_mode").and_then(Value::as_str) {
        Some("terminal") => RewardMode::Terminal,
        Some("delayed_feedback") => RewardMode::DelayedFeedback,
        _ => {
            return Err(fail(
                "invalid_reward_mode",
                StatusCode::BAD_REQUEST,
                "reward_mode must be terminal or delayed_feedback",
            ))
        }
    };
    let task_id = header_str(&headers, TASK_HEADER).ok_or_else(|| {
        fail(
            "invalid_task_request",
            StatusCode::BAD_REQUEST,
            "X-AIGW-RL-Task-Id is required",
        )
    })?;
    let policy_name = header_str(&headers, POLICY_NAME_HEADER).ok_or_else(|| {
        fail(
            "invalid_task_request",
            StatusCode::BAD_REQUEST,
            "X-AIGW-RL-Policy-Name is required",
        )
    })?;
    let policy_model = header_str(&headers, POLICY_MODEL_HEADER).ok_or_else(|| {
        fail(
            "invalid_task_request",
            StatusCode::BAD_REQUEST,
            "X-AIGW-RL-Policy-Model is required",
        )
    })?;

    let spec = TaskSpec {
        rl_task_id: task_id,
        agent_session_id: session_id,
        model_id: state.config().model_id.clone(),
        policy_lora_name: policy_name,
        reward_mode,
        policy_model: Some(policy_model),
    };
    let (record, created) = service
        .registry()
        .start(&spec)
        .await
        .map_err(task_failure)?;
    let status = if created {
        StatusCode::CREATED
    } else {
        StatusCode::OK
    };
    Ok((status, Json(record)))
}

/// `GET /v1/rl/tasks/{rl_task_id}` — the durable task record.
pub(super) async fn get_task(
    State(state): State<RouteState>,
    Path(rl_task_id): Path<String>,
) -> ApiResult<Json<TaskRecord>> {
    let service = state.online()?;
    let record = service
        .registry()
        .get(&rl_task_id)
        .await
        .map_err(task_failure)?;
    Ok(Json(record))
}

/// `POST /v1/rl/tasks/{rl_task_id}/stop` — finalize the task for the user.
pub(super) async fn stop_task(
    State(state): State<RouteState>,
    Path(rl_task_id): Path<String>,
) -> ApiResult<Json<TaskRecord>> {
    let service = state.online()?;
    let record = service
        .registry()
        .finish(&rl_task_id, FinishReason::UserStopped)
        .await
        .map_err(task_failure)?;
    Ok(Json(record))
}

/// `POST /v1/rl/tasks/{rl_task_id}/reward` — project one terminal reward.
pub(super) async fn reward_task(
    State(state): State<RouteState>,
    Path(rl_task_id): Path<String>,
    body: Bytes,
) -> ApiResult<Json<Value>> {
    let service = state.online()?;
    let fields = object_body(&body, "invalid_reward_request")?;
    require_only(
        &fields,
        &["reward"],
        "invalid_reward_request",
        "body must contain only reward",
    )?;
    let reward = fields
        .get("reward")
        .and_then(Value::as_f64)
        .filter(|reward| reward.is_finite() && (0.0..=1.0).contains(reward))
        .ok_or_else(|| {
            fail(
                "invalid_reward",
                StatusCode::BAD_REQUEST,
                "reward must be a finite number between 0 and 1",
            )
        })?;
    let sample_count = service
        .registry()
        .record_reward(&rl_task_id, reward)
        .await
        .map_err(online_failure)?;
    Ok(Json(serde_json::json!({"sample_count": sample_count})))
}

/// `POST /internal/v1/rl/tasks/{*operation}` — `{id}:finish` and `{id}:abort`.
pub(super) async fn internal_task_operation(
    State(state): State<RouteState>,
    Path(operation): Path<String>,
    body: Bytes,
) -> ApiResult<Json<TaskRecord>> {
    let Some((rl_task_id, action)) = operation.rsplit_once(':') else {
        return Err(fail(
            "not_found",
            StatusCode::NOT_FOUND,
            format!("unknown internal operation: {operation}"),
        ));
    };
    match action {
        "finish" => finish_task(&state, rl_task_id, &body).await,
        "abort" => abort_task(&state, rl_task_id, &body).await,
        other => Err(fail(
            "not_found",
            StatusCode::NOT_FOUND,
            format!("unknown internal operation: {other}"),
        )),
    }
}

/// `.../{id}:finish` — the service stops a task (`service_stopped` only).
async fn finish_task(
    state: &RouteState,
    rl_task_id: &str,
    body: &[u8],
) -> ApiResult<Json<TaskRecord>> {
    let service = state.online()?;
    let fields = object_body(body, "invalid_finish_reason")?;
    require_only(
        &fields,
        &["reason"],
        "invalid_finish_reason",
        "reason must be service_stopped",
    )?;
    if fields.get("reason").and_then(Value::as_str) != Some("service_stopped") {
        return Err(fail(
            "invalid_finish_reason",
            StatusCode::BAD_REQUEST,
            "reason must be service_stopped",
        ));
    }
    let record = service
        .registry()
        .finish(rl_task_id, FinishReason::ServiceStopped)
        .await
        .map_err(task_failure)?;
    Ok(Json(record))
}

/// `.../{id}:abort` — the service aborts a task (`capture_failed`/`timeout`).
async fn abort_task(
    state: &RouteState,
    rl_task_id: &str,
    body: &[u8],
) -> ApiResult<Json<TaskRecord>> {
    let service = state.online()?;
    let fields = object_body(body, "invalid_abort_reason")?;
    let reason = match fields.get("reason").and_then(Value::as_str) {
        Some("capture_failed") if fields.len() == 1 => FinishReason::CaptureFailed,
        Some("timeout") if fields.len() == 1 => FinishReason::Timeout,
        _ => {
            return Err(fail(
                "invalid_abort_reason",
                StatusCode::BAD_REQUEST,
                "reason must be capture_failed or timeout",
            ))
        }
    };
    let record = service
        .registry()
        .abort(rl_task_id, reason)
        .await
        .map_err(task_failure)?;
    Ok(Json(record))
}
