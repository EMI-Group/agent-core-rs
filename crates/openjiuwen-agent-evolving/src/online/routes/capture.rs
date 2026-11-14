//! The internal completion hooks: `:before`, `:after` and `:discard`.
//!
//! Python parity: `agent_rl/online/capture_pipeline.py`'s `before`, `after` and
//! `discard`, including the request normalization the gateway depends on
//! (`logprobs`, `top_logprobs`, `return_token_ids` and `stream_options.include_usage`
//! are forced on) and the validation that keeps a capture a complete, single
//! choice, policy-matching exchange.

use axum::body::Bytes;
use axum::extract::State;
use axum::http::StatusCode;
use axum::Json;
use serde_json::{json, Map, Value};

use super::super::task::{RewardMode, TaskRecord};
use super::{
    fail, object_body, online_failure, optional_str, required_str, required_value, ApiResult,
    RouteState, MAX_TURN_TRANSITIONS,
};

/// Why a capture request was rejected, before it becomes an HTTP error.
enum CaptureFailure {
    /// The payload is malformed (`400 invalid_completion`).
    Bad(String),
    /// The payload contradicts the task (`409 completion_conflict`).
    Conflict(String),
}

impl CaptureFailure {
    fn into_error(self) -> crate::service::ApiError {
        match self {
            Self::Bad(message) => fail("invalid_completion", StatusCode::BAD_REQUEST, message),
            Self::Conflict(message) => fail("completion_conflict", StatusCode::CONFLICT, message),
        }
    }
}

/// `POST /internal/v1/completions:before` — reserve a capture, normalize the request.
pub(super) async fn before_completion(
    State(state): State<RouteState>,
    body: Bytes,
) -> ApiResult<Json<Value>> {
    let service = state.online()?;
    let fields = object_body(&body, "invalid_completion")?;
    let rl_task_id = required_str(&fields, "rl_task_id", "invalid_completion")?;
    let capture_id = required_str(&fields, "capture_id", "invalid_completion")?;
    let agent_turn_id = optional_str(&fields, "agent_turn_id", "invalid_completion")?;
    let request = required_value(&fields, "request", "invalid_completion")?;

    let task = service
        .registry()
        .get(&rl_task_id)
        .await
        .map_err(online_failure)?;
    let normalized = normalize_request(&task, &capture_id, agent_turn_id.as_deref(), &request)
        .map_err(CaptureFailure::into_error)?;
    let request_fingerprint = super::fingerprint(&normalized);

    for _ in 0..MAX_TURN_TRANSITIONS {
        let reservation = service
            .registry()
            .begin_capture(
                &rl_task_id,
                &capture_id,
                agent_turn_id.as_deref(),
                &request_fingerprint,
            )
            .await
            .map_err(super::capture_failure)?;
        let Some(transition) = reservation else {
            return Ok(Json(json!({ "request": normalized })));
        };
        // A new delayed-feedback turn: publish the open one, then make the new
        // turn current before reserving the capture again.
        super::publish_turn(service, &rl_task_id, &transition.previous).await?;
        service
            .registry()
            .advance_turn(&rl_task_id, &transition.previous, &transition.next)
            .await
            .map_err(super::capture_failure)?;
    }
    Err(fail(
        "completion_conflict",
        StatusCode::CONFLICT,
        "turn transition did not settle",
    ))
}

/// `POST /internal/v1/completions:after` — commit the response and stage its sample.
pub(super) async fn after_completion(
    State(state): State<RouteState>,
    body: Bytes,
) -> ApiResult<StatusCode> {
    let service = state.online()?;
    let fields = object_body(&body, "invalid_completion")?;
    let rl_task_id = required_str(&fields, "rl_task_id", "invalid_completion")?;
    let capture_id = required_str(&fields, "capture_id", "invalid_completion")?;
    let agent_turn_id = optional_str(&fields, "agent_turn_id", "invalid_completion")?;
    let request = required_value(&fields, "request", "invalid_completion")?;
    let response = required_value(&fields, "response", "invalid_completion")?;

    let task = service
        .registry()
        .get(&rl_task_id)
        .await
        .map_err(online_failure)?;
    let normalized_request =
        normalize_request(&task, &capture_id, agent_turn_id.as_deref(), &request)
            .map_err(CaptureFailure::into_error)?;
    let normalized_response =
        normalize_response(&task, &response).map_err(CaptureFailure::into_error)?;
    let sample = build_sample(
        &task,
        &capture_id,
        agent_turn_id.as_deref(),
        &normalized_request,
        &normalized_response,
    );
    let response_fingerprint = super::fingerprint(&normalized_response);
    service
        .registry()
        .commit_capture(
            &rl_task_id,
            &capture_id,
            agent_turn_id.as_deref(),
            &response_fingerprint,
            &sample,
        )
        .await
        .map_err(super::capture_failure)?;
    Ok(StatusCode::NO_CONTENT)
}

/// `POST /internal/v1/completions:discard` — release an abandoned capture.
pub(super) async fn discard_completion(
    State(state): State<RouteState>,
    body: Bytes,
) -> ApiResult<StatusCode> {
    let service = state.online()?;
    let fields = object_body(&body, "invalid_completion")?;
    let rl_task_id = required_str(&fields, "rl_task_id", "invalid_completion")?;
    let capture_id = required_str(&fields, "capture_id", "invalid_completion")?;
    let agent_turn_id = optional_str(&fields, "agent_turn_id", "invalid_completion")?;
    service
        .registry()
        .discard_capture(&rl_task_id, &capture_id, agent_turn_id.as_deref())
        .await
        .map_err(super::capture_failure)?;
    Ok(StatusCode::NO_CONTENT)
}

/// Validates and normalizes an `OpenAI` chat request against `task`'s policy.
fn normalize_request(
    task: &TaskRecord,
    capture_id: &str,
    agent_turn_id: Option<&str>,
    request: &Value,
) -> Result<Value, CaptureFailure> {
    if capture_id.trim().is_empty() {
        return Err(CaptureFailure::Bad("capture_id is required".to_owned()));
    }
    let Value::Object(fields) = request else {
        return Err(CaptureFailure::Bad("request must be an object".to_owned()));
    };
    let model = fields
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if model != task.policy_model {
        return Err(CaptureFailure::Conflict(format!(
            "request model does not match Task policy: {model}"
        )));
    }
    if !matches!(fields.get("messages"), Some(Value::Array(_))) {
        return Err(CaptureFailure::Bad(
            "request messages must be a list".to_owned(),
        ));
    }
    let single_choice = match fields.get("n") {
        None => true,
        Some(Value::Number(number)) => number.as_i64() == Some(1),
        Some(_) => false,
    };
    if !single_choice {
        return Err(CaptureFailure::Bad("RL capture requires n=1".to_owned()));
    }
    if task.reward_mode == RewardMode::DelayedFeedback
        && agent_turn_id.is_none_or(|turn| turn.trim().is_empty())
    {
        return Err(CaptureFailure::Bad(
            "agent_turn_id is required for delayed_feedback Task".to_owned(),
        ));
    }

    let mut normalized = fields.clone();
    normalized.insert("logprobs".to_owned(), Value::Bool(true));
    normalized.insert("top_logprobs".to_owned(), Value::from(1));
    normalized.insert("return_token_ids".to_owned(), Value::Bool(true));
    if normalized.get("stream") == Some(&Value::Bool(true)) {
        let mut options = match normalized.get("stream_options") {
            Some(Value::Object(existing)) => existing.clone(),
            _ => Map::new(),
        };
        options.insert("include_usage".to_owned(), Value::Bool(true));
        normalized.insert("stream_options".to_owned(), Value::Object(options));
    }
    Ok(Value::Object(normalized))
}

/// Validates a complete `OpenAI` chat completion against `task`'s policy.
fn normalize_response(task: &TaskRecord, response: &Value) -> Result<Value, CaptureFailure> {
    let Value::Object(fields) = response else {
        return Err(CaptureFailure::Bad("response must be an object".to_owned()));
    };
    if fields.get("object").and_then(Value::as_str) != Some("chat.completion") {
        return Err(CaptureFailure::Bad(
            "response object must be chat.completion".to_owned(),
        ));
    }
    let model = fields
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if model != task.policy_model {
        return Err(CaptureFailure::Conflict(format!(
            "response model does not match Task policy: {model}"
        )));
    }
    match fields.get("choices") {
        Some(Value::Array(choices)) if choices.len() == 1 => {}
        _ => {
            return Err(CaptureFailure::Bad(
                "response must contain exactly one choice".to_owned(),
            ))
        }
    }
    let Some(Value::Object(usage)) = fields.get("usage") else {
        return Err(CaptureFailure::Bad("response is missing usage".to_owned()));
    };
    let mut counters = [0_u64; 3];
    for (index, field) in ["prompt_tokens", "completion_tokens", "total_tokens"]
        .into_iter()
        .enumerate()
    {
        let Some(count) = usage.get(field).and_then(Value::as_u64) else {
            return Err(CaptureFailure::Bad(format!(
                "response usage {field} must be a non-negative integer"
            )));
        };
        counters[index] = count;
    }
    if counters[0] + counters[1] != counters[2] {
        return Err(CaptureFailure::Bad(
            "response usage total_tokens does not match prompt_tokens + completion_tokens"
                .to_owned(),
        ));
    }
    Ok(response.clone())
}

/// Builds the sample staged for the capture.
fn build_sample(
    task: &TaskRecord,
    capture_id: &str,
    agent_turn_id: Option<&str>,
    request: &Value,
    response: &Value,
) -> Value {
    json!({
        "sample_id": capture_id,
        "trajectory_id": capture_id,
        "rl_task_id": task.rl_task_id,
        "task_id": task.rl_task_id,
        "user_id": task.model_id,
        "session_id": task.agent_session_id,
        "agent_turn_id": agent_turn_id,
        "policy_version": task.policy_lora_name,
        "source": "rl_task",
        "request": request,
        "response": response,
    })
}
