//! Training-run routes: claim a batch, inspect a run, stop a run.

use axum::body::Bytes;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::Json;
use serde_json::Value;

use super::super::training::TrainingRunRecord;
use super::{fail, json_body, online_failure, ApiResult, RouteState};

/// `POST /v1/rl/training/runs` — claim a batch and drive it to a terminal state.
///
/// Python hands the run to a detached executor and answers with the queued
/// record; the port awaits the run, because the offline executor finishes
/// inline, and therefore answers with the terminal record.
pub(super) async fn start_run(
    State(state): State<RouteState>,
    body: Bytes,
) -> ApiResult<(StatusCode, Json<TrainingRunRecord>)> {
    let service = state.online()?;
    match json_body(&body, "invalid_training_request")? {
        Value::Null => {}
        Value::Object(fields) if fields.is_empty() => {}
        _ => {
            return Err(fail(
                "invalid_training_request",
                StatusCode::BAD_REQUEST,
                "Training Run body must be empty",
            ))
        }
    }

    let (claimed, created, _samples) = service.runner().claim().await.map_err(online_failure)?;
    if !created {
        // The service already owns an active run: reuse it, as Python does.
        return Ok((StatusCode::OK, Json(claimed.public())));
    }
    let finished = service
        .runner()
        .run(&claimed.training_run_id)
        .await
        .map_err(online_failure)?;
    Ok((StatusCode::CREATED, Json(finished.public())))
}

/// `GET /v1/rl/training/runs/{training_run_id}` — one run, public shape.
pub(super) async fn get_run(
    State(state): State<RouteState>,
    Path(training_run_id): Path<String>,
) -> ApiResult<Json<TrainingRunRecord>> {
    let service = state.online()?;
    let run = service
        .runner()
        .get(&training_run_id)
        .await
        .map_err(online_failure)?;
    Ok(Json(run.public()))
}

/// `POST /v1/rl/training/runs/{training_run_id}/stop` — cancel an active run.
pub(super) async fn stop_run(
    State(state): State<RouteState>,
    Path(training_run_id): Path<String>,
) -> ApiResult<Json<TrainingRunRecord>> {
    let service = state.online()?;
    let run = service
        .runner()
        .stop(&training_run_id)
        .await
        .map_err(online_failure)?;
    Ok(Json(run.public()))
}
