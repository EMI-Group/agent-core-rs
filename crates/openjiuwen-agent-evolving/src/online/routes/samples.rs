//! Batch upload and the trajectory read model, both backed by the sample ledger.

use std::collections::BTreeMap;

use axum::body::Bytes;
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::Json;
use serde_json::{json, Map, Value};

use super::super::sample::{SampleRecord, SampleStatus};
use super::{fail, object_body, online_failure, ApiResult, RouteState};

/// Default page size of the trajectory listing (`Query(default=100, ge=1)`).
const DEFAULT_LIST_LIMIT: usize = 100;

/// Every sample status, in the order the unfiltered listing merges them.
const ALL_STATUSES: [SampleStatus; 4] = [
    SampleStatus::Pending,
    SampleStatus::Training,
    SampleStatus::Trained,
    SampleStatus::Failed,
];

/// `POST /v1/gateway/upload/batch` — persist a batch of samples once.
pub(super) async fn upload_batch(
    State(state): State<RouteState>,
    body: Bytes,
) -> ApiResult<Json<Value>> {
    let service = state.online()?;
    let fields = object_body(&body, "invalid_rail_batch")?;
    let protocol_version = fields
        .get("protocol_version")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_owned();
    let Some(Value::Array(samples)) = fields.get("samples") else {
        return Err(fail(
            "invalid_rail_batch",
            StatusCode::BAD_REQUEST,
            "samples must be a list",
        ));
    };
    let limit = state.config().upload_max_samples;
    if samples.len() > limit {
        return Err(fail(
            "invalid_rail_batch",
            StatusCode::BAD_REQUEST,
            format!(
                "batch of {} samples exceeds the {limit}-sample upload limit",
                samples.len()
            ),
        ));
    }

    let mut records = Vec::with_capacity(samples.len());
    for (index, sample) in samples.iter().enumerate() {
        let mut record: SampleRecord = serde_json::from_value(sample.clone()).map_err(|error| {
            fail(
                "invalid_rail_batch",
                StatusCode::BAD_REQUEST,
                format!("sample {index} is malformed: {error}"),
            )
        })?;
        if record.user_id.trim().is_empty() {
            record.user_id.clone_from(&state.config().model_id);
        }
        record.validate().map_err(|error| {
            fail(
                "invalid_rail_batch",
                StatusCode::BAD_REQUEST,
                format!("sample {index}: {error}"),
            )
        })?;
        records.push(record);
    }

    let saved = service
        .ledger()
        .save_once(&records)
        .await
        .map_err(online_failure)?;
    let accepted = saved.len();
    let duplicate = records.len() - accepted;
    Ok(Json(json!({
        "ok": true,
        "result": {
            "accepted": accepted,
            "rejected": 0,
            "duplicate": duplicate,
            "items": [],
            "protocol_version": protocol_version,
        },
    })))
}

/// `GET /v1/rl/trajectories/stats` — ledger counters in Python's shape.
pub(super) async fn trajectory_stats(State(state): State<RouteState>) -> ApiResult<Json<Value>> {
    let service = state.online()?;
    let counters = service.ledger().stats().await.map_err(online_failure)?;
    let mut body = Map::new();
    body.insert("model_id".to_owned(), json!(state.config().model_id));
    body.insert("user_id".to_owned(), json!(state.config().model_id));
    for (key, count) in counters {
        body.insert(key, json!(count));
    }
    Ok(Json(Value::Object(body)))
}

/// `GET /v1/rl/trajectories?status=&limit=` — the ledger listing.
pub(super) async fn list_trajectories(
    State(state): State<RouteState>,
    Query(query): Query<BTreeMap<String, String>>,
) -> ApiResult<Json<Value>> {
    let service = state.online()?;
    let limit = match query.get("limit") {
        None => DEFAULT_LIST_LIMIT,
        Some(raw) => raw.parse::<usize>().map_err(|_| {
            fail(
                "invalid_request",
                StatusCode::BAD_REQUEST,
                format!("limit must be a positive integer: {raw}"),
            )
        })?,
    };
    if limit < 1 {
        return Err(fail(
            "invalid_request",
            StatusCode::BAD_REQUEST,
            "limit must be at least 1",
        ));
    }
    let filter = match query.get("status").map(String::as_str) {
        None | Some("") => None,
        Some("pending") => Some(SampleStatus::Pending),
        Some("training") => Some(SampleStatus::Training),
        Some("trained") => Some(SampleStatus::Trained),
        Some("failed") => Some(SampleStatus::Failed),
        Some(other) => {
            return Err(fail(
                "invalid_request",
                StatusCode::BAD_REQUEST,
                format!("unknown sample status: {other}"),
            ))
        }
    };

    let mut records = if let Some(status) = filter {
        service
            .ledger()
            .list_by_status(status)
            .await
            .map_err(online_failure)?
    } else {
        let mut all = Vec::new();
        for status in ALL_STATUSES {
            all.extend(
                service
                    .ledger()
                    .list_by_status(status)
                    .await
                    .map_err(online_failure)?,
            );
        }
        all.sort_by(|left, right| {
            left.created_seq
                .cmp(&right.created_seq)
                .then_with(|| left.sample_id.cmp(&right.sample_id))
        });
        all
    };
    records.truncate(limit);
    Ok(Json(json!({
        "model_id": state.config().model_id,
        "user_id": state.config().model_id,
        "status": filter.map(|status| status.as_str()),
        "limit": limit,
        "count": records.len(),
        "trajectories": records,
    })))
}

/// `GET /v1/rl/trajectories/{trajectory_id}` — one sample record.
pub(super) async fn get_trajectory(
    State(state): State<RouteState>,
    Path(trajectory_id): Path<String>,
) -> ApiResult<Json<Value>> {
    let service = state.online()?;
    match service
        .ledger()
        .get(&trajectory_id)
        .await
        .map_err(online_failure)?
    {
        Some(record) => Ok(Json(json!(record))),
        None => Err(fail(
            "trajectory_not_found",
            StatusCode::NOT_FOUND,
            format!("trajectory not found: {trajectory_id}"),
        )),
    }
}
