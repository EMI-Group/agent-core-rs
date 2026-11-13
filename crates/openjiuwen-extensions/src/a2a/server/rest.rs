//! The REST (`HTTP+JSON`) binding.
//!
//! Port of the `a2a-sdk` REST app that `a2a_server.py` mounts at `rest_url`
//! (`/a2a/rest`) when the card advertises the `HTTP+JSON` binding. The routes
//! mirror the SDK's v0.3-compatible set under `rest_prefix` (`/v1` by
//! default), and — because both spellings are in use — task actions are also
//! accepted as a `/tasks/{id}/action` sub-path:
//!
//! | Route | Method | Result |
//! |---|---|---|
//! | `{base}/card` | `GET` | the served [`crate::a2a::types::AgentCard`] |
//! | `{base}/message:send`, `{base}/messages:send` | `POST` | a [`crate::a2a::types::Task`] |
//! | `{base}/message:stream` | `POST` | `text/event-stream`, one `data:` frame per event |
//! | `{base}/tasks/{id}` | `GET` | a task, or a `404` envelope |
//! | `{base}/tasks/{id}/cancel`, `{base}/tasks/{id}:cancel` | `POST` | the cancelled task |
//! | `{base}/tasks/{id}/subscribe`, `{base}/tasks/{id}:subscribe` | `GET`, `POST` | SSE: task snapshot, then its current status |
//!
//! The send routes accept the same params object the JSON-RPC binding carries
//! (`{"message": …, "task_id": …, "sessionId": …, …metadata}`), so the two
//! bindings share one request model and one execution path ([`super::ops`]).
//!
//! # Errors
//!
//! A REST failure answers with the HTTP status of the mapped A2A error and the
//! envelope `{"error": {"code": <int>, "status": "<GRPC-ish>", "message":
//! "<msg>"}}`. The mapping lives in [`error_status`]; see the module docs of
//! [`super`] for the table. A known path with an unsupported HTTP method
//! answers `405` with the same envelope, installed through
//! `Router::method_not_allowed_fallback` on this sub-router only, so the
//! JSON-RPC route's behavior is unchanged.
//!
//! Streaming requests validate eagerly: invalid JSON and unusable params return
//! a plain JSON error (with the mapped HTTP status) instead of a broken SSE
//! stream, matching the SDK's eager first-item handling.

use std::sync::Arc;

use axum::body::{Body, Bytes};
use axum::extract::{Path, State};
use axum::http::{header, HeaderValue, Method, StatusCode, Uri};
use axum::response::Response;
use axum::routing::{get, post};
use axum::Router;
use serde::Serialize;
use serde_json::{json, Value};

use super::service::card_handler;
use super::{ops, sse, Inner};
use crate::a2a::jsonrpc::{error_code, JsonRpcError};

/// Builds the REST sub-router; the caller merges it into the main router and
/// applies [`Router::with_state`].
pub(super) fn router(inner: &Inner) -> Router<Arc<Inner>> {
    let base = inner.config.rest_base_path();
    let mut app = Router::new()
        .route(&format!("{base}/card"), get(card_handler))
        .route(&format!("{base}/message:send"), post(send_handler))
        .route(&format!("{base}/messages:send"), post(send_handler))
        .route(&format!("{base}/message:stream"), post(stream_handler))
        // The colon spellings (`/tasks/{id}:cancel`) cannot be expressed as a
        // `matchit` route (a dynamic segment may only be followed by `/` or the
        // end), so one dispatcher splits the segment on its last `:`. Task ids
        // are 32-hex UUIDs, so no task id contains a colon.
        .route(
            &format!("{base}/tasks/{{segment}}"),
            get(tasks_dispatch).post(tasks_dispatch),
        )
        .route(&format!("{base}/tasks/{{id}}/cancel"), post(cancel_subpath))
        .route(
            &format!("{base}/tasks/{{id}}/subscribe"),
            get(subscribe_subpath).post(subscribe_subpath),
        );
    // Bakes the 405 handler into each REST `MethodRouter`; merging afterwards
    // keeps it scoped to these routes.
    app = app.method_not_allowed_fallback(method_not_allowed);
    app
}

/// `POST {base}/message:send` and `{base}/messages:send`.
async fn send_handler(State(inner): State<Arc<Inner>>, body: Bytes) -> Response {
    let params = match parse_params(&body) {
        Ok(params) => params,
        Err(error) => return error_response(&error),
    };
    match ops::send_task(&inner, params.as_ref()).await {
        Ok(task) => json_with_status(StatusCode::OK, &task),
        Err(error) => error_response(&error),
    }
}

/// `POST {base}/message:stream`.
#[allow(clippy::unused_async)]
async fn stream_handler(State(inner): State<Arc<Inner>>, body: Bytes) -> Response {
    let params = match parse_params(&body) {
        Ok(params) => params,
        Err(error) => return error_response(&error),
    };
    match ops::start_stream(&inner, params.as_ref()) {
        Ok(receiver) => sse::value_receiver_response(receiver),
        Err(error) => error_response(&error),
    }
}

/// `POST {base}/tasks/{id}/cancel`.
async fn cancel_subpath(State(inner): State<Arc<Inner>>, Path(id): Path<String>) -> Response {
    cancel_task(&inner, &id).await
}

/// `GET|POST {base}/tasks/{id}/subscribe`.
#[allow(clippy::unused_async)]
async fn subscribe_subpath(State(inner): State<Arc<Inner>>, Path(id): Path<String>) -> Response {
    subscribe_task(&inner, &id)
}

/// `GET|POST {base}/tasks/{segment}`, including the colon action spellings.
async fn tasks_dispatch(
    State(inner): State<Arc<Inner>>,
    Path(segment): Path<String>,
    method: Method,
) -> Response {
    let (task_id, action) = split_action(&segment);
    if method == Method::GET {
        match action {
            "" => stored_task_response(&inner, task_id),
            "subscribe" => subscribe_task(&inner, task_id),
            "cancel" => method_not_allowed_response(&method),
            _ => unknown_action(action),
        }
    } else if method == Method::POST {
        match action {
            "cancel" => cancel_task(&inner, task_id).await,
            "subscribe" => subscribe_task(&inner, task_id),
            "" => method_not_allowed_response(&method),
            _ => unknown_action(action),
        }
    } else {
        method_not_allowed_response(&method)
    }
}

/// Splits `{id}[:action]` on the last colon.
fn split_action(segment: &str) -> (&str, &str) {
    segment.rsplit_once(':').unwrap_or((segment, ""))
}

/// `GET {base}/tasks/{id}`.
fn stored_task_response(inner: &Inner, task_id: &str) -> Response {
    let task_id = match required_task_id(task_id) {
        Ok(task_id) => task_id,
        Err(error) => return error_response(&error),
    };
    match ops::stored_task(inner, task_id) {
        Ok(task) => json_with_status(StatusCode::OK, &task),
        Err(error) => error_response(&error),
    }
}

/// `POST {base}/tasks/{id}[:/]cancel`.
async fn cancel_task(inner: &Inner, task_id: &str) -> Response {
    let task_id = match required_task_id(task_id) {
        Ok(task_id) => task_id,
        Err(error) => return error_response(&error),
    };
    match ops::cancel_stored_task(inner, task_id).await {
        Ok(task) => json_with_status(StatusCode::OK, &task),
        Err(error) => error_response(&error),
    }
}

/// `GET|POST {base}/tasks/{id}[:/]subscribe`: the task snapshot, then its
/// current status.
fn subscribe_task(inner: &Inner, task_id: &str) -> Response {
    let task_id = match required_task_id(task_id) {
        Ok(task_id) => task_id,
        Err(error) => return error_response(&error),
    };
    match ops::task_snapshot(inner, task_id) {
        Ok((task, update)) => {
            sse::value_responses(vec![ops::value_of(&task), ops::value_of(&update)])
        }
        Err(error) => error_response(&error),
    }
}

/// Trims a route task id, reporting `INVALID_PARAMS` when it is empty.
fn required_task_id(task_id: &str) -> std::result::Result<&str, JsonRpcError> {
    let task_id = task_id.trim();
    if task_id.is_empty() {
        return Err(JsonRpcError::new(
            error_code::INVALID_PARAMS,
            "a task id is required",
        ));
    }
    Ok(task_id)
}

/// A `404` envelope for an unrecognized colon action.
fn unknown_action(action: &str) -> Response {
    error_response(&JsonRpcError::new(
        error_code::TASK_NOT_FOUND,
        format!("unknown task action `{action}`"),
    ))
}

/// The `405` fallback installed on the REST sub-router.
async fn method_not_allowed(method: Method, uri: Uri) -> Response {
    error_response(&JsonRpcError::new(
        error_code::METHOD_NOT_FOUND,
        format!("HTTP method {method} is not allowed for {uri}"),
    ))
}

/// A `405` envelope.
fn method_not_allowed_response(method: &Method) -> Response {
    error_response(&JsonRpcError::new(
        error_code::METHOD_NOT_FOUND,
        format!("HTTP method {method} is not allowed on this route"),
    ))
}

/// Parses a REST request body as the shared params object.
fn parse_params(body: &Bytes) -> std::result::Result<Option<Value>, JsonRpcError> {
    if body.is_empty() {
        return Ok(None);
    }
    serde_json::from_slice::<Value>(body)
        .map(Some)
        .map_err(|error| {
            JsonRpcError::new(error_code::PARSE, format!("invalid request body: {error}"))
        })
}

/// The error envelope, with the HTTP status mapped from the A2A error code.
fn error_response(error: &JsonRpcError) -> Response {
    let (status, status_text) = error_status(error.code);
    let body = json!({
        "error": {
            "code": error.code,
            "status": status_text,
            "message": error.message.as_str(),
        }
    });
    json_with_status(status, &body)
}

/// The HTTP status and gRPC-style status string of an A2A error code.
fn error_status(code: i32) -> (StatusCode, &'static str) {
    match code {
        error_code::PARSE | error_code::INVALID_REQUEST | error_code::INVALID_PARAMS => {
            (StatusCode::BAD_REQUEST, "INVALID_ARGUMENT")
        }
        error_code::METHOD_NOT_FOUND => (StatusCode::METHOD_NOT_ALLOWED, "METHOD_NOT_ALLOWED"),
        error_code::TASK_NOT_FOUND => (StatusCode::NOT_FOUND, "NOT_FOUND"),
        error_code::TASK_NOT_CANCELABLE => (StatusCode::CONFLICT, "FAILED_PRECONDITION"),
        error_code::UNSUPPORTED_OPERATION => (StatusCode::NOT_IMPLEMENTED, "UNIMPLEMENTED"),
        error_code::CONTENT_TYPE_NOT_SUPPORTED => {
            (StatusCode::UNSUPPORTED_MEDIA_TYPE, "UNSUPPORTED_MEDIA_TYPE")
        }
        _ => (StatusCode::INTERNAL_SERVER_ERROR, "INTERNAL"),
    }
}

/// Serializes `value` as the body of a JSON response with `status`.
fn json_with_status<T: Serialize + ?Sized>(status: StatusCode, value: &T) -> Response {
    match serde_json::to_vec(value) {
        Ok(body) => json_body(status, body),
        Err(error) => {
            tracing::error!(%error, "failed to serialize a REST response");
            json_body(
                StatusCode::INTERNAL_SERVER_ERROR,
                json!({
                    "error": {
                        "code": error_code::INTERNAL,
                        "status": "INTERNAL",
                        "message": format!("failed to serialize the result: {error}"),
                    }
                })
                .to_string()
                .into_bytes(),
            )
        }
    }
}

/// An `application/json` response with `status` and `body`.
fn json_body(status: StatusCode, body: Vec<u8>) -> Response {
    let mut response = Response::new(Body::from(body));
    *response.status_mut() = status;
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    response
}
