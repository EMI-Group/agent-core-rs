//! axum router and handlers for the mock server.
//!
//! One handler answers both chat-completion paths (`/v1/chat/completions` and
//! the bare `/chat/completions` alias) by reading [`Bytes`] instead of a `Json`
//! extractor, so a malformed body is recorded before the 400 is returned.
//!
//! Handler rules (see the crate `CONTEXT.md`):
//!
//! - exactly one script entry is popped per request, stream or not;
//! - the recording is pushed after the response id is known and before the
//!   response is written, then every `wait_for_requests` caller is woken;
//! - no `std::sync::MutexGuard` is held across an `.await` (it is `!Send`, so
//!   the compiler enforces this);
//! - a bad request never fails the server task.
//!
//! Only chat-completion traffic is recorded: the `/health` probe and the JSON
//! 404s answer without touching the request log, so `request_count` and
//! `last_request` always describe model calls (Python parity: the `httpx`
//! transport fake only ever saw model calls).

use std::collections::BTreeMap;
use std::convert::Infallible;
use std::sync::Arc;

use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{header, HeaderMap, HeaderValue, Method, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::StreamExt;
use openjiuwen_foundation::llm::StreamChunk;
use serde_json::Value;

use super::state::{PoppedResponse, RecordedRequest, ScriptedResponse, ServerState};
use super::wire::{self, MockHttpError};

/// Builds the router for one server instance.
pub(crate) fn router(state: Arc<ServerState>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/chat/completions", post(chat_completions))
        .route("/health", get(health))
        .fallback(not_found)
        .method_not_allowed_fallback(not_found)
        .with_state(state)
}

/// `GET /health` → `{"status":"ok"}`.
async fn health() -> Json<Value> {
    Json(wire::health_body())
}

/// The JSON 404 served for an unknown route or a wrong method.
async fn not_found() -> Response {
    (StatusCode::NOT_FOUND, Json(wire::not_found_body())).into_response()
}

/// `POST /v1/chat/completions` and `POST /chat/completions`.
async fn chat_completions(
    State(state): State<Arc<ServerState>>,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let raw_body = String::from_utf8_lossy(&body).into_owned();
    let parsed = serde_json::from_slice::<Value>(&body).ok();

    let Some(request) = parsed.as_ref().and_then(wire::parse_request) else {
        state.record(recording(
            &method,
            &uri,
            &headers,
            Value::Null,
            &raw_body,
            false,
            None,
            None,
        ));
        return (
            StatusCode::BAD_REQUEST,
            Json(wire::bad_request_body("expected a JSON object")),
        )
            .into_response();
    };

    let stream = request.stream;
    let model = request.model;
    let popped = state.pop();
    let (response_id, response_model) = response_identity(&state, &popped);
    state.record(recording(
        &method,
        &uri,
        &headers,
        parsed.unwrap_or(Value::Null),
        &raw_body,
        stream,
        model,
        response_id.clone(),
    ));

    match &popped.response {
        ScriptedResponse::Error(error) => error_response(error),
        ScriptedResponse::Completion(completion) => {
            let id = response_id.unwrap_or_default();
            if stream {
                sse_response(&state, &id, &response_model, &completion.to_chunks())
            } else {
                (
                    StatusCode::OK,
                    Json(wire::completion_body(&id, &response_model, completion)),
                )
                    .into_response()
            }
        }
        ScriptedResponse::Chunks(chunks) => {
            let id = response_id.unwrap_or_default();
            if stream {
                sse_response(&state, &id, &response_model, chunks)
            } else {
                (
                    StatusCode::OK,
                    Json(wire::chunks_body(&id, &response_model, chunks)),
                )
                    .into_response()
            }
        }
    }
}

/// The `(response_id, model)` pair stamped on the answer of one script.
///
/// A scripted `MockCompletion::id` wins over the deterministic
/// `chatcmpl-<prefix><seq>` id, and a scripted model wins over the configured
/// default. A scripted error stamps no id because its body is the script verbatim.
pub(crate) fn response_identity(
    state: &ServerState,
    popped: &PoppedResponse,
) -> (Option<String>, String) {
    let model = match &popped.response {
        ScriptedResponse::Completion(completion) => completion
            .model
            .clone()
            .unwrap_or_else(|| state.default_model().to_owned()),
        ScriptedResponse::Chunks(_) | ScriptedResponse::Error(_) => {
            state.default_model().to_owned()
        }
    };
    let id = match &popped.response {
        ScriptedResponse::Error(_) => None,
        ScriptedResponse::Completion(completion) => Some(
            completion
                .id
                .clone()
                .unwrap_or_else(|| state.response_id(popped.seq)),
        ),
        ScriptedResponse::Chunks(_) => Some(state.response_id(popped.seq)),
    };
    (id, model)
}

/// Builds the recording of one request.
pub(crate) fn recording(
    method: &Method,
    uri: &Uri,
    headers: &HeaderMap,
    body: Value,
    raw_body: &str,
    stream: bool,
    model: Option<String>,
    response_id: Option<String>,
) -> RecordedRequest {
    let mut recorded_headers: BTreeMap<String, String> = BTreeMap::new();
    for (name, value) in headers {
        let value = String::from_utf8_lossy(value.as_bytes()).into_owned();
        recorded_headers
            .entry(name.as_str().to_owned())
            .and_modify(|existing| {
                existing.push_str(", ");
                existing.push_str(&value);
            })
            .or_insert(value);
    }

    RecordedRequest {
        method: method.as_str().to_owned(),
        path: uri.path().to_owned(),
        query: uri.query().map(str::to_owned),
        headers: recorded_headers,
        body,
        raw_body: raw_body.to_owned(),
        stream,
        model,
        response_id,
    }
}

/// Turns a scripted error into an HTTP response.
fn error_response(error: &MockHttpError) -> Response {
    let status = StatusCode::from_u16(error.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    (status, Json(error.body.clone())).into_response()
}

/// Streams the frames as `text/event-stream`, optionally delaying each one.
fn sse_response(state: &ServerState, id: &str, model: &str, chunks: &[StreamChunk]) -> Response {
    let frames = wire::sse_frames(id, model, chunks);
    let delay = state.chunk_delay();
    let body = futures::stream::iter(frames).then(move |frame| async move {
        if !delay.is_zero() {
            tokio::time::sleep(delay).await;
        }
        Ok::<Bytes, Infallible>(Bytes::from(frame))
    });

    (
        [(
            header::CONTENT_TYPE,
            HeaderValue::from_static("text/event-stream"),
        )],
        Body::from_stream(body),
    )
        .into_response()
}
