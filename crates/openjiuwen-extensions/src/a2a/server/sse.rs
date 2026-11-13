//! Server-sent-event framing shared by the JSON-RPC and REST bindings.
//!
//! The JSON-RPC binding wraps every event in a JSON-RPC success envelope
//! ([`frame`]); the REST binding streams the event payload itself
//! ([`value_frame`]), because the `a2a-sdk` REST dispatcher emits bare
//! `data: <json>` frames. Both use the same `\n\n` frame terminator and the
//! same `text/event-stream` headers, so only the payload shape differs.

use std::convert::Infallible;

use axum::body::{Body, Bytes};
use axum::http::{header, HeaderValue};
use axum::response::Response;
use futures::{Stream, StreamExt};
use serde_json::Value;
use tokio::sync::mpsc;

use crate::a2a::jsonrpc::{self, JsonRpcResponse};

/// One `text/event-stream` frame holding a `JSON-RPC` response.
pub(super) fn frame(response: &JsonRpcResponse) -> Bytes {
    Bytes::from(format!("data: {}\n\n", jsonrpc::serialize(response)))
}

/// Wraps a stream of bytes as a `text/event-stream` response.
fn stream_response<S>(frames: S) -> Response
where
    S: Stream<Item = std::result::Result<Bytes, Infallible>> + Send + 'static,
{
    let mut response = Response::new(Body::from_stream(frames));
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/event-stream"),
    );
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    response
}

/// Wraps a stream of `JSON-RPC` responses as an SSE response.
pub(super) fn response<S>(responses: S) -> Response
where
    S: Stream<Item = JsonRpcResponse> + Send + 'static,
{
    stream_response(responses.map(|response| Ok::<Bytes, Infallible>(frame(&response))))
}

/// An SSE response that ends after `responses` (used by `SubscribeToTask`).
pub(super) fn responses(responses: Vec<JsonRpcResponse>) -> Response {
    response(futures::stream::iter(responses))
}

/// An SSE response carrying one `JSON-RPC` result frame per value received
/// from `receiver`.
pub(super) fn receiver_response(receiver: mpsc::UnboundedReceiver<Value>, id: Value) -> Response {
    let mut receiver = receiver;
    let responses = async_stream::stream! {
        while let Some(value) = receiver.recv().await {
            yield JsonRpcResponse::success(id.clone(), value);
        }
    };
    response(responses)
}

/// One `text/event-stream` frame holding a bare JSON value.
pub(super) fn value_frame(value: &Value) -> Bytes {
    Bytes::from(format!("data: {value}\n\n"))
}

/// Wraps a stream of JSON values as a `text/event-stream` response.
pub(super) fn value_stream<S>(values: S) -> Response
where
    S: Stream<Item = Value> + Send + 'static,
{
    stream_response(values.map(|value| Ok::<Bytes, Infallible>(value_frame(&value))))
}

/// An SSE response that ends after `values`.
pub(super) fn value_responses(values: Vec<Value>) -> Response {
    value_stream(futures::stream::iter(values))
}

/// An SSE response carrying one bare-value frame per value received from
/// `receiver`.
pub(super) fn value_receiver_response(mut receiver: mpsc::UnboundedReceiver<Value>) -> Response {
    let values = async_stream::stream! {
        while let Some(value) = receiver.recv().await {
            yield value;
        }
    };
    value_stream(values)
}
