//! Binding-neutral server operations shared by the JSON-RPC and REST bindings.
//!
//! The two wire bindings differ only in framing and error envelopes: deriving
//! a send plan, running the executor, storing the task, snapshotting a task,
//! and cancelling one are identical. Everything in this module is
//! `pub(super)` so both [`super::service`] and [`super::rest`] drive the same
//! code path; the bindings stay thin adapters over these operations.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::Response;
use openjiuwen_common::error::Result;
use serde::Serialize;
use serde_json::{Map, Value};
use tokio::sync::mpsc;

use super::event_sink::EventSink;
use super::{agent_message, fresh_id, Inner};
use crate::a2a::jsonrpc::{error_code, JsonRpcError};
use crate::a2a::transform::A2aTransformer;
use crate::a2a::types::{
    CancelTaskRequest, Message, Part, Role, Task, TaskState, TaskStatus, TaskStatusUpdateEvent,
};

/// The agent message published together with the `WORKING` status.
const WORKING_MESSAGE: &str = "Processing your request...";

/// The payload keys consumed by the request mapping, never copied into the
/// execution metadata (Python's exclusion set).
const RESERVED_PAYLOAD_KEYS: [&str; 3] = ["query", "task_id", "sessionId"];

/// Everything a send call needs once the ids and message are derived.
pub(super) struct SendPlan {
    /// The publisher the executor drives.
    pub(super) sink: EventSink,
    /// The event stream the sink publishes into.
    pub(super) receiver: mpsc::UnboundedReceiver<Value>,
    /// The per-turn execution context.
    pub(super) context: super::config::A2aExecutionContext,
    /// The request message included in the `SUBMITTED` snapshot.
    pub(super) message: Message,
}

/// Derives the ids, message, execution context, and sink of a send call.
pub(super) fn plan_send(
    inner: &Inner,
    params: Option<&Value>,
) -> std::result::Result<SendPlan, JsonRpcError> {
    let params = params.cloned().unwrap_or(Value::Null);
    let payload = A2aTransformer::from_request(&params)
        .map_err(|error| JsonRpcError::new(error_code::INVALID_PARAMS, error.to_string()))?;
    let task_id = match payload.get("task_id").and_then(Value::as_str) {
        Some(id) if !id.trim().is_empty() => id.trim().to_owned(),
        _ => fresh_id(),
    };
    let context_id = match payload.get("sessionId").and_then(Value::as_str) {
        Some(id) if !id.trim().is_empty() => id.trim().to_owned(),
        _ => fresh_id(),
    };
    let metadata = payload_metadata(&payload);
    let message = request_message(&params, &payload, &task_id, &context_id);
    let (sink, receiver) = EventSink::new(
        task_id.clone(),
        context_id.clone(),
        Arc::clone(&inner.store),
    );
    Ok(SendPlan {
        sink,
        receiver,
        context: super::config::A2aExecutionContext {
            task_id,
            context_id,
            message: message.clone(),
            metadata,
        },
        message,
    })
}

/// The openJiuwen payload entries that carry into the execution metadata.
fn payload_metadata(payload: &Value) -> Map<String, Value> {
    let mut metadata = Map::new();
    if let Some(object) = payload.as_object() {
        for (key, value) in object {
            if !RESERVED_PAYLOAD_KEYS.contains(&key.as_str()) {
                metadata.insert(key.clone(), value.clone());
            }
        }
    }
    metadata
}

/// The request [`Message`], decoded from the params or synthesized.
fn request_message(params: &Value, payload: &Value, task_id: &str, context_id: &str) -> Message {
    let Some(wire) = params.get("message") else {
        return synthesized_message(payload, task_id, context_id);
    };
    match serde_json::from_value::<Message>(wire.clone()) {
        Ok(message) => message,
        Err(error) => {
            tracing::debug!(
                %error,
                "A2A request message did not decode; synthesizing it from the payload"
            );
            synthesized_message(payload, task_id, context_id)
        }
    }
}

/// Builds the request message a bare openJiuwen payload implies.
fn synthesized_message(payload: &Value, task_id: &str, context_id: &str) -> Message {
    let parts = payload
        .get("query")
        .and_then(Value::as_str)
        .map_or_else(Vec::new, |query| vec![Part::text(query)]);
    Message {
        message_id: fresh_id(),
        context_id: Some(context_id.to_owned()),
        task_id: Some(task_id.to_owned()),
        role: Role::User,
        parts,
        metadata: payload_metadata(payload),
        ..Message::default()
    }
}

/// Publishes the `SUBMITTED` snapshot and the `WORKING` status.
fn publish_prelude(
    sink: &EventSink,
    message: &Message,
    context: &super::config::A2aExecutionContext,
) {
    sink.task(Task {
        id: context.task_id.clone(),
        context_id: context.context_id.clone(),
        status: TaskStatus {
            state: TaskState::Submitted,
            message: None,
            timestamp: None,
        },
        artifacts: Vec::new(),
        history: vec![message.clone()],
        metadata: Map::new(),
    });
    sink.working(Some(agent_message(
        WORKING_MESSAGE,
        &context.task_id,
        &context.context_id,
    )));
}

/// Runs the executor and publishes the terminal status when it left the task open.
async fn run_executor(
    executor: Arc<dyn super::config::A2aExecutor>,
    sink: EventSink,
    context: super::config::A2aExecutionContext,
) -> Result<()> {
    let outcome = executor.execute(context, &sink).await;
    match &outcome {
        Ok(()) => {
            if !sink.is_terminal() {
                sink.complete();
            }
        }
        Err(error) => {
            tracing::warn!(
                %error,
                task_id = sink.task_id(),
                "A2A executor failed; publishing FAILED"
            );
            sink.fail(error.to_string());
        }
    }
    outcome
}

/// Runs one non-streaming send and returns the stored task.
pub(super) async fn send_task(
    inner: &Inner,
    params: Option<&Value>,
) -> std::result::Result<Task, JsonRpcError> {
    let plan = plan_send(inner, params)?;
    let SendPlan {
        sink,
        receiver: _receiver,
        context,
        message,
    } = plan;
    let task_id = context.task_id.clone();
    publish_prelude(&sink, &message, &context);
    run_executor(Arc::clone(&inner.executor), sink, context)
        .await
        .map_err(|error| {
            JsonRpcError::new(
                error_code::INTERNAL,
                format!("A2A executor failed: {error}"),
            )
        })?;
    stored_task(inner, &task_id)
}

/// Starts one streaming send: publishes through the sink and returns the event
/// stream the caller frames.
pub(super) fn start_stream(
    inner: &Inner,
    params: Option<&Value>,
) -> std::result::Result<mpsc::UnboundedReceiver<Value>, JsonRpcError> {
    let plan = plan_send(inner, params)?;
    let SendPlan {
        sink,
        receiver,
        context,
        message,
    } = plan;
    let executor = Arc::clone(&inner.executor);
    tokio::spawn(async move {
        publish_prelude(&sink, &message, &context);
        if let Err(error) = run_executor(executor, sink, context).await {
            tracing::debug!(
                %error,
                "A2A streaming execution failed after FAILED was published"
            );
        }
    });
    Ok(receiver)
}

/// The stored task plus the status snapshot a subscribe route publishes.
pub(super) fn task_snapshot(
    inner: &Inner,
    task_id: &str,
) -> std::result::Result<(Task, TaskStatusUpdateEvent), JsonRpcError> {
    let task = stored_task(inner, task_id)?;
    let update = TaskStatusUpdateEvent {
        task_id: task.id.clone(),
        context_id: task.context_id.clone(),
        status: task.status.clone(),
        metadata: Map::new(),
    };
    Ok((task, update))
}

/// The stored task with `task_id`, or a `TASK_NOT_FOUND` error.
pub(super) fn stored_task(inner: &Inner, task_id: &str) -> std::result::Result<Task, JsonRpcError> {
    inner
        .store
        .get(task_id)
        .ok_or_else(|| task_not_found(task_id))
}

/// Asks the executor to cancel the stored task, then publishes `CANCELED`.
pub(super) async fn cancel_stored_task(
    inner: &Inner,
    task_id: &str,
) -> std::result::Result<Task, JsonRpcError> {
    let task = stored_task(inner, task_id)?;
    if task.status.state.is_terminal() {
        return Err(JsonRpcError::new(
            error_code::TASK_NOT_CANCELABLE,
            format!("task {task_id} is already in a terminal state"),
        ));
    }
    if let Err(error) = inner.executor.cancel(context_from_task(&task)).await {
        tracing::warn!(
            %error,
            task_id = %task.id,
            "A2A executor cancellation failed; publishing CANCELED anyway"
        );
    }
    let (sink, _receiver) = EventSink::new(
        task.id.clone(),
        task.context_id.clone(),
        Arc::clone(&inner.store),
    );
    sink.canceled();
    stored_task(inner, &task.id)
}

/// The async cancel variant used by the JSON-RPC `CancelTask` adapter.
pub(super) async fn cancel_request(
    inner: &Inner,
    request: CancelTaskRequest,
) -> std::result::Result<Task, JsonRpcError> {
    let task_id = request.id.trim();
    if task_id.is_empty() {
        return Err(JsonRpcError::new(
            error_code::INVALID_PARAMS,
            "CancelTask requires a task id",
        ));
    }
    cancel_stored_task(inner, task_id).await
}

/// The execution context of an already stored task.
fn context_from_task(task: &Task) -> super::config::A2aExecutionContext {
    let message = task.history.last().cloned().unwrap_or_else(|| Message {
        message_id: fresh_id(),
        context_id: Some(task.context_id.clone()),
        task_id: Some(task.id.clone()),
        ..Message::default()
    });
    super::config::A2aExecutionContext {
        task_id: task.id.clone(),
        context_id: task.context_id.clone(),
        message,
        metadata: task.metadata.clone(),
    }
}

/// The task id a `GetTask`/`SubscribeToTask` params object carries.
///
/// The v1 name is `id`; `taskId` is accepted as well, matching the vendored
/// Python adapter's tolerance for both spellings.
pub(super) fn task_id_param(params: Option<&Value>) -> Option<String> {
    let params = params?.as_object()?;
    ["id", "taskId"].iter().find_map(|key| {
        params
            .get(*key)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
    })
}

/// A `-32001` error for `task_id`.
pub(super) fn task_not_found(task_id: &str) -> JsonRpcError {
    JsonRpcError::new(
        error_code::TASK_NOT_FOUND,
        format!("task not found: {task_id}"),
    )
}

/// Serializes `value`, degrading to `null` on the impossible failure.
pub(super) fn value_of<T: Serialize + ?Sized>(value: &T) -> Value {
    match serde_json::to_value(value) {
        Ok(value) => value,
        Err(error) => {
            tracing::error!(%error, "failed to serialize an A2A payload");
            Value::Null
        }
    }
}

/// A JSON HTTP response body.
pub(super) fn json_bytes(body: Vec<u8>) -> Response {
    let mut response = Response::new(Body::from(body));
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    response
}

/// A plain-text HTTP response, used only for internal serialization failures.
pub(super) fn text_response(status: StatusCode, body: String) -> Response {
    let mut response = Response::new(Body::from(body));
    *response.status_mut() = status;
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/plain; charset=utf-8"),
    );
    response
}
