//! The server itself: the axum routes, JSON-RPC dispatch, and the lifecycle
//! wrapper.

use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;

use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::Response;
use axum::routing::{get, post};
use axum::Router;
use futures::StreamExt;
use openjiuwen_common::error::{Error, Result};
use serde::Serialize;
use serde_json::{Map, Value};
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use super::config::{A2aExecutionContext, A2aExecutor, A2aServerConfig};
use super::event_sink::EventSink;
use super::store::{InMemoryTaskStore, TaskStore};
use super::{agent_message, fresh_id};
use crate::a2a::jsonrpc::{self, error_code, method, JsonRpcError, JsonRpcResponse};
use crate::a2a::transform::A2aTransformer;
use crate::a2a::types::{
    CancelTaskRequest, Message, Part, Role, Task, TaskState, TaskStatus, TaskStatusUpdateEvent,
};

/// The agent message published together with the `WORKING` status.
const WORKING_MESSAGE: &str = "Processing your request...";

/// The payload keys consumed by the request mapping, never copied into the
/// execution metadata (Python's exclusion set).
const RESERVED_PAYLOAD_KEYS: [&str; 3] = ["query", "task_id", "sessionId"];

/// The shared server state behind every route.
struct Inner {
    config: A2aServerConfig,
    executor: Arc<dyn A2aExecutor>,
    store: Arc<dyn TaskStore>,
}

/// An A2A server: a router plus the store and executor it dispatches to.
///
/// Python parity: `A2AServer` in `a2a_server.py` (the transport half) and
/// `A2AServerAdapter` for the start/stop lifecycle. Cloning is cheap and yields
/// another handle onto the same store and executor, which is what
/// [`A2aServerAdapter`] uses to keep a `&self` view while serving.
#[derive(Clone)]
pub struct A2aServer {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for A2aServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The executor and the store are trait objects, so only the configuration
        // and the store's task count are printable.
        f.debug_struct("A2aServer")
            .field("config", &self.inner.config)
            .field("tasks", &self.inner.store.all().len())
            .finish()
    }
}

impl A2aServer {
    /// Builds a server over a fresh [`InMemoryTaskStore`].
    ///
    /// # Errors
    ///
    /// [`Error::Unsupported`] when the card advertises a gRPC interface.
    pub fn new(config: A2aServerConfig, executor: Arc<dyn A2aExecutor>) -> Result<Self> {
        Self::with_store(config, executor, Arc::new(InMemoryTaskStore::new()))
    }

    /// Builds a server over `store`.
    ///
    /// # Errors
    ///
    /// [`Error::Unsupported`] when the card advertises a gRPC interface.
    pub fn with_store(
        config: A2aServerConfig,
        executor: Arc<dyn A2aExecutor>,
        store: Arc<dyn TaskStore>,
    ) -> Result<Self> {
        reject_unsupported_transports(&config)?;
        Ok(Self {
            inner: Arc::new(Inner {
                config,
                executor,
                store,
            }),
        })
    }

    /// The configuration this server was built with.
    #[must_use]
    pub fn config(&self) -> &A2aServerConfig {
        &self.inner.config
    }

    /// The task store this server reads and writes.
    #[must_use]
    pub fn store(&self) -> Arc<dyn TaskStore> {
        Arc::clone(&self.inner.store)
    }

    /// The axum router: the card route plus the `JSON-RPC` route.
    pub fn router(&self) -> Router {
        Router::new()
            .route(&self.inner.config.card_path, get(card_handler))
            .route(&self.inner.config.rpc_path, post(rpc_handler))
            .with_state(Arc::clone(&self.inner))
    }

    /// Serves until `shutdown` is cancelled.
    ///
    /// # Errors
    ///
    /// [`Error::Transport`] when the socket cannot be bound or the HTTP server
    /// fails.
    pub async fn serve(self, shutdown: CancellationToken) -> Result<()> {
        let listener = bind_listener(&self.inner.config).await?;
        let local_addr = listener.local_addr().map_err(|error| {
            Error::transport(format!("failed to read the A2A bind address: {error}"))
        })?;
        tracing::info!(%local_addr, "A2A server listening");
        let router = self.router();
        axum::serve(listener, router)
            .with_graceful_shutdown(async move { shutdown.cancelled().await })
            .await
            .map_err(|error| Error::transport(format!("A2A server failed: {error}")))
    }

    /// Binds now and serves in a background task until shutdown.
    ///
    /// Binding happens before the call returns, so the [`ServerHandle`] knows the
    /// local address (which is how a `port = 0` configuration is used).
    ///
    /// # Errors
    ///
    /// [`Error::Transport`] when the socket cannot be bound.
    pub async fn serve_in_background(self, shutdown: CancellationToken) -> Result<ServerHandle> {
        let listener = bind_listener(&self.inner.config).await?;
        let local_addr = listener.local_addr().map_err(|error| {
            Error::transport(format!("failed to read the A2A bind address: {error}"))
        })?;
        tracing::info!(%local_addr, "A2A server listening in the background");
        // A child token: cancelling either the caller's token or the handle's
        // shuts the server down.
        let token = shutdown.child_token();
        let serve_token = token.clone();
        let router = self.router();
        let join = tokio::spawn(async move {
            axum::serve(listener, router)
                .with_graceful_shutdown(async move { serve_token.cancelled().await })
                .await
                .map_err(|error| Error::transport(format!("A2A server failed: {error}")))
        });
        Ok(ServerHandle {
            local_addr,
            token,
            join,
        })
    }
}

/// Rejects every transport binding this server does not speak.
///
/// gRPC is refused like Python's `ValueError("gRPC transport is not
/// supported.")`; a REST (`HTTP+JSON`) binding is tolerated, but no REST route
/// is mounted.
fn reject_unsupported_transports(config: &A2aServerConfig) -> Result<()> {
    let grpc = config
        .agent_card
        .supported_interfaces
        .iter()
        .any(|interface| interface.protocol_binding.eq_ignore_ascii_case("grpc"));
    if grpc {
        return Err(Error::unsupported("gRPC transport is not supported"));
    }
    Ok(())
}

/// Binds the configured address.
async fn bind_listener(config: &A2aServerConfig) -> Result<TcpListener> {
    TcpListener::bind((config.host.as_str(), config.port))
        .await
        .map_err(|error| {
            Error::transport(format!(
                "failed to bind the A2A server to {}:{}: {error}",
                config.host, config.port
            ))
        })
}

/// A running [`A2aServer`] returned by [`A2aServer::serve_in_background`].
pub struct ServerHandle {
    local_addr: SocketAddr,
    token: CancellationToken,
    join: JoinHandle<Result<()>>,
}

impl ServerHandle {
    /// The address the server is listening on.
    #[must_use]
    pub const fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    /// Requests a graceful shutdown without waiting for it.
    pub fn shutdown(&self) {
        self.token.cancel();
    }

    /// Waits for the server task to finish.
    ///
    /// # Errors
    ///
    /// [`Error::Transport`] when the serving task panicked or was aborted, or
    /// the HTTP server failed.
    pub async fn join(self) -> Result<()> {
        self.join
            .await
            .map_err(|error| Error::transport(format!("A2A server task failed to join: {error}")))?
    }
}

/// The Python-compatible start/stop wrapper around [`A2aServer`].
///
/// Python parity: `A2AServerAdapter` (`a2a_server_adapter.py`), whose `start()`
/// launches the runtime thread and whose `stop()` shuts it down. Both calls are
/// idempotent here, and the adapter binds an ephemeral port when
/// [`A2aServerConfig::port`] is `0`.
pub struct A2aServerAdapter {
    server: A2aServer,
    shutdown: CancellationToken,
    handle: Option<ServerHandle>,
}

impl A2aServerAdapter {
    /// Wraps `server` in a start/stop lifecycle.
    #[must_use]
    pub fn new(server: A2aServer) -> Self {
        Self {
            server,
            shutdown: CancellationToken::new(),
            handle: None,
        }
    }

    /// The wrapped server.
    #[must_use]
    pub const fn server(&self) -> &A2aServer {
        &self.server
    }

    /// Starts the server; a second call while running is a no-op.
    ///
    /// # Errors
    ///
    /// [`Error::Transport`] when the socket cannot be bound.
    pub async fn start(&mut self) -> Result<()> {
        if self.handle.is_some() {
            return Ok(());
        }
        self.shutdown = CancellationToken::new();
        let handle = self
            .server
            .clone()
            .serve_in_background(self.shutdown.child_token())
            .await?;
        self.handle = Some(handle);
        Ok(())
    }

    /// Stops the server and waits for it; stopped twice is a no-op.
    ///
    /// # Errors
    ///
    /// [`Error::Transport`] when the serving task panicked or was aborted.
    pub async fn stop(&mut self) -> Result<()> {
        let Some(handle) = self.handle.take() else {
            return Ok(());
        };
        handle.shutdown();
        handle.join().await
    }

    /// Whether the server is currently running.
    #[must_use]
    pub const fn active(&self) -> bool {
        self.handle.is_some()
    }

    /// The address the running server listens on.
    #[must_use]
    pub fn local_addr(&self) -> Option<SocketAddr> {
        self.handle.as_ref().map(ServerHandle::local_addr)
    }
}

/// Serves the configured agent card.
///
/// `async` because axum only accepts async handlers; the card is already
/// serialized by the time this returns, so nothing is awaited.
#[allow(clippy::unused_async)]
async fn card_handler(State(inner): State<Arc<Inner>>) -> Response {
    match serde_json::to_vec(&inner.config.agent_card) {
        Ok(body) => json_bytes(body),
        Err(error) => {
            tracing::error!(%error, "failed to serialize the A2A agent card");
            text_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("failed to serialize the agent card: {error}"),
            )
        }
    }
}

/// Dispatches one `JSON-RPC` request.
async fn rpc_handler(State(inner): State<Arc<Inner>>, body: Bytes) -> Response {
    let request = match jsonrpc::parse_request(&body) {
        Ok(request) => request,
        Err(error) => return json_response(&JsonRpcResponse::failure(Value::Null, error)),
    };
    tracing::debug!(method = %request.method, "A2A JSON-RPC request");
    match method::canonical(&request.method) {
        Some(method::SEND_MESSAGE) => {
            send_message(&inner, &request.id, request.params.as_ref()).await
        }
        Some(method::SEND_STREAMING_MESSAGE) => {
            stream_message(&inner, &request.id, request.params.as_ref())
        }
        Some(method::SUBSCRIBE_TO_TASK) => {
            subscribe_to_task(&inner, &request.id, request.params.as_ref())
        }
        Some(method::GET_TASK) => get_task(&inner, &request.id, request.params.as_ref()),
        Some(method::CANCEL_TASK) => {
            cancel_task(&inner, &request.id, request.params.as_ref()).await
        }
        _ => json_error(
            &request.id,
            JsonRpcError::new(
                error_code::METHOD_NOT_FOUND,
                format!("unknown method `{}`", request.method),
            ),
        ),
    }
}

/// Everything a `SendMessage` needs once the ids and message are derived.
struct SendPlan {
    sink: EventSink,
    receiver: mpsc::UnboundedReceiver<Value>,
    context: A2aExecutionContext,
    message: Message,
}

/// Derives the ids, message, execution context, and sink of a send call.
fn plan_send(inner: &Inner, params: Option<&Value>) -> std::result::Result<SendPlan, JsonRpcError> {
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
        context: A2aExecutionContext {
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
fn publish_prelude(sink: &EventSink, message: &Message, context: &A2aExecutionContext) {
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
    executor: Arc<dyn A2aExecutor>,
    sink: EventSink,
    context: A2aExecutionContext,
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

/// Non-streaming `SendMessage`: run the turn, then report the stored task.
async fn send_message(inner: &Arc<Inner>, id: &Value, params: Option<&Value>) -> Response {
    let plan = match plan_send(inner, params) {
        Ok(plan) => plan,
        Err(error) => return json_error(id, error),
    };
    let SendPlan {
        sink,
        receiver: _receiver,
        context,
        message,
    } = plan;
    let task_id = context.task_id.clone();
    publish_prelude(&sink, &message, &context);
    let outcome = run_executor(Arc::clone(&inner.executor), sink, context).await;
    match outcome {
        Ok(()) => match inner.store.get(&task_id) {
            Some(task) => json_result(id, &task),
            None => json_error(
                id,
                JsonRpcError::new(
                    error_code::INTERNAL,
                    format!("task {task_id} is missing from the task store"),
                ),
            ),
        },
        Err(error) => json_error(
            id,
            JsonRpcError::new(
                error_code::INTERNAL,
                format!("A2A executor failed: {error}"),
            ),
        ),
    }
}

/// Streaming `SendStreamingMessage`: one SSE frame per published event.
fn stream_message(inner: &Arc<Inner>, id: &Value, params: Option<&Value>) -> Response {
    let plan = match plan_send(inner, params) {
        Ok(plan) => plan,
        Err(error) => return json_error(id, error),
    };
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
    sse_receiver_response(receiver, id.clone())
}

/// `SubscribeToTask`: a snapshot of the stored task plus its current status.
fn subscribe_to_task(inner: &Arc<Inner>, id: &Value, params: Option<&Value>) -> Response {
    let Some(task_id) = task_id_param(params) else {
        return sse_responses(vec![JsonRpcResponse::failure(
            id.clone(),
            JsonRpcError::new(
                error_code::INVALID_PARAMS,
                "SubscribeToTask requires a task id",
            ),
        )]);
    };
    let Some(task) = inner.store.get(&task_id) else {
        return sse_responses(vec![JsonRpcResponse::failure(
            id.clone(),
            task_not_found(&task_id),
        )]);
    };
    let update = TaskStatusUpdateEvent {
        task_id: task.id.clone(),
        context_id: task.context_id.clone(),
        status: task.status.clone(),
        metadata: Map::new(),
    };
    sse_responses(vec![
        JsonRpcResponse::success(id.clone(), value_of(&task)),
        JsonRpcResponse::success(id.clone(), value_of(&update)),
    ])
}

/// `GetTask`: the stored task, or `-32001`.
fn get_task(inner: &Arc<Inner>, id: &Value, params: Option<&Value>) -> Response {
    let Some(task_id) = task_id_param(params) else {
        return invalid_params(id, "GetTask requires a task id");
    };
    inner.store.get(&task_id).map_or_else(
        || json_error(id, task_not_found(&task_id)),
        |task| json_result(id, &task),
    )
}

/// `CancelTask`: ask the executor, then publish `CANCELED`.
async fn cancel_task(inner: &Arc<Inner>, id: &Value, params: Option<&Value>) -> Response {
    let request = match params {
        Some(params) => match serde_json::from_value::<CancelTaskRequest>(params.clone()) {
            Ok(request) => request,
            Err(error) => {
                return invalid_params(id, format!("invalid CancelTask params: {error}"));
            }
        },
        None => return invalid_params(id, "CancelTask requires a task id"),
    };
    let task_id = request.id.trim();
    if task_id.is_empty() {
        return invalid_params(id, "CancelTask requires a task id");
    }
    let Some(task) = inner.store.get(task_id) else {
        return json_error(id, task_not_found(task_id));
    };
    if task.status.state.is_terminal() {
        return json_error(
            id,
            JsonRpcError::new(
                error_code::TASK_NOT_CANCELABLE,
                format!("task {task_id} is already in a terminal state"),
            ),
        );
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
    match inner.store.get(&task.id) {
        Some(task) => json_result(id, &task),
        None => json_error(
            id,
            JsonRpcError::new(
                error_code::INTERNAL,
                format!("task {} is missing from the task store", task.id),
            ),
        ),
    }
}

/// The execution context of an already stored task.
fn context_from_task(task: &Task) -> A2aExecutionContext {
    let message = task.history.last().cloned().unwrap_or_else(|| Message {
        message_id: fresh_id(),
        context_id: Some(task.context_id.clone()),
        task_id: Some(task.id.clone()),
        ..Message::default()
    });
    A2aExecutionContext {
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
fn task_id_param(params: Option<&Value>) -> Option<String> {
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
fn task_not_found(task_id: &str) -> JsonRpcError {
    JsonRpcError::new(
        error_code::TASK_NOT_FOUND,
        format!("task not found: {task_id}"),
    )
}

/// An `INVALID_PARAMS` error envelope.
fn invalid_params(id: &Value, message: impl Into<String>) -> Response {
    json_error(id, JsonRpcError::new(error_code::INVALID_PARAMS, message))
}

/// A `200 application/json` envelope.
fn json_response(response: &JsonRpcResponse) -> Response {
    json_bytes(jsonrpc::serialize(response).into_bytes())
}

/// A `200 application/json` failure envelope.
fn json_error(id: &Value, error: JsonRpcError) -> Response {
    json_response(&JsonRpcResponse::failure(id.clone(), error))
}

/// A `200 application/json` result envelope for `value`.
fn json_result<T: Serialize + ?Sized>(id: &Value, value: &T) -> Response {
    match serde_json::to_value(value) {
        Ok(value) => json_response(&JsonRpcResponse::success(id.clone(), value)),
        Err(error) => {
            tracing::error!(%error, "failed to serialize an A2A result");
            json_error(
                id,
                JsonRpcError::new(
                    error_code::INTERNAL,
                    format!("failed to serialize the result: {error}"),
                ),
            )
        }
    }
}

/// Serializes `value`, degrading to `null` on the impossible failure.
fn value_of<T: Serialize + ?Sized>(value: &T) -> Value {
    match serde_json::to_value(value) {
        Ok(value) => value,
        Err(error) => {
            tracing::error!(%error, "failed to serialize an A2A payload");
            Value::Null
        }
    }
}

/// A JSON HTTP response body.
fn json_bytes(body: Vec<u8>) -> Response {
    let mut response = Response::new(Body::from(body));
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    response
}

/// A plain-text HTTP response, used only for internal serialization failures.
fn text_response(status: StatusCode, body: String) -> Response {
    let mut response = Response::new(Body::from(body));
    *response.status_mut() = status;
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/plain; charset=utf-8"),
    );
    response
}

/// One `text/event-stream` frame holding a `JSON-RPC` response.
fn sse_frame(response: &JsonRpcResponse) -> Bytes {
    Bytes::from(format!("data: {}\n\n", jsonrpc::serialize(response)))
}

/// Wraps a stream of `JSON-RPC` responses as an SSE response.
fn sse_response<S>(responses: S) -> Response
where
    S: futures::Stream<Item = JsonRpcResponse> + Send + 'static,
{
    let frames = responses.map(|response| Ok::<Bytes, Infallible>(sse_frame(&response)));
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

/// An SSE response that ends after `responses` (used by `SubscribeToTask`).
fn sse_responses(responses: Vec<JsonRpcResponse>) -> Response {
    sse_response(futures::stream::iter(responses))
}

/// An SSE response carrying one frame per value received from `receiver`.
fn sse_receiver_response(receiver: mpsc::UnboundedReceiver<Value>, id: Value) -> Response {
    let mut receiver = receiver;
    let responses = async_stream::stream! {
        while let Some(value) = receiver.recv().await {
            yield JsonRpcResponse::success(id.clone(), value);
        }
    };
    sse_response(responses)
}
