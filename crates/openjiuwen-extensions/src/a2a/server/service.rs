//! The server itself: the axum routes for both bindings, JSON-RPC dispatch,
//! and the lifecycle wrapper.

use std::net::SocketAddr;
use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::Response;
use axum::routing::{get, post};
use axum::Router;
use openjiuwen_common::error::{Error, Result};
use serde::Serialize;
use serde_json::Value;
use tokio::net::TcpListener;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use super::config::{A2aExecutor, A2aServerConfig};
use super::ops::{self, json_bytes, text_response};
use super::rest;
use super::sse;
use super::store::{InMemoryTaskStore, TaskStore};
use super::Inner;
use crate::a2a::jsonrpc::{self, error_code, method, JsonRpcError, JsonRpcResponse};
use crate::a2a::types::CancelTaskRequest;

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
    /// [`Error::Unsupported`] when the card advertises a gRPC interface;
    /// [`Error::Config`] when a mount path is not absolute or the REST base
    /// collides with another mounted route.
    pub fn new(config: A2aServerConfig, executor: Arc<dyn A2aExecutor>) -> Result<Self> {
        Self::with_store(config, executor, Arc::new(InMemoryTaskStore::new()))
    }

    /// Builds a server over `store`.
    ///
    /// # Errors
    ///
    /// [`Error::Unsupported`] when the card advertises a gRPC interface;
    /// [`Error::Config`] when a mount path is not absolute or the REST base
    /// collides with another mounted route.
    pub fn with_store(
        config: A2aServerConfig,
        executor: Arc<dyn A2aExecutor>,
        store: Arc<dyn TaskStore>,
    ) -> Result<Self> {
        reject_unsupported_transports(&config)?;
        validate_paths(&config)?;
        let card = super::derive_card(&config);
        Ok(Self {
            inner: Arc::new(Inner {
                config,
                executor,
                store,
                card,
            }),
        })
    }

    /// The configuration this server was built with.
    #[must_use]
    pub fn config(&self) -> &A2aServerConfig {
        &self.inner.config
    }

    /// The card served at every card route.
    ///
    /// Derived from [`Self::config`]'s card at construction time: the
    /// transport interfaces advertise exactly the bindings this server mounts,
    /// while every other card field is passed through unchanged.
    #[must_use]
    pub fn agent_card(&self) -> &crate::a2a::types::AgentCard {
        &self.inner.card
    }

    /// The task store this server reads and writes.
    #[must_use]
    pub fn store(&self) -> Arc<dyn TaskStore> {
        Arc::clone(&self.inner.store)
    }

    /// The axum router: the card route plus the routes of every mounted
    /// binding.
    pub fn router(&self) -> Router {
        let inner = Arc::clone(&self.inner);
        let config = &inner.config;
        let mut router = Router::new().route(&config.card_path, get(card_handler));
        if config.transport.mounts_json_rpc() {
            router = router.route(&config.rpc_path, post(rpc_handler));
        }
        if config.transport.mounts_rest() {
            router = router.merge(rest::router(&inner));
        }
        router.with_state(inner)
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
/// supported.")`; both `JSONRPC` and `HTTP+JSON` are served by
/// [`super::config::A2aTransport`].
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

/// Validates the mount paths before they reach `Router::route` (which panics on
/// a malformed path).
fn validate_paths(config: &A2aServerConfig) -> Result<()> {
    let paths = [
        ("card_path", config.card_path.as_str()),
        ("rpc_path", config.rpc_path.as_str()),
        ("rest_path", config.rest_path.as_str()),
    ];
    for (name, path) in paths {
        if !path.starts_with('/') {
            return Err(Error::config(format!(
                "A2A server {name} must start with `/`: {path}"
            )));
        }
    }
    if config.transport.mounts_rest() {
        let rest = config.rest_base_path();
        if rest.is_empty() || !rest.starts_with('/') {
            return Err(Error::config(format!(
                "A2A server rest path must resolve to a non-empty absolute path: {rest}"
            )));
        }
        if same_path(&rest, &config.card_path) || same_path(&rest, &config.rpc_path) {
            return Err(Error::config(format!(
                "A2A server rest base path `{rest}` collides with another mounted route"
            )));
        }
    }
    Ok(())
}

/// Whether two mount paths name the same route, ignoring a trailing slash.
fn same_path(left: &str, right: &str) -> bool {
    left.trim_end_matches('/') == right.trim_end_matches('/')
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

/// Serves the derived agent card; mounted at the well-known path and the REST
/// card route.
///
/// `async` because axum only accepts async handlers; the card is already
/// serialized by the time this returns, so nothing is awaited.
#[allow(clippy::unused_async)]
pub(super) async fn card_handler(State(inner): State<Arc<Inner>>) -> Response {
    match serde_json::to_vec(&inner.card) {
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

/// Non-streaming `SendMessage`: run the turn, then report the stored task.
async fn send_message(inner: &Arc<Inner>, id: &Value, params: Option<&Value>) -> Response {
    match ops::send_task(inner, params).await {
        Ok(task) => json_result(id, &task),
        Err(error) => json_error(id, error),
    }
}

/// Streaming `SendStreamingMessage`: one SSE frame per published event.
fn stream_message(inner: &Arc<Inner>, id: &Value, params: Option<&Value>) -> Response {
    match ops::start_stream(inner, params) {
        Ok(receiver) => sse::receiver_response(receiver, id.clone()),
        Err(error) => json_error(id, error),
    }
}

/// `SubscribeToTask`: a snapshot of the stored task plus its current status.
fn subscribe_to_task(inner: &Arc<Inner>, id: &Value, params: Option<&Value>) -> Response {
    let Some(task_id) = ops::task_id_param(params) else {
        return sse::responses(vec![JsonRpcResponse::failure(
            id.clone(),
            JsonRpcError::new(
                error_code::INVALID_PARAMS,
                "SubscribeToTask requires a task id",
            ),
        )]);
    };
    match ops::task_snapshot(inner, &task_id) {
        Ok((task, update)) => sse::responses(vec![
            JsonRpcResponse::success(id.clone(), ops::value_of(&task)),
            JsonRpcResponse::success(id.clone(), ops::value_of(&update)),
        ]),
        Err(error) => sse::responses(vec![JsonRpcResponse::failure(id.clone(), error)]),
    }
}

/// `GetTask`: the stored task, or `-32001`.
fn get_task(inner: &Arc<Inner>, id: &Value, params: Option<&Value>) -> Response {
    let Some(task_id) = ops::task_id_param(params) else {
        return invalid_params(id, "GetTask requires a task id");
    };
    match ops::stored_task(inner, &task_id) {
        Ok(task) => json_result(id, &task),
        Err(error) => json_error(id, error),
    }
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
    match ops::cancel_request(inner, request).await {
        Ok(task) => json_result(id, &task),
        Err(error) => json_error(id, error),
    }
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
