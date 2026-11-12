//! The A2A server: axum routes, `JSON-RPC` dispatch, the executor SPI, and the
//! task store.
//!
//! Port of `openjiuwen/extensions/a2a/a2a_server.py`, `a2a_agent_executor.py`,
//! and `a2a_server_adapter.py` (v0.1.20). Python hands the transport to
//! `FastAPI` plus the `a2a-sdk` `DefaultRequestHandler`; this module owns the
//! same surface over axum: two routes, five `JSON-RPC` 2.0 methods, the
//! [`A2aExecutor`] SPI, an [`EventSink`] that mirrors Python's `TaskUpdater`,
//! and a [`TaskStore`] kept coherent by that sink.
//!
//! # Routes
//!
//! | Route | Method | Body |
//! |---|---|---|
//! | [`A2aServerConfig::card_path`] (default `/.well-known/agent-card.json`) | `GET`, `HEAD` | the serialized [`types::AgentCard`] as `application/json` |
//! | [`A2aServerConfig::rpc_path`] (default `/a2a/jsonrpc/`) | `POST` | one `JSON-RPC` envelope, or `text/event-stream` for streaming methods |
//!
//! The `rpc_path` is mounted **literally**, trailing slash included, because
//! `a2a-sdk` clients and cards publish `.../a2a/jsonrpc/` and streaming clients
//! often do not follow the `307` a missing slash would produce. Clients must
//! `POST` to exactly that path.
//!
//! # `JSON-RPC` methods
//!
//! | Canonical (v1) | v0.3 alias | Result |
//! |---|---|---|
//! | `SendMessage` | `message/send` | the stored [`types::Task`] |
//! | `SendStreamingMessage` | `message/stream` | `text/event-stream`, one `data:` frame per event |
//! | `GetTask` | `tasks/get` | the stored [`types::Task`] |
//! | `CancelTask` | `tasks/cancel` | the stored [`types::Task`] in `CANCELED` |
//! | `SubscribeToTask` | `tasks/resubscribe` | a task snapshot frame plus its current status frame |
//!
//! Both spellings resolve through [`jsonrpc::method::canonical`]. Every
//! non-streaming reply is HTTP 200 carrying a `JSON-RPC` envelope, as the
//! `JSON-RPC` convention requires: `-32600`/`-32700` for a malformed request,
//! `-32601` for an unknown method, `-32602` for unusable params, `-32001` for an
//! unknown task, `-32002` for a task that is already terminal, and `-32603` when
//! the executor fails.
//!
//! # Event order (Python `A2AAgentExecutor`)
//!
//! Every `SendMessage` — streaming or not — publishes, in this exact order:
//!
//! 1. a [`types::Task`] snapshot with `state = SUBMITTED` and
//!    `history = [request message]`;
//! 2. a `WORKING` [`types::TaskStatusUpdateEvent`] carrying the agent message
//!    `Processing your request...`;
//! 3. zero or more artifact updates produced by the executor (`last_chunk` is
//!    set only when the executor marks the artifact terminal);
//! 4. for each artifact still open, a closing
//!    [`types::TaskArtifactUpdateEvent`] with empty `parts`, `append = true`,
//!    and `last_chunk = true`, in ascending artifact-id order;
//! 5. the terminal status: `FAILED`, `CANCELED`, `INPUT_REQUIRED`, otherwise
//!    `COMPLETED`.
//!
//! The snapshot has to come first: `a2a-sdk` consumers reject a status update
//! that arrives before the task exists — their error is "Agent should enqueue
//! `Task` before `TaskStatusUpdateEvent`". Executor errors publish `FAILED` and,
//! for a non-streaming call, are also reported as a `-32603` error envelope.
//!
//! # Store semantics
//!
//! Every [`EventSink`] emit also updates the stored [`types::Task`], which is
//! what `GetTask`, `CancelTask`, and the non-streaming `SendMessage` reply read:
//! the first snapshot inserts the task, status updates replace its status, and
//! an artifact update replaces the artifact with the same id or appends it.
//! Closing an open artifact does **not** touch the store — the accumulated
//! parts stay intact, matching Python's "close prior chunks without replacing
//! their accumulated contents". A dropped stream receiver is not an error: the
//! store is still updated, which is exactly what a non-streaming call needs.
//!
//! # Documented deviations from the `CONTEXT.md` contract
//!
//! 1. [`A2aServer::serve_in_background`] is `async`: a socket must be bound
//!    before its local address is known (Python's `port = 0` case picks the port
//!    at bind time), so the caller awaits the bind and receives a
//!    [`ServerHandle`] that is already listening.
//! 2. [`A2aExecutor::cancel`]'s default body returns `Ok(())` and the server
//!    publishes the `CANCELED` status and event itself. The observable behavior
//!    matches Python's "default: emit Canceled", without forcing every executor
//!    to drive the sink.
//! 3. `SubscribeToTask` (`tasks/resubscribe`) answers with a stored task
//!    snapshot plus its current status and then ends the stream; it is not a
//!    live subscription to a running task.
//!
//! # Documented limitations
//!
//! - The gRPC transport is rejected at construction time
//!   ([`Error::Unsupported`]). A REST (`HTTP+JSON`) interface is tolerated on
//!   the card, but no REST route is mounted, exactly like the Python wrapper
//!   when only the JSON-RPC transport is configured.
//! - Tasks live in memory by default ([`InMemoryTaskStore`]); supply another
//!   [`TaskStore`] through [`A2aServer::with_store`] to persist them.

use std::collections::{BTreeMap, BTreeSet};
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

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

use super::jsonrpc::{self, error_code, method, JsonRpcError, JsonRpcResponse};
use super::transform::A2aTransformer;
use super::types::{
    self, Artifact, CancelTaskRequest, Message, Part, Role, Task, TaskArtifactUpdateEvent,
    TaskState, TaskStatus, TaskStatusUpdateEvent,
};

/// The default JSON-RPC mount path, trailing slash included.
pub const DEFAULT_RPC_PATH: &str = "/a2a/jsonrpc/";

/// The default agent-card path.
pub const DEFAULT_CARD_PATH: &str = "/.well-known/agent-card.json";

/// The default protocol version advertised on the card (`1.0`, the A2A v1 wire).
pub const DEFAULT_PROTOCOL_VERSION: &str = "1.0";

/// The default bind host (loopback, like Python's `start(host="127.0.0.1")`).
pub const DEFAULT_HOST: &str = "127.0.0.1";

/// The default bind port (Python's `start(port=8000)`).
pub const DEFAULT_PORT: u16 = 8000;

/// The agent message published together with the `WORKING` status.
const WORKING_MESSAGE: &str = "Processing your request...";

/// The payload keys consumed by the request mapping, never copied into the
/// execution metadata (Python's exclusion set).
const RESERVED_PAYLOAD_KEYS: [&str; 3] = ["query", "task_id", "sessionId"];

/// A fresh 32-lowercase-hex id, shaped like Python's `uuid4().hex`.
fn fresh_id() -> String {
    uuid::Uuid::new_v4().simple().to_string()
}

/// An agent-authored message for `state`, stamped with the task and context ids.
fn agent_message(text: impl Into<String>, task_id: &str, context_id: &str) -> Message {
    Message {
        message_id: fresh_id(),
        context_id: Some(context_id.to_owned()),
        task_id: Some(task_id.to_owned()),
        role: Role::Agent,
        parts: vec![Part::text(text)],
        ..Message::default()
    }
}

/// The stored tasks of one server, keyed by task id.
///
/// Python parity: `a2a.server.tasks.inmemory_task_store.InMemoryTaskStore`'s
/// interface. The server reads through this trait in `GetTask`/`CancelTask` and
/// the [`EventSink`] writes through it, so a deployment can swap in a durable
/// store without touching the protocol layer.
pub trait TaskStore: Send + Sync {
    /// The task with this id, if it is known.
    fn get(&self, id: &str) -> Option<Task>;

    /// Inserts or replaces a task; the task id is the key.
    fn put(&self, task: Task);

    /// Every known task.
    fn all(&self) -> Vec<Task>;
}

/// A process-local [`TaskStore`].
///
/// Tasks are kept in a [`BTreeMap`], so [`TaskStore::all`] returns them in
/// ascending id order. Python's dict yields insertion order instead; the ids are
/// opaque, so the sorted order is a documented, deterministic substitute.
#[derive(Debug, Default)]
pub struct InMemoryTaskStore {
    tasks: Mutex<BTreeMap<String, Task>>,
}

impl InMemoryTaskStore {
    /// An empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl TaskStore for InMemoryTaskStore {
    fn get(&self, id: &str) -> Option<Task> {
        self.tasks
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(id)
            .cloned()
    }

    fn put(&self, task: Task) {
        self.tasks
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(task.id.clone(), task);
    }

    fn all(&self) -> Vec<Task> {
        self.tasks
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .values()
            .cloned()
            .collect()
    }
}

/// The publisher an [`A2aExecutor`] drives while it produces a task.
///
/// Python parity: `a2a.server.tasks.task_updater.TaskUpdater` as used by
/// `A2AAgentExecutor`. Every emit is serialized once, handed to the streaming
/// channel, and folded into the [`TaskStore`] so readers that never see the
/// stream — `GetTask`, `CancelTask`, a non-streaming `SendMessage` — still
/// observe coherent state.
///
/// All methods are infallible: a dropped receiver is ignored (the store update
/// still happens), and a terminal method is a no-op once the task has reached a
/// terminal state, matching Python's "already in a terminal state" behavior.
pub struct EventSink {
    task_id: String,
    context_id: String,
    store: Arc<dyn TaskStore>,
    sender: mpsc::UnboundedSender<Value>,
    open_artifacts: Mutex<BTreeSet<String>>,
    terminal: AtomicBool,
}

impl EventSink {
    /// Creates a sink and the receiver its events are streamed through.
    pub(crate) fn new(
        task_id: String,
        context_id: String,
        store: Arc<dyn TaskStore>,
    ) -> (Self, mpsc::UnboundedReceiver<Value>) {
        let (sender, receiver) = mpsc::unbounded_channel();
        let sink = Self {
            task_id,
            context_id,
            store,
            sender,
            open_artifacts: Mutex::new(BTreeSet::new()),
            terminal: AtomicBool::new(false),
        };
        (sink, receiver)
    }

    /// The id of the task this sink publishes for.
    #[must_use]
    pub fn task_id(&self) -> &str {
        &self.task_id
    }

    /// The context (conversation) id this sink publishes for.
    #[must_use]
    pub fn context_id(&self) -> &str {
        &self.context_id
    }

    /// Whether a terminal status has been published already.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        self.terminal.load(Ordering::Acquire)
    }

    /// Publishes the [`Task`] snapshot and stores it (`state` as supplied).
    pub fn task(&self, task: Task) {
        self.emit(&task);
        self.store.put(task);
    }

    /// Publishes a `WORKING` status update and stores it.
    pub fn working(&self, message: Option<Message>) {
        let status = TaskStatus {
            state: TaskState::Working,
            message,
            timestamp: None,
        };
        self.emit(&TaskStatusUpdateEvent {
            task_id: self.task_id.clone(),
            context_id: self.context_id.clone(),
            status: status.clone(),
            metadata: Map::new(),
        });
        self.update_stored(|task| task.status = status);
    }

    /// Publishes one artifact chunk and stores the artifact.
    ///
    /// An empty [`Artifact::artifact_id`] is filled with a fresh id. While
    /// `last_chunk` is `false` the artifact stays open, so a later terminal
    /// method closes it; with `last_chunk` the artifact is complete.
    pub fn artifact(&self, artifact: Artifact, last_chunk: bool) {
        let artifact = with_artifact_id(artifact);
        let artifact_id = artifact.artifact_id.clone();
        self.emit(&TaskArtifactUpdateEvent {
            task_id: self.task_id.clone(),
            context_id: self.context_id.clone(),
            artifact: artifact.clone(),
            append: false,
            last_chunk,
            metadata: Map::new(),
        });
        self.with_open_artifacts(|open| {
            if last_chunk {
                open.remove(&artifact_id);
            } else {
                open.insert(artifact_id.clone());
            }
        });
        self.update_stored(|task| {
            match task
                .artifacts
                .iter_mut()
                .find(|existing| existing.artifact_id == artifact_id)
            {
                Some(existing) => *existing = artifact,
                None => task.artifacts.push(artifact),
            }
        });
    }

    /// Publishes the `COMPLETED` terminal status.
    pub fn complete(&self) {
        self.finish(TaskState::Completed, None);
    }

    /// Publishes the `FAILED` terminal status with `message` as its text.
    pub fn fail(&self, message: impl Into<String>) {
        let status_message = agent_message(message, &self.task_id, &self.context_id);
        self.finish(TaskState::Failed, Some(status_message));
    }

    /// Publishes the `INPUT_REQUIRED` terminal status.
    pub fn require_input(&self, message: Option<Message>) {
        self.finish(TaskState::InputRequired, message);
    }

    /// Publishes the `CANCELED` terminal status.
    pub fn canceled(&self) {
        self.finish(TaskState::Canceled, None);
    }

    /// Publishes a terminal status: closes open artifacts first, then the status.
    fn finish(&self, state: TaskState, message: Option<Message>) {
        if self.terminal.swap(true, Ordering::AcqRel) {
            return;
        }
        self.close_open_artifacts();
        let status = TaskStatus {
            state,
            message,
            timestamp: None,
        };
        self.emit(&TaskStatusUpdateEvent {
            task_id: self.task_id.clone(),
            context_id: self.context_id.clone(),
            status: status.clone(),
            metadata: Map::new(),
        });
        self.update_stored(|task| task.status = status);
    }

    /// Closes every open artifact without replacing its stored contents.
    fn close_open_artifacts(&self) {
        let open = self.with_open_artifacts(std::mem::take);
        for artifact_id in open {
            self.emit(&TaskArtifactUpdateEvent {
                task_id: self.task_id.clone(),
                context_id: self.context_id.clone(),
                artifact: Artifact {
                    artifact_id,
                    ..Artifact::default()
                },
                append: true,
                last_chunk: true,
                metadata: Map::new(),
            });
        }
    }

    /// Serializes `event` and queues it for the stream.
    fn emit<T: Serialize>(&self, event: &T) {
        let value = match serde_json::to_value(event) {
            Ok(value) => value,
            Err(error) => {
                tracing::error!(%error, "failed to serialize an A2A event");
                return;
            }
        };
        if self.sender.send(value).is_err() {
            tracing::debug!("A2A event receiver dropped; the task store is still updated");
        }
    }

    /// Applies `update` to the stored task, when it exists.
    fn update_stored(&self, update: impl FnOnce(&mut Task)) {
        if let Some(mut task) = self.store.get(&self.task_id) {
            update(&mut task);
            self.store.put(task);
        }
    }

    /// Runs `update` over the open-artifact id set.
    fn with_open_artifacts<R>(&self, update: impl FnOnce(&mut BTreeSet<String>) -> R) -> R {
        let mut open = self
            .open_artifacts
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        update(&mut open)
    }
}

/// Fills an empty artifact id with a fresh one.
fn with_artifact_id(mut artifact: Artifact) -> Artifact {
    if artifact.artifact_id.is_empty() {
        artifact.artifact_id = fresh_id();
    }
    artifact
}

/// The bind address, mount paths, and card a [`A2aServer`] publishes.
///
/// Python parity: the `A2AServer(...)` / `start(host, port)` arguments of
/// `a2a_server.py`, collapsed into one value. Defaults match Python:
/// `rpc_path = /a2a/jsonrpc/`, `card_path = /.well-known/agent-card.json`,
/// `protocol_version = 1.0`, `host = 127.0.0.1`, `port = 8000`.
#[derive(Debug, Clone, PartialEq)]
pub struct A2aServerConfig {
    /// The card served at [`Self::card_path`].
    pub agent_card: types::AgentCard,
    /// The JSON-RPC URL to publish on the card, when it differs from the bind
    /// address (Python's `interface_url`). Passed through untouched; the
    /// response body of the card route is the configured card itself.
    pub interface_url: Option<String>,
    /// The path `JSON-RPC` requests are `POST`ed to, mounted literally.
    pub rpc_path: String,
    /// The path the agent card is served at.
    pub card_path: String,
    /// The protocol version this server speaks.
    pub protocol_version: String,
    /// The host to bind (`serve`/`serve_in_background`).
    pub host: String,
    /// The port to bind; `0` picks an ephemeral port at bind time.
    pub port: u16,
}

impl Default for A2aServerConfig {
    fn default() -> Self {
        Self {
            agent_card: types::AgentCard::default(),
            interface_url: None,
            rpc_path: DEFAULT_RPC_PATH.to_owned(),
            card_path: DEFAULT_CARD_PATH.to_owned(),
            protocol_version: DEFAULT_PROTOCOL_VERSION.to_owned(),
            host: DEFAULT_HOST.to_owned(),
            port: DEFAULT_PORT,
        }
    }
}

impl A2aServerConfig {
    /// A configuration for `agent_card` with every other field defaulted.
    #[must_use]
    pub fn new(agent_card: types::AgentCard) -> Self {
        Self {
            agent_card,
            ..Self::default()
        }
    }

    /// Sets the JSON-RPC URL published on the card.
    #[must_use]
    pub fn with_interface_url(mut self, url: impl Into<String>) -> Self {
        self.interface_url = Some(url.into());
        self
    }

    /// Sets the bind host and port.
    #[must_use]
    pub fn with_bind(mut self, host: impl Into<String>, port: u16) -> Self {
        self.host = host.into();
        self.port = port;
        self
    }

    /// Sets the `JSON-RPC` mount path.
    #[must_use]
    pub fn with_rpc_path(mut self, path: impl Into<String>) -> Self {
        self.rpc_path = path.into();
        self
    }
}

/// One turn handed to an [`A2aExecutor`].
///
/// Python parity: the `RequestContext` members `A2AAgentExecutor` reads
/// (`task_id`, `context_id`, `message`) plus the openJiuwen payload metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct A2aExecutionContext {
    /// The task the turn belongs to (from the request, or freshly generated).
    pub task_id: String,
    /// The conversation the turn belongs to (`sessionId` when the request had
    /// one, otherwise freshly generated).
    pub context_id: String,
    /// The request message exactly as received (or synthesized from the payload).
    pub message: Message,
    /// The openJiuwen payload without its reserved `query`/`task_id`/
    /// `sessionId` keys.
    pub metadata: Map<String, Value>,
}

/// The application hook that produces a task.
///
/// Python parity: `A2AAgentExecutor`, minus the streaming/invoke split — an
/// executor publishes through the [`EventSink`] and the server decides whether
/// the events are streamed or only stored.
#[async_trait::async_trait]
pub trait A2aExecutor: Send + Sync {
    /// Runs one turn, publishing statuses and artifacts through `sink`.
    ///
    /// The server publishes the `SUBMITTED` snapshot and the `WORKING` status
    /// before calling this, and the terminal status after it returns `Ok` (an
    /// executor may publish its own terminal status first, which suppresses the
    /// server's `COMPLETED`).
    ///
    /// # Errors
    ///
    /// Any error is logged, published as a `FAILED` status, and — for a
    /// non-streaming call — reported as a `-32603` `JSON-RPC` error envelope.
    async fn execute(&self, ctx: A2aExecutionContext, sink: &EventSink) -> Result<()>;

    /// Cancels one task.
    ///
    /// The default body returns `Ok(())`; the server publishes the `CANCELED`
    /// status itself, so the observable behavior matches Python's default
    /// "emit Canceled" without forcing every executor to drive the sink.
    ///
    /// # Errors
    ///
    /// Return an error when the cancellation could not be attempted. The server
    /// logs it and publishes `CANCELED` anyway, because cancellation is best
    /// effort (Python's `_cancel_quietly`).
    async fn cancel(&self, _ctx: A2aExecutionContext) -> Result<()> {
        Ok(())
    }
}

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

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use axum::body::Body;
    use axum::http::{Method, Request, StatusCode};
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;
    use tower::ServiceExt;
    use types::AgentInterface;

    /// An executor that publishes an open artifact and a terminal one.
    struct EchoExecutor;

    #[async_trait::async_trait]
    impl A2aExecutor for EchoExecutor {
        async fn execute(&self, _ctx: A2aExecutionContext, sink: &EventSink) -> Result<()> {
            sink.artifact(
                Artifact {
                    parts: vec![Part::text("chunk")],
                    ..Artifact::default()
                },
                false,
            );
            sink.artifact(
                Artifact {
                    artifact_id: String::from("a2"),
                    parts: vec![Part::text("done")],
                    ..Artifact::default()
                },
                true,
            );
            Ok(())
        }
    }

    /// An executor that fails.
    struct FailingExecutor;

    #[async_trait::async_trait]
    impl A2aExecutor for FailingExecutor {
        async fn execute(&self, _ctx: A2aExecutionContext, _sink: &EventSink) -> Result<()> {
            Err(Error::validation("boom"))
        }
    }

    fn card() -> types::AgentCard {
        types::AgentCard {
            name: String::from("test-agent"),
            description: String::from("does things"),
            version: String::from("1.0"),
            capabilities: types::AgentCapabilities {
                streaming: true,
                push_notifications: false,
            },
            ..types::AgentCard::default()
        }
    }

    fn server(executor: Arc<dyn A2aExecutor>) -> A2aServer {
        A2aServer::new(A2aServerConfig::new(card()), executor).unwrap()
    }

    /// A stored task in `state`, for cancel/resubscribe tests.
    fn stored_task(id: &str, state: TaskState) -> Task {
        Task {
            id: String::from(id),
            context_id: String::from("ctx-1"),
            status: TaskStatus {
                state,
                message: None,
                timestamp: None,
            },
            artifacts: Vec::new(),
            history: Vec::new(),
            metadata: Map::new(),
        }
    }

    /// Drains a sink receiver without blocking.
    fn drain(receiver: &mut mpsc::UnboundedReceiver<Value>) -> Vec<Value> {
        std::iter::from_fn(|| receiver.try_recv().ok()).collect()
    }

    /// POSTs one JSON-RPC request through the router and returns the envelope.
    async fn rpc(server: &A2aServer, method_name: &str, params: Value) -> Value {
        let body = json!({
            "jsonrpc": "2.0",
            "id": 7,
            "method": method_name,
            "params": params,
        })
        .to_string();
        let request = Request::builder()
            .method(Method::POST)
            .uri("/a2a/jsonrpc/")
            .header("content-type", "application/json")
            .body(Body::from(body))
            .unwrap();
        let response = server.router().oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = http_body_util::BodyExt::collect(response.into_body())
            .await
            .unwrap()
            .to_bytes();
        serde_json::from_slice(&bytes).unwrap()
    }

    /// The `result` of a successful envelope, panicking on an error envelope.
    fn result(envelope: &Value) -> &Value {
        assert!(
            envelope.get("error").is_none(),
            "unexpected error: {envelope}"
        );
        &envelope["result"]
    }

    #[test]
    fn in_memory_task_store_put_get_and_all() {
        let store = InMemoryTaskStore::new();
        assert!(store.get("missing").is_none());
        assert_eq!(store.all().len(), 0);

        store.put(stored_task("t2", TaskState::Working));
        store.put(stored_task("t1", TaskState::Submitted));
        assert_eq!(store.get("t1").unwrap().status.state, TaskState::Submitted);
        // `all` is sorted by id.
        let ids: Vec<String> = store.all().into_iter().map(|task| task.id).collect();
        assert_eq!(ids, vec![String::from("t1"), String::from("t2")]);

        // `put` replaces by id.
        store.put(stored_task("t1", TaskState::Completed));
        assert_eq!(store.get("t1").unwrap().status.state, TaskState::Completed);
        assert_eq!(store.all().len(), 2);
    }

    #[test]
    fn event_sink_publishes_the_python_event_order() {
        // ports `tests/unit_tests/extensions/a2a/test_a2a_server.py` (event order).
        let store = Arc::new(InMemoryTaskStore::new());
        let (sink, mut receiver) = EventSink::new(
            String::from("t1"),
            String::from("c1"),
            Arc::clone(&store) as Arc<dyn TaskStore>,
        );
        sink.task(stored_task("t1", TaskState::Submitted));
        sink.working(Some(agent_message("working", "t1", "c1")));
        sink.artifact(
            Artifact {
                artifact_id: String::from("open"),
                parts: vec![Part::text("one")],
                ..Artifact::default()
            },
            false,
        );
        sink.artifact(
            Artifact {
                artifact_id: String::from("final"),
                parts: vec![Part::text("two")],
                ..Artifact::default()
            },
            true,
        );
        sink.complete();

        let events = drain(&mut receiver);
        assert_eq!(events.len(), 6, "unexpected events: {events:?}");
        // 1. the Task snapshot.
        assert_eq!(events[0]["id"], json!("t1"));
        assert_eq!(events[0]["status"]["state"], json!("TASK_STATE_SUBMITTED"));
        // 2. WORKING.
        assert_eq!(events[1]["status"]["state"], json!("TASK_STATE_WORKING"));
        assert_eq!(events[1]["taskId"], json!("t1"));
        assert_eq!(events[1]["contextId"], json!("c1"));
        // 3. the open artifact chunk.
        assert_eq!(events[2]["artifact"]["artifactId"], json!("open"));
        assert_eq!(events[2]["lastChunk"], json!(false));
        assert_eq!(events[2]["append"], json!(false));
        // 4. the terminal artifact chunk.
        assert_eq!(events[3]["artifact"]["artifactId"], json!("final"));
        assert_eq!(events[3]["lastChunk"], json!(true));
        // 5. the open artifact is closed without replacing its parts.
        assert_eq!(events[4]["artifact"]["artifactId"], json!("open"));
        assert_eq!(events[4]["artifact"]["parts"], json!([]));
        assert_eq!(events[4]["append"], json!(true));
        assert_eq!(events[4]["lastChunk"], json!(true));
        // 6. the terminal status.
        assert_eq!(events[5]["status"]["state"], json!("TASK_STATE_COMPLETED"));

        let stored = store.get("t1").unwrap();
        assert_eq!(stored.status.state, TaskState::Completed);
        assert_eq!(stored.artifacts.len(), 2);
        let open = &stored.artifacts[0];
        assert_eq!(open.artifact_id, "open");
        assert_eq!(open.parts, vec![Part::text("one")]);
        assert_eq!(stored.artifacts[1].artifact_id, "final");
    }

    #[test]
    fn terminal_events_are_idempotent() {
        let store = Arc::new(InMemoryTaskStore::new());
        let (sink, mut receiver) = EventSink::new(
            String::from("t1"),
            String::from("c1"),
            Arc::clone(&store) as Arc<dyn TaskStore>,
        );
        sink.task(stored_task("t1", TaskState::Working));
        sink.complete();
        assert!(sink.is_terminal());
        let after_first = drain(&mut receiver).len();

        // A second terminal call (any of them) publishes nothing.
        sink.complete();
        sink.fail("late");
        sink.canceled();
        sink.require_input(None);
        assert_eq!(drain(&mut receiver).len(), 0);
        assert_eq!(store.get("t1").unwrap().status.state, TaskState::Completed);
        assert!(after_first > 0);
    }

    #[test]
    fn fail_publishes_the_failure_text() {
        let store = Arc::new(InMemoryTaskStore::new());
        let (sink, mut receiver) = EventSink::new(
            String::from("t1"),
            String::from("c1"),
            Arc::clone(&store) as Arc<dyn TaskStore>,
        );
        sink.task(stored_task("t1", TaskState::Working));
        sink.fail("executor exploded");

        let events = drain(&mut receiver);
        let last = events.last().unwrap();
        assert_eq!(last["status"]["state"], json!("TASK_STATE_FAILED"));
        assert_eq!(
            last["status"]["message"]["parts"][0]["text"],
            json!("executor exploded")
        );
        assert_eq!(last["status"]["message"]["role"], json!("ROLE_AGENT"));
        assert_eq!(last["status"]["message"]["taskId"], json!("t1"));

        let stored = store.get("t1").unwrap();
        assert_eq!(stored.status.state, TaskState::Failed);
        let message = stored.status.message.unwrap();
        assert_eq!(message.parts, vec![Part::text("executor exploded")]);
    }

    #[test]
    fn config_defaults_match_python() {
        let config = A2aServerConfig::default();
        assert_eq!(config.rpc_path, "/a2a/jsonrpc/");
        assert_eq!(config.card_path, "/.well-known/agent-card.json");
        assert_eq!(config.protocol_version, "1.0");
        assert_eq!(config.host, "127.0.0.1");
        assert_eq!(config.port, 8000);
        assert!(config.interface_url.is_none());

        let config = A2aServerConfig::new(card())
            .with_interface_url("http://127.0.0.1:1/a2a/jsonrpc/")
            .with_bind("0.0.0.0", 0)
            .with_rpc_path("/rpc/");
        assert_eq!(config.agent_card.name, "test-agent");
        assert_eq!(
            config.interface_url.as_deref(),
            Some("http://127.0.0.1:1/a2a/jsonrpc/")
        );
        assert_eq!(config.host, "0.0.0.0");
        assert_eq!(config.port, 0);
        assert_eq!(config.rpc_path, "/rpc/");
    }

    #[test]
    fn new_rejects_a_grpc_interface() {
        let mut card = card();
        card.supported_interfaces = vec![AgentInterface {
            url: String::from("http://127.0.0.1:1"),
            protocol_binding: String::from("gRPC"),
            protocol_version: String::from("1.0"),
            tenant: None,
        }];
        let error = A2aServer::new(A2aServerConfig::new(card), Arc::new(EchoExecutor)).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert_eq!(
            error.to_string(),
            "unsupported: gRPC transport is not supported"
        );
    }

    #[tokio::test]
    async fn card_route_serves_the_agent_card() {
        let server = server(Arc::new(EchoExecutor));
        let request = Request::builder()
            .uri("/.well-known/agent-card.json")
            .body(Body::empty())
            .unwrap();
        let response = server.router().oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()[header::CONTENT_TYPE], "application/json");
        let bytes = http_body_util::BodyExt::collect(response.into_body())
            .await
            .unwrap()
            .to_bytes();
        let card: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(card["name"], json!("test-agent"));
        assert_eq!(card["capabilities"]["streaming"], json!(true));
    }

    #[tokio::test]
    async fn send_message_completes_the_task() {
        let server = server(Arc::new(EchoExecutor));
        let envelope = rpc(
            &server,
            "message/send",
            json!({ "message": { "role": "ROLE_USER", "parts": [{ "text": "hi" }] } }),
        )
        .await;
        let task = result(&envelope);
        assert_eq!(task["status"]["state"], json!("TASK_STATE_COMPLETED"));
        assert_eq!(task["id"].as_str().unwrap().len(), 32);
        assert_eq!(task["history"][0]["parts"][0]["text"], json!("hi"));
        assert_eq!(task["artifacts"].as_array().unwrap().len(), 2);
        // The stored task is the one the result reports.
        let stored = server.store().get(task["id"].as_str().unwrap()).unwrap();
        assert_eq!(stored.status.state, TaskState::Completed);
    }

    #[tokio::test]
    async fn send_message_with_the_v0_3_alias_dispatches() {
        let server = server(Arc::new(EchoExecutor));
        let envelope = rpc(&server, "message/send", json!({ "query": "hello" })).await;
        let task = result(&envelope);
        assert_eq!(task["status"]["state"], json!("TASK_STATE_COMPLETED"));
        assert_eq!(task["history"][0]["parts"][0]["text"], json!("hello"));
        assert_eq!(task["history"][0]["role"], json!("ROLE_USER"));
    }

    #[tokio::test]
    async fn unknown_method_reports_method_not_found() {
        let server = server(Arc::new(EchoExecutor));
        let envelope = rpc(&server, "message/teleport", json!({})).await;
        assert_eq!(
            envelope["error"]["code"],
            json!(error_code::METHOD_NOT_FOUND)
        );
        assert!(envelope["error"]["message"]
            .as_str()
            .unwrap()
            .contains("message/teleport"));
    }

    #[tokio::test]
    async fn get_task_reports_an_unknown_id() {
        let server = server(Arc::new(EchoExecutor));
        let envelope = rpc(&server, "tasks/get", json!({ "id": "nope" })).await;
        assert_eq!(envelope["error"]["code"], json!(error_code::TASK_NOT_FOUND));
        // A missing id is a params problem, not a lookup problem.
        let envelope = rpc(&server, "GetTask", json!({})).await;
        assert_eq!(envelope["error"]["code"], json!(error_code::INVALID_PARAMS));
    }

    #[tokio::test]
    async fn get_task_returns_a_stored_task() {
        let store = Arc::new(InMemoryTaskStore::new());
        store.put(stored_task("t1", TaskState::Working));
        let server = A2aServer::with_store(
            A2aServerConfig::new(card()),
            Arc::new(EchoExecutor),
            Arc::clone(&store) as Arc<dyn TaskStore>,
        )
        .unwrap();
        let envelope = rpc(&server, "tasks/get", json!({ "taskId": "t1" })).await;
        assert_eq!(result(&envelope)["id"], json!("t1"));
        assert_eq!(
            result(&envelope)["status"]["state"],
            json!("TASK_STATE_WORKING")
        );
    }

    #[tokio::test]
    async fn cancel_task_publishes_canceled() {
        let store = Arc::new(InMemoryTaskStore::new());
        store.put(stored_task("t1", TaskState::Working));
        let server = A2aServer::with_store(
            A2aServerConfig::new(card()),
            Arc::new(EchoExecutor),
            Arc::clone(&store) as Arc<dyn TaskStore>,
        )
        .unwrap();

        let envelope = rpc(&server, "tasks/cancel", json!({ "id": "t1" })).await;
        assert_eq!(
            result(&envelope)["status"]["state"],
            json!("TASK_STATE_CANCELED")
        );
        assert_eq!(store.get("t1").unwrap().status.state, TaskState::Canceled);

        // Cancelling a terminal task is refused.
        let envelope = rpc(&server, "CancelTask", json!({ "id": "t1" })).await;
        assert_eq!(
            envelope["error"]["code"],
            json!(error_code::TASK_NOT_CANCELABLE)
        );
    }

    #[tokio::test]
    async fn streaming_message_emits_the_python_frame_order() {
        let server = server(Arc::new(EchoExecutor));
        let body = json!({
            "jsonrpc": "2.0",
            "id": 11,
            "method": "message/stream",
            "params": { "message": { "parts": [{ "text": "hi" }] } },
        })
        .to_string();
        let request = Request::builder()
            .method(Method::POST)
            .uri("/a2a/jsonrpc/")
            .body(Body::from(body))
            .unwrap();
        let response = server.router().oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers()[header::CONTENT_TYPE],
            "text/event-stream"
        );
        assert_eq!(response.headers()[header::CACHE_CONTROL], "no-cache");

        let bytes = http_body_util::BodyExt::collect(response.into_body())
            .await
            .unwrap()
            .to_bytes();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        let frames: Vec<Value> = text
            .split("\n\n")
            .filter(|frame| !frame.trim().is_empty())
            .map(|frame| {
                let data = frame.strip_prefix("data: ").unwrap();
                serde_json::from_str::<Value>(data).unwrap()
            })
            .collect();

        assert_eq!(frames.len(), 6, "unexpected frames: {frames:?}");
        for frame in &frames {
            assert_eq!(frame["jsonrpc"], json!("2.0"));
            assert_eq!(frame["id"], json!(11));
        }
        let open_id = frames[2]["result"]["artifact"]["artifactId"]
            .as_str()
            .unwrap()
            .to_owned();
        assert_eq!(open_id.len(), 32);
        // 1. Task snapshot.
        assert_eq!(
            frames[0]["result"]["status"]["state"],
            json!("TASK_STATE_SUBMITTED")
        );
        // 2. WORKING.
        assert_eq!(
            frames[1]["result"]["status"]["state"],
            json!("TASK_STATE_WORKING")
        );
        // 3./4. the executor's artifacts.
        assert_eq!(frames[2]["result"]["lastChunk"], json!(false));
        assert_eq!(frames[3]["result"]["artifact"]["artifactId"], json!("a2"));
        assert_eq!(frames[3]["result"]["lastChunk"], json!(true));
        // 5. the open artifact is closed.
        assert_eq!(
            frames[4]["result"]["artifact"]["artifactId"],
            json!(open_id)
        );
        assert_eq!(frames[4]["result"]["append"], json!(true));
        assert_eq!(frames[4]["result"]["artifact"]["parts"], json!([]));
        // 6. COMPLETED.
        assert_eq!(
            frames[5]["result"]["status"]["state"],
            json!("TASK_STATE_COMPLETED")
        );
    }

    #[tokio::test]
    async fn executor_error_publishes_failed_and_a_jsonrpc_error() {
        let store = Arc::new(InMemoryTaskStore::new());
        let server = A2aServer::with_store(
            A2aServerConfig::new(card()),
            Arc::new(FailingExecutor),
            Arc::clone(&store) as Arc<dyn TaskStore>,
        )
        .unwrap();

        let envelope = rpc(&server, "SendMessage", json!({ "query": "hi" })).await;
        assert_eq!(envelope["error"]["code"], json!(error_code::INTERNAL));
        assert!(envelope["error"]["message"]
            .as_str()
            .unwrap()
            .contains("boom"));

        // The task is still stored, in FAILED, with the failure text.
        let tasks = store.all();
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].status.state, TaskState::Failed);
        let message = tasks[0].status.message.clone().unwrap();
        assert_eq!(message.parts, vec![Part::text("validation: boom")]);
    }

    #[tokio::test]
    async fn subscribe_to_task_answers_with_a_snapshot() {
        let store = Arc::new(InMemoryTaskStore::new());
        store.put(stored_task("t1", TaskState::Working));
        let server = A2aServer::with_store(
            A2aServerConfig::new(card()),
            Arc::new(EchoExecutor),
            Arc::clone(&store) as Arc<dyn TaskStore>,
        )
        .unwrap();

        let body = json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tasks/resubscribe",
            "params": { "id": "t1" },
        })
        .to_string();
        let request = Request::builder()
            .method(Method::POST)
            .uri("/a2a/jsonrpc/")
            .body(Body::from(body))
            .unwrap();
        let response = server.router().oneshot(request).await.unwrap();
        assert_eq!(
            response.headers()[header::CONTENT_TYPE],
            "text/event-stream"
        );
        let bytes = http_body_util::BodyExt::collect(response.into_body())
            .await
            .unwrap()
            .to_bytes();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        let frames: Vec<Value> = text
            .split("\n\n")
            .filter(|frame| !frame.trim().is_empty())
            .map(|frame| {
                serde_json::from_str::<Value>(frame.strip_prefix("data: ").unwrap()).unwrap()
            })
            .collect();
        assert_eq!(frames.len(), 2, "unexpected frames: {frames:?}");
        assert_eq!(frames[0]["result"]["id"], json!("t1"));
        assert_eq!(
            frames[1]["result"]["status"]["state"],
            json!("TASK_STATE_WORKING")
        );

        // An unknown id yields one error frame and ends.
        let body = json!({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "SubscribeToTask",
            "params": { "id": "missing" },
        })
        .to_string();
        let request = Request::builder()
            .method(Method::POST)
            .uri("/a2a/jsonrpc/")
            .body(Body::from(body))
            .unwrap();
        let response = server.router().oneshot(request).await.unwrap();
        let bytes = http_body_util::BodyExt::collect(response.into_body())
            .await
            .unwrap()
            .to_bytes();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        let frame: Value =
            serde_json::from_str(text.trim().strip_prefix("data: ").unwrap()).unwrap();
        assert_eq!(frame["error"]["code"], json!(error_code::TASK_NOT_FOUND));
    }

    #[tokio::test]
    async fn adapter_starts_and_stops_on_an_ephemeral_port() {
        let config = A2aServerConfig::new(card()).with_bind("127.0.0.1", 0);
        let server = A2aServer::new(config, Arc::new(EchoExecutor)).unwrap();
        let mut adapter = A2aServerAdapter::new(server);
        assert!(!adapter.active());
        assert!(adapter.local_addr().is_none());

        adapter.start().await.unwrap();
        assert!(adapter.active());
        let address = adapter.local_addr().unwrap();
        assert_eq!(address.ip().to_string(), "127.0.0.1");
        assert_ne!(address.port(), 0);

        // Starting twice keeps the first server.
        adapter.start().await.unwrap();
        assert_eq!(adapter.local_addr(), Some(address));

        adapter.stop().await.unwrap();
        assert!(!adapter.active());
        assert!(adapter.local_addr().is_none());
        // Stopping twice is a no-op.
        adapter.stop().await.unwrap();
    }
}
