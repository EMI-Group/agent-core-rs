//! Server lifecycle supervision: lazy start, crash detection, exponential-backoff restart.
//!
//! Ports `openjiuwen/harness/lsp/core/instance.py` (`LSPServerInstance`) without the Python
//! `ContentModified` retry loop: one [`ServerSupervisor`] owns exactly one [`ServerKey`], starts
//! its server on first use, and rebuilds it once after a crash.
//!
//! # States
//!
//! `Stopped → Starting → Running` on a successful handshake, `Starting → Failed` on a launch
//! or `initialize` failure, and `Running → Failed` when the client reports that the server
//! stopped. [`ServerStatus::healthy`] is only true in `Running` with a live client.
//!
//! # Crash recovery
//!
//! A crashed server is restarted at most [`RestartPolicy::max_attempts`] consecutive times.
//! [`RestartPolicy::delay_for`] is the deterministic back-off (`base × 2^(attempt-1)`, capped),
//! mirroring Python's `RETRY_BASE_DELAY_MS`/`MAX_CRASH_RECOVERY_ATTEMPTS`; the sleep goes
//! through the injected [`BackoffTimer`] so tests never wait on wall-clock time.
//!
//! # Concurrency
//!
//! Starts are serialized by a `tokio` mutex that is held only across the launch sequence (never
//! across a model-facing operation), because one server per key must be spawned exactly once.
//! The client slot and the status block are separate `parking_lot` mutexes: they are only held
//! for short, non-async updates.
//!
//! An explicit [`ServerSupervisor::shutdown`] is final: later `client()`/`run` calls return a
//! typed "has been shut down" error instead of silently restarting the process.
use std::future::Future;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex as AsyncMutex;

use openjiuwen_common::error::{Error, Result};

use super::client::LspClient;
use super::config::LspServerConfig;
use super::diagnostics::DiagnosticRegistry;
use super::discovery::{server_name, ServerKey};
use super::protocol::client_capabilities;
use super::transport::LspTransport;
use super::uri::path_to_uri;

/// Default number of consecutive crash-recovery attempts (Python's
/// `MAX_CRASH_RECOVERY_ATTEMPTS`).
const DEFAULT_MAX_ATTEMPTS: u32 = 3;
/// Default first back-off step (Python's `RETRY_BASE_DELAY_MS`).
const DEFAULT_BASE_DELAY: Duration = Duration::from_millis(500);
/// Default back-off ceiling.
const DEFAULT_MAX_DELAY: Duration = Duration::from_secs(8);

/// Lifecycle state of one supervised server.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum ServerState {
    /// Never started, or explicitly stopped.
    #[default]
    Stopped,
    /// Launch and `initialize` are in flight.
    Starting,
    /// Handshake completed; the client is usable.
    Running,
    /// Launch failed, `initialize` failed or the server stopped unexpectedly.
    Failed,
}

/// A snapshot of one supervisor's state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct ServerStatus {
    /// Identity of the supervised server instance.
    pub key: ServerKey,
    /// LSP language id the server was configured for.
    pub language_id: String,
    /// Current lifecycle state.
    pub state: ServerState,
    /// True only in [`ServerState::Running`] with a live client.
    pub healthy: bool,
    /// Consecutive failed starts / crash recoveries since the last successful start.
    pub attempts: u32,
    /// Message of the last launch or crash failure, when there was one.
    pub last_error: Option<String>,
}

/// Crash-recovery budget and back-off shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RestartPolicy {
    /// Consecutive attempts allowed before the supervisor reports a budget error.
    pub max_attempts: u32,
    /// Delay before the first retry.
    pub base_delay: Duration,
    /// Upper bound for any single back-off.
    pub max_delay: Duration,
}

impl Default for RestartPolicy {
    fn default() -> Self {
        Self {
            max_attempts: DEFAULT_MAX_ATTEMPTS,
            base_delay: DEFAULT_BASE_DELAY,
            max_delay: DEFAULT_MAX_DELAY,
        }
    }
}

impl RestartPolicy {
    /// The deterministic back-off for `attempt` (1-based): `base × 2^(attempt-1)`, capped.
    ///
    /// There is no jitter, so a test can assert the exact sequence. `attempt` values of 0 are
    /// treated as the first attempt.
    #[must_use]
    pub fn delay_for(&self, attempt: u32) -> Duration {
        let exponent = attempt.saturating_sub(1).min(31);
        let factor = 2_u32.checked_pow(exponent).unwrap_or(u32::MAX);
        self.base_delay
            .checked_mul(factor)
            .unwrap_or(self.max_delay)
            .min(self.max_delay)
    }
}

/// The sleep seam used between crash-recovery attempts.
///
/// Production uses [`TokioTimer`]; tests use [`InstantTimer`] and assert the requested delays.
#[async_trait]
pub trait BackoffTimer: Send + Sync {
    /// Waits for `delay` (or records it, for the test clock).
    async fn sleep(&self, delay: Duration);
}

/// The real clock: waits on the tokio timer.
#[derive(Debug, Clone, Copy, Default)]
pub struct TokioTimer;

impl TokioTimer {
    /// Creates the real timer.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

#[async_trait]
impl BackoffTimer for TokioTimer {
    async fn sleep(&self, delay: Duration) {
        tokio::time::sleep(delay).await;
    }
}

/// The deterministic test clock: records every requested delay and returns immediately.
#[derive(Debug, Clone, Default)]
pub struct InstantTimer {
    /// Every delay requested through [`BackoffTimer::sleep`], in order.
    delays: Arc<Mutex<Vec<Duration>>>,
}

impl InstantTimer {
    /// Creates a timer that has recorded nothing yet.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The delays requested so far, in call order.
    #[must_use]
    pub fn delays(&self) -> Vec<Duration> {
        self.delays.lock().clone()
    }
}

#[async_trait]
impl BackoffTimer for InstantTimer {
    async fn sleep(&self, delay: Duration) {
        self.delays.lock().push(delay);
    }
}

/// The launch seam: turns a config into a transport, or a typed error.
pub type ServerFactory =
    Arc<dyn Fn(&LspServerConfig) -> Result<Box<dyn LspTransport>> + Send + Sync>;

/// Mutable status fields, updated synchronously.
#[derive(Debug)]
struct StatusBlock {
    /// Current lifecycle state.
    state: ServerState,
    /// Failed start / crash recoveries since the last success.
    attempts: u32,
    /// Last failure message.
    last_error: Option<String>,
}

impl Default for StatusBlock {
    fn default() -> Self {
        Self {
            state: ServerState::Stopped,
            attempts: 0,
            last_error: None,
        }
    }
}

/// Owns one language-server instance: lazy start, one automatic crash retry, final shutdown.
pub struct ServerSupervisor {
    /// Server definition.
    config: LspServerConfig,
    /// Transport factory supplied by the host (process or in-memory).
    factory: ServerFactory,
    /// Optional shared diagnostics registry, attached to every started client.
    diagnostics: Option<Arc<DiagnosticRegistry>>,
    /// Back-off clock.
    timer: Arc<dyn BackoffTimer>,
    /// Recovery budget and back-off shape.
    policy: RestartPolicy,
    /// Per-request timeout handed to each client.
    request_timeout: Duration,
    /// Bound for the `initialize` handshake.
    startup_timeout: Duration,
    /// Root used when the config has none.
    workspace_root: Option<PathBuf>,
    /// The live client, when there is one.
    client: Mutex<Option<Arc<LspClient>>>,
    /// Serializes the launch sequence; held only across start.
    start_lock: AsyncMutex<()>,
    /// Lifecycle state, attempts and last error.
    status: Mutex<StatusBlock>,
    /// Set by [`ServerSupervisor::shutdown`]; forbids any later start.
    shutdown: AtomicBool,
}

impl ServerSupervisor {
    /// Creates a supervisor for `config` that launches through `factory`.
    #[must_use]
    pub fn new(config: LspServerConfig, factory: ServerFactory) -> Self {
        Self {
            startup_timeout: config.startup_timeout,
            config,
            factory,
            diagnostics: None,
            timer: Arc::new(TokioTimer),
            policy: RestartPolicy::default(),
            request_timeout: super::config::DEFAULT_REQUEST_TIMEOUT,
            workspace_root: None,
            client: Mutex::new(None),
            start_lock: AsyncMutex::new(()),
            status: Mutex::new(StatusBlock::default()),
            shutdown: AtomicBool::new(false),
        }
    }

    /// Attaches a diagnostics registry to every client this supervisor starts.
    #[must_use]
    pub fn with_diagnostics(mut self, registry: Arc<DiagnosticRegistry>) -> Self {
        self.diagnostics = Some(registry);
        self
    }

    /// Replaces the back-off clock.
    #[must_use]
    pub fn with_timer(mut self, timer: Arc<dyn BackoffTimer>) -> Self {
        self.timer = timer;
        self
    }

    /// Replaces the crash-recovery policy.
    #[must_use]
    pub const fn with_restart_policy(mut self, policy: RestartPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Sets the per-request timeout for started clients.
    #[must_use]
    pub const fn with_request_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = timeout;
        self
    }

    /// Sets the `initialize` bound.
    #[must_use]
    pub const fn with_startup_timeout(mut self, timeout: Duration) -> Self {
        self.startup_timeout = timeout;
        self
    }

    /// Sets the root used when the config has none (also the default `rootUri`).
    #[must_use]
    pub fn with_workspace_root(mut self, root: PathBuf) -> Self {
        self.workspace_root = Some(root);
        self
    }

    /// Identity of the supervised instance: server id plus effective root.
    #[must_use]
    pub fn key(&self) -> ServerKey {
        ServerKey {
            name: server_name(&self.config).to_owned(),
            root: self.effective_root(),
        }
    }

    /// The server definition this supervisor launches.
    #[must_use]
    pub fn config(&self) -> &LspServerConfig {
        &self.config
    }

    /// A snapshot of the lifecycle state.
    #[must_use]
    pub fn status(&self) -> ServerStatus {
        let state = self.status.lock();
        let alive = self
            .client
            .lock()
            .as_ref()
            .is_some_and(|client| client.is_alive());
        ServerStatus {
            key: self.key(),
            language_id: self.config.language_id.clone(),
            state: state.state,
            healthy: state.state == ServerState::Running && alive,
            attempts: state.attempts,
            last_error: state.last_error.clone(),
        }
    }

    /// Consecutive failed starts / crash recoveries since the last successful start.
    #[must_use]
    pub fn attempts(&self) -> u32 {
        self.status.lock().attempts
    }

    /// Returns a live client, starting the server on first use.
    ///
    /// # Errors
    ///
    /// Returns the launch error (attributed and counted) when the transport cannot be created
    /// or `initialize` fails or times out, the typed budget error once
    /// [`RestartPolicy::max_attempts`] is exhausted, and the typed "has been shut down" error
    /// after an explicit [`ServerSupervisor::shutdown`].
    pub async fn client(&self) -> Result<Arc<LspClient>> {
        self.ensure_running_allowed()?;
        if let Some(client) = self.live_client() {
            return Ok(client);
        }
        let _guard = self.start_lock.lock().await;
        self.ensure_running_allowed()?;
        if let Some(client) = self.live_client() {
            return Ok(client);
        }
        self.start().await
    }

    /// Runs `op` against a live client, retrying it once after a crash.
    ///
    /// The retry only happens when `op` failed *and* the client is dead; a live server's error
    /// (a JSON-RPC error, a timeout) is propagated unchanged. Before the retry the supervisor
    /// waits [`RestartPolicy::delay_for`] on the injected timer and restarts the server.
    ///
    /// # Errors
    ///
    /// Propagates `op`'s error when the client survived, and the launch, budget or shutdown
    /// error when the recovery itself fails; the retried `op` error is returned as-is.
    pub async fn run<T, F, Fut>(&self, op: F) -> Result<T>
    where
        F: Fn(Arc<LspClient>) -> Fut,
        Fut: Future<Output = Result<T>>,
    {
        let client = self.client().await?;
        match op(Arc::clone(&client)).await {
            Ok(value) => Ok(value),
            Err(error) => {
                if client.is_alive() {
                    return Err(error);
                }
                let attempt = {
                    let mut status = self.status.lock();
                    status.attempts = status.attempts.saturating_add(1);
                    status.last_error = client
                        .close_reason()
                        .or_else(|| Some(error.message().into_owned()));
                    status.attempts
                };
                let delay = self.policy.delay_for(attempt);
                tracing::warn!(
                    target: "openjiuwen_harness::lsp",
                    server = %self.name(),
                    attempt,
                    delay_ms = delay.as_millis(),
                    "LSP server crashed; restarting"
                );
                self.timer.sleep(delay).await;
                self.restart().await?;
                let fresh = self.client().await?;
                op(fresh).await
            }
        }
    }

    /// Stops the server (when running) and forbids any later start.
    ///
    /// Idempotent; every failure during teardown is best effort, like
    /// [`LspClient::shutdown`].
    pub async fn shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
        self.stop_current().await;
    }

    /// Stops the current server and starts a fresh one immediately.
    ///
    /// Used by hosts that know the workspace changed. The crash-recovery budget and the
    /// shutdown flag still apply.
    ///
    /// # Errors
    ///
    /// Returns the typed budget error when the attempts are exhausted, the shutdown error
    /// after [`ServerSupervisor::shutdown`], and the launch error when the restarted server
    /// cannot complete `initialize`.
    pub async fn restart(&self) -> Result<()> {
        self.ensure_running_allowed()?;
        self.stop_current().await;
        self.client().await?;
        Ok(())
    }

    /// The display name used in error messages.
    fn name(&self) -> &str {
        server_name(&self.config)
    }

    /// The root the server is started for: configured root, workspace root, else `.`.
    fn effective_root(&self) -> PathBuf {
        self.config
            .root
            .clone()
            .or_else(|| self.workspace_root.clone())
            .unwrap_or_else(|| PathBuf::from("."))
    }

    /// Fails with the typed shutdown error once [`ServerSupervisor::shutdown`] ran.
    fn ensure_running_allowed(&self) -> Result<()> {
        if self.shutdown.load(Ordering::SeqCst) {
            return Err(Error::transport(format!(
                "LSP server '{}' has been shut down",
                self.name()
            )));
        }
        Ok(())
    }

    /// The typed "too many recoveries" error.
    fn budget_error(&self) -> Error {
        let last = self
            .status
            .lock()
            .last_error
            .clone()
            .unwrap_or_else(|| "unknown".to_owned());
        Error::transport(format!(
            "LSP server '{}' exceeded max crash recovery attempts ({}); last error: {last}",
            self.name(),
            self.policy.max_attempts
        ))
    }

    /// Returns a live client, dropping a dead one and recording why it stopped.
    fn live_client(&self) -> Option<Arc<LspClient>> {
        let mut slot = self.client.lock();
        match slot.as_ref() {
            Some(client) if client.is_alive() => Some(Arc::clone(client)),
            Some(_) => {
                let dead = slot.take();
                drop(slot);
                if let Some(dead) = dead {
                    let mut status = self.status.lock();
                    if status.state == ServerState::Running {
                        status.state = ServerState::Failed;
                    }
                    status.last_error = dead
                        .close_reason()
                        .or_else(|| Some("LSP server stopped".to_owned()));
                }
                None
            }
            None => None,
        }
    }

    /// The serialized launch sequence: factory, client, diagnostics, initialize.
    async fn start(&self) -> Result<Arc<LspClient>> {
        let attempts = self.attempts();
        if attempts >= self.policy.max_attempts.max(1) {
            return Err(self.budget_error());
        }
        self.status.lock().state = ServerState::Starting;
        let transport = match (self.factory)(&self.config) {
            Ok(transport) => transport,
            Err(error) => return Err(self.record_failure(error)),
        };
        let client = Arc::new(LspClient::new(transport, self.request_timeout));
        if let Some(registry) = &self.diagnostics {
            registry.attach(&client);
        }
        let root_uri = path_to_uri(&self.effective_root());
        let handshake = tokio::time::timeout(
            self.startup_timeout,
            client.initialize(&root_uri, client_capabilities()),
        )
        .await;
        let outcome = match handshake {
            Ok(Ok(_)) => Ok(()),
            Ok(Err(error)) => Err(error),
            Err(_) => Err(Error::timeout(format!(
                "LSP server '{}' did not finish initialize within {} ms",
                self.name(),
                self.startup_timeout.as_millis()
            ))),
        };
        match outcome {
            Ok(()) => {
                *self.client.lock() = Some(Arc::clone(&client));
                let mut status = self.status.lock();
                status.state = ServerState::Running;
                status.attempts = 0;
                status.last_error = None;
                drop(status);
                tracing::info!(
                    target: "openjiuwen_harness::lsp",
                    server = %self.name(),
                    root = %self.effective_root().display(),
                    "LSP server started"
                );
                Ok(client)
            }
            Err(error) => {
                // Dropping the client asks its I/O task to close the transport; a
                // `StdioTransport` kills its child on drop.
                drop(client);
                Err(self.record_failure(error))
            }
        }
    }

    /// Records a failed start and returns either the original or the budget error.
    fn record_failure(&self, error: Error) -> Error {
        let exhausted = {
            let mut status = self.status.lock();
            status.state = ServerState::Failed;
            status.attempts = status.attempts.saturating_add(1);
            status.last_error = Some(error.message().into_owned());
            status.attempts >= self.policy.max_attempts.max(1)
        };
        tracing::warn!(
            target: "openjiuwen_harness::lsp",
            server = %self.name(),
            error = %error.message(),
            "LSP server failed to start"
        );
        if exhausted {
            self.budget_error()
        } else {
            error
        }
    }

    /// Stops and drops the current client, if any, and marks the server stopped.
    async fn stop_current(&self) {
        let existing = self.client.lock().take();
        if let Some(client) = existing {
            if client.is_alive() {
                client.shutdown().await;
            }
            drop(client);
        }
        self.status.lock().state = ServerState::Stopped;
    }
}
