//! The MCP-server manager: registration, connection state, and tool loading.
//!
//! Ports the MCP portion of Python
//! `openjiuwen/core/runner/resources_manager/` (v0.1.20): `add_mcp_server`,
//! `remove_mcp_server`, `get_mcp_server_config`, and `get_mcp_tool_ids` from
//! `resource_manager.py`, plus `ToolMgr`'s `McpServerResource` state and its
//! per-`server_id` async locks from `tool_manager.py`.
//!
//! The manager is the runner's authority on MCP servers: it stores each
//! [`McpServerConfig`] under its `server_id`, owns the client built for it, and
//! tracks the server through [`McpServerState`]. Registration never connects, so
//! [`McpServerMgr::register`] stays synchronous; [`McpServerMgr::connect`] and
//! [`McpServerMgr::load_tools`] establish the session (the latter connects
//! implicitly, mirroring Python's lazy `list_tools`).
//!
//! The client is built through [`McpConnector`] rather than by calling
//! [`connect_mcp_server`] directly. That seam exists for testability: the
//! connector is the only place a transport is constructed, so a unit test can
//! substitute a fake client. `openjiuwen_foundation`'s own `FakeMcpClient` is
//! `#[cfg(test)]`-gated to that crate and therefore not available here.
//!
//! Deferred (see the crate-level `## Deferred Python Behavior`): tags,
//! expiry-time refresh, name-based lookup, and Python's global tool registry.
//! Tool registration is an explicit opt-in through
//! [`McpServerMgr::register_tools`], not a side effect of connecting.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::tool::mcp::{
    connect_mcp_server, load_mcp_tools, McpClient, McpServerConfig, McpTool, McpToolOptions,
};
use openjiuwen_foundation::tool::Tool;
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};

use crate::error::{already_registered, invalid_id, not_found};
use crate::resources::{RegisterPolicy, Registration};

/// The noun used in every error message from this module.
const KIND: &str = "mcp server";

/// The lifecycle state of one registered MCP server.
///
/// Serialized internally tagged on `state`, so a unit state is
/// `{"state": "connected"}` and a failure carries its reason as
/// `{"state": "failed", "reason": "..."}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "state")]
pub enum McpServerState {
    /// Registered but never connected (or reconfigured since its last session).
    Registered,
    /// A live client is held; [`McpServerMgr::client`] returns it.
    Connected,
    /// The last connection attempt failed; no client is held.
    Failed {
        /// Human-readable reason, taken from the failing [`Error`]'s display.
        reason: String,
    },
    /// Explicitly disconnected (or removed and not yet re-registered).
    Disconnected,
}

/// A serializable projection of one registered MCP server.
///
/// [`McpServerMgr::list`] returns these in registration order. The auth
/// material of the underlying [`McpServerConfig`] is deliberately absent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct McpServerInfo {
    /// The resource key (`McpServerConfig::server_id`).
    pub server_id: String,
    /// The human-readable server name (`McpServerConfig::server_name`).
    pub server_name: String,
    /// The current lifecycle state.
    pub state: McpServerState,
}

/// Builds an [`McpClient`] for a server configuration without connecting it.
///
/// This is the manager's transport seam. [`DefaultMcpConnector`] delegates to
/// [`connect_mcp_server`] (the production transports); a test supplies its own
/// implementation because `openjiuwen_foundation`'s fake client is compiled
/// only inside that crate's own tests.
pub trait McpConnector: Send + Sync {
    /// Builds a lazily connecting client for `config`.
    ///
    /// # Errors
    ///
    /// Whatever the underlying transport factory reports: typically
    /// [`Error::Validation`] for an unusable configuration and
    /// [`Error::Unsupported`] for a deferred transport.
    fn build(&self, config: &McpServerConfig) -> Result<Arc<dyn McpClient>>;
}

/// The production connector: builds the transports `openjiuwen-foundation`
/// ships (`stdio`, `sse`, `streamable-http`, `openapi`).
#[derive(Debug, Clone, Copy, Default)]
pub struct DefaultMcpConnector;

impl McpConnector for DefaultMcpConnector {
    fn build(&self, config: &McpServerConfig) -> Result<Arc<dyn McpClient>> {
        connect_mcp_server(config)
    }
}

/// One stored server: its configuration, lifecycle state, and (once connected)
/// its live client.
#[derive(Clone)]
struct Entry {
    config: McpServerConfig,
    state: McpServerState,
    client: Option<Arc<dyn McpClient>>,
}

impl Entry {
    /// The serializable projection of this entry.
    fn info(&self) -> McpServerInfo {
        McpServerInfo {
            server_id: self.config.server_id.clone(),
            server_name: self.config.server_name.clone(),
            state: self.state.clone(),
        }
    }
}

/// The lock-guarded state of an [`McpServerMgr`].
///
/// Entries stay in registration order (`list`/`ids` preserve it) and are
/// indexed by `server_id`; removal compacts the vector exactly like
/// `Registry`.
struct Inner {
    /// Entries in registration order.
    entries: Vec<Entry>,
    /// Server id -> position in `entries`.
    index: HashMap<String, usize>,
    /// Mutation counter; bumped on registration, replacement, removal, a
    /// non-empty clear, and every observable state transition.
    revision: u64,
}

/// The runner's manager of MCP servers, keyed on `McpServerConfig::server_id`.
///
/// Registration does not connect; [`McpServerMgr::connect`] (or
/// [`McpServerMgr::load_tools`]) builds and connects the client, and
/// [`McpServerMgr::disconnect`]/[`McpServerMgr::remove`] tear it down.
///
/// Concurrent `connect`/`disconnect` calls for one server are serialized by a
/// per-server async lock (Python's per-`server_id` `asyncio.Lock`), so two
/// racing callers observe exactly one connection. Config, state, and client live
/// behind one `parking_lot::RwLock` whose guard is never held across an
/// `.await`.
pub struct McpServerMgr {
    /// Builds transports; see [`McpConnector`].
    connector: Arc<dyn McpConnector>,
    /// The registry state.
    inner: RwLock<Inner>,
    /// Per-server async locks, created on first use and kept for the manager's
    /// lifetime (Python keeps the same dictionary).
    locks: Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
}

impl McpServerMgr {
    /// Builds an empty manager that uses [`DefaultMcpConnector`].
    #[must_use]
    pub fn new() -> Self {
        Self::with_connector(Arc::new(DefaultMcpConnector))
    }

    /// Builds an empty manager that builds clients through `connector`.
    ///
    /// A host supplies its own connector to substitute the transport (tests,
    /// or an embedding that pre-wires clients).
    #[must_use]
    pub fn with_connector(connector: Arc<dyn McpConnector>) -> Self {
        Self {
            connector,
            inner: RwLock::new(Inner {
                entries: Vec::new(),
                index: HashMap::new(),
                revision: 0,
            }),
            locks: Mutex::new(HashMap::new()),
        }
    }

    /// Registers `config`, failing when its `server_id` is already present.
    ///
    /// Registration does not connect: the entry starts in
    /// [`McpServerState::Registered`] with no client.
    ///
    /// # Errors
    ///
    /// [`openjiuwen_common::error::Error::Validation`] for an empty or
    /// whitespace-only `server_id` and for a duplicate id
    /// ([`RegisterPolicy::Fail`]).
    pub fn register(&self, config: McpServerConfig) -> Result<()> {
        self.register_with(config, RegisterPolicy::Fail).map(|_| ())
    }

    /// Registers `config` under `policy`.
    ///
    /// [`RegisterPolicy::Replace`] swaps the configuration and returns the entry
    /// to [`McpServerState::Registered`], dropping the previous client **without
    /// awaiting its `disconnect`** (this method is synchronous); call
    /// [`McpServerMgr::disconnect`] or [`McpServerMgr::remove`] first when a
    /// clean shutdown matters.
    ///
    /// # Errors
    ///
    /// [`openjiuwen_common::error::Error::Validation`] for an empty or
    /// whitespace-only `server_id` and, under [`RegisterPolicy::Fail`], for a
    /// duplicate id.
    pub fn register_with(
        &self,
        config: McpServerConfig,
        policy: RegisterPolicy,
    ) -> Result<Registration> {
        let id = config.server_id.clone();
        if id.trim().is_empty() {
            return Err(invalid_id(KIND));
        }
        let name = config.server_name.clone();

        let mut dropped_client: Option<Arc<dyn McpClient>> = None;
        let registration = {
            let mut inner = self.inner.write();
            if let Some(position) = inner.index.get(&id).copied() {
                match policy {
                    RegisterPolicy::Fail => Err(already_registered(KIND, &id)),
                    RegisterPolicy::Keep => {
                        let existing_name = inner
                            .entries
                            .get(position)
                            .map_or_else(String::new, |entry| entry.config.server_name.clone());
                        Ok(Registration {
                            id: id.clone(),
                            name: existing_name,
                            replaced: false,
                        })
                    }
                    RegisterPolicy::Replace => {
                        if let Some(entry) = inner.entries.get_mut(position) {
                            dropped_client = entry.client.take();
                            entry.config = config;
                            entry.state = McpServerState::Registered;
                        }
                        inner.revision = inner.revision.saturating_add(1);
                        Ok(Registration {
                            id: id.clone(),
                            name: name.clone(),
                            replaced: true,
                        })
                    }
                }
            } else {
                let position = inner.entries.len();
                inner.entries.push(Entry {
                    config,
                    state: McpServerState::Registered,
                    client: None,
                });
                inner.index.insert(id.clone(), position);
                inner.revision = inner.revision.saturating_add(1);
                Ok(Registration {
                    id,
                    name,
                    replaced: false,
                })
            }
        };
        drop(dropped_client);

        if let Ok(ref registration) = registration {
            tracing::debug!(
                server_id = %registration.id,
                server_name = %registration.name,
                replaced = registration.replaced,
                "registered MCP server"
            );
        }
        registration
    }

    /// Returns a clone of the configuration registered under `id`.
    #[must_use]
    pub fn config(&self, id: &str) -> Option<McpServerConfig> {
        self.entry(id).map(|entry| entry.config)
    }

    /// Returns the current state of the server registered under `id`.
    #[must_use]
    pub fn state(&self, id: &str) -> Option<McpServerState> {
        self.entry(id).map(|entry| entry.state)
    }

    /// Returns the live client of `id`, or `None` when no session is held.
    ///
    /// Only [`McpServerState::Connected`] entries carry a client.
    #[must_use]
    pub fn client(&self, id: &str) -> Option<Arc<dyn McpClient>> {
        self.entry(id).and_then(|entry| entry.client)
    }

    /// Connects the server `id`, building its client first if needed.
    ///
    /// Already-connected servers are a no-op. Concurrent calls for the same id
    /// are serialized, so the underlying client's `connect` runs once.
    ///
    /// # Errors
    ///
    /// [`openjiuwen_common::error::Error::NotFound`] for an unregistered id;
    /// otherwise whatever the connector's `build` or the client's `connect`
    /// reported, with the entry left in
    /// [`McpServerState::Failed`] and no client.
    pub async fn connect(&self, id: &str) -> Result<()> {
        let lock = self.server_lock(id);
        let _guard = lock.lock().await;

        let Some(entry) = self.entry(id) else {
            return Err(not_found(KIND, id));
        };
        if matches!(entry.state, McpServerState::Connected) && entry.client.is_some() {
            return Ok(());
        }

        let client = match self.connector.build(&entry.config) {
            Ok(client) => client,
            Err(error) => {
                tracing::warn!(
                    server_id = %id,
                    error = %error,
                    "MCP server client could not be built"
                );
                self.record_failure(id, &entry.config, error.to_string());
                return Err(error);
            }
        };

        if let Err(error) = client.connect().await {
            tracing::warn!(
                server_id = %id,
                error = %error,
                "MCP server connection failed"
            );
            self.record_failure(id, &entry.config, error.to_string());
            return Err(error);
        }

        self.record_connected(id, &entry.config, client)?;
        tracing::debug!(server_id = %id, "connected MCP server");
        Ok(())
    }

    /// Loads every tool the server advertises, connecting it first if needed.
    ///
    /// The cards are those [`load_mcp_tools`] projects: id
    /// `"{server_id}.{server_name}.{tool}"`, the remote name, description, and
    /// input schema.
    ///
    /// # Errors
    ///
    /// [`openjiuwen_common::error::Error::NotFound`] for an unregistered id, the
    /// [`McpServerMgr::connect`] error when connecting fails, or the client's
    /// `list_tools` error.
    pub async fn load_tools(&self, id: &str) -> Result<Vec<McpTool>> {
        self.connect(id).await?;
        let client = self.client(id).ok_or_else(|| not_found(KIND, id))?;
        load_mcp_tools(client, McpToolOptions::default()).await
    }

    /// Publishes the server's tools into `mgr` and returns their tool ids.
    ///
    /// Each [`McpTool`] is registered as an `Arc<dyn Tool>` with
    /// [`RegisterPolicy::Keep`], so a repeated call does not fail on the ids it
    /// published last time. The returned ids come from the tools' cards.
    ///
    /// # Errors
    ///
    /// The [`McpServerMgr::load_tools`] error, or the
    /// [`ResourceMgr`](crate::ResourceMgr) registration error (for example an
    /// empty projected id).
    pub async fn register_tools(&self, mgr: &crate::ResourceMgr, id: &str) -> Result<Vec<String>> {
        let tools = self.load_tools(id).await?;
        let mut tool_ids = Vec::with_capacity(tools.len());
        for tool in tools {
            let tool_id = tool.card().id.clone();
            mgr.register_tool_with(tool.boxed(), RegisterPolicy::Keep)?;
            tool_ids.push(tool_id);
        }
        Ok(tool_ids)
    }

    /// Disconnects the server `id` and drops its client.
    ///
    /// A server that is registered but never connected (or already
    /// disconnected) is a no-op and moves to
    /// [`McpServerState::Disconnected`]. The client reference is dropped even
    /// when the underlying `disconnect` fails.
    ///
    /// # Errors
    ///
    /// [`openjiuwen_common::error::Error::NotFound`] for an unregistered id,
    /// else the client's `disconnect` error (the entry is still marked
    /// disconnected).
    pub async fn disconnect(&self, id: &str) -> Result<()> {
        let lock = self.server_lock(id);
        let _guard = lock.lock().await;

        let Some(entry) = self.entry(id) else {
            return Err(not_found(KIND, id));
        };
        let outcome = match &entry.client {
            Some(client) => client.disconnect().await,
            None => Ok(()),
        };
        if let Err(ref error) = outcome {
            tracing::warn!(
                server_id = %id,
                error = %error,
                "MCP server disconnect failed; dropping the client anyway"
            );
        }
        self.mark_disconnected(id);
        tracing::debug!(server_id = %id, "disconnected MCP server");
        outcome
    }

    /// Disconnects the server `id` and unregisters it, returning its config.
    ///
    /// A failing `disconnect` is logged and does not prevent the removal
    /// (Python removes the resource regardless).
    ///
    /// # Errors
    ///
    /// [`openjiuwen_common::error::Error::NotFound`] for an unregistered id.
    pub async fn remove(&self, id: &str) -> Result<McpServerConfig> {
        let lock = self.server_lock(id);
        let _guard = lock.lock().await;

        let Some(entry) = self.entry(id) else {
            return Err(not_found(KIND, id));
        };
        if let Some(client) = &entry.client {
            if let Err(error) = client.disconnect().await {
                tracing::warn!(
                    server_id = %id,
                    error = %error,
                    "MCP server disconnect failed during removal; removing anyway"
                );
            }
        }
        let config = self.unregister(id).ok_or_else(|| not_found(KIND, id))?;
        tracing::debug!(server_id = %id, "removed MCP server");
        Ok(config)
    }

    /// Returns a projection of every server in registration order.
    #[must_use]
    pub fn list(&self) -> Vec<McpServerInfo> {
        self.inner.read().entries.iter().map(Entry::info).collect()
    }

    /// Returns the registered ids in registration order.
    #[must_use]
    pub fn ids(&self) -> Vec<String> {
        self.inner
            .read()
            .entries
            .iter()
            .map(|entry| entry.config.server_id.clone())
            .collect()
    }

    /// Returns the number of registered servers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.read().entries.len()
    }

    /// Returns whether no server is registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.read().entries.is_empty()
    }

    /// Returns the mutation counter.
    ///
    /// It increments on registration, replacement, removal, a non-empty clear,
    /// and every observable state transition (connect success, connect failure,
    /// disconnect).
    #[must_use]
    pub fn revision(&self) -> u64 {
        self.inner.read().revision
    }

    /// Removes every server and returns how many were removed.
    ///
    /// Held clients are dropped **without** awaiting `disconnect` (this method
    /// is synchronous); a non-empty clear increments the revision by one.
    pub fn clear(&self) -> usize {
        self.drain(false)
    }

    /// Removes every server, drops their clients, and zeroes the revision.
    ///
    /// The runner uses this when it releases its resources.
    #[allow(dead_code)] // the facade wires this into its resource release path.
    pub(crate) fn reset(&self) -> usize {
        self.drain(true)
    }

    /// Returns a clone of the stored entry for `id`.
    fn entry(&self, id: &str) -> Option<Entry> {
        let inner = self.inner.read();
        inner
            .index
            .get(id)
            .and_then(|position| inner.entries.get(*position))
            .cloned()
    }

    /// Returns the async lock guarding transitions of `id`.
    ///
    /// The map entry is created on first use and retained afterwards, matching
    /// Python's per-`server_id` lock dictionary.
    fn server_lock(&self, id: &str) -> Arc<tokio::sync::Mutex<()>> {
        let mut locks = self.locks.lock();
        Arc::clone(locks.entry(id.to_owned()).or_default())
    }

    /// Records a successful connection, unless the entry was reconfigured while
    /// the connection was being established.
    fn record_connected(
        &self,
        id: &str,
        config: &McpServerConfig,
        client: Arc<dyn McpClient>,
    ) -> Result<()> {
        let mut inner = self.inner.write();
        let Some(position) = inner.index.get(id).copied() else {
            return Err(not_found(KIND, id));
        };
        let Some(entry) = inner.entries.get_mut(position) else {
            return Err(not_found(KIND, id));
        };
        if entry.config != *config {
            return Err(Error::validation(format!(
                "mcp server '{id}' was reconfigured during connect"
            )));
        }
        let previous = entry.client.replace(client);
        entry.state = McpServerState::Connected;
        inner.revision = inner.revision.saturating_add(1);
        drop(inner);
        drop(previous);
        Ok(())
    }

    /// Records a failed connection attempt.
    ///
    /// The state becomes [`McpServerState::Failed`] and the client slot is
    /// cleared. A reconfigured entry is left untouched.
    fn record_failure(&self, id: &str, config: &McpServerConfig, reason: String) {
        let mut inner = self.inner.write();
        let Some(position) = inner.index.get(id).copied() else {
            return;
        };
        let (changed, previous) = {
            let Some(entry) = inner.entries.get_mut(position) else {
                return;
            };
            if entry.config != *config {
                return;
            }
            let previous = entry.client.take();
            let changed =
                previous.is_some() || !matches!(entry.state, McpServerState::Failed { .. });
            entry.state = McpServerState::Failed { reason };
            (changed, previous)
        };
        if changed {
            inner.revision = inner.revision.saturating_add(1);
        }
        drop(inner);
        drop(previous);
    }

    /// Marks `id` disconnected and drops its client.
    fn mark_disconnected(&self, id: &str) {
        let mut inner = self.inner.write();
        let Some(position) = inner.index.get(id).copied() else {
            return;
        };
        let (changed, previous) = {
            let Some(entry) = inner.entries.get_mut(position) else {
                return;
            };
            let changed = entry.state != McpServerState::Disconnected || entry.client.is_some();
            entry.state = McpServerState::Disconnected;
            (changed, entry.client.take())
        };
        if changed {
            inner.revision = inner.revision.saturating_add(1);
        }
        drop(inner);
        drop(previous);
    }

    /// Removes `id` from the registry and returns its configuration.
    fn unregister(&self, id: &str) -> Option<McpServerConfig> {
        let mut inner = self.inner.write();
        // Reborrow to a plain `&mut Inner` so `entries` and `index` can be
        // borrowed disjointly while the shifted ids are reindexed.
        let inner: &mut Inner = &mut inner;
        let position = inner.index.remove(id)?;
        let entry = inner.entries.remove(position);
        for (offset, shifted) in inner.entries.iter().enumerate().skip(position) {
            inner.index.insert(shifted.config.server_id.clone(), offset);
        }
        inner.revision = inner.revision.saturating_add(1);
        Some(entry.config)
    }

    /// Clears the registry; `reset_revision` selects `reset` semantics.
    fn drain(&self, reset_revision: bool) -> usize {
        let mut inner = self.inner.write();
        let removed = inner.entries.len();
        let clients: Vec<Arc<dyn McpClient>> = inner
            .entries
            .drain(..)
            .filter_map(|mut entry| entry.client.take())
            .collect();
        inner.index.clear();
        if reset_revision {
            inner.revision = 0;
        } else if removed > 0 {
            inner.revision = inner.revision.saturating_add(1);
        }
        drop(inner);
        drop(clients);
        removed
    }
}

impl Default for McpServerMgr {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for McpServerMgr {
    /// Prints the kind, ids in registration order, and the revision.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let inner = self.inner.read();
        let ids: Vec<&str> = inner
            .entries
            .iter()
            .map(|entry| entry.config.server_id.as_str())
            .collect();
        formatter
            .debug_struct("McpServerMgr")
            .field("ids", &ids)
            .field("revision", &inner.revision)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use openjiuwen_common::error::ErrorKind;
    use openjiuwen_foundation::tool::mcp::{McpRemoteTool, McpToolResult};
    use serde_json::{json, Value};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Mutex as StdMutex;
    use std::time::Duration;

    /// A deterministic `McpClient`; failures are configured through flags.
    struct FakeClient {
        server_id: String,
        server_name: String,
        tools: Vec<McpRemoteTool>,
        connects: AtomicUsize,
        disconnects: AtomicUsize,
        connected: AtomicBool,
        connect_error: StdMutex<Option<String>>,
        disconnect_error: StdMutex<Option<String>>,
    }

    impl FakeClient {
        fn new(server_id: &str, server_name: &str, tools: Vec<McpRemoteTool>) -> Arc<Self> {
            Arc::new(Self {
                server_id: server_id.to_owned(),
                server_name: server_name.to_owned(),
                tools,
                connects: AtomicUsize::new(0),
                disconnects: AtomicUsize::new(0),
                connected: AtomicBool::new(false),
                connect_error: StdMutex::new(None),
                disconnect_error: StdMutex::new(None),
            })
        }

        fn fail_connect(&self, reason: &str) {
            *self.connect_error.lock().unwrap() = Some(reason.to_owned());
        }

        fn fail_disconnect(&self, reason: &str) {
            *self.disconnect_error.lock().unwrap() = Some(reason.to_owned());
        }

        fn connect_count(&self) -> usize {
            self.connects.load(Ordering::SeqCst)
        }

        fn disconnect_count(&self) -> usize {
            self.disconnects.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl McpClient for FakeClient {
        fn server_name(&self) -> &str {
            &self.server_name
        }

        fn server_id(&self) -> &str {
            &self.server_id
        }

        async fn connect(&self) -> Result<()> {
            self.connects.fetch_add(1, Ordering::SeqCst);
            // Yield so two concurrent callers really interleave in the manager.
            tokio::task::yield_now().await;
            if let Some(reason) = self.connect_error.lock().unwrap().clone() {
                return Err(Error::Mcp(reason));
            }
            self.connected.store(true, Ordering::SeqCst);
            Ok(())
        }

        async fn disconnect(&self) -> Result<()> {
            self.disconnects.fetch_add(1, Ordering::SeqCst);
            if let Some(reason) = self.disconnect_error.lock().unwrap().clone() {
                return Err(Error::Mcp(reason));
            }
            self.connected.store(false, Ordering::SeqCst);
            Ok(())
        }

        async fn is_connected(&self) -> bool {
            self.connected.load(Ordering::SeqCst)
        }

        async fn list_tools(&self) -> Result<Vec<McpRemoteTool>> {
            Ok(self.tools.clone())
        }

        async fn call_tool_with(
            &self,
            _tool_name: &str,
            _arguments: Value,
            _timeout: Option<Duration>,
            _include_image_content: bool,
        ) -> Result<McpToolResult> {
            Err(Error::unsupported("the fake client does not call tools"))
        }
    }

    /// A connector returning a shared fake client (or a fixed failure).
    struct FakeConnector {
        client: Arc<FakeClient>,
        builds: AtomicUsize,
        error: StdMutex<Option<String>>,
    }

    impl FakeConnector {
        fn new(client: Arc<FakeClient>) -> Arc<Self> {
            Arc::new(Self {
                client,
                builds: AtomicUsize::new(0),
                error: StdMutex::new(None),
            })
        }

        fn fail(&self, reason: &str) {
            *self.error.lock().unwrap() = Some(reason.to_owned());
        }

        fn builds(&self) -> usize {
            self.builds.load(Ordering::SeqCst)
        }
    }

    impl McpConnector for FakeConnector {
        fn build(&self, _config: &McpServerConfig) -> Result<Arc<dyn McpClient>> {
            self.builds.fetch_add(1, Ordering::SeqCst);
            if let Some(reason) = self.error.lock().unwrap().clone() {
                return Err(Error::Mcp(reason));
            }
            Ok(Arc::clone(&self.client) as Arc<dyn McpClient>)
        }
    }

    /// One manager plus the fakes it talks to.
    struct Fixture {
        mgr: McpServerMgr,
        client: Arc<FakeClient>,
        connector: Arc<FakeConnector>,
    }

    fn fixture() -> Fixture {
        let client = FakeClient::new(
            "s1",
            "alpha",
            vec![McpRemoteTool::new(
                "echo",
                "echoes its input",
                json!({"type": "object"}),
            )],
        );
        let connector = FakeConnector::new(Arc::clone(&client));
        let mgr = McpServerMgr::with_connector(Arc::clone(&connector) as Arc<dyn McpConnector>);
        Fixture {
            mgr,
            client,
            connector,
        }
    }

    fn config(server_id: &str, server_name: &str) -> McpServerConfig {
        McpServerConfig::stdio(server_name, "fake-mcp").with_server_id(server_id)
    }

    #[test]
    fn states_round_trip_with_their_tagged_shape() {
        let cases = [
            (McpServerState::Registered, json!({"state": "registered"})),
            (McpServerState::Connected, json!({"state": "connected"})),
            (
                McpServerState::Failed {
                    reason: "boom".to_owned(),
                },
                json!({"state": "failed", "reason": "boom"}),
            ),
            (
                McpServerState::Disconnected,
                json!({"state": "disconnected"}),
            ),
        ];
        for (state, expected) in cases {
            let value = serde_json::to_value(&state).unwrap();
            assert_eq!(value, expected);
            let back: McpServerState = serde_json::from_value(value).unwrap();
            assert_eq!(back, state);
        }
    }

    #[test]
    fn server_info_round_trips() {
        let info = McpServerInfo {
            server_id: "s1".to_owned(),
            server_name: "alpha".to_owned(),
            state: McpServerState::Failed {
                reason: "nope".to_owned(),
            },
        };
        let value = serde_json::to_value(&info).unwrap();
        assert_eq!(
            value,
            json!({
                "server_id": "s1",
                "server_name": "alpha",
                "state": {"state": "failed", "reason": "nope"},
            })
        );
        let back: McpServerInfo = serde_json::from_value(value).unwrap();
        assert_eq!(back, info);
    }

    #[tokio::test]
    async fn register_lists_servers_in_registration_order() {
        let f = fixture();
        f.mgr.register(config("s1", "alpha")).unwrap();
        f.mgr.register(config("s2", "beta")).unwrap();

        assert_eq!(f.mgr.ids(), vec!["s1", "s2"]);
        assert_eq!(f.mgr.len(), 2);
        assert!(!f.mgr.is_empty());
        assert_eq!(f.mgr.revision(), 2);
        assert_eq!(f.mgr.state("s1"), Some(McpServerState::Registered));
        assert!(f.mgr.client("s1").is_none());
        assert_eq!(f.mgr.config("s2").unwrap().server_name, "beta");

        let infos = f.mgr.list();
        assert_eq!(infos.len(), 2);
        assert_eq!(infos[0].server_id, "s1");
        assert_eq!(infos[0].server_name, "alpha");
        assert_eq!(infos[0].state, McpServerState::Registered);
        assert_eq!(infos[1].server_id, "s2");
        assert_eq!(infos[1].server_name, "beta");
    }

    #[tokio::test]
    async fn duplicate_registration_fails_under_fail_policy() {
        let f = fixture();
        f.mgr.register(config("s1", "alpha")).unwrap();

        let error = f.mgr.register(config("s1", "other")).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert_eq!(
            error.to_string(),
            "validation: mcp server 's1' is already registered"
        );
        assert_eq!(f.mgr.config("s1").unwrap().server_name, "alpha");
        assert_eq!(f.mgr.revision(), 1);
    }

    #[tokio::test]
    async fn keep_and_replace_policies_update_the_entry_as_documented() {
        let f = fixture();
        f.mgr.register(config("s1", "alpha")).unwrap();

        let kept = f
            .mgr
            .register_with(config("s1", "replacement"), RegisterPolicy::Keep)
            .unwrap();
        assert!(!kept.replaced);
        assert_eq!(kept.name, "alpha");
        assert_eq!(f.mgr.config("s1").unwrap().server_name, "alpha");
        assert_eq!(f.mgr.revision(), 1);

        f.mgr.connect("s1").await.unwrap();
        assert!(f.mgr.client("s1").is_some());

        let replaced = f
            .mgr
            .register_with(config("s1", "replacement"), RegisterPolicy::Replace)
            .unwrap();
        assert!(replaced.replaced);
        assert_eq!(replaced.name, "replacement");
        assert_eq!(f.mgr.config("s1").unwrap().server_name, "replacement");
        assert_eq!(f.mgr.state("s1"), Some(McpServerState::Registered));
        assert!(f.mgr.client("s1").is_none());
        assert_eq!(f.mgr.revision(), 3);
    }

    #[tokio::test]
    async fn empty_server_ids_are_rejected() {
        let f = fixture();
        for id in ["", "   ", "\t\n"] {
            let error = f.mgr.register(config(id, "ghost")).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Validation);
            assert_eq!(
                error.to_string(),
                "validation: mcp server id cannot be empty or whitespace only"
            );
        }
        assert!(f.mgr.is_empty());
        assert_eq!(f.mgr.revision(), 0);
    }

    #[tokio::test]
    async fn connect_transitions_to_connected_and_exposes_the_client() {
        let f = fixture();
        f.mgr.register(config("s1", "alpha")).unwrap();

        f.mgr.connect("s1").await.unwrap();

        assert_eq!(f.mgr.state("s1"), Some(McpServerState::Connected));
        assert!(f.mgr.client("s1").is_some());
        assert_eq!(f.client.connect_count(), 1);
        assert_eq!(f.connector.builds(), 1);
        assert_eq!(f.mgr.revision(), 2);
    }

    #[tokio::test]
    async fn connect_is_idempotent() {
        let f = fixture();
        f.mgr.register(config("s1", "alpha")).unwrap();

        f.mgr.connect("s1").await.unwrap();
        f.mgr.connect("s1").await.unwrap();

        assert_eq!(f.client.connect_count(), 1);
        assert_eq!(f.connector.builds(), 1);
        assert_eq!(f.mgr.revision(), 2);
    }

    #[tokio::test]
    async fn a_failing_connector_surfaces_the_error_and_marks_the_server_failed() {
        let f = fixture();
        f.mgr.register(config("s1", "alpha")).unwrap();
        f.connector.fail("cannot build");

        let error = f.mgr.connect("s1").await.unwrap_err();

        assert_eq!(error.kind(), ErrorKind::Mcp);
        assert_eq!(error.to_string(), "mcp: cannot build");
        assert_eq!(
            f.mgr.state("s1"),
            Some(McpServerState::Failed {
                reason: "mcp: cannot build".to_owned()
            })
        );
        assert!(f.mgr.client("s1").is_none());
        assert_eq!(f.client.connect_count(), 0);
    }

    #[tokio::test]
    async fn a_failing_connect_surfaces_the_error_and_marks_the_server_failed() {
        let f = fixture();
        f.mgr.register(config("s1", "alpha")).unwrap();
        f.client.fail_connect("transport down");

        let error = f.mgr.connect("s1").await.unwrap_err();

        assert_eq!(error.kind(), ErrorKind::Mcp);
        assert_eq!(error.to_string(), "mcp: transport down");
        assert_eq!(
            f.mgr.state("s1"),
            Some(McpServerState::Failed {
                reason: "mcp: transport down".to_owned()
            })
        );
        assert!(f.mgr.client("s1").is_none());
        assert_eq!(f.client.connect_count(), 1);
    }

    #[tokio::test]
    async fn a_failed_server_can_be_connected_after_the_cause_is_cleared() {
        let f = fixture();
        f.mgr.register(config("s1", "alpha")).unwrap();
        f.client.fail_connect("transport down");
        assert!(f.mgr.connect("s1").await.is_err());

        *f.client.connect_error.lock().unwrap() = None;
        f.mgr.connect("s1").await.unwrap();

        assert_eq!(f.mgr.state("s1"), Some(McpServerState::Connected));
        assert!(f.mgr.client("s1").is_some());
        assert_eq!(f.client.connect_count(), 2);
    }

    #[tokio::test]
    async fn load_tools_connects_implicitly() {
        let f = fixture();
        f.mgr.register(config("s1", "alpha")).unwrap();

        let tools = f.mgr.load_tools("s1").await.unwrap();

        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].card().id, "s1.alpha.echo");
        assert_eq!(tools[0].card().name, "echo");
        assert_eq!(f.mgr.state("s1"), Some(McpServerState::Connected));
        assert_eq!(f.client.connect_count(), 1);
    }

    #[tokio::test]
    async fn register_tools_publishes_the_cards_into_the_resource_mgr() {
        let f = fixture();
        f.mgr.register(config("s1", "alpha")).unwrap();
        let resources = crate::ResourceMgr::new();

        let ids = f.mgr.register_tools(&resources, "s1").await.unwrap();

        assert_eq!(ids, vec!["s1.alpha.echo".to_owned()]);
        assert_eq!(resources.tool_count(), 1);
        assert_eq!(resources.tool_card("s1.alpha.echo").unwrap().name, "echo");
        assert_eq!(
            resources.tool("s1.alpha.echo").unwrap().card().id,
            "s1.alpha.echo"
        );

        // Keep semantics: republishing is a no-op rather than a duplicate error.
        let again = f.mgr.register_tools(&resources, "s1").await.unwrap();
        assert_eq!(again, vec!["s1.alpha.echo".to_owned()]);
        assert_eq!(resources.tool_count(), 1);
    }

    #[tokio::test]
    async fn disconnect_drops_the_client_and_marks_the_server_disconnected() {
        let f = fixture();
        f.mgr.register(config("s1", "alpha")).unwrap();
        f.mgr.connect("s1").await.unwrap();

        f.mgr.disconnect("s1").await.unwrap();

        assert_eq!(f.mgr.state("s1"), Some(McpServerState::Disconnected));
        assert!(f.mgr.client("s1").is_none());
        assert_eq!(f.client.disconnect_count(), 1);

        // Already disconnected: no client, still Ok.
        f.mgr.disconnect("s1").await.unwrap();
        assert_eq!(f.client.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn disconnect_on_a_never_connected_server_is_ok() {
        let f = fixture();
        f.mgr.register(config("s1", "alpha")).unwrap();

        f.mgr.disconnect("s1").await.unwrap();

        assert_eq!(f.mgr.state("s1"), Some(McpServerState::Disconnected));
        assert_eq!(f.client.disconnect_count(), 0);
    }

    #[tokio::test]
    async fn a_failing_disconnect_still_marks_the_server_disconnected() {
        let f = fixture();
        f.mgr.register(config("s1", "alpha")).unwrap();
        f.mgr.connect("s1").await.unwrap();
        f.client.fail_disconnect("socket closed");

        let error = f.mgr.disconnect("s1").await.unwrap_err();

        assert_eq!(error.kind(), ErrorKind::Mcp);
        assert_eq!(error.to_string(), "mcp: socket closed");
        assert_eq!(f.mgr.state("s1"), Some(McpServerState::Disconnected));
        assert!(f.mgr.client("s1").is_none());
    }

    #[tokio::test]
    async fn remove_disconnects_and_unregisters() {
        let f = fixture();
        f.mgr.register(config("s1", "alpha")).unwrap();
        f.mgr.connect("s1").await.unwrap();

        let config = f.mgr.remove("s1").await.unwrap();

        assert_eq!(config.server_id, "s1");
        assert_eq!(config.server_name, "alpha");
        assert!(f.mgr.is_empty());
        assert!(f.mgr.config("s1").is_none());
        assert_eq!(f.client.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn remove_still_removes_when_the_disconnect_fails() {
        let f = fixture();
        f.mgr.register(config("s1", "alpha")).unwrap();
        f.mgr.connect("s1").await.unwrap();
        f.client.fail_disconnect("socket closed");

        let config = f.mgr.remove("s1").await.unwrap();

        assert_eq!(config.server_name, "alpha");
        assert!(f.mgr.is_empty());
    }

    #[tokio::test]
    async fn missing_ids_are_not_found() {
        let f = fixture();

        for error in [
            f.mgr.connect("missing").await.unwrap_err(),
            f.mgr.load_tools("missing").await.unwrap_err(),
            f.mgr.disconnect("missing").await.unwrap_err(),
            f.mgr.remove("missing").await.unwrap_err(),
        ] {
            assert_eq!(error.kind(), ErrorKind::NotFound);
            assert_eq!(
                error.to_string(),
                "not_found: mcp server 'missing' is not registered"
            );
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn concurrent_connects_open_exactly_one_session() {
        let f = fixture();
        f.mgr.register(config("s1", "alpha")).unwrap();

        let (first, second) = tokio::join!(f.mgr.connect("s1"), f.mgr.connect("s1"));
        first.unwrap();
        second.unwrap();

        assert_eq!(f.client.connect_count(), 1);
        assert_eq!(f.connector.builds(), 1);
        assert_eq!(f.mgr.state("s1"), Some(McpServerState::Connected));
        assert!(f.mgr.client("s1").is_some());
    }

    #[tokio::test]
    async fn revision_tracks_mutations_and_clear_reports_the_count() {
        let f = fixture();
        f.mgr.register(config("s1", "alpha")).unwrap();
        f.mgr.register(config("s2", "beta")).unwrap();
        assert_eq!(f.mgr.revision(), 2);

        assert_eq!(f.mgr.clear(), 2);
        assert!(f.mgr.is_empty());
        assert_eq!(f.mgr.revision(), 3);

        assert_eq!(f.mgr.clear(), 0);
        assert_eq!(f.mgr.revision(), 3, "clearing an empty registry is a no-op");
    }

    #[tokio::test]
    async fn reset_clears_and_zeroes_the_revision() {
        let f = fixture();
        f.mgr.register(config("s1", "alpha")).unwrap();
        assert_eq!(f.mgr.reset(), 1);
        assert!(f.mgr.is_empty());
        assert_eq!(f.mgr.revision(), 0);
    }

    #[tokio::test]
    async fn debug_output_lists_ids_and_the_revision() {
        let f = fixture();
        f.mgr.register(config("s1", "alpha")).unwrap();
        let rendered = format!("{:?}", f.mgr);
        assert!(rendered.contains("\"s1\""), "{rendered}");
        assert!(rendered.contains("revision: 1"), "{rendered}");
    }
}
