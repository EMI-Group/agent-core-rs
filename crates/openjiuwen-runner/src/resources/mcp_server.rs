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
    /// Explicitly disconnected by [`McpServerMgr::disconnect`]; removal drops
    /// the entry instead of reaching this state.
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
    /// non-empty clear, and every recorded state change (see
    /// [`McpServerMgr::revision`]).
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
    /// and every state change it records: a successful connect, a failed
    /// connect, and a disconnect. A repeated failure that leaves the entry
    /// `Failed` with no client is not counted, though the stored reason is
    /// refreshed.
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
#[path = "mcp_server/tests.rs"]
mod tests;
