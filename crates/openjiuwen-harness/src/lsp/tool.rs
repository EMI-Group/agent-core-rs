//! The model-facing `lsp` tool.
//!
//! Ports `openjiuwen/harness/tools/lsp_tool/_tool.py` as a Rust [`Tool`]: lazily started
//! language servers, read-only navigation actions, pull and push diagnostics, and a
//! `success=false` payload for every failure. The tool never returns `Err` and never panics
//! on model input.
//!
//! # Actions
//!
//! `hover`, `definition`, `references`, `completion`, `symbols` (document symbols),
//! `workspace_symbols`, `diagnostics`, `implementation`, `prepare_call_hierarchy`,
//! `incoming_calls` and `outgoing_calls`.
//!
//! # Server selection
//!
//! [`LspToolConfig`] resolves the server for one call through three layers, in order:
//!
//! 1. An explicit [`LspToolConfig::server`] always wins (single-server mode). Its key name is
//!    [`LspServerConfig::name`] or `"default"`, and its root is
//!    [`LspServerConfig::root`] or the canonical workspace root.
//! 2. Otherwise [`LspToolConfig::servers`] is indexed by
//!    [`DiscoveryIndex`](super::DiscoveryIndex) and the file's extension, priority and root
//!    markers select the definition.
//! 3. Otherwise, when a transport or transport factory was injected (the offline seam), a
//!    default definition using [`LspToolConfig::language_id`] is synthesized. Without any of
//!    the three the call fails with a typed "no LSP server configured" payload.
//!
//! `workspace_symbols` has no file to inspect, so it uses the explicit server, else
//! [`DiscoveryIndex::resolve_by_language`](super::DiscoveryIndex::resolve_by_language) with
//! [`LspToolConfig::language_id`], else the first discovered definition at the workspace root,
//! else the injected/default path.
//!
//! # Lifecycle
//!
//! One [`ServerSupervisor`] is created lazily per [`ServerKey`](super::ServerKey) and reused
//! for every later call, so a crashed server is restarted (once, with back-off) and the
//! operation retried instead of leaking a process. [`LspTool::shutdown`] stops every server
//! and makes later calls fail; [`LspTool::server_status`] exposes the observed lifecycle.
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::OnceCell;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::tool::{Tool, ToolCard, ToolContext};
use openjiuwen_sysop::workspace::canonicalize_containing;

use crate::tools::{schema_for, tool_card};

use super::config::{LspServerConfig, DEFAULT_MAX_FILE_SIZE, DEFAULT_REQUEST_TIMEOUT};
use super::diagnostics::{DiagnosticRegistry, DiagnosticsConfig};
use super::discovery::{
    builtin_server_configs, server_name, DiscoveryIndex, ResolvedServer, ServerKey,
};
use super::instance::{
    BackoffTimer, RestartPolicy, ServerFactory, ServerStatus, ServerSupervisor, TokioTimer,
};
use super::transport::{LspTransport, StdioTransport};
use super::wire::Diagnostic;

/// The typed message returned when no layer resolves a language server.
pub(super) const NO_SERVER_CONFIGURED: &str =
    "no LSP server configured for the lsp tool; set LspToolConfig::server or inject a transport";

/// The navigation action requested from the language server.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum LspAction {
    /// Type/signature information at a position.
    Hover,
    /// Go to definition at a position.
    Definition,
    /// Find all references at a position.
    References,
    /// Completion candidates at a position.
    Completion,
    /// Document symbols for the whole file.
    Symbols,
    /// Workspace-wide symbol search with `query`.
    WorkspaceSymbols,
    /// Diagnostics for one file (push, then pull) or, without a path, every buffered file.
    Diagnostics,
    /// Go to the implementation(s) of the symbol at a position.
    Implementation,
    /// Prepare the call-hierarchy items at a position.
    PrepareCallHierarchy,
    /// Incoming calls of the symbol at a position.
    IncomingCalls,
    /// Outgoing calls of the symbol at a position.
    OutgoingCalls,
}

impl LspAction {
    /// Every action name, in schema order.
    pub const ALL: [Self; 11] = [
        Self::Hover,
        Self::Definition,
        Self::References,
        Self::Completion,
        Self::Symbols,
        Self::WorkspaceSymbols,
        Self::Diagnostics,
        Self::Implementation,
        Self::PrepareCallHierarchy,
        Self::IncomingCalls,
        Self::OutgoingCalls,
    ];

    /// The `snake_case` name used in JSON.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Hover => "hover",
            Self::Definition => "definition",
            Self::References => "references",
            Self::Completion => "completion",
            Self::Symbols => "symbols",
            Self::WorkspaceSymbols => "workspace_symbols",
            Self::Diagnostics => "diagnostics",
            Self::Implementation => "implementation",
            Self::PrepareCallHierarchy => "prepare_call_hierarchy",
            Self::IncomingCalls => "incoming_calls",
            Self::OutgoingCalls => "outgoing_calls",
        }
    }

    /// Whether the action needs a `path` argument.
    #[must_use]
    pub const fn requires_path(self) -> bool {
        !matches!(self, Self::WorkspaceSymbols | Self::Diagnostics)
    }

    /// The LSP method whose capability/error name describes the action.
    pub(super) const fn method(self) -> &'static str {
        match self {
            Self::Hover => "textDocument/hover",
            Self::Definition => "textDocument/definition",
            Self::References => "textDocument/references",
            Self::Completion => "textDocument/completion",
            Self::Symbols => "textDocument/documentSymbol",
            Self::WorkspaceSymbols => "workspace/symbol",
            Self::Diagnostics => "textDocument/diagnostic",
            Self::Implementation => "textDocument/implementation",
            Self::PrepareCallHierarchy => "textDocument/prepareCallHierarchy",
            Self::IncomingCalls => "callHierarchy/incomingCalls",
            Self::OutgoingCalls => "callHierarchy/outgoingCalls",
        }
    }
}

/// Arguments of the `lsp` tool.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct LspArgs {
    /// The navigation action to perform.
    pub action: LspAction,
    /// File path, absolute or relative to the workspace root.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "String")]
    pub path: Option<String>,
    /// One-based line number; LSP positions are zero-based internally.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "u32")]
    pub line: Option<u32>,
    /// One-based column number; defaults to 1.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "u32")]
    pub column: Option<u32>,
    /// Symbol query, used by `workspace_symbols`; empty returns all symbols.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "String")]
    pub query: Option<String>,
    /// Source lines shown around definition results (0-20).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "u32")]
    pub context_lines: Option<u32>,
}

/// Configuration of one [`LspTool`].
///
/// Server selection follows [`LspToolConfig::server`], then [`LspToolConfig::servers`], then
/// the injected transport; see the module docs for the exact precedence.
#[derive(Debug, Clone)]
pub struct LspToolConfig {
    /// Root that relative paths resolve against; defaults to the current directory.
    pub workspace_root: Option<PathBuf>,
    /// Description language; `"en"` selects English, anything else the Python default `"cn"`.
    pub language: String,
    /// The one server used in single-server mode; it always wins over [`LspToolConfig::servers`].
    pub server: Option<LspServerConfig>,
    /// Discovery definitions consulted per file when [`LspToolConfig::server`] is `None`.
    ///
    /// Populate it with [`LspToolConfig::with_servers`],
    /// [`LspToolConfig::with_discovered_server`] or [`LspToolConfig::with_builtin_servers`].
    pub servers: Vec<LspServerConfig>,
    /// Crash-recovery budget and back-off shape for every started server.
    pub restart_policy: RestartPolicy,
    /// `languageId` used for `didOpen` when no server config supplies one.
    pub language_id: String,
    /// Per-request timeout.
    pub request_timeout: Duration,
    /// Largest file the tool will synchronize, in bytes.
    pub max_file_size: u64,
    /// Default number of source lines shown around definition results.
    pub context_lines: u32,
}

impl Default for LspToolConfig {
    fn default() -> Self {
        Self {
            workspace_root: None,
            language: "cn".to_owned(),
            server: None,
            servers: Vec::new(),
            restart_policy: RestartPolicy::default(),
            language_id: "plaintext".to_owned(),
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
            max_file_size: DEFAULT_MAX_FILE_SIZE,
            context_lines: 0,
        }
    }
}

impl LspToolConfig {
    /// Returns this configuration with a different workspace root.
    #[must_use]
    pub fn with_workspace_root(mut self, root: impl Into<PathBuf>) -> Self {
        self.workspace_root = Some(root.into());
        self
    }

    /// Returns this configuration with a different description language.
    #[must_use]
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = language.into();
        self
    }

    /// Returns this configuration with an explicit server, enabling single-server mode.
    #[must_use]
    pub fn with_server(mut self, server: LspServerConfig) -> Self {
        self.server = Some(server);
        self
    }

    /// Returns this configuration with `servers` as its discovery definitions.
    ///
    /// The list replaces any previous one; order breaks priority ties, as in
    /// [`DiscoveryIndex`](super::DiscoveryIndex).
    #[must_use]
    pub fn with_servers<I>(mut self, servers: I) -> Self
    where
        I: IntoIterator<Item = LspServerConfig>,
    {
        self.servers = servers.into_iter().collect();
        self
    }

    /// Returns this configuration with `server` appended to its discovery definitions.
    #[must_use]
    pub fn with_discovered_server(mut self, server: LspServerConfig) -> Self {
        self.servers.push(server);
        self
    }

    /// Returns this configuration with every built-in definition appended.
    ///
    /// The definitions mirror Python's `servers/servers/*.py`; their commands are not probed,
    /// so a missing binary surfaces as a typed launch failure on first use.
    #[must_use]
    pub fn with_builtin_servers(mut self) -> Self {
        self.servers.extend(builtin_server_configs());
        self
    }

    /// Returns this configuration with a different crash-recovery policy.
    #[must_use]
    pub const fn with_restart_policy(mut self, policy: RestartPolicy) -> Self {
        self.restart_policy = policy;
        self
    }

    /// Returns this configuration with a different fallback `languageId`.
    #[must_use]
    pub fn with_language_id(mut self, language_id: impl Into<String>) -> Self {
        self.language_id = language_id.into();
        self
    }

    /// Returns this configuration with a different per-request timeout.
    #[must_use]
    pub fn with_request_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = timeout;
        self
    }

    /// Returns this configuration with a different maximum synchronized file size.
    #[must_use]
    pub fn with_max_file_size(mut self, bytes: u64) -> Self {
        self.max_file_size = bytes;
        self
    }

    /// Returns this configuration with a different default context-line count.
    #[must_use]
    pub fn with_context_lines(mut self, lines: u32) -> Self {
        self.context_lines = lines;
        self
    }
}

/// The canonical workspace root plus the discovery table built from it.
///
/// Both are computed once per tool, so a later `LspToolConfig` mutation cannot desynchronize
/// path containment from root inference.
pub(super) struct WorkspaceIndex {
    /// Canonical workspace root every resolved path must stay inside.
    pub(super) root: PathBuf,
    /// Extension/priority/root-marker table over the configured definitions.
    discovery: DiscoveryIndex,
}

/// The `lsp` tool: code navigation, diagnostics and call hierarchy through language servers.
pub struct LspTool {
    /// The model-facing card.
    pub(super) card: ToolCard,
    /// Path resolution, language, timeouts, server selection and restart policy.
    pub(super) config: LspToolConfig,
    /// An injected transport factory (the offline seam); `None` launches the resolved server.
    factory: parking_lot::Mutex<Option<ServerFactory>>,
    /// Push-diagnostic buffer shared with every client and with the host.
    pub(super) diagnostics: Arc<DiagnosticRegistry>,
    /// Back-off clock handed to every supervisor.
    timer: Arc<dyn BackoffTimer>,
    /// One supervisor per server instance, created lazily and reused.
    supervisors: parking_lot::Mutex<BTreeMap<ServerKey, Arc<ServerSupervisor>>>,
    /// Last pull-report result id per document URI.
    pub(super) pull_result_ids: parking_lot::Mutex<HashMap<String, String>>,
    /// Last full pull report's diagnostics per document URI, for `unchanged` answers.
    pub(super) pull_items: parking_lot::Mutex<HashMap<String, Vec<Diagnostic>>>,
    /// Canonical workspace root and discovery table, resolved on first use.
    index: OnceCell<WorkspaceIndex>,
    /// Set by [`LspTool::shutdown`]; forbids any later server start.
    shutdown: AtomicBool,
}

impl LspTool {
    /// Builds the tool from `config`; no server is started until the first call.
    #[must_use]
    pub fn new(config: LspToolConfig) -> Self {
        let card = Self::card_for(&config.language);
        Self {
            card,
            config,
            factory: parking_lot::Mutex::new(None),
            diagnostics: Arc::new(DiagnosticRegistry::new(DiagnosticsConfig::default())),
            timer: Arc::new(TokioTimer),
            supervisors: parking_lot::Mutex::new(BTreeMap::new()),
            pull_result_ids: parking_lot::Mutex::new(HashMap::new()),
            pull_items: parking_lot::Mutex::new(HashMap::new()),
            index: OnceCell::new(),
            shutdown: AtomicBool::new(false),
        }
    }

    /// Injects a transport instead of launching the configured server.
    ///
    /// This is the offline test seam: pass a [`ChannelTransport`](super::ChannelTransport)
    /// whose peer stream is driven by an in-process fake server. The transport is used by the
    /// first server started; anything after that falls back to launching the resolved config.
    #[must_use]
    pub fn with_transport(self, transport: Box<dyn LspTransport>) -> Self {
        let slot = Arc::new(parking_lot::Mutex::new(Some(transport)));
        let factory: ServerFactory = Arc::new(move |config: &LspServerConfig| {
            if let Some(transport) = slot.lock().take() {
                return Ok(transport);
            }
            Ok(Box::new(StdioTransport::spawn(config)?))
        });
        *self.factory.lock() = Some(factory);
        self
    }

    /// Replaces the transport factory used to launch every server.
    ///
    /// Hosts use it to route servers through their own process launcher; tests use it to hand
    /// out a fresh in-process fake per [`LspServerConfig`].
    #[must_use]
    pub fn with_transport_factory(self, factory: ServerFactory) -> Self {
        *self.factory.lock() = Some(factory);
        self
    }

    /// Shares `registry` for push diagnostics; by default the tool owns a private registry.
    #[must_use]
    pub fn with_diagnostics(mut self, registry: Arc<DiagnosticRegistry>) -> Self {
        self.diagnostics = registry;
        self
    }

    /// Replaces the back-off clock used between crash-recovery attempts.
    #[must_use]
    pub fn with_backoff_timer(mut self, timer: Arc<dyn BackoffTimer>) -> Self {
        self.timer = timer;
        self
    }

    /// Builds the model-facing card in `language`.
    #[must_use]
    pub fn card_for(language: &str) -> ToolCard {
        let mut input_params = schema_for::<LspArgs>();
        install_action_enum(&mut input_params);
        tool_card("lsp", description(language), input_params, true)
    }

    /// The configuration this tool was built with.
    #[must_use]
    pub fn config(&self) -> &LspToolConfig {
        &self.config
    }

    /// The push-diagnostic registry this tool feeds and reads.
    #[must_use]
    pub fn diagnostics(&self) -> Arc<DiagnosticRegistry> {
        Arc::clone(&self.diagnostics)
    }

    /// A snapshot of every supervised server, ordered by [`ServerKey`](super::ServerKey).
    #[must_use]
    pub fn server_status(&self) -> Vec<ServerStatus> {
        self.supervisors
            .lock()
            .values()
            .map(|supervisor| supervisor.status())
            .collect()
    }

    /// Stops every server, idempotently, and makes later calls fail.
    ///
    /// A host should await this before dropping a tool that launched real processes; dropping
    /// the clients would otherwise only close their transports.
    pub async fn shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
        let supervisors: Vec<Arc<ServerSupervisor>> =
            self.supervisors.lock().values().cloned().collect();
        for supervisor in supervisors {
            supervisor.shutdown().await;
        }
    }

    /// Resolves `path` (or a language-only request) to the server that should handle it.
    ///
    /// See the module docs for the precedence between the explicit server, the discovery
    /// definitions and the injected transport.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when no layer resolves a server, and the workspace-root error
    /// when the root cannot be determined.
    pub(super) async fn resolve_server(&self, path: Option<&Path>) -> Result<ResolvedServer> {
        let index = self.workspace_index().await.map_err(Error::config)?;
        if let Some(server) = &self.config.server {
            return Ok(explicit_server(server, &index.root));
        }
        let discovered = match path {
            Some(path) => index.discovery.resolve(path),
            None => {
                workspace_symbols_server(&index.discovery, &index.root, &self.config.language_id)
            }
        };
        if let Some(resolved) = discovered {
            return Ok(resolved);
        }
        if self.has_factory() {
            return Ok(default_server(&self.config.language_id, &index.root));
        }
        Err(Error::config(NO_SERVER_CONFIGURED))
    }

    /// Returns the supervisor for `resolved`, creating it lazily and reusing it afterwards.
    ///
    /// # Errors
    ///
    /// Returns the typed "has been shut down" error after [`LspTool::shutdown`].
    pub(super) fn supervisor(&self, resolved: &ResolvedServer) -> Result<Arc<ServerSupervisor>> {
        if self.shutdown.load(Ordering::SeqCst) {
            return Err(Error::transport("LSP tool has been shut down"));
        }
        if let Some(existing) = self.supervisors.lock().get(&resolved.key) {
            return Ok(Arc::clone(existing));
        }
        let supervisor = Arc::new(
            ServerSupervisor::new(resolved.config.clone(), self.server_factory())
                .with_diagnostics(Arc::clone(&self.diagnostics))
                .with_timer(Arc::clone(&self.timer))
                .with_restart_policy(self.config.restart_policy)
                .with_request_timeout(self.config.request_timeout)
                .with_workspace_root(resolved.key.root.clone()),
        );
        let mut supervisors = self.supervisors.lock();
        let entry = supervisors
            .entry(resolved.key.clone())
            .or_insert_with(|| Arc::clone(&supervisor));
        Ok(Arc::clone(entry))
    }

    /// The factory that launches one server: the injected one, else a stdio process.
    fn server_factory(&self) -> ServerFactory {
        if let Some(factory) = self.factory.lock().as_ref() {
            return Arc::clone(factory);
        }
        Arc::new(|config: &LspServerConfig| {
            Ok(Box::new(StdioTransport::spawn(config)?) as Box<dyn LspTransport>)
        })
    }

    /// Whether a transport or transport factory was injected.
    fn has_factory(&self) -> bool {
        self.factory.lock().is_some()
    }

    /// Whether any layer (explicit, discovered or injected) can resolve a server.
    pub(super) fn has_server_configuration(&self) -> bool {
        self.config.server.is_some() || !self.config.servers.is_empty() || self.has_factory()
    }

    /// The `languageId` sent in `didOpen` for `resolved`.
    pub(super) fn language_for(&self, resolved: &ResolvedServer) -> String {
        let language_id = &resolved.config.language_id;
        if language_id.is_empty() {
            self.config.language_id.clone()
        } else {
            language_id.clone()
        }
    }

    /// The canonical workspace root and discovery table, computed once.
    pub(super) async fn workspace_index(&self) -> std::result::Result<&WorkspaceIndex, String> {
        self.index
            .get_or_try_init(|| async {
                let root = match &self.config.workspace_root {
                    Some(root) => root.clone(),
                    None => std::env::current_dir()
                        .map_err(|error| format!("cannot determine the workspace root: {error}"))?,
                };
                let root = canonicalize_containing(&root).await;
                let discovery = DiscoveryIndex::new(root.clone(), self.config.servers.clone());
                Ok(WorkspaceIndex { root, discovery })
            })
            .await
    }
}

/// The explicit single-server mode: its config always wins over the discovery definitions.
fn explicit_server(server: &LspServerConfig, workspace_root: &Path) -> ResolvedServer {
    let root = server
        .root
        .clone()
        .unwrap_or_else(|| workspace_root.to_path_buf());
    let mut config = server.clone();
    config.root = Some(root.clone());
    let name = if server.name.is_empty() {
        "default".to_owned()
    } else {
        server.name.clone()
    };
    ResolvedServer {
        key: ServerKey { name, root },
        config,
    }
}

/// The synthesized definition used when only an injected transport is available.
fn default_server(language_id: &str, workspace_root: &Path) -> ResolvedServer {
    let name = if language_id.is_empty() {
        "default".to_owned()
    } else {
        language_id.to_owned()
    };
    let root = workspace_root.to_path_buf();
    let mut config = LspServerConfig::new(String::new())
        .with_name(name.clone())
        .with_language_id(language_id);
    config.root = Some(root.clone());
    ResolvedServer {
        key: ServerKey { name, root },
        config,
    }
}

/// Resolves a language-only (`workspace_symbols`) request without a file path.
fn workspace_symbols_server(
    index: &DiscoveryIndex,
    workspace_root: &Path,
    language_id: &str,
) -> Option<ResolvedServer> {
    if let Some(resolved) = index.resolve_by_language(language_id) {
        return Some(resolved);
    }
    let mut config = index.servers().first()?.clone();
    let root = workspace_root.to_path_buf();
    config.root = Some(root.clone());
    let name = server_name(&config).to_owned();
    Some(ResolvedServer {
        key: ServerKey { name, root },
        config,
    })
}

#[async_trait]
impl Tool for LspTool {
    fn card(&self) -> &ToolCard {
        &self.card
    }

    async fn invoke(&self, inputs: Value, ctx: &ToolContext) -> Result<Value> {
        let payload = self.run(inputs, ctx).await;
        Ok(payload.to_json())
    }

    /// Renders the short result text rather than the JSON envelope.
    fn render_for_llm(&self, output: &Value) -> String {
        let success = output
            .get("success")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        if success {
            if let Some(result) = output
                .get("data")
                .and_then(|data| data.get("result"))
                .and_then(Value::as_str)
            {
                return if result.trim().is_empty() {
                    "No results found.".to_owned()
                } else {
                    result.to_owned()
                };
            }
        }
        output
            .get("error")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned()
    }
}

/// Adds the explicit action enum to the generated schema.
///
/// The generated schema references the Rust enum through `$defs`, which
/// [`schema_for`] strips for model-facing cards; installing the names here keeps `action`
/// self-describing.
fn install_action_enum(schema: &mut Value) {
    let actions: Vec<&str> = LspAction::ALL
        .iter()
        .map(|action| action.as_str())
        .collect();
    if let Some(properties) = schema.get_mut("properties").and_then(Value::as_object_mut) {
        properties.insert(
            "action".to_owned(),
            json!({
                "type": "string",
                "enum": actions,
                "description": "The navigation action to perform.",
            }),
        );
    }
}

/// Maps a client error to the model-facing message, naming unsupported methods clearly.
pub(super) fn map_client_error(error: &Error, method: &str) -> String {
    let message = error.message().into_owned();
    if message.contains("-32601") || message.to_ascii_lowercase().contains("method not found") {
        format!("Operation not supported by this language server: {method}")
    } else {
        format!("LSP request failed: {message}")
    }
}

/// The tool description for `language`; only `"en"` selects English.
fn description(language: &str) -> &'static str {
    if language == "en" {
        "LSP (Language Server Protocol) code navigation: hover, definition, references, \
         completion, document symbols, workspace symbols, implementation, call hierarchy \
         (prepare, incoming and outgoing calls) and diagnostics (push and pull). Positions are \
         one-based line and column. Returns short, typed text results."
    } else {
        "LSP（语言服务器协议）代码导航工具：hover、definition、references、completion、\
         symbols（文档符号）、workspace_symbols、implementation、call hierarchy（prepare_call_hierarchy、\
         incoming_calls、outgoing_calls）和 diagnostics（推送与拉取诊断）。行号和列号从 1 开始，\
         返回简短的结构化文本结果。"
    }
}
