//! The model-facing `lsp` tool.
//!
//! Ports `openjiuwen/harness/tools/lsp_tool/_tool.py` as a Rust [`Tool`]: one lazily started
//! server per tool instance (one workspace root), read-only navigation actions, and a
//! `success=false` payload for every failure. The tool never returns `Err` and never panics
//! on model input.
//!
//! Actions: `hover`, `definition`, `references`, `completion`, `symbols` (document symbols)
//! and `workspace_symbols`; `diagnostics` is accepted by the schema so the model gets a
//! typed "not implemented" answer instead of a parse error.
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::OnceCell;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::tool::{Tool, ToolCard, ToolContext};
use openjiuwen_sysop::workspace::{canonicalize_containing, is_within};

use crate::tools::{schema_for, tool_card, ToolPayload};

use super::client::LspClient;
use super::config::{
    LspServerConfig, DEFAULT_MAX_FILE_SIZE, DEFAULT_REQUEST_TIMEOUT, DEFAULT_STARTUP_TIMEOUT,
};
use super::protocol::{client_capabilities, Location, Position};
use super::render;
use super::transport::{LspTransport, StdioTransport};
use super::uri::{path_to_uri, uri_to_path};

/// Largest `context_lines` value honoured, keeping excerpts short.
const MAX_CONTEXT_LINES: u32 = 20;
/// Largest number of definition locations that get a source excerpt.
const MAX_CONTEXT_LOCATIONS: usize = 5;

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
    /// Diagnostics for the file (accepted but not implemented in this build).
    Diagnostics,
}

impl LspAction {
    /// Every action name, in schema order.
    pub const ALL: [Self; 7] = [
        Self::Hover,
        Self::Definition,
        Self::References,
        Self::Completion,
        Self::Symbols,
        Self::WorkspaceSymbols,
        Self::Diagnostics,
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
        }
    }

    /// Whether the action needs a `path` argument.
    #[must_use]
    pub const fn requires_path(self) -> bool {
        !matches!(self, Self::WorkspaceSymbols | Self::Diagnostics)
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
#[derive(Debug, Clone)]
pub struct LspToolConfig {
    /// Root that relative paths resolve against; defaults to the current directory.
    pub workspace_root: Option<PathBuf>,
    /// Description language; `"en"` selects English, anything else the Python default `"cn"`.
    pub language: String,
    /// The server launched lazily on the first call; may be `None` when a transport is injected.
    pub server: Option<LspServerConfig>,
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

    /// Returns this configuration with a language server to launch lazily.
    #[must_use]
    pub fn with_server(mut self, server: LspServerConfig) -> Self {
        self.server = Some(server);
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

/// The `lsp` tool: code navigation through a lazily started language server.
pub struct LspTool {
    /// The model-facing card.
    card: ToolCard,
    /// Path resolution, language, timeouts and the optional server descriptor.
    config: LspToolConfig,
    /// An injected transport, consumed when the client is first built.
    transport: parking_lot::Mutex<Option<Box<dyn LspTransport>>>,
    /// The lazily initialized client; initialization failures are retried on the next call.
    client: OnceCell<Arc<LspClient>>,
}

impl LspTool {
    /// Builds the tool from `config`; no server is started until the first call.
    #[must_use]
    pub fn new(config: LspToolConfig) -> Self {
        let card = Self::card_for(&config.language);
        Self {
            card,
            config,
            transport: parking_lot::Mutex::new(None),
            client: OnceCell::new(),
        }
    }

    /// Injects a transport instead of launching the configured server.
    ///
    /// This is the offline test seam: pass a [`ChannelTransport`](super::ChannelTransport)
    /// whose peer stream is driven by an in-process fake server.
    #[must_use]
    pub fn with_transport(self, transport: Box<dyn LspTransport>) -> Self {
        *self.transport.lock() = Some(transport);
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

    /// Returns the lazily built client, initializing the server on the first call.
    async fn client(&self) -> Result<Arc<LspClient>> {
        let client = self.client.get_or_try_init(|| self.build_client()).await?;
        Ok(Arc::clone(client))
    }

    /// Builds and initializes the client from the injected transport or the server config.
    async fn build_client(&self) -> Result<Arc<LspClient>> {
        let transport: Box<dyn LspTransport> = if let Some(transport) = self.transport.lock().take()
        {
            transport
        } else if let Some(server) = &self.config.server {
            Box::new(StdioTransport::spawn(server)?)
        } else {
            return Err(Error::config(
                "no LSP server configured for the lsp tool; set LspToolConfig::server or inject a transport",
            ));
        };
        let client = Arc::new(LspClient::new(transport, self.config.request_timeout));
        let root_uri = path_to_uri(&self.server_root());
        let startup_timeout = self
            .config
            .server
            .as_ref()
            .map_or(DEFAULT_STARTUP_TIMEOUT, |server| server.startup_timeout);
        match tokio::time::timeout(
            startup_timeout,
            client.initialize(&root_uri, client_capabilities()),
        )
        .await
        {
            Ok(result) => {
                result?;
                Ok(client)
            }
            Err(_) => Err(Error::timeout(format!(
                "LSP server did not complete initialize within {} ms",
                startup_timeout.as_millis()
            ))),
        }
    }

    /// The root URI sent in `initialize`.
    fn server_root(&self) -> PathBuf {
        self.config
            .server
            .as_ref()
            .and_then(|server| server.root.clone())
            .or_else(|| self.config.workspace_root.clone())
            .or_else(|| std::env::current_dir().ok())
            .unwrap_or_else(|| PathBuf::from("."))
    }

    /// The `languageId` sent in `didOpen`.
    fn language_id(&self) -> &str {
        self.config
            .server
            .as_ref()
            .map_or(self.config.language_id.as_str(), |server| {
                server.language_id.as_str()
            })
    }

    /// Validates the arguments and executes the requested action.
    #[allow(clippy::too_many_lines)] // one dispatch table keeps the payload shapes together
    async fn execute(&self, args: LspArgs) -> std::result::Result<ToolPayload, String> {
        let action = args.action;
        if action == LspAction::Diagnostics {
            return Err(
                "Unsupported action 'diagnostics': diagnostics are deferred in this build."
                    .to_owned(),
            );
        }
        let client = self
            .client()
            .await
            .map_err(|error| error.message().into_owned())?;
        if action == LspAction::WorkspaceSymbols {
            let query = args.query.as_deref().unwrap_or_default();
            let symbols = client
                .workspace_symbols(query)
                .await
                .map_err(|error| map_client_error(&error, "workspace/symbol"))?;
            let result = render::render_workspace_symbols(&symbols);
            return Ok(ToolPayload::ok(json!({
                "action": action.as_str(),
                "result": result,
                "symbols": symbols,
            })));
        }
        let (path, uri, source) = self.resolve_document(&args).await?;
        client
            .did_open(&uri, self.language_id(), &source)
            .await
            .map_err(|error| map_client_error(&error, "textDocument/didOpen"))?;
        let position = Position::from_one_based(args.line.unwrap_or(1), args.column.unwrap_or(1))
            .map_err(|error| error.message().into_owned())?;
        let file_path = path.to_string_lossy().into_owned();
        match action {
            LspAction::Hover => {
                let hover = client
                    .hover(&uri, position)
                    .await
                    .map_err(|error| map_client_error(&error, "textDocument/hover"))?;
                let result = render::render_hover(hover.as_ref());
                Ok(ToolPayload::ok(json!({
                    "action": "hover",
                    "file_path": file_path,
                    "result": result,
                    "hover": hover,
                })))
            }
            LspAction::Definition => {
                let locations = client
                    .definition(&uri, position)
                    .await
                    .map_err(|error| map_client_error(&error, "textDocument/definition"))?;
                let mut result = render::render_definition(&locations);
                let context = args
                    .context_lines
                    .unwrap_or(self.config.context_lines)
                    .min(MAX_CONTEXT_LINES);
                if context > 0 {
                    let excerpt = self.context_excerpt(&locations, context).await;
                    if !excerpt.is_empty() {
                        result.push_str("\n\n");
                        result.push_str(&excerpt);
                    }
                }
                Ok(ToolPayload::ok(json!({
                    "action": "definition",
                    "file_path": file_path,
                    "result": result,
                    "locations": locations,
                })))
            }
            LspAction::References => {
                let locations = client
                    .references(&uri, position, true)
                    .await
                    .map_err(|error| map_client_error(&error, "textDocument/references"))?;
                let result = render::render_references(&locations);
                Ok(ToolPayload::ok(json!({
                    "action": "references",
                    "file_path": file_path,
                    "result": result,
                    "locations": locations,
                })))
            }
            LspAction::Completion => {
                let items = client
                    .completion(&uri, position)
                    .await
                    .map_err(|error| map_client_error(&error, "textDocument/completion"))?;
                let result = render::render_completion(&items);
                Ok(ToolPayload::ok(json!({
                    "action": "completion",
                    "file_path": file_path,
                    "result": result,
                    "items": items,
                })))
            }
            LspAction::Symbols => {
                let symbols = client
                    .document_symbols(&uri)
                    .await
                    .map_err(|error| map_client_error(&error, "textDocument/documentSymbol"))?;
                let result = render::render_symbols(&symbols);
                Ok(ToolPayload::ok(json!({
                    "action": "symbols",
                    "file_path": file_path,
                    "result": result,
                    "symbols": symbols,
                })))
            }
            LspAction::WorkspaceSymbols | LspAction::Diagnostics => {
                Err(format!("Unsupported action '{}'.", action.as_str()))
            }
        }
    }

    /// Resolves the `path` argument and reads the document for synchronization.
    async fn resolve_document(
        &self,
        args: &LspArgs,
    ) -> std::result::Result<(PathBuf, String, String), String> {
        let raw = args
            .path
            .as_deref()
            .filter(|path| !path.is_empty())
            .ok_or_else(|| format!("path is required for '{}'", args.action.as_str()))?;
        let path = self.resolve_path(raw).await?;
        let source = self.read_source(&path).await?;
        let uri = path_to_uri(&path);
        Ok((path, uri, source))
    }

    /// Resolves a model-supplied path inside the workspace root.
    async fn resolve_path(&self, raw: &str) -> std::result::Result<PathBuf, String> {
        let root = match &self.config.workspace_root {
            Some(root) => root.clone(),
            None => std::env::current_dir()
                .map_err(|error| format!("cannot determine the workspace root: {error}"))?,
        };
        let root = canonicalize_containing(&root).await;
        let path = Path::new(raw);
        let candidate = if path.is_absolute() {
            path.to_path_buf()
        } else {
            root.join(path)
        };
        let resolved = canonicalize_containing(&candidate).await;
        if !is_within(&resolved, &root) {
            return Err(format!("Path '{raw}' is outside the workspace root."));
        }
        let metadata = tokio::fs::metadata(&resolved)
            .await
            .map_err(|_| format!("File not found: '{raw}'"))?;
        if !metadata.is_file() {
            return Err(format!("Not a file: '{raw}'"));
        }
        Ok(resolved)
    }

    /// Reads a document, enforcing the configured size limit.
    async fn read_source(&self, path: &Path) -> std::result::Result<String, String> {
        let metadata = tokio::fs::metadata(path)
            .await
            .map_err(|error| format!("Cannot stat '{}': {error}", path.display()))?;
        if metadata.len() > self.config.max_file_size {
            return Err(format!(
                "File '{}' is too large for LSP operations ({} bytes; max {}).",
                path.display(),
                metadata.len(),
                self.config.max_file_size
            ));
        }
        tokio::fs::read_to_string(path)
            .await
            .map_err(|error| format!("Cannot read '{}': {error}", path.display()))
    }

    /// Renders numbered source excerpts around the first few definition locations.
    async fn context_excerpt(&self, locations: &[Location], context: u32) -> String {
        let mut blocks = Vec::new();
        for location in locations.iter().take(MAX_CONTEXT_LOCATIONS) {
            let path = uri_to_path(&location.uri);
            let Ok(text) = tokio::fs::read_to_string(&path).await else {
                continue;
            };
            let lines: Vec<&str> = text.lines().collect();
            if lines.is_empty() {
                continue;
            }
            let target = usize::try_from(location.range.start.line).unwrap_or(0);
            let context = usize::try_from(context).unwrap_or(0);
            let first = target.saturating_sub(context);
            let last = target
                .saturating_add(context)
                .min(lines.len().saturating_sub(1));
            if first > last {
                continue;
            }
            let first_line = u32::try_from(first).unwrap_or(u32::MAX - 1) + 1;
            blocks.push(render::format_context_excerpt(
                &path.to_string_lossy(),
                first_line,
                &lines[first..=last],
            ));
        }
        blocks.join("\n")
    }
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

impl LspTool {
    /// Runs one tool call, converting every failure into a `success=false` payload.
    async fn run(&self, inputs: Value, ctx: &ToolContext) -> ToolPayload {
        let args: LspArgs = match serde_json::from_value(inputs) {
            Ok(args) => args,
            Err(error) => return ToolPayload::fail(format!("Invalid input: {error}")),
        };
        if ctx.cancellation.is_cancelled() {
            return ToolPayload::fail("LSP tool call was cancelled.");
        }
        match self.execute(args).await {
            Ok(payload) => payload,
            Err(message) => ToolPayload::fail(message),
        }
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
fn map_client_error(error: &Error, method: &str) -> String {
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
         completion, document symbols and workspace symbols. Positions are one-based line and \
         column. Returns short, typed text results."
    } else {
        "LSP（语言服务器协议）代码导航工具：hover、definition、references、completion、\
         symbols（文档符号）和 workspace_symbols。行号和列号从 1 开始，返回简短的结构化文本结果。"
    }
}
