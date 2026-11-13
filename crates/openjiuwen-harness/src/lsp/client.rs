//! Request/response correlation for one language server.
//!
//! Ports the single-reader design of `openjiuwen/harness/lsp/core/client.py` with tokio
//! instead of asyncio: one background task owns the [`LspTransport`], serialising outgoing
//! messages and dispatching incoming responses to the waiting request through a `oneshot`
//! channel. Every request is bounded by [`LspClient::request_timeout`] via
//! [`tokio::time::timeout`], so a wedged server cannot hang the agent loop.
//!
//! The task also answers server-initiated requests (`workspace/configuration`,
//! `workspace/workspaceFolders`, `client/registerCapability`) with the same empty payloads
//! Python uses, which many servers require before they answer `initialize`.
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use serde_json::{json, Value};
use tokio::sync::{mpsc, oneshot};

use openjiuwen_common::error::{Error, Result};

use super::protocol::{
    parse_completion_items, parse_document_symbols, parse_hover, parse_locations,
    parse_workspace_symbols, request_message, response_message, CompletionItem, DocumentSymbol,
    Hover, Location, LspError, Position, ServerCapabilities, SymbolInformation,
};
use super::transport::LspTransport;

/// Capacity of the outgoing command queue; requests beyond it wait for the I/O task.
const OUTGOING_CAPACITY: usize = 64;

/// A message the client hands to its I/O task.
enum Outgoing {
    /// Encode and write this JSON-RPC message.
    Message(Value),
    /// Close the transport and stop the task.
    Close,
}

/// Requests awaiting a response, keyed by JSON-RPC id.
type PendingMap = Arc<Mutex<HashMap<u64, oneshot::Sender<Result<Value>>>>>;

/// A JSON-RPC client for one language server.
///
/// Cloning is not supported; wrap in [`Arc`] to share, as [`LspTool`](crate::lsp::LspTool)
/// does. Dropping the last reference asks the I/O task to close the transport (and kill a
/// [`StdioTransport`](crate::lsp::StdioTransport) child); call [`LspClient::shutdown`] for a
/// graceful protocol shutdown first.
pub struct LspClient {
    outgoing: mpsc::Sender<Outgoing>,
    pending: PendingMap,
    next_id: AtomicU64,
    request_timeout: Duration,
    capabilities: Mutex<Option<ServerCapabilities>>,
    versions: Mutex<HashMap<String, i64>>,
}

impl LspClient {
    /// Starts the I/O task over `transport` and bounds each request with `request_timeout`.
    #[must_use]
    pub fn new(transport: Box<dyn LspTransport>, request_timeout: Duration) -> Self {
        let (outgoing, receiver) = mpsc::channel(OUTGOING_CAPACITY);
        let pending: PendingMap = Arc::new(Mutex::new(HashMap::new()));
        let task_pending = Arc::clone(&pending);
        tokio::spawn(run_io(transport, receiver, task_pending));
        Self {
            outgoing,
            pending,
            next_id: AtomicU64::new(1),
            request_timeout,
            capabilities: Mutex::new(None),
            versions: Mutex::new(HashMap::new()),
        }
    }

    /// The per-request timeout this client enforces.
    #[must_use]
    pub const fn request_timeout(&self) -> Duration {
        self.request_timeout
    }

    /// The capabilities captured from `initialize`, when it has completed.
    #[must_use]
    pub fn capabilities(&self) -> Option<ServerCapabilities> {
        self.capabilities.lock().clone()
    }

    /// Whether the `initialize` handshake has completed.
    #[must_use]
    pub fn is_initialized(&self) -> bool {
        self.capabilities.lock().is_some()
    }

    /// Performs the LSP handshake and captures the server capabilities.
    ///
    /// Sends `initialize` with the standard client capabilities, then the `initialized`
    /// notification. `root_uri` is the `file://` URI of the workspace root.
    ///
    /// # Errors
    ///
    /// Propagates transport failures and JSON-RPC errors; a timeout surfaces as
    /// [`Error::Timeout`].
    pub async fn initialize(
        &self,
        root_uri: &str,
        capabilities: Value,
    ) -> Result<ServerCapabilities> {
        let params = json!({
            "processId": std::process::id(),
            "rootUri": root_uri,
            "workspaceFolders": [{ "uri": root_uri, "name": "workspace" }],
            "capabilities": capabilities,
        });
        let result = self.request("initialize", params).await?;
        let capabilities = ServerCapabilities::from_initialize_result(&result);
        *self.capabilities.lock() = Some(capabilities.clone());
        self.notify("initialized", json!({})).await?;
        Ok(capabilities)
    }

    /// Sends `textDocument/didOpen`, or a full-text `didChange` when the document is already
    /// open.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Transport`] when the client has been shut down.
    pub async fn did_open(&self, uri: &str, language_id: &str, text: &str) -> Result<()> {
        let (method, params) = {
            let mut versions = self.versions.lock();
            if let Some(version) = versions.get_mut(uri) {
                *version += 1;
                (
                    "textDocument/didChange",
                    json!({
                        "textDocument": { "uri": uri, "version": *version },
                        "contentChanges": [{ "text": text }],
                    }),
                )
            } else {
                versions.insert(uri.to_owned(), 1);
                (
                    "textDocument/didOpen",
                    json!({
                        "textDocument": {
                            "uri": uri,
                            "languageId": language_id,
                            "version": 1,
                            "text": text,
                        },
                    }),
                )
            }
        };
        self.notify(method, params).await
    }

    /// Sends a full-text `textDocument/didChange`, bumping the document version.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Transport`] when the client has been shut down.
    pub async fn did_change(&self, uri: &str, text: &str) -> Result<()> {
        let version = {
            let mut versions = self.versions.lock();
            let version = versions.entry(uri.to_owned()).or_insert(0);
            *version += 1;
            *version
        };
        self.notify(
            "textDocument/didChange",
            json!({
                "textDocument": { "uri": uri, "version": version },
                "contentChanges": [{ "text": text }],
            }),
        )
        .await
    }

    /// Requests `textDocument/hover` and normalises the result.
    ///
    /// # Errors
    ///
    /// Propagates transport, timeout and JSON-RPC errors.
    pub async fn hover(&self, uri: &str, position: Position) -> Result<Option<Hover>> {
        let result = self
            .request(
                "textDocument/hover",
                json!({ "textDocument": { "uri": uri }, "position": position }),
            )
            .await?;
        Ok(parse_hover(&result))
    }

    /// Requests `textDocument/definition` and normalises the locations.
    ///
    /// # Errors
    ///
    /// Propagates transport, timeout and JSON-RPC errors.
    pub async fn definition(&self, uri: &str, position: Position) -> Result<Vec<Location>> {
        let result = self
            .request(
                "textDocument/definition",
                json!({ "textDocument": { "uri": uri }, "position": position }),
            )
            .await?;
        Ok(parse_locations(&result))
    }

    /// Requests `textDocument/references` and normalises the locations.
    ///
    /// # Errors
    ///
    /// Propagates transport, timeout and JSON-RPC errors.
    pub async fn references(
        &self,
        uri: &str,
        position: Position,
        include_declaration: bool,
    ) -> Result<Vec<Location>> {
        let result = self
            .request(
                "textDocument/references",
                json!({
                    "textDocument": { "uri": uri },
                    "position": position,
                    "context": { "includeDeclaration": include_declaration },
                }),
            )
            .await?;
        Ok(parse_locations(&result))
    }

    /// Requests `textDocument/completion` and normalises the items.
    ///
    /// # Errors
    ///
    /// Propagates transport, timeout and JSON-RPC errors.
    pub async fn completion(&self, uri: &str, position: Position) -> Result<Vec<CompletionItem>> {
        let result = self
            .request(
                "textDocument/completion",
                json!({ "textDocument": { "uri": uri }, "position": position }),
            )
            .await?;
        Ok(parse_completion_items(&result))
    }

    /// Requests `textDocument/documentSymbol` and normalises the symbols.
    ///
    /// # Errors
    ///
    /// Propagates transport, timeout and JSON-RPC errors.
    pub async fn document_symbols(&self, uri: &str) -> Result<Vec<DocumentSymbol>> {
        let result = self
            .request(
                "textDocument/documentSymbol",
                json!({ "textDocument": { "uri": uri } }),
            )
            .await?;
        Ok(parse_document_symbols(&result))
    }

    /// Requests `workspace/symbol` for `query` (empty returns every symbol the server offers).
    ///
    /// # Errors
    ///
    /// Propagates transport, timeout and JSON-RPC errors.
    pub async fn workspace_symbols(&self, query: &str) -> Result<Vec<SymbolInformation>> {
        let result = self
            .request("workspace/symbol", json!({ "query": query }))
            .await?;
        Ok(parse_workspace_symbols(&result))
    }

    /// Sends a JSON-RPC request and waits for its result.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Timeout`] when the server does not answer within
    /// [`LspClient::request_timeout`], [`Error::Transport`] when the connection closes, and
    /// [`Error::Tool`] when the server returns a JSON-RPC error.
    pub async fn request(&self, method: &str, params: Value) -> Result<Value> {
        self.request_with_timeout(method, params, self.request_timeout)
            .await
    }

    /// Sends a JSON-RPC notification; no response is awaited.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Transport`] when the client has been shut down. The write itself
    /// completes asynchronously through the I/O task.
    pub async fn notify(&self, method: &str, params: Value) -> Result<()> {
        let message = super::protocol::notification_message(method, params);
        self.outgoing
            .send(Outgoing::Message(message))
            .await
            .map_err(|_| Error::transport("LSP client is shut down"))
    }

    /// Performs the graceful LSP shutdown: `shutdown` request, `exit` notification, close.
    ///
    /// The `shutdown` request is bounded to two seconds so teardown cannot stall even when
    /// the server is unresponsive; all failures are best effort.
    pub async fn shutdown(&self) {
        let timeout = self.request_timeout.min(Duration::from_secs(2));
        let _ = self
            .request_with_timeout("shutdown", Value::Null, timeout)
            .await;
        let _ = self.notify("exit", Value::Null).await;
        let _ = self.outgoing.send(Outgoing::Close).await;
    }

    /// Sends a request bounded by `timeout` instead of the client default.
    async fn request_with_timeout(
        &self,
        method: &str,
        params: Value,
        timeout: Duration,
    ) -> Result<Value> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (sender, receiver) = oneshot::channel();
        self.pending.lock().insert(id, sender);
        let message = request_message(id, method, params);
        if self
            .outgoing
            .send(Outgoing::Message(message))
            .await
            .is_err()
        {
            self.pending.lock().remove(&id);
            return Err(Error::transport("LSP client is shut down"));
        }
        match tokio::time::timeout(timeout, receiver).await {
            Ok(Ok(outcome)) => outcome,
            Ok(Err(_)) => Err(Error::transport(format!(
                "LSP server closed the connection while waiting for '{method}'"
            ))),
            Err(_) => {
                self.pending.lock().remove(&id);
                Err(Error::timeout(format!(
                    "LSP request '{method}' timed out after {} ms",
                    timeout.as_millis()
                )))
            }
        }
    }
}

impl Drop for LspClient {
    fn drop(&mut self) {
        let _ = self.outgoing.try_send(Outgoing::Close);
    }
}

/// Owns the transport for the lifetime of the client.
async fn run_io(
    mut transport: Box<dyn LspTransport>,
    mut outgoing: mpsc::Receiver<Outgoing>,
    pending: PendingMap,
) {
    loop {
        tokio::select! {
            command = outgoing.recv() => match command {
                Some(Outgoing::Message(message)) => {
                    if let Err(error) = transport.send(message).await {
                        fail_all(&pending, &error);
                        let _ = transport.close().await;
                        break;
                    }
                }
                Some(Outgoing::Close) | None => {
                    let _ = transport.close().await;
                    break;
                }
            },
            incoming = transport.recv() => match incoming {
                Ok(Some(message)) => {
                    if let Err(error) = dispatch(&mut transport, &pending, message).await {
                        fail_all(&pending, &error);
                        let _ = transport.close().await;
                        break;
                    }
                }
                Ok(None) => {
                    fail_all(&pending, &Error::transport("LSP server closed the connection"));
                    let _ = transport.close().await;
                    break;
                }
                Err(error) => {
                    fail_all(&pending, &error);
                    let _ = transport.close().await;
                    break;
                }
            },
        }
    }
    fail_all(&pending, &Error::transport("LSP transport stopped"));
}

/// Dispatches one incoming message: responses resolve waiters, server requests get an answer.
async fn dispatch(
    transport: &mut Box<dyn LspTransport>,
    pending: &PendingMap,
    message: Value,
) -> Result<()> {
    if let Some(method) = message.get("method").and_then(Value::as_str) {
        match message.get("id") {
            Some(id) if !id.is_null() => {
                let result = match method {
                    "workspace/configuration" | "workspace/workspaceFolders" => json!([]),
                    _ => Value::Null,
                };
                tracing::debug!(
                    target: "openjiuwen_harness::lsp",
                    method,
                    "answering server-initiated LSP request"
                );
                transport.send(response_message(id.clone(), result)).await?;
            }
            _ => {
                tracing::debug!(
                    target: "openjiuwen_harness::lsp",
                    method,
                    "ignoring LSP notification"
                );
            }
        }
        return Ok(());
    }
    let Some(id) = message.get("id").and_then(Value::as_u64) else {
        return Ok(());
    };
    let Some(sender) = pending.lock().remove(&id) else {
        return Ok(());
    };
    let outcome = match message.get("error") {
        Some(error) => match serde_json::from_value::<LspError>(error.clone()) {
            Ok(error) => Err(error.into_error()),
            Err(_) => Err(Error::tool("LSP server returned a malformed error object")),
        },
        None => Ok(message.get("result").cloned().unwrap_or(Value::Null)),
    };
    let _ = sender.send(outcome);
    Ok(())
}

/// Fails every pending request with a transport error carrying `error`'s message.
fn fail_all(pending: &PendingMap, error: &Error) {
    let message = error.message().into_owned();
    for (_, sender) in pending.lock().drain() {
        let _ = sender.send(Err(Error::transport(message.clone())));
    }
}
