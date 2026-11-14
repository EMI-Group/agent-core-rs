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
//!
//! Id-less notifications are dispatched to handlers registered through
//! [`LspClient::add_notification_handler`] (`textDocument/publishDiagnostics` is the reason
//! the seam exists), and every exit path records why the client stopped, so a host can watch
//! [`LspClient::closed`] instead of polling a dead server.
use std::collections::HashMap;
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use serde_json::{json, Value};
use tokio::sync::{mpsc, oneshot, watch};

use openjiuwen_common::error::{Error, Result};

use super::protocol::{
    parse_completion_items, parse_document_symbols, parse_hover, parse_locations,
    parse_workspace_symbols, request_message, response_message, CompletionItem, DocumentSymbol,
    Hover, Location, LspError, Position, ServerCapabilities, SymbolInformation,
};
use super::transport::LspTransport;
use super::wire::{
    parse_call_hierarchy_incoming_calls, parse_call_hierarchy_items,
    parse_call_hierarchy_outgoing_calls, parse_document_diagnostic_report,
    CallHierarchyIncomingCall, CallHierarchyItem, CallHierarchyOutgoingCall,
    DocumentDiagnosticReport,
};

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

/// One notification callback, keyed by the LSP method it listens for.
type NotificationHandler = Arc<dyn Fn(&Value) + Send + Sync>;

/// Notification handlers in registration order.
type NotificationHandlers = Arc<Mutex<Vec<(String, NotificationHandler)>>>;

/// Liveness state the I/O task publishes when it exits.
///
/// The first close reason wins, so an explicit [`LspClient::shutdown`] is not overwritten by
/// the task's own teardown path.
struct Lifecycle {
    alive: AtomicBool,
    reason: Mutex<Option<String>>,
    closed: watch::Sender<bool>,
}

impl Lifecycle {
    /// Creates a live lifecycle whose watch channel starts at `false`.
    fn new() -> Self {
        let (closed, _receiver) = watch::channel(false);
        Self {
            alive: AtomicBool::new(true),
            reason: Mutex::new(None),
            closed,
        }
    }

    /// Marks the client closed with `reason`, keeping the first reason recorded.
    fn close(&self, reason: impl Into<String>) {
        {
            let mut slot = self.reason.lock();
            if slot.is_none() {
                *slot = Some(reason.into());
            }
        }
        self.alive.store(false, Ordering::SeqCst);
        self.closed.send_replace(true);
    }

    /// Whether the client is still running.
    fn is_alive(&self) -> bool {
        self.alive.load(Ordering::SeqCst)
    }

    /// The recorded close reason, when the client has stopped.
    fn reason(&self) -> Option<String> {
        self.reason.lock().clone()
    }

    /// A receiver that flips to `true` when the client stops.
    fn subscribe(&self) -> watch::Receiver<bool> {
        self.closed.subscribe()
    }
}

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
    handlers: NotificationHandlers,
    lifecycle: Arc<Lifecycle>,
}

impl LspClient {
    /// Starts the I/O task over `transport` and bounds each request with `request_timeout`.
    #[must_use]
    pub fn new(transport: Box<dyn LspTransport>, request_timeout: Duration) -> Self {
        let (outgoing, receiver) = mpsc::channel(OUTGOING_CAPACITY);
        let pending: PendingMap = Arc::new(Mutex::new(HashMap::new()));
        let task_pending = Arc::clone(&pending);
        let handlers: NotificationHandlers = Arc::new(Mutex::new(Vec::new()));
        let lifecycle = Arc::new(Lifecycle::new());
        tokio::spawn(run_io(
            transport,
            receiver,
            task_pending,
            Arc::clone(&handlers),
            Arc::clone(&lifecycle),
        ));
        Self {
            outgoing,
            pending,
            next_id: AtomicU64::new(1),
            request_timeout,
            capabilities: Mutex::new(None),
            versions: Mutex::new(HashMap::new()),
            handlers,
            lifecycle,
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

    /// Registers `handler` for every id-less notification whose `method` matches.
    ///
    /// Handlers run on the I/O task in registration order; a panicking handler is logged and
    /// the next one still runs. Registering is safe before and after
    /// [`LspClient::initialize`], and a notification with no matching handler stays ignored.
    pub fn add_notification_handler(
        &self,
        method: impl Into<String>,
        handler: Arc<dyn Fn(&Value) + Send + Sync>,
    ) {
        self.handlers.lock().push((method.into(), handler));
    }

    /// Whether the I/O task is still running.
    ///
    /// Turns `false` once the transport closes for any reason, including an explicit
    /// [`LspClient::shutdown`] and a server that dies mid-request.
    #[must_use]
    pub fn is_alive(&self) -> bool {
        self.lifecycle.is_alive()
    }

    /// A watch receiver that flips to `true` when the client stops.
    ///
    /// Hosts use it to stop reusing a client whose server crashed without polling
    /// [`LspClient::is_alive`].
    #[must_use]
    pub fn closed(&self) -> watch::Receiver<bool> {
        self.lifecycle.subscribe()
    }

    /// A short reason for the client stopping, or `None` while it is alive.
    ///
    /// The first recorded reason wins: for example `"LSP server closed the connection"`,
    /// `"LSP transport stopped"` or `"shutdown requested"`.
    #[must_use]
    pub fn close_reason(&self) -> Option<String> {
        self.lifecycle.reason()
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

    /// Requests `textDocument/implementation` and normalises the locations.
    ///
    /// Accepts both `Location` and `LocationLink` results, like
    /// [`LspClient::definition`].
    ///
    /// # Errors
    ///
    /// Propagates transport, timeout and JSON-RPC errors.
    pub async fn implementation(&self, uri: &str, position: Position) -> Result<Vec<Location>> {
        let result = self
            .request(
                "textDocument/implementation",
                json!({ "textDocument": { "uri": uri }, "position": position }),
            )
            .await?;
        Ok(parse_locations(&result))
    }

    /// Requests `textDocument/prepareCallHierarchy` and normalises the items.
    ///
    /// # Errors
    ///
    /// Propagates transport, timeout and JSON-RPC errors.
    pub async fn prepare_call_hierarchy(
        &self,
        uri: &str,
        position: Position,
    ) -> Result<Vec<CallHierarchyItem>> {
        let result = self
            .request(
                "textDocument/prepareCallHierarchy",
                json!({ "textDocument": { "uri": uri }, "position": position }),
            )
            .await?;
        Ok(parse_call_hierarchy_items(&result))
    }

    /// Requests `callHierarchy/incomingCalls` for `item`.
    ///
    /// The normalised item is serialized back into the flat LSP item shape (including `data`
    /// when the server sent one) because servers correlate the request with that payload.
    ///
    /// # Errors
    ///
    /// Propagates transport, timeout and JSON-RPC errors.
    pub async fn incoming_calls(
        &self,
        item: &CallHierarchyItem,
    ) -> Result<Vec<CallHierarchyIncomingCall>> {
        let result = self
            .request(
                "callHierarchy/incomingCalls",
                json!({ "item": call_hierarchy_item_value(item) }),
            )
            .await?;
        Ok(parse_call_hierarchy_incoming_calls(&result))
    }

    /// Requests `callHierarchy/outgoingCalls` for `item`.
    ///
    /// # Errors
    ///
    /// Propagates transport, timeout and JSON-RPC errors.
    pub async fn outgoing_calls(
        &self,
        item: &CallHierarchyItem,
    ) -> Result<Vec<CallHierarchyOutgoingCall>> {
        let result = self
            .request(
                "callHierarchy/outgoingCalls",
                json!({ "item": call_hierarchy_item_value(item) }),
            )
            .await?;
        Ok(parse_call_hierarchy_outgoing_calls(&result))
    }

    /// Requests `textDocument/diagnostic`, passing `previous_result_id` when the caller has a
    /// cached report so the server can answer `unchanged`.
    ///
    /// # Errors
    ///
    /// Propagates transport, timeout and JSON-RPC errors.
    pub async fn document_diagnostic(
        &self,
        uri: &str,
        previous_result_id: Option<&str>,
    ) -> Result<DocumentDiagnosticReport> {
        let mut params = json!({ "textDocument": { "uri": uri } });
        if let Some(previous) = previous_result_id {
            params["previousResultId"] = json!(previous);
        }
        let result = self.request("textDocument/diagnostic", params).await?;
        Ok(parse_document_diagnostic_report(&result))
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
    /// the server is unresponsive; all failures are best effort. The client is marked closed
    /// with the reason `"shutdown requested"`.
    pub async fn shutdown(&self) {
        let timeout = self.request_timeout.min(Duration::from_secs(2));
        let _ = self
            .request_with_timeout("shutdown", Value::Null, timeout)
            .await;
        let _ = self.notify("exit", Value::Null).await;
        self.lifecycle.close("shutdown requested");
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
///
/// Every exit path records why the client stopped, fails the pending requests with that
/// reason, closes the transport and marks [`LspClient::is_alive`] `false`.
async fn run_io(
    mut transport: Box<dyn LspTransport>,
    mut outgoing: mpsc::Receiver<Outgoing>,
    pending: PendingMap,
    handlers: NotificationHandlers,
    lifecycle: Arc<Lifecycle>,
) {
    let reason: String;
    let mut failure: Option<Error> = None;
    loop {
        tokio::select! {
            command = outgoing.recv() => match command {
                Some(Outgoing::Message(message)) => {
                    if let Err(error) = transport.send(message).await {
                        reason = format!("LSP transport failed: {}", error.message());
                        failure = Some(error);
                        break;
                    }
                }
                Some(Outgoing::Close) | None => {
                    reason = "LSP transport stopped".to_owned();
                    break;
                }
            },
            incoming = transport.recv() => match incoming {
                Ok(Some(message)) => {
                    if let Err(error) = dispatch(&mut transport, &pending, &handlers, message).await {
                        reason = format!("LSP transport failed: {}", error.message());
                        failure = Some(error);
                        break;
                    }
                }
                Ok(None) => {
                    reason = "LSP server closed the connection".to_owned();
                    break;
                }
                Err(error) => {
                    reason = error.message().into_owned();
                    failure = Some(error);
                    break;
                }
            },
        }
    }
    let _ = transport.close().await;
    lifecycle.close(reason.clone());
    match failure {
        Some(error) => fail_all(&pending, &error),
        None => fail_all(&pending, &Error::transport(reason)),
    }
}

/// Dispatches one incoming message: responses resolve waiters, server requests get an answer,
/// notifications go to the registered handlers.
async fn dispatch(
    transport: &mut Box<dyn LspTransport>,
    pending: &PendingMap,
    handlers: &NotificationHandlers,
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
            _ => dispatch_notification(handlers, method, &message),
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

/// Runs the handlers registered for `method` with the notification `params`, in order.
///
/// A panicking handler is logged and skipped so one bad callback cannot take the I/O task (and
/// therefore every pending request) down with it.
fn dispatch_notification(handlers: &NotificationHandlers, method: &str, message: &Value) {
    // Clone the matching handlers out of the lock so a handler may register another one.
    let matched: Vec<NotificationHandler> = {
        let handlers = handlers.lock();
        handlers
            .iter()
            .filter(|(name, _)| name == method)
            .map(|(_, handler)| Arc::clone(handler))
            .collect()
    };
    if matched.is_empty() {
        tracing::debug!(
            target: "openjiuwen_harness::lsp",
            method,
            "ignoring LSP notification"
        );
        return;
    }
    let params = message.get("params").unwrap_or(&Value::Null);
    for handler in matched {
        if std::panic::catch_unwind(AssertUnwindSafe(|| handler(params))).is_err() {
            tracing::warn!(
                target: "openjiuwen_harness::lsp",
                method,
                "LSP notification handler panicked"
            );
        }
    }
}

/// Serializes a normalised call-hierarchy item back into the flat LSP item shape.
fn call_hierarchy_item_value(item: &CallHierarchyItem) -> Value {
    let mut value = json!({
        "name": item.name,
        "kind": item.kind,
        "uri": item.uri,
        "range": item.range,
        "selectionRange": item.selection_range,
    });
    if let Some(detail) = &item.detail {
        value["detail"] = json!(detail);
    }
    if !item.tags.is_empty() {
        value["tags"] = json!(item.tags);
    }
    if let Some(data) = &item.data {
        value["data"] = data.clone();
    }
    value
}

/// Fails every pending request with a transport error carrying `error`'s message.
fn fail_all(pending: &PendingMap, error: &Error) {
    let message = error.message().into_owned();
    for (_, sender) in pending.lock().drain() {
        let _ = sender.send(Err(Error::transport(message.clone())));
    }
}
