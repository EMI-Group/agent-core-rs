//! An in-process fake language server used by every client and tool test.
//!
//! The fake reads framed JSON-RPC from one half of a [`tokio::io::duplex`] pair and answers
//! canned hover/definition/references/completion/documentSymbol/workspaceSymbol results. It
//! also sends one `workspace/configuration` request during `initialize` (many real servers do)
//! and records every received message, so tests can assert ordering and versions without a
//! real language server or an installed binary.
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use serde_json::{json, Value};
use tokio::io::{AsyncWriteExt, DuplexStream};

use crate::lsp::protocol::{write_message, LspError, MessageReader};

/// The capabilities the fake server advertises.
#[must_use]
pub(crate) fn capabilities() -> Value {
    json!({
        "hoverProvider": true,
        "definitionProvider": true,
        "referencesProvider": true,
        "completionProvider": { "triggerCharacters": ["."] },
        "documentSymbolProvider": true,
        "workspaceSymbolProvider": true,
    })
}

/// A running fake server plus its recorded message log.
pub(crate) struct FakeServer {
    received: Arc<Mutex<Vec<Value>>>,
    handle: tokio::task::JoinHandle<()>,
}

impl FakeServer {
    /// Every message the fake server received, in order.
    pub(crate) fn messages(&self) -> Vec<Value> {
        self.received.lock().clone()
    }

    /// The `method` field of every received request/notification, in order.
    pub(crate) fn methods(&self) -> Vec<String> {
        self.received
            .lock()
            .iter()
            .filter_map(|message| message.get("method").and_then(Value::as_str))
            .map(ToOwned::to_owned)
            .collect()
    }

    /// The index of the first message with `method`, if any.
    pub(crate) fn position_of(&self, method: &str) -> Option<usize> {
        self.methods()
            .iter()
            .position(|candidate| candidate == method)
    }

    /// Waits until `method` has been received at least `count` times.
    pub(crate) async fn wait_for_count(&self, method: &str, count: usize) -> bool {
        for _ in 0..200 {
            let seen = self
                .methods()
                .iter()
                .filter(|candidate| candidate.as_str() == method)
                .count();
            if seen >= count {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        false
    }

    /// Waits until `method` has been received, polling for at most one second.
    pub(crate) async fn wait_for(&self, method: &str) -> bool {
        for _ in 0..200 {
            if self.position_of(method).is_some() {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        false
    }

    /// Waits for the server task to finish.
    pub(crate) async fn join(self) {
        let _ = self.handle.await;
    }
}

/// Spawns a fake server over `stream` and returns its handle.
pub(crate) fn spawn(stream: DuplexStream) -> FakeServer {
    let received = Arc::new(Mutex::new(Vec::new()));
    let log = Arc::clone(&received);
    let handle = tokio::spawn(async move { run(stream, log).await });
    FakeServer { received, handle }
}

/// Builds one `Location` object.
fn location(uri: &str, line: u32, character: u32) -> Value {
    json!({
        "uri": uri,
        "range": {
            "start": { "line": line, "character": character },
            "end": { "line": line, "character": character + 1 },
        },
    })
}

/// Extracts the document URI from a text-document request.
fn document_uri(message: &Value) -> String {
    message
        .get("params")
        .and_then(|params| params.get("textDocument"))
        .and_then(|document| document.get("uri"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned()
}

/// Builds a success response.
fn response(id: Value, result: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "result": result })
}

/// Builds an error response.
fn error(id: Value, error: &LspError) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "error": error })
}

/// The server loop.
async fn run(stream: DuplexStream, received: Arc<Mutex<Vec<Value>>>) {
    let (read, mut write) = tokio::io::split(stream);
    let mut reader = MessageReader::new(read);
    loop {
        let message = match reader.next().await {
            Ok(Some(message)) => message,
            Ok(None) | Err(_) => break,
        };
        received.lock().push(message.clone());
        let method = message
            .get("method")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        let id = message.get("id").cloned();
        if method.is_empty() {
            // A response to our own `workspace/configuration` request.
            continue;
        }
        let Some(id) = id else {
            if method == "exit" {
                let _ = write.shutdown().await;
                break;
            }
            continue;
        };
        let response = match method.as_str() {
            "initialize" => {
                let request = json!({
                    "jsonrpc": "2.0",
                    "id": 9001_u64,
                    "method": "workspace/configuration",
                    "params": { "items": [] },
                });
                if write_message(&mut write, &request).await.is_err() {
                    break;
                }
                // Wait for the client's answer before completing initialize.
                match reader.next().await {
                    Ok(Some(answer)) => received.lock().push(answer),
                    Ok(None) | Err(_) => break,
                }
                response(id, json!({ "capabilities": capabilities() }))
            }
            "shutdown" => response(id, Value::Null),
            "textDocument/hover" => response(
                id,
                json!({
                    "contents": { "kind": "markdown", "value": "**fake hover**" },
                    "range": {
                        "start": { "line": 1, "character": 4 },
                        "end": { "line": 1, "character": 8 },
                    },
                }),
            ),
            "textDocument/definition" => {
                let uri = document_uri(&message);
                if uri.contains("slow") {
                    tokio::time::sleep(Duration::from_millis(300)).await;
                }
                response(id, json!([location(&uri, 2, 4)]))
            }
            "textDocument/references" => {
                let uri = document_uri(&message);
                response(id, json!([location(&uri, 1, 2), location(&uri, 3, 4)]))
            }
            "textDocument/completion" => {
                let uri = document_uri(&message);
                if uri.contains("unsupported") {
                    error(
                        id,
                        &LspError::new(-32601, "method not found: textDocument/completion"),
                    )
                } else {
                    response(
                        id,
                        json!({
                            "isIncomplete": false,
                            "items": [
                                { "label": "alpha", "kind": 3, "detail": "fake detail" },
                                { "label": "beta", "kind": 6 },
                            ],
                        }),
                    )
                }
            }
            "textDocument/documentSymbol" => response(
                id,
                json!([{
                    "name": "main",
                    "kind": 12,
                    "detail": "fn main()",
                    "range": {
                        "start": { "line": 0, "character": 0 },
                        "end": { "line": 4, "character": 1 },
                    },
                    "selectionRange": {
                        "start": { "line": 0, "character": 3 },
                        "end": { "line": 0, "character": 7 },
                    },
                    "children": [{
                        "name": "helper",
                        "kind": 12,
                        "range": {
                            "start": { "line": 6, "character": 0 },
                            "end": { "line": 8, "character": 1 },
                        },
                        "selectionRange": {
                            "start": { "line": 6, "character": 3 },
                            "end": { "line": 6, "character": 9 },
                        },
                    }],
                }]),
            ),
            "workspace/symbol" => {
                let query = message
                    .get("params")
                    .and_then(|params| params.get("query"))
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                if query == "boom" {
                    error(
                        id,
                        &LspError::new(-32601, "method not found: workspace/symbol"),
                    )
                } else {
                    response(
                        id,
                        json!([{
                            "name": "main",
                            "kind": 12,
                            "location": location("file:///workspace/src/lib.rs", 2, 4),
                            "containerName": "crate",
                        }]),
                    )
                }
            }
            other => error(
                id,
                &LspError::new(-32601, format!("method not found: {other}")),
            ),
        };
        if write_message(&mut write, &response).await.is_err() {
            break;
        }
    }
}
