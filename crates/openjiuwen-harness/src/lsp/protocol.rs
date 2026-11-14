//! JSON-RPC 2.0 framing and the LSP wire types.
//!
//! Ports the framing and result shapes of `openjiuwen/harness/lsp/core/client.py` and
//! `openjiuwen/harness/lsp/core/types.py`. Messages are framed as
//!
//! ```text
//! Content-Length: <byte length>\r\n
//! \r\n
//! <UTF-8 JSON body>
//! ```
//!
//! [`FrameDecoder`] is a synchronous, incremental parser (tested with hand-written byte
//! fixtures), [`MessageReader`] wraps it over an async byte stream, and [`encode_message`] /
//! [`write_message`] produce the same frames. Frames larger than [`MAX_CONTENT_LENGTH`] are
//! rejected instead of buffered.
//!
//! Server results are normalised into compact serde structs ([`Location`], [`Hover`],
//! [`CompletionItem`], [`DocumentSymbol`], [`SymbolInformation`]) instead of leaking raw
//! `Value` through the client, because LSP servers legitimately return several shapes for the
//! same method (for example `Location` versus `LocationLink`, or a `DocumentSymbol` tree
//! versus flat `SymbolInformation`).
// The message builders take owned values so callers can move freshly built params into the
// JSON tree without cloning them; the alternative (`&Value`) only shifts the clone inside.
#![allow(clippy::needless_pass_by_value)]
use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

use openjiuwen_common::error::{Error, Result};

/// The JSON-RPC protocol version every message carries.
pub const JSONRPC_VERSION: &str = "2.0";
/// Largest accepted frame body, in bytes (8 MiB).
pub const MAX_CONTENT_LENGTH: usize = 8 * 1024 * 1024;
/// Largest accepted header block, in bytes (8 KiB).
pub const MAX_HEADER_LENGTH: usize = 8 * 1024;
/// The header name that carries the body length.
const CONTENT_LENGTH: &str = "content-length";
/// The header/body separator.
const HEADER_TERMINATOR: &[u8] = b"\r\n\r\n";

/// An error returned by a language server inside a JSON-RPC error response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LspError {
    /// JSON-RPC error code; `-32601` means "method not found".
    pub code: i64,
    /// Human-readable error message.
    pub message: String,
    /// Optional structured error data.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl LspError {
    /// Builds an error with `code` and `message`.
    #[must_use]
    pub fn new(code: i64, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            data: None,
        }
    }

    /// Whether the server does not implement the requested method.
    #[must_use]
    pub fn is_method_not_found(&self) -> bool {
        self.code == -32601
    }

    /// Converts the JSON-RPC error into the crate's typed error.
    ///
    /// The whole [`LspError`] is kept as the [`Error::External`] source so the numeric code
    /// survives the conversion: callers match on [`LspError::is_method_not_found`] through
    /// [`server_error`] instead of parsing the rendered message. The rendered payload is
    /// unchanged (`LSP error <code>: <message>`), because [`Error::message`] returns the
    /// source's display text.
    pub fn into_error(self) -> Error {
        Error::external(self)
    }
}

/// The typed JSON-RPC error behind `error`, when it came from a language server.
///
/// [`LspError::into_error`] keeps the whole error, so a consumer holding only an [`Error`] can
/// still match on [`LspError::code`] / [`LspError::is_method_not_found`].
#[must_use]
pub fn server_error(error: &Error) -> Option<&LspError> {
    match error {
        Error::External(source) => source.downcast_ref::<LspError>(),
        _ => None,
    }
}

/// Whether `error` is a "method not found" answer from a language server.
///
/// The typed code ([`LspError::is_method_not_found`]) is authoritative: it is the only evidence
/// that survives a server whose message says something else entirely. The message check is a
/// documented last resort for servers that answer "method not found" without the JSON-RPC code
/// (`-32601`) — a non-standard code on an unimplemented request, or an error object without a
/// numeric code at all. It looks for the phrase only: the numeric code is matched typely, so a
/// code that merely appears in text is not treated as evidence.
#[must_use]
pub fn is_method_not_found(error: &Error) -> bool {
    if server_error(error).is_some_and(LspError::is_method_not_found) {
        return true;
    }
    error
        .message()
        .to_ascii_lowercase()
        .contains("method not found")
}

impl fmt::Display for LspError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "LSP error {}: {}", self.code, self.message)
    }
}

impl std::error::Error for LspError {}

/// Builds a JSON-RPC request object.
#[must_use]
pub fn request_message(id: u64, method: &str, params: Value) -> Value {
    json!({
        "jsonrpc": JSONRPC_VERSION,
        "id": id,
        "method": method,
        "params": params,
    })
}

/// Builds a JSON-RPC notification object (no id, no response expected).
#[must_use]
pub fn notification_message(method: &str, params: Value) -> Value {
    json!({
        "jsonrpc": JSONRPC_VERSION,
        "method": method,
        "params": params,
    })
}

/// Builds a successful JSON-RPC response.
#[must_use]
pub fn response_message(id: Value, result: Value) -> Value {
    json!({
        "jsonrpc": JSONRPC_VERSION,
        "id": id,
        "result": result,
    })
}

/// Builds an error JSON-RPC response.
#[must_use]
pub fn error_response(id: Value, error: &LspError) -> Value {
    json!({
        "jsonrpc": JSONRPC_VERSION,
        "id": id,
        "error": error,
    })
}

/// Encodes `message` as a `Content-Length` frame.
///
/// # Errors
///
/// Returns [`Error::Validation`] when the message cannot be serialized or its body exceeds
/// [`MAX_CONTENT_LENGTH`].
pub fn encode_message(message: &Value) -> Result<Vec<u8>> {
    let body = serde_json::to_vec(message)
        .map_err(|error| Error::validation(format!("cannot encode LSP message: {error}")))?;
    if body.len() > MAX_CONTENT_LENGTH {
        return Err(Error::validation(format!(
            "LSP message body is {} bytes, above the {} byte limit",
            body.len(),
            MAX_CONTENT_LENGTH
        )));
    }
    let mut frame = format!("Content-Length: {}\r\n\r\n", body.len()).into_bytes();
    frame.extend_from_slice(&body);
    Ok(frame)
}

/// Writes one framed `message` to `writer` and flushes it.
///
/// # Errors
///
/// Propagates framing failures and write failures as [`Error::Transport`].
pub async fn write_message<W>(writer: &mut W, message: &Value) -> Result<()>
where
    W: AsyncWrite + Unpin + Send,
{
    let frame = encode_message(message)?;
    writer
        .write_all(&frame)
        .await
        .map_err(|error| Error::transport(format!("failed to write LSP frame: {error}")))?;
    writer
        .flush()
        .await
        .map_err(|error| Error::transport(format!("failed to flush LSP frame: {error}")))?;
    Ok(())
}

/// Incremental `Content-Length` frame decoder.
///
/// Feed arbitrary byte chunks with [`FrameDecoder::feed`] and pull complete messages with
/// [`FrameDecoder::try_next`]; a `None` result means "need more bytes". Extra headers are
/// tolerated, header names are case-insensitive, and the body length is measured in bytes
/// (so multi-byte UTF-8 bodies stay correct).
#[derive(Debug, Default)]
pub struct FrameDecoder {
    buffer: Vec<u8>,
}

impl FrameDecoder {
    /// Creates an empty decoder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends `bytes` to the pending buffer.
    pub fn feed(&mut self, bytes: &[u8]) {
        self.buffer.extend_from_slice(bytes);
    }

    /// Returns whether the decoder holds unparsed bytes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Attempts to decode one complete message.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for a header block without `Content-Length`, a
    /// non-numeric or oversized length, a malformed header line, invalid JSON, or a body
    /// that is not a JSON object.
    pub fn try_next(&mut self) -> Result<Option<Value>> {
        if self.buffer.len() > MAX_CONTENT_LENGTH + MAX_HEADER_LENGTH + HEADER_TERMINATOR.len() {
            return Err(Error::validation(
                "LSP frame exceeds the maximum buffered size",
            ));
        }
        let Some(header_end) = find_subslice(&self.buffer, HEADER_TERMINATOR) else {
            if self.buffer.len() > MAX_HEADER_LENGTH {
                return Err(Error::validation(format!(
                    "LSP header block exceeds the {MAX_HEADER_LENGTH} byte limit"
                )));
            }
            return Ok(None);
        };
        if header_end > MAX_HEADER_LENGTH {
            return Err(Error::validation(format!(
                "LSP header block exceeds the {MAX_HEADER_LENGTH} byte limit"
            )));
        }
        let header = std::str::from_utf8(&self.buffer[..header_end])
            .map_err(|_| Error::validation("LSP frame header is not valid UTF-8"))?;
        let length = parse_content_length(header)?;
        if length > MAX_CONTENT_LENGTH {
            return Err(Error::validation(format!(
                "LSP frame body of {length} bytes exceeds the {MAX_CONTENT_LENGTH} byte limit"
            )));
        }
        let body_start = header_end + HEADER_TERMINATOR.len();
        let frame_end = body_start
            .checked_add(length)
            .ok_or_else(|| Error::validation("LSP content length overflows the buffer"))?;
        if self.buffer.len() < frame_end {
            return Ok(None);
        }
        let body = self.buffer[body_start..frame_end].to_vec();
        self.buffer.drain(..frame_end);
        let value: Value = serde_json::from_slice(&body)
            .map_err(|error| Error::validation(format!("invalid LSP JSON body: {error}")))?;
        if !value.is_object() {
            return Err(Error::validation("LSP message body must be a JSON object"));
        }
        Ok(Some(value))
    }

    /// Finishes the stream, rejecting a frame that ended halfway through.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Transport`] when unparsed bytes remain.
    pub fn finish(&self) -> Result<Option<Value>> {
        if self.buffer.is_empty() {
            Ok(None)
        } else {
            Err(Error::transport(
                "LSP stream ended in the middle of a frame",
            ))
        }
    }
}

/// Parses the `Content-Length` header out of a header block.
fn parse_content_length(header: &str) -> Result<usize> {
    let mut length = None;
    for line in header.split("\r\n") {
        if line.is_empty() {
            continue;
        }
        let Some((name, value)) = line.split_once(':') else {
            return Err(Error::validation(format!(
                "malformed LSP header line: {line:?}"
            )));
        };
        if name.trim().eq_ignore_ascii_case(CONTENT_LENGTH) {
            let parsed = value
                .trim()
                .parse::<usize>()
                .map_err(|_| Error::validation(format!("invalid LSP Content-Length: {value:?}")))?;
            length = Some(parsed);
        }
    }
    length.ok_or_else(|| Error::validation("LSP frame is missing the Content-Length header"))
}

/// Returns the index of `needle` in `haystack`.
fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

/// An async reader that yields framed JSON-RPC messages.
#[derive(Debug)]
pub struct MessageReader<R> {
    reader: R,
    decoder: FrameDecoder,
}

impl<R> MessageReader<R>
where
    R: AsyncRead + Unpin + Send,
{
    /// Wraps `reader` in a framed message reader.
    #[must_use]
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            decoder: FrameDecoder::new(),
        }
    }

    /// Reads the next message, or `None` when the stream ends cleanly at a frame boundary.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Transport`] on read failure or a frame truncated by EOF, and
    /// [`Error::Validation`] on a malformed frame.
    pub async fn next(&mut self) -> Result<Option<Value>> {
        loop {
            if let Some(message) = self.decoder.try_next()? {
                return Ok(Some(message));
            }
            let mut buffer = [0_u8; 8192];
            let read =
                self.reader.read(&mut buffer).await.map_err(|error| {
                    Error::transport(format!("failed to read LSP frame: {error}"))
                })?;
            if read == 0 {
                return self.decoder.finish();
            }
            self.decoder.feed(&buffer[..read]);
        }
    }
}

/// A zero-based LSP text position.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema,
)]
pub struct Position {
    /// Zero-based line number.
    pub line: u32,
    /// Zero-based character offset (UTF-16 code units per the protocol default).
    pub character: u32,
}

impl Position {
    /// Builds a zero-based position.
    #[must_use]
    pub const fn new(line: u32, character: u32) -> Self {
        Self { line, character }
    }

    /// Converts the model-facing, one-based `line:column` into an LSP position.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `line` or `column` is zero.
    pub fn from_one_based(line: u32, column: u32) -> Result<Self> {
        if line == 0 || column == 0 {
            return Err(Error::validation(
                "line and column are 1-based and must be at least 1",
            ));
        }
        Ok(Self {
            line: line - 1,
            character: column - 1,
        })
    }

    /// Returns the one-based `(line, column)` an editor displays.
    #[must_use]
    pub const fn to_one_based(self) -> (u32, u32) {
        (self.line + 1, self.character + 1)
    }
}

/// A zero-based LSP range.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema,
)]
pub struct Range {
    /// Inclusive start position.
    pub start: Position,
    /// Exclusive end position.
    pub end: Position,
}

/// An LSP `Location`: a file URI plus a range.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Location {
    /// `file://` URI of the file.
    pub uri: String,
    /// Range inside the file.
    pub range: Range,
}

/// A normalised hover result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Hover {
    /// Markdown or plain text contents, already flattened to a single string.
    pub contents: String,
    /// Optional range the hover applies to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub range: Option<Range>,
}

/// One completion candidate.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(default, rename_all = "camelCase")]
pub struct CompletionItem {
    /// Text shown in the completion list.
    pub label: String,
    /// LSP `CompletionItemKind` number.
    pub kind: Option<u32>,
    /// Short detail line (signature, type).
    pub detail: Option<String>,
    /// Long documentation, flattened to text.
    pub documentation: Option<String>,
    /// Text inserted instead of `label` when present.
    pub insert_text: Option<String>,
}

/// One document symbol, covering both the hierarchical `DocumentSymbol` tree and the flat
/// `SymbolInformation` form.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct DocumentSymbol {
    /// Symbol name.
    pub name: String,
    /// LSP `SymbolKind` number.
    pub kind: u32,
    /// Signature detail, when the server provides one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    /// Full range of the symbol.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub range: Option<Range>,
    /// Range of the identifier itself.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selection_range: Option<Range>,
    /// `file://` URI, set for flat `SymbolInformation` results and absent for trees.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uri: Option<String>,
    /// Enclosing symbol name from flat `SymbolInformation` results.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub container_name: Option<String>,
    /// Nested symbols.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<DocumentSymbol>,
}

/// A flat `SymbolInformation` entry from `workspace/symbol`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SymbolInformation {
    /// Symbol name.
    pub name: String,
    /// LSP `SymbolKind` number.
    pub kind: u32,
    /// Location of the symbol.
    pub location: Location,
    /// Enclosing symbol name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub container_name: Option<String>,
}

/// Capabilities captured from the `initialize` result.
///
/// Only the providers the tool can dispatch to are decoded; [`ServerCapabilities::raw`]
/// keeps the complete object for diagnostics and future use.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ServerCapabilities {
    /// Supports `textDocument/hover`.
    pub hover: bool,
    /// Supports `textDocument/definition`.
    pub definition: bool,
    /// Supports `textDocument/references`.
    pub references: bool,
    /// Supports `textDocument/completion`.
    pub completion: bool,
    /// Supports `textDocument/documentSymbol`.
    pub document_symbol: bool,
    /// Supports `workspace/symbol`.
    pub workspace_symbol: bool,
    /// Supports `textDocument/implementation`.
    pub implementation: bool,
    /// Supports `textDocument/prepareCallHierarchy` and the `callHierarchy/*` methods.
    pub call_hierarchy: bool,
    /// Supports pull diagnostics (`textDocument/diagnostic`).
    pub diagnostics: bool,
    /// The raw `capabilities` object returned by the server.
    pub raw: Value,
}

impl ServerCapabilities {
    /// Extracts capabilities from an `initialize` result object.
    #[must_use]
    pub fn from_initialize_result(result: &Value) -> Self {
        let raw = result.get("capabilities").cloned().unwrap_or(Value::Null);
        let provider = |name: &str| {
            raw.get(name)
                .is_some_and(|value| value.as_bool().unwrap_or(!value.is_null()))
        };
        Self {
            hover: provider("hoverProvider"),
            definition: provider("definitionProvider"),
            references: provider("referencesProvider"),
            completion: provider("completionProvider"),
            document_symbol: provider("documentSymbolProvider"),
            workspace_symbol: provider("workspaceSymbolProvider"),
            implementation: provider("implementationProvider"),
            call_hierarchy: provider("callHierarchyProvider"),
            diagnostics: provider("diagnosticProvider"),
            raw,
        }
    }

    /// Returns whether the server advertises `method`.
    ///
    /// Unknown methods report `true`: absence of a capability entry is not proof that a
    /// server rejects the method, and the client maps a `-32601` response to a clear error.
    #[must_use]
    pub fn supports(&self, method: &str) -> bool {
        match method {
            "textDocument/hover" => self.hover,
            "textDocument/definition" => self.definition,
            "textDocument/references" => self.references,
            "textDocument/completion" => self.completion,
            "textDocument/documentSymbol" => self.document_symbol,
            "workspace/symbol" => self.workspace_symbol,
            "textDocument/implementation" => self.implementation,
            "textDocument/prepareCallHierarchy"
            | "callHierarchy/incomingCalls"
            | "callHierarchy/outgoingCalls" => self.call_hierarchy,
            "textDocument/diagnostic" => self.diagnostics,
            _ => true,
        }
    }
}

/// The client capabilities the harness advertises during `initialize`.
#[must_use]
pub fn client_capabilities() -> Value {
    json!({
        "window": { "workDoneProgress": true },
        "workspace": {
            "applyEdit": true,
            "workspaceEdit": { "documentChanges": true },
            "workspaceFolders": true,
            "configuration": true,
        },
        "textDocument": {
            "synchronization": {
                "didOpen": true,
                "didChange": { "willSave": true, "willSaveWaitUntil": true, "save": true },
                "didSave": true,
            },
            "publishDiagnostics": { "versionSupport": true },
            "hover": { "contentFormat": ["markdown", "plaintext"] },
            "completion": { "completionItem": { "snippetSupport": false } },
            "documentSymbol": { "hierarchicalDocumentSymbolSupport": true },
            "implementation": { "dynamicRegistration": false },
            "callHierarchy": { "dynamicRegistration": false },
            "diagnostic": {
                "dynamicRegistration": false,
                "relatedDocumentSupport": true,
                "versionSupport": true,
            },
        },
    })
}

/// Normalises a hover result into [`Hover`].
///
/// Returns `None` for `null`, a missing `contents` field, and empty contents.
#[must_use]
pub fn parse_hover(value: &Value) -> Option<Hover> {
    let contents = hover_text(value.get("contents")?)?;
    if contents.trim().is_empty() {
        return None;
    }
    let range = value
        .get("range")
        .and_then(|range| serde_json::from_value(range.clone()).ok());
    Some(Hover { contents, range })
}

/// Flattens any LSP hover-contents shape into text.
fn hover_text(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::Array(items) => {
            let parts: Vec<String> = items.iter().filter_map(hover_text).collect();
            if parts.is_empty() {
                None
            } else {
                Some(parts.join("\n"))
            }
        }
        Value::Object(map) => map
            .get("value")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        _ => None,
    }
}

/// Normalises a definition/implementation result into locations.
///
/// Accepts `null`, a single `Location`, a `LocationLink` (using `targetRange`), and an array
/// of either.
#[must_use]
pub fn parse_locations(value: &Value) -> Vec<Location> {
    match value {
        Value::Array(items) => items.iter().filter_map(location_from_value).collect(),
        _ => location_from_value(value).into_iter().collect(),
    }
}

/// Parses one `Location` or `LocationLink`.
fn location_from_value(value: &Value) -> Option<Location> {
    let uri = value
        .get("uri")
        .or_else(|| value.get("targetUri"))
        .and_then(Value::as_str)?;
    let range = value
        .get("range")
        .or_else(|| value.get("targetRange"))
        .or_else(|| value.get("targetSelectionRange"))?;
    let range: Range = serde_json::from_value(range.clone()).ok()?;
    Some(Location {
        uri: uri.to_owned(),
        range,
    })
}

/// Normalises a completion result (a `CompletionItem[]` or a `CompletionList`).
#[must_use]
pub fn parse_completion_items(value: &Value) -> Vec<CompletionItem> {
    let items = match value {
        Value::Array(items) => items.as_slice(),
        Value::Object(map) => map
            .get("items")
            .and_then(Value::as_array)
            .map_or(&[][..], Vec::as_slice),
        _ => return Vec::new(),
    };
    items.iter().filter_map(completion_item).collect()
}

/// Parses one completion item, tolerating object and string documentation.
fn completion_item(value: &Value) -> Option<CompletionItem> {
    let label = value.get("label").and_then(Value::as_str)?.to_owned();
    let kind = value.get("kind").and_then(Value::as_u64).and_then(as_u32);
    let detail = value
        .get("detail")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    let documentation = value.get("documentation").and_then(hover_text);
    let insert_text = value
        .get("insertText")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    Some(CompletionItem {
        label,
        kind,
        detail,
        documentation,
        insert_text,
    })
}

/// Normalises document symbols from either a `DocumentSymbol` tree or flat
/// `SymbolInformation` entries.
#[must_use]
pub fn parse_document_symbols(value: &Value) -> Vec<DocumentSymbol> {
    let Some(items) = value.as_array() else {
        return Vec::new();
    };
    let is_tree = items.first().is_some_and(|first| {
        first.get("selectionRange").is_some() || first.get("children").is_some()
    });
    if is_tree {
        items.iter().filter_map(document_symbol).collect()
    } else {
        items
            .iter()
            .filter_map(symbol_information_as_document)
            .collect()
    }
}

/// Parses one hierarchical `DocumentSymbol`, recursively.
fn document_symbol(value: &Value) -> Option<DocumentSymbol> {
    let name = value.get("name").and_then(Value::as_str)?.to_owned();
    let kind = value.get("kind").and_then(Value::as_u64).and_then(as_u32)?;
    let detail = value
        .get("detail")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    let range = value
        .get("range")
        .and_then(|range| serde_json::from_value(range.clone()).ok());
    let selection_range = value
        .get("selectionRange")
        .and_then(|range| serde_json::from_value(range.clone()).ok());
    let children = value
        .get("children")
        .and_then(Value::as_array)
        .map(|children| children.iter().filter_map(document_symbol).collect())
        .unwrap_or_default();
    Some(DocumentSymbol {
        name,
        kind,
        detail,
        range,
        selection_range,
        uri: None,
        container_name: value
            .get("containerName")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        children,
    })
}

/// Converts one flat `SymbolInformation` entry into a [`DocumentSymbol`].
fn symbol_information_as_document(value: &Value) -> Option<DocumentSymbol> {
    let name = value.get("name").and_then(Value::as_str)?.to_owned();
    let kind = value.get("kind").and_then(Value::as_u64).and_then(as_u32)?;
    let location = value.get("location")?;
    let uri = location.get("uri").and_then(Value::as_str)?.to_owned();
    let range = location
        .get("range")
        .and_then(|range| serde_json::from_value(range.clone()).ok());
    Some(DocumentSymbol {
        name,
        kind,
        detail: None,
        range,
        selection_range: range,
        uri: Some(uri),
        container_name: value
            .get("containerName")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        children: Vec::new(),
    })
}

/// Normalises a `workspace/symbol` result into [`SymbolInformation`] entries.
#[must_use]
pub fn parse_workspace_symbols(value: &Value) -> Vec<SymbolInformation> {
    let Some(items) = value.as_array() else {
        return Vec::new();
    };
    items
        .iter()
        .filter_map(|item| {
            let name = item.get("name").and_then(Value::as_str)?.to_owned();
            let kind = item.get("kind").and_then(Value::as_u64).and_then(as_u32)?;
            let location = location_from_value(item.get("location")?)?;
            Some(SymbolInformation {
                name,
                kind,
                location,
                container_name: item
                    .get("containerName")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned),
            })
        })
        .collect()
}

/// Narrows a JSON integer to `u32`.
fn as_u32(value: u64) -> Option<u32> {
    u32::try_from(value).ok()
}
