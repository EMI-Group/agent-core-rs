//! Action execution for the [`LspTool`](super::LspTool).
//!
//! Every action resolves a server first, then runs through the per-key
//! [`ServerSupervisor`], which starts the server lazily, restarts it once after a crash and
//! retries the operation. Failures are string messages that the tool turns into a
//! `success=false` payload: `invoke` never returns `Err` for model input.
//!
//! Diagnostics are best effort: push data buffered by the
//! [`DiagnosticRegistry`](super::DiagnosticRegistry) wins, an empty buffer falls back to a
//! `textDocument/diagnostic` pull (when the server advertises it), and "nothing anywhere" is a
//! success payload with `source: "none"` rather than an error.
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::{json, Value};

use openjiuwen_common::error::Result;
use openjiuwen_foundation::tool::ToolContext;
use openjiuwen_sysop::workspace::{canonicalize_containing, is_within};

use crate::tools::ToolPayload;

use super::client::LspClient;
use super::protocol::{CompletionItem, DocumentSymbol, Hover, Location, Position};
use super::render;
use super::tool::{map_client_error, LspAction, LspArgs, LspTool, NO_SERVER_CONFIGURED};
use super::uri::{path_to_uri, uri_to_path};
use super::wire::{
    CallHierarchyIncomingCall, CallHierarchyItem, CallHierarchyOutgoingCall, Diagnostic,
    DiagnosticReportKind, DocumentDiagnosticReport,
};

/// Largest `context_lines` value honoured, keeping excerpts short.
const MAX_CONTEXT_LINES: u32 = 20;
/// Largest number of definition locations that get a source excerpt.
const MAX_CONTEXT_LOCATIONS: usize = 5;

/// Whether the tool failed on model input or the server answered with an error.
type ToolResult<T> = std::result::Result<T, String>;

/// The request behind one path-based action.
#[derive(Debug, Clone, Copy)]
enum PositionOp {
    /// `textDocument/hover`.
    Hover,
    /// `textDocument/definition`.
    Definition,
    /// `textDocument/references`.
    References,
    /// `textDocument/completion`.
    Completion,
    /// `textDocument/documentSymbol`.
    Symbols,
    /// `textDocument/implementation`.
    Implementation,
    /// `textDocument/prepareCallHierarchy`.
    PrepareCallHierarchy,
    /// `prepareCallHierarchy` followed by `callHierarchy/incomingCalls`.
    IncomingCalls,
    /// `prepareCallHierarchy` followed by `callHierarchy/outgoingCalls`.
    OutgoingCalls,
}

/// What one path-based request produced before it is turned into a payload.
enum PositionOutcome {
    /// A hover result.
    Hover(Option<Hover>),
    /// Definition, reference or implementation locations.
    Locations(Vec<Location>),
    /// Completion candidates.
    Completion(Vec<CompletionItem>),
    /// Document symbols.
    Symbols(Vec<DocumentSymbol>),
    /// Call-hierarchy items from `prepareCallHierarchy`.
    CallItems(Vec<CallHierarchyItem>),
    /// Incoming calls of the prepared item.
    IncomingCalls {
        /// The prepared item the server answered for.
        item: CallHierarchyItem,
        /// The incoming calls.
        calls: Vec<CallHierarchyIncomingCall>,
    },
    /// Outgoing calls of the prepared item.
    OutgoingCalls {
        /// The prepared item the server answered for.
        item: CallHierarchyItem,
        /// The outgoing calls.
        calls: Vec<CallHierarchyOutgoingCall>,
    },
    /// `prepareCallHierarchy` returned no item for the position.
    NoCallItem,
}

/// The result of the diagnostics pull path.
enum PullOutcome {
    /// The server does not advertise `textDocument/diagnostic`.
    Unsupported,
    /// The server answered with a report.
    Report(DocumentDiagnosticReport),
}

impl LspTool {
    /// Runs one tool call, converting every failure into a `success=false` payload.
    pub(super) async fn run(&self, inputs: Value, ctx: &ToolContext) -> ToolPayload {
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

    /// Dispatches one validated call to its action implementation.
    async fn execute(&self, args: LspArgs) -> ToolResult<ToolPayload> {
        match args.action {
            LspAction::Diagnostics => self.diagnostics_action(&args).await,
            LspAction::WorkspaceSymbols => self.workspace_symbols_action(&args).await,
            action => self.position_action(action, &args).await,
        }
    }

    /// Runs `workspace_symbols`, which needs no document to synchronize.
    async fn workspace_symbols_action(&self, args: &LspArgs) -> ToolResult<ToolPayload> {
        let resolved = self
            .resolve_server(None)
            .await
            .map_err(|error| error.message().into_owned())?;
        let supervisor = self
            .supervisor(&resolved)
            .map_err(|error| error.message().into_owned())?;
        let query = args.query.clone().unwrap_or_default();
        let symbols = supervisor
            .run(move |client| {
                let query = query.clone();
                async move { client.workspace_symbols(&query).await }
            })
            .await
            .map_err(|error| map_client_error(&error, LspAction::WorkspaceSymbols.method()))?;
        let result = render::render_workspace_symbols(&symbols);
        Ok(ToolPayload::ok(json!({
            "action": LspAction::WorkspaceSymbols.as_str(),
            "result": result,
            "symbols": symbols,
        })))
    }

    /// Runs one path-based action after resolving the document and its server.
    async fn position_action(&self, action: LspAction, args: &LspArgs) -> ToolResult<ToolPayload> {
        let Some(operation) = operation_for(action) else {
            return Err(format!("Unsupported action '{}'.", action.as_str()));
        };
        if !self.has_server_configuration() {
            // Preserve the pre-discovery precedence: a tool with no server at all reports that
            // before it inspects the path.
            return Err(NO_SERVER_CONFIGURED.to_owned());
        }
        let (path, uri, source) = self.resolve_document(args).await?;
        let resolved = self
            .resolve_server(Some(&path))
            .await
            .map_err(|error| error.message().into_owned())?;
        let supervisor = self
            .supervisor(&resolved)
            .map_err(|error| error.message().into_owned())?;
        let position = Position::from_one_based(args.line.unwrap_or(1), args.column.unwrap_or(1))
            .map_err(|error| error.message().into_owned())?;
        let language_id = self.language_for(&resolved);
        let file_path = path.to_string_lossy().into_owned();
        let outcome = supervisor
            .run(move |client| {
                let uri = uri.clone();
                let language_id = language_id.clone();
                let source = source.clone();
                async move {
                    synchronize_and_run(client, operation, &uri, &language_id, &source, position)
                        .await
                }
            })
            .await
            .map_err(|error| map_client_error(&error, action.method()))?;
        self.outcome_payload(action, outcome, file_path, args).await
    }

    /// Turns one [`PositionOutcome`] into the action's payload.
    async fn outcome_payload(
        &self,
        action: LspAction,
        outcome: PositionOutcome,
        file_path: String,
        args: &LspArgs,
    ) -> ToolResult<ToolPayload> {
        match outcome {
            PositionOutcome::Hover(hover) => Ok(ToolPayload::ok(json!({
                "action": action.as_str(),
                "file_path": file_path,
                "result": render::render_hover(hover.as_ref()),
                "hover": hover,
            }))),
            PositionOutcome::Completion(items) => Ok(ToolPayload::ok(json!({
                "action": action.as_str(),
                "file_path": file_path,
                "result": render::render_completion(&items),
                "items": items,
            }))),
            PositionOutcome::Symbols(symbols) => Ok(ToolPayload::ok(json!({
                "action": action.as_str(),
                "file_path": file_path,
                "result": render::render_symbols(&symbols),
                "symbols": symbols,
            }))),
            PositionOutcome::Locations(locations) => {
                if action == LspAction::References {
                    return Ok(ToolPayload::ok(json!({
                        "action": action.as_str(),
                        "file_path": file_path,
                        "result": render::render_references(&locations),
                        "locations": locations,
                    })));
                }
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
                    "action": action.as_str(),
                    "file_path": file_path,
                    "result": result,
                    "locations": locations,
                })))
            }
            PositionOutcome::CallItems(items) => Ok(ToolPayload::ok(json!({
                "action": action.as_str(),
                "file_path": file_path,
                "result": render::render_call_hierarchy_items(&items),
                "items": items,
            }))),
            PositionOutcome::IncomingCalls { item, calls } => Ok(ToolPayload::ok(json!({
                "action": action.as_str(),
                "file_path": file_path,
                "result": render::render_incoming_calls(&calls),
                "calls": calls,
                "item": item,
            }))),
            PositionOutcome::OutgoingCalls { item, calls } => Ok(ToolPayload::ok(json!({
                "action": action.as_str(),
                "file_path": file_path,
                "result": render::render_outgoing_calls(&calls),
                "calls": calls,
                "item": item,
            }))),
            PositionOutcome::NoCallItem => Ok(ToolPayload::ok(json!({
                "action": action.as_str(),
                "file_path": file_path,
                "result": render::render_call_hierarchy_items(&[]),
                "calls": Vec::<CallHierarchyIncomingCall>::new(),
                "item": Value::Null,
            }))),
        }
    }

    /// Answers a `diagnostics` call: push buffer first, then a pull, then "none".
    async fn diagnostics_action(&self, args: &LspArgs) -> ToolResult<ToolPayload> {
        let Some(raw) = args.path.as_deref().filter(|path| !path.is_empty()) else {
            return Ok(self.drain_diagnostics_payload());
        };
        let path = self.resolve_path(raw).await?;
        let uri = path_to_uri(&path);
        let file_path = path.to_string_lossy().into_owned();
        if let Some(diagnostics) = self.diagnostics.snapshot(&uri) {
            if !diagnostics.is_empty() {
                return Ok(diagnostics_payload(&file_path, &diagnostics, "push"));
            }
        }
        self.pull_diagnostics(&path, &uri, file_path).await
    }

    /// Drains every buffered document into the no-path diagnostics payload.
    fn drain_diagnostics_payload(&self) -> ToolPayload {
        let files = self.diagnostics.drain();
        let count: usize = files
            .iter()
            .map(|file| file.diagnostics.len())
            .sum::<usize>();
        let result = render::render_diagnostic_files(&files);
        ToolPayload::ok(json!({
            "action": LspAction::Diagnostics.as_str(),
            "files": files,
            "result": result,
            "count": count,
        }))
    }

    /// Pulls diagnostics for one file when no push data is buffered.
    async fn pull_diagnostics(
        &self,
        path: &Path,
        uri: &str,
        file_path: String,
    ) -> ToolResult<ToolPayload> {
        let Ok(resolved) = self.resolve_server(Some(path)).await else {
            return Ok(no_diagnostics_payload(&file_path));
        };
        let Ok(supervisor) = self.supervisor(&resolved) else {
            return Ok(no_diagnostics_payload(&file_path));
        };
        let source = self.read_source(path).await?;
        let language_id = self.language_for(&resolved);
        let document = uri.to_owned();
        let previous = self.pull_result_ids.lock().get(uri).cloned();
        let outcome = supervisor
            .run(move |client| {
                let document = document.clone();
                let language_id = language_id.clone();
                let source = source.clone();
                let previous = previous.clone();
                async move {
                    client.did_open(&document, &language_id, &source).await?;
                    let supported = client.capabilities().is_some_and(|capabilities| {
                        capabilities.supports(LspAction::Diagnostics.method())
                    });
                    if !supported {
                        return Ok(PullOutcome::Unsupported);
                    }
                    let report = client
                        .document_diagnostic(&document, previous.as_deref())
                        .await?;
                    Ok(PullOutcome::Report(report))
                }
            })
            .await;
        let outcome = match outcome {
            Ok(outcome) => outcome,
            Err(error) => {
                if error.message().contains("-32601") {
                    return Ok(no_diagnostics_payload(&file_path));
                }
                return Err(map_client_error(&error, LspAction::Diagnostics.method()));
            }
        };
        match outcome {
            PullOutcome::Unsupported => Ok(no_diagnostics_payload(&file_path)),
            PullOutcome::Report(report) => {
                let diagnostics = match report.kind {
                    DiagnosticReportKind::Full => {
                        if let Some(result_id) = report.result_id.clone() {
                            self.pull_result_ids
                                .lock()
                                .insert(uri.to_owned(), result_id);
                        }
                        self.pull_items
                            .lock()
                            .insert(uri.to_owned(), report.items.clone());
                        report.items
                    }
                    DiagnosticReportKind::Unchanged => {
                        self.pull_items.lock().get(uri).cloned().unwrap_or_default()
                    }
                };
                Ok(diagnostics_payload(&file_path, &diagnostics, "pull"))
            }
        }
    }

    /// Resolves the `path` argument and reads the document for synchronization.
    async fn resolve_document(&self, args: &LspArgs) -> ToolResult<(PathBuf, String, String)> {
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
    async fn resolve_path(&self, raw: &str) -> ToolResult<PathBuf> {
        let root = self.workspace_index().await?.root.clone();
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
    async fn read_source(&self, path: &Path) -> ToolResult<String> {
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

/// The request behind `action`, or `None` for actions without a document position.
fn operation_for(action: LspAction) -> Option<PositionOp> {
    match action {
        LspAction::Hover => Some(PositionOp::Hover),
        LspAction::Definition => Some(PositionOp::Definition),
        LspAction::References => Some(PositionOp::References),
        LspAction::Completion => Some(PositionOp::Completion),
        LspAction::Symbols => Some(PositionOp::Symbols),
        LspAction::Implementation => Some(PositionOp::Implementation),
        LspAction::PrepareCallHierarchy => Some(PositionOp::PrepareCallHierarchy),
        LspAction::IncomingCalls => Some(PositionOp::IncomingCalls),
        LspAction::OutgoingCalls => Some(PositionOp::OutgoingCalls),
        LspAction::WorkspaceSymbols | LspAction::Diagnostics => None,
    }
}

/// Synchronizes the document, then performs one navigation request.
async fn synchronize_and_run(
    client: Arc<LspClient>,
    operation: PositionOp,
    uri: &str,
    language_id: &str,
    source: &str,
    position: Position,
) -> Result<PositionOutcome> {
    client.did_open(uri, language_id, source).await?;
    match operation {
        PositionOp::Hover => Ok(PositionOutcome::Hover(client.hover(uri, position).await?)),
        PositionOp::Definition => Ok(PositionOutcome::Locations(
            client.definition(uri, position).await?,
        )),
        PositionOp::References => Ok(PositionOutcome::Locations(
            client.references(uri, position, true).await?,
        )),
        PositionOp::Completion => Ok(PositionOutcome::Completion(
            client.completion(uri, position).await?,
        )),
        PositionOp::Symbols => Ok(PositionOutcome::Symbols(
            client.document_symbols(uri).await?,
        )),
        PositionOp::Implementation => Ok(PositionOutcome::Locations(
            client.implementation(uri, position).await?,
        )),
        PositionOp::PrepareCallHierarchy => Ok(PositionOutcome::CallItems(
            client.prepare_call_hierarchy(uri, position).await?,
        )),
        PositionOp::IncomingCalls => {
            let items = client.prepare_call_hierarchy(uri, position).await?;
            let Some(item) = items.into_iter().next() else {
                return Ok(PositionOutcome::NoCallItem);
            };
            let calls = client.incoming_calls(&item).await?;
            Ok(PositionOutcome::IncomingCalls { item, calls })
        }
        PositionOp::OutgoingCalls => {
            let items = client.prepare_call_hierarchy(uri, position).await?;
            let Some(item) = items.into_iter().next() else {
                return Ok(PositionOutcome::NoCallItem);
            };
            let calls = client.outgoing_calls(&item).await?;
            Ok(PositionOutcome::OutgoingCalls { item, calls })
        }
    }
}

/// Builds a payload carrying diagnostics found through `source`.
fn diagnostics_payload(file_path: &str, diagnostics: &[Diagnostic], source: &str) -> ToolPayload {
    ToolPayload::ok(json!({
        "action": LspAction::Diagnostics.as_str(),
        "file_path": file_path,
        "result": render::render_diagnostics(diagnostics),
        "diagnostics": diagnostics,
        "source": source,
    }))
}

/// Builds the success payload used when no diagnostics of any kind are available.
fn no_diagnostics_payload(file_path: &str) -> ToolPayload {
    diagnostics_payload(file_path, &[], "none")
}
