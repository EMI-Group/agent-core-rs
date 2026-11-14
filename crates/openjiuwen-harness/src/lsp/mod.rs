//! Language-server client and the model-facing `lsp` tool.
//!
//! Ports `openjiuwen/harness/lsp/` and `openjiuwen/harness/tools/lsp_tool/` as an offline,
//! testable subset: JSON-RPC 2.0 framing, a transport seam with a process and an in-memory
//! implementation, a request/response client, normalised LSP result types, and the `lsp` tool
//! that renders short navigation text for the model.
//!
//! # Layering
//!
//! `protocol` owns framing and the navigation wire types, `wire` owns the pull-diagnostics and
//! call-hierarchy types plus their tolerant parsers, `uri` converts `file://` URIs,
//! `transport` owns the [`LspTransport`] seam, `config` holds [`LspServerConfig`],
//! `client` correlates requests, `render` formats results, and `tool` adapts all of it
//! to the [`Tool`](openjiuwen_foundation::tool::Tool) trait.
//!
//! # Offline testing
//!
//! No test needs a real language server: [`ChannelTransport`] runs an in-process fake server
//! over [`tokio::io::duplex`], and the process transport is exercised with `/bin/cat` when it
//! exists. See `tests/fake_server.rs`.
//!
//! # Deferred
//!
//! The `lsp` tool still reports `diagnostics`, `implementation` and call-hierarchy actions as
//! deferred, and there is no diagnostics registry, per-extension server discovery or crash
//! retry yet. The client and `wire` layer they need are in place.
mod client;
mod config;
pub mod protocol;
pub mod render;
mod tool;
mod transport;
mod uri;
mod wire;

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;

pub use client::LspClient;
pub use config::LspServerConfig;
pub use protocol::{
    client_capabilities, parse_completion_items, parse_document_symbols, parse_hover,
    parse_locations, parse_workspace_symbols, CompletionItem, DocumentSymbol, FrameDecoder, Hover,
    Location, LspError, MessageReader, Position, Range, ServerCapabilities, SymbolInformation,
    MAX_CONTENT_LENGTH,
};
pub use render::symbol_kind_name;
pub use tool::{LspAction, LspArgs, LspTool, LspToolConfig};
pub use transport::{ChannelTransport, LspTransport, StdioTransport};
pub use uri::{path_to_uri, uri_to_path};
pub use wire::{
    parse_call_hierarchy_incoming_calls, parse_call_hierarchy_items,
    parse_call_hierarchy_outgoing_calls, parse_diagnostics, parse_diagnostics_event,
    parse_document_diagnostic_report, CallHierarchyIncomingCall, CallHierarchyItem,
    CallHierarchyOutgoingCall, Diagnostic, DiagnosticCode, DiagnosticRelatedInformation,
    DiagnosticReportKind, DiagnosticsEvent, DocumentDiagnosticReport,
};
