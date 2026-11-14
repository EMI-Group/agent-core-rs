//! Language-server client and the model-facing `lsp` tool.
//!
//! Ports `openjiuwen/harness/lsp/` and `openjiuwen/harness/tools/lsp_tool/` as an offline,
//! testable subset: JSON-RPC 2.0 framing, a transport seam with a process and an in-memory
//! implementation, a request/response client, normalised LSP result types, and the `lsp` tool
//! that renders short navigation text for the model.
//!
//! # Layering
//!
//! `protocol` owns framing and the navigation wire types, `wire` owns the push/pull diagnostics
//! and call-hierarchy types plus their tolerant parsers, `uri` converts `file://` URIs,
//! `transport` owns the [`LspTransport`] seam, `config` holds [`LspServerConfig`], `client`
//! correlates requests, `diagnostics` buffers push diagnostics, `discovery` maps files to
//! servers, `instance` supervises one server's lifecycle, `render` formats results, and `tool`
//! adapts all of it to the [`Tool`](openjiuwen_foundation::tool::Tool) trait.
//!
//! # Offline testing
//!
//! No test needs a real language server: [`ChannelTransport`] runs an in-process fake server
//! over [`tokio::io::duplex`], the process transport is exercised with `/bin/cat` when it
//! exists, discovery tests use temp directories, and crash recovery runs on an injected
//! [`InstantTimer`]. See `tests/fake_server.rs`.
//!
//! # Deferred
//!
//! The `lsp` tool still reports `diagnostics`, `implementation` and call-hierarchy actions as
//! deferred until it owns a [`DiagnosticRegistry`] and a [`ServerSupervisor`]; the client,
//! `wire`, diagnostics, discovery and supervision layers are in place.
mod client;
mod config;
mod diagnostics;
mod discovery;
mod instance;
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
pub use diagnostics::{DiagnosticRegistry, DiagnosticsChange, DiagnosticsConfig, DiagnosticsFile};
pub use discovery::{builtin_server_configs, DiscoveryIndex, ResolvedServer, ServerKey};
pub use instance::{
    BackoffTimer, InstantTimer, RestartPolicy, ServerFactory, ServerState, ServerStatus,
    ServerSupervisor, TokioTimer,
};
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
