# openjiuwen-harness::lsp

## Intent

A language-server protocol (LSP) client plus the model-facing `lsp` tool, ported from `openjiuwen/harness/lsp/` and `openjiuwen/harness/tools/lsp_tool/`.
The subset is offline-testable: JSON-RPC framing, a transport seam with process and in-memory implementations, request/response correlation with timeouts, normalised result types, short text rendering, and a `Tool` that never panics and never returns `Err` on model input.
Python parity target: tool name `lsp`, 1-based model positions, `Defined in path:line:col`/grouped-reference formatting, lazy server start per workspace root, and `success=false` payloads for every failure.

## API Surface

- `LspClient` — `new(transport, request_timeout)`, `initialize(root_uri, capabilities)`, `did_open`, `did_change`, `hover`, `definition`, `references`, `completion`, `document_symbols`, `workspace_symbols`, `request`, `notify`, `shutdown`, `capabilities`, `is_initialized`.
- `LspTransport` — `send(Value)`, `recv() -> Option<Value>`, `close()`; `StdioTransport::spawn(&LspServerConfig)` and `ChannelTransport::new(buffer) -> (Self, DuplexStream)`.
- `LspServerConfig` — serde descriptor `{ command, args, env, root, language_id, startup_timeout }` with builder setters.
- `LspTool` / `LspToolConfig` — `LspTool::new(config)`, `LspTool::card_for(language)`, `with_transport(Box<dyn LspTransport>)` for the offline seam.
- `LspArgs` / `LspAction` — `action` (`hover`, `definition`, `references`, `completion`, `symbols`, `workspace_symbols`, `diagnostics`), `path`, `line`, `column`, `query`, `context_lines`.
- Wire types — `Position`, `Range`, `Location`, `Hover`, `CompletionItem`, `DocumentSymbol`, `SymbolInformation`, `ServerCapabilities`, `LspError`, all serde + schemars.
- Framing — `encode_message`, `write_message`, `FrameDecoder`, `MessageReader`, `MAX_CONTENT_LENGTH` (8 MiB), `request_message`/`notification_message`/`response_message`/`error_response`, `client_capabilities`.
- Rendering — `render::{symbol_kind_name, render_hover, render_definition, render_references, render_completion, render_symbols, render_workspace_symbols, format_context_excerpt, display_uri, format_location}`.
- URIs — `path_to_uri`, `uri_to_path`.

## Constraints

- Errors are `openjiuwen_common::error::{Error, Result}`; malformed frames, EOF, oversize frames, timeouts and JSON-RPC errors are typed.
- `LspTool::invoke` always returns `Ok(payload)`; failures set `success=false` and a human-readable `error`.
- No `unsafe`, no `println!` (tracing only), no locks held across `.await`; the client's I/O task owns the transport and uses `tokio::select!` over an outgoing command queue and incoming frames.
- Paths resolve inside `LspToolConfig::workspace_root` only; `..` and absolute paths outside the root are rejected before any read.
- Files are synchronized before a request (`didOpen`, then full-text `didChange` with monotonic versions) and reads are gated by `max_file_size` (10 MiB default).
- No new dependencies beyond the crate manifest (tokio, serde/serde_json, schemars, async-trait, parking_lot, tracing, sysop path helpers).
- Deferred: diagnostics (push registry and pull requests), call-hierarchy/implementation/workspace-symbol caching, server auto-discovery per extension, crash retry, gitignore filtering.

## Fake-server test recipe

- `tests/fake_server.rs` spawns a task over one `tokio::io::duplex` half that reads frames with `MessageReader` and answers canned hover/definition/references/completion/documentSymbol/workspaceSymbol results.
- Build a client with `LspClient::new(Box::new(ChannelTransport::new(64 * 1024).0), timeout)` and pass the peer stream to `fake_server::spawn`.
- Build a tool with `LspTool::new(LspToolConfig::default().with_workspace_root(dir)).with_transport(Box::new(transport))`.
- The fake sends `workspace/configuration` during `initialize`, so capability capture also proves server-initiated request handling.
- A definition request whose URI contains `slow` sleeps 300 ms, exercising the per-request timeout; a completion request whose URI contains `unsupported` returns `-32601`.
- The process transport is tested with `/bin/cat` as an echo server when it exists (no real LSP binary required); missing binaries are asserted through `StdioTransport::spawn` errors.

## Routing Table

- `./protocol.rs` → JSON-RPC framing, message builders, LSP wire types, result normalisation, server/client capabilities.
- `./uri.rs` → `file://` URI encoding/decoding.
- `./transport.rs` → `LspTransport`, `StdioTransport` (process spawn, stderr drain, kill on drop), `ChannelTransport` (in-memory duplex).
- `./client.rs` → single-transport I/O task, request correlation, timeouts, document versions, typed methods, graceful shutdown.
- `./config.rs` → `LspServerConfig` and timeout/size defaults.
- `./render.rs` → one-based text rendering for every action, symbol/kind maps, context excerpts.
- `./tool.rs` → `LspTool`, `LspToolConfig`, `LspArgs`, schema/card construction, path safety, lazy client, payload envelopes.
- `./tests/` → offline suites: framing fixtures, URI conversion, transport (channel + `/bin/cat`), fake server, client behaviour, tool behaviour.
