# openjiuwen-harness::lsp

## Intent

A language-server protocol (LSP) client plus the model-facing `lsp` tool, ported from `openjiuwen/harness/lsp/` and `openjiuwen/harness/tools/lsp_tool/`.
The subset is offline-testable: JSON-RPC framing, a transport seam with process and in-memory implementations, request/response correlation with timeouts, push/pull diagnostics, call hierarchy, per-extension server discovery, crash recovery, normalised result types, short text rendering, and a `Tool` that never panics and never returns `Err` on model input.
Python parity target: tool name `lsp`, 1-based model positions, `Defined in path:line:col`/grouped-reference formatting, `publishDiagnostics` buffering with dedup/caps/severity order, call-hierarchy and implementation text formats, lazy server start per file/root, and `success=false` payloads for every failure.

## API Surface

- `LspClient` — `new(transport, request_timeout)`, `initialize(root_uri, capabilities)`, `did_open`, `did_change`, `hover`, `definition`, `references`, `completion`, `document_symbols`, `workspace_symbols`, `implementation`, `prepare_call_hierarchy`, `incoming_calls`, `outgoing_calls`, `document_diagnostic(uri, previous_result_id)`, `request`, `notify`, `add_notification_handler`, `is_alive`, `closed`, `close_reason`, `shutdown`, `capabilities`, `is_initialized`.
- `LspTransport` — `send(Value)`, `recv() -> Option<Value>`, `close()`; `StdioTransport::spawn(&LspServerConfig)` and `ChannelTransport::new(buffer) -> (Self, DuplexStream)`.
- `LspServerConfig` — serde descriptor `{ name, command, args, env, root, language_id, extensions, priority, root_markers, startup_timeout }` with builder setters.
- `wire` types (all serde + `schemars::JsonSchema`) — `Diagnostic`, `DiagnosticCode` (string or number), `DiagnosticRelatedInformation`, `DiagnosticsEvent`, `DiagnosticReportKind`, `DocumentDiagnosticReport`, `CallHierarchyItem` (flat URI-based), `CallHierarchyIncomingCall`, `CallHierarchyOutgoingCall`.
- `wire` parsers — `parse_diagnostics`, `parse_diagnostics_event`, `parse_document_diagnostic_report`, `parse_call_hierarchy_items`, `parse_call_hierarchy_incoming_calls`, `parse_call_hierarchy_outgoing_calls`; all tolerate null/malformed input by returning empty/`None`.
- `protocol` — existing framing (`FrameDecoder`, `MessageReader`, `write_message`, `MAX_CONTENT_LENGTH`), message builders, `Position`/`Range`/`Location`, `Hover`, `CompletionItem`, `DocumentSymbol`, `SymbolInformation`, `LspError` (+ `server_error(&Error) -> Option<&LspError>`, `is_method_not_found(&Error)`), `client_capabilities`, `ServerCapabilities` (`hover`, `definition`, `references`, `completion`, `document_symbol`, `workspace_symbol`, `implementation`, `call_hierarchy`, `diagnostics`, `raw`), `parse_*` normalisers.
- `DiagnosticRegistry` — `new(DiagnosticsConfig)`, `publish(DiagnosticsEvent)`, `snapshot(uri)`, `drain()`, `pending_count`, `clear`, `subscribe() -> broadcast::Receiver<DiagnosticsChange>`, `attach(&Arc<Self>, &LspClient)`.
- `DiagnosticsConfig` — `max_per_file` (10), `max_total` (30), `max_files` (256, bounds buffered documents *and* their delivered-key history), `channel_capacity` (16).
- `DiagnosticsFile` — `{ uri, version, diagnostics, local_path }`.
- `DiagnosticsChange` — `Published(DiagnosticsEvent)` or `Cleared { uri }`.
- `DiscoveryIndex` — `new(workspace_root, servers)`, `resolve(path)`, `resolve_by_language(language_id)`, `candidates_for_extension(ext)`, `servers`, `is_empty`.
- `ResolvedServer` — `{ key: ServerKey, config: LspServerConfig }` with `language_id()`/`root()`; `ServerKey` is `{ name, root }` and orders by name then root.
- `builtin_server_configs()` — pyright, rust, typescript, gopls (60 s startup), jdtls; opt-in and never PATH-probed.
- `ServerSupervisor` — `new(config, factory)`, `with_diagnostics`, `with_timer`, `with_restart_policy`, `with_request_timeout`, `with_startup_timeout`, `with_workspace_root`, `client()`, `run(op)`, `restart()`, `shutdown()`, `status()`, `attempts()`, `key()`, `config()`.
- `RestartPolicy` — `{ max_attempts: 3, base_delay: 500 ms, max_delay: 8 s }`, `delay_for(attempt)` = `base × 2^(attempt-1)` capped, no jitter.
- `BackoffTimer` trait with `TokioTimer` (production) and `InstantTimer` (records delays, returns immediately; the deterministic test clock).
- `ServerState` — `Stopped`, `Starting`, `Running`, `Failed`; `ServerStatus` — `{ key, language_id, state, healthy, attempts, last_error }`.
- `LspTool` / `LspToolConfig` — `LspTool::new(config)`, `card_for(language)`, `with_transport`, `with_transport_factory`, `with_diagnostics`, `with_backoff_timer`, `invoke` (impl `Agent`-style `Tool`), `shutdown`, `server_status`, `diagnostics`, `config`.
- `LspToolConfig` — `workspace_root`, `language`, `server`, `servers`, `restart_policy`, `language_id`, `request_timeout`, `max_file_size`, `context_lines`; `new()` is the `Default` value; builders `with_workspace_root`, `with_language`, `with_server`, `with_servers`, `with_discovered_server`, `with_builtin_servers`, `with_restart_policy`, `with_language_id`, `with_request_timeout`, `with_max_file_size`, `with_context_lines`.
- `LspToolConfig::SPEC_KEYS` — the six spec keys accepted by its `FromStr`, in `Display` order: `workspace`, `language`, `language_id`, `request_timeout`, `max_file_size`, `context_lines`.
- `LspAction` — `hover`, `definition`, `references`, `completion`, `symbols`, `workspace_symbols`, `diagnostics`, `implementation`, `prepare_call_hierarchy`, `incoming_calls`, `outgoing_calls`; `ALL` is in that order and `requires_path` is false only for `workspace_symbols`/`diagnostics`.
- External-caller surface (a host outside the crate): construct with `LspTool::new(LspToolConfig)`; `LspToolConfig::new()` or `Default` plus the `with_*` builders, and every field is `pub` so a literal with `..Default::default()` works; `LspAction` also implements `FromStr`/`Display`, and every action keeps its snake_case serde name; `LspArgs::new(action)` plus `with_path`/`with_position`/`with_query`/`with_context_lines`; `LspToolConfig`, `LspAction` and `LspArgs` all round-trip through their `FromStr`/`Display` text forms; the only invocation entry point from outside is the `Tool` trait (`card`, `invoke(inputs, &ToolContext)`), because `LspTool::run` and every helper on the tool are `pub(super)`.
- `LspArgs` — `new(action)`, `with_path`, `with_position(line, column)` (one-based, stored verbatim), `with_query`, `with_context_lines`, plus `FromStr`/`Display`; render helpers — `render::{symbol_kind_name, diagnostic_severity_name, render_hover, render_definition, render_references, render_completion, render_symbols, render_workspace_symbols, render_diagnostics, render_diagnostic_files, render_call_hierarchy_items, render_incoming_calls, render_outgoing_calls, format_context_excerpt, display_uri, format_location}`.
## Text Forms
- `LspToolConfig::from_str` parses a comma-separated `key=value` spec over the scalar knobs; missing keys keep their defaults, so `""` is the default configuration.
- The accepted keys are `workspace` (root path), `language`, `language_id`, `request_timeout` (a whole number with an `ms`, `s` or `ns` suffix, greater than zero), `max_file_size` (bytes, plain integer) and `context_lines`.
- Server definitions (`server`, `servers`) are deliberately not part of the spec: they describe a process (command, args, env) and stay builder-only; parsing them is an unknown-key error.
- An empty item, a token without `=`, a duplicate key, an unknown key, an empty string value or a bad number/duration is `Error::Validation` naming the offending token.
- `LspToolConfig::Display` renders only the knobs that differ from the defaults (comma-separated, `SPEC_KEYS` order), so a default config renders as the empty string and every rendering parses back to the same config.
- `LspAction::from_str` accepts every `LspAction::ALL` name case-insensitively and with `-` in place of `_` (so a kebab flag value parses); an unknown name is `Error::Validation` listing every accepted value; `Display` emits `as_str()`.
- `LspArgs::from_str` parses `<action> [path][:line[:column]] [query...]`: the action, then a path with an optional one-based `line`/`column`, then a query.
- Positions split from the right and only trailing all-digit segments count, so `C:\dir\main.rs:10:5` keeps its Windows path; a non-digit segment after a numeric one (`src/lib.rs:10:x`) and a third numeric segment are `Error::Validation`.
- `line`/`column` must be in `1..=4294967295`; `0` or an overflowing value is `Error::Validation`.
- `workspace_symbols` takes every remaining token as its query joined with single spaces (`workspace_symbols Foo Bar`); no query means "all symbols".
- `diagnostics` takes the next token as its path and no path means "drain every buffered document"; a position suffix is rejected because the action ignores it.
- A missing path for a path-taking action, an unknown action and an unexpected trailing token are `Error::Validation`.
- `LspArgs::Display` renders the canonical `<action> ...` form (only the fields the grammar carries for that action) and round-trips through `FromStr` for whitespace-free paths and space-containing queries.
- Tokens are whitespace-separated, so a path containing whitespace must be built with `LspArgs::with_path`; a path whose last `:`-separated segment is numeric is indistinguishable from a position.


## Request Table

| Action | LSP method | Params | Payload keys |
| --- | --- | --- | --- |
| `hover` | `textDocument/hover` | `textDocument.uri`, `position` | `action`, `file_path`, `result`, `hover` |
| `definition` | `textDocument/definition` | `textDocument.uri`, `position` | `action`, `file_path`, `result`, `locations` |
| `references` | `textDocument/references` | `textDocument.uri`, `position`, `context.includeDeclaration` (true) | `action`, `file_path`, `result`, `locations` |
| `completion` | `textDocument/completion` | `textDocument.uri`, `position` | `action`, `file_path`, `result`, `items` |
| `symbols` | `textDocument/documentSymbol` | `textDocument.uri` | `action`, `file_path`, `result`, `symbols` |
| `workspace_symbols` | `workspace/symbol` | `query` | `action`, `result`, `symbols` |
| `implementation` | `textDocument/implementation` | `textDocument.uri`, `position` | `action`, `file_path`, `result`, `locations` |
| `prepare_call_hierarchy` | `textDocument/prepareCallHierarchy` | `textDocument.uri`, `position` | `action`, `file_path`, `result`, `items` |
| `incoming_calls` | `textDocument/prepareCallHierarchy` then `callHierarchy/incomingCalls` | `textDocument.uri`, `position`; `item` | `action`, `file_path`, `result`, `calls`, `item` |
| `outgoing_calls` | `textDocument/prepareCallHierarchy` then `callHierarchy/outgoingCalls` | `textDocument.uri`, `position`; `item` | `action`, `file_path`, `result`, `calls`, `item` |
| `diagnostics` (with path) | push registry, then `textDocument/diagnostic` | `textDocument.uri`, optional `previousResultId` | `action`, `file_path`, `result`, `diagnostics`, `source` (`push`/`pull`/`none`), plus `stale: true` when an `unchanged` answer had no baseline |
| `diagnostics` (no path) | registry `drain` only | none | `action`, `result`, `files`, `count` |

- Documents are synchronised before a position request (`didOpen`, then full-text `didChange`) and reads are gated by `max_file_size` (10 MiB default).
- `incoming_calls`/`outgoing_calls` prepare the hierarchy and use the first returned item when several are offered.
- Diagnostics pulls store the returned `resultId` per URI and send it back as `previousResultId`; an `unchanged` report reuses the last full items.
- A `-32601` answer maps to `Operation not supported by this language server: <method>`.

## Server Selection Precedence

1. `LspToolConfig::server` (single-server mode) always wins; key name is `LspServerConfig::name` or `"default"` and root is `LspServerConfig::root` or the canonical workspace root.
2. Otherwise `LspToolConfig::servers` is indexed by `DiscoveryIndex`: case-insensitive extension match (leading dot optional), ordered by `(priority, registration order)`, first definition whose root inference succeeds.
3. Otherwise, with an injected transport/factory, a default definition using `LspToolConfig::language_id` is synthesized; without any layer the call fails with the typed `NO_SERVER_CONFIGURED` payload.
- `workspace_symbols` has no path, so it uses the explicit server, then `resolve_by_language(language_id)`, then the first discovery definition at the workspace root, then the injected/default path.
- Root inference walks from the file's parent up to the workspace root, returns the nearest directory containing any `root_marker`, and otherwise falls back to `LspServerConfig::root` (when inside the workspace) or the workspace root.
- Rust definitions follow Python: walk up to the nearest `Cargo.toml` containing `[workspace]` within the workspace bound.
- One `ServerSupervisor` is created lazily per `ServerKey` and reused; concurrent first calls serialize on the supervisor's start lock.

## Crash Recovery Policy

- A server whose I/O task exits is marked not alive; pending requests fail with `Error::Transport` naming the crash/close.
- `ServerSupervisor::run` retries the operation exactly once on a fresh client after sleeping `delay_for(attempt)` on the injected timer.
- `RestartPolicy` defaults: 3 attempts, 500 ms base, exponential doubling, 8 s cap, no jitter (deterministic).
- A successful start resets `attempts` to 0; exhausting the budget sets `ServerState::Failed` and returns `Error::Transport("LSP server '<name>' exceeded max crash recovery attempts (N); last error: ...")`.
- A refused start (budget already exhausted) also records `Failed` with the underlying `last_error` under the same status lock, so a status read after a failed `restart()` never shows a stale `Stopped`/`Starting`.
- Lock order is `client` before `status` on every path (`status`, `live_client`, `stop_current`); no lock is held across `.await` except the start serialization mutex.
- `LspTool::shutdown`/`ServerSupervisor::shutdown` stop servers idempotently and permanently inhibit restarts (later calls fail with a typed "has been shut down" error).
- Status and attempt counters are exposed through `LspTool::server_status()` / `ServerSupervisor::status()`.

## Diagnostics Semantics

- `publish` is version-aware: a non-empty publish with a lower `version` than the stored one is ignored; otherwise it replaces the URI's set and records the version.
- An empty diagnostic list clears the URI, resets its delivered-key history, and emits `DiagnosticsChange::Cleared`.
- Entries dedupe by `message|severity|line:character|code`, are severity-sorted (Error first, unknown severity treated as 3), and are capped per file and per registry (`max_files`, least-recently-updated URI evicted).
- `max_files` bounds *all* per-document state: the delivered-key history is capped at the same URI count (least-recently-drained dropped first), and a document evicted from the buffer loses its history with it, so a long session keeps no key set per URI ever seen.
- `drain` mirrors Python `get_and_clear`: intra-round dedupe, cross-round suppression of delivered keys, per-file and global caps, local paths resolved from URIs.
- The registry is an owned value per tool (shared through `LspTool::with_diagnostics`), not a process-global singleton; Python's `get_instance` singleton is deliberately not reproduced because the crate forbids undocumented global mutable state.
- Push data wins over pull for a file; pull is attempted only when push is empty and the server advertises `diagnosticProvider`.
- An `unchanged` pull answer without a cached full report triggers exactly one extra pull without `previousResultId`; if the server still answers `unchanged`, the payload carries `stale: true` (never a silently empty "clean" list).

## Constraints

- Errors are `openjiuwen_common::error::{Error, Result}`; malformed frames, EOF, oversize frames, timeouts, JSON-RPC errors, launch failures and exhausted recovery budgets are typed.
- `LspError::into_error` keeps the whole error as `Error::External`, so the numeric code survives; `protocol::is_method_not_found` matches on `LspError::is_method_not_found` first and falls back to the message phrase "method not found" only for servers that omit or mis-code it.
- An unreadable `Cargo.toml` during Rust root inference is logged at debug level (path + error) and still treated as "not a workspace root".
- `LspTool::invoke` always returns `Ok(payload)`; failures set `success=false` and a human-readable `error`.
- No `unsafe`, no `println!` (tracing only), no locks held across `.await` except the supervisor's tokio start mutex, which exists to start one server per key and is documented.
- Paths resolve inside `LspToolConfig::workspace_root` only; `..` and absolute paths outside the root are rejected before any read, and `DiscoveryIndex` refuses paths outside its workspace root.
- Files are synchronized before a request (`didOpen`, then full-text `didChange` with monotonic versions); notification handlers are safe to register before and after `initialize`, and a panicking handler cannot kill the I/O task.
- No new dependencies beyond the crate manifest (tokio, serde/serde_json, schemars, async-trait, parking_lot, tracing, sysop path helpers).
- Files stay around ≤1000 lines; `tool.rs` delegates action execution to `tool_ops.rs`.

## Known Issues

- The `lsp` card description is sourced from `src/tools/descriptions.rs` (the single source of truth, so no duplicate string lives here) and its current entry does not yet name implementation, call hierarchy or diagnostics; broadening it is a sibling edit in that table.
- `lsp` has no CLI mounting: the CLI does not resolve a per-workspace server config yet (it mounts thirteen extended tools but not `lsp`).

## Remaining Deferrals

- `.gitignore` filtering of navigation results (`openjiuwen/harness/lsp/core/utils/git_ignore.py`): needs a git-index contract this node does not own; results are returned unfiltered.
- Server discovery from `PATH`/manifest search (`servers/registry.py` `build_configs_async`): definitions come from the built-in table or the host, and a missing binary fails typed at launch.
- Diagnostics subscriptions (`workspace/diagnostic`, refresh requests) and workspace-wide pulls: only per-file `textDocument/diagnostic` and push delivery are implemented.
- LSP formatter/rename/code-action edits are out of scope (read-only navigation tool).
- Workspace-symbol caching: every `workspace/symbol` call queries the server.

## Test Strategy

- `tests/fake_server.rs` drives every client and tool test over one `tokio::io::duplex` half: canned navigation/call-hierarchy/implementation/diagnostic responses, `publishDiagnostics` after `didOpen` (one warning) and after a `clean` `didChange` (empty), plus recorded messages for exact method/params assertions.
- `tests/framing.rs`, `tests/uri.rs`, `tests/transport.rs` cover byte-level framing, URI conversion and both transports (`/bin/cat` as an echo server when present).
- `tests/client.rs` and `tests/client_ext.rs` cover initialize capability capture, server-initiated requests, notification dispatch (including panicking handlers), typed methods, versions, timeouts, malformed frames, EOF, liveness and shutdown reasons.
- `tests/wire.rs` covers tolerant parsing of diagnostics, diagnostic reports and call hierarchy (flat vs nested items, string vs numeric codes).
- `tests/diagnostics.rs` covers version awareness, clearing, caps, cross-round dedupe, the `max_files` bound on both the buffer and the delivered-key history, `subscribe`, and an end-to-end push through the client.
- `tests/discovery.rs` covers extension matching, priority/order, language id, root markers, Cargo workspace roots, containment, builtin definitions, and the debug record left behind by an unreadable `Cargo.toml`.
- `tests/retry.rs` covers one-retry recovery, delay recording, budget exhaustion, refused-start status, shutdown inhibition, concurrent `status()` reads during start/stop, and backoff doubling/capping.
- `tests/tool.rs`, `tests/tool_ops.rs`, `tests/tool_lifecycle.rs` cover the 11-action card, every action payload, typed and message-fallback method-not-found mapping, diagnostics push→pull→unchanged (including the `stale` marker and the forced full re-pull when no snapshot exists), per-extension routing with two fake servers, crash retry, shutdown and status.
- `tests/parse.rs` covers `LspToolConfig::new`, every spec key, the duration/number suffixes, every spec error (unknown/duplicate/empty/malformed/bad value), `Display` round-trips, every action spelling (snake/kebab/upper), the `<action> [path][:line[:column]] [query...]` grammar including Windows paths, every `LspArgs` error (missing path, bad/zero/overflowing position, position on `diagnostics`, trailing token), `LspArgs`/`LspToolConfig` round-trips, and the unchanged serde names and card schema.
- Definition of done: `cargo fmt -p openjiuwen-harness`, `cargo test -p openjiuwen-harness --offline` (default, `--no-default-features`, `--all-features`), `cargo clippy -p openjiuwen-harness --all-targets --all-features --offline -- -D warnings`, `cargo doc -p openjiuwen-harness --no-deps` all clean and offline.

## Routing Table

- `./protocol.rs` → JSON-RPC framing, message builders, LSP position/location/hover/unit types, result normalisers, server/client capabilities.
- `./wire.rs` → diagnostics and call-hierarchy wire types plus tolerant parsers.
- `./uri.rs` → `file://` URI encoding/decoding.
- `./transport.rs` → `LspTransport`, `StdioTransport` (process spawn, stderr drain, kill on drop), `ChannelTransport` (in-memory duplex).
- `./client.rs` → single-transport I/O task, request correlation, timeouts, notification handlers, liveness/close reason, document versions, typed methods, graceful shutdown.
- `./config.rs` → `LspServerConfig` (name, command, args, env, root, language id, extensions, priority, root markers, timeouts).
- `./diagnostics.rs` → bounded, version-aware push-diagnostic registry, drain semantics and change stream.
- `./discovery.rs` → `ServerKey`/`ResolvedServer`/`DiscoveryIndex`, extension/priority/root-marker resolution, builtin definitions.
- `./instance.rs` → `ServerSupervisor`, restart policy, back-off timers, health/attempt status, one-retry `run`.
- `./render.rs` → one-based text rendering for every action, symbol/severity maps, diagnostics and call-hierarchy formats, context excerpts.
- `./tool.rs` → `LspTool`, `LspToolConfig`, `LspAction`, `LspArgs`, card/schema construction, server selection and supervisor map, payload envelopes.
- `./parse.rs` → `FromStr`/`Display` for `LspToolConfig` (`SPEC_KEYS` spec), `LspAction` and `LspArgs` (`<action> [path][:line[:column]] [query...]`), position splitting, duration/number parsing.
- `./tool_ops.rs` → per-action execution: path/URI resolution, document sync, pull diagnostics, call hierarchy, payload shaping and error mapping.
- `./tests/` → offline suites: framing, URI, transport, fake server, client/wire, diagnostics, discovery, retry, tool operations, lifecycle, and the text forms of `parse.rs`.
