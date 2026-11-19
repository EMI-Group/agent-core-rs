//! Offline tests for the LSP framing, transport, client, tool and URI helpers.
// The fake server mirrors wire shapes rather than production signatures, and test fixtures
// intentionally build many small payloads; the pedantic style lints do not apply here.
#![allow(
    clippy::similar_names,
    clippy::needless_pass_by_value,
    clippy::too_many_lines,
    clippy::manual_let_else
)]
mod client;
mod client_ext;
mod diagnostics;
mod discovery;
mod fake_server;
mod framing;
mod parse;
mod retry;
mod stalled;
mod tool;
mod tool_lifecycle;
mod tool_ops;
mod transport;
mod uri;
mod wire;
