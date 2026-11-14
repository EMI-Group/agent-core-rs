//! `openjiuwen watch` — live reload diagnostics for the workspace.
//!
//! The harness owns the workspace watcher ([`openjiuwen_harness::watcher`]): a
//! debounced, coalesced change stream over the files the harness reloads from —
//! skill libraries, agent specs, instruction files (`OPENJIUWEN.md`,
//! `IDENTITY.md`) and the MCP configuration — plus the reload handlers that
//! re-read them. This module owns the terminal end: resolve the watch plan from
//! the workspace, start the watcher (the `openjiuwen-harness` `notify` feature is
//! enabled by this binary), and print one line per change, reload failure, drop
//! notice or watcher error.
//!
//! `chat --watch` reuses the same session with stderr as the sink, so a REPL user
//! sees reloads without the conversation stream on stdout being touched.
use std::time::Duration;

use openjiuwen_common::error::Result;

/// `openjiuwen watch [--json] [--list] [--max-events N]`.
#[derive(Debug, Clone, clap::Args)]
pub(crate) struct WatchCli {
    /// Print one JSON object per line instead of text.
    #[arg(long)]
    pub(crate) json: bool,
    /// Print the resolved watch targets and exit without watching.
    #[arg(long)]
    pub(crate) list: bool,
    /// Debounce window in milliseconds (default 250).
    #[arg(long = "debounce-ms", default_value_t = 250)]
    pub(crate) debounce_ms: u64,
    /// Exit after this many updates; without it the watcher runs until Ctrl-C.
    #[arg(long = "max-events", value_name = "N")]
    pub(crate) max_events: Option<u64>,
}

/// Where a running watcher writes its report lines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WatchSink {
    /// `openjiuwen watch`: the report is the command's result.
    Stdout,
    /// `chat --watch`: the report is a diagnostic beside the conversation.
    Stderr,
}

/// How a watcher session reports.
#[derive(Debug, Clone, Copy)]
pub(crate) struct WatchSpec {
    /// Emit JSON objects instead of text lines.
    pub(crate) json: bool,
    /// Debounce window for coalescing filesystem events.
    pub(crate) debounce: Duration,
    /// Stop after this many updates; `None` runs until cancellation.
    pub(crate) max_events: Option<u64>,
    /// Destination for the report lines.
    pub(crate) sink: WatchSink,
}

impl WatchSpec {
    /// The spec `chat --watch` uses: text lines on stderr, no event limit.
    pub(crate) fn chat_notices(debounce: Duration) -> Self {
        Self {
            json: false,
            debounce,
            max_events: None,
            sink: WatchSink::Stderr,
        }
    }
}

/// A started watcher whose report task runs until [`WatchSession::stop`].
///
/// Dropping the session stops the report task cooperatively; `stop` also shuts
/// the watcher itself down (drains pending events, joins both harness tasks).
pub(crate) struct WatchSession {
    /// Handle to the harness watcher.
    pub(crate) handle: openjiuwen_harness::watcher::WatcherHandle,
}

impl WatchSession {
    /// Shuts the watcher and its report task down.
    ///
    /// # Errors
    ///
    /// Returns `Error::sys_op` when a watcher task panicked; shutdown is
    /// otherwise idempotent and infallible.
    pub(crate) async fn stop(self) -> Result<()> {
        self.handle.shutdown().await
    }
}

/// Runs `openjiuwen watch`.
///
/// # Errors
///
/// Returns a typed error when the watch plan is empty, a target path cannot be
/// watched, or the OS watcher cannot start; option-resolution errors come from
/// the harness.
pub(crate) async fn run(args: &WatchCli, base: &openjiuwen_harness::cli::Cli) -> Result<()> {
    let _ = (args, base);
    Err(openjiuwen_common::error::Error::unsupported(
        "watch command is not wired yet",
    ))
}

/// Starts a watcher that reports to [`WatchSpec::sink`] without blocking.
///
/// Used by `chat --watch`; `run` drives the same session and awaits it.
///
/// # Errors
///
/// Returns the same typed errors [`run`] does.
pub(crate) async fn start(
    _base: &openjiuwen_harness::cli::Cli,
    _spec: WatchSpec,
) -> Result<WatchSession> {
    Err(openjiuwen_common::error::Error::unsupported(
        "watch command is not wired yet",
    ))
}
