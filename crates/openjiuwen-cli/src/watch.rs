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
//! `--list` renders the resolved plan and exits without starting a watcher, so a
//! user can check what *would* be observed offline (nothing has to exist yet:
//! [`WatchPlan::existing_only`] drops the paths whose watch directory is missing
//! and records them as skipped).
//!
//! `chat --watch` reuses the same session with stderr as the sink, so a REPL user
//! sees reloads without the conversation stream on stdout being touched.
use std::io::Write;
use std::time::Duration;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_harness::cli::{CliOptions, ProcessEnv};
use openjiuwen_harness::skills::SkillLibrary;
use openjiuwen_harness::watcher::{
    InstructionReloadHandler, InstructionReloader, McpConfigReloadHandler, McpConfigReloader,
    SkillLibraryReloader, SkillsReloadHandler, WatchPlan, WatchUpdate, WatcherConfig,
    WatcherHandle, WorkspaceWatcher,
};

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
    pub(crate) handle: WatcherHandle,
    /// The task that renders the subscription to [`WatchSpec::sink`].
    task: tokio::task::JoinHandle<()>,
}

impl WatchSession {
    /// Shuts the watcher and its report task down.
    ///
    /// The report task ends once the broadcast channel closes, which happens
    /// when the handle is dropped after the shutdown handshake drained the
    /// pending debounce window.
    ///
    /// # Errors
    ///
    /// Returns `Error::sys_op` when a watcher task panicked; shutdown is
    /// otherwise idempotent and infallible.
    pub(crate) async fn stop(self) -> Result<()> {
        let Self { handle, task } = self;
        let shutdown = handle.shutdown().await;
        // Dropping the last sender closes the subscription stream, so the
        // report task observes `None` and finishes; awaiting it then cannot
        // block on a channel that stays open.
        drop(handle);
        let joined = task.await;
        shutdown?;
        joined.map_err(|error| Error::sys_op(format!("watch report task failed: {error}")))
    }
}

/// Resolves the fully effective CLI options of this invocation.
fn resolve_options(base: &openjiuwen_harness::cli::Cli) -> Result<CliOptions> {
    CliOptions::resolve(base)
}

/// Discovers the watch targets of `options`, keeping only existing roots.
fn resolve_plan(options: &CliOptions) -> WatchPlan {
    let env = ProcessEnv;
    WatchPlan::workspace(&options.workspace, &options.home, &env).existing_only()
}

/// Builds the watcher for `plan`, registers the reload handlers and starts it.
async fn start_watcher(
    options: &CliOptions,
    plan: WatchPlan,
    debounce: Duration,
) -> Result<WatcherHandle> {
    let instructions = InstructionReloader::load(&options.workspace, &options.home)?;
    let mcp_paths =
        openjiuwen_harness::cli::paths::mcp_config_paths(&options.home, &options.workspace, &env());
    let mcp = McpConfigReloader::new(mcp_paths)?;
    let library = SkillLibrary::with_roots([options.workspace.join("skills")]);

    WorkspaceWatcher::new()
        .with_config(WatcherConfig::default().with_debounce(debounce))
        .watch_plan(plan)
        .on_reload(InstructionReloadHandler::new(std::sync::Arc::new(
            instructions,
        )))
        .on_reload(McpConfigReloadHandler::new(std::sync::Arc::new(mcp)))
        .on_reload(SkillsReloadHandler::new(std::sync::Arc::new(
            SkillLibraryReloader::new(library),
        )))
        .start()
        .await
}

/// Returns the process env source (a unit struct, so it costs nothing).
fn env() -> ProcessEnv {
    ProcessEnv
}

/// Prints the resolved plan and returns without starting a watcher.
fn print_plan(plan: &WatchPlan, json: bool) -> Result<()> {
    if json {
        let targets: Vec<serde_json::Value> = plan
            .targets()
            .iter()
            .map(|target| {
                serde_json::json!({
                    "kind": target.kind().as_str(),
                    "path": target.path().display().to_string(),
                    "label": target.label(),
                    "recursive": target.is_recursive(),
                })
            })
            .collect();
        let skipped: Vec<String> = plan
            .skipped()
            .iter()
            .map(|path| path.display().to_string())
            .collect();
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "targets": targets,
                "skipped": skipped,
            }))?
        );
        return Ok(());
    }
    for target in plan.targets() {
        let recursive = if target.is_recursive() {
            " (recursive)"
        } else {
            ""
        };
        println!(
            "{}\t{}{recursive}",
            target.kind().as_str(),
            target.path().display()
        );
    }
    for path in plan.skipped() {
        println!("skipped\t{}", path.display());
    }
    Ok(())
}

/// Announces the number of watched targets on stderr and flushes it.
///
/// stderr rather than stdout keeps `watch --json` and `chat` stdout
/// machine-readable; the flush makes the line a readiness signal for a
/// supervising process.
fn announce(targets: usize) {
    let mut stderr = std::io::stderr();
    // A closed stderr is not an error: the watcher keeps working headless.
    let _ = writeln!(stderr, "watching {targets} target(s)");
    let _ = stderr.flush();
}

/// Renders one update as text.
fn text_line(update: &WatchUpdate) -> String {
    match update {
        WatchUpdate::Event(event) => {
            let mut line = format!(
                "event {} {} {}",
                event.kind.as_str(),
                event.change.as_str(),
                event.path.display()
            );
            if let Some(previous) = &event.previous_path {
                line.push_str(" <- ");
                line.push_str(&previous.display().to_string());
            }
            line
        }
        WatchUpdate::ReloadFailed(failure) => format!(
            "reload failed [{}] {}: {}",
            failure.handler,
            failure.path.display(),
            failure.message
        ),
        WatchUpdate::Dropped(dropped) => {
            format!("dropped {} {}", dropped.dropped, dropped.stage.as_str())
        }
        WatchUpdate::WatcherFailed(failure) => format!("watcher error: {}", failure.message),
    }
}

/// Writes one update to `sink`.
fn report(update: &WatchUpdate, json: bool, sink: WatchSink) {
    let line = if json {
        match serde_json::to_string(update) {
            Ok(line) => line,
            Err(error) => {
                tracing::warn!(%error, "failed to serialize a watcher update");
                return;
            }
        }
    } else {
        text_line(update)
    };
    match sink {
        WatchSink::Stdout => println!("{line}"),
        WatchSink::Stderr => eprintln!("{line}"),
    }
}

/// Returns whether the update budget is exhausted.
fn budget_reached(max_events: Option<u64>, count: u64) -> bool {
    max_events.is_some_and(|max| count >= max)
}

/// Runs `openjiuwen watch`.
///
/// # Errors
///
/// Returns a typed error when the watch plan is empty, a target path cannot be
/// watched, or the OS watcher cannot start; option-resolution errors come from
/// the harness. `--list` never starts a watcher and cannot fail beyond that.
pub(crate) async fn run(args: &WatchCli, base: &openjiuwen_harness::cli::Cli) -> Result<()> {
    let options = resolve_options(base)?;
    let plan = resolve_plan(&options);
    if args.list {
        return print_plan(&plan, args.json);
    }

    let handle = start_watcher(&options, plan, Duration::from_millis(args.debounce_ms)).await?;
    let mut subscription = handle.subscribe();
    announce(handle.target_count());

    // One future for the whole run: re-registering the signal handler on every
    // loop iteration would leak a signal stream per event.
    let ctrl_c = tokio::signal::ctrl_c();
    tokio::pin!(ctrl_c);

    let mut count: u64 = 0;
    loop {
        tokio::select! {
            update = subscription.recv() => {
                let Some(update) = update else { break };
                report(&update, args.json, WatchSink::Stdout);
                count += 1;
                if budget_reached(args.max_events, count) {
                    break;
                }
            }
            _ = &mut ctrl_c => break,
        }
    }

    // Nothing is printed after this point: the handle is shut down before
    // `run` returns, so a supervisor sees the last rendered line and then EOF.
    handle.shutdown().await
}

/// Starts a watcher that reports to [`WatchSpec::sink`] without blocking.
///
/// Used by `chat --watch`; `run` drives the same session and awaits it.
///
/// # Errors
///
/// Returns the same typed errors [`run`] does.
pub(crate) async fn start(
    base: &openjiuwen_harness::cli::Cli,
    spec: WatchSpec,
) -> Result<WatchSession> {
    let options = resolve_options(base)?;
    let plan = resolve_plan(&options);
    let handle = start_watcher(&options, plan, spec.debounce).await?;
    // Subscribe before announcing readiness, so a change made by the process
    // that waits for the readiness line cannot be missed.
    let mut subscription = handle.subscribe();
    announce(handle.target_count());

    let task = tokio::spawn(async move {
        let mut count: u64 = 0;
        while let Some(update) = subscription.recv().await {
            report(&update, spec.json, spec.sink);
            count += 1;
            if budget_reached(spec.max_events, count) {
                break;
            }
        }
    });

    Ok(WatchSession { handle, task })
}
