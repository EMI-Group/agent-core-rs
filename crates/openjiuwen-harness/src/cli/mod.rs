//! clap surface of the `openjiuwen` command-line front-end.
//!
//! The binary in `openjiuwen-cli` parses [`Cli`] and hands it to the CLI entry
//! point; everything else — option resolution, the one-shot runner, the
//! interactive REPL and the permission prompts — lives beside this module.
//!
//! Module layout:
//!
//! - `mod.rs` (this file) — the command grammar: [`Cli`], [`Command`],
//!   [`RunArgs`]/[`ChatArgs`], [`OutputFormat`], [`Dispatch`] and
//!   [`stdin_is_tty`]
//! - [`config`] — `CliOptions` resolution (flags/env/settings/defaults)
//! - [`io`] — `LineReader`/`Writer` traits and the `CliIo` handle
//! - [`render`] — `RenderState`, the chunk-to-terminal renderer
//! - [`run`] — one-shot execution and its output formats
//! - [`repl`] — the interactive REPL and its slash commands
//! - [`permissions`] — approval prompts and the permission host
//! - [`session_store`] — append-only session records under the home directory
//! - [`tests`] — offline unit and integration suite
//!
//! Dispatch is a pure function of the parsed command line plus one injected
//! boolean, so it is testable without touching the process environment or the
//! terminal.

mod config;
mod io;
mod permissions;
mod render;
mod repl;
mod run;
mod session_store;

#[cfg(test)]
mod tests;

// Placeholder modules are still empty: their glob re-exports carry nothing yet,
// which `unused_imports` and `unreachable_pub` would flag. The attributes go
// away together with the globs once each module exposes its items.
#[allow(unused_imports, unreachable_pub)]
pub use config::*;
pub use io::*;
#[allow(unused_imports, unreachable_pub)]
pub use permissions::*;
#[allow(unused_imports, unreachable_pub)]
pub use render::*;
#[allow(unused_imports, unreachable_pub)]
pub use repl::*;
#[allow(unused_imports, unreachable_pub)]
pub use run::*;
pub use session_store::*;

use std::io::IsTerminal;
use std::path::PathBuf;

/// Parsed command line of the `openjiuwen` binary.
///
/// Every option is global so it may appear before or after a subcommand, and
/// every option is optional: the effective value comes from the flag, then the
/// environment, then `~/.openjiuwen/settings.json`, then the defaults.
#[derive(Debug, Clone, clap::Parser)]
#[command(name = "openjiuwen", version, about = "openJiuwen agent CLI")]
pub struct Cli {
    /// Model name to use (for example `gpt-4o`).
    #[arg(short = 'm', long, global = true)]
    pub model: Option<String>,

    /// Model provider to use (for example `openai` or `anthropic`).
    #[arg(long, global = true)]
    pub provider: Option<String>,

    /// API key for the model provider.
    ///
    /// Parsed as plain text and turned into a secret by `CliOptions`; it is
    /// never logged.
    #[arg(long, global = true)]
    pub api_key: Option<String>,

    /// Base URL of the model provider API.
    #[arg(long, global = true)]
    pub api_base: Option<String>,

    /// Workspace directory the agent may read and write.
    #[arg(short = 'w', long, global = true)]
    pub workspace: Option<PathBuf>,

    /// Stream reasoning chunks and extra diagnostics.
    #[arg(short = 'v', long, global = true)]
    pub verbose: bool,

    /// Optional subcommand; without one the mode is chosen from the terminal.
    #[command(subcommand)]
    pub command: Option<Command>,
}

/// Subcommands of the `openjiuwen` binary.
#[derive(Debug, Clone, clap::Subcommand)]
pub enum Command {
    /// Run a single prompt and exit.
    Run(RunArgs),
    /// Start the interactive chat REPL.
    Chat(ChatArgs),
}

/// Options of the `chat` subcommand.
///
/// The REPL takes no options of its own today; global options still apply.
#[derive(Debug, Clone, Default, clap::Args)]
pub struct ChatArgs {}

/// Options of the `run` subcommand.
#[derive(Debug, Clone, Default, PartialEq, Eq, clap::Args)]
pub struct RunArgs {
    /// Prompt to run; when omitted (or `-`) it is read from stdin.
    pub prompt: Option<String>,

    /// How the answer and streamed chunks are printed.
    #[arg(short = 'f', long, value_enum, default_value_t = OutputFormat::Text)]
    pub output_format: OutputFormat,
}

/// Output format of the `run` subcommand.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum OutputFormat {
    /// Human-readable text, rendered while it streams (the default).
    #[default]
    Text,
    /// A single JSON object printed once the run completes.
    Json,
    /// One JSON object per streamed chunk, plus a terminal result line.
    #[value(name = "stream-json")]
    StreamJson,
}

impl OutputFormat {
    /// Returns the wire name of the format: `text`, `json` or `stream-json`.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Json => "json",
            Self::StreamJson => "stream-json",
        }
    }
}

impl std::fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// What the CLI should do for a parsed command line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Dispatch {
    /// Run one prompt and exit.
    Run(RunArgs),
    /// Start the interactive chat REPL.
    Chat,
}

impl Cli {
    /// Resolves the effective command for this command line.
    ///
    /// An explicit subcommand always wins. Without one the CLI chats when
    /// `stdin_is_tty` is true and otherwise runs a single prompt, which is then
    /// read from stdin.
    #[must_use]
    pub fn dispatch(&self, stdin_is_tty: bool) -> Dispatch {
        match &self.command {
            Some(Command::Run(args)) => Dispatch::Run(args.clone()),
            Some(Command::Chat(_)) => Dispatch::Chat,
            None if stdin_is_tty => Dispatch::Chat,
            None => Dispatch::Run(RunArgs::default()),
        }
    }
}

/// Returns whether standard input is a terminal.
///
/// This is the only process-level probe in the module; keeping it here leaves
/// [`Cli::dispatch`] pure and testable.
#[must_use]
pub fn stdin_is_tty() -> bool {
    std::io::stdin().is_terminal()
}
