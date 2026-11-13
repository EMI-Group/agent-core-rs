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
//! - `config` — `CliOptions` resolution (flags/env/settings/defaults)
//! - `io` — `LineReader`/`Writer` traits and the `CliIo` handle
//! - `render` — `RenderState`, the chunk-to-terminal renderer
//! - `run` — one-shot execution and its output formats
//! - `repl` — the interactive REPL and its slash commands
//! - `permissions` — approval prompts and the permission host
//! - `session_store` — append-only session records under the home directory
//! - `allow_rules` — permanent allow-rule persistence to `permissions.yaml`
//! - `compact` — the `/compact` context-compaction slash command
//! - `history` — persistent REPL input history with a retained-entry cap
//! - `instruction` — project instruction files injected into the prompt
//! - `mcp_config` — `mcp.json` discovery and merge
//! - `onboarding` — the first-run onboarding wizard
//! - `paths` — every CLI-owned file path and environment switch
//! - `shell` — the `!cmd` shell-passthrough command
//! - `skill_command` — `/<skill>` slash-command dispatch
//! - `style` — ANSI styling helpers and colour gating
//! - `usage` — token and cost accounting for `/cost`
//! - `tests` — offline unit and integration suite
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

// Public CLI submodules: each owns one user-facing concern and is documented
// in its own file.
pub mod allow_rules;
pub mod compact;
pub mod history;
pub mod instruction;
pub mod mcp_config;
pub mod onboarding;
pub mod paths;
pub mod shell;
pub mod skill_command;
pub mod style;
pub mod usage;

#[cfg(test)]
mod tests;

pub use config::{
    CliOptions, EnvSource, MapEnv, ProcessEnv, Settings, DEFAULT_LANGUAGE, DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_TOKENS, DEFAULT_MODEL, DEFAULT_PROVIDER, MIN_MAX_TOKENS, OPENJIUWEN_API_BASE,
    OPENJIUWEN_API_KEY, OPENJIUWEN_HOME, OPENJIUWEN_MAX_ITERATIONS, OPENJIUWEN_MAX_TOKENS,
    OPENJIUWEN_MODEL, OPENJIUWEN_PROVIDER, OPENJIUWEN_WORKSPACE, SESSION_PREFIX,
};
pub use io::{CliIo, LineReader, StdioReader, StdoutWriter, Writer};
pub use permissions::{render_prompt, render_prompt_styled, CliPermissionHost, PermissionAnswer};
pub use render::RenderState;
pub use repl::{InterruptAction, InterruptCounter, Repl, ReplCommand, TurnOutcome};
pub use run::{
    build_agent, build_agent_with, new_session, result_json_line, run_once, run_with_model,
    session_id, stream_json_line, FoundationMcpToolProvider, McpToolProvider, RunOutcome,
};
pub use session_store::{MessageRecord, SessionRecord, SessionStore, SessionSummary};

use std::io::IsTerminal;
use std::path::PathBuf;

use openjiuwen_common::error::Result;

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

    /// Disable ANSI colour in terminal output.
    ///
    /// Colour is off automatically when the output is not a terminal or when
    /// `NO_COLOR` is set; this flag forces it off unconditionally.
    #[arg(long = "no-color", global = true)]
    pub no_color: bool,

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

/// Returns whether standard output is a terminal.
///
/// Colour gating uses this probe; like [`stdin_is_tty`] it lives here so the
/// decision itself stays a pure function of the parsed flags and the
/// environment (see [`paths::color_enabled`]).
#[must_use]
pub fn stdout_is_tty() -> bool {
    std::io::stdout().is_terminal()
}

/// Runs the CLI end to end: resolve options, dispatch, then run or chat.
///
/// This is the canonical entry point of the `openjiuwen` binary. The workspace
/// directory is created only once a command is going to run, so `--help` and
/// parse errors never touch the disk.
///
/// The one-shot `run` stays uncoloured and non-interactive: it uses
/// [`CliIo::stdio_non_interactive`], so permission prompts fail closed and the
/// first-run wizard never runs.
///
/// `chat` builds one [`CliIo::stdio`] handle, enables colour only on a terminal
/// with `--no-color` and `NO_COLOR` unset (see [`paths::color_enabled`]), runs the
/// onboarding wizard when the home has no settings file yet
/// ([`onboarding::maybe_run`] is a no-op otherwise), and then hands the same
/// handle to the agent (permission host) and to [`Repl::with_io_and_signals`].
/// The REPL self-wires its history, skills and shell commands from
/// `options.home`/`options.workspace`.
///
/// # Errors
///
/// Returns [`Error::Config`](openjiuwen_common::error::Error::Config) when the
/// options cannot be resolved or the workspace cannot be created, and whatever
/// onboarding, model construction, the one-shot run or the REPL reports.
pub async fn run_cli(cli: Cli) -> Result<()> {
    let options = CliOptions::resolve(&cli)?;
    match cli.dispatch(stdin_is_tty()) {
        Dispatch::Run(args) => {
            tracing::debug!(?args, "dispatching one-shot run");
            options.ensure_workspace()?;
            let io = CliIo::stdio_non_interactive();
            run::run_once(&options, &args, &io).await.map(|_| ())
        }
        Dispatch::Chat => {
            tracing::debug!("dispatching interactive chat");
            options.ensure_workspace()?;
            let color = paths::color_enabled(cli.no_color, &ProcessEnv, stdout_is_tty());
            let io = CliIo::stdio().with_color(color);
            // First-run setup: a no-op unless the home has no settings file and
            // the terminal is interactive (see `onboarding::maybe_run`).
            onboarding::maybe_run(&io, &options.home, &ProcessEnv).await?;
            let model = options.build_model().await?;
            let agent = run::build_agent(model, &options, &io).await?;
            Repl::with_io_and_signals(agent, options, io).run().await
        }
    }
}

/// Alias of [`run_cli`] under the name the binary calls.
///
/// `run` is also a private module (one-shot execution); the two live in
/// different namespaces, so the spelling is unambiguous.
///
/// # Errors
///
/// See [`run_cli`].
pub async fn run(cli: Cli) -> Result<()> {
    run_cli(cli).await
}
