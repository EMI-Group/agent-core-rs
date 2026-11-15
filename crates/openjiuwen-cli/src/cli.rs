//! Command grammar of the `openjiuwen` binary.
//!
//! The harness owns the `run` / `chat` grammar ([`openjiuwen_harness::cli::Cli`],
//! [`openjiuwen_harness::cli::RunArgs`], [`openjiuwen_harness::cli::ChatArgs`]),
//! and this module reuses those types verbatim — flattened into thin local
//! wrappers — so the two grammars cannot drift.
//! It owns the outer layer the harness has no words for: the global options
//! (duplicated flag-for-flag, because clap needs one `Command` tree to parse),
//! the `auto-harness` command group, and the harness-surface commands the
//! binary exposes directly (`lsp`, `cron`, `team`, `watch`, `sessions`).
//!
//! Conversion is explicit: [`Cli::to_harness`] builds the harness [`Cli`] from
//! the parsed globals, and [`run`] hands `run`/`chat` back to
//! [`openjiuwen_harness::cli::run_cli`] (through [`crate::extras`], which adds
//! the binary's own `--worktree` / `--watch` wiring). A missing subcommand maps
//! to `command: None`, so the harness keeps its TTY-based dispatch.
//!
//! [`Cli`]: openjiuwen_harness::cli::Cli
use std::path::PathBuf;

use openjiuwen_common::error::Result;
use openjiuwen_harness::cli::{ChatArgs, RunArgs};

use crate::auto_harness::{self, AutoHarnessArgs};

/// Parsed command line of the `openjiuwen` binary.
///
/// Every option is global so it may appear before or after a subcommand, and
/// every option is optional: the effective value comes from the flag, then the
/// environment, then `~/.openjiuwen/settings.json`, then the defaults.
#[derive(Debug, Clone, clap::Parser)]
#[command(name = "openjiuwen", version, about = "openJiuwen agent CLI")]
pub(crate) struct Cli {
    /// Options shared by every subcommand.
    #[command(flatten)]
    pub(crate) global: GlobalArgs,
    /// Optional subcommand; without one the mode is chosen from the terminal.
    #[command(subcommand)]
    pub(crate) command: Option<Command>,
}

/// The global options, duplicated from the harness grammar.
///
/// clap parses one `Command` tree, so the outer grammar has to declare the same
/// flags the harness exposes; the unit tests below pin that duplication against
/// the harness grammar. The field names, short/long flags, value names and help
/// text match [`openjiuwen_harness::cli::Cli`] exactly, and every field is
/// forwarded by [`Cli::to_harness`].
#[derive(Debug, Clone, Default, clap::Args)]
pub(crate) struct GlobalArgs {
    /// Model name to use (for example `gpt-4o`).
    #[arg(short = 'm', long, global = true)]
    pub(crate) model: Option<String>,
    /// Model provider to use (for example `openai` or `anthropic`).
    #[arg(long, global = true)]
    pub(crate) provider: Option<String>,
    /// API key for the model provider.
    ///
    /// Parsed as plain text and turned into a secret by `CliOptions`; it is
    /// never logged.
    #[arg(long, global = true)]
    pub(crate) api_key: Option<String>,
    /// Base URL of the model provider API.
    #[arg(long, global = true)]
    pub(crate) api_base: Option<String>,
    /// Workspace directory the agent may read and write.
    #[arg(short = 'w', long, global = true)]
    pub(crate) workspace: Option<PathBuf>,
    /// Stream reasoning chunks and extra diagnostics.
    #[arg(short = 'v', long, global = true)]
    pub(crate) verbose: bool,
    /// Disable ANSI colour in terminal output.
    ///
    /// Colour is off automatically when the output is not a terminal or when
    /// `NO_COLOR` is set; this flag forces it off unconditionally.
    #[arg(long = "no-color", global = true)]
    pub(crate) no_color: bool,
}

/// `run`: the harness arguments plus the binary's own additions.
#[derive(Debug, Clone, clap::Args)]
pub(crate) struct RunCliArgs {
    /// The harness `run` arguments, reused verbatim.
    #[command(flatten)]
    pub(crate) base: RunArgs,
    /// Create (or reuse) a git worktree of the workspace repository and run in it.
    ///
    /// The name is slugified the same way the `worktree` tool does it; the
    /// repository is discovered from the resolved workspace (or the working
    /// directory), and the run's workspace becomes the worktree path.
    #[arg(long, value_name = "NAME")]
    pub(crate) worktree: Option<String>,
}

/// `chat`: the harness arguments plus the binary's own additions.
#[derive(Debug, Clone, clap::Args)]
pub(crate) struct ChatCliArgs {
    /// The harness `chat` arguments, reused verbatim.
    #[command(flatten)]
    pub(crate) base: ChatArgs,
    /// Watch the workspace and report reloads on stderr while the REPL runs.
    ///
    /// Requires the `openjiuwen-harness` `notify` feature, which this binary
    /// enables; change events are debounced, reloaded, and reported as one
    /// stderr line each so the conversation stream on stdout stays untouched.
    #[arg(long)]
    pub(crate) watch: bool,
    /// Create (or reuse) a git worktree of the workspace repository and chat in it.
    #[arg(long, value_name = "NAME")]
    pub(crate) worktree: Option<String>,
}

/// Subcommands of the `openjiuwen` binary.
///
/// `run` and `chat` wrap the harness argument types so their parsing is the
/// harness's own; `auto-harness`, `lsp`, `cron`, `team` and `watch` are new here
/// and dispatched by this crate.
#[derive(Debug, Clone, clap::Subcommand)]
pub(crate) enum Command {
    /// Run a single prompt and exit.
    Run(RunCliArgs),
    /// Start the interactive chat REPL.
    Chat(ChatCliArgs),
    /// List, inspect and prune the sessions a conversation can be resumed from.
    ///
    /// Shows the CLI session records, the persistent-agent execution journals and
    /// the persistent-agent session records together, newest first; `run`/`chat`
    /// resume any of them with `--resume <ID>` or `--continue`.
    Sessions(crate::sessions::SessionsCli),
    /// Query a language server: diagnostics, navigation, call hierarchy.
    ///
    /// Drives the harness `lsp` tool from the terminal against the built-in
    /// server table (pyright, rust-analyzer, typescript, gopls, jdtls) or an
    /// explicit `--server-command`.
    Lsp(crate::lsp::LspCli),
    /// Inspect and operate the workspace cron schedule.
    ///
    /// A host-side view of the same `cron` tool the agent uses: list, create,
    /// toggle, preview, run and cancel jobs in `<workspace>/cron`.
    Cron(crate::cron::CronCli),
    /// Operate an agent team hosted by another process over its attach socket.
    ///
    /// Talks the `openjiuwen-agent-teams` attach protocol; the socket comes from
    /// `--socket` or `OPENJIUWEN_TEAM_SOCKET`.
    Team(crate::team::TeamCli),
    /// Watch the workspace and report skill, spec, instruction and MCP reloads.
    ///
    /// Prints one line (or one JSON object with `--json`) per debounced change
    /// or reload outcome, and exits on Ctrl-C or after `--max-events`.
    Watch(crate::watch::WatchCli),
    /// Auto Harness Agent: optimise the harness framework itself.
    ///
    /// Ported from Python's `openjiuwen auto-harness` group; task execution is
    /// not implemented in the Rust port, but task resolution, the experience
    /// store and the deferred surfaces are.
    AutoHarness(AutoHarnessArgs),
}

impl Cli {
    /// Builds the harness command line this invocation stands for.
    ///
    /// Every global is forwarded, including `--no-color`, so option resolution
    /// in [`openjiuwen_harness::cli::CliOptions`] sees exactly what the user
    /// typed. The binary's own `--worktree` / `--watch` additions are dropped
    /// here; [`crate::extras`] applies them around the harness call.
    /// Subcommands with no harness counterpart map to `command: None`: the
    /// harness only reads the globals there and never dispatches.
    pub(crate) fn to_harness(&self) -> openjiuwen_harness::cli::Cli {
        let command = match &self.command {
            Some(Command::Run(args)) => {
                Some(openjiuwen_harness::cli::Command::Run(args.base.clone()))
            }
            Some(Command::Chat(args)) => {
                Some(openjiuwen_harness::cli::Command::Chat(args.base.clone()))
            }
            Some(
                Command::Lsp(_)
                | Command::Cron(_)
                | Command::Team(_)
                | Command::Watch(_)
                | Command::Sessions(_)
                | Command::AutoHarness(_),
            )
            | None => None,
        };
        openjiuwen_harness::cli::Cli {
            model: self.global.model.clone(),
            provider: self.global.provider.clone(),
            api_key: self.global.api_key.clone(),
            api_base: self.global.api_base.clone(),
            workspace: self.global.workspace.clone(),
            verbose: self.global.verbose,
            no_color: self.global.no_color,
            command,
        }
    }
}

/// Runs the parsed command line.
///
/// `auto-harness`, `lsp`, `cron`, `team`, `watch` and `sessions` are executed
/// here;
/// `run`/`chat` go through [`crate::extras`], which applies the binary's own
/// flags and then delegates to [`openjiuwen_harness::cli::run_cli`], which
/// resolves options, dispatches on the terminal and runs the one-shot prompt or
/// the REPL.
///
/// # Errors
///
/// Returns whatever the harness CLI or the binary's own commands report; see
/// [`openjiuwen_harness::cli::run_cli`], [`auto_harness::run`], [`crate::lsp::run`],
/// [`crate::cron::run`], [`crate::team::run`], [`crate::watch::run`] and
/// [`crate::sessions::run`].
pub(crate) async fn run(cli: Cli) -> Result<()> {
    match &cli.command {
        Some(Command::AutoHarness(args)) => auto_harness::run(args, &cli.to_harness()).await,
        Some(Command::Lsp(args)) => crate::lsp::run(args, &cli.to_harness()).await,
        Some(Command::Cron(args)) => crate::cron::run(args, &cli.to_harness()).await,
        Some(Command::Team(args)) => crate::team::run(args).await,
        Some(Command::Watch(args)) => crate::watch::run(args, &cli.to_harness()).await,
        Some(Command::Sessions(args)) => crate::sessions::run(args, &cli.to_harness()).await,
        Some(Command::Run(args)) => crate::extras::run_run(args, &cli.to_harness()).await,
        Some(Command::Chat(args)) => crate::extras::run_chat(args, &cli.to_harness()).await,
        None => openjiuwen_harness::cli::run_cli(cli.to_harness()).await,
    }
}

#[cfg(test)]
mod tests {
    //! Grammar tests: flag propagation and the drift guard against the harness.
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use clap::CommandFactory;
    use clap::Parser;

    use super::{Cli, Command};

    /// Parses a command line, panicking with clap's diagnostic on failure.
    fn parse(args: &[&str]) -> Cli {
        Cli::try_parse_from(args).unwrap()
    }

    #[test]
    fn no_color_before_the_subcommand_survives_conversion() {
        let cli = parse(&["openjiuwen", "--no-color", "run", "hello"]);
        assert!(cli.global.no_color, "--no-color was not parsed");
        assert!(cli.to_harness().no_color, "--no-color was not forwarded");
    }

    #[test]
    fn no_color_after_the_subcommand_survives_conversion() {
        let cli = parse(&["openjiuwen", "run", "hello", "--no-color"]);
        assert!(cli.global.no_color, "--no-color was not parsed");
        assert!(cli.to_harness().no_color, "--no-color was not forwarded");
    }

    #[test]
    fn globals_are_forwarded_and_run_chat_map_across() {
        let cli = parse(&[
            "openjiuwen",
            "-m",
            "gpt-4o-mini",
            "--provider",
            "anthropic",
            "--api-key",
            "sk-test",
            "--api-base",
            "https://example.invalid",
            "-w",
            "/tmp/ws",
            "-v",
            "run",
            "hello",
            "-f",
            "json",
        ]);
        let harness = cli.to_harness();
        assert_eq!(harness.model.as_deref(), Some("gpt-4o-mini"));
        assert_eq!(harness.provider.as_deref(), Some("anthropic"));
        assert_eq!(harness.api_key.as_deref(), Some("sk-test"));
        assert_eq!(harness.api_base.as_deref(), Some("https://example.invalid"));
        assert_eq!(
            harness.workspace.as_deref(),
            Some(std::path::Path::new("/tmp/ws"))
        );
        assert!(harness.verbose);
        match harness.command {
            Some(openjiuwen_harness::cli::Command::Run(args)) => {
                assert_eq!(args.prompt.as_deref(), Some("hello"));
                assert_eq!(args.output_format.as_str(), "json");
            }
            other => panic!("expected a forwarded Run command, got {other:?}"),
        }
    }

    #[test]
    fn run_worktree_is_a_cli_flag_and_never_reaches_the_harness() {
        let cli = parse(&["openjiuwen", "run", "hello", "--worktree", "demo"]);
        match &cli.command {
            Some(Command::Run(args)) => assert_eq!(args.worktree.as_deref(), Some("demo")),
            other => panic!("expected a Run command, got {other:?}"),
        }
        match cli.to_harness().command {
            Some(openjiuwen_harness::cli::Command::Run(args)) => {
                assert_eq!(args.prompt.as_deref(), Some("hello"));
            }
            other => panic!("expected a forwarded Run command, got {other:?}"),
        }
    }

    #[test]
    fn chat_watch_is_a_cli_flag_and_never_reaches_the_harness() {
        let cli = parse(&["openjiuwen", "chat", "--watch"]);
        match &cli.command {
            Some(Command::Chat(args)) => assert!(args.watch),
            other => panic!("expected a Chat command, got {other:?}"),
        }
        assert!(matches!(
            cli.to_harness().command,
            Some(openjiuwen_harness::cli::Command::Chat(_))
        ));
    }

    #[test]
    fn harness_surface_commands_map_to_no_harness_command() {
        for args in [
            vec!["openjiuwen", "auto-harness", "history"],
            vec!["openjiuwen", "lsp", "diagnostics"],
            vec!["openjiuwen", "cron", "list"],
            vec!["openjiuwen", "team", "members"],
            vec!["openjiuwen", "watch", "--list"],
        ] {
            let cli = parse(&args);
            assert!(
                cli.to_harness().command.is_none(),
                "{args:?} must not reach the harness grammar"
            );
        }
    }

    #[test]
    fn no_subcommand_stays_none_for_tty_dispatch() {
        let cli = parse(&["openjiuwen"]);
        assert!(cli.to_harness().command.is_none());
    }

    #[test]
    fn every_harness_global_flag_exists_here() {
        let ours: Vec<String> = Cli::command()
            .get_arguments()
            .filter_map(|arg| arg.get_long().map(str::to_owned))
            .collect();
        for arg in openjiuwen_harness::cli::Cli::command().get_arguments() {
            let Some(long) = arg.get_long() else {
                continue;
            };
            if long == "help" || long == "version" {
                continue;
            }
            assert!(
                ours.iter().any(|candidate| candidate == long),
                "global `--{long}` exists in the harness grammar but not in the CLI grammar"
            );
        }
    }

    #[test]
    fn every_harness_subcommand_exists_here() {
        let ours: Vec<String> = Cli::command()
            .get_subcommands()
            .map(|command| command.get_name().to_owned())
            .collect();
        for command in openjiuwen_harness::cli::Cli::command().get_subcommands() {
            let name = command.get_name();
            assert!(
                ours.iter().any(|candidate| candidate == name),
                "subcommand `{name}` exists in the harness grammar but not in the CLI grammar"
            );
        }
        for expected in ["run", "chat"] {
            assert!(
                ours.iter().any(|candidate| candidate == expected),
                "missing `{expected}` subcommand"
            );
        }
    }
}
