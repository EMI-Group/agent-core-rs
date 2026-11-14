//! `openjiuwen lsp` — the harness language-server tool, driven from the terminal.
//!
//! The harness owns the LSP client, the server table and the eleven actions
//! ([`openjiuwen_harness::lsp`]); this module only translates a command line into
//! an [`LspTool`] call and renders the payload. It never speaks JSON-RPC itself.
//!
//! The tool is the same one the model would call, so the terminal sees exactly
//! the model's answers: one payload per action, `success=false` with a reason on
//! an expected failure. Text mode prints the payload's rendered `result`;
//! `--json` prints the whole payload document.
use std::path::PathBuf;

use clap::ValueEnum;
use openjiuwen_common::error::Result;
use openjiuwen_harness::lsp::LspAction;

/// The eleven `lsp` actions, in the tool's own order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "kebab-case")]
pub(crate) enum LspActionArg {
    /// Type information at a position.
    Hover,
    /// Where a symbol is defined.
    Definition,
    /// Every reference to a symbol.
    References,
    /// Completion candidates at a position.
    Completion,
    /// Symbols declared in one document.
    Symbols,
    /// Symbols matching a query across the workspace.
    WorkspaceSymbols,
    /// Push and pull diagnostics for one document.
    Diagnostics,
    /// Implementations of an interface or trait method.
    Implementation,
    /// The call-hierarchy item at a position.
    PrepareCallHierarchy,
    /// Callers of the function at a position.
    IncomingCalls,
    /// Callees of the function at a position.
    OutgoingCalls,
}

impl LspActionArg {
    /// Maps the command-line spelling onto the tool's action enum.
    pub(crate) const fn action(self) -> LspAction {
        match self {
            Self::Hover => LspAction::Hover,
            Self::Definition => LspAction::Definition,
            Self::References => LspAction::References,
            Self::Completion => LspAction::Completion,
            Self::Symbols => LspAction::Symbols,
            Self::WorkspaceSymbols => LspAction::WorkspaceSymbols,
            Self::Diagnostics => LspAction::Diagnostics,
            Self::Implementation => LspAction::Implementation,
            Self::PrepareCallHierarchy => LspAction::PrepareCallHierarchy,
            Self::IncomingCalls => LspAction::IncomingCalls,
            Self::OutgoingCalls => LspAction::OutgoingCalls,
        }
    }
}

/// `openjiuwen lsp <ACTION> [OPTIONS]`.
#[derive(Debug, Clone, clap::Args)]
pub(crate) struct LspCli {
    /// The action to run.
    #[arg(value_enum)]
    pub(crate) action: LspActionArg,
    /// File to act on, relative to the workspace or absolute inside it.
    ///
    /// Required by every action except `workspace-symbols` and `diagnostics`
    /// without a file, which reports the buffered diagnostics of all files.
    #[arg(long, value_name = "PATH")]
    pub(crate) file: Option<PathBuf>,
    /// One-based line of the position (hover, definition, references, …).
    #[arg(long)]
    pub(crate) line: Option<u32>,
    /// One-based column of the position.
    #[arg(long, alias = "col")]
    pub(crate) column: Option<u32>,
    /// Query text for `workspace-symbols`.
    #[arg(long)]
    pub(crate) query: Option<String>,
    /// Context lines to include around a position result.
    #[arg(long = "context-lines")]
    pub(crate) context_lines: Option<u32>,
    /// Built-in server to use, by name (`pyright`, `rust`, `typescript`, `gopls`, `jdtls`).
    #[arg(long, value_name = "NAME")]
    pub(crate) server: Option<String>,
    /// Explicit language-server program to launch instead of a built-in entry.
    #[arg(long = "server-command", value_name = "PROGRAM")]
    pub(crate) server_command: Option<String>,
    /// Argument for `--server-command`; repeatable.
    #[arg(long = "server-arg", value_name = "ARG")]
    pub(crate) server_args: Vec<String>,
    /// LSP language id for `--server-command` (for example `rust`, `python`).
    #[arg(long = "language-id", value_name = "ID")]
    pub(crate) language_id: Option<String>,
    /// Print the payload as one JSON document instead of text.
    #[arg(long)]
    pub(crate) json: bool,
}

/// Runs `openjiuwen lsp`.
///
/// # Errors
///
/// Returns the action's failure as a typed error (exit `1`) when the tool reports
/// `success=false`, and option-resolution errors from the harness.
pub(crate) async fn run(args: &LspCli, base: &openjiuwen_harness::cli::Cli) -> Result<()> {
    let _ = (args, base);
    Err(openjiuwen_common::error::Error::unsupported(
        "lsp command is not wired yet",
    ))
}
