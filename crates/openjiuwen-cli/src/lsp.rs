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
use std::path::{Path, PathBuf};

use clap::ValueEnum;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_core::foundation::tool::{Tool, ToolContext};
use openjiuwen_harness::lsp::{
    builtin_server_configs, LspAction, LspArgs, LspServerConfig, LspTool, LspToolConfig,
};
use openjiuwen_harness::tools::ToolPayload;

/// The built-in server names, as they are offered in the unknown-name diagnostic.
const BUILTIN_SERVERS: &str = "pyright, rust, typescript, gopls, jdtls";

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
/// The command resolves the harness options once (the workspace decides both the
/// server root and how a relative `--file` is read), rejects a missing file before
/// any server is spawned, then drives [`LspTool`] exactly as the model would and
/// renders its single payload: the rendered `result` text by default, the whole
/// document with `--json`.
///
/// # Errors
///
/// Returns [`Error::not_found`] for a `--file` that does not exist, option
/// resolution errors from the harness, [`Error::validation`] for an unknown
/// `--server` name, and the action's failure as [`Error::tool`] (exit `1`) when
/// the tool reports `success=false`.
pub(crate) async fn run(args: &LspCli, base: &openjiuwen_harness::cli::Cli) -> Result<()> {
    let options = openjiuwen_harness::cli::CliOptions::resolve(base)?;
    if let Some(file) = &args.file {
        ensure_exists(&resolve_against(&options.workspace, file)).await?;
    }

    let tool = LspTool::new(server_config(args, &options.workspace)?);
    let input = serde_json::to_value(tool_args(args))?;
    // `shutdown` closes any language server the call started, so it runs on the
    // failure path too; the tool never leaves a child process behind.
    let outcome = tool.invoke(input, &ToolContext::new()).await;
    tool.shutdown().await;

    let raw = outcome?;
    let payload: ToolPayload = serde_json::from_value(raw.clone())?;
    if !payload.success {
        return Err(Error::tool(
            payload
                .error
                .unwrap_or_else(|| "lsp action failed".to_owned()),
        ));
    }
    print_payload(&payload, &raw, args.json)
}

/// Resolves `file` under `workspace` unless it is already absolute.
fn resolve_against(workspace: &Path, file: &Path) -> PathBuf {
    if file.is_absolute() {
        file.to_path_buf()
    } else {
        workspace.join(file)
    }
}

/// Fails with [`Error::not_found`] unless `path` exists.
///
/// The check runs before the tool is built so a typo cannot launch (or probe for)
/// a language server.
async fn ensure_exists(path: &Path) -> Result<()> {
    match tokio::fs::metadata(path).await {
        Ok(_) => Ok(()),
        Err(_) => Err(Error::not_found(format!(
            "file not found: {}",
            path.display()
        ))),
    }
}

/// Builds the tool configuration: explicit program, named built-in, or all built-ins.
fn server_config(args: &LspCli, workspace: &Path) -> Result<LspToolConfig> {
    let config = LspToolConfig::default()
        .with_workspace_root(workspace.to_path_buf())
        .with_language("en");
    let config = match (&args.server_command, &args.server) {
        // An explicit program wins over a named built-in: it describes exactly
        // what to launch, so it cannot be ambiguous.
        (Some(program), _) => {
            let mut server = LspServerConfig::new(program.clone())
                .with_name("cli")
                .with_args(args.server_args.clone())
                .with_root(workspace.to_path_buf());
            if let Some(language_id) = &args.language_id {
                server = server.with_language_id(language_id.clone());
            }
            config.with_server(server)
        }
        (None, Some(name)) => config.with_server(builtin_server(name)?),
        (None, None) => config.with_builtin_servers(),
    };
    // Without an explicit program the language id is a tool-level hint used when
    // the resolved server config does not carry one.
    match (&args.server_command, &args.language_id) {
        (None, Some(language_id)) => Ok(config.with_language_id(language_id.clone())),
        _ => Ok(config),
    }
}

/// Looks a built-in server up by its exact name.
fn builtin_server(name: &str) -> Result<LspServerConfig> {
    builtin_server_configs()
        .into_iter()
        .find(|config| config.name == *name)
        .ok_or_else(|| {
            Error::validation(format!(
                "unknown lsp server '{name}'; built in servers: {BUILTIN_SERVERS}"
            ))
        })
}

/// Translates the parsed command line into the tool's own argument type.
///
/// `path` is the spelling the user typed: the tool resolves it against the
/// workspace root it was configured with, so both forms reach the same file.
fn tool_args(args: &LspCli) -> LspArgs {
    LspArgs {
        action: args.action.action(),
        path: args
            .file
            .as_ref()
            .map(|file| file.to_string_lossy().into_owned()),
        line: args.line,
        column: args.column,
        query: args.query.clone(),
        context_lines: args.context_lines,
    }
}

/// Writes the payload to stdout: one JSON document with `json`, else human text.
///
/// The harness renders the action's human text into `data.result`; when an action
/// has no textual rendering the data is printed as pretty JSON instead. Nothing
/// but the payload reaches stdout, so a caller can pipe either form.
fn print_payload(payload: &ToolPayload, raw: &serde_json::Value, json: bool) -> Result<()> {
    if json {
        println!("{}", serde_json::to_string_pretty(raw)?);
        return Ok(());
    }
    let data = payload.data.clone().unwrap_or(serde_json::Value::Null);
    match data.get("result").and_then(serde_json::Value::as_str) {
        Some(result) if !result.trim().is_empty() => println!("{result}"),
        _ => println!("{}", serde_json::to_string_pretty(&data)?),
    }
    Ok(())
}
