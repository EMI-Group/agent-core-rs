//! Provider-neutral backend layer of the remote sandbox subtree.
//!
//! Python parity: the three providers under
//! `openjiuwen/extensions/sys_operation/sandbox/providers/` copy the same host-side
//! behavior — line selection, newline transforms, chunking, sorting, and local transfer
//! I/O — around three different wire protocols. Rust factors that behavior out once.
//!
//! A provider therefore implements only the wire primitives of [`SandboxBackend`]
//! (`read_bytes`, `write_bytes`, `list`, `search`, `execute`) and may override
//! [`SandboxBackend::code_command`] / [`SandboxBackend::code_cwd`] when its container runs
//! snippets differently. The generic wrappers [`RemoteFs`], [`RemoteShell`], and
//! [`RemoteCode`] implement the kernel's ten [`FsOperation`], four [`ShellOperation`], and
//! two [`CodeOperation`] methods on top, reproducing Python's host-side behavior exactly;
//! they live in the `fs`, `shell`, and `code` submodules of this one. [`provider_set_from`]
//! turns one backend constructor into the three factories a
//! [`SandboxProviderSet`] needs.
//!
//! Failures stay values: operation-level failures are [`OpResult`] codes built with the
//! kernel's `fs_failure` / `shell_failure` / `code_failure` helpers, and transport
//! failures map through [`gateway_failure`] to the kernel's sandbox gateway status
//! (`199007`).

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use openjiuwen_common::error::Result;
use openjiuwen_core::sys_operation::code::types::Language;
use openjiuwen_core::sys_operation::operation::{CodeOperation, FsOperation, ShellOperation};

use super::semantics::posix_quote;
use super::transport::{base64_encode, RemoteError};
use crate::sys_operation::{SandboxEndpoint, SandboxGatewayConfig, SandboxProviderSet};

mod code;
mod fs;
mod shell;

#[cfg(test)]
mod tests;

pub(crate) use code::RemoteCode;
pub(crate) use fs::RemoteFs;
pub(crate) use shell::RemoteShell;

/// How a command reaches the remote container.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ExecCommand {
    /// A program and its arguments, passed without a shell (`JiuwenBox` exec, POSIX-quoted for
    /// string-only transports).
    Argv(Vec<String>),
    /// A command line the remote shell interprets.
    Shell(String),
}

impl ExecCommand {
    /// Builds an argument vector from anything string-like.
    #[must_use]
    pub(crate) fn argv<I, S>(argv: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self::Argv(argv.into_iter().map(Into::into).collect())
    }

    /// Builds a shell command line.
    #[must_use]
    pub(crate) fn shell(command: impl Into<String>) -> Self {
        Self::Shell(command.into())
    }

    /// Renders the command as a shell line: `Shell` stays verbatim while `Argv` is quoted
    /// with [`posix_quote`] and joined with single spaces.
    #[must_use]
    pub(crate) fn to_shell_command(&self) -> String {
        match self {
            Self::Shell(command) => command.clone(),
            Self::Argv(argv) => argv
                .iter()
                .map(|argument| posix_quote(argument))
                .collect::<Vec<_>>()
                .join(" "),
        }
    }
}

/// One remote execution request.
#[derive(Debug, Clone)]
pub(crate) struct ExecRequest {
    /// Command to run.
    pub command: ExecCommand,
    /// Working directory; `None` uses the container's default.
    pub cwd: Option<String>,
    /// Environment overlay; `None` removes a variable.
    pub environment: BTreeMap<String, Option<String>>,
    /// Standard input text, when the transport supports it.
    pub stdin: Option<String>,
    /// Timeout in seconds, when the transport enforces one.
    pub timeout_secs: Option<u64>,
}

impl ExecRequest {
    /// A request running `command` with no cwd, environment, stdin, or timeout.
    #[must_use]
    pub(crate) fn new(command: ExecCommand) -> Self {
        Self {
            command,
            cwd: None,
            environment: BTreeMap::new(),
            stdin: None,
            timeout_secs: None,
        }
    }
}

/// Captured result of one remote execution.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct ExecResult {
    /// Exit status reported by the container.
    pub exit_code: i32,
    /// Captured standard output.
    pub stdout: String,
    /// Captured standard error.
    pub stderr: String,
    /// Whether the container reported the command as timed out.
    pub timed_out: bool,
}

/// One entry a remote directory listing or search returned.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct RemoteEntry {
    /// Entry name (file or directory name).
    pub name: String,
    /// Absolute path of the entry.
    pub path: String,
    /// Size in bytes.
    pub size: u64,
    /// Modification time, formatted like a Python datetime.
    pub modified_time: String,
    /// Whether the entry is a directory.
    pub is_directory: bool,
    /// File suffix including the dot (`".txt"`), absent for directories.
    pub file_type: Option<String>,
}

/// A remote directory listing request.
#[derive(Debug, Clone)]
pub(crate) struct ListRequest {
    /// Directory to list.
    pub path: String,
    /// Recurse into subdirectories.
    pub recursive: bool,
    /// Maximum recursion depth; `None` means unlimited when `recursive`.
    pub max_depth: Option<usize>,
    /// Include regular files.
    pub include_files: bool,
    /// Include directories.
    pub include_dirs: bool,
}

/// A remote search request.
#[derive(Debug, Clone)]
pub(crate) struct SearchRequest {
    /// Directory to search under.
    pub path: String,
    /// Glob pattern to match.
    pub pattern: String,
    /// Globs of paths to exclude.
    pub exclude_patterns: Option<Vec<String>>,
}

/// The wire primitives every remote sandbox provider implements.
///
/// Providers translate these five calls plus, optionally, [`Self::code_command`] and
/// [`Self::code_cwd`]; the generic wrappers in this module reproduce Python's host-side
/// filesystem, shell, and code behavior on top of them.
#[async_trait]
pub(crate) trait SandboxBackend: Send + Sync + 'static {
    /// The sandbox type this backend serves (`"aio"`, `"jiuwenbox"`, `"yuanrong"`, …).
    fn provider_name(&self) -> &'static str;

    /// Reads the whole remote file.
    async fn read_bytes(&self, path: &str) -> std::result::Result<Vec<u8>, RemoteError>;

    /// Writes `data` to `path`, overwriting when `!append` and appending otherwise.
    ///
    /// Parent directories are the backend's job, mirroring Python's providers.
    async fn write_bytes(
        &self,
        path: &str,
        data: &[u8],
        append: bool,
    ) -> std::result::Result<(), RemoteError>;

    /// Lists a remote directory.
    async fn list(
        &self,
        request: &ListRequest,
    ) -> std::result::Result<Vec<RemoteEntry>, RemoteError>;

    /// Searches a remote directory for files matching a glob.
    async fn search(
        &self,
        request: &SearchRequest,
    ) -> std::result::Result<Vec<RemoteEntry>, RemoteError>;

    /// Runs a command inside the container.
    async fn execute(&self, request: &ExecRequest) -> std::result::Result<ExecResult, RemoteError>;

    /// Whether this backend can honour a conditional write.
    ///
    /// `create_if_absent` and `expected_content_sha256` are evaluated host-side
    /// by [`RemoteFs`]: it reads the current remote bytes under a per-path
    /// in-process lock, applies the precondition, and only then writes. A
    /// backend opts in when a read and a write of the same path reach the same
    /// container view; the default keeps both options rejected with a clear
    /// `fs_failure` instead of ignoring them.
    fn supports_conditional_write(&self) -> bool {
        false
    }

    /// The command that runs a code snippet.
    ///
    /// The default is the inline form
    /// `[language.executable(), language.inline_flags().., code]`
    /// (`python3 -u -c`, `node -e`); `force_file` writes the snippet to a temporary file
    /// first, exactly like Python's providers.
    fn code_command(&self, code: &str, language: Language, force_file: bool) -> ExecCommand {
        if force_file {
            let encoded = posix_quote(&base64_encode(code.as_bytes()));
            let suffix = language.file_suffix();
            let executable = language.executable();
            return ExecCommand::shell(format!(
                "tmp=$(mktemp /tmp/ojw_code_XXXXXX{suffix}) && printf %s {encoded} | base64 -d > \"$tmp\" && {executable} \"$tmp\"; status=$?; rm -f \"$tmp\"; exit $status"
            ));
        }
        let mut argv = Vec::with_capacity(language.inline_flags().len() + 2);
        argv.push(language.executable().to_string());
        argv.extend(
            language
                .inline_flags()
                .iter()
                .map(|flag| (*flag).to_string()),
        );
        argv.push(code.to_string());
        ExecCommand::argv(argv)
    }

    /// The working directory a code snippet runs in.
    ///
    /// The default is `/tmp`, where Python runs remote snippets.
    fn code_cwd(&self, _caller: Option<&str>) -> String {
        "/tmp".to_string()
    }

    /// Starts `request` in the background and returns its pid.
    ///
    /// The provided implementation runs
    /// `{ <command>; } >/dev/null 2>&1 & echo $!` through [`Self::execute`] (timeout 10 s,
    /// same cwd and environment) and parses the pid from stdout. A non-zero exit, an
    /// unparsable pid, or a transport error becomes [`RemoteError::Status`] with status `0`.
    async fn spawn_background(
        &self,
        request: &ExecRequest,
    ) -> std::result::Result<u32, RemoteError> {
        let command = request.command.to_shell_command();
        let wrapped = ExecRequest {
            cwd: request.cwd.clone(),
            environment: request.environment.clone(),
            timeout_secs: Some(10),
            ..ExecRequest::new(ExecCommand::shell(format!(
                "{{ {command}; }} >/dev/null 2>&1 & echo $!"
            )))
        };
        let result = match self.execute(&wrapped).await {
            Ok(result) => result,
            Err(error) => {
                return Err(RemoteError::Status {
                    status: 0,
                    message: format!("background command failed: {}", error.message()),
                })
            }
        };
        if result.exit_code != 0 {
            return Err(RemoteError::Status {
                status: 0,
                message: format!(
                    "background command failed with exit code {}: {}",
                    result.exit_code,
                    result.stderr.trim()
                ),
            });
        }
        let stdout = result.stdout.trim();
        stdout.parse::<u32>().map_err(|error| RemoteError::Status {
            status: 0,
            message: format!("could not parse the background pid from '{stdout}': {error}"),
        })
    }
}

/// Builds a full three-factory [`SandboxProviderSet`] from a backend constructor.
///
/// Each factory calls `constructor(endpoint, config)` and wraps the result in the matching
/// generic wrapper, so a provider module only supplies its wire backend.
pub(crate) fn provider_set_from<B, F>(constructor: F) -> SandboxProviderSet
where
    B: SandboxBackend,
    F: Fn(&SandboxEndpoint, &SandboxGatewayConfig) -> Result<B> + Send + Sync + 'static,
{
    let constructor = Arc::new(constructor);
    let fs_constructor = Arc::clone(&constructor);
    let shell_constructor = Arc::clone(&constructor);
    let code_constructor = constructor;
    SandboxProviderSet::new()
        .with_fs(Arc::new(
            move |endpoint: &SandboxEndpoint, config: &SandboxGatewayConfig| {
                let backend = Arc::new(fs_constructor(endpoint, config)?);
                let operation: Arc<dyn FsOperation> = Arc::new(RemoteFs::new(backend));
                Ok(operation)
            },
        ))
        .with_shell(Arc::new(
            move |endpoint: &SandboxEndpoint, config: &SandboxGatewayConfig| {
                let backend = Arc::new(shell_constructor(endpoint, config)?);
                let operation: Arc<dyn ShellOperation> = Arc::new(RemoteShell::new(backend));
                Ok(operation)
            },
        ))
        .with_code(Arc::new(
            move |endpoint: &SandboxEndpoint, config: &SandboxGatewayConfig| {
                let backend = Arc::new(code_constructor(endpoint, config)?);
                let operation: Arc<dyn CodeOperation> = Arc::new(RemoteCode::new(backend));
                Ok(operation)
            },
        ))
}

/// Converts an elapsed duration to whole milliseconds.
fn elapsed_ms(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
}

/// Converts a length or index to the `u64` the result types use.
fn len_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}
