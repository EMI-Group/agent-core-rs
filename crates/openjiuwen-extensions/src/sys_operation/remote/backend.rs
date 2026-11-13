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
//! two [`CodeOperation`] methods on top, reproducing Python's host-side behavior exactly.
//! [`provider_set_from`] turns one backend constructor into the three factories a
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
use futures::StreamExt;
use openjiuwen_common::error::Result;
use openjiuwen_core::sys_operation::code::types::{
    CodeChunk, CodeExecParams, CodeOutput, Language,
};
use openjiuwen_core::sys_operation::constants::DEFAULT_ENCODING;
use openjiuwen_core::sys_operation::fs::types::{
    DownloadFileData, DownloadFileParams, FileContent, FileSystemItem, ListDirsParams,
    ListFilesData, ListFilesParams, ReadChunkData, ReadFileData, ReadFileParams, ReadMode,
    SearchFilesData, SearchFilesParams, SortBy, TransferChunkData, UploadFileData,
    UploadFileParams, WriteFileData, WriteFileParams,
};
use openjiuwen_core::sys_operation::operation::{
    CodeOperation, FsOperation, OpStream, ShellOperation,
};
use openjiuwen_core::sys_operation::result::{code_failure, fs_failure, shell_failure, OpResult};
use openjiuwen_core::sys_operation::shell::types::{
    ShellArgvParams, ShellBackgroundOutput, ShellChunk, ShellExecParams, ShellOutput, StreamKind,
};
use openjiuwen_core::sys_operation::status::SysOpStatus;

use super::local_io::{read_host_file, write_host_file, HostIoError};
use super::semantics::{
    apply_text_newlines, byte_chunks, filter_file_types, normalize_line_selection, posix_quote,
    select_text_lines, sort_items, text_lines,
};
use super::transport::{base64_encode, gateway_failure, RemoteError};
use crate::sys_operation::{SandboxEndpoint, SandboxGatewayConfig, SandboxProviderSet};

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
            return ExecCommand::Shell(format!(
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
        ExecCommand::Argv(argv)
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
            command: ExecCommand::Shell(format!("{{ {command}; }} >/dev/null 2>&1 & echo $!")),
            cwd: request.cwd.clone(),
            environment: request.environment.clone(),
            stdin: None,
            timeout_secs: Some(10),
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

/// The kernel's ten [`FsOperation`] methods implemented over a [`SandboxBackend`].
pub(crate) struct RemoteFs<B: SandboxBackend> {
    backend: Arc<B>,
    conditional_write: bool,
    provider: &'static str,
}

impl<B: SandboxBackend> RemoteFs<B> {
    /// Builds the wrapper; the conditional-write flag comes from the gateway configuration.
    #[must_use]
    pub(crate) fn new(backend: Arc<B>, config: &SandboxGatewayConfig) -> Self {
        Self {
            provider: backend.provider_name(),
            backend,
            conditional_write: config.supports_conditional_write,
        }
    }
}

/// The kernel's four [`ShellOperation`] methods implemented over a [`SandboxBackend`].
pub(crate) struct RemoteShell<B: SandboxBackend> {
    backend: Arc<B>,
    provider: &'static str,
}

impl<B: SandboxBackend> RemoteShell<B> {
    /// Builds the wrapper.
    #[must_use]
    pub(crate) fn new(backend: Arc<B>) -> Self {
        Self {
            provider: backend.provider_name(),
            backend,
        }
    }
}

/// The kernel's two [`CodeOperation`] methods implemented over a [`SandboxBackend`].
pub(crate) struct RemoteCode<B: SandboxBackend> {
    backend: Arc<B>,
    provider: &'static str,
}

impl<B: SandboxBackend> RemoteCode<B> {
    /// Builds the wrapper.
    #[must_use]
    pub(crate) fn new(backend: Arc<B>) -> Self {
        Self {
            provider: backend.provider_name(),
            backend,
        }
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
                let operation: Arc<dyn FsOperation> = Arc::new(RemoteFs::new(backend, config));
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

#[async_trait]
impl<B: SandboxBackend> FsOperation for RemoteFs<B> {
    fn supports_conditional_write(&self) -> bool {
        self.conditional_write
    }

    async fn read_file(&self, params: ReadFileParams) -> OpResult<ReadFileData> {
        let selection = match normalize_line_selection(
            params.mode,
            params.head,
            params.tail,
            params.line_range,
        ) {
            Ok(selection) => selection,
            Err(reason) => return fs_failure("read_file", &reason),
        };
        let bytes = match self.backend.read_bytes(&params.path).await {
            Ok(bytes) => bytes,
            Err(error) => return gateway_failure("read_file", &error),
        };
        let total_bytes = len_u64(bytes.len());
        match params.mode {
            ReadMode::Text => {
                let text = String::from_utf8_lossy(&bytes);
                let content = select_text_lines(&text, &selection);
                OpResult::success(ReadFileData {
                    path: params.path,
                    content: FileContent::Text(content),
                    mode: ReadMode::Text,
                    encoding: (params.encoding != DEFAULT_ENCODING)
                        .then(|| params.encoding.clone()),
                    truncated: false,
                    total_bytes,
                })
            }
            ReadMode::Bytes => OpResult::success(ReadFileData {
                path: params.path,
                content: FileContent::Bytes(bytes),
                mode: ReadMode::Bytes,
                encoding: None,
                truncated: false,
                total_bytes,
            }),
        }
    }

    fn read_file_stream(self: Arc<Self>, params: ReadFileParams) -> OpStream<ReadChunkData> {
        Box::pin(
            futures::stream::once(async move {
                let path = params.path.clone();
                let mode = params.mode;
                let chunk_size = params.chunk_size;
                match self.read_file(params).await {
                    OpResult {
                        code: 0,
                        message: _,
                        data: Some(data),
                    } => read_chunks(data, chunk_size),
                    failure => vec![OpResult {
                        code: failure.code,
                        message: failure.message,
                        data: Some(ReadChunkData {
                            path,
                            mode,
                            is_last_chunk: true,
                            ..ReadChunkData::default()
                        }),
                    }],
                }
            })
            .flat_map(futures::stream::iter),
        )
    }

    async fn write_file(&self, params: WriteFileParams) -> OpResult<WriteFileData> {
        if params.expected_content_sha256.is_some() {
            return fs_failure(
                "write_file",
                &unsupported_conditional(self.provider, "expected_content_sha256"),
            );
        }
        if params.create_if_absent {
            return fs_failure(
                "write_file",
                &unsupported_conditional(self.provider, "create_if_absent"),
            );
        }
        let mode = params.content.mode();
        let payload = match &params.content {
            FileContent::Text(text) => {
                apply_text_newlines(text, params.prepend_newline, params.append_newline)
                    .into_bytes()
            }
            FileContent::Bytes(bytes) => bytes.clone(),
        };
        if !params.create_if_not_exist {
            match self.backend.read_bytes(&params.path).await {
                Ok(_) => {}
                Err(error) if error.is_not_found() => {
                    return fs_failure(
                        "write_file",
                        &format!("File does not exist: {}", params.path),
                    )
                }
                Err(error) => return gateway_failure("write_file", &error),
            }
        }
        let size = len_u64(payload.len());
        match self
            .backend
            .write_bytes(&params.path, &payload, params.append)
            .await
        {
            Ok(()) => OpResult::success(WriteFileData {
                path: params.path,
                size,
                mode,
            }),
            Err(error) => gateway_failure("write_file", &error),
        }
    }

    async fn upload_file(&self, params: UploadFileParams) -> OpResult<UploadFileData> {
        let bytes = match read_host_file(&params.local_path).await {
            Ok(bytes) => bytes,
            Err(reason) => {
                return fs_failure(
                    "upload_file",
                    &format!(
                        "failed to read local file '{}': {reason}",
                        params.local_path
                    ),
                )
            }
        };
        if !params.overwrite {
            match self.backend.read_bytes(&params.target_path).await {
                Ok(_) => {
                    return fs_failure(
                        "upload_file",
                        &format!("File already exists: {}", params.target_path),
                    )
                }
                Err(error) if error.is_not_found() => {}
                Err(error) => return gateway_failure("upload_file", &error),
            }
        }
        let size = len_u64(bytes.len());
        match self
            .backend
            .write_bytes(&params.target_path, &bytes, false)
            .await
        {
            Ok(()) => OpResult::success(UploadFileData {
                local_path: params.local_path,
                target_path: params.target_path,
                size,
            }),
            Err(error) => gateway_failure("upload_file", &error),
        }
    }

    fn upload_file_stream(
        self: Arc<Self>,
        params: UploadFileParams,
    ) -> OpStream<TransferChunkData> {
        Box::pin(
            futures::stream::once(async move {
                let source_path = params.local_path.clone();
                let target_path = params.target_path.clone();
                match self.upload_file(params).await {
                    OpResult {
                        code: 0,
                        message: _,
                        data,
                    } => {
                        let size = data.map_or(0, |data| data.size);
                        vec![OpResult::success(TransferChunkData {
                            source_path,
                            target_path,
                            chunk_size: size,
                            chunk_index: 0,
                            is_last_chunk: true,
                        })]
                    }
                    failure => vec![OpResult {
                        code: failure.code,
                        message: failure.message,
                        data: Some(TransferChunkData {
                            source_path,
                            target_path,
                            is_last_chunk: true,
                            ..TransferChunkData::default()
                        }),
                    }],
                }
            })
            .flat_map(futures::stream::iter),
        )
    }

    async fn download_file(&self, params: DownloadFileParams) -> OpResult<DownloadFileData> {
        let bytes = match self.backend.read_bytes(&params.source_path).await {
            Ok(bytes) => bytes,
            Err(error) => return gateway_failure("download_file", &error),
        };
        let size = len_u64(bytes.len());
        match write_host_file(
            &params.local_path,
            &bytes,
            params.create_parent_dirs,
            params.overwrite,
        )
        .await
        {
            Ok(()) => OpResult::success(DownloadFileData {
                source_path: params.source_path,
                local_path: params.local_path,
                size,
            }),
            Err(HostIoError::AlreadyExists) => fs_failure(
                "download_file",
                &format!("File already exists: {}", params.local_path),
            ),
            Err(HostIoError::Io(reason)) => fs_failure("download_file", &reason),
        }
    }

    fn download_file_stream(
        self: Arc<Self>,
        params: DownloadFileParams,
    ) -> OpStream<TransferChunkData> {
        Box::pin(
            futures::stream::once(async move {
                let source_path = params.source_path.clone();
                let target_path = params.local_path.clone();
                match self.download_file(params).await {
                    OpResult {
                        code: 0,
                        message: _,
                        data,
                    } => {
                        let size = data.map_or(0, |data| data.size);
                        vec![OpResult::success(TransferChunkData {
                            source_path,
                            target_path,
                            chunk_size: size,
                            chunk_index: 0,
                            is_last_chunk: true,
                        })]
                    }
                    failure => vec![OpResult {
                        code: failure.code,
                        message: failure.message,
                        data: Some(TransferChunkData {
                            source_path,
                            target_path,
                            is_last_chunk: true,
                            ..TransferChunkData::default()
                        }),
                    }],
                }
            })
            .flat_map(futures::stream::iter),
        )
    }

    async fn list_files(&self, params: ListFilesParams) -> OpResult<ListFilesData> {
        let request = ListRequest {
            path: params.path.clone(),
            recursive: params.recursive,
            max_depth: params.max_depth,
            include_files: true,
            include_dirs: false,
        };
        let entries = match self.backend.list(&request).await {
            Ok(entries) => entries,
            Err(error) => return gateway_failure("list_files", &error),
        };
        let mut items: Vec<FileSystemItem> = filter_file_types(
            entries.into_iter().map(file_system_item).collect(),
            &params.file_types,
        );
        sort_items(&mut items, params.sort_by, params.sort_descending);
        OpResult::success(ListFilesData {
            total_count: items.len(),
            list_items: items,
            root_path: params.path,
            recursive: params.recursive,
            max_depth: params.max_depth,
        })
    }

    async fn list_directories(&self, params: ListDirsParams) -> OpResult<ListFilesData> {
        let request = ListRequest {
            path: params.path.clone(),
            recursive: params.recursive,
            max_depth: params.max_depth,
            include_files: false,
            include_dirs: true,
        };
        let entries = match self.backend.list(&request).await {
            Ok(entries) => entries,
            Err(error) => return gateway_failure("list_directories", &error),
        };
        let mut items: Vec<FileSystemItem> = entries
            .into_iter()
            .map(|entry| {
                let mut item = file_system_item(entry);
                if item.is_directory {
                    item.file_type = None;
                }
                item
            })
            .collect();
        sort_items(&mut items, params.sort_by, params.sort_descending);
        OpResult::success(ListFilesData {
            total_count: items.len(),
            list_items: items,
            root_path: params.path,
            recursive: params.recursive,
            max_depth: params.max_depth,
        })
    }

    async fn search_files(&self, params: SearchFilesParams) -> OpResult<SearchFilesData> {
        let request = SearchRequest {
            path: params.path.clone(),
            pattern: params.pattern.clone(),
            exclude_patterns: params.exclude_patterns.clone(),
        };
        let entries = match self.backend.search(&request).await {
            Ok(entries) => entries,
            Err(error) => return gateway_failure("search_files", &error),
        };
        let mut items: Vec<FileSystemItem> = entries.into_iter().map(file_system_item).collect();
        sort_items(&mut items, SortBy::Name, false);
        OpResult::success(SearchFilesData {
            total_matches: items.len(),
            matching_files: items,
            search_path: params.path,
            search_pattern: params.pattern,
            exclude_patterns: params.exclude_patterns,
        })
    }
}

#[async_trait]
impl<B: SandboxBackend> ShellOperation for RemoteShell<B> {
    async fn execute_cmd(&self, params: ShellExecParams) -> OpResult<ShellOutput> {
        if params.command.trim().is_empty() {
            return shell_failure("execute_cmd", "command can not be empty");
        }
        let timeout = params.effective_timeout_secs(300);
        let cwd = normalized_cwd(params.cwd.as_deref());
        let request = ExecRequest {
            command: ExecCommand::Shell(params.command.clone()),
            cwd: cwd.clone(),
            environment: params.environment.clone(),
            stdin: None,
            timeout_secs: Some(timeout),
        };
        let started = Instant::now();
        let result = match self.backend.execute(&request).await {
            Ok(result) => result,
            Err(error) => return shell_failure("execute_cmd", &error.message()),
        };
        let output = ShellOutput {
            command: params.command,
            cwd: cwd.unwrap_or_else(|| ".".to_string()),
            exit_code: result.exit_code,
            stdout: result.stdout,
            stderr: result.stderr,
            truncated: false,
            timed_out: result.timed_out,
            cancelled: false,
            duration_ms: elapsed_ms(started),
        };
        if output.timed_out {
            return OpResult {
                code: SysOpStatus::ShellExecutionError.code(),
                message: format!("execution timeout after {timeout} seconds"),
                data: Some(output),
            };
        }
        OpResult::success(output)
    }

    fn execute_cmd_stream(self: Arc<Self>, params: ShellExecParams) -> OpStream<ShellChunk> {
        Box::pin(
            futures::stream::once(async move {
                match self.execute_cmd(params).await {
                    OpResult {
                        code: 0,
                        message: _,
                        data: Some(output),
                    } => shell_chunks(&output),
                    failure => vec![OpResult {
                        code: failure.code,
                        message: failure.message,
                        data: Some(ShellChunk::terminal(0, -1)),
                    }],
                }
            })
            .flat_map(futures::stream::iter),
        )
    }

    async fn execute_cmd_background(
        &self,
        params: ShellExecParams,
    ) -> OpResult<ShellBackgroundOutput> {
        if params.command.trim().is_empty() {
            return shell_failure("execute_cmd_background", "command can not be empty");
        }
        let timeout = params.effective_timeout_secs(300);
        let cwd = normalized_cwd(params.cwd.as_deref());
        let request = ExecRequest {
            command: ExecCommand::Shell(params.command.clone()),
            cwd: cwd.clone(),
            environment: params.environment,
            stdin: None,
            timeout_secs: Some(timeout),
        };
        match self.backend.spawn_background(&request).await {
            Ok(pid) => OpResult::success(ShellBackgroundOutput {
                command: params.command,
                cwd: cwd.unwrap_or_else(|| ".".to_string()),
                pid: Some(pid),
                error: None,
            }),
            Err(error) => shell_failure("execute_cmd_background", &error.message()),
        }
    }

    async fn execute_argv(&self, params: ShellArgvParams) -> OpResult<ShellOutput> {
        if params.argv.is_empty() {
            return shell_failure("execute_argv", "argv can not be empty");
        }
        let timeout = params.effective_timeout_secs(300);
        let cwd = normalized_cwd(params.cwd.as_deref());
        let command = ExecCommand::Argv(params.argv);
        let command_line = command.to_shell_command();
        let request = ExecRequest {
            command,
            cwd: cwd.clone(),
            environment: params.environment,
            stdin: None,
            timeout_secs: Some(timeout),
        };
        let started = Instant::now();
        let result = match self.backend.execute(&request).await {
            Ok(result) => result,
            Err(error) => return shell_failure("execute_argv", &error.message()),
        };
        let output = ShellOutput {
            command: command_line,
            cwd: cwd.unwrap_or_else(|| ".".to_string()),
            exit_code: result.exit_code,
            stdout: result.stdout,
            stderr: result.stderr,
            truncated: false,
            timed_out: result.timed_out,
            cancelled: false,
            duration_ms: elapsed_ms(started),
        };
        if output.timed_out {
            return OpResult {
                code: SysOpStatus::ShellExecutionError.code(),
                message: format!("execution timeout after {timeout} seconds"),
                data: Some(output),
            };
        }
        OpResult::success(output)
    }
}

#[async_trait]
impl<B: SandboxBackend> CodeOperation for RemoteCode<B> {
    async fn execute_code(&self, params: CodeExecParams) -> OpResult<CodeOutput> {
        let language = params.language;
        if params.code.trim().is_empty() {
            return OpResult {
                code: SysOpStatus::CodeExecutionError.code(),
                message: "code can not be empty".to_string(),
                data: Some(CodeOutput {
                    code_content: params.code,
                    language: language.as_str().to_string(),
                    exit_code: -1,
                    ..CodeOutput::default()
                }),
            };
        }
        let timeout = params.effective_timeout_secs(300);
        let mut environment: BTreeMap<String, Option<String>> = language
            .forced_env()
            .iter()
            .map(|(name, value)| ((*name).to_string(), Some((*value).to_string())))
            .collect();
        for (name, value) in params.environment {
            environment.insert(name, value);
        }
        let command = self
            .backend
            .code_command(&params.code, language, params.force_file);
        let request = ExecRequest {
            command,
            cwd: Some(self.backend.code_cwd(params.cwd.as_deref())),
            environment,
            stdin: None,
            timeout_secs: Some(timeout),
        };
        let started = Instant::now();
        let result = match self.backend.execute(&request).await {
            Ok(result) => result,
            Err(error) => return code_failure("execute_code", &error.message()),
        };
        let output = CodeOutput {
            code_content: params.code,
            language: language.as_str().to_string(),
            exit_code: result.exit_code,
            stdout: result.stdout,
            stderr: result.stderr,
            truncated: false,
            timed_out: result.timed_out,
            duration_ms: elapsed_ms(started),
        };
        if output.timed_out {
            return OpResult {
                code: SysOpStatus::CodeExecutionError.code(),
                message: format!("execution timeout after {timeout} seconds"),
                data: Some(output),
            };
        }
        OpResult::success(output)
    }

    fn execute_code_stream(self: Arc<Self>, params: CodeExecParams) -> OpStream<CodeChunk> {
        Box::pin(
            futures::stream::once(async move {
                match self.execute_code(params).await {
                    OpResult {
                        code: 0,
                        message: _,
                        data: Some(output),
                    } => code_chunks(&output),
                    failure => vec![OpResult {
                        code: failure.code,
                        message: failure.message,
                        data: Some(CodeChunk::terminal(0, -1)),
                    }],
                }
            })
            .flat_map(futures::stream::iter),
        )
    }
}

/// Splits read data into the chunks a streamed read emits.
fn read_chunks(data: ReadFileData, chunk_size: usize) -> Vec<OpResult<ReadChunkData>> {
    let path = data.path;
    let mode = data.mode;
    match data.content {
        FileContent::Text(text) => {
            let lines = text_lines(&text);
            if lines.is_empty() {
                return vec![OpResult::success(empty_read_chunk(path, mode))];
            }
            let last = lines.len() - 1;
            lines
                .into_iter()
                .enumerate()
                .map(|(index, line)| {
                    OpResult::success(ReadChunkData {
                        path: path.clone(),
                        chunk_size: len_u64(line.len()),
                        chunk: FileContent::Text(line),
                        mode,
                        chunk_index: len_u64(index),
                        is_last_chunk: index == last,
                    })
                })
                .collect()
        }
        FileContent::Bytes(bytes) => {
            let chunks = byte_chunks(&bytes, chunk_size);
            if chunks.is_empty() {
                return vec![OpResult::success(empty_read_chunk(path, mode))];
            }
            let last = chunks.len() - 1;
            chunks
                .into_iter()
                .enumerate()
                .map(|(index, chunk)| {
                    OpResult::success(ReadChunkData {
                        path: path.clone(),
                        chunk_size: len_u64(chunk.len()),
                        chunk: FileContent::Bytes(chunk),
                        mode,
                        chunk_index: len_u64(index),
                        is_last_chunk: index == last,
                    })
                })
                .collect()
        }
    }
}

/// The single terminal chunk an empty streamed read emits.
fn empty_read_chunk(path: String, mode: ReadMode) -> ReadChunkData {
    ReadChunkData {
        path,
        chunk: match mode {
            ReadMode::Text => FileContent::Text(String::new()),
            ReadMode::Bytes => FileContent::Bytes(Vec::new()),
        },
        mode,
        chunk_size: 0,
        chunk_index: 0,
        is_last_chunk: true,
    }
}

/// Splits a completed shell run into stdout lines, stderr lines, and the terminal chunk.
fn shell_chunks(output: &ShellOutput) -> Vec<OpResult<ShellChunk>> {
    let mut items = Vec::new();
    let mut index = 0_u64;
    for line in text_lines(&output.stdout) {
        items.push(OpResult::success(ShellChunk::new(
            line,
            StreamKind::Stdout,
            index,
        )));
        index += 1;
    }
    for line in text_lines(&output.stderr) {
        items.push(OpResult::success(ShellChunk::new(
            line,
            StreamKind::Stderr,
            index,
        )));
        index += 1;
    }
    items.push(OpResult::success(ShellChunk::terminal(
        index,
        output.exit_code,
    )));
    items
}

/// Splits a completed code run into stdout lines, stderr lines, and the terminal chunk.
fn code_chunks(output: &CodeOutput) -> Vec<OpResult<CodeChunk>> {
    let mut items = Vec::new();
    let mut index = 0_u64;
    for line in text_lines(&output.stdout) {
        items.push(OpResult::success(CodeChunk::new(
            line,
            StreamKind::Stdout,
            index,
        )));
        index += 1;
    }
    for line in text_lines(&output.stderr) {
        items.push(OpResult::success(CodeChunk::new(
            line,
            StreamKind::Stderr,
            index,
        )));
        index += 1;
    }
    items.push(OpResult::success(CodeChunk::terminal(
        index,
        output.exit_code,
    )));
    items
}

/// Maps a backend entry onto the kernel's listing item.
fn file_system_item(entry: RemoteEntry) -> FileSystemItem {
    FileSystemItem {
        name: entry.name,
        path: entry.path,
        size: entry.size,
        modified_time: entry.modified_time,
        is_directory: entry.is_directory,
        file_type: entry.file_type,
    }
}

/// The message Python's providers use for an unsupported conditional-write option.
fn unsupported_conditional(provider: &str, field: &str) -> String {
    format!(
        "{field} is not supported by the {provider} provider (remote sandbox transports have no compare-and-swap)"
    )
}

/// Normalizes the caller's working directory the way the Python providers do.
fn normalized_cwd(cwd: Option<&str>) -> Option<String> {
    cwd.filter(|value| !value.is_empty() && *value != ".")
        .map(str::to_string)
}

/// Converts an elapsed duration to whole milliseconds.
fn elapsed_ms(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
}

/// Converts a length or index to the `u64` the result types use.
fn len_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use tokio::fs;
    use tokio::sync::Mutex;

    /// An in-memory backend with scripted executions.
    #[derive(Default)]
    struct FakeBackend {
        files: Mutex<BTreeMap<String, Vec<u8>>>,
        entries: Mutex<Vec<RemoteEntry>>,
        search_entries: Mutex<Vec<RemoteEntry>>,
        executions: Mutex<Vec<ExecResult>>,
        requests: Mutex<Vec<ExecRequest>>,
    }

    impl FakeBackend {
        fn new() -> Self {
            Self::default()
        }

        async fn put(&self, path: &str, data: &[u8]) {
            self.files
                .lock()
                .await
                .insert(path.to_string(), data.to_vec());
        }

        async fn contents(&self, path: &str) -> Option<Vec<u8>> {
            self.files.lock().await.get(path).cloned()
        }

        async fn set_entries(&self, entries: Vec<RemoteEntry>) {
            *self.entries.lock().await = entries;
        }

        async fn set_search_entries(&self, entries: Vec<RemoteEntry>) {
            *self.search_entries.lock().await = entries;
        }

        async fn script(&self, results: Vec<ExecResult>) {
            *self.executions.lock().await = results;
        }

        async fn requests(&self) -> Vec<ExecRequest> {
            self.requests.lock().await.clone()
        }

        async fn last_request(&self) -> ExecRequest {
            self.requests.lock().await.last().cloned().unwrap()
        }
    }

    #[async_trait]
    impl SandboxBackend for FakeBackend {
        fn provider_name(&self) -> &'static str {
            "fake"
        }

        async fn read_bytes(&self, path: &str) -> std::result::Result<Vec<u8>, RemoteError> {
            self.files
                .lock()
                .await
                .get(path)
                .cloned()
                .ok_or_else(|| RemoteError::NotFound(format!("{path} not found")))
        }

        async fn write_bytes(
            &self,
            path: &str,
            data: &[u8],
            append: bool,
        ) -> std::result::Result<(), RemoteError> {
            let mut files = self.files.lock().await;
            if append {
                files
                    .entry(path.to_string())
                    .or_default()
                    .extend_from_slice(data);
            } else {
                files.insert(path.to_string(), data.to_vec());
            }
            Ok(())
        }

        async fn list(
            &self,
            request: &ListRequest,
        ) -> std::result::Result<Vec<RemoteEntry>, RemoteError> {
            Ok(self
                .entries
                .lock()
                .await
                .iter()
                .filter(|entry| {
                    if entry.is_directory {
                        request.include_dirs
                    } else {
                        request.include_files
                    }
                })
                .cloned()
                .collect())
        }

        async fn search(
            &self,
            _request: &SearchRequest,
        ) -> std::result::Result<Vec<RemoteEntry>, RemoteError> {
            Ok(self.search_entries.lock().await.clone())
        }

        async fn execute(
            &self,
            request: &ExecRequest,
        ) -> std::result::Result<ExecResult, RemoteError> {
            self.requests.lock().await.push(request.clone());
            let mut scripted = self.executions.lock().await;
            if scripted.is_empty() {
                Ok(ExecResult::default())
            } else {
                Ok(scripted.remove(0))
            }
        }
    }

    fn fs_backend(
        backend: &Arc<FakeBackend>,
        config: &SandboxGatewayConfig,
    ) -> RemoteFs<FakeBackend> {
        RemoteFs::new(Arc::clone(backend), config)
    }

    fn entry(name: &str, size: u64, modified_time: &str, file_type: Option<&str>) -> RemoteEntry {
        RemoteEntry {
            name: name.to_string(),
            path: format!("/work/{name}"),
            size,
            modified_time: modified_time.to_string(),
            is_directory: file_type.is_none(),
            file_type: file_type.map(str::to_string),
        }
    }

    fn chunk_texts(chunks: &[OpResult<ReadChunkData>]) -> Vec<String> {
        chunks
            .iter()
            .map(|chunk| match &chunk.data.as_ref().unwrap().chunk {
                FileContent::Text(text) => text.clone(),
                FileContent::Bytes(bytes) => String::from_utf8_lossy(bytes).into_owned(),
            })
            .collect()
    }

    async fn collect<T>(stream: OpStream<T>) -> Vec<OpResult<T>> {
        stream.collect().await
    }

    #[tokio::test]
    async fn text_read_applies_head_tail_and_line_range() {
        let backend = Arc::new(FakeBackend::new());
        backend.put("/work/a.txt", b"one\ntwo\nthree\nfour\n").await;
        let fs = fs_backend(&backend, &SandboxGatewayConfig::default());

        let head = fs
            .read_file(ReadFileParams::text("/work/a.txt").with_head(2))
            .await;
        assert_eq!(head.code, 0);
        assert_eq!(
            head.data.unwrap().content,
            FileContent::Text("one\ntwo\n".to_string())
        );

        let tail = fs
            .read_file(ReadFileParams::text("/work/a.txt").with_tail(1))
            .await;
        assert_eq!(
            tail.data.unwrap().content,
            FileContent::Text("four\n".to_string())
        );

        let range = fs
            .read_file(ReadFileParams::text("/work/a.txt").with_line_range(2, 3))
            .await;
        let data = range.data.unwrap();
        assert_eq!(data.content, FileContent::Text("two\nthree\n".to_string()));
        assert_eq!(data.total_bytes, 19);
        assert_eq!(data.encoding, None);
    }

    #[tokio::test]
    async fn text_read_reports_a_non_default_encoding() {
        let backend = Arc::new(FakeBackend::new());
        backend.put("/work/a.txt", b"hi").await;
        let fs = fs_backend(&backend, &SandboxGatewayConfig::default());
        let result = fs
            .read_file(ReadFileParams::text("/work/a.txt").with_encoding("gbk"))
            .await;
        assert_eq!(result.data.unwrap().encoding, Some("gbk".to_string()));
    }

    #[tokio::test]
    async fn read_validation_errors_use_the_fs_status() {
        let backend = Arc::new(FakeBackend::new());
        let fs = fs_backend(&backend, &SandboxGatewayConfig::default());

        let bytes_head = fs
            .read_file(ReadFileParams::bytes("/work/a.bin").with_head(1))
            .await;
        assert_eq!(bytes_head.code, 199_003);
        assert!(bytes_head.message.contains(
            "Parameters 'head', 'tail', and 'line_range' are only supported in text mode"
        ));

        let both = fs
            .read_file(
                ReadFileParams::text("/work/a.txt")
                    .with_head(1)
                    .with_tail(1),
            )
            .await;
        assert_eq!(both.code, 199_003);
        assert!(both
            .message
            .contains("head and tail cannot be specified simultaneously"));
    }

    #[tokio::test]
    async fn a_missing_file_is_a_gateway_failure() {
        let backend = Arc::new(FakeBackend::new());
        let fs = fs_backend(&backend, &SandboxGatewayConfig::default());
        let result = fs
            .read_file(ReadFileParams::text("/work/missing.txt"))
            .await;
        assert_eq!(result.code, 199_007);
        assert_eq!(
            result.message,
            "sandbox gateway error, operation: read_file, reason: /work/missing.txt not found"
        );
    }

    #[tokio::test]
    async fn bytes_read_returns_the_raw_payload() {
        let backend = Arc::new(FakeBackend::new());
        backend.put("/work/a.bin", &[0_u8, 1, 2]).await;
        let fs = fs_backend(&backend, &SandboxGatewayConfig::default());
        let result = fs.read_file(ReadFileParams::bytes("/work/a.bin")).await;
        let data = result.data.unwrap();
        assert_eq!(data.mode, ReadMode::Bytes);
        assert_eq!(data.content, FileContent::Bytes(vec![0, 1, 2]));
        assert_eq!(data.total_bytes, 3);
        assert!(!data.truncated);
    }

    #[tokio::test]
    async fn read_stream_emits_one_chunk_per_selected_line() {
        let backend = Arc::new(FakeBackend::new());
        backend.put("/work/a.txt", b"one\ntwo\nthree\n").await;
        let fs = Arc::new(fs_backend(&backend, &SandboxGatewayConfig::default()));
        let chunks =
            collect(fs.read_file_stream(ReadFileParams::text("/work/a.txt").with_head(2))).await;

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunk_texts(&chunks), vec!["one\n", "two\n"]);
        let first = chunks[0].data.as_ref().unwrap();
        assert_eq!(first.chunk_index, 0);
        assert_eq!(first.chunk_size, 4);
        assert!(!first.is_last_chunk);
        let last = chunks[1].data.as_ref().unwrap();
        assert_eq!(last.chunk_index, 1);
        assert!(last.is_last_chunk);
    }

    #[tokio::test]
    async fn read_stream_splits_bytes_by_chunk_size() {
        let backend = Arc::new(FakeBackend::new());
        backend.put("/work/a.bin", b"abcde").await;
        let fs = Arc::new(fs_backend(&backend, &SandboxGatewayConfig::default()));
        let mut params = ReadFileParams::bytes("/work/a.bin");
        params.chunk_size = 2;
        let chunks = collect(fs.read_file_stream(params)).await;

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunk_texts(&chunks), vec!["ab", "cd", "e"]);
        let sizes: Vec<u64> = chunks
            .iter()
            .map(|chunk| chunk.data.as_ref().unwrap().chunk_size)
            .collect();
        assert_eq!(sizes, vec![2, 2, 1]);
        assert!(chunks[2].data.as_ref().unwrap().is_last_chunk);
        assert_eq!(chunks[0].data.as_ref().unwrap().mode, ReadMode::Bytes);
    }

    #[tokio::test]
    async fn an_empty_stream_always_yields_one_terminal_chunk() {
        let backend = Arc::new(FakeBackend::new());
        backend.put("/work/empty.txt", b"").await;
        backend.put("/work/empty.bin", b"").await;
        let fs = Arc::new(fs_backend(&backend, &SandboxGatewayConfig::default()));

        let text = collect(
            fs.clone()
                .read_file_stream(ReadFileParams::text("/work/empty.txt")),
        )
        .await;
        assert_eq!(text.len(), 1);
        let data = text[0].data.as_ref().unwrap();
        assert!(data.is_last_chunk);
        assert_eq!(data.chunk, FileContent::Text(String::new()));

        let bytes = collect(fs.read_file_stream(ReadFileParams::bytes("/work/empty.bin"))).await;
        assert_eq!(bytes.len(), 1);
        let data = bytes[0].data.as_ref().unwrap();
        assert!(data.is_last_chunk);
        assert_eq!(data.chunk, FileContent::Bytes(Vec::new()));
    }

    #[tokio::test]
    async fn a_failed_stream_yields_exactly_one_terminal_chunk() {
        let backend = Arc::new(FakeBackend::new());
        let fs = Arc::new(fs_backend(&backend, &SandboxGatewayConfig::default()));
        let chunks = collect(fs.read_file_stream(ReadFileParams::text("/work/missing.txt"))).await;

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].code, 199_007);
        let data = chunks[0].data.as_ref().unwrap();
        assert_eq!(data.path, "/work/missing.txt");
        assert!(data.is_last_chunk);
        assert_eq!(data.mode, ReadMode::Text);
    }

    #[tokio::test]
    async fn write_file_applies_python_newline_defaults_and_append() {
        let backend = Arc::new(FakeBackend::new());
        let fs = fs_backend(&backend, &SandboxGatewayConfig::default());

        let written = fs
            .write_file(WriteFileParams::new("/work/a.txt", "hello"))
            .await;
        let data = written.data.unwrap();
        assert_eq!(data.size, 6);
        assert_eq!(data.mode, ReadMode::Text);
        assert_eq!(backend.contents("/work/a.txt").await.unwrap(), b"\nhello");

        let appended = fs
            .write_file(WriteFileParams::raw("/work/a.txt", "more").with_append())
            .await;
        assert_eq!(appended.code, 0);
        assert_eq!(
            backend.contents("/work/a.txt").await.unwrap(),
            b"\nhellomore"
        );

        let trailing = fs
            .write_file({
                let mut params = WriteFileParams::raw("/work/b.txt", "x");
                params.append_newline = true;
                params
            })
            .await;
        assert_eq!(trailing.data.unwrap().size, 2);
        assert_eq!(backend.contents("/work/b.txt").await.unwrap(), b"x\n");
    }

    #[tokio::test]
    async fn write_file_enforces_create_if_not_exist() {
        let backend = Arc::new(FakeBackend::new());
        let fs = fs_backend(&backend, &SandboxGatewayConfig::default());
        let params = {
            let mut params = WriteFileParams::raw("/work/new.txt", "x");
            params.create_if_not_exist = false;
            params
        };
        let result = fs.write_file(params).await;
        assert_eq!(result.code, 199_003);
        assert!(result
            .message
            .contains("File does not exist: /work/new.txt"));
        assert!(backend.contents("/work/new.txt").await.is_none());
    }

    #[tokio::test]
    async fn conditional_writes_are_rejected_with_the_provider_name() {
        let backend = Arc::new(FakeBackend::new());
        let fs = fs_backend(&backend, &SandboxGatewayConfig::default());

        let sha = fs
            .write_file(WriteFileParams::raw("/work/a.txt", "x").with_expected_sha256("deadbeef"))
            .await;
        assert_eq!(sha.code, 199_003);
        assert_eq!(
            sha.message,
            "file system operation execution error, execution: write_file, reason: \
             expected_content_sha256 is not supported by the fake provider (remote sandbox \
             transports have no compare-and-swap)"
        );

        let absent = fs
            .write_file({
                let mut params = WriteFileParams::raw("/work/a.txt", "x");
                params.create_if_absent = true;
                params
            })
            .await;
        assert_eq!(absent.code, 199_003);
        assert!(absent
            .message
            .contains("create_if_absent is not supported by the fake provider"));
    }

    #[tokio::test]
    async fn upload_and_download_round_trip_through_the_host() {
        let dir = tempfile::tempdir().unwrap();
        let local = dir.path().join("payload.bin");
        fs::write(&local, b"payload").await.unwrap();
        let backend = Arc::new(FakeBackend::new());
        let fs_operation = Arc::new(fs_backend(&backend, &SandboxGatewayConfig::default()));

        let upload = fs_operation
            .upload_file(UploadFileParams {
                local_path: local.to_string_lossy().into_owned(),
                target_path: "/work/payload.bin".to_string(),
                ..UploadFileParams::default()
            })
            .await;
        assert_eq!(upload.code, 0);
        assert_eq!(upload.data.as_ref().unwrap().size, 7);
        assert_eq!(
            backend.contents("/work/payload.bin").await.unwrap(),
            b"payload"
        );

        let target = dir.path().join("out.bin");
        let download = fs_operation
            .download_file(DownloadFileParams {
                source_path: "/work/payload.bin".to_string(),
                local_path: target.to_string_lossy().into_owned(),
                ..DownloadFileParams::default()
            })
            .await;
        assert_eq!(download.code, 0);
        assert_eq!(fs::read(&target).await.unwrap(), b"payload");

        let chunk = collect(
            fs_operation.download_file_stream(DownloadFileParams {
                source_path: "/work/payload.bin".to_string(),
                local_path: dir
                    .path()
                    .join("nested/stream.bin")
                    .to_string_lossy()
                    .into_owned(),
                create_parent_dirs: true,
                overwrite: true,
                ..DownloadFileParams::default()
            }),
        )
        .await;
        assert_eq!(chunk.len(), 1);
        let data = chunk[0].data.as_ref().unwrap();
        assert_eq!(data.chunk_size, 7);
        assert!(data.is_last_chunk);
        assert_eq!(
            fs::read(dir.path().join("nested/stream.bin"))
                .await
                .unwrap(),
            b"payload"
        );
    }

    #[tokio::test]
    async fn transfers_refuse_existing_destinations() {
        let dir = tempfile::tempdir().unwrap();
        let local = dir.path().join("payload.bin");
        fs::write(&local, b"payload").await.unwrap();
        let backend = Arc::new(FakeBackend::new());
        backend.put("/work/payload.bin", b"remote").await;
        let fs_operation = fs_backend(&backend, &SandboxGatewayConfig::default());

        let upload = fs_operation
            .upload_file(UploadFileParams {
                local_path: local.to_string_lossy().into_owned(),
                target_path: "/work/payload.bin".to_string(),
                ..UploadFileParams::default()
            })
            .await;
        assert_eq!(upload.code, 199_003);
        assert!(upload
            .message
            .contains("File already exists: /work/payload.bin"));

        let existing = dir.path().join("existing.bin");
        fs::write(&existing, b"old").await.unwrap();
        let download = fs_operation
            .download_file(DownloadFileParams {
                source_path: "/work/payload.bin".to_string(),
                local_path: existing.to_string_lossy().into_owned(),
                ..DownloadFileParams::default()
            })
            .await;
        assert_eq!(download.code, 199_003);
        assert!(download.message.contains("File already exists"));
        assert_eq!(fs::read(&existing).await.unwrap(), b"old");
    }

    #[tokio::test]
    async fn upload_reports_an_unreadable_local_file() {
        let backend = Arc::new(FakeBackend::new());
        let fs_operation = fs_backend(&backend, &SandboxGatewayConfig::default());
        let result = fs_operation
            .upload_file(UploadFileParams {
                local_path: "/definitely/missing/payload.bin".to_string(),
                target_path: "/work/payload.bin".to_string(),
                ..UploadFileParams::default()
            })
            .await;
        assert_eq!(result.code, 199_003);
        assert!(result.message.contains(
            "failed to read local file '/definitely/missing/payload.bin': \
             failed to read host file '/definitely/missing/payload.bin'"
        ));
    }

    #[tokio::test]
    async fn list_files_sorts_and_filters() {
        let backend = Arc::new(FakeBackend::new());
        backend
            .set_entries(vec![
                entry("b.rs", 30, "2024-01-02 00:00:00", Some(".rs")),
                entry("a.txt", 10, "2024-01-03 00:00:00", Some(".txt")),
                entry("sub", 512, "2024-01-01 00:00:00", None),
            ])
            .await;
        let fs_operation = fs_backend(&backend, &SandboxGatewayConfig::default());

        let mut params = ListFilesParams::new("/work");
        params.sort_by = SortBy::Size;
        params.sort_descending = true;
        let listed = fs_operation.list_files(params).await;
        let data = listed.data.unwrap();
        assert_eq!(data.total_count, 2, "directories are excluded");
        assert_eq!(
            data.list_items
                .iter()
                .map(|item| item.name.as_str())
                .collect::<Vec<_>>(),
            vec!["b.rs", "a.txt"]
        );

        let filtered = fs_operation
            .list_files({
                let mut params = ListFilesParams::new("/work");
                params.file_types = Some(vec![".txt".to_string()]);
                params
            })
            .await;
        let data = filtered.data.unwrap();
        assert_eq!(data.total_count, 1);
        assert_eq!(data.list_items[0].path, "/work/a.txt");
        assert_eq!(data.root_path, "/work");
    }

    #[tokio::test]
    async fn list_directories_returns_only_directories() {
        let backend = Arc::new(FakeBackend::new());
        backend
            .set_entries(vec![
                entry("a.txt", 10, "2024-01-03 00:00:00", Some(".txt")),
                entry("sub", 512, "2024-01-01 00:00:00", None),
            ])
            .await;
        let fs_operation = fs_backend(&backend, &SandboxGatewayConfig::default());
        let listed = fs_operation
            .list_directories(ListDirsParams::new("/work"))
            .await;
        let data = listed.data.unwrap();
        assert_eq!(data.total_count, 1);
        assert!(data.list_items[0].is_directory);
        assert_eq!(data.list_items[0].size, 512);
        assert_eq!(data.list_items[0].file_type, None);
    }

    #[tokio::test]
    async fn search_files_maps_entries_and_sorts_by_name() {
        let backend = Arc::new(FakeBackend::new());
        backend
            .set_search_entries(vec![
                entry("b.txt", 2, "2024-01-01 00:00:00", Some(".txt")),
                entry("a.txt", 1, "2024-01-01 00:00:00", Some(".txt")),
            ])
            .await;
        let fs_operation = fs_backend(&backend, &SandboxGatewayConfig::default());
        let result = fs_operation
            .search_files(SearchFilesParams::new("/work", "*.txt").with_excludes(["*.rs"]))
            .await;
        let data = result.data.unwrap();
        assert_eq!(data.total_matches, 2);
        assert_eq!(
            data.matching_files
                .iter()
                .map(|item| item.name.as_str())
                .collect::<Vec<_>>(),
            vec!["a.txt", "b.txt"]
        );
        assert_eq!(data.search_path, "/work");
        assert_eq!(data.search_pattern, "*.txt");
        assert_eq!(data.exclude_patterns, Some(vec!["*.rs".to_string()]));
    }

    #[tokio::test]
    async fn execute_cmd_reports_success_and_non_zero_exits_as_values() {
        let backend = Arc::new(FakeBackend::new());
        backend
            .script(vec![ExecResult {
                exit_code: 7,
                stdout: "out".to_string(),
                stderr: "err".to_string(),
                timed_out: false,
            }])
            .await;
        let shell = RemoteShell::new(Arc::clone(&backend));
        let params = ShellExecParams::new("false")
            .with_cwd(".")
            .with_timeout_secs(12);
        let result = shell.execute_cmd(params).await;

        assert_eq!(result.code, 0);
        let output = result.data.unwrap();
        assert_eq!(output.exit_code, 7);
        assert_eq!(output.stdout, "out");
        assert_eq!(output.stderr, "err");
        assert_eq!(output.cwd, ".");
        assert!(!output.timed_out);
        assert!(!output.truncated);
        assert!(!output.cancelled);

        let request = backend.last_request().await;
        assert_eq!(request.command, ExecCommand::Shell("false".to_string()));
        assert_eq!(request.cwd, None, "'.' is normalized away");
        assert_eq!(request.timeout_secs, Some(12));
    }

    #[tokio::test]
    async fn execute_cmd_rejects_an_empty_command() {
        let backend = Arc::new(FakeBackend::new());
        let shell = RemoteShell::new(Arc::clone(&backend));
        let result = shell.execute_cmd(ShellExecParams::new("   ")).await;
        assert_eq!(result.code, 199_004);
        assert!(result.message.contains("command can not be empty"));
        assert!(result.data.is_none());
        assert!(backend.requests().await.is_empty());
    }

    #[tokio::test]
    async fn a_timeout_is_a_shell_failure_with_data() {
        let backend = Arc::new(FakeBackend::new());
        backend
            .script(vec![ExecResult {
                exit_code: 124,
                stdout: "partial".to_string(),
                stderr: String::new(),
                timed_out: true,
            }])
            .await;
        let shell = RemoteShell::new(Arc::clone(&backend));
        let result = shell
            .execute_cmd(ShellExecParams::new("sleep 5").with_timeout_secs(5))
            .await;

        assert_eq!(result.code, 199_004);
        assert_eq!(result.message, "execution timeout after 5 seconds");
        let output = result.data.unwrap();
        assert!(output.timed_out);
        assert_eq!(output.exit_code, 124);
        assert_eq!(output.stdout, "partial");
    }

    #[tokio::test]
    async fn execute_argv_quotes_the_command_line() {
        let backend = Arc::new(FakeBackend::new());
        let shell = RemoteShell::new(Arc::clone(&backend));
        let result = shell
            .execute_argv(ShellArgvParams::new(["echo", "hello world"]))
            .await;

        assert_eq!(result.code, 0);
        assert_eq!(result.data.unwrap().command, "echo 'hello world'");
        let request = backend.last_request().await;
        assert_eq!(
            request.command,
            ExecCommand::Argv(vec!["echo".to_string(), "hello world".to_string()])
        );

        let empty = shell.execute_argv(ShellArgvParams::default()).await;
        assert_eq!(empty.code, 199_004);
        assert!(empty.message.contains("argv can not be empty"));
    }

    #[tokio::test]
    async fn execute_cmd_background_parses_the_pid() {
        let backend = Arc::new(FakeBackend::new());
        backend
            .script(vec![ExecResult {
                exit_code: 0,
                stdout: "4321\n".to_string(),
                stderr: String::new(),
                timed_out: false,
            }])
            .await;
        let shell = RemoteShell::new(Arc::clone(&backend));
        let result = shell
            .execute_cmd_background(ShellExecParams::new("sleep 30"))
            .await;

        assert_eq!(result.code, 0);
        let output = result.data.unwrap();
        assert_eq!(output.pid, Some(4321));
        assert_eq!(output.command, "sleep 30");
        assert_eq!(output.cwd, ".");
        assert_eq!(output.error, None);

        let request = backend.last_request().await;
        assert_eq!(request.timeout_secs, Some(10));
        let ExecCommand::Shell(command) = request.command else {
            panic!("expected a shell command");
        };
        assert!(command.contains("echo $!"));
        assert!(command.contains("sleep 30"));
    }

    #[tokio::test]
    async fn execute_cmd_background_surfaces_failures() {
        let backend = Arc::new(FakeBackend::new());
        backend
            .script(vec![ExecResult {
                exit_code: 3,
                stdout: String::new(),
                stderr: "boom".to_string(),
                timed_out: false,
            }])
            .await;
        let shell = RemoteShell::new(Arc::clone(&backend));
        let failed = shell
            .execute_cmd_background(ShellExecParams::new("bad"))
            .await;
        assert_eq!(failed.code, 199_004);
        assert!(failed.message.contains("exit code 3"));
        assert!(failed.message.contains("boom"));

        let backend = Arc::new(FakeBackend::new());
        backend
            .script(vec![ExecResult {
                exit_code: 0,
                stdout: "not-a-pid".to_string(),
                stderr: String::new(),
                timed_out: false,
            }])
            .await;
        let shell = RemoteShell::new(Arc::clone(&backend));
        let unparsable = shell
            .execute_cmd_background(ShellExecParams::new("weird"))
            .await;
        assert_eq!(unparsable.code, 199_004);
        assert!(unparsable.message.contains("background pid"));

        let empty = shell
            .execute_cmd_background(ShellExecParams::new(" "))
            .await;
        assert_eq!(empty.code, 199_004);
        assert!(empty.message.contains("command can not be empty"));
    }

    #[tokio::test]
    async fn shell_stream_orders_stdout_stderr_and_the_terminal_chunk() {
        let backend = Arc::new(FakeBackend::new());
        backend
            .script(vec![ExecResult {
                exit_code: 2,
                stdout: "a\nb\n".to_string(),
                stderr: "e\n".to_string(),
                timed_out: false,
            }])
            .await;
        let shell = Arc::new(RemoteShell::new(Arc::clone(&backend)));
        let chunks = collect(shell.execute_cmd_stream(ShellExecParams::new("cmd"))).await;

        assert_eq!(chunks.len(), 4);
        let data: Vec<_> = chunks
            .iter()
            .map(|chunk| chunk.data.as_ref().unwrap())
            .collect();
        assert_eq!(data[0].text, "a\n");
        assert_eq!(data[0].stream, Some(StreamKind::Stdout));
        assert_eq!(data[1].text, "b\n");
        assert_eq!(data[1].stream, Some(StreamKind::Stdout));
        assert_eq!(data[2].text, "e\n");
        assert_eq!(data[2].stream, Some(StreamKind::Stderr));
        assert_eq!(data[3].chunk_index, 3);
        assert_eq!(data[3].exit_code, Some(2));
        assert_eq!(
            data.iter()
                .map(|chunk| chunk.chunk_index)
                .collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );
    }

    #[tokio::test]
    async fn an_empty_shell_stream_still_terminates() {
        let backend = Arc::new(FakeBackend::new());
        let shell = Arc::new(RemoteShell::new(Arc::clone(&backend)));
        let chunks = collect(shell.execute_cmd_stream(ShellExecParams::new("true"))).await;

        assert_eq!(chunks.len(), 1);
        let data = chunks[0].data.as_ref().unwrap();
        assert_eq!(data.chunk_index, 0);
        assert_eq!(data.exit_code, Some(0));
    }

    #[tokio::test]
    async fn a_failed_shell_stream_yields_one_terminal_chunk() {
        let backend = Arc::new(FakeBackend::new());
        let shell = Arc::new(RemoteShell::new(Arc::clone(&backend)));
        let chunks = collect(shell.execute_cmd_stream(ShellExecParams::new(""))).await;

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].code, 199_004);
        assert!(chunks[0].message.contains("command can not be empty"));
        let data = chunks[0].data.as_ref().unwrap();
        assert_eq!(data.exit_code, Some(-1));
        assert_eq!(data.chunk_index, 0);
    }

    #[tokio::test]
    async fn execute_code_uses_inline_argv_and_forced_environment_defaults() {
        let backend = Arc::new(FakeBackend::new());
        let code = RemoteCode::new(Arc::clone(&backend));
        let params = CodeExecParams::new("print(1)").with_cwd("/custom");
        let mut params = params;
        params
            .environment
            .insert("PYTHONUTF8".to_string(), Some("0".to_string()));
        params
            .environment
            .insert("EXTRA".to_string(), Some("1".to_string()));
        let result = code.execute_code(params).await;

        assert_eq!(result.code, 0);
        assert_eq!(result.data.unwrap().language, "python");

        let request = backend.last_request().await;
        assert_eq!(
            request.command,
            ExecCommand::Argv(vec![
                "python3".to_string(),
                "-u".to_string(),
                "-c".to_string(),
                "print(1)".to_string()
            ])
        );
        assert_eq!(request.cwd, Some("/tmp".to_string()));
        assert_eq!(request.timeout_secs, Some(300));
        assert_eq!(
            request.environment.get("PYTHONIOENCODING"),
            Some(&Some("utf-8".to_string()))
        );
        assert_eq!(
            request.environment.get("PYTHONUTF8"),
            Some(&Some("0".to_string())),
            "the caller wins over the forced default"
        );
        assert_eq!(
            request.environment.get("EXTRA"),
            Some(&Some("1".to_string()))
        );
    }

    #[tokio::test]
    async fn execute_code_rejects_an_empty_snippet() {
        let backend = Arc::new(FakeBackend::new());
        let code = RemoteCode::new(Arc::clone(&backend));
        let result = code.execute_code(CodeExecParams::new(" \n")).await;

        assert_eq!(result.code, 199_005);
        assert_eq!(result.message, "code can not be empty");
        let output = result.data.unwrap();
        assert_eq!(output.exit_code, -1);
        assert_eq!(output.language, "python");
        assert!(backend.requests().await.is_empty());
    }

    #[tokio::test]
    async fn force_file_writes_a_temporary_script() {
        let backend = Arc::new(FakeBackend::new());
        let code = RemoteCode::new(Arc::clone(&backend));
        let params = CodeExecParams::javascript("console.log(1)").with_force_file(true);
        code.execute_code(params).await;

        let request = backend.last_request().await;
        let ExecCommand::Shell(command) = request.command else {
            panic!("force_file must use a shell command");
        };
        assert!(command.contains("/tmp/ojw_code_XXXXXX.js"), "{command}");
        assert!(command.contains("base64 -d"), "{command}");
        assert!(command.contains("node \"$tmp\""), "{command}");
        let encoded = command
            .split("printf %s ")
            .nth(1)
            .unwrap()
            .split(" | base64 -d")
            .next()
            .unwrap();
        assert_eq!(base64_decode_for_test(encoded), "console.log(1)");
    }

    fn base64_decode_for_test(value: &str) -> String {
        use base64::Engine as _;
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(value.trim_matches('\''))
            .unwrap();
        String::from_utf8(decoded).unwrap()
    }

    #[tokio::test]
    async fn a_code_timeout_is_a_code_failure_with_data() {
        let backend = Arc::new(FakeBackend::new());
        backend
            .script(vec![ExecResult {
                exit_code: 124,
                stdout: "partial".to_string(),
                stderr: String::new(),
                timed_out: true,
            }])
            .await;
        let code = RemoteCode::new(Arc::clone(&backend));
        let result = code
            .execute_code(CodeExecParams::new("while True: pass").with_timeout_secs(4))
            .await;

        assert_eq!(result.code, 199_005);
        assert_eq!(result.message, "execution timeout after 4 seconds");
        assert!(result.data.unwrap().timed_out);
    }

    #[tokio::test]
    async fn code_stream_orders_chunks_and_terminates() {
        let backend = Arc::new(FakeBackend::new());
        backend
            .script(vec![ExecResult {
                exit_code: 0,
                stdout: "x\ny\n".to_string(),
                stderr: "z\n".to_string(),
                timed_out: false,
            }])
            .await;
        let code = Arc::new(RemoteCode::new(Arc::clone(&backend)));
        let chunks = collect(code.execute_code_stream(CodeExecParams::new("print(1)"))).await;

        assert_eq!(chunks.len(), 4);
        let data: Vec<_> = chunks
            .iter()
            .map(|chunk| chunk.data.as_ref().unwrap())
            .collect();
        assert_eq!(
            data.iter()
                .map(|chunk| chunk.chunk_index)
                .collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );
        assert_eq!(data[0].stream, Some(StreamKind::Stdout));
        assert_eq!(data[2].stream, Some(StreamKind::Stderr));
        assert_eq!(data[3].exit_code, Some(0));
    }

    #[tokio::test]
    async fn a_failed_code_stream_yields_one_terminal_chunk() {
        let backend = Arc::new(FakeBackend::new());
        let code = Arc::new(RemoteCode::new(Arc::clone(&backend)));
        let chunks = collect(code.execute_code_stream(CodeExecParams::new(""))).await;

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].code, 199_005);
        assert_eq!(chunks[0].message, "code can not be empty");
        assert_eq!(chunks[0].data.as_ref().unwrap().exit_code, Some(-1));
    }

    #[tokio::test]
    async fn provider_set_from_registers_three_factories_and_honours_the_config() {
        let set = provider_set_from(
            |_endpoint: &SandboxEndpoint, _config: &SandboxGatewayConfig| Ok(FakeBackend::new()),
        );
        assert!(set.fs.is_some());
        assert!(set.shell.is_some());
        assert!(set.code.is_some());

        let conditional = SandboxGatewayConfig {
            supports_conditional_write: true,
            ..SandboxGatewayConfig::default()
        };
        let fs = set.fs.as_ref().unwrap();
        let fs = fs(&SandboxEndpoint::default(), &conditional).unwrap();
        assert!(fs.supports_conditional_write());

        let plain = set.fs.as_ref().unwrap();
        let plain = plain(
            &SandboxEndpoint::default(),
            &SandboxGatewayConfig::default(),
        )
        .unwrap();
        assert!(!plain.supports_conditional_write());
    }
}
