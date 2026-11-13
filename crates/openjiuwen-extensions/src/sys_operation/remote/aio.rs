//! The `aio` remote sandbox backend.
//!
//! Python parity: `openjiuwen/extensions/sys_operation/sandbox/providers/aio.py`
//! (v0.1.20), which registers one provider per operation type
//! (`@SandboxRegistry.provider("aio", "fs" | "shell" | "code")`). This module
//! builds the same three factories.
//!
//! # The transport decision
//!
//! Python's provider wraps the third-party `agent_sandbox` SDK, which is not
//! vendored and whose private REST contract is not part of the reference
//! repository. The only wire contract defined there is the kernel gateway
//! envelope: a [`GatewayInvokeRequest`] (`op_type` `"fs"` / `"shell"` /
//! `"code"`, `method`, `params`, `isolation_key`) answered by a
//! [`GatewayResponse`] (`code`, `message`, `data`) where `code == 0` means
//! success (`core/sys_operation/sandbox/gateway/gateway.py` and
//! `gateway_client.py`, which dispatch `params` as the provider method's
//! keyword arguments). This provider is therefore a real HTTP client for that
//! envelope:
//!
//! 1. it POSTs the envelope to `{base_url}{invoke_path}`, where `invoke_path`
//!    defaults to [`DEFAULT_INVOKE_PATH`] and can be overridden through
//!    `launcher_config.extra_params["invoke_path"]` (a leading `/` is added
//!    when missing);
//! 2. it unwraps the [`GatewayResponse`] tolerantly (`code` defaults to `0`,
//!    `message` to `""`, `data` is optional) and returns `data` (`null` when
//!    absent) on success, or a [`RemoteError::Status`] carrying
//!    `aio gateway code <code>: <message>` otherwise — which the wrappers
//!    render as code `199007` through
//!    [`sandbox_gateway_message`](openjiuwen_core::sys_operation::status::sandbox_gateway_message);
//!    because no HTTP status is involved in an envelope rejection, the reason
//!    is rendered verbatim (`reason(error)`) instead of the transport's
//!    `HTTP 0: …` form, while genuine transport failures keep the transport
//!    rendering (including `HTTP 404`);
//! 3. it maps `data` into the kernel result structs.
//!
//! The `params` field names mirror the Python provider signatures exactly
//! (`timeout`, not `timeout_secs`; `line_range` as a `[start, end]` array;
//! bytes content as a padded base64 string with `encoding: "base64"`), because
//! the gateway dispatches them as keyword arguments.
//!
//! # What stays host-side
//!
//! The envelope has **no streaming variant**, so the three stream methods
//! perform their buffered call and chunk the result on the host: text reads by
//! line, byte reads by `chunk_size` (or [`DEFAULT_STREAM_CHUNK_SIZE`]) and
//! shell/code output by line, always ending in a terminal frame. Host-side
//! work is otherwise limited to local file transfer (upload/download) and path
//! handling; the *remote* provider performs text selection
//! (`head`/`tail`/`line_range`) and newline transforms
//! (`prepend_newline`/`append_newline`), which is why those arrive as
//! parameters instead of being applied here.
//!
//! Two further consequences are deliberate:
//!
//! - `upload_file` cannot hand the container a `local_path` (the envelope sees
//!   the provider's filesystem, not the caller's), so the host file is read and
//!   inlined as base64 bytes in a `write_file` call;
//! - `execute_argv` has no Python counterpart — it is sent as `execute_cmd`
//!   with the POSIX-quoted argv joined by single spaces, the same shape
//!   [`posix_quote`] gives the generic remote backend.
//!
//! Container lifecycle (`launcher_config.on_stop`, `idle_ttl_seconds`) is not
//! applied: the kernel has no launcher or lifecycle API, so a provider only
//! talks to the already resolved [`SandboxEndpoint`].
//!
//! # Notes on the mapping
//!
//! - Shell completion is success even for a non-zero exit code (Python
//!   parity); only a timeout becomes a failure (`199004`), and a `124` exit
//!   code — `timeout(1)`'s status — is the timeout signal.
//! - Code completion follows the same rule (`199005`), detecting a timeout
//!   from exit code `124` or a `stderr` containing "timed out".
//! - `write_file` rejects `expected_content_sha256` and `create_if_absent`
//!   because a remote transport has no compare-and-swap.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{json, Value};

use openjiuwen_common::error::Result;
use openjiuwen_core::sys_operation::code::types::{
    CodeChunk, CodeExecParams, CodeOutput, Language,
};
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
use openjiuwen_core::sys_operation::sandbox::config::{
    GatewayInvokeRequest, GatewayResponse, SandboxEndpoint, SandboxGatewayConfig,
};
use openjiuwen_core::sys_operation::shell::types::{
    ShellArgvParams, ShellBackgroundOutput, ShellChunk, ShellExecParams, ShellOutput, ShellType,
    StreamKind,
};
use openjiuwen_core::sys_operation::status::{sandbox_gateway_message, SysOpStatus};

use super::local_io::{read_host_file, write_host_file, HostIoError};
use super::semantics::{
    byte_chunks, file_extension, filter_file_types, normalize_line_selection, posix_quote,
    sort_items, text_lines,
};
use super::transport::{base64_decode, base64_encode, hex_decode, RemoteError, SandboxHttp};
use crate::sys_operation::SandboxProviderSet;

#[cfg(test)]
mod tests;

/// The sandbox type this module implements.
const PROVIDER: &str = "aio";

/// The envelope path a stock AIO gateway listens on.
const DEFAULT_INVOKE_PATH: &str = "/api/v1/invoke";

/// The `launcher_config.extra_params` key overriding [`DEFAULT_INVOKE_PATH`].
const INVOKE_PATH_PARAM: &str = "invoke_path";

/// The timeout applied when the caller omits one (Python `timeout or 300`).
const DEFAULT_TIMEOUT_SECS: u64 = 300;

/// `timeout(1)`'s exit status, which both providers treat as a timeout.
const TIMEOUT_EXIT_CODE: i32 = 124;

/// The encoding name that marks bytes payloads in the envelope.
const BASE64_ENCODING: &str = "base64";

/// The filesystem, shell, and code factories of the `aio` sandbox type.
///
/// Every factory builds its own [`AioClient`] from the endpoint and
/// configuration the registry hands it: the registry calls each factory
/// separately, so the three operation types of one sandbox may legitimately be
/// resolved against different endpoints.
#[must_use]
pub(crate) fn provider_set() -> SandboxProviderSet {
    SandboxProviderSet::new()
        .with_fs(Arc::new(
            |endpoint: &SandboxEndpoint, config: &SandboxGatewayConfig| {
                let client = Arc::new(AioClient::new(endpoint, config)?);
                let operation: Arc<dyn FsOperation> = Arc::new(AioFs { client });
                Ok(operation)
            },
        ))
        .with_shell(Arc::new(
            |endpoint: &SandboxEndpoint, config: &SandboxGatewayConfig| {
                let client = Arc::new(AioClient::new(endpoint, config)?);
                let operation: Arc<dyn ShellOperation> = Arc::new(AioShell { client });
                Ok(operation)
            },
        ))
        .with_code(Arc::new(
            |endpoint: &SandboxEndpoint, config: &SandboxGatewayConfig| {
                let client = Arc::new(AioClient::new(endpoint, config)?);
                let operation: Arc<dyn CodeOperation> = Arc::new(AioCode { client });
                Ok(operation)
            },
        ))
}

/// The envelope client shared by the three operation wrappers.
pub(crate) struct AioClient {
    /// HTTP client carrying the base URL, auth headers, and auth query.
    http: SandboxHttp,
    /// Path the envelope is posted to.
    invoke_path: String,
    /// Resolved isolation key sent with every request (`null` when absent).
    isolation_key: Option<String>,
}

impl AioClient {
    /// Builds a client for `endpoint`, honoring `launcher_config` overrides.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::Config`](openjiuwen_common::error::ErrorKind::Config)
    /// when neither the endpoint nor the launcher configuration carries a base
    /// URL, or when the HTTP client cannot be built.
    pub(crate) fn new(endpoint: &SandboxEndpoint, config: &SandboxGatewayConfig) -> Result<Self> {
        let http = SandboxHttp::new(PROVIDER, endpoint, config)?;
        Ok(Self {
            invoke_path: invoke_path(config),
            isolation_key: endpoint.isolation_key.clone(),
            http,
        })
    }

    /// Posts one envelope and returns its `data` payload.
    ///
    /// # Errors
    ///
    /// Returns [`RemoteError::Status`] with status `0` and the message
    /// `aio gateway code <code>: <message>` when the gateway answers a
    /// non-zero `code`, [`RemoteError::NotFound`] / [`RemoteError::Status`] for
    /// a non-2xx HTTP answer, and [`RemoteError::Transport`] when no usable
    /// response arrives.
    async fn invoke(
        &self,
        op_type: &str,
        method: &str,
        params: Value,
    ) -> std::result::Result<Value, RemoteError> {
        let envelope = GatewayInvokeRequest {
            op_type: op_type.to_string(),
            method: method.to_string(),
            params,
            isolation_key: self.isolation_key.clone(),
        };
        let request = self
            .http
            .request(reqwest::Method::POST, &self.invoke_path)
            .json(&envelope);
        let body = self.http.send_value(request, method).await?;
        let response: GatewayResponse = serde_json::from_value(body).map_err(|error| {
            RemoteError::Transport(format!(
                "{method}: could not decode the {PROVIDER} gateway response: {error}"
            ))
        })?;
        if response.is_success() {
            Ok(response.data.unwrap_or(Value::Null))
        } else {
            Err(RemoteError::Status {
                status: 0,
                message: format!(
                    "{PROVIDER} gateway code {}: {}",
                    response.code, response.message
                ),
            })
        }
    }
}

/// The kernel's ten [`FsOperation`] methods over the envelope.
pub(crate) struct AioFs {
    /// Shared envelope client.
    client: Arc<AioClient>,
}

impl AioFs {
    /// The envelope parameters of `read_file`.
    async fn read_remote(
        &self,
        params: &ReadFileParams,
    ) -> std::result::Result<ReadFileData, RemoteError> {
        let payload = read_file_params(params);
        let data = self.client.invoke("fs", "read_file", payload).await?;
        read_file_data(&params.path, params.mode, &data)
    }

    /// Reads a remote file as raw bytes (`mode: "bytes"`).
    async fn read_remote_bytes(&self, path: &str) -> std::result::Result<Vec<u8>, RemoteError> {
        let payload = read_file_params(&ReadFileParams::bytes(path));
        let data = self.client.invoke("fs", "read_file", payload).await?;
        decode_transfer_content(&data)
    }

    /// Whether the remote path exists, probed through a bytes-mode read.
    async fn remote_exists(&self, path: &str) -> std::result::Result<bool, RemoteError> {
        let payload = read_file_params(&ReadFileParams::bytes(path));
        match self.client.invoke("fs", "read_file", payload).await {
            Ok(_) => Ok(true),
            Err(error) if is_absent(&error) => Ok(false),
            Err(error) => Err(error),
        }
    }

    /// Writes `bytes` to `path` with `mode: "bytes"` (the transfer payload).
    async fn write_remote_bytes(
        &self,
        path: &str,
        bytes: &[u8],
    ) -> std::result::Result<(), RemoteError> {
        let payload = json!({
            "path": path,
            "content": base64_encode(bytes),
            "mode": ReadMode::Bytes.as_str(),
            "prepend_newline": false,
            "append_newline": false,
            "append": false,
            "create_if_not_exist": true,
            "permissions": Value::Null,
            "encoding": BASE64_ENCODING,
        });
        self.client
            .invoke("fs", "write_file", payload)
            .await
            .map(|_| ())
    }

    /// Runs one listing/search call and maps its entries.
    async fn entries(
        &self,
        method: &str,
        payload: Value,
    ) -> std::result::Result<Vec<FileSystemItem>, RemoteError> {
        let data = self.client.invoke("fs", method, payload).await?;
        Ok(response_entries(&data)
            .iter()
            .map(|entry| file_system_item(entry))
            .collect())
    }
}

#[async_trait]
impl FsOperation for AioFs {
    async fn read_file(&self, params: ReadFileParams) -> OpResult<ReadFileData> {
        if let Err(reason) =
            normalize_line_selection(params.mode, params.head, params.tail, params.line_range)
        {
            return fs_failure("read_file", &reason);
        }
        match self.read_remote(&params).await {
            Ok(data) => OpResult::success(data),
            Err(error) => call_failure("read_file", &error),
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
        let conditional = if params.expected_content_sha256.is_some() {
            Some("expected_content_sha256")
        } else if params.create_if_absent {
            Some("create_if_absent")
        } else {
            None
        };
        if let Some(field) = conditional {
            return fs_failure(
                "write_file",
                &format!(
                    "{field} is not supported by the {PROVIDER} provider \
                     (remote sandbox transports have no compare-and-swap)"
                ),
            );
        }
        let mode = params.content.mode();
        let (content, encoding, payload_len) = match &params.content {
            FileContent::Text(text) => (text.clone(), params.encoding.clone(), text.len()),
            FileContent::Bytes(bytes) => (
                base64_encode(bytes),
                BASE64_ENCODING.to_string(),
                bytes.len(),
            ),
        };
        let payload = json!({
            "path": params.path,
            "content": content,
            "mode": mode.as_str(),
            "prepend_newline": params.prepend_newline,
            "append_newline": params.append_newline,
            "append": params.append,
            "create_if_not_exist": params.create_if_not_exist,
            "permissions": params.permissions,
            "encoding": encoding,
        });
        let data = match self.client.invoke("fs", "write_file", payload).await {
            Ok(data) => data,
            Err(error) => return call_failure("write_file", &error),
        };
        OpResult::success(WriteFileData {
            path: text_field(&data, "path").unwrap_or_else(|| params.path.clone()),
            size: json_u64(data.get("size")).unwrap_or(len_u64(payload_len)),
            mode: parse_mode(data.get("mode")).unwrap_or(mode),
        })
    }

    async fn upload_file(&self, params: UploadFileParams) -> OpResult<UploadFileData> {
        let bytes = match read_host_file(&params.local_path).await {
            Ok(bytes) => bytes,
            Err(reason) => return fs_failure("upload_file", &reason),
        };
        if !params.overwrite {
            match self.remote_exists(&params.target_path).await {
                Ok(true) => {
                    return fs_failure(
                        "upload_file",
                        &format!("File already exists: {}", params.target_path),
                    )
                }
                Ok(false) => {}
                Err(error) => return call_failure("upload_file", &error),
            }
        }
        let size = len_u64(bytes.len());
        match self.write_remote_bytes(&params.target_path, &bytes).await {
            Ok(()) => OpResult::success(UploadFileData {
                local_path: params.local_path,
                target_path: params.target_path,
                size,
            }),
            Err(error) => call_failure("upload_file", &error),
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
        let bytes = match self.read_remote_bytes(&params.source_path).await {
            Ok(bytes) => bytes,
            Err(error) => return call_failure("download_file", &error),
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
        let payload = json!({
            "path": params.path,
            "recursive": params.recursive,
            "max_depth": params.max_depth,
            "sort_by": sort_by_name(params.sort_by),
            "sort_descending": params.sort_descending,
            "file_types": params.file_types,
        });
        let entries = match self.entries("list_files", payload).await {
            Ok(entries) => entries,
            Err(error) => return call_failure("list_files", &error),
        };
        let files: Vec<FileSystemItem> = entries
            .into_iter()
            .filter(|entry| !entry.is_directory)
            .collect();
        let mut items = filter_file_types(files, &params.file_types);
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
        let payload = json!({
            "path": params.path,
            "recursive": params.recursive,
            "max_depth": params.max_depth,
            "sort_by": sort_by_name(params.sort_by),
            "sort_descending": params.sort_descending,
        });
        let entries = match self.entries("list_directories", payload).await {
            Ok(entries) => entries,
            Err(error) => return call_failure("list_directories", &error),
        };
        let mut items: Vec<FileSystemItem> = entries
            .into_iter()
            .filter(|entry| entry.is_directory)
            .map(|entry| FileSystemItem {
                size: 0,
                file_type: None,
                ..entry
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
        let payload = json!({
            "path": params.path,
            "pattern": params.pattern,
            "exclude_patterns": params.exclude_patterns,
        });
        let mut items = match self.entries("search_files", payload).await {
            Ok(entries) => entries,
            Err(error) => return call_failure("search_files", &error),
        };
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

/// The kernel's four [`ShellOperation`] methods over the envelope.
pub(crate) struct AioShell {
    /// Shared envelope client.
    client: Arc<AioClient>,
}

impl AioShell {
    /// Sends one `execute_cmd` envelope and maps its output.
    ///
    /// `operation` names the calling kernel method in failure messages, so
    /// `execute_argv` reports itself rather than `execute_cmd`.
    async fn run_command(
        &self,
        operation: &str,
        command: &str,
        cwd: Option<&str>,
        timeout: u64,
        environment: &BTreeMap<String, Option<String>>,
        shell_type: ShellType,
    ) -> OpResult<ShellOutput> {
        let payload = json!({
            "command": command,
            "cwd": cwd,
            "timeout": timeout,
            "environment": environment,
            "shell_type": shell_type.as_str(),
        });
        let data = match self.client.invoke("shell", "execute_cmd", payload).await {
            Ok(data) => data,
            Err(error) => return shell_failure(operation, &reason(&error)),
        };
        let output = shell_output(command, cwd, &data);
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
impl ShellOperation for AioShell {
    async fn execute_cmd(&self, params: ShellExecParams) -> OpResult<ShellOutput> {
        if params.command.trim().is_empty() {
            return shell_failure("execute_cmd", "command can not be empty");
        }
        let timeout = params.effective_timeout_secs(DEFAULT_TIMEOUT_SECS);
        let cwd = normalized_cwd(params.cwd.as_deref());
        self.run_command(
            "execute_cmd",
            &params.command,
            cwd.as_deref(),
            timeout,
            &params.environment,
            params.shell_type,
        )
        .await
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
        let cwd = normalized_cwd(params.cwd.as_deref());
        let payload = json!({
            "command": params.command,
            "cwd": cwd,
            "environment": params.environment,
            "shell_type": params.shell_type.as_str(),
        });
        let data = match self
            .client
            .invoke("shell", "execute_cmd_background", payload)
            .await
        {
            Ok(data) => data,
            Err(error) => return shell_failure("execute_cmd_background", &reason(&error)),
        };
        OpResult::success(ShellBackgroundOutput {
            command: text_field(&data, "command").unwrap_or_else(|| params.command.clone()),
            cwd: text_field(&data, "cwd").unwrap_or_else(|| cwd.unwrap_or_else(|| ".".to_string())),
            pid: json_u64(data.get("pid")).and_then(|pid| u32::try_from(pid).ok()),
            error: text_field(&data, "error"),
        })
    }

    async fn execute_argv(&self, params: ShellArgvParams) -> OpResult<ShellOutput> {
        if params.argv.is_empty() {
            return shell_failure("execute_argv", "argv can not be empty");
        }
        let timeout = params.effective_timeout_secs(DEFAULT_TIMEOUT_SECS);
        let cwd = normalized_cwd(params.cwd.as_deref());
        let command = params
            .argv
            .iter()
            .map(|argument| posix_quote(argument))
            .collect::<Vec<_>>()
            .join(" ");
        self.run_command(
            "execute_argv",
            &command,
            cwd.as_deref(),
            timeout,
            &params.environment,
            ShellType::Auto,
        )
        .await
    }
}

/// The kernel's two [`CodeOperation`] methods over the envelope.
pub(crate) struct AioCode {
    /// Shared envelope client.
    client: Arc<AioClient>,
}

#[async_trait]
impl CodeOperation for AioCode {
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
        let timeout = params.effective_timeout_secs(DEFAULT_TIMEOUT_SECS);
        let payload = json!({
            "code": params.code,
            "language": language.as_str(),
            "timeout": timeout,
            "environment": params.environment,
            "cwd": params.cwd,
            "options": {"force_file": params.force_file},
        });
        let data = match self.client.invoke("code", "execute_code", payload).await {
            Ok(data) => data,
            Err(error) => return code_failure("execute_code", &reason(&error)),
        };
        let output = code_output(&params.code, language, &data);
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

/// The envelope path, honoring the launcher's `invoke_path` override.
fn invoke_path(config: &SandboxGatewayConfig) -> String {
    let configured = config
        .launcher_config
        .as_ref()
        .and_then(|launcher| launcher.extra_params.get(INVOKE_PATH_PARAM))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|path| !path.is_empty());
    match configured {
        Some(path) if path.starts_with('/') => path.to_string(),
        Some(path) => format!("/{path}"),
        None => DEFAULT_INVOKE_PATH.to_string(),
    }
}

/// The envelope parameters of `read_file` (Python `read_file(path, mode, ...)`).
fn read_file_params(params: &ReadFileParams) -> Value {
    json!({
        "path": params.path,
        "mode": params.mode.as_str(),
        "head": params.head,
        "tail": params.tail,
        "line_range": params.line_range.map(|range| vec![range.start, range.end]),
        "encoding": params.encoding,
        "chunk_size": params.chunk_size,
    })
}

/// Maps a `read_file` payload onto the kernel's read data.
fn read_file_data(
    path: &str,
    mode: ReadMode,
    data: &Value,
) -> std::result::Result<ReadFileData, RemoteError> {
    let mode = parse_mode(data.get("mode")).unwrap_or(mode);
    let content = match data.get("content") {
        Some(Value::String(text)) => FileContent::Text(text.clone()),
        Some(Value::Array(_)) => FileContent::Bytes(bytes_from_value(data.get("content"))?),
        Some(Value::Null) | None => empty_content(mode),
        Some(other) => {
            return Err(RemoteError::Transport(format!(
                "read_file: unexpected content in the {PROVIDER} response: {other}"
            )))
        }
    };
    Ok(ReadFileData {
        path: text_field(data, "path").unwrap_or_else(|| path.to_string()),
        total_bytes: len_u64(content.byte_len()),
        content,
        mode,
        encoding: None,
        truncated: false,
    })
}

/// Decodes the bytes a `read_file` answered for a transfer.
///
/// A byte array passes through; a string is decoded per its
/// `content_encoding` (`"hex"` / `"base64"`), defaulting to base64 and falling
/// back to the string's UTF-8 bytes when it is not valid base64.
fn decode_transfer_content(data: &Value) -> std::result::Result<Vec<u8>, RemoteError> {
    let encoding = text_field(data, "content_encoding")
        .map(|encoding| encoding.to_ascii_lowercase())
        .filter(|encoding| !encoding.is_empty());
    match data.get("content") {
        Some(Value::Array(_)) => bytes_from_value(data.get("content")),
        Some(Value::String(text)) => match encoding.as_deref() {
            Some("hex") => hex_decode(text),
            Some("base64") | None => base64_decode(text)
                .or_else(|_| Ok::<Vec<u8>, RemoteError>(text.clone().into_bytes())),
            Some(other) => Err(RemoteError::Invalid(format!(
                "unsupported content_encoding '{other}'"
            ))),
        },
        Some(Value::Null) | None => Ok(Vec::new()),
        Some(other) => Err(RemoteError::Transport(format!(
            "download_file: unexpected content in the {PROVIDER} response: {other}"
        ))),
    }
}

/// Reads a JSON array of `0..=255` values as bytes.
fn bytes_from_value(value: Option<&Value>) -> std::result::Result<Vec<u8>, RemoteError> {
    let items = value
        .and_then(Value::as_array)
        .map_or(&[][..], Vec::as_slice);
    let mut bytes = Vec::with_capacity(items.len());
    for item in items {
        let number = item.as_u64().ok_or_else(|| {
            RemoteError::Transport(format!("content byte {item} is not an unsigned integer"))
        })?;
        let byte = u8::try_from(number).map_err(|_| {
            RemoteError::Transport(format!("content byte {number} is out of range"))
        })?;
        bytes.push(byte);
    }
    Ok(bytes)
}

/// An empty content value of the requested mode.
fn empty_content(mode: ReadMode) -> FileContent {
    match mode {
        ReadMode::Text => FileContent::Text(String::new()),
        ReadMode::Bytes => FileContent::Bytes(Vec::new()),
    }
}

/// The entries a listing or search payload carries.
///
/// Accepts Python's `FileSystemData` (`list_items`) as well as the tolerant
/// `items` and `matching_files` shapes.
fn response_entries(data: &Value) -> Vec<&Value> {
    ["list_items", "items", "matching_files"]
        .iter()
        .find_map(|key| match data.get(*key) {
            Some(Value::Array(items)) => Some(items.iter().collect()),
            _ => None,
        })
        .unwrap_or_default()
}

/// Maps one wire entry onto the kernel's listing item.
fn file_system_item(entry: &Value) -> FileSystemItem {
    let name = text_field(entry, "name").unwrap_or_default();
    let is_directory = entry
        .get("is_directory")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let file_type = text_field(entry, "type")
        .or_else(|| (!is_directory).then(|| file_extension(&name)).flatten());
    FileSystemItem {
        path: text_field(entry, "path").unwrap_or_else(|| name.clone()),
        size: json_u64(entry.get("size")).unwrap_or(0),
        modified_time: text_field(entry, "modified_time").unwrap_or_else(|| "0".to_string()),
        name,
        is_directory,
        file_type,
    }
}

/// Maps an `execute_cmd` payload onto the kernel's shell output.
fn shell_output(request_command: &str, cwd: Option<&str>, data: &Value) -> ShellOutput {
    let exit_code = json_i32(data.get("exit_code")).unwrap_or(0);
    ShellOutput {
        command: text_field(data, "command").unwrap_or_else(|| request_command.to_string()),
        cwd: text_field(data, "cwd").unwrap_or_else(|| cwd.unwrap_or(".").to_string()),
        exit_code,
        stdout: text_field(data, "stdout")
            .or_else(|| text_field(data, "output"))
            .unwrap_or_default(),
        stderr: text_field(data, "stderr").unwrap_or_default(),
        truncated: false,
        timed_out: exit_code == TIMEOUT_EXIT_CODE,
        cancelled: false,
        duration_ms: json_u64(data.get("duration_ms")).unwrap_or(0),
    }
}

/// Maps an `execute_code` payload onto the kernel's code output.
fn code_output(request_code: &str, language: Language, data: &Value) -> CodeOutput {
    let exit_code = json_i32(data.get("exit_code")).unwrap_or(-1);
    let stderr = text_field(data, "stderr").unwrap_or_default();
    CodeOutput {
        code_content: text_field(data, "code_content").unwrap_or_else(|| request_code.to_string()),
        language: text_field(data, "language").unwrap_or_else(|| language.as_str().to_string()),
        exit_code,
        stdout: text_field(data, "stdout").unwrap_or_default(),
        timed_out: exit_code == TIMEOUT_EXIT_CODE
            || stderr.to_ascii_lowercase().contains("timed out"),
        stderr,
        truncated: false,
        duration_ms: json_u64(data.get("duration_ms")).unwrap_or(0),
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
        chunk: empty_content(mode),
        mode,
        chunk_size: 0,
        chunk_index: 0,
        is_last_chunk: true,
    }
}

/// Splits a completed shell run into stdout lines, stderr lines, and the
/// terminal chunk.
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

/// Splits a completed code run into stdout lines, stderr lines, and the
/// terminal chunk.
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

/// The failure reason of a gateway call.
///
/// An envelope rejection travels as [`RemoteError::Status`] with status `0`
/// because no HTTP status is involved, and [`RemoteError::message`] would
/// report it as `HTTP 0: ...`; that prefix is dropped so the reason reads
/// exactly like the gateway's own message. Every other failure keeps the
/// transport rendering (including `HTTP 404` and connection errors).
fn reason(error: &RemoteError) -> String {
    match error {
        RemoteError::Status { status: 0, message } => message.clone(),
        other => other.message(),
    }
}
/// Renders a failed gateway call as the kernel's sandbox gateway status.
///
/// This is [`super::transport::gateway_failure`] with the envelope rejection
/// recognised as such (see [`reason`]); the wrappers therefore answer `199007`
/// with `sandbox gateway error, operation: <op>, reason: <reason>`.
fn call_failure<T>(operation: &str, error: &RemoteError) -> OpResult<T> {
    OpResult::failure(
        SysOpStatus::SandboxGatewayError,
        sandbox_gateway_message(operation, &reason(error)),
    )
}
/// Whether a probe failure means the remote path is absent.
///
/// The envelope signals a missing path as `code != 0` rather than as HTTP
/// `404`, so besides the transport status the gateway message is the only
/// signal available.
fn is_absent(error: &RemoteError) -> bool {
    if error.is_not_found() {
        return true;
    }
    let message = error.message().to_ascii_lowercase();
    [
        "not found",
        "no such file",
        "does not exist",
        "not exist",
        "no exists",
    ]
    .iter()
    .any(|needle| message.contains(needle))
}

/// Parses `"text"` / `"bytes"` into a [`ReadMode`].
fn parse_mode(value: Option<&Value>) -> Option<ReadMode> {
    match value?.as_str()?.trim().to_ascii_lowercase().as_str() {
        "text" => Some(ReadMode::Text),
        "bytes" => Some(ReadMode::Bytes),
        _ => None,
    }
}

/// The wire name of a sort key.
fn sort_by_name(sort_by: SortBy) -> &'static str {
    match sort_by {
        SortBy::Name => "name",
        SortBy::ModifiedTime => "modified_time",
        SortBy::Size => "size",
    }
}

/// A non-empty string field, if present.
fn text_field(value: &Value, key: &str) -> Option<String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .map(str::to_string)
        .filter(|text| !text.is_empty())
}

/// An unsigned integer field, accepting a numeric string too.
fn json_u64(value: Option<&Value>) -> Option<u64> {
    match value? {
        Value::Number(number) => number
            .as_u64()
            .or_else(|| number.as_i64().and_then(|value| u64::try_from(value).ok())),
        Value::String(text) => text.trim().parse().ok(),
        _ => None,
    }
}

/// A signed 32-bit integer field, accepting a numeric string too.
fn json_i32(value: Option<&Value>) -> Option<i32> {
    match value? {
        Value::Number(number) => number.as_i64().and_then(|value| i32::try_from(value).ok()),
        Value::String(text) => text.trim().parse().ok(),
        _ => None,
    }
}

/// Normalizes the caller's working directory the way the Python providers do.
fn normalized_cwd(cwd: Option<&str>) -> Option<String> {
    cwd.filter(|value| !value.is_empty() && *value != ".")
        .map(str::to_string)
}

/// Converts a length or index to the `u64` the result types use.
fn len_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}
