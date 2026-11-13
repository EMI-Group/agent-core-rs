//! The kernel's [`FsOperation`] implemented once over a [`SandboxBackend`].
//!
//! Python parity: every provider in
//! `openjiuwen/extensions/sys_operation/sandbox/providers/` repeats the same host-side
//! filesystem behavior — line selection, newline transforms, transfer I/O, sorting, and
//! type filtering — around its own wire calls. [`RemoteFs`] performs that work once and
//! leaves the provider with `read_bytes` / `write_bytes` / `list` / `search`.

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use openjiuwen_core::sys_operation::constants::DEFAULT_ENCODING;
use openjiuwen_core::sys_operation::fs::types::{
    DownloadFileData, DownloadFileParams, FileContent, FileSystemItem, ListDirsParams,
    ListFilesData, ListFilesParams, ReadChunkData, ReadFileData, ReadFileParams, ReadMode,
    SearchFilesData, SearchFilesParams, SortBy, TransferChunkData, UploadFileData,
    UploadFileParams, WriteFileData, WriteFileParams,
};
use openjiuwen_core::sys_operation::operation::{FsOperation, OpStream};
use openjiuwen_core::sys_operation::result::{fs_failure, OpResult};
use openjiuwen_core::sys_operation::sandbox::config::SandboxGatewayConfig;

use super::super::local_io::{read_host_file, write_host_file, HostIoError};
use super::super::semantics::{
    apply_text_newlines, byte_chunks, filter_file_types, normalize_line_selection,
    select_text_lines, sort_items, text_lines,
};
use super::super::transport::gateway_failure;
use super::{len_u64, ListRequest, RemoteEntry, SandboxBackend, SearchRequest};

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
