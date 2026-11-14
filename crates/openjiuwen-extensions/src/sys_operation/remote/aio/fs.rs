//! The kernel's [`FsOperation`] over the AIO gateway envelope.
//!
//! Python parity: `openjiuwen/extensions/sys_operation/sandbox/providers/aio.py`
//! registers an `fs` provider whose `params` are the Python method's keyword
//! arguments. [`AioFs`] sends those envelopes, maps the `data` payload back into
//! the kernel result structs, and chunks streamed reads on the host (the
//! envelope has no streaming variant).

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{json, Value};

use openjiuwen_core::sys_operation::fs::types::{
    DownloadFileData, DownloadFileParams, FileContent, FileSystemItem, ListDirsParams,
    ListFilesData, ListFilesParams, ReadChunkData, ReadFileData, ReadFileParams, ReadMode,
    SearchFilesData, SearchFilesParams, SortBy, TransferChunkData, UploadFileData,
    UploadFileParams, WriteFileData, WriteFileParams,
};
use openjiuwen_core::sys_operation::operation::{FsOperation, OpStream};
use openjiuwen_core::sys_operation::result::{fs_failure, OpResult};

use super::super::conditional::{lock_path, Precondition};
use super::super::local_io::{read_host_file, write_host_file, HostIoError};
use super::super::semantics::{
    byte_chunks, file_extension, filter_file_types, normalize_line_selection, sort_items,
    text_lines,
};
use super::super::transport::{
    base64_decode, base64_encode, gateway_failure, hex_decode, RemoteError,
};
use super::{is_absent, json_u64, text_field, AioClient, BASE64_ENCODING, PROVIDER};

/// The kernel's ten [`FsOperation`] methods over the envelope.
pub(crate) struct AioFs {
    /// Shared envelope client.
    pub(super) client: Arc<AioClient>,
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
    fn supports_conditional_write(&self) -> bool {
        // The provider evaluates the precondition itself (read, check, write),
        // so the capability is its own and does not depend on the gateway
        // configuration.
        true
    }

    async fn read_file(&self, params: ReadFileParams) -> OpResult<ReadFileData> {
        if let Err(reason) =
            normalize_line_selection(params.mode, params.head, params.tail, params.line_range)
        {
            return fs_failure("read_file", &reason);
        }
        match self.read_remote(&params).await {
            Ok(data) => OpResult::success(data),
            Err(error) => gateway_failure("read_file", &error),
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
        let precondition = Precondition::from_params(&params);
        if let Err(reason) = precondition.validate() {
            return fs_failure("write_file", &reason);
        }
        let conditional = precondition.is_conditional();
        // The lock makes the read-check-write sequence atomic within this
        // process; it is only needed when a precondition must be evaluated.
        let guard = if conditional {
            Some(lock_path(PROVIDER, &params.path).await)
        } else {
            None
        };
        if conditional {
            let current = match self.read_remote_bytes(&params.path).await {
                Ok(bytes) => Some(bytes),
                Err(error) if is_absent(&error) => None,
                Err(error) => return gateway_failure("write_file", &error),
            };
            if let Err(reason) = precondition.check(&params.path, current.as_deref()) {
                // No write request is issued for a rejected precondition.
                return fs_failure("write_file", &reason);
            }
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
            Err(error) => return gateway_failure("write_file", &error),
        };
        let result = OpResult::success(WriteFileData {
            path: text_field(&data, "path").unwrap_or_else(|| params.path.clone()),
            size: json_u64(data.get("size")).unwrap_or(len_u64(payload_len)),
            mode: parse_mode(data.get("mode")).unwrap_or(mode),
        });
        drop(guard);
        result
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
                Err(error) => return gateway_failure("upload_file", &error),
            }
        }
        let size = len_u64(bytes.len());
        match self.write_remote_bytes(&params.target_path, &bytes).await {
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
        let bytes = match self.read_remote_bytes(&params.source_path).await {
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
            Err(error) => return gateway_failure("list_files", &error),
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
            Err(error) => return gateway_failure("list_directories", &error),
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
            Err(error) => return gateway_failure("search_files", &error),
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

/// Converts a length or index to the `u64` the result types use.
fn len_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}
