//! Filesystem parameter and result types (Python `core/sys_operation/result/fs_operation_result.py`).
//!
//! Every operation takes a parameter struct and returns an [`crate::result::OpResult`]
//! wrapping a data struct. Parameter structs derive `Deserialize` with serde defaults, so
//! a tool invocation can deserialize a partial JSON object straight into them, and
//! `JsonSchema` so tool cards are generated from a single source of truth.
//!
//! ```
//! use openjiuwen_sysop::fs::types::ReadFileParams;
//!
//! let params = ReadFileParams::text("notes.txt");
//! assert_eq!(params.path, "notes.txt");
//! assert_eq!(params.encoding, "utf-8");
//! ```

use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::config::BinaryPolicy;
use crate::constants::{DEFAULT_ENCODING, DEFAULT_PERMISSIONS};

/// How file content is interpreted.
///
/// Python parity: the `mode: Literal['text', 'bytes']` argument of every read/write call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum ReadMode {
    /// Decode the bytes with the requested encoding (default).
    #[default]
    Text,
    /// Return the raw bytes.
    Bytes,
}

impl ReadMode {
    /// The serialized name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Bytes => "bytes",
        }
    }

    /// Returns `true` for [`ReadMode::Text`].
    #[must_use]
    pub const fn is_text(self) -> bool {
        matches!(self, Self::Text)
    }
}

/// File content, either decoded text or raw bytes.
///
/// Serializes as a plain JSON string for [`FileContent::Text`] and as an array of byte
/// values for [`FileContent::Bytes`] (Python returns bytes that its JSON encoder renders
/// the same way through pydantic's `arbitrary_types_allowed` models).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum FileContent {
    /// Decoded text.
    Text(String),
    /// Raw bytes.
    Bytes(Vec<u8>),
}

impl FileContent {
    /// Builds text content.
    #[must_use]
    pub fn text(content: impl Into<String>) -> Self {
        Self::Text(content.into())
    }

    /// Builds byte content.
    #[must_use]
    pub fn bytes(content: impl Into<Vec<u8>>) -> Self {
        Self::Bytes(content.into())
    }

    /// Returns `true` when the content is text.
    #[must_use]
    pub const fn is_text(&self) -> bool {
        matches!(self, Self::Text(_))
    }

    /// Returns the text, or `None` for bytes.
    #[must_use]
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(text) => Some(text),
            Self::Bytes(_) => None,
        }
    }

    /// Returns the bytes, or `None` for text.
    #[must_use]
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            Self::Text(_) => None,
            Self::Bytes(bytes) => Some(bytes),
        }
    }

    /// Returns the number of bytes the content occupies once encoded.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        match self {
            Self::Text(text) => text.len(),
            Self::Bytes(bytes) => bytes.len(),
        }
    }

    /// The matching [`ReadMode`].
    #[must_use]
    pub const fn mode(&self) -> ReadMode {
        match self {
            Self::Text(_) => ReadMode::Text,
            Self::Bytes(_) => ReadMode::Bytes,
        }
    }
}

impl From<String> for FileContent {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<&str> for FileContent {
    fn from(value: &str) -> Self {
        Self::Text(value.to_string())
    }
}

impl From<Vec<u8>> for FileContent {
    fn from(value: Vec<u8>) -> Self {
        Self::Bytes(value)
    }
}

/// A 1-based, inclusive line range (`line_range=(start, end)` in Python).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct LineRange {
    /// First line to keep, 1-based.
    pub start: usize,
    /// Last line to keep, 1-based and inclusive.
    pub end: usize,
}

impl LineRange {
    /// Builds a range.
    #[must_use]
    pub const fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }
}

/// Parameters of `read_file` / `read_file_stream`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct ReadFileParams {
    /// Path to read, absolute or relative to the workspace root.
    pub path: String,
    /// Text or bytes.
    pub mode: ReadMode,
    /// Keep only the first `head` lines (text mode only).
    pub head: Option<usize>,
    /// Keep only the last `tail` lines (text mode only).
    pub tail: Option<usize>,
    /// Keep only the given 1-based inclusive line range (text mode only).
    pub line_range: Option<LineRange>,
    /// Text encoding; `"auto"` selects [`crate::encoding::detect_text_encoding`].
    pub encoding: String,
    /// Byte chunk size; `0` reads the whole file.
    pub chunk_size: usize,
    /// Take a shared read lock instead of an exclusive one (Python `only_read`).
    pub only_read: bool,
    /// What to do when the content looks binary in text mode.
    pub binary: BinaryPolicy,
}

impl Default for ReadFileParams {
    fn default() -> Self {
        Self {
            path: String::new(),
            mode: ReadMode::Text,
            head: None,
            tail: None,
            line_range: None,
            encoding: DEFAULT_ENCODING.to_string(),
            chunk_size: 0,
            only_read: false,
            binary: BinaryPolicy::Lossy,
        }
    }
}

impl ReadFileParams {
    /// Text-mode parameters for `path`.
    #[must_use]
    pub fn text(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            ..Self::default()
        }
    }

    /// Bytes-mode parameters for `path`.
    #[must_use]
    pub fn bytes(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            mode: ReadMode::Bytes,
            ..Self::default()
        }
    }

    /// Returns these parameters with `head` set.
    #[must_use]
    pub fn with_head(mut self, head: usize) -> Self {
        self.head = Some(head);
        self
    }

    /// Returns these parameters with `tail` set.
    #[must_use]
    pub fn with_tail(mut self, tail: usize) -> Self {
        self.tail = Some(tail);
        self
    }

    /// Returns these parameters with `line_range` set.
    #[must_use]
    pub fn with_line_range(mut self, start: usize, end: usize) -> Self {
        self.line_range = Some(LineRange::new(start, end));
        self
    }

    /// Returns these parameters with an explicit encoding.
    #[must_use]
    pub fn with_encoding(mut self, encoding: impl Into<String>) -> Self {
        self.encoding = encoding.into();
        self
    }
}

/// Data of `read_file` (Python `ReadFileData`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct ReadFileData {
    /// Resolved path that was read.
    pub path: String,
    /// File content.
    pub content: FileContent,
    /// Text or bytes.
    pub mode: ReadMode,
    /// Encoding used, set when detection (`"auto"`) was requested.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding: Option<String>,
    /// `true` when `chunk_size` truncated the content (bytes mode).
    pub truncated: bool,
    /// Total size of the file in bytes.
    pub total_bytes: u64,
}

impl Default for ReadFileData {
    fn default() -> Self {
        Self {
            path: String::new(),
            content: FileContent::Text(String::new()),
            mode: ReadMode::Text,
            encoding: None,
            truncated: false,
            total_bytes: 0,
        }
    }
}

/// Parameters of the bounded read (`FsOperation::read_bytes_limited`).
///
/// Rust addition: Python's `read_file` always loads the whole file. This request reads at
/// most [`Self::max_bytes`] bytes starting at [`Self::offset`] without touching the rest of
/// the file, so the memory a read needs is bounded by the caller rather than by the file.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct ReadBytesParams {
    /// Path to read, absolute or relative to the workspace root.
    pub path: String,
    /// Byte offset the window starts at; `0` starts at the beginning of the file.
    pub offset: u64,
    /// Maximum number of bytes to return; `0` returns no bytes (only the size facts).
    pub max_bytes: u64,
    /// Skip both lock layers (Python `skip_lock`); default `false`.
    pub only_read: bool,
}

impl ReadBytesParams {
    /// Bounded parameters for the window `[offset, offset + max_bytes)` of `path`.
    #[must_use]
    pub fn new(path: impl Into<String>, offset: u64, max_bytes: u64) -> Self {
        Self {
            path: path.into(),
            offset,
            max_bytes,
            only_read: false,
        }
    }

    /// Returns these parameters with [`Self::only_read`] set.
    #[must_use]
    pub fn with_only_read(mut self, only_read: bool) -> Self {
        self.only_read = only_read;
        self
    }
}

/// Data of the bounded read (`FsOperation::read_bytes_limited`).
///
/// [`Self::bytes`] never holds more than the requested window, [`Self::total_bytes`] is the
/// real length of the whole file, and [`Self::truncated`] says whether bytes were left
/// unread past the window.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct ReadBytesData {
    /// Resolved path that was read.
    pub path: String,
    /// The window's bytes, at most `ReadBytesParams::max_bytes` of them.
    pub bytes: Vec<u8>,
    /// Full size of the file in bytes.
    pub total_bytes: u64,
    /// `true` when the window ended before the end of the file.
    pub truncated: bool,
}

/// One chunk of `read_file_stream` (Python `ReadFileChunkData`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct ReadChunkData {
    /// Resolved path being read.
    pub path: String,
    /// Chunk payload; one line per chunk in text mode.
    pub chunk: FileContent,
    /// Text or bytes.
    pub mode: ReadMode,
    /// Size of this chunk in bytes.
    pub chunk_size: u64,
    /// Zero-based chunk index.
    pub chunk_index: u64,
    /// `true` for the final chunk.
    pub is_last_chunk: bool,
}

impl Default for ReadChunkData {
    fn default() -> Self {
        Self {
            path: String::new(),
            chunk: FileContent::Text(String::new()),
            mode: ReadMode::Text,
            chunk_size: 0,
            chunk_index: 0,
            is_last_chunk: true,
        }
    }
}

/// Parameters of `write_file`.
///
/// The five write flags mirror Python's keyword-only arguments one-to-one, which is why
/// this struct intentionally carries more than three booleans.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct WriteFileParams {
    /// Destination path.
    pub path: String,
    /// Content to write; the variant decides text vs bytes.
    pub content: FileContent,
    /// Prefix a newline to text content (Python default `true`).
    pub prepend_newline: bool,
    /// Append a newline to text content (Python default `false`).
    pub append_newline: bool,
    /// Append instead of overwriting (Python default `false`).
    pub append: bool,
    /// Create the file when it does not exist (Python default `true`).
    pub create_if_not_exist: bool,
    /// Octal permission string applied on Unix (Python default `"644"`).
    pub permissions: Option<String>,
    /// Encoding used to encode text content.
    pub encoding: String,
    /// Reject the write unless the current content hashes to this SHA-256 hex digest.
    pub expected_content_sha256: Option<String>,
    /// Create the file only if it does not exist yet (compare-and-swap variant).
    pub create_if_absent: bool,
}

impl Default for WriteFileParams {
    fn default() -> Self {
        Self {
            path: String::new(),
            content: FileContent::Text(String::new()),
            prepend_newline: true,
            append_newline: false,
            append: false,
            create_if_not_exist: true,
            permissions: Some(DEFAULT_PERMISSIONS.to_string()),
            encoding: DEFAULT_ENCODING.to_string(),
            expected_content_sha256: None,
            create_if_absent: false,
        }
    }
}

impl WriteFileParams {
    /// Overwrite `path` with `content`, using Python's newline defaults.
    #[must_use]
    pub fn new(path: impl Into<String>, content: impl Into<FileContent>) -> Self {
        Self {
            path: path.into(),
            content: content.into(),
            ..Self::default()
        }
    }

    /// Overwrite `path` with `content` and no leading newline (`prepend_newline = false`).
    ///
    /// Python's default `prepend_newline = true` prefixes `"\n"` to every text write; use
    /// this constructor when the file must contain exactly `content`.
    #[must_use]
    pub fn raw(path: impl Into<String>, content: impl Into<FileContent>) -> Self {
        Self {
            path: path.into(),
            content: content.into(),
            prepend_newline: false,
            ..Self::default()
        }
    }

    /// Returns these parameters in append mode.
    #[must_use]
    pub fn with_append(mut self) -> Self {
        self.append = true;
        self
    }
    /// Returns these parameters with an explicit write encoding.
    ///
    /// Text content is encoded with it (see [`crate::encoding::encode_text`]); byte content
    /// ignores it.
    #[must_use]
    pub fn with_encoding(mut self, encoding: impl Into<String>) -> Self {
        self.encoding = encoding.into();
        self
    }

    /// Returns these parameters with an expected-content digest.
    #[must_use]
    pub fn with_expected_sha256(mut self, digest: impl Into<String>) -> Self {
        self.expected_content_sha256 = Some(digest.into());
        self
    }
}

/// Data of `write_file` (Python `WriteFileData`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct WriteFileData {
    /// Path written.
    pub path: String,
    /// Bytes written.
    pub size: u64,
    /// Text or bytes.
    pub mode: ReadMode,
}

impl Default for WriteFileData {
    fn default() -> Self {
        Self {
            path: String::new(),
            size: 0,
            mode: ReadMode::Text,
        }
    }
}

/// Parameters of `upload_file` / `upload_file_stream` (host → workspace).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct UploadFileParams {
    /// Source path on the host; not sandbox-checked (Python parity).
    pub local_path: String,
    /// Destination path inside the workspace; sandbox-checked.
    pub target_path: String,
    /// Overwrite an existing destination.
    pub overwrite: bool,
    /// Create missing destination parent directories.
    pub create_parent_dirs: bool,
    /// Copy the source permissions.
    pub preserve_permissions: bool,
    /// Chunk size in bytes; `0` copies in one step.
    pub chunk_size: u64,
}

impl Default for UploadFileParams {
    fn default() -> Self {
        Self {
            local_path: String::new(),
            target_path: String::new(),
            overwrite: false,
            create_parent_dirs: true,
            preserve_permissions: true,
            chunk_size: 0,
        }
    }
}

/// Data of `upload_file` (Python `UploadFileData`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "snake_case", default)]
pub struct UploadFileData {
    /// Source path.
    pub local_path: String,
    /// Destination path.
    pub target_path: String,
    /// Bytes copied.
    pub size: u64,
}

/// Parameters of `download_file` / `download_file_stream` (workspace → host).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct DownloadFileParams {
    /// Source path inside the workspace; sandbox-checked.
    pub source_path: String,
    /// Destination path on the host; not sandbox-checked.
    pub local_path: String,
    /// Overwrite an existing destination.
    pub overwrite: bool,
    /// Create missing destination parent directories.
    pub create_parent_dirs: bool,
    /// Copy the source permissions.
    pub preserve_permissions: bool,
    /// Chunk size in bytes; `0` copies in one step.
    pub chunk_size: u64,
}

impl Default for DownloadFileParams {
    fn default() -> Self {
        Self {
            source_path: String::new(),
            local_path: String::new(),
            overwrite: false,
            create_parent_dirs: true,
            preserve_permissions: true,
            chunk_size: 0,
        }
    }
}

/// Data of `download_file` (Python `DownloadFileData`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "snake_case", default)]
pub struct DownloadFileData {
    /// Source path.
    pub source_path: String,
    /// Destination path.
    pub local_path: String,
    /// Bytes copied.
    pub size: u64,
}

/// One chunk of `upload_file_stream` / `download_file_stream`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct TransferChunkData {
    /// Source path.
    pub source_path: String,
    /// Destination path.
    pub target_path: String,
    /// Size of this chunk in bytes.
    pub chunk_size: u64,
    /// Zero-based chunk index.
    pub chunk_index: u64,
    /// `true` for the final chunk.
    pub is_last_chunk: bool,
}

impl Default for TransferChunkData {
    fn default() -> Self {
        Self {
            source_path: String::new(),
            target_path: String::new(),
            chunk_size: 0,
            chunk_index: 0,
            is_last_chunk: true,
        }
    }
}

/// Sort key of directory listings.
///
/// Python parity: `sort_by: Literal['name', 'modified_time', 'size']`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SortBy {
    /// Sort by entry name (default).
    #[default]
    Name,
    /// Sort by modification timestamp.
    ModifiedTime,
    /// Sort by size in bytes.
    Size,
}

/// Parameters of `list_files`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "snake_case", default)]
pub struct ListFilesParams {
    /// Directory to list.
    pub path: String,
    /// Recurse into subdirectories.
    pub recursive: bool,
    /// Maximum recursion depth; `None` means unlimited when `recursive`.
    pub max_depth: Option<usize>,
    /// Sort key.
    pub sort_by: SortBy,
    /// Reverse the sort order.
    pub sort_descending: bool,
    /// Keep only files whose suffix matches one of these entries exactly (`".txt"`).
    pub file_types: Option<Vec<String>>,
}

impl ListFilesParams {
    /// Non-recursive parameters for `path`.
    #[must_use]
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            ..Self::default()
        }
    }

    /// Returns these parameters in recursive mode.
    #[must_use]
    pub fn recursive(mut self, max_depth: Option<usize>) -> Self {
        self.recursive = true;
        self.max_depth = max_depth;
        self
    }
}

/// Parameters of `list_directories`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "snake_case", default)]
pub struct ListDirsParams {
    /// Directory to list.
    pub path: String,
    /// Recurse into subdirectories.
    pub recursive: bool,
    /// Maximum recursion depth.
    pub max_depth: Option<usize>,
    /// Sort key.
    pub sort_by: SortBy,
    /// Reverse the sort order.
    pub sort_descending: bool,
}

impl ListDirsParams {
    /// Non-recursive parameters for `path`.
    #[must_use]
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            ..Self::default()
        }
    }
}

/// A listed entry (Python `FileSystemItem`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "snake_case", default)]
pub struct FileSystemItem {
    /// Entry name (file name or directory name).
    pub name: String,
    /// Absolute path of the entry.
    pub path: String,
    /// Size in bytes (`0` for directories).
    pub size: u64,
    /// Modification time formatted like Python's `str(datetime.fromtimestamp(mtime))`.
    pub modified_time: String,
    /// `true` for directories.
    pub is_directory: bool,
    /// File suffix including the dot, `None` for directories (Python field `type`).
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub file_type: Option<String>,
}

/// Data of `list_files` / `list_directories` (Python `FileSystemData`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "snake_case", default)]
pub struct ListFilesData {
    /// Number of entries returned.
    pub total_count: usize,
    /// The entries.
    pub list_items: Vec<FileSystemItem>,
    /// Directory that was listed.
    pub root_path: String,
    /// Whether the listing recursed.
    pub recursive: bool,
    /// Depth limit that was applied.
    pub max_depth: Option<usize>,
}

/// How `search_files_with` interprets its pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum SearchPatternMode {
    /// Path-like glob matching (`*.txt`, `src/**/mod.rs`) — the Python behavior.
    #[default]
    Glob,
    /// Regular-expression matching against the file name.
    Regex,
}

/// Parameters of `search_files` / `search_files_with`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "snake_case", default)]
pub struct SearchFilesParams {
    /// Directory to search under.
    pub path: String,
    /// Glob (Python parity) or regex pattern, depending on the call.
    pub pattern: String,
    /// Globs of paths to exclude from the result.
    pub exclude_patterns: Option<Vec<String>>,
}

impl SearchFilesParams {
    /// Parameters searching `path` for `pattern`.
    #[must_use]
    pub fn new(path: impl Into<String>, pattern: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            pattern: pattern.into(),
            exclude_patterns: None,
        }
    }

    /// Returns these parameters with exclusion globs.
    #[must_use]
    pub fn with_excludes<I, S>(mut self, excludes: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.exclude_patterns = Some(excludes.into_iter().map(Into::into).collect());
        self
    }
}

/// Data of `search_files` (Python `SearchFilesData`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "snake_case", default)]
pub struct SearchFilesData {
    /// Number of matching files.
    pub total_matches: usize,
    /// The matching files.
    pub matching_files: Vec<FileSystemItem>,
    /// Directory that was searched.
    pub search_path: String,
    /// Pattern that was applied.
    pub search_pattern: String,
    /// Exclusions that were applied.
    pub exclude_patterns: Option<Vec<String>>,
}

/// Metadata of a path (`stat`), a Rust addition with no Python counterpart.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "snake_case", default)]
pub struct FileStat {
    /// Absolute path.
    pub path: String,
    /// `true` for regular files.
    pub is_file: bool,
    /// `true` for directories.
    pub is_directory: bool,
    /// `true` when the path is a symbolic link.
    pub is_symlink: bool,
    /// Size in bytes.
    pub size: u64,
    /// Modification time formatted like a Python datetime.
    pub modified_time: String,
    /// Octal permission bits on Unix, `None` elsewhere.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub permissions: Option<String>,
}

/// Parameters of `mkdir`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct MkdirParams {
    /// Directory to create.
    pub path: String,
    /// Create missing parents.
    pub recursive: bool,
}

impl Default for MkdirParams {
    fn default() -> Self {
        Self {
            path: String::new(),
            recursive: true,
        }
    }
}

impl MkdirParams {
    /// Recursive parameters for `path`.
    #[must_use]
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            ..Self::default()
        }
    }
}

/// Data of `mkdir`.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct MkdirData {
    /// Directory that was created.
    pub path: String,
    /// `false` when the directory already existed.
    pub created: bool,
}

/// Parameters of `delete`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "snake_case", default)]
pub struct DeleteParams {
    /// Path to delete.
    pub path: String,
    /// Delete directories recursively (required for non-empty directories).
    pub recursive: bool,
}

impl DeleteParams {
    /// Non-recursive parameters for `path`.
    #[must_use]
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            ..Self::default()
        }
    }
}

/// Data of `delete`.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct DeleteData {
    /// Path that was deleted.
    pub path: String,
    /// Number of filesystem entries removed (`1` for a file or empty directory).
    pub deleted: usize,
}

/// Parameters of `copy`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "snake_case", default)]
pub struct CopyParams {
    /// Source path inside the workspace.
    pub source_path: String,
    /// Destination path inside the workspace.
    pub target_path: String,
    /// Overwrite an existing destination.
    pub overwrite: bool,
    /// Copy directories recursively.
    pub recursive: bool,
}

impl CopyParams {
    /// Parameters copying `source_path` to `target_path`.
    #[must_use]
    pub fn new(source_path: impl Into<String>, target_path: impl Into<String>) -> Self {
        Self {
            source_path: source_path.into(),
            target_path: target_path.into(),
            ..Self::default()
        }
    }
}

/// Data of `copy`.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct CopyData {
    /// Source path.
    pub source_path: String,
    /// Destination path.
    pub target_path: String,
    /// Bytes copied.
    pub size: u64,
}

/// Parameters of `move_path`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "snake_case", default)]
pub struct MoveParams {
    /// Source path inside the workspace.
    pub source_path: String,
    /// Destination path inside the workspace.
    pub target_path: String,
    /// Overwrite an existing destination.
    pub overwrite: bool,
}

impl MoveParams {
    /// Parameters moving `source_path` to `target_path`.
    #[must_use]
    pub fn new(source_path: impl Into<String>, target_path: impl Into<String>) -> Self {
        Self {
            source_path: source_path.into(),
            target_path: target_path.into(),
            overwrite: false,
        }
    }
}

/// Data of `move_path`.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct MoveData {
    /// Source path.
    pub source_path: String,
    /// Destination path.
    pub target_path: String,
}

/// Free-form options accepted by the Python backend and preserved for parity.
///
/// Python threads an `options: Optional[Dict[str, Any]]` through every call (lock
/// timeout, encoding overrides). Rust models the known keys as fields on the parameter
/// structs and keeps the remaining ones here.
pub type Extras = BTreeMap<String, serde_json::Value>;

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn params_deserialize_from_partial_json_with_python_defaults() {
        let params: ReadFileParams = serde_json::from_str(r#"{"path": "a.txt"}"#).unwrap();
        assert_eq!(params.mode, ReadMode::Text);
        assert_eq!(params.encoding, "utf-8");
        assert_eq!(params.chunk_size, 0);
        assert!(!params.only_read);

        let params: WriteFileParams =
            serde_json::from_str(r#"{"path": "a.txt", "content": "hi"}"#).unwrap();
        assert!(
            params.prepend_newline,
            "Python prepend_newline defaults to true"
        );
        assert!(!params.append_newline);
        assert!(params.create_if_not_exist);
        assert_eq!(params.permissions.as_deref(), Some("644"));
    }

    #[test]
    fn write_content_accepts_text_and_bytes() {
        let text: WriteFileParams =
            serde_json::from_str(r#"{"path": "a", "content": "hello"}"#).unwrap();
        assert_eq!(text.content, FileContent::Text("hello".to_string()));

        let bytes: WriteFileParams =
            serde_json::from_str(r#"{"path": "a", "content": [1, 2, 3]}"#).unwrap();
        assert_eq!(bytes.content, FileContent::Bytes(vec![1, 2, 3]));
    }

    #[test]
    fn file_content_helpers() {
        let text = FileContent::text("abc");
        assert!(text.is_text());
        assert_eq!(text.byte_len(), 3);
        assert_eq!(text.mode(), ReadMode::Text);
        assert!(text.as_bytes().is_none());

        let bytes = FileContent::bytes(vec![1_u8, 2]);
        assert_eq!(bytes.mode(), ReadMode::Bytes);
        assert_eq!(bytes.as_bytes(), Some(&[1_u8, 2][..]));
    }

    #[test]
    fn file_system_item_uses_python_type_field() {
        let item = FileSystemItem {
            name: "a.txt".to_string(),
            file_type: Some(".txt".to_string()),
            ..FileSystemItem::default()
        };
        let json = serde_json::to_value(&item).unwrap();
        assert_eq!(json["type"], ".txt");
        assert_eq!(json["is_directory"], false);
    }

    #[test]
    fn builder_helpers_set_overrides() {
        let params = ReadFileParams::text("x").with_head(3).with_line_range(1, 2);
        assert_eq!(params.head, Some(3));
        assert_eq!(params.line_range, Some(LineRange::new(1, 2)));
        let list = ListFilesParams::new("d").recursive(Some(2));
        assert!(list.recursive);
        assert_eq!(list.max_depth, Some(2));
    }

    #[test]
    fn defaults_match_python_transfer_flags() {
        let upload = UploadFileParams::default();
        assert!(upload.create_parent_dirs);
        assert!(upload.preserve_permissions);
        assert!(!upload.overwrite);
        let search = SearchFilesParams::new("d", "*.txt").with_excludes(["*.csv"]);
        assert_eq!(
            search.exclude_patterns.as_deref().map(<[String]>::len),
            Some(1)
        );
    }
}
