//! The operation traits implemented by every backend.
//!
//! Three traits mirror the Python ABCs `BaseFsOperation`, `BaseShellOperation`, and
//! `BaseCodeOperation` (`core/sys_operation/fs.py`, `shell.py`, `code.py`): the local
//! backend and the sandbox backend implement the same surface, so callers are agnostic
//! of where an operation runs.
//!
//! Streaming methods take `self: Arc<Self>` so the returned [`OpStream`] is `'static` and
//! can outlive the borrow that created it.
//!
//! ```
//! use openjiuwen_sysop::fs::types::ReadFileParams;
//! use openjiuwen_sysop::result::OpResult;
//!
//! // Backends return `OpResult<T>`: a failure is a value, not an exception.
//! let failure = OpResult::<ReadFileParams>::failure(
//!     openjiuwen_sysop::status::SysOpStatus::FsExecutionError,
//!     "boom",
//! );
//! assert_eq!(failure.code, 199_003);
//! ```

use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::BoxStream;
use openjiuwen_common::error::{Error, Result};

use crate::code::types::{CodeChunk, CodeExecParams, CodeOutput};
use crate::fs::types::{
    CopyData, CopyParams, DeleteData, DeleteParams, DownloadFileData, DownloadFileParams, FileStat,
    ListDirsParams, ListFilesData, ListFilesParams, MkdirData, MkdirParams, MoveData, MoveParams,
    ReadBytesData, ReadBytesParams, ReadChunkData, ReadFileData, ReadFileParams, SearchFilesData,
    SearchFilesParams, SearchPatternMode, TransferChunkData, UploadFileData, UploadFileParams,
    WriteFileData, WriteFileParams,
};
use crate::result::OpResult;
use crate::shell::types::{
    ShellArgvParams, ShellBackgroundOutput, ShellChunk, ShellExecParams, ShellOutput,
};

/// A stream of operation results.
///
/// Failures appear as a single terminal [`OpResult`] with a non-zero code, matching the
/// Python behavior of yielding one error item from a streamed operation.
pub type OpStream<T> = BoxStream<'static, OpResult<T>>;

/// Filesystem operations shared by the local and sandbox backends.
///
/// The ten required methods are exactly Python's `BaseFsOperation` surface, plus one
/// defaulted Rust addition — [`FsOperation::read_bytes_limited`] — that lets a caller read a
/// window of a file without allocating the whole of it. The extra convenience operations
/// (`mkdir`, `delete`, `copy`, `move_path`, `stat`, `exists`, regex search) live in
/// [`FsExtOperation`] and are not rendered as tool cards, so the generated tool list stays
/// Python-compatible.
#[async_trait]
pub trait FsOperation: Send + Sync {
    /// Whether this backend implements compare-and-swap writes.
    ///
    /// Python parity: `BaseFsOperation.supports_conditional_write`; the local backend
    /// returns `true`, the sandbox backend mirrors its gateway configuration.
    fn supports_conditional_write(&self) -> bool {
        false
    }

    /// Reads a file, honoring head/tail/line-range slicing in text mode.
    ///
    /// The whole file is loaded, so a file above
    /// [`LocalWorkConfig::max_file_size_bytes`](crate::config::LocalWorkConfig::max_file_size_bytes)
    /// is refused instead of allocated: the failure reason comes from
    /// [`PathPolicy::check_read_size`](crate::workspace::PathPolicy::check_read_size) and names
    /// the file size and the limit. Use [`Self::read_bytes_limited`] for a window of a larger
    /// file.
    async fn read_file(&self, params: ReadFileParams) -> OpResult<ReadFileData>;

    /// Reads at most `params.max_bytes` bytes from `params.offset` **without** loading the
    /// rest of the file.
    ///
    /// Rust addition (Python's `read_file` is whole-file only): the returned buffer is bounded
    /// by the caller's window, `total_bytes` reports the real file length, and `truncated`
    /// says whether more bytes remained. Reading the first lines of a multi-gigabyte log costs
    /// the window, not the file.
    ///
    /// The default implementation reports
    /// [`openjiuwen_common::error::ErrorKind::Unsupported`], so a backend that implements only
    /// Python's ten-method surface keeps compiling and says it cannot bound a read instead of
    /// silently degrading to a whole-file read.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Unsupported`] for a backend that does not
    /// implement it, [`openjiuwen_common::error::ErrorKind::Validation`] when `max_bytes` is
    /// above the configured whole-file ceiling (the message names the requested window and the
    /// limit), and [`openjiuwen_common::error::ErrorKind::SysOp`] for path/lock/I/O failures.
    async fn read_bytes_limited(&self, params: ReadBytesParams) -> Result<ReadBytesData> {
        let _ = params;
        Err(Error::unsupported(
            "this filesystem backend does not implement bounded byte reads",
        ))
    }

    /// Streams a file: one line per chunk in text mode, `chunk_size` bytes per chunk in
    /// bytes mode.
    fn read_file_stream(self: Arc<Self>, params: ReadFileParams) -> OpStream<ReadChunkData>;

    /// Writes (or appends to) a file, with optional compare-and-swap.
    async fn write_file(&self, params: WriteFileParams) -> OpResult<WriteFileData>;

    /// Copies a host file into the workspace.
    async fn upload_file(&self, params: UploadFileParams) -> OpResult<UploadFileData>;

    /// Streams a host-to-workspace copy.
    fn upload_file_stream(self: Arc<Self>, params: UploadFileParams)
        -> OpStream<TransferChunkData>;

    /// Copies a workspace file to the host.
    async fn download_file(&self, params: DownloadFileParams) -> OpResult<DownloadFileData>;

    /// Streams a workspace-to-host copy.
    fn download_file_stream(
        self: Arc<Self>,
        params: DownloadFileParams,
    ) -> OpStream<TransferChunkData>;

    /// Lists files under a directory.
    async fn list_files(&self, params: ListFilesParams) -> OpResult<ListFilesData>;

    /// Lists directories under a directory.
    async fn list_directories(&self, params: ListDirsParams) -> OpResult<ListFilesData>;

    /// Searches for files by glob pattern (Python `pathlib.rglob` semantics).
    async fn search_files(&self, params: SearchFilesParams) -> OpResult<SearchFilesData>;
}

/// Additional filesystem operations that are Rust additions (no Python counterpart).
#[async_trait]
pub trait FsExtOperation: Send + Sync {
    /// Returns metadata for a path.
    async fn stat(&self, path: &str) -> OpResult<FileStat>;

    /// Returns `true` when the path exists.
    async fn exists(&self, path: &str) -> OpResult<bool>;

    /// Creates a directory.
    async fn mkdir(&self, params: MkdirParams) -> OpResult<MkdirData>;

    /// Deletes a file or directory.
    async fn delete(&self, params: DeleteParams) -> OpResult<DeleteData>;

    /// Copies a file or directory inside the workspace.
    async fn copy(&self, params: CopyParams) -> OpResult<CopyData>;

    /// Moves or renames a path inside the workspace.
    async fn move_path(&self, params: MoveParams) -> OpResult<MoveData>;

    /// Searches for files using glob or regular-expression matching.
    async fn search_files_with(
        &self,
        params: SearchFilesParams,
        mode: SearchPatternMode,
    ) -> OpResult<SearchFilesData>;
}

/// Shell command execution.
#[async_trait]
pub trait ShellOperation: Send + Sync {
    /// Runs a command line through a shell, capturing bounded output.
    async fn execute_cmd(&self, params: ShellExecParams) -> OpResult<ShellOutput>;

    /// Runs a command line, yielding output chunks as they arrive.
    fn execute_cmd_stream(self: Arc<Self>, params: ShellExecParams) -> OpStream<ShellChunk>;

    /// Starts a command and returns its pid without waiting for completion.
    async fn execute_cmd_background(
        &self,
        params: ShellExecParams,
    ) -> OpResult<ShellBackgroundOutput>;

    /// Runs a program and argument vector without a shell.
    ///
    /// Preferred over [`ShellOperation::execute_cmd`] whenever the caller controls the
    /// argument boundaries: no quoting, globbing, or redirection is interpreted, so a
    /// value can never be split or injected into the command line.
    async fn execute_argv(&self, params: ShellArgvParams) -> OpResult<ShellOutput>;
}

/// Snippet/script execution in a subprocess.
#[async_trait]
pub trait CodeOperation: Send + Sync {
    /// Runs a snippet and returns its captured output.
    async fn execute_code(&self, params: CodeExecParams) -> OpResult<CodeOutput>;

    /// Runs a snippet, yielding output chunks as they arrive.
    fn execute_code_stream(self: Arc<Self>, params: CodeExecParams) -> OpStream<CodeChunk>;
}

/// The three backends produced for one [`crate::SysOperation`].
#[derive(Clone)]
pub struct OperationSet {
    /// Filesystem backend, when the mode provides one.
    pub fs: Option<Arc<dyn FsOperation>>,
    /// Shell backend, when the mode provides one.
    pub shell: Option<Arc<dyn ShellOperation>>,
    /// Code backend, when the mode provides one.
    pub code: Option<Arc<dyn CodeOperation>>,
}

impl std::fmt::Debug for OperationSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OperationSet")
            .field("fs", &self.fs.is_some())
            .field("shell", &self.shell.is_some())
            .field("code", &self.code.is_some())
            .finish()
    }
}

impl OperationSet {
    /// An empty set, used by tests and by modes that expose no backend.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            fs: None,
            shell: None,
            code: None,
        }
    }
}
