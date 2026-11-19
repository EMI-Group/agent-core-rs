//! Defaults shared by the sys-operation backends.
//!
//! The values mirror the Python runtime defaults in
//! `openjiuwen/core/sys_operation/config.py` and `core/sys_operation/fs.py` so that a
//! configuration written for the Python SDK keeps its meaning.

/// First-token shell allowlist used when [`crate::config::LocalWorkConfig`] does not
/// override it.
///
/// Python parity: the default `LocalWorkConfig.shell_allowlist` (`config.py`).
/// Setting the allowlist to `None` or to an empty list allows every command.
pub const DEFAULT_SHELL_ALLOWLIST: &[&str] = &[
    "echo", "rg", "ls", "dir", "cd", "pwd", "python", "python3", "pip", "pip3", "npm", "node",
    "git", "cat", "type", "mkdir", "md", "rm", "rd", "cp", "copy", "mv", "move", "grep", "find",
    "curl", "wget", "ps", "df", "ping",
];

/// Default shell/code execution timeout in seconds (Python `timeout=300`).
pub const DEFAULT_TIMEOUT_SECS: u64 = 300;

/// Default wait for a cross-process file lock before failing with 199003.
///
/// Python's SQLite lock manager blocks until the lock is free; a bounded default keeps a
/// dead peer from wedging a workspace forever. `LocalWorkConfig::lock_timeout_secs = 0`
/// restores the unbounded wait.
pub const DEFAULT_LOCK_TIMEOUT_SECS: u64 = 30;

/// Bytes of captured stdout/stderr retained per stream before truncation.
///
/// Python captures shell and code output without a limit; this crate bounds the buffer at
/// 4 MiB by default and appends [`OUTPUT_TRUNCATION_MARKER`] when the limit is hit.
/// Raise `LocalWorkConfig::max_output_bytes` for parity with Python's unbounded capture.
pub const DEFAULT_MAX_OUTPUT_BYTES: u64 = 4 * 1024 * 1024;

/// Marker appended to a captured stream that hit `LocalWorkConfig::max_output_bytes`.
pub const OUTPUT_TRUNCATION_MARKER: &str = "\n[output truncated]";

/// Default ceiling for a *whole-file* read, in bytes (64 MiB).
///
/// `read_file` and `read_file_stream` load the entire file into memory, so they refuse a file
/// above [`LocalWorkConfig::max_file_size_bytes`](crate::config::LocalWorkConfig::max_file_size_bytes)
/// instead of allocating it. Python's `read_file` has no such ceiling; this is a Rust addition
/// that turns an accidental multi-gigabyte read into a typed failure.
///
/// A caller that needs a window of a larger file uses the bounded read
/// ([`crate::operation::FsOperation::read_bytes_limited`]), which never allocates more than the
/// requested window, or sets `max_file_size_bytes` to `None`
/// ([`LocalWorkConfig::unrestricted`](crate::config::LocalWorkConfig::unrestricted) does that)
/// to restore Python's unlimited behavior.
pub const DEFAULT_MAX_FILE_SIZE_BYTES: u64 = 64 * 1024 * 1024;

/// Default chunk size in bytes for `read_file_stream` (Python `DEFAULT_READ_STREAM_CHUNK_SIZE`).
pub const DEFAULT_READ_STREAM_CHUNK_SIZE: usize = 8192;

/// Default chunk size in bytes for `upload_file_stream`/`download_file_stream`.
pub const DEFAULT_TRANSFER_CHUNK_SIZE: u64 = 1024 * 1024;

/// Chunk size used while scanning the tail of a text file, in bytes.
pub const DEFAULT_TAIL_SCAN_CHUNK_SIZE: usize = 1024;

/// Text chunk size for streamed shell/code output (Python `chunk_size=1024`).
pub const DEFAULT_TEXT_CHUNK_SIZE: usize = 1024;

/// Encoding assumed when a caller does not pass one.
pub const DEFAULT_ENCODING: &str = "utf-8";

/// The value that selects encoding detection in `LocalWorkConfig`-driven reads.
pub const AUTO_ENCODING: &str = "auto";

/// Default octal permission string applied by `write_file` (Python `permissions="644"`).
pub const DEFAULT_PERMISSIONS: &str = "644";

/// Bytes of stderr kept from a backgrounded command (Python `DEFAULT_STDERR_RETENTION`).
pub const DEFAULT_STDERR_RETENTION_BYTES: u64 = 8192;

/// Default launcher type of a sandbox gateway configuration.
pub const DEFAULT_LAUNCHER_TYPE: &str = "pre_deploy";

/// Default sandbox type of a sandbox gateway configuration.
pub const DEFAULT_SANDBOX_TYPE: &str = "aio";

/// Largest command string passed to a process launch through the argument vector on Unix.
///
/// Longer snippets are written to a temporary file (Python `_UNIX_CMD_LIMIT`).
pub const UNIX_COMMAND_LINE_LIMIT: usize = 100_000;

/// Largest command string passed through the argument vector on Windows.
pub const WINDOWS_COMMAND_LINE_LIMIT: usize = 8_000;

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn default_allowlist_matches_python() {
        assert_eq!(DEFAULT_SHELL_ALLOWLIST.len(), 30);
        assert!(DEFAULT_SHELL_ALLOWLIST.contains(&"git"));
        assert_eq!(DEFAULT_SHELL_ALLOWLIST[0], "echo");
        assert_eq!(DEFAULT_SHELL_ALLOWLIST[29], "ping");
    }

    #[test]
    fn defaults_match_python() {
        assert_eq!(DEFAULT_TIMEOUT_SECS, 300);
        assert_eq!(DEFAULT_READ_STREAM_CHUNK_SIZE, 8192);
        assert_eq!(DEFAULT_TRANSFER_CHUNK_SIZE, 1_048_576);
        assert_eq!(DEFAULT_PERMISSIONS, "644");
        assert_eq!(DEFAULT_STDERR_RETENTION_BYTES, 8192);
        assert_eq!(DEFAULT_LAUNCHER_TYPE, "pre_deploy");
        assert_eq!(DEFAULT_SANDBOX_TYPE, "aio");
        assert_eq!(DEFAULT_MAX_FILE_SIZE_BYTES, 67_108_864);
    }
}
