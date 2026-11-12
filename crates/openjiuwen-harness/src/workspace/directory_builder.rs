//! Materializes a directory schema on disk through a sysop backend.
//!
//! Python's `DirectoryBuilder` seeds every node of a workspace schema: files are created
//! with their packaged content, directories get an empty [`.workspace`](WORKSPACE_MARKER)
//! marker, and an existing entry is never overwritten. The marker is what makes a
//! directory recognizable as a workspace node rather than an incidental folder.
//!
//! All I/O goes through [`SysOperation`] so the same builder works against the local
//! backend and a sandbox backend; existence is probed through that backend too, never
//! through the process filesystem.

use std::path::PathBuf;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_sysop::fs::types::{FileContent, ReadFileParams, ReadMode, WriteFileParams};
use openjiuwen_sysop::workspace::clean_path;
use openjiuwen_sysop::{FsOperation, SysOperation};

use crate::workspace::node::DirectoryNode;

/// Marker file that identifies a directory as a workspace node.
pub const WORKSPACE_MARKER: &str = ".workspace";

/// Creates the schema nodes of a workspace inside `root_path`.
#[derive(Debug, Clone)]
pub struct DirectoryBuilder {
    /// Backend used for every read and write.
    pub sys_operation: SysOperation,
    /// Root directory the schema is materialized in.
    pub root_path: PathBuf,
}

impl DirectoryBuilder {
    /// Creates a builder that materializes schemas under `root_path`.
    #[must_use]
    pub fn new(sys_operation: SysOperation, root_path: impl Into<PathBuf>) -> Self {
        Self {
            sys_operation,
            root_path: root_path.into(),
        }
    }

    /// Materializes `directories`, depth-first and in schema order.
    ///
    /// Existing files and markers are left untouched, so rebuilding a workspace is
    /// idempotent and never clobbers agent-authored content.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::Unsupported`](openjiuwen_common::error::ErrorKind::Unsupported)
    /// when the sys-operation has no filesystem backend,
    /// [`ErrorKind::Validation`](openjiuwen_common::error::ErrorKind::Validation) for an
    /// unsafe node path (before any I/O for that node) or a non-UTF-8 root, and the
    /// backend error when an operation fails.
    pub async fn build(&self, directories: &[DirectoryNode]) -> Result<()> {
        let fs = self.sys_operation.fs().ok_or_else(|| {
            Error::unsupported("workspace directory builder requires a filesystem backend")
        })?;
        // Explicit stack instead of async recursion: async fn cannot recurse without boxing.
        let mut stack: Vec<(&DirectoryNode, PathBuf)> = directories
            .iter()
            .rev()
            .map(|node| (node, self.root_path.clone()))
            .collect();
        while let Some((node, base)) = stack.pop() {
            let relative = node.resolved_path();
            if !is_safe_relative_path(relative) {
                return Err(Error::validation(format!(
                    "Unsafe path detected: {relative}"
                )));
            }
            let full = clean_path(&base.join(relative));
            let full = path_to_string(&full)?;
            if node.is_file {
                if !file_exists(fs.as_ref(), &full).await {
                    fs.write_file(WriteFileParams::raw(
                        full.as_str(),
                        FileContent::text(node.default_content.clone()),
                    ))
                    .await
                    .into_result()?;
                }
            } else {
                let marker = path_to_string(&PathBuf::from(&full).join(WORKSPACE_MARKER))?;
                if !file_exists(fs.as_ref(), &marker).await {
                    fs.write_file(WriteFileParams::raw(marker, FileContent::text("")))
                        .await
                        .into_result()?;
                }
            }
            for child in node.children.iter().rev() {
                stack.push((child, PathBuf::from(&full)));
            }
        }
        Ok(())
    }
}

/// Converts a path to a lossless UTF-8 string.
fn path_to_string(path: &std::path::Path) -> Result<String> {
    path.to_str().map(ToString::to_string).ok_or_else(|| {
        Error::validation(format!(
            "workspace path is not valid UTF-8: {}",
            path.display()
        ))
    })
}

/// Probes a path through the backend: a successful one-byte read means it exists.
async fn file_exists(fs: &dyn FsOperation, path: &str) -> bool {
    let params = ReadFileParams {
        path: path.to_string(),
        mode: ReadMode::Bytes,
        chunk_size: 1,
        only_read: true,
        ..ReadFileParams::default()
    };
    fs.read_file(params).await.is_success()
}

/// Returns `true` when `path` is a relative path that stays inside its base directory.
///
/// Rejects empty paths, NUL bytes, absolute paths (`/`, `\`, and UNC prefixes), and
/// Windows drive prefixes (`C:` / `C:\`). `\` is normalized to `/`, `.` segments are
/// ignored, and `..` segments pop a segment — a path that would escape is rejected. At
/// least one segment must survive, so `""`, `"."`, and `"x/.."` are unsafe while `"a/b"`,
/// `"./a/b"`, `"a\\b"`, and `"a/../b"` are safe.
#[must_use]
pub fn is_safe_relative_path(path: &str) -> bool {
    if path.is_empty() || path.contains('\0') {
        return false;
    }
    let bytes = path.as_bytes();
    if matches!(bytes.first(), Some(b'/' | b'\\')) {
        return false;
    }
    if bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':' {
        return false;
    }
    let normalized = path.replace('\\', "/");
    let mut depth = 0_usize;
    for segment in normalized.split('/') {
        match segment {
            "" | "." => {}
            ".." => {
                if depth == 0 {
                    return false;
                }
                depth -= 1;
            }
            _ => depth += 1,
        }
    }
    depth > 0
}
