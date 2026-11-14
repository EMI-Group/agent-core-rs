//! On-disk layout of one coding-memory root: the file names, the derived-state directory,
//! the file change stamp, and the atomic document read/write helpers.
//!
//! Everything here is crate-internal plumbing for `crate::lite::local`; the public
//! constants are re-exported from that module, so their paths are unchanged.
use std::path::Path;
use std::time::{Duration, SystemTime};

use openjiuwen_common::error::{Error, Result};

use crate::lite::markdown::MarkdownMemory;
use crate::record::MemoryScope;

/// File name of the Markdown source of truth.
pub const MEMORY_FILE: &str = "MEMORY.md";

/// Directory (relative to the root) holding the derived index and the lock file.
pub const STATE_DIR: &str = ".memory";

/// File name of the derived full-text index (feature `sqlite`).
pub const INDEX_FILE: &str = "index.db";

/// File name of the cross-process advisory lock.
pub const LOCK_FILE: &str = "index.lock";

/// Debounce window used by
/// [`start_watching`](crate::lite::local::LocalCodingMemory::start_watching).
///
/// Matches the Python `sync.watchDebounceMs` intent at a smaller, test-friendly value:
/// filesystem bursts are coalesced into one reload.
pub const DEFAULT_DEBOUNCE: Duration = Duration::from_millis(200);

/// One file's change stamp: modification time plus length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct FileStamp {
    /// Last modification time, when the platform reports one.
    modified: Option<SystemTime>,
    /// File length in bytes.
    len: u64,
}

impl FileStamp {
    /// Reads the stamp of `path`, or `None` when the file does not exist.
    pub(super) async fn of(path: &Path) -> Result<Option<Self>> {
        match tokio::fs::metadata(path).await {
            Ok(metadata) => Ok(Some(Self {
                modified: metadata.modified().ok(),
                len: metadata.len(),
            })),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => {
                Err(Error::external(error).with_context(format!("stamping {}", path.display())))
            }
        }
    }
}

/// Loads (or initializes) the document at `path` for `scope`.
pub(super) async fn load_document(
    path: &Path,
    scope: &MemoryScope,
) -> Result<(MarkdownMemory, Option<FileStamp>)> {
    let text = match tokio::fs::read_to_string(path).await {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok((MarkdownMemory::new().with_scope(scope.clone()), None));
        }
        Err(error) => {
            return Err(Error::external(error)
                .with_context(format!("reading the memory document {}", path.display())));
        }
    };
    let stamp = FileStamp::of(path).await?;
    if text.trim().is_empty() {
        return Ok((MarkdownMemory::new().with_scope(scope.clone()), stamp));
    }
    let document = MarkdownMemory::parse(&text)?.with_scope(scope.clone());
    Ok((document, stamp))
}

/// Writes `document` to `path` atomically (temporary file plus rename).
pub(super) async fn write_document(path: &Path, document: &MarkdownMemory) -> Result<()> {
    let rendered = document.render();
    let temporary = path.with_extension("md.tmp");
    tokio::fs::write(&temporary, rendered.as_bytes())
        .await
        .map_err(|error| {
            Error::external(error).with_context(format!(
                "writing the temporary memory document {}",
                temporary.display()
            ))
        })?;
    tokio::fs::rename(&temporary, path).await.map_err(|error| {
        Error::external(error)
            .with_context(format!("replacing the memory document {}", path.display()))
    })
}
