//! `zip` reading (stored and DEFLATE) and deterministic stored writing.

use super::ArchiveEntry;
use openjiuwen_common::error::{Error, Result};

/// Unpacks a `zip` archive into entries, in central-directory order.
///
/// # Errors
///
/// Returns [`Error::validation`] for malformed or truncated archives and
/// [`Error::unsupported`] for ZIP64, encrypted entries, and compression methods
/// other than stored and DEFLATE.
pub(crate) fn unpack(_bytes: &[u8]) -> Result<Vec<ArchiveEntry>> {
    Err(Error::unsupported(
        "zip archive unpacking requires a compression dependency that the offline workspace does \
         not provide",
    ))
}

/// Writes entries as a deterministic, stored-only `zip` archive.
///
/// This type is never instantiated: it only groups the entry points so call
/// sites read as [`ArchiveWriter::write_stored`].
#[derive(Debug, Clone, Copy)]
pub struct ArchiveWriter;

impl ArchiveWriter {
    /// Serializes `entries` as a stored `zip` archive.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] for an empty entry list, a directory entry
    /// with content, a file entry whose path ends in `/`, or an unsafe path.
    pub fn write_stored(_entries: &[ArchiveEntry]) -> Result<Vec<u8>> {
        Err(Error::unsupported(
            "zip archive writing is not wired into this build",
        ))
    }
}
