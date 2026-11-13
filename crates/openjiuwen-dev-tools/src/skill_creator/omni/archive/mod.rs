//! Archive sniffing, packing, and unpacking for the [`super::pipeline`] scrape stage.
//!
//! The `skill_omni_creation` pipeline accepts a URL that may resolve to an
//! archive rather than an HTML page, so [`detect`] identifies the container from
//! its magic bytes and [`ArchiveUnpacker`] turns it into ordered
//! [`ArchiveEntry`] values. [`ArchiveWriter`] produces deterministic stored
//! `zip` bytes for tests and for callers that need to round-trip entries.
//!
//! Every format is parsed in-process with no compression dependency:
//!
//! - `tar` (POSIX `ustar` and GNU flavors) in `tar`;
//! - `zip` (stored and DEFLATE, plus a stored writer) in `zip`;
//! - `gzip` (RFC 1952, including gzip-wrapped `tar`/`zip`) in `gzip`;
//! - DEFLATE (RFC 1951) and CRC-32 (IEEE) live in the shared `inflate` and
//!   `crc` modules.
//!
//! Every entry path is validated before it is returned: absolute paths, `..`
//! components, and backslashes are rejected, so a caller can never write outside
//! the bundle root. Unpacking is bounded by [`MAX_ENTRY_UNCOMPRESSED_BYTES`] per
//! entry and [`MAX_ARCHIVE_UNCOMPRESSED_BYTES`] per archive, so a hostile archive
//! fails with a validation error instead of exhausting memory.
mod crc;
mod gzip;
mod inflate;
mod tar;
mod zip;

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Largest number of bytes a single unpacked entry may declare: 8 MiB.
///
/// This matches the default [`HttpFetcher`](super::fetch::HttpFetcher) body cap,
/// so an archive entry can never be larger than the response that carried it.
pub const MAX_ENTRY_UNCOMPRESSED_BYTES: usize = 8 * 1024 * 1024;

/// Largest number of bytes every entry of one archive may declare in total: 32 MiB.
pub const MAX_ARCHIVE_UNCOMPRESSED_BYTES: usize = 32 * 1024 * 1024;

/// A container format recognized by [`detect`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArchiveFormat {
    /// A `tar` archive, including the POSIX `ustar` flavor.
    Tar,
    /// A `zip` archive.
    Zip,
    /// A `gzip`-compressed stream.
    Gzip,
}

impl ArchiveFormat {
    /// The lowercase wire name of the format: `"tar"`, `"zip"`, or `"gzip"`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Tar => "tar",
            Self::Zip => "zip",
            Self::Gzip => "gzip",
        }
    }
}

impl fmt::Display for ArchiveFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One entry read out of an archive.
///
/// `path` is the archive-relative path with `/` separators; `bytes` holds the
/// file contents and is empty for directories; `is_dir` marks directory entries.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ArchiveEntry {
    /// Archive-relative path of the entry.
    pub path: String,
    /// File contents; empty for directory entries.
    pub bytes: Vec<u8>,
    /// Whether the entry is a directory.
    pub is_dir: bool,
}

/// Associated functions for sniffing and unpacking archives.
///
/// This type is never instantiated: it only groups the entry points so call
/// sites read as [`ArchiveUnpacker::unpack`] / [`ArchiveUnpacker::unpack_auto`],
/// matching the associated-function style of `SkillBundle`.
#[derive(Debug, Clone, Copy)]
pub struct ArchiveUnpacker;

impl ArchiveUnpacker {
    /// Unpacks `bytes` as `format`, returning entries in archive order.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] for malformed, truncated, unsafe, or
    /// oversized content, and [`Error::unsupported`] for capabilities this
    /// in-process implementation deliberately does not provide (ZIP64, encrypted
    /// entries, and compression methods other than stored and DEFLATE).
    pub fn unpack(format: ArchiveFormat, bytes: &[u8]) -> Result<Vec<ArchiveEntry>> {
        match format {
            ArchiveFormat::Tar => tar::unpack(bytes),
            ArchiveFormat::Zip => zip::unpack(bytes),
            ArchiveFormat::Gzip => gzip::unpack(bytes),
        }
    }

    /// Sniffs the format with [`detect`] and then [`unpack`](Self::unpack)s it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] when the bytes carry no recognized archive
    /// signature, plus everything [`unpack`](Self::unpack) can return.
    pub fn unpack_auto(bytes: &[u8]) -> Result<Vec<ArchiveEntry>> {
        let format = detect(bytes).ok_or_else(|| {
            Error::validation(format!(
                "unrecognized archive: no gzip, zip, or ustar signature in {} bytes",
                bytes.len()
            ))
        })?;
        Self::unpack(format, bytes)
    }
}

/// Sniffs the archive format of `bytes`, or `None` when nothing matches.
///
/// Recognized signatures are `gzip` (`1f 8b`), `zip` (`PK\x03\x04` and the empty
/// archive `PK\x05\x06`), and `tar` (`ustar` at byte offset 257).
#[must_use]
pub fn detect(bytes: &[u8]) -> Option<ArchiveFormat> {
    if bytes.starts_with(&[0x1f, 0x8b]) {
        Some(ArchiveFormat::Gzip)
    } else if bytes.starts_with(b"PK\x03\x04") || bytes.starts_with(b"PK\x05\x06") {
        Some(ArchiveFormat::Zip)
    } else if bytes.get(257..262) == Some(b"ustar".as_slice()) {
        Some(ArchiveFormat::Tar)
    } else {
        None
    }
}

pub use zip::ArchiveWriter;

/// Rejects paths a caller could use to escape the bundle root.
///
/// The rules are shared by every format: a path must be non-empty, relative
/// (no leading `/` and no Windows drive prefix), free of `..` components and
/// backslashes, and `/`-separated.
pub(crate) fn validate_path(path: &str) -> Result<String> {
    if path.is_empty() {
        return Err(Error::validation("unsafe archive path: empty entry name"));
    }
    if path.starts_with('/') || is_windows_absolute(path) {
        return Err(Error::validation(format!(
            "unsafe archive path {path:?}: absolute paths are not allowed"
        )));
    }
    if path.split('/').any(|component| component == "..") {
        return Err(Error::validation(format!(
            "unsafe archive path {path:?}: `..` components are not allowed"
        )));
    }
    if path.contains('\\') {
        return Err(Error::validation(format!(
            "unsafe archive path {path:?}: backslashes are not allowed"
        )));
    }
    Ok(path.to_string())
}

/// Whether `path` starts with a Windows drive prefix such as `C:`.
fn is_windows_absolute(path: &str) -> bool {
    let bytes = path.as_bytes();
    bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':'
}

/// Bounds-checked slice of `bytes`, naming `offset` in the error.
pub(crate) fn slice_at(bytes: &[u8], offset: usize, len: usize) -> Result<&[u8]> {
    let end = offset
        .checked_add(len)
        .ok_or_else(|| Error::validation(format!("byte offset overflow at {offset} + {len}")))?;
    bytes.get(offset..end).ok_or_else(|| {
        Error::validation(format!(
            "truncated input: {len} bytes needed at offset {offset}, but the input is {} bytes",
            bytes.len()
        ))
    })
}

/// Reads a little-endian `u16` at `offset`.
pub(crate) fn read_u16_le(bytes: &[u8], offset: usize) -> Result<u16> {
    let raw = slice_at(bytes, offset, 2)?;
    Ok(u16::from_le_bytes([raw[0], raw[1]]))
}

/// Reads a little-endian `u32` at `offset`.
pub(crate) fn read_u32_le(bytes: &[u8], offset: usize) -> Result<u32> {
    let raw = slice_at(bytes, offset, 4)?;
    Ok(u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]))
}

/// Enforces the per-entry and aggregate uncompressed-size caps.
///
/// `tar` and `zip` declare entry sizes in their headers; charging a declared
/// size before allocating keeps a hostile archive from exhausting memory.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct CapTracker {
    total: usize,
}

impl CapTracker {
    /// A tracker that has charged nothing yet.
    pub(crate) const fn new() -> Self {
        Self { total: 0 }
    }

    /// Charges `len` bytes to the entry and aggregate caps.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] naming `label` when `len` exceeds
    /// [`MAX_ENTRY_UNCOMPRESSED_BYTES`] or the running total would exceed
    /// [`MAX_ARCHIVE_UNCOMPRESSED_BYTES`].
    pub(crate) fn charge(&mut self, len: usize, label: &str) -> Result<()> {
        if len > MAX_ENTRY_UNCOMPRESSED_BYTES {
            return Err(Error::validation(format!(
                "{label} declares {len} bytes, above the per-entry cap of \
                 {MAX_ENTRY_UNCOMPRESSED_BYTES} bytes"
            )));
        }
        let total = self
            .total
            .checked_add(len)
            .ok_or_else(|| Error::validation(format!("{label}: the archive size overflows")))?;
        if total > MAX_ARCHIVE_UNCOMPRESSED_BYTES {
            return Err(Error::validation(format!(
                "{label}: the archive expands past the aggregate cap of \
                 {MAX_ARCHIVE_UNCOMPRESSED_BYTES} bytes"
            )));
        }
        self.total = total;
        Ok(())
    }

    /// Bytes charged so far, for tests that assert the running total.
    #[cfg(test)]
    pub(crate) const fn total(self) -> usize {
        self.total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_recognizes_known_signatures() {
        assert_eq!(detect(&[0x1f, 0x8b, 0x08, 0x00]), Some(ArchiveFormat::Gzip));
        assert_eq!(detect(b"PK\x03\x04rest"), Some(ArchiveFormat::Zip));
        assert_eq!(detect(b"PK\x05\x06"), Some(ArchiveFormat::Zip));
        assert_eq!(detect(b"plain text, not an archive"), None);
        assert_eq!(ArchiveFormat::Gzip.to_string(), "gzip");
        assert_eq!(ArchiveFormat::Tar.as_str(), "tar");
    }

    #[test]
    fn validate_path_accepts_relative_paths_and_rejects_escapes() {
        assert_eq!(validate_path("a/b.txt").expect("relative path"), "a/b.txt");
        for path in ["", "/etc/passwd", "C:evil", "../evil", "a/../b", "a\\b"] {
            let error = validate_path(path).expect_err(path);
            assert_eq!(
                error.kind(),
                openjiuwen_common::error::ErrorKind::Validation
            );
        }
    }

    #[test]
    fn little_endian_readers_are_bounds_checked() {
        let bytes = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09];
        assert_eq!(read_u16_le(&bytes, 0).expect("u16"), 0x0201);
        assert_eq!(read_u32_le(&bytes, 0).expect("u32"), 0x0403_0201);
        let error = read_u32_le(&bytes, 7).expect_err("truncated");
        assert!(error.to_string().contains("offset 7"), "{error}");
        let error = slice_at(&bytes, usize::MAX, 2).expect_err("overflow");
        assert!(error.to_string().contains("overflow"), "{error}");
    }

    #[test]
    fn cap_tracker_enforces_per_entry_and_aggregate_bounds() {
        let mut caps = CapTracker::new();
        // Four full entries exactly exhaust the 32 MiB aggregate cap.
        for _ in 0..4 {
            caps.charge(MAX_ENTRY_UNCOMPRESSED_BYTES, "entry")
                .expect("a full entry fits");
        }
        assert_eq!(caps.total(), MAX_ARCHIVE_UNCOMPRESSED_BYTES);
        let error = caps
            .charge(MAX_ENTRY_UNCOMPRESSED_BYTES, "entry")
            .expect_err("aggregate cap");
        assert!(error.to_string().contains("aggregate"), "{error}");
        assert_eq!(caps.total(), MAX_ARCHIVE_UNCOMPRESSED_BYTES);

        let error = CapTracker::new()
            .charge(MAX_ENTRY_UNCOMPRESSED_BYTES + 1, "entry")
            .expect_err("per-entry cap");
        assert!(error.to_string().contains("per-entry"), "{error}");
    }
}
