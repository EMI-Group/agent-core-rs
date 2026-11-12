//! Minimal ZIP container reading (`PKZIP` / APPNOTE 6.3).
//!
//! DOCX and XLSX documents are ZIP archives of XML parts, so this module turns
//! an archive into a list of [`ZipEntry`] values and extracts their bytes.
//! Only what a document parser needs is implemented:
//!
//! - end-of-central-directory (EOCD) discovery, scanning backwards so a trailing
//!   archive comment is tolerated;
//! - central-directory entries (name, flags, method, sizes, `CRC-32`, offsets);
//! - methods 0 (stored) and 8 (`deflate`), verified against the recorded `CRC-32`
//!   and uncompressed size, with [`DEFAULT_MAX_OUTPUT_BYTES`] as a bomb guard.
//!
//! ZIP64 archives, multi-disk archives, encryption, and every other compression
//! method are rejected with [`Error::Unsupported`], naming what was found.
//!
//! # Paths
//!
//! ZIP is a byte-oriented format, not a filesystem: entry names are stored with
//! `/` separators, carry no drive letters, and are matched **exactly**
//! (case-sensitively) by [`ZipArchive::entry`], [`ZipArchive::has_entry`], and
//! [`ZipArchive::read_entry`]. [`ZipArchive::read_entry_ignore_case`] is the
//! documented ASCII-case-insensitive convenience for callers that cannot trust
//! the producer's casing.
//!
//! # Error classification
//!
//! - [`Error::Validation`] — the bytes are not a ZIP archive or are structurally
//!   truncated (missing EOCD, bad signatures, inconsistent headers).
//! - [`Error::Retrieval`] — the container is well formed but an entry's payload
//!   is corrupt: `CRC-32` mismatch, wrong decoded size, or a failed `DEFLATE`
//!   stream.
//! - [`Error::Unsupported`] — ZIP64, split archives, encrypted entries, and
//!   compression methods other than stored/`deflate`.
//! - [`Error::NotFound`] — a requested entry (or the file passed to
//!   [`ZipArchive::open`]) does not exist.

use std::path::Path;

use openjiuwen_common::error::{Error, Result};

use crate::parsing::inflate::{crc32, inflate, DEFAULT_MAX_OUTPUT_BYTES};

/// End-of-central-directory signature (`PK\x05\x06`).
const EOCD_SIGNATURE: u32 = 0x0605_4B50;
/// Central-directory file header signature (`PK\x01\x02`).
const CENTRAL_DIRECTORY_SIGNATURE: u32 = 0x0201_4B50;
/// Local file header signature (`PK\x03\x04`).
const LOCAL_HEADER_SIGNATURE: u32 = 0x0403_4B50;
/// Size of the EOCD record without its optional comment.
const EOCD_SIZE: usize = 22;
/// Largest legal archive comment, and therefore the furthest back an EOCD scan
/// needs to look.
const MAX_ARCHIVE_COMMENT: usize = 65_535;
/// Size of a central-directory file header without name/extra/comment fields.
const CENTRAL_HEADER_SIZE: usize = 46;
/// Size of a local file header without name/extra fields.
const LOCAL_HEADER_SIZE: usize = 30;
/// General-purpose flag bit 0: the entry is encrypted.
const FLAG_ENCRYPTED: u16 = 0x0001;
/// General-purpose flag bit 11: the entry name is UTF-8.
const FLAG_UTF8: u16 = 0x0800;

/// Compression method 0: the payload is stored verbatim.
pub const COMPRESSION_METHOD_STORED: u16 = 0;
/// Compression method 8: the payload is a raw `DEFLATE` stream.
pub const COMPRESSION_METHOD_DEFLATE: u16 = 8;

/// Returns the APPNOTE name of a ZIP compression method.
///
/// ```
/// # use openjiuwen_retrieval::parsing::zip::compression_method_name;
/// assert_eq!(compression_method_name(0), "stored");
/// assert_eq!(compression_method_name(8), "deflate");
/// assert_eq!(compression_method_name(12), "bzip2");
/// assert_eq!(compression_method_name(4242), "unknown");
/// ```
pub fn compression_method_name(method: u16) -> &'static str {
    match method {
        COMPRESSION_METHOD_STORED => "stored",
        COMPRESSION_METHOD_DEFLATE => "deflate",
        1 => "shrink",
        2..=5 => "reduce",
        6 => "implode",
        9 => "deflate64",
        10 => "pkware implode",
        12 => "bzip2",
        14 => "lzma",
        18 => "ibm terasea",
        19 => "ibm lz77",
        93 => "zstandard",
        94 => "mp3",
        95 => "xz",
        96 => "jpeg",
        97 => "wavpack",
        98 => "ppmd",
        99 => "aes",
        _ => "unknown",
    }
}

/// One central-directory entry of a ZIP archive.
///
/// Only the fields a document reader needs are kept; timestamps, attributes, and
/// the archive comment are dropped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZipEntry {
    /// Entry name as stored in the archive (`/`-separated, exact case).
    pub name: String,
    /// Compression method: [`COMPRESSION_METHOD_STORED`] or
    /// [`COMPRESSION_METHOD_DEFLATE`] (anything else is rejected at parse time).
    pub compression_method: u16,
    /// General-purpose flag bits from the central directory.
    pub flags: u16,
    /// `CRC-32` of the uncompressed payload.
    pub crc32: u32,
    /// Size of the payload as stored in the archive.
    pub compressed_size: u32,
    /// Size of the payload after decompression.
    pub uncompressed_size: u32,
    /// Offset of this entry's local file header within the archive.
    pub local_header_offset: u32,
}

impl ZipEntry {
    /// Returns `true` for directory entries (names ending in `/`, which carry no payload).
    pub fn is_directory(&self) -> bool {
        self.name.ends_with('/')
    }

    /// Returns `true` when the producer flagged the name as UTF-8 (flag bit 11).
    pub fn is_utf8_name(&self) -> bool {
        self.flags & FLAG_UTF8 != 0
    }

    /// Returns `"stored"` or `"deflate"` for the supported methods.
    pub fn compression_method_name(&self) -> &'static str {
        compression_method_name(self.compression_method)
    }
}

/// A ZIP archive parsed from memory.
///
/// The archive keeps its bytes so entries can be extracted without re-reading the
/// source file; use [`ZipArchive::open`] to load one from disk.
#[derive(Debug, Clone)]
pub struct ZipArchive {
    /// The complete archive bytes.
    data: Vec<u8>,
    /// Entries in central-directory order.
    entries: Vec<ZipEntry>,
}

impl ZipArchive {
    /// Parses a ZIP archive from its complete byte content.
    ///
    /// # Errors
    ///
    /// See the module documentation: structural damage is
    /// [`Error::Validation`], unsupported variants are [`Error::Unsupported`].
    pub fn from_bytes(data: Vec<u8>) -> Result<Self> {
        let eocd = find_end_of_central_directory(&data)?;
        let entries_on_disk = read_u16(&data, eocd + 8)?;
        let total_entries = read_u16(&data, eocd + 10)?;
        let central_directory_size = read_u32(&data, eocd + 12)?;
        let central_directory_offset = read_u32(&data, eocd + 16)?;
        let disk_number = read_u16(&data, eocd + 4)?;
        let central_directory_disk = read_u16(&data, eocd + 6)?;

        if entries_on_disk == u16::MAX
            || total_entries == u16::MAX
            || central_directory_size == u32::MAX
            || central_directory_offset == u32::MAX
            || disk_number == u16::MAX
            || central_directory_disk == u16::MAX
        {
            return Err(Error::Unsupported(
                "ZIP64 archives are not supported".to_owned(),
            ));
        }
        if disk_number != 0 || central_directory_disk != 0 || entries_on_disk != total_entries {
            return Err(Error::Unsupported(
                "multi-disk (split) ZIP archives are not supported".to_owned(),
            ));
        }

        let directory_start = central_directory_offset as usize;
        let directory_end = directory_start
            .checked_add(central_directory_size as usize)
            .filter(|&end| end <= data.len())
            .ok_or_else(|| {
                Error::Validation(
                    "ZIP central directory extends past the end of the archive".to_owned(),
                )
            })?;
        let mut entries = Vec::new();
        let mut offset = directory_start;
        for _ in 0..total_entries {
            let (entry, next) = parse_central_directory_entry(&data, offset)?;
            entries.push(entry);
            offset = next;
        }
        if offset > directory_end {
            return Err(Error::Validation(
                "ZIP central directory entries overrun the declared directory size".to_owned(),
            ));
        }
        Ok(Self { data, entries })
    }

    /// Reads and parses the ZIP archive at `path`.
    ///
    /// The file is read with `tokio::fs`, so this never blocks the async runtime.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] for a missing file, [`Error::Transport`] for
    /// any other I/O error, and the [`ZipArchive::from_bytes`] errors for a
    /// malformed archive.
    pub async fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let data = tokio::fs::read(path)
            .await
            .map_err(|error| match error.kind() {
                std::io::ErrorKind::NotFound => {
                    Error::NotFound(format!("File {} does not exist", path.display()))
                }
                _ => Error::Transport(format!("reading {}: {error}", path.display())),
            })?;
        Self::from_bytes(data)
    }

    /// Returns every entry, in central-directory order (directories included).
    pub fn entries(&self) -> &[ZipEntry] {
        &self.entries
    }

    /// Returns every entry name, in central-directory order.
    pub fn entry_names(&self) -> Vec<&str> {
        self.entries
            .iter()
            .map(|entry| entry.name.as_str())
            .collect()
    }

    /// Returns the entry called exactly `name`, or `None`.
    pub fn entry(&self, name: &str) -> Option<&ZipEntry> {
        self.entries.iter().find(|entry| entry.name == name)
    }

    /// Returns `true` when an entry called exactly `name` exists.
    pub fn has_entry(&self, name: &str) -> bool {
        self.entry(name).is_some()
    }

    /// Number of entries (directories included).
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when the archive holds no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns the total size of the archive in bytes.
    pub fn archive_size(&self) -> usize {
        self.data.len()
    }

    /// Extracts and decompresses the entry called exactly `name`.
    ///
    /// The payload is verified against the recorded `CRC-32`, and deflated
    /// entries are additionally verified against the recorded uncompressed size.
    ///
    /// # Errors
    ///
    /// [`Error::NotFound`] for an unknown name, [`Error::Validation`] for a
    /// directory or a malformed local header, [`Error::Retrieval`] for a
    /// checksum/size mismatch or a corrupt `DEFLATE` payload.
    pub fn read_entry(&self, name: &str) -> Result<Vec<u8>> {
        let entry = self.entry(name).ok_or_else(|| {
            Error::NotFound(format!("ZIP entry {name:?} not found in the archive"))
        })?;
        self.extract(entry)
    }

    /// Extracts the entry called `name`, ignoring ASCII case when the exact name
    /// is absent.
    ///
    /// Exact matches always win; otherwise the first entry whose name matches
    /// case-insensitively (compared ASCII byte by byte) is used. Byte-oriented
    /// archives rarely agree on casing, which is why this helper exists, but ZIP
    /// paths are still `/`-separated and never normalised.
    ///
    /// # Errors
    ///
    /// Same as [`ZipArchive::read_entry`].
    pub fn read_entry_ignore_case(&self, name: &str) -> Result<Vec<u8>> {
        let entry = self.entry_ignore_case(name).ok_or_else(|| {
            Error::NotFound(format!(
                "ZIP entry {name:?} not found in the archive (case-insensitive lookup)"
            ))
        })?;
        self.extract(entry)
    }

    /// Extracts an entry and decodes it as UTF-8, replacing invalid sequences.
    ///
    /// Convenience for XML/text parts; binary parts should use
    /// [`ZipArchive::read_entry`].
    ///
    /// # Errors
    ///
    /// Same as [`ZipArchive::read_entry`].
    pub fn read_text(&self, name: &str) -> Result<String> {
        Ok(String::from_utf8_lossy(&self.read_entry(name)?).into_owned())
    }

    /// Finds an entry by exact name, falling back to an ASCII-case-insensitive match.
    fn entry_ignore_case(&self, name: &str) -> Option<&ZipEntry> {
        if let Some(entry) = self.entry(name) {
            return Some(entry);
        }
        let wanted = name.to_ascii_lowercase();
        self.entries
            .iter()
            .find(|entry| entry.name.to_ascii_lowercase() == wanted)
    }

    /// Extracts and verifies the payload of `entry`.
    fn extract(&self, entry: &ZipEntry) -> Result<Vec<u8>> {
        if entry.is_directory() {
            return Err(Error::Validation(format!(
                "ZIP entry {:?} is a directory and has no payload",
                entry.name
            )));
        }
        let declared_size = entry.uncompressed_size as usize;
        if declared_size > DEFAULT_MAX_OUTPUT_BYTES {
            return Err(Error::Validation(format!(
                "ZIP entry {:?} declares {declared_size} bytes, exceeding the \
                 {DEFAULT_MAX_OUTPUT_BYTES}-byte limit",
                entry.name
            )));
        }

        let header_offset = entry.local_header_offset as usize;
        let signature = read_u32(&self.data, header_offset)?;
        if signature != LOCAL_HEADER_SIGNATURE {
            return Err(Error::Validation(format!(
                "invalid ZIP local header signature {signature:#010x} at offset {header_offset}"
            )));
        }
        let local_method = read_u16(&self.data, header_offset + 8)?;
        if local_method != entry.compression_method {
            return Err(Error::Validation(format!(
                "ZIP entry {:?} declares compression method {} in the local header but {} in the central directory",
                entry.name, local_method, entry.compression_method
            )));
        }
        let name_length = usize::from(read_u16(&self.data, header_offset + 26)?);
        let extra_length = usize::from(read_u16(&self.data, header_offset + 28)?);
        let data_start = header_offset + LOCAL_HEADER_SIZE + name_length + extra_length;
        let data_end = data_start
            .checked_add(entry.compressed_size as usize)
            .filter(|&end| end <= self.data.len())
            .ok_or_else(|| {
                Error::Validation(format!(
                    "ZIP entry {:?} payload extends past the end of the archive",
                    entry.name
                ))
            })?;
        let compressed = &self.data[data_start..data_end];

        let output = if entry.compression_method == COMPRESSION_METHOD_STORED {
            compressed.to_vec()
        } else {
            inflate(compressed, DEFAULT_MAX_OUTPUT_BYTES)?
        };

        if output.len() != declared_size {
            return Err(Error::Retrieval(format!(
                "ZIP entry {:?} size mismatch: header declares {declared_size} bytes, data yields {}",
                entry.name,
                output.len()
            )));
        }
        let actual_crc = crc32(&output);
        if actual_crc != entry.crc32 {
            return Err(Error::Retrieval(format!(
                "ZIP entry {:?} CRC-32 mismatch: header declares {:#010x}, data yields {actual_crc:#010x}",
                entry.name, entry.crc32
            )));
        }
        Ok(output)
    }
}

/// Locates the end-of-central-directory record, scanning backwards.
///
/// A trailing archive comment is allowed; bytes after the record are ignored.
fn find_end_of_central_directory(data: &[u8]) -> Result<usize> {
    if data.len() < EOCD_SIZE {
        return Err(Error::Validation(
            "not a ZIP archive: shorter than an end-of-central-directory record".to_owned(),
        ));
    }
    let lowest = data.len().saturating_sub(EOCD_SIZE + MAX_ARCHIVE_COMMENT);
    let mut offset = data.len() - EOCD_SIZE;
    loop {
        if read_u32(data, offset)? == EOCD_SIGNATURE {
            let comment_length = usize::from(read_u16(data, offset + 20)?);
            if offset + EOCD_SIZE + comment_length <= data.len() {
                return Ok(offset);
            }
        }
        if offset == lowest {
            return Err(Error::Validation(
                "not a ZIP archive: end-of-central-directory record not found".to_owned(),
            ));
        }
        offset -= 1;
    }
}

/// Parses one central-directory file header and returns it with the next offset.
fn parse_central_directory_entry(data: &[u8], offset: usize) -> Result<(ZipEntry, usize)> {
    let signature = read_u32(data, offset)?;
    if signature != CENTRAL_DIRECTORY_SIGNATURE {
        return Err(Error::Validation(format!(
            "invalid ZIP central directory signature {signature:#010x} at offset {offset}"
        )));
    }
    let flags = read_u16(data, offset + 8)?;
    let compression_method = read_u16(data, offset + 10)?;
    let crc = read_u32(data, offset + 16)?;
    let compressed_size = read_u32(data, offset + 20)?;
    let uncompressed_size = read_u32(data, offset + 24)?;
    let name_length = usize::from(read_u16(data, offset + 28)?);
    let extra_length = usize::from(read_u16(data, offset + 30)?);
    let comment_length = usize::from(read_u16(data, offset + 32)?);
    let local_header_offset = read_u32(data, offset + 42)?;

    let name_start = offset + CENTRAL_HEADER_SIZE;
    let name_end = name_start
        .checked_add(name_length)
        .filter(|&end| end <= data.len())
        .ok_or_else(|| {
            Error::Validation(format!(
                "ZIP central directory entry at offset {offset} has a truncated name"
            ))
        })?;
    let name = decode_entry_name(&data[name_start..name_end], flags & FLAG_UTF8 != 0)?;

    if flags & FLAG_ENCRYPTED != 0 {
        return Err(Error::Unsupported(format!(
            "encrypted ZIP entries are not supported (entry {name:?})"
        )));
    }
    if !matches!(
        compression_method,
        COMPRESSION_METHOD_STORED | COMPRESSION_METHOD_DEFLATE
    ) {
        return Err(Error::Unsupported(format!(
            "ZIP compression method {compression_method} ({}) is not supported (entry {name:?})",
            compression_method_name(compression_method)
        )));
    }

    let next = name_end + extra_length + comment_length;
    if next > data.len() {
        return Err(Error::Validation(format!(
            "ZIP central directory entry {name:?} extends past the end of the archive"
        )));
    }
    Ok((
        ZipEntry {
            name,
            compression_method,
            flags,
            crc32: crc,
            compressed_size,
            uncompressed_size,
            local_header_offset,
        },
        next,
    ))
}

/// Decodes an entry name: strict UTF-8 when flagged, lossy otherwise.
fn decode_entry_name(bytes: &[u8], utf8: bool) -> Result<String> {
    if !utf8 {
        return Ok(String::from_utf8_lossy(bytes).into_owned());
    }
    std::str::from_utf8(bytes)
        .map(str::to_owned)
        .map_err(|_| Error::Validation("ZIP entry name is not valid UTF-8".to_owned()))
}

/// Reads a little-endian `u16` at `offset`, rejecting out-of-range offsets.
fn read_u16(data: &[u8], offset: usize) -> Result<u16> {
    let bytes = data
        .get(offset..)
        .and_then(|rest| rest.get(..2))
        .ok_or_else(|| Error::Validation(format!("truncated ZIP structure at offset {offset}")))?;
    Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
}

/// Reads a little-endian `u32` at `offset`, rejecting out-of-range offsets.
fn read_u32(data: &[u8], offset: usize) -> Result<u32> {
    let bytes = data
        .get(offset..)
        .and_then(|rest| rest.get(..4))
        .ok_or_else(|| Error::Validation(format!("truncated ZIP structure at offset {offset}")))?;
    Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;

    /// Builds a 22-byte end-of-central-directory record.
    fn eocd(entries: u16, directory_size: u32, directory_offset: u32) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&EOCD_SIGNATURE.to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes());
        bytes.extend_from_slice(&entries.to_le_bytes());
        bytes.extend_from_slice(&entries.to_le_bytes());
        bytes.extend_from_slice(&directory_size.to_le_bytes());
        bytes.extend_from_slice(&directory_offset.to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes());
        bytes
    }

    #[test]
    fn compression_method_names_cover_known_methods() {
        assert_eq!(compression_method_name(0), "stored");
        assert_eq!(compression_method_name(8), "deflate");
        assert_eq!(compression_method_name(12), "bzip2");
        assert_eq!(compression_method_name(93), "zstandard");
        assert_eq!(compression_method_name(u16::MAX), "unknown");
    }

    #[test]
    fn integer_readers_reject_out_of_range_offsets() {
        let data = [0x01, 0x02, 0x03];
        assert_eq!(read_u16(&data, 1).unwrap(), 0x0302);
        let error = read_u16(&data, 2).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(
            error.to_string().contains("truncated ZIP structure"),
            "{error}"
        );
        assert!(read_u32(&data, 0).is_err());
        assert!(read_u32(&data, usize::MAX).is_err());
    }

    #[test]
    fn names_are_strict_utf8_only_when_flagged() {
        assert_eq!(
            decode_entry_name(b"word/document.xml", false).unwrap(),
            "word/document.xml"
        );
        assert_eq!(
            decode_entry_name(&[0x61, 0xFF, 0x62], false).unwrap(),
            "a\u{fffd}b"
        );
        let error = decode_entry_name(&[0x61, 0xFF, 0x62], true).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
    }

    #[test]
    fn directory_and_utf8_flags_are_exposed() {
        let directory = ZipEntry {
            name: "word/".to_owned(),
            compression_method: COMPRESSION_METHOD_STORED,
            flags: FLAG_UTF8,
            crc32: 0,
            compressed_size: 0,
            uncompressed_size: 0,
            local_header_offset: 0,
        };
        assert!(directory.is_directory());
        assert!(directory.is_utf8_name());
        assert_eq!(directory.compression_method_name(), "stored");
        let file = ZipEntry {
            name: "word/document.xml".to_owned(),
            compression_method: COMPRESSION_METHOD_DEFLATE,
            ..directory
        };
        assert!(!file.is_directory());
        assert_eq!(file.compression_method_name(), "deflate");
    }

    #[test]
    fn empty_archive_parses() {
        let archive = ZipArchive::from_bytes(eocd(0, 0, 0)).unwrap();
        assert!(archive.is_empty());
        assert_eq!(archive.len(), 0);
        assert_eq!(archive.archive_size(), EOCD_SIZE);
        assert_eq!(archive.entries().len(), 0);
        assert!(!archive.has_entry("x"));
    }

    #[test]
    fn eocd_scan_tolerates_a_trailing_comment() {
        let mut bytes = eocd(0, 0, 0);
        bytes[20..22].copy_from_slice(&3u16.to_le_bytes());
        bytes.extend_from_slice(b"hey");
        let archive = ZipArchive::from_bytes(bytes).unwrap();
        assert!(archive.is_empty());
    }

    #[test]
    fn missing_eocd_is_reported() {
        let error = ZipArchive::from_bytes(b"PK\x03\x04not an archive".to_vec()).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(
            error.to_string().contains("end-of-central-directory"),
            "{error}"
        );

        let error = ZipArchive::from_bytes(vec![0u8; 4]).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("shorter than"), "{error}");
    }

    #[test]
    fn zip64_archives_are_unsupported() {
        let mut bytes = eocd(0xFFFF, 0, 0);
        bytes[10..12].copy_from_slice(&0xFFFFu16.to_le_bytes());
        let error = ZipArchive::from_bytes(bytes).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert!(error.to_string().contains("ZIP64"), "{error}");

        // A sentinel central-directory offset is also a ZIP64 marker.
        let error = ZipArchive::from_bytes(eocd(0, 0, u32::MAX)).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
    }

    #[test]
    fn split_archives_are_unsupported() {
        let mut bytes = eocd(0, 0, 0);
        bytes[4..6].copy_from_slice(&1u16.to_le_bytes());
        let error = ZipArchive::from_bytes(bytes).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert!(error.to_string().contains("multi-disk"), "{error}");
    }

    #[test]
    fn central_directory_beyond_the_archive_is_rejected() {
        let error = ZipArchive::from_bytes(eocd(1, 46, 10)).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("past the end"), "{error}");
    }

    #[test]
    fn unknown_entry_reports_not_found() {
        let archive = ZipArchive::from_bytes(eocd(0, 0, 0)).unwrap();
        let error = archive.read_entry("word/document.xml").unwrap_err();
        assert_eq!(error.kind(), ErrorKind::NotFound);
        assert!(error.to_string().contains("not found"), "{error}");
        let error = archive
            .read_entry_ignore_case("word/document.xml")
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::NotFound);
    }
}
