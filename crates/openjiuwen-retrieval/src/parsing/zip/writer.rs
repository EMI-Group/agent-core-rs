//! Minimal ZIP container writing (`PKZIP` / APPNOTE 6.3), stored entries only.
//!
//! [`ZipWriter`] is the writing counterpart of
//! [`ZipArchive`](super::ZipArchive), meant for tests and tools that need to
//! build a small archive in memory without adding a ZIP crate: DOCX and XLSX
//! fixtures are assemblies of XML parts, so the parser suites have to produce
//! containers themselves.
//! Only stored (uncompressed) entries are written — no `DEFLATE`, timestamps,
//! attributes, comments, data descriptors, or ZIP64 — and every name is flagged
//! as UTF-8 (general-purpose flag bit 11), so non-ASCII part names such as
//! `文档.xml` round-trip byte for byte.
//!
//! Each [`ZipWriter::add_stored`] call appends a local file header and the raw
//! payload straight to an internal buffer and records the entry for the central
//! directory; [`ZipWriter::finish`] then appends that directory and the
//! end-of-central-directory record.
//! The limits are the format's own: a name of at most `65_535` bytes, at most
//! `65_535` entries, and a payload or archive offset of at most `u32::MAX` bytes;
//! exceeding any of them returns [`Error::Validation`].
//!
//! # Example
//!
//! ```
//! # use openjiuwen_retrieval::parsing::zip::{ZipArchive, ZipWriter};
//! let mut writer = ZipWriter::new();
//! writer.add_stored("word/document.xml", "<w:document/>").unwrap();
//! let archive = ZipArchive::from_bytes(writer.finish().unwrap()).unwrap();
//! assert_eq!(archive.entry_names(), vec!["word/document.xml"]);
//! assert_eq!(archive.read_entry("word/document.xml").unwrap(), b"<w:document/>");
//! ```

use openjiuwen_common::error::{Error, Result};

use super::{
    CENTRAL_DIRECTORY_SIGNATURE, CENTRAL_HEADER_SIZE, COMPRESSION_METHOD_STORED, EOCD_SIGNATURE,
    FLAG_UTF8, LOCAL_HEADER_SIGNATURE, LOCAL_HEADER_SIZE,
};
use crate::parsing::inflate::crc32;

/// Version made by and version needed to extract: ZIP 2.0, enough for stored
/// entries with UTF-8 names.
const VERSION: u16 = 20;
/// Largest entry name the ZIP format can record, in bytes.
const MAX_NAME_BYTES: usize = u16::MAX as usize;
/// Largest number of entries a non-ZIP64 archive can record.
const MAX_ENTRIES: usize = u16::MAX as usize;

/// What [`ZipWriter::finish`] needs to emit one central-directory header.
#[derive(Debug)]
struct EntryRecord {
    /// Entry name as written (UTF-8, `/`-separated).
    name: String,
    /// `CRC-32` of the payload.
    crc32: u32,
    /// Payload length, identical for the compressed and uncompressed size.
    size: u32,
    /// Offset of this entry's local file header within the archive.
    local_header_offset: u32,
}

/// Builds a stored-entry ZIP archive in memory.
///
/// Entries are written in insertion order and keep their exact names; see the
/// [module documentation](self) for the subset of the format that is emitted.
#[derive(Debug, Default)]
pub struct ZipWriter {
    /// Local file headers and their payloads, in insertion order.
    data: Vec<u8>,
    /// Central-directory metadata for every written entry.
    entries: Vec<EntryRecord>,
}

impl ZipWriter {
    /// Creates an empty writer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a stored entry called `name` whose payload is `data`.
    ///
    /// The local file header and the payload are written immediately, so calls
    /// can be chained; a rejected entry leaves the writer unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `name` is empty or longer than `65_535`
    /// bytes, when the archive already holds `65_535` entries, or when a payload
    /// length or archive offset no longer fits the 32-bit fields of a non-ZIP64
    /// archive.
    pub fn add_stored(
        &mut self,
        name: impl Into<String>,
        data: impl Into<Vec<u8>>,
    ) -> Result<&mut Self> {
        let name = name.into();
        let payload = data.into();
        let name_length = checked_name(&name)?;
        if self.entries.len() >= MAX_ENTRIES {
            return Err(Error::Validation(format!(
                "ZIP archive already holds {} entries, the maximum a non-ZIP64 archive can record",
                self.entries.len()
            )));
        }
        let size = checked_size(payload.len(), &name)?;
        let checksum = crc32(&payload);
        let local_header_offset = checked_offset(self.data.len())?;
        let end = self
            .data
            .len()
            .checked_add(LOCAL_HEADER_SIZE + usize::from(name_length) + payload.len())
            .ok_or_else(|| {
                Error::Validation(format!(
                    "ZIP archive is {} bytes long, exceeding the addressable 32-bit archive size",
                    self.data.len()
                ))
            })?;
        checked_offset(end)?;

        self.data
            .reserve(LOCAL_HEADER_SIZE + usize::from(name_length) + payload.len());
        write_u32(&mut self.data, LOCAL_HEADER_SIGNATURE);
        write_u16(&mut self.data, VERSION);
        write_u16(&mut self.data, FLAG_UTF8);
        write_u16(&mut self.data, COMPRESSION_METHOD_STORED);
        write_u16(&mut self.data, 0); // last modification time
        write_u16(&mut self.data, 0); // last modification date
        write_u32(&mut self.data, checksum);
        write_u32(&mut self.data, size); // compressed size (stored, so equal)
        write_u32(&mut self.data, size); // uncompressed size
        write_u16(&mut self.data, name_length);
        write_u16(&mut self.data, 0); // extra field length
        self.data.extend_from_slice(name.as_bytes());
        self.data.extend_from_slice(&payload);

        self.entries.push(EntryRecord {
            name,
            crc32: checksum,
            size,
            local_header_offset,
        });
        Ok(self)
    }

    /// Appends the central directory and the end-of-central-directory record,
    /// consuming the writer and returning the complete archive bytes.
    ///
    /// A writer with no entries yields exactly the 22-byte
    /// end-of-central-directory record, which readers accept as an empty
    /// archive.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the central-directory offset or size
    /// no longer fits the 32-bit fields of a non-ZIP64 archive.
    pub fn finish(mut self) -> Result<Vec<u8>> {
        let directory_offset = checked_offset(self.data.len())?;
        let mut directory = Vec::with_capacity(self.entries.len() * CENTRAL_HEADER_SIZE);
        for entry in &self.entries {
            write_u32(&mut directory, CENTRAL_DIRECTORY_SIGNATURE);
            write_u16(&mut directory, VERSION); // version made by
            write_u16(&mut directory, VERSION); // version needed to extract
            write_u16(&mut directory, FLAG_UTF8);
            write_u16(&mut directory, COMPRESSION_METHOD_STORED);
            write_u16(&mut directory, 0); // last modification time
            write_u16(&mut directory, 0); // last modification date
            write_u32(&mut directory, entry.crc32);
            write_u32(&mut directory, entry.size); // compressed size
            write_u32(&mut directory, entry.size); // uncompressed size
            let name_length = checked_name(&entry.name)?;
            write_u16(&mut directory, name_length);
            write_u16(&mut directory, 0); // extra field length
            write_u16(&mut directory, 0); // file comment length
            write_u16(&mut directory, 0); // disk number where the entry starts
            write_u16(&mut directory, 0); // internal file attributes
            write_u32(&mut directory, 0); // external file attributes
            write_u32(&mut directory, entry.local_header_offset);
            directory.extend_from_slice(entry.name.as_bytes());
        }
        let directory_size = checked_offset(directory.len())?;
        let entry_count = u16::try_from(self.entries.len()).map_err(|_| {
            Error::Validation(format!(
                "ZIP archive holds {} entries, exceeding the {MAX_ENTRIES}-entry limit",
                self.entries.len()
            ))
        })?;

        self.data.extend_from_slice(&directory);
        write_u32(&mut self.data, EOCD_SIGNATURE);
        write_u16(&mut self.data, 0); // number of this disk
        write_u16(&mut self.data, 0); // disk holding the central directory
        write_u16(&mut self.data, entry_count); // entries on this disk
        write_u16(&mut self.data, entry_count); // total entries
        write_u32(&mut self.data, directory_size);
        write_u32(&mut self.data, directory_offset);
        write_u16(&mut self.data, 0); // archive comment length
        Ok(self.data)
    }

    /// Number of entries written so far.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when no entry has been written.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Validates an entry name and returns its length in bytes as a header field.
fn checked_name(name: &str) -> Result<u16> {
    if name.is_empty() {
        return Err(Error::Validation(
            "ZIP entry name must not be empty".to_owned(),
        ));
    }
    u16::try_from(name.len()).map_err(|_| {
        Error::Validation(format!(
            "ZIP entry name is {} bytes long, exceeding the {MAX_NAME_BYTES}-byte limit",
            name.len()
        ))
    })
}

/// Converts a payload length into the 32-bit size a ZIP header records.
fn checked_size(size: usize, name: &str) -> Result<u32> {
    let limit = u32::MAX as usize;
    u32::try_from(size).map_err(|_| {
        Error::Validation(format!(
            "ZIP entry {name:?} payload is {size} bytes long, exceeding the {limit}-byte limit"
        ))
    })
}

/// Converts an archive offset into the 32-bit offset a ZIP header records.
fn checked_offset(offset: usize) -> Result<u32> {
    let limit = u32::MAX as usize;
    u32::try_from(offset).map_err(|_| {
        Error::Validation(format!(
            "ZIP archive offset {offset} exceeds the {limit}-byte limit of a non-ZIP64 archive"
        ))
    })
}

/// Appends a little-endian `u16` to `buffer`.
fn write_u16(buffer: &mut Vec<u8>, value: u16) {
    buffer.extend_from_slice(&value.to_le_bytes());
}

/// Appends a little-endian `u32` to `buffer`.
fn write_u32(buffer: &mut Vec<u8>, value: u32) {
    buffer.extend_from_slice(&value.to_le_bytes());
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
