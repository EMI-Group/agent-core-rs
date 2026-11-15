//! `zip` reading (stored and `DEFLATE`) and deterministic stored writing.
//!
//! The reader implements the classic 32-bit `zip` container (PKWARE APPNOTE
//! 6.3) on top of the shared in-crate helpers, with no compression dependency:
//!
//! - [`find_eocd`] scans backwards for the end-of-central-directory record
//!   (`PK\x05\x06`) inside the legal `22 + 65535` byte comment window and
//!   prefers the rightmost matching record;
//! - [`parse_central_directory`] walks the `PK\x01\x02` records, which are
//!   authoritative: local-header sizes are never trusted because streamed
//!   archives legally leave them zero and carry a trailing data descriptor
//!   (general-purpose flag bit 3);
//! - [`read_entry`] validates the local header (`PK\x03\x04`), slices the
//!   compressed payload, and verifies the declared uncompressed size and the
//!   CRC-32 of the result.
//!
//! Supported compression methods are stored (`0`) and `DEFLATE` (`8`, decoded
//! by [`inflate`](super::inflate)). Everything else is a typed
//! [`Error::unsupported`]: `ZIP64` (a `PK\x06\x06` record, a `PK\x06\x07`
//! locator, a `0xffff_ffff` size/offset escape value, or `65535` total entries),
//! encrypted entries (flag bit 0), and any other compression method (bzip2,
//! `LZMA`, zstd, AES, ...).
//!
//! Names are decoded as UTF-8 when the UTF-8 flag (bit 11) is set, and strictly
//! so: invalid UTF-8 with that flag is a validation error. Without the flag a
//! name is decoded as UTF-8 when it happens to be valid and otherwise through
//! the CP437 code page, the historical `zip` default, so old archives keep their
//! accented names. Backslashes are never normalised: the decoded name goes
//! through [`validate_path`], which rejects absolute paths, `..` components, and
//! backslashes, and a trailing `/` is kept because it is part of the name.
//!
//! Entries come back in central-directory order. A directory entry (a trailing
//! `/` in the name, or the DOS directory bit `0x10` in the external attributes)
//! carries no data: its declared sizes are ignored and it is charged zero bytes
//! to the caps. Every other entry is charged its declared uncompressed size to
//! [`CapTracker`] before decompressing, so a zip bomb fails with a validation
//! error instead of allocating.
//!
//! [`ArchiveWriter::write_stored`] is the deterministic counterpart used by
//! tests and by callers that need a round trip: stored-only, a fixed DOS
//! timestamp of `1980-01-01 00:00:00` (`time = 0x0000`, `date = 0x0021`), no
//! extras, no comments, and entries in the order given.

use super::crc::crc32;
use super::inflate::inflate;
use super::{
    read_u16_le, read_u32_le, slice_at, validate_path, ArchiveEntry, ArchiveEntryKind, CapTracker,
};
use openjiuwen_common::error::{Error, Result};

/// Signature of a local file header.
const LOCAL_SIGNATURE: &[u8] = b"PK\x03\x04";
/// Signature of a central-directory file header.
const CENTRAL_SIGNATURE: &[u8] = b"PK\x01\x02";
/// Signature of the end-of-central-directory record.
const EOCD_SIGNATURE: &[u8] = b"PK\x05\x06";
/// Signature of the `ZIP64` end-of-central-directory record.
const ZIP64_EOCD_SIGNATURE: &[u8] = b"PK\x06\x06";
/// Signature of the `ZIP64` end-of-central-directory locator.
const ZIP64_LOCATOR_SIGNATURE: &[u8] = b"PK\x06\x07";
/// Fixed size of the end-of-central-directory record, before its comment.
const EOCD_FIXED: usize = 22;
/// Fixed size of a central-directory record, before its variable fields.
const CENTRAL_FIXED: usize = 46;
/// Fixed size of a local file header, before its variable fields.
const LOCAL_FIXED: usize = 30;
/// Fixed size of the `ZIP64` end-of-central-directory record.
const ZIP64_EOCD_FIXED: usize = 56;
/// Fixed size of the `ZIP64` end-of-central-directory locator.
const ZIP64_LOCATOR_FIXED: usize = 20;
/// Longest `zip` comment, in bytes, that hides a preceding record.
const MAX_COMMENT_LEN: usize = u16::MAX as usize;
/// General-purpose flag bit 0: the entry is encrypted.
const FLAG_ENCRYPTED: u16 = 0x0001;
/// General-purpose flag bit 11: the entry name is UTF-8.
const FLAG_UTF8: u16 = 0x0800;
/// Compression method 0: stored.
const METHOD_STORED: u16 = 0;
/// Compression method 8: `DEFLATE`.
const METHOD_DEFLATE: u16 = 8;
/// The DOS directory attribute, the low byte of the external attributes.
const DOS_DIRECTORY_ATTRIBUTE: u32 = 0x10;
/// The escape value `ZIP64` uses in a 32-bit size or offset field.
const ZIP64_MARKER: u32 = 0xFFFF_FFFF;
/// `zip` version 2.0, which introduced `DEFLATE` and folders; written by both
/// the local header and the central-directory record.
const VERSION: u16 = 20;
/// Fixed DOS modification time `00:00:00`, used by the writer.
const DOS_TIME: u16 = 0x0000;
/// Fixed DOS modification date `1980-01-01`, the earliest DOS date.
const DOS_DATE: u16 = 0x0021;
/// Unix mode bits of a regular file (`0o100644`, shifted into the high half of
/// the external attributes).
const UNIX_FILE_MODE: u32 = 0o100_644;
/// Unix mode bits of a directory (`0o040755`).
const UNIX_DIRECTORY_MODE: u32 = 0o040_755;
/// Upper bound on the records pre-allocated from the declared entry count.
const MAX_PREALLOCATED_RECORDS: usize = 1024;

/// Unpacks a `zip` archive into entries, in central-directory order.
///
/// # Errors
///
/// Returns [`Error::validation`] for malformed or truncated archives and
/// [`Error::unsupported`] for ZIP64, encrypted entries, and compression methods
/// other than stored and DEFLATE.
pub(crate) fn unpack(bytes: &[u8]) -> Result<Vec<ArchiveEntry>> {
    let eocd = find_eocd(bytes)?;
    let records = parse_central_directory(bytes, eocd)?;
    let mut caps = CapTracker::new();
    let mut entries = Vec::with_capacity(records.len());
    for record in &records {
        entries.push(read_entry(bytes, record, &mut caps)?);
    }
    Ok(entries)
}

/// The end-of-central-directory record's parsed fields.
#[derive(Debug, Clone, Copy)]
struct Eocd {
    /// Number of central-directory records to read.
    total_entries: usize,
    /// Where the central directory starts.
    cd_offset: usize,
}

/// One parsed central-directory record, the authoritative view of an entry.
#[derive(Debug, Clone)]
struct CentralRecord {
    /// Decoded entry name, including any trailing `/`.
    name: String,
    /// General-purpose flags.
    flags: u16,
    /// Compression method.
    method: u16,
    /// CRC-32 of the uncompressed payload.
    crc: u32,
    /// Size of the compressed payload in the archive.
    compressed_size: usize,
    /// Uncompressed size declared by the central directory.
    declared_size: usize,
    /// External attributes; the low byte holds the DOS attribute flags.
    external_attributes: u32,
    /// Offset of the entry's local file header.
    local_offset: usize,
}

/// Finds and parses the rightmost end-of-central-directory record.
///
/// # Errors
///
/// Returns [`Error::validation`] when no record sits in the scan window or the
/// directory it points at runs past the input, and [`Error::unsupported`] for
/// multi-disk archives and `ZIP64` markers.
fn find_eocd(bytes: &[u8]) -> Result<Eocd> {
    let scanned = bytes.len().min(EOCD_FIXED + MAX_COMMENT_LEN);
    if bytes.len() < EOCD_FIXED {
        return Err(eocd_missing(scanned));
    }
    let window_start = bytes.len() - scanned;
    let last_start = bytes.len() - EOCD_FIXED;
    let offset = (window_start..=last_start)
        .rev()
        .find(|&position| has_eocd_signature(bytes, position))
        .ok_or_else(|| eocd_missing(scanned))?;

    if let Some(reason) = zip64_preceding(bytes, offset) {
        return Err(zip64_error(reason));
    }
    let disk = read_u16_le(bytes, offset + 4)?;
    let cd_start_disk = read_u16_le(bytes, offset + 6)?;
    let entries_on_disk = read_u16_le(bytes, offset + 8)?;
    let total_entries = read_u16_le(bytes, offset + 10)?;
    let cd_size = read_u32_le(bytes, offset + 12)?;
    let cd_offset = read_u32_le(bytes, offset + 16)?;

    if disk != 0 || cd_start_disk != 0 || entries_on_disk != total_entries {
        return Err(Error::validation(format!(
            "multi-disk zip archives are not supported: the end-of-central-directory record at \
             byte {offset} names disk {disk}, central-directory start disk {cd_start_disk}, and \
             {entries_on_disk} of {total_entries} entries on this disk"
        )));
    }
    if total_entries == u16::MAX {
        return Err(zip64_error(format!(
            "the end-of-central-directory record at byte {offset} reports {total_entries} entries, \
             the ZIP64 escape value for the entry count"
        )));
    }
    if cd_size == ZIP64_MARKER || cd_offset == ZIP64_MARKER {
        return Err(zip64_error(format!(
            "the end-of-central-directory record at byte {offset} carries a 0xffffffff \
             central-directory size or offset, the ZIP64 escape value"
        )));
    }

    let cd_offset = cd_offset as usize;
    let cd_size = cd_size as usize;
    let cd_end = cd_offset.checked_add(cd_size).ok_or_else(|| {
        Error::validation(format!(
            "zip central directory at byte {cd_offset} declares an unaddressable size of \
             {cd_size} bytes"
        ))
    })?;
    if cd_end > bytes.len() {
        return Err(Error::validation(format!(
            "truncated zip central directory at byte {cd_offset}: {cd_size} bytes declared but \
             only {} remain in the {} byte archive",
            bytes.len().saturating_sub(cd_offset),
            bytes.len()
        )));
    }
    Ok(Eocd {
        total_entries: usize::from(total_entries),
        cd_offset,
    })
}

/// Whether an end-of-central-directory record starts at `position`.
///
/// A candidate only counts when its fixed part and its declared comment both fit
/// inside the input, so trailing bytes cannot masquerade as a record.
fn has_eocd_signature(bytes: &[u8], position: usize) -> bool {
    if bytes.get(position..position + 4) != Some(EOCD_SIGNATURE) {
        return false;
    }
    read_u16_le(bytes, position + 20).is_ok_and(|comment| {
        position
            .checked_add(EOCD_FIXED)
            .and_then(|start| start.checked_add(usize::from(comment)))
            .is_some_and(|end| end <= bytes.len())
    })
}

/// Builds the validation error for a missing end-of-central-directory record.
fn eocd_missing(scanned: usize) -> Error {
    Error::validation(format!(
        "invalid zip archive: no end-of-central-directory record (PK\x05\x06) in the last \
         {scanned} bytes, the {EOCD_FIXED} + {MAX_COMMENT_LEN} byte scan window"
    ))
}

/// Names the `ZIP64` structure that precedes the end-of-central-directory record.
fn zip64_preceding(bytes: &[u8], eocd: usize) -> Option<&'static str> {
    if eocd >= ZIP64_LOCATOR_FIXED
        && bytes.get(eocd - ZIP64_LOCATOR_FIXED..eocd - ZIP64_LOCATOR_FIXED + 4)
            == Some(ZIP64_LOCATOR_SIGNATURE)
    {
        return Some(
            "a ZIP64 end-of-central-directory locator (PK\x06\x07) precedes the \
             end-of-central-directory record",
        );
    }
    if eocd >= ZIP64_EOCD_FIXED
        && bytes.get(eocd - ZIP64_EOCD_FIXED..eocd - ZIP64_EOCD_FIXED + 4)
            == Some(ZIP64_EOCD_SIGNATURE)
    {
        return Some(
            "a ZIP64 end-of-central-directory record (PK\x06\x06) precedes the \
             end-of-central-directory record",
        );
    }
    None
}

/// Builds the unsupported error for `ZIP64` structures.
fn zip64_error(reason: impl std::fmt::Display) -> Error {
    Error::unsupported(format!(
        "zip64 archive: {reason}; ZIP64 archives are not implemented because this in-process \
         reader is bounded to the classic 32-bit zip layout"
    ))
}

/// Parses every central-directory record, in archive order.
///
/// # Errors
///
/// Returns [`Error::validation`] for a truncated or malformed record and
/// [`Error::unsupported`] for `ZIP64` escape values inside a record.
fn parse_central_directory(bytes: &[u8], eocd: Eocd) -> Result<Vec<CentralRecord>> {
    let mut records = Vec::with_capacity(eocd.total_entries.min(MAX_PREALLOCATED_RECORDS));
    let mut offset = eocd.cd_offset;
    for index in 0..eocd.total_entries {
        let (record, next) = parse_central_record(bytes, offset, index)?;
        records.push(record);
        offset = next;
    }
    Ok(records)
}

/// Parses the central-directory record at `offset`, returning it and the offset
/// of the next record.
///
/// # Errors
///
/// Returns [`Error::validation`] naming the byte offset for a truncated or
/// malformed record, and [`Error::unsupported`] for `ZIP64` escape values.
fn parse_central_record(
    bytes: &[u8],
    offset: usize,
    index: usize,
) -> Result<(CentralRecord, usize)> {
    let header = slice_at(bytes, offset, CENTRAL_FIXED)?;
    if header.get(..4) != Some(CENTRAL_SIGNATURE) {
        return Err(Error::validation(format!(
            "invalid zip central-directory record {index} at byte {offset}: expected the \
             central-directory signature PK\x01\x02, found {:02x?}",
            &header[..4]
        )));
    }
    let flags = read_u16_le(header, 8)?;
    let method = read_u16_le(header, 10)?;
    let crc = read_u32_le(header, 16)?;
    let compressed = read_u32_le(header, 20)?;
    let declared = read_u32_le(header, 24)?;
    let name_len = usize::from(read_u16_le(header, 28)?);
    let extra_len = usize::from(read_u16_le(header, 30)?);
    let comment_len = usize::from(read_u16_le(header, 32)?);
    let external = read_u32_le(header, 38)?;
    let local_offset = read_u32_le(header, 42)?;

    if compressed == ZIP64_MARKER || declared == ZIP64_MARKER || local_offset == ZIP64_MARKER {
        return Err(zip64_error(format!(
            "central-directory record {index} at byte {offset} carries a 0xffffffff size or \
             local-header offset, the ZIP64 escape value"
        )));
    }
    let name_start = offset + CENTRAL_FIXED;
    let raw_name = slice_at(bytes, name_start, name_len)?;
    let name = decode_name(raw_name, flags, offset)?;
    let next = name_start
        .checked_add(name_len)
        .and_then(|value| value.checked_add(extra_len))
        .and_then(|value| value.checked_add(comment_len))
        .ok_or_else(|| {
            Error::validation(format!(
                "zip central-directory record {index} at byte {offset} overflows the archive \
                 length"
            ))
        })?;
    Ok((
        CentralRecord {
            name,
            flags,
            method,
            crc,
            compressed_size: compressed as usize,
            declared_size: declared as usize,
            external_attributes: external,
            local_offset: local_offset as usize,
        },
        next,
    ))
}

/// Decodes a stored entry name.
///
/// The UTF-8 flag (bit 11) demands strict UTF-8; without it a valid UTF-8 name is
/// kept as-is and anything else is decoded through CP437, the historic `zip`
/// code page.
fn decode_name(raw: &[u8], flags: u16, offset: usize) -> Result<String> {
    if flags & FLAG_UTF8 != 0 {
        return std::str::from_utf8(raw)
            .map(str::to_owned)
            .map_err(|error| {
                Error::validation(format!(
                    "invalid zip entry name in the central-directory record at byte {offset}: the \
                     UTF-8 flag (bit 11) is set but the name is not valid UTF-8 ({error})"
                ))
            });
    }
    match std::str::from_utf8(raw) {
        Ok(name) => Ok(name.to_owned()),
        Err(_) => Ok(decode_cp437(raw)),
    }
}

/// The code page 437 glyphs for `0x80..=0xFF`, the `zip` fallback encoding.
const CP437_HIGH: [char; 128] = [
    // 0x80..=0x87
    'Ç', 'ü', 'é', 'â', 'ä', 'à', 'å', 'ç', // 0x88..=0x8f
    'ê', 'ë', 'è', 'ï', 'î', 'ì', 'Ä', 'Å', // 0x90..=0x97
    'É', 'æ', 'Æ', 'ô', 'ö', 'ò', 'û', 'ù', // 0x98..=0x9f
    'ÿ', 'Ö', 'Ü', '¢', '£', '¥', '₧', 'ƒ', // 0xa0..=0xa7
    'á', 'í', 'ó', 'ú', 'ñ', 'Ñ', 'ª', 'º', // 0xa8..=0xaf
    '¿', '⌐', '¬', '½', '¼', '¡', '«', '»', // 0xb0..=0xb7
    '░', '▒', '▓', '│', '┤', '╡', '╢', '╖', // 0xb8..=0xbf
    '╕', '╣', '║', '╗', '╝', '╜', '╛', '┐', // 0xc0..=0xc7
    '└', '┴', '┬', '├', '─', '┼', '╞', '╟', // 0xc8..=0xcf
    '╚', '╔', '╩', '╦', '╠', '═', '╬', '╧', // 0xd0..=0xd7
    '╨', '╤', '╥', '╙', '╘', '╒', '╓', '╫', // 0xd8..=0xdf
    '╪', '┘', '┌', '█', '▄', '▌', '▐', '▀', // 0xe0..=0xe7
    'α', 'ß', 'Γ', 'π', 'Σ', 'σ', 'µ', 'τ', // 0xe8..=0xef
    'Φ', 'Θ', 'Ω', 'δ', '∞', 'φ', 'ε', '∩', // 0xf0..=0xf7
    '≡', '±', '≥', '≤', '⌠', '⌡', '÷', '≈', // 0xf8..=0xff
    '°', '∙', '·', '√', 'ⁿ', '²', '■', '\u{a0}',
];

/// Decodes bytes as CP437: ASCII is itself, `0x80..=0xFF` uses the upper table.
fn decode_cp437(raw: &[u8]) -> String {
    raw.iter()
        .map(|&byte| {
            if byte < 0x80 {
                char::from(byte)
            } else {
                CP437_HIGH[usize::from(byte - 0x80)]
            }
        })
        .collect()
}

/// Whether the record describes a directory.
fn is_directory(record: &CentralRecord) -> bool {
    record.name.ends_with('/') || record.external_attributes & DOS_DIRECTORY_ATTRIBUTE != 0
}

/// Extracts one entry, verifying its path, sizes, and CRC-32.
///
/// # Errors
///
/// Returns [`Error::validation`] for an unsafe path, a truncated local header or
/// payload, a size mismatch, a CRC-32 mismatch, or an entry above the caps, and
/// [`Error::unsupported`] for encrypted entries and unknown methods.
fn read_entry(bytes: &[u8], record: &CentralRecord, caps: &mut CapTracker) -> Result<ArchiveEntry> {
    let path = validate_path(&record.name)?;
    if record.flags & FLAG_ENCRYPTED != 0 {
        return Err(Error::unsupported(format!(
            "encrypted zip entry {path:?}: decrypting entries is not supported"
        )));
    }
    if record.method != METHOD_STORED && record.method != METHOD_DEFLATE {
        return Err(Error::unsupported(format!(
            "zip entry {path:?} uses compression method {}: only stored (0) and DEFLATE (8) are \
             supported",
            record.method
        )));
    }
    if is_directory(record) {
        caps.charge(0, &format!("zip directory entry {path:?}"))?;
        return Ok(ArchiveEntry {
            path,
            bytes: Vec::new(),
            is_dir: true,
            kind: ArchiveEntryKind::Directory,
            ..ArchiveEntry::default()
        });
    }
    if record.compressed_size == 0 && record.declared_size > 0 {
        return Err(Error::validation(format!(
            "zip entry {path:?} declares {} uncompressed bytes in a zero-byte payload",
            record.declared_size
        )));
    }
    // Charge the declared size before decompressing, so a declared zip bomb never
    // reaches an allocation.
    caps.charge(record.declared_size, &format!("zip entry {path:?}"))?;

    let payload = payload_slice(bytes, record)?;
    let data = if record.method == METHOD_DEFLATE {
        let data = inflate(payload, record.declared_size).map_err(|error| {
            Error::validation(format!(
                "invalid DEFLATE data for zip entry {path:?}: {error}"
            ))
        })?;
        if data.len() != record.declared_size {
            return Err(size_mismatch(&path, data.len(), record.declared_size));
        }
        data
    } else {
        // Stored: the payload is the file, so a size mismatch is visible before
        // the copy that would allocate it.
        if payload.len() != record.declared_size {
            return Err(size_mismatch(&path, payload.len(), record.declared_size));
        }
        payload.to_vec()
    };
    let computed = crc32(&data);
    if computed != record.crc {
        return Err(Error::validation(format!(
            "zip entry {path:?} fails its CRC-32 check: the central directory declares \
             0x{declared:08x} but the payload hashes to 0x{computed:08x}",
            declared = record.crc
        )));
    }
    Ok(ArchiveEntry {
        path,
        bytes: data,
        is_dir: false,
        kind: ArchiveEntryKind::File,
        ..ArchiveEntry::default()
    })
}

/// Builds the validation error for a payload whose length contradicts the
/// central directory.
fn size_mismatch(path: &str, actual: usize, declared: usize) -> Error {
    Error::validation(format!(
        "zip entry {path:?} declares {declared} uncompressed bytes but its payload holds \
         {actual} bytes"
    ))
}

/// Returns the compressed payload slice named by the record's local header.
///
/// Only the local header's variable-field lengths are used: its sizes are
/// ignored because a streamed entry keeps them zero and writes a data descriptor
/// instead, so the central directory stays authoritative.
///
/// # Errors
///
/// Returns [`Error::validation`] naming the byte offset for a missing local
/// header signature, a truncated header, or a truncated payload.
fn payload_slice<'a>(bytes: &'a [u8], record: &CentralRecord) -> Result<&'a [u8]> {
    let offset = record.local_offset;
    let signature = slice_at(bytes, offset, 4)?;
    if signature != LOCAL_SIGNATURE {
        return Err(Error::validation(format!(
            "invalid zip local header for entry {:?} at byte {offset}: expected the signature \
             PK\x03\x04, found {signature:02x?}",
            record.name
        )));
    }
    let name_len = usize::from(read_u16_le(bytes, offset + 26)?);
    let extra_len = usize::from(read_u16_le(bytes, offset + 28)?);
    let data_start = offset
        .checked_add(LOCAL_FIXED)
        .and_then(|value| value.checked_add(name_len))
        .and_then(|value| value.checked_add(extra_len))
        .ok_or_else(|| {
            Error::validation(format!(
                "zip local header for entry {:?} at byte {offset} overflows the archive length",
                record.name
            ))
        })?;
    slice_at(bytes, data_start, record.compressed_size)
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
    /// The bytes are deterministic for a fixed input: stored compression, the
    /// fixed DOS timestamp `1980-01-01 00:00:00`, no extra fields, no comments,
    /// entries in the order given, and the UTF-8 flag (bit 11) set only for a
    /// non-ASCII path. Every path is validated with `validate_path` first, so
    /// what this writes the reader accepts.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] for an empty entry list, a directory entry
    /// with content, a file entry whose path ends in `/`, or an unsafe path.
    pub fn write_stored(entries: &[ArchiveEntry]) -> Result<Vec<u8>> {
        let staged = stage_entries(entries)?;
        let mut out = Vec::new();
        let directory = write_local_records(&mut out, &staged)?;
        write_central_directory(&mut out, &directory)?;
        Ok(out)
    }
}

/// One entry as staged for the central directory.
#[derive(Debug, Clone)]
struct StagedEntry {
    /// Entry name, echoed into the central directory.
    name: String,
    /// General-purpose flags, kept identical in both headers.
    flags: u16,
    /// CRC-32 of the payload.
    crc: u32,
    /// Payload size, identical compressed and uncompressed because the writer is
    /// stored-only.
    size: u32,
    /// Whether the entry is a directory.
    is_dir: bool,
    /// Offset of the entry's local file header.
    offset: u32,
}

/// Validates the entries and pairs each borrowed path with its entry.
///
/// # Errors
///
/// Returns [`Error::validation`] for an empty list, more entries than a classic
/// archive can count, an unsafe path, a directory with content, or a file path
/// that ends in `/`.
fn stage_entries(entries: &[ArchiveEntry]) -> Result<Vec<(&str, &ArchiveEntry)>> {
    if entries.is_empty() {
        return Err(Error::validation(
            "cannot write a zip archive with no entries",
        ));
    }
    if entries.len() > usize::from(u16::MAX) {
        return Err(Error::validation(format!(
            "cannot write a zip archive with {} entries: a classic archive counts at most {} and \
             ZIP64 is not implemented",
            entries.len(),
            u16::MAX
        )));
    }
    let mut staged = Vec::with_capacity(entries.len());
    for (index, entry) in entries.iter().enumerate() {
        let path = validate_path(&entry.path)?;
        if entry.is_dir {
            if !entry.bytes.is_empty() {
                return Err(Error::validation(format!(
                    "zip entry {index} {path:?} is a directory but carries {} bytes of content",
                    entry.bytes.len()
                )));
            }
        } else if path.ends_with('/') {
            return Err(Error::validation(format!(
                "zip entry {index} {path:?} is a file but its path ends with `/`"
            )));
        }
        staged.push((entry.path.as_str(), entry));
    }
    Ok(staged)
}

/// Writes every local header and payload, returning the central-directory plan.
///
/// # Errors
///
/// Returns [`Error::validation`] when an offset, a size, or a name length cannot
/// be addressed by the 32-bit `zip` fields.
fn write_local_records(
    out: &mut Vec<u8>,
    staged: &[(&str, &ArchiveEntry)],
) -> Result<Vec<StagedEntry>> {
    let mut directory = Vec::with_capacity(staged.len());
    for &(name, entry) in staged {
        let offset = to_u32(out.len(), "zip local-header offset")?;
        let size = to_u32(
            entry.bytes.len(),
            &format!("the size of zip entry {name:?}"),
        )?;
        let crc = crc32(&entry.bytes);
        let flags = if name.is_ascii() { 0 } else { FLAG_UTF8 };
        write_local_header(out, name, size, crc, flags)?;
        out.extend_from_slice(&entry.bytes);
        directory.push(StagedEntry {
            name: name.to_owned(),
            flags,
            crc,
            size,
            is_dir: entry.is_dir,
            offset,
        });
    }
    Ok(directory)
}

/// Writes one local file header.
///
/// # Errors
///
/// Returns [`Error::validation`] when `name` is longer than a `zip` name field
/// can hold.
fn write_local_header(
    out: &mut Vec<u8>,
    name: &str,
    size: u32,
    crc: u32,
    flags: u16,
) -> Result<()> {
    out.extend_from_slice(LOCAL_SIGNATURE);
    put_u16(out, VERSION);
    put_u16(out, flags);
    put_u16(out, METHOD_STORED);
    put_u16(out, DOS_TIME);
    put_u16(out, DOS_DATE);
    put_u32(out, crc);
    put_u32(out, size);
    put_u32(out, size);
    put_u16(out, name_length(name)?);
    put_u16(out, 0);
    out.extend_from_slice(name.as_bytes());
    Ok(())
}

/// Writes the central directory and its end-of-central-directory record.
///
/// # Errors
///
/// Returns [`Error::validation`] when the directory offset, size, entry count, or
/// a name length cannot be addressed by the corresponding `zip` field.
fn write_central_directory(out: &mut Vec<u8>, directory: &[StagedEntry]) -> Result<()> {
    let cd_start = out.len();
    for entry in directory {
        out.extend_from_slice(CENTRAL_SIGNATURE);
        put_u16(out, VERSION);
        put_u16(out, VERSION);
        put_u16(out, entry.flags);
        put_u16(out, METHOD_STORED);
        put_u16(out, DOS_TIME);
        put_u16(out, DOS_DATE);
        put_u32(out, entry.crc);
        put_u32(out, entry.size);
        put_u32(out, entry.size);
        put_u16(out, name_length(&entry.name)?);
        put_u16(out, 0);
        put_u16(out, 0);
        put_u16(out, 0);
        put_u16(out, 0);
        put_u32(out, external_attributes(entry.is_dir));
        put_u32(out, entry.offset);
        out.extend_from_slice(entry.name.as_bytes());
    }
    let cd_offset = to_u32(cd_start, "zip central-directory offset")?;
    let cd_size = to_u32(out.len() - cd_start, "zip central-directory size")?;
    write_eocd(out, directory.len(), cd_size, cd_offset)
}

/// Writes the end-of-central-directory record for `count` entries.
///
/// # Errors
///
/// Returns [`Error::validation`] when `count` does not fit the 16-bit count
/// fields, which a classic archive cannot express.
fn write_eocd(out: &mut Vec<u8>, count: usize, cd_size: u32, cd_offset: u32) -> Result<()> {
    let count = u16::try_from(count).map_err(|_| {
        Error::validation(format!(
            "cannot write a zip archive with {count} entries: a classic archive counts at most {} \
             and ZIP64 is not implemented",
            u16::MAX
        ))
    })?;
    out.extend_from_slice(EOCD_SIGNATURE);
    put_u16(out, 0);
    put_u16(out, 0);
    put_u16(out, count);
    put_u16(out, count);
    put_u32(out, cd_size);
    put_u32(out, cd_offset);
    put_u16(out, 0);
    Ok(())
}

/// Builds the external attributes for an entry, pairing the Unix mode in the high
/// half with the DOS directory bit in the low byte.
fn external_attributes(is_dir: bool) -> u32 {
    if is_dir {
        (UNIX_DIRECTORY_MODE << 16) | DOS_DIRECTORY_ATTRIBUTE
    } else {
        UNIX_FILE_MODE << 16
    }
}

/// Length of a `zip` name field for `name`.
///
/// # Errors
///
/// Returns [`Error::validation`] when `name` is longer than 65535 bytes.
fn name_length(name: &str) -> Result<u16> {
    u16::try_from(name.len()).map_err(|_| {
        Error::validation(format!(
            "zip entry name {name:?} is {} bytes long, above the 65535-byte zip name limit",
            name.len()
        ))
    })
}

/// Narrows `value` to a `zip` 32-bit field.
///
/// # Errors
///
/// Returns [`Error::validation`] naming `what` when `value` does not fit.
fn to_u32(value: usize, what: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| {
        Error::validation(format!(
            "cannot write a zip archive: {what} needs {value} bytes, above the 4294967295-byte \
             classic zip limit"
        ))
    })
}

/// Appends a little-endian `u16`.
fn put_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

/// Appends a little-endian `u32`.
fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

#[cfg(test)]
mod tests;
