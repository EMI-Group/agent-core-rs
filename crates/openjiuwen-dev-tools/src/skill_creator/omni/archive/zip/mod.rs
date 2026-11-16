//! `zip` reading (stored and `DEFLATE`, classic and `ZIP64`) and deterministic
//! stored writing.
//!
//! The reader implements the `zip` container (PKWARE APPNOTE 6.3) on top of the
//! shared in-crate helpers, with no compression dependency:
//!
//! - [`find_eocd`] scans backwards for the end-of-central-directory record
//!   (`PK\x05\x06`) inside the legal `22 + 65535` byte comment window and
//!   prefers the rightmost matching record. When a `ZIP64` locator
//!   (`PK\x06\x07`) directly precedes it, the `ZIP64` end record (`PK\x06\x06`)
//!   it names supplies the 64-bit entry count, central-directory offset, and
//!   central-directory size; a `ZIP64` record sitting directly before the end
//!   record without a locator is tolerated as well. The classic record's escape
//!   values (`0xffff` entries, `0xffff_ffff` sizes and offsets) are replaced by
//!   those 64-bit values, and an escape value with no `ZIP64` structure behind it
//!   is a validation error naming the absent record;
//! - [`parse_central_directory`] walks the `PK\x01\x02` records, which are
//!   authoritative: local-header sizes are never trusted because streamed
//!   archives legally leave them zero and carry a trailing data descriptor
//!   (general-purpose flag bit 3). A per-entry `ZIP64` extra field (header id
//!   `0x0001`) supplies the sizes and local-header offset of a record that
//!   escaped them, in the `ZIP64` order, and a digital-signature record
//!   (`PK\x06\x05`) after the last entry is skipped;
//! - [`read_entry`] validates the local header (`PK\x03\x04`), slices the
//!   compressed payload, inflates it when needed, and verifies the declared
//!   uncompressed size and the CRC-32 of the result.
//!
//! Supported compression methods are stored (`0`) and `DEFLATE` (`8`, decoded
//! by [`inflate`](super::inflate)).
//!
//! Traditional PKWARE encryption (`ZipCrypto`, general-purpose flag bit 0) is
//! decrypted by the `zipcrypto` module when the caller supplies a password
//! through [`UnpackOptions`](super::UnpackOptions): the 12-byte encryption
//! header is decrypted and its password check byte verified first, then the
//! decrypted remainder is decompressed, so the declared sizes, the CRC-32 check,
//! and the [`CapTracker`] charge all apply to the real payload. An encrypted
//! entry read without a password is a [`Error::validation`] naming the entry and
//! asking for [`UnpackOptions::with_password`](super::UnpackOptions::with_password).
//!
//! Three other encryption schemes are typed [`Error::unsupported`], each naming
//! itself: `WinZip` AES (compression method `99` or extra field `0x9901`), PKWARE
//! strong encryption (flag bit 6), and an encrypted central directory (flag bit
//! 13). Any other encryption marker and any other compression method (bzip2,
//! `LZMA`, zstd, ...) is refused the same way. A 64-bit `ZIP64` size or offset
//! wider than this platform's `usize` is the only `ZIP64` refusal left.
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
//! to the caps. A symlink entry (Unix file type `0o120000` in the high half of
//! the external attributes) carries its target in `link_target` and no bytes. Its
//! target is not path-validated — a relative link that climbs out of the bundle
//! is still only a string in a returned record — while the entry path itself is.
//! The Unix mode found in that high half is reported in `mode` for every entry
//! that declares one. Every other entry is charged its declared uncompressed size
//! to [`CapTracker`] before decompressing, so a zip bomb fails with a validation
//! error instead of allocating.
//!
//! [`ArchiveWriter::write_stored`] is the deterministic counterpart used by
//! tests and by callers that need a round trip: stored-only, a fixed DOS
//! timestamp of `1980-01-01 00:00:00` (`time = 0x0000`, `date = 0x0021`), no
//! extras, no comments, and entries in the order given. It writes the Unix mode
//! of each entry kind into the external attributes (`0o120777` for a symlink,
//! `0o040755` for a directory, `0o100644` otherwise), so a symlink entry round
//! trips through [`unpack`] with its `kind` and `link_target` intact.

mod payload;
mod writer;
mod zip64;
mod zipcrypto;

pub use writer::ArchiveWriter;

use super::{
    read_u16_le, read_u32_le, slice_at, validate_path, ArchiveEntry, ArchiveEntryKind, CapTracker,
};
use openjiuwen_common::error::{Error, Result};
use payload::decode_payload;
use zip64::{Zip64Escapes, ZIP64_MARKER};

/// Signature of a local file header.
const LOCAL_SIGNATURE: &[u8] = b"PK\x03\x04";
/// Signature of a central-directory file header.
const CENTRAL_SIGNATURE: &[u8] = b"PK\x01\x02";
/// Signature of the end-of-central-directory record.
const EOCD_SIGNATURE: &[u8] = b"PK\x05\x06";
/// Fixed size of the end-of-central-directory record, before its comment.
const EOCD_FIXED: usize = 22;
/// Fixed size of a central-directory record, before its variable fields.
const CENTRAL_FIXED: usize = 46;
/// Fixed size of a local file header, before its variable fields.
const LOCAL_FIXED: usize = 30;
/// Longest `zip` comment, in bytes, that hides a preceding record.
const MAX_COMMENT_LEN: usize = u16::MAX as usize;
/// General-purpose flag bit 0: the entry is encrypted.
const FLAG_ENCRYPTED: u16 = 0x0001;
/// General-purpose flag bit 3: the entry's sizes and CRC-32 follow it as a data
/// descriptor, so its local header declares none.
const FLAG_DATA_DESCRIPTOR: u16 = 0x0008;
/// General-purpose flag bit 6: the entry uses PKWARE strong encryption.
const FLAG_STRONG_ENCRYPTION: u16 = 0x0040;
/// General-purpose flag bit 11: the entry name is UTF-8.
const FLAG_UTF8: u16 = 0x0800;
/// General-purpose flag bit 13: the central directory is encrypted, so the local
/// header's values are masked.
const FLAG_MASKED_LOCAL_HEADERS: u16 = 0x2000;
/// Compression method 0: stored.
const METHOD_STORED: u16 = 0;
/// Compression method 8: `DEFLATE`.
const METHOD_DEFLATE: u16 = 8;
/// Compression method 99: `WinZip` AES, whose payload this reader cannot decrypt.
const METHOD_AES: u16 = 99;
/// The DOS directory attribute, the low byte of the external attributes.
const DOS_DIRECTORY_ATTRIBUTE: u32 = 0x10;
/// The mask of the Unix file-type bits inside the external attributes' high half.
const UNIX_FILE_TYPE_MASK: u32 = 0o170_000;
/// The Unix file type of a symbolic link (`S_IFLNK`).
const UNIX_SYMLINK_TYPE: u32 = 0o120_000;
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
/// Unix mode bits of a symbolic link (`0o120777`).
const UNIX_SYMLINK_MODE: u32 = 0o120_777;
/// Upper bound on the records pre-allocated from the declared entry count.
const MAX_PREALLOCATED_RECORDS: usize = 1024;

/// Unpacks a `zip` archive that carries no encrypted entries.
///
/// The no-password shorthand for [`unpack_with_password`].
///
/// # Errors
///
/// Returns [`Error::validation`] for malformed, truncated, unsafe, or oversized
/// archives (including an escape value with no `ZIP64` structure behind it) and
/// for a `ZipCrypto` entry, which needs a password, and
/// [`Error::unsupported`] for a compression method other than stored and
/// `DEFLATE`, a `ZIP64` value wider than this platform's `usize`, and the
/// encryption schemes this reader does not implement (`WinZip` AES, PKWARE strong
/// encryption, an encrypted central directory).
pub(crate) fn unpack(bytes: &[u8]) -> Result<Vec<ArchiveEntry>> {
    unpack_with_password(bytes, None)
}

/// Unpacks a `zip` archive into entries, in central-directory order, decrypting
/// traditional PKWARE (`ZipCrypto`) entries with `password`.
///
/// # Errors
///
/// Returns [`Error::validation`] for malformed, truncated, unsafe, or oversized
/// archives, for a `ZipCrypto` entry when `password` is `None` or does not match
/// the entry's check byte, and for a payload shorter than its encryption header;
/// and [`Error::unsupported`] for a compression method other than stored and
/// `DEFLATE`, a `ZIP64` value wider than this platform's `usize`, and the
/// encryption schemes this reader does not implement (`WinZip` AES, PKWARE strong
/// encryption, an encrypted central directory).
pub(crate) fn unpack_with_password(
    bytes: &[u8],
    password: Option<&str>,
) -> Result<Vec<ArchiveEntry>> {
    let eocd = find_eocd(bytes)?;
    let records = parse_central_directory(bytes, eocd)?;
    let mut caps = CapTracker::new();
    let mut entries = Vec::with_capacity(records.len());
    for record in &records {
        entries.push(read_entry(bytes, record, &mut caps, password)?);
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
    /// Byte offset of the end-of-central-directory record itself, which bounds
    /// the directory region.
    record_offset: usize,
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
    /// Whether the record's extra data carries the `WinZip` AES field (`0x9901`),
    /// which marks an entry traditional `ZipCrypto` cannot decrypt.
    aes_extra: bool,
    /// CRC-32 of the uncompressed payload.
    crc: u32,
    /// Size of the compressed payload in the archive.
    compressed_size: usize,
    /// Uncompressed size declared by the central directory.
    declared_size: usize,
    /// External attributes; the low byte holds the DOS attribute flags and the
    /// high half the Unix mode when the writer declared one.
    external_attributes: u32,
    /// Offset of the entry's local file header.
    local_offset: usize,
}

/// Finds and parses the rightmost end-of-central-directory record, resolving the
/// `ZIP64` values its escape fields stand for.
///
/// # Errors
///
/// Returns [`Error::validation`] when no record sits in the scan window, the
/// directory it points at runs past the input, a `ZIP64` record or locator is
/// truncated or inconsistent, an escape value has no `ZIP64` structure behind
/// it, or the archive spans several disks.
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

    let (total_entries, cd_offset, cd_size) =
        if let Some(directory) = zip64::find_directory(bytes, offset)? {
            (
                directory.total_entries,
                directory.cd_offset,
                directory.cd_size,
            )
        } else {
            classic_directory(offset, total_entries, cd_size, cd_offset)?
        };

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
        total_entries,
        cd_offset,
        record_offset: offset,
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

/// Builds the validation error for an escape value that no `ZIP64` structure
/// backs.
fn missing_zip64(eocd: usize, field: &str) -> Error {
    Error::validation(format!(
        "invalid zip archive: the end-of-central-directory record at byte {eocd} holds the ZIP64 \
         escape value in its {field} field, but no ZIP64 end-of-central-directory record \
         (PK\x06\x06) or locator (PK\x06\x07) precedes it"
    ))
}

/// Resolves the classic (32-bit) end-of-central-directory fields into a directory
/// location, refusing an escape value that no `ZIP64` structure backs.
///
/// # Errors
///
/// Returns [`Error::validation`] naming the escaped field when the record holds
/// the `ZIP64` escape value but neither a `ZIP64` end-of-central-directory record
/// nor a locator precedes it.
fn classic_directory(
    eocd: usize,
    total_entries: u16,
    cd_size: u32,
    cd_offset: u32,
) -> Result<(usize, usize, usize)> {
    if total_entries == u16::MAX {
        return Err(missing_zip64(eocd, "total-entry count"));
    }
    if cd_size == ZIP64_MARKER {
        return Err(missing_zip64(eocd, "central-directory size"));
    }
    if cd_offset == ZIP64_MARKER {
        return Err(missing_zip64(eocd, "central-directory offset"));
    }
    Ok((
        usize::from(total_entries),
        cd_offset as usize,
        cd_size as usize,
    ))
}

///
/// A digital-signature record directly after the last entry is skipped, so an
/// archive that carries one still reads.
///
/// # Errors
///
/// Returns [`Error::validation`] for a truncated or malformed record, for an
/// extra field that is short of a value its record escaped, and for a directory
/// that runs past the end-of-central-directory record.
fn parse_central_directory(bytes: &[u8], eocd: Eocd) -> Result<Vec<CentralRecord>> {
    let mut records = Vec::with_capacity(eocd.total_entries.min(MAX_PREALLOCATED_RECORDS));
    let mut offset = eocd.cd_offset;
    for index in 0..eocd.total_entries {
        let (record, next) = parse_central_record(bytes, offset, index)?;
        records.push(record);
        offset = next;
    }
    // The archive digital-signature record legally follows the central directory.
    let end = zip64::skip_digital_signature(bytes, offset);
    if end > eocd.record_offset {
        return Err(Error::validation(format!(
            "invalid zip archive: the central directory ends at byte {end}, past the \
             end-of-central-directory record at byte {}",
            eocd.record_offset
        )));
    }
    Ok(records)
}

/// Parses the central-directory record at `offset`, returning it and the offset
/// of the next record.
///
/// # Errors
///
/// Returns [`Error::validation`] naming the byte offset for a truncated or
/// malformed record, for an escape value without a `ZIP64` extra-field value
/// behind it, and for an extra field that ends early.
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

    let name_start = offset + CENTRAL_FIXED;
    let raw_name = slice_at(bytes, name_start, name_len)?;
    let name = decode_name(raw_name, flags, offset)?;
    let extra_start = name_start.checked_add(name_len).ok_or_else(|| {
        Error::validation(format!(
            "zip central-directory record {index} at byte {offset} overflows the archive length"
        ))
    })?;
    let escapes = Zip64Escapes {
        declared: declared == ZIP64_MARKER,
        compressed: compressed == ZIP64_MARKER,
        local_offset: local_offset == ZIP64_MARKER,
    };
    let aes_extra = zipcrypto::has_aes_extra(bytes, extra_start, extra_len);
    let extra = zip64::parse_extra(bytes, extra_start, extra_len, escapes, &name, offset)?;
    let compressed_size = zip64::resolve(
        compressed,
        extra.compressed,
        "compressed size",
        &name,
        offset,
    )?;
    let declared_size =
        zip64::resolve(declared, extra.declared, "uncompressed size", &name, offset)?;
    let local_offset = zip64::resolve(
        local_offset,
        extra.local_offset,
        "local-header offset",
        &name,
        offset,
    )?;

    let next = extra_start
        .checked_add(extra_len)
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
            aes_extra,
            crc,
            compressed_size,
            declared_size,
            external_attributes: external,
            local_offset,
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

/// The Unix mode an entry declares, or `None` when the archive stores none.
///
/// The high half of the external attributes carries the mode only for archives
/// written on a Unix host; a `zip` written by a DOS writer leaves it zero, which
/// is why a zero high half reports no mode at all.
fn unix_mode(external_attributes: u32) -> Option<u32> {
    let mode = external_attributes >> 16;
    (mode != 0).then_some(mode)
}

/// Whether a Unix mode names a symbolic link.
fn is_symlink(mode: u32) -> bool {
    mode & UNIX_FILE_TYPE_MASK == UNIX_SYMLINK_TYPE
}

/// Extracts one entry, verifying its path, sizes, and CRC-32.
///
/// # Errors
///
/// Returns [`Error::validation`] for an unsafe path, a truncated local header or
/// payload, an encrypted entry with no password or with a password that fails its
/// check byte, a size mismatch, a CRC-32 mismatch, or an entry above the caps, and
/// [`Error::unsupported`] for a compression method this reader cannot decode, an
/// encryption scheme beyond traditional `ZipCrypto`, and a `ZIP64` value this
/// platform cannot address.
fn read_entry(
    bytes: &[u8],
    record: &CentralRecord,
    caps: &mut CapTracker,
    password: Option<&str>,
) -> Result<ArchiveEntry> {
    let path = validate_path(&record.name)?;
    let mode = unix_mode(record.external_attributes);
    if record.flags & FLAG_STRONG_ENCRYPTION != 0 {
        return Err(Error::unsupported(format!(
            "zip entry {path:?} uses PKWARE strong encryption (general-purpose flag bit 6), which \
             is not supported: only traditional ZipCrypto entries can be decrypted"
        )));
    }
    if record.flags & FLAG_MASKED_LOCAL_HEADERS != 0 {
        return Err(Error::unsupported(format!(
            "zip entry {path:?} belongs to an encrypted central directory (general-purpose flag \
             bit 13), which is not supported: only traditional ZipCrypto entries can be decrypted"
        )));
    }
    if record.method == METHOD_AES || record.aes_extra {
        return Err(Error::unsupported(format!(
            "zip entry {path:?} uses WinZip AES encryption (compression method 99 or extra field \
             0x9901), which is not supported: only traditional ZipCrypto entries can be decrypted"
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
            link_target: None,
            mode,
        });
    }
    let encrypted = record.flags & FLAG_ENCRYPTED != 0;
    if encrypted && password.is_none() {
        return Err(Error::validation(format!(
            "zip entry {path:?} is encrypted with traditional ZipCrypto: pass the password with \
             UnpackOptions::with_password (or ArchiveUnpacker::unpack_with_options) to decrypt it"
        )));
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
    let data = decode_payload(bytes, record, &path, password)?;

    if mode.is_some_and(is_symlink) {
        // The stored payload of a symlink entry is its target. The lossy decode
        // can expand every invalid byte to the three bytes of U+FFFD, so the
        // growth is charged to the caps as well.
        let target = String::from_utf8_lossy(&data).into_owned();
        if target.len() > record.declared_size {
            caps.charge(
                target.len() - record.declared_size,
                &format!("the symlink target of zip entry {path:?}"),
            )?;
        }
        return Ok(ArchiveEntry {
            path,
            bytes: Vec::new(),
            is_dir: false,
            kind: ArchiveEntryKind::Symlink,
            link_target: Some(target),
            mode,
        });
    }
    Ok(ArchiveEntry {
        path,
        bytes: data,
        is_dir: false,
        kind: ArchiveEntryKind::File,
        link_target: None,
        mode,
    })
}

#[cfg(test)]
mod tests;
