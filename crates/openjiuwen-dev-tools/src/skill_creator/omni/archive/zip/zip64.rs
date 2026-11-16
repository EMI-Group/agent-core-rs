//! `ZIP64` records, the end-of-central-directory locator, and the `ZIP64`
//! extra field, decoded on top of the same bounds-checked primitive readers as
//! the classic layout in [`super`].
//!
//! `ZIP64` (PKWARE APPNOTE 6.3 §4.3.14–4.3.15) extends a `zip` archive in three
//! places, and each one is parsed here:
//!
//! - the `ZIP64` end-of-central-directory record (`PK\x06\x06`), whose 44 fixed
//!   bytes after the 64-bit declared size hold the disk numbers, the entry
//!   counts, and the central-directory size and offset as 64-bit values; a
//!   declared size above 44 means an extensible data sector that is skipped;
//! - the `ZIP64` end-of-central-directory locator (`PK\x06\x07`), 20 bytes that
//!   name the record's 64-bit offset, which is what [`find_directory`] follows
//!   when one directly precedes the classic end record;
//! - the per-entry `ZIP64` extra field (header id `0x0001`) that supplies the
//!   values a central-directory record escaped with `0xffff_ffff`
//!   ([`parse_extra`]).
//!
//! The classic record's escape values (`0xffff` entries, `0xffff_ffff` sizes and
//! offsets) are resolved against those structures by [`super::find_eocd`]; a
//! surviving escape value without any `ZIP64` structure is a validation error
//! naming the absent record, because guessing a size would be unsound.
//!
//! A digital-signature record (`PK\x06\x05`) may legally follow the central
//! directory, so [`skip_digital_signature`] steps over one instead of letting its
//! signature be read as another directory record.
//!
//! Every structure is read through [`slice_at`], so a truncated or hostile
//! archive fails with a validation error naming the byte offset instead of
//! panicking.

use super::super::{read_u16_le, read_u32_le, slice_at};
use super::CENTRAL_FIXED;
use openjiuwen_common::error::{Error, Result};

/// Signature of the `ZIP64` end-of-central-directory record.
pub(crate) const ZIP64_EOCD_SIGNATURE: &[u8] = b"PK\x06\x06";
/// Signature of the `ZIP64` end-of-central-directory locator.
pub(crate) const ZIP64_LOCATOR_SIGNATURE: &[u8] = b"PK\x06\x07";
/// Signature of the archive digital-signature record.
pub(crate) const DIGITAL_SIGNATURE: &[u8] = b"PK\x06\x05";
/// Fixed size of the `ZIP64` end-of-central-directory record, before its
/// extensible data sector.
pub(crate) const ZIP64_EOCD_FIXED: usize = 56;
/// Fixed size of the `ZIP64` end-of-central-directory locator.
pub(crate) const ZIP64_LOCATOR_FIXED: usize = 20;
/// Bytes of fixed fields a `ZIP64` end record declares after its size field.
pub(crate) const ZIP64_EOCD_MIN_REMAINING: u64 = 44;
/// Header id of the per-entry `ZIP64` extra field.
pub(crate) const ZIP64_EXTRA_ID: u16 = 0x0001;
/// The escape value `ZIP64` uses in a 32-bit size or offset field.
pub(crate) const ZIP64_MARKER: u32 = 0xFFFF_FFFF;
/// Fixed part of the archive digital-signature record: signature plus a 16-bit
/// length.
const DIGITAL_SIGNATURE_FIXED: usize = 6;
/// The bytes a `ZIP64` end record uses before its declared size field.
const ZIP64_EOCD_HEADER: usize = 12;

/// The central-directory location a `ZIP64` end record declares.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Zip64Directory {
    /// Total number of central-directory records.
    pub(crate) total_entries: usize,
    /// Where the central directory starts.
    pub(crate) cd_offset: usize,
    /// Size of the central directory in bytes.
    pub(crate) cd_size: usize,
}

/// Finds and parses the `ZIP64` end-of-central-directory record of an archive.
///
/// A `PK\x06\x07` locator immediately before the classic end record is followed
/// to the record's declared 64-bit offset; without a locator, a record that sits
/// immediately before the end record and whose declared size lands exactly on it
/// is accepted as the tolerated layout some writers emit.
///
/// # Errors
///
/// Returns [`Error::validation`] when the locator or record is truncated, points
/// outside the input, or names a second disk. Returns `Ok(None)` when the archive
/// carries no `ZIP64` structure at all.
pub(crate) fn find_directory(bytes: &[u8], eocd_offset: usize) -> Result<Option<Zip64Directory>> {
    if let Some(locator_offset) = locator_offset(bytes, eocd_offset) {
        let record_offset = read_locator(bytes, locator_offset)?;
        return read_eocd(bytes, record_offset, Some(locator_offset)).map(Some);
    }
    match record_before(bytes, eocd_offset) {
        Some(record_offset) => read_eocd(bytes, record_offset, Some(eocd_offset)).map(Some),
        None => Ok(None),
    }
}

/// The offset of a `ZIP64` locator that directly precedes the end record.
fn locator_offset(bytes: &[u8], eocd_offset: usize) -> Option<usize> {
    let start = eocd_offset.checked_sub(ZIP64_LOCATOR_FIXED)?;
    has_signature(bytes, start, ZIP64_LOCATOR_SIGNATURE).then_some(start)
}

/// The offset of a `ZIP64` end record that sits immediately before the end
/// record without a locator, when its declared size lands exactly on the end
/// record.
fn record_before(bytes: &[u8], eocd_offset: usize) -> Option<usize> {
    let start = eocd_offset.checked_sub(ZIP64_EOCD_FIXED)?;
    if !has_signature(bytes, start, ZIP64_EOCD_SIGNATURE) {
        return None;
    }
    let declared = read_u64_le(bytes, start + 4).ok()?;
    let end = record_end(start, declared).ok()?;
    (end == eocd_offset).then_some(start)
}

/// Whether `signature` starts at `offset`.
fn has_signature(bytes: &[u8], offset: usize, signature: &[u8]) -> bool {
    bytes
        .get(offset..)
        .is_some_and(|rest| rest.starts_with(signature))
}

/// Reads the `ZIP64` locator at `offset`, returning the record offset it names.
///
/// # Errors
///
/// Returns [`Error::validation`] for a missing signature, a truncated locator, a
/// multi-disk archive, or a record offset outside the input.
fn read_locator(bytes: &[u8], offset: usize) -> Result<usize> {
    let fixed = slice_at(bytes, offset, ZIP64_LOCATOR_FIXED)?;
    if fixed.get(..4) != Some(ZIP64_LOCATOR_SIGNATURE) {
        return Err(Error::validation(format!(
            "invalid ZIP64 end-of-central-directory locator at byte {offset}: expected the \
             signature PK\x06\x07, found {:02x?}",
            fixed.get(..4).unwrap_or_default()
        )));
    }
    let disk = read_u32_le(fixed, 4)?;
    let declared = read_u64_le(fixed, 8)?;
    let total_disks = read_u32_le(fixed, 16)?;
    if disk != 0 || total_disks != 1 {
        return Err(Error::validation(format!(
            "multi-disk zip archives are not supported: the ZIP64 locator at byte {offset} places \
             the ZIP64 end-of-central-directory record at byte {declared} of disk {disk} of \
             {total_disks}"
        )));
    }
    let record_offset = to_usize(
        declared,
        "the ZIP64 end-of-central-directory offset named by the locator",
    )?;
    if record_offset
        .checked_add(ZIP64_EOCD_FIXED)
        .is_none_or(|end| end > bytes.len())
    {
        return Err(Error::validation(format!(
            "invalid ZIP64 end-of-central-directory locator at byte {offset}: it points at byte \
             {record_offset}, but only {} bytes of the {}-byte input follow, fewer than the \
             {ZIP64_EOCD_FIXED} fixed bytes of the record",
            bytes.len().saturating_sub(record_offset),
            bytes.len()
        )));
    }
    Ok(record_offset)
}

/// Parses the `ZIP64` end-of-central-directory record at `offset`.
///
/// `expected_end` is the byte offset the record's declared size must land on: the
/// locator that names it, or the classic end record for the locator-free layout.
///
/// # Errors
///
/// Returns [`Error::validation`] for a missing signature, a declared size below
/// the 44 fixed bytes or past the input, a record whose declared size does not
/// end where the structure after it starts, a multi-disk archive, or a declared
/// entry count that could not fit the declared central directory.
fn read_eocd(bytes: &[u8], offset: usize, expected_end: Option<usize>) -> Result<Zip64Directory> {
    let fixed = slice_at(bytes, offset, ZIP64_EOCD_FIXED)?;
    if fixed.get(..4) != Some(ZIP64_EOCD_SIGNATURE) {
        return Err(Error::validation(format!(
            "invalid ZIP64 end-of-central-directory record at byte {offset}: expected the \
             signature PK\x06\x06, found {:02x?}",
            fixed.get(..4).unwrap_or_default()
        )));
    }
    let declared = read_u64_le(fixed, 4)?;
    if declared < ZIP64_EOCD_MIN_REMAINING {
        return Err(Error::validation(format!(
            "invalid ZIP64 end-of-central-directory record at byte {offset}: it declares \
             {declared} bytes after its size field, below the {ZIP64_EOCD_MIN_REMAINING} bytes of \
             fixed fields"
        )));
    }
    let record_end = record_end(offset, declared)?;
    if record_end > bytes.len() {
        return Err(Error::validation(format!(
            "truncated ZIP64 end-of-central-directory record at byte {offset}: it declares \
             {declared} bytes after its size field, ending at byte {record_end} of the {}-byte \
             input",
            bytes.len()
        )));
    }
    if let Some(expected) = expected_end {
        if record_end != expected {
            return Err(Error::validation(format!(
                "invalid ZIP64 end-of-central-directory record at byte {offset}: it declares {declared} \
                 bytes after its size field and ends at byte {record_end}, but the structure that \
                 follows it starts at byte {expected}"
            )));
        }
    }
    let disk = read_u32_le(fixed, 16)?;
    let cd_start_disk = read_u32_le(fixed, 20)?;
    let entries_on_disk = read_u64_le(fixed, 24)?;
    let total_entries = read_u64_le(fixed, 32)?;
    let cd_size = read_u64_le(fixed, 40)?;
    let cd_offset = read_u64_le(fixed, 48)?;
    if disk != 0 || cd_start_disk != 0 || entries_on_disk != total_entries {
        return Err(Error::validation(format!(
            "multi-disk zip archives are not supported: the ZIP64 end-of-central-directory record \
             at byte {offset} names disk {disk}, central-directory start disk {cd_start_disk}, and \
             {entries_on_disk} of {total_entries} entries on this disk"
        )));
    }
    let total_entries = to_usize(total_entries, "the ZIP64 total entry count")?;
    let cd_size = to_usize(cd_size, "the ZIP64 central-directory size")?;
    let cd_offset = to_usize(cd_offset, "the ZIP64 central-directory offset")?;
    // Every central-directory record is at least 46 bytes, so a declared count
    // that could not fit the declared directory is unusable input.
    if total_entries.saturating_mul(CENTRAL_FIXED) > cd_size {
        return Err(Error::validation(format!(
            "invalid ZIP64 end-of-central-directory record at byte {offset}: it declares \
             {total_entries} entries in a {cd_size}-byte central directory, below the \
             {CENTRAL_FIXED} bytes each record needs"
        )));
    }
    Ok(Zip64Directory {
        total_entries,
        cd_offset,
        cd_size,
    })
}

/// The byte offset just past a `ZIP64` end record of `declared` remaining bytes
/// that starts at `offset`.
///
/// # Errors
///
/// Returns [`Error::unsupported`] when the address arithmetic does not fit this
/// platform's `usize`, or [`Error::validation`] when the record's own size
/// overflows the address space.
fn record_end(offset: usize, declared: u64) -> Result<usize> {
    let declared = to_usize(declared, "a ZIP64 end-of-central-directory record size")?;
    offset
        .checked_add(ZIP64_EOCD_HEADER)
        .and_then(|value| value.checked_add(declared))
        .ok_or_else(|| {
            Error::validation(format!(
                "zip64 archive: the end-of-central-directory record at byte {offset} declares \
                 {declared} bytes, which overflows the archive address space"
            ))
        })
}

/// Which fixed central-directory fields carried the `ZIP64` escape value.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Zip64Escapes {
    /// The uncompressed size holds `0xffff_ffff`.
    pub(crate) declared: bool,
    /// The compressed size holds `0xffff_ffff`.
    pub(crate) compressed: bool,
    /// The local-header offset holds `0xffff_ffff`.
    pub(crate) local_offset: bool,
}

impl Zip64Escapes {
    /// Whether any field holds the escape value.
    const fn is_any(self) -> bool {
        self.declared || self.compressed || self.local_offset
    }
}

/// The 64-bit values a `ZIP64` extra field supplied for escaped fields.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Zip64Extra {
    /// Uncompressed size, when the fixed record escaped it.
    pub(crate) declared: Option<u64>,
    /// Compressed size, when the fixed record escaped it.
    pub(crate) compressed: Option<u64>,
    /// Local-header offset, when the fixed record escaped it.
    pub(crate) local_offset: Option<u64>,
}

/// Reads the `ZIP64` extra field of a central-directory record.
///
/// The extra data is walked as `(id u16, size u16, data)` tuples, and the
/// `ZIP64` field supplies, in order and only for the escaped fields, the
/// uncompressed size, the compressed size, and the local-header offset (the
/// fourth value, the disk number, is not needed because multi-disk archives are
/// rejected). An archive with escaped fields but no `ZIP64` extra field is
/// unusable, because the real value is nowhere else.
///
/// # Errors
///
/// Returns [`Error::validation`] naming the entry and the record offset when the
/// extra data is truncated or the `ZIP64` field is missing or too short.
pub(crate) fn parse_extra(
    bytes: &[u8],
    extra_offset: usize,
    extra_len: usize,
    escapes: Zip64Escapes,
    name: &str,
    offset: usize,
) -> Result<Zip64Extra> {
    if !escapes.is_any() {
        return Ok(Zip64Extra::default());
    }
    let extra = slice_at(bytes, extra_offset, extra_len)?;
    let mut field = 0;
    while field + 4 <= extra.len() {
        let id = read_u16_le(extra, field)?;
        let size = usize::from(read_u16_le(extra, field + 2)?);
        let start = field + 4;
        let data = slice_at(extra, start, size).map_err(|_| {
            Error::validation(format!(
                "invalid zip extra field 0x{id:04x} in the central-directory record at byte \
                 {offset} for entry {name:?}: it declares {size} bytes but only {} remain",
                extra.len().saturating_sub(start)
            ))
        })?;
        if id == ZIP64_EXTRA_ID {
            return decode_extra(data, escapes, name, offset);
        }
        field = start + size;
    }
    Err(Error::validation(format!(
        "invalid zip central-directory record at byte {offset} for entry {name:?}: its sizes or \
         local-header offset hold the ZIP64 escape value 0xffffffff but the record carries no \
         ZIP64 extra field (header id 0x0001)"
    )))
}

/// Decodes the `ZIP64` extra field's payload against the escaped fields.
///
/// # Errors
///
/// Returns [`Error::validation`] naming the entry and offset when the payload is
/// too short for the escaped fields.
fn decode_extra(
    data: &[u8],
    escapes: Zip64Escapes,
    name: &str,
    offset: usize,
) -> Result<Zip64Extra> {
    let mut extra = Zip64Extra::default();
    let mut read = 0;
    if escapes.declared {
        extra.declared = Some(next_u64(data, &mut read, name, offset)?);
    }
    if escapes.compressed {
        extra.compressed = Some(next_u64(data, &mut read, name, offset)?);
    }
    if escapes.local_offset {
        extra.local_offset = Some(next_u64(data, &mut read, name, offset)?);
    }
    Ok(extra)
}

/// Reads the next `u64` of a `ZIP64` extra field, advancing `read`.
///
/// # Errors
///
/// Returns [`Error::validation`] naming the entry and offset when fewer than
/// eight bytes remain.
fn next_u64(data: &[u8], read: &mut usize, name: &str, offset: usize) -> Result<u64> {
    let end = read
        .checked_add(8)
        .ok_or_else(|| short_extra(name, offset))?;
    let raw = data
        .get(*read..end)
        .ok_or_else(|| short_extra(name, offset))?;
    let mut value = [0_u8; 8];
    value.copy_from_slice(raw);
    *read = end;
    Ok(u64::from_le_bytes(value))
}

/// Builds the validation error for a `ZIP64` extra field that is too short.
fn short_extra(name: &str, offset: usize) -> Error {
    Error::validation(format!(
        "invalid ZIP64 extra field in the central-directory record at byte {offset} for entry \
         {name:?}: the fixed record escapes a value but the extra field ends before its 8 bytes"
    ))
}

/// Resolves one escaped 32-bit central-directory field.
///
/// # Errors
///
/// Returns [`Error::validation`] naming the entry when the escape value has no
/// `ZIP64` value behind it, and [`Error::unsupported`] when the 64-bit value does
/// not fit this platform's `usize`.
pub(crate) fn resolve(
    fixed: u32,
    extended: Option<u64>,
    what: &str,
    name: &str,
    offset: usize,
) -> Result<usize> {
    if fixed != ZIP64_MARKER {
        return Ok(fixed as usize);
    }
    let value = extended.ok_or_else(|| {
        Error::validation(format!(
            "invalid zip central-directory record at byte {offset} for entry {name:?}: the {what} \
             field holds the ZIP64 escape value 0xffffffff but no ZIP64 extra field supplies it"
        ))
    })?;
    to_usize(value, &format!("the ZIP64 {what} of zip entry {name:?}"))
}

/// Whether a digital-signature record starts at `offset`.
pub(crate) fn is_digital_signature(bytes: &[u8], offset: usize) -> bool {
    has_signature(bytes, offset, DIGITAL_SIGNATURE)
}

/// Steps over the digital-signature record at `offset` when one is there.
///
/// A digital-signature record is `PK\x06\x05`, a 16-bit length, and that many
/// bytes of signature data. It legally follows the central directory, so the
/// reader skips it instead of failing; a record whose declared length runs past
/// the input is skipped to the end of the input, which the caller's own bounds
/// check then rejects as a directory overrun.
pub(crate) fn skip_digital_signature(bytes: &[u8], offset: usize) -> usize {
    if !is_digital_signature(bytes, offset) {
        return offset;
    }
    let len = read_u16_le(bytes, offset + 4).map_or(0, usize::from);
    offset
        .saturating_add(DIGITAL_SIGNATURE_FIXED)
        .saturating_add(len)
        .min(bytes.len())
}

/// Reads a little-endian `u64` at `offset`.
///
/// # Errors
///
/// Returns [`Error::validation`] naming the offset when fewer than eight bytes
/// remain.
fn read_u64_le(bytes: &[u8], offset: usize) -> Result<u64> {
    let raw = slice_at(bytes, offset, 8)?;
    let mut value = [0_u8; 8];
    value.copy_from_slice(raw);
    Ok(u64::from_le_bytes(value))
}

/// Narrows a 64-bit `ZIP64` value to this platform's `usize`.
///
/// # Errors
///
/// Returns [`Error::unsupported`] naming `what` when the value is wider than an
/// address on this platform, the one `ZIP64` feature an in-process reader cannot
/// represent.
pub(crate) fn to_usize(value: u64, what: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| {
        Error::unsupported(format!(
            "zip64 archive: {what} is {value}, above the {}-bit addressable limit of this platform",
            usize::BITS
        ))
    })
}

#[cfg(test)]
mod tests;
