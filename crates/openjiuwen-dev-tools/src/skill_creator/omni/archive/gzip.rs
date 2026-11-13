//! RFC 1952 `gzip` member parsing on top of the shared DEFLATE decoder.
//!
//! A gzip file is one or more *members*; each member is a 10-byte base header,
//! the optional `FEXTRA`/`FNAME`/`FCOMMENT`/`FHCRC` fields, one raw DEFLATE
//! payload, and an 8-byte trailer holding the CRC-32 and the uncompressed size
//! modulo 2^32 (RFC 1952 §2.3). Members are decoded in order and their payloads
//! concatenated, so `cat a.gz b.gz` behaves like the single stream it claims to
//! be.
//!
//! Everything is in-process and bounded: the payload is decoded by the shared
//! `inflate_stream` against the aggregate cap
//! [`MAX_ARCHIVE_UNCOMPRESSED_BYTES`], the member CRC-32 and `ISIZE` are
//! verified, and the concatenated payload must carry a `tar` or `zip` signature
//! because the omni pipeline only consumes archives. Every read is
//! bounds-checked and every error names the member start offset, so a corrupt
//! file reports where it went wrong instead of panicking.
//!
//! The header-CRC (`FHCRC`) extension is verified: the stored 16-bit value must
//! equal the low 16 bits of the CRC-32 of the header bytes that precede it.
//! `FTEXT` (bit 0 of `FLG`) carries no meaning here and is ignored.

use super::crc::crc32;
use super::inflate::inflate_stream;
use super::{
    detect, read_u16_le, read_u32_le, slice_at, ArchiveEntry, ArchiveFormat, ArchiveUnpacker,
    MAX_ARCHIVE_UNCOMPRESSED_BYTES,
};
use openjiuwen_common::error::{Error, Result};

/// The gzip member magic: `ID1` (`0x1f`) followed by `ID2` (`0x8b`).
const MAGIC: [u8; 2] = [0x1f, 0x8b];
/// Offset of the compression-method byte (`CM`) inside a member.
const CM_OFFSET: usize = 2;
/// Offset of the flag byte (`FLG`) inside a member.
const FLG_OFFSET: usize = 3;
/// Offset of `MTIME`, which is followed by `XFL` and `OS`.
const MTIME_OFFSET: usize = 4;
/// Length of the fixed header tail: `MTIME` (4) + `XFL` (1) + `OS` (1).
const FIXED_TAIL_LEN: usize = 6;
/// Length of a member trailer: `CRC32` (4) + `ISIZE` (4).
const TRAILER_LEN: usize = 8;
/// The only compression method RFC 1952 defines: DEFLATE (`CM` = 8).
const CM_DEFLATE: u8 = 8;
/// `FHCRC`: a 16-bit header CRC follows the other optional fields.
const FLG_FHCRC: u8 = 0x02;
/// `FEXTRA`: a `XLEN`-prefixed extra field follows.
const FLG_FEXTRA: u8 = 0x04;
/// `FNAME`: a `NUL`-terminated original file name follows.
const FLG_FNAME: u8 = 0x08;
/// `FCOMMENT`: a `NUL`-terminated comment follows.
const FLG_FCOMMENT: u8 = 0x10;
/// Reserved `FLG` bits, which RFC 1952 requires to be zero.
const FLG_RESERVED: u8 = 0xE0;

/// Decompresses a `gzip` stream and unpacks the `tar` or `zip` payload it carries.
///
/// Concatenated members are decoded in order; their payloads are concatenated
/// before the format is sniffed, so a `tar` split across members still parses.
///
/// # Errors
///
/// Returns [`Error::validation`] for an empty stream, a truncated or malformed
/// member header, a reserved `FLG` bit, a header-CRC/CRC-32/`ISIZE` mismatch,
/// trailing bytes that do not start a new member, and a payload that is not a
/// `tar` or `zip` archive (a nested gzip payload is rejected as such);
/// [`Error::unsupported`] for a compression method other than DEFLATE.
pub(crate) fn unpack(bytes: &[u8]) -> Result<Vec<ArchiveEntry>> {
    let payload = decompress(bytes)?;
    unpack_payload(&payload)
}

/// Decompresses every member of `bytes`, concatenating their payloads.
///
/// # Errors
///
/// Returns [`Error::validation`] for an empty stream and for anything a member
/// header, payload, or trailer can get wrong (see [`unpack`]).
fn decompress(bytes: &[u8]) -> Result<Vec<u8>> {
    let mut payload = Vec::new();
    let mut offset = 0usize;
    let mut index = 0usize;
    while offset < bytes.len() {
        offset = read_member(bytes, offset, index, &mut payload)?;
        index += 1;
    }
    if index == 0 {
        return Err(Error::validation(format!(
            "empty gzip stream: no member found in {} bytes",
            bytes.len()
        )));
    }
    Ok(payload)
}

/// Decodes the member starting at `start` into `payload`, returning the offset
/// just past its trailer.
///
/// `index` is the zero-based member number, used only to word error messages.
///
/// # Errors
///
/// Returns [`Error::validation`] for a bad header, payload, CRC-32, or `ISIZE`,
/// and [`Error::unsupported`] for a compression method other than DEFLATE.
fn read_member(bytes: &[u8], start: usize, index: usize, payload: &mut Vec<u8>) -> Result<usize> {
    let payload_start = read_header(bytes, start, index)?;
    let budget = remaining_cap(payload.len());
    let outcome = inflate_stream(&bytes[payload_start..], budget).map_err(|error| {
        Error::validation(format!(
            "gzip member {index} at offset {start}: its DEFLATE payload is invalid: {error}"
        ))
    })?;
    let trailer = payload_start
        .checked_add(outcome.consumed)
        .ok_or_else(|| offset_overflow(start))?;
    let stored_crc = read_u32_le(bytes, trailer)
        .map_err(|_| truncated_member(start, trailer, TRAILER_LEN / 2, bytes.len()))?;
    let isize_offset = trailer
        .checked_add(4)
        .ok_or_else(|| offset_overflow(start))?;
    let stored_isize = read_u32_le(bytes, isize_offset)
        .map_err(|_| truncated_member(start, isize_offset, TRAILER_LEN / 2, bytes.len()))?;

    let computed_crc = crc32(&outcome.data);
    if computed_crc != stored_crc {
        return Err(Error::validation(format!(
            "gzip member {index} at offset {start} has a CRC-32 mismatch: the trailer stores \
             {stored_crc:#010x} but the decoded payload hashes to {computed_crc:#010x} ({} bytes)",
            outcome.data.len()
        )));
    }
    let computed_isize = u32::try_from(outcome.data.len()).map_err(|_| {
        Error::validation(format!(
            "gzip member {index} at offset {start} decoded {} bytes, which cannot be compared with \
             a 32-bit ISIZE",
            outcome.data.len()
        ))
    })?;
    if computed_isize != stored_isize {
        return Err(Error::validation(format!(
            "gzip member {index} at offset {start} has an ISIZE mismatch: the trailer stores \
             {stored_isize} but the decoded payload is {computed_isize} bytes"
        )));
    }

    payload.extend_from_slice(&outcome.data);
    trailer
        .checked_add(TRAILER_LEN)
        .ok_or_else(|| offset_overflow(start))
}

/// The aggregate output budget left for a member when `produced` bytes have
/// already been decompressed.
///
/// Members share one budget ([`MAX_ARCHIVE_UNCOMPRESSED_BYTES`]), and the
/// budget saturates at zero instead of underflowing once it is exhausted, so a
/// further member fails with a validation error rather than wrapping.
fn remaining_cap(produced: usize) -> usize {
    MAX_ARCHIVE_UNCOMPRESSED_BYTES.saturating_sub(produced)
}

/// Parses one member header, returning the offset where its DEFLATE payload
/// starts.
///
/// # Errors
///
/// Returns [`Error::validation`] for a missing magic (trailing bytes after a
/// completed member are named as such), truncated fields, reserved `FLG` bits,
/// or a bad header CRC; [`Error::unsupported`] for a non-DEFLATE `CM`.
fn read_header(bytes: &[u8], start: usize, index: usize) -> Result<usize> {
    match bytes.get(start..).and_then(|rest| rest.get(..MAGIC.len())) {
        Some(found) if found == MAGIC.as_slice() => {}
        Some(found) if index == 0 => {
            return Err(Error::validation(format!(
                "not a gzip stream: expected the 1f 8b member magic at offset {start}, found \
                 {found:02x?}"
            )));
        }
        Some(found) => {
            return Err(Error::validation(format!(
                "trailing bytes after the final gzip member at offset {start}: expected a new \
                 member magic, found {found:02x?}"
            )));
        }
        None if index == 0 => {
            return Err(truncated_member(start, start, MAGIC.len(), bytes.len()));
        }
        None => {
            return Err(Error::validation(format!(
                "trailing bytes after the final gzip member at offset {start}: {} byte(s) are too \
                 short to hold a member magic",
                bytes.len().saturating_sub(start)
            )));
        }
    }

    let method = member_byte(bytes, start, start + CM_OFFSET)?;
    if method != CM_DEFLATE {
        return Err(Error::unsupported(format!(
            "gzip member {index} at offset {start} uses compression method {method}; only 8 \
             (DEFLATE) is supported"
        )));
    }
    let flags = member_byte(bytes, start, start + FLG_OFFSET)?;
    if flags & FLG_RESERVED != 0 {
        return Err(Error::validation(format!(
            "gzip member {index} at offset {start} sets reserved FLG bits in flag byte \
             {flags:#04x} (only the low 5 bits are defined)"
        )));
    }
    // `MTIME`, `XFL`, and `OS` are read and ignored.
    member_slice(bytes, start, start + MTIME_OFFSET, FIXED_TAIL_LEN)?;
    let mut offset = start + MTIME_OFFSET + FIXED_TAIL_LEN;

    if flags & FLG_FEXTRA != 0 {
        let extra_len = read_u16_le(bytes, offset)
            .map_err(|_| truncated_member(start, offset, 2, bytes.len()))?;
        offset += 2;
        member_slice(bytes, start, offset, usize::from(extra_len))?;
        offset += usize::from(extra_len);
    }
    if flags & FLG_FNAME != 0 {
        offset = skip_c_string(bytes, start, offset, "file name")?;
    }
    if flags & FLG_FCOMMENT != 0 {
        offset = skip_c_string(bytes, start, offset, "comment")?;
    }
    if flags & FLG_FHCRC != 0 {
        let stored = read_u16_le(bytes, offset)
            .map_err(|_| truncated_member(start, offset, 2, bytes.len()))?;
        let computed = header_crc(bytes, start, offset)?;
        if stored != computed {
            return Err(Error::validation(format!(
                "gzip member {index} at offset {start} has a header CRC mismatch: the header \
                 stores {stored:#06x} but the header bytes hash to {computed:#06x}"
            )));
        }
        offset += 2;
    }
    Ok(offset)
}

/// The low 16 bits of the CRC-32 over the header bytes `start..end`.
///
/// # Errors
///
/// Returns [`Error::validation`] when the header would be truncated.
fn header_crc(bytes: &[u8], start: usize, end: usize) -> Result<u16> {
    let header = member_slice(bytes, start, start, end.saturating_sub(start))?;
    let crc = crc32(header);
    u16::try_from(crc & 0xFFFF).map_err(|_| {
        Error::validation(format!(
            "gzip member at offset {start}: header CRC {crc:#010x} does not fit 16 bits"
        ))
    })
}

/// Skips a `NUL`-terminated string, returning the offset just past the `NUL`.
///
/// # Errors
///
/// Returns [`Error::validation`] naming the member when no terminator exists
/// before the end of the input.
fn skip_c_string(bytes: &[u8], start: usize, offset: usize, what: &str) -> Result<usize> {
    let rest = member_slice(bytes, start, offset, bytes.len().saturating_sub(offset))?;
    let end = rest.iter().position(|&byte| byte == 0).ok_or_else(|| {
        Error::validation(format!(
            "truncated gzip member at offset {start}: the {what} field starting at offset {offset} \
             has no NUL terminator"
        ))
    })?;
    offset
        .checked_add(end)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| offset_overflow(start))
}

/// Sniffs the decompressed `payload` and unpacks the archive it carries.
///
/// # Errors
///
/// Returns [`Error::validation`] when the payload is neither a `tar` nor a `zip`
/// archive, or is a nested gzip stream, plus everything the archive unpackers
/// return.
fn unpack_payload(payload: &[u8]) -> Result<Vec<ArchiveEntry>> {
    match detect(payload) {
        Some(ArchiveFormat::Tar) => ArchiveUnpacker::unpack(ArchiveFormat::Tar, payload),
        Some(ArchiveFormat::Zip) => ArchiveUnpacker::unpack(ArchiveFormat::Zip, payload),
        Some(ArchiveFormat::Gzip) => Err(Error::validation(format!(
            "gzip payload is a nested gzip stream: nested compression is not expanded ({} bytes)",
            payload.len()
        ))),
        None => Err(Error::validation(format!(
            "gzip payload is not an archive: no tar, zip, or gzip signature in {} bytes",
            payload.len()
        ))),
    }
}

/// Reads a single header byte, naming the member on truncation.
///
/// # Errors
///
/// Returns [`Error::validation`] when `offset` is past the end of `bytes`.
fn member_byte(bytes: &[u8], start: usize, offset: usize) -> Result<u8> {
    member_slice(bytes, start, offset, 1).map(|raw| raw[0])
}

/// Bounds-checked header slice, naming the member on truncation.
///
/// # Errors
///
/// Returns [`Error::validation`] when `offset..offset + len` is not in `bytes`.
fn member_slice(bytes: &[u8], start: usize, offset: usize, len: usize) -> Result<&[u8]> {
    slice_at(bytes, offset, len).map_err(|_| truncated_member(start, offset, len, bytes.len()))
}

/// The error for a header or trailer field that runs past the end of the input.
fn truncated_member(start: usize, offset: usize, len: usize, available: usize) -> Error {
    Error::validation(format!(
        "truncated gzip member at offset {start}: {len} byte(s) are required at offset {offset}, \
         but the stream is {available} bytes"
    ))
}

/// The error for an offset computation that cannot be represented.
fn offset_overflow(start: usize) -> Error {
    Error::validation(format!(
        "gzip member at offset {start}: the member offset overflows a usize"
    ))
}

#[cfg(test)]
mod tests;
