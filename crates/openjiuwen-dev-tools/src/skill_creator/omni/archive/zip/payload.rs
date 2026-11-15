//! Locating, decrypting, and decompressing one `zip` entry's payload.
//!
//! The central directory stays authoritative for every size and offset, so this
//! module only reads the local header's variable-field lengths and its DOS
//! modification time, then walks the payload in the order the format requires:
//! decrypt the traditional PKWARE (`ZipCrypto`) layer first, decompress the
//! remainder second, and verify the declared uncompressed size and the CRC-32 of
//! the result last.
//!
//! Both `stored` and `DEFLATE` payloads go through the same path, and an
//! encrypted entry's `compressed_size` covers its 12-byte encryption header, so
//! the ciphertext is that size minus the header and the decrypted remainder is
//! what the stored or `DEFLATE` stream really is.

use super::super::crc::crc32;
use super::super::inflate::inflate;
use super::super::{read_u16_le, slice_at};
use super::{
    zipcrypto, CentralRecord, FLAG_DATA_DESCRIPTOR, FLAG_ENCRYPTED, LOCAL_FIXED, LOCAL_SIGNATURE,
    METHOD_DEFLATE,
};
use openjiuwen_common::error::{Error, Result};

/// The compressed payload of an entry and the header value it carries.
pub(super) struct Payload<'a> {
    /// The plain or encrypted compressed payload the central directory sizes.
    pub(super) bytes: &'a [u8],
    /// The DOS modification time of the local header, the password check byte
    /// source of a data-descriptor-encrypted entry.
    pub(super) dos_time: u16,
}

/// Decodes an entry's payload and verifies its declared size and CRC-32.
///
/// `password` is only consulted for an entry whose general-purpose flag bit 0
/// says it is encrypted; the caller has already refused an encrypted entry it
/// cannot decrypt.
///
/// # Errors
///
/// Returns [`Error::validation`] for a truncated local header or payload, a
/// truncated encryption header, a password that fails the header's check byte, a
/// malformed `DEFLATE` stream, a size mismatch, or a CRC-32 mismatch.
pub(super) fn decode_payload(
    bytes: &[u8],
    record: &CentralRecord,
    path: &str,
    password: Option<&str>,
) -> Result<Vec<u8>> {
    let local = payload_slice(bytes, record)?;
    let decrypted;
    let payload = match password {
        Some(password) if record.flags & FLAG_ENCRYPTED != 0 => {
            // A data-descriptor entry keeps its local CRC field zero, so the
            // password check byte comes from the DOS modification time instead.
            let check = if record.flags & FLAG_DATA_DESCRIPTOR != 0 {
                zipcrypto::CheckByte::DosTime(local.dos_time)
            } else {
                zipcrypto::CheckByte::Crc32(record.crc)
            };
            decrypted = zipcrypto::decrypt(local.bytes, password, check, path)?;
            decrypted.as_slice()
        }
        _ => local.bytes,
    };
    let data = if record.method == METHOD_DEFLATE {
        let data = inflate(payload, record.declared_size).map_err(|error| {
            Error::validation(format!(
                "invalid DEFLATE data for zip entry {path:?}: {error}"
            ))
        })?;
        if data.len() != record.declared_size {
            return Err(size_mismatch(path, data.len(), record.declared_size));
        }
        data
    } else {
        // Stored: the payload is the file, so a size mismatch is visible before
        // the copy that would allocate it.
        if payload.len() != record.declared_size {
            return Err(size_mismatch(path, payload.len(), record.declared_size));
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
    Ok(data)
}

/// Builds the validation error for a payload whose length contradicts the
/// central directory.
fn size_mismatch(path: &str, actual: usize, declared: usize) -> Error {
    Error::validation(format!(
        "zip entry {path:?} declares {declared} uncompressed bytes but its payload holds \
         {actual} bytes"
    ))
}

/// Returns the compressed payload named by the record's local header, plus that
/// header's DOS modification time.
///
/// Only the local header's variable-field lengths and time are used: its sizes are
/// ignored because a streamed entry keeps them zero and writes a data descriptor
/// instead, so the central directory stays authoritative.
///
/// # Errors
///
/// Returns [`Error::validation`] naming the byte offset for a missing local
/// header signature, a truncated header, or a truncated payload.
fn payload_slice<'a>(bytes: &'a [u8], record: &CentralRecord) -> Result<Payload<'a>> {
    let offset = record.local_offset;
    let signature = slice_at(bytes, offset, 4)?;
    if signature != LOCAL_SIGNATURE {
        return Err(Error::validation(format!(
            "invalid zip local header for entry {:?} at byte {offset}: expected the signature \
             PK\x03\x04, found {signature:02x?}",
            record.name
        )));
    }
    let dos_time = read_u16_le(bytes, offset + 10)?;
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
    Ok(Payload {
        bytes: slice_at(bytes, data_start, record.compressed_size)?,
        dos_time,
    })
}
