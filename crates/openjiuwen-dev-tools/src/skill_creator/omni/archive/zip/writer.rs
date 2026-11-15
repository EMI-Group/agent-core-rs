//! Deterministic stored `zip` writing: the [`ArchiveWriter`] entry point and the
//! header record helpers it is built from.
//!
//! The writer is stored-only and fully deterministic: a fixed DOS timestamp
//! (`1980-01-01 00:00:00`), no extra fields, no comments, entries in the order
//! given, and the UTF-8 flag set only for a non-ASCII name. Every entry kind gets
//! its Unix mode in the external attributes, so a symlink entry round trips
//! through [`unpack`](super::unpack) with its `kind` and `link_target` intact.

use super::super::crc::crc32;
use super::{
    validate_path, ArchiveEntry, ArchiveEntryKind, CENTRAL_SIGNATURE, DOS_DATE,
    DOS_DIRECTORY_ATTRIBUTE, DOS_TIME, EOCD_SIGNATURE, FLAG_UTF8, LOCAL_SIGNATURE, METHOD_STORED,
    UNIX_DIRECTORY_MODE, UNIX_FILE_MODE, UNIX_SYMLINK_MODE, VERSION,
};
use openjiuwen_common::error::{Error, Result};

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
    /// The external attributes carry the Unix mode of the entry kind: `0o120777`
    /// for a symlink, a directory mode plus the DOS directory bit for a
    /// directory, and `0o100644` otherwise. A symlink entry stores its target:
    /// `link_target` when it is set, `bytes` otherwise.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] for an empty entry list, a directory entry
    /// with content, a file entry whose path ends in `/`, a symlink entry with no
    /// target, or an unsafe path.
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
    /// External attributes, built from the entry kind.
    external: u32,
    /// Offset of the entry's local file header.
    offset: u32,
}

/// Validates the entries and pairs each borrowed path with its entry.
///
/// # Errors
///
/// Returns [`Error::validation`] for an empty list, more entries than a classic
/// archive can count, an unsafe path, a directory with content, a file path that
/// ends in `/`, or a symlink with no target.
fn stage_entries(entries: &[ArchiveEntry]) -> Result<Vec<(&str, &ArchiveEntry)>> {
    if entries.is_empty() {
        return Err(Error::validation(
            "cannot write a zip archive with no entries",
        ));
    }
    if entries.len() > usize::from(u16::MAX) {
        return Err(Error::validation(format!(
            "cannot write a zip archive with {} entries: a classic archive counts at most {} and \
             this writer emits no ZIP64 end-of-central-directory record",
            entries.len(),
            u16::MAX
        )));
    }
    let mut staged = Vec::with_capacity(entries.len());
    for (index, entry) in entries.iter().enumerate() {
        let path = validate_path(&entry.path)?;
        if entry.is_dir || entry.kind == ArchiveEntryKind::Directory {
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
        if entry.kind == ArchiveEntryKind::Symlink {
            payload_of(entry)?;
        }
        staged.push((entry.path.as_str(), entry));
    }
    Ok(staged)
}

/// The bytes the writer stores for `entry`: the link target for a symlink, the
/// contents otherwise.
///
/// # Errors
///
/// Returns [`Error::validation`] for a symlink entry that carries neither a
/// `link_target` nor a target in `bytes`.
fn payload_of(entry: &ArchiveEntry) -> Result<&[u8]> {
    if entry.kind != ArchiveEntryKind::Symlink {
        return Ok(&entry.bytes);
    }
    match &entry.link_target {
        Some(target) => Ok(target.as_bytes()),
        None if entry.bytes.is_empty() => Err(Error::validation(format!(
            "zip symlink entry {:?} has no target: set `link_target`, or put the target in `bytes`",
            entry.path
        ))),
        None => Ok(&entry.bytes),
    }
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
        let payload = payload_of(entry)?;
        let size = to_u32(payload.len(), &format!("the size of zip entry {name:?}"))?;
        let crc = crc32(payload);
        let flags = if name.is_ascii() { 0 } else { FLAG_UTF8 };
        write_local_header(out, name, size, crc, flags)?;
        out.extend_from_slice(payload);
        directory.push(StagedEntry {
            name: name.to_owned(),
            flags,
            crc,
            size,
            external: external_attributes(entry),
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
        put_u32(out, entry.external);
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
             and this writer emits no ZIP64 end-of-central-directory record",
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

/// Builds the external attributes for an entry: the Unix mode of its kind in the
/// high half, and the DOS directory bit in the low byte for a directory.
fn external_attributes(entry: &ArchiveEntry) -> u32 {
    if entry.is_dir || entry.kind == ArchiveEntryKind::Directory {
        (UNIX_DIRECTORY_MODE << 16) | DOS_DIRECTORY_ATTRIBUTE
    } else if entry.kind == ArchiveEntryKind::Symlink {
        UNIX_SYMLINK_MODE << 16
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
pub(super) fn put_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

/// Appends a little-endian `u32`.
pub(super) fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}
