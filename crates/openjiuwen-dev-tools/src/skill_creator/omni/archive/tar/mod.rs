//! POSIX `ustar` / GNU `tar` parsing.
//!
//! A `tar` archive is a sequence of 512-byte headers, each followed by the
//! entry's data padded to a 512-byte boundary. Two consecutive zero blocks
//! terminate the archive, and everything after them is ignored; a single zero
//! block followed by a valid header continues (the `GNU` tolerance), and a
//! trailing partial block that cannot hold a header is ignored. The parser
//! accepts:
//!
//! - octal size fields and the `GNU` base-256 encoding;
//! - unsigned checksums and the `GNU` signed variant;
//! - the POSIX `ustar` `prefix` field;
//! - `GNU` long names (`L`) and long linknames (`K`), with or without a `NUL`
//!   terminator, each applying to the next header only;
//! - PAX per-file (`x`) and global (`g`) extended headers whose
//!   `<len> <key>=<value>\n` records are parsed and applied: `path`, `linkpath`,
//!   and decimal `size` are honored, while `mtime`, `uid`, `gid`, `uname`,
//!   `gname`, and `comment` are parsed and ignored. A per-file record applies to
//!   the next header only and wins over a global record, which applies to that
//!   header and every following entry; a PAX `path`/`linkpath` also wins over
//!   the `GNU` `L`/`K` value;
//! - regular files (`0`/`\0`), directories (`5`), symbolic links (`2`), and hard
//!   links (`1`).
//!
//! A symlink or hardlink entry carries its target verbatim in
//! [`link_target`](ArchiveEntry::link_target), because a link target may
//! legitimately be absolute or contain `..`; it is never used to write a file.
//! Devices and FIFOs carry no file contents and are skipped. Every entry path
//! goes through [`validate_path`](super::validate_path), and every effective size
//! (the PAX override when present, otherwise the header field) is charged to
//! [`CapTracker`](super::CapTracker) before its data slice is copied.
//!
//! Every byte access is bounds-checked, so a hostile archive yields an
//! [`Error::validation`] instead of a panic.
use super::{validate_path, ArchiveEntry, ArchiveEntryKind, CapTracker};
use openjiuwen_common::error::{Error, Result};
use std::collections::BTreeMap;

#[cfg(test)]
mod tests;

/// The size of one `tar` header or data block.
const BLOCK: usize = 512;
/// Offset of the `tar` name field.
const NAME_OFFSET: usize = 0;
/// Length of the `tar` name field.
const NAME_LEN: usize = 100;
/// Offset of the `tar` mode field.
const MODE_OFFSET: usize = 100;
/// Length of the `tar` mode field.
const MODE_LEN: usize = 8;
/// Offset of the `tar` size field.
const SIZE_OFFSET: usize = 124;
/// Length of the `tar` size field.
const SIZE_LEN: usize = 12;
/// Offset of the `tar` checksum field.
const CHECKSUM_OFFSET: usize = 148;
/// Length of the `tar` checksum field.
const CHECKSUM_LEN: usize = 8;
/// Offset of the `tar` typeflag field.
const TYPEFLAG_OFFSET: usize = 156;
/// Offset of the `tar` linkname field.
const LINKNAME_OFFSET: usize = 157;
/// Length of the `tar` linkname field.
const LINKNAME_LEN: usize = 100;
/// Offset of the `tar` magic field.
const MAGIC_OFFSET: usize = 257;
/// Length of the `tar` magic field.
const MAGIC_LEN: usize = 6;
/// Offset of the `ustar` prefix field.
const PREFIX_OFFSET: usize = 345;
/// Length of the `ustar` prefix field.
const PREFIX_LEN: usize = 155;
/// The POSIX `ustar` magic value, which enables the `prefix` field.
const USTAR_MAGIC: &[u8] = b"ustar\0";

/// Parses a `tar` archive, appending entries in archive order.
///
/// # Errors
///
/// Returns [`Error::validation`] naming the offending byte offset for a bad
/// checksum, an unparsable size, a malformed PAX record, truncated entry data, an
/// unsafe path, or an entry above the size caps.
pub(crate) fn unpack(bytes: &[u8]) -> Result<Vec<ArchiveEntry>> {
    let mut entries = Vec::new();
    let mut caps = CapTracker::new();
    let mut overrides = Overrides::default();
    let mut offset = 0usize;
    let mut previous_zero = false;
    while let Some(end) = offset.checked_add(BLOCK) {
        let Some(header) = bytes.get(offset..end) else {
            // A trailing partial block cannot hold a header, so it is ignored.
            break;
        };
        if header.iter().all(|&byte| byte == 0) {
            if previous_zero {
                break;
            }
            previous_zero = true;
            offset = end;
            continue;
        }
        previous_zero = false;
        verify_checksum(header, offset)?;
        let header_size = parse_size(header, offset)?;
        let typeflag = header.get(TYPEFLAG_OFFSET).copied().unwrap_or(0);
        // An extended-header payload is sized by its own header field; a data
        // entry may be resized by a PAX `size` record.
        let size = if is_metadata(typeflag) {
            header_size
        } else {
            overrides.size(offset)?.unwrap_or(header_size)
        };
        caps.charge(size, &format!("tar entry at byte {offset}"))?;
        let data_start = offset.checked_add(BLOCK).ok_or_else(|| {
            Error::validation(format!("tar header offset overflow at byte {offset}"))
        })?;
        let data_end = data_start.checked_add(size).ok_or_else(|| {
            Error::validation(format!(
                "tar entry at byte {offset} declares an unaddressable size of {size} bytes"
            ))
        })?;
        let data = bytes.get(data_start..data_end).ok_or_else(|| {
            Error::validation(format!(
                "truncated tar entry data at byte {offset}: {size} bytes declared but only {} \
                 remain",
                bytes.len().saturating_sub(data_start)
            ))
        })?;
        let padding = (BLOCK - size % BLOCK) % BLOCK;
        match typeflag {
            b'0' | b'\0' => {
                let path = overrides.path(header)?;
                entries.push(ArchiveEntry {
                    path,
                    bytes: data.to_vec(),
                    is_dir: false,
                    kind: ArchiveEntryKind::File,
                    link_target: None,
                    mode: parse_mode(header),
                });
            }
            b'5' => {
                let path = overrides.path(header)?;
                entries.push(ArchiveEntry {
                    path,
                    bytes: Vec::new(),
                    is_dir: true,
                    kind: ArchiveEntryKind::Directory,
                    link_target: None,
                    mode: parse_mode(header),
                });
            }
            b'1' | b'2' => {
                let path = overrides.path(header)?;
                entries.push(ArchiveEntry {
                    path,
                    bytes: Vec::new(),
                    is_dir: false,
                    kind: if typeflag == b'1' {
                        ArchiveEntryKind::HardLink
                    } else {
                        ArchiveEntryKind::Symlink
                    },
                    link_target: Some(overrides.link(header)),
                    mode: parse_mode(header),
                });
            }
            b'L' => overrides.set_long_name(payload_text(data)),
            b'K' => overrides.set_long_link(payload_text(data)),
            b'x' => overrides.set_local(parse_pax(data, data_start)?),
            b'g' => overrides.merge_global(parse_pax(data, data_start)?),
            // Devices and FIFOs carry no file contents and are skipped, but they
            // still consume the overrides awaiting this header.
            _ => {}
        }
        if !is_metadata(typeflag) {
            overrides.clear_local();
        }
        offset = data_start
            .checked_add(size)
            .and_then(|base| base.checked_add(padding))
            .ok_or_else(|| {
                Error::validation(format!("tar entry offset overflow after byte {offset}"))
            })?;
    }
    Ok(entries)
}

/// Whether `typeflag` marks an extended header rather than a data entry.
fn is_metadata(typeflag: u8) -> bool {
    matches!(typeflag, b'L' | b'K' | b'x' | b'g')
}

/// The name, linkname, and size overrides gathered from `GNU` and PAX headers.
#[derive(Debug, Default)]
struct Overrides {
    /// The `GNU` `L` long-name payload, applying to the next header only.
    long_name: Option<String>,
    /// The `GNU` `K` long-linkname payload, applying to the next header only.
    long_link: Option<String>,
    /// PAX per-file (`x`) records, applying to the next header only.
    local: BTreeMap<String, String>,
    /// PAX global (`g`) records, applying to every following entry.
    global: BTreeMap<String, String>,
}

impl Overrides {
    /// Records a `GNU` `L` long-name payload.
    fn set_long_name(&mut self, name: String) {
        self.long_name = Some(name);
    }

    /// Records a `GNU` `K` long-linkname payload.
    fn set_long_link(&mut self, link: String) {
        self.long_link = Some(link);
    }

    /// Records the per-file (`x`) records for the next header.
    fn set_local(&mut self, records: BTreeMap<String, String>) {
        self.local = records;
    }

    /// Adds a global (`g`) record, replacing an earlier value for the same key.
    fn merge_global(&mut self, records: BTreeMap<String, String>) {
        self.global.extend(records);
    }

    /// Drops every override consumed by a data entry.
    fn clear_local(&mut self) {
        self.long_name = None;
        self.long_link = None;
        self.local.clear();
    }

    /// Looks up a PAX value, preferring the per-file record over the global one.
    fn lookup(&self, key: &str) -> Option<&str> {
        self.local
            .get(key)
            .or_else(|| self.global.get(key))
            .map(String::as_str)
    }

    /// Resolves the entry path: PAX `path`, then the `GNU` `L` value, then the
    /// header's own name and `ustar` prefix.
    fn path(&self, header: &[u8]) -> Result<String> {
        if let Some(path) = self.lookup("path") {
            return validate_path(path);
        }
        if let Some(name) = &self.long_name {
            return validate_path(name);
        }
        validate_path(&header_path(header))
    }

    /// Resolves the link target: PAX `linkpath`, then the `GNU` `K` value, then
    /// the header's linkname field.
    ///
    /// The target is returned verbatim: it may be absolute or contain `..`, and
    /// nothing here writes through it.
    fn link(&self, header: &[u8]) -> String {
        if let Some(link) = self.lookup("linkpath") {
            return link.to_owned();
        }
        if let Some(link) = &self.long_link {
            return link.clone();
        }
        text_field(header, LINKNAME_OFFSET, LINKNAME_LEN)
    }

    /// Resolves the PAX `size` override for the entry at `offset`, in bytes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] naming `offset` when the record is not a
    /// decimal byte count.
    fn size(&self, offset: usize) -> Result<Option<usize>> {
        let Some(raw) = self.lookup("size") else {
            return Ok(None);
        };
        parse_decimal(raw.as_bytes()).map(Some).ok_or_else(|| {
            Error::validation(format!(
                "invalid tar PAX size record {raw:?} on the header at byte {offset}"
            ))
        })
    }
}

/// Parses a PAX extended-header payload into its `<key>=<value>` records.
///
/// Each record is `<len> <key>=<value>\n`, where `len` is the whole record length
/// in bytes, including the length digits and the trailing newline. Unknown keys
/// are retained here and ignored by the caller.
///
/// # Errors
///
/// Returns [`Error::validation`] naming the absolute byte offset of the offending
/// record when its length is not a decimal number, does not cover the record (or
/// does not end in a newline), runs past the payload, or when the record has no
/// `=` separator or an empty key. `base` is the offset of `data` in the archive.
fn parse_pax(data: &[u8], base: usize) -> Result<BTreeMap<String, String>> {
    let mut records = BTreeMap::new();
    let mut cursor = 0usize;
    while cursor < data.len() {
        let offset = base.saturating_add(cursor);
        let Some(space) = data
            .get(cursor..)
            .and_then(|rest| rest.iter().position(|&byte| byte == b' '))
        else {
            return Err(Error::validation(format!(
                "malformed tar PAX record at byte {offset}: no length separator"
            )));
        };
        let Some(len) = data.get(cursor..cursor + space).and_then(parse_decimal) else {
            return Err(Error::validation(format!(
                "malformed tar PAX record at byte {offset}: the record length is not a decimal \
                 number"
            )));
        };
        if len <= space + 1 {
            return Err(Error::validation(format!(
                "malformed tar PAX record at byte {offset}: the declared length {len} does not \
                 cover its own header"
            )));
        }
        let Some(record) = data.get(cursor..cursor.saturating_add(len)) else {
            return Err(Error::validation(format!(
                "truncated tar PAX record at byte {offset}: {len} bytes declared but only {} \
                 remain",
                data.len().saturating_sub(cursor)
            )));
        };
        if record.last() != Some(&b'\n') {
            return Err(Error::validation(format!(
                "malformed tar PAX record at byte {offset}: the declared length {len} does not end \
                 in a newline"
            )));
        }
        let body = record.get(space + 1..record.len() - 1).ok_or_else(|| {
            Error::validation(format!(
                "malformed tar PAX record at byte {offset}: the declared length {len} does not \
                 cover its own header"
            ))
        })?;
        let Some(equals) = body.iter().position(|&byte| byte == b'=') else {
            return Err(Error::validation(format!(
                "malformed tar PAX record at byte {offset}: missing the `key=value` separator"
            )));
        };
        let key = String::from_utf8_lossy(&body[..equals]).into_owned();
        if key.is_empty() {
            return Err(Error::validation(format!(
                "malformed tar PAX record at byte {offset}: empty keyword"
            )));
        }
        let value = String::from_utf8_lossy(&body[equals + 1..]).into_owned();
        records.insert(key, value);
        cursor = cursor.saturating_add(len);
    }
    Ok(records)
}

/// Parses a non-empty run of ASCII decimal digits.
fn parse_decimal(raw: &[u8]) -> Option<usize> {
    if raw.is_empty() {
        return None;
    }
    let mut value = 0usize;
    for &byte in raw {
        if !byte.is_ascii_digit() {
            return None;
        }
        value = value
            .checked_mul(10)?
            .checked_add(usize::from(byte - b'0'))?;
    }
    Some(value)
}

/// Reads the mode field of a header, or `None` when it declares nothing.
fn parse_mode(header: &[u8]) -> Option<u32> {
    parse_number(field(header, MODE_OFFSET, MODE_LEN)).and_then(|mode| u32::try_from(mode).ok())
}

/// Reads `prefix/name` from a `tar` header, honoring the `ustar` prefix field.
fn header_path(header: &[u8]) -> String {
    let name = text_field(header, NAME_OFFSET, NAME_LEN);
    let magic = field(header, MAGIC_OFFSET, MAGIC_LEN);
    let prefix = if magic == USTAR_MAGIC {
        text_field(header, PREFIX_OFFSET, PREFIX_LEN)
    } else {
        String::new()
    };
    if prefix.is_empty() {
        name
    } else {
        format!("{prefix}/{name}")
    }
}

/// Verifies the `tar` header checksum at `offset`.
///
/// The stored octal value is compared against the unsigned sum with the
/// checksum field read as ASCII spaces; the `GNU` signed variant (each byte
/// interpreted as `i8`) is accepted too.
fn verify_checksum(header: &[u8], offset: usize) -> Result<()> {
    let stored = parse_number(field(header, CHECKSUM_OFFSET, CHECKSUM_LEN)).ok_or_else(|| {
        Error::validation(format!(
            "invalid tar checksum field in header at byte {offset}"
        ))
    })?;
    let mut unsigned = 0u64;
    let mut signed = 0i64;
    for (index, &byte) in header.iter().enumerate() {
        let byte = if (CHECKSUM_OFFSET..CHECKSUM_OFFSET + CHECKSUM_LEN).contains(&index) {
            b' '
        } else {
            byte
        };
        unsigned += u64::from(byte);
        signed += i64::from(i8::from_ne_bytes([byte]));
    }
    if stored == unsigned || i64::try_from(stored).is_ok_and(|value| value == signed) {
        Ok(())
    } else {
        Err(Error::validation(format!(
            "tar checksum mismatch in header at byte {offset}: stored {stored}, computed \
             {unsigned}"
        )))
    }
}

/// Parses the octal (or `GNU` base-256) size field of a `tar` header.
fn parse_size(header: &[u8], offset: usize) -> Result<usize> {
    let value = parse_number(field(header, SIZE_OFFSET, SIZE_LEN)).ok_or_else(|| {
        Error::validation(format!("invalid tar size field in header at byte {offset}"))
    })?;
    usize::try_from(value).map_err(|_| {
        Error::validation(format!(
            "tar entry at byte {offset} declares a size too large to address: {value}"
        ))
    })
}

/// Parses a numeric `tar` header field.
///
/// Octal fields are read after leading `NUL`s and spaces, stopping at the first
/// non-octal byte. A high bit set in the first byte selects the `GNU` base-256
/// encoding, read big-endian with the high bit treated as a positive sign bit.
fn parse_number(raw: &[u8]) -> Option<u64> {
    if raw.first().is_some_and(|byte| byte & 0x80 != 0) {
        let mut value = 0u64;
        for (index, &byte) in raw.iter().enumerate() {
            let byte = if index == 0 { byte & 0x7f } else { byte };
            value = value.checked_mul(256)?.checked_add(u64::from(byte))?;
        }
        return Some(value);
    }
    let start = raw.iter().position(|&byte| byte != 0 && byte != b' ')?;
    let mut value = 0u64;
    let mut seen = false;
    for &byte in &raw[start..] {
        if !(b'0'..=b'7').contains(&byte) {
            break;
        }
        value = value.checked_mul(8)?.checked_add(u64::from(byte - b'0'))?;
        seen = true;
    }
    seen.then_some(value)
}

/// Reads an ASCII header field, trimming trailing `NUL`s and surrounding spaces.
fn text_field(header: &[u8], offset: usize, len: usize) -> String {
    let raw = field(header, offset, len);
    let end = raw.iter().position(|&byte| byte == 0).unwrap_or(raw.len());
    String::from_utf8_lossy(&raw[..end])
        .trim_matches(' ')
        .to_string()
}

/// Reads the text payload of a `GNU` long-name header up to its `NUL` terminator.
///
/// A payload without a `NUL` terminator (the `GNU` base-256 flavor used by some
/// writers) is used in full.
fn payload_text(data: &[u8]) -> String {
    let end = data
        .iter()
        .position(|&byte| byte == 0)
        .unwrap_or(data.len());
    String::from_utf8_lossy(&data[..end]).into_owned()
}

/// Bounds-checked slice of one header field.
fn field(header: &[u8], offset: usize, len: usize) -> &[u8] {
    header
        .get(offset..offset.saturating_add(len))
        .unwrap_or(&[])
}
